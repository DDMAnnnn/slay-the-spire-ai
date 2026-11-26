"""Imitate the public combat search with the fork's full-observation policy encoder."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import heart_native as engine
import numpy as np
import torch
from torch.nn import functional as F

from .heart_curriculum import read_model
from .replay import apply_model
from .train import write
from .transfer_probe import EXAMPLE, evaluate, model_code


def dataset(paths, device):
    columns = {k: [] for k in ("obs", "prior", "teacher")}
    for path in paths:
        with np.load(path) as data:
            for name in columns:
                columns[name].append(data[name])
    return tuple(torch.from_numpy(np.concatenate(parts)).to(device) for parts in columns.values())


class FullStudent:
    def __init__(self, device, weights, threads):
        self.device, self.weights, self.threads = device, weights, threads
        self.net, _ = model_code().load(EXAMPLE / "weights.pt", device)
        state = torch.load("build/ironclad-transfer/update-64.pt", map_location=device, weights_only=True)
        self.net.load_state_dict(state["agent"])
        self.reference = copy.deepcopy(self.net).eval().requires_grad_(False)
        self.scale = 1.

    def correction(self, obs, prior, fixed):
        logits = self.net.dist(obs, prior > -1e7).logits
        # Subtract the frozen policy so the initial model reproduces the incumbent exactly.
        delta = logits - fixed
        centered = delta - delta.gather(1, prior.argmax(1, keepdim=True))
        return 24. * (centered / 8.).tanh()

    @torch.no_grad()
    def fixed_logits(self, data):
        obs, prior, _ = data
        return torch.cat([self.reference.dist(o, p > -1e7).logits for o, p in
                          zip(obs.split(256), prior.split(256))])

    @torch.inference_mode()
    def dist(self, obs, mask):
        native = engine.combat_student_priors(self.weights, obs.cpu().numpy(),
                                              mask.cpu().numpy().astype(np.uint8), self.threads)
        prior = torch.as_tensor(native, device=self.device)
        # A sole finite option keeps out-of-scope actions identical to the frozen parent.
        use = (prior > -1e7).sum(1) > 1
        if use.any():
            o, p = obs[use], prior[use]
            fixed = self.reference.dist(o, p > -1e7).logits
            prior[use] = p + self.scale * self.correction(o, p, fixed)
        return SimpleNamespace(logits=prior)


@torch.inference_mode()
def heldout(model, data, fixed, scales):
    obs, prior, teacher = data
    delta = torch.cat([model.correction(o, p, f) for o, p, f in
                       zip(obs.split(256), prior.split(256), fixed.split(256))])
    best = teacher.max(1).values
    parent = prior.argmax(1, keepdim=True)
    original_regret = best - teacher.gather(1, parent).flatten()
    result = []
    for scale in scales:
        pick = (prior + scale * delta).argmax(1, keepdim=True)
        regret = best - teacher.gather(1, pick).flatten()
        changed = original_regret > 1e-5
        result.append(dict(scale=scale, count=len(obs), parent_regret=float(original_regret.clamp(max=24).mean()),
                           regret=float(regret.clamp(max=24).mean()), changed=int(changed.sum()),
                           corrected=int((changed & (regret < 1e-5)).sum()),
                           spoiled=int((~changed & (regret > 1e-5)).sum())))
    return result


def fit(model, out):
    training = dataset(sorted(out.glob("196*.npz")), model.device)
    valid_files = sorted(out.glob("197*.npz"))
    selection = dataset(valid_files[:4], model.device)
    assessment = dataset(valid_files[4:], model.device)
    fixed = model.fixed_logits(training)
    fixed_selection = model.fixed_logits(selection)
    obs, prior, teacher = training
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=.00018, weight_decay=.01)
    history, best = [], float("inf")
    for epoch in range(1, 17):
        for idx in torch.randperm(len(obs), device=model.device).split(192):
            p, y = prior[idx], teacher[idx]
            correction = model.correction(obs[idx], p, fixed[idx])
            candidate = p > -1e7
            pred = p + correction
            # The soft ranking teaches alternatives even when the original move already wins.
            kl = F.kl_div((pred / 2).log_softmax(1), (y / 2).softmax(1), reduction="none").sum(1)
            error = F.smooth_l1_loss(correction, (y - p).clamp(-24, 24), reduction="none")
            loss = (kl + .04 * (error * candidate).sum(1) / candidate.sum(1)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.)
            optimizer.step()
        if epoch in (1, 2, 4, 8, 16):
            scores = heldout(model, selection, fixed_selection, (.25, .5, 1., 2.))
            best_scale = min(scores, key=lambda s: s["regret"])
            history.append(dict(epoch=epoch, scores=scores))
            if best_scale["regret"] < best:
                best = best_scale["regret"]
                selected = dict(epoch=epoch, scale=best_scale["scale"])
                torch.save(model.net.state_dict(), out / "selected.pt")
    model.net.load_state_dict(torch.load(out / "selected.pt", map_location=model.device, weights_only=True))
    model.scale = selected["scale"]
    return dict(training_states=len(obs), selection_seed_range=[197000000, 197000127],
                assessment_seed_range=[197000128, 197000255], selection_history=history,
                selected=selected, assessment=heldout(model, assessment, model.fixed_logits(assessment), [model.scale])[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("build/full-search-student"))
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if len(list(args.out.glob("196*.npz"))) != 48 or len(list(args.out.glob("197*.npz"))) != 8:
        raise ValueError("complete all 56 public observation shards before training")
    torch.manual_seed(20419)
    torch.set_num_threads(2)
    parent = read_model("build/trajectory-teacher/challenger.json.gz")
    model = FullStudent(args.device, apply_model("first", parent), args.threads)
    report_path = args.out / "policy.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["checkpoint_sha256"] != hashlib.sha256((args.out / "selected.pt").read_bytes()).hexdigest():
            raise ValueError("the checkpoint differs from the recorded evaluation")
        model.net.load_state_dict(torch.load(args.out / "selected.pt", map_location=args.device, weights_only=True))
        model.scale = report["selected"]["scale"]
    else:
        report = fit(model, args.out)
        report["checkpoint_sha256"] = hashlib.sha256((args.out / "selected.pt").read_bytes()).hexdigest()
        write(report_path, report)
    def run(start, n, mode):
        return evaluate(model, lambda o, m: m, args.device, mode, parent, start, n, 192, args.threads, "ironclad")
    if "ordinary_screen" not in report:
        report["ordinary_screen"] = {mode: run(204000000, 1024, mode) for mode in ("parent", "combat")}
        write(report_path, report)
    parent_score, student_score = [report["ordinary_screen"][k] for k in ("parent", "combat")]
    assessment = report["assessment"]
    print("assessment", assessment, "screen", report["ordinary_screen"], flush=True)
    if assessment["regret"] >= assessment["parent_regret"] and student_score["act3"] <= parent_score["act3"]:
        report["qualification"] = "Neither held-out search regret nor ordinary Act 3 arrivals improved"
        write(report_path, report)
        return
    panels = report.setdefault("ordinary_panel", [])
    for offset in range(sum(s["runs"] for s in panels), 32768, 2048):
        panels.append(run(194000000 + offset, 2048, "combat"))
        write(report_path, report)


if __name__ == "__main__":
    main()
