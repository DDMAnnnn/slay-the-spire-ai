"""Learn conditional card and boss-relic preferences from observed human offers."""
import argparse
import copy
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .heart_curriculum import read_model, save_model
from .human_menus import TOKENS, collect
from .inventory_search import panel_chunks
from .public_belief import evaluate
from .replay import apply_model
from .train import write


class MenuModel(nn.Module):
    def __init__(self, contextual):
        super().__init__()
        self.contextual = contextual
        self.context = nn.Parameter(torch.randn(TOKENS, 24) * .1)
        self.keys = nn.Parameter(torch.randn(TOKENS, 24) * .1)
        self.bias = nn.Parameter(torch.zeros(3, TOKENS))
        self.upgrade = nn.Parameter(torch.zeros(TOKENS))
        self.repeat = nn.Parameter(torch.zeros(TOKENS))

    def forward(self, ctx, act, menu):
        ids = menu[:, :, 0].long()
        use = ids.clamp_min(0)
        score = self.bias[act[:, None], use] + self.upgrade[use] * menu[:, :, 1]
        if self.contextual:
            ctx = ctx.float()
            weights = F.dropout(ctx, p=.12, training=self.training)
            h = torch.tanh((weights @ self.context) / ctx.sum(1, keepdim=True).clamp_min(1).sqrt())
            score = score + (h[:, None, :] * self.keys[use]).sum(2)
            score = score + self.repeat[use] * ctx.gather(1, use).log1p()
        return score.masked_fill(ids < 0, -1e8)

    def export(self):
        params = (self.context, self.keys, self.bias, self.upgrade, self.repeat)
        return [(x if self.contextual or i in (2, 3) else torch.zeros_like(x)).detach().cpu().tolist()
                for i, x in enumerate(params)]


def load_data(path, device):
    with np.load(path) as z:
        return {k: torch.from_numpy(z[k]).to(device) for k in z.files}


@torch.inference_mode()
def assess(model, data):
    model.eval()
    metrics = np.zeros((2, 4))
    for i in torch.arange(len(data["pick"]), device=data["pick"].device).split(512):
        logits = model(data["context"][i], data["act"][i], data["options"][i])
        label = data["pick"][i]
        loss = F.cross_entropy(logits, label, reduction="none")
        correct = logits.argmax(1) == label
        chance = (data["options"][i, :, 0] >= 0).sum(1).float().reciprocal()
        for category in (0, 1):
            group = data["kind"][i] == category
            metrics[category] += [int(group.sum()), int(correct[group].sum()), float(loss[group].sum()),
                                  float(chance[group].sum())]
    return {name: dict(count=int(n), accuracy=a/n, cross_entropy=b/n, uniform_accuracy=c/n)
            for name, (n, a, b, c) in zip(("cards", "relics"), metrics)}


def fit(data, contextual, device):
    torch.manual_seed(21317)
    net = MenuModel(contextual).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.004, weight_decay=.02)
    train = data["train"]
    history, best = [], float("inf")
    for epoch in range(1, 33):
        net.train()
        for i in torch.randperm(len(train["pick"]), device=device).split(384):
            logits = net(train["context"][i], train["act"][i], train["options"][i])
            loss = F.cross_entropy(logits, train["pick"][i], reduction="none")
            # A run contains many card menus and only two boss-relic menus.
            loss = (loss * (1 + 2 * train["kind"][i])).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            optimizer.step()
        if epoch in (2, 4, 8, 16, 32):
            stats = assess(net, data["validation"])
            history.append(dict(epoch=epoch, metrics=stats))
            rank = stats["cards"]["cross_entropy"] + .3 * stats["relics"]["cross_entropy"]
            if rank < best:
                best, chosen = rank, epoch
                state = copy.deepcopy(net.state_dict())
    net.load_state_dict(state)
    return net.export(), dict(epoch=chosen, history=history, test=assess(net, data["test"]),
                              streamers=assess(net, data["streamers"]))


def rank(result):
    return result["first_boss"] + 2*result["second_boss"] + .1*result["act3"] + 4*result["act4"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("build/human-strategy"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if not (out / "data.json").exists():
        write(out / "data.json", collect(out))
    model_path = out / "fitted.json"
    if not model_path.exists():
        data = {k: load_data(out / f"{k}.npz", args.device) for k in ("train", "validation", "test", "streamers")}
        fitted = {}
        for label in ("frequency", "context"):
            config, metrics = fit(data, label == "context", args.device)
            fitted[label] = dict(config=config, metrics=metrics)
        write(model_path, fitted)
    fitted = json.loads(model_path.read_text())
    report_path = out / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(
        data=json.loads((out / "data.json").read_text()), imitation={k: v["metrics"] for k,v in fitted.items()},
        validation=[], start=214000000, test={})
    parent = read_model("build/trajectory-teacher/challenger.json.gz")
    w = apply_model("first", parent)
    def enable(name, scales):
        engine.set_historical_menu(*fitted[name]["config"], scales)
    if "selected" not in report:
        for scales in ((1., 2.), (4., 6.), (8., 12.), (16., 24.), (40., 50.), (6., 24.)):
            enable("context", scales)
            result = evaluate(w, 213000000, 1024, args.threads)
            report["validation"].append(dict(scales=scales, **result))
        report["selected"] = max(report["validation"], key=rank)["scales"]
        write(report_path, report)
    scales = report["selected"]
    save_model(out / "challenger.json.gz", {**parent, "historical_menu": [*fitted["context"]["config"], scales]})
    for name in ("parent", "context", "frequency", "cards", "relics"):
        if name in report["test"]:
            continue
        if name == "parent":
            engine.reset_historical_menu()
        else:
            active = [0 if name == "relics" else scales[0], 0 if name == "cards" else scales[1]]
            enable("frequency" if name == "frequency" else "context", active)
        report["test"][name] = panel_chunks(out, name, w, 214000000, 32768, args.threads)
        write(report_path, report)
        print(name, report["test"][name], flush=True)


if __name__ == "__main__":
    main()
