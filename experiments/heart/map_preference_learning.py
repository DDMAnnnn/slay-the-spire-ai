"""Improve the demonstrated router with returns from matched alternative paths."""
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .heart_curriculum import read_model, save_model
from .route_irl import RoutePolicy, features
from .train import write


def load_forks(out):
    examples = []
    for act in range(1, 4):
        with gzip.open(out / f"act-{act}.jsonl.gz", "rt") as fp:
            for ctx, board, options in map(json.loads, fp):
                seed, _, *state = ctx
                row, elite_x, elite_y = state[9:]
                grid = np.asarray(board, dtype=np.int16).reshape(15, 7, 2)
                elite = np.zeros((15, 7), dtype=np.uint8)
                if not state[7] and 0 <= elite_x < 7 and 0 <= elite_y < 15:
                    elite[elite_y, elite_x] = 1
                legal = np.zeros(7, dtype=bool)
                values = np.zeros((2, 7), dtype=np.float32)
                for col, floor, end_act, _, heart, _, truncated in options:
                    assert not truncated
                    legal[col] = True
                    values[0, col] = heart
                    values[1, col] = (heart + .20 * (end_act >= 4) + .07 * (floor >= 51)
                                      + .03 * (floor >= 50) + .01 * (end_act >= 3))
                # All visits from a run stay together across acts.
                bucket = ((seed * 2654435761) % (2**32)) % 10
                examples.append((grid[:, :, 0], grid[:, :, 1], elite, row, features(state[:9], row),
                                 legal, values, bucket))
    return [torch.from_numpy(np.asarray(col)) for col in zip(*examples)]


def main():
    out = Path("build/map-preferences")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    *inputs, values, bucket = [p.to(device) for p in load_forks(out)]
    legal = inputs[-1]
    train, val, test = (torch.where(mask)[0] for mask in (bucket < 8, bucket == 8, bucket == 9))
    ref = RoutePolicy(True).to(device)
    ref.load_state_dict(torch.load("build/route-irl/whole_map.pt", map_location=device, weights_only=True))
    def logits(model, ix):
        return model(*(x[ix] for x in inputs))
    with torch.no_grad():
        ref_scores = torch.cat([logits(ref, ix) for ix in torch.arange(len(bucket), device=device).split(256)])
    parent = read_model("build/route-irl/selected-whole_map.json.gz")
    report = {"splits": list(map(len, (train, val, test))), "fits": {}}
    for objective, name in enumerate(("heart", "milestones")):
        target = values[:, objective, :]
        amplitude = target.max(dim=1).values - target.masked_fill(~legal, 9).min(dim=1).values
        informative = train[amplitude[train] > 0]
        if not len(informative):
            raise ValueError(f"No differing outcomes for {name}")
        torch.manual_seed(25118)
        model = copy.deepcopy(ref)
        opt = torch.optim.Adam(model.parameters(), lr=.013)
        @torch.no_grad()
        def assess(ix):
            z = torch.cat([logits(model, part) for part in ix.split(256)])
            base = ref_scores[ix].argmax(1)
            chosen = z.argmax(1)
            y = target[ix]
            diff = y.gather(1, chosen[:, None]) - y.gather(1, base[:, None])
            return dict(forks=len(ix), changed=int((chosen != base).sum()), gain=float(diff.mean()),
                        improved=int((diff > 0).sum()), lost=int((diff < 0).sum()))
        best, history = -float("inf"), []
        for epoch in range(1, 13):
            for _ in range(6):
                ix = informative[torch.randint(len(informative), (96,), device=device)]
                anchor = train[torch.randint(len(train), (96,), device=device)]
                both = torch.cat([ix, anchor])
                z = logits(model, both)
                # Every branch saw the same starting deck and RNG. A within-fork
                # advantage removes the level of difficulty shared by all actions.
                y = target[ix]
                advantage = y - ((y * legal[ix]).sum(1) / legal[ix].sum(1))[:, None]
                denom = amplitude[ix].clamp_min(.08)[:, None]
                improve = -(z[:len(ix)].softmax(1) * advantage / denom).sum(1).mean()
                trust = F.kl_div(z.log_softmax(1), ref_scores[both].softmax(1), reduction="batchmean")
                opt.zero_grad()
                (improve + .12 * trust).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                opt.step()
            if epoch in (1, 3, 6, 12):
                result = assess(val)
                history.append(dict(epoch=epoch, **result))
                if result["gain"] > best:
                    best = result["gain"]
                    chosen_epoch = epoch
                    best_weights = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_weights)
        save_model(out / f"{name}.json.gz", {**parent, "map_policy": [*model.export(), 1.]})
        report["fits"][name] = dict(training_informative=len(informative), validation=history,
                                     selected=chosen_epoch, test=assess(test))
        write(out / "fit.json", report)
        print(name, report["fits"][name], flush=True)


if __name__ == "__main__":
    main()
