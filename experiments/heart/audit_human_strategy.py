"""Verify the exported historical policy against its Python logits on real public reward menus."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts
import torch

from .heart_curriculum import read_model
from .human_menus import STARTERS, TOKENS, PrefixReader
from .human_strategy import MenuModel
from .replay import apply_model, rows
from .train import write


def audit(config):
    model = MenuModel(True).eval()
    with torch.no_grad():
        for parameter, value in zip((model.context, model.keys, model.bias, model.upgrade, model.repeat), config):
            parameter.copy_(torch.tensor(value))
    parent = read_model("build/trajectory-teacher/challenger.json.gz")
    w = apply_model("first", parent)
    reader = PrefixReader()
    base_cards = {reader.cards[s] for s in STARTERS}
    starter_relic = reader.relics["Burning Blood"]
    examples = []
    for seed in range(215000000, 215000016):
        env = sts.Env()
        env.reset(seed, 20, final_act=True, character="ironclad")
        for action in engine.trace(w, seed):
            o, mask = env.observe(), env.mask()
            screen = sts.SCREEN_NAMES[int(o[6])]
            if not o[7] and screen in ("REWARDS", "BOSS_RELIC"):
                options = []
                raw = rows(o, "choices")
                for i, row in enumerate(raw):
                    a = sts.ACT_CHOOSE + i
                    if not mask[a]:
                        continue
                    kind = sts.CHOICE_KINDS[int(row[0])]
                    if kind == "SKIP": item = (0, 0)
                    elif kind == "SINGING_BOWL": item = (1, 0)
                    elif kind == "CARD": item = (int(row[1])+1, int(row[2] > 0))
                    elif kind == "RELIC" and screen == "BOSS_RELIC":
                        item = (1+engine.card_count+int(row[1]), 0)
                    else:
                        options = []
                        break
                    options.append((a, item))
                ids = [item[0] for _, item in options]
                if 0 in ids and max(ids) >= 2:
                    ctx = np.zeros(TOKENS, np.float32)
                    for c in rows(o, "deck"):
                        token = 1+int(c[0])
                        if c[0] and token not in base_cards:
                            ctx[token] = min(3, ctx[token]+1)
                    for r in rows(o, "relics"):
                        token = 1+engine.card_count+int(r[0])
                        if r[0] and token != starter_relic:
                            ctx[token] = 1
                    actions = [a for a,_ in options]
                    examples.append((o.copy(), mask.copy(), ctx, options, screen,
                                     np.asarray(engine.scores(w,o,mask))[actions]))
            if not env.step(action):
                raise RuntimeError("baseline replay diverged")
    scales = [6., 12.]
    engine.set_historical_menu(*config, scales)
    maximum, counts = 0., {}
    with torch.no_grad():
        for o, mask, ctx, options, screen, prior in examples:
            action, items = zip(*options)
            logits = model(torch.tensor(ctx[None]), torch.tensor([min(2,int(o[4])-1)]),
                           torch.tensor([items])).numpy()[0]
            skip = items.index((0,0))
            expected = scales[int(screen == "BOSS_RELIC")]*(logits-logits[skip])
            actual = np.asarray(engine.scores(w,o,mask))[list(action)]-prior
            maximum = max(maximum, float(np.abs(expected-actual).max()))
            counts[screen] = counts.get(screen,0) + 1
    if maximum > 1e-4 or set(counts) != {"REWARDS","BOSS_RELIC"}:
        raise AssertionError((counts,maximum))
    return dict(menus=counts, maximum_absolute_error=maximum, starts=[215000000,215000015])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/human-strategy"))
    args = parser.parse_args()
    config = json.loads((args.out / "fitted.json").read_text())["context"]["config"]
    result = audit(config)
    write(args.out / "parity.json", result)
    print(result)


if __name__ == "__main__":
    main()
