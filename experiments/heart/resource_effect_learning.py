"""Estimate within-position resource effects from crossed HP and potion interventions."""
import json
from pathlib import Path

import numpy as np

from .train import write


def design(act, stage, hp, number, item, kept, items):
    # Piecewise linear HP permits the marginal value of 10 HP to change with current HP.
    spline = np.clip(hp[:, None] * 4 - np.arange(4), 0., 1.)
    inventory = (number[:, None] > np.arange(3)).astype(float)
    inventory = np.column_stack((inventory, number * (hp - .5)))
    core = np.column_stack((spline, inventory))
    x = np.zeros((len(act), 8 * (1 + 3 + 12) + items + 1))
    cols = np.arange(8)
    row = np.arange(len(act))[:, None]
    x[:, :8] = core
    x[row, 8 + 8 * (act[:, None] - 1) + cols] = core
    x[row, 32 + 8 * ((act[:, None] - 1) * 4 + stage[:, None]) + cols] = core
    x[np.arange(len(act)), 128 + item] = kept & (item > 0)
    return x


def main():
    out = Path("build/resource-effects")
    report = json.loads((out / "collection.json").read_text())
    if set(report) != {"1", "2", "3"}:
        raise ValueError("training bank is incomplete")
    items = json.loads(Path("build/resource-policy/sizes.json").read_text())["potions"]
    rows = np.concatenate([np.load(f) for f in sorted(out.glob("act-*.npy"))])
    a = rows[rows[:, 9] == 0]
    seed, act, floor, maxhp, _, hp, initial, item, removed = a[:, :9].T
    stage = np.clip((floor - 1 - 17 * (act - 1)) // 4, 0, 3)
    x = design(act, stage, hp / maxhp, initial - removed, item, removed == 0, items)
    next_act = a[:, 14] > act
    outcomes = np.column_stack(((a[:, 13] >= floor + 5) | next_act, next_act, a[:, 16] > 0)).astype(float)
    old = (hp - 4 * removed)[:, None].astype(float)
    _, group = np.unique(np.column_stack((seed, act)), axis=0, return_inverse=True)
    count = np.bincount(group)
    def center(y):
        mean = np.zeros((len(count), y.shape[1]))
        np.add.at(mean, group, y)
        return y - (mean / count[:, None])[group]
    x, delta, old = center(x), center(outcomes), center(old)
    bucket = ((seed.astype(np.int64) * 2654435761) % 2**32) % 10
    tr, va, te = (bucket < 8, bucket == 8, bucket == 9)
    weights = 1. / count[group]
    def fit(matrix, penalty):
        z = matrix[tr]; w = weights[tr, None]
        return np.linalg.solve(z.T @ (z * w) + np.diag(penalty), z.T @ (delta[tr] * w))
    def mse(pred, mask):
        return np.average((pred[mask] - delta[mask])**2, axis=0, weights=weights[mask]).tolist()
    candidates = {}
    for strength in (.25, 1., 4., 16.):
        penalty = strength * np.repeat([1., 3., 6., 3.], [8, 24, 96, items + 1])
        coefs = fit(x, penalty)
        candidates[strength] = (coefs, mse(x @ coefs, va))
    best = min(candidates, key=lambda r: sum(candidates[r][1][:2]))
    coefs = candidates[best][0]
    naive = old @ fit(old, [1e-6])
    result = dict(collection=report, selected_ridge=best,
                  validation={str(r):v[1] for r,v in candidates.items()},
                  groups=[len(np.unique(group[m])) for m in (tr,va,te)],
                  target_order=["survive five more floors or advance act", "advance act", "enter Heart"],
                  test_mse=mse(x @ coefs, te), fixed_hp_and_potions_mse=mse(naive, te),
                  zero_effect_mse=mse(np.zeros_like(delta), te))
    # A probability change is converted to equivalent HP using the training population.
    # Hold potion count fixed and use the same conversion for all actions.
    table_hp = np.zeros((12,21)); table_pot = np.zeros((12,21,5,items+1))
    objective = .5 * (coefs[:,0] + coefs[:,1])
    for ctx in range(12):
        ca = ctx // 4 + 1; st = ctx % 4
        for h in range(21):
            ids = np.arange(items + 1)
            def value(n, held):
                return design(np.full_like(ids, ca), np.full_like(ids, st), np.full(len(ids), h / 20.),
                                            np.full_like(ids,n), ids, np.full(len(ids),held),items) @ objective
            table_hp[ctx,h] = value(0, False)[0]
            for n in range(1,6):
                table_pot[ctx,h,n-1] = value(n, True) - value(n-1, False)
    norm = np.mean(table_hp[:,-1] - table_hp[:,0]) / np.mean(maxhp[tr])
    if norm <= 0:
        raise ValueError("training did not identify a positive typical HP effect")
    table_hp /= norm
    table_pot /= norm
    which = tr & (initial > 0) & (removed == 0)
    actual = table_pot[(act[which]-1)*4+stage[which],np.rint(20*hp[which]/maxhp[which]).astype(int),
                       initial[which]-1,item[which]]
    result.update(typical_potion_cost=float(np.median(actual)),probability_per_hp=norm,
                  potion_equivalent_hp_percentiles=np.percentile(table_pot[:,:,0,1:], [10,50,90]).tolist())
    write(out / "fit.json", result)
    write(out / "tables.json", dict(hp=table_hp.tolist(),potion=table_pot.tolist()))
    print(result, flush=True)


if __name__ == "__main__":
    main()
