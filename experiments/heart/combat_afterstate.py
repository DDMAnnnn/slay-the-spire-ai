"""Public mid-fight features and dependency-free inference in the source simulator."""
import math
from collections import Counter


def features(row, layout):
    zones = layout["zones"]
    player = zones["player"][0]
    hp, maximum = row[0], max(1, row[1])
    x = [hp, maximum, hp / maximum, row[3], row[4], row[22], row[23],
         sum(v > 0 for v in row[17:22])]
    x.extend(row[player + 2:player + 7])
    # Pool monsters by their displayed statistics. Intent damage is zero if the icon is concealed.
    start, count, width = zones["monsters"]
    monsters = [row[p:p + width] for p in range(start, start + count * width, width)
                if row[p] and row[p + 1] > 0]
    incoming = sum(max(0, m[8]) * max(0, m[9]) for m in monsters)
    x.extend([len(monsters), incoming, max(0, incoming - row[player + 2]),
              min((m[1] for m in monsters), default=0)])
    offsets = [1, 2, 3, *[10 + layout["monster_status"][n]
                          for n in ("STRENGTH", "VULNERABLE", "WEAK", "POISON")]]
    for field in offsets:
        vals = [m[field] for m in monsters]
        x.extend((sum(vals), max(vals, default=0)))
    x.extend(row[player + 8:player + zones["player"][2]])
    # Enemy identity and hand composition are bags. Keep named inputs consistent across runtimes.
    ids = Counter(int(m[0]) for m in monsters)
    x.extend(ids[i] for i in layout["critic_enemies"])
    start, count, width = zones["hand"]
    hand = [row[p:p + width] for p in range(start, start + count * width, width) if row[p]]
    ids = Counter(int(c[0]) for c in hand)
    x.extend(ids[i] for i in layout["critic_cards"])
    x.extend([sum(c[1] for c in hand), sum(c[3] for c in hand), sum(c[2] == 0 for c in hand)])
    return list(map(float, x))


def estimate(model, x):
    value = model["bias"]
    for tree in model["trees"]:
        index = 0
        while True:
            field, threshold, left, right, leaf = tree[index]
            if left == 0:
                value += leaf
                break
            index = left if x[field] <= threshold else right
    return value


def competing_moves(result):
    incumbent = int(result.action.bits)
    roots = {int(a.bits): (v, q / 100.) for a, v, q in
             zip(result.root_actions, result.root_visits, result.root_mean_evaluations)
             if v > 0 and math.isfinite(q)}
    if incumbent not in roots or len(roots) < 2:
        return None
    visits, score = roots[incumbent]
    rivals = [k for k, (v, q) in roots.items()
              if k != incumbent and v * 3 >= visits and abs(q - score) <= 12.]
    if not rivals:
        return None
    rival = max(rivals, key=lambda k: (roots[k][0], roots[k][1]))
    return roots, rival


def pair_features(state, result, scores, layout):
    incumbent = int(result.action.bits)
    roots, rival = competing_moves(result)
    c = state["combat"]
    left, right = roots[incumbent], roots[rival]
    x = [state["act"],state["floor"],c["player_hp"]/max(1,c["player_max_hp"]),
         c["player_block"],c["energy"],c["turn"],len(c["hand"]),len(c["monsters"]),
         left[1],right[1],right[1]-left[1],math.log((right[0]+1)/(left[0]+1)),
         scores[str(incumbent)],scores[str(rival)],scores[str(rival)]-scores[str(incumbent)]]
    for bits in (incumbent,rival):
        kind=bits>>29
        source=bits&0xffff
        target=(bits>>16)&8191
        card = c["hand"][source] if kind==0 and source<len(c["hand"]) else None
        x.extend([int(kind==0),int(kind==1),int(kind==4),int(card[1]) if card else 0,
                  card[2] if card else 0,
                  c["monsters"][target][1] if target<len(c["monsters"]) else 0])
        x.extend(int(card is not None and card[0]+1==i) for i in layout["critic_cards"])
    return x


def choose(env, sts, state, result, seed, model, layout, weight, worlds=3, diagnostic=None):
    from public_demo_export import combat_index, public_row

    incumbent = int(result.action.bits)
    if any(r[0] == model['frozen_eye'] for r in state['relics']):
        return incumbent, False
    contest = competing_moves(result)
    if contest is None:
        return incumbent, False
    roots, rival = contest
    score = roots[incumbent][1]
    totals = {incumbent: 0., rival: 0.}
    for i in range(worlds):
        # The caller only invokes this on an ordinary play prompt. Concealed draws and RNG
        # are resampled before either action. Both moves see the same independent world.
        world = env.clone_with_resampled_hidden((seed + 1000003 * (i + 1)) & ((1 << 64) - 1))
        for bits in totals:
            branch = world.clone()
            transition = branch._native.step(sts.EnvironmentAction(sts.ActionDomain(1), bits))
            for _ in range(12):
                s = branch.snapshot()
                if transition.terminated or transition.exited_combat or not s["in_combat"]:
                    health = max(0, s["hp"])
                    totals[bits] += 35. + health + 4 * sum(v > 1 for v in s["potions"]) if health else 0.
                    break
                legal = branch._native.legal_actions()
                if any(a.bits == 0x80000000 for a in legal):
                    ids = {combat_index(a.bits) for a in legal}
                    obs = public_row(s, ids - {None}, layout)
                    totals[bits] += estimate(model, features(obs, layout)) + 4 * sum(v > 1 for v in s["combat"]["potions"])
                    break
                if not legal:
                    return incumbent, False
                transition = branch._native.step(min(legal, key=lambda a: int(a.bits)))
            else:
                return incumbent, False
    if diagnostic is not None:
        diagnostic.update({str(k): v / worlds for k, v in totals.items()})
    correction=model.get("correction")
    if correction:
        scores={str(k):v/worlds for k,v in totals.items()}
        x=pair_features(state,result,scores,layout)
        predicted=estimated_difference(correction,x)
        return (rival if predicted > correction["threshold"] else incumbent), True
    difference = (totals[rival] - totals[incumbent]) / worlds
    hurdle = max(0., score - roots[rival][1])
    return (rival if weight * difference > hurdle + 1.0 else incumbent), True


def estimated_difference(policy, x):
    if policy['kind']=='tree':
        return estimate(policy,x)
    return policy['bias']+sum(a*b for a,b in zip(policy['coefficients'],x))
