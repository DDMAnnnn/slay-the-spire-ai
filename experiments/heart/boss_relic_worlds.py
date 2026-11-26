"""Compare the first boss-relic offers across the same independently sampled Act 2 futures."""
import hashlib
import json
import random


def branch_value(native, first, sts, router, seed, fights=3):
    move = native.step(sts.EnvironmentAction(sts.ActionDomain(0), first))
    fight_count = 0
    rng = random.Random(seed)
    for _ in range(400):
        state = native.snapshot()
        hp = max(0, state["hp"])
        if move.exited_combat and hp > 0:
            fight_count += 1
        if move.terminated or native.is_terminal:
            return 3. * fight_count
        if fight_count == fights:
            return 35. + hp + 4. * sum(v > 1 for v in state["potions"])
        legal = native.legal_actions()
        if len(legal) == 1:
            action = legal[0]
        elif state["in_combat"]:
            if any(a.bits == 0x80000000 for a in legal):
                result = native.public_belief_combat_search_action(96,4,rng.getrandbits(53),2,True)
                action = result.action
            else:
                action = min(legal,key=lambda a:int(a.bits))
        elif state["screen_state"] == 1 and state["current_event"] == 31:
            action = legal[rng.randrange(len(legal))]
        else:
            selected = router(native,state,legal)
            action = sts.EnvironmentAction(sts.ActionDomain(0),selected)
        move = native.step(action)
    return None


def choose(env, sts, state, legal, incumbent, router, observed_relics, count):
    if state["screen_state"] != 3 or state["act"] != 1:
        raise ValueError("expected the Act 1 boss chest")
    offers = sorted(a.bits for a in legal if int(a.domain) == 0 and a.bits < 4)
    if incumbent not in offers or len(offers) < 2:
        return incumbent, {}
    values = {a:[] for a in offers}
    observed = [sts.RelicId(int(i)) for i in sorted(observed_relics)]
    visible = [state["act"], state["floor"], state["hp"], state["max_hp"], state["gold"],
               sorted((c[0], c[1]) for c in state["deck"]), sorted(tuple(r) for r in state["relics"]),
               list(state["potions"]), list(state["keys"]), list(state["decision"]["boss_relics"])]
    root_seed = int.from_bytes(hashlib.blake2b(json.dumps(visible).encode("ascii"),digest_size=8).digest(),"little")
    for i in range(count):
        seed = (root_seed + 279771 + 100003 * i) % (1 << 64)
        particle = env._native.public_strategic_particle(seed,observed)
        public_offers = set(a.bits for a in particle.legal_actions())
        if not set(offers).issubset(public_offers):
            raise ValueError("resampling changed the offered relics")
        for a in offers:
            score = branch_value(particle.clone(),a,sts,router,seed)
            if score is None:
                return incumbent, {}
            values[a].append(score)
    means = {a:sum(v)/len(v) for a,v in values.items()}
    best = max(offers,key=lambda a:(means[a],a == incumbent))
    # Leave a small advantage to the existing draft scorer for value beyond the horizon.
    return (best if means[best] > means[incumbent] + 3. else incumbent), values
