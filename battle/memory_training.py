"""Train predictions from earlier fights; ingest complete, same-turn game transitions."""
from collections import defaultdict
import random

import torch
from torch.nn import functional as F

from .experience import records, transitions
from .memory import MemoryBank, MemoryInterpreter, integer_field, observe, save_memory, scale


def recording_experiences(path, through=None):
    def prefix():
        for row in records(path):
            yield row
            if through is not None and row.get("ticket") == through and row["event"] == "decision":
                break
    for step in transitions(prefix()):
        if step.after_state is not None:
            fight, seq = step.ticket.rsplit("-",1)
            item = observe(step.state, step.action, step.after_state, fight, int(seq))
            if item:
                yield item


def remember_through(folder, ticket, runtime, output):
    from uuid import UUID
    fight, _ = ticket.rsplit("-",1)
    path = folder / "recordings" / (str(UUID(fight)) + ".jsonl")
    model, bank = runtime
    old_size = len(bank.seen)
    for item in recording_experiences(path, through=ticket):
        bank.remember(item)
    if len(bank.seen) != old_size:
        save_memory(output, model, bank)


def integer_penalty(pred, units, natural):
    """L1 distance to a valid outcome, evaluated on the original game-unit grid."""
    raw = pred * units
    nearest = raw.detach().round()
    nearest = torch.where(natural, nearest.clamp_min(0), nearest)
    return ((raw - nearest).abs() / units).mean()


def fit(episodes, *, seed=19, updates=300, batch_size=32, integer_weight=0.):
    if integer_weight < 0:
        raise ValueError("integer_weight must be nonnegative")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    # An all-zero but observable quantity must remain forecastable on a new card.
    fields = sorted({"hand.arrivals", "energy", "target.current_hp", "player.block"} |
                    {k for e in episodes for k in e.changes})
    model = MemoryInterpreter(fields)
    discrete = [i for i,k in enumerate(fields) if integer_field(k)]
    units = model.prior.weight.new_tensor([scale(fields[i]) for i in discrete])
    natural = torch.tensor([fields[i] == "hand.arrivals" for i in discrete], device=units.device)
    opt = torch.optim.AdamW(model.parameters(), lr=.002)
    # Each query's support was available before that battle began. The cache
    # holds references to episodes, never labels belonging to its query fight.
    support, prior_fights = {}, MemoryBank()
    grouped = defaultdict(list)
    for e in episodes:
        grouped[e.fight].append(e)
    for fight in grouped.values():
        for e in fight:
            support[e.fight,e.cue.key] = prior_fights.candidates(e.cue)
        for e in fight:
            prior_fights.remember(e)
    for update in range(updates):
        queries = rng.choices(episodes, k=batch_size)
        supports = []
        for q in queries:
            choices = support[q.fight,q.cue.key]
            mode = rng.random()
            if mode < .25:
                choices = []
            elif mode < .5:
                choices = [s for s in choices if s.cue.key != q.cue.key]
            supports.append(rng.sample(choices,min(len(choices),16)))
        pred, _, _ = model(*model.tensors([q.cue for q in queries],supports))
        targets = pred.new_tensor([model.labels(q) for q in queries])
        loss = F.smooth_l1_loss(pred, targets)
        if integer_weight and discrete and update >= 3 * updates // 4:
            loss = loss + integer_weight * integer_penalty(pred[:,discrete], units, natural)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    return model.eval()
