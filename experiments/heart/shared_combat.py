"""Public observation fields available with the same meaning in both simulator forks."""
import numpy as np
import sts_env as sts


def schema():
    return dict(size=sts.OBS_SIZE, zones=sts.OBS_LAYOUT, dome=sts.RELIC_NAMES.index("Runic Dome"),
                monster_status={s: sts.MONSTER_STATUS_NAMES.index(s) for s in ("STRENGTH","VULNERABLE","WEAK","POISON")})


def projected(batch, visible_intents=False):
    obs = np.ascontiguousarray(batch)
    out = np.zeros_like(obs)
    columns = [0,1,3,4,5,7,12,13,14,16,17,18,19,20,21,22,23]
    out[:,columns] = obs[:,columns]
    def slots(matrix, name):
        start, rows, width = sts.OBS_LAYOUT[name]
        return matrix[:,start:start+rows*width].reshape(len(matrix),rows,width)
    for name in ("deck","draw","discard","exhaust"):
        src, dest = slots(obs,name), slots(out,name)
        # Sorting removes the original ordering, including the hidden draw order.
        code = np.where(src[:,:,0]>0,2*src[:,:,0]+(src[:,:,1]>0),1e6)
        code.sort(axis=1)
        dest[:,:,0] = np.where(code<1e6,code//2,0)
        dest[:,:,1] = code%2
    relic_ids = slots(obs,"relics")[:,:,0]
    relics = np.sort(np.where(relic_ids>0,relic_ids,1e6),axis=1)
    slots(out,"relics")[:,:,0] = np.where(relics<1e6,relics,0)
    player = slots(out,"player")
    player[:,:,:7] = slots(obs,"player")[:,:,:7]
    player[:,:,8:] = slots(obs,"player")[:,:,8:]
    cols = [0,1,2,3,6,9] + [10+v for v in schema()["monster_status"].values()]
    if visible_intents:
        cols.extend([7,8])
    slots(out,"monsters")[:,:,cols] = slots(obs,"monsters")[:,:,cols]
    slots(out,"hand")[:,:,:5] = slots(obs,"hand")[:,:,:5]
    slots(out,"hand")[:,:,1] = slots(out,"hand")[:,:,1]>0
    return out


def demo_rows(root, split):
    dtype = np.dtype([("seed","<u8"),("step","<u4"),("obs","<i2",sts.OBS_SIZE),
                      ("legal","u1",81),("action","u1")])
    return np.memmap(root/f"obs-{split}.bin", dtype=dtype, mode="r")


def interchangeable(obs, legal, action):
    # Advantage is shared only for plays of equal visible hand rows on the same enemy.
    import torch
    n = len(action)
    ix = torch.arange(n, device=action.device)
    eq = torch.zeros_like(legal)
    eq[ix,action] = True
    off, count, width = sts.OBS_LAYOUT["hand"]
    hand = obs[:,off:off+count*width].reshape(n,count,width)
    source = (action//5).clamp(max=count-1)
    same = (hand == hand[ix,source,None,:]).all(-1) & (action < 50)[:,None]
    for slot in range(count):
        eq[ix,5*slot+action%5] |= same[:,slot]
    return eq & legal
