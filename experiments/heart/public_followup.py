"""Resolve displayed card-selection options without giving the policy pile order or simulator state."""
from math import sqrt

# The source ABI differs from our other simulator starting at DISCARD_ONE (21).
HAND={1,4,5,8,14,15,19,20,21}
DISCARD={10,11,12}
DRAW={16,17,18,22}
MULTI_HAND={6,9}


def menu(state,actions):
    c=state["combat"];task=c["card_select_task"]
    def token(card):
        return 1+2*card[0]+int(card[1]>0)
    options={}
    for bits in actions:
        kind=bits>>29;idx=bits&0xffff
        if kind==2:
            if task in (2,3):
                ids=(1+2*c["card_select_cards"][idx],) if idx<3 else (0,)
            elif task in HAND or task in DISCARD or task==7 or task in DRAW:
                zone=("hand" if task in HAND else "discard_pile" if task in DISCARD else
                      "exhaust_pile" if task==7 else "draw_pile")
                # Only the card represented by this legal menu item is disclosed.
                # The scorer receives a card identity without its underlying pile index.
                if idx>=len(c[zone]):return None
                ids=(token(c[zone][idx]),)
            else:return None
        elif kind==3 and task in MULTI_HAND:
            ids=tuple(sorted(token(card) for i,card in enumerate(c["hand"][:10]) if idx&(1<<i)))
        else:return None
        options[bits]=ids
    contexts=[1.,c["energy"]/4.,c["player_hp"]/max(1,state["max_hp"]),len(c["hand"])/10.,
              min(10,c["turn"])/10.]
    return task,contexts,options


def choose(state,actions,params):
    shown=menu(state,actions)
    if shown is None:return min(actions),False
    task,x,options=shown
    if params is None:return min(options,key=lambda bits:(options[bits],bits)),False
    def score(tokens):
        n=len(tokens)
        size=params["size"][task][min(10,n)]
        contextual=sum(a*b for a,b in zip(size,x))
        return contextual+sum(params["card"][task][t] for t in tokens)/sqrt(max(1,n))
    # Equal model scores break by visible card features. Equivalent copies give the same score.
    return min(options,key=lambda bits:(-score(options[bits]),options[bits],bits)),True
