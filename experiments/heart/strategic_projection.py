"""Project only displayed strategic fields from the reference simulator into our policy ABI."""
import json
import re
from pathlib import Path


class StrategicProjection:
    def __init__(self):
        layout=json.loads(Path("build/public-expert-intent/layout.json").read_text())
        self.size,self.zones=layout["size"],layout["zones"]
        text=Path("build/rl-fork/include/constants/Relics.h").read_text(encoding="utf8")
        relics=re.findall(r'"([^"]*)"',text.split("relicIds[]",1)[1].split("};",1)[0])
        self.counters={i for i,r in enumerate(relics) if r in ("Happy Flower","Incense Burner","InkBottle",
                       "Inserter","Nunchaku","Sundial","Pen Nib","Omamori","Girya","Tiny Chest")}
        self.positive={i for i,r in enumerate(relics) if r in ("NeowsBlessing","WingedGreaves","Matryoshka","NlothsMask")}
        self.doll=relics.index("Du-Vu Doll")
        cards=Path("build/rl-fork/include/constants/Cards.h").read_text(encoding="utf8")
        self.curses={i for i,t in enumerate(re.findall(r"CardType::([A-Z]+)",cards.split("cardTypes[]",1)[1].split("};",1)[0])) if t=="CURSE"}

    def encode(self,s,actions,burning):
        o=[0]*self.size;mapping={}
        screen=s["screen_state"];d=s["decision"];deck=s["deck"];mapx,mapy=s["map_position"]
        o[:8]=[s["hp"],s["max_hp"],s["gold"],s["floor"],s["act"],s["ascension"],screen,0]
        o[8:17]=[s["current_room"]+1,s["current_event"]+1 if screen==1 else 0,mapx,mapy,*s["keys"],
                  s["act_boss"]+1,s["potion_capacity"]]
        for i,p in enumerate(s["potions"][:5]):o[17+i]=p+1 if p>1 else 0
        o[22:26]=[len(deck),len(s["relics"]),*(burning or (-1,-1))]
        if screen==4:o[26:29]=[d["card_select_screen_type"],d["card_select_pick_count"],len(d["selected_cards"])]
        o[29]=-1;o[30]=d["treasure_chest_size"]+1 if screen==6 else 0;o[31]=s["outcome"]
        off,lim,width=self.zones["relics"]
        for i,(r,data) in enumerate(s["relics"][:lim]):
            count=(data if r in self.counters or r in self.positive and data>0 else
                   sum(c[0] in self.curses for c in deck) if r==self.doll else -1)
            o[off+i*width:off+i*width+2]=[r+1,count+1]
        off,lim,width=self.zones["deck"]
        for i,(card,up,*_) in enumerate(deck[:lim]):o[off+i*width:off+i*width+2]=[card+1,int(bool(up))]
        o[self.zones["class"][0]]=s["character_class"]+1
        rooms={}
        off=self.zones["map"][0]
        for x,y,room,edges in s["map"]:
            if not (0<=x<7 and 0<=y<15):continue
            rooms[x,y]=room
            # Row 14's long edge ends at the boss and is irrelevant to our planner.
            edge=sum(1<<(col-x+1) for col in edges if abs(col-x)<=1)
            o[off+2*(7*y+x):off+2*(7*y+x)+2]=[room+1,edge]
        slot=0;gold_index=0;start,limit,w=self.zones["choices"]
        for bits in actions:
            idx=bits&255;second=(bits>>8)&255;kind=(bits>>27)&7
            if bits&0x80000000:
                if idx<5:mapping[75+idx if bits&0x40000000 else 50+5*idx]=bits
                continue
            i=idx;label=0;a=b=extra=0
            if screen==1:
                if s["current_event"]==5:
                    a,b=d["neow_options"][idx];a+=1;b+=1;label=9
                else:
                    label=8;a=i
            elif screen==2:
                i=slot;slot+=1
                if kind==0 and second==5:label=14;extra=idx
                elif kind==0:
                    card,up,*_=d["card_rewards"][idx][second];label,a,b,extra=1,card+1,int(bool(up)),idx
                elif kind==1:
                    label=4;g=d["gold_rewards"];a=g[min(gold_index,len(g)-1)];gold_index+=1
                elif kind==2:label,a=5,1 if d["has_emerald_key"] else 2
                elif kind==3:label,a=3,d["potion_rewards"][idx]+1
                elif kind==4:label,a=2,d["relic_rewards"][idx]+1
                else:label=6
            elif screen==3:
                if idx<3:label,a=2,d["boss_relics"][idx]+1
                else:label=6
            elif screen==4:
                card,up,*_=d["selectable_cards"][idx];label,a,b=1,card+1,int(bool(up))
            elif screen==5:
                label,a,b=7,idx,7 if mapy==14 else rooms[idx,mapy+1]+1
            elif screen==6:label=12 if idx==0 else 6
            elif screen==7:label,a=10,idx
            elif screen==8:
                if kind==0:
                    card,up,_,price=d["shop_cards"][idx];label,a,b,extra=1,card+1,int(bool(up)),price
                elif kind==3:
                    p,price=d["shop_potions"][idx];i,label,a,extra=10+idx,3,p+1,price
                elif kind==4:
                    relic,price=d["shop_relics"][idx];i,label,a,extra=7+idx,2,relic+1,price
                elif kind==5:i,label,extra=13,11,d["shop_remove_cost"]
                elif kind==6:i,label=14,6
                else:raise ValueError(("shop",kind))
            else:raise ValueError(("screen",screen))
            if not 0<=i<limit:continue
            if 81+i in mapping:raise ValueError(("ambiguous menu",screen,i))
            mapping[81+i]=bits
            o[start+w*i:start+w*i+6]=[label,a,b,extra,0,1]
        if not mapping:raise ValueError(("unmapped legal menu",screen,actions))
        return o,mapping
