// Action utilities use only the fork's public observation and legal mask.
// The simulator and episode log are available exclusively to the rollout driver.
#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "constants/CardFacts.h"
#include "constants/Events.h"
#include "constants/RelicPotionFacts.h"
#include "env/Env.h"

namespace py = pybind11;
using namespace sts;
using namespace sts::env;
using C = CardId;
using R = RelicId;

enum Weight { Defense, Kill, Focus, Scaling, Draw, Debuff, Elite, Rest, ShopVisit, Hall,
              Draft, Duplicate, Heal, PotionUse, CostPenalty, KeyRoute,
              Swap, EndgameDraft, EnergyRelic, Finisher, Lookahead, Count };
using Weights = std::array<double, Count>;
using EventTable = std::array<std::array<double,16>,std::size(eventGameNames)>;
static EventTable event_values{};

double card_prior(C id) {
    switch (id) {
    case C::APOTHEOSIS: case C::OFFERING: case C::IMMOLATE: case C::FEED: return 11;
    case C::APPARITION: return 14;
    case C::REAPER: case C::DEMON_FORM: case C::DISARM: case C::IMPERVIOUS: return 9;
    case C::BLUDGEON: case C::SHOCKWAVE: case C::UPPERCUT: case C::BATTLE_TRANCE: case C::FLAME_BARRIER:
    case C::INFLAME: case C::SHRUG_IT_OFF: case C::FIEND_FIRE: case C::SPOT_WEAKNESS: return 8;
    case C::WHIRLWIND: case C::POMMEL_STRIKE: case C::LIMIT_BREAK: case C::PUMMEL:
    case C::CORRUPTION: case C::FEEL_NO_PAIN: case C::ANGER: case C::CARNAGE:
    case C::DARK_EMBRACE: case C::POWER_THROUGH: case C::SECOND_WIND: return 7;
    case C::HEMOKINESIS: case C::PERFECTED_STRIKE: case C::IRON_WAVE: case C::TWIN_STRIKE:
    case C::METALLICIZE: case C::ARMAMENTS: case C::HEAVY_BLADE: case C::SWORD_BOOMERANG:
    case C::HEADBUTT: case C::TRUE_GRIT: case C::BURNING_PACT: case C::CLEAVE:
        return 6;
    default: break;
    }
    if (getCardType(id) == CardType::CURSE || getCardType(id) == CardType::STATUS) return -12;
    if (id == C::STRIKE_RED || id == C::DEFEND_RED) return -2;
    return 3;
}

struct View {
    const float *o, *p;
    std::array<int, std::size(cardNames)> deck{};
    double missing, incoming = 0;
    int live = 0;
    const Weights &w;
    View(const float *obs, const Weights &weights): o(obs), p(obs+OBS_PLAYER), w(weights) {
        missing = 1. - o[0] / std::max(1.f, o[1]);
        for (int i=0; i<MAX_DECK; ++i) {
            int id = int(o[OBS_DECK + i*CARD_W])-1;
            if (id<0) break;
            ++deck[id];
        }
        for (int i=0; i<5; ++i) {
            auto m = enemy(i);
            if (m[6]) { ++live; incoming += m[8]*m[9]; }
        }
    }
    const float *enemy(int t) const { return o+OBS_MONSTERS+t*MONSTER_W; }
    int num(C id) const { return deck[int(id)]; }
    bool relic(R id) const {
        for (int i=0; i<MAX_RELICS; ++i) {
            int r=int(o[OBS_RELICS+i*RELIC_W])-1;
            if (r==int(id)) return true;
            if (r<0) break;
        }
        return false;
    }
    double draft(C id,bool up) const {
        double score=card_prior(id)+1.5*up;
        int scaling=num(C::DEMON_FORM)+num(C::INFLAME)+num(C::SPOT_WEAKNESS);
        if (o[4]>1) {
            if (id==C::DEMON_FORM && !scaling) score+=8;
            if (id==C::SPOT_WEAKNESS && !num(id)) score+=5;
            if ((id==C::HEAVY_BLADE || id==C::PUMMEL || id==C::REAPER) && scaling) score+=3;
            if (id==C::DISARM && num(id)<2) score+=4;
            if (id==C::IMPERVIOUS && num(id)<2) score+=3;
            if (id==C::CORRUPTION && num(C::FEEL_NO_PAIN)) score+=5;
            if (id==C::BODY_SLAM && (num(C::BARRICADE)||num(C::ENTRENCH))) score+=9;
            if (id==C::ENTRENCH && num(C::BARRICADE)) score+=9;
        }
        if (getCardType(id)==CardType::ATTACK && o[4]==1) {
            double printed=std::max(0,getBaseDamage(id,up))*std::max(1,int(cardFacts[up][int(id)][0]));
            if(id==C::PERFECTED_STRIKE) printed+=2*(num(C::STRIKE_RED)+num(C::POMMEL_STRIKE)+1);
            score+=std::min(3.5,printed/6.)*std::max(0.,(17.-o[22])/6.);
        }
        if (id==C::DEMON_FORM && num(id)==0) score+=2;
        if (id==C::CORRUPTION && !num(C::FEEL_NO_PAIN) && !relic(R::DEAD_BRANCH)) score-=3;
        if (id==C::BARRICADE && num(C::IMPERVIOUS)+num(C::POWER_THROUGH)>0) score+=5;
        if (id==C::LIMIT_BREAK) score += (num(C::INFLAME)+num(C::SPOT_WEAKNESS)+num(C::DEMON_FORM)+int(relic(R::VAJRA))) ? 3 : -7;
        if (id==C::CORRUPTION && relic(R::DEAD_BRANCH)) score+=15;
        if ((id==C::DARK_EMBRACE || id==C::FEEL_NO_PAIN) && num(C::CORRUPTION)) score+=4;
        if (id==C::IMMOLATE && o[4]==1) score+=3;
        if (id==C::DEMON_FORM && o[4]>1) score+=3;
        return score-w[Duplicate]*num(id);
    }
    double play(int a) const {
        const float *h=o+OBS_HAND+HAND_W*(a/5), *m=enemy(a%5);
        C id=C(int(h[0])-1);
        bool up=h[1]!=0;
        auto f=cardFacts[up][int(id)];
        double str=p[N_PLAYER_BASE+int(PlayerStatus::STRENGTH)];
        double dex=p[N_PLAYER_BASE+int(PlayerStatus::DEXTERITY)];
        int energy=int(p[3]), cost=std::max(0,int(h[2]));
        double hits=std::max(1,int(f[0])), base=std::max(0,getBaseDamage(id,up));
        if (id==C::WHIRLWIND) hits=energy;
        if (id==C::HEAVY_BLADE) str*=up?5:3;
        if (id==C::BODY_SLAM) base=p[2];
        if (id==C::PERFECTED_STRIKE)
            base+=(up?3:2)*(num(C::STRIKE_RED)+num(C::POMMEL_STRIKE)+num(C::TWIN_STRIKE)+num(id)+num(C::WILD_STRIKE));
        double dmg=base || getCardType(id)==CardType::ATTACK ? std::max(0.,base+str)*hits : 0.;
        if (p[N_PLAYER_BASE+int(PlayerStatus::WEAK)]) dmg*=.75;
        if (m[MONSTER_BASE+int(MonsterStatus::VULNERABLE)] && h[4]) dmg*=1.5;
        if (id==C::FIEND_FIRE) dmg*=std::max(1.f,p[5]-1);
        double targetHp=m[1]+m[3];
        double score=dmg*(f[1]?1.+.72*(live-1):1.);
        if (h[4] && dmg>0) {
            score+=(3+w[Focus])*20./(targetHp+5);
            if (dmg>=targetHp) score+=w[Kill]+m[8]*m[9];
            else if (energy>cost && live>1) {
                // Credit progress toward a kill achievable with another visible card.
                for(int j=0;j<int(p[5]);++j) {
                    auto h2=o+OBS_HAND+j*HAND_W;
                    if(j==a/5 || !h2[3] || h2[2]>energy-cost) continue;
                    auto next=C(int(h2[0])-1);
                    int base2=getBaseDamage(next,h2[1]);
                    if(base2>0 && dmg+base2+str>=targetHp) {
                        score+=w[Finisher]+.65*m[8]*m[9]; break;
                    }
                }
            }
            if (m[MONSTER_BASE+int(MonsterStatus::INTANGIBLE)]) score*=.18;
        }
        double blk=f[3]>0?std::max(0.,f[3]+dex):0;
        if (id==C::SHRUG_IT_OFF && !blk) blk=up?11:8;
        if (p[N_PLAYER_BASE+int(PlayerStatus::FRAIL)]) blk*=.75;
        double need=std::max(0.,incoming-p[2]);
        score+=w[Defense]*std::min(blk,need)+.09*blk;
        score+=w[Draw]*std::min(3,int(f[4]))*(energy>cost?1:.25);
        score+=w[Debuff]*(f[32] && !m[MONSTER_BASE+int(MonsterStatus::VULNERABLE)]?1.:0.);
        if (id==C::BASH && !m[MONSTER_BASE+int(MonsterStatus::VULNERABLE)])
            score+=w[Debuff]*(energy>=3?1.6:1);
        if (id==C::SHOCKWAVE || id==C::DISARM) score+=w[Debuff]*2.5;
        if (id==C::SHRUG_IT_OFF) score+=2;
        if (id==C::FLAME_BARRIER && need>0) score+=std::min(12.,incoming*.65);
        if (id==C::IMPERVIOUS && need<7) score-=10;
        if (id==C::ENTRENCH) score+=std::min(double(p[2]),need)*w[Defense];
        if (id==C::BLOODLETTING && energy>1) score-=10;
        if (id==C::EXHUME && o[OBS_EXHAUST]==0) score-=10;
        if (id==C::APPARITION) score+= (incoming>0 && !p[N_PLAYER_BASE+int(PlayerStatus::INTANGIBLE)])?55:3;
        if (id==C::BARRICADE && live) score+=8;
        if (id==C::SECOND_WIND) {
            int exhausted=0;
            for(int j=0;j<int(p[5]);++j) if (j!=a/5 &&
                getCardType(C(int(o[OBS_HAND+j*HAND_W])-1))!=CardType::ATTACK) ++exhausted;
            score+=w[Defense]*std::min(need,exhausted*(up?7.:5.));
        }
        if(id==C::BASH && m[MONSTER_BASE+int(MonsterStatus::VULNERABLE)]<=1 && live==1) score+=5;
        if(id==C::DISARM && m[MONSTER_BASE+int(MonsterStatus::ARTIFACT)]==0) score+=12;
        if(id==C::REAPER && str>3) score+=std::min(18.,str);
        if (id==C::REAPER && missing<.15 && live==1) score-=6;
        if (getCardType(id)==CardType::SKILL && enemy(0)[MONSTER_BASE+int(MonsterStatus::ENRAGE)])
            score-=13;
        if (id==C::SPOT_WEAKNESS && m[8]>0) score+=w[Scaling]*10;
        if (id==C::LIMIT_BREAK && str>0) score+=w[Scaling]*(4+str*2);
        if (getCardType(id)==CardType::POWER) score+=w[Scaling]*(8+card_prior(id))*(p[4]<5?1.:.75);
        if (id==C::OFFERING || id==C::APOTHEOSIS) score+=35;
        if (id==C::REAPER) score+=std::min(double(o[1]-o[0]),dmg*live)*1.4;
        if (id==C::FEED && dmg>=targetHp) score+=18;
        if (id==C::ARMAMENTS && up && p[5]>2) score+=10;
        if (id==C::ANGER && dmg>0) score+=2;
        score += f[5]*5 - f[6]*.7;
        return (score+.04)/(1+w[CostPenalty]*std::max(0,cost-1)) + (cost==0?1.2:0);
    }
    double map_score(int x) const {
        double next[7]{}, now[7]{};
        for (int y=14; y>int(o[11]); --y) {
            for (int col=0; col<7; ++col) {
                auto node=o+OBS_MAP+(7*y+col)*MAP_W;
                Room room=Room(int(node[0])-1);
                double value=0, tail=-1e8;
                switch(room) {
                case Room::REST: value=w[Rest]*(1+missing); break;
                case Room::SHOP: value=o[2]>130?w[ShopVisit]:-.8; break;
                case Room::ELITE: value=w[Elite]*(1.-2.2*missing)-(o[4]==1?5.:2.5); break;
                case Room::MONSTER: value=w[Hall]*(o[3]<5?1.8:1.); break;
                default: value=.3; break;
                }
                if (!o[13] && col==o[24] && y==o[25])
                    value+=(o[4]==3?100.:w[KeyRoute]*(o[4]==2?.9:.15));
                for(int delta=-1;delta<=1;++delta) {
                    int dest=col+delta;
                    if(dest>=0 && dest<7 && (int(node[1]) & (1<<(delta+1)))) tail=std::max(tail,next[dest]);
                }
                now[col]=value+.94*(tail < -1e7 ? 0.:tail);
            }
            std::copy(now,now+7,next);
        }
        return next[x];
    }
    double relic_value(R id) const {
        switch(id) {
        case R::PANDORAS_BOX: return 20;
        case R::SNECKO_EYE: return 15;
        case R::RUNIC_PYRAMID: return 8;
        case R::COFFEE_DRIPPER: return 4;
        case R::SOZU: case R::FUSION_HAMMER: case R::CURSED_KEY: return w[EnergyRelic];
        case R::VELVET_CHOKER: case R::ECTOPLASM: return 7;
        case R::MARK_OF_PAIN: case R::BUSTED_CROWN: return 5;
        case R::DEAD_BRANCH: case R::TORII: case R::TUNGSTEN_ROD: case R::FOSSILIZED_HELIX: return 13;
        case R::BRIMSTONE: return -8;
        case R::PRISMATIC_SHARD: return -15;
        case R::RUNIC_DOME: return -18;
        default: return 6;
        }
    }
    double potion(int a) const {
        if(a>=ACT_DISCARD) return -80;
        int slot=(a-ACT_POTION)/5;
        auto id=Potion(int(o[17+slot])-1);
        if(id==Potion::FRUIT_JUICE) return 60;
        if(!o[7] || id==Potion::SMOKE_BOMB) return -65;
        bool boss=Room(int(o[8])-1)==Room::BOSS || o[4]==4;
        bool elite=Room(int(o[8])-1)==Room::ELITE;
        double risk=std::max(0.,incoming-p[2]);
        double urgency=w[PotionUse]*(double(boss)*19+double(elite)*15+12*missing+risk*.35)-11;
        if(id==Potion::BLOCK_POTION) return p[3]==0 && risk>3?urgency+risk:-20;
        if(id==Potion::FIRE_POTION || id==Potion::EXPLOSIVE_POTION || id==Potion::POISON_POTION) {
            auto m=enemy((a-ACT_POTION)%5);
            return urgency+(m[1]+m[3]<=20 && m[8]>0?8.:0.)+(p[3]==0?1.:-5.);
        }
        if(id==Potion::ENERGY_POTION || id==Potion::SWIFT_POTION) return p[3]==0?urgency:-15;
        return urgency+(p[4]<=1?4.:0.);
    }
    double score(int a) const {
        if(a<ACT_POTION) return play(a);
        if(a<ACT_END_TURN) return potion(a);
        if(a==ACT_END_TURN) return 0;
        if(a==ACT_CONFIRM) return 4;
        int i=a-ACT_CHOOSE;
        const float *c=o+OBS_CHOICES+i*CHOICE_W;
        ChoiceKind kind=ChoiceKind(int(c[0]));
        if(o[7]) {
            if(kind!=ChoiceKind::CARD) return kind==ChoiceKind::SKIP?-10:1;
            C id=C(int(c[1])-1);
            auto task=CardSelectTask(int(o[26]));
            if(c[3]) return -1000;
            double rank=card_prior(id)+c[2];
            if(task==CardSelectTask::EXHAUST_ONE || task==CardSelectTask::EXHAUST_MANY ||
               task==CardSelectTask::DISCARD || task==CardSelectTask::GAMBLE) return 3-rank;
            return 6+rank;
        }
        auto screen=ScreenState(int(o[6]));
        switch(screen) {
        case ScreenState::EVENT_SCREEN:
            if(kind==ChoiceKind::NEOW_OPTION) return i==3 ? w[Swap]-1 : -.001*i;
            if(kind==ChoiceKind::EVENT_OPTION && i<16)
                return event_values.at(int(o[9])-1)[i] - .0001*i;
            return -.001*i;
        case ScreenState::MAP_SCREEN: return map_score(i);
        case ScreenState::REST_ROOM:
            if(i==2) return !o[12] && (o[4]>=3 || (o[4]>=2 && missing<.18))?45:-4;
            if(i==0) return missing*w[Heal];
            return i==1?8.:i==3?8.5:2.;
        case ScreenState::TREASURE_ROOM: return 4-i;
        case ScreenState::BOSS_RELIC_REWARDS: return kind==ChoiceKind::RELIC?relic_value(R(int(c[1])-1)):0.;
        case ScreenState::CARD_SELECT: {
            C id=C(int(c[1])-1);
            bool remove=int(o[26])==1 || int(o[26])==2 || int(o[26])==4 || int(o[26])==8;
            return remove?-card_prior(id):card_prior(id)+(id==C::BASH && o[4]==1?9:0);
        }
        case ScreenState::SHOP_ROOM:
        case ScreenState::REWARDS: {
            bool shop=screen==ScreenState::SHOP_ROOM;
            switch(kind) {
            case ChoiceKind::KEY: return 90;
            case ChoiceKind::GOLD: return 60;
            case ChoiceKind::RELIC: return shop?relic_value(R(int(c[1])-1))-.027*c[3]:50;
            case ChoiceKind::POTION: return shop?(o[4]>=3?3.:0.)-.03*c[3]:35;
            case ChoiceKind::CARD_REMOVE: return 3.;
            case ChoiceKind::CARD: return draft(C(int(c[1])-1),c[2])-w[Draft]-
                w[EndgameDraft]*std::max(0.,double(o[22])-16) - (shop?.035*c[3]:0.);
            default: return 0;
            }
        }
        default: return -i*.001;
        }
    }
#include "turn_plan.inc"
};

struct Result { int floor=0, act=0, keys=0, steps=0; bool heart=false, truncated=false; };
Result advance(Env &e,const Weights &w,std::vector<int>* trace=nullptr,int stopAct=0) {
    std::array<float,OBS_SIZE> obs{};
    std::array<uint8_t,N_ACTIONS> mask{};
    Result r;
    for (;r.steps<3500 && !e.done();++r.steps) {
        e.observe(obs.data());
        if(stopAct && obs[4]>=stopAct) break;
        e.mask(mask.data());
        View v(obs.data(),w);
        std::array<double,N_ACTIONS> utilities;
        v.all_scores(mask.data(),utilities.data());
        double best=-1e30; int choice=-1;
        for(int i=0;i<N_ACTIONS;++i) if(mask[i]) {
            double s=utilities[i];
            if(s>best) {best=s;choice=i;}
        }
        if(choice<0 || !e.step(choice)) throw std::runtime_error("invalid policy action");
        if(trace) trace->push_back(choice);
    }
    e.observe(obs.data()); r.floor=int(obs[3]); r.act=int(obs[4]);
    r.keys=int(obs[12]+obs[13]+obs[14]); r.truncated=!e.done();
    r.heart=e.won() && r.act==4; // Python verifier additionally requires the logged final encounter.
    return r;
}

Result run(const Weights &w,uint64_t seed,std::vector<int>* trace=nullptr) {
    Env e; e.record=false; e.reset(seed,20,true,CharacterClass::IRONCLAD);
    return advance(e,w,trace);
}

// Natural prefixes from training seeds only. Snapshots initialise a rollout;
// the scorer still receives exactly the public observation used in fresh runs.
struct Bank {
    std::vector<Env> states;
    Bank(const Weights &w,uint64_t seed,int limit,int count,int act) {
        if(count<1 || limit<1 || act<2 || act>4) throw std::invalid_argument("curriculum range");
        for(int i=0;i<limit && int(states.size())<count;++i) {
            Env e; e.record=false; e.reset(seed+i,20,true,CharacterClass::IRONCLAD);
            auto r=advance(e,w,nullptr,act);
            if(!e.done() && r.act==act) states.push_back(e);
        }
    }
    std::vector<std::array<int,6>> evaluate(const Weights &w,int threads) const {
        std::vector<std::array<int,6>> rows(states.size());
        #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
        for(int i=0;i<int(states.size());++i) {
            Env e=states[i]; auto r=advance(e,w);
            rows[i]={r.floor,r.act,r.keys,r.steps,int(r.heart),int(r.truncated)};
        }
        return rows;
    }
};

PYBIND11_MODULE(heart_native,m) {
    py::class_<Bank>(m,"Bank").def(py::init<const Weights&,uint64_t,int,int,int>())
        .def("batch",&Bank::evaluate).def("__len__",[](const Bank &b){return b.states.size();});
    m.def("set_event_table",[](const EventTable &table){event_values=table;});
    m.def("batch",[](const Weights &w,uint64_t seed,int n,int threads){
        if(n<1 || threads<1) throw std::invalid_argument("positive batch dimensions required");
        std::vector<std::array<int,6>> out(n);
        #pragma omp parallel for num_threads(threads) schedule(dynamic,8)
        for(int j=0;j<n;++j) {
            auto r=run(w,seed+j);
            out[j]={r.floor,r.act,r.keys,r.steps,int(r.heart),int(r.truncated)};
        }
        return out;
    });
    m.def("trace",[](const Weights &w,uint64_t seed){std::vector<int> a;run(w,seed,&a);return a;});
    m.def("scores",[](const Weights &w,py::array_t<float,py::array::c_style|py::array::forcecast> obs,
                      py::array_t<uint8_t,py::array::c_style|py::array::forcecast> mask) {
        if(obs.size()!=OBS_SIZE || mask.size()!=N_ACTIONS) throw std::invalid_argument("observation or mask size");
        View v(obs.data(),w);
        std::vector<double> out(N_ACTIONS,-1e30);
        v.all_scores(mask.data(),out.data());
        return out;
    });
}
