// Construct each practice fight from a new game. The planning interface only
// receives View, which contains the public observation and policy weights.
using Encounter=MonsterEncounter;
Encounter scenario_encounter(const View &v,int &primary) {
    Encounter other=Encounter::INVALID;
    for(int j=0;j<5;++j) if(v.enemy(j)[6]) {
        switch(MonsterId(int(v.enemy(j)[0])-1)) {
            case MonsterId::THE_GUARDIAN: primary=j; return Encounter::THE_GUARDIAN;
            case MonsterId::HEXAGHOST: primary=j; return Encounter::HEXAGHOST;
            case MonsterId::SLIME_BOSS: primary=j; return Encounter::SLIME_BOSS;
            case MonsterId::BRONZE_AUTOMATON: primary=j; return Encounter::AUTOMATON;
            case MonsterId::THE_CHAMP: primary=j; return Encounter::CHAMP;
            case MonsterId::THE_COLLECTOR: primary=j; return Encounter::COLLECTOR;
            case MonsterId::AWAKENED_ONE: primary=j; return Encounter::AWAKENED_ONE;
            case MonsterId::TIME_EATER: primary=j; return Encounter::TIME_EATER;
            case MonsterId::DONU: primary=j; return Encounter::DONU_AND_DECA;
            case MonsterId::DECA: primary=j; other=Encounter::DONU_AND_DECA; break;
            case MonsterId::SPIRE_SPEAR: primary=j; return Encounter::SHIELD_AND_SPEAR;
            case MonsterId::SPIRE_SHIELD: primary=j; other=Encounter::SHIELD_AND_SPEAR; break;
            default: break;
        }
    }
    return other;
}

// Only the fields in this key can affect generation. The previous public
// profile on each worker is usually reused throughout a boss fight.
std::vector<int> scenario_key(const View &v,Encounter encounter) {
    const float *o=v.o;
    std::vector<int> key{int(encounter),int(o[4]),int(o[1]),int(o[16]),int(o[5])};
    for(int i=0;i<5;++i) key.push_back(int(o[17+i]));
    for(int i=0;i<MAX_DECK && o[OBS_DECK+i*CARD_W];++i) {
        auto c=o+OBS_DECK+i*CARD_W;
        key.push_back(int(c[0])*256+int(c[1]));
    }
    key.push_back(-1);
    for(int i=0;i<MAX_RELICS && o[OBS_RELICS+i*RELIC_W];++i)
        key.push_back(int(o[OBS_RELICS+i*RELIC_W]));
    return key;
}
void couple_opening_shuffle(GameContext &gc,int sample);
Env scenario_start(const std::vector<int> &key,int sample,int hp=-1,int floor=-1,
                   Room room=Room::INVALID,int gold=0,bool paired=false) {
    Env e;
    e.record=false;
    // Fixed artificial seeds let every tactic face the same shuffled deck and
    // random enemy moves. No real run's seed is passed to this function.
    auto enc=Encounter(key[0]);
    e.reset(0xaf761003ULL+104729ULL*sample,key[4],true,CharacterClass::IRONCLAD);
    auto &gc=e.gc;
    gc.act=key[1]; gc.floorNum=gc.act==1?16:gc.act==2?33:gc.act==3?50:55;
    gc.curRoom=enc==Encounter::SHIELD_AND_SPEAR?Room::ELITE:Room::BOSS;
    gc.maxHp=key[2]; gc.curHp=std::max(1,gc.maxHp*(55+15*(sample%4))/100);
    if(hp>=0) gc.curHp=hp;
    if(floor>=0) gc.floorNum=floor;
    if(room!=Room::INVALID) gc.curRoom=room;
    gc.gold=gold; gc.deck=Deck(); gc.relics=RelicContainer();
    gc.potionCapacity=key[3]; gc.potionCount=0; gc.potions.fill(Potion::EMPTY_POTION_SLOT);
    for(int i=0;i<gc.potionCapacity && i<5;++i) if(key[5+i]) {
        gc.potions[i]=Potion(key[5+i]-1); ++gc.potionCount;
    }
    size_t k=10;
    for(;key[k]!=-1;++k) gc.deck.obtainRaw(Card(C(key[k]/256-1),key[k]%256));
    for(++k;k<key.size();++k) gc.relics.add({R(key[k]-1),0});
    if(paired) couple_opening_shuffle(gc,sample);
    gc.screenState=ScreenState::BATTLE;
    gc.regainControlAction=[](GameContext &g){g.screenState=ScreenState::REWARDS;};
    gc.info.encounter=enc;
    e.inBattle=true;
    e.bc=BattleContext(); e.bc.init(gc,enc);
    return e;
}
double scenario_return(Env e,const Weights &w,int tactic) {
    struct Guard {
        int old=simulated_tactic;
        explicit Guard(int value){simulated_tactic=value;}
        ~Guard(){simulated_tactic=old;}
    } guard(tactic);
    std::array<float,OBS_SIZE> obs;
    std::array<uint8_t,N_ACTIONS> mask;
    std::array<double,N_ACTIONS> scores;
    double hpRemaining=1.;
    for(int step=0;step<380 && e.inBattle && !e.done() && e.bc.outcome==Outcome::UNDECIDED;++step) {
        e.observe(obs.data()); e.mask(mask.data());
        View view(obs.data(),w); view.all_scores(mask.data(),scores.data());
        int primary=0;
        if(scenario_encounter(view,primary)!=Encounter::INVALID)
            hpRemaining=view.enemy(primary)[1]/std::max(1.f,view.enemy(primary)[2]);
        int choice=int(std::max_element(scores.begin(),scores.end())-scores.begin());
        if(!e.step(choice)) throw std::runtime_error("illegal practice fight decision");
    }
    if(e.bc.outcome==Outcome::PLAYER_VICTORY)
        return 1.+.25*std::max(0,e.bc.player.curHp)/std::max(1,e.bc.player.maxHp);
    return .24*(1.-hpRemaining);
}
bool View::scenario_scores(const uint8_t *mask,double *out) const {
    if(Room(int(o[8])-1)!=Room::BOSS && o[4]!=4) return false;
    int primary=0;
    Encounter enc=scenario_encounter(*this,primary);
    if(enc==Encounter::INVALID) return false;
    int choice=simulated_tactic;
    if(choice<0) {
        struct Cache { unsigned version=0; std::vector<int> key; Weights weights{}; int selected=0; };
        thread_local Cache cache;
        auto key=scenario_key(*this,enc);
        if(cache.version!=scenario_version || cache.key!=key || cache.weights!=w) {
            std::array<double,12> totals{};
            for(int s=0;s<scenario_draws;++s) {
                Env entry=scenario_start(key,s);
                for(int i=0;i<12;++i) totals[i]+=scenario_return(entry,w,i);
            }
            cache={scenario_version,std::move(key),w,int(std::max_element(totals.begin(),totals.end())-totals.begin())};
        }
        choice=cache.selected;
    }
    if(choice==0) return false; // Keep the inherited boss policy as one playable option.
    apply_tactics(mask,out,primary,scenario_menu[choice]);
    return true;
}
