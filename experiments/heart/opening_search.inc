Encounter opening_encounter(const View &v,Room room) {
    int slot=0;
    if(room==Room::BOSS) return scenario_encounter(v,slot);
    if(room==Room::ELITE) {
        for(int j=0;j<5;++j) switch(MonsterId(int(v.enemy(j)[0])-1)) {
            case MonsterId::GREMLIN_NOB: return Encounter::GREMLIN_NOB;
            case MonsterId::LAGAVULIN: return Encounter::LAGAVULIN;
            case MonsterId::SENTRY: return Encounter::THREE_SENTRIES;
            case MonsterId::GREMLIN_LEADER: return Encounter::GREMLIN_LEADER;
            case MonsterId::TASKMASTER: return Encounter::SLAVERS;
            case MonsterId::BOOK_OF_STABBING: return Encounter::BOOK_OF_STABBING;
            default: break;
        }
    }
    if(room!=Room::MONSTER || opening_scope==0 || (v.o[4]==1 && opening_scope==1)) return Encounter::INVALID;
    for(int j=0;j<5;++j) switch(MonsterId(int(v.enemy(j)[0])-1)) {
        case MonsterId::CENTURION: return Encounter::CENTURION_AND_HEALER;
        case MonsterId::SNAKE_PLANT: return Encounter::SNAKE_PLANT;
        case MonsterId::SNECKO: return Encounter::SNECKO;
        case MonsterId::SHELLED_PARASITE: return v.live==1?Encounter::SHELL_PARASITE:Encounter::SHELLED_PARASITE_AND_FUNGI;
        case MonsterId::CHOSEN:
            if(v.live==1) return Encounter::CHOSEN;
            for(int k=0;k<5;++k) if(int(v.enemy(k)[0])==1+int(MonsterId::CULTIST)) return Encounter::CULTIST_AND_CHOSEN;
            return Encounter::CHOSEN_AND_BYRDS;
        case MonsterId::MUGGER: return Encounter::TWO_THIEVES;
        case MonsterId::SPHERIC_GUARDIAN: return v.live==1?Encounter::SPHERIC_GUARDIAN:Encounter::SENTRY_AND_SPHERE;
        default: break;
    }
    switch(MonsterId(int(v.enemy(0)[0])-1)) {
        case MonsterId::CULTIST: return v.live==1?Encounter::CULTIST:Encounter::THREE_CULTIST;
        case MonsterId::JAW_WORM: return Encounter::JAW_WORM;
        case MonsterId::BYRD: return Encounter::THREE_BYRDS;
        case MonsterId::LOOTER: return v.live==1?Encounter::LOOTER:Encounter::TWO_THIEVES;
        case MonsterId::GREEN_LOUSE: case MonsterId::RED_LOUSE: return v.live==2?Encounter::TWO_LOUSE:Encounter::THREE_LOUSE;
        default: return Encounter::INVALID;
    }
}

// Match public facts against an independently generated opponent. Conditioning
// on intent preserves information a player can see without reading a hidden move.
bool match_opening(Env &e,const View &v,int sample) {
    if(e.bc.outcome!=Outcome::UNDECIDED) return false;
    std::array<float,OBS_SIZE> obs;
    e.observe(obs.data());
    for(int j=0;j<5;++j) {
        const float *src=v.enemy(j),*dst=obs.data()+OBS_MONSTERS+j*MONSTER_W;
        if(src[0]!=dst[0] || src[6]!=dst[6] || src[7]!=dst[7] || src[8]!=dst[8] || src[9]!=dst[9]) return false;
        if(src[6]) {
            auto &m=e.bc.monsters.arr[j];
            m.curHp=int(src[1]);m.maxHp=int(src[2]);m.block=int(src[3]);
        }
    }
    const auto &p=e.bc.player;
    if(p.curHp!=v.o[0] || obs[OBS_PLAYER+N_PLAYER_BASE+int(PS::STRENGTH)]!=v.p[N_PLAYER_BASE+int(PS::STRENGTH)])
        return false;
    auto &cards=e.bc.cards;
    while(cards.cardsInHand) {
        int j=cards.cardsInHand-1;
        auto card=cards.hand[j];cards.removeFromHandAtIdx(j);cards.moveToDrawPileTop(card);
    }
    for(int j=0;j<int(v.p[5]);++j) {
        auto h=v.o+OBS_HAND+j*HAND_W;
        auto it=std::find_if(cards.drawPile.begin(),cards.drawPile.end(),[&](const CardInstance &c){
            return int(c.id)+1==int(h[0]) && c.upgraded==bool(h[1]);
        });
        if(it==cards.drawPile.end()) return false;
        auto card=*it;cards.removeFromDrawPileAtIdx(int(it-cards.drawPile.begin()));
        card.costForTurn=int(h[2]);cards.moveToHand(card);
    }
    if(int(cards.drawPile.size())!=int(v.p[6]) || cards.drawPile.size()>MAX_PILE) return false;
    std::array<std::array<int,2>,CARD_COUNT> balance{};
    for(int i=0;i<int(cards.drawPile.size());++i) {
        auto row=v.o+OBS_DRAW+i*CARD_W;
        if(row[0]<1 || row[0]>CARD_COUNT) return false;
        --balance[int(row[0])-1][row[1]!=0];
        auto &card=cards.drawPile[i]; ++balance[int(card.id)][card.upgraded];
    }
    for(const auto &pair:balance) if(pair[0] || pair[1]) return false;
    std::mt19937 shuffle(24571u+7919u*sample);
    std::shuffle(cards.drawPile.begin(),cards.drawPile.end(),shuffle);
    e.bc.player.energy=int(v.p[3]); e.bc.player.block=int(v.p[2]);
    return true;
}

double finish_opening(Env e,const Weights &w,int first) {
    const int potions=e.gc.potionCount;
    if(!e.step(first)) throw std::runtime_error("invalid matched first play");
    std::array<float,OBS_SIZE> obs;
    std::array<uint8_t,N_ACTIONS> mask;
    std::array<double,N_ACTIONS> scores;
    double remaining=1.;
    for(int k=0;k<360 && e.inBattle && !e.done();++k) {
        e.observe(obs.data());e.mask(mask.data());View v(obs.data(),w);
        double hp=0,maxHp=0;
        for(int j=0;j<5;++j) if(v.enemy(j)[0]) {hp+=std::max(0.f,v.enemy(j)[1]);maxHp+=v.enemy(j)[2];}
        remaining=hp/std::max(1.,maxHp);
        v.all_scores(mask.data(),scores.data());
        if(!e.step(int(std::max_element(scores.begin(),scores.end())-scores.begin())))
            throw std::runtime_error("invalid continuation in matched opening");
    }
    double health=e.bc.outcome==Outcome::PLAYER_VICTORY?35.+e.bc.player.curHp:-20.*remaining;
    return health-4.*(potions-e.bc.potionCount);
}

void View::opening_scores(const uint8_t *mask,double *out) const {
    if(o[OBS_DISCARD] || o[OBS_EXHAUST] || p[5]<1 || p[6]+p[5]!=o[22]) return;
    auto room=Room(int(o[8])-1);
    Encounter enc=opening_encounter(*this,room);
    if(enc==Encounter::INVALID) return;
    std::vector<int> actions;
    for(int j=0;j<=ACT_END_TURN;++j) if(mask[j] && (j<ACT_DISCARD || j==ACT_END_TURN)) actions.push_back(j);
    std::stable_sort(actions.begin(),actions.end(),[&](int a,int b){return out[a]>out[b];});
    std::vector<int> candidates;
    for(int a:actions) {
        bool same=false;
        if(a<ACT_POTION) for(int b:candidates) if(b<ACT_POTION && a%5==b%5) {
            auto ha=o+OBS_HAND+a/5*HAND_W,hb=o+OBS_HAND+b/5*HAND_W;
            if(ha[0]==hb[0] && ha[1]==hb[1] && ha[2]==hb[2]) same=true;
        }
        if(!same) candidates.push_back(a);
        if(candidates.size()==5) break;
    }
    if(candidates.size()<2) return;
    if(opening_audit) ++(*opening_audit)[0];
    struct Guard { Guard(){opening_branch=true;} ~Guard(){opening_branch=false;} } guard;
    auto key=scenario_key(*this,enc);
    int accepted=0;
    std::array<double,5> totals{};
    for(int s=0;s<5*opening_draws && accepted<opening_draws;++s) {
        auto e=scenario_start(key,83+s,int(o[0]),int(o[3]),room,int(o[2]));
        if(!match_opening(e,*this,s)) continue;
        std::array<uint8_t,N_ACTIONS> synthetic;
        e.mask(synthetic.data());
        if(std::any_of(candidates.begin(),candidates.end(),[&](int a){return !synthetic[a];})) continue;
        for(int j=0;j<int(candidates.size());++j) totals[j]+=finish_opening(e,w,candidates[j]);
        ++accepted;
    }
    if(!accepted) return;
    for(int j=1;j<int(candidates.size());++j) out[candidates[j]]+=opening_scale*(totals[j]-totals[0])/accepted;
    if(opening_audit) {
        ++(*opening_audit)[1]; (*opening_audit)[2]+=accepted;
        (*opening_audit)[3]+=candidates[0]!=int(std::max_element(out,out+N_ACTIONS)-out);
    }
}
