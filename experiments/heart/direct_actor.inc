int actor_item(int a) const {
    if(a<ACT_POTION) return card_index(a);
    if(a<ACT_DISCARD) return CARD_COUNT+int(o[17+(a-ACT_POTION)/5]);
    return 0;
}
int actor_target(int a) const {
    if(a<ACT_POTION && o[OBS_HAND+(a/5)*HAND_W+4]) return a%5;
    if(a>=ACT_POTION && a<ACT_DISCARD && potionRequiresTarget(Potion(actor_item(a)-CARD_COUNT-1)))
        return (a-ACT_POTION)%5;
    return -1;
}
ActorFeatures actor_features(int a,int target) const {
    bool play=a<ACT_POTION,drink=a>=ACT_POTION && a<ACT_DISCARD;
    auto h=o+OBS_HAND+(play?a/5:0)*HAND_W;
    C id=play?C(int(h[0])-1):C::INVALID;
    bool up=play && h[1];
    auto f=cardFacts[up][int(id)];
    auto m=enemy(std::max(0,target));
    float target_in=target>=0?m[8]*m[9]:0;
    float total_hp=0, lowest=1e4, max_hit=0, occupied=0;
    for(int i=0;i<5;++i) {
        auto mob=enemy(i);
        if(mob[6]) {
            total_hp+=mob[1]; lowest=std::min(lowest,mob[1]+mob[3]);
            max_hit=std::max(max_hit,mob[8]*mob[9]);
        }
        occupied+=o[17+i]>0;
    }
    auto room=Room(int(o[8])-1);
    float str=p[N_PLAYER_BASE+int(PS::STRENGTH)];
    float base=play?std::max(0,getBaseDamage(id,up)):0;
    return {o[4]/4,o[0]/80,float(missing),p[2]/30,p[3]/4,p[4]/8,p[5]/8,p[6]/20,
            float(incoming/35),float(live)/3,total_hp/160,(live?lowest:0)/50,max_hit/30,
            str/10,p[N_PLAYER_BASE+int(PS::DEXTERITY)]/5,p[N_PLAYER_BASE+int(PS::WEAK)]/3,
            occupied/std::max(1.f,o[16]),float(room==Room::BOSS),float(room==Room::ELITE),
            target>=0?m[1]/80:0,target>=0?m[3]/20:0,target_in/30,
            target>=0?m[MONSTER_BASE+int(MS::VULNERABLE)]/3:0,
            target>=0?m[MONSTER_BASE+int(MS::THORNS)]/5:0,
            play?h[2]/3:0,float(up),base/15,f[0]/3.f,f[3]/10.f,f[4]/3.f,f[5]/3.f,
            f[6]/6.f,float(f[1]),f[32]/3.f,f[33]/3.f,
            float(play && getCardType(id)==CardType::POWER),float(drink),float(a==ACT_END_TURN),
            float(std::max(0.,incoming-p[2])/std::max(1.f,p[0])),o[3]/56};
}
void direct_scores(const uint8_t *mask,double *out) const {
    std::fill(out,out+N_ACTIONS,-1e30);
    ActorVector hand{};
    const int n=int(p[5]);
    for(int i=0;i<n;++i) {
        const auto &vec=actor_hand[int(o[OBS_HAND+i*HAND_W])];
        for(int j=0;j<ACTOR_WIDTH;++j) hand[j]+=vec[j]/n;
    }
    for(int a=0;a<=ACT_END_TURN;++a) if(mask[a] && (a<ACT_DISCARD || a==ACT_END_TURN)) {
        int t=actor_target(a);
        auto x=actor_features(a,t);
        auto &item=actor_items[actor_item(a)];
        auto &target=actor_targets[t<0?0:int(enemy(t)[0])];
        double value=0;
        for(int j=0;j<ACTOR_WIDTH;++j) {
            float z=actor_bias[j]+hand[j]+item[j]+target[j];
            for(int i=0;i<ACTOR_INPUT;++i) z+=actor_weights[j][i]*x[i];
            value+=std::max(0.f,z)*actor_out[j];
        }
        out[a]=value;
    }
}
