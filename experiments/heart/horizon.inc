// Longer encounters change the return on investing energy this turn.
double lasting(C id,bool up) const {
    double hp=0;
    for(int i=0;i<5;++i) if(enemy(i)[6]) hp+=enemy(i)[1];
    double turns=std::clamp(hp/(18.+std::max(0.,double(p[N_PLAYER_BASE+int(PS::STRENGTH)]))*2),1.,8.);
    double strength=p[N_PLAYER_BASE+int(PS::STRENGTH)];
    bool long_fight=turns>=4;
    auto status=[this](PS id){return p[N_PLAYER_BASE+int(id)];};
    switch(id) {
    case C::DEMON_FORM: return turns*(turns-1)*(up?1.8:1.4);
    case C::INFLAME: return turns*(up?5.:3.5);
    case C::METALLICIZE: return turns*(up?3.5:2.5);
    case C::CORRUPTION: return relic(R::DEAD_BRANCH)?100.:
        (num(C::FEEL_NO_PAIN)+num(C::DARK_EMBRACE)>0?26.:13.);
    case C::FEEL_NO_PAIN: return (status(PS::CORRUPTION) || relic(R::DEAD_BRANCH)?8.:2.)*turns;
    case C::DARK_EMBRACE: return (status(PS::CORRUPTION) || relic(R::DEAD_BRANCH)?6.:1.5)*turns;
    case C::BARRICADE: if(!synergy_enabled) return (num(C::ENTRENCH) || num(C::IMPERVIOUS)?4.:1.5)*turns;
        return (num(C::ENTRENCH) || num(C::IMPERVIOUS)?8.:3.)*turns;
    case C::ENTRENCH: return synergy_enabled && status(PS::BARRICADE)?std::min(90.,double(p[2])*1.8):0.;
    case C::POWER_THROUGH: case C::IMPERVIOUS: return synergy_enabled && status(PS::BARRICADE)?25.:0.;
    case C::BERSERK: return long_fight && (up || incoming==0)?20.:-16.;
    case C::BRUTALITY: return turns*1.8;
    case C::JUGGERNAUT: return turns*(up?6.:4.);
    case C::LIMIT_BREAK: return strength>0?std::min(80.,strength*turns*1.5):-10.;
    case C::DISARM: return long_fight?22.:3.;
    case C::SHOCKWAVE: return std::min(4.,turns)*(live>1?6.:4.);
    case C::REAPER: return strength==0 && p[4]<3 && (status(PS::DEMON_FORM) || relic(R::BRIMSTONE))?-14.:0.;
    case C::OFFERING: return p[5]<8 && p[0]>7?16.:-22.;
    case C::BITE: return missing>0?4.:0.;
    case C::APPARITION: return status(PS::INTANGIBLE) && up?-24.:0.;
    case C::DOUBLE_TAP: return turns>1 && p[3]>=2?14.:-3.;
    case C::FIEND_FIRE: return status(PS::FEEL_NO_PAIN)*std::min(6.f,p[5]-1);
    default: return 0.;
    }
}
double reaction(int a) const {
    auto h=o+OBS_HAND+(a/5)*HAND_W;
    auto id=C(int(h[0])-1);
    auto m=enemy(a%5);
    auto foe=MonsterId(int(m[0])-1);
    double value=lasting(id,h[1]);
    if(h[4] && getCardType(id)==CardType::ATTACK) {
        if(foe==MonsterId::DONU || foe==MonsterId::REPTOMANCER || foe==MonsterId::RED_SLAVER) value+=6;
        if(foe==MonsterId::BRONZE_ORB && enemy(1)[1]<70) value-=10;
        if(foe==MonsterId::SPIKER) value-=m[MONSTER_BASE+int(MS::THORNS)]*2;
        if(foe==MonsterId::WRITHING_MASS && incoming<12 && p[3]<=1 && m[1]>30) value-=12;
    }
    if(getCardType(id)==CardType::ATTACK && p[N_PLAYER_BASE+int(PS::DOUBLE_TAP)]) {
        value+=std::max(0,getBaseDamage(id,h[1]))*.8;
    }
    if(relic(R::DEAD_BRANCH) && (doesCardExhaust(id,h[1]) || (p[N_PLAYER_BASE+int(PS::CORRUPTION)] && getCardType(id)==CardType::SKILL))) value+=5;
    if(enemy(0)[MONSTER_BASE+int(MS::BEAT_OF_DEATH)]) {
        auto f=cardFacts[h[1]!=0][int(id)];
        value-=6;
        if(f[3] && p[2]<3) value+=4;
        if(p[0]<=3 && p[2]<2 && !f[3]) value-=100;
    }
    if(incoming>=p[0]+p[2] && cardFacts[h[1]!=0][int(id)][3]>0) value+=18;
    return value;
}
