int boss_index(int &slot) const {
    for(int i=0;i<5;++i) if(enemy(i)[6]) {
        slot=i;
        switch(MonsterId(int(enemy(i)[0])-1)) {
            case MonsterId::BRONZE_AUTOMATON: return 0;
            case MonsterId::THE_CHAMP: return 1;
            case MonsterId::THE_COLLECTOR: return 2;
            default: break;
        }
    }
    return -1;
}
RecoveryInput recovery_input(int boss,int slot) const {
    RecoveryInput x{};
    auto features=deck_features();
    std::copy(features.begin(),features.begin()+14,x.begin());
    x[14+boss]=1.f;
    auto m=enemy(slot);
    x[17]=m[1]/std::max(1.f,m[2]); x[18]=p[4]/8; x[19]=p[3]/4; x[20]=p[2]/25;
    x[21]=incoming/40; x[22]=live/3.f; x[23]=m[8]*m[9]/40.f;
    x[24]=p[N_PLAYER_BASE+int(PS::STRENGTH)]/6; x[25]=p[N_PLAYER_BASE+int(PS::DEMON_FORM)]/3;
    x[26]=p[N_PLAYER_BASE+int(PS::WEAK)]/3; x[27]=m[MONSTER_BASE+int(MS::VULNERABLE)]/3;
    for(int h=0;h<int(p[5]);++h) {
        auto row=o+OBS_HAND+h*HAND_W;
        C id=C(int(row[0])-1);
        const auto f=cardFacts[row[1]!=0][int(id)];
        x[28]+=std::max(0,getBaseDamage(id,row[1]))*std::max(1,int(f[0]))/60.f;
        x[29]+=f[3]/30.f; x[30]+=float(getCardType(id)==CardType::POWER);
        x[31]+=f[4]/4.f;
    }
    x[32]=num(C::DEMON_FORM)+num(C::INFLAME)+num(C::SPOT_WEAKNESS);
    x[33]=num(C::CORRUPTION)+num(C::DARK_EMBRACE)+num(C::FEEL_NO_PAIN);
    x[34]=num(C::BARRICADE)+num(C::BODY_SLAM)+num(C::ENTRENCH);
    x[35]=num(C::APPARITION); x[36]=num(C::DISARM); x[37]=num(C::REAPER);
    x[38]=o[0]/80.f;
    for(int j=0;j<5;++j) x[39]+=float(o[17+j]>0);
    return x;
}
int recovery_choice(int boss,int slot) const {
    const auto x=recovery_input(boss,slot);
    RecoveryHidden hidden{};
    for(int j=0;j<32;++j) hidden[j]=std::max(0.f,std::inner_product(x.begin(),x.end(),recovery_layer[j].begin(),recovery_bias[j]));
    auto scores=recovery_offset;
    for(int j=0;j<16;++j) scores[j]=std::inner_product(hidden.begin(),hidden.end(),recovery_readout[j].begin(),scores[j]);
    return int(std::max_element(scores.begin(),scores.end())-scores.begin());
}
void apply_boss_tactics(const uint8_t *mask,double *scores) const {
    int slot=0, boss=boss_index(slot);
    if(boss<0) return;
    const auto &q=use_recovery?recovery_menu[recovery_choice(boss,slot)][boss]:boss_tactics[boss];
    apply_tactics(mask,scores,slot,q);
}
void apply_tactics(const uint8_t *mask,double *scores,int slot,const std::array<double,18> &q) const {
    const double early=1./(1.+p[4]*.22);
    const double hpFraction=enemy(slot)[1]/std::max(1.f,enemy(slot)[2]);
    const double incomingLeft=std::max(0.,incoming-p[2]);
    for(int a=0;a<=ACT_END_TURN;++a) if(mask[a]) {
        if(a>=ACT_POTION) {
            if(a<ACT_DISCARD) scores[a]+=q[15]+q[16]*early;
            if(a==ACT_END_TURN) scores[a]+=q[17];
            continue;
        }
        auto h=o+OBS_HAND+a/5*HAND_W;
        C id=C(int(h[0])-1);
        auto facts=cardFacts[h[1]!=0][int(id)];
        bool attack=getCardType(id)==CardType::ATTACK, power=getCardType(id)==CardType::POWER;
        double damage=std::max(0,getBaseDamage(id,h[1]))*std::max(1,int(facts[0]))/10.;
        double block=std::min(incomingLeft,double(std::max(0,int(facts[3]))))/8.;
        auto target=enemy(a%5);
        bool directed=attack && h[4];
        const double other=directed && a%5!=slot?1.:0.;
        std::array<double,15> x={damage,damage*(1.-hpFraction),block,block*missing,
            double(power),power*early,double(facts[4]),double(facts[5]),other,
            other*(1.-target[1]/std::max(1.f,target[2])),double(directed && a%5==slot)*hpFraction,
            double(facts[32]>0 || facts[33]>0 || id==C::DISARM),double(facts[1]!=0)*live,
            double(h[2]),double(doesCardExhaust(id,h[1]))};
        scores[a]+=std::inner_product(x.begin(),x.end(),q.begin(),0.);
    }
}
