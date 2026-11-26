struct ExpectedHand {
    double damage=0, hits=0, block=0;
};
ExpectedHand expected_hand() const {
    // Pool contents are public. If the draw pile runs short, the remainder
    // comes from the discard pile and the cards leaving the hand this turn.
    struct Pool { double n=0,cost=0,damage=0,hits=0,block=0; };
    auto add=[](Pool &pool,const float *row) {
        C id=C(int(row[0])-1);
        bool up=row[1]!=0;
        const auto facts=cardFacts[up][int(id)];
        double hits=std::max(1,int(facts[0]));
        pool.n++;
        int cost=getEnergyCost(id,up);
        if(cost < -1) return;
        pool.cost+=cost==-1?2:cost;
        if(getCardType(id)==CardType::ATTACK) {
            pool.damage+=std::max(0,getBaseDamage(id,up))*hits;
            pool.hits+=hits;
        }
        pool.block+=std::max(0,int(facts[3]));
    };
    Pool draw,other;
    for(int j=0;j<MAX_PILE && o[OBS_DRAW+j*CARD_W];++j) add(draw,o+OBS_DRAW+j*CARD_W);
    for(int j=0;j<MAX_PILE && o[OBS_DISCARD+j*CARD_W];++j) add(other,o+OBS_DISCARD+j*CARD_W);
    for(int j=0;j<int(p[5]);++j) add(other,o+OBS_HAND+j*HAND_W);
    const double d=std::min(draw.n,5.)/std::max(draw.n,1.);
    const double r=std::min(other.n,std::max(0.,5.-draw.n))/std::max(other.n,1.);
    const double playable=std::min(1.,3./std::max(1.,draw.cost*d+other.cost*r));
    return {playable*(draw.damage*d+other.damage*r),playable*(draw.hits*d+other.hits*r),
            playable*(draw.block*d+other.block*r)};
}

double future_value(const Position &s,const ExpectedHand &hand) const {
    auto hp=s.hp;
    double health=o[0]+s.hpChange;
    const double perTurnBlock=p[N_PLAYER_BASE+int(PS::METALLICIZE)]+s.passiveBlock;
    const double growth=p[N_PLAYER_BASE+int(PS::DEMON_FORM)]+s.strengthGrowth;
    double value=forecast_settings[1]*s.hpChange, discount=1.;
    for(int turn=0;turn<=3;++turn) {
        if(turn) {
            // Use one expected playable hand. Allocate its attack to the
            // lowest-HP survivor, with overflow usable against a second target.
            const double strength=s.strength-s.temporaryStrength-p[N_PLAYER_BASE+int(PS::LOSE_STRENGTH)]+turn*growth;
            double damage=std::max(0.,hand.damage+hand.hits*strength);
            for(int j=0;j<5 && damage>0;++j) {
                int target=-1;
                for(int k=0;k<5;++k) if(hp[k]>0 && (target<0 || hp[k]<hp[target])) target=k;
                if(target<0) break;
                double multiplier=s.vulnerable[target]>turn?1.5:1.;
                double hit=std::min(hp[target],damage*multiplier);
                hp[target]-=hit;
                damage-=hit/multiplier;
            }
        }
        double incomingDamage=0;
        for(int k=0;k<5;++k) if(hp[k]>0) {
            auto m=enemy(k);
            double hit=turn?intent_forecast[int(m[0])][int(m[7])][turn-1]:m[8]*m[9];
            hit=std::max(0.,hit-s.disarm[k]*(turn?1.:m[9]));
            if(s.weak[k]>turn && !(m[MONSTER_BASE+int(MS::WEAK)]>turn)) hit*=.75;
            incomingDamage+=hit;
        }
        double loss=std::max(0.,incomingDamage-perTurnBlock-(turn?hand.block:s.block));
        value-=discount*forecast_settings[1]*loss;
        health-=loss;
        if(health<=0) { value-=discount*35.; break; }
        discount*=forecast_settings[3];
    }
    return value-forecast_settings[2]*std::accumulate(hp.begin(),hp.end(),0.);
}

void forecast_scores(const uint8_t *mask,double *scores,const Position &start) const {
    const ExpectedHand hand=expected_hand();
    const double end=future_value(start,hand);
    // Compare each first move and its best visible second play to ending now.
    // Re-plan after each action so that newly revealed cards can participate.
    for(int a=0;a<ACT_POTION;++a) if(mask[a]) {
        Position first=start;
        resolve(first,a);
        double value=future_value(first,hand);
        if(first.energy>=0) for(int b=0;b<ACT_POTION;++b) if(mask[b] && b/5!=a/5) {
            const auto row=o+OBS_HAND+b/5*HAND_W;
            if(row[2]>first.energy || (row[4] && first.hp[b%5]<=0) || (first.used & (1<<(b/5)))) continue;
            Position second=first;
            resolve(second,b);
            value=std::max(value,future_value(second,hand));
        }
        scores[a]+=forecast_settings[0]*(value-end);
    }
}
