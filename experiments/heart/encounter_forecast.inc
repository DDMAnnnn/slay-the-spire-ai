// A reward is evaluated against the same training encounter distribution before and after pickup.
void score_predicted_fights(const uint8_t *mask,double *scores) const {
    int act=int(o[4]);
    if(act<1 || act>3) return;
    auto screen=ScreenState(int(o[6]));
    bool boss=screen==ScreenState::BOSS_RELIC_REWARDS;
    bool smith=screen==ScreenState::CARD_SELECT && int(o[26])==int(CardSelectScreenType::UPGRADE);
    bool remove=screen==ScreenState::CARD_SELECT && int(o[26])==int(CardSelectScreenType::REMOVE);
    if(!boss && !smith && !remove && draft_skip(mask)<0) return;
    constexpr int relic_base=1+2*CARD_COUNT;
    constexpr int potion_base=relic_base+int(std::size(relicNames));
    EncounterHidden bag{};int n=0,potions=0;
    auto add=[&](EncounterHidden &sum,int id,double direction) {
        for(size_t j=0;j<sum.size();++j) sum[j]+=direction*enc_item.at(id)[j];
    };
    for(int i=0;i<MAX_DECK && o[OBS_DECK+i*CARD_W];++i) {
        auto c=o+OBS_DECK+i*CARD_W;add(bag,1+2*(int(c[0])-1)+int(c[1]>0),1.);++n;
    }
    for(int i=0;i<MAX_RELICS && o[OBS_RELICS+i*RELIC_W];++i) {
        add(bag,relic_base+int(o[OBS_RELICS+i*RELIC_W])-1,1.);++n;
    }
    for(int i=0;i<5;++i) if(o[17+i]) {add(bag,potion_base+int(o[17+i])-1,1.);++n;++potions;}
    const double hp=o[0]/std::max(1.f,o[1]);
    const int seen_boss=int(o[15])-1;
    auto value=[&](const EncounterHidden &inventory,int count,int dc,int dr) {
        std::array<double,11> x{1.,hp,hp*hp,(o[1]-75.)/40.,(o[22]+dc)/25.,(o[23]+dr)/12.,
            potions/2.,std::clamp((o[3]-1.-17.*(act-1))/15.,0.,1.),double(act==2),double(act==3),0.};
        EncounterHidden base{};
        const double norm=std::sqrt(std::max(1,count));
        for(size_t j=0;j<base.size();++j)
            base[j]=inventory[j]/norm+std::inner_product(x.begin(),x.end(),enc_context[j].begin(),0.);
        auto outcome=[&](int enemy) {
            auto z=enc_out_bias;
            for(size_t j=0;j<base.size();++j) {
                double h=std::tanh(base[j]+enc_enemy.at(enemy)[j]);
                for(int k=0;k<2;++k) z[k]+=h*enc_readout[k][j];
            }
            return (1./(1.+std::exp(-z[0])))*(.55+.45/(1.+std::exp(-z[1])));
        };
        double mean=0.;
        for(auto [enemy,mass]:enc_banks[act-1]) mean+=mass*outcome(enemy);
        if(seen_boss>=0 && seen_boss<int(enc_enemy.size())) mean+=enc_boss_weight*(outcome(seen_boss)-mean);
        return mean;
    };
    const double base=value(bag,n,0,0);
    for(int i=0;i<N_CHOICES;++i) if(mask[ACT_CHOOSE+i]) {
        const float *c=o+OBS_CHOICES+i*CHOICE_W;
        auto edited=bag;int dc=0,dr=0;
        if(ChoiceKind(int(c[0]))==ChoiceKind::CARD) {
            int t=1+2*(int(c[1])-1)+int(c[2]>0);
            if(smith) {
                if(c[2]) continue;
                add(edited,t,-1.);add(edited,t+1,1.);
            } else {
                dc=remove?-1:1;add(edited,t,dc);
            }
        } else if(boss && ChoiceKind(int(c[0]))==ChoiceKind::RELIC) {
            dr=1;add(edited,relic_base+int(c[1])-1,1.);
        } else continue;
        scores[ACT_CHOOSE+i]+=enc_gain*(value(edited,n+dc+dr,dc,dr)-base);
    }
}
