// The historical logs support card and boss-relic menus. Other decisions keep their native scores.
void historical_menu(const uint8_t *mask,double *scores) const {
    const auto screen=ScreenState(int(o[6]));
    bool boss=screen==ScreenState::BOSS_RELIC_REWARDS;
    if(!boss && screen!=ScreenState::REWARDS) return;
    const double scale=historical_scales[boss];
    if(!scale) return;
    std::vector<std::pair<int,int>> options;
    bool skip=false, item=false;
    for(int i=0;i<N_CHOICES;++i) if(mask[ACT_CHOOSE+i]) {
        auto opt=o+OBS_CHOICES+i*CHOICE_W;
        auto kind=ChoiceKind(int(opt[0]));
        int token;
        if(kind==ChoiceKind::SKIP) {token=0;skip=true;}
        else if(!boss && kind==ChoiceKind::SINGING_BOWL) token=1;
        else if(!boss && kind==ChoiceKind::CARD) {token=1+int(opt[1]);item=true;}
        else if(boss && kind==ChoiceKind::RELIC) {token=1+CARD_COUNT+int(opt[1]);item=true;}
        else return;
        options.emplace_back(i,token);
    }
    if(!skip || !item) return;
    std::array<uint8_t,HISTORICAL_ITEMS> counts{};
    for(int i=0;i<MAX_DECK && o[OBS_DECK+i*CARD_W];++i) {
        int id=int(o[OBS_DECK+i*CARD_W])-1;
        if(C(id)==C::STRIKE_RED || C(id)==C::DEFEND_RED || C(id)==C::BASH || C(id)==C::ASCENDERS_BANE) continue;
        auto &n=counts[2+id]; if(n<3) ++n;
    }
    for(int i=0;i<MAX_RELICS && o[OBS_RELICS+i*RELIC_W];++i) {
        int id=int(o[OBS_RELICS+i*RELIC_W])-1;
        if(R(id)!=R::BURNING_BLOOD) counts[2+CARD_COUNT+id]=1;
    }
    HistoricalHidden hidden{};
    int size=0;
    for(int t=2;t<HISTORICAL_ITEMS;++t) if(counts[t]) {
        size+=counts[t];
        for(size_t j=0;j<hidden.size();++j) hidden[j]+=counts[t]*historical_context[t][j];
    }
    float norm=std::sqrt(float(std::max(1,size)));
    for(float &h:hidden) h=std::tanh(h/norm);
    int act=std::clamp(int(o[4])-1,0,2);
    auto value=[&](int t,int upgrade) {
        return historical_bias[act][t]+historical_upgrade[t]*upgrade+historical_repeat[t]*std::log1p(counts[t])+
               std::inner_product(hidden.begin(),hidden.end(),historical_options[t].begin(),0.);
    };
    const double base=value(0,0);
    const double hp=o[0]/std::max(1.f,o[1])-.65;
    const double size_feature=std::clamp((o[22]-18.)/15.,-1.5,2.);
    const double log_gate=historical_gate[boss?3+std::min(act,1):act]+historical_gate[5+boss]*hp+
                          historical_gate[7+boss]*size_feature;
    const double weight=scale*std::exp(std::clamp(log_gate,-2.,2.));
    for(auto [i,token]:options) {
        bool upgraded=!boss && token>=2 && o[OBS_CHOICES+i*CHOICE_W+2]>0;
        scores[ACT_CHOOSE+i]+=weight*(value(token,upgraded)-base);
        if(!boss && token>=2) scores[ACT_CHOOSE+i]+=historical_gate[9+act];
    }
}
