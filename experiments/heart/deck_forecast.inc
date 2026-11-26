// Compare the inventory after each legal acquisition or removal. Public context
// terms stay constant across alternatives and cancel in the logit difference.
void deck_forecast_scores(const uint8_t *mask,double *scores) const {
    const auto screen=ScreenState(int(o[6]));
    const bool boss=screen==ScreenState::BOSS_RELIC_REWARDS;
    const bool remove=screen==ScreenState::CARD_SELECT && CardSelectScreenType(int(o[26]))==CardSelectScreenType::REMOVE;
    if(!boss && !remove && screen!=ScreenState::REWARDS) return;
    std::vector<std::pair<int,int>> changes;
    bool skip=false;
    for(int i=0;i<N_CHOICES;++i) if(mask[ACT_CHOOSE+i]) {
        auto opt=o+OBS_CHOICES+i*CHOICE_W;
        auto kind=ChoiceKind(int(opt[0]));
        if(kind==ChoiceKind::SKIP || kind==ChoiceKind::SINGING_BOWL) {skip=true;continue;}
        if(boss && kind==ChoiceKind::RELIC) changes.emplace_back(ACT_CHOOSE+i,CARD_COUNT+int(opt[1])-1);
        else if(!boss && kind==ChoiceKind::CARD) changes.emplace_back(ACT_CHOOSE+i,int(opt[1])-1);
        else return;
    }
    if(changes.empty() || (!remove && !skip)) return;
    const int act=std::clamp(int(o[4])-1+boss,0,2);
    ForecastHidden hidden=forecast_stage[act];
    auto add=[&](int token){
        for(size_t j=0;j<hidden.size();++j) hidden[j]+=forecast_items[token][j];
    };
    for(int i=0;i<MAX_DECK && o[OBS_DECK+i*CARD_W];++i) add(int(o[OBS_DECK+i*CARD_W])-1);
    for(int i=0;i<MAX_RELICS && o[OBS_RELICS+i*RELIC_W];++i) add(CARD_COUNT+int(o[OBS_RELICS+i*RELIC_W])-1);
    ForecastHidden before{};
    for(size_t j=0;j<hidden.size();++j) before[j]=std::tanh(hidden[j]);
    const float sign=remove?-1.f:1.f;
    for(auto [a,token]:changes) {
        auto delta=forecast_direct[act][token];
        for(auto &x:delta) x*=sign;
        for(size_t j=0;j<hidden.size();++j) {
            float dh=std::tanh(hidden[j]+sign*forecast_items[token][j])-before[j];
            for(size_t h=0;h<delta.size();++h) delta[h]+=dh*forecast_readout[act][j][h];
        }
        scores[a]+=std::inner_product(delta.begin(),delta.end(),forecast_scale.begin(),0.);
    }
}
