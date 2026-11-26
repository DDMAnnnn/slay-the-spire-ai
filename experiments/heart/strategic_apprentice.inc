// The same item embeddings follow a card from a reward to a shop or upgrade.
// Each decision family learns how to use those shared features and its parent policy score.
void strategic_menu(const uint8_t *mask,double *scores) const {
    if(!(strat_acts & (1<<(std::clamp(int(o[4]),1,4)-1))))return;
    const int screen=int(o[6]),event=int(o[9])-1,edit=int(o[26]);
    int task=-1;
    switch(ScreenState(screen)) {
        case ScreenState::EVENT_SCREEN: task=event==int(Event::NEOW)?0:1;break;
        case ScreenState::REWARDS:task=2;break;
        case ScreenState::BOSS_RELIC_REWARDS:task=3;break;
        case ScreenState::SHOP_ROOM:task=4;break;
        case ScreenState::REST_ROOM:task=5;break;
        case ScreenState::CARD_SELECT:task=edit==4?6:edit==3?7:8;break;
        case ScreenState::TREASURE_ROOM:task=9;break;
        default:return;
    }
    const double mix=strat_mix[task];
    if(!mix || (screen==1 && event==int(Event::MATCH_AND_KEEP))) return;
    std::array<int,STRAT_BASE> counts{};
    StratHidden hidden{};int n=0;
    auto inventory=[&](int token){
        ++counts[token];++n;for(int k=0;k<16;++k) hidden[k]+=strat_context[token][k];
    };
    for(int i=0;i<MAX_DECK && o[OBS_DECK+i*CARD_W];++i) inventory(15+int(o[OBS_DECK+i*CARD_W]));
    for(int i=0;i<MAX_RELICS && o[OBS_RELICS+i*RELIC_W];++i) inventory(15+CARD_COUNT+int(o[OBS_RELICS+i*RELIC_W]));
    for(int i=0;i<5;++i) if(o[17+i]) inventory(15+CARD_COUNT+int(std::size(relicNames))+int(o[17+i]));
    const double norm=std::sqrt(double(std::max(n,1)));
    for(int k=0;k<16;++k) hidden[k]=std::tanh(hidden[k]/norm+strat_task_hidden[task][k]);
    std::vector<std::pair<int,double>> options;
    double parent=-1e30;
    for(int i=0;i<N_CHOICES;++i) if(mask[ACT_CHOOSE+i]) parent=std::max(parent,scores[ACT_CHOOSE+i]);
    double miss=1.-o[0]/std::max(1.f,o[1]);
    StratInput x={1.,miss,miss*miss,(o[1]-75)/50.,std::min(3.,o[2]/200.),double(o[4]==2),double(o[4]>=3),
                 std::max(0,int(o[3])-1)%17/15.,o[12],o[13],o[14],std::min(o[22]/25.,3.),0.,0.,0.,0.,0.};
    for(int i=0;i<N_CHOICES;++i) if(mask[ACT_CHOOSE+i]) {
        auto c=o+OBS_CHOICES+i*CHOICE_W;
        int typ=int(c[0]),item=int(c[1]),up=int(c[2]),token;
        switch(ChoiceKind(typ)) {
            case ChoiceKind::CARD:token=15+item;break;
            case ChoiceKind::RELIC:token=15+CARD_COUNT+item;break;
            case ChoiceKind::POTION:token=15+CARD_COUNT+int(std::size(relicNames))+item;break;
            case ChoiceKind::GOLD:token=1;break;
            case ChoiceKind::KEY:token=item==1?2:3;break;
            case ChoiceKind::SKIP:token=0;break;
            case ChoiceKind::EVENT_OPTION:token=STRAT_BASE+8+16*event+i;break;
            case ChoiceKind::NEOW_OPTION:token=STRAT_NEOW+8*(item-1)+up-1;break;
            case ChoiceKind::REST_OPTION:token=STRAT_BASE+item;break;
            case ChoiceKind::CARD_REMOVE:token=6;break;
            case ChoiceKind::OPEN_CHEST:token=5;break;
            case ChoiceKind::SINGING_BOWL:token=4;break;
            default:return;
        }
        if(token<0 || token>=STRAT_TOKENS)return;
        double price=screen==8?int(c[3]):0.;
        x[12]=typ==int(ChoiceKind::CARD) && up>0;x[13]=price/150.;x[14]=price/std::max(1.f,o[2]);
        x[15]=token<STRAT_BASE?std::min(3,counts[token])/3.:0.;
        x[16]=std::clamp((scores[ACT_CHOOSE+i]-parent)/12.,-8.,0.);
        double z=strat_task_item[task][token]+std::inner_product(hidden.begin(),hidden.end(),strat_action[token].begin(),0.);
        for(int k=0;k<17;++k) z+=x[k]*(strat_item[token][k]+strat_category[task][typ][k]);
        options.emplace_back(ACT_CHOOSE+i,z);
    }
    if(options.size()<2)return;
    double best=-1e30;for(auto &v:options)best=std::max(best,v.second);
    for(auto [a,z]:options)scores[a]=parent+(1.-mix)*(scores[a]-parent)+8.*mix*(z-best);
}
