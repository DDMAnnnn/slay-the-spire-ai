// Compare the deck and resources after each displayed campfire, shop or card-edit option.
void resource_menu(const uint8_t *mask,double *scores) const {
    auto screen=ScreenState(int(o[6]));
    int kind=-1;
    if(screen==ScreenState::REST_ROOM) kind=0;
    if(screen==ScreenState::SHOP_ROOM) kind=1;
    if(screen==ScreenState::CARD_SELECT) {
        auto task=CardSelectScreenType(int(o[26]));
        if(task==CardSelectScreenType::UPGRADE) kind=2;
        if(task==CardSelectScreenType::REMOVE) kind=3;
    }
    if(kind<0 || !resource_mix[kind]) return;
    std::array<int,RESOURCE_TOKENS> counts{};
    int upgrades=0;
    for(int i=0;i<MAX_DECK && o[OBS_DECK+i*CARD_W];++i) {
        auto row=o+OBS_DECK+i*CARD_W;
        ++counts[8+int(row[0])];upgrades+=row[1]>0;
    }
    for(int i=0;i<MAX_RELICS && o[OBS_RELICS+i*RELIC_W];++i)
        ++counts[8+CARD_COUNT+int(o[OBS_RELICS+i*RELIC_W])];
    ResourceHidden hidden{};
    int items=0;
    for(int id=0;id<RESOURCE_TOKENS;++id) if(counts[id]) {
        items+=counts[id];
        for(int j=0;j<int(hidden.size());++j) hidden[j]+=counts[id]*resource_context[id][j];
    }
    const float norm=std::sqrt(float(std::max(1,items)));
    for(auto &x:hidden) x=std::tanh(x/norm);
    const float miss=1-o[0]/std::max(1.f,o[1]);
    const int local=std::max(0,int(o[3])-1)%17;
    int filled=0;
    for(int i=0;i<5;++i) filled+=o[17+i]>0;
    ResourceInput x={1.f,miss,miss*miss,(o[1]-75)/50,std::min(o[2]/200,3.f),float(o[4]==2),float(o[4]>=3),
        local/15.f,float(local>=13),o[12],o[13],o[14],std::min(o[22]/25,3.f),upgrades/std::max(1.f,o[22]),
        num(C::STRIKE_RED)/5.f,num(C::DEFEND_RED)/4.f,filled/std::max(1.f,o[16]),miss*float(local>=13),
        float(o[4]>=3 && !o[12]),0,0,0,0};
    std::vector<std::pair<int,double>> values;
    double original=-1e30;
    for(int i=0;i<N_CHOICES;++i) if(mask[ACT_CHOOSE+i]) {
        auto opt=o+OBS_CHOICES+i*CHOICE_W;
        int token,category;
        switch(ChoiceKind(int(opt[0]))) {
            case ChoiceKind::REST_OPTION: token=2+int(opt[1]);category=2;break;
            case ChoiceKind::CARD: token=8+int(opt[1]);category=3;break;
            case ChoiceKind::RELIC: token=8+CARD_COUNT+int(opt[1]);category=4;break;
            case ChoiceKind::POTION: token=8+CARD_COUNT+int(std::size(relicNames))+int(opt[1]);category=5;break;
            case ChoiceKind::CARD_REMOVE: token=1;category=1;break;
            case ChoiceKind::SKIP: token=0;category=0;break;
            default: return;
        }
        x[19]=opt[3]/150; x[20]=opt[3]/std::max(1.f,o[2]); x[21]=opt[2]>0; x[22]=std::min(3,counts[token])/3.f;
        double z=std::inner_product(x.begin(),x.end(),resource_category[kind][category].begin(),0.)+
                 std::inner_product(x.begin(),x.end(),resource_item[kind][token].begin(),0.)+
                 std::inner_product(hidden.begin(),hidden.end(),resource_option[kind][token].begin(),0.);
        values.emplace_back(ACT_CHOOSE+i,z);
        original=std::max(original,scores[ACT_CHOOSE+i]);
    }
    if(values.size()<2) return;
    const double t=resource_mix[kind];
    double highest=-1e30;
    for(auto &[a,z]:values) {
        z=(1-t)*(scores[a]-original)/8+t*z;highest=std::max(highest,z);
    }
    // Preserve the native scale for any legal non-menu action such as drinking a potion.
    for(auto [a,z]:values) scores[a]=original+8*(z-highest);
}
