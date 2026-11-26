int inventory_card(int id,bool up) const {return id+(up?CARD_COUNT:0);}
int inventory_relic(int id) const {return 2*CARD_COUNT+id;}
double inventory_value(const InventoryHidden &state,int remove,int add) const {
    std::array<double,4> logits=inventory_intercept;
    for(int j=0;j<INVENTORY_WIDTH;++j) {
        float x=state[j];
        if(remove>=0) x-=inventory_tokens[remove][j];
        if(add>=0) x+=inventory_tokens[add][j];
        x=std::tanh(x);
        for(int k=0;k<4;++k) logits[k]+=x*inventory_heads[k][j];
    }
    const int act=std::clamp(int(o[4])-1,0,3);
    double total=0.,sum=0.;
    for(int k=act;k<4;++k) {
        double weight=k==act?1.:inventory_settings[3]/(k-act);
        total+=weight/(1.+std::exp(-std::clamp(logits[k],-30.,30.)));sum+=weight;
    }
    return total/sum;
}

void inventory_scores(const uint8_t *mask,double *scores) const {
    auto screen=ScreenState(int(o[6]));
    bool shop=screen==ScreenState::SHOP_ROOM,reward=screen==ScreenState::REWARDS;
    bool boss=screen==ScreenState::BOSS_RELIC_REWARDS,select=screen==ScreenState::CARD_SELECT;
    bool camp=screen==ScreenState::REST_ROOM;
    if(!shop && !reward && !boss && !select && !camp) return;
    InventoryHidden state=inventory_bias;
    auto include=[&](int id) {
        for(int j=0;j<INVENTORY_WIDTH;++j) state[j]+=inventory_tokens[id][j];
    };
    for(int i=0;i<MAX_DECK && o[OBS_DECK+i*CARD_W];++i) {
        auto c=o+OBS_DECK+i*CARD_W;include(inventory_card(int(c[0])-1,c[1]!=0));
    }
    for(int i=0;i<MAX_RELICS && o[OBS_RELICS+i*RELIC_W];++i)
        include(inventory_relic(int(o[OBS_RELICS+i*RELIC_W])-1));
    double base=inventory_value(state,-1,-1);
    auto diff=[&](int rm,int add){return inventory_value(state,rm,add)-base;};
    double upgrade=0,remove=0;
    if((shop || camp) && inventory_settings[1]) {
        for(int j=0;j<MAX_DECK && o[OBS_DECK+j*CARD_W];++j) {
            auto c=o+OBS_DECK+j*CARD_W;int id=int(c[0])-1;
            int token=inventory_card(id,c[1]!=0);
            Card card{C(id),int(c[1])};
            if(shop && card.canTransform()) remove=std::max(remove,diff(token,-1));
            if(camp && card.canUpgrade()) upgrade=std::max(upgrade,diff(token,inventory_card(id,true)));
        }
    }
    for(int j=0;j<N_CHOICES;++j) if(mask[ACT_CHOOSE+j]) {
        auto opt=o+OBS_CHOICES+j*CHOICE_W;auto kind=ChoiceKind(int(opt[0]));
        int id=int(opt[1])-1;
        double &q=scores[ACT_CHOOSE+j];
        if((reward || shop || boss) && kind==ChoiceKind::RELIC)
            q+=inventory_settings[2]*diff(-1,inventory_relic(id));
        if((reward || shop) && kind==ChoiceKind::CARD)
            q+=inventory_settings[0]*diff(-1,inventory_card(id,opt[2]!=0));
        if(shop && kind==ChoiceKind::CARD_REMOVE) q+=inventory_settings[1]*remove;
        if(camp && j==1) q+=inventory_settings[1]*upgrade;
        if(select && kind==ChoiceKind::CARD) {
            int token=inventory_card(id,opt[2]!=0);
            auto task=CardSelectScreenType(int(o[26]));
            if(task==CardSelectScreenType::UPGRADE) q+=inventory_settings[1]*diff(token,inventory_card(id,true));
            if(task==CardSelectScreenType::REMOVE) q+=inventory_settings[1]*diff(token,-1);
            if(task==CardSelectScreenType::DUPLICATE) q+=inventory_settings[0]*diff(-1,token);
        }
    }
}
