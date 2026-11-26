double edit_score(Edit edit,int id,bool upgraded) const {
    const auto &q=edit_values[std::clamp(int(o[4])-1,0,2)][edit][id];
    return q[0]+q[1]*(edit!=GainPotion && deck[id]>=(edit==AddCard?1:2))
               +q[2]*upgraded+q[3]*missing;
}

void score_edits(const uint8_t *mask,double *scores) const {
    auto screen=ScreenState(int(o[6]));
    bool shop=screen==ScreenState::SHOP_ROOM;
    bool reward=screen==ScreenState::REWARDS;
    bool rest=screen==ScreenState::REST_ROOM;
    if(!shop && !reward && !rest && screen!=ScreenState::CARD_SELECT) return;
    double bestUpgrade=0,bestRemove=0;
    if(shop || rest) {
        for(int j=0;j<MAX_DECK;++j) {
            auto row=o+OBS_DECK+j*CARD_W;
            if(!row[0]) break;
            C id=C(int(row[0])-1);
            Card card(id,int(row[1]));
            if(card.canUpgrade()) bestUpgrade=std::max(bestUpgrade,edit_score(UpgradeCard,int(id),row[1]));
            if(card.canTransform()) bestRemove=std::max(bestRemove,edit_score(RemoveCard,int(id),row[1]));
        }
    }
    for(int j=0;j<N_CHOICES;++j) if(mask[ACT_CHOOSE+j]) {
        auto row=o+OBS_CHOICES+j*CHOICE_W;
        const auto kind=ChoiceKind(int(row[0]));
        double &score=scores[ACT_CHOOSE+j];
        int id=int(row[1])-1;
        if((shop || reward) && kind==ChoiceKind::CARD) score+=edit_score(AddCard,id,row[2]);
        if((shop || reward) && kind==ChoiceKind::POTION) score+=edit_score(GainPotion,id,false);
        if(shop && kind==ChoiceKind::CARD_REMOVE) score+=bestRemove;
        if(rest && j==1) score+=bestUpgrade;
        if(screen!=ScreenState::CARD_SELECT || kind!=ChoiceKind::CARD) continue;
        switch(CardSelectScreenType(int(o[26]))) {
            case CardSelectScreenType::UPGRADE: score+=edit_score(UpgradeCard,id,row[2]); break;
            case CardSelectScreenType::REMOVE: score+=edit_score(RemoveCard,id,row[2]); break;
            case CardSelectScreenType::DUPLICATE: score+=edit_score(AddCard,id,row[2]); break;
            default: break;
        }
    }
}
