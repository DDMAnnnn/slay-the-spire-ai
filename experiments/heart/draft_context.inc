int draft_skip(const uint8_t *mask) const {
    if(o[7] || ScreenState(int(o[6]))!=ScreenState::REWARDS) return -1;
    int skip=-1, cards=0;
    for(int i=0;i<N_CHOICES;++i) if(mask[ACT_CHOOSE+i]) {
        auto kind=ChoiceKind(int(o[OBS_CHOICES+i*CHOICE_W]));
        if(kind==ChoiceKind::SKIP) skip=ACT_CHOOSE+i;
        else if(kind==ChoiceKind::CARD) ++cards;
        else return -1;
    }
    return cards?skip:-1;
}
DraftFeatures deck_features() const {
    DraftFeatures x{1.,o[4]/3.,o[3]/56.,missing,o[22]/25.,o[2]/200.,o[23]/10.};
    const double size=std::max(1.f,o[22]);
    for(int j=0;j<MAX_DECK;++j) {
        const auto card=o+OBS_DECK+j*CARD_W;
        if(!card[0]) break;
        C id=C(int(card[0])-1);
        const auto f=cardFacts[card[1]>0][int(id)];
        x[7]+=double(getCardType(id)==CardType::ATTACK)/size;
        x[8]+=double(getCardType(id)==CardType::POWER)/size;
        x[9]+=std::max(0,getBaseDamage(id,card[1]>0))*std::max(1,int(f[0]))/(12.*size);
        x[10]+=std::max(0,int(f[3]))/(6.*size);
        x[11]+=std::max(0,int(f[4]))/size;
        x[12]+=double(f[1]>0)/size;
        x[13]+=double(card[1]>0)/size;
    }
    return x;
}
DraftFeatures offered_features(DraftFeatures x,const float *c) const {
    C id=C(int(c[1])-1);
    x[14]=num(id);
    x[15]=c[2];
    x[16]=draft(id,c[2])/15.; // Previous draft score is one input with a fitted coefficient.
    return x;
}
