int visible_draft_boss() const {
    if(o[7] || Room(int(o[8])-1)==Room::BOSS || int(o[4])>3) return -1;
    static constexpr MonsterEncounter bosses[]={
        MonsterEncounter::THE_GUARDIAN, MonsterEncounter::HEXAGHOST, MonsterEncounter::SLIME_BOSS,
        MonsterEncounter::AUTOMATON, MonsterEncounter::CHAMP, MonsterEncounter::COLLECTOR,
        MonsterEncounter::AWAKENED_ONE, MonsterEncounter::TIME_EATER, MonsterEncounter::DONU_AND_DECA};
    auto it=std::find(std::begin(bosses),std::end(bosses),MonsterEncounter(int(o[15])-1));
    return it==std::end(bosses)?-1:int(it-std::begin(bosses));
}

BossDraftInput boss_draft_input(const DraftFeatures &context,int boss,const float *card,double prior) const {
    BossDraftInput x{};
    auto offered=offered_features(context,card);
    std::copy(offered.begin(),offered.end(),x.begin());
    x[17+boss]=1;
    x[26]=float(std::clamp(prior,-40.,40.)/12.);
    x[27]=o[12]; x[28]=o[13]; x[29]=o[14];
    for(int j=0;j<5;++j) x[30]+=o[17+j]>0;
    x[31]=o[1]/80;
    for(int j=0;j<CARD_COUNT;++j) x[32+j]=std::min(3,deck[j]);
    return x;
}

void apply_boss_draft(const uint8_t *mask,double *out) const {
    int skip=draft_skip(mask),boss=visible_draft_boss();
    if(skip<0 || boss<0) return;
    const auto context=deck_features();
    for(int j=0;j<N_CHOICES;++j) if(mask[ACT_CHOOSE+j] && ACT_CHOOSE+j!=skip) {
        auto card=o+OBS_CHOICES+j*CHOICE_W;
        const auto x=boss_draft_input(context,boss,card,out[ACT_CHOOSE+j]-out[skip]);
        const auto &q=boss_draft_weights[int(card[1])-1];
        out[ACT_CHOOSE+j]+=std::inner_product(q.begin(),q.end(),x.begin(),0.);
    }
}
