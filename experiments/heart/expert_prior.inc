void apply_expert_prior(const uint8_t *mask,double *scores) const {
    auto screen=ScreenState(int(o[6]));
    bool drafting=screen==ScreenState::REWARDS && draft_skip(mask)>=0;
    bool boss=screen==ScreenState::BOSS_RELIC_REWARDS;
    bool neow=screen==ScreenState::EVENT_SCREEN && int(o[9])-1==int(Event::NEOW);
    if(!drafting && !boss && !neow) return;
    int act=std::clamp(int(o[4])-1,0,2);
    for(int j=0;j<N_CHOICES;++j) if(mask[ACT_CHOOSE+j]) {
        const float *row=o+OBS_CHOICES+j*CHOICE_W;
        auto kind=ChoiceKind(int(row[0]));
        int id=int(row[1])-1;
        double &score=scores[ACT_CHOOSE+j];
        if(drafting && kind==ChoiceKind::CARD) score+=expert_cards[act][id][row[2]>0];
        if(boss && kind==ChoiceKind::RELIC) score+=expert_relics[std::min(1,act)][id];
        if(neow && kind==ChoiceKind::NEOW_OPTION) score+=expert_neow[id][int(row[2])-1];
    }
}
