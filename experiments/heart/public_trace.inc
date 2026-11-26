// Convert an archived simulator action to one or more legal public-interface actions.
std::vector<int> recorded_actions(const Env &e, uint32_t bits) {
    if(e.inBattle) {
        search::Action action(bits);
        const int source=action.getSourceIdx(), target=action.getTargetIdx();
        switch(action.getActionType()) {
            case search::ActionType::CARD:
                if(source>=10 || target>=5) return {};
                return {ACT_PLAY+5*source+target};
            case search::ActionType::POTION:
                if(source>=5) return {};
                if(target==0x1fff) return {ACT_DISCARD+source};
                if(target>=5) return {};
                return {ACT_POTION+5*source+target};
            case search::ActionType::END_TURN: return {ACT_END_TURN};
            case search::ActionType::SINGLE_CARD_SELECT: return {ACT_CHOOSE+action.getSelectIdx()};
            case search::ActionType::MULTI_CARD_SELECT: {
                std::vector<int> picks;
                for(int i:action.getSelectedIdxs()) picks.push_back(ACT_CHOOSE+i);
                picks.push_back(ACT_CONFIRM);
                return picks;
            }
        }
        return {};
    }
    search::GameAction a(bits);
    if(a.isPotionAction()) {
        int slot=a.getIdx1();
        if(slot>=5) return {};
        return {a.isPotionDiscard()?ACT_DISCARD+slot:ACT_POTION+5*slot};
    }
    if(e.gc.screenState==ScreenState::EVENT_SCREEN && e.gc.curEvent==Event::MATCH_AND_KEEP)
        return {ACT_CHOOSE+a.getIdx1(), ACT_CHOOSE+a.getIdx2()};
    for(int i=0;i<std::min(N_CHOICES,e.choiceCount());++i)
        if(e.choiceAction(i).bits==bits) return {ACT_CHOOSE+i};
    return {};
}

auto convert_public_trace(uint64_t seed,const std::vector<std::array<uint32_t,2>> &trace) {
    Env e;
    e.record=false;
    e.reset(seed,20,true,CharacterClass::IRONCLAD);
    std::vector<int> actions;
    // Store half-open ranges so a divergent fight cannot leave partial training labels.
    std::vector<std::array<int,7>> fights;
    std::array<int,7> fight{};
    bool active=false;
    auto observe=[&](){
        bool combat=e.inBattle && !e.done();
        int hp=combat?e.bc.player.curHp:e.gc.curHp;
        if(active && (!combat || e.gc.act!=fight[0] || e.gc.floorNum!=fight[1] || int(e.bc.encounter)!=fight[2])) {
            fight[4]=hp; fight[6]=int(actions.size());
            fights.push_back(fight);
            active=false;
        }
        if(combat && !active) {
            fight={e.gc.act,e.gc.floorNum,int(e.bc.encounter),hp,0,int(actions.size()),0};
            active=true;
        }
    };
    observe();
    int i=0;
    bool took_boss_relic=false;
    std::string reason="end";
    for(const auto &row:trace) {
        // This fork leaves the boss chest open after taking a relic; the recorded fork closes it.
        if(took_boss_relic && !e.inBattle && e.gc.screenState==ScreenState::BOSS_RELIC_REWARDS
           && e.isValid(ACT_CHOOSE+3) && !e.isValid(ACT_CHOOSE) && !e.isValid(ACT_CHOOSE+1)
           && !e.isValid(ACT_CHOOSE+2)) {
            if(!e.step(ACT_CHOOSE+3)) {reason="boss_chest";break;}
            actions.push_back(ACT_CHOOSE+3);
            observe();
            took_boss_relic=false;
        }
        if(!e.inBattle && e.gc.screenState==ScreenState::MAP_SCREEN) took_boss_relic=false;
        if(e.done() || int(row[0])!=int(e.inBattle)) {reason="domain";break;}
        auto moves=recorded_actions(e,row[1]);
        if(moves.empty()) {reason="unmapped";break;}
        for(int a:moves) {
            if(!e.inBattle && e.gc.screenState==ScreenState::BOSS_RELIC_REWARDS
               && a>=ACT_CHOOSE && a<ACT_CHOOSE+3) took_boss_relic=true;
            if(a<0 || a>=N_ACTIONS || !e.step(a)) {reason="illegal";break;}
            actions.push_back(a); observe();
        }
        if(reason!="end") break;
        ++i;
    }
    return std::make_tuple(actions,fights,i,reason,int(e.gc.screenState),int(e.gc.floorNum),e.won());
}
