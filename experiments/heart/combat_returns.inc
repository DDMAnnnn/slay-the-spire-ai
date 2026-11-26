// Matched action-value targets: take each distinct legal action and finish the
// fight with the same continuation policy. Only the labels access future states.
struct FightReturn {
    bool win;
    float health, remaining, potions;
    bool capped;
};

FightReturn finish_combat_action(Env game, const Weights &weights, int first,int tactic=-1) {
    std::array<float, OBS_SIZE> obs;
    std::array<uint8_t, N_ACTIONS> mask;
    std::array<double, N_ACTIONS> scores;
    const int initial_potions=game.bc.potionCount;
    const auto encounter=game.bc.encounter;
    const int floor=game.gc.floorNum;
    int action=first;
    for (int k=0; k<400 && game.inBattle && !game.done(); ++k) {
        if (k) {
            game.observe(obs.data());
            View view(obs.data(),weights);
            game.mask(mask.data());
            action=continuation_action(view,mask.data(),scores,tactic);
        }
        if (!game.step(action)) throw std::runtime_error("invalid combat continuation");
        if (game.inBattle && (encounter!=game.bc.encounter || floor!=game.gc.floorNum))
            return {true,float(game.gc.curHp),0.f,float(initial_potions-game.gc.potionCount),false};
    }
    bool win=game.bc.outcome==Outcome::PLAYER_VICTORY;
    float hp=0, max_hp=0;
    for (int i=0;i<game.bc.monsters.monsterCount;++i) {
        const auto &m=game.bc.monsters.arr[i];
        hp+=std::max(0,m.curHp);max_hp+=m.maxHp;
    }
    float remaining=hp/std::max(1.f,max_hp);
    return {win,win?float(game.bc.player.curHp):0.f,win?0.f:remaining,
            float(initial_potions-game.bc.potionCount),game.inBattle && !game.done()};
}

using CombatReturnRow=std::array<float,11+CURRICULUM_INPUT>;
CombatReturnRow combat_return_row(const View &view,const FightReturn &r,int i,int k,int a,double gap) {
    auto c=view.candidate(a,std::clamp(gap/CURRICULUM_TEMP,-10.,0.));
    CombatReturnRow row{};
    row[0]=i;row[1]=k;row[2]=a;row[3]=c.item;row[4]=c.foe;
    row[5]=r.win;row[6]=r.health;row[7]=r.remaining;row[8]=r.potions;
    row[9]=view.o[0];row[10]=view.o[1];
    std::copy(c.features.begin(),c.features.end(),row.begin()+11);
    return row;
}

py::array_t<float> pack_combat_examples(const std::vector<std::vector<CombatReturnRow>> &episodes) {
    size_t length=0;
    for(const auto &e:episodes) length+=e.size();
    py::array_t<float> rows({py::ssize_t(length),py::ssize_t(CombatReturnRow{}.size())});
    float *ptr=rows.mutable_data();
    for(const auto &e:episodes) for(const auto &r:e) ptr=std::copy(r.begin(),r.end(),ptr);
    return rows;
}

auto combat_returns(const Weights &w,uint64_t start,int n,int threads) {
    if(n<1 || threads<1) throw std::invalid_argument("positive collection dimensions required");
    std::vector<std::vector<CombatReturnRow>> episodes(n);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
    for(int i=0;i<n;++i) {
        Env game;
        game.record=false;
        game.reset(start+i,20,true,CharacterClass::IRONCLAD);
        std::mt19937 rng(static_cast<unsigned>(start+i));
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        for(int k=0;k<3500 && !game.done();++k) {
            game.observe(obs.data());game.mask(mask.data());
            View view(obs.data(),w);view.all_scores(mask.data(),scores.data());
            int parent=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            if(game.inBattle && mask[ACT_END_TURN] && (parent<ACT_DISCARD || parent==ACT_END_TURN) &&
               rng()%(obs[4]>=3?2:obs[4]==2?5:10)==0) {
                auto actions=distinct_combat_actions(parent,obs.data(),mask.data());
                if(actions.size()>1) {
                    size_t before=episodes[i].size();
                    for(int a:actions) {
                        auto result=finish_combat_action(game,w,a);
                        if(result.capped) {episodes[i].resize(before);break;}
                        episodes[i].push_back(combat_return_row(view,result,i,k,a,scores[a]-scores[parent]));
                    }
                }
            }
            if(!game.step(parent)) throw std::runtime_error("invalid policy action during combat collection");
        }
    }
    return pack_combat_examples(episodes);
}
