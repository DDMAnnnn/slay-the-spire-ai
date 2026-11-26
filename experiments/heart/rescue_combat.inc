// Locate the actual fight that ended a natural run, then let the teacher change
// several plays in that fight. The deploying actor never receives this snapshot.
auto rescue_combat(const Weights &w,uint64_t start,int n,int threads) {
    if(n<1 || threads<1) throw std::invalid_argument("positive rescue dimensions required");
    std::vector<std::vector<CombatReturnRow>> episodes(n);
    std::vector<std::array<int,10>> stats(n);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
    for(int i=0;i<n;++i) {
        Env game,entry;
        game.record=false;
        game.reset(start+i,20,true,CharacterClass::IRONCLAD);
        bool tracked=false;
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        for(int k=0;k<3500 && !game.done();++k) {
            if(game.inBattle && (!tracked || entry.gc.floorNum!=game.gc.floorNum || entry.bc.encounter!=game.bc.encounter)) {
                entry=game;tracked=true;
            } else if(!game.inBattle) tracked=false;
            game.observe(obs.data());game.mask(mask.data());
            View view(obs.data(),w);view.all_scores(mask.data(),scores.data());
            if(!game.step(int(std::max_element(scores.begin(),scores.end())-scores.begin())))
                throw std::runtime_error("invalid fatal-fight scout action");
        }
        if(!game.done() || !tracked || game.bc.outcome!=Outcome::PLAYER_LOSS) continue;
        auto &s=stats[i];
        s[8]=entry.gc.curHp;
        s[0]=1;s[1]=entry.gc.act;s[2]=entry.gc.floorNum;s[3]=int(entry.bc.encounter);
        for(int k=0;k<400 && entry.inBattle && !entry.done();++k) {
            entry.observe(obs.data());entry.mask(mask.data());
            View view(obs.data(),w);view.all_scores(mask.data(),scores.data());
            int parent=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            int chosen=parent;
            if(mask[ACT_END_TURN] && (parent<ACT_DISCARD || parent==ACT_END_TURN)) {
                auto actions=distinct_combat_actions(parent,obs.data(),mask.data());
                if(actions.size()>1) {
                    double best=-1e20;
                    bool guaranteed=false;
                    size_t initial=episodes[i].size();
                    for(int a:actions) {
                        auto r=finish_combat_action(entry,w,a);
                        if(r.capped) {episodes[i].resize(initial);chosen=parent;guaranteed=false;break;}
                        episodes[i].push_back(combat_return_row(view,r,i,k,a,scores[a]-scores[parent]));
                        // Winning takes precedence. Enemy health supplies a gradient when every branch loses.
                        double value=r.win?1000.+r.health-.25*r.potions:-100.*r.remaining-.01*r.potions;
                        if(value>best) {best=value;chosen=a;guaranteed=r.win;}
                    }
                    s[4]+=int(episodes[i].size()-initial);s[5]+=chosen!=parent;
                    if(guaranteed) {
                        // Replay the selected exact continuation from the same position.
                        auto result=finish_combat_action(entry,w,chosen);
                        if(!result.win) throw std::runtime_error("combat rescue replay diverged");
                        s[6]=1;s[7]=int(result.health);break;
                    }
                }
            }
            if(!entry.step(chosen)) throw std::runtime_error("invalid rescue action");
        }
        if(!s[6] && entry.bc.outcome==Outcome::PLAYER_VICTORY) {s[6]=1;s[7]=entry.bc.player.curHp;}
        s[9]=int(entry.inBattle && !entry.done() && !s[6]);
    }
    return std::make_pair(pack_combat_examples(episodes),std::move(stats));
}
