// The second play can depend on the new hand or intent the player actually sees.
// For each independent world, use other compatible worlds to select that play.
struct NextCombatGroup {
    std::array<float,OBS_SIZE> shown;
    std::array<uint8_t,N_ACTIONS> mask;
    std::vector<int> options;
    std::vector<size_t> worlds;
};

double fight_utility(const FightReturn &result,const float *o) {
    if(!result.win) return -28.*result.remaining-4.*result.potions;
    const double old=belief_win_value+result.health-4.*result.potions;
    if(!resource_search_mix || o[4]<1 || o[4]>3) return old;
    ResourcePosition after(int(o[4]),int(o[3]),result.exit_health,o[1]);
    ResourcePosition reference(int(o[4]),int(o[3]),.5*o[1],o[1]);
    std::array<int,5> initial{};
    for(int k=0;k<5;++k) initial[k]=int(o[17+k]);
    const double learned=belief_win_value+.5*o[1]+after.health()+after.inventory(result.kept)
                        -reference.health()-reference.inventory(initial);
    return old+resource_search_mix*(learned-old);
}

double crossfit_advantage(const std::vector<std::vector<double>> &returns,double margin) {
    if(returns.size()<2 || returns.front().size()<2) return 0.;
    std::vector<double> totals(returns.front().size());
    for(const auto &row:returns)
        std::transform(row.begin(),row.end(),totals.begin(),totals.begin(),std::plus<double>());
    double gain=0.;
    for(const auto &held:returns) {
        size_t chosen=0;
        double best=totals[0]-held[0]+margin*(returns.size()-1);
        for(size_t a=1;a<totals.size();++a) {
            double estimate=totals[a]-held[a];
            if(estimate>best) {best=estimate;chosen=a;}
        }
        gain+=held[chosen]-held[0];
    }
    return gain;
}

double next_play_gain(const std::vector<Env> &roots,int first,const Weights &w,int width) {
    std::vector<Env> next;
    std::vector<NextCombatGroup> groups;
    next.reserve(roots.size());
    for(const auto &root:roots) {
        auto e=root;
        if(!e.step(first)) throw std::runtime_error("invalid first play in a public particle");
        if(e.done() || !e.inBattle || e.bc.outcome!=Outcome::UNDECIDED ||
           e.gc.floorNum!=root.gc.floorNum || e.bc.encounter!=root.bc.encounter) continue;
        std::array<float,OBS_SIZE> obs;std::array<uint8_t,N_ACTIONS> mask;
        e.observe(obs.data());e.mask(mask.data());
        if(!mask[ACT_END_TURN]) continue;
        View view(obs.data(),w);
        std::array<double,N_ACTIONS> scores;view.all_scores(mask.data(),scores.data());
        int parent=int(std::max_element(scores.begin(),scores.end())-scores.begin());
        if(parent>=ACT_DISCARD && parent!=ACT_END_TURN) continue;
        auto options=distinct_combat_actions(parent,obs.data(),mask.data());
        if(options.size()<2) continue;
        std::stable_sort(options.begin()+1,options.end(),[&](int a,int b){return scores[a]>scores[b];});
        if(int(options.size())>width) options.resize(width);
        // Omit the independently generated map, which is unrelated to this current fight.
        auto same=[&](const NextCombatGroup &g) {
            return mask==g.mask && options==g.options &&
                std::equal(obs.begin(),obs.begin()+9,g.shown.begin()) &&
                std::equal(obs.begin()+16,obs.begin()+22,g.shown.begin()+16) &&
                std::equal(obs.begin()+OBS_RELICS,obs.begin()+OBS_MAP,g.shown.begin()+OBS_RELICS) &&
                std::equal(obs.begin()+OBS_PLAYER,obs.end(),g.shown.begin()+OBS_PLAYER);
        };
        auto it=std::find_if(groups.begin(),groups.end(),same);
        if(it==groups.end()) {
            groups.push_back({obs,mask,std::move(options),{}});
            it=std::prev(groups.end());
        }
        it->worlds.push_back(next.size());
        next.push_back(std::move(e));
    }
    double gain=0.;
    for(const auto &g:groups) if(g.worlds.size()>1) {
        std::vector<std::vector<double>> returns;
        for(auto i:g.worlds) {
            auto &row=returns.emplace_back();
            for(int a:g.options) row.push_back(fight_utility(finish_combat_action(next[i],w,a),g.shown.data()));
        }
        // Require a two-HP advantage on the selecting worlds before changing the second play.
        gain+=crossfit_advantage(returns,2.);
        if(continuation_audit) {++(*continuation_audit)[1];(*continuation_audit)[2]+=int(g.worlds.size());}
    }
    return gain/roots.size();
}

void refine_first_plays(const std::vector<int> &actions,const std::vector<Env> &roots,
                        const Weights &w,double *scores,int width) {
    if(roots.size()<2) return;
    std::vector<int> finalists=actions;
    std::stable_sort(finalists.begin(),finalists.end(),[&](int a,int b){return scores[a]>scores[b];});
    if(finalists.size()>3) finalists.resize(3);
    if(std::find(finalists.begin(),finalists.end(),actions[0])==finalists.end()) finalists.back()=actions[0];
    auto before=int(std::max_element(scores,scores+N_ACTIONS)-scores);
    for(int a:finalists) scores[a]+=belief_scale*next_play_gain(roots,a,w,width);
    if(continuation_audit) {
        ++(*continuation_audit)[0];
        (*continuation_audit)[3]+=before!=int(std::max_element(scores,scores+N_ACTIONS)-scores);
    }
}
