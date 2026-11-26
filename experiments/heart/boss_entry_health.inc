// Intervene before battle initialization so HP-dependent powers and the boss's
// opening action see the new HP. Both arms take the same public map action.
auto boss_entry_health(const Bank &bank,const Weights &w,int bonus,int threads) {
    if(threads<1 || bonus < -1) throw std::invalid_argument("invalid HP intervention");
    std::vector<std::array<float,10>> results(bank.states.size());
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
    for(int i=0;i<int(bank.states.size());++i) {
        Env e=bank.states[i];
        std::array<float,OBS_SIZE> obs;std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> q;
        bool found=false;
        for(int k=0;k<30 && !e.done() && !e.inBattle;++k) {
            e.observe(obs.data());e.mask(mask.data());
            View view(obs.data(),w);view.base_scores(mask.data(),q.data());
            int a=int(std::max_element(q.begin(),q.end())-q.begin());
            Env before=e;
            if(!e.step(a)) throw std::runtime_error("invalid pre-boss action");
            if(!e.inBattle) continue;
            if(e.gc.curRoom!=Room::BOSS) break;
            const auto opponent=e.bc.encounter;
            int natural=before.gc.curHp,maximum=before.gc.maxHp;
            before.gc.curHp=bonus<0?maximum:std::min(maximum,natural+bonus);
            int assigned=before.gc.curHp;
            if(!before.step(a) || !before.inBattle || before.bc.encounter!=opponent)
                throw std::runtime_error("boss or route changed during the HP intervention");
            before.observe(obs.data());before.mask(mask.data());
            View battle(obs.data(),w);battle.all_scores(mask.data(),q.data());
            auto result=finish_combat_action(before,w,int(std::max_element(q.begin(),q.end())-q.begin()));
            results[i]={float(opponent),float(natural),float(maximum),float(assigned),obs[0],float(result.win),
                        result.health,result.remaining,result.potions,float(result.capped)};
            found=true;
        }
        if(!found) results[i][0]=-1;
    }
    return results;
}
