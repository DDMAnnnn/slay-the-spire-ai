// Branches share the deck, inventory and RNG state at the chest. Only training
// receives the returns. The live scorer sees the offered relic ID.
std::vector<std::array<int,3>> practice_first_boss(const Weights &w,uint64_t start,int count,int threads) {
    if(count<1 || threads<1) throw std::invalid_argument("positive practice dimensions required");
    std::vector<std::vector<std::array<int,3>>> examples(count);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,8)
    for(int j=0;j<count;++j) {
        Env e; e.record=false; e.reset(start+j,20,true,CharacterClass::IRONCLAD);
        advance(e,w,nullptr,0,true);
        if(e.done()) continue;
        std::array<float,OBS_SIZE> obs;
        e.observe(obs.data());
        if(obs[4]!=1 || obs[7] || ScreenState(int(obs[6]))!=ScreenState::BOSS_RELIC_REWARDS) continue;
        std::array<uint8_t,N_ACTIONS> mask;
        e.mask(mask.data());
        auto outcome=[&](int action) {
            Env branch=e;
            if(!branch.step(action)) throw std::runtime_error("invalid boss relic alternative");
            auto r=advance(branch,w);
            return r.floor + 6*r.act + 2*r.keys + 100*int(r.heart) - 8*int(r.truncated);
        };
        int skip=-1;
        for(int i=0;i<N_CHOICES;++i)
            if(mask[ACT_CHOOSE+i] && ChoiceKind(int(obs[OBS_CHOICES+i*CHOICE_W]))==ChoiceKind::SKIP)
                skip=ACT_CHOOSE+i;
        if(skip<0) continue;
        const int reference=outcome(skip);
        for(int i=0;i<N_CHOICES;++i) {
            const auto c=obs.data()+OBS_CHOICES+i*CHOICE_W;
            if(mask[ACT_CHOOSE+i] && ChoiceKind(int(c[0]))==ChoiceKind::RELIC)
                examples[j].push_back({j,int(c[1])-1,outcome(ACT_CHOOSE+i)-reference});
        }
    }
    std::vector<std::array<int,3>> rows;
    for(const auto &episode:examples) rows.insert(rows.end(),episode.begin(),episode.end());
    return rows;
}
