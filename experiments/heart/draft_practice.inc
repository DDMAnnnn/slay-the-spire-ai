using DraftExample=std::array<double,19>;
std::vector<DraftExample> draft_practice(const Weights &w,uint64_t start,int count,int threads) {
    if(count<1 || threads<1) throw std::invalid_argument("positive practice dimensions required");
    std::vector<std::vector<DraftExample>> episodes(count);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for(int j=0;j<count;++j) {
        Env e; e.record=false; e.reset(start+j,20,true,CharacterClass::IRONCLAD);
        std::mt19937 rng(static_cast<unsigned>(start+j));
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        int last=-1;
        for(int k=0;k<3500 && !e.done();++k) {
            e.observe(obs.data()); e.mask(mask.data());
            View v(obs.data(),w); v.all_scores(mask.data(),scores.data());
            int skip=v.draft_skip(mask.data());
            if(skip>=0 && last!=int(obs[3]) && rng()%3==0) {
                last=int(obs[3]);
                auto outcome=[&](int a) {
                    Env branch=e;
                    if(!branch.step(a)) throw std::runtime_error("invalid draft alternative");
                    auto r=advance(branch,w);
                    return r.floor+6*r.act+2*r.keys+100*r.heart-8*r.truncated;
                };
                const int reference=outcome(skip);
                const auto deck=v.deck_features();
                for(int i=0;i<N_CHOICES;++i) if(mask[ACT_CHOOSE+i] && ACT_CHOOSE+i!=skip) {
                    const auto c=obs.data()+OBS_CHOICES+i*CHOICE_W;
                    auto x=v.offered_features(deck,c);
                    DraftExample row{}; row[0]=int(c[1])-1;
                    std::copy(x.begin(),x.end(),row.begin()+1);
                    row.back()=outcome(ACT_CHOOSE+i)-reference;
                    episodes[j].push_back(row);
                }
            }
            int a=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            if(!e.step(a)) throw std::runtime_error("invalid draft practice step");
        }
    }
    std::vector<DraftExample> all;
    for(const auto &ep:episodes) all.insert(all.end(),ep.begin(),ep.end());
    return all;
}
