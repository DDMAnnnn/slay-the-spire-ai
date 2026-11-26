struct PreferenceBatch {
    // Potion ID, public features, full-run advantage. Card pairs carry baseline score
    // difference, the two cost normalizers, and a signed end-of-fight HP advantage.
    std::vector<std::array<double,19>> drinks;
    std::vector<std::array<double,6>> cards;
};
auto preference_practice(const Weights &w,uint64_t seed,int n,int threads) {
    if(n<1 || threads<1) throw std::invalid_argument("positive practice dimensions required");
    std::vector<PreferenceBatch> batches(n);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for(int j=0;j<n;++j) {
        Env e; e.record=false; e.reset(seed+j,20,true,CharacterClass::IRONCLAD);
        std::mt19937 rng(static_cast<unsigned>(seed+j));
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        for(int k=0;k<3500 && !e.done();++k) {
            e.observe(obs.data()); e.mask(mask.data());
            View v(obs.data(),w); v.all_scores(mask.data(),scores.data());
            int best=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            if(mask[ACT_END_TURN] && rng()%24==0) {
                int hold=ACT_END_TURN;
                std::vector<int> plays;
                for(int a=0;a<ACT_POTION;++a) if(mask[a]) {
                    if(scores[a]>scores[hold]) hold=a;
                    plays.push_back(a);
                }
                auto branch=[&](int a) {
                    Env copy=e;
                    if(!copy.step(a)) throw std::runtime_error("illegal practice alternative");
                    return copy;
                };
                bool has_reference=false;
                double reference=0;
                auto run_value=[&](int a) {
                    Env copy=branch(a); auto r=advance(copy,w);
                    return r.floor+6*r.act+2*r.keys+100*int(r.heart)-8*int(r.truncated);
                };
                for(int a=ACT_POTION;a<ACT_DISCARD;++a) if(mask[a]) {
                    if(!has_reference) {reference=run_value(hold);has_reference=true;}
                    std::array<double,19> row{};
                    row[0]=int(obs[17+(a-ACT_POTION)/5])-1;
                    auto x=v.drink_features(a);
                    std::copy(x.begin(),x.end(),row.begin()+1);
                    row.back()=run_value(a)-reference;
                    batches[j].drinks.push_back(row);
                }
                if(!plays.empty()) {
                    int alt=plays[rng()%plays.size()];
                    auto id=[&](int a){return a==ACT_END_TURN?-1:int(obs[OBS_HAND+a/5*HAND_W])-1;};
                    auto gain=[&](int a){return a==ACT_END_TURN?0.:
                        1./(1.+v.w[CostPenalty]*std::max(0,int(obs[OBS_HAND+a/5*HAND_W+2])-1));};
                    if(id(alt)!=id(hold)) {
                        Env first=branch(alt), second=branch(hold);
                        double advantage=finish_fight(first,w)-finish_fight(second,w);
                        batches[j].cards.push_back({double(id(alt)),double(id(hold)),
                            scores[alt]-scores[hold],gain(alt),gain(hold),advantage});
                    }
                }
            }
            if(!e.step(best)) throw std::runtime_error("illegal practice move");
        }
    }
    PreferenceBatch all;
    for(auto &b:batches) {
        all.drinks.insert(all.drinks.end(),b.drinks.begin(),b.drinks.end());
        all.cards.insert(all.cards.end(),b.cards.begin(),b.cards.end());
    }
    return std::make_pair(all.drinks,all.cards);
}
