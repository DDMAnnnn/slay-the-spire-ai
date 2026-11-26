// Training only. Compare alternatives after the same naturally reached decision.
// None of these simulator copies or their returns are accessible to the actor.
using Example = std::array<double,4>; // act, feature, paired advantage, floor
std::vector<Example> practice(const Weights& w,uint64_t seed,int episodes,int threads) {
    if(episodes<1 || threads<1) throw std::invalid_argument("positive training dimensions required");
    std::vector<std::vector<Example>> episodes_data(episodes);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,8)
    for(int j=0;j<episodes;++j) {
        std::mt19937 sampler(static_cast<unsigned>(seed+j));
        Env e; e.record=false; e.reset(seed+j,20,true,CharacterClass::IRONCLAD);
        std::array<float,OBS_SIZE> o;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        int last=-1;
        for(int step=0;step<3500 && !e.done();++step) {
            e.observe(o.data()); e.mask(mask.data());
            View v(o.data(),w); v.all_scores(mask.data(),scores.data());
            const int best=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            const auto screen=ScreenState(int(o[6]));
            const int floor=int(o[3]), act=std::min(3,int(o[4]))-1;
            if(!o[7] && floor!=last && (screen==ScreenState::REWARDS || screen==ScreenState::EVENT_SCREEN)) {
                std::vector<int> actions;
                int skip=-1;
                bool other_loot=false;
                for(int i=0;i<N_CHOICES;++i) if(mask[ACT_CHOOSE+i]) {
                    auto c=o.data()+OBS_CHOICES+i*CHOICE_W;
                    auto kind=ChoiceKind(int(c[0]));
                    if(kind==ChoiceKind::SKIP) skip=ACT_CHOOSE+i;
                    else if(kind!=ChoiceKind::CARD && kind!=ChoiceKind::EVENT_OPTION) other_loot=true;
                    if(v.feature(i)>=0) actions.push_back(ACT_CHOOSE+i);
                }
                if(!other_loot && !actions.empty() && (skip>=0 || actions.size()>=2)) {
                    last=floor;
                    if(o[4]>1 || sampler()%3==0) {
                        if(skip>=0) actions.push_back(skip);
                        std::vector<double> returns;
                        for(int a:actions) {
                            Env branch=e;
                            if(!branch.step(a)) throw std::runtime_error("illegal training alternative");
                            auto r=advance(branch,w);
                            returns.push_back(r.floor + 6*r.act + 2*r.keys + 100*r.heart - 8*r.truncated);
                        }
                        double ref=skip>=0?returns.back():std::accumulate(returns.begin(),returns.end(),0.)/returns.size();
                        for(size_t k=0;k<actions.size();++k) {
                            int f=v.feature(actions[k]-ACT_CHOOSE);
                            if(f>=0) episodes_data[j].push_back({double(act),double(f),returns[k]-ref,double(floor)});
                        }
                    }
                }
            }
            if(!e.step(best)) throw std::runtime_error("illegal training action");
        }
    }
    std::vector<Example> result;
    for(const auto &episode:episodes_data) result.insert(result.end(),episode.begin(),episode.end());
    return result;
}
