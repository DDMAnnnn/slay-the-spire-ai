// Training-only branches share the same earned position and future randomness.
// The fitted policy receives only the visible graph and resources.
using MapPracticeRow=std::tuple<std::vector<int>,std::vector<int>,std::vector<std::array<int,7>>>;
std::vector<MapPracticeRow> map_preferences(const Weights &w,const std::vector<uint64_t>& seeds,int act,int threads) {
    std::vector<MapPracticeRow> records(seeds.size());
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for(int ix=0;ix<int(seeds.size());++ix) {
        Env e,candidate;e.record=false;e.reset(seeds[ix],20,true,CharacterClass::IRONCLAD);
        std::mt19937 rng(uint32_t(seeds[ix])^0x637da53u);int eligible=0;
        std::array<float,OBS_SIZE> o;std::array<uint8_t,N_ACTIONS> legal;std::array<double,N_ACTIONS> q;
        for(int step=0;step<3500 && !e.done() && e.gc.act<=act;++step) {
            e.observe(o.data());e.mask(legal.data());
            if(e.gc.act==act && !e.inBattle && e.gc.screenState==ScreenState::MAP_SCREEN && o[11]<14 &&
               std::count(legal.begin()+ACT_CHOOSE,legal.begin()+ACT_CHOOSE+7,1)>1) {
                if(std::uniform_int_distribution<int>(0,eligible++)(rng)==0) {
                    candidate=e;candidate.gc.map=std::make_shared<Map>(*e.gc.map);
                }
            }
            View v(o.data(),w);v.all_scores(legal.data(),q.data());
            if(!e.step(int(std::max_element(q.begin(),q.end())-q.begin()))) throw std::runtime_error("invalid practice route");
        }
        if(!eligible) continue;
        candidate.observe(o.data());candidate.mask(legal.data());
        View v(o.data(),w);v.all_scores(legal.data(),q.data());
        int parent=int(std::max_element(q.begin(),q.end())-q.begin());
        if(parent<ACT_CHOOSE || parent>=ACT_CHOOSE+7) continue;
        int potions=0;for(int k=0;k<5;++k) potions+=o[17+k]>0;
        std::vector<int> context={ix,parent-ACT_CHOOSE,act,int(o[0]),int(o[1]),int(o[2]),potions,int(o[22]),
            int(o[12]),int(o[13]),int(o[14]),int(o[11])+1,int(o[24]),int(o[25])};
        std::vector<int> board,options;
        for(int j=0;j<105;++j) {
            auto m=o.data()+OBS_MAP+j*MAP_W;
            board.push_back(m[0]?int(m[0])-1:8);board.push_back(j>=98?0:int(m[1]));
        }
        for(int j=0;j<7;++j) if(legal[ACT_CHOOSE+j] && ACT_CHOOSE+j!=parent) options.push_back(j);
        std::shuffle(options.begin(),options.end(),rng);if(options.size()>2) options.resize(2);
        options.push_back(parent-ACT_CHOOSE);
        std::vector<std::array<int,7>> returns;
        for(int j:options) {
            Env branch=candidate;branch.gc.map=std::make_shared<Map>(*candidate.gc.map);
            if(!branch.step(ACT_CHOOSE+j)) throw std::runtime_error("illegal counterfactual route");
            auto r=advance(branch,w);
            returns.push_back({j,r.floor,r.act,r.keys,int(r.encounter),int(r.heart),int(r.truncated)});
        }
        records[ix]={std::move(context),std::move(board),std::move(returns)};
    }
    return records;
}
