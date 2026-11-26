using RouteSample=std::array<float,ROUTE_INPUT+8>;
std::vector<RouteSample> route_practice(const Weights &w,uint64_t start,int n,int threads) {
    if(n<1 || threads<1) throw std::invalid_argument("positive route sample dimensions required");
    std::vector<std::vector<RouteSample>> runs(n);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for(int i=0;i<n;++i) {
        Env e;e.record=false;e.reset(start+i,20,true,CharacterClass::IRONCLAD);
        std::mt19937 rng(static_cast<uint32_t>(start+i));
        std::array<float,OBS_SIZE> obs,end;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        for(int step=0;step<3500 && !e.done();++step) {
            e.observe(obs.data());e.mask(mask.data());
            View v(obs.data(),w);
            v.all_scores(mask.data(),scores.data());
            int kind=v.route_kind();
            if(kind>=0 && std::count(mask.begin()+ACT_CHOOSE,mask.begin()+ACT_CHOOSE+7,1)>1 &&
               rng()%(obs[4]>=3?1:kind==1?2:4)==0) {
                for(int j=0;j<7;++j) if(mask[ACT_CHOOSE+j]) {
                    Env branch=e;
                    if(!branch.step(ACT_CHOOSE+j)) throw std::runtime_error("invalid route branch");
                    auto near=advance(branch,w,nullptr,0,false,int(obs[3])+6,3499-step);
                    branch.observe(end.data());
                    auto final=advance(branch,w,nullptr,0,false,0,3499-step-near.steps);
                    RouteSample row{};
                    row[0]=i;row[1]=step;row[2]=kind;
                    // Use the parent policy's score as the feature even during
                    // later passes that follow the network's choice.
                    std::array<double,N_ACTIONS> raw{};
                    for(int k=0;k<7;++k) if(mask[ACT_CHOOSE+k]) raw[ACT_CHOOSE+k]=v.score(ACT_CHOOSE+k);
                    if(use_edits && kind==1) v.score_edits(mask.data(),raw.data());
                    auto x=v.route_input(kind,j,raw[ACT_CHOOSE+j]);
                    std::copy(x.begin(),x.end(),row.begin()+3);
                    row[ROUTE_INPUT+3]=float(near.floor>=int(obs[3])+6)*end[0]/std::max(1.f,end[1]);
                    row[ROUTE_INPUT+4]=final.floor-int(obs[3]);
                    row[ROUTE_INPUT+5]=final.act-int(obs[4]);
                    row[ROUTE_INPUT+6]=final.keys;
                    row[ROUTE_INPUT+7]=near.encounter || final.encounter;
                    runs[i].push_back(row);
                }
            }
            int a=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            if(!e.step(a)) throw std::runtime_error("invalid route policy action");
        }
    }
    std::vector<RouteSample> rows;
    for(auto &run:runs) rows.insert(rows.end(),run.begin(),run.end());
    return rows;
}
