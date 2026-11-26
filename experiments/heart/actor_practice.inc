using ActorExample=std::array<float,17+ACTOR_INPUT>;
std::vector<ActorExample> actor_practice(const Weights &w,uint64_t seed,int n,int threads) {
    if(n<1 || threads<1) throw std::invalid_argument("positive practice dimensions required");
    std::vector<std::vector<ActorExample>> episodes(n);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for(int j=0;j<n;++j) {
        Env e; e.record=false; e.reset(seed+j,20,true,CharacterClass::IRONCLAD);
        std::mt19937 rng(static_cast<unsigned>(seed+j));
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        int last=-1;
        for(int k=0;k<3500 && !e.done();++k) {
            e.observe(obs.data()); e.mask(mask.data());
            View v(obs.data(),w); v.all_scores(mask.data(),scores.data());
            int best=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            int turn=int(obs[3])*1000+int(v.p[4]);
            if(mask[ACT_END_TURN] && turn!=last && rng()%(obs[4]>=3?5:25)==0) {
                last=turn;
                for(int a=0;a<=ACT_END_TURN;++a) if(mask[a] && (a<ACT_DISCARD || a==ACT_END_TURN)) {
                    int target=v.actor_target(a);
                    auto x=v.actor_features(a,target);
                    ActorExample row{};
                    row[0]=j; row[1]=k; row[2]=v.actor_item(a); row[3]=target<0?0:int(v.enemy(target)[0]);
                    for(int h=0;h<v.p[5];++h) row[4+h]=obs[OBS_HAND+h*HAND_W];
                    std::copy(x.begin(),x.end(),row.begin()+14);
                    Env branch=e;
                    if(!branch.step(a)) throw std::runtime_error("invalid action in training branch");
                    row[14+ACTOR_INPUT]=a==best;
                    row[15+ACTOR_INPUT]=float(finish_fight(branch,w));
                    auto r=advance(branch,w);
                    row[16+ACTOR_INPUT]=float(r.floor+6*r.act+2*r.keys+100*r.encounter-8*r.truncated);
                    episodes[j].push_back(row);
                }
            }
            if(!e.step(best)) throw std::runtime_error("invalid behavior action");
        }
    }
    std::vector<ActorExample> out;
    for(auto &ep:episodes) out.insert(out.end(),ep.begin(),ep.end());
    return out;
}
