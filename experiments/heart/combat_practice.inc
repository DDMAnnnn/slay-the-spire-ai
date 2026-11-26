// End-of-fight labels are available only in the training driver.
using CombatSample=std::array<float,25>;
double finish_fight(Env &e,const Weights &w) {
    std::array<float,OBS_SIZE> obs;
    std::array<uint8_t,N_ACTIONS> mask;
    std::array<double,N_ACTIONS> scores;
    int turns=0;
    for(int k=0;k<400 && e.inBattle && !e.done();++k) {
        e.observe(obs.data()); e.mask(mask.data());
        View v(obs.data(),w); v.all_scores(mask.data(),scores.data());
        auto a=int(std::max_element(scores.begin(),scores.end())-scores.begin());
        turns+=a==ACT_END_TURN;
        if(!e.step(a)) throw std::runtime_error("illegal combat branch action");
    }
    e.observe(obs.data());
    if(e.done() && !e.won()) {
        double enemy_hp=0;
        for(int i=0;i<5;++i) enemy_hp+=obs[OBS_MONSTERS+i*MONSTER_W+1];
        return -25-.06*enemy_hp;
    }
    return obs[0]-.1*turns;
}
std::vector<CombatSample> combat_practice(const Weights& w,uint64_t seed,int n,int threads) {
    if(n<1 || threads<1) throw std::invalid_argument("positive training dimensions required");
    std::vector<std::vector<CombatSample>> per_episode(n);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for(int j=0;j<n;++j) {
        std::mt19937 random(static_cast<unsigned>(seed+j));
        Env e; e.record=false; e.reset(seed+j,20,true,CharacterClass::IRONCLAD);
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        int previous=-1;
        for(int k=0;k<3500 && !e.done();++k) {
            e.observe(obs.data()); e.mask(mask.data());
            View v(obs.data(),w); v.all_scores(mask.data(),scores.data());
            int best=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            const int turn=int(obs[3])*100+int(v.p[4]);
            auto room=Room(int(obs[8])-1);
            int stride=obs[4]>=3?2:room==Room::ELITE || room==Room::BOSS?6:24;
            if(mask[ACT_END_TURN] && previous!=turn && random()%stride==0 &&
               (best<ACT_POTION || best==ACT_END_TURN)) {
                previous=turn;
                std::vector<CombatSample> group;
                std::vector<double> returns;
                for(int a=0;a<=ACT_END_TURN;++a) {
                    if(!mask[a] || (a>=ACT_POTION && a!=ACT_END_TURN) || scores[a]<scores[best]-50) continue;
                    auto x=v.combat_features(a,scores[a]-scores[best]);
                    CombatSample row{}; row[0]=float(v.card_index(a));row[1]=float(v.enemy_index(a));
                    std::copy(x.begin(),x.end(),row.begin()+2);
                    Env future=e;
                    if(!future.step(a)) throw std::runtime_error("illegal training action");
                    returns.push_back(finish_fight(future,w));
                    group.push_back(row);
                }
                if(group.size()>1) {
                    // An easy fight can give every move a high return. Center the labels within this decision.
                    // A positive label means this move beat the average alternative under the continuation policy.
                    double average=std::accumulate(returns.begin(),returns.end(),0.)/returns.size();
                    for(size_t i=0;i<group.size();++i) {
                        group[i].back()=float(returns[i]-average);
                        per_episode[j].push_back(group[i]);
                    }
                }
            }
            if(!e.step(best)) throw std::runtime_error("illegal combat practice action");
        }
    }
    std::vector<CombatSample> result;
    for(const auto &episode:per_episode) result.insert(result.end(),episode.begin(),episode.end());
    return result;
}
