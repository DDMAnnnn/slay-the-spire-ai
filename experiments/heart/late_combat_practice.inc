// One row per action. Seed offset and decision index identify paired outcomes.
using LateExample=std::array<float,39>;
std::vector<LateExample> late_combat_practice(const Weights &w,uint64_t start,int n,int threads) {
    if(n<1 || threads<1) throw std::invalid_argument("positive batch dimensions required");
    std::vector<std::vector<LateExample>> samples(n);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for(int j=0;j<n;++j) {
        std::mt19937 rng(static_cast<unsigned>(start+j));
        Env e; e.record=false; e.reset(start+j,20,true,CharacterClass::IRONCLAD);
        advance(e,w,nullptr,2);
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        int last_turn=-1;
        for(int k=0;k<3500 && !e.done();++k) {
            e.observe(obs.data()); e.mask(mask.data());
            View v(obs.data(),w); v.all_scores(mask.data(),scores.data());
            int best=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            int turn=int(obs[3])*1000+int(v.p[4]);
            if(mask[ACT_END_TURN] && (best<ACT_POTION || best==ACT_END_TURN) &&
               last_turn!=turn && rng()%(obs[4]>=3?3:20)==0) {
                last_turn=turn;
                for(int a=0;a<=ACT_END_TURN;++a) {
                    if(!mask[a] || (a>=ACT_POTION && a!=ACT_END_TURN)) continue;
                    LateExample row{};
                    row[0]=float(j); row[1]=float(k); row[2]=float(v.card_index(a)); row[3]=float(v.enemy_index(a));
                    for(int h=0;h<v.p[5];++h) row[4+h]=obs[OBS_HAND+h*HAND_W];
                    auto x=v.combat_features(a,scores[a]-scores[best]);
                    std::copy(x.begin(),x.end(),row.begin()+14);
                    row[36]=float(scores[a]-scores[best]); row[37]=obs[4];
                    Env copy=e;
                    if(!copy.step(a)) throw std::runtime_error("illegal combat alternative");
                    row[38]=float(finish_fight(copy,w));
                    samples[j].push_back(row);
                }
            }
            if(!e.step(best)) throw std::runtime_error("illegal late practice action");
        }
    }
    std::vector<LateExample> all;
    for(auto &ep:samples) all.insert(all.end(),ep.begin(),ep.end());
    return all;
}
