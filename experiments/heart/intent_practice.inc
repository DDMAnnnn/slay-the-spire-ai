// Estimate the attack displayed one, two and three turns later, conditional
// on the same monster still being present and targetable at that time.
using IntentStat=std::array<double,4>;
using IntentStats=std::vector<std::array<IntentStat,3>>;
IntentStats intent_practice(const Weights &w,uint64_t start,int n,int threads) {
    if(n<1 || threads<1) throw std::invalid_argument("positive batch dimensions required");
    const size_t keys=intent_forecast.size()*18;
    std::vector<IntentStats> worker(threads,IntentStats(keys));
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int shard=0;shard<threads;++shard) {
        auto &stats=worker[shard];
        for(int j=shard;j<n;j+=threads) {
            Env env; env.record=false; env.reset(start+j,20,true,CharacterClass::IRONCLAD);
            std::array<float,OBS_SIZE> obs;
            std::array<uint8_t,N_ACTIONS> mask;
            std::array<double,N_ACTIONS> scores;
            struct Seen { int id=0,intent=0,turn=-1; double attack=0; };
            std::array<std::array<Seen,5>,3> history{};
            int floor=-1,previous=-1;
            for(int k=0;k<3500 && !env.done();++k) {
                env.observe(obs.data()); env.mask(mask.data());
                View v(obs.data(),w);
                if(!obs[7] || floor!=int(obs[3])) { history={}; previous=-1; floor=int(obs[3]); }
                const int turn=int(v.p[4]);
                if(mask[ACT_END_TURN] && previous!=turn) {
                    previous=turn;
                    for(int s=0;s<5;++s) {
                        auto m=v.enemy(s);
                        Seen now{int(m[0]),int(m[7]),turn,m[8]*m[9]};
                        if(!m[6]) now.id=0;
                        for(int h=1;h<=3;++h) {
                            const auto &old=history[(turn+3-h)%3][s];
                            if(!now.id || old.id!=now.id || old.turn!=turn-h) break;
                            auto &cell=stats[old.id*18+old.intent][h-1];
                            double err=now.attack-old.attack;
                            cell[0]++; cell[1]+=now.attack; cell[2]+=now.attack*now.attack;
                            cell[3]+=err*err;
                        }
                        history[turn%3][s]=now;
                    }
                }
                v.all_scores(mask.data(),scores.data());
                int choice=int(std::max_element(scores.begin(),scores.end())-scores.begin());
                if(!env.step(choice)) throw std::runtime_error("illegal forecast practice action");
            }
        }
    }
    for(int t=1;t<threads;++t) for(size_t k=0;k<keys;++k)
        for(int h=0;h<3;++h) for(int f=0;f<4;++f) worker[0][k][h][f]+=worker[t][k][h][f];
    return std::move(worker[0]);
}
