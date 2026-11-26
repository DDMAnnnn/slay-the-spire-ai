// Clear and remaining boss health are training labels. The policy only sees
// its public combat observation. Bank states come from natural floor 33 runs.
std::vector<std::array<double,4>> boss_results(const Bank &bank,const Weights &w,int threads) {
    if(threads<1) throw std::invalid_argument("positive threads required");
    std::vector<std::array<double,4>> results(bank.states.size());
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for(int i=0;i<int(bank.states.size());++i) {
        Env e=bank.states[i];
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        int boss=-1;
        double fraction=1.;
        for(int k=0;k<3500 && !e.done();++k) {
            e.observe(obs.data());
            if((boss>=0 && !obs[7]) || obs[4]!=2 || obs[3]!=33) break;
            View v(obs.data(),w);
            int slot=0,current=v.boss_index(slot);
            if(obs[7] && current>=0) {
                boss=current;
                fraction=v.enemy(slot)[1]/std::max(1.f,v.enemy(slot)[2]);
            }
            e.mask(mask.data());
            v.all_scores(mask.data(),scores.data());
            int a=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            if(!e.step(a)) break;
        }
        e.observe(obs.data());
        bool cleared=boss>=0 && !e.done() && !obs[7] && obs[0]>0;
        results[i]={double(boss),double(cleared),cleared?0.:fraction,obs[0]/std::max(1.f,obs[1])};
    }
    return results;
}
