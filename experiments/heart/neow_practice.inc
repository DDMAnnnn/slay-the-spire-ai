// Observe one complete natural run per opening alternative. Only the trainer receives
// these counterfactual outcomes. A deployed policy gets the visible bonus and drawback.
std::vector<std::array<int,7>> neow_practice(const Weights &w,uint64_t seed,int n,int threads) {
    if(n<1 || threads<1) throw std::invalid_argument("positive practice dimensions required");
    std::vector<std::array<int,7>> data(4*n);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,8)
    for(int j=0;j<n;++j) {
        Env e; e.record=false; e.reset(seed+j,20,true,CharacterClass::IRONCLAD);
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        e.observe(obs.data()); e.mask(mask.data());
        for(int i=0;i<4;++i) {
            auto c=obs.data()+OBS_CHOICES+i*CHOICE_W;
            if(!mask[ACT_CHOOSE+i] || ChoiceKind(int(c[0]))!=ChoiceKind::NEOW_OPTION)
                continue;
            Env branch=e; branch.step(ACT_CHOOSE+i);
            auto r=advance(branch,w);
            data[4*j+i]={int(c[1]),int(c[2]),r.floor,r.act,r.keys,int(r.heart),int(r.truncated)};
        }
    }
    return data;
}
