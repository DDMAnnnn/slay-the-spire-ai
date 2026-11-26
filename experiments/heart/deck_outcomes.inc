// At most one snapshot per floor and only before a camp/shop decision a player can see.
std::vector<std::vector<int>> collect_deck_outcomes(const Weights &w,uint64_t start,int count,int threads) {
    std::vector<std::vector<std::vector<int>>> games(count);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,8)
    for(int n=0;n<count;++n) {
        Env e;e.record=false;e.reset(start+n,20,true,CharacterClass::IRONCLAD);
        std::array<float,OBS_SIZE> o;std::array<uint8_t,N_ACTIONS> mask;std::array<double,N_ACTIONS> q;
        int last=-1;
        for(int k=0;k<3500 && !e.done();++k) {
            e.observe(o.data());e.mask(mask.data());
            auto screen=ScreenState(int(o[6]));
            if(!o[7] && o[4]<=3 && last!=o[3] && (screen==ScreenState::REST_ROOM || screen==ScreenState::SHOP_ROOM)
                && std::count(mask.begin()+ACT_CHOOSE,mask.begin()+ACT_CHOOSE+N_CHOICES,1)>1) {
                last=int(o[3]);
                std::vector<int> r={n,0,int(o[4]),last,screen==ScreenState::SHOP_ROOM,int(o[0]),int(o[1]),int(o[2])};
                for(int j=0;j<MAX_DECK && o[OBS_DECK+j*CARD_W];++j) r.push_back(int(o[OBS_DECK+j*CARD_W])-1);
                for(int j=0;j<MAX_RELICS && o[OBS_RELICS+j*RELIC_W];++j) r.push_back(CARD_COUNT+int(o[OBS_RELICS+j*RELIC_W])-1);
                games[n].push_back(std::move(r));
            }
            View v(o.data(),w);v.all_scores(mask.data(),q.data());
            if(!e.step(int(std::max_element(q.begin(),q.end())-q.begin()))) throw std::runtime_error("invalid inventory collection action");
        }
        for(auto &r:games[n]) {
            // Clear means actually advancing to the next act under this policy.
            r[1]=e.gc.act>r[2];
        }
        if(!e.done()) games[n].clear();
    }
    std::vector<std::vector<int>> rows;
    for(auto &game:games) for(auto &row:game) rows.push_back(std::move(row));
    return rows;
}
