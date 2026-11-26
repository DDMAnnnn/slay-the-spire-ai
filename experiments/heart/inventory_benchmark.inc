using Inventory=std::pair<std::vector<int>,std::vector<int>>;
std::vector<Inventory> bank_inventories(const Bank &bank) {
    std::vector<Inventory> result;
    std::array<float,OBS_SIZE> obs;
    for(const auto &e:bank.states) {
        e.observe(obs.data());
        Inventory inv;
        for(int j=0;j<MAX_DECK && obs[OBS_DECK+j*CARD_W];++j) {
            auto c=obs.data()+OBS_DECK+j*CARD_W;
            inv.first.push_back(int(c[0])*256+int(c[1]));
        }
        for(int j=0;j<MAX_RELICS && obs[OBS_RELICS+j*RELIC_W];++j)
            inv.second.push_back(int(obs[OBS_RELICS+j*RELIC_W]));
        result.push_back(std::move(inv));
    }
    return result;
}

void validate_inventory(const Inventory &inv) {
    if(inv.first.empty() || inv.first.size()>Deck::MAX_SIZE || inv.second.size()>MAX_RELICS)
        throw std::invalid_argument("invalid inventory size");
    for(int c:inv.first) if(c/256<2 || c/256>CARD_COUNT || (c%256)>1)
        throw std::invalid_argument("invalid inventory card or upgrade");
    for(int r:inv.second) if(r<1 || r>int(std::size(relicNames)))
        throw std::invalid_argument("invalid inventory relic");
}

std::vector<std::array<int,5>> inventory_benchmark(const Weights &w,const std::vector<Inventory> &profiles,
        int draws,int seedOffset,int threads) {
    if(draws<1 || threads<1 || seedOffset<0) throw std::invalid_argument("invalid diagnostic dimensions");
    for(const auto &inv:profiles) validate_inventory(inv);
    const Encounter fights[]={Encounter::TIME_EATER,Encounter::AWAKENED_ONE,Encounter::DONU_AND_DECA,Encounter::SHIELD_AND_SPEAR};
    std::vector<std::array<int,5>> result(profiles.size()*4*draws);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
    for(int i=0;i<int(profiles.size());++i) {
        for(int fight=0;fight<4;++fight) {
            const auto &inv=profiles[i];
            std::vector<int> key{int(fights[fight]),fight==3?4:3,80,2,20,0,0,0,0,0};
            key.insert(key.end(),inv.first.begin(),inv.first.end());key.push_back(-1);
            key.insert(key.end(),inv.second.begin(),inv.second.end());
            for(int s=0;s<draws;++s) {
                auto e=scenario_start(key,seedOffset+s,64);
                std::array<float,OBS_SIZE> obs;
                std::array<uint8_t,N_ACTIONS> mask;
                std::array<double,N_ACTIONS> scores;
                int k=0;
                for(;k<450 && e.inBattle && !e.done();++k) {
                    e.observe(obs.data());e.mask(mask.data());
                    View v(obs.data(),w);v.all_scores(mask.data(),scores.data());
                    if(!e.step(int(std::max_element(scores.begin(),scores.end())-scores.begin())))
                        throw std::runtime_error("invalid benchmark action");
                }
                bool win=e.bc.outcome==Outcome::PLAYER_VICTORY;
                result[(i*4+fight)*draws+s]={i,fight,int(win),win?e.bc.player.curHp:0,int(k==450 && e.inBattle && !e.done())};
            }
        }
    }
    return result;
}
