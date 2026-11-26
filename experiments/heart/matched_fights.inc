struct MatchedFights {
    std::vector<Env> entries;
    std::vector<std::array<int,6>> info;
    MatchedFights(const Weights &w,uint64_t start,int count,int threads) {
        if(count<1 || threads<1) throw std::invalid_argument("positive collection dimensions required");
        std::vector<std::vector<Env>> games(count);
        std::vector<std::vector<std::array<int,6>>> labels(count);
        #pragma omp parallel for num_threads(threads) schedule(dynamic,2)
        for(int i=0;i<count;++i) {
            Env game,entry,control;
            game.record=false;game.reset(start+i,20,true,CharacterClass::IRONCLAD);
            bool tracking=false;int victories=0;
            std::mt19937 rng(static_cast<uint32_t>(start+i));
            std::array<float,OBS_SIZE> obs;
            std::array<uint8_t,N_ACTIONS> mask;
            std::array<double,N_ACTIONS> scores;
            auto append=[&](const Env &e,int won) {
                games[i].push_back(e);
                labels[i].push_back({i,won,e.gc.act,e.gc.floorNum,int(e.bc.encounter),e.gc.curHp});
            };
            for(int k=0;k<3500 && !game.done();++k) {
                if(game.inBattle && !tracking) {entry=game;tracking=true;}
                game.observe(obs.data());game.mask(mask.data());
                View v(obs.data(),w);v.base_scores(mask.data(),scores.data());
                if(!game.step(int(std::max_element(scores.begin(),scores.end())-scores.begin())))
                    throw std::runtime_error("invalid matched fight scout move");
                if(tracking && (!game.inBattle || game.done() || game.gc.floorNum!=entry.gc.floorNum ||
                                 game.bc.encounter!=entry.bc.encounter)) {
                    if(game.done() && game.bc.outcome==Outcome::PLAYER_LOSS) append(entry,0);
                    else if(std::uniform_int_distribution<int>(0,victories++)(rng)==0) control=entry;
                    tracking=false;
                }
            }
            if(victories) append(control,1);
        }
        for(int i=0;i<count;++i) {
            info.insert(info.end(),labels[i].begin(),labels[i].end());
            entries.insert(entries.end(),games[i].begin(),games[i].end());
        }
    }
    auto evaluate(const Weights &w,int threads) const {
        std::vector<std::array<float,4>> results(entries.size());
        #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
        for(int i=0;i<int(entries.size());++i) {
            Env e=entries[i];
            std::array<float,OBS_SIZE> obs;std::array<uint8_t,N_ACTIONS> mask;
            std::array<double,N_ACTIONS> scores;
            e.observe(obs.data());e.mask(mask.data());
            View v(obs.data(),w);v.all_scores(mask.data(),scores.data());
            auto r=finish_combat_action(e,w,int(std::max_element(scores.begin(),scores.end())-scores.begin()));
            results[i]={float(r.win),r.health,r.potions,float(r.capped)};
        }
        return results;
    }
};
