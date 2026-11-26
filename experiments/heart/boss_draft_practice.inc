using BossDraftRow=std::array<float,32+CARD_COUNT+6>;

// Compare all cards with skipping on exactly the same real run and on the
// same independently generated boss fights. The deployment table uses neither seed.
std::vector<BossDraftRow> boss_draft_practice(const Weights &w,const std::vector<uint64_t> &seeds,
        int fromAct,int samples,int threads) {
    if(threads<1 || samples<1 || fromAct<1 || fromAct>3) throw std::invalid_argument("invalid draft practice");
    std::vector<std::vector<BossDraftRow>> episodes(seeds.size());
    #pragma omp parallel for schedule(dynamic,4) num_threads(threads)
    for(int i=0;i<int(seeds.size());++i) {
        Env e;e.record=false;e.reset(seeds[i],20,true,CharacterClass::IRONCLAD);
        if(fromAct>1) advance(e,w,nullptr,fromAct);
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        std::mt19937 rng(static_cast<uint32_t>(seeds[i]));
        int last=-1;
        for(int k=0;k<3300 && !e.done();++k) {
            e.observe(obs.data());e.mask(mask.data());
            View v(obs.data(),w);v.all_scores(mask.data(),scores.data());
            int skip=v.draft_skip(mask.data()),boss=v.visible_draft_boss();
            if(skip>=0 && boss>=0 && int(obs[3])!=last && obs[22]<MAX_DECK) {
                last=int(obs[3]);
                if(rng()%(obs[4]==1?4:obs[4]==2?2:1)==0) {
                    auto key=scenario_key(v,Encounter(int(obs[15])-1));
                    int first=int(rng()%100000);
                    auto future=[&](int action) {
                        Env copy=e;
                        if(!copy.step(action)) throw std::runtime_error("invalid draft alternative");
                        auto r=advance(copy,w);
                        return .08*r.floor+r.act+.15*r.keys+3*(r.act==4)+6*r.encounter;
                    };
                    auto practice=[&](const std::vector<int> &profile) {
                        double total=0;
                        for(int s=0;s<samples;++s)
                            total+=scenario_return(scenario_start(profile,first+s),w,0);
                        return total/samples;
                    };
                    const auto context=v.deck_features();
                    double reference=future(skip),skipBoss=practice(key);
                    for(int j=0;j<N_CHOICES;++j) if(mask[ACT_CHOOSE+j] && ACT_CHOOSE+j!=skip) {
                        const auto c=obs.data()+OBS_CHOICES+j*CHOICE_W;
                        auto changed=key;
                        auto split=std::find(changed.begin()+10,changed.end(),-1);
                        changed.insert(split,int(c[1])*256+int(c[2]));
                        auto x=v.boss_draft_input(context,boss,c,scores[ACT_CHOOSE+j]-scores[skip]);
                        BossDraftRow row{};
                        row[0]=i;row[1]=k;row[2]=int(c[1])-1;row[3]=obs[4];
                        std::copy(x.begin(),x.end(),row.begin()+4);
                        row[row.size()-2]=float(practice(changed)-skipBoss);
                        row.back()=float(future(ACT_CHOOSE+j)-reference);
                        episodes[i].push_back(row);
                    }
                }
            }
            int a=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            if(!e.step(a)) throw std::runtime_error("invalid draft behavior action");
        }
    }
    std::vector<BossDraftRow> rows;
    for(auto &episode:episodes) rows.insert(rows.end(),episode.begin(),episode.end());
    return rows;
}
