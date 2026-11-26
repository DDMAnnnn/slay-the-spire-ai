// Prefix and suffix share the original RNG state and the same episode step budget.
using StageResult=std::array<int,6>;
StageResult stage_result(const Result &r) {
    return {r.floor,r.act,int(r.encounter),r.steps,int(r.heart),int(r.truncated)};
}

struct ActOpening {
    std::vector<Env> states;
    std::vector<StageResult> prefix;
    std::vector<int> arrivals;
    ActOpening(const Weights &w,uint64_t seed,int count,int threads) {
        if(count<1 || threads<1) throw std::invalid_argument("positive stage batch dimensions required");
        states.resize(count);
        prefix.resize(count);
        #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
        for(int i=0;i<count;++i) {
            auto &env=states[i];
            env.record=false;
            env.reset(seed+i,20,true,CharacterClass::IRONCLAD);
            prefix[i]=stage_result(advance(env,w,nullptr,2));
        }
        for(int i=0;i<count;++i)
            if(prefix[i][1]==2 && !states[i].done() && prefix[i][3]<3500) arrivals.push_back(i);
    }
    std::vector<std::vector<double>> features() const {
        std::vector<std::vector<double>> result;
        std::array<float,OBS_SIZE> obs{};
        const Weights unused{};
        for(int index:arrivals) {
            states[index].observe(obs.data());
            View view(obs.data(),unused);
            auto deck=view.deck_features();
            std::vector<double> x(deck.begin(),deck.begin()+14);
            x.insert(x.end(),{obs[0]/100.,obs[1]/100.,obs[12],obs[13],obs[14]});
            x.insert(x.end(),view.deck.begin(),view.deck.end());
            size_t relics=x.size(),potions=relics+std::size(relicNames);
            x.resize(potions+std::size(potionNames),0.);
            for(int j=0;j<MAX_RELICS;++j) {
                int id=int(obs[OBS_RELICS+j*RELIC_W])-1;
                if(id>=0) x.at(relics+id)=1.;
            }
            for(int j=0;j<5;++j) {
                int id=int(obs[17+j])-1;
                if(id>=0) ++x.at(potions+id);
            }
            result.push_back(std::move(x));
        }
        return result;
    }
    std::vector<StageResult> finish(const Weights &w,const std::vector<int> &ids,int threads) const {
        if(threads<1) throw std::invalid_argument("positive threads required");
        for(int i:ids)
            if(!std::binary_search(arrivals.begin(),arrivals.end(),i)) throw std::invalid_argument("unavailable stage");
        std::vector<StageResult> rows(ids.size());
        #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
        for(int j=0;j<int(ids.size());++j) {
            const int i=ids[j], spent=prefix[i][3];
            Env env=states[i];
            auto r=advance(env,w,nullptr,0,false,0,3500-spent);
            r.steps+=spent;
            rows[j]=stage_result(r);
        }
        return rows;
    }
};
