using CurriculumRow=std::array<float,7+CURRICULUM_INPUT>;
using CurriculumStats=std::array<int,7>;

std::pair<py::array_t<float>,CurriculumStats> curriculum_practice(
        const Bank &bank,const Weights &w,uint32_t seed,int groups,int paths,int window,int threads) {
    if(bank.states.empty() || groups<1 || paths<2 || window<1 || threads<1)
        throw std::invalid_argument("invalid matched rollout dimensions");
    std::vector<std::vector<CurriculumRow>> output(groups);
    std::vector<CurriculumStats> stats(groups);
    #pragma omp parallel for schedule(dynamic,2) num_threads(threads)
    for(int j=0;j<groups;++j) {
        std::mt19937 rng(seed+j*7919u);
        Env initial=bank.states[rng()%bank.states.size()];
        const int delay=rng()%24;
        if(delay) advance(initial,w,nullptr,0,false,0,delay);
        if(initial.done()) continue;
        std::vector<double> rewards(paths);
        std::vector<std::vector<CurriculumRow>> samples(paths);
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores,logits;
        std::uniform_real_distribution<double> uniform(.0000001,.9999999);
        initial.observe(obs.data());
        const int floor=int(obs[3]);
        for(int path=0;path<paths;++path) {
            Env e=initial;
            int decisions=0;
            bool encountered=false;
            double near=-1;
            auto &rows=samples[path];
            for(int k=0;k<180 && decisions<window && !e.done();++k) {
                e.observe(obs.data());e.mask(mask.data());
                View v(obs.data(),w);
                encountered|=int(obs[4])==4 && obs[7] && v.enemy(0)[0]==int(MonsterId::CORRUPT_HEART)+1 && v.enemy(0)[6];
                if(near<0 && int(obs[3])>floor) near=obs[0]/std::max(1.f,obs[1]);
                v.base_scores(mask.data(),scores.data());
                double top=*std::max_element(scores.begin(),scores.end());
                int choice=-1;double maxDraw=-1e20,maxLogit=-1e20;
                size_t start=rows.size();
                for(int a=0;a<N_ACTIONS;++a) if(mask[a] && scores[a]>=top-7*CURRICULUM_TEMP) {
                    double gap=(scores[a]-top)/CURRICULUM_TEMP;
                    auto c=v.candidate(a,gap);
                    double logit=gap+(curriculum_scale?curriculum_scale*v.curriculum_delta(c):0.);
                    logits[a]=logit;
                    double draw=logit-std::log(-std::log(uniform(rng)));
                    if(draw>maxDraw) {maxDraw=draw;choice=a;}
                    maxLogit=std::max(maxLogit,logit);
                    CurriculumRow row{};
                    row[0]=decisions;row[1]=c.item;row[2]=c.foe;row[3]=float(gap);row[4]=a;
                    std::copy(c.features.begin(),c.features.end(),row.begin()+7);
                    rows.push_back(row);
                }
                if(rows.size()==start+1) rows.resize(start);
                else {
                    double sum=0;
                    for(size_t z=start;z<rows.size();++z) sum+=std::exp(logits[int(rows[z][4])]-maxLogit);
                    double chosenLogp=logits[choice]-maxLogit-std::log(sum);
                    for(size_t z=start;z<rows.size();++z) {
                        rows[z][4]=int(rows[z][4])==choice;rows[z][5]=float(chosenLogp);
                    }
                    ++decisions;
                }
                if(choice<0 || !e.step(choice)) throw std::runtime_error("invalid sampled move");
            }
            e.observe(obs.data());
            if(near<0) near=e.done()?0:obs[0]/std::max(1.f,obs[1]);
            auto r=advance(e,w);
            encountered|=r.encounter;
            // The HP term gives short windows feedback when every branch dies on the same floor.
            rewards[path]=r.floor/12.+r.act+.15*r.keys+2.*(r.floor>=51)+4.*(r.act>=4)+12.*encountered+near;
            auto &s=stats[j];++s[0];s[1]+=r.act>=3;s[2]+=r.act>=4;s[3]+=encountered;s[4]+=r.heart;s[5]+=r.truncated;
        }
        double mean=std::accumulate(rewards.begin(),rewards.end(),0.)/paths;
        double var=0;for(double v:rewards) var+=(v-mean)*(v-mean)/paths;
        if(var<1e-10) continue;
        ++stats[j][6];
        for(int path=0;path<paths;++path) {
            for(auto &row:samples[path]) {
                row[0]+=float((j*paths+path)*window);
                row[6]=float((rewards[path]-mean)/(1.+std::sqrt(var)));
            }
            output[j].insert(output[j].end(),samples[path].begin(),samples[path].end());
        }
    }
    CurriculumStats total{};
    size_t length=0;
    for(int j=0;j<groups;++j) {
        length+=output[j].size();
        for(int i=0;i<int(total.size());++i) total[i]+=stats[j][i];
    }
    py::array_t<float> rows({py::ssize_t(length),py::ssize_t(CurriculumRow{}.size())});
    float *ptr=rows.mutable_data();
    for(auto &group:output) for(auto &row:group) ptr=std::copy(row.begin(),row.end(),ptr);
    return {std::move(rows),total};
}
