// Record the public search's advice, including lower-ranked legal moves. A
// zero correction on an unsearched move reproduces the teacher's deployed behavior.
using DistillRow=std::array<float,11+CURRICULUM_INPUT>;
auto belief_examples(const Weights &w,uint64_t start,int n,int threads) {
    if(n<1 || threads<1 || !belief_draws) throw std::invalid_argument("a trained particle bank is required");
    std::vector<std::vector<DistillRow>> episodes(n);
    std::vector<std::array<int,3>> counts(n);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
    for(int i=0;i<n;++i) {
        Env e;e.record=false;e.reset(start+i,20,true,CharacterClass::IRONCLAD);
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores,prior;
        std::mt19937 rng(static_cast<uint32_t>(start+i));
        for(int step=0;step<3500 && !e.done();++step) {
            e.observe(obs.data());e.mask(mask.data());
            View view(obs.data(),w);
            view.base_scores(mask.data(),prior.data());scores=prior;
            int parent=int(std::max_element(prior.begin(),prior.end())-prior.begin());
            std::array<int,5> audit{};
            belief_audit=&audit;
            view.belief_scores(mask.data(),scores.data());
            belief_audit=nullptr;
            int chosen=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            if(audit[2]) {
                auto &s=counts[i];++s[0];s[1]+=parent!=chosen;
                // Keep a proportional sample, independent of whether the teacher changed the action.
                if(rng()%3==0) {
                    auto actions=distinct_combat_actions(parent,obs.data(),mask.data());
                    for(int a:actions) {
                        auto c=view.candidate(a,std::max(-10.,(prior[a]-prior[parent])/CURRICULUM_TEMP));
                        DistillRow r{};
                        r[0]=i;r[1]=step;r[2]=a;r[3]=c.item;r[4]=c.foe;
                        r[5]=float((scores[a]-prior[a])/CURRICULUM_TEMP);r[6]=float(a==chosen);
                        r[7]=float((prior[a]-prior[parent])/CURRICULUM_TEMP);
                        r[8]=float(parent!=chosen);
                        std::copy(c.features.begin(),c.features.end(),r.begin()+11);
                        episodes[i].push_back(r);
                    }
                    ++s[2];
                }
            }
            if(!e.step(chosen)) throw std::runtime_error("invalid public teacher action");
        }
    }
    std::array<int,3> total{};
    for(const auto &s:counts) for(int j=0;j<3;++j) total[j]+=s[j];
    return std::make_pair(pack_combat_examples(episodes),total);
}

// Match the candidate set and score units used for search labels. Outside the
// teacher's combat scope a single finite logit exactly reproduces the incumbent.
auto combat_student_priors(const Weights &weights,
        py::array_t<float,py::array::c_style> observations,
        py::array_t<uint8_t,py::array::c_style> masks,int threads) {
    if(observations.ndim()!=2 || observations.shape(1)!=OBS_SIZE || masks.ndim()!=2 ||
       masks.shape(0)!=observations.shape(0) || masks.shape(1)!=N_ACTIONS || threads<1)
        throw std::invalid_argument("invalid public combat batch");
    const int n=int(observations.shape(0));
    py::array_t<float> result({n,N_ACTIONS});
    float *output=result.mutable_data();
    const float *obs=observations.data();
    const uint8_t *legal=masks.data();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int i=0;i<n;++i) {
        const auto *o=obs+i*OBS_SIZE;
        const auto *mask=legal+i*N_ACTIONS;
        float *row=output+i*N_ACTIONS;
        std::fill(row,row+N_ACTIONS,-1e8f);
        std::array<double,N_ACTIONS> scores;
        View view(o,weights);
        view.base_scores(mask,scores.data());
        const int parent=int(std::max_element(scores.begin(),scores.end())-scores.begin());
        row[parent]=0;
        const auto room=Room(int(o[8])-1);
        if(!o[7] || !mask[ACT_END_TURN] || view.p[3]<1 || view.p[5]<1 ||
           (parent>=ACT_DISCARD && parent!=ACT_END_TURN) ||
           (o[4]==1 && room!=Room::BOSS && room!=Room::ELITE)) continue;
        for(int a:distinct_combat_actions(parent,o,mask))
            row[a]=float((scores[a]-scores[parent])/CURRICULUM_TEMP);
    }
    return result;
}
