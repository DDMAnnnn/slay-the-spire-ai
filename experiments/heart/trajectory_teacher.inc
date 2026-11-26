// Search happens only while collecting labels. Inference has no Env or future draws.
struct PracticePlan {
    std::vector<int> actions;
    double value=-1e9;
    bool won=false;
};

PracticePlan search_fight(Env state,const Weights &w,int variant,std::mt19937 &rng) {
    PracticePlan plan;
    std::array<float,OBS_SIZE> obs;
    std::array<uint8_t,N_ACTIONS> mask;
    std::array<double,N_ACTIONS> scores;
    // The same card-type and target preferences persist across a trajectory.
    // This can discover a whole sequence of blocks or a change of focus target.
    std::normal_distribution<double> noise(0.,1.);
    double scale=variant?3.+(variant%7)*4.:0.;
    std::array<double,8> bias{};
    for(auto &x:bias) x=noise(rng)*scale;
    for(int step=0;step<280 && state.inBattle && !state.done();++step) {
        state.observe(obs.data()); state.mask(mask.data());
        View v(obs.data(),w); v.all_scores(mask.data(),scores.data());
        if(variant && mask[ACT_END_TURN]) {
            for(int a=0;a<=ACT_END_TURN;++a) if(mask[a]) {
                if(a<ACT_POTION) {
                    auto h=obs.data()+OBS_HAND+(a/5)*HAND_W;
                    auto type=getCardType(C(int(h[0])-1));
                    scores[a]+=bias[type==CardType::ATTACK?0:type==CardType::POWER?1:2];
                    if(h[4]) {
                        auto m=v.enemy(a%5);
                        scores[a]+=bias[3]*m[1]/std::max(1.f,m[2])+bias[4]*m[8]*m[9]/25.;
                    }
                } else if(a<ACT_DISCARD) scores[a]+=bias[5];
                else if(a==ACT_END_TURN) scores[a]+=bias[6];
                scores[a]+=scale*.35*noise(rng);
            }
        }
        int a=int(std::max_element(scores.begin(),scores.end())-scores.begin());
        if(!state.step(a)) throw std::runtime_error("invalid search trajectory");
        plan.actions.push_back(a);
    }
    plan.won=state.bc.outcome==Outcome::PLAYER_VICTORY;
    if(state.bc.outcome!=Outcome::UNDECIDED) {
        // Upstream normalizes winning returns on a different scale than losses.
        double utility=search::BattleScumSearcher2::evaluateEndState(state.bc);
        plan.value=plan.won?utility/100.:std::clamp(utility-35.,-80.,-1.);
    }
    return plan;
}

using TeacherRow=std::array<float,58>;
using TeacherStats=std::array<int,5>;
std::pair<std::vector<TeacherRow>,TeacherStats> trajectory_practice(
        const Weights &w,uint64_t start,int n,int paths,int threads) {
    if(n<1 || paths<2 || threads<1) throw std::invalid_argument("invalid teacher practice");
    std::vector<std::vector<TeacherRow>> episodes(n);
    std::vector<TeacherStats> counts(n);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,2)
    for(int j=0;j<n;++j) {
        Env e; e.record=false; e.reset(start+j,20,true,CharacterClass::IRONCLAD);
        std::mt19937 rng(static_cast<unsigned>(start+j));
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        int last_floor=-1;
        for(int k=0;k<3400 && !e.done();++k) {
            e.observe(obs.data()); e.mask(mask.data());
            View v(obs.data(),w); v.all_scores(mask.data(),scores.data());
            int best=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            if(obs[4]>2) break;
            if(mask[ACT_END_TURN] && last_floor!=int(obs[3])) {
                last_floor=int(obs[3]);
                auto room=Room(int(obs[8])-1);
                if(room==Room::BOSS || room==Room::ELITE || rng()%4==0) {
                    auto parent=search_fight(e,w,0,rng),winner=parent;
                    for(int z=1;z<paths;++z) {
                        auto candidate=search_fight(e,w,z,rng);
                        if(candidate.value>winner.value) winner=std::move(candidate);
                    }
                    auto &s=counts[j];
                    ++s[0]; s[1]+=parent.won; s[2]+=winner.won;
                    if(winner.value>parent.value+1) ++s[3];
                    Env practice=e;
                    for(int t=0;t<int(winner.actions.size());++t) {
                        practice.observe(obs.data());practice.mask(mask.data());
                        View seen(obs.data(),w);seen.all_scores(mask.data(),scores.data());
                        int chosen=winner.actions[t];
                        double top=*std::max_element(scores.begin(),scores.end());
                        if(mask[ACT_END_TURN] && (chosen<ACT_DISCARD || chosen==ACT_END_TURN) &&
                           std::count(mask.begin(),mask.begin()+ACT_DISCARD,uint8_t(1))>0) {
                            bool alternate=scores[chosen]+1e-8<top;
                            if(alternate || rng()%5==0) {
                                ++s[4];
                                for(int a=0;a<=ACT_END_TURN;++a) if(mask[a] && (a<ACT_DISCARD || a==ACT_END_TURN)) {
                                    TeacherRow row{};
                                    row[0]=j;row[1]=last_floor*300+t;row[2]=seen.actor_item(a);
                                    int target=seen.actor_target(a);
                                    row[3]=target<0?0:int(seen.enemy(target)[0]);
                                    for(int h=0;h<seen.p[5];++h) row[4+h]=obs[OBS_HAND+h*HAND_W];
                                    auto x=seen.actor_features(a,target);
                                    std::copy(x.begin(),x.end(),row.begin()+14);
                                    row[54]=a==chosen;row[55]=float(scores[a]-top);
                                    row[56]=float(winner.value-parent.value);row[57]=alternate;
                                    episodes[j].push_back(row);
                                }
                            }
                        }
                        if(!practice.step(chosen)) throw std::runtime_error("teacher replay mismatch");
                    }
                }
            }
            if(!e.step(best)) throw std::runtime_error("invalid teacher collection policy");
        }
    }
    TeacherStats total{};
    std::vector<TeacherRow> rows;
    for(int j=0;j<n;++j) {
        for(int k=0;k<5;++k) total[k]+=counts[j][k];
        rows.insert(rows.end(),episodes[j].begin(),episodes[j].end());
    }
    return {std::move(rows),total};
}
