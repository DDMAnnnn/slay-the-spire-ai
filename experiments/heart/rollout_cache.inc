// Cache decisions only inside one search at a frozen public root. The cheap
// fingerprint locates a slot; equality of the entire observation and mask is required for a hit.
struct RolloutEntry {
    uint64_t generation;
    std::array<float,OBS_SIZE> observation;
    std::array<uint8_t,N_ACTIONS> mask;
    std::array<int,3> action;
};
using RolloutCache=std::array<RolloutEntry,256>;
static bool cache_rollout_actions=false;
static thread_local std::unique_ptr<RolloutCache> rollout_cache;
static thread_local uint64_t rollout_generation=0;
static thread_local std::array<int64_t,2>* rollout_cache_audit=nullptr;

int continuation_action(const View &view,const uint8_t *mask,std::array<double,N_ACTIONS> &scores,int tactic) {
    auto greedy=[&](const auto &q){return int(std::max_element(q.begin(),q.end())-q.begin());};
    auto alternative=[&](int id,auto &q){
        if(id>=0 && mask[ACT_END_TURN]) {
            int primary=0;
            for(int i=0;i<5;++i) if(view.enemy(i)[6]) {primary=i;break;}
            view.apply_tactics(mask,q.data(),primary,scenario_menu[id]);
        }
        return greedy(q);
    };
    bool cache=cache_rollout_actions && belief_branch && simulated_tactic<0 &&
               (tactic==-1 || tactic==1 || tactic==2);
    if(!cache) {
        view.all_scores(mask,scores.data());
        return alternative(tactic,scores);
    }
    if(!rollout_cache) rollout_cache=std::make_unique<RolloutCache>();
    const auto *o=view.o;
    uint32_t hash=2166136261u;
    auto add=[&](float v){hash=(hash^uint32_t(int(v)))*16777619u;};
    add(view.p[0]);add(view.p[2]);add(view.p[3]);add(view.p[4]);add(view.p[5]);
    for(int i=0;i<5;++i) {add(view.enemy(i)[1]);add(view.enemy(i)[3]);}
    for(int i=0;i<int(view.p[5]) && i<10;++i) {add(o[OBS_HAND+i*HAND_W]);add(o[OBS_HAND+i*HAND_W+2]);}
    auto &entry=(*rollout_cache)[(hash^(hash>>16))%rollout_cache->size()];
    if(entry.generation==rollout_generation &&
       !std::memcmp(entry.observation.data(),o,sizeof(float)*OBS_SIZE) &&
       !std::memcmp(entry.mask.data(),mask,sizeof(uint8_t)*N_ACTIONS)) {
        if(rollout_cache_audit) ++(*rollout_cache_audit)[0];
        return entry.action[tactic<0?0:tactic];
    }
    if(rollout_cache_audit) ++(*rollout_cache_audit)[1];
    view.all_scores(mask,scores.data());
    entry.generation=rollout_generation;
    std::copy_n(o,OBS_SIZE,entry.observation.begin());
    std::copy_n(mask,N_ACTIONS,entry.mask.begin());
    entry.action.fill(greedy(scores));
    if(belief_plans>1 && mask[ACT_END_TURN]) for(int id=1;id<belief_plans;++id) {
        auto adjusted=scores;
        entry.action[id]=alternative(id,adjusted);
    }
    return entry.action[tactic<0?0:tactic];
}
