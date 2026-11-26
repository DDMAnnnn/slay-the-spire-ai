// Each node selects actions from a public observation. Only simulation outcomes
// supply returns; the live game's Env, RNG and concealed draw order never enter.
struct CombatArm {
    int action;
    double prior;
    int visits=0;
    double total=0.;
};
struct CombatNode {
    std::vector<CombatArm> arms;
    int visits=0;
    double total=0.;
};

CombatNode public_node(const std::vector<int> &actions,const double *scores) {
    CombatNode n;
    const double best=scores[*std::max_element(actions.begin(),actions.end(),[&](int a,int b){return scores[a]<scores[b];})];
    double sum=0.;
    for(int a:actions) {
        double p=std::exp(std::max(-18.,(scores[a]-best)/12.));
        n.arms.push_back({a,p});sum+=p;
    }
    for(auto &arm:n.arms) arm.prior=.75*arm.prior/sum+.25/n.arms.size();
    return n;
}

size_t tree_pick(const CombatNode &node,bool root) {
    // Give every root play two common draws before allocating visits adaptively.
    if(root && node.visits<int(2*node.arms.size())) return node.visits%node.arms.size();
    double best=-1e9;
    size_t choice=0;
    for(size_t i=0;i<node.arms.size();++i) {
        const auto &arm=node.arms[i];
        double q=arm.visits?arm.total/arm.visits:(node.visits?node.total/node.visits:.45);
        double bound=q+1.4*arm.prior*std::sqrt(1.+node.visits)/(1.+arm.visits);
        if(bound>best) {best=bound;choice=i;}
    }
    return choice;
}

uint64_t visible_combat_hash(const float *o,const uint8_t *mask,int depth) {
    uint64_t hash=1469598103934665603ULL;
    auto add=[&](uint64_t v){hash=(hash^v)*1099511628211ULL;};
    add(static_cast<uint64_t>(depth));
    auto span=[&](int begin,int end){
        for(int i=begin;i<end;++i) if(o[i]!=0.f) {
            uint32_t bits;std::memcpy(&bits,o+i,sizeof(bits));add(uint64_t(i)<<32|bits);
        }
    };
    span(0,9);span(16,28);span(OBS_RELICS,OBS_MAP);span(OBS_PLAYER,OBS_SIZE);
    if(!mask[ACT_END_TURN]) span(OBS_CHOICES,OBS_DECK);
    for(int i=0;i<N_ACTIONS;++i) if(mask[i]) add(0xff00000000ULL+i);
    return hash;
}

double tree_return(bool won,float hp,float remaining,int max_hp,float potions) {
    return (won?.55+.45*hp/std::max(1,max_hp):.25*(1.-remaining))-.04*potions;
}

double finished_tree_value(const Env &e,int potions,int max_hp) {
    bool won=e.bc.outcome==Outcome::PLAYER_VICTORY;
    int alive=0,total=0;
    for(int i=0;i<e.bc.monsters.monsterCount;++i) {
        const auto &m=e.bc.monsters.arr[i];
        alive+=std::max(0,int(m.curHp));total+=int(m.maxHp);
    }
    return tree_return(won,won?e.bc.player.curHp:0,float(alive)/std::max(1,total),
                       max_hp,float(potions-e.bc.potionCount));
}

void public_tree_scores(const std::vector<int> &actions,const std::vector<Env> &roots,
                        const Weights &w,double *scores) {
    CombatNode root=public_node(actions,scores);
    std::unordered_map<uint64_t,CombatNode> branches;
    int expanded=0, followups=0;
    const int max_hp=roots.front().bc.player.maxHp;
    const int n=std::max(public_tree_budget,int(2*actions.size()));
    for(int i=0;i<n;++i) {
        size_t arm=tree_pick(root,true);
        // Each root action sees the same sequence of independent worlds by visit count.
        Env g=roots[root.arms[arm].visits%roots.size()];
        const int potions=g.bc.potionCount, floor=g.gc.floorNum;
        const auto encounter=g.bc.encounter;
        std::vector<std::pair<CombatNode*,size_t>> path{{&root,arm}};
        int action=root.arms[arm].action;
        double value=0.;
        for(int depth=1;;++depth) {
            if(!g.step(action)) throw std::runtime_error("invalid public tree action");
            if(g.inBattle && (g.gc.floorNum!=floor || g.bc.encounter!=encounter)) {
                value=tree_return(true,float(g.gc.curHp),0.f,max_hp,float(potions-g.gc.potionCount));break;
            }
            if(g.done() || !g.inBattle || g.bc.outcome!=Outcome::UNDECIDED) {
                value=finished_tree_value(g,potions,max_hp);break;
            }
            std::array<float,OBS_SIZE> obs;std::array<uint8_t,N_ACTIONS> mask;
            std::array<double,N_ACTIONS> prior;
            g.observe(obs.data());g.mask(mask.data());
            auto get_parent=[&](){
                View view(obs.data(),w);view.all_scores(mask.data(),prior.data());
                return int(std::max_element(prior.begin(),prior.end())-prior.begin());
            };
            if(depth>=public_tree_depth) {
                const int used=potions-g.bc.potionCount;
                int parent=get_parent();
                auto r=finish_combat_action(std::move(g),w,parent);
                value=tree_return(r.win,r.health,r.remaining,max_hp,used+r.potions);break;
            }
            uint64_t key=visible_combat_hash(obs.data(),mask.data(),depth);
            auto found=branches.find(key);
            if(found==branches.end()) {
                int parent=get_parent();
                std::vector<int> options;
                if(mask[ACT_END_TURN] && (parent<ACT_DISCARD || parent==ACT_END_TURN)) {
                    options=distinct_combat_actions(parent,obs.data(),mask.data());
                } else {
                    options.push_back(parent);
                    for(int a=0;a<N_ACTIONS;++a) if(mask[a] && a!=parent) options.push_back(a);
                }
                std::stable_sort(options.begin()+1,options.end(),[&](int a,int b){return prior[a]>prior[b];});
                if(options.size()>7) options.resize(7);
                found=branches.emplace(key,public_node(options,prior.data())).first;
            }
            auto &node=found->second;
            arm=tree_pick(node,false);
            action=node.arms[arm].action;
            path.push_back({&node,arm});
            if(node.visits) {++expanded;followups+=!mask[ACT_END_TURN];}
            else {
                const int used=potions-g.bc.potionCount;
                auto r=finish_combat_action(std::move(g),w,action);
                value=tree_return(r.win,r.health,r.remaining,max_hp,used+r.potions);break;
            }
        }
        for(auto [node,k]:path) {
            ++node->visits;node->total+=value;
            ++node->arms[k].visits;node->arms[k].total+=value;
        }
    }
    auto q=[&](const CombatArm &a){return (a.total+2*root.total/root.visits)/(a.visits+2);};
    const double reference=q(root.arms[0]);
    for(const auto &a:root.arms) scores[a.action]+=100.*belief_scale*(q(a)-reference);
    if(tree_audit) {
        ++(*tree_audit)[0];(*tree_audit)[1]+=n;(*tree_audit)[2]+=int(branches.size());
        (*tree_audit)[3]+=expanded;(*tree_audit)[4]+=followups;
    }
}
