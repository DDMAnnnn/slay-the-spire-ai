// Run separate bounded action trees on synthetic battles reconstructed from public fields.
// Each root action has to work across worlds. Later actions may differ in each world.
struct TerminalNode {
    search::Action action;
    int visits=0;
    double total=0., best=-1.;
    std::vector<TerminalNode> children;
};

search::Action terminal_root_action(int a) {
    using T=search::ActionType;
    if(a==ACT_END_TURN) return search::Action(T::END_TURN);
    if(a>=ACT_POTION) return search::Action(T::POTION,(a-ACT_POTION)/5,(a-ACT_POTION)%5);
    return search::Action(T::CARD,(a-ACT_PLAY)/5,(a-ACT_PLAY)%5);
}

double terminal_value(const BattleContext &s,int initial_potions) {
    int live=0,maximum=0;
    for(int i=0;i<s.monsters.monsterCount;++i) {
        live+=std::max(0,int(s.monsters.arr[i].curHp));
        maximum+=int(s.monsters.arr[i].maxHp);
    }
    const double kept=std::min(0,int(s.potionCount)-initial_potions)*.025;
    if(s.outcome==Outcome::PLAYER_VICTORY)
        return .65+.35*s.player.curHp/std::max(1,int(s.player.maxHp))+kept;
    return .25*(1.-double(live)/std::max(1,maximum))+kept;
}

void terminal_choices(search::BattleScumSearcher2 &enumerator,const BattleContext &s,
                      std::vector<search::Action> &choices) {
    choices.clear();
    if(s.inputState==InputState::CARD_SELECT) {
        choices=search::Action::enumerateCardSelectActions(s);
    } else if(s.inputState==InputState::PLAYER_NORMAL) {
        search::BattleScumSearcher2::Node node;
        enumerator.enumerateActionsForNode(node,s);
        for(auto &e:node.edges) {
            auto a=e.action;
            if(a.getActionType()==search::ActionType::POTION && a.getTargetIdx()==8191) continue;
            choices.push_back(a);
        }
    }
    choices.erase(std::remove_if(choices.begin(),choices.end(),[&](auto a){return !a.isValidAction(s);}),choices.end());
}

void terminal_search_scores(const std::vector<int> &actions,const std::vector<Env> &roots,double *scores) {
    std::vector<double> pooled(actions.size());
    for(size_t world=0;world<roots.size();++world) {
        const BattleContext &initial=roots[world].bc;
        search::BattleScumSearcher2 enumerator(initial);
        std::mt19937 rng(525351u+unsigned(world)*199933u);
        TerminalNode root;
        for(int a:actions) root.children.push_back({terminal_root_action(a)});
        std::vector<search::Action> options;
        const int sims=std::max(terminal_budget,int(actions.size())*2);
        for(int i=0;i<sims;++i) {
            BattleContext state=initial;
            TerminalNode *node=&root;
            std::vector<TerminalNode*> path{node};
            int depth=0;
            while(!node->children.empty() && depth<96 && state.outcome==Outcome::UNDECIDED) {
                auto &children=node->children;
                auto untried=std::find_if(children.begin(),children.end(),[](const auto &c){return !c.visits;});
                if(untried!=children.end()) node=&*untried;
                else {
                    const double explore=.7*std::sqrt(std::log(double(node->visits)+1.));
                    auto ucb=[&](const auto &c){return c.total/c.visits+explore/std::sqrt(double(c.visits));};
                    node=&*std::max_element(children.begin(),children.end(),[&](const auto &a,const auto &b){return ucb(a)<ucb(b);});
                }
                if(!node->action.isValidAction(state)) throw std::runtime_error("illegal synthetic tree edge");
                node->action.execute(state);path.push_back(node);++depth;
            }
            if(node->visits && node->children.empty() && depth<96 && state.outcome==Outcome::UNDECIDED) {
                terminal_choices(enumerator,state,options);
                std::shuffle(options.begin(),options.end(),rng);
                for(auto a:options) node->children.push_back({a});
            }
            while(depth++<160 && state.outcome==Outcome::UNDECIDED) {
                terminal_choices(enumerator,state,options);
                if(options.empty()) break;
                const size_t pick=std::uniform_int_distribution<size_t>(0,options.size()-1)(rng);
                options[pick].execute(state);
            }
            const double value=terminal_value(state,initial.potionCount);
            for(auto *p:path) {++p->visits;p->total+=value;p->best=std::max(p->best,value);}
        }
        for(size_t a=0;a<actions.size();++a) {
            const auto &edge=root.children[a];
            pooled[a]+=(1.-terminal_optimism)*edge.total/edge.visits+terminal_optimism*edge.best;
        }
    }
    for(size_t a=0;a<actions.size();++a)
        scores[actions[a]]+=100.*belief_scale*(pooled[a]-pooled[0])/roots.size();
}
