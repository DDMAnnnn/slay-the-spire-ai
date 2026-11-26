// Slot priorities are shared between hypothetical decks. Adding a card inserts it at a
// uniformly random position while leaving the relative order of the existing cards intact.
void couple_opening_shuffle(GameContext &gc,int sample) {
    const int n=gc.deck.size();
    std::vector<int> permutation(n),order(n);
    std::iota(permutation.begin(),permutation.end(),0);
    std::iota(order.begin(),order.end(),0);
    Random shuffle(gc.seed+gc.floorNum);
    java::Collections::shuffle(permutation.begin(),permutation.end(),java::Random(shuffle.randomLong()));
    std::mt19937 rng(119771u+104729u*sample);
    std::vector<uint32_t> priorities(n);
    for(auto &p:priorities) p=rng();
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return priorities[a]<priorities[b];});
    auto original=gc.deck.cards;
    for(int i=0;i<n;++i) gc.deck.cards[permutation[i]]=original[order[i]];
}

void View::inventory_search(const uint8_t *mask,double *scores) const {
    if(o[7] || !inventory_draws || o[22]>=MAX_PILE-1 || o[4]>3) return;
    // These change the inventory on pickup or encode a particular bottled card.
    // The observation lacks enough information to reproduce their setup here.
    if(relic(R::BOTTLED_FLAME) || relic(R::BOTTLED_LIGHTNING) || relic(R::BOTTLED_TORNADO)) return;
    bool upgrade=ScreenState(int(o[6]))==ScreenState::CARD_SELECT && int(o[26])==int(CardSelectScreenType::UPGRADE);
    const int skip=draft_skip(mask);
    if(skip>=0 && (skip<ACT_CHOOSE || skip>=ACT_CONFIRM || !mask[skip]))
        throw std::logic_error("skip must be a legal reward action");
    if(skip<0 && (!upgrade || inventory_mode==0)) return;
    std::vector<int> candidates;
    for(int a=ACT_CHOOSE;a<ACT_CONFIRM;++a) if(mask[a]) {
        auto opt=o+OBS_CHOICES+(a-ACT_CHOOSE)*CHOICE_W;
        if(ChoiceKind(int(opt[0]))==ChoiceKind::CARD || a==skip) candidates.push_back(a);
    }
    if(candidates.size()<2) return;
    std::stable_sort(candidates.begin(),candidates.end(),[&](int a,int b){return scores[a]>scores[b];});
    if(candidates.size()>6) candidates.resize(6);
    // Include declining an offer even when all card priors are favorable.
    if(skip>=0 && std::find(candidates.begin(),candidates.end(),skip)==candidates.end()) candidates.back()=skip;
    const int act=int(o[4])-1;
    const Encounter fights[3][3]={
        {Encounter::JAW_WORM,Encounter::GREMLIN_NOB,Encounter::THREE_SENTRIES},
        {Encounter::SNAKE_PLANT,Encounter::SLAVERS,Encounter::BOOK_OF_STABBING},
        {Encounter::SPIRE_GROWTH,Encounter::GIANT_HEAD,Encounter::REPTOMANCER}};
    const Encounter bosses[3]={Encounter::THE_GUARDIAN,Encounter::CHAMP,Encounter::TIME_EATER};
    std::vector<double> totals(candidates.size());
    struct Guard {
        int old=simulated_tactic;
        Guard(){simulated_tactic=0;}
        ~Guard(){simulated_tactic=old;}
    } guard;
    for(int s=0;s<inventory_draws;++s) {
        bool boss=s%3==0;
        int nonboss=s%3==0?0:(s-s/3-1)%3;
        auto enc=boss?(visible_draft_boss()>=0?Encounter(int(o[15])-1):bosses[act]):fights[act][nonboss];
        auto room=boss?Room::BOSS:nonboss==0?Room::MONSTER:Room::ELITE;
        auto key=scenario_key(*this,enc);
        int sample=9131+s+17*int(o[3]);
        int hp=std::clamp(int(o[0])+(boss?int(.2*o[1]):0),1,int(o[1]));
        auto div=std::find(key.begin()+10,key.end(),-1)-key.begin();
        for(size_t i=0;i<candidates.size();++i) {
            auto option=key;
            int a=candidates[i];
            if(a!=skip) {
                auto c=o+OBS_CHOICES+(a-ACT_CHOOSE)*CHOICE_W;
                int token=int(c[1])*256+int(c[2]);
                if(upgrade) {
                    auto it=std::find(option.begin()+10,option.begin()+div,token);
                    if(it==option.begin()+div) return;
                    ++*it;
                } else option.insert(option.begin()+div,token);
            }
            auto state=scenario_start(option,sample,hp,int(o[3]),room,int(o[2]),true);
            if(state.bc.outcome==Outcome::PLAYER_VICTORY) {totals[i]+=1.;continue;}
            if(state.bc.outcome!=Outcome::UNDECIDED) continue;
            std::array<float,OBS_SIZE> obs;
            std::array<uint8_t,N_ACTIONS> am;
            std::array<double,N_ACTIONS> sc;
            state.observe(obs.data());state.mask(am.data());
            View v(obs.data(),w);v.all_scores(am.data(),sc.data());
            int first=int(std::max_element(sc.begin(),sc.end())-sc.begin());
            auto r=finish_combat_action(state,w,first);
            totals[i]+=r.win?.60+.4*r.health/o[1]-.025*r.potions:.25*(1.-r.remaining)-.025*r.potions;
        }
    }
    for(size_t i=1;i<candidates.size();++i)
        scores[candidates[i]]+=inventory_search_scale*(totals[i]-totals[0])/inventory_draws;
}
