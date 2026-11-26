void remember_enemy(ParticleBucket &bucket,const Env &e,std::mt19937 &rng) {
    int sample=std::uniform_int_distribution<int>(0,bucket.seen++)(rng);
    if(bucket.examples.size()<24) bucket.examples.push_back({e.bc.monsters,e.bc.encounter,e.bc.turn});
    else if(sample<24) bucket.examples[sample]={e.bc.monsters,e.bc.encounter,e.bc.turn};
}

void collect_beliefs(const Weights &w,uint64_t start,int count) {
    if(count<1) throw std::invalid_argument("positive belief collection size required");
    belief_bank.clear();late_particles.clear();practice_decks={};
    std::array<int,3> seen_decks{};
    std::mt19937 deck_rng(20703);
    const int old=belief_draws;
    belief_draws=0;
    std::mt19937 selector(190215);
    for(int i=0;i<count;++i) {
        Env e;e.record=false;e.reset(start+i,20,true,CharacterClass::IRONCLAD);
        int floor=-1,turn=-1;std::array<bool,3> saved{};
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        for(int k=0;k<3500 && !e.done();++k) {
            e.observe(obs.data());e.mask(mask.data());
            if(e.inBattle && mask[ACT_END_TURN] && (floor!=e.gc.floorNum || turn!=e.bc.turn)) {
                floor=e.gc.floorNum;turn=e.bc.turn;
                auto &bucket=belief_bank[public_enemy_key(obs.data())];
                remember_enemy(bucket,e,selector);
                const int act=e.gc.act-1;
                if(act>=0 && act<3 && !saved[act] && floor>=std::array<int,3>{12,28,44}[act] && obs[22]<MAX_PILE-1) {
                    saved[act]=true;
                    int j=std::uniform_int_distribution<int>(0,seen_decks[act]++)(deck_rng);
                    View seen(obs.data(),w);
                    auto profile=scenario_key(seen,Encounter::JAW_WORM);
                    if(practice_decks[act].size()<48) practice_decks[act].push_back(std::move(profile));
                    else if(j<48) practice_decks[act][j]=std::move(profile);
                }
            }
            // Keep the enemy bank fixed when evaluating a learned search student.
            View v(obs.data(),w);v.base_scores(mask.data(),scores.data());
            if(!e.step(int(std::max_element(scores.begin(),scores.end())-scores.begin())))
                throw std::runtime_error("invalid practice action while building particles");
        }
    }
    belief_draws=old;
}

std::array<int,3> enrich_late_particles(const Weights &w,int repetitions) {
    if(repetitions<1 || practice_decks[1].empty()) throw std::invalid_argument("collect natural practice decks first");
    late_particles.clear();
    const std::vector<std::pair<int,Encounter>> opponents={
        {2,Encounter::AUTOMATON},{2,Encounter::COLLECTOR},{2,Encounter::CHAMP},{2,Encounter::SLAVERS},
        {2,Encounter::BOOK_OF_STABBING},{3,Encounter::AWAKENED_ONE},{3,Encounter::DONU_AND_DECA},
        {3,Encounter::TIME_EATER},{3,Encounter::GIANT_HEAD},{3,Encounter::REPTOMANCER},
        {4,Encounter::SHIELD_AND_SPEAR}};
    std::mt19937 rng(201001);
    int states=0;
    for(auto [act,enc]:opponents) for(int i=0;i<repetitions;++i) {
        auto &bank=practice_decks[(act>=3 && !practice_decks[2].empty())?2:1];
        auto key=bank[rng()%bank.size()];
        key[0]=int(enc);key[1]=act;
        Room room=(enc==Encounter::SHIELD_AND_SPEAR || enc==Encounter::GIANT_HEAD || enc==Encounter::REPTOMANCER ||
                   enc==Encounter::SLAVERS || enc==Encounter::BOOK_OF_STABBING)?Room::ELITE:Room::BOSS;
        auto e=scenario_start(key,90451+i*127+int(enc),key[2]*(60+20*(i%3))/100,-1,room);
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        int seen=-1;
        for(int k=0;k<380 && e.inBattle && !e.done() && e.bc.outcome==Outcome::UNDECIDED;++k) {
            e.observe(obs.data());e.mask(mask.data());
            if(mask[ACT_END_TURN] && seen!=e.bc.turn) {
                remember_enemy(late_particles[public_enemy_key(obs.data())],e,rng);
                seen=e.bc.turn;++states;
            }
            View v(obs.data(),w);v.base_scores(mask.data(),scores.data());
            if(!e.step(int(std::max_element(scores.begin(),scores.end())-scores.begin())))
                throw std::runtime_error("invalid synthetic fight action");
        }
    }
    return {int(opponents.size())*repetitions,states,int(late_particles.size())};
}

// These status values are visible; the independently sampled particle supplies the
// unobserved move history. Match special powers exactly instead of guessing their counters.
void match_common_debuffs(Monster &m,const float *s) {
    auto value=[&](MS type){return int(s[MONSTER_BASE+int(type)]);};
    m.strength=value(MS::STRENGTH);
    m.vulnerable=value(MS::VULNERABLE);m.setHasStatus<MS::VULNERABLE>(m.vulnerable!=0);
    m.weak=value(MS::WEAK);m.setHasStatus<MS::WEAK>(m.weak!=0);
    m.artifact=value(MS::ARTIFACT);m.setHasStatus<MS::ARTIFACT>(m.artifact!=0);
}
void match_player_powers(Player &p,const float *shown) {
    p.statusBits0=0;p.statusBits1=0;p.statusMap.clear();p.statusSeq.fill(0);p.statusSeqNext=0;
    p.justAppliedBits=0;
    p.artifact=int(shown[int(PS::ARTIFACT)]);p.dexterity=int(shown[int(PS::DEXTERITY)]);
    p.strength=int(shown[int(PS::STRENGTH)]);p.focus=int(shown[int(PS::FOCUS)]);
    for(int i=1;i<N_PLAYER_STATUS;++i) if(shown[i] && i!=int(PS::ARTIFACT) &&
           i!=int(PS::DEXTERITY) && i!=int(PS::STRENGTH) && i!=int(PS::FOCUS)) {
        if(i<64) p.statusBits0|=uint64_t(1)<<i;
        else p.statusBits1|=uint32_t(1)<<(i-64);
        p.statusMap[PS(i)]=int(shown[i]);p.statusSeq[i]=++p.statusSeqNext;
    }
}
#include "stasis_particles.inc"

bool make_particle(Env &e,const View &v,const EnemyParticle &sample,int draw,const uint8_t *legal) {
    const auto *o=v.o;
    const auto room=Room(int(o[8])-1);
    auto key=scenario_key(v,sample.encounter);
    e=scenario_start(key,6203+draw,int(o[0]),int(o[3]),room,int(o[2]));
    if(e.bc.outcome!=Outcome::UNDECIDED || e.bc.inputState!=InputState::PLAYER_NORMAL) return false;
    auto &bc=e.bc;auto &p=bc.player;
    bc.turn=int(v.p[4]);bc.monsters=sample.monsters;
    std::vector<int> stasis_slots;
    for(int j=0;j<bc.monsters.monsterCount;++j) {
        const auto s=v.enemy(j);
        auto &m=bc.monsters.arr[j];
        if(s[MONSTER_BASE+int(MS::STASIS)]) {
            if(!reconstruct_stasis || m.id!=MonsterId::BRONZE_ORB || sample.encounter!=Encounter::AUTOMATON) return false;
            m.setHasStatus<MS::STASIS>(true);
            m.miscInfo=1;
            if(!s[4]) stasis_slots.push_back(std::min(1,j));
        }
        m.curHp=int(s[1]);m.maxHp=int(s[2]);m.block=int(s[3]);
        match_common_debuffs(m,s);
    }
    p.curHp=int(v.p[0]);p.maxHp=int(v.p[1]);p.block=int(v.p[2]);p.energy=int(v.p[3]);
    p.bloodied=2*p.curHp<=p.maxHp;
    if(v.p[N_PLAYER_BASE+int(PS::THE_BOMB)]) return false;
    match_player_powers(p,v.p+N_PLAYER_BASE);
    for(int i=0;i<MAX_RELICS && o[OBS_RELICS+i*RELIC_W];++i) {
        auto rel=o+OBS_RELICS+i*RELIC_W;
        int counter=int(rel[1])-1;
        if(counter<0) continue;
        switch(R(int(rel[0])-1)) {
        case R::PEN_NIB:p.penNibCounter=counter;break;
        case R::HAPPY_FLOWER:p.happyFlowerCounter=counter;break;
        case R::INCENSE_BURNER:p.incenseBurnerCounter=counter;break;
        case R::INK_BOTTLE:p.inkBottleCounter=counter;break;
        case R::NUNCHAKU:p.nunchakuCounter=counter;break;
        case R::SUNDIAL:p.sundialCounter=counter;break;
        case R::KUNAI: case R::SHURIKEN:p.attacksPlayedThisTurn=counter;break;
        case R::LETTER_OPENER:p.skillsPlayedThisTurn=counter;break;
        case R::VELVET_CHOKER:case R::POCKETWATCH:p.cardsPlayedThisTurn=counter;break;
        default:break;
        }
    }
    auto &cards=bc.cards;cards=CardManager();cards.nextUniqueCardId=e.gc.deck.cards.size();
    for(int j=0;j<int(v.p[5]);++j) {
        auto h=o+OBS_HAND+j*HAND_W;
        CardInstance c(C(int(h[0])-1),h[1]!=0);
        c.costForTurn=int(h[2]);c.retain=o[OBS_CLASS+CLASS_RETAIN+j]!=0;
        cards.createTempCardInHand(c);
    }
    if(int(v.p[6])>MAX_PILE) return false;
    std::vector<int> draw_cards;
    for(int j=0;j<MAX_PILE;++j) {
        auto d=o+OBS_DRAW+j*CARD_W;
        if(!d[0]) break;
        draw_cards.push_back(int(d[0])*256+int(d[1]));
    }
    // Canonicalize first so a reorder of the unobserved draw-pile list can never change a play.
    std::sort(draw_cards.begin(),draw_cards.end());
    std::mt19937 rng(18479u+23917u*draw);
    std::shuffle(draw_cards.begin(),draw_cards.end(),rng);
    for(auto id:draw_cards) cards.createTempCardInDrawPile(cards.drawPile.size(),{C(id/256-1),bool(id%256)});
    for(int j=0;j<MAX_PILE;++j) {
        auto c=o+OBS_DISCARD+j*CARD_W;
        if(!c[0]) break;
        cards.createTempCardInDiscard({C(int(c[0])-1),c[1]!=0});
    }
    for(int j=0;j<MAX_PILE;++j) {
        auto h=o+OBS_EXHAUST+j*CARD_W;
        if(!h[0]) break;
        CardInstance c(C(int(h[0])-1),h[1]!=0);
        c.uniqueId=cards.nextUniqueCardId++;
        cards.exhaustPile.push_back(c);
    }
    std::array<float,OBS_SIZE> check;
    std::array<uint8_t,N_ACTIONS> mask;
    if(!stasis_slots.empty()) {
        auto missing=missing_stasis_cards(o,int(stasis_slots.size()));
        if(missing.empty()) return false;
        if(missing.size()==2 && draw%2) std::swap(missing[0],missing[1]);
        for(size_t i=0;i<missing.size();++i) {
            CardInstance c(C(missing[i]/2-1),missing[i]%2!=0);
            c.uniqueId=cards.nextUniqueCardId++;
            cards.stasisCards[stasis_slots[i]]=c;
        }
    }
    e.observe(check.data());e.mask(mask.data());
    return std::equal(legal,legal+N_ACTIONS,mask.begin()) &&
           std::equal(o+OBS_PLAYER,o+OBS_DRAW,check.begin()+OBS_PLAYER) &&
           std::equal(o+OBS_RELICS,o+OBS_RELICS+MAX_RELICS*RELIC_W,check.begin()+OBS_RELICS);
}

#include "public_continuation.inc"
#include "public_tree.inc"
#include "terminal_search.inc"

void View::belief_scores(const uint8_t *mask,double *out) const {
    if(!o[7] || !mask[ACT_END_TURN] || belief_branch || opening_branch || simulated_tactic>=0) return;
    auto room=Room(int(o[8])-1);
    if(belief_scope==0 && room!=Room::BOSS && room!=Room::ELITE) return;
    if(belief_scope==1 && room!=Room::BOSS && room!=Room::ELITE && o[4]==1) return;
    if(!search_zero_energy && (p[3]<1 || p[5]<1)) return;
    int prior=int(std::max_element(out,out+N_ACTIONS)-out);
    if(prior>=ACT_DISCARD && prior!=ACT_END_TURN) return;
    auto actions=distinct_combat_actions(prior,o,mask);
    if(actions.size()<2) return;
    if(belief_audit) ++(*belief_audit)[0];
    auto identity=public_enemy_key(o);
    auto it=belief_bank.find(identity);
    auto extra=use_late_particles?late_particles.find(identity):late_particles.end();
    if(it==belief_bank.end() && extra==late_particles.end()) return;
    if(belief_audit) ++(*belief_audit)[1];
    struct Guard {Guard(){belief_branch=true;++rollout_generation;} ~Guard(){belief_branch=false;}} guard;
    std::vector<const EnemyParticle*> order;
    if(it!=belief_bank.end()) for(const auto &s:it->second.examples) order.push_back(&s);
    if(extra!=late_particles.end()) for(const auto &s:extra->second.examples) order.push_back(&s);
    auto distance=[&](const EnemyParticle *s){
        double d=std::abs(s->turn-int(p[4]));
        if(use_late_particles) for(int i=0;i<s->monsters.monsterCount;++i) {
            auto &m=s->monsters.arr[i];
            d+=.5*std::abs(double(m.curHp)/std::max(1,m.maxHp)-enemy(i)[1]/std::max(1.f,enemy(i)[2]));
        }
        return d;
    };
    std::stable_sort(order.begin(),order.end(),[&](const auto *a,const auto *b){return distance(a)<distance(b);});
    std::stable_sort(actions.begin()+1,actions.end(),[&](int a,int b){return out[a]>out[b];});
    const size_t limit=belief_action_limit?belief_action_limit:(belief_plans>1?9:5);
    if(actions.size()>limit) actions.resize(limit);
    std::vector<std::array<double,3>> totals(actions.size());
    std::vector<Env> next_play_roots;
    if(next_play_width || public_tree_budget || terminal_budget) next_play_roots.reserve(belief_draws);
    int accepted=0;
    const int attempts=std::min<int>(order.size()*belief_draws,8*belief_draws);
    for(int d=0;d<attempts && accepted<belief_draws;++d) {
        Env e;
        if(!make_particle(e,*this,*order[d%order.size()],d,mask)) continue;
        if(public_tree_budget || terminal_budget) {
            next_play_roots.push_back(std::move(e));++accepted;continue;
        }
        for(size_t j=0;j<actions.size();++j) {
            for(int t=0;t<belief_plans;++t) {
                int tactic=t==0?-1:t==1?1:2;
                auto result=finish_combat_action(e,w,actions[j],tactic);
                totals[j][t]+=fight_utility(result,o);
            }
        }
        if(next_play_width) next_play_roots.push_back(std::move(e));
        ++accepted;
    }
    if(!accepted) return;
    // Select a continuation tactic after averaging its outcome over all particles.
    // Picking it separately for each hidden draw would overvalue clairvoyant plans.
    auto best_plan=[&](size_t j){return *std::max_element(totals[j].begin(),totals[j].begin()+belief_plans);};
    if(terminal_budget) terminal_search_scores(actions,next_play_roots,out);
    else if(public_tree_budget) public_tree_scores(actions,next_play_roots,w,out);
    else {
        double base=best_plan(0);
        for(size_t j=1;j<actions.size();++j) out[actions[j]]+=belief_scale*(best_plan(j)-base)/accepted;
        if(next_play_width) refine_first_plays(actions,next_play_roots,w,out,next_play_width);
    }
    if(belief_audit) {
        ++(*belief_audit)[2];(*belief_audit)[3]+=accepted;
        (*belief_audit)[4]+=prior!=int(std::max_element(out,out+N_ACTIONS)-out);
    }
}
