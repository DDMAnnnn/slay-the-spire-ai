// Included inside View. Predict only the resolution of cards currently in hand.
// Draws, random choices and subsequent enemy turns are deliberately left unresolved.
struct Position {
    std::array<double,5> hp{}, vulnerable{};
    double block=0, strength=0;
    double hpChange=0, strengthGrowth=0, passiveBlock=0, temporaryStrength=0;
    std::array<double,5> weak{}, disarm{};
    int energy=0, used=0;
};
Position start_turn() const {
    Position s;
    s.block=p[2]; s.energy=int(p[3]); s.strength=p[N_PLAYER_BASE+int(PlayerStatus::STRENGTH)];
    for(int j=0;j<5;++j) {
        auto m=enemy(j);
        s.hp[j]=m[6]?m[1]+m[3]:0;
        s.vulnerable[j]=m[MONSTER_BASE+int(MonsterStatus::VULNERABLE)];
    }
    return s;
}
double resolve(Position &s,int a) const {
    int j=a/5,t=a%5;
    auto h=o+OBS_HAND+HAND_W*j;
    C id=C(int(h[0])-1);
    bool u=h[1]!=0;
    auto f=cardFacts[u][int(id)];
    int cost=std::max(0,int(h[2]));
    // X cards use the remaining energy.
    int repeats=id==C::WHIRLWIND?s.energy:std::max(1,int(f[0]));
    s.energy-=h[2]==-1?s.energy:cost;
    s.used|=1<<j;
    double raw=scaling_attack(id,u);
    if(id==C::BODY_SLAM) raw=s.block;
    if(id==C::PERFECTED_STRIKE) raw+=(u?3:2)*(num(id)+num(C::STRIKE_RED)+num(C::TWIN_STRIKE)+num(C::POMMEL_STRIKE)+num(C::WILD_STRIKE));
    if(id==C::FIEND_FIRE) repeats=std::max(1,int(p[5])-1);
    double damage=std::max(0.,raw+s.strength*(id==C::HEAVY_BLADE?(u?5:3):1));
    if(getCardType(id)!=CardType::ATTACK) damage=0;
    double gain=0;
    for(int k=0;k<5;++k) {
        if(s.hp[k]<=0 || (!f[1] && k!=t)) continue;
        auto m=enemy(k);
        double per=std::floor(damage*(s.vulnerable[k]>0?1.5:1.)*(p[N_PLAYER_BASE+int(PlayerStatus::WEAK)]?.75:1.));
        if(m[MONSTER_BASE+int(MonsterStatus::INTANGIBLE)]) per=std::min(per,1.);
        if(m[MONSTER_BASE+int(MonsterStatus::FLIGHT)]) per=std::floor(per*.5);
        double hit=std::min(s.hp[k],per*repeats);
        s.hp[k]-=hit;
        gain+=hit;
        if(hit && s.hp[k]==0) gain+=w[Kill]+w[Defense]*m[8]*m[9];
        if(id==C::FEED && hit && s.hp[k]==0) gain+=22;
        if(id==C::REAPER) gain+=std::min(double(o[1]-o[0]),hit)*1.5;
        if(!m[MONSTER_BASE+int(MonsterStatus::ARTIFACT)]) {
            if(f[32] && !s.vulnerable[k]) gain+=w[Debuff];
            s.vulnerable[k]+=f[32];
            if(f[33] && !m[MONSTER_BASE+int(MonsterStatus::WEAK)]) gain+=.25*m[8]*m[9]+3;
        }
    }
    double block=f[3];
    if(id==C::ENTRENCH) block=s.block;
    if(id==C::SECOND_WIND) block=std::max(1.,p[5]-2.)*(u?7:5);
    if(block) block=std::max(0.,block+p[N_PLAYER_BASE+int(PlayerStatus::DEXTERITY)]);
    if(p[N_PLAYER_BASE+int(PlayerStatus::FRAIL)]) block=std::floor(block*.75);
    double incoming_after=0;
    for(int k=0;k<5;++k) if(s.hp[k]>0) incoming_after+=enemy(k)[8]*enemy(k)[9];
    gain+=std::min(block,std::max(0.,incoming_after-s.block))*w[Defense];
    if (p[N_PLAYER_BASE+int(PlayerStatus::BARRICADE)]) gain+=block*.75;
    s.block+=block;
    if(id==C::INFLAME) s.strength+=u?3:2;
    if(id==C::SPOT_WEAKNESS && enemy(t)[8]>0) {s.strength+=u?4:3;gain+=12;}
    if(id==C::LIMIT_BREAK) {gain+=std::min(35.,s.strength*3); s.strength*=2;}
    if(id==C::FLEX) s.strength+=u?4:2;
    if(getCardType(id)==CardType::POWER) gain+=w[Scaling]*(9+card_prior(id));
    if(id==C::SHOCKWAVE || id==C::DISARM) gain+=14;
    if(id==C::APPARITION && incoming_after>s.block) gain+=45;
    if(id==C::OFFERING || id==C::APOTHEOSIS) gain+=28;
    if(id==C::ARMAMENTS && u && p[5]>2) gain+=9;
    if(id==C::FLAME_BARRIER && incoming_after>s.block-block) gain+=7;
    if(id==C::BITE && missing>0) gain+=4;
    if(id==C::RITUAL_DAGGER && s.hp[t]<=0) gain+=12;
    gain+=f[4]*w[Draw]*(s.energy>0?1.4:.35)+f[5]*3-f[6];
    s.energy=std::min(10,s.energy+int(f[5]));
    if(id==C::FIEND_FIRE) s.used=1023;
    if(getCardType(id)==CardType::SKILL && enemy(0)[MONSTER_BASE+int(MonsterStatus::ENRAGE)]) gain-=12;
    if(id==C::IMPERVIOUS && incoming_after-s.block+block<7) gain-=11;
    if(id==C::REAPER && !missing) gain-=5;
    if(forecast_settings[0]>0) {
        s.hpChange-=f[6];
        if(id==C::BITE) s.hpChange+=std::min(double(o[1]-o[0]),double(f[7]));
        if(id==C::DEMON_FORM) s.strengthGrowth+=u?3:2;
        if(id==C::METALLICIZE) s.passiveBlock+=u?4:3;
        if(id==C::FLEX) s.temporaryStrength+=u?4:2;
        for(int k=0;k<5;++k) if((f[1] || k==t || id==C::SHOCKWAVE) &&
                !enemy(k)[MONSTER_BASE+int(MS::ARTIFACT)]) {
            s.weak[k]+=std::max(0,int(f[33]));
            if(id==C::DISARM) s.disarm[k]+=u?3:2;
        }
    }
    return gain;
}
double continuation(const Position &s,const uint8_t *mask,int depth) const {
    if(depth==0) return 0;
    double best=0;
    for(int j=0;j<int(p[5]);++j) {
        auto h=o+OBS_HAND+j*HAND_W;
        if((s.used & (1<<j)) || !h[3] || h[2]>s.energy) continue;
        for(int t=0;t<(h[4]?5:1);++t) {
            if(!mask[5*j+t] || (h[4] && s.hp[t]<=0)) continue;
            auto next=s;
            double value=resolve(next,5*j+t);
            best=std::max(best,value+.94*continuation(next,mask,depth-1));
        }
    }
    return best;
}
#include "forecast_plan.inc"
void base_scores(const uint8_t *mask,double *out) const {
    if(use_actor && !teacher_scale && o[7] && mask[ACT_END_TURN]) {
        direct_scores(mask,out);
        return;
    }
    for(int i=0;i<N_ACTIONS;++i) {
        bool learned_drink=learned_drinks && mask[ACT_END_TURN] && i>=ACT_POTION && i<ACT_DISCARD;
        out[i]=mask[i] && !learned_drink?score(i):-1e30;
    }
    if(learned_draft && draft_skip(mask)>=0) {
        auto features=deck_features();
        for(int i=0;i<N_CHOICES;++i) if(mask[ACT_CHOOSE+i]) {
            auto c=o+OBS_CHOICES+i*CHOICE_W;
            if(ChoiceKind(int(c[0]))!=ChoiceKind::CARD) continue;
            auto x=offered_features(features,c);
            auto &coefs=draft_table[int(c[1])-1];
            out[ACT_CHOOSE+i]=std::inner_product(x.begin(),x.end(),coefs.begin(),0.);
        }
    }
    if(use_recovery && o[4]<=2 && draft_skip(mask)>=0) {
        for(int j=0;j<N_CHOICES;++j) if(mask[ACT_CHOOSE+j]) {
            auto c=o+OBS_CHOICES+j*CHOICE_W;
            if(ChoiceKind(int(c[0]))!=ChoiceKind::CARD) continue;
            int id=int(c[1])-1;
            const auto &bonus=recovery_draft[id];
            out[ACT_CHOOSE+j]+=bonus[0]+bonus[1]*(deck[id]>0)+bonus[2]*(c[2]>0);
        }
    }
    if(use_edits && !o[7]) score_edits(mask,out);
    if(!inventory_tokens.empty() && !o[7]) inventory_scores(mask,out);
    if(!o[7] && (historical_scales[0] || historical_scales[1])) historical_menu(mask,out);
    if(use_expert && !o[7]) apply_expert_prior(mask,out);
    if(!boss_draft_weights.empty() && !o[7]) apply_boss_draft(mask,out);
    if((route_scale[0] || route_scale[1]) && !o[7]) route_scores(mask,out);
    if(map_mix && !o[7]) learned_map_scores(mask,out);
    if(!o[7] && (forecast_scale[0] || forecast_scale[1] || forecast_scale[2])) deck_forecast_scores(mask,out);
    if(!o[7] && (resource_mix[0] || resource_mix[1] || resource_mix[2] || resource_mix[3])) resource_menu(mask,out);
    if(!o[7] && strat_enabled) strategic_menu(mask,out);
    if(!o[7] && enc_gain) score_predicted_fights(mask,out);
    if(!o[7] || !mask[ACT_END_TURN]) return;
    double before=*std::max_element(out,out+ACT_POTION);
    auto start=start_turn();
    for(int i=0;i<ACT_POTION;++i) if(mask[i]) {
        auto next=start;
        auto now=resolve(next,i);
        // End turn and potions compete on a comparable one-action scale.
        // Lookahead only changes relative scores among the currently playable cards.
        out[i]+=w[Lookahead]*(now+.94*continuation(next,mask,turn_depth-1));
    }
    double after=*std::max_element(out,out+ACT_POTION);
    for(int i=0;i<ACT_POTION;++i) if(mask[i]) out[i]+=reaction(i)-(after-before);
    apply_combat_head(mask,out);
    if(late_combat_scale && o[4]>=2 && o[4]<=late_combat_last_act) apply_late_combat(mask,out);
    if(learned_drinks) {
        // Each prediction is a difference from taking the best non-drink move now.
        double hold=std::max(out[ACT_END_TURN],*std::max_element(out,out+ACT_POTION));
        for(int a=ACT_POTION;a<ACT_DISCARD;++a) if(mask[a]) {
            int id=int(o[17+(a-ACT_POTION)/5])-1;
            auto x=drink_features(a);
            double advantage=std::inner_product(x.begin(),x.end(),drink_table[id].begin(),0.);
            out[a]=hold+advantage-drink_margin;
        }
    }
    if(forecast_settings[0]>0) forecast_scores(mask,out,start);
    if(scenario_draws && o[4]>=scenario_from && scenario_scores(mask,out)) return;
    if((use_boss_tactics || use_recovery) && o[4]==2 && Room(int(o[8])-1)==Room::BOSS) apply_boss_tactics(mask,out);
    if(opening_draws && !opening_branch && simulated_tactic<0 && o[4]<=2 && p[4]==0)
        opening_scores(mask,out);
    if(use_actor && teacher_scale && o[4]<=2) {
        std::array<double,N_ACTIONS> residual;
        direct_scores(mask,residual.data());
        for(int a=0;a<=ACT_END_TURN;++a) if(mask[a] && (a<ACT_DISCARD || a==ACT_END_TURN))
            out[a]+=teacher_scale*residual[a];
    }
}
