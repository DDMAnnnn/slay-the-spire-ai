// A shared single-hidden-layer action-value correction. The two embeddings identify
// the candidate card and target. The remaining inputs are public numeric features.
using Features=std::array<float,22>;
int card_index(int a) const {return a==ACT_END_TURN?0:int(o[OBS_HAND+(a/5)*HAND_W]);}
int enemy_index(int a) const {return int(enemy(a==ACT_END_TURN?0:a%5)[0]);}
Features combat_features(int a,double distance) const {
    auto m=enemy(a==ACT_END_TURN?0:a%5);
    auto h=o+OBS_HAND+(a==ACT_END_TURN?0:a/5)*HAND_W;
    bool play=a!=ACT_END_TURN;
    auto id=play?C(int(h[0])-1):C::INVALID;
    auto f=cardFacts[play && h[1]!=0][int(id)];
    return {float(std::max(-50.,distance)/20.),o[4]/4, float(missing),p[2]/30,p[3]/4,p[4]/8,p[5]/7,
        float(incoming/40),float(live)/3,m[1]/150,m[3]/20,m[8]*m[9]/35,
        p[N_PLAYER_BASE+int(PS::STRENGTH)]/10,p[N_PLAYER_BASE+int(PS::DEXTERITY)]/5,
        float(play?h[2]/3:0),float(play?h[1]:0),float(f[3])/12,float(f[4])/3,
        float(play?std::max(0,getBaseDamage(id,h[1]))/18.:0),float(f[0])/3,
        float(m[MONSTER_BASE+int(MS::VULNERABLE)]>0),float(Room(int(o[8])-1)==Room::BOSS)};
}
void apply_combat_head(const uint8_t *mask,double *out) const {
    if(head_scale==0 || !mask[ACT_END_TURN]) return;
    double top=out[ACT_END_TURN];
    for(int a=0;a<ACT_POTION;++a) top=std::max(top,out[a]);
    for(int a=0;a<=ACT_END_TURN;++a) {
        if(!mask[a] || (a>=ACT_POTION && a!=ACT_END_TURN) || out[a]<top-50) continue;
        auto x=combat_features(a,out[a]-top);
        auto &card=head_cards[card_index(a)];
        auto &target=head_enemies[enemy_index(a)];
        double v=0;
        for(int j=0;j<UNITS;++j) {
            float z=card[j]+target[j]+head_bias[j];
            for(size_t i=0;i<x.size();++i) z+=head_weights[j][i]*x[i];
            v+=std::max(0.f,z)*head_out[j];
        }
        out[a]+=head_scale*v;
    }
}

void apply_late_combat(const uint8_t *mask,double *out) const {
    Hidden hand{};
    int n=int(p[5]);
    for(int i=0;i<n;++i) {
        auto &embedding=late_hand[int(o[OBS_HAND+i*HAND_W])];
        for(int j=0;j<UNITS;++j) hand[j]+=embedding[j]/n;
    }
    double top=std::max(out[ACT_END_TURN],*std::max_element(out,out+ACT_POTION));
    double new_top=-1e30;
    for(int a=0;a<=ACT_END_TURN;++a) {
        if(!mask[a] || (a>=ACT_POTION && a!=ACT_END_TURN)) continue;
        auto x=combat_features(a,out[a]-top);
        auto &card=late_cards[card_index(a)];
        auto &target=late_enemies[enemy_index(a)];
        double value=0;
        for(int j=0;j<UNITS;++j) {
            float z=hand[j]+card[j]+target[j]+late_bias[j];
            for(size_t i=0;i<x.size();++i) z+=late_features[j][i]*x[i];
            value+=std::max(0.f,z)*late_output[j];
        }
        out[a]+=late_combat_scale*value;
        new_top=std::max(new_top,out[a]);
    }
    // Preserve the original best score relative to drinking a potion.
    for(int a=0;a<=ACT_END_TURN;++a)
        if(mask[a] && (a<ACT_POTION || a==ACT_END_TURN)) out[a]+=top-new_top;
}
