void View::attention_scores(const uint8_t *mask,double *scores) const {
    if(!mask[ACT_END_TURN] || !o[7] || simulated_tactic>=0 || opening_branch) return;
    int parent=int(std::max_element(scores,scores+N_ACTIONS)-scores);
    if(parent>=ACT_DISCARD && parent!=ACT_END_TURN) return;
    auto actions=distinct_combat_actions(parent,o,mask);
    const int n=int(actions.size());
    if(n<2) return;
    const auto &model=choice_attention;
    std::array<AttentionHidden,ACT_END_TURN+1> hidden{};
    std::array<AttentionKey,ACT_END_TURN+1> queries{},keys{},values{};
    for(int i=0;i<n;++i) {
        int a=actions[i];
        auto c=candidate(a,std::max(-10.,(scores[a]-scores[parent])/CURRICULUM_TEMP));
        auto &h=hidden[i];
        for(int j=0;j<ATT_WIDTH;++j) {
            float z=model.bias[j]+model.items[c.item][j]+model.foes[c.foe][j];
            h[j]=std::max(0.f,std::inner_product(c.features.begin(),c.features.end(),model.input[j].begin(),z));
        }
        for(int j=0;j<ATT_KEY;++j) {
            queries[i][j]=std::inner_product(h.begin(),h.end(),model.query[j].begin(),0.f);
            keys[i][j]=std::inner_product(h.begin(),h.end(),model.key[j].begin(),0.f);
            values[i][j]=std::inner_product(h.begin(),h.end(),model.value[j].begin(),0.f);
        }
    }
    std::array<double,ACT_END_TURN+1> adjustments{};
    for(int i=0;i<n;++i) {
        std::array<float,ACT_END_TURN+1> logits{};
        for(int k=0;k<n;++k) logits[k]=std::inner_product(queries[i].begin(),queries[i].end(),keys[k].begin(),0.f)/std::sqrt(float(ATT_KEY));
        float top=*std::max_element(logits.begin(),logits.begin()+n),denom=0;
        AttentionKey context{};
        for(int k=0;k<n;++k) {
            float weight=std::exp(logits[k]-top);
            denom+=weight;
            for(int j=0;j<ATT_KEY;++j) context[j]+=weight*values[k][j];
        }
        for(auto &x:context) x/=denom;
        for(int j=0;j<ATT_WIDTH;++j) {
            float z=std::inner_product(context.begin(),context.end(),model.mix[j].begin(),hidden[i][j]);
            adjustments[i]+=model.output[j]*std::max(0.f,z);
        }
    }
    // An identical copy in a different slot gets the same learned correction.
    std::array<double,ACT_END_TURN+1> delta{};
    for(int i=0;i<n;++i) {
        int a=actions[i];
        delta[a]=CURRICULUM_TEMP*model.scale*8*std::tanh((adjustments[i]-adjustments[0])/8);
        if(a>=ACT_POTION) continue;
        const auto h=o+OBS_HAND+(a/5)*HAND_W;
        for(int b=a%5;b<ACT_POTION;b+=5) if(mask[b] && std::equal(h,h+HAND_W,o+OBS_HAND+(b/5)*HAND_W))
            delta[b]=delta[a];
    }
    for(int a=0;a<=ACT_END_TURN;++a) if(mask[a]) scores[a]+=delta[a];
}
