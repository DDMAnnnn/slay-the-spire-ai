struct Candidate {
    int item=0, foe=0;
    CurriculumFeatures features{};
};
Candidate candidate(int a,double gap) const {
    Candidate c;
    const int specials=CARD_COUNT+1+int(std::size(potionNames))+int(std::size(relicNames));
    auto &x=c.features;
    x[40]=float(gap);x[41]=o[2]/250;x[42]=o[22]/25;x[43]=o[23]/12;
    x[44]=o[12];x[45]=o[13];x[46]=o[14];x[47]=o[7];
    x[48]=o[4]/4;x[49]=o[0]/80;x[50]=o[3]/56;x[51]=float(missing);
    if(o[7]) {
        int t=actor_target(a);
        int slot=t;
        if(slot<0) for(int j=0;j<5;++j) if(enemy(j)[6]) {slot=j;break;}
        c.foe=slot<0?0:int(enemy(slot)[0]);
        auto f=actor_features(a,t);
        std::copy(f.begin(),f.end(),x.begin());
    }
    if(a<ACT_POTION) c.item=card_index(a);
    else if(a<ACT_DISCARD) c.item=actor_item(a);
    else if(a<ACT_CHOOSE || a==ACT_CONFIRM) c.item=specials+(a==ACT_END_TURN?0:a==ACT_CONFIRM?1:2);
    else {
        auto opt=o+OBS_CHOICES+(a-ACT_CHOOSE)*CHOICE_W;
        auto kind=ChoiceKind(int(opt[0]));
        int id=int(opt[1]);
        c.item=specials+4+int(kind)*20;
        x[52]=opt[2]/4;x[53]=std::min(400.f,opt[3])/100;
        if(kind==ChoiceKind::CARD && id>0 && id<=CARD_COUNT) {
            c.item=id;x[54]=float(num(C(id-1)));x[55]=float(card_prior(C(id-1))/12);
        } else if(kind==ChoiceKind::POTION && id>0) c.item=CARD_COUNT+id;
        else if(kind==ChoiceKind::RELIC && id>0) c.item=CARD_COUNT+int(std::size(potionNames))+id;
        else if(kind==ChoiceKind::MAP_NODE) c.item+=std::clamp(int(opt[2]),0,18);
        else if(kind==ChoiceKind::EVENT_OPTION || kind==ChoiceKind::REST_OPTION || kind==ChoiceKind::NEOW_OPTION)
            c.item+=std::min(a-ACT_CHOOSE,18);
        x[56]=o[6]/10;x[57]=opt[0]/14;
    }
    return c;
}
double curriculum_delta(const Candidate &c) const {
    double y=0;
    for(int h=0;h<CURRICULUM_WIDTH;++h) {
        float z=curriculum_bias[h]+curriculum_items[c.item][h]+curriculum_foes[c.foe][h];
        z=std::inner_product(c.features.begin(),c.features.end(),curriculum_layer[h].begin(),z);
        y+=std::max(0.f,z)*curriculum_out[h];
    }
    return 8.*std::tanh(y/8.);
}
void all_scores(const uint8_t *mask,double *out) const {
    base_scores(mask,out);
    if(!curriculum_scale || simulated_tactic>=0 || opening_branch) return;
    if((curriculum_scope==1 && !o[7]) || (curriculum_scope==2 && o[7])) return;
    double top=*std::max_element(out,out+N_ACTIONS);
    for(int a=0;a<N_ACTIONS;++a) if(mask[a] && out[a]>=top-7*CURRICULUM_TEMP) {
        double gap=(out[a]-top)/CURRICULUM_TEMP;
        out[a]+=CURRICULUM_TEMP*curriculum_scale*curriculum_delta(candidate(a,gap));
    }
}
