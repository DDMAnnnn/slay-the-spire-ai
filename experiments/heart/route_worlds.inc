// Construct independently seeded continuations from public map, inventory and keys.
bool restore_route_world(Env &e,const View &v,int sample,const uint8_t *mask) {
    const float *o=v.o;
    const int act=int(o[4]),row=int(o[11]),floor=int(o[3]);
    if(act<1 || act>3 || row>=14 || o[22]>=MAX_DECK || o[23]>=MAX_RELICS) return false;
    if(v.relic(R::BOTTLED_FLAME)||v.relic(R::BOTTLED_LIGHTNING)||v.relic(R::BOTTLED_TORNADO)) return false;
    if(v.num(C::RITUAL_DAGGER)||v.num(C::GENETIC_ALGORITHM)) return false;
    const uint64_t independent=7315829ULL+104729ULL*sample+8647ULL*act+593ULL*(row+1);
    e.record=false;e.reset(independent,20,true,CharacterClass::IRONCLAD);
    auto &g=e.gc;
    for(int a=2;a<=act;++a) g.transitionToAct(a);
    g.boss=Encounter(int(o[15])-1);
    if(act==3) {
        std::vector<Encounter> alternatives;
        for(auto boss:{Encounter::AWAKENED_ONE,Encounter::TIME_EATER,Encounter::DONU_AND_DECA})
            if(boss!=g.boss) alternatives.push_back(boss);
        g.secondBoss=alternatives[sample%alternatives.size()];
    }
    g.floorNum=floor;g.curMapNodeX=int(o[10]);g.curMapNodeY=row;
    g.screenState=ScreenState::MAP_SCREEN;g.curRoom=Room(int(o[8])-1);
    g.redKey=o[12];g.greenKey=o[13];g.blueKey=o[14];g.gold=int(o[2]);
    g.maxHp=int(o[1]);g.curHp=int(o[0]);
    g.potionCapacity=int(o[16]);g.potionCount=0;g.potions.fill(Potion::EMPTY_POTION_SLOT);
    for(int i=0;i<g.potionCapacity;++i) if(o[17+i]) {g.potions[i]=Potion(int(o[17+i])-1);++g.potionCount;}
    g.deck=Deck();g.relics=RelicContainer();
    for(int i=0;i<int(o[22]);++i) {
        auto c=o+OBS_DECK+i*CARD_W;
        g.deck.obtainRaw(Card(C(int(c[0])-1),int(c[1])));
    }
    for(int i=0;i<int(o[23]);++i) {
        auto r=o+OBS_RELICS+i*RELIC_W;R id=R(int(r[0])-1);
        g.relics.add({id,std::max(0,int(r[1])-1)});
    }
    for(auto *pool:{&g.commonRelicPool,&g.uncommonRelicPool,&g.rareRelicPool,&g.shopRelicPool,&g.bossRelicPool})
        pool->erase(std::remove_if(pool->begin(),pool->end(),[&](R id){return g.relics.has(id);}),pool->end());
    auto map=std::make_shared<Map>();
    for(int y=0;y<15;++y) for(int col=0;col<7;++col) {
        auto *seen=o+OBS_MAP+(7*y+col)*MAP_W;
        auto &n=map->nodes[y][col];n.x=col;n.y=y;n.room=seen[0]?Room(int(seen[0])-1):Room::NONE;
        // Every occupied top-row room connects to the boss at column three,
        // even when that is farther than a normal adjacent-column edge.
        if(y==14) {if(seen[0]) n.addEdge(3);continue;}
        int edges=int(seen[1]);
        for(int d=-1;d<=1;++d) if(col+d>=0 && col+d<7 && (edges&(1<<(1+d)))) {
            n.addEdge(col+d);
            if(y<14) map->nodes[y+1][col+d].addParent(col);
        }
    }
    map->burningEliteX=int(o[24]);map->burningEliteY=int(o[25]);map->burningEliteBuff=sample%4;
    g.map=std::move(map);
    // The exact past encounter count is absent from this observation. Sample a plausible
    // list position at this depth rather than using the live game's encounter stream.
    g.monsterListOffset=std::clamp((std::max(0,row)+sample%3)/2,0,int(g.monsterList.size())-1);
    g.eliteMonsterListOffset=std::clamp((std::max(0,row)+sample%5)/8,0,int(g.eliteMonsterList.size())-1);
    std::array<float,OBS_SIZE> visible;std::array<uint8_t,N_ACTIONS> legal;
    e.observe(visible.data());e.mask(legal.data());
    if(route_root_difference) {
        for(int j=0;j<N_ACTIONS;++j) if(legal[j]!=mask[j])
            route_root_difference->push_back({double(-1-j),float(mask[j]),float(legal[j])});
        for(int j=0;j<OBS_PLAYER;++j) if((j<29 || j>=OBS_RELICS) && o[j]!=visible[j])
            route_root_difference->push_back({double(j),o[j],visible[j]});
    }
    return std::equal(mask,mask+N_ACTIONS,legal.begin()) &&
        std::equal(o,o+29,visible.begin()) &&
        std::equal(o+OBS_RELICS,o+OBS_PLAYER,visible.begin()+OBS_RELICS);
}

double future_route_value(Env e,const Weights &w,int action,int horizon) {
    int floor=e.gc.floorNum,act=e.gc.act,keys=e.gc.greenKey+e.gc.redKey+e.gc.blueKey;
    // Game contexts can share their map. Isolate branches before a simulated move.
    e.gc.map=std::make_shared<Map>(*e.gc.map);
    std::array<float,OBS_SIZE> o;std::array<uint8_t,N_ACTIONS> mask;std::array<double,N_ACTIONS> scores;
    for(int step=0;step<650;++step) {
        if(!e.step(action)) throw std::runtime_error("illegal simulated route action");
        if(e.done() || e.gc.act>act || (!e.inBattle && e.gc.screenState==ScreenState::MAP_SCREEN && e.gc.floorNum>=floor+horizon)) break;
        e.observe(o.data());e.mask(mask.data());View v(o.data(),w);
        v.base_scores(mask.data(),scores.data());
        action=int(std::max_element(scores.begin(),scores.end())-scores.begin());
    }
    const bool reached=e.gc.act>act || (!e.inBattle && e.gc.screenState==ScreenState::MAP_SCREEN && e.gc.floorNum>=floor+horizon);
    const int hp=e.inBattle?e.bc.player.curHp:e.gc.curHp;
    const double hpfrac=double(std::max(0,hp))/std::max(1,e.gc.maxHp);
    const int delta=e.gc.greenKey+e.gc.redKey+e.gc.blueKey-keys;
    return .2*std::clamp(double(e.gc.floorNum-floor)/horizon,0.,1.)+(reached?1.+.6*hpfrac:0.)+.2*delta;
}

void View::sample_routes(const uint8_t *mask,double *scores) const {
    if(!route_world_count || o[7] || o[4]<route_world_act || o[4]>3 || o[11]>=14 ||
        ScreenState(int(o[6]))!=ScreenState::MAP_SCREEN) return;
    std::vector<int> options;
    for(int i=0;i<7;++i) if(mask[ACT_CHOOSE+i]) options.push_back(ACT_CHOOSE+i);
    if(options.size()<2) return;
    std::array<double,7> values{};
    int accepted=0;
    for(int i=0;i<route_world_count;++i) {
        Env e;
        if(!restore_route_world(e,*this,i,mask)) continue;
        for(size_t j=0;j<options.size();++j) values[j]+=future_route_value(e,w,options[j],route_world_horizon);
        ++accepted;
    }
    if(!accepted) return;
    const int prior=int(std::max_element(scores,scores+N_ACTIONS)-scores);
    int best=0;
    for(size_t j=1;j<options.size();++j) if(scores[options[j]]>scores[options[best]]) best=int(j);
    for(size_t j=0;j<options.size();++j) scores[options[j]]+=route_world_scale*(values[j]-values[best])/accepted;
    if(route_world_audit) {
        ++(*route_world_audit)[0];(*route_world_audit)[1]+=accepted;(*route_world_audit)[2]+=prior!=int(std::max_element(scores,scores+N_ACTIONS)-scores);
    }
}
