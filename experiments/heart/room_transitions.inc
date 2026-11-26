// Roll out each legal next room from a training-only copy. The deployed router sees the map.
using RoomTransition=std::array<int,21>;
std::vector<std::vector<RoomTransition>> room_transitions(
        const Weights &w,const std::vector<uint64_t> &seeds,int threads) {
    std::vector<std::vector<RoomTransition>> result(seeds.size());
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for(int ix=0;ix<int(seeds.size());++ix) {
        Env e;e.record=false;e.reset(seeds[ix],20,true,CharacterClass::IRONCLAD);
        std::mt19937 rng(uint32_t(seeds[ix]) ^ 0xba9188du);
        std::array<float,OBS_SIZE> o;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        auto pick=[&](Env &world) {
            world.observe(o.data());world.mask(mask.data());
            View(o.data(),w).all_scores(mask.data(),scores.data());
            return int(std::max_element(scores.begin(),scores.end())-scores.begin());
        };
        int pending=-1;
        int steps=0;
        for(;steps<3500 && !e.done();++steps) {
            const int a=pick(e);
            bool map=!e.inBattle && e.gc.screenState==ScreenState::MAP_SCREEN;
            if(map && pending>=0) {
                auto &r=result[ix][pending];
                if(r[15]!=e.gc.curHp || r[16]!=e.gc.maxHp || r[17]!=e.gc.act || !r[18])
                    throw std::runtime_error("unchanged room prediction differs from natural next map");
                pending=-1;
            }
            if(map && e.gc.act<=3 && e.gc.curMapNodeY<14 && rng()%3==0 &&
               a>=ACT_CHOOSE && a<ACT_CHOOSE+7) {
                const auto before=o;
                const auto legal=mask;
                int n=0;for(int i=0;i<5;++i) n+=before[17+i]>0;
                const int row=e.gc.curMapNodeY+1;
                for(int x=0;x<7;++x) if(legal[ACT_CHOOSE+x]) {
                    const int room=int(before[OBS_MAP+(7*row+x)*MAP_W])-1;
                    if(room<0 || room>=8) continue;
                    Env b=e;b.gc.map=std::make_shared<Map>(*e.gc.map);
                    if(!b.step(ACT_CHOOSE+x)) throw std::runtime_error("invalid forked map action");
                    int j=0;
                    for(;j<650 && !b.done() && (b.inBattle || b.gc.screenState!=ScreenState::MAP_SCREEN);++j)
                        if(!b.step(pick(b))) throw std::runtime_error("invalid forked room action");
                    if(j==650) continue;
                    bool parent=a==ACT_CHOOSE+x;
                    if(parent) pending=int(result[ix].size());
                    bool burning=int(before[24])==x && int(before[25])==row;
                    result[ix].push_back({ix,e.gc.act,row,e.gc.curHp,e.gc.maxHp,n,int(before[22]),int(before[23]),
                        int(before[2]),int(before[12])+2*int(before[13])+4*int(before[14]),room,int(burning),
                        x,int(parent),int(e.gc.floorNum),int(b.gc.curHp),int(b.gc.maxHp),int(b.gc.act),
                        int(!b.done() || b.won()),0,0});
                }
            }
            if(!e.step(a)) throw std::runtime_error("invalid natural map action");
        }
        if(pending>=0 && e.done()) {
            auto &r=result[ix][pending];
            if(r[15]!=e.gc.curHp || r[17]!=e.gc.act || r[18]!=int(e.won()))
                throw std::runtime_error("unchanged room terminal result differs");
        }
        for(auto &r:result[ix]) {r[19]=e.gc.act;r[20]=int(steps>=3500);}
    }
    return result;
}
