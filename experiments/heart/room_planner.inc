// Back up survival probabilities through the visible map, preserving the HP distribution after each room.
void room_plan_scores(const uint8_t *mask,double *scores) const {
    const int act=int(o[4])-1;
    if(o[7] || act<0 || act>=3 || !(room_plan_acts & (1<<act)) ||
       ScreenState(int(o[6]))!=ScreenState::MAP_SCREEN) return;
    const int start=int(o[11])+1;
    if(start<0 || start>=14) return;
    std::vector<int> legal;
    for(int x=0;x<7;++x) if(mask[ACT_CHOOSE+x]) legal.push_back(x);
    if(legal.size()<2) return;
    int pot=0;for(int i=0;i<5;++i) pot|=o[17+i]>0;
    const int last=std::min(14,start+room_plan_horizon-1);
    using Nodes=std::array<std::array<std::array<double,9>,2>,7>;
    Nodes next{},now{};
    const int tail_stage=std::min(3,(last+1)/4);
    for(auto &node:next) for(int green=0;green<2;++green) for(int hp=0;hp<9;++hp) {
        const int base=(((act*4+tail_stage)*2+pot)*2+green)*9;
        node[green][hp]=room_tail_table[base+hp];
    }
    for(int row=last;row>=start;--row) {
        now={};
        for(int col=0;col<7;++col) {
            const float *cell=o+OBS_MAP+(7*row+col)*MAP_W;
            int room=int(cell[0])-1;
            if(room<0 || room>=8) continue;
            bool burning=room==int(Room::ELITE) && col==int(o[24]) && row==int(o[25]);
            const int tile=burning?8:room;
            const int base=(((act*4+std::min(row/4,3))*2+pot)*9+tile)*9*10;
            for(int green=0;green<2;++green) {
                std::array<double,9> after{};
                int g=green || burning;
                if(row==last) after=next[col][g];
                else for(int d=-1;d<=1;++d) if(col+d>=0 && col+d<7 && int(cell[1])&(1<<(d+1)))
                    for(int k=0;k<9;++k) after[k]=std::max(after[k],next[col+d][g][k]);
                for(int hp=0;hp<9;++hp)
                    now[col][green][hp]=std::inner_product(after.begin(),after.end(),room_transition_table.begin()+base+hp*10+1,0.);
            }
        }
        next=now;
    }
    const double at=8.*std::clamp(double(o[0])/std::max(1.f,o[1]),0.,1.);
    const int lo=int(at),hi=std::min(8,lo+1);
    const int green=int(bool(o[13]));
    std::array<double,7> returns{};
    for(int x:legal) returns[x]=next[x][green][lo]+(at-lo)*(next[x][green][hi]-next[x][green][lo]);
    int parent=*std::max_element(legal.begin(),legal.end(),[&](int a,int b){return scores[ACT_CHOOSE+a]<scores[ACT_CHOOSE+b];});
    for(int x:legal) scores[ACT_CHOOSE+x]+=room_plan_gain*(returns[x]-returns[parent]);
}
