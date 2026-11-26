// A soft Bellman backup values each destination and the public paths it leaves open.
void learned_map_scores(const uint8_t *mask,double *scores) const {
    if(o[7] || ScreenState(int(o[6]))!=ScreenState::MAP_SCREEN || o[11]>=14) return;
    const int act=int(o[4]);
    if(act<1 || act>3 || !(map_active_acts & (1<<(act-1)))) return;
    int row=std::max(0,int(o[11])+1);
    std::vector<int> legal;
    for(int i=0;i<7;++i) if(mask[ACT_CHOOSE+i]) legal.push_back(i);
    if(legal.size()<2) return;
    int pots=0;
    for(int i=0;i<5;++i) pots+=o[17+i]>0;
    double miss=1.-o[0]/std::max(1.f,o[1]);
    std::array<double,13> x={1.,miss,miss*miss,std::min(3.,o[2]/200.),std::min(pots,4)/3.,
        std::min(50.,double(o[22]))/24.,double(o[4]==2),double(o[4]==3),row/14.,
        double(!o[12]),double(!o[13]),double(!o[14]),miss*(o[4]>=2)};
    std::array<double,10> u{};
    for(size_t k=0;k<x.size();++k) for(size_t j=0;j<u.size();++j) u[j]+=x[k]*map_reward[k][j];
    std::array<double,7> next{};
    auto room=[&](int y,int col){int r=int(o[OBS_MAP+(7*y+col)*MAP_W])-1;return std::clamp(r<0?8:r,0,8);};
    for(int y=map_graph?14:row;y>=row;--y) {
        std::array<double,7> now{};
        for(int col=0;col<7;++col) {
            int r=room(y,col);
            now[col]=u[r]+y/14.*map_depth[r];
            if(!o[13] && y==int(o[25]) && col==int(o[24])) now[col]+=u[9]+y/14.*map_depth[9];
            if(!map_graph || y==14) continue;
            int bits=int(o[OBS_MAP+(7*y+col)*MAP_W+1]),count=0;
            std::array<double,3> q{};
            for(int d=-1;d<=1;++d) if(col+d>=0 && col+d<7 && bits&(1<<(d+1))) {
                int v=room(y+1,col+d);
                q[count++]=next[col+d]+map_edge[0][r][v]+miss*map_edge[1][r][v];
            }
            if(!count) continue;
            double best=*std::max_element(q.begin(),q.begin()+count),mass=0;
            for(int k=0;k<count;++k) mass+=std::exp(q[k]-best);
            now[col]+=.9*(best+std::log(mass)+(map_branch[std::clamp(int(o[4])-1,0,2)]-1)*std::log(double(count)));
        }
        next=now;
    }
    double native=-1e20,learned=-1e20;
    for(int a:legal) {native=std::max(native,scores[ACT_CHOOSE+a]);learned=std::max(learned,next[a]);}
    for(int a:legal) scores[ACT_CHOOSE+a]=native+(1-map_mix)*(scores[ACT_CHOOSE+a]-native)+8*map_mix*(next[a]-learned);
}
