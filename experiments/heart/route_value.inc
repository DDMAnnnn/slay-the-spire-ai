int route_kind() const {
    if(o[7]) return -1;
    auto screen=ScreenState(int(o[6]));
    if(screen==ScreenState::MAP_SCREEN && o[11]<14) return 0;
    return screen==ScreenState::REST_ROOM?1:-1;
}
RouteInput route_input(int kind,int choice,double prior) const {
    RouteInput x{};
    auto deckContext=deck_features();
    std::copy(deckContext.begin(),deckContext.begin()+14,x.begin());
    x[14]=o[0]/80; x[15]=o[1]/80; x[16]=o[12]; x[17]=o[13]; x[18]=o[14];
    for(int i=0;i<5;++i) x[19]+=float(o[17+i]>0);
    x[20]=float(std::clamp(prior,-100.,100.)/40.);
    x[21]=num(C::DEMON_FORM)+num(C::INFLAME)+num(C::SPOT_WEAKNESS);
    x[22]=num(C::APPARITION)+num(C::IMPERVIOUS); x[23]=relic(R::BURNING_BLOOD);
    x[24]=relic(R::FUSION_HAMMER); x[25]=relic(R::COFFEE_DRIPPER);
    x[26]=o[11]/14; x[27]=kind;
    if(kind==1) x[28+choice]=1;
    // Track all reachable columns. For each distance window, record both
    // opportunities and unavoidable rooms, since a fork changes their value.
    int mask=1<<(kind==1?std::clamp(int(o[10]),0,6):choice);
    const int start=int(o[11])+(kind==0);
    for(int y=std::max(0,start);y<15 && mask;++y) {
        int next=0,reach=0;
        std::array<int,6> counts{};
        for(int col=0;col<7;++col) if(mask&(1<<col)) {
            auto node=o+OBS_MAP+(7*y+col)*MAP_W;
            int room=int(node[0])-1;
            if(room<0) continue;
            ++reach;
            if(room<6) ++counts[room];
            if(!o[13] && col==int(o[24]) && y==int(o[25])) x[64]=1.f/(1+.15f*(y-start));
            for(int d=-1;d<=1;++d) if(col+d>=0 && col+d<7 && (int(node[1])&(1<<(d+1))))
                next|=1<<(col+d);
        }
        int dist=y-start;
        for(int type=0;type<6;++type) {
            if(dist==0) x[35+type]=counts[type]>0;
            else if(dist<=3) {
                x[41+type]+=float(counts[type]>0)/3;
                x[47+type]+=float(reach>0 && counts[type]==reach)/3;
            } else x[53+type]+=float(counts[type]>0)/10;
        }
        if(dist>0 && dist<=5) x[58+dist]=reach/7.f;
        mask=next;
    }
    return x;
}
void route_scores(const uint8_t *mask,double *scores) const {
    int kind=route_kind();
    if(kind<0 || !route_scale[kind]) return;
    for(int j=0;j<7;++j) if(mask[ACT_CHOOSE+j]) {
        auto &score=scores[ACT_CHOOSE+j];
        const auto x=route_input(kind,j,score);
        float value=0;
        for(int h=0;h<ROUTE_WIDTH;++h)
            value+=route_output[h]*std::max(0.f,std::inner_product(x.begin(),x.end(),route_layer[h].begin(),route_bias[h]));
        score+=route_scale[kind]*value;
    }
}
