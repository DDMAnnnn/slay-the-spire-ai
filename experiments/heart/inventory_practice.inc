std::vector<std::array<float,4>> grade_inventories(const Weights &w,const std::vector<Inventory> &rows,
        unsigned seed,int groupSize,int draws,int threads) {
    if(groupSize<1 || rows.size()%groupSize || draws<1 || threads<1)
        throw std::invalid_argument("expected complete inventory families");
    for(const auto &inv:rows) validate_inventory(inv);
    const Encounter fights[4][5]={
        {Encounter::THE_GUARDIAN,Encounter::SLIME_BOSS,Encounter::HEXAGHOST,Encounter::GREMLIN_NOB,Encounter::THREE_SENTRIES},
        {Encounter::CHAMP,Encounter::COLLECTOR,Encounter::AUTOMATON,Encounter::SLAVERS,Encounter::BOOK_OF_STABBING},
        {Encounter::TIME_EATER,Encounter::AWAKENED_ONE,Encounter::DONU_AND_DECA,Encounter::REPTOMANCER,Encounter::GIANT_HEAD},
        {Encounter::SHIELD_AND_SPEAR,Encounter::SHIELD_AND_SPEAR,Encounter::SHIELD_AND_SPEAR,
         Encounter::SHIELD_AND_SPEAR,Encounter::SHIELD_AND_SPEAR}};
    std::vector<std::array<float,4>> values(rows.size());
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for(int i=0;i<int(rows.size());++i) {
        // Identical sample seeds within a family compare inventory edits under the same opponent and HP.
        std::mt19937 rng(seed+i/groupSize);
        const auto &inv=rows[i];
        for(int k=0;k<4;++k) for(int s=0;s<draws;++s) {
            int enemy=rng()%5,sample=1000+rng()%10000000;
            auto enc=fights[k][enemy];
            std::vector<int> key{int(enc),k+1,80,2,20,0,0,0,0,0};
            key.insert(key.end(),inv.first.begin(),inv.first.end());key.push_back(-1);
            key.insert(key.end(),inv.second.begin(),inv.second.end());
            auto room=k==3 || enemy>=3?Room::ELITE:Room::BOSS;
            auto state=scenario_start(key,sample,-1,-1,room);
            double result=scenario_return(state,w,0);
            double value=result>=1.?.8+.8*(result-1.):.5*result;
            values[i][k]+=float(value/draws);
        }
    }
    return values;
}
