// The true held cards are used only to grade the inference after the public prediction.
std::array<int,5> audit_stasis(const Bank &bank,const Weights &w) {
    std::array<int,5> counts{};
    for(auto e:bank.states) {
        int before=e.gc.floorNum;
        bool seen=false;
        for(int k=0;k<450 && !e.done() && e.gc.floorNum<=before+1;++k) {
            if(e.inBattle && e.bc.encounter!=Encounter::AUTOMATON) break;
            if(e.inBattle) seen=true;
            else if(seen) break;
            std::array<float,OBS_SIZE> o;std::array<uint8_t,N_ACTIONS> mask;
            std::array<double,N_ACTIONS> q;
            e.observe(o.data());e.mask(mask.data());
            if(e.inBattle && mask[ACT_END_TURN]) {
                std::vector<int> actual;
                for(int i=0;i<e.bc.monsters.monsterCount;++i) {
                    const float *s=o.data()+OBS_MONSTERS+i*MONSTER_W;
                    if(s[4] || !s[MONSTER_BASE+int(MS::STASIS)]) continue;
                    const auto &c=e.bc.cards.stasisCards[std::min(1,i)];
                    actual.push_back(2*(1+int(c.id))+c.upgraded);
                }
                if(!actual.empty()) {
                    ++counts[1];
                    auto guess=missing_stasis_cards(o.data(),int(actual.size()));
                    if(!guess.empty()) {
                        ++counts[2];
                        std::sort(actual.begin(),actual.end());
                        counts[3]+=actual==guess;
                        counts[4]+=guess.size()==2 && guess.front()!=guess.back();
                    }
                }
            }
            View v(o.data(),w);v.base_scores(mask.data(),q.data());
            if(!e.step(int(std::max_element(q.begin(),q.end())-q.begin()))) throw std::runtime_error("invalid audit action");
        }
        counts[0]+=seen;
    }
    return counts;
}
