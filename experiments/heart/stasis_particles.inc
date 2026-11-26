// Infer the cards carried by living Bronze Orbs from the permanent deck and the four public piles.
// With two distinct missing cards the observation cannot identify the owner. Sample both assignments.
std::vector<int> missing_stasis_cards(const float *o,int holders) {
    std::array<int,2*(1+CARD_COUNT)> balance{};
    auto count=[&](int begin,int rows,int width,int sign) {
        for(int j=0;j<rows;++j) {
            const float *r=o+begin+j*width;
            if(!r[0]) break;
            balance[2*int(r[0])+(r[1]>0)]+=sign;
        }
    };
    count(OBS_DECK,MAX_DECK,CARD_W,1);
    count(OBS_HAND,10,HAND_W,-1);
    for(int offset:{OBS_DRAW,OBS_DISCARD,OBS_EXHAUST}) count(offset,MAX_PILE,CARD_W,-1);
    std::vector<int> missing;
    for(int k=2;k<int(balance.size());++k) {
        // Generated cards, in-combat upgrades or played powers can make subtraction ambiguous.
        if(balance[k]<0 || int(missing.size())+balance[k]>holders) return {};
        missing.insert(missing.end(),balance[k],k);
    }
    if(int(missing.size())!=holders) return {};
    return missing;
}
