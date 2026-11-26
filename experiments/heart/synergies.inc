// A deck with a scaling attack, persistent block, or an exhaust engine changes
// the marginal value of its next card. All counts come from the public master deck.
double synergy(C id,bool upgraded) const {
    bool exhaust=num(C::CORRUPTION) || relic(R::DEAD_BRANCH);
    bool persistent=num(C::BARRICADE) || relic(R::CALIPERS);
    bool wall=persistent || num(C::ENTRENCH);
    bool searing=num(C::SEARING_BLOW)>0;
    double value=0;
    switch(id) {
    case C::SEARING_BLOW: return !searing && o[3]<12 && !relic(R::FUSION_HAMMER)?18.:-10.;
    case C::CORRUPTION: value+=relic(R::DEAD_BRANCH)?45.:num(C::FEEL_NO_PAIN)?13.:3.; break;
    case C::DARK_EMBRACE: case C::FEEL_NO_PAIN: value+=exhaust?11.:2.; break;
    case C::BARRICADE: value+=wall || num(C::BODY_SLAM) || num(C::IMPERVIOUS)?12.:2.; break;
    case C::ENTRENCH: value+=persistent?17.:num(C::BODY_SLAM)?5.:0.; break;
    case C::BODY_SLAM: value+=wall?15.:num(C::POWER_THROUGH)>0?6.:0.; break;
    case C::IMPERVIOUS: case C::POWER_THROUGH: value+=wall || exhaust?6.:2.; break;
    case C::ARMAMENTS: case C::HEADBUTT: value+=searing?7.:0.; break;
    case C::LIMIT_BREAK: value+=relic(R::BRIMSTONE)?14.:0.; break;
    case C::REAPER: value+=relic(R::BRIMSTONE)?20.:5.; break;
    case C::TRUE_GRIT: case C::SECOND_WIND: value+=exhaust || persistent?5.:0.; break;
    default: break;
    }
    if(searing && getCardType(id)==CardType::ATTACK && id!=C::HEADBUTT && id!=C::REAPER) value-=9.;
    if(num(id) && getCardType(id)==CardType::POWER && id!=C::FEEL_NO_PAIN) value-=5;
    return value;
}
double scaling_attack(C id,bool up) const {
    if(!synergy_enabled) return std::max(0,getBaseDamage(id,up));
    if(id==C::RITUAL_DAGGER) return 15+std::min(25.,o[3]*.4);
    if(id!=C::SEARING_BLOW) return std::max(0,getBaseDamage(id,up));
    // The fork exposes the upgrade flag but omits the number of upgrades.
    // Use a conservative estimate for the repeated-smith drafting strategy.
    int n=up?std::clamp(int(o[3]/7),1,9):0;
    return 12+n*(n+7)/2;
}
