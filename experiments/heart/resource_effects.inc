// Continuation estimates are learned offline from siblings with equal starting inventories.
using HpSurface = std::array<std::array<double,21>,12>;
using PotionSurface = std::array<std::array<std::array<std::vector<double>,5>,21>,12>;
static HpSurface resource_hp{};
static PotionSurface resource_potion{};
static double resource_drink_gain=0.,resource_search_mix=0.,resource_typical_cost=4.;

struct ResourcePosition {
    int context, lo, hi;
    double fraction;
    ResourcePosition(int act,int floor,double health,double maximum) {
        context=4*(act-1)+std::clamp((floor-1-17*(act-1))/4,0,3);
        const double at=20.*std::clamp(health/std::max(1.,maximum),0.,1.);
        lo=int(at);hi=std::min(20,lo+1);fraction=at-lo;
    }
    double interpolate(double a,double b) const {return a+(b-a)*fraction;}
    double health() const {return interpolate(resource_hp[context][lo],resource_hp[context][hi]);}
    double potion(int count,int id) const {
        const int n=std::clamp(count-1,0,4);
        return interpolate(resource_potion[context][lo][n].at(id),resource_potion[context][hi][n].at(id));
    }
    double inventory(const std::array<int,5> &items) const {
        double total=0.;int n=0;
        for(int id:items) if(id>0) {
            ++n;total+=potion(n,0)+potion(1,id)-potion(1,0);
        }
        return total;
    }
};
