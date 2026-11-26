// Action utilities use only the fork's public observation and legal mask.
// The simulator and episode log are available exclusively to the rollout driver.
#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <cstring>
#include <unordered_map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "constants/CardFacts.h"
#include "constants/Events.h"
#include "constants/RelicPotionFacts.h"
#include "env/Env.h"
#include "sim/search/BattleScumSearcher2.h"

namespace py = pybind11;
using namespace sts;
using namespace sts::env;
using C = CardId;
using R = RelicId;

enum Weight { Defense, Kill, Focus, Scaling, Draw, Debuff, Elite, Rest, ShopVisit, Hall,
              Draft, Duplicate, Heal, PotionUse, CostPenalty, KeyRoute,
              Swap, EndgameDraft, EnergyRelic, Finisher, Lookahead, Count };
using Weights = std::array<double, Count>;
static Weights late_weights{};
static Weights final_act_weights{};
static bool use_final_weights=false;
static int late_from=5;
using EventTable = std::array<std::array<double,16>,std::size(eventGameNames)>;
static EventTable event_values{};
constexpr int CARD_COUNT=std::size(cardNames);
constexpr int CHOICE_FEATURES=CARD_COUNT+16*std::size(eventGameNames);
using BossDraftInput=std::array<float,32+CARD_COUNT>;
static std::vector<BossDraftInput> boss_draft_weights;
using ChoiceTable=std::array<std::array<double,CHOICE_FEATURES>,3>;
static ChoiceTable learned_values{};
using BossRelicTable = std::array<double,std::size(relicNames)>;
static BossRelicTable first_boss_bonus{};
constexpr int UNITS=16;
using Hidden=std::array<float,UNITS>;
using Cards=std::array<Hidden,CARD_COUNT+1>;
using Enemies=std::array<Hidden,std::size(monsterIdStrings)+1>;
using Matrix=std::array<std::array<float,22>,UNITS>;
static Cards head_cards{};
static Enemies head_enemies{};
static Matrix head_weights{};
static Hidden head_bias{},head_out{};
static double head_scale=0;
static Cards late_cards{},late_hand{};
static Enemies late_enemies{};
static Matrix late_features{};
static Hidden late_bias{},late_output{};
static double late_combat_scale=0;
static int late_combat_last_act=4;
static bool synergy_enabled=true;
static int turn_depth=2; // Includes the first card being scored.
using IntentForecast=std::array<std::array<std::array<float,3>,18>,std::size(monsterIdStrings)+1>;
static IntentForecast intent_forecast{};
// Strength of the planner, cost of HP, value of unspent enemy HP, future discount.
using ForecastSettings=std::array<double,4>;
static ForecastSettings forecast_settings{};
using BossTactics=std::array<std::array<double,18>,3>;
static BossTactics boss_tactics{};
static bool use_boss_tactics=false;
using RecoveryInput=std::array<float,40>;
using RecoveryHidden=std::array<float,32>;
using RecoveryMenu=std::array<BossTactics,16>;
using RecoveryLayer=std::array<RecoveryInput,32>;
using RecoveryReadout=std::array<RecoveryHidden,16>;
static RecoveryMenu recovery_menu{};
static RecoveryLayer recovery_layer{};
static RecoveryHidden recovery_bias{};
static RecoveryReadout recovery_readout{};
static std::array<float,16> recovery_offset{};
using RecoveryDraft=std::array<std::array<float,3>,CARD_COUNT>;
static RecoveryDraft recovery_draft{};
static bool use_recovery=false;
enum Edit { AddCard, UpgradeCard, RemoveCard, GainPotion, EditCount };
using EditWeights=std::array<float,4>;
using EditTable=std::array<std::array<std::array<EditWeights,CARD_COUNT>,EditCount>,3>;
static EditTable edit_values{};
static bool use_edits=false;
constexpr int ROUTE_INPUT=65, ROUTE_WIDTH=48;
using RouteInput=std::array<float,ROUTE_INPUT>;
using RouteHidden=std::array<float,ROUTE_WIDTH>;
using RouteLayer=std::array<RouteInput,ROUTE_WIDTH>;
static RouteLayer route_layer{};
static RouteHidden route_bias{},route_output{};
static std::array<double,2> route_scale{};
using ScenarioMenu=std::array<std::array<double,18>,12>;
static ScenarioMenu scenario_menu{};
static int scenario_draws=0, scenario_from=3;
static unsigned scenario_version=0;
static thread_local int simulated_tactic=-1;
static int opening_draws=0, opening_scope=0;
static double opening_scale=0.;
static thread_local bool opening_branch=false;
using OpeningAudit=std::array<int,4>;
static thread_local OpeningAudit *opening_audit=nullptr;
using PlayBase = std::array<double,CARD_COUNT>;
PlayBase initial_play_base() {
    PlayBase base{};
    base[int(C::OFFERING)]=base[int(C::APOTHEOSIS)]=35.;
    return base;
}
static PlayBase play_base=initial_play_base();
using DrinkFeatures = std::array<double,17>;
using DrinkTable = std::array<DrinkFeatures,std::size(potionNames)>;
static DrinkTable drink_table{};
static bool learned_drinks=false;
static double drink_margin=0;
using NeowTable=std::array<std::array<double,std::size(Neow::drawbackStrings)>,std::size(Neow::bonusStrings)>;
static NeowTable neow_values{},expert_neow{};
using ExpertCards=std::array<std::array<std::array<double,2>,CARD_COUNT>,3>;
using ExpertRelics=std::array<BossRelicTable,2>;
static ExpertCards expert_cards{};
static ExpertRelics expert_relics{};
static bool use_expert=false;
static bool learned_neow=false;
using DraftFeatures=std::array<double,17>;
using DraftTable=std::array<DraftFeatures,CARD_COUNT>;
static DraftTable draft_table{};
static bool learned_draft=false;
constexpr int ACTOR_WIDTH=32, ACTOR_INPUT=40;
using ActorVector=std::array<float,ACTOR_WIDTH>;
using ActorFeatures=std::array<float,ACTOR_INPUT>;
using ActorItems=std::array<ActorVector,CARD_COUNT+1+std::size(potionNames)>;
using ActorTargets=std::array<ActorVector,std::size(monsterIdStrings)+1>;
using ActorWeights=std::array<ActorFeatures,ACTOR_WIDTH>;
static ActorItems actor_items{},actor_hand{};
static ActorTargets actor_targets{};
static ActorWeights actor_weights{};
static ActorVector actor_bias{},actor_out{};
static bool use_actor=false;
static double teacher_scale=0.;
constexpr int CURRICULUM_INPUT=58,CURRICULUM_WIDTH=20;
constexpr double CURRICULUM_TEMP=6.;
using CurriculumFeatures=std::array<float,CURRICULUM_INPUT>;
using CurriculumHidden=std::array<float,CURRICULUM_WIDTH>;
using CurriculumLayer=std::array<CurriculumFeatures,CURRICULUM_WIDTH>;
using CurriculumItems=std::array<CurriculumHidden,CARD_COUNT+1+std::size(potionNames)+std::size(relicNames)+320>;
using CurriculumFoes=std::array<CurriculumHidden,std::size(monsterIdStrings)+1>;
static CurriculumLayer curriculum_layer{};
static CurriculumHidden curriculum_bias{},curriculum_out{};
static CurriculumItems curriculum_items{};
static CurriculumFoes curriculum_foes{};
static double curriculum_scale=0.;
static int curriculum_scope=0;
constexpr int INVENTORY_WIDTH=48;
using InventoryHidden=std::array<float,INVENTORY_WIDTH>;
using InventoryTokens=std::vector<InventoryHidden>;
using InventoryHeads=std::array<InventoryHidden,4>;
static InventoryTokens inventory_tokens;
static InventoryHidden inventory_bias{};
static InventoryHeads inventory_heads{};
static std::array<double,4> inventory_intercept{},inventory_settings{};
static int inventory_draws=0,inventory_mode=0;
static double inventory_search_scale=0.;

double card_prior(C id) {
    switch (id) {
    case C::APOTHEOSIS: case C::OFFERING: case C::IMMOLATE: case C::FEED: return 11;
    case C::APPARITION: return 14;
    case C::REAPER: case C::DEMON_FORM: case C::DISARM: case C::IMPERVIOUS: return 9;
    case C::BLUDGEON: case C::SHOCKWAVE: case C::UPPERCUT: case C::BATTLE_TRANCE: case C::FLAME_BARRIER:
    case C::INFLAME: case C::SHRUG_IT_OFF: case C::FIEND_FIRE: case C::SPOT_WEAKNESS: return 8;
    case C::WHIRLWIND: case C::POMMEL_STRIKE: case C::LIMIT_BREAK: case C::PUMMEL:
    case C::CORRUPTION: case C::FEEL_NO_PAIN: case C::ANGER: case C::CARNAGE:
    case C::DARK_EMBRACE: case C::POWER_THROUGH: case C::SECOND_WIND: return 7;
    case C::HEMOKINESIS: case C::PERFECTED_STRIKE: case C::IRON_WAVE: case C::TWIN_STRIKE:
    case C::METALLICIZE: case C::ARMAMENTS: case C::HEAVY_BLADE: case C::SWORD_BOOMERANG:
    case C::HEADBUTT: case C::TRUE_GRIT: case C::BURNING_PACT: case C::CLEAVE:
        return 6;
    default: break;
    }
    if (getCardType(id) == CardType::CURSE || getCardType(id) == CardType::STATUS) return -12;
    if (id == C::STRIKE_RED || id == C::DEFEND_RED) return -2;
    return 3;
}

constexpr int HISTORICAL_ITEMS=2+CARD_COUNT+int(std::size(relicNames));
using HistoricalHidden=std::array<float,24>;
using HistoricalEmbeddings=std::vector<HistoricalHidden>;
static HistoricalEmbeddings historical_context,historical_options;
static std::array<std::vector<float>,3> historical_bias;
static std::vector<float> historical_upgrade,historical_repeat;
static std::array<double,2> historical_scales{};
static std::array<double,12> historical_gate{};

#include "choice_attention.inc"
#include "public_belief.inc"

struct View {
    const float *o, *p;
    std::array<int, std::size(cardNames)> deck{};
    double missing, incoming = 0;
    int live = 0;
    const Weights &w;
    View(const float *obs, const Weights &weights): o(obs), p(obs+OBS_PLAYER),
        w(use_final_weights && obs[4]>=3?final_act_weights:int(obs[4])>=late_from?late_weights:weights) {
        missing = 1. - o[0] / std::max(1.f, o[1]);
        for (int i=0; i<MAX_DECK; ++i) {
            int id = int(o[OBS_DECK + i*CARD_W])-1;
            if (id<0) break;
            ++deck[id];
        }
        for (int i=0; i<5; ++i) {
            auto m = enemy(i);
            if (m[6]) { ++live; incoming += m[8]*m[9]; }
        }
    }
    const float *enemy(int t) const { return o+OBS_MONSTERS+t*MONSTER_W; }
    int num(C id) const { return deck[int(id)]; }
    bool relic(R id) const {
        for (int i=0; i<MAX_RELICS; ++i) {
            int r=int(o[OBS_RELICS+i*RELIC_W])-1;
            if (r==int(id)) return true;
            if (r<0) break;
        }
        return false;
    }
#include "synergies.inc"
    double draft(C id,bool up) const {
        double score=card_prior(id)+1.5*up+(synergy_enabled?synergy(id,up):0.);
        int scaling=num(C::DEMON_FORM)+num(C::INFLAME)+num(C::SPOT_WEAKNESS);
        if (id==C::REAPER) score+=5+3*(scaling>0);
        if (id==C::BARRICADE && (num(C::ENTRENCH)||num(C::BODY_SLAM))) score+=6;
        if (id==C::POWER_THROUGH || id==C::FLAME_BARRIER) score+=2;
        if (id==C::CORRUPTION && (relic(R::DEAD_BRANCH)||num(C::DARK_EMBRACE))) score+=10;
        if (id==C::ANGER && o[4]>1) score-=5;
        if (id==C::SPOT_WEAKNESS && scaling==0) score+=3;
        if (o[4]>1) {
            if (id==C::DEMON_FORM && !scaling) score+=8;
            if (id==C::SPOT_WEAKNESS && !num(id)) score+=5;
            if ((id==C::HEAVY_BLADE || id==C::PUMMEL || id==C::REAPER) && scaling) score+=3;
            if (id==C::DISARM && num(id)<2) score+=4;
            if (id==C::IMPERVIOUS && num(id)<2) score+=3;
            if (id==C::CORRUPTION && num(C::FEEL_NO_PAIN)) score+=5;
            if (id==C::BODY_SLAM && (num(C::BARRICADE)||num(C::ENTRENCH))) score+=9;
            if (id==C::ENTRENCH && num(C::BARRICADE)) score+=9;
        }
        if (getCardType(id)==CardType::ATTACK && o[4]==1) {
            double printed=std::max(0,getBaseDamage(id,up))*std::max(1,int(cardFacts[up][int(id)][0]));
            if(id==C::PERFECTED_STRIKE) printed+=2*(num(C::STRIKE_RED)+num(C::POMMEL_STRIKE)+1);
            score+=std::min(3.5,printed/6.)*std::max(0.,(17.-o[22])/6.);
        }
        if (id==C::DEMON_FORM && num(id)==0) score+=2;
        if (id==C::CORRUPTION && !num(C::FEEL_NO_PAIN) && !relic(R::DEAD_BRANCH)) score-=3;
        if (id==C::BARRICADE && num(C::IMPERVIOUS)+num(C::POWER_THROUGH)>0) score+=5;
        if (id==C::LIMIT_BREAK) score += (num(C::INFLAME)+num(C::SPOT_WEAKNESS)+num(C::DEMON_FORM)+int(relic(R::VAJRA))) ? 3 : -7;
        if (id==C::CORRUPTION && relic(R::DEAD_BRANCH)) score+=15;
        if ((id==C::DARK_EMBRACE || id==C::FEEL_NO_PAIN) && num(C::CORRUPTION)) score+=4;
        if (id==C::IMMOLATE && o[4]==1) score+=3;
        if (id==C::DEMON_FORM && o[4]>1) score+=3;
        return score-w[Duplicate]*num(id);
    }
#include "horizon.inc"
    double play(int a) const {
        const float *h=o+OBS_HAND+HAND_W*(a/5), *m=enemy(a%5);
        C id=C(int(h[0])-1);
        bool up=h[1]!=0;
        auto f=cardFacts[up][int(id)];
        double str=p[N_PLAYER_BASE+int(PlayerStatus::STRENGTH)];
        double dex=p[N_PLAYER_BASE+int(PlayerStatus::DEXTERITY)];
        int energy=int(p[3]), cost=std::max(0,int(h[2]));
        double hits=std::max(1,int(f[0])), base=scaling_attack(id,up);
        if (id==C::WHIRLWIND) hits=energy;
        if (id==C::HEAVY_BLADE) str*=up?5:3;
        if (id==C::BODY_SLAM) base=p[2];
        if (id==C::PERFECTED_STRIKE)
            base+=(up?3:2)*(num(C::STRIKE_RED)+num(C::POMMEL_STRIKE)+num(C::TWIN_STRIKE)+num(id)+num(C::WILD_STRIKE));
        double dmg=base || getCardType(id)==CardType::ATTACK ? std::max(0.,base+str)*hits : 0.;
        if (p[N_PLAYER_BASE+int(PlayerStatus::WEAK)]) dmg*=.75;
        if (m[MONSTER_BASE+int(MonsterStatus::VULNERABLE)] && h[4]) dmg*=1.5;
        if (id==C::FIEND_FIRE) dmg*=std::max(1.f,p[5]-1);
        double targetHp=m[1]+m[3];
        double score=dmg*(f[1]?1.+.72*(live-1):1.);
        if (h[4] && dmg>0) {
            score+=(3+w[Focus])*20./(targetHp+5);
            if (dmg>=targetHp) score+=w[Kill]+m[8]*m[9];
            else if (energy>cost && live>1) {
                // Credit progress toward a kill achievable with another visible card.
                for(int j=0;j<int(p[5]);++j) {
                    auto h2=o+OBS_HAND+j*HAND_W;
                    if(j==a/5 || !h2[3] || h2[2]>energy-cost) continue;
                    auto next=C(int(h2[0])-1);
                    int base2=getBaseDamage(next,h2[1]);
                    if(base2>0 && dmg+base2+str>=targetHp) {
                        score+=w[Finisher]+.65*m[8]*m[9]; break;
                    }
                }
            }
            if (m[MONSTER_BASE+int(MonsterStatus::INTANGIBLE)]) score*=.18;
        }
        double blk=f[3]>0?std::max(0.,f[3]+dex):0;
        if (id==C::SHRUG_IT_OFF && !blk) blk=up?11:8;
        if (p[N_PLAYER_BASE+int(PlayerStatus::FRAIL)]) blk*=.75;
        double need=std::max(0.,incoming-p[2]);
        score+=w[Defense]*std::min(blk,need)+.09*blk;
        score+=w[Draw]*std::min(3,int(f[4]))*(energy>cost?1:.25);
        score+=w[Debuff]*(f[32] && !m[MONSTER_BASE+int(MonsterStatus::VULNERABLE)]?1.:0.);
        if (id==C::BASH && !m[MONSTER_BASE+int(MonsterStatus::VULNERABLE)])
            score+=w[Debuff]*(energy>=3?1.6:1);
        if (id==C::SHOCKWAVE || id==C::DISARM) score+=w[Debuff]*2.5;
        if (id==C::SHRUG_IT_OFF) score+=2;
        if (id==C::FLAME_BARRIER && need>0) score+=std::min(12.,incoming*.65);
        if (id==C::IMPERVIOUS && need<7) score-=10;
        if (id==C::ENTRENCH) score+=std::min(double(p[2]),need)*w[Defense];
        if (id==C::BLOODLETTING && energy>1) score-=10;
        if (id==C::EXHUME && o[OBS_EXHAUST]==0) score-=10;
        if (id==C::APPARITION) score+= (incoming>0 && !p[N_PLAYER_BASE+int(PlayerStatus::INTANGIBLE)])?55:3;
        if (id==C::BARRICADE && live) score+=8;
        if (id==C::SECOND_WIND) {
            int exhausted=0;
            for(int j=0;j<int(p[5]);++j) if (j!=a/5 &&
                getCardType(C(int(o[OBS_HAND+j*HAND_W])-1))!=CardType::ATTACK) ++exhausted;
            score+=w[Defense]*std::min(need,exhausted*(up?7.:5.));
        }
        if(id==C::BASH && m[MONSTER_BASE+int(MonsterStatus::VULNERABLE)]<=1 && live==1) score+=5;
        if(id==C::DISARM && m[MONSTER_BASE+int(MonsterStatus::ARTIFACT)]==0) score+=12;
        if(id==C::REAPER && str>3) score+=std::min(18.,str);
        if (id==C::REAPER && missing<.15 && live==1) score-=6;
        if (getCardType(id)==CardType::SKILL && enemy(0)[MONSTER_BASE+int(MonsterStatus::ENRAGE)])
            score-=13;
        if (id==C::SPOT_WEAKNESS && m[8]>0) score+=w[Scaling]*10;
        if (id==C::LIMIT_BREAK && str>0) score+=w[Scaling]*(4+str*2);
        if (getCardType(id)==CardType::POWER) score+=w[Scaling]*(8+card_prior(id))*(p[4]<5?1.:.75);
        score+=play_base[int(id)];
        if (id==C::REAPER) score+=std::min(double(o[1]-o[0]),dmg*live)*1.4;
        if (id==C::FEED && dmg>=targetHp) score+=18;
        if (id==C::ARMAMENTS && up && p[5]>2) score+=10;
        if (id==C::ANGER && dmg>0) score+=2;
        score += f[5]*5 - f[6]*.7;
        return (score+.04)/(1+w[CostPenalty]*std::max(0,cost-1)) + (cost==0?1.2:0);
    }
    double map_score(int x) const {
        // Work backward along the visible map edges. Each room gets credit for the best route it leaves open.
        double next[7]{}, now[7]{};
        for (int y=14; y>int(o[11]); --y) {
            for (int col=0; col<7; ++col) {
                auto node=o+OBS_MAP+(7*y+col)*MAP_W;
                Room room=Room(int(node[0])-1);
                double value=0, tail=-1e8;
                switch(room) {
                case Room::REST: value=w[Rest]*(1+missing)+(synergy_enabled && num(C::SEARING_BLOW)?7.:0.); break;
                case Room::SHOP: value=o[2]>130?w[ShopVisit]:-.8; break;
                case Room::ELITE: value=w[Elite]*(1.-2.2*missing)-(o[4]==1?5.:2.5); break;
                case Room::MONSTER: value=w[Hall]*(o[3]<5?1.8:1.); break;
                case Room::EVENT: value=o[4]>=2?2.2:.8; break;
                default: value=.3; break;
                }
                if (!o[13] && col==o[24] && y==o[25])
                    value+=(o[4]==3?100.:w[KeyRoute]*(o[4]==2?.9:.15));
                for(int delta=-1;delta<=1;++delta) {
                    int dest=col+delta;
                    if(dest>=0 && dest<7 && (int(node[1]) & (1<<(delta+1)))) tail=std::max(tail,next[dest]);
                }
                now[col]=value+.94*(tail < -1e7 ? 0.:tail);
            }
            std::copy(now,now+7,next);
        }
        return next[x];
    }
    double relic_value(R id) const {
        switch(id) {
        case R::PANDORAS_BOX: return 20;
        case R::SNECKO_EYE: return 15;
        case R::RUNIC_PYRAMID: return 8;
        case R::COFFEE_DRIPPER: return 4;
        case R::SOZU: case R::FUSION_HAMMER: case R::CURSED_KEY: return w[EnergyRelic];
        case R::VELVET_CHOKER: case R::ECTOPLASM: return 7;
        case R::MARK_OF_PAIN: case R::BUSTED_CROWN: return 5;
        case R::DEAD_BRANCH: case R::TORII: case R::TUNGSTEN_ROD: case R::FOSSILIZED_HELIX: return 13;
        case R::BRIMSTONE: return num(C::DEMON_FORM)?13:28;
        case R::PANTOGRAPH: case R::MEAT_ON_THE_BONE: case R::INCENSE_BURNER: return 16;
        case R::SACRED_BARK: return 8;
        case R::ASTROLABE: return (num(C::STRIKE_RED)+num(C::DEFEND_RED)>4)?12:6;
        case R::BLACK_BLOOD: return 11;
        case R::PHILOSOPHERS_STONE: return 8;
        case R::PRISMATIC_SHARD: return -15;
        case R::RUNIC_DOME: return -18;
        default: return 6;
        }
    }
    DrinkFeatures drink_features(int a) const {
        int slot=(a-ACT_POTION)/5;
        auto id=Potion(int(o[17+slot])-1);
        auto m=enemy((a-ACT_POTION)%5);
        bool targeted=potionRequiresTarget(id);
        int occupied=0;
        for(int j=0;j<5;++j) occupied+=o[17+j]>0;
        auto room=Room(int(o[8])-1);
        return {1., missing, o[0]/80., p[2]/30., p[3]/4., p[5]/7.,
                incoming/40., std::max(0.,incoming-p[2])/std::max(1.f,o[0]),
                live/3., p[4]/8., o[4]/4., double(room==Room::ELITE), double(room==Room::BOSS),
                occupied/std::max(1.f,o[16]), targeted?m[1]/80.:0.,
                targeted?m[3]/20.:0., targeted?m[8]*m[9]/30.:0.};
    }
    double potion(int a) const {
        if(a>=ACT_DISCARD) return -80;
        int slot=(a-ACT_POTION)/5;
        auto id=Potion(int(o[17+slot])-1);
        if(id==Potion::FRUIT_JUICE) return 60;
        if(id==Potion::BLOOD_POTION) return missing>.27?80:-65;
        if(!o[7]) return -65;
        if(id==Potion::SMOKE_BOMB) return incoming-p[2]>p[0]*.85 && o[4]!=4?55:-65;
        bool boss=Room(int(o[8])-1)==Room::BOSS || o[4]==4;
        bool elite=Room(int(o[8])-1)==Room::ELITE;
        double risk=std::max(0.,incoming-p[2]);
        double urgency=w[PotionUse]*(double(boss)*19+double(elite)*15+12*missing+risk*.35)-11;
        if(id==Potion::REGEN_POTION && missing>.28) return 50;
        if(id==Potion::LIQUID_BRONZE && (o[4]==4 || boss)) return 60;
        if(id==Potion::CULTIST_POTION && boss) return 90;
        if(id==Potion::BLOCK_POTION) return p[3]==0 && risk>3?urgency+risk:-20;
        if(id==Potion::FIRE_POTION || id==Potion::EXPLOSIVE_POTION || id==Potion::POISON_POTION) {
            auto m=enemy((a-ACT_POTION)%5);
            return urgency+(m[1]+m[3]<=20 && m[8]>0?8.:0.)+(p[3]==0?1.:-5.);
        }
        if(id==Potion::ENERGY_POTION || id==Potion::SWIFT_POTION) return p[3]==0?urgency:-15;
        return urgency+(p[4]<=1?4.:0.);
    }
    int feature(int i) const {
        const float *c=o+OBS_CHOICES+i*CHOICE_W;
        auto kind=ChoiceKind(int(c[0]));
        if(kind==ChoiceKind::CARD) return int(c[1])-1;
        if(kind==ChoiceKind::EVENT_OPTION && i<16) return CARD_COUNT+16*(int(o[9])-1)+i;
        return -1;
    }
    double learned(int i) const {
        int f=feature(i);
        return f>=0?learned_values[std::min(3,int(o[4]))-1][f]:0.;
    }
    double score(int a) const {
        if(a<ACT_POTION) return play(a);
        if(a<ACT_END_TURN) return potion(a);
        if(a==ACT_END_TURN) return 0;
        if(a==ACT_CONFIRM) return 4;
        int i=a-ACT_CHOOSE;
        const float *c=o+OBS_CHOICES+i*CHOICE_W;
        ChoiceKind kind=ChoiceKind(int(c[0]));
        if(o[7]) {
            if(kind!=ChoiceKind::CARD) return kind==ChoiceKind::SKIP?-10:1;
            C id=C(int(c[1])-1);
            auto task=CardSelectTask(int(o[26]));
            if(c[3]) return -1000;
            double rank=card_prior(id)+c[2];
            if(synergy_enabled && id==C::SEARING_BLOW && c[2]) rank+=20;
            if(synergy_enabled && id==C::REAPER && missing>.3) rank+=12;
            if(task==CardSelectTask::EXHAUST_ONE || task==CardSelectTask::EXHAUST_MANY ||
               task==CardSelectTask::DISCARD || task==CardSelectTask::GAMBLE) return 3-rank;
            return 6+rank;
        }
        auto screen=ScreenState(int(o[6]));
        switch(screen) {
        case ScreenState::EVENT_SCREEN:
            if(kind==ChoiceKind::NEOW_OPTION) {
                if(learned_neow) return neow_values.at(int(c[1])-1).at(int(c[2])-1);
                return i==3 ? w[Swap]-1 : -.001*i;
            }
            if(kind==ChoiceKind::EVENT_OPTION && i<16)
                return learned(i)+event_values.at(int(o[9])-1)[i] - .0001*i;
            return -.001*i;
        case ScreenState::MAP_SCREEN: return map_score(i);
        case ScreenState::REST_ROOM:
            if(i==2) return !o[12] && (o[4]>=3 || (o[4]>=2 && missing<.18))?45:-4;
            if(i==0) return missing*w[Heal];
            if(synergy_enabled && i==1 && num(C::SEARING_BLOW) && missing<.68 && !relic(R::FUSION_HAMMER)) return 42;
            return i==1?8.:i==3?8.5:2.;
        case ScreenState::TREASURE_ROOM: return 4-i;
        case ScreenState::BOSS_RELIC_REWARDS: {
            if(kind!=ChoiceKind::RELIC) return 0.;
            int id=int(c[1])-1;
            return relic_value(R(id))+(o[4]==1?first_boss_bonus.at(id):0.);
        }
        case ScreenState::CARD_SELECT: {
            C id=C(int(c[1])-1);
            bool remove=int(o[26])==1 || int(o[26])==2 || int(o[26])==4 || int(o[26])==8;
            if(synergy_enabled && !remove && id==C::SEARING_BLOW && int(o[26])==int(CardSelectScreenType::UPGRADE)) return 70;
            return remove?-card_prior(id):card_prior(id)+(id==C::BASH && o[4]==1?9:0);
        }
        case ScreenState::SHOP_ROOM:
        case ScreenState::REWARDS: {
            bool shop=screen==ScreenState::SHOP_ROOM;
            switch(kind) {
            case ChoiceKind::KEY:
                if(c[1]==2 && o[4]<3) {
                    for(int j=0;j<N_CHOICES;++j) {
                        auto item=o+OBS_CHOICES+j*CHOICE_W;
                        if(int(item[0])==int(ChoiceKind::RELIC) && relic_value(R(int(item[1])-1))>=13) return 15;
                    }
                }
                return 90;
            case ChoiceKind::GOLD: return 60;
            case ChoiceKind::RELIC: return shop?relic_value(R(int(c[1])-1))-.027*c[3]:50;
            case ChoiceKind::POTION: {
                auto id=Potion(int(c[1])-1);
                double priority=o[4]==4?14.:o[4]>=2?6.:3.;
                if(id==Potion::FAIRY_POTION || id==Potion::CULTIST_POTION || id==Potion::GHOST_IN_A_JAR) priority+=6;
                return shop?priority-.03*c[3]:35;
            }
            case ChoiceKind::CARD_REMOVE: return 3.;
            case ChoiceKind::CARD: return learned(i)+draft(C(int(c[1])-1),c[2])-w[Draft]-
                w[EndgameDraft]*std::max(0.,double(o[22])-16) - (shop?.035*c[3]:0.);
            default: return 0;
            }
        }
        default: return -i*.001;
        }
    }
#include "combat_head.inc"
#include "direct_actor.inc"
#include "draft_context.inc"
#include "boss_draft.inc"
#include "expert_prior.inc"
#include "human_strategy.inc"
#include "deck_edits.inc"
#include "inventory_value.inc"
#include "route_value.inc"
#include "boss_tactics.inc"
    bool scenario_scores(const uint8_t *mask,double *out) const;
    void attention_scores(const uint8_t *mask,double *out) const;
    void belief_scores(const uint8_t *mask,double *out) const;
    void inventory_search(const uint8_t *mask,double *out) const;
    void opening_scores(const uint8_t *mask,double *out) const;
#include "turn_plan.inc"
#include "curriculum_actor.inc"
};

#include "choice_attention_scores.inc"
#include "scenario.inc"
#include "opening_search.inc"
struct Result { int floor=0, act=0, keys=0, steps=0; bool heart=false, truncated=false, encounter=false; };
Result advance(Env &e,const Weights &w,std::vector<int>* trace=nullptr,int stopAct=0,
               bool stopAtFirstChest=false,int stopFloor=0,int stepLimit=3500) {
    std::array<float,OBS_SIZE> obs{};
    std::array<uint8_t,N_ACTIONS> mask{};
    Result r;
    for (;r.steps<stepLimit && !e.done();++r.steps) {
        e.observe(obs.data());
        if(stopAct && obs[4]>=stopAct) break;
        if(stopFloor && obs[3]>=stopFloor) break;
        if(obs[7] && obs[4]==4 && obs[OBS_MONSTERS]==1+int(MonsterId::CORRUPT_HEART) && obs[OBS_MONSTERS+6])
            r.encounter=true;
        if(stopAtFirstChest && obs[4]==1 && !obs[7] &&
           ScreenState(int(obs[6]))==ScreenState::BOSS_RELIC_REWARDS) break;
        e.mask(mask.data());
        View v(obs.data(),w);
        std::array<double,N_ACTIONS> utilities;
        v.all_scores(mask.data(),utilities.data());
        double best=-1e30; int choice=-1;
        for(int i=0;i<N_ACTIONS;++i) if(mask[i]) {
            double s=utilities[i];
            if(s>best) {best=s;choice=i;}
        }
        if(choice<0 || !e.step(choice)) throw std::runtime_error("invalid policy action");
        if(trace) trace->push_back(choice);
    }
    e.observe(obs.data()); r.floor=int(obs[3]); r.act=int(obs[4]);
    r.keys=int(obs[12]+obs[13]+obs[14]); r.truncated=!e.done();
    r.heart=e.won() && r.act==4; // Python verifier additionally requires the logged final encounter.
    return r;
}

Result run(const Weights &w,uint64_t seed,std::vector<int>* trace=nullptr) {
    Env e; e.record=false; e.reset(seed,20,true,CharacterClass::IRONCLAD);
    return advance(e,w,trace);
}

// Natural prefixes from training seeds only. Snapshots initialise a rollout;
// the scorer still receives exactly the public observation used in fresh runs.
struct Bank {
    std::vector<Env> states;
    Bank()=default;
    Bank(const Weights &w,const std::vector<uint64_t> &seeds,int floor,int threads) {
        if(floor<1 || floor>56 || seeds.empty() || threads<1) throw std::invalid_argument("invalid floor bank");
        states.resize(seeds.size());
        std::vector<int> reached(seeds.size());
        #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
        for(int i=0;i<int(seeds.size());++i) {
            auto &e=states[i]; e.record=false; e.reset(seeds[i],20,true,CharacterClass::IRONCLAD);
            auto r=advance(e,w,nullptr,0,false,floor);
            reached[i]=!e.done() && r.floor>=floor;
        }
        if(std::find(reached.begin(),reached.end(),0)!=reached.end())
            throw std::invalid_argument("seed did not reach the requested floor with this policy");
    }
    Bank(const Weights &w,uint64_t seed,int limit,int count,int act) {
        if(count<1 || limit<1 || act<2 || act>4) throw std::invalid_argument("curriculum range");
        for(int i=0;i<limit && int(states.size())<count;++i) {
            Env e; e.record=false; e.reset(seed+i,20,true,CharacterClass::IRONCLAD);
            auto r=advance(e,w,nullptr,act);
            if(!e.done() && r.act==act) states.push_back(e);
        }
    }
    std::vector<std::array<int,6>> evaluate(const Weights &w,int threads) const {
        std::vector<std::array<int,6>> rows(states.size());
        #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
        for(int i=0;i<int(states.size());++i) {
            Env e=states[i]; auto r=advance(e,w);
            rows[i]={r.floor,r.act,r.keys,r.steps,int(r.heart),int(r.truncated)};
        }
        return rows;
    }
    std::vector<std::array<int,5>> progress(const Weights &w,int threads) const {
        if(threads<1) throw std::invalid_argument("positive threads required");
        std::vector<std::array<int,5>> rows(states.size());
        #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
        for(int i=0;i<int(states.size());++i) {
            Env e=states[i]; auto r=advance(e,w);
            rows[i]={r.floor,r.act,int(r.encounter),int(r.heart),int(r.truncated)};
        }
        return rows;
    }
};

#include "stage_policy.inc"
#include "choice_learning.inc"
#include "boss_relic_practice.inc"
#include "combat_practice.inc"
#include "late_combat_practice.inc"
#include "combat_preferences.inc"
#include "neow_practice.inc"
#include "draft_practice.inc"
#include "actor_practice.inc"
#include "trajectory_teacher.inc"
#include "curriculum_practice.inc"
#include "macro_teacher.inc"
#include "rollout_cache.inc"
#include "combat_returns.inc"
#include "public_belief_search.inc"
#include "belief_distillation.inc"
#include "matched_fights.inc"
#include "inventory_search.inc"
#include "rescue_combat.inc"
#include "inventory_benchmark.inc"
#include "inventory_practice.inc"
#include "boss_draft_practice.inc"
#include "intent_practice.inc"
#include "boss_drill.inc"
#include "boss_entry_health.inc"
#include "recovery_drill.inc"
#include "edit_practice.inc"
#include "route_practice.inc"

std::vector<std::array<int,5>> tactic_progress(const Bank &bank,const Weights &w,int choice,int threads) {
    if(threads<1 || choice<0 || choice>=int(scenario_menu.size()) || !scenario_draws)
        throw std::invalid_argument("configure practice tactics first");
    std::vector<std::array<int,5>> rows(bank.states.size());
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for(int i=0;i<int(bank.states.size());++i) {
        Env e=bank.states[i];
        simulated_tactic=choice;
        auto r=advance(e,w);
        simulated_tactic=-1;
        rows[i]={r.floor,r.act,int(r.encounter),int(r.heart),int(r.truncated)};
    }
    return rows;
}

PYBIND11_MODULE(heart_native,m) {
    m.def("set_opening_search",[](int draws,int scope,double scale){
        if(draws<1 || draws>16 || scope<0 || scope>2 || !std::isfinite(scale) || scale<0)
            throw std::invalid_argument("invalid opening search");
        opening_draws=draws;opening_scope=scope;opening_scale=scale;
    });
    m.def("reset_opening_search",[](){opening_draws=0;});
    m.def("opening_audit",[](const Weights &w,uint64_t start,int n){
        if(n<1) throw std::invalid_argument("positive audit count required");
        OpeningAudit result{};
        struct Guard { ~Guard(){opening_audit=nullptr;} } guard;
        opening_audit=&result;
        for(int i=0;i<n;++i) run(w,start+i);
        return result;
    });
    m.def("route_practice",&route_practice);
    m.def("set_route",[](const RouteLayer &layer,const RouteHidden &bias,const RouteHidden &out,
                         const std::array<double,2> &scale) {
        route_layer=layer;route_bias=bias;route_output=out;route_scale=scale;
    });
    m.def("reset_route",[](){route_scale={};});
    m.def("set_scenarios",[](const ScenarioMenu &menu,int draws,int act){
        if(draws<1 || draws>64 || act<1 || act>4) throw std::invalid_argument("invalid scenario horizon");
        scenario_menu=menu; scenario_draws=draws; scenario_from=act; ++scenario_version;
    });
    m.def("reset_scenarios",[](){scenario_draws=0;++scenario_version;});
    m.def("set_edits",[](const EditTable &table){edit_values=table;use_edits=true;});
    m.def("reset_edits",[](){use_edits=false;});
    m.def("set_recovery",[](const RecoveryMenu &menu,const RecoveryLayer &layer,const RecoveryHidden &bias,
              const RecoveryReadout &readout,const std::array<float,16> &offset,const RecoveryDraft &draft) {
        recovery_menu=menu; recovery_layer=layer; recovery_bias=bias; recovery_readout=readout;
        recovery_offset=offset; recovery_draft=draft; use_recovery=true;
    });
    m.def("reset_recovery",[](){use_recovery=false;});
    m.def("set_boss_tactics",[](const BossTactics &t){boss_tactics=t;use_boss_tactics=true;});
    m.def("reset_boss_tactics",[](){use_boss_tactics=false;});
    m.def("intent_practice",&intent_practice);
    m.def("set_forecast",[](const IntentForecast &table,const ForecastSettings &settings){
        intent_forecast=table; forecast_settings=settings;
    });
    m.def("reset_forecast",[](){forecast_settings={};});
    m.attr("actor_items")=actor_items.size();
    m.def("boss_draft_practice",&boss_draft_practice);
    m.def("reset_expert",[](){use_expert=false;});
    m.def("set_expert",[](const ExpertCards &cards,const ExpertRelics &relics,const NeowTable &neow){
        expert_cards=cards;expert_relics=relics;expert_neow=neow;use_expert=true;
    });
    m.def("set_boss_draft",[](std::vector<BossDraftInput> table){
        if(!table.empty() && table.size()!=CARD_COUNT) throw std::invalid_argument("invalid card table");
        boss_draft_weights=std::move(table);
    });
    m.def("trajectory_practice",&trajectory_practice);
    m.def("grade_inventories",&grade_inventories);
    m.def("set_inventory_value",[](InventoryTokens tokens,const InventoryHidden &bias,
            const InventoryHeads &heads,const std::array<double,4> &offset,const std::array<double,4> &settings){
        if(tokens.size()!=2*CARD_COUNT+std::size(relicNames)) throw std::invalid_argument("invalid inventory tokens");
        inventory_tokens=std::move(tokens);inventory_bias=bias;inventory_heads=heads;
        inventory_intercept=offset;inventory_settings=settings;
    });
    m.def("reset_inventory_value",[](){inventory_tokens.clear();});
    m.def("inventory_benchmark",&inventory_benchmark);
    m.def("bank_inventories",&bank_inventories);
    m.attr("curriculum_items")=curriculum_items.size();
    m.attr("curriculum_foes")=curriculum_foes.size();
    m.def("curriculum_practice",&curriculum_practice);
    m.def("set_curriculum",[](const CurriculumItems &items,const CurriculumFoes &foes,
                                const CurriculumLayer &layer,const CurriculumHidden &bias,
                                const CurriculumHidden &out,double scale){
        curriculum_items=items;curriculum_foes=foes;curriculum_layer=layer;
        curriculum_bias=bias;curriculum_out=out;curriculum_scale=scale;
    });
    m.def("set_curriculum_scope",[](int scope){
        if(scope<0 || scope>5) throw std::invalid_argument("invalid curriculum scope");
        curriculum_scope=scope;
    });
    m.def("reset_curriculum",[](){curriculum_scale=0.;curriculum_scope=0;});
    m.def("set_teacher_scale",[](double scale){teacher_scale=scale;});
    m.def("actor_practice",&actor_practice);
    m.def("set_actor",[](const ActorItems &items,const ActorItems &hand,const ActorTargets &targets,
                         const ActorWeights &weights,const ActorVector &bias,const ActorVector &out){
        actor_items=items; actor_hand=hand; actor_targets=targets;
        actor_weights=weights; actor_bias=bias; actor_out=out; use_actor=true;
    });
    m.def("reset_actor",[](){use_actor=false;});
    m.def("late_combat_practice",&late_combat_practice);
    m.def("set_late_combat",[](const Cards &cards,const Cards &hand,const Enemies &enemies,
                               const Matrix &features,const Hidden &bias,const Hidden &output,double scale){
        late_cards=cards; late_hand=hand; late_enemies=enemies; late_features=features;
        late_bias=bias; late_output=output; late_combat_scale=scale;
    });
    m.def("reset_late_combat",[](){late_combat_scale=0.;});
    m.def("set_late_combat_last_act",[](int act){
        if(act<2 || act>4) throw std::invalid_argument("expected act 2 through 4");
        late_combat_last_act=act;
    });
    m.def("set_final_weights",[](const Weights &w){final_act_weights=w;use_final_weights=true;});
    m.def("reset_final_weights",[](){use_final_weights=false;});
    m.def("set_draft",[](const DraftTable &t){draft_table=t;learned_draft=true;});
    m.def("reset_draft",[](){learned_draft=false;});
    m.def("draft_practice",&draft_practice);
    m.def("set_neow",[](const NeowTable &t){neow_values=t;learned_neow=true;});
    m.def("reset_neow",[](){learned_neow=false;});
    m.def("neow_practice",&neow_practice);
    m.attr("neow_shape")=py::make_tuple(std::size(Neow::bonusStrings),std::size(Neow::drawbackStrings));
    m.def("initial_play_base",&initial_play_base);
    m.def("set_play_base",[](const PlayBase &base){play_base=base;});
    m.def("set_drinks",[](const DrinkTable &table,double margin){
        drink_table=table; drink_margin=margin; learned_drinks=true;
    });
    m.def("reset_drinks",[](){learned_drinks=false;});
    m.def("preference_practice",&preference_practice);
    m.def("macro_teacher",&macro_teacher);
    m.def("set_choice_attention",[](std::vector<AttentionHidden> items,std::vector<AttentionHidden> foes,
            const std::array<CurriculumFeatures,ATT_WIDTH> &input,const AttentionHidden &bias,
            const AttentionProj &query,const AttentionProj &key,const AttentionProj &value,
            const std::array<AttentionKey,ATT_WIDTH> &mix,const AttentionHidden &output,double scale){
        if(items.size()!=curriculum_items.size() || foes.size()!=curriculum_foes.size())
            throw std::invalid_argument("invalid choice attention embedding sizes");
        choice_attention={std::move(items),std::move(foes),input,bias,query,key,value,mix,output,scale};
    });
    m.def("reset_choice_attention",[](){choice_attention.scale=0.;});
    m.def("collect_beliefs",&collect_beliefs);
    m.def("belief_examples",&belief_examples);
    m.def("combat_student_priors",&combat_student_priors);
    m.def("enrich_late_particles",&enrich_late_particles);
    m.def("set_rollout_cache",[](bool enabled){cache_rollout_actions=enabled;});
    m.def("rollout_cache_audit",[](const Weights &w,uint64_t start,int n){
        std::array<int64_t,2> counts{};
        struct Guard {
            explicit Guard(std::array<int64_t,2> &c){rollout_cache_audit=&c;}
            ~Guard(){rollout_cache_audit=nullptr;}
        } guard(counts);
        for(int i=0;i<n;++i) run(w,start+i);
        return counts;
    });
    m.def("set_public_tree",[](int budget,int depth){
        if(budget<0 || budget>8192 || depth<1 || depth>24) throw std::invalid_argument("invalid tree budget");
        public_tree_budget=budget;public_tree_depth=depth;
    });
    m.def("public_tree_audit",[](const Weights &w,uint64_t start,int n){
        std::array<int,5> counts{};
        struct Guard {
            explicit Guard(std::array<int,5> &c){tree_audit=&c;}
            ~Guard(){tree_audit=nullptr;}
        } guard(counts);
        for(int i=0;i<n;++i) run(w,start+i);
        return counts;
    });
    m.def("set_continuation",[](int width,bool free_plays){
        if(width<0 || width==1 || width>8) throw std::invalid_argument("invalid continuation width");
        next_play_width=width;search_zero_energy=free_plays;
    });
    m.def("continuation_audit",[](const Weights &w,uint64_t start,int n){
        std::array<int,4> counters{};
        struct Guard {
            explicit Guard(std::array<int,4> &c){continuation_audit=&c;}
            ~Guard(){continuation_audit=nullptr;}
        } guard(counters);
        for(int i=0;i<n;++i) run(w,start+i);
        return counters;
    });
    m.def("crossfit_advantage",[](const std::vector<std::vector<double>> &r,double margin){
        if(r.empty() || r.front().empty() || std::any_of(r.begin(),r.end(),[&](const auto &x){return x.size()!=r.front().size();}))
            throw std::invalid_argument("rectangular outcome table required");
        return crossfit_advantage(r,margin);
    });
    m.def("set_historical_menu",[](HistoricalEmbeddings context,HistoricalEmbeddings options,
            std::array<std::vector<float>,3> bias,std::vector<float> upgrade,std::vector<float> repeat,
            std::array<double,2> scales){
        if(context.size()!=HISTORICAL_ITEMS || options.size()!=HISTORICAL_ITEMS || upgrade.size()!=HISTORICAL_ITEMS ||
           repeat.size()!=HISTORICAL_ITEMS || std::any_of(bias.begin(),bias.end(),[](const auto &r){return r.size()!=HISTORICAL_ITEMS;}))
            throw std::invalid_argument("invalid historical menu vocabulary");
        historical_context=std::move(context);historical_options=std::move(options);historical_bias=std::move(bias);
        historical_upgrade=std::move(upgrade);historical_repeat=std::move(repeat);historical_scales=scales;
    });
    m.def("set_historical_gate",[](const std::array<double,12> &params){historical_gate=params;});
    m.def("reset_historical_menu",[](){historical_scales={};});
    m.def("set_belief_action_limit",[](int limit){
        if(limit<0 || limit>int(N_ACTIONS)) throw std::invalid_argument("invalid search action limit");
        belief_action_limit=limit;
    });
    m.def("set_belief_plans",[](int tactics,double win,bool enriched){
        if(tactics<1 || tactics>3 || !(win>=0)) throw std::invalid_argument("invalid search objective");
        belief_plans=tactics;belief_win_value=win;use_late_particles=enriched;
    });
    py::class_<MatchedFights>(m,"MatchedFights")
        .def(py::init<const Weights &,uint64_t,int,int>())
        .def_readonly("info",&MatchedFights::info).def("evaluate",&MatchedFights::evaluate);
    m.def("paired_draw_audit",[](int n){
        if(n<1) throw std::invalid_argument("positive audit sample required");
        std::array<int,14> stats{};
        for(int s=0;s<n;++s) for(int coupled=0;coupled<=1;++coupled) {
            std::vector<int> key{int(Encounter::JAW_WORM),1,80,2,20,0,0,0,0,0};
            for(C c:{C::STRIKE_RED,C::DEFEND_RED,C::BASH,C::STRIKE_RED,C::DEFEND_RED,C::ANGER,
                     C::SHRUG_IT_OFF,C::IRON_WAVE,C::STRIKE_RED,C::FLEX}) key.push_back((1+int(c))*256);
            key.push_back(-1);
            auto deck_order=[&](const std::vector<int> &k){
                auto e=scenario_start(k,s+701,70,5,Room::MONSTER,50,coupled!=0);
                std::vector<C> deck;
                for(int i=0;i<e.bc.cards.cardsInHand;++i) deck.push_back(e.bc.cards.hand[i].id);
                const auto &pile=e.bc.cards.drawPile;
                for(int i=int(pile.size())-1;i>=0;--i) deck.push_back(pile[i].id);
                return deck;
            };
            auto base=deck_order(key);
            key.insert(key.end()-1,(1+int(C::POMMEL_STRIKE))*256);
            auto alt=deck_order(key);
            auto it=std::find(alt.begin(),alt.end(),C::POMMEL_STRIKE);
            if(base.size()!=10 || alt.size()!=11 || it==alt.end()) throw std::runtime_error("missing audit card");
            if(coupled) ++stats[2+(it-alt.begin())];
            alt.erase(it);stats[coupled]+=alt==base;
        }
        return stats;
    });
    m.def("set_inventory_search",[](int draws,int mode,double scale){
        if(draws<0 || draws>32 || mode<0 || mode>1) throw std::invalid_argument("invalid inventory search settings");
        inventory_draws=draws;inventory_mode=mode;inventory_search_scale=scale;
    });
    m.def("set_belief",[](int draws,int scope,double scale){
        if(draws<0 || draws>32 || scope<0 || scope>2) throw std::invalid_argument("invalid belief settings");
        belief_draws=draws;belief_scope=scope;belief_scale=scale;
    });
    m.def("belief_audit",[](const Weights &w,uint64_t start,int n){
        std::array<int,5> counts{};
        struct Guard{~Guard(){belief_audit=nullptr;}} guard;
        belief_audit=&counts;
        for(int i=0;i<n;++i) run(w,start+i);
        return counts;
    });
    m.def("combat_returns",&combat_returns);
    m.def("rescue_combat",&rescue_combat);
    m.def("set_turn_depth",[](int depth){
        if(depth<1 || depth>10) throw std::invalid_argument("turn depth must be 1 to 10 cards");
        turn_depth=depth;
    });
    m.def("set_first_boss_bonus",[](const BossRelicTable &table){first_boss_bonus=table;});
    m.attr("relic_count")=first_boss_bonus.size();
    m.def("practice_first_boss",&practice_first_boss);
    m.def("set_choice_table",[](const ChoiceTable &t){learned_values=t;});
    m.def("set_synergies",[](bool enabled){synergy_enabled=enabled;});
    m.attr("feature_count")=CHOICE_FEATURES;
    m.attr("card_count")=CARD_COUNT;
    m.def("practice",&practice);
    m.def("combat_practice",&combat_practice);
    m.def("set_late",[](int act,const Weights &weights){late_from=act;late_weights=weights;});
    m.attr("enemy_count")=head_enemies.size();
    m.def("set_head",[](const Cards &cards,const Enemies &enemies,const Matrix &weights,const Hidden &bias,const Hidden &out,double scale){
        head_cards=cards; head_enemies=enemies; head_weights=weights; head_bias=bias; head_out=out; head_scale=scale;
    });
    py::class_<ActOpening>(m,"ActOpening")
        .def(py::init<const Weights&,uint64_t,int,int>())
        .def_readonly("prefix",&ActOpening::prefix)
        .def_readonly("arrivals",&ActOpening::arrivals)
        .def("features",&ActOpening::features)
        .def("finish",&ActOpening::finish);
    py::class_<Bank>(m,"Bank").def(py::init<const Weights&,uint64_t,int,int,int>())
        .def(py::init<const Weights&,const std::vector<uint64_t>&,int,int>())
        .def("progress",&Bank::progress)
        .def("boss_results",&boss_results).def("entry_health",&boss_entry_health)
        .def("recovery_phase",&recovery_phase).def("recovery_features",&recovery_features)
        .def("recovery_card_trials",&recovery_card_trials)
        .def("edit_practice",&edit_practice)
        .def("tactic_progress",&tactic_progress)
        .def("batch",&Bank::evaluate).def("__len__",[](const Bank &b){return b.states.size();});
    m.def("set_event_table",[](const EventTable &table){event_values=table;});
    m.def("batch",[](const Weights &w,uint64_t seed,int n,int threads){
        if(n<1 || threads<1) throw std::invalid_argument("positive batch dimensions required");
        std::vector<std::array<int,6>> out(n);
        #pragma omp parallel for num_threads(threads) schedule(dynamic,8)
        for(int j=0;j<n;++j) {
            auto r=run(w,seed+j);
            out[j]={r.floor,r.act,r.keys,r.steps,int(r.heart),int(r.truncated)};
        }
        return out;
    });
    m.def("batch_seeds",[](const Weights &w,const std::vector<uint64_t> &seeds,int threads){
        if(threads<1) throw std::invalid_argument("positive thread count required");
        std::vector<std::array<int,7>> result(seeds.size());
        #pragma omp parallel for num_threads(threads) schedule(dynamic,2)
        for(int i=0;i<int(seeds.size());++i) {
            auto r=run(w,seeds[i]);
            result[i]={r.floor,r.act,r.keys,r.steps,int(r.heart),int(r.truncated),int(r.encounter)};
        }
        return result;
    });
    m.def("trace",[](const Weights &w,uint64_t seed){std::vector<int> a;run(w,seed,&a);return a;});
    m.def("prior_batch",[](const Weights &w,py::array_t<float,py::array::c_style> observations,
                           py::array_t<uint8_t,py::array::c_style> masks,int threads){
        if(observations.ndim()!=2 || observations.shape(1)!=OBS_SIZE || masks.ndim()!=2 ||
           masks.shape(0)!=observations.shape(0) || masks.shape(1)!=N_ACTIONS || threads<1)
            throw std::invalid_argument("invalid prior batch dimensions");
        int n=int(observations.shape(0));
        const float *obs=observations.data();const uint8_t *mask=masks.data();
        py::array_t<float> result({n,N_ACTIONS});float *out=result.mutable_data();
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(int i=0;i<n;++i) {
            std::array<double,N_ACTIONS> scores;
            View view(obs+i*OBS_SIZE,w);view.all_scores(mask+i*N_ACTIONS,scores.data());
            int prior=int(std::max_element(scores.begin(),scores.end())-scores.begin());
            for(int a=0;a<N_ACTIONS;++a) out[i*N_ACTIONS+a]=mask[i*N_ACTIONS+a]?
                float(std::max(-16.,(scores[a]-scores[prior])/4.)+1.5*(a==prior)):-1e9f;
        }
        return result;
    });
    m.def("teacher_batch",[](const Weights &weights,
            py::array_t<float,py::array::c_style> observations,
            py::array_t<uint8_t,py::array::c_style> masks,int threads) {
        if(observations.ndim()!=2 || observations.shape(1)!=OBS_SIZE || masks.ndim()!=2 ||
           masks.shape(0)!=observations.shape(0) || masks.shape(1)!=N_ACTIONS || threads<1)
            throw std::invalid_argument("invalid batched observations or masks");
        int n=int(observations.shape(0));
        py::array_t<int32_t> actions(n);
        const float *obs=observations.data();
        const uint8_t *mask=masks.data();
        auto result=actions.mutable_data();
        #pragma omp parallel for schedule(static) num_threads(threads)
        for(int i=0;i<n;++i) {
            std::array<double,N_ACTIONS> scores;
            View v(obs+i*OBS_SIZE,weights);
            v.all_scores(mask+i*N_ACTIONS,scores.data());
            result[i]=int(std::max_element(scores.begin(),scores.end())-scores.begin());
        }
        return actions;
    });
    m.def("scores",[](const Weights &w,py::array_t<float,py::array::c_style|py::array::forcecast> obs,
                      py::array_t<uint8_t,py::array::c_style|py::array::forcecast> mask) {
        if(obs.size()!=OBS_SIZE || mask.size()!=N_ACTIONS) throw std::invalid_argument("observation or mask size");
        View v(obs.data(),w);
        std::vector<double> out(N_ACTIONS,-1e30);
        v.all_scores(mask.data(),out.data());
        return out;
    });
}
