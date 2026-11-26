// Each particle starts from an independent practice fight. No live Env, real seed,
// ordered draw pile or private enemy move is supplied to the decision function.
using BeliefKey=std::array<int,10>;
struct EnemyParticle {
    MonsterGroup monsters;
    MonsterEncounter encounter;
    int turn;
};
struct ParticleBucket {
    int seen=0;
    std::vector<EnemyParticle> examples;
};
static std::map<BeliefKey,ParticleBucket> belief_bank;
static std::map<BeliefKey,ParticleBucket> late_particles;
static std::array<std::vector<std::vector<int>>,3> practice_decks;
static bool use_late_particles=false;
static bool reconstruct_stasis=false;
static int belief_draws=0,belief_scope=0;
static double belief_scale=0.;
static int belief_plans=1;
static int belief_action_limit=0;
static int next_play_width=0;
static int terminal_budget=0;
static double terminal_optimism=0.;
static int public_tree_budget=0,public_tree_depth=1;
static thread_local std::array<int,5>* tree_audit=nullptr;
static bool search_zero_energy=false;
static thread_local std::array<int,4>* continuation_audit=nullptr;
static double belief_win_value=32.;
static thread_local bool belief_branch=false;
static thread_local std::array<int,5>* belief_audit=nullptr;

BeliefKey public_enemy_key(const float *o) {
    BeliefKey key{};
    for(int j=0;j<5;++j) {
        auto m=o+OBS_MONSTERS+j*MONSTER_W;
        key[2*j]=int(m[0]);
        key[2*j+1]=int(m[7])*8+int(m[4])+2*int(m[5])+4*int(m[6]);
    }
    return key;
}
