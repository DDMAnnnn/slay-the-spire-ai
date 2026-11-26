// The learned head reads legal combat options as a set of feature rows. Q, K
// and V refer to those same rows; card positions never enter the computation.
constexpr int ATT_WIDTH=24,ATT_KEY=8;
using AttentionHidden=std::array<float,ATT_WIDTH>;
using AttentionKey=std::array<float,ATT_KEY>;
using AttentionProj=std::array<AttentionHidden,ATT_KEY>;
struct ChoiceAttention {
    std::vector<AttentionHidden> items,foes;
    std::array<CurriculumFeatures,ATT_WIDTH> input;
    AttentionHidden bias;
    AttentionProj query,key,value;
    std::array<AttentionKey,ATT_WIDTH> mix;
    AttentionHidden output;
    double scale=0.;
};
static ChoiceAttention choice_attention{};


std::vector<int> distinct_combat_actions(int prior,const float *obs,const uint8_t *mask) {
    std::vector<int> actions{prior};
    for(int a=0;a<=ACT_END_TURN;++a) {
        if(!mask[a] || a==prior || (a>=ACT_DISCARD && a<ACT_END_TURN)) continue;
        if(a<ACT_POTION) {
            const auto h=obs+OBS_HAND+(a/5)*HAND_W;
            bool duplicate=false;
            for(int b:actions) if(b<ACT_POTION && a%5==b%5)
                duplicate|=std::equal(h,h+HAND_W,obs+OBS_HAND+(b/5)*HAND_W);
            if(duplicate) continue;
        }
        actions.push_back(a);
    }
    return actions;
}
