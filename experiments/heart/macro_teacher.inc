// Offline upper-bound experiment. Every candidate is a legal public choice;
// its label comes from a separate full continuation, never from a policy input.
using MacroOutcome = std::array<int, 5>;
MacroOutcome macro_outcome(const Result &r) {
    return {int(r.encounter), int(r.act >= 4), r.floor, r.keys, int(!r.truncated)};
}

struct MacroExample {
    int step, action, item;
    CurriculumFeatures features;
    MacroOutcome outcome;
};

auto macro_teacher(const Weights &weights, uint64_t start, int count, int threads) {
    if (count < 1 || threads < 1) throw std::invalid_argument("invalid teacher dimensions");
    std::vector<std::array<int, 10>> summary(count);
    std::vector<std::vector<MacroExample>> episodes(count);
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (int i = 0; i < count; ++i) {
        Env state;
        state.record = false;
        state.reset(start + i, 20, true, CharacterClass::IRONCLAD);
        Env baseline = state;
        Result initial = advance(baseline, weights);
        MacroOutcome guaranteed = macro_outcome(initial);
        auto &stats = summary[i];
        stats[0] = initial.floor;
        stats[1] = initial.encounter;
        std::array<float, OBS_SIZE> observation;
        std::array<uint8_t, N_ACTIONS> mask;
        std::array<double, N_ACTIONS> scores;
        int step = 0;
        for (; step < 3500 && !state.done() && !guaranteed[0]; ++step) {
            state.observe(observation.data());
            state.mask(mask.data());
            View view(observation.data(), weights);
            view.all_scores(mask.data(), scores.data());
            int prior = int(std::max_element(scores.begin(), scores.end()) - scores.begin());
            int chosen = prior;
            // Include leaving a shop or skipping a reward when that is the incumbent move.
            if (!state.inBattle && std::count(mask.begin()+ACT_CHOOSE, mask.end(), uint8_t(1)) >= 1) {
                std::vector<int> candidates{prior};
                for (int a = ACT_CHOOSE; a < N_ACTIONS; ++a)
                    if (mask[a] && a != prior) candidates.push_back(a);
                if (candidates.size() > 1) {
                    ++stats[6];
                    const auto incumbent = guaranteed;
                    for (int a : candidates) {
                        auto label = incumbent;
                        if (a != prior) {
                            Env branch = state;
                            if (!branch.step(a)) throw std::runtime_error("invalid macro alternative");
                            auto result = advance(branch, weights, nullptr, 0, false, 0, 3499-step);
                            label = macro_outcome(result);
                            ++stats[7];
                        }
                        auto candidate = view.candidate(a, std::max(-10., (scores[a]-scores[prior])/CURRICULUM_TEMP));
                        episodes[i].push_back({step,a,candidate.item,candidate.features,label});
                        if (label > guaranteed) {
                            guaranteed = label;
                            chosen = a;
                        }
                    }
                    stats[8] += chosen != prior;
                }
            }
            if (!state.step(chosen)) throw std::runtime_error("invalid teacher move");
        }
        // Replay the promised continuation with the ordinary policy to verify the label.
        Result final = advance(state, weights, nullptr, 0, false, 0, 3500-step);
        stats[2] = final.floor;
        stats[3] = final.encounter;
        stats[4] = final.heart;
        stats[5] = final.truncated;
        stats[9] = macro_outcome(final) != guaranteed;
    }
    size_t total = 0;
    for (const auto &rows : episodes) total += rows.size();
    constexpr int width = CURRICULUM_INPUT+9;
    py::array_t<float> result({py::ssize_t(total),py::ssize_t(width)});
    float *p = result.mutable_data();
    for (int i = 0; i < count; ++i) for (const auto &r : episodes[i]) {
        *p++ = float(i); *p++ = float(r.step); *p++ = float(r.action); *p++ = float(r.item);
        p = std::copy(r.outcome.begin(), r.outcome.end(), p);
        p = std::copy(r.features.begin(), r.features.end(), p);
    }
    return std::make_pair(std::move(result),std::move(summary));
}
