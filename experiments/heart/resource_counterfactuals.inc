// Sample one earned postcombat reward menu. Only training labels use the future simulator.
using ResourceOutcome = std::array<int, 19>;
std::vector<std::vector<ResourceOutcome>> resource_outcomes(
        const Weights &w, const std::vector<uint64_t> &seeds, int act, int threads) {
    if (act < 1 || act > 3 || threads < 1) throw std::invalid_argument("invalid resource practice");
    std::vector<std::vector<ResourceOutcome>> all(seeds.size());
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for (int ix = 0; ix < int(seeds.size()); ++ix) {
        Env e, saved;
        e.record = false;
        e.reset(seeds[ix], 20, true, CharacterClass::IRONCLAD);
        std::mt19937 rng(uint32_t(seeds[ix]) ^ 0xe87a113u);
        int count = 0, last_floor = -1;
        std::array<float, OBS_SIZE> obs;
        std::array<uint8_t, N_ACTIONS> legal;
        std::array<double, N_ACTIONS> scores;
        for (int step = 0; step < 3500 && !e.done() && e.gc.act <= act; ++step) {
            auto &g = e.gc;
            if (g.act == act && !e.inBattle && g.screenState == ScreenState::REWARDS &&
                (g.curRoom == Room::MONSTER || g.curRoom == Room::ELITE || g.curRoom == Room::BOSS) &&
                g.floorNum != last_floor) {
                last_floor = g.floorNum;
                if (std::uniform_int_distribution<int>(0, count++)(rng) == 0) {
                    saved = e;
                    saved.gc.map = std::make_shared<Map>(*g.map);
                }
            }
            e.observe(obs.data()); e.mask(legal.data());
            View(obs.data(), w).all_scores(legal.data(), scores.data());
            if (!e.step(int(std::max_element(scores.begin(), scores.end()) - scores.begin())))
                throw std::runtime_error("invalid resource practice action");
        }
        if (!count) continue;
        auto &g = saved.gc;
        std::vector<int> filled;
        for (int j = 0; j < g.potionCapacity; ++j)
            if (g.potions[j] != Potion::EMPTY_POTION_SLOT) filled.push_back(j);
        const int slot = filled.empty() ? -1 : filled[rng() % filled.size()];
        saved.observe(obs.data());
        auto &records = all[ix];
        auto collect = [&](int hp, bool remove, bool control) {
            Env branch = saved;
            branch.gc.map = std::make_shared<Map>(*g.map);
            branch.gc.curHp = hp;
            if (remove) branch.gc.discardPotionAtIdx(slot);
            auto r = advance(branch, w);
            int id = slot < 0 ? 0 : 1 + int(g.potions[slot]);
            records.push_back({ix, act, int(g.floorNum), int(g.maxHp), int(g.curHp), hp,
                               int(g.potionCount), id, int(remove), int(control), int(obs[22]),
                               int(obs[23]), int(obs[2]), r.floor, r.act, r.keys, r.encounter, r.heart, r.truncated});
        };
        collect(g.curHp, false, true);
        // Cross HP and potion interventions so spending one is identifiable at each HP level.
        for (int percent : {15, 45, 75, 95}) {
            int hp = std::max(1, int(g.maxHp) * percent / 100);
            collect(hp, false, false);
            if (slot >= 0) collect(hp, true, false);
        }
    }
    return all;
}
