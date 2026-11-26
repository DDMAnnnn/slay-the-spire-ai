Bank recovery_phase(const Bank &source,const Weights &w,int decisions,int threads) {
    if(decisions<0 || threads<1) throw std::invalid_argument("invalid recovery phase");
    Bank result;
    result.states=source.states;
    std::vector<int> keep(result.states.size());
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for(int i=0;i<int(result.states.size());++i) {
        Env &e=result.states[i];
        std::array<float,OBS_SIZE> obs;
        std::array<uint8_t,N_ACTIONS> mask;
        std::array<double,N_ACTIONS> scores;
        int combatDecisions=0;
        for(int k=0;k<3500 && !e.done();++k) {
            e.observe(obs.data()); e.mask(mask.data());
            if(obs[4]!=2 || obs[3]>33) break;
            View v(obs.data(),w);
            int slot=0;
            if(obs[3]==33 && obs[7] && mask[ACT_END_TURN] && v.boss_index(slot)>=0 &&
               combatDecisions++==decisions) { keep[i]=1; break; }
            v.all_scores(mask.data(),scores.data());
            if(!e.step(int(std::max_element(scores.begin(),scores.end())-scores.begin()))) break;
        }
    }
    size_t count=0;
    for(size_t i=0;i<keep.size();++i) if(keep[i]) {
        if(count!=i) result.states[count]=std::move(result.states[i]);
        ++count;
    }
    result.states.resize(count);
    return result;
}

std::vector<RecoveryInput> recovery_features(const Bank &bank,const Weights &w) {
    std::vector<RecoveryInput> rows;
    std::array<float,OBS_SIZE> obs;
    for(const Env &e:bank.states) {
        e.observe(obs.data());
        View v(obs.data(),w);
        int slot=0,boss=v.boss_index(slot);
        if(boss<0) throw std::invalid_argument("expected active boss");
        rows.push_back(v.recovery_input(boss,slot));
    }
    return rows;
}

// Intervene on one card at floor 32. Skip and all additions share the same
// pre-boss state. obtainRaw isolates deck contents from pickup relic triggers.
std::vector<std::array<int,5>> recovery_card_trials(const Bank &bank,const Weights &w,
                                                     uint32_t seed,int samples,int threads) {
    std::vector<C> pool;
    for(int i=0;i<CARD_COUNT;++i) {
        C id=C(i);
        auto r=getCardRarity(id);
        if(getCardColor(id)==CardColor::RED &&
           (r==CardRarity::COMMON || r==CardRarity::UNCOMMON || r==CardRarity::RARE)) pool.push_back(id);
    }
    if(samples<1 || samples>int(pool.size()) || threads<1) throw std::invalid_argument("invalid paired card trial");
    std::vector<std::array<int,5>> rows(bank.states.size()*samples);
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4)
    for(int i=0;i<int(bank.states.size());++i) {
        const Env &entry=bank.states[i];
        std::array<float,OBS_SIZE> obs;
        entry.observe(obs.data());
        View v(obs.data(),w);
        auto won=[&](C id,bool up) {
            Env e=entry;
            if(id!=C::INVALID) e.gc.deck.obtainRaw(Card(id,up));
            return advance(e,w,nullptr,0,false,34).floor>=34;
        };
        const bool reference=won(C::INVALID,false);
        auto cards=pool;
        std::mt19937 rng(seed+i);
        std::shuffle(cards.begin(),cards.end(),rng);
        for(int j=0;j<samples;++j) {
            C id=cards[j]; bool up=rng()%2==0;
            rows[i*samples+j]={int(id),int(v.num(id)>0),int(up),int(reference),int(won(id,up))};
        }
    }
    return rows;
}
