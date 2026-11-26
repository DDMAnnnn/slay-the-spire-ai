// Every row is a paired change from the identical public pre-boss position.
// Modified decks, RNG states and future outcomes stay inside the training driver.
using EditSample=std::array<float,7>;
std::vector<EditSample> edit_practice(const Bank &bank,const Weights &w,uint32_t seed,int samples,int threads) {
    std::vector<C> cards;
    for(int j=0;j<CARD_COUNT;++j) {
        C id=C(j);
        auto rarity=getCardRarity(id);
        if(getCardColor(id)==CardColor::RED && (rarity==CardRarity::COMMON ||
                rarity==CardRarity::UNCOMMON || rarity==CardRarity::RARE)) cards.push_back(id);
    }
    if(samples<1 || samples>int(cards.size()) || threads<1) throw std::invalid_argument("invalid edit practice");
    std::vector<std::vector<EditSample>> samplesByState(bank.states.size());
    #pragma omp parallel for num_threads(threads) schedule(dynamic,2)
    for(int i=0;i<int(bank.states.size());++i) {
        const Env &entry=bank.states[i];
        if(entry.inBattle || entry.gc.screenState==ScreenState::CARD_SELECT) continue;
        int act=entry.gc.act;
        std::array<float,OBS_SIZE> obs;
        entry.observe(obs.data());
        View v(obs.data(),w);
        auto value=[&](Env &e) {
            Result r=advance(e,w);
            if(act==3) return .6*(r.floor>=51)+1.*(r.act==4)+1.*r.encounter+.2*r.heart;
            int nextBoss=act==1?33:50;
            return 1.*(r.act>act)+.4*(r.floor>=nextBoss)+.6*(r.act>act+1)+r.encounter;
        };
        Env reference=entry;
        double base=value(reference);
        auto record=[&](int op,int id,bool dup,bool up,Env &branch) {
            samplesByState[i].push_back({float(act-1),float(op),float(id),float(dup),
                float(up),float(v.missing),float(value(branch)-base)});
        };
        std::mt19937 rng(seed+i);
        auto alternatives=cards;
        std::shuffle(alternatives.begin(),alternatives.end(),rng);
        for(int j=0;j<samples && entry.gc.deck.size()<Deck::MAX_SIZE;++j) {
            C id=alternatives[j]; bool up=rng()%2==0;
            Env e=entry; e.gc.deck.obtainRaw(Card(id,up));
            record(AddCard,int(id),v.num(id)>0,up,e);
        }
        std::array<std::array<bool,2>,CARD_COUNT> seen{};
        for(int j=0;j<entry.gc.deck.size();++j) {
            const Card c=entry.gc.deck.cards[j];
            int id=int(c.id);
            if(seen[id][c.upgraded] || entry.gc.deck.isCardBottled(j)) continue;
            seen[id][c.upgraded]=true;
            if(c.canUpgrade()) {
                Env e=entry; e.gc.deck.upgrade(j);
                record(UpgradeCard,id,v.num(c.id)>1,c.upgraded,e);
            }
            if(c.canTransform()) {
                Env e=entry; e.gc.deck.remove(e.gc,j);
                record(RemoveCard,id,v.num(c.id)>1,c.upgraded,e);
            }
        }
        if(entry.gc.potionCount<entry.gc.potionCapacity && !v.relic(R::SOZU)) {
            auto &pool=PotionPool::potionPool[int(CharacterClass::IRONCLAD)];
            std::vector<Potion> choices(std::begin(pool),std::end(pool));
            std::shuffle(choices.begin(),choices.end(),rng);
            for(int j=0;j<std::min(samples,int(choices.size()));++j) {
                Env e=entry; e.gc.obtainPotion(choices[j]);
                record(GainPotion,int(choices[j]),false,false,e);
            }
        }
    }
    std::vector<EditSample> result;
    for(auto &group:samplesByState) result.insert(result.end(),group.begin(),group.end());
    return result;
}
