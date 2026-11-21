// Seeded public observations using gamerpuppy/sts_lightspeed at 7476a819.
#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "game/GameContext.h"
#include "combat/BattleContext.h"
#include "sim/search/Action.h"
#include "nlohmann/json.hpp"

using namespace sts;
using json = nlohmann::json;
using Move = sts::search::Action;
using Kind = sts::search::ActionType;

json card(const CardInstance& c) {
    return {{"id", getCardStringId(c.id)}, {"upgrades", c.getUpgradeCount()}, {"cost", int(c.costForTurn)}};
}

template<class Pile>
json bag(const Pile& pile) {
    std::vector<json> cards;
    for (const auto& c : pile) cards.push_back(card(c));
    std::sort(cards.begin(), cards.end());
    return cards;
}

void power(json& owner, const char* id, int amount) {
    if (amount) owner["powers"].push_back({{"id", id}, {"amount", amount}});
}

json snapshot(const BattleContext& b) {
    auto empty = json::array();
    json p = {{"current_hp", b.player.curHp}, {"max_hp", b.player.maxHp}, {"block", b.player.block},
              {"class", "IRONCLAD"}, {"ascension", 20}, {"powers", empty}, {"potions", empty}};
    power(p, "Weak", b.player.getStatus<PlayerStatus::WEAK>());
    power(p, "Frail", b.player.getStatus<PlayerStatus::FRAIL>());
    power(p, "Strength", b.player.strength);
    power(p, "No Draw", b.player.hasStatus<PlayerStatus::NO_DRAW>() ? 1 : 0);
    json s = {{"player",p}, {"energy", b.player.energy}, {"turn", b.turn}, {"hand",empty},
              {"draw",bag(b.cards.drawPile)}, {"discard",bag(b.cards.discardPile)},
              {"exhaust",bag(b.cards.exhaustPile)}, {"monsters",empty}};
    for (int i=0; i<b.cards.cardsInHand; ++i) s["hand"].push_back(card(b.cards.hand[i]));
    for (int i=0; i<b.monsters.monsterCount; ++i) {
        const auto& e = b.monsters.arr[i];
        json m = {{"id",e.getName()}, {"current_hp",std::max(0,e.curHp)}, {"max_hp",e.maxHp},
                  {"block",e.block}, {"powers",empty}};
        power(m,"Vulnerable",e.vulnerable);
        power(m,"Curl Up",e.getStatus<MonsterStatus::CURL_UP>());
        s["monsters"].push_back(m);
    }
    return s;
}

int main(int argc, char** argv) {
    const bool paired = argc == 5 && std::string(argv[4]) == "--paired-draw";
    const bool policy = argc == 5 && std::string(argv[4]) == "--policy";
    if (argc != 4 && !paired && !policy) return 2;
    const int start = std::stoi(argv[1]), count = std::stoi(argv[2]), addPommel = std::stoi(argv[3]);
    const MonsterEncounter enemies[] = {MonsterEncounter::CULTIST, MonsterEncounter::JAW_WORM,
                                        MonsterEncounter::TWO_LOUSE, MonsterEncounter::SMALL_SLIMES};
    for (int seed=start; seed<start+count; ++seed) {
        GameContext g(CharacterClass::IRONCLAD, seed, 20);
        g.floorNum = 1;
        if (addPommel) g.obtainCard(CardId::POMMEL_STRIKE);
        BattleContext b;
        b.init(g, enemies[seed%4]);
        b.executeActions();
        std::mt19937 rng(seed ^ 0x7bec9);
        for (int step=0; b.outcome==Outcome::UNDECIDED; ++step) {
            if (step >= 500 || b.inputState!=InputState::PLAYER_NORMAL) return 3;
            std::vector<Move> legal;
            for (int i=0; i<b.cards.cardsInHand; ++i) {
                int targets = b.cards.hand[i].requiresTarget() ? b.monsters.monsterCount : 1;
                for (int j=0; j<targets; ++j) {
                    Move play(Kind::CARD,i,j);
                    if (play.isValidAction(b)) legal.push_back(play);
                }
            }
            Move move(Kind::END_TURN);
            if (policy) {
                auto state = snapshot(b);
                state["player"]["relics"] = json::array({{{"id","Burning Blood"}}});
                for (int i=0; i<b.monsters.monsterCount; ++i) {
                    auto& row = state["monsters"][i];
                    const auto& monster = b.monsters.arr[i];
                    auto info = monster.getMoveBaseDamage(b);
                    row["targetable"] = monster.isTargetable();
                    row["intent"] = info.attackCount > 0 ? "ATTACK" : "NON_ATTACK";
                    if (info.attackCount > 0) {
                        row["damage"] = monster.calculateDamageToPlayer(b, info.damage);
                        row["hits"] = info.attackCount;
                    }
                    power(row,"Strength",monster.getStatus<MonsterStatus::STRENGTH>());
                    power(row,"Ritual",monster.getStatus<MonsterStatus::RITUAL>());
                }
                json options = json::array();
                for (const auto& play : legal) {
                    int i = play.getSourceIdx();
                    std::string action = "play " + std::to_string(i);
                    if (b.cards.hand[i].requiresTarget()) action += " " + std::to_string(play.getTargetIdx());
                    options.push_back(action);
                }
                options.push_back("end");
                std::cout << json({{"state",state},{"options",options}}).dump() << std::endl;
                int choice;
                if (!(std::cin >> choice) || choice < 0 || choice >= int(options.size())) return 4;
                if (choice < int(legal.size())) move = legal[choice];
                move.execute(b);
                continue;
            }
            if (!legal.empty()) move=legal[std::uniform_int_distribution<int>(0,legal.size()-1)(rng)];
            auto state = snapshot(b);
            std::string action = "end";
            if (move.getActionType()==Kind::CARD) {
                const int slot=move.getSourceIdx();
                action="play "+std::to_string(slot);
                if (b.cards.hand[slot].requiresTarget()) action+=" "+std::to_string(move.getTargetIdx());
                if (paired && b.cards.hand[slot].id == CardId::POMMEL_STRIKE &&
                    !b.player.hasStatus<PlayerStatus::NO_DRAW>() &&
                    b.cards.drawPile.size() + b.cards.discardPile.size() > 0) {
                    auto blocked = b;
                    blocked.player.setHasStatus<PlayerStatus::NO_DRAW>(true);
                    auto blockedState = snapshot(blocked);
                    move.execute(blocked);
                    move.execute(b);
                    if (b.outcome == Outcome::UNDECIDED && blocked.outcome == Outcome::UNDECIDED) {
                        std::cout << json({{"fight",seed},{"step",step},{"action",action},
                            {"allowed",{{"state",state},{"after",snapshot(b)}}},
                            {"blocked",{{"state",blockedState},{"after",snapshot(blocked)}}}}).dump() << '\n';
                        break; // at most one matched pair per fight
                    }
                    continue;
                }
            }
            move.execute(b);
            if (paired) continue;
            std::cout << json({{"fight",seed},{"step",step},{"state",state},{"action",action},
                               {"after",snapshot(b)},{"terminal",b.outcome!=Outcome::UNDECIDED}}).dump() << '\n';
        }
        if (policy) {
            std::cout << json({{"victory", b.outcome == Outcome::PLAYER_VICTORY},
                               {"hp", std::max(0,b.player.curHp)}, {"max_hp", b.player.maxHp},
                               {"fight", seed}}).dump() << std::endl;
        }
    }
}
