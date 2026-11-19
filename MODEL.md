# Model notes

## 18 Nov 2026 — What does the model actually choose?

The first sketch went straight to reading the game and recording runs. The question we meant to start with was one step later. Suppose we already have the hand and the enemies on screen. How do we get from that to a move?

Take a hand with Strike and Defend against two enemies. There are four choices here if both cards are playable: Strike the left enemy, Strike the right one, play Defend, or end the turn. That example helped pin down what needs a score. A Strike can be a great play against the enemy it kills and a terrible play against the one next to it.

One idea from today's discussion is to run the same little network over each choice. It would see the whole fight, the card being played, and the target. We'd use the same card encoder wherever Strike shows up in the hand, and the same enemy encoder for either target. End Turn would get a score too, with the card and target marked absent. For the whole-fight input, a first pass could summarize the hand and enemies, keep their counts, and add HP, energy, and block. We'd keep order wherever the rules make it matter.

The old MathCard model gives us a simpler option to try alongside this: one output per action slot. The shared scorer takes more work per decision. What it might buy us is that learning when to play Strike from slot one also helps when it turns up in slot four. There's an easy sanity check there. Swap two cards whose order makes no difference to the game and see whether the scores follow the cards.

The part I'm least sure about is the hand summary. Sometimes the whole reason to play one card is what it lets you do with the next one. How much of that survives the summary? If we try this design, I'd compare it with the fixed-slot version on the same fights and training budget, looking at both outcomes and time per move. We haven't run either version yet.

Still open for discussion: should picking up a card after combat use anything learned from playing that card in combat? For Strike, “can this kill something now?” and “do I want to draw another copy for the rest of the run?” can have very different answers.

## 18 Nov 2026 — Does each monster need its own model?

The two ideas that started this discussion came from the project author: give each monster a block in one big input, or give each monster its own little model. One detail to pin down is what "each monster" means. A slot for every species in the game and a row for every enemy in this fight are quite different inputs. Two enemies of the same species still need separate HP, intent, and target IDs.

A third option to consider is one row per live enemy and the same small encoder for every row. Keep the species as a feature. That lets both enemies borrow what the model has learned about HP and incoming damage while keeping track of which one a card would hit. A combat policy still needs to compare them in light of the hand and the whole fight. A species-specific model might make more sense if the job were predicting that species' next move.

An inexpensive check would be to compare a fixed-slot model with the shared encoder on fights with duplicate enemies and unfamiliar combinations. Do scores follow the intended targets when we reorder equivalent input rows? Does enemy targeting actually improve at the same training budget? Attention over the enemies is another option if interactions turn out to be hard to learn. These are suggestions from the discussion; we haven't picked or tested one yet.

## 18 Nov 2026 — What would a monster predictor actually learn?

The project author pointed out that enemy moves follow game rules. If an intent is visible, the player already has it. If the rules force the next move given the visible history, the rule itself can supply it. If several moves are still possible, knowing their probabilities doesn't tell us which one the game will draw.

That changes the question about a small model per monster. For choosing a target, a monster encoder could describe the threat posed by this enemy in this fight. The player policy would then use that description to choose an action. We don't need to train it to guess a hidden random draw. One design choice is still open: start with visible intents and past moves alone, or also supply public move rules explicitly?
