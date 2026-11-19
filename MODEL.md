# Model notes

## 18 Nov 2026 — What can two cards share?

I want one set of weights to read damage wherever it appears. A hit from a card and a Lightning discharge share a quantity, but the source, target and timing can change the decision. I keep a mechanic's definition attached to the effect that references it. Adding 3 and 8 to Zap's immediate damage would teach the wrong lesson. I keep the printed order because applying Vulnerable before a hit can change its outcome.

I chose a shared effect layer followed by a recurrent card encoder. A card can have another effect without needing another set of weights. I keep the current cost and an explicit incomplete-reading flag. I still need combat experience to learn the weights that make a card useful in context.

## 18 Nov 2026 — What does the model actually choose?

When I asked about architecture, I meant the model itself. Suppose it can already read the game and I can record a run. How does it go from seeing the hand and the enemies to picking a move?

Take Strike and Defend against two enemies. If both cards are playable, I can Strike the left enemy, Strike the right one, play Defend, or end the turn. The same Strike might kill one enemy and barely scratch the other. Somehow the model has to score the card and its target together.

One suggestion I got was to score each legal move with the same small network. It would read the fight, the card, and the target. Strike would go through the same card encoder wherever it sits in the hand, and either target would go through the same enemy encoder. End Turn would have neither a card nor a target. A first pass at representing the fight could summarize the hand and enemies, keep their counts, and add HP, energy, and block. Order would still matter wherever it affects play.

My MathCard model gives me a simpler baseline: one output per action slot. Scoring moves individually costs time. What I could get in return is a model that learns something about Strike in slot one and can use it in slot four. I could check that by swapping two cards whose order makes no difference and seeing whether the scores follow them.

One question from the discussion is what gets lost in a hand summary. Sometimes I play a card because of what it lets me do with the next one. Would the model see that? I haven't tried either version yet. A fair first comparison would use the same fights and training budget and track both combat outcomes and time per move.

There's also a separate question for later. If the model learns when to play Strike, would that help it decide whether to take another Strike after combat? I can want the one in my hand right now and still skip the one on the reward screen.

## 18 Nov 2026 — Does each monster need its own model?

I started with two ideas. I could use one global model and give each monster a block in the input matrix. Or I could build a separate small model for each monster. Writing down a fight with two enemies of the same species exposes an ambiguity in my first idea. If a block stands for a species, where do I put the second enemy's HP and intent? If a block stands for an enemy on screen, each one already has its own row and can be targeted separately.

A third idea came up in the discussion: keep one row per enemy, but use the same encoder to read every row. Species would be one of the inputs. That way the model could use something it learned about HP or incoming damage across species. It would still need the hand and the rest of the fight to decide which enemy to hit. A separate model for a species would answer a different question if its job were to predict that enemy's next move.

I haven't chosen between these designs. A cheap first check would be to swap equivalent enemy rows and see whether the scores still point to the same targets. A more interesting test would be to compare fixed slots with a shared encoder on duplicate enemies and unfamiliar groups. Would sharing actually improve targeting on the same training budget? Attention also came up as an option if the simpler model struggles with interactions.


## 18 Nov 2026 — How the first combat model fits together

By *intent* I mean the monster's whole behavior pattern: moves, effects, cycles, conditions and random branches. I'll use *current-turn intent* for the one move telegraphed right now. My idea is to give each monster type a place to learn its pattern while keeping one player policy that chooses what to play. Here is a concrete version to try. It hasn't been implemented.

1. **Combat inputs.** At each player decision, read HP, block, energy, hand, public pile information, active effects, and the legal moves. For each enemy, read its visible type, HP, block, powers and current-turn intent. Keep a separate history per enemy in the encounter. Record a telegraph change or an executed enemy move when it happens. Playing three cards against an unchanged telegraph must not advance the cycle three times.
2. **Monster pattern memory.** Give each monster type and Ascension setting its own training records. As a first baseline, count next observed moves after matching the recent move sequence, including the current telegraph, in previous training fights. Back off to shorter matches when evidence is sparse. Output a distribution and the number of matching examples; an unseen type returns an unknown flag. Keep public context alongside the sequence so a later model can distinguish triggers such as HP thresholds. This first table can miss conditional rules. The entire learned relation is the estimate of intent; one prediction only describes the next move. Fit from training fights and freeze the table during evaluation. The current fight's observed history can still advance.
3. **Shared player policy.** As a first baseline, concatenate a fixed-slot hand representation, player state, one enemy block and the pattern-memory output. Pass the resulting vector through a small feedforward network. It produces one score per hand slot plus End Turn. Mask illegal choices. With one enemy the target is implicit. One set of player weights serves every encounter; a new enemy can start with empty pattern memory. For multiple enemies I'll need a target-aware output. A shared candidate scorer is one option to compare at that point.
4. **Learning signal.** The pattern memory learns from observed enemy moves. The player policy learns from combat outcomes. The test reports win rate first and HP lost on wins alongside the win count. The precise training reward and update rule are still open.

I'd start with Ironclad, a fixed starting deck, one enemy species, shuffled draws and an empty potion belt. The first milestone is a complete fight on held-out seeds. Then compare two player policies with the same inputs, capacity and training budget, except that one always gets the unknown forecast. Check prediction quality against the per-type move frequencies as well as combat outcomes. A predictor could learn the cycle perfectly and still never change the best card to play in a particular fight.

If two species share a pattern, their histories may eventually be pooled. For that to justify treating them alike in combat, other relevant mechanics such as passive powers must also match or be represented in the inputs. For the prototype I can keep records separate until there is evidence for pooling.


## 18 Nov 2026 — A small prototype with a full-run design in mind

I want to start with the Ironclad on A20. The prototype can be one fight. The design has to leave room for the decisions I care about across a whole run. One change from the earlier sketch: using a potion belongs in the combat model from the start.

I see two timescales. In combat I choose a card and target, use a potion (with a target if needed), or end the turn. Outside combat I build the deck, choose rewards and potions, spend gold, decide whether to rest or upgrade, and choose a path through events, campfires, and elites. These choices draw on the same HP, deck, inventory, gold, and visible map. I haven't decided that each one needs its own network.

One interface suggested in the discussion is to evaluate what a fight leaves me with. The combat outcome should retain survival, remaining HP, remaining potions, and any other persistent changes. A later run-level model could estimate how valuable that resulting state is given the deck and visible path ahead. This matters for potions: keeping one can be valuable, but the cost of spending it changes with the run. Rest versus upgrade and route choice should likewise be evaluated in the context of the same run.

For a first test I could use one enemy, the starter deck, varied draws, and a single targeted potion type. Offer the legal potion action alongside the cards. Compare fights with and without a potion available, recording use, survival, and HP remaining. A one-fight test tells me whether the agent can make tactical use of a potion. To test whether it learns when to save one, I need a consequence after this fight, such as carrying the potion into a second encounter. The monster intent memory and its no-forecast comparison from the earlier sketch still fit here.

Once the A20 Ironclad policy works, I want to see what happens if I lower the Ascension. A change in win rate would reflect both the change in difficulty and any change in enemy behavior. I could freeze the policy for the first test and compare encounters with changed patterns against those with unchanged patterns. Other characters are a separate extension. The full-run victory condition (Act 3 or the Heart) is still open.
