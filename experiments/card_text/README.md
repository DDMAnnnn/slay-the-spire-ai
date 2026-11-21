# First encounter with “Deal 10 damage twice”

The combat policy sees a synthetic one-decision fight. The player has one energy and two one-cost attack cards. One card kills the enemy through its Block; otherwise the enemy's attack ends the fight. Ending the turn loses. The fixture chooses enemy HP between the two attacks' effective totals so every example distinguishes the choices. This measures lethal-card selection, not full-game win rate.

## What the model sees

Both arms use an opaque token per ID, the same public state, current cost, card type, action mask, width-32 two-layer combat self-attention and joint action scorer. Only the text arm also sees the printed words and numeric magnitudes. Its GRU and word embeddings start from random weights. Neither arm sees a rule program, damage preview, separate damage/hit-count field, glossary label for the synthetic attacks, or episodic memory. Tokenized IDs preserve no English or numeric content. The three unfamiliar test IDs occupy different embedding buckets from the 60 training IDs.

I used 15 training mechanics: one or two hits of 4, 6, 8, 10, 12, 14, 18 or 22, omitting ten twice. Each mechanic has four fixed IDs with identical wording. The same 4,096 training fights, logged random actions and binary terminal outcomes feed each pair of models. Training fits a win score for the *logged* action. Both arms get 480 minibatches of 48, and greedy evaluation uses frozen weights. The slot order varies; the optional Strength value is 0, 2 or 4. No held-out card plays enter training.

## Results

Win rates below are averaged across three initializations evaluated on the same cases.

| Decision set | Cases per model | IDs | IDs + text |
|---|---:|---:|---:|
| Known IDs in fresh fights | 384 | 97.8% | 99.0% |
| Ten twice under a new, fixed ID against familiar alternatives | 384 | 72.7% | 88.8% |
| Both IDs unfamiliar, same state with their descriptions exchanged | 768 | 50.0% | 83.3% |

For seeds 19, 23 and 41 respectively, fixed-ID holdout wins were 192, 320 and 325 for IDs alone and 339, 320 and 364 for IDs + text, out of 384. With one familiar alternative, guessing based on that alternative or the threshold can succeed. The swap probe exposes this: across each pair, the ID-only inputs stay identical, but the correct action switches slots.

On the swap probe the text policy won every ten-twice versus ten-once and ten-twice versus twelve-twice decision across all three seeds. It won 50% against twenty-two-once. At zero Strength, 22 beats 20; at four Strength, 28 beats 26. The policy used the wording for simple choices. These results give little evidence that it learned to apply Strength per hit reliably. The fixed wording, repeated aliases, threshold-selected enemy HP and single decision make this a controlled test of composition. Arbitrary paraphrases, other mechanics and full fights remain untested.

## Reproduce

From the project root with PyTorch installed:

```powershell
.\.venv\Scripts\python.exe -m experiments.card_text.run build/card-text
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The runner saves six policy checkpoints and a results JSON with per-seed counts and the Strength breakdown. It also checks batched evaluation logits against the normal game-policy entry point after reloading the weights.

## Can the policy use Strength with only base wording?

Run `python -m experiments.card_text.strength build/strength-pairs` for both training schedules and all three seeds. This diagnostic rewards choosing the attack that deals more damage. Enemy HP stays at 100 and Block at zero. Every pair of test inputs differs only in the visible Strength status. Training supplies a binary outcome for a logged action. The model never sees adjusted damage, hit labels, or a Strength tooltip.

Both schedules use 4,096 decisions, 480 updates, identical initialization per seed, and training Strength values of 0, 2 and 4. Random practice samples hands and stacks independently before rejecting damage ties. Paired practice replaces half the samples with the same hand shown at 0 and 4, selected so the winner flips. This comparison changes both pairing and the frequency of reversals. The damage objective and fixed HP also differ from the earlier lethal-choice experiment.

The table counts a pair only if the frozen policy chose correctly at **both** stack counts. Counts aggregate seeds 19, 23 and 41. For the held-out attack each seed uses the same 32 variants (two slot orders and sixteen assignments of opaque IDs). These are variations of one matchup, not distinct card mechanics.

| Test | Random practice | Paired practice |
|---|---:|---:|
| Three familiar reversal matchups, 0 and 4 | 288/288 | 288/288 |
| Ten twice vs. twenty-two once, 0 and 4 | 96/96 | 87/96 |
| Same held-out matchup, 1 and 3 | 13/96 | 25/96 |
| Held-out card in two matchups with a fixed winner, 0 and 4 | 192/192 | 182/192 |
| Familiar six twice vs. twenty-two once, single hit always better, 0 and 4 | 96/96 | 96/96 |

The 1-and-3 follow-up used the same saved weights. At 1 Strength the totals are 22 vs. 23. At 3 they are 26 vs. 25. All six models chose correctly at 3. At 1, the per-seed counts out of 32 were 13, 0, 0 for random practice and 0, 25, 0 for paired practice. Clamping Strength to the lower value makes each reversal pair identical at the input and takes the both-correct count to zero. The observed decisions provide evidence of transfer at practiced stack counts and weak interpolation to new counts. No exact damage predictions were requested or measured.

## Training on exact damage

Run `python -m experiments.card_text.exact_damage build/exact-damage` to reproduce the twelve fitted models and their per-seed JSON reports. Each checkpoint uses the combat model's target HP loss output in HP units. The experiment trains this output by mean squared error. For ranking at evaluation it chooses the action with the greatest predicted HP loss. The network's separate policy score receives no training signal in this experiment.

The inputs, hands, random action logs, optimizer settings, minibatches and initialization seeds match the Strength ranking experiment. “Played” supplies the HP loss for the action actually logged, including zero for ending the turn. “All” supplies the simulator's HP loss for both attacks and the end-turn action. All-action feedback therefore supplies three targets per state. The test card description remains hidden throughout training. Only the base description is shown when the frozen model encounters it. The synthetic enemy has 100 HP and zero Block, so immediate HP loss equals attack damage.

Each test compares ten twice (A) with twenty-two once (B) at all five Strength values. There are 32 combinations of unfamiliar IDs and hand order, repeated for three seeds. The reports give mean predicted HP loss, mean absolute error and nearest-integer accuracy separately for each attack and stack count. They also count how many layouts rank the correct card first at both members of a Strength pair. At 2 Strength both attacks deal 24 and only the numerical predictions are assessed.

| Practice | Feedback | Mean absolute error (HP) | Correct rounded HP / 960 | Both 0 and 4 / 96 | Both 1 and 3 / 96 |
|---|---|---:|---:|---:|---:|
| Random | Played | 1.26 | 142 | 93 | 50 |
| Random | All | 0.84 | 353 | 90 | 50 |
| Paired | Played | 1.05 | 279 | 74 | 19 |
| Paired | All | 0.71 | 434 | 93 | 33 |

With random practice and played-action feedback, the mean predictions for A at Strength 0 through 4 were 19.65, 21.61, 23.96, 26.02 and 27.67 HP. For B they were 21.40, 21.79, 23.22, 24.51 and 25.53. At 1 Strength the average predicted advantage for B was only 0.18 HP, even though B wins by 1. Averages can conceal substantial errors on individual IDs. With the original binary objective the 1-and-3 pair scores were 13/96 for random practice and 25/96 for paired practice. Exact played-action feedback improved the first and reduced the second. These are repeated variants of one matchup across three seeds, so the results do not estimate general combat performance.

## Broad paired practice with exact played-action feedback

Each block contains 4,096 logged decisions. Broad paired practice draws 2,048 hands from all 105 combinations of the 15 known attack descriptions, then plays the same logged action in each hand at two distinct Strength values drawn from 0, 2 and 4. At the first block, 1,972 pairs kept the same winner, 17 reversed it and 59 included a tie. The second block had 1,942, 19 and 87 respectively. No winner-based filter is applied. Base wording, card IDs and hand order stay fixed across a pair. A pair contributes two ordinary regression examples and may occur in separate minibatches.

The matched random control uses the identical hands, IDs, logged action indices and marginal Strength counts. It shuffles Strength across examples so a repeated hand can get the same or a different value. The independent random control also redraws the second hand in each pair, representing fresh random hands under the same budget. Every arm uses exact HP loss only for the logged action, the same initialization seeds (19, 23 and 41), batch size 48 and the same sampled minibatch indices within each budget.

Three budgets answer different questions. The original budget uses 480 updates on a 4,096-record pool. The pooled run draws from both blocks (8,192 records) at the same 480-update limit, with 23,040 sampled examples in either case. The extended run continues the original checkpoint and optimizer for 480 additional updates on the second block, totalling 960 updates. It changes both the example stream and the update count. All reports use the same frozen held-out matchup and slot/ID layouts as the earlier exact-damage experiment.

Reproduce both budget settings from the repository root:

```powershell
.\.venv\Scripts\python.exe -m experiments.card_text.broad_pairs build/broad-pairs
.\.venv\Scripts\python.exe -m experiments.card_text.broad_pairs build/broad-pairs --pooled
```

| Pool / updates | Practice | Mean absolute error (HP) | Both 0 and 4 / 96 | Both 1 and 3 / 96 | Both 1 and 3 per seed / 32 |
|---|---|---:|---:|---:|---|
| 4,096 / 480 | Broad paired | 0.729 | 95 | 69 | 22, 26, 21 |
| 4,096 / 480 | Same hands, shuffled Strength | 0.839 | 94 | 51 | 24, 6, 21 |
| 4,096 / 480 | Fresh random hands | 0.931 | 75 | 12 | 10, 0, 2 |
| 8,192 / 480 | Broad paired | 0.867 | 89 | 48 | 19, 14, 15 |
| 8,192 / 480 | Same hands, shuffled Strength | 0.863 | 90 | 44 | 18, 11, 15 |
| 8,192 / 480 | Fresh random hands | 0.758 | 90 | 49 | 25, 5, 19 |
| 8,192 / 960 | Broad paired | 1.334 | 91 | 63 | 14, 29, 20 |
| 8,192 / 960 | Same hands, shuffled Strength | 0.711 | 92 | 46 | 18, 7, 21 |
| 8,192 / 960 | Fresh random hands | 0.906 | 85 | 52 | 17, 22, 13 |

All 96 cases are slot and unfamiliar-ID variants of one matchup across three model seeds. Numerical error averages both cards at all five Strength values. Paired training's early advantage largely disappears for the bigger pool at fixed compute. Extra updates on the second block produce a higher paired flip score than either control, alongside larger numerical errors.

## Does the pairing advantage repeat on other attacks?

For each fold, hold out one two-hit description completely, then train fresh models on the other 15 descriptions. The withheld base amount still appears on a single-hit card, and the test opponent's wording is familiar. Both test IDs are unfamiliar. The original 10-twice fold used the nine existing 480-update checkpoints. Re-running the generalized evaluator reproduced every field in their original reports. The 4-, 6- and 8-twice folds are new.

The test opponent deals twice the held-out base amount plus two in one hit. That gives the same one-HP disadvantage for the double hit at 1 Strength and the same one-HP advantage at 3 in every fold. A pass requires choosing correctly at both. All four matchups tie at 2. Damage error is the mean absolute error across both cards, all five Strength values and all layouts. The 32 layouts vary hand order and opaque IDs. They are repeated measurements on one matchup per fold.

Every schedule uses 4,096 records, 480 updates of 48 examples, the same three initialization seeds, and exact damage for the logged action. Strength ranges over 0, 2 and 4 in training. The matched random control shuffles Strength over the identical repeated hands, and the fresh random control redraws hands. The hand generator, sampled action slots and minibatch draws match across paired and shuffled practice within each fold. The test checks transfer of an attack description within one Strength mechanic at the original budget.

Reproduce the full panel with `python -m experiments.card_text.holdout_panel build/holdout-panel`. The JSON report for each model includes numerical error and predictions at each Strength.

| Withheld base damage (two hits) | Schedule | Correct switches / 96 | Per seed (19, 23, 41) / 32 | Mean absolute error, HP |
|---:|---|---:|---|---:|
| 4 | Broad paired | 45 | 31, 3, 11 | 0.831 |
| 4 | Shuffled Strength | 30 | 24, 3, 3 | 1.070 |
| 4 | Fresh hands | 19 | 8, 3, 8 | 2.255 |
| 6 | Broad paired | 23 | 18, 3, 2 | 0.815 |
| 6 | Shuffled Strength | 60 | 30, 16, 14 | 0.738 |
| 6 | Fresh hands | 66 | 24, 25, 17 | 0.511 |
| 8 | Broad paired | 70 | 28, 22, 20 | 0.691 |
| 8 | Shuffled Strength | 73 | 29, 21, 23 | 0.810 |
| 8 | Fresh hands | 47 | 25, 4, 18 | 1.234 |
| 10 | Broad paired | 69 | 22, 26, 21 | 0.729 |
| 10 | Shuffled Strength | 51 | 24, 6, 21 | 0.839 |
| 10 | Fresh hands | 12 | 10, 0, 2 | 0.931 |

On the three new matchups alone, paired practice passed 138/288 switches with 0.779 HP error, shuffled Strength passed 163/288 with 0.873 HP error, and fresh hands passed 132/288 with 1.333 HP error. Including the original matchup yields 207, 214 and 144 switches out of 384 and errors of 0.767, 0.864 and 1.233 HP respectively. The ordering depends on the metric and the held-out card. These runs measure damage estimates and one-action rankings within a single mechanic.
