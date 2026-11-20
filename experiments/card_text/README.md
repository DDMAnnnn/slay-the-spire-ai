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
