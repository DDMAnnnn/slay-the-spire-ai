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
