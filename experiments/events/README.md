# Learning event buttons from outcomes

An event decision uses the existing combat policy and candidate scorer. The player row contains the event ID, HP, maximum HP, gold, ascension, and floor. Other rows describe the visible master deck (including upgrades), relics and their visible counters, and potion slots. Each legal event button gets one source row containing an opaque joint event/option ID. Its native index stays attached for execution. Option text, prices, reward descriptors, random seeds and branch outcomes never enter the input. Inventory-use actions are outside the event button list.

## How the test works

On each A20 Ironclad seed, follow one deterministic public controller until the first ordinary Act 1 event with at least two buttons. At each legal button, clone the same starting state, press that button, and run the fixed controller until death or reaching Act 2. This controller prioritizes visible question marks, attacks the target with the lowest HP plus block, and uses fixed rules on other screens. Subsequent ordinary events take the last available button. Random outcomes within each branch use the game's simulator. A single continuation per button measures a realized outcome with that controller. It doesn't establish the best choice across future randomness or all possible controllers.

Start from the fork combat checkpoint. Fit its weights with a pairwise logistic ranking loss. The target is an Act 1 clear plus the highest floor reached capped at 17 and divided by 17. Differences in return weight the pairs. Equal-return pairs give no preference signal. This is supervised ranking on simulator outcomes. The lookup comparison remembers each event/button's mean training return and chooses the largest among available buttons (zero for unseen pairs, first legal on ties).

Training used seeds 30000–30159. There were 129 qualifying starts across 18 events. Test seeds 40000–40079 yielded 65 starts across 14 events. Neow and Match and Keep were outside the ordinary-event sample. Data collection kept the qualifying starts before seeing the branches' outcomes.

| Button policy | Best on 20 starts where outcomes differed | Mean floor across all 65 |
|---|---:|---:|
| Event-trained neural policy | 14 | 8.51 |
| Event/button lookup | 14 | 8.68 |
| First legal button | 10 | 8.49 |
| Original frozen combat policy | 9 | 8.37 |
| Uniform legal button (exact expectation) | 9 | 8.29 |

The other 45 starts had equal measured outcomes for every legal choice. No test branch cleared Act 1 under the scripted continuation. On seed 40007, *We Meet Again!* offered slots 0, 1, 2 and 3. The neural policy picked 2, reaching floor 16. The lookup picked 0, reaching floor 14. This is one recorded example. Across all starts the lookup reached a slightly higher mean floor.

## Playing the whole act

A separate test began 16 fresh runs per arm on seeds 50000–50015. In one arm the event-trained checkpoint chose ordinary event buttons and played combat. In the other, the lookup chose the event buttons and the same checkpoint played combat. Both used scripted choices elsewhere, including the opening bonus, map and rewards. The neural arm cleared Act 1 on 0/16 and reached mean floor 8.69. The lookup cleared 1/16 and reached mean floor 8.62. The arms reached the same floor on 14 of 16 seeds. On seed 50015 both were truncated on floor 3 after repeating a combat exhaust-selection grid. Truncations count as unfinished and remain in the floor average. The small sample and sparse clear signal leave the value of state conditioning open.

## Reproduce

Use the [fork setup](../headbutt/README.md) and [combat checkpoint](../combat/README.md), then run from the project root:

```powershell
$env:PYTHONPATH = (Resolve-Path build/fork-binding/Release).Path
.venv/Scripts/python.exe -m experiments.events.run
```

The runner writes a separate `build/event-policy/events.pt` checkpoint and `results.json` with per-button continuations and held-out run logs. Event training updates shared weights. The whole-act comparison holds those weights fixed across arms and measures the choice of event policy. It does not measure changes in combat strength relative to the original checkpoint.
