# Learning which move to play in a full fight

This experiment uses the existing pinned sts_lightspeed simulator to resolve complete A20 Ironclad starter-deck encounters. The combat model reads public observations and hand-authored card and status rules. Its action scores control the simulation one legal choice at a time. It can play cards on specific targets, block or end its turn early. This setup starts with Burning Blood and no potions. Reported HP is measured at combat termination before post-combat healing.

## Training and evaluation

One width-32 combat model, seeded with 19, samples actions in 256 fights (seeds 22000 through 22255). Eight fights are collected per policy update. The REINFORCE return is one point for victory plus the fraction of maximum HP remaining. Other fights in the batch provide the baseline for that episode. The auxiliary target is the signed change in target HP when the next decision still shows the same target on the same turn. The model receives no target HP label across a missing terminal observation. No decision is labelled as the correct choice.

The saved model plays greedily on 64 further seeds (44000 through 44063). All policies see the same starts: 16 Cultist, 16 Jaw Worm, 16 two-Louse and 16 small-Slime encounters. Random play chooses uniformly among legal card-target plays and ends when none remain. Attack-first takes the first legal targeted attack, then the first other play. It provides a simple benchmark for choosing to spend energy on defense. Both baselines are evaluated through the same protocol as the learned policy.

The collector omits hidden draw order and future enemy moves. Its enemy observations distinguish attacks from other intents and give visible attack damage and hit count. The game bridge has richer intent labels, card displays and object metadata. The card catalogue provides rules and numbers to the model; the simulator implements combat effects. The benchmark covers full simulated fights within four encounter types. New seeds test fresh fights of those types.

## Reproduction

Use the pinned simulator checkout and compiler from the [collector instructions](../memory/README.md). On Windows, keep the compiler runtime directory on PATH while building and running:

```powershell
.\.venv\Scripts\python.exe -m experiments.memory.build_collector build/sts_lightspeed build/combat-policy/collect.exe
.\.venv\Scripts\python.exe -m experiments.combat.run build/combat-policy/collect.exe build/combat-policy
.\.venv\Scripts\python.exe -m experiments.combat.run build/combat-policy/collect.exe build/combat-policy --evaluate-hp 20
```

The runner writes a normal battle checkpoint, `combat.pt`, and a per-fight `results.json` with terminal HP, outcome and evaluation action counts.

## First results

All three policies won every standard start. After observing that ceiling, I replayed the same 64 encounter seeds with starting HP set to 20 and the checkpoint frozen. Losses count as zero remaining HP.

| Policy | Standard wins | Standard mean HP | 20-HP wins | 20-HP mean HP |
|---|---:|---:|---:|---:|
| Random legal card play | 64/64 | 48.19 | 34/64 | 4.03 |
| Attack-first | 64/64 | 53.56 | 51/64 | 5.97 |
| Learned combat policy | 64/64 | 55.31 | 56/64 | 7.64 |

At 20 HP the learned policy won nine encounters lost by attack-first and lost four that attack-first won. Its per-encounter wins were 16/16 Cultist, 15/16 Jaw Worm, 15/16 two Louses, and 10/16 small Slimes. This measures one policy initialization and the same four starter encounters used during training. The comparison cannot attribute the advantage to blocking alone; card order and target selection also differ. As a bridge compatibility check, the saved model scored all eight legal actions in a previously recorded real-game frame.
