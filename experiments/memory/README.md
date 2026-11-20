# What did the interpreter remember about Pommel Strike?

I wanted to see what happens the first time an Ironclad interpreter meets Pommel Strike. I let it read the public card text. I withheld its catalogue program and all Pommel experience until the adaptation stage. I measured predictions on fresh fights before adding any of those test observations to memory.

## What changed

The interpreter predicts **public changes following an action**. A cue contains the object's description, type and visible context. Its ID and upgrade level address persistent memory. A text-and-state network learns a prior prediction from starter fights. Cross-attention retrieves earlier action windows and transfers the difference between what happened in those windows and what the prior expected. The prediction and its learned representation feed the existing joint action scorer. The experiment evaluates the interpreter directly.

The memory checkpoint contains the weights and the observed episodes. Each episode carries a source fight and action index, its cue, and changes to HP, Block, powers, energy and pile contents, including counts of individual card IDs. The store accepts new identities and observed fields. The current ingestion path handles plays and potion uses with a visible next observation on the same turn. Delayed consequences and attribution to passive relics remain open.

## How the fights were collected

I used the [sts_lightspeed combat simulator](https://github.com/gamerpuppy/sts_lightspeed/tree/7476a81954020087da31d41d16fddf475746ec2d) at A20. Four encounters rotate across seeds: Cultist, Jaw Worm, two Louses and small Slimes. Every fight begins with the Ironclad starting deck, including Ascender's Bane. The adaptation and Pommel test decks add one unupgraded Pommel Strike. The collector samples uniformly among legal card–target plays and ends the turn when none are available. The simulator handles the effects. The interpreter gets public snapshots, sorted piles and printed card descriptions. Seeds and hidden draw order are excluded from model inputs.

| Data | Seeds | Use |
|---|---|---|
| 96 starter fights | 1000–1095 | Fit shared weights on 1,256 usable actions, including 11 plays of Slimed generated during combat. |
| 16 further starter fights | 2000–2015 | Build a persistent bank of 168 episodes. |
| 40 starter + Pommel fights | 3000–3039 | Take the first eligible Pommel example from each distinct fight until 16 examples are available. |
| 40 final starter fights | 6000–6039 | Test the effects of Strike, Defend and Bash. |
| 60 final starter + Pommel fights | 7000–7059 | Score the same 54 eligible Pommel plays from 45 fights at each memory milestone. |

For each training query, the support pool comes from earlier fights. Some updates mask the entire pool or the queried identity. Training uses 300 AdamW updates, batches of 32 and seed 19. After training I freeze the weights. At 0, 1, 4 and 16 Pommel observations, only the contents of memory change. I exclude end-turn, terminal and incomplete observations from this immediate-effect experiment.

## What the first run changed

The first version pooled all memories before applying attention. On the pilot test seeds 5000–5059, it predicted 0.38 arrivals on average after 16 Pommel observations alongside starter memories. I changed the retrieval rule: attend separately to exact-card and related-card records, then weight the related pool as one prior observation against the retrieved direct observations (up to twelve recent records per identity). This is a chosen bias towards direct evidence. I moved the final tests to untouched seeds before evaluating the revision.

## Starter cards

Mean absolute error (MAE) is measured in the units of the indicated effect. Smaller is better. “Shared weights” has all episodic memory masked. The last column uses those same weights with the starter bank enabled.

| Card and measured change | Plays | Untrained, empty | Shared weights | Shared weights + starter bank |
|---|---:|---:|---:|---:|
| Strike, target HP | 256 | 3.21 | 0.56 | 0.55 |
| Defend, player Block | 229 | 3.03 | 0.11 | 0.12 |
| Bash, enemy Vulnerable | 36 | 1.87 | 0.09 | 0.07 |
| Bash, target HP | 36 | 4.89 | 1.11 | 1.43 |

Shared training accounts for most of the starter-card performance. In particular, episodic retrieval increased Bash's HP error. For an independent check I scored 15 eligible plays from one previously recorded real-game fight with the starter bank. Strike's target HP MAE was 1.14 across seven plays; Defend's Block MAE was 0.34 across seven. There was one Bash play. The old recording lacks card descriptions, so for unupgraded actions I supplied the same public English card text as in the simulator adapter.

## Pommel Strike

All 54 evaluated uses brought one card into the hand after accounting for the copy played. I compute arrivals from card multisets so an unchanged hand size still captures the draw. This number describes what was observed; in other decks, creation or a return from discard can also add a card to the hand.

| Prior Pommel plays | Mean predicted arrivals with starter bank | Raw arrival MAE | Target HP MAE |
|---:|---:|---:|---:|
| 0 | −0.04 | 1.04 | 2.27 |
| 1 | 0.48 | 0.52 | 0.75 |
| 4 | 0.80 | 0.20 | 0.98 |
| 16 | 0.92 | 0.08 | 0.70 |

The prediction head uses unconstrained regression, so its near-zero first prediction can be slightly negative. After four examples, rounding its prediction gave one arrival on every tested play. This test cannot distinguish correctly inferring “draw” from remembering “one arrival typically follows this card.” Draw prevention, upgrades and delayed interactions would distinguish stronger claims.

At zero Pommel experience the untrained model's HP MAE was 5.54. Starter-trained weights with an empty bank reached 2.25; enabling starter memories gave 2.27. At one Pommel example the model with those shared weights and **only the new Pommel memory** predicted 1.01 arrivals (MAE 0.01). Adding the starter bank reduced the mean to 0.48. The initial attack transfer comes mostly from the shared weights. The episodic mechanism quickly copies a novel observable effect; the prior from starter cards slows its adoption in this case. Combat win rate was outside this experiment.

## Reproduce

The experiment needs PyTorch, Git, a C++17 GCC toolchain and the upstream simulator's pinned JSON submodule. From the project root on Windows, with the chosen compiler and its runtime on PATH:

```powershell
$python = '.\.venv\Scripts\python.exe'
git clone https://github.com/gamerpuppy/sts_lightspeed.git build/sts_lightspeed
git -C build/sts_lightspeed checkout 7476a81954020087da31d41d16fddf475746ec2d
git -C build/sts_lightspeed submodule update --init json
& $python -m experiments.memory.build_collector build/sts_lightspeed build/memory-final/collect.exe
& $python -m experiments.memory.run build/memory-final --collect build/memory-final/collect.exe --recordings "$env:USERPROFILE/SpireCombatAI/recordings"
& $python -m unittest discover -s tests -v
```

The recordings option is optional. The runner writes split traces, full metrics, a starter checkpoint and an adapted checkpoint. It reloads the adapted checkpoint and repeats held-out scoring.

To inspect the existing C-model with memory, use `python -m battle inspect POLICY.pt RECORDING.jsonl --memory build/memory-final/adapted.pt`. Live play also accepts `--memory` and `--save-memory OUTPUT.pt`. Confirmed same-turn actions enter memory before scoring the next decision. On restart, use that saved memory checkpoint to continue learning. The interpreter experiment freezes the combat policy and does not establish whether adding the new embeddings improves its choices.
