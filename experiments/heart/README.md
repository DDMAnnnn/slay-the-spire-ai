# A20 Ironclad against the Heart

A successful episode must start at Neow with the unmodified Ironclad starter deck, collect all three keys, reach Act 4 and defeat the Heart. An Act 3 victory and a truncation both count as misses. This experiment uses the pinned native simulator from the [combat setup](../headbutt/README.md).

The actor gives every legal action a score from the public observation. It reads card facts, hand costs, enemy intents, deck composition, relics, potions, HP, gold and the visible map. It estimates reachable future rooms along map edges to plan for the emerald key. Its initial action values include hand-authored Ironclad strategy. The full-run training loop fits the relative weights from episode returns by evaluating positive and negative perturbations on matching training seeds, averaging the stronger proposals and selecting on separate validation seeds. Ordinary event preferences come from the previous neural event checkpoint averaged over its original training starts. Other screens use the action scorer in `policy.cpp`. This is a separate full-run actor. The toolkit's published Defect policy supplies no actions or weights.

Act 2 training snapshots are taken from natural, originally unmodified training runs. Each candidate resumes the same fights, with the real HP, inventory, and deck from those prefixes. The policy sees only the observation. Final evaluations always begin at Neow.

## Run locally

Once the fork core and the Python environment are built, configure this extension from the repository root. Use the same x64 Release toolchain as the core:

```powershell
$py = (Resolve-Path .venv/Scripts/python.exe).Path
$bindings = & $py -c "import pybind11; print(pybind11.get_cmake_dir())"
$fork = (Resolve-Path build/rl-fork).Path
$core = (Resolve-Path build/fork-native/Release/sts_core.lib).Path
cmake -S experiments/heart -B build/heart-binding -A x64 "-DPython_EXECUTABLE=$py" "-Dpybind11_DIR:PATH=$bindings" "-DFORK_SOURCE:PATH=$fork" "-DSTS_CORE_LIBRARY:FILEPATH=$core"
cmake --build build/heart-binding --config Release
$env:PYTHONPATH = "$((Resolve-Path build/fork-binding/Release).Path);$((Resolve-Path build/heart-binding/Release).Path)"
& $py -u -m experiments.heart.train
```

Each checkpoint directory contains its fitted weights, event table, and per-seed evaluation rows. For the first Heart victory, the verifier plays the policy again in a fresh recording-enabled simulator, compares each selected legal action to the original trace, checks the terminal fight log and saves the episode and weights in `heart-win.json`.

## Verified ordinary-start runs

The frozen checkpoint and complete action logs are in [results](results). Every release reference starts at Neow on A20 Ironclad, has Act 4 enabled, and uses the original starter deck. The verifier checks each argmax against the recorded actions, all three keys, the floor-by-floor history, and positive HP after the Corrupt Heart. The policy receives the public observation and legal-action mask. Only the training driver clones game states to label alternative decisions.

| Frozen actor | Fresh starts | Reached Act 2 | Reached Act 3 | Reached Act 4 | Heart kills | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First | 262,144 | 155,372 | 6,525 | 29 | 1 | 9 |
| Deck synergies | 262,144 | 144,218 | 5,883 | 36 | 2 | 7 |

The two panels used different, disjoint seed ranges. Wins are rare and their count is too small to rank these versions by Heart win rate. The first policy won seed 40150942. The deck-synergy variant won seeds 41105607 and 41207418.

### Watch or reproduce

After the build and PYTHONPATH setup above, these commands load the packaged weights and run the actor from the start. Open the generated HTML in a browser, then use **Jump to the Heart** to inspect the final fight. The replay lists the five highest-scoring legal actions at each decision.

```powershell
& $py -m experiments.heart.replay --out build/heart-first.html
& $py -m experiments.heart.replay --variant synergies --out build/heart-synergies.html
& $py -m experiments.heart.replay --variant synergies --seed 41207418 --out build/heart-third.html
& $py -m experiments.heart.evaluate experiments/heart/results/checkpoint.json.gz 42000000 1024 --variant first
```

### How the winning actor learns

1. The native scorer evaluates legal choices on every screen. Its Ironclad card priors and short lookahead over visible cards provide its starting strategy. The weight search in `train.py` compares candidate policies on matching seeds.
2. The choice learner in `learn_choices.py` played each sampled card or event alternative from the same training position. For cards it measured the downstream difference from skipping. For an event it measured each button against the average available button. It learned effects by act and pooled across acts for rare choices. Across 24,576 training starts, the comparison on 4,096 validation starts raised Act 3 arrivals from 25 to 52.
3. The supervised combat head in `fit_combat.py` learned from 305,013 action examples. Its target is the difference between an action's end-of-fight utility and the mean of the alternatives tried at that decision. The utility accounts for survival, HP, and fight length. Card and enemy embeddings plus 22 public numeric inputs feed a 16-unit hidden layer. Its output adjusts the hand-authored score. The earlier attention checkpoint supplied no combat weights to this actor.
4. The late-game optimizer in `late.py` fit a separate set of weights from Act 2 onward. It used 240 naturally reached Act 2 starts and 192 Act 3 starts for training, with separate banks for validation. The second variant also values specific deck combinations and repeated smithing. Both variants use the same learned checkpoint.

The frozen checkpoint contains the full-run weights, event table, learned card/event adjustments, combat head and late-act weights. `evaluation.json` in the results directory records the evaluation ranges and winning seeds. Runs are from the pinned lightspeed fork.
