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
