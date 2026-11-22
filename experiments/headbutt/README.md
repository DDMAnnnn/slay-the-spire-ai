# Headbutt on the RL fork

Run from the project root with the frozen checkpoint from the [combat experiment](../combat/README.md). The fork supplies the card facts, public observation, legal mask and game transitions. Our existing policy scores the legal actions. This first adapter covers Ironclad hand plays, end turn and Headbutt's discard selection. It raises an error if it encounters an unsupported prompt or legal action, including potion actions.

## Build on Windows

Use an x64 Visual Studio C++ toolchain, CMake on PATH and the project's Python 3.11 environment. This recipe links the fork's unmodified core and Python binding source using pybind11 2.13.6. A fresh checkout needs the JSON submodule.

```powershell
git clone --no-checkout https://github.com/bpblakely/sts_lightspeed.git build/rl-fork
git -C build/rl-fork checkout --detach 482b3ef29c93746ada7d6fd0fd16885d7cc30b1f
git -C build/rl-fork submodule update --init --recursive
.venv/Scripts/python.exe -m pip install numpy==2.4.6 pybind11==2.13.6
cmake -S build/rl-fork -B build/fork-native -A x64 -DSTS_PYTHON=OFF
cmake --build build/fork-native --config Release --target sts_core --parallel
$py = (Resolve-Path .venv/Scripts/python.exe).Path
$bindings = & $py -c "import pybind11; print(pybind11.get_cmake_dir())"
$fork = (Resolve-Path build/rl-fork).Path
$core = (Resolve-Path build/fork-native/Release/sts_core.lib).Path
cmake -S experiments/headbutt -B build/fork-binding -A x64 "-DPython_EXECUTABLE=$py" "-Dpybind11_DIR:PATH=$bindings" "-DFORK_SOURCE:PATH=$fork" "-DSTS_CORE_LIBRARY:FILEPATH=$core"
cmake --build build/fork-binding --config Release --target sts_env
$env:PYTHONPATH = (Resolve-Path build/fork-binding/Release).Path
.venv/Scripts/python.exe -m experiments.headbutt.run
```

## What the replay checks

Seed 8 starts at A20 with one Headbutt added to Ironclad's deck. Six fixed legal actions get the replay to a hand with two targetable enemies and two distinct card types in discard. The test position was selected to exercise both decisions with the existing checkpoint. Every candidate at both measured decisions comes from the fork's mask.

1. Each playable card and target gets its own action score. Headbutt on the small Spike Slime wins. The fork resolves nine damage and spends one energy.
2. The fork opens its Headbutt prompt. The player row now names the pending choice. Each legal discard slot supplies the source card row to the same scorer. The target field is empty. To reuse the checkpoint, this uses the existing card-action embedding. This checkpoint has never trained on retrieval decisions.
3. Strike wins. The replay ends the turn and checks the first card in the new hand. In a copy of the same post-hit position it chooses Defend and checks the first draw again. Copies of a given card are equivalent in this fixture. The scoring path uses only public observations, card facts, and the mask. The cloned branch is used only for verification.

The reported numbers are policy logits. The experiment tests whether the decisions reach the simulator correctly. Measuring the strategic value of retrieval will require training and evaluating on fights with these follow-up choices.
