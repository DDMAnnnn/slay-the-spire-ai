# Spire Combat AI

A first-party combat adapter for the original Slay the Spire and a Python policy runner. ModTheSpire loads the jar and BaseMod supplies the update hook. Game state, action enumeration, command validation, and recording are implemented in this project.

## Try one fight on Windows

1. Build with PowerShell 7 and JDK 11 or newer: `./build.ps1 -Game 'E:/SteamLibrary/steamapps/common/SlayTheSpire'`. The script uses the game's libraries and the two Workshop loaders and writes `build/SpireCombatAI.jar`. Place that jar in the game's `mods` directory.
2. Launch the original game with mods. Enable BaseMod and **Spire Combat AI** in ModTheSpire. For the first test, use a disposable run and the vanilla card and potion pool.
3. Double-click **Start combat agent.cmd** or run `py -3 agent.py play`. Choose the character, Ascension, route, and encounter in the game yourself. At a player turn, press **F8**. The sample policy can spend a Fire Potion and play cards. F8 again releases control. Avoid manual actions during automated combat.
4. The agent stops controlling the game at the end of the fight or on an unsupported selection screen. The Python process can wait for another F8 session. Close it with Ctrl+C.

## Card rules and PyTorch exercises

The [original card table](CARD_TABLE.md) is the reference for all 370 registered card IDs. Run `python check_card_table.py path/to/desktop-1.0.jar` with a JDK available to verify coverage and rule syntax. The live wiring policy runs independently of the table.

For the attention exercises, install PyTorch with `python -m pip install -r requirements-model.txt`. On the development machine, the existing CPU installation is also available in PowerShell with `$env:PYTHONPATH = (Resolve-Path build/python-deps).Path`.

## Read a recording

The game writes one UTF-8 JSONL file per session in `%USERPROFILE%/SpireCombatAI/recordings`. From the project directory in PowerShell:

```powershell
$latest = Get-ChildItem "$env:USERPROFILE/SpireCombatAI/recordings/*.jsonl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
py -3 agent.py replay $latest.FullName
```

Add `--full` to inspect every public field. Each decision has a ticket, the observed state, and its legal choices. The matching action row records what the policy requested; the next decision or finish row shows what was observed afterward. A file without a finish row is incomplete. “Combat ended” marks the engine's battle-end signal, which can include escape and precede reward collection. Replays review recorded decisions; they do not reconstruct animation frames or label a terminal HP change as a measured training reward.

## Policy interface

Replace `choose(state, options)` in `agent.py`. The mailbox lives under `%USERPROFILE%/SpireCombatAI` and assumes a single game and agent process. The adapter publishes a decision atomically, then accepts a matching ticket and one action from the current legal set. Indices are zero-based. Example commands are `play 0 1` (hand slot 0, enemy 1), `potion 0 1`, and `end`. State includes hand, energy, HP, block, powers, enemies, visible current-turn intents, potions, relics, deck and piles. The draw pile is sorted, the seed is omitted, and Runic Dome suppresses intent fields. Combat submenus such as choosing a card to exhaust require manual continuation in this prototype.

## 18 Nov 2026 — First connection check

I wanted to own the boundary between the model and the game. The first check uses the real Java mailbox and recorder, a Python agent, and a synthetic decision. It checks stale-ticket rejection, a valid reply, a completed recording, and cleanup. That tells me the two processes agree on the protocol. A real fight is the separate check for game timing, potion effects, and the final observed state. No policy has been trained yet.
