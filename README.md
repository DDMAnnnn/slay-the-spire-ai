# Slay the Spire AI

A reinforcement learning project that plays Ironclad on Ascension 20 in *Slay the Spire*. The goal is to defeat the Corrupt Heart from the ordinary starting deck. The agent chooses cards and targets in combat and makes deck, potion, shop, rest, event, and route decisions using player-visible information.

## Approach

Training compares alternative actions from the same game position. The choice determines how far each comparison runs.

- **Combat.** Card mechanics and a short lookahead over the visible hand give each play an initial score. The simulator tries alternative cards and targets, then finishes each fight with the same policy. A small PyTorch network learns from differences in survival and remaining HP to adjust the scores.
- **Card rewards and events.** For a card reward, skipping gives a reference for measuring the value of adding each offered card. For an event, the agent learns preferences by comparing the downstream outcomes of its available buttons.
- **Late-game practice.** Training revisits naturally reached Act 2 and Act 3 positions with the decks, relics, and HP earned along the way. Full-run evaluation starts at Neow.

## Results

On the same 4,096 validation starts, adding learned card and event preferences increased Act 3 arrivals from **25 to 52**. Two frozen policy versions defeated the Heart **three times across 524,288 fresh simulator starts**. The first winning action sequence also defeated the Heart in the original game, finishing with **22/84 HP**. The real game and simulator briefly differed in their card piles on floor 22 before coming back into sync.

[Checkpoint and replay instructions](experiments/heart/README.md) · [Evaluation counts and seeds](experiments/heart/results/evaluation.json) · [Development notes](MODEL.md) · [Live combat adapter](LIVE_COMBAT.md)

## Roadmap

1. **Make Heart wins more consistent.** Identify common losses in the replays, compare policy changes on matched seeds, and evaluate the resulting player on fresh starts.
2. **Improve decisions across fights.** Use combat outcomes to estimate whether a deck can afford an elite fight, how much HP a potion is likely to save, and when a rest is worth more than an upgrade.

**Possible generalizations.** One candidate experiment is transfer from Ironclad to Silent. Test the frozen policy first to measure zero-shot transfer, then compare fine-tuning with training from scratch under the same budget. Other open directions include transfer across Ascension levels and to modded cards with unfamiliar mechanics.

## Software credits

- **Simulator and RL interface.** gamerpuppy developed the original C++ simulator. Training uses bpblakely's fork for game transitions, card facts, player-visible observations, and legal-action masks. [Original sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed) · [Fork at the training revision](https://github.com/bpblakely/sts_lightspeed/tree/482b3ef29c93746ada7d6fd0fd16885d7cc30b1f)
- **In-game replay.** bpblakely's toolkit supplied the command conversion and playback infrastructure used to replay the winning decisions in the original game. [sts-lightspeed-toolkit at the playback revision](https://github.com/bpblakely/sts-lightspeed-toolkit/tree/f46fed59175646254b284c27e677a9b32ca5b748)

This repository implements the Ironclad policy, the training procedure that compares alternative decisions, the late-game curriculum, and the evaluation experiments.
