# Slay the Spire AI

A reinforcement learning project that plays Ironclad on Ascension 20 in *Slay the Spire*. The goal is to defeat the Corrupt Heart from the ordinary starting deck. The agent chooses cards and targets in combat and makes deck, potion, shop, rest, event, and route decisions using player-visible information.

## Approach

Training compares alternative actions from the same game position. The choice determines how far each comparison runs.

- **Combat.** Card mechanics and a short lookahead over the visible hand give each play an initial score. The simulator tries alternative cards and targets, then finishes each fight with the same policy. A small PyTorch network learns from differences in survival and remaining HP to adjust the scores.
- **Card rewards and events.** For a card reward, skipping gives a reference for measuring the value of adding each offered card. For an event, the agent learns preferences by comparing the downstream outcomes of its available buttons.
- **Late-game practice.** Training revisits naturally reached Act 2 and Act 3 positions with the decks, relics, and HP earned along the way. Full-run evaluation starts at Neow.

## Progress from the first win

1. **Establish that the agent can finish a run.** Two early policies defeated the Heart three times across 524,288 fresh simulator starts. The first winning action sequence also finished in the original game with 22/84 HP. The card piles briefly diverged on floor 22 and later came back into sync. [First checkpoints and replay instructions](experiments/heart/README.md#verified-ordinary-start-runs)
2. **Check whether intermediate gains survive a whole run.** A learned boss-relic preference raised Act 3 arrivals from 748 to 911 on 32,768 matched starts. Act 4 arrivals went from three to two. The project now tracks entry into the Heart fight as its primary milestone, with kills reported separately. [Boss-relic experiment](experiments/heart/results/first-boss-relics.json)
3. **Teach the agent to read a path.** A route model learned from demonstrations and evaluated connected rooms backward through the visible map. This lets it distinguish, for example, an elite followed by a campfire from an elite followed by another fight. Adding the router to the same native player improved Heart encounters on both independent panels of 32,768 starts, from 34 to 60 and from 29 to 55. Combined kills rose from four to nine. [Map model](experiments/heart/results/learned-map.json) · [Paired full-run results](experiments/heart/results/combined-human-search.json)
4. **Test how much stronger combat can contribute.** A second simulator paired this project's strategic policy with an upstream combat planner. For each decision, the planner samples possible hidden states consistent with public information and compares legal plays in a shared search tree. The four-sample player reached the Heart five times on 256 starts and seven times on 256 fresh starts, with three kills in total. A higher-budget version reached the Heart nine times on the original panel and lost all nine fights. Its fresh-seed test is still open. [Replicated player](experiments/heart/results/reference-runtime.json) · [Seven variants on identical starts](experiments/heart/results/source-round-comparison.json)

The current target is to enter the Heart fight on **20% of ordinary A20 Ironclad starts**. The best replicated player reached **12/512 (2.34%)**. The map experiment and the combat-search experiment use different simulator builds and seed panels. Each comparison above holds the simulator fixed.

[Experiment journal and current checkpoints](experiments/heart/README.md) · [Development notes](MODEL.md) · [Live combat adapter](LIVE_COMBAT.md)

## Roadmap

1. **Replicate the higher-budget combat result.** Run the frozen settings on fresh ordinary starts and compare Heart entries and kills against the four-sample player on those same seeds.
2. **Improve decisions across fights.** Use combat outcomes to estimate whether a deck can afford an elite fight, how much HP a potion is likely to save, and when a rest is worth more than an upgrade.

**Possible generalizations.** One candidate experiment is transfer from Ironclad to Silent. Test the frozen policy first to measure zero-shot transfer, then compare fine-tuning with training from scratch under the same budget. Other open directions include transfer across Ascension levels and to modded cards with unfamiliar mechanics.

## Software credits

- **Simulator and RL interface.** gamerpuppy developed the original C++ simulator. Training uses bpblakely's fork for game transitions, card facts, player-visible observations, and legal-action masks. [Original sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed) · [Fork at the training revision](https://github.com/bpblakely/sts_lightspeed/tree/482b3ef29c93746ada7d6fd0fd16885d7cc30b1f)
- **In-game replay.** bpblakely's toolkit supplied the command conversion and playback infrastructure used to replay the winning decisions in the original game. [sts-lightspeed-toolkit at the playback revision](https://github.com/bpblakely/sts-lightspeed-toolkit/tree/f46fed59175646254b284c27e677a9b32ca5b748)
- **Public-information combat search and demonstrations.** The second simulator, combat search and published action traces come from SnyderConsulting. This project integrates its own strategic policy with that search and evaluates the combination from ordinary starts. [sts_ml at the evaluated revision](https://github.com/SnyderConsulting/sts_ml/tree/1e69c2e78a5ab3cb44cca5ffdfcb82803a68701d)
- **Human run histories.** Historical offers and choices supplied supervision for drafting and resource decisions. [MaT1g3R's run collection](https://github.com/MaT1g3R/Slay-the-Spire-data/tree/097aaf3564c2247835162d267cbc7c55d2c9039e) · [Archived winning runs](https://github.com/karlo-delacruz-ieds/sts_1_winning_percentage/tree/411c17ab53b24c3781f5a02c6a4032667bbeda3f)

This repository implements the Ironclad policy, the training procedure that compares alternative decisions, the late-game curriculum, and the evaluation experiments.
