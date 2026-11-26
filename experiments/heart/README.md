# A20 Ironclad against the Heart

A successful episode must start at Neow with the unmodified Ironclad starter deck, collect all three keys, reach Act 4 and defeat the Heart. An Act 3 victory and a truncation both count as misses. This experiment uses the pinned native simulator from the [combat setup](../headbutt/README.md).

## Current ordinary-start benchmarks

My current target is to **enter the Corrupted Heart fight on 20% of ordinary A20 Ironclad runs**. I record defeating it as a separate outcome. Every run in this table began at Neow with the starter deck.

| Frozen player | Starts | Entered the Heart fight | Defeated the Heart |
| --- | ---: | ---: | ---: |
| Local simulator, human draft plus combat and reward search | 32,768 | 34 (0.104%) | 3 |
| Same simulator, same seeds, learned map policy added | 32,768 | 60 (0.183%) | 3 |
| Local simulator, independent panel, original / learned route / route plus learned campfires | 32,768 each | 29 / 55 / 59 | 1 / 6 / 6 |
| Pinned source simulator, local strategic policy plus public shared-tree combat search | 256 | 5 (1.95%) | 1 |
| Same source player, first 256 starts on the independent panel | 256 | 7 (2.73%) | 2 |
| Higher-budget source search on ordinary combat turns, original seed panel | 256 | 9 (3.52%) | 0 |

The first two rows isolate a route change. The source simulator uses a different set of seeds and a smaller panel. Across its two fixed samples the player reached the Heart 12 times in 512 starts (2.34%). The second sample is the first 256 of a planned 32,768 starts. That longer evaluation is paused with 327 complete seed records. The higher-budget search has one completed panel. [Local simulator comparison](results/combined-human-search.json) · [Source configuration, ordinary runs and verified win](results/reference-runtime.json)

The frozen [native route checkpoint](results/learned-route-search.json.gz) reproduces seed 194002634. With the native extension on the Python path, run `python -m experiments.heart.replay --checkpoint experiments/heart/results/learned-route-search.json.gz --seed 194002634 --out build/learned-route-win.html`. For the pinned source simulator, the [strategic checkpoint](results/source-strategic-policy.json.gz) supplies the noncombat choices. After preparing both simulator bindings, run `python -m experiments.heart.strategic_source_play --model experiments/heart/results/source-strategic-policy.json.gz --mode 2 --early 384 --late 2048 --elite 2048 --boss 8192 --seed 2290664653572843552` to reproduce the verified ordinary win.

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

### Learning the first boss relic choice

One chest offered Philosopher's Stone, Busted Crown and Sacred Bark. The original actor took the Stone and died on floor 24. The actor with the learned relic scores took the Crown and reached floor 29. That's the first seed of the held-out panel, 55001000. I wanted to know whether differences like this added up across runs.

I used the first Heart-winning policy to reach 7,275 Act 1 chests from 12,288 training starts. At each chest I ran all three offers and skipping from the same saved position, then let the same policy finish each branch. I scored the run by final floor plus 6 per act reached, 2 per key and 100 for beating the Heart, with an 8-point penalty for a truncated rollout. Each relic's training label was its score minus the skip score. I averaged those differences by relic and added 30 zero-valued observations per relic to temper estimates from small samples. During play the actor adds the learned bonus to its existing score at the Act 1 chest.

I chose the bonus multiplier by Act 3 arrivals on 4,096 separate validation starts. Multipliers of 0, 1, 4 and 16 gave 105, 125, 128 and 136 arrivals. I froze the multiplier at 16 before testing the two actors from Neow on 32,768 fresh matched starts.

| Held-out result | Original | Learned relic scores |
| --- | ---: | ---: |
| Reached Act 2 | 19,281 | 19,281 |
| Reached Act 3 | 748 | 911 |
| Reached Act 4 | 3 | 2 |
| Beat the Heart | 0 | 0 |
| Mean final floor | 21.739 | 22.222 |

The new choice gained 276 Act 3 arrivals and lost 113 on individual seeds. Both actors had one truncated run, included in the totals. This gives me a better way to reach Act 3 for practice. Whether it also leaves me with a deck that can finish the run is still open. The table assigns one average bonus per relic across all decks.

To reproduce the comparison and save the challenger locally, rebuild the native module using the instructions above, then run

```powershell
& $py -m experiments.heart.boss_relics
& $py -m experiments.heart.evaluate build/first-boss-relics/challenger.json.gz 55001000 32768
```

[Training estimates, validation scores and paired test counts](results/first-boss-relics.json)

### How many cards ahead should I look?

Against the Byrds on floor 19 of seed 56000000, I compared two choices from the same hand and 33 HP. The two-card search opened with Pommel Strike and was still fighting at 31 HP on the next turn. The five-card search opened with Bash+, then played Pommel Strike and two Strikes on the same Byrd. That finished the fight at 33 HP.

Depth counts the first card. For each possible first play, the planner projects changes such as energy spent, Vulnerable applied and enemy HP lost. It scores each continuation, discounts each successive card by 0.94, and uses the highest-scoring sequence to rank the first play. The agent plays one card and plans again from the real observation. The search considers cards visible and legal at the original decision. A draw can reveal a new option at the next decision. The approximation also omits some triggers and can score a non-targeted card after projected lethal.

I kept the learned first-boss relic scores, combat weights, training rewards and other choices fixed. On the same 8,192 validation starts, depths 1 through 5 reached Act 3 on 129, 247, 280, 286 and 297 runs. I selected depth 5 by Act 3 arrivals. On 32,768 separate starts it reached Act 3 on 1,109 runs against 943 at depth 2, gaining 414 seeds and losing 248. Act 4 arrivals went from 4 to 5. Neither policy beat the Heart. Depth 5 took 38.43 seconds against 10.44 at depth 2 using ten threads.

Reproduce the depth comparison and run the separate compute benchmark with

```powershell
& $py -m experiments.heart.turn_depth
& $py -m experiments.heart.turn_depth --benchmark
```

**I chose depth 2 for collecting training runs.** With a fixed time budget, I want as many trajectories reaching Act 3 as I can get. I timed five depths on the same 16,384 starts using ten threads. From depth 1 to 2, runtime went from 4.526 to 5.406 seconds (+19.4%), and Act 3 arrivals went from 307 to 475 (+54.7%). From depth 2 to 3, runtime rose another 33.4% for 15.4% more arrivals. Depth 2 gave me the most Act 3 arrivals per minute of the five settings I tested, about 5,272. This benchmark used frozen weights. Whether these trajectories produce a stronger policy after retraining is a separate question.

**For individual runs where search time matters less, I chose depth 5.** It reached Act 3 most often in the depth sweep, and its advantage over depth 2 held up on the separate 32,768-start test above. Heart wins are still too rare here to choose a depth by Heart win rate.

[Depth comparison and paired counts](results/turn-depth.json) · [Full-run throughput at each depth](results/turn-depth-throughput.json)

### Learning when to drink a potion

Blood Potion used to jump from a score of −65 to 80 when missing HP crossed 27%. I tried letting the outcome of drinking teach that decision. From the same combat position I tried each legal drink and target against taking the best card or ending the turn. All branches could use potions on later moves. I ran them to the end of the run so that saving a potion for a later fight could pay off. Each potion learned a base and coefficients for public state features, with data pooled across potions. This experiment covers ordinary combat turns. Out-of-combat drinking, discarding and shopping still use the original policy.

I also initialized one play weight per card from the existing Offering and Apotheosis bonuses of 35 and zero for the remaining entries. The other combat features and lookahead still contribute their own scores. I fitted these weights on pairs of moves labeled by the end of the fight. Offering moved to 32.55. The 8,192 training starts produced 24,148 drink examples and 23,377 card pairs. On 1,024 separate starts the potion model's prediction error (RMSE) was 6.067, close to 6.057 using only a potion's average advantage.

On 32,768 fresh matched starts at depth 2, the original policy reached Act 3 on 945 runs and encountered the Heart twice. The card-only version reached Act 3 on 892 and encountered the Heart six times. The potion-only version produced 961 and two, and the combined version produced 860 and one. All four had zero Heart wins. I kept the original as the reference. The combined result shows why I need to evaluate the components together as well as separately.

For my 20% target I count runs that enter a fight with a live Corrupt Heart, divided by all ordinary A20 Ironclad starts. The two original runs that died to the Heart counted. The two that died to Shield and Spear did not. Run `python -m experiments.heart.preferences` to reproduce the experiment. [Results and seeds](results/combat-preferences.json)

I checked the six-versus-two card result on a larger independent panel beginning at seed 71000000. Across 131,072 matched starts, learned card weights reached the Heart 9 times and the reference reached it 11 times. The learned weights did win seed 71126269 in 683 moves with 38 HP left. I'd keep the reference for encounter-rate comparisons. [Confirmation counts](results/heart-encounter-confirmation.json)

### Is giving up Burning Blood the problem?

I tried all four Neow offers on each of 16,384 training seeds. I measured the downstream return relative to swapping the starter relic and averaged it for each visible bonus and drawback. On 32,768 fresh starts, the learned opening reached the same two Heart encounters as the original. Act 3 arrivals went from 919 to 910. This gives me little reason to replace the opening in the current reference policy. Run `python -m experiments.heart.opening` to reproduce it. [Opening comparison](results/learned-opening.json)

### Should the next copy depend on the deck?

I compared taking each card with skipping from the same reward screen. I collected 73,168 alternatives from 8,192 ordinary training starts. One learner estimated the average value of each card. The other also saw existing copies, upgrade status, deck size and composition, and the old draft score. I let the learned ranking take over on card-or-skip screens and evaluated the two models separately.

On 32,768 fresh starts the per-card averages took Act 3 arrivals from 970 to 1,162. The deck-conditioned scores reached 1,038. The corresponding Heart encounters were 2 for the previous policy, 1 for per-card averages and 2 for the deck-conditioned policy. The previous policy also won on seed 70031312, verified across 747 moves with 11 HP left. The per-card version gives me more late-game training trajectories. Whether I can use those trajectories to improve Heart encounters is open. Run `python -m experiments.heart.draft_context` to repeat the comparison. [Card reward results](results/draft-context.json)

### Learning how to play the hand in later fights

I wanted the combat scorer to see which cards it was holding when it valued a play. I added a mean embedding of the visible hand and fitted a late-act scorer from 234,193 simulated alternatives at 44,292 positions. I used positions from both the reference policy and the card-reward policy that reaches Act 3 more often. Each alternative played out the rest of the fight with the same continuation policy. There were 37,447 positions in Act 2, 6,675 in Act 3 and none in Act 4.

On a separate panel of 65,536 ordinary starts the late scorer raised Act 3 arrivals from 1,880 to 2,115. It encountered the Heart three times against five for the reference. The same-position validation also gave a mixed signal. At the scale chosen on full runs, changing only the first move produced a mean end-of-fight utility difference of −0.003 across 2,416 positions. The observed gain is in clearing Act 2. I haven't seen a corresponding gain in reaching the Heart. [Hand-aware combat experiment](results/late-combat.json)

### What did the floor 49 practice buy me?

I saved naturally reached positions at floors 35 and 49 so I could train against the double boss more often. I used the learned reward-card averages, used the new combat scorer in Act 2 only, and searched late-game weights using floor progress, boss clears and actual Heart encounters. The training bank came from 65,536 ordinary starts before I reduced the run budget. It contained 512 floor 35 positions and 298 floor 49 positions. A separate 32,768-start validation pool supplied 512 and 132 positions respectively.

I then froze the models and tested each on the same 32,768 fresh starts. Using the staged policy took Act 3 arrivals from 945 to 1,294. The additional late-game weight search left them at 1,294, as expected for a change applied after entering Act 3. The reference encountered the Heart once, and both challengers had zero encounters. Each challenger reached Act 4 once. This made it easier to collect Act 3 positions but didn't demonstrate an improvement on my target.

For subsequent comparisons I use 32,768 fresh starts **per frozen policy**. Validation chooses settings from separate seeds. Training can revisit a single saved position many times. The current training pool also defaults to 32,768 starts. To reproduce the earlier training bank, run `python -m experiments.heart.heart_curriculum --training-starts 65536`. [Full-run counts and seed ranges](results/heart-curriculum.json)

### Can I replace the combat scorer entirely?

When I play Offering, the old actor gets a hand-written bonus before the network weighs in. I removed that scoring path for ordinary combat decisions. The new actor compares every legal card and target, drink and target, and end turn, using the visible hand, monster, card facts, and combat state. Each decision outside that scope, including card-selection prompts, still goes through the previous policy.

I trained an imitation version, a version using paired end-of-fight and downstream returns, and a second round with positions visited by a learned actor. The alternatives in a training decision shared the same simulator state. During play the actor scores only its public observation. The training and native scorers agreed to within 0.000001 on 54 candidate actions from a recorded run.

On 32,768 fresh matched starts the reference reached Act 3 on 987 runs and the Heart on four. The imitation actor reached Act 3 on 590 and the Heart on zero. Using outcomes raised those numbers to 721 and zero. The second round reached 728 and one. The small validation panel selected imitation by three-way ranking, but it came last of the three on test. I kept every checkpoint and retained the previous policy as the reference. [All three training and evaluation results](results/direct-actor.json)

### What if I change the whole strategy at once?

A deck chosen by one actor might do poorly when a different actor plays its cards. I searched 160 joint settings across combat, drafting, potions, routing, shops, rests, Neow and lookahead. Each candidate varied two groups of 21 weights and seven structural choices. I compared them on ordinary starts and separately on naturally reached floor 18, 35 and 49 snapshots. Those later positions exposed outcomes that a small batch from Neow would almost never see.

The frozen winner of that search reached the Heart on 2 of 32,768 fresh starts, against 1 for the reference. It reached Act 3 less often, 779 times against 991. Neither defeated the Heart. I saved the candidate separately because the extra Heart encounter is weak evidence and the earlier regression is substantial. [Joint search results and chosen weights](results/joint-strategy.json)

### Could my deck tell me when to switch strategies?

Two different strategies reached the Heart on different seeds. I wanted to know whether my deck, relics, potions, HP, gold and keys at the start of Act 2 could tell me which strategy to use. I kept Act 1 identical in every run. For each of 7,232 training positions I played four continuations from exactly the same point. I trained a ridge regression to predict how much each choice would gain or lose relative to the four-strategy average for that position. The training score rewarded floor progress, Act 3, late Act 3, Act 4 and Heart encounters. I chose the regularization strength on 2,450 separate Act 2 positions.

On 32,768 fresh starts the learned choice got 1,111 runs to Act 3, up from 967 for the reference and 1,066 for the best fixed choice on validation. It reached Act 4 twice and the Heart zero times. The reference reached the Heart twice. I also checked every alternative after the selector made its decision. Even with hindsight, picking among these four strategies once at Act 2 could have reached the Heart on only four starts. That tells me where to stop investing in this design. The downstream strategies need to produce much better continuations for a selector to have a chance.

I reran the full reference panel without the Act 2 split and got the same results on every reported metric. I can run the frozen selector locally with `python -m experiments.heart.stage_policy --replay-start 96000000 --runs 32768` after the native build and Python path setup above. [Matched test and hindsight counts](results/stage-policy.json)

### Does anticipating the next attack change which card I should play?

An enemy can spend this turn buffing and attack next turn. I learned the expected attack shown one, two and three turns later from 8,192 ordinary starts, grouped by enemy and current visible intent. I counted transitions only while the monster stayed targetable. For my estimate of the next hand I used the contents of the public piles. The planner compares a first card and possible second card against ending now, then estimates incoming damage, player health and enemy health over three more turns. The expected hand is an approximation. It reuses the same average for later turns and misses individual draws, many triggers and changes in enemy phases.

On unseen fights the next-turn damage prediction had a root mean squared error of 5.95, compared with 9.05 for predicting the average and 12.19 for repeating the current attack. I tried 24 sets of planner weights and selected from three finalists on separate full runs. On the 32,768-start test the selected planner brought 477 more runs to Act 2 and 22 more to Act 3 than the reference. Each encountered the Heart once, on different seeds. Using the same planner with just average enemy damage reached Act 3 on 947 starts and the Heart on two. Two rare encounters don't settle which policy is better at reaching the Heart.

I replayed 4,096 additional starts to see where the forecast policy failed. Of 889 Act 2 boss fights, it died in 764. The median boss entry was at 83% of maximum HP. Ordinary monster rooms accounted for 1,119 more deaths across 9,807 fights. Both getting to the boss and converting a reasonable starting position into a boss win need work. [Forecast accuracy, settings, matched full runs and encounter breakdown](results/forecast.json)

### What if I practice the Act 2 bosses directly?

Against the Collector, an attack spent on a Torch Head buys time but gives the boss another turn. I let the model change 18 combat weights per boss together, including how it values minion damage, early powers, block, potion timing and ending the turn. I trained on 768 natural boss entries. On 512 separate entries used for selection, boss clears went from 85 to 100.

On 32,768 fresh starts the forecast policy and its specialist reached the same 7,236 Act 2 boss rooms. The specialist got 246 extra runs to Act 3 and lost 232 the forecast policy got through. The net gain was 14. Both reached Act 4 five times. The forecast policy encountered the Heart three times and the specialist twice. On seed 106000015 their first different choice was where to play Strike. The forecast policy hit a Torch Head and eventually died on floor 33. The specialist hit the Collector and made floor 39. Later choices also differed.

I wanted to know how many of the losses were recoverable with the decks already in hand. On a separate panel of 768 boss entries, I played each fight 98 ways. The forecast policy cleared 106, the frozen specialist cleared 116, and at least one tested tactic cleared 309. That last count uses hindsight. It tells me that actions alone can recover many fights with the same HP, potions and cards. I can use those successful continuations to train a policy that chooses from what it sees during the fight. [Boss experiment, example and hindsight audit](results/boss-tactics.json)

### Did I stop training too early?

I replayed the original outcome-trained actor's training run and recovered its archived epoch-13 weights exactly. I evaluated epochs 1, 4, 8, 13, 16 and 32 on the same 4,096 full runs. The Act 3 counts were 52, 83, 91, 71, 88 and 85. Every checkpoint got zero Heart encounters. Epoch 13 had the highest single-move validation gain among those six. Extending training to epoch 32 gave me no evidence that more passes over these examples could solve the full game. [Epoch comparison](results/actor-epochs.json)

### Can a boss fight teach me what to put in the deck?

From 3,072 floor 32 positions I tried 73,728 single-card additions against skipping from the same state. I used the difference in boss clears to adjust the reward score for that card, accounting for upgrades and copies I already held. This measures the isolated addition one floor before the boss. Applying the estimate to earlier reward screens is an extrapolation. For combat I trained on 11,461 positions with 16 matched continuations apiece. A small network chose a tactic again from the visible fight state after each move. I compared five epoch checkpoints on the same 4,096 validation games and froze the combined policy at epoch 128.

On 32,768 fresh starts the combined policy got 1,250 runs to Act 3 against 1,076 for the previous specialist. It entered the Heart fight twice against four times. Since Act 3 arrivals improved, I tried each change separately on the same starts. Just changing the card rewards reached Act 3 on 1,318 starts and the Heart on four. Changing only the combat policy reached Act 3 on 1,011 and the Heart on two. I kept the draft-only checkpoint for the next experiment. The gain I can point to is getting through Act 2. [Combined run and component tests](results/recovery.json)

### What else could I change in the same deck?

I tried adding, upgrading and removing one card, and receiving a potion in an empty slot. Each comparison started from the same pre-boss position and ran to the end. I collected 225,225 paired edits across Acts 1, 2 and 3. Only 158 training runs reached the Act 3 campfire, so I shrank each item's act-specific estimate toward its average across acts. I applied the estimates together at rewards, shops, campfires and deck selection screens. I selected the weights on separate validation seeds.

On 32,768 fresh starts the combined policy reached the Heart six times against once for its parent. On another 32,768 starts with both models frozen it reached the Heart six times against twice. Act 3 arrivals increased on both panels, from 1,323 to 1,644 and from 1,341 to 1,641. On the first panel, changing only card addition, upgrading, removal or potion values produced two, five, one and one Heart encounters respectively. I kept the combined policy. Replaying the 158 training seeds at the Act 3 campfire had shown me the main bottleneck there. Of those, 152 died against one of the bosses, four died against Shield and Spear, and two died against the Heart. [Matched results, follow-up and late-game deaths](results/deck-training.json)

### Could I try a fight with this deck before choosing a tactic?

I selected eleven tactics from 84 candidates by looking for ones that rescued different training positions. The existing policy was the twelfth option. When the real policy sees a supported boss or Shield and Spear, it can build practice fights from the visible encounter, deck, relic identities, potion slots and maximum HP. The practice fights start from independent artificial seeds. I compare all tactics on the same draws and keep the one with the best mean fight return. The estimate assumes a range of starting HP and resets relic counters. It doesn't reconstruct the current hand or the exact combat state.

Validation chose eight practice draws per tactic, starting in Act 3. On 32,768 fresh starts the policy reached the Heart three times against two. On a second panel it reached the Heart five times against four. It beat the Heart once on each panel. Its parent had zero wins across the two. I replayed both victories in a fresh simulator and checked every legal action against the frozen policy. Seed 122022863 finished with 7 HP after 711 decisions. Seed 123006700 finished with 17 HP after 594. Eight encounters across both panels is still well short of my target. [Matched results and verified victories](results/scenarios.json)

### Can the map tell me when a risk is worth taking?

I trained a route and campfire policy by taking different legal branches from the same position. I compared progress, keys and remaining health six floors later. On two sets of 32,768 fresh starts, 41,732 runs reached Act 2 versus 39,636 before the change. Act 3 arrivals went from 3,212 to 3,521 and first Act 3 boss arrivals from 394 to 477. Both policies reached the Heart nine times. On the first set, changing only the map policy recovered 163 of the combined gain of 176 Act 3 arrivals. I kept the route and campfire change for training. I replayed the two new Heart wins, seeds 130016661 and 131002615, from their ordinary starting states. [Results and component tests](results/route-policy.json)

### Would I choose a different first card if I practiced the opening hand?

I rebuilt practice fights from the visible hand, enemy lineup, intent and draw-pile contents. I shuffled the unknown order independently and compared up to five legal plays on the same draws. At 57 HP against the Sentries I chose Sword Boomerang where the old policy chose Anger. Reversing the listed order of the 14 unobserved draw cards left every score unchanged in this example.

In 160 starts I could match 309 of 353 eligible openings and changed 43 first plays. On two fresh panels the search got 57 additional runs to Act 3, but both policies reached the Heart 10 times in 65,536 starts and neither won. Search took about 45 seconds per panel versus 18–19 for the parent. I kept the cheaper route policy as the parent. [Settings, matched runs and the Sentries example](results/opening-search.json)

### How much of a successful fight can I learn from hindsight?

I practiced 24 whole action sequences per fight. Holding the deck, HP and enemy fixed let me find out whether a different sequence could save the position. I used the fork's end-of-fight evaluation and its environment's legal actions. I sampled plays around my existing policy and trained a neural network to adjust its scores. The search was confined to generating labels. In the 1,077 held-out fights, it found 952 wins versus the original 879. The student's agreement with the selected play went from the parent's 38.4% to 52.0% on the sampled decisions. I sampled disagreements more often, so that percentage describes this exercise rather than the frequency of changed plays in ordinary runs.

I compared four training epochs at three strengths, then froze the epoch-2 student with strength 4. Across two sets of 32,768 normal starts, early survival improved again. Act 2 arrivals went from 41,669 to 42,472 and Act 3 from 3,538 to 3,726. The first Act 3 boss saw 466 entries versus 474, and each policy reached the Heart five times. I kept the student as an early-combat candidate. I want to test deck decisions against the boss visible on the map because getting through early fights with more decks has barely moved the late-game result. [Teacher coverage, epoch selection, matched tests and replay check](results/trajectory-teacher.json)

### What would I take before Slime Boss?

On floor 5 of seed 147027981 the old policy took Pommel Strike. The new one took Blood for Blood. I compared each card with skipping in four independent practice fights against the map's boss, then compared the real full-run continuations from the same reward screen. The two estimates depend on the offered card, current deck and boss icon. I reset relic counters in the practice fights and varied starting HP. I fit the final table on 14,103 decisions. On held-out offers the predicted advantage correlated 0.18 with the boss result and 0.06 with full-run progress.

Across two fresh panels, Act 3 arrivals went from 3,741 to 3,929. Heart encounters went from 13 to 10. On the first panel the boss-only and run-only versions reached the Heart three and seven times, versus seven for the parent. I replayed all 706 decisions on the Slime Boss example and saw the new deck reach the Heart. The example shows a real change of pick. The panels give me no reason to promote it as my Heart policy. [Matched tests, component results and example](results/boss-draft.json)

### Can I learn deck choices from people who beat A20?

The run histories record what was offered as well as what was taken. That gives me a choice to imitate even when the player skipped. I used ordinary A20 Ironclad runs from the rotating sample, chegs and panacea108. I kept wins and losses, deduplicated by seed and held out whole runs. I left out the deliberately constrained Neow series. The logs came from [MaT1g3R's public collection](https://github.com/MaT1g3R/Slay-the-Spire-data/tree/097aaf3564c2247835162d267cbc7c55d2c9039e).

I trained on 128 runs and tested imitation on 19 others. On 421 held-out card menus the model matched 64% of picks. Choosing uniformly among the same offers would average 24%. I added the fitted preferences at card rewards, boss relics and Neow. The combined policy got 2,302 runs to Act 3 versus 3,823 for its parent on 65,536 fresh starts. It reached the Heart five times versus eleven. It did win once on seed 151007641. I checked all 1,045 actions and the final 2 HP.

I isolated each change. The boss relic preferences alone reached the Heart seven times versus four for the parent on the first panel. On two further fresh panels they reached it 18 times versus 17. Combining them with the boss-practice card table reached the Heart 11 times on those same starts. I kept the original parent. The available logs tell me which offered items were picked. They don't tell me how to pilot the resulting decks in combat. [Imitation scores, ablations and independent confirmation](results/expert-draft.json)

### How much does playing Defect teach a network about Ironclad?

I used the toolkit's [published Defect PPO network and weights](https://github.com/bpblakely/sts-lightspeed-toolkit/tree/f46fed59175646254b284c27e677a9b32ca5b748/examples/defect_a20). The source config reports 144,179,200 training steps. I first froze it and took its highest-scoring legal action. On the same 512 Ironclad starts it reached Act 3 zero times playing everything, three times playing just combat, and zero times playing just the map and other out-of-combat screens. My existing policy reached Act 3 on 28. These results use greedy play. The upstream example samples actions.

I then trained the network on Ironclad. I let it visit some of its own states during imitation, and I credited either copy when two visually identical cards could perform the same play on the same target. The batched teacher matched the original scorer on 1,040 decisions, including 176 with interchangeable hand actions. After 131,072 imitation steps I switched to PPO for another 131,072 steps. The PPO objective includes a small imitation term. I checked two checkpoints during each stage and picked one per stage using the same 256 validation starts.

On a fresh 1,024-start screen, the frozen network reached Act 2 on 54 starts, the selected imitation checkpoint on 306, and the selected PPO checkpoint on 358. Act 3 arrivals were 0, 1 and 7 respectively. My existing Ironclad policy reached Act 2 on 683 and Act 3 on 68. No policy reached the Heart on this screen. The transferred policy learned, but replacing the existing actor cost too much early survival to promote it. I didn't run it on a full 32,768-start benchmark.

The training trajectories contained zero Heart encounters. I generated 32,768 fresh runs with the existing policy to locate stronger training starts. Of these, 8,476 reached floor 33, 255 reached floor 50 and 12 reached floor 55. I can now sample the approaches to those fights during training while continuing to measure Heart encounters from ordinary starts. [Frozen transfer, checkpoint comparisons and scout counts](results/ironclad-transfer.json)

### What if I give the learner another try from the same late fight?

I kept the old policy's action scores and trained a network to adjust them. Each practice group starts from one real position. I sampled six different continuations, let each vary its decisions for a short window, then played each branch to the end. Subtracting the group's mean return makes a move in a hard position comparable to the alternatives that were actually available there.

Training produced 1,297 Heart entries in 41,208 trajectories. They came from repeated positions, including just 12 distinct floor-55 starts. On two independent panels of 32,768 ordinary starts the new policy reached the Heart four and five times versus one and four for its parent. On a third panel it reached the Heart three times versus seven. That leaves both at 12 encounters in 98,304 starts. I verified a win on seed 165016444, which ended with 45 HP after 801 actions. I kept the parent. More late-fight feedback solved the sampling problem, but this update didn't establish that a fresh run was more likely to reach the Heart. [Training, three matched panels and a verified win](results/curriculum-actor.json)

### Would these relics help if I handed them to a different deck?

I paired 83 floor-50 inventories from the policy with 83 final inventories from human A20 Ironclad Heart wins. I held the combat policy fixed and gave each inventory the same 12 practice draws against each of four late opponents. Every fight started at 64 of 80 HP with an empty potion belt. Relic counters and bottle assignments were reset. The human inventories came from later in the run and were selected from victories, so this exercise measures the policy's ability to use those inventories. It doesn't measure how often it can build them.

Across 3,984 fights per condition, my policy won 835 with its own inventories, 1,412 with the human cards only, 582 with the human relics only, and 2,344 with the human cards and relics together. I then kept all the human decks and relic sets and shuffled which deck received which set. The original pairs cleared 11.8 percentage points more fights than the average of two shuffles. The paired bootstrap interval across inventories was 5.5 to 18.3 points. This gives me a reason to score the change a relic makes to the deck that will actually receive it. [Inventory swaps and opponent breakdown](results/inventory-benchmark.json)

### Can I learn what adding one relic does to the deck I already have?

On seed 169016266 I had 68 HP and a choice of Strawberry, Molten Egg and Captain's Wheel. My existing policy gave all three a score of 50 and took Strawberry. The learned inventory adjustment picked Captain's Wheel. The first run died to Transient on floor 46. The second reached the Heart. I checked both action sequences. Later decisions also changed, so this run shows how the model breaks a tie. I'd need more evidence to recommend the Wheel in this position.

I trained a network on 368,640 standalone fights. For each inventory I tried the same draws with a card added, a card edited, a relic changed, or a small bundle added. At a reward or shop, the policy scores how much the available change would improve the current inventory. On 61,440 held-out fights the selected checkpoint (epoch 24) predicted the direction and size of these changes only weakly: its correlation with observed changes was 0.166. The weakest of the eight adjustment settings I tried won selection.

On three fresh panels of 32,768 ordinary starts each, the Heart encounter counts were 2 vs 1, 12 vs 9, and 4 vs 2 in favor of the learned inventory scores. Across all 98,304 starts the new policy reached the Heart on 14 seeds the parent missed and missed eight seeds where the parent reached the Heart. Four were shared. The exact paired test gave p = 0.286. The parent won one run and the new policy won none. On the first panel I also enabled each adjustment alone. Card additions, edits and relic choices reached the Heart on one, one and two starts respectively. I kept the existing policy as the reference. [Training, paired comparisons, ablations and replay](results/inventory-learning.json)

### If a run died in this fight, could a different play have saved it?

I gave each distinct legal combat action its own copy of the same position and played to the end of the fight. That gave me four labels: survival, exit HP, remaining enemy HP fraction and change in potion count. I trained a network on differences between moves as well as their absolute outcomes. This lets me check whether the learner can tell which play saves HP even when both plays win.

From ordinary positions I trained on 40,565 decisions. On held-out choices my epoch-32 model reduced paired potion-count error from 0.158 to 0.050 and normalized HP error from 0.00547 to 0.00510. Survival error stayed at 0.0297. On 32,768 fresh starts the new policy reached the first Act 3 boss 262 times versus 216 for the parent. It entered the Heart six times versus eight. [Ordinary combat practice and score checks](results/combat-returns.json)

I then went back to the actual fight that ended each run. The teacher tried alternative plays and followed through on earlier changes. On 640 independent losses it found and replayed 387 ways to survive, including 192 of 378 Act 2 losses. During training it rescued 2,385 of 4,095 fatal fights. I merged those action outcomes with the ordinary examples. On another 32,768 ordinary starts, the rescue student reached the Heart four times, the ordinary-only student three times and their parent seven times. These local rescues reveal mistakes I could correct with hindsight. The results leave open how to recognize those mistakes from the information available when I have to play. [Rescue breakdown and three-way test](results/fatal-combat.json)

### How much could I improve if I knew which reward would take me further?

I kept the combat player fixed. At a map, shop, campfire, event or reward, I let a teacher try each legal menu option and finish the run with the old policy. It could take the best option, arrive at another decision, and try again. The teacher could use the outcomes of these hypothetical runs. My playable model only gets to see the current public state.

On 256 held-out starts the teacher reached the Heart ten times versus zero. It raised the mean final floor from 23.84 to 41.78 with 918 changes across 25,323 decisions. I generated labels on a separate 1,024 starts. The teacher reached the Heart 34 times versus one for the incumbent, and I replayed the promised continuations with zero mismatches. This is the performance of one hindsight search procedure. A deeper search could find choices it missed. [Training and held-out hindsight results](results/macro-teacher.json)

The macro student saw 372,777 paired alternatives in training. On the held-out paths only 1,292 of 94,653 alternatives improved the incumbent continuation. Giving those rare examples six times the training weight increased recall from 8% to 23%. Only 8% of the alternatives it predicted would help actually did. I also found that adding a common negative correction to a shortlist could accidentally promote a move outside the shortlist, so I centered the correction on the incumbent. The selected model reached the Heart three times versus seven on 32,768 fresh starts. [Distillation, calibration and matched run results](results/macro-student.json)

### Does Twin Strike need to see the other legal plays?

I gave each legal card and target combination, potion play and end-turn option a row. One model could attend to the other rows before predicting its four combat outcomes. The control model scored the rows independently. I gave them the same training positions, group weighting, hidden width, epochs and validation procedure. This experiment represents the options available now. An unplayable card in the hand doesn't get its own row.

On 58,520 held-out alternatives, self-attention reduced the joint utility error from 2.752 to 2.746. I checked that reversing the row order preserved the utilities and that the exported native scorer reproduced the Python score differences on 198 legal candidates. On 32,768 new games the attention model reached the Heart five times, the independent model four times and the incumbent seven times. I kept the incumbent. [Architecture comparison, outcome errors and run results](results/choice-attention.json)

### What information did the other Ironclad agent see?

I found a project reporting 50.1% Heart wins in 501 original-game A20 runs. The network made the out-of-combat choices. Their combat search could see the true draw order and RNG. I would count a demonstration from that search as a training label, and I'd test my own player with the draw order hidden. The published win rate belongs to their full system. [Author's benchmark](https://github.com/destire-mio/sts-ironclad-agent) and [combat search disclosure](https://github.com/destire-mio/sts-ironclad-agent/blob/main/docs/combat-solver-r11-20261119.md)

A second project reports 66 Heart encounters and 13 wins in 1,000 floor-zero A20 runs. Its combat solver samples possible hidden states, using more search on bosses than on ordinary fights. Their benchmark used a development partition. I can use that result as a reference for what public-information search can do. I haven't reproduced their run or used their weights for the measurements here. [Measured architecture](https://github.com/SnyderConsulting/sts_ml/blob/main/docs/ARCHITECTURE.md) and [benchmark](https://github.com/SnyderConsulting/sts_ml)

### Could I search a fight using only what I can see?

On floor 16 of seed 195000000 I had 73 HP. The incumbent wanted to drink Distilled Chaos. My new search chose End Turn. It built possible fights from my visible hand, piles, powers, potions, enemies and current intents. For the enemy's private move history I sampled from independent practice games. I checked the reconstructed combat rows, relic displays and legal moves before trusting a particle. Remaining unknowns include the order in which powers were applied, temporary card effects and some relic counters. A failed match leaves the incumbent in control.

I checked whether the listing order of the draw pile could influence the answer. Reversing the list made exactly zero difference to the scores on 1,629 live decisions. During 40 training-independent audit runs, 4,478 of 5,825 eligible positions matched with a bank trained on 3,072 starts. I then compared five search budgets on 1,024 common starts. Selected search (three particles on elites, bosses and Act 2 or later fights) reached Act 3 on 92 starts and the first Act 3 boss on 14. The incumbent reached them on 67 and six. I selected this setting before the full 32,768-start comparison. [Search settings, coverage and draw-order audit](results/public-belief-screen.json)

I also tried playing practice fights before taking a card reward. If I add Pommel Strike and reshuffle the entire deck, I change the draw order of all the cards I already had. I instead gave the existing card slots shared random priorities and inserted the new card at its own random priority. Across 256 ten-card examples every existing card kept its relative position, and the new card appeared in all eleven possible positions. I can compare the outcome under the same initial draw conditions, including cases where the added card delays another card I wanted to draw.

### What carried over to ordinary runs?

I froze the three-particle search before running 32,768 fresh starts. The old policy reached the Heart three times and search reached it eleven times. They shared one seed. Search gained ten and lost two, giving a two-sided paired exact p-value of 0.039 on this panel. Neither defeated the Heart. Act 3 arrivals went from 1,775 to 3,008, and first Act 3 boss arrivals from 226 to 434. Evaluation took about 26 minutes versus 23 seconds with other jobs running on the same machine. I kept a second seed range for checking how well this repeats. [Matched results and replay](results/public-belief.json)

I replayed seed 192006772. At 60 HP against Hexaghost the first disagreement came on decision 89. The old policy drank Distilled Chaos and died on floor 16. Search drank a Skill Potion, picked up all three keys and entered the Heart fight with 45 HP. It died four turns later. I checked every one of its 575 decisions against a second simulator replay. Several later decisions changed too.

I also trained a self-attention network on 36,821 public search decisions so I could measure how much of that improvement I could get without running search during play. Even its best epoch had higher average search regret than the original policy on 5,821 held-out decisions (1.227 versus 1.181, lower is better). On a second set of 32,768 starts the student reached the Heart five times, the card reward search also reached it five times, and the parent reached it twice. Both improvements are small at the Heart. I am comparing the two components with combat search enabled on those same starts. [Training, held-out errors and first component results](results/search-student-and-deck.json)

### Which part of deeper search is worth paying for?

I held the starting positions fixed. For every ordinary run I tried to save its fatal fight, and I also kept a randomly selected fight the old policy won. The deeper search can inspect eight public-compatible particles and nine plays. It tries three continuation policies for each play. I average across possible hidden states before picking a continuation, because I won't know which hidden state I'm actually in. I also collected independently simulated late-game enemy phases and increased the value of surviving.

On held-out positions the first search rescued 56 of 255 fatal fights, and deeper search rescued 79. Switching to deeper search gained 35 fights and lost 12. On the 255 winning controls it preserved 254, versus all 255 for the first search. The deeper search took 373 seconds versus 21 seconds on these fights. Most of the additional saves came in Act 1. Only twelve of the fatal positions were in Act 3 and none were in Act 4. I started a floor-zero evaluation and separate checks from naturally reached floors 32 and 49 to find out whether the expensive search improves the late game. [Matched fights by act and search settings](results/matched-fight-search.json)

The compact student saw the available plays, a target and 58 features per play. The search teacher could also see the draw pile, discard pile, other enemies, relics and powers. I am testing that difference by replaying the same decisions and keeping their search targets fixed. Each replay checks the suggested action before saving the full public observation. I reserved different seeds for checkpoint selection and for assessing the selected student. On a 333-decision implementation check the new interface reproduced every parent move with the learned correction set to zero.

### What if the second play changes the value of the first?

I noticed the planner dropped every zero-energy position before looking at the legal moves. In 128 ordinary starts there were 409 otherwise in-scope decisions with a playable card at zero energy and 751 more with only potion use available besides ending the turn. I enabled these choices and added a second play to the search. For example, if the first card draws a card, the second move can depend on the new hand. I group independent practice worlds by the combat information visible after the first move. For each world I choose the second action using the other worlds in its group, then score the action in the one I held out. A hand I saw in only one world keeps the original continuation.

I tested the calculation with two simulated outcomes. A move worth five extra HP in both worlds earned ten in total. A move worth twenty in one world and minus twenty in the other earned minus twenty under held-out selection. On four ordinary audit starts the second step changed 29 first moves in 497 searches with multiple particles. That tells me the implementation changes play. The matched-fight and floor-zero comparisons test whether the changes help.

### Did PPO close the gap on ordinary starts?

I let the Ironclad-adjusted network change the existing policy's scores and trained it for 256 updates of 4,096 steps. None of the 4,449 completed training episodes reached the Heart. Exact outcome feedback was therefore sparse even though the learner received intermediate rewards for progress. I checked five training checkpoints on 512 separate starts. Updates 16, 48, 96, 160 and 256 reached Act 3 on 30, 30, 19, 17 and 22 starts. The parent reached it on 37. I froze update 16 using the predeclared validation score.

On 32,768 new starts PPO reached the Heart five times and never defeated it. The parent reached it six times and defeated it once. Act 2 arrivals increased from 21,089 to 21,239, then Act 3 arrivals fell from 1,853 to 1,501. I kept the parent for further comparisons. [Training, checkpoint selection and frozen evaluation](results/anchored-ppo.json)

### What did combining the combat and reward searches change?

On 32,768 ordinary starts the old policy reached the Heart twice, card-reward search five times, and the combined searches thirteen times. The combined policy retained both original entries and gained eleven. Act 3 arrivals rose from 1,924 to 3,160. None of those three policies beat the Heart on this panel. I am also running combat search alone on these seeds to measure what the card-reward component adds to combat search.

I checked all 632 moves on seed 194001088 in two simulator paths. I saw the combined policy take the Sapphire key on floor 9, Ruby on 32 and Emerald on 44, then enter the Heart fight with 25 HP and die on its first turn. This counts toward the entry objective. Even if every one of its 3,160 Act 3 arrivals had reached the Heart, that would still be below my target. [Seed-level comparison and replay](results/paired-combat-and-rewards.json)

The two-play version rescued 47 of 128 formerly fatal fights and preserved all 128 winning controls. Six-draw search with the same survival objective rescued 40 and preserved 126. On 1,024 fresh starts the two-play version got 108 runs to Act 3 versus 110 for six-draw search. It got 21 to the first Act 3 boss versus 23. I kept six-draw search in the comparison and stopped the two-play test after that screen. [Matched fights, component checks and ordinary starts](results/two-play-search.json)

### Could human choices teach me how to build a deck my policy can play?

I found an archive with 2,648 historical A20 Ironclad victories. I kept 2,630 distinct runs that ended on floor 57 with the Heart as the last recorded fight. All are victories, so I used the offers as examples of human choices. The archive gives me neither labels for the outcomes of the rejected options nor turn-by-turn demonstrations. Most logs came from the July 2020 build. [Archived run data](https://github.com/karlo-delacruz-ieds/sts_1_winning_percentage/tree/411c17ab53b24c3781f5a02c6a4032667bbeda3f)

I trained a model on 57,903 offers from 2,088 runs. For each offer I reconstructed the cards and relics recorded before the choice. Shops, events and same-floor relic gains are deferred until after the labeled menu. Unknown transforms can leave gaps in that history. I compared an item preference by act with a second model that can adjust its preference using the current inventory and copy count. On 268 held-out historical runs, adding inventory context raised card-choice agreement from 59.8% to 65.8%. On 147 separately collected streamer runs it rose from 58.5% to 60.6%. The exported C++ and Python scores agreed within 0.000009 across 189 card menus and 15 boss relic menus.

I applied the learned adjustment to both card and boss relic rewards, selected the scales on 1,024 separate starts, then played 32,768 matched starts. The parent reached the Heart five times, item frequency eight, card adjustments alone nine, relic adjustments alone two, and the contextual model with both adjustments thirteen. I replayed both of its victories, on seeds 214004505 and 214004761, with 35 and 22 HP remaining. On another 32,768 fresh starts the contextual model reached the Heart thirteen times against nine, with zero victories for either. Across both panels it gained 23 entries and lost 11. Three seeds were shared. The paired two-sided p-value was 0.058. I am testing whether the learned reward policy helps when the combat player can also search. [Training, holdouts, both full panels, component tests and verified victories](results/historical-human-strategy.json)

### Can I obtain more Heart examples without changing the game I am training for?

On the 194M panel, nine of the combined policy's thirteen Heart entries came from 244 seeds where the older policy had reached floor 50. I can run every training episode from floor one and spend more of the budget on those seeds. To estimate how a policy would do on an ordinary start, I multiply the mean in each floor group by the fraction of all seeds in that group. The acting policy never receives the old policy's result or the seed.

I checked this sampling design against two separate completed panels before training. For a batch of 1,536 games, uniform sampling would give me 0.61 Heart entries on average in either panel. With the new design I expect 5.08 and 7.83 respectively. The estimated variance of the ordinary-start rate drops to 55% and 29% of the uniform estimator's variance. I will judge the trained policy on a separate contiguous 32,768 starts without weighting. My first experiment jointly changes how much the policy trusts human card and relic preferences by act, health and deck size, and its willingness to take another card.

### Would caching let me afford deeper combat search?

I tried caching decisions inside a simulated combat search. A hit requires the same public observation, legal actions and continuation policy. With caching on and off, all 959 checked decisions agreed. Across two timed repeats of 24 games each, throughput rose by about 8%. I kept this optional because the measured gain is small relative to the cost of deeper search. [Timing and trace checks](results/rollout-cache.json)

### Which single search change rescued the most fights?

I applied each change separately to the same 255 previously lost fights and 255 previously won fights. The original search rescued 56 losses and preserved all 255 wins. Sampling eight possible worlds raised rescues to 65, at a cost of two previously won fights. Adding late-game examples to the enemy bank rescued 63 and preserved all 255 wins. Broadening search to more hallway fights rescued 60 and preserved all 255. The complete set of changes rescued 79 and preserved 254. I would spend a limited budget on better examples of enemy states first, since that change recovered seven fights without spoiling a control here. These counts measure combat recovery on a fixed set of positions. The ordinary-start comparison measures the consequences for the whole run. [Seven individual changes on matched fights](results/search-ablation.json)

### What would arriving at full HP buy me?

I sampled 160 naturally reached positions before each of the Act 2 and first Act 3 bosses. I let the original policy take its ordinary campfire choice and then changed HP immediately before the boss fight. Each comparison uses the same enemy, deck, relics, potions and underlying game randomness. The unmodified control reproduced the original boss outcomes for both panels.

With combat search, Act 2 wins went from 35 at natural HP to 44 with 12 extra HP (capped at maximum), then to 50 with full HP. First Act 3 boss wins went from 50 to 53 to 73. Against Bronze Automaton the policy won 12 of 54 Act 2 fights at natural HP and also 12 at full HP. Against the Champ it went from 12 of 51 to 21. Health matters, but its value depends on which fight and deck it buys time for. The intervention supplies free health without forgoing an upgrade, so the counts do not tell me how often the policy should rest. [Matched health interventions by boss](results/boss-entry-health.json)

### Did training on the actual Heart signal change ordinary play?

I tried 12 changes to the policy together over five generations. I selected a setting on 4,096 independently sampled starts, froze it, then tested on 32,768 consecutive seeds beginning at 222000000. The original policy reached the Heart six times, the policy trained on human choices eight times, and the policy adjusted by encounter feedback eleven times. The adjusted policy gained eight encounters and lost five relative to human imitation, with three in common. None of the three policies killed the Heart on this panel. I kept the candidate for a fresh matched replication because the net change of three games is small. [Training, selection and ordinary-run results](results/stratified-gate.json)

### Would combat search alone explain the combined result?

I finished the last arm on the same 32,768 starts. Combat search alone reached the Heart 13 times, exactly as often as combat search plus reward planning. Five encounters were shared. Eight seeds improved and eight regressed after I added reward planning. Act 3 arrivals rose from 3,055 to 3,160 and Act 4 from 22 to 24. I have evidence for combat search over its parent, but this panel gives me no net Heart benefit for the extra reward search. [Four policies on the same seeds](results/paired-combat-and-rewards.json)

### Can I learn from a stronger public-information agent across simulator versions?

I obtained 1,000 published action traces from SnyderConsulting's Ironclad agent. Their benchmark reports 66 Heart encounters and 13 victories. Six runs ended in errors and I excluded them from imitation. I first translated each recorded action into the interface I use for play. I matched the floor, encounter, entry HP and exit HP for 3,289 fights. All were in Act 1. Continuing after the first divergence would pair potentially incorrect game states and decisions. [Original benchmark and traces](https://github.com/SnyderConsulting/sts_ml/tree/1e69c2e78a5ab3cb44cca5ffdfcb82803a68701d/results)

I then replayed the 994 completed games in the exact published simulator and matched their initial and terminal state hashes. I exported 142,159 regular combat choices with at least two legal actions, including 22,607 in Act 3 and 3,212 in Act 4. I gave my learner only fields I can also construct in my simulator. It sees unordered card piles, HP, block, energy, cards, relic IDs, visible statuses and attack hit counts. It has no access to replay seeds, future actions, private move IDs or draw order. I separated complete games before training. In 515 early-combat decisions that could be reconstructed in both simulators, all 515 action masks agreed. The full input tensors matched in 487. The differences included card costs, upgrades and enemy status stacks. I will evaluate transfer by playing ordinary games in my simulator. [Cross-fork checks and data split](results/public-demonstrations.json)

To reproduce the extraction, I build the local native extension, run `python -m experiments.heart.public_demonstrations`, then `python -m experiments.heart.prepare_public_demo --python313 PATH_TO_WINDOWS_PYTHON313` and `python -m experiments.heart.audit_public_demo`. I train with `python -m experiments.heart.public_demo_student`. The upstream simulator, trajectories and model checkpoints are stored in the local build directory.

### Would the Heart-trained deck policy replicate?

On 32,768 new seeds beginning at 223000000, the frozen policy reached the Heart 15 times against seven for the same policy without the learned adjustments. Both killed the Heart once. Over the two panels I gained 19 encounters, lost eight and kept seven in common (two-sided exact paired p = 0.0522). The observed encounter rate went from 15/65,536 to 26/65,536. I would keep the adjustment for further development while treating the size of its benefit as uncertain.

Act-dependent trust alone reached the Heart 10 and 11 times on the two panels. HP and deck-size conditioning alone reached it nine and 11 times. The change in willingness to take a card alone reached it five and 10 times. The complete policy reached it 11 and 15 times. I will use the complete set of weights as the macro policy in the next combat experiment. [Replication and all three individual changes](results/stratified-gate.json)

### Which imitation checkpoint should play?

I initialized the combat network with my existing Ironclad weights and trained it on 114,558 recorded choices. When two copies of a card have the same visible hand row, I credit either copy on the same target. I compared all eight epochs on 15,119 decisions from separate games. Validation negative log likelihood fell from 1.768 initially to 1.099 at epoch four and rose to 1.134 at epoch eight. I froze epoch four before opening the final 12,482 decisions. It agreed with the demonstrator on 54.9% overall, 63.0% in Act 1, 49.0% in Act 2, 42.1% in Act 3 and 35.3% in Act 4. The Act 4 figure has only 215 decisions.

I also want to know whether this network is more useful after the deck is established. I fixed two candidate switching floors, 35 and 49, and three strengths of the original combat policy. I can test a frozen late specialist on an entire 32,768-start panel while replaying only those seeds that the parent reached the switching floor on. Both policies make identical decisions before that point. Every other seed is an unchanged loss and still counts in my denominator. I check a sample of the unchanged prefix policy against the original native outcomes before selecting the floor on separate seeds. [Epoch selection, assessment and data checks](results/public-demonstrations.json)

### Did the tactical imitator improve actual games?

I played the same 1,024 fresh starts with the macro policy fixed. My existing combat policy reached Act 3 in 53 games. The imitation policy reached it in 28. Mixing imitation with the existing scores raised that to 47 or 52, depending on the strength of the mix. No arm entered Act 4. I rejected the whole-run replacement. The original inputs could distinguish the number of incoming hits, but could not distinguish six incoming damage from thirty. My next version restores the displayed intent and damage and increases the relative training weight of later acts. [Frozen policy screen](results/public-demonstrations.json)

I also completed the late-position test for deeper search. Of 96 naturally reached floor-49 situations, the original policy reached the Heart once, ordinary combat search four times and the deeper search four times. The deeper search got 33 runs to the second Act 3 boss against 23 for ordinary search. Of 96 floor-32 situations, none of the three policies reached the Heart. The deeper search got one to Act 4. [Late-game continuations and matched fight controls](results/matched-fight-search.json)

### Can I reproduce the displayed incoming damage?

I translated the current move into the game's visible icon, damage and hit count. The renderer accounts for Weak, Vulnerable, relics, stance, facing and Intangible. Runic Dome yields a hidden icon and zero damage fields. The learner never receives the original move identifier. On 515 shared early-game decisions the two simulators agreed on all enemy intent fields in 514. All 515 legal masks agreed and reversing the draw pile changed zero inputs. The single damage discrepancy occurred on a position where the simulators also disagreed on enemy Strength.

I trained with the restored display and gave underrepresented acts more weight by the inverse square root of how frequent they are. I preserve the original data split and the previous legal-action objective. For this version the extraction command adds `--intent-output build/public-expert-intent`, and training adds `--root build/public-expert-intent --balance-acts`. [Visible-field comparison](results/public-demonstrations.json)

### Did the full-observation search student retain the teacher's Heart entries?

On the complete 194M panel of 32,768 ordinary starts, the original combat policy reached the Heart twice, the online search reached it 13 times, and my full-observation student of that search reached it three times. The student won zero Hearts. It reached Act 3 in 1,946 runs against 1,924 for the parent and 3,055 for search. I would retain online planning when I need the stronger play. Moving its decisions into this network recovered little of its end-to-end benefit. [Frozen checkpoint, validation and matched ordinary runs](results/full-search-student.json)

### Why did the first trace translation stop before Act 2?

I compared the actions on both sides of the first boss chest. My simulator keeps the chest open after taking a relic and exposes a final confirmation. The recorded simulator returns directly to the map. I inserted that confirmation only after a relic had been taken and every relic choice was disabled. This recovered 111 further matching fights, all in Act 2, bringing the total to 3,400. The imitation dataset already contains all four acts because I extracted it inside the source simulator.

I repeated the observation comparison deeper in combat, including 214 Act 2 choices. Of 473 positions, 437 matched on every input, 470 on every intent field, and 471 on the legal mask. Reversing the draw pile changed zero inputs. The remaining discrepancies matter when interpreting imitation accuracy across the two games. [Trace and observation audits](results/public-demonstrations.json)

The new combat policy selected epoch five. Held-out action agreement went from 54.9% to 55.4% overall and from 35.3% to 42.3% on the 215 Act 4 choices. On 1,024 new starts with the macro policy fixed, the revised standalone version got 35 to Act 3 compared with 28 for the previous standalone version and 53 for the parent. Mixing the revised model with the parent got 55 to Act 3 and one to the Heart. I selected that mix for the full panel. [Frozen checkpoint and screen](results/public-demonstrations.json)

### Did adding the compact student help online search?

On the same 32,768 seeds the compact student alone reached the Heart five times, search alone 13 times, and search with the compact adjustment 14 times. The combined player gained 11 encounters and lost 10 compared with search alone. Act 3 arrivals rose from 3,055 to 3,173. One additional encounter across that many runs gives me little reason to favor the extra component on Heart reach alone. [Frozen policies and paired outcomes](results/search-student-and-deck.json)

### Did a late combat specialist help on ordinary starts?

I tried switching combat policy on floor 35 or 49. I selected the floor-35 conservative blend on separate practice games. On the untouched 32,768-start panel, the 1,774 runs that reached Act 3 got the specialist and all earlier choices stayed identical. It retained the parent's five Heart encounters and added two. Neither policy won against the Heart. With only two seeds separating them, I kept this as an experimental option. [Selection and paired outcomes](results/public-demonstrations.json)

### Can I learn when to rest and what to buy from a stronger player?

I extracted 15,346 campfire, shop, upgrade and removal decisions from 994 verified public games. I kept whole games together when splitting training, checkpoint selection and testing. The scorer compares each legal option using the deck, relics, HP, gold, key status, potion occupancy and price. This lets the same upgrade or purchase receive a different score when the run changes. In 560 real menus the exported C++ scorer agreed with the training implementation within 0.000004 on centered logits. That check caught a floor-zero discrepancy in card removals before I ran the final evaluation.

On 4,096 ordinary selection starts the existing policy got 225 runs to Act 3 and two to the Heart. The selected 70% imitation blend got 170 to Act 3 and two to the Heart. I froze that mix and scheduled separate tests of the four decisions to see which ones help my own combat player. [Training, checkpoint selection and gameplay screen](results/resource-policy.json) and [demonstration source](https://github.com/SnyderConsulting/sts_ml/tree/1e69c2e78a5ab3cb44cca5ffdfcb82803a68701d/results)

### What happened when the learned resource decisions played full runs?

On 32,768 new starts the existing player reached the Heart 16 times and won twice. Changing all four decisions together gave me eight encounters and zero wins. Changing only the shop gave me five encounters, only removals 11, only campfires 17, and only upgrade targets 18. I kept the existing shop policy. On a second 32,768 starts, changing only upgrades again increased Act 3 arrivals (1,816 to 1,931), but Heart encounters fell from nine to seven. Across the two panels the upgrade adjustment added 278 Act 3 arrivals and zero net Heart encounters. [Frozen component comparisons and replication](results/resource-policy.json)

### Can the combat planner continue while a Bronze Orb holds a card?

My planner used to stop simulating actions whenever an orb showed Stasis. I subtracted the hand and all three public piles from the permanent deck. When that left the correct number of cards, I restored those cards in independently simulated fights. For two different stolen cards I varied which orb carried which. A separate grader could compare my inference with the actual stolen cards. Across 3,993 decisions with an active holder, I made 463 predictions and got the set of cards right 458 times. The five errors are why I treat this as an inference.

I played both policies from 487 naturally earned Bronze Automaton fights on fresh seeds. Recovering Stasis took wins from 100 to 113, with 21 newly won fights and eight newly lost ones. The other 1,049 boss fights produced identical results. This measures a specific Act 2 fight, so I report it separately from the ordinary Heart rate. [Stasis reconstruction and matched outcomes](results/stasis-search.json)

### How much does the human draft policy help when I also search combat?

On 32,768 identical starts, the learned draft alone reached the Heart nine times. Combat and reward search without the learned draft reached it 13 times. Using both the learned draft and those searches reached it 34 times and defeated it three times. The combined player gained 31 encounters and lost ten relative to the two searches alone. Its 3,077 Act 3 arrivals were fewer than the searches' 3,160, which makes the gain after Act 3 particularly interesting. I replayed all three victories from Neow and verified the final encounter and remaining HP (30, 30 and 23). I started an independent replication and a comparison that removes reward search while retaining learned drafting and combat search. [Panel, paired changes and verified victories](results/combined-human-search.json)

### Would a model of successful inventories make better draft choices?

I trained a model on visible inventories at shops and campfires using 32,768 of my own runs plus independently verified expert games. It predicts whether the run advances to the next act under either player and predicts which player assembled the inventory. To score a reward or removal I change the relevant item count and measure the difference in its predictions. The C++ scorer and the training model agreed to within 0.000008 on 994 real menus.

On held-out games the survival log loss was 0.591 against 0.602 for knowing only the act and player. The source classifier's log loss was 0.018, easily explained in part by the opening relic. In the Act 1 inventories I inspected the expert always had Burning Blood; my player had exchanged it in every case. Recognizing the player doesn't tell me which item would help my player win.

I selected the scales on separate seed pools using weighted Heart encounters. The selected policy used the two survival estimates and zero weight on source resemblance. On 32,768 untouched starts it reached the Heart twice against 13 for the parent. I rejected it. Being able to forecast what happens to an existing deck offered little protection against choosing the wrong edit. [Forecast accuracy, selection procedure and ordinary runs](results/deck-outcomes.json)

### Should I keep Burning Blood with the current draft policy?

I repeated the opening experiment with my current draft policy. On 32,768 new starts the ordinary opening got 1,854 games to Act 3 and 14 to the Heart. Ranking all four offers using the previously learned values got 1,847 and 12. Forcing the best offer that retained Burning Blood got 1,466 and nine. I kept the ordinary opening. [Controlled opening comparison](results/starter-relic-transfer.json)

### Can my combat policy learn on the late-game decks that the expert actually earned?

I replayed the verified source runs and saved the exact beginning of each sampled fight. I separated whole games into 420 training fights, 160 validation fights and 179 held-out fights. Replaying from those states fixes the deck, HP, enemy and initial shuffle. The learner receives the public observation. The grader can compare the outcome with the recorded expert play.

I first queried a public-information search teacher on 4,121 decisions reached by my student and added its preferred move to training. A control received the same number of updates and original demonstration examples matched by act. On the 179 held-out fights the initial student won 127, the extra demonstration training won 126 and the correction-trained student won 127. The original recorded player had won 158. Follow-up selection screens used the same deterministic fallback in all student arms.

I also tried reinforcement learning. Four sampled attempts started from each identical state. I compared their returns within the group so that a strong starting deck could not earn credit for the actions I sampled. A win earned one point and up to a quarter point for HP left. I limited the update with a clipped policy ratio and a penalty for moving far from the initial action distribution. Validation selected update six of twelve. On held-out fights it won 128 against 127 and left 34.4 HP on average against 33.5. Its clearest difference was Act 3 (42 of 56 against 37); it lost ground in the other acts. I scheduled a full ordinary-start test that deploys it only in Act 3. [Matched fights, training and epoch selection](results/expert-state-learning.json)

The older whole-run imitation blend has now finished its ordinary panel. It got 1,925 runs to Act 3 against 1,825 for the existing player, but reached the Heart only ten times against 15. Each won once. [Frozen ordinary-start evaluation](results/public-demonstrations.json)

### Does understanding the map beyond the next room change the route?

I fitted two routing policies to the same 8,370 expert decisions at map forks. The first scores each immediate room given the resources I carry. The second also computes the value of connected paths backward from the top of the map. I used a differentiable Bellman backup so an observed choice can teach me that an elite leading to a campfire differs from one leading to another fight. Only the room types, edges, visible burning elite and current resources enter the model.

On 984 held-out forks the next-room model matched 73.8% of expert choices; the whole-map model matched 92.7%. I selected the deployment mixtures on separate ordinary starts. On the same 32,768 test games the existing route policy reached the Heart 11 times, the next-room model 12 times and the whole-map model 26 times. The whole-map model reached Act 3 on fewer starts (1,604 versus 1,880) but entered Act 4 twice as often (38 versus 19). It won one Heart. I kept it for independent replication and a combined test with combat search. [Imitation, score parity, tests and paired encounters](results/learned-map.json)


On a second 32,768 ordinary starts my whole-map router reached the Heart 21 times and won twice. The old router reached it 11 times and won zero. Across both panels I counted 47 encounters versus 22. I then switched the new router on in only one act at a time on the second panel. Act 1 alone got 18 encounters, Act 2 alone got 11, Act 3 alone got 10, and Acts 2 and 3 together got 11. I can see an effect from my early route even though the Heart comes three acts later. I kept the all-act router because it won the full comparison on both panels. [Replication and stage-specific results](results/learned-map.json)


### Can I improve an expert route by replaying the road I didn't take?

I saved 6,397 actual forks, replayed up to three legal paths from each one, and finished each run with my existing router. The unchanged path had to reproduce the full original run's floor, act, keys and encounter result before I used either branch. Only 35 forks produced different Heart outcomes, including 28 from Act 3. With whole games assigned to splits, that left 31 informative training forks for a Heart-only objective. I kept a second objective that also valued reaching the bosses and Act 4.

On 8,192 selection starts I counted 11 encounters with Heart-only training, four with the extra milestones and nine with the existing router. On 32,768 fresh starts the selected policy got 22 encounters and one kill; the existing router got 21 and two. Thirteen seeds gained an encounter and 12 lost one. I kept the existing router. [Matched branches, checkpoint selection and ordinary-start results](results/map-preference-learning.json)

### Would seeing the search teacher's scores help me learn its moves?

I used the same 29,240 training positions for three students. I varied the supervision between the teacher's chosen move, the fraction of search visits to each move, and a blend of within-position search evaluations and visits. Every student also saw the recorded expert move. Equivalent copies of a card aimed at the same target shared probability in the loss.

Validation favored the evaluation-informed student, which won 119 of 160 fights against 114 for my initial student. In 179 untouched fights the initial student won 127, the chosen-move student won 130, the visit student won 125, and the evaluation-informed student won 129. That leaves a substantial gap to the original expert's 158 wins from these starting positions. [Comparable search targets and held-out fights](results/root-distillation.json)

I also finished the ordinary-run test of my earlier reinforcement-trained Act 3 specialist. It reached the Heart 11 times on 32,768 starts. The imitation specialist also got 11, and the existing player got 12. The five extra Act 3 fight wins I had seen on expert decks didn't translate into more Heart encounters on my own decks. [Ordinary Act 3 transfer](results/expert-state-learning.json)


### Would a deeper tree rescue more losing fights on my own decks?

I built a tree that plays out the whole fight on each independently generated battle. I compared two simulation budgets and two ways to score a root move: its mean return, or a mixture of the mean and best return. I gave each root move at least two simulations in each world. These worlds came from practice data and the visible battle state; the real future draws stayed unavailable.

I selected 96 simulations per world with mean return. In a fresh bank of 160 naturally lost fights and 160 matched winning controls, my selected tree rescued 26 losses and kept 159 wins. My existing rollout search rescued 44 and kept all 160. I kept the existing search. The new tree plays many continuations with a random legal policy; deeper search alone didn't make those continuations reliable enough to justify deploying it. [Budgets, selection and fresh matched fights](results/terminal-search.json)


I checked how well the external public-search teacher actually plays from the same combat starts. On validation it won 112, 121, 128 and 125 of 160 fights at 64, 384, 2,048 and 8,192 simulations per decision. I'd pick 2,048 from that result. On the separate 179 fights those four budgets won 123, 133, 140 and 145. The original student won 127 and the logged expert had won 158. I used independent search particles and kept the student fallback identical on unhandled decisions. That isolates part of the gap: my original training labels came from a much weaker search budget than the best teacher tested here. These are fight outcomes on earned expert decks. [Budgets, matching and combat outcomes](results/search-teacher-ceiling.json)


On the second 32,768 ordinary starts, the frozen human draft and combat/reward searches reached the Heart 29 times and killed it once. Human drafting alone reached it 13 times and won zero. Across my two evaluations the combined player earned 63 encounters and four victories in 65,536 starts (0.0961%). [Independent replication](results/combined-human-search.json)

### Can a human campfire or event choice teach my router what to do after it arrives?

I reconstructed campfire decisions from the same archive of human A20 Ironclad Heart wins as my draft policy. To keep the HP input honest, I used the previous floor's HP and excluded cases where Eternal Feather was known to heal on entry. Smith targets competed against the subset of unupgraded cards I could reconstruct from earlier transactions. On held-out games the model identified the campfire action 79.7% of the time and the smith target 49.9% of the time. A uniformly sampled target within the reconstructed smith menus would get 19.7%.

On a fresh panel of 32,768 uniformly sampled 64-bit starts, a 35% campfire and smith mixture reached the Heart 28 times with six kills, against 21 and three. Isolating the decisions gave 30 encounters and six kills for campfire actions alone, and 19 and two for smith targets alone. I tested those on the same seeds after selecting the mixture. [Human campfire experiments](results/human-campfires.json)

I also mapped historical event button names to their simulator actions for 23 events. The mapping identifies the choices. I learned their scores from earlier human games. Event frequencies alone matched 75.7% of held-out decisions and adding HP, gold and floor matched 74.3%. I selected the deployment mixture on ordinary starts. A 45% contextual mixture reached the Heart 30 times and won six on 32,768 new starts, versus 22 and three with the same router and its earlier event choices. I checked the deployed scores against the training formula on 64 real menus; the largest difference was below 0.000001. [Human event experiments](results/human-events.json)


I tested the two human-trained decisions together on another 32,768 ordinary starts. The router alone reached 25 Hearts and won one. The event policy reached 30 and won one. The campfire and smith mixture reached 34 and won five. Both together reached 33 and won five. The campfire mixture now has 62 encounters across two 32,768-run panels against 46 for the same router without it. I retained that fast policy. [Shared seeds and all four arms](results/human-campfires.json)

### Should I simulate a few rooms before committing to a late route?

I reconstructed the visible Act 3 map, HP, inventory and keys in fresh worlds. Each legal route experienced the same set of independently generated encounters across a fixed horizon. On my initial audit I recovered 300 usable worlds across 100 branch decisions and changed 24 choices. In 384 earned selection positions the existing policy got seven Heart encounters and the best trial got six. On a separate full panel of 32,768 ordinary starts, all 1,644 Act 3 arrivals stayed identical, while the selected planner reached 421 first bosses against 344, and 27 Hearts against 20. Both had zero wins. The selection result and the fresh result point in different directions, so I started a replication using my retained campfire policy. [Public world checks, selection and whole runs](results/route-rollouts.json)

### Can I hand all the noncombat menus to an imitation model?

I started with a shop. Matching a recorded purchase is measurable, but I care about what happens after the purchase. I trained one scorer on 45,080 choices from verified public A20 traces and split games before fitting. I chose epoch 12 on validation. It matched 93% of held-out shop choices, 94% of removals and 67% of boss relic choices. The original player in that archive reached 65 Hearts and won 13 times in 994 games. Those are the demonstrator's results.

In my simulator, the best inventory mixture from the selection panel reached 12 Hearts on 32,768 fresh starts, against 18 with the existing player. I then let the existing player build its deck through Act 2 before switching just the inventory decisions. It reached 36 Hearts against 25 on selection. On fresh starts it tied at 27 encounters each, with two kills against zero. I kept the incumbent. [Training, menu agreement, early and late deployment](results/strategic-apprentice.json)

I also removed reward search from the combined player while keeping its combat search and human draft weights. On the same 32,768 starts it reached 18 Hearts and killed one. Restoring reward search gave 34 encounters and three kills. [Same-seed reward search comparison](results/combined-human-search.json)

### At the same HP, what changes if I still have the potion?

I saved 8,523 real postcombat reward screens and replayed them at four HP levels. When a potion was present, I kept it in one continuation and removed it in the other. The unchanged branch had to reproduce the original game's final floor, act, keys, Heart encounter and win. Across 53,647 continuations, I fitted the differences within each starting position. This avoids interpreting a bad deck losing both HP and a potion as evidence that spending the potion caused the loss.

On held-out positions, the mean squared error for the effect on reaching the next act fell from 0.0493 to 0.0453 compared with fitting a single scale to HP minus four per potion. The lightest inventory adjustment reached 34 Hearts against 27 on selection. It tied the original policy at 27 on 32,768 new starts and won once against zero. I kept the previous potion policy. [Counterfactual effects and ordinary starts](results/resource-effects.json)

### Where do I lose an already-developed deck?

I replayed the public reference traces to their first Act 3 screen. With all 192 original inventories restored and the noncombat router fixed, my fast combat player reached five Hearts and killed one. My locally trained combat imitation reached two and killed zero. The recorded player had reached 65 and won 13. From the 73 earned Act 4 positions my fast player reached 38 Hearts and killed two. I keep these conditional tests separate from the ordinary-start rate. [Restoration checks and outcomes](results/reference-runtime.json)

I also compared two searches on individual held-out fights. At 384 simulations, a shared public-belief tree won 143 of 179 fights versus 133 for separate trees. Eleven fights switched to wins and one to a loss. At 2,048 simulations, the same shared-tree mode still won 143. I selected the mode using different games. The independent search then won 140 at that budget. These runs used the published simulator's public search without its strategic Heart checkpoint. [Matched source fights](results/shared-search-screen.json)

On the retained campfire policy I repeated the 13-room Act 3 planning experiment. It reached 37 Hearts instead of 34 on 32,768 starts, with three kills instead of five. The first boss count increased from 395 to 460. Clearing the last two fights remains a separate bottleneck. [Late map replication](results/route-rollouts.json)

### Does route learning still help when combat and reward search are both on?

I changed only the map policy on the combined search and human-draft player. On the same 32,768 A20 starts, Heart entries rose from 34 to 60. Six seeds reached the Heart with both policies, 54 gained an encounter and 28 lost one. Both policies killed the Heart three times. The new route sent fewer games into Act 3 (2,664 vs 3,077), but more through both bosses and into Act 4 (80 vs 51). Only 8.1% of starts reached Act 3 at all. To reach my 20% Heart goal, I have to improve early and middle survival as well. [Same-seed route comparison](results/combined-human-search.json)

I also tried predicting the distribution of HP after each visible room and carrying it through the map. I collected 345,966 room continuations across my training and evaluation splits and checked the unchanged branches against ordinary play. On a faster player, planning over the last two acts reached 29 Hearts against 27 on selection and 24 against 19 on fresh starts. The fresh panel contained nine gained encounters and four lost. These counts warrant replication on the stronger search player. [Room model and two full panels](results/room-transitions.json)

### Can an accurate fight prediction tell me which reward to take?

I split 11,680 fights by game, with 9,446 for training, 1,127 for validation and 1,107 for the test. Adding the actual inventory reduced held-out fight utility error from 0.03170 to 0.03028. I then used the model to compare the deck before and after a prospective reward against the same opponent distribution. On 32,768 fresh starts the player reached 16 Hearts against 29 for its parent. The dataset let me check predictions about decks that actually fought. It gave me much weaker evidence about the effect of adding a particular card to a particular deck. I left this scorer out of the player. [Forecast accuracy and decisions in full games](results/encounter-forecast.json)

For resource use, I could make that comparison directly by replaying the same position at different HP or potion levels. Feeding those estimated costs into both the tactical search and the potion decision rescued 36 of 160 previously fatal fights against 35 for the original, and kept all 160 successful controls against 159. Seven fights improved and five got worse. That small net gain gives me little reason to spend a full panel on this change alone. [Matched fight test](results/resource-effects.json)

### What did learning the next card-selection prompt buy me?

I trained a separate scorer for the visible legal choices when a card asks for a follow-up. On held-out prompts it agreed with the recorded player 39.8% of the time versus 28.6% for picking the first legal option. On the same 192 earned Act 3 starts the scorer reached four Hearts and won twice, versus five Hearts and one win. On 256 ordinary starts neither reached the Heart. I also checked the public combat search on 48 Headbutt prompts from 17 games. Changing the concealed draw order while holding the displayed decision fixed preserved the chosen card, root visits and estimated root values on every tested prompt. [Follow-up training and privacy audit](results/public-followups.json) · [Full-game and earned-start comparisons](results/reference-runtime.json)

### Did the stronger combat search help across a whole run?

I gave the public shared tree four independent particles and 384 simulations on early ordinary moves, 2,048 on elites and late moves, and 8,192 on bosses. My strategic policy supplied the noncombat choices. Of 256 ordinary A20 Ironclad starts, 207 reached Act 2, 33 reached Act 3, seven reached Act 4, five entered the Heart fight and one won. Of the 174 runs that stopped in Act 2, 52 ended on the boss floor and 122 ended earlier. This gives me a reason to improve hallway and elite fights along with bosses. I launched a separate 32,768-start evaluation on fresh seeds. [Source-runtime results](results/reference-runtime.json)

I also tried learning which search budget and rollout method to use from the public HP and floor when each fight starts. I could compare four methods from the exact same fight and train directly on the differences. I fitted on 160 development fights with folds grouped by game. On 179 held-out fights the learned choice won 141, versus 143 for always using the cheaper shared tree. Randomly assigning the same mix had an expected 141.8 wins. Even selecting the winner after seeing all four results would rescue only seven more fights (150 wins). I moved on to improving the play itself. [Paired outcomes and simulation counts](results/search-allocation.json)

### Do completed decks teach me what to draft midway through Act 2?

I clustered the final inventories in my human training games and used the clusters as auxiliary labels for learning draft decisions. At decision time the gate sees only the cards and relics already acquired. Validation chose 16 clusters. On held-out card choices the log loss moved from 0.826 to 0.823, with little change in accuracy. In 32,768 fresh games the strongest mixture reached 25 Hearts and killed two, versus 26 encounters and zero kills for its parent. A single-cluster control at the same scale reached 22 and killed two. I kept the parent for my encounter objective. [Human choice prediction and full-run comparison](results/latent-draft.json)

### Can a mid-fight prediction choose between two plausible plays?

I trained on 114,558 public decisions from 7,996 fights. Each fight had equal total training weight. I gave a surviving fight 35 points plus exit HP and a loss zero. On 904 fights from held-out games, using the displayed hand, intents, enemies and statuses reduced mean absolute error from 12.81 to 8.43 score points. My exported decision trees matched the training library's predictions exactly on the sampled test rows.

I then compared two legal moves the search considered competitive. I played both across the same three independently sampled concealed futures and estimated the resulting position with the fitted trees. On 48 tested prompts, altering the real concealed draw order left both estimates and the selected move unchanged. Validation chose the stronger influence of the two settings I tried. On 179 held-out fights it changed 734 moves, rescued three fights and lost four, winning 142 versus 143. I kept the original combat search. The next experiment measures the effect of actually trying each candidate, directly supervising the decision instead of assuming that a good position forecast supplies that difference. [Prediction, privacy and fight outcomes](results/afterstate-critic.json)

I also reproduced the source search player's ordinary win on seed 2290664653572843552. A fresh simulator accepted all 933 recorded actions and reproduced the terminal hash. It entered the Heart with 42 of 42 HP and finished with nine. It collected Sapphire on floor 9, Ruby on 23 and Emerald on 40. The deck that reached the Heart held Corruption, Dark Embrace+, three Apparitions and the Dead Branch relic. [Verified outcome](results/reference-runtime.json)

### Can combat help choose the first boss relic?

My first pilot reused the same three artificial seeds at every chest. I changed the seed source to a hash of the public inventory and the current offers, so different chests sample different possible Act 2 futures. All candidates at one chest still get the same three worlds. I started a fresh evaluation for this version. For every relic, including skipping, I used those same worlds and played the first three fights under a cheaper search. In the actual run I retained my draft choice unless the alternative gained more than three score points across these simulated futures on average. The three-fight horizon covers a risk my ordinary player often encountered early in Act 2. It can also underrate a relic whose value builds up later. For six archived games, changing the underlying unseen future left the computed branch values identical. Continuing the original replay after planning reproduced its terminal hash. [Public-world checks](results/boss-relic-worlds.json)

I stopped an expensive late-boss search comparison after verifying the first 24 unchanged continuations. Only 631 of 32,768 benchmark starts had reached that switch point. Even winning from every one of those positions would put the encounter rate below 2% on that panel. I redirected compute toward decisions made earlier in the run. I have no comparative result for this stopped experiment.

### Does measuring the effect of switching the move change what I learn?

I saved 1,452 tactical decisions from 397 verified games. At each one I tried the searcher's move and one competitive alternative in three independently sampled futures, using the same chance realization for both. I played the fights to their end and trained on the difference in survival, exit HP and remaining potions. Replaying each original game still produced its recorded terminal hash.

On 132 held-out decisions from 36 games, the fitted rule switched six times. Its measured advantage averaged 0.08 score points per decision. A game-level bootstrap interval ran from −0.18 to +0.45. In the fixed 179-fight test it changed 209 moves, rescued three fights and lost one, reaching 145 wins against 143. On the separate 160-fight validation bank it won 127 against 130. I am testing it together with boss-relic planning on ordinary starts, restricted to the 384-simulation combat turns used in training. I am keeping the parent available while that test runs. [Paired data, held-out effects and tactical outcomes](results/move-trials.json)

On the fresh source panel, 222 of the first 256 ordinary starts reached Act 2, 53 reached Act 3, eight entered Act 4 and seven encountered the Heart. Two won. On the original 256 seeds, transferring the combined search player’s strategic choices increased Act 3 arrivals from 33 to 41 but reduced Heart entries from five to one and kills from one to zero. I kept the original strategic checkpoint for public shared-tree combat. [Same-seed macro test and independent tactical replication](results/reference-runtime.json)

I finished the independent native test. On 32,768 identical starts the original combined player reached 29 Hearts and killed one, adding only the learned route reached 55 and killed six, and adding the learned campfire policy as well reached 59 and killed six. The campfire change gained 41 and lost 37 encounters relative to routing alone. Across both full panels, routing raised encounters from 63 to 115 over 65,536 starts and kills from four to nine. The large reproducible gain here is routing. [Full paired comparisons](results/combined-human-search.json)

I also ran the source tree-sharing control on 256 ordinary starts. Both versions received identical sampled particles, rollout policy and total simulation budgets at each corresponding decision. Keeping a separate tree per particle reached three Hearts and killed one, against five encounters and one kill with a shared tree. Four seeds gained an encounter and two lost one. This isolates the search choice, although the number of Heart outcomes is small. [Source ablation](results/reference-runtime.json)

### Which changes held up when I finished the round?

I checked that all seven source variants used exactly the same 256 ordinary starting seeds. Every game ended within the action limit. Increasing the combat search budget on regular turns raised Act 3 arrivals from 33 to 46 and Heart encounters from five to nine. Five seeds gained an encounter and one lost it. The higher-budget player reached the Heart on all nine runs that entered Act 4 and lost all nine fights. Applying that same expensive search to follow-up prompts, such as which card to put on top with Headbutt, lost four of those encounters and gained none.

Planning the first boss relic using three simulated fights reached two Hearts and won zero, against five encounters and one kill for the parent. Adding the learned move correction made choices differently, but the same two seeds still reached the Heart. I kept the four-particle player as the replicated benchmark. The higher-budget regular-turn search is the strongest candidate on this panel. I paused before running it on fresh seeds. [Stage counts and paired comparisons](results/source-round-comparison.json)
