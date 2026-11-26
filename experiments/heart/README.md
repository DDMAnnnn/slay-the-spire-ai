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
