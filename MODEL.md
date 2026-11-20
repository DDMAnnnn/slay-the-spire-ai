# Model notes

## 18 Nov 2026 — What can two cards share?

I want one set of weights to read damage wherever it appears. A hit from a card and a Lightning discharge share a quantity, but the source, target and timing can change the decision. I keep a mechanic's definition attached to the effect that references it. Adding 3 and 8 to Zap's immediate damage would teach the wrong lesson. I keep the printed order because applying Vulnerable before a hit can change its outcome.

I chose a shared effect layer followed by a recurrent card encoder. A card can have another effect without needing another set of weights. I keep the current cost and an explicit incomplete-reading flag. I still need combat experience to learn the weights that make a card useful in context.

## 18 Nov 2026 — What does the model actually choose?

When I asked about architecture, I meant the model itself. Suppose it can already read the game and I can record a run. How does it go from seeing the hand and the enemies to picking a move?

Take Strike and Defend against two enemies. If both cards are playable, I can Strike the left enemy, Strike the right one, play Defend, or end the turn. The same Strike might kill one enemy and barely scratch the other. Somehow the model has to score the card and its target together.

One suggestion I got was to score each legal move with the same small network. It would read the fight, the card, and the target. Strike would go through the same card encoder wherever it sits in the hand, and either target would go through the same enemy encoder. End Turn would have neither a card nor a target. A first pass at representing the fight could summarize the hand and enemies, keep their counts, and add HP, energy, and block. Order would still matter wherever it affects play.

My MathCard model gives me a simpler baseline: one output per action slot. Scoring moves individually costs time. What I could get in return is a model that learns something about Strike in slot one and can use it in slot four. I could check that by swapping two cards whose order makes no difference and seeing whether the scores follow them.

One question from the discussion is what gets lost in a hand summary. Sometimes I play a card because of what it lets me do with the next one. Would the model see that? I haven't tried either version yet. A fair first comparison would use the same fights and training budget and track both combat outcomes and time per move.

There's also a separate question for later. If the model learns when to play Strike, would that help it decide whether to take another Strike after combat? I can want the one in my hand right now and still skip the one on the reward screen.

## 18 Nov 2026 — Does each monster need its own model?

I started with two ideas. I could use one global model and give each monster a block in the input matrix. Or I could build a separate small model for each monster. Writing down a fight with two enemies of the same species exposes an ambiguity in my first idea. If a block stands for a species, where do I put the second enemy's HP and intent? If a block stands for an enemy on screen, each one already has its own row and can be targeted separately.

A third idea came up in the discussion: keep one row per enemy, but use the same encoder to read every row. Species would be one of the inputs. That way the model could use something it learned about HP or incoming damage across species. It would still need the hand and the rest of the fight to decide which enemy to hit. A separate model for a species would answer a different question if its job were to predict that enemy's next move.

I haven't chosen between these designs. A cheap first check would be to swap equivalent enemy rows and see whether the scores still point to the same targets. A more interesting test would be to compare fixed slots with a shared encoder on duplicate enemies and unfamiliar groups. Would sharing actually improve targeting on the same training budget? Attention also came up as an option if the simpler model struggles with interactions.


## 18 Nov 2026 — How the first combat model fits together

By *intent* I mean the monster's whole behavior pattern: moves, effects, cycles, conditions and random branches. I'll use *current-turn intent* for the one move telegraphed right now. My idea is to give each monster type a place to learn its pattern while keeping one player policy that chooses what to play. Here is a concrete version to try. It hasn't been implemented.

1. **Combat inputs.** At each player decision, read HP, block, energy, hand, public pile information, active effects, and the legal moves. For each enemy, read its visible type, HP, block, powers and current-turn intent. Keep a separate history per enemy in the encounter. Record a telegraph change or an executed enemy move when it happens. Playing three cards against an unchanged telegraph must not advance the cycle three times.
2. **Monster pattern memory.** Give each monster type and Ascension setting its own training records. As a first baseline, count next observed moves after matching the recent move sequence, including the current telegraph, in previous training fights. Back off to shorter matches when evidence is sparse. Output a distribution and the number of matching examples; an unseen type returns an unknown flag. Keep public context alongside the sequence so a later model can distinguish triggers such as HP thresholds. This first table can miss conditional rules. The entire learned relation is the estimate of intent; one prediction only describes the next move. Fit from training fights and freeze the table during evaluation. The current fight's observed history can still advance.
3. **Shared player policy.** As a first baseline, concatenate a fixed-slot hand representation, player state, one enemy block and the pattern-memory output. Pass the resulting vector through a small feedforward network. It produces one score per hand slot plus End Turn. Mask illegal choices. With one enemy the target is implicit. One set of player weights serves every encounter; a new enemy can start with empty pattern memory. For multiple enemies I'll need a target-aware output. A shared candidate scorer is one option to compare at that point.
4. **Learning signal.** The pattern memory learns from observed enemy moves. The player policy learns from combat outcomes. The test reports win rate first and HP lost on wins alongside the win count. The precise training reward and update rule are still open.

I'd start with Ironclad, a fixed starting deck, one enemy species, shuffled draws and an empty potion belt. The first milestone is a complete fight on held-out seeds. Then compare two player policies with the same inputs, capacity and training budget, except that one always gets the unknown forecast. Check prediction quality against the per-type move frequencies as well as combat outcomes. A predictor could learn the cycle perfectly and still never change the best card to play in a particular fight.

If two species share a pattern, their histories may eventually be pooled. For that to justify treating them alike in combat, other relevant mechanics such as passive powers must also match or be represented in the inputs. For the prototype I can keep records separate until there is evidence for pooling.


## 18 Nov 2026 — A small prototype with a full-run design in mind

I want to start with the Ironclad on A20. The prototype can be one fight. The design has to leave room for the decisions I care about across a whole run. One change from the earlier sketch: using a potion belongs in the combat model from the start.

I see two timescales. In combat I choose a card and target, use a potion (with a target if needed), or end the turn. Outside combat I build the deck, choose rewards and potions, spend gold, decide whether to rest or upgrade, and choose a path through events, campfires, and elites. These choices draw on the same HP, deck, inventory, gold, and visible map. I haven't decided that each one needs its own network.

One interface suggested in the discussion is to evaluate what a fight leaves me with. The combat outcome should retain survival, remaining HP, remaining potions, and any other persistent changes. A later run-level model could estimate how valuable that resulting state is given the deck and visible path ahead. This matters for potions: keeping one can be valuable, but the cost of spending it changes with the run. Rest versus upgrade and route choice should likewise be evaluated in the context of the same run.

For a first test I could use one enemy, the starter deck, varied draws, and a single targeted potion type. Offer the legal potion action alongside the cards. Compare fights with and without a potion available, recording use, survival, and HP remaining. A one-fight test tells me whether the agent can make tactical use of a potion. To test whether it learns when to save one, I need a consequence after this fight, such as carrying the potion into a second encounter. The monster intent memory and its no-forecast comparison from the earlier sketch still fit here.

Once the A20 Ironclad policy works, I want to see what happens if I lower the Ascension. A change in win rate would reflect both the change in difficulty and any change in enemy behavior. I could freeze the policy for the first test and compare encounters with changed patterns against those with unchanged patterns. Other characters are a separate extension. The full-run victory condition (Act 3 or the Heart) is still open.

## 19 Nov 2026

**What if a card hits once for every relic I own?** That was where the card reader stopped making sense to me. It could translate “deal 5 damage twice,” but “twice” was baked into the label. Giving it another label for three hits would teach it very little about my relic card. I want the rule to say `Repeat(Count(PlayerRelics), Attack(Target, 10))`. With three relics that's three hits of 10. Swap the arguments around and I get ten hits of 3. Even with the same total damage, Strength changes the two attacks by different amounts.

That gave me a way to go through the card library. Start with Strike and define one hit. Twin Strike introduces repetition. By the time I get to Barrage, I already have both pieces. The repeat count comes from occupied orb slots. Spirit Shield was a useful check on that decision. “For each card in hand” changes the amount of a *single* block gain. A power that triggers on gaining block can tell the difference. The reference table now covers 370 registered card IDs. Its check verifies IDs and rule syntax. Testing every interaction in a fight is still open. [Original card table](CARD_TABLE.md)

**Do I need to solve reading before I can learn to play?** I originally put those together because I want to transfer from Ironclad to Silent and eventually to mods. I can give the combat model hand-authored rules for the original game. That lets me ask whether it can use Bash well when it already knows what Bash does. A future text reader can produce rules in that same format.

The existing reader fits logistic regression to labeled clauses. It can recognize a familiar damage phrase and copy the number. I also have a text encoder that keeps unfamiliar wording and keyword definitions available. Changing a definition changes its vector in the test. Whether it understands the change is a different experiment. For that I'd hold out an entire mechanic, give it the tooltip under a made-up name, and change “turn start” to “turn end.” Does the predicted trigger change while the amount and target stay the same? For transfer in *play*, I'd freeze the Ironclad model for the first evaluation, then compare its learning curve on the new character with training from scratch on the same fights.

**Where did my 1 × 292 vector go?** In my previous game I flattened the state before sending it to the network. The PyTorch detail I wanted to understand was what happens if I give a layer three rows. A linear layer from 64 to 128 features takes a hand of shape `[3, 64]` and returns `[3, 128]`. Each card gets the same weights. Drawing a fourth card adds one row. There's still no interaction between the cards at that point.

For the new design, I want to encode each card's rule tree into a vector and keep a row for each card in hand. The tree preserves which number supplies the damage and which supplies the repeat count. Current cost belongs with the card instance. Strength belongs to the player, and the damage value the bridge reads may already include it. I need to keep base damage and the current preview distinct or I could count Strength twice.

The piles raised a different question. Two Strikes in the draw pile matter more than one, but rearranging an unknown draw order should give exactly the same input to the decision. I'd sum the card representations and include the pile size, keeping draw, discard and exhaust separate. Headbutt is the exception that helped me pin this down. If I just put Bash on top, I know my next draw. I'd keep known positions separately and pool the remaining cards. The bridge currently sorts the pile, so recovering that knowledge will take public move history. Finisher gives me a similar check on history within a turn: two identical-looking hands can play differently if I've already used three Attacks.

**Why I want attention for this fight.** Say I have Bash, Twin Strike and Defend, a Strength Potion and a Fire Potion, and two enemies. I want Twin Strike's score to depend on which potion I can drink and which enemy I can kill. Joint self-attention was suggested as a way to let each object's representation depend on the others. That's why I'd choose it for the combat model. A single hand summary might preserve the overall balance of attacks and skills while losing the particular combination I'm deciding whether to use.

Here's the shape calculation I want to be able to reconstruct. Assume both potions and all three cards are usable. The widths are design choices. This example counts only the objects listed.

| Step | Shape | What I keep |
|---|---|---|
| Encode one player, three cards, two potions and two enemies | `[8, 64]` | One row per object, with its type |
| Let the objects attend to one another | `[8, 8]` weights per head → `[8, 64]` contextualized rows | Twin Strike can use information from Bash, either potion and either enemy |
| Enumerate legal moves | 9 moves | Four targeted attacks, Defend, three potion uses including their targets, and End Turn |
| Gather one input per move | `[9, 208]` | Action kind (16), source (64), target (64), player context (64), with designated vectors for missing arguments |
| Score every row with shared weights | `[9, 208] → [9, 128] → [9, 1]` | One score for each legal move |

“Twin Strike on A” and “Twin Strike on B” use the same card row and different target rows. Drinking the Strength Potion could change the card row itself at the next decision. I'd test the wiring by swapping two hand slots where order has no gameplay effect. The scores should follow the cards. Then I'd vary just the enemy HP around Twin Strike's lethal threshold to see whether the *trained* model actually uses the target. The rule table and earlier encoder are in the repo. Connecting the rule trees to this attention model and training the action scorer are still ahead.

**Who gets to ask the question?** I can put Bash and Twin Strike through the same card encoder, but doing that alone gives each one no information about the other. With self-attention each battle object produces a query, a key and a value. The query picks up a weighted mixture of values from the objects in this fight. I can reuse the resulting Twin Strike row for either target. The scorer still gets the particular enemy I chose.

Cross-attention would give each proposed move its own query. That would let “Bash on the left” retrieve a different follow-up card from “Bash on the right.” My reason for choosing object self-attention is that the bridge already tells me the card and the target for each move. I can preserve both and share the work of reading the fight across moves. I'd revisit action queries if the scorer consistently misses a third object, for instance when only one Bash target can be killed by the follow-up. I'd compare on the same fights and training budget, including time per decision. Attention weights on their own wouldn't show that one design plays better.

## 20 Nov 2026 — What if I just give it the sentence?

"Deal 6 damage." "Deal 10 damage." "Deal 6 damage twice." I kept asking how I could give the model the meaning of a card. Looking at those three sentences together made me ask a smaller question. If the words and numbers go into the combat model, can playing teach it enough to choose a fourth card it has never seen?

I took a detour through a separate interpreter because I wanted an unfamiliar card to make sense on the first encounter. Writing down the original card rules helped me see why the amount and the hit count belong in different places. Training a reader to choose among eleven templates helped with unfamiliar wording too. Expanding its examples moved that score from 0/22 to 17/22. But 339 of the 370 original base-card rules were outside what those templates could express. I was choosing a large part of its possible answer before it had even read the card.

I still like the idea of learning from combat. After seeing Pommel Strike sixteen times, the memory model expected about 0.92 cards to enter the hand. Then we tried the same play with and without No Draw. With six examples of each kind in memory, it predicted about 0.63 in both situations. That was the point where I had to separate two questions I'd been treating as one: what a card says it does, and what actually happens in this fight. The memory experiment gave me a way to test that difference. Solving it had also become a project of its own.

For the combat model I tried letting the action scorer learn from the printed description directly. Both versions saw the ID. One also saw the words and numbers. I held out ten damage twice. On one-decision fights with that card against a familiar alternative, IDs alone won 72.7% and IDs plus text won 88.8%, averaged over three runs. My favorite check was swapping the wording behind two unfamiliar IDs while keeping the hand slots and enemy identical. The correct choice switched. IDs alone got 50%; the version with text got 83.3%.

The misses tell me what to ask next. Ten twice beat ten once, and the text model picked twelve twice over ten twice. But put ten twice against twenty-two once and change my Strength: which card wins now? On the swap test it only got half of those right. I have evidence that the wording can change a decision. I still want to see whether it can learn *when the ordering changes*, and whether the advantage survives a whole fight. [Experiment and per-seed results](experiments/card_text/README.md)

## 20 Nov 2026 - Same hand, different Strength

If the card still says "Deal 10 damage twice" at every Strength, can watching what happens teach the model when to play it? I wanted to change just my Strength and see whether the choice changed. In the last experiment the enemy's HP changed too. This time I kept it at 100 and asked which attack would deal more damage. That let me show exactly the same two cards twice.

I left ten twice out of training. Against twenty-two once, the models trained on random hands got both decisions right in 96/96 cases across three runs at 0 and 4 Strength. Giving half the training budget to pairs where Strength flips the answer got 87/96. I didn't need all those paired examples for this comparison. I did need the paired *test* to see what changed the choice.

Then I tried 1 and 3 Strength with the weights frozen. Both numbers were new in training, and now the Strength icon was present in both cases. At 1, ten twice deals 22 and twenty-two once deals 23. At 3, ten twice wins 26 to 25. All six models got the second choice right. Most missed the first. The random schedule got both right in 13/96 cases and the paired schedule in 25/96. I have a reason to keep the base wording. I also have a much better test than asking whether the model recognizes the word Strength. [Strength experiment](experiments/card_text/README.md#can-the-policy-use-strength-with-only-base-wording)

## 20 Nov 2026 - What if I tell it how much damage it dealt?

I was giving the model a 1 for choosing the better attack. If I already know the enemy lost 22 HP, why throw the 22 away? I kept the same hands and trained it to predict how much HP each played action took off the enemy. In a separate run I let the simulator answer for the other card too.

Ten twice was still missing from training. With random practice and feedback only on the played action, the average prediction went from 19.65 to 27.67 as my Strength went from 0 to 4. The actual numbers are 20 and 28. But at 1 Strength it predicted 21.61 for ten twice and 21.79 for twenty-two once. The order of those averages is right. The real gap is a whole point, and the individual choices were much less reliable. It got the 1-and-3 pair right 50 times out of 96, up from 13 with binary feedback. On the paired practice schedule the same change took me from 25 to 19. I want to keep the damage prediction because I can now see where the estimate goes wrong. Choosing the move that wins the fight is still a separate job. [Exact damage experiment](experiments/card_text/README.md#training-on-exact-damage)
