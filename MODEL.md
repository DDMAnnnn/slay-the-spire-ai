# Model notes

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
