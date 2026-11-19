# Notes on running the game

## 18 Nov 2026 — What I'd want to see after a loss

I want the AI to win more runs and, when it loses, to get further. One starting point suggested in the discussion was Ironclad on Ascension 0, counting a win when it beats the Act 3 boss. If I use that setup, I'll save the game version and settings alongside the result so I can repeat the comparison.

My MathCard game already handled move legality and used 37 action slots. One idea I can borrow is having the game tell the policy which moves are available. During a fight that might include playing the same card on either of two enemies. At a reward screen it might be three cards and Skip. The policy could pick from the list in both cases without inheriting the old game's action numbers.

Before settling on that interface, I'd like to watch one actual card play and one reward choice. Does the game ask for the card first and the target second? Can I cancel in between? A decision ID might help here: if a response arrives after the game has moved on, the runner could discard it instead of accidentally playing another card.

When I replay a loss, I'd like to pause before a bad move and see what the AI knew, what it could have done, and what it picked. The runner can save the seed to reproduce the run. The policy should only see draw order when the player can see it, and an event node on the map shouldn't give away the event outcome. The policy's memory should start fresh each run.

A rule-based player would let me check all of that before training. Later I could compare two policies on the same seeds, keep a separate set for final evaluation, and record wins and floors reached along with the policy version and settings. I'd log interrupted runs separately so I can tell a broken adapter from a lost fight. The [model notes](MODEL.md) pick up the architecture question I originally wanted to discuss.
