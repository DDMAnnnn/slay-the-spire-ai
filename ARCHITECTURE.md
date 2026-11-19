# Notes on running the game

## 18 Nov 2026 — What we want to be able to see after a loss

The goal is simple: win more runs, and see how far we get when we lose. One suggested starting point is Ironclad on Ascension 0, counting the Act 3 boss as a win. We'd save those settings and the game version so a later comparison still means something.

The old MathCard game already knew which moves were legal. That seems worth carrying over. It had 37 action slots of its own, though, and there's no particular reason Spire should inherit those. One idea is to ask the game what choices the player has right now. During a fight that might include playing a card on either of two enemies. A minute later it might be three reward cards and Skip. Either way the policy can pick from the list it gets.

A detail to watch when we hook this up: the game might ask for a card first and a target second, with a chance to cancel in between. We should watch one real card play and one reward choice before settling on how to represent them. Waiting for player input and tying each choice to its decision ID should also keep a slow response from playing a card twice during an animation.

What I'd want from the first replay is to pause before a bad move and see exactly what the policy could see, what else it could have done, and what it picked. The runner can keep the seed for replay. The policy gets the draw-pile information a player could know, including the order only if it's been revealed. Same goes for map nodes: the choice to visit an event shouldn't give away its outcome. Any memory the policy keeps starts fresh each run.

A rule-based player would be enough to check this wiring. Later we can run two policy versions on the same seeds and compare wins and floors reached, keeping a separate set for final evaluation. We should save the policy version and settings, and mark a run that broke halfway through as interrupted. Otherwise a bug in the adapter could look exactly like a loss. For now these are notes for the first integration test; the [model notes](MODEL.md) are where the architecture discussion picked up.
