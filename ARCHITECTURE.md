# Slay the Spire AI: game interface and evaluation

Design draft, 18 November 2026. The first gameplay test is pending.

The goal is to win as often as possible using player-visible information and to get as far as possible on losing runs. For the first evaluation, use a fixed game version, play the Ironclad at Ascension 0, and count defeating the Act 3 boss as a win. Record the character, difficulty, and win condition in each evaluation configuration.

## Responsibilities during a run

| Component | Responsibility | Proposed interface |
|---|---|---|
| Game adapter | Wait until the game needs player input, read the visible state, list legal choices, and execute the selected action. Wait through animations and automatic enemy actions. | Emit a decision ID, scene type, observation, and candidate actions. Accept a candidate for that decision ID. If the decision has expired, read the state again to avoid playing a card twice. |
| Policy | Choose an action using the observation, the candidates, and information already observed during this run. Rules, search, or a learned model can implement the policy. | Receive a decision and return one candidate. Reset run-specific memory at the start of each run. |
| Run controller | Start the run, request decisions, execute actions through the adapter, and log each choice until victory, defeat, or interruption. | Record the configuration, observations, legal choices, actions taken, and outcome. Record why a run was interrupted and report the interruption count in evaluations. |

The run controller manages the seed. The policy may see the known contents of the draw pile and any ordering the game has revealed to the player. The same visibility rule applies to candidate actions. A map choice can identify a selectable node, but its metadata must exclude the hidden outcome of an event there.

## Evaluation and training

- **Evaluation.** Compare policies on the same seeds. Report wins, total runs, and floors reached. Reserve separate seeds for debugging and final evaluation. Save the game version, policy version, and configuration so the comparison can be repeated.
- **Training.** A learner can consume interaction data and produce a versioned policy. The controller must also run a hand-written policy independently. The game interface leaves the choice of network architecture and scene-specific models open.

## Decision: represent each choice as a list of legal actions

Combat, card rewards, and route selection present different kinds of choices in the same run. We considered assigning a fixed ID to every possible action and generating candidates at each decision. We propose the candidate list for the game-policy interface.

In combat, playing the second card against the left enemy and playing it against the right enemy are two candidates. At a reward screen, the candidates are the cards on offer and the option to skip. The controller follows the same loop in both cases. The tradeoff is a variable number of candidates. A model can score them individually or map them into fixed slots internally.

MathCard inspired the separation of action legality from action evaluation and already handled both combat and rewards. This design stands independently of its 37 action IDs, DQN implementation, and trained weights.

## First interface check

Use the same decision loop to play one targeted card and make one post-combat reward choice. We expect only the candidate list to change. If the game exposes card selection, targeting, or cancellation as separate decisions, record the actual sequence before deciding which steps to combine. Evaluate playing strength separately.

The follow-up implementation commit should identify the integration used, show what each scene exposed, and record which assumptions held or needed revision.
