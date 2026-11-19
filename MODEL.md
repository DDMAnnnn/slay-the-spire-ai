# Combat policy architecture proposal

18 November 2026. This proposal assumes the decision interface in [ARCHITECTURE.md](ARCHITECTURE.md). Training and validation are pending.

## What the model predicts

Given a player-visible combat state and a list of legal actions, assign a preference score to each action. A targeted card play specifies both the card and its target. The training objective will determine how to interpret the score.

## How to score a candidate

1. Encode each card in hand with a shared card encoder. Encode each enemy with a shared enemy encoder.
2. Aggregate the card and enemy representations. Combine them with player health, energy, and block to form a state vector. Retain entity counts and explicitly encode position or order wherever it affects the rules.
3. For each candidate, pass the state vector, action type, card representation, and target representation through the same small feedforward network. It returns one scalar. For actions such as ending the turn, mark the card and target as absent.

## Why try this architecture?

A feedforward network with a fixed output slot per action is the simplest baseline. A shared scorer lets the model apply the same decision rule across hand positions and enemy targets, at the cost of evaluating each candidate. A simple aggregation may also lose information about combinations of cards. If gameplay analysis reveals that weakness, compare an architecture that explicitly models interactions. Encode ordered zones separately from unordered collections.

The first proposal covers combat. Whether reward selection and routing should share representations or use separate scoring heads remains open. MathCard's legal-action mask informed this design. It assumes fresh model weights.

## How to test the design

- Reorder cards or targets whose positions are equivalent under the game rules. Each action's score should follow the corresponding card and target through the reordering.
- Compare the shared scorer with the fixed-slot baseline using the same visible information, data, and training budget. Measure combat outcomes and inference cost. The reordering check tests a structural property. The baseline comparison tests whether that property is worth its cost in play.
