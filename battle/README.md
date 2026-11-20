# One combat decision

I started with two choices, playing the same Twin Strike on the left enemy or on the right. The game already offers both choices. I want to read the card once and score both uses of it. The checkpoint produced by init has a trained text reader and random policy weights. Training the policy on recorded choices is a separate operation.

| Stage | Input | Output and reason |
|---|---|---|
| Observation | A public decision and its legal action strings | One row for the player and one per card instance, enemy, potion slot, relic and orb. Each card includes its zone. Powers stay with their owner. Two copies can share a meaning while differing in cost. Empty slots and pile counts retain resource capacity and multiplicity. |
| Interpreter | Object kind, ID, text and available metadata | An ordered program tagged catalog, predicted or unresolved. The vanilla card table supplies 370 reference rules with upgrade variants. A small supervised clause classifier shares language features across object kinds and copies numeric arguments into templates. An unknown identifier and available text or glossary stay visible when it abstains. |
| Object encoder | Each row's rule tokens and observed numbers | An ordered GRU reads the tokens. Numeric literals share an embedding and a continuous magnitude. Other scalar fields have a signed log magnitude and a presence bit. These signals combine into a 64-number row. This preserves operand order and distinguishes a missing field from zero. |
| Battle encoder | N rows of width 64 | Two layers of four-head self-attention return N contextual rows. No arbitrary pile positions are supplied. Orb slots carry their meaningful order. Width, depth and head count are capacity choices and have no win-rate evidence yet. |
| Action scorer | Player row, source row, target row and action kind | The same network scores each legal choice. With A choices its input shape is [A,208] and its output is [A]. A learned empty row handles absent sources and targets. The highest-scoring legal string can go straight to the game. |
| Feedback | Ticket-matched choices, next frames and visible HP changes | Behavioral cloning uses the requested action as a label. An auxiliary head predicts player HP change, target HP change and the number of visible decreases in target HP before the next decision. Missing labels are masked. A probability vector beside each predicted rule lets these losses update the reader while supervised rules anchor its meaning. |

## Run it

From the repository root in PowerShell with Python 3.10+ and PyTorch:

```powershell
$python = '.\.venv\Scripts\python.exe'
$ckpt = 'build/battle-v2/reader-and-policy.pt'
& $python -m battle init $ckpt
& $python -m battle read $ckpt card 'Deal 13 damage 4 times.'
$traces = @(Get-ChildItem "$env:USERPROFILE/SpireCombatAI/recordings/*.jsonl" | ForEach-Object FullName)
& $python -m battle inspect $ckpt $traces[0]
& $python -m battle train $ckpt build/battle-v2/imitation.pt $traces
& $python -m unittest discover -s tests -v
```

To select this policy in the existing F8 game session, run `.\.venv\Scripts\python.exe -m battle play build/battle-v2/imitation.pt` from the repository root. The bridge accepts the same atomic play-slot-target commands as before. New schema-2 recordings require the rebuilt mod jar. Older traces work with the fields they contain.

The [reader dataset and evaluation](data/reader/README.md) include the original comparison, split definitions, failure examples and commands to reproduce the scores.

## What the measurements mean

The text reader has eleven supervised program templates, shared across card, potion, enemy, relic and power clauses. It reads short English clauses and some combinations. Its confidence threshold is a heuristic. Unknown concepts, localization, novel control flow and wording made entirely from familiar words can still defeat it. Known enemy attack intents already arrive structured and go straight into a rule. A potion or power with an unseen ID can use its supplied description. Older recordings often have only the ID. A reference entry uses its canonical rule even if a mod reuses the same vanilla ID and overrides its behavior.

Catalog parameters describe fresh cards. Schema 2 additionally provides the card instance's base values, its display values and target kind, and potion descriptions and current potency. Display values are snapshot features. The encoder does no damage simulation. Full deck construction, hidden pile order and enemy future moves have no inputs here.

The upgraded recorder samples visible HP once per game update during an action window. Separate observed drops can distinguish some multi-hit effects. Multiple changes within one update can merge, a blocked hit can have no HP change, and a relic can cause a drop. No event assigns sole credit to the chosen card. An interrupted recording can still supply a demonstrated action. Only observed outcomes become auxiliary labels.

I used 21 recorded choices for the first imitation pass. They supplied 21 player HP deltas and eight target HP deltas. These recordings predate the per-update HP observations. I can use them to exercise training, but a fresh fight is needed to measure the policy's play and verify the new telemetry in-game.
