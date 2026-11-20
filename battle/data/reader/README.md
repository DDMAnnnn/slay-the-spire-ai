# Reader data and evaluation

I wanted to know whether changing 5 to 13 really tested the reader. Both amounts become the same input token. To measure something different, I held out whole phrasings and included descriptions where returning a familiar rule would change the target, timing or number of hits.

- `templates.json` defines the eleven output classes, valid object kinds and copied arguments. The classifier learns weights for these classes.
- `train.jsonl` holds 150 hand-authored, labeled object/phrase examples. They cover 104 distinct word sequences after replacing numeric values, with some wording shared across object kinds. The 30 original examples carry an origin tag. Every training example has a supported label.
- `validation.jsonl` contains 22 cases and `test.jsonl` contains 51. I fixed the classifier, optimizer, seed (19) and confidence cutoff (0.85) for the comparison. I used validation as a diagnostic. I did not fit or select parameters on either split.
- `legacy_schema.json` preserves the feature and class order for format-1 checkpoints. New checkpoints carry their own vocabulary and templates. Inference from a new checkpoint needs no training examples.

## Results

Counts below give exact programs for supported descriptions and correct abstentions for unsupported descriptions. Numeric-copy cases deliberately reuse the wording of training examples. The split test checks that the other held-out descriptions, after normalizing case and numbers, appear in neither training nor an earlier evaluation split. Compositions reuse familiar individual clauses in new sequences. These are small, hand-authored English evaluations by the same author as the expanded training examples.

| Cases | Original | Expanded |
|---|---:|---:|
| Validation, held-out wording | 0/11 | 10/11 |
| Validation, ordered clauses | 3/3 | 3/3 |
| Validation, unsupported descriptions | 7/8 | 7/8 |
| Test, new numbers | 11/11 | 11/11 |
| Test, held-out wording | 0/22 | 17/22 |
| Test, ordered clauses | 6/6 | 6/6 |
| Test, unsupported descriptions | 7/12 | 6/12 |

The expanded reader abstained on five held-out phrasings. On unsupported cases it assigned six incorrect programs. For example, it mapped “Deal 5 damage to yourself” to `Attack(Target,5)`. Teaching it the word “yourself” through self-buff examples made that word available to the classifier. A high softmax score alone does not establish that the target or timing is correct.

## Original game text

The game audit runs the English base description of every registered card through the text classifier with ID lookup bypassed. The expected programs come from the project's hand-authored card catalogue. The audit uses exact canonical program strings. It covers the shipped base wording and includes familiar phrases. It does not measure behavior in real mods, semantic equivalence of alternative programs, or combat win rate.

Only 31 of the 370 reference programs have a structure that the current card templates and clause concatenation can express. Both models matched 20 of those 31 and abstained on the other 11. Both abstained on all 339 outside that output vocabulary. End-to-end text-only coverage was 20/370 (5.4%). The production path still uses catalogue lookup for known original IDs.

## Reproduce

From the repository root in PowerShell:

```powershell
$python = '.\.venv\Scripts\python.exe'
$ckpt = 'build/battle-v2/reader-and-policy.pt'
$jar = 'E:/SteamLibrary/steamapps/common/SlayTheSpire/desktop-1.0.jar'
& $python -m battle init $ckpt
& $python -m battle.evaluate_reader --original --game-jar $jar
& $python -m battle.evaluate_reader $ckpt --game-jar $jar
& $python -m unittest discover -s tests -v
```

The game jar is optional. Add `--errors` to inspect each missed or incorrect prediction. The new checkpoint contains the fitted interpreter and a freshly initialized combat policy.
