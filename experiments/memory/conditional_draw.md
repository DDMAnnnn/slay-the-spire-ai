# Does Pommel Strike draw when No Draw is active?

I wanted to distinguish remembering a common result from remembering when it applies. I cloned an A20 simulator state just before a legal Pommel Strike and set the simulator's No Draw status on one copy. I played the same card on the same target in both copies. The collector exports the status as a public player power. The evaluator verifies that this is the only difference in the two public inputs and checks the actual hand arrivals after both plays.

I used the original starter-trained weights, 12 examples from independent adaptation fights (seeds 10000–10039), and 91 paired states from 120 fresh fight seeds (11000–11119). Each fight supplied at most one nonterminal pair with cards available to draw. Six of the balanced adaptation examples had drawing allowed and six had drawing blocked. The same 12 source fights supplied the allowed-only control. I scored integer predictions using the existing nonnegative card-count readout. All 91 pairs had one actual arrival when allowed and zero when blocked.

| Memory condition | Mean prediction, allowed | Mean prediction, blocked | Correct on both sides |
|---|---:|---:|---:|
| Weights only | −0.051 | −0.059 | 0/91 |
| Starter memories | −0.044 | −0.052 | 0/91 |
| Starter + 12 allowed Pommel plays | 0.920 | 0.912 | 0/91 |
| Starter + 12 balanced Pommel plays | 0.631 | 0.631 | 0/91 |
| Online, starting with starter memories | 0.540 | 0.532 | 3/91 |

The first four runs kept the bank fixed. In the online run I scored both alternatives before revealing either outcome. I then stored one branch, alternating allowed and blocked by fight, for use on subsequent fights. The bank grew from 168 episodes to 259. Weights stayed fixed throughout. This deliberately tests adapting through memory to an interaction absent during starter-deck training. The online model shifted its average forecast but seldom changed its rounded answer correctly across the two conditions.

To reproduce with the compiler runtime on PATH, rebuild the collector from the pinned checkout and run from the repository root:

```powershell
$python = '.\.venv\Scripts\python.exe'
$dir = 'build/memory-conditional'
& $python -m experiments.memory.build_collector build/sts_lightspeed "$dir/collect.exe"
& "$dir/collect.exe" 10000 40 1 --paired-draw > "$dir/adapt.jsonl"
& "$dir/collect.exe" 11000 120 1 --paired-draw > "$dir/test.jsonl"
& $python -m experiments.memory.conditional_draw build/memory-final/starter.pt $dir
```
