# Step 3a: source-plus-sink key validation

Generated 2026-09-02 22:21:35. Condition SCRUB, which is what Step 3c would run.

## Side by side

| metric | sink-only key | source+sink key | stop threshold |
| --- | --- | --- | --- |
| distinct keys | 339 | 759 | |
| both-class keys | 66 (19.5%) | 41 (5.4%) | |
| **samples covered by both-class keys** | **53.8%** | **9.0%** | **>= 35%** |
| pure0 keys / samples | 130 / 24.2% | 235 / 39.9% | |
| pure1 keys / samples | 143 / 22.0% | 483 / 51.2% | |
| mixed keys / samples | 66 / 53.8% | 41 / 9.0% | |
| size max / p90 / median / min | 2392 / 602 / 13 / 1 | 1001 / 163 / 43 / 1 | |
| worst-case fold deviation | 10.9 pts | 6.2 pts | <= ~10 pts |
| **test-fold keys at seed 0** | **116** | **228** | **>= 40** |
| test-fold samples at seed 0 | 17,721 (31.0%) | 17,313 (30.3%) | |
| disjointness overlap | 0 | 0 | must be 0 |

## Verdict: STOP 2 fires. The grid was not run.

Both-class coverage falls from 53.8 percent to 9.0 percent, far below the
35 percent threshold. STOP 1 passes comfortably (228 test keys against a
floor of 40) and fold skew passes at 6.2 points, so resolution is not the
problem. Coverage is.

The mechanism is visible in the polarity buckets. Pure-positive keys go
from 143 covering 22.0 percent of samples to 483 covering 51.2 percent.
The source segment is where polarity lives in source-varying testcases:
"initialize data as a small string" against "initialize data as a large
string" share a sink but differ in source. Joining source and sink
therefore assigns the good and bad members of one pair to two different
compound keys, which is the same failure the naive whole-description key
had, reappearing one level up.

So the compound key does not measure what Step 3 set out to measure. It
fragments the good/bad pairing rather than removing structural overlap
between folds.
