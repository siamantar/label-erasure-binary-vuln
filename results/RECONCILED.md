# Reconciled results

Every figure here is computed directly from `results/runs/*.json`, the 189
per-run records, which are the authoritative store. The aggregated `.txt`
reports in this archive are human-readable derivatives and were not used as
inputs.

`results/` was not modified. `stratification.txt` and `gate_measurements.txt`
are the only new files, and both were written into `deliverables/`.

## 1. The seed-averaging convention

**Chosen: average all available model seeds within a partition, then take the
mean and standard deviation across partitions.**

Where a partition carries several model seeds, their mean is a lower-variance
estimate of that partition's value, so it is the better input to a
partition-level standard deviation. Four reasons for this choice:

1. It uses all 30 BiGRU runs rather than discarding 20 of them.
2. It is a lower-variance estimator of the quantity the partition SD describes.
3. It is deterministic and reproducible.
4. Step 4's variance decomposition already used it, so adopting it means the
   project reports one number rather than two.

**Under this convention, Juliet SCRUB BiGRU, CWE-disjoint, max_len 256, over 10
partitions is 82.33 +/- 3.06.**

### Asymmetry to disclose

Only the BiGRU has three seeds per partition. The logistic regression is
deterministic and has no seed. The transformer has one seed per partition. So
BiGRU per-partition estimates carry less within-partition noise than
transformer estimates, which slightly favors the BiGRU in any comparison
between them. This does not affect the logistic regression, which has no seed
noise to average away.

## 2. Why `step5c_three_way.txt` is excluded

That file is deliberately absent from this archive. Its BiGRU column is not
reproducible.

The generating script assigned into a dictionary keyed by partition, so for a
partition holding three seeds the last file returned by `glob` won. The seed
that survived therefore depended on filesystem enumeration order.

**The seed sequence it actually used, by partition 1 through 10:**

    1, 1, 0, 1, 2, 2, 0, 0, 1, 0

That is a mixture of all three seeds, not a fixed choice. Three candidate
figures for the same cell:

| Convention | Value |
| --- | --- |
| seed-averaged over 3 seeds, **chosen** | **82.33 +/- 3.06** |
| glob-order mixture, `step5c_three_way.txt` | 82.52 +/- 3.17 |
| seed 0 only | 82.39 +/- 3.39 |

The middle row matches no stated convention and should not appear anywhere.

## 3. Three-way model comparison

SCRUB, CWE-disjoint, max_len 256, 10 partitions, seed-averaged. Baseline is
each test fold's own majority class, averaged the same way.

| Corpus | Model | k | Accuracy | Baseline | Lift |
| --- | --- | --- | --- | --- | --- |
| Juliet | Logistic regression | 10 | 80.50 +/- 3.73 | 55.21 +/- 2.71 | +25.29 |
| Juliet | BiGRU | 10 | 82.33 +/- 3.06 | 55.21 +/- 2.71 | +27.12 |
| Juliet | Transformer | 10 | 77.96 +/- 4.50 | 55.21 +/- 2.71 | +22.76 |
| NDSS18 | Logistic regression | 10 | 89.22 +/- 3.57 | 87.06 +/- 4.30 | +2.16 |
| NDSS18 | BiGRU | 10 | 88.84 +/- 2.92 | 87.06 +/- 4.30 | +1.78 |
| NDSS18 | Transformer | 10 | 86.44 +/- 4.02 | 87.06 +/- 4.30 | -0.61 |

All error bars are partition standard deviations. None is a seed standard
deviation.

## 4. Variance decomposition

Juliet SCRUB, BiGRU, max_len 256.

| Protocol | k | Accuracy | Mean within-partition seed SD | Partition SD | Ratio |
| --- | --- | --- | --- | --- | --- |
| CWE-disjoint | 10 | 82.33 +/- 3.06 | 0.802 | 3.056 | 3.81x |
| template-disjoint | 10 | 83.22 +/- 2.59 | 0.422 | 2.593 | 6.14x |

Partition variance dominates seed variance by 3.8x to 6.1x. Any error bar
computed across seeds understates the real uncertainty by that factor.

## 5. Protocol comparison

| Protocol | k | Accuracy |
| --- | --- | --- |
| CWE-disjoint | 10 | 82.33 +/- 3.06 |
| template-disjoint | 10 | 83.22 +/- 2.59 |
| **Difference** | | **-0.89, standard error 1.27** |

The interval spans zero and the sign is reversed relative to the earlier
single-partition comparison. The 4.33-point drop reported before partition
variance was measured came from comparing one arbitrary CWE partition against
one arbitrary template partition, and is retracted.

The standard error is the two-sample form, because the two protocols draw
independent partitions and there is no pairing to exploit.

## 6. Pairwise model differences

Paired on the same partitions, which removes partition difficulty and is why
these standard errors are far smaller than the partition SDs above.

| Corpus | Comparison | Difference | SE | 95% CI | Verdict |
| --- | --- | --- | --- | --- | --- |
| Juliet | LR minus BiGRU | -1.83 | 0.36 | [-2.54, -1.13] | excludes zero |
| Juliet | LR minus Transformer | +2.54 | 0.77 | [+1.02, +4.05] | excludes zero |
| Juliet | BiGRU minus Transformer | +4.37 | 0.84 | [+2.72, +6.01] | excludes zero |
| NDSS18 | LR minus BiGRU | +0.38 | 0.46 | [-0.53, +1.29] | **includes zero** |
| NDSS18 | LR minus Transformer | +2.78 | 0.35 | [+2.09, +3.46] | excludes zero |
| NDSS18 | BiGRU minus Transformer | +2.40 | 0.74 | [+0.94, +3.85] | excludes zero |

On Juliet every pair separates. On NDSS18 the logistic regression and the
BiGRU are indistinguishable.

## 7. Corrected stratification figures

The claim "polarity-stratified assignment reduces worst-case fold class
deviation from 10.9 to 5.8 points" is **wrong as written**. It pairs two
stratified numbers measured on two different conditions and presents the
difference as a stratification effect. 10.9 is stratified on SCRUB; 5.8 is
stratified on RAW.

Correct figures, unstratified against stratified **within one condition**:

| Condition | Unstratified | Polarity-stratified | Improvement |
| --- | --- | --- | --- |
| RAW | 10.1 pts | 5.8 pts | 4.3 |
| SCRUB | 17.6 pts | 10.9 pts | 6.7 |

**Quote the SCRUB row.** Every template-split experiment in this project ran on
SCRUB. RAW and SCRUB differ in deduplication and in class balance, so a figure
from one does not describe the other.

Full validator output for both conditions is in `stratification.txt`.

## 8. Notes on `runs.csv`

189 data rows, one per completed run, 15 columns.

- **`arch` for 112 BiGRU records was inferred, not read.** Those runs predate
  the `--arch` flag, so their `config` has no `arch` key. Absence of the key
  means the recurrent model, and they are filled as `bigru` on that basis. The
  20 transformer records carry `arch` explicitly; the 47 logistic regression
  records are identified by a top-level `model` field instead.
- `model_seed`, `epochs`, and `best_epoch` are empty for all 47 logistic
  regression rows. That model is deterministic and has no epochs.
- `test_templates` is populated for 45 of the 50 template-split rows. The 5
  empty ones are logistic regression runs, whose records carry no
  `test_groups` field. No value was invented for them.
- `lift` is computed as `accuracy - fold_baseline` from the same record.
- `split_seed` is empty for the canonical CWE partition, which the code
  identifies by seed 20260902 rather than by a stored split seed.

## 9. Gaps: what the paper needs that is not on disk

1. **Per-CWE model accuracy across partitions.** Per-CWE support appears in
   `gate_inspect_excerpt.txt` section 3b, and per-CWE collision and accuracy
   tables appear in `ceiling.txt`, but only for the canonical partition
   20260902. The per-run JSON records carry no per-CWE accuracy field, so a
   per-CWE figure with partition error bars is not recoverable without new
   runs.
2. **The stratification figures**, which existed only as terminal output.
   Captured in `stratification.txt` in this archive, which resolves the gap.
3. **The Step 5a gate measurements**, which existed only in `sessions.md`
   prose. Captured in `gate_measurements.txt`, which resolves the gap, except
   that the transformer temperature samples were never written down and are
   not reconstructed.
4. **`step5c_three_way.txt` is unsafe to parse** for the reason in section 2.
   It is excluded from this archive. Table generators should read
   `results/runs/*.json`, or `runs.csv` in this archive, directly.
