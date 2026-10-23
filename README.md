# Label Erasure and the Limits of Binary Vulnerability Benchmarks

This repository holds the code and the per-run results for the paper
*Label Erasure and the Limits of Binary Vulnerability Benchmarks* by Siam
Shibly Antar and Syem Shibly Ador, to appear in the Proceedings of the 2026
International Conference on Cybertechnology and Future Networks (ICCFN).

## Summary

Researchers evaluate binary vulnerability detection almost entirely on
synthetic corpora, because real vulnerable binaries with reliable labels are
scarce.
Published systems report accuracies above 99% on the largest of these
corpora, and researchers use those figures to rank methods. The paper audits
the Juliet Test Suite as researchers use it for this task. A substring rule that
performs no learning outscores every published result, because Juliet's
convention of naming vulnerable functions `bad` survives compilation as
mangled symbols. On identical test samples, removing that label costs a
recurrent model less than its own seed variation, so the accuracy that
remains does not come from leakage. Two further explanations for the
corpus's difficulty also fail, and accuracy stays roughly 27 points above
the majority-class baseline under every control the paper applies. Partition
variance exceeds seed variance by a factor of 3.8 to 6.1 while the field
reports seed error bars, so readers should not trust small differences on
this benchmark.

## What this repository contains

| Path | Contents |
| --- | --- |
| `scrub.py` | Builds the three input conditions from the released corpora |
| `lethe_split.py` | Corpus loading, deduplication, the CWE-disjoint partitions, identical test samples, and the label-permutation control |
| `template_split.py` | The template-disjoint partition and its validation report |
| `bigru.py` | Trains the BiGRU and the transformer encoder (`--arch transformer`, class `TransformerEnc`) |
| `lr_run.py` | Trains the logistic regression |
| `ceiling.py`, `lr256.py` | Duplicate collisions and the logistic regression on the initial partition |
| `leak_position.py`, `gate_inspect.py` | Where the label sits in each sequence, and the structure of both corpora |
| `figures/mkfigs.py` | Renders the two figures of the paper from `results/runs.csv` |
| `scripts/tables.py` | Computes the statistics of Table 3 and Table 4 from `results/runs.csv` |
| `scripts/fetch_data.sh` | Downloads and verifies the two corpora |
| `results/runs/` | The 189 per-run results, one JSON file per run |
| `results/runs.csv` | The same 189 runs flattened into one table |
| `results/*.txt`, `results/*.md` | Text records of the analyses, described below |
| `results/MANIFEST.sha256` | SHA-256 digests of every file under `results/` |

The scripts print the working name LETHE in their banners. It refers to this
project.

The text records in `results/`:

| File | Produced by |
| --- | --- |
| `RECONCILED.md` | How the per-run results reconcile into the reported figures, including the seed-averaging convention. It refers to some working files that this repository does not include. |
| `ceiling.txt` | `python ceiling.py` |
| `lr256.txt` | `python lr256.py` |
| `leak_position.txt` | `python leak_position.py` |
| `gate_inspect_excerpt.txt` | `python gate_inspect.py`, sections 3b, 3d, 3e, 3f and 3h. The full output also prints complete corpus samples, which this repository does not redistribute. |
| `stratification.txt` | `python template_split.py --validate --cond RAW` and `--cond SCRUB` |
| `residual_probe.txt` | The non-lexical control. The analysis command that wrote it is not part of this repository. |
| `step3a_source_sink_validation.md` | The source-and-sink partition check. Its analysis command is not part of this repository. |
| `step4_variance.txt` | The variance decomposition and label-permutation control summaries, computed from the per-run results. Its analysis command is not part of this repository. |
| `step5d_retrofit.txt` | Where each initial-partition figure sits in the distribution over ten partitions, computed from the per-run results. Its analysis command is not part of this repository. |
| `gate_measurements.txt` | GPU and memory measurements transcribed from the run notes |

`scrub.py` prints its own record of the three conditions (vocabulary size,
mean sequence length, replaced tokens, and the substring rules) each time it
runs. `results/scrub.txt` is the record of the condition build used in the
paper, with its absolute paths made relative to the repository root and no
other change.

## Setup

We produced the results with Python 3.12.14.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

The two thread variables keep the parallel workers in `ceiling.py` from
oversubscribing the CPU. Rendering the figures and computing the tables
needs only Python and, for the figures, matplotlib.

## Data

The experiments use the labelled Juliet and NDSS18 token corpora from release
`ds-v0.0.1` of
[L1NNA/Juliet-NDSS18-BinaryCodeSum](https://github.com/L1NNA/Juliet-NDSS18-BinaryCodeSum/releases/tag/ds-v0.0.1).
This repository does not redistribute them. Download and verify them with:

```bash
bash scripts/fetch_data.sh
```

The script writes both files to `data/` and checks these digests, which match
the digests shown on the release page:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `mne-opr-desp-juliet-labelled.pkl` | 85,882,805 | `8a13310e7ecac1fe118997b45e7a737dad7b33528b76b8e6c5b8051e847920e3` |
| `mne-opr-desp-ndss-labelled.pkl` | 1,109,847 | `691d6d890411dfe8635bb7b69b0600219c7d2c8780236abcb763a0a899982c29` |

Each file is a gzipped pickle holding a dict of four parallel lists: `x`, the
token sequence as one space-separated string; `y`, a natural-language
description; `z1`, the binary label; and `z2`, the CWE category. Juliet holds
83,326 samples and NDSS18 holds 14,166. If you use the data, also cite the
works that the source release lists.

## The three input conditions

The code names the conditions of the paper as follows:

| Paper | Code |
| --- | --- |
| original | `RAW` |
| anonymized | `SCRUB` |
| fully anonymized | `MASK` |

The anonymized condition replaces mangled C++ symbols, any token whose name
contains `bad`, `good` or a CWE identifier, and the names of Juliet's
test-case support functions with the placeholder token `SYM`. The fully
anonymized condition also replaces recognized C library function names with a
second placeholder token, `LIBFN`. Replacement is one-for-one, so sequence
lengths do not change.

Regenerate all three conditions for both corpora with:

```bash
python scrub.py
```

It reads `data/*.pkl` and writes six files to `data/derived/`, named
`{juliet,ndss}-{raw,scrub,mask}.pkl.gz`. Every training script loads these.

The conditions `RAW-matched` and `MASK-matched` give the identical test
samples of the paper. They load the original or fully anonymized corpus but
deduplicate it once with the hashes of the anonymized corpus, so all three
conditions train and test on the same samples.

To print the template-disjoint partition's validation report:

```bash
python template_split.py --validate --cond SCRUB
```

## How the models were trained

You do not need to train anything to reproduce the tables and figures:
`results/` holds the runs used in the paper. The commands below produced
those runs. Each command writes `results/runs/<run_id>.json`, so a rerun with
the same arguments overwrites the stored result. Pass `--results-dir` with
another directory to keep the stored results intact. The neural runs used one
GPU each, with `--gpu 0 --checkpoint-dir checkpoints` added to every
`bigru.py` command.

```bash
# BiGRU, initial partition, all five conditions, three seeds
for f in juliet ndss; do
  for c in RAW SCRUB MASK RAW-matched MASK-matched; do
    for s in 0 1 2; do
      python bigru.py --file $f --cond $c --max-len 256 --seed $s --split-mode cwe
    done
  done
done

# BiGRU, template-disjoint, partition 0, Juliet, all five conditions
for c in RAW SCRUB MASK RAW-matched MASK-matched; do
  for s in 0 1 2; do
    python bigru.py --file juliet --cond $c --max-len 256 --seed $s \
        --split-mode template --split-seed 0
  done
done

# BiGRU, template-disjoint, partitions 1 to 9, Juliet, anonymized
for p in 1 2 3 4 5 6 7 8 9; do
  for s in 0 1 2; do
    python bigru.py --file juliet --cond SCRUB --max-len 256 --seed $s \
        --split-mode template --split-seed $p
  done
done

# BiGRU, the ten sampled CWE-disjoint partitions, anonymized
# (three seeds on Juliet, one on NDSS18)
for p in 1 2 3 4 5 6 7 8 9 10; do
  for s in 0 1 2; do
    python bigru.py --file juliet --cond SCRUB --max-len 256 --seed $s \
        --split-mode random-cwe --split-seed $p
  done
  python bigru.py --file ndss --cond SCRUB --max-len 256 --seed 0 \
      --split-mode random-cwe --split-seed $p
done

# Transformer, the ten sampled CWE-disjoint partitions, anonymized
for f in juliet ndss; do
  for p in 1 2 3 4 5 6 7 8 9 10; do
    python bigru.py --arch transformer --file $f --cond SCRUB --max-len 256 \
        --seed 0 --split-mode random-cwe --split-seed $p
  done
done

# Label-permutation control: labels permuted within the training set only
for s in 0 1 2; do
  python bigru.py --file juliet --cond SCRUB --max-len 256 --seed $s \
      --split-mode cwe --shuffle-labels --shuffle-scope train
  python bigru.py --file ndss --cond SCRUB --max-len 256 --seed $s \
      --split-mode cwe --shuffle-labels --shuffle-scope train
  python bigru.py --file juliet --cond SCRUB --max-len 256 --seed $s \
      --split-mode template --split-seed 0 --shuffle-labels --shuffle-scope train
done

# Whole-corpus permutation, one run (not the paper's control)
python bigru.py --file juliet --cond SCRUB --max-len 256 --seed 0 \
    --split-mode cwe --shuffle-labels --shuffle-scope all

# Logistic regression, initial partition, full sequence (0), 256 and 2048 tokens
for f in juliet ndss; do
  for c in RAW SCRUB MASK; do
    for n in 0 256 2048; do
      python lr_run.py --file $f --cond $c --max-len $n --split-mode cwe
    done
  done
  for c in RAW-matched MASK-matched; do
    python lr_run.py --file $f --cond $c --max-len 256 --split-mode cwe
  done
done

# Logistic regression, template-disjoint, partition 0, Juliet
for c in RAW SCRUB MASK RAW-matched MASK-matched; do
  python lr_run.py --file juliet --cond $c --max-len 256 \
      --split-mode template --split-seed 0
done

# Logistic regression, the ten sampled CWE-disjoint partitions, anonymized
for f in juliet ndss; do
  for p in 1 2 3 4 5 6 7 8 9 10; do
    python lr_run.py --file $f --cond SCRUB --max-len 256 \
        --split-mode random-cwe --split-seed $p
  done
done
```

The corpus analyses print their reports to standard output:

```bash
python gate_inspect.py
python leak_position.py
python ceiling.py
python lr256.py
```

Run `python bigru.py --help`, `python lr_run.py --help` or
`python template_split.py --help` for every option.

## Reproducing the tables and figures from results/

Check the stored results first:

```bash
cd results && sha256sum -c MANIFEST.sha256 && cd ..
```

Table 3 and Table 4:

```bash
python scripts/tables.py
python scripts/tables.py --per-seed   # Table 4 again for each BiGRU seed
```

The script reads only `results/runs.csv`. It keeps the anonymized corpus,
the ten sampled CWE-disjoint partitions and the 256-token window, and it
drops the control runs. Table 3 gives, per corpus and model, the mean and
standard deviation across partitions of accuracy and of the majority-class
baseline of each test set, and the margin, which is mean accuracy minus mean
baseline. It averages the BiGRU seeds within each partition first. Table 4
gives the differences between models paired on partition, with the standard
error of the paired difference and a 95% interval of 1.96 standard errors.

The two figures:

```bash
cd figures
python mkfigs.py            # reads ../results/runs.csv
```

This writes `fig_partitions.pdf` and `fig_variance.pdf` to `figures/`.

## Terms in the code and in the paper

The code, the run IDs and the columns of `results/runs.csv` keep the names
used during the experiments, so the stored results load unchanged. This
table maps them to the terms of the paper.

| Code or results | Where it appears | Paper term |
| --- | --- | --- |
| `RAW` | `--cond`, `condition` column, run IDs, `data/derived/*-raw.pkl.gz` | original |
| `SCRUB` | same places, `*-scrub.pkl.gz` | anonymized |
| `MASK` | same places, `*-mask.pkl.gz` | fully anonymized |
| `RAW-matched`, `MASK-matched` | `--cond`, `condition` column, run IDs | original and fully anonymized on identical test samples, deduplicated once on the anonymized corpus |
| `SYM`, `LIBFN` | `scrub.py` | the placeholder token and the second placeholder token |
| `HARNESS`, harness | `scrub.py` | test-case support functions |
| `lift` | `results/runs.csv` column, script output | margin over the majority-class baseline of each test set |
| `fold_baseline`, `meta.baseline` | `results/runs.csv` column, run JSON | majority-class baseline of each test set |
| fold, `train_mask`, `test_mask` | code comments and output | training set, test set |
| `cwe` split mode, seed 20260902, canonical | `--split-mode cwe`, empty `split_seed` | CWE-disjoint, initial partition |
| `random-cwe`, split seeds 1 to 10 | `split_mode` column, run IDs `-ss1` to `-ss10` | CWE-disjoint, the ten sampled partitions |
| `template`, split seeds 0 to 9 | `split_mode` column, run IDs | template-disjoint |
| `template_key`, last `and` segment | `template_split.py` | template identifier, the final clause of the description |
| `pure0`, `pure1`, `mixed`, polarity, both-class | `template_split.py`, records | templates by label; templates with both vulnerable and non-vulnerable samples |
| `source-sink` | `--split-mode`, `--key`, `step3a_source_sink_validation.md` | source-and-sink partition |
| `--shuffle-labels --shuffle-scope train`, `-shuftrain`, random-label control | `bigru.py`, run IDs, records | label-permutation control |
| `--shuffle-scope all`, `-shuffled` | `bigru.py`, one run ID | whole-corpus permutation, not the paper's control |
| structural, `-struct` | `residual_probe.txt`, `figures/mkfigs.py` | non-lexical control |
| `bigru`, `transformer`, `lr` | `arch` column, `--arch`, run ID prefixes `tfm-` and `lr-` | BiGRU, transformer, logistic regression |
| `juliet`, `ndss` | `--file`, `corpus` column | Juliet, NDSS18 |
| `max_len`, `--max-len`, `ml256` | flags, column, run IDs | the 256-token window |
| cell | `--help` text | configuration |
| token stream | comments and records | token sequence |
| run records | `results/runs/*.json` | per-run results |

## Citation

```bibtex
@inproceedings{antar2026labelerasure,
  title     = {Label Erasure and the Limits of Binary Vulnerability Benchmarks},
  author    = {Antar, Siam Shibly and Ador, Syem Shibly},
  booktitle = {Proceedings of the 2026 International Conference on Cybertechnology and Future Networks (ICCFN)},
  publisher = {Springer},
  year      = {2026},
  note      = {to appear}
}
```

## License

The code and the results in this repository are released under the Apache
License 2.0. See `LICENSE` and `NOTICE`.

The Juliet and NDSS18 data are not part of this repository and remain under
the terms of their source release,
[L1NNA/Juliet-NDSS18-BinaryCodeSum](https://github.com/L1NNA/Juliet-NDSS18-BinaryCodeSum).
That release does not state a license; consult it and its authors before
reusing the data.
