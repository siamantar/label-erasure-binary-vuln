"""Table 3 and Table 4 of the paper, computed from results/runs.csv.

Both tables use the anonymized corpus (condition SCRUB), CWE-disjoint
evaluation over the ten sampled partitions (split_mode random-cwe,
split_seed 1 to 10), and the 256-token window (max_len 256). Control runs,
whose run IDs end in -shuffled, -shuftrain, or -struct, are excluded.

Table 3 (model classes over ten partitions): for each corpus and model, the
mean and sample standard deviation across partitions of accuracy and of the
majority-class baseline of each test set, and the margin, which is mean
accuracy minus mean baseline. Where a partition has several BiGRU training
seeds, the seeds are averaged within the partition first.

Table 4 (pairwise differences, paired on partition): LR minus BiGRU, LR
minus transformer, and BiGRU minus transformer, with the BiGRU averaged over
its seeds. The standard error is the sample standard deviation of the paired
differences divided by the square root of the number of partitions, and the
95% interval is the mean plus or minus 1.96 standard errors.

With --per-seed, Table 4 is repeated once for each BiGRU training seed, so
that every model contributes a single run per partition.

Reads only the CSV. Needs no corpus, no GPU, and no third-party package.
"""

import argparse
import csv
import math
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNS = os.path.join(REPO, "results", "runs.csv")

CONTROL_SUFFIXES = ("-shuffled", "-shuftrain", "-struct")
CORPORA = [("juliet", "Juliet"), ("ndss", "NDSS18")]
MODELS = [("lr", "Logistic regression"), ("bigru", "BiGRU"),
          ("transformer", "Transformer")]
PAIRS = [("lr", "bigru", "LR - BiGRU"),
         ("lr", "transformer", "LR - Transformer"),
         ("bigru", "transformer", "BiGRU - Transformer")]
N_PARTITIONS = 10


def load(path):
    """Per-partition records for the configuration both tables use.

    Returns {(corpus, arch): {split_seed: [(model_seed, accuracy, baseline)]}}.
    """
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["run_id"].endswith(CONTROL_SUFFIXES):
                continue
            if (r["split_mode"] != "random-cwe" or r["condition"] != "SCRUB"
                    or r["max_len"] != "256"):
                continue
            key = (r["corpus"], r["arch"])
            part = int(r["split_seed"])
            seed = int(r["model_seed"]) if r["model_seed"] else None
            out.setdefault(key, {}).setdefault(part, []).append(
                (seed, float(r["accuracy"]), float(r["fold_baseline"])))
    for corpus, _ in CORPORA:
        for arch, _ in MODELS:
            parts = out.get((corpus, arch), {})
            if len(parts) != N_PARTITIONS:
                sys.exit(f"error: expected {N_PARTITIONS} partitions for "
                         f"{corpus}/{arch}, found {len(parts)} in {path}")
    return out


def partition_means(runs):
    """Average accuracy and baseline over the runs in each partition."""
    acc = {p: statistics.mean(a for _, a, _ in rs) for p, rs in runs.items()}
    base = {p: statistics.mean(b for _, _, b in rs) for p, rs in runs.items()}
    return acc, base


def paired(a, b):
    """Mean, standard error, and 95% interval of a - b over shared partitions."""
    parts = sorted(set(a) & set(b))
    d = [a[p] - b[p] for p in parts]
    m = statistics.mean(d)
    se = statistics.stdev(d) / math.sqrt(len(d))
    return m, se, m - 1.96 * se, m + 1.96 * se


def table3(data):
    print("Table 3. Model classes over ten partitions, anonymized corpus, "
          "CWE-disjoint, 256 tokens.")
    print("Intervals are partition standard deviations. Margin is mean "
          "accuracy minus mean baseline.\n")
    print(f"  {'Corpus':<8} {'Model':<20} {'Runs/partition':>14} "
          f"{'Accuracy':>16} {'Baseline':>16} {'Margin':>8}")
    for corpus, clab in CORPORA:
        for arch, mlab in MODELS:
            runs = data[(corpus, arch)]
            acc, base = partition_means(runs)
            counts = sorted({len(rs) for rs in runs.values()})
            am, asd = statistics.mean(acc.values()), statistics.stdev(acc.values())
            bm, bsd = statistics.mean(base.values()), statistics.stdev(base.values())
            print(f"  {clab:<8} {mlab:<20} {','.join(map(str, counts)):>14} "
                  f"{am:7.2f} +/- {asd:4.2f} {bm:7.2f} +/- {bsd:4.2f} "
                  f"{am - bm:+8.2f}")
    print()


def table4(data, title, pick):
    """Paired differences; pick(runs) gives per-partition accuracy per model."""
    print(title)
    print(f"  {'Corpus':<8} {'Comparison':<21} {'Diff.':>7} {'SE':>6} "
          f"{'95% CI':>18}  Zero")
    for corpus, clab in CORPORA:
        acc = {arch: pick(arch, data[(corpus, arch)]) for arch, _ in MODELS}
        if any(v is None for v in acc.values()):
            continue
        for x, y, lab in PAIRS:
            m, se, lo, hi = paired(acc[x], acc[y])
            zero = "excluded" if lo > 0 or hi < 0 else "included"
            print(f"  {clab:<8} {lab:<21} {m:+7.2f} {se:6.2f} "
                  f"[{lo:+6.2f}, {hi:+6.2f}]  {zero}")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compute the statistics of Table 3 and Table 4 from the "
                    "per-run results in results/runs.csv.")
    ap.add_argument("--runs", default=DEFAULT_RUNS,
                    help="path to the per-run results table "
                         "(default: results/runs.csv in this repository)")
    ap.add_argument("--per-seed", action="store_true",
                    help="also repeat Table 4 once per BiGRU training seed")
    a = ap.parse_args(argv)

    data = load(a.runs)
    table3(data)
    table4(data, "Table 4. Pairwise differences, paired on partition, "
                 "BiGRU averaged over its seeds.",
           lambda arch, runs: partition_means(runs)[0])

    if a.per_seed:
        seeds = sorted({s for (c, arch), runs in data.items() if arch == "bigru"
                        for rs in runs.values() for s, _, _ in rs})
        for s in seeds:
            def pick(arch, runs, s=s):
                if arch != "bigru":
                    return partition_means(runs)[0]
                got = {p: [acc for seed, acc, _ in rs if seed == s]
                       for p, rs in runs.items()}
                if any(len(v) != 1 for v in got.values()):
                    return None
                return {p: v[0] for p, v in got.items()}
            table4(data, f"Table 4 with the BiGRU restricted to training "
                         f"seed {s} (corpora without that seed on every "
                         f"partition are skipped).", pick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
