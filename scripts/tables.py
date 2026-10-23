"""Tables 3 to 6 of the paper and the input-length figures of Section 4.4,
computed from results/runs.csv.

Tables 3 and 4 use the anonymized corpus (condition SCRUB), CWE-disjoint
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
that every model contributes a single run per partition. The Juliet rows of
that output form Table 5.

With --variance, the script prints Table 6 instead: Juliet, anonymized
corpus, BiGRU, 256-token window, for the ten CWE-disjoint sampled partitions
and the ten template-disjoint partitions. Per partition it takes the sample
standard deviation of accuracy across training seeds, the mean accuracy and
the mean baseline. It reports the mean of the seed standard deviations, the
sample standard deviation of the partition means, their ratio, the lowest
and highest partition mean, and the mean and sample standard deviation over
partitions of the margin (mean accuracy minus mean baseline).

With --window, the script prints the figures of Section 4.4 instead: the
logistic regression on the anonymized corpus and the initial CWE-disjoint
partition at 256 tokens, 2,048 tokens and the full sequence, with accuracy,
baseline, margin and test-set size for each corpus, and the change in
accuracy from 256 tokens to the full sequence.

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


def read_rows(path):
    """Every non-control row of the per-run results table."""
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f)
                if not r["run_id"].endswith(CONTROL_SUFFIXES)]


def table6(path):
    """Variance components over partitions, Juliet, anonymized, BiGRU."""
    rows = [r for r in read_rows(path)
            if r["corpus"] == "juliet" and r["condition"] == "SCRUB"
            and r["arch"] == "bigru" and r["max_len"] == "256"]
    print("Table 6. Variance components over ten partitions per protocol, "
          "Juliet, anonymized corpus, BiGRU, 256 tokens.")
    print("Ratio is the standard deviation of partition means divided by the "
          "mean seed standard deviation.")
    print("Margin is the mean and partition standard deviation of accuracy "
          "minus the majority-class baseline.\n")
    print(f"  {'Protocol':<18} {'Partitions':>10} {'Seeds':>6} {'Seed SD':>8} "
          f"{'Partition SD':>13} {'Ratio':>6} {'Accuracy range':>15} "
          f"{'Margin':>15}")
    for mode, label in (("random-cwe", "CWE-disjoint"),
                        ("template", "Template-disjoint")):
        parts = {}
        for r in rows:
            if r["split_mode"] == mode:
                parts.setdefault(int(r["split_seed"]), []).append(
                    (float(r["accuracy"]), float(r["fold_baseline"])))
        if len(parts) != N_PARTITIONS or any(len(v) < 2 for v in parts.values()):
            sys.exit(f"error: expected {N_PARTITIONS} {mode} partitions with "
                     f"at least two seeds each in {path}")
        seed_sd = statistics.mean(statistics.stdev(a for a, _ in v)
                                  for v in parts.values())
        means = [statistics.mean(a for a, _ in v) for v in parts.values()]
        margins = [statistics.mean(a for a, _ in v) -
                   statistics.mean(b for _, b in v) for v in parts.values()]
        part_sd = statistics.stdev(means)
        seeds = ",".join(map(str, sorted({len(v) for v in parts.values()})))
        rng = f"{min(means):.2f}-{max(means):.2f}"
        print(f"  {label:<18} {len(parts):>10} {seeds:>6} {seed_sd:8.3f} "
              f"{part_sd:13.3f} {part_sd / seed_sd:6.2f} {rng:>15} "
              f"{statistics.mean(margins):+8.2f} +/- "
              f"{statistics.stdev(margins):4.2f}")
    print()


def window(path):
    """Logistic regression at three input lengths, initial partition."""
    rows = [r for r in read_rows(path)
            if r["arch"] == "lr" and r["split_mode"] == "cwe"
            and r["condition"] == "SCRUB"]
    lengths = [("256", "256 tokens"), ("2048", "2,048 tokens"),
               ("0", "full sequence")]
    print("Section 4.4. Logistic regression at three input lengths, "
          "anonymized corpus, CWE-disjoint, initial partition.")
    print("Margin is accuracy minus the majority-class baseline of the "
          "test set.\n")
    print(f"  {'Corpus':<8} {'Input':<14} {'Accuracy':>9} {'Baseline':>9} "
          f"{'Margin':>8} {'Test samples':>13}")
    for corpus, clab in CORPORA:
        got = {}
        for ml, mlab in lengths:
            r = [x for x in rows if x["corpus"] == corpus and x["max_len"] == ml]
            if len(r) != 1:
                sys.exit(f"error: expected one logistic regression run for "
                         f"{corpus} at max_len {ml} in {path}, found {len(r)}")
            r = r[0]
            got[ml] = r
            print(f"  {clab:<8} {mlab:<14} {float(r['accuracy']):9.2f} "
                  f"{float(r['fold_baseline']):9.2f} {float(r['lift']):+8.2f} "
                  f"{int(r['test_n']):>13,}")
        same = len({got[ml]["test_n"] for ml, _ in lengths}) == 1
        gain = float(got["0"]["accuracy"]) - float(got["256"]["accuracy"])
        print(f"  {clab:<8} full sequence minus 256 tokens: {gain:+.2f} points; "
              f"same test set at all three lengths: {'yes' if same else 'NO'}")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compute the statistics of Tables 3 to 6 and the "
                    "input-length figures of Section 4.4 from the per-run "
                    "results in results/runs.csv. Without an option the "
                    "script prints Tables 3 and 4.")
    ap.add_argument("--runs", default=DEFAULT_RUNS,
                    help="path to the per-run results table "
                         "(default: results/runs.csv in this repository)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--per-seed", action="store_true",
                      help="also repeat Table 4 once per BiGRU training seed; "
                           "the Juliet rows form Table 5")
    mode.add_argument("--variance", action="store_true",
                      help="print Table 6, the variance components, instead")
    mode.add_argument("--window", action="store_true",
                      help="print the logistic regression at 256 tokens, "
                           "2,048 tokens and the full sequence (Section 4.4) "
                           "instead")
    a = ap.parse_args(argv)

    if a.variance:
        table6(a.runs)
        return 0
    if a.window:
        window(a.runs)
        return 0

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
