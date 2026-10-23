"""TF-IDF logistic regression on one file/condition/length/split cell.

The linear counterpart to bigru.py. Same corpora, same splits, same
deduplication, so the two models land in one table without cross-paper
arithmetic. Writes a per-run JSON next to the BiGRU results.

max_len 0 means the full sequence with no truncation, which is what
ceiling.py used.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from lethe_split import (ALL_CONDITIONS, FILES, SPLIT_MODES, SplitError,
                         build_split, dedup_hashes, load,
                         majority_baseline)

BLAS_THREADS = 6      # measured knee: OpenBLAS scaling turns negative past 6
BAR = "=" * 78


def die(msg, code=2):
    print(f"\nERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def run_id(a):
    ml = "full" if a.max_len == 0 else str(a.max_len)
    bits = ["lr", a.file, a.cond, a.split_mode, f"ml{ml}"]
    if a.split_seed is not None:
        bits.append(f"ss{a.split_seed}")
    if a.shuffle_labels:
        bits.append("shuffled")
    return "-".join(bits)


def truncate(xs, n):
    if n <= 0:
        return xs
    return [" ".join(s.split()[:n]) for s in xs]


def make_parser():
    p = argparse.ArgumentParser(
        prog="lr_run.py",
        description="TF-IDF unigram+bigram logistic regression on one cell.",
        epilog="Example:\n"
               "  python lr_run.py --file juliet --cond SCRUB --max-len 2048",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, choices=FILES)
    p.add_argument("--cond", required=True, choices=ALL_CONDITIONS)
    p.add_argument("--max-len", type=int, default=256,
                   help="truncation length; 0 means the full sequence")
    p.add_argument("--split-mode", default="cwe", choices=SPLIT_MODES)
    p.add_argument("--split-seed", type=int, default=None)
    p.add_argument("--shuffle-labels", action="store_true")
    p.add_argument("--results-dir", default="results/runs")
    p.add_argument("--min-df", type=int, default=5)
    return p


def main(argv=None):
    a = make_parser().parse_args(argv)
    if a.split_mode == "random-cwe" and a.split_seed is None:
        die("--split-mode random-cwe requires --split-seed.")
    if a.split_mode in ("template", "source-sink") and a.file != "juliet":
        die(f"--split-mode template is only validated for juliet, "
            f"not {a.file!r}.")
    if a.max_len < 0:
        die("--max-len cannot be negative; use 0 for the full sequence.")

    rid = run_id(a)
    os.makedirs(a.results_dir, exist_ok=True)
    res_path = os.path.join(a.results_dir, rid + ".json")

    print(BAR)
    print(f"LETHE LR  run {rid}")
    print(BAR)
    try:
        d = load(a.file, a.cond)
        sp = build_split(a.file, d, hashes=dedup_hashes(a.file, a.cond),
                         mode=a.split_mode, split_seed=a.split_seed,
                         shuffle_labels=a.shuffle_labels, shuffle_seed=0)
    except SplitError as e:
        die(str(e))

    xd, labels = sp["x"], sp["labels"]
    tr, te = sp["train_mask"], sp["test_mask"]
    Xtr_txt = truncate([xd[i] for i in np.flatnonzero(tr)], a.max_len)
    Xte_txt = truncate([xd[i] for i in np.flatnonzero(te)], a.max_len)
    ytr, yte = labels[tr], labels[te]
    maj_class, maj = majority_baseline(ytr, yte)

    print(f"  dedup {sp['n_before']:,} -> {sp['n_after']:,}   "
          f"train {len(Xtr_txt):,}  test {len(Xte_txt):,}")
    print(f"  grouping {sp['group_kind']}, {sp['n_groups']:,} groups")
    print(f"  max_len {'full' if a.max_len == 0 else a.max_len}")
    print(f"  majority class {maj_class} -> baseline {maj:.2f}%")
    if len(np.unique(ytr)) < 2:
        die(f"training fold is single-class for {rid}; the split is "
            f"degenerate. Try a different --split-seed.")

    t0 = time.perf_counter()
    vec = TfidfVectorizer(analyzer="word", tokenizer=str.split,
                          token_pattern=None, lowercase=False,
                          ngram_range=(1, 2), min_df=a.min_df,
                          sublinear_tf=True, dtype=np.float32)
    Xtr = vec.fit_transform(Xtr_txt)
    Xte = vec.transform(Xte_txt)
    t_vec = time.perf_counter() - t0
    print(f"  features {Xtr.shape[1]:,}  train nnz {Xtr.nnz:,}  ({t_vec:.1f}s)")

    t0 = time.perf_counter()
    with threadpool_limits(limits=BLAS_THREADS):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    acc = float((clf.predict(Xte) == yte).mean() * 100)
    tracc = float((clf.predict(Xtr) == ytr).mean() * 100)
    print(f"  fit {t_fit:.1f}s, n_iter={int(clf.n_iter_[0])}")
    print(f"\n  RESULT test {acc:.2f}%  baseline {maj:.2f}%  "
          f"lift {acc-maj:+.2f}  train {tracc:.2f}%")

    out = {
        "run_id": rid, "status": "done", "model": "lr",
        "config": {"file": a.file, "cond": a.cond, "max_len": a.max_len,
                   "split_mode": a.split_mode, "split_seed": a.split_seed,
                   "shuffle_labels": bool(a.shuffle_labels),
                   "min_df": a.min_df},
        "meta": {"n_before": sp["n_before"], "n_after": sp["n_after"],
                 "n_train": int(tr.sum()), "n_test": int(te.sum()),
                 "group_kind": sp["group_kind"],
                 "n_groups": int(sp["n_groups"]),
                 "features": int(Xtr.shape[1]),
                 "majority_class": maj_class, "baseline": maj},
        "test_acc": acc, "train_acc": tracc,
        "elapsed": t_vec + t_fit,
    }
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  wrote {res_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
