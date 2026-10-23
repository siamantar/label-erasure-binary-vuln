"""Accuracy ceiling under RAW, SCRUB, and MASK.

4a  exact sequence collisions and the irreducible error they impose
4b  trivial baselines under a CWE-disjoint split, deduplicated first
"""

import collections
import gzip
import hashlib
import os
import pickle
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from lethe_split import (CONDITIONS, DERIVED, FILES, MIN_SUPPORT, SEED,
                         TEST_FRACTION, cwe_disjoint_split, hash_seqs, load,
                         majority_baseline, qualifying_cwes)

HERE = os.path.dirname(os.path.abspath(__file__))
BLAS_THREADS = 6          # measured: OpenBLAS scaling turns negative past 6
N_JOBS = min(20, os.cpu_count() or 1)
BAR = "=" * 78


def h2(t):
    print("\n" + "-" * 78); print(t); print("-" * 78)


def collision_stats(hashes, labels):
    """Distinct sequences, conflicting sequences, and minimum error count."""
    groups = collections.defaultdict(collections.Counter)
    for h, l in zip(hashes, labels):
        groups[h][l] += 1
    distinct = len(groups)
    conflict = {h: cc for h, cc in groups.items() if len(cc) > 1}
    errors = sum(sum(cc.values()) - max(cc.values()) for cc in conflict.values())
    conflict_samples = sum(sum(cc.values()) for cc in conflict.values())
    return {
        "n": len(hashes), "distinct": distinct, "conflict_seqs": len(conflict),
        "conflict_samples": conflict_samples, "errors": errors,
    }


def cwe_collision(cwe, hashes, labels, z2):
    idx = [i for i, c in enumerate(z2) if c == cwe]
    s = collision_stats([hashes[i] for i in idx], [labels[i] for i in idx])
    s["cwe"] = cwe
    return s


def run_4a(key, cond, d, hashes, quals):
    z1, z2 = d["z1"], d["z2"]
    h2(f"4a. EXACT COLLISIONS   {key.upper()} / {cond}")
    s = collision_stats(hashes, z1)
    n = s["n"]
    print(f"  total samples                       : {n:,}")
    print(f"  distinct sequences                  : {s['distinct']:,}")
    print(f"  duplicate redundancy (n - distinct) : {n - s['distinct']:,} "
          f"({(n-s['distinct'])/n*100:.2f}%)")
    print(f"  sequences with BOTH z1 labels       : {s['conflict_seqs']:,}")
    print(f"  samples in those sequences          : {s['conflict_samples']:,} "
          f"({s['conflict_samples']/n*100:.2f}%)")
    print("\n  Irreducible error arithmetic:")
    print("    errors = sum over conflicting sequences of "
          "(group_size - majority_count)")
    print(f"    errors = {s['errors']:,}")
    print(f"    irreducible error rate = {s['errors']:,} / {n:,} "
          f"= {s['errors']/n*100:.4f}%")
    print(f"    accuracy ceiling       = 100 - {s['errors']/n*100:.4f} "
          f"= {100 - s['errors']/n*100:.4f}%")

    print(f"\n  Per-CWE, for the {len(quals)} CWEs clearing {MIN_SUPPORT} "
          f"samples with both classes (n_jobs={N_JOBS}, 1 thread each):")
    rows = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(cwe_collision)(c, hashes, z1, z2) for c in quals)
    rows.sort(key=lambda r: (-r["n"], r["cwe"]))
    print(f"    {'CWE':<10} {'n':>7} {'distinct':>9} {'conflict seq':>13} "
          f"{'errors':>8} {'irred %':>9} {'ceiling %':>10}")
    tot_e = 0
    for r in rows:
        tot_e += r["errors"]
        print(f"    {r['cwe']:<10} {r['n']:>7,} {r['distinct']:>9,} "
              f"{r['conflict_seqs']:>13,} {r['errors']:>8,} "
              f"{r['errors']/r['n']*100:8.4f}% "
              f"{100-r['errors']/r['n']*100:9.4f}%")
    covered = sum(r["n"] for r in rows)
    print(f"    {'TOTAL':<10} {covered:>7,} {'':>9} {'':>13} {tot_e:>8,} "
          f"{tot_e/covered*100:8.4f}% {100-tot_e/covered*100:9.4f}%")
    return s


def run_4b(key, cond, d, hashes, quals):
    x = d["x"]
    h2(f"4b. CWE-DISJOINT SPLIT   {key.upper()} / {cond}")

    sp = cwe_disjoint_split(key, d, hashes=hashes, quals=quals)
    xd, yd, cd = sp["x"], sp["y"], sp["cwe"]
    test_cwes, train_cwes = sp["test_cwes"], sp["train_cwes"]
    print(f"  before dedup : {sp['n_before']:,} samples")
    print(f"  after  dedup : {sp['n_after']:,} samples "
          f"({(1-sp['n_after']/sp['n_before'])*100:.2f}% removed)")
    print(f"\n  seed = {SEED}, test fraction = {TEST_FRACTION}")
    print(f"  TEST CWEs  ({len(test_cwes)}): {', '.join(test_cwes)}")
    print(f"  TRAIN CWEs ({len(train_cwes)}): {', '.join(train_cwes)}")
    print(f"  disjoint check: overlap = none (asserted in lethe_split)")

    te, tr = sp["test_mask"], sp["train_mask"]
    Xtr_txt = [xd[i] for i in np.flatnonzero(tr)]
    Xte_txt = [xd[i] for i in np.flatnonzero(te)]
    ytr, yte = yd[tr], yd[te]
    print(f"\n  train samples : {len(Xtr_txt):,}   "
          f"class balance {np.bincount(ytr, minlength=2).tolist()}")
    print(f"  test  samples : {len(Xte_txt):,}   "
          f"class balance {np.bincount(yte, minlength=2).tolist()}")
    if len(Xte_txt) == 0 or len(np.unique(ytr)) < 2:
        print("  SKIP: degenerate split")
        return None

    maj_class, maj_test = majority_baseline(ytr, yte)
    maj_test_oracle = float(max(yte.mean(), 1 - yte.mean()) * 100)
    print(f"  majority class from TRAIN = {maj_class}; "
          f"predicting it on test = {maj_test:.2f}%")
    print(f"  (oracle majority on test itself = {maj_test_oracle:.2f}%)")

    t0 = time.perf_counter()
    vec = TfidfVectorizer(analyzer="word", tokenizer=str.split,
                          token_pattern=None, lowercase=False,
                          ngram_range=(1, 2), min_df=5, sublinear_tf=True,
                          dtype=np.float32)
    Xtr = vec.fit_transform(Xtr_txt)
    Xte = vec.transform(Xte_txt)
    t_vec = time.perf_counter() - t0
    print(f"\n  TF-IDF unigram+bigram, min_df=5, sublinear_tf")
    print(f"    features : {Xtr.shape[1]:,}")
    print(f"    train nnz: {Xtr.nnz:,}   test nnz: {Xte.nnz:,}")
    print(f"    fit time : {t_vec:.1f}s")

    t0 = time.perf_counter()
    with threadpool_limits(limits=BLAS_THREADS):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    acc = float((clf.predict(Xte) == yte).mean() * 100)
    tracc = float((clf.predict(Xtr) == ytr).mean() * 100)
    print(f"    LR fit   : {t_fit:.1f}s "
          f"(threadpool_limits={BLAS_THREADS}), n_iter={clf.n_iter_[0]}")
    print(f"\n  RESULT  held-out CWE accuracy : {acc:6.2f}%")
    print(f"          majority baseline     : {maj_test:6.2f}%")
    print(f"          lift over baseline    : {acc - maj_test:+6.2f} pts")
    print(f"          train accuracy        : {tracc:6.2f}%")

    print(f"\n  Per test CWE (n_jobs={N_JOBS}, 1 thread each):")
    pred = clf.predict(Xte)
    cte = [cd[i] for i in np.flatnonzero(te)]

    def per_cwe(c):
        m = np.array([cc == c for cc in cte])
        if m.sum() == 0:
            return None
        a = float((pred[m] == yte[m]).mean() * 100)
        b = float(max(yte[m].mean(), 1 - yte[m].mean()) * 100)
        return (c, int(m.sum()), a, b)

    rows = [r for r in Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(per_cwe)(c) for c in test_cwes) if r]
    rows.sort(key=lambda r: -r[1])
    print(f"    {'CWE':<10} {'n':>7} {'accuracy':>9} {'majority':>9} "
          f"{'lift':>8}")
    for c, k, a, b in rows:
        print(f"    {c:<10} {k:>7,} {a:8.2f}% {b:8.2f}% {a-b:+7.2f}")
    return {"acc": acc, "maj": maj_test, "n_test": len(Xte_txt),
            "features": Xtr.shape[1]}


def main():
    print(BAR)
    print("LETHE: accuracy ceiling under RAW, SCRUB, MASK")
    print(BAR)
    print(f"python {sys.version.split()[0]}  numpy {np.__version__}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    print(f"joblib n_jobs={N_JOBS} (workers single-threaded), "
          f"threadpool_limits={BLAS_THREADS} around the LR solve")
    print(f"seed={SEED}  min CWE support={MIN_SUPPORT}")

    summary = {}
    for key in FILES:
        base = load(key, "RAW")
        quals, _ = qualifying_cwes(base["z1"], base["z2"])
        print(f"\n{BAR}")
        print(f"{key.upper()}: {len(quals)} CWEs clear {MIN_SUPPORT} samples "
              f"with both classes")
        print(BAR)
        del base
        for cond in CONDITIONS:
            d = load(key, cond)
            hashes = hash_seqs(d["x"])
            print(f"\n{BAR}\n### {key.upper()} / {cond}\n{BAR}")
            a = run_4a(key, cond, d, hashes, quals)
            b = run_4b(key, cond, d, hashes, quals)
            summary[(key, cond)] = (a, b)
            del d, hashes

    print("\n" + BAR)
    print("SUMMARY")
    print(BAR)
    print(f"  {'file':<8} {'cond':<6} {'distinct':>10} {'conflict':>9} "
          f"{'irred %':>9} {'held-out acc':>13} {'baseline':>9} {'lift':>8}")
    for (key, cond), (a, b) in summary.items():
        if b is None:
            print(f"  {key:<8} {cond:<6} {a['distinct']:>10,} "
                  f"{a['conflict_seqs']:>9,} {a['errors']/a['n']*100:8.4f}% "
                  f"{'skipped':>13}")
            continue
        print(f"  {key:<8} {cond:<6} {a['distinct']:>10,} "
              f"{a['conflict_seqs']:>9,} {a['errors']/a['n']*100:8.4f}% "
              f"{b['acc']:12.2f}% {b['maj']:8.2f}% {b['acc']-b['maj']:+7.2f}")


if __name__ == "__main__":
    main()
