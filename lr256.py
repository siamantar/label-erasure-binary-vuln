"""Stage 1: TF-IDF logistic regression on the first 256 tokens only.

Closes the 2x2 against ceiling.py's full-sequence LR, isolating truncation
from architecture. Same split, same vectorizer settings, same solver.
"""

import os
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from lethe_split import (CONDITIONS, FILES, SEED, TEST_FRACTION,
                         cwe_disjoint_split, load, majority_baseline)

MAX_LEN = 256
BLAS_THREADS = 6
BAR = "=" * 78


def truncate(xs, n):
    return [" ".join(s.split()[:n]) for s in xs]


def main():
    print(BAR)
    print(f"LETHE Stage 1: LR on the first {MAX_LEN} tokens")
    print(BAR)
    print(f"python {sys.version.split()[0]}  numpy {np.__version__}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    print(f"split: seed={SEED} test_fraction={TEST_FRACTION}, "
          f"asserted against results/ceiling.txt")
    print(f"threadpool_limits={BLAS_THREADS} around the LR solve")

    rows = []
    for key in FILES:
        for cond in CONDITIONS:
            d = load(key, cond)
            sp = cwe_disjoint_split(key, d)
            xd, yd = sp["x"], sp["y"]
            tr, te = sp["train_mask"], sp["test_mask"]

            print("\n" + BAR)
            print(f"### {key.upper()} / {cond} / max_len={MAX_LEN}")
            print(BAR)
            print(f"  dedup      : {sp['n_before']:,} -> {sp['n_after']:,}")
            print(f"  test CWEs  : {', '.join(sp['test_cwes'])}")

            Xtr_txt = truncate([xd[i] for i in np.flatnonzero(tr)], MAX_LEN)
            Xte_txt = truncate([xd[i] for i in np.flatnonzero(te)], MAX_LEN)
            ytr, yte = yd[tr], yd[te]
            print(f"  train      : {len(Xtr_txt):,}  "
                  f"class balance {np.bincount(ytr, minlength=2).tolist()}")
            print(f"  test       : {len(Xte_txt):,}  "
                  f"class balance {np.bincount(yte, minlength=2).tolist()}")

            maj_class, maj = majority_baseline(ytr, yte)
            print(f"  majority class from train = {maj_class}, "
                  f"baseline on test = {maj:.2f}%")

            t0 = time.perf_counter()
            vec = TfidfVectorizer(analyzer="word", tokenizer=str.split,
                                  token_pattern=None, lowercase=False,
                                  ngram_range=(1, 2), min_df=5,
                                  sublinear_tf=True, dtype=np.float32)
            Xtr = vec.fit_transform(Xtr_txt)
            Xte = vec.transform(Xte_txt)
            t_vec = time.perf_counter() - t0
            print(f"  features   : {Xtr.shape[1]:,}  "
                  f"train nnz {Xtr.nnz:,}  ({t_vec:.1f}s)")

            t0 = time.perf_counter()
            with threadpool_limits(limits=BLAS_THREADS):
                clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
                clf.fit(Xtr, ytr)
            t_fit = time.perf_counter() - t0
            acc = float((clf.predict(Xte) == yte).mean() * 100)
            tracc = float((clf.predict(Xtr) == ytr).mean() * 100)
            print(f"  LR fit     : {t_fit:.1f}s, n_iter={clf.n_iter_[0]}")
            print(f"\n  RESULT held-out accuracy : {acc:6.2f}%")
            print(f"         majority baseline  : {maj:6.2f}%")
            print(f"         lift               : {acc-maj:+6.2f} pts")
            print(f"         train accuracy     : {tracc:6.2f}%")
            rows.append((key, cond, acc, maj, Xtr.shape[1], tracc))
            del d, sp, Xtr, Xte, Xtr_txt, Xte_txt

    print("\n" + BAR)
    print(f"SUMMARY  LR on first {MAX_LEN} tokens")
    print(BAR)
    print(f"  {'file':<8} {'cond':<6} {'features':>9} {'acc':>8} "
          f"{'baseline':>9} {'lift':>8} {'train acc':>10}")
    for key, cond, acc, maj, nf, tracc in rows:
        print(f"  {key:<8} {cond:<6} {nf:>9,} {acc:7.2f}% {maj:8.2f}% "
              f"{acc-maj:+7.2f} {tracc:9.2f}%")


if __name__ == "__main__":
    main()
