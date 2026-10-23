"""Shared corpus loading and every split protocol used by the project.

Single source of truth. ceiling.py, lr256.py, and bigru.py all import from
here so the logistic regression and the neural model see identical partitions.

Split modes
-----------
cwe         the canonical CWE-disjoint partition at seed 20260902, asserted
            against the CWE list recorded in results/ceiling.txt
random-cwe  the same construction at an arbitrary seed, for measuring how
            much the headline number depends on which CWEs were held out
template    template-disjoint, see template_split.py
"""

import collections
import gzip
import hashlib
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DERIVED = os.path.join(HERE, "data", "derived")
FILES = ["juliet", "ndss"]
CONDITIONS = ["RAW", "SCRUB", "MASK"]
SPLIT_MODES = ["cwe", "template", "random-cwe", "source-sink"]

# Test-set matching.
#
# Each condition deduplicates by its own sequence hashes, and scrubbing merges
# sequences that differed only in symbol names. Juliet RAW loses 0 percent to
# deduplication while SCRUB loses 31.32 percent, so the plain RAW and SCRUB
# rows are scored on different test items and their difference conflates a
# representation change with a test-set change.
#
# A "-matched" condition loads its own corpus but deduplicates using the
# SCRUB-condition hashes, so RAW-matched, SCRUB, and MASK-matched all train and
# test on exactly the same items and differ only in input representation.
# SCRUB itself defines the mask, so SCRUB-matched would be SCRUB and is not
# offered.
MATCH_SUFFIX = "-matched"
MATCHED_CONDITIONS = ["RAW" + MATCH_SUFFIX, "MASK" + MATCH_SUFFIX]
ALL_CONDITIONS = CONDITIONS + MATCHED_CONDITIONS
DEDUP_REFERENCE = "SCRUB"


def base_cond(cond):
    """Underlying corpus for a condition label."""
    return cond[:-len(MATCH_SUFFIX)] if cond.endswith(MATCH_SUFFIX) else cond


def is_matched(cond):
    return cond.endswith(MATCH_SUFFIX)

SEED = 20260902
TEST_FRACTION = 0.30
MIN_SUPPORT = 200

# Recorded in results/ceiling.txt. Drift here invalidates every comparison,
# so the canonical split is asserted against these on every run.
EXPECTED_TEST_CWES = {
    "juliet": ["CWE121", "CWE123", "CWE127", "CWE190", "CWE23", "CWE401",
               "CWE415", "CWE427", "CWE476", "CWE680", "CWE762"],
    "ndss": ["CWE119", "CWE122", "CWE134"],
}


class SplitError(RuntimeError):
    """Raised when a split cannot be built or drifts from its reference."""


def corpus_path(key, cond):
    return os.path.join(DERIVED, f"{key}-{base_cond(cond).lower()}.pkl.gz")


def load(key, cond):
    """Load a derived corpus, with an actionable message if it is missing."""
    p = corpus_path(key, cond)
    if not os.path.exists(p):
        raise SplitError(
            f"Derived corpus not found: {p}\n"
            f"  Build it first:  python scrub.py\n"
            f"  scrub.py needs the two source .pkl files in data/. See the "
            f"Data section of README.md for the URLs and sha256 values."
        )
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def hash_seqs(xs):
    return [hashlib.sha1(s.encode("utf-8", "surrogatepass")).hexdigest()
            for s in xs]


def dedup_hashes(key, cond):
    """Hashes to deduplicate by. Matched conditions borrow the reference's.

    Returns None when the condition deduplicates by its own sequences, which
    lets build_split fall through to its normal path.
    """
    if not is_matched(cond):
        return None
    ref = load(key, DEDUP_REFERENCE)
    own = load(key, cond)
    if len(ref["x"]) != len(own["x"]):
        raise SplitError(
            f"cannot match {cond!r} to {DEDUP_REFERENCE!r} for {key!r}: "
            f"corpora have {len(own['x']):,} and {len(ref['x']):,} samples. "
            f"They must be aligned row for row. Rebuild both with scrub.py."
        )
    if own["z1"] != ref["z1"] or own["z2"] != ref["z2"]:
        raise SplitError(
            f"cannot match {cond!r} to {DEDUP_REFERENCE!r} for {key!r}: the "
            f"label or CWE columns differ, so the rows are not aligned."
        )
    return hash_seqs(ref["x"])


def qualifying_cwes(z1, z2):
    """CWEs with at least MIN_SUPPORT samples and both classes present."""
    per = collections.defaultdict(collections.Counter)
    for c, l in zip(z2, z1):
        per[c][l] += 1
    out = []
    for c, cc in per.items():
        if sum(cc.values()) >= MIN_SUPPORT and cc.get(0, 0) > 0 and \
                cc.get(1, 0) > 0:
            out.append(c)
    return sorted(out), per


def dedup_indices(hashes):
    """Indices of the first occurrence of each distinct sequence, in order."""
    seen = set()
    keep = []
    for i, h in enumerate(hashes):
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep


def pick_test_cwes(quals, seed=SEED, test_fraction=TEST_FRACTION):
    """The CWE partition. Deterministic given quals and seed."""
    rng = np.random.default_rng(seed)
    pool = list(quals)
    order = rng.permutation(len(pool))
    n_test = max(1, int(round(len(pool) * test_fraction)))
    return sorted(pool[i] for i in order[:n_test])


def _dedup(d, hashes=None):
    x, z1, z2 = d["x"], d["z1"], d["z2"]
    y = d.get("y", [""] * len(x))
    if hashes is None:
        hashes = hash_seqs(x)
    keep = dedup_indices(hashes)
    return {
        "x": [x[i] for i in keep],
        "y": [y[i] for i in keep],
        "labels": np.array([z1[i] for i in keep], dtype=int),
        "cwe": [z2[i] for i in keep],
        "keep": keep,
        "n_before": len(x),
        "n_after": len(keep),
    }


def cwe_disjoint_split(key, d, hashes=None, quals=None, seed=SEED,
                       test_fraction=TEST_FRACTION, assert_expected=True):
    """Deduplicate by sequence hash, then split so no test CWE trains.

    Returns a dict with the deduplicated corpus and boolean train/test masks.
    When assert_expected is true the computed test CWE set must match
    EXPECTED_TEST_CWES[key], which pins the canonical protocol.
    """
    base = _dedup(d, hashes)
    if quals is None:
        quals, _ = qualifying_cwes(d["z1"], d["z2"])
    cd = base["cwe"]

    test_cwes = pick_test_cwes(quals, seed=seed, test_fraction=test_fraction)
    train_cwes = sorted(set(c for c in cd if c not in set(test_cwes)))

    expected = EXPECTED_TEST_CWES.get(key)
    if assert_expected and expected is not None and \
            sorted(test_cwes) != sorted(expected):
        raise SplitError(
            f"CWE-disjoint split drifted for {key!r}.\n"
            f"  expected test CWEs ({len(expected)}): {sorted(expected)}\n"
            f"  computed test CWEs ({len(test_cwes)}): {sorted(test_cwes)}\n"
            f"  seed={seed} test_fraction={test_fraction} "
            f"min_support={MIN_SUPPORT} qualifying={len(quals)}\n"
            f"  This means the corpus or the protocol changed. Do not work "
            f"around it; find out which."
        )
    overlap = set(test_cwes) & set(train_cwes)
    if overlap:
        raise SplitError(f"train/test CWE overlap for {key!r}: "
                         f"{sorted(overlap)}")

    te = np.array([c in set(test_cwes) for c in cd])
    out = dict(base)
    out.update({
        "cwe": cd,
        "desc": base["y"],          # the natural-language description field
        "y": base["labels"],        # alias: ceiling.py and lr256.py read "y"
        "test_cwes": test_cwes, "train_cwes": train_cwes,
        "test_mask": te, "train_mask": ~te,
        "quals": quals, "mode": "cwe", "split_seed": seed,
        "group_kind": "cwe",
        "n_groups": len(set(cd)),
    })
    return out


def build_split(key, d, mode="cwe", split_seed=None, test_fraction=TEST_FRACTION,
                shuffle_labels=False, shuffle_seed=0, hashes=None,
                shuffle_scope="train"):
    """Unified entry point used by bigru.py.

    Returns a dict with keys: x, labels, cwe, y, train_mask, test_mask, and
    metadata describing how the partition was made.
    """
    if mode not in SPLIT_MODES:
        raise SplitError(f"unknown split mode {mode!r}; "
                         f"choose one of {SPLIT_MODES}")

    if mode == "cwe":
        seed = SEED if split_seed is None else split_seed
        sp = cwe_disjoint_split(key, d, hashes=hashes, seed=seed,
                                test_fraction=test_fraction,
                                assert_expected=(seed == SEED))
    elif mode == "random-cwe":
        if split_seed is None:
            raise SplitError("--split-mode random-cwe requires --split-seed")
        sp = cwe_disjoint_split(key, d, hashes=hashes, seed=split_seed,
                                test_fraction=test_fraction,
                                assert_expected=False)
        sp["mode"] = "random-cwe"
    else:
        from template_split import template_disjoint_split
        seed = 0 if split_seed is None else split_seed
        sp = template_disjoint_split(key, d, hashes=hashes, seed=seed,
                                     test_fraction=test_fraction,
                                     key_kind=mode)

    labels = sp["labels"]
    if shuffle_labels:
        # Random-label control. Two scopes, and they answer different
        # questions, so the caller must choose deliberately.
        #
        #   train: permute labels inside the TRAINING fold only; the test fold
        #          keeps its true labels. The model learns noise, so on a
        #          truthfully labelled test set it must land at the fold's own
        #          baseline. Anything above that means the protocol itself
        #          leaks test-label information into training.
        #
        #   all:   permute across the whole deduplicated corpus, test fold
        #          included. Test labels are then random too, so this only
        #          confirms the pipeline cannot beat chance when no signal
        #          exists anywhere. It cannot detect a leaking split.
        if shuffle_scope not in ("train", "all"):
            raise SplitError(f"shuffle_scope must be 'train' or 'all', "
                             f"got {shuffle_scope!r}")
        rng = np.random.default_rng(shuffle_seed)
        y = labels.copy()
        if shuffle_scope == "all":
            y = y[rng.permutation(len(y))]
        else:
            idx = np.flatnonzero(sp["train_mask"])
            y[idx] = y[idx[rng.permutation(len(idx))]]
        sp["labels"] = y
        sp["y"] = y
        sp["labels_shuffled"] = True
        sp["shuffle_scope"] = shuffle_scope
        sp["shuffle_seed"] = shuffle_seed
    else:
        sp["labels_shuffled"] = False
        sp["shuffle_scope"] = None
    return sp


def majority_baseline(ytr, yte):
    """Predict the training majority class on the test set."""
    maj = int(np.bincount(ytr, minlength=2).argmax())
    return maj, float((yte == maj).mean() * 100)
