"""Template-disjoint splitting for Juliet.

Juliet generates each testcase from a flow variant: a source paired with a
sink. The same variant is reused across many CWEs, so a CWE-disjoint split
still lets a model meet the same code template in training and test. This
module derives the template from the description field y and builds a split
where no template appears in both folds.

Why the naive key fails
-----------------------
Keying on the whole description gives 806 templates on Juliet, and every one
of them is class-pure: good and bad variants of the same flow carry different
descriptions, so the key separates them by construction.

The key used here
-----------------
A description reads "<source> and <sink>". For the source-varying testcases
that dominate Juliet, the sink names the flow variant and is shared between
the good and bad members of a pair, for example

    initialize data as a small string and copy data to string using strcpy  (0)
    initialize data as a large string and copy data to string using strcpy  (1)

so the last "and"-separated segment is the template. That key gives 339
templates, 66 of which carry both classes, covering 56.9 percent of samples.

The remaining templates are still class-pure, because in sink-varying
testcases the polarity lives in the sink instead. Assigning templates to folds
uniformly at random therefore leaves fold class balance to chance: measured
over 12 seeds, the test fold ranged from 40.2 to 60.1 percent positive.

The fix is to stratify the assignment by template polarity. Templates are
bucketed into pure-negative, pure-positive, and mixed, and each bucket
contributes the same sample fraction to the test fold. That holds the test
fold to 46.7 to 53.8 percent positive over the same 12 seeds, worst deviation
3.8 points instead of 10.1, while keeping folds strictly template-disjoint.

Run this file directly to reproduce those validation numbers.
"""

import argparse
import collections
import re
import sys

import numpy as np

SEG_RE = re.compile(r"\s+and\s+")
STRATA = ("pure0", "pure1", "mixed")


def segments(desc):
    return [p.strip() for p in SEG_RE.split(desc.strip()) if p.strip()]


def template_key(desc):
    """The flow variant: the last 'and'-separated segment of the description."""
    segs = segments(desc)
    return segs[-1] if segs else desc.strip()


def source_sink_key(desc):
    """Compound key over BOTH the source and the sink.

    Juliet's structure is CWE x source x sink x flow variant. Keying on the
    sink alone leaves source patterns shared across folds, so a sink-disjoint
    split can still let a model meet familiar source scaffolding in test. This
    key joins the first segment (source) and the last (sink), so folds are
    disjoint on the pair.

    A description with a single segment has no source/sink distinction to
    make, so the segment itself is the key.
    """
    segs = segments(desc)
    if not segs:
        return desc.strip()
    if len(segs) == 1:
        return segs[0]
    return f"{segs[0]} || {segs[-1]}"


KEY_FUNCS = {"template": template_key, "source-sink": source_sink_key}


def template_keys(descs, kind="template"):
    fn = KEY_FUNCS[kind]
    return [fn(s) for s in descs]


def template_table(keys, labels):
    """template -> [n_negative, n_positive]."""
    tbl = collections.defaultdict(lambda: [0, 0])
    for k, l in zip(keys, labels):
        tbl[k][int(l)] += 1
    return dict(tbl)


def stratum_of(counts):
    neg, pos = counts
    if pos == 0:
        return "pure0"
    if neg == 0:
        return "pure1"
    return "mixed"


def assign_test_templates(tbl, seed, test_fraction, stratified=True):
    """Choose the templates that form the test fold.

    Stratified assignment draws the same sample fraction from each polarity
    bucket, which is what keeps both folds class-balanced.
    """
    names = sorted(tbl)
    if not stratified:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(names))
        n_test = max(1, int(round(len(names) * test_fraction)))
        return set(names[i] for i in order[:n_test])

    buckets = collections.defaultdict(list)
    for k in names:
        buckets[stratum_of(tbl[k])].append(k)

    test = set()
    for si, s in enumerate(STRATA):
        ks = sorted(buckets[s])
        if not ks:
            continue
        # A distinct, deterministic stream per bucket, derived from the seed.
        rng = np.random.default_rng([seed, si])
        rng.shuffle(ks)
        quota = test_fraction * sum(sum(tbl[k]) for k in ks)
        got = 0
        for k in ks:
            if got >= quota:
                break
            test.add(k)
            got += sum(tbl[k])
    return test


def template_disjoint_split(key, d, hashes=None, seed=0, test_fraction=0.30,
                            stratified=True, key_kind="template"):
    """Deduplicate, derive templates from y, then split template-disjoint."""
    from lethe_split import SplitError, _dedup

    base = _dedup(d, hashes)
    descs, labels, cwes = base["y"], base["labels"], base["cwe"]
    if not any(s.strip() for s in descs):
        raise SplitError(
            f"template split needs the description field y, but {key!r} has "
            f"none populated. Only Juliet and NDSS carry descriptions; check "
            f"that the derived corpus was built by scrub.py, which preserves "
            f"y unchanged."
        )

    keys = template_keys(descs, key_kind)
    tbl = template_table(keys, labels)
    if len(tbl) < 4:
        raise SplitError(
            f"template split produced only {len(tbl)} templates for {key!r}, "
            f"too few to partition. The description field is probably not in "
            f"the expected '<source> and <sink>' form."
        )
    test_tpl = assign_test_templates(tbl, seed, test_fraction, stratified)
    te = np.array([k in test_tpl for k in keys])

    if te.all() or not te.any():
        raise SplitError(
            f"template split degenerate for {key!r}: test fold holds "
            f"{int(te.sum())} of {len(te)} samples."
        )
    tr = ~te
    for nm, m in (("train", tr), ("test", te)):
        if len(np.unique(labels[m])) < 2:
            raise SplitError(
                f"template split left the {nm} fold single-class for {key!r} "
                f"at seed {seed}. Try a different --split-seed, and report it: "
                f"stratified assignment is supposed to prevent this."
            )

    out = dict(base)
    out.update({
        "cwe": cwes, "desc": descs, "y": labels,
        "test_mask": te, "train_mask": tr,
        "mode": key_kind, "split_seed": seed,
        "group_kind": key_kind,
        "n_groups": len(tbl),
        "templates": tbl,
        "test_templates": sorted(test_tpl),
        "test_cwes": sorted(set(cwes)),      # templates cut across CWEs
        "train_cwes": sorted(set(cwes)),
    })
    return out


def validate(corpus_key="juliet", cond="RAW", seeds=12, test_fraction=0.30,
             key_kind="template"):
    """Reproduce the validation numbers quoted in this module's docstring."""
    from lethe_split import _dedup, load

    d = load(corpus_key, cond)
    base = _dedup(d)
    descs, labels = base["y"], base["labels"]
    n = len(labels)
    print("=" * 78)
    print(f"SPLIT VALIDATION  {corpus_key}/{cond}  key={key_kind}")
    print("=" * 78)
    print(f"  deduplicated samples : {n:,}")
    print(f"  overall class balance: {labels.mean()*100:.2f}% positive")

    naive = collections.defaultdict(lambda: [0, 0])
    for s, l in zip(descs, labels):
        naive[s][int(l)] += 1
    pure = sum(1 for v in naive.values() if v[0] == 0 or v[1] == 0)
    print(f"\n  NAIVE KEY (whole description)")
    print(f"    templates            : {len(naive):,}")
    print(f"    class-pure templates : {pure:,} "
          f"({pure/len(naive)*100:.1f}%)  <-- this is why it fails")

    keys = template_keys(descs, key_kind)
    tbl = template_table(keys, labels)
    sizes = sorted((sum(v) for v in tbl.values()), reverse=True)
    both = [k for k in tbl if tbl[k][0] and tbl[k][1]]
    both_n = sum(sum(tbl[k]) for k in both)
    label = ("last 'and' segment (sink)" if key_kind == "template"
             else "first + last 'and' segments (source + sink)")
    print(f"\n  KEY: {label}")
    print(f"    distinct keys        : {len(tbl):,}")
    print(f"    both-class keys      : {len(both):,} "
          f"({len(both)/len(tbl)*100:.1f}% of keys)")
    print(f"    samples they cover   : {both_n:,} "
          f"({both_n/n*100:.1f}% of samples)")
    print(f"    size distribution    : max {sizes[0]:,}  p90 "
          f"{sizes[int(len(sizes)*0.10)]:,}  median {sizes[len(sizes)//2]:,}  "
          f"min {sizes[-1]:,}")
    buckets = collections.Counter(stratum_of(v) for v in tbl.values())
    print(f"    polarity buckets:")
    for s in STRATA:
        ks = [k for k in tbl if stratum_of(tbl[k]) == s]
        tot = sum(sum(tbl[k]) for k in ks)
        print(f"      {s:<6} {len(ks):>4} templates  {tot:>7,} samples "
              f"({tot/n*100:5.1f}%)")

    print(f"\n  FOLD BALANCE over {seeds} seeds, test_fraction={test_fraction}")
    print(f"    {'assignment':<22} {'test frac':>18} {'train pos':>18} "
          f"{'test pos':>18} {'worst dev':>10}")
    for label, strat in (("naive (unstratified)", False),
                         ("polarity-stratified", True)):
        rows = []
        for sd in range(seeds):
            test_tpl = assign_test_templates(tbl, sd, test_fraction, strat)
            te = np.array([k in test_tpl for k in keys])
            if te.all() or not te.any():
                continue
            rows.append((te.mean(), labels[~te].mean(), labels[te].mean()))
        a = np.array(rows)
        worst = np.abs(a[:, 2] - 0.5).max() * 100
        print(f"    {label:<22} "
              f"{a[:,0].mean()*100:5.1f}% [{a[:,0].min()*100:4.1f}-"
              f"{a[:,0].max()*100:4.1f}] "
              f"{a[:,1].mean()*100:5.1f}% [{a[:,1].min()*100:4.1f}-"
              f"{a[:,1].max()*100:4.1f}] "
              f"{a[:,2].mean()*100:5.1f}% [{a[:,2].min()*100:4.1f}-"
              f"{a[:,2].max()*100:4.1f}] "
              f"{worst:9.1f}")
    print("\n  Stratified assignment is the shipped default. Lower worst "
          "deviation is better.")

    print(f"\n  DISJOINTNESS CHECK (seed 0)")
    test_tpl = assign_test_templates(tbl, 0, test_fraction, True)
    te0 = np.array([k in test_tpl for k in keys])
    print(f"    test fold: {len(test_tpl):,} keys, {int(te0.sum()):,} samples "
          f"({te0.mean()*100:.1f}%), positive rate {labels[te0].mean()*100:.2f}%")
    te = np.array([k in test_tpl for k in keys])
    tr_keys = set(k for k, m in zip(keys, ~te) if m)
    te_keys = set(k for k, m in zip(keys, te) if m)
    overlap = tr_keys & te_keys
    print(f"    train templates {len(tr_keys):,}, test templates "
          f"{len(te_keys):,}, overlap {len(overlap)}")
    if overlap:
        print("    FAIL: templates appear in both folds")
        return 1
    print("    PASS: no template appears in both folds")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Derive and validate the Juliet template-disjoint split.")
    ap.add_argument("--validate", action="store_true",
                    help="print the validation report and exit")
    ap.add_argument("--file", default="juliet", help="corpus key (default juliet)")
    ap.add_argument("--cond", default="RAW", choices=["RAW", "SCRUB", "MASK"])
    ap.add_argument("--key", default="template",
                    choices=sorted(KEY_FUNCS),
                    help="which key to validate (default template)")
    ap.add_argument("--seeds", type=int, default=12,
                    help="how many assignment seeds to sweep (default 12)")
    ap.add_argument("--test-fraction", type=float, default=0.30)
    a = ap.parse_args()
    if not a.validate:
        ap.print_help()
        print("\nNothing to do. Pass --validate to run the validation report.")
        return 0
    return validate(a.file, a.cond, a.seeds, a.test_fraction, a.key)


if __name__ == "__main__":
    sys.exit(main())
