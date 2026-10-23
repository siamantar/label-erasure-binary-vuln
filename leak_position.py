"""Where in the token stream does the Juliet label leak sit?

Prior work truncates at 256 tokens. This measures whether the leak survives
that truncation, and at what position the first leak token appears.
"""

import gzip
import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FILES = [
    ("JULIET", "mne-opr-desp-juliet-labelled.pkl"),
    ("NDSS", "mne-opr-desp-ndss-labelled.pkl"),
]

CWE_RE = re.compile(r"cwe\d+")
# A pure numeric or hex immediate, e.g. 0BADh or 0BADF00Dh. Excluded so a
# hex constant that happens to spell 'bad' is not counted as a symbol leak.
IMM_RE = re.compile(r"^[0-9][0-9a-f]*h?$")
BAR = "=" * 78


def is_leak(tok_lower):
    """A leak token carries the label or the CWE id in a symbol name."""
    if CWE_RE.search(tok_lower):
        return True
    if "bad" in tok_lower or "good" in tok_lower:
        return not IMM_RE.match(tok_lower)
    return False


def first_leak_index(toks_lower):
    for i, t in enumerate(toks_lower):
        if is_leak(t):
            return i
    return -1


def rules(toks_lower, limit=None):
    """Substring rules over a token window."""
    w = toks_lower if limit is None else toks_lower[:limit]
    joined = " ".join(w)
    return ("bad" in joined), ("good" in joined)


def run(tag, fname):
    path = os.path.join(DATA, fname)
    print("\n" + BAR)
    print(f"FILE: {tag}  ({fname})")
    print(BAR)
    d = pickle.load(gzip.open(path, "rb"))
    x, z1 = d["x"], d["z1"]
    n = len(x)
    lab = np.array(z1, dtype=int)

    any_leak = np.zeros(n, bool)
    in256 = np.zeros(n, bool)
    in128 = np.zeros(n, bool)
    first = np.full(n, -1, dtype=np.int64)
    seqlen = np.zeros(n, dtype=np.int64)
    imm_guard_hits = 0

    hb_full = np.zeros(n, bool); hg_full = np.zeros(n, bool)
    hb_256 = np.zeros(n, bool); hg_256 = np.zeros(n, bool)

    for i, s in enumerate(x):
        toks = [t.lower() for t in s.split()]
        seqlen[i] = len(toks)
        fi = first_leak_index(toks)
        first[i] = fi
        any_leak[i] = fi >= 0
        in256[i] = 0 <= fi < 256
        in128[i] = 0 <= fi < 128
        for t in toks:
            if ("bad" in t or "good" in t) and IMM_RE.match(t):
                imm_guard_hits += 1
        hb_full[i], hg_full[i] = rules(toks)
        hb_256[i], hg_256[i] = rules(toks, 256)

    print(f"  samples                : {n:,}")
    print(f"  mean sequence length   : {seqlen.mean():.1f}")
    print(f"  hex immediates spelling bad/good, excluded by guard : "
          f"{imm_guard_hits:,}")

    print("\n  LEAK PRESENCE")
    print(f"    at least one leak token anywhere        : "
          f"{int(any_leak.sum()):>7,} / {n:,} = {any_leak.mean()*100:6.2f}%")
    print(f"    at least one leak token in first 256    : "
          f"{int(in256.sum()):>7,} / {n:,} = {in256.mean()*100:6.2f}%")
    print(f"    at least one leak token in first 128    : "
          f"{int(in128.sum()):>7,} / {n:,} = {in128.mean()*100:6.2f}%")
    print(f"    no leak token at all                    : "
          f"{int((~any_leak).sum()):>7,} / {n:,} = "
          f"{(~any_leak).mean()*100:6.2f}%")

    print("\n  POSITION OF FIRST LEAK TOKEN (samples that have one)")
    have = first[first >= 0]
    if have.size:
        print(f"    count   : {have.size:,}")
        print(f"    min     : {int(have.min()):,}")
        print(f"    median  : {np.median(have):.1f}")
        for p in (90, 95, 99):
            print(f"    p{p:<6}: {np.percentile(have, p):.1f}")
        print(f"    max     : {int(have.max()):,}")
    else:
        print("    none")

    print("\n  SUBSTRING RULES: FULL SEQUENCE vs 256-TOKEN PREFIX")
    maj = max(lab.mean(), 1 - lab.mean()) * 100
    print(f"    majority-class baseline : {maj:.2f}%")
    print(f"    {'rule':<36} {'full seq':>10} {'256 prefix':>12}")
    for name, pf, pp in (
        ("predict 1 if 'bad' present", hb_full.astype(int),
         hb_256.astype(int)),
        ("predict 1 if 'bad' and not 'good'", (hb_full & ~hg_full).astype(int),
         (hb_256 & ~hg_256).astype(int)),
        ("predict 0 if 'good' present", (~hg_full).astype(int),
         (~hg_256).astype(int)),
    ):
        print(f"    {name:<36} {(pf==lab).mean()*100:9.2f}% "
              f"{(pp==lab).mean()*100:11.2f}%")
    return


def main():
    print(BAR)
    print("LETHE: leak position analysis")
    print(BAR)
    print(f"python : {sys.version.split()[0]}")
    print(f"numpy  : {np.__version__}")
    print("\nLeak token definition:")
    print("  normalized (lowercased) token that either matches cwe\\d+, or")
    print("  contains 'bad' or 'good' while not being a pure numeric/hex")
    print("  immediate such as 0BADh.")
    for tag, fname in FILES:
        run(tag, fname)


if __name__ == "__main__":
    main()
