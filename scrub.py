"""Build three corpus conditions from the released token streams.

RAW    tokens exactly as released
SCRUB  label-carrying symbols replaced by SYM, instruction semantics kept
MASK   SCRUB plus every recognized C library name replaced by LIBFN

Deterministic: no randomness, no data-dependent ordering. Re-running on the
same inputs produces byte-identical decisions.
"""

import collections
import gzip
import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DERIVED = os.path.join(DATA, "derived")
FILES = [
    ("juliet", "mne-opr-desp-juliet-labelled.pkl"),
    ("ndss", "mne-opr-desp-ndss-labelled.pkl"),
]
CONDITIONS = ["RAW", "SCRUB", "MASK"]
BAR = "=" * 78

SYM = "SYM"
LIBFN = "LIBFN"

CWE_RE = re.compile(r"cwe\d+")
# Itanium C++ mangling as emitted by IDA: _Z, __Z, and the j___Z thunk form.
MANGLED_RE = re.compile(r"^j?_+Z")
# Pure numeric or hex immediate, e.g. 0BADh. Guarded so a constant that
# happens to spell 'bad' is not mistaken for a symbol.
IMM_RE = re.compile(r"^[0-9][0-9a-f]*h?$")

# Juliet and NDSS test harness scaffolding. Not part of the C library and not
# part of the instruction semantics, so it goes to SYM.
HARNESS = {
    "printline", "printwline", "printintline", "printlongline",
    "printlonglongline", "printhexcharline", "printstructline",
    "printfloatline", "printdoubleline", "printunsignedline",
    "printsizetline", "printhexunsignedcharline", "printwcharline",
    "printbytesline", "printhexbytesline",
    "goodg2b", "goodb2g", "goodg2b1", "goodg2b2", "goodb2g1", "goodb2g2",
    "badsink", "badsource", "goodsink", "goodsource",
    "staticreturnstrue", "staticreturnsfalse",
    "globalreturnstrue", "globalreturnsfalse",
    "globalreturnstrueorfalse", "globalreturnsfalseortrue",
    "staticfive", "staticfalse", "statictrue",
}

# Recognized C library / runtime names, canonical spelling. Matching is
# normalized: leading and trailing underscores stripped, case folded, so
# strcpy, _strcpy, and STRCPY all collapse to the same entry.
LIBNAMES = [
    "strcpy", "strncpy", "strcat", "strncat", "memcpy", "memmove", "memset",
    "malloc", "calloc", "realloc", "free", "alloca",
    "printf", "sprintf", "snprintf", "fprintf", "wprintf", "vprintf",
    "puts", "gets", "fgets", "fgetws", "fputs", "scanf", "sscanf", "fscanf",
    "swscanf", "wscanf",
    "strlen", "strcmp", "strncmp", "strchr", "strstr", "strtok", "strdup",
    "wcscpy", "wcsncpy", "wcscat", "wcsncat", "wcslen", "wmemset", "wmemcpy",
    "fopen", "fclose", "fread", "fwrite", "fseek", "ftell", "remove",
    "rename", "unlink", "open", "close", "read", "write", "stat", "chmod",
    "socket", "recv", "send", "accept", "listen", "bind", "connect",
    "atoi", "atol", "atof", "strtol", "strtoul", "rand", "srand", "time",
    "exit", "abort", "system", "getenv", "setenv", "execl", "execlp",
    "execv", "popen", "pclose",
    "isxdigit", "iswxdigit", "isalpha", "isdigit", "toupper", "tolower",
]
LIBSET = {n.lower() for n in LIBNAMES}


def norm(tok):
    """Normalized form used for library-name matching."""
    return tok.strip("_").lower()


def is_leaky(tok):
    low = tok.lower()
    if CWE_RE.search(low):
        return True
    if ("bad" in low or "good" in low) and not IMM_RE.match(low):
        return True
    return False


def build_mapper(condition):
    """Return f(token) -> token for the given condition, plus a stats dict."""
    cache = {}
    stats = collections.Counter()

    def decide(tok):
        # order matters: mangled and leaky symbols outrank library matching
        if MANGLED_RE.match(tok):
            return SYM, "mangled"
        if is_leaky(tok):
            return SYM, "leaky"
        n = norm(tok)
        if n in HARNESS:
            return SYM, "harness"
        if condition == "MASK" and n in LIBSET:
            return LIBFN, "libfn"
        return tok, None

    def f(tok):
        hit = cache.get(tok)
        if hit is None:
            out, why = decide(tok)
            cache[tok] = (out, why)
            hit = (out, why)
        out, why = hit
        if why:
            stats[why] += 1
        return out

    return f, stats, cache


def substring_rules(xs, labels):
    """Accuracy of the trivial label-leak rules on a corpus."""
    n = len(xs)
    hb = np.zeros(n, bool)
    hg = np.zeros(n, bool)
    for i, s in enumerate(xs):
        low = s.lower()
        hb[i] = "bad" in low
        hg[i] = "good" in low
    lab = np.asarray(labels, dtype=int)
    return {
        "bad": float(((hb.astype(int)) == lab).mean() * 100),
        "bad_not_good": float((((hb & ~hg).astype(int)) == lab).mean() * 100),
        "not_good": float((((~hg).astype(int)) == lab).mean() * 100),
        "majority": float(max(lab.mean(), 1 - lab.mean()) * 100),
    }


def corpus_stats(xs):
    vocab = collections.Counter()
    total = 0
    lens = []
    for s in xs:
        t = s.split()
        vocab.update(t)
        total += len(t)
        lens.append(len(t))
    return len(vocab), total, float(np.mean(lens)), vocab


def main():
    os.makedirs(DERIVED, exist_ok=True)
    print(BAR)
    print("LETHE: corpus scrubbing pipeline")
    print(BAR)
    print(f"python : {sys.version.split()[0]}")
    print(f"numpy  : {np.__version__}")
    print(f"output : {DERIVED}")
    print("\nSCRUB replaces a token with SYM when it is a C++ mangled symbol")
    print("(_Z, __Z, j___Z), or carries 'bad'/'good'/CWE<digits> in a symbol")
    print("name, or is test-harness scaffolding. The bad/good test applies to")
    print("every token position, not only call targets, because operand names")
    print("such as [rbp+dataBadBuffer] leak the label too. Pure numeric and")
    print("hex immediates are guarded so a constant like 0BADh is preserved.")
    print("MASK additionally replaces recognized C library names with LIBFN.")

    summary = {}
    for key, fname in FILES:
        path = os.path.join(DATA, fname)
        print("\n" + BAR)
        print(f"FILE: {key.upper()}  ({fname})")
        print(BAR)
        d = pickle.load(gzip.open(path, "rb"))
        x, y, z1, z2 = d["x"], d["y"], d["z1"], d["z2"]
        n = len(x)
        print(f"  samples : {n:,}")

        for cond in CONDITIONS:
            if cond == "RAW":
                xs = x
                replaced = collections.Counter()
                cache = {}
            else:
                f, replaced, cache = build_mapper(cond)
                xs = [" ".join(f(t) for t in s.split()) for s in x]

            vsz, total, meanlen, vocab = corpus_stats(xs)
            nrep = sum(replaced.values())
            rules = substring_rules(xs, z1)

            out = os.path.join(DERIVED, f"{key}-{cond.lower()}.pkl.gz")
            with gzip.open(out, "wb", compresslevel=6) as fh:
                pickle.dump({"x": xs, "y": y, "z1": z1, "z2": z2}, fh,
                            protocol=4)

            print(f"\n  --- {cond} ---")
            print(f"    vocabulary size        : {vsz:>12,}")
            print(f"    total tokens           : {total:>12,}")
            print(f"    mean sequence length   : {meanlen:>12.2f}")
            print(f"    tokens replaced        : {nrep:>12,} "
                  f"({nrep/total*100:.2f}% of all tokens)")
            if replaced:
                for w, c in sorted(replaced.items()):
                    print(f"      by {w:<9}: {c:>12,}")
                print(f"    distinct token forms replaced : "
                      f"{sum(1 for v in cache.values() if v[1]):,}")
            print(f"    written                : {out} "
                  f"({os.path.getsize(out):,} bytes)")
            print(f"    substring rules (majority baseline "
                  f"{rules['majority']:.2f}%):")
            for lbl, k in (("predict 1 if 'bad'", "bad"),
                           ("predict 1 if 'bad' and not 'good'",
                            "bad_not_good"),
                           ("predict 0 if 'good'", "not_good")):
                print(f"      {lbl:<36} {rules[k]:6.2f}%")
            if cond in ("SCRUB", "MASK"):
                worst = max(rules["bad"], rules["bad_not_good"],
                            rules["not_good"])
                delta = worst - rules["majority"]
                ok = delta <= 1.0
                print(f"    CHECK: worst rule {worst:.2f}% vs baseline "
                      f"{rules['majority']:.2f}%, delta {delta:+.2f} pts "
                      f"=> {'PASS' if ok else 'FAIL'}")
            summary[(key, cond)] = {
                "vocab": vsz, "total": total, "mean": meanlen,
                "replaced": nrep, "rules": rules,
            }
            if cond != "RAW":
                del xs

        # MASK name list, printed once per file
        f, _, _ = build_mapper("MASK")
        masked = sorted({norm(t) for t in
                         collections.Counter(
                             tok for s in x for tok in s.split()).keys()
                         if norm(t) in LIBSET and not MANGLED_RE.match(t)
                         and not is_leaky(t)})
        print(f"\n  MASK: recognized C library names actually present "
              f"({len(masked)}):")
        for i in range(0, len(masked), 6):
            print("    " + "  ".join(f"{m:<14}" for m in masked[i:i + 6]))
        del d, x, y, z1, z2

    print("\n" + BAR)
    print("SUMMARY")
    print(BAR)
    print(f"  {'file':<8} {'cond':<6} {'vocab':>10} {'tokens':>14} "
          f"{'mean len':>9} {'replaced':>12} {'worst rule':>11} "
          f"{'baseline':>9}")
    for (key, cond), s in summary.items():
        worst = max(s["rules"]["bad"], s["rules"]["bad_not_good"],
                    s["rules"]["not_good"])
        print(f"  {key:<8} {cond:<6} {s['vocab']:>10,} {s['total']:>14,} "
              f"{s['mean']:>9.1f} {s['replaced']:>12,} {worst:>10.2f}% "
              f"{s['rules']['majority']:>8.2f}%")

    jr = summary[("juliet", "RAW")]["vocab"]
    js = summary[("juliet", "SCRUB")]["vocab"]
    nr = summary[("ndss", "RAW")]["vocab"]
    print(f"\n  Vocabulary collapse check:")
    print(f"    Juliet RAW   : {jr:,}")
    print(f"    Juliet SCRUB : {js:,}  ({js/jr*100:.2f}% of RAW)")
    print(f"    NDSS RAW     : {nr:,}  (reference scale)")
    if js > 10 * nr:
        print(f"    *** Juliet SCRUB vocabulary is still {js/nr:.1f}x the NDSS")
        print(f"        scale. Some symbol form was missed.")
    else:
        print(f"    Juliet SCRUB is {js/nr:.1f}x NDSS scale; the symbol "
              f"explosion is gone.")


if __name__ == "__main__":
    main()
