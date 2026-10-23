"""Structural inspection of the Juliet and NDSS18 labelled token datasets.

Read-only. Loads each gzipped pickle, measures shape, class balance, per-CWE
support, vocabulary, sequence length, duplicate structure, and platform split.
"""

import collections
import gzip
import hashlib
import os
import pickle
import statistics
import sys

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = [
    ("JULIET", "mne-opr-desp-juliet-labelled.pkl", 83326),
    ("NDSS",   "mne-opr-desp-ndss-labelled.pkl",   14166),
]
LIBFUNCS = [
    "strcpy", "strncpy", "strcat", "strncat", "memcpy", "memmove", "malloc",
    "free", "printf", "sprintf", "snprintf", "gets", "fgets", "scanf",
    "wcscpy", "wcsncpy", "wcscat", "alloca", "CreateFile",
    "WideCharToMultiByte",
]
BAR = "=" * 78


def h1(t):
    print("\n" + BAR); print(t); print(BAR)


def h2(t):
    print("\n" + "-" * 78); print(t); print("-" * 78)


def load(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def pct(v, p):
    return float(np.percentile(v, p))


def inspect(tag, fname, expected_n):
    path = os.path.join(DATA, fname)
    h1(f"FILE: {tag}  ({fname})")
    print(f"  path       : {path}")
    print(f"  bytes      : {os.path.getsize(path):,}")
    d = load(path)
    x, y, z1, z2 = d["x"], d["y"], d["z1"], d["z2"]

    # ---------------------------------------------------- 3a shape/integrity
    h2("3a. SHAPE AND INTEGRITY")
    print(f"  top-level type   : {type(d).__name__}, keys = {list(d.keys())}")
    for k in ("x", "y", "z1", "z2"):
        v = d[k]
        print(f"  {k:<3} type={type(v).__name__:<5} len={len(v):>7,}  "
              f"elem type={type(v[0]).__name__}")
    lens = {k: len(d[k]) for k in ("x", "y", "z1", "z2")}
    same = len(set(lens.values())) == 1
    print(f"  all four same length : {same}   ({sorted(set(lens.values()))})")
    n = len(x)
    print(f"  sample count  actual : {n:,}")
    print(f"  sample count  expected: {expected_n:,}   "
          f"MATCH: {n == expected_n}")

    c1 = collections.Counter(z1)
    print(f"\n  class balance from z1 : {dict(sorted(c1.items()))}")
    for k in sorted(c1):
        print(f"    label {k} : {c1[k]:>7,}  ({c1[k]/n*100:5.2f}%)")

    cwes = collections.Counter(z2)
    print(f"\n  distinct CWE values in z2 : {len(cwes)}")

    # ---------------------------------------------------- 3b per-CWE support
    h2("3b. PER-CWE SUPPORT")
    per = collections.defaultdict(lambda: collections.Counter())
    for c, l in zip(z2, z1):
        per[c][l] += 1
    rows = []
    for c, cc in per.items():
        vuln, non = cc.get(1, 0), cc.get(0, 0)
        rows.append((c, vuln + non, vuln, non))
    rows.sort(key=lambda r: (-r[1], r[0]))
    print(f"  {'CWE':<12} {'total':>8} {'vuln(1)':>9} {'nonvuln(0)':>11} "
          f"{'both?':>6} {'>=200+both':>11}")
    ok = 0
    for c, t, v, nv in rows:
        both = v > 0 and nv > 0
        clears = both and t >= 200
        if clears:
            ok += 1
        print(f"  {c:<12} {t:>8,} {v:>9,} {nv:>11,} {str(both):>6} "
              f"{str(clears):>11}")
    print(f"\n  TOTAL CWEs                                : {len(rows)}")
    print(f"  CWEs with both classes present            : "
          f"{sum(1 for r in rows if r[2] > 0 and r[3] > 0)}")
    print(f"  *** CWEs with >=200 samples AND both classes : {ok}")

    # ---------------------------------------------------- 3c raw samples
    h2("3c. RAW SAMPLES, VERBATIM AND UNTRUNCATED")
    print(f"  type of an x element : {type(x[0]).__name__}")
    probe = x[0]
    if isinstance(probe, str):
        print("  x is a SINGLE STRING of space-separated tokens, "
              "not a list of strings.")
        print(f"  x[0] character length : {len(probe):,}")
        print(f"  x[0] token count      : {len(probe.split()):,}")
    for i in (0, 1, 2):
        print(f"\n  ---------- SAMPLE INDEX {i} ----------")
        print(f"  x[{i}] =")
        print(x[i])
        print(f"\n  y[{i}] =")
        print(y[i])
        print(f"\n  z1[{i}] = {z1[i]!r}   (type {type(z1[i]).__name__})")
        print(f"  z2[{i}] = {z2[i]!r}   (type {type(z2[i]).__name__})")

    # ------------------------------- streaming pass: vocab, lengths, dupes
    vocab = collections.Counter()
    doc_freq = collections.Counter()
    seq_len = np.empty(n, dtype=np.int64)
    hashes = [None] * n
    call_targets = collections.Counter()
    call_windows = []
    reg_sysv = np.zeros(n, dtype=np.int64)
    reg_win = np.zeros(n, dtype=np.int64)
    SYSV = {"rdi", "rsi", "edi", "esi"}
    WIN = {"rcx", "rdx", "r8", "r9", "ecx", "edx", "r8d", "r9d"}
    NORM_LIB = {f.lower() for f in LIBFUNCS}
    api_hit = 0

    for i, s in enumerate(x):
        toks = s.split()
        seq_len[i] = len(toks)
        vocab.update(toks)
        uniq = set(toks)
        doc_freq.update(uniq)
        hashes[i] = hashlib.sha1(s.encode("utf-8", "surrogatepass")).hexdigest()
        for j, t in enumerate(toks):
            if t == "call":
                if j + 1 < len(toks):
                    call_targets[toks[j + 1]] += 1
                if len(call_windows) < 10:
                    call_windows.append((i, j, toks[max(0, j - 5):j + 6]))
        if any(t.strip("_").lower() in NORM_LIB for t in uniq):
            api_hit += 1
        reg_sysv[i] = sum(1 for t in toks if t in SYSV)
        reg_win[i] = sum(1 for t in toks if t in WIN)

    # ---------------------------------------------------- 3c vocab
    h2("3c (cont). VOCABULARY")
    print(f"  total tokens (with repeats) : {sum(vocab.values()):,}")
    print(f"  vocabulary size (distinct)  : {len(vocab):,}")
    print(f"\n  60 most frequent tokens:")
    print(f"  {'rank':>4} {'token':<28} {'count':>14} {'% of tokens':>12}")
    tot_tok = sum(vocab.values())
    for r, (t, c) in enumerate(vocab.most_common(60), 1):
        print(f"  {r:>4} {t:<28} {c:>14,} {c/tot_tok*100:11.4f}%")

    # ---------------------------------------------------- 3d library funcs
    h2("3d. DO LIBRARY FUNCTION NAMES SURVIVE?")
    print("  Matching is normalized: leading/trailing underscores stripped and")
    print("  case folded, because IDA emits names like _strcpy, ___stack_chk_fail")
    print("  and SNPRINTF. Every raw variant found is listed.\n")

    norm_map = collections.defaultdict(list)
    for t in vocab:
        norm_map[t.strip("_").lower()].append(t)

    print(f"  {'requested':<22} {'present':>8} {'variants (raw tokens)':<44} "
          f"{'freq':>12} {'samples':>9} {'%samples':>9}")
    found_any = []
    n_present = 0
    for fn in LIBFUNCS:
        variants = sorted(norm_map.get(fn.lower(), []), key=lambda t: -vocab[t])
        if variants:
            n_present += 1
            found_any.extend(variants)
            freq = sum(vocab[v] for v in variants)
            samp = max(doc_freq[v] for v in variants)
            shown = ", ".join(variants[:6]) + ("..." if len(variants) > 6 else "")
            print(f"  {fn:<22} {'YES':>8} {shown:<44} {freq:>12,} "
                  f"{samp:>9,} {samp/n*100:8.2f}%")
            for v in variants:
                print(f"  {'':<22} {'':>8}   {v:<42} {vocab[v]:>12,} "
                      f"{doc_freq[v]:>9,} {doc_freq[v]/n*100:8.2f}%")
        else:
            print(f"  {fn:<22} {'no':>8} {'-':<44} {'-':>12} {'-':>9} {'-':>9}")

    print(f"\n  requested names present : {n_present} / {len(LIBFUNCS)}")
    print(f"  samples containing >=1 requested library name (any variant) : "
          f"{api_hit:,} / {n:,} = {api_hit/n*100:.2f}%")

    print(f"\n  Other recognizable API / runtime names among call targets:")
    KNOWN = ["wmemset", "wcslen", "wcsncat", "calloc", "realloc", "fopen",
             "fclose", "fprintf", "fscanf", "fgetws", "fwrite", "fread",
             "strlen", "sscanf", "swscanf", "wprintf", "puts", "rand",
             "srand", "time", "exit", "socket", "recv", "send", "accept",
             "listen", "bind", "atoi", "getenv", "system", "execl", "open",
             "close", "read", "write", "chmod", "stat", "unlink", "rename",
             "iswxdigit", "isxdigit", "stack_chk_fail", "cxa_throw",
             "unwind_resume", "znwm", "znam", "zdlpv", "zdapv"]
    seen = set()
    rows = []
    for k in KNOWN:
        vs = norm_map.get(k.lower(), [])
        if vs:
            freq = sum(vocab[v] for v in vs)
            samp = max(doc_freq[v] for v in vs)
            rows.append((freq, k, ", ".join(sorted(vs, key=lambda t: -vocab[t])[:4]),
                         samp))
            seen.add(k)
    rows.sort(reverse=True)
    print(f"    {'name':<20} {'variants':<34} {'freq':>12} {'samples':>9} "
          f"{'%samples':>9}")
    for freq, k, vs, samp in rows:
        print(f"    {k:<20} {vs:<34} {freq:>12,} {samp:>9,} "
              f"{samp/n*100:8.2f}%")

    print(f"\n  Top 60 call targets actually observed (token following 'call'):")
    print(f"  {'rank':>4} {'target':<52} {'count':>12}")
    for r, (t, c) in enumerate(call_targets.most_common(60), 1):
        print(f"  {r:>4} {t:<52} {c:>12,}")
    print(f"\n  distinct call targets : {len(call_targets):,}")
    print(f"  total 'call' tokens   : {vocab.get('call', 0):,}")

    print(f"\n  10 token windows around 'call' (5 before, token, 5 after):")
    for (i, j, w) in call_windows:
        print(f"    sample {i:>6} pos {j:>6} : {' '.join(w)}")

    # ---------------------------------------------------- 3e lengths
    h2("3e. SEQUENCE LENGTHS (tokens)")
    print(f"  min     : {int(seq_len.min()):,}")
    print(f"  max     : {int(seq_len.max()):,}")
    print(f"  mean    : {seq_len.mean():.2f}")
    print(f"  median  : {statistics.median(seq_len.tolist()):.1f}")
    for p in (90, 95, 99):
        print(f"  p{p:<6}: {pct(seq_len, p):.1f}")
    over = int((seq_len > 256).sum())
    print(f"\n  samples with length > 256 : {over:,} / {n:,} = {over/n*100:.2f}%")
    print(f"  tokens retained if truncated at 256 : "
          f"{int(np.minimum(seq_len, 256).sum()):,} of {int(seq_len.sum()):,} "
          f"= {np.minimum(seq_len,256).sum()/seq_len.sum()*100:.2f}%")

    # ---------------------------------------------------- 3f duplicates
    h2("3f. EXACT DUPLICATE SEQUENCES")
    hc = collections.Counter(hashes)
    distinct = len(hc)
    dup_groups = {h: c for h, c in hc.items() if c > 1}
    n_dup_seq = sum(dup_groups.values())
    print(f"  total samples              : {n:,}")
    print(f"  distinct x sequences       : {distinct:,}")
    print(f"  duplicated distinct seqs   : {len(dup_groups):,}")
    print(f"  samples in duplicate groups: {n_dup_seq:,} "
          f"({n_dup_seq/n*100:.2f}%)")
    print(f"  redundancy (n - distinct)  : {n - distinct:,}")

    by_hash_lab = collections.defaultdict(collections.Counter)
    by_hash_cwe = collections.defaultdict(collections.Counter)
    for h, l, c in zip(hashes, z1, z2):
        by_hash_lab[h][l] += 1
        by_hash_cwe[h][c] += 1
    conflict = {h: cc for h, cc in by_hash_lab.items() if len(cc) > 1}
    conflict_samples = sum(sum(cc.values()) for cc in conflict.values())
    min_err = sum(sum(cc.values()) - max(cc.values()) for cc in conflict.values())
    print(f"\n  sequences appearing with BOTH z1 labels : {len(conflict):,}")
    print(f"  samples involved in those sequences      : {conflict_samples:,} "
          f"({conflict_samples/n*100:.2f}%)")
    print("\n  Irreducible error arithmetic:")
    print("    For each conflicting sequence, any deterministic classifier must")
    print("    output one label, so it errs on the minority members.")
    print("      errors = sum over conflicting seqs of "
          "(group_size - majority_count)")
    print(f"      errors = {min_err:,}")
    print(f"      lower bound on error rate = {min_err:,} / {n:,} "
          f"= {min_err/n*100:.4f}%")
    print(f"      => ceiling on achievable accuracy = "
          f"{100 - min_err/n*100:.4f}%")

    print("\n  Conflicting duplicates per CWE:")
    percwe_conf = collections.Counter()
    percwe_err = collections.Counter()
    for h, cc in conflict.items():
        grp = by_hash_cwe[h]
        tot = sum(cc.values())
        err = tot - max(cc.values())
        for c, k in grp.items():
            percwe_conf[c] += k
            percwe_err[c] += err * (k / tot)
    if percwe_conf:
        print(f"    {'CWE':<12} {'conflict samples':>17} {'CWE total':>10} "
              f"{'% of CWE':>9} {'est errors':>11}")
        for c, k in sorted(percwe_conf.items(), key=lambda kv: -kv[1]):
            tot_c = sum(per[c].values())
            print(f"    {c:<12} {k:>17,} {tot_c:>10,} "
                  f"{k/tot_c*100:8.2f}% {percwe_err[c]:>11.1f}")
    else:
        print("    none")

    # ---------------------------------------------------- 3g platform split
    h2("3g. PLATFORM SPLIT PROBE (SysV vs Windows x64 register usage)")
    print(f"  SysV markers  : {sorted(SYSV)}")
    print(f"  Win64 markers : {sorted(WIN)}")
    print(f"  mean SysV-reg tokens per sample : {reg_sysv.mean():.2f}")
    print(f"  mean Win64-reg tokens per sample: {reg_win.mean():.2f}")
    sysv_dom = reg_sysv > reg_win
    win_dom = reg_win > reg_sysv
    tie = ~(sysv_dom | win_dom)
    print(f"  samples SysV-dominant  : {int(sysv_dom.sum()):,}")
    print(f"  samples Win64-dominant : {int(win_dom.sum()):,}")
    print(f"  samples tied           : {int(tie.sum()):,}")

    # Linux/Windows textual markers
    for marker in ("fs:28h", "__stack_chk_fail", "__security_cookie",
                   "GetLastError", "WideCharToMultiByte", "wcscpy"):
        print(f"  samples containing {marker:<22}: {doc_freq.get(marker,0):>7,}")

    print("\n  Contiguity check: is there a boundary near index 7,160?")
    for cut in (7000, 7100, 7160, 7200, 7500):
        if cut < n:
            a = sysv_dom[:cut].mean() * 100
            b = sysv_dom[cut:].mean() * 100
            print(f"    cut at {cut:>6,}: SysV-dominant before = {a:6.2f}%, "
                  f"after = {b:6.2f}%,  separation = {abs(a-b):6.2f} pts")
    # best cut by separation
    best = None
    step = max(1, n // 2000)
    for cut in range(step, n, step):
        a = sysv_dom[:cut].mean()
        b = sysv_dom[cut:].mean()
        sep = abs(a - b)
        if best is None or sep > best[1]:
            best = (cut, sep)
    print(f"    scanned {n//step:,} candidate cuts; strongest separation at "
          f"index {best[0]:,} (separation {best[1]*100:.2f} pts)")

    # register-width split, the discriminator that actually separates NDSS
    b32 = np.zeros(n, bool); b64 = np.zeros(n, bool); glibc = np.zeros(n, bool)
    for i, sx in enumerate(x):
        t = set(sx.split())
        b64[i] = bool(t & {"rbp", "rsp"})
        b32[i] = bool(t & {"ebp", "esp"}) and not b64[i]
        glibc[i] = bool(t & {"fs:28h", "__stack_chk_fail", "___stack_chk_fail"})
    print("\n  Register-width split (x86 vs x86-64):")
    print(f"    pure 32-bit (ebp/esp, no rbp/rsp) : {int(b32.sum()):,}")
    print(f"    64-bit (rbp/rsp present)          : {int(b64.sum()):,}")
    print(f"    sum                               : {int(b32.sum()+b64.sum()):,}")
    print(f"    glibc/SysV marker present         : {int(glibc.sum()):,}")
    print("\n  Is the file ordered by platform? marker density per decile:")
    for k in range(10):
        a, b = k * n // 10, (k + 1) * n // 10
        print(f"    idx {a:>7,}-{b:>7,} : glibc {glibc[a:b].mean()*100:6.2f}%   "
              f"32-bit {b32[a:b].mean()*100:6.2f}%")
    alt = int((glibc[1:] != glibc[:-1]).sum())
    print(f"    alternations between marker states : {alt:,}")
    print("    A contiguous platform split would show ~1 alternation.")

    # ---------------------------------------------------- 3h label leakage
    h2("3h. LABEL LEAKAGE PROBE (symbol names vs ground truth)")
    hb = np.zeros(n, bool); hg = np.zeros(n, bool); hc = np.zeros(n, bool)
    for i, sx in enumerate(x):
        low = sx.lower()
        hb[i] = "bad" in low
        hg[i] = "good" in low
        hc[i] = z2[i].lower() in low
    lab = np.array(z1, dtype=int)
    print(f"  samples containing substring 'bad'      : {int(hb.sum()):,} "
          f"({hb.mean()*100:.2f}%)")
    print(f"  samples containing substring 'good'     : {int(hg.sum()):,} "
          f"({hg.mean()*100:.2f}%)")
    print(f"  samples whose own z2 CWE id appears in x: {int(hc.sum()):,} "
          f"({hc.mean()*100:.2f}%)")
    maj = max(lab.mean(), 1 - lab.mean()) * 100
    print(f"  majority-class baseline accuracy        : {maj:.2f}%")
    for nm, pred in (("predict 1 if 'bad' present", hb.astype(int)),
                     ("predict 1 if 'bad' and not 'good'",
                      (hb & ~hg).astype(int)),
                     ("predict 0 if 'good' present", (~hg).astype(int))):
        print(f"  rule [{nm:<34}] accuracy = {(pred==lab).mean()*100:6.2f}%")
    print("  A trivial substring rule beating the majority baseline by a wide")
    print("  margin means the symbol names carry the label directly.")

    del d, x, y, z1, z2
    return {"n": n, "cwe_ok": ok, "found": found_any, "vocab": len(vocab)}


def main():
    print(BAR)
    print("LETHE dataset gate inspection")
    print(BAR)
    print(f"python     : {sys.version.split()[0]}")
    print(f"numpy      : {np.__version__}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    summary = {}
    for tag, fname, exp in FILES:
        summary[tag] = inspect(tag, fname, exp)
    h1("SUMMARY")
    for tag, s in summary.items():
        print(f"  {tag:<8} n={s['n']:>7,}  vocab={s['vocab']:>7,}  "
              f"CWEs >=200 with both classes = {s['cwe_ok']}")
        print(f"           library tokens found: "
              f"{sorted(set(s['found'])) if s['found'] else 'NONE'}")


if __name__ == "__main__":
    main()
