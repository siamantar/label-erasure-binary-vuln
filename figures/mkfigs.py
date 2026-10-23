"""Generate the two result figures from the measured run records.

Figures are generated, never hand-edited. Rerunning after new results
keeps the paper in sync with the data.

FONT COMPLIANCE: matplotlib defaults to Type 3 fonts in PDF output,
which Springer LNCS rejects. The rcParams block below forces CID
TrueType subsets of Computer Modern, which is compliant and matches
the body typeface. axes.unicode_minus must stay False because cmr10
has no Unicode minus glyph and fig_partitions carries a negative margin
value in the NDSS18 transformer panel. Verify with pdffonts.

    python3 mkfigs.py --runs ../../results/runs.csv
"""
import argparse, csv, statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'serif',
    'font.serif': ['cmr10', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.formatter.use_mathtext': True,
})
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--runs', default='../results/runs.csv',
                help='path to the consolidated run records')
args = ap.parse_args()

rows = list(csv.DictReader(open(args.runs)))
CTRL = ('-shuffled', '-shuftrain', '-struct')
real = [r for r in rows if not any(r['run_id'].endswith(c) for c in CTRL)]

def F(v):
    try: return float(v)
    except: return None

P = defaultdict(lambda: defaultdict(list))
B = defaultdict(lambda: defaultdict(list))
for r in real:
    if (r['split_mode'] == 'random-cwe' and r['condition'] == 'SCRUB'
            and r['max_len'] == '256'):
        P[(r['corpus'], r['arch'])][r['split_seed']].append(F(r['accuracy']))
        B[(r['corpus'], r['arch'])][r['split_seed']].append(F(r['fold_baseline']))

MODELS = [('lr', 'Logistic regression'), ('bigru', 'BiGRU'),
          ('transformer', 'Transformer')]
CORP = [('juliet', 'Juliet'), ('ndss', 'NDSS18')]

# ---- Figure: models x corpora over ten partitions ----
fig, axes = plt.subplots(2, 3, figsize=(6.4, 3.6), sharex=True, sharey=True)
handles = None
for i, (corp, clab) in enumerate(CORP):
    ref = P[(corp, 'bigru')]
    order = sorted(ref, key=lambda s: st.mean(ref[s]))
    for j, (arch, mlab) in enumerate(MODELS):
        ax = axes[i][j]
        d, b = P[(corp, arch)], B[(corp, arch)]
        xs = list(range(1, len(order) + 1))
        means = [st.mean(d[s]) for s in order]
        base = [st.mean(b[s]) for s in order]
        ax.fill_between(xs, base, means,
                        where=[m >= bb for m, bb in zip(means, base)],
                        color='#1f4e79', alpha=0.16, lw=0, zorder=1)
        ax.fill_between(xs, base, means,
                        where=[m < bb for m, bb in zip(means, base)],
                        color='#c0392b', alpha=0.22, lw=0, zorder=1)
        for x, s in zip(xs, order):
            if len(d[s]) > 1:
                ax.plot([x] * len(d[s]), d[s], 'o', ms=1.8, color='0.72', zorder=2)
        hb, = ax.plot(xs, base, 's--', ms=2.2, lw=0.7, color='0.45',
                      zorder=3, label='majority-class baseline')
        ha, = ax.plot(xs, means, 'o-', ms=3, lw=1.0, color='#1f4e79',
                      zorder=4, label='accuracy')
        if handles is None:
            handles = [ha, hb]
        m, sd = st.mean(means), st.stdev(means)
        lift = m - st.mean(base)
        ax.set_ylim(45, 102)
        ax.set_yticks([50, 60, 70, 80, 90, 100])
        if j == 0:
            ax.set_ylabel(f'{clab}\naccuracy (%)')
        if i == 0:
            ax.set_title(mlab, fontsize=8, pad=3)
        if i == 1:
            ax.set_xlabel('partition (sorted)')
        ax.set_xticks([1, 5, 10])
        ax.text(0.04, 0.965, f'{m:.1f} $\\pm$ {sd:.2f}\nmargin {lift:+.1f}',
                transform=ax.transAxes, fontsize=6.5, color='#1f4e79',
                va='top', ha='left', linespacing=1.35)
fig.legend(handles, ['accuracy', 'majority-class baseline'], fontsize=7, ncol=2,
           loc='lower center', frameon=False, bbox_to_anchor=(0.5, -0.015),
           handlelength=2.0, columnspacing=1.8)
fig.tight_layout(pad=0.4, w_pad=0.4, h_pad=0.7, rect=[0, 0.055, 1, 1])
fig.savefig('fig_partitions.pdf')

# ---- Figure: variance decomposition ----
CANON = 88.85
fig2, ax = plt.subplots(figsize=(3.2, 2.3))
ref = P[('juliet', 'bigru')]
order = sorted(ref, key=lambda s: st.mean(ref[s]))
means = [st.mean(ref[s]) for s in order]
m, sd = st.mean(means), st.stdev(means)
ax.axhspan(m - sd, m + sd, color='#c0392b', alpha=0.09, zorder=0)
ax.axhline(m, lw=0.7, color='#c0392b', zorder=2)
for x, s in enumerate(order, 1):
    ax.plot([x, x], [min(ref[s]), max(ref[s])], '-', lw=0.8, color='0.72', zorder=1)
    ax.plot([x] * len(ref[s]), ref[s], 'o', ms=2.6, color='0.72', zorder=1)
ax.plot(range(1, 11), means, 'o-', ms=3.2, lw=1.0, color='#1f4e79', zorder=3)
ax.axhline(CANON, lw=0.9, ls=':', color='#7d3c98', zorder=3)
ax.plot([10.5], [CANON], '*', ms=8, color='#7d3c98', zorder=4, clip_on=False)
ax.text(10.9, CANON, f'initial\npartition {CANON}', fontsize=6.3,
        color='#7d3c98', va='center', linespacing=1.25)
ax.annotate('', xy=(9.55, m - sd), xytext=(9.55, m + sd),
            arrowprops=dict(arrowstyle='<->', lw=0.7, color='#c0392b'))
ax.text(9.35, m, f'partition\nSD {sd:.2f}', fontsize=6.3, color='#c0392b',
        va='center', ha='right', linespacing=1.25)
k = order[0]
seedsd = st.mean([st.stdev(v) for v in ref.values() if len(v) > 1])
ax.annotate('', xy=(1, min(ref[k])), xytext=(1, max(ref[k])),
            arrowprops=dict(arrowstyle='<->', lw=0.7, color='0.30'))
ax.text(1.35, st.mean(ref[k]) - 0.6, f'seed SD {seedsd:.2f}', fontsize=6.3,
        color='0.30', va='center')
ax.set_xticks(range(1, 11))
ax.set_xlabel('partition (sorted by difficulty)')
ax.set_ylabel('accuracy (%)')
ax.set_ylim(75, 91)
ax.set_xlim(0.3, 10.8)
fig2.tight_layout(pad=0.3)
fig2.savefig('fig_variance.pdf')
print("both figures regenerated")
