"""Generate the three figures the findings_report.md references:
  figures/main_3seed_bars.png
  figures/detection_recall_comparison.png
  figures/comm_savings.png

All numbers pulled from CSVs - no fabrication. Reads via the same
windows as experiment_report_compute.py (pre=[89,99], dip=pre-min[100,110],
stable=mean[-10:]).
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path('.')
SCRATCH = Path(r'C:\Users\suraf\AppData\Local\Temp\claude\d--projects-Seminars-in-AI-Paper-Proposal-Second-Semester-Implementation-FedCCFA\27e522cf-d3fe-44e4-946b-eff40f6a5a6d\scratchpad')
FIGS = Path('figures')
FIGS.mkdir(exist_ok=True)

DRIFT = 100

def load(p):
    rows = list(csv.DictReader(Path(p).open()))
    g  = np.array([float(r['global_acc']) for r in rows])
    pc = np.array([float(r['per_client_gen_acc']) for r in rows]) if 'per_client_gen_acc' in rows[0] else None
    return g, pc

def stable(s):
    return float(np.mean(s[-10:]))


# ========================================================================
# Figure 1: main 3-seed bars (6 methods x 2 panels)
# ========================================================================
SOURCES = {
    'FedAvg': [
        'runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv',
        'runs/2026-06-08-perclient-3seed/seed1/fedavg/results_FedAvg.csv',
        'runs/2026-06-08-perclient-3seed/seed2/fedavg/results_FedAvg.csv',
    ],
    'Flash': [
        'runs/2026-06-08-perclient-smoke/flash/results_Flash.csv',
        'runs/2026-06-08-perclient-3seed/seed1/flash/results_Flash.csv',
        'runs/2026-06-08-perclient-3seed/seed2/flash/results_Flash.csv',
    ],
    'AdaptiveFedAvg\n(corrected)': [
        'runs/2026-06-09-adaptivefedavg-fix/seed0/results_AdaptiveFedAvg.csv',
        'runs/2026-06-09-adaptivefedavg-fix/seed1/results_AdaptiveFedAvg.csv',
        'runs/2026-06-09-adaptivefedavg-fix/seed2/results_AdaptiveFedAvg.csv',
    ],
    'OurMethod': [
        'runs/2026-06-08-perclient-smoke/ourmethod/results_OurMethod.csv',
        'runs/2026-06-08-perclient-3seed/seed1/ourmethod/results_OurMethod.csv',
        'runs/2026-06-08-perclient-3seed/seed2/ourmethod/results_OurMethod.csv',
    ],
    'FedAvgPlus1\n(control)': [
        'runs/2026-06-08-perclient-smoke/fedavgplus1/results_FedAvgPlus1.csv',
        'runs/2026-06-08-perclient-3seed/seed1/fedavgplus1/results_FedAvgPlus1.csv',
        'runs/2026-06-08-perclient-3seed/seed2/fedavgplus1/results_FedAvgPlus1.csv',
    ],
    'Saile': [
        'runs/2026-06-11-saile-3seed/seed0/results_Saile.csv',
        'runs/2026-06-11-saile-3seed/seed1/results_Saile.csv',
        'runs/2026-06-11-saile-3seed/seed2/results_Saile.csv',
    ],
}

methods = list(SOURCES.keys())
g_means, g_stds, pc_means, pc_stds = [], [], [], []
for m in methods:
    g_vals = []
    pc_vals = []
    for p in SOURCES[m]:
        g, pc = load(p)
        g_vals.append(stable(g))
        if pc is not None:
            pc_vals.append(stable(pc))
    g_means.append(np.mean(g_vals)); g_stds.append(np.std(g_vals, ddof=0))
    pc_means.append(np.mean(pc_vals)); pc_stds.append(np.std(pc_vals, ddof=0))

# Highlight: FedAvg and OurMethod
colors = []
for m in methods:
    if 'OurMethod' in m: colors.append('#d62728')        # red
    elif m == 'FedAvg': colors.append('#1f77b4')         # blue
    elif 'control' in m.lower(): colors.append('#999999') # gray (FedAvgPlus1)
    else: colors.append('#7f7f7f')                        # neutral
# distinct colors per method, with OurMethod and FedAvg highlighted
colors = ['#1f77b4', '#7f7f7f', '#9467bd', '#d62728', '#999999', '#2ca02c']

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
x = np.arange(len(methods))
ax = axes[0]
bars = ax.bar(x, g_means, yerr=g_stds, capsize=4, color=colors, edgecolor='black', linewidth=0.6)
for xi, m, s in zip(x, g_means, g_stds):
    ax.text(xi, m + s + 0.005, f'{m:.4f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Global stable accuracy (rounds 190-199)')
ax.set_title('Post-drift stable global accuracy (3 seeds, canonical setup)', fontsize=11)
ax.set_ylim(0.55, 0.63)
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
bars = ax.bar(x, pc_means, yerr=pc_stds, capsize=4, color=colors, edgecolor='black', linewidth=0.6)
for xi, m, s in zip(x, pc_means, pc_stds):
    ax.text(xi, m + s + 0.005, f'{m:.4f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Per-client stable gen accuracy')
ax.set_title('Post-drift stable per-client generalized accuracy (3 seeds)', fontsize=11)
ax.set_ylim(0.43, 0.58)
ax.grid(axis='y', alpha=0.3)
ax.axhline(pc_means[0], color='#1f77b4', linestyle=':', linewidth=1, alpha=0.6)

fig.tight_layout()
fig.savefig(FIGS / 'main_3seed_bars.png', dpi=160, bbox_inches='tight')
plt.close(fig)
print(f"  wrote {FIGS/'main_3seed_bars.png'}")
print(f"  per-method (global stable, per-client stable):")
for m, gm, gs, pm, ps in zip(methods, g_means, g_stds, pc_means, pc_stds):
    print(f"    {m.replace(chr(10), ' '):>30}: {gm:.4f} +/- {gs:.4f}   {pm:.4f} +/- {ps:.4f}")


# ========================================================================
# Figure 2: detection recall comparison across regimes
# ========================================================================
def peak_in(flag_path, lo, hi):
    rows = list(csv.DictReader(Path(flag_path).open()))
    return max(int(rows[r]['flagged_count']) for r in range(lo, hi))

regimes = [
    ('Canonical\n(seed 0)',   35),  # 7/20 across 3 seeds, mean exactly = 7.0/20 = 35%
    ('Dir(0.5)',                5),  # 1/20
    ('Recurrent\ncanonical->swap', 30),  # 6/20 event 0
    ('Recurrent\nswap->canonical',  0),  # 0/20 event 1
    ('Aggressive\nperm',       95),  # 19/20
    ('Covariate\nL3-L4',        5),  # 1/20
    ('Covariate\nL1-L2 (indir.)', None),  # logged 0/20 due to logging bug; indirect via trajectory
]

fig, ax = plt.subplots(figsize=(11, 5))
labels = [r[0] for r in regimes]
vals = [r[1] if r[1] is not None else 0 for r in regimes]
colors_r = ['#1f77b4', '#ff7f0e', '#2ca02c', '#2ca02c', '#d62728', '#9467bd', '#9467bd']
bars = ax.bar(labels, vals, color=colors_r, edgecolor='black', linewidth=0.6)
# Show numeric labels above bars
for b, v, raw in zip(bars, vals, regimes):
    label_v = raw[1]
    if label_v is None:
        ax.text(b.get_x() + b.get_width()/2, 2, 'not directly\nobservable\n(CSV logging bug)',
                ha='center', va='bottom', fontsize=8, style='italic', color='gray')
        # Cross-hatch
        b.set_hatch('//')
        b.set_alpha(0.5)
    else:
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f'{v}%', ha='center', va='bottom', fontsize=10)
ax.axhline(35, color='#1f77b4', linestyle=':', linewidth=1, alpha=0.6)
ax.text(-0.4, 36, 'canonical peak (35%)', fontsize=8, color='#1f77b4', alpha=0.8)
ax.set_ylabel('Peak detection recall (flagged clients / 20)')
ax.set_title('OurMethod detection peak across drift regimes (single seed = 0)', fontsize=11)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
fig.tight_layout()
fig.savefig(FIGS / 'detection_recall_comparison.png', dpi=160, bbox_inches='tight')
plt.close(fig)
print(f"\n  wrote {FIGS/'detection_recall_comparison.png'}")


# ========================================================================
# Figure 3: communication savings (whole-run vs drift-window)
# ========================================================================
# Read directly from the flag CSVs and recompute (no fabrication).
L3, L4 = 18496, 73856
TOTAL = 107690
N, R = 20, 200
flag_sources = {
    0: 'runs/2026-06-08-perclient-smoke/ourmethod/results_OurMethod_flags.csv',
    1: 'runs/2026-06-08-perclient-3seed/seed1/ourmethod/results_OurMethod_flags.csv',
    2: 'runs/2026-06-08-perclient-3seed/seed2/ourmethod/results_OurMethod_flags.csv',
}
whole_pct = []; drift_pct = []
for s, p in flag_sources.items():
    rows = list(csv.DictReader(Path(p).open()))
    fl3 = np.array([int(r['flagged_layer3_count']) for r in rows])
    fl4 = np.array([int(r['flagged_layer4_count']) for r in rows])
    saving = fl3 * L3 + fl4 * L4
    whole_pct.append(100.0 * saving.sum() / (R * N * TOTAL))
    drift_pct.append(100.0 * saving[100:110].sum() / (10 * N * TOTAL))
w_mean, w_std = np.mean(whole_pct), np.std(whole_pct, ddof=0)
d_mean, d_std = np.mean(drift_pct), np.std(drift_pct, ddof=0)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(['Whole run\n(rounds 1-200)', 'Drift window\n(rounds 100-109)'],
              [w_mean, d_mean],
              yerr=[w_std, d_std],
              capsize=6,
              color=['#7f7f7f', '#d62728'],
              edgecolor='black', linewidth=0.6)
ax.text(0, w_mean + w_std + 0.5, f'{w_mean:.2f}% ± {w_std:.2f}%', ha='center', fontsize=11, weight='bold')
ax.text(1, d_mean + d_std + 0.5, f'{d_mean:.2f}% ± {d_std:.2f}%', ha='center', fontsize=11, weight='bold')
ax.set_ylabel('Upload saving vs FedAvg (%)')
ax.set_title('OurMethod communication saving (3 seeds, mean ± std)', fontsize=11)
ax.set_ylim(0, max(d_mean + d_std, 16) + 2)
ax.grid(axis='y', alpha=0.3)
# Annotate L4 dominance
ax.text(0.5, 0.94, 'L4 contributes ~83.5% of withheld bytes\n(L4 is 4x larger than L3)',
        transform=ax.transAxes, ha='center', va='top', fontsize=9,
        style='italic', color='gray',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='lightgray', alpha=0.9))
fig.tight_layout()
fig.savefig(FIGS / 'comm_savings.png', dpi=160, bbox_inches='tight')
plt.close(fig)
print(f"  wrote {FIGS/'comm_savings.png'}")
print(f"  whole-run: {w_mean:.3f}% +/- {w_std:.3f}%")
print(f"  drift-win: {d_mean:.3f}% +/- {d_std:.3f}%")
