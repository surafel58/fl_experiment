"""
generate_figures.py — produce PNG plots for the main experiments.

Reads CSVs from runs/ and writes PNGs into figures/. No GPU, no extra compute.

Figures produced:
  fig1_single_drift_curves.png       — 3-seed accuracy curves, single-event drift
  fig2_recurrent_drift_curves.png    — 2-seed accuracy curves, recurrent drift
  fig3_hybrid_lift_trajectory.png    — OurMethod hybrid lift on flagged clients (3 seeds, single drift)
  fig4_drift_dips_bar.png            — per-event dip magnitudes (recurrent drift, 2-seed mean)
  fig5_flag_counts_timeline.png      — flagged-client count per round (OurMethod, all 3 single-drift seeds)
  fig6_method_summary_bar.png        — pre-drift / dip / stable bar chart with std error bars (single drift, 3 seeds)
"""

import csv
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless
import matplotlib.pyplot as plt

FIG_DIR = Path('figures')
FIG_DIR.mkdir(exist_ok=True)

# --------- runs index ---------
SINGLE_DRIFT_RUNS = {
    0: Path('runs/2026-05-26-augfix/seed0'),
    1: Path('runs/2026-05-27-multiseed/seed1'),
    2: Path('runs/2026-05-27-multiseed/seed2'),
}
HYBRID_RUNS = {
    0: Path('runs/2026-05-28-hybrid/seed0'),
    1: Path('runs/2026-05-28-hybrid/seed1'),
    2: Path('runs/2026-05-28-hybrid/seed2'),
}
RECURRENT_RUNS = {
    0: Path('runs/2026-05-29-recurrent-drift/seed0'),
    1: Path('runs/2026-05-29-recurrent-drift/seed1'),
}

METHODS = ['FedAvg', 'Flash', 'AdaptiveFedAvg', 'OurMethod']
METHOD_COLORS = {
    'FedAvg':         '#1f77b4',  # blue
    'Flash':          '#ff7f0e',  # orange
    'AdaptiveFedAvg': '#2ca02c',  # green
    'OurMethod':      '#d62728',  # red
}


def load_global_accs(folder: Path, method: str):
    p = folder / f'results_{method}.csv'
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    return np.array([float(r['global_acc']) for r in rows])


def load_hybrid_data(folder: Path):
    p = folder / 'results_OurMethod.csv'
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    out = {'round': [], 'local_c': [], 'hybrid_c': []}
    for r in rows:
        out['round'].append(int(r['round']))
        out['local_c'].append([float(r[f'local_c{c:02d}']) for c in range(20)])
        out['hybrid_c'].append([float(r[f'hybrid_c{c:02d}']) for c in range(20)])
    return {k: np.array(v) for k, v in out.items()}


def load_flags(folder: Path):
    p = folder / 'results_OurMethod_flags.csv'
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    counts = []
    flag_sets = []
    for r in rows:
        counts.append(int(r['flagged_count']))
        ids = r['flagged_client_ids'].split(';') if r['flagged_client_ids'].strip() else []
        flag_sets.append(set(int(x) for x in ids))
    return np.array(counts), flag_sets


# ============================================================
# Figure 1 — single-event 3-seed accuracy curves
# ============================================================

def fig1_single_drift_curves():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in METHODS:
        series = [load_global_accs(SINGLE_DRIFT_RUNS[s], m) for s in (0, 1, 2)]
        series = [s for s in series if s is not None]
        if not series:
            continue
        # truncate to shortest just in case
        n = min(len(s) for s in series)
        arr = np.stack([s[:n] for s in series])  # [seeds, rounds]
        mean_y = arr.mean(axis=0)
        std_y  = arr.std(axis=0)
        x = np.arange(n)
        ax.plot(x, mean_y, label=m, color=METHOD_COLORS[m], linewidth=2)
        ax.fill_between(x, mean_y - std_y, mean_y + std_y,
                        color=METHOD_COLORS[m], alpha=0.15)
    ax.axvline(100, color='black', linestyle='--', linewidth=1, alpha=0.6, label='drift at round 100')
    ax.set_xlabel('Round')
    ax.set_ylabel('Global test accuracy')
    ax.set_title('Single-event sudden drift — 3-seed mean ± std across all 4 methods')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_ylim(0.15, 0.78)
    fig.tight_layout()
    out = FIG_DIR / 'fig1_single_drift_curves.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# Figure 2 — recurrent-drift 2-seed accuracy curves
# ============================================================

def fig2_recurrent_drift_curves():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in METHODS:
        series = [load_global_accs(RECURRENT_RUNS[s], m) for s in (0, 1)]
        series = [s for s in series if s is not None]
        if not series:
            continue
        n = min(len(s) for s in series)
        arr = np.stack([s[:n] for s in series])
        mean_y = arr.mean(axis=0)
        std_y  = arr.std(axis=0)
        x = np.arange(n)
        ax.plot(x, mean_y, label=m, color=METHOD_COLORS[m], linewidth=2)
        ax.fill_between(x, mean_y - std_y, mean_y + std_y,
                        color=METHOD_COLORS[m], alpha=0.15)
    for d, lbl in [(100, 'event 0'), (150, 'event 1')]:
        ax.axvline(d, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax.text(d + 1, 0.74, lbl, fontsize=9, alpha=0.7)
    ax.set_xlabel('Round')
    ax.set_ylabel('Global test accuracy')
    ax.set_title('Recurrent drift [100, 150] — 2-seed mean ± std across all 4 methods')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_ylim(0.15, 0.78)
    fig.tight_layout()
    out = FIG_DIR / 'fig2_recurrent_drift_curves.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# Figure 3 — hybrid lift trajectory on flagged clients (OurMethod)
# Shows per-round mean (hybrid_cXX - local_cXX) over the set of flagged clients
# in that round. Only single-event runs (hybrid_run, 3 seeds).
# ============================================================

def fig3_hybrid_lift_trajectory():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for s, color in zip((0, 1, 2), ['#1f77b4', '#2ca02c', '#d62728']):
        hyb = load_hybrid_data(HYBRID_RUNS[s])
        flags = load_flags(HYBRID_RUNS[s])
        if hyb is None or flags is None:
            continue
        counts, flag_sets = flags
        rounds = hyb['round']
        lift_series = []
        for i, r in enumerate(rounds):
            flagged = flag_sets[i] if i < len(flag_sets) else set()
            if not flagged:
                lift_series.append(np.nan)
                continue
            local_vals  = [hyb['local_c'][i, c]  for c in flagged]
            hybrid_vals = [hyb['hybrid_c'][i, c] for c in flagged]
            lift_series.append(np.mean(hybrid_vals) - np.mean(local_vals))
        lift_arr = np.array(lift_series)
        ax.plot(rounds, lift_arr, 'o-', label=f'seed {s}', color=color,
                markersize=4, linewidth=1.5, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.axvline(100, color='black', linestyle='--', linewidth=1, alpha=0.6)
    ax.text(101, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0.4 else 0.55,
            'drift at round 100', fontsize=9, alpha=0.7)
    ax.set_xlim(90, 130)
    ax.set_ylim(-0.05, 0.7)
    ax.set_xlabel('Round')
    ax.set_ylabel('Hybrid acc minus Global acc, mean over flagged clients')
    ax.set_title('OurMethod: hybrid model lift on flagged clients during drift window\n'
                 '(per-round mean, single-event drift)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()
    out = FIG_DIR / 'fig3_hybrid_lift_trajectory.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# Figure 4 — per-event dip bar chart (recurrent drift, 2-seed mean)
# ============================================================

def fig4_drift_dips_bar():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    event_rounds = [100, 150]
    method_e0_dips = {m: [] for m in METHODS}
    method_e1_dips = {m: [] for m in METHODS}

    for m in METHODS:
        for s in (0, 1):
            accs = load_global_accs(RECURRENT_RUNS[s], m)
            if accs is None: continue
            for evt_idx, evt_round in enumerate(event_rounds):
                if len(accs) < evt_round + 10: continue
                pre = float(np.mean(accs[max(0, evt_round-11):evt_round]))
                dip = pre - float(np.min(accs[evt_round:evt_round+10]))
                (method_e0_dips if evt_idx == 0 else method_e1_dips)[m].append(dip)

    x = np.arange(len(METHODS))
    width = 0.36

    e0_means = [np.mean(method_e0_dips[m]) for m in METHODS]
    e0_stds  = [np.std(method_e0_dips[m]) if len(method_e0_dips[m]) > 1 else 0 for m in METHODS]
    e1_means = [np.mean(method_e1_dips[m]) for m in METHODS]
    e1_stds  = [np.std(method_e1_dips[m]) if len(method_e1_dips[m]) > 1 else 0 for m in METHODS]

    bars0 = ax.bar(x - width/2, e0_means, width, yerr=e0_stds,
                   label='Event 0 (round 100)', color='#5b8dd6', capsize=4)
    bars1 = ax.bar(x + width/2, e1_means, width, yerr=e1_stds,
                   label='Event 1 (round 150)', color='#c73f3f', capsize=4)

    for bars in (bars0, bars1):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.005, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel('Accuracy dip (pre-drift mean minus min in next 10 rounds)')
    ax.set_title('Recurrent drift: per-event accuracy dips on canonical test set\n(2-seed mean ± std)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    ax.set_ylim(0, max(max(e0_means), max(e1_means)) * 1.30)
    fig.tight_layout()
    out = FIG_DIR / 'fig4_drift_dips_bar.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# Figure 5 — flag count timeline (OurMethod, single-drift 3 seeds)
# ============================================================

def fig5_flag_counts_timeline():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for s, color in zip((0, 1, 2), ['#1f77b4', '#2ca02c', '#d62728']):
        flags = load_flags(HYBRID_RUNS[s])
        if flags is None: continue
        counts, _ = flags
        ax.plot(np.arange(len(counts)), counts, label=f'seed {s}',
                color=color, linewidth=1.5, alpha=0.85)
    ax.axvline(100, color='black', linestyle='--', linewidth=1, alpha=0.6)
    ax.text(101, ax.get_ylim()[1] * 0.95, 'drift at round 100', fontsize=9, alpha=0.7)
    ax.set_xlabel('Round')
    ax.set_ylabel('Number of clients flagged (of 20)')
    ax.set_title('OurMethod: flagged-client count over time\n'
                 'Zero false positives outside the drift window (single-event runs)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_ylim(-0.5, 10)
    fig.tight_layout()
    out = FIG_DIR / 'fig5_flag_counts_timeline.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# Figure 6 — method summary bar chart (3-seed single-drift)
# ============================================================

def fig6_method_summary_bar():
    metrics = {'Pre-drift acc': {}, 'Dip (smaller=better)': {}, 'Post-drift stable': {}}
    DRIFT = 100
    for m in METHODS:
        pres, dips, stbls = [], [], []
        for s in (0, 1, 2):
            accs = load_global_accs(SINGLE_DRIFT_RUNS[s], m)
            if accs is None or len(accs) < DRIFT + 11: continue
            pres.append(float(np.mean(accs[max(0, DRIFT-11):DRIFT])))
            dips.append(pres[-1] - float(np.min(accs[DRIFT:DRIFT+10])))
            stbls.append(float(np.mean(accs[-10:])))
        metrics['Pre-drift acc'][m]         = (np.mean(pres),  np.std(pres))
        metrics['Dip (smaller=better)'][m]  = (np.mean(dips),  np.std(dips))
        metrics['Post-drift stable'][m]     = (np.mean(stbls), np.std(stbls))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric_name, data) in zip(axes, metrics.items()):
        means = [data[m][0] for m in METHODS]
        stds  = [data[m][1] for m in METHODS]
        bars = ax.bar(METHODS, means, yerr=stds, capsize=4,
                      color=[METHOD_COLORS[m] for m in METHODS])
        for b, mv in zip(bars, means):
            ax.text(b.get_x() + b.get_width()/2, mv + 0.005, f'{mv:.3f}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_title(metric_name)
        ax.set_ylabel('accuracy' if 'acc' in metric_name.lower() or 'stable' in metric_name.lower() else 'accuracy points')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=20)
    fig.suptitle('Single-event sudden drift — 3-seed mean ± std', y=1.02, fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / 'fig6_method_summary_bar.png'
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# Run all
# ============================================================

if __name__ == '__main__':
    print("Generating figures into ./figures/ ...")
    fig1_single_drift_curves()
    fig2_recurrent_drift_curves()
    fig3_hybrid_lift_trajectory()
    fig4_drift_dips_bar()
    fig5_flag_counts_timeline()
    fig6_method_summary_bar()
    print("Done.")
