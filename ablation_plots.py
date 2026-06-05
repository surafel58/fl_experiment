"""
ablation_plots.py — produce PNG plots for the 2026-06-04 OurMethod ablation study.

Reads the 5 ablation CSVs in runs/2026-06-04-ablations/<variant>/ and writes
PNGs into runs/2026-06-04-ablations/plots/. No GPU, no extra compute, no rerun.

Plots produced:
  ablation_trajectories.png   — global acc vs round, all 5 variants, drift line @ 100
  ablation_metrics_bars.png   — grouped bars: pre-drift, dip, post-drift stable, recovery
  ablation_deltas.png         — bars of delta-vs-baseline on pre-drift and post-drift-stable

Plot style matches generate_figures.py: matplotlib Agg, dpi=160, same figsize/grid
conventions, explicit per-variant colors stable across all three figures.

The metric formulas mirror summarize_method() in all_experiments_optimized.py
exactly, so the numbers in these plots match SUMMARY.md.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless
import matplotlib.pyplot as plt

# --------- paths ---------
ROOT = Path('runs/2026-06-04-ablations')
PLOTS_DIR = ROOT / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

# --------- variant order and colors (stable across all figures) ---------
VARIANTS = ['baseline', 'no-detection', 'all-layers', 'tau-low', 'tau-high']
VARIANT_LABEL = {
    'baseline':     'baseline (OurMethod, default)',
    'no-detection': 'no-detection (τ = ∞)',
    'all-layers':   'all-layers flaggable',
    'tau-low':      'tau-low (τ = 1.2)',
    'tau-high':     'tau-high (τ = 1.6)',
}
VARIANT_COLOR = {
    'baseline':     '#d62728',   # red (matches OurMethod color from prior figures)
    'no-detection': '#ff7f0e',   # orange
    'all-layers':   '#2ca02c',   # green
    'tau-low':      '#1f77b4',   # blue
    'tau-high':     '#9467bd',   # purple
}

# --------- metric formulas — same as summarize_method() ---------
DRIFT_ROUND = 100   # DRIFT_SCHEDULE[0]
MAX_ROUNDS  = 200
RECOVERY_CAP = MAX_ROUNDS - DRIFT_ROUND   # = 100 — value shown when "not recovered"


def load_accs(variant: str):
    p = ROOT / variant / 'results_OurMethod.csv'
    rows = list(csv.DictReader(p.open()))
    accs = np.array([float(r['global_acc']) for r in rows])
    return accs


def metrics_for(accs: np.ndarray):
    """Mirrors summarize_method in all_experiments_optimized.py."""
    if len(accs) < DRIFT_ROUND + 11:
        return None
    pre   = float(np.mean(accs[max(0, DRIFT_ROUND-11):DRIFT_ROUND]))
    dip   = pre - float(np.min(accs[DRIFT_ROUND:DRIFT_ROUND+10]))
    stable = float(np.mean(accs[-10:]))
    # recovery: first round >= drift_round where acc >= pre - 0.02
    rec = None
    for i in range(DRIFT_ROUND, len(accs)):
        if accs[i] >= pre - 0.02:
            rec = i - DRIFT_ROUND
            break
    return {
        'pre': pre,
        'dip': dip,
        'stable': stable,
        'recovery': rec,                                      # None means not recovered
        'recovery_display': rec if rec is not None else RECOVERY_CAP,
        'recovered': rec is not None,
    }


# ============================================================
# Load all 5 variants
# ============================================================

data = {}
for v in VARIANTS:
    accs = load_accs(v)
    m = metrics_for(accs)
    data[v] = {'accs': accs, 'metrics': m}

# --------- print metric table so we can cross-check vs SUMMARY.md ---------
print(f"{'variant':<14}{'pre':>10}{'dip':>10}{'stable':>10}{'recovery':>11}")
print('-' * 55)
for v in VARIANTS:
    m = data[v]['metrics']
    rec_str = f'{m["recovery"]}' if m['recovered'] else 'NR'
    print(f"{v:<14}{m['pre']:>10.4f}{m['dip']:>10.4f}{m['stable']:>10.4f}{rec_str:>11}")

# ============================================================
# Figure 1 — Accuracy trajectories
# ============================================================

def fig_trajectories():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for v in VARIANTS:
        accs = data[v]['accs']
        rounds = np.arange(len(accs))
        ax.plot(rounds, accs, color=VARIANT_COLOR[v], linewidth=2,
                label=VARIANT_LABEL[v])
    ax.axvline(DRIFT_ROUND, color='black', linestyle='--', linewidth=1, alpha=0.6,
               label=f'drift at round {DRIFT_ROUND}')
    ax.set_xlabel('Round')
    ax.set_ylabel('Global test accuracy')
    ax.set_title('OurMethod ablations — global accuracy vs round (seed 0)')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_ylim(0.15, 0.78)
    fig.tight_layout()
    out = PLOTS_DIR / 'ablation_trajectories.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# Figure 2 — Grouped bars: pre-drift, dip, post-drift stable, recovery rounds
# ============================================================

def fig_metrics_bars():
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    metric_keys = ['pre', 'dip', 'stable', 'recovery_display']
    metric_titles = ['Pre-drift acc', 'Accuracy dip (smaller=better)',
                     'Post-drift stable acc', 'Recovery rounds']
    metric_ylabels = ['accuracy', 'accuracy', 'accuracy', 'rounds since drift']

    for ax, key, title, ylabel in zip(axes, metric_keys, metric_titles, metric_ylabels):
        values = [data[v]['metrics'][key] for v in VARIANTS]
        colors = [VARIANT_COLOR[v] for v in VARIANTS]
        bars = ax.bar(VARIANTS, values, color=colors)
        ymax = max(values) if max(values) > 0 else 1.0
        for v, bar, val in zip(VARIANTS, bars, values):
            recovered = data[v]['metrics']['recovered']
            if key == 'recovery_display' and not recovered:
                # short label, placed INSIDE the bar so 5 NRs do not collide
                ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                        'NR', ha='center', va='center', fontsize=10,
                        color='white', fontweight='bold')
            elif key == 'recovery_display':
                ax.text(bar.get_x() + bar.get_width() / 2, val + ymax * 0.02,
                        f'{val:d}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, val + ymax * 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=20)
        # leave headroom for value labels
        ax.set_ylim(0, ymax * 1.18)
        if key == 'recovery_display':
            ax.axhline(RECOVERY_CAP, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
            # add a footnote explaining NR
            ax.text(0.5, -0.20, 'NR = not recovered to within 2pp of pre-drift by run end (cap = 100)',
                    transform=ax.transAxes, ha='center', va='top', fontsize=8, color='gray', style='italic')

    fig.suptitle('OurMethod ablations — derived metrics (seed 0)', y=1.02, fontsize=12)
    fig.tight_layout()
    out = PLOTS_DIR / 'ablation_metrics_bars.png'
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# Figure 3 — Deltas vs baseline on pre-drift and post-drift stable
# ============================================================

def fig_deltas():
    base = data['baseline']['metrics']
    others = [v for v in VARIANTS if v != 'baseline']

    delta_pre    = [data[v]['metrics']['pre']    - base['pre']    for v in others]
    delta_stable = [data[v]['metrics']['stable'] - base['stable'] for v in others]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(others))
    width = 0.36

    bars_pre = ax.bar(x - width/2, delta_pre, width,
                      color=[VARIANT_COLOR[v] for v in others],
                      label='Δ Pre-drift acc', edgecolor='black', linewidth=0.5)
    bars_stb = ax.bar(x + width/2, delta_stable, width,
                      color=[VARIANT_COLOR[v] for v in others],
                      label='Δ Post-drift stable', edgecolor='black',
                      linewidth=0.5, hatch='//', alpha=0.85)

    # Y-range with explicit padding so labels stay inside the plot box.
    all_vals = delta_pre + delta_stable
    ymin = min(min(all_vals), 0.0)
    ymax = max(max(all_vals), 0.0)
    span = ymax - ymin
    ax.set_ylim(ymin - span * 0.18, ymax + span * 0.20)

    pad = span * 0.04
    for bars, vals in ((bars_pre, delta_pre), (bars_stb, delta_stable)):
        for bar, val in zip(bars, vals):
            offset = pad if val >= 0 else -pad
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                    f'{val * 100:+.2f}pp', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=8)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABEL[v] for v in others], rotation=15, ha='right')
    ax.set_ylabel('Δ accuracy vs unmodified baseline (accuracy points)')
    ax.set_title('OurMethod ablations — deltas vs baseline (seed 0)\nPositive = variant beats baseline; negative = variant hurts')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper left')
    fig.tight_layout()
    out = PLOTS_DIR / 'ablation_deltas.png'
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# Run all
# ============================================================

if __name__ == '__main__':
    print(f"Generating ablation plots into {PLOTS_DIR}/ ...\n")
    fig_trajectories()
    fig_metrics_bars()
    fig_deltas()
    print("\nDone.")
