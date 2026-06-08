"""perclient_summary.py — build SUMMARY.md for the 5-method per-client smoke.

Inputs (one per method, each containing `results_<Method>.csv`):
  runs/2026-06-08-perclient-smoke/fedavg/
  runs/2026-06-08-perclient-smoke/flash/
  runs/2026-06-08-perclient-smoke/adaptive/
  runs/2026-06-08-perclient-smoke/ourmethod/
  runs/2026-06-08-perclient-smoke/fedavgplus1/

Reports per method:
  - global_acc:          pre-drift / dip / post-drift stable
  - per_client_gen_acc:  pre-drift / dip / post-drift stable
  - delta-per-client vs FedAvg (the key column)

Also reports the FedAvg vs FedAvgPlus1 global_acc trajectory divergence.
CUDA convs run with cudnn.benchmark=True are NOT bit-reproducible across
processes; expect small drift. We report the actual divergence so the reader
can decide if the control's global trajectory tracks plain FedAvg closely
enough for the per-client comparison to be meaningful.

Run from the project root after all 5 CSVs land:
    python3 perclient_summary.py
"""

import csv
from pathlib import Path

import numpy as np

ROOT = Path('runs/2026-06-08-perclient-smoke')
DRIFT_ROUND = 100
NUM_ROUNDS  = 200

METHODS = [
    ('FedAvg',         'fedavg/results_FedAvg.csv'),
    ('Flash',          'flash/results_Flash.csv'),
    ('AdaptiveFedAvg', 'adaptive/results_AdaptiveFedAvg.csv'),
    ('FedAvgPlus1',    'fedavgplus1/results_FedAvgPlus1.csv'),
    ('OurMethod',      'ourmethod/results_OurMethod.csv'),
]


def load_series(path: Path):
    rows = list(csv.DictReader(path.open()))
    global_acc = np.array([float(r['global_acc']) for r in rows])
    pc_acc = np.array([float(r.get('per_client_gen_acc') or 'nan') for r in rows])
    return global_acc, pc_acc


def pre_dip_stable(series):
    pre    = float(np.mean(series[max(0, DRIFT_ROUND-11):DRIFT_ROUND]))
    dip    = pre - float(np.min(series[DRIFT_ROUND:DRIFT_ROUND+10]))
    stable = float(np.mean(series[-10:]))
    return pre, dip, stable


def main():
    rows = {}
    for name, rel in METHODS:
        p = ROOT / rel
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        g, pc = load_series(p)
        rows[name] = {
            'global': pre_dip_stable(g),
            'pc':     pre_dip_stable(pc),
            'g_series': g,
            'pc_series': pc,
        }

    # FedAvg vs FedAvgPlus1 global trajectory divergence (cudnn.benchmark=True
    # means we expect small drift, NOT bit-identity; report it honestly).
    fa = rows.get('FedAvg', {}).get('g_series')
    fp = rows.get('FedAvgPlus1', {}).get('g_series')
    if fa is not None and fp is not None:
        diff = fp - fa
        gate = {
            'max_abs': float(np.max(np.abs(diff))),
            'mean_abs': float(np.mean(np.abs(diff))),
            'final_abs': float(abs(diff[-1])),
        }
    else:
        gate = None

    # Build the markdown.
    lines = []
    lines.append("# Per-client smoke — 5 methods, single seed (seed 0)\n")
    lines.append("**Setup:** CIFAR-10, Dir(0.1), 20 clients, 200 rounds, 5 local epochs, batch 64, single sudden drift at round 100. Per-client metric = FedCCFA's per-client generalized accuracy (validated at Step A — bit-exact pre-drift identity, see `validate_perclient_metric.py`).\n")
    lines.append("**Hardware:** GCP L4 (g2-standard-4), torch 2.9.1+cu129. `cudnn.benchmark = True` (convs non-deterministic across processes).\n")
    lines.append("**Methods:**\n")
    lines.append("- **FedAvg** — reference baseline (global model on cohort-swapped full test sets).")
    lines.append("- **Flash, AdaptiveFedAvg** — drift-aware baselines.")
    lines.append("- **FedAvgPlus1** — control: FedAvg + 1 local epoch on each client's own data immediately before per-client eval. Isolates trivial last-step personalization as a confound on the per-client metric.")
    lines.append("- **OurMethod** — drift-triggered selective layer adaptation.\n")

    lines.append("## Results\n")
    lines.append("All values are mean over the indicated window. `pre` = rounds [89, 99]; `dip` = pre − min(rounds [100, 109]); `stable` = mean(rounds [190, 199]).\n")

    lines.append("### Global accuracy (canonical CIFAR-10 test set, undrifted labels)\n")
    lines.append("| Method | Pre | Dip | Stable |")
    lines.append("|---|---:|---:|---:|")
    for name, _ in METHODS:
        if name not in rows: continue
        pre, dip, stable = rows[name]['global']
        lines.append(f"| {name} | {pre:.4f} | {dip:.4f} | {stable:.4f} |")
    lines.append("")

    lines.append("### Per-client generalized accuracy (FedCCFA protocol, cohort-swapped full test sets)\n")
    lines.append("| Method | Pre | Dip | Stable | Δ stable vs FedAvg |")
    lines.append("|---|---:|---:|---:|---:|")
    base_pc_stable = rows.get('FedAvg', {}).get('pc', (None, None, None))[2]
    for name, _ in METHODS:
        if name not in rows: continue
        pre, dip, stable = rows[name]['pc']
        delta = (stable - base_pc_stable) * 100 if base_pc_stable is not None else float('nan')
        delta_str = f"{delta:+.2f}pp" if not np.isnan(delta) else "—"
        if name == 'FedAvg':
            delta_str = '0 (ref)'
        lines.append(f"| {name} | {pre:.4f} | {dip:.4f} | {stable:.4f} | {delta_str} |")
    lines.append("")

    lines.append("## FedAvg vs FedAvgPlus1 — global-trajectory divergence gate\n")
    if gate:
        lines.append("With `cudnn.benchmark = True` (our standard config) and the same seed, two")
        lines.append("PyTorch processes running the SAME training loop will not produce bit-identical")
        lines.append("global accuracy because CUDA convolution kernels are non-deterministic across")
        lines.append("processes. The +1-epoch fine-tune in FedAvgPlus1 does NOT touch the global model")
        lines.append("(it operates on a per-client scratch clone), so the only difference between")
        lines.append("FedAvg and FedAvgPlus1's global trajectories should be CUDA non-determinism.\n")
        lines.append(f"- max |Δglobal| across 200 rounds: **{gate['max_abs']:.4f}** ({gate['max_abs']*100:.2f}pp)")
        lines.append(f"- mean |Δglobal|: **{gate['mean_abs']:.4f}** ({gate['mean_abs']*100:.2f}pp)")
        lines.append(f"- final-round |Δglobal|: **{gate['final_abs']:.4f}** ({gate['final_abs']*100:.2f}pp)\n")
        lines.append("If `max_abs` is comparable to single-seed noise (~0.5–1.5pp), the global")
        lines.append("trajectories are 'effectively the same model' and the per-client metric comparison")
        lines.append("between FedAvg and FedAvgPlus1 is interpretable as 'same global model, +1 personalization epoch'.")
    else:
        lines.append("(FedAvgPlus1 or FedAvg output not found — gate not computable.)")
    lines.append("")

    lines.append("## Caveats\n")
    lines.append("- **Single seed (seed 0).** Prior 3-seed studies measured 1-σ noise on stable post-drift accuracy at ~0.5–1.5pp. Treat sub-1pp deltas as noise.")
    lines.append("- **FedAvgPlus1 is a noise-floor control, not a competitive method.** Its per-client metric tells us how much per-client 'lift' is achievable by trivial last-step personalization with no drift mechanism at all. OurMethod's per-client metric must beat this control by a meaningful margin to claim the selective per-layer adaptation actually adds value beyond fine-tuning.")
    lines.append("- **Non-determinism caveat above.** The 'global model unchanged' control claim is approximate, bounded by the divergence numbers above.\n")

    out = ROOT / 'SUMMARY.md'
    out.write_text("\n".join(lines), encoding='utf-8')
    print(f"Wrote {out}")
    print()
    print("\n".join(lines))


if __name__ == '__main__':
    main()
