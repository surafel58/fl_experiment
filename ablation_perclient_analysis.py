"""
ablation_perclient_analysis.py — secondary per-client lens for the 2026-06-04
ablation study, computed from the existing CSVs only (no re-runs).

Important: this is NOT the FedCCFA per-client generalized-accuracy metric. The
FedCCFA metric evaluates the global model on the FULL test set with each
cohort's label swap applied; our CSVs only contain per-client accuracy on each
client's own TRAINING partition under its training-label swap. Different eval
set, different swap target. See SUMMARY_perclient.md for the full comparison.

This script reports the existing `local_cXX` and `hybrid_cXX` columns as a
secondary lens, explicitly labeled.
"""

import csv
from pathlib import Path

import numpy as np

ROOT = Path('runs/2026-06-04-ablations')
VARIANTS = ['baseline', 'no-detection', 'all-layers', 'tau-low', 'tau-high']
NUM_CLIENTS = 20
LAST_K = 10  # average over last 10 rounds, matches summarize_method's stable window
DRIFT_ROUND = 100


def load_rows(variant: str):
    p = ROOT / variant / 'results_OurMethod.csv'
    return list(csv.DictReader(p.open()))


def stable_mean_of(rows, col):
    vals = [float(r[col]) for r in rows[-LAST_K:]]
    return float(np.mean(vals))


def post_drift_mean_of(rows, col):
    """Mean over all post-drift rounds [DRIFT_ROUND, end]."""
    vals = [float(r[col]) for r in rows[DRIFT_ROUND:]]
    return float(np.mean(vals))


def variant_metrics(variant: str):
    rows = load_rows(variant)

    # global acc (canonical test set, NOT drifted — what summarize_method reports)
    global_stable = stable_mean_of(rows, 'global_acc')

    # secondary lens 1: per-client local-data accuracy (global model on each
    # client's training partition with their label swap). Mean across last-10
    # rounds AND all 20 clients.
    local_stable_per_client = [stable_mean_of(rows, f'local_c{cid:02d}')
                               for cid in range(NUM_CLIENTS)]
    local_stable = float(np.mean(local_stable_per_client))

    # secondary lens 2: per-client hybrid accuracy (per-client view of the
    # model each client actually uses — for OurMethod this folds in the per-
    # client selective-layer choices). Mean across last-10 rounds AND all 20
    # clients.
    hybrid_stable_per_client = [stable_mean_of(rows, f'hybrid_c{cid:02d}')
                                for cid in range(NUM_CLIENTS)]
    hybrid_stable = float(np.mean(hybrid_stable_per_client))

    return {
        'global_stable': global_stable,
        'local_stable': local_stable,
        'hybrid_stable': hybrid_stable,
        'local_per_client': local_stable_per_client,
        'hybrid_per_client': hybrid_stable_per_client,
    }


if __name__ == '__main__':
    data = {v: variant_metrics(v) for v in VARIANTS}

    print(f"{'variant':<14}{'global':>10}{'local-mean':>12}{'hybrid-mean':>13}")
    print('-' * 49)
    for v in VARIANTS:
        m = data[v]
        print(f"{v:<14}{m['global_stable']:>10.4f}"
              f"{m['local_stable']:>12.4f}{m['hybrid_stable']:>13.4f}")

    base = data['baseline']
    print()
    print("Deltas vs baseline (variant - baseline), expressed in pp:")
    print(f"{'variant':<14}{'global':>10}{'local-mean':>12}{'hybrid-mean':>13}")
    print('-' * 49)
    for v in VARIANTS:
        if v == 'baseline':
            continue
        m = data[v]
        dg = (m['global_stable'] - base['global_stable']) * 100
        dl = (m['local_stable'] - base['local_stable']) * 100
        dh = (m['hybrid_stable'] - base['hybrid_stable']) * 100
        print(f"{v:<14}{dg:>+9.2f}p{dl:>+11.2f}p{dh:>+12.2f}p")
