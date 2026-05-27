"""
drift_window_analysis.py — re-analyze the 3-seed data with metrics that
actually exercise the adaptation mechanism.

The earlier "post-drift stable" metric averages rounds 190-199 — that's
~75 rounds AFTER the last flag clears in our runs, so the adaptation
window cannot show up in that number. This script measures during the
adaptation window itself (rounds 100-115).

Metrics computed:
  1. Drift-window mean global acc   (rounds 100-115)
  2. Drift-window minimum global acc (the trough)
  3. Recovery-window area-under-curve (rounds 100-130)
  4. Time to recover to 95% of pre-drift baseline
  5. Per-flagged-client local acc during the flag window (seeds 1, 2 only)
  6. Per-drift-group local acc during the flag window (seeds 1, 2 only)

No new compute — pure analysis of existing CSVs.
"""

import csv
from pathlib import Path
from statistics import mean, stdev

NUM_CLIENTS  = 20
NUM_ROUNDS   = 200
DRIFT_ROUND  = 100
WINDOW_END   = 115         # flag window typically clears here in our runs
RECOVERY_END = 130         # post-drift recovery period
METHODS      = ['FedAvg', 'Flash', 'AdaptiveFedAvg', 'OurMethod']
SEEDS_GLOBAL    = [0, 1, 2]    # all seeds have global_acc
SEEDS_PERCLIENT = [1, 2]       # only these have local_cXX columns

GROUP_A = [i for i in range(NUM_CLIENTS) if i % 10 < 3]
GROUP_B = [i for i in range(NUM_CLIENTS) if 3 <= i % 10 < 6]
GROUP_C = [i for i in range(NUM_CLIENTS) if i % 10 >= 6]

RUN_PATHS = {
    0: Path('runs/2026-05-26-augfix/seed0'),
    1: Path('runs/2026-05-27-multiseed/seed1'),
    2: Path('runs/2026-05-27-multiseed/seed2'),
}


def load_rows(method, seed):
    path = RUN_PATHS[seed] / f'results_{method}.csv'
    if not path.exists(): return None
    return list(csv.DictReader(path.open()))


def load_flags(seed):
    path = RUN_PATHS[seed] / 'results_OurMethod_flags.csv'
    if not path.exists(): return None
    return list(csv.DictReader(path.open()))


# ============================================================
# Metric 1-4: drift window global acc
# ============================================================

def global_acc_metrics(rows):
    """Returns: pre, dip_min, win_mean (100-115), rec_auc (100-130), t95"""
    accs = [float(r['global_acc']) for r in rows]
    pre = mean(accs[DRIFT_ROUND-11:DRIFT_ROUND])
    dip_min = min(accs[DRIFT_ROUND:WINDOW_END+1])
    win_mean = mean(accs[DRIFT_ROUND:WINDOW_END+1])
    rec_auc = sum(accs[DRIFT_ROUND:RECOVERY_END+1]) / (RECOVERY_END - DRIFT_ROUND + 1)
    # Time to recover to 95% of pre-drift
    target = 0.95 * pre
    t95 = None
    for i in range(DRIFT_ROUND, len(accs)):
        if accs[i] >= target:
            t95 = i - DRIFT_ROUND
            break
    return pre, dip_min, win_mean, rec_auc, t95


def section_global():
    print("="*100)
    print("METRIC 1-4: GLOBAL ACCURACY DURING DRIFT WINDOW")
    print("="*100)
    print(f"  Drift round: {DRIFT_ROUND} | Flag window end: ~{WINDOW_END} "
          f"| Recovery end: {RECOVERY_END}")
    print()

    # Collect per-method per-seed metrics
    data = {m: [] for m in METHODS}
    for m in METHODS:
        for s in SEEDS_GLOBAL:
            rows = load_rows(m, s)
            if rows: data[m].append(global_acc_metrics(rows))

    print(f"  {'Method':<18}{'pre-drift':>13}{'min (100-115)':>16}"
          f"{'mean (100-115)':>17}{'auc (100-130)':>16}{'rounds to 95%':>16}")
    print("  " + "-"*94)
    for m in METHODS:
        if not data[m]: continue
        pres = [d[0] for d in data[m]]
        mins = [d[1] for d in data[m]]
        means = [d[2] for d in data[m]]
        aucs = [d[3] for d in data[m]]
        t95s = [d[4] if d[4] is not None else NUM_ROUNDS - DRIFT_ROUND for d in data[m]]
        def f(xs):
            return f"{mean(xs):.4f}" if len(xs)==1 else f"{mean(xs):.4f}+/-{stdev(xs):.3f}"
        print(f"  {m:<18}{f(pres):>13}{f(mins):>16}{f(means):>17}"
              f"{f(aucs):>16}{f([float(t) for t in t95s]):>16}")
    print()

    # Pairwise deltas OurMethod vs baselines
    print("  PAIRWISE DELTAS (OurMethod - baseline, mean over seeds):")
    print(f"  {'vs':<18}{'pre Δ':>10}{'min Δ':>10}{'mean Δ':>10}{'auc Δ':>10}{'t95 Δ':>10}")
    print("  " + "-"*68)
    if data['OurMethod']:
        om_metrics = data['OurMethod']
        for baseline in ['FedAvg', 'Flash', 'AdaptiveFedAvg']:
            if not data[baseline]: continue
            bm = data[baseline]
            d_pre  = mean([o[0]-b[0] for o,b in zip(om_metrics, bm)])
            d_min  = mean([o[1]-b[1] for o,b in zip(om_metrics, bm)])
            d_mean = mean([o[2]-b[2] for o,b in zip(om_metrics, bm)])
            d_auc  = mean([o[3]-b[3] for o,b in zip(om_metrics, bm)])
            d_t95  = mean([(o[4] if o[4] else NUM_ROUNDS-DRIFT_ROUND) -
                           (b[4] if b[4] else NUM_ROUNDS-DRIFT_ROUND)
                           for o,b in zip(om_metrics, bm)])
            print(f"  vs {baseline:<15}{d_pre:>+10.4f}{d_min:>+10.4f}"
                  f"{d_mean:>+10.4f}{d_auc:>+10.4f}{d_t95:>+10.1f}")
    print()
    print("  Reading: for pre/min/mean/auc, POSITIVE delta = OurMethod better.")
    print("           for t95 (rounds to recover), NEGATIVE delta = OurMethod faster.")
    print()


# ============================================================
# Metric 5: per-flagged-client local acc during flag window
# ============================================================

def section_flagged_clients():
    print("="*100)
    print("METRIC 5: PER-FLAGGED-CLIENT LOCAL ACC DURING FLAG WINDOW (seeds 1+2)")
    print("="*100)
    print(f"  Window: rounds {DRIFT_ROUND}-{WINDOW_END} inclusive")
    print(f"  Compares OurMethod adaptation vs FedAvg passive averaging for the")
    print(f"  same clients that OurMethod flagged.")
    print()

    # Determine flagged-ever set per seed
    flagged_per_seed = {}
    for s in SEEDS_PERCLIENT:
        frows = load_flags(s)
        if not frows: continue
        flagged = set()
        for r in frows:
            for col in ('flagged_layer3_ids', 'flagged_layer4_ids'):
                if r[col].strip():
                    flagged.update(int(x) for x in r[col].split(';'))
        flagged_per_seed[s] = sorted(flagged)
        print(f"  Seed {s} flagged-ever clients: {flagged_per_seed[s]}  ({len(flagged_per_seed[s])} clients)")
    print()

    # Per-method mean local acc over the flagged subset over the window
    print(f"  {'Method':<14}{'Seed':<6}{'Flagged clients win-mean local acc':>40}")
    print("  " + "-"*60)
    cross_methods = {m: {} for m in METHODS}
    for m in METHODS:
        for s in SEEDS_PERCLIENT:
            if s not in flagged_per_seed: continue
            rows = load_rows(m, s)
            if not rows: continue
            flagged = flagged_per_seed[s]
            window_rows = rows[DRIFT_ROUND:WINDOW_END+1]
            # For each round in window, compute mean local acc over flagged clients
            per_round_means = []
            for r in window_rows:
                vals = [float(r[f'local_c{cid:02d}']) for cid in flagged]
                per_round_means.append(mean(vals))
            win_mean = mean(per_round_means)
            cross_methods[m][s] = win_mean
            print(f"  {m:<14}{s:<6}{win_mean:>40.4f}")
    print()

    # OurMethod vs FedAvg delta on the flagged subset, per seed and averaged
    print("  PAIRWISE DELTA on flagged clients' local acc (OurMethod - FedAvg):")
    fa_vals, om_vals, deltas = [], [], []
    for s in SEEDS_PERCLIENT:
        if s in cross_methods['FedAvg'] and s in cross_methods['OurMethod']:
            fa = cross_methods['FedAvg'][s]
            om = cross_methods['OurMethod'][s]
            fa_vals.append(fa); om_vals.append(om)
            deltas.append(om - fa)
            print(f"  seed={s}  FedAvg={fa:.4f}  OurMethod={om:.4f}  delta={om-fa:+.4f}")
    if deltas:
        print(f"  MEAN delta: {mean(deltas):+.4f} ({mean(deltas)*100:+.2f}pp)")
    print()


# ============================================================
# Metric 6: per-drift-group local acc during flag window
# ============================================================

def section_drift_groups():
    print("="*100)
    print("METRIC 6: PER-DRIFT-GROUP LOCAL ACC DURING FLAG WINDOW (seeds 1+2)")
    print("="*100)
    print(f"  Window: rounds {DRIFT_ROUND}-{WINDOW_END} inclusive")
    print()

    # For each method, per drift group, mean local acc over window across seeds
    group_defs = [('A (1<->2)', GROUP_A), ('B (3<->4)', GROUP_B), ('C (5<->6)', GROUP_C)]

    print(f"  {'Method':<18}{'Group A win-mean':>20}{'Group B win-mean':>20}{'Group C win-mean':>20}")
    print("  " + "-"*78)

    data = {m: {} for m in METHODS}
    for m in METHODS:
        for label, grp in group_defs:
            all_means = []
            for s in SEEDS_PERCLIENT:
                rows = load_rows(m, s)
                if not rows: continue
                window_rows = rows[DRIFT_ROUND:WINDOW_END+1]
                for r in window_rows:
                    vals = [float(r[f'local_c{cid:02d}']) for cid in grp]
                    all_means.append(mean(vals))
            if all_means:
                data[m][label] = mean(all_means)

    for m in METHODS:
        row = f"  {m:<18}"
        for label, _ in group_defs:
            v = data[m].get(label)
            row += f"{v:>20.4f}" if v is not None else f"{'-':>20}"
        print(row)
    print()

    print("  PAIRWISE DELTAS OurMethod - FedAvg per group (in pp):")
    for label, _ in group_defs:
        o = data['OurMethod'].get(label)
        f = data['FedAvg'].get(label)
        if o is not None and f is not None:
            print(f"  Group {label}: {(o-f)*100:+.2f}pp")
    print()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import io, sys
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    section_global()
    section_flagged_clients()
    section_drift_groups()
    sys.stdout = old

    report = buf.getvalue()
    print(report)

    with open('drift_window_analysis_report.txt', 'w') as f:
        f.write(report)
    print("\nReport saved: drift_window_analysis_report.txt")
