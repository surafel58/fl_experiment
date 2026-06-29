"""saile_gate_analysis.py - compare FedAvg vs Saile-at-best (lr=0.01) across
regimes for the Saile-instability headroom gate.

Reads:
  Canonical Dir(0.1) sudden (seed 0):
    runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv
    runs/2026-06-11-saile-3seed/seed0/results_Saile.csv
  Dir(0.5) sudden (seed 0):
    runs/2026-06-11-dir05-test/fedavg/results_FedAvg.csv
    runs/2026-06-24-saile-gate/dir05/results_Saile.csv
  Recurrent alternating (seed 0):
    runs/2026-06-13-recurrent-gain-test/fedavg/results_FedAvg.csv
    runs/2026-06-24-saile-gate/recurrent/results_Saile.csv
  Aggressive permutation (seed 0):
    SCRATCH/saile_gate/aggressive/FedAvg.csv  (from aggressive-concept-drift-test branch)
    runs/2026-06-24-saile-gate/aggressive/results_Saile.csv
  Covariate drift L3/L4 (seed 0):
    SCRATCH/saile_gate/covariate/FedAvg.csv  (from covariate-drift-test branch)
    runs/2026-06-24-saile-gate/covariate/results_Saile.csv

Reports per regime:
  global_acc pre / dip / stable for both methods
  per_client_gen_acc pre / dip / stable for both methods
  delta Saile - FedAvg on both stable metrics (+ on dip if drift recurrent)
  Saile LR trajectory around drift event(s): min/mean/max client_lr
"""

import csv
import numpy as np
from pathlib import Path

SCRATCH = Path(r'C:\Users\suraf\AppData\Local\Temp\claude\d--projects-Seminars-in-AI-Paper-Proposal-Second-Semester-Implementation-FedCCFA\27e522cf-d3fe-44e4-946b-eff40f6a5a6d\scratchpad')

REGIMES = [
    ('Canonical Dir(0.1) sudden', 100, [100],
     'runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv',
     'runs/2026-06-11-saile-3seed/seed0/results_Saile.csv'),
    ('Dir(0.5) sudden', 100, [100],
     'runs/2026-06-11-dir05-test/fedavg/results_FedAvg.csv',
     'runs/2026-06-24-saile-gate/dir05/results_Saile.csv'),
    ('Recurrent alternating', 40, [40, 80, 120, 160],
     'runs/2026-06-13-recurrent-gain-test/fedavg/results_FedAvg.csv',
     'runs/2026-06-24-saile-gate/recurrent/results_Saile.csv'),
    ('Aggressive permutation', 100, [100],
     str(SCRATCH / 'saile_gate/aggressive/FedAvg.csv'),
     'runs/2026-06-24-saile-gate/aggressive/results_Saile.csv'),
    ('Covariate drift', 100, [100],
     str(SCRATCH / 'saile_gate/covariate/FedAvg.csv'),
     'runs/2026-06-24-saile-gate/covariate/results_Saile.csv'),
]


def load(p):
    if not Path(p).exists():
        return None
    rows = list(csv.DictReader(Path(p).open()))
    g  = np.array([float(r['global_acc']) for r in rows])
    pc = np.array([float(r.get('per_client_gen_acc') or 'nan') for r in rows]) if 'per_client_gen_acc' in rows[0] else None
    return rows, g, pc


def windows(s, first_drift):
    if s is None: return (None, None, None)
    pre    = float(np.mean(s[max(0, first_drift-11):first_drift]))
    dip    = pre - float(np.min(s[first_drift:first_drift+10]))
    stable = float(np.mean(s[-10:]))
    return pre, dip, stable


def lr_around_event(rows, event_round, span=5):
    """Return list of (round, base_lr, min, mean, max) around event_round."""
    out = []
    for r in rows:
        rnd = int(r['round'])
        if event_round - span <= rnd <= event_round + span:
            out.append((
                rnd,
                float(r.get('base_lr') or 'nan'),
                float(r.get('min_client_lr') or 'nan'),
                float(r.get('mean_client_lr') or 'nan'),
                float(r.get('max_client_lr') or 'nan'),
            ))
    return out


print("=" * 100)
print("SAILE-INSTABILITY HEADROOM GATE — single-seed regime sweep")
print("(Saile @ lr=0.01, best stable from prior canonical sweep)")
print("=" * 100)

for name, first_drift, drift_events, fa_path, sa_path in REGIMES:
    print(f"\n>>> {name}")
    fa = load(fa_path)
    sa = load(sa_path)
    if fa is None:
        print(f"    MISSING FedAvg: {fa_path}")
        continue
    if sa is None:
        print(f"    MISSING Saile: {sa_path}")
        continue
    _, fa_g, fa_pc = fa
    sa_rows, sa_g, sa_pc = sa

    fa_gw  = windows(fa_g,  first_drift)
    sa_gw  = windows(sa_g,  first_drift)
    fa_pcw = windows(fa_pc, first_drift) if fa_pc is not None else (None, None, None)
    sa_pcw = windows(sa_pc, first_drift) if sa_pc is not None else (None, None, None)

    print(f"    global:    FedAvg pre={fa_gw[0]:.4f} dip={fa_gw[1]:.4f} stable={fa_gw[2]:.4f}")
    print(f"               Saile  pre={sa_gw[0]:.4f} dip={sa_gw[1]:.4f} stable={sa_gw[2]:.4f}")
    if fa_pc is not None and sa_pc is not None:
        print(f"    per-cli:   FedAvg pre={fa_pcw[0]:.4f} dip={fa_pcw[1]:.4f} stable={fa_pcw[2]:.4f}")
        print(f"               Saile  pre={sa_pcw[0]:.4f} dip={sa_pcw[1]:.4f} stable={sa_pcw[2]:.4f}")
    d_g_stable  = (sa_gw[2] - fa_gw[2]) * 100
    d_pc_stable = (sa_pcw[2] - fa_pcw[2]) * 100 if (fa_pc is not None and sa_pc is not None) else None
    d_g_dip     = (sa_gw[1] - fa_gw[1]) * 100
    print(f"    delta Saile-FedAvg stable: global {d_g_stable:+.2f}pp   per-cli {d_pc_stable:+.2f}pp" if d_pc_stable is not None else f"    delta Saile-FedAvg stable: global {d_g_stable:+.2f}pp")
    print(f"    delta Saile-FedAvg dip:    global {d_g_dip:+.2f}pp  (positive = Saile DIPS DEEPER, worse)")

    # LR trajectory around drift events
    if 'base_lr' in sa_rows[0]:
        for ev in drift_events:
            tr = lr_around_event(sa_rows, ev, span=4)
            print(f"    Saile LR around event @round {ev}:")
            print(f"      round   base_lr   min       mean      max")
            for rnd, base, mn, me, mx in tr:
                marker = "  *" if rnd == ev else ""
                print(f"      {rnd:>5}  {base:.5f}  {mn:.5f}  {me:.5f}  {mx:.5f}{marker}")

print()
print("=" * 100)
print("Headline: regimes where Saile clearly underperforms FedAvg are candidates")
print("for the multi-seed confirm + robustification probe.")
print("=" * 100)
