"""headroom_gate_analysis.py - does oracle drift-boost FedAvg beat both plain
FedAvg AND always-higher-LR FedAvg? Also: confirm FedAvgAdam (Flash action,
no trigger) sits below plain FedAvg, isolating the action ceiling.

Inputs (all canonical Dir(0.1) sudden drift @ rd 100, seed 0):
  Plain FedAvg:   runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv
  Oracle 2x/3x/5x:  runs/2026-06-25-headroom-gate/oracle_{2,3,5}x/results_FedAvg.csv
  Always 2x/3x/5x:  runs/2026-06-25-headroom-gate/always_{2,3,5}x/results_FedAvg.csv
  FedAvgAdam:       runs/2026-06-25-headroom-gate/fedavg_adam/results_FedAvgAdam.csv
"""
import csv
import numpy as np
from pathlib import Path


def load(p):
    if not Path(p).exists(): return None
    rows = list(csv.DictReader(Path(p).open()))
    return {
        'global': np.array([float(r['global_acc']) for r in rows]),
        'pc':     np.array([float(r.get('per_client_gen_acc') or 'nan') for r in rows])
                  if 'per_client_gen_acc' in rows[0] else None,
    }


def metrics(d, first_drift=100):
    if d is None: return None
    g = d['global']
    n_rounds = len(g)
    if n_rounds < first_drift + 5:
        return None
    pre   = float(np.mean(g[max(0, first_drift-11):first_drift]))
    post_end = min(first_drift+10, n_rounds)
    minp  = float(np.min(g[first_drift:post_end]))
    dip   = pre - minp
    rec = None
    target = pre - 0.02
    for r in range(first_drift, n_rounds):
        if g[r] >= target:
            rec = r - first_drift; break
    # stable: use last 10 if we have full 200, else use last 10 of what we have
    stable = float(np.mean(g[-10:]))
    pc_stable = float(np.mean(d['pc'][-10:])) if d['pc'] is not None else None
    return {'pre': pre, 'dip': dip, 'min_post': minp, 'rec': rec, 'stable': stable, 'pc_stable': pc_stable, 'n_rounds': n_rounds}


configs = [
    ('Plain FedAvg',         'runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv'),
    ('FedAvgAdam (no trigger)','runs/2026-06-25-headroom-gate/fedavg_adam/results_FedAvgAdam.csv'),
    ('Oracle boost 2x',      'runs/2026-06-25-headroom-gate/oracle_2x/results_FedAvg.csv'),
    ('Oracle boost 3x',      'runs/2026-06-25-headroom-gate/oracle_3x/results_FedAvg.csv'),
    ('Oracle boost 5x',      'runs/2026-06-25-headroom-gate/oracle_5x/results_FedAvg.csv'),
    ('Always-higher 2x (lr=0.02)','runs/2026-06-25-headroom-gate/always_2x/results_FedAvg.csv'),
    ('Always-higher 3x (lr=0.03)','runs/2026-06-25-headroom-gate/always_3x/results_FedAvg.csv'),
    ('Always-higher 5x (lr=0.05)','runs/2026-06-25-headroom-gate/always_5x/results_FedAvg.csv'),
]

print("=" * 110)
print("HEADROOM GATE - does oracle drift-boost beat BOTH plain FedAvg AND always-higher-LR?")
print("(canonical Dir(0.1) sudden drift @ rd 100, single seed)")
print("=" * 110)
print(f"\n{'Config':<32}{'pre':>9}{'dip':>9}{'min_post':>11}{'rec':>6}{'stable':>10}{'pc_stable':>12}")
print("-" * 110)

results = {}
for name, p in configs:
    d = load(p)
    if d is None:
        print(f"{name:<32}  MISSING: {p}")
        continue
    m = metrics(d)
    results[name] = m
    rec_str = str(m['rec']) if m['rec'] is not None else 'no'
    pc_str = f"{m['pc_stable']:.4f}" if m['pc_stable'] is not None else '-'
    n_str = f" [n={m['n_rounds']}]" if m['n_rounds'] < 200 else ''
    print(f"{name:<32}{m['pre']:>9.4f}{m['dip']:>9.4f}{m['min_post']:>11.4f}{rec_str:>6}{m['stable']:>10.4f}{pc_str:>12}{n_str}")

# Deltas
print("\n" + "=" * 110)
print("DELTAS")
print("=" * 110)
if 'Plain FedAvg' in results:
    plain = results['Plain FedAvg']
    print(f"\nPlain FedAvg: stable={plain['stable']:.4f}  dip={plain['dip']:.4f}  rec={plain['rec']}")
    print()
    for name, m in results.items():
        if name == 'Plain FedAvg': continue
        d_stable = (m['stable'] - plain['stable']) * 100
        d_dip    = (m['dip'] - plain['dip']) * 100
        d_pc     = (m['pc_stable'] - plain['pc_stable']) * 100 if (m['pc_stable'] is not None and plain['pc_stable'] is not None) else None
        d_rec    = (m['rec'] - plain['rec']) if (m['rec'] is not None and plain['rec'] is not None) else None
        d_pc_str = f"  pc_delta={d_pc:+.2f}pp" if d_pc is not None else ""
        d_rec_str = f"  rec_delta={d_rec:+d} rds" if d_rec is not None else ""
        print(f"  {name:<32} stable_delta={d_stable:+.2f}pp  dip_delta={d_dip:+.2f}pp{d_rec_str}{d_pc_str}")

# Verdict comparison: oracle vs always at same factor
print("\n" + "=" * 110)
print("ORACLE vs ALWAYS-HIGHER (the critical control)")
print("=" * 110)
print("If oracle wins ONLY by what always-higher already gets, the trigger adds nothing.")
print()
for factor in [2, 3, 5]:
    ok_name = f'Oracle boost {factor}x'
    al_name = f'Always-higher {factor}x (lr=0.0{factor})'
    if ok_name in results and al_name in results:
        ok, al = results[ok_name], results[al_name]
        d_stable = (ok['stable'] - al['stable']) * 100
        d_dip    = (ok['dip'] - al['dip']) * 100
        d_pc     = (ok['pc_stable'] - al['pc_stable']) * 100 if (ok['pc_stable'] is not None and al['pc_stable'] is not None) else None
        d_pc_str = f"   pc_delta={d_pc:+.2f}pp" if d_pc is not None else ""
        print(f"  {factor}x:  Oracle stable={ok['stable']:.4f}  Always stable={al['stable']:.4f}  "
              f"Oracle-Always stable_delta={d_stable:+.2f}pp  dip_delta={d_dip:+.2f}pp{d_pc_str}")

# Action ceiling
print("\n" + "=" * 110)
print("ACTION CEILING (FedAvgAdam vs Plain FedAvg)")
print("=" * 110)
if 'FedAvgAdam (no trigger)' in results and 'Plain FedAvg' in results:
    fad = results['FedAvgAdam (no trigger)']
    plain = results['Plain FedAvg']
    d = (fad['stable'] - plain['stable']) * 100
    print(f"  FedAvgAdam stable = {fad['stable']:.4f}")
    print(f"  Plain FedAvg stable = {plain['stable']:.4f}")
    print(f"  delta = {d:+.2f}pp  (negative = Adam action below simple averaging; the action ceiling)")

print("\n" + "=" * 110)
print("VERDICT:")
print("  PASS (direction alive): oracle beats plain FedAvg meaningfully AND beats always-higher at same factor.")
print("  FAIL (direction dead):   oracle ~= plain FedAvg, OR oracle ~= always-higher.")
print("=" * 110)
