"""flash_multiseed_analysis.py - per-seed + 3-seed aggregate report for the
Flash confounding gate's MULTI-SEED CONFIRM on the two promising regimes.

Reads 3 seeds per regime:
  No-drift Dir(0.1):
    seed 0: runs/2026-06-24-flash-gate/nodrift_d01/results_{FedAvg,Flash}.csv
    seed 1: runs/2026-06-24-flash-multiseed/nodrift_d01/seed1/results_{FedAvg,Flash}.csv
    seed 2: runs/2026-06-24-flash-multiseed/nodrift_d01/seed2/results_{FedAvg,Flash}.csv
  Partial-coverage A:
    seed 0: runs/2026-06-24-flash-gate/partial_A/results_{FedAvg,Flash}.csv
    seed 1: runs/2026-06-24-flash-multiseed/partial_A/seed1/results_{FedAvg,Flash}.csv
    seed 2: runs/2026-06-24-flash-multiseed/partial_A/seed2/results_{FedAvg,Flash}.csv

Reports per regime:
  Per-seed: FedAvg stable, Flash stable, Delta global, Delta per-cli, mean(first 30 amp)
  3-seed:   mean +/- std (sample) on each metric
  Verdict:  is the gap robust across seeds (every seed shows Flash < FedAvg)?
            is the spurious-firing signal consistent across seeds?
"""

import csv
import numpy as np
from pathlib import Path

REGIMES = [
    # (name, first_drift_round_or_None, paths)
    ('No-drift Dir(0.1)', None, [
        ('seed 0', 'runs/2026-06-24-flash-gate/nodrift_d01/results_FedAvg.csv',
                   'runs/2026-06-24-flash-gate/nodrift_d01/results_Flash.csv'),
        ('seed 1', 'runs/2026-06-24-flash-multiseed/nodrift_d01/seed1/results_FedAvg.csv',
                   'runs/2026-06-24-flash-multiseed/nodrift_d01/seed1/results_Flash.csv'),
        ('seed 2', 'runs/2026-06-24-flash-multiseed/nodrift_d01/seed2/results_FedAvg.csv',
                   'runs/2026-06-24-flash-multiseed/nodrift_d01/seed2/results_Flash.csv'),
    ]),
    ('Partial-coverage A @ rd 100', 100, [
        ('seed 0', 'runs/2026-06-24-flash-gate/partial_A/results_FedAvg.csv',
                   'runs/2026-06-24-flash-gate/partial_A/results_Flash.csv'),
        ('seed 1', 'runs/2026-06-24-flash-multiseed/partial_A/seed1/results_FedAvg.csv',
                   'runs/2026-06-24-flash-multiseed/partial_A/seed1/results_Flash.csv'),
        ('seed 2', 'runs/2026-06-24-flash-multiseed/partial_A/seed2/results_FedAvg.csv',
                   'runs/2026-06-24-flash-multiseed/partial_A/seed2/results_Flash.csv'),
    ]),
]


def load_csv(p):
    if not Path(p).exists():
        return None
    rows = list(csv.DictReader(Path(p).open()))
    out = {
        'global': np.array([float(r['global_acc']) for r in rows]),
        'pc':     np.array([float(r.get('per_client_gen_acc') or 'nan') for r in rows])
                  if 'per_client_gen_acc' in rows[0] else None,
    }
    if 'flash_amp' in rows[0]:
        out['amp'] = np.array([float(r.get('flash_amp') or 'nan') for r in rows])
    return out


def stable(acc):
    return float(np.mean(acc[-10:]))


print("=" * 100)
print("FLASH CONFOUNDING GATE - MULTI-SEED CONFIRM (3 seeds, 2 promising regimes)")
print("=" * 100)

for name, first_drift, seeds in REGIMES:
    print(f"\n>>> {name}")
    print(f"    {'seed':<8}{'FedAvg g':>10}{'Flash g':>10}{'dg':>10}{'FedAvg pc':>11}{'Flash pc':>10}{'dpc':>10}{'30-amp':>10}")
    d_g, d_pc, amp30, missing = [], [], [], []
    for label, fa_p, fl_p in seeds:
        fa = load_csv(fa_p)
        fl = load_csv(fl_p)
        if fa is None or fl is None:
            missing.append(label)
            print(f"    {label:<8}  MISSING ({'FedAvg' if fa is None else 'Flash'})")
            continue
        fa_g = stable(fa['global'])
        fl_g = stable(fl['global'])
        fa_pc = stable(fa['pc']) if fa['pc'] is not None else None
        fl_pc = stable(fl['pc']) if fl['pc'] is not None else None
        dg  = (fl_g - fa_g) * 100
        dpc = (fl_pc - fa_pc) * 100 if (fa_pc is not None and fl_pc is not None) else None
        a30 = float(np.mean(fl['amp'][:30])) if 'amp' in fl else None
        d_g.append(dg)
        if dpc is not None: d_pc.append(dpc)
        if a30 is not None: amp30.append(a30)
        dpc_str = f"{dpc:+.2f}pp" if dpc is not None else "-"
        a30_str = f"{a30:.4f}" if a30 is not None else "-"
        print(f"    {label:<8}{fa_g:>10.4f}{fl_g:>10.4f}{dg:>+9.2f}pp{fa_pc:>11.4f}{fl_pc:>10.4f}{dpc_str:>10}{a30_str:>10}")

    if missing:
        print(f"    NOTE: {missing} missing -- aggregates below are over {len(d_g)} available seeds")

    if len(d_g) >= 2:
        dg_arr = np.array(d_g)
        print(f"    3-SEED mean: dg = {dg_arr.mean():+.2f}pp ± {dg_arr.std(ddof=1):.2f}pp", end='')
        if d_pc:
            dpc_arr = np.array(d_pc)
            print(f"   dpc = {dpc_arr.mean():+.2f}pp ± {dpc_arr.std(ddof=1):.2f}pp", end='')
        if amp30:
            a_arr = np.array(amp30)
            print(f"   first-30 amp = {a_arr.mean():.4f} ± {a_arr.std(ddof=1):.4f}")
        else:
            print()
        # Robustness check
        all_negative = all(g < 0 for g in d_g)
        all_pc_neg   = all(p < 0 for p in d_pc) if d_pc else False
        all_above_noise = all(abs(g) > 1.0 for g in d_g)
        print(f"    Robustness: every seed shows Flash < FedAvg on global? {all_negative}")
        print(f"                every seed shows Flash < FedAvg on per-cli? {all_pc_neg}")
        print(f"                every seed gap > 1pp (above noise floor)?   {all_above_noise}")

print()
print("=" * 100)
print("VERDICT MATRIX")
print("  PASS = gap robust across all 3 seeds AND beyond noise floor (>1pp) AND")
print("         no-drift regime shows consistent spurious firing (first-30 amp meaningfully > 0)")
print("  PARTIAL = gap survives multi-seed but one/two seeds within noise, or partial robustness")
print("  FAIL  = single-seed gap doesn't survive: one or more seeds tie or flip sign")
print("=" * 100)
