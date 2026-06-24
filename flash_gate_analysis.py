"""flash_gate_analysis.py - compare FedAvg vs Flash across 4 regimes for the
Flash confounding gate (does Flash's update-magnitude trigger spuriously
fire under static heterogeneity / partial-coverage drift?).

Reads:
  No-drift Dir(0.1):
    runs/2026-06-24-flash-gate/nodrift_d01/results_FedAvg.csv
    runs/2026-06-24-flash-gate/nodrift_d01/results_Flash.csv
  Partial cohort A drift @ rd 100:
    runs/2026-06-24-flash-gate/partial_A/results_FedAvg.csv
    runs/2026-06-24-flash-gate/partial_A/results_Flash.csv
  Canonical Dir(0.1) sudden drift (FedAvg reused from prior smoke):
    runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv  [REUSED]
    runs/2026-06-24-flash-gate/canonical_d01/results_Flash.csv
  Dir(0.5) sudden drift (FedAvg reused from Saile gate):
    runs/2026-06-11-dir05-test/fedavg/results_FedAvg.csv       [REUSED]
    runs/2026-06-24-flash-gate/dir05/results_Flash.csv

Reports per regime:
  global_acc + per_client_gen_acc:  pre / dip / stable for both methods
  Delta Flash - FedAvg on stable (positive = Flash better)
  Delta Flash - FedAvg on dip (positive = Flash dips deeper, worse)
  Flash trigger diagnostic:
    pre-drift window mean(flash_amp), max(flash_amp), mean(delta_mom_norm)
    For no-drift: entire run is "pre-drift"; we report whole-run averages
    plus the late-round (last 30) mean as the steady-state firing level.
"""

import csv
import numpy as np
from pathlib import Path

REGIMES = [
    # (name, first_drift_round_or_None, fedavg_csv, flash_csv)
    ('No-drift Dir(0.1)',                None,
     'runs/2026-06-24-flash-gate/nodrift_d01/results_FedAvg.csv',
     'runs/2026-06-24-flash-gate/nodrift_d01/results_Flash.csv'),
    ('Partial-coverage drift (cohort A @ rd 100)', 100,
     'runs/2026-06-24-flash-gate/partial_A/results_FedAvg.csv',
     'runs/2026-06-24-flash-gate/partial_A/results_Flash.csv'),
    ('Canonical Dir(0.1) sudden @ rd 100', 100,
     'runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv',
     'runs/2026-06-24-flash-gate/canonical_d01/results_Flash.csv'),
    ('Dir(0.5) sudden @ rd 100',          100,
     'runs/2026-06-11-dir05-test/fedavg/results_FedAvg.csv',
     'runs/2026-06-24-flash-gate/dir05/results_Flash.csv'),
]


def load(p):
    if not Path(p).exists():
        return None
    rows = list(csv.DictReader(Path(p).open()))
    out = {'rows': rows}
    out['global']  = np.array([float(r['global_acc']) for r in rows])
    if 'per_client_gen_acc' in rows[0]:
        out['pc'] = np.array([float(r.get('per_client_gen_acc') or 'nan') for r in rows])
    else:
        out['pc'] = None
    if 'flash_amp' in rows[0]:
        out['amp']        = np.array([float(r.get('flash_amp') or 'nan') for r in rows])
        out['delta_mom']  = np.array([float(r.get('delta_mom_norm') or 'nan') for r in rows])
    return out


def windows(acc, first_drift):
    """Return (pre, dip, stable). If first_drift is None, pre=mean(first 30),
    dip=N/A, stable=mean(last 10)."""
    if acc is None: return (None, None, None)
    if first_drift is None:
        return (float(np.mean(acc[:30])), None, float(np.mean(acc[-10:])))
    pre    = float(np.mean(acc[max(0, first_drift-11):first_drift]))
    dip    = pre - float(np.min(acc[first_drift:first_drift+10]))
    stable = float(np.mean(acc[-10:]))
    return pre, dip, stable


print("=" * 100)
print("FLASH CONFOUNDING GATE - single-seed regime sweep")
print("=" * 100)

summary = []

for name, first_drift, fa_path, fl_path in REGIMES:
    print(f"\n>>> {name}")
    fa = load(fa_path)
    fl = load(fl_path)
    if fa is None:
        print(f"    MISSING FedAvg: {fa_path}")
        continue
    if fl is None:
        print(f"    MISSING Flash: {fl_path}")
        continue
    fa_g, fa_pc = fa['global'], fa['pc']
    fl_g, fl_pc = fl['global'], fl['pc']

    fa_gw  = windows(fa_g, first_drift)
    fl_gw  = windows(fl_g, first_drift)
    fa_pcw = windows(fa_pc, first_drift) if fa_pc is not None else (None, None, None)
    fl_pcw = windows(fl_pc, first_drift) if fl_pc is not None else (None, None, None)

    print(f"    global:    FedAvg pre={fa_gw[0]:.4f} dip={fa_gw[1] if fa_gw[1] is not None else float('nan'):.4f} stable={fa_gw[2]:.4f}")
    print(f"               Flash  pre={fl_gw[0]:.4f} dip={fl_gw[1] if fl_gw[1] is not None else float('nan'):.4f} stable={fl_gw[2]:.4f}")
    if fa_pc is not None and fl_pc is not None:
        print(f"    per-cli:   FedAvg pre={fa_pcw[0]:.4f} dip={fa_pcw[1] if fa_pcw[1] is not None else float('nan'):.4f} stable={fa_pcw[2]:.4f}")
        print(f"               Flash  pre={fl_pcw[0]:.4f} dip={fl_pcw[1] if fl_pcw[1] is not None else float('nan'):.4f} stable={fl_pcw[2]:.4f}")

    d_g_stable  = (fl_gw[2] - fa_gw[2]) * 100
    d_pc_stable = (fl_pcw[2] - fa_pcw[2]) * 100 if (fa_pc is not None and fl_pc is not None) else None
    d_g_dip     = (fl_gw[1] - fa_gw[1]) * 100 if (fa_gw[1] is not None and fl_gw[1] is not None) else None

    if d_pc_stable is not None:
        print(f"    delta Flash-FedAvg stable: global {d_g_stable:+.2f}pp   per-cli {d_pc_stable:+.2f}pp")
    else:
        print(f"    delta Flash-FedAvg stable: global {d_g_stable:+.2f}pp")
    if d_g_dip is not None:
        print(f"    delta Flash-FedAvg dip:    global {d_g_dip:+.2f}pp  (positive = Flash dips deeper = worse)")

    # Flash trigger diagnostic
    if 'amp' in fl:
        amp = fl['amp']
        dm  = fl['delta_mom']
        if first_drift is None:
            # Entire run is "pre-drift"; report whole-run + late-round (last 30)
            print(f"    Flash trigger (NO-DRIFT regime: any firing here is spurious by construction):")
            print(f"      whole-run    mean(flash_amp)={float(np.mean(amp)):.4f}  max={float(np.max(amp)):.4f}  "
                  f"mean(delta_mom_norm)={float(np.mean(dm)):.5f}")
            print(f"      first 30 rds mean(flash_amp)={float(np.mean(amp[:30])):.4f}")
            print(f"      last  30 rds mean(flash_amp)={float(np.mean(amp[-30:])):.4f}  (steady state)")
        else:
            # Pre-drift window (rounds 0 to first_drift-1)
            pre_amp = amp[:first_drift]
            print(f"    Flash trigger (pre-drift window rds 0..{first_drift-1}):")
            print(f"      mean(flash_amp)={float(np.mean(pre_amp)):.4f}  max={float(np.max(pre_amp)):.4f}  "
                  f"mean(delta_mom_norm)={float(np.mean(dm[:first_drift])):.5f}")
            print(f"      drift-round flash_amp={float(amp[first_drift]):.4f}  delta_mom_norm={float(dm[first_drift]):.5f}")
            print(f"      drift+5 mean(flash_amp)={float(np.mean(amp[first_drift:first_drift+5])):.4f}")

    summary.append((name, fa_gw[2], fl_gw[2], d_g_stable, d_pc_stable, d_g_dip,
                    float(np.mean(fl['amp'])) if 'amp' in fl else None))

# Compact summary table
print()
print("=" * 100)
print("SUMMARY TABLE")
print("=" * 100)
print(f"{'Regime':<48}{'FedAvg':>10}{'Flash':>10}{'d_g_stable':>14}{'d_pc_stable':>14}{'d_g_dip':>10}{'mean_amp':>11}")
for name, fa_s, fl_s, dg, dpc, dd, mamp in summary:
    dgs   = f"{dg:+.2f}pp"
    dpcs  = f"{dpc:+.2f}pp" if dpc is not None else "-"
    dds   = f"{dd:+.2f}pp"  if dd is not None else "-"
    mamps = f"{mamp:.4f}"   if mamp is not None else "-"
    print(f"{name[:48]:<48}{fa_s:>10.4f}{fl_s:>10.4f}{dgs:>14}{dpcs:>14}{dds:>10}{mamps:>11}")

print()
print("KEY DIAGNOSTIC FOR THE GATE:")
print("  No-drift Dir(0.1):   mean(flash_amp) - if substantially > 0, Flash is firing spuriously")
print("                       (any non-zero firing is spurious by construction - no drift to detect).")
print("  No-drift Dir(0.1):   delta Flash-FedAvg stable - if negative, the spurious firing HURTS.")
print("                       That is the gate's PASS evidence: spurious firing + cost = exploitable.")
print("  Partial-coverage:    Flash should hurt the non-drifting majority if its trigger over-reacts")
print("                       to the cohort-A drift signal mixed with static heterogeneity.")
