"""flashnorm_smoke_analysis.py - V1 (Option II) smoke + attribution control.

Compares:
  V1 (FlashNormTrigger, Option II) no-drift Dir(0.1)
  V1 (FlashNormTrigger, Option II) canonical Dir(0.1) sudden @ rd 100
  Control (FlashColdInit, no-drift): Flash + cold-start init only, no normalization
  Reused baselines from prior gate (seed 0):
    FedAvg no-drift, Flash no-drift, FedAvg canonical, Flash canonical
"""
import csv
import numpy as np
from pathlib import Path


def load(p, amp_col=None):
    if not Path(p).exists():
        return None
    rows = list(csv.DictReader(Path(p).open()))
    g  = np.array([float(r['global_acc']) for r in rows])
    pc = np.array([float(r.get('per_client_gen_acc') or 'nan') for r in rows]) if 'per_client_gen_acc' in rows[0] else None
    amp = None
    if amp_col and amp_col in rows[0]:
        amp = np.array([float(r.get(amp_col) or 'nan') for r in rows])
    ratio = None
    if 'ratio_norm' in rows[0]:
        ratio = np.array([float(r.get('ratio_norm') or 'nan') for r in rows])
    return rows, g, pc, amp, ratio


def windows(s, first_drift):
    if s is None: return (None, None, None)
    if first_drift is None:
        return (float(np.mean(s[:30])), None, float(np.mean(s[-10:])))
    pre    = float(np.mean(s[max(0, first_drift-11):first_drift]))
    dip    = pre - float(np.min(s[first_drift:first_drift+10]))
    stable = float(np.mean(s[-10:]))
    return pre, dip, stable


print("=" * 100)
print("FLASHNORMTRIGGER V1 (Option II) - smoke + attribution control")
print("=" * 100)

# ----- NO-DRIFT Dir(0.1) -----
print("\n>>> NO-DRIFT Dir(0.1)  (any firing here is spurious by construction)\n")
fa_no   = load('runs/2026-06-24-flash-gate/nodrift_d01/results_FedAvg.csv')
fl_no   = load('runs/2026-06-24-flash-gate/nodrift_d01/results_Flash.csv', amp_col='flash_amp')
v1_no   = load('runs/2026-06-25-flashnorm-smoke-v2/nodrift/results_FlashNormTrigger.csv', amp_col='flash_amp_new')
ci_no   = load('runs/2026-06-25-flashnorm-smoke/nodrift_coldinit/results_FlashColdInit.csv', amp_col='flash_amp')

for name, d in [('FedAvg', fa_no), ('Flash', fl_no), ('FlashColdInit (control)', ci_no), ('V1 FlashNormTrigger', v1_no)]:
    if d is None: print(f"  {name}: MISSING"); continue
    _, g, pc, amp, ratio = d
    pre, _, stable = windows(g, None)
    pc_pre, _, pc_stable = windows(pc, None) if pc is not None else (None, None, None)
    print(f"  {name:<30}  pre30={pre:.4f}  stable={stable:.4f}", end='')
    if pc is not None:
        print(f"   pc_pre30={pc_pre:.4f}  pc_stable={pc_stable:.4f}", end='')
    if amp is not None:
        print(f"   amp_30={float(np.mean(amp[:30])):.4f}  amp_stable={float(np.mean(amp[-30:])):.4f}", end='')
    print()

if v1_no is not None and ci_no is not None and fl_no is not None and fa_no is not None:
    _, _, _, v1_amp, _ = v1_no
    _, _, _, ci_amp, _ = ci_no
    _, _, _, fl_amp, _ = fl_no
    print(f"\n  Mechanism check (no-drift first-30 mean amp; smaller = quieter):")
    print(f"    Flash (stock)           : {float(np.mean(fl_amp[:30])):.4f}")
    print(f"    FlashColdInit (control) : {float(np.mean(ci_amp[:30])):.4f}")
    print(f"    V1 (Option II)          : {float(np.mean(v1_amp[:30])):.4f}")

    _, fa_g, _, _, _ = fa_no
    _, fl_g, _, _, _ = fl_no
    _, ci_g, _, _, _ = ci_no
    _, v1_g, _, _, _ = v1_no
    print(f"\n  Accuracy (no-drift stable, mean last 10):")
    print(f"    FedAvg                  : {float(np.mean(fa_g[-10:])):.4f}")
    print(f"    Flash (stock)           : {float(np.mean(fl_g[-10:])):.4f}  (delta vs FedAvg = {(np.mean(fl_g[-10:])-np.mean(fa_g[-10:]))*100:+.2f}pp)")
    print(f"    FlashColdInit (control) : {float(np.mean(ci_g[-10:])):.4f}  (delta vs FedAvg = {(np.mean(ci_g[-10:])-np.mean(fa_g[-10:]))*100:+.2f}pp)")
    print(f"    V1 (Option II)          : {float(np.mean(v1_g[-10:])):.4f}  (delta vs FedAvg = {(np.mean(v1_g[-10:])-np.mean(fa_g[-10:]))*100:+.2f}pp)")
    print(f"    V1 vs Flash             : {(np.mean(v1_g[-10:])-np.mean(fl_g[-10:]))*100:+.2f}pp")
    print(f"    V1 vs FlashColdInit     : {(np.mean(v1_g[-10:])-np.mean(ci_g[-10:]))*100:+.2f}pp")

# ----- CANONICAL Dir(0.1) sudden @ rd 100 -----
print("\n\n>>> CANONICAL Dir(0.1) sudden @ rd 100")
fa_ca   = load('runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv')
fl_ca   = load('runs/2026-06-24-flash-gate/canonical_d01/results_Flash.csv', amp_col='flash_amp')
v1_ca   = load('runs/2026-06-25-flashnorm-smoke-v2/canonical/results_FlashNormTrigger.csv', amp_col='flash_amp_new')

for name, d in [('FedAvg', fa_ca), ('Flash', fl_ca), ('V1 FlashNormTrigger', v1_ca)]:
    if d is None: print(f"  {name}: MISSING"); continue
    _, g, pc, amp, ratio = d
    pre, dip, stable = windows(g, 100)
    pc_pre, pc_dip, pc_stable = windows(pc, 100) if pc is not None else (None, None, None)
    print(f"  {name:<22}  pre={pre:.4f}  dip={dip:.4f}  stable={stable:.4f}", end='')
    if pc is not None:
        print(f"   pc_stable={pc_stable:.4f}", end='')
    print()

if v1_ca is not None and fl_ca is not None:
    _, _, _, v1_amp, _ = v1_ca
    _, _, _, fl_amp, _ = fl_ca
    print(f"\n  Drift spike check (rd 100 = drift event):")
    print(f"    Flash amp:    pre@95-99={float(np.mean(fl_amp[95:100])):.4f}  drift@100={float(fl_amp[100]):.4f}  +5avg={float(np.mean(fl_amp[100:105])):.4f}")
    print(f"    V1 amp_new:   pre@95-99={float(np.mean(v1_amp[95:100])):.4f}  drift@100={float(v1_amp[100]):.4f}  +5avg={float(np.mean(v1_amp[100:105])):.4f}")
    print(f"    V1/Flash drift-spike ratio @100: {float(v1_amp[100]/(fl_amp[100]+1e-10)):.3f}x")
