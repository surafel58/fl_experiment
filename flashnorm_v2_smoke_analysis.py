"""V2 smoke analysis vs the prior baselines + ColdInit floor."""
import csv
import numpy as np
from pathlib import Path


def load(p, amp_col=None):
    if not Path(p).exists(): return None
    rows = list(csv.DictReader(Path(p).open()))
    out = {
        'rows': rows,
        'global': np.array([float(r['global_acc']) for r in rows]),
        'pc':     np.array([float(r.get('per_client_gen_acc') or 'nan') for r in rows])
                  if 'per_client_gen_acc' in rows[0] else None,
    }
    if amp_col and amp_col in rows[0]:
        out['amp'] = np.array([float(r.get(amp_col) or 'nan') for r in rows])
    if 'ratio_norm' in rows[0]:
        out['ratio'] = np.array([float(r.get('ratio_norm') or 'nan') for r in rows])
    if 'scaled_delta_norm' in rows[0]:
        out['scaled'] = np.array([float(r.get('scaled_delta_norm') or 'nan') for r in rows])
    if 'B_layer_norm' in rows[0]:
        out['B'] = np.array([float(r.get('B_layer_norm') or 'nan') for r in rows])
    return out


def stable(acc): return float(np.mean(acc[-10:]))


print("=" * 100)
print("V2 (FlashNormV2 per-tensor B) smoke vs baselines  -- single seed, no-drift + canonical")
print("=" * 100)

# ---- NO-DRIFT ----
print("\n>>> NO-DRIFT Dir(0.1)")
runs_no = [
    ('FedAvg',                  'runs/2026-06-24-flash-gate/nodrift_d01/results_FedAvg.csv', None),
    ('Flash (stock)',           'runs/2026-06-24-flash-gate/nodrift_d01/results_Flash.csv', 'flash_amp'),
    ('FlashColdInit (control)', 'runs/2026-06-25-flashnorm-smoke/nodrift_coldinit/results_FlashColdInit.csv', 'flash_amp'),
    ('V1 Option II',            'runs/2026-06-25-flashnorm-smoke-v2/nodrift/results_FlashNormTrigger.csv', 'flash_amp_new'),
    ('V2 (per-tensor B)',       'runs/2026-06-25-v2-smoke/nodrift/results_FlashNormV2.csv', 'flash_amp_new'),
]
for name, p, amp in runs_no:
    d = load(p, amp_col=amp)
    if d is None:
        print(f"  {name:<28}  MISSING"); continue
    s = stable(d['global'])
    pre30 = float(np.mean(d['global'][:30]))
    s_pc = stable(d['pc']) if d['pc'] is not None else None
    amp_str = ""
    if 'amp' in d:
        amp30 = float(np.mean(d['amp'][:30]))
        amp_end = float(np.mean(d['amp'][-30:]))
        amp_str = f"   amp_first30={amp30:.4f}  amp_last30={amp_end:.4f}"
    pc_str = f"   pc_stable={s_pc:.4f}" if s_pc is not None else ""
    print(f"  {name:<28}  pre30={pre30:.4f}  stable={s:.4f}{pc_str}{amp_str}")

# Deltas vs FedAvg + bars
fa = load('runs/2026-06-24-flash-gate/nodrift_d01/results_FedAvg.csv')
fl = load('runs/2026-06-24-flash-gate/nodrift_d01/results_Flash.csv', amp_col='flash_amp')
ci = load('runs/2026-06-25-flashnorm-smoke/nodrift_coldinit/results_FlashColdInit.csv', amp_col='flash_amp')
v2 = load('runs/2026-06-25-v2-smoke/nodrift/results_FlashNormV2.csv', amp_col='flash_amp_new')

if fa is not None and v2 is not None:
    fa_s = stable(fa['global']); fl_s = stable(fl['global']); ci_s = stable(ci['global']); v2_s = stable(v2['global'])
    print(f"\n  Deltas (no-drift stable global_acc):")
    print(f"    V2 vs FedAvg:        {(v2_s - fa_s)*100:+.2f}pp  (FedAvg = {fa_s:.4f}, V2 = {v2_s:.4f})")
    print(f"    V2 vs Flash:         {(v2_s - fl_s)*100:+.2f}pp")
    print(f"    V2 vs FlashColdInit: {(v2_s - ci_s)*100:+.2f}pp  (ColdInit-floor delta vs Flash was {(ci_s - fl_s)*100:+.2f}pp)")
    print(f"    V2 first-30 amp:     {float(np.mean(v2['amp'][:30])):.4f}  (vs Flash {float(np.mean(fl['amp'][:30])):.4f}, ColdInit {float(np.mean(ci['amp'][:30])):.4f})")
    print(f"    V2 last-30 amp:      {float(np.mean(v2['amp'][-30:])):.4f}  (silence check: smaller better)")

# ---- CANONICAL ----
print("\n\n>>> CANONICAL Dir(0.1) sudden @ rd 100")
runs_ca = [
    ('FedAvg',            'runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv', None),
    ('Flash (stock)',     'runs/2026-06-24-flash-gate/canonical_d01/results_Flash.csv', 'flash_amp'),
    ('V1 Option II',      'runs/2026-06-25-flashnorm-smoke-v2/canonical/results_FlashNormTrigger.csv', 'flash_amp_new'),
    ('V2 (per-tensor B)', 'runs/2026-06-25-v2-smoke/canonical/results_FlashNormV2.csv', 'flash_amp_new'),
]
for name, p, amp in runs_ca:
    d = load(p, amp_col=amp)
    if d is None:
        print(f"  {name:<22}  MISSING"); continue
    g = d['global']
    pre  = float(np.mean(g[max(0, 100-11):100]))
    dip  = pre - float(np.min(g[100:110]))
    s    = stable(g)
    s_pc = stable(d['pc']) if d['pc'] is not None else None
    extra = f"   pc_stable={s_pc:.4f}" if s_pc is not None else ""
    print(f"  {name:<22}  pre={pre:.4f}  dip={dip:.4f}  stable={s:.4f}{extra}")

# Drift spike check
fl_ca = load('runs/2026-06-24-flash-gate/canonical_d01/results_Flash.csv', amp_col='flash_amp')
v2_ca = load('runs/2026-06-25-v2-smoke/canonical/results_FlashNormV2.csv', amp_col='flash_amp_new')
if fl_ca is not None and v2_ca is not None:
    print(f"\n  Drift spike check @ rd 100:")
    print(f"    Flash amp:    pre@95-99={float(np.mean(fl_ca['amp'][95:100])):.4f}   drift@100={float(fl_ca['amp'][100]):.4f}   +5avg={float(np.mean(fl_ca['amp'][100:105])):.4f}")
    print(f"    V2 amp_new:   pre@95-99={float(np.mean(v2_ca['amp'][95:100])):.4f}   drift@100={float(v2_ca['amp'][100]):.4f}   +5avg={float(np.mean(v2_ca['amp'][100:105])):.4f}")
    ratio_drift = float(v2_ca['amp'][100] / (fl_ca['amp'][100] + 1e-10))
    print(f"    V2/Flash drift-spike ratio @100: {ratio_drift:.3f}x")

    fa_ca = load('runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv')
    if fa_ca is not None:
        print(f"\n  Canonical deltas (stable):")
        print(f"    V2 vs FedAvg: {(stable(v2_ca['global']) - stable(fa_ca['global']))*100:+.2f}pp")
        print(f"    V2 vs Flash:  {(stable(v2_ca['global']) - stable(fl_ca['global']))*100:+.2f}pp")

# V2 trajectory diagnostics
if v2 is not None:
    print(f"\n>>> V2 NO-DRIFT trajectory (10-round samples):")
    for r in [0, 10, 20, 25, 30, 40, 60, 100, 150, 199]:
        g = float(v2['global'][r])
        amp = float(v2['amp'][r]) if 'amp' in v2 else 0.0
        ratio = float(v2['ratio'][r]) if 'ratio' in v2 else 0.0
        B = float(v2['B'][r]) if 'B' in v2 else 0.0
        warmup_mark = " (WARMUP)" if r < 25 else ""
        print(f"    rd {r:>3}  global={g:.4f}   amp_new={amp:.4f}   ratio={ratio:.2f}   B_norm={B:.4f}{warmup_mark}")

if v2_ca is not None:
    print(f"\n>>> V2 CANONICAL trajectory around drift (rd 95-110):")
    for r in range(95, 111):
        g = v2_ca['global'][r]
        amp = v2_ca['amp'][r]
        ratio = v2_ca.get('ratio', [0]*200)[r] if 'ratio' in v2_ca else 0
        marker = "  <-- DRIFT" if r == 100 else ""
        print(f"    rd {r:>3}  global={g:.4f}   amp_new={amp:.4f}   ratio={ratio:.2f}{marker}")
