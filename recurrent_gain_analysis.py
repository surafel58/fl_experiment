"""recurrent_gain_analysis.py - go/no-go analysis for alternating-drift test.

Reads FedAvg + OurMethod CSVs from runs/2026-06-13-recurrent-gain-test/
under the alternating-drift schedule [40, 80, 120, 160]:
  Phase 1 (canonical):  rounds [0,   39]
  Phase 2 (swapped):    rounds [40,  79]
  Phase 3 (canonical):  rounds [80,  119]
  Phase 4 (swapped):    rounds [120, 159]
  Phase 5 (canonical):  rounds [160, 199]

Reports:
  - Accuracy trajectory windowed by phase (pre/dip/recovered, both metrics).
  - Forgetting on canonical concept:
      IMMEDIATE = peak(phase1) - mean(rounds 80-89)   [discriminator]
      RESIDUAL  = peak(phase1) - mean(rounds 110-119) [after re-adaptation]
  - Whole-post-drift mean (rounds 40-199), since there's no single plateau.
  - OurMethod detection peak at each of the 4 events.
  - OurMethod - FedAvg deltas on accuracy + forgetting.
"""

import csv
from pathlib import Path
import numpy as np


def load(p):
    rows = list(csv.DictReader(Path(p).open()))
    g  = np.array([float(r['global_acc']) for r in rows])
    pc = np.array([float(r.get('per_client_gen_acc') or 'nan') for r in rows])
    return g, pc


# Load both methods
fa_g, fa_pc = load('runs/2026-06-13-recurrent-gain-test/fedavg/results_FedAvg.csv')
om_g, om_pc = load('runs/2026-06-13-recurrent-gain-test/ourmethod/results_OurMethod.csv')

# Load OurMethod flag CSV for per-event detection
flag_rows = list(csv.DictReader(Path('runs/2026-06-13-recurrent-gain-test/ourmethod/results_OurMethod_flags.csv').open()))


# ============================================================
# Phase windows
# ============================================================
PHASE1 = (0,   40)    # canonical
PHASE2 = (40,  80)    # swapped
PHASE3 = (80,  120)   # canonical (after first swap-back)
PHASE4 = (120, 160)   # swapped
PHASE5 = (160, 200)   # canonical (after second swap-back)

DRIFT_EVENTS = [40, 80, 120, 160]


def phase_stats(series, start, end):
    """Return (peak, min, mean) over rounds [start, end)."""
    w = series[start:end]
    return float(w.max()), float(w.min()), float(w.mean())


def fmt(x):
    return f"{x:.4f}"


# ============================================================
# Per-phase peak/min/mean for both methods on both metrics
# ============================================================
print("=" * 80)
print("PHASE-WINDOWED ACCURACY (peak / min / mean)")
print("=" * 80)
phases = [('P1 canon ', PHASE1),
          ('P2 swap  ', PHASE2),
          ('P3 canon ', PHASE3),
          ('P4 swap  ', PHASE4),
          ('P5 canon ', PHASE5)]
print(f"{'phase':>10} | {'FedAvg G':>22} | {'OurM G':>22} | {'FedAvg PC':>22} | {'OurM PC':>22}")
print(f"{'':>10} | {'pk/mn/mean':>22} | {'pk/mn/mean':>22} | {'pk/mn/mean':>22} | {'pk/mn/mean':>22}")
print('-' * 110)
for name, (s, e) in phases:
    fa_g_s  = phase_stats(fa_g, s, e)
    om_g_s  = phase_stats(om_g, s, e)
    fa_pc_s = phase_stats(fa_pc, s, e)
    om_pc_s = phase_stats(om_pc, s, e)
    print(f"{name:>10} | {fa_g_s[0]:.4f}/{fa_g_s[1]:.4f}/{fa_g_s[2]:.4f} | "
          f"{om_g_s[0]:.4f}/{om_g_s[1]:.4f}/{om_g_s[2]:.4f} | "
          f"{fa_pc_s[0]:.4f}/{fa_pc_s[1]:.4f}/{fa_pc_s[2]:.4f} | "
          f"{om_pc_s[0]:.4f}/{om_pc_s[1]:.4f}/{om_pc_s[2]:.4f}")


# ============================================================
# Forgetting metric (canonical concept)
# ============================================================
print()
print("=" * 80)
print("FORGETTING (canonical concept)")
print("=" * 80)
# Peak phase 1 (rounds 0-39)
fa_peak = float(fa_g[0:40].max())
om_peak = float(om_g[0:40].max())

# IMMEDIATE forgetting: peak(phase1) - mean(rounds 80-89)
# = forgetting measured RIGHT AFTER swap-back, before re-adaptation
fa_imm_post = float(fa_g[80:90].mean())
om_imm_post = float(om_g[80:90].mean())
fa_imm_F = fa_peak - fa_imm_post
om_imm_F = om_peak - om_imm_post

# RESIDUAL forgetting: peak(phase1) - mean(rounds 110-119)
# = after 30 rounds of re-adaptation on canonical
fa_res_post = float(fa_g[110:120].mean())
om_res_post = float(om_g[110:120].mean())
fa_res_F = fa_peak - fa_res_post
om_res_F = om_peak - om_res_post

print(f"  FedAvg     peak(P1)={fa_peak:.4f}  mean(80..89)={fa_imm_post:.4f}  IMMEDIATE F={fa_imm_F:+.4f}")
print(f"  OurMethod  peak(P1)={om_peak:.4f}  mean(80..89)={om_imm_post:.4f}  IMMEDIATE F={om_imm_F:+.4f}")
print(f"  Delta (OurMethod - FedAvg) on IMMEDIATE forgetting: {(om_imm_F - fa_imm_F)*100:+.2f}pp")
print(f"    (negative -> OurMethod forgets LESS; positive -> OurMethod forgets MORE)")
print()
print(f"  FedAvg     peak(P1)={fa_peak:.4f}  mean(110..119)={fa_res_post:.4f}  RESIDUAL F={fa_res_F:+.4f}")
print(f"  OurMethod  peak(P1)={om_peak:.4f}  mean(110..119)={om_res_post:.4f}  RESIDUAL F={om_res_F:+.4f}")
print(f"  Delta (OurMethod - FedAvg) on RESIDUAL forgetting: {(om_res_F - fa_res_F)*100:+.2f}pp")


# ============================================================
# Whole-post-drift mean (rounds 40-199)
# ============================================================
print()
print("=" * 80)
print("WHOLE-POST-DRIFT MEAN (rounds 40-199, no single 'stable' plateau exists)")
print("=" * 80)
fa_g_pd  = float(fa_g[40:].mean())
om_g_pd  = float(om_g[40:].mean())
fa_pc_pd = float(fa_pc[40:].mean())
om_pc_pd = float(om_pc[40:].mean())
print(f"  FedAvg    global mean[40..199] = {fa_g_pd:.4f} | per-client = {fa_pc_pd:.4f}")
print(f"  OurMethod global mean[40..199] = {om_g_pd:.4f} | per-client = {om_pc_pd:.4f}")
print(f"  Delta global (OurM - FA)    = {(om_g_pd - fa_g_pd)*100:+.2f}pp")
print(f"  Delta per-client (OurM - FA)= {(om_pc_pd - fa_pc_pd)*100:+.2f}pp")


# ============================================================
# Final-phase stable (rounds 190-199, in phase 5 = canonical)
# ============================================================
print()
print("=" * 80)
print("FINAL-PHASE STABLE (rounds 190-199, in phase 5 canonical)")
print("=" * 80)
fa_g_st  = float(fa_g[-10:].mean())
om_g_st  = float(om_g[-10:].mean())
fa_pc_st = float(fa_pc[-10:].mean())
om_pc_st = float(om_pc[-10:].mean())
print(f"  FedAvg    global stable = {fa_g_st:.4f} | per-client = {fa_pc_st:.4f}")
print(f"  OurMethod global stable = {om_g_st:.4f} | per-client = {om_pc_st:.4f}")
print(f"  Delta global (OurM - FA)    = {(om_g_st - fa_g_st)*100:+.2f}pp")
print(f"  Delta per-client (OurM - FA)= {(om_pc_st - fa_pc_st)*100:+.2f}pp")


# ============================================================
# OurMethod detection at each of the 4 events
# ============================================================
print()
print("=" * 80)
print("OURMETHOD DETECTION AT EACH EVENT")
print("=" * 80)
print(f"{'event':>5} | {'round':>5} | {'direction':>20} | {'flag_count[ev..ev+9]':>22} | {'peak / 20'}")
print('-' * 80)
def direction_for(idx):
    # event 0: 40 -> swapped; event 1: 80 -> canonical;
    # event 2: 120 -> swapped; event 3: 160 -> canonical
    return "-> swapped" if idx % 2 == 0 else "-> canonical (swap back)"
for idx, rnd in enumerate(DRIFT_EVENTS):
    window = [int(flag_rows[r]['flagged_count']) for r in range(rnd, min(rnd + 10, 200))]
    print(f"{idx:>5} | {rnd:>5} | {direction_for(idx):>20} | {str(window):>22} | {max(window):>5}/20")


# ============================================================
# Bottom line
# ============================================================
print()
print("=" * 80)
print("GO/NO-GO BOTTOM LINE")
print("=" * 80)
dom_g_pd  = (om_g_pd  - fa_g_pd)  * 100
dom_pc_pd = (om_pc_pd - fa_pc_pd) * 100
dom_imm_F = (om_imm_F - fa_imm_F) * 100
print(f"  OurMethod - FedAvg, mean post-drift global  : {dom_g_pd:+.2f}pp")
print(f"  OurMethod - FedAvg, mean post-drift per-cli : {dom_pc_pd:+.2f}pp")
print(f"  OurMethod - FedAvg, IMMEDIATE forgetting    : {dom_imm_F:+.2f}pp  (negative -> OurMethod forgets less)")
print(f"  OurMethod - FedAvg, final-phase stable      : {(om_g_st - fa_g_st)*100:+.2f}pp global, {(om_pc_st - fa_pc_st)*100:+.2f}pp per-client")
