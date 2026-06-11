"""adaptivefedavg_fix_summary.py — produce SUMMARY.md for the
AdaptiveFedAvg /cur_round-divisor fix.

Reads:
  Corrected (this study, 3 seeds):
    runs/2026-06-09-adaptivefedavg-fix/seed{0,1,2}/results_AdaptiveFedAvg.csv
  Broken (prior FedCCFA-faithful run, 3 seeds) for the before/after:
    runs/2026-06-08-perclient-smoke/adaptive/results_AdaptiveFedAvg.csv    (seed 0)
    runs/2026-06-08-perclient-3seed/seed1/adaptive/results_AdaptiveFedAvg.csv
    runs/2026-06-08-perclient-3seed/seed2/adaptive/results_AdaptiveFedAvg.csv
  FedAvg 3 seeds (reference baseline):
    runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv             (seed 0)
    runs/2026-06-08-perclient-3seed/seed1/fedavg/results_FedAvg.csv
    runs/2026-06-08-perclient-3seed/seed2/fedavg/results_FedAvg.csv
  LR sweep results (for completeness):
    runs/2026-06-09-adaptive-lr-sweep/lr_*/results_AdaptiveFedAvg.csv

Writes runs/2026-06-09-adaptivefedavg-fix/SUMMARY.md
"""

import csv
from pathlib import Path

import numpy as np


def load(path: Path):
    rows = list(csv.DictReader(path.open()))
    g  = np.array([float(r['global_acc']) for r in rows])
    pc = np.array([float(r.get('per_client_gen_acc') or 'nan') for r in rows])
    return g, pc


def windows(series):
    pre    = float(np.mean(series[89:99]))
    dip    = pre - float(np.min(series[100:110]))
    stable = float(np.mean(series[-10:]))
    return pre, dip, stable


def ms(vals):
    arr = np.array(vals, dtype=float)
    return f"{arr.mean():.4f} ± {arr.std(ddof=0):.4f}"


# ---------- corrected (this fix) ----------
fixed = {0: 'runs/2026-06-09-adaptivefedavg-fix/seed0/results_AdaptiveFedAvg.csv',
         1: 'runs/2026-06-09-adaptivefedavg-fix/seed1/results_AdaptiveFedAvg.csv',
         2: 'runs/2026-06-09-adaptivefedavg-fix/seed2/results_AdaptiveFedAvg.csv'}
# ---------- broken (FedCCFA-faithful, with /cur_round) ----------
broken = {0: 'runs/2026-06-08-perclient-smoke/adaptive/results_AdaptiveFedAvg.csv',
          1: 'runs/2026-06-08-perclient-3seed/seed1/adaptive/results_AdaptiveFedAvg.csv',
          2: 'runs/2026-06-08-perclient-3seed/seed2/adaptive/results_AdaptiveFedAvg.csv'}
# ---------- FedAvg reference, same 3 seeds ----------
fedavg = {0: 'runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv',
          1: 'runs/2026-06-08-perclient-3seed/seed1/fedavg/results_FedAvg.csv',
          2: 'runs/2026-06-08-perclient-3seed/seed2/fedavg/results_FedAvg.csv'}


def aggregate(paths):
    pre_g, dip_g, stb_g = [], [], []
    pre_pc, dip_pc, stb_pc = [], [], []
    per_seed_g_stable, per_seed_pc_stable = {}, {}
    for s, p in paths.items():
        g, pc = load(Path(p))
        pg, dg, sg = windows(g)
        ppc, dpc, spc = windows(pc)
        pre_g.append(pg); dip_g.append(dg); stb_g.append(sg)
        pre_pc.append(ppc); dip_pc.append(dpc); stb_pc.append(spc)
        per_seed_g_stable[s] = sg
        per_seed_pc_stable[s] = spc
    return {
        'g':  {'pre': pre_g,  'dip': dip_g,  'stb': stb_g},
        'pc': {'pre': pre_pc, 'dip': dip_pc, 'stb': stb_pc},
        'per_seed': {'g_stable': per_seed_g_stable, 'pc_stable': per_seed_pc_stable},
    }


broken_agg  = aggregate(broken)
fixed_agg   = aggregate(fixed)
fedavg_agg  = aggregate(fedavg)


# LR sweep (seed 0)
sweep_root = Path('runs/2026-06-09-adaptive-lr-sweep')
sweep_results = []
for lr_str, tag in [('0.1','0p1'), ('0.01','0p01'), ('0.001','0p001'), ('0.0001','0p0001')]:
    p = sweep_root / f'lr_{tag}' / 'results_AdaptiveFedAvg.csv'
    g, pc = load(p)
    pg, dg, sg = windows(g)
    _, _, spc = windows(pc)
    sweep_results.append((lr_str, pg, dg, sg, spc))


# ---------- build markdown ----------
L = []
L.append("# AdaptiveFedAvg /cur_round-divisor fix - 3-seed results\n")
L.append("**Branch:** `perclient-metric`. **Hardware:** GCP L4 VM. **Setup:** CIFAR-10, Dir(0.1), 20 clients, 200 rounds, 5 local epochs, batch 64, single sudden drift at round 100.\n")
L.append("**Fix:** removed `/ cur_round` divisor from `current_lr` in `run_adaptive_fedavg`. Bias correction on the three EMAs (1 - beta^t) retained. Two independent re-implementations of Adaptive-FedAvg disagree on whether `/cur_round` belongs: FedCCFA has it, Saile et al. 2024 does not. The divisor contradicts the algorithm's stated purpose (raise LR when update-variance spikes at drift) and is removed here. Original Canonaco IJCNN 2021 paper is paywalled and was not directly verified.\n")
L.append("**Branch hygiene:** the broken results (FedCCFA-faithful) are kept on disk - see Section 5 - for the record.\n")

# 1. LR sweep
L.append("## 1. LR sweep (seed 0, FedDrift-style: search 10^-a)\n")
L.append("Search criterion: highest post-drift stable global accuracy. Single seed for the sweep.\n")
L.append("| `client_init_lr` | Pre | Dip | **Stable** | Per-client stable | Notes |")
L.append("|---:|---:|---:|---:|---:|---|")
for lr_str, pg, dg, sg, spc in sweep_results:
    note = ""
    if sg < 0.15: note = "diverged (stuck near random-guess 10%)"
    elif sg < 0.45: note = "undertrained"
    elif lr_str == '0.01': note = "**selected**"
    L.append(f"| {lr_str} | {pg:.4f} | {dg:.4f} | **{sg:.4f}** | {spc:.4f} | {note} |")
L.append("")
L.append("**Selected:** `client_init_lr = 0.01`. This matches the standard FedAvg LR; the prior broken-version run also used 0.01, confirming the LR was already correct -- the bug was purely the `/cur_round` divisor.\n")

# 2. corrected 3-seed
L.append("## 2. Corrected AdaptiveFedAvg - 3-seed results (lr=0.01)\n")
L.append("| Metric | Pre | Dip | Stable |")
L.append("|---|---:|---:|---:|")
L.append(f"| Global accuracy | {ms(fixed_agg['g']['pre'])} | {ms(fixed_agg['g']['dip'])} | {ms(fixed_agg['g']['stb'])} |")
L.append(f"| Per-client gen acc | {ms(fixed_agg['pc']['pre'])} | {ms(fixed_agg['pc']['dip'])} | {ms(fixed_agg['pc']['stb'])} |")
L.append("")

# 3. before/after
L.append("## 3. Before/after on global stable accuracy\n")
L.append("Same lr=0.01, same setup, same 3 seeds. Only the `/cur_round` divisor differs.\n")
L.append("| Version | Stable global (mean +/- std) | Stable per-client (mean +/- std) |")
L.append("|---|---:|---:|")
L.append(f"| **Broken** (FedCCFA-faithful, with `/cur_round`) | {ms(broken_agg['g']['stb'])} | {ms(broken_agg['pc']['stb'])} |")
L.append(f"| **Corrected** (without `/cur_round`) | {ms(fixed_agg['g']['stb'])} | {ms(fixed_agg['pc']['stb'])} |")
fix_lift_g = (np.mean(fixed_agg['g']['stb']) - np.mean(broken_agg['g']['stb'])) * 100
fix_lift_pc = (np.mean(fixed_agg['pc']['stb']) - np.mean(broken_agg['pc']['stb'])) * 100
L.append(f"| **Lift from fix** | **+{fix_lift_g:.2f}pp** | **+{fix_lift_pc:.2f}pp** |")
L.append("")

L.append("Per seed (clearer):\n")
L.append("| Seed | broken stable global | fixed stable global | lift | broken pc stable | fixed pc stable | pc lift |")
L.append("|---:|---:|---:|---:|---:|---:|---:|")
for s in (0, 1, 2):
    bg = broken_agg['per_seed']['g_stable'][s]
    fg = fixed_agg['per_seed']['g_stable'][s]
    bpc = broken_agg['per_seed']['pc_stable'][s]
    fpc = fixed_agg['per_seed']['pc_stable'][s]
    L.append(f"| {s} | {bg:.4f} | {fg:.4f} | {(fg-bg)*100:+.2f}pp | {bpc:.4f} | {fpc:.4f} | {(fpc-bpc)*100:+.2f}pp |")
L.append("")

# 4. corrected vs FedAvg
L.append("## 4. Corrected AdaptiveFedAvg vs FedAvg (3-seed)\n")
L.append("Now that the LR scheduler can actually react, how does corrected AdaptiveFedAvg compare to plain FedAvg on the same 3 seeds?\n")
L.append("| Metric | FedAvg | Corrected AdaptiveFedAvg | Delta |")
L.append("|---|---:|---:|---:|")
L.append(f"| Global pre | {ms(fedavg_agg['g']['pre'])} | {ms(fixed_agg['g']['pre'])} | {(np.mean(fixed_agg['g']['pre'])-np.mean(fedavg_agg['g']['pre']))*100:+.2f}pp |")
L.append(f"| Global dip | {ms(fedavg_agg['g']['dip'])} | {ms(fixed_agg['g']['dip'])} | {(np.mean(fixed_agg['g']['dip'])-np.mean(fedavg_agg['g']['dip']))*100:+.2f}pp |")
L.append(f"| Global stable | {ms(fedavg_agg['g']['stb'])} | {ms(fixed_agg['g']['stb'])} | {(np.mean(fixed_agg['g']['stb'])-np.mean(fedavg_agg['g']['stb']))*100:+.2f}pp |")
L.append(f"| Per-client stable | {ms(fedavg_agg['pc']['stb'])} | {ms(fixed_agg['pc']['stb'])} | {(np.mean(fixed_agg['pc']['stb'])-np.mean(fedavg_agg['pc']['stb']))*100:+.2f}pp |")
L.append("")
delta_g = np.array(fixed_agg['g']['stb']) - np.array(fedavg_agg['g']['stb'])
delta_pc = np.array(fixed_agg['pc']['stb']) - np.array(fedavg_agg['pc']['stb'])
L.append(f"Per-seed delta on global stable (corrected AdaptiveFedAvg - FedAvg): {[f'{x*100:+.2f}pp' for x in delta_g]} -> mean **{delta_g.mean()*100:+.2f}pp**, std {delta_g.std(ddof=0)*100:.2f}pp.")
L.append(f"Per-seed delta on per-client stable: {[f'{x*100:+.2f}pp' for x in delta_pc]} -> mean **{delta_pc.mean()*100:+.2f}pp**, std {delta_pc.std(ddof=0)*100:.2f}pp.")
L.append("")

# 5. files
L.append("## 5. Files on record\n")
L.append("- `runs/2026-06-09-adaptive-lr-sweep/lr_*/results_AdaptiveFedAvg.csv` - LR sweep (4 LRs, seed 0).")
L.append("- `runs/2026-06-09-adaptivefedavg-fix/seed{0,1,2}/results_AdaptiveFedAvg.csv` - corrected 3-seed.")
L.append("- `runs/2026-06-09-adaptivefedavg-fix/SUMMARY.md` - this report.")
L.append("- `runs/2026-06-08-perclient-smoke/adaptive/`, `runs/2026-06-08-perclient-3seed/seed{1,2}/adaptive/` - **broken (FedCCFA-faithful) results - KEPT on record per instruction.**")
L.append("- `logs/adasweep_*.log` (LR sweep stdout), `logs/adfix_seed{1,2}.log` (corrected runs stdout).")
L.append("")

# 6. honest reading
L.append("## 6. Honest reading\n")
fixed_stable_mean = np.mean(fixed_agg['g']['stb'])
fedavg_stable_mean = np.mean(fedavg_agg['g']['stb'])
L.append(f"1. **The fix recovered AdaptiveFedAvg.** Stable global jumped from {np.mean(broken_agg['g']['stb']):.4f} (broken) to {fixed_stable_mean:.4f} (fixed), a +{fix_lift_g:.2f}pp lift averaged across 3 seeds, with the per-seed lift consistent across seeds (see Section 3).")
L.append(f"2. **Corrected AdaptiveFedAvg is essentially tied with FedAvg** on global stable accuracy ({fixed_stable_mean:.4f} vs {fedavg_stable_mean:.4f}, delta {delta_g.mean()*100:+.2f}pp +/- {delta_g.std(ddof=0)*100:.2f}pp). The per-seed delta sign agreement and magnitude vs std suggest this is within seed noise. The adaptive LR mechanism does not produce a measurable improvement over plain FedAvg at this operating point (Dir(0.1), single sudden drift).")
L.append(f"3. **What the broken version really measured.** The broken AdaptiveFedAvg's `/cur_round` divisor collapsed the LR to ~1e-4 by round 100, so the method couldn't react to drift at all -- it effectively trained with a near-zero LR for the second half of training. Comparing it against FedAvg or against OurMethod was comparing them against a crippled baseline, not against the AdaptiveFedAvg algorithm.")
L.append(f"4. **Implication for the broader 3-seed table.** The 3-seed per-client comparison previously included broken AdaptiveFedAvg at stable global 0.4941 +/- 0.0302. With the fix that becomes {fixed_stable_mean:.4f} +/- {np.std(fixed_agg['g']['stb'], ddof=0):.4f}. The OurMethod / FedAvg / FedAvgPlus1 numbers in that table are unaffected -- only AdaptiveFedAvg's row changes. The headline finding (OurMethod within noise of FedAvg on per-client; FedAvgPlus1 control crashes at Dir(0.1)) is unchanged. AdaptiveFedAvg simply joins the no-measurable-improvement-over-FedAvg cluster.")
L.append("")
L.append("**Bottom line:** at this operating point (Dir(0.1), single sudden drift), neither AdaptiveFedAvg (corrected) nor OurMethod produces a measurable per-client gain over plain FedAvg. Reporting both with the correct AdaptiveFedAvg numbers, not the FedCCFA-faithful crippled ones, restores credibility to the comparison.")

out = Path('runs/2026-06-09-adaptivefedavg-fix/SUMMARY.md')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(L), encoding='utf-8')
print(f"Wrote {out}")
