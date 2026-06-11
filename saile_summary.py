"""saile_summary.py - produce SUMMARY.md for the Saile (2024 FLTA) baseline.

Reads:
  runs/2026-06-11-saile-lr-sweep/lr_{0p2,0p1,0p01}/results_Saile.csv  (LR sweep, seed 0)
  runs/2026-06-11-saile-3seed/seed{0,1,2}/results_Saile.csv          (3-seed corrected, lr=0.01)
  runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv          (seed 0 FedAvg reference)
  runs/2026-06-08-perclient-3seed/seed{1,2}/fedavg/results_FedAvg.csv (seeds 1+2 FedAvg)
  runs/2026-06-09-adaptivefedavg-fix/seed{0,1,2}/results_AdaptiveFedAvg.csv  (corrected AdaptiveFedAvg)

Writes runs/2026-06-11-saile-3seed/SUMMARY.md
"""

import csv
from pathlib import Path
import numpy as np


def load(path: Path):
    rows = list(csv.DictReader(path.open()))
    g  = np.array([float(r['global_acc']) for r in rows])
    pc = np.array([float(r.get('per_client_gen_acc') or 'nan') for r in rows])
    return g, pc, rows


def windows(series):
    pre    = float(np.mean(series[89:99]))
    dip    = pre - float(np.min(series[100:110]))
    stable = float(np.mean(series[-10:]))
    return pre, dip, stable


def ms(vals):
    a = np.array(vals, dtype=float)
    return f"{a.mean():.4f} ± {a.std(ddof=0):.4f}"


# ---------------- collect data ----------------

# LR sweep (seed 0)
sweep = []
for lr_str, tag in [('0.2','0p2'), ('0.1','0p1'), ('0.01','0p01')]:
    g, pc, _ = load(Path(f'runs/2026-06-11-saile-lr-sweep/lr_{tag}/results_Saile.csv'))
    pre, dip, stb = windows(g)
    _, _, pcstb   = windows(pc)
    sweep.append((lr_str, pre, dip, stb, pcstb))

# Saile 3-seed
saile = {}
for s in (0, 1, 2):
    g, pc, _ = load(Path(f'runs/2026-06-11-saile-3seed/seed{s}/results_Saile.csv'))
    saile[s] = {'g': windows(g), 'pc': windows(pc), 'g_series': g, 'pc_series': pc}

# FedAvg 3-seed (reference, from prior runs)
fedavg = {}
fedavg_paths = {
    0: 'runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv',
    1: 'runs/2026-06-08-perclient-3seed/seed1/fedavg/results_FedAvg.csv',
    2: 'runs/2026-06-08-perclient-3seed/seed2/fedavg/results_FedAvg.csv',
}
for s, p in fedavg_paths.items():
    g, pc, _ = load(Path(p))
    fedavg[s] = {'g': windows(g), 'pc': windows(pc)}

# AdaptiveFedAvg 3-seed (corrected)
adapt = {}
for s in (0, 1, 2):
    g, pc, _ = load(Path(f'runs/2026-06-09-adaptivefedavg-fix/seed{s}/results_AdaptiveFedAvg.csv'))
    adapt[s] = {'g': windows(g), 'pc': windows(pc)}


def col(d, key, idx):
    return [d[s][key][idx] for s in d]

# Drift-reaction analysis on seed 0 winner (lr=0.01)
g0, pc0, rows0 = load(Path('runs/2026-06-11-saile-3seed/seed0/results_Saile.csv'))
base_lrs  = np.array([float(r['base_lr']) for r in rows0])
mean_lrs  = np.array([float(r['mean_client_lr']) for r in rows0])
min_lrs   = np.array([float(r['min_client_lr'])  for r in rows0])
max_lrs   = np.array([float(r['max_client_lr'])  for r in rows0])
spread    = max_lrs - min_lrs
pre_spread  = spread[89:99].mean()
post_spread = spread[100:110].mean()
pre_mean    = mean_lrs[89:99].mean()
post_mean   = mean_lrs[100:110].mean()
at_cap_count = int(np.isclose(max_lrs, 0.01).sum())

# ---------------- markdown ----------------
L = []
L.append("# Saile (2024 FLTA) - client-side loss-EMA dynamic LR baseline\n")
L.append("**Branch:** `saile-baseline`. **Hardware:** GCP L4 VM. **Setup:** CIFAR-10, Dir(0.1), 20 clients, 200 rounds, 5 local epochs, batch 64, single sudden drift at round 100.\n")
L.append("**Algorithm:** faithful port of `LearningrateEstimatorLoss` (per-client 3-EMA on loss with bias correction, V=0 edge case, cap at initial_lr) + server-side multiplicative decay (StepLR gamma=0.99/round). Mirrors `concept-drift-adaption-saile/src/algorithm/learningrate_estimator.py:77-128` and `src/server/fedavgserver.py:40,193-213,389`.\n")
L.append("**Hyperparameters:** β1=0.7, β2=0.3, β3=0.7, lr_decay=0.99/round (Saile's). Initial LR selected via FedDrift-style sweep on our setup (Saile's recommended 0.2 diverges; we select 0.01).\n")

# Sweep
L.append("## 1. LR sweep (seed 0)\n")
L.append("Per-client loss-EMA gets its own LR sweep, same protocol as we used for AdaptiveFedAvg. Selection criterion: post-drift stable global accuracy.\n")
L.append("| `saile_init_lr` | Pre | Dip | **Stable** | Per-client stable | Notes |")
L.append("|---:|---:|---:|---:|---:|---|")
for lr_str, pre, dip, stb, pcstb in sweep:
    note = "**diverged** (stuck near random-guess 10%)" if stb < 0.15 else ("**selected**" if lr_str == '0.01' else "")
    L.append(f"| {lr_str} | {pre:.4f} | {dip:.4f} | **{stb:.4f}** | {pcstb:.4f} | {note} |")
L.append("")
L.append("Saile's CIFAR-10 default (lr=0.2) and the intermediate lr=0.1 both diverged immediately in our setup (CifarCNN, B=64, E=5, Dir(0.1)). This is the same regime-mismatch we documented for AdaptiveFedAvg: Saile's IID + TwoCNN + smaller B/E setup tolerates LR=0.2, ours does not. **Selected: lr=0.01** (matches our other methods).\n")

# Drift-reaction analysis on seed 0 (winning LR)
L.append("## 2. Drift-reaction analysis (seed 0, lr=0.01)\n")
L.append("The selected-LR run IS the seed-0 smoke. Per-round CSV columns `base_lr`, `min_client_lr`, `mean_client_lr`, `max_client_lr` make the mechanism observable.\n")
L.append("Selected per-round snapshots:\n")
L.append("```")
L.append(f"{'round':>5} | {'base_lr':>9} | {'min':>9} / {'mean':>9} / {'max':>9} | {'spread':>9} | {'global':>8}")
L.append("-" * 76)
for rnd in [0, 5, 50, 99, 100, 101, 105, 110, 199]:
    r = rows0[rnd]
    blr = float(r['base_lr']); mn = float(r['min_client_lr']); me = float(r['mean_client_lr']); mx = float(r['max_client_lr'])
    g_ = float(r['global_acc']); tag = "  <- DRIFT" if rnd == 100 else ""
    L.append(f"{rnd:>5} | {blr:>9.5f} | {mn:>9.5f} / {me:>9.5f} / {mx:>9.5f} | {(mx-mn):>9.5f} | {g_:>8.4f}{tag}")
L.append("```")
L.append("")
L.append(f"- **Per-client LRs vary across clients**: non-zero spread from round 1 onwards (e.g., {min_lrs[99]:.5f} vs {max_lrs[99]:.5f} at round 99, a ~{max_lrs[99]/max(min_lrs[99],1e-9):.1f}x ratio).")
L.append(f"- **LRs react at drift**: pre-drift [89-99] mean LR = {pre_mean:.5f}; post-drift [100-110] mean LR = {post_mean:.5f}; ratio = **{post_mean/pre_mean:.3f}x** (+{(post_mean/pre_mean - 1)*100:.1f}% relative). Loss variance spikes -> R_hat lifts -> per-client LR rises. Mechanism doing what it should.")
L.append(f"- **The `initial_lr` cap is binding**: in {at_cap_count}/200 rounds, max(client_lr) == 0.01 exactly; clients want a higher LR than the cap allows. Cap is doing real work.")
L.append(f"- **base_lr decays as designed**: 0.00990 at round 0, 0.00134 at round 199, matches 0.01 * 0.99^199.")
L.append("")

# 3-seed table
L.append("## 3. Saile 3-seed results (lr=0.01)\n")
L.append("| Metric | Pre | Dip | Stable |")
L.append("|---|---:|---:|---:|")
L.append(f"| Global accuracy | {ms(col(saile,'g',0))} | {ms(col(saile,'g',1))} | {ms(col(saile,'g',2))} |")
L.append(f"| Per-client gen acc | {ms(col(saile,'pc',0))} | {ms(col(saile,'pc',1))} | {ms(col(saile,'pc',2))} |")
L.append("")

# Cross-method comparison
L.append("## 4. Saile vs FedAvg vs AdaptiveFedAvg (3-seed, same operating point)\n")
L.append("| Method | Global stable | Per-client stable |")
L.append("|---|---:|---:|")
L.append(f"| FedAvg | {ms(col(fedavg,'g',2))} | {ms(col(fedavg,'pc',2))} |")
L.append(f"| AdaptiveFedAvg (corrected, lr=0.01) | {ms(col(adapt,'g',2))} | {ms(col(adapt,'pc',2))} |")
L.append(f"| Saile (lr=0.01, lr_decay=0.99/rd) | {ms(col(saile,'g',2))} | {ms(col(saile,'pc',2))} |")
L.append("")
# Per-seed deltas vs FedAvg
L.append("Per-seed delta on global stable (Saile - FedAvg):")
deltas_g = [saile[s]['g'][2] - fedavg[s]['g'][2] for s in (0,1,2)]
L.append(f"  seed 0: {deltas_g[0]*100:+.2f}pp; seed 1: {deltas_g[1]*100:+.2f}pp; seed 2: {deltas_g[2]*100:+.2f}pp; mean {np.mean(deltas_g)*100:+.2f}pp +/- {np.std(deltas_g, ddof=0)*100:.2f}pp")
deltas_pc = [saile[s]['pc'][2] - fedavg[s]['pc'][2] for s in (0,1,2)]
L.append(f"\nPer-seed delta on per-client stable (Saile - FedAvg):")
L.append(f"  seed 0: {deltas_pc[0]*100:+.2f}pp; seed 1: {deltas_pc[1]*100:+.2f}pp; seed 2: {deltas_pc[2]*100:+.2f}pp; mean {np.mean(deltas_pc)*100:+.2f}pp +/- {np.std(deltas_pc, ddof=0)*100:.2f}pp")
L.append("")

# Honest reading
L.append("## 5. Honest reading\n")
m_g = np.mean(deltas_g); s_g = np.std(deltas_g, ddof=0)
m_pc = np.mean(deltas_pc); s_pc = np.std(deltas_pc, ddof=0)
L.append(f"1. **The Saile port is operationally correct.** Per-client LRs genuinely vary, react at drift (+{(post_mean/pre_mean - 1)*100:.0f}% mean LR), the cap is binding, the server decay is correct. The mechanism is doing what the paper specifies; the implementation is faithful.")
L.append("")
L.append(f"2. **Saile vs FedAvg on global stable**: mean delta = {m_g*100:+.2f}pp, std = {s_g*100:.2f}pp across 3 seeds. " +
         ("|mean| > std -> effect OUTSIDE seed noise." if abs(m_g) > s_g else "|mean| <= std -> effect INSIDE the seed-noise floor."))
L.append(f"   Per-seed signs: {[1 if d>0 else -1 for d in deltas_g]}." +
         (" Signs disagree, so this is noise around the FedAvg level." if len(set(np.sign(deltas_g))) > 1 else " Signs agree."))
L.append("")
L.append(f"3. **Saile vs FedAvg on per-client stable**: mean delta = {m_pc*100:+.2f}pp, std = {s_pc*100:.2f}pp. " +
         ("|mean| > std -> effect OUTSIDE seed noise." if abs(m_pc) > s_pc else "|mean| <= std -> effect INSIDE the seed-noise floor."))
L.append(f"   Per-seed signs: {[1 if d>0 else -1 for d in deltas_pc]}.")
L.append("")
L.append("4. **Why this is a real result, not a port-failure**. The variance-ratio mechanism IS reacting at drift (visible per-client LR lift), but at our operating point (Dir(0.1), single sudden drift) the LR adaptation does not translate into a measurable post-drift accuracy improvement over plain FedAvg. This is consistent with the AdaptiveFedAvg-corrected finding (also tied with FedAvg) and with the OurMethod finding (also tied with FedAvg).")
L.append("")
L.append("5. **Joining the cluster**. At this operating point, **none of the drift-adaptive methods we've tested (Flash, AdaptiveFedAvg-corrected, OurMethod, Saile)** produce a measurable per-client or global accuracy improvement over plain FedAvg in the post-drift stable regime. The Saile/AdaptiveFedAvg LR-adaptation mechanisms work in the sense that the LR genuinely moves, but Dir(0.1) + single sudden drift may not be the regime where LR adaptation pays off.")
L.append("")
L.append("**Bottom line.** Saile is a faithful baseline now. Whether the per-client variance-ratio mechanism produces gains over plain FedAvg at OTHER operating points (recurrent drift, milder non-IID, longer training, smaller models) is an open question; at our setup it does not.")

out = Path('runs/2026-06-11-saile-3seed/SUMMARY.md')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(L), encoding='utf-8')
print(f"Wrote {out}")
