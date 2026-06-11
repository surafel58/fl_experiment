# AdaptiveFedAvg /cur_round-divisor fix - 3-seed results

**Branch:** `perclient-metric`. **Hardware:** GCP L4 VM. **Setup:** CIFAR-10, Dir(0.1), 20 clients, 200 rounds, 5 local epochs, batch 64, single sudden drift at round 100.

**Fix:** removed `/ cur_round` divisor from `current_lr` in `run_adaptive_fedavg`. Bias correction on the three EMAs (1 - beta^t) retained. Two independent re-implementations of Adaptive-FedAvg disagree on whether `/cur_round` belongs: FedCCFA has it, Saile et al. 2024 does not. The divisor contradicts the algorithm's stated purpose (raise LR when update-variance spikes at drift) and is removed here. Original Canonaco IJCNN 2021 paper is paywalled and was not directly verified.

**Branch hygiene:** the broken results (FedCCFA-faithful) are kept on disk - see Section 5 - for the record.

## 1. LR sweep (seed 0, FedDrift-style: search 10^-a)

Search criterion: highest post-drift stable global accuracy. Single seed for the sweep.

| `client_init_lr` | Pre | Dip | **Stable** | Per-client stable | Notes |
|---:|---:|---:|---:|---:|---|
| 0.1 | 0.1000 | 0.0000 | **0.1000** | 0.1000 | diverged (stuck near random-guess 10%) |
| 0.01 | 0.7008 | 0.1279 | **0.5812** | 0.5413 | **selected** |
| 0.001 | 0.6138 | 0.0915 | **0.5423** | 0.5167 |  |
| 0.0001 | 0.3730 | 0.0118 | **0.3654** | 0.3630 | undertrained |

**Selected:** `client_init_lr = 0.01`. This matches the standard FedAvg LR; the prior broken-version run also used 0.01, confirming the LR was already correct -- the bug was purely the `/cur_round` divisor.

## 2. Corrected AdaptiveFedAvg - 3-seed results (lr=0.01)

| Metric | Pre | Dip | Stable |
|---|---:|---:|---:|
| Global accuracy | 0.7118 ± 0.0079 | 0.1208 ± 0.0183 | 0.5916 ± 0.0164 |
| Per-client gen acc | 0.7118 ± 0.0079 | 0.1665 ± 0.0092 | 0.5511 ± 0.0098 |

## 3. Before/after on global stable accuracy

Same lr=0.01, same setup, same 3 seeds. Only the `/cur_round` divisor differs.

| Version | Stable global (mean +/- std) | Stable per-client (mean +/- std) |
|---|---:|---:|
| **Broken** (FedCCFA-faithful, with `/cur_round`) | 0.4941 ± 0.0302 | 0.4653 ± 0.0173 |
| **Corrected** (without `/cur_round`) | 0.5916 ± 0.0164 | 0.5511 ± 0.0098 |
| **Lift from fix** | **+9.75pp** | **+8.58pp** |

Per seed (clearer):

| Seed | broken stable global | fixed stable global | lift | broken pc stable | fixed pc stable | pc lift |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4656 | 0.5812 | +11.56pp | 0.4441 | 0.5413 | +9.71pp |
| 1 | 0.5360 | 0.6148 | +7.88pp | 0.4866 | 0.5645 | +7.79pp |
| 2 | 0.4808 | 0.5789 | +9.81pp | 0.4651 | 0.5475 | +8.24pp |

## 4. Corrected AdaptiveFedAvg vs FedAvg (3-seed)

Now that the LR scheduler can actually react, how does corrected AdaptiveFedAvg compare to plain FedAvg on the same 3 seeds?

| Metric | FedAvg | Corrected AdaptiveFedAvg | Delta |
|---|---:|---:|---:|
| Global pre | 0.7127 ± 0.0045 | 0.7118 ± 0.0079 | -0.09pp |
| Global dip | 0.1153 ± 0.0137 | 0.1208 ± 0.0183 | +0.55pp |
| Global stable | 0.5958 ± 0.0129 | 0.5916 ± 0.0164 | -0.42pp |
| Per-client stable | 0.5525 ± 0.0065 | 0.5511 ± 0.0098 | -0.14pp |

Per-seed delta on global stable (corrected AdaptiveFedAvg - FedAvg): ['-0.74pp', '+0.08pp', '-0.59pp'] -> mean **-0.42pp**, std 0.36pp.
Per-seed delta on per-client stable: ['-0.64pp', '+0.27pp', '-0.07pp'] -> mean **-0.14pp**, std 0.38pp.

## 5. Files on record

- `runs/2026-06-09-adaptive-lr-sweep/lr_*/results_AdaptiveFedAvg.csv` - LR sweep (4 LRs, seed 0).
- `runs/2026-06-09-adaptivefedavg-fix/seed{0,1,2}/results_AdaptiveFedAvg.csv` - corrected 3-seed.
- `runs/2026-06-09-adaptivefedavg-fix/SUMMARY.md` - this report.
- `runs/2026-06-08-perclient-smoke/adaptive/`, `runs/2026-06-08-perclient-3seed/seed{1,2}/adaptive/` - **broken (FedCCFA-faithful) results - KEPT on record per instruction.**
- `logs/adasweep_*.log` (LR sweep stdout), `logs/adfix_seed{1,2}.log` (corrected runs stdout).

## 6. Honest reading

1. **The fix recovered AdaptiveFedAvg.** Stable global jumped from 0.4941 (broken) to 0.5916 (fixed), a +9.75pp lift averaged across 3 seeds, with the per-seed lift consistent across seeds (see Section 3).
2. **Corrected AdaptiveFedAvg is essentially tied with FedAvg** on global stable accuracy (0.5916 vs 0.5958, delta -0.42pp +/- 0.36pp). The per-seed delta sign agreement and magnitude vs std suggest this is within seed noise. The adaptive LR mechanism does not produce a measurable improvement over plain FedAvg at this operating point (Dir(0.1), single sudden drift).
3. **What the broken version really measured.** The broken AdaptiveFedAvg's `/cur_round` divisor collapsed the LR to ~1e-4 by round 100, so the method couldn't react to drift at all -- it effectively trained with a near-zero LR for the second half of training. Comparing it against FedAvg or against OurMethod was comparing them against a crippled baseline, not against the AdaptiveFedAvg algorithm.
4. **Implication for the broader 3-seed table.** The 3-seed per-client comparison previously included broken AdaptiveFedAvg at stable global 0.4941 +/- 0.0302. With the fix that becomes 0.5916 +/- 0.0164. The OurMethod / FedAvg / FedAvgPlus1 numbers in that table are unaffected -- only AdaptiveFedAvg's row changes. The headline finding (OurMethod within noise of FedAvg on per-client; FedAvgPlus1 control crashes at Dir(0.1)) is unchanged. AdaptiveFedAvg simply joins the no-measurable-improvement-over-FedAvg cluster.

**Bottom line:** at this operating point (Dir(0.1), single sudden drift), neither AdaptiveFedAvg (corrected) nor OurMethod produces a measurable per-client gain over plain FedAvg. Reporting both with the correct AdaptiveFedAvg numbers, not the FedCCFA-faithful crippled ones, restores credibility to the comparison.