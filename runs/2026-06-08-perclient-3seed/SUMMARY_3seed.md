# Per-client comparison — 3 seeds (0, 1, 2)

**Setup:** CIFAR-10, Dir(0.1), 20 clients, 200 rounds, 5 local epochs, batch 64, single sudden drift at round 100.
Per-client metric = FedCCFA's per-client generalized accuracy (faithful port, validated bit-exact pre-drift at Step A).

**Hardware:** GCP L4 VM, `cudnn.benchmark = True` (convs non-deterministic across processes).

**Seeds aggregated:** [0, 1, 2]  (n = 3)

All values mean ± std across seeds; std is population (ddof=0).
`pre` = mean rounds [89, 99]; `dip` = pre − min(rounds [100, 109]); `stable` = mean(rounds [190, 199]).

## Global accuracy (canonical CIFAR-10 test set, undrifted labels)

| Method | Pre | Dip | Stable |
|---|---:|---:|---:|
| FedAvg | 0.7128 ± 0.0043 | 0.1154 ± 0.0136 | 0.5958 ± 0.0129 |
| Flash | 0.6845 ± 0.0095 | 0.1703 ± 0.0132 | 0.5899 ± 0.0192 |
| AdaptiveFedAvg | 0.5755 ± 0.0224 | 0.0507 ± 0.0264 | 0.4941 ± 0.0302 |
| FedAvgPlus1 | 0.7135 ± 0.0018 | 0.1170 ± 0.0135 | 0.5984 ± 0.0169 |
| OurMethod | 0.7117 ± 0.0055 | 0.1186 ± 0.0174 | 0.5917 ± 0.0180 |

## Per-client generalized accuracy (FedCCFA protocol)

| Method | Pre | Dip | Stable |
|---|---:|---:|---:|
| FedAvg | 0.7128 ± 0.0043 | 0.1642 ± 0.0071 | 0.5525 ± 0.0065 |
| Flash | 0.6845 ± 0.0095 | 0.1760 ± 0.0118 | 0.5457 ± 0.0125 |
| AdaptiveFedAvg | 0.5755 ± 0.0224 | 0.1146 ± 0.0151 | 0.4653 ± 0.0173 |
| FedAvgPlus1 | 0.5105 ± 0.0131 | 0.0930 ± 0.0049 | 0.4613 ± 0.0111 |
| OurMethod | 0.7117 ± 0.0055 | 0.1631 ± 0.0096 | 0.5508 ± 0.0086 |

## Key per-client deltas (stable, mean ± std across seeds)

| Comparison | Δ per-client stable |
|---|---:|
| OurMethod − FedAvg | -0.18pp ± 0.22pp |
| OurMethod − FedAvgPlus1 (control) | +8.95pp ± 0.75pp |

## Per-seed breakdown — per-client stable (rounds [190, 199] mean)

Shows seed-to-seed consistency vs noise for the three methods the comparison turns on.

| Seed | FedAvg | FedAvgPlus1 (control) | OurMethod | OurMethod − FedAvg | OurMethod − Control |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5476 | 0.4460 | 0.5438 | -0.38pp | +9.78pp |
| 1 | 0.5617 | 0.4720 | 0.5630 | +0.12pp | +9.09pp |
| 2 | 0.5482 | 0.4658 | 0.5455 | -0.27pp | +7.97pp |

## FedAvg vs FedAvgPlus1 — global-trajectory non-determinism gate (per seed)

With `cudnn.benchmark = True` and the same seed, two PyTorch processes running the same training do NOT produce bit-identical
global accuracy across processes (CUDA convs are non-deterministic). The +1-epoch fine-tune in FedAvgPlus1 does NOT touch the
global model. So the FedAvg vs FedAvgPlus1 global divergence is a pure CUDA-noise floor. Reported per seed so the per-client
control comparison is interpretable.

| Seed | max |Δglobal| | mean |Δglobal| | final |Δglobal| |
|---:|---:|---:|---:|
| 0 | 0.0217 (2.17pp) | 0.0052 (0.52pp) | 0.0008 (0.08pp) |
| 1 | 0.0223 (2.23pp) | 0.0052 (0.52pp) | 0.0068 (0.68pp) |
| 2 | 0.0218 (2.18pp) | 0.0047 (0.47pp) | 0.0066 (0.66pp) |

## Honest reading

### The two comparisons that matter

- **OurMethod vs FedAvg** (per-client stable): mean Δ = **−0.18pp**, std = **0.22pp** across n=3 seeds. |mean| ≤ std → effect is INSIDE the seed-noise floor. Per-seed deltas: **−0.38, +0.12, −0.27** — the sign FLIPS across seeds (one positive, two negative). At n=3 with sign disagreement, this is unambiguously noise around zero. **OurMethod's mechanism produces no measurable per-client gain over plain FedAvg at this operating point.**

- **OurMethod vs FedAvgPlus1 (control)** (per-client stable): mean Δ = **+8.95pp**, std = **0.75pp** across n=3 seeds. |mean| ≫ std → effect is OUTSIDE the seed-noise floor by ~12σ. Per-seed deltas: **+9.78, +9.09, +7.97** — all positive, all > 7.5pp. This is a real, consistent effect at n=3.

### What the two findings mean together

The control isn't measuring "what a competent drift-aware method does" — it's measuring "what trivial last-step personalization does to the per-client metric at our operating point". At Dir(0.1) with ~500 samples per client covering 2–3 classes, one extra local SGD epoch on top of the aggregated global model collapses the model toward the client's class subset and destroys per-client generalized accuracy on the cohort-swapped full test set. The +8.95pp gap of OurMethod over this control is consistent and large but is not evidence the mechanism works — it's evidence the mechanism doesn't catastrophically over-personalize, which plain FedAvg also does.

**The right comparison is OurMethod vs plain FedAvg, and there OurMethod produces no measurable gain on the per-client metric** (Δ −0.18 ± 0.22pp, signs disagreeing across seeds). At Dir(0.1) with single sudden drift at round 100, the selective per-layer adaptation does not improve per-client generalized accuracy in the stable post-drift regime measured here.

### What's still defensible to claim

- OurMethod is **not worse** than plain FedAvg on either metric (global or per-client). All deltas are within noise.
- OurMethod **avoids** the catastrophic per-client collapse seen with trivial local fine-tuning (FedAvgPlus1).
- The dip in global accuracy at drift is comparable to FedAvg's (mean 0.1186 vs 0.1154; std overlaps).
- The FedCCFA per-client metric port is correct (Step A bit-exact pre-drift identity; per-seed identity check confirms it in production runs).

### What is NOT defensible

- Claiming the selective-layer mechanism improves the per-client metric on this setup. It doesn't, at n=3.
- Comparing the +2.76pp Dir(0.5) "FedAvg_baseline.py" finding directly to FedAvgPlus1's −9pp here — they're different control variants and different operating points; the only common thread is "trivial local personalization is the mechanism, and at Dir(0.1) it goes the wrong direction".

### What this means for the thesis

This is a clean, single-event single-drift, Dir(0.1) result. It establishes that the per-client metric does not show OurMethod's mechanism working at this operating point. The mechanism may still show value at other points (recurrent drift, milder non-IID, different drift schedule, etc.), and the existing global-accuracy dip story for the τ=1.2 ablation is unchanged by this finding. But on the per-client metric as a standalone deliverable for the comparison sought here, the honest answer is: **no measurable gain at this operating point**, and that needs to be the reported finding.
