# Saile-instability headroom gate — verdict: **FAIL (NO-GO)**

Date: 2026-06-24. Seed: 0 (single-seed scan; no multi-seed confirm planned).
Method index: 6 (Saile). Saile init_lr = 0.01 (best stable from prior canonical sweep).
FedAvg baseline LR = 0.01.

## Hypothesis being tested

> Saile's per-client 3-EMA-on-loss dynamic LR, **even at its best-tuned stable setting**, underperforms FedAvg in some regime (because its loss-EMA mis-tracks drift), leaving headroom for a more robust per-client adapter to beat **both** Saile and FedAvg there.

## Procedure

5 regimes scanned, each as 1 FedAvg run + 1 Saile (lr=0.01) run, 200 rounds, CIFAR-10, 20 clients, distributed cohort drift A/B/C:

| Regime | Drift type | First drift round | FedAvg source | Saile source |
|---|---|---:|---|---|
| Canonical Dir(0.1) sudden | pairwise label swap, sudden | 100 | runs/2026-06-08-perclient-smoke/fedavg | runs/2026-06-11-saile-3seed/seed0 |
| Dir(0.5) sudden | pairwise label swap, sudden | 100 | runs/2026-06-11-dir05-test/fedavg | runs/2026-06-24-saile-gate/dir05 |
| Recurrent alternating | swap, alternating @ 40/80/120/160 | 40 | runs/2026-06-13-recurrent-gain-test/fedavg | runs/2026-06-24-saile-gate/recurrent |
| Aggressive permutation | per-cohort full 10-label permutation, sudden | 100 | scratch (aggressive-concept-drift-test branch) | runs/2026-06-24-saile-gate/aggressive |
| Covariate drift | per-cohort image corruption (noise/blur/contrast), sudden | 100 | scratch (covariate-drift-test branch) | runs/2026-06-24-saile-gate/covariate |

## Result table

| Regime | FedAvg global stable | Saile global stable | Δ global | Δ per-cli | Δ global dip |
|---|---:|---:|---:|---:|---:|
| Canonical Dir(0.1) sudden | 0.5887 | 0.5844 | **−0.43pp** | +0.01pp | −0.25 |
| Dir(0.5) sudden | 0.6661 | 0.6736 | +0.76pp | +0.19pp | +0.38 |
| Recurrent alternating | 0.7046 | 0.7108 | +0.62pp | +0.62pp | −1.37 |
| Aggressive permutation | 0.1512 | 0.1551 | +0.39pp | +0.47pp | −0.27 |
| Covariate drift | 0.6770 | 0.6819 | +0.49pp | +0.21pp | −0.78 |

Stable window: mean of last 10 rounds. Dip: pre-drift baseline minus minimum global_acc within 10 rounds after first drift.

## Saile per-client LR trajectory around drift (proof that the adapter responds)

Across all 5 regimes, Saile's bias-corrected EMA loss adapter detects the drift round and re-inflates per-client LR to near the cap, then cools down within 5–10 rounds. Representative spike at the first drift round:

| Regime | LR mean (round 99) | LR mean (drift round) | LR mean (drift+5) |
|---|---:|---:|---:|
| Canonical (rd 100) | 0.0065 | 0.0092 | (cooled) |
| Dir(0.5) (rd 100) | 0.0059 | 0.0092 | (cooled) |
| Recurrent (rd 40) | 0.0097 | 0.0097 (already saturated pre-drift) | 0.0094 |
| Aggressive (rd 100) | 0.0065 | **0.0100** (cap) | 0.0097 |
| Covariate (rd 100) | 0.0073 | 0.0091 | 0.0076 |

The adapter is mechanically working. It just doesn't change the final accuracy enough to either help or hurt.

## Verdict & why

**FAIL.** No regime shows best-Saile clearly underperforming FedAvg. All 5 deltas fall within ±1pp on both global and per-client stable accuracy — single-seed FL noise floor on this benchmark. The recurrent regime (the most likely to break Saile's loss-EMA per the hypothesis) actually shows Saile **slightly better** (+0.62pp both metrics), the opposite of what the headroom claim predicted.

Mechanistic read: Saile's per-client LR variability is bounded by `initial_lr=0.01` (the cap) and the bias-corrected EMA cools back to a base trajectory within 5–10 rounds. With that cap = FedAvg's LR, the "responsiveness" is effectively a no-op around drift — too gentle to harm, too gentle to help.

## Why we're killing the direction, not running multi-seed confirm

Multi-seed at 3 seeds would tighten the ±1pp band to ~±0.3pp. Even if every regime came out significant in some direction, the gap is fundamentally too small to be exploitable headroom for a research method. A robustified per-client adapter would at best recover a sub-1pp gap — not worth the implementation+writeup investment vs. directions with larger headroom.

## Provenance

- Harness used for aggressive/covariate runs: temp branch `saile-gate-tmp` (local-only, not pushed), which is `main` + `aggressive-concept-drift-test` + `covariate-drift-test` merged. Conflicts (`apply_test_drift_event`, helper-block ordering, argparse) resolved in commit `31855f4`.
- Pair-2 runs: 2026-06-24 09:49 UTC → 10:28 UTC on GCP L4 (fl-experiment, northamerica-northeast2-b), VM stopped after.
- Sweep analysis script: `saile_gate_analysis.py`. Reproduces the table above by reading the per-round CSVs.
- Saile method implementation untouched between pair-1 and pair-2; both pairs read the same `methods/Saile.py` semantics (only the drift harness differs).
