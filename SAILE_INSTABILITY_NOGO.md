# Saile-instability headroom gate — NO-GO

**Date:** 2026-06-24
**Verdict:** FAIL. Direction killed.

## Question

Does Saile's per-client 3-EMA-on-loss dynamic LR (FLTA 2024), **even at its best-tuned stable setting**, underperform FedAvg in some drift regime — leaving headroom for a more robust per-client adapter to beat both?

## Setup

CIFAR-10, CifarCNN (107,690 params), 20 clients, distributed cohort drift (A/B/C), 200 rounds, single seed (seed 0). Saile @ `initial_lr=0.01` (best stable from prior canonical work). FedAvg @ `lr=0.01`. Same harness, same partitioning, same seeds.

## Sweep table

Stable window: mean of last 10 rounds. Dip: pre-drift baseline minus minimum global_acc in the 10 rounds after first drift.

| Regime | FedAvg global stable | Saile global stable | Δ global | Δ per-cli | Δ global dip |
|---|---:|---:|---:|---:|---:|
| Canonical Dir(0.1) sudden | 0.5887 | 0.5844 | **−0.43pp** | +0.01pp | −0.25 |
| Dir(0.5) sudden | 0.6661 | 0.6736 | +0.76pp | +0.19pp | +0.38 |
| Recurrent alternating | 0.7046 | 0.7108 | +0.62pp | +0.62pp | −1.37 |
| Aggressive permutation | 0.1512 | 0.1551 | +0.39pp | +0.47pp | −0.27 |
| Covariate drift | 0.6770 | 0.6819 | +0.49pp | +0.21pp | −0.78 |

All deltas within **±1pp** single-seed noise. No regime where best-Saile clearly underperforms FedAvg.

## Headline

- **All 5 regimes tie within noise.** No exploitable gap.
- **The regime hypothesized to break Saile** (recurrent alternating — its loss-EMA was supposed to mis-track when drift reverses) showed Saile slightly **better** (+0.62pp both metrics), the opposite direction predicted by the headroom claim.

## Mechanistic reason

Saile's bias-corrected EMA LR is bounded by `initial_lr=0.01` (= FedAvg's LR) with a short cooldown (~5–10 rounds). At safe tuning the per-client LR cap matches FedAvg, so even when the EMA correctly detects a loss spike at drift, the response is too gentle to change the final accuracy in either direction. The adapter mechanically fires (LR mean snaps from ~0.006 to ~0.009 at drift in every regime, then cools), but the firing is bounded by the same LR that FedAvg uses statically. **Per-client triggers appear confound-robust by construction** — and at a safe cap they're also too tame to help.

## Why we're not running multi-seed

3 seeds would tighten the ±1pp band to ~±0.3pp. Even if every regime emerged statistically distinguishable, a sub-1pp gap is not an exploitable research-method headroom. Compute is better spent on the next candidate direction.

## Provenance

- Pair-1 (dir05 + recurrent): 2026-06-24 09:00–09:38 UTC, GCP L4 (fl-experiment, northamerica-northeast2-b). CSVs under `runs/2026-06-24-saile-gate/{dir05,recurrent}/`.
- Pair-2 (aggressive + covariate): 2026-06-24 09:49–10:28 UTC, same VM. CSVs under `runs/2026-06-24-saile-gate/{aggressive,covariate}/`. Harness was the temp local merge `saile-gate-tmp` (main + aggressive-concept-drift-test + covariate-drift-test); not pushed.
- Reference FedAvg baselines: canonical (`runs/2026-06-08-perclient-smoke/fedavg/`), Dir(0.5) (`runs/2026-06-11-dir05-test/fedavg/`), recurrent (`runs/2026-06-13-recurrent-gain-test/fedavg/`), aggressive + covariate FedAvg pulled from `aggressive-concept-drift-test` and `covariate-drift-test` branch CSVs into scratch.
- Sweep analysis: `saile_gate_analysis.py` reproduces the table from the per-round CSVs.
- VM stopped after pair-2 completed (TERMINATED).

## What the next candidate should look like

Per-client triggers tied FedAvg here because they're confound-robust. The documented confounding-weakness target is methods whose trigger is **update-magnitude / server-influenced** (e.g. Flash) — those over-react to static heterogeneity, not to genuine drift. That's the next gate.
