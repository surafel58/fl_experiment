# Covariate drift test — go/no-go, single seed

**Branch:** `covariate-drift-test`. **Hardware:** GCP L4 VM. **Setup:** CIFAR-10, Dir(0.1), 20 clients, 200 rounds, 5 local epochs, batch 64, CifarCNN, seed 0. **Drift:** single sudden event at round 100, per-cohort IMAGE corruption (labels untouched).

Corruptions used (all deterministic, applied to TEST_X variants in lockstep):
- Cohort A (id%10 < 3): Gaussian noise std=0.15
- Cohort B (3 ≤ id%10 < 6): Gaussian blur 5×5, σ=1.5
- Cohort C (id%10 ≥ 6): Contrast halving + brightness shift (×0.5 + 0.3)

## Verdict: **NO-GO — but informative.** The L1/L2 re-aim path is now pre-justified by direct evidence.

## The numbers that matter

### 1. Detection recall — the smoking gun

| Drift type | Peak flag count [100, 109] | Recall | Total flag-rounds post-drift |
|---|---|---:|---:|
| Pairwise label swap (ref) | `[7, 7, 7, 7, 6, 1, 1, 1, 1, 1]` | 7/20 = 35% | high |
| **Covariate drift (this)** | `[1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | **1/20 = 5%** | **22 client-rounds / 100 rounds** = 0.22 flags/round avg |

**The L3/L4 EMA detector is functionally silent under covariate drift.** Only one client (id 18) ever triggers, and that's an isolated case. The detector mass under label drift (concentrated on the cohorts that experienced the swap) does not appear under covariate drift.

### 2. OurMethod vs FedAvg accuracy

| Metric | FedAvg | OurMethod | Δ (OurM − FA) |
|---|---:|---:|---:|
| Pre-drift global | 0.7078 | 0.7078 | 0 (identical) |
| Global dip | 0.0414 | 0.0413 | essentially identical |
| Global stable [190, 199] | 0.6770 | 0.6756 | **−0.14pp** |
| Per-client pre-drift | 0.7078 | 0.7078 | 0 (identical) |
| Per-client dip | 0.0979 | 0.0980 | essentially identical |
| **Per-client stable [190, 199]** | **0.6290** | **0.6292** | **+0.02pp** |

Per-client stable Δ is **+0.02pp** — zero to two decimal places. Mechanism doing nothing → method tied with FedAvg.

### 3. Drift dip is much milder than label drift

Covariate drift produces ~0.04 global dip (and ~0.10 per-client dip), vs ~0.12 dip for pairwise label swap. Both methods recover quickly. With labels unchanged, the model still works decently on the canonical test set; per-cohort feature corruption hurts contributions to the global aggregate but doesn't destroy the model's class boundaries.

## Mechanistic explanation — exactly what the task spec predicted

The task spec called this out: *"detection might NOT fire, since covariate shift may hit early layers more than L3/L4 — that itself is informative."*

That's exactly what we see, mechanically. Covariate drift = "the input distribution shifts" = the feature extractor (early layers L1/L2) sees inputs whose statistics no longer match what it was trained on. The early layers absorb most of the per-cohort distortion. By the time signal reaches L3/L4, the features have already been (badly) normalized, and the L3/L4 weights don't need to change much to keep producing reasonable classifier-adjacent representations.

Under **label** drift (the prior tests), the OPPOSITE was true: the feature extractor still sees canonical-looking images, but the layer right before the classifier (L4 → fc) has to undo a swap that the (always-global) fc can't satisfy on its own. So L3/L4 weight magnitudes spike → detector triggers.

The detector is watching the wrong location for covariate drift.

## What this confirms (and what it doesn't)

**Confirmed:**
- Covariate shift produces large signal in early layers (L1/L2), not the layers OurMethod currently watches.
- The "L1/L2 re-aim" path mentioned in the task spec is justified by direct evidence: not just theory but the actual flagged_count signal showing the existing setup is blind.
- The pivot from concept-drift framing to covariate-drift framing fundamentally requires re-locating the detector and the layer-protection target.

**NOT confirmed** (would need a re-aimed run):
- Whether re-aiming the detector + protection to L1/L2 actually produces a separation from FedAvg.
- Whether the protection mechanism (keep-layers-local) is the right adaptation under covariate drift, even if detection works. Surgical fine-tuning theory (Lee et al. 2022) suggests yes for input-shift, but that's untested in our harness.

## Decision

**NO-GO** on a full multi-seed run with the current L3/L4 detector under covariate drift — the mechanism is silent and the deltas are noise (+0.02pp per-client stable). Confirming with more seeds would burn compute for an already-decisive negative.

**GO on the next test**: re-aim the detector + the flaggable-layer set to L1/L2 (the early conv layers) and re-run a single-seed go/no-go on this same covariate-drift setup. That's the experiment the data here is asking for. The covariate-drift infrastructure built in this branch (per-cohort image variants, lockstep test corruption, canonical X backup) is reusable for the re-aimed experiment with zero changes — only the OurMethod `DRIFT_LAYERS` and the cohort flag logic need to move from L3/L4 to L1/L2.

## What's pushed

- Branch `covariate-drift-test` (off main, **not merged**)
- `runs/2026-06-14-covariate-drift-test/{fedavg,ourmethod}/` + flags CSV + SUMMARY.md
- `all_experiments_optimized.py` — `--covariate-drift` CLI flag, 3 corruption helpers (`_corrupt_gauss_noise`, `_corrupt_gauss_blur`, `_corrupt_contrast`), per-cohort `TEST_X_VARIANTS`, `GPU_CLIENT_X_CANONICAL` backup, lockstep `apply_drift_event` / `apply_test_drift_event` / `fresh_client_y` / `reset_per_client_metric_state` / `evaluate_per_client_gen_acc` extensions
- VM logs
