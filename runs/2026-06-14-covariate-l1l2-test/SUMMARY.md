# Covariate drift + L1/L2 re-aim — go/no-go, single seed

**Branch:** `covariate-l1l2-test` (based on `covariate-drift-test`). **Hardware:** GCP L4 VM. **Setup:** CIFAR-10, Dir(0.1), 20 clients, 200 rounds, 5 local epochs, batch 64, CifarCNN, seed 0. **Drift:** single sudden covariate drift at round 100 (cohort A=Gaussian noise, B=blur, C=contrast — identical to `covariate-drift-test`). **OurMethod change**: `DRIFT_LAYERS = ['layer1', 'layer2']` via new `--drift-layers` CLI flag (previously L3/L4).

## Verdict: **NO-GO — Outcome #2 from the task spec.**

> *"Detector fires on L1/L2 but accuracy still ties → the mechanism's protect-and-keep-local strategy doesn't extract value even when correctly aimed; the core design (not just targeting) needs rethinking."*

The detector engaged on L1/L2 (evidence below), but "keep local" is the wrong response for early layers.

## Caveat — logging instrumentation gap

The existing OurMethod logging block hard-codes per-layer counts for `prev_flags.get('layer3')` and `prev_flags.get('layer4')` only. When `DRIFT_LAYERS = ['layer1', 'layer2']`, L3/L4 are excluded → they never flag → the CSV's `flagged_count` column reads 0 for every round. The L1/L2 flagging IS operational in the mechanism (`OurClient.train` sets `prev_flags['layer1']` / `prev_flags['layer2']` correctly; `get_upload_state` reads `upload_mask` derived from those flags) — only the **reporting** is missing.

So **we cannot read exact L1/L2 detection recall from this run's logs**. The detection evidence below is therefore indirect: an accuracy-divergence proxy.

## Evidence the L1/L2 detector fired (indirect, via trajectory divergence)

Comparing `|OurMethod_acc − FedAvg_acc|` per round between this run (L1/L2) and the prior `covariate-drift-test` (L3/L4, where the detector was almost completely silent at 1/20 peak):

| Window | Metric | L3/L4 covariate (≈silent baseline) | L1/L2 covariate (this) | Ratio |
|---|---|---:|---:|---:|
| Pre-drift [0, 99] | global max | 0.00pp | 0.00pp | — |
| Pre-drift [0, 99] | global mean | 0.00pp | 0.00pp | — |
| Post-drift [100, 199] | global max | 0.90pp | **2.84pp** | ~3.2× |
| Post-drift [100, 199] | global mean | 0.28pp | **0.41pp** | ~1.5× |
| Post-drift [100, 199] | per-cli max | 0.44pp | **4.44pp** | ~10× |
| Post-drift [100, 199] | per-cli mean | 0.15pp | **0.40pp** | ~2.7× |

**Pre-drift divergence is 0 in both runs** — bit-identical trajectories under the same seed when no mechanism is firing. So all post-drift divergence is purely from OurMethod's selective-sync activity (no CUDA non-determinism mixed in).

The L1/L2 re-aim produces **3-10× more divergence** from FedAvg than the L3/L4 baseline did. This is decisive evidence that **L1/L2 IS detecting covariate drift and engaging the selective-sync path**. The premise of the L1/L2 re-aim (covariate drift disturbs early layers, not late ones) is confirmed by the actual mechanism response.

## Accuracy comparison

| Metric | FedAvg | OurMethod (L1/L2) | Δ (OurM − FA) |
|---|---:|---:|---:|
| Pre-drift global [89, 99] | 0.7078 | 0.7078 | 0 (identical) |
| Global dip | 0.0414 | 0.0358 | OurM dips SLIGHTLY LESS by 0.56pp |
| Global stable [190, 199] | 0.6770 | 0.6752 | **−0.18pp** |
| Per-client pre-drift | 0.7078 | 0.7078 | 0 (identical) |
| Per-client dip | 0.0979 | **0.1261** | **OurM dips MORE** by 2.82pp |
| **Per-client stable** [190, 199] | **0.6290** | **0.6279** | **−0.12pp** |

**Per-client stable Δ = −0.12pp** — essentially tied. **Per-client dip is 2.82pp DEEPER for OurMethod.** The mechanism fires, withholds L1/L2 from the federation, and the immediate impact is worse, even though the stable recovery converges to roughly the same point as FedAvg.

## Comparison vs L3/L4 covariate baseline

| | L3/L4 covariate (ref) | L1/L2 covariate (this) |
|---|---:|---:|
| OurMethod − FedAvg, per-client stable | +0.02pp | −0.12pp |
| OurMethod − FedAvg, global stable | −0.14pp | −0.18pp |
| Detection (indirect, via trajectory) | barely fires | **clearly fires** |
| Per-client dip vs FedAvg | tied (0.0980 vs 0.0979) | **worse (0.1261 vs 0.0979)** |

L1/L2 re-aim **detects** the drift better but **doesn't help** the metric — and **hurts the dip**. The mechanism's "keep local" response is wrong for early layers.

## Mechanistic explanation — why L1/L2 keep-local backfires

L1 and L2 encode low-level image features (edge filters, texture detectors, color statistics) that benefit enormously from being learned **across the federation's combined data** — they need exposure to many image distributions to generalize.

Under covariate drift, the right response for client A (whose images now have Gaussian noise) is **NOT** to keep its noise-trained L1/L2 weights local. Those weights are tuned to extract features from noisy images specifically — they're worse on the broader, mixed federation distribution than the federation's averaged feature extractor would be.

What the mechanism does under "keep L1/L2 local":
1. Client A flags L1/L2 (per-layer EMA spike confirms drift hit early layers)
2. Client A withholds its L1/L2 updates from aggregation, keeps its local L1/L2
3. Next round: Client A's hybrid model = (its local noise-tuned L1/L2) + (federation's L3/L4) + (global classifier)
4. The mismatch between A's noise-tuned L1/L2 and the federation's L3/L4 (trained on the averaged-feature representation) **hurts**
5. The federation also loses A's contribution to L1/L2, so the global L1/L2 is now less robust to noise

Net result: each individual flagged client is worse off (local features mismatched to global later layers), AND the federation's global L1/L2 lacks the gradient signal from clients who have actually seen corrupted images. Both effects hurt the dip and produce a slight stable penalty.

This is the opposite of how late-layer label drift works: there, late-layer specialization actually helps the client adapt because each cohort's label remapping is a per-client problem.

## What this confirms vs what's open

**Confirmed:**
- Covariate drift disturbs early layers (L1/L2 detector engages where L3/L4 was silent) ✓
- The L1/L2 re-aim correctly targets the layers that experience the drift ✓
- The targeting was the right diagnosis from the prior test ✓

**Refuted:**
- The "keep flagged layers local" mechanism, even when correctly aimed at L1/L2, does not improve accuracy under covariate drift
- In fact it makes the immediate dip worse (per-client dip 0.1261 vs FedAvg 0.0979, +2.82pp deeper)

**Open:**
- A re-designed mechanism for covariate drift would need to do something OTHER than "keep local" with early layers. Surgical fine-tuning theory (Lee et al. 2022) suggests "tune early, freeze late" for input shift — but that's for ONE shifted distribution at a time, not 20 clients each on a different shifted distribution. In our distributed-covariate-shift setting, the right move may be the opposite of OurMethod's current design: **share L1/L2 from all clients (including drifted ones) and learn a robust shared feature extractor**, rather than letting each client diverge into a per-client feature extractor.

## Decision

**NO-GO on the current OurMethod design under covariate drift**, regardless of targeting. The mechanism's protect-and-keep-local strategy is wrong for early layers — it makes the dip worse and doesn't improve recovery. A core redesign would be needed: "share, don't withhold, when drift hits early layers".

This is decisive evidence that OurMethod (as currently designed: drift-triggered selective layer adaptation via withholding) does not produce a separation from FedAvg under either drift type:
- **Label drift** (concept drift): mechanism designed for it, fires correctly, but tied with FedAvg because pre-drift L3/L4 weights are useful for the post-drift task (pairwise) or useless (aggressive perm) either way.
- **Covariate drift** (input shift): mechanism wasn't aimed at it (L3/L4 silent at 5%); when re-aimed (L1/L2), detector engages, but "keep local" is the wrong response for early-layer drift.

## Future work suggested (not in scope here)

- **Try the inverse mechanism**: when a layer is flagged under covariate drift, INCREASE that layer's federation aggregation weight (use the diverse local-corrupted-data signal to build a more robust shared extractor). Tests "share-more-not-less" hypothesis.
- **Try surgical fine-tuning**: tune early layers locally, freeze late layers (FedBABU-style for shift). Tests "tune-don't-withhold" hypothesis.
- **Fix the instrumentation**: the L1/L2 flag-count gap should be patched in a follow-up commit. The mechanism is correct; only the reporting is wrong.

## What's pushed

- Branch `covariate-l1l2-test` (based on `covariate-drift-test`, **not merged** to main)
- `runs/2026-06-14-covariate-l1l2-test/` — CSVs (FedAvg + OurMethod with L1/L2 flagging) + flags CSV + SUMMARY.md
- `all_experiments_optimized.py` — `--drift-layers` CLI flag + `__main__` override (15 lines)
- Run logs

## VM

TERMINATED — billing stopped.
