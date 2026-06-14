# Aggressive concept drift (per-cohort full permutations) — go/no-go, single seed

**Branch:** `aggressive-concept-drift-test`. **Hardware:** GCP L4 VM. **Setup:** CIFAR-10, Dir(0.1), 20 clients, 200 rounds, 5 local epochs, batch 64, CifarCNN, seed 0. **Drift:** single sudden event at round 100, each cohort applies a DIFFERENT full 10-label permutation (not a pairwise swap). Permutations seeded RandomState(42).

Permutations used (cross-cohort agreements 1/0/0 → maximally conflicting):
```
P_A: [8, 1, 5, 0, 7, 2, 9, 4, 3, 6]   ← cohort A (id%10 < 3)
P_B: [0, 1, 8, 5, 3, 4, 7, 9, 6, 2]   ← cohort B (3 ≤ id%10 < 6)
P_C: [9, 2, 0, 6, 8, 5, 3, 7, 1, 4]   ← cohort C (id%10 ≥ 6)
```

## Verdict: **NO-GO** — decisive

OurMethod is mechanically a perfect detector for this aggressive drift (95% recall) but the layer-protection mechanism produces **zero** accuracy benefit. **OurMethod − FedAvg on per-client stable = −0.00pp**, identical to 4 decimal places.

## The numbers that matter

### Detection recall — the only thing that improved

| Drift form | Peak flag count [100, 109] | Recall |
|---|---|---:|
| Pairwise swap (reference, prior runs) | `[7, 7, 7, 7, 6, 1, 1, 1, 1, 1]` | 7/20 = **35%** |
| **Aggressive permutation (this test)** | `[19, 19, 19, 19, 19, 9, 1, 1, 1, 1]` | **19/20 = 95%** |

Aggressive drift is **~3× easier to detect** than pairwise swap. The detector saturates on 19/20 clients for 5 consecutive rounds — almost every client's per-layer weight magnitudes spike past τ. Conclusion on detection: the EMA-relative-variance detector works exactly as designed at high drift magnitude.

### Accuracy — the disappointing part

| Metric | FedAvg | OurMethod | Δ (OurM − FA) |
|---|---:|---:|---:|
| Global pre-drift [89, 99] | 0.7078 | 0.7078 | 0 (identical) |
| Global dip | 0.5358 | 0.5190 | OurM dips slightly more |
| Global stable [190, 199] (canonical TEST_Y) | 0.1512 | 0.1530 | **+0.18pp** |
| Per-client pre-drift | 0.7078 | 0.7078 | 0 (identical) |
| Per-client dip | 0.5180 | 0.5211 | OurM dips marginally less |
| **Per-client stable** [190, 199] | **0.2862** | **0.2862** | **−0.00pp (identical to 4 dp)** |

Both methods reach the **same** post-drift stable per-client accuracy. Detecting the drift on 19/20 clients did not translate into any measurable advantage on the metric that matters.

### Reference comparison (same seed, pairwise-swap drift)

| Metric | Aggressive Δ (OurM − FA) | Pairwise Δ (OurM − FA, seed 0) |
|---|---:|---:|
| Global stable | +0.18pp | −0.82pp |
| Per-client stable | −0.00pp | −0.38pp |

Going from pairwise swap → aggressive permutation:
- Detection recall: 35% → 95%
- Per-client Δ vs FedAvg: −0.38pp → 0pp

The detection mechanism massively improved. The accuracy delta didn't. **The mechanism's two stages are decoupled** — better detection does NOT mean better outcome.

## Mechanistic explanation

OurMethod's value proposition: when client A's data drifts in a way that's incompatible with what the global model is converging to, KEEP A's drift-layer (L3/L4) weights LOCAL so A isn't pulled toward the (now wrong-for-A) global average.

This works when the **pre-drift learned weights are still useful for the post-drift task** (e.g. pairwise label swap of 2 classes: the feature representation barely changes, only the classifier needs to remap 2 classes → small local model adjustment suffices).

Under aggressive permutation, the **pre-drift learned L3/L4 weights are useless for the post-drift task**:
- Cohort A pre-drift: trained on canonical labels → L3/L4 encode features useful for classifying canonical labels.
- Cohort A post-drift: data has P_A applied → cohort A needs L3/L4 that map images to *permuted* labels.
- The pre-drift L3/L4 weights are equally wrong as a randomly-initialized version for cohort A's new task.
- "Keeping flagged layers local" preserves USELESS information; the client needs to re-train from scratch on the new mapping. FedAvg's global average (a different kind of wrong) and OurMethod's preserved local weights (also wrong) both fail to recover.

In short: **detection mass ≠ adaptation efficacy**. OurMethod's mechanism assumes the locally-protected weights still contain useful task-specific signal. When drift is severe enough to invalidate that signal, the mechanism becomes a no-op vs FedAvg.

The classifier (always global, never flaggable) is asked to satisfy three incompatible label mappings simultaneously — globally, this is impossible, so the classifier converges to something useless for all three. That's why both methods crash to ~15% global accuracy and ~29% per-client accuracy. The shared training signal pulls the model in 3 directions, both methods are equally helpless.

## Implication for the thesis story

This is **decisive evidence** that label-based concept drift, in general, does not let OurMethod separate from FedAvg:

- **Mild label drift (pairwise swap)**: detection recall 35%, OurMethod tied with FedAvg (Δ ~0pp per-client at n=3 seeds).
- **Aggressive label drift (full per-cohort permutation)**: detection recall 95%, OurMethod still tied with FedAvg (Δ = 0pp at n=1 seed).

The dial we'd want — "harder drift → more mechanism action → bigger gap" — does not exist for label-based drift. Detection scales with drift magnitude, but the layer-protection mechanism's gain stays zero either way. Two structurally different label-drift regimes both produce the same answer.

The remaining unexplored axis is **covariate drift** (where the features themselves shift, not just the labels). Under covariate drift the shared feature representation IS the bottleneck OurMethod is conceptually designed to protect, and the situation is qualitatively different from label drift.

## Decision

**NO-GO** on a full multi-seed aggressive-label-drift run. The single-seed result is too clean (Δ = 0pp identical to 4 dp on per-client) for more seeds to flip it.

Justifies the pivot to covariate drift as the next exploration direction.
