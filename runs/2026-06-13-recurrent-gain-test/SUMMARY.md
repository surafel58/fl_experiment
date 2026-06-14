# Recurrent-gain hypothesis test - go/no-go, single seed

**Branch:** `recurrent-gain-test`. **Hardware:** GCP L4 VM. **Setup:** CIFAR-10, Dir(0.1), 20 clients, 200 rounds, 5 local epochs, batch 64, CifarCNN, seed 0. **Drift:** alternating, events at rounds 40/80/120/160, same cohort swap (A:1↔2, B:3↔4, C:5↔6) at every event - involution toggles the concept canonical → swapped → canonical → swapped → canonical.

## Verdict: **NO-GO**

OurMethod is tied with FedAvg on all relevant metrics. Recurrent drift does NOT rescue a separation. **Additionally, the detection mechanism is structurally asymmetric** - it fires on 2 of 4 events.

## Phase windows

| Phase | Rounds | Concept |
|---|---|---|
| 1 | 0-39 | canonical |
| 2 | 40-79 | swapped |
| 3 | 80-119 | canonical (1st swap-back) |
| 4 | 120-159 | swapped |
| 5 | 160-199 | canonical (2nd swap-back) |

## The numbers that matter

### 1. OurMethod − FedAvg deltas (single seed)

| Metric | FedAvg | OurMethod | Δ (OurM − FA) |
|---|---:|---:|---:|
| Mean post-drift global [40, 199] | 0.6475 | 0.6491 | **+0.16pp** |
| Mean post-drift per-client [40, 199] | 0.6261 | 0.6263 | **+0.02pp** |
| Final-phase stable global [190, 199] | 0.7046 | 0.7073 | **+0.27pp** |
| Final-phase stable per-client [190, 199] | 0.7046 | 0.7073 | **+0.27pp** |
| **IMMEDIATE forgetting** = peak(P1) − mean(80, 89) | −0.0185 | −0.0165 | **+0.19pp (slightly worse)** |
| RESIDUAL forgetting = peak(P1) − mean(110, 119) | −0.0262 | −0.0244 | **+0.18pp (slightly worse)** |

All Δ are well within the seed-noise floor (~0.5-1.5pp from prior 3-seed studies). On all 4 accuracy/forgetting axes, OurMethod is statistically tied with FedAvg. **The single number you cared most about** (immediate forgetting) shows OurMethod forgetting SLIGHTLY MORE, not less.

### 2. OurMethod detection at each event (the structural finding)

| Event | Round | Direction | Flagged peak | Per-round window |
|---|---:|---|---:|---|
| 0 | 40 | canonical → swapped | **6/20 (30%)** | `[6, 6, 6, 6, 6, 4, 2, 0, 0, 0]` |
| 1 | 80 | swapped → canonical | **0/20 (0%)** | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` |
| 2 | 120 | canonical → swapped | **5/20 (25%)** | `[5, 5, 5, 5, 3, 1, 1, 1, 1, 1]` |
| 3 | 160 | swapped → canonical | **0/20 (0%)** | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` |

**The detector is asymmetric across drift direction.** It fires reliably on `canonical → swapped` transitions (6/20 and 5/20 — same magnitude as Dir(0.1) single-drift, ~30% recall), but NEVER on `swapped → canonical` transitions despite identical loss-spike magnitude.

Why: the EMA detector keys on per-layer weight-change magnitude *relative to recent variance*. At round 80, the model has been training on swapped concept for 40 rounds and the recent variance is still elevated from event-0 drift recovery. The swap-back produces a similar absolute spike but a smaller *relative* spike → doesn't exceed τ. By round 160, same story (recent variance still elevated from event-2 recovery).

## What this means for the recurrent-gain hypothesis

The hypothesis was: under recurrent drift, OurMethod's mechanism fires often, FedAvg degrades cumulatively, gap opens.

What actually happened:
1. **The mechanism does NOT fire often** under alternating drift. It only fires on the "into swapped" half of events (2 of 4 = 50%). The "swap back" events are silent.
2. **FedAvg does not degrade cumulatively** in any visible way - its final-phase stable (0.7046) is higher than its phase-1 peak (0.6848), because the model is still in the rising-curve regime at 200 rounds.
3. **Both methods recover from each drift faster than the initial learn-up** in phase 1, which is why both "forgetting" metrics are NEGATIVE (the model is BETTER at canonical after a swapped phase than it was in phase 1).

The premise of the hypothesis (FedAvg fails harder under recurrent drift) is itself wrong in this setting. Drift here is "training-data label swap" of 3 disjoint class pairs, not catastrophic forgetting in the continual-learning sense. The shared lower-layer features keep improving across phases, and both methods benefit equally.

## Decision

**NO-GO on a full multi-seed multi-method recurrent-drift run.** Three reasons:

1. **Accuracy deltas are noise** on every metric (max +0.27pp at final stable, with no consistent positive direction).
2. **Immediate forgetting** (the discriminator you asked for) shows OurMethod slightly WORSE, not better.
3. **The detection mechanism is mechanically incapable of firing on swap-back events** in the alternating setting, so it operates as a single-shot mechanism even when drift recurs. This is a structural property of EMA-relative-variance detection, not a hyperparameter issue.

A full multi-seed run would just confirm these single-seed tied results at slightly tighter error bars. The mechanism is doing what it can mechanistically do (fire on the first drift direction, then go silent), and that's not enough to separate from FedAvg here.

## What a "real recurrent-drift" experiment would need (for the record)

If we wanted to genuinely stress-test the mechanism on recurrent concepts, we would need either:
- A detector that resets its EMA after a flagged event (so the next drift produces a fresh relative spike).
- Slower recurrence (so the model fully restabilizes between events, making both directions detectable).
- A different drift type where the recurring concepts are not perfectly anti-symmetric (e.g., 4 distinct concept rotations instead of 2 alternating).

None of these are in scope here. They're notes for any future recurrent-drift work.
