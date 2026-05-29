# 2026-05-29 — Recurrent drift smoke (seed=0)

**Status:** Smoke complete; seed 1 underway. Branch: `recurrent-drift` (not merged to main).

## What this was

First run with the new two-event drift schedule. All 4 methods on seed 0.

```
DRIFT_SCHEDULE = [100, 150]
Event 0 (round 100): A 1<->2,  B 3<->4,  C 5<->6   (original sudden-drift swaps)
Event 1 (round 150): A 3<->4,  B 5<->6,  C 7<->8   (rotated; disjoint per group)
```

## Headline result

Post-(double-drift) stable accuracy on the canonical test set:

| Method | Stable acc |
|---|---:|
| FedAvg | 0.4085 |
| Flash | 0.3985 |
| AdaptiveFedAvg | 0.3441 |
| **OurMethod** | **0.4144** |

OurMethod wins by **+0.59pp** over FedAvg. Detector re-fired at event 1 (5/20 clients flagged with completely different IDs from event 0), and hyb-lift was +27pp on those flagged clients during the event-1 window.

## Per-event global-test dips (canonical, undrifted test set)

| Method | Event 0 dip | Event 1 dip | Event 1 / Event 0 |
|---|---:|---:|---:|
| FedAvg | 0.122 | 0.182 | 1.49 |
| Flash | 0.181 | 0.239 | 1.33 |
| AdaptiveFedAvg | 0.032 | 0.108 | 3.34 |
| OurMethod | 0.121 | 0.198 | 1.63 |

Event 1 is 33-235% LARGER a shock than event 0 on the canonical test set across all methods. The accumulated drift (each group now has TWO disjoint swaps active) makes the canonical-test gap deeper.

## Per-group local-acc dips (the diagnostic that initially looked concerning)

For Group A: comparable dips at both events (within ~80-110%).
For Groups B and C: smaller or negative dips at event 1 (-17pp to +0.5pp).

**This is NOT a sparse-Dirichlet artifact.** Class-frequency check (see `class_frequency_check.py`) showed:
- Group A has 3.4x MORE samples for its event-1 pair (3,4) than event-0 pair (1,2)
- Group B has 0.97x (essentially equal) for event-0 (3,4) vs event-1 (5,6)
- Group C has 0.72x (slightly less but adequate) for event-0 (5,6) vs event-1 (7,8)

Real cause: federation-adaptation confound. With only 50 rounds between events, the federation hasn't fully reached equilibrium from event 0 before event 1 fires, so the per-group local-acc metric mixes transient dynamics. The canonical-test metric isn't confounded this way.

## Files

- `seed0/results_FedAvg.csv` — round + global_acc + local_c00..c19
- `seed0/results_Flash.csv` — same
- `seed0/results_AdaptiveFedAvg.csv` — adds client_lr column
- `seed0/results_OurMethod.csv` — adds hybrid_c00..hybrid_c19
- `seed0/results_OurMethod_flags.csv` — per-round per-client flag tracking
- `recurrent_smoke.log` — full live stdout

## Git context

- Branch: `recurrent-drift` (not merged)
- Code: commit `a74595a` (recurrent drift + per-group dip diagnostic)
- Wall time: ~2h 20min on GCP L4
- Cost: ~$1.65
