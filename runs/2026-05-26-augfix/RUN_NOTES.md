# 2026-05-26 — All 4 methods, seed=0, post augment-fix

**Status:** Active. This is the canonical seed=0 result for the 4-method comparison.

## What this was

Full 4-method run after the GPU-augment padding bug was fixed. With true-black
padding matching torchvision, this is the first "clean" 4-method comparison.

No per-client local-accuracy logging — that feature was added later, so this run
has only the global accuracy column per round.

## Final results

| Method | Pre-drift | Dip | Post-drift stable |
|---|---:|---:|---:|
| FedAvg | 0.7081 | 0.1219 | 0.5887 |
| Flash | 0.6847 | 0.1805 | 0.5716 |
| AdaptiveFedAvg | 0.5558 | 0.0325 | 0.4892 |
| **OurMethod** | **0.7136** | 0.1201 | **0.5952** |

OurMethod wins 3 of 4 metrics vs FedAvg (pre-drift, dip, post-drift stable),
matching the handoff's original single-seed pattern.

Drift detection: 7/20 clients flagged at round 100, all clears by round 114,
zero false positives pre- and post-drift.

## How it was launched

```bash
python3 -u all_experiments_optimized.py --methods all > all4_run.log 2>&1
```

## Files

| File | Description |
|---|---|
| `seed0/results_FedAvg.csv` | 200 rounds, columns: round, global_acc |
| `seed0/results_Flash.csv` | same shape as FedAvg |
| `seed0/results_AdaptiveFedAvg.csv` | adds `client_lr` column |
| `seed0/results_OurMethod.csv` | same shape as FedAvg |
| `seed0/results_OurMethod_flags.csv` | per-round per-client flag tracking |
| `all4_run.log` | full stdout of the run |

## Git context

- Code: post-fix gpu_augment (commit `3a70360` — "Add full 4-method run results with augmentation fix applied")
- Pipeline: GPU-resident + GPU augmentation (true-black pad), BS=64, drop_last=True, num_workers=0
- Wall time: 2h 14min on GCP L4
- Cost: ~$1.60
