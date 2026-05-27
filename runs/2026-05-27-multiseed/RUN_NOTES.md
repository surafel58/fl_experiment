# 2026-05-27 — Multi-seed sweep, seeds 1 and 2

**Status:** Active. Companion to [../2026-05-26-augfix/](../2026-05-26-augfix/) (seed 0).

## What this was

Two additional seeds (1 and 2) for variance estimation across the 4-method
comparison. First run that uses per-client local-accuracy logging, so per-round
CSVs include 20 extra columns `local_c00 .. local_c19`.

Combined with seed 0, gives a 3-seed paired comparison; see [`../../multiseed_analysis.py`](../../multiseed_analysis.py)
and `../../multiseed_analysis_report.txt`.

## Final results (mean ± std over seeds 1 and 2)

| Method | Pre-drift | Dip | Post-drift stable |
|---|---:|---:|---:|
| FedAvg | 0.7151 ± 0.005 | 0.1190 ± 0.032 | 0.5982 ± 0.022 |
| Flash | 0.6944 ± 0.001 | 0.1896 ± 0.040 | 0.6035 ± 0.032 |
| AdaptiveFedAvg | 0.5879 ± 0.001 | 0.0692 ± 0.039 | 0.5033 ± 0.034 |
| **OurMethod** | **0.7264 ± 0.011** | 0.1236 ± 0.024 | **0.6072 ± 0.040** |

Per seed:
- seed 1: OurMethod pre 0.7342 / dip 0.1067 / stable 0.6353 (strongest)
- seed 2: OurMethod pre 0.7185 / dip 0.1405 / stable 0.5791

## Flag detection consistency

Across both seeds, OurMethod's selective layer adaptation fired with 0 false
positives in pre-drift (rounds 0-99) and 0 false positives in post-recovery
(rounds 120+). Peak flag count at the drift moment varied slightly by seed
due to the different client partition (re-seeded `partition_dataset`).

## How it was launched

```bash
python3 -u all_experiments_optimized.py --methods all --seed 1 && \
python3 -u all_experiments_optimized.py --methods all --seed 2
```

Both invocations ran back-to-back on the VM, results auto-tagged with seed in the
filename. After download, files were renamed to drop the `_seed<N>` suffix because
the seed is encoded in the parent folder.

## Files

Each `seed<N>/` folder contains the same 5 CSV files:
| File | Description |
|---|---|
| `results_FedAvg.csv` | round, global_acc, local_c00 .. local_c19 |
| `results_Flash.csv` | same shape as FedAvg |
| `results_AdaptiveFedAvg.csv` | adds `client_lr` column |
| `results_OurMethod.csv` | same shape as FedAvg |
| `results_OurMethod_flags.csv` | per-round per-client flag tracking |

`multiseed_run.log` is the live stdout of both invocations concatenated.

## Git context

- Code: per-client logging + --seed flag (commit `a0850fa`)
- Wall time: ~4h 35min on GCP L4 (~2h 17min per seed)
- Cost: ~$3.30
