# Experiment runs index

All FL experiment runs in this project, in chronological order. Each subdirectory
holds the CSVs, the live run log, and a `RUN_NOTES.md` explaining what produced it.

Naming: `runs/<YYYY-MM-DD>-<short-tag>/[seed<N>/]`. The tag describes the run
intent; seed subdirectories group per-seed CSV outputs.

## Runs

| Folder | Date | Methods | Seeds | Code state | Notes |
|---|---|---|---|---|---|
| [2026-05-26-bug-ourmethod-only](2026-05-26-bug-ourmethod-only/) | 2026-05-26 | OurMethod | 0 | GPU-resident pipeline, **buggy mid-gray pad** | First end-to-end test of new pipeline. CSVs overwritten by subsequent run; only the log survives. |
| [2026-05-26-bug-all4](2026-05-26-bug-all4/) | 2026-05-26 | FedAvg, Flash, AdaptiveFedAvg | 0 | GPU-resident pipeline, **buggy mid-gray pad** | Companion all-3 run to complete the first 4-method comparison. CSVs overwritten; only the log survives. Misleadingly favored FedAvg — that bug led to the parity test that found the pad value mismatch. |
| [2026-05-26-augfix](2026-05-26-augfix/) | 2026-05-26 | all 4 | 0 | **Augment pad fix applied** (true black, matches torchvision) | First clean 4-method seed=0 run. No per-client logging yet. |
| [2026-05-27-multiseed](2026-05-27-multiseed/) | 2026-05-27 | all 4 | 1, 2 | Augment fix + **per-client local-accuracy logging** + `--seed` flag | Two additional seeds for mean±std. Per-client CSV columns `local_c00..local_c19`. |

## Conventions

- CSV filenames inside a seed folder do **not** include the seed (the folder
  path already encodes it). Example: `runs/2026-05-27-multiseed/seed1/results_OurMethod.csv`.
- Each run's RUN_NOTES.md records: launching command, git commit at run time,
  total wall time, final metric summary, anything worth knowing.
- Live run logs (`*_run.log`) are kept alongside the CSVs.

## Reproducing a run

```bash
git checkout <commit-from-RUN_NOTES.md>
python3 all_experiments_optimized.py --methods all --seed <N> --out-dir runs/<new-tag>/seed<N>
```
