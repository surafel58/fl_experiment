# 2026-05-26 — OurMethod single-seed (buggy aug)

**Status:** Superseded. CSVs overwritten by [../2026-05-26-augfix/](../2026-05-26-augfix/) — only the log survives.

## What this was

The first full 200-round OurMethod run on the refactored GPU-resident pipeline.
Used as a sanity check that the refactor reproduced the prior single-seed numbers
from the handoff (pre-drift 0.7124, post-drift stable 0.5975).

## Result at the time

| Metric | Prior (handoff) | This run | Δ |
|---|---:|---:|---:|
| Pre-drift | 0.7124 | 0.7142 | +0.18pp |
| Dip | 0.1107 | 0.1148 | +0.41pp |
| Post-drift stable | 0.5975 | 0.6057 | +0.82pp |
| Peak flagged at drift | 7/20 | 7/20 | exact match |

Numbers looked great and matched prior to within ~1pp.

## Why superseded

Later, the all-3 baseline run ([../2026-05-26-bug-all4/](../2026-05-26-bug-all4/))
showed FedAvg unexpectedly beating OurMethod, which led to the augment parity
test that found a real bug: the GPU augment was padding the **already-normalized**
tensor with constant 0 (= mid-gray after un-normalizing), while torchvision pads
the raw uint8 image with 0 (= true black). The fix is in `gpu_augment` —
see [../2026-05-26-augfix/RUN_NOTES.md](../2026-05-26-augfix/RUN_NOTES.md).

## Files

- `our_run.log` — full stdout of the 200-round run

## Git context

- Code: pre-fix gpu_augment (commit `6644da0` — "Add OurMethod 200-round results")
- Pipeline: GPU-resident dataset + GPU-side aug, BS=64, drop_last=True
- Wall time: ~33 min on GCP L4
