# 2026-05-26 — FedAvg + Flash + AdaptiveFedAvg (buggy aug)

**Status:** Superseded. CSVs overwritten by [../2026-05-26-augfix/](../2026-05-26-augfix/) — only the log survives.

## What this was

Companion to [../2026-05-26-bug-ourmethod-only/](../2026-05-26-bug-ourmethod-only/) —
the 3 baseline methods (FedAvg, Flash, AdaptiveFedAvg) needed to complete the
4-method comparison. Ran sequentially with `--methods 1 2 3`.

## Result at the time — and why it mattered

| Method | Pre-drift | Dip | Post-drift stable |
|---|---:|---:|---:|
| FedAvg | 0.7147 | 0.1114 | **0.6082** |
| Flash | 0.6810 | 0.1803 | 0.5824 |
| AdaptiveFedAvg | 0.5580 | 0.0377 | 0.4854 |
| OurMethod (from bug-ourmethod-only) | 0.7142 | 0.1148 | 0.6057 |

**FedAvg unexpectedly beat OurMethod on all 3 main metrics**, flipping the
ordering from the prior single-seed in the handoff. This was the trigger to
investigate.

Per-seed delta vs the handoff baseline showed FedAvg jumped +2.9pp post-drift
while OurMethod only moved +0.8pp — a suspicious asymmetric improvement from
a "neutral" pipeline change.

## Diagnosis

Wrote `augment_parity_test.py` comparing torchvision vs the GPU augment per-channel
mean and std over many augmented batches. Found:

| Pipeline | mean (R, G, B) | std (R, G, B) |
|---|---|---|
| torchvision | -0.428, -0.522, -0.503 | 1.391, 1.364, 1.353 |
| GPU augment (buggy) | -0.102, -0.197, -0.204 | 1.148, 1.145, 1.175 |
| GPU augment (fixed: pad with -MEAN/STD) | -0.428, -0.522, -0.503 | 1.391, 1.364, 1.353 |

The bug: `F.pad(value=0)` padded the already-normalized tensor with 0 in
normalized space, i.e. mid-gray after un-normalizing. Torchvision pads the
raw uint8 image with 0 (true black) before normalization. After fixing the
pad value to per-channel `-MEAN/STD`, the per-channel stats match torchvision
to 6 decimal places.

## Files

- `full_run.log` — full stdout of the 3-method run

## Git context

- Code: pre-fix gpu_augment (commit `3e2bce7`)
- Wall time: ~1h 43min on GCP L4
