# 2026-05-28 — OurMethod hybrid-logging run, 3 seeds

**Status:** Active. The decisive run for the thesis adaptation claim.

## What this was

Re-ran OurMethod on all 3 seeds with new instrumentation: alongside the
per-client `local_cXX` columns (global model on each client), the CSV now
also logs `hybrid_cXX` columns — the accuracy of the actual model each
client uses next round (global for unflagged layers + classifier, local
for flagged layers).

This is the metric the earlier analyses were missing: previous per-client
columns measured the GLOBAL model on each client, which by design excludes
flagged clients' updates for the flagged layers and is therefore not
representative of what those clients actually experience.

## Headline findings

Across 3 seeds, during the flag window (rounds 100–115, 33 flagged rounds total):

| Metric | Value |
|---|---|
| Mean hybrid - global on flagged clients (lift) | **+35.5pp** |
| Peak single-round lift | **+60.0pp** (seed 1 round 103) |
| Min single-round lift | +7.4pp (L3-only phase, seed 1) |

Per seed:
| Seed | Flagged rounds | Mean hyb-lift |
|---|---:|---:|
| 0 | 13 | +0.3454 (+34.5pp) |
| 1 | 13 | +0.3277 (+32.8pp) |
| 2 | 7  | +0.4255 (+42.6pp) |

## Hybrid vs FedAvg on the same clients (seeds 1, 2)

The truly thesis-relevant comparison: what OurMethod gives a drifted client
(its hybrid) vs what FedAvg gives the same drifted client (its global). Both
methods evaluated on the same flagged clients during the flag window:

| Seed | OurMethod hybrid | FedAvg local | Δ |
|---|---:|---:|---:|
| 1 | 0.7610 | 0.4385 | **+0.3225 (+32.3pp)** |
| 2 | 0.8580 | 0.4602 | **+0.3978 (+39.8pp)** |

OurMethod's selective layer adaptation produces a model for flagged clients
that is ~32–40 percentage points more accurate on their (drifted) local data
than FedAvg's federation-averaged model would be.

## Layer attribution

When both L3 and L4 are flagged: lift ≈ +45–60pp (peak).
When only L3 is flagged: lift drops to +7–25pp.

This means **most of the adaptation value sits in L4** (Linear 576→128,
73,856 params = 86% of the flaggable parameter surface). L3's contribution
is real but smaller in this configuration.

## How it was launched

```bash
python3 -u all_experiments_optimized.py --methods 4 --seed 0 \
    --out-dir runs/2026-05-28-hybrid/seed0 && \
python3 -u all_experiments_optimized.py --methods 4 --seed 1 \
    --out-dir runs/2026-05-28-hybrid/seed1 && \
python3 -u all_experiments_optimized.py --methods 4 --seed 2 \
    --out-dir runs/2026-05-28-hybrid/seed2
```

## Files

Each `seed<N>/` folder contains:
- `results_OurMethod.csv` — round, global_acc, local_c00..local_c19, hybrid_c00..hybrid_c19
- `results_OurMethod_flags.csv` — per-round per-client flag tracking

`hybrid_run.log` is the live stdout of all 3 seeds.

## Git context

- Code: hybrid logging added (commit `49e405f`)
- Wall time: ~1h 50min on GCP L4 (~37 min per seed)
- Cost: ~$1.30

## Caveat / minor difference vs prior runs

The new `hybrid_temp = get_model()` allocation in `run_our_method` consumes
RNG state, shifting the per-seed trajectory slightly. The pre-drift /
dip / stable numbers differ from the seed=0 run in `../2026-05-26-augfix/`
and from the seed=1/2 runs in `../2026-05-27-multiseed/` by ~0.5–2pp.
Within-seed comparisons inside this folder are valid; cross-folder
comparisons should use the corresponding FedAvg/Flash/Adaptive results
from the appropriate runs/ folder.
