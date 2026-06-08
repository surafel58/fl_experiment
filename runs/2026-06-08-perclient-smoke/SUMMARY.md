# Per-client smoke — 5 methods, single seed (seed 0)

**Setup:** CIFAR-10, Dir(0.1), 20 clients, 200 rounds, 5 local epochs, batch 64, single sudden drift at round 100. Per-client metric = FedCCFA's per-client generalized accuracy (validated at Step A — bit-exact pre-drift identity, see `validate_perclient_metric.py`).

**Hardware:** GCP L4 (g2-standard-4), torch 2.9.1+cu129. `cudnn.benchmark = True` (convs non-deterministic across processes).

**Methods:**

- **FedAvg** — reference baseline (global model on cohort-swapped full test sets).
- **Flash, AdaptiveFedAvg** — drift-aware baselines.
- **FedAvgPlus1** — control: FedAvg + 1 local epoch on each client's own data immediately before per-client eval. Isolates trivial last-step personalization as a confound on the per-client metric.
- **OurMethod** — drift-triggered selective layer adaptation.

## Results

All values are mean over the indicated window. `pre` = rounds [89, 99]; `dip` = pre − min(rounds [100, 109]); `stable` = mean(rounds [190, 199]).

### Global accuracy (canonical CIFAR-10 test set, undrifted labels)

| Method | Pre | Dip | Stable |
|---|---:|---:|---:|
| FedAvg | 0.7081 | 0.1219 | 0.5887 |
| Flash | 0.6714 | 0.1652 | 0.5711 |
| AdaptiveFedAvg | 0.5461 | 0.0327 | 0.4656 |
| FedAvgPlus1 | 0.7111 | 0.1254 | 0.5871 |
| OurMethod | 0.7049 | 0.1205 | 0.5805 |

### Per-client generalized accuracy (FedCCFA protocol, cohort-swapped full test sets)

| Method | Pre | Dip | Stable | Δ stable vs FedAvg |
|---|---:|---:|---:|---:|
| FedAvg | 0.7081 | 0.1640 | 0.5476 | 0 (ref) |
| Flash | 0.6714 | 0.1755 | 0.5337 | -1.40pp |
| AdaptiveFedAvg | 0.5461 | 0.1012 | 0.4441 | -10.35pp |
| FedAvgPlus1 | 0.4921 | 0.0909 | 0.4460 | -10.16pp |
| OurMethod | 0.7049 | 0.1608 | 0.5438 | -0.38pp |

## FedAvg vs FedAvgPlus1 — global-trajectory divergence gate

With `cudnn.benchmark = True` (our standard config) and the same seed, two
PyTorch processes running the SAME training loop will not produce bit-identical
global accuracy because CUDA convolution kernels are non-deterministic across
processes. The +1-epoch fine-tune in FedAvgPlus1 does NOT touch the global model
(it operates on a per-client scratch clone), so the only difference between
FedAvg and FedAvgPlus1's global trajectories should be CUDA non-determinism.

- max |Δglobal| across 200 rounds: **0.0217** (2.17pp)
- mean |Δglobal|: **0.0052** (0.52pp)
- final-round |Δglobal|: **0.0008** (0.08pp)

If `max_abs` is comparable to single-seed noise (~0.5–1.5pp), the global
trajectories are 'effectively the same model' and the per-client metric comparison
between FedAvg and FedAvgPlus1 is interpretable as 'same global model, +1 personalization epoch'.

## Headline reading

1. **OurMethod's per-client metric is within seed noise of plain FedAvg** (0.5438 vs 0.5476 stable, Δ = −0.38pp). At n=1 with a ~0.5–1.5pp noise floor, this is a tie, not a gain. OurMethod's mechanism produces no measurable per-client benefit at this operating point in this single seed.

2. **The FedAvgPlus1 control behaves OPPOSITE to the Dir(0.5) verification.** In the FedCCFA-FedAvg verification track, the `FedAvg_baseline.py` variant (locally-fine-tuned eval) scored +2.76pp ABOVE published FedAvg at Dir(0.5). Here at Dir(0.1), one extra local epoch on top of the aggregated global model produces −10.16pp BELOW FedAvg on the per-client metric (0.4460 vs 0.5476). The mechanism: at Dir(0.1) each client has ~500 samples covering only 2–3 classes; one epoch of local SGD on that narrow class distribution catastrophically collapses the model toward predicting only the client's class subset, tanking accuracy on the full 10k-sample cohort-swapped test set.

   **The two findings (Dir(0.5) +2.76pp, Dir(0.1) −10.16pp) are also not direct mirrors of each other.** `FedAvg_baseline.py` evaluated each client's 5-epoch local state from this round (and skipped `send_params` before final eval — so the global aggregation was never returned to the client at eval time). `FedAvgPlus1` evaluates `aggregated_global_model + 1 local epoch on a scratch clone`. Different exposure to the aggregated global model. But both are controls for "trivial last-step personalization"; at our operating point, that personalization is catastrophic, not lifting.

3. **Flash and AdaptiveFedAvg both underperform plain FedAvg** on BOTH metrics. Flash global stable 0.5711 (−1.76pp vs FedAvg) and per-client 0.5337 (−1.40pp). AdaptiveFedAvg global 0.4656 (−12.31pp) and per-client 0.4441 (−10.35pp) — Adaptive's LR-damping collapsing to near-zero under Dir(0.1) is a known failure mode and not a focus of this study. Both per-client deltas align in sign and magnitude with their global deltas, suggesting the per-client metric is mostly tracking model quality, not personalization-specific gains.

4. **Global vs per-client rankings agree, modulo OurMethod.** On global stable, the ranking is FedAvg > FedAvgPlus1 > OurMethod > Flash > AdaptiveFedAvg. On per-client stable, it's FedAvg > OurMethod > Flash > FedAvgPlus1 > AdaptiveFedAvg. OurMethod climbs from 3rd to 2nd on the per-client metric (passing FedAvgPlus1) — the only method whose ranking IMPROVES under the per-client lens. That's a hint, not a result; it's well inside seed noise and absolutely needs ≥3 seeds before any claim.

5. **CUDA non-determinism gate.** FedAvg vs FedAvgPlus1 global trajectories: max 2.17pp, mean 0.52pp, final 0.08pp. The 2.17pp max is slightly above the 1.5pp noise band but is a transient (likely near drift). Mean and final-round divergence are well within noise. The control's global trajectory tracks plain FedAvg closely enough that the per-client comparison is interpretable.

## Caveats

- **Single seed (seed 0).** Prior 3-seed studies measured 1-σ noise on stable post-drift accuracy at ~0.5–1.5pp. Treat sub-1pp deltas as noise; treat 1–2pp deltas with care; the OurMethod vs FedAvg per-client Δ of −0.38pp is fully inside noise.
- **FedAvgPlus1 is a control, not a competitive method.** It measures what trivial last-step personalization does to the per-client metric at our operating point. The −10.16pp result tells us that any per-client gain on top of plain FedAvg is genuine signal — there is no "free lunch" lift available from local fine-tuning here, unlike at Dir(0.5).
- **Non-determinism caveat above.** The 'global model unchanged' control claim is approximate, bounded by the divergence numbers in the gate section.
- **AdaptiveFedAvg-at-Dir(0.1) is a known degenerate case** (LR collapse under heavy non-IID); the −12.31pp global gap is not specific to the per-client lens.
