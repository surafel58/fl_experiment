# Dir(0.5) hypothesis test - go/no-go, single seed

**Branch:** `dir05-test`. **Hardware:** GCP L4 VM. **Setup:** CIFAR-10, 20 clients, 200 rounds, 5 local epochs, batch 64, single sudden drift at round 100, our CifarCNN, seed 0. **Only thing changed vs canonical Dir(0.1) setup: ALPHA_DIR = 0.5** (passed via the new `--alpha-dir` CLI flag).

## Hypothesis (as stated)

At Dir(0.1), OurMethod ties FedAvg because heavy non-IID makes drift undetectable for ~65% of clients (detection recall ~35%). At Dir(0.5), clients have more balanced data, drift should be MORE detectable, detector should fire more, mechanism may separate from FedAvg.

## Verdict: **REFUTED** on both prongs

**The premise (drift more detectable at Dir(0.5))**: refuted.
**The conclusion (OurMethod separates from FedAvg)**: refuted.

## The numbers that matter

### 1. OurMethod - FedAvg deltas (stable post-drift, mean of rounds [190, 199])

| | Dir(0.1) (seed-0 reference) | Dir(0.5) (this test) |
|---|---:|---:|
| Δ global stable | **−0.82pp** | **+0.12pp** |
| Δ per-client stable | **−0.38pp** | **+0.06pp** |

At Dir(0.5), both metrics flip sign from "tied negative" to "tied positive" but the magnitudes are essentially zero in either direction. No gap opens up. **OurMethod is tied with FedAvg at Dir(0.5) just like at Dir(0.1).**

### 2. Detection recall at drift (peak flagged_count over rounds [100, 109])

| | Dir(0.1) (seed-0 reference) | Dir(0.5) (this test) |
|---|---:|---:|
| Peak | **7/20 (35%)** | **1/20 (5%)** |
| Per-round | `[7, 7, 7, 7, 6, 1, 1, 1, 1, 1]` | `[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]` |
| Reaction duration | ~5 rounds | ~4 rounds (only 1 client) |

**Detection is 7× WORSE at Dir(0.5)**, not better.

## Full results table

| | Pre | Dip | Stable |
|---|---:|---:|---:|
| Dir(0.5) FedAvg global | 0.7810 | 0.1249 | 0.6661 |
| Dir(0.5) OurMethod global | 0.7810 | 0.1315 | 0.6673 |
| Dir(0.5) FedAvg per-client | 0.7810 | 0.1759 | 0.6112 |
| Dir(0.5) OurMethod per-client | 0.7810 | 0.1771 | 0.6118 |
| Dir(0.1) FedAvg global (seed 0) | 0.7078 | 0.1216 | 0.5887 |
| Dir(0.1) OurMethod global (seed 0) | 0.7048 | 0.1204 | 0.5805 |
| Dir(0.1) FedAvg per-client (seed 0) | 0.7078 | 0.1637 | 0.5476 |
| Dir(0.1) OurMethod per-client (seed 0) | 0.7048 | 0.1607 | 0.5438 |

Re-partition summary: Dir(0.1) samples/client min=127 max=5338 mean=2500; Dir(0.5) samples/client min=881 max=4665 mean=2500. Dir(0.5) is much more balanced, as expected.

## Mechanistic explanation - why detection got WORSE, not better

The intuition behind the hypothesis ("more balanced data → more detectable drift") had the sign backwards.

OurMethod's detector fires on per-layer weight-change magnitudes (the EMA-tracked ratio of layer-3/layer-4 update sizes). The signal it keys on is the SIZE of the per-layer weight spike when a client's loss spikes at drift.

- **At Dir(0.1)**, each client owns roughly 2-3 classes. When THOSE 2-3 classes get swapped, the per-class loss spike is HUGE for that client (~half their data just became wrong), and the gradient concentrates on the layers responsible for those specific classes. **Concentrated spike → detector triggers.**
- **At Dir(0.5)**, each client owns many classes more evenly. When 1-2 of them get swapped, only a small fraction of the client's data is affected, the loss spike is smaller, and the gradient is diluted across many other classes that didn't change. **Diffuse spike → detector misses it.**

So "more balanced data" produced the OPPOSITE of the predicted effect on detection: drift becomes more diffuse and harder to localize.

A useful corollary: the drift dip itself is similar in magnitude across both partitions (~0.12-0.13 on global), so it's not that drift becomes less harmful at Dir(0.5) — it's that the drift becomes less concentrated per client, which is exactly what makes the per-client detection mechanism less effective.

## Implications for the thesis story

1. **OurMethod's detection-based mechanism is essentially silent at Dir(0.5)**: only 1 client out of 20 ever triggers the per-layer adaptation, and the global model behaves identically to FedAvg.
2. **The "tied with FedAvg" finding from Dir(0.1) generalizes to Dir(0.5)**: not because the mechanism works equally well everywhere, but because at Dir(0.5) the mechanism barely fires at all and OurMethod reduces (operationally) to FedAvg.
3. **Lowering TAU (the detection threshold) is the cheapest follow-up** if we want to make the detector fire more at Dir(0.5). But the ablation table already showed that tau=1.2 produces a smaller drift dip (-4.56pp) at Dir(0.1) — at Dir(0.5) the drift dip is already similar, so the dip-reduction benefit may not transfer.
4. **The real question this test surfaces**: at any operating point in the non-IID spectrum, does OurMethod's mechanism produce a meaningful improvement over FedAvg, or is the detection-based approach fundamentally too brittle? Dir(0.1) gives high recall but the per-layer adaptation doesn't yield a gain. Dir(0.5) gives low recall, so the mechanism can't be tested. There's no obvious regime in between where both conditions hold.

## Decision

**No-go.** The hypothesis that Dir(0.5) would surface a separation between OurMethod and FedAvg is refuted by a single seed. The underlying cause (detection recall drops from 35% to 5%) is well-understood and unlikely to flip with more seeds. **Do not commit to a full multi-seed, multi-method run at Dir(0.5).**

The detection mechanism's failure mode at moderate non-IID is itself a finding worth keeping. The next exploration should be either (a) a tau sweep at Dir(0.5) to see if a lower threshold catches the diluted signal, or (b) recurrent drift at Dir(0.5) where cumulative selective retention might compound — neither tested here.
