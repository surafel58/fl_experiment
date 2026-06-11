# Saile (2024 FLTA) - client-side loss-EMA dynamic LR baseline

**Branch:** `saile-baseline`. **Hardware:** GCP L4 VM. **Setup:** CIFAR-10, Dir(0.1), 20 clients, 200 rounds, 5 local epochs, batch 64, single sudden drift at round 100.

**Algorithm:** faithful port of `LearningrateEstimatorLoss` (per-client 3-EMA on loss with bias correction, V=0 edge case, cap at initial_lr) + server-side multiplicative decay (StepLR gamma=0.99/round). Mirrors `concept-drift-adaption-saile/src/algorithm/learningrate_estimator.py:77-128` and `src/server/fedavgserver.py:40,193-213,389`.

**Hyperparameters:** β1=0.7, β2=0.3, β3=0.7, lr_decay=0.99/round (Saile's). Initial LR selected via FedDrift-style sweep on our setup (Saile's recommended 0.2 diverges; we select 0.01).

## 1. LR sweep (seed 0)

Per-client loss-EMA gets its own LR sweep, same protocol as we used for AdaptiveFedAvg. Selection criterion: post-drift stable global accuracy.

| `saile_init_lr` | Pre | Dip | **Stable** | Per-client stable | Notes |
|---:|---:|---:|---:|---:|---|
| 0.2 | 0.1000 | 0.0000 | **0.1000** | 0.1000 | **diverged** (stuck near random-guess 10%) |
| 0.1 | 0.1000 | 0.0000 | **0.1000** | 0.1000 | **diverged** (stuck near random-guess 10%) |
| 0.01 | 0.7070 | 0.1196 | **0.5844** | 0.5477 | **selected** |

Saile's CIFAR-10 default (lr=0.2) and the intermediate lr=0.1 both diverged immediately in our setup (CifarCNN, B=64, E=5, Dir(0.1)). This is the same regime-mismatch we documented for AdaptiveFedAvg: Saile's IID + TwoCNN + smaller B/E setup tolerates LR=0.2, ours does not. **Selected: lr=0.01** (matches our other methods).

## 2. Drift-reaction analysis (seed 0, lr=0.01)

The selected-LR run IS the seed-0 smoke. Per-round CSV columns `base_lr`, `min_client_lr`, `mean_client_lr`, `max_client_lr` make the mechanism observable.

Selected per-round snapshots:

```
round |   base_lr |       min /      mean /       max |    spread |   global
----------------------------------------------------------------------------
    0 |   0.00990 |   0.01000 /   0.01000 /   0.01000 |   0.00000 |   0.2119
    5 |   0.00941 |   0.00393 /   0.00541 /   0.00744 |   0.00351 |   0.4037
   50 |   0.00599 |   0.00604 /   0.00888 /   0.01000 |   0.00396 |   0.6845
   99 |   0.00366 |   0.00295 /   0.00650 /   0.01000 |   0.00705 |   0.7044
  100 |   0.00362 |   0.00465 /   0.00923 /   0.01000 |   0.00535 |   0.6154  <- DRIFT
  101 |   0.00359 |   0.00404 /   0.00897 /   0.01000 |   0.00596 |   0.6362
  105 |   0.00345 |   0.00381 /   0.00869 /   0.01000 |   0.00619 |   0.5914
  110 |   0.00328 |   0.00309 /   0.00763 /   0.01000 |   0.00691 |   0.5869
  199 |   0.00134 |   0.00125 /   0.00329 /   0.01000 |   0.00875 |   0.5809
```

- **Per-client LRs vary across clients**: non-zero spread from round 1 onwards (e.g., 0.00295 vs 0.01000 at round 99, a ~3.4x ratio).
- **LRs react at drift**: pre-drift [89-99] mean LR = 0.00665; post-drift [100-110] mean LR = 0.00854; ratio = **1.284x** (+28.4% relative). Loss variance spikes -> R_hat lifts -> per-client LR rises. Mechanism doing what it should.
- **The `initial_lr` cap is binding**: in 178/200 rounds, max(client_lr) == 0.01 exactly; clients want a higher LR than the cap allows. Cap is doing real work.
- **base_lr decays as designed**: 0.00990 at round 0, 0.00134 at round 199, matches 0.01 * 0.99^199.

## 3. Saile 3-seed results (lr=0.01)

| Metric | Pre | Dip | Stable |
|---|---:|---:|---:|
| Global accuracy | 0.7139 ± 0.0058 | 0.1228 ± 0.0114 | 0.5985 ± 0.0148 |
| Per-client gen acc | 0.7139 ± 0.0058 | 0.1662 ± 0.0084 | 0.5567 ± 0.0072 |

## 4. Saile vs FedAvg vs AdaptiveFedAvg (3-seed, same operating point)

| Method | Global stable | Per-client stable |
|---|---:|---:|
| FedAvg | 0.5958 ± 0.0129 | 0.5525 ± 0.0065 |
| AdaptiveFedAvg (corrected, lr=0.01) | 0.5916 ± 0.0164 | 0.5511 ± 0.0098 |
| Saile (lr=0.01, lr_decay=0.99/rd) | 0.5985 ± 0.0148 | 0.5567 ± 0.0072 |

Per-seed delta on global stable (Saile - FedAvg):
  seed 0: -0.43pp; seed 1: +0.50pp; seed 2: +0.73pp; mean +0.27pp +/- 0.50pp

Per-seed delta on per-client stable (Saile - FedAvg):
  seed 0: +0.01pp; seed 1: +0.36pp; seed 2: +0.88pp; mean +0.42pp +/- 0.36pp

## 5. Honest reading

1. **The Saile port is operationally correct.** Per-client LRs genuinely vary, react at drift (+28% mean LR), the cap is binding, the server decay is correct. The mechanism is doing what the paper specifies; the implementation is faithful.

2. **Saile vs FedAvg on global stable**: mean delta = +0.27pp, std = 0.50pp across 3 seeds. |mean| <= std -> effect INSIDE the seed-noise floor.
   Per-seed signs: [-1, 1, 1]. Signs disagree, so this is noise around the FedAvg level.

3. **Saile vs FedAvg on per-client stable**: mean delta = +0.42pp, std = 0.36pp. |mean| > std -> effect OUTSIDE seed noise.
   Per-seed signs: [1, 1, 1].

4. **The per-client result is suggestive, not definitive.** All 3 seeds positive (+0.01, +0.36, +0.88pp on per-client stable vs FedAvg), mean +0.42pp marginally exceeds std 0.36pp. At n=3 with one near-zero sample, this is a "hint of a real effect", not a confirmed result. A multi-seed (n>=5) replication is the next step before claiming Saile's per-client mechanism helps. **Global stable is uniformly noise** (signs disagree, |mean| < std).

5. **Saile vs the other drift-adaptive methods at our operating point.** Flash, AdaptiveFedAvg-corrected, and OurMethod each came back within seed noise of plain FedAvg on both metrics. Saile is the **first method we've tested where the per-client metric direction is consistent across seeds**, although the magnitude is small and the n=3 evidence is marginal. AdaptiveFedAvg-corrected and Saile are mechanically similar (3-EMA on a variance ratio with bias correction), but they differ in input signal (Saile uses per-client loss, AdaptiveFedAvg uses global weight variance) and granularity (Saile is per-client, AdaptiveFedAvg is global). The per-client signal may be what matters here.

**Bottom line.** Saile is a faithful baseline. **Global accuracy: tied with FedAvg.** **Per-client accuracy: consistent positive direction across 3 seeds at +0.42pp +/- 0.36pp** - suggestive but inside the "is it real?" gray zone at n=3. Worth replicating with more seeds before making a claim. The mechanism is operationally correct (per-client LR variation + drift reaction visible), and the result is a more interesting signal than the other drift-adaptive baselines produced at this operating point.