# Communication-cost analysis: OurMethod vs FedAvg

**Branch:** `comm-analysis`. Read-only analysis - no new training. Uses committed flag CSVs from the 3-seed single-drift runs (perclient-metric track).

## 1. Per-layer parameter counts (CifarCNN, 107,690 total)

Derived from `LAYER_GROUPS` at all_experiments_optimized.py:1104-1112.

| Layer | Module | Op | Params | % of total |
|---|---|---|---:|---:|
| layer1 | hidden_layers.0  | Conv2d(3,16,5)             |  1,216 |  1.13% |
| layer2 | hidden_layers.3  | Conv2d(16,32,5,padding=1)  | 12,832 | 11.92% |
| layer3 | hidden_layers.6  | Conv2d(32,64,3,padding=1)  | 18,496 | 17.18% |
| layer4 | hidden_layers.10 | Linear(576, 128)            | 73,856 | 68.58% |
| fc (classifier) | fc | Linear(128, 10) |  1,290 |  1.20% |
| **total** | - | - | **107,690** | 100.00% |

- Always uploaded (L1 + L2 + fc): **15,338 params** = 14.24% of model.
- Flaggable layers (L3 + L4): **92,352 params** = 85.76% of model.
- L4 alone is **73,856 = 68.58%** of the model - so an L4 flag saves ~4x what an L3 flag saves.

## 2. Per-round upload budget (20 clients participating, every round)

- **FedAvg (constant)**: 20 clients * 107,690 params = **2.15M params/round**, every round.
- **OurMethod (variable)**: per-round upload = 20 * (L1 + L2 + fc) + (20 - flagged_L3) * L3 + (20 - flagged_L4) * L4.
  - Minimum (every client flags both L3 and L4): 20 * 15,338 = 306.76k params/round.
  - Maximum (no flags fire, the baseline state): 2.15M params/round = identical to FedAvg.

OurMethod can NEVER upload more than FedAvg - it can only withhold layers, never add them. The saving is therefore always non-negative per round.

## 3. Whole-run totals (200 rounds, per seed)

FedAvg constant baseline: 200 rounds * 2.15M = **430.76M params total** uploaded per run.

| Seed | OurMethod uploaded | FedAvg uploaded | Saving (params) | Saving (%) |
|---:|---:|---:|---:|---:|
| 0 | 427.73M | 430.76M | 3.03M | **0.703%** |
| 1 | 427.97M | 430.76M | 2.79M | **0.647%** |
| 2 | 427.40M | 430.76M | 3.36M | **0.780%** |
| **mean +/- std** | - | - | 3.06M | **0.710% +/- 0.054%** |

Whole-run saving across 3 seeds: **mean 0.710%, std 0.054%**. Well under 1%, as expected. The mechanism is dormant for the vast majority of training (rounds 0-99 pre-drift, rounds ~105-199 post-drift); the saving comes from a brief window where flags fire.

## 4. Drift-window vs rest-of-run breakdown

Drift window = rounds [100, 109] (10 rounds immediately after drift), where the detection mechanism actually fires. Outside this window, OurMethod uploads ~identically to FedAvg.

| Seed | drift-window saving (% of drift-window FedAvg) | rest-of-run saving (% of rest-of-run FedAvg) |
|---:|---:|---:|
| 0 | **13.806%** | 0.0136% |
| 1 | **12.605%** | 0.0181% |
| 2 | **15.606%** | 0.0000% |
| **mean +/- std** | **14.006% +/- 1.233%** | 0.0105% +/- 0.0077% |

During the drift window the saving is **~14.0%** of communication, ~20x the whole-run average. Outside the drift window the saving is essentially zero (0.0105%) - the detector doesn't fire, OurMethod uploads exactly what FedAvg uploads.

## 5. Layer attribution of the savings

Of the total params saved across the run, how much came from L3 flags vs L4 flags?

| Seed | L3 share | L4 share | total flags fired (sum over rounds) |
|---:|---:|---:|---|
| 0 | 19.5% | 80.5% | L3 flags: 32, L4 flags: 33 |
| 1 | 17.9% | 82.1% | L3 flags: 27, L4 flags: 31 |
| 2 | 12.1% | 87.9% | L3 flags: 22, L4 flags: 40 |
| **mean** | 16.5% | 83.5% | - |

L4 dominates the saving (83% on average) even though L3 and L4 are flagged at similar frequencies, because L4 has ~4x as many params (73.9k vs 18.5k). The detector fires on L3 and L4 roughly equally, but in communication terms an L4 flag is worth ~4 L3 flags.

## 6. Honest reading

1. **The whole-run communication saving is small**: **mean 0.710% +/- 0.054%** across 3 seeds. The mechanism is dormant outside the drift window; OurMethod and FedAvg are operationally identical for ~190 of the 200 rounds in a single-drift run.

2. **The drift-window saving is meaningfully larger**: **mean 14.006% +/- 1.233%** during rounds [100, 109]. "When the mechanism actually fires, it withholds roughly 14% of the layer uploads from the flagged clients." This is the honest "when it matters" number.

3. **The asymmetry property**: OurMethod NEVER uploads more than FedAvg - it only ever withholds layers, never adds them. So the property is **"never worse on communication, modestly better during detected drift"**. There is no scenario where this mechanism increases communication cost.

4. **Why the saving is concentrated, not constant**: the EMA detector fires only when per-layer weight-change magnitudes spike, which happens almost exclusively at drift (rounds 100-104 in seed 0, similar in other seeds). After the spike subsides (typically by round 105-110), the detector goes silent and OurMethod's selective-sync path becomes a no-op. The dip-mitigation benefit and the communication-saving benefit are co-located in time.

5. **Scaling note**: at our setup (200 rounds, single drift), the saving is small in absolute % terms. In a setting with MORE frequent drift events (e.g. continuous incremental drift, or recurrent drift every K rounds), the drift-window fraction of total training would be larger, and the whole-run saving would scale proportionally. **The whole-run % saving is essentially (drift-window saving %) * (drift-fraction of training)**. At our 10/200 = 5% drift fraction, a ~14% drift-window saving yields ~0.70% whole-run saving, which matches the observed ~0.71%.

**Bottom line:** OurMethod's communication saving over FedAvg is **0.71% whole-run, 14.0% during drift, never worse**. Small in absolute terms at a single-drift operating point; would scale up with more frequent drift events. The mechanism is a layer-withholding *side effect* of the drift-detection mechanism, not its primary purpose - the accuracy effect (small/no measurable gain) is the load-bearing claim, and the communication saving is a documented side benefit.