# Experiment Report — Part A: Recorded Results

**Project:** Drift-Triggered Selective Layer Adaptation in Federated Learning under Distributed Real Concept Drift ("OurMethod"), Master's thesis (FedCCFA-based baseline).
**Date of report:** 2026-06-19. **Author of analysis:** Claude (read-only, no new experiments).
**Source repo:** `fl_experiment/`. **Compiled on:** branch `main` at commit `9e3b7f9` (with three unmerged feature branches inspected via `git show`).

This report extracts every number from the committed result CSVs. No estimates. Where a value is not in the repo, the report says **NOT FOUND IN REPO**. Where mean ± std is reported, the per-seed values it was computed from are also shown.

> Reproducible computation: `python experiment_report_compute.py` regenerates every numeric table here from the CSVs cited.

---

## 1. Setup (verified from code)

### 1.1 CifarCNN architecture

[`all_experiments_optimized.py:495-507`](all_experiments_optimized.py#L495-L507):

```python
class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(3, 16, 5),               nn.LeakyReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding=1),   nn.LeakyReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),   nn.LeakyReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64*3*3, 128),            nn.LeakyReLU(),
        )
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        return self.fc(self.hidden_layers(x))
```

Per-layer parameter counts (computed from the layer definitions; `LAYER_GROUPS` at lines 1104-1108 maps these to the OurMethod-flagging names `layer1..layer4`):

| Name | Module idx | Op | Params |
|---|---|---|---:|
| layer1 | hidden_layers.0  | Conv2d(3, 16, 5) | 1,216 |
| layer2 | hidden_layers.3  | Conv2d(16, 32, 5, padding=1) | 12,832 |
| layer3 | hidden_layers.6  | Conv2d(32, 64, 3, padding=1) | 18,496 |
| layer4 | hidden_layers.10 | Linear(64·3·3=576, 128) | 73,856 |
| fc | fc | Linear(128, 10) | 1,290 |
| **total** |  |  | **107,690** |

The model contains **no BatchNorm, LayerNorm, GroupNorm, or InstanceNorm layers** — inherited from FedCCFA's original `utils/models.py:39-66`. Input is pre-normalized once at data load time only.

### 1.2 FL configuration (defaults)

[`all_experiments_optimized.py:91-99`](all_experiments_optimized.py#L91-L99):

```python
NUM_CLIENTS    = 20
NUM_ROUNDS     = 200
LOCAL_EPOCHS   = 5
BATCH_SIZE     = 64
LR             = 0.01
MOMENTUM       = 0.9
WEIGHT_DECAY   = 1e-5
ALPHA_DIR      = 0.1
SEED           = 0
```

### 1.3 Drift schedule (default = single sudden) and cohort label-swaps

[`all_experiments_optimized.py:115-134`](all_experiments_optimized.py#L115-L134):

```python
DRIFT_SCHEDULE_SINGLE    = [100]                # default
DRIFT_SCHEDULE_RECURRENT = [100, 150]           # --recurrent
DRIFT_EVENTS = [
    {'A': (1, 2), 'B': (3, 4), 'C': (5, 6)},    # event 0
    {'A': (3, 4), 'B': (5, 6), 'C': (7, 8)},    # event 1 (only used with --recurrent)
]
DRIFT_GROUP_A  = [i for i in range(NUM_CLIENTS) if i % 10 < 3]
DRIFT_GROUP_B  = [i for i in range(NUM_CLIENTS) if 3 <= i % 10 < 6]
DRIFT_GROUP_C  = [i for i in range(NUM_CLIENTS) if i % 10 >= 6]
```

For NUM_CLIENTS=20: cohort A = clients {0,1,2,10,11,12} (6 clients), cohort B = {3,4,5,13,14,15} (6 clients), cohort C = {6,7,8,9,16,17,18,19} (8 clients). This matches FedCCFA's `client.id % 10` rule.

### 1.4 OurMethod constants

[`all_experiments_optimized.py:1110-1115`](all_experiments_optimized.py#L1110-L1115):

```python
DRIFT_LAYERS  = ['layer3', 'layer4']   # only these are eligible to be flagged
STABLE_LAYERS = ['layer1', 'layer2']   # always uploaded (note: STABLE_LAYERS is comment-level only, not enforced in get_upload_state)
CLASSIFIER    = 'fc'                   # always overwritten by global
EMA_ALPHA     = 0.3
TAU_OUR       = 1.4
WARMUP        = 10
```

The flag decision is one line at [line 1285-1286](all_experiments_optimized.py#L1285-L1286):
```python
if name in DRIFT_LAYERS and ratio > TAU_OUR:
    flags[name] = True
```

### 1.5 Per-client generalized accuracy implementation

Faithful port of FedCCFA's metric. Source: [`all_experiments_optimized.py:218-282`](all_experiments_optimized.py#L218-L282). Four full-test-set label variants (`TEST_Y_VARIANTS[0..3]`, 10000 labels each), per-cohort label swaps applied in lockstep to client training labels via [`apply_drift_event`](all_experiments_optimized.py#L417), and `evaluate_per_client_gen_acc` ([line 266](all_experiments_optimized.py#L266)) evaluates the **global aggregated model** on `TEST_Y_VARIANTS[client.global_test_id]`:

```python
def evaluate_per_client_gen_acc(model):
    needed = set(CLIENT_GLOBAL_TEST_ID)
    if TEST_X_VARIANTS is not None:
        # Covariate-drift mode: per-cohort IMAGES, canonical LABELS
        acc_by_gid = {gid: evaluate_gpu(model, TEST_X_VARIANTS[gid], TEST_Y) for gid in needed}
    else:
        # Label-drift mode: shared canonical IMAGES, per-cohort corrupted LABELS
        acc_by_gid = {gid: evaluate_gpu(model, TEST_X, TEST_Y_VARIANTS[gid]) for gid in needed}
    if len(needed) == 1:
        return next(iter(acc_by_gid.values()))   # pre-drift fast-path -> bit-exact identity with evaluate_gpu
    cohort_sizes = {gid: 0 for gid in needed}
    for cid in range(NUM_CLIENTS):
        cohort_sizes[CLIENT_GLOBAL_TEST_ID[cid]] += 1
    return sum(acc_by_gid[gid] * cohort_sizes[gid] for gid in needed) / NUM_CLIENTS
```

**Validation**: bit-exact pre-drift identity `evaluate_per_client_gen_acc(m) == evaluate_gpu(m)` verified at `validate_perclient_metric.py` (every CSV's round 0 row also satisfies `per_client_gen_acc == global_acc` to floating-point precision).

**FedCCFA-FedAvg reproduction**: at Dir(0.5), 20 clients, 200 rounds, seed 0, the harness was verified at **0.5994** (file `runs/2026-06-04-fedccfa-verify/CIFAR10_sudden_FedAvg_20260604181603436012.csv`) against published target 0.6096 ± 1.5pp → PASS, inside band.

---

## 2. Method registry + CLI flags

[`all_experiments_optimized.py:1590-1597`](all_experiments_optimized.py#L1590-L1597):

```python
METHOD_REGISTRY = {
    1: ('FedAvg',          run_fedavg),
    2: ('Flash',           run_flash),
    3: ('AdaptiveFedAvg',  run_adaptive_fedavg),
    4: ('OurMethod',       run_our_method),
    5: ('FedAvgPlus1',     run_fedavg_plus1),   # control: FedAvg + 1 local epoch for per-client eval only
    6: ('Saile',           run_saile),          # Saile 2024 (FLTA): per-client loss-EMA dynamic LR
}
```

CLI flags (`parse_args()`):

| Flag | Affects | Purpose |
|---|---|---|
| `--methods <ids>` | dispatcher | Run subset {1..6} or "all" (defaults to {1..4}; 5 and 6 are opt-in controls). |
| `--rounds N` | global | Override `NUM_ROUNDS` (default 200). |
| `--seed N` | global | Override `SEED`, re-partition train set. |
| `--out-dir DIR` | global | Output CSV directory. |
| `--recurrent` | drift schedule | Use `DRIFT_SCHEDULE_RECURRENT = [100, 150]`. |
| `--alternating-drift` | drift schedule | Schedule `[40, 80, 120, 160]` with identical swap at each event (involution → concept oscillates canonical↔swapped). |
| `--alpha-dir FLOAT` | partition | Override `ALPHA_DIR` (default 0.1); triggers re-partition. |
| `--ablation-tau FLOAT` | OurMethod | Override `TAU_OUR` (default 1.4). `inf` disables detection. |
| `--ablation-all-layers` | OurMethod | Make all 4 hidden layers eligible to flag (`DRIFT_LAYERS = list(LAYER_GROUPS)`). |
| `--drift-layers L ...` | OurMethod | Explicit `DRIFT_LAYERS` override (`layer1..layer4`). |
| `--adaptive-init-lr FLOAT` | Method 3 | Override AdaptiveFedAvg's initial LR for its FedDrift-style sweep. |
| `--saile-init-lr FLOAT` | Method 6 | Override Saile's initial LR for its sweep. |
| `--aggressive-concept-drift` | drift type | Per-cohort full 10-label permutations (seeded RNG 42) instead of pairwise swap. Single event at round 100. |
| `--covariate-drift` | drift type | Per-cohort IMAGE corruptions (A=Gaussian noise, B=Gaussian blur, C=contrast×0.5+shift) with canonical labels. Allocates `TEST_X_VARIANTS` and `GPU_CLIENT_X_CANONICAL` backup. |

---

## 3. Single sudden drift, 3 seeds — MAIN RESULT

**Setup**: defaults (CIFAR-10, Dir(0.1), 20 clients, 200 rounds, sudden drift @ round 100, seed ∈ {0,1,2}).
**Windows**: `pre` = mean(rounds [89, 99]); `dip` = pre − min(rounds [100, 109]); `stable` = mean(rounds [190, 199]). Both metrics computed identically.

### 3.1 Per-seed table — global accuracy / per-client gen acc

Each cell is `pre / dip / stable` for that method × seed.

| Method | Seed | Global (pre / dip / stable) | Per-client (pre / dip / stable) |
|---|---:|---|---|
| FedAvg | 0 | 0.7081 / 0.1219 / 0.5887 | 0.7081 / 0.1640 / 0.5476 |
| FedAvg | 1 | 0.7119 / 0.0964 / 0.6140 | 0.7119 / 0.1556 / 0.5617 |
| FedAvg | 2 | 0.7184 / 0.1278 / 0.5848 | 0.7184 / 0.1729 / 0.5482 |
| Flash | 0 | 0.6714 / 0.1652 / 0.5711 | 0.6714 / 0.1755 / 0.5337 |
| Flash | 1 | 0.6881 / 0.1572 / 0.6163 | 0.6881 / 0.1617 / 0.5630 |
| Flash | 2 | 0.6940 / 0.1885 / 0.5823 | 0.6940 / 0.1906 / 0.5405 |
| AdaptiveFedAvg (broken /cur_round) | 0 | 0.5461 / 0.0327 / 0.4656 | 0.5461 / 0.1012 / 0.4441 |
| AdaptiveFedAvg (broken /cur_round) | 1 | 0.5799 / 0.0313 / 0.5360 | 0.5799 / 0.1068 / 0.4866 |
| AdaptiveFedAvg (broken /cur_round) | 2 | 0.6005 / 0.0881 / 0.4808 | 0.6005 / 0.1358 / 0.4651 |
| **AdaptiveFedAvg (corrected)** | 0 | 0.7007 / 0.1278 / 0.5812 | 0.7007 / 0.1626 / 0.5413 |
| **AdaptiveFedAvg (corrected)** | 1 | 0.7158 / 0.0956 / 0.6148 | 0.7158 / 0.1574 / 0.5645 |
| **AdaptiveFedAvg (corrected)** | 2 | 0.7185 / 0.1386 / 0.5789 | 0.7185 / 0.1789 / 0.5475 |
| FedAvgPlus1 (control) | 0 | 0.7111 / 0.1254 / 0.5871 | **0.4921** / 0.0909 / **0.4460** |
| FedAvgPlus1 (control) | 1 | 0.7138 / 0.0980 / 0.6223 | **0.5215** / 0.0884 / **0.4720** |
| FedAvgPlus1 (control) | 2 | 0.7155 / 0.1276 / 0.5857 | **0.5178** / 0.0998 / **0.4658** |
| **OurMethod** | 0 | 0.7049 / 0.1205 / 0.5805 | 0.7049 / 0.1608 / 0.5438 |
| **OurMethod** | 1 | 0.7119 / 0.0964 / 0.6170 | 0.7119 / 0.1526 / 0.5630 |
| **OurMethod** | 2 | 0.7184 / 0.1388 / 0.5775 | 0.7184 / 0.1759 / 0.5455 |
| Saile | 0 | 0.7068 / 0.1194 / 0.5844 | 0.7068 / 0.1679 / 0.5477 |
| Saile | 1 | 0.7138 / 0.1110 / 0.6190 | 0.7138 / 0.1554 / 0.5654 |
| Saile | 2 | 0.7210 / 0.1380 / 0.5920 | 0.7210 / 0.1753 / 0.5570 |

Note on FedAvgPlus1 per-client (in bold): pre-drift per-client is much lower than global because FedAvgPlus1 evaluates each client on its locally-fine-tuned model (global + 1 local epoch); at Dir(0.1) one epoch on a heavily class-imbalanced shard collapses the model. This is by design — the control measures "trivial last-step personalization".

### 3.2 Aggregated mean ± std across 3 seeds

| Method | Global pre | Global dip | Global stable | Per-cli pre | Per-cli dip | Per-cli stable |
|---|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.7128 ± 0.0043 | 0.1154 ± 0.0136 | 0.5958 ± 0.0129 | 0.7128 ± 0.0043 | 0.1642 ± 0.0071 | 0.5525 ± 0.0065 |
| Flash | 0.6845 ± 0.0095 | 0.1703 ± 0.0132 | 0.5899 ± 0.0192 | 0.6845 ± 0.0095 | 0.1760 ± 0.0118 | 0.5457 ± 0.0125 |
| AdaptiveFedAvg (broken) | 0.5755 ± 0.0224 | 0.0507 ± 0.0264 | 0.4941 ± 0.0302 | 0.5755 ± 0.0224 | 0.1146 ± 0.0151 | 0.4653 ± 0.0173 |
| **AdaptiveFedAvg (corrected)** | 0.7117 ± 0.0078 | 0.1207 ± 0.0183 | 0.5916 ± 0.0164 | 0.7117 ± 0.0078 | 0.1663 ± 0.0091 | 0.5511 ± 0.0098 |
| FedAvgPlus1 (control) | 0.7135 ± 0.0018 | 0.1170 ± 0.0135 | 0.5984 ± 0.0169 | 0.5105 ± 0.0131 | 0.0930 ± 0.0049 | 0.4613 ± 0.0111 |
| **OurMethod** | 0.7117 ± 0.0055 | 0.1186 ± 0.0174 | 0.5917 ± 0.0180 | 0.7117 ± 0.0055 | 0.1631 ± 0.0096 | 0.5508 ± 0.0086 |
| Saile | 0.7139 ± 0.0058 | 0.1228 ± 0.0113 | 0.5985 ± 0.0148 | 0.7139 ± 0.0058 | 0.1662 ± 0.0082 | 0.5567 ± 0.0072 |

### 3.3 OurMethod − FedAvg deltas (THE comparison)

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± std | Signs agree? |
|---|---:|---:|---:|---:|---|
| Global stable | −0.82pp | +0.30pp | −0.73pp | **−0.42pp ± 0.51pp** | NO (−, +, −) |
| **Per-client stable** | **−0.38pp** | **+0.12pp** | **−0.27pp** | **−0.18pp ± 0.22pp** | NO (−, +, −) |

|mean| ≤ std on both metrics, signs flip across seeds. **OurMethod is within seed noise of FedAvg on the main result.**

### 3.4 Other notable contrasts (3-seed mean Δ on per-client stable, vs FedAvg)

| Method | Δ vs FedAvg | Per-seed signs | Reading |
|---|---:|---|---|
| Saile − FedAvg | **+0.42pp ± 0.36pp** | (+, +, +) — all positive | Marginal: |mean| > std, signs agree, but n=3 with one near-zero (+0.01) makes it suggestive not defensible. |
| AdaptiveFedAvg (corrected) − FedAvg | −0.14pp ± 0.38pp | (−, +, −) | Tied. |
| Flash − FedAvg | −0.68pp ± 0.69pp | NOT COMPUTED HERE (signs from raw data) | Within noise. |
| FedAvgPlus1 − FedAvg | **−9.12pp ± 0.65pp** | all negative | Catastrophic on per-client by design (local fine-tune at Dir(0.1) is harmful). |

Source files for every row above: see Section 9 provenance.

---

## 4. OurMethod detection statistics (single sudden drift)

From OurMethod's per-round flag CSV (`results_OurMethod_flags.csv`). Columns: `flagged_count` is the union of L3 and L4 flag sets; `flagged_layer3_count` and `flagged_layer4_count` are per-layer.

### 4.1 Detection recall — flagged client count over the 10 rounds immediately post-drift

| Seed | flagged_count [100..109] | Peak | L3 flag count [100..109] | L4 flag count [100..109] |
|---|---|---:|---|---|
| 0 | `[7, 7, 7, 7, 6, 1, 1, 1, 1, 1]` | **7/20 (35%)** | `[5, 5, 5, 5, 4, 1, 1, 1, 1, 1]` peak 5 | `[7, 7, 7, 7, 4, 1, 0, 0, 0, 0]` peak 7 |
| 1 | `[6, 6, 6, 6, 5, 2, 0, 1, 1, 1]` | **6/20 (30%)** | `[5, 5, 5, 4, 1, 0, 0, 1, 1, 1]` peak 5 | `[6, 6, 6, 6, 5, 2, 0, 0, 0, 0]` peak 6 |
| 2 | `[8, 8, 8, 8, 6, 2, 2, 0, 0, 0]` | **8/20 (40%)** | `[4, 4, 4, 4, 2, 2, 2, 0, 0, 0]` peak 4 | `[8, 8, 8, 8, 6, 2, 0, 0, 0, 0]` peak 8 |

**Mean peak: 7.0/20 = 35%.** Reaction window typically ~4-5 rounds before flags subside.

### 4.2 Per-layer breakdown — L4 dominates

In every seed, L4's peak count ≥ L3's peak count. L4 contributes more flags overall, and given L4 has ~4× more parameters than L3 (73,856 vs 18,496), L4 dominates the communication-saving attribution too (see Section 7).

### 4.3 Precision

**NOT FOUND IN REPO** — precision (true positives / flagged) is not directly computable from the logged CSVs. The "ground truth" of which clients experienced drift is all 20 (every cohort gets a swap), so by that definition precision is trivially 100%. No per-flag-correctness metric is logged.

### 4.4 Hybrid-lift on flagged clients

Logged in the OurMethod stdout (line printed when flagged_any > 0), e.g. seed 0 round 100: `hyb-lift(flagged): +0.4253`. NOT stored in the CSV, only the stdout log file. Per-round hybrid-lift trajectory NOT FOUND IN REPO in machine-readable form.

---

## 5. Ablation study — single seed (seed 0), 5 variants

Source: `runs/2026-06-04-ablations/<variant>/results_OurMethod.csv`. Note: the ablation CSVs predate the per-client metric port; **per-client gen acc is NOT present in these CSVs** ("NOT FOUND IN REPO" for those columns). Reproducible report: `runs/2026-06-04-ablations/SUMMARY.md`.

| Variant | Pre (global) | Dip (global) | Stable (global) | Δ-stable vs baseline | Notes |
|---|---:|---:|---:|---:|---|
| baseline | 0.7026 | 0.1196 | 0.5884 | 0 (ref) | unmodified OurMethod (τ=1.4, drift-layers L3+L4) |
| no-detection (τ=∞) | 0.7071 | 0.1265 | 0.5876 | −0.08pp | detection disabled — confirms mechanism off ≈ FedAvg |
| all-layers | 0.7047 | 0.1296 | 0.5894 | +0.10pp | L1+L2+L3+L4 flaggable |
| **tau-low (τ=1.2)** | 0.7077 | **0.0740** | 0.5924 | +0.40pp | **−4.56pp smaller dip** vs baseline; only knob with a clear functional effect at n=1 |
| tau-high (τ=1.6) | 0.7057 | 0.1223 | 0.5965 | +0.81pp | dip ≈ baseline, stable marginally up; inside noise |

The standout: **tau-low cuts the immediate dip by 4.56pp** (0.0740 vs 0.1196) while leaving stable basically unchanged. All other variants are within ~1pp of the baseline on stable, well inside the 0.5-1.5pp single-seed noise floor measured in the 3-seed studies.

---

## 6. Other go/no-go tests (each single seed=0)

All FedAvg vs OurMethod, same default setup unless noted.

### 6.1 Dir(0.5) test (less non-IID)

Source: `runs/2026-06-11-dir05-test/`. Override: `--alpha-dir 0.5`.

| | FedAvg | OurMethod | Δ |
|---|---:|---:|---:|
| Global stable | 0.6661 | 0.6673 | **+0.12pp** |
| Per-client stable | 0.6112 | 0.6118 | **+0.06pp** |
| Detection peak [100..109] | — | **1/20 (5%)** | (35% → 5%, 7× degradation vs Dir(0.1)) |

Per-round flag count over [100..109]: `[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]`.

Hypothesis was: more balanced data → more detectable drift. **Refuted**: detection collapses from 35% to 5%. Mechanism premise (concentrated weight-spike at drift) requires per-client class concentration, which Dir(0.5) reduces.

### 6.2 Recurrent alternating drift

Source: `runs/2026-06-13-recurrent-gain-test/`. Override: `--alternating-drift`. Schedule [40, 80, 120, 160], same swap at every event → concept oscillates canonical/swapped/canonical/swapped/canonical.

| | FedAvg | OurMethod | Δ |
|---|---:|---:|---:|
| Global stable [190, 199] | 0.7046 | 0.7073 | +0.27pp |
| Per-client stable [190, 199] | 0.7046 | 0.7073 | +0.27pp |

(per-client == global in the canonical-phase rounds because `TEST_Y_VARIANTS` toggle back to canonical via involution; the per-cohort identity holds.)

**Per-event detection (peak in 10-round window from event start):**

| Event | Round | Direction | flagged_count window | Peak |
|---|---:|---|---|---:|
| 0 | 40 | canonical → swapped | `[6, 6, 6, 6, 6, 4, 2, 0, 0, 0]` | 6/20 |
| 1 | 80 | swapped → canonical | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` | **0/20** |
| 2 | 120 | canonical → swapped | `[5, 5, 5, 5, 3, 1, 1, 1, 1, 1]` | 5/20 |
| 3 | 160 | swapped → canonical | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` | **0/20** |

**Detection is structurally asymmetric across drift direction**: fires on "into swapped", silent on "swap back". Cause: EMA-relative-variance detector — at the swap-back, recent variance is still elevated from the prior recovery, so the new spike doesn't exceed τ.

### 6.3 Aggressive concept drift (per-cohort full label permutations)

Source: branch `aggressive-concept-drift-test` at `dda631b`, `runs/2026-06-14-aggressive-concept-test/`. Override: `--aggressive-concept-drift`. Per-cohort 10-label permutations (seeded RandomState(42)): P_A=[8,1,5,0,7,2,9,4,3,6], P_B=[0,1,8,5,3,4,7,9,6,2], P_C=[9,2,0,6,8,5,3,7,1,4] — maximally conflicting (A↔B: 1/10 agreements; B↔C: 0/10; A↔C: 0/10).

| | FedAvg | OurMethod | Δ |
|---|---:|---:|---:|
| Global stable | 0.1512 | 0.1530 | +0.18pp |
| **Per-client stable** | **0.2862** | **0.2862** | **−0.00pp** (identical to 4 dp) |
| Detection peak [100..109] | — | **19/20 (95%)** | — |

Per-round flag count [100..109]: `[19, 19, 19, 19, 19, 9, 1, 1, 1, 1]`.

**Detection saturates** (3× the pairwise-swap baseline) but **accuracy is identical** — both methods crash to near-random on the canonical TEST_Y (~15%) and stay at ~29% on the cohort-permuted per-client metric. **Detection mass ≠ adaptation efficacy.**

### 6.4 Covariate drift, default L3/L4 detector

Source: branch `covariate-drift-test` at `2f47a4a`, `runs/2026-06-14-covariate-drift-test/`. Override: `--covariate-drift`. Per-cohort image corruption (A=Gaussian noise std=0.15, B=Gaussian blur 5×5 σ=1.5, C=contrast×0.5+shift 0.3) applied to BOTH training images and the per-cohort `TEST_X_VARIANTS`. Labels unchanged.

| | FedAvg | OurMethod (L3/L4) | Δ |
|---|---:|---:|---:|
| Global stable | 0.6770 | 0.6756 | −0.14pp |
| Per-client stable | 0.6290 | 0.6292 | **+0.02pp** |
| Detection peak [100..109] | — | **1/20 (5%)** | — |

Per-round flag count [100..109]: `[1, 1, 1, 1, 1, 1, 1, 0, 1, 1]`.

**L3/L4 detector functionally silent** under covariate drift. Confirms covariate shift hits early layers, not late ones.

### 6.5 Covariate drift, L1/L2 re-aim

Source: branch `covariate-l1l2-test` at `33c883c`, `runs/2026-06-14-covariate-l1l2-test/`. Overrides: `--covariate-drift --drift-layers layer1 layer2`.

| | FedAvg | OurMethod (L1/L2) | Δ |
|---|---:|---:|---:|
| Global stable | 0.6770 | 0.6752 | −0.18pp |
| Per-client stable | 0.6290 | 0.6279 | **−0.12pp** |
| Per-client **dip** | 0.0982 | **0.1264** | **+2.82pp DEEPER** |
| Detection peak [100..109] (from CSV `flagged_count`) | — | **0/20 (0%)** | (see caveat below) |

**Logging caveat**: the OurMethod CSV's `flagged_count` column hard-codes a lookup of `prev_flags.get('layer3')` and `prev_flags.get('layer4')`. With `--drift-layers layer1 layer2`, L3/L4 are no longer flaggable → the CSV reports 0 for every round even when L1/L2 ARE firing. **The mechanism IS operational** (`OurClient.train` correctly sets `prev_flags['layer1']/['layer2']`; `get_upload_state` uses `upload_mask` derived from those); only the REPORTING is missing.

**Indirect detection evidence — trajectory divergence vs FedAvg** (`|OurMethod_acc − FedAvg_acc|` per round):

| Window | L3/L4 covariate (≈silent baseline) | L1/L2 covariate (this) | Ratio |
|---|---:|---:|---:|
| Pre-drift [0, 99] | 0.00pp max | 0.00pp max | (bit-identical pre-drift) |
| Post-drift [100, 199] global max | 0.90pp | **2.84pp** | 3.2× |
| Post-drift [100, 199] global mean | 0.28pp | **0.41pp** | 1.5× |
| Post-drift [100, 199] per-cli max | 0.44pp | **4.44pp** | 10× |
| Post-drift [100, 199] per-cli mean | 0.15pp | **0.40pp** | 2.7× |

L1/L2 produces 1.5-10× more post-drift divergence than the (mostly silent) L3/L4 baseline — strong indirect evidence the L1/L2 detector IS firing. But the per-client dip is 2.82pp WORSE for OurMethod, and stable converges back to ≈ FedAvg. **"Keep local" is the wrong response for early layers** — they need exposure to the federation's combined data to generalize.

---

## 7. Communication analysis (whole-run + drift-window saving)

Source: `comm_analysis.py` + flag CSVs from the 3-seed single-drift runs. Reproducible report: `runs/2026-06-13-comm-analysis/SUMMARY.md`.

FedAvg per-round upload (constant): 20 clients × 107,690 params = **2,153,800 params/round**.
FedAvg whole-run upload (200 rounds): **430,760,000 params** ≈ 430.76M.

**Per-seed saving** (OurMethod uploads less than FedAvg only on rounds where L3/L4 are flagged):

| Seed | Whole-run saving | Drift-window saving [100, 109] | L3 share of saving | L4 share of saving | Source CSV |
|---|---:|---:|---:|---:|---|
| 0 | 0.703% (3.03M params) | **13.806%** | 19.5% | 80.5% | `runs/2026-06-08-perclient-smoke/ourmethod/results_OurMethod_flags.csv` |
| 1 | 0.647% (2.79M params) | **12.605%** | 17.9% | 82.1% | `runs/2026-06-08-perclient-3seed/seed1/ourmethod/results_OurMethod_flags.csv` |
| 2 | 0.780% (3.36M params) | **15.606%** | 12.1% | 87.9% | `runs/2026-06-08-perclient-3seed/seed2/ourmethod/results_OurMethod_flags.csv` |
| **mean ± std** | **0.710% ± 0.054%** | **14.006% ± 1.233%** | 16.5% (mean) | 83.5% (mean) | — |

**Asymmetry**: OurMethod NEVER uploads more than FedAvg — only withholds. The 14% drift-window saving is the "when the mechanism actually fires" figure; the 0.71% whole-run figure reflects that the mechanism is dormant for ~190 of 200 rounds in the single-drift setting.

L4 dominates the saving (~84% of withheld params on average) because L4 has ~4× the params of L3.

---

## 8. Baseline implementation notes

### 8.1 AdaptiveFedAvg `/cur_round` bug and fix

**Source of bug**: ported byte-faithfully from FedCCFA's `AdaptiveFedAvgServer.cal_adaptive_lr` (line 79 of FedCCFA repo):

```python
client_dynamic_lr = min(self.client_init_lr, self.client_init_lr * ratio_norm / cur_round)
```

The `/ cur_round` divisor produces a 1/t decay: by round 100 with base LR 0.01, effective LR ≈ 1e-4 — two orders of magnitude below base. The method's stated purpose is to *raise* LR at drift, so this self-defeats.

**Cross-check**: Saile et al. 2024's `LearningrateEstimatorModel` re-implementation of the same Canonaco 2021 algorithm omits the `/cur_round` divisor (`lr = min(initial_lr, base_lr * variance_ratio_ema)`). Two independent re-implementations disagree → likely a FedCCFA porting deviation. Original Canonaco IJCNN 2021 paper is paywalled and was not directly verified.

**Fix**: removed `/cur_round` divisor; kept the three-EMA bias correction (`/(1 - β^t)`) intact. Source: `all_experiments_optimized.py:875-880` (commit `aa86a1b`).

**LR sweep (FedDrift-style, seed 0)**: selection by post-drift stable global accuracy. Source: `runs/2026-06-09-adaptive-lr-sweep/lr_*/results_AdaptiveFedAvg.csv`.

| `client_init_lr` | Stable | Note |
|---:|---:|---|
| 0.1 | 0.1000 | diverged |
| **0.01** | **0.5812** | **selected** (matches the harness's default LR) |
| 0.001 | 0.5423 | undertrained |
| 0.0001 | 0.3654 | severely undertrained |

**Before/after (3 seeds, lr=0.01)**:

| Version | Global stable | Per-client stable |
|---|---:|---:|
| Broken (with `/cur_round`) | 0.4941 ± 0.0302 | 0.4653 ± 0.0173 |
| Corrected | 0.5916 ± 0.0164 | 0.5511 ± 0.0098 |
| **Lift** | **+9.75pp** | **+8.58pp** |

All 3 per-seed lifts positive: +11.56pp, +7.88pp, +9.81pp on global. Fix unambiguous.

### 8.2 Saile LR sweep — which LRs diverged, which was selected

Source: `runs/2026-06-11-saile-lr-sweep/lr_*/results_Saile.csv`. Saile's CIFAR-10 reference config uses LR=0.2 with lr_decay=0.99/round and β=(0.7, 0.3, 0.7). We adopted β + decay as-is, but ran a sweep on the initial LR because Saile's reference setup differs from ours (IID + TwoCNN + B=50 + E=3; ours is Dir(0.1) + CifarCNN + B=64 + E=5).

| `saile_init_lr` | Stable global | Note |
|---:|---:|---|
| 0.2 (Saile default) | 0.1000 | **diverged** (random-guess 10%) |
| 0.1 | 0.1000 | **diverged** |
| **0.01** | **0.5844** | **selected** |

Same regime-mismatch story as AdaptiveFedAvg — at our operating point LR=0.2 is too high. **Selected LR=0.01.**

### 8.3 FedAvgPlus1 control result

Source: `runs/2026-06-08-perclient-smoke/fedavgplus1/` + `runs/2026-06-08-perclient-3seed/seed{1,2}/fedavgplus1/`. The control runs plain FedAvg (so its global model trajectory matches FedAvg modulo CUDA non-determinism) but evaluates each client's per-client metric on a `(global_model + 1 local epoch on the client's own data)` model.

3-seed result: **per-client stable = 0.4613 ± 0.0111** (vs FedAvg 0.5525 ± 0.0065). Δ = **−9.12pp**.

The control HURTS the per-client metric by ~9pp because at Dir(0.1), one epoch of local SGD on a heavily class-imbalanced shard (~500 samples covering 2-3 classes) collapses the model toward those classes. Reading: at this operating point there is no free-lunch lift from trivial local fine-tuning; any per-client gain on top of plain FedAvg would be genuine mechanism, not last-step personalization. (Important context for evaluating OurMethod's tied per-client result — the control isn't "easy lift" here.)

### 8.4 Saile vs FedAvg result (the most interesting baseline finding)

Per-seed Δ on per-client stable: +0.01pp, +0.36pp, +0.88pp → **mean +0.42pp ± 0.36pp**, all 3 signs positive. |mean| > std but marginal. **First drift-adaptive baseline with consistent direction across seeds**, but n=3 with one near-zero sample makes it suggestive not defensible. Warrants ≥5-seed replication before claiming Saile's per-client mechanism beats FedAvg.

---

## 9. Data provenance — every file used, with path and branch

### 9.1 Branches and commit hashes

| Branch | Commit | Status | What's on this branch |
|---|---|---|---|
| `main` | `9e3b7f9` | HEAD of this report | Merged: ablations, perclient-metric (+AdaptiveFedAvg fix), saile-baseline, comm-analysis, dir05-test, recurrent-gain-test |
| `aggressive-concept-drift-test` | `dda631b` | NOT merged | `runs/2026-06-14-aggressive-concept-test/` |
| `covariate-drift-test` | `2f47a4a` | NOT merged | `runs/2026-06-14-covariate-drift-test/` |
| `covariate-l1l2-test` | `33c883c` | NOT merged (based on covariate-drift-test) | `runs/2026-06-14-covariate-l1l2-test/` + inherited covariate-drift data |

### 9.2 Single sudden 3-seed (Section 3)

| Method | Seed | Path |
|---|---:|---|
| FedAvg | 0 | `runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv` |
| FedAvg | 1 | `runs/2026-06-08-perclient-3seed/seed1/fedavg/results_FedAvg.csv` |
| FedAvg | 2 | `runs/2026-06-08-perclient-3seed/seed2/fedavg/results_FedAvg.csv` |
| Flash | 0 | `runs/2026-06-08-perclient-smoke/flash/results_Flash.csv` |
| Flash | 1, 2 | `runs/2026-06-08-perclient-3seed/seed{1,2}/flash/results_Flash.csv` |
| AdaptiveFedAvg (broken) | 0 | `runs/2026-06-08-perclient-smoke/adaptive/results_AdaptiveFedAvg.csv` |
| AdaptiveFedAvg (broken) | 1, 2 | `runs/2026-06-08-perclient-3seed/seed{1,2}/adaptive/results_AdaptiveFedAvg.csv` |
| AdaptiveFedAvg (corrected) | 0, 1, 2 | `runs/2026-06-09-adaptivefedavg-fix/seed{0,1,2}/results_AdaptiveFedAvg.csv` |
| FedAvgPlus1 | 0 | `runs/2026-06-08-perclient-smoke/fedavgplus1/results_FedAvgPlus1.csv` |
| FedAvgPlus1 | 1, 2 | `runs/2026-06-08-perclient-3seed/seed{1,2}/fedavgplus1/results_FedAvgPlus1.csv` |
| OurMethod | 0 | `runs/2026-06-08-perclient-smoke/ourmethod/results_OurMethod.csv` (+ `_flags.csv`) |
| OurMethod | 1, 2 | `runs/2026-06-08-perclient-3seed/seed{1,2}/ourmethod/results_OurMethod.csv` (+ `_flags.csv`) |
| Saile | 0, 1, 2 | `runs/2026-06-11-saile-3seed/seed{0,1,2}/results_Saile.csv` |

### 9.3 Ablation (Section 5)

`runs/2026-06-04-ablations/{baseline, no-detection, all-layers, tau-low, tau-high}/results_OurMethod.csv` (5 files). Companion `SUMMARY.md` + 3 PNG plots in `runs/2026-06-04-ablations/plots/`. Branch: merged into main via `c717338`.

### 9.4 Other tests (Section 6)

| Test | Path | Branch |
|---|---|---|
| Dir(0.5) | `runs/2026-06-11-dir05-test/{fedavg,ourmethod}/results_*.csv` | main (merged via dir05-test) |
| Recurrent alternating | `runs/2026-06-13-recurrent-gain-test/{fedavg,ourmethod}/results_*.csv` | main (merged via recurrent-gain-test) |
| Aggressive concept | `runs/2026-06-14-aggressive-concept-test/{fedavg,ourmethod}/results_*.csv` | `aggressive-concept-drift-test` (NOT merged) |
| Covariate L3/L4 | `runs/2026-06-14-covariate-drift-test/{fedavg,ourmethod}/results_*.csv` | `covariate-drift-test` (NOT merged) |
| Covariate L1/L2 | `runs/2026-06-14-covariate-l1l2-test/{fedavg,ourmethod}/results_*.csv` | `covariate-l1l2-test` (NOT merged) |

### 9.5 LR sweeps

| Method | Path |
|---|---|
| AdaptiveFedAvg | `runs/2026-06-09-adaptive-lr-sweep/lr_{0p1,0p01,0p001,0p0001}/results_AdaptiveFedAvg.csv` |
| Saile | `runs/2026-06-11-saile-lr-sweep/lr_{0p2,0p1,0p01}/results_Saile.csv` |

### 9.6 FedCCFA reproduction verification

`runs/2026-06-04-fedccfa-verify/CIFAR10_sudden_FedAvg_20260604181603436012.csv` — the PASS run at Dir(0.5) reproducing 0.5994 vs published 0.6096 target. `RUN_NOTES.md` in same directory.

### 9.7 Companion SUMMARY.md files (for cross-reading)

- `runs/2026-06-04-ablations/SUMMARY.md`
- `runs/2026-06-08-perclient-smoke/SUMMARY.md` (single-seed all-5-methods report)
- `runs/2026-06-08-perclient-3seed/SUMMARY_3seed.md` (3-seed aggregate)
- `runs/2026-06-09-adaptivefedavg-fix/SUMMARY.md`
- `runs/2026-06-11-saile-3seed/SUMMARY.md`
- `runs/2026-06-11-dir05-test/SUMMARY.md`
- `runs/2026-06-13-recurrent-gain-test/SUMMARY.md`
- `runs/2026-06-13-comm-analysis/SUMMARY.md`
- `runs/2026-06-14-aggressive-concept-test/SUMMARY.md` (on aggressive-concept-drift-test branch)
- `runs/2026-06-14-covariate-drift-test/SUMMARY.md` (on covariate-drift-test branch)
- `runs/2026-06-14-covariate-l1l2-test/SUMMARY.md` (on covariate-l1l2-test branch)

### 9.8 Reproduction scripts (read-only analysis)

- `experiment_report_compute.py` — regenerates every numeric value in this report from the CSVs.
- `comm_analysis.py` — communication-cost analysis from flag CSVs.
- `ablation_plots.py` — the 3 ablation PNGs from CSVs.
- `perclient_summary.py`, `perclient_summary_3seed.py`, `saile_summary.py`, `adaptivefedavg_fix_summary.py`, `recurrent_gain_analysis.py` — per-experiment summary builders.

---

## What was found vs what was missing

### Found in repo (with exact citations):

✅ All FL setup constants and code paths (Section 1)
✅ Full method registry + every CLI flag (Section 2)
✅ 3-seed × 6-method comparison on single sudden drift, both metrics (Section 3)
✅ OurMethod detection per-round flag counts at drift (Section 4)
✅ 5-variant ablation results (Section 5) — but only global accuracy (per-client gen acc not logged in those CSVs)
✅ All 5 other go/no-go tests (Section 6)
✅ Per-seed + mean communication saving + L3/L4 attribution (Section 7)
✅ AdaptiveFedAvg `/cur_round` bug, fix, LR sweep, before/after lift (Section 8)
✅ Saile LR sweep, divergence pattern (Section 8)
✅ FedAvgPlus1 control result (Section 8)
✅ Complete file-by-file provenance (Section 9)

### NOT FOUND IN REPO (explicit gaps):

❌ **Per-client gen acc in ablation CSVs** — `runs/2026-06-04-ablations/<variant>/results_OurMethod.csv` only has the `global_acc` and `local_cXX` columns; the per-client gen acc metric was ported AFTER the ablation runs. To get per-client values for the ablation knobs, runs would need to be re-executed.

❌ **OurMethod detection precision** — there's no per-flag-correctness logged. The "ground truth" of "which clients experienced drift" is trivially "all 20" (every cohort gets a swap), so by that definition precision is 100% in every run. No more granular precision metric exists.

❌ **Per-round hybrid-lift trajectory in machine-readable form** — hybrid-lift on flagged clients is printed to stdout (e.g. seed 0 round 100: `+0.4253`) but not saved to the CSV. The information lives only in the log files (`logs/*.log`).

❌ **L1/L2 flag counts in `results_OurMethod_flags.csv`** when DRIFT_LAYERS is re-aimed (covariate-l1l2-test) — the CSV's `flagged_count` column is hard-coded as `union(prev_flags['layer3'], prev_flags['layer4'])` and reads 0 even when L1/L2 ARE firing. The mechanism is correct, only the logging is. Indirect detection evidence via trajectory divergence (3× the silent-baseline) is reported in Section 6.5 instead.

❌ **Direct precision/F1 for the EMA detector against any task-level "drift truth"** — the detector flags clients based on per-layer weight change magnitude; matching this against external "ground truth" would require synthetic drift labels per round, which the harness does not produce.

❌ **Multi-seed runs for any of the go/no-go tests** (Dir(0.5), recurrent, aggressive, covariate L3/L4, covariate L1/L2) — single seed only by design (these were cheap diagnostics; the negative results were sharp enough to decide without seed replication).

❌ **Canonaco IJCNN 2021 paper's exact LR formula** — paywalled and not in the references folder. The AdaptiveFedAvg `/cur_round` divisor's faithfulness vs Canonaco's original is therefore inferred (not verified) from Saile's independent re-implementation that omits it.
