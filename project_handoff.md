# Project Handoff Document

**Project:** Master's Thesis on Federated Learning Under Concept Drift
**Student:** Surafel Sentayehu, MSc AI, Addis Ababa University
**Advisor:** Dr. Adane Letta
**Target completion:** Summer 2026

---

## 1. The thesis at a glance

### 1.1 Title

"Drift-Triggered Selective Layer Adaptation in Federated Learning Under Distributed Real Concept Drift"

### 1.2 Problem we are solving

In federated learning, multiple clients train a shared model on their own private data. The data each client sees can change over time — for example, what counted as "spam" last year might not be spam this year. When this happens differently on different clients (some clients see one kind of change, others see different changes), it is called **distributed concept drift**.

The big issue: standard federated learning averages all client updates together. If half the clients are adapting to one kind of change and the other half to a different change, averaging them produces a confused model that helps nobody. The model's accuracy drops at the drift moment and never fully recovers.

### 1.3 Research gap we identified

Looking at the literature (2024-2026), no published method does what we propose: let each client decide, **layer by layer, every round**, whether the drift it is experiencing has affected that specific layer enough to keep it locally vs share it back.

Existing methods either:
- Treat the entire model uniformly (no per-layer decisions)
- Run drift detection only on the server (clients have no autonomy)
- Use fixed architectural rules (same handling for every client every round)

We bridge this gap with a method that is **client-side, per-layer, per-round, and reversible.**

### 1.4 Our solution architecture

A five-step protocol that each client runs every round:

**Step 1 — Selective Sync**
Receive the global model. For layers the client had flagged as "drifted" last round, keep the locally adapted version. For unflagged layers, accept the global version. Classifier always accepts global.

**Step 2 — Local Training**
Standard SGD for 5 epochs on local data. Save the trained state.

**Step 3 — Weight Divergence**
For each layer, compute how much the trained weights moved from the global weights received at the start of the round, normalized by the global weight magnitude. This is one number per layer.

**Step 4 — EMA-Based Drift Detection**
Compare the current divergence to the layer's own exponential moving average baseline. If the ratio exceeds a threshold (1.4 in our setup) AND the layer is one of the upper layers (layer3 or layer4), flag it. Lower layers (layer1, layer2) are never flagged because they encode general features that should always be shared.

**Step 5 — Selective Upload**
Send only the unflagged layers and the classifier back to the server. The flagged layers stay on the client and continue adapting independently.

**Server side:** Per-layer averaging only from clients who uploaded that layer. Layers that nobody uploaded keep their current global value. Classifier always averages from all clients.

### 1.5 Theoretical grounding

Two ICLR papers directly support our design:

- **Lee et al. (ICLR 2023, Surgical Fine-Tuning):** Distribution shifts affect different layers disproportionately. Upper layers need more adaptation under concept-type drift.
- **Ramasesh et al. (ICLR 2021, Anatomy of Catastrophic Forgetting):** Lower layers learn general features that transfer across distributions; upper layers learn task-specific representations that need re-learning.

This is why layer1 and layer2 are stable while layer3 and layer4 are drift-flaggable.

---

## 2. The experimental setup

### 2.1 Dataset and model

- **Dataset:** CIFAR-10 (50,000 training images, 10,000 test images, 10 classes)
- **Clients:** 20 with Dirichlet partition alpha = 0.1 (highly non-IID, each client mostly sees 2-3 classes)
- **Model:** CifarCNN from FedCCFA's codebase — 4 hidden layers (3 conv + 1 FC) + 1 classifier, total 107,690 parameters
- **Training:** SGD momentum=0.9, lr=0.01, batch_size=64, 5 local epochs per round, 200 rounds total

### 2.2 Drift protocol (FedCCFA's sudden_drift)

At round 100, label swaps applied to client groups based on `client_id % 10`:

| Group | Client IDs | Label swap |
|---|---|---|
| Group A | 0, 1, 2, 10, 11, 12 (6 clients) | 1 ↔ 2 |
| Group B | 3, 4, 5, 13, 14, 15 (6 clients) | 3 ↔ 4 |
| Group C | 6, 7, 8, 9, 16, 17, 18, 19 (8 clients) | 5 ↔ 6 |

**All 20 of 20 clients are drifted, just in different ways.** No stable clients.

### 2.3 Baselines

| # | Method | Year | Detection | Adaptation |
|---|---|---|---|---|
| 1 | FedAvg | 2017 | None | None |
| 2 | Flash | 2023 | Server-side gradient variance | Server-side Adam-like step |
| 3 | Adaptive-FedAvg | 2021 | Server-side parameter variance | Client-side via adjusted lr |
| 4 | OurMethod | 2026 | Client-side per-layer divergence | Client-side per-layer upload mask |

CDA-FedAvg was attempted but failed (confidence-based detection cannot distinguish drift from natural training fluctuations on highly non-IID image data). Documented as a negative finding, not in the comparison table.

### 2.4 Single-seed results (already obtained)

| Method | Pre-drift | Dip | Post-drift stable |
|---|---|---|---|
| FedAvg | 70.33% | 11.96% | 57.92% |
| Flash | 67.93% | 17.87% | 57.63% |
| Adaptive-FedAvg | 54.90% | 2.02% | 46.80% |
| **OurMethod** | **71.24%** | **11.07%** | **59.75%** |

OurMethod wins on 3 of 4 metrics. No method fully recovers — this matches the FedCCFA paper's prediction that no single-model method can recover from this kind of drift.

OurMethod detection at drift moment: 7 of 20 drifted clients flagged. By round 120, all flags cleared.

### 2.5 What still needs to happen

- Multi-seed runs (seeds 1, 2) for all 4 methods — confirms 1-2% gains are above noise
- 5 ablation studies (no-detection, all-layers, tau sweep, alpha sweep, warmup sweep)
- Per-client local accuracy logging
- Communication cost measurement

---

## 3. Recent work — speed optimization attempt

### 3.1 Why we tried to optimize

Previous experimental timings were very slow:
- RTX 4050 (local): ~8 hours per method
- A100 PCIE (vast.ai): ~4 hours per method (16 hours for all 4)
- Colab T4: ~5 hours per method

This made multi-seed runs prohibitively expensive in both time and money.

### 3.2 Diagnosis — CPU bottleneck, not GPU

Profiling showed the GPU was idle 60-70% of the time. The bottleneck was the CPU data pipeline:
- Loading image batches from disk
- Applying random crop/flip/normalize transforms
- Moving data to GPU memory

Throwing a faster GPU at this did not help much — going from RTX 4050 to A100 only halved the time despite the A100 being 5x faster on compute. The GPU was waiting for the CPU.

### 3.3 The optimization we applied

Three small changes to the DataLoader configuration:

**Change 1 — Parallel data loading:**
```python
num_workers=NUM_WORKERS   # parallel CPU workers prepare batches
```
Where `NUM_WORKERS = min(4, os.cpu_count() or 1)` adapts to the machine.

**Change 2 — Pinned memory transfer:**
```python
pin_memory=True   # direct CPU-to-GPU transfer, no intermediate copy
```

**Change 3 — Persistent workers across epochs:**
```python
persistent_workers=True   # workers stay alive between epochs (saves spawn time)
```

These changes do not affect training math at all. Same model, same hyperparameters, same convergence. Only the data loading is faster.

### 3.4 Additional logging features added

- **TeeLogger:** All stdout/stderr is captured to a timestamped log file while still printing to terminal
- **Per-client per-round flag tracking:** OurMethod now produces `results_OurMethod_flags.csv` with which specific client IDs flagged which specific layers in every single round
- **Enhanced round output:** Around the drift moment (rounds 100-120), prints flagged client IDs explicitly

### 3.5 Expected speedup

| Setup | Original code | Optimized code |
|---|---|---|
| RTX 4050 local (8 cores) | ~8 hr/method | ~5 hr/method |
| Colab T4 (2 cores) | ~5 hr/method | ~4-5 hr/method (only 2 workers possible) |
| GCP L4 (4 cores) | ~3-4 hr/method | ~1.5-2 hr/method |
| vast.ai A100 (8+ cores) | ~4 hr/method | ~1 hr/method |

The optimization gives roughly 2x speedup when num_workers matches available CPU cores. On systems with few cores (like Colab's 2 vCPUs) the benefit is smaller.

### 3.6 Status at end of session

- Optimized file `all_experiments_optimized.py` is ready
- GCP L4 VM created in northamerica-northeast2-b with PyTorch pre-installed, scipy installed
- VM external IP: 34.130.221.101 (will change when VM is stopped and restarted)
- SSH access works via direct SSH with the gcp_key file (gcloud's bundled PuTTY is broken)
- A Colab test run was started but Colab only has 2 vCPUs so the optimization is not fully effective there

### 3.7 What to do next

Run the optimized file on the GCP L4 VM:

```bash
ssh -i "$env:USERPROFILE\.ssh\gcp_key" suraf@<vm-ip>
nohup python3 -u all_experiments_optimized.py > nohup.out 2>&1 &
tail -f experiment_log_*.txt
```

Estimated runtime: 6-8 hours for all 4 methods on 1 seed.
Estimated cost: ~$5 at $0.71/hour.

Stop the VM as soon as experiments finish to stop billing.

---

## 4. Code reference

### 4.1 Files involved

| File | Purpose |
|---|---|
| `all_experiments.py` | Original code, runs all 4 methods sequentially |
| `all_experiments_optimized.py` | Optimized version with DataLoader speedups + logging |
| `OurMethod_entity.py` | Client and Server classes for FedCCFA repo integration |
| `OurMethod_method.py` | Training loop wrapper for FedCCFA |
| `OurMethod.yaml` | Configuration for FedCCFA integration |

### 4.2 The exact changes between original and optimized

**Added imports:**
```python
import sys
from datetime import datetime
```

**Added at top (adaptive worker count):**
```python
NUM_WORKERS = min(4, os.cpu_count() or 1)
```

**Added TeeLogger class (captures stdout to log file):**
```python
class TeeLogger:
    def __init__(self, file_path, original_stream):
        self.file   = open(file_path, 'a', buffering=1)
        self.stream = original_stream
    def write(self, message):
        self.file.write(message)
        self.stream.write(message)
    def flush(self):
        self.file.flush()
        self.stream.flush()
    def isatty(self):
        return self.stream.isatty()

_log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE       = f"experiment_log_{_log_timestamp}.txt"
sys.stdout = TeeLogger(LOG_FILE, sys.__stdout__)
sys.stderr = TeeLogger(LOG_FILE, sys.__stderr__)
```

**Changed get_loader (added 3 parameters):**

Original:
```python
def get_loader(client_sets, cid):
    return DataLoader(client_sets[cid], batch_size=BATCH_SIZE,
                      shuffle=True, drop_last=True)
```

Optimized:
```python
def get_loader(client_sets, cid):
    return DataLoader(client_sets[cid], batch_size=BATCH_SIZE,
                      shuffle=True, drop_last=True,
                      num_workers=NUM_WORKERS, pin_memory=True,
                      persistent_workers=(NUM_WORKERS > 0))
```

**Changed global_loader:**

Original:
```python
global_loader = DataLoader(global_test_ds, batch_size=512, shuffle=False)
```

Optimized:
```python
global_loader = DataLoader(global_test_ds, batch_size=512, shuffle=False,
                           num_workers=min(2, NUM_WORKERS), pin_memory=True)
```

**Added per-client flag logging in run_our_method:**

The OurMethod runner now also produces `results_OurMethod_flags.csv` with per-round per-client flag tracking. Prints flagged client IDs explicitly around the drift moment.

### 4.3 The 5-step OurMethod protocol in code (pseudocode)

```python
# Per-client state, persisted across rounds
ema_baselines = None             # EMA of weight divergence per layer
prev_flags    = {all False}      # last round's flag decisions
local_state   = None             # locally trained weights
rounds_seen   = 0

LAYER_GROUPS  = {'layer1', 'layer2', 'layer3', 'layer4'}
DRIFT_LAYERS  = {'layer3', 'layer4'}      # only upper layers flaggable
STABLE_LAYERS = {'layer1', 'layer2'}      # always uploaded
CLASSIFIER    = 'fc'                       # always uploaded, always overwritten

for each round r:
    # STEP 1: Selective Sync
    new_state = copy(global_state)
    if local_state is not None:
        for layer in LAYER_GROUPS:
            if prev_flags[layer]:
                new_state[layer] = local_state[layer]
    new_state[CLASSIFIER] = global_state[CLASSIFIER]
    model.load(new_state)

    # STEP 2: Local Training
    SGD train 5 epochs
    local_state = copy(model.state())

    # STEP 3: Weight Divergence
    for layer in LAYER_GROUPS:
        d[layer] = ||local[layer] - global[layer]||_F / ||global[layer]||_F

    # STEP 4: EMA Drift Detection
    rounds_seen += 1
    warmup = rounds_seen <= 10
    if ema_baselines is None:
        ema_baselines = {layer: d[layer]}

    new_flags = {}
    for layer in LAYER_GROUPS:
        ratio = d[layer] / ema_baselines[layer]
        if warmup:
            ema_baselines[layer] = d[layer]
            new_flags[layer] = False
        else:
            ema_baselines[layer] = 0.3 * d[layer] + 0.7 * ema_baselines[layer]
            if layer in DRIFT_LAYERS and ratio > 1.4:
                new_flags[layer] = True
            else:
                new_flags[layer] = False
    prev_flags = new_flags

    # STEP 5: Selective Upload
    upload_state = {}
    for layer in LAYER_GROUPS:
        if not new_flags[layer]:
            upload_state[layer] = local_state[layer]
    upload_state[CLASSIFIER] = local_state[CLASSIFIER]
    send_to_server(upload_state)

# Server: per-layer FedAvg over uploaders only
for layer in LAYER_GROUPS:
    uploaders = [c for c in clients if c uploaded this layer]
    if uploaders:
        global[layer] = weighted_avg(uploaders, weight by data_size)
    # else: keep current global[layer] unchanged
global[CLASSIFIER] = weighted_avg(all clients)
```

---

## 5. Key configuration parameters

### 5.1 OurMethod-specific hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| EMA alpha | 0.3 | Tuned empirically — balances responsiveness and stability |
| tau (flag threshold) | 1.4 | Tuned empirically — sharp enough to catch drift, conservative enough to avoid false positives |
| warmup_rounds | 10 | Long enough for EMA baseline to stabilize, short enough to protect before drift at round 100 |
| flag-eligible layers | layer3, layer4 only | Grounded in Lee 2023 + Ramasesh 2021 + own diagnostics |
| classifier handling | always global | Preserves shared label semantics across federation |

### 5.2 Shared experiment parameters

| Parameter | Value |
|---|---|
| Clients | 20 |
| Alpha (Dirichlet) | 0.1 |
| Local epochs per round | 5 |
| Total rounds | 200 |
| Drift round | 100 |
| Batch size | 64 |
| Learning rate | 0.01 (Adaptive-FedAvg adjusts dynamically) |
| Momentum | 0.9 |
| Weight decay | 1e-5 |

---

## 6. Output files reference

After a successful run, these files are produced:

| File | Content |
|---|---|
| `experiment_log_YYYYMMDD_HHMMSS.txt` | Full stdout/stderr from the entire run |
| `results_FedAvg.csv` | Per-round global accuracy for FedAvg |
| `results_Flash.csv` | Per-round global accuracy for Flash |
| `results_CDAFedAvg.csv` | Per-round global accuracy + flagged count for CDA-FedAvg |
| `results_OurMethod.csv` | Per-round global accuracy for OurMethod |
| `results_OurMethod_flags.csv` | Per-round per-client flag details (which clients flagged which layers) |

---

## 7. Quick-start command sequence

For the next person continuing this work:

```bash
# On GCP L4 VM (after SSH connection)
nvidia-smi   # verify GPU
pip install scipy --break-system-packages

# Upload all_experiments_optimized.py via scp from local
# scp -i "<keypath>" all_experiments_optimized.py suraf@<vm-ip>:~/

# Run in background, watch progress
nohup python3 -u all_experiments_optimized.py > nohup.out 2>&1 &
tail -f experiment_log_*.txt

# When finished, stop the VM to stop billing
sudo shutdown -h now
```
