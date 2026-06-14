"""
all_experiments_optimized.py — refactored 2026-05-26

Federated learning under distributed concept drift (CIFAR-10).

Active methods (selectable via --methods):
  1  FedAvg
  2  Flash             (Panchal et al. 2023)
  3  AdaptiveFedAvg    (Canonji 2021, ported from FedCCFA)
  4  OurMethod         (drift-triggered selective layer adaptation)

Legacy methods kept in this file but NOT exposed via CLI:
  CDA-FedAvg (Casado et al. 2022) — documented negative finding.
  To run it, edit __main__ to call run_cda_fedavg() explicitly.

Pipeline (active methods):
  - GPU-resident client + test tensors (one-time copy at startup)
  - GPU-side per-sample augmentation: RandomCrop(32, pad=4) + HFlip(p=0.5)
  - num_workers=0 (workers added overhead on 4-vCPU L4, see profile reports)
  - Batch size 64, drop_last=True (matches FedCCFA semantics)

Logging:
  - Full stdout/stderr tee'd to experiment_log_<ts>.txt
  - Per-method CSV is appended + flushed on every round, so a crash never
    loses more than the in-flight round.
  - OurMethod additionally produces results_OurMethod_flags.csv (per-round
    per-client flag tracking).

Usage:
  python3 all_experiments_optimized.py                  # all 4 active
  python3 all_experiments_optimized.py --methods 1      # FedAvg only
  python3 all_experiments_optimized.py --methods 1 2 4  # subset
  python3 all_experiments_optimized.py --methods all    # explicit all
"""

import argparse
import copy
import csv
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy.stats import beta as beta_dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Dataset


# ============================================================
# LOGGING — capture all stdout/stderr to a timestamped log file
# ============================================================

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

print("=" * 60)
print(f"Experiment log: {LOG_FILE}")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)


# ============================================================
# CONFIG
# ============================================================

NUM_CLIENTS    = 20
NUM_ROUNDS     = 200
LOCAL_EPOCHS   = 5
BATCH_SIZE     = 64
LR             = 0.01
MOMENTUM       = 0.9
WEIGHT_DECAY   = 1e-5
ALPHA_DIR      = 0.1
SEED           = 0
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Drift schedule.
# DEFAULT: single sudden-drift event at round 100 (original FedCCFA setup).
# Pass --recurrent on the CLI to override to the recurrent schedule [100, 150].
#
# Each entry in DRIFT_SCHEDULE is the round at which a drift event fires. The
# per-event swap mapping in DRIFT_EVENTS must have an entry for each scheduled
# round (and DRIFT_EVENTS keeps both mappings so the recurrent flag can flip
# the schedule without changing the swap table).
#
# CONSTRAINT (recurrent only): per group, the label pair at event k+1 must be
# DISJOINT from the pair at event k for that group. Otherwise applying
# _swap_labels_gpu twice on the same pair would TOGGLE (involution) and undo
# the drift. The default DRIFT_EVENTS table satisfies this constraint.
DRIFT_SCHEDULE_SINGLE    = [100]              # default
DRIFT_SCHEDULE_RECURRENT = [100, 150]         # used when --recurrent is passed
DRIFT_SCHEDULE = list(DRIFT_SCHEDULE_SINGLE)  # active schedule; overridable in __main__
DRIFT_EVENTS = [
    # Event 0 (round 100) — original sudden-drift swaps from FedCCFA
    {'A': (1, 2), 'B': (3, 4), 'C': (5, 6)},
    # Event 1 (round 150) — rotated, each group's pair is disjoint from its event-0 pair.
    # Only consumed when DRIFT_SCHEDULE has >= 2 entries (--recurrent enabled).
    {'A': (3, 4), 'B': (5, 6), 'C': (7, 8)},
]
assert len(DRIFT_EVENTS) >= len(DRIFT_SCHEDULE_RECURRENT), \
    "Need a swap mapping in DRIFT_EVENTS for every entry in DRIFT_SCHEDULE_RECURRENT"

# Sudden drift groups — FedCCFA layout, client_id % 10 rule.
# Group membership does NOT change across events; only the label pair that
# gets swapped for that group at each event changes.
DRIFT_GROUP_A  = [i for i in range(NUM_CLIENTS) if i % 10 < 3]
DRIFT_GROUP_B  = [i for i in range(NUM_CLIENTS) if 3 <= i % 10 < 6]
DRIFT_GROUP_C  = [i for i in range(NUM_CLIENTS) if i % 10 >= 6]
DRIFT_GROUPS   = {'A': DRIFT_GROUP_A, 'B': DRIFT_GROUP_B, 'C': DRIFT_GROUP_C}

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print(f"Device: {DEVICE}")
print(f"CPU cores detected: {os.cpu_count()}")
print(f"Config: {NUM_CLIENTS} clients | alpha={ALPHA_DIR} | "
      f"{LOCAL_EPOCHS} epochs | {NUM_ROUNDS} rounds | drift schedule {DRIFT_SCHEDULE}")
print(f"Pipeline: GPU-resident dataset + GPU-side augmentation")
print(f"Batch size: {BATCH_SIZE} | drop_last=True | num_workers=0")


# ============================================================
# DATA — load, partition, build GPU-resident tensors
# ============================================================

print("Downloading CIFAR-10...")
raw_train = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True)
raw_test  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)


def partition_dataset(dataset, n, alpha, seed):
    """FedCCFA-style Dirichlet partition with a small per-class warm-up split."""
    labels = np.array(dataset.targets)
    nc     = len(set(labels))
    cidx   = [[] for _ in range(n)]
    for k in range(nc):
        ci = np.where(labels == k)[0]
        es = np.array_split(ci[:n*5], n)
        cidx = [c + e.tolist() for c, e in zip(cidx, es)]
        rem  = ci[n*5:]
        random.seed(seed + k); np.random.seed(seed + k)
        props  = np.random.dirichlet(np.repeat(alpha, n))
        splits = (np.cumsum(props/props.sum()) * len(rem)).astype(int)[:-1]
        cidx   = [c + ch.tolist() for c, ch in
                  zip(cidx, np.split(rem, splits))]
    for c in cidx: random.shuffle(c)
    return cidx


print("Partitioning...")
train_idx = partition_dataset(raw_train, NUM_CLIENTS, ALPHA_DIR, SEED)
sizes = [len(train_idx[i]) for i in range(NUM_CLIENTS)]
print(f"Samples/client: min={min(sizes)} max={max(sizes)} mean={int(np.mean(sizes))}")


# --------- GPU-resident tensors (built once at startup) ---------
print("Pre-loading GPU-resident tensors...")
MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=DEVICE).view(1, 3, 1, 1)
STD  = torch.tensor([0.2023, 0.1994, 0.2010], device=DEVICE).view(1, 3, 1, 1)


def _to_gpu_tensors(base, indices):
    """uint8 HWC -> normalized fp32 NCHW on GPU; labels -> long on GPU."""
    arr = base.data[indices].astype(np.float32) / 255.0
    x   = torch.from_numpy(arr).permute(0, 3, 1, 2).to(DEVICE)
    x   = (x - MEAN) / STD
    y   = torch.tensor(np.array(base.targets)[indices],
                       dtype=torch.long, device=DEVICE)
    return x, y


# x tensors never mutate. y tensors are the *clean* (un-drifted) labels;
# each method clones them so drift mutations don't leak across methods.
GPU_CLIENT_X       = {}
GPU_CLIENT_Y_CLEAN = {}
for _cid in range(NUM_CLIENTS):
    _x, _y = _to_gpu_tensors(raw_train, train_idx[_cid])
    GPU_CLIENT_X[_cid]       = _x
    GPU_CLIENT_Y_CLEAN[_cid] = _y

TEST_X, TEST_Y = _to_gpu_tensors(raw_test, list(range(len(raw_test))))


# ============================================================
# PER-CLIENT GENERALIZED ACCURACY (FedCCFA protocol — faithful port)
# ============================================================
#
# FedCCFA's metric (utils/gen_dataset.py:149, utils/drift.py:23-42,
# entities/base.py:64-67, methods/FedAvg.py:50-51):
#   1. 4 copies of the FULL test set, all initially undrifted.
#   2. At drift, copies [1]/[2]/[3] get cohort A/B/C's label swaps; copy [0]
#      stays undrifted. Each client is assigned a global_test_id in {0,1,2,3}.
#      Initial value is 0 (matches FedCCFA's `Client(..., 0)` init).
#   3. server.send_params(clients) before last_round_evaluate, so per-client
#      eval uses the GLOBAL aggregated model.
#   4. Per-client metric = mean across clients of acc(global_model,
#      copy[client.global_test_id]).
#
# Memory-efficient port: ONE TEST_X (10000 images) shared by all four variants;
# FOUR TEST_Y variants (10000 longs each). TEST_Y_VARIANTS[0] is the undrifted
# reference; [1]/[2]/[3] are mutated in lockstep with apply_drift_event via
# apply_test_drift_event below.

TEST_Y_VARIANTS = {
    0: TEST_Y.clone(),
    1: TEST_Y.clone(),
    2: TEST_Y.clone(),
    3: TEST_Y.clone(),
}
CLIENT_GLOBAL_TEST_ID = [0] * NUM_CLIENTS

GROUP_TO_GID = {'A': 1, 'B': 2, 'C': 3}


def apply_test_drift_event(event_idx):
    """Mutate TEST_Y_VARIANTS[1..3] and CLIENT_GLOBAL_TEST_ID for the event.
    Called from apply_drift_event so the test-set swaps stay in lockstep with
    the per-client training-label swaps. TEST_Y_VARIANTS[0] is never mutated.
    """
    swaps = DRIFT_EVENTS[event_idx]
    for group_label, group_clients in DRIFT_GROUPS.items():
        a, b = swaps[group_label]
        gid  = GROUP_TO_GID[group_label]
        _swap_labels_gpu(TEST_Y_VARIANTS[gid], a, b)
        for cid in group_clients:
            CLIENT_GLOBAL_TEST_ID[cid] = gid


def reset_per_client_metric_state():
    """Restore pre-drift test-set state. Called at the start of each method
    run so cross-method drift mutations don't leak."""
    for gid in TEST_Y_VARIANTS:
        TEST_Y_VARIANTS[gid].copy_(TEST_Y)
    for cid in range(NUM_CLIENTS):
        CLIENT_GLOBAL_TEST_ID[cid] = 0


def evaluate_per_client_gen_acc(model):
    """FedCCFA per-client generalized accuracy.

    For each client, evaluate the GLOBAL model on the full test set with that
    client's cohort label swap, then average across clients. Clients in the
    same cohort share a global_test_id, so we evaluate once per active variant
    and weight by cohort size — numerically equivalent to a per-client loop.

    Pre-drift (every client at gid=0) is handled by a fast path that returns
    the single-variant accuracy directly. This makes the pre-drift identity
    `evaluate_per_client_gen_acc(m) == evaluate_gpu(m)` bit-exact (no
    sum(N copies of x)/N round-trip and its associated ULP rounding).
    """
    needed = set(CLIENT_GLOBAL_TEST_ID)
    acc_by_gid = {gid: evaluate_gpu(model, TEST_X, TEST_Y_VARIANTS[gid])
                  for gid in needed}
    if len(needed) == 1:
        return next(iter(acc_by_gid.values()))
    cohort_sizes = {gid: 0 for gid in needed}
    for cid in range(NUM_CLIENTS):
        cohort_sizes[CLIENT_GLOBAL_TEST_ID[cid]] += 1
    return sum(acc_by_gid[gid] * cohort_sizes[gid]
               for gid in needed) / NUM_CLIENTS

_resident_mb = sum(t.element_size() * t.numel() for t in
                   list(GPU_CLIENT_X.values()) +
                   list(GPU_CLIENT_Y_CLEAN.values()) +
                   [TEST_X, TEST_Y]) / (1024 * 1024)
print(f"GPU-resident dataset memory: {_resident_mb:.1f} MiB")


def fresh_client_y():
    """Per-method working copy of client labels (so drift doesn't leak)."""
    return {cid: GPU_CLIENT_Y_CLEAN[cid].clone() for cid in range(NUM_CLIENTS)}


CLIENT_WEIGHTS = [GPU_CLIENT_X[i].size(0) for i in range(NUM_CLIENTS)]


# ============================================================
# DRIFT — in-place mutation of per-method GPU label tensors
# ============================================================

def _swap_labels_gpu(y, a, b):
    """Swap label values a and b in-place. Mask-then-assign avoids interference."""
    mask_a = (y == a)
    mask_b = (y == b)
    y[mask_a] = b
    y[mask_b] = a


def apply_drift_event(client_y, event_idx):
    """
    Apply the swap mapping of DRIFT_EVENTS[event_idx] in-place on the per-method
    client_y dict (each entry is a per-client GPU label tensor).

    Each event introduces a NEW perturbation; per-group pairs at successive
    events are required to be disjoint so that _swap_labels_gpu (an involution)
    does not toggle previous swaps back to canonical.

    The canonical TEST_Y tensor (used by `evaluate_gpu` for global accuracy)
    is never mutated. The four TEST_Y_VARIANTS used by the FedCCFA per-client
    metric ARE mutated here via apply_test_drift_event, so the per-cohort test
    swaps stay in lockstep with the per-client training-label swaps.
    """
    swaps = DRIFT_EVENTS[event_idx]
    rnd   = DRIFT_SCHEDULE[event_idx]
    print(f"\n  *** Drift event {event_idx} at round {rnd} ***")
    for group_label, group_clients in DRIFT_GROUPS.items():
        a, b = swaps[group_label]
        print(f"  Group {group_label} {group_clients}: swap {a}<->{b}")
        for cid in group_clients:
            _swap_labels_gpu(client_y[cid], a, b)
    apply_test_drift_event(event_idx)


# ============================================================
# GPU-SIDE AUGMENTATION
# Mirrors torchvision.transforms.RandomCrop(32, padding=4, padding_mode='constant')
# + RandomHorizontalFlip(p=0.5). Per-sample crop offsets, per-sample flip.
#
# Padding subtlety: torchvision pads the raw uint8 image with 0 (true black)
# BEFORE normalization. After normalization the padded zeros become
# (0 - MEAN) / STD = -MEAN/STD per channel (~-2.4). To stay faithful, we pad
# the already-normalized tensor with -MEAN/STD per channel — NOT a constant 0,
# which would correspond to mid-gray (raw ~ MEAN). Verified to match torchvision
# pixel statistics to 6 decimal places in augment_parity_test.py.
# ============================================================

# Per-channel pad value: image of "raw 0" after normalization.
PAD_VALUE = (-MEAN / STD)  # shape [1, 3, 1, 1]


def gpu_augment(x, pad=4, crop_size=32):
    n = x.size(0)
    # Per-sample horizontal flip with p=0.5
    flip = torch.rand(n, device=x.device) < 0.5
    if flip.any():
        x = torch.where(flip[:, None, None, None], torch.flip(x, dims=[3]), x)
    # Build padded tensor with per-channel "black" fill (= raw 0 after normalization)
    _, c, h, w = x.shape
    ph, pw = h + 2 * pad, w + 2 * pad
    padded = PAD_VALUE.expand(n, c, ph, pw).contiguous()
    padded[:, :, pad:pad + h, pad:pad + w] = x
    x = padded
    _, c, h, _ = x.shape
    max_off = h - crop_size  # = 2*pad
    h_off   = torch.randint(0, max_off + 1, (n,), device=x.device)
    w_off   = torch.randint(0, max_off + 1, (n,), device=x.device)
    rows = h_off[:, None] + torch.arange(crop_size, device=x.device)[None, :]
    cols = w_off[:, None] + torch.arange(crop_size, device=x.device)[None, :]
    bi   = torch.arange(n, device=x.device)[:, None, None, None]
    ci   = torch.arange(c, device=x.device)[None, :, None, None]
    ri   = rows[:, None, :, None].expand(n, c, crop_size, crop_size)
    cj   = cols[:, None, None, :].expand(n, c, crop_size, crop_size)
    return x[bi, ci, ri, cj]


# ============================================================
# MODEL — FedCCFA CifarCNN
# ============================================================

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


def get_model():
    return CifarCNN().to(DEVICE)


print(f"Model: CifarCNN | Params: {sum(p.numel() for p in get_model().parameters()):,}")


# ============================================================
# SHARED UTILITIES (GPU pipeline)
# ============================================================

def evaluate_gpu(model, test_x=None, test_y=None, batch=512):
    if test_x is None: test_x = TEST_X
    if test_y is None: test_y = TEST_Y
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for s in range(0, test_x.size(0), batch):
            out = model(test_x[s:s+batch])
            correct += (out.argmax(1) == test_y[s:s+batch]).sum().item()
            total   += test_y[s:s+batch].size(0)
    return correct / total


def local_evaluate_gpu(model, x, y, batch=1024):
    """
    Accuracy of the GLOBAL model on a single client's local data
    (with whatever labels they currently hold — may be drifted).
    Measures whether the federated model serves this client well.
    """
    model.eval()
    correct = 0
    n = x.size(0)
    with torch.no_grad():
        for s in range(0, n, batch):
            out = model(x[s:s+batch])
            correct += (out.argmax(1) == y[s:s+batch]).sum().item()
    return correct / n if n > 0 else 0.0


def evaluate_all_clients(model, client_y):
    """Return [acc_0, acc_1, ..., acc_{N-1}] of the global model on each client's data."""
    return [local_evaluate_gpu(model, GPU_CLIENT_X[cid], client_y[cid])
            for cid in range(NUM_CLIENTS)]


# Column names for per-client local accuracy (used by all methods)
CLIENT_FIELDS = [f'local_c{cid:02d}' for cid in range(NUM_CLIENTS)]
# Column names for per-client HYBRID accuracy (OurMethod only — see run_our_method)
HYBRID_FIELDS = [f'hybrid_c{cid:02d}' for cid in range(NUM_CLIENTS)]


def build_local_row(rnd, global_acc, local_accs, extra=None):
    """Build a CSV row dict with round, global_acc, optional extras, and per-client local accs."""
    row = {'round': rnd, 'global_acc': global_acc}
    if extra:
        row.update(extra)
    for cid, a in enumerate(local_accs):
        row[f'local_c{cid:02d}'] = a
    return row


def local_train_gpu(model, x, y, epochs=LOCAL_EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    """
    SGD on one client's GPU-resident data with GPU augmentation.
    drop_last=True semantics (matches FedCCFA's DataLoader).
    """
    model.train()
    opt  = optim.SGD(model.parameters(), lr=lr,
                     momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    n   = x.size(0)
    end = (n // batch_size) * batch_size
    for _ in range(epochs):
        perm = torch.randperm(n, device=x.device)
        for s in range(0, end, batch_size):
            idx = perm[s:s+batch_size]
            xb  = gpu_augment(x[idx])
            opt.zero_grad()
            crit(model(xb), y[idx]).backward()
            opt.step()


def fedavg_aggregate(global_model, local_states, weights):
    gs    = global_model.state_dict()
    total = sum(weights)
    new   = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in gs.items()}
    for state, w in zip(local_states, weights):
        for key in gs:
            new[key] += (w / total) * state[key].float()
    global_model.load_state_dict(new)


OUT_DIR = ''   # Empty = current directory; overridden by --out-dir


def seeded_path(base_name):
    """Resolve the output path for a CSV.

    Rules:
      - If OUT_DIR is set (--out-dir), write into OUT_DIR with the bare
        base name (no _seed<N> suffix — the folder encodes the seed).
      - Otherwise (no --out-dir): seed=0 uses the bare base_name (back-compat),
        non-zero seeds get a `_seed<N>` filename suffix.
    """
    if OUT_DIR:
        os.makedirs(OUT_DIR, exist_ok=True)
        return os.path.join(OUT_DIR, base_name)
    if SEED == 0:
        return base_name
    if base_name.endswith('.csv'):
        return base_name[:-4] + f'_seed{SEED}.csv'
    return base_name + f'_seed{SEED}'


class LiveCSV:
    """Append-as-you-go CSV that flushes on every write."""
    def __init__(self, path, fieldnames):
        self.path       = path
        self.fieldnames = fieldnames
        self.f          = open(path, 'w', newline='', buffering=1)
        self.writer     = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.writer.writeheader()
        self.f.flush()
    def write(self, row):
        self.writer.writerow(row)
        self.f.flush()
    def close(self):
        try: self.f.close()
        except Exception: pass


# ============================================================
# METHOD 1 — FedAvg
# ============================================================

def run_fedavg():
    print("\n" + "="*60)
    print("METHOD 1: FedAvg")
    print("="*60)

    client_y = fresh_client_y()
    gm       = get_model()
    reset_per_client_metric_state()
    log      = []
    csv_out  = LiveCSV(seeded_path('results_FedAvg.csv'),
                       ['round', 'global_acc', 'per_client_gen_acc'] + CLIENT_FIELDS)

    try:
        for rnd in range(NUM_ROUNDS):
            if rnd in DRIFT_SCHEDULE:
                apply_drift_event(client_y, DRIFT_SCHEDULE.index(rnd))

            states, weights = [], []
            for cid in range(NUM_CLIENTS):
                lm = get_model()
                lm.load_state_dict({k: v.clone() for k, v in gm.state_dict().items()})
                local_train_gpu(lm, GPU_CLIENT_X[cid], client_y[cid])
                states.append(lm.state_dict())
                weights.append(CLIENT_WEIGHTS[cid])

            fedavg_aggregate(gm, states, weights)
            acc        = evaluate_gpu(gm)
            pc_acc     = evaluate_per_client_gen_acc(gm)
            local_accs = evaluate_all_clients(gm, client_y)
            row        = build_local_row(rnd, acc, local_accs,
                                         extra={'per_client_gen_acc': pc_acc})
            log.append(row)
            csv_out.write(row)

            if rnd % 10 == 0 or rnd in DRIFT_SCHEDULE:
                tag = "  <-- DRIFT" if rnd in DRIFT_SCHEDULE else ""
                mean_local = sum(local_accs) / len(local_accs)
                print(f"  Round {rnd:03d} | Global: {acc:.4f} | "
                      f"per-client: {pc_acc:.4f} | "
                      f"mean local: {mean_local:.4f}{tag}")
    finally:
        csv_out.close()
    print(f"  Results: {csv_out.path}")
    return log


# ============================================================
# METHOD 5 (control) — FedAvg + 1 local epoch
#
# Identical to plain FedAvg in every respect that touches the GLOBAL model:
#   - Same client local-training (LOCAL_EPOCHS epochs per round)
#   - Same fedavg_aggregate over the same client states with the same weights
#   - Same global_acc evaluation on canonical TEST_Y
# With the same seed, this method's global_acc trajectory is BIT-IDENTICAL
# to plain FedAvg. The ONLY divergence is the per-client metric:
#   plain FedAvg's per_client_gen_acc = acc(GLOBAL model, cohort-swapped test)
#   FedAvg+1's    per_client_gen_acc = acc(GLOBAL + 1 local epoch on client's
#                                          own data, cohort-swapped test)
# This isolates trivial last-step personalization as a confound on the
# per-client metric. Mirrors the FedAvg_baseline.py finding: +2.76pp over
# published FedAvg at Dir(0.5).
#
# Plain FedAvg's run_fedavg() is NOT modified. The two are independent
# entries in METHOD_REGISTRY (1 = FedAvg, 5 = FedAvg+1).
# ============================================================

def evaluate_per_client_gen_acc_finetuned(global_model, client_y, scratch_model):
    """Per-client generalized accuracy where each client's eval model
    = global_model + 1 local epoch on the client's own (possibly drifted) data.

    No cohort fast-path here: each client has a DIFFERENT fine-tuned model,
    so we cannot pre-cohort the forward passes — 20 fine-tune+eval cycles per
    round. The 1 local epoch uses the SAME lr/optimizer/batch size as plain
    FedAvg local training (local_train_gpu defaults: lr=LR, batch=BATCH_SIZE,
    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, gpu_augment).
    """
    accs = []
    g_state = global_model.state_dict()
    for cid in range(NUM_CLIENTS):
        scratch_model.load_state_dict({k: v.clone() for k, v in g_state.items()})
        local_train_gpu(scratch_model, GPU_CLIENT_X[cid], client_y[cid], epochs=1)
        gid = CLIENT_GLOBAL_TEST_ID[cid]
        accs.append(evaluate_gpu(scratch_model, TEST_X, TEST_Y_VARIANTS[gid]))
    return sum(accs) / len(accs)


def run_fedavg_plus1():
    print("\n" + "="*60)
    print("METHOD 5 (control): FedAvg + 1 local epoch")
    print("  global eval     -> global aggregated model, canonical TEST_Y")
    print("  per-client eval -> each client's (global + 1 local epoch) model")
    print("="*60)

    client_y         = fresh_client_y()
    gm               = get_model()
    finetune_scratch = get_model()
    reset_per_client_metric_state()
    log     = []
    csv_out = LiveCSV(seeded_path('results_FedAvgPlus1.csv'),
                      ['round', 'global_acc', 'per_client_gen_acc'] + CLIENT_FIELDS)

    try:
        for rnd in range(NUM_ROUNDS):
            if rnd in DRIFT_SCHEDULE:
                apply_drift_event(client_y, DRIFT_SCHEDULE.index(rnd))

            states, weights = [], []
            for cid in range(NUM_CLIENTS):
                lm = get_model()
                lm.load_state_dict({k: v.clone() for k, v in gm.state_dict().items()})
                local_train_gpu(lm, GPU_CLIENT_X[cid], client_y[cid])
                states.append(lm.state_dict())
                weights.append(CLIENT_WEIGHTS[cid])

            fedavg_aggregate(gm, states, weights)
            acc        = evaluate_gpu(gm)
            pc_acc     = evaluate_per_client_gen_acc_finetuned(gm, client_y,
                                                               finetune_scratch)
            local_accs = evaluate_all_clients(gm, client_y)
            row        = build_local_row(rnd, acc, local_accs,
                                         extra={'per_client_gen_acc': pc_acc})
            log.append(row)
            csv_out.write(row)

            if rnd % 10 == 0 or rnd in DRIFT_SCHEDULE:
                tag = "  <-- DRIFT" if rnd in DRIFT_SCHEDULE else ""
                mean_local = sum(local_accs) / len(local_accs)
                print(f"  Round {rnd:03d} | Global: {acc:.4f} | "
                      f"per-client(finetuned): {pc_acc:.4f} | "
                      f"mean local: {mean_local:.4f}{tag}")
    finally:
        csv_out.close()
    print(f"  Results: {csv_out.path}")
    return log


# ============================================================
# METHOD 2 — Flash (Panchal et al. 2023)
# ============================================================

def run_flash():
    print("\n" + "="*60)
    print("METHOD 2: Flash")
    print("="*60)

    # From Flash.yaml
    SERVER_LR      = 0.01
    LOSS_DECREMENT = 0.004
    BETA1          = 0.9
    BETA2          = 0.99
    TAU_FLASH      = 0.001

    client_y = fresh_client_y()
    gm       = get_model()
    reset_per_client_metric_state()
    log      = []
    csv_out  = LiveCSV(seeded_path('results_Flash.csv'),
                       ['round', 'global_acc', 'per_client_gen_acc'] + CLIENT_FIELDS)
    crit     = nn.CrossEntropyLoss()

    first_mom  = 0
    second_mom = TAU_FLASH ** 2
    prev_2mom  = 0
    delta_mom  = 0
    beta3      = 0
    prev_val_loss = {cid: -1 for cid in range(NUM_CLIENTS)}

    try:
        for rnd in range(NUM_ROUNDS):
            if rnd in DRIFT_SCHEDULE:
                apply_drift_event(client_y, DRIFT_SCHEDULE.index(rnd))

            updates, weights = [], []
            for cid in range(NUM_CLIENTS):
                lm = get_model()
                lm.load_state_dict({k: v.clone() for k, v in gm.state_dict().items()})
                lm.train()
                init_params = parameters_to_vector(lm.parameters()).detach()
                opt = optim.SGD(lm.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
                x_cid = GPU_CLIENT_X[cid]
                y_cid = client_y[cid]
                n     = x_cid.size(0)
                end   = (n // BATCH_SIZE) * BATCH_SIZE

                for epoch in range(LOCAL_EPOCHS):
                    perm = torch.randperm(n, device=DEVICE)
                    for s in range(0, end, BATCH_SIZE):
                        idx = perm[s:s+BATCH_SIZE]
                        xb  = gpu_augment(x_cid[idx])
                        opt.zero_grad()
                        crit(lm(xb), y_cid[idx]).backward()
                        opt.step()

                    # Per-epoch val-loss check (mirrors original — augmented,
                    # un-shuffled sweep across the client's data)
                    lm.eval()
                    val_loss = 0.0
                    vb       = 0
                    with torch.no_grad():
                        for s in range(0, n, 256):
                            xb_v = gpu_augment(x_cid[s:s+256])
                            val_loss += crit(lm(xb_v), y_cid[s:s+256]).item()
                            vb       += 1
                    val_loss /= max(vb, 1)

                    if prev_val_loss[cid] != -1:
                        delta = prev_val_loss[cid] - val_loss
                        if 0 < delta < LOSS_DECREMENT / (epoch + 1):
                            break
                    prev_val_loss[cid] = val_loss
                    lm.train()

                cur_params = parameters_to_vector(lm.parameters()).detach()
                updates.append((cur_params - init_params).cpu().numpy())
                weights.append(CLIENT_WEIGHTS[cid])

            # Flash server aggregation
            total = sum(weights)
            agg   = sum(u * (w/total) for u, w in zip(updates, weights))

            first_mom  = BETA1 * first_mom + (1 - BETA1) * agg
            prev_2mom  = second_mom
            second_mom = BETA2 * second_mom + (1 - BETA2) * np.square(agg)
            beta3      = (np.abs(prev_2mom) /
                          (np.abs(np.square(agg) - second_mom) +
                           np.abs(prev_2mom) + 1e-10))
            delta_mom  = (beta3 * delta_mom +
                          (1 - beta3) * (np.square(agg) - second_mom))
            agg_update = (SERVER_LR * first_mom /
                          (np.sqrt(second_mom) - delta_mom + TAU_FLASH))

            cur_global = parameters_to_vector(gm.parameters()).detach()
            new_global = cur_global + torch.tensor(
                agg_update, dtype=torch.float32).to(DEVICE)
            vector_to_parameters(new_global, gm.parameters())

            acc        = evaluate_gpu(gm)
            pc_acc     = evaluate_per_client_gen_acc(gm)
            local_accs = evaluate_all_clients(gm, client_y)
            row        = build_local_row(rnd, acc, local_accs,
                                         extra={'per_client_gen_acc': pc_acc})
            log.append(row)
            csv_out.write(row)

            if rnd % 10 == 0 or rnd in DRIFT_SCHEDULE:
                tag = "  <-- DRIFT" if rnd in DRIFT_SCHEDULE else ""
                mean_local = sum(local_accs) / len(local_accs)
                print(f"  Round {rnd:03d} | Global: {acc:.4f} | "
                      f"per-client: {pc_acc:.4f} | "
                      f"mean local: {mean_local:.4f}{tag}")
    finally:
        csv_out.close()
    print(f"  Results: {csv_out.path}")
    return log


# ============================================================
# METHOD 3 — AdaptiveFedAvg (ported from FedCCFA)
# Server tracks EMA of global-param mean/variance to compute a
# bias-corrected variance ratio that scales the client LR each round.
# Math mirrors FedCCFA AdaptiveFedAvgServer.cal_adaptive_lr exactly.
# ============================================================

ADAPTIVE_BETA1 = 0.7    # mean EMA
ADAPTIVE_BETA2 = 0.3    # variance EMA
ADAPTIVE_BETA3 = 0.7    # ratio EMA

# AdaptiveFedAvg's client_init_lr (the LR the scheduler scales). Defaults to
# the harness's LR. Overridable via --adaptive-init-lr for FedDrift-style LR
# search on AdaptiveFedAvg (Jothimurugesan et al. 2023 § 5.3 — Adaptive-FedAvg
# uses its own internal LR scheduler so it gets a separate LR sweep).
ADAPTIVE_INIT_LR = LR


# ============================================================
# Saile et al. 2024 (FLTA) — client-side loss-EMA dynamic LR.
# Faithful port of src/algorithm/learningrate_estimator.py:LearningrateEstimatorLoss
# (LR multiplier) and src/server/fedavgserver.py StepLR (server decay).
#
# Default hyperparameters are Saile's CIFAR-10 config (lr=0.2, lr_decay=0.99/round,
# b1=0.7, b2=0.3, b3=0.7). Their initial LR is 20x our LR=0.01 - we treat Saile
# as its own method with its own LR (same protocol used for AdaptiveFedAvg fix),
# and SAILE_INIT_LR is overridable via --saile-init-lr for the FedDrift-style
# LR sweep over Saile's recommended LRs.
# ============================================================
SAILE_B1            = 0.7
SAILE_B2            = 0.3
SAILE_B3            = 0.7
SAILE_INIT_LR       = 0.2           # Saile CIFAR-10 default; override via --saile-init-lr
SAILE_LR_DECAY      = 0.99          # multiplicative server-side per-round decay
SAILE_PROBE_BUDGET  = 50            # Saile's `while batch_count <= 50` in update_estimator


class SaileEstimator:
    """Faithful port of LearningrateEstimatorLoss (Saile et al. 2024).

    One instance per client. estimate(loss, current_round, base_lr) returns
    that client's LR for this round = min(initial_lr, base_lr * R_hat) where
    R_hat is the third (bias-corrected) EMA of the variance ratio of the loss.

    Mirrors src/algorithm/learningrate_estimator.py:77-128. Including the V=0
    edge case at line 109-113 (paper Algorithm 2 lines 5-6).
    """
    def __init__(self, initial_lr, b1, b2, b3):
        self.initial_lr = initial_lr
        self.b1, self.b2, self.b3 = b1, b2, b3
        self.loss_ema = 0.0
        self.prev_loss_ema = 0.0
        self.prev_loss_ema_na = 0.0
        self.variance_ema = 0.0
        self.prev_variance_ema = 0.0
        self.prev_variance_ema_na = 0.0
        self.variance_ratio_ema = 0.0
        self.prev_variance_ratio_ema_na = 0.0

    def estimate(self, loss, current_round, base_lr):
        # EMA on loss mean + bias correction
        self.loss_ema = self.b1 * self.prev_loss_ema_na + (1 - self.b1) * loss
        self.prev_loss_ema_na = self.loss_ema
        self.loss_ema = self.loss_ema / (1 - pow(self.b1, current_round))

        # EMA on loss variance + bias correction
        self.variance_ema = (self.b2 * self.prev_variance_ema_na
                             + (1 - self.b2) * (loss - self.prev_loss_ema) ** 2)
        self.prev_variance_ema_na = self.variance_ema
        self.variance_ema = self.variance_ema / (1 - pow(self.b2, current_round))

        # V=0 edge case (paper Algorithm 2 lines 5-6, Saile code lines 109-113)
        if self.prev_variance_ema == 0:
            ratio = 1.0
        else:
            ratio = self.variance_ema / self.prev_variance_ema

        # EMA on variance ratio + bias correction
        self.variance_ratio_ema = (self.b3 * self.prev_variance_ratio_ema_na
                                   + (1 - self.b3) * ratio)
        self.prev_variance_ratio_ema_na = self.variance_ratio_ema
        self.variance_ratio_ema = self.variance_ratio_ema / (1 - pow(self.b3, current_round))

        # Snapshot for next call's ratio computation
        self.prev_loss_ema = self.loss_ema
        self.prev_variance_ema = self.variance_ema

        return float(min(self.initial_lr, base_lr * self.variance_ratio_ema))


def _saile_probe_loss(model, x, y,
                      probe_budget=SAILE_PROBE_BUDGET, batch_size=BATCH_SIZE):
    """Mean cross-entropy of the GLOBAL model on a random subset of (x, y).

    Mirrors src/client/fedavgclient.py:79-101 update_estimator(): a
    `while batch_count <= probe_budget` loop sampling random batches and
    averaging the per-batch loss. No grad. With our BATCH_SIZE=64 and the
    default probe_budget=50, one batch fires (~64 samples) — equivalent to
    Saile's behavior at B=50 (two batches ~100 samples).
    """
    n = x.size(0)
    if n == 0:
        return 0.0
    crit = nn.CrossEntropyLoss()
    model.eval()
    losses = []
    batch_count = 0
    with torch.no_grad():
        while batch_count <= probe_budget:
            idx = torch.randperm(n, device=x.device)[:batch_size]
            losses.append(crit(model(x[idx]), y[idx]).item())
            batch_count += batch_size
    return float(np.mean(losses))


def run_saile():
    """METHOD 6: Saile et al. 2024 (FLTA), client-side loss-EMA dynamic LR.

    Each round (faithful to Saile's flow):
      1. Server broadcasts the global model gm.
      2. Each client computes a loss probe on gm (~SAILE_PROBE_BUDGET samples,
         no grad).
      3. Each client's SaileEstimator updates 3 EMAs (loss, loss-variance,
         variance-ratio) + bias correction + V=0 edge case, returning a
         per-client LR = min(SAILE_INIT_LR, base_lr * R_hat).
      4. Each client trains locally at its OWN LR for LOCAL_EPOCHS.
      5. FedAvg aggregate.
      6. Server applies stepwise decay: base_lr <- base_lr * SAILE_LR_DECAY.

    Hyperparameters are Saile's CIFAR-10 defaults by default. The init LR is
    overridable via --saile-init-lr for FedDrift-style LR sweeps.
    """
    print("\n" + "="*60)
    print(f"METHOD 6: Saile (per-client loss-EMA dynamic LR)")
    print(f"  initial_lr={SAILE_INIT_LR} | lr_decay={SAILE_LR_DECAY}/round | "
          f"b1={SAILE_B1} b2={SAILE_B2} b3={SAILE_B3}")
    print(f"  Decay brings base_lr from {SAILE_INIT_LR} to "
          f"{SAILE_INIT_LR * SAILE_LR_DECAY**100:.4f} at round 100, "
          f"{SAILE_INIT_LR * SAILE_LR_DECAY**199:.4f} at round 199.")
    print("="*60)

    client_y   = fresh_client_y()
    gm         = get_model()
    reset_per_client_metric_state()
    estimators = [SaileEstimator(SAILE_INIT_LR, SAILE_B1, SAILE_B2, SAILE_B3)
                  for _ in range(NUM_CLIENTS)]
    base_lr    = SAILE_INIT_LR
    log        = []
    csv_out    = LiveCSV(seeded_path('results_Saile.csv'),
                         ['round', 'global_acc', 'per_client_gen_acc',
                          'base_lr', 'mean_client_lr',
                          'min_client_lr', 'max_client_lr'] + CLIENT_FIELDS)

    try:
        for rnd in range(NUM_ROUNDS):
            if rnd in DRIFT_SCHEDULE:
                apply_drift_event(client_y, DRIFT_SCHEDULE.index(rnd))

            # Loss probe + per-client LR on the BROADCAST global model gm.
            # current_round is 1-indexed (Saile's framework rounds start at 1,
            # avoids divide-by-zero in 1 - b^0).
            client_lrs = []
            for cid in range(NUM_CLIENTS):
                loss = _saile_probe_loss(gm, GPU_CLIENT_X[cid], client_y[cid])
                lr_c = estimators[cid].estimate(loss, rnd + 1, base_lr)
                client_lrs.append(lr_c)

            # Local training at each client's individual LR.
            states, weights = [], []
            for cid in range(NUM_CLIENTS):
                lm = get_model()
                lm.load_state_dict({k: v.clone() for k, v in gm.state_dict().items()})
                local_train_gpu(lm, GPU_CLIENT_X[cid], client_y[cid], lr=client_lrs[cid])
                states.append(lm.state_dict())
                weights.append(CLIENT_WEIGHTS[cid])

            fedavg_aggregate(gm, states, weights)

            # Server-side stepwise decay (after aggregation, before next round).
            base_lr = base_lr * SAILE_LR_DECAY

            acc        = evaluate_gpu(gm)
            pc_acc     = evaluate_per_client_gen_acc(gm)
            local_accs = evaluate_all_clients(gm, client_y)
            row = build_local_row(rnd, acc, local_accs, extra={
                'per_client_gen_acc': pc_acc,
                'base_lr':            base_lr,
                'mean_client_lr':     float(np.mean(client_lrs)),
                'min_client_lr':      float(np.min(client_lrs)),
                'max_client_lr':      float(np.max(client_lrs)),
            })
            log.append(row)
            csv_out.write(row)

            if rnd % 10 == 0 or rnd in DRIFT_SCHEDULE:
                tag = "  <-- DRIFT" if rnd in DRIFT_SCHEDULE else ""
                print(f"  Round {rnd:03d} | Global: {acc:.4f} | "
                      f"per-client: {pc_acc:.4f} | base_lr={base_lr:.5f} | "
                      f"lr min/mean/max="
                      f"{min(client_lrs):.5f}/{np.mean(client_lrs):.5f}/"
                      f"{max(client_lrs):.5f}{tag}")
    finally:
        csv_out.close()
    print(f"  Results: {csv_out.path}")
    return log


def run_adaptive_fedavg():
    print("\n" + "="*60)
    print("METHOD 3: Adaptive-FedAvg (Canonji 2021, ported from FedCCFA)")
    print(f"  beta1={ADAPTIVE_BETA1} beta2={ADAPTIVE_BETA2} beta3={ADAPTIVE_BETA3}")
    print("="*60)

    client_y = fresh_client_y()
    gm       = get_model()
    reset_per_client_metric_state()
    log      = []
    csv_out  = LiveCSV(seeded_path('results_AdaptiveFedAvg.csv'),
                       ['round', 'global_acc', 'per_client_gen_acc', 'client_lr'] + CLIENT_FIELDS)

    # Server-side adaptive-LR state
    prev_mean          = 0.0
    prev_mean_norm     = 0.0
    prev_variance      = 0.0
    prev_variance_norm = 0.0
    prev_ratio         = 0.0
    client_init_lr     = ADAPTIVE_INIT_LR
    current_lr         = ADAPTIVE_INIT_LR

    try:
        for rnd in range(NUM_ROUNDS):
            if rnd in DRIFT_SCHEDULE:
                apply_drift_event(client_y, DRIFT_SCHEDULE.index(rnd))

            states, weights = [], []
            for cid in range(NUM_CLIENTS):
                lm = get_model()
                lm.load_state_dict({k: v.clone() for k, v in gm.state_dict().items()})
                local_train_gpu(lm, GPU_CLIENT_X[cid], client_y[cid], lr=current_lr)
                states.append(lm.state_dict())
                weights.append(CLIENT_WEIGHTS[cid])

            fedavg_aggregate(gm, states, weights)

            # Adaptive-LR update on the new global parameter vector
            cur_round  = rnd + 1   # 1-indexed, prevents division by zero
            cur_params = parameters_to_vector(gm.parameters()).detach().cpu().numpy()

            mean      = ADAPTIVE_BETA1 * prev_mean + (1 - ADAPTIVE_BETA1) * cur_params
            mean_norm = mean / (1 - pow(ADAPTIVE_BETA1, cur_round))

            variance      = (ADAPTIVE_BETA2 * prev_variance +
                             (1 - ADAPTIVE_BETA2) *
                             np.mean((cur_params - prev_mean_norm) ** 2))
            variance_norm = variance / (1 - pow(ADAPTIVE_BETA2, cur_round))

            if cur_round == 1:
                ratio = ADAPTIVE_BETA3 * prev_ratio + (1 - ADAPTIVE_BETA3)
            else:
                ratio = (ADAPTIVE_BETA3 * prev_ratio +
                         (1 - ADAPTIVE_BETA3) *
                         (variance_norm / prev_variance_norm))
            ratio_norm = ratio / (1 - pow(ADAPTIVE_BETA3, cur_round))

            prev_mean          = mean
            prev_mean_norm     = mean_norm
            prev_variance      = variance
            prev_variance_norm = variance_norm
            prev_ratio         = ratio

            # NOTE: FedCCFA's AdaptiveFedAvgServer.cal_adaptive_lr divides the
            # final LR by `cur_round`, producing a 1/t decay that drives the LR
            # near zero before drift hits (e.g. 1e-4 at round 100 with base
            # 1e-2). That divisor contradicts the algorithm's stated purpose —
            # to RAISE the LR when update-variance spikes at drift — and is not
            # present in Saile et al. 2024's independent implementation of the
            # same algorithm. We remove the /cur_round divisor here. Bias
            # correction on the three EMAs (1 - beta^t) is retained.
            current_lr = float(min(client_init_lr,
                                   client_init_lr * ratio_norm))

            acc        = evaluate_gpu(gm)
            pc_acc     = evaluate_per_client_gen_acc(gm)
            local_accs = evaluate_all_clients(gm, client_y)
            row        = build_local_row(rnd, acc, local_accs,
                                         extra={'per_client_gen_acc': pc_acc,
                                                'client_lr': current_lr})
            log.append(row)
            csv_out.write(row)

            if rnd % 10 == 0 or rnd in DRIFT_SCHEDULE:
                tag = "  <-- DRIFT" if rnd in DRIFT_SCHEDULE else ""
                mean_local = sum(local_accs) / len(local_accs)
                print(f"  Round {rnd:03d} | Global: {acc:.4f} | "
                      f"per-client: {pc_acc:.4f} | "
                      f"mean local: {mean_local:.4f} | "
                      f"lr={current_lr:.5f}{tag}")
    finally:
        csv_out.close()
    print(f"  Results: {csv_out.path}")
    return log


# ============================================================
# METHOD 4 — OurMethod
# Per-layer drift-triggered selective layer adaptation.
# ============================================================

LAYER_GROUPS = {
    'layer1': 'hidden_layers.0',
    'layer2': 'hidden_layers.3',
    'layer3': 'hidden_layers.6',
    'layer4': 'hidden_layers.10',
}
DRIFT_LAYERS  = ['layer3', 'layer4']    # only these are eligible to be flagged
STABLE_LAYERS = ['layer1', 'layer2']    # always uploaded
CLASSIFIER    = 'fc'                    # always overwritten by global
EMA_ALPHA     = 0.3
TAU_OUR       = 1.4
WARMUP        = 10


def get_layer_vec(state, prefix):
    parts = [v.float().flatten() for k, v in state.items()
             if k.startswith(prefix)]
    return torch.cat(parts) if parts else torch.zeros(1)


def weight_divergence(local_state, global_state):
    divs = {}
    for name, prefix in LAYER_GROUPS.items():
        lv = get_layer_vec(local_state, prefix)
        gv = get_layer_vec(global_state, prefix)
        divs[name] = ((lv - gv).norm() / gv.norm()).item() if gv.norm() > 0 else 0.0
    return divs


class OurClient:
    def __init__(self, cid):
        self.cid         = cid
        self.model       = get_model()
        self.ema         = None
        self.prev_flags  = {l: False for l in LAYER_GROUPS}
        self.local_state = None
        self.upload_mask = {l: True for l in LAYER_GROUPS}
        self.rounds_seen = 0

    def selective_sync(self, global_state):
        state = {k: v.clone() for k, v in global_state.items()}
        if self.local_state is not None:
            for name, prefix in LAYER_GROUPS.items():
                if self.prev_flags.get(name, False):
                    for key in state:
                        if key.startswith(prefix):
                            state[key] = self.local_state[key].clone()
        for key in state:
            if key.startswith(CLASSIFIER):
                state[key] = global_state[key].clone()
        self.model.load_state_dict(state)

    def train(self, global_state, x, y):
        self.rounds_seen += 1
        warmup = self.rounds_seen <= WARMUP

        self.selective_sync(global_state)
        local_train_gpu(self.model, x, y)

        self.local_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        divs = weight_divergence(self.local_state, global_state)

        if self.ema is None:
            self.ema = {l: divs[l] for l in LAYER_GROUPS}

        flags = {l: False for l in LAYER_GROUPS}
        for name in LAYER_GROUPS:
            d_l   = divs[name]
            ema_l = self.ema[name]
            ratio = d_l / ema_l if ema_l > 0 else 1.0
            if warmup:
                self.ema[name] = d_l
            else:
                self.ema[name] = EMA_ALPHA * d_l + (1 - EMA_ALPHA) * ema_l
                if name in DRIFT_LAYERS and ratio > TAU_OUR:
                    flags[name] = True

        self.prev_flags  = flags
        self.upload_mask = {l: not flags[l] for l in LAYER_GROUPS}

    def get_upload_state(self):
        full = self.model.state_dict()
        out  = {}
        for name, prefix in LAYER_GROUPS.items():
            if self.upload_mask[name]:
                for k, v in full.items():
                    if k.startswith(prefix):
                        out[k] = v.clone()
        for k, v in full.items():
            if k.startswith(CLASSIFIER):
                out[k] = v.clone()
        return out


def compute_hybrid_state(client, global_state):
    """
    Build the state_dict the client would use AFTER selective_sync next round —
    i.e. global model for unflagged layers + classifier, client's local_state
    for flagged layers. Pure function, no side effects on client.model.
    """
    state = {k: v.clone() for k, v in global_state.items()}
    if client.local_state is not None:
        for name, prefix in LAYER_GROUPS.items():
            if client.prev_flags.get(name, False):
                for key in state:
                    if key.startswith(prefix):
                        state[key] = client.local_state[key].clone()
    # Classifier always taken from global (the design pins shared label semantics)
    for key in state:
        if key.startswith(CLASSIFIER):
            state[key] = global_state[key].clone()
    return state


def evaluate_all_clients_hybrid(global_model, clients, client_y, temp_model):
    """
    Per-client accuracy of the HYBRID model (what the client actually uses
    next round) on the client's own data. For OurMethod the hybrid differs
    from the global model exactly when c.prev_flags has any True entries.
    """
    global_state = global_model.state_dict()
    accs = []
    for c in clients:
        h_state = compute_hybrid_state(c, global_state)
        temp_model.load_state_dict(h_state)
        accs.append(local_evaluate_gpu(temp_model,
                                       GPU_CLIENT_X[c.cid],
                                       client_y[c.cid]))
    return accs


def per_layer_fedavg_our(global_model, clients, weights):
    gs  = global_model.state_dict()
    new = {k: v.clone() for k, v in gs.items()}

    for name, prefix in LAYER_GROUPS.items():
        uploaders = [(c, w) for c, w in zip(clients, weights)
                     if c.upload_mask.get(name, True)]
        if not uploaders:
            continue
        total = sum(w for _, w in uploaders)
        for key in gs:
            if key.startswith(prefix):
                new[key] = torch.zeros_like(gs[key], dtype=torch.float32)
                for c, w in uploaders:
                    us = c.get_upload_state()
                    if key in us:
                        new[key] += (w/total) * us[key].float()

    total_all = sum(weights)
    for key in gs:
        if key.startswith(CLASSIFIER):
            new[key] = torch.zeros_like(gs[key], dtype=torch.float32)
            for c, w in zip(clients, weights):
                us = c.get_upload_state()
                if key in us:
                    new[key] += (w/total_all) * us[key].float()

    global_model.load_state_dict(new)


def run_our_method():
    print("\n" + "="*60)
    print("METHOD 4: OurMethod — Drift-Triggered Selective Layer Adaptation")
    print(f"  EMA alpha={EMA_ALPHA} | TAU={TAU_OUR} | Warmup={WARMUP}")
    print("="*60)

    client_y    = fresh_client_y()
    gm          = get_model()
    reset_per_client_metric_state()
    clients     = [OurClient(i) for i in range(NUM_CLIENTS)]
    hybrid_temp = get_model()   # reused buffer for hybrid-model evaluation
    log         = []

    csv_out  = LiveCSV(seeded_path('results_OurMethod.csv'),
                       ['round', 'global_acc', 'per_client_gen_acc']
                       + CLIENT_FIELDS + HYBRID_FIELDS)
    flag_csv = LiveCSV(seeded_path('results_OurMethod_flags.csv'), [
        'round', 'flagged_count',
        'flagged_layer3_count', 'flagged_layer4_count',
        'flagged_client_ids', 'flagged_layer3_ids',
        'flagged_layer4_ids', 'global_acc'])

    try:
        for rnd in range(NUM_ROUNDS):
            if rnd in DRIFT_SCHEDULE:
                apply_drift_event(client_y, DRIFT_SCHEDULE.index(rnd))

            gs = {k: v.clone() for k, v in gm.state_dict().items()}
            for c in clients:
                c.train(gs, GPU_CLIENT_X[c.cid], client_y[c.cid])

            weights = [CLIENT_WEIGHTS[c.cid] for c in clients]
            per_layer_fedavg_our(gm, clients, weights)

            acc        = evaluate_gpu(gm)
            pc_acc     = evaluate_per_client_gen_acc(gm)
            local_accs = evaluate_all_clients(gm, client_y)
            # HYBRID: per-client accuracy of the model each client actually uses
            # next round (global for unflagged layers + classifier, local for flagged).
            hybrid_accs = evaluate_all_clients_hybrid(gm, clients, client_y, hybrid_temp)

            row = build_local_row(rnd, acc, local_accs,
                                  extra={'per_client_gen_acc': pc_acc})
            for cid, h in enumerate(hybrid_accs):
                row[f'hybrid_c{cid:02d}'] = h
            log.append(row)
            csv_out.write(row)

            flagged_layer3 = [c.cid for c in clients if c.prev_flags.get('layer3', False)]
            flagged_layer4 = [c.cid for c in clients if c.prev_flags.get('layer4', False)]
            flagged_any    = sorted(set(flagged_layer3) | set(flagged_layer4))
            flag_csv.write({
                'round': rnd,
                'flagged_count': len(flagged_any),
                'flagged_layer3_count': len(flagged_layer3),
                'flagged_layer4_count': len(flagged_layer4),
                'flagged_client_ids': ';'.join(map(str, flagged_any)),
                'flagged_layer3_ids': ';'.join(map(str, flagged_layer3)),
                'flagged_layer4_ids': ';'.join(map(str, flagged_layer4)),
                'global_acc': acc,
            })

            if (rnd % 10 == 0 or rnd in DRIFT_SCHEDULE or
                    any(d <= rnd <= d + 20 for d in DRIFT_SCHEDULE)):
                tag = "  <-- DRIFT" if rnd in DRIFT_SCHEDULE else ""
                # Hybrid lift on flagged clients (the metric the thesis cares about)
                if flagged_any:
                    fh = [hybrid_accs[c] for c in flagged_any]
                    fl = [local_accs[c]  for c in flagged_any]
                    delta = sum(fh)/len(fh) - sum(fl)/len(fl)
                    hyb_str = f" | hyb-lift(flagged): {delta:+.4f}"
                else:
                    hyb_str = ""
                print(f"  Round {rnd:03d} | Global: {acc:.4f} | "
                      f"per-client: {pc_acc:.4f} | "
                      f"Flagged: {len(flagged_any)}/{NUM_CLIENTS} "
                      f"(L3:{len(flagged_layer3)}, L4:{len(flagged_layer4)}) "
                      f"IDs: {flagged_any}{hyb_str}{tag}")
    finally:
        csv_out.close()
        flag_csv.close()
    print(f"  Results: {csv_out.path}")
    print(f"  Per-client flags: {flag_csv.path}")
    return log


# ============================================================
# LEGACY — CDA-FedAvg (Casado et al. 2022)
# Retained in the file for the record but NOT exposed via the CLI.
# Uses the older CPU DataLoader pipeline; porting the LTM-rehearsal
# machinery to GPU-resident is non-trivial and out of scope.
# To run: edit __main__ and call run_cda_fedavg() explicitly.
# ============================================================

class ClientDataset(Dataset):
    def __init__(self, base, indices, transform=None):
        self.data      = base.data[indices]
        self.targets   = list(np.array(base.targets)[indices])
        self.transform = transform
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        img, label = self.data[idx], int(self.targets[idx])
        if self.transform: img = self.transform(img)
        return img, label


_legacy_train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
_legacy_test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


def build_client_sets_legacy():
    return {i: ClientDataset(raw_train, train_idx[i], _legacy_train_tf)
            for i in range(NUM_CLIENTS)}


def drift_dataset_legacy(dataset, a, b):
    targets = np.array(dataset.targets)
    ia = np.where(targets == a)[0]
    ib = np.where(targets == b)[0]
    for i in ia: dataset.targets[i] = -1
    for i in ib: dataset.targets[i] = a
    for i in np.where(np.array(dataset.targets) == -1)[0]:
        dataset.targets[i] = b


def apply_drift_legacy(client_sets, event_idx):
    """LEGACY (CPU) counterpart of apply_drift_event for CDA-FedAvg's older path.
    Same DRIFT_EVENTS mapping; mutates dataset.targets in place."""
    swaps = DRIFT_EVENTS[event_idx]
    rnd   = DRIFT_SCHEDULE[event_idx]
    print(f"\n  *** Drift event {event_idx} at round {rnd} (legacy) ***")
    for group_label, group_clients in DRIFT_GROUPS.items():
        a, b = swaps[group_label]
        for cid in group_clients:
            drift_dataset_legacy(client_sets[cid], a, b)


def evaluate_legacy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total


LAMBDA_CDA = 0.05
DELTA_CDA  = 100
N_MAX_CDA  = 1000
H_CDA      = 2.995732
LTM_SIZE   = 2000


def fit_beta_mom_cda(x):
    mu  = np.clip(x.mean(), 1e-6, 1.0 - 1e-6)
    var = x.var()
    max_var = mu * (1.0 - mu) - 1e-8
    if max_var <= 0 or var <= 0:
        return None, None
    var    = min(var, max_var)
    common = mu * (1.0 - mu) / var - 1.0
    return max(mu * common, 1e-6), max((1.0 - mu) * common, 1e-6)


def detect_drift_cda(Q_confidences):
    N = len(Q_confidences)
    if N < 2 * DELTA_CDA:
        return False
    Q = np.clip(np.array(Q_confidences), 1e-6, 1.0 - 1e-6)
    for k in range(DELTA_CDA, N - DELTA_CDA):
        Qa, Qb = Q[:k], Q[k:]
        ma, mb = Qa.mean(), Qb.mean()
        if ma > (1.0 - LAMBDA_CDA) * mb:
            continue
        alpha_a, beta_a = fit_beta_mom_cda(Qa)
        alpha_b, beta_b = fit_beta_mom_cda(Qb)
        if alpha_a is None or alpha_b is None:
            continue
        try:
            log_pa = beta_dist.logpdf(Q, alpha_a, beta_a)
            log_pb = beta_dist.logpdf(Q, alpha_b, beta_b)
            log_pa = np.where(np.isfinite(log_pa), log_pa, 0.0)
            log_pb = np.where(np.isfinite(log_pb), log_pb, 0.0)
            sk = (log_pa - log_pb).sum()
        except Exception:
            continue
        if sk > H_CDA:
            return True
    return False


class CDAClient:
    """LEGACY — CDA-FedAvg client (Casado et al. 2022)."""
    def __init__(self, cid, dataset):
        self.cid         = cid
        self.dataset     = dataset
        self.model       = get_model()
        self.Q           = []
        self.Q_data      = []
        self.LTM         = []
        self.n_concepts  = 0
        self.initialized = False

    def set_params(self, global_state):
        self.model.load_state_dict({k: v.clone() for k, v in global_state.items()})

    def _ltm_loader(self):
        class LTMDataset(Dataset):
            def __init__(s, data): s.data = data
            def __len__(s): return len(s.data)
            def __getitem__(s, i):
                img, label = s.data[i]
                return img, torch.tensor(label, dtype=torch.long)
        return DataLoader(LTMDataset(self.LTM), batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=False)

    def train(self):
        crit   = nn.CrossEntropyLoss()
        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE,
                            shuffle=True, drop_last=True)
        if not self.initialized:
            init_data = []
            for imgs, labels in loader:
                for i in range(imgs.size(0)):
                    init_data.append((imgs[i].cpu(), int(labels[i])))
            if len(init_data) > LTM_SIZE:
                init_data = random.sample(init_data, LTM_SIZE)
            self.LTM.extend(init_data)
            self.n_concepts = 1
            self.initialized = True
        train_loader = self._ltm_loader() if self.LTM else loader
        self.model.train()
        opt = optim.SGD(self.model.parameters(), lr=LR,
                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        for _ in range(LOCAL_EPOCHS):
            for imgs, labels in train_loader:
                imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
                opt.zero_grad()
                crit(self.model(imgs.float()), labels).backward()
                opt.step()
        current_data = []
        drift_found  = False
        self.model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(DEVICE)
                probs = torch.softmax(self.model(imgs.float()), dim=1)
                confs = probs.max(1).values.cpu().numpy()
                for i, qi in enumerate(confs):
                    label   = int(labels[i])
                    img_cpu = imgs[i].cpu()
                    current_data.append((img_cpu, label))
                    if len(self.Q) >= N_MAX_CDA:
                        self.Q.pop(0); self.Q_data.pop(0)
                    self.Q.append(float(qi))
                    self.Q_data.append((img_cpu, label))
                    if random.random() < math.exp(-2.0 * float(qi)):
                        if detect_drift_cda(self.Q):
                            drift_found = True; break
                if drift_found: break
        if drift_found:
            sampled = current_data
            if len(sampled) > LTM_SIZE:
                sampled = random.sample(sampled, LTM_SIZE)
            self.LTM.extend(sampled)
            if len(self.LTM) > LTM_SIZE * 5:
                self.LTM = self.LTM[-(LTM_SIZE * 5):]
            self.n_concepts += 1
            self.Q.clear(); self.Q_data.clear()


def run_cda_fedavg():
    """LEGACY — not wired into the CLI. To run, call from __main__ explicitly."""
    print("\n" + "="*60)
    print("METHOD (LEGACY): CDA-FedAvg")
    print("="*60)
    client_sets   = build_client_sets_legacy()
    test_ds       = ClientDataset(raw_test, list(range(len(raw_test))), _legacy_test_tf)
    global_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    gm      = get_model()
    clients = [CDAClient(i, client_sets[i]) for i in range(NUM_CLIENTS)]
    log     = []
    csv_out = LiveCSV('results_CDAFedAvg.csv', ['round', 'global_acc'])
    try:
        for rnd in range(NUM_ROUNDS):
            if rnd in DRIFT_SCHEDULE:
                apply_drift_legacy(client_sets, DRIFT_SCHEDULE.index(rnd))
                for c in clients:
                    c.dataset = client_sets[c.cid]
            gs = {k: v.clone() for k, v in gm.state_dict().items()}
            for c in clients:
                c.set_params(gs); c.train()
            states  = [c.model.state_dict() for c in clients]
            weights = [len(client_sets[c.cid]) for c in clients]
            fedavg_aggregate(gm, states, weights)
            for c in clients:
                c.set_params(gm.state_dict())
            acc = evaluate_legacy(gm, global_loader)
            log.append({'round': rnd, 'global_acc': acc})
            csv_out.write({'round': rnd, 'global_acc': acc})
            if rnd % 10 == 0 or rnd in DRIFT_SCHEDULE:
                tag = "  <-- DRIFT" if rnd in DRIFT_SCHEDULE else ""
                print(f"  Round {rnd:03d} | Global: {acc:.4f}{tag}")
    finally:
        csv_out.close()
    return log


# ============================================================
# MAIN — CLI + dispatch
# ============================================================

METHOD_REGISTRY = {
    1: ('FedAvg',          run_fedavg),
    2: ('Flash',           run_flash),
    3: ('AdaptiveFedAvg',  run_adaptive_fedavg),
    4: ('OurMethod',       run_our_method),
    5: ('FedAvgPlus1',     run_fedavg_plus1),   # control: FedAvg + 1 local epoch for per-client eval only
    6: ('Saile',           run_saile),          # Saile 2024 (FLTA): per-client loss-EMA dynamic LR
}


def parse_args():
    p = argparse.ArgumentParser(
        description="FL drift experiments on CIFAR-10. GPU-resident pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Method IDs:\n"
                "  1  FedAvg\n"
                "  2  Flash\n"
                "  3  AdaptiveFedAvg\n"
                "  4  OurMethod\n"
                "  5  FedAvgPlus1 (control: FedAvg + 1 local epoch for per-client eval only)\n"
                "  6  Saile (2024 FLTA: per-client loss-EMA dynamic LR)\n"
                "\nExamples:\n"
                "  python3 all_experiments_optimized.py\n"
                "  python3 all_experiments_optimized.py --methods 1\n"
                "  python3 all_experiments_optimized.py --methods 1 2 4\n"
                "  python3 all_experiments_optimized.py --methods all\n"))
    p.add_argument('--methods', nargs='+', default=['all'],
                   help="Method IDs (1..6) or 'all' (default; methods 1..6 only).")
    p.add_argument('--rounds', type=int, default=None,
                   help=f"Override total rounds (default {NUM_ROUNDS}). "
                        f"For smoke tests, use a small value like 10.")
    p.add_argument('--seed', type=int, default=None,
                   help=f"Random seed (default {SEED}). Drives partition + "
                        f"init. For seed != 0, output CSVs are named "
                        f"results_X_seed<N>.csv so multi-seed runs coexist.")
    p.add_argument('--out-dir', type=str, default=None,
                   help="Directory to write result CSVs into. If set, the "
                        "_seed<N> filename suffix is dropped (the folder "
                        "encodes the seed). Use this to land results "
                        "straight into a runs/<tag>/seed<N>/ folder.")
    p.add_argument('--recurrent', action='store_true',
                   help=f"Enable recurrent drift. With this flag, drift fires "
                        f"at rounds {DRIFT_SCHEDULE_RECURRENT}; without it, "
                        f"the default schedule {DRIFT_SCHEDULE_SINGLE} (single "
                        f"sudden-drift event) is used. Per-event swap mappings "
                        f"come from DRIFT_EVENTS.")
    p.add_argument('--alternating-drift', action='store_true',
                   help="Frequent ALTERNATING drift: schedule [40, 80, 120, 160] "
                        "with the SAME cohort swap (A:1<->2, B:3<->4, C:5<->6) "
                        "applied at every event. Because _swap_labels_gpu is an "
                        "involution, applying the same swap twice toggles labels "
                        "back to canonical. So the concept oscillates: "
                        "canonical -> swapped -> canonical -> swapped -> canonical. "
                        "Used for recurrent-gain analysis (forgetting + accuracy "
                        "across recurring concepts). Overrides --recurrent.")
    # Ablation knobs (OurMethod only). Affect only `TAU_OUR` and `DRIFT_LAYERS`,
    # which together gate the single line at OurClient.train:
    #     if name in DRIFT_LAYERS and ratio > TAU_OUR: flags[name] = True
    # Nothing else in OurMethod's machinery (selective_sync, get_upload_state,
    # per_layer_fedavg_our, classifier handling, EMA update) is changed.
    p.add_argument('--ablation-tau', type=float, default=None,
                   help=f"Override OurMethod's TAU_OUR (default {TAU_OUR}). "
                        f"Pass `inf` to disable detection entirely (no layer "
                        f"ever flags; OurMethod reduces to FedAvg-equivalent "
                        f"aggregation through the existing selective-sync path).")
    p.add_argument('--ablation-all-layers', action='store_true',
                   help=f"Make all 4 hidden layers (L1, L2, L3, L4) eligible "
                        f"for flagging instead of only L3+L4. The classifier "
                        f"continues to be always-global, unchanged.")
    p.add_argument('--adaptive-init-lr', type=float, default=None,
                   help=f"Override the initial LR for AdaptiveFedAvg (method 3) "
                        f"only. Other methods continue to use LR={LR}. Used for "
                        f"the FedDrift-style LR sweep on AdaptiveFedAvg (its "
                        f"internal scheduler needs its own LR selected).")
    p.add_argument('--saile-init-lr', type=float, default=None,
                   help=f"Override the initial LR for Saile (method 6) only. "
                        f"Default is Saile's CIFAR-10 LR={SAILE_INIT_LR}. Used "
                        f"for the FedDrift-style LR sweep (Saile has its own "
                        f"internal LR scheduler so it needs its own LR selected).")
    return p.parse_args()


def resolve_methods(arg_list):
    if any(str(a).lower() == 'all' for a in arg_list):
        return [1, 2, 3, 4]
    out = []
    for a in arg_list:
        try:
            n = int(a)
        except ValueError:
            raise SystemExit(f"Unknown --methods value: {a!r}. "
                             f"Use 1..6 or 'all'.")
        if n not in METHOD_REGISTRY:
            raise SystemExit(f"Unknown method id: {n}. Valid: 1..6")
        if n not in out:
            out.append(n)
    return out


def _per_group_acc_at_round(row, group_clients, col_prefix='local_c'):
    """Mean of per-client local-acc columns over clients in the group, for one round."""
    vals = []
    for cid in group_clients:
        key = f'{col_prefix}{cid:02d}'
        if key in row:
            vals.append(float(row[key]))
    return float(np.mean(vals)) if vals else None


def summarize_method(name, log):
    """
    Headline metrics around the first drift event (back-compat with downstream
    analyses) plus per-event and per-group dip recap for multi-event schedules.
    """
    accs = [r['global_acc'] for r in log]
    first_drift = DRIFT_SCHEDULE[0]
    if len(accs) < first_drift + 11:
        return  # short run, skip

    # Headline metrics — anchored at the FIRST drift event (unchanged interpretation)
    pre   = float(np.mean(accs[max(0, first_drift-11):first_drift]))
    dip   = pre - min(accs[first_drift:first_drift+10])
    rec   = next((i - first_drift for i in range(first_drift, len(accs))
                  if accs[i] >= pre - 0.02), None)
    stable = float(np.mean(accs[-10:]))
    print(f"\n{name}:")
    print(f"  Pre-drift acc:     {pre:.4f}")
    print(f"  Accuracy dip:      {dip:.4f}  (at first drift event, round {first_drift})")
    print(f"  Recovery rounds:   {rec if rec is not None else 'Not recovered'}")
    print(f"  Post-drift stable: {stable:.4f}")

    # Per-event GLOBAL-accuracy dip recap (one row per drift event)
    if len(DRIFT_SCHEDULE) > 1:
        print(f"  Per-event global dips:")
        print(f"    {'event':<6}{'round':>7}{'pre':>10}{'min(+10)':>12}{'dip':>10}")
        for k, d in enumerate(DRIFT_SCHEDULE):
            if len(accs) < d + 10:
                continue
            pre_d = float(np.mean(accs[max(0, d-11):d]))
            min_d = min(accs[d:d+10])
            dip_d = pre_d - min_d
            print(f"    {k:<6}{d:>7}{pre_d:>10.4f}{min_d:>12.4f}{dip_d:>10.4f}")

    # Per-event PER-GROUP local-accuracy dip recap. Only meaningful if the log
    # contains per-client columns (which all 4 active methods produce). Uses
    # local_cXX (federation-view of each client). For OurMethod this is the
    # raw shock to the system; hybrid_cXX would measure adaptation residual.
    has_local_cols = bool(log) and any(k.startswith('local_c') for k in log[0])
    if has_local_cols and len(DRIFT_SCHEDULE) >= 1:
        print(f"  Per-event per-group local-acc dips (using local_cXX):")
        header = f"    {'event':<6}{'round':>7}"
        for gl in DRIFT_GROUPS:
            header += f"{'pre('+gl+')':>11}{'min('+gl+')':>11}{'dip('+gl+')':>11}"
        print(header)
        for k, d in enumerate(DRIFT_SCHEDULE):
            if len(log) < d + 10:
                continue
            line = f"    {k:<6}{d:>7}"
            for gl, gc in DRIFT_GROUPS.items():
                pre_vals = [_per_group_acc_at_round(log[r], gc)
                            for r in range(max(0, d-11), d)]
                pre_vals = [v for v in pre_vals if v is not None]
                post_vals = [_per_group_acc_at_round(log[r], gc)
                             for r in range(d, d+10)]
                post_vals = [v for v in post_vals if v is not None]
                if not pre_vals or not post_vals:
                    line += f"{'-':>11}{'-':>11}{'-':>11}"
                    continue
                pre_g = float(np.mean(pre_vals))
                min_g = min(post_vals)
                dip_g = pre_g - min_g
                line += f"{pre_g:>11.4f}{min_g:>11.4f}{dip_g:>11.4f}"
            print(line)


if __name__ == '__main__':
    args        = parse_args()
    method_ids  = resolve_methods(args.methods)
    method_names = [METHOD_REGISTRY[m][0] for m in method_ids]

    if args.rounds is not None:
        if args.rounds < 1:
            raise SystemExit("--rounds must be >= 1")
        NUM_ROUNDS = args.rounds
        print(f"\n[--rounds override] NUM_ROUNDS={NUM_ROUNDS}")

    if args.recurrent:
        DRIFT_SCHEDULE[:] = DRIFT_SCHEDULE_RECURRENT
        print(f"[--recurrent] DRIFT_SCHEDULE = {DRIFT_SCHEDULE} (multi-event drift)")

    if args.alternating_drift:
        # Frequent alternating drift: same cohort swap applied at every event.
        # _swap_labels_gpu is an involution -> two consecutive applications cancel,
        # producing canonical -> swapped -> canonical -> swapped -> canonical.
        # Overrides --recurrent if both were passed.
        DRIFT_SCHEDULE[:] = [40, 80, 120, 160]
        DRIFT_EVENTS.clear()
        for _ in range(len(DRIFT_SCHEDULE)):
            DRIFT_EVENTS.append({'A': (1, 2), 'B': (3, 4), 'C': (5, 6)})
        print(f"[--alternating-drift] DRIFT_SCHEDULE = {DRIFT_SCHEDULE}")
        print(f"[--alternating-drift] DRIFT_EVENTS = {len(DRIFT_EVENTS)} x "
              f"{DRIFT_EVENTS[0]} (involution -> concept oscillates "
              f"canonical/swapped/canonical/swapped/canonical)")

    if args.ablation_tau is not None:
        TAU_OUR = args.ablation_tau
        print(f"[--ablation-tau] TAU_OUR = {TAU_OUR}"
              + (" (detection disabled — no layer will ever flag)"
                 if TAU_OUR == float('inf') else ""))
    if args.ablation_all_layers:
        DRIFT_LAYERS = list(LAYER_GROUPS.keys())
        print(f"[--ablation-all-layers] DRIFT_LAYERS = {DRIFT_LAYERS}")

    if args.adaptive_init_lr is not None:
        ADAPTIVE_INIT_LR = args.adaptive_init_lr
        print(f"[--adaptive-init-lr] ADAPTIVE_INIT_LR = {ADAPTIVE_INIT_LR}")

    if args.saile_init_lr is not None:
        SAILE_INIT_LR = args.saile_init_lr
        print(f"[--saile-init-lr] SAILE_INIT_LR = {SAILE_INIT_LR}")

    if args.out_dir is not None:
        OUT_DIR = args.out_dir
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"[--out-dir] CSVs will be written to: {OUT_DIR}/")

    if args.seed is not None and args.seed != SEED:
        print(f"\n[--seed override] re-seeding to {args.seed} and re-partitioning...")
        SEED = args.seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        # Re-partition with the new seed
        train_idx = partition_dataset(raw_train, NUM_CLIENTS, ALPHA_DIR, SEED)
        # Rebuild GPU-resident tensors (x is also re-built since partition changed)
        GPU_CLIENT_X.clear()
        GPU_CLIENT_Y_CLEAN.clear()
        for _cid in range(NUM_CLIENTS):
            _x, _y = _to_gpu_tensors(raw_train, train_idx[_cid])
            GPU_CLIENT_X[_cid]       = _x
            GPU_CLIENT_Y_CLEAN[_cid] = _y
        # CLIENT_WEIGHTS is module-level; mutate in place
        for _i in range(NUM_CLIENTS):
            CLIENT_WEIGHTS[_i] = GPU_CLIENT_X[_i].size(0)
        sizes = [GPU_CLIENT_X[i].size(0) for i in range(NUM_CLIENTS)]
        print(f"  Re-partitioned: samples/client min={min(sizes)} "
              f"max={max(sizes)} mean={int(np.mean(sizes))}")

    print("\n" + "="*60)
    print(f"RUNNING METHODS: {' -> '.join(method_names)}")
    print(f"Rounds per method: {NUM_ROUNDS}")
    print("="*60)

    logs = {}
    for mid in method_ids:
        name, fn = METHOD_REGISTRY[mid]
        try:
            logs[name] = fn()
        except KeyboardInterrupt:
            print(f"\n!! Interrupted during {name}. "
                  f"results_{name}.csv on disk has all completed rounds.")
            raise

    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for name in method_names:
        if name in logs:
            summarize_method(name, logs[name])

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Full output log: {LOG_FILE}")
    print("CSV files saved per method (live):")
    for mid in method_ids:
        print(f"  - results_{METHOD_REGISTRY[mid][0]}.csv")
    if 4 in method_ids:
        print("  - results_OurMethod_flags.csv (per-client per-round flag data)")
