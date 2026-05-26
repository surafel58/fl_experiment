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
DRIFT_ROUND    = 100
SEED           = 0
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sudden drift groups — FedCCFA layout, client_id % 10 rule
DRIFT_GROUP_A  = [i for i in range(NUM_CLIENTS) if i % 10 < 3]    # swap 1<->2
DRIFT_GROUP_B  = [i for i in range(NUM_CLIENTS) if 3 <= i % 10 < 6] # swap 3<->4
DRIFT_GROUP_C  = [i for i in range(NUM_CLIENTS) if i % 10 >= 6]   # swap 5<->6

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print(f"Device: {DEVICE}")
print(f"CPU cores detected: {os.cpu_count()}")
print(f"Config: {NUM_CLIENTS} clients | alpha={ALPHA_DIR} | "
      f"{LOCAL_EPOCHS} epochs | {NUM_ROUNDS} rounds | drift at {DRIFT_ROUND}")
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


def apply_drift_gpu(client_y):
    """Apply FedCCFA sudden_drift label swaps to the per-method y dict."""
    print(f"\n  *** Sudden drift at round {DRIFT_ROUND} ***")
    print(f"  Group A {DRIFT_GROUP_A}: swap 1<->2")
    print(f"  Group B {DRIFT_GROUP_B}: swap 3<->4")
    print(f"  Group C {DRIFT_GROUP_C}: swap 5<->6")
    for cid in DRIFT_GROUP_A: _swap_labels_gpu(client_y[cid], 1, 2)
    for cid in DRIFT_GROUP_B: _swap_labels_gpu(client_y[cid], 3, 4)
    for cid in DRIFT_GROUP_C: _swap_labels_gpu(client_y[cid], 5, 6)


# ============================================================
# GPU-SIDE AUGMENTATION
# Mirrors torchvision.transforms.RandomCrop(32, padding=4, padding_mode='constant')
# + RandomHorizontalFlip(p=0.5). Per-sample crop offsets, per-sample flip decision.
# ============================================================

def gpu_augment(x, pad=4, crop_size=32):
    n = x.size(0)
    # Per-sample horizontal flip with p=0.5
    flip = torch.rand(n, device=x.device) < 0.5
    if flip.any():
        x = torch.where(flip[:, None, None, None], torch.flip(x, dims=[3]), x)
    # Zero-pad then per-sample random crop
    x = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
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
    log      = []
    csv_out  = LiveCSV('results_FedAvg.csv', ['round', 'global_acc'])

    try:
        for rnd in range(NUM_ROUNDS):
            if rnd == DRIFT_ROUND:
                apply_drift_gpu(client_y)

            states, weights = [], []
            for cid in range(NUM_CLIENTS):
                lm = get_model()
                lm.load_state_dict({k: v.clone() for k, v in gm.state_dict().items()})
                local_train_gpu(lm, GPU_CLIENT_X[cid], client_y[cid])
                states.append(lm.state_dict())
                weights.append(CLIENT_WEIGHTS[cid])

            fedavg_aggregate(gm, states, weights)
            acc = evaluate_gpu(gm)
            row = {'round': rnd, 'global_acc': acc}
            log.append(row)
            csv_out.write(row)

            if rnd % 10 == 0 or rnd == DRIFT_ROUND:
                tag = "  <-- DRIFT" if rnd == DRIFT_ROUND else ""
                print(f"  Round {rnd:03d} | Global: {acc:.4f}{tag}")
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
    log      = []
    csv_out  = LiveCSV('results_Flash.csv', ['round', 'global_acc'])
    crit     = nn.CrossEntropyLoss()

    first_mom  = 0
    second_mom = TAU_FLASH ** 2
    prev_2mom  = 0
    delta_mom  = 0
    beta3      = 0
    prev_val_loss = {cid: -1 for cid in range(NUM_CLIENTS)}

    try:
        for rnd in range(NUM_ROUNDS):
            if rnd == DRIFT_ROUND:
                apply_drift_gpu(client_y)

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

            acc = evaluate_gpu(gm)
            row = {'round': rnd, 'global_acc': acc}
            log.append(row)
            csv_out.write(row)

            if rnd % 10 == 0 or rnd == DRIFT_ROUND:
                tag = "  <-- DRIFT" if rnd == DRIFT_ROUND else ""
                print(f"  Round {rnd:03d} | Global: {acc:.4f}{tag}")
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


def run_adaptive_fedavg():
    print("\n" + "="*60)
    print("METHOD 3: Adaptive-FedAvg (Canonji 2021, ported from FedCCFA)")
    print(f"  beta1={ADAPTIVE_BETA1} beta2={ADAPTIVE_BETA2} beta3={ADAPTIVE_BETA3}")
    print("="*60)

    client_y = fresh_client_y()
    gm       = get_model()
    log      = []
    csv_out  = LiveCSV('results_AdaptiveFedAvg.csv',
                       ['round', 'global_acc', 'client_lr'])

    # Server-side adaptive-LR state
    prev_mean          = 0.0
    prev_mean_norm     = 0.0
    prev_variance      = 0.0
    prev_variance_norm = 0.0
    prev_ratio         = 0.0
    client_init_lr     = LR
    current_lr         = LR

    try:
        for rnd in range(NUM_ROUNDS):
            if rnd == DRIFT_ROUND:
                apply_drift_gpu(client_y)

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

            current_lr = float(min(client_init_lr,
                                   client_init_lr * ratio_norm / cur_round))

            acc = evaluate_gpu(gm)
            row = {'round': rnd, 'global_acc': acc, 'client_lr': current_lr}
            log.append(row)
            csv_out.write(row)

            if rnd % 10 == 0 or rnd == DRIFT_ROUND:
                tag = "  <-- DRIFT" if rnd == DRIFT_ROUND else ""
                print(f"  Round {rnd:03d} | Global: {acc:.4f} | "
                      f"client_lr={current_lr:.5f}{tag}")
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

    client_y = fresh_client_y()
    gm       = get_model()
    clients  = [OurClient(i) for i in range(NUM_CLIENTS)]
    log      = []

    csv_out  = LiveCSV('results_OurMethod.csv', ['round', 'global_acc'])
    flag_csv = LiveCSV('results_OurMethod_flags.csv', [
        'round', 'flagged_count',
        'flagged_layer3_count', 'flagged_layer4_count',
        'flagged_client_ids', 'flagged_layer3_ids',
        'flagged_layer4_ids', 'global_acc'])

    try:
        for rnd in range(NUM_ROUNDS):
            if rnd == DRIFT_ROUND:
                apply_drift_gpu(client_y)

            gs = {k: v.clone() for k, v in gm.state_dict().items()}
            for c in clients:
                c.train(gs, GPU_CLIENT_X[c.cid], client_y[c.cid])

            weights = [CLIENT_WEIGHTS[c.cid] for c in clients]
            per_layer_fedavg_our(gm, clients, weights)

            acc = evaluate_gpu(gm)
            log.append({'round': rnd, 'global_acc': acc})
            csv_out.write({'round': rnd, 'global_acc': acc})

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

            if (rnd % 10 == 0 or rnd == DRIFT_ROUND or
                    DRIFT_ROUND <= rnd <= DRIFT_ROUND + 20):
                tag = "  <-- DRIFT" if rnd == DRIFT_ROUND else ""
                print(f"  Round {rnd:03d} | Global: {acc:.4f} | "
                      f"Flagged: {len(flagged_any)}/{NUM_CLIENTS} "
                      f"(L3:{len(flagged_layer3)}, L4:{len(flagged_layer4)}) "
                      f"IDs: {flagged_any}{tag}")
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


def apply_drift_legacy(client_sets):
    print(f"\n  *** Sudden drift at round {DRIFT_ROUND} ***")
    for cid in DRIFT_GROUP_A: drift_dataset_legacy(client_sets[cid], 1, 2)
    for cid in DRIFT_GROUP_B: drift_dataset_legacy(client_sets[cid], 3, 4)
    for cid in DRIFT_GROUP_C: drift_dataset_legacy(client_sets[cid], 5, 6)


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
            if rnd == DRIFT_ROUND:
                apply_drift_legacy(client_sets)
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
            if rnd % 10 == 0 or rnd == DRIFT_ROUND:
                tag = "  <-- DRIFT" if rnd == DRIFT_ROUND else ""
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
                "\nExamples:\n"
                "  python3 all_experiments_optimized.py\n"
                "  python3 all_experiments_optimized.py --methods 1\n"
                "  python3 all_experiments_optimized.py --methods 1 2 4\n"
                "  python3 all_experiments_optimized.py --methods all\n"))
    p.add_argument('--methods', nargs='+', default=['all'],
                   help="Method IDs (1..4) or 'all' (default).")
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
                             f"Use 1..4 or 'all'.")
        if n not in METHOD_REGISTRY:
            raise SystemExit(f"Unknown method id: {n}. Valid: 1..4")
        if n not in out:
            out.append(n)
    return out


def summarize_method(name, log):
    accs = [r['global_acc'] for r in log]
    if len(accs) < DRIFT_ROUND + 11:
        return  # short run, skip
    pre   = float(np.mean(accs[max(0, DRIFT_ROUND-11):DRIFT_ROUND]))
    dip   = pre - min(accs[DRIFT_ROUND:DRIFT_ROUND+10])
    rec   = next((i - DRIFT_ROUND for i in range(DRIFT_ROUND, len(accs))
                  if accs[i] >= pre - 0.02), None)
    stable = float(np.mean(accs[-10:]))
    print(f"\n{name}:")
    print(f"  Pre-drift acc:     {pre:.4f}")
    print(f"  Accuracy dip:      {dip:.4f}")
    print(f"  Recovery rounds:   {rec if rec is not None else 'Not recovered'}")
    print(f"  Post-drift stable: {stable:.4f}")


if __name__ == '__main__':
    args        = parse_args()
    method_ids  = resolve_methods(args.methods)
    method_names = [METHOD_REGISTRY[m][0] for m in method_ids]

    print("\n" + "="*60)
    print(f"RUNNING METHODS: {' -> '.join(method_names)}")
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
