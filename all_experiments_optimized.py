# ============================================================
# All-in-one FL Experiment — runs all 4 methods sequentially
# Methods: FedAvg, Flash, CDA-FedAvg, Our Method
# Dataset: CIFAR-10, 20 clients, alpha=0.1, sudden drift round 100
# Results saved to CSV files in current directory
#
# Run: python all_experiments.py
# Requirements: pip install torch torchvision numpy scipy
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import random
import math
import os
import csv
import sys
from datetime import datetime
from scipy.stats import beta as beta_dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters


# Adaptive worker count — uses up to 4 workers but never more than CPU cores
NUM_WORKERS = min(4, os.cpu_count() or 1)


# ============================================================
# LOGGING — capture ALL stdout/stderr to a timestamped log file
# while still printing to terminal. No print statements need
# to be changed; this wraps the entire output stream.
# ============================================================

class TeeLogger:
    """Write to both a file and the original stream (terminal)."""
    def __init__(self, file_path, original_stream):
        self.file   = open(file_path, 'a', buffering=1)  # line-buffered
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

print(f"=" * 60)
print(f"Experiment log: {LOG_FILE}")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"=" * 60)

# ============================================================
# CONFIG — same for all methods
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

# FedCCFA sudden drift groups (mirrors drift.py exactly)
# 30% per group — client_id % 10 logic from FedCCFA
DRIFT_GROUP_A  = [i for i in range(NUM_CLIENTS) if i % 10 < 3]   # swap 1<->2
DRIFT_GROUP_B  = [i for i in range(NUM_CLIENTS) if 3 <= i%10 < 6] # swap 3<->4
DRIFT_GROUP_C  = [i for i in range(NUM_CLIENTS) if i % 10 >= 6]   # swap 5<->6
DRIFTING       = DRIFT_GROUP_A + DRIFT_GROUP_B + DRIFT_GROUP_C

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print(f"Device: {DEVICE}")
print(f"CPU cores detected: {os.cpu_count()} | DataLoader workers: {NUM_WORKERS}")
print(f"Config: {NUM_CLIENTS} clients, alpha={ALPHA_DIR}, "
      f"{LOCAL_EPOCHS} epochs, {NUM_ROUNDS} rounds, drift at {DRIFT_ROUND}")


# ============================================================
# DATA — FedCCFA Dirichlet partition
# ============================================================

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

print("Downloading CIFAR-10...")
raw_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
raw_test  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)


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


def partition_dataset(dataset, n, alpha, seed):
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


print("Partitioning data...")
train_idx = partition_dataset(raw_train, NUM_CLIENTS, ALPHA_DIR, SEED)

# Rebuild client datasets fresh for each method run
def build_client_sets():
    return {i: ClientDataset(raw_train, train_idx[i], train_transform)
            for i in range(NUM_CLIENTS)}

global_test_ds = ClientDataset(raw_test, list(range(len(raw_test))), test_transform)
global_loader  = DataLoader(global_test_ds, batch_size=512, shuffle=False,
                            num_workers=min(2, NUM_WORKERS), pin_memory=True)

sizes = [len(train_idx[i]) for i in range(NUM_CLIENTS)]
print(f"Samples/client: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.0f}")


# ============================================================
# DRIFT — FedCCFA exact sudden_drift
# ============================================================

def drift_dataset(dataset, a, b):
    targets = np.array(dataset.targets)
    ia = np.where(targets == a)[0]
    ib = np.where(targets == b)[0]
    for i in ia: dataset.targets[i] = -1
    for i in ib: dataset.targets[i] = a
    for i in np.where(np.array(dataset.targets) == -1)[0]:
        dataset.targets[i] = b

def apply_drift(client_sets):
    print(f"\n  *** Sudden drift at round {DRIFT_ROUND} ***")
    print(f"  Group A {DRIFT_GROUP_A[:3]}...: swap 1<->2")
    print(f"  Group B {DRIFT_GROUP_B[:3]}...: swap 3<->4")
    print(f"  Group C {DRIFT_GROUP_C[:3]}...: swap 5<->6")
    for cid in DRIFT_GROUP_A: drift_dataset(client_sets[cid], 1, 2)
    for cid in DRIFT_GROUP_B: drift_dataset(client_sets[cid], 3, 4)
    for cid in DRIFT_GROUP_C: drift_dataset(client_sets[cid], 5, 6)

def get_loader(client_sets, cid):
    return DataLoader(client_sets[cid], batch_size=BATCH_SIZE,
                      shuffle=True, drop_last=True,
                      num_workers=NUM_WORKERS, pin_memory=True,
                      persistent_workers=(NUM_WORKERS > 0))


# ============================================================
# MODEL — FedCCFA CifarCNN
# ============================================================

class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(3, 16, 5),  nn.LeakyReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding=1), nn.LeakyReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64*3*3, 128), nn.LeakyReLU()
        )
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        return self.fc(self.hidden_layers(x.float()))

def get_model():
    return CifarCNN().to(DEVICE)

print(f"Model: CifarCNN | "
      f"Params: {sum(p.numel() for p in get_model().parameters()):,}")


# ============================================================
# SHARED UTILITIES
# ============================================================

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total

def local_train(model, loader, epochs):
    model.train()
    opt  = optim.SGD(model.parameters(), lr=LR,
                     momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            crit(model(imgs.float()), labels).backward()
            opt.step()

def fedavg_aggregate(global_model, local_states, weights):
    gs  = global_model.state_dict()
    new = copy.deepcopy(gs)
    total = sum(weights)
    for key in gs:
        new[key] = torch.zeros_like(gs[key], dtype=torch.float32)
        for state, w in zip(local_states, weights):
            new[key] += (w / total) * state[key].float()
    global_model.load_state_dict(new)

def save_results(results, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results saved to {filename}")


# ============================================================
# METHOD 1 — FedAvg
# ============================================================

def run_fedavg():
    print("\n" + "="*60)
    print("METHOD 1: FedAvg")
    print("="*60)
    client_sets = build_client_sets()
    gm          = get_model()
    log         = []

    for rnd in range(NUM_ROUNDS):
        if rnd == DRIFT_ROUND:
            apply_drift(client_sets)

        states, weights = [], []
        for cid in range(NUM_CLIENTS):
            lm = get_model()
            lm.load_state_dict(copy.deepcopy(gm.state_dict()))
            local_train(lm, get_loader(client_sets, cid), LOCAL_EPOCHS)
            states.append(lm.state_dict())
            weights.append(len(client_sets[cid]))

        fedavg_aggregate(gm, states, weights)
        acc = evaluate(gm, global_loader)
        log.append({'round': rnd, 'global_acc': acc})

        if rnd % 10 == 0 or rnd == DRIFT_ROUND:
            tag = "  <-- DRIFT" if rnd == DRIFT_ROUND else ""
            print(f"  Round {rnd:03d} | Global: {acc:.4f}{tag}")

    save_results(log, 'results_FedAvg.csv')
    return log


# ============================================================
# METHOD 2 — Flash
# ============================================================

def run_flash():
    print("\n" + "="*60)
    print("METHOD 2: Flash")
    print("="*60)

    # Flash hyperparameters (from Flash.yaml)
    SERVER_LR      = 0.01
    LOSS_DECREMENT = 0.004
    BETA1          = 0.9
    BETA2          = 0.99
    TAU_FLASH      = 0.001

    client_sets  = build_client_sets()
    gm           = get_model()
    log          = []
    crit         = nn.CrossEntropyLoss()

    # Flash server momentum state
    first_mom  = 0
    second_mom = TAU_FLASH ** 2
    prev_2mom  = 0
    delta_mom  = 0
    beta3      = 0

    # Per-client previous val loss for early stopping
    prev_val_loss = {cid: -1 for cid in range(NUM_CLIENTS)}

    for rnd in range(NUM_ROUNDS):
        if rnd == DRIFT_ROUND:
            apply_drift(client_sets)

        updates, weights = [], []

        for cid in range(NUM_CLIENTS):
            lm = get_model()
            lm.load_state_dict(copy.deepcopy(gm.state_dict()))
            lm.train()

            init_params = parameters_to_vector(lm.parameters()).detach()
            opt = optim.SGD(lm.parameters(), lr=LR,
                            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            loader = get_loader(client_sets, cid)

            for epoch in range(LOCAL_EPOCHS):
                for imgs, labels in loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    opt.zero_grad()
                    crit(lm(imgs.float()), labels).backward()
                    opt.step()

                # Flash early stopping: check val loss decrease
                lm.eval()
                val_loss = 0.0
                with torch.no_grad():
                    vl = DataLoader(client_sets[cid], batch_size=256)
                    for vi, vlab in vl:
                        vi, vlab = vi.to(DEVICE), vlab.to(DEVICE)
                        val_loss += crit(lm(vi.float()), vlab).item()
                val_loss /= max(len(vl), 1)

                if prev_val_loss[cid] != -1:
                    delta = prev_val_loss[cid] - val_loss
                    if 0 < delta < LOSS_DECREMENT / (epoch + 1):
                        break
                prev_val_loss[cid] = val_loss
                lm.train()

            cur_params = parameters_to_vector(lm.parameters()).detach()
            updates.append((cur_params - init_params).cpu().numpy())
            weights.append(len(client_sets[cid]))

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

        acc = evaluate(gm, global_loader)
        log.append({'round': rnd, 'global_acc': acc})

        if rnd % 10 == 0 or rnd == DRIFT_ROUND:
            tag = "  <-- DRIFT" if rnd == DRIFT_ROUND else ""
            print(f"  Round {rnd:03d} | Global: {acc:.4f}{tag}")

    save_results(log, 'results_Flash.csv')
    return log


# ============================================================
# METHOD 3 — CDA-FedAvg (Casado et al. 2022)
# ============================================================

# Detection hyperparameters — all from paper Section 4.1 (Casado et al. 2022)
LAMBDA_CDA = 0.05                        # sensitivity to change
DELTA_CDA  = 100                         # minimum sub-window size
N_MAX_CDA  = 1000                        # max sliding window size
H_CDA      = 2.995732            # Th = -log(lambda) per paper Section 4.1
LTM_SIZE   = 2000                        # 200 samples/class x 10 classes (paper Section 4.2)


def fit_beta_mom_cda(x):
    """
    Method of moments for beta distribution parameters.
    Paper Algorithm 5 lines 9-10.
    Guards against degenerate distributions (over-confident CIFAR CNNs).
    """
    mu  = np.clip(x.mean(), 1e-6, 1.0 - 1e-6)
    var = x.var()
    max_var = mu * (1.0 - mu) - 1e-8
    if max_var <= 0 or var <= 0:
        return None, None
    var    = min(var, max_var)
    common = mu * (1.0 - mu) / var - 1.0
    return max(mu * common, 1e-6), max((1.0 - mu) * common, 1e-6)


def detect_drift_cda(Q_confidences):
    """
    Algorithm 5 from Casado et al. 2022 — corrected implementation.

    Key fixes vs broken version:
    1. Th = -log(lambda) = 2.9957, NOT 0.05
    2. sk = sum over ALL Q for each split k (vectorized)
    3. Beta PDF clipped to [1e-6, 1-1e-6] for over-confident CNNs
    4. Method-of-moments guards prevent negative alpha/beta
    """
    N = len(Q_confidences)
    if N < 2 * DELTA_CDA:
        return False

    # Clip: CIFAR CNNs produce max softmax = 1.0 -> log(beta_pdf(1)) = -inf
    Q = np.clip(np.array(Q_confidences), 1e-6, 1.0 - 1e-6)

    for k in range(DELTA_CDA, N - DELTA_CDA):
        Qa, Qb = Q[:k], Q[k:]
        ma, mb = Qa.mean(), Qb.mean()

        # Condition 1: detect only confidence drops (paper line 7)
        if ma > (1.0 - LAMBDA_CDA) * mb:
            continue

        alpha_a, beta_a = fit_beta_mom_cda(Qa)
        alpha_b, beta_b = fit_beta_mom_cda(Qb)
        if alpha_a is None or alpha_b is None:
            continue

        try:
            log_pa = beta_dist.logpdf(Q, alpha_a, beta_a)
            log_pb = beta_dist.logpdf(Q, alpha_b, beta_b)
            # Replace non-finite values with 0
            log_pa = np.where(np.isfinite(log_pa), log_pa, 0.0)
            log_pb = np.where(np.isfinite(log_pb), log_pb, 0.0)
            sk = (log_pa - log_pb).sum()
        except Exception:
            continue

        # Condition 2: Th = -log(lambda) per paper Section 4.1
        if sk > H_CDA:
            return True

    return False



class CDAClient:
    """
    CDA-FedAvg client — Algorithms 3 and 4 from Casado et al. 2022.
    Adapted to synchronous FL rounds.
    """
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
        self.model.load_state_dict(copy.deepcopy(global_state))

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

        # Step 1: initialize LTM on first round
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

        # Step 2: rehearsal training on LTM
        train_loader = self._ltm_loader() if self.LTM else loader
        self.model.train()
        opt = optim.SGD(self.model.parameters(), lr=LR,
                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        for _ in range(LOCAL_EPOCHS):
            for imgs, labels in train_loader:
                imgs   = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                opt.zero_grad()
                crit(self.model(imgs.float()), labels).backward()
                opt.step()

        # Steps 3-5: confidence computation and drift detection
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
                        self.Q.pop(0)
                        self.Q_data.pop(0)
                    self.Q.append(float(qi))
                    self.Q_data.append((img_cpu, label))
                    # Fire detection with prob e^{-2*qi} (paper Algorithm 4 line 12)
                    if random.random() < math.exp(-2.0 * float(qi)):
                        if detect_drift_cda(self.Q):
                            drift_found = True
                            break
                if drift_found:
                    break

        if drift_found:
            sampled = current_data
            if len(sampled) > LTM_SIZE:
                sampled = random.sample(sampled, LTM_SIZE)
            self.LTM.extend(sampled)
            # Keep LTM bounded to last 5 concepts
            if len(self.LTM) > LTM_SIZE * 5:
                self.LTM = self.LTM[-(LTM_SIZE * 5):]
            self.n_concepts += 1
            self.Q.clear()
            self.Q_data.clear()


def run_cda_fedavg():
    print("\n" + "="*60)
    print("METHOD 3: CDA-FedAvg (Casado et al. 2022)")
    print("="*60)
    client_sets = build_client_sets()
    gm          = get_model()
    clients     = [CDAClient(i, client_sets[i]) for i in range(NUM_CLIENTS)]
    log         = []

    for rnd in range(NUM_ROUNDS):
        if rnd == DRIFT_ROUND:
            apply_drift(client_sets)
            # Update client datasets after drift
            for c in clients:
                c.dataset = client_sets[c.cid]

        gs = copy.deepcopy(gm.state_dict())
        for c in clients:
            c.set_params(gs)
            c.train()

        # Standard FedAvg aggregation
        states  = [c.model.state_dict() for c in clients]
        weights = [len(client_sets[c.cid]) for c in clients]
        fedavg_aggregate(gm, states, weights)

        # Send updated params back
        for c in clients:
            c.set_params(gm.state_dict())

        acc = evaluate(gm, global_loader)
        log.append({'round': rnd, 'global_acc': acc})

        if rnd % 10 == 0 or rnd == DRIFT_ROUND:
            drifted = sum(1 for c in clients if c.n_concepts > 1)
            tag = "  <-- DRIFT" if rnd == DRIFT_ROUND else ""
            print(f"  Round {rnd:03d} | Global: {acc:.4f} | "
                  f"Detected: {drifted}/{NUM_CLIENTS}{tag}")

    save_results(log, 'results_CDAFedAvg.csv')
    return log


# ============================================================
# METHOD 4 — Our Method
# ============================================================

LAYER_GROUPS = {
    'layer1': 'hidden_layers.0',
    'layer2': 'hidden_layers.3',
    'layer3': 'hidden_layers.6',
    'layer4': 'hidden_layers.10',
}
DRIFT_LAYERS  = ['layer3', 'layer4']
STABLE_LAYERS = ['layer1', 'layer2']
CLASSIFIER    = 'fc'
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
        divs[name] = ((lv-gv).norm() / gv.norm()).item() if gv.norm() > 0 else 0.0
    return divs


class OurClient:
    def __init__(self, cid, dataset):
        self.cid         = cid
        self.dataset     = dataset
        self.model       = get_model()
        self.ema         = None
        self.prev_flags  = {l: False for l in LAYER_GROUPS}
        self.local_state = None
        self.upload_mask = {l: True for l in LAYER_GROUPS}
        self.rounds_seen = 0

    def selective_sync(self, global_state):
        state = copy.deepcopy(global_state)
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

    def train(self, global_state):
        self.rounds_seen += 1
        warmup = self.rounds_seen <= WARMUP
        crit   = nn.CrossEntropyLoss()

        self.selective_sync(global_state)

        self.model.train()
        opt = optim.SGD(self.model.parameters(), lr=LR,
                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        for _ in range(LOCAL_EPOCHS):
            for imgs, labels in get_loader({self.cid: self.dataset}, self.cid):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad()
                crit(self.model(imgs.float()), labels).backward()
                opt.step()

        self.local_state = copy.deepcopy(self.model.state_dict())
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
                self.ema[name] = EMA_ALPHA * d_l + (1-EMA_ALPHA) * ema_l
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
    new = copy.deepcopy(gs)

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

    total = sum(weights)
    for key in gs:
        if key.startswith(CLASSIFIER):
            new[key] = torch.zeros_like(gs[key], dtype=torch.float32)
            for c, w in zip(clients, weights):
                us = c.get_upload_state()
                if key in us:
                    new[key] += (w/total) * us[key].float()

    global_model.load_state_dict(new)


def run_our_method():
    print("\n" + "="*60)
    print("METHOD 4: Our Method — Drift-Triggered Selective Adaptation")
    print(f"EMA alpha={EMA_ALPHA} | TAU={TAU_OUR} | Warmup={WARMUP}")
    print("="*60)
    client_sets = build_client_sets()
    gm          = get_model()
    clients     = [OurClient(i, client_sets[i]) for i in range(NUM_CLIENTS)]
    log         = []
    flag_log    = []   # per-round per-client flag tracking

    for rnd in range(NUM_ROUNDS):
        if rnd == DRIFT_ROUND:
            apply_drift(client_sets)
            for c in clients:
                c.dataset = client_sets[c.cid]

        gs = copy.deepcopy(gm.state_dict())
        for c in clients:
            c.train(gs)

        weights = [len(client_sets[c.cid]) for c in clients]
        per_layer_fedavg_our(gm, clients, weights)

        acc = evaluate(gm, global_loader)
        log.append({'round': rnd, 'global_acc': acc})

        # Record per-client flag data EVERY round
        flagged_layer3 = [c.cid for c in clients if c.prev_flags.get('layer3', False)]
        flagged_layer4 = [c.cid for c in clients if c.prev_flags.get('layer4', False)]
        flagged_any    = sorted(set(flagged_layer3) | set(flagged_layer4))
        flag_log.append({
            'round': rnd,
            'flagged_count': len(flagged_any),
            'flagged_layer3_count': len(flagged_layer3),
            'flagged_layer4_count': len(flagged_layer4),
            'flagged_client_ids': ';'.join(map(str, flagged_any)),
            'flagged_layer3_ids': ';'.join(map(str, flagged_layer3)),
            'flagged_layer4_ids': ';'.join(map(str, flagged_layer4)),
            'global_acc': acc,
        })

        if rnd % 10 == 0 or rnd == DRIFT_ROUND or (
            DRIFT_ROUND <= rnd <= DRIFT_ROUND + 20):
            tag = "  <-- DRIFT" if rnd == DRIFT_ROUND else ""
            print(f"  Round {rnd:03d} | Global: {acc:.4f} | "
                  f"Flagged: {len(flagged_any)}/{NUM_CLIENTS} "
                  f"(L3:{len(flagged_layer3)}, L4:{len(flagged_layer4)}) "
                  f"IDs: {flagged_any}{tag}")

    save_results(log, 'results_OurMethod.csv')

    # Save per-round per-client flag data
    with open('results_OurMethod_flags.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'round', 'flagged_count',
            'flagged_layer3_count', 'flagged_layer4_count',
            'flagged_client_ids', 'flagged_layer3_ids',
            'flagged_layer4_ids', 'global_acc'])
        writer.writeheader()
        writer.writerows(flag_log)
    print(f"  Per-client flag log saved to results_OurMethod_flags.csv")
    return log


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STARTING ALL EXPERIMENTS")
    print("="*60)

    fedavg_log  = run_fedavg()
    flash_log   = run_flash()
    cda_log     = run_cda_fedavg()
    our_log     = run_our_method()

    # Print final comparison
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    def get_metrics(log, drift_round=DRIFT_ROUND, window=10, tol=0.02):
        accs   = [r['global_acc'] for r in log]
        pre    = float(np.mean(accs[max(0, drift_round-11):drift_round]))
        dip    = pre - min(accs[drift_round:drift_round+window])
        rec    = next((i-drift_round for i in range(drift_round, len(accs))
                       if accs[i] >= pre-tol), None)
        stable = float(np.mean(accs[-10:]))
        return pre, dip, rec, stable

    for name, log in [("FedAvg",    fedavg_log),
                      ("Flash",     flash_log),
                      ("CDA-FedAvg",cda_log),
                      ("OurMethod", our_log)]:
        pre, dip, rec, stable = get_metrics(log)
        print(f"\n{name}:")
        print(f"  Pre-drift acc:     {pre:.4f}")
        print(f"  Accuracy dip:      {dip:.4f}")
        print(f"  Recovery rounds:   "
              f"{rec if rec is not None else 'Not recovered'}")
        print(f"  Post-drift stable: {stable:.4f}")

    print("\nAll CSV files saved. Done.")
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Full output log: {LOG_FILE}")
    print(f"CSV files:")
    print(f"  - results_FedAvg.csv")
    print(f"  - results_Flash.csv")
    print(f"  - results_CDAFedAvg.csv")
    print(f"  - results_OurMethod.csv")
    print(f"  - results_OurMethod_flags.csv (per-round per-client flag data)")
