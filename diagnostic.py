# ============================================================
# diagnostic.py
# Signal verification before full implementation
#
# PURPOSE: Confirm that weight divergence produces a visible
#   asymmetric signal on upper vs lower layers when drift hits,
#   using the correct config (alpha=0.1, 5 epochs, 20 clients).
#
# Run this BEFORE the full implementation.
# If conv3/conv4 ratios spike clearly after round 15 on drifting
# clients but NOT on stable clients → signal is valid → proceed.
#
# Takes ~20-30 minutes on RTX 4050.
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ---- Config ----
NUM_CLIENTS = 20
NUM_ROUNDS = 30       # enough to see pre and post drift
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
DRIFT_ROUND = 15       # FedCCFA fires at round 100 normally,
# we use 15 here for quick validation
ALPHA_DIR = 0.1      # LOW — high non-IID, strong drift signal
WARMUP_ROUNDS = 10
EMA_ALPHA = 0.3
SEED = 0

# FedCCFA sudden drift groups (mirrors drift.py exactly)
# 30% of clients per group — matches FedCCFA's client_id % 10 logic
DRIFT_GROUP_A = list(range(0, 6))    # swap 1 <-> 2
DRIFT_GROUP_B = list(range(6, 12))   # swap 3 <-> 4
DRIFT_GROUP_C = list(range(12, 18))  # swap 5 <-> 6
STABLE_CLIENTS = [18, 19]
DRIFTING_CLIENTS = DRIFT_GROUP_A + DRIFT_GROUP_B + DRIFT_GROUP_C

LAYER_NAMES = ['conv1', 'conv2', 'conv3', 'conv4']

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"Config: {NUM_CLIENTS} clients, alpha={ALPHA_DIR}, "
      f"{LOCAL_EPOCHS} epochs, drift at round {DRIFT_ROUND}")
print(f"Drifting: {len(DRIFTING_CLIENTS)}/20 clients "
      f"| Stable: {STABLE_CLIENTS}")


# ============================================================
# Data — FedCCFA exact partition
# ============================================================

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

raw_train = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True)


class ClientDataset(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.data = base_dataset.data[indices]
        self.targets = list(np.array(base_dataset.targets)[indices])
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, label = self.data[idx], int(self.targets[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


def partition_dataset(dataset, client_num, alpha, seed):
    labels = np.array(dataset.targets)
    num_classes = len(set(labels))
    client_indices = [[] for _ in range(client_num)]
    essential_num = 5
    for k in range(num_classes):
        class_idx = np.where(labels == k)[0]
        essential = np.array_split(
            class_idx[:client_num * essential_num], client_num)
        client_indices = [ci + e.tolist()
                          for ci, e in zip(client_indices, essential)]
        remaining = class_idx[client_num * essential_num:]
        random.seed(seed + k)
        np.random.seed(seed + k)
        props = np.random.dirichlet(np.repeat(alpha, client_num))
        splits = (np.cumsum(props / props.sum()) * len(remaining)
                  ).astype(int)[:-1]
        client_indices = [ci + chunk.tolist() for ci, chunk in
                          zip(client_indices, np.split(remaining, splits))]
    for ci in client_indices:
        random.shuffle(ci)
    return client_indices


print("Partitioning (Dirichlet alpha=0.1)...")
train_indices = partition_dataset(raw_train, NUM_CLIENTS, ALPHA_DIR, SEED)
client_sets = {
    i: ClientDataset(raw_train, train_indices[i], train_transform)
    for i in range(NUM_CLIENTS)
}
sizes = [len(client_sets[i]) for i in range(NUM_CLIENTS)]
print(f"Samples/client: min={min(sizes)}, max={max(sizes)}, "
      f"mean={np.mean(sizes):.0f}")


# ============================================================
# Drift — FedCCFA exact
# ============================================================

def drift_dataset(dataset, class_a, class_b):
    targets = np.array(dataset.targets)
    idx_a = np.where(targets == class_a)[0]
    idx_b = np.where(targets == class_b)[0]
    for i in idx_a:
        dataset.targets[i] = -1
    for i in idx_b:
        dataset.targets[i] = class_a
    for i in np.where(np.array(dataset.targets) == -1)[0]:
        dataset.targets[i] = class_b


def apply_sudden_drift():
    print(f"\n  *** SUDDEN DRIFT at round {DRIFT_ROUND} ***")
    print(f"  Group A {DRIFT_GROUP_A[:3]}...: swap 1<->2")
    print(f"  Group B {DRIFT_GROUP_B[:3]}...: swap 3<->4")
    print(f"  Group C {DRIFT_GROUP_C[:3]}...: swap 5<->6")
    for cid in DRIFT_GROUP_A:
        drift_dataset(client_sets[cid], 1, 2)
    for cid in DRIFT_GROUP_B:
        drift_dataset(client_sets[cid], 3, 4)
    for cid in DRIFT_GROUP_C:
        drift_dataset(client_sets[cid], 5, 6)


def get_loader(cid):
    return DataLoader(client_sets[cid], batch_size=BATCH_SIZE,
                      shuffle=True, drop_last=True)


# ============================================================
# Model — FedCCFA's CifarCNN (matches their experiments)
# ============================================================

class CifarCNN(nn.Module):
    """
    FedCCFA's exact CifarCNN from models.py — using this
    instead of SimpleCNN to match their experimental conditions.
    Named blocks for per-layer divergence tracking.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.LeakyReLU())
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.classifier(x)


# Update layer names to match CifarCNN
LAYER_NAMES = ['conv1', 'conv2', 'conv3', 'fc']
DRIFT_LAYERS = ['conv3', 'fc']   # upper layers
STABLE_LAYERS = ['conv1', 'conv2']


def get_model():
    return CifarCNN().to(device)


print(f"Model: CifarCNN (FedCCFA exact) | "
      f"Params: {sum(p.numel() for p in get_model().parameters()):,}")
print(f"Drift-flaggable: {DRIFT_LAYERS} | Always-global: {STABLE_LAYERS}")


# ============================================================
# Weight Divergence Signal
# ============================================================

def local_train(model, loader, epochs):
    model.train()
    opt = optim.SGD(model.parameters(), lr=LR,
                    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            crit(model(imgs), labels).backward()
            opt.step()


def compute_weight_divergence(local_state, global_state):
    """d_l = ||W_l^local - W_l^global||_F / ||W_l^global||_F"""
    divs = {}
    for name in LAYER_NAMES:
        lp, gp = [], []
        for key in local_state:
            if key.startswith(name):
                lp.append(local_state[key].float().flatten())
                gp.append(global_state[key].float().flatten())
        if not lp:
            divs[name] = 0.0
            continue
        lv = torch.cat(lp)
        gv = torch.cat(gp)
        divs[name] = ((lv-gv).norm() / gv.norm()
                      ).item() if gv.norm() > 0 else 0.0
    return divs


# ============================================================
# Diagnostic run — track divergence per round
# ============================================================

# Store: {(rnd, cid): {layer: divergence}}
divergence_log = {}

gm = get_model()

for rnd in range(1, NUM_ROUNDS+1):
    if rnd == DRIFT_ROUND:
        apply_sudden_drift()

    gs = copy.deepcopy(gm.state_dict())
    states = []

    for cid in range(NUM_CLIENTS):
        lm = get_model()
        lm.load_state_dict(copy.deepcopy(gs))
        local_train(lm, get_loader(cid), LOCAL_EPOCHS)
        trained = copy.deepcopy(lm.state_dict())
        states.append(trained)

        # Record divergence for diagnostic clients only
        if cid in [DRIFT_GROUP_A[0], STABLE_CLIENTS[0]]:
            divs = compute_weight_divergence(trained, gs)
            divergence_log[(rnd, cid)] = divs

    # Standard FedAvg aggregation
    ns = copy.deepcopy(states[0])
    for key in ns:
        ns[key] = torch.stack(
            [s[key].float() for s in states]).mean(0)
    gm.load_state_dict(ns)

    if rnd % 5 == 0 or rnd == DRIFT_ROUND:
        print(f"  Round {rnd:02d} done"
              f"{'  <-- DRIFT' if rnd == DRIFT_ROUND else ''}")

print("\nTraining complete. Printing divergence diagnostics...\n")


# ============================================================
# Print diagnostic table
# ============================================================

for cid, label in [(DRIFT_GROUP_A[0], "DRIFTING (Group A, swap 1<->2)"),
                   (STABLE_CLIENTS[0], "STABLE")]:
    print(f"CLIENT {cid} — {label}")
    print(f"{'Rnd':<5} "
          f"{'d_conv1':>12} "
          f"{'d_conv2':>12} "
          f"{'d_conv3':>12} "
          f"{'d_fc':>12}  "
          f"ratio_c3/c1")
    print("-" * 72)

    prev = None
    for rnd in range(1, NUM_ROUNDS+1):
        d = divergence_log.get((rnd, cid))
        if d:
            # ratio of upper to lower as asymmetry measure
            upper = (d['conv3'] + d['fc']) / 2
            lower = (d['conv1'] + d['conv2']) / 2
            ratio = upper / lower if lower > 0 else 0
            tag = " <--DRIFT" if rnd == DRIFT_ROUND else ""
            print(f"{rnd:<5} "
                  f"{d['conv1']:>12.6f} "
                  f"{d['conv2']:>12.6f} "
                  f"{d['conv3']:>12.6f} "
                  f"{d['fc']:>12.6f}  "
                  f"{ratio:>6.3f}{tag}")
    print()

print("="*60)
print("WHAT TO LOOK FOR:")
print("  If the signal works:")
print("    - ratio_c3/c1 should INCREASE after drift for drifting client")
print("    - ratio_c3/c1 should stay FLAT for stable client")
print("    - The increase should be > 1.4x to cross TAU")
print()
print("  If ratio_c3/c1 looks the same for both clients after drift:")
print("    - The 2-class label swap still does not affect upper layers")
print("    - Need stronger drift or different architecture")
print("="*60)
