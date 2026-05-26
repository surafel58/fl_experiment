# ============================================================
# Drift-Triggered Selective Layer Adaptation in FL
# Local experiment (Windows / RTX GPU)
#
# DRIFT TYPE: Input distribution shift via image corruption
#   At drift round, drifting clients' data is replaced with
#   corrupted versions (Gaussian noise + contrast shift).
#   This changes P(X) — the visual features themselves change.
#   Upper extractor layers must adapt their feature detectors.
#   Lower layers (edges, textures) remain relatively stable.
#   This matches the assumption in Lee et al. (ICLR 2023) and
#   Ramasesh et al. (ICLR 2021).
#
# SIGNAL: Per-layer weight divergence
#   d_l = ||W_l^local - W_l^global||_F / ||W_l^global||_F
#   EMA-based detection: flag if d_l/ema_l > TAU
#
# Setup: 6 clients, 20 rounds, drift at round 10
#   Group A (0,1): Gaussian noise corruption at drift round
#   Group B (2,3): Contrast + brightness corruption at drift round
#   Stable (4,5): clean data throughout
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ---- Hyperparameters ----
NUM_CLIENTS = 6
NUM_ROUNDS = 20
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
DRIFT_ROUND = 10
ALPHA_DIR = 0.5
WARMUP_ROUNDS = 5
EMA_ALPHA = 0.3
TAU = 1.4
SEED = 0

DRIFT_GROUP_A = [0, 1]   # Gaussian noise corruption
DRIFT_GROUP_B = [2, 3]   # Contrast shift corruption
STABLE_CLIENTS = [4, 5]
DRIFTING_CLIENTS = DRIFT_GROUP_A + DRIFT_GROUP_B

LAYER_NAMES = ['conv1', 'conv2', 'conv3', 'conv4']
DRIFT_LAYERS = ['conv3', 'conv4']
STABLE_LAYERS = ['conv1', 'conv2']

# Corruption parameters — strong enough to force feature adaptation
NOISE_STD = 0.5    # Gaussian noise std (on normalized images)
CONTRAST_FACTOR = 0.3    # contrast reduction factor

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f"Signal: per-layer WEIGHT DIVERGENCE")
print(f"  d_l = ||W_local - W_global||_F / ||W_global||_F")
print(f"  EMA alpha={EMA_ALPHA}, tau={TAU}, flag if d_l/ema_l > {TAU}")
print(f"Drift at round {DRIFT_ROUND}:")
print(f"  Group A {DRIFT_GROUP_A}: Gaussian noise (std={NOISE_STD})")
print(f"  Group B {DRIFT_GROUP_B}: Contrast shift (factor={CONTRAST_FACTOR})")
print(f"  Stable {STABLE_CLIENTS}: clean throughout")


# ============================================================
# Data and Corruption Transforms
# ============================================================

# Standard clean transform
clean_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# Post-normalize Gaussian noise (applied in __getitem__ after normalization)
# Contrast shift: reduces contrast by mixing toward gray mean
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

raw_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True)
raw_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True)


class ClientDataset(Dataset):
    """
    Client dataset with switchable corruption.
    corruption: None | 'noise' | 'contrast'
    Corruption is applied AFTER normalization to simulate
    a real distribution shift in the feature space.
    """

    def __init__(self, base_dataset, indices,
                 base_transform=None, corruption=None):
        self.data = base_dataset.data[indices]
        self.targets = list(np.array(base_dataset.targets)[indices])
        self.base_transform = base_transform
        self.corruption = corruption

    def set_corruption(self, corruption):
        self.corruption = corruption

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, label = self.data[idx], int(self.targets[idx])

        if self.base_transform:
            img = self.base_transform(img)  # returns tensor

        # Apply corruption AFTER normalization
        if self.corruption == 'noise':
            # Gaussian noise — disrupts fine-grained features
            # Upper layers that encode complex patterns are hit harder
            noise = torch.randn_like(img) * NOISE_STD
            img = img + noise
            img = torch.clamp(img, -3.0, 3.0)

        elif self.corruption == 'contrast':
            # Contrast reduction — reduces distinction between features
            # Equivalent to mixing toward zero mean in normalized space
            img = img * CONTRAST_FACTOR

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


print("Partitioning data (Dirichlet alpha=0.5)...")
train_indices = partition_dataset(raw_train, NUM_CLIENTS, ALPHA_DIR, SEED)

# All clients start with clean data
client_train_sets = {
    i: ClientDataset(raw_train, train_indices[i],
                     base_transform=clean_transform,
                     corruption=None)
    for i in range(NUM_CLIENTS)
}

global_test_ds = ClientDataset(raw_test, list(range(len(raw_test))),
                               base_transform=test_transform,
                               corruption=None)
global_test_loader = DataLoader(global_test_ds, batch_size=256, shuffle=False)

sizes = [len(client_train_sets[i]) for i in range(NUM_CLIENTS)]
print(f"Samples per client: min={min(sizes)}, max={max(sizes)}, "
      f"mean={np.mean(sizes):.0f}")


# ============================================================
# Drift — apply corruption at drift round
# ============================================================

def apply_sudden_drift():
    """
    Switch drifting clients to corrupted input distributions.
    Group A: Gaussian noise (std=0.5) — disrupts high-frequency features
    Group B: Contrast reduction (factor=0.3) — suppresses mid-level features
    Both force the upper extractor layers to adapt their representations.
    """
    print(f"  Input distribution shift at round {DRIFT_ROUND}:")
    print(f"    Group A {DRIFT_GROUP_A}: + Gaussian noise (std={NOISE_STD})")
    print(
        f"    Group B {DRIFT_GROUP_B}: + Contrast reduction ({CONTRAST_FACTOR}x)")
    for cid in DRIFT_GROUP_A:
        client_train_sets[cid].set_corruption('noise')
    for cid in DRIFT_GROUP_B:
        client_train_sets[cid].set_corruption('contrast')


def reset_datasets():
    for cid in range(NUM_CLIENTS):
        client_train_sets[cid].set_corruption(None)


def get_loader(cid):
    return DataLoader(client_train_sets[cid], batch_size=BATCH_SIZE,
                      shuffle=True, drop_last=True)


# ============================================================
# Model
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.classifier(x.view(x.size(0), -1))


def get_model():
    return SimpleCNN().to(device)


print(f"Model params: {sum(p.numel() for p in get_model().parameters()):,}")


# ============================================================
# Training + Weight Divergence + Drift Detection
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


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


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
        divs[name] = ((lv - gv).norm() / gv.norm()
                      ).item() if gv.norm() > 0 else 0.0
    return divs


_ema = {}
_diag = {}


def compute_flags_divergence(divs, client_id, rnd, warmup):
    global _ema
    if client_id not in _ema:
        _ema[client_id] = {l: divs[l] for l in LAYER_NAMES}

    flags = {l: False for l in LAYER_NAMES}
    diag = {}

    for name in LAYER_NAMES:
        d_l = divs[name]
        ema_l = _ema[client_id][name]
        ratio = d_l / ema_l if ema_l > 0 else 1.0
        diag[name] = (round(d_l, 6), round(ema_l, 6), round(ratio, 3))

        if warmup:
            _ema[client_id][name] = d_l   # reset EMA to current during warmup
        else:
            _ema[client_id][name] = EMA_ALPHA * d_l + (1 - EMA_ALPHA) * ema_l
            if name in DRIFT_LAYERS and ratio > TAU:
                flags[name] = True

    _diag[(rnd, client_id)] = diag
    return flags


def build_local_model(global_state, prev_local_state, prev_flags):
    """
    Step 1 — Selective sync:
    Unflagged layers: use global (accept aggregated knowledge)
    Flagged layers: restore from previous local trained weights
    Classifier: always from global
    """
    state = copy.deepcopy(global_state)
    if prev_local_state is not None:
        for name in LAYER_NAMES:
            if prev_flags.get(name, False):
                for key in state:
                    if key.startswith(name):
                        state[key] = prev_local_state[key].clone()
    for key in state:
        if key.startswith('classifier'):
            state[key] = global_state[key].clone()
    lm = get_model()
    lm.load_state_dict(state)
    return lm


def per_layer_fedavg(global_model, client_uploads):
    gs = global_model.state_dict()
    new = copy.deepcopy(gs)
    for lname in LAYER_NAMES:
        contrib = [sd for sd, mask in client_uploads
                   if mask.get(lname, True)]
        if not contrib:
            continue
        for key in gs:
            if key.startswith(lname):
                new[key] = torch.stack(
                    [sd[key].float() for sd in contrib]).mean(0)
    all_sds = [sd for sd, _ in client_uploads]
    for key in gs:
        if key.startswith('classifier'):
            new[key] = torch.stack(
                [sd[key].float() for sd in all_sds]).mean(0)
    global_model.load_state_dict(new)


# ============================================================
# FedAvg Baseline
# ============================================================

def run_fedavg():
    print("\n" + "="*55)
    print("FedAvg Baseline")
    print("="*55)
    reset_datasets()
    gm = get_model()
    acc = []
    for rnd in range(1, NUM_ROUNDS+1):
        if rnd == DRIFT_ROUND:
            apply_sudden_drift()
        states = []
        for cid in range(NUM_CLIENTS):
            lm = get_model()
            lm.load_state_dict(copy.deepcopy(gm.state_dict()))
            local_train(lm, get_loader(cid), LOCAL_EPOCHS)
            states.append(lm.state_dict())
        ns = copy.deepcopy(states[0])
        for key in ns:
            ns[key] = torch.stack(
                [s[key].float() for s in states]).mean(0)
        gm.load_state_dict(ns)
        a = evaluate(gm, global_test_loader)
        acc.append(a)
        if rnd % 5 == 0 or rnd == DRIFT_ROUND:
            tag = "  <-- DRIFT" if rnd == DRIFT_ROUND else ""
            print(f"  Round {rnd:02d} | Acc: {a:.4f}{tag}")
    return acc


# ============================================================
# Our Method
# ============================================================

def run_our_method():
    print("\n" + "="*55)
    print("Our Method — Weight Divergence + Local Persistence")
    print(f"EMA alpha={EMA_ALPHA} | TAU={TAU}")
    print("="*55)
    reset_datasets()
    global _ema, _diag
    _ema = {}
    _diag = {}

    gm = get_model()
    acc = []
    prev_flags = {cid: {l: False for l in LAYER_NAMES}
                  for cid in range(NUM_CLIENTS)}
    flag_history = {cid: {l: [] for l in LAYER_NAMES}
                    for cid in range(NUM_CLIENTS)}
    client_local_states = {cid: None for cid in range(NUM_CLIENTS)}

    for rnd in range(1, NUM_ROUNDS+1):
        if rnd == DRIFT_ROUND:
            apply_sudden_drift()

        uploads = []
        current_flags = {}
        gs = copy.deepcopy(gm.state_dict())

        for cid in range(NUM_CLIENTS):
            loader = get_loader(cid)

            # Step 1 — Selective sync (with local persistence)
            lm = build_local_model(gs, client_local_states[cid],
                                   prev_flags[cid])

            # Step 2 — Local training
            local_train(lm, loader, LOCAL_EPOCHS)

            # Store trained weights for next round
            trained_state = copy.deepcopy(lm.state_dict())
            client_local_states[cid] = trained_state

            # Step 3+4 — Weight divergence + EMA detection
            divs = compute_weight_divergence(trained_state, gs)
            flags = compute_flags_divergence(
                divs, cid, rnd, warmup=(rnd <= WARMUP_ROUNDS))

            current_flags[cid] = flags
            for l in LAYER_NAMES:
                flag_history[cid][l].append(flags[l])

            # Step 5 — Selective upload
            uploads.append((trained_state,
                            {l: not flags[l] for l in LAYER_NAMES}))

        per_layer_fedavg(gm, uploads)
        prev_flags = current_flags

        a = evaluate(gm, global_test_loader)
        acc.append(a)

        if rnd % 5 == 0 or rnd == DRIFT_ROUND:
            tag = "  <-- DRIFT" if rnd == DRIFT_ROUND else ""
            rc = DRIFT_GROUP_A[0]
            fl = [l for l in LAYER_NAMES if current_flags[rc][l]]
            print(f"  Round {rnd:02d} | Acc: {a:.4f} | "
                  f"C{rc} flags: {fl if fl else 'none'}{tag}")

    return acc, flag_history


# ============================================================
# Run, Diagnostics, Results, Plots
# ============================================================

fedavg_acc = run_fedavg()
our_acc, flag_history = run_our_method()


def print_diag(client_id, label):
    print(f"\nWEIGHT DIVERGENCE DIAGNOSTICS — Client {client_id} ({label})")
    print(f"tau={TAU} | flag if ratio > {TAU} (increase only)")
    print(f"{'Rnd':<5} "
          f"{'d_c1':>10} {'r_c1':>6} "
          f"{'d_c2':>10} {'r_c2':>6} "
          f"{'d_c3':>10} {'r_c3':>6} "
          f"{'d_c4':>10} {'r_c4':>6}  flags")
    print("-" * 90)
    for rnd in range(1, NUM_ROUNDS+1):
        d = _diag.get((rnd, client_id))
        if d:
            tag = " <-DRIFT" if rnd == DRIFT_ROUND else ""
            fl = []
            if rnd > WARMUP_ROUNDS:
                for l in DRIFT_LAYERS:
                    if d[l][2] > TAU:
                        fl.append(l)
            print(f"{rnd:<5} "
                  f"{d['conv1'][0]:>10.6f} {d['conv1'][2]:>6.3f} "
                  f"{d['conv2'][0]:>10.6f} {d['conv2'][2]:>6.3f} "
                  f"{d['conv3'][0]:>10.6f} {d['conv3'][2]:>6.3f} "
                  f"{d['conv4'][0]:>10.6f} {d['conv4'][2]:>6.3f}  "
                  f"{fl if fl else ''}{tag}")


print_diag(0, "drifting — Group A noise")
print_diag(4, "STABLE")


def compute_metrics(accs, drift_round, window=5, tol=0.02):
    pre = float(np.mean(accs[max(0, drift_round-6):drift_round-1]))
    dip = pre - min(accs[drift_round-1:drift_round-1+window])
    rec = next((i-(drift_round-1)
                for i in range(drift_round-1, len(accs))
                if accs[i] >= pre-tol), None)
    stable = float(np.mean(accs[-5:]))
    return pre, dip, rec, stable


print("\n" + "="*55)
print("RESULTS SUMMARY")
print("="*55)
for label, accs in [("FedAvg", fedavg_acc), ("Our Method", our_acc)]:
    pre, dip, rec, stable = compute_metrics(accs, DRIFT_ROUND)
    print(f"\n{label}:")
    print(f"  Pre-drift accuracy:    {pre:.4f}")
    print(f"  Accuracy dip:          {dip:.4f}")
    print(f"  Recovery rounds:       "
          f"{rec if rec is not None else 'Not recovered'}")
    print(f"  Post-drift stable acc: {stable:.4f}")


rounds = list(range(1, NUM_ROUNDS+1))
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.plot(rounds, fedavg_acc, 'o-', color='#E07B54',
        lw=2, ms=4, label='FedAvg')
ax.plot(rounds, our_acc, 's-', color='#2E86AB',
        lw=2, ms=4, label='Our Method')
ax.axvline(x=DRIFT_ROUND, color='red', ls='--', lw=1.5,
           label=f'Drift onset (round {DRIFT_ROUND})')
ax.set_xlabel('Round', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title(f'Test Accuracy — Input Distribution Shift Drift\n'
             f'({NUM_CLIENTS} clients, noise+contrast corruption, '
             f'EMA α={EMA_ALPHA}, τ={TAU})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, NUM_ROUNDS)
ax.set_ylim(0, 1)

ax2 = axes[1]
vc = DRIFT_GROUP_A[0]
fmat = np.array([[1 if flag_history[vc][l][r] else 0
                  for r in range(NUM_ROUNDS)]
                 for l in LAYER_NAMES])
ax2.imshow(fmat, aspect='auto', cmap='RdYlGn_r',
           vmin=0, vmax=1, interpolation='nearest')
ax2.set_yticks(range(len(LAYER_NAMES)))
ax2.set_yticklabels(LAYER_NAMES, fontsize=11)
ax2.set_xlabel('Round', fontsize=12)
ax2.set_title(f'Layer Flags — Client {vc} (noise corruption)\n'
              f'Green = stable | Red = drifted (kept local)',
              fontsize=12, fontweight='bold')
ax2.axvline(x=DRIFT_ROUND-1.5, color='red', ls='--', lw=2)
ax2.set_xticks(range(0, NUM_ROUNDS, 5))
ax2.set_xticklabels(range(1, NUM_ROUNDS+1, 5))
ax2.legend(
    handles=[mpatches.Patch(color='#4CAF50', label='Stable — uploaded'),
             mpatches.Patch(color='#F44336', label='Drifted — kept local')],
    loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('drift_fl_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as drift_fl_results.png")
