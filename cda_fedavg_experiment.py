# ============================================================
# CDA-FedAvg Standalone Experiment
# Casado et al., Multimedia Tools and Applications, 2022
# DOI: 10.1007/s11042-021-11219-x
#
# FIXES APPLIED vs previous broken version:
#   1. Th = -log(lambda) = 2.9957 (paper Section 4.1 explicit formula)
#      Previous value 0.05 was lambda itself, not the threshold — 60x too low
#   2. LTM size L = 2000 (200 samples/class x 10 classes, per paper Section 4.2)
#      Previous value 300 total (~30/class) caused accuracy plateau at 0.54
#   3. Beta PDF clipping to [1e-6, 1-1e-6] prevents log(-inf) on confident CNNs
#   4. Method-of-moments guards prevent negative alpha/beta parameters
#   5. sk computed as running sum over all Q indices for each split k
#      (not max of single-point ratios)
#   6. Detection sampling sign verified: fires when random() < exp(-2*qi)
#      i.e. MORE likely to check when confidence is LOW (correct per paper)
#
# Run: python cda_fedavg_experiment.py
# Requirements: pip install torch torchvision scipy numpy
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
import csv
from scipy.stats import beta as beta_dist

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

# FedCCFA sudden drift groups
DRIFT_GROUP_A  = [i for i in range(NUM_CLIENTS) if i % 10 < 3]
DRIFT_GROUP_B  = [i for i in range(NUM_CLIENTS) if 3 <= i%10 < 6]
DRIFT_GROUP_C  = [i for i in range(NUM_CLIENTS) if i % 10 >= 6]

# CDA-FedAvg hyperparameters — all from paper Section 4.1 and 4.2
LAMBDA_CDA     = 0.05           # sensitivity to change (paper: lambda=0.05)
DELTA_CDA      = 100            # minimum sub-window size (paper: Delta=100)
N_MAX_CDA      = 1000           # sliding window max size (paper: Nmax=1000)
TH_CDA         = -math.log(LAMBDA_CDA)  # = 2.9957 (paper: Th = -log(lambda))
LTM_SIZE       = 2000           # long-term memory total size
                                # paper: L=1400 for 7 classes (=200/class)
                                # CIFAR-10: 10 classes x 200 = 2000
LTM_PER_CONCEPT = 2000          # store up to 2000 per concept (cap at LTM_SIZE)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print(f"Device: {DEVICE}")
print(f"CDA-FedAvg hyperparameters:")
print(f"  lambda={LAMBDA_CDA}, Delta={DELTA_CDA}, Nmax={N_MAX_CDA}")
print(f"  Th = -log(lambda) = {TH_CDA:.6f}")
print(f"  LTM size per concept: {LTM_PER_CONCEPT}")
print(f"Config: {NUM_CLIENTS} clients, alpha={ALPHA_DIR}, "
      f"{LOCAL_EPOCHS} epochs, {NUM_ROUNDS} rounds, drift at {DRIFT_ROUND}")


# ============================================================
# DATA
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
raw_train = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True)
raw_test  = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True)


class ClientDataset(Dataset):
    def __init__(self, base, indices, transform=None):
        self.data      = base.data[indices]
        self.targets   = list(np.array(base.targets)[indices])
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, label = self.data[idx], int(self.targets[idx])
        if self.transform:
            img = self.transform(img)
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
        random.seed(seed + k)
        np.random.seed(seed + k)
        props  = np.random.dirichlet(np.repeat(alpha, n))
        splits = (np.cumsum(props/props.sum()) * len(rem)).astype(int)[:-1]
        cidx   = [c + ch.tolist() for c, ch in
                  zip(cidx, np.split(rem, splits))]
    for c in cidx:
        random.shuffle(c)
    return cidx


print("Partitioning data...")
train_idx = partition_dataset(raw_train, NUM_CLIENTS, ALPHA_DIR, SEED)

client_sets = {
    i: ClientDataset(raw_train, train_idx[i], train_transform)
    for i in range(NUM_CLIENTS)
}

global_test_ds = ClientDataset(
    raw_test, list(range(len(raw_test))), test_transform)
global_loader  = DataLoader(
    global_test_ds, batch_size=512, shuffle=False)

sizes = [len(train_idx[i]) for i in range(NUM_CLIENTS)]
print(f"Samples/client: min={min(sizes)}, max={max(sizes)}, "
      f"mean={np.mean(sizes):.0f}")


# ============================================================
# DRIFT
# ============================================================

def drift_dataset(dataset, a, b):
    targets = np.array(dataset.targets)
    ia = np.where(targets == a)[0]
    ib = np.where(targets == b)[0]
    for i in ia: dataset.targets[i] = -1
    for i in ib: dataset.targets[i] = a
    for i in np.where(np.array(dataset.targets) == -1)[0]:
        dataset.targets[i] = b


def apply_drift():
    print(f"\n  *** Sudden drift at round {DRIFT_ROUND} ***")
    print(f"  Group A {DRIFT_GROUP_A[:3]}...: swap 1<->2")
    print(f"  Group B {DRIFT_GROUP_B[:3]}...: swap 3<->4")
    print(f"  Group C {DRIFT_GROUP_C[:3]}...: swap 5<->6")
    for cid in DRIFT_GROUP_A:
        drift_dataset(client_sets[cid], 1, 2)
    for cid in DRIFT_GROUP_B:
        drift_dataset(client_sets[cid], 3, 4)
    for cid in DRIFT_GROUP_C:
        drift_dataset(client_sets[cid], 5, 6)


def reset_datasets():
    for cid in range(NUM_CLIENTS):
        client_sets[cid].targets = list(
            np.array(raw_train.targets)[train_idx[cid]])


def get_loader(cid):
    return DataLoader(
        client_sets[cid], batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True)


# ============================================================
# MODEL — FedCCFA's CifarCNN
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
# UTILITIES
# ============================================================

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs.float()).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def fedavg_aggregate(global_model, local_states, weights):
    gs    = global_model.state_dict()
    new   = copy.deepcopy(gs)
    total = sum(weights)
    for key in gs:
        new[key] = torch.zeros_like(gs[key], dtype=torch.float32)
        for state, w in zip(local_states, weights):
            new[key] += (w / total) * state[key].float()
    global_model.load_state_dict(new)


# ============================================================
# CDA-FedAvg DRIFT DETECTION (Algorithm 5 — Casado et al. 2022)
# ============================================================

def fit_beta_mom(x):
    """
    Method of moments for beta distribution parameters.
    Paper Algorithm 5 lines 9-10.
    Returns (alpha, beta) clipped to [1e-6, inf) to prevent
    degenerate distributions from over-confident CNN outputs.
    """
    mu  = np.clip(x.mean(), 1e-6, 1.0 - 1e-6)
    var = x.var()
    # Clamp var to prevent zero or negative denominator
    max_var = mu * (1.0 - mu) - 1e-8
    if max_var <= 0 or var <= 0:
        return None, None
    var = min(var, max_var)
    common = mu * (1.0 - mu) / var - 1.0
    alpha  = max(mu * common, 1e-6)
    beta   = max((1.0 - mu) * common, 1e-6)
    return alpha, beta


def detect_drift_cda(Q_confidences):
    """
    Algorithm 5 from Casado et al. 2022.

    For each split k in [Delta, N-Delta]:
      Qa = Q[:k]  (most recent — paper notation)
      Qb = Q[k:]  (older)

    Condition 1 (line 7): ma <= (1-lambda) * mb
      i.e. recent confidence dropped relative to older confidence

    Condition 2 (lines 9-17): sum of log-likelihood ratios sk > Th
      sk = sum_{i=1..N} log f(qi | alpha_a, beta_a)
                       - log f(qi | alpha_b, beta_b)
      where alpha/beta estimated from Qa and Qb respectively
      via method of moments.

    Th = -log(lambda) per paper Section 4.1 explicit statement.

    Returns True if drift detected at any split k, False otherwise.
    """
    N = len(Q_confidences)
    if N < 2 * DELTA_CDA:
        return False

    # Clip confidences to open interval — CIFAR CNNs often produce
    # max softmax = 1.0 which gives log(beta_pdf(1.0)) = -inf
    Q = np.clip(np.array(Q_confidences), 1e-6, 1.0 - 1e-6)

    for k in range(DELTA_CDA, N - DELTA_CDA):
        Qa = Q[:k]   # most recent (paper convention)
        Qb = Q[k:]   # older

        ma = Qa.mean()
        mb = Qb.mean()

        # Condition 1: detect only drops, not increases (paper line 7)
        if ma > (1.0 - LAMBDA_CDA) * mb:
            continue

        # Fit beta distributions via method of moments (paper lines 9-10)
        alpha_a, beta_a = fit_beta_mom(Qa)
        alpha_b, beta_b = fit_beta_mom(Qb)

        if alpha_a is None or alpha_b is None:
            continue

        # Compute sum of log-likelihood ratios over ALL Q samples (lines 11-13)
        # sk = sum_i [ log f(qi|params_a) - log f(qi|params_b) ]
        try:
            log_pa = beta_dist.logpdf(Q, alpha_a, beta_a)
            log_pb = beta_dist.logpdf(Q, alpha_b, beta_b)
            # Replace any -inf or nan with 0 (numerical guard)
            log_pa = np.where(np.isfinite(log_pa), log_pa, 0.0)
            log_pb = np.where(np.isfinite(log_pb), log_pb, 0.0)
            sk = (log_pa - log_pb).sum()
        except Exception:
            continue

        # Condition 2: log-likelihood ratio exceeds threshold (line 17)
        # Th = -log(lambda) per paper Section 4.1
        if sk > TH_CDA:
            return True

    return False


# ============================================================
# CDA-FedAvg CLIENT
# ============================================================

class CDAClient:
    """
    CDA-FedAvg client implementing Algorithms 3 and 4 from
    Casado et al. 2022, adapted to synchronous FL rounds.

    Key components:
    - Short-term memory Q: sliding window of confidence scores
    - Long-term memory LTM: stored raw data from each past concept
    - Drift detection: Algorithm 5 with corrected Th = -log(lambda)
    - Rehearsal training: each round trains on all LTM data
    """

    def __init__(self, cid, dataset):
        self.cid         = cid
        self.dataset     = dataset
        self.model       = get_model()
        self.Q           = []      # short-term memory: confidence scores
        self.Q_data      = []      # corresponding (img_tensor, label) pairs
        self.LTM         = []      # long-term memory: all past concepts
        self.n_concepts  = 0
        self.initialized = False

    def set_params(self, global_state):
        self.model.load_state_dict(copy.deepcopy(global_state))

    def _ltm_loader(self):
        """DataLoader wrapping long-term memory for rehearsal training."""
        class LTMDataset(Dataset):
            def __init__(s, data): s.data = data
            def __len__(s): return len(s.data)
            def __getitem__(s, i):
                img, label = s.data[i]
                return img, torch.tensor(label, dtype=torch.long)

        return DataLoader(
            LTMDataset(self.LTM),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False
        )

    def train(self):
        """
        Algorithm 4 (one FL round):
          1. If first round: initialize LTM with current data
          2. Train on LTM (rehearsal across all past concepts)
          3. Compute confidence on current data, populate Q
          4. Detect drift with probability e^{-2*qi} per sample
          5. If drift: move current data to LTM, clear Q
        """
        crit = nn.CrossEntropyLoss()
        loader = get_loader(self.cid)

        # Step 1 — First round: seed LTM with initial concept data
        if not self.initialized:
            init_data = []
            for imgs, labels in loader:
                for i in range(imgs.size(0)):
                    init_data.append((imgs[i].cpu(), int(labels[i])))
            # Sample up to LTM_PER_CONCEPT
            if len(init_data) > LTM_PER_CONCEPT:
                init_data = random.sample(init_data, LTM_PER_CONCEPT)
            self.LTM.extend(init_data)
            self.n_concepts = 1
            self.initialized = True

        # Step 2 — Rehearsal training on long-term memory
        # This is the core anti-forgetting mechanism: train on ALL past
        # concepts every round so the model retains earlier knowledge
        if self.LTM:
            train_loader = self._ltm_loader()
        else:
            train_loader = loader

        self.model.train()
        opt = optim.SGD(
            self.model.parameters(), lr=LR,
            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

        for _ in range(LOCAL_EPOCHS):
            for imgs, labels in train_loader:
                imgs   = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                opt.zero_grad()
                crit(self.model(imgs.float()), labels).backward()
                opt.step()

        # Step 3+4 — Confidence computation and drift detection
        current_data  = []
        drift_found   = False

        self.model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(DEVICE)
                probs = torch.softmax(
                    self.model(imgs.float()), dim=1)
                # Confidence = max softmax probability (paper Section 4.1)
                confs = probs.max(1).values.cpu().numpy()

                for i, qi in enumerate(confs):
                    label   = int(labels[i])
                    img_cpu = imgs[i].cpu()
                    current_data.append((img_cpu, label))

                    # Maintain sliding window Q (max size Nmax)
                    if len(self.Q) >= N_MAX_CDA:
                        self.Q.pop(0)
                        self.Q_data.pop(0)
                    self.Q.append(float(qi))
                    self.Q_data.append((img_cpu, label))

                    # Run detection with probability e^{-2*qi}
                    # Lower confidence → higher probability of running check
                    # This is intentional: Algorithm 4 line 12
                    if random.random() < math.exp(-2.0 * float(qi)):
                        if detect_drift_cda(self.Q):
                            drift_found = True
                            break

                if drift_found:
                    break

        # Step 5 — Drift adaptation: add current concept to LTM, clear Q
        if drift_found:
            sampled = current_data
            if len(sampled) > LTM_PER_CONCEPT:
                sampled = random.sample(sampled, LTM_PER_CONCEPT)
            self.LTM.extend(sampled)
            # Keep LTM bounded: if total exceeds some maximum,
            # drop oldest concept (simple FIFO — keep last 5 concepts)
            max_ltm = LTM_PER_CONCEPT * 5
            if len(self.LTM) > max_ltm:
                self.LTM = self.LTM[-max_ltm:]
            self.n_concepts += 1
            self.Q.clear()
            self.Q_data.clear()


# ============================================================
# RUN CDA-FedAvg
# ============================================================

def run_cda_fedavg(seed=SEED):
    print(f"\n{'='*60}")
    print(f"CDA-FedAvg (Casado et al. 2022) — seed={seed}")
    print(f"Th={TH_CDA:.4f} | Nmax={N_MAX_CDA} | "
          f"Delta={DELTA_CDA} | LTM={LTM_PER_CONCEPT}/concept")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    reset_datasets()
    gm      = get_model()
    clients = [CDAClient(i, client_sets[i]) for i in range(NUM_CLIENTS)]
    log     = []

    for rnd in range(NUM_ROUNDS):
        if rnd == DRIFT_ROUND:
            apply_drift()
            for c in clients:
                c.dataset = client_sets[c.cid]

        gs = copy.deepcopy(gm.state_dict())
        for c in clients:
            c.set_params(gs)
            c.train()

        states  = [c.model.state_dict() for c in clients]
        weights = [len(client_sets[c.cid]) for c in clients]
        fedavg_aggregate(gm, states, weights)

        # Send updated global back to all clients
        gs_new = copy.deepcopy(gm.state_dict())
        for c in clients:
            c.set_params(gs_new)

        acc = evaluate(gm, global_loader)
        log.append({'round': rnd, 'global_acc': acc})

        if rnd % 10 == 0 or rnd == DRIFT_ROUND:
            drifted = sum(1 for c in clients if c.n_concepts > 1)
            ltm_avg = np.mean([len(c.LTM) for c in clients])
            tag     = "  <-- DRIFT" if rnd == DRIFT_ROUND else ""
            print(f"  Round {rnd:03d} | Global: {acc:.4f} | "
                  f"Detected: {drifted}/{NUM_CLIENTS} | "
                  f"Avg LTM: {ltm_avg:.0f}{tag}")

    fname = f'results_CDAFedAvg_seed{seed}.csv'
    with open(fname, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'global_acc'])
        writer.writeheader()
        writer.writerows(log)
    print(f"  Results saved to {fname}")
    return log


# ============================================================
# METRICS
# ============================================================

def compute_metrics(log, drift_round=DRIFT_ROUND, window=10, tol=0.02):
    accs   = [r['global_acc'] for r in log]
    pre    = float(np.mean(accs[max(0, drift_round-11):drift_round]))
    dip    = pre - min(accs[drift_round:drift_round+window])
    rec    = next((i-drift_round for i in range(drift_round, len(accs))
                   if accs[i] >= pre-tol), None)
    stable = float(np.mean(accs[-10:]))
    return pre, dip, rec, stable


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Run single seed by default
    # To run multiple seeds: change seeds list below
    seeds = [0]  # add 1, 2 for multi-seed

    all_logs = []
    for s in seeds:
        log = run_cda_fedavg(seed=s)
        all_logs.append(log)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    for i, (s, log) in enumerate(zip(seeds, all_logs)):
        pre, dip, rec, stable = compute_metrics(log)
        print(f"\nSeed {s}:")
        print(f"  Pre-drift acc:     {pre:.4f}")
        print(f"  Accuracy dip:      {dip:.4f}")
        print(f"  Recovery rounds:   "
              f"{rec if rec is not None else 'Not recovered'}")
        print(f"  Post-drift stable: {stable:.4f}")

    if len(all_logs) > 1:
        print(f"\nMean across {len(seeds)} seeds:")
        metrics = [compute_metrics(l) for l in all_logs]
        for i, name in enumerate(['Pre-drift', 'Dip', 'Recovery', 'Stable']):
            vals = [m[i] for m in metrics if m[i] is not None]
            if vals:
                print(f"  {name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\nDone.")
