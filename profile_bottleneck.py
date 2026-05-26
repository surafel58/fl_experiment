# ============================================================
# Profiling harness — finds the REAL bottleneck in FedAvg
#
# Runs 3 rounds of FedAvg with fine-grained per-section timing
# under 4 configurations to compare:
#   A) baseline           — num_workers=0, baseline DataLoader
#   B) current "optimized" — num_workers=4, pin, persistent
#   C) gpu-resident       — pre-loaded GPU tensors, num_workers=0
#   D) gpu-resident + AMP — same as C plus mixed precision
#
# Per-section timers (all CUDA-synchronized):
#   - per_round_loader_build  — DataLoader instantiation cost
#   - per_round_state_copy    — copy.deepcopy(state_dict) cost
#   - per_round_compute       — actual fwd+bwd+step
#   - per_round_h2d           — host->device transfer cost
#   - per_round_aggregate     — fedavg averaging cost
#   - per_round_eval          — global eval cost
#
# Output: profile_report_<ts>.txt with a side-by-side table.
#
# Run:  python profile_bottleneck.py
# ============================================================

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision, torchvision.transforms as transforms
import numpy as np, copy, random, time, os, sys
from datetime import datetime
from contextlib import contextmanager

# --------- config (mirrors all_experiments_optimized.py) ---------
NUM_CLIENTS  = 20
PROFILE_ROUNDS = 3        # enough to amortize warmup
LOCAL_EPOCHS = 5
BATCH_SIZE   = 64
LR, MOM, WD  = 0.01, 0.9, 1e-5
ALPHA_DIR    = 0.1
SEED         = 0
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS  = min(4, os.cpu_count() or 1)

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# --------- CUDA-synced timer ---------
@contextmanager
def timed(bucket, key):
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    bucket[key] = bucket.get(key, 0.0) + (time.perf_counter() - t0)

# --------- data ---------
print("Loading CIFAR-10...")
raw_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
raw_test  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class ClientDataset(Dataset):
    def __init__(self, base, indices, transform=None):
        self.data, self.targets = base.data[indices], list(np.array(base.targets)[indices])
        self.transform = transform
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        img, label = self.data[idx], int(self.targets[idx])
        if self.transform: img = self.transform(img)
        return img, label

def partition(dataset, n, alpha, seed):
    labels = np.array(dataset.targets); nc = len(set(labels))
    cidx = [[] for _ in range(n)]
    for k in range(nc):
        ci = np.where(labels == k)[0]
        es = np.array_split(ci[:n*5], n)
        cidx = [c + e.tolist() for c, e in zip(cidx, es)]
        rem  = ci[n*5:]
        random.seed(seed + k); np.random.seed(seed + k)
        props  = np.random.dirichlet(np.repeat(alpha, n))
        splits = (np.cumsum(props/props.sum()) * len(rem)).astype(int)[:-1]
        cidx   = [c + ch.tolist() for c, ch in zip(cidx, np.split(rem, splits))]
    for c in cidx: random.shuffle(c)
    return cidx

print("Partitioning...")
train_idx = partition(raw_train, NUM_CLIENTS, ALPHA_DIR, SEED)
client_sets = {i: ClientDataset(raw_train, train_idx[i], train_transform)
               for i in range(NUM_CLIENTS)}
global_test = ClientDataset(raw_test, list(range(len(raw_test))), test_transform)

# --------- pre-build GPU-resident client tensors ---------
# For configs C and D: pre-normalize and keep on GPU. We skip the random
# augmentations to keep the comparison clean — math differs slightly, but
# this experiment is about wall-clock not accuracy.
print("Pre-loading GPU-resident tensors...")
MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=DEVICE).view(1,3,1,1)
STD  = torch.tensor([0.2023, 0.1994, 0.2010], device=DEVICE).view(1,3,1,1)

def to_gpu_tensors(base, indices):
    # raw is HWC uint8 [N,32,32,3] -> NCHW float [N,3,32,32], normalized, on GPU
    arr = base.data[indices].astype(np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(0,3,1,2).to(DEVICE)
    t   = (t - MEAN) / STD
    y   = torch.tensor(np.array(base.targets)[indices], dtype=torch.long, device=DEVICE)
    return t, y

gpu_client_data = {i: to_gpu_tensors(raw_train, train_idx[i]) for i in range(NUM_CLIENTS)}
gpu_test_x, gpu_test_y = to_gpu_tensors(raw_test, list(range(len(raw_test))))

# --------- model ---------
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
    def forward(self, x): return self.fc(self.hidden_layers(x.float()))

def new_model(): return CifarCNN().to(DEVICE)

# --------- training inner loops ---------
def train_dataloader(model, loader, epochs, bucket):
    model.train()
    opt  = optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for imgs, labels in loader:
            with timed(bucket, 'h2d'):
                imgs   = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
            with timed(bucket, 'compute'):
                opt.zero_grad()
                crit(model(imgs), labels).backward()
                opt.step()

def train_gpu_resident(model, x, y, epochs, bucket, amp=False):
    model.train()
    opt  = optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD)
    crit = nn.CrossEntropyLoss()
    n = x.size(0)
    scaler = torch.amp.GradScaler('cuda') if amp else None
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n - BATCH_SIZE + 1, BATCH_SIZE):
            idx = perm[s:s+BATCH_SIZE]
            with timed(bucket, 'compute'):
                opt.zero_grad()
                if amp:
                    with torch.amp.autocast('cuda'):
                        loss = crit(model(x[idx]), y[idx])
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
                else:
                    crit(model(x[idx]), y[idx]).backward()
                    opt.step()

# --------- fedavg aggregation ---------
def fedavg_deepcopy(gm, states, weights, bucket):
    with timed(bucket, 'aggregate'):
        gs  = gm.state_dict()
        new = copy.deepcopy(gs)
        total = sum(weights)
        for key in gs:
            new[key] = torch.zeros_like(gs[key], dtype=torch.float32)
            for state, w in zip(states, weights):
                new[key] += (w / total) * state[key].float()
        gm.load_state_dict(new)

# --------- eval ---------
def eval_dataloader(model, bucket):
    with timed(bucket, 'eval'):
        loader = DataLoader(global_test, batch_size=512, shuffle=False)
        model.eval(); correct = total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        return correct / total

def eval_gpu_resident(model, bucket):
    with timed(bucket, 'eval'):
        model.eval(); correct = total = 0
        with torch.no_grad():
            for s in range(0, gpu_test_x.size(0), 512):
                out = model(gpu_test_x[s:s+512])
                correct += (out.argmax(1) == gpu_test_y[s:s+512]).sum().item()
                total   += min(512, gpu_test_x.size(0) - s)
        return correct / total

# --------- config runners ---------
def run_config(name, *, use_workers, use_gpu_resident, use_amp):
    print(f"\n{'='*60}\nCONFIG: {name}\n{'='*60}")
    bucket = {}
    gm = new_model()
    for rnd in range(PROFILE_ROUNDS):
        states, weights = [], []
        for cid in range(NUM_CLIENTS):
            with timed(bucket, 'state_copy'):
                lm = new_model()
                lm.load_state_dict(copy.deepcopy(gm.state_dict()))
            if use_gpu_resident:
                x, y = gpu_client_data[cid]
                train_gpu_resident(lm, x, y, LOCAL_EPOCHS, bucket, amp=use_amp)
            else:
                with timed(bucket, 'loader_build'):
                    loader = DataLoader(
                        client_sets[cid], batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True,
                        num_workers=(NUM_WORKERS if use_workers else 0),
                        pin_memory=use_workers,
                        persistent_workers=(use_workers and NUM_WORKERS > 0)
                    )
                train_dataloader(lm, loader, LOCAL_EPOCHS, bucket)
                # explicit cleanup so worker shutdown counts
                del loader
            states.append(lm.state_dict())
            weights.append(len(client_sets[cid]))
        fedavg_deepcopy(gm, states, weights, bucket)
        if use_gpu_resident:
            acc = eval_gpu_resident(gm, bucket)
        else:
            acc = eval_dataloader(gm, bucket)
        print(f"  Round {rnd} | acc={acc:.4f}")
    return bucket

# --------- entry point ---------
def main():
    print(f"Device: {DEVICE} | CPU cores: {os.cpu_count()} | workers: {NUM_WORKERS}")
    print(f"Rounds: {PROFILE_ROUNDS} | clients: {NUM_CLIENTS} | epochs: {LOCAL_EPOCHS} | batch: {BATCH_SIZE}")

    results = {}
    results['A_baseline']        = run_config('A: num_workers=0',     use_workers=False, use_gpu_resident=False, use_amp=False)
    results['B_current_optim']   = run_config('B: current "optimized"',use_workers=True,  use_gpu_resident=False, use_amp=False)
    results['C_gpu_resident']    = run_config('C: GPU-resident',       use_workers=False, use_gpu_resident=True,  use_amp=False)
    results['D_gpu_resident_amp']= run_config('D: GPU-resident + AMP', use_workers=False, use_gpu_resident=True,  use_amp=True)

    # --------- report ---------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"profile_report_{ts}.txt"
    keys = ['loader_build', 'state_copy', 'h2d', 'compute', 'aggregate', 'eval']
    cols = list(results.keys())

    lines = []
    lines.append(f"Profile run @ {ts} | device={DEVICE} | rounds={PROFILE_ROUNDS}\n")
    header = f"{'section':<16}" + "".join(f"{c:>22}" for c in cols)
    lines.append(header); lines.append("-"*len(header))
    for k in keys:
        row = f"{k:<16}" + "".join(f"{results[c].get(k, 0.0):>22.2f}" for c in cols)
        lines.append(row)
    lines.append("-"*len(header))
    totals = {c: sum(results[c].values()) for c in cols}
    lines.append(f"{'TOTAL (s)':<16}" + "".join(f"{totals[c]:>22.2f}" for c in cols))
    lines.append("")
    base = totals['A_baseline']
    lines.append("Speedup vs A_baseline:")
    for c in cols:
        lines.append(f"  {c:<24} {base/totals[c]:>5.2f}x  (total {totals[c]:.1f}s)")

    report = "\n".join(lines)
    print("\n" + report)
    with open(report_path, 'w') as f: f.write(report)
    print(f"\nReport saved to {report_path}")

if __name__ == '__main__':
    main()
