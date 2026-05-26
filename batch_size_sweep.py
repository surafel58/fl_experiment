# ============================================================
# Batch-size sweep — find the throughput sweet spot on L4
#
# The profile run showed GPU at 3% utilization with batch=64.
# Hypothesis: kernel-launch + Python overhead dominates, and
# larger batches amortize that fixed cost.
#
# Sweep: 64, 128, 256, 512, 1024 — measured on the GPU-resident
# config (the winning config from profile_bottleneck.py).
#
# Per batch size, runs 3 rounds of FedAvg-style: 20 clients,
# 5 local epochs each. CUDA-synced wall time. Also reports
# steps/sec and final accuracy (sanity check).
#
# IMPORTANT — this run is about WALL CLOCK, not accuracy.
# Augmentation is skipped. Accuracy numbers are for sanity only.
#
# Output: batch_sweep_<ts>.txt
# Run:    python3 batch_size_sweep.py
# ============================================================

import torch, torch.nn as nn, torch.optim as optim
import torchvision
import numpy as np, copy, random, time, os
from datetime import datetime
from contextlib import contextmanager

# --------- config ---------
NUM_CLIENTS  = 20
SWEEP_ROUNDS = 3
LOCAL_EPOCHS = 5
LR, MOM, WD  = 0.01, 0.9, 1e-5
ALPHA_DIR    = 0.1
SEED         = 0
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZES  = [64, 128, 256, 512, 1024]

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# --------- timer ---------
@contextmanager
def cuda_timed():
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0
    if DEVICE.type == 'cuda': torch.cuda.synchronize()

# --------- data load + partition (same as profile harness) ---------
print("Loading CIFAR-10...")
raw_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
raw_test  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

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
print("Pre-loading GPU-resident tensors...")
MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=DEVICE).view(1,3,1,1)
STD  = torch.tensor([0.2023, 0.1994, 0.2010], device=DEVICE).view(1,3,1,1)

def to_gpu(base, indices):
    arr = base.data[indices].astype(np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(0,3,1,2).to(DEVICE)
    t   = (t - MEAN) / STD
    y   = torch.tensor(np.array(base.targets)[indices], dtype=torch.long, device=DEVICE)
    return t, y

gpu_clients   = {i: to_gpu(raw_train, train_idx[i]) for i in range(NUM_CLIENTS)}
test_x, test_y = to_gpu(raw_test, list(range(len(raw_test))))
client_sizes  = [len(train_idx[i]) for i in range(NUM_CLIENTS)]
print(f"Client sample counts: min={min(client_sizes)} max={max(client_sizes)} mean={np.mean(client_sizes):.0f}")

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
    def forward(self, x): return self.fc(self.hidden_layers(x))

def new_model(): return CifarCNN().to(DEVICE)

# --------- training (GPU-resident, no augmentation) ---------
def train_client(model, x, y, epochs, batch_size):
    model.train()
    opt  = optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD)
    crit = nn.CrossEntropyLoss()
    n = x.size(0)
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        # drop_last semantics — match original code
        for s in range(0, n - batch_size + 1, batch_size):
            idx = perm[s:s+batch_size]
            opt.zero_grad()
            crit(model(x[idx]), y[idx]).backward()
            opt.step()
            steps += 1
    return steps

def fedavg(gm, states, weights):
    gs  = gm.state_dict()
    total = sum(weights)
    new = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in gs.items()}
    for st, w in zip(states, weights):
        for k in gs:
            new[k] += (w/total) * st[k].float()
    gm.load_state_dict(new)

def eval_model(model):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for s in range(0, test_x.size(0), 512):
            out = model(test_x[s:s+512])
            correct += (out.argmax(1) == test_y[s:s+512]).sum().item()
            total   += min(512, test_x.size(0) - s)
    return correct / total

# --------- one sweep iteration ---------
def run_bs(bs):
    # reset model + seed for reproducibility
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    gm = new_model()
    total_steps = 0
    last_acc = 0.0
    with cuda_timed() as elapsed:
        for rnd in range(SWEEP_ROUNDS):
            states, weights = [], []
            for cid in range(NUM_CLIENTS):
                lm = new_model()
                lm.load_state_dict({k: v.clone() for k, v in gm.state_dict().items()})
                x, y = gpu_clients[cid]
                # skip if client has fewer samples than batch_size
                if x.size(0) < bs:
                    continue
                steps = train_client(lm, x, y, LOCAL_EPOCHS, bs)
                total_steps += steps
                states.append(lm.state_dict())
                weights.append(len(x))
            fedavg(gm, states, weights)
            last_acc = eval_model(gm)
        sec = elapsed()
    return sec, total_steps, last_acc, len(weights)

# --------- main ---------
def main():
    print(f"\nDevice: {DEVICE} | CPU cores: {os.cpu_count()}")
    print(f"Sweep: {BATCH_SIZES}")
    print(f"Per BS: {SWEEP_ROUNDS} rounds * {NUM_CLIENTS} clients * {LOCAL_EPOCHS} local epochs\n")

    results = []
    for bs in BATCH_SIZES:
        print(f"--- batch_size = {bs} ---")
        sec, steps, acc, n_clients = run_bs(bs)
        sps  = steps / sec if sec > 0 else 0
        results.append({
            'bs': bs, 'sec': sec, 'steps': steps,
            'steps_per_sec': sps, 'acc': acc, 'n_clients': n_clients
        })
        print(f"  time={sec:.2f}s  steps={steps}  steps/s={sps:.1f}  "
              f"acc={acc:.3f}  clients_used={n_clients}/{NUM_CLIENTS}")
        print()

    # --------- report ---------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"batch_sweep_{ts}.txt"
    lines = []
    lines.append(f"Batch-size sweep @ {ts}  device={DEVICE}")
    lines.append(f"  {SWEEP_ROUNDS} rounds * {NUM_CLIENTS} clients * {LOCAL_EPOCHS} local epochs each")
    lines.append("")
    header = f"{'BS':>6}{'time(s)':>10}{'steps':>10}{'steps/s':>12}{'acc':>8}{'clients':>10}"
    lines.append(header); lines.append("-"*len(header))
    base_time = results[0]['sec']
    for r in results:
        lines.append(f"{r['bs']:>6}{r['sec']:>10.2f}{r['steps']:>10}"
                     f"{r['steps_per_sec']:>12.1f}{r['acc']:>8.3f}{r['n_clients']:>10}")
    lines.append("")
    lines.append("Speedup vs BS=64:")
    for r in results:
        lines.append(f"  BS={r['bs']:>4}  {base_time/r['sec']:>5.2f}x  ({r['sec']:.2f}s)")

    report = "\n".join(lines)
    print("\n" + report)
    with open(path, 'w') as f: f.write(report)
    print(f"\nReport saved: {path}")

if __name__ == '__main__':
    main()
