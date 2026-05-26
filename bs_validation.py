# ============================================================
# Batch-size validation — does BS=128 match BS=64 convergence?
#
# Runs FedAvg for 40 rounds under 3 configs and saves
# per-round accuracy curves. Same seed, same GPU-resident
# pipeline, same GPU-side augmentation across all configs —
# the ONLY thing varying is batch size and LR.
#
#   X1: BS=64,  LR=0.01,  drop_last=True   (matches original code)
#   X2: BS=128, LR=0.01,  drop_last=False  (naive 2x)
#   X3: BS=128, LR=0.014, drop_last=False  (sqrt(2)-scaled LR)
#
# Pass criteria: X2 or X3 final accuracy within ~2% of X1 AND
# the curves stay visually close throughout. If both lag X1,
# we revert to BS=64.
#
# Output:
#   bs_validation_<ts>.csv   — per-round acc for all 3 configs
#   bs_validation_<ts>.txt   — summary table
# ============================================================

import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import torchvision
import numpy as np, random, copy, os, csv, time
from datetime import datetime

# --------- config ---------
NUM_CLIENTS  = 20
ROUNDS       = 40
LOCAL_EPOCHS = 5
MOM, WD      = 0.9, 1e-5
ALPHA_DIR    = 0.1
SEED         = 0
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configs: (label, batch_size, lr, drop_last)
CONFIGS = [
    ('X1_bs64',         64,  0.01,  True),
    ('X2_bs128_lr01',   128, 0.01,  False),
    ('X3_bs128_lrsq2',  128, 0.014, False),
]

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# --------- data: load + partition + GPU-resident tensors ---------
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

print("Partitioning + loading to GPU...")
train_idx = partition(raw_train, NUM_CLIENTS, ALPHA_DIR, SEED)
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
sizes = [len(train_idx[i]) for i in range(NUM_CLIENTS)]
print(f"Clients: {NUM_CLIENTS}  samples min/max/mean: {min(sizes)}/{max(sizes)}/{int(np.mean(sizes))}")

# --------- GPU-side augmentation (mirrors torchvision behavior) ---------
def gpu_augment(x, pad=4, crop_size=32):
    # Random horizontal flip per sample
    flip = torch.rand(x.size(0), device=x.device) < 0.5
    if flip.any():
        x = torch.where(flip[:, None, None, None], torch.flip(x, dims=[3]), x)
    # Random crop with zero padding (matches torchvision default for RandomCrop)
    x = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
    n, c, h, w = x.shape
    max_off = h - crop_size  # = 2*pad
    h_off = torch.randint(0, max_off + 1, (n,), device=x.device)
    w_off = torch.randint(0, max_off + 1, (n,), device=x.device)
    rows = h_off[:, None] + torch.arange(crop_size, device=x.device)[None, :]   # [n, 32]
    cols = w_off[:, None] + torch.arange(crop_size, device=x.device)[None, :]   # [n, 32]
    batch_idx = torch.arange(n, device=x.device)[:, None, None, None]           # [n,1,1,1]
    chan_idx  = torch.arange(c, device=x.device)[None, :, None, None]           # [1,c,1,1]
    row_idx   = rows[:, None, :, None].expand(n, c, crop_size, crop_size)
    col_idx   = cols[:, None, None, :].expand(n, c, crop_size, crop_size)
    return x[batch_idx, chan_idx, row_idx, col_idx]

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

def new_model():
    # deterministic init per config so the only variable is batch size + lr
    torch.manual_seed(SEED)
    return CifarCNN().to(DEVICE)

# --------- training ---------
def train_client(model, x, y, epochs, bs, lr, drop_last):
    model.train()
    opt  = optim.SGD(model.parameters(), lr=lr, momentum=MOM, weight_decay=WD)
    crit = nn.CrossEntropyLoss()
    n = x.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        # drop_last=True  -> only full batches
        # drop_last=False -> last (possibly partial) batch is used
        step_end = (n // bs) * bs if drop_last else n
        for s in range(0, step_end, bs):
            idx = perm[s:s+bs]
            if idx.numel() == 0: continue
            xb = gpu_augment(x[idx])
            opt.zero_grad()
            crit(model(xb), y[idx]).backward()
            opt.step()

def fedavg(gm, states, weights):
    gs = gm.state_dict()
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
            total   += test_y[s:s+512].size(0)
    return correct / total

# --------- one config run ---------
def run_config(label, bs, lr, drop_last):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    gm = new_model()
    accs = []
    t0 = time.perf_counter()
    for rnd in range(ROUNDS):
        states, weights = [], []
        for cid in range(NUM_CLIENTS):
            x, y = gpu_clients[cid]
            # If drop_last and client has < bs samples, that client contributes 0 batches.
            # Mirrors the original FedCCFA behavior exactly.
            if drop_last and x.size(0) < bs:
                continue
            lm = new_model()
            lm.load_state_dict({k: v.clone() for k, v in gm.state_dict().items()})
            train_client(lm, x, y, LOCAL_EPOCHS, bs, lr, drop_last)
            states.append(lm.state_dict())
            weights.append(x.size(0))
        fedavg(gm, states, weights)
        acc = eval_model(gm)
        accs.append(acc)
        if rnd % 5 == 0 or rnd == ROUNDS - 1:
            print(f"  [{label}] round {rnd:02d} | acc={acc:.4f}  clients={len(weights)}")
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] total {elapsed:.1f}s  final acc={accs[-1]:.4f}")
    return accs, elapsed

# --------- main ---------
def main():
    print(f"\nDevice: {DEVICE} | rounds: {ROUNDS} | clients: {NUM_CLIENTS}")
    print("Configs:")
    for c in CONFIGS: print(f"  {c[0]}: bs={c[1]} lr={c[2]} drop_last={c[3]}")
    print()

    all_accs, timings = {}, {}
    for label, bs, lr, drop_last in CONFIGS:
        print(f"\n=== {label} (bs={bs}, lr={lr}, drop_last={drop_last}) ===")
        accs, sec = run_config(label, bs, lr, drop_last)
        all_accs[label] = accs
        timings[label]  = sec

    # --------- save CSV (one row per round, one col per config) ---------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"bs_validation_{ts}.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['round'] + [c[0] for c in CONFIGS])
        for r in range(ROUNDS):
            w.writerow([r] + [f"{all_accs[c[0]][r]:.4f}" for c in CONFIGS])
    print(f"\nPer-round CSV: {csv_path}")

    # --------- summary table ---------
    txt_path = f"bs_validation_{ts}.txt"
    lines = []
    lines.append(f"BS validation @ {ts}  device={DEVICE}")
    lines.append(f"  rounds={ROUNDS}  clients={NUM_CLIENTS}  local_epochs={LOCAL_EPOCHS}")
    lines.append("")
    header = f"{'config':<22}{'final acc':>12}{'mean last 5':>14}{'wall (s)':>12}"
    lines.append(header); lines.append("-"*len(header))
    base_final = all_accs[CONFIGS[0][0]][-1]
    for c in CONFIGS:
        a = all_accs[c[0]]
        final = a[-1]
        last5 = sum(a[-5:]) / 5
        lines.append(f"{c[0]:<22}{final:>12.4f}{last5:>14.4f}{timings[c[0]]:>12.1f}")
    lines.append("")
    lines.append(f"Deltas vs {CONFIGS[0][0]} (last-5 mean):")
    base_l5 = sum(all_accs[CONFIGS[0][0]][-5:])/5
    for c in CONFIGS[1:]:
        l5 = sum(all_accs[c[0]][-5:])/5
        delta = l5 - base_l5
        lines.append(f"  {c[0]:<22} {delta:+.4f}  ({'PASS' if abs(delta) < 0.02 else 'FAIL'} at 2% threshold)")
    report = "\n".join(lines)
    print("\n" + report)
    with open(txt_path, 'w') as f: f.write(report)
    print(f"\nSummary: {txt_path}")

if __name__ == '__main__':
    main()
