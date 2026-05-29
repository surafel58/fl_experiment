"""
class_frequency_check.py — verify the sparse-Dirichlet hypothesis.

Question: are Group B / Group C clients holding so few examples of their
event-1 swap-pair classes that the swap is nearly a no-op for them?

For seed=0's Dirichlet(alpha=0.1) partition, this computes:
  - Per drift group, total samples in the EVENT-0 swap-pair classes
  - Per drift group, total samples in the EVENT-1 swap-pair classes
  - Per-client breakdowns

If event-1's swap-pair has dramatically fewer samples in some group,
that's the sparse-Dirichlet smoking gun.

Uses identical partition logic to all_experiments_optimized.py.
"""
import random
import numpy as np
import torchvision

NUM_CLIENTS = 20
ALPHA_DIR   = 0.1
SEED        = 0

GROUP_A = [i for i in range(NUM_CLIENTS) if i % 10 < 3]
GROUP_B = [i for i in range(NUM_CLIENTS) if 3 <= i % 10 < 6]
GROUP_C = [i for i in range(NUM_CLIENTS) if i % 10 >= 6]

DRIFT_EVENTS = [
    {'A': (1, 2), 'B': (3, 4), 'C': (5, 6)},  # event 0
    {'A': (3, 4), 'B': (5, 6), 'C': (7, 8)},  # event 1
]


def partition_dataset(dataset, n, alpha, seed):
    labels = np.array(dataset.targets)
    nc = len(set(labels))
    cidx = [[] for _ in range(n)]
    for k in range(nc):
        ci = np.where(labels == k)[0]
        es = np.array_split(ci[:n*5], n)
        cidx = [c + e.tolist() for c, e in zip(cidx, es)]
        rem = ci[n*5:]
        random.seed(seed + k); np.random.seed(seed + k)
        props = np.random.dirichlet(np.repeat(alpha, n))
        splits = (np.cumsum(props/props.sum()) * len(rem)).astype(int)[:-1]
        cidx = [c + ch.tolist() for c, ch in zip(cidx, np.split(rem, splits))]
    for c in cidx: random.shuffle(c)
    return cidx


print("Loading CIFAR-10...")
raw_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
labels = np.array(raw_train.targets)

print("Partitioning (seed=0)...")
random.seed(SEED); np.random.seed(SEED)
train_idx = partition_dataset(raw_train, NUM_CLIENTS, ALPHA_DIR, SEED)

# Per-client class counts
client_class_counts = {}
for cid, idx_list in enumerate(train_idx):
    cnt = np.bincount(labels[idx_list], minlength=10)
    client_class_counts[cid] = cnt

print()
print("Per-client class distribution (rows = client, cols = class 0..9):")
print(f"  {'cid':<4}{'grp':<5}{'total':<7}" + ''.join(f'  c{i}'.ljust(6) for i in range(10)))
print("  " + "-"*82)
for cid in range(NUM_CLIENTS):
    grp = 'A' if cid in GROUP_A else ('B' if cid in GROUP_B else 'C')
    cnt = client_class_counts[cid]
    total = int(cnt.sum())
    print(f"  {cid:<4}{grp:<5}{total:<7}" + ''.join(f'{int(c):>5d} ' for c in cnt))

print()
print("=" * 80)
print("PER-GROUP TOTALS, FOCUS ON EVENT-0 vs EVENT-1 SWAP PAIRS")
print("=" * 80)

for grp_label, grp_clients in [('A', GROUP_A), ('B', GROUP_B), ('C', GROUP_C)]:
    e0_pair = DRIFT_EVENTS[0][grp_label]
    e1_pair = DRIFT_EVENTS[1][grp_label]
    total = sum(client_class_counts[c].sum() for c in grp_clients)
    e0_count = sum(client_class_counts[c][e0_pair[0]] + client_class_counts[c][e0_pair[1]]
                   for c in grp_clients)
    e1_count = sum(client_class_counts[c][e1_pair[0]] + client_class_counts[c][e1_pair[1]]
                   for c in grp_clients)
    e0_frac = e0_count / total
    e1_frac = e1_count / total

    print(f"\nGroup {grp_label} (clients {grp_clients}, total samples {total}):")
    print(f"  Event 0 swap pair classes {e0_pair}: {e0_count} samples "
          f"({e0_frac*100:.1f}% of group's data)")
    print(f"  Event 1 swap pair classes {e1_pair}: {e1_count} samples "
          f"({e1_frac*100:.1f}% of group's data)")
    ratio = e1_count / e0_count if e0_count > 0 else float('inf')
    verdict = "comparable" if 0.7 <= ratio <= 1.4 else ("event-1 STARVED" if ratio < 0.7 else "event-1 OVER-served")
    print(f"  Event-1 / Event-0 samples ratio: {ratio:.2f}  ({verdict})")

    # Per-client event-0 vs event-1 sample counts
    print(f"  Per-client breakdown for Group {grp_label}:")
    print(f"    {'cid':<4}{'e0_pair':<10}{'e1_pair':<10}{'e0_count':>10}{'e1_count':>10}")
    for cid in grp_clients:
        cnt = client_class_counts[cid]
        e0c = int(cnt[e0_pair[0]] + cnt[e0_pair[1]])
        e1c = int(cnt[e1_pair[0]] + cnt[e1_pair[1]])
        print(f"    {cid:<4}{str(e0_pair):<10}{str(e1_pair):<10}{e0c:>10d}{e1c:>10d}")
