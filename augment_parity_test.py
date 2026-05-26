"""
augment_parity_test.py — diagnose whether gpu_augment matches torchvision
RandomCrop(32, padding=4) + HorizontalFlip(p=0.5).

The suspected bug: gpu_augment pads with constant 0 AFTER normalization,
which corresponds to raw pixel ~mean-gray. Torchvision pads with 0 in
the raw uint8 PIL image BEFORE normalization, i.e. true black -> after
normalization that becomes -MEAN/STD per channel (~-2.4).

We run both pipelines on the same CIFAR-10 images N times and compare
per-channel mean and std of the augmented batches. If they differ
meaningfully (much more than sampling noise), the bug is real.

Then we test a *fixed* gpu_augment that pads with -MEAN/STD per channel,
and see if it reproduces torchvision's stats.

Run locally on Windows (CPU is fine; no GPU needed for this test):
  python augment_parity_test.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

DEVICE     = torch.device('cpu')          # test is fast enough on CPU
N_TRIALS   = 5000                          # augmentations per image
N_IMAGES   = 32                            # number of CIFAR-10 images sampled
SEED       = 0
torch.manual_seed(SEED); np.random.seed(SEED)

MEAN_T = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD_T  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

# --------- load a small CIFAR-10 sample ---------
print("Loading CIFAR-10 (train set head)...")
raw = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
imgs_uint8 = raw.data[:N_IMAGES]  # [N, 32, 32, 3] uint8
print(f"Sample {N_IMAGES} images, running {N_TRIALS} augmentations each")


# --------- torchvision pipeline (reference) ---------
tv_aug = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def run_torchvision(images_np, trials):
    """Apply torchvision aug to each image `trials` times, stack all outputs."""
    out = []
    for img in images_np:
        for _ in range(trials):
            out.append(tv_aug(img))
    # [N*trials, 3, 32, 32]
    return torch.stack(out)


# --------- GPU-style augmentations (current and fixed) ---------
def to_normalized_tensor_batch(images_np):
    """uint8 HWC batch -> normalized fp32 NCHW batch (no augmentation)."""
    arr = images_np.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)  # [N, 3, 32, 32]
    return (t - MEAN_T) / STD_T


def gpu_augment_v1_zero_in_norm(x, pad=4, crop_size=32):
    """CURRENT implementation: F.pad(value=0) in normalized space (= mean-gray)."""
    n = x.size(0)
    flip = torch.rand(n) < 0.5
    if flip.any():
        x = torch.where(flip[:, None, None, None], torch.flip(x, dims=[3]), x)
    x = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
    _, c, h, _ = x.shape
    max_off = h - crop_size
    h_off = torch.randint(0, max_off + 1, (n,))
    w_off = torch.randint(0, max_off + 1, (n,))
    rows = h_off[:, None] + torch.arange(crop_size)[None, :]
    cols = w_off[:, None] + torch.arange(crop_size)[None, :]
    bi = torch.arange(n)[:, None, None, None]
    ci = torch.arange(c)[None, :, None, None]
    ri = rows[:, None, :, None].expand(n, c, crop_size, crop_size)
    cj = cols[:, None, None, :].expand(n, c, crop_size, crop_size)
    return x[bi, ci, ri, cj]


def gpu_augment_v2_black_pad(x, pad=4, crop_size=32):
    """FIXED: pad with -MEAN/STD per channel, i.e. normalized 'true black'."""
    n = x.size(0)
    flip = torch.rand(n) < 0.5
    if flip.any():
        x = torch.where(flip[:, None, None, None], torch.flip(x, dims=[3]), x)
    # build padded tensor with per-channel "black" fill
    _, c, h, w = x.shape
    ph, pw = h + 2 * pad, w + 2 * pad
    pad_val = (-MEAN_T / STD_T).expand(n, c, ph, pw).clone()
    pad_val[:, :, pad:pad + h, pad:pad + w] = x
    x = pad_val
    _, c, h, _ = x.shape
    max_off = h - crop_size
    h_off = torch.randint(0, max_off + 1, (n,))
    w_off = torch.randint(0, max_off + 1, (n,))
    rows = h_off[:, None] + torch.arange(crop_size)[None, :]
    cols = w_off[:, None] + torch.arange(crop_size)[None, :]
    bi = torch.arange(n)[:, None, None, None]
    ci = torch.arange(c)[None, :, None, None]
    ri = rows[:, None, :, None].expand(n, c, crop_size, crop_size)
    cj = cols[:, None, None, :].expand(n, c, crop_size, crop_size)
    return x[bi, ci, ri, cj]


def run_gpu_aug(aug_fn, images_np, trials):
    """Apply gpu_augment-style fn to each image `trials` times, stack."""
    base = to_normalized_tensor_batch(images_np)  # [N, 3, 32, 32]
    n = base.size(0)
    # Replicate the base batch trials times and shuffle order to match torchvision's per-call randomness
    big = base.unsqueeze(0).expand(trials, n, 3, 32, 32).reshape(trials * n, 3, 32, 32)
    return aug_fn(big)


# --------- stats helpers ---------
def per_channel_stats(t):
    """Return dict of mean, std per channel + global pixel min/max."""
    means = t.mean(dim=(0, 2, 3)).tolist()
    stds  = t.std(dim=(0, 2, 3)).tolist()
    return {
        'mean_R': means[0], 'mean_G': means[1], 'mean_B': means[2],
        'std_R':  stds[0],  'std_G':  stds[1],  'std_B':  stds[2],
        'min': t.min().item(), 'max': t.max().item(),
    }


def fmt_stats(name, s):
    return (f"{name:<28} "
            f"mean=({s['mean_R']:+.4f},{s['mean_G']:+.4f},{s['mean_B']:+.4f}) "
            f"std=({s['std_R']:.4f},{s['std_G']:.4f},{s['std_B']:.4f}) "
            f"range=[{s['min']:.2f}, {s['max']:.2f}]")


# --------- run all three pipelines ---------
print("\n--- Running torchvision reference ---")
tv  = run_torchvision(imgs_uint8, N_TRIALS)
s_tv = per_channel_stats(tv)

print("--- Running gpu_augment v1 (current; pad=0 in normalized) ---")
v1  = run_gpu_aug(gpu_augment_v1_zero_in_norm, imgs_uint8, N_TRIALS)
s_v1 = per_channel_stats(v1)

print("--- Running gpu_augment v2 (fixed; pad=-MEAN/STD per channel) ---")
v2  = run_gpu_aug(gpu_augment_v2_black_pad, imgs_uint8, N_TRIALS)
s_v2 = per_channel_stats(v2)

# --------- report ---------
print("\n" + "="*100)
print("RESULTS")
print("="*100)
print(fmt_stats("torchvision (reference)", s_tv))
print(fmt_stats("gpu_augment v1 (current)", s_v1))
print(fmt_stats("gpu_augment v2 (fixed)", s_v2))

# Deltas
print("\nDelta vs torchvision (smaller = closer match):")
for name, s in [("v1 current", s_v1), ("v2 fixed", s_v2)]:
    d_means = [s[k] - s_tv[k] for k in ('mean_R', 'mean_G', 'mean_B')]
    d_stds  = [s[k] - s_tv[k] for k in ('std_R', 'std_G', 'std_B')]
    mean_l2 = float(np.linalg.norm(d_means))
    std_l2  = float(np.linalg.norm(d_stds))
    print(f"  {name:<14} "
          f"|Δmean|_2 = {mean_l2:.4f}   |Δstd|_2 = {std_l2:.4f}")
