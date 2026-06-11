"""perclient_summary_3seed.py — combine seed 0 + seed 1 + seed 2 for the
per-client smoke and produce SUMMARY_3seed.md with mean ± std across seeds.

Inputs:
  runs/2026-06-08-perclient-smoke/<method>/        (seed 0 — already present)
  runs/2026-06-08-perclient-3seed/seed1/<method>/  (seed 1 — new)
  runs/2026-06-08-perclient-3seed/seed2/<method>/  (seed 2 — new)

Outputs:
  runs/2026-06-08-perclient-3seed/SUMMARY_3seed.md
"""

import csv
import math
from pathlib import Path

import numpy as np

OUTROOT = Path('runs/2026-06-08-perclient-3seed')
DRIFT_ROUND = 100
NUM_ROUNDS  = 200

# (display_name, csv_basename, subdir_name_under_outdir)
METHODS = [
    ('FedAvg',         'results_FedAvg.csv',         'fedavg'),
    ('Flash',          'results_Flash.csv',          'flash'),
    ('AdaptiveFedAvg', 'results_AdaptiveFedAvg.csv', 'adaptive'),
    ('FedAvgPlus1',    'results_FedAvgPlus1.csv',    'fedavgplus1'),
    ('OurMethod',      'results_OurMethod.csv',      'ourmethod'),
]

SEED_ROOTS = {
    0: Path('runs/2026-06-08-perclient-smoke'),
    1: OUTROOT / 'seed1',
    2: OUTROOT / 'seed2',
}


def load_series(path: Path):
    rows = list(csv.DictReader(path.open()))
    g  = np.array([float(r['global_acc']) for r in rows])
    pc = np.array([float(r.get('per_client_gen_acc') or 'nan') for r in rows])
    return g, pc


def pre_dip_stable(series):
    pre    = float(np.mean(series[max(0, DRIFT_ROUND-11):DRIFT_ROUND]))
    dip    = pre - float(np.min(series[DRIFT_ROUND:DRIFT_ROUND+10]))
    stable = float(np.mean(series[-10:]))
    return pre, dip, stable


def fmt_mean_std(vals):
    """Format mean ± std (population std, divisor n; for n=3 this matches the
    1-sigma seed spread used in our prior 3-seed studies)."""
    arr = np.array(vals, dtype=float)
    if len(arr) == 0 or np.isnan(arr).any():
        return "—"
    m = float(arr.mean())
    s = float(arr.std(ddof=0))
    return f"{m:.4f} ± {s:.4f}"


def main():
    # data[seed][method] = {'global': (pre,dip,stable), 'pc': (pre,dip,stable),
    #                       'g_series': np.array, 'pc_series': np.array}
    data = {}
    missing = []
    for seed, root in SEED_ROOTS.items():
        data[seed] = {}
        for name, csv_base, subdir in METHODS:
            p = root / subdir / csv_base
            if not p.exists():
                missing.append(str(p))
                continue
            g, pc = load_series(p)
            data[seed][name] = {
                'global': pre_dip_stable(g),
                'pc':     pre_dip_stable(pc),
                'g_series': g,
                'pc_series': pc,
            }

    if missing:
        print("MISSING CSVs (will appear as — in tables):")
        for m in missing: print(f"  {m}")
        print()

    seeds_present = [s for s in SEED_ROOTS if data.get(s)]
    n_seeds = len(seeds_present)

    # Aggregate by method × window × metric across seeds.
    def collect(name, metric, window_idx):
        out = []
        for s in seeds_present:
            v = data[s].get(name)
            if v is None or name not in data[s]: continue
            out.append(data[s][name][metric][window_idx])
        return out

    # OurMethod - FedAvg delta on per-client stable, per seed.
    def per_seed_delta(method_a, method_b, metric, window_idx):
        out = []
        for s in seeds_present:
            a = data[s].get(method_a)
            b = data[s].get(method_b)
            if a is None or b is None: continue
            out.append(a[metric][window_idx] - b[metric][window_idx])
        return out

    # FedAvg vs FedAvgPlus1 global trajectory gate (per seed).
    gate_per_seed = {}
    for s in seeds_present:
        fa = data[s].get('FedAvg')
        fp = data[s].get('FedAvgPlus1')
        if fa is None or fp is None: continue
        diff = fp['g_series'] - fa['g_series']
        gate_per_seed[s] = {
            'max_abs': float(np.max(np.abs(diff))),
            'mean_abs': float(np.mean(np.abs(diff))),
            'final_abs': float(abs(diff[-1])),
        }

    # ---------- markdown build ----------
    L = []
    L.append("# Per-client comparison — 3 seeds (0, 1, 2)\n")
    L.append("**Setup:** CIFAR-10, Dir(0.1), 20 clients, 200 rounds, 5 local epochs, batch 64, single sudden drift at round 100.")
    L.append("Per-client metric = FedCCFA's per-client generalized accuracy (faithful port, validated bit-exact pre-drift at Step A).\n")
    L.append("**Hardware:** GCP L4 VM, `cudnn.benchmark = True` (convs non-deterministic across processes).\n")
    L.append(f"**Seeds aggregated:** {seeds_present}  (n = {n_seeds})\n")
    L.append("All values mean ± std across seeds; std is population (ddof=0).")
    L.append("`pre` = mean rounds [89, 99]; `dip` = pre − min(rounds [100, 109]); `stable` = mean(rounds [190, 199]).\n")

    # Global table
    L.append("## Global accuracy (canonical CIFAR-10 test set, undrifted labels)\n")
    L.append("| Method | Pre | Dip | Stable |")
    L.append("|---|---:|---:|---:|")
    for name, _, _ in METHODS:
        pre = fmt_mean_std(collect(name, 'global', 0))
        dip = fmt_mean_std(collect(name, 'global', 1))
        stb = fmt_mean_std(collect(name, 'global', 2))
        L.append(f"| {name} | {pre} | {dip} | {stb} |")
    L.append("")

    # Per-client table
    L.append("## Per-client generalized accuracy (FedCCFA protocol)\n")
    L.append("| Method | Pre | Dip | Stable |")
    L.append("|---|---:|---:|---:|")
    for name, _, _ in METHODS:
        pre = fmt_mean_std(collect(name, 'pc', 0))
        dip = fmt_mean_std(collect(name, 'pc', 1))
        stb = fmt_mean_std(collect(name, 'pc', 2))
        L.append(f"| {name} | {pre} | {dip} | {stb} |")
    L.append("")

    # Key deltas
    L.append("## Key per-client deltas (stable, mean ± std across seeds)\n")
    L.append("| Comparison | Δ per-client stable |")
    L.append("|---|---:|")
    d_om_fa = per_seed_delta('OurMethod', 'FedAvg', 'pc', 2)
    d_om_fp = per_seed_delta('OurMethod', 'FedAvgPlus1', 'pc', 2)
    arr_a = np.array(d_om_fa) * 100 if d_om_fa else None
    arr_b = np.array(d_om_fp) * 100 if d_om_fp else None
    if arr_a is not None:
        L.append(f"| OurMethod − FedAvg | {arr_a.mean():+.2f}pp ± {arr_a.std(ddof=0):.2f}pp |")
    if arr_b is not None:
        L.append(f"| OurMethod − FedAvgPlus1 (control) | {arr_b.mean():+.2f}pp ± {arr_b.std(ddof=0):.2f}pp |")
    L.append("")

    # Per-seed breakdown for the 3 methods that matter
    L.append("## Per-seed breakdown — per-client stable (rounds [190, 199] mean)\n")
    L.append("Shows seed-to-seed consistency vs noise for the three methods the comparison turns on.")
    L.append("")
    L.append("| Seed | FedAvg | FedAvgPlus1 (control) | OurMethod | OurMethod − FedAvg | OurMethod − Control |")
    L.append("|---:|---:|---:|---:|---:|---:|")
    for s in seeds_present:
        fa  = data[s].get('FedAvg',      {}).get('pc', (None,None,None))[2]
        fp  = data[s].get('FedAvgPlus1', {}).get('pc', (None,None,None))[2]
        om  = data[s].get('OurMethod',   {}).get('pc', (None,None,None))[2]
        def fmt(x): return f"{x:.4f}" if x is not None else "—"
        def dpp(a, b): return f"{(a-b)*100:+.2f}pp" if (a is not None and b is not None) else "—"
        L.append(f"| {s} | {fmt(fa)} | {fmt(fp)} | {fmt(om)} | {dpp(om, fa)} | {dpp(om, fp)} |")
    L.append("")

    # CUDA non-determinism gate per seed
    L.append("## FedAvg vs FedAvgPlus1 — global-trajectory non-determinism gate (per seed)\n")
    L.append("With `cudnn.benchmark = True` and the same seed, two PyTorch processes running the same training do NOT produce bit-identical")
    L.append("global accuracy across processes (CUDA convs are non-deterministic). The +1-epoch fine-tune in FedAvgPlus1 does NOT touch the")
    L.append("global model. So the FedAvg vs FedAvgPlus1 global divergence is a pure CUDA-noise floor. Reported per seed so the per-client")
    L.append("control comparison is interpretable.")
    L.append("")
    L.append("| Seed | max |Δglobal| | mean |Δglobal| | final |Δglobal| |")
    L.append("|---:|---:|---:|---:|")
    for s in seeds_present:
        g = gate_per_seed.get(s)
        if g is None:
            L.append(f"| {s} | — | — | — |")
        else:
            L.append(f"| {s} | {g['max_abs']:.4f} ({g['max_abs']*100:.2f}pp) | {g['mean_abs']:.4f} ({g['mean_abs']*100:.2f}pp) | {g['final_abs']:.4f} ({g['final_abs']*100:.2f}pp) |")
    L.append("")

    # Honest reading section
    L.append("## Honest reading\n")
    if arr_a is not None and arr_b is not None:
        mom_fa = arr_a.mean(); som_fa = arr_a.std(ddof=0)
        mom_fp = arr_b.mean(); som_fp = arr_b.std(ddof=0)
        # Effect-vs-noise test: |mean| > std?
        sig_fa = abs(mom_fa) > som_fa
        sig_fp = abs(mom_fp) > som_fp
        L.append(f"- **OurMethod vs FedAvg** (per-client stable): mean Δ = **{mom_fa:+.2f}pp**, std = **{som_fa:.2f}pp** across n={n_seeds} seeds. "
                 f"|mean| {'>' if sig_fa else '≤'} std → effect is {'OUTSIDE' if sig_fa else 'INSIDE'} the seed-noise floor.")
        L.append(f"- **OurMethod vs FedAvgPlus1 (control)** (per-client stable): mean Δ = **{mom_fp:+.2f}pp**, std = **{som_fp:.2f}pp**. "
                 f"|mean| {'>' if sig_fp else '≤'} std → effect is {'OUTSIDE' if sig_fp else 'INSIDE'} the seed-noise floor.")
        L.append("")
        L.append("Bottom line guidance: an effect 'inside the seed-noise floor' at n=3 is not a result; report it as such. At n=3 even a clear")
        L.append("signed mean only becomes a defensible claim if |mean| comfortably exceeds std AND the per-seed deltas all agree in sign.")
    L.append("")

    out = OUTROOT / 'SUMMARY_3seed.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding='utf-8')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
