"""experiment_report_compute.py - read-only aggregation of every committed
result CSV into one structured stdout dump, used to build EXPERIMENT_REPORT_PARTA.md.

Cites WHICH CSV each number comes from. Does not estimate. Where a file is
missing prints NOT_FOUND. Where mean+/-std is reported, the per-seed values
are also printed for traceability.

Windows used (matching the harness's summarize_method):
  pre    = mean(global_acc[89..99])         (10 rounds just before drift @100)
  dip    = pre - min(global_acc[100..110])  (10 rounds just after drift)
  stable = mean(global_acc[-10:])           (final 10 rounds)
Same windows used for per-client gen acc.
"""

import csv
from pathlib import Path
import numpy as np

# Paths used (on disk after main checkout + unmerged-branch extraction)
REPO = Path('.')
SCRATCH = Path(r'C:\Users\suraf\AppData\Local\Temp\claude\d--projects-Seminars-in-AI-Paper-Proposal-Second-Semester-Implementation-FedCCFA\27e522cf-d3fe-44e4-946b-eff40f6a5a6d\scratchpad')

DRIFT_ROUND = 100


def load(p: Path):
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    g  = np.array([float(r['global_acc']) for r in rows])
    pc_col = 'per_client_gen_acc'
    pc = np.array([float(r[pc_col]) for r in rows]) if pc_col in rows[0] else None
    return g, pc, rows


def windows(s):
    if s is None or len(s) < DRIFT_ROUND + 10:
        return None, None, None
    pre = float(np.mean(s[max(0,DRIFT_ROUND-11):DRIFT_ROUND]))
    dip = pre - float(np.min(s[DRIFT_ROUND:DRIFT_ROUND+10]))
    stable = float(np.mean(s[-10:]))
    return pre, dip, stable


def report_one(label, csv_path):
    out = load(csv_path)
    if out is None:
        print(f"  {label:>34}: NOT_FOUND  ({csv_path})")
        return None
    g, pc, _ = out
    gw = windows(g)
    pcw = windows(pc) if pc is not None else (None,None,None)
    print(f"  {label:>34}: g={gw[0]:.4f}/{gw[1]:.4f}/{gw[2]:.4f}  "
          f"pc={pcw[0] if pcw[0] is None else f'{pcw[0]:.4f}'}/"
          f"{pcw[1] if pcw[1] is None else f'{pcw[1]:.4f}'}/"
          f"{pcw[2] if pcw[2] is None else f'{pcw[2]:.4f}'}    [{csv_path}]")
    return gw, pcw


# ----------------------------------------------------------------
# Section 3: Single sudden drift, 3 seeds, MAIN RESULT
# ----------------------------------------------------------------
print("=" * 100)
print("SECTION 3: SINGLE SUDDEN DRIFT, 3 SEEDS (the main result)")
print("=" * 100)

# Three seeds for each method. Paths per method:
SEED_SOURCES = {
    'FedAvg': {
        0: 'runs/2026-06-08-perclient-smoke/fedavg/results_FedAvg.csv',
        1: 'runs/2026-06-08-perclient-3seed/seed1/fedavg/results_FedAvg.csv',
        2: 'runs/2026-06-08-perclient-3seed/seed2/fedavg/results_FedAvg.csv',
    },
    'Flash': {
        0: 'runs/2026-06-08-perclient-smoke/flash/results_Flash.csv',
        1: 'runs/2026-06-08-perclient-3seed/seed1/flash/results_Flash.csv',
        2: 'runs/2026-06-08-perclient-3seed/seed2/flash/results_Flash.csv',
    },
    'AdaptiveFedAvg-broken': {
        0: 'runs/2026-06-08-perclient-smoke/adaptive/results_AdaptiveFedAvg.csv',
        1: 'runs/2026-06-08-perclient-3seed/seed1/adaptive/results_AdaptiveFedAvg.csv',
        2: 'runs/2026-06-08-perclient-3seed/seed2/adaptive/results_AdaptiveFedAvg.csv',
    },
    'AdaptiveFedAvg-corrected': {
        0: 'runs/2026-06-09-adaptivefedavg-fix/seed0/results_AdaptiveFedAvg.csv',
        1: 'runs/2026-06-09-adaptivefedavg-fix/seed1/results_AdaptiveFedAvg.csv',
        2: 'runs/2026-06-09-adaptivefedavg-fix/seed2/results_AdaptiveFedAvg.csv',
    },
    'FedAvgPlus1': {
        0: 'runs/2026-06-08-perclient-smoke/fedavgplus1/results_FedAvgPlus1.csv',
        1: 'runs/2026-06-08-perclient-3seed/seed1/fedavgplus1/results_FedAvgPlus1.csv',
        2: 'runs/2026-06-08-perclient-3seed/seed2/fedavgplus1/results_FedAvgPlus1.csv',
    },
    'OurMethod': {
        0: 'runs/2026-06-08-perclient-smoke/ourmethod/results_OurMethod.csv',
        1: 'runs/2026-06-08-perclient-3seed/seed1/ourmethod/results_OurMethod.csv',
        2: 'runs/2026-06-08-perclient-3seed/seed2/ourmethod/results_OurMethod.csv',
    },
    'Saile': {
        0: 'runs/2026-06-11-saile-3seed/seed0/results_Saile.csv',
        1: 'runs/2026-06-11-saile-3seed/seed1/results_Saile.csv',
        2: 'runs/2026-06-11-saile-3seed/seed2/results_Saile.csv',
    },
}


def collect_3seed(method, paths):
    pre_g, dip_g, stb_g = [], [], []
    pre_pc, dip_pc, stb_pc = [], [], []
    per_seed = {}
    for s, p in paths.items():
        out = load(Path(p))
        if out is None:
            per_seed[s] = ('MISSING', p)
            continue
        g, pc, _ = out
        gw = windows(g); pcw = windows(pc) if pc is not None else (None,None,None)
        pre_g.append(gw[0]); dip_g.append(gw[1]); stb_g.append(gw[2])
        if pc is not None:
            pre_pc.append(pcw[0]); dip_pc.append(pcw[1]); stb_pc.append(pcw[2])
        per_seed[s] = (gw, pcw, p)
    return {
        'global': {'pre': pre_g, 'dip': dip_g, 'stb': stb_g},
        'pc':     {'pre': pre_pc, 'dip': dip_pc, 'stb': stb_pc},
        'per_seed': per_seed,
    }


def ms(vals):
    if len(vals) == 0:
        return "n/a"
    a = np.array(vals, dtype=float)
    return f"{a.mean():.4f} +/- {a.std(ddof=0):.4f}"


data = {}
for method, paths in SEED_SOURCES.items():
    data[method] = collect_3seed(method, paths)
    print(f"\n{method}:")
    for s, info in data[method]['per_seed'].items():
        if info[0] == 'MISSING':
            print(f"  seed {s}: MISSING  ({info[1]})")
        else:
            gw, pcw, p = info
            print(f"  seed {s}: g={gw[0]:.4f}/{gw[1]:.4f}/{gw[2]:.4f}  "
                  f"pc={pcw[0]:.4f}/{pcw[1]:.4f}/{pcw[2]:.4f}    [{p}]")
    print(f"  mean+/-std GLOBAL:   pre={ms(data[method]['global']['pre'])}  "
          f"dip={ms(data[method]['global']['dip'])}  "
          f"stable={ms(data[method]['global']['stb'])}")
    print(f"  mean+/-std PER-CLI:  pre={ms(data[method]['pc']['pre'])}  "
          f"dip={ms(data[method]['pc']['dip'])}  "
          f"stable={ms(data[method]['pc']['stb'])}")


# OurMethod - FedAvg deltas per seed
print()
print("-" * 100)
print("OurMethod - FedAvg per-seed deltas:")
for s in (0,1,2):
    om = data['OurMethod']['per_seed'][s]
    fa = data['FedAvg']['per_seed'][s]
    if om[0] == 'MISSING' or fa[0] == 'MISSING': continue
    om_gw, om_pcw, _ = om
    fa_gw, fa_pcw, _ = fa
    dg = (om_gw[2] - fa_gw[2]) * 100
    dpc = (om_pcw[2] - fa_pcw[2]) * 100
    print(f"  seed {s}: global_stable {dg:+.2f}pp,  per_client_stable {dpc:+.2f}pp")
om_pc_stb = data['OurMethod']['pc']['stb']
fa_pc_stb = data['FedAvg']['pc']['stb']
om_g_stb  = data['OurMethod']['global']['stb']
fa_g_stb  = data['FedAvg']['global']['stb']
d_pc = np.array(om_pc_stb) - np.array(fa_pc_stb)
d_g  = np.array(om_g_stb)  - np.array(fa_g_stb)
print(f"  OverAll OurMethod - FedAvg (pc stable): "
      f"mean {d_pc.mean()*100:+.2f}pp +/- {d_pc.std(ddof=0)*100:.2f}pp; "
      f"signs {[1 if d>0 else -1 for d in d_pc]}; signs_agree={len(set(np.sign(d_pc)))==1}")
print(f"  OverAll OurMethod - FedAvg (g  stable): "
      f"mean {d_g.mean()*100:+.2f}pp +/- {d_g.std(ddof=0)*100:.2f}pp; "
      f"signs {[1 if d>0 else -1 for d in d_g]}")


# ----------------------------------------------------------------
# Section 4: OurMethod detection stats (3 seeds, single drift)
# ----------------------------------------------------------------
print()
print("=" * 100)
print("SECTION 4: OurMethod detection stats (single sudden drift, 3 seeds)")
print("=" * 100)

FLAG_SRC = {
    0: 'runs/2026-06-08-perclient-smoke/ourmethod/results_OurMethod_flags.csv',
    1: 'runs/2026-06-08-perclient-3seed/seed1/ourmethod/results_OurMethod_flags.csv',
    2: 'runs/2026-06-08-perclient-3seed/seed2/ourmethod/results_OurMethod_flags.csv',
}
for s, p in FLAG_SRC.items():
    rows = list(csv.DictReader(Path(p).open()))
    fc = [int(rows[r]['flagged_count']) for r in range(100, 110)]
    fl3 = [int(rows[r]['flagged_layer3_count']) for r in range(100, 110)]
    fl4 = [int(rows[r]['flagged_layer4_count']) for r in range(100, 110)]
    print(f"  seed {s} [{p}]:")
    print(f"    flagged_count   [100..109] = {fc}  peak={max(fc)}/20 ({max(fc)*5}%)")
    print(f"    layer3 flag cnt [100..109] = {fl3}  peak={max(fl3)}/20")
    print(f"    layer4 flag cnt [100..109] = {fl4}  peak={max(fl4)}/20")


# ----------------------------------------------------------------
# Section 5: Ablation results
# ----------------------------------------------------------------
print()
print("=" * 100)
print("SECTION 5: Ablation results (single seed=0, 200 rounds)")
print("=" * 100)
ABL_VARIANTS = ['baseline', 'no-detection', 'all-layers', 'tau-low', 'tau-high']
for v in ABL_VARIANTS:
    p = Path(f'runs/2026-06-04-ablations/{v}/results_OurMethod.csv')
    report_one(v, p)


# ----------------------------------------------------------------
# Section 6: Other tests (dir05, aggressive, covariate L3/L4, covariate L1/L2, recurrent)
# ----------------------------------------------------------------
print()
print("=" * 100)
print("SECTION 6: Other go/no-go tests (single seed=0)")
print("=" * 100)

OTHER_TESTS = [
    ('Dir(0.5) test', {
        'FedAvg':    'runs/2026-06-11-dir05-test/fedavg/results_FedAvg.csv',
        'OurMethod': 'runs/2026-06-11-dir05-test/ourmethod/results_OurMethod.csv',
        'flags':     'runs/2026-06-11-dir05-test/ourmethod/results_OurMethod_flags.csv',
    }),
    ('Recurrent alternating drift', {
        'FedAvg':    'runs/2026-06-13-recurrent-gain-test/fedavg/results_FedAvg.csv',
        'OurMethod': 'runs/2026-06-13-recurrent-gain-test/ourmethod/results_OurMethod.csv',
        'flags':     'runs/2026-06-13-recurrent-gain-test/ourmethod/results_OurMethod_flags.csv',
    }),
    ('Aggressive concept (full perms)', {
        'FedAvg':    str(SCRATCH / 'aggressive' / 'FedAvg.csv'),
        'OurMethod': str(SCRATCH / 'aggressive' / 'OurMethod.csv'),
        'flags':     str(SCRATCH / 'aggressive' / 'OurMethod_flags.csv'),
    }),
    ('Covariate drift, L3/L4 flag (default)', {
        'FedAvg':    str(SCRATCH / 'covariate' / 'FedAvg.csv'),
        'OurMethod': str(SCRATCH / 'covariate' / 'OurMethod.csv'),
        'flags':     str(SCRATCH / 'covariate' / 'OurMethod_flags.csv'),
    }),
    ('Covariate drift, L1/L2 flag (re-aim)', {
        'FedAvg':    str(SCRATCH / 'covariate_l1l2' / 'FedAvg.csv'),
        'OurMethod': str(SCRATCH / 'covariate_l1l2' / 'OurMethod.csv'),
        'flags':     str(SCRATCH / 'covariate_l1l2' / 'OurMethod_flags.csv'),
    }),
]

for tname, src in OTHER_TESTS:
    print(f"\n{tname}:")
    fa_out = load(Path(src['FedAvg']))
    om_out = load(Path(src['OurMethod']))
    if fa_out is None or om_out is None:
        print(f"  MISSING  fa={src['FedAvg']}  om={src['OurMethod']}")
        continue
    fa_g, fa_pc, _ = fa_out
    om_g, om_pc, _ = om_out
    fa_gw = windows(fa_g); fa_pcw = windows(fa_pc)
    om_gw = windows(om_g); om_pcw = windows(om_pc)
    print(f"  FedAvg     g={fa_gw[0]:.4f}/{fa_gw[1]:.4f}/{fa_gw[2]:.4f}  pc={fa_pcw[0]:.4f}/{fa_pcw[1]:.4f}/{fa_pcw[2]:.4f}")
    print(f"  OurMethod  g={om_gw[0]:.4f}/{om_gw[1]:.4f}/{om_gw[2]:.4f}  pc={om_pcw[0]:.4f}/{om_pcw[1]:.4f}/{om_pcw[2]:.4f}")
    print(f"  Delta global stable:    {(om_gw[2]-fa_gw[2])*100:+.2f}pp")
    print(f"  Delta per-client stable: {(om_pcw[2]-fa_pcw[2])*100:+.2f}pp")
    # Flag stats
    if Path(src['flags']).exists():
        rows = list(csv.DictReader(Path(src['flags']).open()))
        # For recurrent there are multiple drift events; check around 40,80,120,160
        if 'recurrent' in tname.lower() or 'alternating' in tname.lower():
            for ev_round in [40, 80, 120, 160]:
                fc = [int(rows[r]['flagged_count']) for r in range(ev_round, min(ev_round+10, 200))]
                print(f"  Detection event @round {ev_round}: flagged_count = {fc}  peak={max(fc)}/20")
        else:
            fc = [int(rows[r]['flagged_count']) for r in range(100, 110)]
            print(f"  Detection [100..109]: flagged_count = {fc}  peak={max(fc)}/20  ({max(fc)*5}%)")


# ----------------------------------------------------------------
# Section 7: Communication analysis - read computed values from prior run
# ----------------------------------------------------------------
print()
print("=" * 100)
print("SECTION 7: Communication analysis (whole-run + drift-window saving)")
print("=" * 100)

# Reproduce the comm-analysis numbers directly so they're cited from CSVs not from comm_analysis.py
L1_PARAMS = 3 * 16 * 5 * 5 + 16
L2_PARAMS = 16 * 32 * 5 * 5 + 32
L3_PARAMS = 32 * 64 * 3 * 3 + 64
L4_PARAMS = 576 * 128 + 128
FC_PARAMS = 128 * 10 + 10
TOTAL = L1_PARAMS + L2_PARAMS + L3_PARAMS + L4_PARAMS + FC_PARAMS
ALWAYS_UP = L1_PARAMS + L2_PARAMS + FC_PARAMS
N_CLIENTS, N_ROUNDS = 20, 200
print(f"  Per-layer params: L1={L1_PARAMS}, L2={L2_PARAMS}, L3={L3_PARAMS}, L4={L4_PARAMS}, fc={FC_PARAMS}, total={TOTAL}")
print(f"  FedAvg per-round upload = {N_CLIENTS * TOTAL:,} params (constant)")
print(f"  FedAvg whole-run upload = {N_ROUNDS * N_CLIENTS * TOTAL:,} params")
print()
for s, p in FLAG_SRC.items():
    rows = list(csv.DictReader(Path(p).open()))
    fl3 = np.array([int(r['flagged_layer3_count']) for r in rows])
    fl4 = np.array([int(r['flagged_layer4_count']) for r in rows])
    saving = fl3 * L3_PARAMS + fl4 * L4_PARAMS  # params NOT uploaded
    total_saving = int(saving.sum())
    fedavg_total = N_ROUNDS * N_CLIENTS * TOTAL
    pct_whole = 100 * total_saving / fedavg_total
    drift_saving = int(saving[100:110].sum())
    fedavg_drift = 10 * N_CLIENTS * TOTAL
    pct_drift = 100 * drift_saving / fedavg_drift
    l3_share = 100 * fl3.sum() * L3_PARAMS / max(1, total_saving)
    l4_share = 100 * fl4.sum() * L4_PARAMS / max(1, total_saving)
    print(f"  seed {s}: whole-run saving = {pct_whole:.3f}%  ({total_saving:,} params)  "
          f"drift-window saving = {pct_drift:.3f}%  L3 share={l3_share:.1f}%  L4 share={l4_share:.1f}%  [{p}]")
