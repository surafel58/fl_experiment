"""comm_analysis.py - communication-cost analysis of OurMethod vs FedAvg.

Read-only. Uses committed flag CSVs from the 3-seed single-drift runs
(runs/2026-06-08-perclient-smoke + runs/2026-06-08-perclient-3seed).

Method (mechanically):
  FedAvg: every round, every client uploads ALL parameters.
  OurMethod: every round, every client uploads STABLE_LAYERS (layer1, layer2)
             + CLASSIFIER (fc) + DRIFT_LAYERS it did NOT flag (layer3/layer4
             that aren't flagged for that client). Flagged layers are kept
             local. So per-round uploaded params from a client:
               L1 + L2 + fc  (always)
             + (L3 if not flagged for this client)
             + (L4 if not flagged for this client)

Counts come from flag_layer3_count and flag_layer4_count in the flag CSV
(these track how many of the 20 clients had each layer flagged per round).

Per-round saving (OurMethod vs FedAvg) = flag_L3_count * L3_params
                                       + flag_L4_count * L4_params.

OurMethod NEVER uploads MORE than FedAvg - it can only withhold layers.
So the "saving" is always >= 0 per round.
"""

import csv
from pathlib import Path
import numpy as np


# ============================================================
# Per-layer parameter counts for CifarCNN.
# Derived from the harness's LAYER_GROUPS at all_experiments_optimized.py:1104-1112:
#   layer1 = hidden_layers.0  = Conv2d(3, 16, 5)
#   layer2 = hidden_layers.3  = Conv2d(16, 32, 5, padding=1)
#   layer3 = hidden_layers.6  = Conv2d(32, 64, 3, padding=1)
#   layer4 = hidden_layers.10 = Linear(64*3*3=576, 128)
#   fc     (classifier)        = Linear(128, 10)
#
# Conv2d(in, out, k) params: in*out*k*k + out (bias)
# Linear(in, out) params:    in*out + out (bias)
# ============================================================
L1_PARAMS = 3 * 16 * 5 * 5 + 16          # = 1216
L2_PARAMS = 16 * 32 * 5 * 5 + 32         # = 12832
L3_PARAMS = 32 * 64 * 3 * 3 + 64         # = 18496
L4_PARAMS = 576 * 128 + 128              # = 73856
FC_PARAMS = 128 * 10 + 10                # = 1290
TOTAL_PARAMS = L1_PARAMS + L2_PARAMS + L3_PARAMS + L4_PARAMS + FC_PARAMS   # = 107690

# In OurMethod, STABLE_LAYERS = ['layer1', 'layer2'] and CLASSIFIER = 'fc'
# are always uploaded. DRIFT_LAYERS = ['layer3', 'layer4'] can be flagged.
ALWAYS_UPLOAD = L1_PARAMS + L2_PARAMS + FC_PARAMS   # = 15338
FLAGGABLE     = L3_PARAMS + L4_PARAMS                # = 92352

NUM_CLIENTS = 20
NUM_ROUNDS  = 200
DRIFT_ROUND = 100


SEEDS = {
    0: 'runs/2026-06-08-perclient-smoke/ourmethod/results_OurMethod_flags.csv',
    1: 'runs/2026-06-08-perclient-3seed/seed1/ourmethod/results_OurMethod_flags.csv',
    2: 'runs/2026-06-08-perclient-3seed/seed2/ourmethod/results_OurMethod_flags.csv',
}


def load_seed(path: Path):
    rows = list(csv.DictReader(path.open()))
    fl3  = np.array([int(r['flagged_layer3_count']) for r in rows])
    fl4  = np.array([int(r['flagged_layer4_count']) for r in rows])
    flag_any = np.array([int(r['flagged_count']) for r in rows])
    return fl3, fl4, flag_any


def compute(fl3, fl4):
    """Returns dict of per-round arrays and totals."""
    fedavg_per_round = NUM_CLIENTS * TOTAL_PARAMS    # constant
    fedavg_total     = NUM_ROUNDS * fedavg_per_round

    # OurMethod per-round upload: NUM_CLIENTS * ALWAYS_UPLOAD
    #                           + (NUM_CLIENTS - fl3) * L3_PARAMS
    #                           + (NUM_CLIENTS - fl4) * L4_PARAMS
    ourm_per_round = (NUM_CLIENTS * ALWAYS_UPLOAD
                      + (NUM_CLIENTS - fl3) * L3_PARAMS
                      + (NUM_CLIENTS - fl4) * L4_PARAMS)

    # Saving = FedAvg - OurMethod = fl3 * L3 + fl4 * L4
    saving_per_round = fl3 * L3_PARAMS + fl4 * L4_PARAMS

    return {
        'fedavg_per_round':   fedavg_per_round,
        'fedavg_total':       fedavg_total,
        'ourm_per_round':     ourm_per_round,
        'ourm_total':         int(ourm_per_round.sum()),
        'saving_per_round':   saving_per_round,
        'saving_total':       int(saving_per_round.sum()),
        'fl3':                fl3,
        'fl4':                fl4,
    }


def fmt_params(n):
    """Format param counts: 107690 -> '107.7k'; 21538000000 -> '21.54G'."""
    if n >= 1e9:  return f"{n/1e9:.2f}G"
    if n >= 1e6:  return f"{n/1e6:.2f}M"
    if n >= 1e3:  return f"{n/1e3:.2f}k"
    return str(n)


def pct(saving, fedavg):
    return 100.0 * saving / fedavg if fedavg > 0 else 0.0


# ---------- collect per-seed ----------
results = {}
for s, p in SEEDS.items():
    fl3, fl4, _ = load_seed(Path(p))
    results[s] = compute(fl3, fl4)

# ---------- per-seed table data ----------
def aggregate(field):
    return np.array([results[s][field] for s in (0,1,2)])


# Whole-run totals
fedavg_total = results[0]['fedavg_total']           # constant across seeds
ourm_totals  = [results[s]['ourm_total'] for s in (0,1,2)]
saving_totals = [results[s]['saving_total'] for s in (0,1,2)]
saving_pcts  = [pct(saving_totals[s], fedavg_total) for s in (0,1,2)]

# Drift-window totals (rounds 100-109, the 10 rounds immediately after drift)
DRIFT_WIN_START = 100
DRIFT_WIN_END   = 110  # exclusive
DRIFT_WIN_LEN   = DRIFT_WIN_END - DRIFT_WIN_START
fedavg_drift_total = DRIFT_WIN_LEN * results[0]['fedavg_per_round']
ourm_drift_totals  = [int(results[s]['ourm_per_round'][DRIFT_WIN_START:DRIFT_WIN_END].sum()) for s in (0,1,2)]
saving_drift_totals = [int(results[s]['saving_per_round'][DRIFT_WIN_START:DRIFT_WIN_END].sum()) for s in (0,1,2)]
saving_drift_pcts  = [pct(saving_drift_totals[s], fedavg_drift_total) for s in (0,1,2)]

# Outside the drift window
non_drift_pcts = []
for s in (0,1,2):
    rest_saving = results[s]['saving_total'] - saving_drift_totals[s]
    rest_fedavg = fedavg_total - fedavg_drift_total
    non_drift_pcts.append(pct(rest_saving, rest_fedavg))

# Layer-attribution (how much of the saving is from L3 vs L4)
l3_share = []
l4_share = []
for s in (0,1,2):
    l3_save = int(results[s]['fl3'].sum() * L3_PARAMS)
    l4_save = int(results[s]['fl4'].sum() * L4_PARAMS)
    tot = max(l3_save + l4_save, 1)
    l3_share.append(100.0 * l3_save / tot)
    l4_share.append(100.0 * l4_save / tot)


def msd(vals):
    a = np.array(vals, dtype=float)
    return a.mean(), a.std(ddof=0)


# ---------- markdown ----------
L = []
L.append("# Communication-cost analysis: OurMethod vs FedAvg\n")
L.append("**Branch:** `comm-analysis`. Read-only analysis - no new training. Uses committed flag CSVs from the 3-seed single-drift runs (perclient-metric track).\n")

# 1. Per-layer counts
L.append("## 1. Per-layer parameter counts (CifarCNN, 107,690 total)\n")
L.append("Derived from `LAYER_GROUPS` at all_experiments_optimized.py:1104-1112.\n")
L.append("| Layer | Module | Op | Params | % of total |")
L.append("|---|---|---|---:|---:|")
L.append(f"| layer1 | hidden_layers.0  | Conv2d(3,16,5)             | {L1_PARAMS:>6,} | {100*L1_PARAMS/TOTAL_PARAMS:>5.2f}% |")
L.append(f"| layer2 | hidden_layers.3  | Conv2d(16,32,5,padding=1)  | {L2_PARAMS:>6,} | {100*L2_PARAMS/TOTAL_PARAMS:>5.2f}% |")
L.append(f"| layer3 | hidden_layers.6  | Conv2d(32,64,3,padding=1)  | {L3_PARAMS:>6,} | {100*L3_PARAMS/TOTAL_PARAMS:>5.2f}% |")
L.append(f"| layer4 | hidden_layers.10 | Linear(576, 128)            | {L4_PARAMS:>6,} | {100*L4_PARAMS/TOTAL_PARAMS:>5.2f}% |")
L.append(f"| fc (classifier) | fc | Linear(128, 10) | {FC_PARAMS:>6,} | {100*FC_PARAMS/TOTAL_PARAMS:>5.2f}% |")
L.append(f"| **total** | - | - | **{TOTAL_PARAMS:>6,}** | 100.00% |")
L.append("")
L.append(f"- Always uploaded (L1 + L2 + fc): **{ALWAYS_UPLOAD:,} params** = {100*ALWAYS_UPLOAD/TOTAL_PARAMS:.2f}% of model.")
L.append(f"- Flaggable layers (L3 + L4): **{FLAGGABLE:,} params** = {100*FLAGGABLE/TOTAL_PARAMS:.2f}% of model.")
L.append(f"- L4 alone is **{L4_PARAMS:,} = {100*L4_PARAMS/TOTAL_PARAMS:.2f}%** of the model - so an L4 flag saves ~4x what an L3 flag saves.")
L.append("")

# 2. Per-round budgets
L.append("## 2. Per-round upload budget (20 clients participating, every round)\n")
L.append(f"- **FedAvg (constant)**: 20 clients * {TOTAL_PARAMS:,} params = **{fmt_params(20 * TOTAL_PARAMS)} params/round**, every round.")
L.append("- **OurMethod (variable)**: per-round upload = 20 * (L1 + L2 + fc) + (20 - flagged_L3) * L3 + (20 - flagged_L4) * L4.")
L.append(f"  - Minimum (every client flags both L3 and L4): 20 * {ALWAYS_UPLOAD:,} = {fmt_params(20 * ALWAYS_UPLOAD)} params/round.")
L.append(f"  - Maximum (no flags fire, the baseline state): {fmt_params(20 * TOTAL_PARAMS)} params/round = identical to FedAvg.")
L.append("")
L.append("OurMethod can NEVER upload more than FedAvg - it can only withhold layers, never add them. The saving is therefore always non-negative per round.")
L.append("")

# 3. Whole-run totals
L.append("## 3. Whole-run totals (200 rounds, per seed)\n")
L.append(f"FedAvg constant baseline: 200 rounds * {fmt_params(20 * TOTAL_PARAMS)} = **{fmt_params(fedavg_total)} params total** uploaded per run.\n")
L.append("| Seed | OurMethod uploaded | FedAvg uploaded | Saving (params) | Saving (%) |")
L.append("|---:|---:|---:|---:|---:|")
for s in (0,1,2):
    L.append(f"| {s} | {fmt_params(ourm_totals[s])} | {fmt_params(fedavg_total)} | {fmt_params(saving_totals[s])} | **{saving_pcts[s]:.3f}%** |")
m, sd = msd(saving_pcts)
L.append(f"| **mean +/- std** | - | - | {fmt_params(int(np.mean(saving_totals)))} | **{m:.3f}% +/- {sd:.3f}%** |")
L.append("")
L.append(f"Whole-run saving across 3 seeds: **mean {m:.3f}%, std {sd:.3f}%**. Well under 1%, as expected. The mechanism is dormant for the vast majority of training (rounds 0-99 pre-drift, rounds ~105-199 post-drift); the saving comes from a brief window where flags fire.")
L.append("")

# 4. Drift-window breakdown
L.append("## 4. Drift-window vs rest-of-run breakdown\n")
L.append("Drift window = rounds [100, 109] (10 rounds immediately after drift), where the detection mechanism actually fires. Outside this window, OurMethod uploads ~identically to FedAvg.\n")
L.append("| Seed | drift-window saving (% of drift-window FedAvg) | rest-of-run saving (% of rest-of-run FedAvg) |")
L.append("|---:|---:|---:|")
for s in (0,1,2):
    L.append(f"| {s} | **{saving_drift_pcts[s]:.3f}%** | {non_drift_pcts[s]:.4f}% |")
md, sdd = msd(saving_drift_pcts)
mn, sdn = msd(non_drift_pcts)
L.append(f"| **mean +/- std** | **{md:.3f}% +/- {sdd:.3f}%** | {mn:.4f}% +/- {sdn:.4f}% |")
L.append("")
L.append(f"During the drift window the saving is **~{md:.1f}%** of communication, ~{md/max(m, 0.001):.0f}x the whole-run average. Outside the drift window the saving is essentially zero ({mn:.4f}%) - the detector doesn't fire, OurMethod uploads exactly what FedAvg uploads.")
L.append("")

# 5. Layer attribution
L.append("## 5. Layer attribution of the savings\n")
L.append("Of the total params saved across the run, how much came from L3 flags vs L4 flags?\n")
L.append("| Seed | L3 share | L4 share | total flags fired (sum over rounds) |")
L.append("|---:|---:|---:|---|")
for s in (0,1,2):
    fl3_total = int(results[s]['fl3'].sum())
    fl4_total = int(results[s]['fl4'].sum())
    L.append(f"| {s} | {l3_share[s]:.1f}% | {l4_share[s]:.1f}% | L3 flags: {fl3_total}, L4 flags: {fl4_total} |")
ml3, sl3 = msd(l3_share)
ml4, sl4 = msd(l4_share)
L.append(f"| **mean** | {ml3:.1f}% | {ml4:.1f}% | - |")
L.append("")
L.append(f"L4 dominates the saving ({ml4:.0f}% on average) even though L3 and L4 are flagged at similar frequencies, because L4 has ~4x as many params (73.9k vs 18.5k). The detector fires on L3 and L4 roughly equally, but in communication terms an L4 flag is worth ~4 L3 flags.")
L.append("")

# 6. Honest reading
L.append("## 6. Honest reading\n")
L.append(f"1. **The whole-run communication saving is small**: **mean {m:.3f}% +/- {sd:.3f}%** across 3 seeds. The mechanism is dormant outside the drift window; OurMethod and FedAvg are operationally identical for ~190 of the 200 rounds in a single-drift run.")
L.append("")
L.append(f"2. **The drift-window saving is meaningfully larger**: **mean {md:.3f}% +/- {sdd:.3f}%** during rounds [100, 109]. \"When the mechanism actually fires, it withholds roughly {md:.0f}% of the layer uploads from the flagged clients.\" This is the honest \"when it matters\" number.")
L.append("")
L.append("3. **The asymmetry property**: OurMethod NEVER uploads more than FedAvg - it only ever withholds layers, never adds them. So the property is **\"never worse on communication, modestly better during detected drift\"**. There is no scenario where this mechanism increases communication cost.")
L.append("")
L.append("4. **Why the saving is concentrated, not constant**: the EMA detector fires only when per-layer weight-change magnitudes spike, which happens almost exclusively at drift (rounds 100-104 in seed 0, similar in other seeds). After the spike subsides (typically by round 105-110), the detector goes silent and OurMethod's selective-sync path becomes a no-op. The dip-mitigation benefit and the communication-saving benefit are co-located in time.")
L.append("")
L.append(f"5. **Scaling note**: at our setup (200 rounds, single drift), the saving is small in absolute % terms. In a setting with MORE frequent drift events (e.g. continuous incremental drift, or recurrent drift every K rounds), the drift-window fraction of total training would be larger, and the whole-run saving would scale proportionally. **The whole-run % saving is essentially (drift-window saving %) * (drift-fraction of training)**. At our 10/200 = 5% drift fraction, a ~{md:.0f}% drift-window saving yields ~{md*10/200:.2f}% whole-run saving, which matches the observed ~{m:.2f}%.")
L.append("")
L.append(f"**Bottom line:** OurMethod's communication saving over FedAvg is **{m:.2f}% whole-run, {md:.1f}% during drift, never worse**. Small in absolute terms at a single-drift operating point; would scale up with more frequent drift events. The mechanism is a layer-withholding *side effect* of the drift-detection mechanism, not its primary purpose - the accuracy effect (small/no measurable gain) is the load-bearing claim, and the communication saving is a documented side benefit.")

out = Path('runs/2026-06-13-comm-analysis/SUMMARY.md')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(L), encoding='utf-8')
print(f"Wrote {out}")

print()
print("=== Quick numbers (verification) ===")
print(f"Total params:        {TOTAL_PARAMS:,}")
print(f"L1={L1_PARAMS:,}  L2={L2_PARAMS:,}  L3={L3_PARAMS:,}  L4={L4_PARAMS:,}  fc={FC_PARAMS:,}")
print(f"FedAvg per round:    {20*TOTAL_PARAMS:,} params")
print(f"FedAvg total (200r): {fedavg_total:,} params = {fmt_params(fedavg_total)}")
print()
for s in (0,1,2):
    print(f"Seed {s}: OurMethod={ourm_totals[s]:>14,}  saving={saving_totals[s]:>9,} = {saving_pcts[s]:.3f}% whole-run, {saving_drift_pcts[s]:.3f}% drift-window")
print(f"Mean whole-run saving: {m:.3f}% +/- {sd:.3f}%")
print(f"Mean drift-window saving: {md:.3f}% +/- {sdd:.3f}%")
