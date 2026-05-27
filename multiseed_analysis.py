"""
multiseed_analysis.py — analyses on existing CSVs (no VM, no extra compute).

STEP 1: Per-client local-accuracy breakdown
  For seeds 1 and 2 (which have per-client columns), compare each method's
  per-client recovery vs pre-drift baseline. Especially: does OurMethod help
  the *flagged* clients more than FedAvg does?

STEP 2: Communication-cost accounting (BONUS METRIC, not primary)
  Compare total bytes uploaded across 200 rounds. FedAvg/Flash/Adaptive all
  send the full model every round; OurMethod skips drift-flagged layers.

Output:
  - Printed report
  - multiseed_analysis_report.md saved to disk
"""

import csv
from pathlib import Path
from statistics import mean, stdev

NUM_CLIENTS  = 20
NUM_ROUNDS   = 200
DRIFT_ROUND  = 100
METHODS      = ['FedAvg', 'Flash', 'AdaptiveFedAvg', 'OurMethod']
SEEDS        = [1, 2]   # only these have per-client columns

# Drift groups (mirrors script)
GROUP_A = [i for i in range(NUM_CLIENTS) if i % 10 < 3]    # 1<->2
GROUP_B = [i for i in range(NUM_CLIENTS) if 3 <= i % 10 < 6]  # 3<->4
GROUP_C = [i for i in range(NUM_CLIENTS) if i % 10 >= 6]   # 5<->6

# Param counts per layer (CifarCNN)
LAYER_PARAMS = {
    'layer1':     1216,  # Conv 3->16, 5x5  + bias
    'layer2':    12832,  # Conv 16->32, 5x5 + bias
    'layer3':    18496,  # Conv 32->64, 3x3 + bias
    'layer4':    73856,  # Linear 576->128  + bias
    'classifier': 1290,  # Linear 128->10   + bias
}
TOTAL_PARAMS = sum(LAYER_PARAMS.values())
BYTES_PER_FLOAT = 4
assert TOTAL_PARAMS == 107690, f"Param count mismatch: {TOTAL_PARAMS}"


# ============================================================
# Data loading
# ============================================================

RUN_PATHS = {
    0: Path('runs/2026-05-26-augfix/seed0'),
    1: Path('runs/2026-05-27-multiseed/seed1'),
    2: Path('runs/2026-05-27-multiseed/seed2'),
}


def load_method_csv(method, seed):
    path = RUN_PATHS[seed] / f'results_{method}.csv'
    if not path.exists():
        return None
    return list(csv.DictReader(path.open()))


def load_flags(seed):
    path = RUN_PATHS[seed] / 'results_OurMethod_flags.csv'
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    out = {}
    for r in rows:
        rnd = int(r['round'])
        out[rnd] = {
            'l3': [int(x) for x in r['flagged_layer3_ids'].split(';') if x.strip()],
            'l4': [int(x) for x in r['flagged_layer4_ids'].split(';') if x.strip()],
        }
    return out


# ============================================================
# STEP 1: Per-client local-accuracy analysis
# ============================================================

def per_client_metrics(rows):
    """Return {client_id: (pre_acc, drift_round_acc, stable_acc)}."""
    out = {}
    for cid in range(NUM_CLIENTS):
        col = f'local_c{cid:02d}'
        vals = [float(r[col]) for r in rows]
        pre   = mean(vals[DRIFT_ROUND-11:DRIFT_ROUND])
        atdrift = vals[DRIFT_ROUND]
        stable = mean(vals[-10:])
        out[cid] = (pre, atdrift, stable)
    return out


def step1_per_client():
    print("="*100)
    print("STEP 1: PER-CLIENT LOCAL ACCURACY (seeds 1 + 2)")
    print("="*100)
    print()

    # Aggregate per-client metrics across seeds and methods
    # data[method][cid] = list of (pre, atdrift, stable) tuples across seeds
    data = {m: {cid: [] for cid in range(NUM_CLIENTS)} for m in METHODS}

    for m in METHODS:
        for s in SEEDS:
            rows = load_method_csv(m, s)
            if rows is None:
                continue
            metrics = per_client_metrics(rows)
            for cid, vals in metrics.items():
                data[m][cid].append(vals)

    # 1.1 — Group-level mean recovery (drifted clients only — all clients are drifted)
    print(">> Per-DRIFT-GROUP mean accuracy (seeds 1+2 averaged):")
    print()
    print(f"  {'Method':<18}  {'Group A (1<->2)':<22}  {'Group B (3<->4)':<22}  {'Group C (5<->6)':<22}")
    print(f"  {'':<18}  {'pre / drift / stable':<22}  {'pre / drift / stable':<22}  {'pre / drift / stable':<22}")
    print("  " + "-"*92)

    for m in METHODS:
        row = f"  {m:<18}"
        for label, grp in [('A', GROUP_A), ('B', GROUP_B), ('C', GROUP_C)]:
            pre_vals    = [t[0] for cid in grp for t in data[m][cid]]
            drift_vals  = [t[1] for cid in grp for t in data[m][cid]]
            stable_vals = [t[2] for cid in grp for t in data[m][cid]]
            cell = f"{mean(pre_vals):.3f}/{mean(drift_vals):.3f}/{mean(stable_vals):.3f}"
            row += f"  {cell:<22}"
        print(row)
    print()

    # 1.2 - Per-client comparison: OurMethod vs FedAvg
    print(">> Per-client POST-DRIFT STABLE acc: OurMethod minus FedAvg")
    print("   (positive = OurMethod recovers this client BETTER post-drift)")
    print()
    diffs = []
    print(f"  {'Client':<8}{'Group':<8}{'FedAvg':>12}{'OurMethod':>12}{'Delta':>12}")
    print("  " + "-"*52)
    for cid in range(NUM_CLIENTS):
        grp = 'A' if cid in GROUP_A else ('B' if cid in GROUP_B else 'C')
        fa_vals = [t[2] for t in data['FedAvg'][cid]]
        om_vals = [t[2] for t in data['OurMethod'][cid]]
        fa_mean = mean(fa_vals)
        om_mean = mean(om_vals)
        delta = om_mean - fa_mean
        diffs.append((cid, grp, fa_mean, om_mean, delta))
        marker = "  <-- OurMethod wins" if delta > 0.01 else ("  <-- FedAvg wins" if delta < -0.01 else "")
        print(f"  {cid:<8}{grp:<8}{fa_mean:>12.4f}{om_mean:>12.4f}{delta:>+12.4f}{marker}")
    print()

    om_wins = sum(1 for d in diffs if d[4] > 0.01)
    fa_wins = sum(1 for d in diffs if d[4] < -0.01)
    ties    = NUM_CLIENTS - om_wins - fa_wins
    avg_delta = mean([d[4] for d in diffs])
    print(f"  Summary: OurMethod wins {om_wins} clients, FedAvg wins {fa_wins}, ties (|Δ|<0.01) = {ties}")
    print(f"  Mean per-client delta: {avg_delta:+.4f} ({avg_delta*100:+.2f}pp)")
    print()

    # 1.3 — How do the FLAGGED clients fare vs UNFLAGGED?
    # Use seed 1's flag CSV — which clients ever flagged?
    flags_s1 = load_flags(1)
    flagged_ever = set()
    if flags_s1:
        for rnd_data in flags_s1.values():
            flagged_ever.update(rnd_data['l3'])
            flagged_ever.update(rnd_data['l4'])

    unflagged_ever = [cid for cid in range(NUM_CLIENTS) if cid not in flagged_ever]
    print(f">> Flagged-vs-unflagged: did OurMethod help the clients it flagged?")
    print(f"   Flagged-ever (seed 1): {sorted(flagged_ever)}  ({len(flagged_ever)} clients)")
    print(f"   Unflagged-ever        : {sorted(unflagged_ever)}  ({len(unflagged_ever)} clients)")
    print()
    for label, group in [('FLAGGED   ', flagged_ever), ('UNFLAGGED ', unflagged_ever)]:
        fa_pre  = mean([t[0] for cid in group for t in data['FedAvg'][cid]])
        fa_stb  = mean([t[2] for cid in group for t in data['FedAvg'][cid]])
        om_pre  = mean([t[0] for cid in group for t in data['OurMethod'][cid]])
        om_stb  = mean([t[2] for cid in group for t in data['OurMethod'][cid]])
        d_pre = om_pre - fa_pre
        d_stb = om_stb - fa_stb
        print(f"  {label}  FedAvg pre/stable: {fa_pre:.3f}/{fa_stb:.3f}  "
              f"OurMethod: {om_pre:.3f}/{om_stb:.3f}  "
              f"deltas: {d_pre:+.3f}/{d_stb:+.3f}")
    print()


# ============================================================
# STEP 2: Communication-cost accounting (BONUS — not primary)
# ============================================================

def step2_communication():
    print("="*100)
    print("STEP 2: COMMUNICATION COST  (bonus metric — secondary to accuracy)")
    print("="*100)
    print()

    fedavg_bytes_total = NUM_ROUNDS * NUM_CLIENTS * TOTAL_PARAMS * BYTES_PER_FLOAT
    print(f"  Model size: {TOTAL_PARAMS:,} params  ({TOTAL_PARAMS * BYTES_PER_FLOAT / 1024:.1f} KiB / client / round)")
    print(f"  Full upload per round (20 clients): {NUM_CLIENTS * TOTAL_PARAMS * BYTES_PER_FLOAT / 1024:.1f} KiB")
    print(f"  Full run total (200 rounds, all clients): {fedavg_bytes_total / 1024 / 1024:.1f} MiB")
    print()

    # FedAvg / Flash / Adaptive all upload full model every round
    print("  Baseline methods (FedAvg, Flash, AdaptiveFedAvg): all upload the FULL model")
    print(f"    every round from every client => {fedavg_bytes_total / 1024 / 1024:.1f} MiB total per run")
    print()

    # OurMethod: skip flagged layers
    print("  OurMethod: skips flagged layers per (client, round). Per-seed breakdown:")
    print()
    print(f"  {'Seed':<6}{'Bytes uploaded':>20}{'vs FedAvg':>14}{'Saved':>12}{'Skips':>10}")
    print("  " + "-"*60)

    for s in SEEDS:
        flags = load_flags(s)
        if flags is None:
            print(f"  {s:<6}  (no flag CSV)")
            continue

        bytes_total_om = 0
        layer_skips = 0
        for rnd in range(NUM_ROUNDS):
            f = flags.get(rnd, {'l3': [], 'l4': []})
            for cid in range(NUM_CLIENTS):
                # OurClient always uploads layer1, layer2, classifier
                params = LAYER_PARAMS['layer1'] + LAYER_PARAMS['layer2'] + LAYER_PARAMS['classifier']
                # layer3 only if NOT flagged
                if cid not in f['l3']:
                    params += LAYER_PARAMS['layer3']
                else:
                    layer_skips += 1
                # layer4 only if NOT flagged
                if cid not in f['l4']:
                    params += LAYER_PARAMS['layer4']
                else:
                    layer_skips += 1
                bytes_total_om += params * BYTES_PER_FLOAT

        saved = fedavg_bytes_total - bytes_total_om
        pct = 100.0 * bytes_total_om / fedavg_bytes_total
        saved_pct = 100.0 - pct
        print(f"  {s:<6}"
              f"{bytes_total_om/1024/1024:>17.2f} MiB"
              f"{pct:>13.2f}%"
              f"{saved/1024/1024:>9.2f} MiB"
              f"{layer_skips:>10}")

    print()
    print("  Interpretation: total savings are modest (~few MiB) because flagging")
    print("  is concentrated in the ~14-round drift window. The structural property")
    print("  is what matters: OurMethod transmits STRICTLY LESS during drift, and")
    print("  the SAME as FedAvg in steady state — never more.")
    print()


# ============================================================
# Run + save report
# ============================================================

if __name__ == '__main__':
    import io
    import sys

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    step1_per_client()
    step2_communication()
    sys.stdout = old_stdout

    report = buf.getvalue()
    print(report)

    with open('multiseed_analysis_report.txt', 'w') as f:
        f.write(report)
    print("\nReport saved: multiseed_analysis_report.txt")
