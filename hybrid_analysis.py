"""
hybrid_analysis.py — does the hybrid model help the flagged clients?

Reads the new OurMethod CSVs (which now include hybrid_cXX columns next to
local_cXX) and answers three questions:

  Q1 — Hybrid lift vs OurMethod's own global model
       For each flagged client during the flag window, how much more accurate
       is the HYBRID (what they actually use) than the global (federation's
       view of them)? Positive = the selective_sync adaptation is helping.

  Q2 — Hybrid vs FedAvg for the same clients (seeds 1, 2)
       Compare OurMethod's hybrid acc vs FedAvg's local_cXX (also "what FedAvg
       gives this client") on the flagged subset. Positive = OurMethod's full
       mechanism beats FedAvg for the clients OurMethod identifies as drifted.

  Q3 — Drift-window global acc (recap from drift_window_analysis.py).

Reads:
  runs/2026-05-28-hybrid/seed{0,1,2}/results_OurMethod.csv  (hybrid columns)
  runs/2026-05-28-hybrid/seed{0,1,2}/results_OurMethod_flags.csv
  runs/2026-05-27-multiseed/seed{1,2}/results_FedAvg.csv  (for cross comparison)
"""

import csv
from pathlib import Path
from statistics import mean, stdev

NUM_CLIENTS  = 20
DRIFT_ROUND  = 100
WINDOW_END   = 115

HYBRID_RUN  = Path('runs/2026-05-28-hybrid')
MULTISEED   = Path('runs/2026-05-27-multiseed')

SEEDS = [0, 1, 2]


def load_om(seed):
    p = HYBRID_RUN / f'seed{seed}' / 'results_OurMethod.csv'
    if not p.exists(): return None
    return list(csv.DictReader(p.open()))


def load_flags(seed):
    p = HYBRID_RUN / f'seed{seed}' / 'results_OurMethod_flags.csv'
    if not p.exists(): return None
    return list(csv.DictReader(p.open()))


def load_fedavg_seed(seed):
    """Per-seed FedAvg with local_cXX (only exists for seeds 1, 2)."""
    p = MULTISEED / f'seed{seed}' / 'results_FedAvg.csv'
    if not p.exists(): return None
    return list(csv.DictReader(p.open()))


def flagged_set(frows):
    s = set()
    for r in frows:
        for col in ('flagged_layer3_ids', 'flagged_layer4_ids'):
            if r[col].strip():
                s.update(int(x) for x in r[col].split(';'))
    return sorted(s)


def per_round_flagged(frows):
    """Map round -> set of client_ids flagged in that round."""
    out = {}
    for r in frows:
        rnd = int(r['round'])
        ids = set()
        for col in ('flagged_layer3_ids', 'flagged_layer4_ids'):
            if r[col].strip():
                ids.update(int(x) for x in r[col].split(';'))
        out[rnd] = ids
    return out


# ============================================================
# Q1 — Hybrid lift vs OurMethod's own global (per seed)
# ============================================================

def q1_hybrid_lift():
    print("=" * 100)
    print("Q1 — HYBRID vs GLOBAL on flagged clients (OurMethod internal)")
    print("=" * 100)
    print(f"  For each (seed, round in 100..{WINDOW_END}), for the clients flagged")
    print(f"  in THAT round, compute: mean(hybrid_cXX) - mean(local_cXX)")
    print(f"  Positive = the hybrid (what clients use) beats the global (what we logged before).")
    print()
    print(f"  {'Seed':<6}{'Mean hyb-lift on flagged (per round, mean over window)':>60}")
    print("  " + "-"*68)

    per_seed_lifts = {}
    for s in SEEDS:
        om = load_om(s)
        fr = load_flags(s)
        if not om or not fr: continue
        flagged_at = per_round_flagged(fr)
        lifts = []
        for r in om[DRIFT_ROUND:WINDOW_END + 1]:
            rnd = int(r['round'])
            ids = flagged_at.get(rnd, set())
            if not ids: continue
            hybrid = mean([float(r[f'hybrid_c{c:02d}']) for c in ids])
            local  = mean([float(r[f'local_c{c:02d}'])  for c in ids])
            lifts.append(hybrid - local)
        if lifts:
            per_seed_lifts[s] = lifts
            print(f"  {s:<6}{mean(lifts):>50.4f}  ({len(lifts)} flagged rounds in window)")

    print()
    if per_seed_lifts:
        all_lifts = [v for L in per_seed_lifts.values() for v in L]
        print(f"  Across all seeds:")
        print(f"    Mean hyb-lift per flagged round = {mean(all_lifts):+.4f}")
        print(f"    Max hyb-lift seen              = {max(all_lifts):+.4f}")
        print(f"    Min hyb-lift seen              = {min(all_lifts):+.4f}")
        print(f"    N (flagged rounds total)        = {len(all_lifts)}")
    print()


# ============================================================
# Q2 — OurMethod hybrid vs FedAvg local for same clients (seeds 1, 2)
# ============================================================

def q2_vs_fedavg():
    print("=" * 100)
    print("Q2 — OurMethod HYBRID vs FedAvg LOCAL on same flagged clients (seeds 1, 2)")
    print("=" * 100)
    print(f"  For each seed in {{1, 2}}, for each round in 100..{WINDOW_END} when any")
    print(f"  client was flagged: compare OurMethod's hybrid_cXX (what the client")
    print(f"  actually uses) vs FedAvg's local_cXX (what FedAvg gives the client).")
    print()
    print(f"  {'Seed':<6}{'Mean hybrid':>14}{'Mean FedAvg':>14}{'Δ':>10}{'#rounds':>10}")
    print("  " + "-"*54)

    for s in [1, 2]:
        om = load_om(s); fr = load_flags(s); fa = load_fedavg_seed(s)
        if not (om and fr and fa): continue
        flagged_at = per_round_flagged(fr)
        h_means, f_means = [], []
        n = 0
        for om_r, fa_r in zip(om[DRIFT_ROUND:WINDOW_END + 1],
                               fa[DRIFT_ROUND:WINDOW_END + 1]):
            rnd = int(om_r['round'])
            ids = flagged_at.get(rnd, set())
            if not ids: continue
            h = mean([float(om_r[f'hybrid_c{c:02d}']) for c in ids])
            f = mean([float(fa_r[f'local_c{c:02d}'])  for c in ids])
            h_means.append(h); f_means.append(f)
            n += 1
        if h_means:
            print(f"  {s:<6}{mean(h_means):>14.4f}{mean(f_means):>14.4f}"
                  f"{mean(h_means)-mean(f_means):>+10.4f}{n:>10}")
    print()


# ============================================================
# Q3 — Global drift-window recap (same as before, sanity check)
# ============================================================

def q3_recap():
    print("=" * 100)
    print("Q3 — OurMethod drift-window global acc recap (with hybrid version of code)")
    print("=" * 100)

    metrics = []
    for s in SEEDS:
        om = load_om(s)
        if not om: continue
        accs = [float(r['global_acc']) for r in om]
        pre = mean(accs[DRIFT_ROUND-11:DRIFT_ROUND])
        dip_min = min(accs[DRIFT_ROUND:WINDOW_END+1])
        win_mean = mean(accs[DRIFT_ROUND:WINDOW_END+1])
        stable = mean(accs[-10:])
        metrics.append((s, pre, dip_min, win_mean, stable))

    print(f"  {'Seed':<6}{'pre':>10}{'min':>10}{'mean(100-115)':>16}{'stable(190-199)':>17}")
    print("  " + "-"*60)
    for s, p, m, w, st in metrics:
        print(f"  {s:<6}{p:>10.4f}{m:>10.4f}{w:>16.4f}{st:>17.4f}")
    if len(metrics) >= 2:
        pres   = [m[1] for m in metrics]
        mins   = [m[2] for m in metrics]
        windows = [m[3] for m in metrics]
        stables = [m[4] for m in metrics]
        print(f"  mean  {mean(pres):>10.4f}{mean(mins):>10.4f}{mean(windows):>16.4f}{mean(stables):>17.4f}")
        if len(metrics) >= 3:
            print(f"  std   {stdev(pres):>10.4f}{stdev(mins):>10.4f}{stdev(windows):>16.4f}{stdev(stables):>17.4f}")
    print()


if __name__ == '__main__':
    import io, sys
    buf = io.StringIO()
    old = sys.stdout; sys.stdout = buf
    q3_recap()
    q1_hybrid_lift()
    q2_vs_fedavg()
    sys.stdout = old

    report = buf.getvalue()
    print(report)

    with open('hybrid_analysis_report.txt', 'w') as f:
        f.write(report)
    print("\nReport saved: hybrid_analysis_report.txt")
