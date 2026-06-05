# Ablation study — combined honest summary (global + secondary per-client lens)

**Branch:** `ablations` (off main). Single seed (seed 0). Setup: CIFAR-10, Dir(0.1), 20 clients, 200 rounds, 5 local epochs, batch 64, single sudden drift at round 100. OurMethod defaults: EMA α=0.3, warmup=10, drift-layers L3+L4, τ=1.4.

This file is the meeting-ready combined view. It pairs the canonical global-accuracy ablation table (rigorous, what `SUMMARY.md` already had) with a secondary per-client *local-data* lens (NOT the FedCCFA verified metric), and is explicit about what cannot be computed without re-running.

---

## 1. Gate-decision recap (full details: `SUMMARY_perclient.md`)

I tried to produce a per-client generalized-accuracy version of the ablation table computed *identically* to the FedCCFA per-client metric we just verified on a clean VM (FedAvg.py reproduction PASSED at 59.94% vs paper target 60.96%, inside ±1.5pp).

**Result: GATE B with recompute impossible.**

- The existing per-client columns in our CSVs (`local_cXX`) evaluate the global model on each client's **training partition** (~500 samples per client, only the classes that client owns), not on the full test set. FedCCFA's verified metric evaluates the global model on four copies of the **full 10000-sample test set** with cohort-specific label swaps applied to the test labels. Different eval set, different swap target — these are not the same metric.
- The ablation runs saved CSVs only, not model checkpoints. Without round-199 global-model parameters, the FedCCFA-correct metric cannot be computed offline. Re-running is forbidden by the task.

→ **No FedCCFA-compatible per-client ablation table appears in this summary.** Below I report (a) the canonical global table (rigorous), and (b) a secondary per-client local-data lens clearly labeled as NOT the FedCCFA metric.

---

## 2. Global-accuracy ablation table (rigorous, canonical 10k test set, undrifted)

Numbers reproduced from `SUMMARY.md` — same single-seed data, same formulas as `summarize_method`.

| Variant | Pre-drift acc | Dip | Post-drift stable | Δ-pre vs baseline | Δ-stable vs baseline | Recovery |
|---|---:|---:|---:|---:|---:|---|
| baseline | **0.7026** | **0.1196** | **0.5884** | 0 (ref) | 0 (ref) | not recovered |
| no-detection | **0.7071** | **0.1265** | **0.5876** | +0.45pp | **−0.08pp** | not recovered |
| all-layers | **0.7047** | **0.1296** | **0.5894** | +0.21pp | **+0.10pp** | not recovered |
| tau-low | **0.7077** | **0.0740** | **0.5924** | +0.51pp | **+0.40pp** | not recovered |
| tau-high | **0.7057** | **0.1223** | **0.5965** | +0.31pp | **+0.81pp** | not recovered |

- **Pre-drift acc:** mean global acc over rounds [89, 99].
- **Dip:** Pre minus min(acc) over rounds [100, 109].
- **Post-drift stable:** mean global acc over last 10 rounds.
- Test set is the canonical CIFAR-10 test set (10000 samples), labels NOT drifted.

Plots: `plots/ablation_trajectories.png`, `plots/ablation_metrics_bars.png`, `plots/ablation_deltas.png`.

---

## 3. Secondary per-client local-data lens — NOT the FedCCFA metric

> **Read this label first.** The two columns marked "(NOT FedCCFA)" below are computed from `local_cXX` / `hybrid_cXX` columns in the existing CSVs. They evaluate the global model (or each client's actual-use model) on **each client's TRAINING-set partition** with that client's label swap. They are NOT comparable to the FedCCFA per-client generalized accuracy metric, which uses the full test set. Treat them as a different lens, not as a substitute. See `SUMMARY_perclient.md` Section 1 for the code-cited eval-path comparison.

Stable means over the last 10 rounds, then averaged across all 20 clients.

| Variant | global (canonical 10k test) | per-client local-data (NOT FedCCFA) | per-client hybrid (NOT FedCCFA) |
|---|---:|---:|---:|
| baseline | 0.5884 | 0.5783 | 0.5783 |
| no-detection | 0.5876 | 0.5807 | 0.5807 |
| all-layers | 0.5894 | 0.5824 | 0.5820 |
| tau-low | 0.5924 | 0.5798 | 0.5798 |
| tau-high | 0.5965 | 0.5821 | 0.5821 |

**Deltas vs baseline (pp):**

| Variant | Δ global | Δ local-data (NOT FedCCFA) | Δ hybrid (NOT FedCCFA) |
|---|---:|---:|---:|
| no-detection | −0.09 | +0.24 | +0.24 |
| all-layers   | +0.10 | +0.41 | +0.38 |
| tau-low      | +0.40 | +0.15 | +0.15 |
| tau-high     | +0.80 | +0.38 | +0.38 |

Note that the local-data lens and the global lens disagree in both magnitude and direction for several variants (e.g. `no-detection` is global −0.09pp but local-data +0.24pp). That divergence is exactly the kind of thing a proper per-client generalized-accuracy metric on the full test set would clarify — and is exactly why I will not pretend the local-data lens is that metric.

(`hybrid` and `local-data` agree to ≤0.04pp because for most client × round combinations the "model the client actually uses" equals the global model — OurMethod's per-client layer choices only diverge from FedAvg in narrow circumstances after drift.)

---

## 4. Caveats — read before drawing conclusions

1. **Single seed.** All numbers in both tables are seed 0. Prior 3-seed studies on this setup measured 1-σ seed variance on stable post-drift accuracy at ~0.5–1.5pp. The largest delta in either table is 0.81pp (`tau-high` global). **None of the deltas in either table are outside the established single-seed noise floor.** A multi-seed (≥3) replication is the next step.

2. **Local-fitting sensitivity is unverified for the ablations.** When we ran the FedCCFA verification track, we found that `FedAvg_baseline.py` (which evaluates a locally-fine-tuned model) scored +2.76pp above the published FedAvg baseline (63.72% vs 60.96%). This proved that *per-client* metrics on this setup are highly sensitive to local fitting. **For OurMethod's ablations we have NOT yet run the proper control "FedAvg + one local epoch right before eval."** Without that control, any per-client gain (canonical *or* hypothetical FedCCFA-style) cannot be cleanly attributed to OurMethod's selective per-layer mechanism vs trivial last-step personalization. The local-data lens in Section 3 is even more exposed to this effect than the canonical global lens because it evaluates on each client's training partition — where local fitting helps most.

3. **The verified FedCCFA metric was at a different operating point.** The FedAvg verification ran at Dir(0.5), the ablations run at Dir(0.1). The Dir(0.5) verification establishes that the harness reproduces published FedAvg, not that the per-client metric numbers transfer to Dir(0.1). When the FedCCFA-style per-client metric is eventually computed for OurMethod's ablations, the proper FedAvg reference for *that* operating point would also need to be computed at Dir(0.1), single drift, 20 clients — a quantity we do not have either.

4. **OurMethod's headline knob effect (`tau-low`) is still on the dip, not on stable accuracy.** Across both tables, `tau-low` (τ=1.2) shows its strongest signal in the global *dip* (0.0740 vs baseline 0.1196 = −4.56pp smaller dip, the largest effect in the study), with stable accuracy unchanged on every lens. The mechanism story is "softer landing at drift", not "higher stable accuracy". A per-client generalized-accuracy metric would mostly affect how we measure the stable phase; the dip story is unchanged.

---

## 5. Plain-language reading (4–5 sentences)

The canonical global-accuracy ablation table is intact and rigorous: `tau-low` produces a meaningfully smaller drift-dip (−4.56pp vs baseline) while leaving stable accuracy basically unchanged, and that is the cleanest result in the study. Every other knob (`no-detection` at τ=∞, `all-layers`, `tau-high` at τ=1.6) moves the global stable accuracy by less than 1pp, which is inside the single-seed noise floor established by prior 3-seed runs. The secondary per-client local-data lens (NOT the FedCCFA metric) tells a slightly different story — several variants flip sign or change magnitude under it — but at deltas this small (max 0.41pp) and on training-partition data, that's not informative without a multi-seed replication and without the FedAvg+1-local-epoch control. **The FedCCFA-compatible per-client generalized-accuracy metric is not in this summary** because the ablation runs did not save model checkpoints; the next iteration should add either inline FedCCFA-metric logging or end-of-run checkpoint saving so this lens can be computed cleanly. Bottom line: the global story is unchanged (`tau-low` reduces the dip, other knobs are noise at n=1), the per-client metric question is **deferred** until the harness supports it, and no per-client claim in this ablation set should be made until at least a FedAvg+1-local-epoch control is also run.
