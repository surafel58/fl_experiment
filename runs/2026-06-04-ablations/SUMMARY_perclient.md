# Per-client ablation analysis — eval-path gate result

**Branch:** `ablations`.
**Goal:** produce a per-client generalized-accuracy version of the ablation table, computed *identically* to the FedCCFA metric we just verified (FedAvg.py reproduction PASSED at 59.94%, inside ±1.5pp of paper target 60.96%).
**Outcome:** **Gate B with recompute impossible.** The FedCCFA-compatible per-client metric **cannot** be produced from the existing ablation artifacts. The closest existing column is computed by a different eval path and is **not** comparable to the verified FedCCFA metric. Details, evidence, and the secondary lens that *can* be reported are below.

---

## 1. Eval-path comparison (the gate)

### FedCCFA's verified metric — what we matched at 59.94%

From the FedCCFA source we just ran on the VM:

1. **Dataset construction.** `utils/gen_dataset.py:149` builds **four identical copies of the FULL test set** (10000 CIFAR-10 samples each):
   ```python
   global_test_sets = [ClientDataset(test_set, list(range(len(test_set))), transform=test_transform) for _ in range(4)]
   ```

2. **Drift application.** `utils/drift.py:23-42` (`sudden_drift`) applies label swaps to **both** the per-client train sets **and** the full-test-set copies:
   - `global_test_sets[1]`: labels 1↔2 swapped
   - `global_test_sets[2]`: labels 3↔4 swapped
   - `global_test_sets[3]`: labels 5↔6 swapped
   - Each client is assigned one `global_test_id ∈ {1, 2, 3}` based on `client.id % 10`, so 3 cohorts.

3. **Per-client eval.** `entities/base.py:65`:
   ```python
   def global_test(self, global_tests):
       return get_accuracy(self.model, global_tests[self.global_test_id], self.args["device"])
   ```
   `self.model` is the **global aggregated model** at the time of evaluation, because `server.send_params(clients)` was called just before `last_round_evaluate` in the training loop (this is the bug we found in the locally-added `FedAvg_baseline.py` — it doesn't call `send_params`, which inflated the metric by +2.76pp).

4. **Aggregation.** The published number is the mean across all clients of `client.global_test(global_test_sets)`.

**Summary of the verified eval path:** global aggregated model → evaluated on **full test set (10000 samples)** → labels in that test set drifted per the client's cohort swap → mean across 20 clients.

### Our harness's existing per-client columns (`local_cXX`)

From `all_experiments_optimized.py:361-364`:
```python
def evaluate_all_clients(model, client_y):
    return [local_evaluate_gpu(model, GPU_CLIENT_X[cid], client_y[cid])
            for cid in range(NUM_CLIENTS)]
```

- `model` = `gm` (the global aggregated model). ✅ Same as FedCCFA.
- `GPU_CLIENT_X[cid]` = each client's **training-set partition** under Dir(0.1) (~500 samples per client for 20 clients), **NOT** the full test set. ❌ Different from FedCCFA.
- `client_y[cid]` = each client's **training labels** with their cohort's label swap applied (the swap is mutated in-place by `apply_drift_event`).
- The harness comment at line 247-248 makes the absence of test-set drift explicit:
  > "The TEST_Y tensor is never passed to this function — only per-method client label dicts are. Test labels remain canonical."

**Summary of our `local_cXX` eval path:** global aggregated model → evaluated on each client's **training-set partition (~500 samples, heavily class-imbalanced under Dir(0.1))** → labels in that training set drifted per the client's swap → mean across 20 clients.

### The two differ in two places

| Aspect | FedCCFA verified | Our `local_cXX` |
|---|---|---|
| Model evaluated | global aggregated | global aggregated ✅ |
| Eval set | full test set (10000) | client training partition (~500) ❌ |
| Where the swap lives | test-set labels | training-set labels ❌ |
| Class coverage of eval set | all 10 classes per cohort | only the classes that client owns |
| Sample size per client | 10000 | ~500 |

**Verdict: GATE B.** Our existing per-client columns are NOT computed by the same eval path as the FedCCFA verified metric. Reporting `mean(local_cXX)` as if it were the FedCCFA per-client generalized accuracy would be wrong.

---

## 2. Can the FedCCFA metric be recomputed from existing artifacts?

To recompute the FedCCFA-correct metric for the ablations without re-running training, I would need at minimum:
- The round-199 **global model parameters** for each of the 5 variants (so I could evaluate each variant's final global model on the four full-test-set copies).
- The same data partitioning seed and the same cohort assignment so the per-client mean is over the right groups.

**Checked the run directory.** The ablation runs saved only the CSVs:
```
runs/2026-06-04-ablations/
├── all-layers/{results_OurMethod.csv, results_OurMethod_flags.csv}
├── baseline/{results_OurMethod.csv, results_OurMethod_flags.csv}
├── no-detection/{results_OurMethod.csv, results_OurMethod_flags.csv}
├── tau-low/{results_OurMethod.csv, results_OurMethod_flags.csv}
├── tau-high/{results_OurMethod.csv, results_OurMethod_flags.csv}
├── plots/{ablation_trajectories,ablation_metrics_bars,ablation_deltas}.png
└── SUMMARY.md
```

A whole-tree search for `*.pt`, `*.pth`, `*.ckpt` turns up no model checkpoints from any of these runs. The harness `all_experiments_optimized.py` does not save model state — it appends per-round metric rows to a `LiveCSV` and discards the model when the method finishes.

**Therefore: the FedCCFA-correct per-client metric cannot be computed from existing ablation artifacts without re-running training.** Per the task instructions ("no experiment re-runs"), I do not re-run, and I do not present a differently-computed metric as if it were the verified FedCCFA metric.

This is the **Gate B with recompute impossible** terminal state.

---

## 3. What I *can* report — a clearly-labeled secondary lens

The harness DID log two per-client quantities for each variant that, while NOT the FedCCFA verified metric, are still informative as a secondary lens. Both are stable-window means (last 10 rounds, mean over all 20 clients):

| Variant | global (canonical 10k test) | per-client local-data acc (NOT FedCCFA) | per-client hybrid acc (NOT FedCCFA) |
|---|---:|---:|---:|
| baseline | 0.5884 | 0.5783 | 0.5783 |
| no-detection | 0.5876 | 0.5807 | 0.5807 |
| all-layers | 0.5894 | 0.5824 | 0.5820 |
| tau-low | 0.5924 | 0.5798 | 0.5798 |
| tau-high | 0.5965 | 0.5821 | 0.5821 |

**Deltas vs baseline (pp):**

| Variant | Δ global | Δ local-data | Δ hybrid |
|---|---:|---:|---:|
| no-detection | −0.09 | +0.24 | +0.24 |
| all-layers   | +0.10 | +0.41 | +0.38 |
| tau-low      | +0.40 | +0.15 | +0.15 |
| tau-high     | +0.80 | +0.38 | +0.38 |

**What these columns mean:**
- `local-data`: per-client accuracy of the **global aggregated model** evaluated on each **client's training partition** with that client's label-swap applied to training labels. Eval set is ~500 samples per client, heavily class-imbalanced under Dir(0.1).
- `hybrid`: per-client accuracy of the model **each client actually uses** (for OurMethod this folds in whichever layers that client kept local vs synced). For most clients × rounds this is identical or nearly identical to `local-data`, which is why the columns agree to within ≤0.04pp.

**Why this is NOT a substitute for the FedCCFA metric.** Different eval set, different swap target, vastly different sample size and class coverage. Several deltas even point in different directions between the canonical global metric and the local-data lens (e.g. `no-detection` is global −0.09pp but local-data +0.24pp). This divergence is the whole reason a proper per-client generalized-accuracy metric on the full test set would be informative — and is exactly why I will not pretend the local-data lens is that metric.

---

## 4. Single-seed and noise floor

Every number in this file is a single-seed (seed 0) result. Earlier 3-seed studies on this setup showed 1-σ seed variance on stable post-drift accuracy on the order of 0.5–1.5pp. **All deltas in the table above (max |Δ| = 0.80pp) are inside that noise floor.** Treat the table as ranking with substantial uncertainty.

---

## 5. What to do next (if a per-client metric is needed for the paper)

The cheapest path forward is one or more of:
1. **Save round-199 global model checkpoints** in the next ablation re-run (one `torch.save` call per variant at end of training; ~10 MB each). With those, an offline script can compute the FedCCFA-compatible per-client metric without re-training.
2. **Add the FedCCFA metric inline** to the harness: instantiate the four full-test-set copies once at startup, mutate their labels at the drift round in lockstep with the per-client swaps, and add a `per_client_gen_acc` column to the CSV. ~30 lines of code, no slower than the current eval since it only runs once per round.
3. **Single-seed verification first**: do (1) for ONE variant first, port the metric, sanity-check against FedCCFA-FedAvg's 59.94% reference, then re-run the 5 ablations with the new column.

Option (2) is the right long-term answer. Option (1) is the fastest patch on the existing CSVs.

**This is a decision for after the morning meeting — not done autonomously.**
