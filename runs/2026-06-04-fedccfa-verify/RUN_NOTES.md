# 2026-06-04 — FedCCFA paper-metric verification

**Branch:** `perclient-metric` (off main). Do not merge.
**Status:** Stage 1 — FedAvg verification PASSED. FedCCFA verification (Run C) not yet started.

## Goal

Verify FedCCFA's "per-client generalized accuracy" metric by reproducing
their Table 2 CIFAR-10 sudden-drift result with their own code, before
porting the metric into our harness.

- FedCCFA paper target (CIFAR-10, Dir(0.5), 20 clients, full participation, sudden drift @ 100, 200 rounds):
  - FedAvg ≈ 60.96%
  - FedCCFA ≈ 73.00%
- Pass criteria:
  - FedAvg within ±1.5pp → [59.46, 62.46]
  - FedCCFA within ±2pp → [71.00, 75.00]; hard fail if <70%.

## Setup

- VM: GCP L4 (g2-standard-4)
- Code: FedCCFA repo at HEAD (commit `647d8b0`), restored published files
  `configs/FedAvg.yaml` and `methods/FedAvg.py` from HEAD (they had been
  locally deleted in our working tree at some point in past thesis work).
- venv on VM: isolated, torch 2.2.2+cu121, torchvision 0.17.2 (per their
  requirements.txt with relaxed PyYAML).
- Seed: 0 (FedCCFA's default in both YAMLs)

## Runs in this folder

### Run A — `FedAvg_baseline.py` (LOCALLY-ADDED, not FedCCFA-published)

**File:** `CIFAR10_sudden_FedAvg_20260604071909842619.csv` + `fedavg_verify.log`

This run was launched against `methods/FedAvg_baseline.py` and
`configs/FedAvg_baseline.yaml` — files that are present in our working
tree but are NOT in FedCCFA's HEAD (they were locally added during prior
thesis work). The training loop in this script does NOT call
`server.send_params(clients)` before `last_round_evaluate`, so each
client's `client.model` at evaluation time is the locally-trained model
from round 199, NOT the aggregated global model.

**Result:** mean global_accuracy = **0.6372** (63.72%)

**Verdict against the published-FedAvg target (60.96%, ±1.5pp = [59.46, 62.46]):**
**FAIL by +2.76pp above the band.**

**Why this matters for our thesis (advisor's call):** this is effectively
"FedAvg + one local SGD epoch per client right before evaluation" — a
common personalization trick. The +2.76pp lift over published FedAvg
shows the per-client generalized-accuracy metric is **highly sensitive
to local retention**, which is exactly the kind of effect OurMethod's
selective per-layer adaptation produces. We KEEP this number as a
critical baseline ("naive local fine-tuning") against which OurMethod
must demonstrate a meaningful per-layer-adaptation gain. If OurMethod
only matches naive-local-fine-tune, the per-layer mechanism adds nothing
beyond trivial personalization.

### Run B — `FedAvg.py` (FedCCFA-published) — PASSED ✅

**File:** `CIFAR10_sudden_FedAvg_20260604181603436012.csv` + `fedavg_pub_verify.log`

Restored `methods/FedAvg.py` and `configs/FedAvg.yaml` from FedCCFA HEAD,
edited the YAML's three setup fields to match the verification target:
- `client_num: 100 → 20`
- `sample_ratio: 0.2 → 1.0`
- `drift_pattern: false → sudden`
(α, seed, all other hyperparams unchanged.)

This run uses the published training loop, which DOES call
`server.send_params(clients)` before `last_round_evaluate`, so each
client's evaluation uses the AGGREGATED GLOBAL model — the apples-to-apples
counterpart to Table 2's FedAvg number.

**Result:** mean global_accuracy = **0.5994** (59.94%)

**Verdict against the published-FedAvg target (60.96%, ±1.5pp = [59.46, 62.46]):**
**PASS by −1.02pp below target, inside the band.**

The FedCCFA harness reproduces the paper's FedAvg baseline at our
verification settings. The per-client generalized-accuracy metric pipeline
in their code is therefore trustworthy for the next step (FedCCFA.py).

### Run C — `FedCCFA.py` (FedCCFA-published, not yet started)

Gated on Run B passing — which it has. Next action: launch FedCCFA.py
against the same YAML setup (Dir(0.5), 20 clients, full participation,
sudden drift @ 100) and verify the result lands in [71.00, 75.00].

## Naming clarification

The two CSVs here are for two distinct FedAvg variants:
- `CIFAR10_sudden_FedAvg_20260604181603436012.csv` — Run B, published
  `FedAvg.py` (global-eval). The PASS run.
- `FedAvg_baseline_LOCAL_FINETUNED_seed0.csv` — Run A, the
  locally-added `FedAvg_baseline.py` variant (one-extra-local-epoch).
  Renamed from the original output to make the role explicit.
