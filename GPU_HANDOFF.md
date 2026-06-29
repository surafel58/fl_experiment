# GPU handoff — instructions for another Claude Code session

This is a shared GCP L4 VM. Read this fully before launching anything. Costs are real, transient failures are common, and prior sessions hit several specific traps.

---

## 1. The VM

| Field | Value |
|---|---|
| Name | `fl-experiment` |
| Zone | `northamerica-northeast2-b` |
| GPU | 1× NVIDIA L4 (24 GB) |
| Current state | **TERMINATED** (must be started before use) |
| User | `suraf` |
| SSH key | `~/.ssh/gcp_key` (Windows side: `C:\Users\suraf\.ssh\gcp_key`) |
| Cost | ~$0.70–1.00/hr when RUNNING. **Always stop when done.** |

**Do NOT use `gcloud compute ssh`** — it's broken on this Windows setup. Use plain `ssh` directly.

---

## 2. Start the VM

```bash
gcloud compute instances start fl-experiment --zone=northamerica-northeast2-b
```

Then get the external IP:
```bash
gcloud compute instances list --filter="name=fl-experiment" \
    --format="value(status,networkInterfaces[0].accessConfigs[0].natIP)"
```

**STOCKOUT is common** in this zone. If you get `resources not available`, poll with retry (60s intervals — the user is OK waiting up to ~30 min for capacity).

Example poll script (use Monitor tool):
```bash
until gcloud compute instances start fl-experiment --zone=northamerica-northeast2-b 2>&1 | grep -q "RUNNING\|done\."; do
  sleep 60
done
```

---

## 3. SSH

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i ~/.ssh/gcp_key suraf@<IP>
```

The IP changes per restart — always re-fetch it after starting.

---

## 4. Stop the VM (DO THIS WHEN DONE)

```bash
gcloud compute instances stop fl-experiment --zone=northamerica-northeast2-b
```

If you leave it running you're costing the user money for compute they're not using. **End every session with a stop.**

---

## 5. The harness

The VM has `~/all_experiments_optimized.py` pre-installed with PyTorch + CUDA + CIFAR-10 already downloaded. You upload your version with scp:

```bash
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i ~/.ssh/gcp_key \
    /path/to/your/all_experiments_optimized.py \
    suraf@<IP>:~/all_experiments_optimized.py
```

Multiple Claude sessions overwrite each other's harness. **Coordinate or work on separate branches.** The harness file lives in `Implementation/Experiment/fl_experiment/all_experiments_optimized.py` on the host side.

### Methods available (METHOD_REGISTRY)

```
1   FedAvg            (plain weighted averaging)
2   Flash             (Panchal et al. 2023; server-side Adam with drift correction)
3   AdaptiveFedAvg    (ported from FedCCFA)
4   OurMethod         (legacy; experimental)
5   FedAvgPlus1       (FedAvg + 1 local epoch — control for trivial finetune confound)
6   Saile             (FLTA 2024; per-client loss-EMA dynamic LR)
7   FlashNormTrigger  (V1, failed — slow-EMA normalized trigger)
8   FlashColdInit     (attribution control: Flash + cold-start init only)
9   FlashNormV2       (V2, failed — per-tensor B + warmup window)
11  FedAvgAdam        (action-ceiling control: Flash's action with delta_mom=0)
```

### CLI flags (full list)

```
--methods 1 2 4         method IDs to run (or 'all' = 1..6)
--seed N                random seed (default 0)
--rounds N              total rounds (default 200; use small for smoke)
--alpha-dir 0.5         Dirichlet alpha for partition (default 0.1)
--no-drift              clears DRIFT_SCHEDULE (static heterogeneity only)
--partial-cohorts A     only cohort A drifts (A/B/C)
--recurrent             use [100,150] drift schedule
--alternating-drift     [40,80,120,160] alternating drift
--aggressive-concept-drift     per-cohort full label permutation (ONLY on aggressive-concept-drift-test branch)
--covariate-drift              per-cohort image corruption (ONLY on covariate-drift-test branch)
--saile-init-lr 0.01    override Saile's initial LR
--adaptive-init-lr X    override AdaptiveFedAvg's initial LR
--flashnorm-beta-b X    override slow-EMA rate for FlashNormTrigger
--lr X                  override base LR for ALL methods (default 0.01)
--lr-boost-factor X     multiplier during boost window (default 1.0 = none)
--lr-boost-start N      boost window start round (default 100)
--lr-boost-end N        boost window end (exclusive, default 110)
--out-dir DIR           write CSVs to DIR (else write to cwd)
```

### Typical command

```bash
ssh -i ~/.ssh/gcp_key suraf@<IP> \
    "python3 -u all_experiments_optimized.py \
        --seed 0 --methods 1 --rounds 200 \
        --out-dir runs/your-experiment/baseline/"
```

CSVs land at `~/runs/your-experiment/baseline/results_FedAvg.csv` etc. with one row per round.

---

## 6. **Critical**: launching long-running jobs

Many prior sessions lost ~3 hours of compute to chain scripts dying mid-run. **The lesson:**

### DON'T do this (will die after first pair):
```bash
nohup bash my_chain.sh > log 2>&1 & disown
```
The chain bash gets SIGHUP'd somehow after the first sub-process finishes, even though it's nohup+disown'd. Cause unknown — possibly related to SSH connection lifecycle or session manager.

### DO this (true detached):
```bash
setsid nohup bash my_chain.sh < /dev/null > log 2>&1 & disown
```
The `setsid` creates a new session disconnected from SSH; `< /dev/null` cuts stdin. This survives.

### Or even better: launch each run directly (no chain script)

For a 2-run pair:
```bash
ssh -i ~/.ssh/gcp_key suraf@<IP> << 'EOF'
  setsid nohup python3 -u all_experiments_optimized.py --seed 0 --methods 1 \
      --out-dir runs/X/ < /dev/null > logs/X.log 2>&1 & disown
  sleep 3
  setsid nohup python3 -u all_experiments_optimized.py --seed 0 --methods 2 \
      --out-dir runs/Y/ < /dev/null > logs/Y.log 2>&1 & disown
EOF
```

Then use the Monitor tool to poll for completion:
```bash
until ! ssh -i ~/.ssh/gcp_key suraf@<IP> \
    "pgrep -f all_experiments_optimized.py >/dev/null" 2>/dev/null; do
  sleep 60
done
```

---

## 7. Resource limits + pairing

| Procs in parallel | GPU util | Per-proc slowdown | Notes |
|---:|---:|---:|---|
| 1 | ~50–80% | baseline | best |
| 2 | ~95% | ~1.2–1.5× | proven safe, used for most pairs |
| 3 | 100% | ~2–3× | works but slow; OK for smoke |
| 4+ | risk OOM | — | not recommended |

200-round runs take **~25–40 min per proc when paired 2-at-a-time** on an idle L4. Triple-share is ~50 min/proc.

GPU memory: each proc uses ~2.4 GB. 4 procs would fit memory but compute-bottleneck badly.

---

## 8. Coordinating with other sessions

**Check for existing procs before launching:**
```bash
ssh -i ~/.ssh/gcp_key suraf@<IP> "pgrep -af all_experiments_optimized.py"
```

If procs are running, you'll be sharing GPU. If you must run, do it in a way that doesn't OOM (cap to 1–2 additional procs on top of existing).

**Use distinct `--out-dir`s.** Don't clobber another session's output. Convention: `runs/YYYY-MM-DD-<your-experiment-name>/`.

**Don't reupload the harness mid-run.** Python loads the source once at startup — re-uploading doesn't affect running procs, but if your changes break the file syntactically, the next launch will fail.

---

## 9. Common transient failures (don't panic)

| Symptom | Cause | Fix |
|---|---|---|
| `Connection reset by peer` during scp/ssh | Network blip | Just retry |
| `STOCKOUT` on `instances start` | GCP capacity issue | Poll every 60s; usually resolves in 5–30 min |
| `Permission denied (publickey)` | Wrong key path | Confirm `~/.ssh/gcp_key` exists |
| Procs gone but no CSV completion message | Either OOM, NaN, or VM reboot — check log file tail |
| Chain script dies after first pair | Use `setsid` (see §6) |

---

## 10. Git state on host (where the harness lives)

Host directory: `d:\projects\Seminars in AI\Paper\Proposal\Second Semester\Implementation\Experiment\fl_experiment\`

Branches with stable harness + evidence (most recent first):
- `headroom-gate-fail` — oracle LR boost test (FAILED)
- `flashnorm-v2-failed` — V1/V2 trigger fixes (FAILED) + FlashColdInit + FedAvgAdam
- `flash-confounding-gate-pass` — Flash confounding gate evidence (PASSED, 3 seeds)
- `main` — production
- `aggressive-concept-drift-test` — has `--aggressive-concept-drift` flag (NOT on main)
- `covariate-drift-test` — has `--covariate-drift` flag (NOT on main)

**Plain commits, no Co-Authored-By trailer.** This is a hard rule from the user.

---

## 11. Conventions that matter for any robust claim

- **±1pp is the single-seed noise floor** on this benchmark. Anything within ±1pp at single seed is noise, not signal.
- **Multi-seed = seeds 0, 1, 2** (3 seeds is the minimum for a robust claim).
- **Per-seed reporting is mandatory.** A 3-seed mean isn't enough — you must show each seed's number. Direction must be consistent across all 3 seeds, and each seed's magnitude must exceed the noise floor.
- **Fair tuning matters.** A method tested with crippled hyperparameters doesn't count as falsified. Use published defaults from the original paper.
- **Sanity-check method behavior at its design point.** If a method designed for canonical Dir(0.1) drift doesn't tie FedAvg there, your tuning is broken.

---

## 12. What's already been gated (so you don't re-litigate)

| Direction | Verdict | Why |
|---|---|---|
| FedDrift-clustering (FedCCFA) | PAUSED | DBSCAN ARI 0.70 but actual ~= oracle on accuracy; clustering not load-bearing |
| Saile per-client trigger | NO-GO | Best-tuned Saile ties FedAvg across 5 regimes |
| Flash confound | PASS (gate); FAIL (method) | Spurious firing real; trigger fixable but action ceiling kills the fix |
| Drift-triggered FedAvg LR boost | FAIL | Client-side label-relearning is the bottleneck; no aggregator knob helps |
| Drift-type routing | NO-GO | Type recoverable but no action headroom |
| FSSL-DP | NO-GO | FSSL more DP-robust than supervised |

**Implication:** trigger-class methods on this benchmark are exhausted. If you're picking a new gate, target a different mechanism class.

---

## 13. End-of-session checklist

- [ ] All `pgrep all_experiments_optimized.py` returns nothing
- [ ] CSVs pulled to host (`scp -r suraf@<IP>:~/runs/your-experiment/ ./runs/`)
- [ ] `gcloud compute instances stop fl-experiment --zone=northamerica-northeast2-b`
- [ ] Status confirmed: `gcloud compute instances list --filter="name=fl-experiment" --format="value(status)"` should return `TERMINATED`
- [ ] Branch committed with plain commits (no Co-Authored-By)

If you skip the stop, the user is paying ~$0.85/hr for nothing.
