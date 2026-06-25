# Heterogeneity-normalized drift trigger — DESIGN SPEC (for review, not built)

**Goal.** Beat both Flash and FedAvg in no-drift Dir(0.1) and partial-coverage drift, at-worst-tie both in canonical / Dir(0.5) clean-drift regimes. Per-seed reporting, ≥3 seeds.

**Working name.** `FlashNormTrigger` (or `NormFlash`). Pick later.

---

## 1. The trigger

**Flash's current trigger (recap).** Server-side, per server-step:
```
agg_t        = sum_i w_i * (theta_i_t - theta_global_t)        # weighted mean of client updates
second_mom_t = beta2 * second_mom_{t-1} + (1 - beta2) * agg_t^2
delta_mom_t  = beta3 * delta_mom_{t-1} + (1 - beta3) * (agg_t^2 - second_mom_t)
agg_update_t = server_lr * first_mom_t / (sqrt(second_mom_t) - delta_mom_t + tau)
```
`delta_mom` is large when squared-aggregate diverges from its long-run EMA — i.e. when update magnitude *changes*. The intent is drift detection. The problem the gate exposed: in early training under static heterogeneity, `agg^2` is naturally large because the model is fitting a hard non-IID problem from cold init, NOT because anything drifted. Result: `delta_mom` over-fires in early rounds → −2.3pp permanent gap in no-drift Dir(0.1).

**Proposed trigger (new).** Replace `delta_mom` with a normalized version:
```
norm_delta_t = (agg_t^2 - second_mom_t) / (B_t + eps)
delta_mom_t  = beta3 * delta_mom_{t-1} + (1 - beta3) * norm_delta_t
```
where `B_t` is a **per-client heterogeneity baseline** (see §2). The action stays identical: `agg_update = server_lr * first_mom / (sqrt(second_mom) - delta_mom + tau)`.

**Why this should work.** Under static heterogeneity, the baseline `B` and the current `agg^2` both grow roughly proportionally — so `norm_delta` stays near zero. Under real drift, `agg^2` spikes faster than `B` updates → `norm_delta` fires. Spurious firing on heterogeneity is suppressed; firing on temporal change in the distribution survives.

**Decision required.** Should `norm_delta` be computed elementwise (per-parameter ratio) and then summarized into a scalar for the divisor scaling, or computed only on the scalar `||agg^2 - second_mom||_2`?
- **Elementwise** preserves Flash's per-parameter Adam-like geometry — recommended; this matches Flash's contract that the divisor is per-parameter.
- **Scalar-norm** is simpler but loses per-parameter resolution and is a strictly weaker change.

Recommendation: elementwise (do not break Flash's per-parameter mechanic).

---

## 2. Baseline window: fixed warmup vs slow-updating EMA

The whole point of normalization is `B` shouldn't follow `agg^2` quickly enough to absorb real drift, but should match `agg^2` under static heterogeneity. Two candidates:

### Option A — Fixed warmup window
`B = mean(agg_t^2 over t in [warmup_start, warmup_end])`, frozen after warmup.
- **Pros:** Doesn't absorb real drift, ever. Mechanism is dead-simple to argue and audit.
- **Cons:** Goes stale if the heterogeneity itself shifts (covariate drift, client churn, slow learning-rate decay changing the natural magnitude of `agg`). The gate's no-drift Dir(0.1) regime would be fine because the natural `agg^2` decays smoothly as the model converges — but a fixed mean from rounds 5–25 would be too large by round 100, attenuating *real* drift detection too.
- **Subtle danger:** Under a fixed window, `norm_delta` would shrink as training progresses (since `agg^2` shrinks but `B` is frozen). The trigger would gradually *under*-fire over time — including on real drift. Bad.

### Option B — Slow-updating EMA (recommended)
`B_t = beta_b * B_{t-1} + (1 - beta_b) * agg_t^2` with `beta_b >> beta2` (Flash's β2=0.99, so `beta_b` ∈ [0.999, 0.9995] — much slower).
- **Pros:** Tracks the natural decay of `agg^2` under stationary training. Stays calibrated as the model converges.
- **Cons:** A slow EMA *will* eventually absorb real drift if `beta_b` is too small. The drift event becomes invisible to `norm_delta` after a few hundred rounds.
- **Mitigation:** Pick `beta_b` such that the EMA's effective window (`1/(1-beta_b)` ≈ 1000–2000 rounds) is much longer than any reasonable drift event recovery window (~10–50 rounds in our regimes). Real drift will cause a short visible spike before the slow EMA catches up; that's exactly the window during which the trigger should fire.

**Tradeoff statement.** Fixed-warmup is mechanically purest but doesn't survive natural training dynamics. EMA is the standard solution and we lean into it.

**Recommendation.** Option B with `beta_b = 0.999` (effective window ≈ 1000 rounds, vs Flash's β2 effective window of ≈ 100 rounds). Validate by inspecting `norm_delta` trajectory in no-drift Dir(0.1) (should stay near zero) and canonical Dir(0.1) sudden (should spike at round 100 and decay over 10–20 rounds).

**Decision required.** A or B? My recommendation: B at `beta_b = 0.999`. Open to A if you want the simpler/more auditable thing accepting the staleness risk.

---

## 3. Client-side vs server-side normalization

The hypothesis on the table: "each client normalizes its own update against its own baseline since heterogeneity is per-client." Let me work through both.

### Server-side normalization (one global `B`)
`B_t` is a single tensor on the server, updated from the server's `agg_t^2` after aggregation.
- **Pros:** Drop-in change to Flash's server step. No client communication change. Reproduces today's failure mode cleanly: spurious firing happens on the SERVER's `agg^2`, so normalizing it on the server addresses where the problem manifests.
- **Cons:** "Heterogeneity is per-client" — clients with high local data difficulty contribute large per-client updates; the server's `agg^2` aggregates these and a single global `B` averages over them. If one cohort drifts (partial-coverage), the cohort's outlier update would spike `agg^2`, and a global `B` would still detect it correctly — but with less sensitivity than a client-level signal.

### Client-side normalization (each client has its own `B_i`)
Each client `i` maintains `B_i_t = beta_b * B_i_{t-1} + (1 - beta_b) * ||u_i_t||^2` where `u_i_t = theta_i_t - theta_global_t` is its local update. Server collects pre-normalized signals: each client reports `n_i_t = ||u_i_t||^2 / B_i_t` (or the full per-parameter ratio), and the server's drift signal becomes `(weighted mean of n_i_t) - 1`.
- **Pros:** Each client's heterogeneity baseline normalizes its own contribution. A naturally hard client (high `B_i`) doesn't dominate the drift signal. A drifting client (sudden change in `||u_i||^2` away from its OWN `B_i`) lights up regardless of its absolute magnitude.
- **Cons:** Adds per-client state that must persist across rounds (server has to remember client IDs ↔ `B_i`). Need to handle new/dropped clients. Each client must keep `B_i` locally OR the server stores it and sends `B_i_t` back each round. Slightly more communication and statefulness.

**Recommendation.** Start client-side. The mechanism better matches the literature claim ("heterogeneity is per-client") and matches the gate's mechanistic finding (Flash fires because individual client updates are large under heterogeneity, not because the global aggregation introduces spurious noise). If client-side underperforms server-side in our regimes, that's an interesting finding in itself.

**Caveat to think about.** Client-side requires `B_i` to be defined before the client's first contribution. Cold start: use the first-round `||u_i||^2` as the initial `B_i` (so `n_i_1 = 1` for all clients, trigger is silent in round 1). Server-side has the same cold-start (uses initial `second_mom`).

**Decision required.** Client-side or server-side as the V1 method? My recommendation: client-side. If you'd rather see the simpler server-side variant first, I'll do that — but it leaves the per-client mechanism for V2 and we'd have to justify why we didn't do it from the start.

---

## 4. The action — KEEP IDENTICAL to Flash

**Confirmed: keep Flash's adaptation action unchanged.**
- Flash's action: `agg_update = server_lr * first_mom_t / (sqrt(second_mom_t) - delta_mom_t + tau)`.
- New method's action: same expression, with `delta_mom_t` computed from the normalized signal per §1–3.

**Why this is the right call.**
- Isolates the contribution to the trigger. Any gap vs Flash is attributable to the trigger, not the action. Any tie with FedAvg is interpretable.
- The Saile-NO-GO and Flash-PASS combined story is: "the trigger family matters; the action is secondary." Keeping the action constant makes that story testable.
- Implementation risk is minimal: one function to change (the `delta_mom` update line and the `B` bookkeeping).
- If we change the action too, we introduce a second variable and can't tell which one bought the gain.

**One subtlety.** Flash's `delta_mom` is an EMA of `(agg^2 - second_mom)`, which is signed. Our normalized version `(agg^2 - second_mom) / B` is also signed (negative when `agg^2 < second_mom`). The action is the same expression `sqrt(second_mom) - delta_mom + tau`, so a negative `delta_mom` makes the denominator LARGER and attenuates the step (Flash's intent: when updates are smaller than expected, be conservative). The sign preservation matters; do not absolute-value the normalized signal.

---

## 5. Evaluation plan

**Primary metric.** Per-seed `(global_acc_stable, per_client_gen_acc_stable)` mean of last 10 of 200 rounds.

**Required outcomes (all on ≥3 seeds, single CIFAR-10 / CifarCNN / 20-client harness):**

### Beat both Flash and FedAvg
- **No-drift Dir(0.1)** — target gap vs Flash: ≥ +1.5pp (recovering most of Flash's −2.28pp deficit). Target gap vs FedAvg: ≥ 0 (no regression). Every seed positive vs Flash. 3-seed mean significantly > 0 (mean > std).
- **Partial-coverage A** — target gap vs Flash: ≥ +1.0pp (recovering most of Flash's −1.96pp). Target gap vs FedAvg: ≥ 0.

### At-worst-tie both
- **Canonical Dir(0.1) sudden** — Δ vs FedAvg and Δ vs Flash both within ±1pp (this is where Flash already ties FedAvg; we must not regress).
- **Dir(0.5) sudden** — Δ vs FedAvg within ±1pp; vs Flash within ±1pp.

### Diagnostic instrumentation (re-use the gate's apparatus)
- Log per-round `flash_amp_new` (the normalized analog). Confirm in no-drift Dir(0.1) that the new trigger's first-30-rounds mean is **near zero** (not 0.045 like Flash).
- Log per-round `B_norm` (or per-client `B_i_norm` summarized to mean/min/max). Confirm `B` rises with `agg^2` in no-drift Dir(0.1) (cancellation) and lags `agg^2` at the drift event in canonical Dir(0.1) (so `norm_delta` spikes).

### Per-seed reporting
- Both regimes, all 4 deltas (vs Flash global / vs FedAvg global / vs Flash per-cli / vs FedAvg per-cli), per seed and 3-seed mean ± std.
- The robustness gate: every seed must show the targeted direction. Mean-only is not sufficient (lesson from the prior shrinkage-gate false positive cited in user-spec history).

### Verdict thresholds
- **WIN:** all four target conditions above hold across 3 seeds. Direction stays alive; proceed to write-up.
- **PARTIAL WIN:** beats Flash in no-drift Dir(0.1) but doesn't fully reach FedAvg, OR ties in partial-coverage instead of beating. → diagnose: is it the baseline window? the normalization scope? Reconsider.
- **FAIL:** regresses vs Flash in any clean-drift regime, or fails to beat Flash in no-drift Dir(0.1). → the normalization broke the action; reconsider what to normalize or how.

### Compute budget
~12 runs to confirm V1 (3 seeds × 4 regimes; reuse Flash + FedAvg baselines from the gate). At ~40 min/pair on L4 with 2 parallel runs ≈ ~4 hours wall.

---

## 6. Things I'd push back on / open questions

1. **Should we include client-count scaling?** The harness is fixed at 20 clients. The normalized trigger might behave differently at larger client counts. Out of scope for V1; flag for the future.

2. **Should we test in the Saile-killing regimes too?** The Saile gate killed per-client triggers; the new method has a client-side baseline but a server-side action. The Saile result doesn't directly apply (Saile changed the action; we're keeping the action). But running canonical/recurrent/aggressive once each to verify the method also at-worst-ties in those Saile-tested regimes would be cheap and add evidence the trigger generalizes.

3. **What if `beta_b = 0.999` is wrong?** A small ablation (`beta_b in {0.99, 0.995, 0.999, 0.9995}`) on no-drift + canonical, single-seed, would tell us the sensitivity. Cheap enough to do before claiming V1 is the final answer. Recommend including this in the plan.

4. **Naming.** I called it `FlashNormTrigger`/`NormFlash` above. If you have a preference, name it now so the implementation, CSVs, and analysis scripts all use the same string.

---

## Summary of decisions to confirm

| # | Decision | My recommendation | Need your call? |
|---|---|---|---|
| 1 | Elementwise vs scalar-norm `norm_delta` | Elementwise | Confirm |
| 2 | Baseline window: fixed warmup vs slow EMA | Slow EMA, `beta_b = 0.999` | Confirm |
| 3 | Client-side vs server-side normalization | Client-side | Confirm |
| 4 | Action: keep identical to Flash | Yes | Confirm |
| 5 | Eval plan: beat both on 2 regimes + tie on 2 regimes, 3 seeds | Yes | Confirm |
| 6 | Include `beta_b` sensitivity ablation | Yes, single-seed | Confirm |
| 7 | Test in Saile-killing regimes for generalization | Optional add-on | Defer or now? |
| 8 | Name | `FlashNormTrigger` (suggest) | Pick or override |

Ready to build on your sign-off.
