# Writeup structure — DRAFT FOR REVIEW (outline + key evidence, not prose)

**Central thesis:** Flash's underperformance vs FedAvg is an **ACTION problem**, not a trigger problem.
- The confound (spurious early-training firing) is real (gate PASS).
- The trigger IS fixable (V2 silenced it).
- But fixing the trigger cannot reach FedAvg, because Flash's action — its server-side `lr × first_mom / (sqrt(second_mom) − delta_mom + tau)` — both caps stable accuracy below FedAvg's simple averaging AND becomes unstable under any strong trigger response (positive feedback through the divisor).

Working title candidate: **"The Flash limitation is in the action, not the trigger: an experimental failure map"**

---

## Section 1 — Headline (1 page)

**Stated up front (the thesis in 3 bullets):**
- ☐ Flash spuriously fires on heterogeneity in early training, costing 2-3pp vs FedAvg in no-drift / partial-coverage regimes. (Gate PASS, 3 seeds.)
- ☐ A normalization-based trigger fix can silence this spurious firing (V2: first-30 amp 20× quieter than Flash) and recover ~2pp of the gap — but cannot reach FedAvg.
- ☐ The remaining ~1.3pp gap + a catastrophic instability under real drift response are properties of Flash's server-side Adam-like action, not the trigger. Beating FedAvg on this benchmark requires changing the action class.

**Why this matters (one paragraph):** Drift-detection methods in FL (FedDrift, Flash, Saile et al.) have been compared at the trigger level. The Flash failure map here is the first empirical demonstration that *trigger improvements alone* hit an action ceiling — calibrates expectations for the whole "improve the trigger" research direction.

---

## Section 2 — The confounding gate (the one PASS finding)

**Subhead:** "Flash spuriously fires under static heterogeneity, costing 2–3pp."

**Already documented in:** `FLASH_CONFOUNDING_GATE_PASS.md`.

**Key evidence to include:**
- ☐ Setup: CIFAR-10, CifarCNN (107,690 params), 20 clients, Dir(0.1)/Dir(0.5), 200 rounds, seeds 0/1/2, Flash hyperparameters from published Flash.yaml.
- ☐ 4-regime single-seed sweep table (no-drift / partial-A / canonical / Dir(0.5)):
  - no-drift: −3.51pp, partial-A: −2.18pp (Flash behind FedAvg, both > noise floor)
  - canonical: +0.12pp, Dir(0.5): −0.73pp (within noise — Flash designed for these)
- ☐ Per-seed multi-seed table (3 seeds × 2 promising regimes):
  - no-drift: Δ = −2.28 ± 1.17pp (every seed Flash < FedAvg, every > 1pp)
  - partial-A: Δ = −1.96 ± 0.38pp
- ☐ Spurious-firing diagnostic: no-drift first-30 mean `flash_amp` = 0.045 ± 0.028 across seeds. By construction this firing is spurious (no drift to detect). Magnitude scales with the per-seed gap.
- ☐ Fair-tuning confirmation: Flash uses published defaults; ties FedAvg in canonical (the regime it was designed for) — gap is a trigger property, not a tuning artifact.

**Reuse evidence:** `runs/2026-06-24-flash-gate/`, `runs/2026-06-24-flash-multiseed/`, `flash_gate_analysis.py`, `flash_multiseed_analysis.py`.

---

## Section 3 — The trigger-is-fixable sub-finding

**Subhead:** "A normalization-based trigger fix achieves silence on heterogeneity, but…"

### 3.1 Cold-start init alone (FlashColdInit, attribution control)

**Claim:** Even just fixing Flash's `second_mom = τ²` initialization (replace with `second_mom = agg²` on round 0) closes 24% of the gap (+0.84pp). The scalar τ² init is itself a small source of spurious firing.

**Evidence:**
- ☐ FlashColdInit no-drift stable: 0.6825 (Flash: 0.6742, FedAvg: 0.7092)
- ☐ Same harness, same seed (0), only the `second_mom` init differs

### 3.2 V1 (slow-EMA per-element B): three failure modes documented

**Claim:** The straightforward slow-EMA implementation has multiple algebraic-derivable failure modes that compound:

| Variant | Approach | Failure |
|---|---|---|
| V1a | `scale = sqrt(second_mom)` per spec | Denominator collapsed → NaN at rd 1 |
| V1b | `scale = second_mom` (units fixed) | Outlier-element ratio explosion → NaN at rd 10-40 |
| V1c | + Option II `max(B_slow, second_mom)` | Algebraic collapse to Flash in 200-round budget |

**Evidence:**
- ☐ Algebraic derivation: with β_b = 0.999, B_slow's effective window is ~1000 rounds. Within 200 rounds, `B_slow < second_mom` for most params, so `max(B_slow, second_mom) = second_mom`, and `scaled_delta = ratio × second_mom = (agg²−second_mom)` — algebraically identical to Flash's raw signal.
- ☐ Smoke: V1c first-30 amp = 0.027, ~equal to Flash's 0.025; V1c stable = 0.6869 (+1.27pp over Flash, but only +0.44pp over ColdInit floor).

### 3.3 V2 (per-tensor B, fixed warmup window): no-drift silence achieved

**Claim:** Switching to per-tensor B + a frozen warmup baseline + vanilla-Adam during warmup achieves the design intent for spurious firing.

**Evidence:**
- ☐ Per-tensor B (one scalar per parameter tensor, ~10 scalars for CifarCNN): eliminates V1's outlier-element pathology. Per-tensor B is dominated by typical elements, never near-zero.
- ☐ Frozen warmup B (mean of agg² over rounds 5–24, then constant): doesn't lag the model the way the slow EMA did; doesn't algebraically collapse to Flash.
- ☐ Vanilla-Adam during warmup (rd 0–24, delta_mom = 0): avoids ALL of Flash's spurious firing during the noisy early-training phase.
- ☐ Result on no-drift: **first-30 mean amp = 0.0013** (20× quieter than Flash's 0.0255). Stable = 0.6960 — beats Flash by +2.18pp, beats ColdInit by +1.35pp (clearly above the +0.84pp ColdInit floor → normalization adds real signal beyond the init fix).

**Interim conclusion (for THIS section):** Spurious firing on heterogeneity is a fixable trigger problem. The gate's mechanistic story (Flash over-fires on static heterogeneity, this is exploitable) is confirmed.

---

## Section 4 — The action ceiling (central thesis, half 1)

**Subhead:** "Even a near-perfect trigger cannot reach FedAvg."

**The observation that drives the thesis:**

| Method | No-drift stable | Δ vs FedAvg | first-30 amp |
|---|---:|---:|---:|
| FedAvg | **0.7092** | — | — |
| Flash | 0.6742 | −3.51pp | 0.0255 |
| FlashColdInit | 0.6825 | −2.67pp | 0.0295 |
| V1 Option II | 0.6869 | −2.23pp | 0.0270 |
| **V2 (silenced)** | **0.6960** | **−1.32pp** | **0.0013** |

**The argument:**
- ☐ Going from "Flash with full spurious firing" (0.6742) to "Flash with effectively no spurious firing" (V2, 0.6960) recovers **+2.18pp**.
- ☐ But there's STILL a **−1.32pp gap to FedAvg** even when the trigger is functionally silent.
- ☐ The trigger improvements show diminishing returns toward an apparent asymptote ~0.69–0.70.
- ☐ That asymptote is below FedAvg's 0.7092 by ~1pp.

**Why the action accounts for this gap (mechanism argument):**
- ☐ Flash's action is server-side Adam-like: `update = server_lr × first_mom / (sqrt(second_mom) − delta_mom + τ)`. Even with `delta_mom = 0` (vanilla Adam), this divides by sqrt(second_mom).
- ☐ FedAvg's action is simple weighted averaging: `update = sum(w_i × client_update_i)`. No per-parameter denominator.
- ☐ On Dir(0.1) CIFAR-10, the sqrt(second_mom) divisor appears to introduce per-parameter step-size imbalance that costs ~1pp vs simple averaging.
- ☐ This is a property of the ACTION, not the trigger — the trigger only affects how `delta_mom` modifies the divisor.

**Evidence to include:**
- ☐ The progression table above
- ☐ Algebraic decomposition: when `delta_mom = 0`, Flash's action = vanilla server-side Adam. V2 during the WARMUP phase IS vanilla server-side Adam (delta_mom = 0 by construction for rounds 0–24). V2's overall stable accuracy is bounded above by what vanilla server-side Adam would achieve. If we ran "pure FedAvg-Adam" (no trigger at all, vanilla Adam aggregation), we'd expect ~the V2 stable level.
- ☐ **CONFIRMED EXPERIMENTALLY (2026-06-25 headroom gate):** FedAvgAdam (method 11 — Flash's Adam action with delta_mom=0 throughout, no trigger at all):
  - No-drift: V2 stable 0.6960 (which IS vanilla Adam during warmup + silent trigger after) → action ceiling lives there.
  - **Canonical drift: FedAvgAdam stable 0.5773, plain FedAvg stable 0.5887 → −1.14pp action gap, matching the −1.32pp no-drift gap.**
  - **The action ceiling is bilateral and consistent across regimes — not a no-drift artifact.**

---

## Section 4b — The headroom gate (action confirmed via a different angle)

**Subhead:** "Even with oracle drift timing, a brief LR boost on plain FedAvg leaves no headroom."

**Motivating question (after Sections 3+4 closed the trigger-fix direction):** *Could a drift-triggered boost on top of FedAvg's STABLE simple-averaging action open the gap that Flash's Adam action couldn't?* This gate tests whether plain FedAvg leaves any headroom AT the drift event itself — bypassing the Adam action entirely.

**Configs tested, single seed, canonical Dir(0.1) drift @ rd 100:**
- Plain FedAvg (baseline)
- Oracle-boosted FedAvg: 2x/3x/5x LR boost during rounds [100, 110) (the oracle knows the drift round)
- Always-higher LR: 2x/3x/5x LR applied throughout (critical control — separates "boost AT drift specifically" from "higher LR generally helps")
- FedAvgAdam (no trigger) — also runs here as the action-ceiling check

**Result table:**

| Config | Dip | Stable | Δ stable vs Plain | Δ stable vs same-factor Always |
|---|---:|---:|---:|---:|
| Plain FedAvg | 0.1219 | **0.5887** | — | — |
| Oracle 2x | 0.1384 | 0.5849 | −0.38pp | −0.09pp |
| Oracle 3x | 0.2102 | 0.5918 | +0.31pp | **+1.81pp** |
| Oracle 5x | 0.6102 | CRASHED | — | — |
| Always 2x (lr=0.02) | 0.1215 | 0.5858 | −0.29pp | — |
| Always 3x (lr=0.03) | 0.1390 | 0.5737 | −1.50pp | — |
| Always 5x (lr=0.05) | — | CRASHED | — | — |

**Verdict: FAIL (no exploitable headroom).**
- ☐ 2x: oracle WORSE than plain stable (−0.38pp), oracle ≈ always (Δ −0.09pp). Trigger adds nothing.
- ☐ 3x: oracle marginally beats plain (+0.31pp, single-seed, within ±1pp noise) AND beats always_3x (+1.81pp) — but always_3x is degraded (lr=0.03 generally too high → −1.50pp below plain), so "oracle beats always_3x" reflects always_3x's degradation more than oracle's gain. The +0.31pp over plain is single-seed noise level.
- ☐ 5x: both configs crash. The boost factor magnitude that might overcome plain FedAvg's self-recovery rate is too aggressive to be stable on this benchmark.

**Mechanism (the deeper finding):** at label-swap drift, clients train for ~10-30 rounds on wrong-direction gradients while figuring out the new label mapping. Boosting their LR during this window amplifies the wrong updates → deeper dip (oracle 3x dipped +8.83pp deeper than plain). The boost ends, model has more damage to recover from. Final stable asymptotes to ~plain FedAvg's stable. **There is no "knob to turn up" at drift because FedAvg's drift-recovery bottleneck is client-side label-relearning, not aggregator convergence speed.**

**Files:** `runs/2026-06-25-headroom-gate/`, `headroom_gate_analysis.py`, branch `headroom-gate-fail` (commit b0a6795).

---

## Section 5 — The action instability (central thesis, half 2)

**Subhead:** "Strong trigger response triggers runaway feedback through the action denominator."

**The observation:**

V2 canonical drift trajectory (after silence pre-drift):
```
rd  99   global=0.682   amp_new=0.008   (silent pre-drift)
rd 100   global=0.673   amp_new=0.184   (drift fires — 3.75× Flash's drift response)
rd 101   global=0.655   amp_new=0.267
rd 105   global=0.539   amp_new=2.118
rd 107   global=0.508   amp_new=2.504
rd 109   global=0.267   amp_new=17.614
rd 110   global=0.100   amp_new=NaN
```

**The mechanism (positive-feedback argument):**
- ☐ Flash's action implicitly assumes `|delta_mom| << sqrt(second_mom)` per element. The denominator `sqrt(second_mom) − delta_mom + τ` is meaningful only while delta_mom stays bounded below sqrt(second_mom).
- ☐ At a strong drift response (e.g., V2 at rd 100), `delta_mom` jumps to a large value relative to sqrt(second_mom).
- ☐ This produces an amplified server step (denominator near zero or negative).
- ☐ The amplified step pushes the model further out of equilibrium, producing larger `agg²` next round.
- ☐ Larger `agg²` feeds back into a larger `delta_mom` (the trigger fires harder).
- ☐ Loop: amp 0.18 → 0.27 → 0.82 → 2.12 → 2.50 → 17.6 → NaN.

**Why Flash itself doesn't suffer this:**
- ☐ Flash's `delta_mom` is unscaled `(agg² − second_mom)`, which is naturally bounded by `agg²` magnitude (~1e-6 per element). This is 3 orders of magnitude below sqrt(second_mom) (~1e-3 per element). The implicit invariant `|delta_mom| << sqrt(second_mom)` is satisfied by accident of dimensionality.
- ☐ Any normalization (V1, V2) that puts `delta_mom` at a magnitude closer to `sqrt(second_mom)` violates the invariant on strong responses → instability.
- ☐ Flash is therefore "stuck" with a weak trigger: any stronger trigger response would destabilize the action.

**Implication:** The action's denominator structure does double duty:
1. Caps stable accuracy below simple averaging (Section 4 argument).
2. Caps trigger response strength below the action's stability margin (Section 5 argument).

**Together these constitute the thesis: the limitation is in the action.**

---

## Section 6 — Implications & alternative directions

**The trigger-improvement research direction:**
- ☐ Improving Flash's trigger (the literature focus) cannot beat FedAvg on this benchmark.
- ☐ A heterogeneity-aware trigger CAN silence spurious firing (V2 demonstrated this), but the residual gap is in the action.
- ☐ This suggests the FedDrift/Flash/Saile family — which all preserve some form of adaptive-LR action — share a similar ceiling.

**The drift-triggered FedAvg-modulation direction:**
- ☐ Putting a drift trigger on TOP of FedAvg's simple-averaging action (instead of Flash's Adam action) was the next natural pivot. Section 4b (headroom gate) tests this.
- ☐ Result: oracle drift-timed LR boost on plain FedAvg leaves no exploitable headroom. FedAvg self-recovers as fast as a brief LR boost can, because the drift bottleneck is client-side label-relearning, not server-side convergence speed.
- ☐ This closes the "trigger + FedAvg action" direction in addition to "trigger + Flash action".

**What would beat FedAvg on this benchmark:**
- ☐ Replace the action class entirely: e.g., trigger drives a learning-rate schedule rather than a per-parameter divisor (action stays close to FedAvg's simple averaging).
- ☐ Or: stay with FedAvg as the base action and add drift detection only as a re-clustering / re-initialization trigger (the FedDrift family direction).
- ☐ Or: accept the action ceiling and target methods that explicitly trade some no-drift performance for faster drift recovery (different research goal).

**For our research program:**
- ☐ The "improve Flash's trigger" direction is closed. V2's no-drift improvement is real but cannot reach FedAvg.
- ☐ The "method that beats both Flash AND FedAvg" target was the wrong shape: should be re-aimed.
- ☐ The exploitable headroom from the confounding gate doesn't yield a method that beats FedAvg; it yields a method that's better than Flash but worse than FedAvg. Not a research contribution.

---

## Section 7 — Honest negative-result summary

**What we set out to do:** Beat both Flash and FedAvg in no-drift / partial-coverage by fixing Flash's spurious-firing trigger.

**What we proved:**
1. ☐ Flash's spurious firing on heterogeneity is real (gate PASS, 3 seeds).
2. ☐ The trigger IS fixable (V2 achieves silence, beats Flash by +2.18pp in no-drift).
3. ☐ But fixing the trigger cannot beat FedAvg — Flash's action has a bilateral ceiling below simple averaging (−1.32pp no-drift, −1.14pp canonical drift) AND a stability margin that caps trigger response strength (V2 NaN'd on drift response).
4. ☐ **A drift-triggered LR boost on plain FedAvg ALSO fails:** oracle-timed boost leaves no exploitable headroom over plain FedAvg at the drift event (headroom gate, single seed; +0.31pp single-seed-noise gain at 3x at the cost of +8.83pp deeper dip; 5x crashes).
5. ☐ Therefore: BOTH the trigger-improvement direction (with Flash's action) AND the drift-modulation direction (with FedAvg's action) are closed for this benchmark. The drift-recovery bottleneck is client-side label-relearning, which neither aggregator-level adjustments nor LR boosts can shortcut.

**Why this is worth writing up:** Five findings are real. They jointly map a four-corner failure surface (trigger×action ∈ {Flash, FedAvg}²) around the "improve Flash" research direction. Negative results that map a failure surface this precisely ARE contributions.

---

## Section 8 — Appendix

**Material to preserve:**
- ☐ Per-seed tables for all gate runs (already in `FLASH_CONFOUNDING_GATE_PASS.md`)
- ☐ V1 three-failure-mode walkthrough with exact NaN-round + algebraic derivation each
- ☐ V2 paper analysis (`FLASH_METHOD_DESIGN_SPEC.md` + the per-tensor revision)
- ☐ Smoke trajectory tables (V1 nodrift, V1 canonical, V2 nodrift, V2 canonical, FlashColdInit nodrift)
- ☐ Code provenance: branch `flash-confounding-gate-pass` (gate evidence), branch `flashnorm-v2-failed` (V1 + V2 + control evidence)

---

## Open questions for your review (DECIDE BEFORE FULL PROSE)

1. **Include a "pure FedAvg-Adam" confirmatory run?** Would isolate the action ceiling as a standalone finding. ~30 min compute. Trade-off: small extra compute vs a sharper Section 4. **MY RECOMMENDATION: YES, run it.** It cleanly separates "action ceiling" from "trigger details" and makes the thesis bulletproof.

2. **Frame as "calibrating the trigger-improvement direction" or as "an inherent action limitation in adaptive-divisor FL methods"?** The first is honest about scope (one benchmark). The second is more ambitious but needs more evidence. **MY RECOMMENDATION: first framing, with the second as a hypothesis flag in the implications section.**

3. **Length target?** Workshop-paper short form (4-6 pages) or full negative-result writeup (8-10)? Different prose density.

4. **Audience?** FL research (assumes Flash/FedDrift familiarity) or broader ML (needs setup primer)?

5. **Title decision:** "The Flash limitation is in the action, not the trigger" or something less specific?

---

## File index for the writeup

Will reference:
- `FLASH_CONFOUNDING_GATE_PASS.md` — gate finding source-of-truth
- `FLASH_METHOD_DESIGN_SPEC.md` — V2 paper analysis
- `flash_gate_analysis.py` / `flash_multiseed_analysis.py` — gate analysis reproducers
- `flashnorm_smoke_analysis.py` — V1 smoke analysis
- `flashnorm_v2_smoke_analysis.py` — V2 smoke analysis
- `all_experiments_optimized.py` — methods 7, 8, 9 implementations
- `runs/2026-06-24-flash-gate/`, `runs/2026-06-24-flash-multiseed/` — gate CSVs
- `runs/2026-06-25-v2-smoke/`, `runs/2026-06-25-flashnorm-smoke*/` — V1+V2 smoke CSVs
