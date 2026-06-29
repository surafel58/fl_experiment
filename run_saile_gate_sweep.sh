#!/usr/bin/env bash
# Saile-instability gate: 4 Saile (lr=0.01) single-seed runs across regimes.
# 2-at-a-time on the L4. Touches ~/SAILE_GATE_DONE on completion.
#
# Each run uses Saile (method 6) at its best stable LR from the prior
# canonical sweep (0.01). The CSV that lands captures per-round
# base_lr / min/mean/max_client_lr (instrumented in the harness already),
# so the LR trajectory around drift is observable post-hoc.

set -u
cd ~
mkdir -p logs

wait_method_6_drained() {
    while pgrep -f "all_experiments_optimized.py.*--methods 6 " > /dev/null; do
        sleep 30
    done
}

run_pair() {
    local A_FLAGS="$1"   local A_TAG="$2"
    local B_FLAGS="$3"   local B_TAG="$4"
    local OUT_ROOT="runs/2026-06-24-saile-gate"

    mkdir -p "$OUT_ROOT/$A_TAG" "$OUT_ROOT/$B_TAG"

    echo "[gate] launching: tag=$A_TAG  ($A_FLAGS)  +  tag=$B_TAG  ($B_FLAGS)"
    nohup python3 -u all_experiments_optimized.py --seed 0 --methods 6 --rounds 200 \
          --saile-init-lr 0.01 $A_FLAGS \
          --out-dir "$OUT_ROOT/$A_TAG/" \
          > "logs/saile_gate_${A_TAG}.log" 2>&1 &
    sleep 3
    nohup python3 -u all_experiments_optimized.py --seed 0 --methods 6 --rounds 200 \
          --saile-init-lr 0.01 $B_FLAGS \
          --out-dir "$OUT_ROOT/$B_TAG/" \
          > "logs/saile_gate_${B_TAG}.log" 2>&1 &
    sleep 3
    echo "[gate] PIDs: $(pgrep -f 'all_experiments_optimized.py.*--methods 6' | tr '\n' ' ')"
    wait_method_6_drained
    echo "[gate] pair done at $(date -u)"
}

echo "[gate] start at $(date -u)"

# Pair 1: Dir(0.5) sudden  +  recurrent alternating
run_pair "--alpha-dir 0.5"      "dir05"      "--alternating-drift" "recurrent"

# Pair 2: aggressive permutation  +  covariate drift
run_pair "--aggressive-concept-drift" "aggressive" "--covariate-drift"   "covariate"

touch ~/SAILE_GATE_DONE
echo "[gate] ALL DONE at $(date -u)"
