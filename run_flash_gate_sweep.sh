#!/usr/bin/env bash
# Flash confounding gate: 4 regimes x (FedAvg + Flash), single seed.
# Pair-wise on the L4. Touches ~/FLASH_GATE_DONE on completion.
#
# Regimes:
#   nodrift_d01    --no-drift                 (alpha=0.1 default; static heterogeneity only)
#   partial_A      --partial-cohorts A        (only 6 of 20 clients drift at round 100)
#   canonical_d01  (no flags)                 (full-coverage Dir(0.1) sudden drift)
#   dir05          --alpha-dir 0.5            (full-coverage Dir(0.5) sudden drift)
#
# Reused FedAvg baselines (already pulled from prior runs):
#   canonical_d01  -> existing FedAvg CSV (canonical Dir(0.1) sudden)
#   dir05          -> existing FedAvg CSV (Dir(0.5) sudden, from Saile gate)
# So we run FedAvg only for nodrift_d01 and partial_A; Flash for all 4.

set -u
cd ~
mkdir -p logs
OUT_ROOT="runs/2026-06-24-flash-gate"

wait_drained() {
    while pgrep -f "all_experiments_optimized.py" > /dev/null; do
        sleep 30
    done
}

launch_pair() {
    local A_TAG="$1"   local A_METHOD="$2"   local A_FLAGS="$3"
    local B_TAG="$4"   local B_METHOD="$5"   local B_FLAGS="$6"
    mkdir -p "$OUT_ROOT/$A_TAG" "$OUT_ROOT/$B_TAG"

    echo "[flash-gate] launching: $A_TAG (m=$A_METHOD, $A_FLAGS) + $B_TAG (m=$B_METHOD, $B_FLAGS)"
    nohup python3 -u all_experiments_optimized.py --seed 0 --methods $A_METHOD --rounds 200 \
          $A_FLAGS --out-dir "$OUT_ROOT/$A_TAG/" \
          > "logs/flash_gate_${A_TAG}_m${A_METHOD}.log" 2>&1 &
    sleep 3
    nohup python3 -u all_experiments_optimized.py --seed 0 --methods $B_METHOD --rounds 200 \
          $B_FLAGS --out-dir "$OUT_ROOT/$B_TAG/" \
          > "logs/flash_gate_${B_TAG}_m${B_METHOD}.log" 2>&1 &
    sleep 3
    echo "[flash-gate] PIDs: $(pgrep -f 'all_experiments_optimized.py' | tr '\n' ' ')"
    wait_drained
    echo "[flash-gate] pair done at $(date -u)"
}

echo "[flash-gate] start at $(date -u)"

# Pair 1: no-drift FedAvg + no-drift Flash
launch_pair  "nodrift_d01" 1 "--no-drift"  "nodrift_d01" 2 "--no-drift"

# Pair 2: partial-A FedAvg + partial-A Flash
launch_pair  "partial_A" 1 "--partial-cohorts A"  "partial_A" 2 "--partial-cohorts A"

# Pair 3: canonical Flash + Dir(0.5) Flash (FedAvg baselines reused from prior runs)
launch_pair  "canonical_d01" 2 ""  "dir05" 2 "--alpha-dir 0.5"

touch ~/FLASH_GATE_DONE
echo "[flash-gate] ALL DONE at $(date -u)"
