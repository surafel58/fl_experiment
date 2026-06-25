#!/usr/bin/env bash
# Headroom gate: does oracle drift-boost FedAvg leave room beyond what
# always-higher-LR FedAvg already captures? Plus FedAvgAdam action-ceiling
# control. All canonical Dir(0.1) sudden drift @ rd 100, seed 0, 200 rounds.
#
# 4 pairs of 2 parallel runs. Touches ~/HEADROOM_GATE_DONE on completion.
#   Pair 1: oracle 2x + always 2x
#   Pair 2: oracle 3x + always 3x
#   Pair 3: oracle 5x + always 5x
#   Pair 4: FedAvgAdam (single run, paired with idle; or run solo)

set -u
cd ~
mkdir -p logs
OUT=runs/2026-06-25-headroom-gate

wait_drained() {
    while pgrep -f "all_experiments_optimized.py" > /dev/null; do
        sleep 30
    done
}

launch_pair() {
    local A_TAG="$1"   local A_FLAGS="$2"
    local B_TAG="$3"   local B_FLAGS="$4"
    mkdir -p "$OUT/$A_TAG" "$OUT/$B_TAG"

    echo "[headroom] launching: $A_TAG ($A_FLAGS) + $B_TAG ($B_FLAGS)"
    nohup python3 -u all_experiments_optimized.py --seed 0 --methods 1 --rounds 200 \
          $A_FLAGS --out-dir "$OUT/$A_TAG/" \
          > "logs/headroom_${A_TAG}.log" 2>&1 &
    sleep 3
    nohup python3 -u all_experiments_optimized.py --seed 0 --methods 1 --rounds 200 \
          $B_FLAGS --out-dir "$OUT/$B_TAG/" \
          > "logs/headroom_${B_TAG}.log" 2>&1 &
    sleep 3
    echo "[headroom] PIDs: $(pgrep -f 'all_experiments_optimized.py' | tr '\n' ' ')"
    wait_drained
    echo "[headroom] pair done at $(date -u)"
}

echo "[headroom] start at $(date -u)"

# Oracle boost = LR boost only during rd 100-110.
# Always-higher = global LR override (no boost window).

launch_pair "oracle_2x" "--lr-boost-factor 2.0"  "always_2x" "--lr 0.02"
launch_pair "oracle_3x" "--lr-boost-factor 3.0"  "always_3x" "--lr 0.03"
launch_pair "oracle_5x" "--lr-boost-factor 5.0"  "always_5x" "--lr 0.05"

# FedAvgAdam: action-ceiling control, no trigger
mkdir -p "$OUT/fedavg_adam"
echo "[headroom] launching: fedavg_adam"
nohup python3 -u all_experiments_optimized.py --seed 0 --methods 11 --rounds 200 \
      --out-dir "$OUT/fedavg_adam/" \
      > "logs/headroom_fedavg_adam.log" 2>&1 &
sleep 3
wait_drained
echo "[headroom] fedavg_adam done at $(date -u)"

touch ~/HEADROOM_GATE_DONE
echo "[headroom] ALL DONE at $(date -u)"
