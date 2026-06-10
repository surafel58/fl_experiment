#!/usr/bin/env bash
# 4-LR sweep for AdaptiveFedAvg (corrected), seed 0 only.
# 2-at-a-time on the L4. Writes ~/ADAPTIVE_SWEEP_DONE when finished.

set -u
cd ~
mkdir -p logs

wait_method_3_drained() {
    while pgrep -f "all_experiments_optimized.py.*--methods 3 " > /dev/null; do
        sleep 30
    done
}

run_pair() {
    local A_LR="$1"   local A_TAG="$2"
    local B_LR="$3"   local B_TAG="$4"
    local OUT_ROOT="runs/2026-06-09-adaptive-lr-sweep"

    mkdir -p "$OUT_ROOT/lr_${A_TAG}" "$OUT_ROOT/lr_${B_TAG}"

    echo "[sweep] launching pair: lr=${A_LR} (tag ${A_TAG}) + lr=${B_LR} (tag ${B_TAG})"
    nohup python3 -u all_experiments_optimized.py --seed 0 --methods 3 --rounds 200 \
          --adaptive-init-lr "$A_LR" \
          --out-dir "$OUT_ROOT/lr_${A_TAG}/" \
          > "logs/adasweep_${A_TAG}.log" 2>&1 &
    sleep 3
    nohup python3 -u all_experiments_optimized.py --seed 0 --methods 3 --rounds 200 \
          --adaptive-init-lr "$B_LR" \
          --out-dir "$OUT_ROOT/lr_${B_TAG}/" \
          > "logs/adasweep_${B_TAG}.log" 2>&1 &
    sleep 3
    echo "[sweep] PIDs: $(pgrep -f 'all_experiments_optimized.py.*--methods 3' | tr '\n' ' ')"
    wait_method_3_drained
    echo "[sweep] pair done at $(date -u)"
}

echo "[sweep] start at $(date -u)"
run_pair 0.1     "0p1"     0.01    "0p01"
run_pair 0.001   "0p001"   0.0001  "0p0001"

touch ~/ADAPTIVE_SWEEP_DONE
echo "[sweep] ALL DONE at $(date -u)"
