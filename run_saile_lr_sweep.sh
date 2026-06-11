#!/usr/bin/env bash
# 3-LR sweep for Saile (corrected), seed 0 only.
# 2-at-a-time on the L4. Writes ~/SAILE_SWEEP_DONE when finished.

set -u
cd ~
mkdir -p logs

wait_method_6_drained() {
    while pgrep -f "all_experiments_optimized.py.*--methods 6 " > /dev/null; do
        sleep 30
    done
}

run_one() {
    local LR="$1"   local TAG="$2"
    local OUT_ROOT="runs/2026-06-11-saile-lr-sweep"
    mkdir -p "$OUT_ROOT/lr_${TAG}"
    echo "[sweep] launching lr=${LR} (tag ${TAG}) at $(date -u)"
    nohup python3 -u all_experiments_optimized.py --seed 0 --methods 6 --rounds 200 \
          --saile-init-lr "$LR" \
          --out-dir "$OUT_ROOT/lr_${TAG}/" \
          > "logs/sailesweep_${TAG}.log" 2>&1 &
    sleep 3
}

echo "[sweep] start at $(date -u)"

# Pair 1: lr=0.2 and lr=0.1 in parallel
run_one 0.2  "0p2"
run_one 0.1  "0p1"
echo "[sweep] pair 1 PIDs: $(pgrep -f 'all_experiments_optimized.py.*--methods 6' | tr '\n' ' ')"
wait_method_6_drained
echo "[sweep] pair 1 done at $(date -u)"

# Pair 2: lr=0.01 alone
run_one 0.01 "0p01"
echo "[sweep] pair 2 PID: $(pgrep -f 'all_experiments_optimized.py.*--methods 6' | tr '\n' ' ')"
wait_method_6_drained
echo "[sweep] pair 2 done at $(date -u)"

touch ~/SAILE_SWEEP_DONE
echo "[sweep] ALL DONE at $(date -u)"
