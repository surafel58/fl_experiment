#!/usr/bin/env bash
# Flash confounding gate, MULTI-SEED CONFIRM on the 2 promising regimes.
# Single-seed gate already showed -3.51pp (no-drift) and -2.18pp (partial-A).
# Seed 0 reused from runs/2026-06-24-flash-gate/{nodrift_d01,partial_A}/.
# Here we add seeds 1 and 2, FedAvg + Flash each = 4 pairs of 2 parallel runs.
#
# Touches ~/FLASH_MULTISEED_DONE on completion.

set -u
cd ~
mkdir -p logs
OUT_ROOT="runs/2026-06-24-flash-multiseed"

wait_drained() {
    while pgrep -f "all_experiments_optimized.py" > /dev/null; do
        sleep 30
    done
}

launch_pair() {
    local TAG="$1"   local SEED="$2"   local FLAGS="$3"
    local DIR="$OUT_ROOT/${TAG}/seed${SEED}"
    mkdir -p "$DIR"

    echo "[flash-ms] launching pair: ${TAG} seed=${SEED} (FedAvg + Flash, $FLAGS)"
    nohup python3 -u all_experiments_optimized.py --seed $SEED --methods 1 --rounds 200 \
          $FLAGS --out-dir "$DIR/" \
          > "logs/flash_ms_${TAG}_s${SEED}_m1.log" 2>&1 &
    sleep 3
    nohup python3 -u all_experiments_optimized.py --seed $SEED --methods 2 --rounds 200 \
          $FLAGS --out-dir "$DIR/" \
          > "logs/flash_ms_${TAG}_s${SEED}_m2.log" 2>&1 &
    sleep 3
    echo "[flash-ms] PIDs: $(pgrep -f 'all_experiments_optimized.py' | tr '\n' ' ')"
    wait_drained
    echo "[flash-ms] pair done at $(date -u)"
}

echo "[flash-ms] start at $(date -u)"

# 4 pairs sequential, 2 procs each in parallel (proven config from gate sweep)
launch_pair "nodrift_d01" 1 "--no-drift"
launch_pair "nodrift_d01" 2 "--no-drift"
launch_pair "partial_A"   1 "--partial-cohorts A"
launch_pair "partial_A"   2 "--partial-cohorts A"

touch ~/FLASH_MULTISEED_DONE
echo "[flash-ms] ALL DONE at $(date -u)"
