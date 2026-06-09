#!/usr/bin/env bash
# Run all 5 methods at seeds 1 and 2 on the VM, 2-at-a-time per seed.
# Writes ~/ALL_DONE_perclient_3seed when finished.

set -u
cd ~
mkdir -p logs

wait_for_method() {
    local id="$1"
    while pgrep -f "all_experiments_optimized.py.*--methods ${id} " > /dev/null; do
        sleep 30
    done
}

run_seed() {
    local SEED="$1"
    local OUT_ROOT="runs/2026-06-08-perclient-3seed/seed${SEED}"
    mkdir -p "$OUT_ROOT/fedavg" "$OUT_ROOT/flash" "$OUT_ROOT/adaptive" "$OUT_ROOT/ourmethod" "$OUT_ROOT/fedavgplus1"

    echo "[chain] === SEED $SEED B1 (methods 1 + 5) ==="
    nohup python3 -u all_experiments_optimized.py --seed "$SEED" --methods 1 --rounds 200 --out-dir "$OUT_ROOT/fedavg/" > "logs/pc3_seed${SEED}_fedavg.log" 2>&1 &
    sleep 3
    nohup python3 -u all_experiments_optimized.py --seed "$SEED" --methods 5 --rounds 200 --out-dir "$OUT_ROOT/fedavgplus1/" > "logs/pc3_seed${SEED}_fedavgplus1.log" 2>&1 &
    sleep 3
    echo "[chain] seed $SEED B1 launched at $(date). PIDs: $(pgrep -f 'all_experiments_optimized.py.*--seed '$SEED | tr '\n' ' ')"
    wait_for_method 1
    wait_for_method 5
    echo "[chain] seed $SEED B1 done at $(date)."

    echo "[chain] === SEED $SEED B2 (methods 2 + 3) ==="
    nohup python3 -u all_experiments_optimized.py --seed "$SEED" --methods 2 --rounds 200 --out-dir "$OUT_ROOT/flash/" > "logs/pc3_seed${SEED}_flash.log" 2>&1 &
    sleep 3
    nohup python3 -u all_experiments_optimized.py --seed "$SEED" --methods 3 --rounds 200 --out-dir "$OUT_ROOT/adaptive/" > "logs/pc3_seed${SEED}_adaptive.log" 2>&1 &
    sleep 3
    echo "[chain] seed $SEED B2 launched at $(date)."
    wait_for_method 2
    wait_for_method 3
    echo "[chain] seed $SEED B2 done at $(date)."

    echo "[chain] === SEED $SEED B3 (method 4) ==="
    nohup python3 -u all_experiments_optimized.py --seed "$SEED" --methods 4 --rounds 200 --out-dir "$OUT_ROOT/ourmethod/" > "logs/pc3_seed${SEED}_ourmethod.log" 2>&1 &
    sleep 3
    echo "[chain] seed $SEED B3 launched at $(date)."
    wait_for_method 4
    echo "[chain] seed $SEED B3 done at $(date)."
}

run_seed 1
run_seed 2

touch ~/ALL_DONE_perclient_3seed
echo "[chain] ALL_DONE marker written at $(date)."
