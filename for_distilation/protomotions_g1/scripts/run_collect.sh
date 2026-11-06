#!/usr/bin/env bash
# usage: run_collect.sh [seed] [steps]  - 5 motions, one CPU process each (MuJoCo backend = 1 env / process); writes $R/data_pm/
source "$(dirname "$0")/common.sh"
SEED=${1:-0}; STEPS=${2:-40000}; OUT=$R/data_pm; mkdir -p "$OUT"
cd "$PM"
for m in walk wave high5 jumpjack dance; do
  PM_OUT=$OUT/${m}_s${SEED}.npz PM_STEPS=$STEPS PM_SEED=$SEED PM_MOTION_NAME=$m \
    "$PY" "$OURS/pm_collect.py" --simulator mujoco --headless --num-envs 1 --checkpoint $CK \
    --motion-file data/g1-kimodo-generated/proto/output_${m}.motion > "$OUT/${m}_s${SEED}.log" 2>&1 &
done
wait
echo ALL_COLLECTED_$SEED > "$OUT/collected_$SEED.txt"
