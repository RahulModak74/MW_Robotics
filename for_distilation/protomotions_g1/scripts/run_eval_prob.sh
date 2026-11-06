#!/usr/bin/env bash
# usage: run_eval_prob.sh <jobs file> [parallel]     jobs: "<tag> <student .pt or 'teacher'> <motion> <noise> <seed> [gate S:T]"
# Writes $R/eval_prob/<tag>.log (ProtoMotions evaluator output) and <tag>.npz (per-step uncertainty, prob students only). Resumable.
source "$(dirname "$0")/common.sh"
JOBS=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); P=${2:-4}; OUT=${OUT:-$R/eval_prob}; mkdir -p "$OUT"
export R PY PM OURS CK OUT
run_one() {
  tag=$1; st=$2; m=$3; nz=$4; sd=$5; gt=$6
  [ -f "$OUT/$tag.done" ] && return
  extra=""; [ "$st" != "teacher" ] && extra="--student $st --uq-out $OUT/$tag.npz"
  [ -n "$gt" ] && extra="$extra --gate $gt"
  case "$m" in /*|*.pt) mf=$m ;; *) mf=data/g1-kimodo-generated/proto/output_${m}.motion ;; esac   # motion name or a motion-lib path
  cd "$PM" && "$PY" "$OURS/pm_eval_student.py" $extra --act-noise $nz --seed $sd --simulator mujoco --headless --num-envs 1 --full-eval \
     --checkpoint $CK --motion-file "$mf" > "$OUT/$tag.log" 2>&1 && touch "$OUT/$tag.done"
}
export -f run_one
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | xargs -P "$P" -L 1 bash -c 'run_one "$@"' _
