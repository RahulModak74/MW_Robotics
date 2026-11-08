#!/usr/bin/env bash
# m500.sh - 500 prompted Kimodo training motions (125 per category, dance excluded) + the 4 original motions -> students -> dance test.
# Light on the machine: at most 3 worker processes, and every stage / process waits until CPU and memory are both < LIMIT % (default 70).
#   ./m500.sh            all stages: select convert collect train eval agg
#   ./m500.sh <stage>    one stage
# Dance scoring: the clean (noise 0) clip is deterministic, so it is run ONCE; noisy runs use 3 seeds each -> 7 independent runs per arm.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); R=$(cd "$HERE/.." && pwd); M=$HERE/m500; mkdir -p $M/logs
CONV=$HOME/ProtoMotions/data/scripts/convert_g1_csv_to_proto.py; P=3; LIMIT=${LIMIT:-70}
source "$R/scripts/common.sh"
ST=$R/students

usage() {   # prints "cpu mem" in % (cpu over 2 s)
  local a b; a=($(head -1 /proc/stat)); sleep 2; b=($(head -1 /proc/stat))
  local idle=$(( (b[4]+b[5]) - (a[4]+a[5]) )); local tot=0; for i in 1 2 3 4 5 6 7 8; do tot=$(( tot + b[i] - a[i] )); done
  local mem=$(free | awk '/Mem:/ {printf "%d", 100 * ($2 - $7) / $2}')
  echo $(( 100 * (tot - idle) / (tot > 0 ? tot : 1) )) $mem
}
wait_ok() { while true; do read c m <<< "$(usage)"; [ $c -lt $LIMIT ] && [ $m -lt $LIMIT ] && return; echo "[$(date +%T)] waiting: cpu $c% mem $m%"; sleep 30; done; }

select_() {
  python3 - "$M" <<'EOF'
import json, os, glob, shutil, sys
from collections import defaultdict
M = sys.argv[1]; P = json.load(open("prompts.json")); cat = {p["id"]: p["cat"] for p in P}
split = {p["id"]: ("heldout_dance" if p["cat"] == "dance" else ("heldout_seen" if int(p["id"].split("_")[-1]) % 10 == 0 else "train")) for p in P}
by = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob("csv/*/*.csv")):
    b = os.path.basename(f); pid = "_".join(b.split("_")[:2]); by[split[pid]][pid].append(os.path.abspath(f))
def take(prompts, n):                      # round-robin over prompts: as many distinct prompts as possible
    out, k = [], 0
    while len(out) < n and any(len(v) > k for v in prompts.values()):
        for pid in sorted(prompts):
            if len(prompts[pid]) > k and len(out) < n: out.append(prompts[pid][k])
        k += 1
    return out
sel = {"train": []}
for c in ("locomotion", "gesture", "exercise", "everyday"):
    sel["train"] += take({p: v for p, v in by["train"].items() if cat[p] == c}, 125)
sel["eval_familiar"] = [v[0] for p, v in sorted(by["heldout_seen"].items())][:30]
sel["eval_dance"] = [v[0] for p, v in sorted(by["heldout_dance"].items())][:30]
for s, fs in sel.items():
    d = f"{M}/csv_{s}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in fs: os.symlink(f, f"{d}/{os.path.basename(f)}")
    print(s, len(fs), "motions,", len({'_'.join(os.path.basename(f).split('_')[:2]) for f in fs}), "prompts")
EOF
}
convert() {
  for s in train eval_familiar eval_dance; do wait_ok
    (cd $PM && $PY $CONV --input-dir $M/csv_$s --output-dir $M/proto_$s --input-fps 30 --output-fps 30 --pos-units m --rot-format quat_wxyz \
       --joint-units rad --no-has-header --no-has-frame-column > $M/logs/conv_$s.log 2>&1)
  done
  python3 -c "
import os
fs = sorted(os.listdir('$M/proto_train'))
for i in range($P):
    d = f'$M/shard{i}'; os.makedirs(d, exist_ok=True)
    for f in fs[i::$P]:
        if not os.path.exists(f'{d}/{f}'): os.symlink(f'$M/proto_train/{f}', f'{d}/{f}')
print(len(fs), 'training motions in $P shards')"
  for d in shard0 shard1 shard2 proto_eval_familiar proto_eval_dance; do wait_ok
    (cd $PM && $PY protomotions/components/motion_lib.py --motion-path $M/$d --output-file $M/lib_$d.pt > $M/logs/lib_$d.log 2>&1)
  done; grep -h "total length" $M/logs/lib_*.log
}
collect() {
  for i in 0 1 2; do
    [ -f $R/data_pm/m500_s$i.npz ] && continue
    wait_ok
    secs=$(grep -o "total length of [0-9.]*" $M/logs/lib_shard$i.log | awk '{print $4}')
    steps=$(awk -v s="$secs" 'BEGIN { printf "%d", s * 50 }')
    (cd $PM && PM_OUT=$R/data_pm/m500_s$i.npz PM_STEPS=$steps PM_SEED=$((60+i)) PM_MOTION_NAME=m500 \
       $PY $OURS/pm_collect.py --simulator mujoco --headless --num-envs 1 --checkpoint $CK --motion-file $M/lib_shard$i.pt > $M/logs/col_$i.log 2>&1) &
    sleep 60                                                     # let each process settle before checking resources again
  done; wait; ls -la $R/data_pm/m500_s*.npz
}
train() {
  cd $OURS; MOT=walk,wave,high5,jumpjack,m500; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  wait_ok; python3 train_student.py      --kind mlp   --name r0_mlp_m500  --train_motions $MOT --seed 0 > $M/logs/train_r0.log 2>&1
  wait_ok; python3 train_student_prob.py --kind gauss --name r3_gll_m500  --train_motions $MOT --seed 0 --lastlayer > $M/logs/train_r3.log 2>&1
  wait_ok; python3 train_student_prob.py --kind mwv2  --name r5_mwll_m500 --train_motions $MOT --seed 0 --lastlayer > $M/logs/train_r5.log 2>&1
  grep -h '^{' $M/logs/train_r*.log | cut -c1-200; grep -h "teacher fell" $M/logs/train_r0.log | grep m500
}
evaluate() {
  J=$M/jobs.txt; : > $J
  for arm in r0_mlp r3_gll r5_mwll; do
    echo "${arm}-m500__dance__n0__s0 $ST/${arm}_m500.pt dance 0 0" >> $J
    for nz in 0.1 0.2; do for es in 0 1 2; do echo "${arm}-m500__dance__n${nz}__s${es} $ST/${arm}_m500.pt dance $nz $es" >> $J; done; done
    for k in familiar dance; do echo "${arm}-m500__heldout_k${k}__n0__s0 $ST/${arm}_m500.pt $M/lib_proto_eval_$k.pt 0 0" >> $J; done
  done
  for k in familiar dance; do echo "teacher__heldout_k${k}__n0__s0 teacher $M/lib_proto_eval_$k.pt 0 0" >> $J; done
  # run in batches of P, checking resources before each batch
  split -l $P $J $M/jobs_batch_
  for b in $M/jobs_batch_*; do wait_ok; OUT=$R/eval_prob/m500 $R/scripts/run_eval_prob.sh $b $P; done
  rm -f $M/jobs_batch_*
}
agg() {
  cd $R/scripts && python3 aggregate_retry.py --eval $R/eval_prob/m500 --data $R/data_pm --train_motions walk,wave,high5,jumpjack,m500 \
     --out ../results/SUMMARY_m500.md > /dev/null && echo "summary -> results/SUMMARY_m500.md"
}
cd $HERE
case ${1:-all} in
  select) select_ ;; convert) convert ;; collect) collect ;; train) train ;; eval) evaluate ;; agg) agg ;;
  all) for st in select_ convert collect train evaluate agg; do echo "[$(date +%T)] $st"; $st; done; echo "[$(date +%T)] M500 DONE" ;;
esac
