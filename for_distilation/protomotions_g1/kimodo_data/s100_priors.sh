#!/usr/bin/env bash
# s100_priors.sh - 100 motions chosen evenly (25 each: locomotion, exercise, everyday, gesture; s100_motions.txt) from the m500 teacher
# rollouts: strong physics priors (MW v3, lambda 140, 10-step look-ahead) vs the identical control (lambda 0). Needs m500.sh collect first.
#   ./s100_priors.sh [train|eval|agg|all]      (GPU for training if available: PRI_DEVICE=cuda, default; PRI_DEVICE=cpu otherwise)
set -e
HERE=$(cd "$(dirname "$0")" && pwd); R=$(cd "$HERE/.." && pwd); M=$HERE/s100; mkdir -p $M/logs; E=$HERE/m500; P=3; LIMIT=${LIMIT:-70}
source "$R/scripts/common.sh"
ST=$R/students; MOT=walk,wave,high5,jumpjack,m500
usage() { local a b; a=($(head -1 /proc/stat)); sleep 2; b=($(head -1 /proc/stat))
  local idle=$(( (b[4]+b[5]) - (a[4]+a[5]) )); local tot=0; for i in 1 2 3 4 5 6 7 8; do tot=$(( tot + b[i] - a[i] )); done
  local mem=$(free | awk '/Mem:/ {printf "%d", 100 * ($2 - $7) / $2}'); echo $(( 100 * (tot - idle) / (tot > 0 ? tot : 1) )) $mem; }
wait_ok() { while true; do read c m <<< "$(usage)"; [ $c -lt $LIMIT ] && [ $m -lt $LIMIT ] && return; echo "[$(date +%T)] waiting: cpu $c% mem $m%"; sleep 30; done; }
COMMON="--motion_subset $HERE/s100_motions.txt --device ${PRI_DEVICE:-cuda} --ckpt_every 500 --train_motions $MOT --seed 0 --lastlayer --threads 6 --steps 6000 --tol_q 0.8 --horizon 10 --law_weights 1,1,0.5,0.5"
train() {
  cd $OURS; export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
  [ -f $ST/r7c_mwv3ctl_s100.pt ] || { wait_ok; python3 train_student_priors.py --name r7c_mwv3ctl_s100 --lam 0   $COMMON >> $M/logs/train_r7c.log 2>&1; }
  [ -f $ST/r7_mwpri_s100.pt ]    || { wait_ok; python3 train_student_priors.py --name r7_mwpri_s100    --lam 140 $COMMON >> $M/logs/train_r7.log 2>&1; }
  grep -h '^{' $M/logs/train_r7*.log
}
evaluate() {
  J=$M/jobs_s100.txt; : > $J
  for arm in r7c_mwv3ctl r7_mwpri; do
    echo "${arm}-s100__dance__n0__s0 $ST/${arm}_s100.pt dance 0 0" >> $J
    for nz in 0.1 0.2; do for es in 0 1 2; do echo "${arm}-s100__dance__n${nz}__s${es} $ST/${arm}_s100.pt dance $nz $es" >> $J; done; done
    for k in familiar dance; do echo "${arm}-s100__heldout_k${k}__n0__s0 $ST/${arm}_s100.pt $E/lib_proto_eval_$k.pt 0 0" >> $J; done
  done
  split -l $P $J $M/jobs_s100_batch_
  for b in $M/jobs_s100_batch_*; do wait_ok; OUT=$R/eval_prob/s100 $R/scripts/run_eval_prob.sh $b $P; done
  rm -f $M/jobs_s100_batch_*
}
agg() {
  cd $R/scripts && python3 aggregate_retry.py --eval $R/eval_prob/s100 --data $R/data_pm --train_motions $MOT \
     --out ../results/SUMMARY_s100_priors.md > /dev/null && echo "summary -> results/SUMMARY_s100_priors.md"
}
cd $HERE
case ${1:-all} in
  train) train ;; eval) evaluate ;; agg) agg ;;
  all) for st in train evaluate agg; do echo "[$(date +%T)] $st"; $st; done; echo "[$(date +%T)] S100 PRIORS DONE" ;;
esac
