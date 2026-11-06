#!/usr/bin/env bash
# quick100.sh - quick check: do ~100 extra (unconditional Kimodo) motions help the students vs the 4-motion students?
# gen 120 unconditional motions -> 100 train + 20 held out; teacher rollouts on the 100; retrain r0/r3/r5 on 4 original + 100;
# evaluate new and old (dance fold) students on dance (held out for both) and on the 20 held-out motions.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); R=$(cd "$HERE/.." && pwd); Q=$HERE/quick100; mkdir -p $Q/logs
KPY=$HOME/kimodo/venv/bin/python; CONV=$HOME/ProtoMotions/data/scripts/convert_g1_csv_to_proto.py
source "$R/scripts/common.sh"
STAGE=${1:-all}
gen() {
  python3 -c "import json; json.dump([{'id': f'uncond_{i:03d}', 'cat': 'uncond', 'text': ''} for i in range(15)], open('$Q/prompts.json', 'w'))"
  mkdir -p $Q/emb; (cd $HERE && $KPY gen_motions.py --prompts $Q/prompts.json --emb $Q/emb --out $Q/csv --samples 8 > $Q/logs/gen.log 2>&1)
  rm -rf $Q/csv_train $Q/csv_test; mkdir -p $Q/csv_train $Q/csv_test
  i=0; for f in $(ls $Q/csv/uncond/*.csv | sort); do if [ $i -lt 100 ]; then ln -s $f $Q/csv_train/; else ln -s $f $Q/csv_test/; fi; i=$((i+1)); done
  for s in train test; do
    (cd $PM && $PY $CONV --input-dir $Q/csv_$s --output-dir $Q/proto_$s --input-fps 30 --output-fps 30 --pos-units m --rot-format quat_wxyz \
       --joint-units rad --no-has-header --no-has-frame-column > $Q/logs/conv_$s.log 2>&1)
    (cd $PM && $PY protomotions/components/motion_lib.py --motion-path $Q/proto_$s --output-file $Q/lib_$s.pt > $Q/logs/lib_$s.log 2>&1)
  done; grep -h "total length" $Q/logs/lib_*.log
}
collect() {   # 5 processes x 10k steps on the 100-motion library -> data_pm/uncond_s<i>.npz
  for i in 0 1 2 3 4; do
    (cd $PM && PM_OUT=$R/data_pm/uncond_s$i.npz PM_STEPS=10000 PM_SEED=$((20+i)) PM_MOTION_NAME=uncond \
       $PY $OURS/pm_collect.py --simulator mujoco --headless --num-envs 1 --checkpoint $CK --motion-file $Q/lib_train.pt > $Q/logs/col_$i.log 2>&1) &
  done; wait; ls -la $R/data_pm/uncond_s*.npz
}
train() {
  cd $OURS; M=walk,wave,high5,jumpjack,uncond
  { python3 train_student.py      --kind mlp   --name r0_mlp_q100_s0  --train_motions $M --seed 0
    python3 train_student_prob.py --kind gauss --name r3_gll_q100_s0  --train_motions $M --seed 0 --lastlayer
    python3 train_student_prob.py --kind mwv2  --name r5_mwll_q100_s0 --train_motions $M --seed 0 --lastlayer
  } > $Q/logs/train.log 2>&1; grep -E '^\{|teacher fell|data:' $Q/logs/train.log
}
evaluate() {
  J=$Q/jobs.txt; : > $J; ST=$R/students
  for nz in 0 0.1 0.2; do
    for arm in r0_mlp r3_gll r5_mwll; do
      for es in 0 1 2; do echo "${arm}-q100__dance__n${nz}__s${es} $ST/${arm}_q100_s0.pt dance $nz $es" >> $J; done
      echo "${arm}-q100__heldout_uncond__n${nz}__s0 $ST/${arm}_q100_s0.pt $Q/lib_test.pt $nz 0" >> $J
      echo "${arm}-4mot__heldout_uncond__n${nz}__s0 $ST/${arm}_dance_s0.pt $Q/lib_test.pt $nz 0" >> $J
    done
    echo "teacher__heldout_uncond__n${nz}__s0 teacher $Q/lib_test.pt $nz 0" >> $J
  done
  OUT=$R/eval_prob/quick100 $R/scripts/run_eval_prob.sh $J 5
  # the 4-motion students' dance runs already exist in eval_prob/dance_s0; link them in for one table
  for f in $R/eval_prob/dance_s0/{r0_mlp,r3_gll,r5_mwll,teacher}__dance__*; do
    b=$(basename $f); a=${b%%__*}; [ "$a" = teacher ] && n=$b || n=${a}-4mot__${b#*__}; ln -sf $f $R/eval_prob/quick100/$n; done
  cd $R/scripts && python3 aggregate_retry.py --eval $R/eval_prob/quick100 --data $R/data_pm --train_motions walk,wave,high5,jumpjack,uncond \
     --out ../results/SUMMARY_quick100.md > /dev/null && echo "summary -> results/SUMMARY_quick100.md"
}
case $STAGE in gen) gen ;; collect) collect ;; train) train ;; eval) evaluate ;; all) gen; collect; train; evaluate ;; esac
