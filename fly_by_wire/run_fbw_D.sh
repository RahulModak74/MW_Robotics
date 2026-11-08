#!/usr/bin/env bash
# run_fbw_D.sh <student.pt> [tag]  -  the 58-test FBW protocol with config D (full envelope), one run at a time, resumable.
#   7 dance-clip runs (noise 0 once; noise 0.1 and 0.2, seeds 0-2) + 21 unseen familiar + 30 unseen generated dances (clean).
#   Libraries: ../for_distilation/protomotions_g1/kimodo_data/m500/lib_proto_eval_{familiar,dance}.pt (made by m500.sh).
#   Output: results_runs/fbwD_<tag>__*.log / .npz / _fbw.npz. Then: python fbw_tracking.py (see README).
HERE=$(cd "$(dirname "$0")" && pwd); G=$HERE/../for_distilation/protomotions_g1
ST=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); TAG=${2:-pilot}; E=$G/kimodo_data/m500; O=$HERE/results_runs; mkdir -p $O
# look-ahead and search: 20 control steps (0.4 s), 64 candidates, 4 noise draws, correction spread per joint group
export FBW_H=20 FBW_N=64 FBW_S=4 FBW_ITERS=1 FBW_SIG_LEG=0.25 FBW_SIG_WAIST=0.2 FBW_SIG_ARM=0.35
# envelope D: capture point within 5 cm of the support, torso within 50 deg (cos = 0.643), predicted-fall term on;
# weight 50 per law, act when the pilot's expected cost exceeds 1.0; height and spin laws off
export FBW_CP_TOL=0.05 FBW_UP_MIN=0.643 FBW_W_CP=50 FBW_W_UP=50 FBW_FALL=1 FBW_W_Z=0 FBW_W_SPIN=0 FBW_Z_MIN=0.6 FBW_QUIET=1.0
J=$O/jobs_$TAG.txt
{ echo "fbwD_${TAG}__dance__n0__s0 $ST dance 0 0"
  for nz in 0.1 0.2; do for s in 0 1 2; do echo "fbwD_${TAG}__dance__n${nz}__s$s $ST dance $nz $s"; done; done
  for k in familiar dance; do echo "fbwD_${TAG}__heldout_k${k}__n0__s0 $ST $E/lib_proto_eval_$k.pt 0 0"; done; } > $J
OUT=$O $G/scripts/run_eval_prob.sh $J 1
grep -H -E "success_rate|\[fbw\] steps" $O/fbwD_${TAG}__*.log
