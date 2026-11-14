# sourced by the run_*.sh scripts here. Set PM to your ProtoMotions clone and PY to its Python (MuJoCo backend), as in ../../student.
C=${C:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}            # this folder (step1_extraction/cpp_student)
S=$C/../student; F=$C/../flight_computer
PM=${PM:-$HOME/ProtoMotions}
PY=${PY:-python}
OURS=$C/gate
export PYTHONPATH="$PM:$C/gate:$C/kernel:$S/ours:$F" TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CK=data/pretrained_models/motion_tracker/g1-bones-deploy/inference_last.ckpt   # relative to $PM
ST=${ST:?set ST to the 100-motion student checkpoint (r7c_mwv3ctl_s100.pt, from ../../student/kimodo_data/s100_priors.sh)}
E=${E:?set E to the folder holding lib_proto_eval_familiar.pt and lib_proto_eval_dance.pt (the 51 held-out motions)}
OUT=${OUT:-$C/results_runs}
# config D: the envelope and search that scored 58/58 (same values as ../../flight_computer/run_fbw_D.sh)
export FBW_SIG_LEG=0.25 FBW_SIG_WAIST=0.2 FBW_SIG_ARM=0.35 FBW_ITERS=1 FBW_Z_MIN=0.6 FBW_CP_TOL=0.05 FBW_UP_MIN=0.643 \
       FBW_W_Z=0 FBW_W_SPIN=0 FBW_QUIET=1.0 FBW_W_CP=50 FBW_W_UP=50 FBW_FALL=1 FBW_THREADS=${FBW_THREADS:-6}

run_one() {     # tag motion noise seed   (resumable: skips a tag that has a .done file)
  tag=$1; m=$2; nz=$3; sd=$4
  [ -f "$O/$tag.done" ] && return
  case "$m" in /*|*.pt) mf=$m ;; *) mf=data/g1-kimodo-generated/proto/output_${m}.motion ;; esac
  echo "[$(date +%T)] $tag"
  (cd "$PM" && FBW_LA_NOISE=$nz FBW_SEED=$sd nice -n 10 "$PY" "$OURS/pm_eval_fbw2.py" --student "$ST" --fbw2 --fbw2-out "$O/${tag}_fbw2.npz" \
     --act-noise $nz --seed $sd --simulator mujoco --headless --num-envs 1 --full-eval --checkpoint $CK --motion-file "$mf" > "$O/$tag.log" 2>&1) \
     && touch "$O/$tag.done"
  grep -h "success_rate\|\[fbw2\] steps" "$O/$tag.log"
}

all_58() {      # the 58 tests: the dance clip clean + noise 0.1 / 0.2 x 3 seeds, 21 familiar motions, 30 generated dances
  run_one dance__n0__s0 dance 0 0
  for nz in 0.1 0.2; do for es in 0 1 2; do run_one dance__n${nz}__s${es} dance $nz $es; done; done
  for k in familiar dance; do run_one heldout_k${k}__n0__s0 "$E/lib_proto_eval_$k.pt" 0 0; done
}

score() {
  "$PY" - "$O" <<'PY'
import sys, glob, re, numpy as np
O = sys.argv[1]; ok = n = 0; G = []; T = []; P = []
for l in sorted(glob.glob(f"{O}/*.log")):
    t = open(l, errors="ignore").read(); m = re.search(r"success_rate: ([\d.]+)", t)
    if not m: continue
    k = 21 if "kfamiliar" in l else 30 if "kdance" in l else 1
    s = round(float(m.group(1)) * k); ok += s; n += k
    z = np.load(l[:-4] + "_fbw2.npz"); G.append(z["gate"]); T.append(z["t_fbw"]); P.append(z["t_pilot"])
    print(f"{l.split('/')[-1][:-4]:32s} {s:>2d}/{k}  gate open {z['gate'].mean():.3f}  flight computer p50 {np.median(z['t_fbw'])*1e3:.1f} "
          f"p99 {np.percentile(z['t_fbw'],99)*1e3:.0f} max {z['t_fbw'].max()*1e3:.0f} ms")
if n:
    G, T, P = map(np.concatenate, (G, T, P)); tot = T + P
    print(f"\nscore {ok}/{n}   (student alone 40/58; full search on every step 58/58)")
    print(f"steps {len(G)}; gate open {G.mean():.3f}; decision time per step (student + gate + flight computer): median {np.median(tot)*1e3:.2f} ms, "
          f"p99 {np.percentile(tot,99)*1e3:.0f} ms, max {tot.max()*1e3:.0f} ms; within 20 ms: {(tot <= 0.02).mean():.3f}; "
          f"within 200 ms: {(tot <= 0.2).mean():.3f}; gated steps mean {T[G].mean()*1e3:.0f} ms")
PY
}
