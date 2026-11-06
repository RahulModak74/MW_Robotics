#!/usr/bin/env bash
# run_pipeline.sh <stage>  - Kimodo-generated G1 motion data for the scaled-up distillation (Retry_MW_distillation.md, data lever).
#   encode   : LLM2Vec (4-bit) embeddings for prompts.json            (GPU, Kimodo venv)
#   generate : Kimodo-G1-RP-v1 motions, SAMPLES per prompt -> csv/   (GPU, Kimodo venv)
#   convert  : csv -> ProtoMotions .motion, split into train / heldout_seen / heldout_dance, package motion libs (CPU, venv_pm)
#   collect  : teacher rollouts on the train shards, P processes -> ../data_pm_kimodo/ (CPU, venv_pm)
# Split (fixed, by prompt): dance category = held out entirely; in the other 4 categories every 10th prompt id = held out
# ("new motion of a familiar kind"). Everything else = training.
set -e
STAGE=$1; HERE=$(cd "$(dirname "$0")" && pwd); R=$(cd "$HERE/.." && pwd)
KPY=${KPY:-$HOME/kimodo/venv/bin/python}; SAMPLES=${SAMPLES:-8}; P=${P:-6}; SHARDS=${SHARDS:-$P}
CONV=${CONV:-$HOME/ProtoMotions/data/scripts/convert_g1_csv_to_proto.py}     # not in the vendored snapshot
source "$R/scripts/common.sh"
cd "$HERE"
case $STAGE in
encode)   LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH HF_DEACTIVATE_ASYNC_LOAD=1 HF_HUB_OFFLINE=1 $KPY encode_prompts.py --prompts prompts.json --out emb --work work ;;
generate) $KPY gen_motions.py --prompts prompts.json --emb emb --out csv --samples $SAMPLES --steps ${GEN_STEPS:-50} ;;
convert)
  python3 - <<'EOF'
import json, os, shutil
P = json.load(open("prompts.json"))
split = {}
for p in P:
    n = int(p["id"].split("_")[-1])
    split[p["id"]] = "heldout_dance" if p["cat"] == "dance" else ("heldout_seen" if n % 10 == 0 else "train")
json.dump(split, open("split.json", "w"), indent=0)
for s in ("train", "heldout_seen", "heldout_dance"):
    d = f"csv_split/{s}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
for cat in os.listdir("csv"):
    if not os.path.isdir(f"csv/{cat}"):
        continue
    for f in os.listdir(f"csv/{cat}"):
        pid = "_".join(f.split("_")[:2])
        os.symlink(os.path.abspath(f"csv/{cat}/{f}"), f"csv_split/{split[pid]}/{f}")
print({s: len(os.listdir(f"csv_split/{s}")) for s in ("train", "heldout_seen", "heldout_dance")})
EOF
  for s in train heldout_seen heldout_dance; do
    (cd "$PM" && "$PY" "$CONV" --input-dir "$HERE/csv_split/$s" --output-dir "$HERE/proto/$s" --input-fps 30 --output-fps 30 \
       --pos-units m --rot-format quat_wxyz --joint-units rad --no-has-header --no-has-frame-column > "$HERE/convert_$s.log" 2>&1)
  done
  # training motions -> SHARDS libs (one per collector process); held-out sets -> one lib each
  python3 - "$SHARDS" <<'EOF'
import os, sys, shutil
n = int(sys.argv[1]); fs = sorted(os.listdir("proto/train"))
for i in range(n):
    d = f"proto/train_shard{i}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in fs[i::n]:
        os.symlink(os.path.abspath(f"proto/train/{f}"), f"{d}/{f}")
print(len(fs), "training motions in", n, "shards")
EOF
  mkdir -p libs
  for d in proto/train_shard* proto/heldout_seen proto/heldout_dance; do
    b=$(basename $d); (cd "$PM" && "$PY" protomotions/components/motion_lib.py --motion-path "$HERE/$d" --output-file "$HERE/libs/$b.pt" > "$HERE/libs/$b.log" 2>&1) &
  done; wait; ls -la libs/*.pt ;;
collect)
  # steps per shard = COVER x its total clip length at the 50 Hz control rate (COVER=1: random clip sampling visits ~63 % of clips)
  OUT=$R/data_pm_kimodo; mkdir -p "$OUT"
  for lib in libs/train_shard*.pt; do
    i=${lib##*shard}; i=${i%.pt}
    secs=$(grep -o "total length of [0-9.]*" libs/train_shard$i.log | awk '{print $4}')
    steps=$(awk -v s="$secs" -v c="${COVER:-1.0}" 'BEGIN { printf "%d", s * 50 * c }')
    [ -f "$OUT/kimodo_s$i.npz" ] && continue
    (cd "$PM" && PM_OUT=$OUT/kimodo_s$i.npz PM_STEPS=$steps PM_SEED=$i PM_MOTION_NAME=kimodo_shard$i \
       "$PY" "$OURS/pm_collect.py" --simulator mujoco --headless --num-envs 1 --checkpoint $CK --motion-file "$HERE/$lib" \
       > "$OUT/kimodo_s$i.log" 2>&1) &
  done; wait; ls -la "$OUT" ;;
*) echo "stage: encode | generate | convert | collect"; exit 1 ;;
esac
