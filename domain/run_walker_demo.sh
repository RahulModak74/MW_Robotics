#!/usr/bin/env bash
# run_walker_demo.sh — launch the interactive Walker2d demo.
# Bootstrap: creates walker_demo/ dir with walker_app.py, walker_animation.py,
# and static/index.html if missing. Then starts Flask on :5051.
#
#   cd ~/MW_Robotics/domain && chmod +x run_walker_demo.sh
#   ./run_walker_demo.sh              # start the server
#   ./run_walker_demo.sh --install    # only bootstrap files, don't start
#
# Policies loaded:
#   ../cem_best_theta.npy    (balance, 920-step)   — run ./run_plan_a.sh if missing
#   ../cem_walk_theta.npy    (walk)                — run ./run_demo_walker.sh if missing
# MW detector:
#   ../mw_to_protomotions/walker_mw_detector.ckpt  (already in repo)

set -euo pipefail
if [[ -t 1 ]]; then GRN=$'\033[32m'; RED=$'\033[31m'; YEL=$'\033[33m'; BLU=$'\033[34m'; RST=$'\033[0m'
else GRN=""; RED=""; YEL=""; BLU=""; RST=""; fi
say(){ printf "%s [%s] %s\n" "$(date -u +%H:%M:%SZ)" "${BLU}walker${RST}" "$*"; }
ok(){  printf "%s [%s] %s\n" "$(date -u +%H:%M:%SZ)" "${GRN}walker${RST}" "$*"; }
die(){ printf "%s [%s] %s\n" "$(date -u +%H:%M:%SZ)" "${RED}walker${RST}" "$*" >&2; exit 1; }

INSTALL_ONLY=0
[[ "${1:-}" == "--install" ]] && INSTALL_ONLY=1

TRAIN_DET=0
if [[ "${1:-}" == "--train-detector" ]]; then
    TRAIN_DET=1
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../mw_to_protomotions/cem_policy_search.py" ]]; then
    REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "$SCRIPT_DIR/mw_to_protomotions/cem_policy_search.py" ]]; then
    REPO="$SCRIPT_DIR"
else
    die "cannot find mw_to_protomotions/cem_policy_search.py — run from MW_Robotics/domain/"
fi
DEMO_DIR="$REPO/domain/walker_demo"
mkdir -p "$DEMO_DIR/static"

# ── check policies
[[ -f "$REPO/cem_best_theta.npy" ]] || say "${YEL}warning: cem_best_theta.npy missing — run ./run_plan_a.sh first${RST}"
[[ -f "$REPO/cem_walk_theta.npy" ]] || say "${YEL}warning: cem_walk_theta.npy missing — run ./run_demo_walker.sh first (walk button will 404)${RST}"

# ── check deps
PY="${PYTHON:-python3}"
"$PY" -c "import flask, gymnasium, numpy, pandas, torch, pyro" \
    || die "missing deps: pip install flask 'gymnasium[mujoco]' numpy pandas torch pyro-ppl"
export MW_DEVICE=cpu CUDA_VISIBLE_DEVICES=""
ok "Python + deps present"

# ── ensure MW detector exists ────────────────────────────────────────────────
DET_CKPT="$REPO/mw_to_protomotions/walker_mw_detector.ckpt"
if [[ ! -f "$DET_CKPT" || $TRAIN_DET -eq 1 ]]; then
    say "MW detector missing (or --train-detector set) - training now (~2 min)"
    mkdir -p "$REPO/domain/logs"
    "$PY" - <<PYEOF 2>&1 | tee -a "$REPO/domain/logs/detector_train_$(date -u +%Y%m%d_%H%M%S).log"
import sys, os
sys.path.insert(0, "$REPO")
import pyro
from mw_framework import MWDetector
from mw_framework.core import set_seed
from mw_to_protomotions.walker_domain import WalkerPostureVAE, FEATURES
from mw_to_protomotions.datagen import collect

set_seed(0); pyro.clear_param_store()
print("  collecting PD trajectories...", flush=True)
df = collect(n_episodes=60, max_steps=300)
normal = df[df.label == "normal"].reset_index(drop=True)
print(f"  {len(normal)} normal rows, training WalkerPostureVAE...", flush=True)
det = MWDetector(vae=WalkerPostureVAE(), epochs=200, lr=2e-3).fit(normal[FEATURES])
det.save("$REPO/mw_to_protomotions/walker_mw_detector.ckpt")
print("  saved $REPO/mw_to_protomotions/walker_mw_detector.ckpt")
PYEOF
    ok "detector trained"
fi

if [[ $INSTALL_ONLY -eq 1 ]]; then
    say "--install: files should already exist if you pasted them; nothing to do"
    exit 0
fi

say "starting Flask on http://127.0.0.1:5051"
say "  policies loaded from: $REPO/cem_best_theta.npy, $REPO/cem_walk_theta.npy"
say "  MW detector       : $REPO/mw_to_protomotions/walker_mw_detector.ckpt"
cd "$DEMO_DIR"
exec "$PY" walker_app.py
