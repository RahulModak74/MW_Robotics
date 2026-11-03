#!/usr/bin/env bash
# run_demo_walker.sh — train a WALKING (not standing) Walker2d policy.
#
# Produces cem_walk_theta.npy (repo root), used by run_walker_demo.sh's
# "walk" button. Uses forward-progress fitness, not survival — this is the
# fix for the reward-hacking pattern race_demo.py documented: survival alone
# → stands still; forward distance alone → faceplants; forward + small
# survival → walks.
#
# Usage:
#   cd ~/MW_Robotics/domain
#   chmod +x run_demo_walker.sh
#   ./run_demo_walker.sh              # full train (~25 min CPU) + verify
#   ./run_demo_walker.sh --quick      # 5 gens, ~2 min, smoke test
#   ./run_demo_walker.sh --verify     # just check what's on disk, no training
#   ./run_demo_walker.sh --retrain    # ignore existing theta, force retrain
#   ./run_demo_walker.sh --dry-run
#   ./run_demo_walker.sh --with-video # also record MP4s (needs imageio)
#
# Exits 0 = theta produced/verified.  1 = setup error.

set -euo pipefail
if [[ -t 1 ]]; then
    GRN=$'\033[32m'; RED=$'\033[31m'; YEL=$'\033[33m'; BLU=$'\033[34m'; RST=$'\033[0m'
else
    GRN=""; RED=""; YEL=""; BLU=""; RST=""
fi
ts()  { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say() { printf "%s [%s] %s\n" "$(ts)" "${BLU}walk${RST}"  "$*"; }
ok()  { printf "%s [%s] %s\n" "$(ts)" "${GRN}walk${RST}"  "$*"; }
warn(){ printf "%s [%s] %s\n" "$(ts)" "${YEL}walk${RST}"  "$*" >&2; }
die() { printf "%s [%s] %s\n" "$(ts)" "${RED}walk${RST}"  "$*" >&2; exit 1; }

# ── args ─────────────────────────────────────────────────────────────────────
DRY=0; QUICK=0; VERIFY=0; RETRAIN=0; WITH_VIDEO=0
GENERATIONS=30; POP=80; MAX_STEPS=600
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY=1 ;;
        --quick)      QUICK=1; GENERATIONS=5; POP=32 ;;
        --verify)     VERIFY=1 ;;
        --retrain)    RETRAIN=1 ;;
        --with-video) WITH_VIDEO=1 ;;
        --generations) GENERATIONS="${2:?}"; shift ;;
        --pop)         POP="${2:?}"; shift ;;
        -h|--help)     sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown arg: $1 (see --help)" ;;
    esac
    shift
done

# ── repo root detection (same pattern as run_plan_a.sh — the one that works) ─
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../mw_to_protomotions/cem_policy_search.py" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "$SCRIPT_DIR/mw_to_protomotions/cem_policy_search.py" ]]; then
    REPO_ROOT="$SCRIPT_DIR"
else
    die "cannot find mw_to_protomotions/cem_policy_search.py — run from MW_Robotics/domain/"
fi
cd "$REPO_ROOT"
DOMAIN_DIR="$REPO_ROOT/domain"
mkdir -p "$DOMAIN_DIR/logs"
LOG_FILE="$DOMAIN_DIR/logs/walk_$(date -u +%Y%m%d_%H%M%S).log"
THETA_PATH="$REPO_ROOT/cem_walk_theta.npy"
SUMMARY_PATH="$REPO_ROOT/cem_walk_summary.json"

say "repo root : $REPO_ROOT"
say "theta     : $THETA_PATH"
say "log       : $LOG_FILE"

# ── verify mode: report and exit ─────────────────────────────────────────────
if [[ $VERIFY -eq 1 ]]; then
    say "verify mode — checking artifacts"
    for f in cem_best_theta.npy cem_walk_theta.npy cem_summary.json cem_walk_summary.json; do
        if [[ -f "$REPO_ROOT/$f" ]]; then
            size=$(stat -c%s "$REPO_ROOT/$f")
            mtime=$(stat -c%y "$REPO_ROOT/$f" | cut -d. -f1)
            ok "✓ $f  (${size} bytes, $mtime)"
        else
            warn "✗ $f  (missing)"
        fi
    done
    exit 0
fi

# ── dep check ────────────────────────────────────────────────────────────────
PY="${PYTHON:-python3}"
"$PY" -c "import sys; assert sys.version_info >= (3,9)" || die "need Python 3.9+"
"$PY" -c "import gymnasium, numpy, torch" \
    || die "missing deps: pip install 'gymnasium[mujoco]' numpy torch"
if [[ $WITH_VIDEO -eq 1 ]]; then
    "$PY" -c "import imageio" 2>/dev/null \
        || die "--with-video needs imageio: pip install imageio imageio-ffmpeg"
fi
ok "Python + deps present"
export MW_DEVICE=cpu CUDA_VISIBLE_DEVICES=""
say "MW_DEVICE=cpu, CUDA_VISIBLE_DEVICES=''"

# ── dry-run ──────────────────────────────────────────────────────────────────
if [[ $DRY -eq 1 ]]; then
    say "dry-run — would:"
    if [[ -f "$THETA_PATH" && $RETRAIN -eq 0 ]]; then
        say "  reuse existing $THETA_PATH (use --retrain to force retrain)"
    else
        say "  train: pop=$POP gens=$GENERATIONS max_steps=$MAX_STEPS"
        say "  fitness: forward distance + 0.02 * survival steps"
        say "  save: $THETA_PATH"
    fi
    [[ $WITH_VIDEO -eq 1 ]] && say "  record: domain/walker_demo/walk_forward.mp4"
    say "dry-run complete."
    exit 0
fi

# ── reuse or train ───────────────────────────────────────────────────────────
if [[ -f "$THETA_PATH" && $RETRAIN -eq 0 ]]; then
    ok "cem_walk_theta.npy already exists — skipping training (use --retrain to force)"
    ok "run ./run_walker_demo.sh to see it in the browser"
    exit 0
fi

say "step 1/2 — training walk policy (forward-progress fitness)"
say "            pop=$POP gens=$GENERATIONS max_steps=$MAX_STEPS"
say "            wall-clock: ~$((QUICK ? 120 : 1500))s on a laptop CPU"

set +e
"$PY" - "$GENERATIONS" "$POP" "$MAX_STEPS" "$THETA_PATH" "$SUMMARY_PATH" <<'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import sys, os, time, json
sys.path.insert(0, os.getcwd())
import numpy as np
import gymnasium as gym
from mw_to_protomotions.cem_policy_search import (
    get_features, policy_action, DIM)

GENERATIONS = int(sys.argv[1])
POP         = int(sys.argv[2])
MAX_STEPS   = int(sys.argv[3])
THETA_PATH  = sys.argv[4]
SUMMARY_PATH = sys.argv[5]

SURVIVAL_WEIGHT = 0.02   # small, so forward distance dominates but falling still costs


def rollout_forward(theta, seed, max_steps):
    """Net forward distance + small survival bonus. The exact fitness fix
    documented in race_demo.py bug #2."""
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    x0 = data.qpos[0]
    steps = 0
    for _ in range(max_steps):
        obs, r, term, trunc, info = env.step(policy_action(theta, get_features(data)))
        steps += 1
        if term:
            break
    x_final = data.qpos[0]
    env.close()
    return float((x_final - x0) + SURVIVAL_WEIGHT * steps)


def fitness(theta, seeds, max_steps):
    return float(np.mean([rollout_forward(theta, s, max_steps) for s in seeds]))


def cem():
    """CEM with FRESH random seeds each generation — the overfitting fix
    from stage4/6 (fixed training seeds generalize poorly to held-out)."""
    rng = np.random.default_rng(0)
    mean = np.zeros(DIM)
    std  = np.ones(DIM) * 0.5
    n_elite = max(1, int(POP * 0.2))
    for gen in range(GENERATIONS):
        pop = mean + std * rng.standard_normal((POP, DIM))
        gen_seeds = rng.integers(0, 100_000, size=3).tolist()   # fresh every gen
        fits = np.array([fitness(t, gen_seeds, MAX_STEPS) for t in pop])
        elite = pop[np.argsort(fits)[-n_elite:]]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
        print(f"  gen {gen:2d}  pop_mean={fits.mean():8.3f}  "
              f"pop_best={fits.max():8.3f}  std={np.linalg.norm(std):.3f}",
              flush=True)
    return mean


t0 = time.time()
theta = cem()
search_time = time.time() - t0
print(f"\n[search] {search_time:.1f}s, |theta|={np.linalg.norm(theta):.3f}", flush=True)

# Held-out eval on 20 fresh seeds (disjoint from training)
held_out_seeds = list(range(2000, 2020))
dists, survs = [], []
for s in held_out_seeds:
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    obs, info = env.reset(seed=s)
    data = env.unwrapped.data
    x0 = data.qpos[0]
    steps = 0
    for _ in range(1000):
        obs, r, term, trunc, info = env.step(policy_action(theta, get_features(data)))
        steps += 1
        if term:
            break
    dists.append(float(data.qpos[0] - x0))
    survs.append(steps)
    env.close()

dists, survs = np.array(dists), np.array(survs)
print(f"\n[held-out, {len(held_out_seeds)} seeds, cap 1000]")
print(f"  mean forward distance = {dists.mean():.2f} m")
print(f"  mean steps survived   = {survs.mean():.1f} / 1000")
print(f"  fall rate             = {(survs < 1000).mean():.1%}")

np.save(THETA_PATH, theta.astype(np.float32))
print(f"\n[save] {THETA_PATH}")

summary = dict(
    config=dict(pop=POP, gens=GENERATIONS, max_steps=MAX_STEPS,
                n_params=int(DIM), survival_weight=SURVIVAL_WEIGHT),
    fitness="forward_distance + 0.02 * survival_steps",
    search_time_s=float(search_time),
    held_out=dict(n=len(held_out_seeds),
                  forward_mean_m=float(dists.mean()),
                  forward_median_m=float(np.median(dists)),
                  steps_mean=float(survs.mean()),
                  fall_rate=float((survs < 1000).mean())),
    theta_path=THETA_PATH,
)
with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[summary] {SUMMARY_PATH}")
PYEOF
PY_RC=${PIPESTATUS[0]}
set -e

[[ $PY_RC -eq 0 ]] || die "training failed (exit $PY_RC) — see $LOG_FILE"

# ── step 2: verify ───────────────────────────────────────────────────────────
say "step 2/2 — verifying artifacts"
[[ -f "$THETA_PATH" ]]   || die "theta not produced: $THETA_PATH"
[[ -f "$SUMMARY_PATH" ]] || die "summary not produced: $SUMMARY_PATH"

FWD=$(python3 -c "import json; s=json.load(open('$SUMMARY_PATH')); print(f\"{s['held_out']['forward_mean_m']:.2f}\")")
STEPS=$(python3 -c "import json; s=json.load(open('$SUMMARY_PATH')); print(f\"{s['held_out']['steps_mean']:.0f}\")")
FALL=$(python3 -c "import json; s=json.load(open('$SUMMARY_PATH')); print(f\"{s['held_out']['fall_rate']*100:.1f}%\")")

{
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  WALK POLICY — READY"
    echo "════════════════════════════════════════════════════════════════"
    echo "  theta file           : $THETA_PATH"
    echo "  held-out forward     : $FWD m  (mean across 20 seeds)"
    echo "  held-out survival    : $STEPS / 1000 steps"
    echo "  fall rate            : $FALL"
    echo ""
    echo "  Next: restart the Flask demo so it picks up the new policy"
    echo "    ./run_walker_demo.sh"
    echo "    open http://127.0.0.1:5051   →  click 'walk'"
    echo "════════════════════════════════════════════════════════════════"
} | tee -a "$LOG_FILE"

# ── optional: record MP4 ─────────────────────────────────────────────────────
if [[ $WITH_VIDEO -eq 1 ]]; then
    say "recording walk MP4 (~1 min)"
    "$PY" - "$THETA_PATH" <<'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import gymnasium as gym
import imageio.v2 as iio
from mw_to_protomotions.cem_policy_search import get_features, policy_action

theta = np.load(sys.argv[1])
out = "domain/walker_demo/walk_forward.mp4"
os.makedirs(os.path.dirname(out), exist_ok=True)

env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False,
               render_mode="rgb_array")
obs, info = env.reset(seed=0)
data = env.unwrapped.data
x0 = data.qpos[0]
frames, steps = [], 0
while steps < 600:
    frames.append(env.render())
    obs, r, term, trunc, info = env.step(policy_action(theta, get_features(data)))
    steps += 1
    if term:
        break
env.close()
iio.mimsave(out, frames, fps=60)
print(f"[video] {out}  ({steps} steps, {data.qpos[0]-x0:.2f} m forward)")
PYEOF
    ok "video recorded"
fi

exit 0
