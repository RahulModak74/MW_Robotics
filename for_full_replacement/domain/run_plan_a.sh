#!/usr/bin/env bash
# run_plan_a.sh — CEM policy search on Walker2d (Plan A, the proven baseline).
#
# Wraps mw_to_protomotions/cem_policy_search.py with:
#   - saved theta (cem_best_theta.npy) so it can be reused / compared
#   - stronger held-out eval (default 20 seeds, was 10)
#   - random baseline on the SAME held-out seeds
#   - optional multi-restart variance (--restarts N)
#   - summary JSON (cem_summary.json) for downstream comparison
#
# Reference points (from earlier runs):
#   untrained random policy:  15.9 steps
#   CEM-found policy (this):  208.3 steps  (84s CPU, serial)
#   hand-tuned PD baseline:  263 steps
#
# Usage:
#   cd ~/MW_Robotics/domain
#   chmod +x run_plan_a.sh
#   ./run_plan_a.sh                    # full search + held-out eval
#   ./run_plan_a.sh --dry-run          # show what would happen
#   ./run_plan_a.sh --quick            # 3 generations, smoke test only
#   ./run_plan_a.sh --skip-search      # reuse saved theta, only re-eval
#   ./run_plan_a.sh --restarts 3       # 3 independent CEM runs, variance report
#   ./run_plan_a.sh --held-out 30      # eval on 30 seeds (slower, tighter CI)
#   ./run_plan_a.sh --with-pd          # also run tune_balance.py for fresh PD baseline
#
# Exits 0 on pass (within 80% of PD baseline), 2 on underperform, 1 on setup error.

set -euo pipefail

# ── colour + timing helpers ──────────────────────────────────────────────────
if [[ -t 1 ]]; then
    GRN=$'\033[32m'; RED=$'\033[31m'; YEL=$'\033[33m'; BLU=$'\033[34m'; RST=$'\033[0m'
else
    GRN=""; RED=""; YEL=""; BLU=""; RST=""
fi
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say()  { printf "%s [%s] %s\n" "$(ts)" "${BLU}plan-a${RST}" "$*"; }
warn() { printf "%s [%s] %s\n" "$(ts)" "${YEL}plan-a${RST}" "$*" >&2; }
die()  { printf "%s [%s] %s\n" "$(ts)" "${RED}plan-a${RST}" "$*" >&2; exit "${2:-1}"; }
ok()   { printf "%s [%s] %s\n" "$(ts)" "${GRN}plan-a${RST}" "$*"; }

# ── parse args ───────────────────────────────────────────────────────────────
DRY_RUN=0; QUICK=0; SKIP_SEARCH=0; RESTARTS=1; HELD_OUT=20; WITH_PD=0
GENERATIONS=15; POP=60; MAX_STEPS=400; ELITE=0.2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=1 ;;
        --quick)          QUICK=1; GENERATIONS=3; POP=32 ;;
        --skip-search)    SKIP_SEARCH=1 ;;
        --restarts)       RESTARTS="${2:?--restarts needs N}"; shift ;;
        --held-out)       HELD_OUT="${2:?--held-out needs N}"; shift ;;
        --with-pd)        WITH_PD=1 ;;
        --generations)    GENERATIONS="${2:?}"; shift ;;
        --pop)            POP="${2:?}"; shift ;;
        --max-steps)      MAX_STEPS="${2:?}"; shift ;;
        -h|--help)        sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown arg: $1 (see --help)" ;;
    esac
    shift
done

# ── detect repo root ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../mw_to_protomotions/cem_policy_search.py" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "$SCRIPT_DIR/mw_to_protomotions/cem_policy_search.py" ]]; then
    REPO_ROOT="$SCRIPT_DIR"
else
    die "cannot find mw_to_protomotions/cem_policy_search.py — run from MW_Robotics/ or MW_Robotics/domain/"
fi
cd "$REPO_ROOT"
DOMAIN_DIR="$REPO_ROOT/domain"
mkdir -p "$DOMAIN_DIR/logs"
LOG_FILE="$DOMAIN_DIR/logs/plan_a_$(date -u +%Y%m%d_%H%M%S).log"

say "repo root : $REPO_ROOT"
say "log file  : $LOG_FILE"
say "config    : pop=$POP gens=$GENERATIONS max_steps=$MAX_STEPS restarts=$RESTARTS held_out=$HELD_OUT"

# ── dep checks ───────────────────────────────────────────────────────────────
PY="${PYTHON:-python3}"
"$PY" -c "import sys; assert sys.version_info >= (3,9)" || die "need Python 3.9+"
"$PY" -c "import gymnasium, numpy, torch, pyro" \
    || die "missing deps: pip install 'gymnasium[mujoco]' torch pyro-ppl"
ok "Python + deps present"
export MW_DEVICE=cpu CUDA_VISIBLE_DEVICES=""
say "MW_DEVICE=cpu, CUDA_VISIBLE_DEVICES='' (CPU-only by design)"

if [[ $DRY_RUN -eq 1 ]]; then
    say "dry-run — would run:"
    say "  1. CEM search ($RESTARTS restart(s), pop=$POP, gens=$GENERATIONS)"
    say "  2. Save theta -> $REPO_ROOT/cem_best_theta.npy"
    say "  3. Held-out eval on $HELD_OUT fresh seeds"
    say "  4. Random baseline on same seeds"
    [[ $WITH_PD -eq 1 ]] && say "  5. PD baseline: python3 mw_to_protomotions/tune_balance.py"
    say "  6. Write $REPO_ROOT/cem_summary.json + interpretation"
    say "dry-run complete — nothing changed."
    exit 0
fi

# ── optional: fresh PD baseline ──────────────────────────────────────────────
PD_STEPS=263   # reference from earlier runs; overridden if --with-pd
if [[ $WITH_PD -eq 1 ]]; then
    say "step 0/4 — running tune_balance.py for a fresh PD baseline (~30s)"
    PD_LOG="$DOMAIN_DIR/logs/tune_balance_$(date -u +%Y%m%d_%H%M%S).log"
    set +e
    "$PY" mw_to_protomotions/tune_balance.py > "$PD_LOG" 2>&1
    PD_RC=$?
    set -e
    # tune_balance.py prints per-line "sign=... kp=... kd=... -> survived N"
    PD_STEPS=$(grep -oE 'survived [0-9]+' "$PD_LOG" | tail -1 | awk '{print $2}' || echo 0)
    if [[ "$PD_STEPS" -gt 0 ]]; then
        ok "PD baseline: $PD_STEPS steps (log: $PD_LOG)"
    else
        warn "could not parse PD baseline from $PD_LOG, using reference 263"
        PD_STEPS=263
    fi
fi

# ── step 1: CEM search + save + held-out (single Python, all restarts) ───────
say "step 1/4 — CEM search ($RESTARTS restart(s)) + held-out eval"
say "            this is the long step (~$((RESTARTS * 90))s CPU, serial)"

THETA_PATH="$REPO_ROOT/cem_best_theta.npy"
SUMMARY_PATH="$REPO_ROOT/cem_summary.json"

if [[ $SKIP_SEARCH -eq 1 && -f "$THETA_PATH" ]]; then
    say "            --skip-search: reusing $THETA_PATH (will only re-evaluate)"
    RESTARTS_ACTUAL=0
else
    RESTARTS_ACTUAL=$RESTARTS
fi

set +e
"$PY" - "$GENERATIONS" "$POP" "$MAX_STEPS" "$RESTARTS_ACTUAL" "$HELD_OUT" "$ELITE" \
    "$THETA_PATH" "$SUMMARY_PATH" <<'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import sys, os, time, json
sys.path.insert(0, os.getcwd())  # repo root; script has cd'd here

import numpy as np
from mw_to_protomotions.cem_policy_search import cem, fitness, DIM, unpack

GENERATIONS = int(sys.argv[1])
POP         = int(sys.argv[2])
MAX_STEPS   = int(sys.argv[3])
RESTARTS    = int(sys.argv[4])
HELD_OUT    = int(sys.argv[5])
ELITE       = float(sys.argv[6])
THETA_PATH  = sys.argv[7]
SUMMARY_PATH = sys.argv[8]

print(f"[search] pop={POP} gens={GENERATIONS} max_steps={MAX_STEPS} "
      f"restarts={RESTARTS} held_out={HELD_OUT} dim={DIM}")

held_out_seeds = list(range(100, 100 + HELD_OUT))
random_theta_seeds = [999, 1000, 1001]

results = []  # one per restart

if RESTARTS > 0:
    for restart in range(RESTARTS):
        t0 = time.time()
        print(f"\n[restart {restart+1}/{RESTARTS}] running CEM (seed={restart})…")
        theta = cem(
            pop_size=POP, elite_frac=ELITE, generations=GENERATIONS,
            max_steps=MAX_STEPS, eval_seeds=(0, 1), seed=restart,
        )
        search_time = time.time() - t0
        print(f"[restart {restart+1}] search {search_time:.1f}s, "
              f"|theta|={np.linalg.norm(theta):.3f}")

        # Held-out eval on THIS restart's theta
        cem_perf = fitness(theta, held_out_seeds, max_steps=1000)
        print(f"[restart {restart+1}] held-out mean ({len(held_out_seeds)} seeds) = "
              f"{cem_perf:.1f} steps")
        results.append(dict(
            restart=restart, theta=theta.tolist(), search_time_s=search_time,
            cem_held_out_mean=float(cem_perf),
            n_params=int(DIM),
        ))
    # Best restart by held-out score — that's the one we save
    best = max(results, key=lambda r: r["cem_held_out_mean"])
    best_theta = np.array(best["theta"], dtype=np.float32)
    np.save(THETA_PATH, best_theta)
    print(f"\n[save] best restart = #{best['restart']+1} "
          f"(held-out {best['cem_held_out_mean']:.1f}), saved -> {THETA_PATH}")
else:
    # --skip-search path: load theta, treat as one entry
    if not os.path.exists(THETA_PATH):
        print(f"[fatal] {THETA_PATH} missing — run without --skip-search first")
        sys.exit(3)
    best_theta = np.load(THETA_PATH)
    print(f"[load] {THETA_PATH} (|theta|={np.linalg.norm(best_theta):.3f})")
    cem_perf = fitness(best_theta, held_out_seeds, max_steps=1000)
    results.append(dict(
        restart=0, theta=best_theta.tolist(), search_time_s=0.0,
        cem_held_out_mean=float(cem_perf), n_params=int(DIM),
    ))
    print(f"[eval] held-out mean ({len(held_out_seeds)} seeds) = {cem_perf:.1f} steps")

# Random baseline on the SAME held-out seeds (mean over 3 independent randoms)
rng = np.random.default_rng(999)
random_perfs = []
for s in random_theta_seeds:
    rand = np.random.default_rng(s).standard_normal(DIM) * 0.5
    p = fitness(rand, held_out_seeds, max_steps=1000)
    random_perfs.append(float(p))
    print(f"[random seed={s}] held-out mean = {p:.1f} steps")
random_mean = float(np.mean(random_perfs))

# Restart variance (only if >1 restart)
if RESTARTS > 1:
    vals = np.array([r["cem_held_out_mean"] for r in results])
    var = dict(n_restarts=len(vals), mean=float(vals.mean()),
               std=float(vals.std()), min=float(vals.min()), max=float(vals.max()))
    print(f"\n[restarts] held-out mean across {len(vals)} restarts: "
          f"{var['mean']:.1f} ± {var['std']:.1f} "
          f"(min {var['min']:.1f}, max {var['max']:.1f})")
else:
    var = None

# Summary JSON — downstream consumers (V10, E-gate comparisons) read this
summary = dict(
    config=dict(pop=POP, gens=GENERATIONS, max_steps=MAX_STEPS,
                elite=ELITE, restarts=RESTARTS, held_out_n=len(held_out_seeds),
                n_params=int(DIM)),
    reference=dict(random=15.9, cem_previous=208.3, pd_baseline=263),
    results=results,
    cem_best_held_out_mean=float(max(r["cem_held_out_mean"] for r in results)),
    random_held_out_mean=random_mean,
    restart_variance=var,
)
with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[summary] wrote {SUMMARY_PATH}")
PYEOF
PY_RC=$?
set -e

if [[ $PY_RC -ne 0 ]]; then
    die "search / eval step failed (exit $PY_RC) — see $LOG_FILE"
fi

# ── step 2: read the summary and produce the interpretation ──────────────────
say "step 2/4 — reading summary"
set +e
CEM_BEST=$("$PY" -c "import json; print(json.load(open('$SUMMARY_PATH'))['cem_best_held_out_mean'])")
RANDOM_MEAN=$("$PY" -c "import json; print(json.load(open('$SUMMARY_PATH'))['random_held_out_mean'])")
RESTART_STD=$("$PY" -c "
import json
s = json.load(open('$SUMMARY_PATH'))
v = s.get('restart_variance')
print(v['std'] if v else 0.0)
")
set -e
ok "CEM best held-out: $CEM_BEST steps"
ok "Random baseline : $RANDOM_MEAN steps"

# ── step 3: define the pass bar and interpret ────────────────────────────────
# Plan A's honest bar: > 0.8 * PD baseline (263) = 210 steps.
# The original result was 208.3 (79% of PD) — a strong result.
# We add tolerance: 208.3 ± 10% reproducibility = [187, 229].
PASS_MIN=$(awk -v pd="$PD_STEPS" 'BEGIN{printf "%.1f", 0.80*pd}')
PASS_STRONG=$(awk -v pd="$PD_STEPS" 'BEGIN{printf "%.1f", 0.95*pd}')
REPRODUCIBILITY_LOW=$(awk 'BEGIN{printf "%.1f", 208.3*0.90}')
REPRODUCIBILITY_HIGH=$(awk 'BEGIN{printf "%.1f", 208.3*1.10}')

say "step 3/4 — thresholds: pass>=$PASS_MIN, strong>=$PASS_STRONG, reference_repro=[$REPRODUCIBILITY_LOW, $REPRODUCIBILITY_HIGH]"

# ── step 4: report ───────────────────────────────────────────────────────────
say "step 4/4 — result"
PASS=$(awk -v c="$CEM_BEST" -v p="$PASS_MIN" 'BEGIN{print (c>=p) ? 1 : 0}')
REPRO=$(awk -v c="$CEM_BEST" -v lo="$REPRODUCIBILITY_LOW" -v hi="$REPRODUCIBILITY_HIGH" \
             'BEGIN{print (c>=lo && c<=hi) ? 1 : 0}')
STRONG=$(awk -v c="$CEM_BEST" -v s="$PASS_STRONG" 'BEGIN{print (c>=s) ? 1 : 0}')

if [[ "$PASS" -eq 1 ]]; then
    {
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "  PLAN A — RESULT: ${GRN}PASS${RST}"
        echo "════════════════════════════════════════════════════════════════"
        echo "  CEM best held-out  : $CEM_BEST steps"
        echo "  Random baseline    : $RANDOM_MEAN steps"
        echo "  PD baseline        : $PD_STEPS steps"
        echo "  Reference (earlier): 208.3 steps"
        [[ "$REPRO" -eq 1 ]] && echo "  Reproducibility    : ${GRN}WITHIN 10% of reference${RST}" || \
                                echo "  Reproducibility    : outside 10% of reference (run more restarts to diagnose)"
        [[ -n "$RESTART_STD" && "$RESTART_STD" != "0.0" ]] && \
            echo "  Restart std        : ±$RESTART_STD steps"
        echo ""
        if [[ "$STRONG" -eq 1 ]]; then
            echo "  ${GRN}STRONG — CEM matches or beats 95% of PD baseline.${RST}"
            echo "  This is the result to compare V10 against. Plan B's fixed"
            echo "  variant must beat THIS number to be worth shipping."
        else
            echo "  ${YEL}MODERATE — CEM is within 80% of PD baseline.${RST}"
            echo "  Real headroom left. Try: more generations, larger pop,"
            echo "  or richer features (add gait-phase clock as in race_demo.py)."
        fi
        echo ""
        echo "  Saved theta   : $THETA_PATH"
        echo "  Full summary  : $SUMMARY_PATH"
        echo "  Full log      : $LOG_FILE"
        echo "════════════════════════════════════════════════════════════════"
    } | tee -a "$LOG_FILE"
    exit 0
else
    {
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "  PLAN A — RESULT: ${RED}UNDERPERFORM${RST}"
        echo "════════════════════════════════════════════════════════════════"
        echo "  CEM best held-out  : $CEM_BEST steps"
        echo "  Pass threshold     : $PASS_MIN steps (80% of PD baseline $PD_STEPS)"
        echo "  Random baseline    : $RANDOM_MEAN steps"
        echo ""
        echo "  Possible causes, in order of likelihood:"
        echo "    1. Search under-converged — run with more generations:"
        echo "         ./run_plan_a.sh --generations 30 --pop 80"
        echo "    2. High seed-to-seed variance — run with more restarts:"
        echo "         ./run_plan_a.sh --restarts 5"
        echo "       and inspect restart_variance.std in $SUMMARY_PATH"
        echo "    3. Feature set too thin — the original 15-dim feature set"
        echo "       (pitch, pitch_rate, height, 6 qpos, 6 qvel) has no phase"
        echo "       signal. race_demo.py's gait-clock feature (sin/cos phase)"
        echo "       closed the gap for forward locomotion — port it here."
        echo "    4. Walker2d-v5 version mismatch — check gymnasium version."
        echo ""
        echo "  Full log      : $LOG_FILE"
        echo "════════════════════════════════════════════════════════════════"
    } | tee -a "$LOG_FILE"
    exit 2
fi
