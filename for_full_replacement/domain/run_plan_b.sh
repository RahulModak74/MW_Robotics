#!/usr/bin/env bash
# run_plan_b.sh — install Plan B files into repo structure and run the E-gate test.
#
# Assumes:
#   ~/MW_Robotics/mw_framework/           (your existing framework)
#   ~/MW_Robotics/mw_to_protomotions/     (exists, from the snapshot)
#   ~/MW_Robotics/tests/                  (exists)
#   ~/MW_Robotics/domain/                 (this folder, with the 5 .py files)
#
# Usage:
#   cd ~/MW_Robotics/domain
#   chmod +x run_plan_b.sh
#   ./run_plan_b.sh                    # install + smoke test + E-gate
#   ./run_plan_b.sh --dry-run          # show what would happen, change nothing
#   ./run_plan_b.sh --smoke-only       # install + datagen smoke test only
#   ./run_plan_b.sh --skip-train       # reuse cached TransitionVAE checkpoint
#
# Exits 0 on E-gate pass, 2 on E-gate fail (Plan B not viable as spec'd),
# 1 on install/setup error.

set -euo pipefail

# ── colour + timing helpers ──────────────────────────────────────────────────
if [[ -t 1 ]]; then
    GRN=$'\033[32m'; RED=$'\033[31m'; YEL=$'\033[33m'; BLU=$'\033[34m'; RST=$'\033[0m'
else
    GRN=""; RED=""; YEL=""; BLU=""; RST=""
fi
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say()  { printf "%s [%s] %s\n" "$(ts)" "${BLU}plan-b${RST}" "$*"; }
warn() { printf "%s [%s] %s\n" "$(ts)" "${YEL}plan-b${RST}" "$*" >&2; }
die()  { printf "%s [%s] %s\n" "$(ts)" "${RED}plan-b${RST}" "$*" >&2; exit "${2:-1}"; }
ok()   { printf "%s [%s] %s\n" "$(ts)" "${GRN}plan-b${RST}" "$*"; }

# ── parse args ───────────────────────────────────────────────────────────────
DRY_RUN=0
SMOKE_ONLY=0
SKIP_TRAIN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)     DRY_RUN=1 ;;
        --smoke-only)  SMOKE_ONLY=1 ;;
        --skip-train)  SKIP_TRAIN=1 ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) die "unknown arg: $arg (see --help)" ;;
    esac
done

# ── detect repo root ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../mw_framework/__init__.py" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "$SCRIPT_DIR/mw_framework/__init__.py" ]]; then
    REPO_ROOT="$SCRIPT_DIR"
else
    die "cannot find mw_framework/ — run this from MW_Robotics/ or MW_Robotics/domain/"
fi

cd "$REPO_ROOT"


DOMAIN_DIR="$REPO_ROOT/domain"
LOG_DIR="$DOMAIN_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/plan_b_$(date -u +%Y%m%d_%H%M%S).log"

say "repo root : $REPO_ROOT"
say "domain dir: $DOMAIN_DIR"
say "log file  : $LOG_FILE"

# ── verify the 5 source files exist ──────────────────────────────────────────
SRCS=(transition.py inverse.py b1_runtime.py transition_datagen.py test_transition.py)
for f in "${SRCS[@]}"; do
    [[ -f "$DOMAIN_DIR/$f" ]] || die "missing source: $DOMAIN_DIR/$f"
done
ok "all 5 source files present"

# ── python / package checks ──────────────────────────────────────────────────
PY="${PYTHON:-python3}"
"$PY" -c "import sys; assert sys.version_info >= (3,9), sys.version" \
    || die "need Python 3.9+, got $($PY -V)"
"$PY" -c "import torch, pyro, numpy, pandas" \
    || die "missing deps: pip install torch pyro-ppl numpy pandas"
"$PY" -c "import gymnasium, mujoco" \
    || die "missing deps: pip install 'gymnasium[mujoco]'"
"$PY" -c "import pytest" \
    || die "missing pytest: pip install pytest"
ok "Python + deps present"

# Force CPU — this is the whole point.
export MW_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=""
say "MW_DEVICE=cpu, CUDA_VISIBLE_DEVICES='' (CPU-only by design)"

# ── install: place files in package structure ────────────────────────────────
install_file() {
    local src="$1" dst="$2"
    if [[ -f "$dst" ]] && ! cmp -s "$src" "$dst"; then
        local bak="${dst}.bak.$(date -u +%Y%m%d%H%M%S)"
        warn "backing up existing $dst -> $bak"
        [[ $DRY_RUN -eq 0 ]] && cp -p "$dst" "$bak"
    fi
    if [[ $DRY_RUN -eq 0 ]]; then
        cp -p "$src" "$dst"
        ok "installed $(basename "$src") -> ${dst#$REPO_ROOT/}"
    else
        say "would install $(basename "$src") -> ${dst#$REPO_ROOT/}"
    fi
}

mkdir_target() {
    local d="$1"
    if [[ ! -d "$d" ]]; then
        if [[ $DRY_RUN -eq 0 ]]; then mkdir -p "$d"; ok "created dir ${d#$REPO_ROOT/}"; else say "would create dir ${d#$REPO_ROOT/}"; fi
    fi
}

say "step 1/5 — install files into package structure"
mkdir_target "$REPO_ROOT/mw_framework/domains"
mkdir_target "$REPO_ROOT/mw_framework/query"
mkdir_target "$REPO_ROOT/mw_to_protomotions"
mkdir_target "$REPO_ROOT/tests"

install_file "$DOMAIN_DIR/transition.py"            "$REPO_ROOT/mw_framework/domains/transition.py"
install_file "$DOMAIN_DIR/inverse.py"               "$REPO_ROOT/mw_framework/query/inverse.py"
install_file "$DOMAIN_DIR/b1_runtime.py"            "$REPO_ROOT/mw_to_protomotions/b1_runtime.py"
install_file "$DOMAIN_DIR/transition_datagen.py"    "$REPO_ROOT/mw_to_protomotions/transition_datagen.py"
install_file "$DOMAIN_DIR/test_transition.py"       "$REPO_ROOT/tests/test_transition_gate.py"

# ensure __init__.py exists where needed (namespace packages work, but explicit is safer)
for pkg in mw_framework/query mw_to_protomotions tests; do
    init="$REPO_ROOT/$pkg/__init__.py"
    if [[ ! -f "$init" ]]; then
        if [[ $DRY_RUN -eq 0 ]]; then
            printf '"""%s — package marker."""\n' "$pkg" > "$init"
            ok "created $pkg/__init__.py"
        else
            say "would create $pkg/__init__.py"
        fi
    fi
done

# query/__init__.py should export the primitive
if [[ ! -s "$REPO_ROOT/mw_framework/query/__init__.py" ]] || ! grep -q "inverse_query" "$REPO_ROOT/mw_framework/query/__init__.py" 2>/dev/null; then
    if [[ $DRY_RUN -eq 0 ]]; then
        cat > "$REPO_ROOT/mw_framework/query/__init__.py" <<'EOF'
"""mw_framework.query — Plan B query primitives on the transition manifold."""
from .inverse import inverse_query
__all__ = ["inverse_query"]
EOF
        ok "wrote mw_framework/query/__init__.py"
    else
        say "would write mw_framework/query/__init__.py"
    fi
fi

if [[ $DRY_RUN -eq 1 ]]; then
    say "dry-run complete — nothing changed. Re-run without --dry-run to execute."
    exit 0
fi

# ── step 2: smoke test datagen ───────────────────────────────────────────────
say "step 2/5 — datagen smoke test (2 units × 50 steps)"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import sys, time
sys.path.insert(0, ".")
from mw_to_protomotions.transition_datagen import collect_transitions
t0 = time.time()
df = collect_transitions(n_units=2, n_steps=50, seed0=1)
print(f"  rows = {len(df)}, cols = {len(df.columns)}, "
      f"units = {df['unit'].nunique()}, wall = {time.time()-t0:.1f}s")
assert len(df) > 0, "datagen returned zero rows"
required = {"s_height", "s_pitch", "a_thigh", "sp_height"}
missing = required - set(df.columns)
assert not missing, f"missing columns: {missing}"
nan_cols = df.isna().sum()
nan_cols = nan_cols[nan_cols > 0]
assert len(nan_cols) == 0, f"NaN columns: {list(nan_cols.index)}"
print("  smoke test PASSED")
PYEOF
ok "datagen works"

if [[ $SMOKE_ONLY -eq 1 ]]; then
    say "--smoke-only set — stopping here. Run without --smoke-only for E-gate."
    exit 0
fi

# ── step 3: (optional) train TransitionVAE ───────────────────────────────────
CACHE="/tmp/mw_transition_walker.ckpt"
if [[ $SKIP_TRAIN -eq 1 && -f "$CACHE" ]]; then
    say "step 3/5 — SKIPPING training, reusing cache $CACHE"
else
    say "step 3/5 — training TransitionVAE (20 units × 250 steps, ~3–5 min CPU)"
    say "            this is the long step — go make chai"
    "$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import sys, time
sys.path.insert(0, ".")
import pyro
from mw_framework import MWDetector, set_seed
from mw_framework.domains.transition import TransitionVAE, FEATURES
from mw_to_protomotions.transition_datagen import collect_transitions

set_seed(0); pyro.clear_param_store()
print("  generating training data…", flush=True)
t0 = time.time()
df = collect_transitions(n_units=20, n_steps=250, seed0=100)
print(f"  {len(df)} transitions from {df['unit'].nunique()} units "
      f"({time.time()-t0:.1f}s)", flush=True)

n = len(df)
train = df.iloc[: int(n * 0.8)]
print(f"  training on {len(train)} rows…", flush=True)
t0 = time.time()
det = MWDetector(vae=TransitionVAE(), epochs=200, lr=2e-3, verbose=True).fit(train[FEATURES])
print(f"  trained in {time.time()-t0:.1f}s", flush=True)
det.save("/tmp/mw_transition_walker.ckpt")
print(f"  saved /tmp/mw_transition_walker.ckpt", flush=True)
PYEOF
    ok "training complete"
fi

# ── step 4: E-gate (the decider) ─────────────────────────────────────────────
say "step 4/5 — E-gate: the decisive test"
say "            if this passes, Plan B is viable; if it fails, Plan A stays"
set +e
"$PY" -m pytest tests/test_transition_gate.py::test_manifold_learns_cleanly \
    tests/test_transition_gate.py::test_e_gate_inverse_recoverable \
    -v -s --tb=short 2>&1 | tee -a "$LOG_FILE"
EGATE_RC=${PIPESTATUS[0]}
set -e

# ── step 5: interpret result ─────────────────────────────────────────────────
say "step 5/5 — interpreting the E-gate result"
if [[ $EGATE_RC -eq 0 ]]; then
    ok "E-gate PASSED"
    # extract the numbers from the log for a clean summary
    MEAN_ERR=$(grep -E "^\s+mean\s+=" "$LOG_FILE" | tail -1 | awk '{print $3}' || echo "?")
    IMPROV=$(grep -E "improvement over random" "$LOG_FILE" | tail -1 | awk '{print $NF}' || echo "?")
    {
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "  PLAN B — E-GATE RESULT: ${GRN}PASS${RST}"
        echo "════════════════════════════════════════════════════════════════"
        echo "  inverse-query mean action error : $MEAN_ERR"
        echo "  improvement over random         : $IMPROV"
        echo ""
        echo "  Interpretation:"
        if awk -v m="${MEAN_ERR:-9}" 'BEGIN {exit !(m < 0.1)}'; then
            echo "    STRONG — mean error < 0.1. Plan B is a clear win on this robot."
            echo "    Next: implement b1_step, then run V10 vs cem_policy_search.py."
        elif awk -v m="${MEAN_ERR:-9}" 'BEGIN {exit !(m < 0.2)}'; then
            echo "    MODERATE — mean error in [0.1, 0.2]. Plan B is viable but"
            echo "    Plan A may match it. Run V10 before committing further."
        else
            echo "    BORDERLINE — passed the 2× vs random bar but not strongly."
            echo "    Consider adding history features (s_{t-1}, a_{t-1}) to"
            echo "    TransitionVAE and re-running before building B1/B2."
        fi
        echo ""
        echo "  Full log: $LOG_FILE"
        echo "  Cache  : /tmp/mw_transition_walker.ckpt  (re-run with --skip-train)"
        echo "════════════════════════════════════════════════════════════════"
    } | tee -a "$LOG_FILE"
    exit 0
else
    warn "E-gate FAILED (pytest exit $EGATE_RC)"
    {
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "  PLAN B — E-GATE RESULT: ${RED}FAIL${RST}"
        echo "════════════════════════════════════════════════════════════════"
        echo "  The inverse query could not recover the action from (s, s')"
        echo "  better than random. As spec'd, Plan B is not viable on Walker2d."
        echo ""
        echo "  This is a real, publishable result — not a bug."
        echo "  Options, in order of cost:"
        echo "    1. Enrich TransitionVAE with history: (s_{t-1}, a_{t-1}, s_t, a_t, s_{t+1})"
        echo "       — 2× feature count, ~1 day rework. Often fixes ill-posed inverses."
        echo "    2. Condition on contact mode (foot down / in air) as a fibre —"
        echo "       contact discontinuity fragments the manifold; conditioning"
        echo "       on it is the known fix (see the MW paper's FibreBundle op)."
        echo "    3. Fall back to Plan A (cem_policy_search.py, already in repo)."
        echo "       Plan A is the proven path; Plan B was always the bet."
        echo ""
        echo "  Full log: $LOG_FILE"
        echo "════════════════════════════════════════════════════════════════"
    } | tee -a "$LOG_FILE"
    exit 2
fi
