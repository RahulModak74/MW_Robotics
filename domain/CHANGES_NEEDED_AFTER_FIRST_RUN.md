The walk policy works (1.33 m forward!) — the only bug is the MW gauge reading —
Two things to note before the fix:

Your walk policy is working. 1.33 m forward in 600 steps on Walker2d is a real walking gait at this policy size — the reward-hacking fix (forward distance + 0.02 × survival) is doing exactly what it was supposed to do. That's the promotion demo.

The push test is also working. Fell at step 172 with push at step 100 → the robot survived 72 steps after the kick, then fell. That's the correct behavior for a strong push.

The — for MW distance is a silent failure. Not a physics problem — a code problem. The walker_animation.py file has a try/except: pass that swallows the real error, so the frontend gets an empty list and prints —.

The likely cause is that walker_mw_detector.ckpt either doesn't exist anymore, or was trained with an older schema that no longer matches the current walker_domain.FEATURES. Both fixable. Here are the concrete additions.

Add 1 — domain/walker_demo/walker_animation.py (stop swallowing the error)
Find this block:

python
    mw_distances = []
    if det is not None and posture_rows:
        from walker_domain import FEATURES
        try:
            d = det.mw_distance(pd.DataFrame(posture_rows)[FEATURES])
            mw_distances = [float(v) for v in d]
        except Exception:
            mw_distances = []
Replace with:

python
    mw_distances = []
    mw_error = None
    if det is None:
        mw_error = "detector_not_loaded"
    elif posture_rows:
        from walker_domain import FEATURES
        try:
            df_posture = pd.DataFrame(posture_rows)[FEATURES]
            d = det.mw_distance(df_posture)
            mw_distances = [float(v) for v in d]
        except Exception as e:
            import traceback
            mw_error = f"{type(e).__name__}: {e}"
            print(f"[MW] mw_distance failed: {mw_error}", flush=True)
            traceback.print_exc()
And in the return dict, add "mw_error": mw_error:

python
    return {
        "qpos": qpos_log,
        "qvel": qvel_log,
        "mw_distances": mw_distances,
        "mw_error": mw_error,                     # ← ADD THIS LINE
        "steps": steps,
        "forward_distance": float(data.qpos[0] - x0),
        "fell_at": fell_at,
        "seed": seed,
        "push_at": push_at,
        "push": push,
    }
Why this matters. The moment you add this, the Flask console will print the actual exception. It's almost certainly going to be RuntimeError: input is missing columns: [...] or FileNotFoundError or a pickle error — the message will tell you exactly which.

Add 2 — domain/walker_demo/walker_app.py (validate the detector at startup)
Find this block:

python
DET = None
det_candidates = [
    os.path.join(_REPO, "mw_to_protomotions", "walker_mw_detector.ckpt"),
    os.path.join(_REPO, "walker_mw_detector.ckpt"),
]
for p in det_candidates:
    if os.path.exists(p):
        try:
            DET = MWDetector.load(p)
            print(f"loaded MW detector from {p}")
            break
        except Exception as e:
            print(f"could not load {p}: {e}")
Replace with a version that validates the loaded detector, not just loads it:

python
DET = None
DET_PATH = None
det_candidates = [
    os.path.join(_REPO, "mw_to_protomotions", "walker_mw_detector.ckpt"),
    os.path.join(_REPO, "walker_mw_detector.ckpt"),
]
for p in det_candidates:
    if not os.path.exists(p):
        continue
    try:
        candidate = MWDetector.load(p)
        # ── validate on a known-good posture ─────────────────────────────
        import pandas as pd
        from walker_domain import FEATURES as WFEATURES
        sample = pd.DataFrame([{
            "height": 1.25, "pitch": 0.05, "pitch_rate": 0.0,
            "thigh": -0.5, "leg": -0.5, "foot": 0.0,
            "thigh_left": -0.5, "leg_left": -0.5, "foot_left": 0.0,
        }])
        d = candidate.mw_distance(sample[WFEATURES])
        val = float(d[0])
        if not (val == val):    # NaN check
            raise ValueError(f"mw_distance returned NaN on sample")
        DET = candidate
        DET_PATH = p
        print(f"loaded MW detector from {p}, sample distance = {val:.3f}")
        break
    except Exception as e:
        import traceback
        print(f"could not load/validate {p}: {type(e).__name__}: {e}")
        traceback.print_exc()
        continue

if DET is None:
    print("[MW] WARNING: no usable detector — mw_distances will be empty")
    print("[MW] train one with: ./run_demo_walker.sh --train-detector")
Add a diagnostics endpoint right before if __name__ == "__main__"::

python
@app.route("/api/walker/diagnostics")
def diagnostics():
    return jsonify({
        "policies_loaded": list(POLICIES.keys()),
        "detector_loaded": DET is not None,
        "detector_path": DET_PATH,
        "detector_feature_names": getattr(DET, "feature_names_", None) if DET else None,
        "detector_calibration": {
            "threshold": DET.calibration_.threshold,
            "sigma": DET.calibration_.sigma,
        } if DET else None,
    })
Now curl http://127.0.0.1:5051/api/walker/diagnostics tells you exactly what the state is.

Add 3 — domain/run_walker_demo.sh (auto-train the detector if missing)
Find the block:

bash
# ── check deps
PY="${PYTHON:-python3}"
"$PY" -c "import flask, gymnasium, numpy, pandas, torch, pyro" \
    || die "missing deps: pip install flask 'gymnasium[mujoco]' numpy pandas torch pyro-ppl"
export MW_DEVICE=cpu CUDA_VISIBLE_DEVICES=""
ok "Python + deps present"
Replace with (adds a detector check + auto-train):

bash
# ── check deps
PY="${PYTHON:-python3}"
"$PY" -c "import flask, gymnasium, numpy, pandas, torch, pyro" \
    || die "missing deps: pip install flask 'gymnasium[mujoco]' numpy pandas torch pyro-ppl"
export MW_DEVICE=cpu CUDA_VISIBLE_DEVICES=""
ok "Python + deps present"

# ── ensure MW detector exists ────────────────────────────────────────────────
DET_CKPT="$REPO/mw_to_protomotions/walker_mw_detector.ckpt"
TRAIN_DET=0
[[ "${1:-}" == "--train-detector" ]] && TRAIN_DET=1
if [[ ! -f "$DET_CKPT" || $TRAIN_DET -eq 1 ]]; then
    say "MW detector missing (or --train-detector set) — training now (~2 min)"
    "$PY" - <<'PYEOF' 2>&1 | tee -a "$REPO/domain/logs/detector_train_$(date -u +%Y%m%d_%H%M%S).log"
import sys, os
sys.path.insert(0, os.getcwd())
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
det.save("mw_to_protomotions/walker_mw_detector.ckpt")
print("  saved mw_to_protomotions/walker_mw_detector.ckpt")
PYEOF
    ok "detector trained"
fi
Also accept --train-detector as an arg at the top. Insert after INSTALL_ONLY=0:

bash
TRAIN_DET=0
if [[ "${1:-}" == "--train-detector" ]]; then
    TRAIN_DET=1
    shift
fi
Add 4 — domain/walker_demo/static/index.html (show the error, don't just show —)
Find the renderFrame function, and replace the MW/status update block:

javascript
  const mw = current.mw_distances[idx];
  els.step.textContent = `${idx + 1} / ${current.qpos.length}`;
  els.fwd.textContent = (current.qpos[idx][0] - current.qpos[0][0]).toFixed(2);
  els.mw.textContent = mw != null ? mw.toFixed(2) : '—';
  const status = mw == null ? '—' : (mw > 6.0 ? 'THREAT' : mw > 3.0 ? 'WATCH' : 'OK');
  els.status.textContent = status;
  els.status.className = 'value ' + (status === 'THREAT' ? 'status-bad' : status === 'WATCH' ? 'status-warn' : 'status-ok');
with:

javascript
  const mw = current.mw_distances[idx];
  els.step.textContent = `${idx + 1} / ${current.qpos.length}`;
  els.fwd.textContent = (current.qpos[idx][0] - current.qpos[0][0]).toFixed(2);

  if (current.mw_error) {
    els.mw.textContent = 'n/a';
    els.mw.title = current.mw_error;
    els.status.textContent = 'MW OFF';
    els.status.className = 'value status-warn';
  } else if (mw != null) {
    els.mw.textContent = mw.toFixed(2);
    els.mw.title = '';
    const status = mw > 6.0 ? 'THREAT' : mw > 3.0 ? 'WATCH' : 'OK';
    els.status.textContent = status;
    els.status.className = 'value ' +
      (status === 'THREAT' ? 'status-bad' :
       status === 'WATCH' ? 'status-warn' : 'status-ok');
  } else {
    els.mw.textContent = '—';
    els.status.textContent = '—';
    els.status.className = 'value';
  }
Also add a small banner when the run starts:

javascript
  // in runPolicy(), just after the fetch:
  if (current.mw_error) {
    els.eq.innerHTML = `<div class="eq" style="color:#f87171">MW unavailable: ${current.mw_error}</div>` +
                       (current.equations || []).map(e => `<div class="eq">· ${e}</div>`).join('');
  } else {
    els.eq.innerHTML = (current.equations || []).map(e => `<div class="eq">· ${e}</div>`).join('');
  }
After applying all four additions, run in order
bash
cd ~/MW_Robotics/domain

# 1. Confirm what's actually on disk
ls -la ~/MW_Robotics/mw_to_protomotions/walker_mw_detector.ckpt 2>&1

# 2. Start the server. If the detector is missing or invalid, it auto-trains
#    (or use --train-detector to force a retrain with the current schema)
./run_walker_demo.sh --train-detector

# 3. In a second terminal, check the diagnostics
curl -s http://127.0.0.1:5051/api/walker/diagnostics | python3 -m json.tool

# 4. Reload the page, click "walk" — MW gauge should now populate
The diagnostics output will tell you definitively what was wrong. Expected healthy output:

json
{
  "policies_loaded": ["balance", "walk"],
  "detector_loaded": true,
  "detector_path": "/home/rahul/MW_Robotics/mw_to_protomotions/walker_mw_detector.ckpt",
  "detector_feature_names": ["height", "pitch", "pitch_rate", "thigh", ...],
  "detector_calibration": {"threshold": 4.231, "sigma": 1.054}
}
If detector_loaded: false or detector_path: null, the console (from Addition 2) will have printed the specific traceback. That's the answer to "why is the MW gauge empty."

While you're there — one observation about the balance policy
The balance policy showing -0.75 m forward means it drifts slightly backward while standing. That's not a bug — a policy that learned "survive 920 steps" has no incentive to stay stationary, only to not fall. The subtle backward lean is probably a stable equilibrium.

If you want it to show 0.00 m, you'd need either:

A "balance" fitness with a small penalty on forward drift, or

A slightly longer training run so CEM finds a truly stationary equilibrium (unlikely — CEM has no reason to prefer stationary)

For the promotion narrative, -0.75 m is fine: "the balance policy holds for 920 steps with minor drift" is honest. The walk policy at +1.33 m is the one to feature.

The honest status update for the roadmap
Artifact	Status	Note
run_plan_a.sh	✅ Working	920.9 steps (stand-still policy, correct for survival fitness)
run_demo_walker.sh	✅ Working	1.33 m forward — real gait
run_walker_demo.sh	✅ Working (Flask)	Serving both policies
Demo frontend	✅ Working	qpos animates, forward distance live
MW gauge	⚠️ Broken	Silent failure — diagnose + fix with the 4 additions above
ONNX export	❌ Not yet	Roadmap Gap 5
Retargeting (GMR)	❌ Not yet	Roadmap Gap 1
mjbatch parallelism	❌ Not yet	Roadmap Gap 2
Two of six gaps closed. The MW-gauge fix is a 20-minute patch, not a research problem. Once it works, you have the full demo: physics running, policy walking, manifold validating in real time. That's the video to record.

The remaining four gaps are the actual roadmap to ProtoMotions parity. The MW gauge fix is just the thing you need to show that the manifold is already part of the story — not just a research artifact, but a live component of the demo.
