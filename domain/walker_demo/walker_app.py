"""Analogous to full_flow_animation/app.py. Same schema, same route shape,
two policies (balance / walk) plus a push test."""
from __future__ import annotations
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_DOMAIN = os.path.dirname(_HERE)
_REPO = os.path.dirname(_DOMAIN)
sys.path.insert(0, _HERE)
sys.path.insert(0, _DOMAIN)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "mw_to_protomotions"))  # walker_domain.py lives
# here, and the checkpoint was saved expecting it importable as a top-level
# module (not mw_to_protomotions.walker_domain) - the real bug behind the
# "silent" MW failure: MWDetector.load() raised ModuleNotFoundError for
# 'walker_domain' because this path was missing, not a schema mismatch.

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from walker_animation import run_policy_rollout
from mw_framework import MWDetector

app = Flask(__name__,
            static_folder=os.path.join(_HERE, "static"),
            static_url_path="")

POLICIES = {}
for name, relpath in [("balance", "cem_best_theta.npy"),
                      ("walk", "cem_walk_theta.npy")]:
    path = os.path.join(_REPO, relpath)
    if os.path.exists(path):
        POLICIES[name] = np.load(path)
        print(f"loaded policy '{name}' from {relpath}")

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
        if not (val == val):  # NaN check
            raise ValueError("mw_distance returned NaN on sample")
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
    print("[MW] WARNING: no usable detector - mw_distances will be empty")
    print("[MW] train one with: ./run_walker_demo.sh --train-detector")

if not POLICIES:
    raise SystemExit("No policies found. Run ./run_demo_walker.sh first.")

DOMAIN_EQ = [
    "joint limits: thigh/leg ∈ [-2.618, 0], foot ∈ [-0.785, 0.785]",
    "reachability: |Δqpos| per step bounded by physics",
    "MW distance: deviation from the trained standing/walking manifold",
]


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/walker/policies")
def list_policies():
    return jsonify({
        "available": list(POLICIES.keys()),
        "mw_active": DET is not None,
    })


@app.route("/api/walker/run", methods=["POST"])
def run():
    body = request.get_json(silent=True) or {}
    policy = body.get("policy", "balance")
    theta = POLICIES.get(policy)
    if theta is None:
        return jsonify({"ok": False, "error": f"no policy '{policy}'"}), 404

    result = run_policy_rollout(
        theta,
        max_steps=int(body.get("max_steps", 1000)),
        push_at=int(body.get("push_at", -1)),
        push=float(body.get("push", 0.0)),
        det=DET,
        seed=int(body.get("seed", 0)),
    )
    return jsonify({
        **result,
        "policy": policy,
        "domain": "walker2d",
        "equations": DOMAIN_EQ,
    })


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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5051, debug=False)
