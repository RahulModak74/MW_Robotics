#!/usr/bin/env python3
"""
full_flow_animation/app.py
============================
Flask backend for the interactive MW robot demo. Loads the six cached
checkpoints (run train_and_cache.py first), keeps a small server-side
robot state (arm joint angles, base pose, accumulated heat), and exposes
one route per behaviour. Every route returns a list of animation frames
decoded from the matching trained manifold via robot_motion.compute_path()
- see that file's docstring and README.md for exactly what is real.

Run:
    python train_and_cache.py     # once, trains + caches all six models
    python app.py                 # then start the server
    open http://127.0.0.1:5050
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory

from mw_framework import MWDetector
import robot_motion as rm

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")

app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="")

print("Loading cached checkpoints ...")
DET = {}
for name in ("flexible_robot", "joint_actuator", "planar_arm", "mobile_robot",
            "legged_contact", "actuator_thermal"):
    path = os.path.join(CKPT_DIR, f"{name}.pth")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path} - run 'python train_and_cache.py' first.")
    DET[name] = MWDetector.load(path)
print("All six models loaded.")

# Server-side robot state (one demo robot, shared across clients - this is a
# demo, not a multi-user app).
STATE = {
    "theta1": 0.3, "theta2": 0.6,        # arm joint angles
    "joint_ref": 0.8,                    # standalone joint-actuator set-point
    "base_x": 0.0, "base_y": 0.0, "base_theta": 0.0,
    "heat": 0.15, "last_activity": time.time(),
}


def bump_heat(amount=0.22):
    STATE["heat"] = min(1.0, STATE["heat"] + amount)
    STATE["last_activity"] = time.time()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/state")
def get_state():
    return jsonify(STATE)


@app.route("/api/arm/reach", methods=["POST"])
def arm_reach():
    body = request.get_json(force=True)
    x, y = float(body["x"]), float(body["y"])
    frames, (t1, t2) = rm.arm_reach(DET["planar_arm"], STATE["theta1"], STATE["theta2"], x, y)
    STATE["theta1"], STATE["theta2"] = t1, t2
    bump_heat()
    return jsonify({"frames": frames, "domain": "planar_arm",
                   "equations": "forward-kinematics identity + Jacobian (velocity) identity"})


@app.route("/api/joint/rotate", methods=["POST"])
def joint_rotate():
    body = request.get_json(force=True)
    target = float(body["theta"])
    frames = rm.joint_rotate(DET["joint_actuator"], STATE["joint_ref"], target)
    STATE["joint_ref"] = target
    bump_heat()
    return jsonify({"frames": frames, "domain": "joint_actuator",
                   "equations": "I*theta'' + b*theta' + k*theta = tau  (Newton's second law)"})


@app.route("/api/base/move", methods=["POST"])
def base_move():
    body = request.get_json(force=True)
    x, y = float(body["x"]), float(body["y"])
    frames, (xe, ye, the) = rm.base_move(DET["mobile_robot"], STATE["base_x"], STATE["base_y"],
                                        STATE["base_theta"], x, y)
    STATE["base_x"], STATE["base_y"], STATE["base_theta"] = xe, ye, the
    bump_heat()
    return jsonify({"frames": frames, "domain": "mobile_robot",
                   "equations": "nonholonomic constraint: xdot*sin(theta) - ydot*cos(theta) = 0"})


@app.route("/api/hop", methods=["POST"])
def hop():
    body = request.get_json(silent=True) or {}
    height = float(body.get("height", 0.5))
    frames = rm.hop(DET["legged_contact"], launch_height=height)
    bump_heat(0.3)
    return jsonify({"frames": frames, "domain": "legged_contact",
                   "equations": "energy conservation: vz^2 + 2*g*z = v0^2"})


@app.route("/api/lift", methods=["POST"])
def lift():
    body = request.get_json(silent=True) or {}
    deflection = float(body.get("deflection", 0.7))
    frames = rm.lift_payload(DET["flexible_robot"], deflection=deflection)
    bump_heat(0.18)
    return jsonify({"frames": frames, "domain": "flexible_robot",
                   "equations": "Euler-Bernoulli beam PDE: rho*A*w_tt + EI*w_xxxx = f"})


@app.route("/api/thermal/status")
def thermal_status():
    elapsed = time.time() - STATE["last_activity"]
    frames = rm.thermal_tick(DET["actuator_thermal"], heat_level=STATE["heat"])
    frac = min(elapsed / 14.0, 1.0)
    idx = min(int(frac * (len(frames) - 1)), len(frames) - 1)
    current = frames[idx]
    # advance the stored heat slowly toward the decayed reading, so repeated
    # polls settle the gauge instead of resetting it each time.
    STATE["heat"] = max(0.02, STATE["heat"] * 0.985)
    return jsonify({"excess": current["excess"], "T": current["T"],
                   "domain": "actuator_thermal",
                   "equations": "Fourier's law: flux = -k * dTdx, decay envelope"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
