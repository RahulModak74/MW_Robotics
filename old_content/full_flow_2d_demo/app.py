#!/usr/bin/env python3
"""
full_flow_animation/app.py
============================
Flask backend for the interactive MW robot demo. Loads the six cached
checkpoints (run train_and_cache.py first), keeps server-side state for
TWO named robots ("a", "b"), and exposes one route per behaviour plus a
choreography endpoint that, if a sibling mw_agentic/ package is present
(not part of this repo), drives both robots from its agentic layer - a
recipe first, an LLM only when needed. Without it, /api/choreograph
returns a clear 501 and every per-behaviour manual route still works
unchanged. Every per-behaviour route returns a list of animation frames
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

try:
    from mw_agentic.orchestrate import choreograph as mw_choreograph
except ImportError:
    mw_choreograph = None

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


def _default_state():
    return {"theta1": 0.3, "theta2": 0.6, "joint_ref": 0.8,
            "base_x": 0.0, "base_y": 0.0, "base_theta": 0.0,
            "heat": 0.15, "last_activity": time.time()}


# Two named robots ("a", "b"). The single-robot demo (index.html before the
# choreography feature) always used robot "a" implicitly - every route below
# still defaults to "a" so that page keeps working unchanged.
STATES = {"a": _default_state(), "b": _default_state()}
STATES["b"]["base_x"] = 1.5     # start the two robots apart, not stacked


def bump_heat(robot: str, amount: float = 0.22) -> None:
    s = STATES[robot]
    s["heat"] = min(1.0, s["heat"] + amount)
    s["last_activity"] = time.time()


DOMAIN_EQ = {
    "planar_arm": "forward-kinematics identity + Jacobian (velocity) identity",
    "joint_actuator": "I*theta'' + b*theta' + k*theta = tau  (Newton's second law)",
    "mobile_robot": "nonholonomic constraint: xdot*sin(theta) - ydot*cos(theta) = 0",
    "legged_contact": "energy conservation: vz^2 + 2*g*z = v0^2",
    "flexible_robot": "Euler-Bernoulli beam PDE: rho*A*w_tt + EI*w_xxxx = f",
}


def _run_tool(robot: str, tool: str, params: dict) -> dict:
    """Shared by the per-behaviour routes AND /api/choreograph: run one
    action for one robot's CURRENT state, advance that state, return
    {frames, domain, equations}. Params match mw_agentic/choreography.py's
    schema (dx, x/y, height, deflection, theta)."""
    s = STATES[robot]
    if tool == "walk":
        frames, (xe, ye, the) = rm.base_move(DET["mobile_robot"], s["base_x"], s["base_y"],
                                             s["base_theta"], s["base_x"] + params["dx"], s["base_y"])
        s["base_x"], s["base_y"], s["base_theta"] = xe, ye, the
        bump_heat(robot)
        return {"frames": frames, "domain": "mobile_robot", "equations": DOMAIN_EQ["mobile_robot"]}
    if tool == "reach":
        frames, (t1, t2) = rm.arm_reach(DET["planar_arm"], s["theta1"], s["theta2"], params["x"], params["y"])
        s["theta1"], s["theta2"] = t1, t2
        bump_heat(robot)
        return {"frames": frames, "domain": "planar_arm", "equations": DOMAIN_EQ["planar_arm"]}
    if tool == "hop":
        frames = rm.hop(DET["legged_contact"], launch_height=params["height"])
        bump_heat(robot, 0.3)
        return {"frames": frames, "domain": "legged_contact", "equations": DOMAIN_EQ["legged_contact"]}
    if tool == "lift":
        frames = rm.lift_payload(DET["flexible_robot"], deflection=params["deflection"])
        bump_heat(robot, 0.18)
        return {"frames": frames, "domain": "flexible_robot", "equations": DOMAIN_EQ["flexible_robot"]}
    if tool == "joint":
        frames = rm.joint_rotate(DET["joint_actuator"], s["joint_ref"], params["theta"])
        s["joint_ref"] = params["theta"]
        bump_heat(robot)
        return {"frames": frames, "domain": "joint_actuator", "equations": DOMAIN_EQ["joint_actuator"]}
    raise ValueError(f"unknown tool {tool!r}")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/state")
def get_state():
    robot = request.args.get("robot", "a")
    return jsonify(STATES[robot])


@app.route("/api/arm/reach", methods=["POST"])
def arm_reach():
    body = request.get_json(force=True)
    robot = body.get("robot", "a")
    out = _run_tool(robot, "reach", {"x": float(body["x"]), "y": float(body["y"])})
    return jsonify(out)


@app.route("/api/joint/rotate", methods=["POST"])
def joint_rotate():
    body = request.get_json(force=True)
    robot = body.get("robot", "a")
    out = _run_tool(robot, "joint", {"theta": float(body["theta"])})
    return jsonify(out)


@app.route("/api/base/move", methods=["POST"])
def base_move():
    body = request.get_json(force=True)
    robot = body.get("robot", "a")
    s = STATES[robot]
    dx = float(body["x"]) - s["base_x"]      # the route's own contract is an absolute world x,y target;
    out = _run_tool(robot, "walk", {"dx": dx})  # _run_tool takes a relative dx - convert once, here
    return jsonify(out)


@app.route("/api/hop", methods=["POST"])
def hop():
    body = request.get_json(silent=True) or {}
    robot = body.get("robot", "a")
    out = _run_tool(robot, "hop", {"height": float(body.get("height", 0.5))})
    return jsonify(out)


@app.route("/api/lift", methods=["POST"])
def lift():
    body = request.get_json(silent=True) or {}
    robot = body.get("robot", "a")
    out = _run_tool(robot, "lift", {"deflection": float(body.get("deflection", 0.7))})
    return jsonify(out)


@app.route("/api/thermal/status")
def thermal_status():
    robot = request.args.get("robot", "a")
    s = STATES[robot]
    elapsed = time.time() - s["last_activity"]
    frames = rm.thermal_tick(DET["actuator_thermal"], heat_level=s["heat"])
    frac = min(elapsed / 14.0, 1.0)
    idx = min(int(frac * (len(frames) - 1)), len(frames) - 1)
    current = frames[idx]
    s["heat"] = max(0.02, s["heat"] * 0.985)
    return jsonify({"excess": current["excess"], "T": current["T"],
                   "domain": "actuator_thermal", "equations": DOMAIN_EQ.get("actuator_thermal", "")})


@app.route("/api/choreograph", methods=["POST"])
def choreograph_route():
    """Ask mw_agentic for a choreography (recipe first, LLM only if needed -
    see mw_agentic/orchestrate.py), then resolve every action into real
    decoded frames by running it through the matching trained manifold, in
    each robot's own time order. Returns the spec, which path was taken
    (recipe/llm), how many model calls it cost, and the resolved per-action
    frames for the frontend to schedule and play."""
    if mw_choreograph is None:
        return jsonify({"ok": False, "errors": ["mw_agentic is not installed - the "
                       "agentic choreography layer is not part of this repo"]}), 501

    body = request.get_json(force=True)
    task = (body.get("task") or "").strip()
    if not task:
        return jsonify({"ok": False, "errors": ["task is required"]}), 400

    result = mw_choreograph(task)
    if not result["ok"]:
        return jsonify(result), 422

    spec = result["spec"]
    by_robot = {"a": [], "b": []}
    for action in spec["actions"]:
        by_robot[action["robot"]].append(action)
    for robot in by_robot:
        by_robot[robot].sort(key=lambda a: a["t"])

    resolved = []
    for robot, actions in by_robot.items():
        for action in actions:
            out = _run_tool(robot, action["tool"], action["params"])
            resolved.append({"robot": robot, "t": action["t"], "tool": action["tool"], **out})

    return jsonify({**result, "resolved": resolved})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
