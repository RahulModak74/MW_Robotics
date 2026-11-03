#!/usr/bin/env python3
"""
app_3d.py - 3D version of the MW robot demo, driven by a POLICY instead of six stitched 1D manifolds.

The MW policy manifold (train_mw_3d.py: an mw_framework MWVAE with physics priors, trained by Pyro SVI through MWDetector)
IS the controller: every step, state -> latent query on the manifold -> action block -> torques,
fed into a real MuJoCo 3D body (Humanoid-v5 / Ant-v5). The browser (Three.js) draws the world pose of every
MuJoCo geom, frame by frame. Nothing is hand-animated.

Run:
    python train_policy_3d.py --env Humanoid-v5     # CEM teacher + plain-CVAE baseline
    python train_mw_3d.py --env Humanoid-v5         # MW manifold (Pyro) - what the app serves
    python app_3d.py                                 # then open http://127.0.0.1:5052

HONEST SCOPE: see README.md. A policy is only as good as its teacher; on Humanoid-v5 that teacher is the
documented CPU "3D balance wall" (for_full_replacement/domain/29DoF/PROGRESS_LOG.md) - the report numbers are shown in the UI as-is.
"""
import os, sys, json
# SERVING IS CPU-ONLY BY DESIGN. Training may use a consumer GPU (train_mw_3d.py picks CUDA when it exists), but the
# trained manifold is queried one tiny optimisation per control step, where a GPU is slower than the CPU and where
# MWDetector.load() would otherwise put the model on CUDA while ManifoldPolicy builds its tensors on the CPU
# (device mismatch). Pin it before mw_framework is imported; export MW_DEVICE=cuda to override on purpose.
os.environ.setdefault("MW_DEVICE", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import mujoco
from flask import Flask, jsonify, request, send_from_directory

from policy_3d import CVAE, LinearPolicy, make_env, features
from mw_framework import MWDetector
from mw_policy_3d import ManifoldPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "checkpoints_3d")
torch.set_num_threads(int(os.environ.get("MW_THREADS", "2")))
app = Flask(__name__, static_folder=os.path.join(HERE, "static_3d"), static_url_path="")

ENVS = {"humanoid": "Humanoid-v5", "ant": "Ant-v5"}
MAX_STEPS = 1000
BODIES = {}      # tag -> {env_name, cvae, theta, report}
for tag in ("humanoid_ppo", "humanoid", "ant"):       # humanoid_ppo = MW manifold distilled from the PPO teacher
    mwp = os.path.join(CK, f"{tag}_mw.pth")
    if not os.path.exists(mwp):
        continue
    base = tag.replace("_ppo", "")
    try:
        det = MWDetector.load(mwp)
    except Exception as e:                            # a bad/old checkpoint must not take the other bodies down
        print(f"skipping {tag}: cannot load {mwp}: {e}", file=sys.stderr)
        continue
    rp = os.path.join(CK, f"{tag}_mw_report.json")
    BODIES[tag] = {"env": ENVS[base], "mw_det": det,
                   "hi": det.vae_.hi_, "fdim": det.vae_.s_dim_, "adim": det.vae_.a_dim_,
                   "cvae": None, "theta": None, "ppo": None,
                   "report": json.load(open(rp)) if os.path.exists(rp) else {}}
    cp = os.path.join(CK, f"{base}_cvae.pt")
    if tag == base and os.path.exists(cp):
        ck = torch.load(cp, map_location="cpu")
        m = CVAE(ck["fdim"], ck["adim"]); m.load_state_dict(ck["state"]); m.eval()
        BODIES[tag].update(cvae=m, theta=np.load(os.path.join(CK, f"{base}_theta.npy")))
if not BODIES:
    raise SystemExit(f"No trained MW manifold in {CK} (expected <body>_mw.pth). Train one first, e.g.\n"
                     "  python train_mw_3d.py --env Humanoid-v5 --teacher ppo    (needs the PPO teacher files)\n"
                     "  python train_policy_3d.py --env Humanoid-v5 && python train_mw_3d.py --env Humanoid-v5")
print(f"3D policies loaded: {list(BODIES)}  (inference device: {os.environ['MW_DEVICE']})")


class PPOTeacher:
    """The stable-baselines3 teacher, so the UI can show teacher and student side by side. Optional: needs
    stable-baselines3 and the *_ppo.zip + *_ppo_vecnorm.pkl files; if either is missing the mode is simply absent."""
    def __init__(self, base, env_name):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        path = os.path.join(CK, f"{base}_ppo")
        self.vn = VecNormalize.load(path + "_vecnorm.pkl", DummyVecEnv([lambda: make_env(env_name)]))
        self.vn.training, self.vn.norm_reward = False, False
        self.m = PPO.load(path, device="cpu")

    def act(self, obs, hi):
        a, _ = self.m.predict(self.vn.normalize_obs(obs[None]), deterministic=True)
        return np.clip(a[0], -hi, hi)


for _tag, _b in BODIES.items():
    _base = _tag.replace("_ppo", "")
    if _tag.endswith("_ppo") and os.path.exists(os.path.join(CK, f"{_base}_ppo.zip")) \
            and os.path.exists(os.path.join(CK, f"{_base}_ppo_vecnorm.pkl")):
        try:
            _b["ppo"] = PPOTeacher(_base, _b["env"])
        except Exception as e:
            print(f"PPO teacher mode unavailable for {_tag}: {e}", file=sys.stderr)


def _geoms(model):
    out = []
    for i in range(model.ngeom):
        gt = int(model.geom_type[i])
        if gt not in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_SPHERE,
                      mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX,
                      mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CYLINDER):
            continue
        out.append({"i": i, "type": mujoco.mjtGeom(gt).name.replace("mjGEOM_", "").lower(),
                    "size": [round(float(x), 4) for x in model.geom_size[i]],
                    "rgba": [round(float(x), 3) for x in model.geom_rgba[i]],
                    "body": int(model.geom_bodyid[i])})
    return out


def _pose(data, gids):
    quat = np.zeros(4)
    P, Q = [], []
    for i in gids:
        mujoco.mju_mat2Quat(quat, data.geom_xmat[i])
        P.append([round(float(x), 3) for x in data.geom_xpos[i]])
        Q.append([round(float(x), 3) for x in quat])        # w, x, y, z
    return P, Q


def run_rollout(tag, mode, steps, seed, noise, push):
    b = BODIES[tag]
    env = make_env(b["env"])
    obs, _ = env.reset(seed=seed)
    model, data = env.unwrapped.model, env.unwrapped.data
    gids = [g["i"] for g in _geoms(model)]
    pol = LinearPolicy(b["theta"], b["fdim"], b["adim"], b["hi"]) if b["theta"] is not None else None
    rng = np.random.default_rng(seed)
    wanted = mode
    if (mode == "teacher" and b["theta"] is None) or (mode.startswith("cvae") and b["cvae"] is None):
        mode = "mw_manifold"          # this body has no CEM/plain-CVAE baseline (PPO-distilled)
    if mode == "ppo_teacher" and b["ppo"] is None:
        mode = "mw_manifold"
    mwpol = ManifoldPolicy(b["mw_det"]) if mode == "mw_manifold" else None
    frames, total, fell_at = [], 0.0, None
    for t in range(steps):
        f = features(data)
        if mode == "mw_manifold":
            a = mwpol.act(mwpol.observe(data))        # each manifold knows which state layout it was trained on
        elif mode == "ppo_teacher":
            a = b["ppo"].act(obs, b["hi"])
        elif mode == "teacher":
            a = pol(f)
        else:
            a = b["cvae"].act(torch.as_tensor(f[None], dtype=torch.float32),
                              stochastic=(mode == "cvae_stochastic")).numpy()[0] * b["hi"]
        if noise:
            a = np.clip(a + noise * b["hi"] * rng.standard_normal(a.shape), -b["hi"], b["hi"])
        if push and t == 60:                                    # one lateral shove, to see it react
            data.qvel[0:3] += np.array([0.0, push, 0.0])
        obs, r, term, trunc, _ = env.step(a)
        total += r
        P, Q = _pose(data, gids)
        frames.append({"p": P, "q": Q, "x": round(float(data.qpos[0]), 3), "z": round(float(data.qpos[2]), 3)})
        if term:
            fell_at = t + 1
            break
        if trunc:                                               # the environment's own time limit, not a fall
            break
    env.close()
    # mode_used makes a silent fallback visible: the UI must not label MW-manifold frames as the teacher's.
    return frames, {"steps": len(frames), "fell_at": fell_at, "reward": round(total, 1),
                    "forward_m": round(frames[-1]["x"] - frames[0]["x"], 3),
                    "mode_requested": wanted, "mode_used": mode}


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index_3d.html")


@app.route("/api3d/bodies")
def bodies():
    return jsonify({t: {"env": b["env"], "report": b["report"],
                        "modes": ["mw_manifold"] + (["ppo_teacher"] if b["ppo"] else [])
                                 + (["teacher", "cvae", "cvae_stochastic"] if b["cvae"] is not None else [])}
                    for t, b in BODIES.items()})


@app.route("/api3d/health")
def health():
    return jsonify({"ok": True, "bodies": list(BODIES), "inference_device": os.environ["MW_DEVICE"],
                    "torch": torch.__version__, "cuda_visible": torch.cuda.is_available(),
                    "mujoco": mujoco.__version__})


@app.route("/api3d/model")
def model_info():
    tag = request.args.get("body") or next(iter(BODIES))
    if tag not in BODIES:
        return jsonify({"error": f"body {tag!r} not trained; available: {list(BODIES)}"}), 404
    env = make_env(BODIES[tag]["env"]); env.reset(seed=0)
    g = _geoms(env.unwrapped.model); env.close()
    return jsonify({"geoms": g})


@app.route("/api3d/rollout", methods=["POST"])
def rollout():
    b = request.get_json(silent=True) or {}
    tag = b.get("body") or next(iter(BODIES))          # default: the first trained body
    if tag not in BODIES:
        return jsonify({"error": f"body {tag!r} not trained; available: {list(BODIES)}"}), 404
    try:
        steps, seed = max(1, min(int(b.get("steps", 400)), MAX_STEPS)), int(b.get("seed", 1))
        noise, push = float(b.get("noise", 0.0)), float(b.get("push", 0.0))
    except (TypeError, ValueError):
        return jsonify({"error": "steps/seed must be integers, noise/push numbers"}), 400
    frames, summary = run_rollout(tag, str(b.get("mode", "mw_manifold")), steps, seed, noise, push)
    return jsonify({"frames": frames, "summary": summary, "dt": 0.03})


if __name__ == "__main__":
    app.run(host=os.environ.get("MW_HOST", "127.0.0.1"), port=int(os.environ.get("MW_PORT", "5052")), debug=False)
