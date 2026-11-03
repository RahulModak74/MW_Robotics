#!/usr/bin/env python3
"""
train_mw_3d.py - MW-framework training of the 3D policy manifold (Pyro SVI via MWDetector).

  1. teacher theta (CEM, from train_policy_3d.py) -> roll out in MuJoCo, collect (s, a) pairs (clean + noisy)
  2. MWDetector(vae=PolicyManifold3DVAE(physics priors from the MuJoCo model)).fit(pairs)   <- Pyro ELBO + physics laws
  3. act(s) = latent query on the manifold (ManifoldPolicy); evaluate vs teacher on the SAME held-out seeds
  4. save checkpoints_3d/<env>_mw.pth + <env>_mw_report.json
Run from full_flow_animation/ after train_policy_3d.py:
    python train_mw_3d.py --env Humanoid-v5
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, json, time, argparse
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
import train_policy_3d as tp
from policy_3d import make_env, features, feat_dim
from mw_policy_3d import PolicyManifold3DVAE, spec_from_env, ManifoldPolicy
from mw_framework import MWDetector


def _cpu_only():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""      # MuJoCo rollout workers: CPU inference, no CUDA context
    torch.set_num_threads(1)


def _eval_mw(args):
    path, seed, max_steps, env_name = args
    torch.set_num_threads(1)
    det = MWDetector.load(path); pol = ManifoldPolicy(det)
    env = make_env(env_name); env.reset(seed=seed); d = env.unwrapped.data
    tot, n = 0.0, 0
    for _ in range(max_steps):
        _, r, term, trunc, _ = env.step(pol.act(features(d)))
        tot += r; n += 1
        if term or trunc: break
    return tot, n, float(d.qpos[0])

det_env = ["Humanoid-v5"]
_ppo = {}


def _ppo_init(env_name, max_steps, path):
    tp._init(env_name, max_steps)
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    ve = VecNormalize.load(path + "_vecnorm.pkl", DummyVecEnv([lambda: make_env(env_name)]))
    ve.training = False; ve.norm_reward = False
    _ppo["m"], _ppo["v"] = PPO.load(path, device="cpu"), ve


def _collect_ppo(args):
    """(s, a) with s = framework features, a = PPO's action (PPO sees the full normalised obs)."""
    seed, noise = args
    ve, m, env = _ppo["v"], _ppo["m"], tp._env
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=int(seed)); d = env.unwrapped.data
    hi = tp._cfg["hi"]; S, A, steps, tot = [], [], 0, 0.0
    for _ in range(tp._cfg["max_steps"]):
        a, _ = m.predict(ve.normalize_obs(obs[None]), deterministic=True)
        a = np.clip(a[0], -hi, hi)
        S.append(features(d)); A.append(a / hi)
        ae = np.clip(a + noise * hi * rng.standard_normal(a.shape), -hi, hi) if noise else a
        obs, r, term, trunc, _ = env.step(ae); steps += 1; tot += r
        if term or trunc: break
    return S, A, steps, tot, float(d.qpos[0])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Humanoid-v5"); ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--max_steps", type=int, default=500); ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--teacher", default="cem", choices=["cem", "ppo"]); ap.add_argument("--latent", type=int, default=8); ap.add_argument("--n_eval", type=int, default=10)
    a = ap.parse_args(); det_env[0] = a.env
    tag = a.env.split("-")[0].lower()
    theta = np.load(os.path.join(tp.CK, f"{tag}_theta.npy")) if a.teacher == "cem" else None
    probe = make_env(a.env); spec = spec_from_env(probe); probe.close()
    print("physics spec:", {k: (v if not isinstance(v, list) else f"[{len(v)}]") for k, v in spec.items()}, flush=True)

    if a.teacher == "ppo":
        ppo_path = os.path.join(tp.CK, f"{tag}_ppo")
        pool = Pool(a.workers, initializer=_ppo_init, initargs=(a.env, a.max_steps, ppo_path))
        data = pool.map(_collect_ppo, [(s, nz) for s in range(200) for nz in (0.0, 0.1, 0.3)])
        held = pool.map(_collect_ppo, [(s, 0.0) for s in range(5000, 5000 + a.n_eval)])
        teacher_rep = {"reward": float(np.mean([h[3] for h in held])), "steps": float(np.mean([h[2] for h in held])),
                       "x": float(np.mean([h[4] for h in held]))}
        print("PPO TEACHER held-out:", teacher_rep, flush=True)
        tag = tag + "_ppo"
    else:
        teacher_rep = None
        pool = Pool(a.workers, initializer=tp._init, initargs=(a.env, a.max_steps))
        data = pool.map(tp._collect, [(theta, s, nz) for s in range(200) for nz in (0.0, 0.1, 0.3)])
    S = np.array([x for d in data for x in d[0]], dtype=np.float32)
    A = np.array([x for d in data for x in d[1]], dtype=np.float32) * spec["hi"]      # physical torques
    names = [f"s{i}" for i in range(spec["s_dim"])] + [f"a{i}" for i in range(spec["a_dim"])]
    df = pd.DataFrame(np.concatenate([S, A], 1), columns=names)
    print(f"collected {len(df)} (s,a) rows", flush=True)

    from mw_framework.core import get_device
    dev = get_device()
    print(f"MW VAE (Pyro SVI) training on: {dev}", flush=True)
    if dev.type == "cpu":
        torch.set_num_threads(a.workers)
    t0 = time.time()
    vae = PolicyManifold3DVAE(latent_dim=a.latent, **spec)
    det = MWDetector(vae=vae, epochs=a.epochs, lr=1e-3, batch_size=512, verbose=True).fit(df)
    path = os.path.join(tp.CK, f"{tag}_mw.pth"); det.save(path)
    print(f"MWDetector (Pyro SVI) trained in {time.time()-t0:.0f}s -> {path}", flush=True)
    r = det.mw_distance(df.sample(2000, random_state=0))
    print(f"MW distance on training manifold: median {np.median(r):.3f}  p99 {np.percentile(r,99):.3f}", flush=True)

    import multiprocessing as mp
    pool.close(); pool = mp.get_context("spawn").Pool(a.workers, initializer=_cpu_only)
    ev = pool.map(_eval_mw, [(path, s, a.max_steps, a.env) for s in range(5000, 5000 + a.n_eval)])
    m = [float(np.mean([e[i] for e in ev])) for i in range(3)]
    print(f"MW-MANIFOLD POLICY held-out: reward={m[0]:.1f} steps={m[1]:.1f} x={m[2]:.2f}", flush=True)
    rp = json.load(open(os.path.join(tp.CK, f"{tag}_report.json"))) if os.path.exists(os.path.join(tp.CK, f"{tag}_report.json")) else {}
    pool.close()
    json.dump({"env": a.env, "teacher": teacher_rep or rp.get("teacher"), "plain_cvae_baseline": rp.get("cvae"),
               "mw_manifold": {"reward": m[0], "steps": m[1], "x": m[2]}, "n_rows": len(df),
               "mw_median": float(np.median(r)), "train_s": time.time() - t0},
              open(os.path.join(tp.CK, f"{tag}_mw_report.json"), "w"), indent=1)
