#!/usr/bin/env python3
"""
train_mw_3d.py - MW-framework training of the 3D policy manifold (Pyro SVI via MWDetector).

  1. teacher -> roll out in MuJoCo, collect (s, a) pairs (clean + noisy)       [cached with --data_cache]
  2. MWDetector(vae=PolicyManifold3DVAE(physics priors from the MuJoCo model)).fit(pairs)   <- Pyro ELBO + physics laws
  3. act(s) = latent query on the manifold (ManifoldPolicy); evaluate vs teacher on the SAME held-out seeds
  4. save checkpoints_3d/<env>[_ppo]<name>_mw.pth + _mw_report.json
Run from for_distilation/ after the teacher exists (a PPO teacher ships in checkpoints_3d/):
    python train_mw_3d.py --env Humanoid-v5 --teacher ppo
Improvements from ../STUDENT_COLLAPSE_FIXES.md are switchable, so each can be compared on the same teacher data:
    --features gravity     orientation as a gravity vector in the torso frame (instead of a raw quaternion)
    --ang_prior            P5: torso angular speed bounded by what the teacher does (needs --features gravity)
    --sym_prior            P6: soft left/right hip anti-phase prior (reports the measured correlation first)
    --data_cache F.npz     collect the teacher data once, reuse it for every later run
    --name _tag            suffix for the output files, so runs do not overwrite each other
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, time, argparse
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
import train_policy_3d as tp
from policy_3d import make_env, features, features_gravity, feat_dim
from mw_policy_3d import PolicyManifold3DVAE, spec_from_env, hip_pair
from eval_student import evaluate

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
    """(s, a) with s = framework features, a = PPO's action (PPO sees the full normalised obs).
    Both state layouts are recorded in the same pass so either can be trained from one cached dataset."""
    seed, noise = args
    ve, m, env = _ppo["v"], _ppo["m"], tp._env
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=int(seed)); d = env.unwrapped.data
    hi = tp._cfg["hi"]; S, Sg, A, steps, tot = [], [], [], 0, 0.0
    for _ in range(tp._cfg["max_steps"]):
        a, _ = m.predict(ve.normalize_obs(obs[None]), deterministic=True)
        a = np.clip(a[0], -hi, hi)
        S.append(features(d)); Sg.append(features_gravity(d)); A.append(a / hi)
        ae = np.clip(a + noise * hi * rng.standard_normal(a.shape), -hi, hi) if noise else a
        obs, r, term, trunc, _ = env.step(ae); steps += 1; tot += r
        if term or trunc: break
    return S, A, steps, tot, float(d.qpos[0]), Sg


def _posterior_report(det, df, n=2000):
    """Is the latent being used, and how well does the manifold reproduce (s, a)?  (collapse + under-fit check)"""
    vae = det.vae_
    x = torch.as_tensor(det.norm_.transform(df.iloc[:n].to_numpy(dtype=np.float32)), dtype=torch.float32)
    with torch.no_grad():
        mu, sig = vae.encode(x)
        kl = 0.5 * (mu ** 2 + sig ** 2 - 2 * torch.log(sig) - 1).sum(-1).mean().item()
        active = int((mu.var(0) > 0.01).sum().item())
        xh = vae.decode(mu)
        err = ((x - xh) ** 2).mean(0)                                  # data is z-scored, so this is a fraction of variance
    sd = vae.s_dim_
    rep = {"kl_mean": kl, "active_latent_dims": active, "latent_dims": int(vae.latent_dim),
           "recon_state_frac_var": float(err[:sd].mean()), "recon_action_frac_var": float(err[sd:].mean())}
    print(f"posterior check: KL={kl:.2f}  active latent dims={active}/{rep['latent_dims']}  "
          f"reconstruction error (fraction of variance): state={rep['recon_state_frac_var']:.3f} "
          f"action={rep['recon_action_frac_var']:.3f}", flush=True)
    if kl < 0.1 or active == 0:
        print("note: the latent is barely used - try a larger latent/hidden size, or a smaller kl_weight", flush=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Humanoid-v5"); ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--max_steps", type=int, default=500); ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--teacher", default="cem", choices=["cem", "ppo"]); ap.add_argument("--latent", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128); ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--n_eval", type=int, default=10)
    ap.add_argument("--features", default="legacy", choices=["legacy", "gravity"])
    ap.add_argument("--ang_prior", action="store_true"); ap.add_argument("--sym_prior", action="store_true")
    ap.add_argument("--data_cache", default=""); ap.add_argument("--name", default="")
    ap.add_argument("--query_steps", type=int, default=25); ap.add_argument("--query_lr", type=float, default=0.2)
    a = ap.parse_args(); det_env[0] = a.env
    tag = a.env.split("-")[0].lower()
    if a.features == "gravity" and a.teacher == "cem":
        sys.exit("--features gravity needs --teacher ppo (the CEM teacher's linear policy is tied to the legacy features)")
    if a.ang_prior and a.features != "gravity":
        sys.exit("--ang_prior is defined on the gravity layout: add --features gravity")
    theta = np.load(os.path.join(tp.CK, f"{tag}_theta.npy")) if a.teacher == "cem" else None
    probe = make_env(a.env); probe.reset(seed=0); spec = spec_from_env(probe, a.features); hip = hip_pair(probe.unwrapped.model)
    probe.close()
    print("physics spec:", {k: (v if not isinstance(v, list) else f"[{len(v)}]") for k, v in spec.items()}, flush=True)

    teacher_rep, pool = None, None
    cache = a.data_cache and os.path.exists(a.data_cache)
    if cache:
        z = np.load(a.data_cache, allow_pickle=False)
        S_all = {"legacy": z["S_legacy"], "gravity": z["S_gravity"]}[a.features]; A_raw = z["A"]
        teacher_rep = json.loads(str(z["teacher_rep"])) or None
        print(f"loaded {len(A_raw)} cached teacher rows from {a.data_cache}", flush=True)
    elif a.teacher == "ppo":
        ppo_path = os.path.join(tp.CK, f"{tag}_ppo")
        pool = Pool(a.workers, initializer=_ppo_init, initargs=(a.env, a.max_steps, ppo_path))
        data = pool.map(_collect_ppo, [(s, nz) for s in range(200) for nz in (0.0, 0.1, 0.3)])
        held = pool.map(_collect_ppo, [(s, 0.0) for s in range(5000, 5000 + a.n_eval)])
        teacher_rep = {"reward": float(np.mean([h[3] for h in held])), "steps": float(np.mean([h[2] for h in held])),
                       "x": float(np.mean([h[4] for h in held]))}
        print("PPO TEACHER held-out:", teacher_rep, flush=True)
        S_leg = np.array([x for d in data for x in d[0]], dtype=np.float32)
        S_grv = np.array([x for d in data for x in d[5]], dtype=np.float32)
        A_raw = np.array([x for d in data for x in d[1]], dtype=np.float32)
        S_all = {"legacy": S_leg, "gravity": S_grv}[a.features]
        if a.data_cache:
            np.savez_compressed(a.data_cache, S_legacy=S_leg, S_gravity=S_grv, A=A_raw, teacher_rep=json.dumps(teacher_rep))
            print(f"cached teacher data -> {a.data_cache}", flush=True)
        pool.close()
    else:
        pool = Pool(a.workers, initializer=tp._init, initargs=(a.env, a.max_steps))
        data = pool.map(tp._collect, [(theta, s, nz) for s in range(200) for nz in (0.0, 0.1, 0.3)])
        S_all = np.array([x for d in data for x in d[0]], dtype=np.float32)
        A_raw = np.array([x for d in data for x in d[1]], dtype=np.float32)
        pool.close()
    out_tag = (tag + "_ppo" if a.teacher == "ppo" else tag) + a.name
    S, A = S_all, A_raw * spec["hi"]                                   # physical torques
    names = [f"s{i}" for i in range(spec["s_dim"])] + [f"a{i}" for i in range(spec["a_dim"])]
    df = pd.DataFrame(np.concatenate([S, A], 1), columns=names)
    if a.rows and a.rows < len(df):
        df = df.sample(a.rows, random_state=0).reset_index(drop=True)
    print(f"collected {len(df)} (s,a) rows", flush=True)

    extra = {}
    if a.ang_prior:                                                   # P5: limit = what the teacher itself does, with margin
        w = np.linalg.norm(df[["s3", "s4", "s5"]].to_numpy(), axis=1)
        extra["ang_limit"] = float(np.percentile(w, 99.9) * 1.25)
        print(f"P5 angular-speed limit {extra['ang_limit']:.2f} rad/s (teacher 99.9th pct x1.25)", flush=True)
    if hip is not None:
        c = float(np.corrcoef(df[f"a{hip[0]}"], df[f"a{hip[1]}"])[0, 1])
        print(f"left/right hip_y torque correlation in the teacher data: {c:+.3f}"
              f"{'  (negative = anti-phase)' if c < 0 else '  (positive: an anti-phase prior would fight the teacher)'}", flush=True)
        if a.sym_prior:
            extra["sym_pair"] = list(hip)
    elif a.sym_prior:
        print("this body has no right_hip_y/left_hip_y actuators: --sym_prior ignored", flush=True)

    from mw_framework.core import get_device
    dev = get_device()
    print(f"MW VAE (Pyro SVI) training on: {dev}", flush=True)
    if dev.type == "cpu":
        torch.set_num_threads(a.workers)
    t0 = time.time()
    vae = PolicyManifold3DVAE(latent_dim=a.latent, hidden_dim=a.hidden, **spec, **extra)
    from mw_framework import MWDetector
    det = MWDetector(vae=vae, epochs=a.epochs, lr=1e-3, batch_size=512, verbose=True).fit(df)
    path = os.path.join(tp.CK, f"{out_tag}_mw.pth"); det.save(path)
    train_s = time.time() - t0
    print(f"MWDetector (Pyro SVI) trained in {train_s:.0f}s -> {path}", flush=True)
    r = det.mw_distance(df.sample(min(2000, len(df)), random_state=0))
    print(f"MW distance on training manifold: median {np.median(r):.3f}  p99 {np.percentile(r, 99):.3f}", flush=True)
    post = _posterior_report(det, df)

    ev = evaluate(path, a.env, a.n_eval, a.max_steps, a.query_steps, a.query_lr, a.workers)
    print(f"MW-MANIFOLD POLICY held-out: reward={ev['reward']:.1f} steps={ev['steps']:.1f} x={ev['x']:.2f}  "
          f"(query steps={a.query_steps}, lr={a.query_lr})", flush=True)
    rp_path = os.path.join(tp.CK, f"{tag}_report.json")
    rp = json.load(open(rp_path)) if os.path.exists(rp_path) else {}
    json.dump({"env": a.env, "teacher": teacher_rep or rp.get("teacher"), "plain_cvae_baseline": rp.get("cvae"),
               "mw_manifold": {"reward": ev["reward"], "steps": ev["steps"], "x": ev["x"]},
               "settings": {"features": a.features, "latent": a.latent, "hidden": a.hidden, "epochs": a.epochs,
                            "rows": len(df), "ang_prior": a.ang_prior, "sym_prior": a.sym_prior,
                            "query_steps": a.query_steps, "query_lr": a.query_lr},
               "posterior": post, "n_rows": len(df), "mw_median": float(np.median(r)), "train_s": train_s},
              open(os.path.join(tp.CK, f"{out_tag}_mw_report.json"), "w"), indent=1)
