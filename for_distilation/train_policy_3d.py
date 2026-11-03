#!/usr/bin/env python3
"""
train_policy_3d.py - step 1 of the 3D demo: train a policy on a true-3D MuJoCo body, then
compress it into a CVAE (the "VAE as policy" object from domain/EXP_03.md).

  1. CEM (multiprocess, CPU only) over a linear policy  a = hi*tanh(W f + b)
     fitness = mean env reward over a few seeds (healthy + forward-velocity - ctrl cost)
  2. Roll the CEM policy out, collect (s, a) pairs (with action-noise for state coverage)
  3. Train a CVAE on the pairs, evaluate it against the teacher on the SAME seeds
  4. Save checkpoints_3d/<env>_{theta.npy,cvae.pt,report.json}

Run from for_distilation/:
    python train_policy_3d.py --env Humanoid-v5 --pop 240 --gens 60
    python train_policy_3d.py --env Ant-v5      --pop 120 --gens 40
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.optim as optim
from multiprocessing import Pool

from policy_3d import make_env, features, feat_dim, LinearPolicy, CVAE

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "checkpoints_3d")
os.makedirs(CK, exist_ok=True)

_env = None
_cfg = None


def _init(env_name, max_steps):
    global _env, _cfg
    torch.set_num_threads(1)
    _env = make_env(env_name)
    _cfg = dict(adim=_env.action_space.shape[0], hi=float(_env.action_space.high[0]),
                fdim=feat_dim(_env), max_steps=max_steps)


def _rollout(theta, seed, model=None, noise=0.0, record=False):
    env = _env
    pol = LinearPolicy(theta, _cfg["fdim"], _cfg["adim"], _cfg["hi"]) if model is None else None
    rng = np.random.default_rng(seed)
    env.reset(seed=int(seed))
    data = env.unwrapped.data
    total, steps, S, A = 0.0, 0, [], []
    for _ in range(_cfg["max_steps"]):
        f = features(data)
        if model is None:
            a = pol(f)
        else:
            a = model.act(torch.as_tensor(f[None], dtype=torch.float32)).numpy()[0] * _cfg["hi"]
        if record:
            S.append(f); A.append(a / _cfg["hi"])
        if noise:
            a = np.clip(a + noise * _cfg["hi"] * rng.standard_normal(a.shape), -_cfg["hi"], _cfg["hi"])
        _, r, term, trunc, info = env.step(a)
        total += r; steps += 1
        if term or trunc:
            break
    x = float(data.qpos[0])
    return total, steps, x, S, A


def _fit(args):
    theta, seeds = args
    out = [_rollout(theta, s) for s in seeds]
    return np.mean([o[0] for o in out])


def _collect(args):
    theta, seed, noise = args
    _, steps, _, S, A = _rollout(theta, seed, noise=noise, record=True)
    return S, A


def _eval_theta(args):
    theta, seed = args
    r, st, x, _, _ = _rollout(theta, seed)
    return r, st, x


def _eval_cvae(args):
    state, sdim, adim, seed = args
    m = CVAE(sdim, adim); m.load_state_dict(state); m.eval()
    r, st, x, _, _ = _rollout(None, seed, model=m)
    return r, st, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Humanoid-v5")
    ap.add_argument("--pop", type=int, default=240)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    tag = a.env.split("-")[0].lower()

    probe = make_env(a.env)
    adim, hi, fdim = probe.action_space.shape[0], float(probe.action_space.high[0]), feat_dim(probe)
    probe.close()
    dim = adim * fdim + adim
    print(f"[{a.env}] actions={adim} (hi={hi}) feats={fdim} params={dim}", flush=True)

    pool = Pool(a.workers, initializer=_init, initargs=(a.env, a.max_steps))
    rng = np.random.default_rng(0)
    mean, std = np.zeros(dim), np.full(dim, 0.3)
    n_elite = max(2, int(a.pop * 0.1))
    t0 = time.time(); best_theta, best_fit = mean.copy(), -1e9
    for g in range(a.gens):
        pop = mean + std * rng.standard_normal((a.pop, dim))
        seeds = rng.integers(0, 100000, a.n_seeds).tolist()
        fits = np.array(pool.map(_fit, [(t, seeds) for t in pop]))
        el = np.argsort(fits)[-n_elite:]
        mean, std = pop[el].mean(0), pop[el].std(0) + 0.02
        if fits.max() > best_fit:
            best_fit, best_theta = fits.max(), pop[fits.argmax()].copy()
        print(f"  gen {g:3d} mean={fits.mean():8.1f} best={fits.max():8.1f} ({time.time()-t0:.0f}s)", flush=True)
    theta = mean
    np.save(os.path.join(CK, f"{tag}_theta.npy"), theta)

    held = list(range(5000, 5020))
    ev = pool.map(_eval_theta, [(theta, s) for s in held])
    t_rew, t_steps, t_x = (np.mean([e[i] for e in ev]) for i in range(3))
    print(f"TEACHER held-out: reward={t_rew:.1f} steps={t_steps:.1f} x={t_x:.2f}", flush=True)

    # --- collect (s, a): clean + noisy rollouts of the teacher
    jobs = [(theta, s, nz) for s in range(200) for nz in (0.0, 0.1, 0.3)]
    data = pool.map(_collect, jobs)
    S = np.array([x for d in data for x in d[0]], dtype=np.float32)
    A = np.array([x for d in data for x in d[1]], dtype=np.float32)
    print(f"collected {len(S)} (s,a) pairs", flush=True)

    torch.set_num_threads(a.workers)
    model = CVAE(fdim, adim)
    model.s_mean.copy_(torch.as_tensor(S.mean(0))); model.s_std.copy_(torch.as_tensor(S.std(0) + 1e-3))
    opt = optim.Adam(model.parameters(), lr=1e-3)
    St, At = torch.as_tensor(S), torch.as_tensor(A)
    n = len(S)
    for ep in range(a.epochs):
        perm = torch.randperm(n); tl = tr = tk = 0.0; nb = 0
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            loss, rec, kl = model(St[idx], At[idx], beta=0.01)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); tr += rec.item(); tk += kl.item(); nb += 1
        if ep % 25 == 0 or ep == a.epochs - 1:
            print(f"  cvae ep {ep:3d} recon={tr/nb:.5f} kl={tk/nb:.4f}", flush=True)
    model.eval()
    torch.save({"state": model.state_dict(), "fdim": fdim, "adim": adim, "hi": hi, "env": a.env},
               os.path.join(CK, f"{tag}_cvae.pt"))
    ev2 = pool.map(_eval_cvae, [(model.state_dict(), fdim, adim, s) for s in held])
    c_rew, c_steps, c_x = (np.mean([e[i] for e in ev2]) for i in range(3))
    print(f"CVAE    held-out: reward={c_rew:.1f} steps={c_steps:.1f} x={c_x:.2f}", flush=True)
    json.dump({"env": a.env, "params": dim, "pop": a.pop, "gens": a.gens, "max_steps": a.max_steps,
               "teacher": {"reward": t_rew, "steps": t_steps, "x": t_x},
               "cvae": {"reward": c_rew, "steps": c_steps, "x": c_x}, "n_pairs": int(n),
               "train_s": time.time() - t0},
              open(os.path.join(CK, f"{tag}_report.json"), "w"), indent=1)
    pool.close()


if __name__ == "__main__":
    main()
