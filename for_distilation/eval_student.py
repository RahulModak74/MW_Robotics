#!/usr/bin/env python3
"""
eval_student.py - score a trained MW policy manifold on held-out MuJoCo seeds (CPU, parallel).

    python eval_student.py --ckpt checkpoints_3d/humanoid_ppo_mw.pth
    python eval_student.py --ckpt checkpoints_3d/humanoid_ppo_mw.pth --query_steps 10 --query_lr 0.1 --n_eval 20

Handy for comparing latent-query settings on one trained model without retraining.
"""
import os, sys, json, argparse
os.environ.setdefault("MW_DEVICE", "cpu")                 # the policy is served on CPU
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import multiprocessing as mp
from policy_3d import make_env
from mw_policy_3d import ManifoldPolicy


def _cpu_only():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""               # MuJoCo rollout workers: no CUDA context
    torch.set_num_threads(1)


def _eval_one(args):
    path, seed, max_steps, env_name, qsteps, qlr = args
    torch.set_num_threads(1)
    from mw_framework import MWDetector
    pol = ManifoldPolicy(MWDetector.load(path), steps=qsteps, lr=qlr)
    env = make_env(env_name); env.reset(seed=seed); d = env.unwrapped.data
    tot, n = 0.0, 0
    for _ in range(max_steps):
        _, r, term, trunc, _ = env.step(pol.act(pol.observe(d)))
        tot += r; n += 1
        if term or trunc:
            break
    return tot, n, float(d.qpos[0])


def evaluate(path, env_name="Humanoid-v5", n_eval=10, max_steps=500, qsteps=25, qlr=0.2, workers=10, first_seed=5000):
    """Mean reward / steps / forward distance over held-out seeds (the same seeds the teacher is scored on)."""
    jobs = [(path, s, max_steps, env_name, qsteps, qlr) for s in range(first_seed, first_seed + n_eval)]
    with mp.get_context("spawn").Pool(min(workers, n_eval), initializer=_cpu_only) as pool:
        ev = pool.map(_eval_one, jobs)
    return {"reward": float(np.mean([e[0] for e in ev])), "steps": float(np.mean([e[1] for e in ev])),
            "x": float(np.mean([e[2] for e in ev])), "per_seed_steps": [e[1] for e in ev],
            "query_steps": qsteps, "query_lr": qlr, "n_eval": n_eval}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--env", default="Humanoid-v5")
    ap.add_argument("--n_eval", type=int, default=10); ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--query_steps", type=int, default=25); ap.add_argument("--query_lr", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    r = evaluate(a.ckpt, a.env, a.n_eval, a.max_steps, a.query_steps, a.query_lr, a.workers)
    print(json.dumps(r, indent=1))
