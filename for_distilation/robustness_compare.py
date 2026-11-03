#!/usr/bin/env python3
"""
robustness_compare.py - do the physics laws buy robustness?  Policies are scored under perturbations they were never trained on.

    python robustness_compare.py --policy mlp=data_cache/std_mlp.pt --policy cvae=data_cache/std_cvae.pt \
        --policy bundle_laws=checkpoints_3d/humanoid_ppo_bundle_b5_s02.pt --policy bundle_nolaws=...pt

Conditions: clean | action noise 0.1 / 0.3 (x the torque limit) | a lateral shove of 1 / 2 m/s at step 60.
Metric: mean steps survived (cap 500) over `--n` held-out seeds, plus the share of episodes that reach the cap.
"""
import os, sys, json, argparse, time
os.environ.setdefault("MW_DEVICE", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import multiprocessing as mp
from policy_3d import make_env, features, CVAE

HI = 0.4
CONDITIONS = {"clean": (0.0, 0.0), "noise0.1": (0.1, 0.0), "noise0.3": (0.3, 0.0), "push1": (0.0, 1.0), "push2": (0.0, 2.0)}


def _cpu_only():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)


def _load(kind_path):
    kind, path = kind_path
    if path.endswith("std_mlp.pt") or kind.startswith("mlp"):
        from compare_mw_vs_std import MLPPolicy
        ck = torch.load(path, map_location="cpu")
        m = MLPPolicy(ck["fdim"], ck["adim"], ck["hidden"]); m.load_state_dict(ck["state"]); m.eval()
        return lambda f: HI * m(torch.as_tensor(f[None], dtype=torch.float32))[0].detach().numpy()
    if kind.startswith("cvae"):
        ck = torch.load(path, map_location="cpu")
        m = CVAE(ck["fdim"], ck["adim"], hidden=ck["hidden"]); m.load_state_dict(ck["state"]); m.eval()
        return lambda f: HI * m.act(torch.as_tensor(f[None], dtype=torch.float32)).numpy()[0]
    from mw_bundle_policy import BundlePolicy
    p = BundlePolicy.load(path)
    return p.act


def _run(args):
    kind, path, cond, seed, max_steps = args
    torch.set_num_threads(1)
    noise, push = CONDITIONS[cond]
    act = _load((kind, path))
    env = make_env("Humanoid-v5"); env.reset(seed=seed); d = env.unwrapped.data
    rng = np.random.default_rng(seed)
    n = 0
    for t in range(max_steps):
        a = act(features(d))
        if noise:
            a = np.clip(a + noise * HI * rng.standard_normal(a.shape), -HI, HI)
        if push and t == 60:
            d.qvel[0:3] += np.array([0.0, push, 0.0])
        _, _, term, trunc, _ = env.step(a); n += 1
        if term or trunc:
            break
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True, help="name=path (MLP / CVAE / bundle checkpoint)")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="logs_3d/robustness_compare.json")
    a = ap.parse_args()
    pols = [p.split("=", 1) for p in a.policy]
    jobs = [(k, pth, c, s, 500) for k, pth in pols for c in CONDITIONS for s in range(5000, 5000 + a.n)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_cpu_only) as pool:
        res = pool.map(_run, jobs, chunksize=4)
    table, i = {}, 0
    for k, _ in pols:
        table[k] = {}
        for c in CONDITIONS:
            v = res[i:i + a.n]; i += a.n
            table[k][c] = {"steps": float(np.mean(v)), "reach_cap": float(np.mean([x >= 500 for x in v]))}
    hdr = f"{'policy':22s}" + "".join(f"{c:>14s}" for c in CONDITIONS)
    print(hdr); print("-" * len(hdr))
    for k, _ in pols:
        print(f"{k:22s}" + "".join(f"{table[k][c]['steps']:8.0f} ({table[k][c]['reach_cap']*100:3.0f}%)" for c in CONDITIONS))
    print(f"(mean steps, share of episodes reaching the 500-step cap; {a.n} seeds per cell; {time.time()-t0:.0f}s)")
    json.dump({"table": table, "n": a.n}, open(a.out, "w"), indent=1)
