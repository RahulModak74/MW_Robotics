#!/usr/bin/env python3
"""
train_mw_bundle_3d.py - train the MW fibre-bundle policy (state base + action fibre, Pyro SVI) from cached teacher data.

    python train_mw_bundle_3d.py --data_cache data_cache/ppo_humanoid.npz --name _b1
Teacher data comes from `train_mw_3d.py --teacher ppo --data_cache ...` (collected once, shared by every experiment).
Output: checkpoints_3d/humanoid_ppo_bundle<name>.pt (+ _report.json). Held-out scoring uses the same seeds as the teacher.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from policy_3d import make_env
from mw_policy_3d import spec_from_env, hip_pair
from mw_bundle_policy import build_bundle, BundlePolicy, evaluate_bundle
from mw_framework import MWNormaliser, MWTrainer
from mw_framework.core import get_device

CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_3d")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cache", required=True); ap.add_argument("--env", default="Humanoid-v5")
    ap.add_argument("--rows", type=int, default=100000); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--latent_s", type=int, default=32); ap.add_argument("--latent_a", type=int, default=2)
    ap.add_argument("--hidden_s", type=int, default=256); ap.add_argument("--hidden_a", type=int, default=256)
    ap.add_argument("--sigma_s", type=float, default=0.15); ap.add_argument("--sigma_a", type=float, default=0.2)
    ap.add_argument("--no_priors", action="store_true"); ap.add_argument("--sym_prior", action="store_true")
    ap.add_argument("--batch", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_eval", type=int, default=10); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    tag = a.env.split("-")[0].lower()

    probe = make_env(a.env); probe.reset(seed=0); spec = spec_from_env(probe, "legacy"); hip = hip_pair(probe.unwrapped.model); probe.close()
    z = np.load(a.data_cache)
    S_all, A_all = z["S_legacy"], z["A"] * spec["hi"]                       # physical torques
    teacher = json.loads(str(z["teacher_rep"]))
    idx = pd.Series(np.arange(len(S_all))).sample(min(a.rows, len(S_all)), random_state=0).values   # same rows as train_mw_3d
    X = np.concatenate([S_all[idx], A_all[idx]], 1).astype(np.float32)
    names = [f"s{i}" for i in range(spec["s_dim"])] + [f"a{i}" for i in range(spec["a_dim"])]
    norm = MWNormaliser.fit(X, names)
    cfg = dict(latent_s=a.latent_s, latent_a=a.latent_a, hidden_s=a.hidden_s, hidden_a=a.hidden_a, sigma_s=a.sigma_s,
               sigma_a=a.sigma_a, priors=not a.no_priors, sym_pair=list(hip) if (a.sym_prior and hip) else None)
    alg = build_bundle(spec, norm, **cfg)
    print(alg.describe().split("\n")[0:2], f"+ {len(alg.glues)} physics glue laws", flush=True)
    dev = get_device(); print(f"MW fibre-bundle (Pyro SVI) training on: {dev}   rows={len(X)}", flush=True)
    t0 = time.time()
    MWTrainer(alg, lr=a.lr, epochs=a.epochs, batch_size=a.batch, verbose=True, log_every=10).fit(norm.transform(X))
    train_s = time.time() - t0
    alg = alg.cpu().eval()
    path = os.path.join(CK, f"{tag}_ppo_bundle{a.name}.pt")
    pol = BundlePolicy(alg, norm, spec, cfg); pol.save(path)
    print(f"trained in {train_s:.0f}s -> {path}", flush=True)

    # how well does ONE forward pass (state -> action) reproduce the teacher's action?   (fraction of action variance)
    sd = spec["s_dim"]; sub = np.random.default_rng(0).choice(len(X), 3000, replace=False)
    pred = np.stack([pol.act(X[i, :sd]) for i in sub]); true = X[sub, sd:]
    act_err = float(((pred - true) ** 2).mean() / true.var(0).mean())
    print(f"state -> action, one pass: error = {act_err:.4f} of action variance", flush=True)
    ev = evaluate_bundle(path, a.env, a.n_eval, 500, a.workers)
    print(f"MW-BUNDLE POLICY held-out: steps={ev['steps']:.1f} x={ev['x']:.2f}   per-seed {ev['per_seed_steps']}", flush=True)
    json.dump({"env": a.env, "teacher": teacher, "mw_bundle": ev, "action_error_frac_var": act_err, "settings": vars(a),
               "cfg": cfg, "train_s": train_s}, open(os.path.join(CK, f"{tag}_ppo_bundle{a.name}_report.json"), "w"), indent=1)
