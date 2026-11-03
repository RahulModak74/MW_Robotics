#!/usr/bin/env python3
"""EXP_03 / E2b — §4 follow-up: train the same CVAE recipe on the WALK policy
(cem_walk_theta.npy) and check whether the CVAE walks. Original walk policy is
evaluated on the same seeds for a fair baseline."""
import sys, os, time
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, gymnasium as gym, torch
from exp_e1_vae_policy_compress import collect, train_cvae, eval_policy_replacement
from exp_e2_vae_policy_transfer import eval_walk
from mw_to_protomotions.cem_policy_search import get_features, policy_action

def walk_original(theta, seeds, max_steps=600):
    fwd, steps = [], []
    for seed in seeds:
        env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
        env.reset(seed=seed); d = env.unwrapped.data; x0 = d.qpos[0]; n = 0
        for _ in range(max_steps):
            _, _, term, _, _ = env.step(policy_action(theta, get_features(d))); n += 1
            if term: break
        fwd.append(float(d.qpos[0] - x0)); steps.append(n); env.close()
    return np.array(fwd), np.array(steps)

theta = np.load("cem_walk_theta.npy")
t0 = time.time()
S, A = collect(theta, seeds=list(range(0, 100)), max_steps=600)
Sh, Ah = collect(theta, seeds=list(range(1000, 1020)), max_steps=600)
print(f"collected {len(S)} train / {len(Sh)} held-out pairs in {time.time()-t0:.0f}s", flush=True)
model = train_cvae(S, A, latent_dim=8, hidden=64, beta=0.01)
err, _ = eval_policy_replacement(model, Sh, Ah)
print(f"policy replacement error (held-out): {err:.4f}")
seeds = list(range(3000, 3020))
fo, so = walk_original(theta, seeds)
fc, sc = eval_walk(model, seeds)
print(f"original walk policy : fwd {fo.mean():.2f} m (median {np.median(fo):.2f}), steps {so.mean():.1f}/600")
print(f"walk-trained CVAE    : fwd {fc.mean():.2f} m (median {np.median(fc):.2f}), steps {sc.mean():.1f}/600")
print(f"ratio fwd CVAE/orig  : {fc.mean()/fo.mean():.2f}")
torch.save({"state_dict": model.state_dict(),
            "config": dict(state_dim=15, action_dim=6, latent_dim=8, hidden=64)}, "exp_03_cvae_walk.pt")
