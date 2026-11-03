#!/usr/bin/env python3
"""
EXP_03 / E2 — Does the balance-trained CVAE generalise to a walk task?
Tests whether "VAE as policy" generalises when the task changes.

Loads exp_03_cvae_balance.pt, runs it as a policy on Walker2d with the
forward-walk task. Expected outcomes:
  - ≥ 1.0 m forward: the CVAE learned general Walker2d behavior
  - 0.2–1.0 m: partial transfer, the manifold has a "good action" prior
  - < 0.2 m but survives 500+: it stands, doesn't walk (expected, not a failure)
  - < 0.2 m and falls: overfit to balance, diagnose
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import gymnasium as gym
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_e1_vae_policy_compress import CVAE, eval_rollout
from mw_to_protomotions.cem_policy_search import get_features


def eval_walk(model, seeds, max_steps=600):
    """Runs the CVAE as a policy with forward-distance measurement."""
    fwd, steps = [], []
    for seed in seeds:
        env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
        obs, info = env.reset(seed=seed)
        data = env.unwrapped.data
        x0 = data.qpos[0]
        s_count = 0
        for _ in range(max_steps):
            s = torch.as_tensor(get_features(data)[None], dtype=torch.float32)
            act = model.act_deterministic(s).numpy()[0]
            obs, r, term, trunc, info = env.step(act)
            s_count += 1
            if term:
                break
        fwd.append(float(data.qpos[0] - x0))
        steps.append(s_count)
        env.close()
    return np.array(fwd), np.array(steps)


if __name__ == "__main__":
    print("EXP_03 / E2 — VAE Policy Transfer: balance-trained CVAE on walk task")
    print("=" * 68)

    # Load the CVAE from E1
    path = "exp_03_cvae_balance.pt"
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found — run exp_e1_vae_policy_compress.py first")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = CVAE(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"loaded CVAE from {path}\n")

    # Run the CVAE as a policy on the walk task
    print("[1/2] running balance-trained CVAE on walk task (20 fresh seeds)…")
    fwd, steps = eval_walk(model, seeds=list(range(3000, 3020)))
    print(f"  mean forward: {fwd.mean():.2f} m")
    print(f"  mean steps:   {steps.mean():.1f} / 600")
    print(f"  reference:    linear walk policy = 1.33 m, "
          f"linear balance policy ≈ -0.75 m\n")

    # Verdict
    print("[2/2] verdict")
    print("=" * 68)
    if fwd.mean() >= 1.0:
        print("TRANSFERS — CVAE walks without walk-specific training.")
        print("  → the balance-trained manifold contains general walker2d behavior")
        print("  → strong evidence for 'VAE as policy, trained on any good data'")
    elif fwd.mean() >= 0.2:
        print("PARTIAL — CVAE moves forward somewhat without being trained to.")
        print("  → likely the CVAE encodes 'standing upright' behavior which "
              "translates to slow forward lean")
        print("  → train a walk-trained CVAE next for a real comparison")
    elif steps.mean() >= 500:
        print("STANDS — CVAE holds balance but doesn't walk.")
        print("  → this is expected; the balance CVAE was never trained to walk")
        print("  → the interesting follow-up: train CVAE on walk data, "
              "does it walk?")
    else:
        print("FAILS — CVAE cannot act as a walk policy.")
        print("  → check whether act_deterministic(s) produces "
              "physically-reasonable actions on walk-task states")
        print("  → likely the balance-trained distribution is too narrow")
    print("=" * 68)
