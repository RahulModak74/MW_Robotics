#!/usr/bin/env python3
"""
SPEC_with_MW_3.md section 5.2 - B2, short-horizon MPC on the manifold.

Honest scope note, found while building this: the TransitionVAE's 9 state
features (height, pitch, pitch_rate, 6 joint angles) exclude x-position/
velocity, the same deliberate choice walker_domain.py made (x is
"unconstrained," not part of physical validity). That means this manifold
cannot predict forward progress - testing B2 on a LOCOMOTION goal would
need x-velocity added to the feature set and the VAE retrained (not done
here). Tested instead on the BALANCE goal, which the manifold does
represent - an honest substitution, not the original target task.

forward_query implementation note: a flat joint VAE (this repo's scope
choice, see transition_domain.py's docstring) doesn't give a direct
p(s'|s,a) sampler the way a proper fibre-bundle conditional model would.
Approximated here via iterative imputation: start from s'=s (no change),
encode/decode, read off the predicted s' block, repeat a few times until
self-consistent. Cheap - just a few MLP forward passes, no search.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from mw_framework import MWDetector
from transition_domain import FEATURES, STATE_FEATURES, ACTION_FEATURES, NEXT_FEATURES

det = MWDetector.load("transition_mw_walker.ckpt")
vae, norm = det.vae_, det.norm_
N_STATE = len(STATE_FEATURES)


def forward_query(S, A, n_iters=3):
    """S: (n,9), A: (n,6) -> predicted S' (n,9), via iterative imputation."""
    n = len(S)
    sp_guess = S.copy()
    for _ in range(n_iters):
        X = np.concatenate([S, A, sp_guess], axis=1)
        Xn = norm.transform(X)
        with torch.no_grad():
            mu, _ = vae.encode(torch.tensor(Xn, dtype=torch.float32))
            xhat_n = vae.decode(mu).numpy()
        xhat = norm.inverse(xhat_n)
        sp_guess = xhat[:, N_STATE + 6:N_STATE + 6 + N_STATE]
    return sp_guess


def balance_reward(S, A, SP):
    """Dense reward for the BALANCE goal (spec 6.3's shape, substituting a
    goal the manifold can actually represent, per the scope note above)."""
    pitch, height = SP[:, 1], SP[:, 0]
    r_pitch = -(pitch ** 2)
    r_height = -np.maximum(0, 0.8 - height) ** 2 - np.maximum(0, height - 2.0) ** 2
    r_ctrl = -0.01 * (A ** 2).sum(axis=1)
    df = pd.DataFrame(np.concatenate([S, A, SP], axis=1), columns=FEATURES)
    mw_d = det.mw_distance(df[FEATURES])
    r_mw = -0.1 * mw_d
    return r_pitch + r_height + r_ctrl + r_mw


def get_state9(data):
    return np.array([data.qpos[1], data.qpos[2], data.qvel[2],
                     data.qpos[3], data.qpos[4], data.qpos[5],
                     data.qpos[6], data.qpos[7], data.qpos[8]])


def b2_plan(s_t, N=96, H=8, rng=None):
    rng = rng or np.random.default_rng()
    seqs = rng.uniform(-1, 1, size=(N, H, 6))
    s = np.repeat(s_t[None, :], N, axis=0)
    total_r = np.zeros(N)
    for h in range(H):
        a_h = seqs[:, h, :]
        sp = forward_query(s, a_h, n_iters=2)
        total_r += balance_reward(s, a_h, sp)
        s = sp
    best = np.argmax(total_r)
    return seqs[best, 0, :], total_r[best]


def run_one(N, H, seed, max_steps=200):
    rng = np.random.default_rng(seed)
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    t0 = time.time()
    steps = 0
    for t in range(max_steps):
        s_t = get_state9(data)
        a, r = b2_plan(s_t, N=N, H=H, rng=rng)
        obs, rew, term, trunc, info = env.step(a)
        steps += 1
        if term:
            break
    dt = time.time() - t0
    env.close()
    return steps, dt / steps


if __name__ == "__main__":
    N, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (96, 8)
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    results = [run_one(N, H, seed) for seed in range(n_seeds)]
    steps_arr = np.array([r[0] for r in results])
    print(f"B2 (N={N}, H={H}, {n_seeds} seeds): "
          f"mean survival = {steps_arr.mean():.1f}/200 (std={steps_arr.std():.1f}), "
          f"individual={list(steps_arr)}, "
          f"~{np.mean([r[1] for r in results])*1000:.0f}ms/decision")
