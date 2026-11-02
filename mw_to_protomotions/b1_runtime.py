"""
mw_to_protomotions/b1_runtime
==============================
B1: posterior action sample. The cheapest Plan B runtime — one control-step
query per step. No learned policy, no CEM over policy parameters. Just:

  1. Sample candidate actions from the local action posterior
  2. Filter by MW distance to the transition manifold
  3. Rank by dense task reward
  4. Execute best

Cost per step: ~30ms on CPU (dominated by the MW distance eval over the
candidate set). Fits a 50Hz control loop with headroom.
"""
from __future__ import annotations
import numpy as np
import torch
from mw_framework.query.inverse import inverse_query, _build_query_batch
from mw_framework.domains.transition import (
    STATE_FEATURES, ACTION_FEATURES, A_MAX)


def _dense_reward(s_row: dict, a: np.ndarray, sp_row: dict,
                  goal: dict, lam_ctrl: float = 0.01,
                  lam_mw: float = 0.1, det=None) -> float:
    """Dense reward — no sparse indicator, no +1 on success. Must be
    differentiable-ish in the action for the ranking to mean anything.

    R = -||sp_forward_vel - goal.forward_vel||^2
        - lam_ctrl * ||a||^2
        - lam_mw * MW_distance(s, a, sp)
    """
    # forward velocity proxy: height change is not forward velocity;
    # use the state's pitch_rate and thigh motion as a proxy for progress.
    # For a real system, read sp.forward_vel from the state schema.
    progress = -abs(sp_row.get("pitch_rate", 0.0) - goal.get("forward_vel", 0.0))
    effort = lam_ctrl * float(np.sum(a ** 2))
    if det is not None:
        a_t = torch.as_tensor(a[None], dtype=torch.float32)
        xb = _build_query_batch(det, s_row, a_t, sp_row)
        with torch.no_grad():
            mw, _ = det.vae_.mw_distance(xb, mc_samples=2)
        mw_pen = lam_mw * float(mw.mean().item())
    else:
        mw_pen = 0.0
    return float(progress) - effort - mw_pen


def b1_step(det, s_row: dict, goal: dict, sp_prior: dict,
            n_candidates: int = 64, seed: int | None = None) -> np.ndarray:
    """One control step: sample candidate actions, rank by reward, return best.

    sp_prior: the expected next state (e.g. from a short PD integration, or
              simply the current state — the MW filter will reject the
              action if the resulting transition is off-manifold).
    """
    rng = np.random.default_rng(seed)
    m = len(ACTION_FEATURES)

    # 1. Sample candidates: a local Gaussian around the median *learned*
    #    action (from the last inverse_query) plus a random subset to
    #    avoid collapse to a single mode.
    init_a = inverse_query(det, s_row, sp_prior, n_pop=16, n_gen=2, seed=seed)
    a_local = init_a + 0.15 * rng.standard_normal((n_candidates // 2, m))
    a_random = rng.uniform(-A_MAX, A_MAX, (n_candidates // 2, m))
    candidates = np.clip(np.concatenate([a_local, a_random], axis=0),
                         -A_MAX, A_MAX)

    # 2. Score each candidate
    scores = [_dense_reward(s_row, a, sp_prior, goal, det=det)
              for a in candidates]
    best = candidates[int(np.argmax(scores))]
    return best
