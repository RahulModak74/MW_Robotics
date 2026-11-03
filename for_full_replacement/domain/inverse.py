"""
mw_framework.query.inverse
============================
inverse_query(det, s, s') → a* — solve for the action that most plausibly
carried state s to state s' under the learned transition manifold.

Method: small CEM over the action block only, minimizing MW distance to the
manifold. m = 6 for Walker2d, ~30 for Humanoid. n_pop = 32, n_gen = 3 →
96 VAE forward passes. ~30ms on a commodity CPU core.

This is the primitive Plan B substitutes for a policy network: instead of
"policy(s) → a", we ask "which a makes (s, a, s') look like a real transition?"
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Optional


def _build_query_batch(det, s_row: dict, a_batch: torch.Tensor,
                        sp_row: dict) -> torch.Tensor:
    """Assemble (N, n_features) batch: s fixed, a varies, s' fixed.
    Returns a NORMALISED tensor on the model's device."""
    n = a_batch.shape[0]
    from ..domains.transition import STATE_FEATURES, ACTION_FEATURES
    # physical rows
    s_vec = np.array([s_row[f] for f in STATE_FEATURES], dtype=np.float32)
    sp_vec = np.array([sp_row[f] for f in STATE_FEATURES], dtype=np.float32)
    a_np = a_batch.detach().cpu().numpy().astype(np.float32)
    s_block = np.tile(s_vec, (n, 1))
    sp_block = np.tile(sp_vec, (n, 1))
    full = np.concatenate([s_block, a_np, sp_block], axis=1)
    xn = det.norm_.transform(full)
    dev = next(det.vae_.parameters()).device
    return torch.as_tensor(xn, dtype=torch.float32, device=dev)


def inverse_query(det, s_row: dict, sp_row: dict,
                  n_pop: int = 32, n_gen: int = 3, elite_frac: float = 0.2,
                  init_std: float = 0.5, seed: Optional[int] = None) -> np.ndarray:
    """Return a* (physical units) that minimises MW distance to the manifold
    for the triple (s_row, a*, sp_row).

    s_row / sp_row: dicts keyed by STATE_FEATURES (e.g. {"height": 1.25, ...}).
    Returns: numpy array of 6 torques, physical units, clipped to ±A_MAX.
    """
    from ..domains.transition import ACTION_FEATURES, A_MAX
    rng = np.random.default_rng(seed)
    m = len(ACTION_FEATURES)
    n_elite = max(1, int(n_pop * elite_frac))

    mean = np.zeros(m, dtype=np.float32)
    std = np.ones(m, dtype=np.float32) * init_std

    for _ in range(n_gen):
        # sample in physical action units directly (not normalised)
        pop = mean + std * rng.standard_normal((n_pop, m))
        pop = np.clip(pop, -A_MAX, A_MAX)
        a_t = torch.as_tensor(pop, dtype=torch.float32)
        xbatch = _build_query_batch(det, s_row, a_t, sp_row)
        with torch.no_grad():
            dist, _ = det.vae_.mw_distance(xbatch, mc_samples=2)
        d = dist.cpu().numpy()
        elite_idx = np.argsort(d)[:n_elite]
        elite = pop[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-3

    return np.clip(mean, -A_MAX, A_MAX).astype(np.float32)
