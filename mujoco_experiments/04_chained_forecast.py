#!/usr/bin/env python3
"""
04 - Chained forecasting: does error compound, or stay bounded?
===================================================================
Everything in 02/03 tested a SINGLE hop between two REAL, known states.

IMPORTANT, found while building this: the current architecture (straight-line
latent interpolation between two encoded endpoints) has NO way to predict a
future state from a single current state alone - every hop needs a KNOWN
target to interpolate toward. This matches what ROBOTICS.md already says:
"forecast endpoints come from an external physics model; the manifold shapes
the path between them." So a true target-free forecast chain cannot be built
with what exists yet - that is itself a finding, not a workaround to hide.

What CAN be tested honestly: chain the STARTING point only. Each hop still
interpolates toward the real known checkpoint ahead (the only mode this
architecture supports), but hop N+1 starts from hop N's FORECASTED state
instead of the real one. This isolates whether a slightly-wrong start
compounds into bigger errors over many hops, or self-corrects because the
target is always real - the real question behind "does long-range chaining
work", answered as precisely as the current architecture allows.

Run: python mujoco_experiments/04_chained_forecast.py   (a few minutes)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyro
import torch

from mw_framework import MWDetector
from mw_framework.core import set_seed
from mw_framework.datasets import LEGGED_FEATURES
from mw_framework.domains import HoppingContactVAE
from bounce_env import simulate_drop, extract_flight_segments, get_post_bounce_segment

# Train the same way as 03 (the richer, pybamm_like-style recipe).
DAMPRATIO_LO, DAMPRATIO_HI = 0.22, 0.52
HEIGHT_LO, HEIGHT_HI = 0.4, 3.2
N_UNITS = 80
rng = np.random.default_rng(0)
rows = []
for unit_id in range(N_UNITS):
    h0 = float(rng.uniform(HEIGHT_LO, HEIGHT_HI))
    dr = float(rng.uniform(DAMPRATIO_LO, DAMPRATIO_HI))
    t, z, vz = simulate_drop(h0, dr, n_steps=6000)
    rows.append(extract_flight_segments(t, z, vz, n_sample=40))
mujoco_like = pd.concat(rows, ignore_index=True)

set_seed(0); pyro.clear_param_store()
det = MWDetector(vae=HoppingContactVAE(), epochs=200, lr=2e-3).fit(mujoco_like[LEGGED_FEATURES])
norm, vae = det.norm_, det.vae_


def to_latent(row):
    x = norm.transform(np.array([list(row)]))
    xt = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        mu, _ = vae.encode(xt)
    return mu[0].detach().clone()


def hop(start_row, end_row, frac=0.9):
    """One straight-line latent hop from start_row TOWARD end_row, decoded at
    `frac` of the way there (NOT frac=1.0 - at the exact endpoint, a straight
    line is just z1 itself, independent of z0, which makes the "forecast"
    trivially ignore the start point. frac<1 keeps the result a genuine
    function of both ends, which is required for chaining to mean anything)."""
    z0, z1 = to_latent(start_row), to_latent(end_row)
    z_at = (1 - frac) * z0 + frac * z1
    with torch.no_grad():
        phys = vae.to_physical(vae.decode(z_at[None]))
    return float(phys["z"][0]), float(phys["vz"][0])


# Held-out trajectory
t_full, z_full, vz_full = simulate_drop(2.0, 0.3)
seg_t, seg_z, seg_vz, v0_true, e_true = get_post_bounce_segment(t_full, z_full, vz_full, which=1)
z_range, vz_range = seg_z.max() - seg_z.min(), seg_vz.max() - seg_vz.min()
print(f"Held-out segment: {len(seg_t)} steps, duration={seg_t[-1]:.3f}s, v0={v0_true:.3f}, e={e_true:.3f}")

HOP_STEPS = 50          # 0.1s hops - near the single-hop speed/accuracy crossover
checkpoints = list(range(0, len(seg_t) - 1, HOP_STEPS))
if checkpoints[-1] != len(seg_t) - 1:
    checkpoints.append(len(seg_t) - 1)
n_hops = len(checkpoints) - 1
FRAC = 0.9       # decode 90% of the way to the next checkpoint - see hop()'s docstring
print(f"Chaining {n_hops} hops of {HOP_STEPS} steps (0.1s) each, each decoded at "
     f"{FRAC*100:.0f}% of the way to its target checkpoint.\n")

# ── Chained: hop N+1 starts from hop N's FORECASTED (z,vz), not the real one ──
cur_z, cur_vz = float(seg_z[0]), float(seg_vz[0])
chained_results = [(seg_t[0], cur_z, cur_vz)]
for i in range(n_hops):
    idx0, idx1 = checkpoints[i], checkpoints[i + 1]
    start_row = (seg_t[idx0], cur_z, cur_vz, v0_true, e_true)         # FORECASTED start
    end_row = (seg_t[idx1], seg_z[idx1], seg_vz[idx1], v0_true, e_true)  # REAL target (architecture needs it)
    cur_z, cur_vz = hop(start_row, end_row, frac=FRAC)
    t_at = seg_t[idx0] + FRAC * (seg_t[idx1] - seg_t[idx0])
    chained_results.append((t_at, cur_z, cur_vz))

# ── Re-anchored (control): every hop starts from the REAL state, no chaining ──
reanchored_results = [(seg_t[0], float(seg_z[0]), float(seg_vz[0]))]
for i in range(n_hops):
    idx0, idx1 = checkpoints[i], checkpoints[i + 1]
    start_row = (seg_t[idx0], seg_z[idx0], seg_vz[idx0], v0_true, e_true)  # REAL start
    end_row = (seg_t[idx1], seg_z[idx1], seg_vz[idx1], v0_true, e_true)
    z_f, vz_f = hop(start_row, end_row, frac=FRAC)
    t_at = seg_t[idx0] + FRAC * (seg_t[idx1] - seg_t[idx0])
    reanchored_results.append((t_at, z_f, vz_f))

print(f"{'hop':>4} {'t':>7} {'chained z err':>14} {'chained vz err':>15} "
     f"{'re-anchored z err':>18} {'re-anchored vz err':>19}")
for i, (t_i, cz, cvz) in enumerate(chained_results):
    true_z = np.interp(t_i, seg_t, seg_z)
    true_vz = np.interp(t_i, seg_t, seg_vz)
    cz_err = abs(cz - true_z) / max(z_range, 1e-9) * 100
    cvz_err = abs(cvz - true_vz) / max(vz_range, 1e-9) * 100
    rt_i, rz, rvz = reanchored_results[i]
    rz_err = abs(rz - true_z) / max(z_range, 1e-9) * 100
    rvz_err = abs(rvz - true_vz) / max(vz_range, 1e-9) * 100
    print(f"{i:>4} {t_i:>7.3f} {cz_err:>13.1f}% {cvz_err:>14.1f}% "
         f"{rz_err:>17.1f}% {rvz_err:>18.1f}%")

print("\nIf the 'chained' columns grow much faster than the 're-anchored' columns, a wrong start")
print("compounds into bigger error over hops. If they track closely, the always-real target")
print("pulls the forecast back toward truth regardless of a slightly-wrong start.")
