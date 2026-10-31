#!/usr/bin/env python3
"""
02 - Can MW forecast forward in time, not just validate a given state?
===========================================================================
Train HoppingContactVAE directly ON REAL MUJOCO ROLLOUTS (not the shipped
closed-form dataset). Pick a genuine POST-BOUNCE held-out segment (segment
0, the initial fall, is NEVER in the training distribution by construction
- an earlier version of this test picked it by mistake; see FORECASTER.md).
Encode its real start and end states, interpolate between them in latent
space (geodesic walk, and a cheaper straight-line variant), decode, and
compare against MuJoCo's own fine-grained trajectory over the same
interval. Then time both against re-stepping MuJoCo for the same interval,
at a range of hop sizes.

Run: python mujoco_experiments/02_forecaster_check.py   (a few minutes)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyro
import torch

from mw_framework import MWDetector
from mw_framework.core import set_seed
from mw_framework.datasets import LEGGED_FEATURES
from mw_framework.domains import HoppingContactVAE
from mw_framework.geometry import geodesic
from bounce_env import simulate_drop, extract_flight_segments, get_post_bounce_segment

print("Simulating real MuJoCo ball drops for TRAINING data (10 heights x 4 damping settings)...")
rows = []
for h0 in np.linspace(0.8, 3.0, 10):
    for dampratio in (0.25, 0.3, 0.35, 0.4):
        t, z, vz = simulate_drop(float(h0), dampratio)
        rows.append(extract_flight_segments(t, z, vz, n_sample=60))
mujoco_train = pd.concat(rows, ignore_index=True)
print(f"Trained on {len(mujoco_train)} real MuJoCo rows (not closed-form synthetic data).")

set_seed(0); pyro.clear_param_store()
det = MWDetector(vae=HoppingContactVAE(), epochs=200, lr=2e-3).fit(mujoco_train[LEGGED_FEATURES])
norm, vae = det.norm_, det.vae_


def to_latent(row):
    x = norm.transform(np.array([list(row)]))
    xt = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        mu, _ = vae.encode(xt)
    return mu[0].detach().clone()


def reconstruct(rows_df):
    """Encode -> decode immediately, no interpolation. The reconstruction floor."""
    x = norm.transform(rows_df[LEGGED_FEATURES].values)
    xt = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        mu, _ = vae.encode(xt)
        phys = vae.to_physical(vae.decode(mu))
    return pd.DataFrame({k: v.numpy() for k, v in phys.items()})


# ── A held-out drop, DIFFERENT from any training height/damping combo ──
t_full, z_full, vz_full = simulate_drop(2.0, 0.3)
seg_t, seg_z, seg_vz, v0_true, e_true = get_post_bounce_segment(t_full, z_full, vz_full, which=1)
print(f"\nHeld-out POST-BOUNCE segment: {len(seg_t)} steps, duration={seg_t[-1]:.3f}s, "
     f"v0={v0_true:.3f}, e={e_true:.3f}")

sample_idx = np.linspace(0, len(seg_t) - 1, 20).astype(int)
sample_df = pd.DataFrame({"t": seg_t[sample_idx], "z": seg_z[sample_idx], "vz": seg_vz[sample_idx],
                          "v0": v0_true, "e": e_true})
recon = reconstruct(sample_df)
z_recon_err = np.abs(recon["z"].values - sample_df["z"].values).mean()
vz_recon_err = np.abs(recon["vz"].values - sample_df["vz"].values).mean()
z_range = seg_z.max() - seg_z.min()
vz_range = seg_vz.max() - seg_vz.min()
print(f"\nPure reconstruction (no interpolation) on this held-out segment's own points:")
print(f"  z error:  {z_recon_err:.4f} ({z_recon_err/max(z_range,1e-9)*100:.1f}% of range)")
print(f"  vz error: {vz_recon_err:.4f} ({vz_recon_err/max(vz_range,1e-9)*100:.1f}% of range)")


def eval_hop(dt_hop, n_decode_points=4, use_optimizer=False, opt_steps=300, n_hops=8):
    n_steps_per_hop = max(1, int(dt_hop / 0.002))
    starts = list(range(0, len(seg_t) - n_steps_per_hop - 1, max(1, n_steps_per_hop)))[:n_hops]
    if not starts:
        return float("nan"), float("nan"), float("nan")
    errs_z, errs_vz, times_ms = [], [], []
    for s in starts:
        e_idx = s + n_steps_per_hop
        row0 = (seg_t[s], seg_z[s], seg_vz[s], v0_true, e_true)
        row1 = (seg_t[e_idx], seg_z[e_idx], seg_vz[e_idx], v0_true, e_true)
        z0, z1 = to_latent(row0), to_latent(row1)
        t0 = time.perf_counter()
        if use_optimizer:
            path, _, _ = geodesic(vae, z0, z1, n_points=n_decode_points, steps=opt_steps)
        else:
            tt = torch.linspace(0, 1, n_decode_points)[:, None]
            path = (1 - tt) * z0 + tt * z1
        with torch.no_grad():
            decoded = vae.to_physical(vae.decode(path))
        dt_ms = (time.perf_counter() - t0) * 1000
        pred = pd.DataFrame({k: v.numpy() for k, v in decoded.items()}).sort_values("t")
        t_window = seg_t[s:e_idx + 1] - seg_t[s]
        true_z = np.interp(pred["t"].values, t_window, seg_z[s:e_idx + 1])
        true_vz = np.interp(pred["t"].values, t_window, seg_vz[s:e_idx + 1])
        errs_z.append(np.abs(pred["z"].values - true_z).mean())
        errs_vz.append(np.abs(pred["vz"].values - true_vz).mean())
        times_ms.append(dt_ms)
    return float(np.mean(errs_z)), float(np.mean(errs_vz)), float(np.mean(times_ms))


print("\nStraight-line latent interpolation (no optimiser) vs optimised geodesic, "
     "same hop (half the segment):")
half = seg_t[-1] / 2
for label, use_opt in [("optimised geodesic (300-step Adam)", True), ("straight line, no optimisation", False)]:
    ez, evz, tms = eval_hop(half, n_decode_points=10, use_optimizer=use_opt)
    print(f"  {label}: z_err={ez:.4f}  vz_err={evz:.4f}  time={tms:.3f}ms")

print("\nStraight-line hops at increasing size vs. re-stepping MuJoCo for the same interval "
     "(measured, not assumed):")
mujoco_step_ms = None
t0 = time.perf_counter()
simulate_drop(2.0, 0.3, n_steps=1000)
mujoco_step_ms = (time.perf_counter() - t0) * 1000 / 1000      # ms per MuJoCo step, this machine
for dt_hop in (0.02, 0.05, 0.1, 0.2, 0.4):
    n_steps_per_hop = int(dt_hop / 0.002)
    if n_steps_per_hop >= len(seg_t) - 2:
        continue
    ez, evz, tms = eval_hop(dt_hop, n_decode_points=4, use_optimizer=False)
    mj_ms = n_steps_per_hop * mujoco_step_ms
    print(f"  dt_hop={dt_hop:.2f}s ({n_steps_per_hop} MuJoCo steps): "
         f"z_err={ez:.4f} ({ez/max(z_range,1e-9)*100:.1f}% of range)  "
         f"vz_err={evz:.4f} ({evz/max(vz_range,1e-9)*100:.1f}% of range)  "
         f"MW={tms:.3f}ms  MuJoCo~{mj_ms:.3f}ms  "
         f"{'MW faster' if tms < mj_ms else 'MuJoCo faster'}")

print("""
TAKEAWAY: see FORECASTER.md for the full interpretation, including the two bugs that had to be
fixed to get here (a v0 detection-lag, a z coordinate offset, and a held-out-segment selection
bug that tested on an out-of-distribution trajectory type) and what the results above do and do
not support.
""")
