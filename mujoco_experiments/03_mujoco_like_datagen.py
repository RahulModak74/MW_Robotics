#!/usr/bin/env python3
"""
03 - "MuJoCo-like" data generation, same recipe as the battery pybamm_like_gen scripts
=========================================================================================
The battery generators (~/SANDIA_TEST_MW/pybamm_like_gen_*.py) that produced the
validated Sandia/HUST results do NOT sample from a sparse fixed grid of conditions.
They simulate hundreds of independent UNITS, each with its own randomly-drawn physical
parameters, then densely sample CONTINUOUSLY varying operating conditions within each
unit's own lifetime (MultiLevelDataGenerator.simulate_battery: num_batteries x
samples_per_battery, soc/dod/c_rate/temperature all drawn per-sample, not from a grid).

01_validator_check.py and 02_forecaster_check.py both trained on a sparse grid instead:
10 heights x 4 damping settings (40 fixed combinations). This script applies the SAME
recipe the battery work used - many independent "drops", each with CONTINUOUSLY randomised
height and contact damping, densely sampled - and re-measures both the validator and
forecaster accuracy to see whether a correctly-shaped (denser, continuous) training
manifold does better, as the battery precedent suggests it should.

Run: python mujoco_experiments/03_mujoco_like_datagen.py   (several minutes)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyro
import torch

from mw_framework import MWDetector, roc_auc
from mw_framework.core import set_seed
from mw_framework.datasets import LEGGED_FEATURES
from mw_framework.domains import HoppingContactVAE
from bounce_env import simulate_drop, extract_flight_segments, get_post_bounce_segment, \
    split_flight_segments

# Stable, physical bounce range found earlier (dampratio ~0.15 and below goes
# numerically unstable - apparent energy GAIN, a timestep/stiffness artifact).
DAMPRATIO_LO, DAMPRATIO_HI = 0.22, 0.52
HEIGHT_LO, HEIGHT_HI = 0.4, 3.2
N_UNITS = 80            # independent "drops", each its own randomly-drawn physics -
                        # the battery generator's "num_batteries"

rng = np.random.default_rng(0)
print(f"Simulating {N_UNITS} independent MuJoCo drops, each with CONTINUOUSLY randomised "
     f"height (U[{HEIGHT_LO},{HEIGHT_HI}]) and contact damping (U[{DAMPRATIO_LO},{DAMPRATIO_HI}]) "
     f"- the pybamm_like_gen recipe, not a fixed grid.")
rows = []
for unit_id in range(N_UNITS):
    h0 = float(rng.uniform(HEIGHT_LO, HEIGHT_HI))
    dr = float(rng.uniform(DAMPRATIO_LO, DAMPRATIO_HI))
    t, z, vz = simulate_drop(h0, dr, n_steps=6000)
    df = extract_flight_segments(t, z, vz, n_sample=40)
    df["unit_id"] = unit_id
    rows.append(df)
mujoco_like = pd.concat(rows, ignore_index=True)
print(f"Generated {len(mujoco_like)} rows from {mujoco_like.unit_id.nunique()} units "
     f"(was 5382 rows from 40 fixed grid combinations before).")
print(f"e range: {mujoco_like.e.min():.3f}-{mujoco_like.e.max():.3f}  "
     f"v0 range: {mujoco_like.v0.min():.3f}-{mujoco_like.v0.max():.3f}")

set_seed(0); pyro.clear_param_store()
det = MWDetector(vae=HoppingContactVAE(), epochs=200, lr=2e-3).fit(mujoco_like[LEGGED_FEATURES])
norm, vae = det.norm_, det.vae_

# ── Held-out drops at heights/dampratios NOT in the training set ──
print("\n-- Validator check: held-out drops at heights/dampratios never trained on --")
held_rows = []
for h0 in (0.6, 1.4, 2.3, 3.0):
    for dr in (0.25, 0.35, 0.48):
        t, z, vz = simulate_drop(h0, dr, n_steps=6000)
        held_rows.append(extract_flight_segments(t, z, vz, n_sample=40))
held_df = pd.concat(held_rows, ignore_index=True)
base = np.median(det.mw_distance(mujoco_like[LEGGED_FEATURES]))
held_dist = det.mw_distance(held_df[LEGGED_FEATURES])
print(f"  held-out trajectories: median distance = {np.median(held_dist)/base:.2f}x training baseline")
print(f"  flagged (risk>=50): {(det.predict(held_df[LEGGED_FEATURES]) == -1).mean():.1%}  "
     f"(compare: grid-trained model flagged 15.6% of REAL MuJoCo data in 01_validator_check.py)")


def to_latent(row):
    x = norm.transform(np.array([list(row)]))
    xt = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        mu, _ = vae.encode(xt)
    return mu[0].detach().clone()


def reconstruct(rows_df):
    x = norm.transform(rows_df[LEGGED_FEATURES].values)
    xt = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        mu, _ = vae.encode(xt)
        phys = vae.to_physical(vae.decode(mu))
    return pd.DataFrame({k: v.numpy() for k, v in phys.items()})


print("\n-- Forecaster check: same held-out post-bounce segment as 02_forecaster_check.py --")
t_full, z_full, vz_full = simulate_drop(2.0, 0.3)
seg_t, seg_z, seg_vz, v0_true, e_true = get_post_bounce_segment(t_full, z_full, vz_full, which=1)
z_range, vz_range = seg_z.max() - seg_z.min(), seg_vz.max() - seg_vz.min()

sample_idx = np.linspace(0, len(seg_t) - 1, 20).astype(int)
sample_df = pd.DataFrame({"t": seg_t[sample_idx], "z": seg_z[sample_idx], "vz": seg_vz[sample_idx],
                          "v0": v0_true, "e": e_true})
recon = reconstruct(sample_df)
z_err = np.abs(recon["z"].values - sample_df["z"].values).mean()
vz_err = np.abs(recon["vz"].values - sample_df["vz"].values).mean()
print(f"  pure reconstruction: z_err={z_err:.4f} ({z_err/z_range*100:.1f}% of range)  "
     f"vz_err={vz_err:.4f} ({vz_err/vz_range*100:.1f}% of range)")
print(f"  (compare: grid-trained model measured 2.5% / 1.0% in 02_forecaster_check.py)")


def eval_hop(dt_hop, n_decode_points=4, n_hops=8):
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
        tt = torch.linspace(0, 1, n_decode_points)[:, None]
        t0 = time.perf_counter()
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


print("\n  hop-size accuracy (straight-line interpolation), compare to 02_forecaster_check.py:")
t0 = time.perf_counter()
simulate_drop(2.0, 0.3, n_steps=1000)
mujoco_step_ms = (time.perf_counter() - t0) * 1000 / 1000
for dt_hop in (0.02, 0.05, 0.1, 0.2, 0.4):
    n_steps_per_hop = int(dt_hop / 0.002)
    if n_steps_per_hop >= len(seg_t) - 2:
        continue
    ez, evz, tms = eval_hop(dt_hop)
    mj_ms = n_steps_per_hop * mujoco_step_ms
    print(f"    dt_hop={dt_hop:.2f}s ({n_steps_per_hop} steps): "
         f"z_err={ez/z_range*100:.1f}%  vz_err={evz/vz_range*100:.1f}%  "
         f"MW={tms:.3f}ms  MuJoCo~{mj_ms:.3f}ms  {'MW faster' if tms < mj_ms else 'MuJoCo faster'}")
