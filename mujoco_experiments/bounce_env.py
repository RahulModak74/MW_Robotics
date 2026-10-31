"""
mujoco_experiments.bounce_env
===============================
Shared MuJoCo utilities for the legged_contact validator and forecaster
experiments: simulate a ball dropped onto a plane with MuJoCo's own
contact solver, and extract flight-phase readings in the exact
LEGGED_FEATURES schema that mw_framework.domains.legged_contact.HoppingContactVAE
was built for.

Two coordinate/measurement fixes baked in here, found while building this
(see FORECASTER.md for the full story - both were real bugs, not footnotes):
  1. v0 (launch speed) is computed analytically from the apex height
     (v0 = sqrt(2*g*apex)), not read off vz at the first "airborne" sample -
     that sample lags true liftoff by a few timesteps and biased the energy
     residual by a near-constant offset.
  2. z is shifted by ball_radius so "touching the ground" is z=0, matching
     the closed-form dataset's convention - MuJoCo's ball CENTER rests at
     z=ball_radius, not z=0.

MuJoCo has no literal "coefficient of restitution" parameter; its contact
is a compliant spring-damper. `dampratio` (the second solref value) is
calibrated empirically here: ~1.0 is critically damped (no bounce at all,
e~=0), ~0.5 gives e~=0.3-0.5, lower still becomes numerically unstable
(apparent energy GAIN - a timestep/stiffness artifact, not real physics).
Keep dampratio in roughly [0.25, 0.5] for stable, physical bounces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import mujoco as mj

from mw_framework.datasets import LEGGED_FEATURES

BALL_RADIUS = 0.05
GROUND_TOL = 0.01

_XML_TEMPLATE = """
<mujoco model="bounce">
  <option gravity="0 0 -1" timestep="{dt}"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" solref="{solref0} {dampratio}" solimp="0.9 0.95 0.001"/>
    <body name="ball" pos="0 0 {h0}">
      <joint type="free"/>
      <geom name="ball_geom" type="sphere" size="{radius}" mass="1.0" solref="{solref0} {dampratio}" solimp="0.9 0.95 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


def simulate_drop(h0: float, dampratio: float, dt: float = 0.002, solref0: float = 0.01,
                  n_steps: int = 6000):
    """Drop a ball from height h0 with contact damping `dampratio`.
    Returns (t, z, vz) arrays, one entry per simulation step. z is the RAW
    MuJoCo ball-centre height (not yet shifted - see extract_flight_segments)."""
    xml = _XML_TEMPLATE.format(dt=dt, solref0=solref0, dampratio=dampratio, h0=h0, radius=BALL_RADIUS)
    m = mj.MjModel.from_xml_string(xml)
    d = mj.MjData(m)
    t, z, vz = [], [], []
    for _ in range(n_steps):
        mj.mj_step(m, d)
        t.append(d.time)
        z.append(float(d.qpos[2]))
        vz.append(float(d.qvel[2]))
    return np.array(t), np.array(z), np.array(vz)


def split_flight_segments(t: np.ndarray, z_raw: np.ndarray, vz: np.ndarray):
    """Shift z to the ground-at-zero convention and split into contiguous
    airborne segments. Returns (z_shifted, list_of_index_arrays).
    IMPORTANT: segment 0 is always the INITIAL FALL from release height -
    it is the longest/highest segment by construction (no bounce with
    e < 1 can exceed the starting height) and `extract_flight_segments`
    deliberately excludes it from training data, below. Any held-out test
    must pick a segment with index >= 1, or it is testing on a trajectory
    type the model was never shown - this was a real bug in an earlier
    version of this experiment (see FORECASTER.md)."""
    z = z_raw - BALL_RADIUS
    idx = np.where(z > GROUND_TOL)[0]
    if len(idx) == 0:
        return z, []
    splits = np.where(np.diff(idx) > 1)[0]
    segments = np.split(idx, splits + 1)
    return z, [s for s in segments if len(s) >= 5]


def extract_flight_segments(t: np.ndarray, z_raw: np.ndarray, vz: np.ndarray,
                            n_sample: int = 60) -> pd.DataFrame:
    """Training-data extraction: every post-bounce flight segment (segment 0,
    the untrained initial fall, is skipped), sampled at n_sample points each."""
    z, segments = split_flight_segments(t, z_raw, vz)
    rows, prev_apex = [], None
    for seg in segments:
        seg_t, seg_z, seg_vz = t[seg], z[seg], vz[seg]
        apex = seg_z.max()
        if prev_apex is not None and prev_apex > GROUND_TOL + 0.02:
            e_meas = float(np.clip((apex / prev_apex) ** 0.5, 0.0, 1.5))
            v0_launch = float((2.0 * max(apex - GROUND_TOL, 1e-9)) ** 0.5)
            t0 = seg_t[0]
            picks = np.linspace(0, len(seg) - 1, min(n_sample, len(seg))).astype(int)
            for p in picks:
                rows.append({"t": seg_t[p] - t0, "z": seg_z[p], "vz": seg_vz[p],
                            "v0": abs(v0_launch), "e": e_meas})
        prev_apex = apex
    return pd.DataFrame(rows, columns=LEGGED_FEATURES)


def get_post_bounce_segment(t: np.ndarray, z_raw: np.ndarray, vz: np.ndarray, which: int = 1):
    """A genuine post-bounce segment (which=1 is the first one, right after
    the first bounce) for held-out testing - NEVER segment 0. Returns
    (seg_t_relative, seg_z, seg_vz, v0, e) for the whole segment at full
    per-timestep resolution."""
    z, segments = split_flight_segments(t, z_raw, vz)
    if which >= len(segments) or which < 1:
        raise ValueError(f"need a post-bounce segment (which>=1); only {len(segments)} segments found")
    seg = segments[which]
    prev_apex = z[segments[which - 1]].max()
    apex = z[seg].max()
    e = float(np.clip((apex / prev_apex) ** 0.5, 0.0, 1.5))
    v0 = float((2.0 * apex) ** 0.5)
    seg_t = t[seg] - t[seg][0]
    return seg_t, z[seg], vz[seg], v0, e
