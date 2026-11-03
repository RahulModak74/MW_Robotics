"""
robot_motion.py
================
Turns a user interaction (a mouse click target) into an animation: a
sequence of frames decoded from one of the six trained MW manifolds, using
the SAME straight-line latent-interpolation mechanism measured in
mujoco_experiments/02_forecaster_check.py and 03/04 (encode a start state,
encode a target state, interpolate in latent space, decode each point).

Nothing here re-derives physics from scratch for the animation: both the
start and target states passed to each manifold are computed from the
SAME exact closed-form formulas each domain was trained on
(mw_framework.datasets' private _joint_response/_arm_fk/_unicycle_state/
_beam_state/_thermal_state helpers), so what you see animate is the
manifold's own interpolation between two states it recognises, not an
invented trajectory.

HONEST SCOPE: this is a demo/visualisation, not a validated controller.
See full_flow_animation/README.md for exactly what is and is not real.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from mw_framework.datasets import (
    _joint_response, _arm_fk, _unicycle_state, _beam_state, _thermal_state,
    L1, L2, G_NOM,
)


def wrap_angle(theta: float) -> float:
    """Wrap to [-pi, pi]. mobile_robot trained on theta0 ~ Uniform(-pi, pi);
    letting theta accumulate unwrapped across many moves pushes it further
    out of the trained distribution each time - confirmed as the cause of a
    68% positioning error after five chained moves before this fix."""
    return math.atan2(math.sin(theta), math.cos(theta))


def compute_path(det, start_row: dict, target_row: dict, feature_order, n_points: int = 16):
    """Encode start_row and target_row (dicts: feature_name -> value), walk a
    straight line between their latents, decode n_points frames. Returns a
    list of n_points dicts, one per feature, matching mujoco_experiments'
    tested mechanism exactly."""
    norm, vae = det.norm_, det.vae_

    def to_latent(row):
        x = np.array([[row[f] for f in feature_order]])
        xn = norm.transform(x)
        xt = torch.tensor(xn, dtype=torch.float32)
        with torch.no_grad():
            mu, _ = vae.encode(xt)
        return mu[0].detach().clone()

    z0, z1 = to_latent(start_row), to_latent(target_row)
    tt = torch.linspace(0, 1, n_points)[:, None]
    path = (1 - tt) * z0 + tt * z1
    with torch.no_grad():
        phys = vae.to_physical(vae.decode(path))
    frames = []
    for i in range(n_points):
        frames.append({k: float(v[i]) for k, v in phys.items()})
    return frames


# ── Per-behaviour wrappers: build physically-consistent start/target rows
#    from the SAME closed-form formulas each domain trained on, then hand
#    off to compute_path(). ──────────────────────────────────────────────

def joint_rotate(det, current_theta_ref: float, target_theta_ref: float, n_points=16):
    """joint_actuator: settle from the old set-point to a new one."""
    theta0, omega0, alpha0, tau0, e0 = _joint_response(np.array([0.08]), np.array([current_theta_ref]))
    theta1, omega1, alpha1, tau1, e1 = _joint_response(np.array([11.5]), np.array([target_theta_ref]))
    start = {"t": 0.08, "theta": float(theta0[0]), "omega": float(omega0[0]),
            "alpha": float(alpha0[0]), "tau": float(tau0[0]), "energy": float(e0[0])}
    target = {"t": 11.5, "theta": float(theta1[0]), "omega": float(omega1[0]),
             "alpha": float(alpha1[0]), "tau": float(tau1[0]), "energy": float(e1[0])}
    from mw_framework.datasets import JOINT_FEATURES
    return compute_path(det, start, target, JOINT_FEATURES, n_points)


def arm_reach(det, cur_theta1: float, cur_theta2: float, target_x: float, target_y: float, n_points=16):
    """planar_arm: 2-link inverse kinematics for the target, then interpolate
    the resting-state reach (zero joint velocities at both ends)."""
    r = math.hypot(target_x, target_y)
    r = min(r, L1 + L2 - 1e-3)
    r = max(r, abs(L1 - L2) + 1e-3)
    cos_t2 = (r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    t2 = math.acos(cos_t2)
    k1, k2 = L1 + L2 * math.cos(t2), L2 * math.sin(t2)
    t1 = math.atan2(target_y, target_x) - math.atan2(k2, k1)

    x0, y0, xd0, yd0 = _arm_fk(cur_theta1, cur_theta2, 0.0, 0.0)
    x1, y1, xd1, yd1 = _arm_fk(t1, t2, 0.0, 0.0)
    start = {"theta1": cur_theta1, "theta2": cur_theta2, "theta1dot": 0.0, "theta2dot": 0.0,
            "x": float(x0), "y": float(y0), "xdot": 0.0, "ydot": 0.0}
    target = {"theta1": t1, "theta2": t2, "theta1dot": 0.0, "theta2dot": 0.0,
             "x": float(x1), "y": float(y1), "xdot": 0.0, "ydot": 0.0}
    from mw_framework.datasets import ARM_FEATURES
    return compute_path(det, start, target, ARM_FEATURES, n_points), (t1, t2)


def base_move(det, x0: float, y0: float, theta0: float, target_x: float, target_y: float,
             n_points=16, n_segments_each=3):
    """mobile_robot: turn to face the target, then drive straight to it - two
    phases, each CHAINED through real intermediate checkpoints.

    An earlier version used a single pure-pursuit circular arc (the tangent
    arc through the target). That is only valid when the target is roughly
    ahead of the robot; for a target behind or to the side it requires the
    arc's "major" solution, which the formula did not handle, and testing
    it (not assuming it was fine) showed the robot driving to a completely
    wrong point. Turn-then-drive has no such ambiguity for any direction,
    at the cost of being a less smooth, two-step motion rather than one
    graceful curve - a correctness-for-elegance tradeoff made deliberately
    after the arc version was shown to fail."""
    from mw_framework.datasets import MOBILE_FEATURES

    def chain(x0_, y0_, th0_, v, omega, T, n_seg):
        checkpoint_ts = np.linspace(0.02, max(T, 0.05), n_seg + 1)
        xs, ys, ths, xds, yds, thds = _unicycle_state(checkpoint_ts, x0_, y0_, th0_, v, omega)
        frames_ = []
        for i in range(n_seg):
            start = {"x": float(xs[i]), "y": float(ys[i]), "theta": wrap_angle(float(ths[i])), "v": v, "omega": omega,
                    "xdot": float(xds[i]), "ydot": float(yds[i]), "thetadot": float(thds[i])}
            end = {"x": float(xs[i + 1]), "y": float(ys[i + 1]), "theta": wrap_angle(float(ths[i + 1])), "v": v, "omega": omega,
                  "xdot": float(xds[i + 1]), "ydot": float(yds[i + 1]), "thetadot": float(thds[i + 1])}
            seg_frames = compute_path(det, start, end, MOBILE_FEATURES, max(2, n_points // n_seg))
            frames_.extend(seg_frames if i == 0 else seg_frames[1:])
        return frames_, (float(xs[-1]), float(ys[-1]), wrap_angle(float(ths[-1])))

    theta0 = wrap_angle(theta0)        # defend against any already-unwrapped theta in the caller's state
    dx, dy = target_x - x0, target_y - y0
    dist = max(math.hypot(dx, dy), 1e-6)
    target_heading = math.atan2(dy, dx)
    heading_err = math.atan2(math.sin(target_heading - theta0), math.cos(target_heading - theta0))

    frames = []
    x_cur, y_cur, th_cur = x0, y0, theta0
    if abs(heading_err) > 0.08:                     # phase 1: rotate in place to face the target
        omega_turn = max(-1.1, min(heading_err / 1.0, 1.1))
        T_turn = abs(heading_err / omega_turn)
        f, (x_cur, y_cur, th_cur) = chain(x_cur, y_cur, th_cur, 0.0, omega_turn, T_turn, n_segments_each)
        frames.extend(f)
    if dist > 0.05:                                  # phase 2: drive straight to the target
        v = min(0.9, dist / 1.0)
        T_drive = dist / v
        f, (x_cur, y_cur, th_cur) = chain(x_cur, y_cur, th_cur, v, 0.0, T_drive, n_segments_each)
        frames.extend(f if not frames else f[1:])
    return frames, (x_cur, y_cur, th_cur)


def hop(det, launch_height: float = 0.5, e: float = 0.8, n_points=16, n_segments=4):
    """legged_contact: one ballistic hop of the given launch height.
    CHAINED through n_segments real checkpoints spanning launch to landing
    (same reason as base_move: a single hop across nearly the whole flight
    badly undershoots the true apex)."""
    v0 = math.sqrt(max(2.0 * G_NOM * launch_height, 1e-6))
    t_apex = v0 / G_NOM
    t_end = min(1.9 * t_apex, 2.5)

    from mw_framework.datasets import LEGGED_FEATURES
    checkpoint_ts = np.linspace(0.03, t_end, n_segments + 1)
    z_cp = v0 * checkpoint_ts - 0.5 * G_NOM * checkpoint_ts ** 2
    vz_cp = v0 - G_NOM * checkpoint_ts
    frames = []
    for i in range(n_segments):
        start = {"t": float(checkpoint_ts[i]), "z": max(float(z_cp[i]), 0.0), "vz": float(vz_cp[i]), "v0": v0, "e": e}
        target = {"t": float(checkpoint_ts[i + 1]), "z": max(float(z_cp[i + 1]), 0.0),
                 "vz": float(vz_cp[i + 1]), "v0": v0, "e": e}
        seg_frames = compute_path(det, start, target, LEGGED_FEATURES, max(2, n_points // n_segments))
        frames.extend(seg_frames if i == 0 else seg_frames[1:])
    return frames


def lift_payload(det, deflection: float = 0.7, n_points=16):
    """flexible_robot: a payload-induced deflection, settling under damping -
    animates the arm's last segment flexing then recovering. EI_NOM=1.0
    (flexible_robot.py's constant) so M = w_xx exactly, same as training."""
    EI_NOM = 1.0
    x = np.array([0.9])          # near the free (tip) end of the beam
    w0, wt0, wxx0, e0 = _beam_state(x, 0.08, (deflection, 0.15 * deflection))
    w1, wt1, wxx1, e1 = _beam_state(x, 23.0, (deflection, 0.15 * deflection))
    start = {"x": 0.9, "t": 0.08, "w": float(w0[0]), "w_t": float(wt0[0]),
            "w_xx": float(wxx0[0]), "M": EI_NOM * float(wxx0[0]), "energy": float(e0[0])}
    target = {"x": 0.9, "t": 23.0, "w": float(w1[0]), "w_t": float(wt1[0]),
             "w_xx": float(wxx1[0]), "M": EI_NOM * float(wxx1[0]), "energy": float(e1[0])}
    from mw_framework.datasets import ROBOT_FEATURES
    return compute_path(det, start, target, ROBOT_FEATURES, n_points)


def thermal_tick(det, heat_level: float = 0.6, n_points=16):
    """actuator_thermal: a heated state cooling back toward ambient -
    polled periodically to animate the temperature gauge decaying."""
    x = np.array([0.5])
    T0, dTdx0, flux0, ex0 = _thermal_state(x, 0.1, heat_level, 0.2 * heat_level)
    T1, dTdx1, flux1, ex1 = _thermal_state(x, 14.0, heat_level, 0.2 * heat_level)
    start = {"x": 0.5, "t": 0.1, "T": float(T0[0]), "dTdx": float(dTdx0[0]),
            "flux": float(flux0[0]), "excess": 1.0}
    target = {"x": 0.5, "t": 14.0, "T": float(T1[0]), "dTdx": float(dTdx1[0]),
             "flux": float(flux1[0]), "excess": 0.02}
    from mw_framework.datasets import THERMAL_FEATURES
    return compute_path(det, start, target, THERMAL_FEATURES, n_points)
