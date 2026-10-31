"""
mw_framework.datasets
=====================
Small synthetic datasets generated FROM PHYSICS, so every example runs
offline in minutes. Each returns pandas DataFrames.

WHY synthetic?
    You can only check a detector against ground truth you control.
    Each generator knows the true law, so you can see exactly what MW
    learns, what it catches, and where it fails. Swap in real data later.

Conventions: `normal` = data obeying the physics; `anomaly` = labelled
departures, with an `anomaly_type` column describing the failure.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd



# ─────────────────────────────────────────────────────────────────────────────
# 00  circle - the simplest manifold
# ─────────────────────────────────────────────────────────────────────────────

def circle(n_normal: int = 1000, n_anomaly: int = 200, noise: float = 0.03,
           seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normal points lie on the unit circle. Anomalies sit INSIDE it,
    so every coordinate is within its normal range - only the shape differs."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n_normal)
    r = 1 + rng.normal(0, noise, n_normal)
    normal = pd.DataFrame({"x": r * np.cos(th), "y": r * np.sin(th)})
    th = rng.uniform(0, 2 * np.pi, n_anomaly)
    r = rng.uniform(0.2, 0.7, n_anomaly)
    anom = pd.DataFrame({"x": r * np.cos(th), "y": r * np.sin(th)})
    anom["anomaly_type"] = "inside_circle"
    return normal, anom


def arc(n: int = 1500, span_deg: float = 300.0, noise: float = 0.02,
        seed: int = 0) -> pd.DataFrame:
    """A 300-degree arc: its two ends are close in space, far along the curve."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, np.deg2rad(span_deg), n)
    return pd.DataFrame({"x": np.cos(th) + rng.normal(0, noise, n),
                         "y": np.sin(th) + rng.normal(0, noise, n),
                         "theta": th})


# ─────────────────────────────────────────────────────────────────────────────
# cantilever_beam - Euler-Bernoulli PDE, exact modal solution
#   (flexible-link robot arm / soft-robot segment released from a deflection)
# ─────────────────────────────────────────────────────────────────────────────

ROBOT_FEATURES = ["x", "t", "w", "w_t", "w_xx", "M", "energy"]

BETA = (1.8751040687, 4.6940911330)      # cantilever eigenvalues, beta_n * L, L = 1
OMEGA = tuple(b ** 2 for b in BETA)       # nondimensional natural frequencies
ZETA_BEAM = 0.02                          # modal damping ratio
EI_NOM = 1.0


def _cantilever_sigma(beta: float) -> float:
    return (np.sinh(beta) - np.sin(beta)) / (np.cosh(beta) + np.cos(beta))


_SIGMA = tuple(_cantilever_sigma(b) for b in BETA)


def _mode_shape(x: np.ndarray, n: int):
    """phi_n(x) and its second derivative phi_n''(x) (curvature shape)."""
    b, s = BETA[n], _SIGMA[n]
    phi = np.cosh(b * x) - np.cos(b * x) - s * (np.sinh(b * x) - np.sin(b * x))
    phi_xx = b ** 2 * (np.cosh(b * x) + np.cos(b * x) - s * (np.sinh(b * x) + np.sin(b * x)))
    return phi, phi_xx


def _beam_state(x, t, q0, mode_amp=(1.0, 1.0)):
    """
    Exact state of a cantilever released from rest with initial modal
    amplitudes q0 = (q1_0, q2_0), under linear viscous modal damping.
    mode_amp scales each mode's CONTRIBUTION after release — used to build
    a cheaper/coarser approximation (mode_amp[1] -> 0 drops mode 2 entirely).

    Returns w, w_t, w_xx, energy (all nondimensional) at each (x, t) pair.
    """
    w = np.zeros_like(x, dtype=float)
    w_t = np.zeros_like(x, dtype=float)
    w_xx = np.zeros_like(x, dtype=float)
    energy = np.zeros_like(x, dtype=float)
    for n in range(2):
        om, ze, a0 = OMEGA[n], ZETA_BEAM, q0[n] * mode_amp[n]
        om_d = om * np.sqrt(max(1e-9, 1 - ze ** 2))
        env = np.exp(-ze * om * t)
        q = a0 * env * np.cos(om_d * t)
        q_t = a0 * env * (-ze * om * np.cos(om_d * t) - om_d * np.sin(om_d * t))
        phi, phi_xx = _mode_shape(x, n)
        w += q * phi
        w_t += q_t * phi
        w_xx += q * phi_xx
        # unit modal mass -> generalised KE + PE for this mode
        energy += 0.5 * q_t ** 2 + 0.5 * om ** 2 * q ** 2
    return w, w_t, w_xx, energy


def cantilever_beam(n_normal: int = 4000, n_per_anomaly: int = 200,
                    t_max: float = 25.0, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A flexible-link robot arm released from an initial deflection, decaying
    under viscous damping — the exact 2-mode solution of the Euler-Bernoulli
    beam PDE. See mw_framework.domains.flexible_robot for the governing
    equation and the laws it must obey.

    Anomalies (all realistic flexible-link / soft-robot faults):
        stiffness_loss    local crack: curvature sensor reads higher than
                          the moment sensor (same load) implies -> breaks C1
        moment_glitch     moment-estimate sensor fault, curvature unaffected
        no_decay          energy fails to decay (stuck damper / loose joint) -
                          a t=0 state relabelled as a late-time reading
    """
    rng = np.random.default_rng(seed)

    def make(n, t_fixed=None):
        x = rng.uniform(0, 1, n)
        t = rng.uniform(0.1, t_max, n) if t_fixed is None else np.full(n, float(t_fixed))
        q0 = np.stack([rng.uniform(0.6, 1.2, n), rng.uniform(-0.3, 0.3, n)], axis=1)
        w, w_t, w_xx, e = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            wi, wti, wxxi, ei = _beam_state(np.array([x[i]]), t[i], (q0[i, 0], q0[i, 1]))
            w[i], w_t[i], w_xx[i], e[i] = wi[0], wti[0], wxxi[0], ei[0]
        M = EI_NOM * w_xx
        e0 = np.array([_beam_state(np.array([x[i]]), 0.0, (q0[i, 0], q0[i, 1]))[3][0]
                       for i in range(n)])
        X = np.stack([x, t, w, w_t, w_xx, M, e / np.maximum(e0, 1e-9)], axis=1)
        X[:, 2:] += rng.normal(0, 0.01, (n, 5))
        return pd.DataFrame(X, columns=ROBOT_FEATURES)

    normal = make(n_normal)

    a = make(n_per_anomaly)
    a["w_xx"] *= 1.0 + rng.uniform(0.3, 0.6, n_per_anomaly)       # curvature reads too high
    a["anomaly_type"] = "stiffness_loss"

    b = make(n_per_anomaly)
    b["M"] += rng.choice([-1, 1], n_per_anomaly) * rng.uniform(0.3, 0.5, n_per_anomaly)
    b["anomaly_type"] = "moment_glitch"

    c = make(n_per_anomaly, t_fixed=0.1)
    c["t"] = t_max * 0.8                                         # late time, undamped energy
    c["anomaly_type"] = "no_decay"

    return normal, pd.concat([a, b, c], ignore_index=True)


def cantilever_beam_coarse(n: int = 1000, fidelity: float = 0.0,
                           t_max: float = 25.0, seed: int = 1) -> pd.DataFrame:
    """
    A cheap approximate simulator for the SAME beam, for the validator test:
    `fidelity` in [0, 1] interpolates from mode-1-only + a crude Euler
    time-step phase error (0, worst) to the exact 2-mode solution (1, exact).
    This stands in for a coarse FEM / reduced-order robot simulator whose
    error is dialled by a knob (mesh size, timestep, mode truncation).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    t = rng.uniform(0.1, t_max, n)
    q0 = np.stack([rng.uniform(0.6, 1.2, n), rng.uniform(-0.3, 0.3, n)], axis=1)

    mode2_amp = fidelity                      # 0 = mode 2 dropped entirely
    phase_err = (1 - fidelity) * 0.35          # crude integrator mistunes frequency

    w, w_t, w_xx, e, e0 = (np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n))
    for i in range(n):
        om1, ze = OMEGA[0] * (1 + phase_err), ZETA_BEAM
        om_d = om1 * np.sqrt(max(1e-9, 1 - ze ** 2))

        def mode1(tt):
            env = np.exp(-ze * om1 * tt)
            q = q0[i, 0] * env * np.cos(om_d * tt)
            q_t = q0[i, 0] * env * (-ze * om1 * np.cos(om_d * tt) - om_d * np.sin(om_d * tt))
            return q, q_t, 0.5 * q_t ** 2 + 0.5 * om1 ** 2 * q ** 2

        q, q_t, e1 = mode1(t[i])
        phi1, phi1_xx = _mode_shape(np.array([x[i]]), 0)
        w1, w1_t, w1_xx = q * phi1[0], q_t * phi1[0], q * phi1_xx[0]
        w2, w2_t, w2_xx, e2 = _beam_state(np.array([x[i]]), t[i], (0.0, q0[i, 1]), mode_amp=(0.0, mode2_amp))
        w[i], w_t[i], w_xx[i] = w1 + w2[0], w1_t + w2_t[0], w1_xx + w2_xx[0]
        e[i] = e1 + e2[0]

        # energy(0) for THIS same (mistuned) simulator — same normalisation convention
        # as cantilever_beam's e/e0, so fidelity 1.0 reduces to an identical distribution.
        _, _, e1_0 = mode1(0.0)
        _, _, _, e2_0 = _beam_state(np.array([x[i]]), 0.0, (0.0, q0[i, 1]), mode_amp=(0.0, mode2_amp))
        e0[i] = e1_0 + e2_0[0]

    M = EI_NOM * w_xx
    X = np.stack([x, t, w, w_t, w_xx, M, e / np.maximum(e0, 1e-9)], axis=1)
    df = pd.DataFrame(X, columns=ROBOT_FEATURES)
    df["fidelity"] = fidelity
    return df


# ─────────────────────────────────────────────────────────────────────────────
# joint_actuator - a single PD-controlled robot joint (damped 2nd-order ODE)
# ─────────────────────────────────────────────────────────────────────────────
JOINT_FEATURES = ["t", "theta", "omega", "alpha", "tau", "energy"]


def _joint_response(t, theta_ref, I=1.0, b=0.4, k=1.0):
    """Exact step response of I*theta'' + b*theta' + k*theta = k*theta_ref,
    released from rest at theta=0 (underdamped: b < 2*sqrt(I*k)). theta(t) is
    the textbook underdamped step response; omega, alpha are its exact first
    and second derivatives (closed form, not finite differences)."""
    wn = np.sqrt(k / I)
    zeta = b / (2 * np.sqrt(I * k))
    wd = wn * np.sqrt(max(1e-9, 1 - zeta ** 2))
    r = zeta / np.sqrt(1 - zeta ** 2)
    env = np.exp(-zeta * wn * t)
    c, s = np.cos(wd * t), np.sin(wd * t)
    theta = theta_ref * (1 - env * (c + r * s))
    # d/dt[env*(c + r*s)] = env*(-zeta*wn*(c+r*s) + (-wd*s + r*wd*c))
    d1 = env * (-zeta * wn * (c + r * s) + (-wd * s + r * wd * c))
    omega = -theta_ref * d1
    # d/dt of d1
    d2 = (-zeta * wn * d1
         + env * (-wd * (-zeta * wn * s + wd * c) + r * wd * (-zeta * wn * c - wd * s)))
    alpha = -theta_ref * d2
    tau = I * alpha + b * omega + k * theta           # equation-of-motion identity, by construction
    energy = 0.5 * I * omega ** 2 + 0.5 * k * theta ** 2
    return theta, omega, alpha, tau, energy


def joint_actuator(n_normal: int = 4000, n_per_anomaly: int = 200,
                   t_max: float = 12.0, seed: int = 0):
    """
    A single PD-controlled robot joint (motor + link), modelled as a damped
    second-order linear system driven to a reference angle - the ODE every
    joint controller's inner loop assumes. See mw_framework.domains.joint_actuator.

    Anomalies (realistic actuator faults):
        friction_increase   bearing wear: reported omega is lower than the
                            torque sensor and geometry imply -> breaks C1
        torque_sensor_glitch  torque sensor fault, dynamics unaffected
        no_settle           energy fails to decay (stuck/oscillating joint) -
                            a t=0 state relabelled as a late-time reading
    """
    rng = np.random.default_rng(seed)

    def make(n, t_fixed=None):
        t = rng.uniform(0.1, t_max, n) if t_fixed is None else np.full(n, float(t_fixed))
        theta_ref = rng.uniform(0.6, 1.4, n)
        theta, omega, alpha, tau, e = (np.zeros(n) for _ in range(5))
        for i in range(n):
            theta[i], omega[i], alpha[i], tau[i], e[i] = _joint_response(t[i], theta_ref[i])
        X = np.stack([t, theta, omega, alpha, tau, e], axis=1)
        X[:, 1:] += rng.normal(0, 0.01, (n, 5))
        return pd.DataFrame(X, columns=JOINT_FEATURES)

    normal = make(n_normal)

    a = make(n_per_anomaly)
    a["omega"] *= 1.0 - rng.uniform(0.3, 0.55, n_per_anomaly)
    a["anomaly_type"] = "friction_increase"

    b = make(n_per_anomaly)
    b["tau"] += rng.choice([-1, 1], n_per_anomaly) * rng.uniform(0.3, 0.5, n_per_anomaly)
    b["anomaly_type"] = "torque_sensor_glitch"

    c = make(n_per_anomaly, t_fixed=0.1)
    c["t"] = t_max * 0.8
    c["anomaly_type"] = "no_settle"

    return normal, pd.concat([a, b, c], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# planar_arm - 2-link planar manipulator: forward kinematics + Jacobian
# ─────────────────────────────────────────────────────────────────────────────
ARM_FEATURES = ["theta1", "theta2", "theta1dot", "theta2dot", "x", "y", "xdot", "ydot"]
L1, L2 = 1.0, 0.8


def _arm_fk(theta1, theta2, theta1dot, theta2dot):
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s12, c12 = np.sin(theta1 + theta2), np.cos(theta1 + theta2)
    xdot = -L1 * s1 * theta1dot - L2 * s12 * (theta1dot + theta2dot)
    ydot = L1 * c1 * theta1dot + L2 * c12 * (theta1dot + theta2dot)
    return x, y, xdot, ydot


def planar_arm(n_normal: int = 4000, n_per_anomaly: int = 200, seed: int = 0):
    """
    A 2-link planar robot arm: joint encoders report (theta1, theta2) and
    their rates; a separate sensor (vision, or an IMU on the end-effector)
    reports end-effector position and velocity (x, y, xdot, ydot). Forward
    kinematics and the velocity Jacobian say what the second sensor MUST
    read if the first is right - two independent measurements of the same
    physical configuration. See mw_framework.domains.planar_arm.

    Anomalies (realistic arm faults):
        encoder_drift    joint-1 encoder biased -> FK position disagrees
                        with vision/end-effector sensor
        link_bent       link 2 effectively shorter/longer than nominal
                        (a bent or damaged link) -> FK position disagrees
        velocity_glitch  end-effector velocity sensor fault, position fine
    """
    rng = np.random.default_rng(seed)

    def make(n):
        th1 = rng.uniform(-1.2, 1.2, n)
        th2 = rng.uniform(-2.0, 2.0, n)
        th1d = rng.uniform(-1.0, 1.0, n)
        th2d = rng.uniform(-1.0, 1.0, n)
        x, y, xd, yd = _arm_fk(th1, th2, th1d, th2d)
        X = np.stack([th1, th2, th1d, th2d, x, y, xd, yd], axis=1)
        X += rng.normal(0, 0.01, X.shape)
        return pd.DataFrame(X, columns=ARM_FEATURES)

    normal = make(n_normal)

    a = make(n_per_anomaly)
    bias = rng.uniform(0.15, 0.3, n_per_anomaly) * rng.choice([-1, 1], n_per_anomaly)
    x2, y2, _, _ = _arm_fk(a["theta1"].values + bias, a["theta2"].values,
                          a["theta1dot"].values, a["theta2dot"].values)
    a["x"], a["y"] = x2, y2
    a["anomaly_type"] = "encoder_drift"

    b = make(n_per_anomaly)
    L2_true = L2 * (1 + rng.uniform(0.15, 0.3, n_per_anomaly) * rng.choice([-1, 1], n_per_anomaly))
    th1, th2 = b["theta1"].values, b["theta2"].values
    s1, c1 = np.sin(th1), np.cos(th1)
    s12, c12 = np.sin(th1 + th2), np.cos(th1 + th2)
    b["x"] = L1 * c1 + L2_true * c12
    b["y"] = L1 * s1 + L2_true * s12
    b["anomaly_type"] = "link_bent"

    c = make(n_per_anomaly)
    c["xdot"] += rng.choice([-1, 1], n_per_anomaly) * rng.uniform(0.3, 0.5, n_per_anomaly)
    c["ydot"] += rng.choice([-1, 1], n_per_anomaly) * rng.uniform(0.3, 0.5, n_per_anomaly)
    c["anomaly_type"] = "velocity_glitch"

    return normal, pd.concat([a, b, c], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# mobile_robot - differential-drive / unicycle kinematics (nonholonomic ODE)
# ─────────────────────────────────────────────────────────────────────────────
MOBILE_FEATURES = ["x", "y", "theta", "v", "omega", "xdot", "ydot", "thetadot"]
V_MAX = 2.0


def _unicycle_state(t, x0, y0, theta0, v, omega):
    """Exact closed-form pose after time t at constant (v, omega) -
    a circular arc (or straight line if omega == 0)."""
    theta = theta0 + omega * t
    small = np.abs(omega) < 1e-6
    om_safe = np.where(small, 1.0, omega)
    x = np.where(small, x0 + v * t * np.cos(theta0),
                x0 + (v / om_safe) * (np.sin(theta) - np.sin(theta0)))
    y = np.where(small, y0 + v * t * np.sin(theta0),
                y0 - (v / om_safe) * (np.cos(theta) - np.cos(theta0)))
    xdot = v * np.cos(theta)
    ydot = v * np.sin(theta)
    thetadot = np.full_like(t, 0.0) + omega
    return x, y, theta, xdot, ydot, thetadot


def mobile_robot(n_normal: int = 4000, n_per_anomaly: int = 200,
                 t_max: float = 4.0, seed: int = 0):
    """
    A differential-drive mobile robot: wheel odometry reports (v, omega);
    a second source (IMU gyro + visual odometry) reports the instantaneous
    (xdot, ydot, thetadot). For a wheeled robot that is not slipping, the
    body cannot move sideways - the classic nonholonomic constraint - and
    the heading rate must match the commanded turn rate. See
    mw_framework.domains.mobile_robot.

    Anomalies (realistic mobile-robot faults):
        wheel_slip      a lateral velocity component appears (ice, gravel,
                        a stuck caster) -> breaks the nonholonomic constraint
        gyro_glitch     heading-rate sensor fault, motion otherwise normal
        overspeed       commanded speed exceeds the drivetrain's physical limit
    """
    rng = np.random.default_rng(seed)

    def make(n):
        t = rng.uniform(0.05, t_max, n)
        x0 = rng.uniform(-1, 1, n)
        y0 = rng.uniform(-1, 1, n)
        theta0 = rng.uniform(-np.pi, np.pi, n)
        v = rng.uniform(-1.0, 1.0, n)
        omega = rng.uniform(-1.2, 1.2, n)
        x, y, theta, xd, yd, thd = _unicycle_state(t, x0, y0, theta0, v, omega)
        X = np.stack([x, y, theta, v, omega, xd, yd, thd], axis=1)
        X += rng.normal(0, 0.01, X.shape)
        return pd.DataFrame(X, columns=MOBILE_FEATURES)

    normal = make(n_normal)

    a = make(n_per_anomaly)
    lat = rng.uniform(0.3, 0.6, n_per_anomaly) * rng.choice([-1, 1], n_per_anomaly)
    a["xdot"] += -lat * np.sin(a["theta"].values)
    a["ydot"] += lat * np.cos(a["theta"].values)
    a["anomaly_type"] = "wheel_slip"

    b = make(n_per_anomaly)
    b["thetadot"] += rng.choice([-1, 1], n_per_anomaly) * rng.uniform(0.3, 0.5, n_per_anomaly)
    b["anomaly_type"] = "gyro_glitch"

    c = make(n_per_anomaly)
    extra = rng.uniform(2.1, 2.8, n_per_anomaly)   # guarantees |v_new| > V_MAX=2.0 regardless of original |v| in [0,1]
    c["v"] = np.sign(c["v"].values) * (np.abs(c["v"].values) + extra)
    c["xdot"] = c["v"].values * np.cos(c["theta"].values)
    c["ydot"] = c["v"].values * np.sin(c["theta"].values)
    c["anomaly_type"] = "overspeed"

    return normal, pd.concat([a, b, c], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# legged_contact - ballistic flight phase of a hopping/legged robot (+ bounce)
# ─────────────────────────────────────────────────────────────────────────────
LEGGED_FEATURES = ["t", "z", "vz", "v0", "e"]
G_NOM = 1.0   # nondimensional gravity (matches the beam/mobile conventions: L=1-style scaling)


def legged_contact(n_normal: int = 4000, n_per_anomaly: int = 200, seed: int = 0):
    """
    The ballistic flight phase of a hopping/legged robot's foot (or body, in
    a SLIP-model abstraction) between ground contacts: z(t) = v0*t - 0.5*g*t^2,
    vz(t) = v0 - g*t, exact projectile motion, nondimensional g=1. `e` is the
    coefficient of restitution of the PREVIOUS bounce (0 < e <= 1) carried as
    a feature. See mw_framework.domains.legged_contact.

    Anomalies (realistic legged-robot faults):
        ground_penetration  z < 0 - the foot/body appears to pass through
                            the ground (sensor fault or control failure)
        energy_gain         vz exceeds what energy conservation allows for
                            the given height and launch speed (an actuator
                            "cheating" during ballistic flight - impossible)
        bad_restitution     e outside (0, 1] - the reported bounce would add
                            energy, which no passive contact can do
    """
    rng = np.random.default_rng(seed)

    def make(n):
        v0 = rng.uniform(0.3, 1.5, n)
        t_apex = v0 / G_NOM
        t = rng.uniform(0.0, 1.95, n) * t_apex        # stay in flight (z stays positive)
        z = v0 * t - 0.5 * G_NOM * t ** 2
        vz = v0 - G_NOM * t
        e = rng.uniform(0.6, 0.98, n)
        X = np.stack([t, z, vz, v0, e], axis=1)
        X[:, :3] += rng.normal(0, 0.005, (n, 3))
        return pd.DataFrame(X, columns=LEGGED_FEATURES)

    normal = make(n_normal)

    a = make(n_per_anomaly)
    a["z"] = -rng.uniform(0.05, 0.3, n_per_anomaly)
    a["anomaly_type"] = "ground_penetration"

    b = make(n_per_anomaly)
    b["vz"] += rng.uniform(0.4, 0.8, n_per_anomaly) * np.sign(rng.standard_normal(n_per_anomaly))
    # keep z physical (>=0) so only the energy law is broken, not the ground bound
    b["z"] = b["z"].clip(lower=0.01)
    b["anomaly_type"] = "energy_gain"

    c = make(n_per_anomaly)
    c["e"] = rng.uniform(1.05, 1.4, n_per_anomaly)
    c["anomaly_type"] = "bad_restitution"

    return normal, pd.concat([a, b, c], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# actuator_thermal - 1D heat diffusion in a motor winding (insulated-rod PDE)
# ─────────────────────────────────────────────────────────────────────────────
THERMAL_FEATURES = ["x", "t", "T", "dTdx", "flux", "excess"]
ALPHA_TH = 0.05     # nondimensional thermal diffusivity
T_AMB = 0.0         # ambient, nondimensional baseline
K_COND = 1.0        # nondimensional conductivity (Fourier's law constant)
T_MAX = 3.0         # over-temperature shutdown limit; normal data peaks lower


def _thermal_state(x, t, A1, A2):
    """Exact 2-mode cosine-series solution of the 1D heat equation on [0,1]
    with insulated ends (dT/dx=0 at x=0,1): T = T_amb + sum A_n cos(n pi x) exp(-alpha (n pi)^2 t).
    A localised hotspot (uneven winding current) is A1, A2 != 0 at t=0, decaying as heat spreads."""
    w1, w2 = np.pi, 2 * np.pi
    e1 = np.exp(-ALPHA_TH * w1 ** 2 * t)
    e2 = np.exp(-ALPHA_TH * w2 ** 2 * t)
    T = T_AMB + A1 * np.cos(w1 * x) * e1 + A2 * np.cos(w2 * x) * e2
    dTdx = -A1 * w1 * np.sin(w1 * x) * e1 - A2 * w2 * np.sin(w2 * x) * e2
    flux = -K_COND * dTdx                       # Fourier's law
    excess0 = np.abs(A1) + np.abs(A2)           # excess temp envelope basis at t=0 (worst case over x)
    return T, dTdx, flux, excess0


def actuator_thermal(n_normal: int = 4000, n_per_anomaly: int = 200,
                     t_max: float = 15.0, seed: int = 0):
    """
    Temperature along a motor winding / actuator housing, modelled as 1D
    heat diffusion from an initial hotspot (uneven current distribution),
    decaying as it spreads - the exact 2-mode cosine-series solution of the
    insulated-rod heat equation. See mw_framework.domains.actuator_thermal.

    Anomalies (realistic actuator thermal faults):
        hotspot_short    a partial winding short: the curvature sensor
                         (dT/dx) reads a sharper gradient than the
                         independent flux estimate implies -> breaks C1
        flux_sensor_glitch  flux-sensor fault, temperature field unaffected
        cooling_failure  temperature fails to decay (fan/heatsink failure) -
                         a t=0 state relabelled as a late-time reading
    """
    rng = np.random.default_rng(seed)

    def make(n, t_fixed=None):
        x = rng.uniform(0, 1, n)
        t = rng.uniform(0.1, t_max, n) if t_fixed is None else np.full(n, float(t_fixed))
        A1 = rng.uniform(0.4, 1.0, n)
        A2 = rng.uniform(-0.3, 0.3, n)
        T, dTdx, flux, e0 = _thermal_state(x, t, A1, A2)
        excess = (np.abs(A1 * np.cos(np.pi * x) * np.exp(-ALPHA_TH * np.pi ** 2 * t))
                 + np.abs(A2 * np.cos(2 * np.pi * x) * np.exp(-ALPHA_TH * (2 * np.pi) ** 2 * t))) / e0
        X = np.stack([x, t, T, dTdx, flux, excess], axis=1)
        X[:, 2:] += rng.normal(0, 0.01, (n, 4))
        return pd.DataFrame(X, columns=THERMAL_FEATURES)

    normal = make(n_normal)

    a = make(n_per_anomaly)
    a["dTdx"] *= 1.0 + rng.uniform(0.4, 0.7, n_per_anomaly)
    a["anomaly_type"] = "hotspot_short"

    b = make(n_per_anomaly)
    b["flux"] += rng.choice([-1, 1], n_per_anomaly) * rng.uniform(0.3, 0.5, n_per_anomaly)
    b["anomaly_type"] = "flux_sensor_glitch"

    c = make(n_per_anomaly, t_fixed=0.1)
    c["t"] = t_max * 0.85
    c["anomaly_type"] = "cooling_failure"

    return normal, pd.concat([a, b, c], ignore_index=True)
