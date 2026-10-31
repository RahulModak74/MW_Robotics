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
