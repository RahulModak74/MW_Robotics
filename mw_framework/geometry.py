"""
mw_framework.geometry
=====================
The Riemannian (and Lorentzian) geometry of a learned MW manifold.

A trained decoder f: z -> x is a map from a small latent space onto a
curved surface in sensor space. Geometry tells you how to measure on it.

PULLBACK METRIC
    G(z) = J(z)^T  W  J(z),     J = df/dz,  W = diag(1 / obs_sigma^2)
    A small step dz in latent space moves the sensors by J dz.
    Its length, in noise units, is sqrt(dz^T G dz).
    WHY: latent coordinates are arbitrary; G turns them into physics units.

MAGNIFICATION   sqrt(det G(z))
    How much a unit latent area is stretched on the real manifold.
    Large where the sensors change quickly (stress regions, transitions).

GEODESIC
    The shortest path ON the manifold between two states.
    Straight lines in latent space are NOT shortest in sensor space.
    WHY: the geodesic length is the physically honest "how different are
    these two operating states". The path itself is the most plausible
    way the system could move from one to the other.

LORENTZIAN CONE (arrow of time)
    Flip the sign of ONE coordinate - time - and you get a light cone:
        ds^2 = -(c dt)^2 + |dx|_W^2
    timelike  (ds^2 < 0): change is slow enough to be physical
    spacelike (ds^2 > 0): change is too fast - impossible for this physics
    Add an ARROW for irreversible quantities (battery health only falls):
    an increase beyond noise is a time-reversal - physically forbidden.
    WHY: a Riemannian manifold says WHERE a state may be. A Lorentzian
    cone also says HOW FAST and IN WHICH DIRECTION it may move.
    That catches tampering and abuse that look perfectly normal snapshot
    by snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.func import jacrev, vmap

# All functions accept tensors on any device, compute on the model's device,
# and return results on the caller's device.


def _dev(vae) -> torch.device:
    return vae.obs_sigma.device


# ─────────────────────────────────────────────────────────────────────────────
# Riemannian part
# ─────────────────────────────────────────────────────────────────────────────

def _decoder(vae):
    def f(z):
        return vae.decode(z.unsqueeze(0)).squeeze(0) / vae.obs_sigma
    return f


def decoder_jacobian(vae, z: torch.Tensor) -> torch.Tensor:
    """J(z) in noise-scaled units. z: (B, d) -> (B, n, d)."""
    vae.eval()
    src = z.device
    return vmap(jacrev(_decoder(vae)))(z.to(_dev(vae))).to(src)


def pullback_metric(vae, z: torch.Tensor) -> torch.Tensor:
    """G(z) = J^T W J.  z: (B, d) -> (B, d, d)."""
    J = decoder_jacobian(vae, z)
    return J.transpose(1, 2) @ J


def magnification(vae, z: torch.Tensor) -> torch.Tensor:
    """sqrt(det G(z)): local area stretch of the manifold. (B,)"""
    G = pullback_metric(vae, z)
    return torch.sqrt(torch.clamp(torch.linalg.det(G), min=0.0))


def curve_length(vae, path: torch.Tensor) -> torch.Tensor:
    """Length of a discrete latent path (T, d) measured ON the manifold."""
    x = vae.decode(path.to(_dev(vae))) / vae.obs_sigma
    return (x[1:] - x[:-1]).norm(dim=-1).sum().to(path.device)


def geodesic(vae, z0: torch.Tensor, z1: torch.Tensor, n_points: int = 24,
             steps: int = 300, lr: float = 0.02) -> Tuple[torch.Tensor, float, float]:
    """
    Shortest path on the manifold between latent points z0 and z1.
    Minimises the discrete curve ENERGY  sum |f(z_{i+1}) - f(z_i)|^2
    (energy minimisers are constant-speed geodesics).

    CAVEAT: a decoder also produces outputs where it never saw data. If
    latent_dim is larger than the true dimension of the system, those
    invented regions can bridge real gaps, and the geodesic will take the
    shortcut. Use latent_dim close to the intrinsic dimension, and check
    every path with path_support().

    Returns (path (n_points, d), geodesic_length, straight_line_length).
    """
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    src = z0.device
    z0, z1 = z0.to(_dev(vae)), z1.to(_dev(vae))
    t = torch.linspace(0, 1, n_points, device=z0.device)[:, None]
    straight = (1 - t) * z0 + t * z1
    inner = straight[1:-1].clone().requires_grad_(True)
    opt = torch.optim.Adam([inner], lr=lr)
    for _ in range(steps):
        path = torch.cat([z0[None], inner, z1[None]])
        x = vae.decode(path) / vae.obs_sigma
        energy = ((x[1:] - x[:-1]) ** 2).sum()
        opt.zero_grad()
        energy.backward()
        opt.step()
    for p in vae.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        path = torch.cat([z0[None], inner.detach(), z1[None]])
        return path.to(src), float(curve_length(vae, path)), float(curve_length(vae, straight))


def path_support(vae, path: torch.Tensor, X_data: torch.Tensor) -> torch.Tensor:
    """
    For each point on a latent path: distance (noise units) from its decoded
    state to the NEAREST real training reading. Values of a few units are
    fine (sensor noise). Large values mean the path runs through states the
    system was never seen in - the decoder invented them.
    """
    with torch.no_grad():
        xp = vae.decode(path.to(_dev(vae))) / vae.obs_sigma
        xd = X_data.to(_dev(vae)) / vae.obs_sigma
        return torch.cdist(xp, xd).min(dim=1).values.to(path.device)


def geodesic_distance(vae, x0: torch.Tensor, x1: torch.Tensor, **kw) -> float:
    """Encode two (normalised) readings, return the geodesic length between them."""
    with torch.no_grad():
        z0 = vae.encode(x0[None])[0][0]
        z1 = vae.encode(x1[None])[0][0]
    return geodesic(vae, z0, z1, **kw)[1]


# ─────────────────────────────────────────────────────────────────────────────
# Lorentzian part: the arrow of time
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConeVerdict:
    interval: float          # ds^2 = -(c dt)^2 + |dx|_W^2
    speed: float             # |dx|_W / dt
    label: str               # TIMELIKE | SPACELIKE | ARROW_VIOLATION | NO_TIME_STEP


class LorentzianCone:
    """
    Minkowski-style metric on (time, state):
        ds^2 = -(c * dt)^2 + sum_i w_i * dx_i^2

    Parameters
    ----------
    state_features : names of the space-like coordinates
    weights        : w_i = 1 / scale_i^2  (puts every feature in comparable units)
    c              : the "speed of light" - max physical rate of change.
                     Use .fit() to learn it from normal trajectories.
    arrow          : {feature: -1 or +1} irreversible quantities and the only
                     direction they may move (-1: may only fall, e.g. SOH).
    arrow_tol      : noise allowance before an arrow violation is called.
    """

    def __init__(self, state_features: Sequence[str], weights: Optional[Sequence[float]] = None,
                 c: float = 1.0, arrow: Optional[Dict[str, int]] = None,
                 arrow_tol: Optional[Dict[str, float]] = None):
        self.state_features = list(state_features)
        self.w = np.asarray(weights if weights is not None else [1.0] * len(state_features),
                            dtype=np.float64)
        self.c = float(c)
        self.arrow = arrow or {}
        self.arrow_tol = arrow_tol or {}

    def _speed(self, dt, dx):
        return np.sqrt((self.w * dx ** 2).sum(-1)) / np.maximum(dt, 1e-12)

    def fit(self, trajectories: List[np.ndarray], times: List[np.ndarray],
            quantile: float = 0.999, margin: float = 1.5) -> "LorentzianCone":
        """
        Learn c from NORMAL trajectories:  c = margin * quantile(speed).
        trajectories[k]: (T_k, n_state)   times[k]: (T_k,)
        """
        speeds = []
        for X, t in zip(trajectories, times):
            dt, dx = np.diff(t), np.diff(X, axis=0)
            ok = dt > 0
            speeds.append(self._speed(dt[ok], dx[ok]))
        self.c = float(margin * np.quantile(np.concatenate(speeds), quantile))
        return self

    def classify(self, t0: float, x0: np.ndarray, t1: float, x1: np.ndarray) -> ConeVerdict:
        dt = float(t1 - t0)
        dx = np.asarray(x1, dtype=np.float64) - np.asarray(x0, dtype=np.float64)
        if dt <= 0:
            return ConeVerdict(float("nan"), float("inf"), "NO_TIME_STEP")
        ds2 = -(self.c * dt) ** 2 + float((self.w * dx ** 2).sum())
        speed = float(self._speed(dt, dx))
        for f, sign in self.arrow.items():
            i = self.state_features.index(f)
            if sign * dx[i] < -self.arrow_tol.get(f, 0.0):
                return ConeVerdict(ds2, speed, "ARROW_VIOLATION")
        return ConeVerdict(ds2, speed, "TIMELIKE" if ds2 <= 0 else "SPACELIKE")

    def scan(self, times: np.ndarray, X: np.ndarray) -> List[ConeVerdict]:
        """Classify every consecutive step of one trajectory."""
        return [self.classify(times[i], X[i], times[i + 1], X[i + 1])
                for i in range(len(times) - 1)]
