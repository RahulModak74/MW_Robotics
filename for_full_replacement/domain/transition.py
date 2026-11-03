"""
mw_framework.domains.transition
=================================
A transition manifold over (state, action, next_state) — Plan B's core object.

Unlike walker_domain.py's WalkerPostureVAE (which learns a manifold over STATES
only), this learns the manifold over TRANSITIONS. The physics priors make the
decoder's output lie on the admissible-transition manifold. The joint query
"given s and s', what a?" then becomes a MW-distance minimization problem,
solved by a small CEM over the action block only (6 dims for Walker2d, ~30
for Humanoid) — cheap enough for real-time CPU.

This is the object Plan B replaces a policy network with. There is no learned
controller; the manifold IS the model, and control is a posterior sample.
"""

from __future__ import annotations
from typing import Dict
import torch
from ..core import MWVAE, constraint, bound

# 24 features: 9 state + 6 action + 9 next_state
STATE_FEATURES = ["height", "pitch", "pitch_rate",
                  "thigh", "leg", "foot",
                  "thigh_left", "leg_left", "foot_left"]
ACTION_FEATURES = ["a_thigh", "a_leg", "a_foot",
                   "a_thigh_left", "a_leg_left", "a_foot_left"]
FEATURES = (["s_" + f for f in STATE_FEATURES]
            + ACTION_FEATURES
            + ["sp_" + f for f in STATE_FEATURES])

# Walker2d-v5 physical bounds (from MuJoCo jnt_range / actuator_forcerange)
THIGH_LIM = (-2.61799388, 0.0)
LEG_LIM = (-2.61799388, 0.0)
FOOT_LIM = (-0.78539816, 0.78539816)
A_MAX = 1.0

# One-step reachability priors (loose but anti-teleport)
# Walker2d-v5 dt=0.008; a joint can move at most ~20 rad/s comfortably.
DT = 0.008
JOINT_DELTA_MAX = 20.0 * DT   # 0.16 rad
HEIGHT_DELTA_MAX = 5.0 * DT   # 0.04 m
PITCH_DELTA_MAX = 10.0 * DT   # 0.08 rad


class TransitionVAE(MWVAE):
    """Joint VAE over (s, a, s'). Physics priors: joint limits, torque limits,
    one-step reachability, height/pitch envelope."""

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 128,
                 dropout: float = 0.05, law_tolerance: float = 0.05, **kwargs):
        kwargs.setdefault("n_features", len(FEATURES))
        kwargs.setdefault("feature_names", FEATURES)
        kwargs.setdefault("obs_sigma", [0.05] * len(FEATURES))
        super().__init__(latent_dim=latent_dim, hidden_dim=hidden_dim,
                         dropout=dropout, **kwargs)
        self.law_tolerance = law_tolerance

    def get_config(self):
        cfg = super().get_config()
        cfg.update(law_tolerance=self.law_tolerance)
        return cfg

    def physics(self, x: Dict[str, torch.Tensor]) -> None:
        tol = self.law_tolerance
        # C1 — exact joint limits on s and s'
        for prefix in ("s_", "sp_"):
            for j, (lo, hi) in [("thigh", THIGH_LIM), ("leg", LEG_LIM),
                                ("foot", FOOT_LIM), ("thigh_left", THIGH_LIM),
                                ("leg_left", LEG_LIM), ("foot_left", FOOT_LIM)]:
                bound(f"c1_{prefix}{j}", x[f"{prefix}{j}"], low=lo, high=hi, tolerance=tol)

        # C2 — actuator torque limits
        for name in ACTION_FEATURES:
            bound(f"c2_{name}", x[name], low=-A_MAX, high=A_MAX, tolerance=tol)

        # C3 — one-step kinematic reachability: |s' - s| <= bound (anti-teleport)
        for j in STATE_FEATURES:
            if j in ("height", "pitch", "pitch_rate"):
                dmax = HEIGHT_DELTA_MAX if j == "height" else PITCH_DELTA_MAX
            else:
                dmax = JOINT_DELTA_MAX
            delta = (x[f"sp_{j}"] - x[f"s_{j}"]).abs()
            bound(f"c3_reach_{j}", delta, high=dmax, tolerance=2 * tol)

        # C4 — pitch envelope (soft; the controller's own stable range)
        bound("c4_s_pitch", x["s_pitch"].abs(), high=1.5, tolerance=tol)
        bound("c4_sp_pitch", x["sp_pitch"].abs(), high=1.5, tolerance=tol)
        # C4 — height cannot be negative (no body under the floor)
        bound("c4_s_height_low", x["s_height"], low=0.0, tolerance=tol)
        bound("c4_sp_height_low", x["sp_height"], low=0.0, tolerance=tol)

    @staticmethod
    def residuals(x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Useful summary residuals for diagnostics / failure analysis."""
        return {
            "reach_hip": (x["sp_thigh"] - x["s_thigh"]).abs(),
            "reach_knee": (x["sp_leg"] - x["s_leg"]).abs(),
            "height_delta": (x["sp_height"] - x["s_height"]).abs(),
            "pitch_delta": (x["sp_pitch"] - x["s_pitch"]).abs(),
        }
