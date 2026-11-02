"""
Plan B (SPEC_with_MW_3.md): the manifold is now over TRANSITIONS (s, a, s'),
not just states - this is what makes it a dynamics manifold instead of a
validity check. Follows walker_domain.py's exact style (kept in this
experiment folder, not mw_framework/domains/, same precedent as
walker_domain.py itself - new/unvalidated domains stay with the experiment
that's testing them, not in the shared library, until proven).

One deliberate scope cut from the spec's section 2.3: this is a single
FLAT joint VAE over all 24 features, not the full fibre-bundle
(ManifoldAlgebra state->action->next_state) construction. The gating
experiment in section 8 only needs det.mw_distance() on the concatenated
(s, a, s') row - which a flat VAE gives directly - so the fibre-bundle
refinement is deferred until/unless the gating experiment says this
direction is worth the extra infrastructure.

FEATURES (24): 9 state + 6 action (normalized torques, [-1,1]) + 9 next-state.
"""
from __future__ import annotations
from typing import Dict
import torch
from mw_framework.core import MWVAE, bound

STATE_FEATURES = ["height", "pitch", "pitch_rate", "thigh", "leg", "foot",
                  "thigh_left", "leg_left", "foot_left"]
ACTION_FEATURES = ["tau_thigh", "tau_leg", "tau_foot",
                   "tau_thigh_left", "tau_leg_left", "tau_foot_left"]
NEXT_FEATURES = [f"{f}_next" for f in STATE_FEATURES]
FEATURES = STATE_FEATURES + ACTION_FEATURES + NEXT_FEATURES

# exact joint limits, same as walker_domain.py (confirmed via model.jnt_range)
THIGH_LO, THIGH_HI = -2.61799388, 0.0
LEG_LO, LEG_HI = -2.61799388, 0.0
FOOT_LO, FOOT_HI = -0.78539816, 0.78539816
DELTA_MAX = 0.25   # rad/step soft reachability bound - generous, catches teleports


class TransitionVAE(MWVAE):
    def __init__(self, latent_dim: int = 10, hidden_dim: int = 128,
                 dropout: float = 0.05, law_tolerance: float = 0.05, **kwargs):
        kwargs.setdefault("n_features", len(FEATURES))
        kwargs.setdefault("feature_names", FEATURES)
        kwargs.setdefault("obs_sigma", [0.05] * len(FEATURES))
        super().__init__(latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.law_tolerance = law_tolerance

    def get_config(self):
        cfg = super().get_config()
        cfg.update(law_tolerance=self.law_tolerance)
        return cfg

    def physics(self, x: Dict[str, torch.Tensor]) -> None:
        tol = self.law_tolerance
        # C3 (spec 2.1): exact joint limits, on BOTH s and s' - real hardware
        # facts, not approximations
        for side in ("", "_next"):
            bound(f"c1_thigh{side}", x[f"thigh{side}"], low=THIGH_LO, high=THIGH_HI, tolerance=tol)
            bound(f"c1_leg{side}", x[f"leg{side}"], low=LEG_LO, high=LEG_HI, tolerance=tol)
            bound(f"c1_foot{side}", x[f"foot{side}"], low=FOOT_LO, high=FOOT_HI, tolerance=tol)
            bound(f"c1_thigh_left{side}", x[f"thigh_left{side}"], low=THIGH_LO, high=THIGH_HI, tolerance=tol)
            bound(f"c1_leg_left{side}", x[f"leg_left{side}"], low=LEG_LO, high=LEG_HI, tolerance=tol)
            bound(f"c1_foot_left{side}", x[f"foot_left{side}"], low=FOOT_LO, high=FOOT_HI, tolerance=tol)
            bound(f"c2_pitch_band{side}", x[f"pitch{side}"], low=-1.0, high=1.0, tolerance=tol)
            bound(f"c2_height_band{side}", x[f"height{side}"], low=0.8, high=2.0, tolerance=tol)

        # C2 (spec 2.1): actuator limits - Walker2d's normalized torque range
        for a in ACTION_FEATURES:
            bound(f"c3_{a}_limit", x[a], low=-1.0, high=1.0, tolerance=tol)

        # C1 (spec 2.1): kinematic reachability - s' can't "teleport" from s
        # in one dt. Soft bound on the one-step delta, not an exact law (full
        # coupled-body dynamics don't reduce to a closed form - same honest
        # choice walker_domain.py already made for this body).
        for f in ["height", "pitch", "thigh", "leg", "foot",
                  "thigh_left", "leg_left", "foot_left"]:
            bound(f"c4_reach_{f}", x[f"{f}_next"] - x[f], low=-DELTA_MAX, high=DELTA_MAX, tolerance=tol)

    @staticmethod
    def residuals(x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"pitch_abs": x["pitch"].abs(), "pitch_next_abs": x["pitch_next"].abs()}
