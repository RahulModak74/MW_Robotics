"""
mw_policy_3d.py - the MW-framework policy manifold for a 3D MuJoCo body.

PolicyManifold3DVAE is an mw_framework.core.MWVAE (Pyro model/guide, trained with SVI through MWDetector) over the
JOINT vector x = (s, a):  s = [torso z, torso quat(4), joint angles, root+joint velocities],  a = actuator torques.
Physics priors (declared with mw_framework.core.bound/constraint, exactly like the shipped domains):
  P1 torso height inside the healthy envelope        P2 unit-norm torso quaternion (a rotation, not a free 4-vector)
  P3 joint angles inside MuJoCo jnt_range            P4 torques inside actuator limits
  P5 (optional) torso angular speed inside the range seen in the teacher data
  P6 (optional) left/right hip torques anti-phase (soft)
Two state layouts: feat="legacy" (raw quaternion) and feat="gravity" (gravity direction in the torso frame, see
policy_3d.features_gravity); with "gravity", P2 becomes |gravity vector| = 1.
The manifold IS the policy:  act(s) = the action block of the manifold point whose state block best matches s
(latent query: argmin_z ||decode(z)_s - s||^2 + kl*|z|^2).  No separate controller network.
The same object gives the MW distance of any (s, a): how far a state/torque pair is from the learned manifold.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from mw_framework.core import MWVAE, bound, constraint
from policy_3d import get_features


class PolicyManifold3DVAE(MWVAE):
    def __init__(self, s_dim=22, a_dim=17, hi=0.4, z_lo=1.0, z_hi=2.0,
                 joint_lo=None, joint_hi=None, joint_start=5, latent_dim=8, hidden_dim=128,
                 dropout=0.0, law_tolerance=0.05, feat="legacy", ang_limit=None, sym_pair=None, sym_tol=0.4, **kw):
        self.s_dim_, self.a_dim_, self.hi_ = s_dim, a_dim, hi
        self.z_lo_, self.z_hi_, self.joint_start_ = z_lo, z_hi, joint_start
        self.joint_lo_ = list(joint_lo or []); self.joint_hi_ = list(joint_hi or [])
        self.law_tolerance_ = law_tolerance
        self.feat_, self.ang_limit_ = feat, ang_limit
        self.sym_pair_ = list(sym_pair) if sym_pair else None
        self.sym_tol_ = sym_tol
        names = [f"s{i}" for i in range(s_dim)] + [f"a{i}" for i in range(a_dim)]
        kw.setdefault("n_features", len(names)); kw.setdefault("feature_names", names)
        kw.setdefault("obs_sigma", [0.15] * s_dim + [0.05] * a_dim)
        super().__init__(latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout, **kw)

    def get_config(self):
        c = super().get_config()
        c.update(s_dim=self.s_dim_, a_dim=self.a_dim_, hi=self.hi_, z_lo=self.z_lo_, z_hi=self.z_hi_,
                 joint_lo=self.joint_lo_, joint_hi=self.joint_hi_, joint_start=self.joint_start_,
                 law_tolerance=self.law_tolerance_, feat=self.feat_, ang_limit=self.ang_limit_,
                 sym_pair=self.sym_pair_, sym_tol=self.sym_tol_)
        return c

    def physics(self, x):
        tol = self.law_tolerance_
        if self.feat_ == "gravity":                        # s0:3 gravity (unit), s3:6 ang. velocity, s9 height
            bound("p1_height", x["s9"], low=self.z_lo_, high=self.z_hi_, tolerance=tol)
            g2 = sum(x[f"s{i}"] ** 2 for i in range(3))
            constraint("p2_unit_gravity", g2 - 1.0, tolerance=tol)
            if self.ang_limit_:                            # P5: torso angular speed inside what the teacher ever does
                w2 = sum(x[f"s{i}"] ** 2 for i in range(3, 6))
                bound("p5_ang_speed", w2, high=self.ang_limit_ ** 2, tolerance=tol * self.ang_limit_ ** 2)
        else:                                              # legacy: s0 height, s1:5 raw quaternion
            bound("p1_height", x["s0"], low=self.z_lo_, high=self.z_hi_, tolerance=tol)
            q2 = sum(x[f"s{i}"] ** 2 for i in range(1, 5))
            constraint("p2_unit_quat", q2 - 1.0, tolerance=tol)
        for k, (lo, hi) in enumerate(zip(self.joint_lo_, self.joint_hi_)):
            bound(f"p3_joint{k}", x[f"s{self.joint_start_ + k}"], low=lo, high=hi, tolerance=tol)
        for j in range(self.a_dim_):
            bound(f"p4_torque{j}", x[f"a{j}"], low=-self.hi_, high=self.hi_, tolerance=tol)
        if self.sym_pair_:                                 # P6: left/right hip torques in anti-phase (soft, Laplace)
            i, j = self.sym_pair_
            constraint("p6_sym_hip", x[f"a{i}"] + x[f"a{j}"], tolerance=self.sym_tol_ * self.hi_, kind="laplace")


def hip_pair(model):
    """Actuator indices of (right_hip_y, left_hip_y) read from the MuJoCo model by name, or None (e.g. Ant)."""
    import mujoco
    ids = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}
    r, l = ids.get("right_hip_y"), ids.get("left_hip_y")
    return (r, l) if r is not None and l is not None else None


def spec_from_env(env, feat="legacy"):
    """Physics-prior constructor kwargs read from the MuJoCo model (limits are not hand-typed).
    `env` must have been reset once (the gravity features read the torso rotation matrix)."""
    m = env.unwrapped.model
    rng = m.jnt_range[1:]                                   # skip the free joint
    hi = float(env.action_space.high[0])
    s_dim = len(get_features(feat)(env.unwrapped.data))
    return dict(s_dim=s_dim, a_dim=m.nu, hi=hi, feat=feat,
                z_lo=float(env.unwrapped._healthy_z_range[0]) if hasattr(env.unwrapped, "_healthy_z_range") else 0.2,
                z_hi=float(env.unwrapped._healthy_z_range[1]) if hasattr(env.unwrapped, "_healthy_z_range") else 2.0,
                joint_lo=[float(r[0]) for r in rng], joint_hi=[float(r[1]) for r in rng],
                joint_start=10 if feat == "gravity" else 5)


class ManifoldPolicy:
    """act(state_features) -> torques, by querying the MWDetector's manifold."""
    def __init__(self, det, steps=25, lr=0.2):
        self.det, self.steps, self.lr = det, steps, lr
        self.vae = det.vae_.eval()
        self.features = get_features(getattr(self.vae, "feat_", "legacy"))   # the layout this manifold was trained on          # runs on whatever device MWDetector placed it on (MWVAE device rule)
        self.sd = self.vae.s_dim_
        self.mean = torch.as_tensor(det.norm_.mean, dtype=torch.float32)
        self.std = torch.as_tensor(det.norm_.std, dtype=torch.float32)
        self.sig = self.vae.obs_sigma[:self.sd].cpu()
        self.z_prev = None

    def reset(self):
        self.z_prev = None

    def observe(self, data):
        """State features for this manifold, straight from MuJoCo `data`."""
        return self.features(data)

    def act(self, f):
        s = ((torch.as_tensor(f, dtype=torch.float32) - self.mean[:self.sd]) / self.std[:self.sd]).clamp(-8, 8)
        with torch.no_grad():
            if self.z_prev is None:                                       # warm start: encode with a = mean
                x = torch.cat([s, torch.zeros(self.vae.a_dim_)])[None]
                z0 = self.vae.encode(x)[0][0]
            else:
                z0 = self.z_prev
        z = z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad()
            xh = self.vae.decode(z[None])[0]
            loss = (((xh[:self.sd] - s) / self.sig) ** 2).mean() + self.vae.kl_weight * (z ** 2).sum()
            loss.backward(); opt.step()
        self.z_prev = z.detach()
        with torch.no_grad():
            xh = self.vae.decode(z[None])[0]
        a = xh[self.sd:] * self.std[self.sd:] + self.mean[self.sd:]
        hi = self.vae.hi_
        return a.clamp(-hi, hi).numpy().astype(np.float64)
