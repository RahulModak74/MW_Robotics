"""priors.py - motion-agnostic physical priors for G1 tracker students (dance / balance priors), differentiable, any ProtoMotions motion.

All laws are evaluated on a (predicted) NEXT state, given the current observation. They use only quantities in the deployable
observation (joint angles/velocities, torso angular velocity, gravity in the torso frame, the reference motion's future poses), plus a
differentiable G1 body model (g1_kin.py, matches MuJoCo to 1e-8 m). No per-motion rules: the reference motion is read from the
observation, so the same priors apply to walking, gestures, dance or anything ProtoMotions can play.

Law values (>= 0, "how unusual"), one row per sample:
  balance   capture-point offset (m): | horiz(CoM - stance) + horiz(v_CoM rel. stance) / omega0 |, omega0 = sqrt(g / CoM height)
  upright   torso tilt: 1 + g_torso_z  (0 = upright, 1 = horizontal)
  limits    joint-limit excess (rad), summed
  reference mean |next joint angles - reference joint angles one step ahead| (rad)
Tolerances are calibrated on the TEACHER's own transitions (quantile, default 95 %): the prior says "stay within what the expert does".

Observation layout (349): proprio 0:64 = dof_pos 0:29, dof_vel 29:58, torso ang vel 58:61, gravity in torso frame 61:64;
future targets 64:320 = 4 steps x [6 rel. torso rot (tan-norm), 29 ref dof_vel, 29 ref dof_pos]; previous actions 320:349.
"""
import numpy as np
import torch

from g1_kin import G1Kin

DT = 0.02                                    # control period (unified_pipeline.yaml: control_dt)
LAWS = ("balance", "upright", "limits", "reference")
Q, QD, W, G = slice(0, 29), slice(29, 58), slice(58, 61), slice(61, 64)
REF_Q1 = slice(64 + 35, 64 + 64)            # reference joint angles at +1 step


class Priors(torch.nn.Module):
    def __init__(self, kin=None, q_lo=None, q_hi=None):
        super().__init__()
        self.kin = kin or G1Kin.from_mjcf()
        self.torso = self.kin.names.index("torso_link")
        if q_lo is None:
            q_lo, q_hi = joint_limits()
        self.register_buffer("q_lo", torch.as_tensor(q_lo, dtype=torch.float32))
        self.register_buffer("q_hi", torch.as_tensor(q_hi, dtype=torch.float32))
        self.register_buffer("tol", torch.zeros(len(LAWS)))
        self.register_buffer("scale", torch.ones(len(LAWS)))

    def _frame(self, q, g_torso):
        out = self.kin(q)
        Rt = self._torso_rot(q)                               # torso orientation in the pelvis frame
        g = torch.nn.functional.normalize((Rt @ g_torso.unsqueeze(-1)).squeeze(-1), dim=-1)   # "down" in the pelvis frame
        return out["com"], out["feet"], g

    def _torso_rot(self, q):
        # recompute rotations along the chain to the torso only (cheap: pelvis -> waist yaw/roll/pitch -> torso)
        B = q.shape[0]; k = self.kin
        chain, i = [], self.torso
        while i >= 0:
            chain.append(i); i = k.parent[i]
        R = torch.eye(3, dtype=q.dtype, device=q.device).expand(B, 3, 3)
        from g1_kin import axis_angle
        for i in reversed(chain):
            R = R @ k.rot[i]
            if k.jaxis[i] is not None:
                R = R @ axis_angle(k.jaxis[i], q[:, k.jdof[i]])
        return R

    def values(self, x_now, p_next):
        """x_now: [B,349] raw observation now; p_next: [B,64] raw proprioception at the next step (true or predicted)."""
        q0, q1 = x_now[:, Q], p_next[:, Q]
        com0, feet0, g0 = self._frame(q0, x_now[:, G])
        com1, feet1, g1 = self._frame(q1, p_next[:, G])
        low = (feet1 * g1.unsqueeze(1)).sum(-1)                # larger = lower along gravity
        wts = torch.softmax(low / 0.01, dim=1).unsqueeze(-1)    # stance weights (both feet ~0.5 in double support)
        st1 = (wts * feet1).sum(1); st0 = (wts * feet0).sum(1)
        r1, r0 = com1 - st1, com0 - st0
        v = (r1 - r0) / DT + torch.cross(p_next[:, W], r1, dim=-1)
        hz = lambda u: u - (u * g1).sum(-1, keepdim=True) * g1
        height = (-(r1 * g1).sum(-1)).clamp(min=0.3)
        omega0 = torch.sqrt(9.81 / height).unsqueeze(-1)
        balance = (hz(r1) + hz(v) / omega0).norm(dim=-1)
        gt = torch.nn.functional.normalize(p_next[:, G], dim=-1)
        upright = 1.0 + gt[:, 2]
        limits = (torch.relu(q1 - self.q_hi) + torch.relu(self.q_lo - q1)).sum(-1)
        reference = (q1 - x_now[:, REF_Q1]).abs().mean(-1)
        return torch.stack([balance, upright, limits, reference], -1)

    def penalty(self, vals, weights=None):
        """Soft prior: 0 inside the teacher's range, linear (in units of the teacher's spread) beyond it. [B]"""
        z = torch.relu(vals - self.tol) / self.scale
        if weights is not None:
            z = z * torch.as_tensor(weights, dtype=z.dtype, device=z.device)
        return z.sum(-1)

    @torch.no_grad()
    def calibrate(self, x_now, p_next, q=0.95):
        v = self.values(x_now, p_next)
        self.tol.copy_(torch.quantile(v, q, dim=0))
        self.scale.copy_((torch.quantile(v, 0.99, dim=0) - torch.quantile(v, 0.5, dim=0)).clamp(min=1e-3))
        return v


def pd_map():
    """Deployment action mapping: joint target = offset + scale * action (unified_pipeline.yaml). The observation's previous-action
    block is exactly this joint target (checked: max |diff| 0.00000)."""
    import os, yaml
    y = yaml.safe_load(open(os.path.expanduser("" + os.environ.get("PM", os.path.expanduser("~/ProtoMotions")) + "/data/pretrained_models/motion_tracker/"
                                                 "g1-bones-deploy/compiled_models/unified_pipeline.yaml")))
    return torch.tensor(y["control"]["pd_action_offset"], dtype=torch.float32), torch.tensor(y["control"]["action_scale"], dtype=torch.float32)


def next_obs(x, p_next, a, off, sc):
    """Roll the 349-d observation one control step: new proprioception, reference targets shifted one step (+1,+2,+4,+8 interpolated),
    previous-action block = joint target of the action just taken. Differentiable."""
    x2 = x.clone()
    x2[:, :64] = p_next
    x2[:, 320:349] = off + sc * a
    B = [x[:, 64 + 64 * k: 128 + 64 * k] for k in range(4)]
    x2[:, 64:128] = B[1]; x2[:, 128:192] = 0.5 * (B[1] + B[2]); x2[:, 192:256] = 0.75 * B[2] + 0.25 * B[3]; x2[:, 256:320] = B[3]
    return x2


def joint_limits():
    import mujoco
    from g1_kin import load_mj, DOF_NAMES
    m = load_mj()
    r = np.array([m.jnt_range[m.joint(n).id] for n in DOF_NAMES])
    return r[:, 0], r[:, 1]


def transitions(files, drop_falls=True):
    """(x_t, a_t, p_{t+1}) within episodes from pm_collect npz files; drops episodes where the teacher fell."""
    Xs, As, Ps = [], [], []
    for f in files:
        z = np.load(f); X, A, ep = z["X"], z["A"], z["ep"]
        ok = np.ones(len(X), bool)
        if drop_falls and "root_z" in z.files:
            bad = np.unique(ep[(z["root_z"] < 0.45) | (z["upright"] < 0.5)]); ok = ~np.isin(ep, bad)
        same = (ep[:-1] == ep[1:]) & ok[:-1] & ok[1:]
        Xs.append(X[:-1][same]); As.append(A[:-1][same]); Ps.append(X[1:][same][:, :64])
    return np.concatenate(Xs), np.concatenate(As), np.concatenate(Ps)


if __name__ == "__main__":
    import glob, os, sys
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_pm")
    pats = sys.argv[2].split(",") if len(sys.argv) > 2 else ["walk", "wave", "high5", "jumpjack", "uncond", "dance"]
    pri = Priors()
    for p in pats:
        fs = sorted(glob.glob(os.path.join(d, f"{p}_s*.npz")))
        if not fs:
            continue
        X, A, P = transitions(fs)
        idx = np.random.RandomState(0).choice(len(X), min(20000, len(X)), replace=False)
        v = pri.values(torch.tensor(X[idx]), torch.tensor(P[idx])).numpy()
        qs = np.percentile(v, [50, 95, 99], axis=0)
        print(f"{p:9s} n={len(idx):6d} | " + " | ".join(f"{n} p50 {qs[0, i]:.3f} p95 {qs[1, i]:.3f} p99 {qs[2, i]:.3f}" for i, n in enumerate(LAWS)))
