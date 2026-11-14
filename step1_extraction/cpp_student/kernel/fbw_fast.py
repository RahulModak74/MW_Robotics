"""fbw_fast.py - FlyByWire with the look-ahead (_score) in C++ (fbw_kernel). Drop-in: same constructor, same filter(), same stats.

    from fbw_fast import FlyByWireCpp
    fbw = FlyByWireCpp(sim, DOF_NAMES, FEET, student=student)       # student = the pilot policy (student_prob / student_net model)

What changes vs fbw.py
  exact mode (default)  the same computation, natively: physics, proprioception, envelope laws, next observation and the student's
                        action path all run in one C++ loop (OpenMP threads, Eigen float32 MLP). Two exact shortcuts:
                          * the S noise draws are identical when FBW_LA_NOISE = 0 (all clean runs): roll out one draw, not S;
                          * a rollout that has fallen (FBW_FALL=1) is not simulated further (its cost is already final).
  FBW_KEEP_WARM=1       carry MuJoCo's solver warm start along each look-ahead rollout, seeded from the world's, as the world
                        itself steps (fbw.py's mujoco.rollout() zeroes it every control step). Not bit-identical to fbw.py.
  fast mode             FBW_LA_DT = look-ahead physics step (s), e.g. 0.005 instead of 0.001 (decimation 4 instead of 20);
                        FBW_H_CORR = horizon used when scoring corrections (the pilot's own plan keeps FBW_H).
                        Both change the prediction, so the 58-test protocol must be re-run before use.
The pilot function passed to filter() is not called inside the look-ahead (the kernel runs the student's weights itself); the
pilot's command now (T_pilot) is still taken from the caller.
"""
import os
import numpy as np
import torch
from fbw import FlyByWire

import fbw_kernel


def _lin(L):
    return {"W": L.weight.detach().float().numpy(), "b": L.bias.detach().float().numpy()}


def pilot_spec(student, off, sc):
    """the student's action path as plain arrays: normalisation, [encoder, z_mu], body, mean head."""
    m = student
    while hasattr(m, "base"):                                   # LastLayer: the action is the base model's
        m = m.base
    if hasattr(m, "members"):
        raise NotImplementedError("ensembles: run one kernel per member or average in Python")
    if not hasattr(m, "mean"):
        raise NotImplementedError(f"student kind {getattr(m, 'kind', type(m))}: only gauss / mwv2 / mwv3 (+ lastlayer)")
    spec = {k: getattr(m, k).detach().float().numpy() for k in ("xm", "xs", "ym", "ys")}
    spec["body"] = [_lin(L) for L in m.body if isinstance(L, torch.nn.Linear)]
    spec["mean"] = _lin(m.mean)
    if hasattr(m, "enc"):
        spec["enc"] = [_lin(L) for L in m.enc if isinstance(L, torch.nn.Linear)]
        spec["zmu"] = _lin(m.z_mu)
    spec["off"], spec["sc"] = np.asarray(off, np.float64), np.asarray(sc, np.float64)
    return spec


class FlyByWireCpp(FlyByWire):
    def __init__(self, sim, dof_names, feet, cfg=None, student=None):
        super().__init__(sim, dof_names, feet, cfg)
        if student is None:
            raise ValueError("FlyByWireCpp needs the student (its weights run inside the kernel)")
        la_dt = float(os.environ.get("FBW_LA_DT", 0))
        self.c["keep_warm"] = int(os.environ.get("FBW_KEEP_WARM", 0))
        self.c["la_dt"] = la_dt; self.c["h_corr"] = int(os.environ.get("FBW_H_CORR", self.c["H"]))
        decim = self.decim if la_dt <= 0 else int(round(self.decim * self.m.opt.timestep / la_dt))
        self.k = fbw_kernel.Kernel(self.m._address, self.act.tolist(), self.qad.tolist(), self.dof.tolist(), self.torso, self.root,
                                   self.feet, decim, self.c["threads"], pilot_spec(student, self.off, self.sc),
                                   self.lo, self.hi, la_dt)
        self._warm = np.zeros(self.nv)
        self.ccfg = {k: self.c[k] for k in ("w_fall", "w_cp", "w_up", "w_z", "w_spin", "cp_tol", "up_min", "z_min", "spin_max", "fall", "keep_warm")}

    def filter(self, d, x, T_pilot, pilot):
        self._warm = np.array(d.qacc_warmstart)                                 # the world's solver warm start now
        return super().filter(d, x, T_pilot, pilot)

    def _score(self, K, E, s0, x, T_pilot, pilot):
        c, n, S = self.c, len(K), len(E)
        H = c["H"] if n == 1 and not K.any() else c["h_corr"]  # the pilot's plan: full horizon; corrections: FBW_H_CORR
        w = np.linspace(0, 1, c["H"])[None, :, None]
        dT = (1 - w) * K[:, 0][:, None] + w * K[:, 1][:, None]                  # [n, H, J], as fbw.py
        Ek = E[:1] if not E.any() else E                                        # identical draws: one is enough (exact)
        parts, fallen = self.k.score(np.ascontiguousarray(dT[:, :H]), np.ascontiguousarray(Ek[:, :H]), s0, x, T_pilot, self.ccfg, self._warm)
        Sk = len(Ek)
        viol = parts.sum(1).reshape(n, Sk).mean(1); pfall = fallen.reshape(n, Sk).mean(1)
        self._parts = parts.reshape(n, Sk, 5).mean(1)
        dev = (dT ** 2).mean(axis=(1, 2)) / (self.sig ** 2).mean()
        return viol + c["w_dev"] * dev, viol, pfall, dT
