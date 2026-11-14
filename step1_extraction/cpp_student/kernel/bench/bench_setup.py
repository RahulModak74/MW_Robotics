"""bench_setup.py - common setup: G1 stand-in sim, a random student of the r7c architecture (LastLayer over MWBundleV3, hidden 512,
latent 16), a standing robot, and a 349-d observation. Weights are random (none shipped); compute cost does not depend on them.
The mean head is scaled down so the pilot's targets stay near the default pose and the robot stands with realistic foot contacts."""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path[:0] = [HERE, os.path.dirname(HERE), f"{REPO}/student/ours", f"{REPO}/flight_computer"]
torch.set_num_threads(1)                                      # as in their scripts (OMP_NUM_THREADS=1)

# config D (run_fbw_D.sh)
D = dict(FBW_H=20, FBW_N=64, FBW_S=4, FBW_ITERS=1, FBW_SIG_LEG=0.25, FBW_SIG_WAIST=0.2, FBW_SIG_ARM=0.35, FBW_CP_TOL=0.05,
         FBW_UP_MIN=0.643, FBW_W_CP=50, FBW_W_UP=50, FBW_FALL=1, FBW_W_Z=0, FBW_W_SPIN=0, FBW_Z_MIN=0.6, FBW_QUIET=1.0)
for k, v in D.items():
    os.environ.setdefault(k, str(v))
os.environ.setdefault("FBW_THREADS", str(os.cpu_count()))

import mujoco
from g1_model import FakeSim
from g1_kin import DOF_NAMES, FEET
from priors import pd_map, next_obs
import student_prob as SP


def make_student(seed=0):
    torch.manual_seed(seed)
    base = SP.MWBundleV3(hidden=512, latent_s=16)
    with torch.no_grad():
        base.mean.weight.mul_(0.05); base.mean.bias.zero_()
        base.ys.fill_(0.05); base.ym.zero_()
    st = SP.LastLayer(base, hidden=512)
    with torch.no_grad():
        A = torch.randn(513, 513) * 0.01; st.Linv.copy_(torch.eye(513) + torch.tril(A))
    return st.eval()


OFF, SC = (t.numpy().astype(np.float64) for t in pd_map())


def make_pilot(student):
    def pilot(X):                                             # exactly pm_eval_student.py's pilot
        Xt = torch.as_tensor(X, dtype=torch.float32)
        A = student.uq(Xt)["action"].numpy().astype(np.float64)
        return OFF + SC * A
    return pilot


def standing(sim, settle=1.0):
    m, d = sim.model, sim.data
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.80
    act = {}
    for a in range(m.nu):
        act[int(m.actuator_trnid[a, 0])] = a
    jid = [m.joint(n).id for n in DOF_NAMES]
    for i, j in enumerate(jid):
        d.qpos[m.jnt_qposadr[j]] = OFF[i]
    ctrl = np.zeros(m.nu)
    for i, j in enumerate(jid):
        ctrl[act[j]] = OFF[i]
    d.ctrl[:] = ctrl
    for _ in range(int(settle / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return d


def observation(fbw, d):
    st = np.empty(fbw.ns); mujoco.mj_getState(fbw.m, d, st, fbw.spec)
    p, _, _ = fbw.proprio(st[None])
    x = np.zeros(349)
    x[:64] = p[0]
    for b in range(4):
        x[64 + 64 * b: 64 + 64 * b + 6] = [1, 0, 0, 0, 0, 1]
        x[64 + 64 * b + 35: 64 + 64 * b + 64] = OFF
    x[320:349] = OFF
    return x
