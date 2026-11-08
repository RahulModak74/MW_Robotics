"""full_cpu_trainer_mw_bayesian.py - TEACHER-FREE MW Bayesian trainer for the G1 motion-tracking task. CPU only (MuJoCo through the
ProtoMotions environment). No GPU, no RL, no teacher actions: the NVIDIA checkpoint is loaded only because ProtoMotions' launcher builds
the environment from it; its policy is never queried (enforced: BaseEvaluator._policy_action raises).

Stage A (this file): conjugate Bayesian "note keeping". Learning = adding sufficient statistics; the posterior is closed form.
  * Dynamics belief  p_{t+1} = W^T phi(p_t, T_t) + noise,  phi = [q, qd, w_torso, g_torso, g x w, T (PD joint targets), 1]
    one Normal-Inverse-Gamma per output (64): prior MEAN mu = physics, prior PRECISION = how much we trust it (pseudo-samples):
      - each joint: exact discretisation of its PD loop  I q'' = kp (T - q) - kd q'  (kp, kd from the deployed pipeline, I = MuJoCo's
        joint inertia incl. armature) -> coefficients on q, qd, T;
      - gravity seen from the torso rotates with it: g' = g + dt (g x w);  torso angular velocity persists;  everything else ~ 0.
    Every observed transition just ADDS to S_xx, S_xy, S_yy, n; refit = 64 small linear solves (milliseconds).
  * Policy = closed-form Bayesian MPC on the posterior mean: H-step look-ahead, quadratic laws (track the reference joint angles and
    velocities read from the observation, keep the torso upright, damp torso spin, smooth targets near the reference), joint limits as
    a hard projection. One 29H x 29H solve per control step (precomputed factor; ~0.1 ms).
  * Rounds: act in MuJoCo with the current belief (+ small exploration noise) -> add the notes -> refit. Round 0 uses the physics prior
    alone (no data at all). A kinematic baseline (target = reference pose) is run first for comparison.
Stage B (next, not here): Pyro SVI with reparameterised gradients for the nonlinear parts (policy net through this conjugate dynamics, the
capture-point balance law), keeping the conjugate pieces as closed-form updates.

Run inside ProtoMotions (like pm_collect.py; source ../step1_extraction/student/scripts/common.sh for PM, PY, CK):
    cd $PM && MWB_OUT=<dir> MWB_NAME=mwcpu_s100 $PY <repo>/step2_policy_design/full_cpu_trainer_mw_bayesian.py --simulator mujoco --headless --num-envs 1 \
        --checkpoint $CK --motion-file <a motion library, e.g. from step1_extraction/student/kimodo_data>
Env: MWB_ROUNDS (8), MWB_STEPS per round (3000), MWB_KIN_STEPS (2000), MWB_H (4), MWB_NOISE (0.03 rad on targets), MWB_SEED (0).
"""
import os, sys, json, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, '../step1_extraction/student/ours')]   # priors, g1_kin, pm_launch
DT = 0.02
NJ = 29
# observation layout (349): proprio 0:64 = q 0:29, qd 29:58, torso ang vel 58:61, gravity in torso frame 61:64;
# 4 future reference blocks (+1, +2, +4, +8 steps) of 64 = [6 rel. torso rot, 29 ref dof vel, 29 ref dof pos]; previous targets 320:349
REF0, PREV = 64, slice(320, 349)
F_Q, F_QD, F_W, F_G, F_C, F_T, F_B = slice(0, 29), slice(29, 58), slice(58, 61), slice(61, 64), slice(64, 67), slice(67, 96), 96
NF, NY = 97, 64
KEYS = ["noisy_reduced_coords_obs", "noisy_mimic_reduced_coords_target_poses", "historical_previous_processed_actions"]


def env_i(k, d, cast=float):
    return cast(os.environ.get(k, d))


def expm2(A):
    """matrix exponential of a small matrix (scaling and squaring + Taylor)."""
    s = max(0, int(np.ceil(np.log2(max(np.abs(A).sum(1).max(), 1e-12)))) + 4)
    X = A / 2 ** s; E = np.eye(len(A)); term = np.eye(len(A))
    for k in range(1, 20):
        term = term @ X / k; E = E + term
    for _ in range(s):
        E = E @ E
    return E


def features(x):
    """x [B,349] observation -> current proprio p [B,64] and phi without the target block (filled by the caller)."""
    p = x[:, :64]
    phi = np.zeros((len(x), NF)); phi[:, :64] = p
    phi[:, F_C] = np.cross(p[:, F_G], p[:, F_W]); phi[:, F_B] = 1.0
    return p, phi


def refs(x, H):
    """reference joint angles / velocities for look-ahead steps 1..H, from the observation's +1,+2,+4,+8 blocks (linear interp)."""
    blk = lambda b: (x[64 + 64 * b + 35: 64 + 64 * b + 64], x[64 + 64 * b + 6: 64 + 64 * b + 35])
    at = {1: blk(0), 2: blk(1), 4: blk(2), 8: blk(3)}
    out = []
    for k in range(1, H + 1):
        if k in at:
            out.append(at[k])
        else:
            lo = max(s for s in at if s < k); hi = min(s for s in at if s > k); w = (k - lo) / (hi - lo)
            out.append(tuple((1 - w) * at[lo][i] + w * at[hi][i] for i in range(2)))
    return out


class Belief:
    """Per-output Normal-Inverse-Gamma linear-Gaussian dynamics. Learning = additions to (Sxx, Sxy, Syy, n)."""

    def __init__(self, M0, L0, a0, b0):
        self.M0, self.L0, self.a0, self.b0 = M0, L0, a0, b0                        # [NF,NY], [NF,NY] (diag prec.), [NY], [NY]
        self.Sxx = np.zeros((NF, NF)); self.Sxy = np.zeros((NF, NY)); self.Syy = np.zeros(NY); self.n = 0
        self.refit()

    def add(self, Phi, Y):                                                      # the "notes": pure additions
        self.Sxx += Phi.T @ Phi; self.Sxy += Phi.T @ Y; self.Syy += (Y ** 2).sum(0); self.n += len(Phi)

    def refit(self):
        self.W = np.zeros((NF, NY)); self.s2 = np.zeros(NY); self.Linv_diag = np.zeros((NF, NY))
        for j in range(NY):
            L0 = np.diag(self.L0[:, j]); Ln = L0 + self.Sxx
            m = np.linalg.solve(Ln, L0 @ self.M0[:, j] + self.Sxy[:, j])
            an = self.a0[j] + self.n / 2
            bn = self.b0[j] + 0.5 * (self.Syy[j] + self.M0[:, j] @ L0 @ self.M0[:, j] - m @ Ln @ m)
            self.W[:, j] = m; self.s2[j] = max(bn, 1e-12) / (an - 1)
            self.Linv_diag[:, j] = np.diag(np.linalg.inv(Ln))
        return self

    def predict(self, Phi):
        return Phi @ self.W


class MPC:
    """Closed-form H-step MPC on the belief's mean dynamics (linear in p and T; the g x w feature is frozen at the current state)."""

    def __init__(self, cfg):
        self.c = cfg; self.H = cfg["H"]

    def build(self, W):
        H, c = self.H, self.c
        A = W[:64].T; B = W[F_T].T                                              # p' = A p + B T + (bias + cross)
        P = [np.eye(64)]
        for _ in range(H):
            P.append(A @ P[-1])
        self.P = P; self.A, self.B = A, B
        Su = np.zeros((64 * H, NJ * H))
        for k in range(1, H + 1):
            for i in range(k):
                Su[64 * (k - 1):64 * k, NJ * i:NJ * (i + 1)] = P[k - 1 - i] @ B
        self.Su = Su
        q = np.zeros(64); q[F_Q] = c["w_q"]; q[F_QD] = c["w_qd"]; q[F_W] = c["w_w"]; q[F_G] = c["w_g"]
        self.Qd = np.tile(q, H)
        D = np.eye(NJ * H) - np.eye(NJ * H, k=-NJ)                              # target differences (first row: T_0 - T_prev)
        self.D = D
        Hm = Su.T @ (self.Qd[:, None] * Su) + c["r_smooth"] * D.T @ D + c["r_ref"] * np.eye(NJ * H)
        self.Hinv = np.linalg.inv(Hm)
        self.Wc, self.Wb = W[F_C].T, W[F_B]
        return self

    def act(self, x, lo, hi):
        H, c = self.H, self.c
        p = x[:64]; cross = np.cross(p[F_G], p[F_W]); cst = self.Wc @ cross + self.Wb
        sx = np.zeros(64 * H); acc = p.copy(); off = np.zeros(64)
        for k in range(1, H + 1):                                                # free response (no targets): A^k p + sum A^i cst
            off = self.A @ off + cst; sx[64 * (k - 1):64 * k] = self.P[k] @ p + off
        rf = refs(x, H); r = np.zeros(64 * H); qref = np.concatenate([a for a, _ in rf])
        for k, (qa, qv) in enumerate(rf):
            r[64 * k:64 * k + 29] = qa; r[64 * k + 29:64 * k + 58] = qv; r[64 * k + 61:64 * k + 64] = c["g_des"]
        d0 = np.zeros(NJ * H); d0[:NJ] = x[PREV]
        rhs = self.Su.T @ (self.Qd * (r - sx)) + c["r_smooth"] * self.D.T @ d0 + c["r_ref"] * qref
        U = self.Hinv @ rhs
        return np.clip(U[:NJ], lo, hi)


def physics_prior(kp, kd, I, s2_out):
    """prior mean (physics) and prior precision (pseudo-samples x feature scale^2) per output."""
    M0 = np.zeros((NF, NY)); K = np.full((NF, NY), 2.0); K[F_B] = 0.01            # default: coefficient ~ 0, weak (2 pseudo-samples)
    for j in range(NJ):
        Ac = np.array([[0, 1], [-kp[j] / I[j], -kd[j] / I[j]]]); Bc = np.array([0.0, kp[j] / I[j]])
        Ad = expm2(Ac * DT); Bd = np.linalg.solve(Ac, (Ad - np.eye(2)) @ Bc)
        M0[F_Q.start + j, j], M0[F_QD.start + j, j], M0[F_T.start + j, j] = Ad[0, 0], Ad[0, 1], Bd[0]
        M0[F_Q.start + j, 29 + j], M0[F_QD.start + j, 29 + j], M0[F_T.start + j, 29 + j] = Ad[1, 0], Ad[1, 1], Bd[1]
        for col in (j, 29 + j):
            K[[F_Q.start + j, F_QD.start + j, F_T.start + j], col] = 30.0           # physics coefficients: worth 30 samples
    for i in range(3):
        M0[F_W.start + i, 58 + i] = 1.0; K[F_W.start + i, 58 + i] = 10.0         # torso spin persists
        M0[F_G.start + i, 61 + i] = 1.0; K[F_G.start + i, 61 + i] = 30.0         # gravity direction persists ...
        M0[F_C.start + i, 61 + i] = DT; K[F_C.start + i, 61 + i] = 30.0          # ... and rotates with the torso: g' = g + dt (g x w)
    scale = np.zeros(NF); scale[F_Q] = 0.5; scale[F_QD] = 3.0; scale[F_W] = 1.5; scale[F_G] = 0.7; scale[F_C] = 1.0; scale[F_T] = 0.5; scale[F_B] = 1.0
    L0 = K * scale[:, None] ** 2
    a0 = np.full(NY, 2.0); b0 = a0 * s2_out
    return M0, L0, a0, b0


CFG = dict(H=4, w_q=1.0, w_qd=0.01, w_w=0.02, w_g=1.0, g_des=[0.0, 0.0, -1.0], r_smooth=0.05, r_ref=0.01)


class MWCPUPolicy(torch.nn.Module):
    """Deployable policy: observation [B,349] -> action [B,29] (closed-form MPC on the saved belief). For pm_eval_student.py."""

    def __init__(self, ck):
        super().__init__()
        self.ck = ck; self.mpc = MPC(ck["cfg"]).build(ck["W"])
        self.off, self.sc, self.lo, self.hi = (np.asarray(ck[k]) for k in ("pd_offset", "pd_scale", "T_lo", "T_hi"))

    def forward(self, x):
        xs = x.detach().cpu().numpy().astype(np.float64)
        T = np.stack([self.mpc.act(r, self.lo, self.hi) for r in xs])
        return torch.as_tensor((T - self.off) / self.sc, dtype=torch.float32)


def load_policy(path, device="cpu"):
    return MWCPUPolicy(torch.load(path, map_location="cpu", weights_only=False))


ECFG = dict(H=4, eps=0.02, w_q=1.0, w_qd=0.005, w_tilt=4.0, w_spin=0.02, w_cp=2.0, r=0.02, cp_tol=0.05)


class EOMPrior:
    """Iron-clad prior: the robot's full equations of motion from its MJCF (mass matrix = joint coupling, gravity, contacts with friction,
    position actuators kp/kd, limits), integrated exactly as the control step does (decimation x mj_step), on a PRIVATE copy of the model
    (`m_prior`, the nominal robot) - never the world's own data. Laws on the predicted states: track reference joint angles/velocities,
    torso upright, low spin, capture point inside the support (balance). Control = one 29x29 solve per step on finite-difference
    sensitivities of the H-step look-ahead to a target correction dT (held over the horizon). Residual belief: conjugate NIG on what the
    equations get wrong (KOH discrepancy) - additions only; its mean corrects the predicted joint angles/velocities."""

    def __init__(self, m_prior, decim, jid, act_idx, feet, torso, root, cfg):
        import mujoco
        self.mj, self.m, self.decim, self.c = mujoco, m_prior, decim, cfg
        self.d2 = mujoco.MjData(m_prior); self.qad = [int(m_prior.jnt_qposadr[j]) for j in jid]; self.dof = [int(m_prior.jnt_dofadr[j]) for j in jid]
        self.act, self.feet, self.torso, self.root = act_idx, feet, torso, root
        self.ny = 29 + 29 + 2 + 3 + 2
        w = np.concatenate([np.full(29, cfg["w_q"]), np.full(29, cfg["w_qd"]), np.full(2, cfg["w_tilt"]), np.full(3, cfg["w_spin"]), np.full(2, cfg["w_cp"])])
        self.Q = np.tile(w, cfg["H"])
        M0 = np.zeros((NF, 58)); L0 = np.full((NF, 58), 5.0 * 0.5 ** 2); L0[F_B] = 1.0         # residual prior: mean 0 (trust the equations)
        self.res = Belief.__new__(Belief); self.res_dims = 58
        self.res.M0, self.res.L0, self.res.a0, self.res.b0 = M0, L0, np.full(58, 2.0), 2.0 * np.concatenate([np.full(29, 1e-6), np.full(29, 1e-4)])
        self.res.Sxx = np.zeros((NF, NF)); self.res.Sxy = np.zeros((NF, 58)); self.res.Syy = np.zeros(58); self.res.n = 0
        self._refit_res()

    def _refit_res(self):
        r = self.res; r.W = np.zeros((NF, 58)); r.s2 = np.zeros(58)
        for j in range(58):
            L0 = np.diag(r.L0[:, j]); Ln = L0 + r.Sxx; mj = np.linalg.solve(Ln, r.Sxy[:, j]); r.W[:, j] = mj
            r.s2[j] = max(r.b0[j] + 0.5 * (r.Syy[j] - mj @ Ln @ mj), 1e-12) / (r.a0[j] + r.n / 2 - 1)

    def load(self, d):
        d2 = self.d2; d2.time = d.time; d2.qpos[:] = d.qpos; d2.qvel[:] = d.qvel; d2.act[:] = d.act; d2.qacc_warmstart[:] = d.qacc_warmstart

    def feat(self):
        mj, m, d2 = self.mj, self.m, self.d2
        mj.mj_forward(m, d2); mj.mj_subtreeVel(m, d2)
        R = d2.xmat[self.torso].reshape(3, 3)
        com, vcom = d2.subtree_com[self.root], d2.subtree_linvel[self.root]
        fz = np.array([d2.xpos[f][2] for f in self.feet]); wts = np.exp(-(fz - fz.min()) / 0.02); wts /= wts.sum()
        sup = (wts[:, None] * np.array([d2.xpos[f] for f in self.feet])).sum(0)
        h = max(com[2] - sup[2], 0.3); cp = (com[:2] - sup[:2]) + vcom[:2] / np.sqrt(9.81 / h)
        n = np.linalg.norm(cp); cp = cp * max(0.0, n - self.c["cp_tol"]) / max(n, 1e-9)              # only beyond the tolerance
        return np.concatenate([d2.qpos[self.qad], d2.qvel[self.dof], R[:2, 2], d2.qvel[3:6], cp])

    def rollout(self, d, Tseq):
        self.load(d); ys = []
        for T in Tseq:
            self.d2.ctrl[self.act] = T
            for _ in range(self.decim):
                self.mj.mj_step(self.m, self.d2)
            ys.append(self.feat())
        return np.concatenate(ys)

    def act_on(self, d, x, lo, hi):
        c, H = self.c, self.c["H"]
        rf = refs(x, H); Tn = np.stack([np.clip(qa, lo, hi) for qa, _ in rf])
        ystar = np.concatenate([np.concatenate([qa, qv, np.zeros(2), np.zeros(3), np.zeros(2)]) for qa, qv in rf])
        y0 = self.rollout(d, Tn)
        if self.res.n:                                                         # learned discrepancy on the first predicted step
            _, phi = features(x[None]); phi[0, F_T] = Tn[0]; y0[:58] += (phi @ self.res.W)[0]
        G = np.zeros((len(y0), NJ))
        for i in range(NJ):
            Tp = Tn.copy(); Tp[:, i] += c["eps"]; G[:, i] = (self.rollout(d, Tp) - y0) / c["eps"]
        A = G.T @ (self.Q[:, None] * G) + c["r"] * np.eye(NJ)
        dT = np.linalg.solve(A, G.T @ (self.Q * (ystar - y0)))
        T0 = np.clip(Tn[0] + dT, lo, hi)
        self.last_pred = self.rollout(d, T0[None])[:58]                        # equations' prediction for the target actually sent
        return T0

    def note(self, x, T0, x_next):                                             # residual note: actual - equations (additions only)
        _, phi = features(x[None]); phi[0, F_T] = T0; r = x_next[:58] - self.last_pred
        self.res.Sxx += phi.T @ phi; self.res.Sxy += phi.T @ r[None]; self.res.Syy += r ** 2; self.res.n += 1
        return r


def train(self):
    import mujoco
    from priors import pd_map
    from g1_kin import DOF_NAMES
    from protomotions.agents.evaluators.base_evaluator import BaseEvaluator

    def no_teacher(*a, **k):
        raise RuntimeError("teacher-free trainer: the NVIDIA policy must never be queried")
    BaseEvaluator._policy_action = no_teacher

    out_dir = os.environ.get("MWB_OUT", os.path.join(os.path.dirname(HERE), "students")); name = os.environ.get("MWB_NAME", "mwcpu")
    os.makedirs(out_dir, exist_ok=True)
    rounds, steps, kin_steps = env_i("MWB_ROUNDS", 8, int), env_i("MWB_STEPS", 3000, int), env_i("MWB_KIN_STEPS", 2000, int)
    noise = env_i("MWB_NOISE", 0.03); rng = np.random.default_rng(env_i("MWB_SEED", 0, int))
    cfg = dict(CFG, H=env_i("MWB_H", 4, int))

    sim = self.env.simulator; m, d = sim.model, sim.data
    import yaml
    y = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "third_party/ProtoMotions/data/pretrained_models/motion_tracker/"
                                         "g1-bones-deploy/compiled_models/unified_pipeline.yaml")))["control"]
    kp, kd = np.array(y["stiffness"]), np.array(y["damping"])
    off, sc = (t.numpy().astype(np.float64) for t in pd_map())
    jid = [m.joint(n).id for n in DOF_NAMES]; dof = [int(m.jnt_dofadr[j]) for j in jid]; qad = [int(m.jnt_qposadr[j]) for j in jid]
    lo, hi = m.jnt_range[jid, 0].astype(np.float64), m.jnt_range[jid, 1].astype(np.float64)
    M = np.zeros((m.nv, m.nv)); mujoco.mj_forward(m, d); (mujoco.mj_fullM(m, d, M) if not hasattr(d, "qM") else mujoco.mj_fullM(m, M, d.qM)); I = np.diag(M)[dof]
    print("joint inertia (kg m^2, incl. armature):", np.round(I, 3).tolist(), flush=True)
    s2_out = np.concatenate([np.full(29, 0.01 ** 2), np.full(29, 0.5 ** 2), np.full(3, 0.3 ** 2), np.full(3, 0.02 ** 2)])
    bel = Belief(*physics_prior(kp, kd, I, s2_out)); mpc = MPC(cfg).build(bel.W)

    self.agent.eval()
    done_idx, log, t_all = None, [], time.time()

    def obs_now():
        nonlocal done_idx
        obs, _ = self.env.reset(done_idx)
        self.agent.pre_collect_step(0)
        obs = self.agent.add_agent_info_to_obs(obs)
        td = self.agent.obs_dict_to_tensordict(obs)
        return torch.cat([td[k].reshape(td[k].shape[0], -1) for k in KEYS], -1)[0].cpu().numpy().astype(np.float64)

    def run(tag, n, mode):
        nonlocal done_idx
        X, T, EP, RZ, UP, ends = [], [], [], [], [], []; PR.clear()
        ep, ep_len, t0 = 0, 0, time.time()
        for _ in range(n):
            x = obs_now()
            if not X:
                qd_ = d.qpos[qad]; print(f"  [{tag}] obs vs MuJoCo joint angles max |diff| {np.abs(x[F_Q] - qd_).max():.4f}", flush=True)
            if mode == "kin":
                Tt = np.clip(x[64 + 35: 64 + 64], lo, hi)                       # baseline: target = reference pose at +1
            elif mode == "eom":
                Tt = eom.act_on(d, x, lo, hi); PR.append(eom.last_pred)
            else:
                Tt = mpc.act(x, lo, hi)
            Tx = np.clip(Tt + noise * rng.standard_normal(NJ), lo, hi) if (noise > 0 and mode != "eom") else Tt
            a = torch.as_tensor(((Tx - off) / sc)[None], dtype=torch.float32)
            X.append(x); T.append(Tx); EP.append(ep)
            _, _, dones, _, _ = self.env.step(a)
            RZ.append(float(d.qpos[2])); UP.append(float(1.0 - 2.0 * (d.qpos[4] ** 2 + d.qpos[5] ** 2))); ep_len += 1
            done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
            fell = RZ[-1] < 0.45 or UP[-1] < 0.5                                 # pelvis below 0.45 m or tilted > 60 deg
            if fell and not len(done_idx):
                done_idx = torch.zeros(1, dtype=torch.long, device=dones.device)  # the env does not end on a fall here: reset now
            if len(done_idx):
                ends.append((ep_len, fell)); ep += 1; ep_len = 0
        X, T, EP = np.array(X), np.array(T), np.array(EP)
        same = EP[:-1] == EP[1:]
        _, Phi = features(X[:-1][same]); Phi[:, F_T] = T[:-1][same]; Y = X[1:][same][:, :64]
        pred_rmse = np.sqrt(((bel.predict(Phi) - Y) ** 2).mean(0))                # belief BEFORE these notes: honest out-of-sample
        track = np.abs(X[:, F_Q] - X[:, 99:128]).mean()                          # |joint angles - reference at +1|
        full = ends
        if mode == "eom":                                                      # residual notes: actual next - equations' prediction
            Pp = np.array(PR); idx = np.where(same)[0]; Rr = X[idx + 1][:, :58] - Pp[idx]
            _, Pr = features(X[idx]); Pr[:, F_T] = T[idx]
            res_rmse = np.sqrt((Rr ** 2).mean(0)); eom.res.Sxx += Pr.T @ Pr; eom.res.Sxy += Pr.T @ Rr; eom.res.Syy += (Rr ** 2).sum(0); eom.res.n += len(Pr)
            eom._refit_res()
            extra = dict(eom_resid_rmse_q=float(res_rmse[:29].mean()), eom_resid_rmse_qd=float(res_rmse[29:].mean()), residual_notes=int(eom.res.n))
        else:
            extra = {}
        rec = dict(round=tag, steps=n, episodes=len(full), falls=int(sum(f for _, f in full)),
                   mean_ep_steps=float(np.mean([l for l, _ in full])) if full else float(n), steps_to_fall_median=float(np.median([l for l, f in full if f])) if any(f for _, f in full) else None, track_err_rad=float(track),
                   upright_mean=float(np.mean(UP)), root_z_mean=float(np.mean(RZ)),
                   pred_rmse_q=float(pred_rmse[F_Q].mean()), pred_rmse_qd=float(pred_rmse[F_QD].mean()), pred_rmse_g=float(pred_rmse[F_G].mean()),
                   transitions=int(len(Phi)), secs=round(time.time() - t0, 1), **extra)
        print(f"[{tag}] " + json.dumps(rec), flush=True)
        return Phi, Y, rec

    PR = []
    if os.environ.get("MWB_PRIOR", "linear") == "eom":                        # iron-clad prior: the full equations of motion
        import copy
        m_prior = copy.deepcopy(m)                                             # the nominal robot (prior); the world may differ:
        payload = env_i("MWB_WORLD_PAYLOAD", 0.0)
        torso = m.body("torso_link").id
        if payload:
            m.body_mass[torso] += payload; mujoco.mj_setConst(m, d); print(f"WORLD has +{payload} kg on the torso; the prior does not", flush=True)
        act_of = {int(m.actuator_trnid[i, 0]): i for i in range(m.nu)}; act_idx = [act_of[j] for j in jid]
        from g1_kin import FEET
        ecfg = dict(ECFG, H=env_i("MWB_H", ECFG["H"], int))
        eom = EOMPrior(m_prior, int(sim.decimation), jid, act_idx, [m.body(f).id for f in FEET], torso, m.body("pelvis").id, ecfg)
        print(f"EOM prior: decimation {sim.decimation}, dt {m.opt.timestep}, H {ecfg['H']}, feet {FEET}", flush=True)
        Phi, Y, rec = run("kinematic", kin_steps, "kin"); log.append(rec)
        for r in range(rounds):
            Phi, Y, rec = run(f"eom_round{r}", steps, "eom"); log.append(rec)
        ck = dict(kind="mwcpu_eom", cfg=ecfg, residual=dict(W=eom.res.W, s2=eom.res.s2, n=eom.res.n), payload=payload)
        torch.save(ck, os.path.join(out_dir, f"{name}.pt")); json.dump(log, open(os.path.join(out_dir, f"{name}_rounds.json"), "w"), indent=1)
        print(f"saved {name} ({(time.time() - t_all) / 60:.1f} min total)", flush=True)
        return
    Phi, Y, rec = run("kinematic", kin_steps, "kin"); log.append(rec); bel.add(Phi, Y); bel.refit()   # baseline data are notes too
    for r in range(rounds):
        # round 0 acts on the physics prior ALONE (zero notes); later rounds on the belief with all notes so far
        mpc.build(Belief(bel.M0, bel.L0, bel.a0, bel.b0).W if r == 0 else bel.W)
        Phi, Y, rec = run(f"round{r}{'_prior_only' if r == 0 else ''}", steps, "mpc"); rec["notes_before"] = int(bel.n); log.append(rec)
        bel.add(Phi, Y); bel.refit()
    mpc.build(bel.W)
    ck = dict(kind="mwcpu", W=bel.W, s2=bel.s2, cfg=cfg, pd_offset=off, pd_scale=sc, T_lo=lo, T_hi=hi, n_notes=int(bel.n),
              stats=dict(Sxx=bel.Sxx, Sxy=bel.Sxy, Syy=bel.Syy, n=bel.n), prior=dict(M0=bel.M0, L0=bel.L0, a0=bel.a0, b0=bel.b0))
    path = os.path.join(out_dir, f"{name}.pt"); torch.save(ck, path)
    json.dump(log, open(os.path.join(out_dir, f"{name}_rounds.json"), "w"), indent=1)
    print(f"saved {path} ({bel.n} notes, {(time.time() - t_all) / 60:.1f} min total)", flush=True)


if __name__ == "__main__":
    import pm_launch                                              # single-device Fabric patch
    from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
    BaseEvaluator.simple_test_policy = train
    from protomotions.inference_agent import main
    main()
