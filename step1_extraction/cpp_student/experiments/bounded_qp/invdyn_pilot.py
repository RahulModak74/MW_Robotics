"""invdyn_pilot.py - test C (v2, 10 Nov: stance feet held by a hard constraint in a whole-body solve; upright vector sign fixed; ridge on contact forces): a teacher-free pilot with balance feedback built in, no search and no learning. Per control step:
joint accelerations follow the reference (PD); the capture point sets the centre-of-mass acceleration; the torso is kept upright; contact forces are solved with friction limits (NNLS);
floating-base inverse dynamics then gives joint torques, sent as PD targets. World state read only, on a private copy."""
import numpy as np, mujoco
from scipy.optimize import nnls


class InvDynPilot:
    def __init__(self, m, jid, act_idx, root, torso, Kp=30., Kd=20., mu=.7, kxi=.5, reg=1e-2, wb=True, w_com=1., w_up=.5, w_post=1e-2, kfoot=10., w_pel=.5, joint=True, w_eq=1e3):
        self.m, self.d = m, mujoco.MjData(m); self.root, self.torso = root, torso; self.qad = np.array([m.jnt_qposadr[j] for j in jid]); self.dof = np.array([m.jnt_dofadr[j] for j in jid])
        self.kp = m.actuator_gainprm[act_idx, 0].copy(); self.kd = -m.actuator_biasprm[act_idx, 2].copy(); self.eff = m.actuator_forcerange[act_idx, 1].copy(); self.Kp, self.Kd, self.mu, self.kxi, self.reg = Kp, Kd, mu, kxi, reg; self.last_resid = 0.; self.wb, self.w_com, self.w_up, self.w_post, self.kfoot, self.w_pel, self.joint, self.w_eq = wb, w_com, w_up, w_post, kfoot, w_pel, joint, w_eq; self.stance = []; self.foot_of = {}; self.base = np.arange(6); self.n = m.nv; self.mass = float(m.body_subtreemass[root]); self.t_solve = []

    def contacts(self):
        m, d = self.m, self.d; G = []; P = []; st = set()
        for i in range(d.ncon):
            c = d.contact[i]; b = [m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]]
            if 0 not in b or c.dist > .005: continue
            body = b[1] if b[0] == 0 else b[0]; st.add(self.foot_of.get(body, body)); s = 1. if b[0] == 0 else -1.; R = np.array(c.frame).reshape(3, 3); n, t1, t2 = s * R[0], R[1], R[2]
            jp = np.zeros((3, self.n)); mujoco.mj_jac(m, d, jp, None, c.pos, body); P.append(c.pos[:2].copy())
            for dr in (t1, -t1, t2, -t2): G.append(jp.T @ (n + self.mu * dr))
        self.stance = sorted(x for x in st if x in self.feet)
        return (np.array(G).T, np.array(P)) if G else (None, None)

    def act_on(self, w, x, lo, hi, qa1, qv1, qv2, dt=.02):
        import time; t0 = time.perf_counter(); m, d = self.m, self.d; d.qpos[:] = w.qpos; d.qvel[:] = w.qvel; d.ctrl[:] = 0; mujoco.mj_forward(m, d); q, v = d.qpos[self.qad], d.qvel[self.dof]
        qdd_j = (qv2 - qv1) / dt + self.Kp * (qa1 - q) + self.Kd * (qv1 - v)
        M = np.zeros((self.n, self.n)); mujoco.mj_fullM(m, d, M); h = d.qfrc_bias.copy()
        Jc = np.zeros((3, self.n)); mujoco.mj_jacSubtreeCom(m, d, Jc, self.root); Jw = np.zeros((3, self.n)); mujoco.mj_jacBody(m, d, None, Jw, self.torso)
        com, vc = d.subtree_com[self.root].copy(), Jc @ d.qvel; G, P = self.contacts()
        if getattr(self, "stance_hold", None) is not None:                     # double-support prior: keep named feet planted even through a momentary contact loss (env reset lands slightly airborne); correct for standing / non-locomotion clips
            self.stance = [f for f in self.stance_hold if float(d.xpos[f][2]) < self.stance_hold_z]
        zf = 0.; om = np.sqrt(9.81 / max(com[2], .3)); xi = com[:2] + vc[:2] / om; ctr = P.mean(0) if P is not None else xi
        p = xi + self.kxi * (xi - ctr)                                         # cop that drives the capture point back towards the support centre
        if P is not None: p = np.clip(p, P.min(0) - .02, P.max(0) + .02)
        az = getattr(self, "kcz", 40.) * (self.z_des(qa1) - com[2]) - getattr(self, "dcz", 12.) * vc[2]; ac = np.array([*(om ** 2 * (com[:2] - p)), az])
        up = d.xmat[self.torso].reshape(3, 3)[:, 2]; wb = Jw @ d.qvel; aw = 80. * np.cross(up, [0, 0, 1.]) - 16. * wb; aw = d.xmat[self.torso].reshape(3, 3).T @ aw if False else aw
        if self.wb:                                                            # whole-body: stance feet held still (hard), CoM / torso / posture as weighted tasks
            S = np.zeros((29, self.n)); S[np.arange(29), self.dof] = 1.
            Jp = np.zeros((3, self.n)); mujoco.mj_jacBody(m, d, None, Jp, self.root); upp = d.xmat[self.root].reshape(3, 3)[:, 2]; ap_ = 80. * np.cross(upp, [0, 0, 1.]) - 16. * (Jp @ d.qvel)
            At = np.vstack([Jc, Jw, Jp, S]); bt = np.concatenate([ac, aw, ap_, qdd_j]); wt = np.concatenate([np.full(3, self.w_com), np.full(3, self.w_up), np.full(3, self.w_pel), np.full(29, self.w_post)])
            Cs, cs = [], []
            for b in self.stance:
                jp, jr = np.zeros((3, self.n)), np.zeros((3, self.n)); mujoco.mj_jacBody(m, d, jp, jr, b); Jf = np.vstack([jp, jr]); Cs.append(Jf); cs.append(-self.kfoot * (Jf @ d.qvel))
            Hm = At.T @ (wt[:, None] * At) + 1e-6 * np.eye(self.n); g = At.T @ (wt * bt)
            if Cs:
                C, c = np.vstack(Cs), np.concatenate(cs); K = np.block([[Hm, C.T], [C, -1e-8 * np.eye(len(C))]]); qdd = np.linalg.solve(K, np.concatenate([g, c]))[:self.n]
            else: qdd = np.linalg.solve(Hm, g)
        else:
            A = np.vstack([Jc[:, :6], Jw[:, :6]]); b = np.concatenate([ac - Jc[:, 6:] @ self.jorder(qdd_j), aw - Jw[:, 6:] @ self.jorder(qdd_j)]); qb = np.linalg.lstsq(A, b, rcond=1e-3)[0]
            qdd = np.concatenate([qb, self.jorder(qdd_j)])
        rhs = M @ qdd + h; f = np.zeros(self.n); self.dbg = dict(com=com[:2].copy(), xi=xi.copy(), ctr=ctr.copy(), p=p.copy(), ac=ac.copy(), sup=(P.min(0), P.max(0)) if P is not None else None, ac_qdd=Jc @ qdd, qdd=qdd.copy(), Jc=Jc.copy(), tau=None)
        if G is not None:                                                     # base rows: contacts must carry the base; ridge spreads forces over all corners
            nc = G.shape[1]; Ga = np.vstack([G[:6], np.sqrt(self.reg) * np.eye(nc)]); lam, _ = nnls(Ga, np.concatenate([rhs[:6], np.zeros(nc)]), maxiter=50 * nc); f = G @ lam
            self.last_resid = float(np.linalg.norm(G[:6] @ lam - rhs[:6]) / (np.linalg.norm(rhs[:6]) + 1e-9))
        tau = np.clip((rhs - f)[self.dof], -self.eff, self.eff); T = q + (tau + self.kd * v) / self.kp; self.dbg['tau'] = tau; self.t_solve.append(time.perf_counter() - t0)
        return np.clip(T, lo, hi)

    def jorder(self, u):
        o = np.zeros(self.n - 6); o[self.dof - 6] = u; return o

    def z_des(self, qa):
        d0 = getattr(self, "_d", None)
        if d0 is None: self._d = d0 = mujoco.MjData(self.m)
        d0.qpos[:] = self.d.qpos; d0.qpos[self.qad] = qa; mujoco.mj_kinematics(self.m, d0); mujoco.mj_comPos(self.m, d0); f = min(d0.xpos[b, 2] for b in self.feet); fw = min(self.d.xpos[b, 2] for b in self.feet); return d0.subtree_com[self.root, 2] - f + fw

    # ---------------- motion tracking (PLAN_10_Nov step 3a) ----------------
    def act_track(self, w, ref, lo, hi, gains=None):
        """ref: dict com/vcom/acom [3], pel_R/tor_R [3x3] + pel_w/tor_w [3] (world), qa/qv/qacc [29], foot_p/foot_v [2,3], foot_R [2][3x3], foot_c [2] (reference contact).
        Stance = feet in contact in the world AND in the reference (held still, hard). Swing feet track the reference footholds. CoM follows the reference,
        with the centre of pressure that this implies clipped into the real support (capture-point logic). Torso and pelvis track the reference orientation."""
        import time; t0 = time.perf_counter(); g_ = dict(kc=30., dc=10., kz=40., dz=12., ko=80., do=16., kf=150., df=25., w_sw=1., margin=.01, land=.02, kflat=100.); g_.update(gains or {})
        m, d = self.m, self.d; d.qpos[:] = w.qpos; d.qvel[:] = w.qvel; d.ctrl[:] = 0; mujoco.mj_forward(m, d); q, v = d.qpos[self.qad], d.qvel[self.dof]; n = self.n
        M = np.zeros((n, n)); mujoco.mj_fullM(m, d, M); h = d.qfrc_bias.copy()
        Jc = np.zeros((3, n)); mujoco.mj_jacSubtreeCom(m, d, Jc, self.root); com = d.subtree_com[self.root].copy(); vc = Jc @ d.qvel
        foot_in = {f: False for f in self.feet}; cols = []
        for i in range(d.ncon):
            c = d.contact[i]; b = [m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]]
            if 0 not in b or c.dist > .002: continue
            body = b[1] if b[0] == 0 else b[0]
            if body in foot_in: foot_in[body] = True
            cols.append((body, c))
        down = [f for f in self.feet if foot_in[f]]                             # a loaded foot stays stance until the other foot is down; only then may it lift
        stance = [f for k, f in enumerate(self.feet) if foot_in[f] and (ref["foot_c"][k] or len(down) < 2)]; self.stance = stance
        G, P = [], []; _cpf = {}; _cap = getattr(self, "max_cpf", 99)
        for body, c in cols:
            if body not in stance: continue
            if _cpf.get(body, 0) >= _cap: continue                              # cap contact points per foot -> bound nc -> bound QP size
            _cpf[body] = _cpf.get(body, 0) + 1
            s = 1. if m.geom_bodyid[c.geom1] == 0 else -1.; R = np.array(c.frame).reshape(3, 3); nn, t1, t2 = s * R[0], R[1], R[2]
            jp = np.zeros((3, n)); mujoco.mj_jac(m, d, jp, None, c.pos, body); P.append(c.pos[:2].copy())
            for dr in (t1, -t1, t2, -t2): G.append(jp.T @ (nn + self.mu * dr))
        om = np.sqrt(9.81 / max(com[2], .3))
        S_ = np.array([q_ for f in stance for q_ in sole_points(m, d, f)]) if stance else (np.array(P) if P else None)   # support = whole soles of stance feet
        kc_, dc_ = g_["kc"], g_["dc"]; self.cp_boost = 0.
        if getattr(self, "cp_filter", False) and S_ is not None:               # analytic capture-point filter: firm the CoM tracking per-state as the capture point xi nears the support edge (the d(s) surrogate); minimal (no change) when xi is central
            xi = com[:2] + vc[:2] / om; lo_s, hi_s = S_.min(0), S_.max(0); ctr = .5 * (lo_s + hi_s); half = .5 * (hi_s - lo_s) + 1e-6
            b = float(np.clip((np.abs(xi - ctr) / half - self.cp_tol) / (1. - self.cp_tol), 0., 1.).max())   # 0 until xi passes cp_tol toward edge, saturates at 1 on the edge
            self.cp_boost = b; kc_ = g_["kc"] * (1. + self.cp_k * b); dc_ = g_["dc"] * (1. + self.cp_k * b)   # firm CoM position + velocity tracking so xi does not build
        axy = ref["acom"][:2] + kc_ * (ref["com"][:2] - com[:2]) + dc_ * (ref["vcom"][:2] - vc[:2])
        p = com[:2] - axy / om ** 2                                             # centre of pressure this acceleration needs
        if S_ is not None: p = np.clip(p, S_.min(0) + g_["margin"], S_.max(0) - g_["margin"]); axy = om ** 2 * (com[:2] - p)
        P = S_ if S_ is not None else []
        az = ref["acom"][2] + g_["kz"] * (ref["com"][2] - com[2]) + g_["dz"] * (ref["vcom"][2] - vc[2]); ac = np.array([*axy, az])
        def ori(body, Rr, wr):
            J = np.zeros((3, n)); mujoco.mj_jacBody(m, d, None, J, body); Rb = d.xmat[body].reshape(3, 3); e = rotvec(Rr @ Rb.T); return J, g_["ko"] * e + g_["do"] * (wr - J @ d.qvel)
        Jp, ap_ = ori(self.root, ref["pel_R"], ref["pel_w"]); Jw, aw = ori(self.torso, ref["tor_R"], ref["tor_w"])
        S = np.zeros((29, n)); S[np.arange(29), self.dof] = 1.; qdd_j = ref["qacc"] + self.Kp * (ref["qa"] - q) + self.Kd * (ref["qv"] - v)
        rows, bs, ws = [Jc, Jw, Jp, S], [ac, aw, ap_, qdd_j], [np.full(3, self.w_com), np.full(3, self.w_up), np.full(3, self.w_pel), np.full(29, self.w_post)]
        Cs, cs = [], []
        for k, f in enumerate(self.feet):
            jp, jr = np.zeros((3, n)), np.zeros((3, n)); mujoco.mj_jacBody(m, d, jp, jr, f); Jf = np.vstack([jp, jr])
            if f in stance:                                                    # held still, and turned back flat (towards the reference foot orientation)
                vf = Jf @ d.qvel; er = rotvec(ref["foot_R"][k] @ d.xmat[f].reshape(3, 3).T); er[2] = 0.
                Cs.append(Jf); cs.append(np.concatenate([-self.kfoot * vf[:3], g_["kflat"] * er - self.kfoot * vf[3:]]))
            else:
                tgt = ref["foot_p"][k].copy()
                if ref["foot_c"][k]: tgt[2] -= g_["land"]                                  # reference says down: press it to the floor
                ep = tgt - d.xpos[f]; er = rotvec(ref["foot_R"][k] @ d.xmat[f].reshape(3, 3).T); vf = Jf @ d.qvel
                rows.append(Jf); bs.append(np.concatenate([g_["kf"] * ep + g_["df"] * (ref["foot_v"][k] - vf[:3]), g_["ko"] * er - g_["do"] * vf[3:]])); ws.append(np.full(6, g_["w_sw"]))
        At, bt, wt = np.vstack(rows), np.concatenate(bs), np.concatenate(ws)
        if self.joint and G:                                                   # one bounded least squares over [qdd, contact forces]: base equations hold, forces only push
            from scipy.optimize import lsq_linear
            G = np.array(G).T; nc = G.shape[1]; sw = np.sqrt(wt); we = np.sqrt(self.w_eq); sm = 1. / self.mass
            blocks = [np.hstack([sw[:, None] * At, np.zeros((len(At), nc))])]; rhsb = [sw * bt]
            if Cs: C, c = np.vstack(Cs), np.concatenate(cs); blocks.append(we * np.hstack([C, np.zeros((len(C), nc))])); rhsb.append(we * c)
            blocks.append(we * sm * np.hstack([M[:6], -G[:6]])); rhsb.append(-we * sm * h[:6])
            blocks.append(np.hstack([np.zeros((nc, n)), np.sqrt(self.reg) * sm * np.eye(nc)])); rhsb.append(np.zeros(nc))
            blocks.append(np.hstack([1e-3 * np.eye(n), np.zeros((n, nc))])); rhsb.append(np.zeros(n))
            sol = lsq_linear(np.vstack(blocks), np.concatenate(rhsb), bounds=(np.r_[np.full(n, -np.inf), np.zeros(nc)], np.full(n + nc, np.inf)), method="bvls", max_iter=getattr(self, "qp_iters", 500))
            qdd, lam = sol.x[:n], sol.x[n:]; rhs = M @ qdd + h; f = G @ lam
            self.last_resid = float(np.linalg.norm(rhs[:6] - f[:6]) / (np.linalg.norm(rhs[:6]) + 1e-9))
        else:
            Hm = At.T @ (wt[:, None] * At) + 1e-6 * np.eye(n); gg = At.T @ (wt * bt)
            if Cs: C, c = np.vstack(Cs), np.concatenate(cs); qdd = np.linalg.solve(np.block([[Hm, C.T], [C, -1e-8 * np.eye(len(C))]]), np.concatenate([gg, c]))[:n]
            else: qdd = np.linalg.solve(Hm, gg)
            rhs = M @ qdd + h; f = np.zeros(n)
            if G:
                G = np.array(G).T; nc = G.shape[1]; lam, _ = nnls(np.vstack([G[:6], np.sqrt(self.reg) * np.eye(nc)]), np.concatenate([rhs[:6], np.zeros(nc)]), maxiter=50 * nc); f = G @ lam
                self.last_resid = float(np.linalg.norm(G[:6] @ lam - rhs[:6]) / (np.linalg.norm(rhs[:6]) + 1e-9))
        tau = np.clip((rhs - f)[self.dof], -self.eff, self.eff); T = q + (tau + self.kd * v) / self.kp; self.t_solve.append(time.perf_counter() - t0)
        self.dbg = dict(stance=list(stance), p=p, com=com, ac=ac, ac_qdd=Jc @ qdd, sup=(np.min(P, 0), np.max(P, 0)) if len(P) else None)
        return np.clip(T, lo, hi)


def rotvec(R):
    """rotation vector (axis * angle) of a rotation matrix"""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); a = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (0.5 / np.sinc(a / np.pi)) if a > 1e-6 else 0.5 * v


def plan_com(c0, v0, c_ref, v_ref, lo, hi, om, dt, w_c=1., w_v=.1, w_dp=1e-3, w_end=10.):
    """Linear-inverted-pendulum MPC, one horizontal axis (projection onto the feasible set): choose centre-of-pressure p_k in [lo_k, hi_k]
    (the planned feet) so the CoM stays near the reference while staying balanced. c_ref, v_ref, lo, hi: [K]. Returns p [K] and the
    CoM acceleration to command now, om^2 (c0 - p0). Exact LIPM discretisation, bounded least squares (scipy lsq_linear)."""
    from scipy.optimize import lsq_linear
    K = len(c_ref); ch, sh = np.cosh(om * dt), np.sinh(om * dt)
    Ad = np.array([[ch, sh / om], [om * sh, ch]]); Bd = np.array([1 - ch, -om * sh])
    Phi = np.zeros((2 * K, 2)); Gam = np.zeros((2 * K, K)); x = np.eye(2); Bk = []
    for k in range(K):
        x = Ad @ x; Phi[2 * k:2 * k + 2] = x
        Gam[2 * k:2 * k + 2, :k + 1] = Ad @ Gam[2 * k - 2:2 * k, :k + 1] if k else 0
        Gam[2 * k:2 * k + 2, k] = Bd
    free = Phi @ np.array([c0, v0]); W = np.tile([w_c, w_v], K); W[-2:] = [w_end, w_end]
    D = np.eye(K) - np.eye(K, k=-1)
    Am = np.vstack([np.sqrt(W)[:, None] * Gam, np.sqrt(w_dp) * D]); tgt = np.empty(2 * K); tgt[0::2] = c_ref; tgt[1::2] = v_ref
    bm = np.concatenate([np.sqrt(W) * (tgt - free), np.zeros(K)]); bm[2 * K] += np.sqrt(w_dp) * np.clip(c0, lo[0], hi[0])
    lo2, hi2 = np.minimum(lo, hi - 1e-4), np.maximum(hi, lo + 1e-4)
    p = lsq_linear(Am, bm, bounds=(lo2, hi2), method="bvls", max_iter=200).x
    return p, om ** 2 * (c0 - p[0])


def sole_points(m, d, foot):
    """xy of the end points of the foot's sole capsules (the foot's support outline)"""
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == foot and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE and (m.geom_contype[g] | m.geom_conaffinity[g]):
            a = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g, 1]; out += [(d.geom_xpos[g] + a)[:2], (d.geom_xpos[g] - a)[:2]]
    return out
