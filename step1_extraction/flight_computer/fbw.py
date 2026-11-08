"""fbw.py - MW fly-by-wire: a flight computer between a pilot policy (here the distilled MW student) and the Unitree G1.

  pilot command   = the policy's joint targets T (target = offset + scale * action); the pilot is frozen, never retrained
  airframe model  = the robot's full equations of motion: a private copy of the MuJoCo model (contacts, friction, gravity, position
                    actuators), stepped exactly like the control step. check_prior() verifies it (one-step error ~1e-8 here)
  flight envelope = balanced states: capture point near the support (segment between grounded feet), torso upright, pelvis height,
                    bounded spin; plus a predicted-fall term
  protection      = look ahead H control steps with the pilot IN THE LOOP (it reacts to every predicted state). The pilot's own plan
                    is scored first; if its expected envelope cost is <= quiet, the command passes UNCHANGED. Otherwise N-1 smooth
                    corrections are rolled out under S draws of the known actuator-noise level, and the best few are blended (cost +
                    deviation from the pilot), so the pilot is overruled only as much as needed.

Settings come from FBW_* environment variables (CFG below); config D (the full envelope, 58/58) is in run_fbw_D.sh. A law is off when
its weight is 0. FBW_FALL=0 drops the predicted-fall term. stats["parts"] logs, per step, the pilot's cost split by law
[fall, cp, upright, height, spin]. CPU only: MuJoCo's multithreaded rollout.
"""
import copy, os
import numpy as np


def _env(k, d, cast=float):
    return cast(os.environ.get(k, d))


CFG = dict(H=_env("FBW_H", 15, int), N=_env("FBW_N", 64, int), threads=_env("FBW_THREADS", 6, int),
           sig_leg=_env("FBW_SIG_LEG", 0.12), sig_waist=_env("FBW_SIG_WAIST", 0.10), sig_arm=_env("FBW_SIG_ARM", 0.20),
           cp_tol=_env("FBW_CP_TOL", 0.06), up_min=_env("FBW_UP_MIN", 0.85), z_min=_env("FBW_Z_MIN", 0.60), spin_max=_env("FBW_SPIN", 3.0),
           w_cp=_env("FBW_W_CP", 50.0), w_up=_env("FBW_W_UP", 50.0), w_z=_env("FBW_W_Z", 100.0), w_spin=_env("FBW_W_SPIN", 1.0),
           w_fall=_env("FBW_W_FALL", 1e3), S=_env("FBW_S", 4, int), la_noise=_env("FBW_LA_NOISE", 0.0), temp=_env("FBW_TEMP", 0.5), elites=_env("FBW_ELITES", 4, int), iters=_env("FBW_ITERS", 2, int), w_dev=_env("FBW_W_DEV", 1.0), quiet=_env("FBW_QUIET", 0.0), seed=_env("FBW_SEED", 0, int), fall=_env("FBW_FALL", 1, int))


def refs_abs(x, H):
    """reference joint angles for look-ahead steps 1..H from the observation's +1,+2,+4,+8 blocks (linear interp; held after +8)."""
    at = {1: 0, 2: 1, 4: 2, 8: 3}
    blk = {k: x[64 + 64 * b + 35: 64 + 64 * b + 64] for k, b in at.items()}
    out = []
    for k in range(1, H + 1):
        if k in blk:
            out.append(blk[k])
        elif k > 8:
            out.append(blk[8])
        else:
            lo = max(s for s in blk if s < k); hi = min(s for s in blk if s > k); w = (k - lo) / (hi - lo)
            out.append((1 - w) * blk[lo] + w * blk[hi])
    return np.stack(out)


class FlyByWire:
    def __init__(self, sim, dof_names, feet, cfg=None):
        import mujoco
        from mujoco import rollout
        self.mj, self.ro, self.c = mujoco, rollout, dict(CFG, **(cfg or {}))
        m = sim.model
        self.m = copy.deepcopy(m)                                              # the prior: nominal robot, never the world's own data
        self.decim = int(sim.decimation)
        jid = [m.joint(n).id for n in dof_names]
        act_of = {int(m.actuator_trnid[i, 0]): i for i in range(m.nu)}
        self.act = np.array([act_of[j] for j in jid])
        self.lo, self.hi = m.jnt_range[jid, 0].astype(np.float64), m.jnt_range[jid, 1].astype(np.float64)
        self.feet = [m.body(f).id for f in feet]; self.torso = m.body("torso_link").id; self.root = m.body("pelvis").id
        self.spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.ns = mujoco.mj_stateSize(m, self.spec)
        self.datas = [mujoco.MjData(self.m) for _ in range(self.c["threads"])]
        self.df = mujoco.MjData(self.m)                                        # for feature evaluation
        sig = []
        for n in dof_names:
            sig.append(self.c["sig_leg"] if any(s in n for s in ("hip", "knee", "ankle")) else
                       self.c["sig_waist"] if "waist" in n else self.c["sig_arm"])
        self.sig = np.array(sig)
        self.rng = np.random.default_rng(self.c["seed"])
        self.prev = np.zeros((2, len(jid)))                                    # previous best correction (2 knots), warm start
        self.nq, self.nv = m.nq, m.nv
        self.qad = np.array([int(m.jnt_qposadr[j]) for j in jid]); self.dof = np.array([int(m.jnt_dofadr[j]) for j in jid])
        from priors import pd_map
        self.off, self.sc = (t.numpy().astype(np.float64) for t in pd_map())
        self.stats = dict(steps=0, interventions=0, pred_viol_pilot=[], pred_viol_sent=[], dev=[])

    # ------------------------------------------------------------------ physics look-ahead
    def _state(self, d):
        s = np.empty(self.ns); self.mj.mj_getState(self.m, d, s, self.spec); return s

    def _features(self, qpos, qvel):
        """per predicted state: capture-point distance beyond the support, torso up-component, pelvis height, torso spin."""
        mj, m, d = self.mj, self.m, self.df
        d.qpos[:] = qpos; d.qvel[:] = qvel
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d); mj.mj_comVel(m, d); mj.mj_subtreeVel(m, d)
        com, v = d.subtree_com[self.root], d.subtree_linvel[self.root]
        P = np.array([d.xpos[f] for f in self.feet]); fz = P[:, 2]
        ground = P[fz < fz.min() + 0.03][:, :2]
        h = max(com[2] - fz.min(), 0.3)
        cp = com[:2] + v[:2] / np.sqrt(9.81 / h)
        if len(ground) == 1:
            dist = np.linalg.norm(cp - ground[0])
        else:                                                                   # distance to the segment between the grounded feet
            a, b = ground[0], ground[1]; ab = b - a; t = np.clip((cp - a) @ ab / max(ab @ ab, 1e-9), 0, 1)
            dist = np.linalg.norm(cp - (a + t * ab))
        up = d.xmat[self.torso].reshape(3, 3)[2, 2]
        return dist, up, qpos[2], np.linalg.norm(qvel[3:6])

    # ------------------------------------------------------------------ the filter
    def proprio(self, st):
        """observation proprioception [N,64] from MuJoCo states (verified against ProtoMotions' obs: exact): joint angles, joint
        velocities, pelvis angular velocity (local frame), gravity in the torso frame."""
        mj, m, d = self.mj, self.m, self.df
        q = st[:, 1:1 + self.nq]; v = st[:, 1 + self.nq:1 + self.nq + self.nv]
        g = np.empty((len(st), 3))
        for i in range(len(st)):
            d.qpos[:] = q[i]; mj.mj_kinematics(m, d); g[i] = -d.xmat[self.torso].reshape(3, 3)[2]
        return np.concatenate([q[:, self.qad], v[:, self.dof], v[:, 3:6], g], 1), q, v

    def filter(self, d, x, T_pilot, pilot):
        """d = the world's MjData now (read only); x = the 349-d observation; T_pilot = pilot's joint targets now;
        pilot(X [B,349]) -> joint targets [B,29] = the same policy, run CLOSED-LOOP inside the look-ahead on predicted states.
        Candidate 0 = the pilot alone; others = pilot + a smooth correction. Each candidate is scored over S draws of the actuator
        noise at its KNOWN level (a prior: the flight computer knows the turbulence level, not the gusts; same draws for all
        candidates). Sent = pilot + cost-weighted blend of corrections (MPPI), only if the pilot's expected violation > quiet."""
        c, H, J = self.c, self.c["H"], len(T_pilot)
        N = c["N"]
        K = self.rng.standard_normal((N, 2, J)) * self.sig
        K[0] = 0.0; K[1] = self.prev; K[2:N // 2] += self.prev               # half the samples around the last solution
        E = self.rng.standard_normal((c["S"], H, J)) * c["la_noise"] * self.sc   # actuator noise draws (same for every candidate)
        s0 = self._state(d)
        total, viol, pfall, dT = self._score(K[:1], E, s0, x, T_pilot, pilot)   # the pilot's own plan first
        self.stats["steps"] += 1; self.stats["pred_viol_pilot"].append(float(viol[0])); self.stats.setdefault("pfall", []).append(float(pfall[0]))
        self.stats.setdefault("parts", []).append(self._parts[0].copy())
        if viol[0] <= c["quiet"]:                                               # the pilot stays on the manifold: hands off
            self.prev *= 0.5; self.stats["pred_viol_sent"].append(float(viol[0])); self.stats["dev"].append(0.0)
            self.stats.setdefault("pfall_best", []).append(float(pfall[0]))
            return T_pilot, False
        t1, v1, p1, d1 = self._score(K[1:], E, s0, x, T_pilot, pilot)          # not quiet: score the corrections
        total, viol, pfall, dT = (np.concatenate(u) for u in ((total, t1), (viol, v1), (pfall, p1), (dT, d1)))
        for _ in range(c["iters"] - 1):                                         # cross-entropy refinement around the best candidates
            el = np.argsort(total)[:c["elites"]]
            mu = K[el].mean(0); n2 = max(N // 2, c["elites"])
            K2 = mu[None] + self.rng.standard_normal((n2, 2, J)) * self.sig * 0.5
            t2, v2, p2, d2 = self._score(K2, E, s0, x, T_pilot, pilot)
            K = np.concatenate([K, K2]); total = np.concatenate([total, t2]); viol = np.concatenate([viol, v2])
            pfall = np.concatenate([pfall, p2]); dT = np.concatenate([dT, d2])
        self.stats.setdefault("pfall_best", []).append(float(pfall.min()))
        el = np.argsort(total)[:c["elites"]]                                   # blend only the best few (a full MPPI blend averages
        wts = np.zeros(len(K)); z_ = total[el] - total[el].min()                # distinct rescues away - measured 2026-11-07)
        wts[el] = np.exp(-z_ / max(c["temp"] * (z_.max() + 1e-9), 1e-9)); wts /= wts.sum()
        corr0 = (wts[:, None] * dT[:, 0]).sum(0)
        self.prev = np.stack([(wts[:, None] * dT[:, min(1, H - 1)]).sum(0), (wts[:, None] * K[:, 1]).sum(0)])
        self.stats["interventions"] += 1
        self.stats["pred_viol_sent"].append(float((wts * viol).sum())); self.stats["dev"].append(float(np.abs(corr0).mean()))
        return np.clip(T_pilot + corr0, self.lo, self.hi), True

    def _score(self, K, E, s0, x, T_pilot, pilot):
        """roll every candidate correction K [n,2,J] out H steps with the pilot in the loop, under each noise draw E [S,H,J]."""
        from priors import next_obs
        import torch
        c, H, n, J, S = self.c, self.c["H"], len(K), len(T_pilot), len(E)
        w = np.linspace(0, 1, H)[None, :, None]
        dT = (1 - w) * K[:, 0][:, None] + w * K[:, 1][:, None]                  # [n, H, J]
        dTb = np.repeat(dT, S, 0); Eb = np.tile(E, (n, 1, 1)); B = n * S       # batch b = candidate b // S, draw b % S
        X = np.repeat(x[None], B, 0); st = np.repeat(s0[None], B, 0)
        viol = np.zeros(B); fallen = np.zeros(B, bool); parts = np.zeros((B, 5)); stop = np.zeros(B, bool)
        for k in range(H):
            Tp = T_pilot[None].repeat(B, 0) if k == 0 else pilot(X)
            Ts = np.clip(Tp + dTb[:, k] + Eb[:, k], self.lo, self.hi)
            ctrl = np.zeros((B, self.decim, self.m.nu)); ctrl[:, :, self.act] = Ts[:, None]
            out, _ = self.ro.rollout(self.m, self.datas, st, ctrl, persistent_pool=True)
            st = out[:, -1]
            p, q, v = self.proprio(st)
            for i in range(B):
                if stop[i]:
                    continue
                dist, up, z, spin = self._features(q[i], v[i])
                if (z < 0.45 or up < 0.5) and not fallen[i]:                   # predicted fall: the earlier, the worse
                    fallen[i] = True
                    if c["fall"]:
                        parts[i, 0] += c["w_fall"] * (H - k); stop[i] = True; continue
                parts[i, 1:] += (c["w_cp"] * max(dist - c["cp_tol"], 0) ** 2, c["w_up"] * max(c["up_min"] - up, 0) ** 2,
                                 c["w_z"] * max(c["z_min"] - z, 0) ** 2, c["w_spin"] * max(spin - c["spin_max"], 0) ** 2)
            viol = parts.sum(1)
            X = next_obs(torch.as_tensor(X), torch.as_tensor(p), torch.as_tensor((Ts - self.off) / self.sc), torch.as_tensor(self.off),
                         torch.as_tensor(self.sc)).numpy()
        viol = viol.reshape(n, S).mean(1); pfall = fallen.reshape(n, S).mean(1); self._parts = parts.reshape(n, S, 5).mean(1)
        dev = (dT ** 2).mean(axis=(1, 2)) / (self.sig ** 2).mean()
        return viol + c["w_dev"] * dev, viol, pfall, dT

    def reset(self):
        """new episode: drop the warm start (the previous correction belongs to another motion / state)."""
        self.prev[:] = 0.0

    def check_prior(self, d, T):
        """one-step prediction of the world (call before stepping it), for verifying the prior is exact."""
        ctrl = np.zeros((1, self.decim, self.m.nu)); ctrl[:, :, self.act] = T
        st, _ = self.ro.rollout(self.m, self.datas[:1], self._state(d)[None], ctrl)
        return st[0, -1, 1:1 + self.nq].copy()
