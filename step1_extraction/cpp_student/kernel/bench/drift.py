"""drift.py - what fast mode costs in fidelity. On world states along noisy trajectories (standing, sagging, falling), compare the
look-ahead with a coarse physics step against the exact 1 ms one:
  * the pilot plan's envelope cost: pass/correct decision at quiet = 1.0, fall agreement
  * corrections: overlap of the 4 elites, and the exact-model cost of the blend each mode would send (regret)
  * one-step (20 ms) prediction error of the robot's qpos vs the world (fbw.py's check_prior, 1e-7 when exact)."""
import os, sys
import numpy as np
from bench_setup import *
from fbw_fast import FlyByWireCpp

sim = FakeSim(); st = make_student(); pilot = make_pilot(st)
modes = {"exact 1 ms": 0.0, "2 ms": 0.002, "5 ms": 0.005, "5 ms N16 H10": 0.005}
F = {}
for name, dt in modes.items():
    os.environ["FBW_LA_DT"] = str(dt)
    os.environ["FBW_H_CORR"] = "10" if "H10" in name else "20"
    F[name] = FlyByWireCpp(sim, DOF_NAMES, FEET, cfg={"threads": os.cpu_count()}, student=st)
os.environ.pop("FBW_LA_DT")
ex = F["exact 1 ms"]
rng = np.random.default_rng(0)

states = []
for traj in range(int(sys.argv[1]) if len(sys.argv) > 1 else 6):
    d = standing(sim, 0.2 + 0.1 * traj)
    noise = 0.05 * traj
    for t in range(40):
        x = observation(ex, d); T = pilot(x[None])[0]
        if t % 4 == 0:
            states.append((ex._state(d).copy(), x.copy(), T.copy(), np.array(d.qacc_warmstart)))
        d.ctrl[ex.act] = np.clip(T + noise * rng.standard_normal(29) * SC, ex.lo, ex.hi)
        for _ in range(sim.decimation):
            mujoco.mj_step(sim.model, d)
        if d.qpos[2] < 0.3:
            break
print(f"{len(states)} world states")

res = {n: dict(dec=0, fall=0, viol_rel=[], elite=[], regret=[], onestep=[]) for n in modes}
for s0, x, T, w in states:
    K = rng.standard_normal((64, 2, 29)) * ex.sig; K[0] = 0
    E = np.zeros((1, 20, 29))
    out = {}
    for n, f in F.items():
        f._warm = w
        tp, vp, pp, _ = f._score(K[:1], E, s0, x, T, pilot)
        KK = K[1:16] if "N16" in n else K[1:]
        tc, vc, pc, dT = f._score(KK, E, s0, x, T, pilot)
        out[n] = (vp[0], pp[0], tc, np.argsort(tc)[:4])
    tref = out["exact 1 ms"][2]
    for n in modes:
        vp, pp, tc, el = out[n]
        r = res[n]
        r["dec"] += (vp > 1.0) == (out["exact 1 ms"][0] > 1.0)
        r["fall"] += pp == out["exact 1 ms"][1]
        r["viol_rel"].append(abs(vp - out["exact 1 ms"][0]) / max(out["exact 1 ms"][0], 1.0))
        r["elite"].append(len(set(el.tolist()) & set(out["exact 1 ms"][3].tolist())) / 4)
        r["regret"].append(tref[el[0]] - tref.min())                         # exact cost of this mode's pick - exact best
for n in modes:
    r = res[n]; N = len(states)
    print(f"{n:10s}: pass/correct decision agrees {r['dec']}/{N} | fall flag agrees {r['fall']}/{N} | pilot cost rel. err "
          f"median {np.median(r['viol_rel']):.3f} max {np.max(r['viol_rel']):.3f} | elite overlap {np.mean(r['elite']):.2f} | "
          f"pick regret median {np.median(r['regret']):.3f}, >1 in {np.sum(np.array(r['regret'])>1)}/{N}, "
          f"pick falls under exact model in {np.sum(np.array(r['regret'])>100)}/{N}")

# one-step (20 ms) prediction error vs the 1 ms world, at the same states and the pilot's command
modes.pop("5 ms N16 H10"); errs = {n: [] for n in modes}
for s0, x, T, w in states:
    ref = None
    for n, dt in modes.items():
        import copy
        mm = copy.deepcopy(sim.model)
        if dt:
            mm.opt.timestep = dt
        dd = mujoco.MjData(mm); mujoco.mj_setState(mm, dd, s0, mujoco.mjtState.mjSTATE_FULLPHYSICS); dd.qacc_warmstart[:] = w
        dd.ctrl[ex.act] = T
        for _ in range(int(round(0.02 / mm.opt.timestep))):
            mujoco.mj_step(mm, dd)
        q = dd.qpos.copy()
        if ref is None:
            ref = q
        errs[n].append(np.abs(q - ref).max())
for n in modes:
    print(f"{n:10s}: one-step (20 ms) qpos error vs 1 ms world: median {np.median(errs[n]):.1e} max {np.max(errs[n]):.1e}")
