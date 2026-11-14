"""verify.py - the C++ look-ahead against the original Python one, on identical inputs.
  1. the student's action: kernel MLP vs torch uq()["action"] on 500 observations
  2. _score for the pilot's plan and for 63 corrections, clean (la_noise 0) and noisy (la_noise 0.2, S=4)
  3. filter(): the sent command and the intervention decision, over 20 ticks of a moving robot."""
import numpy as np
from bench_setup import *
from fbw import FlyByWire
from fbw_fast import FlyByWireCpp

sim = FakeSim(); d = standing(sim, 0.3); st = make_student(); pilot = make_pilot(st)
fp = FlyByWire(sim, DOF_NAMES, FEET); fc = FlyByWireCpp(sim, DOF_NAMES, FEET, student=st)
x = observation(fp, d); T = pilot(x[None])[0]

# 1. pilot
rng = np.random.default_rng(0)
X = x[None] + rng.standard_normal((500, 349)) * 0.3
e = np.abs(fc.k.pilot(X) - pilot(X)).max()
print(f"[1] pilot action: max |C++ - torch| = {e:.2e} rad (float32 MLP)")

# 2. _score
s0 = fp._state(d)
for la in (0.0, 0.2):
    K = rng.standard_normal((64, 2, 29)) * fp.sig; K[0] = 0
    E = rng.standard_normal((4, 20, 29)) * la * fp.sc
    for name, KK in (("pilot plan", K[:1]), ("63 corrections", K[1:])):
        tp, vp, pp, _ = fp._score(KK, E, s0, x, T, pilot); partsp = fp._parts
        tc, vc, pc, _ = fc._score(KK, E, s0, x, T, pilot); partsc = fc._parts
        rel = np.abs(vc - vp).max() / max(np.abs(vp).max(), 1e-9)
        same_best = np.argsort(tp)[:4].tolist() == np.argsort(tc)[:4].tolist()
        print(f"[2] la_noise {la} {name:15s}: viol py {vp.min():8.4f}..{vp.max():8.4f} | max |diff| {np.abs(vc-vp).max():.2e}"
              f" (rel {rel:.1e}) | pfall equal {np.array_equal(pp, pc)} | same 4 elites {same_best}")

# 3. filter() over 20 ticks while the world moves (the robot sags under the default pose: real interventions)
fp2 = FlyByWire(sim, DOF_NAMES, FEET); fc2 = FlyByWireCpp(sim, DOF_NAMES, FEET, student=st)
agree, dmax = 0, 0.0
for t in range(20):
    x = observation(fp2, d); T = pilot(x[None])[0]
    Tp, ip = fp2.filter(d, x, T, pilot); Tc, ic = fc2.filter(d, x, T, pilot)
    agree += ip == ic; dmax = max(dmax, np.abs(Tp - Tc).max())
    d.ctrl[fp2.act] = Tp
    for _ in range(sim.decimation):
        mujoco.mj_step(sim.model, d)
print(f"[3] 20 ticks: same decision {agree}/20, interventions py {fp2.stats['interventions']} cpp {fc2.stats['interventions']}, "
      f"max |sent command diff| {dmax:.2e} rad")
