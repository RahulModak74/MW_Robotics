"""bench_python.py - time the ORIGINAL fbw.py filter() on the G1 stand-in: quiet ticks and forced-correction ticks."""
import sys, time
import numpy as np
from bench_setup import *
from fbw import FlyByWire

sim = FakeSim(); d = standing(sim, 0.3)
print("model nq", sim.model.nq, "nv", sim.model.nv, "nu", sim.model.nu, "ncon after settle", d.ncon, "pelvis z", round(d.qpos[2], 3))
st = make_student(); pilot = make_pilot(st)
f = FlyByWire(sim, DOF_NAMES, FEET)
x = observation(f, d); T = pilot(x[None])[0]
print("cfg threads", f.c["threads"], "H", f.c["H"], "N", f.c["N"], "S", f.c["S"])

t0 = time.perf_counter(); T2, inter = f.filter(d, x, T, pilot); t1 = time.perf_counter()
print(f"tick 1: intervened={inter} viol={f.stats['pred_viol_pilot'][-1]:.4f}  {1e3*(t1-t0):.1f} ms")
f.c['quiet'] = 1e18; qt = []
for _ in range(3):
    t0 = time.perf_counter(); f.filter(d, x, T, pilot); qt.append(time.perf_counter() - t0)
print(f"quiet-path ticks (S={f.c['S']}): median {1e3*np.median(qt):.1f} ms")
f.c["quiet"] = -1.0
ct = []
for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 1):
    t0 = time.perf_counter(); f.filter(d, x, T, pilot); ct.append(time.perf_counter() - t0)
print(f"correction ticks: median {1e3*np.median(ct):.0f} ms")
