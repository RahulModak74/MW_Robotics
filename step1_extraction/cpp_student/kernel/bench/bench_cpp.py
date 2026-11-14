"""bench_cpp.py - time filter() with the C++ kernel: quiet and correction ticks, clean (la_noise 0) and noisy (0.2), per thread count,
for exact mode and fast-mode settings. Also the physics floor: one mj_step of the G1 here."""
import os, time, json, sys
import numpy as np
from bench_setup import *
from fbw_fast import FlyByWireCpp

sim = FakeSim(); d0 = standing(sim, 0.3); st = make_student(); pilot = make_pilot(st)

# physics floor
dd = mujoco.MjData(sim.model); mujoco.mj_copyData(dd, sim.model, d0)
t0 = time.perf_counter()
for _ in range(4000):
    mujoco.mj_step(sim.model, dd)
us = (time.perf_counter() - t0) / 4000 * 1e6
print(f"mj_step G1, 1 ms, one core: {us:.1f} us   (one 0.4 s look-ahead = 400 steps = {0.4*us:.1f} ms)")


def tick_times(env, threads, reps=5):
    for k, v in env.items():
        os.environ[k] = str(v)
    os.environ["FBW_THREADS"] = str(threads)
    f = FlyByWireCpp(sim, DOF_NAMES, FEET, cfg={"threads": threads, **({"N": env["FBW_N"]} if "FBW_N" in env else {})}, student=st)
    for k in ("FBW_LA_DT", "FBW_H_CORR", "FBW_N"):
        os.environ.pop(k, None)
    x = observation(f, d0); T = pilot(x[None])[0]
    out = {}
    for la in (0.0, 0.2):
        f.c["la_noise"] = la
        for mode, quiet in (("quiet", 1e18), ("correct", -1.0)):
            f.c["quiet"] = quiet; f.filter(d0, x, T, pilot)
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); f.filter(d0, x, T, pilot); ts.append(time.perf_counter() - t0)
            out[(la, mode)] = 1e3 * np.median(ts)
    return out


configs = [("exact (config D)", {}),
           ("fast A: dt 5 ms", {"FBW_LA_DT": 0.005}),
           ("fast B: dt 5 ms, N 16", {"FBW_LA_DT": 0.005, "FBW_N": 16}),
           ("fast C: dt 5 ms, N 16, H_corr 10", {"FBW_LA_DT": 0.005, "FBW_N": 16, "FBW_H_CORR": 10}),
           ("fast D: dt 2 ms, N 16, H_corr 10", {"FBW_LA_DT": 0.002, "FBW_N": 16, "FBW_H_CORR": 10})]
rows = []
for name, env in configs:
    for th in (1, os.cpu_count()):
        r = tick_times(env, th)
        rows.append((name, th, r))
        print(f"{name:34s} threads {th}: quiet {r[(0.0,'quiet')]:7.1f} | quiet noisy {r[(0.2,'quiet')]:7.1f} | "
              f"correct {r[(0.0,'correct')]:7.1f} | correct noisy {r[(0.2,'correct')]:7.1f}  ms", flush=True)
json.dump([(n, t, {f"{k[0]}_{k[1]}": v for k, v in r.items()}) for n, t, r in rows], open("bench_cpp.json", "w"), indent=1)
