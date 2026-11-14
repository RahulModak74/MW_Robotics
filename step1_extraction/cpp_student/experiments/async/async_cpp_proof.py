#!/usr/bin/env python3
"""async_cpp_proof.py — the claim this bundle makes: with the C++ (GIL-free) kernel on a worker
thread, the Python control heartbeat stays <20 ms WHILE the ~260-370 ms correction search runs
truly in parallel, and the correction lands ~19 control steps after the warning.

Main loop each control step (real-time paced at 20 ms): run a realistic heartbeat (student forward,
~1-2 ms) + time it. At the first gate trip, submit the state to the worker. Keep ticking; record
when the correction lands and the heartbeat tick-time distribution DURING the worker-busy window.

Run (PYTHONPATH needs cpp/, ours/, ProtoMotions; CPU in `performance` governor for real numbers):
    FBW_THREADS=6 python async_cpp_proof.py \
        --student students/r7c_mwv3ctl_s100.pt \
        --motion g1_kimodo_data/m500/proto_train/exercise_009_04.motion
"""
import argparse, os, sys, time
import numpy as np, torch, mujoco
torch.set_num_threads(1)
# repo helpers (ms_vs_ppo_standalone) + cpp on path are expected via PYTHONPATH
import ms_vs_ppo_standalone as H
from g1_kin import DOF_NAMES, FEET
from async_fbw_cpp import AsyncFBWCpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protomotions_g1_distillation/eval_prob/stand/g1_env.mjb")
    ap.add_argument("--json", default="protomotions_g1_distillation/eval_prob/stand/g1_env.json")
    ap.add_argument("--student", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--thr", type=float, default=0.0136)
    ap.add_argument("--budget-ms", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=120)
    a = ap.parse_args()

    m, info = H.load_env(a.model, a.json); info["ground"] = True
    import student_prob; s = student_prob.load(a.student); ys = H.action_sd(s)
    student_fn = lambda X: H._student_action_from_obs(X, s, info["off"], info["sc"])

    sup = AsyncFBWCpp(H.Sim(m, mujoco.MjData(m), info["dec"]), DOF_NAMES, FEET, student=s)
    sup.set_pilot(student_fn)

    clip = H.load_clip(a.motion, m, info["qad"], info["dof"], info["pelvis"], info["torso"], info["feet"])
    d = mujoco.MjData(m); H._reset_to_clip_start(m, d, clip, info); prev = np.zeros(29, np.float32)
    budget = a.budget_ms / 1e3
    submitted = False; submit_t = submit_step = land_t = land_step = solve_ms = None
    busy_ticks, idle_ticks = [], []

    for step in range(a.steps):
        t = step * H.DT; rg = lambda lag: H.ref_at(clip, t + lag * H.DT)
        obs = H.build_obs(d, info["qad"], info["dof"], info["torso"], rg, prev)
        tb = time.perf_counter()                                    # ---- HEARTBEAT (<20 ms) ----
        with torch.no_grad():
            u = s.uq(torch.as_tensor(obs[None], dtype=torch.float32))
        epi = float((u["sig_epi"] / ys).mean()); T = info["off"] + info["sc"] * u["action"].numpy()[0]
        tick_ms = (time.perf_counter() - tb) * 1e3
        if (not submitted) and epi > a.thr:
            sup.submit(d, obs, T); submitted = True; submit_t = time.perf_counter(); submit_step = step
        if submitted and land_t is None and sup._result is not None:   # correction landed
            land_t = time.perf_counter(); land_step = step
            solve_ms = sup._result[2] * 1e3
        (busy_ticks if (submitted and land_t is None) else idle_ticks).append(tick_ms)
        d.ctrl[info["act"]] = T
        for _ in range(info["dec"]):
            mujoco.mj_step(m, d)
        prev = ((T - info["off"]) / info["sc"]).astype(np.float32)
        time.sleep(max(0.0, budget - (time.perf_counter() - tb)))   # real-time pacing
    sup.close()

    def stat(x):
        x = np.array(x)
        return (f"n={len(x):3d}  median={np.median(x):5.2f}  p99={np.percentile(x,99):5.2f}  MAX={x.max():6.2f} ms"
                if len(x) else "n=0")
    print(f"motion: {os.path.basename(a.motion)}  threads={os.environ.get('FBW_THREADS','6')}")
    if solve_ms is not None:
        print(f"C++ correction solve        : {solve_ms:.0f} ms")
        print(f"latency submit->land        : {(land_t-submit_t)*1e3:.0f} ms = {land_step-submit_step} control steps @{a.budget_ms:.0f}ms")
    else:
        print("correction did not land within the run window")
    print(f"HEARTBEAT while worker busy  : {stat(busy_ticks)}")
    print(f"HEARTBEAT idle               : {stat(idle_ticks)}")
    ok = busy_ticks and max(busy_ticks) < a.budget_ms
    print(f"=> main loop max during busy window {'< budget  PASS' if ok else 'OVER budget'}")


if __name__ == "__main__":
    main()
