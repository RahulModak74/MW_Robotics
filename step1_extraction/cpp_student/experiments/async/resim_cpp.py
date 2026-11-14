#!/usr/bin/env python3
"""resim_cpp.py — the MEASURED re-sim of the async supervisor with the C++ (GIL-free) kernel.

Identical to ../ASYNC_APPROACH/integrated_gate.py, but fbw is FlyByWireCpp: solve_ms is the real C++
time, so the modeled landing latency L = ceil(solve_ms/20ms) is ~18 steps, not ~185. This COUNTS
actual survivals (re-simulated), it is not the lead-time estimate. Compare --latency zero (synchronous
fbw_2 anchor) vs --latency sim (C++ async, <20ms heartbeat).

--- original header ---

The guaranteed bounded act_track heartbeat (<20ms, max_cpf=3) PLUS fbw as an asynchronous
supervisor: when the MW gate is open, fbw.filter plans 0.3s ahead on a state snapshot and returns
a joint-space delta; the heartbeat blends that (decayed-by-age) correction into act_track's output.
fbw never runs on the heartbeat, so the <20ms guarantee is untouched.

FIDELITY (deviation from the plan, on purpose): the plan ages corrections by WALL clock, but this
sim runs faster than real-time, so wall-age mis-models staleness. Here latency is modeled in SIM
CONTROL STEPS from the MEASURED solve time: a solve fired at step k with wall cost t_ms lands at
step k + ceil(t_ms/20ms) and is usable only from then on. Deterministic, no thread races, and
exactly the real-time behaviour. The heartbeat time is logged EXCLUDING the fbw solve (which is
off-loop in deployment) so the <20ms guarantee is checked honestly.

Blend:  T = clip( base + w(age) * delta ),  delta = T_fbw - T_student  (fbw's correction to the student)
  --base act     : base = act_track(d,ref)        (the integration: intent carried by act_track)   [default]
  --base student : base = T_student               (reproduces fbw_2 when latency=0, no decay — the Phase C check)

Phases (set via flags):
  C (correctness): --base student --latency zero --k-refresh 0 --max-age 100000  -> must match fbw_2 (9/20 ±1)
  B (realistic)  : --base act    --latency sim  --k-refresh 7 --max-age 20        -> expect 12-15/20
  D (sweep)      : vary --max-age {10,20,30} --k-refresh {3,5,7} --thr {.010,.0136,.017}
"""
import argparse, math, os, sys, time
import numpy as np, torch, mujoco
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ms_vs_ppo_standalone as H
from fbw_fast import FlyByWireCpp
from g1_kin import DOF_NAMES, FEET

DT = H.DT


def run_clip(m, info, clip, student, pilot, fbw, student_fn, ys, args):
    H._reset_to_clip_start(m, (d := mujoco.MjData(m)), clip, info)
    pilot.stance = []; fbw.reset()
    n = int((clip["F"] - 1) / clip["fps"] / DT) - 2
    prev = np.zeros(29, np.float32)
    active, hold, calm = False, 0, 0
    # async supervisor state (sim-step latency model)
    pending = None          # (land_step, delta)  in-flight solve
    current = None          # (delta, done_step)  latest landed correction
    last_fire = -10 ** 9
    hb_ms, solve_ms, lat_steps, wlog, dmax = [], [], [], [], []
    fell = None
    for step in range(n):
        t = step * DT; ref = H.ref_at(clip, t + DT); rg = lambda lag: H.ref_at(clip, t + lag * DT)
        t0 = time.perf_counter()
        obs = H.build_obs(d, info["qad"], info["dof"], info["torso"], rg, prev)
        with torch.no_grad():
            u = student.uq(torch.as_tensor(obs[None], dtype=torch.float32))
        epi = float((u["sig_epi"] / ys).mean())
        T_student = info["off"] + info["sc"] * u["action"].numpy()[0]
        warn = epi > args.thr
        if warn:
            active, hold, calm = True, args.latch, 0
        g = active
        w_applied, dmag = 0.0, 0.0
        if g:
            hold -= 1
            # promote a landed solve
            if pending is not None and step >= pending[0]:
                current = (pending[1], step); pending = None
            T_act = pilot.act_track(d, ref, info["wlo"], info["whi"], {})
            base = T_student if args.base == "student" else T_act
            # decayed correction from the latest landed solve
            delta = np.zeros(29)
            if current is not None:
                age = step - current[1]
                if age < args.max_age:
                    w_applied = 1.0 - age / args.max_age
                    delta = current[0]; dmag = float(np.abs(delta).max())
            T = np.clip(base + w_applied * delta, info["wlo"], info["whi"])
            hb_ms.append((time.perf_counter() - t0) * 1e3)   # heartbeat time EXCLUDES the fbw solve below
            # fire a supervisor solve (worker idle + refresh elapsed); cost modeled as latency, not charged to hb
            if warn and pending is None and (step - last_fire) >= args.k_refresh:
                ts = time.perf_counter()
                T_fbw, inter = fbw.filter(d, obs, T_student, student_fn)
                sms = (time.perf_counter() - ts) * 1e3; solve_ms.append(sms)
                dl = (T_fbw - T_student) if inter else np.zeros(29)
                L = 0 if args.latency == "zero" else max(1, math.ceil(sms / (DT * 1e3)))
                lat_steps.append(L); pending = (step + L, dl); last_fire = step
            calm = calm + 1 if not warn else 0
            if calm >= args.calm and hold <= 0:
                active = False
        else:
            T = T_student; current = None; pending = None; fbw.reset()
            hb_ms.append((time.perf_counter() - t0) * 1e3)
        wlog.append(w_applied); dmax.append(dmag)
        d.ctrl[info["act"]] = T
        for _ in range(info["dec"]):
            mujoco.mj_step(m, d)
        prev = ((T - info["off"]) / info["sc"]).astype(np.float32)
        if d.qpos[2] < 0.45 or (1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2)) < 0.5:
            fell = step; break
    return dict(passed=fell is None, steps=step + 1, of=n, fell=fell,
                hb=np.array(hb_ms), solve=np.array(solve_ms), lat=np.array(lat_steps),
                w=np.array(wlog), dmax=np.array(dmax))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--json", required=True)
    ap.add_argument("--student", required=True); ap.add_argument("--clips", required=True)
    ap.add_argument("--thr", type=float, default=0.0136)
    ap.add_argument("--latch", type=int, default=10); ap.add_argument("--calm", type=int, default=10)
    ap.add_argument("--max-cpf", type=int, default=3)
    ap.add_argument("--base", choices=["act", "student"], default="act")
    ap.add_argument("--latency", choices=["sim", "zero"], default="sim")
    ap.add_argument("--k-refresh", type=int, default=7, help="min sim steps between fbw solves")
    ap.add_argument("--max-age", type=int, default=20, help="correction decays to 0 over this many sim steps")
    ap.add_argument("--budget-ms", type=float, default=20.0)
    a = ap.parse_args()

    m, info = H.load_env(a.model, a.json); info["ground"] = True
    import student_prob; s = student_prob.load(a.student); ys = H.action_sd(s)
    pilot = H.make_pilot(m, info, 30., 20., .5); pilot.max_cpf = a.max_cpf
    fbw = FlyByWireCpp(H.Sim(m, mujoco.MjData(m), info["dec"]), DOF_NAMES, FEET, student=s)  # C++ GIL-free kernel
    student_fn = lambda X: H._student_action_from_obs(X, s, info["off"], info["sc"])
    clips = [ln.strip() for ln in open(a.clips) if ln.strip()]

    print(f"base={a.base} latency={a.latency} k_refresh={a.k_refresh} max_age={a.max_age} thr={a.thr} max_cpf={a.max_cpf}\n")
    rows, allhb = [], []
    for p in clips:
        clip = H.load_clip(p, m, info["qad"], info["dof"], info["pelvis"], info["torso"], info["feet"])
        r = run_clip(m, info, clip, s, pilot, fbw, student_fn, ys, a)
        rows.append((os.path.basename(p), r)); allhb.append(r["hb"])
        sv = r["solve"]; lat = r["lat"]
        print(f"[{os.path.basename(p):22s}] {'PASS' if r['passed'] else 'FAIL':4s} {r['steps']:4d}/{r['of']}  "
              f"hb med {np.median(r['hb']):.2f} max {r['hb'].max():.1f}ms  "
              f"solves {len(sv):3d} (lat med {int(np.median(lat)) if lat.size else 0} st)  "
              f"w~{r['w'][r['w']>0].mean() if (r['w']>0).any() else 0:.2f}  |delta|max {r['dmax'].max():.3f}", flush=True)
    HB = np.concatenate(allhb)
    npass = sum(r["passed"] for _, r in rows)
    over = (HB > a.budget_ms).mean()
    print("\n" + "=" * 72)
    print(f"PASS RATE: {npass}/{len(rows)}   (act_track-only 3/20 · fbw_2 synchronous 9/20, same clips)")
    print(f"HEARTBEAT (act_track path, fbw off-loop): median {np.median(HB):.2f}  p99 {np.percentile(HB,99):.1f}  max {HB.max():.1f} ms")
    print(f"  steps OVER {a.budget_ms:.0f} ms: {100*over:.2f}%   -> {'HARD <20ms HOLDS' if over==0 else 'BREACH'}")


if __name__ == "__main__":
    main()
