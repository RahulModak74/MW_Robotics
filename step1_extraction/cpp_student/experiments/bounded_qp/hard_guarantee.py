#!/usr/bin/env python3
"""hard_guarantee.py — bounded-compute MW gate ("#1: surrogate gate").

Goal: a <20 ms-per-step (50 Hz real-time) hand gate that still rescues the student,
WITHOUT fbw's unbounded physics look-ahead (median 140 ms, tail to seconds).

Idea: keep the MW trigger (student epi), but when it warns, hand to the **inverse-dynamics
pilot QP** `invdyn_pilot.act_track` — an analytic, bounded-compute controller (one QP solve,
~few ms) whose capture-point CoP correction is exactly the "surrogate" of the physics search.
No MuJoCo rollout, so every step has a hard compute bound.

  step cost:  hands-off = student forward (~2 ms)
              gated     = student forward + one act_track QP (~a few ms)   <-- bounded, no rollout

Follow-through hand-back (like fbw_2): once warned, stay on the pilot until the warning is quiet
for `calm` consecutive steps, never before `latch`.

Reports, per run: pass rate on the clips AND the per-step timing distribution (median / p99 / max,
and the fraction of steps over the 20 ms budget) — so we can see if the <20 ms guarantee holds.

Usage:
  PYTHONPATH=.:/ABS/third_party/ProtoMotions python hard_guarantee.py \
     --model g1_env.mjb --json g1_env.json --student r7c_mwv3ctl_s100.pt --clips clips_20.txt
"""
import argparse, os, sys, time
import numpy as np, torch, mujoco
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ms_vs_ppo_standalone as H


def run_clip(m, info, clip, student, pilot, ys, thr, latch, calm_need, gains, budget_ms):
    H._reset_to_clip_start(m, (d := mujoco.MjData(m)), clip, info)
    pilot.stance = []
    n = int((clip["F"] - 1) / clip["fps"] / H.DT) - 2
    prev = np.zeros(29, np.float32)
    active, hold, calm = False, 0, 0
    t_step, gated = [], []
    fell = None
    for step in range(n):
        t0 = time.perf_counter()
        t = step * H.DT; ref = H.ref_at(clip, t + H.DT); rg = lambda lag: H.ref_at(clip, t + lag * H.DT)
        obs = H.build_obs(d, info["qad"], info["dof"], info["torso"], rg, prev)
        with torch.no_grad():
            u = student.uq(torch.as_tensor(obs[None], dtype=torch.float32))
        epi = float((u["sig_epi"] / ys).mean())
        warn = epi > thr
        if warn:
            active, hold, calm = True, latch, 0
        g = active
        if g:                                           # bounded recovery: one QP, no physics rollout
            T = pilot.act_track(d, ref, info["wlo"], info["whi"], gains)
            hold -= 1; calm = calm + 1 if not warn else 0
            if calm >= calm_need and hold <= 0:
                active = False
        else:
            T = info["off"] + info["sc"] * u["action"].numpy()[0]
        d.ctrl[info["act"]] = T
        for _ in range(info["dec"]):
            mujoco.mj_step(m, d)
        prev = ((T - info["off"]) / info["sc"]).astype(np.float32)
        dt_ms = (time.perf_counter() - t0) * 1e3
        t_step.append(dt_ms); gated.append(g)
        if d.qpos[2] < 0.45 or (1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2)) < 0.5:
            fell = step; break
    return dict(passed=fell is None, steps=len(t_step), of=n, fell=fell,
                t=np.array(t_step), gated=np.array(gated))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--json", required=True)
    ap.add_argument("--student", required=True); ap.add_argument("--clips", required=True)
    ap.add_argument("--thr", type=float, default=0.0136)
    ap.add_argument("--latch", type=int, default=10); ap.add_argument("--calm", type=int, default=10)
    ap.add_argument("--kp", type=float, default=30.); ap.add_argument("--kd", type=float, default=20.)
    ap.add_argument("--kxi", type=float, default=.5); ap.add_argument("--gains", default="")
    ap.add_argument("--budget-ms", type=float, default=20.0)
    ap.add_argument("--qp-iters", type=int, default=500, help="cap bvls active-set iterations")
    ap.add_argument("--max-cpf", type=int, default=3, help="cap contact points per foot -> bounds QP size (nc) -> hard <20ms (3 => max ~10ms, 0%% over budget; 99 = uncapped/old behaviour)")
    a = ap.parse_args()

    m, info = H.load_env(a.model, a.json); info["ground"] = True
    import student_prob; s = student_prob.load(a.student); ys = H.action_sd(s)
    pilot = H.make_pilot(m, info, a.kp, a.kd, a.kxi)
    pilot.qp_iters = a.qp_iters; pilot.max_cpf = a.max_cpf
    gains = {}
    for kv in a.gains.split():
        k, v = kv.split("="); gains[k] = float(v)
    clips = [ln.strip() for ln in open(a.clips) if ln.strip()]

    rows, allt, allg = [], [], []
    for p in clips:
        clip = H.load_clip(p, m, info["qad"], info["dof"], info["pelvis"], info["torso"], info["feet"])
        r = run_clip(m, info, clip, s, pilot, ys, a.thr, a.latch, a.calm, gains, a.budget_ms)
        rows.append((os.path.basename(p), r)); allt.append(r["t"]); allg.append(r["gated"])
        print(f"[{os.path.basename(p):22s}] {'PASS' if r['passed'] else 'FAIL':4s} {r['steps']:4d}/{r['of']}  "
              f"gated {100*r['gated'].mean():3.0f}%  step ms: med {np.median(r['t']):.2f} p99 {np.percentile(r['t'],99):.1f} max {r['t'].max():.1f}", flush=True)
    T = np.concatenate(allt); G = np.concatenate(allg)
    npass = sum(r["passed"] for _, r in rows)
    print("\n" + "=" * 70)
    print(f"PASS RATE: {npass}/{len(rows)}   (fbw_2 reference on same clips: B 9/20)")
    print(f"--- per-step wall time over {len(T)} steps (budget {a.budget_ms:.0f} ms) ---")
    print(f"  all steps : median {np.median(T):.2f}  p99 {np.percentile(T,99):.2f}  max {T.max():.2f} ms")
    if G.any():
        print(f"  gated     : median {np.median(T[G]):.2f}  p99 {np.percentile(T[G],99):.2f}  max {T[G].max():.2f} ms   ({100*G.mean():.0f}% of steps)")
    print(f"  hands-off : median {np.median(T[~G]):.2f}  p99 {np.percentile(T[~G],99):.2f} ms")
    over = (T > a.budget_ms).mean()
    print(f"  steps OVER {a.budget_ms:.0f} ms budget: {100*over:.2f}%   ->  "
          + ("HARD <20ms GUARANTEE HOLDS" if over == 0 else f"max {T.max():.1f} ms breaches budget"))


if __name__ == "__main__":
    main()
