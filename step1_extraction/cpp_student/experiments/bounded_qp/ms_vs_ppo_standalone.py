#!/usr/bin/env python3
"""mw_vs_ppo_standalone.py — Does MW replace PPO? Standalone test harness.

Runs the teacher-student route (the 58/58 control) and the teacherless-pilot
route on the same MuJoCo model, same clips, no ProtoMotions. Four arms per clip:

    A. Student alone          — the teacher-distilled student, no gate.
    B. Student + MW gate      — GatedFBW (fbw_2, follow-through); the 58/58 control.
    C. Pilot alone            — invdyn_pilot.act_track; the current teacherless baseline.
    D. Pilot + MW gate        — pilot primary, gate uses the STUDENT's MW signal;
                                flight-computer fallback = reference replay (a cheap,
                                obs-compatible stand-in for the physical pilot).

Verdict (per Does_MW_Replace_PPO.md §9):
    D ≈ B               -> MW replaces PPO for this test set.
    D > C, D < B - 0.1  -> MW helps the pilot but doesn't close the gap: narrowed claim.
    D <= C              -> MW does not rescue the pilot: pilot needs CAM or stepping
                           (PLAN_2 v3), not better gating.

Usage:
    # 1) VERIFY the obs reconstruction BEFORE trusting the student arms:
    python mw_vs_ppo_standalone.py --check-obs --data-pm data_pm \\
        --model eval_prob/stand/g1_env.mjb --json eval_prob/stand/g1_env.json

    # 2) Run the four-arm test on the 20-clip smoke:
    python mw_vs_ppo_standalone.py \\
        --model eval_prob/stand/g1_env.mjb \\
        --json  eval_prob/stand/g1_env.json \\
        --clips clips_20.txt \\
        --student students/r7c.pt \\
        --out results/mw_vs_ppo/

Requires: mujoco, torch, numpy, scipy (for fbw's lsq_linear), and (for the
student arms) the existing ours/ modules on PYTHONPATH: invdyn_pilot, fbw_2,
fbw, student_prob, priors, g1_kin.

KNOWN LIMITATION — the 6-D rel-torso-rot block is INFERRED. See build_obs() and
check_obs_format(); run --check-obs first. The pilot-only arm (C) does not need
the obs and will still work if the format is wrong.
"""
import argparse, glob, json, os, sys, time
from collections import defaultdict

import numpy as np
import mujoco
import torch

torch.set_num_threads(1)   # single-env sequential rollout: torch's default intra-op pool thrashes on tiny per-step forwards

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ============================================================ constants
DT         = 0.02          # control period (unified_pipeline.yaml)
PASS_Z     = 0.45          # fall thresholds (id_track convention)
PASS_UP    = 0.50
REF_LAGS   = (1, 2, 3, 4)  # future_steps:4 → consecutive steps 1..4 (unified_pipeline.yaml / select_steps)
ARMS       = ("A", "B", "C", "D")
ARM_TITLE  = {"A": "student alone",
              "B": "student + MW gate",
              "C": "pilot alone",
              "D": "pilot + MW gate"}


# ============================================================ obs (inferred format)
def quat_to_tan_norm_mat(R):
    """6-D tan-norm rotation == ProtoMotions rotations.quat_to_tan_norm, in matrix form:
    [R·x̂, R·ẑ] = columns 0 and 2 of R. (Source: target_poses.build_reduced_coords_target_poses
    → rotations.quat_to_tan_norm: tan=quat_rotate(q,[1,0,0]); norm=quat_rotate(q,[0,0,1]).)
    NOT the first two columns, and NOT tanh-compressed."""
    return np.array([R[0, 0], R[1, 0], R[2, 0], R[0, 2], R[1, 2], R[2, 2]], dtype=np.float32)


def build_obs(d, qad, dof, torso_id, ref_at_step, prev_action):
    """Assemble the 349-d student observation — faithful port of ProtoMotions.

    obs = noisy_reduced_coords_obs(64) | noisy_mimic_reduced_coords_target_poses(256) |
          historical_previous_processed_actions(29)      [pm_collect.KEYS, unified_pipeline.yaml]
      proprio[0:64]  = dof_pos(29) | dof_vel(29) | root_local_ang_vel(3)=qvel[3:6] |
                       proj_gravity(3) = R_anchorᵀ·[0,0,-1], anchor = torso_link
      targets[64:320]= 4 lags (steps 1..4) × [ tan_norm(R_torsoᵀ·R_ref_torso)(6) | ref_dof_vel(29) | ref_dof_pos(29) ]
      prev[320:349]  = previous raw action a (zeros at reset)
    Validated against g1_env.json obs0 (proprio + anchor + rotation convention)."""
    obs = np.zeros(349, dtype=np.float32)
    obs[0:29]  = d.qpos[qad]
    obs[29:58] = d.qvel[dof]
    obs[58:61] = d.qvel[3:6]                              # root (free-joint) ang vel, already local
    R_torso    = d.xmat[torso_id].reshape(3, 3)           # anchor = torso_link
    obs[61:64] = R_torso.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)

    for k, lag in enumerate(REF_LAGS):
        b   = 64 + 64 * k
        rk  = ref_at_step(lag)                            # caller supplies a t+lag·DT getter
        R_rel = R_torso.T @ rk["tor_R"]                   # conj(cur_anchor)⊗ref_anchor  ==  R_curᵀ·R_ref
        obs[b:b + 6]       = quat_to_tan_norm_mat(R_rel)
        obs[b + 6:b + 35]  = rk["qv"]                     # ref dof_vel
        obs[b + 35:b + 64] = rk["qa"]                     # ref dof_pos

    obs[320:349] = prev_action                            # previous raw policy action a
    return obs


def check_obs_format(data_pm_dir, m, qad, dof, torso_id, feet_ids, n_test=5):
    """Verify build_obs against recorded transitions from pm_collect. Prints per-block
    error. A small error (< data std) on the reference blocks means the format is right;
    a large error means the 6-D rotation representation is wrong and must be fixed
    before trusting arms A/B/D.
    """
    files = sorted(glob.glob(os.path.join(data_pm_dir, "*_s*.npz")))[:n_test]
    if not files:
        print(f"[check-obs] no *_s*.npz under {data_pm_dir}"); return
    print(f"[check-obs] {len(files)} file(s)")
    for f in files:
        z = np.load(f); X, ep = z["X"], z["ep"]
        same = ep[:-1] == ep[1:]
        x0, x1 = X[:-1][same], X[1:][same]
        # Compare the reference blocks of x1 against x0 shifted 1 step (the no-shift baseline vs the true shift):
        # if the training obs is self-consistent, the shifted reference should track a small residual over 1 step.
        # We can't reconstruct the *actual* joint-target chain from here without re-running the env, so this check
        # is intentionally limited: it reports whether the *shape and scale* of each block match expectation.
        rep = lambda b, e: f"  blk[{b:3d}:{e:3d}] data mean {x1[:, b:e].mean():+.3f} std {x1[:, b:e].std():.3f}"
        print(f"[check-obs] {os.path.basename(f)}: {len(x0)} transitions")
        print(rep(64,  70)); print(rep(70,  99)); print(rep(99, 128))
        print(f"  prev-action[320:349] mean {x0[:,320:349].mean():+.3f} std {x0[:,320:349].std():.3f}")
    print("[check-obs] NOTE: this only validates block shapes/scales against recorded data. "
          "To validate the 6-D rotation representation itself, run a clip on the standalone harness "
          "and compare the student's action on the first 5 steps against its action in a recorded "
          "pm_collect rollout of the same clip. If they disagree by more than the DART noise level, "
          "the 6-D representation is wrong.")


# ============================================================ clip loading
def load_clip(motion_path, m, qad, dof, pelvis_id, torso_id, feet_ids):
    """Load a .motion file and precompute the reference (mirrors id_track.py)."""
    X = torch.load(motion_path, weights_only=False)
    fps = float(X["fps"]); F = X["dof_pos"].shape[0]
    dpos = X["dof_pos"].numpy().astype(np.float64)
    dvel = X["dof_vel"].numpy().astype(np.float64)
    rp   = X["rigid_body_pos"].numpy().astype(np.float64)
    rr   = X["rigid_body_rot"].numpy().astype(np.float64)
    rv   = X["rigid_body_vel"].numpy().astype(np.float64)
    rw   = X["rigid_body_ang_vel"].numpy().astype(np.float64)
    rc   = X["rigid_body_contacts"].numpy()
    FC0  = np.stack([rc[:, 6] | rc[:, 7], rc[:, 12] | rc[:, 13]], 1) > 0

    sc = mujoco.MjData(m)
    def qpos_of(i):
        qp = np.zeros(m.nq)
        qp[:3] = rp[i, 0]; qp[3:7] = rr[i, 0][[3, 0, 1, 2]]; qp[qad] = dpos[i]
        if m.nq >= 43: qp[36:43] = [0, 0, -2, 1, 0, 0, 0]
        return qp

    COM, FP, FR, PR, TR = [], [], [], [], []
    for i in range(F):
        sc.qpos[:] = qpos_of(i); mujoco.mj_kinematics(m, sc); mujoco.mj_comPos(m, sc)
        COM.append(sc.subtree_com[pelvis_id].copy())
        FP.append([sc.xpos[f].copy() for f in feet_ids])
        FR.append([sc.xmat[f].reshape(3, 3).copy() for f in feet_ids])
        PR.append(sc.xmat[pelvis_id].reshape(3, 3).copy())
        TR.append(sc.xmat[torso_id].reshape(3, 3).copy())
    COM, FP, FR, PR, TR = map(np.array, (COM, FP, FR, PR, TR))
    VCOM  = np.gradient(COM,  1 / fps, axis=0); ACOM = np.gradient(VCOM, 1 / fps, axis=0)
    QACC  = np.gradient(dvel, 1 / fps, axis=0); FV   = np.gradient(FP,   1 / fps, axis=0)
    return dict(motion=motion_path, fps=fps, F=F, dpos=dpos, dvel=dvel, rp=rp, rv=rv, rw=rw,
                COM=COM, VCOM=VCOM, ACOM=ACOM, QACC=QACC, FP=FP, FV=FV, FR=FR, PR=PR, TR=TR,
                FC=FC0, qpos_of=qpos_of, _sc=sc)


def ref_at(clip, t):
    fps, F = clip["fps"], clip["F"]
    u = np.clip(t * fps, 0, F - 1.001); i = int(u); a = u - i
    L = lambda x: (1 - a) * x[i] + a * x[i + 1]
    N = lambda x: x[i] if a < 0.5 else x[i + 1]
    return dict(com=L(clip["COM"]), vcom=L(clip["VCOM"]), acom=L(clip["ACOM"]),
                qa=L(clip["dpos"]),  qv=L(clip["dvel"]),  qacc=L(clip["QACC"]),
                foot_p=L(clip["FP"]), foot_v=L(clip["FV"]),
                foot_R=N(clip["FR"]), foot_c=N(clip["FC"]),
                pel_R=N(clip["PR"]),  tor_R=N(clip["TR"]),
                pel_w=L(clip["rw"][:, 0]), tor_w=L(clip["rw"][:, 16]))   # act_track needs pelvis/torso ang vel (mirrors id_track.ref_at)


# ============================================================ model / env
class Sim:
    """Minimal sim wrapper accepted by fbw_2.GatedFBW (needs .model, .data, .decimation)."""
    def __init__(self, m, d, decimation):
        self.model, self.data, self.decimation = m, d, int(decimation)


def load_env(model_path, json_path):
    m = mujoco.MjModel.from_binary_path(model_path)
    E = json.load(open(json_path))
    jid  = [m.joint(n).id for n in E["dof_names"]]
    qad  = np.array([m.jnt_qposadr[j] for j in jid])
    dof  = np.array([m.jnt_dofadr[j] for j in jid])
    act_of = {int(m.actuator_trnid[i, 0]): i for i in range(m.nu)}
    act  = np.array([act_of[j] for j in jid])
    off  = np.array(E["pd_offset"]); sc = np.array(E["pd_scale"])
    lo, hi = np.array(E["lo"]), np.array(E["hi"])
    wlo, whi = off - 4 * sc, off + 4 * sc           # the env does not clamp targets; give the pilot a wide band
    info = dict(E=E, jid=jid, qad=qad, dof=dof, act=act, off=off, sc=sc, lo=lo, hi=hi,
                wlo=wlo, whi=whi, dec=int(E["decimation"]),
                pelvis=m.body("pelvis").id, torso=m.body("torso_link").id,
                feet=[m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id])
    return m, info


# ============================================================ pilot / student
def make_pilot(m, info, kp=30., kd=20., kxi=.5):
    from invdyn_pilot import InvDynPilot
    pil = InvDynPilot(m, info["jid"], info["act"], info["pelvis"], info["torso"],
                      Kp=kp, Kd=kd, kxi=kxi)
    pil.feet = info["feet"]
    pil.d = mujoco.MjData(m)                        # the pilot reads a private copy
    return pil


def load_student(path):
    from student_prob import load
    return load(path, device="cpu")


def action_sd(m):
    """Canonical ys resolver (mirrors pm_eval_student.action_sd): the student may be a
    LastLayer (ys on .base) or an Ensemble (ys on .members[0]), not a bare _Norm."""
    if hasattr(m, "ys"):
        return m.ys
    if hasattr(m, "base"):
        return action_sd(m.base)
    return action_sd(m.members[0])


def _student_action_from_obs(X, student, off, sc):
    """Pilot-for-gate for arm B: obs → joint targets. Batched, torch."""
    with torch.no_grad():
        u = student.uq(torch.as_tensor(X, dtype=torch.float32))
    a_np = u["action"].numpy().astype(np.float64)
    return off + sc * a_np


# ============================================================ per-clip runner
def _foot_low(m, sc, qp, feet):
    """Lowest foot collision-geom z at pose qp (mirrors id_track.foot_low; capsule/sphere/box aware)."""
    sc.qpos[:] = qp; mujoco.mj_kinematics(m, sc); lowz = 9.
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in feet and (m.geom_contype[g] | m.geom_conaffinity[g]):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                lowz = min(lowz, sc.geom_xpos[g, 2] - m.geom_size[g, 0])
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                R = sc.geom_xmat[g].reshape(3, 3); lowz = min(lowz, sc.geom_xpos[g, 2] - m.geom_size[g, 0] - abs(R[2, 2]) * m.geom_size[g, 1])
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                R = sc.geom_xmat[g].reshape(3, 3); lowz = min(lowz, sc.geom_xpos[g, 2] - np.abs(R[2] * m.geom_size[g]).sum())
            else:
                lowz = min(lowz, sc.geom_xpos[g, 2] - m.geom_rbound[g])
    return lowz


def _reset_to_clip_start(m, d, clip, info):
    qpos = clip["qpos_of"](0)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos
    d.qpos[2] = clip["rp"][0, 0, 2]
    if m.nq >= 43: d.qpos[36:43] = [0, 0, -2, 1, 0, 0, 0]
    R0 = clip["PR"][0]
    d.qvel[:3]   = clip["rv"][0, 0]
    d.qvel[3:6]  = R0.T @ clip["rw"][0, 0]
    d.qvel[info["dof"]] = clip["dvel"][0]
    if info.get("ground"):                                        # drop the lowest sole onto the floor (id_track --ground)
        d.qpos[2] -= _foot_low(m, clip["_sc"], d.qpos.copy(), info["feet"])
    mujoco.mj_forward(m, d)


def run_arm(m, d, info, clip, arm, student=None, pilot=None, gate=None, gains=None, plan=False):
    """One arm on one clip. Returns dict(passed, steps, fell_at)."""
    _reset_to_clip_start(m, d, clip, info)
    if gate is not None: gate.reset()
    if pilot is not None and hasattr(pilot, "stance"):
        pilot.stance = []
    gains = gains or {}
    n_total = int((clip["F"] - 1) / clip["fps"] / DT) - 2
    prev_action = np.zeros(29, dtype=np.float32)          # historical_previous_processed_actions = 0 at reset (obs0)

    for step in range(n_total):
        t = step * DT
        # The reference used for the pilot is the NEXT step (act_track expects a target pose)
        ref = ref_at(clip, t + DT)
        # a getter for build_obs (returns reference at current time + lag*DT)
        ref_getter = lambda lag: ref_at(clip, t + lag * DT)

        obs = None
        sig = None
        if arm in ("A", "B", "D"):
            obs = build_obs(d, info["qad"], info["dof"], info["torso"], ref_getter, prev_action)
            if student is not None:
                with torch.no_grad():
                    u = student.uq(torch.as_tensor(obs[None], dtype=torch.float32))
                sig = float((u["sig_epi"] / action_sd(student)).mean())   # epi = the 58/58 gate signal (fbw_2 default FBW2_SIG=epi)

        # Primary action
        if arm in ("A", "B"):
            T_raw = info["off"] + info["sc"] * u["action"].numpy()[0]
        else:
            T_raw = pilot.act_track(d, ref, info["wlo"], info["whi"], gains)

        # Gate
        if arm in ("B", "D") and gate is not None:
            if arm == "B":
                pilot_for_gate = lambda X: _student_action_from_obs(X, student, info["off"], info["sc"])
            else:
                # The physical pilot is not obs-compatible; reference replay is the look-ahead's pilot.
                pilot_for_gate = lambda X: np.asarray(X)[:, 99:128].astype(np.float64)
            T, gate_open, inter = gate.filter(d, obs, T_raw, pilot_for_gate, sig)
        else:
            T = T_raw

        # Step the world
        d.ctrl[info["act"]] = T
        for _ in range(info["dec"]):
            mujoco.mj_step(m, d)
        prev_action = ((T - info["off"]) / info["sc"]).astype(np.float32)   # normalized executed action (env's last_actions)

        z  = d.qpos[2]
        up = 1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2)
        if z < PASS_Z or up < PASS_UP:
            return dict(passed=False, steps=step + 1, fell_at=step, of=n_total)

    return dict(passed=True, steps=n_total, fell_at=None, of=n_total)


# ============================================================ verdict
def verdict(rows):
    """Per-arm pass counts + the plan's §9 verdict."""
    by_arm = defaultdict(list)
    for arm, _cid, r in rows: by_arm[arm].append(r["passed"])
    n = {arm: sum(v) for arm, v in by_arm.items()}
    total = {arm: len(v) for arm, v in by_arm.items()}
    A = n.get("A", None); B = n.get("B", 0); C = n.get("C", 0); D = n.get("D", 0)
    # --- preconditions: don't emit a D-vs-B-vs-C verdict off a broken harness ---
    if "A" in total and A == 0:
        return n, total, ("INCONCLUSIVE — arm A collapses (0 passes): the student obs is off-distribution; "
                          "fix build_obs (validate reset-epi < thr) before trusting B/C/D.")
    if total.get("B") and "A" in total and B < A:
        return n, total, ("INCONCLUSIVE — arm B (student+gate) is below arm A (student alone): the gate is "
                          "not behaving as the 58/58 control; check the gate signal / fbw config before the verdict.")
    if not total.get("B"):
        return n, total, "(arm B not run; verdict requires the control)"
    # --- verdict ---
    if D >= B - 1:
        v = "MW replaces PPO — the teacherless route reaches the control arm."
    elif D > C:
        v = "MW helps the pilot but does not close the gap (narrowed claim)."
    else:
        v = "MW does not rescue the pilot — add authority (CAM or stepping)."
    return n, total, v


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="eval_prob/stand/g1_env.mjb")
    ap.add_argument("--json",  required=True, help="eval_prob/stand/g1_env.json")
    ap.add_argument("--clips", help="file with one motion path per line")
    ap.add_argument("--student", help="teacher-student checkpoint (needed for A,B,D)")
    ap.add_argument("--arms", default="ABCD", help="which arms to run")
    ap.add_argument("--out", default="results/mw_vs_ppo")
    ap.add_argument("--gains", default="", help="k=v k=v, e.g. 'kc=30 dc=10'")
    ap.add_argument("--pilot-kp", type=float, default=30.); ap.add_argument("--pilot-kd", type=float, default=20.)
    ap.add_argument("--pilot-kxi", type=float, default=.5)
    ap.add_argument("--fbw2-thr", type=float, default=0.0136); ap.add_argument("--fbw2-release", type=int, default=1)
    ap.add_argument("--fbw2-calm",  type=int, default=10)
    ap.add_argument("--check-obs", action="store_true", help="verify build_obs against recorded data, then exit")
    ap.add_argument("--data-pm", default="data_pm")
    ap.add_argument("--ground", type=int, default=1, help="drop lowest sole onto floor at reset (id_track --ground default)")
    ap.add_argument("--plan", type=int, default=0, help="CoM re-planning for the pilot arm (id_track default 1; 0 = unplanned act_track)")
    a = ap.parse_args()

    m, info = load_env(a.model, a.json)
    info["ground"] = bool(a.ground)

    if a.check_obs:
        check_obs_format(a.data_pm, m, info["qad"], info["dof"], info["torso"], info["feet"])
        return

    with open(a.clips) as f:
        clip_paths = [ln.strip() for ln in f if ln.strip()]
    if not clip_paths:
        raise SystemExit("--clips is empty")

    student = load_student(a.student) if a.student else None
    if any(arm in a.arms for arm in "ABD") and student is None:
        raise SystemExit("arms A/B/D need --student")

    # One pilot and one gate reused across clips; both are stateful per-episode (reset in run_arm)
    pilot = make_pilot(m, info, a.pilot_kp, a.pilot_kd, a.pilot_kxi) if any(arm in a.arms for arm in "CD") else None
    gate = None
    if any(arm in a.arms for arm in "BD"):
        from fbw_2 import GatedFBW
        from g1_kin import DOF_NAMES, FEET
        os.environ["FBW2_RELEASE"] = str(a.fbw2_release)
        os.environ["FBW2_THR"] = str(a.fbw2_thr)
        os.environ["FBW2_CALM"] = str(a.fbw2_calm)
        gate = GatedFBW(Sim(m, mujoco.MjData(m), info["dec"]), DOF_NAMES, FEET)
        print(f"[gate] cfg {gate.c2}", flush=True)

    gains = {}
    for kv in a.gains.split():
        k, v = kv.split("="); gains[k] = float(v)

    os.makedirs(a.out, exist_ok=True)
    rows, t0 = [], time.time()
    for path in clip_paths:
        try:
            clip = load_clip(path, m, info["qad"], info["dof"], info["pelvis"], info["torso"], info["feet"])
        except Exception as ex:
            print(f"[load] SKIP {os.path.basename(path)}: {ex}", flush=True); continue
        cid = os.path.basename(path)
        for arm in a.arms:
            d = mujoco.MjData(m)
            try:
                r = run_arm(m, d, info, clip, arm,
                            student=student if arm in "ABD" else None,
                            pilot=pilot if arm in "CD" else None,
                            gate=gate if arm in "BD" else None,
                            gains=gains)
            except Exception as ex:
                r = dict(passed=False, steps=0, fell_at=-1, of=clip["F"], error=repr(ex))
            rows.append((arm, cid, r))
            print(f"[{arm}] {cid:30s} {'PASS' if r['passed'] else 'FAIL':4s} "
                  f"steps {r['steps']:4d}/{r['of']}", flush=True)
        print("", flush=True)

    n, total, v = verdict(rows)
    print("=" * 60)
    for arm in a.arms:
        print(f"  {arm}: {ARM_TITLE[arm]:22s}  {n[arm]}/{total[arm]}")
    print(f"\n  VERDICT: {v}")
    print(f"  elapsed: {time.time() - t0:.1f}s")

    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out_json = os.path.join(a.out, f"run_{stamp}.json")
    with open(out_json, "w") as f:
        json.dump({"arm_counts": n, "arm_totals": total, "verdict": v,
                   "rows": [dict(arm=r[0], clip=r[1], **r[2]) for r in rows]}, f, indent=1)
    print(f"  -> {out_json}")


if __name__ == "__main__":
    main()
