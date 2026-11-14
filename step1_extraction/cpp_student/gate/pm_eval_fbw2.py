"""pm_eval_fbw2.py - copy of pm_eval_student.py with --fbw2 (fbw_2.GatedFBW: MW-warning-gated flight computer). Score a student (or the teacher) with ProtoMotions' OWN evaluator, on one motion, MuJoCo CPU.

    python pm_eval_student.py --student students/x.pt [--act-noise 0.1] [--seed 0] [--uq-out run.npz] [--gate epi:0.05] \
        --simulator mujoco --headless --num-envs 1 --full-eval --checkpoint <teacher ckpt> --motion-file <.motion>
Without --student the teacher runs unchanged (the reference row). Their evaluate() reports tracking errors, success rate, etc.

Students: student_net.py checkpoints (mlp, bundle) or student_prob.py checkpoints (gauss, ensemble, mwv2, lastlayer).
--uq-out   per step: the student's uncertainty signals, its action, the teacher's action at the same observation, whether the step was
           handed to the teacher, and the step at which ProtoMotions' evaluator latched a failure (fail_step, -1 = none).
--gate S:T hand the step to the teacher when signal S > T. S in {epi, mw, ale}: epi/ale = mean over joints of sigma / action sd;
           mw = MW distance. Set T on training motions (Retry_MW_distillation.md section 6).
"""
import os, sys, json
import numpy as np
import pm_launch
import torch
from protomotions.agents.evaluators.base_evaluator import BaseEvaluator

KEYS = ["noisy_reduced_coords_obs", "noisy_mimic_reduced_coords_target_poses", "historical_previous_processed_actions"]
argv = sys.argv[1:]


def pop(flag, cast=str, default=None):
    if flag in argv:
        i = argv.index(flag); v = cast(argv[i + 1]); del argv[i:i + 2]; return v
    return default


student_path = pop("--student")
noise = pop("--act-noise", float, 0.0)
seed = pop("--seed", int, 0)
uq_out = pop("--uq-out")
qpos_out = pop("--qpos-out")                                  # per policy step: MuJoCo qpos (for rendering videos)
QPOS, QNAMES = [], []
gate = pop("--gate")
use_fbw = "--fbw2" in argv
if use_fbw:
    argv.remove("--fbw2")
fbw2_out = pop("--fbw2-out")
import time
TIM = {"t_pilot": [], "t_fbw": [], "ep": []}
PILOT_REF = os.environ.get("PILOT_REF") == "1"                # test A: the pilot is plain reference replay (+1 joint angles); no teacher, no student
FBW = {"obj": None, "pred": None, "err": [], "inter": [], "viol": []}
torch.manual_seed(seed); np.random.seed(seed)

student, prob = None, False
if student_path:
    ck = torch.load(student_path, map_location="cpu", weights_only=False)
    if ck["kind"] in ("mlp", "bundle"):
        from student_net import load
    else:
        from student_prob import load
        prob = True
    student = load(student_path, "cpu")


def action_sd(m):
    if hasattr(m, "ys"):
        return m.ys
    if hasattr(m, "base"):
        return action_sd(m.base)
    return action_sd(m.members[0])


gate_sig, gate_thr = (gate.split(":")[0], float(gate.split(":")[1])) if gate else (None, None)
if gate and not prob:
    raise SystemExit("--gate needs a probabilistic student (student_prob.py checkpoint)")
LOG = {k: [] for k in ("epi", "ale", "mw", "law_range", "law_rate", "a_student", "a_teacher", "handed", "obs", "sig_epi_j", "sig_ale_j")}
SIG_NOW = {"v": None}
STATE = {"step": 0, "fail_step": -1, "episode": -1}

# --- async-latency supervisor (FBW_ASYNC=1) ----------------------------------
# Sim-step latency
# model into the REAL exam. The fbw correction is applied L = ceil(solve_ms/20)
# steps LATE (one search in flight), blended onto the then-current student torque
# and decayed by age: T = T_student + w(age)*(T_fbw - T_student), w = 1-age/max_age.
# Default exam is unaffected (flag off). solve_ms is the live-measured python fbw
# solve / speedup (speedup=1 => python latency; 4.5 => measured C++ kernel).
import math as _math
# --- proportional (PID-style P term) authority ramp on the MW signal --------
# Instead of binary gate (full correction iff sig>thr), ramp the applied correction
# authority w = clip((sig - lo)/(full - lo), 0, 1): control eases in as MW rises and
# reaches FULL force at sig=full, pre-compensating the few steps before the fall commits.
# Fire the search from `lo` (set FBW2_THR=lo); scale output by w here. T_app = T + w*(T_fbw - T).
PID_ON   = os.environ.get("FBW_PID") == "1"
PID_LO   = float(os.environ.get("FBW_PID_LO", "0.0136"))
PID_FULL = float(os.environ.get("FBW_PID_FULL", "0.025"))
PID_GAMMA = float(os.environ.get("FBW_PID_GAMMA", "1.0"))   # >1 = concave (slow start), <1 = convex (fast start)
PID_GAIN = float(os.environ.get("FBW_PID_GAIN", "1.0"))     # magnitude multiplier; >1 = overdrive (harder than fbw's own correction)
# ENVELOPE mode (FBW_PID_SOFT>0): a force envelope that engages the instant MW rises.
#   sig <  PID_LO            -> w = 0        (normal operation, no force)
#   PID_LO <= sig < PID_FULL -> w = PID_SOFT (soft envelope, e.g. 0.6*hard: reasonably hard, in play early)
#   sig >= PID_FULL          -> w = 1.0      (hard envelope, threshold crossed)
# The early soft force keeps good motions from escalating, so MW rarely climbs to PID_FULL.
PID_SOFT = float(os.environ.get("FBW_PID_SOFT", "0"))       # 0 = disabled (linear ramp); >0 = two-level envelope soft strength
# TEACHERLESS replay base (FBW_REPLAY=1): base pilot = the reference joint angles at +1 step
# (obs[99:128] = priors.REF_Q1), NO student / NO NVIDIA / NO MW. fbw's own physics gate decides
# when to correct. Measures the fully-teacherless expert: reference + physics search.
REPLAY_ON = os.environ.get("FBW_REPLAY") == "1"
# DAgger (FBW_DAGGER=1): STUDENT drives the body (visits its own mistake states); the teacherless fbw
# oracle (reference base + physics search) runs as an OBSERVER and labels each student-visited state.
# Used with FBW_COLLECT to grow the dataset with on-policy states the one-shot BC student actually reaches.
DAGGER_ON = os.environ.get("FBW_DAGGER") == "1"
REF_Q1 = slice(99, 128)
# AMORTIZATION data collection (FBW_COLLECT=<path.npz>): save (obs349, fbw_action29) pairs for distillation.
COLLECT = os.environ.get("FBW_COLLECT")
COLLECT_BUF = {"x": [], "a": []}
if COLLECT:
    import atexit
    def _dump_collect():
        if COLLECT_BUF["x"]:
            np.savez_compressed(COLLECT, x=np.asarray(COLLECT_BUF["x"], np.float32), a=np.asarray(COLLECT_BUF["a"], np.float32))
            print(f"[collect] saved {len(COLLECT_BUF['x'])} (obs,action) pairs -> {COLLECT}", flush=True)
    atexit.register(_dump_collect)
ASYNC_ON = os.environ.get("FBW_ASYNC") == "1"
# mode: "delta"  = freeze knot0 and decay by age (the point-in-time delta; fails, 40/58)
#       "spline" = fly the 2-knot plan FORWARD: apply corr(phase=steps_since_landing) ramping
#                  knot0 -> knot1 over H steps then holding knot1 (the "student rolls it forward")
ASYNC = {"k_refresh": int(os.environ.get("FBW_ASYNC_KREFRESH", "7")),
         "max_age":   int(os.environ.get("FBW_ASYNC_MAXAGE", "20")),
         "speedup":   float(os.environ.get("FBW_ASYNC_SPEEDUP", "4.5")),
         "zero_lat":  os.environ.get("FBW_ASYNC_LAT", "sim") == "zero",
         "mode":      os.environ.get("FBW_ASYNC_MODE", "delta"),
         "H":         int(os.environ.get("FBW_H", "20")),
         "pending": None, "current": None, "last_fire": -(10**9), "step": 0}


def async_apply(T_student, T_fbw, gate_open, solve_s, knot1=None):
    """Return the torque actually applied this control step under the latency model.
    knot0 = T_fbw - T_student (the correction at phase 0); knot1 = the plan's end knot (fbw.prev[1])."""
    A = ASYNC; k = A["step"]; A["step"] += 1; H = A["H"]
    # land an in-flight solve
    if A["pending"] is not None and k >= A["pending"][0]:
        A["current"] = (*A["pending"][1], A["pending"][0]); A["pending"] = None   # (knot0, knot1, land_step)
    applied = T_student
    if A["current"] is not None:
        knot0, kn1, land = A["current"]; age = k - land
        if age < A["max_age"]:
            if A["mode"] == "spline" and kn1 is not None:
                ph = min(age, H - 1) / max(H - 1, 1)                               # ramp knot0->knot1 over H, then hold knot1
                applied = T_student + (1.0 - ph) * knot0 + ph * kn1
            else:                                                                  # delta: freeze knot0, linear age-decay
                w = 1.0 - age / A["max_age"]; applied = T_student + w * knot0
        else:
            A["current"] = None
    # fire a supervisor solve (worker idle + refresh elapsed); cost = latency, not in-loop
    if gate_open and A["pending"] is None and (k - A["last_fire"]) >= A["k_refresh"]:
        sms = (solve_s * 1e3) / A["speedup"]
        L = 0 if A["zero_lat"] else max(1, _math.ceil(sms / 20.0))
        kn1 = knot1.copy() if knot1 is not None else None
        A["pending"] = (k + L, ((T_fbw - T_student).copy(), kn1)); A["last_fire"] = k
        if os.environ.get("FBW_ASYNC_DIAG") == "1" and kn1 is not None:
            k0 = T_fbw - T_student
            A.setdefault("_diag", []).append((float(np.abs(k0).mean()), float(np.abs(kn1).mean()), float(np.abs(kn1 - k0).mean())))
            if len(A["_diag"]) <= 6:
                print(f"[async-diag] fire: |knot0|={A['_diag'][-1][0]:.4f} |knot1|={A['_diag'][-1][1]:.4f} |knot1-knot0|={A['_diag'][-1][2]:.4f}", flush=True)
    return applied
# -----------------------------------------------------------------------------
EP_AT_STEP, EP_FAIL = [], {}                    # per policy step: evaluation episode index; per episode: step of its latched failure
orig = BaseEvaluator._policy_action
orig_check = BaseEvaluator._check_evaluation_failures


@torch.no_grad()
def policy_action(self, obs_td):
    if qpos_out:
        sim = self.env.simulator
        QPOS.append(np.array(sim.data.qpos, dtype=np.float32))
        if not QNAMES:
            import mujoco
            QNAMES.extend(mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in range(sim.model.njnt))
    if PILOT_REF:
        from priors import pd_map
        off_, sc_ = pd_map()
        a = (torch.cat([obs_td[k].reshape(obs_td[k].shape[0], -1) for k in KEYS], -1).cpu()[:, 99:128].float() - off_.float()) / sc_.float()
        a = a.to(obs_td[KEYS[0]].device)
    elif student is None:
        a = orig(self, obs_td)
    else:
        x = torch.cat([obs_td[k].reshape(obs_td[k].shape[0], -1) for k in KEYS], -1).cpu()
        if prob:
            _t0 = time.perf_counter(); u = student.uq(x); TIM["t_pilot"].append(time.perf_counter() - _t0)
            a, handed = u["action"], False
            ys = action_sd(student)
            sig = {"epi": float((u["sig_epi"] / ys).mean()), "ale": float((u["sig_ale"] / ys).mean()), "mw": float(u["mw_dist"][0])}
            SIG_NOW["v"] = sig
            a_t = orig(self, obs_td).cpu() if (uq_out or gate) else None
            if gate and sig[gate_sig] > gate_thr:
                a, handed = a_t, True
            if uq_out:
                for k in ("epi", "ale", "mw"):
                    LOG[k].append(sig[k])
                LOG["law_range"].append(float(u["law_range"][0])); LOG["law_rate"].append(float(u["law_rate"][0]))
                LOG["a_student"].append(u["action"][0].numpy()); LOG["a_teacher"].append(a_t[0].numpy())
                LOG["handed"].append(handed); LOG["obs"].append(x[0].numpy())
                LOG["sig_epi_j"].append(u["sig_epi"][0].numpy()); LOG["sig_ale_j"].append(u["sig_ale"][0].numpy())
        else:
            a = student(x)
        a = a.to(obs_td[KEYS[0]].device)
    if use_fbw:                                                # MW-gated fly-by-wire (fbw_2.py)
        from fbw_2 import GatedFBW
        from priors import pd_map
        from g1_kin import DOF_NAMES, FEET
        sim = self.env.simulator
        if FBW["obj"] is None:
            FBW["obj"] = GatedFBW(sim, DOF_NAMES, FEET, student=student); FBW["off"], FBW["sc"] = (t.numpy().astype(np.float64) for t in pd_map())
            print("[fbw2] on:", FBW["obj"].c, flush=True)
            if REPLAY_ON or DAGGER_ON:                         # teacherless: pilot = reference joint targets (obs[99:128])
                def pilot(X):
                    return np.asarray(X, dtype=np.float64)[:, REF_Q1].copy()
            else:
                def pilot(X):
                    Xt = torch.as_tensor(X, dtype=torch.float32)
                    A = student.uq(Xt)["action"].numpy().astype(np.float64)
                    return FBW["off"] + FBW["sc"] * A
            FBW["pilot"] = pilot
        xo = torch.cat([obs_td[k].reshape(obs_td[k].shape[0], -1) for k in KEYS], -1)[0].cpu().numpy().astype(np.float64)
        a_student = a.clone()                                  # preserve the student action (DAgger applies this, not the oracle's)
        if REPLAY_ON or DAGGER_ON:                             # teacherless base = reference targets; always let fbw's physics gate decide
            T = xo[REF_Q1].copy(); sig_now = 1e9
        else:
            T = FBW["off"] + FBW["sc"] * a[0].cpu().numpy().astype(np.float64)
            sig_now = float(SIG_NOW["v"][FBW["obj"].c2["sig"]])
        T2, gate_open, inter = FBW["obj"].filter(sim.data, xo, T, FBW["pilot"], sig_now)
        if PID_ON:                                             # proportional / envelope authority on MW
            if PID_SOFT > 0:                                    # two-level force envelope: soft early, hard past threshold
                w = 1.0 if sig_now >= PID_FULL else (PID_SOFT if sig_now >= PID_LO else 0.0)
            else:                                              # linear ramp 0->1 over [PID_LO, PID_FULL], full force at PID_FULL
                w = (sig_now - PID_LO) / max(PID_FULL - PID_LO, 1e-9)
                w = min(max(w, 0.0), 1.0) ** PID_GAMMA
            T2 = T + PID_GAIN * w * (T2 - T)                    # PID_GAIN>1 = overdrive
        if ASYNC_ON:                                           # apply the correction L steps late (delta) or rolled fwd (spline)
            kn1 = None
            if gate_open and getattr(FBW["obj"], "fbw", None) is not None:
                kn1 = FBW["obj"].fbw.prev[1]                    # the plan's end knot (blend of K[:,1]); = knot1 in torque-delta space
            T2 = async_apply(T, T2, gate_open, FBW["obj"].log["t_fbw"][-1], kn1)
        TIM["ep"].append(STATE["episode"])
        if COLLECT and (REPLAY_ON or DAGGER_ON):              # expert label = the fbw-corrected action over the reference base
            COLLECT_BUF["x"].append(xo.astype(np.float32).copy()); COLLECT_BUF["a"].append((((T2 - FBW["off"]) / FBW["sc"]).astype(np.float32)).copy())
        if DAGGER_ON:                                          # body follows the STUDENT (on-policy rollout); oracle only labeled above
            a = a_student
        else:
            a = torch.as_tensor(((T2 - FBW["off"]) / FBW["sc"])[None], dtype=a.dtype, device=a.device)
    STATE["step"] += 1
    out = a + noise * torch.randn_like(a) if noise else a
    return out


def check(self, active_envs, trial_ids):
    out = orig_check(self, active_envs, trial_ids)
    EP_AT_STEP.append(STATE["episode"])
    if len(trial_ids) and bool(self._evaluation.failed[trial_ids].any()):
        if STATE["fail_step"] < 0:
            STATE["fail_step"] = STATE["step"] - 1             # index of the action whose step latched the (first) failure
        EP_FAIL.setdefault(STATE["episode"], STATE["step"] - 1)
    return out


def ep_report(e):
    try:
        _ep_report(e)
    except Exception as ex:                                    # progress printing must never break an exam run
        print(f"[fbw2-ep] report error: {ex!r}", flush=True)


def _ep_report(e):
    """one line per finished episode (motion): fell or not, gate share, decision time - for live progress."""
    if e < 0 or not use_fbw or FBW["obj"] is None:
        return
    ep = np.array(TIM["ep"]); sel = ep == e
    if not sel.any():
        return
    g = np.array(FBW["obj"].log["gate"])[sel]; tot = np.array(FBW["obj"].log["t_fbw"])[sel] + np.array(TIM["t_pilot"])[-len(ep):][sel]
    print(f"[fbw2-ep] ep {e} {'FELL' if e in EP_FAIL else 'OK'} steps {int(sel.sum())} gate {g.mean():.3f} "
          f"median {np.median(tot)*1e3:.2f} ms within20ms {(tot <= 0.02).mean():.3f}", flush=True)


def _wrap_start(fn):
    def on_start(self, env_ids):                               # one evaluation episode per motion (num_envs = 1)
        ep_report(STATE["episode"])
        STATE["episode"] += 1
        if use_fbw:
            if FBW["obj"] is not None:
                FBW["obj"].reset()
            if ASYNC_ON:                                       # per-episode reset of the latency supervisor
                ASYNC.update(pending=None, current=None, last_fire=-(10**9), step=0)
        return fn(self, env_ids)
    return on_start


BaseEvaluator._policy_action = policy_action
BaseEvaluator._check_evaluation_failures = check
import importlib
for mod, name in (("protomotions.agents.evaluators.base_evaluator", "BaseEvaluator"),
                  ("protomotions.agents.evaluators.mimic_evaluator", "MimicEvaluator"),
                  ("protomotions.agents.evaluators.masked_mimic_evaluator", "MaskedMimicEvaluator")):
    try:
        cls = getattr(importlib.import_module(mod), name)
    except (ImportError, AttributeError):
        continue
    if "_on_episode_start" in cls.__dict__:                     # subclasses override it; wrap each definition once
        cls._on_episode_start = _wrap_start(cls.__dict__["_on_episode_start"])

if __name__ == "__main__":
    from protomotions.inference_agent import main
    try:
        main(argv)
    finally:
        ep_report(STATE["episode"])
        print(f"[run] steps {STATE['step']} fail_step {STATE['fail_step']}", flush=True)
        if use_fbw and fbw2_out and FBW["obj"] is not None:
            L = FBW["obj"].log; g = np.array(L["gate"]); tf = np.array(L["t_fbw"]); tp = np.array(TIM["t_pilot"])
            print(f"[fbw2] steps {len(g)} gate_open {g.mean():.3f} interventions {int(np.sum(L['inter']))} "
                  f"t_fbw mean {tf.mean()*1e3:.1f} ms p50 {np.median(tf)*1e3:.2f} ms p99 {np.percentile(tf, 99)*1e3:.0f} ms max {tf.max()*1e3:.0f} ms; "
                  f"t_pilot mean {tp.mean()*1e3:.2f} ms", flush=True)
            np.savez_compressed(fbw2_out, **{k: np.array(v) for k, v in L.items()}, t_pilot=tp, episode=np.array(TIM["ep"]),
                                ep_fail=np.array([[k, v] for k, v in sorted(EP_FAIL.items())]).reshape(-1, 2), cfg=json.dumps(FBW["obj"].c, default=str))
        if qpos_out:
            np.savez_compressed(qpos_out, qpos=np.array(QPOS), joint_names=np.array(QNAMES), episode=np.array(EP_AT_STEP), dt=0.02)
            print(f"[qpos] saved {len(QPOS)} frames -> {qpos_out}", flush=True)
        if uq_out:
            np.savez_compressed(uq_out, **{k: np.array(v) for k, v in LOG.items()}, fail_step=STATE["fail_step"],
                                n_steps=STATE["step"], episode=np.array(EP_AT_STEP),
                                ep_fail=np.array([[k, v] for k, v in sorted(EP_FAIL.items())]).reshape(-1, 2), noise=noise, seed=seed, gate=json.dumps(gate), student=str(student_path))
            if False:
                print(f"[fbw] steps {len(FBW['inter'])} interventions {sum(FBW['inter'])} prior one-step max|err| "
                      f"{(max(FBW['err']) if FBW['err'] else float('nan')):.2e} (median {np.median(FBW['err']) if FBW['err'] else float('nan'):.2e})", flush=True)
                np.savez_compressed(uq_out.replace(".npz", "_fbw.npz"), inter=np.array(FBW["inter"]), viol=np.array(FBW["viol"]), err=np.array(FBW["err"]), pfall=np.array(FBW.get("pfall", [])), pfall_best=np.array(FBW["obj"].stats.get("pfall_best", [])), parts=np.array(FBW["obj"].stats.get("parts", [])), cfg=json.dumps(FBW["obj"].c))
            print(f"[uq] saved {STATE['step']} steps, fail_step={STATE['fail_step']} -> {uq_out}", flush=True)
