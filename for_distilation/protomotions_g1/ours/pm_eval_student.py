"""pm_eval_student.py - score a student (or the teacher) with ProtoMotions' OWN evaluator, on one motion, MuJoCo CPU.

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
use_fbw = "--fbw" in argv
if use_fbw:
    argv.remove("--fbw")
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
STATE = {"step": 0, "fail_step": -1, "episode": -1}
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
    if student is None:
        a = orig(self, obs_td)
    else:
        x = torch.cat([obs_td[k].reshape(obs_td[k].shape[0], -1) for k in KEYS], -1).cpu()
        if prob:
            u = student.uq(x)
            a, handed = u["action"], False
            ys = action_sd(student)
            sig = {"epi": float((u["sig_epi"] / ys).mean()), "ale": float((u["sig_ale"] / ys).mean()), "mw": float(u["mw_dist"][0])}
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
    if use_fbw:                                                # MW fly-by-wire between the pilot (this policy) and the robot
        from fbw import FlyByWire
        from priors import pd_map
        from g1_kin import DOF_NAMES, FEET
        sim = self.env.simulator
        if FBW["obj"] is None:
            FBW["obj"] = FlyByWire(sim, DOF_NAMES, FEET); FBW["off"], FBW["sc"] = (t.numpy().astype(np.float64) for t in pd_map())
            print("[fbw] on:", FBW["obj"].c, flush=True)
            def pilot(X):                                      # the same policy, batched, on predicted observations
                Xt = torch.as_tensor(X, dtype=torch.float32)
                A = (student.uq(Xt)["action"] if prob else student(Xt)).numpy().astype(np.float64)
                return FBW["off"] + FBW["sc"] * A
            FBW["pilot"] = pilot
        if FBW["pred"] is not None:                            # was the prior's one-step prediction exact?
            FBW["err"].append(float(np.abs(sim.data.qpos - FBW["pred"]).max()))
        xo = torch.cat([obs_td[k].reshape(obs_td[k].shape[0], -1) for k in KEYS], -1)[0].cpu().numpy().astype(np.float64)
        T = FBW["off"] + FBW["sc"] * a[0].cpu().numpy().astype(np.float64)
        T2, inter = FBW["obj"].filter(sim.data, xo, T, FBW["pilot"])
        FBW["inter"].append(bool(inter)); FBW["viol"].append(FBW["obj"].stats["pred_viol_pilot"][-1]); FBW.setdefault("pfall", []).append(FBW["obj"].stats["pfall"][-1])
        a = torch.as_tensor(((T2 - FBW["off"]) / FBW["sc"])[None], dtype=a.dtype, device=a.device)
    STATE["step"] += 1
    out = a + noise * torch.randn_like(a) if noise else a
    if use_fbw:
        FBW["pred"] = FBW["obj"].check_prior(self.env.simulator.data, FBW["off"] + FBW["sc"] * out[0].cpu().numpy().astype(np.float64))
    return out


def check(self, active_envs, trial_ids):
    out = orig_check(self, active_envs, trial_ids)
    EP_AT_STEP.append(STATE["episode"])
    if len(trial_ids) and bool(self._evaluation.failed[trial_ids].any()):
        if STATE["fail_step"] < 0:
            STATE["fail_step"] = STATE["step"] - 1             # index of the action whose step latched the (first) failure
        EP_FAIL.setdefault(STATE["episode"], STATE["step"] - 1)
    return out


def _wrap_start(fn):
    def on_start(self, env_ids):                               # one evaluation episode per motion (num_envs = 1)
        STATE["episode"] += 1
        if use_fbw:                                            # the reset teleports the robot: no one-step check across it
            FBW["pred"] = None
            if FBW["obj"] is not None:
                FBW["obj"].reset()
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
        if qpos_out:
            np.savez_compressed(qpos_out, qpos=np.array(QPOS), joint_names=np.array(QNAMES), episode=np.array(EP_AT_STEP), dt=0.02)
            print(f"[qpos] saved {len(QPOS)} frames -> {qpos_out}", flush=True)
        if uq_out:
            np.savez_compressed(uq_out, **{k: np.array(v) for k, v in LOG.items()}, fail_step=STATE["fail_step"],
                                n_steps=STATE["step"], episode=np.array(EP_AT_STEP),
                                ep_fail=np.array([[k, v] for k, v in sorted(EP_FAIL.items())]).reshape(-1, 2), noise=noise, seed=seed, gate=json.dumps(gate), student=str(student_path))
            if use_fbw:
                print(f"[fbw] steps {len(FBW['inter'])} interventions {sum(FBW['inter'])} prior one-step max|err| "
                      f"{(max(FBW['err']) if FBW['err'] else float('nan')):.2e} (median {np.median(FBW['err']) if FBW['err'] else float('nan'):.2e})", flush=True)
                np.savez_compressed(uq_out.replace(".npz", "_fbw.npz"), inter=np.array(FBW["inter"]), viol=np.array(FBW["viol"]), err=np.array(FBW["err"]), pfall=np.array(FBW.get("pfall", [])), pfall_best=np.array(FBW["obj"].stats.get("pfall_best", [])), parts=np.array(FBW["obj"].stats.get("parts", [])), cfg=json.dumps(FBW["obj"].c))
            print(f"[uq] saved {STATE['step']} steps, fail_step={STATE['fail_step']} -> {uq_out}", flush=True)
