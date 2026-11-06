"""pm_collect.py - record (deployable observation, teacher action) pairs from the ProtoMotions G1 tracker running in MuJoCo (CPU).

Teacher input recorded per step (349 numbers - the tracker's deployable observation, no privileged max-coords):
    noisy_reduced_coords_obs (64) | noisy_mimic_reduced_coords_target_poses (256) | historical_previous_processed_actions (29)
Label: the teacher's clean 29-dim action. The EXECUTED action gets extra Gaussian noise (sigma cycles per episode over PM_NOISES) so the
data covers states the student will drift into (DART-style); the label stays the teacher's action for the state actually reached.

Env vars: PM_OUT (npz path), PM_STEPS, PM_NOISES (comma list), PM_SEED.  Run via the patched launcher (see run_collect.sh).
"""
import os, sys, time, numpy as np, torch
import pm_launch                                              # single-device Fabric patch
from protomotions.agents.evaluators.base_evaluator import BaseEvaluator

KEYS = ["noisy_reduced_coords_obs", "noisy_mimic_reduced_coords_target_poses", "historical_previous_processed_actions"]


def student_obs(obs_td):
    return torch.cat([obs_td[k].reshape(obs_td[k].shape[0], -1) for k in KEYS], -1)


@torch.no_grad()
def collect(self):
    out, steps = os.environ["PM_OUT"], int(os.environ.get("PM_STEPS", "20000"))
    noises = [float(x) for x in os.environ.get("PM_NOISES", "0,0.05,0.1,0.2").split(",")]
    torch.manual_seed(int(os.environ.get("PM_SEED", "0")))
    self.agent.eval()
    X, A, EP, NZ, DONE = [], [], [], [], []
    MID, RZ, UP = [], [], []                                  # motion id, pelvis height (m), pelvis uprightness (cos tilt)
    mm = self.env.reference_motion.motion_manager
    sim = self.env.simulator
    done_idx, ep, t0 = None, 0, time.time()
    for step in range(steps):
        obs, _ = self.env.reset(done_idx)
        self.agent.pre_collect_step(step)
        obs = self.agent.add_agent_info_to_obs(obs)
        obs_td = self.agent.obs_dict_to_tensordict(obs)
        a = self._policy_action(obs_td).detach()
        sigma = noises[ep % len(noises)]
        X.append(student_obs(obs_td)[0].cpu().numpy()); A.append(a[0].cpu().numpy()); EP.append(ep); NZ.append(sigma)
        MID.append(int(mm.motion_ids[0])) if mm is not None else MID.append(-1)
        q = sim.data.qpos
        RZ.append(float(q[2])); UP.append(float(1.0 - 2.0 * (q[4] ** 2 + q[5] ** 2)))
        _, _, dones, _, _ = self.env.step(a + sigma * torch.randn_like(a))
        done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
        DONE.append(bool(len(done_idx)))
        if len(done_idx):
            ep += 1
        if (step + 1) % 2000 == 0:
            print(f"[collect] {step + 1}/{steps} steps, {ep} episodes, {(step + 1) / (time.time() - t0):.0f} steps/s", flush=True)
    np.savez_compressed(out, X=np.array(X, np.float32), A=np.array(A, np.float32), ep=np.array(EP), sigma=np.array(NZ, np.float32),
                        done=np.array(DONE), motion=os.environ.get("PM_MOTION_NAME", ""),
                        mid=np.array(MID), root_z=np.array(RZ, np.float32), upright=np.array(UP, np.float32),
                        motion_files=np.array([str(f) for f in getattr(mm.motion_lib, "motion_files", ())]) if mm is not None else np.array([]))
    print(f"[collect] saved {len(X)} pairs ({ep} episodes) -> {out}", flush=True)


BaseEvaluator.simple_test_policy = collect
if __name__ == "__main__":
    from protomotions.inference_agent import main
    main()
