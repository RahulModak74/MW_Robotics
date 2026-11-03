# MW Robotics 3D — a policy manifold instead of six stitched models

`app_3d` is the 3D counterpart of `app.py`. The 2D demo plays back six small, separate manifolds on a side-view
drawing. Here a **single trained MW object is the controller** of a real MuJoCo 3D body (Humanoid-v5, 17 actuators),
and the browser (Three.js) draws MuJoCo's world pose of every geom, frame by frame. Nothing is hand-animated.

Context: `../NEW_POLICY_CPU_GPU.md` (the framing — GPU may be used to *train*, the trained object deploys on CPU) and
`../domain/OUTPUT_EXP_03.md` / `../domain/29DoF/PROGRESS_LOG.md` (VAE-as-policy results; the 3D-balance wall).

## The pipeline

```
 teacher (s,a)  ──►  PolicyManifold3DVAE  (mw_framework MWVAE, Pyro SVI via MWDetector, physics priors)  ──►  app_3d
 CEM or PPO         x = (s, a)                                                                 act(s) = latent query
```

| Piece | File | What it is |
|---|---|---|
| Teacher A (CPU, baseline) | `train_policy_3d.py` | multiprocess CEM over a linear policy, plus a **plain-PyTorch CVAE** baseline (no physics, no Pyro) |
| Teacher B (stronger) | `train_ppo_teacher_3d.py` | stable-baselines3 PPO, 11 parallel envs |
| **MW policy manifold** | `mw_policy_3d.py`, `train_mw_3d.py` | `PolicyManifold3DVAE(MWVAE)` trained by Pyro SVI through `MWDetector` |
| Server + viewer | `app_3d.py`, `static_3d/` | Flask API + Three.js; modes: MW manifold / CEM teacher / plain CVAE |
| Shared | `policy_3d.py` | features, linear policy, plain CVAE, env factory |

**Physics priors in the MW manifold** (declared with `mw_framework.core.bound/constraint`, limits read from the MuJoCo model, not typed in):
P1 torso height inside the healthy envelope · P2 unit-norm torso quaternion · P3 joint angles inside `jnt_range` ·
P4 torques inside actuator limits.

**The policy is a query on the manifold**, not a separate network: `act(s)` = action block of the manifold point whose
state block best matches `s` (`argmin_z ‖decode(z)_s − s‖² + kl·‖z‖²`, warm-started from the previous step).
The same object also gives the **MW distance** of any (s, a) — how far a state/torque pair is from the learned manifold.

## Run it

```bash
cd full_flow_animation
pip install -r requirements.txt            # flask (+ torch, pyro-ppl, mujoco, gymnasium, stable-baselines3 from the parent repo)

python train_policy_3d.py --env Humanoid-v5 --pop 240 --gens 60      # CEM teacher + plain CVAE baseline  (~10 min CPU)
python train_ppo_teacher_3d.py --env Humanoid-v5 --timesteps 10000000 --envs 11   # PPO teacher (see timing below)
python train_mw_3d.py --env Humanoid-v5 --teacher ppo                # MW manifold on PPO data   -> humanoid_ppo_mw.pth
python train_mw_3d.py --env Humanoid-v5                              # MW manifold on CEM data   -> humanoid_mw.pth
python app_3d.py                                                     # http://127.0.0.1:5051
```

`train_mw_3d.py` prints `MW VAE (Pyro SVI) training on: cuda|cpu`. Device selection is the framework's own
(`mw_framework.core.get_device`); force it with `MW_DEVICE=cpu`. MuJoCo rollouts always run on CPU worker processes.

## Running on a GPU machine (e.g. the 16 GB Windows box)

1. Install a CUDA build of PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cu124` or newer), then check
   `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
2. `pip install mujoco gymnasium[mujoco] pyro-ppl stable-baselines3 flask pandas scikit-learn`.
3. Run the same commands. On Windows keep the `if __name__ == "__main__":` guards (already present) because the
   scripts use `multiprocessing`; lower `--envs/--workers` to your physical core count.
4. What the GPU speeds up: the PPO network updates and the Pyro SVI step. What it does **not**: MuJoCo simulation
   (CPU), which is the bulk of PPO wall-clock for Humanoid. Expect a modest PPO speed-up, a larger one for large-batch SVI.
5. Copy `checkpoints_3d/*_mw.pth` back; `app_3d.py` runs inference on plain CPU.

### This machine (RTX 3050 6 GB Laptop, not 16 GB)

`torch.cuda.is_available()` is `False` even though `nvidia-smi` works. Diagnosed: the installed torch is already a
CUDA 13.0 build (`2.13.0+cu130`) and the driver is 595.99 (CUDA 13.2). Calling the driver directly
(`cuInit(0)`) returns **error 999 (CUDA_ERROR_UNKNOWN)**, i.e. it fails *below* PyTorch. **Reinstalling PyTorch will not
fix it.** Likely a stale `nvidia_uvm` kernel module; try
`sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`, otherwise reboot, then re-run the `cuInit` check. (Needs `sudo`, which the
agent session does not have.) Until then everything runs on CPU.

## Results (held-out seeds, cap 500 steps; Humanoid-v5)

| Controller | Steps | Forward | Notes |
|---|---|---|---|
| CEM teacher (linear, 770 params) | 89.4 | 0.69 m | measured; matches the documented 3D wall (58–113 steps) |
| Plain-PyTorch CVAE on CEM data | 91.7 | 0.70 m | measured; faithfully copies a weak teacher |
| PPO teacher | _pending_ | | training in progress |
| MW manifold on PPO data | _pending_ | | |

The pitch-table figures in `NEW_POLICY_CPU_GPU.md` (850 / 750 / 870 steps, "beats teacher after DAgger") are
**projections, not measurements** — nothing here supports them yet. The PPO claim of "800+ steps in about an hour" is
likewise unverified: the CEM numbers above are the only measured Humanoid-v5 results so far.

## Honest scope

- A distilled policy can only be as good as its teacher; a manifold trained on 90-step data balances for ~90 steps.
- MW priors constrain *what states/torques are physically admissible*, which is not the same as *balancing*. Whether they
  let the student exceed the teacher (the DAgger idea in `NEW_POLICY_CPU_GPU.md`) is untested.
- Not a safety system; do not use to control a real robot.
