# Step 1 — GPU teacher, CPU student (`for_distilation/`)

A pragmatic first step: use a consumer GPU where it helps most (training), and keep **deployment on a plain CPU**.

```
 PPO teacher  ──(state, action) data──►  MW policy manifold  ──►  app_3d (CPU)
 (GPU or CPU)        ~250-350k rows      PolicyManifold3DVAE        act(s) = latent query
                                          Pyro SVI, physics priors
```

The MW policy manifold is a Bayesian VAE whose decoder is bound by physics laws read from the MuJoCo model (torso height
envelope, unit quaternion, joint limits, torque limits). The controller **is** a query on that manifold: given the current
state, find the latent point whose decoded state matches best and read off its action block. The same object also reports
how far any `(state, action)` pair is from the learned manifold (the MW distance), which is what makes a classical fallback
auditable.

## What is here

| File | Purpose |
|---|---|
| `train_ppo_teacher_3d.py` | stable-baselines3 PPO teacher on `Humanoid-v5` (parallel CPU environments) |
| `train_policy_3d.py` | CEM teacher over a small linear policy, plus a plain-PyTorch CVAE baseline |
| `train_mw_3d.py` | **MW policy manifold** training (Pyro SVI through `MWDetector`), then held-out evaluation |
| `mw_policy_3d.py` | `PolicyManifold3DVAE` (physics priors) and `ManifoldPolicy` (the latent query) |
| `policy_3d.py` | features, linear policy, plain CVAE, environment factory |
| `mw_bundle_policy.py`, `train_mw_bundle_3d.py` | The MW policy as a **fibre bundle** (`ManifoldAlgebra`): the state is the base manifold and the actions are a fibre over it, so the policy is one forward pass, with physics laws as glue maps and per-piece MW distances |
| `compare_mw_vs_std.py`, `robustness_compare.py` | Side-by-side scoring of MW policies against plain MLP / CVAE policies, clean and under noise and pushes |
| `eval_student.py` | Score a trained student on held-out seeds, with adjustable latent-query settings |
| `app_3d.py`, `static_3d/` | Flask + Three.js viewer; always runs inference on CPU |
| `checkpoints_3d/` | a trained PPO teacher (3M steps) and reports, so you can start from the data step |
| `docs/NEW_APPROACH.md` | notes on taking the teacher further (longer training, entropy bonus, LR schedule, reward shaping) |
| `../STUDENT_COLLAPSE_FIXES.md` | the planned improvements to the student (features, priors, diagnostics) |

## Run it

```bash
cd for_distilation
pip install -r requirements.txt

# 1) Teacher. A 3M-step PPO teacher is already in checkpoints_3d/, so you can skip this and go to step 2.
python train_ppo_teacher_3d.py --env Humanoid-v5 --timesteps 5000000 --envs 10

# 2) Student: collect teacher data and train the MW manifold (uses the GPU automatically if one is available).
python train_mw_3d.py --env Humanoid-v5 --teacher ppo --latent 32 --hidden 256 --epochs 100 --rows 100000

# 3) Watch it:  http://127.0.0.1:5052
python app_3d.py
```

Useful knobs for `train_mw_3d.py`: `--latent`, `--hidden`, `--epochs`, `--rows` (subsample the teacher data),
`--workers` (CPU processes for MuJoCo rollouts), `--n_eval` (held-out seeds). Device selection is the framework's own
(`mw_framework.core.get_device`); force it with `MW_DEVICE=cpu` or `MW_DEVICE=cuda`.

Tools for experimenting with the student (all optional; defaults are unchanged):
`--data_cache data_cache/ppo_humanoid.npz` collects the teacher data once and reuses it across runs; `--name _tag` keeps
runs from overwriting each other; `--features gravity` switches orientation to a gravity vector in the torso frame, with
`--ang_prior` and `--sym_prior` adding angular-speed and left/right-hip priors; each run prints a posterior and
reconstruction check. `python eval_student.py --ckpt checkpoints_3d/<name>_mw.pth --query_steps 25 --query_lr 0.2` scores any
trained student on the held-out seeds and lets you vary the latent-query settings without retraining.

Fibre-bundle policy (state base, action fibre) — a good starting point:
```bash
python train_mw_3d.py --env Humanoid-v5 --teacher ppo --data_cache data_cache/ppo_humanoid.npz --epochs 1   # collects + caches teacher data
python train_mw_bundle_3d.py --data_cache data_cache/ppo_humanoid.npz --sigma_a 0.02 --epochs 40 --name _mine
```
`--sigma_a` is the action-noise scale of the fibre (smaller makes the state→action map tighter); `--latent_s/--hidden_s`
size the state base; `--no_priors` drops the physics laws for comparison.

Notes for `app_3d.py`: serving is pinned to CPU by default (`MW_DEVICE=cpu`) because the per-step latent query is tiny;
`MW_PORT`, `MW_HOST` and `MW_THREADS` are configurable; `/api3d/health` reports the device and loaded bodies; a
`ppo_teacher` mode appears automatically when the teacher files are present, so teacher and student can be compared side
by side. Trained student weights (`*.pth`) are not stored in git — train one locally with step 2.

## Hardware

- **Training:** a consumer NVIDIA GPU (an RTX 4060 Ti 16 GB was used here) speeds up the Pyro SVI step for the student.
  MuJoCo simulation itself runs on CPU cores, so a many-core CPU helps just as much for the teacher.
- **Deployment:** any CPU. No CUDA, no Omniverse, no Isaac.

## Results so far (held-out seeds, cap 500 steps, `Humanoid-v5`)

| Controller | Steps | Forward | Notes |
|---|---|---|---|
| CEM teacher (linear policy, ~770 params) | 89.4 | 0.69 m | measured |
| Plain-PyTorch CVAE on CEM data | 91.7 | 0.70 m | measured |
| **PPO teacher** (3M steps) | **577.8** (20 seeds) | **8.30 m** | measured |
| MW manifold distilled from the PPO teacher | — | — | **in progress** — tuning latent size, training length and the physics priors |

The teacher is the strong, repeatable part of the pipeline today; the work now is bringing the MW student up to it.
The planned improvements are listed in [../STUDENT_COLLAPSE_FIXES.md](../STUDENT_COLLAPSE_FIXES.md): gravity-vector
features, stronger priors (angular momentum, left/right symmetry), a posterior diagnostic, and a gentler latent query.
Projected figures in the older planning documents are targets, not measurements.

## Where this goes next

1. Student reaches the teacher's level (this directory).
2. DAgger-style self-improvement: query the teacher only where the student is uncertain.
3. Real-time CPU inference with a classical PD/LQR fallback.
4. Step 2 — remove the teacher entirely: [../for_full_replacement/](../for_full_replacement/).

See [../ROADMAP.md](../ROADMAP.md).

*Research code — not a safety system; do not use it to control a real robot.*
