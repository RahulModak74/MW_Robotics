# Step 2 — CPU-native MW policy learning (`for_full_replacement/`)

**Goal:** make the MW VAE the primary policy learner, so there is no RL teacher and no GPU anywhere between training and
deployment. Physics priors (joint limits, torque limits, ZMP, symmetry, energy) do the job that billions of simulation steps
do in conventional RL: they remove the physically invalid part of the search space before the first gradient step.

**Status: in progress.** The building blocks below are in place and have produced useful results; the end-to-end
teacher-free policy learner is the active research frontier. Contributions and ideas are very welcome.

## What is already in place

| Building block | Where | What it gives Step 2 |
|---|---|---|
| Coupled-body validator (Walker2d, 17-joint Humanoid) | `mw_to_protomotions/` (`walker_domain.py`, `stage1_validator_test.py`, `stretch_humanoid*.py`) | MW distance as a per-step "is this state/torque physically sensible?" signal |
| Early-warning monitor | `mw_to_protomotions/stage7_*.py` | On the simulated falls tested, the warning came before the simulator's own termination |
| Richer-data recipe | `mw_to_protomotions/stage4_*.py` | Diverse training data reduces false alarms on unseen physics (a measured dose-response curve) |
| CPU gradient-free search (CEM) over compact policies | `mw_to_protomotions/cem_policy_search.py`, `domain/29DoF/` | Teacher-free policy search that runs entirely on CPU; strong on Walker2d |
| Classical PD / LQR controllers | `mw_to_protomotions/classical_controller.py`, `lqr_balance.py`, `tune_balance.py` | The safety fallback for the "predict → (action, uncertainty, fallback)" contract |
| Transition model on the manifold | `mw_g1_racer/` (`transition_domain.py`, `train_transition_vae.py`, `b2_mpc.py`) | A learned dynamics model — the starting point for training a policy through the manifold |
| Forecasting against MuJoCo | `../mujoco_experiments/`, `../docs/FORECASTER.md` | Evidence that the manifold can stand in for a simulator at larger time steps |
| VAE-as-policy experiments | `domain/exp_e*.py`, `domain/EXP_03.md` | Compressing a policy into a VAE; the data pipeline Step 1 builds on |
| Lab notes | `domain/*.md`, `*/PROGRESS_LOG.md`, `mw_to_protomotions/STATUS.md` | Day-by-day record of every experiment and its numbers |

## The design choice that defines Step 2

An ELBO is a generative objective, so a reward needs a route into it. [ROADMAP.md](../ROADMAP.md) (Addendum B) lays out three
mechanisms to try, in order of how directly they use the MW machinery:

1. **Reparameterized rollouts through the learned dynamics model.** The VAE is the world model; the policy is amortized
   inference over it. The most direct form of "SVI as the primary learner".
2. **Variational policy gradient.** Sample from the decoder and use a baseline; physics priors prune the action space so far
   fewer samples are needed.
3. **Amortized inference + self-improvement.** Fit `q(z|s)` on seed data, then improve with uncertainty-gated queries
   (DAgger). Likely the first version to ship.

## First experiment

Before scaling the number of joints, the cheap decisive test (ROADMAP, Addendum C): a **planar hopper or 2-DoF biped with a
sparse forward reward, CPU-native SVI, no teacher and no reward shaping**. It is small enough to run in minutes on a laptop
CPU, and whichever way it goes it tells us exactly where to push next. Results will be added here.

## Running the existing experiments

Run from this directory so the packages `mw_to_protomotions` and `mw_g1_racer` resolve as before:

```bash
cd for_full_replacement
pip install -e ..                                        # mw_framework
python mw_to_protomotions/stage1_validator_test.py       # coupled-body validator
python mw_to_protomotions/cem_policy_search.py           # CPU CEM policy search on Walker2d
(cd mw_g1_racer && python validate_transition_vae.py)    # transition model check (reads its checkpoint from its own folder)
```

Most scripts print the settings they use, and the saved `*_full.log` files next to them show a complete run.
Shell runners for the earlier experiment plans are in `domain/` (`run_plan_a.sh`, `run_plan_b.sh`, `run_walker_demo.sh`).

## Where this fits

Step 1 ([../for_distilation/](../for_distilation/)) gets a working, CPU-deployable policy today with a GPU-trained teacher.
Step 2 removes the teacher. Both end in the same place: a physics-constrained policy that reports its own uncertainty and
runs on a $500-class CPU.

*Research code — not a safety system; do not use it to control a real robot.*
