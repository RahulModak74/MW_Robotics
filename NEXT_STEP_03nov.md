# NEXT_STEP_03nov.md — handoff for the next session (any machine, any agent)

Written 3 Nov 2026. Read this first, then `README.md` and `ROADMAP.md`. Everything below is reproducible from the repo
alone; nothing depends on the machine it was produced on.

## 1. Where we are

The project has two steps ([ROADMAP.md](ROADMAP.md)): **Step 1** a consumer-GPU-trained RL teacher distilled into an MW
policy that runs on CPU (`for_distilation/`), and **Step 2** the same without any teacher (`for_full_replacement/`, early).
All work in this session was Step 1 on `Humanoid-v5`.

**Headline:** the MW policy rebuilt as a **fibre bundle** (state = base manifold, 17 actions = fibre over it, one forward
pass) now matches the PPO teacher when nothing disturbs it. The remaining gap is **robustness to disturbances**, and that is
the next piece of work.

Measured on 30 held-out seeds (5000–5029), clean, 1000-step cap, mean steps survived (share reaching the cap):

| Policy | Clean | Action noise 0.3 | Lateral push 1 m/s |
|---|---|---|---|
| PPO teacher (3M steps, shipped in `checkpoints_3d/`) | 602 (20%) | 476 | 351 |
| Plain MLP behaviour cloning | 802 (60%) | 512 | 305 |
| Plain CVAE | 620 (20%) | 470 | 304 |
| **MW fibre bundle, `--sigma_a 0.02`, laws on** | **615 (13%)** | 306 | 159 |
| MW fibre bundle, latent 64 / hidden 512, 80 epochs | 609 (13%) | 272 | 220 |
| Earlier flat MW student (joint VAE over (s, a), latent query) | 92 | 88 | 90 |

Same policies at a 500-step cap: teacher 431, MLP 459, CVAE 439, MW bundle 440. (Your own earlier teacher number of 577.8
used a 1000-step cap on 20 seeds; at that basis the teacher is ~600, consistent.) Differences of a few tens of steps between
rows are inside the noise of 30 seeds — do not rank rows that close.

## 2. Reproduce the current state (about 30 minutes on a consumer GPU + 10 CPU cores)

```bash
git clone https://github.com/RahulModak74/MW_Robotics.git && cd MW_Robotics
python -m venv .venv && . .venv/bin/activate                       # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the CPU wheel; everything also runs on CPU
pip install -e . && pip install -r for_distilation/requirements.txt pytest scikit-learn pandas
cd for_distilation

# 1) collect + cache the teacher data once (600 rollouts: seeds 0-199 x noise 0/0.1/0.3; ~246k rows). ~5 min.
python train_mw_3d.py --env Humanoid-v5 --teacher ppo --data_cache data_cache/ppo_humanoid.npz --epochs 1 --workers 10 --n_eval 2
# 2) train the fibre-bundle policy (the recommended setting)                           ~15 min with laws on (36 glue laws)
python train_mw_bundle_3d.py --data_cache data_cache/ppo_humanoid.npz --sigma_a 0.02 --epochs 40 --name _b5_s02 --workers 10
# 3) plain baselines on the same rows (produces data_cache/std_mlp.pt, std_cvae.pt) and the comparison table
python compare_mw_vs_std.py --data_cache data_cache/ppo_humanoid.npz --mw_ckpt checkpoints_3d/humanoid_ppo_e0_legacy_mw.pth
python robustness_compare.py --n 30 --cap 1000 --conditions clean,noise0.3,push1 \
  --policy ppo_teacher=checkpoints_3d/humanoid_ppo.zip --policy mlp=data_cache/std_mlp.pt --policy cvae=data_cache/std_cvae.pt \
  --policy mw_bundle=checkpoints_3d/humanoid_ppo_bundle_b5_s02.pt
```

Notes: the step-3 comparison script needs an old flat checkpoint (`..._e0_legacy_mw.pth`); to avoid that, train one with
`python train_mw_3d.py --teacher ppo --latent 32 --hidden 256 --epochs 60 --rows 100000 --data_cache data_cache/ppo_humanoid.npz --name _e0_legacy`,
or just skip the `mwflat`/`mw_z` rows. `*.pth`, `data_cache/`, logs and experiment reports are git-ignored on purpose.

## 3. What each file does (all under `for_distilation/` unless noted)

| File | Role |
|---|---|
| `mw_bundle_policy.py` | `build_bundle()` (ManifoldAlgebra: product "state" + fibre "action" + 36 physics glue laws), `BundlePolicy` (`act`, `distances`, save/load), `evaluate_bundle()` |
| `train_mw_bundle_3d.py` | trains the bundle by Pyro SVI from the cached data; prints a one-pass state→action error and held-out steps |
| `train_mw_3d.py` | flat MW manifold; also the **teacher-data collector/cache** (`--data_cache`), posterior check, `--features gravity`, priors |
| `eval_student.py` | scores a flat MW checkpoint with adjustable latent-query settings (CPU workers) |
| `compare_mw_vs_std.py` | trains the plain MLP/CVAE on the same rows; compares policies; early-warning AUROC (MW distance vs Mahalanobis vs kNN) |
| `robustness_compare.py` | all policies under action noise and pushes, any cap, includes the PPO teacher |
| `app_3d.py` | Flask + Three.js viewer; **serves flat MW checkpoints only** (`<body>_mw.pth`); CPU-pinned; has a `ppo_teacher` mode. Not yet wired to the bundle policy |
| `mw_framework/algebra.py` | `ManifoldAlgebra`; this session added an optional `hidden=` to `product()` / `fibre()` |
| `../STUDENT_COLLAPSE_FIXES.md` | the earlier diagnosis document; see section 5 for what was checked against it |

## 4. Next steps, in priority order

1. **DAgger gated by MW distance** (aimed at the robustness gap). Roll out the bundle student (with action noise and pushes in
   the mix), keep the states where its MW distance is high (`BundlePolicy.distances` gives total and per-piece distance),
   label those states with the PPO teacher (`PPO.predict` on the full observation — cheap), append, retrain 10–20 epochs,
   repeat 3–4 times. Success = the noise-0.3 and push-1 columns above rise toward the teacher's (about 476 and 351) without the
   clean column dropping. Needs a new script (suggested: `dagger_bundle_3d.py`); the data cache stores both feature layouts
   but is clean-rollout data only, so DAgger rollouts must be collected fresh.
2. **Make the physics laws earn their place.** The current laws (height envelope, unit quaternion, joint limits, torque
   limits) did not improve robustness in the ablation (`--no_priors` runs were equal or slightly better under noise). Try laws
   that describe balance and dynamics: ZMP / capture point, torso angular-momentum bound, per-step energy; the hip anti-phase
   law (`--sym_prior`; correlation of right/left hip_y torque in the teacher data is -0.35, so use a loose tolerance); and
   tolerance / weighting of the existing glue laws (their early-training weight is large — try warm-up).
3. **MW distance as the fallback trigger.** `BundlePolicy.distances` + a classical PD/LQR controller from
   `for_full_replacement/mw_to_protomotions/classical_controller.py`. As a fall warning on time-matched windows MW distance
   scored AUROC ~0.86, about equal to Mahalanobis (0.87) and kNN (0.86); the open question is whether per-piece distances
   (state vs action) or the physics terms beat those.
4. **Serve the bundle in `app_3d.py`** (load `<body>_bundle*.pt` next to the `_mw.pth` bodies; `act` has the same signature
   as `ManifoldPolicy.act`, `observe(data)` as well) and add a "compare" view with the teacher.
5. **Teacher quality:** the shipped PPO teacher was trained for 3M steps; `docs/NEW_APPROACH.md` describes the longer
   (5M–10M step, entropy bonus, LR anneal) recipe. A stronger teacher raises the ceiling for every student.
6. **Step 2** (`for_full_replacement/README.md`): the first experiment is a planar hopper / 2-DoF biped with sparse forward
   reward, CPU-native SVI, no teacher (ROADMAP, Addenda B and C). Nothing has been run for this yet.

## 5. Things already tried (do not repeat), with the outcome

- **Gravity-vector features** (`--features gravity`) instead of the raw quaternion: not better for the flat MW student (58 vs
  97 steps, 10 seeds). Kept as an option; default stays `legacy`. The bundle uses the legacy layout.
- **Angular-speed prior** (`--ang_prior`) on the flat student: no gain (63 vs 58).
- **Latent-query settings** for the flat student (steps 5–100, lr 0.05–0.3): the default 25 steps at lr 0.2 was best; fewer steps
  were worse. Choosing the action by MW distance (CEM from the latent query's action) gave 125 vs 97 steps.
- **Capacity and epochs** for the flat student: latent 8 → 32 and hidden 128 → 256 doubled its score; 7× more epochs at the
  small size added little.
- **Posterior check**: no collapse (KL ~82, all 32 latent dims active). The flat model reconstructs (s, a) well when given
  both; the loss comes from inferring the action from the state alone, which the fibre bundle removes.
- **Fibre-bundle settings:** `--sigma_a` is the dominant knob: 0.2 → 188 steps, 0.05 → 322, 0.02 → 432 (10 seeds, 500 cap).
  Latent 64 / hidden 512 / 80 epochs is about as good as the default, not better.
- **Corrections to `STUDENT_COLLAPSE_FIXES.md`** found while implementing it: the torso is MuJoCo body 1 (body 0 is the world);
  a free joint's angular velocity is already in the body frame; the hip actuators are `a5` (right_hip_y) and `a9`
  (left_hip_y), not `a2`/`a8`; and the proposed energy prior's limit of 10 can never trigger (17 actuators x 0.4 = 6.8 max).

## 6. Practical notes and gotchas

- **Environment used:** Windows 11, Python 3.12, torch 2.11+cu128, mujoco 3.17, gymnasium 1.4, stable-baselines3 2.9,
  pyro-ppl 1.9, RTX 4060 Ti 16 GB, 12 logical CPU cores. Teacher rollouts and every evaluation run on CPU worker processes
  (`spawn`); keep the `if __name__ == "__main__":` guards. Use `--workers` ~ your core count minus 1–2.
- **CPU serving is intentional.** `app_3d.py` sets `MW_DEVICE=cpu`; `eval_student.py` does too. Training scripts use CUDA
  automatically when available (`MW_DEVICE=cpu` forces CPU).
- **Do not use a broad `taskkill`/`pkill` on python** while other training runs are active; stop jobs by PID or command line.
- **`pytest`:** `tests/test_mw.py::test_sklearn_sign_convention` fails on the original repo too (a statistical assertion), and
  three `tests/test_transition_gate.py` tests error on Windows because they hard-code `/tmp`. Neither is related to this work.
- **Noise level of the comparisons:** with 10 seeds the run-to-run spread is about ±40 steps; use 30 seeds for anything you
  intend to claim, and always state the cap (500 or 1000).
- **Layout:** `old_content/` is reference material only. `for_full_replacement/` scripts expect their own folder as the working
  directory (they import `mw_to_protomotions` / `mw_g1_racer` as top-level packages; `conftest.py` handles this for pytest).
- **Credentials:** GitHub access in this session used a personal token kept outside the repo. Never commit a token; pass it per
  command (for example via `git -c http.extraheader=...`) or use your normal git credential helper.
- **Public wording:** README and ROADMAP describe robustness / off-distribution safety as goals. Keep them as goals until a
  measured result supports them; update the results table in `for_distilation/README.md` when the bundle numbers are final.

*Research code. Not a safety system.*
