# Part 1: policy distillation, powered by Bayesian methods (`for_distilation/`)

Take a policy that a lab trained with large GPUs (and whose licence allows derivative models), watch it drive the robot in
simulation, and distill it into a small **MW policy** that runs on any CPU and reports, every step, how sure it is.

```
 teacher policy ──(what it sees, what it does)──► MW policy: state manifold + Bayesian action head ──► CPU / edge device
  (GPU-trained)        a few hundred motions        (Pyro, consumer GPU or CPU to train)                action + uncertainty
```

## Start here: [`protomotions_g1/`](protomotions_g1/README.md)

The ready result. NVIDIA ProtoMotions' Unitree G1 tracker distilled into a 7.7 MB MW policy trained on 500 generated motions:

* an **unseen dance clip, 7/7 runs**, including heavy action noise (plain MLP 6/7);
* **warning before a fall at AUROC 0.996** on unseen dances;
* **2.3 ms per step on one laptop CPU thread**.

Code, result tables, video and the full reproduction recipe are in that folder. To take another policy through the same steps:
[`../docs/POLICY_TO_PRODUCTION.md`](../docs/POLICY_TO_PRODUCTION.md).

## Also here: the first pipeline (MuJoCo `Humanoid-v5`, PPO teacher)

Our first distillation pipeline, kept because it is self-contained (no ProtoMotions needed) and comes with a 3D browser viewer. The
teacher is strong; the MW student on this task is **still below the teacher** (work in progress), so treat it as a sandbox.

| File | Purpose |
|---|---|
| `train_ppo_teacher_3d.py` | stable-baselines3 PPO teacher on `Humanoid-v5` (parallel CPU environments) |
| `train_mw_3d.py`, `mw_policy_3d.py` | MW policy manifold (`PolicyManifold3DVAE`, physics priors) trained by Pyro SVI on the teacher's data; the controller is a query on the manifold |
| `mw_bundle_policy.py`, `train_mw_bundle_3d.py` | the MW policy as a fibre bundle: state = base manifold, actions = fibre over it, one forward pass |
| `train_policy_3d.py`, `policy_3d.py` | a small CEM teacher and a plain CVAE baseline; features and environment factory |
| `compare_mw_vs_std.py`, `robustness_compare.py`, `eval_student.py` | side-by-side scoring against plain MLP / CVAE policies, clean and under noise and pushes |
| `app_3d.py`, `static_3d/` | Flask + Three.js viewer (inference on CPU) |
| `checkpoints_3d/` | a trained 3M-step PPO teacher (577.8 held-out steps over 20 seeds, 8.3 m), so you can start from the data step |

```bash
cd for_distilation && pip install -r requirements.txt
python train_ppo_teacher_3d.py --env Humanoid-v5 --timesteps 5000000 --envs 10      # optional: a teacher is included
python train_mw_3d.py --env Humanoid-v5 --teacher ppo --latent 32 --hidden 256 --epochs 100 --rows 100000
python app_3d.py                                                                      # http://127.0.0.1:5052
```

Device selection follows `mw_framework.core.get_device` (`MW_DEVICE=cpu` or `cuda` to force it). Trained student weights are
not stored in git.

## Hardware

* **Training:** a consumer NVIDIA GPU (gaming laptop class) makes training fast; a CPU works, slower. Simulation runs on CPU cores.
* **Deployment:** any CPU. No CUDA, no vendor runtime, no licence server.

*Research code, simulation results; not a certified safety system.*
