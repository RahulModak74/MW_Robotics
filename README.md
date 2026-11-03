# MW Robotics

**Physics-grounded robot control that trains on a consumer GPU (or CPU) and deploys on a commodity CPU.**
Built on the Modak-Walawalkar (MW) method: learn from known physics instead of from billions of simulation steps.

We want a humanoid that balances, walks and reacts to pushes — the kind of capability
[ProtoMotions](https://nvlabs.github.io/ProtoMotions/tutorials/) shows — without needing a GPU cluster to get there and
without being locked to one vendor's runtime. The approach is deliberately pragmatic, in two steps:

| | What it is | Where | Status |
|---|---|---|---|
| **Step 1** | A GPU-trained RL **teacher** produces high-quality `(state, action)` data. The MW physics-constrained VAE learns from it and **runs on a plain CPU**. A consumer card (RTX 4060-class) is enough for training. | [`for_distilation/`](for_distilation/) | Teacher working; student under active development |
| **Step 2** | Remove the teacher: the MW VAE becomes the primary policy learner, with physics priors as the inductive bias, so **training and deployment both run on CPU**. | [`for_full_replacement/`](for_full_replacement/) | In progress — building blocks in place, first experiments underway |

The plan, with the reasoning behind each stage, is in **[ROADMAP.md](ROADMAP.md)**.
The next improvements planned for the Step 1 student are in **[STUDENT_COLLAPSE_FIXES.md](STUDENT_COLLAPSE_FIXES.md)**.

## What is already solid

These results come from the MW framework itself and all run on CPU:

- **Six physics domains** with working validators — a bending beam, a motor joint, a planar arm, a wheeled robot, a hopping
  leg, and motor heating — using one unmodified core method ([docs/ROBOTICS.md](docs/ROBOTICS.md)).
- **Forecasting** the next state faster than re-simulating, checked against MuJoCo on the hopping leg
  ([docs/FORECASTER.md](docs/FORECASTER.md)).
- **A coupled body** (Walker2d, then a 17-joint Humanoid): the validator carries over, and on the simulated falls tested
  it raised its warning before the simulator's own termination every time
  ([for_full_replacement/mw_to_protomotions/STATUS.md](for_full_replacement/mw_to_protomotions/STATUS.md)).
- **A strong 3D teacher**: PPO on Humanoid-v5 reaches ~578 steps and ~8.3 m on held-out seeds after 3M steps, and the
  trained teacher ships in the repo so you can start from it.
- **The same mechanism in production** outside robotics (battery-health prediction), with no RL and no GPU.

## Repository map

```
mw_framework/            the MW core library (physics-constrained Bayesian VAE, Pyro SVI)  -- shared by everything
examples/                six robot-physics examples (run_all_examples.sh)
mujoco_experiments/      the forecaster tested against MuJoCo
tests/                   unit tests
docs/                    ROBOTICS, FORECASTER, FIXING_THE_GAPS, LIMITATIONS

for_distilation/         STEP 1  GPU teacher -> MW student that deploys on CPU (training scripts + 3D demo app)
for_full_replacement/    STEP 2  CPU-native MW policy learning, no teacher (experiments and building blocks)
old_content/             earlier specs, plans, snapshots and the 2D demo, kept for reference

ROADMAP.md               the full plan
STUDENT_COLLAPSE_FIXES.md  next steps for the Step 1 student
```

## Quick start

```bash
git clone https://github.com/RahulModak74/MW_Robotics.git
cd MW_Robotics
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .                                       # installs mw_framework
pip install -r for_distilation/requirements.txt        # mujoco, gymnasium, stable-baselines3, flask, ...

python examples/00_flexible_robot_validator.py         # or 01..05
pytest -q
```

For GPU training, install a CUDA build of PyTorch first (any recent consumer NVIDIA card works), e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu128`, then check
`python -c "import torch; print(torch.cuda.is_available())"`. Everything also runs on CPU, just more slowly.
Serving a trained policy always runs on CPU.

Then continue in [for_distilation/README.md](for_distilation/README.md) for the 3D humanoid pipeline, or
[for_full_replacement/README.md](for_full_replacement/README.md) for the Step 2 work.

## Get involved

This is research code and it is moving quickly. Reproductions, new robot examples and experiment results — including your
own numbers — are very welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Questions and ideas: open an issue.

## Scope

Research code, not a finished robot and not a safety system. Do not use it to control real hardware without your own
safety engineering.

## License

Apache-2.0 (see `LICENSE`). If you use this, please cite `CITATION.cff` and the MW theory paper.
