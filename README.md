# MW Robotics

**Humanoid control policies that train and run on a CPU — for teams who can't afford the Isaac stack.**

Modern humanoid control (NVIDIA Isaac / ProtoMotions) is built on data-center GPUs and a vendor runtime. MuJoCo gives
everyone a free, excellent physics engine, but it is a simulator, not a policy trainer: there is no MuJoCo-native equivalent
of Isaac's training stack. **MW Robotics fills that gap** with a physics-constrained policy learner
(the Modak-Walawalkar method) that needs thousands of samples rather than billions, and is designed to deploy on a
$500-class CPU.

## The goal: CPU only, end to end

No GPU for training. No GPU for deployment. No Omniverse, no Isaac, no licence server. Physics priors (joint and torque
limits, balance, symmetry) replace brute-force simulation, and every policy reports how far it is from what it was trained on,
so a classical controller can take over when it should.

## Two steps, one destination

Many teams in the Global South already own a consumer GPU. So we get there in two steps:

| Step | What it is | Status |
|---|---|---|
| **Step 1 — consumer-GPU support** [`for_distilation/`](for_distilation/) | Use the card you already have (an RTX 4060-class is plenty) to train a standard RL teacher, and distill it into an MW policy that **runs on a plain CPU**. Useful today. | Teacher working, student in development |
| **Step 2 — full replacement** [`for_full_replacement/`](for_full_replacement/) | Drop the teacher and the GPU entirely: the MW model learns the policy directly on CPU. **This is the end goal.** | In progress |

The plan, stage by stage: **[ROADMAP.md](ROADMAP.md)**. Next steps for the Step 1 student:
**[STUDENT_COLLAPSE_FIXES.md](STUDENT_COLLAPSE_FIXES.md)**.

## What already works (all on CPU)

- **Six physics domains** with validators — beam, motor joint, planar arm, wheeled robot, hopping leg, motor heat
  ([docs/ROBOTICS.md](docs/ROBOTICS.md)).
- **Forecasting** the next state faster than re-simulating, checked against MuJoCo
  ([docs/FORECASTER.md](docs/FORECASTER.md)).
- **Coupled bodies** (Walker2d and a 17-joint Humanoid): on the simulated falls tested, the validator warned before the
  simulator's own termination ([for_full_replacement/mw_to_protomotions/STATUS.md](for_full_replacement/mw_to_protomotions/STATUS.md)).
- **A strong humanoid teacher** (PPO, ~578 held-out steps, 8.3 m), shipped so you can start from it.
- **The same mechanism in production** outside robotics (battery-health prediction): no RL, no GPU.

## Quick start

```bash
git clone https://github.com/RahulModak74/MW_Robotics.git && cd MW_Robotics
python -m venv .venv && . .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e . && pip install -r for_distilation/requirements.txt

python examples/00_flexible_robot_validator.py          # or 01..05
pytest -q
```

Have a consumer NVIDIA GPU? Install a CUDA build of PyTorch first
(`pip install torch --index-url https://download.pytorch.org/whl/cu128`) and Step 1 will use it. Without one, everything
still runs on CPU. Then continue in [for_distilation/](for_distilation/README.md) or
[for_full_replacement/](for_full_replacement/README.md).

## Repository map

```
mw_framework/            MW core library (physics-constrained Bayesian VAE, Pyro SVI)
examples/  tests/  mujoco_experiments/  docs/
for_distilation/         Step 1: consumer-GPU teacher -> CPU policy, plus a 3D demo viewer
for_full_replacement/    Step 2: CPU-native policy learning
old_content/             earlier specs, plans and the 2D demo, kept for reference
```

## Join in

Reproductions, new robot examples and your own results are very welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

*Research code, not a finished robot and not a safety system. Apache-2.0 (see `LICENSE`); please cite `CITATION.cff`.*
