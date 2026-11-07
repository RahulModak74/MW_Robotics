# MW Robotics (MW: Modak-Walawalkar)

**Take a robot policy trained by a leading lab, turn it into a small Bayesian policy that runs on an ordinary CPU, and that warns
you before it is about to fail.**

| | What it is | Status |
|---|---|---|
| **Part 1: policy distillation, powered by Bayesian methods** ([`for_distilation/`](for_distilation/)) | A licensed teacher policy (for example NVIDIA ProtoMotions' Unitree G1 tracker) is distilled into a ~7 MB MW policy that runs on any CPU, down to edge devices, and raises a warning before a mistake happens | **Ready.** Unitree G1 dances a clip it never saw, 7/7 runs; warning before falls at AUROC 0.996 |
| **Part 2: policy generation, together** ([`ROADMAP.md`](ROADMAP.md)) | Heavier, physics-fortified policies, trained with whatever compute is needed, aimed at replacing parts of today's policy-generation pipelines | Research, open to partners |

[![MW student (top) vs teacher (bottom) on the unseen dance clip](for_distilation/protomotions_g1/results/video/mw_vs_teacher_dance.jpg)](mw_vs_teacher_dance.mp4)

*MW student vs NVIDIA's teacher on a dance clip the student never saw ([video](mw_vs_teacher_dance.mp4)). Simulation (MuJoCo).*

## Part 1 in one picture

```
 teacher policy (licence allows        drive the robot in simulation,       MW policy, ~7 MB          any CPU: laptop,
 derivative models), e.g. NVIDIA  ───► record (what it sees, what it  ───► state manifold      ───►  robot computer,
 ProtoMotions' G1 tracker              does), on a few hundred motions      + Bayesian action          edge device
                                                                            head                       │
                                                                                                       ▼
                                                                    action  +  "how sure am I?"  ──►  warning / safe fallback
```

Leading labs already write excellent robot policies with large GPU clusters. Part 1 does not compete with that; it makes those
policies **cheap to run, easy to audit, and safe to deploy**:

1. **Small:** the dancing G1 policy has 1.65 million parameters (7.7 MB) and takes **2.3 ms per control step on one thread of a
   laptop CPU** (Intel i5-12450H), including its uncertainty. The robot needs one decision every 20 ms.
2. **Honest about what it knows (the main value):** every step, the policy reports how far the situation is from what it learned. On
   unseen dances that signal **predicts a fall within the next 0.5 s with AUROC 0.996** (0.997 on unseen everyday motions), so a
   fallback controller can take over *before* the wrong step happens.
3. **Robust:** it kept the robot dancing under heavy action noise where the plain copy fell.

## The result: Unitree G1 dances a clip it never saw

Teacher: NVIDIA ProtoMotions' G1 whole-body tracker (`g1-bones-deploy`). Training data: the teacher driving the robot on 4 original
motions plus **324 distinct motions generated with NVIDIA's open Kimodo text-to-motion model** (walking, gestures, exercise, everyday
actions; **no dance at all**). We prepared 500, but the teacher's rollouts picked motions at random and reached 324 of them; a run that
visits all 500 is under way. Exam: a dance clip, run clean once and with random action noise 0.1 and 0.2 (3 seeds each), scored by
ProtoMotions' own evaluator in MuJoCo.

| Policy | Dance clip (7 runs) | under heavy noise (0.2) | Warns before a fall? |
|---|---|---|---|
| Plain MLP (a straight copy of the teacher) | 6/7 | 2/3 | no signal of its own |
| MW level 1 (Bayesian action head only) | 5/7 | 1/3 | yes: AUROC 0.990 |
| **MW level 2 (state manifold + Bayesian action head)** | **7/7** | **3/3** | **yes: AUROC 0.996** |
| Teacher (NVIDIA, for reference) | 7/7 | 3/3 | – |

The manifold is what made the difference: the same Bayesian head without it did worse than the plain copy. Full tables,
including 21 unseen familiar motions and 30 unseen generated dances, and how to reproduce:
[`for_distilation/protomotions_g1/`](for_distilation/protomotions_g1/README.md).

### Two kinds of students, nothing in between

To keep this simple we compare only two families:

* **Plain MLP:** an ordinary neural network fitted to the teacher's actions. It copies well, but it cannot tell you when it is out of
  its depth.
* **MW, at increasing levels:** every MW policy is Bayesian and reports its own uncertainty.
  * *MW level 1:* Bayesian action head only (an exact posterior over the last layer).
  * *MW level 2:* **state manifold + Bayesian action head.** This is the 7/7 policy. It first places the robot's situation on a
    learned map of normal body states (the manifold), then reads the action from that place.
  * *MW level 3 (in progress):* level 2 plus physics priors (balance, upright torso, joint limits) checked on the consequences
    of each action. See [`docs/BEYOND_THE_TEACHER.md`](docs/BEYOND_THE_TEACHER.md).

How a policy becomes a manifold, in plain words: [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md).

### More motions, better policy

| Training motions (besides the 4 originals) | MW policy on the dance clip |
|---|---|
| none | 0/9 runs |
| 100, drawn at random | 1/9 |
| 100, chosen evenly across the four kinds of motion | 3/7 |
| **324 generated motions** (of 500 prepared) | **7/7** |

The student needs to have seen enough of the space of normal body states; the dance is then just a new path through it.

## Which teacher policies you may use

Only policies whose licence allows it. Distillation creates a model *derived* from the teacher, so check, before you start:

* the **teacher's licence** allows derivative models (and commercial use, if you need it). NVIDIA's G1 tracker is under the
  NVIDIA Open Model License, which permits derivative models, including commercially, with an attribution notice when
  you redistribute them;
* the licences of the **motions** you train on and of any **text encoder** used to generate them;
* **attribution** notices travel with what you ship.

This repository contains our code only: no NVIDIA weights, no generated motions, no trained student weights. The scripts
regenerate everything. Step by step: [`docs/POLICY_TO_PRODUCTION.md`](docs/POLICY_TO_PRODUCTION.md).

## Part 2: policy generation, together

Distillation needs a teacher. Part 2 is about producing **heavily fortified policies** in the first place, and replacing *parts*
of today's policy-generation pipelines with Bayesian, physics-first pieces: physics priors that rule out impossible behaviour
before training starts, a manifold of accepted (balanced, recoverable) states, and an uncertainty signal built in from day one.

This is heavy work, and the enormous power of GPUs can and should be used for it. **Our stance on GPUs is pragmatic, not
dogmatic:** we use whatever is easily available, such as the consumer GPUs (gaming laptops) that most teams in the Global South
already own, and we welcome what large clusters can do. **We oppose barriers, not technology:** closed runtimes, licence
servers, and hardware that only a few can buy. We are looking for labs and teams to build these policies together. See
[`ROADMAP.md`](ROADMAP.md).

## What MW is

MW (Modak-Walawalkar) is a Bayesian method: physics knowledge goes in as **priors**, which cut the huge space of possible
behaviours down to a low-dimensional **manifold** of physically valid states. A policy then becomes a map on that manifold, and
"how far is this from the manifold?" is a built-in safety signal. Theory and non-robotics examples (for example battery health in
production, no GPU) are in [MW_Geometric_Intelligence](https://github.com/RahulModak74/MW_Geometric_Intelligence).

## Quick start

```bash
git clone https://github.com/RahulModak74/MW_Robotics.git && cd MW_Robotics
python -m venv .venv && . .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e . && pip install -r for_distilation/requirements.txt
python for_distilation/protomotions_g1/examples/policy_priors_demo.py   # the "beyond the teacher" idea, CPU, a few minutes
pytest -q
```

The G1 pipeline itself needs ProtoMotions, MuJoCo and the teacher checkpoint: see
[`for_distilation/protomotions_g1/README.md`](for_distilation/protomotions_g1/README.md#reproduce). A consumer NVIDIA GPU speeds up
training (install a CUDA build of PyTorch first); everything also runs on CPU.

## Repository map

```
for_distilation/          Part 1
  protomotions_g1/        the G1 dance result: code, results, video          <- start here
  *.py, app_3d.py         earlier distillation pipeline on MuJoCo Humanoid-v5 (PPO teacher), with a 3D viewer
for_full_replacement/     Part 2 building blocks (validators, early warning, classical fallbacks, transition models)
mw_framework/             MW core library (physics-constrained Bayesian VAE, Pyro SVI)
docs/                     how it works, policy to production, beyond the teacher, six physics examples, limitations
examples/ tests/ mujoco_experiments/
```

## Documents

| Read | For |
|---|---|
| [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) | how a teacher policy becomes an MW manifold policy, in plain words |
| [`docs/POLICY_TO_PRODUCTION.md`](docs/POLICY_TO_PRODUCTION.md) | taking a new policy from licence check to a CPU or edge deployment, with the tests it must pass |
| [`docs/BEYOND_THE_TEACHER.md`](docs/BEYOND_THE_TEACHER.md) | how physics priors let a student outgrow its teacher (pendulum demo) |
| [`ROADMAP.md`](ROADMAP.md) | Part 2 and what comes next |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | where priors help, where they do not |

## Join in

Reproductions, new teacher policies (with licences that allow it), new robots and honest failures are all welcome: see
[`CONTRIBUTING.md`](CONTRIBUTING.md). Research code, simulation results; not a certified safety system. Apache-2.0 (see
[`LICENSE`](LICENSE)); please cite [`CITATION.cff`](CITATION.cff).
