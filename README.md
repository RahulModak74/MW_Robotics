# MW Robotics (MW: Modak-Walawalkar)

**An experimental, GPU-free approach to robot control: replacing data policies with physics policies.**

Today's best humanoid controllers are data policies: very large networks trained on enormous motion libraries in GPU data
centres. We are trying a different route. Keep the policy small enough for any CPU, write down the physics the robot must obey
(balance, capture point, torso upright), and let that physics do the safety work at run time. The policy and its flight computer need no GPU to run.

**Where it stands, in simulation: an 8 MB student policy plus a physics flight computer completes 58 of 58 of NVIDIA's tests, as
NVIDIA's own teacher does.\*** The flight computer is not yet fast enough for a real robot. Our near-real-time version completes 50 of
58, and closing that gap is an open problem that we are publishing unfinished, so that you can try:
[`step1_extraction/cpp_student/`](step1_extraction/cpp_student/README.md).

[![Student alone (left) falls; the same student + fly-by-wire (right) finishes](step1_extraction/flight_computer/results/video/fbw_D_dance_rescues.jpg)](step1_extraction/flight_computer/results/video/fbw_D_dance_rescues.mp4)

*Unitree G1 in MuJoCo, a dance it never saw, with action noise. Left: the student alone falls. Right: the same student with the
MW fly-by-wire finishes; the badge shows, per step, whether the flight computer passed the student's command through or corrected
it. [Video](step1_extraction/flight_computer/results/video/fbw_D_dance_rescues.mp4).*

| | What | Status |
|---|---|---|
| **Step 1: Bayesian policy extraction** ([`step1_extraction/`](step1_extraction/)) | A large policy (NVIDIA ProtoMotions' G1 tracker) → a small Bayesian student that runs on any CPU and knows when it is unsure → a physics flight computer that keeps it inside the balance envelope | **Done, in simulation: 58/58** |
| **Step 1b: the C++ student** ([`step1_extraction/cpp_student/`](step1_extraction/cpp_student/README.md)) | The same protection inside the robot's 20 ms control step: the student's uncertainty as a gate, and the look-ahead in C++ | **Open problem: 58/58 when slow, 50/58 when fast. Come and try** |
| **Step 2: CPU-compatible policy design** ([`step2_policy_design/`](step2_policy_design/README.md)) | Policies designed from scratch with Bayesian priors about the hardware, with no teacher; new bodies | **Plan, not yet a result** |

**What "GPU-free" means here, exactly.** At run time everything is CPU: the student, the flight computer and the tests.
Preparing the student (generating motions, recording the teacher, training) is a gaming-laptop job. The teacher we distil from
was trained by NVIDIA on GPUs; we do not retrain it. Removing the teacher as well is Step 2, and it is not done.

## An invitation

You do not need a lab, a robot or a GPU to work on this. A laptop, MuJoCo and curiosity are enough. We have written down what we
measured, including what failed, and ten ideas we have not had time to try, each with a cheap first test:
[`OPEN_PROBLEM.md`](step1_extraction/cpp_student/OPEN_PROBLEM.md). If one of them works, or if you see what we are missing, open
an issue. A clear negative result is just as welcome.

## How it works

```
 large policy (licence allows         small MW student, 8 MB             MW fly-by-wire (flight computer)
 derivative models), e.g. NVIDIA ───►  state manifold + Bayesian    ───►  exact physics model, 0.4 s look-ahead   ───► robot
 ProtoMotions' G1 tracker              action head: action + "how          with the student in the loop; passes the
                                       sure am I?"                         command unchanged unless the predicted state
                                                                           leaves the envelope (capture point, torso)
```

Like aircraft fly-by-wire: the student is the pilot and states intent; the flight computer knows the airframe exactly and keeps
it inside its envelope, overruling the pilot only as much as needed. The student's Bayesian uncertainty says *when* it is lost
(fall warning, AUROC 0.994–0.996); the flight computer is what *acts*.

## What we measured (Unitree G1, MuJoCo, ProtoMotions' own evaluator)

Teacher: NVIDIA ProtoMotions' G1 whole-body tracker (`g1-bones-deploy`). Students learn from the teacher driving the robot on 4 original
motions plus motions generated with NVIDIA's open Kimodo text-to-motion model (walking, gestures, exercise, everyday actions;
**no dance**). Exam: an unseen dance clip, run clean once and with action noise 0.1 and 0.2 (3 seeds each); 21 unseen familiar
motions; 30 unseen generated dances.

| Policy | Dance clip, 7 runs (95 % interval) | 21 unseen familiar motions | 30 unseen generated dances | Fall warning (AUROC, 95 % interval) |
|---|---|---|---|---|
| **MW student (100 motions) + MW fly-by-wire (envelope D)** | **7/7 (0.65–1.00)** | **21/21** | **30/30** | – (no falls left to warn about) |
| MW student (100 motions), alone | 3/7 (0.16–0.75) | 17/21 | 20/30 | 0.994 (0.989–0.998) |
| MW level 2 (324 motions), alone | 7/7 (0.65–1.00) | 20/21 | 24/30 | 0.996 (0.992–0.999) |
| MW level 1: Bayesian action head only (324 motions) | 5/7 (0.36–0.92) | 19/21 | 27/30 | 0.992 (0.978–1.000) |
| Plain MLP (324 motions) | 6/7 (0.49–0.97) | 18/21 | 26/30 | none of its own |
| Teacher (NVIDIA), trained on 142,000 motions | 7/7 | 21/21 | 30/30 | – |

* **58/58 with zero regressions.** The flight computer rescued all 18 tests the student failed alone and broke none that it passed.
  The student was not retrained. The flight computer's model matched the simulator to at most 1e-7 per step on every run.
* **What it costs:** joint-tracking error on the 40 tests that pass both ways rises by a median of 3.8 % (worst +23 %); the
  flight computer steers on about 20 % of steps. It trades a little tracking for balance.
* **What makes it work:** the capture-point law. An ablation on the dance clip: predicted falls only 5/7; + torso upright 4/7;
  + capture point 7/7. How the envelope was designed, and how to design one for another robot:
  [`docs/DESIGNING_THE_ENVELOPE.md`](docs/DESIGNING_THE_ENVELOPE.md).
* **The warning** is the students' own: a step counts as positive if a ProtoMotions failure follows within 0.5 s; every one of 26
  failures across three students was warned 0.16–0.96 s ahead. Per run, with definitions and intervals:
  [`EVIDENCE.md`](step1_extraction/student/results/EVIDENCE.md).
* **Size and speed:** the level-2 student is 1.65 M parameters (7.7 MB) and runs in **2.3 ms per control step on one thread of a
  laptop CPU** (Intel i5-12450H); the 100-motion student used with the flight computer is 8.4 MB (it adds a next-state head). The
  robot needs a decision every 20 ms.

\* **Limits, stated plainly.** Simulation only. One student with the flight computer so far; one run per motion library. The
flight computer's model is the simulator itself, and it reads the true state. It runs on CPU but **not yet in real time**: about
0.16 s per step when it lets the student through and about 3 s when it corrects. Next, to make the claim watertight: real-time
CPU, other students, stronger disturbances, a deliberately degraded robot with an inexact model, then hardware.

## The road to real time (the C++ student)

The robot needs a decision every 20 ms. What we have measured so far, same 58 tests, same student, one run each:

| Version | Score | Time per decision |
|---|---|---|
| Flight computer on every step | 58/58 | about 0.16 s to approve, about 3 s to correct |
| The student's own uncertainty decides when the flight computer acts | 58/58 | median 2.6 ms; 78.7 % of steps within 20 ms; the rest take seconds |
| The same, with a cheaper search | 50/58 | 82 % of steps within 20 ms; the rest up to about 0.3 s in Python |
| The search running in the background, its answer arriving late | 40/58 | every step under 20 ms |

The look-ahead now also runs in C++, with the student's weights inside it: the same decisions as the Python version, about
4.5 times faster (a full correction in about 0.26 s on a laptop). That is still 13 control steps. What we learned, the code and
the ideas still open: [`step1_extraction/cpp_student/`](step1_extraction/cpp_student/README.md).

## Verify it

* **In minutes, any CPU:** `python examples/policy_priors_demo.py 0,1,2,3,4` prints falls in 35 push tests for a plain student
  vs one trained with physics priors (`plain 17–18 | prior 0`, a pendulum; [`docs/BEYOND_THE_TEACHER.md`](docs/BEYOND_THE_TEACHER.md)).
* **The G1 result:** needs ProtoMotions, MuJoCo, NVIDIA's checkpoint and Kimodo; a gaming laptop is enough, but it is a day's
  setup. We ship no weights (they derive from NVIDIA's model); the scripts regenerate everything:
  [`step1_extraction/student/`](step1_extraction/student/README.md#reproduce), then
  [`step1_extraction/flight_computer/run_fbw_D.sh`](step1_extraction/flight_computer/README.md#reproduce).

## Which teacher policies you may use

Only policies whose licence allows it. Distillation creates a model *derived* from the teacher, so check: the **teacher's licence**
allows derivative models (and commercial use, if you need it; NVIDIA's G1 tracker is under the NVIDIA Open Model License, which
allows both, with an attribution notice when you redistribute); the licences of the **motions** and of any **text encoder** used to
generate them; and the **attribution** you must ship. Step by step: [`docs/POLICY_TO_PRODUCTION.md`](docs/POLICY_TO_PRODUCTION.md).

## Repository map

```
step1_extraction/
  student/            the small Bayesian student: collect teacher rollouts, train (MLP / Bayesian head / manifold), evaluate
  flight_computer/    the MW fly-by-wire: fbw.py, the 58-test protocol, tracking analysis, results and video     <- start here
  cpp_student/        towards real time: the uncertainty gate, the C++ look-ahead kernel, experiments, the open problem
step2_policy_design/  CPU-compatible policy design (plan) and the teacher-free pieces that exist
docs/                 how it works, designing the envelope, policy to production, beyond the teacher, limitations
examples/             pendulum demo: physics priors in training (minutes, any CPU)
```

Earlier material (a Humanoid-v5 distillation sandbox, Part 2 building blocks, six small PDE robot-physics examples and the MW core
library `mw_framework`) is kept under the git tag [`archive/pre-fbw`](https://github.com/RahulModak74/MW_Robotics/tree/archive/pre-fbw).
The general MW framework and its non-robotics examples: [MW_Geometric_Intelligence](https://github.com/RahulModak74/MW_Geometric_Intelligence).

## What MW is

MW (Modak-Walawalkar) is one recipe applied to different systems: **write the physics you know as Bayesian priors, learn a
low-dimensional manifold of the states that satisfy them, and keep the system on it.** In Step 1 the manifold is the set of normal
body states of a policy (the student), and the flight computer keeps the robot inside the physically accepted states. Background
papers by the authors (preprints on Zenodo, not peer reviewed):

* R. Modak, R. Walawalkar, *A Student-t(ν = 0.8) Prior on Measurement Processes Yields a Riemannian Manifold Simultaneously
  Consistent with General Relativity and Quantum Mechanics* (2026), [doi:10.5281/zenodo.19304813](https://doi.org/10.5281/zenodo.19304813).
* R. Modak, R. Walawalkar, *World Models with an Irreversible Time Axis* (2026),
  [doi:10.5281/zenodo.22954118](https://doi.org/10.5281/zenodo.22954118); not yet applied to the robotics code here.

## Quick start

```bash
git clone https://github.com/RahulModak74/MW_Robotics.git && cd MW_Robotics
python -m venv .venv && . .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python examples/policy_priors_demo.py 0,1,2,3,4
```

## Why we build it this way

Most of the world's engineers and students do not have a GPU cluster. They have a laptop. If good robot control needs only a
small policy and the laws of physics, then anyone with a laptop can study it, improve it and build on it, with no closed
runtimes, licence servers or hardware that only a few can buy. That is who this repository is for. We use whatever compute is at
hand, including the consumer GPUs many teams already own; our stance on GPUs is pragmatic, not dogmatic.

## Join in

Reproductions (successful or not), attempts at the [open problem](step1_extraction/cpp_student/OPEN_PROBLEM.md), new policies with licences that allow it, new robots and honest failures:
[`CONTRIBUTING.md`](CONTRIBUTING.md). Research code; simulation results only; not a safety system. Apache-2.0 ([`LICENSE`](LICENSE));
please cite [`CITATION.cff`](CITATION.cff).
