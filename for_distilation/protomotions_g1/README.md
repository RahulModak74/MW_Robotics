# MW student for NVIDIA ProtoMotions' Unitree G1 tracker — the dance test (`for_distilation/protomotions_g1/`)

A 1.6-million-parameter **MW student** (about 7 MB, runs on a CPU) distilled from NVIDIA ProtoMotions' G1 whole-body motion tracker
(`g1-bones-deploy`, trained by NVIDIA with reinforcement learning on the BONES-SEED motion corpus). The exam is a **dance clip the
student never saw**; not one dance motion is in its training data. Everything runs in MuJoCo and is scored by **ProtoMotions' own evaluator**.

![MW student (top) vs teacher (bottom) on the held-out dance clip](results/video/mw_vs_teacher_dance.jpg)

Video: [`results/video/mw_vs_teacher_dance.mp4`](results/video/mw_vs_teacher_dance.mp4) (MW student left, teacher right, 5 s).

## Result

Training data: the 4 original ProtoMotions motions (walk, wave, high-five, jumping jacks) plus **500 motions generated with NVIDIA's
open Kimodo text-to-motion model** (locomotion, gestures, exercise, everyday actions; all dance prompts held out).
The dance clip is run clean once and with random action noise 0.1 and 0.2 (3 seeds each): 7 independent runs per student.

| Student | Dance clip (7 runs) | under heavy noise (0.2) | 21 unseen familiar motions | 30 unseen Kimodo dances |
|---|---|---|---|---|
| Plain MLP | 6/7 | 2/3 | 18/21 | 26/30 |
| Gaussian + last-layer Laplace (Bayesian) | 5/7 | 1/3 | 19/21 | 27/30 |
| **MW v2 + last layer** | **7/7** | **3/3** | **20/21** | 24/30 |
| Teacher | 7/7 | 3/3 | 21/21 | 30/30 |

* **Only the MW student never fell on the dance clip**, including every heavy-noise run, and it tracked it most closely of the students
  (worst joint error 0.25 rad vs about 0.40; position error 0.13 m vs 0.27 m).
* **It knows when it is in trouble:** its Bayesian uncertainty predicts an upcoming fall (within 25 steps) with AUROC **0.996** on the unseen
  dances; a standard novelty score (Mahalanobis distance on the observation) gets 0.935. The MW manifold distance alone gets 0.946.
* **More evidence is what made the difference:** trained on the 4 original motions only, MW failed the dance in all runs (0/9); with
  100 extra motions 1/9; with 500, 7/7. Full tables: [`results/SUMMARY_dance_s0.md`](results/SUMMARY_dance_s0.md) (4 motions),
  [`results/SUMMARY_quick100.md`](results/SUMMARY_quick100.md) (+100), [`results/SUMMARY_m500.md`](results/SUMMARY_m500.md) (+500).
* **Open:** on the wider set of 30 unseen dances MW completes 24 so far. Next: more generated motions and the physics priors below.

This is distillation: the student learns from a teacher that already learned from the full motion corpus. The claim is that a small
CPU model can take on the skill from a few hundred motions, robustly and with a reliable "I am about to fail" signal.

## What the MW student is

* **State manifold:** an encoder q(z|o) maps the 349-number observation to a 16-dimensional latent; a decoder reconstructs the observation
  (the MW distance = how far a state is from the learned manifold of valid body states).
* **Action head on [o, z]:** the action is read from the observation *and* its place on the manifold (a skip path keeps precision).
* **Last-layer Laplace:** an exact Bayesian posterior over the final layer (closed form, one pass) gives the epistemic uncertainty that warns before falls.
* Trained with a conditional ELBO (Pyro SVI), accuracy first. Code: `ours/student_prob.py` (`MWBundleV2`, `LastLayer`), `ours/train_student_prob.py --kind mwv2 --lastlayer`.

## Physics ("dance") priors — in progress

`ours/priors.py` and `ours/g1_kin.py` (differentiable G1 kinematics, matches MuJoCo to 3e-9 m) define motion-agnostic laws on the predicted
next state: capture-point balance, upright torso, joint limits, follow the reference. Tolerances are the teacher's own range. `ours/train_student_priors.py`
(MW v3) enforces them on the consequences of the student's own actions, up to 10 steps ahead, including on states the teacher never visited.
The pendulum demo [`examples/policy_priors_demo.py`](examples/policy_priors_demo.py) shows the idea in plain PyTorch: under hard pushes a plain student
falls 18/35 times, a student with a physics prior 0/35. The humanoid run (priors vs identical control on the 500 motions) is training now; results will be added here.

## Reproduce

Needs a ProtoMotions clone with its MuJoCo dependencies and the `g1-bones-deploy` checkpoint (`PM=~/ProtoMotions` by default; see `scripts/common.sh`),
Kimodo (`~/kimodo`, for generating motions) and PyTorch + Pyro (GPU optional; the students train in minutes on a laptop GPU, and run on CPU).

```bash
# 1. generate motions (Kimodo; the Llama-3-8B text encoder is loaded in 4-bit)
cd kimodo_data && ~/kimodo/venv/bin/python encode_prompts.py --prompts prompts.json --out emb/ \
   && ~/kimodo/venv/bin/python gen_motions.py --prompts prompts.json --emb emb/ --out csv/
# 2. select 500 training motions, convert, collect teacher rollouts, train the 3 students, evaluate, summarise
./m500.sh                        # stages: select convert collect train eval agg (each waits for CPU/memory < 70 %)
# 3. video of any run
#    pm_eval_student.py ... --qpos-out run.npz ; python scripts/render_dance.py out.mp4 "MW=run.npz" "Teacher=teacher.npz"
```

`ours/pm_eval_student.py` wraps ProtoMotions' evaluator: `--student`, `--act-noise`, `--seed`, `--uq-out` (per-step uncertainty), `--qpos-out` (for videos).
`render_dance.py` needs the real G1 meshes (ProtoMotions' are Git-LFS pointers; set `G1_MESHES`).

## Licences and what is not here

Our code: the repository's licence. **Not included:** NVIDIA's tracker weights, the generated motions, teacher rollouts and our trained student
weights (derived from the NVIDIA tracker's outputs; regenerate them with the scripts). ProtoMotions code is Apache-2.0; the tracker and Kimodo
weights are under the NVIDIA Open Model License; Kimodo's text encoder uses Llama 3 (Meta licence). Simulation only; not a real-robot result.
