# Step 2: CPU-compatible policy design (plan, not yet a result)

Step 1 needs a teacher. Step 2 designs policies **without one**, for a specific robot, with Bayesian priors about that robot's
hardware: trained on a consumer GPU for speed, then measured on a CPU for timing.

## The idea

| piece | what it means (Unitree G1 first) |
|---|---|
| **Hardware prior** | a distribution over the robot's real parameters (link masses, motor strength, friction, latency, gear play), centred on the maker's model and updated from data. Not blind randomisation. |
| **Robust manifold** | the balanced, recoverable states that stay recoverable across that whole distribution |
| **Policy** | small and Bayesian from the start: maps a state on that manifold to an action, with its own uncertainty |
| **Flight computer in training** | the Step 1 fly-by-wire shields the policy while it learns (fewer falls, fewer samples); later the policy absorbs its corrections |
| **Skills** | sequences of manifold states: stand → recover → transitions → skills |

## Plan

1. **Gap scan (CPU, no training):** NVIDIA's tracker on existing motions with a payload in the hands, weakened motors, a friction
   change and latency: find where it fails. Those cases become the exam.
2. **G1 stand and push recovery across hardware variants**, MuJoCo on GPU (MJX) on a 6 GB laptop GPU, against PPO with domain
   randomisation at an equal budget; then CPU timing.
3. **Floor work:** kneel, sit, get up; floor cleaning.
4. **New bodies:** a purpose-built cleaner with four legs and mop arms (static stability margin as the main envelope law), then a
   wheeled disc. Same recipe: the body's model file, its envelope laws, its observation builder.

## What exists here

* [`full_cpu_trainer_mw_bayesian.py`](full_cpu_trainer_mw_bayesian.py): a teacher-free CPU trainer for G1 tracking. Conjugate
  Bayesian dynamics belief whose prior mean is the exact discretised PD and gravity physics, closed-form look-ahead control, and
  rounds of acting and adding sufficient statistics. **Status:** the physics prior is exact; the short look-ahead controller still
  falls, which is why the flight computer (0.4 s, sampled corrections) came next.
* Physics priors as a training loss: [`../step1_extraction/student/ours/train_student_priors.py`](../step1_extraction/student/ours/train_student_priors.py)
  with [`priors.py`](../step1_extraction/student/ours/priors.py). **Negative result so far:** with the priors on, the 100-motion
  student scored 0/7 on the dance clip (the same student without them: 3/7; with the flight computer at run time: 7/7). Details:
  [`../docs/BEYOND_THE_TEACHER.md`](../docs/BEYOND_THE_TEACHER.md). Lesson: on the G1, the priors work at run time, as a flight
  computer, not as a loss.
