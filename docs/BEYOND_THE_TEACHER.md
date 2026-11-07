# Beyond the teacher: how a student can outgrow the policy it learned from

*Demo: [`for_distilation/protomotions_g1/examples/policy_priors_demo.py`](../for_distilation/protomotions_g1/examples/policy_priors_demo.py)
(plain PyTorch, CPU, a few minutes).*

## The baby and the parent

A baby learns to walk by copying its parents, but the parents are not its only teacher. **Gravity teaches too.** Every wobble
the parents never showed is still judged by physics: lean too far and you fall. So a child ends up able to do things nobody
demonstrated, because the laws of physics fill in where the examples run out.

A plain distilled policy has only the parent. It copies the teacher's actions on the situations the teacher happened to visit,
and outside them it guesses. An MW policy also gets **physics priors**: laws written down from the robot's physics, checked on
the *consequences* of each action. They teach the student in exactly the places the teacher never went.

## The demo: balancing a pole (inverted pendulum)

* **Teacher:** a good controller that balances the pole.
* **Data:** the teacher's own trajectories from small leans, the way real distillation data looks: lean and speed always move
  together, so the data cannot tell "react to the lean" apart from "react to the speed". Many functions fit it equally well.
* **Plain student:** fits the teacher's actions.
* **Prior student:** fits the same actions **plus one law**: *after the action, the capture point (lean + speed / natural frequency,
  where the pole is heading) must shrink.* The law is checked on extra states nobody labelled (bigger pushes, faster spins); no
  teacher is needed there, just the physics.

### Result 1: the same data, very different robots

Pushes from upright at speeds the data never contained (5 random seeds, 35 push tests each):

| | Fell |
|---|---|
| Plain student | **18 of 35** pushes (every seed falls from about 3 rad/s up) |
| Prior student | **0 of 35** |
| Teacher | 0 of 7 |

Both students fit the teacher's data equally well (mean squared error 0.0000 to 0.0012). **The data cannot tell them apart; only the prior
decides what happens outside it.**

### Result 2: better than the teacher

Now a *soft* teacher: smooth but slow to recover. Success = the pole stays within 2 degrees after a push. The prior asks for faster
recovery than the teacher ever shows:

| | Pushes where the lean went past 2° (of 35) | Maximum lean per unit of push |
|---|---|---|
| Teacher | 25 | about 10.9° per rad/s |
| Plain student | 25 | same as the teacher |
| **Prior student** | **8** | **about 5.5° per rad/s (half)** |

The student beat its teacher, because the law contains knowledge the teacher's data does not. The limit is honest physics: a
teacher already at the robot's limits (full torque against huge pushes) can be matched but not beaten.

Run it: `python policy_priors_demo.py 0,1,2,3,4` (result 1) and `RHO=0.85 PRIOR_W=3 python policy_priors_demo.py exceed 0,1,2,3,4`
(result 2; the prior asks the capture point to shrink to 85 % each step, weight 3; with the default, gentler prior RHO=0.95 and weight 1
the prior student exceeds 2° on 21 of 35).

## On the humanoid (MW level 3, in progress)

The same laws exist for the Unitree G1, written from its model and checked against MuJoCo
([`ours/priors.py`](../for_distilation/protomotions_g1/ours/priors.py), [`ours/g1_kin.py`](../for_distilation/protomotions_g1/ours/g1_kin.py),
kinematics exact to 3e-9 m): capture point inside the support (balance), upright torso, joint limits, follow the reference.
[`ours/train_student_priors.py`](../for_distilation/protomotions_g1/ours/train_student_priors.py) applies them to the student's own
actions up to 10 steps ahead, including on states the teacher never visited.

What we have learned so far: the laws must be judged with **accurate physics**. In a first strong-prior test on 100 motions the
consequences were predicted by a small learned dynamics model; a strong prior pushing on an inaccurate prediction made the policy
worse (0/7 on the dance vs 3/7 for the identical student without priors). The next version judges the laws with the robot's
exact equations of motion (already working: one-step prediction error about 1e-8). Results will be added here when they pass.
