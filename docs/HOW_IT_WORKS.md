# How a teacher policy becomes an MW manifold policy

*Plain-words version. Code: [`for_distilation/protomotions_g1/ours/student_prob.py`](../for_distilation/protomotions_g1/ours/student_prob.py)
(`MWBundleV2`, `LastLayer`) and [`train_student_prob.py`](../for_distilation/protomotions_g1/ours/train_student_prob.py).*

## 1. A policy is just a function

A robot policy looks at the robot's situation and answers "what do I do now?", 50 times a second. For the Unitree G1 tracker:

* **what it sees (the observation, 349 numbers):** the 29 joint angles and their speeds, how the body is turning, which way gravity
  points from the torso, where the reference motion wants the body to be 1, 2, 4 and 8 steps from now, and the last command sent;
* **what it does (the action, 29 numbers):** a target angle for every joint; the motors' own controllers pull the joints there.

NVIDIA trained that function with reinforcement learning on large GPUs. We do not need to know how; we only need to **watch it work**.

## 2. Watch the teacher

We let the teacher drive the robot in simulation (MuJoCo) on a few hundred motions: the 4 originals plus 324 distinct motions generated
from text prompts ("walk in a circle", "wave with both hands", "do a squat", ...). Every step we record the pair
*(what it saw, what it did)*. Collecting and training take hours on a gaming laptop, not weeks on a cluster.

## 3. Learn the map of normal states (the manifold)

The 349 numbers are not random: a balanced robot's joints, speeds and posture move together. So the MW policy first learns to
**compress every situation into 16 numbers and rebuild it** (an encoder and a decoder, trained as a Bayesian model with Pyro).
The set of situations it can rebuild well is the **manifold of normal states**: a 16-dimensional surface sitting inside the
349-dimensional space.

Two things come out of this for free:

* **a position on the map**: the 16 numbers say *which kind of moment* this is (mid-step, arms up, crouching ...);
* **a distance to the map**: how badly the situation can be rebuilt. A large distance means "this is not a situation I know".

## 4. Read the action from the map

The action head looks at the situation **and** its place on the manifold, and outputs the 29 joint targets. Because the manifold
groups similar moments together, a dance step that was never in the data can still land near moments that were (a squat, a turn,
an arm swing), and the action is read from there. That is why more varied training motions helped so much: 0/9 dance runs with
no extra motions, 7/7 with 324.

## 5. Know how sure you are (the Bayesian part)

After training, the last layer of the action head gets an **exact Bayesian posterior** (a closed-form, one-pass calculation called a
last-layer Laplace approximation). For any new situation it answers: *given everything I was shown, how uncertain is my action
here?* This uncertainty rises when the robot drifts into situations unlike the training data, which is exactly what happens in
the moments before a fall. On unseen dances it predicts a fall within the next 25 steps (0.5 s) with **AUROC 0.996**.

The plain MLP has none of this: it gives an answer with the same confidence whether the situation is familiar or not.

## 6. Run it anywhere

The result is one small network: 1.65 million parameters, 7.7 MB, about 3.3 million multiply-adds per step. It runs in **2.3 ms per
step on one thread of a laptop CPU**, uncertainty included, against a 20 ms budget. That size class fits robot computers and edge
boards, and phone-class CPUs on paper (not yet tested on a phone). Today it runs with plain PyTorch on CPU; ONNX packaging for
phone and embedded runtimes is the next packaging step.

At run time, every step returns three things:

```
action (29 joint targets)   +   uncertainty (per joint)   +   distance to the manifold
                                      │                               │
                                      └─────── above threshold? ──────┘──► hand over to a safe fallback controller
```

## 7. In one line

**The teacher shows the way; the manifold remembers where the robot can be; the Bayesian head knows when it does not know.**

Next level (in progress): add physics priors (balance, upright torso, joint limits) so the policy is shaped by laws, not only by
examples. See [BEYOND_THE_TEACHER.md](BEYOND_THE_TEACHER.md).
