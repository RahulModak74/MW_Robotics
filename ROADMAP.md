# MW Robotics: roadmap

## Where we are

| | Status |
|---|---|
| **Part 1: policy distillation, powered by Bayesian methods** | **Ready.** NVIDIA ProtoMotions' Unitree G1 tracker distilled into a 7.7 MB MW policy: unseen dance 7/7 (plain MLP 6/7), warning before falls at AUROC 0.996, 2.3 ms per step on one laptop CPU thread. [`for_distilation/protomotions_g1/`](for_distilation/protomotions_g1/README.md) |
| **Part 2: policy generation, together** | Research. Building blocks in [`for_full_replacement/`](for_full_replacement/README.md); looking for partners |

## Part 1: from "ready" to "deployed everywhere"

1. **More teachers.** Apply the same recipe ([`docs/POLICY_TO_PRODUCTION.md`](docs/POLICY_TO_PRODUCTION.md)) to other policies whose
   licences allow derivative models: other trackers, locomotion and manipulation policies, other robots.
2. **Packaging for the edge.** ONNX export of the MW policy (dense layers plus one triangular product for the uncertainty) and
   latency on phone-class and embedded CPUs.
3. **Fallback in the loop.** A standard `predict → (action, uncertainty, fallback flag)` interface, with the handover tested in
   simulation for every release.
4. **Toward hardware.** Randomised physics in the exam (masses, friction, motor gains, delays), then supervised real-robot trials
   with the fallback armed.

## Part 2: policy generation, together

Distillation needs a teacher. Part 2 is about producing **heavily fortified policies** in the first place, and replacing *parts*
of today's policy-generation pipelines with Bayesian, physics-first pieces. The idea, in one sentence:

> **Any robotic action is made of states that live on a physical surface (the manifold of accepted, balanced and recoverable
> states), plus enough look-ahead to make sure the trajectory does not leave that surface.**

The pieces we are building, each one meant to plug into existing pipelines rather than replace them wholesale:

| Piece | What it replaces or strengthens | Status |
|---|---|---|
| **Iron-clad physics priors**: the robot's equations of motion, balance (capture point), upright torso, joint and torque limits, written from the robot's model file | behaviour a network would otherwise have to discover from billions of samples | equations of motion for the G1 working (one-step prediction error about 1e-8); kinematics exact to 3e-9 m |
| **Manifold of accepted states** for a robot: the poses *and velocities* it can stand, sit and move in and still recover | hand-tuned termination rules and reward terms | design stage; G1 first |
| **Look-ahead on the manifold**: before acting, check that the next 0.1–0.5 s stays on it (longer for a jump) | reactive-only control | first tests on the G1 |
| **Uncertainty from day one**: every policy reports how far it is from what it knows | black-box policies | working in Part 1 (AUROC 0.996) |
| **Priors in training** (MW level 3): laws judged on the consequences of each action, on states the data never covered | more data, more compute | works on a pendulum (student beats teacher); on the G1, needs the exact physics above ([`docs/BEYOND_THE_TEACHER.md`](docs/BEYOND_THE_TEACHER.md)) |

Then: skills as sequences of accepted states. For the G1 the target tasks are everyday ones: playing a piano, picking fruit,
watering plants, each one a goal for the hand plus a path that stays on the manifold.

### Compute: pragmatic, not dogmatic

This is heavy work, and the enormous power of GPUs can and should be used for it. Our stance on GPUs is **pragmatic, not
dogmatic**: we use what is easily available, such as the consumer GPUs in the gaming laptops that most teams in the Global South
already own, and we welcome what large clusters can do. **We oppose barriers, not technology:** closed runtimes, licence servers,
and hardware only a few can buy. The rule we keep: whatever is trained, anyone should be able to **run** it on an ordinary CPU.

### How to work with us

* **Labs with policies:** if your licence allows derivative models, we will distill your policy into an MW policy and publish the
  comparison (success under noise, warning AUROC, CPU latency) with you.
* **Teams with GPUs:** help train fortified teachers with physics priors in the loop; we bring the priors, the manifold and the
  uncertainty layer.
* **Teams without GPUs:** reproduce Part 1 on a laptop and report what breaks: that feedback is the point.

## Why this can be cheap

The policy network itself is small; the cost of conventional training is the **number of samples**. On-policy reinforcement
learning throws each batch away after one update and needs very many simulated steps, hence thousands of parallel simulators.
MW attacks the sample count three ways: physics goes into the model as priors instead of being rediscovered; the priors rule out
impossible behaviour before training starts; and uncertainty is an output, so data collection can focus on exactly the
situations the policy does not yet know.
