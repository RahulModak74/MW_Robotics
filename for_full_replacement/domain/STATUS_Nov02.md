# RL, CEM, and the Manifold Policy Project — An Engineer's Introduction

**Audience.** 1–2 year Python developer. No prior exposure to reinforcement
learning, control theory, robotics, or Bayesian methods assumed. If you can
read a PyTorch training loop and know what a neural network is, you're the
target reader.

**Purpose.** Explain what this project is, why it exists, what's been tried,
what worked, what failed, and where you can start contributing. By the end,
you should be able to run the existing pipeline, read the experiment logs,
and know which parts are load-bearing and which are dead ends.

**Honesty note.** This project has real failures (Plan B) and real wins
(Walker2d). Both are documented. If the writeup sounds too polished
somewhere, trust the numbers over the prose.

---

## 1. What this project is

We're building a **CPU-only pipeline for robot control policy learning**,
as an alternative to the GPU-heavy training that dominates the field. The
specific comparison point is NVIDIA's **ProtoMotions** framework, which
trains humanoid tracking policies using:

- **PPO** (a gradient-based RL algorithm) over neural networks with
  millions of parameters
- **IsaacGym** (GPU-accelerated physics simulation) with thousands of
  parallel instances
- **~24 A100 GPUs**, training for ~12 hours (for the smaller runs; the
  full Metatron policy trains on 142K motions over days)

The economic argument: **a 12-A100-hour run costs roughly $50–150 in
cloud compute**, plus the capital cost of the hardware if you own it, plus
the engineering time. For research labs in the US with grant money and
national labs with cluster access, this is fine. For a robotics team in
India, or a startup in the Global South, or a small team anywhere without
a spare cluster, it's a hard wall.

Our claim (still being tested): **for many real robotics problems, a
gradient-free search over a small policy can get surprisingly close to the
GPU-trained result, at 1/1000th the compute cost.** The thesis is
"**math as a substitute for capital**" — use a better search algorithm
and a more compact policy representation instead of throwing compute at
the problem.

Whether that thesis is true for humanoids is an open question. It's true
for one simplified 2D robot (Walker2d). It's *not yet* true for 3D
humanoids. The middle of this document explains why.

---

## 2. RL in 60 seconds, for someone who hasn't done it

Skip this section if you already know RL. Otherwise:

**Reinforcement learning** trains an agent (a robot, a game player, a
trading bot) by letting it try things and rewarding good outcomes.

The agent is a function `π(s) → a`: given a state `s` (what the robot
senses), produce an action `a` (what the robot does). This function is
called the **policy**. The goal is to find a `π` that maximizes total
reward over time.

There are two classical approaches to finding `π`:

### Approach 1: Gradient descent (PPO, SAC, etc.)

Represent `π` as a neural network `π_θ(s)` with parameters `θ`. Compute
the gradient of expected reward with respect to `θ`, and take steps
down that gradient.

**Analogy:** you're coaching an athlete. After every play, you say
"that was 7% better than the last one, do more of that." The athlete
gradually improves by following incremental corrections.

**Strengths:** very sample-efficient when the reward signal is smooth,
scales to huge networks, well-understood.

**Weaknesses:** needs a smooth, differentiable reward signal. Needs
differentiable physics if you're training in simulation (or a
differentiable surrogate). Needs backprop, which means GPUs, which means
money. And most importantly: **the physics of a walking robot is NOT
smooth** — the moment a foot hits the ground, forces jump discontinuously.
Gradient methods cope by averaging over the noise, but that's expensive.

### Approach 2: Gradient-free search (CEM, ES, CMA-ES)

Don't compute gradients. Instead, **maintain a population of candidate
policies, test all of them, keep the winners, and try again around the
winners.**

**Cross-Entropy Method (CEM)** is the specific algorithm we use. Here's
the loop in pseudocode:

```python
mean = zeros(num_params)        # center of the search distribution
std  = ones(num_params) * 0.5   # spread of the search distribution

for generation in range(num_generations):
    # 1. Sample a population from N(mean, std)
    population = [mean + std * randn(num_params) for _ in range(pop_size)]
    
    # 2. Test each candidate (roll it out in the simulator, count reward)
    scores = [evaluate(theta) for theta in population]
    
    # 3. Keep the top 20% (the "elites")
    elites = top_k(population, scores, k=int(0.2 * pop_size))
    
    # 4. Refit the distribution to the elites
    mean = mean_of(elites)
    std  = std_of(elites) + 1e-3   # small floor to avoid collapse

return mean   # the final best candidate
```

That's it. **Sample, test, keep winners, resample around winners, repeat.**
The distribution "sharpens" over generations as it concentrates around
good regions.

**Analogy:** you're running a tournament. You have 60 candidates, run all
60 through a test, keep the best 12, and then create 60 new candidates by
slightly varying the best 12. Over 30 generations, the whole population
drifts toward high-quality candidates, without ever computing a gradient.

**Strengths:**
- No gradient needed — works on arbitrarily non-smooth problems
  (contact, jumps, discrete decisions)
- Trivially parallelizable — every candidate is independent
- Very sample-efficient for low-dimensional problems (small number of
  parameters)

**Weaknesses:**
- **Sample cost grows with parameter count.** In practice, doubling the
  parameters of the policy roughly doubles the population size needed
  for the search to converge. This is the fundamental scaling wall.
- No gradient means no signal about *how* to improve, only *which* is
  better. So convergence is slower per generation.

The whole "CPU-only" bet lives in this trade: **worse per-sample
efficiency, better per-dollar efficiency** because you don't need GPUs.

---

## 3. The specific problem we're solving

We want a **policy for a walking robot**. The robot is:

- Walker2d: a simple 2D stick figure with 6 joints. Can walk forward.
  Cannot fall sideways (2D-constrained). This is the toy problem.
- Humanoid-v5: a 3D humanoid with 17 joints (5 torso + 2×6 legs, no
  hands). Can fall sideways, forward, backward. This is the intermediate.
- Unitree G1: a 29-DoF commercial humanoid. This is the real target that
  ProtoMotions focuses on.

A policy for each is a function that maps "current joint angles, velocities,
and root pose" to "torques to apply to each joint." We need to *learn* the
policy because writing it by hand is hard (that's the entire field of
classical control, and it doesn't scale to 29-DoF walking).

The pipeline in this repo has two parts:

1. **CEM** finds a policy by search.
2. **MW (Modak-Walawalkar) manifold** is a separate object that scores
   whether a given state is physically plausible — i.e., whether the
   policy's outputs correspond to states the robot could actually be in.

The MW manifold is a Bayesian variational autoencoder trained on
simulated robot trajectories. It learns the *shape* of physically
admissible states (a low-dimensional surface embedded in the full state
space). Off that surface = the state is physically impossible (a joint
beyond its limit, an energy violation, a contact constraint broken).

The MW distance — how far a state is from the learned surface — can
serve as:

- A **safety monitor** in the demo (the "MW gauge" in the walker demo)
- Potentially a **reward-shaping term** during training (not yet tested
  at scale)
- Potentially the **policy itself** (Plan B — see below)

---

## 4. The manifold-as-policy idea (Plan B)

This is the philosophically most interesting part of the project and also
the part that failed. Worth understanding both the idea and the failure.

**The idea.** What if the manifold *is* the policy? Instead of:

```
Policy = neural network, trained to produce actions
Manifold = separate validator
```

we could have:

```
Manifold = trained on (state, action, next_state) triples
Policy = implicit: given a state, sample actions from the manifold's
         action-fibre, keep those that lead to good next states,
         execute the best
```

In this framing, "the policy" is entirely defined by:

1. The manifold (a learned representation of what transitions are physically valid)
2. A reward function (a preference ordering over admissible transitions)

No neural network between state and action. The manifold supplies the
constraint; the reward supplies the objective.

**Why this would be beautiful.** The manifold is small. It runs on CPU.
If the policy lives entirely on the manifold, then the CPU pipeline is
not "approximating" a GPU policy — it's a fundamentally different, better
object.

**Why it failed (Plan B on Walker2d).** We trained a manifold over
`(s, a, s')` triples from the Walker2d simulator. Then we ran the
"E-gate" test: given `(s, s')`, can the manifold recover the action `a`
that produced that transition?

The result: **mean error 2.35 vs random baseline 1.94** — the manifold
found actions *worse than random*. Mean error > random means it wasn't
just noisy; it was actively wrong. Diagnosis: the joint manifold had no
incentive to use the action `a` at all, because `s'` is strongly
correlated with `s` alone (deltas are small per step). The maximum-
likelihood solution was to memorize `s' ≈ s + noise` and ignore `a`
completely. The action-fibre was a phantom.

**The proposed fix (not yet tested).** Conditional VAE: instead of
`p(s, a, s')`, learn `p(s' | s, a)`. Force the model to use `a` to
predict `s'`. This is the correct causal structure. Dreamer, TD-MPC2, and
PETS all use it for exactly this reason. It's a ~2 day rework.

**The honest status.** Plan B failed as originally implemented. The
conditional fix is written but not yet run. If it passes, the
"manifold-as-policy" vision survives as a scaled-down but real thing. If
it fails even in conditional form, the vision is dead and Plan A (CEM
over a small explicit policy) is the only CPU path.

**Important:** even if Plan B passes, "policy as trained manifold" is a
weaker claim than "policy as neural network." The manifold gives you
*constraints*, not *optimality*. The optimal action for a task might live
just off the manifold. If so, sampling on the manifold is structurally
suboptimal and no amount of search will fix it. The conditional VAE test
is designed to catch exactly this.

---

## 5. The experiments — chronological, honest

The project has run through a series of experiments across several months.
Not all have been documented in one place; this is the consolidated
history. Each entry is: what it tested, what it found, what it means.

### Experiment 1 — "Can CEM learn Walker2d at all?"

**Setup:** 96-parameter linear policy on Walker2d (6 joints, 2D). CEM
with population 60, 15 generations. Fitness = survival steps.

**Result:** 208 steps held-out. Original run was ~84 seconds on CPU.

**Meaning:** Yes, CEM can learn a Walker2d policy on CPU. Fast. This is
the founding result — the proof that a gradient-free search over a tiny
policy is a viable approach for this class of problem.

### Experiment 2 — "How good can it get with more budget?"

**Setup:** Same, but population 80, 30 generations.

**Result:** 920.9 steps held-out. Wall-clock ~19 minutes on a laptop CPU.

**Meaning:** The 208-step result was not the ceiling — it was mid-search.
With 10× the budget, the policy goes from "barely falls less than random"
to "essentially never falls." **The dimension that mattered was search
budget, not a hard limit.** Also note: 920.9 with survival fitness means
"stand still for 1000 steps" — the policy learned to not move. That's the
correct optimum for that fitness. It's not a walking gait.

### Experiment 3 — "Can we make it walk, not stand?"

**Setup:** Same policy, but fitness = `forward_distance + 0.02 × survival`.
Same 96 parameters.

**Result:** 1.33 m forward in 600 steps. The stick figure visibly walks
in the demo.

**Meaning:** The fitness function determines what the policy learns. This
is a fundamental lesson that will recur. **A good algorithm with the wrong
fitness learns the wrong thing.** We now have three fitness regimes:
survival (stands), forward (walks), tracking (follows a reference — not
yet implemented).

### Experiment 4 — "Does rank-4 work on Walker2d?" (the load-bearing test)

**Setup:** Replace the 96-parameter linear policy (`a = tanh(W φ(s) + b)`)
with a 90-parameter rank-4 head (`a = tanh(U (V φ(s)) + b)` with rank
`V, U` small). Same budget, same fitness.

**Result:** 896.2 steps held-out (97.3% match with linear's 920.9).

**Meaning:** Rank reduction is a viable way to keep the search dimension
small at higher DoF. The `rank-4` head is the mechanism that should let
us scale from 6 to 17 to 29 joints without blowing past CEM's population
requirements. **This test passed cleanly.** Before this test, "rank-4
scales" was a hypothesis; after it, it's a confirmed mechanism (at least
in 2D).

### Experiment 5 — "Does it scale to Humanoid-v5?" (the wall)

**Setup:** Try three variants on Humanoid-v5 (17 joints, true 3D):
- Flat linear policy: 731 parameters, same budget as Walker2d's 920.9 run
- Rank-4 policy: 253 parameters, same budget
- Rank-4 with arms frozen: 223 parameters, 11 active joints

**Result:**
- Flat linear: 97.2 steps (barely above random 26.1)
- Rank-4 (30 gens): 99.0 steps
- Rank-4 (90 gens): 112.7 steps
- Rank-4 arms-frozen: 102.0 steps

All clustered in the 97–113 range. **None came close to Walker2d's 900+.**

**Meaning (as originally concluded):** The wall at 3D humanoid is not
parameter count — even rank-4 (which matched linear on Walker2d) does no
better. The wall is something else. The original `PROGRESS_LOG.md` entry
concluded: "3D balance itself is the wall."

**Meaning (as of EXP_02, still being tested):** The original conclusion
is *premature*. Three of the five methods in that comparison have concrete
bugs (see EXP_02 below). The 97–113 band might be a broken pipeline, not
a fundamental limit.

### Experiment 6 — "Does classical PD work?" (the disambiguation)

**Setup:** Hand-written PD controller, no learning. Proportional to
orientation error, derivative to rate. Meant as "is CEM doing anything
useful, or is a classical controller fine?"

**Result:** 58.1 steps.

**Meaning (as originally concluded):** PD also plateaus near the CEM
numbers, so the wall is physics, not the search algorithm.

**Meaning (as of EXP_02, still being tested):** The PD controller has
*three of its four* balance corrections applied to the wrong joints. The
index mapping was wrong (`target[2]` was believed to be "right hip y" but
is actually "abdomen x" — a completely different joint). 58.1 steps is
the result of a scrambled controller, not a well-tuned one. A correctly
wired PD should get 300–500 based on the control literature.

### Experiment 7 — "Plan A confirmed, V10 comparison" (in progress)

**Setup:** Run the CEM at higher budget and compare against Plan B's
eventual output. Track both as "the CPU baseline to beat."

**Status:** Plan A gave 920.9 steps (stand-still policy, correct for
survival fitness). V10 comparison against Plan B requires Plan B to
actually work (see Experiment 4 in section 4 above), which is pending the
conditional VAE fix.

### Experiment 8 — Plan B (the manifold-as-policy experiment)

**Setup:** Train a joint VAE over `(s, a, s')`. Test whether the action
can be recovered from `(s, s')` on the manifold.

**Result:** Mean error 2.35 vs random 1.94 — worse than random.

**Meaning:** Failed. The joint formulation ignores the action. A
conditional rework (`p(s'|s,a)`) is written but not yet tested.

### Experiment 9 — EXP_02 (three bug-fixes, currently running)

**Setup:** Three targeted experiments to check whether the "3D is the
wall" conclusion survives obvious bug-fixes:

- **E1:** Fixed-PD controller with correct joint indices
- **E2:** Rank-4 humanoid with root linear velocities added to features
- **E3:** Rank-4 humanoid with reward-based fitness instead of
  survival-only

**Status:** Pending (the spec is written; the runs are queued).

**Possible outcomes:**
- **All three < 130 steps:** the wall is confirmed, with valid methods.
- **One or more ≥ 250 steps:** the wall was bugs, not physics. The
  pipeline was testing a broken configuration.

The decision tree in `EXP_02_Nov2.md` maps each outcome to next steps.

### Summary table — what works, what doesn't, what's unknown

| Component | Status | Notes |
|---|---|---|
| CEM on Walker2d (survival) | ✅ 920.9 steps | The founding result |
| CEM on Walker2d (walking) | ✅ 1.33 m forward | Working demo |
| Rank-4 on Walker2d | ✅ 896.2 steps | Scaling mechanism confirmed |
| CEM on Humanoid-v5 | ⚠️ 97–113 steps | Under investigation (EXP_02) |
| Plan B (manifold-as-policy) | ❌ Failed | Joint VAE ignores action |
| Conditional VAE (Plan B fix) | ❓ Written, untested | The load-bearing unknown |
| G1 (29 DoF) | ❌ Not attempted | Blocked on humanoid resolution |
| ONNX export | ❌ Not built | Roadmap |
| GMR retargeting | ❌ Not built | Roadmap |

---

## 6. Why this matters — the economics

The naive reading of "24 A100s vs 1 CPU" is: "we're 100,000× slower, how
can we compete?" That's the wrong frame.

The right frame is: **the GPU pipeline pays for its speed with capital
and infrastructure.** Every training run requires:

- GPU hardware ($30K–$200K per card, $200K–$1M+ for a cluster)
- Cloud rental if you don't own it ($2–$4 per GPU-hour)
- The engineering to parallelize across GPUs
- The wall-clock for even one experiment (hours to days)

The CPU pipeline trades all of that for:

- A commodity laptop ($1K) or a used workstation ($3K)
- Cloud rental at ~$0.05 per CPU-hour, or $0 for owned hardware
- No parallelization infrastructure needed (CEM is trivially parallel)
- 20-minute turnaround on experiments

**Per-experiment, CPU is 10–100× cheaper.** The question is whether the
CPU pipeline can *solve the same problems*, or only simpler ones.

For Walker2d: **yes, it solves the problem at competitive quality.**

For Humanoid-v5: **unclear.** The current experiments are all
underperforming, but as EXP_02 will test, the underperformance might be
pipeline bugs, not fundamental limits.

For the G1 (the ProtoMotions target): **unknown.** Even if humanoid works
with fixes, G1 at 29 DoF introduces additional scaling that hasn't been
tested. The 12-DoF intermediate is the next real test.

**The honest claim to make today:**

> We have a working CPU pipeline that solves a 6-DoF locomotion task in
> 20 minutes on a laptop, with an MW manifold that runs alongside as a
> physics validator. Whether it scales to 17 or 29 DoF is an open
> empirical question that we're testing with three targeted experiments.

That claim is weaker than "we replace ProtoMotions." But it's defensible,
and it's true, and it's the honest position.

---

## 7. Current state — what to run and what to read

### Files you'll see

```
domain/
├── run_plan_a.sh              # CEM walker2d survival training (920.9 steps)
├── run_demo_walker.sh         # CEM walker2d walking training (1.33 m)
├── run_walker_demo.sh         # Flask server for the demo
├── run_plan_b.sh              # Plan B (manifold-as-policy), FAILED
├── run_conditional.sh         # Plan B fix (not yet run)
├── run_exp_02.sh              # EXP_02 (three humanoid fixes, pending)
├── rank4_walker.py            # rank-4 walker2d (896.2 steps)
├── rank4_humanoid.py          # rank-4 humanoid (99-113 steps)
├── pd_humanoid.py             # buggy PD controller (58.1 steps)
├── cem_humanoid.py            # flat linear humanoid (97.2 steps)
├── walker_demo/               # Flask demo (physics + MW gauge)
└── logs/                      # Every experiment's log

mw_to_protomotions/
├── cem_policy_search.py       # The CEM core, imported by everything
├── walker_domain.py           # The MW manifold for Walker2d states
├── datagen.py                 # PD controller trajectories for training
└── PROGRESS_LOG.md            # Chronological experiments log

mw_framework/
├── core.py                    # The MW VAE class
├── detector.py                # sklearn-style detector
├── geometry.py                # Riemannian geometry utilities
└── domains/                   # Per-domain physics priors
```

### First command to run

```bash
cd ~/MW_Robotics/domain

# 1. Sanity check the environment
./run_plan_a.sh --quick        # ~2 min, smoke test

# 2. Real result (this is the proven baseline)
./run_plan_a.sh --generations 30 --pop 80   # ~19 min, produces 920.9-step policy

# 3. Walking demo (~25 min, produces the walk policy)
./run_demo_walker.sh

# 4. Launch the interactive demo
./run_walker_demo.sh
# → open http://127.0.0.1:5051
```

### First code to read

In order:

1. **`mw_to_protomotions/cem_policy_search.py`** — the core algorithm.
   ~200 lines. Read the CEM loop and the fitness function carefully. This
   is the heart of the CPU pipeline.

2. **`mw_to_protomotions/walker_domain.py`** — the MW manifold. ~60 lines.
   See how physics priors (joint limits, torque limits, reachability) are
   encoded as constraints on the decoder.

3. **`domain/walker_demo/walker_animation.py`** — how the demo connects
   the policy to the manifold. ~100 lines. This is where the "policy +
   validator" architecture is visible.

4. **`PROGRESS_LOG.md`** — the honest experiment history. Then
   **`EXP_02_Nov2.md`** — the current investigation of the humanoid wall.

---

## 8. What to work on next

If you're joining this project, the priority order is:

### Priority 1 — Run EXP_02 and interpret the results

The `EXP_02_Nov2.md` spec has three experiments (E1, E2, E3), each
targeting a specific bug in the humanoid pipeline. Total wall-clock: ~1
hour. The results determine the entire roadmap.

Run `./run_exp_02.sh`. Read the summary. Then:

- If all three come back < 130 steps: the humanoid wall is real, and the
  next sprint is the 2D-constrained intermediate body.
- If any cross 250 steps: the wall was bugs, and the pipeline needs the
  fixes applied permanently.

### Priority 2 — The conditional VAE for Plan B

If Plan B is going to survive, the conditional rework needs to run its
E-gate test. It's ~4 hours of implementation + ~10 min training + ~1 min
E-gate. If it passes, "policy as trained manifold" is back on the table.
If it fails, Plan B is definitively dead.

### Priority 3 — CPU simulation parallelization (`mjbatch`)

Currently CEM evaluates candidates serially. On a 16-core machine, we
waste 15× the available compute. `mjbatch` releases the GIL and runs
thousands of MuJoCo instances in parallel. This is a ~1-day change that
gives 10–15× wall-clock speedup. It doesn't change any result, it just
makes every experiment faster.

### Priority 4 — GMR retargeting

Required for the ProtoMotions comparison. GMR (General Motion
Retargeting) is open source and CPU-native. Integration is ~2–3 days.
Without it, we can only test on toy environments (Walker2d, Humanoid-v5)
that don't match real robot tasks.

### Priority 5 — ONNX export

Required for deployment. A 96-parameter policy exports in a few lines.
A rank-4 head or a waypoint policy also exports trivially. This closes
the "can run on the robot" loop. ~1–2 days.

### Priority 6 — The 12-DoF intermediate

If EXP_02 confirms the 3D wall (all < 130), the next sprint is a
custom 12-DoF humanoid that is 2D-constrained. This isolates "is the wall
3D physics or joint count?" It's ~1–2 weeks of work (URDF design, GMR
retargeting, CEM training, validation).

---

## 9. What to be skeptical of

A few things that look plausible but aren't:

### "The GPU pipeline is 100,000× faster per sample"

True but irrelevant. The question is per-solution, not per-sample. A CPU
pipeline that needs 100× more samples but costs 1000× less per sample is
still cheaper.

### "The MW manifold makes the policy safer"

Unproven at scale. The manifold catches obvious violations (joints beyond
limits, unphysical velocities) but has not been shown to catch the
subtle failure modes that matter in practice (pre-falls, near-misses).
Treat it as a monitor, not a guarantee.

### "Plan B failed so the whole CPU thesis is dead"

No. Plan A (CEM over a small policy) is the primary path and it's working
(Walker2d, 920.9 steps). Plan B is an ambitious side-experiment. Its
failure is informative but not fatal.

### "The humanoid wall is definitely real"

Only if EXP_02 confirms it. The current evidence is contaminated by three
concrete bugs. Wait for the results before drawing conclusions.

### "We can replace ProtoMotions"

We can't, at least not yet. ProtoMotions has:
- Training on 142K reference motions (we have zero)
- A deployed ONNX pipeline at production quality (we don't have ONNX)
- Motion retargeting from human motion capture (we don't have GMR)
- A full browser demo with six robots (we have one stick figure)

What we might have: **a CPU alternative to the *learning component* of
ProtoMotions, for a subset of robot morphologies and tasks, at much lower
cost.** That's a real claim, but it's narrower.

---

## 10. The overall thesis, in one page

**What we're trying to do.** Build a CPU-only policy-learning pipeline
for robotics that can compete with GPU-heavy gradient-based approaches on
a meaningful subset of tasks.

**The core technique.** CEM (gradient-free search over a small policy) +
MW (a physics manifold as validator / potential policy representation).

**What works.** Walker2d, both policy parameterizations (linear: 920.9
steps; rank-4: 896.2 steps). Full interactive demo. MW validator running
alongside. This is real and demonstrated.

**What's uncertain.** Humanoid-v5 (97–113 steps, three bugs identified,
EXP_02 will test fixes). Plan B (manifold-as-policy, failed as joint VAE,
conditional rework pending). G1 (29 DoF, not attempted, blocked on
humanoid resolution).

**Why it matters economically.** Cloud GPU rental for ProtoMotions-scale
training is $50–150 per run. CPU is $0.05–5. Per-experiment cost is
100–1000× cheaper. If the CPU pipeline can solve even a subset of the
same problems, it opens the field to teams who can't afford GPUs.

**Why it matters scientifically.** Gradient-based methods fail on
non-smooth problems (contact, impacts). CEM is designed for these.
Demonstrating that CEM can compete on contact-rich robotics — even on
just a subset of morphologies — is a contribution to the field
independent of the economic argument.

**What "success" looks like.** A defensible public claim: "on 12-DoF
humanoid walking, CEM over a rank-4 policy on CPU achieves X% of
ProtoMotions-quality tracking, in under 60 minutes wall-clock, at 1/500th
the cost." If we get there in the next 6 months, the thesis is validated.

**What "failure" looks like.** If humanoid and 12-DoF both plateau (like
the current 97–113 band) even with all bugs fixed, the wall is real and
CEM is a tool for simple bodies, not humanoids. That's an honest,
publishable finding too — it tells the field *exactly which problems CEM
is appropriate for*.

**The 2-year view.** If this succeeds, it's a paper ("Gradient-Free
Policy Learning for Humanoid Locomotion at 1/500th the Cost") and a small
open-source toolkit. If it fails, it's a paper explaining exactly why,
which is also valuable. Either way, the work is worth doing because
nobody has run this specific experiment at this specific scale with this
specific architecture, and the result — positive or negative — is
informative.

---

## 11. Epilogue — for the junior engineer

If you're reading this and feeling overwhelmed: don't. Everything here
was built by one person over a few months of focused work. The math is
approachable. The code is Python. The experiments are small.

You don't need to understand Riemannian geometry to contribute. You don't
need to have written a VAE before. You don't need to know what CEM's
convergence proof looks like.

What you need to do:

1. Run the commands in section 7. Understand what the output means.
2. Read `PROGRESS_LOG.md` and `EXP_02_Nov2.md`. Understand what we've
   tried, what failed, and what's pending.
3. Pick up one of the priority-1 tasks in section 8. Run it. Interpret
   it. Log it.
4. If you find something wrong (a bug, a mistaken conclusion, a bad
   assumption), say so honestly. This project has survived on honesty —
   the Plan B failure is documented, not hidden. That's the culture.

The specific claim to be skeptical of: **any claim that sounds too
polished**. The GPU pipeline really does work. Our CPU pipeline really is
behind on humanoids. We're not yet at ProtoMotions parity. What we're
doing is building a different, cheaper, sometimes-works pipeline, and
testing how far it goes. If you keep that in mind, you'll contribute
well.

Welcome aboard.
