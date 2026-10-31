# MW Robotics

**The Modak-Walawalkar (MW) method, applied to robotics.**

MW is a way to learn the physics of a system from data, instead of solving the physics
equations directly every time you need an answer. It's already proven in one hard domain —
battery analytics, where it runs in production and holds up against public datasets and an
independent researcher's own data.

This repo takes that same core method, unchanged, and tries it on six different kinds of
robot physics: a bending beam, a motor joint, a robot arm, a wheeled robot, a hopping leg,
and motor heating. Same code, six new problems. It's also been tested against a real physics
engine (MuJoCo) to see if it can act as a cheap stand-in for a simulator.

Two papers back the theory: the main one (DOI 10.5281/zenodo.19304813), and a follow-up,
["World Models with an Irreversible Time Axis"](https://doi.org/10.5281/zenodo.22954118)
(DOI 10.5281/zenodo.22954118), about using this idea to predict the future, not just check
the present.

**Two different claims, kept apart on purpose:**
- **The method itself works.** That's the proven part — real battery use in production, plus
  it worked unchanged on six brand-new robot physics problems it had never seen.
- **Each robot example here is a first attempt, not a finished product.** They're new code,
  tested once, doing well on some things and needing more work on others. They're shared so
  other people can build on them instead of starting from zero. See `ROBOTICS.md`.

## Where the speed comes from

Normally, checking whether a robot's state makes physical sense means solving its equations
of motion — a search over every possible combination of numbers the state could take. That
search space is enormous: millions to billions of combinations for a real system.

Almost none of those combinations are actually possible. A robot's joint angle, a battery's
voltage, a beam's curve — each one is tied down by an equation. The real system can only ever
sit in the tiny sliver of that huge space where all the equations hold true at once.

MW learns that sliver directly. The physics equations go in as **priors** — not as decoration,
but as the actual rules the model is trained to obey (see `constraint()` and `bound()` in each
file under `domains/*.py`). Training carves the huge search space down to a small, curved
surface where the physics really holds.

Once that surface is learned, checking a new state is just one quick pass through a small
neural network — not a search, not a solve. That's the entire trick behind every speed number
in this repo.

```
the old way:    equations -> solve them numerically over every possible state -> answer
the MW way:     equations -> used as training rules -> learn the small surface where
                they hold -> one quick pass over that surface                   -> answer
```

## The six robot examples

| # | File | What it is | Physics | Rules it learns |
|---|---|---|---|---|
| 00 | `flexible_robot.py` | A bendy robot arm or soft-robot segment | Beam bending (a PDE) | how it bends, how vibration dies down, how much strain is safe |
| 01 | `joint_actuator.py` | One motor-driven joint | A damped spring-like motion (an ODE) | Newton's law for the joint, how its energy settles, torque limits |
| 02 | `planar_arm.py` | A 2-joint robot arm | Pure geometry, no solving needed | where the hand ends up, how fast it moves, joint limits |
| 03 | `mobile_robot.py` | A wheeled robot | How it drives and turns (an ODE) | it can't slide sideways, how fast it can turn, speed limits |
| 04 | `legged_contact.py` | A hopping robot's foot in the air | Falling + bouncing | energy is conserved in the air, it can't sink through the floor, bounce limits |
| 05 | `actuator_thermal.py` | A motor's heat | Heat spreading (a PDE) | how heat flows, how it cools down, a safe temperature limit |

Each one is built the same way: data generated from the exact real equations (no slow
simulator needed), three physics rules taught as priors, and three realistic faults to test
against. Try any of them:

```bash
pip install -r requirements.txt
python examples/00_flexible_robot_validator.py      # or 01..05 - uses a GPU automatically if you have one
pytest -q
```

## What the six examples show

Full numbers and discussion in [ROBOTICS.md](ROBOTICS.md). The short version:

- **The method works the same way on all six, with no changes to the underlying code** — a
  PDE, two different ODEs, pure geometry, a steering constraint, and a bounce/impact problem.
- **Clearly broken physics is caught very well everywhere** — a motor that lost its damping,
  a cooling failure, overspeed, a bad bounce, energy appearing from nowhere. These all score
  0.97–1.00 out of 1.00 (1.00 is a perfect catch).
- **Subtle, scaled-down faults are harder to catch** — a bit more friction, a slightly bent
  link, a small short in a motor winding. These score 0.55–0.67. This shows up the same way
  in several unrelated examples, so it's a real pattern, not bad luck. [FIXING_THE_GAPS.md](FIXING_THE_GAPS.md)
  explains what to try next.
- **One test (the beam, with a coarse simulator) came out backwards.** Also explained, with
  a specific fix proposed, in [FIXING_THE_GAPS.md](FIXING_THE_GAPS.md).

In short: a working robot simulator isn't proven yet. What is proven is that the same method,
unchanged, works across six very different kinds of robot physics — and every weak spot found
so far comes with a concrete next step, not a shrug.

## MW and MuJoCo: not alternatives — MW runs on top of MuJoCo

A natural question, since MuJoCo (a real physics simulator) is already free and runs fine on a
CPU: why use MW at all? Because they do different jobs, and the MuJoCo work in this repo
(`mujoco_experiments/`, [FORECASTER.md](FORECASTER.md)) has MW running *on top of* MuJoCo, not
instead of it.

- **MuJoCo is the ground truth.** Given a robot model, it simulates real physics step by step
  and tells you exactly what happens next. It needs no training data, and it will happily
  simulate whatever state you give it, realistic or not — it has no opinion on that.
- **MW is a cheap, learned stand-in, trained from MuJoCo's own data.** Once trained, it does
  two things a raw simulator doesn't do for you:
  - **Checks whether a state looks physically real** — a validator, catching things a
    simulator will simulate without complaint (see `01_validator_check.py` and `ROBOTICS.md`).
  - **Guesses a future state without re-running the simulator** — a forecaster, faster than
    MuJoCo once the jump is big enough, at a real, measured accuracy cost (see
    [FORECASTER.md](FORECASTER.md)).

**So yes: "MuJoCo plus MW" is fairly described as a simulator with a built-in, cheap
validator** — the one thing a raw physics engine doesn't give you on its own. That's proven,
concretely, for one domain so far (the hopping-leg / bouncing-ball example, `legged_contact`)
— not yet shown across every kind of MuJoCo model, which is the natural next step rather than
something to claim early.

This is also why MW doesn't solve the actual GPU bottleneck robotics startups run into:
training a sophisticated RL control policy at IsaacGym/ProtoMotions scale needs massively
parallel simulation, and MW isn't an RL algorithm today — it doesn't train control policies.
What it adds right now, cheaply, on a CPU, is the validator and short-horizon-forecaster role
on top of whichever simulator a team is already using, GPU or not. There is one credible,
not-yet-built path for MW to actually attack that GPU bottleneck itself, not just work around
it — see the next section for exactly what that would take and where it could fail.

## A real alternative to GPU-scale RL training — scoped, not built yet

GPUs matter for RL training for one specific reason: training a policy needs millions of real
physics samples, and a GPU lets you run thousands of simulations in parallel to generate them
fast. That's a brute-force answer to "I need a lot of samples." There's a different, well-known
way to need *fewer* samples in the first place: train the policy mostly against a cheap,
learned stand-in for the simulator, and only occasionally check against the real, expensive
one to keep it honest. This is called model-based RL, and it's the same family of idea as
Dreamer's "world models" — and it's exactly what the theory paper behind MW, ["World Models
with an Irreversible Time Axis"](https://doi.org/10.5281/zenodo.22954118), already points
toward. So this isn't a stretch; it's the natural next direction, just not built here yet.

**What's missing today, concretely — three real gaps, not a vague "needs more work":**

1. **No action input.** What's proven in [FORECASTER.md](FORECASTER.md) is: given two
   *already-known* states, guess what's in between. Model-based RL needs the opposite shape:
   given the current state *and the action the policy just chose*, predict what happens next.
   The current method can't take an action as input at all — this needs a genuinely different
   kind of model, not a tweak to the existing one.
2. **No reward signal.** A policy needs to know whether an action helped. MW's manifold has no
   concept of "did this help the robot walk forward" — that has to be added separately. Not
   hard in principle (reward is usually a known function of state), but not built.
3. **Model-exploitation risk, untested in this setting.** The chaining test in
   [FORECASTER.md](FORECASTER.md) (Part 4) showed error doesn't pile up when every jump still
   aims at a real, known target. Training a policy on *imagined, open-loop* rollouts with no
   real target to correct toward is a harder, different test — the classic failure mode in
   model-based RL is a policy that learns to exploit cracks in the learned model instead of
   learning real physics. That has to be checked for directly, not assumed away.

**A minimal, honest way to actually find out if this works:**

1. Start with `joint_actuator`, not a humanoid — it already has an exact closed-form law and a
   real control input (`tau`, the applied torque), so it's the cheapest domain to build an
   action-conditioned model for and the easiest to check for correctness against ground truth.
2. Build a model that maps `(current state, action, dt) → next state`, instead of today's
   `(state, state) → path between them`. Check its accuracy against the real, exact dynamics —
   this domain already has the closed-form answer to check against.
3. Add a simple reward (e.g. "track this reference angle") and train a small policy using
   *only* imagined rollouts through that learned model — no real simulator calls during
   training at all.
4. Test the trained policy against the *real* dynamics (or MuJoCo), not the learned model.
   This is the real test: did training inside the cheap model actually transfer, or did the
   policy just find a way to look good to a model that doesn't match reality?
5. Count real simulator samples used, end to end, and compare against what plain model-free RL
   would need for the same task. That comparison — not a general claim — is the actual
   evidence for or against "MW offers a GPU-free alternative here."

**What would count as a real result, either way:** if a policy trained almost entirely on
MW-imagined rollouts, using far fewer real physics samples than model-free RL needs, still
performs reasonably well against real dynamics, that's genuine evidence of a GPU-free
alternative for at least this class of problem. If it doesn't transfer — the model-exploitation
failure — that's an equally real, useful finding, and the gap it exposes would need closing
before claiming anything broader. Either outcome is worth having; the point of scoping it this
precisely is to make sure the answer, whichever way it goes, is trustworthy.

## Speed: no GPU needed, measured

How long it takes to check whether one state is physical (median time, after warm-up, on an
i5-12450H laptop CPU vs. an RTX 3050 laptop GPU):

| Example | CPU, one check | GPU, one check | CPU, 1000 at once | GPU, 1000 at once |
|---|---|---|---|---|
| flexible_beam | 1.14 ms | 3.43 ms | 107.4 ms (0.107 ms each) | 3.63 ms |
| joint_actuator | 1.30 ms | 2.57 ms | 50.3 ms (0.050 ms each) | 2.43 ms |
| planar_arm | 1.08 ms | 2.07 ms | 87.3 ms (0.087 ms each) | 2.04 ms |
| mobile_robot | 1.08 ms | 3.39 ms | 90.0 ms (0.090 ms each) | 3.47 ms |
| legged_contact | 1.28 ms | 2.08 ms | 70.1 ms (0.070 ms each) | 2.11 ms |
| actuator_thermal | 1.23 ms | 3.56 ms | 58.3 ms (0.058 ms each) | 3.53 ms |

**Checking one reading at a time — the normal way you'd actually use this — the CPU is 2-3x
faster than the GPU.** That's because sending work to the GPU has a fixed overhead that only
pays off when there's a lot of work to send at once. The GPU only wins when you check many
states in one batch (1000 at a time: 15-30x faster) — useful for big sweeps, not for reading
one sensor value as it comes in.

Training (same example, `flexible_robot`, 200 epochs on 4000 rows, nothing changed except
which device it ran on):

| Device | Total time | Per epoch |
|---|---|---|
| GPU (RTX 3050 laptop) | 105.6 s | 527.7 ms |
| CPU (i5-12450H) | 181.7 s | 908.4 ms |

Training on CPU alone takes about 1.7x longer than on a GPU — but it's still just 3 minutes
on a laptop. You don't need a GPU to train this; it's just a bit faster if you have one.

## What's in this repo

```
mw_framework/
  core.py          the core model: training, the physics-rule machinery, scoring
  config.py        lets you define a new domain from a JSON file instead of writing code
  detector.py      the main user-facing class: train it, score things, explain a result
  algebra.py       combines manifolds from different domains together
  geometry.py       the math for measuring distance and paths on the learned manifold
  domains/
    flexible_robot.py    bendy beam
    joint_actuator.py    motor joint
    planar_arm.py        2-joint arm
    mobile_robot.py       wheeled robot
    legged_contact.py    hopping/bounce
    actuator_thermal.py  motor heat
  datasets.py      generates the training data for every domain above
  checkpoint.py    saves/loads a trained model to one file
  cli.py           command-line train / score / demo-data
examples/          one runnable script per domain, 00 through 05
docs/
  LIMITATIONS.md   honest limits of this whole approach - still applies here
tests/             the test suite (one known flaky test, explained below)
mujoco_experiments/  testing MW against a real third-party physics engine, MuJoCo:
  bounce_env.py          shared helper code
  01_validator_check.py  does MW correctly judge real MuJoCo data as physical?
  02_forecaster_check.py can MW predict the future state, instead of just checking one?
ROBOTICS.md        full results for all six domains, as a physics checker
FORECASTER.md      full results for MW as a predictor, tested against MuJoCo
FIXING_THE_GAPS.md  concrete next steps for every weak spot found so far
```

## One known pre-existing issue

`tests/test_mw.py::test_sklearn_sign_convention` sometimes fails at this fixed random seed.
We checked: it fails the same way in the original, unmodified `MW_Framework` repo too, so it's
not something this robotics version broke. Left as-is rather than quietly adjusted to pass.

## Read next

- **[ROBOTICS.md](ROBOTICS.md)** — full results for all six domains, and what to try next.
- **[FORECASTER.md](FORECASTER.md)** — can MW predict the future, not just check the present? Tested against a real simulator (MuJoCo).
- **[FIXING_THE_GAPS.md](FIXING_THE_GAPS.md)** — specific fixes to try for every weak spot found.
- **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)** — where this whole approach falls short, in general.
- Modak & Walawalkar, ["World Models with an Irreversible Time Axis"](https://doi.org/10.5281/zenodo.22954118)
  — the fuller idea behind using MW to predict the future. FORECASTER.md tests a simpler,
  related version of this idea, not the full thing described in the paper.

## Scope

This is research code with six worked examples — not a finished robot simulator, and not a
safety system. See [ROBOTICS.md](ROBOTICS.md) and [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## License

Apache-2.0 (see `LICENSE`). If you use this, please cite `CITATION.cff` and the MW theory paper.
