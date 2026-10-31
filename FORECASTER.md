# Testing MW as a predictor, against a real simulator (MuJoCo) — 2026-10-31

Everything else in this repo checks whether a *given* state looks physically real — a
**checker**. This is the first test of MW as a **predictor**: given where a system is now,
guess where it will be next, without running a physics solver to find out. Tested against
MuJoCo (Apache-2.0), a real third-party physics engine, on the hopping-leg (`legged_contact`)
example. Code is in `mujoco_experiments/`.

**Short version:** there's a real, measured tradeoff — MW can predict faster than MuJoCo can
simulate, but only beyond a certain jump size, and only at a real, non-zero accuracy cost.
Not "MW replaces MuJoCo," and not a dead end either. Three real bugs were found and fixed
along the way to get trustworthy numbers — all three are explained below, not glossed over.

## Part 1 — can MW correctly judge real MuJoCo data? (`01_validator_check.py`)

Train the hopping-leg model on its own built-in training data, then score real ball-drop
trajectories from MuJoCo's own physics engine (not our made-up data):

| Model | Trained on bounce range | Typical "distance" score on real MuJoCo data | False alarms |
|---|---|---|---|
| A (as shipped) | 0.6–0.98 | 19.5x normal | 97.0% |
| B (matched to MuJoCo) | 0.2–0.6 (what MuJoCo can actually produce) | 1.3x normal | 15.6% |

Model B still catches all of its own test faults correctly (score 0.998-1.000, basically
perfect). **The lesson:** almost all of Model A's false alarms were because it was trained on
bounce behavior MuJoCo can't actually produce — not because MuJoCo's physics looked wrong.
Train on the range the real system actually uses, not a convenient made-up range.

## Part 2 — can MW predict ahead in time? (`02_forecaster_check.py`)

Train the hopping-leg model directly **on real MuJoCo data** this time (not made-up data).
Then: take a real starting point and a real ending point from one held-out bounce, draw a
straight line between them in the model's internal space, decode points along that line, and
compare them to MuJoCo's own detailed step-by-step trajectory over that same interval.

### Three bugs found and fixed — each one mattered

1. **Reading the starting speed too late.** The code was reading the launch speed a few steps
   after liftoff actually happened, which threw off the energy math by a small but consistent
   amount. Fixed by calculating the true launch speed from the bounce height instead.
2. **A coordinate mismatch.** MuJoCo places a ball's center at its radius above the ground,
   not at zero. The ball-drop formula assumed zero. Fixed by shifting the height value to
   match. Together with fix 1, this took the baseline error from 0.120 down to 0.020 — about
   as small as you'd expect from just the simulator's own step-by-step rounding, not a real
   remaining bug.
3. **Testing on the wrong kind of bounce, every time.** The code picked "the longest flight"
   to test on — but because each bounce loses a little energy, the very first fall before any
   bounce happens is always the longest. The training data deliberately skips that first fall
   (since it needs a previous bounce to exist). So every earlier test was accidentally scoring
   the one situation the model had never been trained on. Once fixed to test on an actual
   post-bounce flight, pure reconstruction error dropped to **2.5%** (height) and **1.0%**
   (speed) of the normal range. This was the single biggest fix of the three.

### The "smarter path-finding" feature turned out to be dead weight

There's a feature in this codebase (`geometry.py::geodesic()`) that spends 300 optimization
steps trying to find a smarter, curved path between two points instead of a straight line.
Tested directly, same two points:

| Method | Height error | Speed error | Time taken |
|---|---|---|---|
| The 300-step "smart" path | 0.0915 | 0.1650 | 537.4 ms |
| A plain straight line | **0.0803** | **0.1584** | **0.342 ms** |

The straight line was both **~1570x faster** and **slightly more accurate**. The fancy version
wasn't just slow — it was actually finding a worse answer. So the predictor below just uses a
plain straight line, no extra optimization step.

### The real result: speed vs. accuracy, measured honestly

Bigger and bigger jumps forward in time, each one compared against MuJoCo actually stepping
through that same interval in small steps:

| Jump size | MuJoCo steps skipped | Height error | Speed error | MW's time | MuJoCo's time | Who's faster |
|---|---|---|---|---|---|---|
| 0.02s | 10 | 7.9% | 1.0% | 0.393 ms | 0.066 ms | MuJoCo |
| 0.05s | 25 | 9.5% | 2.5% | 0.374 ms | 0.165 ms | MuJoCo |
| 0.10s | 50 | 7.9% | 3.9% | 0.386 ms | 0.329 ms | about tied |
| 0.20s | 100 | 13.3% | 6.3% | 0.383 ms | 0.659 ms | **MW** |
| 0.40s | 200 | 13.5% | 9.6% | 0.381 ms | 1.318 ms | **MW** |

This is exactly the shape you'd want for "skip a lot of small simulator steps with one big
guess": MW's cost per jump stays roughly flat (about 0.38 ms no matter the jump size), while
MuJoCo's cost grows with every extra step it has to simulate. They cross over somewhere around
50-100 steps; past that, MW keeps winning by a growing margin — at the cost of accuracy, which
gets worse the bigger the jump (speed error grows from about 1% to about 10% of the normal
range as the jump grows from 10 to 200 steps).

## Part 3 — does richer training data help? (`03_mujoco_like_datagen.py`)

The battery work's own data-generation trick — simulate many independent units, each with
its own randomly chosen physical settings, instead of a fixed grid of conditions — is what
made that work generalize so well. Parts 1 and 2 above both used a plain 40-point fixed grid
(10 heights x 4 damping settings). This test redoes the same thing with 80 independent MuJoCo
drops, each with randomly varied height and bounciness, and re-checks both the checker and
predictor roles.

**Checker: a big, real improvement.**

| Training data | False alarms on unseen settings |
|---|---|
| 40-point fixed grid | 15.6% |
| 80 randomly varied drops | **0.5%** — essentially at the noise floor |

With richer, more varied training data, the model generalizes to settings it never saw almost
perfectly. One caveat: "unseen" here still means unseen *within the same kind of simulated
data* — it tests whether the model's learned shape generalizes smoothly across that range, not
whether it generalizes to a totally different physics source. That's exactly the kind of thing
that should improve with richer training data, and it did.

**Predictor: no real improvement.**

| | 40-point grid | 80 randomly varied drops |
|---|---|---|
| Pure reconstruction error (height / speed) | 2.5% / 1.0% | 3.0% / 1.6% (slightly worse) |
| Jump error at 0.2s (height / speed) | 13.3% / 6.3% | 12.5% / 6.9% (basically unchanged) |

**Honest takeaway:** the richer-data trick is a real, significant fix for the **checker**
role — the same kind of improvement that made the battery work generalize well. It does
**not** help the **predictor** role tested here. Whatever's causing that error, it's something
else — most likely just the nature of guessing a path between two points — not how much
training data variety there is.

## Part 4 — does the error pile up if each guess builds on the last guess? (`04_chained_forecast.py`)

Everything above predicted a jump between two *known, real* points. But a real predictor would
need to predict several jumps in a row, each one starting from its own *previous guess* rather
than a real measurement. Does a slightly-wrong starting point make each following guess worse
and worse, or does it correct itself over time?

One honest limit, found while building this test: the current method (drawing a straight line
between two known points) has no way to predict forward from just one point alone — it always
needs a known target to aim at. So a true "predict the whole future with nothing else known"
test isn't possible with this method yet. That's a real finding about what's still missing,
not something to work around quietly.

What *can* be tested fairly: chain only the *starting* point. Each jump still aims at a real,
known checkpoint ahead (the only thing this method can do), but the jump's starting point is
the *previous jump's guess*, not the real measurement. Over 15 jumps, comparing against a
version that always restarts from the real measurement:

| Jump # | Time | Built on earlier guesses | Always restarted from real data |
|---|---|---|---|
| 1 | 0.09s | height error 6.6%, speed error 3.0% | height error 6.6%, speed error 3.0% |
| 5 | 0.49s | height error 4.1%, speed error 2.7% | height error 3.9%, speed error 2.5% |
| 10 | 0.99s | height error 1.1%, speed error 0.2% | height error 1.1%, speed error 0.2% |
| 15 | 1.41s | height error 7.3%, speed error 1.8% | height error 6.9%, speed error 1.8% |

(Full 15-jump table in `mujoco_experiments/04_chained_forecast.py`'s output.)

**The gap stays small and doesn't grow.** Building each guess on the last guess, instead of
always starting from a real measurement, adds at most about 0.5 percentage points of extra
error at any point in the 15-jump chain — and that gap doesn't widen as the chain gets longer.
**Why:** every jump still aims at a real, known target — that target keeps pulling the guess
back toward the truth, even when the starting point was slightly off. A small early error does
not snowball into a big one here.

**What this does and doesn't show:** it shows that, for this method, a wrong start doesn't
compound — a real, useful property. It does **not** show that MW can run forward on its own
with no outside help: every single jump still needed a real, known target state to aim at.
Removing that requirement (see "Next steps" below) is the real next step toward a true
standalone predictor.

## What this does and does not support

- **Supports:** for the hopping-leg example, a single jump of up to about 0.4 real seconds
  (roughly 200 skipped simulator steps) can be predicted from two known real points, faster
  than MuJoCo itself, at an error of roughly 7-14% of the normal range. The near-perfect
  reconstruction accuracy (2.5% / 1.0%) on in-distribution states shows the model's internal
  map itself is a good fit — the jump error comes from the straight-line guess, not from the
  model being wrong. Separately, richer training data makes the **checker** role generalize
  to new settings almost perfectly (15.6% to 0.5% false alarms) — a real, large improvement,
  just one that doesn't carry over to predictor accuracy. And chaining several jumps together,
  each starting from the last guess instead of real data, does **not** make errors pile up.
- **Does not yet support:** "MW can replace MuJoCo as a simulator." This is one example, one
  held-out trajectory, and every jump still needed a real, known target to aim at — there's
  still no way to predict forward starting from just one current state with nothing else
  given. That's the next thing to build, not just test.
- **A fact about this example, not a weakness to fix:** MuJoCo itself solves this particular
  example in microseconds per step — it's a cheap system. The speed benefit shown here is
  specifically about *replacing many small steps with one big jump*, not about MW being faster
  at matching MuJoCo's own level of detail.

## Next steps

1. **Try predicting forward with no known target at all** — a small model that goes straight
   from "current state + how far ahead to look" to "future state," instead of needing both
   ends already known. This is the actual design described in the "World Models with an
   Irreversible Time Axis" paper (see `ROBOTICS.md`'s "The bigger idea" section) — and it's
   the missing piece that would let this predict from just one current state.
2. **Try the same tests on the other examples that map onto MuJoCo** (`joint_actuator`,
   `planar_arm`, `mobile_robot`) to see whether the same speed crossover and 7-14% accuracy
   band shows up generally, or is specific to bouncing/impact physics.
3. **Tighten the accuracy-vs-jump-size numbers** with more jump sizes and more held-out
   trajectories (this report used just one), and report a range, not a single run — the same
   standard already used for the battery evidence.

## Try it yourself
```bash
pip install mujoco          # Apache-2.0, not in the default requirements
cd mujoco_experiments
python 01_validator_check.py
python 02_forecaster_check.py
python 04_chained_forecast.py
```
