# Evidence behind the G1 numbers: per run, with definitions and uncertainty

Everything here is MuJoCo simulation, scored by ProtoMotions' own evaluator. Raw summaries: [`SUMMARY_m500.md`](SUMMARY_m500.md) (the
324-motion students), [`SUMMARY_s100_priors.md`](SUMMARY_s100_priors.md) (100 motions), [`SUMMARY_quick100.md`](SUMMARY_quick100.md),
[`SUMMARY_dance_s0.md`](SUMMARY_dance_s0.md).

## Definitions

* **Failure** = ProtoMotions' own rule for this model: the anchor (pelvis) height differs from the reference by more than 0.25 m
  (`anchor_height_error`, in the model's `experiment_config.py`). In practice, a fall.
* **Success** of a run = no failure until the end of the motion.
* **Dance clip runs:** clean (noise 0) once, because a clean run is deterministic; action noise 0.1 and 0.2 with seeds 0, 1, 2.
  Action noise = Gaussian noise of that standard deviation added to every action before the robot executes it.
* **Warning signal** = the policy's epistemic uncertainty (last-layer Laplace), mean over joints of σ / action standard deviation.
* **AUROC (per step):** positive = a step within the 25 steps (0.5 s) before a failure; negative = every step of episodes that did not
  fail, plus steps more than 50 steps before a failure. Threshold-free. Script: `scripts/aggregate_retry.py`.

## Dance clip, run by run (324-motion students)

| Run | Plain MLP | MW level 1 | MW level 2 |
|---|---|---|---|
| noise 0 | pass | pass | pass |
| noise 0.1, seeds 0 / 1 / 2 | pass / pass / pass | pass / pass / pass | pass / pass / pass |
| noise 0.2, seed 0 | **fail** at step 173 | **fail** at step 194 | pass |
| noise 0.2, seed 1 | pass | pass | pass |
| noise 0.2, seed 2 | pass | **fail** at step 221 | pass |
| **Total (95 % interval)** | 6/7 (0.49 to 0.97) | 5/7 (0.36 to 0.92) | **7/7 (0.65 to 1.00)** |

The clip is 248 steps (about 5 s). **With 7 runs these intervals overlap: the dance result shows that MW level 2 matches the teacher
(7/7) on an unseen dance; it does not by itself show that MW beats the plain MLP.**

## Unseen motion sets (one run each, many motions)

| | Plain MLP | MW level 1 | MW level 2 | Teacher |
|---|---|---|---|---|
| 21 unseen familiar motions | 18/21 (0.65 to 0.95) | 19/21 | 20/21 (0.77 to 0.99) | 21/21 |
| 30 unseen generated dances | 26/30 (0.70 to 0.95) | 27/30 | 24/30 (0.63 to 0.90) | 30/30 |

## The warning: how many failures it rests on

AUROC with a 95 % interval from resampling **episodes** with replacement (1000 resamples):

| Student | Set | Failed episodes | AUROC | 95 % interval |
|---|---|---|---|---|
| MW level 2 (324 motions) | unseen generated dances | 6 of 30 | 0.996 | 0.992 to 0.999 |
| MW level 2 (324 motions) | unseen familiar motions | 1 of 21 | 0.997 | 0.990 to 1.000 |
| MW level 1 (324 motions) | both sets | 5 of 51 | 0.992 | 0.978 to 1.000 |
| MW, 100 motions (another training run) | both sets | 14 of 51 | 0.994 | 0.989 to 0.998 |

* The difference between level 1 (0.990 to 0.992) and level 2 (0.996) is **within the intervals: no real difference in warning
  quality.** Both Bayesian students warn well; what level 2 adds is task performance (the manifold), not a better warning.
* A per-step AUROC is flattered by the many steps far from any failure, so here is an **operating point** instead. Threshold = the
  signal's 99.9th percentile over steps of episodes that did not fail; alarm = above threshold for 3 consecutive steps:

| Student | Failures caught | Warning lead time | False alarms during successful running |
|---|---|---|---|
| MW level 2 (324 motions) | **7 of 7** | 0.20 to 0.76 s | 2 in 5.8 min |
| MW level 1 (324 motions) | 5 of 5 | 0.26 to 0.52 s | 2 in 6.0 min |
| MW, 100 motions | 14 of 14 | 0.16 to 0.96 s | 3 in 4.8 min |

  Caveat: this threshold was chosen on the same exam's successful episodes (in-sample). For deployment set it on training motions
  only ([`../../../docs/POLICY_TO_PRODUCTION.md`](../../../docs/POLICY_TO_PRODUCTION.md), step 6); that check is not yet in these tables.

## Verify it yourself

* **In minutes, on any CPU:** `python examples/policy_priors_demo.py 0,1,2,3,4` (pendulum; prints "plain 18 | prior 0") and
  `pytest -q` from the repository root (currently 17 of 20 pass; the 3 failing tests are small stochastic checks in the older
  detector and transition-gate code, not the code on this page).
* **The G1 result:** needs ProtoMotions, MuJoCo, NVIDIA's `g1-bones-deploy` checkpoint and Kimodo (to generate motions). On our gaming
  laptop (i5-12450H, 16 GB, consumer GPU): teacher rollouts about 45 min, training minutes per student, evaluation about 30 min,
  plus motion generation with Kimodo on the GPU. Not an afternoon's job, but a laptop's. We do not ship weights (they derive from NVIDIA's model); the scripts regenerate them.
* **The video** ([`video/mw_vs_teacher_dance.mp4`](video/mw_vs_teacher_dance.mp4)) is versioned in git (commit `8a157a1`), SHA-256
  `cda303323f7f1538a165a69c0aabd000a41026f9364b207b00d069a0cc3407ac`; rendered with `scripts/render_dance.py`.

## Fly-by-wire (the 100-motion student + a physics flight computer)

Same exam, same seeds. The student alone: 3/7 on the dance clip (0.16 to 0.75), 17/21, 20/30. With fly-by-wire envelope D: **7/7
(0.65 to 1.00), 21/21, 30/30**, 0 regressions, and a median tracking cost of +3.8 %. Per run, the design of the envelope and the
limits: [`../../../fly_by_wire/README.md`](../../../fly_by_wire/README.md).
