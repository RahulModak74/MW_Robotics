# Taking a new policy to production with MW (Part 1)

This is the path we followed for NVIDIA ProtoMotions' Unitree G1 tracker, written so you can follow it for another teacher policy
or another robot. Every step names the script that does it for the G1
([`for_distilation/protomotions_g1/`](../for_distilation/protomotions_g1/)).

```
 0 licence  →  1 teacher in simulation  →  2 motions  →  3 record the teacher  →  4 train MLP + MW  →  5 exam  →  6 warning threshold
            →  7 fallback  →  8 package for CPU / edge  →  9 the real robot
```

## 0. Check the licence first

Distillation produces a model **derived** from the teacher. Use only teachers whose licence allows derivative models (and
commercial use, if you need it). Also check the licences of the motion data, of any model used to generate motions (for
example a text encoder) and the attribution you must ship. For the G1: the tracker and Kimodo are under the NVIDIA Open Model
License (derivative models allowed, commercially usable, attribution notice on redistribution); Kimodo's text encoder is Llama 3
(Meta licence); ProtoMotions' code is Apache-2.0. Keep a `NOTICE` file with what you ship.

## 1. Put the teacher in a simulator you can script

You need to run the teacher, read exactly what it sees, and score any policy with the **teacher's own evaluator** so the exam is
not ours to bend. For the G1 we run ProtoMotions with its MuJoCo backend on CPU and wrap its evaluator:
[`ours/pm_eval_student.py`](../for_distilation/protomotions_g1/ours/pm_eval_student.py) (`--student`, `--act-noise`, `--seed`,
`--uq-out` for the per-step uncertainty, `--gate` for handing over, `--qpos-out` for videos). Write down the observation layout
and the action mapping (here: joint target = offset + scale × action) and check them against the simulator.

## 2. Choose the training motions, and lock away the exam

* **Variety beats volume.** The student must see the space of normal body states, not one skill. We used 500 motions generated from
  text prompts, **spread evenly** over four kinds: locomotion, gestures, exercise, everyday actions.
  ([`kimodo_data/prompts.json`](../for_distilation/protomotions_g1/kimodo_data/prompts.json), `gen_motions.py`)
* **The exam must be unseen.** No dance was in training; the exam is a dance clip, plus 30 unseen generated dances and 21 unseen
  familiar motions.
* Measured on the G1: no extra motions 0/9 dance runs, 100 random 1/9, 100 evenly chosen 3/7, **500: 7/7**.

## 3. Record the teacher

Run the teacher on the training motions and save every step's *(observation, action)*. The executed action gets some **noise**
(0 to 0.2, cycling per episode) while the recorded label stays the teacher's clean action, so the data also shows how the teacher
recovers from small mistakes. Collection runs on CPU cores, one process per motion shard.
([`ours/pm_collect.py`](../for_distilation/protomotions_g1/ours/pm_collect.py), `scripts/run_collect.sh`, stage `collect` of `kimodo_data/m500.sh`)

## 4. Train a plain MLP and the MW policy, on the same data

Always train the plain MLP alongside: it is your baseline, and it tells you whether the data is good enough.

```bash
cd ours
python3 train_student.py      --kind mlp  --name r0_mlp_m500  --train_motions $MOT --seed 0            # plain MLP baseline
python3 train_student_prob.py --kind mwv2 --name r5_mwll_m500 --train_motions $MOT --seed 0 --lastlayer  # MW level 2
```

A consumer GPU trains each student in minutes; CPU works too. (Stage `train` of `m500.sh` runs both with the settings we used.)

## 5. The exam: does it do the job, under stress?

Run every policy (teacher, MLP, MW) on the unseen motions, clean and with action noise (0.1 and 0.2; 3 seeds each), with the
teacher's evaluator. (Stage `eval`; `scripts/run_eval_prob.sh`; summary tables by `scripts/aggregate_retry.py`.)

**Pass bar we use:** the MW policy completes the unseen exam at least as often as the plain MLP, including under noise; for the G1:
7/7 on the dance clip (MLP 6/7).

## 6. The warning: calibrate it, then prove it

The MW policy reports per step its **uncertainty** and its **distance to the manifold**. Two checks:

1. **Does it warn in time?** Score each signal as a predictor of "the evaluator will record a failure within the next 25 steps
   (0.5 s)": the AUROC. **Pass bar: ≥ 0.99 on unseen motions.** G1: 0.996 (unseen dances), 0.997 (unseen familiar motions).
2. **Set the threshold on training motions only**, never on the exam, then report on the exam how many failures it caught and how
   many false alarms it raised.

## 7. Decide what happens when it warns

The warning is only useful if something takes over. Options, from simplest: hold the current posture and slow down; switch to a
classical balance controller (PD / LQR; examples in [`for_full_replacement/`](../for_full_replacement/)); hand the step to the
teacher if it is available on board (`--gate S:T` in `pm_eval_student.py` tests exactly this); stop safely. Test the handover in
simulation like any other policy.

## 8. Package for CPU and edge devices

* The G1 MW policy: 1.65 M parameters, 7.7 MB, **2.3 ms per step on one laptop CPU thread** (Intel i5-12450H) including the
  uncertainty; budget 20 ms at 50 Hz. Measure on your target hardware with the same input size.
* Today: plain PyTorch on CPU. Next: ONNX export for phone-class and embedded runtimes (the network is a stack of small dense
  layers plus one triangular matrix product for the uncertainty, so it maps directly).
* Ship the normalisation constants, the warning threshold, the action mapping and the `NOTICE` file with the weights.

## 9. The real robot

Everything above is simulation. Before hardware: randomise the simulator (masses, friction, motor gains, delays) in the exam,
check that the warning still fires before failures, run on the robot on a gantry or harness first, and keep the fallback armed.
Real-robot results are not part of this repository yet.

## Checklist

- [ ] licence allows derivative models; `NOTICE` prepared
- [ ] teacher runs in a scriptable simulator; its own evaluator wraps any policy
- [ ] training motions varied and evenly spread; exam motions never seen
- [ ] plain MLP and MW trained on the same data
- [ ] exam passed clean and under noise, at least as well as the MLP
- [ ] warning AUROC ≥ 0.99 on unseen motions; threshold set on training motions
- [ ] fallback chosen and tested in simulation
- [ ] latency measured on the target CPU; package includes constants and threshold
