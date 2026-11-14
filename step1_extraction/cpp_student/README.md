# The C++ student: making the flight computer fast enough for a real robot

**Status: experimental, and unfinished on purpose.** The slow version scores 58 of 58. The fastest version that keeps every
decision near the robot's 20 ms control step scores 50 of 58. Closing that gap is an open problem, and this folder holds
everything we built and measured so that you can study it, rerun it and try your own idea:
[`OPEN_PROBLEM.md`](OPEN_PROBLEM.md).

## Why this student is needed

Step 1 of this repository produced two parts:

* **The student** ([`../student/`](../student/README.md)): an 8 MB network distilled from NVIDIA's G1 tracker. It decides in
  2.3 ms on one laptop CPU thread and reports how unsure it is. Alone it completes 40 of 58 tests.
* **The flight computer** ([`../flight_computer/`](../flight_computer/README.md)): a physics look-ahead that checks the
  student's command against the laws of balance and corrects it when needed. With it the pair completes 58 of 58.

The flight computer is what turns 40 into 58, but it thinks slowly. The robot needs a command every 20 ms. The Python flight
computer takes about 0.16 s to approve a command and about 3 s to correct one. In simulation that is fine, because the simulator
waits. A real robot does not wait: it falls.

So the question this folder works on is: **can the same physics protection run inside the robot's control step, on an ordinary
CPU?** That would let a small student, with no GPU at run time, be as safe as the large policy it was distilled from. We call the
result the "C++ student" because the student's own weights run inside the C++ look-ahead, next to the physics.

## Where it sits

```
                      every 20 ms
 observation ──► student (pilot) ── joint targets ──► MW gate ──► robot
                   │   2.3 ms                           │
                   └─ "how sure am I?" (epi) ───────────┤
                                                        │ quiet: pass the command through unchanged (no physics)
                                                        │ warning: ask the flight computer
                                                        ▼
                                        flight computer (C++ kernel)
                                        exact MuJoCo model of the robot, 0.4 s look-ahead,
                                        the student's weights running inside the loop,
                                        envelope laws: capture point, torso upright, predicted fall
```

Three pieces, each in its own folder:

| Folder | What it is | What it buys |
|---|---|---|
| [`gate/`](gate/) | `fbw_2.py`: the student's own uncertainty decides **when** the flight computer acts. Follow-through: once warned, the flight computer keeps control until the robot is stable again. `pm_eval_fbw2.py` runs it under ProtoMotions' evaluator. | The physics is skipped on about 80 % of steps, with no loss: still 58/58 |
| [`kernel/`](kernel/README.md) | `fbw_kernel.cpp`: the look-ahead in C++ (MuJoCo C API, OpenMP, the student as a float32 network in Eigen). `fbw_fast.py` is the drop-in wrapper. | The same decisions as the Python search (costs agree to 1e-9), about 4.5 times faster |
| [`experiments/`](experiments/README.md) | Three attempts to fit the correction into 20 ms: a cheaper search with a two-level envelope, an asynchronous search, and a bounded QP in place of the search. | Measured answers, including the ones that did not work |

## What we measured

Unitree G1 in MuJoCo, ProtoMotions' own evaluator, the same 58 tests and the same frozen student as the rest of Step 1. Every row
is one run of the 58 tests unless it says otherwise. Laptop: Intel i5-12450H.

| Version | Score | Time per decision | Where |
|---|---|---|---|
| Student alone | 40/58 | 2.3 ms | [`../student/`](../student/README.md) |
| Flight computer on every step (config D, Python) | **58/58** | about 0.16 s to approve, about 3 s to correct | [`../flight_computer/`](../flight_computer/README.md) |
| **MW gate + follow-through**, full search (Python) | **58/58** | median 2.6 ms; 78.7 % of steps within 20 ms; gated steps average 3.2 s | [`results/gated_58of58.md`](results/gated_58of58.md) |
| MW gate, cheap search (N 16, H 15, S 2), two-level envelope | 50/58 | 82 % of steps within 20 ms; gated steps p99 195 ms in Python | [`results/two_envelope_50of58.txt`](results/two_envelope_50of58.txt) |
| Asynchronous full search (the answer arrives about 14 steps late) | 40/58 | every step under 20 ms | [`results/async_exam_40of58.txt`](results/async_exam_40of58.txt) |
| Bounded QP in place of the search (20 development clips) | 3/20, against 9/20 for the search | every step under 10 ms | [`results/bounded_qp_dev20.txt`](results/bounded_qp_dev20.txt) |

The C++ kernel, measured separately ([`kernel/README.md`](kernel/README.md)):

* It reproduces the Python search: costs agree to 1e-9, and 20 of 20 live decisions are the same.
* Full correction: about 260 ms on an idle machine with the performance governor (6 threads), against about 1.2 s in Python.
* Quiet check (the student's own plan, 0.4 s ahead): about 13 ms.
* One exam clip through the gate with the kernel on (`FBW_CPP=1`, dance, no noise): completed, same as Python. The laptop was in
  power-save mode that day, so the 69 corrections took about 0.9 s each. That is one clip, not a timing claim.

**What has not been run:** the full 58 tests with the kernel inside the gate. The "about 43 ms p99" quoted for the cheap search
in C++ is the Python time divided by the measured 4.5 times speed-up, not a separate run.

## What the numbers say

1. **The gate is free.** Trusting the student's uncertainty removes the physics from four steps in five and loses nothing.
2. **C++ is not the missing piece.** The kernel is exact and 4.5 times faster, and a full correction is still about 13 control
   steps long. An exact 0.4 s look-ahead is 400 physics steps; 63 candidates are 25,200.
3. **A late answer is no answer.** The asynchronous search kept the loop under 20 ms and rescued nothing: it saved 3 tests and
   broke 3. A joint correction computed for one instant is wrong 14 steps later.
4. **A cheap search gets half the rescues.** 10 of the 18, with most steps in time.
5. **The search itself is what protects.** Replacing it with one bounded QP lost most of the rescues.

So the open problem is narrow and well posed: **find the 8 remaining rescues without a 25,000-step search, or make the search
fit.** Ideas we have not tried, with a cheap first test for each: [`OPEN_PROBLEM.md`](OPEN_PROBLEM.md).

## Run it

Set up ProtoMotions, the teacher checkpoint, the motions and the 100-motion student as in
[`../student/README.md`](../student/README.md#reproduce). We ship no weights; the scripts regenerate them. Then:

```bash
export PM=<ProtoMotions clone> PY=<its python>
export ST=<path>/r7c_mwv3ctl_s100.pt          # the 100-motion student
export E=<folder with lib_proto_eval_familiar.pt and lib_proto_eval_dance.pt>

cd kernel && ./build.sh && python bench/verify.py && cd ..     # build the kernel, check it against the Python search

scripts/run_gated_58.sh                       # 58/58 version, Python search (hours on a laptop)
FBW_CPP=1 scripts/run_gated_58.sh             # the same search in C++
scripts/run_envelope_58.sh                    # 50/58 version: cheap search, two-level envelope
scripts/run_gated_58.sh score                 # print the score and timing table again
```

Each run is resumable and writes one log and one `.npz` per test to `results_runs/`: per step, the warning signal, whether the
gate was open, whether the flight computer corrected, and the time it took. The score table reports the share of steps within
20 ms and within 200 ms. Run one job at a time, and set the CPU governor to `performance` before you trust a timing.

Every setting is an environment variable: `FBW_*` for the search and envelope (see `CFG` in
[`../flight_computer/fbw.py`](../flight_computer/fbw.py)), `FBW2_*` for the gate (see [`gate/fbw_2.py`](gate/fbw_2.py)),
`FBW_PID_*` for the two-level envelope and `FBW_ASYNC*` for the latency model (see [`gate/pm_eval_fbw2.py`](gate/pm_eval_fbw2.py)).

## Limits, stated plainly

Simulation only. One student, one run per configuration, one laptop. The flight computer's model is the simulator itself and it
reads the true state. The gate threshold (0.0136) was set on the student's training motions, not on the exam; keep it that way.
