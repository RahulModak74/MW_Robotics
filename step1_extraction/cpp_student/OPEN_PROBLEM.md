# The open problem: 58 of 58 inside the control step

We have not solved this. We are publishing it half-done because the missing idea may well be yours. Everything below is either
measured (with the file that holds the numbers) or clearly marked as untried.

## The problem in one paragraph

A small student policy drives a Unitree G1 and falls on 18 of 58 tests. A physics flight computer rescues all 18 by searching,
at each dangerous step, over 63 candidate corrections, each simulated 0.4 s ahead in an exact model. That search costs about
260 ms in C++ on a laptop. The robot needs a command every 20 ms. **Find a way to keep the 18 rescues with every decision
inside the control step, on a CPU.** Reaching 58 of 58 at 200 ms per corrected step would already be a useful result in
simulation; 20 ms is the real target.

## What is already known

| Fact | Evidence |
|---|---|
| The student's uncertainty is a good enough alarm: gating on it keeps 58/58 and skips the physics on about 80 % of steps | [`results/gated_58of58.md`](results/gated_58of58.md) |
| The alarm comes early: on 16 development clips where the student fell, it led the fall by 25 to 151 steps | [`results/capture_point_warning_dev20.txt`](results/capture_point_warning_dev20.txt) |
| Adding a capture-point alarm does not help: it fired earlier on 6 of 16 falls and added no useful rescue | same file |
| The search cost is two-valued: about 100 ms when no correction is needed, seconds (Python) when one is | same file |
| The exact look-ahead has a floor: 400 physics steps, about 13 ms, for one candidate | [`kernel/README.md`](kernel/README.md) |
| A correction applied about 14 steps late rescues nothing (40/58: saved 3, broke 3). Rolling the plan forward did not fix it | [`results/async_exam_40of58.txt`](results/async_exam_40of58.txt) |
| A search a quarter of the size gets 10 of the 18 rescues (50/58) | [`results/two_envelope_50of58.txt`](results/two_envelope_50of58.txt) |
| How hard the correction is applied matters at the edge: soft gain 0.6 gave 6/7 on the dance runs, 0.618 gave 5/7, 1.8 times overdrive 4/7 | same file |
| One bounded QP with a capture-point term, in place of the search: 3/20 against 9/20 | [`results/bounded_qp_dev20.txt`](results/bounded_qp_dev20.txt) |
| Coarser look-ahead physics changes little at 2 ms (pass/correct agreed on 60/60 states; 1/60 chosen corrections fell under the exact model) and more at 5 ms with a small search (16/60) | [`kernel/README.md`](kernel/README.md), on a stand-in student with random weights |

Two results from our other experiments, not in this repository, point the same way. On a quadruped model, a network trained to
copy the flight computer's corrections and a learned look-ahead in place of the physics both lost almost all the protection.
On the G1, cloning a search-based controller into a feedforward network (behaviour cloning, then four rounds of DAgger) scored
0 of 58. The skill seems to live in searching again at every step, not in a map from state to action.

## Ideas we have not tried

Each has a cheap first test. If you run one, please tell us the result either way.

1. **Full search with 2 ms look-ahead physics.** `FBW_CPP=1 FBW_LA_DT=0.002 scripts/run_gated_58.sh`. It should roughly halve the
   cost of a correction. The drift test says the decisions barely change, but that was on a stand-in student. This is the
   cheapest untested run in the folder.
2. **Search sizes between the cheap one and the full one.** N 16 / H 15 / S 2 gives 50; N 64 / H 20 / S 4 gives 58. Nobody has
   measured N 32 or N 48, or H 20 with a small N. Find the smallest search that keeps 58 and time it.
3. **A correction that stays valid when it arrives late.** The asynchronous search failed because its answer was a fixed joint
   offset. Return a small feedback law instead (for example, a gain on the capture-point error, or a short plan that is
   re-anchored to the current state) and the 14-step delay may stop mattering. First test: the four dance rescue runs with
   `FBW_ASYNC=1`.
4. **Search in fewer dimensions.** The search samples 2 knots times 29 joints. The corrections that win may lie in a handful of
   directions (ankles, hips, arms as a counterweight). Log the winning corrections, find their main directions, and sample only
   there with far fewer candidates.
5. **Spend the warning time.** The alarm leads the fall by tens of steps, and the search is idle for most of them. Spread one
   search over several control steps while the first, cheap corrections are already being applied.
6. **A learned envelope with an exact check.** Train a small network to predict the 0.4 s envelope cost and use it to pick three
   or four candidates, then verify only those in the exact model (about 13 ms each). The network proposes; the physics decides.
   Offline first test: on logged gated states, does the true best correction appear among the network's top four? Be warned that
   a network used in place of the physics, with no exact check, is the version that failed on the quadruped.
7. **A simpler body model inside the search.** Score most candidates with a centroidal or pendulum model and only the finalists
   with the full model.
8. **Batched physics on a consumer GPU** (MuJoCo MJX or similar). 63 rollouts in parallel is a natural fit for the laptop GPU many
   teams already own. It leaves the CPU-only goal, so it is a comparison point.
9. **More authority, less search.** Some falls may need a step or an arm swing that small joint corrections cannot produce. A
   capture-point stepping rule or an angular-momentum task could turn hard rescues into easy ones.
10. **Give the student memory.** A recurrent student, or one that sees a short history, might learn part of the recovery itself
    and call the flight computer less often.

## How to report a result

Use the same 58 tests and the same frozen student. Report the score out of 58, how many of the 40 tests that pass alone still
pass, and the decision time per step (median, 99th percentile, worst, share within 20 ms). Say which CPU, which governor, how
many threads and how many runs. Do not tune thresholds on the 58 tests; use training motions. A clean negative result is
welcome: add it to the table above in a pull request.
