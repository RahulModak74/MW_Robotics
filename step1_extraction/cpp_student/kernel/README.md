# The kernel: fly-by-wire look-ahead and the student, in C++

`fbw_kernel.cpp` runs `FlyByWire._score` (the look-ahead loop of [`../../flight_computer/fbw.py`](../../flight_computer/fbw.py)) natively. `FlyByWireCpp` (`fbw_fast.py`) subclasses `FlyByWire` and replaces
only `_score`. `filter()`, the warm start, MPPI blending and stats stay as they are.

Per horizon step, for every rollout in parallel (OpenMP, one `MjData` per thread):
1. target = clip(pilot + correction + noise)
2. 20 × `mj_step`
3. one kinematics pass: proprioception, capture point, torso and fall checks
4. envelope cost
5. `next_obs`

Then the student's action path runs once on all live rollouts, as a float32 MLP in Eigen.

## Build and use

```bash
pip install pybind11 && sudo apt-get install -y libeigen3-dev      # build-time only
./build.sh                       # g++ -O3 -march=native -fopenmp, Eigen 3.4, pybind11, the mujoco wheel's own libmujoco
PM=<ProtoMotions clone> python bench/verify.py                     # C++ vs fbw.py: must agree to about 1e-9
FBW_CPP=1 ST=<student.pt> E=<held-out motions> ../scripts/run_gated_58.sh
```

`FBW_CPP=1` makes `../gate/fbw_2.py` build `FlyByWireCpp` instead of `FlyByWire`. Use the same Python for `build.sh` as for the runs.

Settings (environment):

| Variable | Default | Effect |
|---|---|---|
| `FBW_CPP=1` | off | the gated flight computer uses `FlyByWireCpp` |
| `FBW_THREADS` | 6 | Rollout threads. Keep it modest: the quiet check is one rollout and does not parallelise |
| `FBW_KEEP_WARM=1` | off | Carry the solver warm start along each rollout, seeded from the world's. About 10 % faster. No longer bit-identical to `fbw.py`. |
| `FBW_LA_DT` | 0 = 1 ms (exact) | Look-ahead physics step, e.g. 0.002. Changes the prediction. |
| `FBW_H_CORR` | = `FBW_H` | Horizon for scoring corrections. Changes the result. |

## Exact shortcuts (same results as `fbw.py`)

* **Noise draws:** when `FBW_LA_NOISE` = 0 (all clean tests), the S noise draws are identical, so one is rolled out instead of S.
* **Fallen rollouts:** a rollout that has fallen (`FBW_FALL=1`) is not simulated further, because its cost is already final.
* **Student action path only:** `uq()` also runs the MW decoder, logvar and, for LastLayer, the network a second time plus a 513×513
  product. None of that changes the action.

## Verified (`bench/verify.py`)

| Check | Result |
|---|---|
| Student action vs torch `uq()["action"]` (lastlayer-mwv3, mwv2, gauss) | max diff 1e-7 rad (float32) |
| `_score` costs: pilot plan and 63 corrections, noise 0 and 0.2 | max diff 2e-9 (relative 1e-12); same fall flags, same 4 elites |
| `filter()` over 20 live ticks | same decision 20/20; sent command diff 1e-10 rad |

## Timing (`bench/bench_cpp.py`, 2-core Xeon at 2.1 GHz, config D)

| Tick | `fbw.py` | C++ exact |
|---|---|---|
| Quiet, clean | 110 ms | 24–26 ms (1 thread) |
| Quiet, noisy (S=4) | 110 ms | 51 ms on 2 threads; about 26 ms with ≥ 4 threads |
| Correction, clean (63 rollouts) | 2.9 s | 0.63 s (2 threads) |
| Correction, noisy (252 rollouts) | 2.9 s* | 2.5 s (2 threads) |

*The `fbw.py` baseline was timed with zero noise, but it always simulates all 4 draws, so its noisy tick costs about the same.

The quiet tick is now pure physics: 400 sequential `mj_step` calls of the G1 at 1 ms, 25.6 ms on this core. Everything else takes
under 1 ms. That is the floor for an exact 0.4 s look-ahead. Run `bench/bench_cpp.py` on the target CPU to read off its own floor.

**Corrections cannot reach 20 ms with the exact model.** 63 × 400 = 25,200 steps is about 0.8 s of physics even on a fast core.

## Fast mode costs fidelity (`bench/drift.py`, 60 world states vs the exact model)

| Mode | Pass/correct agrees | Chosen correction falls under the exact model | One-step qpos error |
|---|---|---|---|
| 2 ms physics | 60/60 | 1/60 | 3e-3 median |
| 5 ms physics | 60/60 | 2/60 | 1.3e-2 median |
| 5 ms, N 16, H_corr 10 | 60/60 | 16/60 | – |

Timing with 2 threads: 5 ms quiet 8 ms, correction 191 ms; 5 ms + N 16 + H_corr 10, correction 35 ms.

These states come from a random-weight stand-in for the student, and corrections are sampled without the warm start. Any fast
setting must pass the 58-test protocol before use.

## Bench setup

`bench/g1_model.py` rebuilds ProtoMotions' MuJoCo G1 from its MJCF and robot config:
- 1 kHz physics, decimation 20
- implicit PD actuators with the config's kp/kd/effort limits and armature
- friction 1.0
- visual meshes dropped (contype 0, density 0, so no effect on physics)

`bench_setup.py` uses a random-weight student of the r7c architecture (LastLayer over MWBundleV3, 512 hidden, 16 latent). Compute
cost does not depend on the weights.

## Timing on a laptop (Intel i5-12450H, exact mode, config D)

| | Performance governor, idle machine | Power-save governor (0.4-1 GHz) |
|---|---|---|
| One `mj_step` of the G1 | 32 us | about 62 us |
| Quiet check (400 steps) | about 13 ms | 32 ms (bench); about 78 ms inside the exam |
| Full correction (63 rollouts, 6 threads) | about 260 ms; 370 ms next to a live control loop | about 0.9 s inside the exam (one dance clip, 69 corrections) |

Set the governor before timing anything: `sudo cpupower frequency-set -g performance`. `bench/bench_cpp.json` holds the throttled
numbers. Sources: [`../results/kernel_timing_and_async_proof.txt`](../results/kernel_timing_and_async_proof.txt).
