# Experiments: three attempts to fit the correction into 20 ms

These are development scripts, kept as they were when the numbers were taken. They are here to be read and adapted. The raw
results are in [`../results/`](../results/README.md).

| Attempt | Code | Result |
|---|---|---|
| **Cheap search + two-level envelope.** Search with N 16, H 15, S 2. Apply 0.6 of the correction as soon as the warning rises and all of it past a higher threshold. | [`../gate/pm_eval_fbw2.py`](../gate/pm_eval_fbw2.py) (`FBW_PID=1`), [`../scripts/run_envelope_58.sh`](../scripts/run_envelope_58.sh) | 50/58; 82 % of steps within 20 ms |
| **Asynchronous search.** The control loop never waits; the C++ search runs on a worker thread and its correction is applied when it arrives, fading with age. | [`async/`](async/), and `FBW_ASYNC=1` in `pm_eval_fbw2.py` for the 58 tests | Loop under 3 ms while the search runs; 40/58, the same as the student alone |
| **Bounded QP in place of the search.** On a warning, hand over to an inverse-dynamics controller with a capture-point term: one QP, no rollouts. | [`bounded_qp/`](bounded_qp/) | Every step under 10 ms; 3/20 development clips, against 9/20 for the search |

## Running the standalone scripts

`async/` and `bounded_qp/` do not use ProtoMotions' evaluator. They step a MuJoCo G1 directly through
`bounded_qp/ms_vs_ppo_standalone.py`, which rebuilds the student's observation itself. They need:

* the simulator's model and metadata exported once from a ProtoMotions run (`g1_env.mjb`, `g1_env.json`; see `load_env` in
  `ms_vs_ppo_standalone.py` for the fields);
* `.motion` clips (`bounded_qp/clips_20.txt` lists the 20 development clips by name) and the student checkpoint;
* this on the path: `PYTHONPATH=bounded_qp:async:../kernel:../gate:../../student/ours:../../flight_computer:$PM`.

Run `python bounded_qp/ms_vs_ppo_standalone.py --check-obs ...` first: it verifies the rebuilt observation against recorded ones.

`async/lead_time_cpp.py` estimated 54/58 by assuming that a correction which arrives before the fall is a rescue. The measured
runs disproved that (`../results/async_resim_dev20.txt`, `../results/async_exam_40of58.txt`). The script is kept as a record of
the mistake: an arrival is not a rescue.
