# Results

Raw result files from the development tree, unedited apart from this index. Script and folder names inside them refer to that
tree: `fbw_2.py` and `pm_eval_fbw2.py` are in [`../gate/`](../gate/), `cpp/` is [`../kernel/`](../kernel/), `ASYNC_CPP` is
[`../experiments/async/`](../experiments/async/), `hard_guarantee` is [`../experiments/bounded_qp/`](../experiments/bounded_qp/).
One run each, Intel i5-12450H.

| File | What it shows |
|---|---|
| [`gated_58of58.md`](gated_58of58.md) | MW gate + follow-through, full search: 58/58; median decision 2.6 ms; 78.7 % of steps within 20 ms |
| [`two_envelope_50of58.txt`](two_envelope_50of58.txt) | Cheap search with a two-level envelope: 50/58 per test, timing, and the gate-threshold table |
| [`async_exam_40of58.txt`](async_exam_40of58.txt) | Asynchronous search on the 58 tests: 40/58 (saved 3, broke 3) |
| [`async_resim_dev20.txt`](async_resim_dev20.txt) | The same question on 20 development clips: 3/20 asynchronous, 6/20 synchronous in that harness |
| [`kernel_timing_and_async_proof.txt`](kernel_timing_and_async_proof.txt) | Kernel timing at full clock; the control loop stays under 3 ms while the search runs. Its closing estimate (54/58) is retracted in the file itself |
| [`bounded_qp_dev20.txt`](bounded_qp_dev20.txt) | Bounded QP in place of the search: timing and 3/20 |
| [`capture_point_warning_dev20.txt`](capture_point_warning_dev20.txt) | Warning lead times per clip; a capture-point alarm adds no rescue |
