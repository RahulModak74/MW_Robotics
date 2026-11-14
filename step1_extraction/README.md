# Step 1: Bayesian policy extraction

| folder | what |
|---|---|
| [`student/`](student/README.md) | NVIDIA ProtoMotions' G1 tracker distilled into small students (plain MLP, Bayesian action head, state manifold + Bayesian head); collection, training, evaluation with ProtoMotions' own evaluator, per-run evidence |
| [`flight_computer/`](flight_computer/README.md) | the MW fly-by-wire: an exact MuJoCo model, 0.4 s look-ahead with the student in the loop, envelope D; **58/58** with the 100-motion student |
| [`cpp_student/`](cpp_student/README.md) | towards real time: the student's uncertainty as a gate (still 58/58, physics skipped on about 80 % of steps), the look-ahead in C++ (same decisions, about 4.5 times faster), and the open problem (50/58 at near real-time speed) |

Order to reproduce: the student folder's `Reproduce` section (motions → teacher rollouts → students), then
`flight_computer/run_fbw_D.sh <student.pt>`. How the envelope was designed: [`../docs/DESIGNING_THE_ENVELOPE.md`](../docs/DESIGNING_THE_ENVELOPE.md).
