# Step 1: Bayesian policy extraction

| folder | what |
|---|---|
| [`student/`](student/README.md) | NVIDIA ProtoMotions' G1 tracker distilled into small students (plain MLP, Bayesian action head, state manifold + Bayesian head); collection, training, evaluation with ProtoMotions' own evaluator, per-run evidence |
| [`flight_computer/`](flight_computer/README.md) | the MW fly-by-wire: an exact MuJoCo model, 0.4 s look-ahead with the student in the loop, envelope D; **58/58** with the 100-motion student |

Order to reproduce: the student folder's `Reproduce` section (motions → teacher rollouts → students), then
`flight_computer/run_fbw_D.sh <student.pt>`. How the envelope was designed: [`../docs/DESIGNING_THE_ENVELOPE.md`](../docs/DESIGNING_THE_ENVELOPE.md).
