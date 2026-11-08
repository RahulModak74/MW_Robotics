# Contributing

Welcome. The most useful things you could add:

1. **A new teacher policy distilled with MW** (Step 1): any policy whose licence allows derivative models. Follow
   [`docs/POLICY_TO_PRODUCTION.md`](docs/POLICY_TO_PRODUCTION.md) and report the same numbers we do: success on unseen motions,
   clean and under noise, against a plain MLP trained on the same data; warning AUROC; CPU latency. Say which licence applies.
2. **Reproductions** of the G1 result ([`step1_extraction/student/`](step1_extraction/student/README.md)) on your own
   hardware, especially consumer GPUs and plain CPUs. Open an issue with your setup and numbers, good or bad.
3. **Physics priors and fortified policies** (Step 2, [`step2_policy_design/`](step2_policy_design/README.md)): new laws, new robots, teams with GPUs who want
   to train with priors in the loop.
4. **Honest failures:** a case where MW loses to a plain MLP teaches more than a case where it wins. Add it to
   [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

Before a pull request:

```bash
python examples/policy_priors_demo.py 0,1,2,3,4      # must print: plain 18 | prior 0
```
and, if you changed the student or the flight computer, one run of `step1_extraction/flight_computer/run_fbw_D.sh`.

Style: short functions, docstrings that say *why*, physical units in names (`temp_c`, `r_int_mohm`). Never commit teacher weights,
generated motions or student weights derived from a licensed teacher; ship the scripts that regenerate them.
