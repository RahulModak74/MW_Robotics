# Contributing

Welcome. The most useful contributions:

1. **A new robotics PDE/ODE domain** — a class in `mw_framework/domains/` (or a JSON config — see `mw_framework/config.py` for the schema), plus a generator in `datasets.py` and an example. Every law needs a one-line `why`.
2. **Reproductions on real data** — open an issue with your setup and numbers, good or bad.
3. **Honest failures** — a case where MW loses to a baseline teaches more than one where it wins. Add it to `docs/LIMITATIONS.md`.

Before a pull request:

```bash
pytest -q
./run_all_examples.sh --quick
```

Style: short functions, docstrings that say *why*, physical units in names (`temp_c`, `r_int_mohm`).
