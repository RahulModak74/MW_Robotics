# Contributing

Welcome. The most useful things you could add:

1. **A new robot example** — a new physics system, as a class in `mw_framework/domains/` (or
   a JSON config, see `mw_framework/config.py`), plus a data generator in `datasets.py` and an
   example script. Every physics rule needs a one-line note on *why* it's there.
2. **Tests on real data** — open an issue with your setup and your numbers, whether they're
   good or bad.
3. **Honest failures** — a case where MW loses to a simpler baseline teaches more than a case
   where it wins. Add it to `docs/LIMITATIONS.md`.

Before a pull request:

```bash
pytest -q
./run_all_examples.sh --quick
```

Style: short functions, docstrings that say *why*, physical units in names (`temp_c`, `r_int_mohm`).
