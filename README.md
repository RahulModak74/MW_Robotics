# MW Robotics — code companion

**Modak-Walawalkar (MW) Framework, trimmed to one domain: robotics.**

The MW Framework learns the physics of a system from data and Bayesian priors, with no GPU
cluster and no PDE solver at inference time — a physics-constrained manifold that scores or
forecasts a state in one forward pass on a CPU. It is validated and in production in one hard
domain: battery fleet analytics, where it holds up against open datasets (Sandia, HUST) and
against an independent researcher's own data. That is the proven core.

This repository takes that same core and points it at a new frontier: **robotics**. It adds one
new domain, a genuine robotics PDE (a flexible-link robot arm's beam dynamics), with zero
changes to the underlying machinery, and runs the first experiment to see how far the approach
carries. See [ROBOTICS.md](ROBOTICS.md) for the result — strong in parts, and a clear map of
where the next iteration should focus, which is exactly what a first experiment on new ground
should produce.

Two Zenodo preprints underlie the approach: the theory above (DOI 10.5281/zenodo.19304813),
and ["World Models with an Irreversible Time Axis"](https://doi.org/10.5281/zenodo.22954118)
(DOI 10.5281/zenodo.22954118), which frames MW as a fourth family of world models alongside
Dreamer, JEPA and generative video — battery evidence so far; see "Read next" below for what
it opens up for robotics.

**Two claims, kept separate so credit lands where it's earned:**
- **The core MW mechanism works, and that is the headline.** It is pushed on the strength of
  real evidence (battery: production use, open-dataset validation, an independent
  researcher's sweep) and it carried over unedited to a brand-new robotics PDE on the first
  try — no small thing for a method with no GPU requirement.
- **The robotics domain in this repo is the first exploration of new ground**, not a finished
  product: new laws, tested once, strong on one fault and still being tuned on two others.
  It is public exactly so other researchers can build on it faster than starting from scratch.
  See `ROBOTICS.md`'s "Status, in two tiers."

## The idea in four lines

1. Physics confines a real system to a thin surface — a **manifold** — inside its sensor space.
2. A Bayesian VAE **learns** that surface from normal data.
3. Known laws enter as **priors and likelihood terms**, so the learned surface must obey them.
4. Anything new is scored by its **MW distance** from the surface. One forward pass. O(n) in the number of features.

```
classical:  physics -> PDEs -> numerical solve (grid, time steps)     -> answer
MW:         physics -> Bayesian priors -> learned manifold -> one pass -> answer
```

## The robotics domain in this repo

`mw_framework/domains/flexible_robot.py` encodes the **Euler-Bernoulli beam PDE**
(`rho*A*w_tt + EI*w_xxxx = f`) — the governing equation for a flexible-link robot arm or a
soft-robot segment (a rigid robot's joints are ODEs; its compliant links and soft actuators
are this PDE). The dataset (`mw_framework/datasets.py::cantilever_beam`) is the exact
closed-form two-mode modal solution for a cantilever released from an initial deflection and
decaying under damping — no numerical solver, the same spirit as the other domains' exact
solutions in the parent repo.

Three laws, written as Bayesian priors, no PDE solved:
- **C1** moment-curvature constitutive law (two independent sensors must agree)
- **C2** a viscous energy-decay envelope
- **C3** a strain/yield limit

## What the first experiment shows

```bash
pip install -r requirements.txt
python examples/00_flexible_robot_validator.py      # a few minutes; GPU auto-used if present
pytest -q
```

Measured result (full writeup in [ROBOTICS.md](ROBOTICS.md)):
- The MW *mechanism* (same `core.py`, zero edits) trains and runs on this new, genuine
  robotics PDE with no changes to the framework — the part worth taking seriously.
- One labelled fault (a stuck damper / lost damping) is caught cleanly: ROC-AUC 1.00.
- Two labelled faults (a local stiffness loss / crack, a moment-sensor glitch) are only
  partly separated so far: ROC-AUC ~0.57–0.64 — a clear target for the next iteration of
  the laws, not a dead end.
- The "validate a coarse simulator" test — the robotics analogue of what the parent repo's
  Navier-Stokes example does for a fluid solver — came back the wrong way round: MW distance
  fell as the approximate simulator's fidelity dropped, instead of rising. That is a specific,
  diagnosable gap in what these three laws currently penalise (see ROBOTICS.md), not a
  dead end for the approach.

A working robotics validator or simulator isn't proven yet. What is proven is that the
mechanism carries over to a new PDE with zero framework changes, and the first run already
shows exactly where to push next.

## What is in the box

```
mw_framework/
  core.py          MWNormaliser, MWVAE, constraint(), bound(), MWTrainer, calibrate(), risk_score()
  config.py        JSON domain configs -> ConfigVAE (linear / range / ratio laws) — generic, no config shipped here
  detector.py      MWDetector: scikit-learn API, explain(), save/load
  algebra.py       ManifoldAlgebra: product, fibre, glue, marginal_distance
  geometry.py      pullback metric, geodesics, path_support, LorentzianCone
  domains/
    flexible_robot.py   FlexibleRobotVAE — the one domain in this trim
  datasets.py      circle/arc (generic framework sanity-check toys) + cantilever_beam, cantilever_beam_coarse
  checkpoint.py    one .pth file = weights + config + normaliser + calibration
  cli.py           train / score / demo-data
examples/
  00_flexible_robot_validator.py
docs/
  LIMITATIONS.md   general framework limits — still fully applicable here
tests/             fast pytest suite (one known pre-existing flaky test, see below)
ROBOTICS.md        what is and is not shown
```

## Known issue carried over from the parent repo

`tests/test_mw.py::test_sklearn_sign_convention` fails intermittently at this fixed seed —
confirmed to fail identically in the unmodified parent `MW_Framework` repo too, so it predates
this trim and is not a robotics-specific regression. Left as-is rather than quietly tuned to pass.

## Read next

- **[ROBOTICS.md](ROBOTICS.md)** — the full result and what it would take to push it further.
- **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)** — what the MW approach is not, and where it fails in general.
- Modak & Walawalkar, ["World Models with an Irreversible Time Axis"](https://doi.org/10.5281/zenodo.22954118)
  (Zenodo, 2026) — the forward-only-time / causal-cone architecture that points toward closing
  this repo's validator-vs-forecaster gap (battery evidence so far; see ROBOTICS.md's "Related
  theory" section for what it opens up for a robotics version).

## Scope

Research code and a single worked example, not a production robotics simulator. Not a safety
system. See [ROBOTICS.md](ROBOTICS.md) and [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## License

Apache-2.0 (see `LICENSE`). If you use this, please cite `CITATION.cff` and the MW theory document.
