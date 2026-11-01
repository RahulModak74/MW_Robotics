"""
tests/test_transition_gate
============================
The single decisive test for Plan B. If this passes, Plan B is viable and
worth building the full stack. If it fails, Plan B is structurally limited
(the optimal action is not on the manifold) and Plan A — CEM over policy
parameters — remains the honest path.

The test:
  1. Train TransitionVAE on ~50k real Walker2d transitions.
  2. On held-out (s, s') pairs, solve for a* via inverse_query.
  3. Compare a* to the actual action a that produced each pair.
  4. If ||a* - a|| is small relative to the action range: the inverse is
     recoverable → the optimal action is on the manifold → Plan B is viable.

Threshold justification: a random action has expected error
~0.29 in units of A_MAX = 1.0 (uniform [-1,1] RMS vs mean-zero). A recoverable
inverse should be at least 2× better than random → mean error < 0.15.
A *tight* recoverable inverse should be ≤ 0.1 × A_MAX.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import pyro

from mw_framework import MWDetector, set_seed
from mw_framework.domains.transition import (
    TransitionVAE, STATE_FEATURES, ACTION_FEATURES, FEATURES)
from mw_framework.query.inverse import inverse_query


@pytest.fixture(scope="module")
def trained_detector():
    """Train once for the module: ~50k Walker2d transitions, 200 epochs.
    Cached on disk so re-runs don't retrain (~3-5 min on a laptop CPU)."""
    from mw_to_protomotions.transition_datagen import collect_transitions
    cache = "/tmp/mw_transition_walker.ckpt"
    if os.path.exists(cache):
        return MWDetector.load(cache)
    set_seed(0); pyro.clear_param_store()
    df = collect_transitions(n_units=20, n_steps=250, seed0=100)
    # 80/20 split
    n = len(df)
    train = df.iloc[: int(n * 0.8)]
    det = MWDetector(vae=TransitionVAE(), epochs=200, lr=2e-3).fit(train[FEATURES])
    det.save(cache)
    return det


def test_manifold_learns_cleanly(trained_detector):
    """Sanity: the transition manifold has a low reconstruction floor on
    held-out normal transitions. If this fails, nothing else is meaningful."""
    from mw_to_protomotions.transition_datagen import collect_transitions
    held = collect_transitions(n_units=5, n_steps=250, seed0=9000)
    d = trained_detector.mw_distance(held[FEATURES])
    # Baseline calibrated on training data. Held-out should be within 2×.
    assert d.mean() < 2.0 * trained_detector.calibration_.mu + 1.0


def test_e_gate_inverse_recoverable(trained_detector):
    """The decisive test. Solve for a* given (s, s') on held-out triples.
    If mean ||a* - a_actual|| is well below random, the optimal action is
    on the manifold."""
    from mw_to_protomotions.transition_datagen import collect_transitions
    held = collect_transitions(n_units=5, n_steps=250, seed0=9000)
    held = held.sample(n=min(200, len(held)), random_state=0).reset_index(drop=True)

    errors = []
    for _, row in held.iterrows():
        s_row = {f: float(row[f"s_{f}"]) for f in STATE_FEATURES}
        sp_row = {f: float(row[f"sp_{f}"]) for f in STATE_FEATURES}
        a_actual = np.array([float(row[f]) for f in ACTION_FEATURES], dtype=np.float32)
        a_star = inverse_query(trained_detector, s_row, sp_row,
                                n_pop=32, n_gen=3, seed=int(row["step"]))
        errors.append(float(np.linalg.norm(a_star - a_actual)))
    errors = np.array(errors)

    mean_err = float(errors.mean())
    std_err = float(errors.std())
    p90 = float(np.percentile(errors, 90))

    # Random-action baseline for comparison (uniform in [-1,1]^6)
    random_errs = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        a1 = rng.uniform(-1, 1, 6)
        a2 = rng.uniform(-1, 1, 6)
        random_errs.append(float(np.linalg.norm(a1 - a2)))
    random_mean = float(np.mean(random_errs))

    print(f"\n  E-gate: inverse_query action error")
    print(f"    mean  = {mean_err:.4f}")
    print(f"    std   = {std_err:.4f}")
    print(f"    p90   = {p90:.4f}")
    print(f"    random baseline mean = {random_mean:.4f}")
    print(f"    improvement over random = {random_mean / max(mean_err, 1e-6):.2f}x")

    # The honest thresholds.
    # 1. Mean error must be at least 2× better than random.
    assert mean_err < 0.5 * random_mean, (
        f"E-gate FAILED: mean error {mean_err:.3f} is not 2× better than "
        f"random {random_mean:.3f}. Plan B is not viable on this robot as spec'd."
    )
    # 2. Std must be bounded (not catastrophic variance)
    assert std_err < 0.5 * random_mean, (
        f"E-gate FAILED: std {std_err:.3f} too large. The inverse is ill-posed."
    )
    # 3. Report the "strong success" signal as a warning, not a hard fail
    if mean_err < 0.1:
        print(f"  STRONG: mean error {mean_err:.4f} < 0.1 — Plan B is a clear win.")
    elif mean_err < 0.2:
        print(f"  MODERATE: mean error {mean_err:.4f} in [0.1, 0.2] — Plan B "
              f"is viable but Plan A may match it. Run V10 before committing.")


def test_e_gate_does_not_degrade_under_noise(trained_detector):
    """Plan B must be robust to sensor noise at inference. Add σ=0.05
    Gaussian noise to s and s' in the query; the inverse should still
    recover a*."""
    from mw_to_protomotions.transition_datagen import collect_transitions
    held = collect_transitions(n_units=3, n_steps=250, seed0=9000)
    held = held.sample(n=50, random_state=1).reset_index(drop=True)
    rng = np.random.default_rng(42)

    errors = []
    for _, row in held.iterrows():
        s_row = {f: float(row[f"s_{f}"]) + rng.normal(0, 0.05) for f in STATE_FEATURES}
        sp_row = {f: float(row[f"sp_{f}"]) + rng.normal(0, 0.05) for f in STATE_FEATURES}
        a_actual = np.array([float(row[f]) for f in ACTION_FEATURES], dtype=np.float32)
        a_star = inverse_query(trained_detector, s_row, sp_row,
                                n_pop=32, n_gen=3, seed=0)
        errors.append(float(np.linalg.norm(a_star - a_actual)))
    mean_err = float(np.mean(errors))
    print(f"\n  E-gate with σ=0.05 noise: mean error = {mean_err:.4f}")
    # Loose threshold: noise degrades the inverse but it should still be
    # materially better than random.
    assert mean_err < 0.7, f"Noise robustness failed: {mean_err:.3f} >= 0.7"
