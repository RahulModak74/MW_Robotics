# Limitations — read before deploying

Scope, inherited from the general [MW Framework](https://github.com/RahulModak74/MW_Geometric_Intelligence)
this repo is trimmed from. Each point was observed while building that repo's seven examples;
the numbered examples referenced below (00, 01, 05, 06) live there, not in this trim — this
repo's own result, with its own mixed findings, is in [ROBOTICS.md](../ROBOTICS.md).

**1. Physics does not always beat a plain VAE.**
On clean, plentiful data with simple laws, they tie (examples 01 and 06). The prior pays off with dirty or sparse data, for learning constants, and for laws the data rarely exercises.

**2. Tolerances can hurt.**
A law that is tighter than the decoder's reconstruction noise forces the model to trade fit for law-compliance. Normal data then scores worse and detection drops. Start loose. Tighten while watching held-out AUC.

**3. Posterior widths are optimistic.**
The battery example recovers Ea within a few percent, but its reported sd is far smaller than the true error. Mean-field variational inference and the decoder's approximation both shrink uncertainty. Treat posterior sd as a lower bound. Validate constants against a lab test.

**4. Latent dimension matters.**
Too few, and the manifold cannot fit (a closed loop needs 2 latent dims, example 00). Too many, and the decoder invents states in regions with no data; geodesics will cut through them (example 05). Use `path_support()` and aim near the system's true degrees of freedom.

**5. MW sees what is off the manifold — not what is subtle along it.**
An attacker or fault that moves the system to another *normal* state is invisible to a snapshot score. Use the Lorentzian cone for rates and direction over time; use domain constraints for known bad regions.

**6. Sensitivity has a floor.**
The validator in example 06 does not notice small solver errors (grid step ≤ 0.2). Scores average over all features, so one slightly wrong feature can hide in noise. For single critical laws, monitor the law residual directly as well.

**7. The training set defines "normal".**
Operating regimes absent from training (a new season, a new firmware) will raise false alarms. Retrain or add data. Calibration assumes the held-out normal data looks like future normal data.

**8. O(n) is about inference.**
Scoring one state costs one encoder and one decoder pass. Training still needs data, epochs and tuning. The comparison with PDE solvers is about *checking or emulating states*, not about solving the PDE to the same accuracy.

**9. Synthetic data.**
All examples use physics-generated data with known truth. Results on real systems will be lower and noisier. Treat the numbers here as a demonstration of mechanism, not as benchmarks.

**10. Not a safety system.**
This is research software. Do not use it as the sole protection for equipment or people.
