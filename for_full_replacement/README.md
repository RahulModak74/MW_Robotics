# Part 2 building blocks: toward policy generation (`for_full_replacement/`)

Part 2 of MW Robotics is about producing **heavily fortified policies** in the first place, and replacing parts of today's
policy-generation pipelines with Bayesian, physics-first pieces, together with labs and teams who bring compute and policies. The
plan is in [`../ROADMAP.md`](../ROADMAP.md). Part 1 (distillation) is ready and lives in [`../for_distilation/`](../for_distilation/).

**Status: research.** This folder holds the earlier building blocks that Part 2 reuses. They are experiments with their logs, not
finished components.

| Building block | Where | What it gives Part 2 |
|---|---|---|
| Coupled-body validator (Walker2d, 17-joint Humanoid) | `mw_to_protomotions/` (`walker_domain.py`, `stage1_validator_test.py`, `stretch_humanoid*.py`) | MW distance as a per-step "is this state / torque physically sensible?" signal |
| Early warning | `mw_to_protomotions/stage7_*.py` | on the simulated falls tested, the warning came before the simulator's own termination ([`mw_to_protomotions/STATUS.md`](mw_to_protomotions/STATUS.md)) |
| Richer data, fewer false alarms | `mw_to_protomotions/stage4_*.py` | a measured dose-response curve: more diverse data, fewer false alarms on unseen physics |
| Classical fallback controllers | `mw_to_protomotions/classical_controller.py`, `lqr_balance.py`, `tune_balance.py` | the safe fallback when a policy warns |
| CPU policy search without a teacher | `mw_to_protomotions/cem_policy_search.py`, `domain/cem_humanoid.py` | gradient-free search over compact policies, entirely on CPU; strong on Walker2d, weak on the full humanoid |
| Transition model on the manifold | `mw_g1_racer/` (`transition_domain.py`, `train_transition_vae.py`, `b2_mpc.py`) | a learned dynamics model and a first look-ahead controller |
| Policy compression into a VAE | `domain/exp_e*.py` | the data pipeline Part 1 grew out of |

Logs (`*_full.log`, `logs/`) next to the scripts show complete runs with their numbers.

```bash
cd for_full_replacement
pip install -e ..                                        # mw_framework
python mw_to_protomotions/stage1_validator_test.py       # coupled-body validator
python mw_to_protomotions/cem_policy_search.py           # CPU policy search on Walker2d
(cd mw_g1_racer && python validate_transition_vae.py)    # transition model check
```

What we learned here shaped Part 2: a teacher-free policy needs the physics to be exact, not learned (a learned dynamics model
misleads a strong prior), and it needs look-ahead, not just a one-step check. Both are now the core of the plan.

*Research code, simulation results; not a certified safety system.*
