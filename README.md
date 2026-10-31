# MW Robotics

**The Modak-Walawalkar (MW) method, applied to robotics.**

## What we want MW Robotics to be

We want a robot that can move, balance, and react in the real world — the way
[ProtoMotions](https://nvlabs.github.io/ProtoMotions/tutorials/) already does — but built on
MW's approach: learning from known physics, not from millions of expensive simulations on a
GPU farm. Real robot skill, without needing a GPU cluster to get there.

## What ProtoMotions does

[ProtoMotions](https://nvlabs.github.io/ProtoMotions/tutorials/) is an NVIDIA project that
trains a simulated humanoid robot to move realistically — walking, running, reacting to being
pushed, and many other skills — by trial and error inside a physics simulator. It can also
take real human motion-capture data and teach a robot to copy it, then run that on a real
robot (like the Unitree G1). This is a genuinely impressive result. It's also expensive:
training it needs a lot of GPU power running in parallel for a long time.

## Where we are now

MW Robotics can already check whether a robot's state is physically normal or broken, for
six simple parts of a robot — a bending beam, a motor joint, an arm, a wheeled robot, a leg,
and a motor's heat. Proven, tested, documented.

It can also guess a robot's next state quickly, without running a full simulator — proven on
one of those six parts (the leg), tested against a real physics engine (MuJoCo).

The same trick — guess the next state without a simulator — is also already working in a
completely different field: predicting battery health, already running in production. That
shows the idea is real, not just a theory.

None of this needs a GPU.

## What needs to be done

Getting from here to a full ProtoMotions-style robot needs several more steps: putting many
joints together into one coupled body, real-time control that reacts to a push or a stumble,
learning from real human motion data, and eventually running on a real robot.

The full plan, one step at a time, with how likely each step is to actually work, is here:

**[path_to_full_Robotics.md](path_to_full_Robotics.md)**

## Try it

```bash
pip install -r requirements.txt
python examples/00_flexible_robot_validator.py      # or 01..05
pytest -q
```

## More detail

- **[path_to_full_Robotics.md](path_to_full_Robotics.md)** — the plan to get to full robotics.
- **[ROBOTICS.md](ROBOTICS.md)** — full results for the six robot parts proven so far.
- **[FORECASTER.md](FORECASTER.md)** — the leg, tested as a predictor against MuJoCo.
- **[FIXING_THE_GAPS.md](FIXING_THE_GAPS.md)** — fixes to try for the weak spots found so far.
- **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)** — where this approach falls short, in general.
- **[full_flow_animation/](full_flow_animation/)** — an interactive demo you can click around in.

## Scope

This is research code, not a finished robot and not a safety system.

## License

Apache-2.0 (see `LICENSE`). If you use this, please cite `CITATION.cff` and the MW theory paper.
