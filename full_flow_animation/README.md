# MW Robotics — interactive demo

An interactive Flask + D3 demo: **two** humanoid figures, side view, each driven by the same six
trained MW manifolds in `../mw_framework/domains/`. Click-control one directly (unit A). The
choreography text field (`POST /api/choreograph`) is wired up but the agentic layer behind it
is not part of this repo - without it the route returns a clear 501 and every manual control
still works unchanged. See "Two robots, and the agentic choreography endpoint" below.

## Run it
```bash
cd full_flow_animation
pip install -r requirements.txt          # just flask - everything else is the parent repo's
python train_and_cache.py                # trains and caches all six models, ~a few minutes
python app.py
# open http://127.0.0.1:5050
```

## What each action is driven by

| Figure element | Domain | Trained on |
|---|---|---|
| Walking left/right | `mobile_robot` | nonholonomic kinematics ODE (x only - see "What is not claimed") |
| The reaching arm | `planar_arm` | forward kinematics + Jacobian identity |
| The scanning head | `joint_actuator` | damped 2nd-order ODE (Newton's second law) |
| The jump | `legged_contact` | ballistic motion + restitution |
| The lifting arm / payload | `flexible_robot` | Euler-Bernoulli beam PDE |
| The outline glow | `actuator_thermal` | 1D heat-diffusion PDE |

## Two robots, and the agentic choreography endpoint

`app.py` keeps independent server-side state for robots `"a"` and `"b"`; every route accepts an
optional `"robot"` field (default `"a"`, so the manual single-robot controls always work).
`POST /api/choreograph {"task": "..."}` tries to import a sibling `mw_agentic` package (an
agentic layer: hand-written recipes tried first with zero model calls, a model - `claude` CLI or
a raw `ANTHROPIC_API_KEY` - called only for a novel request, then learned for next time) and, if
present, calls its `orchestrate.choreograph()`. **That package is not part of this public repo.**
Without it the route returns `501` with a clear error, and the frontend's choreography field
just logs `CHOREOGRAPHY FAILED: ...` - no crash, every manual control keeps working.

## What is real here, precisely

- **Every animated motion is decoded from a trained manifold**, using the exact same
  straight-line latent-interpolation mechanism measured and reported in
  `../mujoco_experiments/02_forecaster_check.py` (train once, encode a start state, encode a
  target state, interpolate in latent space, decode each frame). See `robot_motion.py`.
- **Both the start and target states fed to each manifold are physically consistent**, computed
  from the same closed-form formulas each domain's training data generator uses
  (`mw_framework.datasets`' `_joint_response`, `_arm_fk`, `_unicycle_state`, `_beam_state`,
  `_thermal_state`). Nothing invents an arbitrary target; every click or choreography action is
  converted into a real, in-distribution-shaped state before the manifold interpolates to it.
- **The accuracy numbers already measured for this mechanism apply here too**: roughly 1–14%
  of the state's range depending on hop size (see `../FORECASTER.md`). This demo does not
  re-measure accuracy; it reuses the same interpolation approach the measurements are for.
- **When the agentic layer is present**, its choreography validator's parameter ranges match the
  trained domains exactly - a request outside what a manifold was actually shown is rejected
  before it ever reaches one, whether it came from a hand-written recipe or a model.

## What is not claimed

- **This is not a unified 3D physics system.** Each figure composes six separate, mostly 1D
  trained domains into one visual humanoid, side view. Walking is `mobile_robot`'s x-translation
  only - its y output and true 2D floor navigation are not visualised, a deliberate
  simplification for a readable side-on figure, not a new physics claim. There is no single
  equation here governing a robot moving through true 3D space - see `../ROBOTICS.md`'s "Where
  the next iteration should focus".
- **This is not a validated controller.** It is a visualisation of what six trained manifolds
  produce when asked to interpolate between two plausible states, for demonstration. Do not
  use it, or anything derived from it, to control a real robot. Same scope note as the rest
  of this repo: not a safety system.
- **The arm is mounted on the figure in the rendering only.** `planar_arm`'s own training data
  has no body coupling; the frontend positions the decoded local coordinates at the shoulder.
  The domains were not trained together and do not know about each other.
- **The agentic choreography layer is not shipped in this repo.** `/api/choreograph` is a real,
  wired-up endpoint - it is just not backed by anything here.

## Files
```
train_and_cache.py   trains all six models once, saves checkpoints/*.pth
robot_motion.py       compute_path() + one wrapper per behaviour (shared backend logic)
app.py                 Flask routes (per-behaviour + /api/choreograph), two named robot states
static/index.html, app.js, style.css   the D3 frontend - two figures, a choreography input,
                                        a Terminator-style HUD in Bayesian Cybersecurity's brand blue
checkpoints/            cached trained models (git-ignored; regenerate with train_and_cache.py)
```
