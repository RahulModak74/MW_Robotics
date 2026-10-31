# MW Robotics — interactive demo

An interactive Flask + D3 demo: one composite robot, driven by the six trained MW manifolds
in `../mw_framework/domains/`, click-controlled.

## Run it
```bash
cd full_flow_animation
pip install -r requirements.txt          # just flask - everything else is the parent repo's
python train_and_cache.py                # trains and caches all six models, ~a few minutes
python app.py
# open http://127.0.0.1:5050
```

## What each action is driven by

| Scene element | Domain | Trained on |
|---|---|---|
| Driving / turning the base | `mobile_robot` | nonholonomic kinematics ODE |
| Reaching with the arm | `planar_arm` | forward kinematics + Jacobian identity |
| The standalone joint dial | `joint_actuator` | damped 2nd-order ODE (Newton's second law) |
| The hop | `legged_contact` | ballistic motion + restitution |
| The lift / flex | `flexible_robot` | Euler-Bernoulli beam PDE |
| The thermal gauge | `actuator_thermal` | 1D heat-diffusion PDE |

## What is real here, precisely

- **Every animated motion is decoded from a trained manifold**, using the exact same
  straight-line latent-interpolation mechanism measured and reported in
  `../mujoco_experiments/02_forecaster_check.py` (train once, encode a start state, encode a
  target state, interpolate in latent space, decode each frame). See `robot_motion.py`.
- **Both the start and target states fed to each manifold are physically consistent**, computed
  from the same closed-form formulas each domain's training data generator uses
  (`mw_framework.datasets`' `_joint_response`, `_arm_fk`, `_unicycle_state`, `_beam_state`,
  `_thermal_state`). Nothing invents an arbitrary target; every click is converted into a
  real, in-distribution-shaped state before the manifold is asked to interpolate to it.
- **The accuracy numbers already measured for this mechanism apply here too**: roughly 1–14%
  of the state's range depending on hop size (see `../FORECASTER.md`). This demo does not
  re-measure accuracy; it reuses the same interpolation approach the measurements are for.

## What is not claimed

- **This is not a unified 3D physics system.** The scene composes six separate, mostly 1D/2D
  trained domains into one visual robot. `mobile_robot` and `planar_arm` share a genuine 2D
  floor plane (shown top-down); `legged_contact` and `flexible_robot` are 1D-vertical (shown
  side-on, in a separate strip, not merged into the floor view — merging a top-down and a
  side-on projection into one scene would misrepresent the physics). There is no single
  equation here governing a robot moving through true 3D space. That remains future work,
  separate from this demo — see `../ROBOTICS.md`'s "Where the next iteration should focus".
- **This is not a validated controller.** It is a visualisation of what six trained manifolds
  produce when asked to interpolate between two plausible states, for demonstration. Do not
  use it, or anything derived from it, to control a real robot. Same scope note as the rest
  of this repo: not a safety system.
- **The arm is mounted on the moving base in the rendering only.** `planar_arm`'s own training
  data has no base-motion coupling; the frontend applies a rotation/translation to the arm's
  decoded local coordinates to draw it attached to the base. The two domains were not trained
  together and do not know about each other.

## Files
```
train_and_cache.py   trains all six models once, saves checkpoints/*.pth
robot_motion.py       compute_path() + one wrapper per behaviour (shared backend logic)
app.py                 Flask routes, one per behaviour, plus a small server-side robot state
static/index.html, app.js, style.css   the D3 frontend
checkpoints/            cached trained models (git-ignored; regenerate with train_and_cache.py)
```
