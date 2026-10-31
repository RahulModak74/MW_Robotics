# MW Robotics — interactive demo

A Flask + D3 demo you can click around in: **two** humanoid figures, side view, each driven
by the same six trained MW models in `../mw_framework/domains/`. Click-control one directly
(unit A). There's also a text box for typing a task in plain words — it's wired up to an
endpoint (`/api/choreograph`), but the layer that actually understands the text isn't part
of this public repo (more on that below).

## Run it
```bash
cd full_flow_animation
pip install -r requirements.txt          # just flask - everything else is the parent repo's
python train_and_cache.py                # trains and saves all six models, takes a few minutes
python app.py
# open http://127.0.0.1:5050
```

## What drives each part of the figure

| What you see | Trained model | What it learned |
|---|---|---|
| Walking left/right | `mobile_robot` | how a wheeled robot steers (see "What this isn't" below) |
| The reaching arm | `planar_arm` | the geometry of a 2-joint arm |
| The scanning head | `joint_actuator` | how a motor-driven joint settles |
| The jump | `legged_contact` | falling and bouncing |
| The lifting arm / payload | `flexible_robot` | how a bendy beam flexes |
| The outline glow | `actuator_thermal` | how heat spreads and cools |

## Two robots, and the choreography endpoint

`app.py` keeps separate state for robot `"a"` and robot `"b"`; every route takes an optional
`"robot"` field (it defaults to `"a"`, so the original single-robot controls still work
exactly as before). `POST /api/choreograph {"task": "..."}` looks for a sibling package that
turns plain-English text into a sequence of actions for both robots. **That package isn't
part of this public repo** — without it, the endpoint just returns a clear error (HTTP 501),
and the text box quietly tells you choreography isn't available. Every manual button and
slider still works fine either way.

## What's actually real here

- **Every motion you see is decoded from a trained model** — not hand-animated. The same
  method used and measured in `../mujoco_experiments/02_forecaster_check.py`: train once,
  encode a start state, encode a target state, draw a straight line between them in the
  model's internal space, and decode each point along that line into a frame. See
  `robot_motion.py` for the code.
- **Both the start and target states are physically real**, computed from the same exact
  formulas used to generate each model's training data. Nothing made up — every click or
  action is turned into a real, physically consistent state before the model draws a path to
  it.
- **The accuracy already measured for this method applies here too**: predictions are off by
  roughly 1-14% of the normal range, depending on how big a jump is being asked for (see
  `../FORECASTER.md`). This demo doesn't re-measure accuracy — it just reuses the same method
  the measurements were taken on.

## What this isn't

- **This is not one unified 3D physics simulation.** Each figure is six separate, mostly 1D
  trained models stitched together into one visual humanoid, seen from the side. "Walking" is
  really just the wheeled-robot model's left-right motion — there's no single equation here
  governing a robot moving through real 3D space. See `../ROBOTICS.md` for where that could go
  next.
- **This is not a tested robot controller.** It's a visualization of what six trained models
  produce when asked to move between two plausible states, for demonstration purposes. Don't
  use it, or anything built on it, to control a real robot.
- **The arm's position on the figure is just drawing, not physics.** The arm model was never
  trained with a body attached to it — the frontend just places its output at the shoulder for
  looks. None of the six models know about each other.

## Files
```
train_and_cache.py   trains all six models once, saves them to checkpoints/*.pth
robot_motion.py       the shared motion logic: builds a path between two states for each behaviour
app.py                 the Flask server - one route per behaviour, plus /api/choreograph, two robot states
static/index.html, app.js, style.css   the D3 frontend - two figures, a text box, a dark HUD in
                                        Bayesian Cybersecurity's brand blue
checkpoints/            the saved trained models (not stored in git - run train_and_cache.py to regenerate)
```
