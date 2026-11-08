# Designing the flight envelope

*How envelope D of the MW fly-by-wire was designed for the Unitree G1, and how to design one for another policy or robot. Code:
[`step1_extraction/flight_computer/fbw.py`](../step1_extraction/flight_computer/fbw.py); settings:
[`run_fbw_D.sh`](../step1_extraction/flight_computer/run_fbw_D.sh).*

The flight computer predicts the pilot's plan 0.4 s ahead with an exact model of the robot and scores the predicted states against
a set of physical laws, the *envelope*. If the plan stays inside, the pilot's command passes unchanged; if not, the smallest
correction that keeps it inside is sent. So the envelope decides everything: too loose and falls get through, too tight and the
flight computer fights a pilot that was fine.

## How envelope D was designed (G1)

The envelope is a set of physical laws on the *predicted* states, each with a tolerance and a weight:

| law | D's setting | why |
|---|---|---|
| predicted fall (pelvis below 0.45 m or torso flatter than 60°) | on, cost 1000 × steps remaining | the hard boundary; earlier falls cost more |
| **capture point** within 5 cm of the support | weight 50 | the point where the robot would have to step to stop. Measured as distance to the segment between the grounded feet, not to the foot centre, so normal dancing and walking do not trigger it |
| torso upright within 50° | weight 50 | keeps the trunk from tipping during fast upper-body motion |
| pelvis height, spin | off | did not add anything here |
| `quiet` (act when the pilot's expected cost exceeds) | 1.0 | about 3 cm or 8° of sustained violation |

It was chosen by ablation on the dance clip, the same student and seeds each time:

| config | laws | dance clip |
|---|---|---|
| A | predicted fall only | 5/7 |
| B | fall + upright | 4/7 |
| **D** | **fall + upright + capture point** | **7/7** |

**The capture point is the law that matters.** On the dance runs, every step where D corrected was triggered by the capture-point law;
upright fired on about 1 % of them, and the fall term almost never. A only reacts once a fall is predicted. D keeps the robot inside
its balance limits before a fall is ever predicted, which is envelope protection in the fly-by-wire sense. D was then held to the
**no-regression gate** on 51 motions it had never been tuned on (21 familiar + 30 dances): 51/51, 0 regressions.

## Designing an envelope for another policy or robot

1. **Model.** Load the robot's MJCF into a private MuJoCo model and check it with `check_prior()`. A one-step error near 1e-7 means
   the prior is exact; a real robot needs a nominal model plus online residual learning (not done yet).
2. **Pilot in the loop.** Write `pilot(X) -> joint targets` for a batch of predicted observations, and rebuild the policy's
   observation from a predicted state (`proprio()` and `next_obs()` here). This is the main integration work for a new policy.
3. **Laws from the platform's physics.** Legged: capture point vs support, torso tilt, a fall test. Arms: joint and torque limits,
   end-effector speed, collisions. Start with the fall test only (config A), then add one law at a time.
4. **Tune on one clip, accept on held-out motions.** Run an ablation on a hard clip with action noise. Then run the best config on
   held-out motions and require zero regressions against the policy alone, plus the paired tracking cost (`fbw_tracking.py`). If the
   tracking cost is too high, raise `quiet` so it acts only near the envelope boundary.
5. **Watch the takeover rate.** If the flight computer steers on most steps of a run, the pilot is the bottleneck: improve the
   pilot (more data), not the shield.

## Other bodies (planned, not yet built)

| body | main law | other laws |
|---|---|---|
| humanoid (G1, done) | capture point within the support | torso tilt, predicted fall |
| four legs + arms | static stability margin: centre of gravity inside the feet polygon, including arm loads | tip-over moment when reaching, leg-lift order, contact force of a tool on the floor |
| wheeled disc | tipping at edges and steps | wheel slip, collisions, contact force |
