 Here is exactly what you can and cannot claim.
The 6-DoF walk demo proves the mechanism works. It does not prove scaling. The specific gaps between "works on Walker2d" and "works on G1" are concrete, documented, and each has a known fix. But none are fixed yet.

What you can claim today (honest, defensible)
"CEM over a 96-parameter linear policy, running entirely on CPU with no GPU and no backprop, learns a forward-walking gait on Walker2d (6 joints) in ~25 minutes. The MW manifold runs alongside as a real-time physics validator."

Every word of that is true. It's the promotion claim. It's what the video shows.

What you cannot claim yet
~~"The same pipeline scales to 29 DoF."~~

This is a hypothesis, not a claim. Four specific blockers stand between the current state and that claim.

Blocker 1 — the dimensionality wall (hard blocker)
Current: 96 parameters (6 actions × 15 features + 6 bias). CEM's population 60–80 covers it.

G1: 29 actions. Even a modest feature set of 30 features → ~900 parameters. CEM's population would need to scale roughly linearly with parameter count; at 900, the population needed to cover the search space is not tractable on CPU in any reasonable wall-clock.

The fix: Rank-4 head (~265 params) or waypoint lifting (8–12 waypoints × 29 joints ≈ 240–360 params). Waypoint lifting is the more proven of the two — the Lifting paper reports 3.8× lower mean joint error versus direct action search. Without one of these, the 29-DoF attempt is a foregone conclusion.

Effort: 2–3 days. Status: not started.

Blocker 2 — fitness is wrong for humanoid tasks
Current: survival (Plan A) or forward distance (walk demo). Neither is a tracking fitness.

G1: the task is to track a reference motion (walk cycle, jump, etc.). You need -mean_joint_error(rollout, reference) as the fitness. This is Gap 3 from the roadmap.

The fix: implement tracking_fitness(theta, reference, seeds).

Effort: 2–3 days. Status: not started.

Blocker 3 — no retargeting pipeline
Current: the Walker2d environment has a built-in forward-locomotion task. There's no "reference motion" concept in the current pipeline.

G1: a reference motion is mandatory. That requires retargeting human motion data to the G1's 29-DoF kinematic chain, which is what GMR does.

The fix: integrate GMR (open source, CPU-native, real-time).

Effort: 2–3 days. Status: not started.

Blocker 4 — MW validator is untested on humanoid postures
Current: WalkerPostureVAE was trained on Walker2d postures. Its physics priors are Walker2d's joint limits.

G1: a new detector trained on G1 data, with G1's joint limits and physics priors. Whether the manifold still catches the same fault classes at 29 DoF is an empirical question, not a given. The Plan B failure (joint VAE over transitions) showed that "the same architecture scales" is not a safe assumption.

The fix: train a HumanoidPostureVAE on G1 rollouts, run the same fault-detection tests that stage1_validator_test.py runs on Walker2d.

Effort: 2 days. Status: not started.

The intermediate claim you can earn in two weeks
There is a smaller claim that is genuinely achievable and gives you a real stepping stone:

"The same pipeline scales from 6 DoF (Walker2d) to 12 DoF (legs-only humanoid) in two weeks on CPU."

Why 12 DoF and not 29: a legs-only humanoid (two 6-DoF legs, no arms, fixed torso) has enough DoF to demonstrate the dimensionality scaling without crossing the wall. Rank-4 head works at 12 DoF without waypoint lifting. Tracking fitness works at 12 DoF with a single reference clip. GMR has legs-only retargeting profiles for most humanoid URDFs.

The two-week sprint:

Week	Task	Deliverable
1, days 1–2	Rank-4 head + tracking fitness on Walker2d (prove the mechanism on the current env)	Walker2d tracking walk, MPJPE measurement
1, days 3–5	Legs-only humanoid URDF + GMR retargeting one clip	Reference motion in the pipeline
2, days 1–3	Train CEM on the 12-DoF humanoid, rank-4 head, tracking fitness	Held-out MPJPE, wall-clock
2, days 4–5	Port the walker demo to 12-DoF humanoid (qpos animation + MW gauge)	Video demo
Probability of success: 65–75%. The rank-4 head and tracking fitness have both been validated in the literature. GMR is mature. The 12-DoF action space is small enough that the dimensionality wall does not bite.

If the sprint succeeds, the claim becomes:

"CEM over a rank-4 policy on CPU tracks a retargeted reference motion on a 12-DoF humanoid in under 60 minutes, with MW validation running alongside. Same pipeline as the 6-DoF demo; only the joint count changed."

That's a real, defensible claim. It's the "we can scale" claim.

The 29-DoF claim requires a second sprint
After the 12-DoF success:

Week	Task	Risk
3–4	Add waypoint lifting for 29 DoF; retrain on G1	High — the dimensionality wall at 29 DoF is genuinely open
5	Full G1 demo: retarget, track, deploy	Medium — depends on week 3–4 success
Probability: 30–40%. Not high. But the question is empirical and the answer is worth having — even a failed 29-DoF run with a documented failure mechanism is a paper.

The honest percentage table
Claim	Probability	When
6-DoF walk, CPU, MW validator	100%	Done
12-DoF humanoid walk, CPU, rank-4 + tracking	65–75%	2 weeks
29-DoF humanoid walk, CPU, waypoint lifting	30–40%	4–6 weeks
29-DoF tracking a reference motion	20–30%	6–8 weeks
Full g1-racer parity	10–15%	3+ months
What to do before making any scaling claim
Three things, in order, all small:

Fix the MW gauge in the demo (20 min, the four additions from last message). This makes the claim "MW is part of the pipeline" true, not aspirational. Without it, the honest claim is "CEM works, MW exists separately."

Add forward-velocity features to the state (30 min, Gap 3 from the roadmap). Currently the walk policy walks because data.qpos[0] is read externally; the policy itself doesn't see velocity. Adding rootx_dot to STATE_FEATURES makes the policy velocity-aware, which is required before tracking works.

Run the rank-4 head experiment on Walker2d (2–3 days). Prove that a = U @ (V @ φ(s)) + b with V ∈ R^{4×F}, U ∈ R^{6×4} produces a comparable walk to the current 96-param linear policy. If rank-4 matches linear on 6 DoF, it's the same mechanism that will scale to 29 DoF. If rank-4 fails on 6 DoF, the scaling hypothesis is already broken and no amount of G1 work will fix it.

Step 3 is the load-bearing experiment. It's a 2-day task that tells you whether the scaling claim is even worth pursuing. Everything else is downstream of it.

The framing to keep
Your instinct — "math is our only option" — is right, and the 6-DoF result is real. But the honest claim right now is "the mechanism works on 6 DoF", not "the mechanism scales." The scaling claim is earned by the rank-4 experiment on Walker2d (2 days), then the 12-DoF sprint (2 weeks), then the 29-DoF sprint (4–6 weeks). Each step is a real milestone that either passes and earns the claim, or fails and gives you a publishable finding.

Don't skip ahead. The Plan B failure on 6 DoF already showed what happens when you assume scaling works without testing it. The rank-4 test on Walker2d is the smallest, cheapest thing that tells you whether the assumption holds this time. Run it before making any claim about 29 DoF.
