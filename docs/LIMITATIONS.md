# Limitations — read before deploying

These limits carry over from the general [MW Framework](https://github.com/RahulModak74/MW_Geometric_Intelligence)
this repo is trimmed from. Each one was found while building that repo's seven examples
(numbered 00, 01, 05, 06 below — those live in the parent repo, not here). This repo's own
results, with their own mixed findings, are in [ROBOTICS.md](../ROBOTICS.md).

**1. Physics rules don't always beat a plain model.**
With clean, plentiful data and simple physics, a plain model does just as well (examples 01
and 06). Adding physics rules pays off most with messy or sparse data, when you need to learn
an unknown constant, or for rules the data rarely tests.

**2. Rules that are too strict can backfire.**
If a physics rule is tighter than the model's own noise, it forces the model to choose between
fitting the data and obeying the rule. Either way, detection gets worse. Start with loose
rules and tighten them gradually while watching how well it catches faults.

**3. The model's uncertainty estimates are too confident.**
In the battery example, the model gets a physical constant right to within a few percent, but
says it's more certain than it actually is. Treat its uncertainty numbers as a floor, not the
real answer — check important constants against a real lab test.

**4. How much "room" the model has matters.**
Too little room, and it can't learn the shape it needs to (example 00). Too much, and it
starts inventing states in places with no real data behind them (example 05). Use
`path_support()` to check, and aim for about as much room as the system actually needs.

**5. MW catches a state that's clearly off, not a subtle shift between two normal-looking states.**
A fault that moves the system from one "normal-looking" state to another won't show up in a
single snapshot. For that, you need to look at how the state changes over time (the
Lorentzian cone), or add a specific rule for the exact bad region you're worried about.

**6. There's a floor to how small a fault it can notice.**
In one example, it misses a small enough simulation error. Scores average across every
feature, so a problem in just one feature can get lost in the noise. If one specific rule
really matters, watch that rule's own error directly too, not just the overall score.

**7. The training data defines what counts as "normal."**
A situation the model never saw during training (a new season, new equipment) will look like
a false alarm. You'll need to retrain or add more data. Calibration assumes future normal data
looks like the normal data it was trained on.

**8. The speed gain is about checking, not solving.**
Checking one state costs one quick pass through the model. Training still takes real data,
time, and tuning. The comparison to a physics solver is about *checking or predicting a
state*, not about solving the underlying equations to the same precision a solver would.

**9. All the data here is simulated, not real-world.**
Every example uses data generated from known physics, where the right answer is known for
sure. Real-world results will likely be noisier and a bit worse. Treat these numbers as
proof the method works, not as a guaranteed real-world benchmark.

**10. This is not a safety system.**
It's research software. Don't rely on it as the only thing protecting equipment or people.
