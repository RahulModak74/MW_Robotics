"""policy_priors_demo.py - a policy is a function; physical priors cut function space the way a PDE residual does in a PINN.

Inverted pendulum (theta = lean angle, 0 = upright):  theta'' = (g/l) sin(theta) + u, actuator limit |u| <= 25.  A teacher policy balances it.
Training data: the TEACHER'S OWN TRAJECTORIES from small leans (|theta0| < 0.1 rad), as in real distillation - lean and speed move
together (corr ~ -0.88), so the data cannot separate "react to lean" from "react to speed". Test: pushes from upright with angular
speeds the data never contained (0.2 - 2.5 rad/s).

  plain student : fit the teacher's actions on the data (MSE)                       -> one function among many that fit the data
  prior student : same data loss + a BALANCE PRIOR enforced on collocation states   -> PINN-style, no labels needed there
                  prior = "the capture point c = theta + theta_dot / w0 must shrink after the action" (known physics), i.e. a law on
                  the CONSEQUENCE of the action (next state), not on the action itself - the policy-world analogue of a PDE residual.

Measured 2026-11-06 (seeds 0-4; pushes 1.0-4.5 rad/s, all recoverable - the teacher survives every one; collocation |speed| <= 4.5):
  plain student: 18 of 35 push tests FELL (every seed falls from 3.0-3.5 rad/s up); prior student: 0 of 35; teacher: 0 of 7.
  Both fit the teacher's data equally (MSE 0.0000-0.0002): the data cannot tell them apart; only the prior decides what happens off it.
  (An earlier version with collocation |speed| <= 2 and pushes <= 2.5: plain 1 of 15 fell, prior 0 - the prior must cover the test range.)
Mode "exceed" - can a student beat its teacher? (python3 policy_priors_demo.py exceed 0,1,2,3,4 ; env RHO, PRIOR_W)
  Soft teacher (KP=6, KD=3: smooth, slow recovery); success = lean stays within 2 degrees after a push of 0.1-0.4 rad/s.
  The prior asks for FASTER balance recovery than the teacher shows - physics the teacher's data does not contain.
  RHO=0.95, weight 1: exceeded 2 deg - teacher 25/35, plain 25/35, prior 21/35 (prior keeps lean ~17 % lower on every test).
  RHO=0.85, weight 3: exceeded 2 deg - teacher 25/35, plain 25/35, prior  8/35; max lean at the hardest push 4.37 -> 2.2 deg.
  Max lean grows linearly with push (linear regime): teacher ~10.9 deg per rad/s, prior student ~5.5 - the prior halves the slope.
  Caveats: the student deliberately does not copy the teacher here (its action error vs the teacher was not measured); a teacher already
  at the physical limit (the default mode, full torque against large pushes) cannot be beaten this way.
Run: python3 policy_priors_demo.py [seeds, e.g. 0,1,2,3,4]   (CPU, a few minutes; the data itself is fixed by the seed set at import)
"""
import math, os, torch
torch.manual_seed(2)
G_L, DT = 9.81 / 1.0, 0.02
W0 = math.sqrt(G_L)


def step(s, u):                                     # physics (semi-implicit Euler); s = [theta, theta_dot]
    th, om = s[..., 0], s[..., 1]
    om = om + DT * (G_L * torch.sin(th) + u.squeeze(-1).clamp(-25, 25))          # actuator torque limit
    return torch.stack([th + DT * om, om], -1)


KP, KD = 25.0, 10.0                                  # teacher gains (mode "exceed" uses a softer teacher)
RHO = 0.97                                           # prior: capture point must shrink to <= RHO x its value each step


def teacher(s):                                     # cancels gravity, then a PD law
    th, om = s[..., 0], s[..., 1]
    return (-G_L * torch.sin(th) - KP * th - KD * om).unsqueeze(-1)


def capture(s):
    return s[..., 0] + s[..., 1] / W0


def make_policy():
    return torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))


# data: the TEACHER'S OWN TRAJECTORIES from small leans (how distillation data really looks): lean and speed always move together,
# so the data cannot tell "react to lean" from "react to speed" - many functions fit it, and they disagree off the trajectories.
def teacher_trajectories(n=40, T=60):
    out = []
    for th0 in torch.empty(n).uniform_(-0.1, 0.1):
        s = torch.tensor([[float(th0), 0.0]])
        for _ in range(T):
            out.append(s.clone()); s = step(s, teacher(s))
    return torch.cat(out)
S = teacher_trajectories(); U = teacher(S)
print(f"data: {len(S)} states, lean in [{S[:, 0].min():.2f}, {S[:, 0].max():.2f}], speed in [{S[:, 1].min():.2f}, {S[:, 1].max():.2f}], "
      f"corr(lean, speed) = {torch.corrcoef(S.T)[0, 1]:.2f}")


def train(prior_weight):
    pi = make_policy(); opt = torch.optim.Adam(pi.parameters(), 3e-3)
    for it in range(3000):
        loss = ((pi(S) - U) ** 2).mean()                                   # fit the data
        if prior_weight:
            # collocation states: where we have NO teacher labels (wide leans and speeds)
            Sc = torch.stack([torch.empty(512).uniform_(-0.8, 0.8), torch.empty(512).uniform_(-4.5, 4.5)], -1)
            c0, c1 = capture(Sc).abs(), capture(step(Sc, pi(Sc))).abs()
            loss = loss + prior_weight * torch.relu(c1 - RHO * c0).pow(2).mean() / DT ** 2   # "balance must improve"
        opt.zero_grad(); loss.backward(); opt.step()
    return pi


def rollout(pi, th0, om0=0.0, T=300, limit=1.2):
    s = torch.tensor([[th0, om0]])
    with torch.no_grad():
        for t in range(T):
            s = step(s, pi(s))
            if s[0, 0].abs() > limit:
                return f"FELL at {t * DT:.2f} s"
    return f"balanced, final lean {float(s[0, 0]):+.3f} rad"


def max_lean(pi, om0, T=300):
    s = torch.tensor([[0.0, om0]]); m = 0.0
    with torch.no_grad():
        for _ in range(T):
            s = step(s, pi(s)); m = max(m, float(s[0, 0].abs()))
            if m > 1.2:
                break
    return m


def exceed_mode(seeds):
    """Can a student beat its teacher? Soft teacher (KP=6, KD=3: smooth, slow recovery) + a prior demanding faster recovery (RHO=0.95);
    success = lean stays within 2 degrees after a push. Data are re-generated from the soft teacher."""
    global KP, KD, RHO, S, U
    KP, KD, RHO = 6.0, 3.0, float(os.environ.get("RHO", "0.95"))
    torch.manual_seed(123); S = teacher_trajectories(); U = teacher(S)
    lim = math.radians(2.0); pushes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    t_row = "".join("." if max_lean(teacher, p) <= lim else "X" for p in pushes)
    print(f"[exceed] soft teacher KP={KP}, KD={KD}; prior RHO={RHO}, weight {os.environ.get('PRIOR_W', '1.0')}; success = max lean <= 2 deg; pushes {pushes} rad/s")
    print(f"  teacher        {t_row}   (X = exceeded 2 deg)  max lean deg: {[round(math.degrees(max_lean(teacher, p)), 2) for p in pushes]}")
    tot = {"plain": 0, "prior": 0}
    for sd in seeds:
        torch.manual_seed(sd); plain, prior = train(0.0), train(float(os.environ.get("PRIOR_W", "1.0")))
        rows = {n: "".join("." if max_lean(pi, p) <= lim else "X" for p in pushes) for n, pi in (("plain", plain), ("prior", prior))}
        for n in rows: tot[n] += rows[n].count("X")
        print(f"  seed {sd}: plain {rows['plain']}  prior {rows['prior']}   max lean deg prior: {[round(math.degrees(max_lean(prior, p)), 2) for p in pushes]}")
    print(f"  TOTAL over {len(seeds) * len(pushes)} tests: teacher {t_row.count('X') * len(seeds)} | plain {tot['plain']} | prior {tot['prior']} exceeded 2 deg")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "exceed":
        exceed_mode([int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 4]); sys.exit()
    seeds = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2, 3, 4]
    pushes = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    print(f"data: {len(S)} teacher states, |speed| <= {S[:, 1].abs().max():.2f} rad/s, corr(lean, speed) = {torch.corrcoef(S.T)[0, 1]:.2f}")
    feasible = [p for p in pushes if not rollout(teacher, 0.0, p).startswith("FELL")]
    print("teacher survives pushes:", feasible, "| physically unrecoverable (teacher falls too, not counted):", [p for p in pushes if p not in feasible])
    tot = {"plain": 0, "prior": 0}; n = 0
    for sd in seeds:
        torch.manual_seed(sd)
        plain, prior = train(0.0), train(1.0)
        with torch.no_grad():
            fit = (float(((plain(S) - U) ** 2).mean()), float(((prior(S) - U) ** 2).mean()))
        row = {"plain": [], "prior": []}
        for p in feasible:
            for name, pi in (("plain", plain), ("prior", prior)):
                r = rollout(pi, 0.0, p); row[name].append("F" if r.startswith("FELL") else ".")
                tot[name] += r.startswith("FELL")
            n += 1
        print(f"seed {sd}: plain {''.join(row['plain'])}  prior {''.join(row['prior'])}   (F = fell, . = balanced; pushes {feasible}) "
              f"| data fit MSE plain {fit[0]:.4f} prior {fit[1]:.4f}")
    print(f"TOTAL falls over {n} feasible push tests: plain {tot['plain']} | prior {tot['prior']}")
