#!/usr/bin/env python3
"""aggregate_retry.py - tables for Retry_MW_distillation.md from eval_prob/ (logs + per-step npz).

    python3 aggregate_retry.py --eval ../eval_prob --data ../data_pm --train_motions walk,wave,high5,jumpjack --out results/retry/SUMMARY.md

Tags must look like <arm>__<motion>__n<noise>__s<seed>[__gate<...>] (see make_jobs in run_retry.sh).
E1  evaluator: success, anchor height error, max joint error, global tracking error, action delta.
E2  calibration: share of teacher actions inside the 50 / 90 % predictive interval (sigma^2 = ale^2 + epi^2), per joint, pooled.
E3  failure prediction: per step, positive = within H steps before the evaluator's failure; negative = steps of runs that did not
    fail, or more than 2H before a failure. AUROC of each signal; Mahalanobis of the raw observation (fit on training motions) as baseline.
"""
import os, re, glob, json, argparse
import numpy as np

KEYS = ["success_rate", "anchor_height_error/mean", "max_joint_error/mean", "gt_error/mean", "action_delta_mean_rad"]


def parse_log(p):
    out = {}
    for line in open(p, errors="ignore"):
        m = re.match(r"\s*eval/(\S+): ([-\d.eE+naif]+)", line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_maha(data_dir, motions, max_n=60000):
    X = []
    for m in motions:
        for f in sorted(glob.glob(os.path.join(data_dir, f"{m}_s*.npz"))):
            X.append(np.load(f)["X"])
    X = np.concatenate(X)
    if len(X) > max_n:
        X = X[np.random.RandomState(0).choice(len(X), max_n, replace=False)]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    C = np.cov(Z, rowvar=False); C = 0.9 * C + 0.1 * np.eye(len(C))          # shrinkage
    return mu, sd, np.linalg.inv(C)


def maha(obs, mu, sd, P):
    Z = (obs - mu) / sd
    return np.einsum("ij,jk,ik->i", Z, P, Z) / Z.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval_prob"); ap.add_argument("--data", default="data_pm")
    ap.add_argument("--train_motions", default="walk,wave,high5,jumpjack"); ap.add_argument("--out", default="SUMMARY.md")
    ap.add_argument("--H", type=int, default=25, help="steps before failure counted as positive (25 = 0.5 s at 50 Hz)")
    a = ap.parse_args()
    train = a.train_motions.split(",")
    mu, sd, P = fit_maha(a.data, train)
    rows, e3 = {}, {}
    for log in sorted(glob.glob(os.path.join(a.eval, "*.log"))):
        tag = os.path.basename(log)[:-4]
        parts = tag.split("__")
        if len(parts) < 4:
            continue
        arm, motion, nz, seed = parts[0], parts[1], parts[2][1:], parts[3][1:]
        gate = parts[4] if len(parts) > 4 else ""
        r = parse_log(log)
        if "success_rate" not in r:
            continue
        split = motion if motion.startswith(("train", "heldout")) else ("seen" if motion in train else "held-out")
        key = (arm + (f" [{gate}]" if gate else ""), split, nz)
        rec = {k: r.get(k, np.nan) for k in KEYS}
        npz = log[:-4] + ".npz"
        if os.path.exists(npz):
            d = np.load(npz, allow_pickle=True)
            n = len(d["epi"])
            if n:
                rec["handed"] = float(np.mean(d["handed"]))
                rec["n_steps"] = n
                S = np.sqrt(d["sig_ale_j"] ** 2 + d["sig_epi_j"] ** 2) + 1e-9
                zabs = np.abs(d["a_teacher"] - d["a_student"]) / S
                rec["cov50"], rec["cov90"] = float((zabs <= 0.674).mean()), float((zabs <= 1.645).mean())
                fs = int(d["fail_step"])
                sigs = {"epi": d["epi"], "mw": d["mw"], "ale": d["ale"], "law_rate": d["law_rate"],
                        "maha_obs": maha(d["obs"], mu, sd, P)}
                t = np.arange(n)
                if "episode" in d.files and len(d["episode"]) == n:
                    # many motions per run: positives = H steps before each episode's failure; negatives = all steps of
                    # episodes that did not fail (true negatives) + steps more than 2H before a failure
                    ep = d["episode"]; pos = np.zeros(n, bool); neg = np.ones(n, bool)
                    for e, f in (d["ep_fail"].reshape(-1, 2) if d["ep_fail"].size else []):
                        in_e = ep == e
                        pos |= in_e & (t >= f - a.H) & (t <= f)
                        neg &= ~(in_e & (t >= f - 2 * a.H))
                    rec["episodes"] = int(len(np.unique(ep))); rec["failed_eps"] = int(d["ep_fail"].reshape(-1, 2).shape[0])
                elif fs >= 0:
                    pos = (t >= fs - a.H) & (t <= fs); neg = t < fs - 2 * a.H
                else:
                    pos = np.zeros(n, bool); neg = np.ones(n, bool)
                if not gate:
                    for s, v in sigs.items():
                        e3.setdefault((arm, split, s), {"pos": [], "neg": []})
                        e3[(arm, split, s)]["pos"].append(v[pos]); e3[(arm, split, s)]["neg"].append(v[neg])
                rec["act_err_vs_teacher"] = float(np.sqrt(((d["a_student"] - d["a_teacher"]) ** 2).mean()))
        rows.setdefault(key, []).append(rec)

    L = ["# Retry_MW_distillation - results", "",
         "Generated by `scripts/aggregate_retry.py`. MuJoCo simulation, ProtoMotions' own evaluator. "
         f"Training motions: {', '.join(train)}. Values are means over runs (n = runs).", "",
         "## E1 - evaluator scores", "",
         "| arm | split | noise | n | success | anchor height err (m) | max joint err | gt err (m) | action delta (rad) | RMS action vs teacher | handed to teacher | cover 50 % | cover 90 % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (arm, split, nz), recs in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        m = lambda k: np.nanmean([r.get(k, np.nan) for r in recs])
        L.append(f"| {arm} | {split} | {nz} | {len(recs)} | {m('success_rate'):.2f} | {m('anchor_height_error/mean'):.3f} | "
                 f"{m('max_joint_error/mean'):.3f} | {m('gt_error/mean'):.3f} | {m('action_delta_mean_rad'):.3f} | "
                 f"{m('act_err_vs_teacher'):.3f} | {m('handed'):.2f} | {m('cov50'):.2f} | {m('cov90'):.2f} |")
    L += ["", f"## E3 - failure prediction (AUROC; positive = within {a.H} steps before the evaluator's failure)", "",
          "| arm | split | signal | AUROC | positive steps | negative steps |", "|---|---|---|---|---|---|"]
    for (arm, split, s), v in sorted(e3.items()):
        pos = np.concatenate(v["pos"]) if v["pos"] else np.array([]); neg = np.concatenate(v["neg"]) if v["neg"] else np.array([])
        if len(pos) == 0:
            continue
        L.append(f"| {arm} | {split} | {s} | {auroc(pos, neg):.3f} | {len(pos)} | {len(neg)} |")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
