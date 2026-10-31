"""
mw_framework.cli
================
Command line: train on a CSV with a JSON config, score another CSV.
No JSON config ships with this robotics-only trim - write your own (see
mw_framework/config.py for the schema) or train directly against a VAE
object instead (see examples/00_flexible_robot_validator.py).

    python -m mw_framework.cli demo-data --out data/
    python -m mw_framework.cli score --model robot.pth --data data/robot_test.csv \
                                     --out scored.csv --label-col anomaly_type
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def _train(a):
    from .detector import MWDetector
    df = pd.read_csv(a.data)
    if a.normal_col:
        df = df[df[a.normal_col].astype(str) == a.normal_value]
        print(f"normal-only filter: {len(df)} rows where {a.normal_col} == {a.normal_value}")
    det = MWDetector(config=a.config, epochs=a.epochs, verbose=True).fit(df)
    det.save(a.out)
    c = det.calibration_
    print(f"\nsaved {a.out}\ncalibration: threshold={c.threshold:.4f} "
          f"(p{c.percentile:.0f} of held-out normal), sigma={c.sigma:.4f}")


def _score(a):
    from .detector import MWDetector
    det = MWDetector.load(a.model)
    df = pd.read_csv(a.data)
    out = df.copy()
    out["mw_distance"] = det.mw_distance(df)
    out["uncertainty"] = det.uncertainty(df)
    out["risk"] = det.risk_score(df).round(1)
    out["status"] = det.status(df)
    attr = det.feature_attribution(df)
    out["top_driver"] = attr.idxmax(axis=1).values

    n = len(out)
    counts = out["status"].value_counts()
    print(f"\nscored {n} rows with domain '{det.domain_}'")
    for s in ["NORMAL", "MONITOR", "THREAT"]:
        k = int(counts.get(s, 0))
        print(f"  {s:<8} {k:>6}  ({k / n:.1%})")
    if a.label_col and a.label_col in out.columns:
        print(f"\nflag rate (risk >= 50) by '{a.label_col}':")
        g = out.assign(flag=out["risk"] >= 50).groupby(a.label_col)["flag"].mean()
        for k, v in g.items():
            print(f"  {str(k):<16} {v:.1%}")
    if a.out:
        out.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")
    if a.json:
        print(json.dumps(out[["risk", "status", "top_driver"]].head(a.json).to_dict("records"), indent=2))


def _demo(a):
    from .datasets import cantilever_beam
    os.makedirs(a.out, exist_ok=True)
    normal, anom = cantilever_beam(n_normal=4000, seed=0)
    normal.iloc[:3000].to_csv(os.path.join(a.out, "robot_train.csv"), index=False)
    test = pd.concat([normal.iloc[3000:].assign(anomaly_type="NORMAL"), anom])
    test.sample(frac=1, random_state=0).to_csv(os.path.join(a.out, "robot_test.csv"), index=False)
    print(f"wrote {a.out}/robot_train.csv (normal) and {a.out}/robot_test.csv (mixed, labelled)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="mw", description="Modak-Walawalkar Framework CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a detector from a JSON config + CSV")
    t.add_argument("--config", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--out", default="mw_model.pth")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--normal-col", default=None, help="keep only rows where this column ...")
    t.add_argument("--normal-value", default="NORMAL", help="... equals this value")
    t.set_defaults(fn=_train)

    s = sub.add_parser("score", help="score a CSV with a saved detector")
    s.add_argument("--model", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--label-col", default=None)
    s.add_argument("--json", type=int, default=0, help="print first N rows as JSON")
    s.set_defaults(fn=_score)

    d = sub.add_parser("demo-data", help="write synthetic RF train/test CSVs")
    d.add_argument("--out", default="data")
    d.set_defaults(fn=_demo)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
