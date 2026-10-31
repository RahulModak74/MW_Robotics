"""Shared helpers for the examples: banners, tables, optional plots."""

import argparse
import os
import sys
import textwrap
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # run without pip install
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:                                  # plots are optional
    HAS_PLT = False


def args(description: str):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true", help="few epochs, smoke test")
    return p.parse_args()


def banner(title: str, why: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)
    for para in why.strip().split("\n\n"):
        print(textwrap.fill(" ".join(para.split()), 76, initial_indent="  ",
                            subsequent_indent="  "))
        print()


def step(msg: str) -> None:
    print(f"\n-- {msg}")


def table(rows, headers) -> None:
    w = [max(len(str(h)), *(len(f"{r[i]}") for r in rows)) for i, h in enumerate(headers)]
    line = "  " + "  ".join(f"{h:<{w[i]}}" for i, h in enumerate(headers))
    print(line)
    print("  " + "  ".join("-" * x for x in w))
    for r in rows:
        print("  " + "  ".join(f"{str(c):<{w[i]}}" for i, c in enumerate(r)))


def takeaway(text: str) -> None:
    print("\n  TAKEAWAY")
    print(textwrap.fill(" ".join(text.split()), 76, initial_indent="  > ",
                        subsequent_indent="    "))
    print()


def save(fig, name: str) -> None:
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  plot saved -> examples/outputs/{name}")
