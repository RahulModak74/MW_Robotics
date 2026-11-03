"""pytest path setup.

The Step 2 experiments live in for_full_replacement/ and import each other as top-level packages
(mw_to_protomotions, mw_g1_racer), exactly as they did before the repo was reorganised.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("for_full_replacement",):
    p = os.path.join(HERE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)
