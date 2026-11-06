"""pm_launch.py - run ProtoMotions' own entry points on a single Windows machine.

Their runner builds a Lightning Fabric with a DDP strategy even for one process, and torch's gloo backend cannot create a device
from this machine's hostname on Windows. We swap the strategy for Fabric's single-device default ("auto") BEFORE their code builds it;
nothing in the ProtoMotions tree is modified.

    python pm_launch.py inference --simulator mujoco --headless --num-envs 1 --checkpoint ... --motion-file ...
"""
import sys
import protomotions.utils.fabric_config as fc

_orig = fc.FabricConfig.as_kwargs


def _single_device(self):
    kw = _orig(self)
    kw["strategy"] = "auto"
    return kw


fc.FabricConfig.as_kwargs = _single_device

if __name__ == "__main__":
    which = sys.argv.pop(1)
    if which == "inference":
        from protomotions.inference_agent import main
    else:
        raise SystemExit(f"unknown entry point {which!r}")
    main()
