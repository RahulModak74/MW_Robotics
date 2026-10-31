"""
Domain models: MWVAE subclasses whose physics() encodes a field's laws.

Each file is a template. Copy one, change FEATURES and physics(), done.
"""

from .flexible_robot import FlexibleRobotVAE
from .joint_actuator import JointActuatorVAE
from .planar_arm import PlanarArmVAE
from .mobile_robot import MobileRobotVAE
from .legged_contact import HoppingContactVAE
from .actuator_thermal import ActuatorThermalVAE

__all__ = ["FlexibleRobotVAE", "JointActuatorVAE", "PlanarArmVAE", "MobileRobotVAE",
          "HoppingContactVAE", "ActuatorThermalVAE"]
