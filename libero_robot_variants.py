"""Register LIBERO robot variants used by the Sawyer Oracle pipeline (Panda default + Sawyer)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.models.robots.robot_model import register_robot
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_XEMB = Path(os.environ.get("OPENPI_XEMB_ROOT", str(Path.home() / "openpi_ws" / "openpi-xemb")))
# Prefer vendored Sawyer assets in this repo; fall back to OPENPI_XEMB_ROOT.
_LOCAL_SAWYER = _REPO_ROOT / "assets/robots/sawyer/robot.xml"
SAWYER_XML = _LOCAL_SAWYER if _LOCAL_SAWYER.is_file() else (_DEFAULT_XEMB / "assets/robots/sawyer/robot.xml")

ROBOT_VARIANT_CHOICES = ("default", "sawyer")


def sawyer_table_base_z() -> float:
    """Z offset for Sawyer mount on LIBERO table (negative = lower robot vs table)."""
    return float(os.environ.get("SAWYER_TABLE_BASE_Z", "0"))


def _table_offsets(base_z: float) -> dict:
    return {
        "bins": (-0.5, -0.1, base_z),
        "empty": (-0.6, 0, base_z),
        "table": lambda table_length, z=base_z: (-0.16 - table_length / 2, 0, z),
        "study_table": lambda table_length, z=base_z: (-0.25 - table_length / 2, 0, z),
        "kitchen_table": lambda table_length, z=base_z: (-0.16 - table_length / 2, 0, z),
    }


class _MountedSawyerBase(ManipulatorModel):
    def __init__(self, idn=0):
        super().__init__(str(SAWYER_XML), idn=idn)

    @property
    def default_gripper(self):
        return "RethinkGripper"

    @property
    def default_controller_config(self):
        return "default_sawyer"

    @property
    def init_qpos(self):
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, -1.57])

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


class MountedSawyerCustom(_MountedSawyerBase):
    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def base_xpos_offset(self):
        return _table_offsets(sawyer_table_base_z())


class OnTheGroundSawyerCustom(_MountedSawyerBase):
    @property
    def default_mount(self):
        return None

    @property
    def base_xpos_offset(self):
        base_z = sawyer_table_base_z()
        return {
            "bins": (-0.5, -0.1, base_z),
            "empty": (-0.6, 0, base_z),
            "table": lambda table_length, z=base_z: (-0.16 - table_length / 2, 0, z),
            "coffee_table": lambda table_length, z=base_z: (-0.16 - table_length / 2, 0, 0.41 + z),
            "living_room_table": lambda table_length, z=base_z: (-0.16 - table_length / 2, 0, 0.42 + z),
        }


_ALL_VARIANT_ROBOTS = (
    MountedSawyerCustom,
    OnTheGroundSawyerCustom,
)


def register_robot_variants() -> None:
    for cls in _ALL_VARIANT_ROBOTS:
        register_robot(cls)
    ROBOT_CLASS_MAPPING.update(
        {cls.__name__: SingleArm for cls in _ALL_VARIANT_ROBOTS}
    )


def robots_arg_for_variant(variant: str) -> list[str]:
    if variant in ("default", "a"):
        return ["Panda"]
    if variant == "sawyer":
        return ["SawyerCustom"]
    raise ValueError(f"Unknown robot variant: {variant!r}. Expected one of {ROBOT_VARIANT_CHOICES}")
