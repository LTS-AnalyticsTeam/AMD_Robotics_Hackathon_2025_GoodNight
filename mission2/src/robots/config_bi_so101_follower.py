#!/usr/bin/env python

# Out-of-tree config to register a bimanual SO-101 follower robot with LeRobot.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("bi_so101_follower")
@dataclass
class BiSO101FollowerConfig(RobotConfig):
    left_arm_port: str
    right_arm_port: str

    left_arm_disable_torque_on_disconnect: bool = True
    left_arm_max_relative_target: float | dict[str, float] | None = None
    left_arm_use_degrees: bool = False
    right_arm_disable_torque_on_disconnect: bool = True
    right_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_use_degrees: bool = False

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
