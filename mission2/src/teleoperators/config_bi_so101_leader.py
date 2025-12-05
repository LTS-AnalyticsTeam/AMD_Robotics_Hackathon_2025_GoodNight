#!/usr/bin/env python

# Out-of-tree config to register a bimanual SO-101 leader teleoperator with LeRobot.

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_so101_leader")
@dataclass
class BiSO101LeaderConfig(TeleoperatorConfig):
    left_arm_port: str
    right_arm_port: str
    left_arm_use_degrees: bool = False
    right_arm_use_degrees: bool = False
