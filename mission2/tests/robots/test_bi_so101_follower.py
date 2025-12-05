#!/usr/bin/env python

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

# Make external extensions available (src/robots).
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from robots.bi_so101_follower import BiSO101Follower  # noqa: E402
from robots.config_bi_so101_follower import BiSO101FollowerConfig  # noqa: E402
from lerobot.robots.so101_follower import SO101Follower  # noqa: E402


def _make_bus_mock() -> MagicMock:
    """Return a bus mock with just the attributes used by the robot."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm

    return bus


@pytest.fixture
def bi_follower():
    bus_mocks: list[MagicMock] = []

    def _bus_side_effect(*_args, **kwargs):
        bus_mock = _make_bus_mock()
        bus_mock.motors = kwargs["motors"]
        motors_order: list[str] = list(bus_mock.motors)

        bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus_mock.sync_write.return_value = None
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.enable_torque.return_value = None
        bus_mock.is_calibrated = True
        bus_mocks.append(bus_mock)
        return bus_mock

    with (
        patch(
            "lerobot.robots.so101_follower.so101_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO101Follower, "configure", lambda self: None),
    ):
        cfg = BiSO101FollowerConfig(left_arm_port="/dev/null_left", right_arm_port="/dev/null_right", cameras={})
        robot = BiSO101Follower(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(bi_follower):
    assert not bi_follower.is_connected

    bi_follower.connect()
    assert bi_follower.is_connected

    bi_follower.disconnect()
    assert not bi_follower.is_connected


def test_get_observation(bi_follower):
    bi_follower.connect()
    obs = bi_follower.get_observation()

    expected_keys = {f"left_{m}.pos" for m in bi_follower.left_arm.bus.motors} | {
        f"right_{m}.pos" for m in bi_follower.right_arm.bus.motors
    }
    assert set(obs.keys()) == expected_keys

    for idx, motor in enumerate(bi_follower.left_arm.bus.motors, 1):
        assert obs[f"left_{motor}.pos"] == idx
    for idx, motor in enumerate(bi_follower.right_arm.bus.motors, 1):
        assert obs[f"right_{motor}.pos"] == idx


def test_send_action(bi_follower):
    bi_follower.connect()

    left_action = {f"left_{m}.pos": i * 10 for i, m in enumerate(bi_follower.left_arm.bus.motors, 1)}
    right_action = {f"right_{m}.pos": i * 20 for i, m in enumerate(bi_follower.right_arm.bus.motors, 1)}
    action = {**left_action, **right_action}

    returned = bi_follower.send_action(action)

    assert returned == action

    goal_pos_left = {m: (i + 1) * 10 for i, m in enumerate(bi_follower.left_arm.bus.motors)}
    goal_pos_right = {m: (i + 1) * 20 for i, m in enumerate(bi_follower.right_arm.bus.motors)}

    bi_follower.left_arm.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos_left)
    bi_follower.right_arm.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos_right)
