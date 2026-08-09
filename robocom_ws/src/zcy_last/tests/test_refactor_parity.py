#!/usr/bin/env python3
# coding=utf-8
"""拆分前后的纯算法行为必须保持一致。"""

import importlib.util
import sys
import types
from pathlib import Path

import cv2
import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _install_ros_stubs():
    rospy = sys.modules.setdefault("rospy", types.ModuleType("rospy"))
    rospy.is_shutdown = lambda: False
    geometry_msgs = sys.modules.setdefault(
        "geometry_msgs", types.ModuleType("geometry_msgs")
    )
    geometry_msgs_msg = sys.modules.setdefault(
        "geometry_msgs.msg", types.ModuleType("geometry_msgs.msg")
    )
    geometry_msgs_msg.Twist = type("Twist", (), {})
    geometry_msgs.msg = geometry_msgs_msg


def _load_legacy():
    _install_ros_stubs()
    path = SRC_ROOT / "line_cy_task.py"
    spec = importlib.util.spec_from_file_location("zcy_last_legacy_task", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_install_ros_stubs()
from zcy_last.algorithms import vision as current  # noqa: E402
from zcy_last import config  # noqa: E402

legacy = _load_legacy()


def test_route_and_control_parameters_match_legacy():
    names = (
        "TASK_TURN_COMMANDS",
        "KP",
        "KD",
        "LARGE_ERROR_KP",
        "LARGE_ERROR_KD",
        "LANE_WIDTH_PIXELS",
        "STOP_NEAR_RATIO",
    )
    for name in names:
        assert getattr(config, name) == getattr(legacy, name)


def test_lane_observation_matches_legacy():
    binary = np.zeros((480, 640), dtype=np.uint8)
    cv2.line(binary, (30, 440), (120, 90), 255, 12)
    cv2.line(binary, (610, 440), (520, 90), 255, 12)

    before = legacy.LaneDetector().observe(binary, 620.0)
    after = current.LaneDetector().observe(binary, 620.0)

    assert after.valid == before.valid
    assert after.dual_rows == before.dual_rows
    assert after.follow_side == before.follow_side
    assert after.left_points == before.left_points
    assert after.right_points == before.right_points
    assert after.center_x == before.center_x


def test_crosswalk_result_matches_legacy():
    binary = np.zeros((480, 640), dtype=np.uint8)
    for x in (210, 300, 390):
        polygon = cv2.boxPoints(((float(x), 230.0), (32.0, 105.0), 3.0))
        cv2.fillConvexPoly(binary, polygon.astype(np.int32), 255)
    cv2.line(binary, (130, 350), (510, 350), 255, 18)

    before = legacy.CrosswalkDetector().detect(binary)
    after = current.CrosswalkDetector().detect(binary)

    assert after.candidate == before.candidate
    assert after.stop_angle == before.stop_angle
    assert after.stop_bottom == before.stop_bottom
    assert len(after.stripe_polygons) == len(before.stripe_polygons)


def test_route_context_matches_legacy():
    for index in range(len(config.TASK_TURN_COMMANDS)):
        assert current.yolo_route_context(index, "FOLLOW") == \
            legacy.yolo_route_context(index, "FOLLOW")
