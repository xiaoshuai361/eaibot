#!/usr/bin/env python3
# coding=utf-8

import sys
import types

import numpy as np
import pytest


class _Vector(object):
    def __init__(self):
        self.x = self.y = self.z = 0.0


class _Twist(object):
    def __init__(self):
        self.linear = _Vector()
        self.angular = _Vector()


rospy = sys.modules.setdefault("rospy", types.ModuleType("rospy"))
rospy.get_time = lambda: 10.0
rospy.loginfo = lambda *args, **kwargs: None
rospy.logwarn = lambda *args, **kwargs: None
rospy.logerr = lambda *args, **kwargs: None
geometry_msgs = sys.modules.setdefault(
    "geometry_msgs", types.ModuleType("geometry_msgs"))
geometry_msgs_msg = sys.modules.setdefault(
    "geometry_msgs.msg", types.ModuleType("geometry_msgs.msg"))
geometry_msgs_msg.Twist = _Twist
geometry_msgs.msg = geometry_msgs_msg

from zcy_last.algorithms.building_delivery import (  # noqa: E402
    build_calibration_entry,
    compute_building_alignment_command,
    empty_building_calibration,
    load_building_calibration,
    save_building_calibration,
)
from zcy_last.algorithms.vision import (  # noqa: E402
    YoloDetection,
    YoloObstacleDetector,
)
from zcy_last.task.competition import LaneFollower  # noqa: E402


def detection(class_name="Fire Building", box=(128, 96, 192, 144),
              confidence=0.9):
    return YoloDetection(
        0, class_name, confidence, box, (240, 320, 3), 0.5)


def test_letterbox_decode_restores_box_to_original_320x240_coordinates():
    detector = YoloObstacleDetector.__new__(YoloObstacleDetector)
    detector.image_size = 320
    detector.names = {0: "Fire Building"}
    detector.confidence = 0.5
    detector.nms_threshold = 0.45
    detector.center_band_ratio = 0.5
    # Original box=(110,80,210,160). 320x240 letterbox adds 40px top/bottom.
    output = np.asarray([[160, 160, 100, 80, 1.0, 0.9]], dtype=np.float32)

    decoded = detector._decode(output, (240, 320, 3), 1.0, 0, 40)

    assert decoded[0].box == pytest.approx((110, 80, 210, 160))
    assert decoded[0].frame_shape[:2] == (240, 320)


def test_four_calibrations_are_independent_and_use_median_mad(tmp_path):
    path = tmp_path / "building.json"
    payload = empty_building_calibration(320, 240, "building.onnx")
    samples = [(0.48, 0.20), (0.50, 0.22), (0.90, 0.80)]
    entry = build_calibration_entry(
        1, "Electrical Fault Building", samples,
        320, 240, "building.onnx")
    payload["targets"]["1"] = entry
    payload["targets"]["2"] = build_calibration_entry(
        2, "Fire Building", [(0.4, 0.1)] * 3,
        320, 240, "building.onnx")
    save_building_calibration(str(path), payload)

    loaded = load_building_calibration(
        str(path), 320, 240, "building.onnx")

    assert loaded["targets"]["1"]["center_x_ratio"] == pytest.approx(0.50)
    assert loaded["targets"]["1"]["center_x_mad"] == pytest.approx(0.02)
    assert loaded["targets"]["1"]["scale_ratio"] == pytest.approx(0.22)
    assert loaded["targets"]["2"]["scale_ratio"] == pytest.approx(0.10)
    with pytest.raises(RuntimeError, match="宽度"):
        load_building_calibration(str(path), 640, 240, "building.onnx")


def test_alignment_centers_approaches_stops_and_rejects_overclose():
    entry = {"center_x_ratio": 0.5, "scale_ratio": 0.2}
    centered_far = detection(box=(144, 108, 176, 132))
    off_center = detection(box=(250, 108, 282, 132))
    ready = detection(box=(128, 96, 192, 144))
    overclose = detection(box=(112, 84, 208, 156))

    kwargs = dict(
        calibration_entry=entry, frame_shape=(240, 320, 3),
        drive_speed=0.012, center_tolerance_ratio=0.05,
        stop_scale_factor=0.95, overclose_scale_factor=1.10,
        angular_gain=1.0, max_angular=0.12)
    assert compute_building_alignment_command(
        centered_far, **kwargs).status == "approaching"
    centering = compute_building_alignment_command(off_center, **kwargs)
    assert centering.status == "centering"
    assert centering.linear_x == 0.0
    assert compute_building_alignment_command(ready, **kwargs).status == "ready"
    assert compute_building_alignment_command(
        overclose, **kwargs).status == "overclose"


class _Coordinator(object):
    def __init__(self):
        self.calls = []

    def start_delivery(self, source, ids):
        self.calls.append((source, list(ids)))


def make_align_follower(samples):
    follower = LaneFollower.__new__(LaneFollower)
    follower.state = "BUILDING_DELIVERY_ALIGN"
    follower.state_started = 0.0
    follower.velocity_owner = "line"
    follower.yolo_building_confidence = 0.5
    follower.building_delivery_event = types.SimpleNamespace(
        kind="building", area="楼宇B", class_name="Fire Building",
        display_name="火灾楼宇")
    follower.building_delivery_entry = {
        "item_id": 2, "class_name": "Fire Building",
        "center_x_ratio": 0.5, "scale_ratio": 0.2,
    }
    follower.building_delivery_stable_hits = 0
    follower.building_delivery_last_fresh_time = 10.0
    follower.untagged_delivery_failed_ids = set()
    follower.untagged_inventory = [2]
    follower.active_delivery_source = None
    follower.active_delivery_id = None
    follower.grasp_coordinator = _Coordinator()
    follower.commands = []
    follower.publish = lambda linear, angular, force=False: \
        follower.commands.append((linear, angular, force)) or True
    sample_iter = iter(samples)
    follower._poll_yolo_detections = lambda: next(sample_iter)

    def set_state(state):
        follower.state = state

    follower._set_state = set_state
    return follower


def test_three_stable_fresh_frames_are_required_before_arm_delivery():
    target = detection()
    follower = make_align_follower([(True, [target])] * 3)

    for now in (10.1, 10.2):
        follower._handle_building_delivery_align(now)
        assert follower.grasp_coordinator.calls == []
    follower._handle_building_delivery_align(10.3)

    assert follower.grasp_coordinator.calls == [("untagged", [2])]
    assert follower.state == "DELIVERING"
    assert follower.velocity_owner == "grasp"


@pytest.mark.parametrize("samples,now", [
    ([(True, [])], 10.1),
    ([(False, [])], 10.6),
])
def test_lost_or_stale_building_stops_and_never_starts_arm(samples, now):
    follower = make_align_follower(samples)

    follower._handle_building_delivery_align(now)

    assert follower.commands[-1] == (0, 0, True)
    assert follower.grasp_coordinator.calls == []
    assert follower.untagged_delivery_failed_ids == {2}
    assert follower.state == "FOLLOW"


def test_alignment_timeout_stops_without_reading_or_starting_arm():
    follower = make_align_follower([])
    follower._poll_yolo_detections = lambda: pytest.fail(
        "timeout must stop before polling")

    follower._handle_building_delivery_align(25.0)

    assert follower.commands[-1] == (0, 0, True)
    assert follower.grasp_coordinator.calls == []
    assert follower.state == "FOLLOW"


def test_overclose_frame_stops_and_never_starts_arm():
    too_large = detection(box=(112, 84, 208, 156))
    follower = make_align_follower([(True, [too_large])])

    follower._handle_building_delivery_align(10.1)

    assert follower.commands[-1] == (0, 0, True)
    assert follower.grasp_coordinator.calls == []
    assert follower.untagged_delivery_failed_ids == {2}
    assert follower.state == "FOLLOW"
