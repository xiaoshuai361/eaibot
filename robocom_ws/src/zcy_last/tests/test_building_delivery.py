#!/usr/bin/env python3
# coding=utf-8

import json
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
    build_distance_calibration_entry,
    empty_building_calibration,
    estimate_building_distance_mm,
    load_building_calibration,
    save_building_calibration,
)
from zcy_last.algorithms.vision import (  # noqa: E402
    YoloDetection,
    YoloObstacleDetector,
    YoloTaskLedger,
    draw_yolo_boxes,
)
from zcy_last.config import YOLO_BUILDING_CENTER_ROI_X_RATIO  # noqa: E402
def detection(class_name="Fire Building", box=(135, 105, 185, 135),
              confidence=0.9):
    return YoloDetection(
        0, class_name, confidence, box, (240, 320, 3), 0.5)


def distance_samples(a_width=20000.0, b=50.0):
    samples = []
    for distance_mm in (350, 400, 450, 550, 600):
        for delta in (-0.2, 0.2):
            samples.append((
                distance_mm,
                a_width / (distance_mm - b) + delta,
            ))
    return samples


def calibration_entry(item_id=2, class_name="Fire Building"):
    return build_distance_calibration_entry(
        item_id, class_name, distance_samples(),
        320, 240, "building.onnx", 450.0)


def test_letterbox_decode_restores_box_to_original_320x240_coordinates():
    detector = YoloObstacleDetector.__new__(YoloObstacleDetector)
    detector.image_size = 320
    detector.names = {0: "Fire Building"}
    detector.confidence = 0.5
    detector.nms_threshold = 0.45
    detector.center_band_ratio = 0.5
    output = np.asarray([[160, 160, 100, 80, 1.0, 0.9]], dtype=np.float32)

    decoded = detector._decode(output, (240, 320, 3), 1.0, 0, 40)

    assert decoded[0].box == pytest.approx((110, 80, 210, 160))
    assert decoded[0].frame_shape[:2] == (240, 320)


def test_building_roi_is_drawn_red_and_controls_building_stop_event():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    shown = draw_yolo_boxes(
        frame, [], 0.65, draw_center_band=False,
        center_roi_x_ratio=YOLO_BUILDING_CENTER_ROI_X_RATIO)
    assert shown[120, 54].tolist() == [0, 0, 255]
    assert shown[120, 173].tolist() == [0, 0, 255]

    inside_red = detection(box=(50, 100, 90, 140))
    outside_red = detection(box=(200, 100, 240, 140))
    ledger = YoloTaskLedger()
    context = {"kind": "building", "area": "楼宇A"}

    assert inside_red.in_center is False
    assert ledger.select_event(
        context, [inside_red], 0.5, building_confidence=0.5) is not None
    assert ledger.select_event(
        context, [outside_red], 0.5, building_confidence=0.5) is None


def test_real_distance_fit_uses_only_box_width():
    entry = calibration_entry()

    center, distance_mm = estimate_building_distance_mm(
        detection(), entry, (240, 320, 3))

    assert center == pytest.approx(0.5)
    assert distance_mm == pytest.approx(450.0, abs=2.0)
    assert entry["distance_point_count"] == 5
    assert entry["sample_count"] == 10
    assert entry["reference_distance_mm"] == 450.0
    assert entry["min_distance_mm"] == 350.0
    assert entry["max_distance_mm"] == 600.0
    assert entry["width"]["rmse_mm"] < 2.0
    assert "height" not in entry


def test_four_distance_calibrations_are_independent_and_validate_format(tmp_path):
    path = tmp_path / "building.json"
    payload = empty_building_calibration(320, 240, "building.onnx")
    payload["targets"]["1"] = build_distance_calibration_entry(
        1, "Electrical Fault Building", distance_samples(),
        320, 240, "building.onnx", 450.0)
    payload["targets"]["2"] = build_distance_calibration_entry(
        2, "Fire Building", distance_samples(24000.0),
        320, 240, "building.onnx", 450.0)
    save_building_calibration(str(path), payload)

    loaded = load_building_calibration(
        str(path), 320, 240, "building.onnx")

    assert loaded["targets"]["1"]["width"]["a"] != \
        loaded["targets"]["2"]["width"]["a"]
    with pytest.raises(RuntimeError, match="宽度"):
        load_building_calibration(str(path), 640, 240, "building.onnx")

    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="多距离版本"):
        load_building_calibration(str(old_path))


def test_top_bottom_crop_is_allowed_but_left_right_crop_is_rejected():
    entry = calibration_entry()
    _center, distance_mm = estimate_building_distance_mm(
        detection(box=(135, 0, 185, 240)), entry, (240, 320, 3))
    assert distance_mm == pytest.approx(450.0, abs=2.0)

    with pytest.raises(RuntimeError, match="左右边界"):
        estimate_building_distance_mm(
            detection(box=(0, 100, 100, 140)), entry,
            (240, 320, 3))
    with pytest.raises(RuntimeError, match="左右边界"):
        estimate_building_distance_mm(
            detection(box=(220, 100, 319, 140)), entry,
            (240, 320, 3))


def test_distance_outside_sampled_range_is_rejected():
    entry = calibration_entry()
    too_far = detection(box=(145.6, 111.3, 174.4, 128.7))  # about 750mm

    with pytest.raises(RuntimeError, match="超出标定范围"):
        estimate_building_distance_mm(
            too_far, entry, (240, 320, 3))
