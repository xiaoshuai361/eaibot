#!/usr/bin/env python3
# coding=utf-8
"""Pure helpers for repeatable YOLO-guided building delivery alignment."""

import json
import math
import os

import numpy as np


CALIBRATION_VERSION = 1


class BuildingAlignmentCommand(object):
    def __init__(self, status, linear_x, angular_z):
        self.status = str(status)
        self.linear_x = float(linear_x)
        self.angular_z = float(angular_z)


def _finite(value, label):
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise RuntimeError("%s must be finite" % label)
    return number


def _positive_int(value, label):
    number = int(value)
    if number <= 0:
        raise RuntimeError("%s must be positive" % label)
    return number


def building_box_measurement(detection, frame_shape):
    height, width = [int(value) for value in frame_shape[:2]]
    if width <= 0 or height <= 0:
        raise RuntimeError("building frame dimensions must be positive")
    x1, y1, x2, y2 = [float(value) for value in detection.box]
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0.0 or box_height <= 0.0:
        raise RuntimeError("building detection box must have positive area")
    center_x_ratio = ((x1 + x2) * 0.5) / float(width)
    scale_ratio = math.sqrt(
        (box_width / float(width)) * (box_height / float(height)))
    return center_x_ratio, scale_ratio


def _median_and_mad(values, label):
    data = np.asarray([_finite(value, label) for value in values],
                      dtype=np.float64)
    if data.size == 0:
        raise RuntimeError("%s has no samples" % label)
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    return median, mad


def build_calibration_entry(item_id, class_name, measurements,
                            frame_width, frame_height, model_name):
    if not measurements:
        raise RuntimeError("building calibration needs at least one sample")
    centers = [item[0] for item in measurements]
    scales = [item[1] for item in measurements]
    center_median, center_mad = _median_and_mad(
        centers, "center_x_ratio")
    scale_median, scale_mad = _median_and_mad(scales, "scale_ratio")
    if not 0.0 <= center_median <= 1.0:
        raise RuntimeError("building center calibration is outside the frame")
    if not 0.0 < scale_median <= 1.0:
        raise RuntimeError("building scale calibration is invalid")
    return {
        "item_id": int(item_id),
        "class_name": str(class_name),
        "center_x_ratio": center_median,
        "center_x_mad": center_mad,
        "scale_ratio": scale_median,
        "scale_mad": scale_mad,
        "sample_count": len(measurements),
        "frame_width": _positive_int(frame_width, "frame_width"),
        "frame_height": _positive_int(frame_height, "frame_height"),
        "model_name": os.path.basename(str(model_name)),
    }


def empty_building_calibration(frame_width, frame_height, model_name):
    return {
        "version": CALIBRATION_VERSION,
        "frame_width": _positive_int(frame_width, "frame_width"),
        "frame_height": _positive_int(frame_height, "frame_height"),
        "model_name": os.path.basename(str(model_name)),
        "targets": {},
    }


def load_building_calibration(path, expected_width=None,
                              expected_height=None, expected_model=None,
                              allow_missing=False):
    if not os.path.isfile(path):
        if allow_missing:
            return None
        raise RuntimeError("楼宇投递标定文件不存在：%s" % path)
    try:
        with open(path, "r") as handle:
            payload = json.load(handle)
    except (IOError, OSError, ValueError) as exc:
        raise RuntimeError("无法读取楼宇投递标定：%s" % exc)
    if payload.get("version") != CALIBRATION_VERSION:
        raise RuntimeError("不支持的楼宇投递标定版本：%r" %
                           payload.get("version"))
    width = _positive_int(payload.get("frame_width"), "frame_width")
    height = _positive_int(payload.get("frame_height"), "frame_height")
    if expected_width is not None and width != int(expected_width):
        raise RuntimeError("楼宇标定宽度%d与运行宽度%d不一致" %
                           (width, int(expected_width)))
    if expected_height is not None and height != int(expected_height):
        raise RuntimeError("楼宇标定高度%d与运行高度%d不一致" %
                           (height, int(expected_height)))
    model_name = os.path.basename(str(payload.get("model_name", "")))
    if expected_model is not None and model_name != os.path.basename(
            str(expected_model)):
        raise RuntimeError("楼宇标定模型%s与运行模型%s不一致" %
                           (model_name, os.path.basename(str(expected_model))))
    if not isinstance(payload.get("targets"), dict):
        raise RuntimeError("楼宇投递标定缺少 targets")
    return payload


def save_building_calibration(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if hasattr(os, "replace"):
            os.replace(tmp_path, path)
        else:
            os.rename(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def require_building_target(calibration, item_id, class_name=None):
    targets = calibration.get("targets", {})
    entry = targets.get(str(int(item_id)))
    if not isinstance(entry, dict):
        raise RuntimeError("楼宇投递缺少ID%d视觉标定" % int(item_id))
    if int(entry.get("item_id", -1)) != int(item_id):
        raise RuntimeError("楼宇投递ID%d标定内容不一致" % int(item_id))
    if class_name is not None and str(entry.get("class_name")) != str(class_name):
        raise RuntimeError("楼宇投递ID%d类别不一致：%s != %s" % (
            int(item_id), entry.get("class_name"), class_name))
    for key in ("center_x_ratio", "scale_ratio"):
        entry[key] = _finite(entry.get(key), key)
    if not 0.0 <= entry["center_x_ratio"] <= 1.0:
        raise RuntimeError("楼宇投递ID%d中心标定无效" % int(item_id))
    if not 0.0 < entry["scale_ratio"] <= 1.0:
        raise RuntimeError("楼宇投递ID%d尺度标定无效" % int(item_id))
    return entry


def select_locked_building_detection(detections, class_name, confidence):
    candidates = [
        item for item in detections
        if item.class_name == str(class_name)
        and float(item.confidence) >= float(confidence)
    ]
    return max(candidates, key=lambda item: item.confidence) \
        if candidates else None


def compute_building_alignment_command(
        detection, calibration_entry, frame_shape, drive_speed,
        center_tolerance_ratio, stop_scale_factor, overclose_scale_factor,
        angular_gain, max_angular):
    center_ratio, scale_ratio = building_box_measurement(
        detection, frame_shape)
    target_center = _finite(
        calibration_entry["center_x_ratio"], "center_x_ratio")
    target_scale = _finite(
        calibration_entry["scale_ratio"], "scale_ratio")
    center_error = center_ratio - target_center
    tolerance = abs(_finite(
        center_tolerance_ratio, "center_tolerance_ratio"))
    stop_scale = target_scale * _finite(
        stop_scale_factor, "stop_scale_factor")
    overclose_scale = target_scale * _finite(
        overclose_scale_factor, "overclose_scale_factor")
    angular = max(-abs(float(max_angular)), min(
        abs(float(max_angular)), -float(angular_gain) * center_error))
    if scale_ratio > overclose_scale:
        status, linear, angular = "overclose", 0.0, 0.0
    elif abs(center_error) > tolerance:
        status, linear = "centering", 0.0
    elif scale_ratio >= stop_scale:
        status, linear, angular = "ready", 0.0, 0.0
    else:
        status, linear = "approaching", abs(float(drive_speed))
    return BuildingAlignmentCommand(status, linear, angular)
