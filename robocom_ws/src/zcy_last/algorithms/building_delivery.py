#!/usr/bin/env python3
# coding=utf-8
"""YOLO building-box calibration for mechanical-arm delivery distance."""

import json
import math
import os

import numpy as np


CALIBRATION_VERSION = 2


def _finite(value, label):
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("%s must be finite" % label)
    return number


def _positive_int(value, label):
    number = int(value)
    if number <= 0:
        raise RuntimeError("%s must be positive" % label)
    return number


def building_box_geometry(detection, frame_shape, edge_margin_px=1.0):
    height, width = [int(value) for value in frame_shape[:2]]
    if width <= 0 or height <= 0:
        raise RuntimeError("building frame dimensions must be positive")
    x1, y1, x2, y2 = [float(value) for value in detection.box]
    margin = max(0.0, float(edge_margin_px))
    if x1 <= margin or y1 <= margin \
            or x2 >= width - 1 - margin or y2 >= height - 1 - margin:
        raise RuntimeError("楼宇检测框接触画面边缘，不能用于距离估计")
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0.0 or box_height <= 0.0:
        raise RuntimeError("building detection box must have positive area")
    return {
        "center_x_ratio": ((x1 + x2) * 0.5) / float(width),
        "width_px": box_width,
        "height_px": box_height,
    }


def _aggregate_by_distance(samples):
    groups = {}
    for distance_mm, width_px, height_px in samples:
        values = tuple(_finite(value, "distance sample") for value in (
            distance_mm, width_px, height_px))
        if any(value <= 0.0 for value in values):
            raise RuntimeError("building distance samples must be positive")
        groups.setdefault(round(values[0], 3), []).append(values)
    return [tuple(np.median(groups[key], axis=0).tolist())
            for key in sorted(groups)]


def _fit_axis(samples, index):
    distances = np.asarray([item[0] for item in samples], dtype=np.float64)
    pixels = np.asarray([item[index] for item in samples], dtype=np.float64)
    design = np.column_stack((1.0 / pixels, np.ones_like(pixels)))
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        design, distances, rcond=None)
    predicted = design.dot(coefficients)
    errors = predicted - distances
    if coefficients[0] <= 0.0:
        raise RuntimeError("楼宇框尺寸与真实距离不满足反比例关系")
    return {
        "a": float(coefficients[0]),
        "b": float(coefficients[1]),
        "rmse_mm": float(np.sqrt(np.mean(errors ** 2))),
        "max_error_mm": float(np.max(np.abs(errors))),
    }


def build_distance_calibration_entry(
        item_id, class_name, samples, frame_width, frame_height,
        model_name, reference_distance_mm):
    if len(samples) < 6:
        raise RuntimeError("楼宇距离标定至少需要6帧有效样本")
    aggregated = _aggregate_by_distance(samples)
    if len(aggregated) < 3:
        raise RuntimeError("楼宇距离标定至少需要3个不同真实距离")
    return {
        "item_id": int(item_id),
        "class_name": str(class_name),
        "reference_distance_mm": _finite(
            reference_distance_mm, "reference_distance_mm"),
        "min_distance_mm": float(aggregated[0][0]),
        "max_distance_mm": float(aggregated[-1][0]),
        "width": _fit_axis(aggregated, 1),
        "height": _fit_axis(aggregated, 2),
        "sample_count": len(samples),
        "distance_point_count": len(aggregated),
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
        raise RuntimeError(
            "楼宇标定不是多距离版本，请备份并重新标定四类：version=%r"
            % payload.get("version"))
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
    temporary = path + ".tmp"
    try:
        with open(temporary, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def require_building_target(calibration, item_id, class_name=None):
    entry = calibration.get("targets", {}).get(str(int(item_id)))
    if not isinstance(entry, dict):
        raise RuntimeError("楼宇投递缺少ID%d距离标定" % int(item_id))
    if int(entry.get("item_id", -1)) != int(item_id):
        raise RuntimeError("楼宇投递ID%d标定内容不一致" % int(item_id))
    if class_name is not None and str(entry.get("class_name")) != str(class_name):
        raise RuntimeError("楼宇投递ID%d类别不一致：%s != %s" % (
            int(item_id), entry.get("class_name"), class_name))
    reference = _finite(
        entry.get("reference_distance_mm"), "reference_distance_mm")
    minimum = _finite(entry.get("min_distance_mm"), "min_distance_mm")
    maximum = _finite(entry.get("max_distance_mm"), "max_distance_mm")
    if reference <= 0.0 or minimum <= 0.0 or maximum <= minimum:
        raise RuntimeError("楼宇投递距离范围无效")
    if not minimum <= reference <= maximum:
        raise RuntimeError("楼宇投递示教参考距离不在标定范围内")
    for axis in ("width", "height"):
        model = entry.get(axis)
        if not isinstance(model, dict) \
                or _finite(model.get("a"), axis + ".a") <= 0.0:
            raise RuntimeError("楼宇投递ID%d缺少%s距离模型" %
                               (int(item_id), axis))
        _finite(model.get("b"), axis + ".b")
    return entry


def select_locked_building_detection(detections, class_name, confidence):
    candidates = [item for item in detections
                  if item.class_name == str(class_name)
                  and float(item.confidence) >= float(confidence)]
    return max(candidates, key=lambda item: item.confidence) \
        if candidates else None


def estimate_building_distance_mm(detection, calibration_entry, frame_shape,
                                  max_axis_disagreement_mm):
    geometry = building_box_geometry(detection, frame_shape)
    estimates = []
    for axis, pixel_key in (("width", "width_px"),
                            ("height", "height_px")):
        model = calibration_entry[axis]
        estimates.append(
            float(model["a"]) / geometry[pixel_key] + float(model["b"]))
    disagreement = abs(estimates[0] - estimates[1])
    if disagreement > float(max_axis_disagreement_mm):
        raise RuntimeError(
            "楼宇框宽高估距相差%.1fmm，超过%.1fmm" %
            (disagreement, float(max_axis_disagreement_mm)))
    distance_mm = float(np.median(estimates))
    minimum = float(calibration_entry["min_distance_mm"])
    maximum = float(calibration_entry["max_distance_mm"])
    if not minimum <= distance_mm <= maximum:
        raise RuntimeError(
            "楼宇估距%.1fmm超出标定范围%.1f~%.1fmm" %
            (distance_mm, minimum, maximum))
    return geometry["center_x_ratio"], distance_mm
