#!/usr/bin/env python3
"""YOLO-guided tag16h5 full-frame quiet-zone preprocessing."""

import argparse
import base64
import json
import math
import os
import sys

import cv2
import numpy as np


EXPECTED_MODEL_NAMES = {0: "ID1", 1: "ID2", 2: "ID3", 3: "ID4"}
ALLOWED_TAG_IDS = (1, 2, 3, 4)
DEFAULT_MODEL = "/home/eaibot/handeye-calib/src/model/yolov5/tag_yolov5n_640_best.onnx"
DEFAULT_TAG_SIZE_M = 0.0145
YOLO_INPUT_SIZE = 640


class TagDetectionError(RuntimeError):
    pass


def _finite(value, name):
    if isinstance(value, bool):
        raise TagDetectionError("%s must be a finite number" % name)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise TagDetectionError("%s must be a finite number" % name)
    if math.isnan(number) or math.isinf(number):
        raise TagDetectionError("%s must be a finite number" % name)
    return number


def _int_id(value, name):
    if isinstance(value, bool):
        raise TagDetectionError("%s must be an integer" % name)
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        raise TagDetectionError("%s must be an integer" % name)
    if number != float(value):
        raise TagDetectionError("%s must be an integer" % name)
    return number


def resolve_model_path(model_path):
    if os.path.isfile(model_path) and model_path.lower().endswith(".onnx"):
        return model_path
    if os.path.isfile(model_path):
        raise TagDetectionError("YOLO model must be an .onnx file: %s" % model_path)
    if not os.path.isdir(model_path):
        raise TagDetectionError("Model file or directory does not exist: %s" % model_path)
    candidates = []
    for root, _, filenames in os.walk(model_path):
        for name in filenames:
            if name.lower().endswith(".onnx"):
                candidates.append(os.path.join(root, name))
    if not candidates:
        raise TagDetectionError("No .onnx model file found in directory: %s" % model_path)
    preferences = [
        lambda path: os.path.basename(path).lower() == "tag_yolov5n_640_best.onnx",
        lambda path: path.lower().endswith(".onnx") and "yolov5" in path.lower().split(os.path.sep),
        lambda path: path.lower().endswith(".onnx"),
    ]
    for preference in preferences:
        preferred = [path for path in candidates if preference(path)]
        if preferred:
            return sorted(preferred)[0]
    return sorted(candidates)[0]


def load_model(model_path):
    model_path = resolve_model_path(model_path)
    return OnnxYoloV5Model(model_path)


class OnnxYoloV5Model(object):
    names = EXPECTED_MODEL_NAMES

    def __init__(self, model_path, session=None):
        self.model_path = model_path
        self.session = session
        if self.session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise TagDetectionError("Could not import onnxruntime: %s" % exc)
            try:
                self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            except Exception as exc:
                raise TagDetectionError("Could not load ONNX YOLO model: %s" % exc)
        inputs = self.session.get_inputs()
        if not inputs:
            raise TagDetectionError("ONNX YOLO model has no inputs")
        self.input_name = inputs[0].name

    def detect(self, image_bgr, confidence_threshold=0.25):
        return _infer_onnx_yolov5(
            self.session, self.input_name, image_bgr,
            confidence_threshold, YOLO_INPUT_SIZE)


def _image_from_source(source):
    if isinstance(source, np.ndarray):
        _validate_image(source)
        return source
    if isinstance(source, str):
        image = cv2.imread(source, cv2.IMREAD_COLOR)
        if image is None:
            raise TagDetectionError("Could not read image: %s" % source)
        return image
    raise TagDetectionError("YOLO source must be a BGR image or image path")


def _letterbox_image(image_bgr, input_size):
    input_size = _int_id(input_size, "YOLO input size")
    if input_size <= 0:
        raise TagDetectionError("YOLO input size must be positive")
    height, width = image_bgr.shape[:2]
    scale = min(float(input_size) / float(width), float(input_size) / float(height))
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    pad_x = (input_size - resized_width) / 2.0
    pad_y = (input_size - resized_height) / 2.0
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top:top + resized_height, left:left + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor[np.newaxis, :, :, :], scale, float(left), float(top)


def _infer_onnx_yolov5(session, input_name, image_bgr, confidence_threshold, input_size):
    _validate_image(image_bgr)
    confidence_threshold = _finite(confidence_threshold, "confidence threshold")
    tensor, scale, pad_x, pad_y = _letterbox_image(image_bgr, input_size)
    try:
        outputs = session.run(None, {input_name: tensor})
    except Exception as exc:
        raise TagDetectionError("ONNX YOLO inference failed: %s" % exc)
    if not outputs:
        raise TagDetectionError("ONNX YOLO inference returned no outputs")
    return _parse_onnx_yolov5_output(
        outputs[0], image_bgr.shape[1], image_bgr.shape[0],
        scale, pad_x, pad_y, confidence_threshold)


def _parse_onnx_yolov5_output(output, image_width, image_height,
                              scale, pad_x, pad_y, confidence_threshold):
    predictions = np.asarray(output, dtype=np.float32)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise TagDetectionError("ONNX YOLO output must have shape (N, attrs) or (1, N, attrs)")
    detections = []
    for row in predictions:
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        if row.size >= 9:
            objectness = float(row[4])
            class_scores = row[5:9]
            class_id = int(np.argmax(class_scores))
            confidence = objectness * float(class_scores[class_id])
            if confidence < confidence_threshold:
                continue
            center_x, center_y, width, height = [float(value) for value in row[:4]]
            box = [
                center_x - width / 2.0,
                center_y - height / 2.0,
                center_x + width / 2.0,
                center_y + height / 2.0,
            ]
        elif row.size >= 6:
            confidence = float(row[4])
            class_id = int(round(float(row[5])))
            if confidence < confidence_threshold:
                continue
            box = [float(value) for value in row[:4]]
        else:
            raise TagDetectionError("ONNX YOLO output row must contain at least six values")
        if class_id not in EXPECTED_MODEL_NAMES:
            continue
        original_box = _map_letterbox_box_to_image(box, image_width, image_height, scale, pad_x, pad_y)
        if original_box is None:
            continue
        detections.append({
            "class_id": class_id,
            "class_name": EXPECTED_MODEL_NAMES[class_id],
            "confidence": confidence,
            "box": original_box,
        })
    return _nms_detections(detections)


def _map_letterbox_box_to_image(box, image_width, image_height, scale, pad_x, pad_y):
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale
    x1 = max(0.0, min(float(image_width - 1), x1))
    y1 = max(0.0, min(float(image_height - 1), y1))
    x2 = max(0.0, min(float(image_width - 1), x2))
    y2 = max(0.0, min(float(image_height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _nms_detections(detections, iou_threshold=0.45):
    kept = []
    for class_id in sorted(EXPECTED_MODEL_NAMES):
        class_detections = [
            detection for detection in detections
            if detection["class_id"] == class_id
        ]
        class_detections.sort(key=lambda detection: detection["confidence"], reverse=True)
        while class_detections:
            current = class_detections.pop(0)
            kept.append(current)
            class_detections = [
                detection for detection in class_detections
                if _box_iou(current["box"], detection["box"]) <= iou_threshold
            ]
    return sorted(kept, key=lambda detection: (detection["class_id"], -detection["confidence"]))


def _box_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _box_coordinates(value):
    try:
        values = value.tolist()
    except AttributeError:
        values = value
    if (
        isinstance(values, (list, tuple))
        and len(values) == 1
        and isinstance(values[0], (list, tuple))
    ):
        values = values[0]
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise TagDetectionError("Detection box must have four coordinates")
    coordinates = [_finite(item, "Detection box coordinate") for item in values]
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        raise TagDetectionError("Detection box must have positive width and height")
    return coordinates


def infer_yolo_detections(model, image_path, confidence_threshold):
    return infer_yolo_detections_from_source(model, image_path, confidence_threshold)


def infer_yolo_detections_from_image(model, image_bgr, confidence_threshold):
    _validate_image(image_bgr)
    return infer_yolo_detections_from_source(model, image_bgr, confidence_threshold)


def infer_yolo_detections_from_source(model, source, confidence_threshold):
    confidence_threshold = _finite(confidence_threshold, "confidence threshold")
    if not 0.0 < confidence_threshold <= 1.0:
        raise TagDetectionError("confidence threshold must be in (0, 1]")
    if not hasattr(model, "detect"):
        raise TagDetectionError("YOLO model must provide detect(image_bgr, confidence_threshold)")
    try:
        detections = model.detect(_image_from_source(source), confidence_threshold)
    except TagDetectionError:
        raise
    except Exception as exc:
        raise TagDetectionError("YOLO inference failed: %s" % exc)
    return [_validate_detection(detection) for detection in detections]


def _validate_detection(detection):
    if not isinstance(detection, dict):
        raise TagDetectionError("Detection must be a dictionary")
    class_id = _int_id(detection.get("class_id"), "Detection class_id")
    if class_id not in EXPECTED_MODEL_NAMES:
        raise TagDetectionError("Detection class_id is outside ID1-ID4")
    confidence = _finite(detection.get("confidence"), "Detection confidence")
    if not 0.0 <= confidence <= 1.0:
        raise TagDetectionError("Detection confidence must be in [0, 1]")
    return {
        "class_id": class_id,
        "class_name": EXPECTED_MODEL_NAMES[class_id],
        "confidence": confidence,
        "box": _box_coordinates(detection.get("box")),
    }


def select_all_target_boxes(detections, confidence_threshold):
    selected = []
    for target_id in ALLOWED_TAG_IDS:
        target_class = target_id - 1
        matches = [
            detection for detection in detections
            if detection["class_id"] == target_class
            and detection["confidence"] >= confidence_threshold
        ]
        if not matches:
            continue
        best = dict(max(matches, key=lambda detection: detection["confidence"]))
        best["class_name"] = EXPECTED_MODEL_NAMES[target_class]
        selected.append(best)
    return selected


def select_target_box(detections, target_id, confidence_threshold):
    target_id = _int_id(target_id, "target_id")
    if target_id not in ALLOWED_TAG_IDS:
        raise TagDetectionError("target_id must be one of 1, 2, 3, 4")
    target_class = target_id - 1
    confidence_threshold = _finite(confidence_threshold, "confidence threshold")
    matches = [
        detection for detection in detections
        if detection["class_id"] == target_class
        and detection["confidence"] >= confidence_threshold
    ]
    if not matches:
        raise TagDetectionError("No YOLO detection matched target ID%d" % target_id)
    if len(matches) > 1:
        raise TagDetectionError("Multiple YOLO detections matched target ID%d" % target_id)
    selected = dict(matches[0])
    selected["class_name"] = EXPECTED_MODEL_NAMES[target_class]
    return selected


def _validate_image(image_bgr):
    if not isinstance(image_bgr, np.ndarray):
        raise TagDetectionError("Image must be a numpy array")
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise TagDetectionError("Image must be BGR uint8")
    if image_bgr.shape[0] <= 0 or image_bgr.shape[1] <= 0:
        raise TagDetectionError("Image must be non-empty")


def build_roi_variants(image_bgr, box, margin_ratio=0.35, upscale=3):
    _validate_image(image_bgr)
    if len(box) != 4:
        raise TagDetectionError("YOLO box must have four values")
    x1, y1, x2, y2 = [_finite(value, "YOLO box coordinate") for value in box]
    if x2 <= x1 or y2 <= y1:
        raise TagDetectionError("YOLO box must have positive width and height")
    margin_ratio = _finite(margin_ratio, "margin ratio")
    if margin_ratio < 0.0 or margin_ratio > 2.0:
        raise TagDetectionError("margin ratio must be in [0, 2]")
    if isinstance(upscale, bool) or int(upscale) != upscale or upscale < 1 or upscale > 8:
        raise TagDetectionError("upscale must be an integer in [1, 8]")
    upscale = int(upscale)

    height, width = image_bgr.shape[:2]
    side = max(x2 - x1, y2 - y1)
    margin = side * margin_ratio
    crop_x1 = max(0, int(math.floor(x1 - margin)))
    crop_y1 = max(0, int(math.floor(y1 - margin)))
    crop_x2 = min(width, int(math.ceil(x2 + margin)))
    crop_y2 = min(height, int(math.ceil(y2 + margin)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise TagDetectionError("Expanded YOLO ROI is empty")

    crop = np.full((crop_y2 - crop_y1, crop_x2 - crop_x1, 3), 255, dtype=np.uint8)
    box_x1 = max(0, int(math.floor(x1)))
    box_y1 = max(0, int(math.floor(y1)))
    box_x2 = min(width, int(math.ceil(x2)))
    box_y2 = min(height, int(math.ceil(y2)))
    if box_x2 <= box_x1 or box_y2 <= box_y1:
        raise TagDetectionError("Clipped YOLO box is empty")
    crop[
        box_y1 - crop_y1:box_y2 - crop_y1,
        box_x1 - crop_x1:box_x2 - crop_x1,
    ] = image_bgr[box_y1:box_y2, box_x1:box_x2]
    target_w = max(int(math.ceil((x2 - x1 + 2.0 * margin) * upscale)), crop.shape[1] * upscale)
    target_h = max(int(math.ceil((y2 - y1 + 2.0 * margin) * upscale)), crop.shape[0] * upscale)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    resized = cv2.resize(crop, (crop.shape[1] * upscale, crop.shape[0] * upscale), interpolation=cv2.INTER_CUBIC)
    offset_x = int((target_w - resized.shape[1]) // 2)
    offset_y = int((target_h - resized.shape[0]) // 2)
    canvas[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized

    variants = []
    base = {
        "name": "padded_color",
        "image": canvas,
        "crop_origin": (float(crop_x1), float(crop_y1)),
        "scale": float(upscale),
        "canvas_offset": (float(offset_x), float(offset_y)),
    }
    variants.append(base)

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(dict(base, name="clahe_gray", image=clahe))
    blurred = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.6, blurred, -0.6, 0)
    variants.append(dict(base, name="sharp_gray", image=sharp))
    thresholded = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 5)
    variants.append(dict(base, name="adaptive_threshold", image=thresholded))
    return variants


def apply_white_quiet_zones(image_bgr, detections, margin_ratio=0.35, box_expand_pixels=0.0):
    """Return a full-size image with white rectangles around every YOLO tag box.

    The original tag box pixels are copied back after the larger white rectangle
    is painted.  Image size and camera geometry are unchanged.
    """
    _validate_image(image_bgr)
    margin_ratio = _finite(margin_ratio, "margin ratio")
    if margin_ratio < 0.0 or margin_ratio > 2.0:
        raise TagDetectionError("margin ratio must be in [0, 2]")
    box_expand_pixels = _finite(box_expand_pixels, "box expand pixels")
    if box_expand_pixels < 0.0:
        raise TagDetectionError("box expand pixels must be non-negative")
    output = image_bgr.copy()
    image_height, image_width = image_bgr.shape[:2]
    boxes = []
    for detection in detections:
        box = detection.get("box") if isinstance(detection, dict) else None
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise TagDetectionError("Detection box must have four coordinates")
        x1, y1, x2, y2 = [_finite(value, "Detection box coordinate") for value in box]
        if x2 <= x1 or y2 <= y1:
            raise TagDetectionError("Detection box must have positive width and height")
        inner_source_box = [x1, y1, x2, y2]
        x1 = x1 - box_expand_pixels
        y1 = y1 - box_expand_pixels
        x2 = x2 + box_expand_pixels
        y2 = y2 + box_expand_pixels
        side = max(x2 - x1, y2 - y1)
        margin = side * margin_ratio
        outer_x1 = max(0, int(math.floor(x1 - margin)))
        outer_y1 = max(0, int(math.floor(y1 - margin)))
        outer_x2 = min(image_width, int(math.ceil(x2 + margin)))
        outer_y2 = min(image_height, int(math.ceil(y2 + margin)))
        inner_x1 = max(0, int(math.floor(x1)))
        inner_y1 = max(0, int(math.floor(y1)))
        inner_x2 = min(image_width, int(math.ceil(x2)))
        inner_y2 = min(image_height, int(math.ceil(y2)))
        if outer_x2 <= outer_x1 or outer_y2 <= outer_y1:
            raise TagDetectionError("Expanded YOLO ROI is empty")
        if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
            raise TagDetectionError("Clipped YOLO box is empty")
        tag_pixels = image_bgr[inner_y1:inner_y2, inner_x1:inner_x2].copy()
        output[outer_y1:outer_y2, outer_x1:outer_x2] = 255
        output[inner_y1:inner_y2, inner_x1:inner_x2] = tag_pixels
        boxes.append({
            "class_id": detection.get("class_id"),
            "class_name": detection.get("class_name"),
            "confidence": detection.get("confidence"),
            "box": inner_source_box,
            "inner_box": [inner_x1, inner_y1, inner_x2, inner_y2],
            "outer_box": [outer_x1, outer_y1, outer_x2, outer_y2],
        })
    return output, boxes


def apply_cached_quiet_zones(image_bgr, cached_boxes, margin_ratio=0.35, box_expand_pixels=0.0):
    if not cached_boxes:
        _validate_image(image_bgr)
        return image_bgr.copy(), []
    detections = []
    for item in cached_boxes:
        detections.append({
            "class_id": item.get("class_id"),
            "class_name": item.get("class_name"),
            "confidence": item.get("confidence"),
            "box": item.get("box"),
        })
    return apply_white_quiet_zones(
        image_bgr, detections, margin_ratio, box_expand_pixels=box_expand_pixels)


def should_refresh_yolo_boxes(now, last_update, interval, has_cached_boxes):
    interval = _finite(interval, "refresh interval")
    if interval <= 0.0:
        return True
    if not has_cached_boxes:
        return True
    now = _finite(now, "now")
    last_update = _finite(last_update, "last update")
    return now - last_update >= interval


def render_full_frame_debug(image_bgr, boxes, output_path):
    debug = image_bgr.copy()
    for item in boxes:
        outer = [int(round(value)) for value in item["outer_box"]]
        inner = [int(round(value)) for value in item["box"]]
        cv2.rectangle(debug, (outer[0], outer[1]), (outer[2], outer[3]), (255, 0, 0), 1)
        cv2.rectangle(debug, (inner[0], inner[1]), (inner[2], inner[3]), (0, 0, 255), 2)
        label = item.get("class_name") or "tag"
        cv2.putText(debug, label, (inner[0], max(15, inner[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    directory = os.path.dirname(output_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    if not cv2.imwrite(output_path, debug):
        raise TagDetectionError("Could not write debug image: %s" % output_path)


def draw_yolo_debug_overlay(image_bgr, boxes):
    _validate_image(image_bgr)
    output = image_bgr.copy()
    height, width = output.shape[:2]
    for item in boxes or []:
        outer = [int(round(value)) for value in item["outer_box"]]
        x1 = max(0, min(width - 1, outer[0]))
        y1 = max(0, min(height - 1, outer[1]))
        x2 = max(0, min(width - 1, outer[2]))
        y2 = max(0, min(height - 1, outer[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 255), 1)
        label = item.get("class_name") or "tag"
        if item.get("confidence") is not None:
            label = "%s %.2f" % (label, float(item["confidence"]))
        text_y = y1 - 4
        if text_y < 10:
            text_y = min(height - 2, y2 + 12)
        cv2.putText(output, label, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    return output


def map_variant_corners_to_image(corners, variant):
    points = np.asarray(corners, dtype=np.float64)
    if points.shape != (4, 2):
        raise TagDetectionError("Tag corners must have shape (4, 2)")
    offset_x, offset_y = variant["canvas_offset"]
    crop_x, crop_y = variant["crop_origin"]
    scale = variant["scale"]
    mapped = np.empty_like(points)
    mapped[:, 0] = (points[:, 0] - offset_x) / scale + crop_x
    mapped[:, 1] = (points[:, 1] - offset_y) / scale + crop_y
    return mapped.tolist()


def _aruco_dictionary():
    if not hasattr(cv2, "aruco"):
        raise TagDetectionError("cv2.aruco is unavailable in this Python environment")
    dictionary_id = getattr(cv2.aruco, "DICT_APRILTAG_16H5", None)
    if dictionary_id is None:
        dictionary_id = getattr(cv2.aruco, "DICT_APRILTAG_16h5", None)
    if dictionary_id is None:
        raise TagDetectionError("cv2.aruco does not provide DICT_APRILTAG_16H5")
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def _detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()
    if hasattr(parameters, "cornerRefinementMethod"):
        parameters.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_APRILTAG", 3)
    return parameters


def detect_apriltag_in_variant(variant):
    dictionary = _aruco_dictionary()
    parameters = _detector_parameters()
    image = variant["image"]
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    if ids is None or len(ids) == 0:
        return []
    detections = []
    for tag_corners, tag_id in zip(corners, ids.reshape(-1)):
        detections.append({
            "id": int(tag_id),
            "corners": np.asarray(tag_corners, dtype=np.float64).reshape(4, 2),
        })
    return detections


def confirm_decoded_tag(yolo_tag_id, decoded_tag_id):
    yolo_tag_id = _int_id(yolo_tag_id, "yolo_tag_id")
    decoded_tag_id = _int_id(decoded_tag_id, "decoded_tag_id")
    if decoded_tag_id != yolo_tag_id:
        raise TagDetectionError("Decoded tag ID%d does not match YOLO target ID%d" % (
            decoded_tag_id, yolo_tag_id))
    return decoded_tag_id


def _rotation_matrix_to_quaternion_xyzw(rotation):
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0 or not np.isfinite(norm):
        raise TagDetectionError("Pose quaternion is invalid")
    return (quaternion / norm).tolist()


def solve_tag_pose(corners, tag_size_m, camera_matrix, distortion):
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2):
        raise TagDetectionError("Tag corners must have shape (4, 2)")
    tag_size_m = _finite(tag_size_m, "tag size")
    if tag_size_m <= 0.0:
        raise TagDetectionError("tag size must be positive")
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise TagDetectionError("camera matrix must have shape (3, 3)")
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    half = tag_size_m / 2.0
    object_points = np.asarray([
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
    ], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, corners, camera_matrix, distortion,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise TagDetectionError("cv2.solvePnP failed")
    rotation, _ = cv2.Rodrigues(rvec)
    return {
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3).tolist(),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3).tolist(),
        "quaternion_xyzw": _rotation_matrix_to_quaternion_xyzw(rotation),
    }


def camera_info_from_payload(payload):
    info = payload.get("camera_info")
    if not isinstance(info, dict):
        raise TagDetectionError("camera_info must be an object")
    try:
        matrix = np.asarray(info["K"], dtype=np.float64).reshape(3, 3)
    except (KeyError, TypeError, ValueError):
        raise TagDetectionError("camera_info.K must contain 9 numbers")
    distortion = np.asarray(info.get("D", []), dtype=np.float64).reshape(-1)
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise TagDetectionError("camera focal lengths must be positive")
    return matrix, distortion


def detect_tag_pose(payload, model):
    image_path = payload.get("image_path")
    if not isinstance(image_path, str) or not os.path.isfile(image_path):
        raise TagDetectionError("Image path does not exist or is not a regular file")
    target_id = _int_id(payload.get("target_id"), "target_id")
    confidence = _finite(payload.get("confidence", 0.25), "confidence")
    tag_size_m = _finite(payload.get("tag_size_m", DEFAULT_TAG_SIZE_M), "tag_size_m")
    margin_ratio = _finite(payload.get("margin_ratio", 0.35), "margin_ratio")
    upscale = _int_id(payload.get("upscale", 3), "upscale")
    debug_image = payload.get("debug_image")
    camera_matrix, distortion = camera_info_from_payload(payload)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise TagDetectionError("Could not read image: %s" % image_path)
    yolo_detections = infer_yolo_detections(model, image_path, confidence)
    selected = select_target_box(yolo_detections, target_id, confidence)
    variants = build_roi_variants(image, selected["box"], margin_ratio, upscale)

    decode_attempts = []
    for variant in variants:
        detections = detect_apriltag_in_variant(variant)
        decode_attempts.append({
            "variant": variant["name"],
            "ids": [detection["id"] for detection in detections],
        })
        matching = [detection for detection in detections if detection["id"] == target_id]
        if len(matching) != 1:
            continue
        confirm_decoded_tag(target_id, matching[0]["id"])
        image_corners = map_variant_corners_to_image(matching[0]["corners"], variant)
        pose = solve_tag_pose(image_corners, tag_size_m, camera_matrix, distortion)
        if isinstance(debug_image, str) and debug_image.strip():
            render_debug_image(image, selected["box"], image_corners, debug_image, target_id)
        return {
            "ok": True,
            "target_id": target_id,
            "class_id": selected["class_id"],
            "class_name": selected["class_name"],
            "confidence": selected["confidence"],
            "box": selected["box"],
            "corners": image_corners,
            "pose": pose,
            "variant": variant["name"],
            "attempts": decode_attempts,
        }
    raise TagDetectionError("YOLO found ID%d but AprilTag decode did not confirm it: %s" % (
        target_id, decode_attempts))


def generate_full_frame(payload, model):
    image_path = payload.get("image_path")
    if not isinstance(image_path, str) or not os.path.isfile(image_path):
        raise TagDetectionError("Image path does not exist or is not a regular file")
    output_image_path = payload.get("output_image_path")
    if not isinstance(output_image_path, str) or not output_image_path.strip():
        raise TagDetectionError("output_image_path must be non-empty text")
    confidence = _finite(payload.get("confidence", 0.25), "confidence")
    margin_ratio = _finite(payload.get("margin_ratio", 0.35), "margin_ratio")
    box_expand_pixels = _finite(payload.get("box_expand_pixels", 0.0), "box_expand_pixels")
    debug_image = payload.get("debug_image")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise TagDetectionError("Could not read image: %s" % image_path)
    full_frame, boxes = generate_quiet_frame_from_image(
        image, model, confidence_threshold=confidence, margin_ratio=margin_ratio,
        box_expand_pixels=box_expand_pixels)
    if not boxes:
        raise TagDetectionError("YOLO did not find any ID1-ID4 tag")
    directory = os.path.dirname(output_image_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    if not cv2.imwrite(output_image_path, full_frame):
        raise TagDetectionError("Could not write enhanced full-frame image: %s" % output_image_path)
    if isinstance(debug_image, str) and debug_image.strip():
        render_full_frame_debug(full_frame, boxes, debug_image)
    return {
        "ok": True,
        "output_image_path": output_image_path,
        "detections": boxes,
    }


def generate_quiet_frame_from_image(image_bgr, model, confidence_threshold=0.25,
                                    margin_ratio=0.35, box_expand_pixels=0.0):
    _validate_image(image_bgr)
    confidence_threshold = _finite(confidence_threshold, "confidence threshold")
    margin_ratio = _finite(margin_ratio, "margin ratio")
    box_expand_pixels = _finite(box_expand_pixels, "box_expand_pixels")
    yolo_detections = infer_yolo_detections_from_image(model, image_bgr, confidence_threshold)
    selected = select_all_target_boxes(yolo_detections, confidence_threshold)
    if not selected:
        return image_bgr.copy(), []
    return apply_white_quiet_zones(
        image_bgr, selected, margin_ratio, box_expand_pixels=box_expand_pixels)


def generate_quiet_frame_from_encoded(payload, model, cached_boxes=None):
    image_data = payload.get("image_bgr_png_base64")
    if image_data is None:
        image_data = payload.get("image_bgr_jpeg_base64")
    if not isinstance(image_data, str) or not image_data:
        raise TagDetectionError("image_bgr_png_base64 must be non-empty text")
    try:
        encoded = base64.b64decode(image_data.encode("ascii"))
    except Exception as exc:
        raise TagDetectionError("Could not decode image_bgr_png_base64: %s" % exc)
    buffer = np.frombuffer(encoded, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise TagDetectionError("Could not decode JPEG frame")
    confidence = _finite(payload.get("confidence", 0.25), "confidence")
    margin_ratio = _finite(payload.get("margin_ratio", 0.35), "margin_ratio")
    box_expand_pixels = _finite(payload.get("box_expand_pixels", 0.0), "box_expand_pixels")
    refresh_boxes = bool(payload.get("refresh_boxes", True)) or not cached_boxes
    if refresh_boxes:
        full_frame, boxes = generate_quiet_frame_from_image(
            image, model, confidence_threshold=confidence, margin_ratio=margin_ratio,
            box_expand_pixels=box_expand_pixels)
    else:
        full_frame, boxes = apply_cached_quiet_zones(
            image, cached_boxes or [], margin_ratio=margin_ratio,
            box_expand_pixels=box_expand_pixels)
    if bool(payload.get("draw_yolo_overlay", False)) and boxes:
        full_frame = draw_yolo_debug_overlay(full_frame, boxes)
    ok, encoded_frame = cv2.imencode(".png", full_frame)
    if not ok:
        raise TagDetectionError("Could not encode quiet-zone frame")
    return {
        "ok": True,
        "image_bgr_png_base64": base64.b64encode(encoded_frame.tobytes()).decode("ascii"),
        "detections": boxes,
    }


def render_debug_image(image_bgr, box, corners, output_path, target_id):
    output = image_bgr.copy()
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
    polygon = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(output, [polygon], True, (0, 255, 0), 2)
    center = np.mean(np.asarray(corners, dtype=np.float64), axis=0)
    cv2.circle(output, (int(round(center[0])), int(round(center[1]))), 4, (255, 0, 0), -1)
    cv2.putText(output, "ID%d" % target_id, (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    directory = os.path.dirname(output_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    if not cv2.imwrite(output_path, output):
        raise TagDetectionError("Could not write debug image: %s" % output_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="YOLO-guided tag16h5 ROI detector")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["full_frame", "pose", "quiet_frame", "worker"], default="full_frame")
    return parser.parse_args(argv)


def handle_payload(payload, model, mode):
    if mode == "pose":
        return detect_tag_pose(payload, model)
    if mode == "quiet_frame":
        return generate_quiet_frame_from_encoded(payload, model)
    return generate_full_frame(payload, model)


def worker_loop(model, stdin=None, stdout=None):
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    cached_boxes = []
    for line in stdin:
        try:
            payload = json.loads(line)
            response = generate_quiet_frame_from_encoded(
                payload, model, cached_boxes=cached_boxes)
            if payload.get("refresh_boxes", True) and response.get("ok"):
                cached_boxes = response.get("detections", [])
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        stdout.write(json.dumps(response, ensure_ascii=True, allow_nan=False) + "\n")
        stdout.flush()
    return 0


def main(argv=None, stdin=None, stdout=None):
    args = parse_args(argv)
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    try:
        model = load_model(args.model)
        if args.mode == "worker":
            return worker_loop(model, stdin=stdin, stdout=stdout)
        payload = json.loads(stdin.readline())
        if args.mode in ("full_frame", "pose"):
            image_path = payload.get("image_path")
            if not isinstance(image_path, str) or not os.path.isfile(image_path):
                raise TagDetectionError("Image path does not exist or is not a regular file")
        response = handle_payload(payload, model, args.mode)
        stdout.write(json.dumps(response, ensure_ascii=True, allow_nan=False) + "\n")
        stdout.flush()
        return 0
    except Exception as exc:
        response = {"ok": False, "error": str(exc)}
        stdout.write(json.dumps(response, ensure_ascii=True, allow_nan=False) + "\n")
        stdout.flush()
        return 1


if __name__ == "__main__":
    sys.exit(main())
