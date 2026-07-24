from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np


try:
    STRING_TYPES = (basestring,)
except NameError:
    STRING_TYPES = (str,)


class LocalizationError(RuntimeError):
    pass


DEFAULT_TARGET_CLASSES = {
    "power": {"class_id": 0, "class_name": "Emergency power supply device"},
    "fire": {"class_id": 1, "class_name": "Fire extinguishing device"},
    "gas": {"class_id": 2, "class_name": "Gas purification device"},
    "support": {"class_id": 3, "class_name": "Structural support device"},
}


DEFAULT_CONFIG = {
    "model_path": (
        "/home/eaibot/handeye-calib/src/model/yolov5/"
        "Block_v5n_yolov5n_640_best.onnx"
    ),
    "target_classes": DEFAULT_TARGET_CLASSES,
    "target_size_mm": 30.0,
    "distance_method": "theory",
    "frames_required": 10,
    "confidence_min": 0.70,
    "nms_iou": 0.45,
    "input_size": 640,
    "box_width_min_px": 30.0,
    "box_aspect_ratio_min": 0.75,
    "box_aspect_ratio_max": 1.30,
    "center_std_max_px": 2.0,
    "width_cv_max": 0.03,
    "rgb_topic": "/camera/rgb/image_raw",
    "camera_info_topic": "/camera/rgb/camera_info",
    "camera_frame": "camera_rgb_optical_frame",
    "base_frame": "base",
    "distance_models": {
        "power": {"a": None, "b": None},
        "fire": {"a": None, "b": None},
        "gas": {"a": None, "b": None},
        "support": {"a": None, "b": None},
    },
    "fixed_z_mm": None,
    "target_offset_mm": [0.0, 0.0, 0.0],
    "tool_offset_mm": None,
    "fixed_orientation_xyzw": None,
    "pregrasp_distance_mm": 50.0,
    "suction_compression_mm": 3.0,
    "velocity_scale": 0.05,
    "acceleration_scale": 0.05,
    "planning_time": 5.0,
    "tf_timeout": 5.0,
    "base_min_z_mm": 40.0,
    "base_max_radius_mm": 500.0,
}


def _isfinite(value):
    return bool(np.isfinite(value))


def finite_scalar(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise LocalizationError("%s must be a finite number" % name)
    if not _isfinite(number):
        raise LocalizationError("%s must be a finite number" % name)
    return number


def finite_vector(values, name, expected_length=None):
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise LocalizationError("%s must contain finite numbers" % name)
    if vector.ndim != 1:
        raise LocalizationError("%s must be one-dimensional" % name)
    if expected_length is not None and vector.size != expected_length:
        raise LocalizationError("%s must contain %d values" % (name, expected_length))
    if not np.all(np.isfinite(vector)):
        raise LocalizationError("%s must contain finite numbers" % name)
    return vector


def box_geometry(box):
    vector = finite_vector(box, "YOLO box", 4)
    x1, y1, x2, y2 = [float(item) for item in vector.tolist()]
    if x2 <= x1 or y2 <= y1:
        raise LocalizationError("YOLO box must have positive width and height")
    width = x2 - x1
    height = y2 - y1
    return {
        "u": (x1 + x2) * 0.5,
        "v": (y1 + y2) * 0.5,
        "w": width,
        "h": height,
        "aspect": width / height,
    }


def is_detection_usable(detection, rules):
    try:
        confidence = finite_scalar(detection.get("confidence"), "confidence")
        geometry = box_geometry(detection.get("box"))
        confidence_min = finite_scalar(rules.get("confidence_min"), "confidence_min")
        width_min = finite_scalar(rules.get("box_width_min_px"), "box_width_min_px")
        aspect_min = finite_scalar(
            rules.get("box_aspect_ratio_min"), "box_aspect_ratio_min"
        )
        aspect_max = finite_scalar(
            rules.get("box_aspect_ratio_max"), "box_aspect_ratio_max"
        )
    except LocalizationError as exc:
        return False, str(exc)

    if confidence < confidence_min:
        return False, "confidence %.3f below %.3f" % (confidence, confidence_min)
    if geometry["w"] < width_min:
        return False, "width %.2f below %.2f" % (geometry["w"], width_min)
    if geometry["aspect"] < aspect_min or geometry["aspect"] > aspect_max:
        return False, "aspect %.3f outside [%.3f, %.3f]" % (
            geometry["aspect"],
            aspect_min,
            aspect_max,
        )
    return True, ""


def detection_to_observation(detection):
    geometry = box_geometry(detection.get("box"))
    return {
        "u": geometry["u"],
        "v": geometry["v"],
        "w": geometry["w"],
        "h": geometry["h"],
        "confidence": finite_scalar(detection.get("confidence"), "confidence"),
        "box": tuple(float(value) for value in detection.get("box")),
        "class_id": int(detection.get("class_id")),
        "class_name": detection.get("class_name", str(detection.get("class_id"))),
    }


def stable_median_observation(
    observations, frames_required, center_std_max_px, width_cv_max
):
    if isinstance(frames_required, bool) or int(frames_required) <= 0:
        raise LocalizationError("frames_required must be a positive integer")
    frames_required = int(frames_required)
    if len(observations) < frames_required:
        raise LocalizationError(
            "only %d valid observations, need %d" % (len(observations), frames_required)
        )
    center_std_max_px = finite_scalar(center_std_max_px, "center_std_max_px")
    width_cv_max = finite_scalar(width_cv_max, "width_cv_max")
    if center_std_max_px <= 0.0 or width_cv_max <= 0.0:
        raise LocalizationError("stability limits must be positive")

    recent = observations[-frames_required:]
    matrix = np.asarray(
        [[item["u"], item["v"], item["w"], item["h"], item["confidence"]] for item in recent],
        dtype=np.float64,
    )
    if matrix.shape != (frames_required, 5) or not np.all(np.isfinite(matrix)):
        raise LocalizationError("observations must contain finite u/v/w/h/confidence")

    center_std = max(float(np.std(matrix[:, 0])), float(np.std(matrix[:, 1])))
    width_mean = float(np.mean(matrix[:, 2]))
    if width_mean <= 0.0:
        raise LocalizationError("width mean must be positive")
    width_cv = float(np.std(matrix[:, 2]) / width_mean)
    if center_std > center_std_max_px:
        raise LocalizationError(
            "center std %.3f px exceeds %.3f px" % (center_std, center_std_max_px)
        )
    if width_cv > width_cv_max:
        raise LocalizationError(
            "width cv %.5f exceeds %.5f" % (width_cv, width_cv_max)
        )

    median = np.median(matrix, axis=0)
    return {
        "u": float(median[0]),
        "v": float(median[1]),
        "w": float(median[2]),
        "h": float(median[3]),
        "confidence": float(median[4]),
        "center_std_px": center_std,
        "width_cv": width_cv,
        "frames_used": frames_required,
    }


def estimate_distance_mm(
    method, width_px, fx_px, target_size_mm, target, distance_models,
    fixed_z_mm=None
):
    if not isinstance(method, STRING_TYPES):
        raise LocalizationError("distance method must be text")
    method = method.strip().lower()
    width_px = finite_scalar(width_px, "width_px")
    fx_px = finite_scalar(fx_px, "fx_px")
    target_size_mm = finite_scalar(target_size_mm, "target_size_mm")
    if width_px <= 0.0 or fx_px <= 0.0 or target_size_mm <= 0.0:
        raise LocalizationError("width, focal length and target size must be positive")

    if method == "theory":
        return fx_px * target_size_mm / width_px
    if method == "calibrated":
        model = {}
        if isinstance(distance_models, dict):
            model = distance_models.get(target) or {}
        try:
            a = finite_scalar(model.get("a"), "%s distance calibration a" % target)
            b = finite_scalar(model.get("b"), "%s distance calibration b" % target)
        except LocalizationError as exc:
            raise LocalizationError("missing distance calibration: %s" % exc)
        if a <= 0.0:
            raise LocalizationError("distance calibration a must be positive")
        return a / width_px + b
    if method == "fixed_plane":
        fixed = finite_scalar(fixed_z_mm, "fixed_z_mm")
        if fixed <= 0.0:
            raise LocalizationError("fixed_z_mm must be positive")
        return fixed
    raise LocalizationError("unsupported distance method: %s" % method)


def deproject_pixel_to_camera_mm(u, v, z_mm, fx_px, fy_px, cx_px, cy_px):
    u = finite_scalar(u, "pixel u")
    v = finite_scalar(v, "pixel v")
    z_mm = finite_scalar(z_mm, "z_mm")
    fx_px = finite_scalar(fx_px, "fx_px")
    fy_px = finite_scalar(fy_px, "fy_px")
    cx_px = finite_scalar(cx_px, "cx_px")
    cy_px = finite_scalar(cy_px, "cy_px")
    if z_mm <= 0.0 or fx_px <= 0.0 or fy_px <= 0.0:
        raise LocalizationError("z and focal lengths must be positive")
    return ((u - cx_px) * z_mm / fx_px, (v - cy_px) * z_mm / fy_px, z_mm)


def merge_config(base, override):
    result = {}
    for key, value in base.items():
        if isinstance(value, dict):
            result[key] = merge_config(value, {})
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path=None):
    config = merge_config(DEFAULT_CONFIG, {})
    if path:
        if not os.path.isfile(path):
            raise LocalizationError("config file does not exist: %s" % path)
        try:
            import yaml
        except ImportError as exc:
            raise LocalizationError("PyYAML is required to read config: %s" % exc)
        with open(path, "r") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, dict):
            raise LocalizationError("config file must contain a YAML mapping")
        config = merge_config(config, loaded)
    return normalize_config(config)


def normalize_config(config):
    result = merge_config(DEFAULT_CONFIG, config or {})
    if "target_classes" not in result and "class_names" in result:
        result["target_classes"] = {}
        for index, target in enumerate(("power", "fire", "gas", "support")):
            result["target_classes"][target] = {
                "class_id": index,
                "class_name": result["class_names"].get(target, target),
            }
    for target, metadata in DEFAULT_TARGET_CLASSES.items():
        configured = result["target_classes"].setdefault(target, {})
        configured.setdefault("class_id", metadata["class_id"])
        configured.setdefault("class_name", metadata["class_name"])
    return result


def target_metadata(config, target):
    classes = (config or {}).get("target_classes") or DEFAULT_TARGET_CLASSES
    metadata = classes.get(target)
    if not isinstance(metadata, dict):
        raise LocalizationError("unknown target: %s" % target)
    class_id = metadata.get("class_id")
    if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
        raise LocalizationError("target class_id must be a non-negative integer")
    class_name = metadata.get("class_name", str(class_id))
    return {"class_id": class_id, "class_name": class_name}


def class_count_from_config(config):
    classes = (config or {}).get("target_classes") or DEFAULT_TARGET_CLASSES
    max_id = max(int(item["class_id"]) for item in classes.values())
    return max_id + 1


def select_target_detection(detections, target, config):
    metadata = target_metadata(config, target)
    matches = [
        item for item in detections
        if int(item.get("class_id", -1)) == metadata["class_id"]
    ]
    if not matches:
        raise LocalizationError("No detection matched target %s" % target)
    matches.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    best = dict(matches[0])
    best["class_name"] = metadata["class_name"]
    return best


def letterbox_image(image_bgr, input_size):
    try:
        import cv2
    except ImportError as exc:
        raise LocalizationError("OpenCV is required for ONNX preprocessing: %s" % exc)
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise LocalizationError("image must be a BGR uint8 HWC array")
    input_size = int(finite_scalar(input_size, "input_size"))
    if input_size <= 0:
        raise LocalizationError("input_size must be positive")
    height, width = image.shape[:2]
    scale = min(input_size / float(width), input_size / float(height))
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - resized_w) / 2.0
    pad_y = (input_size - resized_h) / 2.0
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top:top + resized_h, left:left + resized_w] = resized
    return canvas, scale, (float(left), float(top))


def yolo_blob(image_bgr, input_size):
    boxed, scale, pad = letterbox_image(image_bgr, input_size)
    rgb = boxed[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[None, :, :, :]
    return np.ascontiguousarray(blob), scale, pad


def _prediction_matrix(output, class_count):
    array = np.asarray(output)
    if isinstance(output, (list, tuple)) and len(output) == 1:
        array = np.asarray(output[0])
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise LocalizationError("YOLO output must be a 2D prediction matrix")
    expected = 5 + int(class_count)
    if array.shape[0] == expected and array.shape[1] != expected:
        array = array.T
    if array.shape[1] != expected:
        raise LocalizationError(
            "YOLOv5 output must have %d columns: cx,cy,w,h,obj,classes" % expected
        )
    return array.astype(np.float64, copy=False)


def _nms_indices(boxes, scores, nms_iou):
    if not boxes:
        return []
    try:
        import cv2
        xywh = []
        for x1, y1, x2, y2 in boxes:
            xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
        indices = cv2.dnn.NMSBoxes(xywh, scores, 0.0, float(nms_iou))
        flattened = np.asarray(indices).reshape(-1).tolist()
        return [int(index) for index in flattened]
    except Exception:
        return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)


def decode_yolov5_output(
    output,
    image_shape,
    input_shape,
    scale,
    pad,
    confidence_min,
    nms_iou,
    class_count,
):
    predictions = _prediction_matrix(output, class_count)
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    input_h, input_w = int(input_shape[0]), int(input_shape[1])
    if image_h <= 0 or image_w <= 0 or input_h <= 0 or input_w <= 0:
        raise LocalizationError("image/input shapes must be positive")
    scale = finite_scalar(scale, "letterbox scale")
    pad_x, pad_y = finite_vector(pad, "letterbox pad", 2).tolist()
    confidence_min = finite_scalar(confidence_min, "confidence_min")
    nms_iou = finite_scalar(nms_iou, "nms_iou")
    if scale <= 0.0:
        raise LocalizationError("letterbox scale must be positive")

    boxes = []
    scores = []
    class_ids = []
    for row in predictions:
        if not np.all(np.isfinite(row)):
            continue
        objectness = float(row[4])
        class_scores = row[5:5 + int(class_count)]
        class_id = int(np.argmax(class_scores))
        confidence = float(objectness * class_scores[class_id])
        if confidence < confidence_min:
            continue
        center_x, center_y, width, height = [float(value) for value in row[:4]]
        if width <= 0.0 or height <= 0.0:
            continue
        x1 = (center_x - width * 0.5 - pad_x) / scale
        y1 = (center_y - height * 0.5 - pad_y) / scale
        x2 = (center_x + width * 0.5 - pad_x) / scale
        y2 = (center_y + height * 0.5 - pad_y) / scale
        x1 = max(0.0, min(float(image_w), x1))
        y1 = max(0.0, min(float(image_h), y1))
        x2 = max(0.0, min(float(image_w), x2))
        y2 = max(0.0, min(float(image_h), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        scores.append(confidence)
        class_ids.append(class_id)

    detections = []
    for index in _nms_indices(boxes, scores, nms_iou):
        detections.append({
            "class_id": class_ids[index],
            "confidence": float(scores[index]),
            "box": [float(value) for value in boxes[index]],
        })
    return detections


class OnnxYoloDetector(object):
    def __init__(self, model_path, config):
        if not os.path.isfile(model_path):
            raise LocalizationError("ONNX model file does not exist: %s" % model_path)
        self.model_path = model_path
        self.config = normalize_config(config)
        self.input_size = int(self.config.get("input_size", 640))
        self.class_count = class_count_from_config(self.config)
        self._runtime = None
        self._session = None
        self._input_name = None
        self._load_backend()

    def _load_backend(self):
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self._runtime = "onnxruntime"
            return
        except Exception:
            pass
        try:
            import cv2
            self._session = cv2.dnn.readNetFromONNX(self.model_path)
            self._runtime = "opencv"
        except Exception as exc:
            raise LocalizationError("Could not load ONNX model: %s" % exc)

    def detect(self, image_bgr):
        image = np.asarray(image_bgr)
        blob, scale, pad = yolo_blob(image, self.input_size)
        if self._runtime == "onnxruntime":
            outputs = self._session.run(None, {self._input_name: blob})
            raw = outputs[0]
        else:
            self._session.setInput(blob)
            raw = self._session.forward()
        detections = decode_yolov5_output(
            raw,
            image.shape[:2],
            (self.input_size, self.input_size),
            scale,
            pad,
            self.config.get("confidence_min", 0.70),
            self.config.get("nms_iou", 0.45),
            self.class_count,
        )
        class_names = {}
        for metadata in self.config.get("target_classes", {}).values():
            class_names[int(metadata["class_id"])] = metadata["class_name"]
        for detection in detections:
            detection["class_name"] = class_names.get(
                int(detection["class_id"]), str(detection["class_id"])
            )
        return detections

    def detect_path(self, image_path):
        try:
            import cv2
        except ImportError as exc:
            raise LocalizationError("OpenCV is required to read image path: %s" % exc)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise LocalizationError("could not read image: %s" % image_path)
        return self.detect(image)


def draw_debug_detections(image_bgr, detections, observations=None, text_lines=None):
    try:
        import cv2
    except ImportError as exc:
        raise LocalizationError("OpenCV is required for debug display: %s" % exc)
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise LocalizationError("debug image must be BGR uint8")
    output = image.copy()
    if isinstance(detections, dict):
        detections = [detections]
    detections = list(detections or [])
    if observations is None:
        observations = [None] * len(detections)
    elif isinstance(observations, dict):
        observations = [observations]
    else:
        observations = list(observations)
    while len(observations) < len(detections):
        observations.append(None)

    colors = [
        (0, 255, 0),
        (0, 128, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            continue
        color = colors[index % len(colors)]
        box = [int(round(value)) for value in detection.get("box", [])]
        obs = observations[index] or detection_to_observation(detection)
        center = (int(round(obs["u"])), int(round(obs["v"])))
        if len(box) == 4:
            cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.drawMarker(output, center, (255, 0, 0), cv2.MARKER_CROSS, 18, 2)
        label = "%s %.2f w=%.1f" % (
            detection.get("class_name", detection.get("class_id", "?")),
            float(detection.get("confidence", obs.get("confidence", 0.0))),
            obs["w"],
        )
        cv2.putText(
            output,
            label,
            (max(0, box[0] if len(box) == 4 else center[0]), max(20, center[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    for index, line in enumerate(text_lines or []):
        cv2.putText(
            output, str(line), (8, 24 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA
        )
    return output


def draw_debug_image(image_bgr, detection, observation=None, text_lines=None):
    return draw_debug_detections(
        image_bgr, [detection], [observation] if observation is not None else None,
        text_lines=text_lines)
