from __future__ import absolute_import, division, print_function

import io
import json
import os

import numpy as np


try:
    STRING_TYPES = (basestring,)
except NameError:
    STRING_TYPES = (str,)


class LocalizationError(RuntimeError):
    pass


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config",
    "block_mono_grasp.yaml",
)


def _read_yaml_mapping(path):
    if not os.path.isfile(path):
        raise LocalizationError("config file does not exist: %s" % path)
    try:
        import yaml
    except ImportError as exc:
        raise LocalizationError("PyYAML is required to read config: %s" % exc)
    # Windows 的系统默认编码不是 UTF-8；配置含中文注释时必须显式指定。
    with io.open(path, "r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise LocalizationError("config file must contain a YAML mapping")
    return loaded


# All adjustable defaults live in this YAML file. Both the Python 3 detector
# and Python 2 ROS child import this same mapping.
DEFAULT_CONFIG = _read_yaml_mapping(DEFAULT_CONFIG_PATH)
DEFAULT_TARGET_CLASSES = DEFAULT_CONFIG["target_classes"]


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


def roi_box_pixels(image_shape, roi_ratio):
    if len(image_shape) < 2:
        raise LocalizationError("image shape must contain height and width")
    height = int(image_shape[0])
    width = int(image_shape[1])
    if height <= 0 or width <= 0:
        raise LocalizationError("image dimensions must be positive")
    x1, y1, x2, y2 = finite_vector(
        roi_ratio, "grasp_roi_ratio", 4).tolist()
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise LocalizationError(
            "grasp_roi_ratio must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1")
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def observation_in_roi(observation, image_shape, roi_ratio):
    x1, y1, x2, y2 = roi_box_pixels(image_shape, roi_ratio)
    u = finite_scalar(observation.get("u"), "observation u")
    v = finite_scalar(observation.get("v"), "observation v")
    if x1 <= u <= x2 and y1 <= v <= y2:
        return True, ""
    return False, (
        "center (%.1f, %.1f) outside grasp ROI [%d, %d, %d, %d]"
        % (u, v, x1, y1, x2, y2))


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
    observations, frames_required, center_std_max_px, width_cv_max,
    mad_scale=3.5, mad_floor_px=1.0
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

    recent = observations[-max(frames_required * 3, frames_required):]
    matrix = np.asarray(
        [[item["u"], item["v"], item["w"], item["h"], item["confidence"]] for item in recent],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[1] != 5 or not np.all(np.isfinite(matrix)):
        raise LocalizationError("observations must contain finite u/v/w/h/confidence")

    mad_scale = finite_scalar(mad_scale, "mad_scale")
    mad_floor_px = finite_scalar(mad_floor_px, "mad_floor_px")
    if mad_scale <= 0.0 or mad_floor_px <= 0.0:
        raise LocalizationError("MAD limits must be positive")
    median_geometry = np.median(matrix[:, :4], axis=0)
    axis_mad = np.median(np.abs(matrix[:, :4] - median_geometry), axis=0)
    limits = np.maximum(axis_mad * mad_scale, mad_floor_px)
    inlier_mask = np.all(
        np.abs(matrix[:, :4] - median_geometry) <= limits,
        axis=1,
    )
    inliers = matrix[inlier_mask]
    if len(inliers) < frames_required:
        raise LocalizationError(
            "only %d stable inliers after MAD filtering, need %d"
            % (len(inliers), frames_required)
        )
    inliers = inliers[-frames_required:]

    center_std = max(float(np.std(inliers[:, 0])), float(np.std(inliers[:, 1])))
    width_mean = float(np.mean(inliers[:, 2]))
    if width_mean <= 0.0:
        raise LocalizationError("width mean must be positive")
    width_cv = float(np.std(inliers[:, 2]) / width_mean)
    if center_std > center_std_max_px:
        raise LocalizationError(
            "center std %.3f px exceeds %.3f px" % (center_std, center_std_max_px)
        )
    if width_cv > width_cv_max:
        raise LocalizationError(
            "width cv %.5f exceeds %.5f" % (width_cv, width_cv_max)
        )

    median = np.median(inliers, axis=0)
    return {
        "u": float(median[0]),
        "v": float(median[1]),
        "w": float(median[2]),
        "h": float(median[3]),
        "confidence": float(median[4]),
        "center_std_px": center_std,
        "width_cv": width_cv,
        "frames_used": frames_required,
        "sample_count": int(len(matrix)),
        "inlier_count": int(np.count_nonzero(inlier_mask)),
        "axis_mad_px": [float(value) for value in axis_mad.tolist()],
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


def _calibrated_axis_distance(model, axis, pixels):
    axis_model = model.get(axis)
    if isinstance(axis_model, dict):
        a_value = axis_model.get("a")
        b_value = axis_model.get("b")
    elif axis == "width":
        # Backward compatibility with the original {a, b} width-only model.
        a_value = model.get("a")
        b_value = model.get("b")
    else:
        return None
    try:
        a_value = finite_scalar(a_value, "%s calibration a" % axis)
        b_value = finite_scalar(b_value, "%s calibration b" % axis)
    except LocalizationError:
        return None
    if a_value <= 0.0:
        return None
    return a_value / pixels + b_value


def estimate_distance_from_box_mm(
    method, width_px, height_px, fx_px, fy_px,
    target_width_mm, target_height_mm, target, distance_models,
    fixed_z_mm=None, max_axis_disagreement_mm=None
):
    """Estimate optical depth without a depth image.

    Calibrated mode combines independent width and height models. Set
    max_axis_disagreement_mm to zero to accept their median without a gate.
    """
    method = str(method).strip().lower()
    width_px = finite_scalar(width_px, "width_px")
    height_px = finite_scalar(height_px, "height_px")
    if width_px <= 0.0 or height_px <= 0.0:
        raise LocalizationError("box width and height must be positive")
    if method == "fixed_plane":
        return estimate_distance_mm(
            method, width_px, fx_px, target_width_mm, target,
            distance_models, fixed_z_mm)
    if method == "theory":
        width_distance = finite_scalar(fx_px, "fx_px") * finite_scalar(
            target_width_mm, "target_width_mm") / width_px
        height_distance = finite_scalar(fy_px, "fy_px") * finite_scalar(
            target_height_mm, "target_height_mm") / height_px
        return float(np.median([width_distance, height_distance]))
    if method != "calibrated":
        raise LocalizationError("unsupported distance method: %s" % method)

    model = (distance_models or {}).get(target) or {}
    estimates = []
    width_distance = _calibrated_axis_distance(model, "width", width_px)
    height_distance = _calibrated_axis_distance(model, "height", height_px)
    if width_distance is not None:
        estimates.append(width_distance)
    if height_distance is not None:
        estimates.append(height_distance)
    if not estimates:
        raise LocalizationError(
            "missing distance calibration for %s; provide width and/or height models"
            % target
        )
    if len(estimates) == 2:
        disagreement = abs(estimates[0] - estimates[1])
        if max_axis_disagreement_mm is None:
            max_axis_disagreement_mm = DEFAULT_CONFIG[
                "max_axis_distance_disagreement_mm"]
        limit = finite_scalar(
            max_axis_disagreement_mm, "max_axis_disagreement_mm")
        if limit < 0.0:
            raise LocalizationError(
                "max_axis_disagreement_mm must be non-negative")
        if limit > 0.0 and disagreement > limit:
            raise LocalizationError(
                "width/height distance disagreement %.2f mm exceeds %.2f mm "
                "(box_width=%.2f px, box_height=%.2f px, "
                "width_distance=%.2f mm, height_distance=%.2f mm)"
                % (
                    disagreement,
                    limit,
                    width_px,
                    height_px,
                    width_distance,
                    height_distance,
                )
            )
    return float(np.median(estimates))


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


def scale_box_width_for_distance(width_px, image_width,
                                 distance_model_frame_width=None):
    width_px = finite_scalar(width_px, "width_px")
    image_width = finite_scalar(image_width, "image_width")
    if width_px <= 0.0 or image_width <= 0.0:
        raise LocalizationError("box/image width must be positive")
    if distance_model_frame_width is None:
        return width_px
    model_width = finite_scalar(
        distance_model_frame_width, "distance_model_frame_width")
    if model_width <= 0.0:
        raise LocalizationError("distance_model_frame_width must be positive")
    return width_px * model_width / image_width


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
        config = merge_config(config, _read_yaml_mapping(path))
    return load_external_distance_calibration(normalize_config(config))


def load_external_distance_calibration(config):
    path = config.get("distance_calibration_file")
    if not path:
        return config
    try:
        with io.open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (IOError, OSError, ValueError) as exc:
        raise LocalizationError(
            "could not read distance_calibration_file: %s" % exc)
    if payload.get("version") != 3:
        raise LocalizationError(
            "distance_calibration_file version 3 is required")
    frame_width = int(payload.get("frame_width", 0))
    if frame_width <= 0 or not isinstance(payload.get("targets"), dict):
        raise LocalizationError("distance calibration dimensions/targets are invalid")
    models = {}
    ranges = {}
    for target, metadata in config["target_classes"].items():
        entry = payload["targets"].get(str(int(metadata["target_id"])))
        if not isinstance(entry, dict):
            raise LocalizationError(
                "distance calibration is missing target %s" % target)
        if str(entry.get("class_name")) != str(metadata["class_name"]):
            raise LocalizationError(
                "distance calibration class mismatch for target %s" % target)
        width_model = entry.get("width")
        if not isinstance(width_model, dict):
            raise LocalizationError(
                "distance calibration is missing width model for %s" % target)
        a_value = finite_scalar(width_model.get("a"), "%s width.a" % target)
        b_value = finite_scalar(width_model.get("b"), "%s width.b" % target)
        minimum = finite_scalar(
            entry.get("min_distance_mm"), "%s min_distance_mm" % target)
        maximum = finite_scalar(
            entry.get("max_distance_mm"), "%s max_distance_mm" % target)
        if a_value <= 0.0 or minimum <= 0.0 or maximum <= minimum:
            raise LocalizationError(
                "distance calibration values are invalid for %s" % target)
        models[target] = {"width": {"a": a_value, "b": b_value}}
        ranges[target] = [minimum, maximum]
    config["distance_models"] = models
    config["distance_model_frame_width"] = frame_width
    config["distance_ranges_mm"] = ranges
    return config


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
        configured.setdefault("target_id", metadata["target_id"])
        configured.setdefault("class_id", metadata["class_id"])
        configured.setdefault("class_name", metadata["class_name"])
    return result


def target_aliases(config=None):
    aliases = {}
    classes = normalize_config(config or DEFAULT_CONFIG)["target_classes"]
    for target, metadata in classes.items():
        target_name = str(target)
        try:
            target_id = int(metadata.get("target_id"))
        except (TypeError, ValueError, OverflowError):
            raise LocalizationError("target_id for %s must be a positive integer" % target)
        if target_id <= 0:
            raise LocalizationError("target_id for %s must be a positive integer" % target)
        alias = str(target_id)
        if alias in aliases and aliases[alias] != target_name:
            raise LocalizationError("duplicate target_id: %s" % alias)
        aliases[target_name] = target_name
        aliases[alias] = target_name
    return aliases


def resolve_target_alias(value, config=None):
    alias = str(value).strip()
    target = target_aliases(config).get(alias)
    if target is None:
        raise LocalizationError("unknown target or target ID: %s" % alias)
    return target


def parse_target_sequence(text, config=None):
    if not isinstance(text, STRING_TYPES) or not text.strip():
        raise LocalizationError("target sequence must be comma separated IDs or names")
    targets = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        target = resolve_target_alias(item, config)
        if target in targets:
            raise LocalizationError("target sequence contains a duplicate: %s" % item)
        targets.append(target)
    if not targets:
        raise LocalizationError("target sequence must contain at least one target")
    return targets


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
        self.input_size = int(self.config["input_size"])
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
            self.config["confidence_min"],
            self.config["nms_iou"],
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


def draw_debug_detections(
        image_bgr, detections, observations=None, text_lines=None,
        roi_ratio=None):
    try:
        import cv2
    except ImportError as exc:
        raise LocalizationError("OpenCV is required for debug display: %s" % exc)
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise LocalizationError("debug image must be BGR uint8")
    output = image.copy()
    if roi_ratio is not None:
        roi = roi_box_pixels(output.shape, roi_ratio)
        cv2.rectangle(output, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)
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
    abbreviations = {
        "power": "POW",
        "fire": "FIR",
        "gas": "GAS",
        "support": "SUP",
    }
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
        target = str(detection.get(
            "target", detection.get("class_name", detection.get("class_id", "?"))))
        short_name = abbreviations.get(target.lower(), target[:3].upper() or "?")
        confidence_percent = int(round(
            100.0 * float(detection.get("confidence", obs.get("confidence", 0.0)))))
        label = "%s%d" % (short_name, confidence_percent)
        label_x = max(0, box[0] if len(box) == 4 else center[0])
        label_y = max(12, (box[1] - 5) if len(box) == 4 else center[1] - 12)
        cv2.putText(
            output,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    for index, line in enumerate(text_lines or []):
        cv2.putText(
            output, str(line), (8, 24 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA
        )
    return output


def draw_debug_image(
        image_bgr, detection, observation=None, text_lines=None,
        roi_ratio=None):
    return draw_debug_detections(
        image_bgr, [detection], [observation] if observation is not None else None,
        text_lines=text_lines, roi_ratio=roi_ratio)
