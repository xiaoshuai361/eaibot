from __future__ import absolute_import, division, print_function

import math

import cv2
import numpy as np


try:
    _STRING_TYPES = (basestring,)
except NameError:
    _STRING_TYPES = (str,)


class LocalizationError(RuntimeError):
    pass


def _isfinite(value):
    return bool(np.isfinite(value))


def _finite_scalar(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise LocalizationError("%s must be a finite number" % name)
    if not _isfinite(result):
        raise LocalizationError("%s must be a finite number" % name)
    return result


def _finite_vector(values, name="vector", expected_length=None):
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


def _nonzero_unit_vector(values, name):
    vector = _finite_vector(values, name, 3)
    length = float(np.linalg.norm(vector))
    if not _isfinite(length) or length <= 0.0:
        raise LocalizationError("%s must be non-zero" % name)
    return vector / length


def _validate_bgr_image(image_bgr):
    if not isinstance(image_bgr, np.ndarray):
        raise LocalizationError("image must be a numpy array")
    if (
        image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.shape[0] <= 0
        or image_bgr.shape[1] <= 0
    ):
        raise LocalizationError("image must be a non-empty HWC BGR uint8 array")
    return image_bgr


def _validate_detector_box(detector_box):
    box = _finite_vector(detector_box, "detector box", 4)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise LocalizationError("detector box must satisfy x2 > x1 and y2 > y1")
    return box


def detector_box_center(detector_box):
    box = _validate_detector_box(detector_box)
    center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
    return tuple(_finite_vector(center, "detector box center", 2).tolist())


def render_yolo_center_debug_image(image_bgr, detector_box, center, depth_radius):
    image = _validate_bgr_image(image_bgr)
    box = _validate_detector_box(detector_box)
    center = _finite_vector(center, "YOLO box center", 2)
    depth_radius = _finite_scalar(depth_radius, "depth radius")
    if depth_radius < 0.0 or depth_radius != math.floor(depth_radius):
        raise LocalizationError("depth radius must be a non-negative integer")
    depth_radius = int(depth_radius)

    output = image.copy()
    box_points = np.rint(box).astype(np.int32)
    center_x = int(math.floor(center[0] + 0.5))
    center_y = int(math.floor(center[1] + 0.5))
    cv2.rectangle(
        output,
        (box_points[0], box_points[1]),
        (box_points[2], box_points[3]),
        (0, 255, 0),
        2,
    )
    cv2.drawMarker(
        output,
        (center_x, center_y),
        (255, 0, 0),
        cv2.MARKER_CROSS,
        18,
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        output,
        (center_x - depth_radius, center_y - depth_radius),
        (center_x + depth_radius, center_y + depth_radius),
        (255, 255, 0),
        1,
    )
    return output


def render_depth_debug_image(
    depth_image, center, encoding, min_depth_m, max_depth_m, depth_radius
):
    if not isinstance(encoding, _STRING_TYPES):
        raise LocalizationError("depth encoding must be a string")
    encoding = encoding.upper()
    if encoding not in ("16UC1", "MONO16", "32FC1"):
        raise LocalizationError("unsupported depth encoding: %s" % encoding)

    center = _finite_vector(center, "depth debug center", 2)
    if np.any(center < 0.0):
        raise LocalizationError("depth debug center must use non-negative pixels")
    depth_radius = _finite_scalar(depth_radius, "depth radius")
    if depth_radius < 0.0 or depth_radius != math.floor(depth_radius):
        raise LocalizationError("depth radius must be a non-negative integer")
    depth_radius = int(depth_radius)
    min_depth_m = _finite_scalar(min_depth_m, "minimum depth")
    max_depth_m = _finite_scalar(max_depth_m, "maximum depth")
    if min_depth_m <= 0.0 or max_depth_m <= min_depth_m:
        raise LocalizationError("depth range must be positive and increasing")

    try:
        depth = np.asarray(depth_image)
    except (TypeError, ValueError):
        raise LocalizationError("depth image must be a two-dimensional array")
    if depth.ndim != 2 or depth.shape[0] <= 0 or depth.shape[1] <= 0:
        raise LocalizationError("depth image must be a non-empty two-dimensional array")
    if encoding in ("16UC1", "MONO16") and depth.dtype != np.uint16:
        raise LocalizationError("16UC1/mono16 depth must use uint16 dtype")
    if encoding == "32FC1" and depth.dtype != np.float32:
        raise LocalizationError("32FC1 depth must use float32 dtype")

    center_x = int(math.floor(float(center[0]) + 0.5))
    center_y = int(math.floor(float(center[1]) + 0.5))
    height, width = depth.shape
    if center_x >= width or center_y >= height:
        raise LocalizationError("depth debug center is outside the image")

    depth_m = depth.astype(np.float64, copy=False)
    if encoding in ("16UC1", "MONO16"):
        depth_m = depth_m * 0.001
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= min_depth_m)
        & (depth_m <= max_depth_m)
    )
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    scale = 255.0 / (max_depth_m - min_depth_m)
    normalized[valid] = np.clip(
        (depth_m[valid] - min_depth_m) * scale, 0.0, 255.0
    ).astype(np.uint8)

    output = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    output[~valid] = (0, 0, 0)
    cv2.rectangle(
        output,
        (center_x - depth_radius, center_y - depth_radius),
        (center_x + depth_radius, center_y + depth_radius),
        (255, 255, 0),
        1,
    )
    cv2.circle(output, (center_x, center_y), 1, (255, 0, 0), -1)
    return output


def sample_depth_m(
    depth_image,
    center,
    encoding,
    radius,
    min_depth_m,
    max_depth_m,
    min_valid_ratio,
    max_mad_m,
):
    if not isinstance(encoding, _STRING_TYPES):
        raise LocalizationError("depth encoding must be a string")
    encoding = encoding.upper()
    if encoding not in ("16UC1", "MONO16", "32FC1"):
        raise LocalizationError("unsupported depth encoding: %s" % encoding)

    center_values = _finite_vector(center, "depth patch center", 2)
    if np.any(center_values < 0.0):
        raise LocalizationError("depth patch center must use non-negative pixels")
    radius_value = _finite_scalar(radius, "depth patch radius")
    if radius_value < 0.0 or radius_value != math.floor(radius_value):
        raise LocalizationError("depth patch radius must be a non-negative integer")
    radius_value = int(radius_value)

    min_depth_m = _finite_scalar(min_depth_m, "minimum depth")
    max_depth_m = _finite_scalar(max_depth_m, "maximum depth")
    min_valid_ratio = _finite_scalar(min_valid_ratio, "minimum valid ratio")
    max_mad_m = _finite_scalar(max_mad_m, "maximum depth MAD")
    if min_depth_m <= 0.0 or max_depth_m <= min_depth_m:
        raise LocalizationError("depth range must be positive and increasing")
    if min_valid_ratio <= 0.0 or min_valid_ratio > 1.0:
        raise LocalizationError("minimum valid ratio must be in (0, 1]")
    if max_mad_m < 0.0:
        raise LocalizationError("maximum depth MAD must be non-negative")

    try:
        depth = np.asarray(depth_image)
    except (TypeError, ValueError):
        raise LocalizationError("depth image must be a two-dimensional array")
    if depth.ndim != 2:
        raise LocalizationError("depth image must be a two-dimensional array")

    center_x = int(math.floor(float(center_values[0]) + 0.5))
    center_y = int(math.floor(float(center_values[1]) + 0.5))
    height, width = depth.shape
    x_start = max(0, center_x - radius_value)
    x_stop = min(width, center_x + radius_value + 1)
    y_start = max(0, center_y - radius_value)
    y_stop = min(height, center_y + radius_value + 1)
    if x_start >= x_stop or y_start >= y_stop:
        raise LocalizationError("depth patch is empty or outside the image")

    try:
        patch = depth[y_start:y_stop, x_start:x_stop].astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError):
        raise LocalizationError("depth patch must contain numeric values")
    if patch.size == 0:
        raise LocalizationError("depth patch is empty")
    if encoding in ("16UC1", "MONO16"):
        patch = patch * 0.001

    finite_mask = np.isfinite(patch)
    valid_mask = np.zeros(patch.shape, dtype=np.bool_)
    valid_mask[finite_mask] = (
        (patch[finite_mask] >= min_depth_m)
        & (patch[finite_mask] <= max_depth_m)
    )
    valid_count = int(np.count_nonzero(valid_mask))
    valid_ratio = valid_count / float(patch.size)
    if valid_count == 0 or valid_ratio < min_valid_ratio:
        raise LocalizationError(
            "depth patch valid ratio %.3f is below %.3f"
            % (valid_ratio, min_valid_ratio)
        )

    valid_depths = patch[valid_mask]
    median_m = float(np.median(valid_depths))
    mad_m = float(np.median(np.abs(valid_depths - median_m)))
    if not _isfinite(median_m) or not _isfinite(mad_m):
        raise LocalizationError("depth statistics are not finite")
    if mad_m > max_mad_m:
        raise LocalizationError(
            "depth MAD %.6f m exceeds %.6f m" % (mad_m, max_mad_m)
        )
    return median_m, {"valid_ratio": valid_ratio, "mad_m": mad_m}


def deproject_pixel(u, v, depth_m, fx, fy, cx, cy):
    u = _finite_scalar(u, "pixel u")
    v = _finite_scalar(v, "pixel v")
    depth_m = _finite_scalar(depth_m, "depth")
    fx = _finite_scalar(fx, "focal length fx")
    fy = _finite_scalar(fy, "focal length fy")
    cx = _finite_scalar(cx, "principal point cx")
    cy = _finite_scalar(cy, "principal point cy")
    if depth_m <= 0.0:
        raise LocalizationError("depth must be positive")
    if fx <= 0.0 or fy <= 0.0:
        raise LocalizationError("focal lengths must be positive")
    point = ((u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m)
    return tuple(_finite_vector(point, "deprojected point", 3).tolist())


def undistort_pixel(u, v, camera_matrix, distortion, distortion_model):
    u = _finite_scalar(u, "pixel u")
    v = _finite_scalar(v, "pixel v")
    if distortion_model not in ("", "plumb_bob"):
        raise LocalizationError(
            "unsupported distortion model: %s" % distortion_model
        )

    try:
        matrix = np.asarray(camera_matrix, dtype=np.float64)
        coefficients = np.asarray(distortion, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise LocalizationError("camera calibration must contain finite numbers")
    if matrix.shape != (3, 3):
        raise LocalizationError("camera matrix must have shape (3, 3)")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(coefficients)):
        raise LocalizationError("camera calibration must contain finite numbers")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise LocalizationError("camera focal lengths must be positive")

    points = np.array([[[u, v]]], dtype=np.float64)
    distortion_arg = coefficients.reshape(-1) if coefficients.size else None
    try:
        corrected = cv2.undistortPoints(points, matrix, distortion_arg, P=matrix)
    except cv2.error as error:
        raise LocalizationError("could not undistort pixel: %s" % error)
    corrected_vector = _finite_vector(
        np.asarray(corrected).reshape(-1), "undistorted pixel", 2
    )
    return tuple(corrected_vector.tolist())


def rotate_vector_by_quaternion(vector, quaternion_xyzw):
    vector = _finite_vector(vector, "vector", 3)
    quaternion = _finite_vector(quaternion_xyzw, "quaternion", 4)
    quaternion_length = float(np.linalg.norm(quaternion))
    if not _isfinite(quaternion_length) or quaternion_length <= 0.0:
        raise LocalizationError("quaternion must be non-zero")
    quaternion = quaternion / quaternion_length

    quaternion_vector = quaternion[:3]
    quaternion_w = quaternion[3]
    intermediate = 2.0 * np.cross(quaternion_vector, vector)
    rotated = vector + quaternion_w * intermediate + np.cross(
        quaternion_vector, intermediate
    )
    if not np.all(np.isfinite(rotated)):
        raise LocalizationError("rotated vector is not finite")
    return tuple(rotated.tolist())


def compute_link_targets(surface_base, tcp_vector_base, approach_gap_m):
    surface = _finite_vector(surface_base, "surface point", 3)
    tcp_vector = _finite_vector(tcp_vector_base, "TCP vector", 3)
    tcp_length = float(np.linalg.norm(tcp_vector))
    if not _isfinite(tcp_length) or tcp_length <= 0.0 or tcp_length > 0.30:
        raise LocalizationError("TCP vector length must be in (0, 0.30] m")
    approach_gap_m = _finite_scalar(approach_gap_m, "approach gap")
    if approach_gap_m <= 0.0 or approach_gap_m > 0.15:
        raise LocalizationError("approach gap must be in (0, 0.15] m")

    axis = tcp_vector / tcp_length
    contact = surface - tcp_vector
    precontact = contact - axis * approach_gap_m
    return tuple(contact.tolist()), tuple(precontact.tolist())


def tool_axis_vector(axis_name, length_m):
    """Return the signed Link6-to-TCP vector expressed in Link6 coordinates."""
    if not isinstance(axis_name, _STRING_TYPES):
        raise LocalizationError("tool axis must be a string")
    length_m = _finite_scalar(length_m, "tool offset")
    if length_m < 0.0 or length_m > 0.30:
        raise LocalizationError("tool offset must be in [0, 0.30] m")
    axes = {
        "x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
    }
    if axis_name not in axes:
        raise LocalizationError("unsupported tool axis: %s" % axis_name)
    return tuple(component * length_m for component in axes[axis_name])


def validate_workspace_points(contact, precontact, minimum_z_m, maximum_radius_m):
    minimum_z_m = _finite_scalar(minimum_z_m, "minimum base z")
    maximum_radius_m = _finite_scalar(maximum_radius_m, "maximum base radius")
    if minimum_z_m < 0.0:
        raise LocalizationError("minimum base z must be non-negative")
    if maximum_radius_m <= 0.0:
        raise LocalizationError("maximum base radius must be positive")

    validated = []
    for label, point in (("contact", contact), ("precontact", precontact)):
        point = _finite_vector(point, "%s point" % label, 3)
        if point[2] < minimum_z_m:
            raise LocalizationError("%s point is below minimum z" % label)
        if math.hypot(float(point[0]), float(point[1])) > maximum_radius_m:
            raise LocalizationError("%s point exceeds maximum radius" % label)
        validated.append(tuple(point.tolist()))
    return tuple(validated)


def validate_rgbd_metadata(
    rgb,
    depth,
    rgb_header,
    depth_header,
    camera_info,
    depth_encoding,
    slop,
    stamp_to_sec,
):
    """Validate that one RGB/depth pair is truly registered and calibrated."""
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.shape[0] <= 0
        or rgb.shape[1] <= 0
    ):
        raise LocalizationError("RGB image must be a non-empty HWC BGR uint8 array")
    if depth.ndim != 2 or depth.shape[0] <= 0 or depth.shape[1] <= 0:
        raise LocalizationError("depth image must be a non-empty single-channel array")
    if depth.shape != rgb.shape[:2]:
        raise LocalizationError("registered depth and RGB dimensions must match")

    if not isinstance(depth_encoding, _STRING_TYPES):
        raise LocalizationError("depth encoding must be text")
    encoding = depth_encoding.upper()
    if encoding in ("16UC1", "MONO16"):
        if depth.dtype != np.uint16:
            raise LocalizationError("16UC1/mono16 depth must use uint16 dtype")
    elif encoding == "32FC1":
        if depth.dtype != np.float32:
            raise LocalizationError("32FC1 depth must use float32 dtype")
    else:
        raise LocalizationError("unsupported depth encoding: %s" % depth_encoding)

    slop = _finite_scalar(slop, "RGB-D synchronization slop")
    if slop < 0.0:
        raise LocalizationError("RGB-D synchronization slop must be non-negative")
    info_header = getattr(camera_info, "header", None)
    try:
        stamp_values = (
            ("RGB", stamp_to_sec(rgb_header.stamp)),
            ("depth", stamp_to_sec(depth_header.stamp)),
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        raise LocalizationError(
            "RGB and depth headers must contain timestamps"
        )
    validated_stamps = {}
    for label, value in stamp_values:
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise LocalizationError(
                "%s timestamp must be finite and positive" % label
            )
        if not _isfinite(value) or value <= 0.0:
            raise LocalizationError(
                "%s timestamp must be finite and positive" % label
            )
        validated_stamps[label] = value
    rgb_stamp = validated_stamps["RGB"]
    depth_stamp = validated_stamps["depth"]
    if abs(rgb_stamp - depth_stamp) > slop:
        raise LocalizationError("RGB/depth timestamp delta exceeds synchronization slop")

    rgb_frame = getattr(rgb_header, "frame_id", "")
    depth_frame = getattr(depth_header, "frame_id", "")
    info_frame = getattr(info_header, "frame_id", "")
    if not rgb_frame or rgb_frame != depth_frame or rgb_frame != info_frame:
        raise LocalizationError(
            "registered RGB, depth and CameraInfo must use the same frame; "
            "verify the real camera driver publishes registered depth"
        )

    try:
        info_width = int(camera_info.width)
        info_height = int(camera_info.height)
    except (AttributeError, TypeError, ValueError, OverflowError):
        raise LocalizationError("CameraInfo dimensions are invalid")
    if info_width != rgb.shape[1] or info_height != rgb.shape[0]:
        raise LocalizationError("CameraInfo dimensions must match RGB/depth")

    matrix = _finite_vector(getattr(camera_info, "K", []), "camera matrix", 9)
    distortion = _finite_vector(
        getattr(camera_info, "D", []), "camera distortion"
    )
    if matrix[0] <= 0.0 or matrix[4] <= 0.0:
        raise LocalizationError("camera focal lengths must be positive")
    return {
        "fx": float(matrix[0]),
        "fy": float(matrix[4]),
        "cx": float(matrix[2]),
        "cy": float(matrix[5]),
        "K": tuple(matrix.tolist()),
        "D": tuple(distortion.tolist()),
        "distortion_model": getattr(camera_info, "distortion_model", ""),
        "encoding": encoding,
    }


def validate_axis_alignment(
    suction_axis_base, camera_forward_base, max_angle_deg
):
    suction_axis = _nonzero_unit_vector(suction_axis_base, "suction axis")
    camera_forward = _nonzero_unit_vector(camera_forward_base, "camera forward axis")
    max_angle_deg = _finite_scalar(max_angle_deg, "maximum alignment angle")
    if max_angle_deg <= 0.0 or max_angle_deg >= 90.0:
        raise LocalizationError("maximum alignment angle must be in (0, 90) degrees")

    dot_product = float(np.dot(suction_axis, camera_forward))
    dot_product = max(-1.0, min(1.0, dot_product))
    angle_deg = math.degrees(math.acos(dot_product))
    if not _isfinite(angle_deg):
        raise LocalizationError("axis alignment angle is not finite")
    if angle_deg > max_angle_deg:
        raise LocalizationError(
            "axis alignment %.3f degrees exceeds %.3f degrees"
            % (angle_deg, max_angle_deg)
        )
    return angle_deg
