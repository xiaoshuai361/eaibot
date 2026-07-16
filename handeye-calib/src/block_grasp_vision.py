from __future__ import absolute_import, division, print_function

import math

import cv2
import numpy as np


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
    if encoding not in ("16UC1", "MONO16", "32FC1"):
        raise LocalizationError("unsupported depth encoding: %s" % encoding)

    center_values = _finite_vector(center, "depth patch center", 2)
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

    center_x = int(round(float(center_values[0])))
    center_y = int(round(float(center_values[1])))
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

    valid_mask = (
        np.isfinite(patch) & (patch >= min_depth_m) & (patch <= max_depth_m)
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
