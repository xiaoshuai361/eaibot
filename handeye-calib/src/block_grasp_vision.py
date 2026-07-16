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


def _polygon_iou(first, second):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    first_area = abs(float(cv2.contourArea(first)))
    second_area = abs(float(cv2.contourArea(second)))
    if first_area <= 0.0 or second_area <= 0.0:
        return 0.0
    try:
        intersection_area = float(cv2.intersectConvexConvex(first, second)[0])
    except cv2.error:
        return 0.0
    union_area = first_area + second_area - intersection_area
    if union_area <= 0.0:
        return 0.0
    return max(0.0, min(1.0, intersection_area / union_area))


def _candidate_score(corners, center, area, rectangularity, detector_box):
    x1, y1, x2, y2 = detector_box
    detector_center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    detector_polygon = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
    )
    detector_area = float((x2 - x1) * (y2 - y1))
    try:
        intersection_area = float(
            cv2.intersectConvexConvex(
                np.asarray(corners, dtype=np.float32), detector_polygon
            )[0]
        )
    except cv2.error:
        intersection_area = 0.0
    detector_coverage = max(0.0, min(1.0, intersection_area / detector_area))
    contains_center = 1.0 if cv2.pointPolygonTest(
        np.asarray(corners, dtype=np.float32), detector_center, False
    ) >= 0 else 0.0

    corner_array = np.asarray(corners, dtype=np.float64)
    extent = max(
        float(np.ptp(corner_array[:, 0])),
        float(np.ptp(corner_array[:, 1])),
        1.0,
    )
    center_distance = math.hypot(
        center[0] - detector_center[0], center[1] - detector_center[1]
    )
    center_closeness = max(0.0, 1.0 - center_distance / extent)
    # YOLO often encloses only the printed artwork. Reward a surrounding surface,
    # but saturate quickly so a very large background rectangle cannot dominate.
    scale_score = min(1.0, math.sqrt(area / detector_area) / 2.0)
    score = (
        0.30 * contains_center
        + 0.25 * detector_coverage
        + 0.15 * center_closeness
        + 0.15 * rectangularity
        + 0.15 * scale_score
    )
    return float(score)


def find_block_quadrilateral(
    image_bgr,
    detector_box,
    roi_margin,
    min_area_pixels,
    max_aspect_error,
    min_rectangularity,
    ambiguity_ratio,
):
    image = _validate_bgr_image(image_bgr)
    box = _validate_detector_box(detector_box)
    roi_margin = _finite_scalar(roi_margin, "ROI margin")
    min_area_pixels = _finite_scalar(min_area_pixels, "minimum contour area")
    max_aspect_error = _finite_scalar(max_aspect_error, "maximum aspect error")
    min_rectangularity = _finite_scalar(
        min_rectangularity, "minimum rectangularity"
    )
    ambiguity_ratio = _finite_scalar(ambiguity_ratio, "ambiguity ratio")
    if roi_margin < 0.0 or roi_margin > 2.0:
        raise LocalizationError("ROI margin must be in [0, 2]")
    if min_area_pixels <= 0.0:
        raise LocalizationError("minimum contour area must be positive")
    if max_aspect_error < 0.0 or max_aspect_error > 1.0:
        raise LocalizationError("maximum aspect error must be in [0, 1]")
    if min_rectangularity <= 0.0 or min_rectangularity > 1.0:
        raise LocalizationError("minimum rectangularity must be in (0, 1]")
    if ambiguity_ratio <= 0.0 or ambiguity_ratio > 1.0:
        raise LocalizationError("ambiguity ratio must be in (0, 1]")

    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    expansion = roi_margin * max(box_width, box_height)
    image_height, image_width = image.shape[:2]
    roi_x1 = max(0, int(math.floor(box[0] - expansion)))
    roi_y1 = max(0, int(math.floor(box[1] - expansion)))
    roi_x2 = min(image_width, int(math.ceil(box[2] + expansion)))
    roi_y2 = min(image_height, int(math.ceil(box[3] + expansion)))
    if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
        raise LocalizationError("expanded detector ROI is empty")

    roi_image = image[roi_y1:roi_y2, roi_x1:roi_x2]
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    if int(np.max(gray)) - int(np.min(gray)) < 5:
        raise LocalizationError("no reliable white quadrilateral candidate")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), dtype=np.uint8)
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    candidates = []
    offset = np.array([roi_x1, roi_y1], dtype=np.float64)
    for contour in contours:
        perimeter = float(cv2.arcLength(contour, True))
        if not _isfinite(perimeter) or perimeter <= 0.0:
            continue
        approximated = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximated) != 4 or not cv2.isContourConvex(approximated):
            continue
        area = abs(float(cv2.contourArea(approximated)))
        if not _isfinite(area) or area < min_area_pixels:
            continue
        rectangle = cv2.minAreaRect(approximated)
        rect_width, rect_height = rectangle[1]
        rect_width = float(rect_width)
        rect_height = float(rect_height)
        if rect_width <= 0.0 or rect_height <= 0.0:
            continue
        aspect_error = 1.0 - min(rect_width, rect_height) / max(
            rect_width, rect_height
        )
        if aspect_error > max_aspect_error:
            continue
        rectangle_area = rect_width * rect_height
        rectangularity = area / rectangle_area
        if not _isfinite(rectangularity) or rectangularity < min_rectangularity:
            continue
        rectangularity = min(1.0, rectangularity)
        corners = approximated.reshape(4, 2).astype(np.float64) + offset
        center_array = np.mean(corners, axis=0)
        center = (float(center_array[0]), float(center_array[1]))
        score = _candidate_score(corners, center, area, rectangularity, box)
        candidate = {
            "corners": corners,
            "center": center,
            "area": area,
            "rectangularity": rectangularity,
            "roi": (roi_x1, roi_y1, roi_x2, roi_y2),
            "score": score,
        }

        duplicate = False
        for existing in candidates:
            center_delta = math.hypot(
                center[0] - existing["center"][0],
                center[1] - existing["center"][1],
            )
            area_ratio = min(area, existing["area"]) / max(area, existing["area"])
            if _polygon_iou(corners, existing["corners"]) >= 0.90 or (
                center_delta <= 2.0 and area_ratio >= 0.95
            ):
                duplicate = True
                if score > existing["score"]:
                    existing.update(candidate)
                break
        if not duplicate:
            candidates.append(candidate)

    if not candidates:
        raise LocalizationError("no reliable white quadrilateral candidate")
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        best_score = best["score"]
        second_score = candidates[1]["score"]
        if best_score <= 0.0 or second_score / best_score >= ambiguity_ratio:
            raise LocalizationError("Ambiguous white quadrilateral candidates")
    return best


def render_debug_image(image_bgr, detector_box, localization, depth_radius):
    image = _validate_bgr_image(image_bgr)
    box = _validate_detector_box(detector_box)
    depth_radius = _finite_scalar(depth_radius, "depth radius")
    if depth_radius < 0.0 or depth_radius != math.floor(depth_radius):
        raise LocalizationError("depth radius must be a non-negative integer")
    depth_radius = int(depth_radius)
    if not isinstance(localization, dict):
        raise LocalizationError("localization must be a dictionary")
    corners = _finite_vector(
        np.asarray(localization.get("corners", [])).reshape(-1),
        "localization corners",
        8,
    ).reshape(4, 2)
    center = _finite_vector(localization.get("center", []), "localization center", 2)

    output = image.copy()
    box_points = np.rint(box).astype(np.int32)
    polygon = np.rint(corners).astype(np.int32).reshape((-1, 1, 2))
    center_x = int(math.floor(center[0] + 0.5))
    center_y = int(math.floor(center[1] + 0.5))
    cv2.rectangle(
        output,
        (box_points[0], box_points[1]),
        (box_points[2], box_points[3]),
        (0, 0, 255),
        2,
    )
    cv2.polylines(output, [polygon], True, (0, 255, 0), 2)
    cv2.circle(output, (center_x, center_y), 4, (255, 0, 0), -1)
    cv2.rectangle(
        output,
        (center_x - depth_radius, center_y - depth_radius),
        (center_x + depth_radius, center_y + depth_radius),
        (255, 255, 0),
        1,
    )
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
