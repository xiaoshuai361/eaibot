# Tagless Block Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single Python 3 command that detects one requested tagless supply block, computes a registered-depth 3D target, and reuses the existing Python 2 Mirobot script for safe dry-run and front suction grasp.

**Architecture:** A Python 3 parent process owns Ultralytics YOLOv8 and starts the existing Python 2 `mirobot_pick_test.py` through two dedicated JSON pipes. Python 2 captures synchronized RGB and registered depth, asks Python 3 for a class-specific ROI, refines the 100 mm white square center, transforms the measured surface point into `base`, then uses a verified `Link6 -> suction_tcp` vector to generate contact and pre-contact poses before reusing current MoveIt and pump helpers.

**Tech Stack:** Python 3.8, Ultralytics YOLOv8, Python 2.7, ROS Melodic, `rospy`, `message_filters`, `cv_bridge`, OpenCV, NumPy, TF, MoveIt, pytest.

---

## File Map

- Create: `handeye-calib/src/mirobot_pick_test.py.bak_20260716_before_block_grasp` — exact pre-change backup requested by the user.
- Create: `handeye-calib/src/block_grasp_vision.py` — Python 2/3-compatible pure OpenCV/NumPy geometry, depth validation, deprojection, and debug rendering.
- Create: `handeye-calib/src/block_detector_protocol.py` — Python 2/3-compatible JSON-line pipe protocol.
- Create: `handeye-calib/src/block_grasp_sequence.py` — dependency-injected, testable dry-run/pre-grasp/grasp state machine.
- Create: `handeye-calib/src/block_pick_main.py` — the only operator-facing Python 3 entry point and YOLO owner.
- Modify: `handeye-calib/src/mirobot_pick_test.py` — add `block_grasp`, RGB-D capture, TF conversion, and reuse of current motion/pump helpers.
- Create: `handeye-calib/tests/conftest.py` — import the source directory without installing a ROS package.
- Create: `handeye-calib/tests/test_block_grasp_vision.py` — pure vision, depth, and projection tests.
- Create: `handeye-calib/tests/test_block_detector_protocol.py` — protocol framing and request matching tests.
- Create: `handeye-calib/tests/test_block_pick_main.py` — target mapping and unique-detection selection tests without loading a real model.
- Create: `handeye-calib/tests/test_block_grasp_sequence.py` — fake arm/pump safety and failure-state tests.
- Create: `handeye-calib/tests/python2_smoke.py` — standard-library runner for real Python 2 pipes, NumPy, OpenCV, and geometry.
- Modify: `zcy/机械臂操作.txt` — WSL sync note, dry-run command, pre-contact command, and low-speed single-block command.

The new pure modules stay importable under both Python versions: no f-strings, annotations, dataclasses, pathlib, or Python 3-only OpenCV APIs.

### Task 1: Preserve the Baseline and Add Test Scaffolding

**Files:**
- Create: `handeye-calib/src/mirobot_pick_test.py.bak_20260716_before_block_grasp`
- Create: `handeye-calib/tests/conftest.py`

- [ ] **Step 1: Verify the source has not changed since planning**

Run:

```bash
sha256sum handeye-calib/src/mirobot_pick_test.py
git status --short handeye-calib/src/mirobot_pick_test.py
```

Expected: a SHA-256 line is printed and Git reports no tracked change for this ignored source path. If the file changed after this plan was written, inspect it before continuing and update line references rather than overwriting the change.

- [ ] **Step 2: Create and verify the requested backup**

Run:

```bash
cp -p handeye-calib/src/mirobot_pick_test.py handeye-calib/src/mirobot_pick_test.py.bak_20260716_before_block_grasp
cmp handeye-calib/src/mirobot_pick_test.py handeye-calib/src/mirobot_pick_test.py.bak_20260716_before_block_grasp
```

Expected: `cmp` exits 0 without output.

- [ ] **Step 3: Add the test import path**

Create `handeye-calib/tests/conftest.py`:

```python
from __future__ import absolute_import

import os
import sys


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
```

- [ ] **Step 4: Compile the test scaffolding**

Run:

```bash
python3 -m py_compile handeye-calib/tests/conftest.py
```

Expected: no output and exit code 0.

- [ ] **Step 5: Commit the baseline artifacts**

Run:

```bash
git add -f handeye-calib/src/mirobot_pick_test.py.bak_20260716_before_block_grasp handeye-calib/tests/conftest.py
git commit -m "chore: preserve pre-block-grasp arm script"
```

Expected: the commit contains only the backup and test scaffolding.

### Task 2: Implement Depth Validation and Pixel Deprojection

**Files:**
- Create: `handeye-calib/src/block_grasp_vision.py`
- Create: `handeye-calib/tests/test_block_grasp_vision.py`

- [ ] **Step 1: Write failing depth and projection tests**

Create `handeye-calib/tests/test_block_grasp_vision.py` with these first tests:

```python
import numpy as np
import pytest

from block_grasp_vision import (
    LocalizationError,
    compute_link_targets,
    deproject_pixel,
    rotate_vector_by_quaternion,
    sample_depth_m,
    undistort_pixel,
    validate_axis_alignment,
)


def test_sample_depth_16uc1_uses_median_and_ignores_zero():
    depth = np.zeros((9, 9), dtype=np.uint16)
    depth[2:7, 2:7] = 800
    depth[4, 4] = 1200
    value, stats = sample_depth_m(
        depth, (4.0, 4.0), '16UC1', radius=2,
        min_depth_m=0.2, max_depth_m=1.5,
        min_valid_ratio=0.8, max_mad_m=0.01)
    assert value == pytest.approx(0.8)
    assert stats['valid_ratio'] == pytest.approx(1.0)


def test_sample_depth_rejects_sparse_values():
    depth = np.zeros((9, 9), dtype=np.uint16)
    depth[4, 4] = 800
    with pytest.raises(LocalizationError, match='valid depth ratio'):
        sample_depth_m(
            depth, (4.0, 4.0), '16UC1', radius=2,
            min_depth_m=0.2, max_depth_m=1.5,
            min_valid_ratio=0.8, max_mad_m=0.01)


def test_sample_depth_rejects_unstable_patch():
    depth = np.array([
        [600, 650, 700],
        [750, 800, 850],
        [900, 950, 1000],
    ], dtype=np.uint16)
    with pytest.raises(LocalizationError, match='depth MAD'):
        sample_depth_m(
            depth, (1.0, 1.0), '16UC1', radius=1,
            min_depth_m=0.2, max_depth_m=1.5,
            min_valid_ratio=1.0, max_mad_m=0.05)


def test_deproject_pixel_uses_rgb_intrinsics():
    xyz = deproject_pixel(
        u=420.0, v=290.0, depth_m=0.8,
        fx=400.0, fy=400.0, cx=320.0, cy=240.0)
    assert xyz == pytest.approx((0.2, 0.1, 0.8))


def test_compute_link_targets_follow_rotated_tcp_vector():
    contact, precontact = compute_link_targets(
        surface_base=(0.20, 0.10, 0.30),
        tcp_vector_base=(0.12, 0.0, 0.0), approach_gap_m=0.03)
    assert contact == pytest.approx((0.08, 0.10, 0.30))
    assert precontact == pytest.approx((0.05, 0.10, 0.30))


def test_compute_link_targets_uses_tcp_direction_for_approach():
    contact, precontact = compute_link_targets(
        surface_base=(0.20, 0.10, 0.30),
        tcp_vector_base=(0.0, 0.12, 0.0), approach_gap_m=0.03)
    assert contact == pytest.approx((0.20, -0.02, 0.30))
    assert precontact == pytest.approx((0.20, -0.05, 0.30))


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf'), -float('inf')])
def test_geometry_rejects_non_finite_values(bad_value):
    with pytest.raises(LocalizationError, match='finite'):
        compute_link_targets(
            surface_base=(bad_value, 0.10, 0.30),
            tcp_vector_base=(0.12, 0.0, 0.0), approach_gap_m=0.03)


def test_rotate_vector_by_quaternion_rotates_local_tcp_into_base():
    rotated = rotate_vector_by_quaternion(
        (0.12, 0.0, 0.0),
        (0.0, 0.0, 0.70710678, 0.70710678))
    assert rotated == pytest.approx((0.0, 0.12, 0.0), abs=1e-6)


def test_axis_alignment_rejects_wrong_tool_axis():
    with pytest.raises(LocalizationError, match='angle'):
        validate_axis_alignment(
            suction_axis_base=(1.0, 0.0, 0.0),
            camera_forward_base=(0.0, 0.0, 1.0),
            max_angle_deg=20.0)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
python3 -m pytest handeye-calib/tests/test_block_grasp_vision.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'block_grasp_vision'`.

- [ ] **Step 3: Implement the minimal depth and projection API**

Create `handeye-calib/src/block_grasp_vision.py` with:

```python
from __future__ import absolute_import, division, print_function

import numpy as np


class LocalizationError(RuntimeError):
    pass


def _depth_scale(encoding):
    normalized = encoding.upper()
    if normalized in ('16UC1', 'MONO16'):
        return 0.001
    if normalized == '32FC1':
        return 1.0
    raise LocalizationError('Unsupported depth encoding: %s' % encoding)


def sample_depth_m(depth_image, center, encoding, radius,
                   min_depth_m, max_depth_m, min_valid_ratio, max_mad_m):
    u = int(round(center[0]))
    v = int(round(center[1]))
    height, width = depth_image.shape[:2]
    x1 = max(0, u - radius)
    x2 = min(width, u + radius + 1)
    y1 = max(0, v - radius)
    y2 = min(height, v + radius + 1)
    patch = depth_image[y1:y2, x1:x2].astype(np.float64) * _depth_scale(encoding)
    valid_mask = np.isfinite(patch) & (patch >= min_depth_m) & (patch <= max_depth_m)
    valid_ratio = float(np.count_nonzero(valid_mask)) / float(patch.size)
    if valid_ratio < min_valid_ratio:
        raise LocalizationError(
            'Insufficient valid depth ratio: %.3f < %.3f' %
            (valid_ratio, min_valid_ratio))
    values = patch[valid_mask]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > max_mad_m:
        raise LocalizationError(
            'Unstable depth MAD: %.4f m > %.4f m' % (mad, max_mad_m))
    return median, {'valid_ratio': valid_ratio, 'mad_m': mad}


def deproject_pixel(u, v, depth_m, fx, fy, cx, cy):
    if depth_m <= 0.0:
        raise LocalizationError('Depth must be positive.')
    if fx <= 0.0 or fy <= 0.0:
        raise LocalizationError('Camera focal lengths must be positive.')
    x = (float(u) - float(cx)) * float(depth_m) / float(fx)
    y = (float(v) - float(cy)) * float(depth_m) / float(fy)
    return x, y, float(depth_m)


def undistort_pixel(u, v, camera_matrix, distortion, distortion_model):
    if distortion_model not in ('', 'plumb_bob'):
        raise LocalizationError('Unsupported RGB distortion model: %s' % distortion_model)
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape((3, 3))
    coefficients = np.asarray(distortion, dtype=np.float64)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(coefficients)):
        raise LocalizationError('Camera calibration values must be finite.')
    points = np.asarray([[[float(u), float(v)]]], dtype=np.float64)
    corrected = cv2.undistortPoints(points, matrix, coefficients, P=matrix)
    return tuple(corrected.reshape(2).tolist())


def _finite_vector(name, values):
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise LocalizationError('%s must contain three finite values.' % name)
    return vector


def rotate_vector_by_quaternion(vector, quaternion_xyzw):
    vector = _finite_vector('vector', vector)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise LocalizationError('Quaternion must contain four finite values.')
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise LocalizationError('Quaternion must be non-zero.')
    x, y, z, w = quaternion / norm
    rotation = np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])
    return tuple(np.dot(rotation, vector).tolist())


def compute_link_targets(surface_base, tcp_vector_base, approach_gap_m):
    surface = _finite_vector('surface_base', surface_base)
    tcp_vector = _finite_vector('tcp_vector_base', tcp_vector_base)
    if not np.isfinite(approach_gap_m):
        raise LocalizationError('Approach gap must be finite.')
    if approach_gap_m <= 0.0 or approach_gap_m > 0.15:
        raise LocalizationError('Approach gap must be within (0.0, 0.15] m.')
    tool_length = float(np.linalg.norm(tcp_vector))
    if tool_length <= 0.0 or tool_length > 0.30:
        raise LocalizationError('TCP vector length must be within (0.0, 0.30] m.')
    axis = tcp_vector / tool_length
    contact = surface - tcp_vector
    precontact = contact - axis * float(approach_gap_m)
    return tuple(contact.tolist()), tuple(precontact.tolist())


def validate_axis_alignment(suction_axis_base, camera_forward_base, max_angle_deg):
    suction = _finite_vector('suction_axis_base', suction_axis_base)
    camera = _finite_vector('camera_forward_base', camera_forward_base)
    if not np.isfinite(max_angle_deg) or not 0.0 < max_angle_deg < 90.0:
        raise LocalizationError('Maximum axis angle must be finite and within (0, 90) degrees.')
    suction /= np.linalg.norm(suction)
    camera /= np.linalg.norm(camera)
    cosine = float(np.clip(np.dot(suction, camera), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cosine)))
    if angle_deg > max_angle_deg:
        raise LocalizationError(
            'Suction axis angle %.2f exceeds %.2f degrees.' %
            (angle_deg, max_angle_deg))
    return angle_deg
```

- [ ] **Step 4: Run the depth and projection tests**

Run:

```bash
python3 -m pytest handeye-calib/tests/test_block_grasp_vision.py -q
```

Expected: all depth, projection, finite-value, and tool-axis tests pass.

- [ ] **Step 5: Commit the pure geometry foundation**

Run:

```bash
git add -f handeye-calib/src/block_grasp_vision.py handeye-calib/tests/test_block_grasp_vision.py
git commit -m "feat: validate block depth and deproject pixels"
```

### Task 3: Refine the White Square Center Inside a YOLO ROI

**Files:**
- Modify: `handeye-calib/src/block_grasp_vision.py`
- Modify: `handeye-calib/tests/test_block_grasp_vision.py`

- [ ] **Step 1: Add failing contour and ambiguity tests**

Append to `handeye-calib/tests/test_block_grasp_vision.py`:

```python
import cv2

from block_grasp_vision import find_block_quadrilateral, render_debug_image


def _synthetic_block_image():
    image = np.full((300, 420, 3), 35, dtype=np.uint8)
    cv2.rectangle(image, (120, 70), (320, 270), (245, 245, 245), -1)
    cv2.circle(image, (220, 170), 55, (20, 100, 220), -1)
    return image


def test_quadrilateral_center_comes_from_outer_white_square():
    result = find_block_quadrilateral(
        _synthetic_block_image(), detector_box=(145, 95, 295, 245),
        roi_margin=0.35, min_area_pixels=10000,
        max_aspect_error=0.20, min_rectangularity=0.85,
        ambiguity_ratio=0.85)
    assert result['center'] == pytest.approx((220.0, 170.0), abs=2.0)
    assert result['area'] > 39000


def test_quadrilateral_rejects_scene_without_square():
    image = np.full((300, 420, 3), 35, dtype=np.uint8)
    with pytest.raises(LocalizationError, match='white quadrilateral'):
        find_block_quadrilateral(
            image, detector_box=(145, 95, 295, 245),
            roi_margin=0.35, min_area_pixels=10000,
            max_aspect_error=0.20, min_rectangularity=0.85,
            ambiguity_ratio=0.85)


def test_quadrilateral_rejects_two_similarly_scored_squares():
    image = np.full((300, 500, 3), 35, dtype=np.uint8)
    cv2.rectangle(image, (70, 80), (210, 220), (245, 245, 245), -1)
    cv2.rectangle(image, (290, 80), (430, 220), (245, 245, 245), -1)
    with pytest.raises(LocalizationError, match='Ambiguous'):
        find_block_quadrilateral(
            image, detector_box=(100, 90, 400, 210),
            roi_margin=0.20, min_area_pixels=10000,
            max_aspect_error=0.20, min_rectangularity=0.85,
            ambiguity_ratio=0.85)


def test_debug_image_marks_center_and_depth_patch():
    image = _synthetic_block_image()
    result = find_block_quadrilateral(
        image, detector_box=(145, 95, 295, 245),
        roi_margin=0.35, min_area_pixels=10000,
        max_aspect_error=0.20, min_rectangularity=0.85,
        ambiguity_ratio=0.85)
    debug = render_debug_image(image, (145, 95, 295, 245), result, depth_radius=5)
    assert debug.shape == image.shape
    assert not np.array_equal(debug, image)
```

- [ ] **Step 2: Run only the new tests and verify missing symbols**

Run:

```bash
python3 -m pytest handeye-calib/tests/test_block_grasp_vision.py -q
```

Expected: import fails because `find_block_quadrilateral` and `render_debug_image` do not exist.

- [ ] **Step 3: Implement ROI expansion, candidate scoring, and debug rendering**

Add to `block_grasp_vision.py`:

```python
import cv2


def _expanded_roi(box, image_width, image_height, margin):
    x1, y1, x2, y2 = [float(value) for value in box]
    width = x2 - x1
    height = y2 - y1
    return (
        max(0, int(round(x1 - width * margin))),
        max(0, int(round(y1 - height * margin))),
        min(image_width, int(round(x2 + width * margin))),
        min(image_height, int(round(y2 + height * margin))),
    )


def find_block_quadrilateral(image_bgr, detector_box, roi_margin,
                             min_area_pixels, max_aspect_error,
                             min_rectangularity, ambiguity_ratio):
    image_height, image_width = image_bgr.shape[:2]
    rx1, ry1, rx2, ry2 = _expanded_roi(
        detector_box, image_width, image_height, roi_margin)
    roi = image_bgr[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours_data = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_data[-2]
    detector_center = (
        (float(detector_box[0]) + float(detector_box[2])) * 0.5,
        (float(detector_box[1]) + float(detector_box[3])) * 0.5,
    )
    candidates = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        area = float(cv2.contourArea(polygon))
        if area < min_area_pixels:
            continue
        rectangle = cv2.minAreaRect(polygon)
        rect_width, rect_height = rectangle[1]
        if rect_width <= 0.0 or rect_height <= 0.0:
            continue
        aspect = max(rect_width, rect_height) / min(rect_width, rect_height)
        if abs(aspect - 1.0) > max_aspect_error:
            continue
        rectangularity = area / float(rect_width * rect_height)
        if rectangularity < min_rectangularity:
            continue
        points = polygon.reshape(4, 2).astype(np.float64)
        points[:, 0] += rx1
        points[:, 1] += ry1
        center = tuple(np.mean(points, axis=0).tolist())
        center_distance = np.hypot(
            center[0] - detector_center[0], center[1] - detector_center[1])
        score = area / (1.0 + center_distance)
        candidates.append((score, points, center, area, rectangularity))
    if not candidates:
        raise LocalizationError('No reliable white quadrilateral found in detector ROI.')
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] / candidates[0][0] >= ambiguity_ratio:
        raise LocalizationError('Ambiguous white quadrilateral candidates.')
    best = candidates[0]
    return {
        'corners': best[1],
        'center': best[2],
        'area': best[3],
        'rectangularity': best[4],
        'roi': (rx1, ry1, rx2, ry2),
    }


def render_debug_image(image_bgr, detector_box, localization, depth_radius):
    debug = image_bgr.copy()
    x1, y1, x2, y2 = [int(round(value)) for value in detector_box]
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
    corners = np.round(localization['corners']).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug, [corners], True, (0, 255, 0), 2)
    u, v = [int(round(value)) for value in localization['center']]
    cv2.circle(debug, (u, v), 5, (255, 0, 0), -1)
    cv2.rectangle(
        debug, (u - depth_radius, v - depth_radius),
        (u + depth_radius, v + depth_radius), (255, 255, 0), 1)
    return debug
```

- [ ] **Step 4: Run all pure vision tests**

Run:

```bash
python3 -m pytest handeye-calib/tests/test_block_grasp_vision.py -q
```

Expected: all tests pass, including rejection of two similarly scored white squares. If the synthetic outer contour touches the expanded ROI boundary, increase `roi_margin` in the test and implementation default rather than falling back to the YOLO center.

- [ ] **Step 5: Commit white-square refinement**

Run:

```bash
git add -f handeye-calib/src/block_grasp_vision.py handeye-calib/tests/test_block_grasp_vision.py
git commit -m "feat: refine tagless block center from white contour"
```

### Task 4: Add a Python 2/3 JSON Pipe Protocol

**Files:**
- Create: `handeye-calib/src/block_detector_protocol.py`
- Create: `handeye-calib/tests/test_block_detector_protocol.py`

- [ ] **Step 1: Write failing protocol tests**

Create `handeye-calib/tests/test_block_detector_protocol.py`:

```python
from io import StringIO

import pytest

from block_detector_protocol import DetectorClient, ProtocolError, read_message, write_message


def test_message_round_trip_preserves_unicode_and_box():
    stream = StringIO()
    write_message(stream, {'id': 7, 'class_name': '灭火装置', 'box': [1, 2, 3, 4]})
    stream.seek(0)
    assert read_message(stream) == {
        'id': 7, 'class_name': '灭火装置', 'box': [1, 2, 3, 4]}


def test_client_rejects_mismatched_response_id():
    request_stream = StringIO()
    response_stream = StringIO('{"id": 9, "ok": true}\n')
    client = DetectorClient(request_stream, response_stream)
    with pytest.raises(ProtocolError, match='response id'):
        client.detect('/tmp/frame.png', 'fire')


def test_client_surfaces_detector_error():
    request_stream = StringIO()
    response_stream = StringIO(
        '{"id": 1, "ok": false, "error": "target not found"}\n')
    client = DetectorClient(request_stream, response_stream)
    with pytest.raises(ProtocolError, match='target not found'):
        client.detect('/tmp/frame.png', 'fire')
```

- [ ] **Step 2: Run and verify the module is missing**

Run:

```bash
python3 -m pytest handeye-calib/tests/test_block_detector_protocol.py -q
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement line-delimited JSON with request IDs**

Create `handeye-calib/src/block_detector_protocol.py`:

```python
from __future__ import absolute_import, print_function

import json


class ProtocolError(RuntimeError):
    pass


def write_message(stream, payload):
    # ASCII escaping keeps the same framing on Python 2 byte streams and Python 3 text streams.
    stream.write(json.dumps(payload, ensure_ascii=True) + '\n')
    stream.flush()


def read_message(stream):
    line = stream.readline()
    if not line:
        raise EOFError('Detector pipe closed.')
    try:
        return json.loads(line)
    except ValueError as exc:
        raise ProtocolError('Invalid detector JSON: %s' % exc)


class DetectorClient(object):
    def __init__(self, request_stream, response_stream):
        self.request_stream = request_stream
        self.response_stream = response_stream
        self.next_request_id = 1

    def detect(self, image_path, target):
        request_id = self.next_request_id
        self.next_request_id += 1
        write_message(self.request_stream, {
            'id': request_id,
            'image_path': image_path,
            'target': target,
        })
        response = read_message(self.response_stream)
        if response.get('id') != request_id:
            raise ProtocolError(
                'Unexpected detector response id: %r != %r' %
                (response.get('id'), request_id))
        if not response.get('ok'):
            raise ProtocolError(response.get('error', 'Detector request failed.'))
        return response
```

- [ ] **Step 4: Run protocol tests**

Run:

```bash
python3 -m pytest handeye-calib/tests/test_block_detector_protocol.py -q
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit the protocol**

Run:

```bash
git add -f handeye-calib/src/block_detector_protocol.py handeye-calib/tests/test_block_detector_protocol.py
git commit -m "feat: add detector pipe protocol"
```

### Task 5: Build the Single Python 3 YOLO Entry Point

**Files:**
- Create: `handeye-calib/src/block_pick_main.py`
- Create: `handeye-calib/tests/test_block_pick_main.py`

- [ ] **Step 1: Write failing mapping and selection tests**

Create `handeye-calib/tests/test_block_pick_main.py`:

```python
import pytest

from block_pick_main import (
    DetectionError,
    TARGET_CLASS_IDS,
    parse_args,
    select_unique_detection,
    validate_model_names,
    validate_runtime_args,
)


def test_target_class_mapping_matches_model_metadata():
    assert TARGET_CLASS_IDS == {
        'power': 0,
        'fire': 1,
        'gas': 2,
        'support': 3,
    }


def test_select_unique_detection_returns_requested_class():
    detections = [
        {'class_id': 0, 'confidence': 0.91, 'box': [10, 20, 30, 40]},
        {'class_id': 1, 'confidence': 0.88, 'box': [50, 60, 70, 80]},
    ]
    selected = select_unique_detection(detections, target_class_id=1, confidence=0.5)
    assert selected['box'] == [50, 60, 70, 80]


def test_select_unique_detection_rejects_missing_target():
    with pytest.raises(DetectionError, match='not found'):
        select_unique_detection([], target_class_id=1, confidence=0.5)


def test_select_unique_detection_rejects_ambiguous_target():
    detections = [
        {'class_id': 1, 'confidence': 0.88, 'box': [10, 20, 30, 40]},
        {'class_id': 1, 'confidence': 0.82, 'box': [50, 60, 70, 80]},
    ]
    with pytest.raises(DetectionError, match='Multiple'):
        select_unique_detection(detections, target_class_id=1, confidence=0.5)


def test_model_metadata_rejects_swapped_classes():
    class FakeModel(object):
        names = {
            0: 'Fire extinguishing device',
            1: 'Emergency power supply device',
            2: 'Gas purification device',
            3: 'Structural support device',
        }

    with pytest.raises(DetectionError, match='metadata'):
        validate_model_names(FakeModel())


def test_runtime_args_reject_nan():
    args = parse_args(['--target', 'fire', '--dry-run', '--approach-gap', 'nan'])
    with pytest.raises(DetectionError, match='finite'):
        validate_runtime_args(args)
```

- [ ] **Step 2: Run and verify the entry module is missing**

Run:

```bash
python3 -m pytest handeye-calib/tests/test_block_pick_main.py -q
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement imports, mapping, CLI, and unique selection**

Create `block_pick_main.py` with these public pieces before adding subprocess code:

```python
#!/usr/bin/env python3

from __future__ import print_function

import argparse
import math
import os
import subprocess
import sys

from block_detector_protocol import read_message, write_message


TARGET_CLASSES = {
    'power': (0, 'Emergency power supply device'),
    'fire': (1, 'Fire extinguishing device'),
    'gas': (2, 'Gas purification device'),
    'support': (3, 'Structural support device'),
}
TARGET_CLASS_IDS = dict((key, value[0]) for key, value in TARGET_CLASSES.items())


class DetectionError(RuntimeError):
    pass


def select_unique_detection(detections, target_class_id, confidence):
    matches = [item for item in detections
               if item['class_id'] == target_class_id and
               item['confidence'] >= confidence]
    if not matches:
        raise DetectionError('Requested target was not found above confidence threshold.')
    if len(matches) != 1:
        raise DetectionError('Multiple requested targets were found; refusing ambiguous grasp.')
    return matches[0]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Detect and grasp one tagless supply block.')
    parser.add_argument('--target', required=True, choices=sorted(TARGET_CLASS_IDS))
    parser.add_argument('--model', default='/home/eaibot/models/Block_yolov8n_640/Block_yolov8n_640_best.pt')
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--python2', default='python2')
    parser.add_argument('--arm-script', default='/home/eaibot/handeye-calib/src/mirobot_pick_test.py')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-at-pre-grasp', action='store_true')
    parser.add_argument('--tool-offset', type=float)
    parser.add_argument('--tool-axis', choices=['x', '-x', 'y', '-y', 'z', '-z'])
    parser.add_argument('--max-tool-camera-angle-deg', type=float, default=20.0)
    parser.add_argument('--approach-gap', type=float, default=0.03)
    parser.add_argument('--velocity-scale', type=float, default=0.05)
    parser.add_argument('--acceleration-scale', type=float, default=0.05)
    parser.add_argument('--debug-image', default='/tmp/block_grasp_debug.png')
    return parser.parse_args(argv)


def validate_runtime_args(args):
    numeric = [args.confidence, args.approach_gap,
               args.velocity_scale, args.acceleration_scale]
    if args.tool_offset is not None:
        numeric.append(args.tool_offset)
    if not all(math.isfinite(value) for value in numeric):
        raise DetectionError('All numeric arguments must be finite.')
    if not 0.0 < args.confidence <= 1.0:
        raise DetectionError('--confidence must be in (0, 1].')
    if not 0.0 < args.velocity_scale <= 1.0 or not 0.0 < args.acceleration_scale <= 1.0:
        raise DetectionError('Velocity and acceleration scales must be in (0, 1].')
    if not 0.0 < args.approach_gap <= 0.15:
        raise DetectionError('--approach-gap must be in (0, 0.15].')
    if args.tool_offset is not None and not 0.0 <= args.tool_offset <= 0.30:
        raise DetectionError('--tool-offset must be in [0, 0.30].')
    if (args.tool_offset is None) != (args.tool_axis is None):
        raise DetectionError('--tool-offset and --tool-axis must be provided together.')


def validate_model_names(model):
    expected = dict((value[0], value[1]) for value in TARGET_CLASSES.values())
    actual = dict(model.names)
    if actual != expected:
        raise DetectionError('Model class metadata does not match the block target mapping.')
```

- [ ] **Step 4: Run the selection tests**

Run:

```bash
python3 -m pytest handeye-calib/tests/test_block_pick_main.py -q
```

Expected: all mapping, selection, metadata, and finite-argument tests pass without importing Ultralytics at module import time.

- [ ] **Step 5: Add lazy YOLO loading and request serving**

Add functions that import Ultralytics only inside `load_model`, convert boxes to plain dictionaries, answer each JSON request, and turn exceptions into `ok: false` responses:

```python
def load_model(model_path):
    from ultralytics import YOLO
    if not os.path.isfile(model_path):
        raise DetectionError('Model file does not exist: %s' % model_path)
    model = YOLO(model_path)
    validate_model_names(model)
    return model


def infer_detections(model, image_path):
    results = model.predict(source=image_path, imgsz=640, conf=0.01, verbose=False)
    detections = []
    for box in results[0].boxes:
        detections.append({
            'class_id': int(box.cls.item()),
            'confidence': float(box.conf.item()),
            'box': [float(value) for value in box.xyxy[0].tolist()],
        })
    return detections


def serve_requests(model, request_stream, response_stream, confidence):
    while True:
        try:
            request = read_message(request_stream)
        except EOFError:
            return
        request_id = request.get('id')
        try:
            target = request['target']
            if target not in TARGET_CLASS_IDS:
                raise DetectionError('Unknown target: %s' % target)
            detections = infer_detections(model, request['image_path'])
            selected = select_unique_detection(
                detections, TARGET_CLASS_IDS[target], confidence)
            write_message(response_stream, {
                'id': request_id,
                'ok': True,
                'target': target,
                'class_id': selected['class_id'],
                'class_name': model.names[selected['class_id']],
                'confidence': selected['confidence'],
                'box': selected['box'],
            })
        except Exception as exc:
            write_message(response_stream, {
                'id': request_id,
                'ok': False,
                'error': str(exc),
            })
```

- [ ] **Step 6: Add dedicated pipes and the Python 2 child command**

Implement `main` so normal ROS output remains attached to the terminal while only protocol JSON uses inherited file descriptors:

```python
def build_child_command(args, request_fd, response_fd):
    command = [
        args.python2, args.arm_script,
        '--mode', 'block_grasp',
        '--block-target', args.target,
        '--detector-request-fd', str(request_fd),
        '--detector-response-fd', str(response_fd),
        '--approach-gap', str(args.approach_gap),
        '--max-tool-camera-angle-deg', str(args.max_tool_camera_angle_deg),
        '--velocity-scale', str(args.velocity_scale),
        '--acceleration-scale', str(args.acceleration_scale),
        '--debug-image', args.debug_image,
    ]
    if args.dry_run:
        command.append('--dry-run')
    if args.stop_at_pre_grasp:
        command.append('--stop-at-pre-grasp')
    if args.tool_offset is not None:
        command.extend(['--tool-offset', str(args.tool_offset)])
        command.extend(['--tool-axis', args.tool_axis])
    return command


def main(argv=None):
    args = parse_args(argv)
    validate_runtime_args(args)
    if not args.dry_run and (args.tool_offset is None or args.tool_axis is None):
        raise DetectionError('--tool-offset and --tool-axis are required for a real block grasp.')
    model = load_model(args.model)
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()
    child = subprocess.Popen(
        build_child_command(args, request_write, response_read),
        pass_fds=(request_write, response_read))
    os.close(request_write)
    os.close(response_read)
    try:
        with os.fdopen(request_read, 'r') as request_stream:
            with os.fdopen(response_write, 'w') as response_stream:
                serve_requests(model, request_stream, response_stream, args.confidence)
    except BaseException:
        if child.poll() is None:
            child.terminate()
        child.wait()
        raise
    return_code = child.wait()
    if return_code != 0:
        raise RuntimeError('Python 2 arm process failed with exit code %d.' % return_code)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        sys.stderr.write('block_pick_main: %s\n' % exc)
        sys.exit(1)
```

- [ ] **Step 7: Add tests for child command safety**

Append to `test_block_pick_main.py`:

```python
from block_pick_main import build_child_command, main, parse_args


def test_child_command_forwards_dry_run_and_pre_grasp_stop():
    args = parse_args(['--target', 'fire', '--dry-run', '--stop-at-pre-grasp'])
    command = build_child_command(args, request_fd=11, response_fd=12)
    assert '--dry-run' in command
    assert '--stop-at-pre-grasp' in command
    assert '--tool-offset' not in command
    assert command[command.index('--detector-request-fd') + 1] == '11'
    assert command[command.index('--detector-response-fd') + 1] == '12'


def test_child_command_forwards_measured_tool_offset():
    args = parse_args([
        '--target', 'fire', '--tool-offset', '0.12', '--tool-axis', 'x'])
    command = build_child_command(args, request_fd=11, response_fd=12)
    assert command[command.index('--tool-offset') + 1] == '0.12'
    assert command[command.index('--tool-axis') + 1] == 'x'


def test_real_run_rejects_missing_tool_offset_before_model_load(monkeypatch):
    def fail_if_called(_model_path):
        pytest.fail('model must not load before real-run safety validation')

    monkeypatch.setattr('block_pick_main.load_model', fail_if_called)
    with pytest.raises(DetectionError, match='tool-offset'):
        main(['--target', 'fire'])
```

Add child lifecycle helpers exactly once in `block_pick_main.py`:

```python
def close_fd_safely(fd):
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def stop_child(child, timeout_sec=3.0):
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=timeout_sec)
```

Add these concrete lifecycle tests using fake children; add a separate monkeypatched `Popen` failure test that asserts all four pipe FDs are closed:

```python
def test_stop_child_terminates_then_kills_after_timeout():
    class FakeChild(object):
        def __init__(self):
            self.calls = []
            self.wait_count = 0

        def poll(self):
            return None

        def terminate(self):
            self.calls.append('terminate')

        def kill(self):
            self.calls.append('kill')

        def wait(self, timeout=None):
            self.wait_count += 1
            if self.wait_count == 1:
                raise subprocess.TimeoutExpired('child', timeout)
            self.calls.append(('wait', timeout))
            return -9

    child = FakeChild()
    stop_child(child, timeout_sec=0.1)
    assert child.calls == ['terminate', 'kill', ('wait', 0.1)]
```

Add separate parent-error tests with explicit fakes: `os.pipe` returns four real temporary FDs, `subprocess.Popen` returns a fake child, and `serve_requests` raises `KeyboardInterrupt`. Assert `stop_child` receives that child and each FD raises `OSError` when closed again. Repeat with `BrokenPipeError`. Add a `Popen`-failure test with the same FD assertions. The production `main` uses `try/finally` from the first `os.pipe()` onward so every unwrapped FD is closed, then calls `stop_child` before re-raising.

Run:

```bash
python3 -m pytest handeye-calib/tests/test_block_pick_main.py -q
```

Expected: all mapping, selection, command, and early-safety tests pass.

- [ ] **Step 8: Commit the Python 3 entry**

Run:

```bash
chmod +x handeye-calib/src/block_pick_main.py
git add -f handeye-calib/src/block_pick_main.py handeye-calib/tests/test_block_pick_main.py
git commit -m "feat: add tagless block YOLO entry point"
```

### Task 6: Add RGB-D Localization to the Existing Python 2 Script

**Files:**
- Modify: `handeye-calib/src/mirobot_pick_test.py:6`
- Modify: `handeye-calib/src/mirobot_pick_test.py:53`
- Modify: `handeye-calib/src/mirobot_pick_test.py:820`

- [ ] **Step 1: Add a source-level regression test before editing**

Append to `test_block_pick_main.py` a test that reads `mirobot_pick_test.py` and initially fails because `block_grasp` is not in the mode choices and the old modes are not represented as a reusable constant:

```python
from pathlib import Path


def test_arm_script_declares_block_grasp_without_removing_old_modes():
    source = Path('handeye-calib/src/mirobot_pick_test.py').read_text(encoding='utf-8')
    for mode in ('home', 'pump', 'grasp', 'place', 'pick_place',
                 'pick_lift_place', 'current_pose', 'wrist_forward', 'block_grasp'):
        assert "'%s'" % mode in source
```

Run:

```bash
python3 -m pytest handeye-calib/tests/test_block_pick_main.py::test_arm_script_declares_block_grasp_without_removing_old_modes -q
```

Expected: FAIL because `'block_grasp'` is absent.

- [ ] **Step 2: Add Python 2-compatible imports and CLI arguments**

Add standard imports `os`, `tempfile`, and `threading`, plus the shared modules below. Add lazy ROS image imports inside the new capture helper so old modes do not gain unnecessary ROS image import failures. Extend mode choices with `block_grasp` and add:

```python
from block_detector_protocol import DetectorClient
from block_grasp_vision import (
    compute_link_targets,
    deproject_pixel,
    find_block_quadrilateral,
    render_debug_image,
    rotate_vector_by_quaternion,
    sample_depth_m,
)
```

```python
    parser.add_argument('--block-target', choices=['power', 'fire', 'gas', 'support'])
    parser.add_argument('--detector-request-fd', type=int)
    parser.add_argument('--detector-response-fd', type=int)
    parser.add_argument('--rgb-topic', default='/camera/rgb/image_raw')
    parser.add_argument('--registered-depth-topic', default='/camera/depth_registered/image_raw')
    parser.add_argument('--rgb-camera-info-topic', default='/camera/rgb/camera_info')
    parser.add_argument('--rgbd-timeout', type=float, default=5.0)
    parser.add_argument('--rgbd-slop', type=float, default=0.15)
    parser.add_argument('--depth-radius', type=int, default=5)
    parser.add_argument('--min-depth', type=float, default=0.20)
    parser.add_argument('--max-depth', type=float, default=1.50)
    parser.add_argument('--min-valid-depth-ratio', type=float, default=0.60)
    parser.add_argument('--max-depth-mad', type=float, default=0.01)
    parser.add_argument('--block-roi-margin', type=float, default=0.35)
    parser.add_argument('--block-min-area', type=float, default=900.0)
    parser.add_argument('--block-max-aspect-error', type=float, default=0.25)
    parser.add_argument('--block-min-rectangularity', type=float, default=0.80)
    parser.add_argument('--block-ambiguity-ratio', type=float, default=0.85)
    parser.add_argument('--tool-offset', type=float)
    parser.add_argument('--tool-axis', choices=['x', '-x', 'y', '-y', 'z', '-z'])
    parser.add_argument('--max-tool-camera-angle-deg', type=float, default=20.0)
    parser.add_argument('--stop-at-pre-grasp', action='store_true')
    parser.add_argument('--debug-image', default='/tmp/block_grasp_debug.png')
    parser.add_argument('--block-min-base-z', type=float, default=0.04)
    parser.add_argument('--block-max-base-radius', type=float, default=0.40)
```

Add `require_block_args(args)` that requires target and both detector FDs; rejects non-finite or out-of-range safety parameters; and requires both a measured tool offset and an explicitly verified local tool axis for non-dry-run execution.

- [ ] **Step 3: Add an approximate-time single RGB-D capture helper**

Implement a small `RgbdCapture` object that:

- lazily imports `message_filters`, `CvBridge`, `Image`, and `CameraInfo`;
- subscribes to RGB and registered depth;
- uses `ApproximateTimeSynchronizer(..., queue_size=10, slop=args.rgbd_slop)`;
- stores only the first synchronized pair under a lock and signals a `threading.Event`;
- obtains RGB `CameraInfo` with `rospy.wait_for_message`;
- unregisters subscribers after capture;
- validates RGB, depth, and CameraInfo dimensions before returning arrays and metadata.

The validation must compare `rgb.shape[:2]`, `depth.shape[:2]`, `camera_info.width`, and `camera_info.height`; mismatch raises `RuntimeError` before any MoveIt call.

Use this concrete interface and state handling:

```python
def capture_rgbd_once(args):
    import message_filters
    from cv_bridge import CvBridge
    from sensor_msgs.msg import CameraInfo, Image

    bridge = CvBridge()
    captured = {}
    capture_lock = threading.Lock()
    capture_event = threading.Event()

    def callback(rgb_msg, depth_msg):
        with capture_lock:
            if capture_event.is_set():
                return
            try:
                captured['rgb'] = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
                captured['depth'] = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                captured['rgb_header'] = copy.deepcopy(rgb_msg.header)
                captured['depth_header'] = copy.deepcopy(depth_msg.header)
                captured['depth_encoding'] = depth_msg.encoding
            except Exception as exc:
                captured['error'] = exc
            capture_event.set()

    rgb_sub = message_filters.Subscriber(args.rgb_topic, Image)
    depth_sub = message_filters.Subscriber(args.registered_depth_topic, Image)
    try:
        synchronizer = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=args.rgbd_slop)
        synchronizer.registerCallback(callback)
        camera_info = rospy.wait_for_message(
            args.rgb_camera_info_topic, CameraInfo, timeout=args.rgbd_timeout)
        if not capture_event.wait(args.rgbd_timeout):
            raise RuntimeError('Timed out waiting for synchronized RGB and registered depth.')
    finally:
        rgb_sub.unregister()
        depth_sub.unregister()
    if 'error' in captured:
        raise RuntimeError('CvBridge conversion failed: %s' % captured['error'])
    rgb_height, rgb_width = captured['rgb'].shape[:2]
    depth_height, depth_width = captured['depth'].shape[:2]
    if (rgb_width, rgb_height) != (depth_width, depth_height):
        raise RuntimeError('Registered depth size does not match RGB size.')
    if (camera_info.width, camera_info.height) != (rgb_width, rgb_height):
        raise RuntimeError('RGB CameraInfo size does not match RGB image size.')
    time_delta = abs((captured['rgb_header'].stamp - captured['depth_header'].stamp).to_sec())
    if time_delta > args.rgbd_slop:
        raise RuntimeError('RGB/depth timestamp difference exceeds configured slop.')
    rgb_frame = captured['rgb_header'].frame_id.lstrip('/')
    depth_frame = captured['depth_header'].frame_id.lstrip('/')
    info_frame = camera_info.header.frame_id.lstrip('/')
    if not rgb_frame or depth_frame != rgb_frame or info_frame != rgb_frame:
        raise RuntimeError(
            'RGB, registered depth, and RGB CameraInfo must use the same optical frame.')
    if len(camera_info.K) != 9 or not all(math.isfinite(value) for value in camera_info.K):
        raise RuntimeError('RGB CameraInfo K must contain nine finite values.')
    if camera_info.K[0] <= 0.0 or camera_info.K[4] <= 0.0:
        raise RuntimeError('RGB CameraInfo focal lengths must be positive.')
    if captured['depth'].ndim != 2:
        raise RuntimeError('Registered depth must be a single-channel image.')
    expected_dtypes = {'16UC1': np.uint16, 'MONO16': np.uint16, '32FC1': np.float32}
    expected_dtype = expected_dtypes.get(captured['depth_encoding'].upper())
    if expected_dtype is None or captured['depth'].dtype != expected_dtype:
        raise RuntimeError('Registered depth encoding and dtype do not match.')
    captured['camera_info'] = camera_info
    return captured
```

- [ ] **Step 4: Add detector request and local refinement**

Implement `request_block_detection(args, image_bgr)`:

1. Open the inherited request FD for line-buffered writing and response FD for reading.
2. Save the exact captured RGB frame to a `mkstemp` PNG.
3. Call `DetectorClient.detect(temp_path, args.block_target)`.
4. Always unlink the temporary frame.
5. Call `find_block_quadrilateral` with the returned box.
6. Call `sample_depth_m` with the registered depth encoding.
7. Read `fx, fy, cx, cy` from `CameraInfo.K`.
8. Call `deproject_pixel`.
9. Save `render_debug_image` to `args.debug_image` and fail if `cv2.imwrite` returns false.

Return one dictionary containing target, confidence, box, corners, center, depth stats, and camera XYZ. Log every value with fixed units.

Implement the function as follows:

```python
def localize_block(args, captured):
    import cv2

    request_stream = os.fdopen(args.detector_request_fd, 'w', 1)
    response_stream = os.fdopen(args.detector_response_fd, 'r')
    detector = DetectorClient(request_stream, response_stream)
    temp_fd, temp_path = tempfile.mkstemp(prefix='block_rgb_', suffix='.png')
    os.close(temp_fd)
    try:
        if not cv2.imwrite(temp_path, captured['rgb']):
            raise RuntimeError('Failed to write detector input: %s' % temp_path)
        detection = detector.detect(temp_path, args.block_target)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    refined = find_block_quadrilateral(
        captured['rgb'], detection['box'], args.block_roi_margin,
        args.block_min_area, args.block_max_aspect_error,
        args.block_min_rectangularity, args.block_ambiguity_ratio)
    depth_m, depth_stats = sample_depth_m(
        captured['depth'], refined['center'], captured['depth_encoding'],
        args.depth_radius, args.min_depth, args.max_depth,
        args.min_valid_depth_ratio, args.max_depth_mad)
    camera_info = captured['camera_info']
    corrected_u, corrected_v = undistort_pixel(
        refined['center'][0], refined['center'][1], camera_info.K,
        camera_info.D, camera_info.distortion_model)
    camera_xyz = deproject_pixel(
        corrected_u, corrected_v, depth_m,
        camera_info.K[0], camera_info.K[4], camera_info.K[2], camera_info.K[5])
    debug = render_debug_image(
        captured['rgb'], detection['box'], refined, args.depth_radius)
    if not cv2.imwrite(args.debug_image, debug):
        raise RuntimeError('Failed to write debug image: %s' % args.debug_image)
    return {
        'target': args.block_target,
        'confidence': detection['confidence'],
        'box': detection['box'],
        'corners': refined['corners'],
        'center': refined['center'],
        'depth_m': depth_m,
        'depth_stats': depth_stats,
        'camera_xyz': camera_xyz,
        'camera_header': captured['rgb_header'],
    }
```

- [ ] **Step 5: Transform the surface point, then offset along the verified suction TCP vector**

Before using the existing `transform_pose`, change its `waitForTransform` call to wait at `pose_stamped.header.stamp` rather than `rospy.Time(0)`. The wait time and the actual `transformPose` request must refer to the same RGB acquisition timestamp. Add a source regression assertion for this exact argument.

Create only the measured surface `PoseStamped` in the actual RGB optical frame, then transform it to `base`. Do not subtract tool length along camera optical Z.

Define `--tool-offset` as the measured distance from the `Link6` origin to the suction contact center and `--tool-axis` as the verified `Link6` local direction from `Link6` toward that contact center. The pair defines the fixed `Link6 -> suction_tcp` translation vector. Rotate that vector into `base` with the same end-effector orientation used for execution.

The desired suction TCP is the measured surface point. Therefore:

```text
tcp_vector_base = R(base <- Link6) * tcp_vector_link6
link6_contact_base = surface_base - tcp_vector_base
suction_axis_base = normalize(tcp_vector_base)
link6_precontact_base = link6_contact_base - approach_gap * suction_axis_base
```

For first-stage dry-run and motion, require the arm to already be in the separately verified wrist-forward joint5 pose within tolerance. This makes the displayed and executed orientation identical. If the joint5 check fails, stop and instruct the operator to run `--mode wrist_forward`; do not move the wrist automatically inside localization.

Reject when any scalar/vector is non-finite, the tool offset or gap is out of bounds, the local axis is unknown, joint5 is outside tolerance, or either contact/pre-contact violates:

```text
pose_base.z < block_min_base_z
hypot(pose_base.x, pose_base.y) > block_max_base_radius
```

Publish surface, contact, and pre-contact poses through `publish_debug_geometry` using extra debug topics. Do not call `warn_if_grasp_target_is_too_low` because this mode has hard rejection rather than warning-only behavior.

Implement the point construction explicitly:

```python
def make_camera_point_pose(header, x_value, y_value, z_value):
    pose = PoseStamped()
    pose.header = copy.deepcopy(header)
    pose.pose.position.x = x_value
    pose.pose.position.y = y_value
    pose.pose.position.z = z_value
    pose.pose.orientation.w = 1.0
    return pose


TOOL_AXES = {
    'x': (1.0, 0.0, 0.0), '-x': (-1.0, 0.0, 0.0),
    'y': (0.0, 1.0, 0.0), '-y': (0.0, -1.0, 0.0),
    'z': (0.0, 0.0, 1.0), '-z': (0.0, 0.0, -1.0),
}


def build_block_poses(args, localization, listener, current_orientation):
    camera_x, camera_y, surface_depth = localization['camera_xyz']
    surface_camera = make_camera_point_pose(
        localization['camera_header'], camera_x, camera_y, surface_depth)
    camera_forward_reference = make_camera_point_pose(
        localization['camera_header'], camera_x, camera_y, surface_depth + 0.01)
    surface_base = transform_pose(listener, args.base_frame, surface_camera, args.tf_timeout)
    forward_base = transform_pose(
        listener, args.base_frame, camera_forward_reference, args.tf_timeout)
    local_axis = TOOL_AXES[args.tool_axis]
    local_tcp = tuple(value * args.tool_offset for value in local_axis)
    quaternion = (
        current_orientation.x, current_orientation.y,
        current_orientation.z, current_orientation.w)
    tcp_vector_base = rotate_vector_by_quaternion(local_tcp, quaternion)
    camera_forward_base = (
        forward_base.pose.position.x - surface_base.pose.position.x,
        forward_base.pose.position.y - surface_base.pose.position.y,
        forward_base.pose.position.z - surface_base.pose.position.z)
    validate_axis_alignment(
        tcp_vector_base, camera_forward_base, args.max_tool_camera_angle_deg)
    surface_xyz = (
        surface_base.pose.position.x, surface_base.pose.position.y,
        surface_base.pose.position.z)
    contact_xyz, precontact_xyz = compute_link_targets(
        surface_xyz, tcp_vector_base, args.approach_gap)
    grasp_pose = build_absolute_pose(
        args.base_frame, contact_xyz[0], contact_xyz[1], contact_xyz[2],
        current_orientation)
    pre_grasp_pose = build_absolute_pose(
        args.base_frame, precontact_xyz[0], precontact_xyz[1], precontact_xyz[2],
        current_orientation)
    for label, pose in [('contact', grasp_pose), ('precontact', pre_grasp_pose)]:
        values = (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError('%s block pose contains a non-finite value.' % label)
        if pose.pose.position.z < args.block_min_base_z:
            raise RuntimeError('%s block pose is below the safety height.' % label)
        if math.hypot(pose.pose.position.x, pose.pose.position.y) > args.block_max_base_radius:
            raise RuntimeError('%s block pose is outside the safety radius.' % label)
    return surface_camera, surface_base, pre_grasp_pose, grasp_pose
```

If `--tool-offset/--tool-axis` are omitted in dry-run, publish only the measured surface point and debug image; do not fabricate Link6 contact/pre-contact poses. A real run and `--stop-at-pre-grasp` remain rejected unless both measured values are supplied.

- [ ] **Step 6: Add `compute_block_context` without routing motion yet**

Implement:

```python
def compute_block_context(args, arm):
    require_block_args(args)
    captured = capture_rgbd_once(args)
    localization = localize_block(args, captured)
    listener = tf.TransformListener()
    current_pose = arm.get_current_pose()
    surface_camera = make_camera_point_pose(
        localization['camera_header'], localization['camera_xyz'][0],
        localization['camera_xyz'][1], localization['camera_xyz'][2])
    surface_base = transform_pose(
        listener, args.base_frame, surface_camera, args.tf_timeout)
    if args.tool_offset is None or args.tool_axis is None:
        return localization, surface_camera, surface_base, None, None
    if not is_wrist_forward_reached(arm, args.wrist_forward_joint5, tolerance=0.02):
        raise RuntimeError('Run --mode wrist_forward before computing Link6 block targets.')
    surface_camera, surface_base, pre_grasp_pose, grasp_pose = build_block_poses(
        args, localization, listener, current_pose.pose.orientation)
    return localization, surface_camera, surface_base, pre_grasp_pose, grasp_pose
```

At this task, update only argument parsing, shared imports, capture/localization helpers, TF timestamp handling, and this context function. Do not dispatch `block_grasp` until Task 7 has created `do_block_grasp` and its state-machine tests.

Preserve these constraints for the later route:

- MoveGroup is built for `block_grasp`.
- Pump proxy is acquired for `block_grasp` only for a full contact grasp, not for dry-run or `--stop-at-pre-grasp`.
- `resolve_tag_id` and `resolve_grasp_offsets` remain limited to tag modes.
- Existing mode defaults and code paths remain byte-for-byte equivalent where possible.

- [ ] **Step 7: Run static and source regression checks**

Run:

```bash
python3 -m py_compile handeye-calib/src/block_grasp_vision.py handeye-calib/src/block_detector_protocol.py handeye-calib/src/block_pick_main.py handeye-calib/src/mirobot_pick_test.py
python3 -m pytest handeye-calib/tests -q
cmp handeye-calib/src/mirobot_pick_test.py.bak_20260716_before_block_grasp handeye-calib/src/mirobot_pick_test.py
```

Expected: compilation and tests pass; the final `cmp` must report a difference, proving the backup remains the original rather than being overwritten.

- [ ] **Step 8: Commit RGB-D localization wiring**

Run:

```bash
git add -f handeye-calib/src/mirobot_pick_test.py handeye-calib/tests/test_block_pick_main.py
git commit -m "feat: localize tagless blocks from registered depth"
```

### Task 7: Reuse Existing Motion and Pump Helpers for a Safe Front Grasp

**Files:**
- Create: `handeye-calib/src/block_grasp_sequence.py`
- Create: `handeye-calib/tests/test_block_grasp_sequence.py`
- Modify: `handeye-calib/src/mirobot_pick_test.py:506`
- Modify: `handeye-calib/src/mirobot_pick_test.py:613`
- Modify: `handeye-calib/src/mirobot_pick_test.py:820`

- [ ] **Step 1: Add source assertions for dry-run and motion reuse**

Add this narrow source regression test because ROS Melodic Python 2 modules are unavailable in WSL:

```python
def test_block_grasp_dry_run_precedes_motion_and_reuses_helpers():
    source = Path('handeye-calib/src/mirobot_pick_test.py').read_text(encoding='utf-8')
    start = source.index('def do_block_grasp(')
    end = source.index('\ndef ', start + 1)
    body = source[start:end]
    dry_run_index = body.index('if args.dry_run:')
    sequence_index = body.index('run_block_sequence(')
    assert dry_run_index < sequence_index
    assert 'go_wrist_forward(' not in body
    assert 'compute_block_context(' in body
```

Also create `handeye-calib/tests/test_block_grasp_sequence.py` before the implementation. Use fake callables to verify behavior rather than relying only on source order:

```python
from block_grasp_sequence import run_block_sequence


def test_dry_run_calls_no_motion_or_pump():
    calls = []
    result = run_block_sequence(
        dry_run=True, stop_at_pre_grasp=False,
        confirm_pump_off=lambda: calls.append('pump_off'),
        move_pre=lambda: calls.append('move_pre'),
        move_contact=lambda: calls.append('move_contact'),
        pump_on=lambda: calls.append('pump_on'),
        retreat=lambda: calls.append('retreat'),
        log=lambda message: calls.append(('log', message)))
    assert result == 'dry_run'
    assert [item for item in calls if isinstance(item, str)] == []


def test_stop_at_pre_grasp_moves_once_without_pump():
    calls = []
    result = run_block_sequence(
        dry_run=False, stop_at_pre_grasp=True,
        confirm_pump_off=lambda: calls.append('pump_off'),
        move_pre=lambda: calls.append('move_pre'),
        move_contact=lambda: calls.append('move_contact'),
        pump_on=lambda: calls.append('pump_on'),
        retreat=lambda: calls.append('retreat'),
        log=lambda message: None)
    assert result == 'pre_grasp'
    assert calls == ['move_pre']


def test_pump_exception_reports_unknown_state():
    messages = []

    def uncertain_pump():
        raise RuntimeError('response timeout')

    with pytest.raises(RuntimeError, match='response timeout'):
        run_block_sequence(
            dry_run=False, stop_at_pre_grasp=False,
            confirm_pump_off=lambda: None,
            move_pre=lambda: None, move_contact=lambda: None,
            pump_on=uncertain_pump, retreat=lambda: None,
            log=messages.append)
    assert any('UNKNOWN' in message for message in messages)
```

Run and expect collection to fail because `block_grasp_sequence` does not exist.

- [ ] **Step 2: Implement `do_block_grasp` dry-run first**

The function must:

1. capture and localize the target;
2. build and transform surface, contact, and pre-contact poses;
3. log `block_surface_camera`, `block_surface_base`, `block_pre_grasp`, and `block_grasp`;
4. publish debug geometry;
5. if `args.dry_run`, optionally hold debug topics and return before wrist, pump, or motion calls.

This gives a true no-motion path even if MoveIt and the pump service are available.

Use this complete function boundary; the injected sequence owns all action ordering:

```python
def do_block_grasp(args, arm, pump_proxy):
    localization, surface_camera, surface_base, pre_grasp_pose, grasp_pose = \
        compute_block_context(args, arm)
    rospy.loginfo(pose_to_text('block_surface_camera', surface_camera))
    rospy.loginfo(pose_to_text('block_surface_base', surface_base))
    if pre_grasp_pose is not None:
        rospy.loginfo(pose_to_text('block_pre_grasp', pre_grasp_pose))
        rospy.loginfo(pose_to_text('block_grasp', grasp_pose))
    publish_debug_geometry(
        args.base_frame, arm.get_current_pose(), surface_base,
        pre_grasp_pose, grasp_pose)

    if args.dry_run:
        rospy.logwarn('Dry run: no wrist, pump, or arm motion executed.')
        if args.debug_hold_seconds > 0.0:
            rospy.sleep(args.debug_hold_seconds)
        return
    if pre_grasp_pose is None or grasp_pose is None:
        raise RuntimeError('Real block motion requires measured tool geometry.')

    run_block_sequence(
        dry_run=False,
        stop_at_pre_grasp=args.stop_at_pre_grasp,
        confirm_pump_off=(lambda: set_pump(pump_proxy, False)),
        move_pre=(lambda: execute_pose(arm, pre_grasp_pose, 'block_pre_grasp')),
        move_contact=(lambda: execute_cartesian_pose(
            arm, grasp_pose, 'block_grasp_contact')),
        pump_on=(lambda: set_pump(pump_proxy, True)),
        retreat=(lambda: execute_cartesian_pose(
            arm, pre_grasp_pose, 'block_grasp_retreat')),
        log=rospy.logwarn)
```

After this function and `block_grasp_sequence.py` tests pass, update `main`: build MoveGroup for `block_grasp`, acquire the pump proxy only when `not args.dry_run and not args.stop_at_pre_grasp`, and dispatch `do_block_grasp` explicitly before the final `pick_place` fallback.

Create `block_grasp_sequence.py` and make `do_block_grasp` delegate its action ordering to it:

```python
from __future__ import absolute_import, print_function


def run_block_sequence(dry_run, stop_at_pre_grasp, confirm_pump_off,
                       move_pre, move_contact, pump_on, retreat, log):
    if dry_run:
        log('Dry run: no wrist, pump, or arm motion executed.')
        return 'dry_run'
    move_pre()
    if stop_at_pre_grasp:
        log('Stopped at pre-grasp. No pump-on command was sent in this run.')
        return 'pre_grasp'
    confirm_pump_off()
    move_contact()
    pump_attempted = True
    try:
        pump_on()
    except Exception:
        if pump_attempted:
            log('Pump state is UNKNOWN and may be ON; recover manually.')
        raise
    try:
        retreat()
    except Exception:
        log('Retreat failed after pump-on; pump may remain ON. Recover manually.')
        raise
    return 'grasped'
```

- [ ] **Step 3: Add the real motion path using existing helpers**

Wire the complete `do_block_grasp` function from Step 2 to the existing helpers through `run_block_sequence`. Do not duplicate a second inline action sequence in `mirobot_pick_test.py`; the state machine is the single ordering source.

Do not call `go_home` and do not turn the pump off in first-stage `block_grasp`; the user requested holding the object for later manual handling.

- [ ] **Step 4: Add exception safety around pump state**

Before contact, explicitly command pump OFF and require its response while the arm is still at pre-grasp. Track whether pump-on was attempted, not only whether its response succeeded. If the pump service times out, report the state as unknown and possibly ON. If retreat fails after pump-on, log a high-severity error and keep the pump on rather than dropping the block. Do not automatically issue a second motion command from the exception handler. Use this control shape:

```python
    pump_on_attempted = False
    try:
        execute_pose(arm, pre_grasp_pose, 'block_pre_grasp')
        set_pump(pump_proxy, False)
        rospy.sleep(0.5)
        execute_cartesian_pose(arm, grasp_pose, 'block_grasp_contact')
        rospy.sleep(0.5)
        pump_on_attempted = True
        set_pump(pump_proxy, True)
        rospy.sleep(0.8)
        execute_cartesian_pose(arm, pre_grasp_pose, 'block_grasp_retreat')
    except Exception:
        if pump_on_attempted:
            rospy.logerr('Pump state is UNKNOWN and may be ON. Stop and recover manually.')
        raise
```

- [ ] **Step 5: Run the complete WSL suite**

Run:

```bash
python3 -m pytest handeye-calib/tests -q
python3 -m py_compile handeye-calib/src/block_grasp_vision.py handeye-calib/src/block_detector_protocol.py handeye-calib/src/block_pick_main.py handeye-calib/src/mirobot_pick_test.py
```

Expected: all tests pass and all four Python files compile. Note explicitly that Python 2 runtime imports, ROS topics, TF, MoveIt, and pump behavior remain unverified in WSL because `python2` and the true hardware chain are absent.

- [ ] **Step 6: Commit motion reuse**

Run:

```bash
git add -f handeye-calib/src/mirobot_pick_test.py handeye-calib/tests/test_block_pick_main.py
git commit -m "feat: execute safe front grasp for tagless blocks"
```

### Task 8: Document Operator Commands and Sync Requirements

**Files:**
- Modify: `zcy/机械臂操作.txt`

- [ ] **Step 1: Add a documented-model-path test**

Add to `test_block_pick_main.py`:

```python
def test_default_model_path_matches_true_machine_documentation():
    args = parse_args(['--target', 'fire', '--dry-run'])
    assert args.model == (
        '/home/eaibot/models/Block_yolov8n_640/'
        'Block_yolov8n_640_best.pt')
```

Run:

```bash
python3 -m pytest handeye-calib/tests/test_block_pick_main.py::test_default_model_path_matches_true_machine_documentation -q
```

Expected: PASS before editing the documentation.

- [ ] **Step 2: Document the true-machine prerequisites**

Add a section explaining:

- the code was edited in WSL and must be synced to `/home/eaibot`;
- sync `block_pick_main.py`, `block_detector_protocol.py`, `block_grasp_vision.py`, `mirobot_pick_test.py`, and the backup;
- sync `/home/zcy/models/Block_yolov8n_640` to `/home/eaibot/models/Block_yolov8n_640`;
- source ROS Melodic, `mirobot_ws`, and `handeye-calib` in that order;
- start Astra, Mirobot/MoveIt, and the hand-eye TF publisher before running the entry command;
- confirm `/camera/depth_registered/image_raw` exists and is aligned with RGB.

- [ ] **Step 3: Document the dry-run command**

Use a command in this form:

```bash
python3 /home/eaibot/handeye-calib/src/block_pick_main.py \
  --target fire \
  --dry-run \
  --debug-image /home/eaibot/zcy/block_grasp_debug.png
```

State that `power`, `fire`, `gas`, and `support` map to the four model classes and that dry-run performs no wrist, pump, or arm movement.

- [ ] **Step 4: Document pre-contact and first-contact safety**

Explain that the first real run requires a measured `--tool-offset`, starts at velocity/acceleration 0.05, and must be performed in two stages on the true machine:

1. run with `--stop-at-pre-grasp` so the arm moves only to the verified pre-contact point with the pump off;
2. execute one low-speed grasp only after the pre-contact point is verified.

Include the real command template without inventing a tool length:

```bash
python3 /home/eaibot/handeye-calib/src/block_pick_main.py \
  --target fire \
  --tool-offset MEASURED_METERS \
  --tool-axis VERIFIED_LINK6_AXIS \
  --approach-gap 0.03 \
  --velocity-scale 0.05 \
  --acceleration-scale 0.05 \
  --stop-at-pre-grasp
```

After that command reaches the correct pre-grasp point, repeat it without `--stop-at-pre-grasp` to permit contact, pump-on, and retreat.

The literals `MEASURED_METERS` and `VERIFIED_LINK6_AXIS` are documentation for required operator measurements, not software defaults. Determine the local axis from the actual Link6/吸盘 assembly in RViz and a low-risk orientation check; the program rejects a real run when either value is absent or the transformed tool axis differs from the camera forward direction by more than the configured angle.

- [ ] **Step 5: Run documentation and regression checks**

Run:

```bash
rg -n "block_pick_main|depth_registered|tool-offset|power|fire|gas|support" zcy/机械臂操作.txt
python3 -m pytest handeye-calib/tests -q
git diff --check
```

Expected: the new commands and safety notes are found, all tests pass, and no whitespace error is reported.

- [ ] **Step 6: Commit documentation**

Run:

```bash
git add -f zcy/机械臂操作.txt handeye-calib/tests/test_block_pick_main.py
git commit -m "docs: add tagless block grasp procedure"
```

### Task 9: Perform Final Offline Verification and Prepare True-Machine Checklist

**Files:**
- Verify: `handeye-calib/src/block_grasp_vision.py`
- Verify: `handeye-calib/src/block_detector_protocol.py`
- Verify: `handeye-calib/src/block_pick_main.py`
- Verify: `handeye-calib/src/mirobot_pick_test.py`
- Verify: `zcy/机械臂操作.txt`

- [ ] **Step 1: Run all offline tests from the repository root**

Run:

```bash
python3 -m pytest handeye-calib/tests -q
```

Expected: all tests pass with zero skips caused by implementation errors.

- [ ] **Step 2: Run syntax checks**

Run:

```bash
python3 -m py_compile \
  handeye-calib/src/block_grasp_vision.py \
  handeye-calib/src/block_detector_protocol.py \
  handeye-calib/src/block_pick_main.py \
  handeye-calib/src/mirobot_pick_test.py
```

Expected: no output and exit code 0. Record that this is only a grammar check for `mirobot_pick_test.py`, not proof that ROS Melodic Python 2 imports work.

- [ ] **Step 3: Re-run real model inference on the preserved test split**

Run:

```bash
python3 - <<'PY'
from ultralytics import YOLO
model = YOLO('/home/zcy/models/Block_yolov8n_640/Block_yolov8n_640_best.pt')
metrics = model.val(
    data='/home/zcy/models/Block_yolov8n_640/Block_yolov8n_640_data.yaml',
    split='test', imgsz=640, device='cpu', plots=False, verbose=False)
print('P=%.3f R=%.3f mAP50=%.3f mAP50-95=%.3f' % (
    metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map))
PY
```

Expected: approximately `P=0.937 R=1.000 mAP50=0.995 mAP50-95=0.775`.

- [ ] **Step 4: Verify the backup and review only intended files**

Record the baseline commit before Task 1:

```bash
BASELINE_COMMIT=$(git rev-parse HEAD)
printf '%s\n' "$BASELINE_COMMIT" > /tmp/tagless_block_grasp_baseline
```

At final review run:

Run:

```bash
cmp handeye-calib/src/mirobot_pick_test.py.bak_20260716_before_block_grasp handeye-calib/src/mirobot_pick_test.py
git status --short
git diff --stat "$(cat /tmp/tagless_block_grasp_baseline)"..HEAD
```

Expected: `cmp` reports a difference; the review shows only tagless-grasp source/tests/docs plus pre-existing unrelated user changes. Do not stage or revert the unrelated `robocom_ws` and `zcy` changes.

- [ ] **Step 5: Hand off true-machine commands without executing them in WSL**

The final report must separate:

- completed WSL checks;
- files to sync to `/home/eaibot`;
- model directory to sync;
- true-machine read-only checks for RGB, registered depth, CameraInfo, and TF;
- true-machine dry-run;
- pre-contact verification;
- measured tool-offset low-speed grasp.

No ROS node, serial device, camera, MoveIt motion, or pump command is executed in WSL.

- [ ] **Step 6: Include Python 2 dependency checks in the true-machine handoff**

Run these only on the competition machine after sourcing ROS and both workspaces:

```bash
python2 - <<'PY'
import cv2
import numpy
import rospy
import tf
import moveit_commander
import message_filters
from cv_bridge import CvBridge
from mirobot_urdf_2.srv import mirobotPump
print('python2 block-grasp dependencies: OK')
PY

python2 -m py_compile \
  /home/eaibot/handeye-calib/src/block_grasp_vision.py \
  /home/eaibot/handeye-calib/src/block_detector_protocol.py \
  /home/eaibot/handeye-calib/src/block_grasp_sequence.py \
  /home/eaibot/handeye-calib/src/mirobot_pick_test.py

python2 /home/eaibot/handeye-calib/tests/python2_smoke.py

python3 - <<'PY'
from ultralytics import YOLO
YOLO('/home/eaibot/models/Block_yolov8n_640/Block_yolov8n_640_best.pt')
print('python3 YOLO model: OK')
PY
```

`python2_smoke.py` must use `os.pipe()` and real `os.fdopen()` streams to round-trip a Unicode JSON message, then import OpenCV/NumPy and run one depth-median, one deprojection, one white-square, and one rotated-TCP geometry assertion without pytest. Expected: compilation is silent and all three commands print `OK`. A missing module or smoke failure blocks dry-run and must be fixed on the competition machine before starting the entry command; do not alter the ROS Python version in WSL to hide the failure.

- [ ] **Step 7: Verify live topic metadata before the first dry-run**

Run only on the competition machine:

```bash
rostopic echo -n 1 /camera/rgb/image_raw/header
rostopic echo -n 1 /camera/depth_registered/image_raw/header
rostopic echo -n 1 /camera/rgb/camera_info
```

Expected: RGB, registered depth, and CameraInfo report the same non-empty RGB optical frame, matched dimensions, valid `K/D`, and RGB/depth timestamps close enough for the configured synchronization slop. Inspect an RGB/depth edge overlay as a hard gate; matching dimensions alone do not prove registration.

- [ ] **Step 8: Document pump recovery and post-grasp release**

The true-machine checklist must state that a pump service timeout means the pump state is unknown and may be ON. Stop motion and recover manually. Once the held block is supported at a safe destination, release it with the existing `--mode pump` test only if its automatic on/off cycle is safe for the situation, or use the verified `switch_pump_status(False)` service procedure documented for the machine. Never command pump-off while the block is unsupported.
