from __future__ import absolute_import, division, print_function

import math

import numpy as np
import pytest

import block_grasp_vision
from block_grasp_vision import (
    LocalizationError,
    _finite_vector,
    compute_link_targets,
    deproject_pixel,
    rotate_vector_by_quaternion,
    sample_depth_m,
    undistort_pixel,
    validate_axis_alignment,
)


def test_finite_checks_do_not_require_python3_math_isfinite(monkeypatch):
    monkeypatch.delattr(block_grasp_vision.math, "isfinite")

    assert deproject_pixel(0, 0, 1, 1, 1, 0, 0) == (0.0, 0.0, 1.0)


def test_sample_depth_16uc1_uses_median_and_ignores_zero():
    depth = np.array(
        [[0, 1000, 1010], [990, 0, 1000], [1020, 980, 0]], dtype=np.uint16
    )

    value, quality = sample_depth_m(depth, (1, 1), "16UC1", 1, 0.5, 2.0, 0.6, 0.1)

    assert value == pytest.approx(1.0)
    assert quality["valid_ratio"] == pytest.approx(6.0 / 9.0)
    assert quality["mad_m"] == pytest.approx(0.01)


def test_sample_depth_mono16_is_millimetres():
    depth = np.array([[1250]], dtype=np.uint16)

    value, quality = sample_depth_m(depth, (0, 0), "MONO16", 0, 0.1, 2.0, 1.0, 0.01)

    assert value == pytest.approx(1.25)
    assert quality == {"valid_ratio": 1.0, "mad_m": 0.0}


def test_sample_depth_accepts_ros_lowercase_mono16():
    value, _ = sample_depth_m(
        np.array([[1250]], dtype=np.uint16),
        (0, 0),
        "mono16",
        0,
        0.1,
        2.0,
        1.0,
        0.01,
    )

    assert value == pytest.approx(1.25)


def test_sample_depth_rounds_half_pixels_up_consistently():
    depth = np.array(
        [[1000, 1100, 1200], [1300, 1400, 1500], [1600, 1700, 1800]],
        dtype=np.uint16,
    )

    value, _ = sample_depth_m(depth, (0.5, 1.5), "16uc1", 0, 0.1, 2.0, 1.0, 0.01)

    assert value == pytest.approx(1.7)


def test_sample_depth_32fc1_uses_metres_and_ignores_nonfinite():
    depth = np.array([[1.0, np.nan], [1.02, np.inf]], dtype=np.float32)

    value, quality = sample_depth_m(depth, (0, 0), "32FC1", 1, 0.5, 2.0, 0.5, 0.1)

    assert value == pytest.approx(1.01)
    assert quality["valid_ratio"] == pytest.approx(0.5)
    assert quality["mad_m"] == pytest.approx(0.01)


def test_sample_depth_rejects_sparse_patch():
    depth = np.array([[1000, 0], [0, 0]], dtype=np.uint16)

    with pytest.raises(LocalizationError, match="valid ratio"):
        sample_depth_m(depth, (0, 0), "16UC1", 1, 0.5, 2.0, 0.5, 0.1)


def test_sample_depth_rejects_excessive_mad():
    depth = np.array([[500, 1000, 1500]], dtype=np.uint16)

    with pytest.raises(LocalizationError, match="MAD"):
        sample_depth_m(depth, (1, 0), "16UC1", 1, 0.1, 2.0, 1.0, 0.2)


def test_sample_depth_clips_patch_at_image_boundary():
    depth = np.array([[1000, 1100], [1200, 1300]], dtype=np.uint16)

    value, quality = sample_depth_m(depth, (0, 0), "16UC1", 2, 0.1, 2.0, 1.0, 1.0)

    assert value == pytest.approx(1.15)
    assert quality["valid_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "depth,center",
    [
        (np.empty((0, 2), dtype=np.uint16), (0, 0)),
        (np.ones((2, 2), dtype=np.uint16), (10, 10)),
    ],
)
def test_sample_depth_rejects_empty_patch(depth, center):
    with pytest.raises(LocalizationError, match="patch"):
        sample_depth_m(depth, center, "16UC1", 1, 0.1, 2.0, 0.1, 0.1)


def test_sample_depth_rejects_unknown_encoding():
    with pytest.raises(LocalizationError, match="encoding"):
        sample_depth_m(np.ones((1, 1)), (0, 0), "8UC1", 0, 0.1, 2.0, 1.0, 0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"center": (np.nan, 0)},
        {"radius": -1},
        {"min_depth_m": np.nan},
        {"max_depth_m": np.inf},
        {"min_depth_m": 2.0, "max_depth_m": 1.0},
        {"min_valid_ratio": 0.0},
        {"min_valid_ratio": 1.1},
        {"max_mad_m": -0.1},
    ],
)
def test_sample_depth_rejects_invalid_parameters(kwargs):
    parameters = dict(
        depth_image=np.ones((3, 3), dtype=np.uint16) * 1000,
        center=(1, 1),
        encoding="16UC1",
        radius=1,
        min_depth_m=0.1,
        max_depth_m=2.0,
        min_valid_ratio=0.5,
        max_mad_m=0.1,
    )
    parameters.update(kwargs)

    with pytest.raises(LocalizationError):
        sample_depth_m(**parameters)


def test_deproject_pixel_uses_pinhole_equations():
    point = deproject_pixel(420.0, 290.0, 2.0, 500.0, 400.0, 320.0, 250.0)

    assert point == pytest.approx((0.4, 0.2, 2.0))


def test_deproject_pixel_rejects_nonfinite_computed_coordinates():
    with pytest.raises(LocalizationError, match="finite"):
        deproject_pixel(1e308, 0, 1e308, 1e-308, 1, 0, 0)


@pytest.mark.parametrize(
    "arguments",
    [
        (np.nan, 0, 1, 1, 1, 0, 0),
        (0, np.inf, 1, 1, 1, 0, 0),
        (0, 0, 0, 1, 1, 0, 0),
        (0, 0, 1, 0, 1, 0, 0),
        (0, 0, 1, 1, -1, 0, 0),
    ],
)
def test_deproject_pixel_rejects_invalid_parameters(arguments):
    with pytest.raises(LocalizationError):
        deproject_pixel(*arguments)


@pytest.mark.parametrize("model", ["", "plumb_bob"])
def test_undistort_pixel_with_zero_distortion_is_unchanged(model):
    camera_matrix = np.array(
        [[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]]
    )

    corrected = undistort_pixel(123.5, 222.25, camera_matrix, np.zeros(5), model)

    assert corrected == pytest.approx((123.5, 222.25), abs=1e-5)


def test_undistort_pixel_rejects_unsupported_model():
    with pytest.raises(LocalizationError, match="distortion model"):
        undistort_pixel(1, 2, np.eye(3), np.zeros(5), "equidistant")


@pytest.mark.parametrize(
    "camera_matrix,distortion",
    [
        (np.eye(2), np.zeros(5)),
        (np.array([[1.0, 0, 0], [0, np.nan, 0], [0, 0, 1]]), np.zeros(5)),
        (np.eye(3), np.array([0, 0, np.inf, 0, 0])),
    ],
)
def test_undistort_pixel_rejects_invalid_calibration(camera_matrix, distortion):
    with pytest.raises(LocalizationError):
        undistort_pixel(1, 2, camera_matrix, distortion, "plumb_bob")


def test_finite_vector_returns_float_array():
    vector = _finite_vector([1, 2, 3], "test vector", 3)

    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float64
    assert vector.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("vector", [[1, 2], [1, np.nan, 3], [1, np.inf, 3]])
def test_finite_vector_rejects_wrong_size_or_nonfinite(vector):
    with pytest.raises(LocalizationError):
        _finite_vector(vector, "test vector", 3)


def test_rotate_vector_by_quaternion_rotates_ninety_degrees_about_z():
    half_sqrt = math.sqrt(0.5)

    rotated = rotate_vector_by_quaternion((1, 0, 0), (0, 0, half_sqrt, half_sqrt))

    assert rotated == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_rotate_vector_by_quaternion_normalizes_quaternion():
    rotated = rotate_vector_by_quaternion((1, 0, 0), (0, 0, 2, 2))

    assert rotated == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


@pytest.mark.parametrize(
    "vector,quaternion",
    [
        ((1, 0, np.nan), (0, 0, 0, 1)),
        ((1, 0, 0), (0, 0, 0, 0)),
        ((1, 0, 0), (0, 0, np.inf, 1)),
    ],
)
def test_rotate_vector_rejects_invalid_inputs_or_zero_quaternion(vector, quaternion):
    with pytest.raises(LocalizationError):
        rotate_vector_by_quaternion(vector, quaternion)


def test_compute_link_targets_uses_complete_tcp_vector_and_axis():
    contact, precontact = compute_link_targets((1.0, 2.0, 3.0), (0.0, 0.0, 0.2), 0.05)

    assert contact == pytest.approx((1.0, 2.0, 2.8))
    assert precontact == pytest.approx((1.0, 2.0, 2.75))


def test_compute_link_targets_normalizes_non_unit_tcp_for_gap_only():
    contact, precontact = compute_link_targets((1.0, 1.0, 1.0), (0.03, 0.04, 0.0), 0.1)

    assert contact == pytest.approx((0.97, 0.96, 1.0))
    assert precontact == pytest.approx((0.91, 0.88, 1.0))


@pytest.mark.parametrize(
    "surface,tcp,gap",
    [
        ((np.nan, 0, 0), (0, 0, 0.1), 0.01),
        ((0, 0, 0), (0, 0, 0), 0.01),
        ((0, 0, 0), (0, 0, 0.301), 0.01),
        ((0, 0, 0), (0, 0, 0.1), 0.0),
        ((0, 0, 0), (0, 0, 0.1), 0.151),
        ((0, 0, 0), (0, 0, 0.1), np.inf),
    ],
)
def test_compute_link_targets_rejects_invalid_length_or_gap(surface, tcp, gap):
    with pytest.raises(LocalizationError):
        compute_link_targets(surface, tcp, gap)


def test_compute_link_targets_accepts_inclusive_boundaries():
    contact, precontact = compute_link_targets((0, 0, 1), (0, 0, 0.3), 0.15)

    assert contact == pytest.approx((0, 0, 0.7))
    assert precontact == pytest.approx((0, 0, 0.55))


def test_validate_axis_alignment_returns_angle_for_aligned_axes():
    angle = validate_axis_alignment((0, 0, 4), (0, 0, 2), 10.0)

    assert angle == pytest.approx(0.0)


def test_validate_axis_alignment_rejects_wrong_axis():
    with pytest.raises(LocalizationError, match="alignment"):
        validate_axis_alignment((1, 0, 0), (0, 0, 1), 20.0)


@pytest.mark.parametrize(
    "suction,camera,max_angle",
    [
        ((0, 0, 0), (0, 0, 1), 10),
        ((0, 0, 1), (0, 0, 0), 10),
        ((0, np.nan, 1), (0, 0, 1), 10),
        ((0, 0, 1), (0, 0, 1), 0),
        ((0, 0, 1), (0, 0, 1), 90),
        ((0, 0, 1), (0, 0, 1), np.inf),
    ],
)
def test_validate_axis_alignment_rejects_invalid_inputs(suction, camera, max_angle):
    with pytest.raises(LocalizationError):
        validate_axis_alignment(suction, camera, max_angle)
