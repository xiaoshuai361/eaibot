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
    find_block_quadrilateral,
    render_debug_image,
    rotate_vector_by_quaternion,
    sample_depth_m,
    undistort_pixel,
    validate_axis_alignment,
)


def _localization_image(polygons, size=(320, 420)):
    image = np.full((size[0], size[1], 3), 25, dtype=np.uint8)
    for corners in polygons:
        polygon = np.asarray(corners, dtype=np.int32)
        cv2 = block_grasp_vision.cv2
        cv2.fillConvexPoly(image, polygon, (245, 245, 245))
    return image


def _find(image, box, **overrides):
    parameters = dict(
        image_bgr=image,
        detector_box=box,
        roi_margin=2.0,
        min_area_pixels=1000,
        max_aspect_error=0.25,
        min_rectangularity=0.75,
        ambiguity_ratio=0.92,
    )
    parameters.update(overrides)
    return find_block_quadrilateral(**parameters)


def test_find_block_uses_outer_white_square_when_detector_only_covers_offset_art():
    image = _localization_image([[(70, 60), (270, 60), (270, 260), (70, 260)]])
    block_grasp_vision.cv2.rectangle(image, (190, 100), (235, 145), (40, 40, 180), -1)

    result = _find(image, (185, 95, 245, 155))

    assert result["center"] == pytest.approx((170.0, 160.0), abs=2.0)
    assert result["corners"].shape == (4, 2)
    assert result["area"] == pytest.approx(40000, rel=0.05)
    assert result["rectangularity"] > 0.95
    assert np.isfinite(result["score"])


def test_find_block_accepts_lightly_rotated_perspective_quadrilateral():
    corners = [(90, 72), (260, 55), (280, 225), (105, 244)]
    image = _localization_image([corners])

    result = _find(image, (145, 115, 220, 180))

    assert result["center"] == pytest.approx(np.mean(corners, axis=0), abs=3.0)


def test_find_block_rejects_no_square_and_long_rectangle():
    empty = _localization_image([])
    rectangle = _localization_image([[(60, 110), (330, 110), (330, 190), (60, 190)]])

    with pytest.raises(LocalizationError, match="candidate"):
        _find(empty, (150, 120, 210, 180))
    with pytest.raises(LocalizationError, match="candidate"):
        _find(rectangle, (150, 120, 210, 180))


@pytest.mark.parametrize(
    "overrides",
    [
        {"detector_box": (20, 20, 20, 40)},
        {"detector_box": (20, 20, np.nan, 40)},
        {"roi_margin": -0.1},
        {"roi_margin": 2.1},
        {"min_area_pixels": 0},
        {"max_aspect_error": -0.1},
        {"max_aspect_error": 1.1},
        {"min_rectangularity": 0},
        {"min_rectangularity": 1.1},
        {"ambiguity_ratio": 0},
        {"ambiguity_ratio": 1.1},
    ],
)
def test_find_block_rejects_invalid_parameters(overrides):
    image = _localization_image([[(50, 50), (200, 50), (200, 200), (50, 200)]])
    parameters = dict(
        detector_box=(90, 90, 140, 140),
        roi_margin=2,
        min_area_pixels=500,
        max_aspect_error=0.3,
        min_rectangularity=0.7,
        ambiguity_ratio=0.9,
    )
    parameters.update(overrides)

    with pytest.raises(LocalizationError):
        find_block_quadrilateral(image, **parameters)


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 20, 3), dtype=np.uint8),
        np.zeros((20, 20), dtype=np.uint8),
        np.zeros((20, 20, 4), dtype=np.uint8),
        np.zeros((20, 20, 3), dtype=np.float32),
    ],
)
def test_find_block_rejects_invalid_image(image):
    with pytest.raises(LocalizationError, match="image"):
        _find(image, (1, 1, 10, 10))


def test_find_block_clips_roi_to_image_boundary():
    image = _localization_image([[(-15, -10), (105, 0), (100, 105), (0, 100)]], (180, 180))

    result = _find(image, (-20, -20, 45, 45), min_area_pixels=500)

    assert result["roi"] == (0, 0, 175, 175)
    assert result["center"][0] < 60
    assert result["center"][1] < 60


def test_find_block_rejects_empty_clipped_roi():
    image = _localization_image([])

    with pytest.raises(LocalizationError, match="ROI"):
        _find(image, (500, 500, 550, 550))


def test_find_block_rejects_two_equally_plausible_squares_as_ambiguous():
    image = _localization_image(
        [
            [(35, 80), (155, 80), (155, 200), (35, 200)],
            [(245, 80), (365, 80), (365, 200), (245, 200)],
        ]
    )

    with pytest.raises(LocalizationError, match="Ambiguous"):
        _find(image, (140, 105, 260, 175), roi_margin=1.0, ambiguity_ratio=0.85)


def test_find_block_selects_candidate_that_covers_detector_box():
    image = _localization_image(
        [
            [(40, 70), (190, 70), (190, 220), (40, 220)],
            [(270, 95), (370, 95), (370, 195), (270, 195)],
        ]
    )

    result = _find(image, (92, 115, 145, 170), ambiguity_ratio=0.8)

    assert result["center"] == pytest.approx((115, 145), abs=2)


def test_find_block_rejects_unrelated_square_only_visible_in_expanded_roi():
    image = _localization_image(
        [[(220, 80), (300, 80), (300, 160), (220, 160)]]
    )

    with pytest.raises(LocalizationError, match="associat"):
        _find(image, (120, 100, 180, 160), min_area_pixels=500)


def test_find_block_deduplicates_nearly_identical_contours(monkeypatch):
    image = _localization_image([[(70, 60), (250, 60), (250, 240), (70, 240)]])
    contour = np.array([[[70, 60]], [[250, 60]], [[250, 240]], [[70, 240]]], dtype=np.int32)
    duplicate = contour.copy()
    duplicate[:, 0, :] += 1

    monkeypatch.setattr(
        block_grasp_vision.cv2,
        "findContours",
        lambda *args, **kwargs: ([contour, duplicate], None),
    )

    result = _find(image, (120, 110, 190, 180))

    assert result["center"] == pytest.approx((160, 150), abs=2)


def test_render_debug_image_draws_without_modifying_input():
    image = _localization_image([[(70, 60), (250, 60), (250, 240), (70, 240)]])
    localization = _find(image, (115, 100, 195, 180))
    original = image.copy()

    rendered = render_debug_image(image, (115, 100, 195, 180), localization, 6)

    assert np.array_equal(image, original)
    assert rendered.shape == image.shape
    assert rendered.dtype == image.dtype
    assert not np.array_equal(rendered, image)
    assert tuple(rendered[100, 115]) == (0, 0, 255)


@pytest.mark.parametrize("radius", [-1, 1.5, np.nan, np.inf])
def test_render_debug_image_rejects_invalid_radius(radius):
    image = _localization_image([])
    localization = {
        "corners": np.array([[1, 1], [10, 1], [10, 10], [1, 10]], dtype=float),
        "center": (5, 5),
    }

    with pytest.raises(LocalizationError, match="radius"):
        render_debug_image(image, (1, 1, 10, 10), localization, radius)


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
