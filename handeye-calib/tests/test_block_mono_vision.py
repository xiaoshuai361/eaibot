import math

import numpy as np
import pytest

from block_mono_vision import (
    LocalizationError,
    box_geometry,
    decode_yolov5_output,
    deproject_pixel_to_camera_mm,
    draw_debug_detections,
    estimate_distance_mm,
    is_detection_usable,
    stable_median_observation,
)


def test_box_geometry_returns_center_size_and_aspect():
    result = box_geometry([10.0, 20.0, 40.0, 80.0])

    assert result["u"] == pytest.approx(25.0)
    assert result["v"] == pytest.approx(50.0)
    assert result["w"] == pytest.approx(30.0)
    assert result["h"] == pytest.approx(60.0)
    assert result["aspect"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "detection, message",
    [
        ({"confidence": 0.69, "box": [0, 0, 60, 60]}, "confidence"),
        ({"confidence": 0.90, "box": [0, 0, 20, 20]}, "width"),
        ({"confidence": 0.90, "box": [0, 0, 80, 20]}, "aspect"),
    ],
)
def test_is_detection_usable_rejects_low_confidence_small_width_and_bad_aspect(
    detection, message
):
    rules = {
        "confidence_min": 0.70,
        "box_width_min_px": 30,
        "box_aspect_ratio_min": 0.75,
        "box_aspect_ratio_max": 1.30,
    }

    usable, reason = is_detection_usable(detection, rules)

    assert usable is False
    assert message in reason


def test_stable_median_observation_returns_robust_center_width_and_stats():
    observations = [
        {"u": 100.0, "v": 50.0, "w": 60.0, "h": 61.0, "confidence": 0.91},
        {"u": 101.0, "v": 51.0, "w": 61.0, "h": 60.0, "confidence": 0.92},
        {"u": 99.0, "v": 49.5, "w": 59.5, "h": 60.5, "confidence": 0.90},
    ]

    result = stable_median_observation(
        observations,
        frames_required=3,
        center_std_max_px=2.0,
        width_cv_max=0.03,
    )

    assert result["u"] == pytest.approx(100.0)
    assert result["v"] == pytest.approx(50.0)
    assert result["w"] == pytest.approx(60.0)
    assert result["center_std_px"] < 2.0
    assert result["width_cv"] < 0.03


def test_stable_median_observation_rejects_unstable_width():
    observations = [
        {"u": 100.0, "v": 50.0, "w": 50.0, "h": 50.0, "confidence": 0.91},
        {"u": 100.0, "v": 50.0, "w": 60.0, "h": 60.0, "confidence": 0.92},
        {"u": 100.0, "v": 50.0, "w": 70.0, "h": 70.0, "confidence": 0.90},
    ]

    with pytest.raises(LocalizationError, match="width"):
        stable_median_observation(
            observations,
            frames_required=3,
            center_std_max_px=2.0,
            width_cv_max=0.03,
        )


def test_estimate_distance_supports_theory_and_calibrated_models():
    theory = estimate_distance_mm(
        method="theory",
        width_px=60.0,
        fx_px=600.0,
        target_size_mm=30.0,
        target="fire",
        distance_models={},
    )
    calibrated = estimate_distance_mm(
        method="calibrated",
        width_px=60.0,
        fx_px=600.0,
        target_size_mm=30.0,
        target="fire",
        distance_models={"fire": {"a": 21000.0, "b": 5.0}},
    )

    assert theory == pytest.approx(300.0)
    assert calibrated == pytest.approx(355.0)


def test_estimate_distance_rejects_missing_calibration_for_real_model():
    with pytest.raises(LocalizationError, match="calibration"):
        estimate_distance_mm(
            method="calibrated",
            width_px=60.0,
            fx_px=600.0,
            target_size_mm=30.0,
            target="fire",
            distance_models={"fire": {"a": None, "b": None}},
        )


def test_deproject_pixel_to_camera_mm_uses_rgb_intrinsics():
    point = deproject_pixel_to_camera_mm(
        u=330.0,
        v=250.0,
        z_mm=300.0,
        fx_px=600.0,
        fy_px=500.0,
        cx_px=320.0,
        cy_px=240.0,
    )

    assert point == pytest.approx((5.0, 6.0, 300.0))


def test_decode_yolov5_output_uses_objectness_times_class_probability():
    # YOLOv5 exported ONNX commonly returns cx, cy, w, h, objectness, class scores.
    output = np.array([[[320.0, 320.0, 160.0, 80.0, 0.80, 0.10, 0.90, 0.20, 0.30]]])

    detections = decode_yolov5_output(
        output,
        image_shape=(480, 640),
        input_shape=(640, 640),
        scale=1.0,
        pad=(0.0, 80.0),
        confidence_min=0.25,
        nms_iou=0.45,
        class_count=4,
    )

    assert len(detections) == 1
    assert detections[0]["class_id"] == 1
    assert detections[0]["confidence"] == pytest.approx(0.72)
    assert detections[0]["box"] == pytest.approx([240.0, 200.0, 400.0, 280.0])


def test_decode_yolov5_output_handles_transposed_prediction_and_rejects_nonfinite_values():
    output = np.array([
        [[math.nan], [320.0], [160.0], [80.0], [0.80], [0.10], [0.90], [0.20], [0.30]]
    ])

    assert decode_yolov5_output(
        output,
        image_shape=(480, 640),
        input_shape=(640, 640),
        scale=1.0,
        pad=(0.0, 80.0),
        confidence_min=0.25,
        nms_iou=0.45,
        class_count=4,
    ) == []


def test_draw_debug_detections_draws_multiple_boxes():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    detections = [
        {"class_id": 0, "class_name": "power", "confidence": 0.9, "box": [10, 10, 40, 40]},
        {"class_id": 1, "class_name": "fire", "confidence": 0.8, "box": [60, 20, 90, 50]},
    ]

    output = draw_debug_detections(image, detections)

    assert output.shape == image.shape
    assert int(np.count_nonzero(output)) > 0
