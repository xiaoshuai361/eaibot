import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from block_mono_vision import (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_PATH,
    LocalizationError,
    box_geometry,
    decode_yolov5_output,
    deproject_pixel_to_camera_mm,
    draw_debug_detections,
    estimate_distance_from_box_mm,
    estimate_distance_mm,
    is_detection_usable,
    observation_in_roi,
    parse_target_sequence,
    resolve_target_alias,
    roi_box_pixels,
    stable_median_observation,
)


def test_default_config_is_loaded_from_the_canonical_yaml():
    with Path(DEFAULT_CONFIG_PATH).open(encoding="utf-8") as stream:
        assert DEFAULT_CONFIG == yaml.safe_load(stream)


def test_default_grasp_uses_same_five_stable_samples_as_tag_workflow():
    assert DEFAULT_CONFIG["frames_required"] == 5
    assert DEFAULT_CONFIG["max_axis_distance_disagreement_mm"] == 0.0


def test_numeric_target_ids_map_to_existing_block_targets():
    assert [resolve_target_alias(str(index)) for index in range(1, 5)] == [
        "power", "fire", "gas", "support"]
    assert resolve_target_alias("gas") == "gas"
    assert parse_target_sequence("4,2,1") == ["support", "fire", "power"]
    with pytest.raises(LocalizationError, match="duplicate"):
        parse_target_sequence("1,power")


def test_box_geometry_returns_center_size_and_aspect():
    result = box_geometry([10.0, 20.0, 40.0, 80.0])

    assert result["u"] == pytest.approx(25.0)
    assert result["v"] == pytest.approx(50.0)
    assert result["w"] == pytest.approx(30.0)
    assert result["h"] == pytest.approx(60.0)
    assert result["aspect"] == pytest.approx(0.5)


def test_grasp_roi_converts_ratios_and_rejects_outside_center():
    roi = roi_box_pixels((480, 640, 3), [0.06, 0.0, 0.24, 1.0])

    assert roi == (38, 0, 154, 480)
    assert observation_in_roi(
        {"u": 100, "v": 240}, (480, 640, 3), [0.06, 0.0, 0.24, 1.0]
    ) == (True, "")
    usable, reason = observation_in_roi(
        {"u": 200, "v": 240}, (480, 640, 3), [0.06, 0.0, 0.24, 1.0]
    )
    assert usable is False
    assert "outside grasp ROI" in reason


def test_grasp_roi_rejects_invalid_ratios():
    with pytest.raises(LocalizationError, match="grasp_roi_ratio"):
        roi_box_pixels((480, 640, 3), [0.4, 0.0, 0.2, 1.0])


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


def test_stable_median_observation_rejects_outlier_and_keeps_fresh_cluster():
    observations = [
        {"u": 100.0 + delta, "v": 50.0, "w": 60.0, "h": 61.0,
         "confidence": 0.9}
        for delta in (-0.4, 0.0, 0.3, 0.2, -0.2)
    ]
    observations.insert(2, {
        "u": 180.0, "v": 120.0, "w": 20.0, "h": 20.0,
        "confidence": 0.95,
    })

    result = stable_median_observation(
        observations, frames_required=5,
        center_std_max_px=2.0, width_cv_max=0.03)

    assert result["u"] == pytest.approx(100.0, abs=0.3)
    assert result["inlier_count"] == 5


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


def test_estimate_distance_from_box_combines_width_and_height_models():
    distance = estimate_distance_from_box_mm(
        method="calibrated",
        width_px=60.0,
        height_px=80.0,
        fx_px=600.0,
        fy_px=600.0,
        target_width_mm=30.0,
        target_height_mm=40.0,
        target="fire",
        distance_models={
            "fire": {
                "width": {"a": 18000.0, "b": 0.0},
                "height": {"a": 24000.0, "b": 0.0},
            }
        },
    )

    assert distance == pytest.approx(300.0)


def test_estimate_distance_from_box_rejects_axis_disagreement_with_details():
    with pytest.raises(LocalizationError) as error:
        estimate_distance_from_box_mm(
            method="calibrated",
            width_px=60.0,
            height_px=60.0,
            fx_px=600.0,
            fy_px=600.0,
            target_width_mm=30.0,
            target_height_mm=30.0,
            target="fire",
            distance_models={
                "fire": {
                    "width": {"a": 18000.0, "b": 0.0},
                    "height": {"a": 24000.0, "b": 0.0},
                }
            },
            max_axis_disagreement_mm=20.0,
        )

    message = str(error.value)
    assert "disagreement 100.00 mm exceeds 20.00 mm" in message
    assert "box_width=60.00 px" in message
    assert "box_height=60.00 px" in message
    assert "width_distance=300.00 mm" in message
    assert "height_distance=400.00 mm" in message


def test_estimate_distance_from_box_accepts_axis_disagreement_when_gate_is_zero():
    distance = estimate_distance_from_box_mm(
        method="calibrated",
        width_px=60.0,
        height_px=60.0,
        fx_px=600.0,
        fy_px=600.0,
        target_width_mm=30.0,
        target_height_mm=30.0,
        target="fire",
        distance_models={
            "fire": {
                "width": {"a": 18000.0, "b": 0.0},
                "height": {"a": 24000.0, "b": 0.0},
            }
        },
        max_axis_disagreement_mm=0.0,
    )

    assert distance == pytest.approx(350.0)


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


def test_draw_debug_detections_uses_compact_non_overlapping_labels(monkeypatch):
    labels = []

    class FakeCv2:
        MARKER_CROSS = 0
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 0

        @staticmethod
        def rectangle(*_args):
            pass

        @staticmethod
        def drawMarker(*_args):
            pass

        @staticmethod
        def putText(_image, label, position, *_args):
            labels.append((label, position))

    monkeypatch.setitem(sys.modules, "cv2", FakeCv2)
    detections = [
        {"target": "power", "class_id": 0, "confidence": 0.91,
         "box": [10, 30, 45, 65]},
        {"target": "fire", "class_id": 1, "confidence": 0.82,
         "box": [55, 30, 90, 65]},
    ]

    draw_debug_detections(np.zeros((80, 100, 3), dtype=np.uint8), detections)

    assert labels == [("POW91", (10, 25)), ("FIR82", (55, 25))]
