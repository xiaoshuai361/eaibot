import json
import subprocess
import sys
from io import StringIO

import cv2
import numpy as np
import pytest

import tag_yolo_roi_detector as detector


EXPECTED_NAMES = {0: "ID1", 1: "ID2", 2: "ID3", 3: "ID4"}


def yolo_detection(class_id=0, confidence=0.8, xyxy=None):
    return {
        "class_id": class_id,
        "class_name": EXPECTED_NAMES[class_id],
        "confidence": confidence,
        "box": xyxy or [20, 30, 80, 90],
    }


class Model:
    names = EXPECTED_NAMES

    def __init__(self, detections):
        self.detections = detections
        self.calls = []

    def detect(self, image_bgr, confidence_threshold):
        self.calls.append((image_bgr, confidence_threshold))
        return self.detections


def test_resolve_model_path_accepts_file_or_directory(tmp_path):
    model_file = tmp_path / "tag_new_yolov5n_640_best.onnx"
    model_file.write_bytes(b"fake")
    assert detector.resolve_model_path(str(model_file)) == str(model_file)
    assert detector.resolve_model_path(str(tmp_path)) == str(model_file)

    (tmp_path / "other.onnx").write_bytes(b"fake")
    assert detector.resolve_model_path(str(tmp_path)) == str(model_file)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(detector.TagDetectionError, match="No .onnx model"):
        detector.resolve_model_path(str(empty_dir))


def test_resolve_model_path_prefers_nested_yolov5_onnx(tmp_path):
    root_model = tmp_path / "other.onnx"
    root_model.write_bytes(b"fake")
    nested = tmp_path / "yolov5"
    nested.mkdir()
    onnx_model = nested / "tag_new_yolov5n_640_best.onnx"
    onnx_model.write_bytes(b"fake")

    assert detector.resolve_model_path(str(tmp_path)) == str(onnx_model)
    assert detector.DEFAULT_MODEL.endswith("/model/yolov5/tag_new_yolov5n_640_best.onnx")


def test_resolve_model_path_rejects_non_onnx_files(tmp_path):
    legacy_model = tmp_path / "tag_yolo_best.weights"
    legacy_model.write_bytes(b"fake")

    with pytest.raises(detector.TagDetectionError, match="must be an .onnx file"):
        detector.resolve_model_path(str(legacy_model))


def test_select_target_box_requires_matching_class_and_unique_result():
    selected = detector.select_target_box(
        [
            {"class_id": 0, "confidence": 0.9, "box": [1, 2, 30, 40]},
            {"class_id": 2, "confidence": 0.7, "box": [5, 6, 35, 46]},
        ],
        target_id=3,
        confidence_threshold=0.5,
    )

    assert selected["class_id"] == 2
    assert selected["class_name"] == "ID3"

    with pytest.raises(detector.TagDetectionError, match="No YOLO detection"):
        detector.select_target_box([selected], target_id=4, confidence_threshold=0.5)
    with pytest.raises(detector.TagDetectionError, match="Multiple"):
        detector.select_target_box([selected, dict(selected)], target_id=3, confidence_threshold=0.5)


def test_infer_yolo_detections_returns_plain_validated_boxes(tmp_path):
    image = tmp_path / "frame.png"
    assert cv2.imwrite(str(image), np.zeros((20, 30, 3), dtype=np.uint8))
    model = Model([yolo_detection(class_id=2, confidence=0.75, xyxy=[1, 2, 20, 18])])

    detections = detector.infer_yolo_detections(model, str(image), 0.25)

    assert detections == [{
        "class_id": 2,
        "class_name": "ID3",
        "confidence": 0.75,
        "box": [1.0, 2.0, 20.0, 18.0],
    }]
    assert len(model.calls) == 1
    assert model.calls[0][0].shape == (20, 30, 3)
    assert model.calls[0][1] == 0.25


def test_infer_yolo_detections_from_image_uses_in_memory_frame():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    model = Model([yolo_detection(class_id=1, confidence=0.85, xyxy=[10, 20, 40, 60])])

    detections = detector.infer_yolo_detections_from_image(model, image, 0.25)

    assert detections == [{
        "class_id": 1,
        "class_name": "ID2",
        "confidence": 0.85,
        "box": [10.0, 20.0, 40.0, 60.0],
    }]
    assert model.calls[0][0] is image
    assert model.calls[0][1] == 0.25


class FakeInput:
    name = "images"


class FakeOnnxSession:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def get_inputs(self):
        return [FakeInput()]

    def run(self, output_names, feed):
        self.calls.append((output_names, feed))
        return self.outputs


def test_onnx_yolov5_inference_maps_letterboxed_xywh_to_original_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    output = np.array([[
        [320.0, 320.0, 80.0, 60.0, 0.90, 0.10, 0.80, 0.05, 0.05],
        [120.0, 100.0, 40.0, 40.0, 0.20, 0.90, 0.02, 0.03, 0.04],
    ]], dtype=np.float32)
    model = detector.OnnxYoloV5Model("fake.onnx", session=FakeOnnxSession([output]))

    detections = detector.infer_yolo_detections_from_image(model, image, 0.25)

    assert detections == [{
        "class_id": 1,
        "class_name": "ID2",
        "confidence": pytest.approx(0.72),
        "box": [280.0, 210.0, 360.0, 270.0],
    }]
    assert model.session.calls[0][0] is None
    assert model.session.calls[0][1]["images"].shape == (1, 3, 640, 640)


def test_onnx_yolov5_inference_accepts_nms_xyxy_output():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    output = np.array([[
        [280.0, 290.0, 360.0, 350.0, 0.84, 2.0],
    ]], dtype=np.float32)
    model = detector.OnnxYoloV5Model("fake.onnx", session=FakeOnnxSession([output]))

    detections = detector.infer_yolo_detections_from_image(model, image, 0.25)

    assert detections == [{
        "class_id": 2,
        "class_name": "ID3",
        "confidence": pytest.approx(0.84),
        "box": [280.0, 210.0, 360.0, 270.0],
    }]


def test_build_roi_variants_pads_with_white_and_maps_corners_back():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[30:70, 40:80] = (20, 20, 20)

    variants = detector.build_roi_variants(image, [40, 30, 80, 70], margin_ratio=0.25, upscale=2)

    assert variants
    raw = variants[0]
    assert raw["image"].shape[:2] == (120, 120)
    assert tuple(raw["image"][0, 0]) == (255, 255, 255)
    points = np.array([[20.0, 20.0], [100.0, 20.0], [100.0, 100.0], [20.0, 100.0]])
    mapped = detector.map_variant_corners_to_image(points, raw)
    np.testing.assert_allclose(mapped, [[40, 30], [80, 30], [80, 70], [40, 70]])


def test_apply_white_quiet_zones_keeps_full_frame_geometry_and_tag_pixels():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    image[30:70, 40:80] = (1, 2, 3)

    output, boxes = detector.apply_white_quiet_zones(
        image,
        [{"class_id": 0, "class_name": "ID1", "confidence": 0.9, "box": [40, 30, 80, 70]}],
        margin_ratio=0.25,
    )

    assert output.shape == image.shape
    assert tuple(output[20, 30]) == (255, 255, 255)
    assert tuple(output[50, 60]) == (1, 2, 3)
    assert tuple(output[5, 5]) == (10, 20, 30)
    assert boxes[0]["outer_box"] == [30, 20, 90, 80]


def test_apply_white_quiet_zones_expands_inner_yolo_box_before_painting_white():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    image[25:75, 35:85] = (1, 2, 3)

    output, boxes = detector.apply_white_quiet_zones(
        image,
        [{"class_id": 0, "class_name": "ID1", "confidence": 0.9, "box": [40, 30, 80, 70]}],
        margin_ratio=0.25,
        box_expand_pixels=5,
    )

    assert boxes[0]["box"] == [40.0, 30.0, 80.0, 70.0]
    assert boxes[0]["inner_box"] == [35, 25, 85, 75]
    assert boxes[0]["outer_box"] == [22, 12, 98, 88]
    assert tuple(output[27, 37]) == (1, 2, 3)
    assert tuple(output[20, 30]) == (255, 255, 255)
    assert tuple(output[5, 5]) == (10, 20, 30)


def test_generate_full_frame_writes_full_size_processed_image(tmp_path):
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    image[30:70, 40:80] = (1, 2, 3)
    image_path = tmp_path / "frame.png"
    output_path = tmp_path / "enhanced.png"
    assert cv2.imwrite(str(image_path), image)
    model = Model([yolo_detection(class_id=0, confidence=0.95, xyxy=[40, 30, 80, 70])])

    result = detector.generate_full_frame({
        "image_path": str(image_path),
        "output_image_path": str(output_path),
        "confidence": 0.25,
        "margin_ratio": 0.25,
    }, model)

    processed = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert result["ok"] is True
    assert processed.shape == image.shape
    assert tuple(processed[20, 30]) == (255, 255, 255)
    assert tuple(processed[50, 60]) == (1, 2, 3)


def test_generate_full_frame_processes_all_four_id_boxes_on_one_image(tmp_path):
    image = np.zeros((160, 180, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    boxes = [
        [10, 10, 30, 30],
        [60, 10, 80, 30],
        [10, 90, 30, 110],
        [60, 90, 80, 110],
    ]
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        image[y1:y2, x1:x2] = (index + 1, index + 2, index + 3)
    image_path = tmp_path / "frame.png"
    output_path = tmp_path / "enhanced.png"
    assert cv2.imwrite(str(image_path), image)
    model = Model([
        yolo_detection(class_id=0, confidence=0.95, xyxy=boxes[0]),
        yolo_detection(class_id=1, confidence=0.95, xyxy=boxes[1]),
        yolo_detection(class_id=2, confidence=0.95, xyxy=boxes[2]),
        yolo_detection(class_id=3, confidence=0.95, xyxy=boxes[3]),
    ])

    result = detector.generate_full_frame({
        "image_path": str(image_path),
        "output_image_path": str(output_path),
        "confidence": 0.25,
        "margin_ratio": 0.25,
    }, model)

    processed = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert processed.shape == image.shape
    assert [item["class_name"] for item in result["detections"]] == ["ID1", "ID2", "ID3", "ID4"]
    assert tuple(processed[5, 5]) == (255, 255, 255)
    assert tuple(processed[5, 55]) == (255, 255, 255)
    assert tuple(processed[85, 5]) == (255, 255, 255)
    assert tuple(processed[85, 55]) == (255, 255, 255)
    assert tuple(processed[20, 20]) == (1, 2, 3)
    assert tuple(processed[20, 70]) == (2, 3, 4)
    assert tuple(processed[100, 20]) == (3, 4, 5)
    assert tuple(processed[100, 70]) == (4, 5, 6)


def test_generate_quiet_frame_from_image_processes_all_four_boxes_without_files():
    image = np.zeros((160, 180, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    boxes = [
        [10, 10, 30, 30],
        [60, 10, 80, 30],
        [10, 90, 30, 110],
        [60, 90, 80, 110],
    ]
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        image[y1:y2, x1:x2] = (index + 1, index + 2, index + 3)
    model = Model([
        yolo_detection(class_id=0, confidence=0.95, xyxy=boxes[0]),
        yolo_detection(class_id=1, confidence=0.95, xyxy=boxes[1]),
        yolo_detection(class_id=2, confidence=0.95, xyxy=boxes[2]),
        yolo_detection(class_id=3, confidence=0.95, xyxy=boxes[3]),
    ])

    processed, detections = detector.generate_quiet_frame_from_image(
        image, model, confidence_threshold=0.25, margin_ratio=0.25)

    assert processed.shape == image.shape
    assert [item["class_name"] for item in detections] == ["ID1", "ID2", "ID3", "ID4"]
    assert tuple(processed[5, 5]) == (255, 255, 255)
    assert tuple(processed[5, 55]) == (255, 255, 255)
    assert tuple(processed[85, 5]) == (255, 255, 255)
    assert tuple(processed[85, 55]) == (255, 255, 255)
    assert tuple(processed[20, 20]) == (1, 2, 3)
    assert tuple(processed[20, 70]) == (2, 3, 4)
    assert tuple(processed[100, 20]) == (3, 4, 5)
    assert tuple(processed[100, 70]) == (4, 5, 6)


def test_generate_quiet_frame_from_image_uses_highest_confidence_duplicate_class():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    image[10:30, 10:30] = (1, 2, 3)
    image[60:80, 60:80] = (4, 5, 6)
    model = Model([
        yolo_detection(class_id=0, confidence=0.30, xyxy=[10, 10, 30, 30]),
        yolo_detection(class_id=0, confidence=0.95, xyxy=[60, 60, 80, 80]),
    ])

    processed, detections = detector.generate_quiet_frame_from_image(
        image, model, confidence_threshold=0.25, margin_ratio=0.25)

    assert len(detections) == 1
    assert detections[0]["box"] == [60.0, 60.0, 80.0, 80.0]
    assert tuple(processed[55, 55]) == (255, 255, 255)
    assert tuple(processed[70, 70]) == (4, 5, 6)
    assert tuple(processed[5, 5]) == (10, 20, 30)


def test_apply_cached_quiet_zones_reuses_boxes_on_new_frame_without_yolo():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    image[30:70, 40:80] = (8, 9, 10)
    cached_boxes = [{
        "class_id": 0,
        "class_name": "ID1",
        "confidence": 0.95,
        "box": [40.0, 30.0, 80.0, 70.0],
        "outer_box": [30, 20, 90, 80],
    }]

    processed, boxes = detector.apply_cached_quiet_zones(image, cached_boxes, margin_ratio=0.25)

    assert processed.shape == image.shape
    assert tuple(processed[20, 30]) == (255, 255, 255)
    assert tuple(processed[50, 60]) == (8, 9, 10)
    assert boxes[0]["class_name"] == "ID1"


def test_draw_yolo_debug_overlay_does_not_touch_tag_inner_box():
    image = np.full((120, 140, 3), 255, dtype=np.uint8)
    image[40:80, 50:90] = (1, 2, 3)
    boxes = [{
        "class_id": 0,
        "class_name": "ID1",
        "confidence": 0.95,
        "box": [50.0, 40.0, 90.0, 80.0],
        "outer_box": [36, 26, 104, 94],
    }]

    output = detector.draw_yolo_debug_overlay(image, boxes)

    assert output.shape == image.shape
    assert tuple(output[60, 70]) == (1, 2, 3)
    assert not np.array_equal(output[26, 36], image[26, 36])


def test_should_refresh_yolo_boxes_respects_interval_and_empty_cache():
    assert detector.should_refresh_yolo_boxes(now=10.0, last_update=9.9, interval=1.0, has_cached_boxes=True) is False
    assert detector.should_refresh_yolo_boxes(now=10.0, last_update=8.9, interval=1.0, has_cached_boxes=True) is True
    assert detector.should_refresh_yolo_boxes(now=10.0, last_update=9.9, interval=1.0, has_cached_boxes=False) is True
    assert detector.should_refresh_yolo_boxes(now=10.0, last_update=9.9, interval=0.0, has_cached_boxes=True) is True


def test_generate_quiet_frame_from_encoded_returns_encoded_full_frame():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    image[20:50, 30:60] = (1, 2, 3)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    model = Model([yolo_detection(class_id=0, confidence=0.95, xyxy=[30, 20, 60, 50])])

    result = detector.generate_quiet_frame_from_encoded({
        "image_bgr_png_base64": detector.base64.b64encode(encoded.tobytes()).decode("ascii"),
        "confidence": 0.25,
        "margin_ratio": 0.25,
    }, model)

    decoded = cv2.imdecode(
        np.frombuffer(detector.base64.b64decode(result["image_bgr_png_base64"]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert result["ok"] is True
    assert decoded.shape == image.shape
    assert result["detections"][0]["class_name"] == "ID1"
    assert tuple(decoded[15, 25]) == (255, 255, 255)


def test_worker_loop_reuses_cached_boxes_when_refresh_is_false():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    image[20:50, 30:60] = (1, 2, 3)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    refresh_payload = json.dumps({
        "image_bgr_png_base64": detector.base64.b64encode(encoded.tobytes()).decode("ascii"),
        "confidence": 0.25,
        "margin_ratio": 0.25,
        "refresh_boxes": True,
    })
    cached_payload = json.dumps({
        "image_bgr_png_base64": detector.base64.b64encode(encoded.tobytes()).decode("ascii"),
        "confidence": 0.25,
        "margin_ratio": 0.25,
        "refresh_boxes": False,
    })
    model = Model([yolo_detection(class_id=0, confidence=0.95, xyxy=[30, 20, 60, 50])])
    stdout = StringIO()

    result_code = detector.worker_loop(model, stdin=StringIO(refresh_payload + "\n" + cached_payload + "\n"), stdout=stdout)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert result_code == 0
    assert len(responses) == 2
    assert responses[0]["ok"] is True
    assert responses[1]["ok"] is True
    assert len(model.calls) == 1
    assert responses[1]["detections"][0]["class_name"] == "ID1"


def test_worker_loop_clears_cached_boxes_when_refresh_finds_no_boxes():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    image[20:50, 30:60] = (1, 2, 3)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    payload = json.dumps({
        "image_bgr_png_base64": detector.base64.b64encode(encoded.tobytes()).decode("ascii"),
        "confidence": 0.25,
        "margin_ratio": 0.25,
        "refresh_boxes": True,
    })
    model = Model([yolo_detection(class_id=0, confidence=0.95, xyxy=[30, 20, 60, 50])])
    original_detect = model.detect

    def detect_once_then_empty(image_bgr, confidence_threshold):
        if len(model.calls) == 0:
            return original_detect(image_bgr, confidence_threshold)
        model.calls.append((image_bgr, confidence_threshold))
        return []

    model.detect = detect_once_then_empty
    stdout = StringIO()

    cached_payload = json.dumps({
        "image_bgr_png_base64": detector.base64.b64encode(encoded.tobytes()).decode("ascii"),
        "confidence": 0.25,
        "margin_ratio": 0.25,
        "refresh_boxes": False,
    })

    result_code = detector.worker_loop(
        model,
        stdin=StringIO(payload + "\n" + payload + "\n" + cached_payload + "\n"),
        stdout=stdout)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert result_code == 0
    assert responses[0]["detections"][0]["class_name"] == "ID1"
    assert responses[1]["detections"] == []
    assert responses[2]["detections"] == []


def test_solve_tag_pose_uses_original_image_corners_and_intrinsics():
    corners = np.array([[300, 220], [340, 220], [340, 260], [300, 260]], dtype=np.float64)
    camera_matrix = np.array([[530.0, 0.0, 320.0], [0.0, 530.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    pose = detector.solve_tag_pose(corners, tag_size_m=0.0145, camera_matrix=camera_matrix, distortion=[0, 0, 0, 0, 0])

    assert pose["tvec"][2] > 0.0
    assert len(pose["quaternion_xyzw"]) == 4
    assert np.linalg.norm(pose["quaternion_xyzw"]) == pytest.approx(1.0)


def test_confirm_decoded_tag_rejects_yolo_decode_mismatch():
    with pytest.raises(detector.TagDetectionError, match="does not match"):
        detector.confirm_decoded_tag(yolo_tag_id=2, decoded_tag_id=3)
    assert detector.confirm_decoded_tag(yolo_tag_id=2, decoded_tag_id=2) == 2


def test_cli_reports_json_error_for_missing_image(tmp_path):
    missing = tmp_path / "missing.png"
    payload = {
        "image_path": str(missing),
        "target_id": 1,
        "camera_info": {
            "K": [530, 0, 320, 0, 530, 240, 0, 0, 1],
            "D": [0, 0, 0, 0, 0],
        },
    }

    result = subprocess.run(
        [sys.executable, detector.__file__, "--model", str(tmp_path / "missing.onnx")],
        input=json.dumps(payload) + "\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    response = json.loads(result.stdout)
    assert result.returncode == 1
    assert response["ok"] is False
    assert "Model file" in response["error"]


def test_detect_tag_pose_confirms_yolo_class_with_real_apriltag_decode(tmp_path):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
    marker = cv2.aruco.generateImageMarker(dictionary, 1, 80)
    image = np.full((180, 220, 3), 245, dtype=np.uint8)
    image[50:130, 70:150] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    image_path = tmp_path / "tag.png"
    assert cv2.imwrite(str(image_path), image)
    model = Model([yolo_detection(class_id=0, confidence=0.95, xyxy=[70, 50, 150, 130])])
    payload = {
        "image_path": str(image_path),
        "target_id": 1,
        "confidence": 0.25,
        "tag_size_m": 0.0145,
        "margin_ratio": 0.35,
        "upscale": 3,
        "camera_info": {
            "K": [530, 0, 110, 0, 530, 90, 0, 0, 1],
            "D": [0, 0, 0, 0, 0],
        },
    }

    result = detector.detect_tag_pose(payload, model)

    assert result["ok"] is True
    assert result["target_id"] == 1
    assert result["class_name"] == "ID1"
    assert result["pose"]["tvec"][2] > 0
    assert result["variant"] in {"padded_color", "clahe_gray", "sharp_gray", "adaptive_threshold"}
