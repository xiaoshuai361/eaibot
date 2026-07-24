import io
import os
from types import SimpleNamespace

import pytest

import block_pick_main as main


def args(*extra):
    return main.parse_args(["--target", "fire"] + list(extra))


def test_default_model_and_config_paths_match_robot_layout():
    parsed = args("--dry-run")

    assert parsed.model == (
        "/home/eaibot/handeye-calib/src/model/yolov5/"
        "Block_v5n_yolov5n_640_best.onnx"
    )
    assert parsed.config == "/home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml"
    assert parsed.arm_script == "/home/eaibot/handeye-calib/src/mirobot_pick_test.py"


def test_missing_config_file_is_created_with_default_yolov5_model(tmp_path):
    config_path = tmp_path / "config" / "block_mono_grasp.yaml"

    main.ensure_config_file(str(config_path))

    text = config_path.read_text(encoding="utf-8")
    assert "Block_v5n_yolov5n_640_best.onnx" in text
    assert "distance_method: theory" in text


def test_action_mode_is_required_and_mutually_exclusive():
    with pytest.raises(SystemExit):
        main.parse_args(["--target", "fire"])
    with pytest.raises(SystemExit):
        main.parse_args(["--target", "fire", "--dry-run", "--execute"])


def test_target_is_optional_for_all_detection_dry_run():
    parsed = main.parse_args(["--dry-run", "--show-rgb"])

    main.validate_runtime_args(parsed, {"distance_method": "theory"})
    command = main.build_child_command(parsed, request_fd=11, response_fd=12)

    assert parsed.target is None
    assert "--block-target" not in command
    assert "--show-rgb" in command


def test_target_is_required_for_motion_actions():
    parsed = main.parse_args(["--execute"])

    with pytest.raises(ValueError, match="--target"):
        main.validate_runtime_args(parsed, {"distance_method": "calibrated"})


def test_validate_runtime_args_allows_theory_only_for_non_motion():
    dry = args("--dry-run")
    execute = args("--execute")

    main.validate_runtime_args(dry, {"distance_method": "theory"})
    with pytest.raises(ValueError, match="theory"):
        main.validate_runtime_args(execute, {"distance_method": "theory"})
    main.validate_runtime_args(execute, {"distance_method": "calibrated"})


def test_calib_record_requires_known_z_mm():
    with pytest.raises(ValueError, match="known-z-mm"):
        main.validate_runtime_args(args("--calib-record"), {"distance_method": "theory"})


def test_build_child_command_forwards_action_and_ros_debug_flags():
    parsed = args(
        "--dry-run",
        "--show-rgb",
        "--confidence",
        "0.55",
        "--frames",
        "7",
        "--known-z-mm",
        "500",
    )

    command = main.build_child_command(parsed, request_fd=11, response_fd=12)

    assert command[:2] == [main.DEFAULT_PYTHON2, parsed.arm_script]
    assert "--mode" in command
    assert "block_mono" in command
    assert "--block-target" in command
    assert "fire" in command
    assert "--detector-request-fd" in command
    assert "11" in command
    assert "--detector-response-fd" in command
    assert "12" in command
    assert "--dry-run" in command
    assert "--show-rgb" in command
    assert "--confidence" in command
    assert "0.55" in command
    assert "--frames" in command
    assert "7" in command


def test_build_child_command_forwards_taught_block_actions():
    teach = args(
        "--teach-block",
        "--preset-file",
        "/tmp/block_presets.json",
        "--overwrite",
    )
    run = args(
        "--run-taught-block",
        "--preset-file",
        "/tmp/block_presets.json",
    )

    teach_command = main.build_child_command(teach, request_fd=11, response_fd=12)
    run_command = main.build_child_command(run, request_fd=21, response_fd=22)

    assert "--teach-block" in teach_command
    assert "--run-taught-block" in run_command
    assert "--preset-file" in teach_command
    assert "/tmp/block_presets.json" in teach_command
    assert "--preset-file" in run_command
    assert "/tmp/block_presets.json" in run_command
    assert "--overwrite" in teach_command


def test_taught_block_actions_require_target_and_non_theory_distance():
    parsed = main.parse_args(["--teach-block"])

    with pytest.raises(ValueError, match="--target"):
        main.validate_runtime_args(parsed, {"distance_method": "fixed_plane"})

    parsed = args("--run-taught-block")
    with pytest.raises(ValueError, match="theory"):
        main.validate_runtime_args(parsed, {"distance_method": "theory"})
    main.validate_runtime_args(parsed, {"distance_method": "fixed_plane"})


def test_serve_requests_returns_selected_detection(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-real-image")
    requests = io.StringIO(
        '{"id":1,"target":"fire","image_path":"%s"}\n' % str(image)
    )
    responses = io.StringIO()

    class Detector:
        def detect_path(self, path):
            assert path == str(image)
            return [
                {"class_id": 1, "class_name": "Fire extinguishing device", "confidence": 0.91, "box": [1, 2, 31, 42]},
                {"class_id": 0, "class_name": "Emergency power supply device", "confidence": 0.92, "box": [5, 6, 35, 46]},
            ]

    main.serve_requests(
        Detector(),
        main.DEFAULT_CONFIG,
        requests,
        responses,
    )
    responses.seek(0)

    assert responses.readline().strip() == (
        '{"id":1,"ok":true,"target":"fire","class_id":1,'
        '"class_name":"Fire extinguishing device","confidence":0.91,'
        '"box":[1,2,31,42]}'
    )


def test_serve_requests_returns_all_usable_detections_when_target_is_omitted(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-real-image")
    requests = io.StringIO('{"id":1,"image_path":"%s"}\n' % str(image))
    responses = io.StringIO()

    class Detector:
        def detect_path(self, path):
            assert path == str(image)
            return [
                {"class_id": 1, "confidence": 0.91, "box": [1, 2, 41, 42]},
                {"class_id": 0, "confidence": 0.92, "box": [5, 6, 45, 46]},
                {"class_id": 2, "confidence": 0.10, "box": [9, 9, 49, 49]},
            ]

    main.serve_requests(Detector(), main.DEFAULT_CONFIG, requests, responses)
    responses.seek(0)
    response = responses.readline().strip()

    assert response.startswith('{"id":1,"ok":true,"target":"all","detections":[')
    assert '"target":"power"' in response
    assert '"target":"fire"' in response
    assert '"target":"gas"' not in response


def test_serve_requests_reports_business_error_without_crashing(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-real-image")
    requests = io.StringIO(
        '{"id":1,"target":"fire","image_path":"%s"}\n' % str(image)
    )
    responses = io.StringIO()

    class Detector:
        def detect_path(self, _path):
            return []

    main.serve_requests(Detector(), main.DEFAULT_CONFIG, requests, responses)
    responses.seek(0)

    assert '"ok":false' in responses.readline()
