import io
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import block_pick_main as main


def args(*extra):
    return main.parse_args(["--target", "fire"] + list(extra))


def test_default_model_and_config_paths_match_robot_layout():
    parsed = args("--dry-run")

    assert parsed.model == (
        "/home/eaibot/handeye-calib/src/model/yolov5/"
        "block_occlusion_yolov5n_640_best.onnx"
    )
    assert parsed.config == "/home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml"
    assert parsed.arm_script == "/home/eaibot/handeye-calib/src/mirobot_pick_test.py"


def test_target_accepts_numeric_aliases():
    parsed = main.parse_args(["--target", "3", "--dry-run"])

    assert parsed.target == "3"


def test_missing_config_file_is_copied_from_canonical_config(tmp_path):
    config_path = tmp_path / "config" / "block_mono_grasp.yaml"

    main.ensure_config_file(str(config_path))

    assert config_path.read_bytes() == Path(main.PACKAGED_CONFIG_PATH).read_bytes()


def test_action_mode_is_required_and_mutually_exclusive():
    with pytest.raises(SystemExit):
        main.parse_args(["--target", "fire"])
    with pytest.raises(SystemExit):
        main.parse_args(["--target", "fire", "--dry-run", "--run-taught-block"])


def test_target_is_optional_for_all_detection_dry_run():
    parsed = main.parse_args(["--dry-run", "--show-rgb"])

    main.validate_runtime_args(parsed, {"distance_method": "theory"})
    command = main.build_child_command(parsed, request_fd=11, response_fd=12)

    assert parsed.target is None
    assert "--block-target" not in command
    assert "--show-rgb" in command


def test_live_preview_is_target_optional_and_forwards_rate():
    parsed = main.parse_args(["--live-preview", "--preview-hz", "1.5"])

    main.validate_runtime_args(parsed, {"distance_method": "theory"})
    command = main.build_child_command(parsed, request_fd=11, response_fd=12)

    assert "--live-preview" in command
    assert command[command.index("--preview-hz") + 1] == "1.5"
    assert "--block-target" not in command


def test_pregrasp_distance_must_be_positive_and_is_forwarded():
    parsed = args(
        "--stop-at-taught-pre-grasp", "--pregrasp-distance-mm", "100")

    main.validate_runtime_args(parsed, {"distance_method": "fixed_plane"})
    command = main.build_child_command(parsed, request_fd=11, response_fd=12)
    assert command[command.index("--pregrasp-distance-mm") + 1] == "100.0"

    invalid = args(
        "--stop-at-taught-pre-grasp", "--pregrasp-distance-mm", "0")
    with pytest.raises(ValueError, match="positive"):
        main.validate_runtime_args(invalid, {"distance_method": "fixed_plane"})


def test_target_is_required_for_motion_actions():
    parsed = main.parse_args(["--run-taught-block"])

    with pytest.raises(ValueError, match="--target"):
        main.validate_runtime_args(parsed, {"distance_method": "calibrated"})


def test_validate_runtime_args_allows_theory_only_for_non_motion():
    dry = args("--dry-run")
    execute = args("--run-taught-block")

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
    assert command[command.index("--supervisor-pid") + 1] == str(os.getpid())
    assert "--dry-run" in command
    assert "--show-rgb" in command
    assert "--confidence" in command
    assert "0.55" in command
    assert "--frames" in command
    assert "7" in command


def test_build_child_command_forwards_taught_block_actions():
    teach = args(
        "--teach-block-grasp",
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

    assert "--teach-block-grasp" in teach_command
    assert "--run-taught-block" in run_command
    assert "--preset-file" in teach_command
    assert "/tmp/block_presets.json" in teach_command
    assert "--preset-file" in run_command
    assert "/tmp/block_presets.json" in run_command
    assert "--overwrite" in teach_command


def test_chassis_sequence_is_targetless_and_forwards_short_options():
    parsed = main.parse_args([
        "--run-chassis-sequence",
        "--sequence", "power,fire,gas,support",
        "--wait-key-between-targets",
        "--show-rgb",
    ])

    main.validate_runtime_args(parsed, {"distance_method": "calibrated"})
    command = main.build_child_command(parsed, request_fd=11, response_fd=12)

    assert "--block-target" not in command
    assert "--run-chassis-sequence" in command
    assert command[command.index("--sequence") + 1] == "power,fire,gas,support"
    assert "--wait-key-between-targets" in command
    assert "--show-rgb" in command


def test_taught_block_actions_require_target_and_non_theory_distance():
    parsed = main.parse_args(["--teach-block-grasp"])

    with pytest.raises(ValueError, match="--target"):
        main.validate_runtime_args(parsed, {"distance_method": "fixed_plane"})

    parsed = args("--run-taught-block")
    with pytest.raises(ValueError, match="theory"):
        main.validate_runtime_args(parsed, {"distance_method": "theory"})
    main.validate_runtime_args(parsed, {"distance_method": "fixed_plane"})


def test_place_teaching_supports_single_target_or_sequence():
    single = main.parse_args(["--target", "3", "--teach-block-place"])
    sequence = main.parse_args([
        "--teach-block-place", "--sequence", "1,2,3,4"])

    main.validate_runtime_args(single, {"distance_method": "calibrated"})
    main.validate_runtime_args(sequence, {"distance_method": "calibrated"})
    single_command = main.build_child_command(single, 11, 12)
    sequence_command = main.build_child_command(sequence, 21, 22)

    assert "--block-target" in single_command
    assert "--sequence" not in single_command
    assert "--block-target" not in sequence_command
    assert sequence_command[sequence_command.index("--sequence") + 1] == "1,2,3,4"
    assert "--run-chassis-sequence" not in sequence_command


def test_interactive_teaching_has_no_fixed_child_timeout():
    for option in (
        "--teach-block-grasp",
        "--teach-block-place",
        "--teach-block-idle",
        "--teach-block-carry",
    ):
        argv = [option]
        if option == "--teach-block-grasp":
            argv = ["--target", "1", option]
        parsed = main.parse_args(argv)
        assert main.child_wait_timeout(parsed) is None

    automatic = args("--run-taught-block")
    assert main.child_wait_timeout(automatic) == pytest.approx(
        main.NORMAL_CHILD_TIMEOUT)


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


def test_parent_interrupt_always_stops_active_arm_child(monkeypatch):
    parsed = args("--teach-block-grasp")

    class FakeChild:
        def __init__(self):
            self.running = True
            self.terminated = False

        def poll(self):
            return None if self.running else 0

        def terminate(self):
            self.terminated = True
            self.running = False

        def wait(self, timeout=None):
            if not self.terminated:
                raise KeyboardInterrupt()
            return 0

        def kill(self):
            self.running = False

    child = FakeChild()
    monkeypatch.setattr(main, "ensure_config_file", lambda _path: False)
    runtime_config = dict(main.DEFAULT_CONFIG)
    runtime_config["distance_method"] = "calibrated"
    monkeypatch.setattr(main, "load_config", lambda _path: runtime_config)
    monkeypatch.setattr(main, "OnnxYoloDetector", lambda *_items: object())
    monkeypatch.setattr(main, "serve_requests", lambda *_items: None)
    monkeypatch.setattr(subprocess, "Popen", lambda *_items, **_kwargs: child)

    with pytest.raises(KeyboardInterrupt):
        main.run_parent(parsed)

    assert child.terminated is True


def test_parent_handles_terminal_close_and_termination_signals(monkeypatch):
    installed = {}
    monkeypatch.setattr(
        main.signal, "signal",
        lambda signum, handler: installed.setdefault(signum, handler),
    )

    main.install_shutdown_handlers()

    assert installed[main.signal.SIGTERM] is main.request_shutdown
    if hasattr(main.signal, "SIGHUP"):
        assert installed[main.signal.SIGHUP] is main.request_shutdown
