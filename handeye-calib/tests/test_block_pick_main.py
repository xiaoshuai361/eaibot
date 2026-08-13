import io
import json
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


def test_default_no_tag_probe_distances_match_tag_flow():
    assert main.DEFAULT_CONFIG["teach_assist_distance_mm"] == pytest.approx(85.0)
    assert main.DEFAULT_CONFIG["approach_gap_mm"] == pytest.approx(30.0)
    assert main.DEFAULT_CONFIG["contact_probe"]["max_travel_mm"] == pytest.approx(65.0)
    assert main.DEFAULT_CONFIG["contact_probe"]["staging_step_mm"] == pytest.approx(5.0)
    assert main.DEFAULT_CONFIG["contact_probe"]["step_mm"] == pytest.approx(2.0)
    assert main.DEFAULT_CONFIG["contact_probe"]["retreat_extra_mm"] == pytest.approx(30.0)
    assert "place_preset_file" not in main.DEFAULT_CONFIG


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


def test_approach_gap_must_be_positive_and_is_forwarded():
    parsed = args(
        "--stop-at-taught-pre-grasp", "--approach-gap-mm", "25")

    main.validate_runtime_args(parsed, {"distance_method": "fixed_plane"})
    command = main.build_child_command(parsed, request_fd=11, response_fd=12)
    assert command[command.index("--approach-gap-mm") + 1] == "25.0"

    invalid = args(
        "--stop-at-taught-pre-grasp", "--approach-gap-mm", "0")
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
        "--teach-block-pick-place",
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

    assert "--teach-block-pick-place" in teach_command
    assert "--run-taught-block" in run_command
    assert "--preset-file" in teach_command
    assert "/tmp/block_presets.json" in teach_command
    assert "--preset-file" in run_command
    assert "/tmp/block_presets.json" in run_command
    assert "--overwrite" in teach_command


def test_building_contact_teach_forwards_delivery_and_block_presets():
    parsed = args(
        "--teach-building-contact-release",
        "--preset-file", "/tmp/block_presets.json",
        "--delivery-file", "/tmp/delivery_presets.json",
        "--overwrite",
    )

    main.validate_runtime_args(parsed, {"distance_method": "calibrated"})
    command = main.build_child_command(parsed, request_fd=11, response_fd=12)

    assert "--teach-building-contact-release" in command
    assert command[command.index("--preset-file") + 1] == \
        "/tmp/block_presets.json"
    assert command[command.index("--delivery-file") + 1] == \
        "/tmp/delivery_presets.json"
    assert "--overwrite" in command


def test_building_teach_assist_constrains_position_only():
    source = Path(main.__file__).with_name(
        "mirobot_pick_test.py").read_text(encoding="utf-8")
    execute_start = source.index("def execute_pose")
    execute_source = source[
        execute_start:source.index("\ndef ", execute_start + 1)]
    start = source.index("def teach_building_contact_release")
    function_source = source[start:source.index("\ndef ", start + 1)]

    assert "arm.set_position_target" in execute_source
    assert "position_only=True" in function_source
    assert 'config["teach_assist_base_z_mm"]' in function_source
    assert 'pickup_model["orientation_xyzw_base"]' not in function_source


def test_chassis_sequence_is_targetless_and_ignores_legacy_wait_option():
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
    assert "--wait-key-between-targets" not in command
    assert "--show-rgb" in command


def test_chassis_sequence_forwards_target_limit_and_strict_mode():
    parsed = main.parse_args([
        "--run-chassis-sequence",
        "--sequence", "1,2,3,4",
        "--max-targets", "2",
        "--fail-on-skip",
    ])

    main.validate_runtime_args(parsed, {"distance_method": "calibrated"})
    command = main.build_child_command(parsed, request_fd=11, response_fd=12)

    assert command[command.index("--max-targets") + 1] == "2"
    assert "--fail-on-skip" in command


def test_chassis_sequence_forwards_result_file():
    parsed = main.parse_args([
        "--run-chassis-sequence",
        "--sequence", "1,2,3,4",
        "--max-targets", "2",
        "--result-file", "/tmp/untagged-result.json",
    ])

    main.validate_runtime_args(parsed, {"distance_method": "calibrated"})
    command = main.build_child_command(parsed, request_fd=11, response_fd=12)

    assert command[command.index("--result-file") + 1] == \
        "/tmp/untagged-result.json"


def test_chassis_sequence_forwards_right_side_search_handshake():
    parsed = main.parse_args([
        "--run-chassis-sequence",
        "--search-before-chassis",
        "--search-ready-file", "/tmp/search-ready",
        "--search-trigger-file", "/tmp/search-trigger",
        "--search-release-file", "/tmp/search-release",
        "--search-roi-ratio", "0.60,0.05,0.98,0.95",
        "--search-stable-frames", "4",
        "--search-poll-hz", "2.5",
    ])

    command = main.build_child_command(parsed, request_fd=11, response_fd=12)

    assert "--search-before-chassis" in command
    assert command[command.index("--search-ready-file") + 1] == \
        "/tmp/search-ready"
    assert command[command.index("--search-trigger-file") + 1] == \
        "/tmp/search-trigger"
    assert command[command.index("--search-release-file") + 1] == \
        "/tmp/search-release"
    assert command[command.index("--search-roi-ratio") + 1] == \
        "0.60,0.05,0.98,0.95"
    assert command[command.index("--search-stable-frames") + 1] == "4"
    assert command[command.index("--search-poll-hz") + 1] == "2.5"


def test_result_file_is_only_valid_for_chassis_sequence():
    parsed = args("--dry-run", "--result-file", "/tmp/result.json")

    with pytest.raises(ValueError, match="result-file"):
        main.validate_runtime_args(parsed, {"distance_method": "theory"})


def test_taught_block_actions_require_target_and_non_theory_distance():
    parsed = main.parse_args(["--teach-block-pick-place"])

    with pytest.raises(ValueError, match="--target"):
        main.validate_runtime_args(parsed, {"distance_method": "fixed_plane"})

    parsed = args("--run-taught-block")
    with pytest.raises(ValueError, match="theory"):
        main.validate_runtime_args(parsed, {"distance_method": "theory"})
    main.validate_runtime_args(parsed, {"distance_method": "fixed_plane"})


def test_no_tag_supports_combined_and_separate_teaching_entries():
    source = Path(main.__file__).read_text(encoding="utf-8")

    assert "--teach-block-grasp" not in source
    assert "--teach-block-pick-place" in source
    assert "--teach-block-pregrasp" in source
    assert "--teach-block-place" in source


def test_interactive_teaching_has_no_fixed_child_timeout():
    for option in (
        "--teach-block-pick-place",
        "--teach-block-pregrasp",
        "--teach-block-place",
        "--teach-block-idle",
        "--teach-block-carry",
        "--teach-building-contact-release",
    ):
        argv = [option]
        if option in ("--teach-block-pick-place", "--teach-block-pregrasp",
                      "--teach-block-place",
                      "--teach-building-contact-release"):
            argv = ["--target", "1", option]
        parsed = main.parse_args(argv)
        assert main.child_wait_timeout(parsed) is None

    automatic = args("--run-taught-block")
    assert main.child_wait_timeout(automatic) == pytest.approx(
        main.NORMAL_CHILD_TIMEOUT)


def test_serve_requests_returns_selected_detection(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-real-image")
    requests = io.StringIO(json.dumps({
        "id": 1, "target": "fire", "image_path": str(image),
    }) + "\n")
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
    requests = io.StringIO(json.dumps({
        "id": 1, "image_path": str(image),
    }) + "\n")
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
    requests = io.StringIO(json.dumps({
        "id": 1, "target": "fire", "image_path": str(image),
    }) + "\n")
    responses = io.StringIO()

    class Detector:
        def detect_path(self, _path):
            return []

    main.serve_requests(Detector(), main.DEFAULT_CONFIG, requests, responses)
    responses.seek(0)

    assert '"ok":false' in responses.readline()


def test_parent_interrupt_always_stops_active_arm_child(monkeypatch):
    parsed = args("--teach-block-pick-place")

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


def test_repeated_interrupt_during_cleanup_force_kills_child():
    class FakeChild:
        def __init__(self):
            self.running = True
            self.terminated = False
            self.killed = False

        def poll(self):
            return None if self.running else -9

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            if not self.killed:
                raise KeyboardInterrupt()
            self.running = False
            return -9

        def kill(self):
            self.killed = True

    child = FakeChild()

    assert main.stop_child(child) is None
    assert child.terminated is True
    assert child.killed is True
    assert child.poll() == -9


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
    if hasattr(main.signal, "SIGTSTP"):
        assert installed[main.signal.SIGTSTP] is main.request_shutdown
