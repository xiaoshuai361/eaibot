import io
import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import block_pick_main as main
from block_detector_protocol import read_message, write_message


EXPECTED = {
    "power": {"class_id": 0, "class_name": "Emergency power supply device"},
    "fire": {"class_id": 1, "class_name": "Fire extinguishing device"},
    "gas": {"class_id": 2, "class_name": "Gas purification device"},
    "support": {"class_id": 3, "class_name": "Structural support device"},
}


def args(*extra):
    return main.parse_args(["--target", "fire"] + list(extra))


def test_class_metadata_and_derived_ids_are_exact():
    assert main.TARGET_CLASSES == EXPECTED
    assert main.TARGET_CLASS_IDS == {key: value["class_id"] for key, value in EXPECTED.items()}


@pytest.mark.parametrize("names", [
    {i: value["class_name"] for i, value in enumerate(EXPECTED.values())},
    [value["class_name"] for value in EXPECTED.values()],
])
def test_model_names_accept_exact_dict_or_list(names):
    main.validate_model_names(names)


@pytest.mark.parametrize("names", [
    {0: "Fire extinguishing device", 1: "Emergency power supply device", 2: "Gas purification device", 3: "Structural support device"},
    {0: "Emergency power supply device", 1: "Fire extinguishing device", 2: "Gas purification device"},
    {0: "Emergency power supply device", 1: "Fire extinguishing device", 2: "Gas purification device", 3: "Structural support device", 4: "extra"},
])
def test_model_names_reject_swapped_missing_or_extra(names):
    with pytest.raises(main.DetectionError):
        main.validate_model_names(names)


def test_defaults_and_required_target():
    parsed = args()
    assert parsed.model == "/home/eaibot/models/Block_yolov8n_640/Block_yolov8n_640_best.pt"
    assert parsed.arm_script == "/home/eaibot/handeye-calib/src/mirobot_pick_test.py"
    assert parsed.confidence == 0.25
    assert parsed.python2 == "python2"
    assert parsed.debug_image == "/tmp/block_grasp_debug.png"
    assert parsed.arm_timeout == 180.0
    with pytest.raises(SystemExit):
        main.parse_args([])


def test_real_argv_accepts_negative_tool_axis(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "block_pick_main.py", "--target", "fire", "--dry-run",
        "--tool-offset", "0.1", "--tool-axis", "-x",
    ])
    assert main.parse_args(None).tool_axis == "-x"


def test_negative_tool_axis_equals_form_is_accepted():
    parsed = main.parse_args([
        "--target", "fire", "--dry-run", "--tool-offset", "0.1",
        "--tool-axis=-x",
    ])
    assert parsed.tool_axis == "-x"


def test_real_argv_rejects_unknown_negative_tool_axis(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "block_pick_main.py", "--target", "fire", "--dry-run",
        "--tool-offset", "0.1", "--tool-axis", "-q",
    ])
    with pytest.raises(SystemExit):
        main.parse_args(None)


def test_script_real_argv_reaches_model_validation_with_negative_tool_axis(tmp_path):
    missing_model = tmp_path / "missing.pt"
    result = subprocess.run(
        [
            sys.executable, main.__file__, "--target", "fire", "--dry-run",
            "--tool-offset", "0.1", "--tool-axis", "-x",
            "--model", str(missing_model),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 1
    assert "Model file does not exist" in result.stderr
    assert "expected one argument" not in result.stderr


@pytest.mark.parametrize("option", [
    "--confidence", "--tool-offset", "--max-tool-camera-angle-deg",
    "--approach-gap", "--velocity-scale", "--acceleration-scale", "--arm-timeout",
])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_all_numeric_cli_values_reject_nonfinite(option, value):
    extra = [option, value, "--dry-run"]
    if option == "--tool-offset":
        extra += ["--tool-axis", "x"]
    with pytest.raises(ValueError, match="finite"):
        main.validate_runtime_args(args(*extra))


@pytest.mark.parametrize("extra", [
    ("--confidence", "0"), ("--confidence", "1.01"),
    ("--velocity-scale", "0"), ("--acceleration-scale", "1.01"),
    ("--approach-gap", "0"), ("--approach-gap", ".151"),
    ("--tool-offset", "-.01", "--tool-axis", "x"),
    ("--tool-offset", ".301", "--tool-axis", "x"),
    ("--max-tool-camera-angle-deg", "0"),
    ("--max-tool-camera-angle-deg", "90"),
    ("--arm-timeout", "0"),
    ("--arm-timeout", "-1"),
])
def test_numeric_boundaries_rejected(extra):
    with pytest.raises(ValueError):
        main.validate_runtime_args(args("--dry-run", *extra))


def test_valid_boundaries_and_tool_pair_rules():
    main.validate_runtime_args(args("--dry-run", "--confidence", "1", "--approach-gap", ".15"))
    with pytest.raises(ValueError, match="together"):
        main.validate_runtime_args(args("--dry-run", "--tool-axis", "x"))
    with pytest.raises(ValueError, match="together"):
        main.validate_runtime_args(args("--dry-run", "--tool-offset", ".1"))
    with pytest.raises(ValueError, match="tool"):
        main.validate_runtime_args(args())
    with pytest.raises(ValueError, match="tool"):
        main.validate_runtime_args(args("--dry-run", "--stop-at-pre-grasp"))
    main.validate_runtime_args(args("--tool-offset", ".1", "--tool-axis", "-z"))


def test_select_unique_detection_requires_exactly_one_matching_confident_box():
    detection = {"class_id": 1, "confidence": .8, "box": [1, 2, 3, 4]}
    assert main.select_unique_detection([detection], 1, .25) is detection
    with pytest.raises(main.DetectionError, match="No detection"):
        main.select_unique_detection([detection], 0, .25)
    with pytest.raises(main.DetectionError, match="No detection"):
        main.select_unique_detection([dict(detection, confidence=.1)], 1, .25)
    with pytest.raises(main.DetectionError, match="Multiple"):
        main.select_unique_detection([detection, dict(detection)], 1, .25)


class Value:
    def __init__(self, value):
        self.value = value
    def tolist(self):
        return self.value


class Model:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []
    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=self.boxes)]


def box(class_id=1, confidence=.75, xyxy=None):
    return SimpleNamespace(cls=Value([class_id]), conf=Value([confidence]), xyxy=Value([xyxy or [1, 2, 101, 102]]))


def test_infer_returns_plain_boxes_and_uses_requested_threshold(tmp_path):
    model = Model([box()])
    result = main.infer_detections(model, str(tmp_path / "image.jpg"), .2)
    assert result == [{"class_id": 1, "confidence": .75, "box": [1.0, 2.0, 101.0, 102.0]}]
    assert model.calls == [{"source": str(tmp_path / "image.jpg"), "imgsz": 640, "conf": .2, "verbose": False}]


@pytest.mark.parametrize("bad_box", [
    box(confidence=float("nan")), box(xyxy=[1, 2, 3]),
    box(xyxy=[1, 2, 1, 3]), box(class_id=float("nan")),
])
def test_infer_rejects_nan_and_bad_shapes(bad_box):
    with pytest.raises(main.DetectionError):
        main.infer_detections(Model([bad_box]), "/tmp/image.jpg", .25)


def test_load_model_is_lazy_and_validates_file_and_names(tmp_path, monkeypatch):
    assert "ultralytics" not in main.__dict__
    with pytest.raises(main.DetectionError, match="file"):
        main.load_model(str(tmp_path / "missing.pt"))
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"fake")
    fake_model = SimpleNamespace(names=[value["class_name"] for value in EXPECTED.values()])
    fake_module = SimpleNamespace(YOLO=lambda path: fake_model)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    assert main.load_model(str(model_file)) is fake_model


def request_stream(*payloads):
    stream = io.StringIO()
    for payload in payloads:
        write_message(stream, payload)
    stream.seek(0)
    return stream


def test_serve_success_error_then_eof(tmp_path, monkeypatch):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    requests = request_stream(
        {"id": 1, "target": "fire", "image_path": str(image)},
        {"id": 2, "target": "unknown", "image_path": str(image)},
        {"id": 3, "target": "fire", "image_path": str(tmp_path / "none")},
    )
    responses = io.StringIO()
    monkeypatch.setattr(main, "infer_detections", lambda *unused: [{"class_id": 1, "confidence": .9, "box": [1, 2, 3, 4]}])
    main.serve_requests(object(), requests, responses, .25)
    responses.seek(0)
    success = read_message(responses)
    assert success == {"id": 1, "ok": True, "target": "fire", "class_id": 1, "class_name": "Fire extinguishing device", "confidence": .9, "box": [1, 2, 3, 4]}
    assert read_message(responses)["ok"] is False
    assert read_message(responses)["ok"] is False
    with pytest.raises(EOFError):
        read_message(responses)


def test_serve_marks_action_phase_before_success_response_write(
    tmp_path, monkeypatch
):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    requests = request_stream(
        {"id": 1, "target": "fire", "image_path": str(image)}
    )
    events = []
    monkeypatch.setattr(
        main,
        "infer_detections",
        lambda *unused: [
            {"class_id": 1, "confidence": .9, "box": [1, 2, 3, 4]}
        ],
    )
    monkeypatch.setattr(
        main,
        "write_message",
        lambda stream, payload: events.append(("write", payload["ok"])),
    )
    main.serve_requests(
        object(), requests, io.StringIO(), .25,
        before_success_response=lambda: events.append("action_phase"),
    )
    assert events == ["action_phase", ("write", True)]


def test_serve_does_not_mark_action_phase_for_error_response(monkeypatch):
    requests = request_stream(
        {"id": 1, "target": "unknown", "image_path": "/missing"}
    )
    events = []
    monkeypatch.setattr(
        main,
        "write_message",
        lambda stream, payload: events.append(("write", payload["ok"])),
    )
    main.serve_requests(
        object(), requests, io.StringIO(), .25,
        before_success_response=lambda: events.append("action_phase"),
    )
    assert events == [("write", False)]


def test_build_child_command_forwards_every_setting():
    parsed = args(
        "--dry-run", "--stop-at-pre-grasp", "--tool-offset", ".12", "--tool-axis", "-x",
        "--approach-gap", ".04", "--max-tool-camera-angle-deg", "19",
        "--velocity-scale", ".06", "--acceleration-scale", ".07", "--debug-image", "/tmp/debug.jpg",
    )
    command = main.build_child_command(parsed, 11, 22)
    assert command[:4] == ["python2", parsed.arm_script, "--mode", "block_grasp"]
    for option, value in {
        "--block-target": "fire", "--detector-request-fd": "11", "--detector-response-fd": "22",
        "--tool-offset": "0.12", "--tool-axis": "-x", "--approach-gap": "0.04",
        "--max-tool-camera-angle-deg": "19.0", "--velocity-scale": "0.06",
        "--acceleration-scale": "0.07", "--debug-image": "/tmp/debug.jpg",
    }.items():
        index = command.index(option)
        assert command[index + 1] == value
    assert "--dry-run" in command
    assert "--stop-at-pre-grasp" in command


def test_build_child_command_always_forwards_default_debug_image():
    command = main.build_child_command(args("--dry-run"), 11, 22)
    index = command.index("--debug-image")
    assert command[index + 1] == "/tmp/block_grasp_debug.png"


def test_arm_timeout_is_parent_only_and_accepts_custom_positive_value():
    parsed = main.validate_runtime_args(args("--dry-run", "--arm-timeout", "245.5"))
    assert parsed.arm_timeout == 245.5
    assert "--arm-timeout" not in main.build_child_command(parsed, 11, 22)


class FakeChild:
    def __init__(self, code=None, timeout=False):
        self.code = code
        self.timeout = timeout
        self.terminated = False
        self.killed = False
        self.wait_timeouts = []
    def poll(self):
        return self.code
    def terminate(self):
        self.terminated = True
    def kill(self):
        self.killed = True
        self.code = -9
    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.timeout and not self.killed:
            raise subprocess.TimeoutExpired("fake", timeout)
        if self.code is None:
            self.code = 0
        return self.code


class NeverReapedChild:
    def __init__(self):
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts = []
    def poll(self):
        return None
    def terminate(self):
        self.terminate_calls += 1
    def kill(self):
        self.kill_calls += 1
    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        raise subprocess.TimeoutExpired("never-reaped", timeout)


class TimeoutThenExitChild(FakeChild):
    pid = 4321

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if len(self.wait_timeouts) == 1:
            raise subprocess.TimeoutExpired("slow-arm", timeout)
        self.code = 0
        return self.code


class ActionPhaseFailureChild(FakeChild):
    pid = 5432

    def __init__(self, error):
        super().__init__(code=None)
        self.error = error

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        raise self.error


class TimeoutThenActionPhaseFailureChild(ActionPhaseFailureChild):
    pid = 6543

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if len(self.wait_timeouts) == 1:
            raise subprocess.TimeoutExpired("slow-arm", timeout)
        raise self.error


def test_stop_child_terminates_then_kills_on_timeout():
    child = FakeChild(timeout=True)
    assert main.stop_child(child) is None
    assert child.terminated and child.killed
    assert child.wait_timeouts == [3.0, 3.0]


def test_stop_child_returns_error_when_child_cannot_be_reaped():
    child = NeverReapedChild()
    cleanup_error = main.stop_child(child)
    assert isinstance(cleanup_error, subprocess.TimeoutExpired)
    assert child.terminate_calls == 1
    assert child.kill_calls == 1
    assert child.wait_timeouts == [main.STOP_CHILD_TIMEOUT, main.STOP_CHILD_TIMEOUT]


def test_main_validates_before_loading(monkeypatch):
    loaded = []
    monkeypatch.setattr(main, "load_model", lambda path: loaded.append(path))
    with pytest.raises(ValueError):
        main.main(["--target", "fire"])
    assert loaded == []


def _run_main(monkeypatch, child, serve=None, track_fds=False, extra=None):
    monkeypatch.setattr(main, "load_model", lambda unused: object())
    monkeypatch.setattr(main, "serve_requests", serve or (lambda *unused: None))
    seen = {}
    if track_fds:
        real_pipe = os.pipe
        pairs = []
        def tracked_pipe():
            pair = real_pipe()
            pairs.append(pair)
            return pair
        monkeypatch.setattr(main.os, "pipe", tracked_pipe)
        seen["pairs"] = pairs
    def popen(command, **kwargs):
        seen.update(command=command, kwargs=kwargs)
        return child
    monkeypatch.setattr(main.subprocess, "Popen", popen)
    result = main.main(["--target", "fire", "--dry-run"] + list(extra or []))
    return result, seen


def test_main_passes_only_child_pipe_ends_and_normal_zero_has_timeout(monkeypatch):
    child = FakeChild(code=0)
    result, seen = _run_main(monkeypatch, child, track_fds=True)
    assert result == 0
    assert seen["kwargs"]["close_fds"] is True
    assert seen["kwargs"]["start_new_session"] is True
    (request_read, request_write), (response_read, response_write) = seen["pairs"]
    assert seen["kwargs"]["pass_fds"] == (request_write, response_read)
    command = seen["command"]
    assert command[command.index("--detector-request-fd") + 1] == str(request_write)
    assert command[command.index("--detector-response-fd") + 1] == str(response_read)
    assert request_read not in seen["kwargs"]["pass_fds"]
    assert response_write not in seen["kwargs"]["pass_fds"]
    assert child.wait_timeouts == [main.NORMAL_CHILD_TIMEOUT]


def test_main_uses_configured_arm_timeout_after_detector_eof(monkeypatch):
    child = FakeChild(code=0)
    result, unused = _run_main(monkeypatch, child, extra=["--arm-timeout", "234"])
    assert result == 0
    assert child.wait_timeouts == [234.0]


def test_main_reports_immediate_nonzero_child(monkeypatch):
    with pytest.raises(RuntimeError, match="status 7"):
        _run_main(monkeypatch, FakeChild(code=7))


@pytest.mark.parametrize("error", [KeyboardInterrupt(), BrokenPipeError("gone")])
def test_main_stops_child_on_serve_failure(monkeypatch, error):
    child = FakeChild(code=None)
    real_pipe = os.pipe
    fds = []
    def tracked_pipe():
        pair = real_pipe()
        fds.extend(pair)
        return pair
    monkeypatch.setattr(main.os, "pipe", tracked_pipe)
    def fail(*unused):
        raise error
    with pytest.raises(type(error)):
        _run_main(monkeypatch, child, fail)
    assert child.terminated
    assert child.wait_timeouts == [main.STOP_CHILD_TIMEOUT]
    assert child.poll() is not None
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("error", [KeyboardInterrupt(), BrokenPipeError("gone")])
def test_main_never_kills_after_success_response_boundary(
    monkeypatch, capsys, error
):
    child = FakeChild(code=None)
    child.pid = 7654

    def success_then_fail(*call_args):
        before_success_response = call_args[4]
        before_success_response()
        raise error

    with pytest.raises(type(error)) as raised:
        _run_main(monkeypatch, child, success_then_fail)
    assert raised.value is error
    assert not child.terminated
    assert not child.killed
    assert child.wait_timeouts == []
    warning = capsys.readouterr().err
    assert "CRITICAL" in warning
    assert "7654" in warning
    assert "UNKNOWN" in warning


@pytest.mark.parametrize("active_error", [KeyboardInterrupt(), BrokenPipeError("gone")])
def test_main_preserves_active_error_when_child_cannot_be_reaped(
    monkeypatch, active_error, capsys
):
    child = NeverReapedChild()
    def fail(*unused):
        raise active_error
    with pytest.raises(type(active_error)) as raised:
        _run_main(monkeypatch, child, fail)
    assert raised.value is active_error
    assert child.terminate_calls == 1
    assert child.kill_calls == 1
    assert child.wait_timeouts == [main.STOP_CHILD_TIMEOUT, main.STOP_CHILD_TIMEOUT]
    stderr = capsys.readouterr().err
    assert "CRITICAL:" in stderr
    assert "arm child may still be running" in stderr
    assert "never-reaped" in stderr


def test_action_timeout_warns_then_waits_without_terminating_child(
    monkeypatch, capsys
):
    child = TimeoutThenExitChild()
    result, unused = _run_main(monkeypatch, child)
    assert result == 0
    assert child.wait_timeouts == [main.NORMAL_CHILD_TIMEOUT, None]
    assert not child.terminated
    assert not child.killed
    warning = capsys.readouterr().err
    assert "CRITICAL" in warning
    assert "4321" in warning
    assert "still be moving" in warning
    assert "UNKNOWN" in warning
    assert "emergency stop" in warning


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("wait failed")])
def test_action_phase_failure_never_kills_child_and_preserves_error(
    monkeypatch, capsys, error
):
    child = ActionPhaseFailureChild(error)
    with pytest.raises(type(error)) as raised:
        _run_main(monkeypatch, child)
    assert raised.value is error
    assert not child.terminated
    assert not child.killed
    assert child.wait_timeouts == [main.NORMAL_CHILD_TIMEOUT]
    warning = capsys.readouterr().err
    assert "CRITICAL" in warning
    assert "5432" in warning
    assert "UNKNOWN" in warning


def test_interrupt_after_action_warning_still_never_kills_and_reports_running(
    monkeypatch, capsys
):
    error = KeyboardInterrupt()
    child = TimeoutThenActionPhaseFailureChild(error)
    with pytest.raises(KeyboardInterrupt) as raised:
        _run_main(monkeypatch, child)
    assert raised.value is error
    assert not child.terminated
    assert not child.killed
    assert child.wait_timeouts == [main.NORMAL_CHILD_TIMEOUT, None]
    warning = capsys.readouterr().err
    assert warning.count("CRITICAL") == 2
    assert "6543" in warning
    assert "UNKNOWN" in warning


def test_popen_failure_closes_every_created_fd(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda unused: object())
    real_pipe = os.pipe
    fds = []
    def tracked_pipe():
        pair = real_pipe()
        fds.extend(pair)
        return pair
    monkeypatch.setattr(main.os, "pipe", tracked_pipe)
    monkeypatch.setattr(main.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError, match="boom"):
        main.main(["--target", "fire", "--dry-run"])
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def _function_node(source, function_name):
    module = ast.parse(source)
    return next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def test_arm_script_block_grasp_delegates_actions_after_dry_run_guard():
    source = Path("handeye-calib/src/mirobot_pick_test.py").read_text(
        encoding="utf-8"
    )
    function = _function_node(source, "do_block_grasp")
    calls = [
        node.func.id for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "compute_block_context" in calls
    assert calls.count("run_block_sequence") == 1
    assert "go_wrist_forward" not in calls
    assert "go_home" not in calls
    assert "publish_debug_geometry" not in calls

    pump_guards = [
        node for node in function.body if isinstance(node, ast.If)
        and "pump_proxy" in ast.dump(node.test, include_attributes=False)
        and "stop_at_pre_grasp" in ast.dump(node.test, include_attributes=False)
    ]
    assert len(pump_guards) == 1
    assert any(isinstance(node, ast.Raise) for node in pump_guards[0].body)

    dry_guard = next(
        node for node in function.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "dry_run"
    )
    assert any(isinstance(node, ast.Return) for node in dry_guard.body)
    sequence_statement_index = next(
        index for index, statement in enumerate(function.body)
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_block_sequence"
            for node in ast.walk(statement)
        )
    )
    assert function.body.index(dry_guard) < sequence_statement_index


def test_arm_main_block_branch_has_safe_arm_and_pump_acquisition_guards():
    source = Path("handeye-calib/src/mirobot_pick_test.py").read_text(
        encoding="utf-8"
    )
    function = _function_node(source, "main")
    rendered = ast.dump(function, include_attributes=False)
    assert "block_grasp" in rendered
    assert "do_block_grasp" in rendered

    build_conditions = []
    pump_conditions = []
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        body_calls = [
            call.func.id for call in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        ]
        condition = ast.dump(node.test, include_attributes=False)
        if "build_move_group" in body_calls:
            build_conditions.append(condition)
        if "get_pump_proxy" in body_calls:
            pump_conditions.append(condition)
    assert any("block_grasp" in condition for condition in build_conditions)
    assert any(
        "block_grasp" in condition
        and "dry_run" in condition
        and "stop_at_pre_grasp" in condition
        for condition in pump_conditions
    )
