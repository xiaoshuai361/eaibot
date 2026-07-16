import io
import os
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
    with pytest.raises(SystemExit):
        main.parse_args([])


@pytest.mark.parametrize("option", [
    "--confidence", "--tool-offset", "--max-tool-camera-angle-deg",
    "--approach-gap", "--velocity-scale", "--acceleration-scale",
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


def test_build_child_command_forwards_every_setting():
    parsed = args(
        "--dry-run", "--stop-at-pre-grasp", "--tool-offset", ".12", "--tool-axis", "-x",
        "--approach-gap", ".04", "--max-tool-camera-angle-deg", "19",
        "--velocity-scale", ".06", "--acceleration-scale", ".07", "--debug-image", "/tmp/debug.jpg",
    )
    command = main.build_child_command(parsed, 11, 22)
    assert command[:4] == ["python2", parsed.arm_script, "--mode", "block_grasp"]
    for option, value in {
        "--target": "fire", "--detector-request-fd": "11", "--detector-response-fd": "22",
        "--tool-offset": "0.12", "--tool-axis": "-x", "--approach-gap": "0.04",
        "--max-tool-camera-angle-deg": "19.0", "--velocity-scale": "0.06",
        "--acceleration-scale": "0.07", "--debug-image": "/tmp/debug.jpg",
    }.items():
        index = command.index(option)
        assert command[index + 1] == value
    assert "--dry-run" in command
    assert "--stop-at-pre-grasp" in command


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
        return self.code if self.code is not None else 0


def test_stop_child_terminates_then_kills_on_timeout():
    child = FakeChild(timeout=True)
    main.stop_child(child)
    assert child.terminated and child.killed
    assert child.wait_timeouts == [3.0, 3.0]


def test_main_validates_before_loading(monkeypatch):
    loaded = []
    monkeypatch.setattr(main, "load_model", lambda path: loaded.append(path))
    with pytest.raises(ValueError):
        main.main(["--target", "fire"])
    assert loaded == []


def _run_main(monkeypatch, child, serve=None):
    monkeypatch.setattr(main, "load_model", lambda unused: object())
    monkeypatch.setattr(main, "serve_requests", serve or (lambda *unused: None))
    seen = {}
    def popen(command, **kwargs):
        seen.update(command=command, kwargs=kwargs)
        return child
    monkeypatch.setattr(main.subprocess, "Popen", popen)
    result = main.main(["--target", "fire", "--dry-run"])
    return result, seen


def test_main_passes_only_child_pipe_ends_and_normal_zero_has_timeout(monkeypatch):
    child = FakeChild(code=0)
    result, seen = _run_main(monkeypatch, child)
    assert result == 0
    assert seen["kwargs"]["close_fds"] is True
    assert len(seen["kwargs"]["pass_fds"]) == 2
    assert child.wait_timeouts == [main.NORMAL_CHILD_TIMEOUT]


def test_main_reports_immediate_nonzero_child(monkeypatch):
    with pytest.raises(RuntimeError, match="status 7"):
        _run_main(monkeypatch, FakeChild(code=7))


@pytest.mark.parametrize("error", [KeyboardInterrupt(), BrokenPipeError("gone")])
def test_main_stops_child_on_serve_failure(monkeypatch, error):
    child = FakeChild(code=None)
    def fail(*unused):
        raise error
    with pytest.raises(type(error)):
        _run_main(monkeypatch, child, fail)
    assert child.terminated


def test_main_normal_wait_timeout_stops_child_and_reports(monkeypatch):
    child = FakeChild(code=None, timeout=True)
    with pytest.raises(RuntimeError, match="timed out"):
        _run_main(monkeypatch, child)
    assert child.terminated and child.killed


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
