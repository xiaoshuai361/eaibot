import ast
import os
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "src" / "tag_yolo_quiet_zone_relay.py"


def load_functions(*names):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return [namespace[name] for name in names]


def test_default_model_points_to_yolov5_onnx():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
    ]
    values = {}
    for node in assignments:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = ast.literal_eval(node.value)

    assert values["DEFAULT_MODEL"].endswith("/model/yolov5/tag_yolov5n_640_best.onnx")


def test_resolve_python3_executable_prefers_ww_environment_with_yolo_runtime():
    resolve_python3, = load_functions("resolve_python3_executable")
    existing = {
        "/home/eaibot/anaconda3/envs/ww/bin/python3",
        "/usr/bin/python3",
    }
    calls = []

    def exists(path):
        return path in existing

    def can_import(path):
        calls.append(path)
        return path.endswith("/envs/ww/bin/python3")

    result = resolve_python3(
        "auto",
        exists=exists,
        can_import_yolo_runtime=can_import,
        find_executable=lambda name: "/usr/bin/python3",
    )

    assert result == "/home/eaibot/anaconda3/envs/ww/bin/python3"
    assert calls[0] == "/home/eaibot/anaconda3/envs/ww/bin/python3"


def test_resolve_python3_executable_accepts_ww_python_when_python3_name_is_absent():
    resolve_python3, = load_functions("resolve_python3_executable")
    existing = {
        "/home/eaibot/anaconda3/envs/ww/bin/python",
        "/usr/bin/python3",
    }

    def exists(path):
        return path in existing

    def can_import(path):
        return path == "/home/eaibot/anaconda3/envs/ww/bin/python"

    result = resolve_python3(
        "auto",
        exists=exists,
        can_import_yolo_runtime=can_import,
        find_executable=lambda name: "/usr/bin/python3",
    )

    assert result == "/home/eaibot/anaconda3/envs/ww/bin/python"


def test_resolve_python3_executable_accepts_explicit_path_without_auto_detection():
    resolve_python3, = load_functions("resolve_python3_executable")

    result = resolve_python3(
        "/custom/ww/bin/python3",
        exists=lambda path: False,
        can_import_yolo_runtime=lambda path: False,
        find_executable=lambda name: None,
    )

    assert result == "/custom/ww/bin/python3"


def test_should_process_frame_respects_publish_interval():
    should_process, = load_functions("should_process_frame")

    assert should_process(now=10.0, last_publish=9.95, interval=0.2) is False
    assert should_process(now=10.0, last_publish=9.70, interval=0.2) is True
    assert should_process(now=10.0, last_publish=9.95, interval=0.0) is True


def test_relay_exposes_and_forwards_box_expand_pixels():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "DEFAULT_BOX_EXPAND_PIXELS = 0" in source
    assert "parser.add_argument('--box-expand-pixels'" in source
    assert "'box_expand_pixels': float(box_expand_pixels)" in source
    assert "self.args.box_expand_pixels" in source


def test_build_detections_payload_exposes_tag_ids_and_image_size():
    build_payload, = load_functions("build_detections_payload")
    header = type("Header", (), {})()
    header.stamp = type("Stamp", (), {"secs": 12, "nsecs": 34})()
    detections = [{
        "class_id": 3,
        "class_name": "ID4",
        "confidence": 0.87,
        "box": [10.0, 20.0, 30.0, 40.0],
        "outer_box": [5, 15, 35, 45],
    }]

    payload = build_payload(header, image_width=640, image_height=480,
                            detections=detections)

    assert payload["stamp"] == {"secs": 12, "nsecs": 34}
    assert payload["image_width"] == 640
    assert payload["image_height"] == 480
    assert payload["detections"] == [{
        "tag_id": 4,
        "class_id": 3,
        "class_name": "ID4",
        "confidence": 0.87,
        "box": [10.0, 20.0, 30.0, 40.0],
        "outer_box": [5.0, 15.0, 35.0, 45.0],
    }]
