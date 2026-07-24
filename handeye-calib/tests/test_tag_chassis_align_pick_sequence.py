import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "src" / "tag_chassis_align_pick_sequence.py"


def load_symbols(*names):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.Assign, ast.ClassDef))
    ]
    namespace = {
        "json": json,
        "math": __import__("math"),
        "os": __import__("os"),
        "subprocess": __import__("subprocess"),
        "sys": __import__("sys"),
        "time": __import__("time"),
        "STRING_TYPES": (str,),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return [namespace[name] for name in names]


def test_parse_roi_ratio_and_convert_to_pixels():
    parse_roi_ratio, roi_ratio_to_pixels = load_symbols(
        "parse_roi_ratio", "roi_ratio_to_pixels")

    ratio = parse_roi_ratio("0.06,0.00,0.24,1.00")
    assert ratio == pytest.approx([0.06, 0.0, 0.24, 1.0])
    assert roi_ratio_to_pixels(ratio, image_width=640, image_height=480) == pytest.approx(
        [38.4, 0.0, 153.6, 480.0])

    with pytest.raises(RuntimeError, match="target ROI"):
        parse_roi_ratio("0.2,0,0.1,1")


def test_select_detection_for_tag_uses_id_and_confidence():
    select_detection_for_tag, = load_symbols("select_detection_for_tag")
    message = {
        "image_width": 640,
        "image_height": 480,
        "detections": [
            {"tag_id": 2, "confidence": 0.2, "box": [200, 10, 240, 50]},
            {"tag_id": 4, "confidence": 0.9, "box": [80, 10, 120, 50]},
        ],
    }

    selected = select_detection_for_tag(message, tag_id=4, min_confidence=0.5)

    assert selected["tag_id"] == 4
    assert selected["box"] == [80.0, 10.0, 120.0, 50.0]
    assert select_detection_for_tag(message, tag_id=2, min_confidence=0.5) is None


def test_compute_drive_command_moves_forward_when_target_is_right_of_roi():
    compute_drive_command, = load_symbols("compute_drive_command")
    roi = [40.0, 0.0, 160.0, 480.0]

    right = compute_drive_command(
        detection={"box": [220.0, 10.0, 260.0, 50.0]},
        roi_pixels=roi,
        drive_speed=0.02,
        tolerance_px=12.0,
        target_right_forward=True)
    left = compute_drive_command(
        detection={"box": [0.0, 10.0, 20.0, 50.0]},
        roi_pixels=roi,
        drive_speed=0.02,
        tolerance_px=12.0,
        target_right_forward=True)
    inside = compute_drive_command(
        detection={"box": [80.0, 10.0, 120.0, 50.0]},
        roi_pixels=roi,
        drive_speed=0.02,
        tolerance_px=12.0,
        target_right_forward=True)

    assert right.linear_x == pytest.approx(0.02)
    assert right.aligned is False
    assert left.linear_x == pytest.approx(-0.02)
    assert inside.linear_x == pytest.approx(0.0)
    assert inside.aligned is True


def test_left_to_right_order_sorts_available_tag_ids_by_x_center():
    left_to_right_order, = load_symbols("left_to_right_order")
    message = {
        "detections": [
            {"tag_id": 4, "confidence": 0.8, "box": [300, 0, 340, 40]},
            {"tag_id": 1, "confidence": 0.8, "box": [10, 0, 50, 40]},
            {"tag_id": 2, "confidence": 0.8, "box": [160, 0, 200, 40]},
        ],
    }

    assert left_to_right_order(message, allowed_sequence=[1, 2, 3, 4],
                               min_confidence=0.5) == [1, 2, 4]


def test_pick_command_includes_home_after_idle_and_requested_scales():
    build_pick_command, = load_symbols("build_pick_command")
    args = SimpleNamespace(
        python2="/usr/bin/python2",
        pick_script="/home/eaibot/handeye-calib/src/mirobot_pick_test_tag.py",
        preset_file="/home/eaibot/handeye-calib/config/tag_pick_place_presets.json",
        pick_velocity_scale=0.1,
        pick_acceleration_scale=0.2,
        pick_motion_settle_seconds=0.4,
        disable_replanning=True,
    )

    command = build_pick_command(args, tag_id=4)

    assert command[:3] == [
        "/usr/bin/python2",
        "/home/eaibot/handeye-calib/src/mirobot_pick_test_tag.py",
        "--mode",
    ]
    assert "--home-after-idle" in command
    assert "--disable-replanning" in command
    assert command[command.index("--sequence") + 1] == "4"
    assert command[command.index("--velocity-scale") + 1] == "0.1"
    assert command[command.index("--acceleration-scale") + 1] == "0.2"
