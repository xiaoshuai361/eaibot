import ast
import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "src" / "mirobot_pick_test.py"


class PoseStamped:
    def __init__(self):
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.pose = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )


def load_symbols(*names):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.Assign, ast.ClassDef))
        and getattr(node, "name", None) in names
    ]
    namespace = {
        "copy": copy,
        "json": json,
        "math": math,
        "os": os,
        "PoseStamped": PoseStamped,
        "STRING_TYPES": (str,),
        "BLOCK_PRESET_VERSION": 1,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return [namespace[name] for name in names]


def make_pose(x=0.0, y=0.0, z=0.0, q=None, frame="base"):
    pose = PoseStamped()
    pose.header.frame_id = frame
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    q = q or [0.0, 0.0, 0.0, 1.0]
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    return pose


def test_normalize_vector_rejects_zero_and_returns_unit_vector():
    _, finite_vector3, normalize_vector = load_symbols(
        "finite_scalar", "finite_vector3", "normalize_vector"
    )

    assert normalize_vector((0.0, 3.0, 4.0), "axis") == pytest.approx((0.0, 0.6, 0.8))
    with pytest.raises(RuntimeError, match="axis"):
        normalize_vector((0.0, 0.0, 0.0), "axis")


def test_build_block_motion_points_use_camera_forward_and_tool_offset():
    _, finite_vector3, normalize_vector, build_block_motion_points = load_symbols(
        "finite_scalar",
        "finite_vector3",
        "normalize_vector",
        "build_block_motion_points",
    )

    result = build_block_motion_points(
        surface_base_mm=(100.0, 200.0, 300.0),
        camera_forward_base=(0.0, 0.0, 1.0),
        tool_offset_base_mm=(0.0, 0.0, 20.0),
        target_offset_mm=(1.0, -2.0, 0.0),
        pregrasp_distance_mm=50.0,
        suction_compression_mm=3.0,
    )

    assert result["surface_tcp_mm"] == pytest.approx((101.0, 198.0, 300.0))
    assert result["pregrasp_link_mm"] == pytest.approx((101.0, 198.0, 230.0))
    assert result["contact_link_mm"] == pytest.approx((101.0, 198.0, 283.0))


def test_taught_block_transform_replays_end_effector_relative_to_anchor():
    (
        finite_scalar,
        quaternion_msg_to_tuple,
        normalize_quaternion,
        quaternion_to_matrix,
        quaternion_from_matrix,
        pose_to_matrix,
        transform_to_matrix,
        matrix_multiply,
        inverse_rigid_matrix,
        matrix_to_pose,
        matrix_to_transform,
        compute_grasp_ee_in_block,
        compute_taught_grasp_pose,
    ) = load_symbols(
        "finite_scalar",
        "quaternion_msg_to_tuple",
        "normalize_quaternion",
        "quaternion_to_matrix",
        "quaternion_from_matrix",
        "pose_to_matrix",
        "transform_to_matrix",
        "matrix_multiply",
        "inverse_rigid_matrix",
        "matrix_to_pose",
        "matrix_to_transform",
        "compute_grasp_ee_in_block",
        "compute_taught_grasp_pose",
    )
    taught_anchor = make_pose(0.10, 0.20, 0.30)
    taught_ee = make_pose(0.15, 0.18, 0.34)

    grasp_ee_in_block = compute_grasp_ee_in_block(taught_anchor, taught_ee)
    moved_anchor = make_pose(0.40, -0.10, 0.20)
    replay = compute_taught_grasp_pose(moved_anchor, grasp_ee_in_block, "base")

    assert grasp_ee_in_block["position"] == pytest.approx([0.05, -0.02, 0.04])
    assert replay.pose.position.x == pytest.approx(0.45)
    assert replay.pose.position.y == pytest.approx(-0.12)
    assert replay.pose.position.z == pytest.approx(0.24)


def test_block_preset_roundtrip_and_overwrite_rules(tmp_path):
    save_block_preset, load_block_preset = load_symbols(
        "save_block_preset", "load_block_preset"
    )
    path = tmp_path / "block_preset.json"
    preset = {
        "version": 1,
        "base_frame": "base",
        "targets": {
            "fire": {
                "grasp_ee_in_block": {
                    "position": [0.1, 0.2, 0.3],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "place_ee_in_base": {
                    "position": [0.4, 0.5, 0.6],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
    }

    save_block_preset(str(path), preset, overwrite=False)

    assert load_block_preset(str(path)) == preset
    with pytest.raises(RuntimeError, match="already exists"):
        save_block_preset(str(path), preset, overwrite=False)


def test_block_anchor_pose_and_taught_pregrasp_use_camera_forward():
    (
        finite_scalar,
        finite_vector3,
        normalize_vector,
        normalize_quaternion,
        block_anchor_pose_from_localization,
        build_pregrasp_from_grasp,
    ) = load_symbols(
        "finite_scalar",
        "finite_vector3",
        "normalize_vector",
        "normalize_quaternion",
        "block_anchor_pose_from_localization",
        "build_pregrasp_from_grasp",
    )
    localization = {
        "base_frame": "base",
        "base_xyz_mm": (100.0, 200.0, 300.0),
        "camera_forward_base": (0.0, 0.0, 100.0),
    }

    anchor = block_anchor_pose_from_localization(
        localization, {"block_anchor_orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    )
    grasp = make_pose(0.10, 0.20, 0.30)
    pregrasp = build_pregrasp_from_grasp(
        grasp, localization["camera_forward_base"], 80.0, "base"
    )

    assert anchor.pose.position.x == pytest.approx(0.10)
    assert anchor.pose.position.y == pytest.approx(0.20)
    assert anchor.pose.position.z == pytest.approx(0.30)
    assert pregrasp.pose.position.x == pytest.approx(0.10)
    assert pregrasp.pose.position.y == pytest.approx(0.20)
    assert pregrasp.pose.position.z == pytest.approx(0.22)


def test_require_motion_config_rejects_missing_tool_offset_for_motion():
    _, finite_vector3, require_motion_config = load_symbols(
        "finite_scalar", "finite_vector3", "require_motion_config"
    )
    config = {"tool_offset_mm": None, "distance_method": "calibrated"}

    with pytest.raises(RuntimeError, match="tool_offset_mm"):
        require_motion_config(config, action="execute")


def test_require_motion_config_rejects_theory_for_motion():
    _, finite_vector3, require_motion_config = load_symbols(
        "finite_scalar", "finite_vector3", "require_motion_config"
    )
    config = {"tool_offset_mm": [0.0, 0.0, 0.0], "distance_method": "theory"}

    with pytest.raises(RuntimeError, match="theory"):
        require_motion_config(config, action="stop_at_pre_grasp")


def test_get_action_supports_taught_actions():
    get_action, = load_symbols("get_action")
    base = {
        "dry_run": False,
        "stop_at_pre_grasp": False,
        "execute": False,
        "calib_record": False,
        "teach_block": False,
        "run_taught_block": False,
    }

    values = dict(base)
    values["teach_block"] = True
    assert get_action(SimpleNamespace(**values)) == "teach_block"

    values = dict(base)
    values["run_taught_block"] = True
    assert get_action(SimpleNamespace(**values)) == "run_taught_block"

    with pytest.raises(RuntimeError, match="exactly one"):
        get_action(SimpleNamespace(**base))


def test_format_localization_summary_contains_pixels_camera_and_base_coords():
    format_triplet, format_localization_summary = load_symbols(
        "format_triplet", "format_localization_summary"
    )
    localization = {
        "target": "fire",
        "confidence": 0.91,
        "box": [1.0, 2.0, 31.0, 42.0],
        "u": 16.0,
        "v": 22.0,
        "w": 30.0,
        "h": 40.0,
        "distance_method": "theory",
        "z_mm": 300.0,
        "camera_xyz_mm": (1.0, 2.0, 300.0),
        "base_xyz_mm": (100.0, 200.0, 300.0),
    }

    text = format_localization_summary(localization)

    assert "目标=fire" in text
    assert "置信度=0.910" in text
    assert "框中心=(16.00,22.00)" in text
    assert "框宽px=30.00" in text
    assert "相机坐标mm=(1.00,2.00,300.00)" in text
    assert "机械臂坐标mm=(100.00,200.00,300.00)" in text


def test_show_rgb_debug_waits_forever_when_requested():
    show_rgb_debug, = load_symbols("show_rgb_debug")
    calls = {}
    key_events = [-1, ord("q")]

    class FakeCv2:
        MARKER_CROSS = 0
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 0

        @staticmethod
        def imshow(name, image):
            calls["window"] = name

        @staticmethod
        def waitKey(milliseconds):
            calls.setdefault("milliseconds", []).append(milliseconds)
            return key_events.pop(0)

        @staticmethod
        def rectangle(*args):
            pass

        @staticmethod
        def drawMarker(*args):
            pass

        @staticmethod
        def putText(*args):
            pass

    def draw_debug_image(image, detection, observation):
        return image

    show_rgb_debug.__globals__.update({
        "draw_debug_image": draw_debug_image,
        "rospy": type(
            "Rospy",
            (),
            {
                "loginfo": staticmethod(lambda *args: None),
                "logwarn": staticmethod(lambda *args: None),
            },
        ),
        "__import__": __import__,
    })
    import sys
    sys.modules["cv2"] = FakeCv2
    try:
        show_rgb_debug(
            image=[[0]],
            detection={"box": [0, 0, 1, 1], "confidence": 0.9},
            observation={"u": 0.5, "v": 0.5, "w": 1.0, "h": 1.0},
            milliseconds=0,
        )
    finally:
        sys.modules.pop("cv2", None)

    assert calls["milliseconds"] == [100, 100]


def test_collect_observations_uses_nonblocking_rgb_preview_before_summary():
    collect_observations, = load_symbols("collect_observations")
    capture = {"rgb": object()}
    detection = {"box": [0, 0, 40, 40], "confidence": 0.9}
    observation = {"u": 20.0, "v": 20.0, "w": 40.0, "h": 40.0}
    calls = {}

    collect_observations.__globals__.update({
        "get_action": lambda args: "dry_run",
        "capture_rgb_once": lambda config: capture,
        "request_detection": lambda detector, target, rgb: detection,
        "is_detection_usable": lambda detected, rules: (True, ""),
        "detection_to_observation": lambda detected: observation,
        "show_rgb_debug": lambda image, detected, observed, wait_ms: calls.setdefault(
            "wait_ms", []
        ).append(wait_ms) or True,
        "rospy": type("Rospy", (), {"logwarn": staticmethod(lambda *args: None)}),
    })

    observations, returned_capture = collect_observations(
        SimpleNamespace(frames=1, block_target="fire", show_rgb=True, known_z_mm=None),
        {
            "frames_required": 1,
            "confidence_min": 0.7,
            "box_width_min_px": 30.0,
            "box_aspect_ratio_min": 0.75,
            "box_aspect_ratio_max": 1.30,
        },
        detector=object(),
    )

    assert observations == [observation]
    assert returned_capture is capture
    assert calls["wait_ms"] == [1]


def test_response_detections_handles_single_and_all_responses():
    response_detections, = load_symbols("response_detections")
    single = {"target": "fire", "box": [0, 0, 1, 1], "confidence": 0.9}
    multi = {"target": "all", "detections": [
        {"target": "power", "box": [0, 0, 1, 1], "confidence": 0.8},
        {"target": "fire", "box": [1, 1, 2, 2], "confidence": 0.9},
    ]}

    assert response_detections(single) == [single]
    assert response_detections(multi) == multi["detections"]


def test_collect_all_observations_groups_visible_targets_and_shows_all():
    collect_all_observations, = load_symbols("collect_all_observations")
    capture = {"rgb": object()}
    response = {
        "target": "all",
        "detections": [
            {"target": "fire", "box": [0, 0, 40, 40], "confidence": 0.9},
            {"target": "power", "box": [50, 0, 90, 40], "confidence": 0.8},
        ],
    }
    shown = {}

    def observation_from_detection(detection):
        return {
            "target": detection["target"],
            "confidence": detection["confidence"],
            "box": detection["box"],
            "u": 20.0,
            "v": 20.0,
            "w": 40.0,
            "h": 40.0,
        }

    collect_all_observations.__globals__.update({
        "get_action": lambda args: "dry_run",
        "capture_rgb_once": lambda config: capture,
        "request_detection": lambda detector, target, rgb: response,
        "response_detections": lambda detector_response: detector_response["detections"],
        "is_detection_usable": lambda detected, rules: (True, ""),
        "detection_to_observation": observation_from_detection,
        "show_rgb_debug": lambda image, detections, observations, wait_ms: shown.update({
            "count": len(detections),
            "wait_ms": wait_ms,
        }) or False,
        "rospy": type("Rospy", (), {"logwarn": staticmethod(lambda *args: None)}),
    })

    observations_by_target, returned_capture = collect_all_observations(
        SimpleNamespace(frames=1, block_target=None, show_rgb=True),
        {
            "frames_required": 1,
            "confidence_min": 0.7,
            "box_width_min_px": 30.0,
            "box_aspect_ratio_min": 0.75,
            "box_aspect_ratio_max": 1.30,
        },
        detector=object(),
    )

    assert sorted(observations_by_target) == ["fire", "power"]
    assert len(observations_by_target["fire"]) == 1
    assert len(observations_by_target["power"]) == 1
    assert returned_capture is capture
    assert shown == {"count": 2, "wait_ms": 1}
