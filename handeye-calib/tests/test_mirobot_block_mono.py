import ast
import copy
import json
import math
import os
import time
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
        "time": time,
        "PoseStamped": PoseStamped,
        "STRING_TYPES": (str,),
        "BLOCK_PRESET_VERSION": 2,
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


def test_constrained_block_grasp_replays_translation_with_fixed_orientation():
    (
        finite_scalar,
        finite_vector3,
        normalize_quaternion,
        compute_constrained_block_grasp_pose,
    ) = load_symbols(
        "finite_scalar",
        "finite_vector3",
        "normalize_quaternion",
        "compute_constrained_block_grasp_pose",
    )
    moved_anchor = make_pose(0.40, -0.10, 0.20, q=[0.5, 0.5, 0.5, 0.5])
    pickup_model = {
        "orientation_xyzw_base": [0.0, 0.0, 0.0, 1.0],
        "approach_axis_xyz_base": [-1.0, 0.0, 0.0],
    }
    entry = {"grasp_offset_xyz_base": [0.05, -0.02, 0.04]}
    replay = compute_constrained_block_grasp_pose(
        moved_anchor, pickup_model, entry, "base")

    assert replay.pose.position.x == pytest.approx(0.45)
    assert replay.pose.position.y == pytest.approx(-0.12)
    assert replay.pose.position.z == pytest.approx(0.24)
    assert replay.pose.orientation.x == pytest.approx(0.0)
    assert replay.pose.orientation.y == pytest.approx(0.0)
    assert replay.pose.orientation.z == pytest.approx(0.0)
    assert replay.pose.orientation.w == pytest.approx(1.0)


def test_block_preset_roundtrip_and_overwrite_rules(tmp_path):
    save_block_preset, load_block_preset = load_symbols(
        "save_block_preset", "load_block_preset"
    )
    path = tmp_path / "block_preset.json"
    preset = {
        "version": 2,
        "base_frame": "base",
        "pickup_model": {
            "orientation_xyzw_base": [0.0, 0.0, 0.0, 1.0],
            "approach_axis_xyz_base": [-1.0, 0.0, 0.0],
        },
        "targets": {
            "fire": {
                "grasp_offset_xyz_base": [0.1, 0.2, 0.3],
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


def test_no_tag_places_match_tag_places():
    handeye_root = SCRIPT.parents[1]
    tag_preset = json.loads(
        (handeye_root / "config" / "tag_pick_place_presets.json").read_text(
            encoding="utf-8"))
    block_preset = json.loads(
        (handeye_root / "config" / "block_mono_pick_place_presets.json").read_text(
            encoding="utf-8"))
    mapping = {"power": "1", "fire": "2", "gas": "3", "support": "4"}

    assert block_preset["version"] == 2
    assert block_preset["idle_joint_values"] == tag_preset["idle_joint_values"]
    for target, tag_id in mapping.items():
        assert block_preset["targets"][target]["place_ee_in_base"] == (
            tag_preset["tags"][tag_id]["place_ee_in_base"])


def test_old_block_preset_is_rejected(tmp_path):
    load_block_preset, = load_symbols("load_block_preset")
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"version": 1, "targets": {}}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="version 2"):
        load_block_preset(str(path))


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


def test_get_action_supports_taught_actions():
    get_action, = load_symbols("get_action")
    base = {
        "dry_run": False,
        "live_preview": False,
        "calib_record": False,
        "teach_block_grasp": False,
        "teach_block_place": False,
        "teach_block_idle": False,
        "teach_block_carry": False,
        "preview_taught_block": False,
        "stop_at_taught_pre_grasp": False,
        "run_taught_block": False,
    }

    values = dict(base)
    values["teach_block_grasp"] = True
    assert get_action(SimpleNamespace(**values)) == "teach_block_grasp"

    values = dict(base)
    values["live_preview"] = True
    assert get_action(SimpleNamespace(**values)) == "live_preview"

    values = dict(base)
    values["run_taught_block"] = True
    assert get_action(SimpleNamespace(**values)) == "run_taught_block"

    with pytest.raises(RuntimeError, match="exactly one"):
        get_action(SimpleNamespace(**base))


def test_active_script_does_not_restore_removed_direct_motion_route():
    source = SCRIPT.read_text(encoding="utf-8")

    for removed in (
        "--stop-at-pre-grasp",
        'parser.add_argument("--execute"',
        'parser.add_argument("--teach-block"',
        "build_block_motion_points",
        "require_motion_config",
        "tool_offset_mm",
        "suction_compression_mm",
    ):
        assert removed not in source

    assert 'rospy.init_node("mirobot_pick_test", anonymous=True)' in source


def test_block_grasp_teaching_never_moves_the_arm_automatically():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "record_block_grasp"
    )
    function_source = ast.get_source_segment(
        SCRIPT.read_text(encoding="utf-8"), function)

    assert "execute_pose" not in function_source
    assert "execute_cartesian_pose" not in function_source
    assert "不会自动移动机械臂" in function_source


def test_taught_pose_workspace_validation_uses_pose_in_millimeters():
    finite_scalar, finite_vector3, validate_workspace, validate_pose_workspace = load_symbols(
        "finite_scalar", "finite_vector3", "validate_workspace", "validate_pose_workspace"
    )
    config = {"base_min_z_mm": 40.0, "base_max_radius_mm": 500.0}

    validate_pose_workspace(make_pose(0.20, 0.10, 0.05), config, "valid")
    with pytest.raises(RuntimeError, match="below"):
        validate_pose_workspace(make_pose(0.20, 0.10, 0.03), config, "low")
    with pytest.raises(RuntimeError, match="radius"):
        validate_pose_workspace(make_pose(0.60, 0.0, 0.10), config, "far")


def test_format_localization_summary_contains_pixels_camera_and_base_coords():
    safe_log_text, format_triplet, format_localization_summary = load_symbols(
        "safe_log_text", "format_triplet", "format_localization_summary"
    )
    format_localization_summary.__globals__["safe_log_text"] = safe_log_text
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


def test_safe_log_text_decodes_utf8_from_python2_boundary():
    safe_log_text, ascii_log_text = load_symbols("safe_log_text", "ascii_log_text")
    ascii_log_text.__globals__["safe_log_text"] = safe_log_text

    assert safe_log_text("目标".encode("utf-8")) == "目标"
    assert ascii_log_text("目标") == r"\u76ee\u6807"


def test_ros_logs_do_not_receive_non_ascii_literals():
    source = SCRIPT.read_text(encoding="utf-8")

    for line in source.splitlines():
        if "rospy.log" in line:
            assert line.isascii()


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

    def draw_debug_image(image, detection, observation, **_kwargs):
        return image

    show_rgb_debug.__globals__.update({
        "draw_debug_image": draw_debug_image,
        "ascii_log_text": str,
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
    finite_scalar, collect_observations = load_symbols(
        "finite_scalar", "collect_observations")
    capture = {"rgb": SimpleNamespace(shape=(480, 640, 3)), "stamp_ns": 1}
    detection = {"box": [0, 0, 40, 40], "confidence": 0.9}
    observation = {"u": 20.0, "v": 20.0, "w": 40.0, "h": 40.0}
    calls = {}

    collect_observations.__globals__.update({
        "get_action": lambda args: "dry_run",
        "capture_rgb_once": lambda config: capture,
        "request_detection": lambda detector, target, rgb: detection,
        "is_detection_usable": lambda detected, rules: (True, ""),
        "detection_to_observation": lambda detected: observation,
        "observation_in_roi": lambda *args: (True, ""),
        "DEFAULT_CONFIG": {"grasp_roi_ratio": [0.0, 0.0, 1.0, 1.0]},
        "ascii_log_text": str,
        "stable_median_observation": lambda *args: {},
        "show_rgb_debug": lambda image, detected, observed, wait_ms, **kwargs: calls.setdefault(
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
            "center_std_max_px": 2.0,
            "width_cv_max": 0.03,
        },
        detector=object(),
    )

    assert observations[0]["stamp_ns"] == 1
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


def test_live_preview_reports_each_target_only_once():
    finite_scalar, new_live_preview_labels = load_symbols(
        "finite_scalar", "new_live_preview_labels")
    reported = set()
    detections = [
        {"target": "power", "confidence": 0.91},
        {"target": "fire", "confidence": 0.82},
    ]

    assert new_live_preview_labels(detections, reported) == [
        "power POW91", "fire FIR82"]
    assert new_live_preview_labels(detections, reported) == []
    assert new_live_preview_labels(
        [{"target": "gas", "confidence": 0.76}], reported
    ) == ["gas GAS76"]


def test_collect_all_observations_groups_visible_targets_and_shows_all():
    collect_all_observations, = load_symbols("collect_all_observations")
    capture = {"rgb": SimpleNamespace(shape=(480, 640, 3))}
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
        "observation_in_roi": lambda *args: (True, ""),
        "DEFAULT_CONFIG": {"grasp_roi_ratio": [0.0, 0.0, 1.0, 1.0]},
        "ascii_log_text": str,
        "show_rgb_debug": lambda image, detections, observations, wait_ms, **kwargs: shown.update({
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
