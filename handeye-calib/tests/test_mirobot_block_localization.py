import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from block_grasp_vision import (
    LocalizationError,
    tool_axis_vector,
    validate_rgbd_metadata,
    validate_workspace_points,
)


SCRIPT = Path(__file__).parents[1] / "src" / "mirobot_pick_test.py"


class Header:
    def __init__(self, frame_id="camera_rgb_optical_frame", stamp=10.0):
        self.frame_id = frame_id
        self.stamp = stamp


class Info:
    width = 4
    height = 3
    K = [500.0, 0.0, 2.0, 0.0, 501.0, 1.5, 0.0, 0.0, 1.0]
    D = [0.0, 0.0, 0.0, 0.0, 0.0]
    distortion_model = "plumb_bob"
    header = Header()


def valid_capture():
    return dict(
        rgb=np.zeros((3, 4, 3), dtype=np.uint8),
        depth=np.ones((3, 4), dtype=np.uint16),
        rgb_header=Header(stamp=10.0),
        depth_header=Header(stamp=10.02),
        camera_info=Info(),
        depth_encoding="16UC1",
        slop=0.05,
        stamp_to_sec=float,
    )


def load_pure_script_functions(*names):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "math": math,
        "STRING_TYPES": (str,),
        "TOOL_AXES": ("x", "-x", "y", "-y", "z", "-z"),
        "BLOCK_TARGETS": ("power", "fire", "gas", "support"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return [namespace[name] for name in names]


def valid_block_args(**changes):
    values = dict(
        block_target="fire", detector_request_fd=5, detector_response_fd=6,
        rgb_topic="/rgb", registered_depth_topic="/registered",
        rgb_camera_info_topic="/info", debug_image="/tmp/debug.png",
        rgbd_timeout=5.0, rgbd_slop=0.05, depth_radius=3,
        depth_min_m=0.1, depth_max_m=2.0, depth_min_valid_ratio=0.5,
        depth_max_mad_m=0.01, roi_margin=0.4, roi_min_area_pixels=1000.0,
        roi_max_aspect_error=0.25, roi_min_rectangularity=0.75,
        roi_ambiguity_ratio=0.9, approach_gap=0.03, tool_offset=None,
        tool_axis=None, max_tool_camera_angle_deg=20.0, base_min_z=0.04,
        base_max_radius=0.5, wrist_forward_tolerance=0.03,
        velocity_scale=0.05, acceleration_scale=0.05, tf_timeout=5.0,
        wrist_forward_joint5=-1.5709534265016345,
        dry_run=True, stop_at_pre_grasp=False,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_rgbd_metadata_accepts_registered_uint16_pair():
    result = validate_rgbd_metadata(**valid_capture())
    assert result["fx"] == 500.0
    assert result["fy"] == 501.0


def test_rgbd_metadata_accepts_registered_float32_pair():
    capture = valid_capture()
    capture["depth"] = np.ones((3, 4), dtype=np.float32)
    capture["depth_encoding"] = "32FC1"
    assert validate_rgbd_metadata(**capture)["encoding"] == "32FC1"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"depth": np.ones((3, 4), dtype=np.float32)}, "uint16"),
        ({"depth_encoding": "32FC1"}, "float32"),
        ({"depth_header": Header(frame_id="other")}, "same frame"),
        ({"depth_header": Header(stamp=10.2)}, "slop"),
        ({"rgb": np.zeros((3, 4), dtype=np.float32)}, "BGR uint8"),
    ],
)
def test_rgbd_metadata_fails_closed(change, message):
    capture = valid_capture()
    capture.update(change)
    with pytest.raises(LocalizationError, match=message):
        validate_rgbd_metadata(**capture)


def test_rgbd_metadata_rejects_bad_info_and_nonfinite_distortion():
    info = Info()
    info.K = [0.0] * 9
    capture = valid_capture()
    capture["camera_info"] = info
    with pytest.raises(LocalizationError, match="focal"):
        validate_rgbd_metadata(**capture)

    info = Info()
    info.D = [float("nan")]
    capture["camera_info"] = info
    with pytest.raises(LocalizationError, match="distortion"):
        validate_rgbd_metadata(**capture)


@pytest.mark.parametrize("stamp", [0.0, -1.0, float("nan"), float("inf")])
def test_rgbd_metadata_rejects_nonpositive_or_nonfinite_stamps(stamp):
    for field in ("rgb_header", "depth_header"):
        capture = valid_capture()
        capture[field] = Header(stamp=stamp)
        with pytest.raises(LocalizationError, match="timestamp.*positive"):
            validate_rgbd_metadata(**capture)

    capture = valid_capture()
    info = Info()
    info.header = Header(stamp=stamp)
    capture["camera_info"] = info
    with pytest.raises(LocalizationError, match="timestamp.*positive"):
        validate_rgbd_metadata(**capture)


def test_rgbd_metadata_rejects_stale_camera_info():
    capture = valid_capture()
    info = Info()
    info.header = Header(stamp=9.0)
    capture["camera_info"] = info
    with pytest.raises(LocalizationError, match="CameraInfo.*slop"):
        validate_rgbd_metadata(**capture)


def test_tool_axis_preserves_sign():
    assert tool_axis_vector("x", 0.12) == (0.12, 0.0, 0.0)
    assert tool_axis_vector("-x", 0.12) == (-0.12, 0.0, 0.0)
    assert tool_axis_vector("-z", 0.12) == (0.0, 0.0, -0.12)


def test_block_arg_validation_supports_surface_only_dry_run():
    _, require = load_pure_script_functions("_require_finite", "require_block_args")
    args = valid_block_args()
    assert require(args) is args


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"depth_min_m": float("nan")}, "finite"),
        ({"tool_offset": 0.12}, "provided together"),
        ({"tool_axis": "-x"}, "provided together"),
        ({"dry_run": False}, "required outside"),
        ({"stop_at_pre_grasp": True}, "required outside"),
        ({"detector_request_fd": 6}, "must differ"),
        ({"debug_image": ""}, "non-empty"),
        ({"block_target": "unknown"}, "unsupported"),
    ],
)
def test_block_arg_validation_fails_closed(changes, message):
    _, require = load_pure_script_functions("_require_finite", "require_block_args")
    with pytest.raises(RuntimeError, match=message):
        require(valid_block_args(**changes))


def test_signed_child_values_are_normalized_for_argparse():
    normalize, = load_pure_script_functions("_normalize_signed_args")
    assert normalize(["--tool-axis", "-x", "--tool-offset", "-inf"]) == [
        "--tool-axis=-x", "--tool-offset=-inf"
    ]


def test_subscriber_cleanup_supports_public_and_wrapped_ros_versions():
    unsubscribe, = load_pure_script_functions("_unsubscribe_message_filter")

    class Handle:
        def __init__(self):
            self.calls = 0

        def unregister(self):
            self.calls += 1

    public = Handle()
    unsubscribe(public)
    assert public.calls == 1
    wrapped_handle = Handle()
    wrapped = SimpleNamespace(sub=wrapped_handle)
    unsubscribe(wrapped)
    assert wrapped_handle.calls == 1


def test_debug_geometry_does_not_mutate_inputs_and_uses_block_extra_names():
    publish_debug_geometry, = load_pure_script_functions("publish_debug_geometry")

    class HeaderValue:
        def __init__(self, frame_id, stamp):
            self.frame_id = frame_id
            self.stamp = stamp

    class PoseValue:
        def __init__(self, stamp):
            self.header = HeaderValue("base", stamp)
            self.pose = SimpleNamespace()

    class Publisher:
        def __init__(self, topic, *_args, **_kwargs):
            self.topic = topic
            self.messages = []
            fake_rospy.publishers[topic] = self

        def publish(self, message):
            self.messages.append(message)

    fake_rospy = SimpleNamespace(
        publishers={},
        Publisher=Publisher,
        sleep=lambda _duration: None,
        Time=SimpleNamespace(now=lambda: 999.0),
    )

    class FakeMarkerArray:
        def __init__(self):
            self.markers = []

    current = PoseValue(10.0)
    surface = PoseValue(10.0)
    pre = PoseValue(10.0)
    grasp = PoseValue(10.0)
    namespace = publish_debug_geometry.__globals__
    namespace.update({
        "copy": copy,
        "rospy": fake_rospy,
        "PoseStamped": object,
        "MarkerArray": FakeMarkerArray,
        "create_debug_marker": lambda *values: values,
    })
    publish_debug_geometry(
        "base", current, None, None, None,
        extra_pose_topics={
            "block_surface_base": surface,
            "block_pre_grasp": pre,
            "block_grasp": grasp,
        },
    )

    assert current.header.stamp == 10.0
    assert surface.header.stamp == 10.0
    assert pre.header.stamp == 10.0
    assert grasp.header.stamp == 10.0
    assert set(fake_rospy.publishers) >= {
        "mirobot_pick_debug/current_pose",
        "mirobot_pick_debug/block_surface_base",
        "mirobot_pick_debug/block_pre_grasp",
        "mirobot_pick_debug/block_grasp",
    }
    published_surface = fake_rospy.publishers[
        "mirobot_pick_debug/block_surface_base"
    ].messages[0]
    assert published_surface is not surface
    assert published_surface.header.stamp == 999.0

    fake_rospy.publishers.clear()
    old_current = PoseValue(20.0)
    tag = PoseValue(20.0)
    old_pre = PoseValue(20.0)
    old_grasp = PoseValue(20.0)
    publish_debug_geometry("base", old_current, tag, old_pre, old_grasp)
    assert [pose.header.stamp for pose in
            (old_current, tag, old_pre, old_grasp)] == [20.0] * 4
    assert set(fake_rospy.publishers) >= {
        "mirobot_pick_debug/current_pose",
        "mirobot_pick_debug/tag_in_base",
        "mirobot_pick_debug/pre_grasp",
        "mirobot_pick_debug/grasp",
    }


def test_block_context_warms_listener_before_capture_and_reuses_it_for_tf():
    compute_block_context, = load_pure_script_functions("compute_block_context")
    events = []
    listener = object()
    current_pose = object()
    surface_camera = object()
    surface_base = object()
    localization = {
        "rgb_header": object(),
        "camera_xyz": (0.0, 0.0, 0.5),
        "target": "fire",
        "class_name": "Fire extinguishing device",
        "confidence": 0.9,
        "center": (10.0, 20.0),
        "depth_m": 0.5,
    }

    class Arm:
        def get_current_pose(self):
            events.append("current")
            return current_pose

    args = SimpleNamespace(
        base_frame="base", tf_timeout=5.0, tool_offset=None, dry_run=True,
    )
    globals_ = compute_block_context.__globals__
    globals_.update({
        "require_block_args": lambda _args: events.append("validate"),
        "tf": SimpleNamespace(
            TransformListener=lambda: (events.append("listener") or listener)
        ),
        "warmup_transform_listener": lambda value: (
            events.append("warmup") if value is listener else None
        ),
        "capture_rgbd_once": lambda _args: (events.append("capture") or object()),
        "localize_block": lambda _args, _capture: (
            events.append("localize") or localization
        ),
        "make_camera_point_pose": lambda _header, _xyz: surface_camera,
        "transform_pose_at_stamp": lambda value, _frame, pose, _timeout: (
            events.append("transform") or surface_base
            if value is listener and pose is surface_camera else None
        ),
        "pose_to_text": lambda *_args: "pose",
        "publish_debug_geometry": lambda *_args, **_kwargs: events.append("debug"),
        "rospy": SimpleNamespace(
            loginfo=lambda *_args: None,
            logwarn=lambda *_args: None,
        ),
    })
    result = compute_block_context(args, Arm())

    assert events[:7] == [
        "validate", "listener", "warmup", "capture", "localize", "current",
        "transform",
    ]
    assert result["surface_base"] is surface_base


def test_workspace_validates_contact_and_precontact():
    validate_workspace_points((0.20, 0.10, 0.08), (0.18, 0.10, 0.10), 0.04, 0.50)
    with pytest.raises(LocalizationError, match="contact.*minimum z"):
        validate_workspace_points((0.20, 0.10, 0.03), (0.18, 0.10, 0.10), 0.04, 0.50)
    with pytest.raises(LocalizationError, match="precontact.*radius"):
        validate_workspace_points((0.20, 0.10, 0.08), (0.60, 0.10, 0.10), 0.04, 0.50)


def test_source_preserves_localization_contract_when_motion_is_dispatched():
    source = SCRIPT.read_text(encoding="utf-8")
    for mode in (
        "home", "pump", "grasp", "place", "pick_place", "pick_lift_place",
        "current_pose", "wrist_forward", "block_grasp",
    ):
        assert "'{}'".format(mode) in source
    for name in (
        "require_block_args", "capture_rgbd_once", "localize_block",
        "transform_pose_at_stamp", "make_camera_point_pose",
        "is_wrist_forward_reached", "build_block_poses", "compute_block_context",
    ):
        assert "def {}(".format(name) in source
    for option in (
        "--block-target", "--detector-request-fd", "--detector-response-fd",
        "--rgb-topic", "--registered-depth-topic", "--rgb-camera-info-topic",
        "--depth-radius", "--roi-margin", "--tool-offset", "--tool-axis",
        "--stop-at-pre-grasp", "--debug-image", "--base-min-z",
        "--base-max-radius",
    ):
        assert option in source
    assert "elif args.mode == 'block_grasp'" in source
    assert "def do_block_grasp(" in source
    assert "else:\n            do_pick_place(args, arm, pump_proxy)" not in source
    assert "BLOCK_TARGETS = ('power', 'fire', 'gas', 'support')" in source
    assert "choices=BLOCK_TARGETS" in source
    for topic_name in ("block_surface_base", "block_pre_grasp", "block_grasp"):
        assert "'{}'".format(topic_name) in source
