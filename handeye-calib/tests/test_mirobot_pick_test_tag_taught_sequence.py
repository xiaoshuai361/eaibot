import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "src" / "mirobot_pick_test_tag.py"


class PoseStamped:
    def __init__(self):
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.pose = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )


def load_module_symbols(*names):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.Assign, ast.ClassDef))
    ]
    namespace = {
        "copy": copy,
        "json": json,
        "math": __import__("math"),
        "os": __import__("os"),
        "PoseStamped": PoseStamped,
        "STRING_TYPES": (str,),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return [namespace[name] for name in names]


class FakeStdin:
    def __init__(self, lines):
        self.lines = list(lines)

    def readline(self):
        if not self.lines:
            return ""
        return self.lines.pop(0)


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


def make_v3_preset(tag_ids=(1, 2), idle_joint_values=None,
                   carry_joint_values=None):
    preset = {
        "version": 3,
        "base_frame": "base",
        "camera_frame": "camera",
        "pickup_model": {
            "orientation_xyzw_base": [0.0, 0.0, 0.0, 1.0],
            "approach_axis_xyz_base": [-1.0, 0.0, 0.0],
        },
        "tags": {},
    }
    for tag_id in tag_ids:
        preset["tags"][str(tag_id)] = {
            "grasp_offset_xyz_base": [-0.03, 0.0, 0.0],
            "place_ee_in_base": {
                "position": [0.4, 0.0, 0.1],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
    if idle_joint_values is not None:
        preset["idle_joint_values"] = list(idle_joint_values)
    if carry_joint_values is not None:
        preset["carry_joint_values"] = list(carry_joint_values)
    return preset


def test_parse_sequence_accepts_comma_list_and_rejects_invalid_values():
    parse_sequence, = load_module_symbols("parse_sequence")

    assert parse_sequence("1,2,3,4") == [1, 2, 3, 4]
    assert parse_sequence(" 4, 2 ") == [4, 2]

    with pytest.raises(RuntimeError, match="--sequence"):
        parse_sequence("")
    with pytest.raises(RuntimeError, match="positive integer"):
        parse_sequence("1,bad")


def test_parse_args_has_no_post_pick_place_joint_alignment():
    parse_args, = load_module_symbols("parse_args")
    parse_args.__globals__.update({
        "argparse": __import__("argparse"),
        "rospy": SimpleNamespace(myargv=lambda argv: argv),
    })

    args = parse_args(["prog", "--mode", "run_taught_sequence"])

    assert not hasattr(args, "place_align_joints")
    assert not hasattr(args, "carry_joint6_lock")
    assert not hasattr(args, "tag_sample_seconds")
    assert args.tag_min_samples == 3
    assert args.tag_max_age_seconds == pytest.approx(2.0)
    assert args.tf_timeout == pytest.approx(12.0)
    assert args.approach_gap == pytest.approx(0.030)
    assert args.assist_front_gap == pytest.approx(0.065)
    assert args.place_approach_gap == pytest.approx(0.05)
    assert args.velocity_scale == pytest.approx(0.4)
    assert args.acceleration_scale == pytest.approx(0.4)
    assert parse_args(["prog", "--mode", "teach_carry"]).mode == "teach_carry"
    assert parse_args(["prog", "--mode", "teach_place_start"]).mode == \
        "teach_place_start"


def test_teach_tag_pose_rejects_stale_tf_instead_of_falling_back():
    wait_for_tag_pose_in_base, = load_module_symbols("wait_for_tag_pose_in_base")

    class FakeDuration:
        def __init__(self, seconds):
            self.seconds = float(seconds)

        def to_sec(self):
            return self.seconds

    class FakeTime:
        current = 0.0

        def __init__(self, seconds=0.0):
            self.seconds = float(seconds)

        @staticmethod
        def now():
            FakeTime.current += 0.25
            return FakeTime(FakeTime.current)

        def __add__(self, duration):
            return FakeTime(self.seconds + duration.seconds)

        def __sub__(self, other):
            return FakeDuration(self.seconds - other.seconds)

        def __lt__(self, other):
            return self.seconds < other.seconds

        def to_nsec(self):
            return int(self.seconds * 1000000000)

    class FakeListener:
        def getLatestCommonTime(self, base_frame, tag_frame):
            return FakeTime(-10.0)

        def lookupTransform(self, base_frame, tag_frame, stamp):
            return [0.24, 0.08, 0.11], [0.0, 0.0, 0.0, 1.0]

    args = SimpleNamespace(
        mode="teach_tag_grasp",
        base_frame="base",
        camera_frame="camera_rgb_optical_frame",
        tf_timeout=0.5,
        tag_min_samples=2,
        tag_max_mad_m=0.005,
        tag_max_age_seconds=0.5,
    )
    wait_for_tag_pose_in_base.__globals__.update({
        "rospy": SimpleNamespace(
            Time=FakeTime,
            Duration=FakeDuration,
            is_shutdown=lambda: False,
            sleep=lambda seconds: None,
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
        ),
        "tf": SimpleNamespace(
            Exception=Exception,
            LookupException=Exception,
            ConnectivityException=Exception,
            ExtrapolationException=Exception,
        ),
    })

    with pytest.raises(RuntimeError, match="visible but too old"):
        wait_for_tag_pose_in_base(FakeListener(), args, 4)


def test_filter_tag_translation_samples_requires_five_inliers_after_mad_filtering():
    filter_tag_translation_samples, = load_module_symbols("filter_tag_translation_samples")
    samples = [
        {"stamp_ns": 1, "position": [0.100, 0.200, 0.300], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        {"stamp_ns": 2, "position": [0.101, 0.199, 0.300], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        {"stamp_ns": 3, "position": [0.099, 0.201, 0.300], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        {"stamp_ns": 4, "position": [0.100, 0.200, 0.301], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        {"stamp_ns": 5, "position": [0.102, 0.198, 0.299], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        {"stamp_ns": 6, "position": [0.180, 0.260, 0.300], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
    ]

    filtered = filter_tag_translation_samples(
        samples, min_samples=5, max_axis_mad_m=0.005)

    assert filtered["inlier_count"] == 5
    assert filtered["sample_count"] == 6
    assert filtered["position"] == pytest.approx([0.100, 0.200, 0.300])


def test_wait_for_tag_pose_resets_inactivity_timeout_for_each_unique_tf():
    wait_for_tag_pose_in_base, = load_module_symbols("wait_for_tag_pose_in_base")
    logs = []

    class FakeDuration:
        def __init__(self, seconds):
            self.seconds = float(seconds)

        def to_sec(self):
            return self.seconds

    class FakeTime:
        current = 0.0

        def __init__(self, seconds=0.0):
            self.seconds = float(seconds)

        @staticmethod
        def now():
            FakeTime.current += 0.01
            return FakeTime(FakeTime.current)

        def __add__(self, duration):
            return FakeTime(self.seconds + duration.seconds)

        def __sub__(self, other):
            return FakeDuration(self.seconds - other.seconds)

        def __lt__(self, other):
            return self.seconds < other.seconds

        def to_nsec(self):
            return int(self.seconds * 1000000000)

    class FakeListener:
        def __init__(self):
            self.index = -1
            self.positions = [
                [0.100, 0.200, 0.300],
                [0.101, 0.199, 0.300],
                [0.099, 0.201, 0.300],
                [0.100, 0.200, 0.301],
                [0.180, 0.260, 0.300],
                [0.102, 0.198, 0.299],
            ]

        def getLatestCommonTime(self, base_frame, tag_frame):
            self.index = min(self.index + 1, len(self.positions) - 1)
            return FakeTime(FakeTime.current)

        def lookupTransform(self, base_frame, tag_frame, stamp):
            return self.positions[self.index], [0.0, 0.0, 0.0, 1.0]

    args = SimpleNamespace(
        mode="run_taught_sequence",
        base_frame="base",
        camera_frame="camera_rgb_optical_frame",
        # Five samples take longer than this in total in the fake clock. The
        # function must still succeed because every unique TF resets the timer.
        tf_timeout=0.05,
        tag_min_samples=5,
        tag_max_mad_m=0.005,
        tag_max_age_seconds=2.0,
    )
    wait_for_tag_pose_in_base.__globals__.update({
        "rospy": SimpleNamespace(
            Time=FakeTime,
            Duration=FakeDuration,
            is_shutdown=lambda: False,
            sleep=lambda seconds: None,
            loginfo=lambda *items: logs.append(items),
            logwarn=lambda *items: None,
        ),
        "tf": SimpleNamespace(
            Exception=Exception,
            LookupException=Exception,
            ConnectivityException=Exception,
            ExtrapolationException=Exception,
        ),
    })

    pose = wait_for_tag_pose_in_base(FakeListener(), args, 1)

    assert pose.pose.position.x == pytest.approx(0.100)
    assert pose.pose.position.y == pytest.approx(0.200)
    assert pose.pose.position.z == pytest.approx(0.300)
    assert any("稳定位姿锁存" in items[0] for items in logs)


def test_preplace_moves_along_base_z():
    build_pre_place_pose, = load_module_symbols("build_pre_place_pose")
    place_pose = make_pose(0.4, -0.2, 0.1)

    pre_place = build_pre_place_pose(place_pose, 0.02, "base")

    assert pre_place.pose.position.x == pytest.approx(0.4)
    assert pre_place.pose.position.y == pytest.approx(-0.2)
    assert pre_place.pose.position.z == pytest.approx(0.12)


def test_runtime_staging_pose_is_behind_taught_pre_grasp():
    build_backoff_pose, = load_module_symbols("build_backoff_pose")
    taught_pre_grasp = make_pose(0.24, 0.10, 0.12)
    pickup_model = {
        "approach_axis_xyz_base": [-1.0, 0.0, 0.0],
    }

    staging_pose = build_backoff_pose(
        taught_pre_grasp, pickup_model, 0.02, "base")

    assert staging_pose.pose.position.x == pytest.approx(0.22)
    assert staging_pose.pose.position.y == pytest.approx(0.10)
    assert staging_pose.pose.position.z == pytest.approx(0.12)


def test_reteach_records_pre_grasp_and_uses_existing_shared_axis_for_staging():
    (prompt_and_record_grasp,
     compute_taught_pre_grasp_pose,
     build_backoff_pose) = load_module_symbols(
        "prompt_and_record_grasp",
        "compute_taught_pre_grasp_pose",
        "build_backoff_pose",
    )
    tag_pose = make_pose(0.25, 0.10, 0.12)
    taught_pre_grasp = make_pose(0.19, 0.055, 0.12)
    preset = make_v3_preset()
    preset["pickup_model"]["approach_axis_xyz_base"] = [-0.8, -0.6, 0.0]

    class FakeArm:
        def get_current_pose(self):
            return copy.deepcopy(taught_pre_grasp)

    args = SimpleNamespace(
        approach_gap=0.06,
        base_frame="base",
        pickup_approach_axis_base=[-1.0, 0.0, 0.0],
        teach_settle_seconds=0.0,
    )
    prompt_and_record_grasp.__globals__.update({
        "prompt_enter": lambda message: None,
        "rospy": SimpleNamespace(loginfo=lambda *items: None),
    })

    prompt_and_record_grasp(
        args, FakeArm(), preset, 1, tag_pose,
        update_pickup_model=False)
    rebuilt_pre_grasp = compute_taught_pre_grasp_pose(
        tag_pose, preset["pickup_model"], preset["tags"]["1"], "base")
    staging_pose = build_backoff_pose(
        rebuilt_pre_grasp, preset["pickup_model"], 0.06, "base")

    assert rebuilt_pre_grasp.pose.position.x == pytest.approx(
        taught_pre_grasp.pose.position.x)
    assert rebuilt_pre_grasp.pose.position.y == pytest.approx(
        taught_pre_grasp.pose.position.y)
    assert staging_pose.pose.position.x == pytest.approx(0.142)
    assert staging_pose.pose.position.y == pytest.approx(0.019)
    assert staging_pose.pose.position.z == pytest.approx(0.12)


def test_horizontal_tag_outward_axis_keeps_face_yaw():
    horizontal_tag_outward_axis, = load_module_symbols(
        "horizontal_tag_outward_axis")
    root_half = 2 ** 0.5 / 2.0
    sin_15 = __import__("math").sin(__import__("math").radians(15.0))
    cos_15 = __import__("math").cos(__import__("math").radians(15.0))
    tag_pose = make_pose(q=[
        -sin_15 * root_half,
        cos_15 * root_half,
        sin_15 * root_half,
        cos_15 * root_half,
    ])

    axis = horizontal_tag_outward_axis(tag_pose)

    assert axis == pytest.approx([3 ** 0.5 / 2.0, 0.5, 0.0])


def test_teach_assist_pose_stays_at_tag_height_and_uses_horizontal_orientation():
    build_teach_assist_pose, = load_module_symbols("build_teach_assist_pose")
    root_half = 2 ** 0.5 / 2.0
    tag_pose = make_pose(0.20, -0.10, 0.07, q=[0.0, root_half, 0.0, root_half])

    assist_pose = build_teach_assist_pose(
        tag_pose,
        front_gap=0.08,
        orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        base_frame="base",
    )

    assert assist_pose.pose.position.x == pytest.approx(0.28)
    assert assist_pose.pose.position.y == pytest.approx(-0.10)
    assert assist_pose.pose.position.z == pytest.approx(0.07)
    assert assist_pose.pose.orientation.x == pytest.approx(0.0)
    assert assist_pose.pose.orientation.y == pytest.approx(0.0)
    assert assist_pose.pose.orientation.z == pytest.approx(0.0)
    assert assist_pose.pose.orientation.w == pytest.approx(1.0)


def test_teach_assist_pose_rejects_non_horizontal_tag_normal():
    build_teach_assist_pose, = load_module_symbols("build_teach_assist_pose")

    with pytest.raises(RuntimeError, match="horizontal"):
        build_teach_assist_pose(
            make_pose(0.0, 0.0, 0.1),
            front_gap=0.08,
            orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
            base_frame="base",
        )


def test_preset_roundtrip_and_overwrite_rules(tmp_path):
    save_preset, load_preset = load_module_symbols("save_preset", "load_preset")
    path = tmp_path / "preset.json"
    preset = make_v3_preset()

    save_preset(str(path), preset, overwrite=False)
    assert load_preset(str(path)) == preset

    with pytest.raises(RuntimeError, match="already exists"):
        save_preset(str(path), preset, overwrite=False)


def test_record_place_stores_pose_orientation_and_approach_axis_only():
    record_tag_place_in_preset, = load_module_symbols("record_tag_place_in_preset")
    place_pose = make_pose(0.12, -0.2, 0.05, q=[0.1, 0.2, 0.3, 0.9])
    preset = {}

    record_tag_place_in_preset(preset, 2, place_pose)

    entry = preset["tags"]["2"]
    assert entry["place_ee_in_base"]["position"] == pytest.approx([0.12, -0.2, 0.05])
    assert entry["place_orientation_in_base"] == pytest.approx(
        [0.10259783520851541, 0.20519567041703082,
         0.3077935056255462, 0.9233805168766387])
    assert entry["place_approach_axis_in_base"] == pytest.approx([0.0, 0.0, 1.0])
    assert "place_joint_values" not in entry


def test_place_teach_start_is_stored_as_a_full_link6_pose():
    record_start, = load_module_symbols(
        "record_place_teach_start_in_preset")
    preset = {}
    start_pose = make_pose(
        0.22, -0.03, 0.18, q=[0.1, 0.2, 0.3, 0.9])

    record_start(preset, start_pose)

    stored = preset["place_teach_start_ee_in_base"]
    assert stored["position"] == pytest.approx([0.22, -0.03, 0.18])
    assert stored["orientation_xyzw"] == pytest.approx(
        [0.1, 0.2, 0.3, 0.9])


def test_place_teach_requires_the_shared_start_pose():
    require_start, = load_module_symbols("require_place_teach_start")

    with pytest.raises(RuntimeError, match="teach_place_start"):
        require_start({})


def test_place_teach_moves_to_shared_start_before_manual_adjustment():
    prompt_and_record_place, = load_module_symbols("prompt_and_record_place")
    start_pose = make_pose(0.2, 0.0, 0.1)
    final_pose = make_pose(0.4, -0.2, 0.05, q=[0.2, 0.1, 0.3, 0.9])
    preset = make_v3_preset()
    preset["place_teach_start_ee_in_base"] = {
        "position": [0.2, 0.0, 0.1],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    events = []
    prompt_and_record_place.__globals__.update({
        "transform_to_pose": lambda frame, transform: start_pose,
        "execute_pose": lambda arm, pose, label: events.append(
            ("move", pose, label)),
        "prompt_enter": lambda message: events.append(("prompt", message)),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            sleep=lambda seconds: events.append(("sleep", seconds))),
    })
    args = SimpleNamespace(base_frame="base", teach_settle_seconds=0.0)
    arm = SimpleNamespace(get_current_pose=lambda: final_pose)

    prompt_and_record_place(args, arm, preset, 1)

    assert events[0] == ("move", start_pose, "place_teach_start")
    assert events[1][0] == "prompt"
    stored = preset["tags"]["1"]["place_ee_in_base"]
    assert stored["position"] == pytest.approx([0.4, -0.2, 0.05])
    assert stored["orientation_xyzw"] == pytest.approx([0.2, 0.1, 0.3, 0.9])


def test_tag_grasp_also_updates_the_shared_place_teach_start():
    prompt_and_record_grasp, = load_module_symbols("prompt_and_record_grasp")
    probe_start = make_pose(
        0.25, -0.04, 0.17, q=[0.1, 0.2, 0.3, 0.9])
    tag_pose = make_pose(
        0.31, -0.04, 0.17, q=[0.0, 0.70710678, 0.0, 0.70710678])
    preset = {"tags": {"2": {}}}
    prompt_and_record_grasp.__globals__.update({
        "prompt_enter": lambda message: None,
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None, sleep=lambda seconds: None),
    })
    args = SimpleNamespace(
        approach_gap=0.065, base_frame="base", teach_settle_seconds=0.0,
        pickup_approach_axis_base=[-1.0, 0.0, 0.0])
    arm = SimpleNamespace(get_current_pose=lambda: probe_start)

    prompt_and_record_grasp(
        args, arm, preset, 2, tag_pose, update_pickup_model=True)

    stored = preset["place_teach_start_ee_in_base"]
    assert stored["position"] == pytest.approx([0.25, -0.04, 0.17])
    assert stored["orientation_xyzw"] == pytest.approx([0.1, 0.2, 0.3, 0.9])


def test_load_preset_reports_missing_corrupt_and_missing_tag(tmp_path):
    load_preset, require_preset_tags = load_module_symbols(
        "load_preset",
        "require_preset_tags",
    )

    with pytest.raises(RuntimeError, match="does not exist"):
        load_preset(str(tmp_path / "missing.json"))

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="无法解析"):
        load_preset(str(corrupt))

    with pytest.raises(RuntimeError, match="tag 2"):
        require_preset_tags({"tags": {"1": {}}}, [1, 2])


def test_existing_id2_grasp_is_shared_by_all_ids_without_migration():
    require_shared_grasp, = load_module_symbols(
        "require_shared_grasp_offset")
    preset = make_v3_preset(tag_ids=(1, 2, 3, 4))
    preset["tags"]["2"]["grasp_offset_xyz_base"] = [-0.07, 0.01, 0.02]

    offset = require_shared_grasp(preset)

    assert offset == pytest.approx([-0.07, 0.01, 0.02])


def test_prompt_enter_accepts_enter_and_allows_abort():
    prompt_enter, UserAbort = load_module_symbols("prompt_enter", "UserAbort")
    fake_select = SimpleNamespace(
        select=lambda read, write, error, timeout: (read, [], []),
        error=OSError,
    )
    prompt_enter.__globals__.update({
        "errno": __import__("errno"),
        "select": fake_select,
        "sys": SimpleNamespace(stdin=FakeStdin(["\n"])),
        "rospy": SimpleNamespace(is_shutdown=lambda: False),
    })

    prompt_enter("ready")

    prompt_enter.__globals__["sys"] = SimpleNamespace(stdin=FakeStdin(["q\n"]))
    with pytest.raises(UserAbort, match="Aborted"):
        prompt_enter("abort")

    prompt_enter.__globals__.update({
        "select": SimpleNamespace(
            select=lambda read, write, error, timeout: ([], [], []),
            error=OSError,
        ),
        "rospy": SimpleNamespace(is_shutdown=lambda: True),
    })
    with pytest.raises(UserAbort, match="ROS 已中断"):
        prompt_enter("shutdown")


def test_teach_carry_records_current_joint_values(tmp_path):
    teach_carry, = load_module_symbols("teach_carry")
    preset = make_v3_preset()
    saved = {}

    class FakeArm:
        def get_current_joint_values(self):
            return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    args = SimpleNamespace(
        preset_file=str(tmp_path / "preset.json"),
        overwrite=True,
        base_frame="base",
        camera_frame="camera",
    )
    teach_carry.__globals__.update({
        "load_or_create_preset": lambda args: (preset, True),
        "prompt_enter": lambda text: None,
        "save_preset": lambda path, data, overwrite: saved.update({
            "path": path,
            "preset": copy.deepcopy(data),
            "overwrite": overwrite,
        }),
        "rospy": SimpleNamespace(loginfo=lambda *items: None),
    })

    teach_carry(args, FakeArm())

    assert saved["path"] == args.preset_file
    assert saved["overwrite"] is True
    assert saved["preset"]["carry_joint_values"] == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


def test_run_taught_sequence_dry_run_does_not_move_or_pump():
    run_taught_sequence, = load_module_symbols("run_taught_sequence")
    args = SimpleNamespace(
        sequence=[1],
        preset_file="/tmp/unused.json",
        base_frame="base",
        camera_frame="camera",
        tf_timeout=1.0,
        approach_gap=0.03,
        place_approach_gap=0.02,
        dry_run=True,
        debug_hold_seconds=0.0,
        assist_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    preset = make_v3_preset(idle_joint_values=[0.0, 0.1, 0.2])
    events = []

    run_taught_sequence.__globals__.update({
        "load_preset": lambda path: preset,
        "wait_for_tag_pose_in_base": lambda listener, args, tag_id: make_pose(0.2, 0.0, 0.1),
        "publish_debug_geometry": lambda *items, **kwargs: events.append("debug"),
        "execute_pose": lambda arm, pose, label: events.append(label),
        "execute_cartesian_pose": lambda arm, pose, label, **kwargs: (
            events.append(label)),
        "execute_joint_values": lambda *items: events.append("idle"),
        "set_pump": lambda *items: events.append("pump"),
        "tf": SimpleNamespace(TransformListener=lambda: object()),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: events.append("dry_run"),
            sleep=lambda seconds: events.append("sleep"),
        ),
    })

    run_taught_sequence(args, object(), None)

    assert "debug" in events
    assert "dry_run" in events
    assert "execute_pose" not in events
    assert "cartesian" not in events
    assert "idle" not in events
    assert "pump" not in events


def test_run_taught_sequence_moves_to_idle_after_each_successful_tag_before_next_tag():
    run_taught_sequence, = load_module_symbols("run_taught_sequence")
    args = SimpleNamespace(
        sequence=[1, 2],
        preset_file="/tmp/unused.json",
        base_frame="base",
        camera_frame="camera",
        tf_timeout=1.0,
        approach_gap=0.03,
        place_approach_gap=0.02,
        dry_run=False,
        debug_hold_seconds=0.0,
        home_after_idle=False,
        assist_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    preset = make_v3_preset(
        tag_ids=(1, 2), idle_joint_values=[0.0, 0.1, 0.2])
    events = []

    run_taught_sequence.__globals__.update({
        "load_preset": lambda path: preset,
        "wait_for_tag_pose_in_base": lambda listener, args, tag_id: (
            events.append(("wait_tag", tag_id)) or make_pose(0.2, 0.0, 0.1)
        ),
        "publish_debug_geometry": lambda *items, **kwargs: events.append("debug"),
        "execute_pose": lambda arm, pose, label: events.append(label),
        "execute_cartesian_pose": lambda arm, pose, label, **kwargs: (
            events.append(label)),
        "execute_joint_values": lambda arm, values, label: events.append(("idle", values, label)),
        "run_contact_approach": lambda *items: events.append("probe") or True,
        "set_pump": lambda *items: events.append("pump"),
        "tf": SimpleNamespace(TransformListener=lambda: object()),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })

    run_taught_sequence(
        args, object(), object(), contact_proxies=(object(), object()))

    idle_event = ("idle", [0.0, 0.1, 0.2], "idle")
    assert events.count(idle_event) == 2
    assert events.count("approach_staging") == 2
    assert events.count("probe") == 2
    first_wait = events.index(("wait_tag", 1))
    second_wait = events.index(("wait_tag", 2))
    first_staging = events.index("approach_staging", first_wait)
    first_probe = events.index("probe", first_staging)
    assert first_staging < first_probe < second_wait
    second_staging = events.index("approach_staging", second_wait)
    second_probe = events.index("probe", second_staging)
    assert second_staging < second_probe
    assert events.index(idle_event) < events.index(("wait_tag", 2))


def test_run_taught_sequence_moves_to_carry_between_grasp_and_place():
    run_taught_sequence, = load_module_symbols("run_taught_sequence")
    args = SimpleNamespace(
        sequence=[1],
        preset_file="/tmp/unused.json",
        base_frame="base",
        camera_frame="camera",
        tf_timeout=1.0,
        approach_gap=0.03,
        place_approach_gap=0.02,
        dry_run=False,
        debug_hold_seconds=0.0,
        home_after_idle=False,
        assist_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    carry_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    preset = make_v3_preset(carry_joint_values=carry_values)
    events = []
    cartesian_targets = {}

    def execute_cartesian(_arm, pose, label, *items, **kwargs):
        events.append(label)
        cartesian_targets[label] = copy.deepcopy(pose)

    run_taught_sequence.__globals__.update({
        "load_preset": lambda path: preset,
        "wait_for_tag_pose_in_base": lambda listener, args, tag_id: make_pose(0.2, 0.0, 0.1),
        "publish_debug_geometry": lambda *items, **kwargs: None,
        "execute_pose": lambda arm, pose, label: events.append(label),
        "execute_cartesian_pose": execute_cartesian,
        "execute_joint_values": lambda arm, values, label: events.append((label, list(values))),
        "run_contact_approach": lambda *items: True,
        "set_pump": lambda *items: events.append("pump"),
        "tf": SimpleNamespace(TransformListener=lambda: object()),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })

    run_taught_sequence(
        args, object(), object(), contact_proxies=(object(), object()))

    carry_event = ("carry", carry_values)
    assert carry_event in events
    assert events.index("approach_staging") < events.index("pickup_retreat")
    assert events.index("pickup_retreat") < events.index(carry_event)
    assert events.index(carry_event) < events.index("taught_pre_place")
    # tag x=0.20, taught offset=-0.03, approach axis=-X:
    # Taught pre-grasp x=0.17; retreat ends 30mm behind it at x=0.14.
    assert cartesian_targets["pickup_retreat"].pose.position.x == pytest.approx(0.14)


def test_run_taught_sequence_ignores_stale_place_joint_values_after_pickup():
    run_taught_sequence, = load_module_symbols("run_taught_sequence")
    args = SimpleNamespace(
        sequence=[1],
        preset_file="/tmp/unused.json",
        base_frame="base",
        camera_frame="camera",
        tf_timeout=1.0,
        approach_gap=0.03,
        place_approach_gap=0.02,
        dry_run=False,
        debug_hold_seconds=0.0,
        home_after_idle=False,
        assist_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    preset = make_v3_preset()
    preset["tags"]["1"]["place_joint_values"] = [
        0.0, 0.1, 0.2, 0.3, 0.4, 1.5]
    events = []

    class FakeArm:
        def get_current_joint_values(self):
            events.append("read_joints")
            return [1.0, 1.1, 1.2, 1.3, 1.4, -2.0]

    def fake_execute_joint_values(arm, values, label):
        events.append((label, list(values)))

    run_taught_sequence.__globals__.update({
        "load_preset": lambda path: preset,
        "wait_for_tag_pose_in_base": lambda listener, args, tag_id: make_pose(0.2, 0.0, 0.1),
        "publish_debug_geometry": lambda *items, **kwargs: None,
        "execute_pose": lambda arm, pose, label: events.append(label),
        "execute_cartesian_pose": lambda arm, pose, label, *args, **kwargs: events.append(label),
        "execute_joint_values": fake_execute_joint_values,
        "run_contact_approach": lambda *items: True,
        "set_pump": lambda *items: events.append("pump"),
        "tf": SimpleNamespace(TransformListener=lambda: object()),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })

    run_taught_sequence(
        args, FakeArm(), object(), contact_proxies=(object(), object()))

    assert not any(
        isinstance(event, tuple) and event[0] == "taught_place_align_joints"
        for event in events)
    assert events.index("taught_pre_place") < events.index("taught_place")


def test_execute_cartesian_pose_retries_without_collision_check_then_falls_back_to_pose():
    execute_cartesian_pose, = load_module_symbols("execute_cartesian_pose")
    target_pose = make_pose(0.1, 0.2, 0.3)
    events = []

    class FakeArm:
        def set_start_state_to_current_state(self):
            events.append("start")

        def compute_cartesian_path(self, waypoints, eef_step, jump_threshold, avoid_collisions):
            events.append(("cartesian", avoid_collisions))
            return SimpleNamespace(joint_trajectory=SimpleNamespace(points=[])), 0.5

        def set_pose_target(self, pose):
            events.append("pose_target")

        def go(self, wait=True):
            events.append("go")
            return True

        def stop(self):
            events.append("stop")

        def clear_pose_targets(self):
            events.append("clear")

    execute_cartesian_pose.__globals__.update({
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: events.append("warn"),
            sleep=lambda seconds: events.append(("sleep", seconds)),
        ),
    })

    execute_cartesian_pose(
        FakeArm(),
        target_pose,
        "taught_place",
        retry_without_collisions=True,
        fallback_to_pose=True,
    )

    assert ("cartesian", True) in events
    assert ("cartesian", False) in events
    assert "pose_target" in events
    assert "go" in events


def test_execute_cartesian_pose_resyncs_replans_and_retries_from_current_state_after_execute_failure():
    execute_cartesian_pose, = load_module_symbols("execute_cartesian_pose")
    target_pose = make_pose(0.1, 0.2, 0.3)
    events = []

    class FakeArm:
        def __init__(self):
            self.execute_calls = 0

        def set_start_state_to_current_state(self):
            events.append("start")

        def compute_cartesian_path(self, waypoints, eef_step, jump_threshold, avoid_collisions):
            events.append(("cartesian", len(events), avoid_collisions))
            return SimpleNamespace(joint_trajectory=SimpleNamespace(points=[object()])), 1.0

        def execute(self, plan, wait=True):
            self.execute_calls += 1
            events.append(("execute", self.execute_calls))
            return self.execute_calls >= 2

        def stop(self):
            events.append("stop")

        def clear_pose_targets(self):
            events.append("clear")

    execute_cartesian_pose.__globals__.update({
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: events.append("warn"),
            sleep=lambda seconds: events.append(("sleep", seconds)),
        ),
    })

    execute_cartesian_pose(FakeArm(), target_pose, "taught_grasp")

    assert events.count("start") == 2
    assert ("execute", 1) in events
    assert ("execute", 2) in events
    assert "warn" in events
    assert ("sleep", 0.5) in events


def test_execute_pose_waits_after_successful_motion_to_let_actionlib_settle():
    execute_pose, = load_module_symbols("execute_pose")
    target_pose = make_pose(0.1, 0.2, 0.3)
    events = []

    class FakeArm:
        def set_start_state_to_current_state(self):
            events.append("start")

        def set_pose_target(self, pose):
            events.append("pose_target")

        def go(self, wait=True):
            events.append("go")
            return True

        def stop(self):
            events.append("stop")

        def clear_pose_targets(self):
            events.append("clear")

    execute_pose.__globals__.update({
        "MOTION_SETTLE_SECONDS": 0.25,
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            sleep=lambda seconds: events.append(("sleep", seconds)),
        ),
    })

    execute_pose(FakeArm(), target_pose, "taught_pre_grasp")

    assert events[-1] == ("sleep", 0.25)


def test_execute_pose_resyncs_and_retries_after_first_go_failure():
    execute_pose, = load_module_symbols("execute_pose")
    target_pose = make_pose(0.1, 0.2, 0.3)
    events = []

    class FakeArm:
        def __init__(self):
            self.go_calls = 0

        def set_start_state_to_current_state(self):
            events.append("start")

        def set_pose_target(self, pose):
            events.append("pose_target")

        def go(self, wait=True):
            self.go_calls += 1
            events.append(("go", self.go_calls))
            return self.go_calls >= 2

        def stop(self):
            events.append("stop")

        def clear_pose_targets(self):
            events.append("clear")

        def get_current_pose(self):
            return make_pose(9.0, 9.0, 9.0)

    execute_pose.__globals__.update({
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: events.append("warn"),
            sleep=lambda seconds: events.append(("sleep", seconds)),
        ),
    })

    execute_pose(FakeArm(), target_pose, "taught_pre_place")

    assert events.count("start") == 2
    assert ("go", 1) in events
    assert ("go", 2) in events
    assert "warn" in events
    assert ("sleep", 0.3) in events


def test_execute_joint_values_resyncs_and_retries_after_first_go_failure():
    execute_joint_values, = load_module_symbols("execute_joint_values")
    events = []

    class FakeArm:
        def __init__(self):
            self.go_calls = 0

        def set_start_state_to_current_state(self):
            events.append("start")

        def set_joint_value_target(self, values):
            events.append(("joint_target", list(values)))

        def go(self, wait=True):
            self.go_calls += 1
            events.append(("go", self.go_calls))
            return self.go_calls >= 2

        def stop(self):
            events.append("stop")

        def clear_pose_targets(self):
            events.append("clear")

    execute_joint_values.__globals__.update({
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: events.append("warn"),
            sleep=lambda seconds: events.append(("sleep", seconds)),
        ),
    })

    execute_joint_values(FakeArm(), [0, 0.1, 0.2], "idle")

    assert events.count("start") == 2
    assert events.count(("joint_target", [0.0, 0.1, 0.2])) == 2
    assert ("go", 1) in events
    assert ("go", 2) in events
    assert "warn" in events
    assert ("sleep", 0.5) in events


def test_execute_pose_does_not_retry_when_failed_go_already_reached_target():
    execute_pose, = load_module_symbols("execute_pose")
    target_pose = make_pose(0.1, 0.2, 0.3)
    events = []

    class FakeArm:
        def set_start_state_to_current_state(self):
            events.append("start")

        def set_pose_target(self, pose):
            events.append("pose_target")

        def go(self, wait=True):
            events.append("go")
            return False

        def stop(self):
            events.append("stop")

        def clear_pose_targets(self):
            events.append("clear")

        def get_current_pose(self):
            events.append("get_current_pose")
            return make_pose(0.101, 0.201, 0.301)

    execute_pose.__globals__.update({
        "rospy": SimpleNamespace(
            loginfo=lambda *items: events.append("info"),
            logwarn=lambda *items: events.append("warn"),
            sleep=lambda seconds: events.append(("sleep", seconds)),
        ),
    })

    execute_pose(FakeArm(), target_pose, "taught_pre_place")

    assert events.count("go") == 1
    assert "get_current_pose" in events
    assert "warn" in events


def test_run_taught_sequence_turns_pump_off_if_failure_happens_after_pickup():
    run_taught_sequence, = load_module_symbols("run_taught_sequence")
    args = SimpleNamespace(
        sequence=[1],
        preset_file="/tmp/unused.json",
        base_frame="base",
        camera_frame="camera",
        tf_timeout=1.0,
        approach_gap=0.03,
        place_approach_gap=0.02,
        dry_run=False,
        debug_hold_seconds=0.0,
        home_after_idle=False,
        assist_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    preset = make_v3_preset()
    pump_events = []

    def fake_cartesian(arm, pose, label, *args, **kwargs):
        if label == "taught_place":
            raise RuntimeError("place failed")

    run_taught_sequence.__globals__.update({
        "load_preset": lambda path: preset,
        "wait_for_tag_pose_in_base": lambda listener, args, tag_id: make_pose(0.2, 0.0, 0.1),
        "publish_debug_geometry": lambda *items, **kwargs: None,
        "execute_pose": lambda *items: None,
        "execute_cartesian_pose": fake_cartesian,
        "execute_joint_values": lambda *items: None,
        "run_contact_approach": lambda *items: True,
        "set_pump": lambda pump_proxy, enabled: pump_events.append(enabled),
        "tf": SimpleNamespace(TransformListener=lambda: object()),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })

    with pytest.raises(RuntimeError, match="place failed"):
        run_taught_sequence(
            args, object(), object(), contact_proxies=(object(), object()))

    assert pump_events == [False, True, False]


def test_run_taught_sequence_can_run_startup_home_after_idle_when_requested():
    run_taught_sequence, = load_module_symbols("run_taught_sequence")
    args = SimpleNamespace(
        sequence=[1],
        preset_file="/tmp/unused.json",
        base_frame="base",
        camera_frame="camera",
        tf_timeout=1.0,
        approach_gap=0.03,
        place_approach_gap=0.02,
        dry_run=False,
        debug_hold_seconds=0.0,
        home_after_idle=True,
        assist_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    preset = make_v3_preset(idle_joint_values=[0.0, 0.1, 0.2])
    events = []

    run_taught_sequence.__globals__.update({
        "load_preset": lambda path: preset,
        "wait_for_tag_pose_in_base": lambda listener, args, tag_id: make_pose(0.2, 0.0, 0.1),
        "publish_debug_geometry": lambda *items, **kwargs: None,
        "execute_pose": lambda *items: events.append("pose"),
        "execute_cartesian_pose": lambda *items, **kwargs: events.append("cartesian"),
        "execute_joint_values": lambda arm, values, label: events.append(label),
        "run_startup_home": lambda args: events.append("startup_home"),
        "run_contact_approach": lambda *items: True,
        "set_pump": lambda *items: events.append("pump"),
        "tf": SimpleNamespace(TransformListener=lambda: object()),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })

    run_taught_sequence(
        args, object(), object(), contact_proxies=(object(), object()))

    assert "idle" in events
    assert "startup_home" in events
    assert events.index("idle") < events.index("startup_home")


def test_contact_guard_covers_move_to_p_then_sixty_five_mm_probe():
    run_contact_probe, = load_module_symbols("run_contact_approach")
    start = make_pose(0.10, 0.20, 0.30)
    pickup_model = {"approach_axis_xyz_base": [-1.0, 0.0, 0.0]}
    enable_events = []
    targets = []
    cartesian_options = []
    state_results = iter([False, False, True])

    def enable_proxy(enabled):
        enable_events.append(enabled)
        return SimpleNamespace(success=True, message="ok")

    def state_proxy():
        triggered = next(state_results)
        return SimpleNamespace(
            success=triggered,
            message="TRIGGERED" if triggered else "NOT_TRIGGERED")

    run_contact_probe.__globals__.update({
        "execute_cartesian_pose": lambda arm, pose, label, **kwargs: (
            targets.append(copy.deepcopy(pose)),
            cartesian_options.append(dict(kwargs))),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })

    assert run_contact_probe(
        object(), start, pickup_model, "base",
        enable_proxy, state_proxy) is True

    assert enable_events == [True, False]
    assert [pose.pose.position.x for pose in targets] == pytest.approx(
        [0.10, 0.165])
    assert all(pose.pose.position.y == pytest.approx(0.20)
               for pose in targets)
    assert all(pose.pose.position.z == pytest.approx(0.30)
               for pose in targets)
    assert all(options["stop_after"] is False
               for options in cartesian_options)
    assert all(options["settle"] is False
               for options in cartesian_options)
    staging_options = {
        "eef_step": pytest.approx(0.005),
        "quiet": True,
        "settle": False,
        "stop_after": False,
        "min_point_interval": pytest.approx(0.5),
    }
    probe_options = dict(staging_options)
    probe_options["eef_step"] = pytest.approx(0.002)
    assert cartesian_options == [staging_options, probe_options]


def test_contact_probe_stops_at_sixty_five_mm_when_switch_never_triggers():
    run_contact_probe, = load_module_symbols("run_contact_approach")
    start = make_pose(0.10, 0.20, 0.30)
    targets = []

    run_contact_probe.__globals__.update({
        "execute_cartesian_pose": lambda arm, pose, label, **kwargs: (
            targets.append(copy.deepcopy(pose))),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })
    enable_proxy = lambda enabled: SimpleNamespace(success=True, message="ok")
    state_proxy = lambda: SimpleNamespace(
        success=False, message="NOT_TRIGGERED")

    assert run_contact_probe(
        object(), start, {"approach_axis_xyz_base": [-1.0, 0.0, 0.0]},
        "base", enable_proxy, state_proxy) is False

    assert len(targets) == 2
    assert targets[0].pose.position.x == pytest.approx(0.10)
    assert targets[-1].pose.position.x == pytest.approx(0.165)


def test_contact_guard_skips_probe_when_switch_triggers_before_p():
    run_contact_approach, = load_module_symbols("run_contact_approach")
    start = make_pose(0.10, 0.20, 0.30)
    targets = []
    state_results = iter([False, True])
    run_contact_approach.__globals__.update({
        "execute_cartesian_pose": lambda arm, pose, label, **kwargs: (
            targets.append(copy.deepcopy(pose))),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })
    enable_proxy = lambda enabled: SimpleNamespace(success=True, message="ok")

    def state_proxy():
        triggered = next(state_results)
        return SimpleNamespace(
            success=triggered,
            message="TRIGGERED" if triggered else "NOT_TRIGGERED")

    assert run_contact_approach(
        object(), start, {"approach_axis_xyz_base": [-1.0, 0.0, 0.0]},
        "base", enable_proxy, state_proxy) is True
    assert len(targets) == 1
    assert targets[0].pose.position.x == pytest.approx(0.10)


def test_contact_state_serial_error_is_a_hard_failure():
    contact_is_triggered, = load_module_symbols("contact_is_triggered")

    with pytest.raises(RuntimeError, match="读取限位开关失败"):
        contact_is_triggered(lambda: SimpleNamespace(
            success=False, message="ERROR: serial unavailable"))


def test_contact_probe_miss_retreats_keeps_pump_off_and_reports_incomplete():
    run_taught_sequence, contact_error = load_module_symbols(
        "run_taught_sequence", "ContactProbeIncomplete")
    args = SimpleNamespace(
        sequence=[1], preset_file="/tmp/unused.json",
        base_frame="base", camera_frame="camera", tf_timeout=1.0,
        approach_gap=0.04, place_approach_gap=0.02,
        dry_run=False, debug_hold_seconds=0.0, home_after_idle=False,
        assist_orientation_xyzw=[0.0, 0.0, 0.0, 1.0])
    preset = make_v3_preset(idle_joint_values=[0.0, 0.1, 0.2])
    pump_events = []
    cartesian_labels = []
    cartesian_targets = {}
    joint_labels = []

    run_taught_sequence.__globals__.update({
        "load_preset": lambda path: preset,
        "wait_for_tag_pose_in_base": lambda listener, args, tag_id: (
            make_pose(0.2, 0.0, 0.1)),
        "publish_debug_geometry": lambda *items, **kwargs: None,
        "execute_pose": lambda *items: None,
        "run_contact_approach": lambda *items: False,
        "execute_cartesian_pose": lambda arm, pose, label, **kwargs: (
            cartesian_labels.append(label),
            cartesian_targets.update({label: copy.deepcopy(pose)})),
        "execute_joint_values": lambda arm, values, label: (
            joint_labels.append(label)),
        "set_pump": lambda proxy, enabled: pump_events.append(enabled),
        "tf": SimpleNamespace(TransformListener=lambda: object()),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })

    with pytest.raises(contact_error):
        run_taught_sequence(
            args, object(), object(),
            contact_proxies=(object(), object()))

    assert pump_events == [False, False]
    assert cartesian_labels == ["contact_probe_miss_retreat"]
    # Tag x=0.20, shared pre-grasp offset=-0.03, then another 30mm back.
    assert cartesian_targets[
        "contact_probe_miss_retreat"].pose.position.x == pytest.approx(0.14)
    assert joint_labels == ["idle"]


def test_source_contract_removes_old_tuning_modes_and_parameters():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "teach_tag_sequence" in source
    assert "teach_carry" in source
    assert "teach_idle" in source
    assert "run_taught_sequence" in source
    for old_text in [
        "face_pick_place",
        "pick_lift_place",
        "--mode pick_place",
        "teach_step",
        "--teach-step",
        "--tag-id",
        "--grasp-x",
        "--y-offset",
        "--z-offset",
        "--tool-axis",
        "--tag-detection-backend",
        "--place-align-joints",
        "--grasp-align-joints",
        "taught_place_align_joints",
        "taught_grasp_align_joints",
        "build_place_align_joint_values",
        "wait_for_enhanced_tag_pose",
    ]:
        assert old_text not in source


def test_teach_sequence_prompts_are_chinese_and_step_based():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "步骤 1" in source
    assert "步骤 2" in source
    assert "步骤 3" in source
    assert "在 RViz 里微调" in source
    assert "输入 q 再回车退出" in source
