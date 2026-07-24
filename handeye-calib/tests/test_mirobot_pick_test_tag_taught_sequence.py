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


def test_compute_grasp_transform_stores_end_effector_relative_to_tag():
    compute_grasp_ee_in_tag, compute_grasp_pose = load_module_symbols(
        "compute_grasp_ee_in_tag",
        "compute_grasp_pose",
    )
    tag_at_teach = make_pose(1.0, 2.0, 0.5)
    taught_ee = make_pose(1.1, 2.2, 0.8)

    grasp_ee_in_tag = compute_grasp_ee_in_tag(tag_at_teach, taught_ee)
    moved_tag = make_pose(0.2, -0.1, 0.4)
    replay_grasp = compute_grasp_pose(moved_tag, grasp_ee_in_tag, "base")

    assert grasp_ee_in_tag["position"] == pytest.approx([0.1, 0.2, 0.3])
    assert replay_grasp.pose.position.x == pytest.approx(0.3)
    assert replay_grasp.pose.position.y == pytest.approx(0.1)
    assert replay_grasp.pose.position.z == pytest.approx(0.7)


def test_position_stable_grasp_replay_ignores_current_tag_rotation_noise():
    (
        record_tag_grasp_in_preset,
        compute_grasp_pose_from_entry,
        build_pre_grasp_pose_from_entry,
    ) = load_module_symbols(
        "record_tag_grasp_in_preset",
        "compute_grasp_pose_from_entry",
        "build_pre_grasp_pose_from_entry",
    )
    root_half = 2 ** 0.5 / 2.0
    taught_tag = make_pose(1.0, 2.0, 0.5)
    taught_grasp = make_pose(
        1.10,
        2.20,
        0.80,
        q=[0.1, 0.2, 0.3, 0.9],
    )
    preset = {}

    record_tag_grasp_in_preset(preset, 4, taught_tag, taught_grasp)
    entry = preset["tags"]["4"]
    moved_tag_with_bad_rotation = make_pose(
        0.2,
        -0.1,
        0.4,
        q=[0.0, 0.0, root_half, root_half],
    )

    replay_grasp = compute_grasp_pose_from_entry(
        moved_tag_with_bad_rotation, entry, "base")
    pre_grasp = build_pre_grasp_pose_from_entry(
        moved_tag_with_bad_rotation, replay_grasp, entry, 0.03, "base")

    assert entry["grasp_position_offset_in_base"] == pytest.approx([0.1, 0.2, 0.3])
    assert entry["grasp_orientation_in_base"] == pytest.approx(
        [0.10259783520851541, 0.20519567041703082,
         0.3077935056255462, 0.9233805168766387])
    assert replay_grasp.pose.position.x == pytest.approx(0.3)
    assert replay_grasp.pose.position.y == pytest.approx(0.1)
    assert replay_grasp.pose.position.z == pytest.approx(0.7)
    assert replay_grasp.pose.orientation.z == pytest.approx(0.3077935056255462)
    assert pre_grasp.pose.position.x == pytest.approx(0.3)
    assert pre_grasp.pose.position.y == pytest.approx(0.1)
    assert pre_grasp.pose.position.z == pytest.approx(0.73)


def test_record_grasp_stores_taught_joint_values_when_available():
    record_tag_grasp_in_preset, = load_module_symbols("record_tag_grasp_in_preset")
    preset = {}

    record_tag_grasp_in_preset(
        preset,
        4,
        make_pose(1.0, 2.0, 0.5),
        make_pose(1.1, 2.2, 0.8),
        grasp_joint_values=[0.0, 0.1, 0.2, 0.3, 0.4, 1.5],
    )

    assert preset["tags"]["4"]["grasp_joint_values"] == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4, 1.5])


def test_pregrasp_moves_along_tag_plus_z_and_preplace_moves_along_base_z():
    build_pre_grasp_pose, build_pre_place_pose = load_module_symbols(
        "build_pre_grasp_pose",
        "build_pre_place_pose",
    )
    tag_pose = make_pose(0.0, 0.0, 0.0)
    grasp_pose = make_pose(0.3, 0.1, 0.2)
    place_pose = make_pose(0.4, -0.2, 0.1)

    pre_grasp = build_pre_grasp_pose(tag_pose, grasp_pose, 0.03, "base")
    pre_place = build_pre_place_pose(place_pose, 0.02, "base")

    assert pre_grasp.pose.position.x == pytest.approx(0.3)
    assert pre_grasp.pose.position.y == pytest.approx(0.1)
    assert pre_grasp.pose.position.z == pytest.approx(0.23)
    assert pre_place.pose.position.x == pytest.approx(0.4)
    assert pre_place.pose.position.y == pytest.approx(-0.2)
    assert pre_place.pose.position.z == pytest.approx(0.12)


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
    preset = {
        "version": 1,
        "base_frame": "base",
        "camera_frame": "camera",
        "tags": {
            "1": {
                "grasp_ee_in_tag": {
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


def test_joint_alignment_uses_current_joints_but_taught_selected_joints():
    build_joint_align_values, = load_module_symbols("build_joint_align_values")

    align_values = build_joint_align_values(
        current_joint_values=[1.0, 1.1, 1.2, 1.3, 1.4, -2.0],
        taught_joint_values=[0.0, 0.1, 0.2, 0.3, 0.4, 1.5],
        align_joints=[5, 6],
        option='--grasp-align-joints',
    )

    assert align_values == pytest.approx([1.0, 1.1, 1.2, 1.3, 0.4, 1.5])

    assert build_joint_align_values(
        current_joint_values=[1.0, 1.1, 1.2, 1.3, 1.4, 1.51],
        taught_joint_values=[0.0, 0.1, 0.2, 0.3, 0.4, 1.5],
        align_joints=[5, 6],
        option='--grasp-align-joints',
    ) == pytest.approx([1.0, 1.1, 1.2, 1.3, 0.4, 1.51])

    assert build_joint_align_values(
        current_joint_values=[1.0, 1.1, 1.2, 1.3, 0.41, 1.51],
        taught_joint_values=[0.0, 0.1, 0.2, 0.3, 0.4, 1.5],
        align_joints=[5, 6],
        option='--grasp-align-joints',
    ) is None

    assert build_joint_align_values(
        current_joint_values=[1.0, 1.1, 1.2, 1.3, 1.4, -2.0],
        taught_joint_values=[0.0, 0.1, 0.2, 0.3, 0.4, 1.5],
        align_joints=[],
        option='--grasp-align-joints',
    ) is None

    with pytest.raises(RuntimeError, match="joint value length"):
        build_joint_align_values(
            [0.0, 0.1], [0.0], align_joints=[5],
            option='--grasp-align-joints')

    with pytest.raises(RuntimeError, match="--grasp-align-joints"):
        build_joint_align_values(
            [0.0] * 6, [0.0] * 6, align_joints=[7],
            option='--grasp-align-joints')


def test_load_preset_reports_missing_corrupt_and_missing_tag(tmp_path):
    load_preset, require_preset_tags = load_module_symbols(
        "load_preset",
        "require_preset_tags",
    )

    with pytest.raises(RuntimeError, match="does not exist"):
        load_preset(str(tmp_path / "missing.json"))

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Could not parse"):
        load_preset(str(corrupt))

    with pytest.raises(RuntimeError, match="tag 2"):
        require_preset_tags({"tags": {"1": {}}}, [1, 2])


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
    with pytest.raises(UserAbort, match="Interrupted"):
        prompt_enter("shutdown")


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
    preset = {
        "idle_joint_values": [0.0, 0.1, 0.2],
        "tags": {
            "1": {
                "grasp_ee_in_tag": {
                    "position": [0.1, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "place_ee_in_base": {
                    "position": [0.4, 0.0, 0.1],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
    }
    events = []

    run_taught_sequence.__globals__.update({
        "load_preset": lambda path: preset,
        "wait_for_tag_pose_in_base": lambda listener, args, tag_id: make_pose(0.2, 0.0, 0.1),
        "publish_debug_geometry": lambda *items, **kwargs: events.append("debug"),
        "execute_pose": lambda *items: events.append("execute_pose"),
        "execute_cartesian_pose": lambda *items, **kwargs: events.append("cartesian"),
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
    preset = {
        "idle_joint_values": [0.0, 0.1, 0.2],
        "tags": {
            "1": {
                "grasp_ee_in_tag": {
                    "position": [0.1, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "place_ee_in_base": {
                    "position": [0.4, 0.0, 0.1],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "2": {
                "grasp_ee_in_tag": {
                    "position": [0.1, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "place_ee_in_base": {
                    "position": [0.4, 0.0, 0.1],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
    }
    events = []

    run_taught_sequence.__globals__.update({
        "load_preset": lambda path: preset,
        "wait_for_tag_pose_in_base": lambda listener, args, tag_id: (
            events.append(("wait_tag", tag_id)) or make_pose(0.2, 0.0, 0.1)
        ),
        "publish_debug_geometry": lambda *items, **kwargs: events.append("debug"),
        "execute_pose": lambda *items: events.append("execute_pose"),
        "execute_cartesian_pose": lambda *items, **kwargs: events.append("cartesian"),
        "execute_joint_values": lambda arm, values, label: events.append(("idle", values, label)),
        "set_pump": lambda *items: events.append("pump"),
        "tf": SimpleNamespace(TransformListener=lambda: object()),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })

    run_taught_sequence(args, object(), object())

    idle_event = ("idle", [0.0, 0.1, 0.2], "idle")
    assert events.count(idle_event) == 2
    assert events.index(idle_event) < events.index(("wait_tag", 2))


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
    preset = {
        "tags": {
            "1": {
                "grasp_ee_in_tag": {
                    "position": [0.1, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "place_ee_in_base": {
                    "position": [0.4, 0.0, 0.1],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "place_joint_values": [0.0, 0.1, 0.2, 0.3, 0.4, 1.5],
            },
        },
    }
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
        "set_pump": lambda *items: events.append("pump"),
        "tf": SimpleNamespace(TransformListener=lambda: object()),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })

    run_taught_sequence(args, FakeArm(), object())

    assert not any(
        isinstance(event, tuple) and event[0] == "taught_place_align_joints"
        for event in events)
    assert events.index("taught_pre_place") < events.index("taught_place")


def test_run_taught_sequence_aligns_taught_grasp_joints_before_pre_grasp_pose():
    run_taught_sequence, = load_module_symbols("run_taught_sequence")
    args = SimpleNamespace(
        sequence=[1],
        preset_file="/tmp/unused.json",
        base_frame="base",
        camera_frame="camera",
        tf_timeout=1.0,
        approach_gap=0.03,
        place_approach_gap=0.02,
        grasp_align_joints=[6],
        dry_run=False,
        debug_hold_seconds=0.0,
        home_after_idle=False,
        assist_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    preset = {
        "tags": {
            "1": {
                "grasp_ee_in_tag": {
                    "position": [0.1, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "grasp_joint_values": [0.0, 0.1, 0.2, 0.3, 0.4, 1.5],
                "place_ee_in_base": {
                    "position": [0.4, 0.0, 0.1],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
    }
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
        "set_pump": lambda *items: events.append("pump"),
        "tf": SimpleNamespace(TransformListener=lambda: object()),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })

    run_taught_sequence(args, FakeArm(), object())

    align_event = (
        "taught_grasp_align_joints",
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    )
    assert align_event in events
    assert events.index(align_event) < events.index("taught_pre_grasp")


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
    preset = {
        "tags": {
            "1": {
                "grasp_ee_in_tag": {
                    "position": [0.1, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "place_ee_in_base": {
                    "position": [0.4, 0.0, 0.1],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
    }
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
        "set_pump": lambda pump_proxy, enabled: pump_events.append(enabled),
        "tf": SimpleNamespace(TransformListener=lambda: object()),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })

    with pytest.raises(RuntimeError, match="place failed"):
        run_taught_sequence(args, object(), object())

    assert pump_events == [True, False]


def test_run_taught_sequence_can_home_after_idle_when_requested():
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
    preset = {
        "idle_joint_values": [0.0, 0.1, 0.2],
        "tags": {
            "1": {
                "grasp_ee_in_tag": {
                    "position": [0.1, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "place_ee_in_base": {
                    "position": [0.4, 0.0, 0.1],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
    }
    events = []

    run_taught_sequence.__globals__.update({
        "load_preset": lambda path: preset,
        "wait_for_tag_pose_in_base": lambda listener, args, tag_id: make_pose(0.2, 0.0, 0.1),
        "publish_debug_geometry": lambda *items, **kwargs: None,
        "execute_pose": lambda *items: events.append("pose"),
        "execute_cartesian_pose": lambda *items, **kwargs: events.append("cartesian"),
        "execute_joint_values": lambda arm, values, label: events.append(label),
        "execute_named_target": lambda arm, name, label: events.append((label, name)),
        "set_pump": lambda *items: events.append("pump"),
        "tf": SimpleNamespace(TransformListener=lambda: object()),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            sleep=lambda seconds: None,
        ),
    })

    run_taught_sequence(args, object(), object())

    assert "idle" in events
    assert ("home", "home") in events
    assert events.index("idle") < events.index(("home", "home"))


def test_source_contract_removes_old_tuning_modes_and_parameters():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "teach_tag_sequence" in source
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
        "taught_place_align_joints",
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
