import ast
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
        "copy": __import__("copy"),
        "json": json,
        "math": __import__("math"),
        "os": __import__("os"),
        "PoseStamped": PoseStamped,
        "STRING_TYPES": (str,),
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


def test_load_or_create_preset_preserves_existing_tags(tmp_path):
    load_or_create_preset, save_preset, load_preset = load_module_symbols(
        "load_or_create_preset",
        "save_preset",
        "load_preset",
    )
    path = tmp_path / "tag_pick_place_presets.json"
    path.write_text(json.dumps({
        "version": 1,
        "base_frame": "base",
        "camera_frame": "camera_rgb_optical_frame",
        "tags": {
            "2": {
                "grasp_ee_in_tag": {
                    "position": [0.2, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "place_ee_in_base": {
                    "position": [0.4, 0.0, 0.1],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        },
    }), encoding="utf-8")

    args = SimpleNamespace(
        preset_file=str(path),
        base_frame="base",
        camera_frame="camera_rgb_optical_frame",
    )
    preset, existed = load_or_create_preset(args)
    preset["tags"]["1"] = {"new": True}
    save_preset(str(path), preset, overwrite=existed)

    reloaded = load_preset(str(path))
    assert reloaded["version"] == 3
    assert "1" in reloaded["tags"]
    assert "2" in reloaded["tags"]
    assert "place_ee_in_base" in reloaded["tags"]["2"]
    assert "grasp_ee_in_tag" not in reloaded["tags"]["2"]


def test_record_idle_joint_values_stores_float_list():
    make_empty_preset, record_idle_in_preset = load_module_symbols(
        "make_empty_preset",
        "record_idle_in_preset",
    )
    arm = SimpleNamespace(get_current_joint_values=lambda: [0, 1.5, -2])
    preset = make_empty_preset("base", "camera_rgb_optical_frame")

    record_idle_in_preset(preset, arm)

    assert preset["idle_joint_values"] == pytest.approx([0.0, 1.5, -2.0])


def test_record_tag_grasp_preserves_existing_place_point():
    record_tag_grasp_in_preset, = load_module_symbols("record_tag_grasp_in_preset")
    preset = {
        "tags": {
            "1": {
                "place_ee_in_base": {
                    "position": [0.4, 0.0, 0.1],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
    }

    record_tag_grasp_in_preset(
        preset,
        1,
        make_pose(1.0, 2.0, 0.5),
        make_pose(1.1, 2.2, 0.8),
    )

    assert preset["tags"]["1"]["place_ee_in_base"]["position"] == pytest.approx([0.4, 0.0, 0.1])
    assert preset["tags"]["1"]["grasp_offset_xyz_base"] == pytest.approx(
        [0.1, 0.2, 0.3])
    assert preset["pickup_model"]["approach_axis_xyz_base"] == pytest.approx(
        [-1.0, 0.0, 0.0])


def test_record_tag_place_preserves_existing_grasp_point():
    record_tag_place_in_preset, = load_module_symbols("record_tag_place_in_preset")
    preset = {
        "tags": {
            "1": {
                "grasp_offset_xyz_base": [0.1, 0.2, 0.3],
            },
        },
    }

    record_tag_place_in_preset(preset, 1, make_pose(0.4, -0.1, 0.2))

    assert preset["tags"]["1"]["grasp_offset_xyz_base"] == pytest.approx(
        [0.1, 0.2, 0.3])
    assert preset["tags"]["1"]["place_ee_in_base"]["position"] == pytest.approx([0.4, -0.1, 0.2])


def test_prompt_and_record_grasp_waits_for_teach_pose_to_settle_before_sampling():
    prompt_and_record_grasp, = load_module_symbols("prompt_and_record_grasp")
    events = []
    args = SimpleNamespace(
        teach_settle_seconds=0.8,
        pickup_approach_axis_base=[-1.0, 0.0, 0.0])
    preset = {"tags": {"1": {"place_ee_in_base": {"position": [0, 0, 0]}}}}

    class FakeArm:
        def get_current_pose(self):
            events.append("get_pose")
            return make_pose(1.1, 2.2, 0.8)

    prompt_and_record_grasp.__globals__.update({
        "prompt_enter": lambda message: events.append("prompt"),
        "rospy": SimpleNamespace(
            sleep=lambda seconds: events.append(("sleep", seconds)),
            loginfo=lambda *items: None,
        ),
    })

    prompt_and_record_grasp(args, FakeArm(), preset, 1, make_pose(1.0, 2.0, 0.5))

    assert events[:3] == ["prompt", ("sleep", 0.8), "get_pose"]


def test_prompt_and_record_place_waits_for_teach_pose_to_settle_before_sampling():
    prompt_and_record_place, = load_module_symbols("prompt_and_record_place")
    events = []
    args = SimpleNamespace(teach_settle_seconds=0.8)
    preset = {"tags": {"1": {"grasp_offset_xyz_base": [0, 0, 0]}}}

    class FakeArm:
        def get_current_pose(self):
            events.append("get_pose")
            return make_pose(0.4, -0.1, 0.2)

        def get_current_joint_values(self):
            return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    prompt_and_record_place.__globals__.update({
        "prompt_enter": lambda message: events.append("prompt"),
        "rospy": SimpleNamespace(
            sleep=lambda seconds: events.append(("sleep", seconds)),
            loginfo=lambda *items: None,
        ),
    })

    prompt_and_record_place(args, FakeArm(), preset, 1)

    assert events[:3] == ["prompt", ("sleep", 0.8), "get_pose"]
