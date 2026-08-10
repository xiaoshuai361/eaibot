import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "src" / "mirobot_delivery.py"


def load_symbols(*names):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.Assign, ast.ClassDef))
    ]
    namespace = {
        "argparse": __import__("argparse"),
        "copy": __import__("copy"),
        "json": json,
        "math": __import__("math"),
        "os": __import__("os"),
        "STRING_TYPES": (str,),
        "arm_api": SimpleNamespace(
            DEFAULT_STARTUP_HOME_SERVICE="/mirobot_startup_home"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"),
         namespace)
    return [namespace[name] for name in names]


def make_item(seed):
    return {
        "cargo_pick_joint_values": [seed + value for value in range(6)],
        "transit_joint_values": [seed + 10 + value for value in range(6)],
        "delivery_joint_values": [seed + 20 + value for value in range(6)],
    }


def make_delivery_preset(seed=0.1, item_ids=(1,)):
    return {
        "version": 2,
        "cargo_pick_joint_values_by_id": {
            str(item_id): [seed + item_id + value for value in range(6)]
            for item_id in item_ids
        },
        "transit_joint_values": [seed + 10 + value for value in range(6)],
        "delivery_joint_values": [seed + 20 + value for value in range(6)],
    }


def write_tag_preset(path, idle=None):
    path.write_text(json.dumps({
        "version": 3,
        "idle_joint_values": idle or [0, 1, 2, 3, 4, 5],
        "tags": {},
    }), encoding="utf-8")


def test_parse_args_defaults_to_three_point_delivery_workflow():
    parse_args, = load_symbols("parse_args")
    parse_args.__globals__["rospy"] = SimpleNamespace(myargv=lambda argv: argv)

    args = parse_args(["prog", "--mode", "run_delivery"])

    assert args.sequence == [1, 2, 3, 4]
    assert args.delivery_file.endswith("/config/delivery_presets.json")
    assert args.tag_preset_file.endswith("/config/tag_pick_place_presets.json")
    assert args.velocity_scale == pytest.approx(0.2)
    assert args.acceleration_scale == pytest.approx(0.2)


def test_home_ready_requires_all_six_joints_near_zero():
    home_joint_state_is_ready, = load_symbols("home_joint_state_is_ready")

    assert home_joint_state_is_ready([0.01, -0.02, 0.0, 0.03, 0.0, -0.01])
    assert not home_joint_state_is_ready(
        [0.01, -0.02, 0.0, 0.09, 0.0, -0.01])
    assert not home_joint_state_is_ready([0.0, 0.0])


def test_vertical_offset_moves_only_base_z_by_five_centimeters():
    build_vertical_offset_pose, = load_symbols("build_vertical_offset_pose")
    current = SimpleNamespace(
        header=SimpleNamespace(stamp=1),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.12, y=-0.08, z=0.06),
            orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9)))
    build_vertical_offset_pose.__globals__["rospy"] = SimpleNamespace(
        Time=SimpleNamespace(now=lambda: 2))

    lifted = build_vertical_offset_pose(current, 0.05)

    assert lifted.pose.position.x == pytest.approx(0.12)
    assert lifted.pose.position.y == pytest.approx(-0.08)
    assert lifted.pose.position.z == pytest.approx(0.11)
    assert lifted.pose.orientation.w == pytest.approx(0.9)
    assert current.pose.position.z == pytest.approx(0.06)


def test_fk_request_uses_taught_joint_values_and_link6():
    fill_fk_request, = load_symbols("fill_fk_request")
    request = SimpleNamespace(
        header=SimpleNamespace(frame_id="", stamp=None),
        fk_link_names=[],
        robot_state=SimpleNamespace(joint_state=SimpleNamespace(
            header=SimpleNamespace(stamp=None), name=[], position=[])))

    fill_fk_request(
        request, "base", ["joint1", "joint2"], [0.1, 0.2],
        "Link6", 123)

    assert request.header.frame_id == "base"
    assert request.fk_link_names == ["Link6"]
    assert request.robot_state.joint_state.name == ["joint1", "joint2"]
    assert request.robot_state.joint_state.position == [0.1, 0.2]


def test_delivery_preset_roundtrip_and_validation(tmp_path):
    (empty_delivery_preset, save_delivery_preset, load_delivery_preset,
     require_delivery_items) = load_symbols(
        "empty_delivery_preset", "save_delivery_preset",
        "load_delivery_preset", "require_delivery_items")
    path = tmp_path / "delivery.json"
    preset = empty_delivery_preset()
    preset.update(make_delivery_preset(0.1, (2, 3)))

    save_delivery_preset(str(path), preset)
    loaded = load_delivery_preset(str(path))
    items = require_delivery_items(loaded, [2, 3])

    assert items["2"]["cargo_pick_joint_values"] == pytest.approx(
        make_delivery_preset(0.1, (2, 3))[
            "cargo_pick_joint_values_by_id"]["2"])
    assert items["2"]["transit_joint_values"] == pytest.approx(
        items["3"]["transit_joint_values"])
    assert items["2"]["delivery_joint_values"] == pytest.approx(
        items["3"]["delivery_joint_values"])
    with pytest.raises(RuntimeError, match="ID4"):
        require_delivery_items(loaded, [4])


def test_three_delivery_points_are_taught_by_separate_modes(tmp_path):
    teach_delivery_point, load_delivery_preset = load_symbols(
        "teach_delivery_point", "load_delivery_preset")
    events = []
    samples = iter([
        [1, 2, 3, 4, 5, 6],
        [11, 12, 13, 14, 15, 16],
        [21, 22, 23, 24, 25, 26],
    ])
    path = tmp_path / "delivery.json"
    arm = SimpleNamespace(get_current_joint_values=lambda: next(samples))
    teach_delivery_point.__globals__.update({
        "prompt_enter": lambda message: events.append(message),
        "rospy": SimpleNamespace(
            sleep=lambda seconds: events.append(("sleep", seconds)),
            loginfo=lambda *items: None),
    })

    for mode in ("teach_cargo_pick", "teach_transit", "teach_release"):
        args = SimpleNamespace(
            mode=mode, delivery_file=str(path), sequence=[4],
            overwrite=False, teach_settle_seconds=0.8)
        teach_delivery_point(args, arm)
    preset = load_delivery_preset(str(path))

    assert len(events) == 6
    assert preset["cargo_pick_joint_values_by_id"]["4"] == pytest.approx(
        [1, 2, 3, 4, 5, 6])
    assert preset["transit_joint_values"] == pytest.approx(
        [11, 12, 13, 14, 15, 16])
    assert preset["delivery_joint_values"] == pytest.approx(
        [21, 22, 23, 24, 25, 26])


def test_run_delivery_orders_home_pump_three_points_and_shared_idle(tmp_path):
    run_delivery, = load_symbols("run_delivery")
    delivery_path = tmp_path / "delivery.json"
    tag_path = tmp_path / "tag.json"
    delivery_path.write_text(
        json.dumps(make_delivery_preset()), encoding="utf-8")
    write_tag_preset(tag_path)
    events = []

    fake_api = SimpleNamespace(
        run_startup_home=lambda args: events.append(("home",)),
        set_pump=lambda proxy, enabled: events.append(("pump", enabled)),
        execute_pose=lambda arm, pose, label: events.append(("pose", label)),
        execute_joint_values=lambda arm, values, label: events.append(
            ("joint", label, list(values))),
        execute_cartesian_pose=lambda arm, pose, label: events.append(
            ("cartesian", label)),
    )
    run_delivery.__globals__.update({
        "arm_api": fake_api,
        "compute_fk_pose": lambda args, arm, joints: events.append(
            ("fk",)) or "cargo_pose",
        "wait_for_home_joint_state": lambda arm: events.append(("home_ready",)),
        "build_vertical_offset_pose": lambda pose, distance: events.append(
            ("build_pre_pick", distance)) or "pre_pick_pose",
        "rospy": SimpleNamespace(
            sleep=lambda seconds: events.append(("sleep", seconds)),
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            logerr=lambda *items: None),
    })
    args = SimpleNamespace(
        delivery_file=str(delivery_path),
        tag_preset_file=str(tag_path), sequence=[1], dry_run=False,
        pump_on_settle_seconds=1.0, pump_off_settle_seconds=0.7)

    run_delivery(args, object(), object())

    assert [event[:2] for event in events] == [
        ("fk",),
        ("build_pre_pick", 0.05),
        ("home",),
        ("home_ready",),
        ("pump", False),
        ("pose", "delivery_1_pre_pick_5cm"),
        ("joint", "delivery_1_cargo_pick"),
        ("pump", True),
        ("sleep", 1.0),
        ("cartesian", "delivery_1_lift_5cm"),
        ("joint", "delivery_1_transit"),
        ("joint", "delivery_1_release"),
        ("pump", False),
        ("sleep", 0.7),
        ("joint", "idle"),
    ]


def test_dry_run_does_not_home_move_or_operate_pump(tmp_path):
    run_delivery, = load_symbols("run_delivery")
    delivery_path = tmp_path / "delivery.json"
    tag_path = tmp_path / "tag.json"
    delivery_path.write_text(
        json.dumps(make_delivery_preset()), encoding="utf-8")
    write_tag_preset(tag_path)
    run_delivery.__globals__.update({
        "arm_api": SimpleNamespace(),
        "rospy": SimpleNamespace(
            loginfo=lambda *items: None,
            logwarn=lambda *items: None),
    })
    args = SimpleNamespace(
        delivery_file=str(delivery_path),
        tag_preset_file=str(tag_path), sequence=[1], dry_run=True)

    run_delivery(args, object(), None)


def test_failure_while_carrying_keeps_pump_enabled(tmp_path):
    run_delivery, = load_symbols("run_delivery")
    delivery_path = tmp_path / "delivery.json"
    tag_path = tmp_path / "tag.json"
    delivery_path.write_text(
        json.dumps(make_delivery_preset()), encoding="utf-8")
    write_tag_preset(tag_path)
    pump_events = []

    def execute(arm, values, label):
        if label.endswith("transit"):
            raise RuntimeError("motion failed")

    run_delivery.__globals__.update({
        "arm_api": SimpleNamespace(
            run_startup_home=lambda args: None,
            set_pump=lambda proxy, enabled: pump_events.append(enabled),
            execute_pose=lambda arm, pose, label: None,
            execute_joint_values=execute,
            execute_cartesian_pose=lambda arm, pose, label: None),
        "compute_fk_pose": lambda args, arm, joints: object(),
        "wait_for_home_joint_state": lambda arm: None,
        "build_vertical_offset_pose": lambda pose, distance: object(),
        "rospy": SimpleNamespace(
            sleep=lambda seconds: None,
            loginfo=lambda *items: None,
            logwarn=lambda *items: None,
            logerr=lambda *items: None),
    })
    args = SimpleNamespace(
        delivery_file=str(delivery_path),
        tag_preset_file=str(tag_path), sequence=[1], dry_run=False,
        pump_on_settle_seconds=0.0, pump_off_settle_seconds=0.0)

    with pytest.raises(RuntimeError, match="motion failed"):
        run_delivery(args, object(), object())

    assert pump_events == [False, True]
