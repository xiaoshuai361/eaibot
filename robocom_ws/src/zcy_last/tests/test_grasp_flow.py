#!/usr/bin/env python3
# coding=utf-8

import json
import sys
import time
import types


class _Vector(object):
    def __init__(self):
        self.x = self.y = self.z = 0.0


class _Twist(object):
    def __init__(self):
        self.linear = _Vector()
        self.angular = _Vector()


def _install_ros_stubs():
    rospy = sys.modules.setdefault("rospy", types.ModuleType("rospy"))
    rospy.is_shutdown = lambda: False
    rospy.get_time = lambda: 10.0
    rospy.loginfo = lambda *args, **kwargs: None
    rospy.logwarn = lambda *args, **kwargs: None
    rospy.logerr = lambda *args, **kwargs: None
    geometry_msgs = sys.modules.setdefault(
        "geometry_msgs", types.ModuleType("geometry_msgs"))
    geometry_msgs_msg = sys.modules.setdefault(
        "geometry_msgs.msg", types.ModuleType("geometry_msgs.msg"))
    geometry_msgs_msg.Twist = _Twist
    geometry_msgs.msg = geometry_msgs_msg


_install_ros_stubs()

from zcy_last.control.grasp import GraspCoordinator  # noqa: E402
from zcy_last.task.competition import LaneFollower  # noqa: E402


class FakeSupervisor(object):
    def __init__(self, result=0):
        self.result = result
        self.calls = []
        self.command = None

    def start_astra(self):
        self.calls.append("start_astra")

    def start_arm_common(self):
        self.calls.append("start_arm_common")

    def stop_astra(self):
        self.calls.append("stop_astra")

    def stop_tag_stack(self):
        self.calls.append("stop_tag_stack")

    def stop_arm_common(self):
        self.calls.append("stop_arm_common")

    def run_job(self, name, command):
        self.calls.append(name)
        self.command = list(command)
        if "--result-file" in command and self.result == 0:
            path = command[command.index("--result-file") + 1]
            count = int(command[command.index("--max-targets") + 1])
            with open(path, "w") as handle:
                json.dump({"completed_ids": list(range(1, count + 1))}, handle)
        return self.result


class FakeCoordinator(object):
    def __init__(self):
        self.calls = []

    def start(self, kind, count):
        self.calls.append((kind, count))

    def start_delivery(self, source, item_ids):
        self.calls.append(("delivery", source, list(item_ids)))

    def completed_items(self):
        return [1, 2]

    def poll(self):
        return None, None


def _wait_result(coordinator):
    deadline = time.time() + 1.0
    while time.time() < deadline:
        result, error = coordinator.poll()
        if result is not None:
            return result, error
        time.sleep(0.01)
    raise AssertionError("coordinator did not finish")


def test_tag_pick_command_uses_count_strict_mode_and_releases_camera():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(
        supervisor, keep_arm_after_tag=True, python3="/env/python3")

    coordinator.start("tag", 2)
    result, error = _wait_result(coordinator)

    assert result is True
    assert error is None
    assert supervisor.command[
        supervisor.command.index("--max-targets") + 1] == "2"
    assert "--fail-on-skip" in supervisor.command
    assert supervisor.calls[-2:] == ["stop_tag_stack", "stop_astra"]
    assert "stop_arm_common" not in supervisor.calls


def test_untagged_pick_failure_stops_camera_and_arm_stack():
    supervisor = FakeSupervisor(result=3)
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")

    coordinator.start("untagged", 1)
    result, error = _wait_result(coordinator)

    assert result is False
    assert error is not None
    assert supervisor.calls[0] == "start_astra"
    assert supervisor.calls[-2:] == ["stop_astra", "stop_arm_common"]


def test_untagged_pick_reports_actual_inventory_and_can_keep_arm():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(
        supervisor, keep_arm_after_untagged=True, python3="/env/python3")

    coordinator.start("untagged", 2)
    result, error = _wait_result(coordinator)

    assert result is True
    assert error is None
    assert coordinator.completed_items() == [1, 2]
    assert "--result-file" in supervisor.command
    assert supervisor.calls[-1] == "stop_astra"
    assert "stop_arm_common" not in supervisor.calls


def test_delivery_command_uses_only_requested_inventory_id():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")

    coordinator.start_delivery("tag", [3])
    result, error = _wait_result(coordinator)

    assert result is True
    assert error is None
    assert supervisor.calls[:2] == ["start_arm_common", "delivery"]
    assert supervisor.command[
        supervisor.command.index("--sequence") + 1] == "3"


def test_untagged_delivery_uses_its_own_motion_presets():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")

    coordinator.start_delivery("untagged", [2])
    result, error = _wait_result(coordinator)

    assert result is True
    assert error is None
    assert supervisor.command[
        supervisor.command.index("--delivery-file") + 1].endswith(
            "/untagged_delivery_presets.json")
    assert supervisor.command[
        supervisor.command.index("--tag-preset-file") + 1].endswith(
            "/block_mono_pick_place_presets.json")


def test_line_publisher_is_suppressed_while_grasp_owns_chassis():
    follower = LaneFollower.__new__(LaneFollower)
    follower.velocity_owner = "grasp"
    follower.state = "B_PICKING"
    follower.maneuver_phase = "NONE"
    follower.turn_angular = 0.5
    follower.last_command_angular = 0.0
    follower.dry_run = False
    published = []
    follower.pub = types.SimpleNamespace(publish=published.append)

    assert follower.publish(0.16, 0.2) is False
    assert published == []
    assert follower.publish(0.0, 0.0, force=True) is True
    assert len(published) == 1


def test_third_intersection_triggers_untagged_pick_before_follow():
    follower = LaneFollower.__new__(LaneFollower)
    follower.task_index = 2
    follower.turn_cmd = "right"
    follower.enable_untagged_pick = True
    follower.untagged_pick_completed = False
    follower.velocity_owner = "line"
    follower.states = []
    follower.publish = lambda *args, **kwargs: True
    follower._shutdown_yolo = lambda: follower.states.append("yolo_stopped")
    follower._set_state = lambda state: follower.states.append(state)
    follower._switch_yolo_profile_if_needed = lambda: follower.states.append(
        "profile_switched")

    follower._complete_intersection()

    assert follower.task_index == 3
    assert follower.turn_cmd == "left"
    assert follower.states == ["yolo_stopped", "A_PICK_PREPARE"]


def test_b_pick_is_started_once_before_following():
    follower = LaneFollower.__new__(LaneFollower)
    follower.state = "B_PICK_PREPARE"
    follower.state_started = 0.0
    follower.tag_pick_count = 2
    follower.grasp_coordinator = FakeCoordinator()
    follower.velocity_owner = "line"
    follower.active_pick_kind = None
    follower.publish = lambda *args, **kwargs: True

    def set_state(state):
        follower.state = state

    follower._set_state = set_state
    follower._handle_pick_without_frame(10.0)
    follower._handle_pick_without_frame(11.0)

    assert follower.state == "B_PICKING"
    assert follower.velocity_owner == "grasp"
    assert follower.grasp_coordinator.calls == [("tag", 2)]


def test_completed_untagged_pick_is_not_triggered_twice():
    follower = LaneFollower.__new__(LaneFollower)
    follower.task_index = 2
    follower.turn_cmd = "right"
    follower.enable_untagged_pick = True
    follower.untagged_pick_completed = True
    follower.states = []
    follower._set_state = follower.states.append
    follower._switch_yolo_profile_if_needed = lambda: follower.states.append(
        "profile_switched")

    follower._complete_intersection()

    assert follower.task_index == 3
    assert follower.states == ["profile_switched", "FOLLOW"]


def test_completed_untagged_pick_records_actual_inventory():
    follower = LaneFollower.__new__(LaneFollower)
    follower.grasp_coordinator = FakeCoordinator()
    follower.grasp_coordinator.completed_items = lambda: [1, 2]
    follower.untagged_pick_count = 2
    follower.untagged_inventory = []
    follower.untagged_pick_completed = False
    follower.velocity_owner = "grasp"
    follower.bridge = types.SimpleNamespace(reset=lambda _width: None)
    follower.lane_width = 620.0
    follower.stop_hits = 3
    follower.states = []
    follower.publish = lambda *args, **kwargs: True
    follower._resume_yolo = lambda profile: profile == "building"
    follower._set_state = follower.states.append

    follower._finish_pick("untagged")

    assert follower.untagged_inventory == [1, 2]
    assert follower.untagged_pick_completed is True
    assert follower.velocity_owner == "line"
    assert follower.states == ["PICK_RECOVER"]


def test_street_event_delivers_only_matching_tag_inventory():
    follower = LaneFollower.__new__(LaneFollower)
    follower.enable_tag_delivery = True
    follower.enable_untagged_delivery = False
    follower.tag_inventory = [1, 3]
    follower.untagged_inventory = []
    follower.tag_delivery_failed_ids = set()
    follower.untagged_delivery_failed_ids = set()
    follower.grasp_coordinator = FakeCoordinator()
    follower.velocity_owner = "line"
    follower.active_delivery_source = None
    follower.active_delivery_id = None
    follower.states = []
    follower.publish = lambda *args, **kwargs: True
    follower._set_state = follower.states.append
    event = types.SimpleNamespace(
        kind="street", area="C区", class_name="Recyclable waste",
        display_name="可回收垃圾")

    assert follower._start_delivery_for_event(event) is True
    assert follower.active_delivery_id == 3
    assert follower.velocity_owner == "grasp"
    assert follower.grasp_coordinator.calls == [("delivery", "tag", [3])]
    assert follower.states == ["DELIVERING"]


def test_building_event_delivers_only_matching_untagged_inventory():
    follower = LaneFollower.__new__(LaneFollower)
    follower.enable_tag_delivery = False
    follower.enable_untagged_delivery = True
    follower.tag_inventory = []
    follower.untagged_inventory = [1, 4]
    follower.tag_delivery_failed_ids = set()
    follower.untagged_delivery_failed_ids = set()
    follower.grasp_coordinator = FakeCoordinator()
    follower.velocity_owner = "line"
    follower.active_delivery_source = None
    follower.active_delivery_id = None
    follower.states = []
    follower.publish = lambda *args, **kwargs: True
    follower._set_state = follower.states.append
    event = types.SimpleNamespace(
        kind="building", area="楼宇A", class_name="Collapsed Building",
        display_name="坍塌楼宇")

    assert follower._start_delivery_for_event(event) is True
    assert follower.active_delivery_source == "untagged"
    assert follower.active_delivery_id == 4
    assert follower.grasp_coordinator.calls == [
        ("delivery", "untagged", [4])]
    assert follower.states == ["DELIVERING"]


def test_delivery_failure_warns_and_resumes_follow_without_retry():
    follower = LaneFollower.__new__(LaneFollower)
    follower.active_delivery_source = "tag"
    follower.active_delivery_id = 2
    follower.tag_inventory = [2]
    follower.untagged_inventory = []
    follower.tag_delivery_failed_ids = set()
    follower.untagged_delivery_failed_ids = set()
    follower.enable_untagged_pick = False
    follower.process_supervisor = FakeSupervisor()
    follower.velocity_owner = "grasp"
    follower.states = []
    follower.publish = lambda *args, **kwargs: True
    follower._set_state = follower.states.append

    follower._finish_delivery(False, RuntimeError("motion failed"))

    assert follower.tag_inventory == [2]
    assert follower.tag_delivery_failed_ids == {2}
    assert follower.velocity_owner == "line"
    assert follower.states == ["FOLLOW"]
