#!/usr/bin/env python3
# coding=utf-8

import json
import os
import sys
import threading
import time
import types

import pytest


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
from zcy_last.task.competition import (  # noqa: E402
    LaneFollower,
    initial_competition_position,
)


class FakeSupervisor(object):
    def __init__(self, result=0):
        self.result = result
        self.calls = []
        self.command = None

    def start_astra(self):
        self.calls.append("start_astra")

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
        self.search_ready = False
        self.search_enabled = False
        self.search_triggered = False
        self.search_released = False

    def start(self, kind, count):
        self.calls.append((kind, count))

    def start_delivery(self, source, item_ids, distance_offset_m=0.0):
        self.calls.append((
            "delivery", source, list(item_ids), float(distance_offset_m)))

    def start_untagged_search(self, count):
        self.calls.append(("untagged_search", count))

    def untagged_search_ready(self):
        return self.search_ready

    def untagged_search_triggered(self):
        return self.search_triggered

    def enable_untagged_search(self):
        self.search_enabled = True
        self.calls.append(("untagged_search_enable",))

    def release_untagged_search(self):
        self.search_released = True
        self.calls.append(("untagged_search_release",))

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


def test_untagged_aligned_debug_entry_starts_at_fourth_left_turn():
    assert initial_competition_position(False, True) == (
        3, "left", "A_PICK_PREPARE")
    assert initial_competition_position(False, False) == (
        0, "right", "FOLLOW")
    assert initial_competition_position(True, False) == (
        0, "right", "B_PICK_PREPARE")


def test_tag_pick_command_uses_partial_mode_and_releases_camera():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(
        supervisor, keep_arm_after_tag=True, python3="/env/python3")

    coordinator.start("tag", 2)
    result, error = _wait_result(coordinator)

    assert result is True
    assert error is None
    assert supervisor.command[
        supervisor.command.index("--max-targets") + 1] == "2"
    assert supervisor.command[
        supervisor.command.index("--tag-tf-wait-seconds") + 1] == "18.0"
    assert "--allow-partial" in supervisor.command
    assert "--fail-on-skip" not in supervisor.command
    assert supervisor.command[
        supervisor.command.index("--pick-approach-gap") + 1] == "0.030"
    assert "--show-debug-window" in supervisor.command
    assert supervisor.calls[-2:] == ["stop_tag_stack", "stop_astra"]
    assert "stop_arm_common" not in supervisor.calls


def test_untagged_pick_command_enables_detection_window():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")

    command = coordinator._untagged_command(1)

    assert "--show-rgb" in command
    assert command[command.index("--sequence") + 1] == "2,3"


def test_untagged_pick_rejects_more_than_two_targets():
    coordinator = GraspCoordinator(FakeSupervisor(), python3="/env/python3")

    with pytest.raises(ValueError, match="1 到 2"):
        coordinator.start("untagged", 3)


def test_untagged_search_command_uses_full_frame_and_handshake_files():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")

    command = coordinator._untagged_command(2, search_before_pick=True)

    assert command[command.index("--sequence") + 1] == "2,3"
    assert command[command.index("--max-targets") + 1] == "2"
    assert "--search-before-chassis" in command
    assert "--show-rgb" in command
    assert "--allow-partial" in command
    assert "--fail-on-skip" not in command
    assert "--search-roi-ratio" not in command
    assert command[command.index("--search-stable-frames") + 1] == "3"
    assert command[command.index("--search-ready-file") + 1] == \
        coordinator.untagged_search_ready_file
    assert command[command.index("--search-enable-file") + 1] == \
        coordinator.untagged_search_enable_file
    assert command[command.index("--search-trigger-file") + 1] == \
        coordinator.untagged_search_trigger_file
    assert command[command.index("--search-release-file") + 1] == \
        coordinator.untagged_search_release_file


def test_untagged_search_clears_stale_handshake_before_background_start():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")
    stale_paths = (
        coordinator.untagged_search_ready_file,
        coordinator.untagged_search_enable_file,
        coordinator.untagged_search_trigger_file,
        coordinator.untagged_search_release_file,
    )
    for path in stale_paths:
        with open(path, "w") as handle:
            handle.write("stale\n")

    coordinator.start_untagged_search(2)

    assert all(not os.path.exists(path) for path in stale_paths)
    _wait_result(coordinator)


def test_delivery_directly_runs_preset_without_dependency_recheck():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(
        supervisor, keep_arm_after_tag=True, python3="/env/python3")

    coordinator.start_delivery("tag", [2])
    result, error = _wait_result(coordinator)

    assert result is True
    assert error is None
    assert supervisor.calls == ["delivery"]
    assert supervisor.command[:2] == ["/usr/bin/python2", "-u"]
    assert supervisor.command[2].endswith("/src/mirobot_delivery.py")
    assert supervisor.command[
        supervisor.command.index("--sequence") + 1] == "2"
    assert "--release-ready-file" in supervisor.command
    assert supervisor.command[
        supervisor.command.index("--pump-off-settle-seconds") + 1] == "0.0"


@pytest.mark.parametrize("source", ["tag", "untagged"])
def test_delivery_reports_success_after_release_delay_while_arm_returns_idle(
        tmp_path, source):
    class BlockingSupervisor(FakeSupervisor):
        def __init__(self):
            super(BlockingSupervisor, self).__init__()
            self.log_dir = str(tmp_path)
            self.started = threading.Event()
            self.finish = threading.Event()

        def run_job(self, name, command):
            self.calls.append(name)
            self.command = list(command)
            self.started.set()
            assert self.finish.wait(1.0)
            return 0

    supervisor = BlockingSupervisor()
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")
    coordinator.start_delivery(source, [2])
    assert supervisor.started.wait(1.0)
    marker = supervisor.command[
        supervisor.command.index("--release-ready-file") + 1]

    with open(marker, "w") as handle:
        handle.write("ID2 pump_off\n")

    result, error = coordinator.poll()
    assert result is True
    assert error is None
    assert coordinator.completed_items() == [2]
    assert coordinator.arm_job_active() is True

    supervisor.finish.set()
    coordinator.join(1.0)
    assert coordinator.arm_job_active() is False


def test_tag_pick_can_request_all_four_targets_in_left_to_right_order():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(
        supervisor, keep_arm_after_tag=True, python3="/env/python3")

    coordinator.start("tag", 4)
    result, error = _wait_result(coordinator)

    assert result is True
    assert error is None
    assert coordinator.completed_items() == [1, 2, 3, 4]
    assert supervisor.command[
        supervisor.command.index("--max-targets") + 1] == "4"
    assert supervisor.command[
        supervisor.command.index("--order") + 1] == "left_to_right"
    assert "--skip-startup-home" not in supervisor.command


def test_tag_pick_accepts_partial_inventory():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(
        supervisor, keep_arm_after_tag=True, python3="/env/python3")
    with open(coordinator.tag_result_file, "w") as handle:
        json.dump({"completed_ids": [2, 4]}, handle)

    assert coordinator._read_pick_result(
        coordinator.tag_result_file, 4, "有 Tag", allow_partial=True
    ) == [2, 4]


def test_tag_pick_accepts_empty_inventory_in_partial_mode():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(
        supervisor, keep_arm_after_tag=True, python3="/env/python3")
    with open(coordinator.tag_result_file, "w") as handle:
        json.dump({"completed_ids": []}, handle)

    assert coordinator._read_pick_result(
        coordinator.tag_result_file, 4, "有 Tag", allow_partial=True
    ) == []


def test_untagged_pick_failure_stops_camera_and_arm_stack():
    supervisor = FakeSupervisor(result=3)
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")

    coordinator.start("untagged", 1)
    result, error = _wait_result(coordinator)

    assert result is False
    assert error is not None
    assert supervisor.calls[0] == "start_astra"
    assert supervisor.calls[-2:] == ["stop_astra", "stop_arm_common"]


def test_astra_cleanup_failure_is_reported_without_crashing_worker_thread():
    class CleanupFailSupervisor(FakeSupervisor):
        def stop_astra(self):
            raise RuntimeError("camera cleanup failed")

    supervisor = CleanupFailSupervisor()
    coordinator = GraspCoordinator(
        supervisor, keep_arm_after_untagged=True, python3="/env/python3")

    coordinator.start("untagged", 1)
    result, error = _wait_result(coordinator)

    assert result is False
    assert "camera cleanup failed" in str(error)


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
    assert "--skip-startup-home" not in supervisor.command
    assert supervisor.calls[0] == "start_astra"
    assert supervisor.calls[-1] == "stop_astra"
    assert "stop_arm_common" not in supervisor.calls


def test_untagged_pick_accepts_partial_inventory():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")
    with open(coordinator.untagged_result_file, "w") as handle:
        json.dump({"completed_ids": [1, 3]}, handle)

    assert coordinator._read_pick_result(
        coordinator.untagged_result_file, 4, "无 Tag", allow_partial=True
    ) == [1, 3]


def test_untagged_child_error_reports_real_runtime_error(tmp_path):
    supervisor = FakeSupervisor(result=1)
    supervisor.log_dir = str(tmp_path)
    log_path = tmp_path / "pick_untagged.log"
    log_path.write_text(
        "Traceback (most recent call last):\n"
        "RuntimeError: Target 2=fire chassis alignment timed out.\n"
        "Error: Arm child exited with status 1\n",
        encoding="utf-8",
    )
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")

    error = coordinator._job_failure(
        "pick_untagged", 1, ["python3", "block_pick_main.py"])

    assert "退出码1" in str(error)
    assert "Target 2=fire chassis alignment timed out" in str(error)
    assert str(log_path) in str(error)


def test_delivery_timeout_reports_last_motion_stage(tmp_path):
    supervisor = FakeSupervisor(result=124)
    supervisor.log_dir = str(tmp_path)
    log_path = tmp_path / "delivery.log"
    log_path.write_text(
        "DELIVERY_STATUS ID4 前往固定投递位姿\n"
        "DELIVERY_TIMEOUT 阶段 35.0 秒无进展，最后阶段："
        "ID4 前往固定投递位姿\n",
        encoding="utf-8",
    )
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")

    error = coordinator._job_failure(
        "delivery", 124, ["python2", "mirobot_delivery.py"])

    assert "投递子进程退出码124" in str(error)
    assert "最后阶段：ID4 前往固定投递位姿" in str(error)


def test_delivery_command_uses_only_requested_inventory_id():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")

    coordinator.start_delivery("tag", [3])
    result, error = _wait_result(coordinator)

    assert result is True
    assert error is None
    assert supervisor.calls == ["delivery"]
    assert supervisor.command[
        supervisor.command.index("--sequence") + 1] == "3"


def test_untagged_delivery_uses_its_own_motion_presets():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")

    coordinator.start_delivery("untagged", [2], 0.1)
    result, error = _wait_result(coordinator)

    assert result is True
    assert error is None
    assert supervisor.command[
        supervisor.command.index("--delivery-file") + 1].endswith(
            "/untagged_delivery_presets.json")
    assert supervisor.command[
        supervisor.command.index("--cargo-pick-file") + 1].endswith(
            "/delivery_presets.json")
    assert supervisor.command[
        supervisor.command.index("--tag-preset-file") + 1].endswith(
            "/block_mono_pick_place_presets.json")
    assert "--release-ready-file" in supervisor.command
    assert supervisor.command[
        supervisor.command.index("--pump-off-settle-seconds") + 1] == "0.7"
    assert supervisor.command[
        supervisor.command.index("--release-ready-delay-seconds") + 1] == "3.0"
    assert "--contact-release" in supervisor.command
    assert "--force-release-on-contact-miss" in supervisor.command
    assert supervisor.command[
        supervisor.command.index("--contact-staging-gap") + 1] == "0.030"
    assert supervisor.command[
        supervisor.command.index("--contact-distance-offset") + 1] == "0.1"


def test_both_delivery_sources_share_tag_cargo_pick_points():
    supervisor = FakeSupervisor()
    coordinator = GraspCoordinator(supervisor, python3="/env/python3")

    tag_command = coordinator._delivery_command("tag", [1])
    untagged_command = coordinator._delivery_command("untagged", [1])

    tag_cargo_file = tag_command[tag_command.index("--cargo-pick-file") + 1]
    untagged_cargo_file = untagged_command[
        untagged_command.index("--cargo-pick-file") + 1]
    assert tag_cargo_file == untagged_cargo_file
    assert tag_cargo_file.endswith("/delivery_presets.json")


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


def test_third_right_exit_uses_finer_a_pick_alignment_only_when_needed():
    follower = LaneFollower.__new__(LaneFollower)
    follower.task_index = 2
    follower.turn_cmd = "right"
    follower.enable_untagged_pick = True
    follower.untagged_pick_completed = False

    assert follower._exit_alignment_parameters() == (
        1.0, 0.012, 0.05, 0.12)

    follower.untagged_pick_completed = True
    assert follower._exit_alignment_parameters() == (
        2.0, 0.018, 0.08, 0.20)


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


def test_a_pick_prepare_waits_until_search_child_is_actually_ready():
    follower = LaneFollower.__new__(LaneFollower)
    follower.state = "A_PICK_PREPARE"
    follower.state_started = 0.0
    follower.untagged_pick_count = 4
    follower.untagged_search_started = False
    follower.untagged_search_enabled = False
    follower.grasp_coordinator = FakeCoordinator()
    follower.velocity_owner = "line"
    follower.active_pick_kind = None
    follower.stop_hits = 2
    follower.publish = lambda *args, **kwargs: True
    follower._set_state = lambda state: setattr(follower, "state", state)

    follower._handle_pick_without_frame(10.0)

    assert follower.state == "A_PICK_PREPARE"
    assert follower.grasp_coordinator.calls == [("untagged_search", 4)]
    follower.grasp_coordinator.search_ready = True
    follower._handle_pick_without_frame(10.1)

    assert follower.state == "A_PICK_SEARCH"
    assert follower.velocity_owner == "line"
    assert follower.stop_hits == 0
    assert follower.grasp_coordinator.search_enabled is False


def test_a_pick_prepare_stays_stopped_until_model_is_ready():
    follower = LaneFollower.__new__(LaneFollower)
    follower.state = "A_PICK_PREPARE"
    follower.state_started = 10.0
    follower.untagged_pick_count = 4
    follower.untagged_search_started = False
    follower.untagged_search_enabled = False
    follower.grasp_coordinator = FakeCoordinator()
    commands = []
    follower.publish = lambda *args, **kwargs: commands.append(args) or True
    follower._set_state = lambda state: setattr(follower, "state", state)

    follower._handle_pick_without_frame(10.0)

    assert commands == [(0, 0)]
    assert follower.grasp_coordinator.calls == [("untagged_search", 4)]
    assert follower.state == "A_PICK_PREPARE"
    follower.grasp_coordinator.search_ready = True

    follower._handle_pick_without_frame(10.1)

    assert commands[-1] == (0, 0)
    assert follower.grasp_coordinator.calls == [("untagged_search", 4)]
    assert follower.state == "A_PICK_SEARCH"


def test_a_pick_prepare_does_not_poll_stale_result_before_search_starts():
    follower = LaneFollower.__new__(LaneFollower)
    follower.state = "A_PICK_PREPARE"
    follower.state_started = 10.0
    follower.untagged_pick_count = 4
    follower.untagged_search_started = False
    follower.untagged_search_enabled = False
    follower.grasp_coordinator = FakeCoordinator()
    stale_result = {"present": True}

    def start_search(count):
        follower.grasp_coordinator.calls.append(("untagged_search", count))
        stale_result["present"] = False

    def poll_search():
        if stale_result["present"]:
            raise AssertionError("搜索启动前不应读取旧任务结果")
        return None, None

    follower.grasp_coordinator.start_untagged_search = start_search
    follower.grasp_coordinator.poll = poll_search
    follower.publish = lambda *args, **kwargs: True

    follower._handle_pick_without_frame(10.0)

    assert follower.state == "A_PICK_PREPARE"
    assert follower.grasp_coordinator.calls == [("untagged_search", 4)]


def test_a_pick_search_stops_before_releasing_chassis_to_grasp():
    follower = LaneFollower.__new__(LaneFollower)
    follower.grasp_coordinator = FakeCoordinator()
    follower.grasp_coordinator.search_triggered = True
    follower.untagged_search_enabled = True
    follower.untagged_search_speed = 0.03
    follower.velocity_owner = "line"
    follower.state = "A_PICK_SEARCH"
    order = []
    follower.publish = lambda *args, **kwargs: order.append("stop") or True
    follower.grasp_coordinator.release_untagged_search = \
        lambda: order.append("release")
    follower._set_state = lambda state: order.append(state)
    observation = types.SimpleNamespace(valid=True, center_x=320.0)
    cross = types.SimpleNamespace(candidate=False, stripe_polygons=[])

    follower._handle_untagged_search(observation, cross, 640)

    assert order == ["stop", "release", "A_PICKING"]
    assert follower.velocity_owner == "grasp"


def test_a_pick_search_drives_default_speed_for_configured_time(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    follower = LaneFollower.__new__(LaneFollower)
    follower.grasp_coordinator = FakeCoordinator()
    follower.stop_hits = 0
    follower.untagged_forward_started_at = 9.0
    follower.untagged_search_forward_time = 2.0
    follower.untagged_search_speed = 0.03
    follower.untagged_search_enabled = False
    commands = []
    follower._control = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("A 点搜索阶段不应使用巡线控制"))
    follower.publish = lambda *args, **kwargs: commands.append(args) or True
    follower._pick_failed = lambda message: (_ for _ in ()).throw(
        AssertionError(message))
    observation = types.SimpleNamespace(valid=True, center_x=333.0)
    cross = types.SimpleNamespace(candidate=False, stripe_polygons=[])

    follower._handle_untagged_search(observation, cross, 640)

    assert commands == [(0.16, 0.0)]
    assert follower.grasp_coordinator.search_enabled is False
    assert follower.grasp_coordinator.search_released is False


def test_a_pick_search_switches_to_slow_speed_after_forward_time(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    follower = LaneFollower.__new__(LaneFollower)
    follower.grasp_coordinator = FakeCoordinator()
    follower.stop_hits = 0
    follower.untagged_forward_started_at = 7.0
    follower.untagged_search_forward_time = 2.0
    follower.untagged_search_speed = 0.03
    follower.untagged_search_enabled = False
    commands = []
    follower.publish = lambda *args, **kwargs: commands.append(args) or True
    follower._pick_failed = lambda message: (_ for _ in ()).throw(
        AssertionError(message))
    observation = types.SimpleNamespace(valid=True, center_x=333.0)
    cross = types.SimpleNamespace(candidate=False, stripe_polygons=[])

    follower._handle_untagged_search(observation, cross, 640)

    assert follower.grasp_coordinator.calls == [
        ("untagged_search_enable",)]
    assert follower.untagged_search_enabled is True
    assert commands == [(0.03, 0.0)]


def test_a_pick_search_timer_starts_with_first_forward_command(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    follower = LaneFollower.__new__(LaneFollower)
    follower.grasp_coordinator = FakeCoordinator()
    follower.untagged_search_forward_time = 2.0
    follower.untagged_search_speed = 0.03
    follower.untagged_search_enabled = False
    follower.untagged_forward_started_at = None
    commands = []
    follower.publish = lambda *args, **kwargs: commands.append(args) or True
    follower._pick_failed = lambda message: (_ for _ in ()).throw(
        AssertionError(message))

    follower._handle_untagged_search(None, None, None)

    assert follower.untagged_forward_started_at == 10.0
    assert commands == [(0.16, 0.0)]
    assert follower.grasp_coordinator.search_enabled is False


def test_a_pick_search_without_lane_frame_uses_dedicated_handler():
    follower = LaneFollower.__new__(LaneFollower)
    follower.state = "A_PICK_SEARCH"
    calls = []
    follower._handle_untagged_search = lambda *args: calls.append(args)

    follower._handle_pick_without_frame(12.0)

    assert calls == [(None, None, None)]


def test_a_pick_search_keeps_driving_if_fourth_entry_arrives_without_target():
    follower = LaneFollower.__new__(LaneFollower)
    follower.grasp_coordinator = FakeCoordinator()
    follower.stop_hits = 2
    follower.untagged_search_enabled = True
    follower.untagged_search_speed = 0.03
    follower.entry_accept_after = 0.0
    failures = []
    follower._pick_failed = failures.append
    follower._control = lambda *args, **kwargs: None
    commands = []
    follower.publish = lambda *args, **kwargs: commands.append(args) or True
    observation = types.SimpleNamespace(valid=True, center_x=320.0)
    cross = types.SimpleNamespace(candidate=True, stripe_polygons=[object()])

    follower._handle_untagged_search(observation, cross, 640)

    assert failures == []
    assert commands == [(0.03, 0.0)]


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
    follower._entry_ready_state = lambda: "TRAFFIC_WAIT"
    follower._set_state = follower.states.append
    follower.untagged_pick_next_entry_time = 5.4
    follower.untagged_pick_next_turn_time = 3.2

    follower._finish_pick("untagged")

    assert follower.untagged_inventory == [1, 2]
    assert follower.untagged_pick_completed is True
    assert follower.untagged_pick_next_maneuver is True
    assert follower.velocity_owner == "line"
    assert follower.states == ["TRAFFIC_WAIT"]


def test_partial_untagged_inventory_continues_competition():
    follower = LaneFollower.__new__(LaneFollower)
    follower.grasp_coordinator = FakeCoordinator()
    follower.grasp_coordinator.completed_items = lambda: [1]
    follower.untagged_pick_count = 4
    follower.untagged_inventory = []
    follower.untagged_pick_completed = False
    follower.velocity_owner = "grasp"
    follower.bridge = types.SimpleNamespace(reset=lambda _width: None)
    follower.lane_width = 620.0
    follower.stop_hits = 3
    follower.states = []
    follower.publish = lambda *args, **kwargs: True
    follower._resume_yolo = lambda profile: profile == "building"
    follower._entry_ready_state = lambda: "TRAFFIC_WAIT"
    follower._set_state = follower.states.append
    follower.untagged_pick_next_entry_time = 5.4
    follower.untagged_pick_next_turn_time = 3.2
    failures = []
    follower._pick_failed = failures.append

    follower._finish_pick("untagged")

    assert failures == []
    assert follower.untagged_inventory == [1]
    assert follower.untagged_pick_completed is True
    assert follower.states == ["TRAFFIC_WAIT"]


def test_completed_tag_pick_waits_for_green_before_first_right_turn():
    follower = LaneFollower.__new__(LaneFollower)
    follower.grasp_coordinator = FakeCoordinator()
    follower.grasp_coordinator.completed_items = lambda: [1, 2, 3, 4]
    follower.tag_pick_count = 4
    follower.tag_inventory = []
    follower.tag_pick_completed = False
    follower.velocity_owner = "grasp"
    follower.bridge = types.SimpleNamespace(reset=lambda _width: None)
    follower.lane_width = 620.0
    follower.stop_hits = 3
    follower.states = []
    follower.publish = lambda *args, **kwargs: True
    follower._resume_yolo = lambda profile: profile == "street"
    follower._entry_ready_state = lambda: "TRAFFIC_WAIT"
    follower._set_state = follower.states.append
    follower.tag_pick_first_entry_time = 5.2
    follower.tag_pick_first_turn_time = 3.1

    follower._finish_pick("tag")

    assert follower.tag_inventory == [1, 2, 3, 4]
    assert follower.tag_pick_completed is True
    assert follower.tag_pick_first_maneuver is True
    assert follower.velocity_owner == "line"
    assert follower.states == ["TRAFFIC_WAIT"]


def test_partial_tag_pick_still_waits_for_green_and_continues():
    follower = LaneFollower.__new__(LaneFollower)
    follower.grasp_coordinator = FakeCoordinator()
    follower.grasp_coordinator.completed_items = lambda: [2]
    follower.tag_pick_count = 4
    follower.tag_inventory = []
    follower.tag_pick_completed = False
    follower.velocity_owner = "grasp"
    follower.bridge = types.SimpleNamespace(reset=lambda _width: None)
    follower.lane_width = 620.0
    follower.stop_hits = 3
    follower.publish = lambda *args, **kwargs: True
    follower._resume_yolo = lambda profile: profile == "street"
    follower._entry_ready_state = lambda: "TRAFFIC_WAIT"
    follower._set_state = lambda state: setattr(follower, "state", state)
    follower.tag_pick_first_entry_time = 5.2
    follower.tag_pick_first_turn_time = 3.1

    follower._finish_pick("tag")

    assert follower.tag_inventory == [2]
    assert follower.tag_pick_completed is True
    assert follower.state == "TRAFFIC_WAIT"


def test_zero_tag_pick_still_continues_without_delivery_inventory():
    follower = LaneFollower.__new__(LaneFollower)
    follower.grasp_coordinator = FakeCoordinator()
    follower.grasp_coordinator.completed_items = lambda: []
    follower.tag_pick_count = 4
    follower.tag_inventory = [1]
    follower.tag_pick_completed = False
    follower.velocity_owner = "grasp"
    follower.bridge = types.SimpleNamespace(reset=lambda _width: None)
    follower.lane_width = 620.0
    follower.stop_hits = 3
    follower.publish = lambda *args, **kwargs: True
    follower._resume_yolo = lambda profile: profile == "street"
    follower._entry_ready_state = lambda: "TRAFFIC_WAIT"
    follower._set_state = lambda state: setattr(follower, "state", state)
    follower.tag_pick_first_entry_time = 5.2
    follower.tag_pick_first_turn_time = 3.1

    follower._finish_pick("tag")

    assert follower.tag_inventory == []
    assert follower.tag_pick_completed is True
    assert follower.state == "TRAFFIC_WAIT"


def test_first_tag_pick_right_turn_uses_independent_times(monkeypatch):
    follower = LaneFollower.__new__(LaneFollower)
    follower.tag_pick_first_maneuver = True
    follower.task_index = 0
    follower.turn_cmd = "right"
    follower.turn_entry_time = 6.5
    follower.turn_time = 4.0
    follower.tag_pick_first_entry_time = 5.2
    follower.tag_pick_first_turn_time = 3.1
    follower.maneuver_phase = "ENTRY"
    follower.maneuver_phase_started = 10.0
    follower.turn_speed = 0.16
    follower.turn_angular = 0.58
    transitions = []
    commands = []

    def set_phase(phase, now=None):
        follower.maneuver_phase = phase
        follower.maneuver_phase_started = float(now)
        transitions.append(phase)

    follower._set_maneuver_phase = set_phase
    follower.publish = lambda linear, angular: commands.append(
        (linear, angular))

    follower._run_timed_turn_phase(15.1)
    assert transitions == []
    follower._run_timed_turn_phase(15.21)

    assert transitions == ["TURN"]
    assert commands[-1] == (0.16, -0.58)


def test_untagged_pick_next_left_turn_uses_independent_times():
    follower = LaneFollower.__new__(LaneFollower)
    follower.untagged_pick_next_maneuver = True
    follower.task_index = 3
    follower.turn_cmd = "left"
    follower.turn_entry_time = 6.5
    follower.turn_time = 4.0
    follower.untagged_pick_next_entry_time = 5.3
    follower.untagged_pick_next_turn_time = 3.2
    follower.maneuver_phase = "ENTRY"
    follower.maneuver_phase_started = 10.0
    follower.turn_speed = 0.16
    follower.turn_angular = 0.58
    transitions = []
    commands = []

    def set_phase(phase, now=None):
        follower.maneuver_phase = phase
        follower.maneuver_phase_started = float(now)
        transitions.append(phase)

    follower._set_maneuver_phase = set_phase
    follower.publish = lambda linear, angular: commands.append(
        (linear, angular))

    follower._run_timed_turn_phase(15.2)
    assert transitions == []
    follower._run_timed_turn_phase(15.31)

    assert transitions == ["TURN"]
    assert commands[-1] == (0.16, 0.58)


def test_third_right_before_untagged_pick_uses_independent_times():
    follower = LaneFollower.__new__(LaneFollower)
    follower.tag_pick_first_maneuver = False
    follower.untagged_pick_next_maneuver = False
    follower.task_index = 2
    follower.turn_cmd = "right"
    follower.enable_untagged_pick = True
    follower.untagged_pick_completed = False
    follower.turn_entry_time = 6.5
    follower.turn_time = 4.0
    follower.a_pick_third_right_entry_time = 7.2
    follower.a_pick_third_right_turn_time = 3.6
    follower.maneuver_phase = "ENTRY"
    follower.maneuver_phase_started = 10.0
    follower.turn_speed = 0.16
    follower.turn_angular = 0.58
    transitions = []
    commands = []

    def set_phase(phase, now=None):
        follower.maneuver_phase = phase
        follower.maneuver_phase_started = float(now)
        transitions.append(phase)

    follower._set_maneuver_phase = set_phase
    follower.publish = lambda linear, angular: commands.append(
        (linear, angular))

    follower._run_timed_turn_phase(17.1)
    assert transitions == []
    follower._run_timed_turn_phase(17.21)

    assert transitions == ["TURN"]
    assert commands[-1] == (0.16, -0.58)


def test_third_right_without_untagged_pick_keeps_normal_times():
    follower = LaneFollower.__new__(LaneFollower)
    follower.tag_pick_first_maneuver = False
    follower.untagged_pick_next_maneuver = False
    follower.task_index = 2
    follower.turn_cmd = "right"
    follower.enable_untagged_pick = False
    follower.untagged_pick_completed = True
    follower.turn_entry_time = 6.5
    follower.turn_time = 4.0
    follower.a_pick_third_right_entry_time = 1.0
    follower.a_pick_third_right_turn_time = 1.0
    follower.maneuver_phase = "ENTRY"
    follower.maneuver_phase_started = 10.0
    follower.turn_speed = 0.16
    follower.turn_angular = 0.58
    transitions = []
    follower._set_maneuver_phase = lambda phase, now=None: transitions.append(
        phase)
    follower.publish = lambda *_args: None

    follower._run_timed_turn_phase(11.1)

    assert transitions == []


def test_normal_intersections_do_not_use_tag_pick_first_times():
    follower = LaneFollower.__new__(LaneFollower)
    follower.tag_pick_first_maneuver = True
    follower.task_index = 1
    follower.turn_cmd = "straight"
    follower.turn_entry_time = 6.5
    follower.turn_time = 4.0
    follower.tag_pick_first_entry_time = 1.0
    follower.tag_pick_first_turn_time = 1.0
    follower.maneuver_phase = "ENTRY"
    follower.maneuver_phase_started = 10.0
    follower.turn_speed = 0.16
    follower.turn_angular = 0.58
    transitions = []
    follower._set_maneuver_phase = lambda phase, now=None: transitions.append(
        phase)
    follower.publish = lambda *_args: None

    follower._run_timed_turn_phase(11.1)

    assert transitions == []


def test_first_intersection_completion_clears_tag_pick_special_timing():
    follower = LaneFollower.__new__(LaneFollower)
    follower.task_index = 0
    follower.turn_cmd = "right"
    follower.tag_pick_first_maneuver = True
    follower.enable_untagged_pick = False
    follower.untagged_pick_completed = True
    follower.states = []
    follower._switch_yolo_profile_if_needed = lambda: None
    follower._set_state = follower.states.append

    follower._complete_intersection()

    assert follower.tag_pick_first_maneuver is False
    assert follower.task_index == 1
    assert follower.turn_cmd == "straight"
    assert follower.states == ["FOLLOW"]


def test_fourth_intersection_completion_clears_untagged_special_timing():
    follower = LaneFollower.__new__(LaneFollower)
    follower.task_index = 3
    follower.turn_cmd = "left"
    follower.tag_pick_first_maneuver = False
    follower.untagged_pick_next_maneuver = True
    follower.enable_untagged_pick = True
    follower.untagged_pick_completed = True
    follower.states = []
    follower._switch_yolo_profile_if_needed = lambda: None
    follower._set_state = follower.states.append

    follower._complete_intersection()

    assert follower.untagged_pick_next_maneuver is False
    assert follower.task_index == 4
    assert follower.turn_cmd == "straight"
    assert follower.states == ["FOLLOW"]


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
    assert follower.grasp_coordinator.calls == [
        ("delivery", "tag", [3], 0.0)]
    assert follower.states == ["DELIVERING"]


def test_next_delivery_waits_when_previous_arm_is_still_returning_idle():
    class BusyCoordinator(FakeCoordinator):
        @staticmethod
        def arm_job_active():
            return True

    follower = LaneFollower.__new__(LaneFollower)
    follower.enable_tag_delivery = True
    follower.enable_untagged_delivery = False
    follower.tag_inventory = [3]
    follower.untagged_inventory = []
    follower.tag_delivery_failed_ids = set()
    follower.untagged_delivery_failed_ids = set()
    follower.grasp_coordinator = BusyCoordinator()
    follower.delivery_arm_wait_reported = False
    follower.states = []
    follower.publish = lambda *args, **kwargs: True
    follower._set_state = follower.states.append
    event = types.SimpleNamespace(
        kind="street", area="C区", class_name="Recyclable waste",
        display_name="可回收垃圾")

    assert follower._start_delivery_for_event(event) is None
    assert follower.delivery_arm_wait_reported is True
    assert follower.grasp_coordinator.calls == []
    assert follower.states == []
    assert follower.tag_delivery_failed_ids == set()


def test_early_delivery_success_keeps_background_arm_return_running():
    class BusyCoordinator(FakeCoordinator):
        @staticmethod
        def arm_job_active():
            return True

    follower = LaneFollower.__new__(LaneFollower)
    follower.enable_untagged_pick = True
    follower.tag_inventory = []
    follower.untagged_inventory = [4]
    follower.tag_delivery_failed_ids = set()
    follower.untagged_delivery_failed_ids = set()
    follower.active_delivery_source = "untagged"
    follower.active_delivery_id = 4
    follower.grasp_coordinator = BusyCoordinator()
    follower.process_supervisor = FakeSupervisor()
    follower.states = []
    follower.publish = lambda *args, **kwargs: True
    follower._set_state = follower.states.append

    follower._finish_delivery(True)

    assert follower.untagged_inventory == []
    assert "stop_arm_common" not in follower.process_supervisor.calls
    assert follower.states == ["FOLLOW"]


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
    follower.building_delivery_calibration = {
        "targets": {
            "4": {
                "item_id": 4,
                "class_name": "Collapsed Building",
                "reference_distance_mm": 450.0,
                "min_distance_mm": 350.0,
                "max_distance_mm": 600.0,
                "width": {"a": 20000.0, "b": 50.0},
            },
        },
    }
    follower.states = []
    follower.publish = lambda *args, **kwargs: True
    follower._set_state = follower.states.append
    event = types.SimpleNamespace(
        kind="building", area="楼宇A", class_name="Collapsed Building",
        display_name="坍塌楼宇",
        detection=types.SimpleNamespace(
            box=(140.0, 108.0, 180.0, 132.0),
            frame_shape=(240, 320, 3)))

    assert follower._start_delivery_for_event(event) is True
    assert follower.active_delivery_source == "untagged"
    assert follower.active_delivery_id == 4
    call = follower.grasp_coordinator.calls[0]
    assert call[:3] == ("delivery", "untagged", [4])
    assert abs(call[3] - 0.06) < 1e-9
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
