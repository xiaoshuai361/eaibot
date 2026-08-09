#!/usr/bin/env python3
# coding=utf-8
"""一键入口参数组合和依赖进程防护测试。"""

import sys
import types

import pytest


def _install_ros_stubs():
    rospy = sys.modules.setdefault("rospy", types.ModuleType("rospy"))
    rospy.ROSInterruptException = RuntimeError
    geometry_msgs = sys.modules.setdefault(
        "geometry_msgs", types.ModuleType("geometry_msgs"))
    geometry_msgs_msg = sys.modules.setdefault(
        "geometry_msgs.msg", types.ModuleType("geometry_msgs.msg"))
    geometry_msgs_msg.Twist = type("Twist", (), {})
    geometry_msgs.msg = geometry_msgs_msg


_install_ros_stubs()

from zcy_last import main as task_main  # noqa: E402
from zcy_last.control import processes  # noqa: E402
from zcy_last.control import runtime  # noqa: E402


@pytest.mark.parametrize(
    "argv,tag_enabled,untagged_enabled",
    [
        (["--no-tag-pick", "--no-untagged-pick"], False, False),
        (["--tag-pick", "--no-untagged-pick"], True, False),
        (["--no-tag-pick", "--untagged-pick"], False, True),
        (["--tag-pick", "--untagged-pick"], True, True),
    ],
)
def test_parse_four_pick_switch_combinations(
        argv, tag_enabled, untagged_enabled):
    options, ros_args = task_main.parse_args(argv)

    assert options.tag_pick is tag_enabled
    assert options.untagged_pick is untagged_enabled
    assert ros_args == []


@pytest.mark.parametrize("option", ["--tag-pick-count", "--untagged-pick-count"])
@pytest.mark.parametrize("count", [0, 5])
def test_pick_count_must_be_between_one_and_four(option, count):
    with pytest.raises(SystemExit):
        task_main.parse_args([option, str(count)])


@pytest.mark.parametrize(
    "argv,expected_calls",
    [
        (
            ["--no-tag-pick", "--no-untagged-pick"],
            ["base_check", "run", "shutdown"],
        ),
        (
            ["--tag-pick", "--no-untagged-pick"],
            ["base_check", "arm", "astra", "tag", "run", "shutdown"],
        ),
        (
            ["--no-tag-pick", "--untagged-pick"],
            ["base_check", "arm", "run", "shutdown"],
        ),
        (
            ["--tag-pick", "--untagged-pick"],
            ["base_check", "arm", "astra", "tag", "run", "shutdown"],
        ),
    ],
)
def test_main_starts_only_required_initial_dependencies(
        monkeypatch, argv, expected_calls):
    calls = []

    class Supervisor(object):
        def __init__(self, **_kwargs):
            pass

        def require_external_base(self):
            calls.append("base_check")

        def start_arm_common(self):
            calls.append("arm")

        def start_astra(self):
            calls.append("astra")

        def start_tag_stack(self):
            calls.append("tag")

        def shutdown(self):
            calls.append("shutdown")

    class Coordinator(object):
        def __init__(self, *_args, **_kwargs):
            pass

    class Follower(object):
        def __init__(self, **_kwargs):
            pass

        def run(self):
            calls.append("run")

    monkeypatch.setattr(task_main, "ProcessSupervisor", Supervisor)
    monkeypatch.setattr(task_main, "GraspCoordinator", Coordinator)
    monkeypatch.setattr(task_main, "LaneFollower", Follower)
    monkeypatch.setattr(sys, "argv", ["zcy_last"])

    task_main.main(argv)

    assert calls == expected_calls


def test_astra_start_rejects_missing_calibration(monkeypatch, tmp_path):
    supervisor = processes.ProcessSupervisor(
        enabled=True, log_root=str(tmp_path))
    monkeypatch.setattr(
        processes, "ASTRA_CAMERA_INFO_FILE", str(tmp_path / "missing.yaml"))

    with pytest.raises(RuntimeError, match="内参文件不存在"):
        supervisor.start_astra()


def test_shared_camera_occupancy_is_rejected(monkeypatch, tmp_path):
    supervisor = processes.ProcessSupervisor(
        enabled=True, log_root=str(tmp_path))
    monkeypatch.setattr(processes.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(supervisor, "_probe", lambda *_args, **_kwargs: True)

    with pytest.raises(RuntimeError, match="正被其他进程占用"):
        supervisor._assert_shared_camera_available()


def test_camera_reader_applies_requested_resolution(monkeypatch):
    settings = []

    class Capture(object):
        def isOpened(self):
            return True

        def set(self, name, value):
            settings.append((name, value))

        def release(self):
            pass

    class Thread(object):
        def __init__(self, target):
            self.target = target
            self.daemon = False

        def start(self):
            pass

        def join(self, _timeout):
            pass

    monkeypatch.setattr(runtime.cv2, "VideoCapture", lambda *_args: Capture())
    monkeypatch.setattr(runtime.threading, "Thread", Thread)

    reader = runtime.CameraReader(2, 320, 240)
    reader.release()

    assert settings == [
        (runtime.cv2.CAP_PROP_FRAME_WIDTH, 320),
        (runtime.cv2.CAP_PROP_FRAME_HEIGHT, 240),
        (runtime.cv2.CAP_PROP_BUFFERSIZE, 1),
    ]


def test_pid_output_is_limited_without_algorithm_module_dependency(monkeypatch):
    clock = iter([1.0, 1.1])
    monkeypatch.setattr(runtime.rospy, "get_time", lambda: next(clock))
    pid = runtime.PID(kp=10.0, kd=0.0, limit=0.5)

    assert pid.update(-100.0) == pytest.approx(0.5)
    assert pid.update(100.0) == pytest.approx(-0.5)
