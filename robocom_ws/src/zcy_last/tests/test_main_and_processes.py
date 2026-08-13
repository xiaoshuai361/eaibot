#!/usr/bin/env python3
# coding=utf-8
"""一键入口参数组合和依赖进程防护测试。"""

import os
import sys
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
    rospy.ROSInterruptException = RuntimeError
    geometry_msgs = sys.modules.setdefault(
        "geometry_msgs", types.ModuleType("geometry_msgs"))
    geometry_msgs_msg = sys.modules.setdefault(
        "geometry_msgs.msg", types.ModuleType("geometry_msgs.msg"))
    geometry_msgs_msg.Twist = _Twist
    geometry_msgs.msg = geometry_msgs_msg


_install_ros_stubs()

from zcy_last import main as task_main  # noqa: E402
from zcy_last import launch as task_launch  # noqa: E402
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


def test_tag_delivery_requires_tag_pick_and_can_be_disabled():
    enabled, _ = task_main.parse_args(["--tag-pick", "--tag-delivery"])
    disabled, _ = task_main.parse_args([
        "--tag-pick", "--no-tag-delivery"])
    no_inventory, _ = task_main.parse_args([
        "--no-tag-pick", "--tag-delivery"])

    assert enabled.tag_delivery is True
    assert disabled.tag_delivery is False
    assert no_inventory.tag_delivery is False


def test_tag_pick_defaults_to_all_four_targets():
    options, _ = task_main.parse_args(["--tag-pick"])

    assert options.tag_pick_count == 4


def test_untagged_delivery_requires_untagged_pick_and_can_be_disabled():
    enabled, _ = task_main.parse_args([
        "--untagged-pick", "--untagged-delivery"])
    disabled, _ = task_main.parse_args([
        "--untagged-pick", "--no-untagged-delivery"])
    no_inventory, _ = task_main.parse_args([
        "--no-untagged-pick", "--untagged-delivery"])

    assert enabled.untagged_delivery is True
    assert disabled.untagged_delivery is False
    assert no_inventory.untagged_delivery is False


@pytest.mark.parametrize(
    "argv,expected",
    [
        (
            ["--no-tag-pick", "--no-untagged-pick"],
            (False, False, False, False),
        ),
        (
            ["--tag-pick", "--tag-pick-count", "4", "--tag-delivery",
             "--no-untagged-pick"],
            (True, True, False, False),
        ),
        (
            ["--tag-pick", "--tag-pick-count", "4", "--no-tag-delivery",
             "--no-untagged-pick"],
            (True, False, False, False),
        ),
        (
            ["--no-tag-pick", "--untagged-pick",
             "--untagged-pick-count", "3", "--untagged-delivery"],
            (False, False, True, True),
        ),
        (
            ["--no-tag-pick", "--untagged-pick",
             "--untagged-pick-count", "3", "--no-untagged-delivery"],
            (False, False, True, False),
        ),
    ],
)
def test_five_official_competition_commands(argv, expected):
    options, ros_args = task_main.parse_args(argv)

    actual = (
        options.tag_pick,
        options.tag_delivery,
        options.untagged_pick,
        options.untagged_delivery,
    )
    assert actual == expected
    assert ros_args == []


@pytest.mark.parametrize(
    "argv,expected_calls",
    [
        (
            ["--no-tag-pick", "--no-untagged-pick"],
            ["run", "shutdown"],
        ),
        (
            ["--tag-pick", "--no-untagged-pick"],
            ["astra", "tag", "run", "shutdown"],
        ),
        (
            ["--no-tag-pick", "--untagged-pick"],
            ["run", "shutdown"],
        ),
        (
            ["--tag-pick", "--untagged-pick"],
            ["astra", "tag", "run", "shutdown"],
        ),
    ],
)
def test_main_starts_only_required_initial_dependencies(
        monkeypatch, argv, expected_calls):
    calls = []

    class Supervisor(object):
        def __init__(self, **_kwargs):
            pass

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


def test_runtime_interface_check_rejects_mixed_module_versions(monkeypatch):
    class OldLaneFollower(object):
        def __init__(self, enable_tag_pick=False):
            pass

    monkeypatch.setattr(task_main, "LaneFollower", OldLaneFollower)

    with pytest.raises(RuntimeError, match="模块版本不一致.*enable_tag_delivery"):
        task_main.validate_runtime_interfaces()


def test_astra_start_falls_back_to_driver_calibration(monkeypatch, tmp_path):
    supervisor = processes.ProcessSupervisor(
        enabled=True, log_root=str(tmp_path))
    monkeypatch.setattr(
        processes, "ASTRA_CAMERA_INFO_FILE", str(tmp_path / "missing.yaml"))
    monkeypatch.setattr(
        supervisor, "_assert_astra_not_running", lambda: None)
    started = []
    waited = []
    monkeypatch.setattr(
        supervisor, "start",
        lambda name, command: started.append((name, command)))
    monkeypatch.setattr(
        supervisor, "wait_until",
        lambda description, *_args, **_kwargs: waited.append(description))

    supervisor.start_astra()

    assert started[0][0] == "astra"
    assert "rgb_camera_info_url" not in started[0][1]
    assert waited == ["Astra RGB 有效内参"]


def test_astra_start_prefers_explicit_calibration(monkeypatch, tmp_path):
    calibration = tmp_path / "astra.yaml"
    calibration.write_text("camera_name: camera\n", encoding="utf-8")
    supervisor = processes.ProcessSupervisor(
        enabled=True, log_root=str(tmp_path))
    monkeypatch.setattr(
        processes, "ASTRA_CAMERA_INFO_FILE", str(calibration))
    monkeypatch.setattr(
        supervisor, "_assert_astra_not_running", lambda: None)
    started = []
    monkeypatch.setattr(
        supervisor, "start",
        lambda name, command: started.append((name, command)))
    monkeypatch.setattr(supervisor, "wait_until", lambda *_args, **_kwargs: None)

    supervisor.start_astra()

    assert "rgb_camera_info_url:=file://%s" % calibration in started[0][1]


def test_external_astra_is_rejected(monkeypatch, tmp_path):
    supervisor = processes.ProcessSupervisor(
        enabled=True, log_root=str(tmp_path))
    monkeypatch.setattr(supervisor, "_probe", lambda *_args, **_kwargs: True)

    with pytest.raises(RuntimeError, match="外部 Astra"):
        supervisor._assert_astra_not_running()


def test_astra_check_does_not_require_v4l_device(monkeypatch, tmp_path):
    supervisor = processes.ProcessSupervisor(
        enabled=True, log_root=str(tmp_path))
    monkeypatch.setattr(supervisor, "_probe", lambda *_args, **_kwargs: False)

    supervisor._assert_astra_not_running()


def test_tag_stack_keeps_detector_input_clean_and_avoids_duplicate_window(
        monkeypatch, tmp_path):
    supervisor = processes.ProcessSupervisor(
        enabled=True, log_root=str(tmp_path))
    started = []
    monkeypatch.setattr(
        supervisor, "start",
        lambda name, command: started.append((name, command)))
    monkeypatch.setattr(supervisor, "wait_until", lambda *_args, **_kwargs: None)

    supervisor.start_tag_stack()

    commands = dict(started)
    assert "--show-yolo-boxes" not in commands["tag_relay"]
    assert "publish_tag_detections_image:=true" in commands["apriltag"]
    assert "show_image:=false" in commands["apriltag"]


def test_wait_until_reports_owned_process_exit_and_log_tail(tmp_path, capsys):
    supervisor = processes.ProcessSupervisor(
        enabled=True, log_root=str(tmp_path))
    log_path = tmp_path / os.path.basename(supervisor.log_dir) / \
        "handeye_tf.log"
    log_path.write_text("calibration file missing\n", encoding="utf-8")
    process = types.SimpleNamespace(poll=lambda: 7, returncode=7)
    supervisor.processes["handeye_tf"] = types.SimpleNamespace(
        process=process)

    with pytest.raises(RuntimeError, match="handeye_tf.*状态码 7"):
        supervisor.wait_until(
            "手眼标定 TF", lambda: False,
            timeout=0.1, watched=("handeye_tf",))

    assert "calibration file missing" in capsys.readouterr().out


def test_arm_common_reuses_prestarted_external_stack(monkeypatch, tmp_path):
    supervisor = processes.ProcessSupervisor(
        enabled=True, log_root=str(tmp_path))
    monkeypatch.setattr(supervisor, "_arm_services_ready", lambda: True)
    monkeypatch.setattr(supervisor, "_handeye_tf_ready", lambda: True)
    started = []
    monkeypatch.setattr(
        supervisor, "start",
        lambda name, command: started.append((name, command)))

    supervisor.start_arm_common()
    supervisor.stop_arm_common()

    assert started == []


def test_main_arm_check_never_starts_common_stack(monkeypatch, tmp_path):
    supervisor = processes.ProcessSupervisor(
        enabled=True, log_root=str(tmp_path))
    waited = []
    started = []
    monkeypatch.setattr(
        supervisor, "wait_until",
        lambda description, *_args, **_kwargs: waited.append(description))
    monkeypatch.setattr(
        supervisor, "start",
        lambda name, command: started.append((name, command)))

    supervisor.require_external_arm_common()

    assert waited == ["外部机械臂服务", "外部手眼标定 TF"]
    assert started == []


def test_arm_common_forces_python2_for_melodic_handeye(monkeypatch, tmp_path):
    supervisor = processes.ProcessSupervisor(
        enabled=True, log_root=str(tmp_path))
    monkeypatch.setattr(supervisor, "_arm_services_ready", lambda: False)
    monkeypatch.setattr(supervisor, "_handeye_tf_ready", lambda: False)
    started = []
    monkeypatch.setattr(
        supervisor, "start",
        lambda name, command: started.append((name, command)))
    monkeypatch.setattr(
        supervisor, "wait_until", lambda *_args, **_kwargs: None)

    supervisor.start_arm_common()

    commands = dict(started)
    assert "export PATH=/usr/bin:/bin:$PATH" in commands["handeye_tf"]
    assert "easy_handeye publish.launch" in commands["handeye_tf"]


def test_dependency_launcher_keeps_common_stack_and_releases_temporary_astra():
    calls = []

    class Supervisor(object):
        def start_base(self):
            calls.append("base")

        def start_astra(self):
            calls.append("astra")

        def start_arm_common(self):
            calls.append("arm")

        def stop_astra(self):
            calls.append("astra_stop")

    task_launch.launch_dependencies(Supervisor())

    assert calls == ["base", "astra", "arm", "astra_stop"]


def test_dependency_launcher_main_always_cleans_owned_processes(monkeypatch):
    calls = []

    class Supervisor(object):
        def __init__(self, **_kwargs):
            calls.append("supervisor")

        def shutdown(self):
            calls.append("shutdown")

    def fail_after_start(_supervisor):
        calls.append("launch")
        raise RuntimeError("stop launcher")

    monkeypatch.setattr(task_launch, "ProcessSupervisor", Supervisor)
    monkeypatch.setattr(task_launch, "launch_dependencies", fail_after_start)

    with pytest.raises(RuntimeError, match="stop launcher"):
        task_launch.main()

    assert calls == ["supervisor", "launch", "shutdown"]


def test_process_shutdown_stops_owned_processes_in_reverse_order(
        monkeypatch, tmp_path):
    supervisor = processes.ProcessSupervisor(
        enabled=True, log_root=str(tmp_path))
    supervisor.processes = {
        "base": object(),
        "moveit": object(),
        "handeye_tf": object(),
    }
    stopped = []
    monkeypatch.setattr(supervisor, "stop", lambda name: stopped.append(name))

    supervisor.shutdown()

    assert stopped == ["handeye_tf", "moveit", "base"]


def test_process_shutdown_continues_after_one_stop_fails(
        monkeypatch, tmp_path):
    supervisor = processes.ProcessSupervisor(
        enabled=True, log_root=str(tmp_path))
    supervisor.processes = {
        "base": object(),
        "moveit": object(),
        "handeye_tf": object(),
    }
    stopped = []

    def stop(name):
        stopped.append(name)
        if name == "moveit":
            raise OSError("stop failed")

    monkeypatch.setattr(supervisor, "stop", stop)

    supervisor.shutdown()

    assert stopped == ["handeye_tf", "moveit", "base"]


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
    monkeypatch.setattr(
        runtime.rospy, "get_time", lambda: next(clock), raising=False)
    pid = runtime.PID(kp=10.0, kd=0.0, limit=0.5)

    assert pid.update(-100.0) == pytest.approx(0.5)
    assert pid.update(100.0) == pytest.approx(-0.5)
