#!/usr/bin/env python3
# coding=utf-8
"""比赛依赖进程的启动、就绪检查和按所有权关闭。"""

import datetime
import os
import signal
import subprocess
import sys
import time

from ..config import (
    ASTRA_CAMERA_INFO_FILE,
    DEPLOY_HOME,
    PICK_BASE_FRAME,
    PICK_CAMERA_FRAME,
    PROCESS_LOG_ROOT,
    PROCESS_START_TIMEOUT,
    PROCESS_STOP_TIMEOUT,
)


class ManagedProcess(object):
    def __init__(self, name, process, log_handle, command):
        self.name = str(name)
        self.process = process
        self.log_handle = log_handle
        self.command = list(command)


class ProcessSupervisor(object):
    """只终止由本实例启动的进程，避免误杀现场已有节点。"""

    def __init__(self, enabled=True, python3=None, log_root=PROCESS_LOG_ROOT):
        self.enabled = bool(enabled)
        self.python3 = python3 or sys.executable
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(os.path.expanduser(log_root), stamp)
        self.processes = {}
        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)
            self._info("本次运行日志目录：%s" % self.log_dir)

    @staticmethod
    def _info(message):
        print("[zcy_last] %s" % message, flush=True)

    @classmethod
    def _print_log_tail(cls, log_path, line_count=40):
        try:
            with open(log_path, "rb") as handle:
                lines = handle.read().decode("utf-8", "replace").splitlines()
        except (IOError, OSError) as exc:
            cls._info("无法读取日志 %s：%s" % (log_path, exc))
            return
        cls._info("%s 末尾日志：" % log_path)
        for line in lines[-max(1, int(line_count)):]:
            print("  %s" % line, flush=True)

    @staticmethod
    def _shell_command(command):
        return ["/bin/bash", "-lc", command]

    @staticmethod
    def _process_alive(item):
        return item is not None and item.process.poll() is None

    def start(self, name, shell_command):
        if not self.enabled:
            return None
        current = self.processes.get(name)
        if self._process_alive(current):
            return current
        log_path = os.path.join(self.log_dir, "%s.log" % name)
        log_handle = open(log_path, "ab", buffering=0)
        command = self._shell_command(shell_command)
        self._info("正在启动 %s，日志：%s" % (name, log_path))
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        item = ManagedProcess(name, process, log_handle, command)
        self.processes[name] = item
        time.sleep(0.4)
        if process.poll() is not None:
            code = process.returncode
            self._print_log_tail(log_path)
            self.stop(name)
            raise RuntimeError(
                "%s 启动失败（状态码 %s），查看日志 %s"
                % (name, code, log_path))
        self._info("%s 进程已启动" % name)
        return item

    def stop(self, name):
        item = self.processes.pop(name, None)
        if item is None:
            return
        try:
            if item.process.poll() is None:
                os.killpg(item.process.pid, signal.SIGTERM)
                try:
                    item.process.wait(timeout=PROCESS_STOP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    os.killpg(item.process.pid, signal.SIGKILL)
                    item.process.wait(timeout=PROCESS_STOP_TIMEOUT)
        finally:
            item.log_handle.close()

    def _probe(self, command, timeout=3.0):
        try:
            result = subprocess.run(
                self._shell_command(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def _assert_astra_not_running(self):
        """Astra 按 USB 设备启动，不依赖不稳定的 /dev/videoN 编号。"""
        if self._process_alive(self.processes.get("astra")):
            return
        if self._probe(
                "rosnode list 2>/dev/null | grep -qE '^/camera(/|$)'",
                timeout=2.0):
            raise RuntimeError(
                "检测到外部 Astra 相机节点；请先关闭临时启动的 "
                "astrapro.launch，再运行比赛主程序")

    def wait_until(self, description, probe, timeout=PROCESS_START_TIMEOUT,
                   watched=()):
        if not self.enabled:
            return
        self._info("正在等待%s（最长 %.1f 秒）" % (
            description, float(timeout)))
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            if probe():
                self._info("%s已就绪" % description)
                return
            for name in watched:
                item = self.processes.get(name)
                if item is None or item.process.poll() is None:
                    continue
                log_path = os.path.join(self.log_dir, "%s.log" % name)
                self._print_log_tail(log_path)
                raise RuntimeError(
                    "等待%s时 %s 异常退出（状态码 %s），"
                    "查看日志 %s" % (
                        description, name, item.process.returncode, log_path))
            time.sleep(0.5)
        for name in watched:
            log_path = os.path.join(self.log_dir, "%s.log" % name)
            if os.path.isfile(log_path):
                self._print_log_tail(log_path)
        raise RuntimeError(
            "等待%s超时，检查日志目录 %s"
            % (description, self.log_dir))

    def _base_ready(self):
        return self._probe(
            "rosnode list | grep -qx '/xnode_comm' && "
            "rosnode list | grep -qx '/xnode_vehicle'")

    def start_base(self):
        if not self.enabled:
            return
        if self._base_ready():
            self._info("检测到外部底盘节点，直接复用")
            return
        self.start(
            "base",
            "source /opt/ros/melodic/setup.bash && "
            "source {0}/robocom_ws/devel/setup.bash && "
            "exec roslaunch xpkg_bringup bringup_basic_ctrl.launch".format(
                DEPLOY_HOME),
        )
        self.wait_until(
            "底盘节点 /xnode_comm 和 /xnode_vehicle",
            self._base_ready,
            watched=("base",),
        )

    def require_external_base(self):
        """确认人工启动的底盘节点可用，但不取得其进程所有权。"""
        if not self.enabled:
            return
        self.wait_until(
            "外部底盘节点 /xnode_comm 和 /xnode_vehicle",
            self._base_ready,
            timeout=5.0,
        )

    def _arm_services_ready(self):
        return self._probe(
            "rosservice info /switch_pump_status >/dev/null 2>&1 && "
            "rosservice info /mirobot_startup_home >/dev/null 2>&1 && "
            "rostopic list | grep -q '^/move_group/status$'")

    def _handeye_tf_ready(self):
        return self._probe(
            "timeout 4 rosrun tf tf_echo {0} {1} 2>/dev/null | "
            "grep -m1 -q Translation".format(
                PICK_BASE_FRAME, PICK_CAMERA_FRAME),
            timeout=5.0)

    def start_arm_common(self):
        if not self.enabled:
            return
        source = (
            "source /opt/ros/melodic/setup.bash && "
            "source {0}/mirobot_ws/devel/setup.bash && "
            "source {0}/handeye-calib/devel/setup.bash && "
        ).format(DEPLOY_HOME)
        if self._arm_services_ready():
            self._info("检测到外部 MoveIt 和机械臂服务，直接复用")
        else:
            self.start(
                "moveit", source +
                "exec roslaunch mirobot_moveit_config mirobot.launch "
                "start_rviz:=false")
        if self._handeye_tf_ready():
            self._info("检测到外部手眼标定 TF，直接复用")
        else:
            self.start(
                "handeye_tf", source +
                # ROS Melodic 的 tf2_py 由 Python 2 编译。比赛主程序
                # 运行在 Conda Python 3 中，而 easy_handeye/publish.py
                # 使用 /usr/bin/env python，因此只对该 ROS 子进程
                # 优先使用系统 Python 2，不改变其他 Python 3 任务。
                "export PATH=/usr/bin:/bin:$PATH && "
                "exec roslaunch --screen easy_handeye publish.launch "
                "eye_on_hand:=false tracking_base_frame:=camera_link")
        self.wait_until(
            "机械臂服务",
            self._arm_services_ready,
            watched=("moveit",),
        )
        self.wait_until(
            "手眼标定 TF",
            self._handeye_tf_ready,
            watched=("moveit", "handeye_tf"),
        )

    def require_external_arm_common(self):
        """确认 launch.py 持有的机械臂公共依赖可用，不启动它们。"""
        if not self.enabled:
            return
        self.wait_until(
            "外部机械臂服务",
            self._arm_services_ready,
            timeout=5.0,
        )
        self.wait_until(
            "外部手眼标定 TF",
            self._handeye_tf_ready,
            timeout=5.0,
        )

    def stop_arm_common(self):
        self.stop("handeye_tf")
        self.stop("moveit")

    def start_astra(self):
        if not self.enabled:
            return
        self._assert_astra_not_running()
        calibration_argument = ""
        if os.path.isfile(ASTRA_CAMERA_INFO_FILE):
            calibration_argument = \
                " rgb_camera_info_url:=file://%s" % ASTRA_CAMERA_INFO_FILE
            self._info("Astra 使用指定 RGB 内参：%s" %
                       ASTRA_CAMERA_INFO_FILE)
        else:
            self._info(
                "警告：未找到指定 Astra RGB 内参文件 %s；"
                "按手动流程使用驱动默认内参启动，"
                "启动后仍会检查 CameraInfo.K。" %
                ASTRA_CAMERA_INFO_FILE)
        self.start(
            "astra",
            "source /opt/ros/melodic/setup.bash && "
            "source {0}/mirobot_ws/devel/setup.bash && "
            "exec roslaunch astra_camera astrapro.launch{1}".format(
                DEPLOY_HOME, calibration_argument),
        )
        try:
            self.wait_until(
                "Astra RGB 有效内参",
                lambda: self._probe(
                    "rostopic echo -n 1 /camera/rgb/camera_info 2>/dev/null | "
                    "grep -E -q '^K: \\[[^]]*[1-9]'",
                    timeout=6.0),
                watched=("astra",),
            )
        except RuntimeError as exc:
            if not calibration_argument:
                raise RuntimeError(
                    "%s；驱动默认 CameraInfo.K 为空，需要将已标定"
                    "文件放到 %s" % (exc, ASTRA_CAMERA_INFO_FILE))
            raise

    def stop_astra(self):
        self.stop("astra")

    def start_tag_stack(self):
        if not self.enabled:
            return
        source = (
            "source /opt/ros/melodic/setup.bash && "
            "source {0}/mirobot_ws/devel/setup.bash && "
            "source {0}/handeye-calib/devel/setup.bash && "
        ).format(DEPLOY_HOME)
        self.start(
            "tag_relay",
            source +
            "exec /usr/bin/python2 {0}/handeye-calib/src/"
            "tag_yolo_quiet_zone_relay.py --python3 {1} "
            "--image-topic /camera/rgb/image_rect_color --yolo-hz 5.0 "
            "--publish-hz 8.0 --confidence 0.08".format(
                DEPLOY_HOME, self.python3),
        )
        self.start(
            "apriltag",
            source +
            "exec roslaunch apriltag_ros continuous_detection.launch "
            "camera_name:=/tag_yolo_quiet image_topic:=image_raw "
            "publish_tag_detections_image:=true show_image:=false "
            "node_output:=log",
        )
        self.wait_until(
            "Tag 补白相机信息",
            lambda: self._probe(
                "rostopic echo -n 1 /tag_yolo_quiet/camera_info >/dev/null 2>&1",
                timeout=6.0),
            watched=("tag_relay", "apriltag"),
        )

    def stop_tag_stack(self):
        self.stop("apriltag")
        self.stop("tag_relay")

    def run_job(self, name, command):
        if not self.enabled:
            return subprocess.call(command)
        log_path = os.path.join(self.log_dir, "%s.log" % name)
        log_handle = open(log_path, "ab", buffering=0)
        process = subprocess.Popen(
            list(command), stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True)
        item = ManagedProcess(name, process, log_handle, command)
        self.processes[name] = item
        try:
            return process.wait()
        finally:
            self.processes.pop(name, None)
            log_handle.close()

    def check_owned_processes(self):
        failed = []
        for name, item in list(self.processes.items()):
            if name.startswith("pick_"):
                continue
            code = item.process.poll()
            if code is not None:
                failed.append((name, code))
        return failed

    def shutdown(self):
        errors = []
        for name in list(self.processes.keys())[::-1]:
            try:
                self.stop(name)
            except Exception as exc:
                errors.append((name, exc))
                self._info("关闭 %s 失败：%s；继续清理其他进程" % (
                    name, exc))
        if errors:
            self._info("进程清理完成，但有 %d 个进程关闭异常" % len(errors))
