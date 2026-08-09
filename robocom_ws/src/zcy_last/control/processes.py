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
    SHARED_OBJECT_CAMERA_INDEX,
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
            self.stop(name)
            raise RuntimeError(
                "%s 启动失败，查看日志 %s" % (name, log_path))
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

    def _assert_shared_camera_available(self):
        device = "/dev/video%d" % SHARED_OBJECT_CAMERA_INDEX
        if not os.path.exists(device):
            raise RuntimeError("共享物体摄像头不存在：%s" % device)
        if self._probe("fuser -s %s" % device, timeout=2.0):
            raise RuntimeError("共享物体摄像头正被其他进程占用：%s" % device)

    def wait_until(self, description, probe, timeout=PROCESS_START_TIMEOUT):
        if not self.enabled:
            return
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            if probe():
                return
            time.sleep(0.5)
        raise RuntimeError("等待%s超时" % description)

    def require_external_base(self):
        """确认人工启动的底盘节点可用，但不取得其进程所有权。"""
        if not self.enabled:
            return
        self.wait_until(
            "外部底盘节点 /xnode_comm 和 /xnode_vehicle",
            lambda: self._probe(
                "rosnode list | grep -qx '/xnode_comm' && "
                "rosnode list | grep -qx '/xnode_vehicle'"),
            timeout=5.0,
        )

    def start_arm_common(self):
        if not self.enabled:
            return
        source = (
            "source /opt/ros/melodic/setup.bash && "
            "source {0}/mirobot_ws/devel/setup.bash && "
            "source {0}/handeye-calib/devel/setup.bash && "
        ).format(DEPLOY_HOME)
        self.start(
            "moveit", source +
            "exec roslaunch mirobot_moveit_config mirobot.launch start_rviz:=false")
        self.start(
            "handeye_tf", source +
            "exec roslaunch easy_handeye publish.launch "
            "eye_on_hand:=false tracking_base_frame:=camera_link")
        self.wait_until(
            "机械臂服务",
            lambda: self._probe(
                "rosservice info /switch_pump_status >/dev/null 2>&1 && "
                "rosservice info /mirobot_startup_home >/dev/null 2>&1 && "
                "rostopic list | grep -q '^/move_group/status$'"),
        )
        self.wait_until(
            "手眼标定 TF",
            lambda: self._probe(
                "timeout 4 rosrun tf tf_echo {0} {1} 2>/dev/null | "
                "grep -m1 -q Translation".format(
                    PICK_BASE_FRAME, PICK_CAMERA_FRAME),
                timeout=5.0),
        )

    def stop_arm_common(self):
        self.stop("handeye_tf")
        self.stop("moveit")

    def start_astra(self):
        if not self.enabled:
            return
        if not os.path.isfile(ASTRA_CAMERA_INFO_FILE):
            raise RuntimeError("Astra RGB 内参文件不存在：%s" %
                               ASTRA_CAMERA_INFO_FILE)
        self._assert_shared_camera_available()
        self.start(
            "astra",
            "source /opt/ros/melodic/setup.bash && "
            "source {0}/mirobot_ws/devel/setup.bash && "
            "exec roslaunch astra_camera astrapro.launch "
            "rgb_camera_info_url:=file://{1}".format(
                DEPLOY_HOME, ASTRA_CAMERA_INFO_FILE),
        )
        self.wait_until(
            "Astra RGB 和内参",
            lambda: self._probe(
                "rostopic echo -n 1 /camera/rgb/camera_info 2>/dev/null | "
                "grep -E -q '^K: \\[[^]]*[1-9]'",
                timeout=6.0),
        )

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
            "publish_tag_detections_image:=true show_image:=false node_output:=log",
        )
        self.wait_until(
            "Tag 检测图像",
            lambda: self._probe(
                "rostopic echo -n 1 /tag_yolo_quiet/camera_info >/dev/null 2>&1",
                timeout=6.0),
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
        for name in list(self.processes.keys())[::-1]:
            self.stop(name)
