#!/usr/bin/env python3
# coding=utf-8
"""B 点有 Tag 与 A 点无 Tag 抓取协调器。"""

import subprocess
import sys
import threading

from ..config import (
    PICK_CANDIDATE_IDS,
    TAG_ALIGN_SCRIPT,
    TAG_PRESET_FILE,
    UNTAGGED_CONFIG_FILE,
    UNTAGGED_PICK_SCRIPT,
    UNTAGGED_PRESET_FILE,
)


class GraspCoordinator(object):
    def __init__(self, supervisor, keep_arm_after_tag=False, python3=None):
        self.supervisor = supervisor
        self.keep_arm_after_tag = bool(keep_arm_after_tag)
        self.python3 = python3 or sys.executable
        self.thread = None
        self.kind = None
        self.result = None
        self.error = None
        self.lock = threading.Lock()

    @staticmethod
    def _sequence_text():
        return ",".join(str(item) for item in PICK_CANDIDATE_IDS)

    def _tag_command(self, count):
        return [
            "/usr/bin/python2", TAG_ALIGN_SCRIPT,
            "--sequence", self._sequence_text(),
            "--order", "left_to_right",
            "--max-targets", str(int(count)),
            "--fail-on-skip",
            "--preset-file", TAG_PRESET_FILE,
            "--pick-velocity-scale", "0.2",
            "--pick-acceleration-scale", "0.2",
            "--tag-tf-wait-seconds", "12.0",
        ]

    def _untagged_command(self, count):
        return [
            self.python3, UNTAGGED_PICK_SCRIPT,
            "--run-chassis-sequence",
            "--sequence", self._sequence_text(),
            "--max-targets", str(int(count)),
            "--fail-on-skip",
            "--confidence", "0.5",
            "--config", UNTAGGED_CONFIG_FILE,
            "--preset-file", UNTAGGED_PRESET_FILE,
        ]

    def _run(self, kind, count):
        success = False
        error = None
        try:
            if kind == "tag":
                command = self._tag_command(count)
            elif kind == "untagged":
                self.supervisor.start_astra()
                command = self._untagged_command(count)
            else:
                raise ValueError("未知抓取类型：%s" % kind)
            code = self.supervisor.run_job("pick_%s" % kind, command)
            if code != 0:
                raise subprocess.CalledProcessError(code, command)
            success = True
        except Exception as exc:
            error = exc
        finally:
            if kind == "tag":
                self.supervisor.stop_tag_stack()
                self.supervisor.stop_astra()
                if not self.keep_arm_after_tag:
                    self.supervisor.stop_arm_common()
            else:
                self.supervisor.stop_astra()
                self.supervisor.stop_arm_common()
            with self.lock:
                self.result = success
                self.error = error

    def start(self, kind, count):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError("已有抓取流程正在运行")
            if not 1 <= int(count) <= len(PICK_CANDIDATE_IDS):
                raise ValueError("抓取数量必须在 1 到 4 之间")
            self.kind = str(kind)
            self.result = None
            self.error = None
            self.thread = threading.Thread(
                target=self._run, args=(self.kind, int(count)))
            self.thread.daemon = True
            self.thread.start()

    def poll(self):
        with self.lock:
            if self.result is None:
                return None, None
            return bool(self.result), self.error

    def join(self, timeout=1.0):
        thread = self.thread
        if thread is not None:
            thread.join(timeout)
