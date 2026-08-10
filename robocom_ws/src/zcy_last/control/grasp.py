#!/usr/bin/env python3
# coding=utf-8
"""有 Tag/无 Tag 抓取与载物仓投递协调器。"""

import json
import os
import subprocess
import sys
import tempfile
import threading

from ..config import (
    PICK_CANDIDATE_IDS,
    TAG_ALIGN_SCRIPT,
    TAG_DELIVERY_PRESET_FILE,
    TAG_DELIVERY_SCRIPT,
    TAG_PRESET_FILE,
    UNTAGGED_CONFIG_FILE,
    UNTAGGED_DELIVERY_PRESET_FILE,
    UNTAGGED_PICK_SCRIPT,
    UNTAGGED_PRESET_FILE,
)


class GraspCoordinator(object):
    def __init__(self, supervisor, keep_arm_after_tag=False,
                 keep_arm_after_untagged=False, python3=None):
        self.supervisor = supervisor
        self.keep_arm_after_tag = bool(keep_arm_after_tag)
        self.keep_arm_after_untagged = bool(keep_arm_after_untagged)
        self.python3 = python3 or sys.executable
        self.thread = None
        self.kind = None
        self.result = None
        self.result_items = []
        self.error = None
        self.lock = threading.Lock()
        result_directory = getattr(supervisor, "log_dir", None)
        if not result_directory or not os.path.isdir(result_directory):
            result_directory = tempfile.gettempdir()
        self.tag_result_file = os.path.join(
            result_directory, "tag_pick_result_%d.json" % os.getpid())
        self.untagged_result_file = os.path.join(
            result_directory, "untagged_pick_result_%d.json" % os.getpid())

    @staticmethod
    def _sequence_text():
        return ",".join(str(item) for item in PICK_CANDIDATE_IDS)

    def _tag_command(self, count):
        try:
            os.unlink(self.tag_result_file)
        except OSError:
            pass
        return [
            "/usr/bin/python2", TAG_ALIGN_SCRIPT,
            "--sequence", self._sequence_text(),
            "--order", "left_to_right",
            "--max-targets", str(int(count)),
            "--fail-on-skip",
            "--result-file", self.tag_result_file,
            "--preset-file", TAG_PRESET_FILE,
            "--pick-velocity-scale", "0.2",
            "--pick-acceleration-scale", "0.2",
            "--tag-tf-wait-seconds", "12.0",
        ]

    def _delivery_command(self, source, item_ids):
        sequence = ",".join(str(int(item_id)) for item_id in item_ids)
        if source == "tag":
            delivery_file = TAG_DELIVERY_PRESET_FILE
            idle_preset_file = TAG_PRESET_FILE
        elif source == "untagged":
            delivery_file = UNTAGGED_DELIVERY_PRESET_FILE
            idle_preset_file = UNTAGGED_PRESET_FILE
        else:
            raise ValueError("未知投递来源：%s" % source)
        return [
            "/usr/bin/python2", TAG_DELIVERY_SCRIPT,
            "--mode", "run_delivery",
            "--sequence", sequence,
            "--delivery-file", delivery_file,
            "--tag-preset-file", idle_preset_file,
        ]

    def _read_pick_result(self, path, expected_count, label):
        try:
            with open(path, "r") as handle:
                payload = json.load(handle)
        except (IOError, OSError, ValueError) as exc:
            raise RuntimeError("无法读取%s抓取结果：%s" % (label, exc))
        completed_ids = payload.get("completed_ids")
        if not isinstance(completed_ids, list):
            raise RuntimeError("%s抓取结果缺少 completed_ids" % label)
        completed_ids = [int(item_id) for item_id in completed_ids]
        if (len(completed_ids) != int(expected_count)
                or len(set(completed_ids)) != len(completed_ids)
                or any(item_id not in PICK_CANDIDATE_IDS
                       for item_id in completed_ids)):
            raise RuntimeError(
                "%s抓取库存异常：期望 %d 个，实际 %s"
                % (label, int(expected_count), completed_ids))
        return completed_ids

    def _untagged_command(self, count):
        try:
            os.unlink(self.untagged_result_file)
        except OSError:
            pass
        return [
            self.python3, UNTAGGED_PICK_SCRIPT,
            "--run-chassis-sequence",
            "--sequence", self._sequence_text(),
            "--max-targets", str(int(count)),
            "--fail-on-skip",
            "--result-file", self.untagged_result_file,
            "--confidence", "0.5",
            "--config", UNTAGGED_CONFIG_FILE,
            "--preset-file", UNTAGGED_PRESET_FILE,
        ]

    def _run(self, kind, payload):
        success = False
        error = None
        result_items = []
        try:
            if kind == "tag":
                command = self._tag_command(payload)
            elif kind == "untagged":
                self.supervisor.start_astra()
                command = self._untagged_command(payload)
            elif kind == "delivery":
                self.supervisor.start_arm_common()
                source, item_ids = payload
                result_items = [int(item_id) for item_id in item_ids]
                command = self._delivery_command(source, result_items)
            else:
                raise ValueError("未知抓取类型：%s" % kind)
            job_name = "delivery" if kind == "delivery" else "pick_%s" % kind
            code = self.supervisor.run_job(job_name, command)
            if code != 0:
                raise subprocess.CalledProcessError(code, command)
            if kind == "tag":
                result_items = self._read_pick_result(
                    self.tag_result_file, payload, "有 Tag")
            elif kind == "untagged":
                result_items = self._read_pick_result(
                    self.untagged_result_file, payload, "无 Tag")
            success = True
        except Exception as exc:
            error = exc
        finally:
            if kind == "tag":
                self.supervisor.stop_tag_stack()
                self.supervisor.stop_astra()
                if not self.keep_arm_after_tag:
                    self.supervisor.stop_arm_common()
            elif kind == "untagged":
                self.supervisor.stop_astra()
                if not self.keep_arm_after_untagged:
                    self.supervisor.stop_arm_common()
            with self.lock:
                self.result = success
                self.result_items = result_items
                self.error = error

    def start(self, kind, count):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError("已有抓取流程正在运行")
            if not 1 <= int(count) <= len(PICK_CANDIDATE_IDS):
                raise ValueError("抓取数量必须在 1 到 4 之间")
            self.kind = str(kind)
            self.result = None
            self.result_items = []
            self.error = None
            self.thread = threading.Thread(
                target=self._run, args=(self.kind, int(count)))
            self.thread.daemon = True
            self.thread.start()

    def start_delivery(self, source, item_ids):
        source = str(source)
        if source not in ("tag", "untagged"):
            raise ValueError("投递来源必须是 tag 或 untagged")
        item_ids = [int(item_id) for item_id in item_ids]
        if (not item_ids or len(set(item_ids)) != len(item_ids)
                or any(item_id not in PICK_CANDIDATE_IDS
                       for item_id in item_ids)):
            raise ValueError("投递 ID 必须是 1 到 4 且不能重复")
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError("已有机械臂流程正在运行")
            self.kind = "delivery"
            self.result = None
            self.result_items = []
            self.error = None
            self.thread = threading.Thread(
                target=self._run, args=(self.kind, (source, item_ids)))
            self.thread.daemon = True
            self.thread.start()

    def poll(self):
        with self.lock:
            if self.result is None:
                return None, None
            return bool(self.result), self.error

    def completed_items(self):
        with self.lock:
            return list(self.result_items)

    def join(self, timeout=1.0):
        thread = self.thread
        if thread is not None:
            thread.join(timeout)
