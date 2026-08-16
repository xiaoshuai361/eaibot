#!/usr/bin/env python3
# coding=utf-8
"""有 Tag/无 Tag 抓取与载物仓投递协调器。"""

import json
import os
import sys
import tempfile
import threading

from ..config import (
    PICK_CANDIDATE_IDS,
    PICK_DEBUG_VIEW,
    TAG_ALIGN_SCRIPT,
    TAG_PICK_TF_WAIT_SECONDS,
    TAG_DELIVERY_PRESET_FILE,
    TAG_DELIVERY_SCRIPT,
    TAG_PRESET_FILE,
    UNTAGGED_CONFIG_FILE,
    UNTAGGED_DELIVERY_PRESET_FILE,
    UNTAGGED_PICK_SCRIPT,
    UNTAGGED_PRESET_FILE,
    UNTAGGED_SEARCH_POLL_HZ,
    UNTAGGED_SEARCH_STABLE_FRAMES,
)

UNTAGGED_PICK_IDS = (2, 3)


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
        self.untagged_search_ready_file = os.path.join(
            result_directory, "untagged_search_ready_%d" % os.getpid())
        self.untagged_search_enable_file = os.path.join(
            result_directory, "untagged_search_enable_%d" % os.getpid())
        self.untagged_search_trigger_file = os.path.join(
            result_directory, "untagged_search_trigger_%d" % os.getpid())
        self.untagged_search_release_file = os.path.join(
            result_directory, "untagged_search_release_%d" % os.getpid())
        self.delivery_release_ready_file = os.path.join(
            result_directory, "delivery_release_ready_%d" % os.getpid())
        self.delivery_source = None

    @staticmethod
    def _sequence_text(item_ids=PICK_CANDIDATE_IDS):
        return ",".join(str(item) for item in item_ids)

    def _tag_command(self, count):
        try:
            os.unlink(self.tag_result_file)
        except OSError:
            pass
        command = [
            "/usr/bin/python2", "-u", TAG_ALIGN_SCRIPT,
            "--sequence", self._sequence_text(),
            "--order", "left_to_right",
            "--max-targets", str(int(count)),
            "--allow-partial",
            "--result-file", self.tag_result_file,
            "--preset-file", TAG_PRESET_FILE,
            "--pick-velocity-scale", "0.2",
            "--pick-acceleration-scale", "0.2",
            "--pick-approach-gap", "0.030",
            "--tag-tf-wait-seconds", str(float(TAG_PICK_TF_WAIT_SECONDS)),
        ]
        if PICK_DEBUG_VIEW:
            command.append("--show-debug-window")
        return command

    def _delivery_command(self, source, item_ids,
                          contact_distance_offset_m=0.0):
        sequence = ",".join(str(int(item_id)) for item_id in item_ids)
        if source == "tag":
            delivery_file = TAG_DELIVERY_PRESET_FILE
            idle_preset_file = TAG_PRESET_FILE
        elif source == "untagged":
            delivery_file = UNTAGGED_DELIVERY_PRESET_FILE
            idle_preset_file = UNTAGGED_PRESET_FILE
        else:
            raise ValueError("未知投递来源：%s" % source)
        command = [
            "/usr/bin/python2", "-u", TAG_DELIVERY_SCRIPT,
            "--mode", "run_delivery",
            "--sequence", sequence,
            "--delivery-file", delivery_file,
            "--cargo-pick-file", TAG_DELIVERY_PRESET_FILE,
            "--tag-preset-file", idle_preset_file,
        ]
        if source == "tag":
            command.extend([
                "--release-ready-file", self.delivery_release_ready_file,
                "--pump-off-settle-seconds", "0.0",
            ])
        if source == "untagged":
            command.extend([
                "--release-ready-file", self.delivery_release_ready_file,
                "--pump-off-settle-seconds", "0.7",
                "--release-ready-delay-seconds", "3.0",
                "--contact-release",
                "--force-release-on-contact-miss",
                "--contact-staging-gap", "0.030",
                "--contact-staging-step", "0.005",
                "--contact-probe-step", "0.002",
                "--contact-probe-max-travel", "0.065",
                "--contact-distance-offset",
                str(float(contact_distance_offset_m)),
            ])
        return command

    def _read_pick_result(self, path, expected_count, label,
                          allow_partial=False):
        try:
            with open(path, "r") as handle:
                payload = json.load(handle)
        except (IOError, OSError, ValueError) as exc:
            raise RuntimeError("无法读取%s抓取结果：%s" % (label, exc))
        completed_ids = payload.get("completed_ids")
        if not isinstance(completed_ids, list):
            raise RuntimeError("%s抓取结果缺少 completed_ids" % label)
        completed_ids = [int(item_id) for item_id in completed_ids]
        count_valid = len(completed_ids) <= int(expected_count) \
            if allow_partial else len(completed_ids) == int(expected_count)
        if (not count_valid or len(set(completed_ids)) != len(completed_ids)
                or any(item_id not in PICK_CANDIDATE_IDS
                       for item_id in completed_ids)):
            raise RuntimeError(
                "%s抓取库存异常：%s %d 个，实际 %s"
                % (label, "最多" if allow_partial else "期望",
                   int(expected_count), completed_ids))
        return completed_ids

    @staticmethod
    def _remove_files(paths):
        for path in paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _untagged_command(self, count, search_before_pick=False):
        command = [
            self.python3, UNTAGGED_PICK_SCRIPT,
            "--run-chassis-sequence",
            "--sequence", self._sequence_text(UNTAGGED_PICK_IDS),
            "--max-targets", str(int(count)),
            "--allow-partial",
            "--result-file", self.untagged_result_file,
            "--confidence", "0.5",
            "--config", UNTAGGED_CONFIG_FILE,
            "--preset-file", UNTAGGED_PRESET_FILE,
        ]
        # 正式 A 点移动搜索必须始终显示识别窗口；窗口直接显示后续
        # 慢速对齐的抓取 ROI，发现阶段本身按全画面统计目标种类。
        if PICK_DEBUG_VIEW or search_before_pick:
            command.append("--show-rgb")
        if search_before_pick:
            command.extend([
                "--search-before-chassis",
                "--search-ready-file", self.untagged_search_ready_file,
                "--search-enable-file", self.untagged_search_enable_file,
                "--search-trigger-file", self.untagged_search_trigger_file,
                "--search-release-file", self.untagged_search_release_file,
                "--search-stable-frames",
                str(int(UNTAGGED_SEARCH_STABLE_FRAMES)),
                "--search-poll-hz", str(float(UNTAGGED_SEARCH_POLL_HZ)),
            ])
        return command

    def _job_failure(self, job_name, code, command):
        log_dir = getattr(self.supervisor, "log_dir", None)
        log_path = os.path.join(log_dir, "%s.log" % job_name) \
            if log_dir else None
        detail = None
        if log_path and os.path.isfile(log_path):
            try:
                with open(log_path, "rb") as handle:
                    lines = handle.read().decode(
                        "utf-8", "replace").splitlines()
                # block_pick_main 最后一行通常只是二次包装的
                # "Arm child exited"，优先找更早的真实 RuntimeError。
                for line in reversed(lines):
                    text = line.strip()
                    if "RuntimeError:" in text:
                        detail = text.split("RuntimeError:", 1)[1].strip()
                        break
                    if "DELIVERY_TIMEOUT " in text:
                        detail = text.split("DELIVERY_TIMEOUT ", 1)[1].strip()
                        break
                if detail is None:
                    for line in reversed(lines):
                        text = line.strip()
                        if (text and "Arm child exited with status" not in text
                                and not text.startswith("Traceback")):
                            detail = text
                            break
            except (IOError, OSError):
                pass
        labels = {
            "pick_untagged": "无Tag抓取",
            "delivery": "投递",
        }
        label = labels.get(job_name, job_name)
        message = "%s子进程退出码%d" % (label, int(code))
        if detail:
            message += "：%s" % detail
        if log_path:
            message += "；日志 %s" % log_path
        if not detail and not log_path:
            message += "；命令 %s" % " ".join(command)
        return RuntimeError(message)

    def _run(self, kind, payload):
        success = False
        error = None
        result_items = []
        try:
            if kind == "tag":
                command = self._tag_command(payload)
            elif kind in ("untagged", "untagged_search"):
                self.supervisor.start_astra()
                command = self._untagged_command(
                    payload, search_before_pick=(kind == "untagged_search"))
            elif kind == "delivery":
                source, item_ids, contact_distance_offset_m = payload
                result_items = [int(item_id) for item_id in item_ids]
                with self.lock:
                    self.result_items = list(result_items)
                command = self._delivery_command(
                    source, result_items, contact_distance_offset_m)
            else:
                raise ValueError("未知抓取类型：%s" % kind)
            job_name = "delivery" if kind == "delivery" else \
                "pick_%s" % ("untagged" if kind == "untagged_search" else kind)
            code = self.supervisor.run_job(job_name, command)
            if code != 0:
                raise self._job_failure(job_name, code, command)
            if kind == "tag":
                result_items = self._read_pick_result(
                    self.tag_result_file, payload, "有 Tag",
                    allow_partial=True)
            elif kind in ("untagged", "untagged_search"):
                result_items = self._read_pick_result(
                    self.untagged_result_file, payload, "无 Tag",
                    allow_partial=True)
            success = True
        except Exception as exc:
            error = exc
        finally:
            try:
                if kind == "tag":
                    self.supervisor.stop_tag_stack()
                    self.supervisor.stop_astra()
                    if not self.keep_arm_after_tag:
                        self.supervisor.stop_arm_common()
                elif kind in ("untagged", "untagged_search"):
                    self.supervisor.stop_astra()
                    if not self.keep_arm_after_untagged:
                        self.supervisor.stop_arm_common()
            except Exception as exc:
                success = False
                if error is None:
                    error = exc
            with self.lock:
                self.result = success
                self.result_items = result_items
                self.error = error

    def start(self, kind, count):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError("已有抓取流程正在运行")
            self.kind = str(kind)
            maximum = len(UNTAGGED_PICK_IDS) \
                if self.kind in ("untagged", "untagged_search") \
                else len(PICK_CANDIDATE_IDS)
            if not 1 <= int(count) <= maximum:
                raise ValueError("抓取数量必须在 1 到 %d 之间" % maximum)
            if self.kind in ("untagged", "untagged_search"):
                # 必须在后台启动 Astra 之前同步清除旧握手文件，避免主
                # 状态机把上一次 ready 当成本次就绪并过早写 enable。
                self._remove_files((
                    self.untagged_result_file,
                    self.untagged_search_ready_file,
                    self.untagged_search_enable_file,
                    self.untagged_search_trigger_file,
                    self.untagged_search_release_file,
                ))
            self.result = None
            self.result_items = []
            self.error = None
            self.thread = threading.Thread(
                target=self._run, args=(self.kind, int(count)))
            self.thread.daemon = True
            self.thread.start()

    def start_untagged_search(self, count):
        self.start("untagged_search", count)

    def untagged_search_ready(self):
        return os.path.isfile(self.untagged_search_ready_file)

    def untagged_search_triggered(self):
        return os.path.isfile(self.untagged_search_trigger_file)

    def enable_untagged_search(self):
        with open(self.untagged_search_enable_file, "w") as handle:
            handle.write("enable\n")

    def release_untagged_search(self):
        with open(self.untagged_search_release_file, "w") as handle:
            handle.write("release\n")

    def start_delivery(self, source, item_ids,
                       contact_distance_offset_m=0.0):
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
            self._remove_files((self.delivery_release_ready_file,))
            self.kind = "delivery"
            self.delivery_source = source
            self.result = None
            self.result_items = []
            self.error = None
            self.thread = threading.Thread(
                target=self._run,
                args=(self.kind, (
                    source, item_ids, float(contact_distance_offset_m))))
            self.thread.daemon = True
            self.thread.start()

    def poll(self):
        with self.lock:
            if (self.kind == "delivery"
                    and self.delivery_source in ("tag", "untagged")
                    and self.result is None
                    and os.path.isfile(self.delivery_release_ready_file)):
                return True, None
            if self.result is None:
                return None, None
            return bool(self.result), self.error

    def arm_job_active(self):
        with self.lock:
            return self.thread is not None and self.thread.is_alive()

    def completed_items(self):
        with self.lock:
            return list(self.result_items)

    def join(self, timeout=1.0):
        thread = self.thread
        if thread is not None:
            thread.join(timeout)
