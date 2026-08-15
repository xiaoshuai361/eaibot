#!/usr/bin/env python3
# coding=utf-8
"""九路口比赛状态机。"""
import os
import shutil
import threading
import time

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist

from ..algorithms.building_delivery import (
    estimate_building_distance_mm,
    limited_building_offset_mm,
    load_building_calibration,
    require_building_target,
)
from ..algorithms.vision import *  # noqa: F401,F403
from ..config import *  # noqa: F401,F403
from ..control.runtime import CameraReader, PID

try:
    from ..algorithms.traffic_light import (
        TrafficLightDetector,
        configure_traffic_camera,
        draw_traffic_light,
        set_capture_resolution,
        update_green_hits,
    )
    TRAFFIC_LIGHT_MODULE_AVAILABLE = True
except ImportError as traffic_light_import_error:
    TRAFFIC_LIGHT_MODULE_AVAILABLE = False
    TRAFFIC_LIGHT_IMPORT_ERROR = traffic_light_import_error
    TrafficLightDetector = None

    def configure_traffic_camera(_camera_index):
        return False

    def set_capture_resolution(capture, width, height):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    def draw_traffic_light(frame, _detections, _color, _green_hits,
                           _green_required):
        return frame

    def update_green_hits(_detections, _current_hits, _required_hits):
        return 0, False, None


def initial_competition_position(enable_tag_pick, start_untagged_aligned):
    if start_untagged_aligned:
        index = UNTAGGED_TRIGGER_INTERSECTION
        return index, TASK_TURN_COMMANDS[index], "A_PICK_PREPARE"
    index = 0
    state = "B_PICK_PREPARE" if enable_tag_pick else "FOLLOW"
    return index, TASK_TURN_COMMANDS[index], state


class LaneFollower(object):
    def __init__(self, grasp_coordinator=None, process_supervisor=None,
                 enable_tag_pick=False, tag_pick_count=1,
                 enable_tag_delivery=False,
                 enable_untagged_pick=False, untagged_pick_count=1,
                 enable_untagged_delivery=False,
                 start_untagged_aligned=False):
        rospy.init_node("line_cy_task", anonymous=True)
        self.grasp_coordinator = grasp_coordinator
        self.process_supervisor = process_supervisor
        self.enable_tag_pick = bool(enable_tag_pick)
        self.tag_pick_count = int(tag_pick_count)
        self.enable_tag_delivery = bool(
            enable_tag_delivery and enable_tag_pick)
        self.enable_untagged_pick = bool(enable_untagged_pick)
        self.untagged_pick_count = int(untagged_pick_count)
        self.enable_untagged_delivery = bool(
            enable_untagged_delivery and enable_untagged_pick)
        self.start_untagged_aligned = bool(start_untagged_aligned)
        if self.start_untagged_aligned and not self.enable_untagged_pick:
            raise ValueError(
                "start_untagged_aligned requires enable_untagged_pick")
        self.tag_pick_completed = (
            self.start_untagged_aligned or not self.enable_tag_pick)
        self.untagged_pick_completed = not self.enable_untagged_pick
        self.active_pick_kind = None
        self.tag_inventory = []
        self.untagged_inventory = []
        self.tag_delivery_failed_ids = set()
        self.untagged_delivery_failed_ids = set()
        self.active_delivery_source = None
        self.active_delivery_id = None
        self.delivery_arm_wait_reported = False
        self.untagged_search_started = False
        self.untagged_search_enabled = False
        self.untagged_forward_started_at = None
        self.tag_pick_first_maneuver = False
        self.untagged_pick_next_maneuver = False
        self.pick_recover_hits = 0
        self.velocity_owner = "line"
        self.camera_index = int(rospy.get_param("~camera_index", CAMERA_INDEX))
        self.process_width = int(rospy.get_param("~process_width", PROCESS_WIDTH))
        self.dry_run = bool(rospy.get_param("~dry_run", DRY_RUN))
        self.debug_view = bool(rospy.get_param("~debug_view", DEBUG_VIEW))
        requested_traffic_light = bool(rospy.get_param(
            "~traffic_light_enabled", TRAFFIC_LIGHT_ENABLED
        ))
        self.traffic_light_enabled = requested_traffic_light
        if requested_traffic_light and not TRAFFIC_LIGHT_MODULE_AVAILABLE:
            message = "未找到 traffic_light_vision，禁止绕过红绿灯：%s" % \
                TRAFFIC_LIGHT_IMPORT_ERROR
            rospy.logerr("line_cy_task %s", message)
            rospy.signal_shutdown(message)
        self.traffic_light_camera_index = int(rospy.get_param(
            "~traffic_light_camera_index", TRAFFIC_LIGHT_CAMERA_INDEX
        ))
        self.traffic_light_model_path = str(rospy.get_param(
            "~traffic_light_model_path", TRAFFIC_LIGHT_MODEL_PATH
        ))
        self.traffic_light_confidence = clamp(float(rospy.get_param(
            "~traffic_light_confidence", TRAFFIC_LIGHT_CONFIDENCE
        )), 0.01, 1.0)
        self.traffic_green_stable_frames = max(1, int(rospy.get_param(
            "~traffic_green_stable_frames", TRAFFIC_GREEN_STABLE_FRAMES
        )))
        self.turn_entry_time = max(0.0, float(rospy.get_param(
            "~turn_entry_time", TURN_ENTRY_TIME
        )))
        self.turn_speed = max(0.0, float(rospy.get_param(
            "~turn_speed", TURN_SPEED
        )))
        self.turn_angular = clamp(abs(float(rospy.get_param(
            "~turn_angular", TURN_ANGULAR
        ))), 0.01, 1.0)
        self.turn_time = max(0.1, float(rospy.get_param(
            "~turn_time", TURN_TIME
        )))
        self.tag_pick_first_entry_time = max(0.0, float(rospy.get_param(
            "~tag_pick_first_entry_time", TAG_PICK_FIRST_ENTRY_TIME
        )))
        self.tag_pick_first_turn_time = max(0.1, float(rospy.get_param(
            "~tag_pick_first_turn_time", TAG_PICK_FIRST_TURN_TIME
        )))
        self.a_pick_third_right_entry_time = max(0.0, float(rospy.get_param(
            "~a_pick_third_right_entry_time", A_PICK_THIRD_RIGHT_ENTRY_TIME
        )))
        self.a_pick_third_right_turn_time = max(0.1, float(rospy.get_param(
            "~a_pick_third_right_turn_time", A_PICK_THIRD_RIGHT_TURN_TIME
        )))
        self.untagged_search_forward_time = max(0.0, float(rospy.get_param(
            "~untagged_search_forward_time", UNTAGGED_SEARCH_FORWARD_TIME
        )))
        self.untagged_search_speed = max(0.0, float(rospy.get_param(
            "~untagged_search_speed", UNTAGGED_SEARCH_SPEED
        )))
        self.untagged_pick_next_entry_time = max(0.0, float(rospy.get_param(
            "~untagged_pick_next_entry_time", UNTAGGED_PICK_NEXT_ENTRY_TIME
        )))
        self.untagged_pick_next_turn_time = max(0.1, float(rospy.get_param(
            "~untagged_pick_next_turn_time", UNTAGGED_PICK_NEXT_TURN_TIME
        )))
        self.final_exit_time = max(0.0, float(rospy.get_param(
            "~final_exit_time", FINAL_EXIT_TIME
        )))
        self.yolo_enabled = bool(rospy.get_param("~yolo_enabled", YOLO_ENABLED))
        self.yolo_requested = self.yolo_enabled
        self.yolo_stop_enabled = bool(rospy.get_param(
            "~yolo_stop_enabled", YOLO_STOP_ENABLED
        ))
        self.yolo_debug_view = bool(rospy.get_param(
            "~yolo_debug_view", self.debug_view
        ))
        self.yolo_camera_index = int(rospy.get_param(
            "~yolo_camera_index", YOLO_CAMERA_INDEX
        ))
        legacy_yolo_model_path = str(rospy.get_param(
            "~yolo_model_path", YOLO_STREET_MODEL_PATH
        ))
        self.yolo_street_model_path = str(rospy.get_param(
            "~yolo_street_model_path", legacy_yolo_model_path
        ))
        self.yolo_building_model_path = str(rospy.get_param(
            "~yolo_building_model_path", YOLO_BUILDING_MODEL_PATH
        ))
        self.yolo_model_path = self.yolo_street_model_path
        self.yolo_frame_interval = max(1, int(rospy.get_param(
            "~yolo_frame_interval", YOLO_FRAME_INTERVAL
        )))
        self.yolo_people_stable_frames = max(1, int(rospy.get_param(
            "~yolo_people_stable_frames", YOLO_PEOPLE_STABLE_FRAMES
        )))
        self.yolo_trash_stable_frames = max(1, int(rospy.get_param(
            "~yolo_trash_stable_frames", YOLO_TRASH_STABLE_FRAMES
        )))
        self.yolo_building_stable_frames = max(1, int(rospy.get_param(
            "~yolo_building_stable_frames", YOLO_BUILDING_STABLE_FRAMES
        )))
        self.yolo_confidence = clamp(float(rospy.get_param(
            "~yolo_confidence", YOLO_CONFIDENCE
        )), 0.001, 1.0)
        self.yolo_trash_confidence = YOLO_TRASH_CONFIDENCE
        self.yolo_building_confidence = YOLO_BUILDING_CONFIDENCE
        self.yolo_center_band_ratio = clamp(float(rospy.get_param(
            "~yolo_center_band_ratio", YOLO_CENTER_BAND_RATIO
        )), 0.01, 1.0)
        self.yolo_image_size = max(32, int(rospy.get_param(
            "~yolo_image_size", YOLO_IMAGE_SIZE
        )))
        self.yolo_nms_threshold = clamp(float(rospy.get_param(
            "~yolo_nms_threshold", YOLO_NMS_THRESHOLD
        )), 0.0, 1.0)
        legacy_class_names = rospy.get_param("~yolo_class_names", None)
        street_class_names = rospy.get_param(
            "~yolo_street_class_names", legacy_class_names
        )
        building_class_names = rospy.get_param(
            "~yolo_building_class_names", None
        )
        self.yolo_street_class_names = self._normalize_ros_class_names(
            street_class_names, YOLO_STREET_CLASS_NAMES
        )
        self.yolo_building_class_names = self._normalize_ros_class_names(
            building_class_names, YOLO_BUILDING_CLASS_NAMES
        )
        self.yolo_class_names = self.yolo_street_class_names
        self.yolo_save_dir = str(rospy.get_param(
            "~yolo_save_dir", YOLO_SAVE_DIR
        ))
        self.yolo_stop_time = max(0.0, float(rospy.get_param(
            "~yolo_stop_time", YOLO_STOP_TIME
        )))
        self.yolo_event_ignore_time = max(0.0, float(rospy.get_param(
            "~yolo_event_ignore_time", YOLO_EVENT_IGNORE_TIME
        )))
        self.task_index, self.turn_cmd, initial_state = \
            initial_competition_position(
                self.enable_tag_pick, self.start_untagged_aligned)
        rospy.loginfo(
            "line_cy_task route=%s entry=%.2f speed=%.2f angular=%.2f "
            "turn_time=%.2f final_exit=%.2f",
            ",".join(TASK_TURN_COMMANDS),
            self.turn_entry_time, self.turn_speed,
            self.turn_angular, self.turn_time, self.final_exit_time,
        )
        rospy.loginfo(
            "line_cy_task intersection %d/%d command=%s",
            self.task_index + 1, len(TASK_TURN_COMMANDS), self.turn_cmd,
        )
        if self.enable_untagged_pick:
            rospy.loginfo(
                "line_cy_task A点前第3右转专用时序："
                "前进=%.2f秒 右转=%.2f秒",
                self.a_pick_third_right_entry_time,
                self.a_pick_third_right_turn_time,
            )
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.vision = BinaryVision()
        self.lanes = LaneDetector(
            fill_width=FILL_WIDTH_PIXELS,
            left_fill_width=LEFT_FILL_WIDTH_PIXELS,
            right_fill_width=RIGHT_FILL_WIDTH_PIXELS,
        )
        self.crosswalk = CrosswalkDetector()
        self.camera = CameraReader(self.camera_index)
        self.yolo_camera = None
        self.traffic_camera = None
        self.traffic_camera_owned = False
        self.traffic_camera_configured = False
        self.traffic_detector = None
        self.traffic_green_hits = 0
        self.traffic_last_color = None
        self.traffic_retry_after = 0.0
        self.yolo_detector = None
        self.yolo_counter = 0
        self.yolo_lock = threading.Lock()
        self.yolo_switch_lock = threading.Lock()
        self.yolo_thread = None
        self.yolo_running = False
        self.yolo_worker_active = False
        self.yolo_latest_seq = 0
        self.yolo_read_seq = 0
        self.yolo_latest_detections = []
        self.yolo_latest_frame = None
        self.yolo_ready = False
        self.yolo_active_profile = None
        self.yolo_stop_detection = None
        self.yolo_stop_event = None
        self.yolo_stop_reported = False
        self.yolo_stop_report_seq = 0
        self.building_delivery_calibration = None
        self.yolo_segment_key = None
        self.yolo_segment_start_seq = 0
        self.yolo_accept_after = 0.0
        self.task_ledger = YoloTaskLedger()
        self.pid = PID(KP, KD, MAX_ANGULAR)
        self.lane_width = LANE_WIDTH_PIXELS if LANE_WIDTH_PIXELS > 0 else PROCESS_WIDTH * DEFAULT_LANE_WIDTH_RATIO
        self.bridge = DualLineBridge(self.lane_width, fill_width=FILL_WIDTH_PIXELS)
        self.state = initial_state
        self.state_started = rospy.get_time()
        self.stop_hits = self.lost_hits = self.align_hits = 0
        self.wait_recover_hits = 0
        self.clear_hits = self.exit_hits = 0
        self.entry_cleared = False
        self.maneuver_phase = "NONE"
        self.maneuver_phase_started = self.state_started
        self.entry_accept_after = 0.0
        self.align_lock = None
        self.align_last_angle = None
        self.last_angular = 0.0
        self.last_command_angular = 0.0
        self.last_control_target = None
        self.last_observation = None
        self.last_crosswalk = CrosswalkResult()
        self.last_binary = None
        self.cleaned = False
        if not self.camera.cap.isOpened():
            rospy.signal_shutdown("cannot open lane camera")
        self._prepare_yolo_save_dir()
        if self.enable_untagged_delivery:
            self._load_building_delivery_calibration()
        if (self.yolo_enabled and not self.enable_tag_pick
                and not self.start_untagged_aligned):
            self._init_yolo()
        if self.start_untagged_aligned:
            rospy.logwarn(
                "line_cy_task 调试入口：假定第3个右转已完成且车身摆正，"
                "从 A_PICK_PREPARE 开始；当前路口4/9 command=%s",
                self.turn_cmd)
        rospy.on_shutdown(self.cleanup)

    def _normalize_ros_class_names(self, value, defaults):
        if not value:
            return tuple(defaults)
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",")
                     if item.strip()]
        return tuple(value)

    def _load_building_delivery_calibration(self):
        path = str(rospy.get_param(
            "~building_delivery_calibration_file",
            BUILDING_DELIVERY_CALIBRATION_FILE,
        ))
        calibration = load_building_calibration(
            path,
            expected_width=YOLO_FRAME_WIDTH,
            expected_height=YOLO_FRAME_HEIGHT,
            expected_model=self.yolo_building_model_path,
        )
        for class_name, item_id in \
                UNTAGGED_DELIVERY_ID_BY_BUILDING_CLASS.items():
            require_building_target(calibration, item_id, class_name)
        self.building_delivery_calibration = calibration
        rospy.loginfo("line_cy_task 已加载楼宇投递视觉标定：%s", path)

    def _yolo_profile_settings(self, profile):
        if profile == "building":
            return (
                self.yolo_building_model_path,
                self.yolo_building_class_names,
                self.yolo_building_confidence,
            )
        return (
            self.yolo_street_model_path,
            self.yolo_street_class_names,
            min(self.yolo_confidence, self.yolo_trash_confidence),
        )

    def _create_yolo_detector(self, profile):
        model_path, class_names, confidence = \
            self._yolo_profile_settings(profile)
        return YoloObstacleDetector(
            model_path,
            confidence=confidence,
            center_band_ratio=self.yolo_center_band_ratio,
            image_size=self.yolo_image_size,
            nms_threshold=self.yolo_nms_threshold,
            class_names=class_names,
        )

    def _init_yolo(self, profile=None):
        initial_profile = profile or yolo_model_profile(self.task_index)
        _, initial_class_names, _ = self._yolo_profile_settings(initial_profile)
        try:
            self.yolo_detector = self._create_yolo_detector(initial_profile)
            rospy.loginfo(
                "line_cy_task 正在加载并预热%s模型：%s",
                initial_profile,
                self.yolo_detector.model_path,
            )
            self.publish(0, 0)
            # 启动时完成加载，避免进入识别路段后再承担首次加载延迟。
            self.yolo_detector.load()
            self.yolo_active_profile = initial_profile
            self.yolo_model_path = self.yolo_detector.model_path
            self.yolo_class_names = tuple(initial_class_names)
        except Exception as exc:
            rospy.logwarn("line_cy_task YOLO disabled: %s", exc)
            self.yolo_enabled = False
            self.yolo_detector = None
            self.yolo_active_profile = None
            self.yolo_ready = False
            return
        traffic_camera_index = getattr(
            self, "traffic_light_camera_index", TRAFFIC_LIGHT_CAMERA_INDEX
        )
        if self.yolo_camera_index == traffic_camera_index:
            try:
                configure_traffic_camera(self.yolo_camera_index)
                self.traffic_camera_configured = True
            except Exception as exc:
                self.traffic_camera_configured = False
                rospy.logwarn(
                    "line_cy_task 摄像头 %d 参数设置失败：%s",
                    self.yolo_camera_index, exc,
                )
        self.yolo_camera = CameraReader(
            self.yolo_camera_index,
            YOLO_FRAME_WIDTH,
            YOLO_FRAME_HEIGHT,
        )
        if not self.yolo_camera.cap.isOpened():
            rospy.logwarn(
                "line_cy_task YOLO camera %d cannot open, YOLO disabled",
                self.yolo_camera_index,
            )
            self.yolo_enabled = False
            self.yolo_ready = False
            self.yolo_camera.release()
            self.yolo_camera = None
            return
        rospy.loginfo("line_cy_task waiting for YOLO first frame warmup")
        ok, frame = self.yolo_camera.read(3.0)
        if not ok:
            rospy.logwarn(
                "line_cy_task YOLO camera %d has no warmup frame, YOLO disabled",
                self.yolo_camera_index,
            )
            self.yolo_enabled = False
            self.yolo_ready = False
            self.yolo_camera.release()
            self.yolo_camera = None
            return
        try:
            detections = self.yolo_detector.detect(frame)
        except Exception as exc:
            rospy.logwarn("line_cy_task YOLO warmup inference failed: %s", exc)
            self.yolo_enabled = False
            self.yolo_ready = False
            self.yolo_camera.release()
            self.yolo_camera = None
            return
        self._store_yolo_result(frame, detections)
        self.yolo_ready = True
        rospy.loginfo(
            "line_cy_task YOLO %s模型加载和预热完成，"
            "enabled camera=%d backend=%s model=%s interval=%d imgsz=%d "
            "people_stable=%d trash_stable=%d building_stable=%d "
            "people_conf=%.2f trash_conf=%.2f "
            "building_conf=%.2f nms=%.2f "
            "stop=%s debug=%s",
            self.yolo_active_profile,
            self.yolo_camera_index, self.yolo_detector.backend_name,
            self.yolo_detector.model_path, self.yolo_frame_interval,
            self.yolo_image_size, self.yolo_people_stable_frames,
            self.yolo_trash_stable_frames,
            self.yolo_building_stable_frames,
            self.yolo_confidence,
            self.yolo_trash_confidence, self.yolo_building_confidence,
            self.yolo_nms_threshold,
            self.yolo_stop_enabled,
            self.yolo_debug_view,
        )
        self.yolo_running = True
        self.yolo_thread = threading.Thread(target=self._yolo_loop)
        self.yolo_thread.daemon = True
        self.yolo_thread.start()

    def _shutdown_yolo(self):
        """释放任务 YOLO 和共享 video2，不改变用户的启用配置。"""
        self.yolo_running = False
        if self.yolo_thread is not None:
            self.yolo_thread.join(2.0)
            self.yolo_thread = None
        with self.yolo_switch_lock:
            if self.yolo_detector is not None:
                self.yolo_detector.close()
                self.yolo_detector = None
        if self.yolo_camera is not None:
            self.yolo_camera.release()
            self.yolo_camera = None
        self.yolo_ready = False
        self.yolo_active_profile = None
        self._clear_yolo_cache()
        try:
            cv2.destroyWindow(YOLO_WINDOW_NAME)
        except cv2.error:
            pass

    def _resume_yolo(self, profile=None):
        if not self.yolo_requested:
            return True
        self.yolo_enabled = True
        self._init_yolo(profile)
        return bool(self.yolo_ready)

    def _clear_yolo_cache(self):
        with self.yolo_lock:
            self.yolo_latest_detections = []
            self.yolo_latest_frame = None
            self.yolo_read_seq = self.yolo_latest_seq
        self.yolo_segment_key = None
        self.yolo_segment_start_seq = self._latest_yolo_seq()
        self.yolo_counter = 0

    def _switch_yolo_profile_if_needed(self):
        """在第三个右转完成后释放街道模型并加载楼宇模型。"""
        if not getattr(self, "yolo_enabled", False):
            return True
        desired_profile = yolo_model_profile(self.task_index)
        if desired_profile == getattr(self, "yolo_active_profile", None):
            return True
        if self.yolo_camera is None:
            rospy.logwarn("line_cy_task YOLO 摄像头不可用，无法切换模型")
            self.yolo_enabled = False
            self.yolo_ready = False
            return False

        model_path, class_names, _ = self._yolo_profile_settings(
            desired_profile
        )
        rospy.loginfo(
            "line_cy_task 到达物资点切换位置，停车并切换为%s模型：%s",
            desired_profile, model_path,
        )
        self.publish(0, 0)
        self.yolo_ready = False
        detector = None
        try:
            with self.yolo_switch_lock:
                old_detector = self.yolo_detector
                self.yolo_detector = None
                if old_detector is not None:
                    old_detector.close()
                detector = self._create_yolo_detector(desired_profile)
                detector.load()
                ok, frame = self.yolo_camera.read(3.0)
                if not ok:
                    raise RuntimeError("模型切换后摄像头没有新画面")
                detections = detector.detect(frame)
                self.yolo_detector = detector
        except Exception as exc:
            if detector is not None:
                detector.close()
            self.yolo_detector = None
            self.yolo_active_profile = None
            self.yolo_enabled = False
            self.yolo_ready = False
            rospy.logwarn("line_cy_task YOLO 模型切换失败，已关闭任务识别：%s",
                          exc)
            return False

        self.yolo_active_profile = desired_profile
        self.yolo_model_path = self.yolo_detector.model_path
        self.yolo_class_names = tuple(class_names)
        self._clear_yolo_cache()
        self._store_yolo_result(frame, detections)
        self.yolo_ready = True
        rospy.loginfo(
            "line_cy_task YOLO 已切换为%s模型并完成预热：%s",
            desired_profile, self.yolo_detector.model_path,
        )
        return True

    def _resize(self, frame):
        height, width = frame.shape[:2]
        if self.process_width <= 0 or width <= self.process_width:
            return frame
        scale = float(self.process_width) / width
        return cv2.resize(frame, (self.process_width, int(height * scale)), interpolation=cv2.INTER_AREA)

    def publish(self, linear, angular, force=False):
        if self.velocity_owner == "grasp" and not force:
            return False
        angular_limit = MAX_ANGULAR
        if (getattr(self, "state", None) == "MANEUVER"
                and getattr(self, "maneuver_phase", None) == "TURN"):
            angular_limit = max(MAX_ANGULAR, abs(self.turn_angular))
        angular = clamp(float(angular), -angular_limit, angular_limit)
        self.last_command_angular = angular
        command = Twist()
        command.linear.x = float(linear)
        command.linear.y = command.linear.z = 0.0
        command.angular.x = command.angular.y = 0.0
        command.angular.z = angular
        if motion_enabled(self.dry_run):
            self.pub.publish(command)
        return True

    def _control(self, center_x, width, speed, bias_pixels=0.0):
        target_x = control_target_x(center_x, bias_pixels)
        self.last_control_target = target_x
        deviation = target_x - width * 0.5
        kp, kd, _ = pd_gains(deviation)
        raw = self.pid.update(deviation, kp, kd)
        direction_scale = FOLLOW_LEFT_ANGULAR_SCALE \
            if raw > 0.0 else FOLLOW_RIGHT_ANGULAR_SCALE
        raw *= direction_scale
        angular = ANGULAR_SMOOTH * self.last_angular + (1.0 - ANGULAR_SMOOTH) * raw
        self.last_angular = angular
        self.publish(speed, angular)

    def _set_maneuver_phase(self, phase, now=None):
        if phase == self.maneuver_phase:
            return
        rospy.loginfo("line_cy_task maneuver phase: %s -> %s",
                      self.maneuver_phase, phase)
        self.maneuver_phase = phase
        self.maneuver_phase_started = rospy.get_time() if now is None else float(now)
        self.pid.reset()
        self.last_angular = 0.0
        self.last_control_target = None

    def _run_timed_turn_phase(self, now):
        elapsed = float(now) - self.maneuver_phase_started
        entry_time = self.turn_entry_time
        turn_time = self.turn_time
        if (getattr(self, "tag_pick_first_maneuver", False)
                and self.task_index == 0 and self.turn_cmd == "right"):
            entry_time = self.tag_pick_first_entry_time
            turn_time = self.tag_pick_first_turn_time
        elif (self.task_index + 1 == UNTAGGED_TRIGGER_INTERSECTION
              and self.turn_cmd == "right"
              and getattr(self, "enable_untagged_pick", False)
              and not getattr(self, "untagged_pick_completed", True)):
            entry_time = self.a_pick_third_right_entry_time
            turn_time = self.a_pick_third_right_turn_time
        elif (getattr(self, "untagged_pick_next_maneuver", False)
              and self.task_index == UNTAGGED_TRIGGER_INTERSECTION
              and self.turn_cmd == "left"):
            entry_time = self.untagged_pick_next_entry_time
            turn_time = self.untagged_pick_next_turn_time
        next_phase = turn_phase_next(
            self.maneuver_phase, elapsed,
            entry_time, turn_time,
        )
        if next_phase is not None:
            self._set_maneuver_phase(next_phase, now)

        if self.maneuver_phase in ("ENTRY", "EXIT_STRAIGHT"):
            self.publish(self.turn_speed, 0.0)
        elif self.maneuver_phase == "TURN":
            linear, angular = fixed_turn_command(
                self.turn_cmd, self.turn_speed, self.turn_angular
            )
            self.publish(linear, angular)
        else:
            self.publish(0, 0)

    def _entry_ready_state(self):
        enabled = getattr(self, "traffic_light_enabled", TRAFFIC_LIGHT_ENABLED)
        return "TRAFFIC_WAIT" if enabled else "MANEUVER"

    def _close_traffic_light(self):
        detector = getattr(self, "traffic_detector", None)
        if detector is not None:
            detector.close()
        self.traffic_detector = None
        camera = getattr(self, "traffic_camera", None)
        if camera is not None and getattr(self, "traffic_camera_owned", False):
            camera.release()
        self.traffic_camera = None
        self.traffic_camera_owned = False
        self.traffic_green_hits = 0
        self.traffic_last_color = None
        try:
            cv2.destroyWindow(TRAFFIC_LIGHT_WINDOW_NAME)
        except cv2.error:
            pass

    def _open_traffic_light(self):
        shared_camera = getattr(self, "yolo_camera", None)
        use_shared = (
            shared_camera is not None
            and shared_camera.cap.isOpened()
            and self.yolo_camera_index == self.traffic_light_camera_index
        )
        if use_shared:
            if not getattr(self, "traffic_camera_configured", False):
                configure_traffic_camera(self.traffic_light_camera_index)
                self.traffic_camera_configured = True
            camera = shared_camera
            camera_owned = False
        else:
            camera = CameraReader(
                self.traffic_light_camera_index,
                TRAFFIC_LIGHT_FRAME_WIDTH,
                TRAFFIC_LIGHT_FRAME_HEIGHT,
            )
            camera_owned = True
            if not camera.cap.isOpened():
                camera.release()
                raise RuntimeError("无法打开红绿灯摄像头 %d" %
                                   self.traffic_light_camera_index)
            try:
                configure_traffic_camera(self.traffic_light_camera_index)
                self.traffic_camera_configured = True
            except Exception:
                camera.release()
                raise
        detector = TrafficLightDetector(
            self.traffic_light_model_path,
            confidence=self.traffic_light_confidence,
        )
        try:
            detector.load()
        except Exception:
            if camera_owned:
                camera.release()
            raise
        self.traffic_camera = camera
        self.traffic_camera_owned = camera_owned
        self.traffic_detector = detector
        rospy.loginfo(
            "line_cy_task 红绿灯模型已在停止线加载：camera=%d model=%s",
            self.traffic_light_camera_index,
            self.traffic_light_model_path,
        )

    def _handle_traffic_light_wait(self, now):
        self.publish(0, 0)
        # 原任务模型可能刚开始一帧推理，等它退出后再独占摄像头和 CPU。
        if getattr(self, "yolo_worker_active", False):
            return
        if self.traffic_detector is None or self.traffic_camera is None:
            if float(now) < self.traffic_retry_after:
                return
            try:
                self._open_traffic_light()
            except Exception as exc:
                self._close_traffic_light()
                self.traffic_retry_after = float(now) + TRAFFIC_LIGHT_RETRY_TIME
                rospy.logwarn("line_cy_task 红绿灯识别启动失败，保持停车：%s", exc)
                return
        ok, frame = self.traffic_camera.read(0.2)
        if not ok:
            return
        try:
            detections = self.traffic_detector.detect(frame)
        except Exception as exc:
            self._close_traffic_light()
            self.traffic_retry_after = float(now) + TRAFFIC_LIGHT_RETRY_TIME
            rospy.logwarn("line_cy_task 红绿灯推理失败，保持停车：%s", exc)
            return
        self.traffic_green_hits, green_ready, color = update_green_hits(
            detections, self.traffic_green_hits,
            self.traffic_green_stable_frames,
        )
        self.traffic_last_color = color
        if self.debug_view:
            try:
                cv2.imshow(
                    TRAFFIC_LIGHT_WINDOW_NAME,
                    draw_traffic_light(
                        frame, detections, color, self.traffic_green_hits,
                        self.traffic_green_stable_frames,
                    ),
                )
                cv2.waitKey(1)
            except cv2.error:
                pass
        if green_ready:
            rospy.loginfo("line_cy_task 连续识别到绿灯，释放模型并进入路口")
            self._set_state("MANEUVER")

    def _lock_entry_alignment(self, now=None, angle=None):
        if angle is None:
            cross = getattr(self, "last_crosswalk", None)
            if cross is None:
                self.align_lock = None
                return False
            if getattr(cross, "candidate", False) and cross.stop_angle is not None:
                angle = cross.stop_angle
            else:
                angle = cross.tracking_angle
        if angle is None:
            self.align_lock = None
            return False
        now = rospy.get_time() if now is None else float(now)
        angle = float(angle)
        magnitude = clamp(abs(angle) * ALIGN_KP,
                          ALIGN_MIN_ANGULAR, ALIGN_MAX_ANGULAR)
        angular = 0.0 if abs(angle) <= ALIGN_TOLERANCE_DEG \
            else (-magnitude if angle > 0.0 else magnitude)
        rotate_time = 0.0 if angular == 0.0 else clamp(
            np.radians(abs(angle)) / max(magnitude, 1e-6)
            * ALIGN_OPEN_LOOP_TIME_SCALE,
            ALIGN_OPEN_LOOP_MIN_TIME,
            ALIGN_OPEN_LOOP_MAX_TIME,
        )
        self.align_lock = {
            "angle": angle,
            "angular": angular,
            "rotate_until": now + rotate_time,
            "settle_until": now + rotate_time + ALIGN_LOCK_SETTLE_TIME,
        }
        rospy.loginfo(
            "line_cy_task lost-bar align angle=%.1f angular=%.2f "
            "rotate=%.2fs settle=%.2fs",
            self.align_lock["angle"], self.align_lock["angular"],
            rotate_time, ALIGN_LOCK_SETTLE_TIME,
        )
        return True

    def _run_locked_entry_alignment(self, now):
        if self.align_lock is None:
            return False, None
        now = float(now)
        if now < self.align_lock["rotate_until"]:
            self.publish(0, self.align_lock["angular"])
            return True, None
        if now < self.align_lock["settle_until"]:
            self.publish(0, 0)
            return True, None
        self.publish(0, 0)
        return True, "MANEUVER"

    def _pick_failed(self, message):
        self.velocity_owner = "line"
        self.publish(0, 0, force=True)
        rospy.logerr("line_cy_task 抓取流程失败，永久停车：%s", message)
        self._set_state("PICK_FAILED")

    def _start_pick(self, kind, count, picking_state):
        if self.grasp_coordinator is None:
            self._pick_failed("未配置抓取协调器")
            return
        try:
            self.active_pick_kind = kind
            self.velocity_owner = "grasp"
            self.grasp_coordinator.start(kind, count)
            self._set_state(picking_state)
        except Exception as exc:
            self._pick_failed(exc)

    def _start_untagged_search(self):
        if self.grasp_coordinator is None:
            self._pick_failed("未配置抓取协调器")
            return
        try:
            self.active_pick_kind = "untagged"
            self.untagged_search_started = True
            self.untagged_forward_started_at = None
            self.grasp_coordinator.start_untagged_search(
                self.untagged_pick_count)
            rospy.loginfo("line_cy_task 正在加载 A 点无 Tag 搜索模型")
        except Exception as exc:
            self._pick_failed(exc)

    def _handle_untagged_search(self, observation, cross, frame_width):
        success, error = self.grasp_coordinator.poll()
        if success is not None:
            self._pick_failed(error or "A 点搜索进程提前退出")
            return
        if not self.untagged_search_enabled:
            forward_time = max(0.0, float(getattr(
                self, "untagged_search_forward_time",
                UNTAGGED_SEARCH_FORWARD_TIME,
            )))
            # 这段要求的是现实世界中完整的直行秒数，不能使用可能受
            # /use_sim_time 或 ROS 时钟跳变影响的 rospy.get_time()。
            now = time.monotonic()
            if self.untagged_forward_started_at is None:
                self.untagged_forward_started_at = now
                rospy.loginfo(
                    "line_cy_task A 点开始普通速度直行，"
                    "按单调时钟计满 %.2f 秒",
                    forward_time,
                )
            elapsed = now - self.untagged_forward_started_at
            if elapsed < forward_time:
                self.publish(FOLLOW_SPEED, 0.0)
                return
            try:
                self.grasp_coordinator.enable_untagged_search()
                self.untagged_search_enabled = True
            except Exception as exc:
                self._pick_failed("无法启动 A 点低速搜索：%s" % exc)
                return
            rospy.loginfo(
                "line_cy_task A 点直行完成：要求 %.2f 秒，实际 %.2f 秒；"
                "切换低速 %.2f 边走边搜索",
                forward_time, elapsed, self.untagged_search_speed)
        if self.grasp_coordinator.untagged_search_triggered():
            self.publish(0, 0, force=True)
            self.velocity_owner = "grasp"
            try:
                self.grasp_coordinator.release_untagged_search()
            except Exception as exc:
                self._pick_failed("无法交接 A 点底盘控制权：%s" % exc)
                return
            self._set_state("A_PICKING")
            rospy.loginfo(
                "line_cy_task A 点右侧稳定检测到无 Tag 物块，"
                "停车并切换到慢速对准抓取")
            return
        # 第 3 个路口的出口已经摆正。A 点搜索阶段不再根据车道线或
        # 第 4 个路口横条停车，只保持零角速度直行，直到搜索子进程
        # 确认足够数量的物块并触发底盘交接。
        self.publish(self.untagged_search_speed, 0.0)

    def _finish_pick(self, kind):
        self.velocity_owner = "line"
        self.publish(0, 0, force=True)
        label = "有 Tag" if kind == "tag" else "无 Tag"
        expected_count = self.tag_pick_count if kind == "tag" \
            else self.untagged_pick_count
        try:
            completed_ids = self.grasp_coordinator.completed_items()
        except Exception as exc:
            self._pick_failed("无法读取%s抓取库存：%s" % (label, exc))
            return
        count_valid = len(completed_ids) <= expected_count
        if not count_valid:
            self._pick_failed(
                "%s抓取库存数量异常：最多 %d，实际 %s"
                % (label, expected_count, completed_ids))
            return
        if kind == "tag":
            self.tag_inventory = list(completed_ids)
            self.tag_pick_completed = True
            profile = "street"
            if len(completed_ids) < expected_count:
                rospy.logwarn(
                    "line_cy_task 有 Tag 抓取部分完成：成功 %d/%d 个；"
                    "按实际库存继续比赛",
                    len(completed_ids), expected_count,
                )
            rospy.loginfo(
                "line_cy_task 有 Tag 物资已入仓，库存 ID=%s",
                self.tag_inventory,
            )
        else:
            self.untagged_inventory = list(completed_ids)
            self.untagged_pick_completed = True
            profile = "building"
            if len(completed_ids) < expected_count:
                rospy.logwarn(
                    "line_cy_task 无 Tag 抓取部分完成：成功 %d/%d 个；"
                    "按实际库存继续比赛",
                    len(completed_ids), expected_count,
                )
            rospy.loginfo(
                "line_cy_task 无 Tag 物资已入仓，库存 ID=%s",
                self.untagged_inventory,
            )
        if not self._resume_yolo(profile):
            self._pick_failed("共享摄像头释放后任务 YOLO 未能恢复")
            return
        self.active_pick_kind = None
        self.pick_recover_hits = 0
        self.bridge.reset(self.lane_width)
        self.stop_hits = 0
        if kind == "tag":
            self.tag_pick_first_maneuver = True
            next_state = self._entry_ready_state()
            self._set_state(next_state)
            rospy.loginfo(
                "line_cy_task B 点抓取完成，先等待绿灯，再按专用时序"
                "直行 %.2f 秒、右转 %.2f 秒",
                self.tag_pick_first_entry_time,
                self.tag_pick_first_turn_time,
            )
        else:
            self.untagged_pick_next_maneuver = True
            next_state = self._entry_ready_state()
            self._set_state(next_state)
            rospy.loginfo(
                "line_cy_task A 点抓取完成，先等待绿灯，再按专用时序"
                "直行 %.2f 秒、左转 %.2f 秒",
                self.untagged_pick_next_entry_time,
                self.untagged_pick_next_turn_time,
            )

    def _delivery_id_for_event(self, event):
        if event is None:
            return None
        if event.kind == "street":
            return TAG_DELIVERY_ID_BY_STREET_CLASS.get(event.class_name)
        if event.kind == "building":
            return UNTAGGED_DELIVERY_ID_BY_BUILDING_CLASS.get(
                event.class_name)
        return None

    def _delivery_context_for_event(self, event):
        if event is None:
            return None
        if event.kind == "street" and self.enable_tag_delivery:
            return (
                "tag", "有 Tag", self.tag_inventory,
                self.tag_delivery_failed_ids,
            )
        if event.kind == "building" and self.enable_untagged_delivery:
            return (
                "untagged", "无 Tag", self.untagged_inventory,
                self.untagged_delivery_failed_ids,
            )
        return None

    def _start_delivery_for_event(self, event):
        context = self._delivery_context_for_event(event)
        if context is None:
            return False
        source, label, inventory, failed_ids = context
        item_id = self._delivery_id_for_event(event)
        if (item_id is None or item_id not in inventory
                or item_id in failed_ids):
            return False
        arm_job_active = getattr(
            self.grasp_coordinator, "arm_job_active", None)
        if callable(arm_job_active) and arm_job_active():
            if not getattr(self, "delivery_arm_wait_reported", False):
                rospy.loginfo(
                    "line_cy_task 上一次投递已关泵但机械臂仍在回idle；"
                    "当前目标处等待归位后再开始下一次投递"
                )
                self.delivery_arm_wait_reported = True
            return None
        if self.grasp_coordinator is None:
            failed_ids.add(item_id)
            rospy.logwarn(
                "line_cy_task %s ID%d 需要投递，但未配置机械臂协调器，继续循迹",
                label, item_id,
            )
            return False
        distance_offset_m = 0.0
        if source == "untagged":
            try:
                entry = require_building_target(
                    self.building_delivery_calibration,
                    item_id, event.class_name)
                _center_ratio, distance_mm = estimate_building_distance_mm(
                    event.detection, entry, event.detection.frame_shape)
                reference_mm = float(entry["reference_distance_mm"])
                raw_offset_mm, limited_offset_mm = limited_building_offset_mm(
                    distance_mm, reference_mm,
                    BUILDING_DELIVERY_OFFSET_MIN_MM,
                    BUILDING_DELIVERY_OFFSET_MAX_MM,
                )
                distance_offset_m = limited_offset_mm * 0.001
                rospy.loginfo(
                    "line_cy_task 无 Tag ID%d 停车估距 %.1fmm，"
                    "示教参考 %.1fmm，原始P修正 %+.1fmm，"
                    "限幅后 %+.1fmm（范围 %+.1f~%+.1fmm）",
                    item_id, distance_mm, reference_mm,
                    raw_offset_mm, limited_offset_mm,
                    BUILDING_DELIVERY_OFFSET_MIN_MM,
                    BUILDING_DELIVERY_OFFSET_MAX_MM,
                )
            except Exception as exc:
                failed_ids.add(item_id)
                rospy.logwarn(
                    "line_cy_task 无 Tag ID%d 楼宇估距失败：%s；不重试",
                    item_id, exc,
                )
                return False
        return self._launch_delivery_process(event, source, label, item_id,
                                             failed_ids, distance_offset_m)

    def _launch_delivery_process(self, event, source, label, item_id,
                                 failed_ids, distance_offset_m=0.0):
        try:
            self.publish(0, 0, force=True)
            self.velocity_owner = "grasp"
            self.active_delivery_source = source
            self.active_delivery_id = item_id
            self.delivery_arm_wait_reported = False
            if source == "untagged":
                self.grasp_coordinator.start_delivery(
                    source, [item_id], distance_offset_m)
            else:
                self.grasp_coordinator.start_delivery(source, [item_id])
            self._set_state("DELIVERING")
            rospy.loginfo(
                "line_cy_task %s识别到%s，开始投递%s ID%d",
                event.area, event.display_name, label, item_id,
            )
            return True
        except Exception as exc:
            self.velocity_owner = "line"
            self.active_delivery_source = None
            self.active_delivery_id = None
            failed_ids.add(item_id)
            self.publish(0, 0, force=True)
            rospy.logwarn(
                "line_cy_task %s ID%d 投递未启动：%s；继续循迹",
                label, item_id, exc,
            )
            return False

    def _finish_delivery(self, success, error=None):
        source = self.active_delivery_source
        item_id = self.active_delivery_id
        if source == "untagged":
            label = "无 Tag"
            inventory = self.untagged_inventory
            failed_ids = self.untagged_delivery_failed_ids
        else:
            label = "有 Tag"
            inventory = self.tag_inventory
            failed_ids = self.tag_delivery_failed_ids
        self.velocity_owner = "line"
        self.publish(0, 0, force=True)
        if success:
            if item_id in inventory:
                inventory.remove(item_id)
            rospy.loginfo(
                "line_cy_task %s ID%d 投递完成，剩余库存=%s",
                label, item_id, inventory,
            )
            arm_no_longer_needed = (
                source == "untagged" and not inventory
            ) or (
                source == "tag" and not inventory
                and not self.enable_untagged_pick
            )
            arm_job_active = getattr(
                self.grasp_coordinator, "arm_job_active", None)
            arm_still_returning = (
                callable(arm_job_active) and arm_job_active())
            if (arm_no_longer_needed and not arm_still_returning
                    and self.process_supervisor is not None):
                self.process_supervisor.stop_arm_common()
        else:
            failed_ids.add(item_id)
            rospy.logwarn(
                "line_cy_task %s ID%d 投递失败：%s；不重试并继续循迹",
                label, item_id, error or "投递子进程返回失败",
            )
        self.active_delivery_source = None
        self.active_delivery_id = None
        self._set_state("FOLLOW")

    def _handle_pick_without_frame(self, now):
        if self.state == "B_PICK_PREPARE":
            self.publish(0, 0, force=True)
            if now - self.state_started >= GRASP_SETTLE_TIME:
                self._start_pick("tag", self.tag_pick_count, "B_PICKING")
            return
        if self.state == "A_PICK_PREPARE":
            # 出口摆正后先保持停车，等 Astra、模型和搜索子进程全部
            # ready。只有 ready 后才启用检测并按默认循迹速度直行。
            self.publish(0, 0, force=True)
            if not self.untagged_search_started:
                self._start_untagged_search()
            if self.state != "A_PICK_PREPARE":
                return
            # 搜索尚未启动时不能读取协调器里的旧任务结果，否则 B 点
            # 抓取的完成状态会被误判成 A 点搜索进程提前退出。
            if not self.untagged_search_started:
                return
            success, error = self.grasp_coordinator.poll()
            if success is not None:
                self._pick_failed(error or "A 点搜索进程提前退出")
            elif self.grasp_coordinator.untagged_search_ready():
                self.velocity_owner = "line"
                self.stop_hits = 0
                self._set_state("A_PICK_SEARCH")
                rospy.loginfo(
                    "line_cy_task A 点模型和识别窗口全部就绪，"
                    "先按默认速度 %.2f 直行 %.2f 秒，"
                    "再以 %.2f 低速搜索",
                    FOLLOW_SPEED,
                    getattr(self, "untagged_search_forward_time",
                            UNTAGGED_SEARCH_FORWARD_TIME),
                    getattr(self, "untagged_search_speed",
                            UNTAGGED_SEARCH_SPEED))
            return
        if self.state == "A_PICK_SEARCH":
            # 已完成第三个路口摆正，搜索阶段不需要循迹图像。独立按
            # 20Hz 刷新速度，确保底盘真正收到完整的快速直行时长。
            self._handle_untagged_search(None, None, None)
            return
        if self.state in ("B_PICKING", "A_PICKING"):
            success, error = self.grasp_coordinator.poll()
            if success is None:
                return
            if not success:
                self._pick_failed(error or "抓取子进程返回失败")
                return
            self._finish_pick(self.active_pick_kind)
            return
        if self.state == "DELIVERING":
            success, error = self.grasp_coordinator.poll()
            if success is not None:
                self._finish_delivery(success, error)
            return
        if self.state == "PICK_FAILED":
            self.publish(0, 0, force=True)

    def _runtime_failure(self):
        if self.process_supervisor is None:
            return None
        failed = self.process_supervisor.check_owned_processes()
        return failed[0] if failed else None

    def _set_state(self, state):
        if state == self.state:
            return
        previous_state = self.state
        if previous_state == "TRAFFIC_WAIT" and state != "TRAFFIC_WAIT":
            self._close_traffic_light()
        rospy.loginfo("line_cy_task state: %s -> %s", previous_state, state)
        self.state = state
        self.state_started = rospy.get_time()
        self.pid.reset()
        self.last_angular = 0.0
        self.last_control_target = None
        self.lost_hits = self.align_hits = 0
        self.align_lock = None
        self.align_last_angle = None
        if state == "EXIT_ALIGN":
            tolerance_deg, align_kp, min_angular, max_angular = \
                self._exit_alignment_parameters()
            if tolerance_deg == A_PICK_EXIT_ALIGN_TOLERANCE_DEG:
                rospy.loginfo(
                    "line_cy_task 第3路口 A 点专用出口精调："
                    "容差=%.1f度 kp=%.3f angular=%.2f~%.2f",
                    tolerance_deg, align_kp, min_angular, max_angular,
                )
        if (state == "FOLLOW"
                and previous_state in ("EXIT_ALIGN", "MANEUVER",
                                       "PICK_RECOVER")):
            self.stop_hits = 0
            self.entry_accept_after = (
                self.state_started + EXIT_ENTRY_IGNORE_TIME
            )
        if state == "FOLLOW" and previous_state in (
                "YOLO_STOP", "DELIVERING"):
            ignore_time = max(0.0, float(getattr(
                self, "yolo_event_ignore_time", YOLO_EVENT_IGNORE_TIME
            )))
            self.yolo_accept_after = self.state_started + ignore_time
            rospy.loginfo(
                "line_cy_task 任务识别保护 %.1f 秒",
                ignore_time,
            )
        if state == "A_PICK_SEARCH":
            self.stop_hits = 0
            self.entry_accept_after = (
                self.state_started + EXIT_ENTRY_IGNORE_TIME
            )
        if state in ("FOLLOW", "MANEUVER", "FINAL_EXIT", "YOLO_STOP",
                     "TRAFFIC_WAIT", "PICK_RECOVER", "DELIVERING",
                     "A_PICK_SEARCH"):
            self.crosswalk.unlock_bar()
        if state == "TRAFFIC_WAIT":
            self.traffic_retry_after = self.state_started
            self.traffic_green_hits = 0
            self.traffic_last_color = None
        if state == "MANEUVER":
            self.entry_cleared = False
            self.clear_hits = self.exit_hits = 0
            self.maneuver_phase = (
                "ENTRY" if maneuver_follow_side(self.turn_cmd) is not None
                else "STRAIGHT"
            )
            self.maneuver_phase_started = self.state_started
        else:
            self.maneuver_phase = "NONE"

    def _complete_intersection(self):
        completed = self.task_index + 1
        rospy.loginfo(
            "line_cy_task intersection %d/%d completed command=%s",
            completed, len(TASK_TURN_COMMANDS), self.turn_cmd,
        )
        if completed >= len(TASK_TURN_COMMANDS):
            self._set_state("FINAL_EXIT")
            return

        if completed == 1:
            self.tag_pick_first_maneuver = False
        if completed == UNTAGGED_TRIGGER_INTERSECTION + 1:
            self.untagged_pick_next_maneuver = False

        self.task_index += 1
        self.turn_cmd = TASK_TURN_COMMANDS[self.task_index]
        if (completed == UNTAGGED_TRIGGER_INTERSECTION
                and self.enable_untagged_pick
                and not self.untagged_pick_completed):
            self.publish(0, 0, force=True)
            self._shutdown_yolo()
            self.untagged_search_started = False
            self.untagged_search_enabled = False
            self._set_state("A_PICK_PREPARE")
            rospy.loginfo(
                "line_cy_task 已完成第 %d 个路口，"
                "停车加载 A 点抓取模型，全部就绪后再直行搜索",
                completed,
            )
            return
        self._switch_yolo_profile_if_needed()
        self._set_state("FOLLOW")
        rospy.loginfo(
            "line_cy_task intersection %d/%d command=%s",
            self.task_index + 1, len(TASK_TURN_COMMANDS), self.turn_cmd,
        )

    def _exit_alignment_parameters(self):
        third_right_before_a_pick = (
            self.task_index + 1 == UNTAGGED_TRIGGER_INTERSECTION
            and self.turn_cmd == "right"
            and self.enable_untagged_pick
            and not self.untagged_pick_completed
        )
        if third_right_before_a_pick:
            return (
                A_PICK_EXIT_ALIGN_TOLERANCE_DEG,
                A_PICK_EXIT_ALIGN_KP,
                A_PICK_EXIT_ALIGN_MIN_ANGULAR,
                A_PICK_EXIT_ALIGN_MAX_ANGULAR,
            )
        return (
            ALIGN_TOLERANCE_DEG,
            ALIGN_KP,
            ALIGN_MIN_ANGULAR,
            ALIGN_MAX_ANGULAR,
        )

    def _update_lane_width(self, observation, frame_width):
        measured = observation.measured_width
        if LANE_WIDTH_PIXELS > 0 or measured is None:
            return
        if frame_width * MIN_LANE_WIDTH_RATIO <= measured <= frame_width * MAX_LANE_WIDTH_RATIO:
            self.lane_width = 0.90 * self.lane_width + 0.10 * measured
            self.bridge.lane_width = self.lane_width

    def _update_bridge(self, binary, center_hint=None, target_y=None):
        height, width = binary.shape[:2]
        raw_left, raw_right = self.lanes.points(binary, center_hint)
        if self.bridge.left_model is None:
            left_points = raw_left
        else:
            left_points = self.lanes.points_near_model(binary, self.bridge.left_model)
        if self.bridge.right_model is None:
            right_points = raw_right
        else:
            right_points = self.lanes.points_near_model(binary, self.bridge.right_model)
        target_y = int(height * ROI_BOTTOM) if target_y is None else int(target_y)
        return self.bridge.update(
            left_points, right_points, target_y, center_hint,
            frame_width=width, validation_top_y=int(height * ROI_TOP),
        )

    def _prepare_yolo_save_dir(self):
        ensure_clean_directory(self.yolo_save_dir)

    def _save_yolo_event_image(self, event, detections):
        with self.yolo_lock:
            frame = None if self.yolo_latest_frame is None \
                else self.yolo_latest_frame.copy()
        if frame is None:
            frame = np.zeros((1, 1, 3), dtype=np.uint8)
        event_confidence = self.yolo_confidence
        if event.kind == "building":
            event_confidence = self.yolo_building_confidence
        elif (event.kind == "street"
              and YOLO_STREET_MESSAGES[event.class_name][0] == "trash"):
            event_confidence = self.yolo_trash_confidence
        event_detections = [
            item for item in detections
            if item.class_name == event.class_name
            and item.confidence >= event_confidence
        ]
        if not event_detections and event.detection is not None:
            event_detections = [event.detection]
        boxed = draw_yolo_boxes(
            frame, event_detections, self.yolo_center_band_ratio,
            draw_center_band=False,
            center_roi_x_ratio=(YOLO_BUILDING_CENTER_ROI_X_RATIO
                                if event.kind == "building" else None),
        )
        self.task_ledger.save_index += 1
        result = event.display_name
        filename = "%02d_%s_%s.jpg" % (
            self.task_ledger.save_index,
            safe_filename_text(event.area),
            safe_filename_text(result),
        )
        path = os.path.join(self.yolo_save_dir, filename)
        cv2.imwrite(path, boxed)
        return path

    def _report_yolo_task_event(self, detections):
        event = self.task_ledger.pending_event
        if event is None:
            return
        if event.kind == "street":
            target_kind, _ = YOLO_STREET_MESSAGES[event.class_name]
            if target_kind == "people":
                rospy.loginfo("%s识别到%s", event.area, event.display_name)
            else:
                rospy.loginfo(
                    "%s检测到垃圾桶：%s",
                    event.area, event.display_name,
                )
        elif event.kind == "building":
            rospy.loginfo("%s检测到%s", event.area, event.display_name)
        self._save_yolo_event_image(event, detections)
        self.task_ledger.pending_event = None

    def _store_yolo_result(self, frame, detections):
        display_frame = None if frame is None else frame.copy()
        with self.yolo_lock:
            self.yolo_latest_seq += 1
            self.yolo_latest_detections = list(detections)
            self.yolo_latest_frame = display_frame

    def _yolo_loop(self):
        while self.yolo_running and not rospy.is_shutdown():
            if not self.yolo_enabled or self.yolo_detector is None \
                    or self.yolo_camera is None:
                time.sleep(0.05)
                continue
            if not self._yolo_inference_allowed():
                time.sleep(0.05)
                continue
            ok, frame = self.yolo_camera.read(0.2)
            if not ok:
                continue
            self.yolo_counter += 1
            if self.yolo_counter % self.yolo_frame_interval != 0:
                continue
            try:
                with self.yolo_switch_lock:
                    detector = self.yolo_detector
                    if detector is None:
                        continue
                    self.yolo_worker_active = True
                    detections = detector.detect(frame)
            except Exception as exc:
                rospy.logwarn("line_cy_task YOLO inference failed: %s", exc)
                detections = []
            finally:
                self.yolo_worker_active = False
            self._store_yolo_result(frame, detections)

    def _poll_yolo_detections(self):
        if not self.yolo_enabled or self.yolo_detector is None \
                or self.yolo_camera is None:
            return False, []
        with self.yolo_lock:
            if self.yolo_latest_seq == self.yolo_read_seq:
                return False, []
            self.yolo_read_seq = self.yolo_latest_seq
            return True, list(self.yolo_latest_detections)

    def _latest_yolo_seq(self):
        with self.yolo_lock:
            return self.yolo_latest_seq

    def _current_yolo_context(self):
        return yolo_route_context(
            getattr(self, "task_index", 0),
            getattr(self, "state", "FOLLOW"),
        )

    def _yolo_context_key(self, context):
        if context.get("kind") == "street":
            return ("street", tuple(context.get("areas", ())))
        if context.get("kind") == "building":
            return ("building", context.get("area"))
        return ("off", None)

    def _mark_yolo_segment_if_needed(self):
        context = self._current_yolo_context()
        key = self._yolo_context_key(context)
        if key != self.yolo_segment_key:
            self.yolo_segment_key = key
            self.yolo_segment_start_seq = self._latest_yolo_seq()
        return context

    def _yolo_inference_allowed(self):
        if not self.yolo_enabled:
            return False
        if getattr(self, "state", None) not in (
                "FOLLOW", "YOLO_STOP"):
            return False
        return (
            getattr(self, "state", None) == "YOLO_STOP"
            or self._current_yolo_context().get("kind") != "off"
        )

    def _yolo_segment_has_fresh_result(self):
        if self._current_yolo_context().get("kind") == "off":
            return True
        return self._latest_yolo_seq() > self.yolo_segment_start_seq

    def _wait_for_yolo_ready_if_needed(self):
        context = self._mark_yolo_segment_if_needed()
        if context.get("kind") == "off" or not self.yolo_enabled:
            return True
        if self.yolo_ready and self._yolo_segment_has_fresh_result():
            return True
        self.publish(0, 0)
        return False

    def _select_yolo_stop_event(self, detections, now=None):
        return self.task_ledger.select_event(
            self._current_yolo_context(), detections, self.yolo_confidence,
            getattr(self, "yolo_building_confidence", self.yolo_confidence),
            getattr(self, "yolo_people_stable_frames",
                    YOLO_PEOPLE_STABLE_FRAMES),
            getattr(self, "yolo_trash_confidence", YOLO_TRASH_CONFIDENCE),
            getattr(self, "yolo_trash_stable_frames",
                    YOLO_TRASH_STABLE_FRAMES),
            getattr(self, "yolo_building_stable_frames",
                    YOLO_BUILDING_STABLE_FRAMES),
        )

    def _maybe_enter_yolo_stop(self, observation):
        if self.state != "FOLLOW" or not self.yolo_enabled:
            return False
        if rospy.get_time() < getattr(self, "yolo_accept_after", 0.0):
            return False
        if self._current_yolo_context().get("kind") == "off":
            return False
        if not self._wait_for_yolo_ready_if_needed():
            return False
        sampled, detections = self._poll_yolo_detections()
        if not sampled:
            return False
        event = self._select_yolo_stop_event(detections)
        if event is None or not self.yolo_stop_enabled:
            return False
        self.task_ledger.accept(event)
        self.yolo_stop_detection = event.detection
        self.yolo_stop_event = event
        self.yolo_stop_reported = False
        self.yolo_stop_report_seq = self._latest_yolo_seq()
        self.delivery_arm_wait_reported = False
        self._set_state("YOLO_STOP")
        self.publish(0, 0)
        return True

    def _handle_yolo_stop(self, now):
        if self.state != "YOLO_STOP":
            return False
        self.publish(0, 0)
        if not self.yolo_stop_reported:
            sampled, detections = self._poll_yolo_detections()
            if sampled and self.yolo_read_seq > self.yolo_stop_report_seq:
                self._report_yolo_task_event(detections)
                self.yolo_stop_reported = True
        if (self.yolo_stop_reported
                and float(now) - self.state_started >= self.yolo_stop_time):
            event = self.yolo_stop_event
            self.yolo_stop_event = None
            if not self._start_delivery_for_event(event):
                self._set_state("FOLLOW")
        return True

    def process(self, raw_frame):
        frame = self._resize(raw_frame)
        binary = self.vision.apply(frame)
        current_left, current_right = self.lanes.points(binary)
        lane_tracks = [current_left, current_right]
        if self.last_observation is not None:
            lane_tracks.extend([self.last_observation.left_points,
                                self.last_observation.right_points])
        allow_strong_lane_override = strong_lane_override_enabled(
            self.state, self.turn_cmd, self.maneuver_phase
        )
        cross = self.crosswalk.detect(
            binary, lane_points=lane_tracks,
            allow_strong_lane_override=allow_strong_lane_override,
        )
        lane_binary = mask_crosswalk(binary, cross)
        observation = self.lanes.observe(lane_binary, self.lane_width)
        self._update_lane_width(observation, frame.shape[1])
        self.last_crosswalk = cross
        self.last_observation = observation
        self.last_binary = lane_binary
        now = rospy.get_time()

        if self.state == "PICK_RECOVER":
            self.publish(0, 0, force=True)
            self.pick_recover_hits = self.pick_recover_hits + 1 \
                if observation.valid else 0
            if self.pick_recover_hits >= PICK_RECOVER_STABLE_FRAMES:
                rospy.loginfo(
                    "line_cy_task 抓取后车道已稳定 %d 帧，恢复循迹",
                    self.pick_recover_hits,
                )
                self._set_state("FOLLOW")
            elif now - self.state_started >= PICK_RECOVER_TIMEOUT:
                self._pick_failed("抓取后未能重新识别车道")

        elif self.state == "A_PICK_SEARCH":
            self._handle_untagged_search(
                observation, cross, frame.shape[1])

        elif self.state == "FOLLOW":
            entry_allowed = entry_acceptance_enabled(
                now, getattr(self, "entry_accept_after", 0.0)
            )
            entry_candidate = (
                entry_allowed
                and cross.candidate
                and len(cross.stripe_polygons) >= ENTRY_MIN_STRIPES
            )
            self.stop_hits = follow_entry_hits(
                entry_candidate, self.stop_hits
            )
            if self.stop_hits >= STOP_STABLE_FRAMES:
                self.stop_hits = 0
                self.crosswalk.lock_current_bar()
                self.bridge.reset(self.lane_width)
                self._set_state("APPROACH")
            if (self.state == "FOLLOW"
                    and not self._wait_for_yolo_ready_if_needed()):
                pass
            elif self.state == "FOLLOW" and self._maybe_enter_yolo_stop(observation):
                pass
            elif self.state != "FOLLOW":
                self.publish(0, 0)
            elif observation.valid:
                self._control(observation.center_x, frame.shape[1], FOLLOW_SPEED,
                              FOLLOW_CENTER_BIAS_PIXELS)
            else:
                self.publish(0, 0)

        elif self.state == "YOLO_STOP":
            self._handle_yolo_stop(now)

        elif self.state == "APPROACH":
            bridge_binary = mask_crosswalk(binary, cross, include_loose=True)
            self._update_bridge(bridge_binary, frame.shape[1] * 0.5)
            # tracking_polygon 只是 Hough 跟踪结果，未必通过纯横条几何校验。
            visible = cross.candidate
            self.lost_hits = 0 if visible else self.lost_hits + 1
            bottom = polygon_bottom_in_center_band(
                cross.stop_polygon, frame.shape[1]
            ) if visible else 0
            next_state = approach_next_state(
                visible, bottom, frame.shape[0], self.lost_hits
            )
            if next_state == "FOLLOW":
                self.stop_hits = 0
                self._set_state("FOLLOW")
                if observation.valid:
                    self._control(observation.center_x, frame.shape[1], FOLLOW_SPEED,
                                  FOLLOW_CENTER_BIAS_PIXELS)
                else:
                    self.publish(0, 0)
            elif next_state == "ALIGN":
                self._set_state("ALIGN")
                self.publish(0, 0)
            elif observation.valid:
                self._control(observation.center_x, frame.shape[1], APPROACH_SPEED)
            else:
                self.publish(0, 0)

        elif self.state == "ALIGN":
            bridge_binary = mask_crosswalk(binary, cross, include_loose=True)
            self._update_bridge(bridge_binary, frame.shape[1] * 0.5)
            if self.align_lock is not None:
                _, next_state = self._run_locked_entry_alignment(now)
                if next_state is not None:
                    self._set_state(self._entry_ready_state())
            else:
                angle = cross.stop_angle if cross.candidate else cross.tracking_angle
                if angle is None:
                    self.lost_hits += 1
                    if (ALIGN_LOST_FALLBACK_ENABLED
                            and self.align_last_angle is not None
                            and self._lock_entry_alignment(
                                now, self.align_last_angle
                            )):
                        _, next_state = self._run_locked_entry_alignment(now)
                        if next_state is not None:
                            self._set_state(self._entry_ready_state())
                    else:
                        self.publish(0, 0)
                elif abs(angle) <= ALIGN_TOLERANCE_DEG:
                    self.align_last_angle = float(angle)
                    self.lost_hits = 0
                    self.align_hits += 1
                    self.publish(0, 0)
                else:
                    self.align_last_angle = float(angle)
                    self.lost_hits = 0
                    self.align_hits = 0
                    magnitude = clamp(abs(angle) * ALIGN_KP,
                                      ALIGN_MIN_ANGULAR, ALIGN_MAX_ANGULAR)
                    self.publish(0, -magnitude if angle > 0 else magnitude)
                if self.align_hits >= ALIGN_STABLE_FRAMES:
                    self._set_state(self._entry_ready_state())

        elif self.state == "TRAFFIC_WAIT":
            self._handle_traffic_light_wait(now)

        elif self.state == "EXIT_ALIGN":
            tolerance_deg, align_kp, min_angular, max_angular = \
                self._exit_alignment_parameters()
            angle = cross.stop_angle if cross.candidate else cross.tracking_angle
            visible = angle is not None
            if angle is None:
                self.lost_hits += 1
                self.publish(0, 0)
            elif abs(angle) <= tolerance_deg:
                self.lost_hits = 0
                self.align_hits += 1
                self.publish(0, 0)
            else:
                self.lost_hits = 0
                self.align_hits = 0
                magnitude = clamp(abs(angle) * align_kp,
                                  min_angular, max_angular)
                self.publish(0, -magnitude if angle > 0 else magnitude)
            next_state = exit_alignment_next_state(
                self.align_hits, self.lost_hits, now - self.state_started
            )
            if next_state is not None:
                self._complete_intersection()
                if self.state == "FOLLOW" and observation.valid:
                    self._control(
                        observation.center_x, frame.shape[1], FOLLOW_SPEED,
                        FOLLOW_CENTER_BIAS_PIXELS,
                    )
                elif self.state == "FINAL_EXIT":
                    self.publish(FOLLOW_SPEED, 0.0)
                else:
                    self.publish(0, 0)

        elif self.state == "FINAL_EXIT":
            if now - self.state_started >= self.final_exit_time:
                self.publish(0, 0)
                rospy.loginfo(
                    "line_cy_task route completed, final exit %.2f seconds",
                    self.final_exit_time,
                )
                self._set_state("DONE")
                rospy.signal_shutdown("line_cy_task completed")
            else:
                self.publish(self.turn_speed, 0.0)

        elif self.state == "WAIT":
            angle = cross.stop_angle if cross.candidate else cross.tracking_angle
            visible = angle is not None
            safe_visible = (
                visible and len(cross.stripe_polygons) >= ALIGN_ENTRY_MIN_STRIPES
                and abs(float(angle)) <= ALIGN_ENTRY_MAX_ANGLE
            )
            self.wait_recover_hits = self.wait_recover_hits + 1 if safe_visible else 0
            next_state = wait_recovery_state(
                angle, visible, self.wait_recover_hits,
                len(cross.stripe_polygons),
            )
            self.publish(0, 0)
            if next_state is not None:
                self._set_state(self._entry_ready_state())

        elif self.state == "MANEUVER":
            lane_binary = mask_crosswalk(binary, cross, include_loose=True)
            side = maneuver_follow_side(self.turn_cmd)
            if side is None:
                target_y = int(frame.shape[0] * MANEUVER_LOOKAHEAD_RATIO)
                center, left_model, right_model = self._update_bridge(
                    lane_binary, frame.shape[1] * 0.5, target_y
                )
                if center is None:
                    center = frame.shape[1] * 0.5
                bias = 0.0
                if self.bridge.selected_side == "left":
                    bias = abs(MANEUVER_CENTER_BIAS_PIXELS)
                elif self.bridge.selected_side == "right":
                    bias = -abs(MANEUVER_CENTER_BIAS_PIXELS)
                self._control(
                    center, frame.shape[1], MANEUVER_SPEED, bias,
                )
            else:
                self.last_binary = lane_binary
                self._run_timed_turn_phase(now)

            if self.state == "MANEUVER":
                cross_visible = cross.candidate or len(cross.stripe_polygons) >= 3
                self.clear_hits = 0 if cross_visible else self.clear_hits + 1
                if self.clear_hits >= ENTRY_CLEAR_FRAMES:
                    self.entry_cleared = True
                exit_ready = (side is None
                              or self.maneuver_phase == "EXIT_STRAIGHT")
                stop_bottom = polygon_bottom_in_center_band(
                    cross.stop_polygon, frame.shape[1]
                ) if cross.candidate else 0
                exit_visible = exit_ready and self.entry_cleared and cross.candidate \
                    and stop_bottom >= frame.shape[0] * BAR_TRACK_MIN_BOTTOM_RATIO
                self.exit_hits, exit_near = maneuver_exit(
                    self.entry_cleared, self.exit_hits, exit_visible,
                    stop_bottom, frame.shape[0]
                )
                if exit_near and now - self.state_started >= MANEUVER_MIN_TIME:
                    self.crosswalk.lock_current_bar()
                    self._set_state("EXIT_ALIGN")
                elif maneuver_timeout_exits_to_follow(now - self.state_started):
                    rospy.logwarn(
                        "line_cy_task maneuver timeout, complete current intersection"
                    )
                    self._complete_intersection()
                    if self.state == "FOLLOW" and observation.valid:
                        self._control(observation.center_x, frame.shape[1], FOLLOW_SPEED,
                                      FOLLOW_CENTER_BIAS_PIXELS)
                    elif self.state == "FINAL_EXIT":
                        self.publish(self.turn_speed, 0.0)
                    else:
                        self.publish(0, 0)

        else:
            self.publish(0, 0)

        if self.debug_view:
            self.draw_debug(frame)
        if getattr(self, "yolo_debug_view", False):
            self.draw_yolo_debug()

    def draw_yolo_debug(self):
        with self.yolo_lock:
            if self.yolo_latest_frame is None:
                return
            frame = self.yolo_latest_frame.copy()
            detections = list(self.yolo_latest_detections)

        frame = draw_yolo_boxes(
            frame, detections,
            getattr(self, "yolo_center_band_ratio", YOLO_CENTER_BAND_RATIO),
            draw_center_band=(
                getattr(self, "yolo_active_profile", None) != "building"),
            center_roi_x_ratio=(YOLO_BUILDING_CENTER_ROI_X_RATIO
                                if getattr(self, "yolo_active_profile", None)
                                == "building" else None),
        )
        status = "YOLO frame_interval={} detections={}".format(
            getattr(self, "yolo_frame_interval", YOLO_FRAME_INTERVAL),
            len(detections)
        )
        cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
        try:
            cv2.imshow(YOLO_WINDOW_NAME, frame)
            cv2.waitKey(1)
        except cv2.error:
            self.yolo_debug_view = False

    def draw_debug(self, frame):
        height, width = frame.shape[:2]
        observation = self.last_observation
        target_x = observation.center_x if self.last_control_target is None \
            else self.last_control_target
        virtual_display = clip_points_for_display(
            observation.virtual_left_points + observation.virtual_right_points, width
        )
        center_path = sorted(observation.center_points, key=lambda point: point[1])
        side = observation.follow_side or ("dual" if observation.dual_rows else "none")
        if self.last_crosswalk.stop_polygon is not None:
            bar_state = "stop"
        elif self.last_crosswalk.tracking_polygon is not None:
            bar_state = "track"
        else:
            bar_state = "none"
        bar_angle = (self.last_crosswalk.stop_angle
                     if self.last_crosswalk.stop_angle is not None
                     else self.last_crosswalk.tracking_angle)
        angle_text = "--" if bar_angle is None else "{:.1f}".format(bar_angle)
        align_lock = getattr(self, "align_lock", None)
        align_angle = (getattr(self, "align_last_angle", None)
                       if align_lock is None else align_lock["angle"])
        lock_text = "--" if align_angle is None else "{:.1f}".format(align_angle)
        text = ("task={}/{} state={} cmd={} phase={} side={} lane={:.0f} dual={} "
                "ctrl={:+.2f} stripes={} bar={} angle={} lock={} hits={} "
                "cross={:.2f}").format(
            self.task_index + 1, len(TASK_TURN_COMMANDS),
            self.state, self.turn_cmd, self.maneuver_phase,
            side, self.lane_width,
            observation.dual_rows,
            getattr(self, "last_command_angular", 0.0),
            len(self.last_crosswalk.stripe_polygons),
            bar_state, angle_text, lock_text, self.crosswalk.bar_only_hits,
            self.last_crosswalk.confidence)
        try:
            cv2.imshow(WINDOW_NAME, frame)
            processed = cv2.cvtColor(self.last_binary, cv2.COLOR_GRAY2BGR)
            top, bottom = int(height * ROI_TOP), int(height * ROI_BOTTOM)
            cv2.rectangle(processed, (0, top), (width - 1, bottom), (0, 180, 0), 2)
            cv2.line(processed, (width // 2, top), (width // 2, bottom), (100, 100, 100), 1)
            stop_half_width = width * clamp(STOP_CENTER_WIDTH_RATIO, 0.0, 1.0) * 0.5
            stop_left = int(round(width * 0.5 - stop_half_width))
            stop_right = int(round(width * 0.5 + stop_half_width))
            stop_top = int(round(height * STOP_NEAR_RATIO))
            cv2.rectangle(processed, (stop_left, stop_top),
                          (stop_right, height - 1), (0, 165, 255), 2)
            for x, y in observation.left_points + observation.right_points:
                cv2.circle(processed, (x, y), 4, (255, 255, 0), -1)
            for x, y in virtual_display:
                cv2.circle(processed, (x, y), 5, (255, 0, 255), 2)
            if center_path:
                for x, y in center_path:
                    cv2.circle(processed, (int(x), int(y)), 4, (0, 255, 0), -1)
            target_display = int(clamp(target_x, 0, width - 1))
            cv2.line(processed, (target_display, bottom - 15),
                     (target_display, bottom + 15), (0, 255, 0), 3)
            for polygon in self.last_crosswalk.stripe_polygons:
                cv2.polylines(processed, [np.asarray(polygon, np.int32)], True,
                              (0, 255, 255), 2)
            if self.last_crosswalk.stop_polygon is not None:
                cv2.polylines(processed,
                              [np.asarray(self.last_crosswalk.stop_polygon, np.int32)],
                              True, (0, 0, 255), 3)
            elif (self.state in ("APPROACH", "ALIGN", "EXIT_ALIGN")
                  and self.last_crosswalk.tracking_polygon is not None):
                cv2.polylines(
                    processed,
                    [np.asarray(self.last_crosswalk.tracking_polygon, np.int32)],
                    True, (0, 128, 255), 2,
                )
            if self.state == "MANEUVER" and self.turn_cmd == "straight":
                y1, y2 = int(height * ROI_TOP), int(height * ROI_BOTTOM)
                models = (
                    (self.bridge.left_model, (255, 128, 0), 3),
                    (self.bridge.right_model, (0, 255, 0), 3),
                    (self.bridge.center_model, (255, 0, 255), 2),
                )
                for model, color, thickness in models:
                    if model is None:
                        continue
                    x1 = int(clamp(model.x_at(y1), 0, width - 1))
                    x2 = int(clamp(model.x_at(y2), 0, width - 1))
                    cv2.line(processed, (x1, y1), (x2, y2), color, thickness)
                lookahead_y = int(height * MANEUVER_LOOKAHEAD_RATIO)
                cv2.circle(processed, (target_display, lookahead_y),
                           7, (0, 0, 255), 2)
            cv2.putText(processed, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2)
            cv2.imshow(PROCESSED_WINDOW_NAME, processed)
            cv2.waitKey(1)
        except cv2.error:
            self.debug_view = False

    def run(self):
        rate = rospy.Rate(20)
        try:
            while not rospy.is_shutdown():
                if self.state not in (
                        "B_PICK_PREPARE", "B_PICKING",
                        "A_PICK_PREPARE", "A_PICKING", "DELIVERING",
                        "PICK_FAILED"):
                    failed = self._runtime_failure()
                    if failed is not None:
                        self._pick_failed(
                            "托管进程 %s 异常退出，状态码 %s" % failed)
                if self.state in (
                        "B_PICK_PREPARE", "B_PICKING",
                        "A_PICK_PREPARE", "A_PICK_SEARCH", "A_PICKING", "DELIVERING",
                        "PICK_FAILED"):
                    self._handle_pick_without_frame(rospy.get_time())
                    rate.sleep()
                    continue
                ok, frame = self.camera.read(1.0)
                if ok:
                    self.process(frame)
                else:
                    self.publish(0, 0)
                rate.sleep()
        finally:
            self.cleanup()

    def cleanup(self):
        if self.cleaned:
            return
        self.cleaned = True
        try:
            self.velocity_owner = "line"
            self.publish(0, 0, force=True)
            self._shutdown_yolo()
            self._close_traffic_light()
            self.camera.release()
            if self.grasp_coordinator is not None:
                self.grasp_coordinator.join(1.0)
        except Exception:
            pass
