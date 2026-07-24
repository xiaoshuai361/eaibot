#!/usr/bin/env python
# coding=utf-8
"""line_cy 的原图叠加版。

保留 line_cy.py 中现有的二值化巡线、斑马线停车和路口补线逻辑；
调试窗口改为显示叠加了补线/停车信息的原始彩色图像。
如有需要，仍可按参数开启程序内录像，但默认关闭。
"""

import os
import time

import cv2
import rospy

from eaibot.robocom_ws.src.line_cy_old import LaneFollower as BaseLaneFollower
from eaibot.robocom_ws.src.line_cy_old import clamp


RECORD_VIDEO = False
RECORD_FPS = 20.0
RECORD_CODEC = "MJPG"
RECORD_OUTPUT_DIR = os.path.expanduser("~/robocom_ws/videos")
RECORD_PREFIX = "line_cy"
RECORD_PREVIEW = False
OVERLAY_WINDOW_NAME = "line_cy_overlay"


class LaneFollowerWithVideo(BaseLaneFollower):
    def __init__(self):
        super(LaneFollowerWithVideo, self).__init__()
        self.record_video = bool(rospy.get_param("~record_video", RECORD_VIDEO))
        self.record_fps = float(rospy.get_param("~record_fps", RECORD_FPS))
        self.record_codec = str(rospy.get_param("~record_codec", RECORD_CODEC)).upper().strip()
        self.record_output_dir = os.path.expanduser(
            str(rospy.get_param("~record_output_dir", RECORD_OUTPUT_DIR)).strip()
        )
        self.record_prefix = str(rospy.get_param("~record_prefix", RECORD_PREFIX)).strip() or RECORD_PREFIX
        self.record_preview = bool(rospy.get_param("~record_preview", RECORD_PREVIEW))
        self.video_writer = None
        self.video_path = None
        self.record_frames = 0
        self.last_centers = []
        self.last_processed_shape = None

        if self.record_video:
            rospy.loginfo(
                "录像已开启: dir=%s fps=%.1f codec=%s",
                self.record_output_dir,
                self.record_fps,
                self.record_codec,
            )

    def _writer_candidates(self):
        primary = self.record_codec if len(self.record_codec) == 4 else RECORD_CODEC
        for codec in (primary, "XVID", "MJPG", "mp4v"):
            yield codec

    def _ensure_writer(self, frame):
        if not self.record_video or self.video_writer is not None:
            return

        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            return

        try:
            os.makedirs(self.record_output_dir)
        except OSError:
            if not os.path.isdir(self.record_output_dir):
                rospy.logerr("无法创建录像目录: %s", self.record_output_dir)
                self.record_video = False
                return

        stamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = "%s_%s" % (self.record_prefix, stamp)

        for codec in self._writer_candidates():
            ext = ".mp4" if codec.lower() == "mp4v" else ".avi"
            path = os.path.join(self.record_output_dir, base_name + "_" + codec.lower() + ext)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(path, fourcc, self.record_fps, (width, height))
            if writer.isOpened():
                self.video_writer = writer
                self.video_path = path
                rospy.loginfo("开始录像: %s", self.video_path)
                return
            writer.release()

        rospy.logerr("无法初始化录像器，已关闭录像功能")
        self.record_video = False

    def _scale_point(self, x, y, raw_shape, processed_shape):
        raw_h, raw_w = raw_shape[:2]
        proc_h, proc_w = processed_shape[:2]
        scale_x = float(raw_w) / float(proc_w)
        scale_y = float(raw_h) / float(proc_h)
        sx = int(round(clamp(x * scale_x, 0, raw_w - 1)))
        sy = int(round(clamp(y * scale_y, 0, raw_h - 1)))
        return sx, sy

    def _scale_box(self, box, raw_shape, processed_shape):
        x, y, w, h = box
        x1, y1 = self._scale_point(x, y, raw_shape, processed_shape)
        x2, y2 = self._scale_point(x + w, y + h, raw_shape, processed_shape)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    def build_overlay_frame(self, raw_frame, stop_result=None):
        if raw_frame is None:
            return None

        display = raw_frame.copy()
        raw_h, raw_w = display.shape[:2]
        processed_shape = self.last_processed_shape or raw_frame.shape
        proc_h, proc_w = processed_shape[:2]

        top = self.last_debug.get("search_top", int(proc_h * self.vision.roi_top_ratio))
        bot = self.last_debug.get("search_bot", int(proc_h * self.vision.roi_bottom_ratio))
        _, top_y = self._scale_point(0, top, display.shape, processed_shape)
        _, bot_y = self._scale_point(0, bot, display.shape, processed_shape)

        cv2.rectangle(display, (0, top_y), (raw_w - 1, bot_y), (0, 180, 0), 2)
        cv2.line(display, (raw_w // 2, 0), (raw_w // 2, raw_h - 1), (90, 90, 90), 1)

        for x, y, left, right in self.last_debug.get("groups", []):
            px, py = self._scale_point(x, y, display.shape, processed_shape)
            pleft, _ = self._scale_point(left, y, display.shape, processed_shape)
            pright, _ = self._scale_point(right, y, display.shape, processed_shape)
            cv2.circle(display, (px, py), 3, (255, 80, 0), -1)
            cv2.line(display, (pleft, py), (pright, py), (255, 80, 0), 1)

        for x, y, left, right in self.last_debug.get("ignored", []):
            px, py = self._scale_point(x, y, display.shape, processed_shape)
            pleft, _ = self._scale_point(left, y, display.shape, processed_shape)
            pright, _ = self._scale_point(right, y, display.shape, processed_shape)
            cv2.circle(display, (px, py), 3, (40, 40, 180), -1)
            cv2.line(display, (pleft, py), (pright, py), (40, 40, 180), 1)

        for row in self.last_debug.get("lane_rows", []):
            center_x, center_y = self._scale_point(row["center_x"], row["y"], display.shape, processed_shape)
            for key in ("left_x", "right_x"):
                if row.get(key) is not None:
                    edge_x, edge_y = self._scale_point(row[key], row["y"], display.shape, processed_shape)
                    cv2.line(display, (edge_x, edge_y - 12), (edge_x, edge_y + 12), (255, 255, 0), 2)
            if row.get("virtual_x") is not None:
                vx, vy = self._scale_point(row["virtual_x"], row["y"], display.shape, processed_shape)
                cv2.line(display, (vx, vy - 18), (vx, vy + 18), (255, 0, 255), 2)
            cv2.circle(display, (center_x, center_y), 5, (0, 255, 0), -1)

        for x, y in self.last_centers:
            px, py = self._scale_point(x, y, display.shape, processed_shape)
            cv2.circle(display, (px, py), 8, (0, 255, 0), 2)

        if stop_result:
            if stop_result.get("stop_box"):
                x, y, w, h = self._scale_box(stop_result["stop_box"], display.shape, processed_shape)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
            for box in stop_result.get("stripe_boxes", []):
                x, y, w, h = self._scale_box(box, display.shape, processed_shape)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)

        status = "state={} hint={} raw={} single={} conf={:.2f} width={:.0f} fail={} v={:.2f} w={:.2f}".format(
            self.state,
            self.last_debug.get("side_hint"),
            self.last_debug.get("raw_single_turn"),
            self.last_debug.get("single_turn"),
            0.0 if not stop_result else stop_result.get("confidence", 0.0),
            self.lane_width(proc_w),
            self.failed_count,
            self.twist.linear.x,
            self.twist.angular.z,
        )
        cv2.putText(display, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)
        cv2.putText(
            display,
            "RAW overlay | ROI green | active cyan/blue | ignored dark-red | virtual magenta | center green",
            (10, raw_h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 200, 255),
            1,
        )

        return display

    def show_overlay_debug(self, raw_frame, stop_result=None):
        display = self.build_overlay_frame(raw_frame, stop_result)
        if display is None:
            return

        height, width = display.shape[:2]
        if self.debug_max_width > 0 and width > self.debug_max_width:
            scale = float(self.debug_max_width) / float(width)
            display = cv2.resize(display, (self.debug_max_width, int(height * scale)))

        try:
            cv2.imshow(OVERLAY_WINDOW_NAME, display)
            cv2.waitKey(1)
        except cv2.error as exc:
            rospy.logwarn("原图调试窗口打开失败，关闭 debug_view: %s", exc)
            self.debug_view = False

    def write_record_frame(self, raw_frame):
        if not self.record_video:
            return

        frame = self.build_overlay_frame(raw_frame, self.last_stop)
        if frame is None:
            return

        self._ensure_writer(frame)
        if self.video_writer is None:
            return

        self.video_writer.write(frame)
        self.record_frames += 1

        if self.record_preview:
            try:
                cv2.imshow("line_cy_record", frame)
                cv2.waitKey(1)
            except cv2.error as exc:
                rospy.logwarn("录像预览窗口打开失败，关闭 record_preview: %s", exc)
                self.record_preview = False

    def process_frame(self, frame):
        raw_frame = frame.copy()
        processed_frame = self.resize_frame(frame)
        binary = self.vision.mask_black(processed_frame)
        self.last_processed_shape = processed_frame.shape
        self.last_stop = self.vision.detect_stopline_before_crosswalk(binary)

        now = rospy.get_time()
        if self.last_stop["detected"] and now >= self.stop_cooldown_until:
            self.stop_hits += 1
        else:
            self.stop_hits = max(0, self.stop_hits - 1)

        if self.stop_hits >= self.stop_stable_frames and not self.detect_only:
            self.state = "STOPPED"
            self.last_centers = []
            if self.debug_view:
                self.show_overlay_debug(raw_frame, self.last_stop)
            self.write_record_frame(raw_frame)
            self.stop_robot(self.stop_hold_time)
            self.run_maneuver(self.get_turn_cmd())
            return

        centers, _ = self.line_control(processed_frame, binary, "normal")
        self.last_centers = centers
        if self.debug_view:
            self.show_overlay_debug(raw_frame, self.last_stop)
        self.write_record_frame(raw_frame)

    def run_maneuver(self, cmd):
        mode, bias = self.maneuver_mode(cmd)
        self.state = "MANEUVER"
        self.pid.reset()
        dual_stable = 0
        crosswalk_clear = 0
        start = rospy.get_time()
        rate = rospy.Rate(20)
        rospy.loginfo("进入路口补线: cmd=%s mode=%s", cmd, mode)

        while not rospy.is_shutdown() and rospy.get_time() - start <= self.intersection_max_time:
            ok, frame = self.cap.read()
            if not ok:
                rospy.logerr("路口补线中无法读取图像")
                break

            raw_frame = frame.copy()
            processed_frame = self.resize_frame(frame)
            binary = self.vision.mask_black(processed_frame)
            self.last_processed_shape = processed_frame.shape
            self.last_stop = self.vision.detect_stopline_before_crosswalk(binary)
            lane_binary = self.suppress_crosswalk_regions(binary, self.last_stop)
            elapsed = rospy.get_time() - start
            if elapsed < self.enter_intersection_straight_time:
                active_mode, active_bias = "right", self.straight_bias
            else:
                active_mode, active_bias = mode, bias
            centers, _ = self.line_control(processed_frame, lane_binary, active_mode, self.side_follow_speed, active_bias)
            self.last_centers = centers

            crosswalk_visible = (
                self.last_stop.get("confidence", 0.0) >= self.crosswalk_clear_confidence
                or len(self.last_stop.get("stripe_boxes", [])) >= 3
                or self.last_stop.get("detected", False)
            )
            if crosswalk_visible:
                crosswalk_clear = 0
            else:
                crosswalk_clear += 1

            recover_allowed_time = self.enter_intersection_straight_time + self.intersection_min_time
            can_recover = (
                elapsed >= recover_allowed_time
                and crosswalk_clear >= self.crosswalk_clear_frames
                and self.last_debug.get("dual_rows", 0) >= 2
                and self.failed_count <= 1
            )
            if can_recover:
                dual_stable += 1
            else:
                dual_stable = max(0, dual_stable - 1)

            if self.debug_view:
                self.show_overlay_debug(raw_frame, self.last_stop)
            self.write_record_frame(raw_frame)
            if dual_stable >= self.recover_dual_frames:
                break
            rate.sleep()

        elapsed = rospy.get_time() - start
        if elapsed >= self.intersection_max_time:
            rospy.logwarn("路口补线达到 %.1f 秒上限，按保护逻辑恢复巡线", self.intersection_max_time)
        self.state = "FOLLOW_LINE"
        self.pid.reset()
        self.failed_count = 0
        self.stop_hits = 0
        self.stop_cooldown_until = rospy.get_time() + self.stop_cooldown_time
        rospy.loginfo("路口动作完成，恢复巡线")

    def cleanup(self):
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None
            if self.video_path:
                rospy.loginfo("录像已保存: %s, frames=%d", self.video_path, self.record_frames)
        super(LaneFollowerWithVideo, self).cleanup()


if __name__ == "__main__":
    try:
        LaneFollowerWithVideo().run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass