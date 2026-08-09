#!/usr/bin/env python3
# coding=utf-8
"""摄像头读取和基础控制器。"""
import threading
import time

import cv2
import rospy


def _clamp(value, low, high):
    return max(low, min(high, value))


def _set_capture_resolution(capture, width, height):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class CameraReader(object):
    def __init__(self, index, frame_width=None, frame_height=None):
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.read_seq = -1
        self.running = self.cap.isOpened()
        self.thread = None
        if self.running:
            if frame_width is not None and frame_height is not None:
                _set_capture_resolution(self.cap, frame_width, frame_height)
            else:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.thread = threading.Thread(target=self._loop)
            self.thread.daemon = True
            self.thread.start()

    def _loop(self):
        while self.running and not rospy.is_shutdown():
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.seq += 1
            else:
                time.sleep(0.02)

    def read(self, timeout=0.5):
        end = time.time() + timeout
        while time.time() < end and not rospy.is_shutdown():
            with self.lock:
                if self.frame is not None and self.seq != self.read_seq:
                    self.read_seq = self.seq
                    return True, self.frame.copy()
            time.sleep(0.01)
        return False, None

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(0.8)
        self.cap.release()


class PID(object):
    def __init__(self, kp, kd, limit):
        self.kp = float(kp)
        self.kd = float(kd)
        self.limit = float(limit)
        self.last_error = 0.0
        self.last_time = None

    def reset(self):
        self.last_error = 0.0
        self.last_time = None

    def update(self, deviation, kp=None, kd=None):
        now = rospy.get_time()
        dt = 0.05 if self.last_time is None else max(0.001, now - self.last_time)
        error = -float(deviation)
        active_kp = self.kp if kp is None else float(kp)
        active_kd = self.kd if kd is None else float(kd)
        output = active_kp * error + active_kd * (error - self.last_error) / dt
        self.last_error = error
        self.last_time = now
        return _clamp(output, -self.limit, self.limit)
