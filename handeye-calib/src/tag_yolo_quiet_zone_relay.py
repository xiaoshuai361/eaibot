#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import base64
import copy
import json
import os
import subprocess
import sys
import threading

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


DEFAULT_PYTHON3 = 'auto'
DEFAULT_DETECTOR_SCRIPT = '/home/eaibot/handeye-calib/src/tag_yolo_roi_detector.py'
DEFAULT_MODEL = '/home/eaibot/handeye-calib/src/model/yolov5/tag_yolov5n_640_best.onnx'
DEFAULT_BOX_EXPAND_PIXELS = 0


def can_import_yolo_runtime(python_path):
    try:
        process = subprocess.Popen(
            [python_path, '-c', 'import onnxruntime'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        process.communicate()
        return process.returncode == 0
    except OSError:
        return False


def find_executable(name):
    for directory in os.environ.get('PATH', '').split(os.pathsep):
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def resolve_python3_executable(value, exists=None,
                               can_import_yolo_runtime=None,
                               find_executable=None):
    exists = os.path.exists if exists is None else exists
    can_import_yolo_runtime = globals()['can_import_yolo_runtime'] if can_import_yolo_runtime is None else can_import_yolo_runtime
    find_executable = globals()['find_executable'] if find_executable is None else find_executable
    if value and value != 'auto':
        return value
    candidates = [
        '/home/eaibot/anaconda3/envs/ww/bin/python3',
        '/home/eaibot/anaconda3/envs/ww/bin/python',
        '/home/eaibot/miniconda3/envs/ww/bin/python3',
        '/home/eaibot/miniconda3/envs/ww/bin/python',
        '/home/eaibot/.conda/envs/ww/bin/python3',
        '/home/eaibot/.conda/envs/ww/bin/python',
        '/home/eaibot/ww/bin/python3',
        '/home/eaibot/ww/bin/python',
    ]
    path_python3 = find_executable('python3')
    if path_python3:
        candidates.append(path_python3)
    candidates.append('/usr/bin/python3')

    checked = set()
    for candidate in candidates:
        if candidate in checked:
            continue
        checked.add(candidate)
        if exists(candidate) and can_import_yolo_runtime(candidate):
            return candidate
    raise RuntimeError(
        'Could not find a Python3 executable that can import onnxruntime. '
        'Pass --python3 /path/to/ww/bin/python3.'
    )


def should_process_frame(now, last_publish, interval):
    if interval <= 0.0:
        return True
    return now - last_publish >= interval


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Realtime YOLO quiet-zone relay for AprilTag.')
    parser.add_argument('--python3', default=DEFAULT_PYTHON3)
    parser.add_argument('--detector-script', default=DEFAULT_DETECTOR_SCRIPT)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--image-topic', default='/camera/rgb/image_raw')
    parser.add_argument('--camera-info-topic', default='/camera/rgb/camera_info')
    parser.add_argument('--output-image-topic', default='/tag_yolo_quiet/image_raw')
    parser.add_argument('--output-camera-info-topic', default='/tag_yolo_quiet/camera_info')
    parser.add_argument('--detections-topic', default='/tag_yolo_quiet/detections_json')
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--margin-ratio', type=float, default=0.35)
    parser.add_argument('--box-expand-pixels', type=float,
                        default=DEFAULT_BOX_EXPAND_PIXELS,
                        help='Expand each YOLO box before preserving tag pixels.')
    parser.add_argument('--yolo-hz', type=float, default=2.0,
                        help='YOLO box refresh rate. Cached boxes are reused between refreshes.')
    parser.add_argument('--publish-hz', type=float, default=8.0,
                        help='Output image publish rate. Lower this if apriltag_ros or display is laggy.')
    parser.add_argument('--show-yolo-boxes', action='store_true',
                        help='Draw YOLO outer boxes outside tag interiors for debugging.')
    parser.add_argument('--queue-size', type=int, default=1)
    return parser.parse_args(rospy.myargv(argv)[1:])


def build_detections_payload(header, image_width, image_height, detections):
    stamp = getattr(header, 'stamp', None)
    payload = {
        'stamp': {
            'secs': int(getattr(stamp, 'secs', 0)),
            'nsecs': int(getattr(stamp, 'nsecs', 0)),
        },
        'frame_id': getattr(header, 'frame_id', ''),
        'image_width': int(image_width),
        'image_height': int(image_height),
        'detections': [],
    }
    for detection in detections or []:
        box = detection.get('box', [])
        outer_box = detection.get('outer_box', box)
        class_id = int(detection.get('class_id'))
        payload['detections'].append({
            'tag_id': class_id + 1,
            'class_id': class_id,
            'class_name': detection.get('class_name'),
            'confidence': float(detection.get('confidence')),
            'box': [float(value) for value in box],
            'outer_box': [float(value) for value in outer_box],
        })
    return payload


class QuietZoneWorker(object):
    def __init__(self, args):
        python3 = resolve_python3_executable(args.python3)
        rospy.loginfo('YOLO quiet-zone worker Python3: %s', python3)
        command = [
            python3,
            args.detector_script,
            '--model',
            args.model,
            '--mode',
            'worker',
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0)
        self._stderr_thread = threading.Thread(target=self._drain_stderr)
        self._stderr_thread.daemon = True
        self._stderr_thread.start()

    def _drain_stderr(self):
        for line in iter(self.process.stderr.readline, b''):
            text = line.decode('utf-8', 'replace').strip()
            if text:
                rospy.logwarn('tag_yolo worker: %s', text)

    def process_frame(self, image_bgr, confidence, margin_ratio, box_expand_pixels,
                      refresh_boxes, draw_yolo_overlay=False):
        import cv2
        import numpy as np

        ok, encoded = cv2.imencode('.png', image_bgr)
        if not ok:
            raise RuntimeError('Failed to encode frame for YOLO quiet-zone worker.')
        request = {
            'image_bgr_png_base64': base64.b64encode(encoded.tobytes()).decode('ascii'),
            'confidence': float(confidence),
            'margin_ratio': float(margin_ratio),
            'box_expand_pixels': float(box_expand_pixels),
            'refresh_boxes': bool(refresh_boxes),
            'draw_yolo_overlay': bool(draw_yolo_overlay),
        }
        line = json.dumps(request, ensure_ascii=True, allow_nan=False) + '\n'
        try:
            self.process.stdin.write(line.encode('ascii'))
            self.process.stdin.flush()
            response_line = self.process.stdout.readline()
        except Exception as exc:
            raise RuntimeError('YOLO quiet-zone worker communication failed: {}'.format(exc))
        if not response_line:
            raise RuntimeError('YOLO quiet-zone worker exited unexpectedly.')
        try:
            response = json.loads(response_line.decode('utf-8', 'replace'))
        except ValueError as exc:
            raise RuntimeError('YOLO quiet-zone worker returned invalid JSON: {}'.format(exc))
        if not response.get('ok'):
            raise RuntimeError('YOLO quiet-zone worker failed: {}'.format(
                response.get('error', 'unknown error')))
        image_data = response.get('image_bgr_png_base64')
        if not image_data:
            raise RuntimeError('YOLO quiet-zone worker response has no image.')
        decoded = np.frombuffer(base64.b64decode(image_data.encode('ascii')), dtype=np.uint8)
        quiet_bgr = cv2.imdecode(decoded, cv2.IMREAD_COLOR)
        if quiet_bgr is None:
            raise RuntimeError('Failed to decode quiet-zone frame from worker.')
        return quiet_bgr, response.get('detections', [])

    def close(self):
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception:
            pass
        if self.process.poll() is None:
            try:
                self.process.terminate()
            except Exception:
                pass


class QuietZoneRelay(object):
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.worker = QuietZoneWorker(args)
        self.latest_camera_info = None
        self.busy = False
        self.last_yolo_update = rospy.Time(0)
        self.last_publish = rospy.Time(0)
        self.last_error_log_time = rospy.Time(0)
        self.lock = threading.Lock()
        self.image_pub = rospy.Publisher(args.output_image_topic, Image, queue_size=1)
        self.info_pub = rospy.Publisher(args.output_camera_info_topic, CameraInfo, queue_size=1)
        self.detections_pub = rospy.Publisher(args.detections_topic, String, queue_size=1)
        self.info_sub = rospy.Subscriber(
            args.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)
        self.image_sub = rospy.Subscriber(
            args.image_topic, Image, self.image_callback, queue_size=args.queue_size)
        rospy.loginfo('YOLO quiet-zone relay: %s + %s -> %s + %s',
                      args.image_topic, args.camera_info_topic,
                      args.output_image_topic, args.output_camera_info_topic)

    def camera_info_callback(self, message):
        self.latest_camera_info = message

    def image_callback(self, message):
        if self.latest_camera_info is None:
            rospy.logwarn_throttle(5.0, 'Waiting for CameraInfo: %s', self.args.camera_info_topic)
            return
        with self.lock:
            if self.busy:
                return
            publish_interval = 1.0 / self.args.publish_hz if self.args.publish_hz > 0.0 else 0.0
            now = rospy.Time.now()
            if not should_process_frame(now.to_sec(), self.last_publish.to_sec(), publish_interval):
                return
            self.busy = True
        try:
            yolo_interval = 1.0 / self.args.yolo_hz if self.args.yolo_hz > 0.0 else 0.0
            refresh_boxes = (
                yolo_interval <= 0.0 or
                (rospy.Time.now() - self.last_yolo_update).to_sec() >= yolo_interval
            )
            image_bgr = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            quiet_bgr, detections = self.worker.process_frame(
                image_bgr, self.args.confidence, self.args.margin_ratio,
                self.args.box_expand_pixels, refresh_boxes,
                self.args.show_yolo_boxes)
            if refresh_boxes:
                self.last_yolo_update = rospy.Time.now()
            output = self.bridge.cv2_to_imgmsg(quiet_bgr, encoding='bgr8')
            output.header = message.header
            camera_info = copy.deepcopy(self.latest_camera_info)
            camera_info.header = message.header
            self.image_pub.publish(output)
            self.info_pub.publish(camera_info)
            payload = build_detections_payload(
                message.header, image_bgr.shape[1], image_bgr.shape[0], detections)
            self.detections_pub.publish(
                String(data=json.dumps(payload, ensure_ascii=True, allow_nan=False)))
            self.last_publish = rospy.Time.now()
            rospy.logdebug_throttle(
                2.0,
                'YOLO quiet-zone relay publishing; cached_yolo_boxes=%d refresh_yolo=%s',
                len(detections), refresh_boxes)
        except Exception as exc:
            now = rospy.Time.now()
            if (now - self.last_error_log_time).to_sec() >= 2.0:
                rospy.logerr('YOLO quiet-zone relay failed: %s', exc)
                self.last_error_log_time = now
        finally:
            with self.lock:
                self.busy = False

    def close(self):
        self.worker.close()


def main(argv=None):
    argv = sys.argv if argv is None else argv
    rospy.init_node('tag_yolo_quiet_zone_relay', anonymous=False)
    args = parse_args(argv)
    relay = QuietZoneRelay(args)
    rospy.on_shutdown(relay.close)
    rospy.spin()


if __name__ == '__main__':
    main()
