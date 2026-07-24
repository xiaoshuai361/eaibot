#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import math
import os
import time


DEFAULT_OUTPUT_DIR = "/home/eaibot/handeye-calib/diagnostics"


def is_finite(value):
    value = float(value)
    return not math.isnan(value) and not math.isinf(value)


def median(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise RuntimeError("Cannot compute a median from no values.")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def median_absolute_deviation(values, center=None):
    center = median(values) if center is None else float(center)
    return median([abs(float(value) - center) for value in values])


def append_unique_tf_sample(samples, seen_stamps, stamp_ns, position,
                            orientation_xyzw, **metadata):
    stamp_ns = int(stamp_ns)
    if stamp_ns in seen_stamps:
        return False
    sample = {
        "stamp_ns": stamp_ns,
        "position": [float(value) for value in position],
        "orientation_xyzw": [float(value) for value in orientation_xyzw],
    }
    sample.update(metadata)
    seen_stamps.add(stamp_ns)
    samples.append(sample)
    return True


def robust_position_summary(samples, mad_scale=3.5):
    if not samples:
        raise RuntimeError("No TF samples were collected.")
    axes = list(zip(*[sample["position"] for sample in samples]))
    centers = [median(axis) for axis in axes]
    mads = [median_absolute_deviation(axis, center)
            for axis, center in zip(axes, centers)]
    inliers = []
    for sample in samples:
        keep = True
        for value, center, axis_mad in zip(
                sample["position"], centers, mads):
            limit = max(float(axis_mad) * float(mad_scale), 1e-6)
            if abs(float(value) - center) > limit:
                keep = False
                break
        if keep:
            inliers.append(sample)
    if not inliers:
        raise RuntimeError("All TF samples were rejected as outliers.")
    inlier_axes = list(zip(*[sample["position"] for sample in inliers]))
    return {
        "sample_count": len(samples),
        "inlier_count": len(inliers),
        "median_position_m": [median(axis) for axis in inlier_axes],
        "position_mad_m": [
            median_absolute_deviation(axis) for axis in inlier_axes
        ],
        "inlier_range_m": [
            max(axis) - min(axis) for axis in inlier_axes
        ],
    }


def validate_camera_info(width, height, intrinsic_k,
                         expected_width=640, expected_height=480):
    if int(width) != int(expected_width) or int(height) != int(expected_height):
        raise RuntimeError(
            "CameraInfo resolution is %dx%d; expected %dx%d."
            % (width, height, expected_width, expected_height))
    if len(intrinsic_k) != 9:
        raise RuntimeError("CameraInfo K must contain 9 values.")
    fx = float(intrinsic_k[0])
    fy = float(intrinsic_k[4])
    if not is_finite(fx) or not is_finite(fy) or fx <= 0.0 or fy <= 0.0:
        raise RuntimeError("CameraInfo fx/fy must be finite and positive.")
    return True


def depth_region_stats(values_m, min_depth_m=0.1, max_depth_m=2.0):
    values = list(values_m)
    valid = [
        float(value) for value in values
        if is_finite(value) and
        float(min_depth_m) <= float(value) <= float(max_depth_m)
    ]
    result = {
        "total_count": len(values),
        "valid_count": len(valid),
        "valid_ratio": (float(len(valid)) / len(values)) if values else 0.0,
        "median_m": None,
        "mad_m": None,
    }
    if valid:
        result["median_m"] = median(valid)
        result["mad_m"] = median_absolute_deviation(
            valid, result["median_m"])
    return result


def stamp_to_ns(stamp):
    return int(stamp.secs) * 1000000000 + int(stamp.nsecs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure AprilTag TF and Mirobot endpoint repeatability.")
    parser.add_argument(
        "--mode", choices=["tag_static", "arm_repeatability", "depth_check"],
        default="tag_static")
    parser.add_argument("--base-frame", default="base")
    parser.add_argument("--target-frame", default="tag_1")
    parser.add_argument("--tag-id", type=int, default=1)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--camera-info-topic", default="/camera/rgb/camera_info")
    parser.add_argument("--detections-topic",
                        default="/tag_yolo_quiet/detections_json")
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument(
        "--depth-topic", default="/camera/depth_registered/image_raw")
    parser.add_argument("--expected-width", type=int, default=640)
    parser.add_argument("--expected-height", type=int, default=480)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


class RosDiagnosticsCollector(object):
    def __init__(self, args, modules):
        self.args = args
        self.rospy = modules["rospy"]
        self.tf = modules["tf"]
        self.Image = modules["Image"]
        self.CameraInfo = modules["CameraInfo"]
        self.JointState = modules["JointState"]
        self.String = modules["String"]
        self.listener = self.tf.TransformListener()
        self.camera_info = None
        self.detections = None
        self.joint_state = None
        self.depth_image = None
        self.rospy.Subscriber(
            args.camera_info_topic, self.CameraInfo,
            self._camera_info_callback, queue_size=1)
        self.rospy.Subscriber(
            args.detections_topic, self.String,
            self._detections_callback, queue_size=1)
        self.rospy.Subscriber(
            args.joint_states_topic, self.JointState,
            self._joint_state_callback, queue_size=1)
        if args.mode == "depth_check":
            self.rospy.Subscriber(
                args.depth_topic, self.Image,
                self._depth_callback, queue_size=1)

    def _camera_info_callback(self, message):
        self.camera_info = message

    def _detections_callback(self, message):
        try:
            self.detections = json.loads(message.data)
        except (TypeError, ValueError):
            self.rospy.logwarn_throttle(
                2.0, "Ignoring malformed detections JSON.")

    def _joint_state_callback(self, message):
        self.joint_state = message

    def _depth_callback(self, message):
        self.depth_image = message

    def _selected_detection(self):
        if not self.detections:
            return None
        matches = [
            detection for detection in self.detections.get("detections", [])
            if int(detection.get("tag_id", -1)) == self.args.tag_id
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: float(item.get("confidence", 0.0)))

    def _joint_snapshot(self):
        if self.joint_state is None:
            return {}
        return {
            "joint_stamp_ns": stamp_to_ns(self.joint_state.header.stamp),
            "joint_names": list(self.joint_state.name),
            "joint_positions": [float(value)
                                for value in self.joint_state.position],
        }

    def collect_window(self, trial):
        samples = []
        seen_stamps = set()
        deadline = self.rospy.Time.now() + self.rospy.Duration(
            self.args.duration)
        rate = self.rospy.Rate(30)
        while not self.rospy.is_shutdown() and self.rospy.Time.now() < deadline:
            try:
                common_time = self.listener.getLatestCommonTime(
                    self.args.base_frame, self.args.target_frame)
                translation, rotation = self.listener.lookupTransform(
                    self.args.base_frame, self.args.target_frame, common_time)
            except (self.tf.LookupException,
                    self.tf.ConnectivityException,
                    self.tf.ExtrapolationException):
                rate.sleep()
                continue
            detection = self._selected_detection()
            metadata = self._joint_snapshot()
            metadata["trial"] = int(trial)
            metadata["detection"] = detection
            append_unique_tf_sample(
                samples, seen_stamps, stamp_to_ns(common_time),
                translation, rotation, **metadata)
            rate.sleep()
        if len(samples) < self.args.min_samples:
            raise RuntimeError(
                "Only %d unique TF samples were collected; need at least %d."
                % (len(samples), self.args.min_samples))
        return samples

    def camera_info_snapshot(self):
        if self.camera_info is None:
            raise RuntimeError(
                "No CameraInfo received from %s."
                % self.args.camera_info_topic)
        validate_camera_info(
            self.camera_info.width, self.camera_info.height,
            self.camera_info.K,
            self.args.expected_width, self.args.expected_height)
        return {
            "width": int(self.camera_info.width),
            "height": int(self.camera_info.height),
            "K": [float(value) for value in self.camera_info.K],
            "D": [float(value) for value in self.camera_info.D],
            "distortion_model": self.camera_info.distortion_model,
        }

    def depth_snapshot(self):
        if self.depth_image is None:
            return None
        detection = self._selected_detection()
        if not detection:
            return None
        try:
            import numpy
        except ImportError:
            raise RuntimeError("depth_check requires numpy.")
        image = self.depth_image
        if image.encoding == "16UC1":
            array = numpy.frombuffer(image.data, dtype=numpy.uint16)
            scale = 0.001
        elif image.encoding == "32FC1":
            array = numpy.frombuffer(image.data, dtype=numpy.float32)
            scale = 1.0
        else:
            raise RuntimeError(
                "Unsupported depth encoding: %s" % image.encoding)
        array = array.reshape((image.height, image.width))
        x1, y1, x2, y2 = [
            int(round(value)) for value in detection["box"]
        ]
        x1 = max(0, min(image.width - 1, x1))
        x2 = max(x1 + 1, min(image.width, x2))
        y1 = max(0, min(image.height - 1, y1))
        y2 = max(y1 + 1, min(image.height, y2))
        values_m = array[y1:y2, x1:x2].astype(
            numpy.float64).reshape(-1) * scale
        return depth_region_stats(values_m)


def write_results(args, camera_info, samples, trial_summaries, depth_stats):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(
        args.output_dir, "%s_%s" % (args.mode, timestamp))
    payload = {
        "mode": args.mode,
        "base_frame": args.base_frame,
        "target_frame": args.target_frame,
        "camera_info": camera_info,
        "trial_summaries": trial_summaries,
        "depth_stats": depth_stats,
        "samples": samples,
    }
    with open(prefix + ".json", "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    with open(prefix + ".csv", "w") as output:
        writer = csv.writer(output)
        writer.writerow([
            "trial", "stamp_ns", "x_m", "y_m", "z_m",
            "qx", "qy", "qz", "qw", "box", "joint_positions"
        ])
        for sample in samples:
            writer.writerow([
                sample["trial"], sample["stamp_ns"],
                sample["position"][0], sample["position"][1],
                sample["position"][2],
                sample["orientation_xyzw"][0],
                sample["orientation_xyzw"][1],
                sample["orientation_xyzw"][2],
                sample["orientation_xyzw"][3],
                json.dumps((sample.get("detection") or {}).get("box")),
                json.dumps(sample.get("joint_positions")),
            ])
    return prefix


def main(argv=None):
    args = parse_args(argv)
    import rospy
    import tf
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from std_msgs.msg import String
    modules = {
        "rospy": rospy,
        "tf": tf,
        "CameraInfo": CameraInfo,
        "Image": Image,
        "JointState": JointState,
        "String": String,
    }
    rospy.init_node("tag_arm_diagnostics", anonymous=True)
    collector = RosDiagnosticsCollector(args, modules)
    rospy.sleep(1.0)
    camera_info = collector.camera_info_snapshot()
    all_samples = []
    summaries = []
    trials = args.trials if args.mode == "arm_repeatability" else 1
    for trial in range(1, trials + 1):
        if args.mode == "arm_repeatability":
            prompt = (
                "Move the arm to endpoint trial %d/%d, wait until it is idle, "
                "then press Enter to sample..." % (trial, trials))
            try:
                raw_input(prompt)
            except NameError:
                input(prompt)
        samples = collector.collect_window(trial)
        all_samples.extend(samples)
        summary = robust_position_summary(samples)
        summary["trial"] = trial
        summaries.append(summary)
        rospy.loginfo(
            "Trial %d: inliers=%d/%d ranges_mm=[%.2f, %.2f, %.2f]",
            trial, summary["inlier_count"], summary["sample_count"],
            summary["inlier_range_m"][0] * 1000.0,
            summary["inlier_range_m"][1] * 1000.0,
            summary["inlier_range_m"][2] * 1000.0)
    depth_stats = collector.depth_snapshot()
    prefix = write_results(
        args, camera_info, all_samples, summaries, depth_stats)
    rospy.loginfo("Diagnostics written to %s.json and %s.csv", prefix, prefix)


if __name__ == "__main__":
    main()
