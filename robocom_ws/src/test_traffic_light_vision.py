#!/usr/bin/env python3
# coding=utf-8
import os
import tempfile
import unittest

import cv2
import numpy as np

import traffic_light_vision as traffic
import traffic_light_test as traffic_test


class FakeDetection(object):
    def __init__(self, class_name, confidence=0.9):
        self.class_name = class_name
        self.confidence = confidence


class FakeNet(object):
    def __init__(self, class_id=0):
        self.class_id = class_id
        self.input_blob = None

    def setPreferableBackend(self, backend):
        pass

    def setPreferableTarget(self, target):
        pass

    def setInput(self, blob):
        self.input_blob = blob

    def forward(self):
        row = np.zeros((1, 8), dtype=np.float32)
        row[0, :5] = [160.0, 160.0, 80.0, 60.0, 0.9]
        row[0, 5 + self.class_id] = 0.9
        return row.reshape(1, 1, 8)


class TrafficLightVisionTests(unittest.TestCase):
    def test_visual_test_script_uses_camera_zero_and_required_model(self):
        self.assertEqual(traffic_test.CAMERA_INDEX, 0)
        self.assertEqual((traffic_test.FRAME_WIDTH,
                          traffic_test.FRAME_HEIGHT), (320, 240))
        self.assertEqual(
            traffic_test.MODEL_PATH,
            "/home/eaibot/handeye-calib/src/model/yolov5/"
            "traffic_lights_yolov5n_320_best.onnx",
        )

    def test_camera_command_matches_required_controls(self):
        calls = []

        command = traffic.configure_traffic_camera(
            0, runner=lambda value: calls.append(value)
        )

        expected = [
            "v4l2-ctl", "-d", "/dev/video0",
            "-c", "exposure_auto=1",
            "-c", "exposure_absolute=15",
            "-c", "white_balance_temperature_auto=0",
            "-c", "white_balance_temperature=4600",
            "-c", "exposure_auto_priority=0",
        ]
        self.assertEqual(command, expected)
        self.assertEqual(calls, [expected])

    def test_debug_view_displays_configured_exposure(self):
        texts = []
        original_put_text = cv2.putText
        cv2.putText = lambda image, text, *args, **kwargs: (
            texts.append(text) or image
        )
        try:
            traffic.draw_traffic_light(
                np.zeros((240, 320, 3), dtype=np.uint8), []
            )
        finally:
            cv2.putText = original_put_text

        self.assertIn("exposure_absolute=15", texts)

    def test_green_requires_consecutive_frames_and_red_resets_hits(self):
        green = [FakeDetection("Green")]
        red = [FakeDetection("Red")]

        hits, ready, color = traffic.update_green_hits(green, 0, 2)
        self.assertEqual((hits, ready, color), (1, False, "Green"))
        hits, ready, color = traffic.update_green_hits(red, hits, 2)
        self.assertEqual((hits, ready, color), (0, False, "Red"))
        hits, ready, _ = traffic.update_green_hits(green, hits, 2)
        hits, ready, color = traffic.update_green_hits(green, hits, 2)
        self.assertEqual((hits, ready, color), (2, True, "Green"))

    def test_model_is_loaded_lazily_and_close_releases_it(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "traffic.onnx")
            open(path, "wb").close()
            detector = traffic.TrafficLightDetector(path)
            original_read = cv2.dnn.readNetFromONNX
            calls = []
            cv2.dnn.readNetFromONNX = lambda value: calls.append(value) or FakeNet()
            try:
                self.assertFalse(detector.loaded)
                self.assertEqual(calls, [])
                detector.load()
                self.assertTrue(detector.loaded)
                detector.close()
                self.assertFalse(detector.loaded)
            finally:
                cv2.dnn.readNetFromONNX = original_read
        self.assertEqual(calls, [path])

    def test_yolov5_output_decodes_green_detection(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "traffic.onnx")
            open(path, "wb").close()
            detector = traffic.TrafficLightDetector(path, confidence=0.5)
            detector.model = FakeNet(class_id=0)

            detections = detector.detect(
                np.zeros((240, 320, 3), dtype=np.uint8)
            )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "Green")
        self.assertAlmostEqual(detections[0].confidence, 0.81, places=4)


if __name__ == "__main__":
    unittest.main()
