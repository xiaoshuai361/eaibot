import sys

import capture_images


def parse_args(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["capture_images.py", *args])
    return capture_images.parse_args()


def test_resolution_sets_width_and_height(monkeypatch):
    args = parse_args(monkeypatch, "--resolution", "320")

    assert args.width == 320
    assert args.height == 320


def test_width_and_height_override_resolution(monkeypatch):
    args = parse_args(
        monkeypatch,
        "--resolution",
        "320",
        "--width",
        "640",
        "--height",
        "480",
    )

    assert args.width == 640
    assert args.height == 480
