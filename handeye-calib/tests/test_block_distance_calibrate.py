import csv

import pytest

from block_distance_calibrate import (
    aggregate_samples_by_distance,
    fit_axis,
    read_samples,
)


def write_samples(path, target="fire"):
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["known_z_mm", "target", "conf", "x1", "y1", "x2",
                         "y2", "u", "v", "w", "h"])
        for distance in (280.0, 320.0, 360.0):
            for _repeat in range(2):
                writer.writerow([
                    distance, target, 0.9, 0, 0, 1, 1, 10, 10,
                    18000.0 / distance, 24000.0 / distance,
                ])


def test_distance_calibration_recovers_width_and_height_models(tmp_path):
    path = tmp_path / "samples.csv"
    write_samples(path)

    samples = read_samples([str(path)], "fire")
    width = fit_axis(samples, 1)
    height = fit_axis(samples, 2)

    assert width["a"] == pytest.approx(18000.0)
    assert width["b"] == pytest.approx(0.0, abs=1e-8)
    assert height["a"] == pytest.approx(24000.0)
    assert height["b"] == pytest.approx(0.0, abs=1e-8)


def test_distance_calibration_requires_three_distances(tmp_path):
    path = tmp_path / "samples.csv"
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        for _repeat in range(6):
            writer.writerow([300, "fire", 0.9, 0, 0, 1, 1, 10, 10, 60, 80])

    with pytest.raises(RuntimeError, match="3 different distances"):
        read_samples([str(path)], "fire")


def test_distance_calibration_uses_per_distance_median():
    samples = [
        (280.0, 64.0, 70.0),
        (280.0, 65.0, 71.0),
        (280.0, 200.0, 220.0),
        (320.0, 56.0, 62.0),
        (320.0, 57.0, 63.0),
        (320.0, 5.0, 6.0),
    ]

    assert aggregate_samples_by_distance(samples) == pytest.approx([
        (280.0, 65.0, 71.0),
        (320.0, 56.0, 62.0),
    ])
