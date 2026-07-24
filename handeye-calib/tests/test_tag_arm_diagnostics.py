from __future__ import absolute_import

import math

import pytest

from tag_arm_diagnostics import (
    append_unique_tf_sample,
    depth_region_stats,
    robust_position_summary,
    validate_camera_info,
)


def test_duplicate_tf_timestamps_are_not_counted_twice():
    samples = []
    seen = set()

    assert append_unique_tf_sample(
        samples, seen, 100, [0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0])
    assert not append_unique_tf_sample(
        samples, seen, 100, [9.0, 9.0, 9.0], [0.0, 0.0, 0.0, 1.0])
    assert len(samples) == 1
    assert samples[0]["position"] == [0.1, 0.2, 0.3]


def test_robust_position_summary_rejects_large_outlier():
    samples = [
        {"position": [0.100, 0.200, 0.300]},
        {"position": [0.101, 0.199, 0.301]},
        {"position": [0.099, 0.201, 0.299]},
        {"position": [0.500, -0.300, 1.200]},
    ]

    summary = robust_position_summary(samples, mad_scale=3.5)

    assert summary["sample_count"] == 4
    assert summary["inlier_count"] == 3
    assert summary["median_position_m"] == pytest.approx([0.100, 0.200, 0.300])
    assert max(summary["inlier_range_m"]) <= 0.0021


def test_camera_info_requires_valid_640x480_intrinsics():
    valid_k = [520.0, 0.0, 320.0, 0.0, 521.0, 240.0, 0.0, 0.0, 1.0]

    assert validate_camera_info(640, 480, valid_k, 640, 480) is True
    with pytest.raises(RuntimeError, match="fx/fy"):
        validate_camera_info(640, 480, [0.0] * 9, 640, 480)
    with pytest.raises(RuntimeError, match="resolution"):
        validate_camera_info(1280, 720, valid_k, 640, 480)


def test_depth_stats_report_valid_ratio_median_and_mad():
    values = [0.40, 0.401, float("nan"), 0.0, 0.399, 3.0]

    stats = depth_region_stats(values, min_depth_m=0.1, max_depth_m=2.0)

    assert stats["total_count"] == 6
    assert stats["valid_count"] == 3
    assert stats["valid_ratio"] == pytest.approx(0.5)
    assert stats["median_m"] == pytest.approx(0.4)
    assert stats["mad_m"] == pytest.approx(0.001)
