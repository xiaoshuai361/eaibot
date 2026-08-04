from __future__ import absolute_import

import json
from types import SimpleNamespace

import pytest

from test_mirobot_pick_test_tag_taught_sequence import (
    load_module_symbols,
    make_pose,
)


def v3_preset():
    return {
        "version": 3,
        "base_frame": "base",
        "camera_frame": "camera_rgb_optical_frame",
        "pickup_model": {
            "orientation_xyzw_base": [0.1, -0.2, 0.3, 0.9],
            "approach_axis_xyz_base": [-1.0, 0.0, 0.0],
        },
        "tags": {
            "1": {
                "grasp_offset_xyz_base": [-0.027, 0.004, -0.012],
                "place_ee_in_base": {
                    "position": [0.12, -0.14, 0.05],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
    }


def test_old_preset_is_rejected_for_runtime(tmp_path):
    load_preset, = load_module_symbols("load_preset")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "version": 1,
        "tags": {"1": {"grasp_ee_in_tag": {}, "place_ee_in_base": {}}},
    }), encoding="utf-8")

    with pytest.raises(RuntimeError, match="version 3"):
        load_preset(str(path))


def test_legacy_migration_preserves_places_but_requires_new_grasp_teaching():
    migrate, = load_module_symbols("migrate_legacy_preset_for_teach")
    legacy = {
        "version": 1,
        "base_frame": "base",
        "camera_frame": "camera",
        "idle_joint_values": [0.1, 0.2],
        "tags": {
            "1": {
                "grasp_ee_in_tag": {"position": [9, 9, 9]},
                "place_ee_in_base": {
                    "position": [0.12, -0.14, 0.05],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
            },
        },
    }

    migrated = migrate(legacy)

    assert migrated["version"] == 3
    assert migrated["idle_joint_values"] == [0.1, 0.2]
    assert migrated["tags"]["1"]["place_ee_in_base"] == (
        legacy["tags"]["1"]["place_ee_in_base"])
    assert "grasp_ee_in_tag" not in migrated["tags"]["1"]
    assert "grasp_offset_xyz_base" not in migrated["tags"]["1"]


def test_version_2_migration_discards_fixed_z_grasp_but_preserves_place():
    migrate, = load_module_symbols("migrate_legacy_preset_for_teach")
    legacy = v3_preset()
    legacy["version"] = 2
    legacy["pickup_model"]["contact_z_base"] = 0.108
    legacy["tags"]["1"]["grasp_offset_xy_base"] = [-0.027, 0.004]

    migrated = migrate(legacy)

    assert migrated["version"] == 3
    assert "pickup_model" not in migrated
    assert "grasp_offset_xyz_base" not in migrated["tags"]["1"]
    assert migrated["tags"]["1"]["place_ee_in_base"] == (
        legacy["tags"]["1"]["place_ee_in_base"])


def test_constrained_grasp_uses_tag_xyz_but_fixed_orientation():
    compute, = load_module_symbols("compute_constrained_grasp_pose")
    preset = v3_preset()
    root_half = 2 ** 0.5 / 2.0
    noisy_rotated_tag = make_pose(
        0.25, 0.10, 0.22,
        q=[0.0, 0.0, root_half, root_half])

    grasp = compute(
        noisy_rotated_tag, preset["pickup_model"],
        preset["tags"]["1"], "base")

    assert grasp.pose.position.x == pytest.approx(0.223)
    assert grasp.pose.position.y == pytest.approx(0.104)
    assert grasp.pose.position.z == pytest.approx(0.208)
    norm = (0.1 ** 2 + 0.2 ** 2 + 0.3 ** 2 + 0.9 ** 2) ** 0.5
    assert [
        grasp.pose.orientation.x,
        grasp.pose.orientation.y,
        grasp.pose.orientation.z,
        grasp.pose.orientation.w,
    ] == pytest.approx([0.1 / norm, -0.2 / norm, 0.3 / norm, 0.9 / norm])


def test_constrained_pregrasp_uses_fixed_base_approach_axis():
    compute, build_pre = load_module_symbols(
        "compute_constrained_grasp_pose",
        "build_constrained_pre_grasp_pose")
    preset = v3_preset()
    grasp = compute(
        make_pose(0.25, 0.10, 0.2),
        preset["pickup_model"], preset["tags"]["1"], "base")

    pre = build_pre(
        grasp, preset["pickup_model"], approach_gap=0.03,
        base_frame="base")

    assert pre.pose.position.x == pytest.approx(grasp.pose.position.x - 0.03)
    assert pre.pose.position.y == pytest.approx(grasp.pose.position.y)
    assert pre.pose.position.z == pytest.approx(grasp.pose.position.z)


def test_tag_z_translation_moves_grasp_z_by_same_amount():
    compute, = load_module_symbols("compute_constrained_grasp_pose")
    preset = v3_preset()

    low = compute(
        make_pose(0.25, 0.10, 0.10),
        preset["pickup_model"], preset["tags"]["1"], "base")
    high = compute(
        make_pose(0.25, 0.10, 0.13),
        preset["pickup_model"], preset["tags"]["1"], "base")

    assert high.pose.position.z - low.pose.position.z == pytest.approx(0.03)
    assert high.pose.orientation.x == pytest.approx(low.pose.orientation.x)
    assert high.pose.orientation.y == pytest.approx(low.pose.orientation.y)
    assert high.pose.orientation.z == pytest.approx(low.pose.orientation.z)
    assert high.pose.orientation.w == pytest.approx(low.pose.orientation.w)


def test_tag_filter_deduplicates_timestamps_and_rejects_outlier():
    append_sample, filter_samples = load_module_symbols(
        "append_unique_tag_sample", "filter_tag_translation_samples")
    samples = []
    seen = set()
    rotations = [0.0, 0.0, 0.0, 1.0]

    assert append_sample(samples, seen, 1, [0.200, 0.100, 0.120], rotations)
    assert not append_sample(samples, seen, 1, [9.0, 9.0, 9.0], rotations)
    append_sample(samples, seen, 2, [0.201, 0.099, 0.121], rotations)
    append_sample(samples, seen, 3, [0.199, 0.101, 0.119], rotations)
    append_sample(samples, seen, 4, [0.500, -0.300, 0.800], rotations)

    filtered = filter_samples(
        samples, min_samples=3, mad_scale=3.5, max_axis_mad_m=0.005)

    assert filtered["position"] == pytest.approx([0.200, 0.100, 0.120])
    assert filtered["sample_count"] == 4
    assert filtered["inlier_count"] == 3


def test_tag_filter_accepts_two_low_rate_samples():
    filter_samples, = load_module_symbols("filter_tag_translation_samples")
    samples = [
        {"stamp_ns": 1, "position": [0.200, 0.100, 0.120],
         "orientation_xyzw": [0, 0, 0, 1]},
        {"stamp_ns": 2, "position": [0.202, 0.098, 0.122],
         "orientation_xyzw": [0, 0, 0, 1]},
    ]

    filtered = filter_samples(
        samples, min_samples=2, mad_scale=3.5, max_axis_mad_m=0.005)

    assert filtered["position"] == pytest.approx([0.201, 0.099, 0.121])
    assert filtered["sample_count"] == 2
    assert filtered["inlier_count"] == 2


def test_tag_filter_rejects_unstable_translation():
    filter_samples, = load_module_symbols("filter_tag_translation_samples")
    samples = [
        {"stamp_ns": index, "position": [0.1 + index * 0.01, 0.2, 0.3],
         "orientation_xyzw": [0, 0, 0, 1]}
        for index in range(10)
    ]

    with pytest.raises(RuntimeError, match="unstable"):
        filter_samples(
            samples, min_samples=10, mad_scale=20.0,
            max_axis_mad_m=0.003)
