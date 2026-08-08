from types import SimpleNamespace

import pytest

import block_distance_collect as collect


def test_parse_targets_and_distances():
    assert collect.parse_targets("fire,power") == ["fire", "power"]
    assert collect.parse_distances("280,300,320") == [280, 300, 320]
    with pytest.raises(Exception):
        collect.parse_targets("unknown")
    with pytest.raises(Exception):
        collect.parse_distances("280,300")


def test_default_collection_uses_ten_frames():
    args = collect.parse_args([])

    assert args.frames == 10
    assert args.distances == [
        340, 360, 370, 380, 390, 400,
        410, 420, 430, 440, 460,
    ]
    assert args.output_dir.endswith(
        "block_distance_samples_occlusion640_400")
    assert not args.overwrite


def test_collection_schedule_finishes_all_targets_before_next_distance():
    assert collect.sample_schedule(
        ["power", "fire", "gas", "support"], [280, 300]
    ) == [
        ("power", 280),
        ("fire", 280),
        ("gas", 280),
        ("support", 280),
        ("power", 300),
        ("fire", 300),
        ("gas", 300),
        ("support", 300),
    ]


def test_build_collect_command_contains_target_distance_and_frames(monkeypatch):
    monkeypatch.setattr(collect.sys, "executable", "/env/bin/python3")
    args = SimpleNamespace(
        block_pick_main="/src/block_pick_main.py",
        frames=20,
        confidence=0.5,
        config="/config/grasp.yaml",
    )

    command = collect.build_collect_command(args, "fire", 320)

    assert command[:2] == ["/env/bin/python3", "/src/block_pick_main.py"]
    assert command[command.index("--target") + 1] == "fire"
    assert command[command.index("--known-z-mm") + 1] == "320"
    assert command[command.index("--frames") + 1] == "20"


def test_existing_sample_path_is_deterministic(tmp_path):
    assert collect.sample_path(str(tmp_path), "gas", 340) == str(
        tmp_path / "gas_340.csv")


def test_run_and_tee_only_keeps_successful_output(tmp_path):
    success = tmp_path / "success.csv"
    failed = tmp_path / "failed.csv"

    assert collect.run_and_tee(
        [collect.sys.executable, "-c", "print('sample')"], str(success)) == 0
    assert success.read_text().strip() == "sample"

    assert collect.run_and_tee(
        [collect.sys.executable, "-c", "import sys; print('bad'); sys.exit(2)"],
        str(failed),
    ) == 2
    assert not failed.exists()
    assert not (tmp_path / "failed.csv.tmp").exists()


def test_main_preserves_existing_samples_and_includes_them_in_fit(
        tmp_path, monkeypatch, capsys):
    paths = []
    original_contents = {}
    for distance in (280, 300, 320):
        path = tmp_path / ("power_%d.csv" % distance)
        contents = "existing sample at %d\n" % distance
        path.write_text(contents)
        paths.append(str(path))
        original_contents[str(path)] = contents

    fitted = []
    monkeypatch.setattr(
        collect,
        "prompt_sample",
        lambda *_args: pytest.fail("existing samples must not be prompted again"),
    )
    monkeypatch.setattr(
        collect,
        "fit_target",
        lambda _args, target, sample_paths: fitted.append(
            (target, list(sample_paths))),
    )

    assert collect.main([
        "--targets", "power",
        "--distances", "280,300,320",
        "--output-dir", str(tmp_path),
    ]) == 0

    assert fitted == [("power", paths)]
    assert "续采模式：保留并跳过 3 个已有样本" in capsys.readouterr().out
    for path, contents in original_contents.items():
        assert open(path).read() == contents
