# Tagless Taught Block Mono Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a no-Tag teaching workflow that records suction pose relative to a YOLO monocular block anchor, then replays it when the block moves.

**Architecture:** Keep Python3 as the ONNX detector parent and Python2 as the ROS/MoveIt worker. The block anchor is a pose in `base`: position from YOLO monocular localization, orientation from `block_anchor_orientation_xyzw` or identity. Teaching stores `grasp_ee_in_block` plus fixed `place_ee_in_base`; replay recomputes the current anchor and applies the stored transform.

**Tech Stack:** ROS Melodic Python2, Python3 ONNX parent, MoveIt, `block_mono_vision.py`, JSON presets.

## Global Constraints

- Do not move hardware unless the selected action explicitly requires motion.
- All no-Tag taught motion requires an explicit `--target`.
- `--dry-run` without target remains all-detection debug only.
- Presets default to `/home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json`.

---

### Task 1: Core transform helpers and preset I/O

**Files:**
- Modify: `/home/zcy/eaibot/handeye-calib/src/mirobot_pick_test.py`
- Test: `/home/zcy/eaibot/handeye-calib/tests/test_mirobot_block_mono.py`

**Interfaces:**
- Produces: `compute_grasp_ee_in_block(anchor_pose, ee_pose) -> dict`
- Produces: `compute_taught_grasp_pose(anchor_pose, grasp_ee_in_block, base_frame) -> PoseStamped`
- Produces: `save_block_preset(path, preset, overwrite=False)` and `load_block_preset(path)`

- [ ] Write failing tests for relative transform replay and JSON round-trip.
- [ ] Implement matrix helpers copied in simplified form from `mirobot_pick_test_tag.py`.
- [ ] Run targeted tests.

### Task 2: Python2 teach/replay actions

**Files:**
- Modify: `/home/zcy/eaibot/handeye-calib/src/mirobot_pick_test.py`
- Test: `/home/zcy/eaibot/handeye-calib/tests/test_mirobot_block_mono.py`

**Interfaces:**
- Produces actions `teach_block` and `run_taught_block` inside `do_block_mono`.
- Consumes existing `compute_block_localization()`.

- [ ] Write failing tests for action dispatch and preset shape.
- [ ] Implement `teach_block_mono`: localize, move to pre-grasp assist, prompt, record current EE grasp and place pose, save preset.
- [ ] Implement `run_taught_block_mono`: localize current anchor, compute taught grasp/pre-grasp/place/pre-place, execute or log.
- [ ] Run targeted tests.

### Task 3: Python3 CLI forwarding

**Files:**
- Modify: `/home/zcy/eaibot/handeye-calib/src/block_pick_main.py`
- Test: `/home/zcy/eaibot/handeye-calib/tests/test_block_pick_main.py`

**Interfaces:**
- Adds actions `--teach-block` and `--run-taught-block`.
- Adds options `--preset-file` and `--overwrite`.

- [ ] Write failing tests for CLI validation and child command forwarding.
- [ ] Implement parser, validation, and command forwarding.
- [ ] Run targeted tests.

### Task 4: Verification and handoff

**Files:**
- Modify docs only if commands change in `/home/zcy/eaibot/zcy/机械臂操作.md`.

- [ ] Run Python compile check.
- [ ] Run all `handeye-calib/tests`.
- [ ] Report exact copy-to-robot files and example commands.
