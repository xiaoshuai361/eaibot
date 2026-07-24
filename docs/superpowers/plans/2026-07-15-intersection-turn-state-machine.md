# Intersection Turn State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the old left/right intersection turn phases in `line_cy_new.py` while preserving its two-stage PD and overall maneuver timeout recovery.

**Architecture:** Straight intersections keep the existing dual-line bridge. Left and right intersections use the old `ENTRY -> TURN -> EXIT_APPROACH` sub-state machine, then reuse the existing exit-bar alignment flow. Only the six turn tuning values and their supporting state are added.

**Tech Stack:** Python, ROS `rospy`, OpenCV, `unittest`.

---

### Task 1: Restore Turn Decisions

**Files:**
- Modify: `robocom_ws/src/line_cy_new.py`
- Test: `robocom_ws/src/test_line_cy_new.py`

- [ ] Run the existing tests for the six tuning constants, fixed turn direction, capture delay, lane validation, and phase transitions; confirm they fail because the restored file lacks the old APIs.
- [ ] Add `TURN_ENTRY_TIME`, `TURN_SPEED`, `TURN_ANGULAR`, `TURN_CAPTURE_DELAY`, `TURN_CAPTURE_FRAMES`, and `TURN_TIMEOUT` beside the motion constants.
- [ ] Add `fixed_turn_command`, `turn_lane_capture_valid`, `turn_side_capture_valid`, `turn_side_control_target`, `update_turn_capture_hits`, and `turn_phase_next` from `line_cy_old.py`.
- [ ] Re-run the focused decision tests and confirm they pass.

### Task 2: Restore Runtime Phases

**Files:**
- Modify: `robocom_ws/src/line_cy_new.py`
- Test: `robocom_ws/src/test_line_cy_new.py`

- [ ] Load the six values from ROS parameters and initialize maneuver phase state.
- [ ] Allow fixed-turn angular velocity to exceed normal-follow `MAX_ANGULAR` only during the `TURN` phase.
- [ ] Add `_set_maneuver_phase` and initialize `ENTRY` for left/right or `STRAIGHT` for straight intersections.
- [ ] Replace the current left/right side-only maneuver branch with the old `ENTRY`, `TURN`, and `EXIT_APPROACH` flow.
- [ ] Keep `maneuver_timeout_exits_to_follow` as the final 20-second fallback and start the exit re-entry guard when it returns to `FOLLOW`.

### Task 3: Verify

**Files:**
- Verify: `robocom_ws/src/line_cy_new.py`
- Verify: `robocom_ws/src/test_line_cy_new.py`

- [ ] Run `python3 -m py_compile line_cy_new.py test_line_cy_new.py`.
- [ ] Run focused turn, PD, exit-bar, timeout, and re-entry guard tests.
- [ ] Inspect the final constants and state transitions to ensure no unrelated old-code features were copied.
