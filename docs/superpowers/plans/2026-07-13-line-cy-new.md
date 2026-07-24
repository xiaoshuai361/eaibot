# line_cy_new Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a smaller standalone ROS lane follower with center-out lane selection, partial stop-bar detection, and straight-through right-line RANSAC bridging.

**Architecture:** Keep pure vision and geometry functions testable without ROS, then wrap them in a compact five-state controller. Use deterministic point-pair RANSAC implemented with NumPy so deployment adds no dependencies.

**Tech Stack:** Python 3 compatible syntax, ROS1 `rospy`, OpenCV, NumPy, `unittest`.

---

### Task 1: Pure lane geometry

**Files:**
- Create: `robocom_ws/src/test_line_cy_new.py`
- Create: `robocom_ws/src/line_cy_new.py`

- [ ] Add failing tests for center-out first segments, external-noise rejection, dual center, and single-side reconstruction.
- [ ] Run `python3 -m unittest test_line_cy_new.LaneGeometryTests -v` and confirm failures are caused by missing implementation.
- [ ] Implement row segmentation and lane observation with a small result dictionary.
- [ ] Re-run the lane geometry tests.

### Task 2: Crosswalk and partial stop bar

**Files:**
- Modify: `robocom_ws/src/test_line_cy_new.py`
- Modify: `robocom_ws/src/line_cy_new.py`

- [ ] Add failing tests for grouped stripes plus a partially visible connected stop bar and for a lone diagonal lane rejection.
- [ ] Implement rotated-rectangle stripe grouping and local Hough stop-bar matching.
- [ ] Run the focused crosswalk tests and retain the detected polygons for masking/debugging.

### Task 3: Straight bridge and turn-side reconstruction

**Files:**
- Modify: `robocom_ws/src/test_line_cy_new.py`
- Modify: `robocom_ws/src/line_cy_new.py`

- [ ] Add failing tests where straight right-edge points coexist with inward-curving outliers.
- [ ] Implement deterministic RANSAC `x = a*y + b`, least-squares inlier refit, model continuity, and center projection.
- [ ] Add tests for left-turn left-edge and right-turn right-edge reconstruction.
- [ ] Run the focused bridge tests.

### Task 4: ROS controller and state transitions

**Files:**
- Modify: `robocom_ws/src/test_line_cy_new.py`
- Modify: `robocom_ws/src/line_cy_new.py`

- [ ] Add pure transition tests for entry confirmation, approach, alignment, maneuver exit by stop bar, fallback exit by restored dual lanes, timeout, and dry-run.
- [ ] Implement camera reader, PID, command publishing, controller loop, and debug view.
- [ ] Keep only the approved top-level tuning parameters and `~turn_cmd`.

### Task 5: Verification

**Files:**
- Verify: `robocom_ws/src/line_cy_new.py`
- Verify: `robocom_ws/src/test_line_cy_new.py`

- [ ] Run `python3 -m unittest test_line_cy_new -v`.
- [ ] Run `python3 -m py_compile line_cy_new.py test_line_cy_new.py`.
- [ ] Run `git diff --check -- robocom_ws/src/line_cy_new.py robocom_ws/src/test_line_cy_new.py`.
- [ ] Confirm `line_cy.py` remains unchanged by this refactor.
