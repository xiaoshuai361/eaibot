# Timed Left/Right Turn and Exit-Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace left/right lane-capture steering with a timed entry-turn-exit-straight sequence and make strong exit bars survive contaminated same-frame lane models without allowing plain boundaries through.

**Architecture:** Straight intersections retain the existing dual-line bridge. Left/right intersections use `ENTRY -> TURN -> EXIT_STRAIGHT`, with fixed commands and time-only transitions, then reuse `EXIT_ALIGN`. Crosswalk detection gives precedence only to bars with strict multi-stripe geometry; weak bars keep the existing lane-parallel rejection.

**Tech Stack:** Python, ROS `rospy`, OpenCV, `unittest`.

---

### Task 1: Timed Left/Right Phase Decisions

**Files:**
- Modify: `robocom_ws/src/test_line_cy_new.py`
- Modify: `robocom_ws/src/line_cy_new.py`

- [ ] **Step 1: Replace stale capture-decision tests with failing timed-phase tests**

Add tests asserting that the only left/right tuning values are `TURN_ENTRY_TIME`,
`TURN_SPEED`, `TURN_ANGULAR`, and `TURN_TIME`, and that:

```python
self.assertIsNone(line_new.turn_phase_next("ENTRY", 0.9, 1.0, 1.6))
self.assertEqual(line_new.turn_phase_next("ENTRY", 1.0, 1.0, 1.6), "TURN")
self.assertIsNone(line_new.turn_phase_next("TURN", 1.5, 1.0, 1.6))
self.assertEqual(
    line_new.turn_phase_next("TURN", 1.6, 1.0, 1.6),
    "EXIT_STRAIGHT",
)
```

Also assert that `TURN_CAPTURE_DELAY`, `TURN_CAPTURE_FRAMES`, and
`TURN_TIMEOUT` are absent.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m unittest \
  test_line_cy_new.StateTests.test_turn_parameters_are_limited_to_timed_motion_values \
  test_line_cy_new.StateTests.test_turn_phases_advance_only_by_elapsed_time
```

Expected: failures because `TURN_TIME` and the new `turn_phase_next` contract do
not exist yet.

- [ ] **Step 3: Implement minimal timed phase decisions**

In `line_cy_new.py`:

```python
TURN_TIME = 1.6

def turn_phase_next(phase, elapsed, entry_time, turn_time):
    if phase == "ENTRY" and elapsed >= entry_time:
        return "TURN"
    if phase == "TURN" and elapsed >= turn_time:
        return "EXIT_STRAIGHT"
    return None
```

Remove the three capture constants and the four lane-capture/counter helpers.

- [ ] **Step 4: Re-run the focused tests and verify GREEN**

Run the command from Step 2. Expected: both tests pass.

### Task 2: Timed Left/Right Runtime Commands

**Files:**
- Modify: `robocom_ws/src/test_line_cy_new.py`
- Modify: `robocom_ws/src/line_cy_new.py`

- [ ] **Step 1: Add failing runtime tests**

Add focused tests around a `LaneFollower` created with `__new__` asserting:

```python
# TURN ignores a valid lane and publishes the fixed signed turn command.
self.assertEqual(published[-1], (follower.turn_speed, -follower.turn_angular))

# EXIT_STRAIGHT ignores a valid lane and publishes fixed straight motion.
self.assertEqual(published[-1], (follower.turn_speed, 0.0))
```

Verify `_set_state("MANEUVER")` still selects `STRAIGHT` for `turn_cmd=straight`
and `ENTRY` for left/right.

- [ ] **Step 2: Run runtime tests and verify RED**

Run the newly added runtime tests by full unittest name. Expected: the
`EXIT_STRAIGHT` command test fails because the phase does not exist and the old
branch still invokes lane capture/PD control.

- [ ] **Step 3: Replace the left/right runtime branch**

Load and log `~turn_time`, remove capture ROS parameters and state, and implement:

```python
if self.maneuver_phase == "ENTRY":
    self.publish(self.turn_speed, 0.0)
elif self.maneuver_phase == "TURN":
    linear, angular = fixed_turn_command(
        self.turn_cmd, self.turn_speed, self.turn_angular
    )
    self.publish(linear, angular)
elif self.maneuver_phase == "EXIT_STRAIGHT":
    self.publish(self.turn_speed, 0.0)
```

Apply `turn_phase_next` after publishing and transition with
`_set_maneuver_phase`. Make exit-bar eligibility for left/right require
`EXIT_STRAIGHT`. Remove bridge updates, side observations, capture counters,
and timeout warnings from this branch. Keep straight-maneuver bridge control
unchanged.

- [ ] **Step 4: Run runtime and existing safety tests**

Expected: timed command tests, fixed-turn angular-limit test, straight phase
test, maneuver timeout test, and exit guard tests pass.

### Task 3: Strong Exit-Bar Evidence

**Files:**
- Modify: `robocom_ws/src/test_line_cy_new.py`
- Modify: `robocom_ws/src/line_cy_new.py`

- [ ] **Step 1: Add failing geometry tests**

Create synthetic bar dictionaries and lane models to verify:

```python
# A strong top-end bar crossing the vehicle axis is not vetoed as a lane.
self.assertFalse(detector._bar_matches_lane(strong_bar, [same_model], 640))

# The same line without matched stripes remains a lane and is rejected.
self.assertTrue(detector._bar_matches_lane(plain_bar, [same_model], 640))
```

Add an `_hough_bars` synthetic image test with two stripe polygons whose top
ends touch the bar, and assert at least two matches.

- [ ] **Step 2: Run geometry tests and verify RED**

Expected: the strong bar is vetoed by the contaminated lane model and the
top-end stripes are not matched.

- [ ] **Step 3: Implement strict strong-bar precedence**

Set `STRIPE_LONG_MAX_RATIO = 0.45`, add:

```python
BAR_STRIPE_TOP_ABOVE_RATIO = 0.14
BAR_STRIPE_TOP_BELOW_RATIO = 0.10
```

During stripe matching, compute `stripe_top` and accept `near_bottom or
near_top`. Add a helper that requires matched count, matched-center x span, and
vehicle-axis coverage. `_bar_matches_lane` returns `False` early only for this
strong geometry. Use the same helper for fallback and final `strong`
confirmation. Keep `BAR_ONLY_MAX_ABS_ANGLE = 20.0` and the normal lane veto.

- [ ] **Step 4: Run geometry tests and verify GREEN**

Expected: strong top-end bar passes; plain boundary remains rejected.

### Task 4: Cleanup and Verification

**Files:**
- Modify: `robocom_ws/src/test_line_cy_new.py`
- Modify: `robocom_ws/src/line_cy_new.py`

- [ ] **Step 1: Remove obsolete capture tests and references**

Delete tests for lane capture delay, capture hits, side capture, and
`EXIT_APPROACH`. Search both files and require no results for:

```bash
rg "TURN_CAPTURE|TURN_TIMEOUT|turn_capture|turn_lane_capture|turn_side_capture|turn_side_control_target|EXIT_APPROACH" \
  robocom_ws/src/line_cy_new.py robocom_ws/src/test_line_cy_new.py
```

- [ ] **Step 2: Compile**

```bash
python3 -m py_compile \
  robocom_ws/src/line_cy_new.py \
  robocom_ws/src/test_line_cy_new.py
```

Expected: exit code 0.

- [ ] **Step 3: Run focused regression tests**

Run all `StateTests` methods related to timed phases, fixed commands, straight
maneuvers, exit bars, maneuver timeout, and re-entry guard, plus the new
crosswalk geometry tests. Expected: all selected tests pass.

- [ ] **Step 4: Inspect final scope**

Confirm the straight bridge path, normal FOLLOW PD, user-tuned speed/PID values,
and unrelated dirty-worktree files are unchanged.

