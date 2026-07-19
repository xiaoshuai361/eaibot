# YOLO Task Ledger Design

## Goal

`robocom_ws/src/line_cy_task.py` keeps the existing nine-intersection route and
line-following behavior, but changes YOLO from a simple stop trigger into a
task-level recognition ledger.

The robot must:

- Recognize two people classes, two trash-bin classes, and four building classes.
- Count people when a people class is recognized.
- Record each street target class once per full run.
- Record each street area once per full run.
- Record each building area once per full run.
- Record each building class once per full run.
- Save one boxed YOLO image for each accepted task result.
- Print concise area-aware results to the terminal.
- Pause YOLO inference on route segments where camera 0 should be reserved for
  future arm/grasp work.

This change does not add grasping or delivery behavior.

## Route Recognition Map

The existing route remains:

```text
right, straight, right, left, straight, left, right, straight, right
```

Recognition is enabled only on these route segments:

```text
task_index=0, start -> intersection 1: YOLO off
task_index=1, intersection 1 -> 2: street areas C, then P
task_index=2, intersection 2 -> 3: street areas A, then S
task_index=3, intersection 3 -> 4: YOLO off, middle material area
task_index=4, intersection 4 -> 5: building B
task_index=5, intersection 5 -> 6: building C
task_index=6, intersection 6 -> 7: YOLO off, middle material area
task_index=7, intersection 7 -> 8: building A
task_index=8, intersection 8 -> 9: building D
FINAL_EXIT: YOLO off
```

Street segments contain two possible areas. When a new valid street target is
accepted, it is assigned to the first unrecorded area in that segment's ordered
area list. Global class de-duplication prevents one object from being accepted
again as the next area.

## Target Classes

Street target classes:

```text
Medical population
General population
可回收垃圾
有害垃圾
```

Building target classes:

```text
Fire Building
Collapsed Building
Toxic Gas-contaminated Building
Electrical Fault Building
```

Street target classes are globally unique in the match: each appears in at most
one street area. Building target classes are also globally unique: each building
class appears in at most one building area.

## Task Ledger

Add a small task ledger owned by `LaneFollower`. It should be separate from the
YOLO detector and from the line-following state machine.

Ledger state:

```text
street_results: C/P/A/S -> accepted result or None
street_seen_classes: accepted street class names
building_results: A/B/C/D -> accepted result or None
building_seen_classes: accepted building class names
save_index: monotonic image number for this run
pending_yolo_event: accepted event waiting for YOLO_STOP reporting
```

Accepted event fields:

```text
kind: "street" or "building"
area: C/P/A/S or A/B/C/D
class_name: model class name
display_name: Chinese terminal/file display text
people_counts: only for people events
detection: selected YOLO detection
```

Startup clears `/home/eaibot/zcy/保存图片` and recreates it if needed.

## Acceptance Rules

YOLO is sampled from the existing background thread.

For each new YOLO result in `FOLLOW`:

1. Ignore the result when the current route segment has YOLO disabled.
2. Ignore detections outside the configured center band or below confidence.
3. Prefer building detections on building segments.
4. Prefer street detections on street segments.
5. Reject a street detection if its class was already accepted.
6. Reject a street detection if the current segment has no remaining street
   areas.
7. Reject a building detection if the current building area already has a
   result.
8. Reject a building detection if its building class was already accepted.
9. Otherwise create a pending event and enter `YOLO_STOP`.

No terminal output is produced for rejected detections.

## Stop, Report, And Save Flow

The existing stop behavior is retained:

```text
FOLLOW receives accepted event
-> enter YOLO_STOP and publish zero velocity
-> wait for a newer YOLO inference result
-> report and save using that newer frame/result
-> remain stopped until YOLO_STOP_TIME has elapsed
-> return to FOLLOW
```

The report frame is saved with YOLO boxes and labels drawn on it.

File names do not include a date. They include a sequence number, area, and
result:

```text
01_C区_医疗人群2个.jpg
02_P区_可回收垃圾.jpg
03_楼宇B_坍塌楼宇.jpg
```

If a file name would collide, the sequence number prevents overwrite.

Terminal output examples:

```text
C区检测到人群：医疗人群2个
P区检测到垃圾桶：可回收垃圾
楼宇B检测到坍塌楼宇
```

People counts are computed from the report YOLO result for the accepted people
class only. Trash-bin and building events print the accepted class name.

## YOLO Pause And Resume

The YOLO model is still loaded and warmed up at startup before driving begins.

On disabled route segments, the YOLO worker should skip inference but keep the
camera and model objects open. This avoids camera re-open risk and leaves the
current code path close to the tested behavior.

When entering a route segment that requires YOLO, driving should wait until the
YOLO worker has produced at least one fresh inference result for that segment.
If no fresh result is available, the robot publishes zero velocity and remains
waiting. This protects the first recognizable area after a disabled middle
segment.

## Debug View

The existing YOLO debug window can stay. It should draw boxes on the latest YOLO
frame and may optionally show the current route segment and whether YOLO is
enabled or waiting.

## Testing

Focused tests should cover:

- Route segment to recognition context mapping.
- Street class de-duplication across C/P/A/S.
- Street area de-duplication.
- Building class de-duplication across A/B/C/D.
- Building area de-duplication.
- Disabled route segments never enter `YOLO_STOP`.
- Entering an enabled segment waits for a fresh YOLO result.
- Report output and saved file names for people, trash, and building events.
- Startup image directory cleanup.

Full `test_line_cy_new.py` discovery currently has unrelated legacy failures, so
verification should use focused tests plus `py_compile` for the edited files.
