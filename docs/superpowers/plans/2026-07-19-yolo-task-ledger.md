# YOLO Task Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a task-level YOLO recognition ledger for the nine-intersection route, with area-aware output, boxed image saving, one-time class/area recognition, and route-based YOLO pause/resume.

**Architecture:** Keep the existing YOLO detector thread and line-following state machine in `line_cy_task.py`. Add small pure helpers for route context and event selection, then integrate those helpers into `LaneFollower` so the state machine only receives accepted task events.

**Tech Stack:** Python 2/3-compatible ROS script, `rospy`, OpenCV `cv2`, NumPy, `unittest`, existing fake follower tests in `robocom_ws/src/test_line_cy_new.py`.

## Global Constraints

- The route remains exactly `right, straight, right, left, straight, left, right, straight, right`.
- YOLO is enabled only on route segments `task_index=1,2,4,5,7,8`.
- Route segment `task_index=1` maps accepted street targets to `C区`, then `P区`.
- Route segment `task_index=2` maps accepted street targets to `A区`, then `S区`.
- Route segments `task_index=4,5,7,8` map to `楼宇B`, `楼宇C`, `楼宇A`, `楼宇D`.
- Route segments `task_index=0,3,6`, `FINAL_EXIT`, and `DONE` keep YOLO inference off.
- Street target classes are `Medical population`, `General population`, `可回收垃圾`, `有害垃圾`.
- Keep the old 14-class model runnable: default class-name decoding stays compatible with the existing model, and the new 8-class task model can be selected with a ROS parameter override.
- Building target classes are `Fire Building`, `Collapsed Building`, `Toxic Gas-contaminated Building`, `Electrical Fault Building`.
- Street target classes are globally unique per run.
- Building target classes are globally unique per run.
- Save boxed YOLO images to `/home/eaibot/zcy/保存图片`.
- Clear `/home/eaibot/zcy/保存图片` at startup.
- Saved filenames do not include dates.
- Keep camera 0 and the loaded model open while YOLO inference is paused.
- Do not add grasping or delivery behavior.

---

## File Structure

- Modify `robocom_ws/src/line_cy_task.py`
  - Add task route constants, target display maps, a lightweight `YoloTaskEvent`, and a `YoloTaskLedger`.
  - Replace old building/people stop-count logic with ledger event selection.
  - Add boxed-frame rendering and image save helpers.
  - Gate the YOLO worker by route context and wait for fresh inference when entering enabled route segments.
- Modify `robocom_ws/src/test_line_cy_new.py`
  - Keep existing ONNX decode and debug-window tests.
  - Replace old stop-limit/cooldown tests with route ledger tests.
  - Add focused tests for save directory cleanup, boxed save/report, disabled segments, and fresh-result waiting.

---

### Task 1: Route Context And Ledger Selection

**Files:**
- Modify: `robocom_ws/src/line_cy_task.py`
- Modify: `robocom_ws/src/test_line_cy_new.py`

**Interfaces:**
- Consumes: existing `YoloDetection`, `TASK_TURN_COMMANDS`, `YOLO_CONFIDENCE`.
- Produces:
  - `yolo_route_context(task_index, state) -> dict`
  - `class YoloTaskEvent(object)`
  - `class YoloTaskLedger(object)`
  - `YoloTaskLedger.select_event(context, detections, confidence) -> YoloTaskEvent or None`
  - `YoloTaskLedger.accept(event) -> None`

- [ ] **Step 1: Write failing route-context tests**

Add these tests inside `TaskYoloTests` in `/home/zcy/eaibot/robocom_ws/src/test_line_cy_new.py`:

```python
    def test_yolo_route_context_maps_task_segments(self):
        expected = {
            0: {"kind": "off"},
            1: {"kind": "street", "areas": ("C区", "P区")},
            2: {"kind": "street", "areas": ("A区", "S区")},
            3: {"kind": "off"},
            4: {"kind": "building", "area": "楼宇B"},
            5: {"kind": "building", "area": "楼宇C"},
            6: {"kind": "off"},
            7: {"kind": "building", "area": "楼宇A"},
            8: {"kind": "building", "area": "楼宇D"},
        }
        for task_index, context in expected.items():
            self.assertEqual(
                line_task.yolo_route_context(task_index, "FOLLOW"),
                context,
            )
        self.assertEqual(
            line_task.yolo_route_context(8, "FINAL_EXIT"),
            {"kind": "off"},
        )

    def test_yolo_target_classes_include_people_trash_and_buildings(self):
        self.assertEqual(
            line_task.YOLO_TARGET_CLASS_NAMES,
            (
                "Collapsed Building",
                "Electrical Fault Building",
                "Fire Building",
                "Toxic Gas-contaminated Building",
                "General population",
                "Medical population",
                "可回收垃圾",
                "有害垃圾",
            ),
        )
```

- [ ] **Step 2: Run route-context tests and verify failure**

Run:

```bash
python3 -m unittest \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_yolo_route_context_maps_task_segments \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_yolo_target_classes_include_people_trash_and_buildings
```

Expected: failure because `yolo_route_context` does not exist and target names do not include trash classes.

- [ ] **Step 3: Implement route constants and target class maps**

In `/home/zcy/eaibot/robocom_ws/src/line_cy_task.py`, update the YOLO constants block:

```python
YOLO_SAVE_DIR = "/home/eaibot/zcy/保存图片" # 任务识别图片保存目录，启动时清空。
YOLO_LEGACY_CLASS_NAMES = (
    "Collapsed Building",
    "Electrical Fault Building",
    "Emergency power supply device",
    "Fire Building",
    "Fire extinguishing device",
    "Gas purification device",
    "General population",
    "ID1",
    "ID2",
    "ID3",
    "ID4",
    "Medical population",
    "Structural support device",
    "Toxic Gas-contaminated Building",
)
YOLO_TASK_CLASS_NAMES = (
    "Collapsed Building",
    "Electrical Fault Building",
    "Fire Building",
    "Toxic Gas-contaminated Building",
    "General population",
    "Medical population",
    "可回收垃圾",
    "有害垃圾",
)
YOLO_CLASS_NAMES = YOLO_LEGACY_CLASS_NAMES
YOLO_TARGET_CLASS_NAMES = (
    "Collapsed Building",
    "Electrical Fault Building",
    "Fire Building",
    "Toxic Gas-contaminated Building",
    "General population",
    "Medical population",
    "可回收垃圾",
    "有害垃圾",
)
YOLO_STREET_MESSAGES = {
    "Medical population": ("people", "医疗人群"),
    "General population": ("people", "普通人群"),
    "可回收垃圾": ("trash", "可回收垃圾"),
    "有害垃圾": ("trash", "有害垃圾"),
}
YOLO_BUILDING_MESSAGES = (
    ("Fire Building", "火灾楼宇"),
    ("Collapsed Building", "坍塌楼宇"),
    ("Toxic Gas-contaminated Building", "有毒气体楼宇"),
    ("Electrical Fault Building", "电力故障楼宇"),
)
YOLO_BUILDING_MESSAGE_BY_CLASS = dict(YOLO_BUILDING_MESSAGES)
YOLO_STREET_ROUTE_AREAS = {
    1: ("C区", "P区"),
    2: ("A区", "S区"),
}
YOLO_BUILDING_ROUTE_AREAS = {
    4: "楼宇B",
    5: "楼宇C",
    7: "楼宇A",
    8: "楼宇D",
}
```

Add this helper near other pure helpers:

```python
def yolo_route_context(task_index, state):
    if state in ("FINAL_EXIT", "DONE"):
        return {"kind": "off"}
    index = int(task_index)
    if index in YOLO_STREET_ROUTE_AREAS:
        return {"kind": "street", "areas": YOLO_STREET_ROUTE_AREAS[index]}
    if index in YOLO_BUILDING_ROUTE_AREAS:
        return {"kind": "building", "area": YOLO_BUILDING_ROUTE_AREAS[index]}
    return {"kind": "off"}
```

- [ ] **Step 4: Run route-context tests and verify pass**

Run the command from Step 2.

Expected: both tests pass.

- [ ] **Step 5: Write failing ledger selection tests**

Add these tests inside `TaskYoloTests`:

```python
    def test_task_ledger_accepts_street_target_once_per_class_and_area(self):
        ledger = line_task.YoloTaskLedger()
        context = line_task.yolo_route_context(1, "FOLLOW")
        medical = line_task.YoloDetection(
            5, "Medical population", 0.9,
            (80, 30, 120, 80), (100, 200, 3), 0.8
        )
        repeat = line_task.YoloDetection(
            5, "Medical population", 0.88,
            (82, 31, 122, 81), (100, 200, 3), 0.8
        )
        trash = line_task.YoloDetection(
            6, "可回收垃圾", 0.91,
            (70, 20, 130, 90), (100, 200, 3), 0.8
        )

        first = ledger.select_event(context, [medical], 0.5)
        ledger.accept(first)
        self.assertEqual(first.kind, "street")
        self.assertEqual(first.area, "C区")
        self.assertEqual(first.class_name, "Medical population")

        self.assertIsNone(ledger.select_event(context, [repeat], 0.5))

        second = ledger.select_event(context, [trash], 0.5)
        ledger.accept(second)
        self.assertEqual(second.area, "P区")
        self.assertEqual(second.display_name, "可回收垃圾")

        general = line_task.YoloDetection(
            4, "General population", 0.95,
            (70, 20, 130, 90), (100, 200, 3), 0.8
        )
        self.assertIsNone(ledger.select_event(context, [general], 0.5))

    def test_task_ledger_accepts_building_area_and_class_once(self):
        ledger = line_task.YoloTaskLedger()
        context = line_task.yolo_route_context(4, "FOLLOW")
        fire = line_task.YoloDetection(
            2, "Fire Building", 0.9,
            (80, 30, 120, 80), (100, 200, 3), 0.8
        )
        event = ledger.select_event(context, [fire], 0.5)
        ledger.accept(event)

        self.assertEqual(event.kind, "building")
        self.assertEqual(event.area, "楼宇B")
        self.assertEqual(event.display_name, "火灾楼宇")
        self.assertIsNone(ledger.select_event(context, [fire], 0.5))

        same_class_new_area = ledger.select_event(
            line_task.yolo_route_context(5, "FOLLOW"), [fire], 0.5
        )
        self.assertIsNone(same_class_new_area)

    def test_task_ledger_ignores_off_route_and_non_center_targets(self):
        ledger = line_task.YoloTaskLedger()
        fire = line_task.YoloDetection(
            2, "Fire Building", 0.9,
            (0, 30, 20, 80), (100, 200, 3), 0.5
        )
        self.assertIsNone(
            ledger.select_event({"kind": "off"}, [fire], 0.5)
        )
        self.assertIsNone(
            ledger.select_event(
                line_task.yolo_route_context(4, "FOLLOW"), [fire], 0.5
            )
        )
```

- [ ] **Step 6: Run ledger tests and verify failure**

Run:

```bash
python3 -m unittest \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_task_ledger_accepts_street_target_once_per_class_and_area \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_task_ledger_accepts_building_area_and_class_once \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_task_ledger_ignores_off_route_and_non_center_targets
```

Expected: failure because `YoloTaskLedger` does not exist.

- [ ] **Step 7: Implement `YoloTaskEvent` and `YoloTaskLedger`**

Add near `YoloDetection` in `line_cy_task.py`:

```python
class YoloTaskEvent(object):
    def __init__(self, kind, area, class_name, display_name, detection):
        self.kind = str(kind)
        self.area = str(area)
        self.class_name = str(class_name)
        self.display_name = str(display_name)
        self.detection = detection


class YoloTaskLedger(object):
    def __init__(self):
        self.street_results = dict((area, None) for area in ("C区", "P区", "A区", "S区"))
        self.street_seen_classes = set()
        self.building_results = dict(
            (area, None) for area in ("楼宇A", "楼宇B", "楼宇C", "楼宇D")
        )
        self.building_seen_classes = set()
        self.pending_event = None
        self.save_index = 0

    def _target_candidates(self, detections, confidence):
        return [
            item for item in detections
            if item.target and item.in_center
            and item.confidence >= float(confidence)
        ]

    def _next_street_area(self, areas):
        for area in areas:
            if self.street_results.get(area) is None:
                return area
        return None

    def select_event(self, context, detections, confidence):
        kind = context.get("kind")
        candidates = self._target_candidates(detections, confidence)
        if kind == "street":
            area = self._next_street_area(context.get("areas", ()))
            if area is None:
                return None
            street = [
                item for item in candidates
                if item.class_name in YOLO_STREET_MESSAGES
                and item.class_name not in self.street_seen_classes
            ]
            if not street:
                return None
            selected = max(street, key=lambda item: item.confidence)
            _, display_name = YOLO_STREET_MESSAGES[selected.class_name]
            return YoloTaskEvent(
                "street", area, selected.class_name, display_name, selected
            )
        if kind == "building":
            area = context.get("area")
            if self.building_results.get(area) is not None:
                return None
            buildings = [
                item for item in candidates
                if item.class_name in YOLO_BUILDING_MESSAGE_BY_CLASS
                and item.class_name not in self.building_seen_classes
            ]
            if not buildings:
                return None
            selected = max(buildings, key=lambda item: item.confidence)
            return YoloTaskEvent(
                "building", area, selected.class_name,
                YOLO_BUILDING_MESSAGE_BY_CLASS[selected.class_name],
                selected,
            )
        return None

    def accept(self, event):
        self.pending_event = event
        if event is None:
            return
        if event.kind == "street":
            self.street_results[event.area] = event
            self.street_seen_classes.add(event.class_name)
        elif event.kind == "building":
            self.building_results[event.area] = event
            self.building_seen_classes.add(event.class_name)
```

- [ ] **Step 8: Run Task 1 tests and commit**

Run:

```bash
python3 -m unittest \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_yolo_route_context_maps_task_segments \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_yolo_target_classes_include_people_trash_and_buildings \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_task_ledger_accepts_street_target_once_per_class_and_area \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_task_ledger_accepts_building_area_and_class_once \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_task_ledger_ignores_off_route_and_non_center_targets
```

Expected: all selected tests pass.

Commit:

```bash
git -C /home/zcy/eaibot add -f robocom_ws/src/line_cy_task.py robocom_ws/src/test_line_cy_new.py
git -C /home/zcy/eaibot commit -m "feat: add yolo task ledger"
```

---

### Task 2: Boxed Image Save And Area-Aware Reporting

**Files:**
- Modify: `robocom_ws/src/line_cy_task.py`
- Modify: `robocom_ws/src/test_line_cy_new.py`

**Interfaces:**
- Consumes:
  - `YoloTaskLedger.pending_event`
  - `YoloTaskEvent(kind, area, class_name, display_name, detection)`
- Produces:
  - `draw_yolo_boxes(frame, detections, center_band_ratio) -> frame`
  - `ensure_clean_directory(path) -> None`
  - `LaneFollower._prepare_yolo_save_dir() -> None`
  - `LaneFollower._report_yolo_task_event(detections) -> None`

- [ ] **Step 1: Write failing cleanup/save/report tests**

Add inside `TaskYoloTests`:

```python
    def test_prepare_yolo_save_dir_clears_previous_images(self):
        root = tempfile.mkdtemp()
        try:
            old_path = os.path.join(root, "old.jpg")
            nested = os.path.join(root, "nested")
            os.mkdir(nested)
            open(old_path, "w").close()
            open(os.path.join(nested, "x.txt"), "w").close()
            follower = self._follower(now=30.0)
            self._restore_rospy = follower._restore_rospy
            follower.yolo_save_dir = root

            follower._prepare_yolo_save_dir()

            self.assertTrue(os.path.isdir(root))
            self.assertEqual(os.listdir(root), [])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_report_yolo_task_event_logs_and_saves_people_boxed_image(self):
        root = tempfile.mkdtemp()
        try:
            follower = self._follower(now=30.0)
            self._restore_rospy = follower._restore_rospy
            follower.yolo_save_dir = root
            follower.task_ledger = line_task.YoloTaskLedger()
            detection = line_task.YoloDetection(
                5, "Medical population", 0.9,
                (20, 20, 80, 80), (100, 120, 3), 0.8
            )
            event = line_task.YoloTaskEvent(
                "street", "C区", "Medical population", "医疗人群", detection
            )
            follower.task_ledger.accept(event)
            with follower.yolo_lock:
                follower.yolo_latest_frame = np.zeros((100, 120, 3), dtype=np.uint8)
            logs = []
            original_loginfo = line_task.rospy.loginfo
            line_task.rospy.loginfo = lambda message, *args: logs.append(
                message % args if args else message
            )
            try:
                follower._report_yolo_task_event([detection])
            finally:
                line_task.rospy.loginfo = original_loginfo

            self.assertIn("C区检测到人群：医疗人群1个", logs)
            files = os.listdir(root)
            self.assertEqual(files, ["01_C区_医疗人群1个.jpg"])
            saved = line_task.cv2.imread(os.path.join(root, files[0]))
            self.assertGreater(int(np.count_nonzero(saved)), 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_report_yolo_task_event_logs_trash_and_building(self):
        root = tempfile.mkdtemp()
        try:
            follower = self._follower(now=30.0)
            self._restore_rospy = follower._restore_rospy
            follower.yolo_save_dir = root
            follower.task_ledger = line_task.YoloTaskLedger()
            trash = line_task.YoloDetection(
                6, "有害垃圾", 0.9,
                (20, 20, 80, 80), (100, 120, 3), 0.8
            )
            event = line_task.YoloTaskEvent(
                "street", "P区", "有害垃圾", "有害垃圾", trash
            )
            follower.task_ledger.accept(event)
            with follower.yolo_lock:
                follower.yolo_latest_frame = np.zeros((100, 120, 3), dtype=np.uint8)
            logs = []
            original_loginfo = line_task.rospy.loginfo
            line_task.rospy.loginfo = lambda message, *args: logs.append(
                message % args if args else message
            )
            try:
                follower._report_yolo_task_event([trash])
            finally:
                line_task.rospy.loginfo = original_loginfo
            self.assertIn("P区检测到垃圾桶：有害垃圾", logs)
            self.assertEqual(os.listdir(root), ["01_P区_有害垃圾.jpg"])

            follower.task_ledger.pending_event = line_task.YoloTaskEvent(
                "building", "楼宇B", "Collapsed Building", "坍塌楼宇", trash
            )
            with follower.yolo_lock:
                follower.yolo_latest_frame = np.zeros((100, 120, 3), dtype=np.uint8)
            follower._report_yolo_task_event([trash])
            self.assertIn("楼宇B检测到坍塌楼宇", logs)
            self.assertIn("02_楼宇B_坍塌楼宇.jpg", os.listdir(root))
        finally:
            shutil.rmtree(root, ignore_errors=True)
```

Also add `import shutil` near the top of `test_line_cy_new.py`.

- [ ] **Step 2: Run cleanup/save/report tests and verify failure**

Run:

```bash
python3 -m unittest \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_prepare_yolo_save_dir_clears_previous_images \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_report_yolo_task_event_logs_and_saves_people_boxed_image \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_report_yolo_task_event_logs_trash_and_building
```

Expected: failure because save/report helpers do not exist.

- [ ] **Step 3: Implement save directory and boxed drawing helpers**

Add `import shutil` to `line_cy_task.py`.

Add helpers near `find_contours`:

```python
def ensure_clean_directory(path):
    path = os.path.expanduser(str(path))
    if os.path.isdir(path):
        for name in os.listdir(path):
            item = os.path.join(path, name)
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
    else:
        os.makedirs(path)


def safe_filename_text(text):
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "")


def draw_yolo_boxes(frame, detections, center_band_ratio):
    output = frame.copy()
    height, width = output.shape[:2]
    ratio = clamp(float(center_band_ratio), 0.0, 1.0)
    left = int(round(width * (1.0 - ratio) * 0.5))
    right = int(round(width - left))
    cv2.line(output, (left, 0), (left, height - 1), (255, 255, 0), 1)
    cv2.line(output, (right, 0), (right, height - 1), (255, 255, 0), 1)
    for item in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in item.box]
        color = (0, 255, 0) if item.target and item.in_center else (0, 255, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = "{} {:.2f}".format(item.class_name, item.confidence)
        cv2.putText(output, label, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return output
```

- [ ] **Step 4: Implement `LaneFollower` save/report methods**

In `LaneFollower.__init__`, set:

```python
        self.yolo_save_dir = str(rospy.get_param("~yolo_save_dir", YOLO_SAVE_DIR))
        self.task_ledger = YoloTaskLedger()
        self._prepare_yolo_save_dir()
```

Add methods near YOLO methods:

```python
    def _prepare_yolo_save_dir(self):
        ensure_clean_directory(self.yolo_save_dir)

    def _people_count_for_event(self, event, detections):
        return sum(
            1 for item in detections
            if item.class_name == event.class_name
            and item.confidence >= self.yolo_confidence
        )

    def _save_yolo_event_image(self, event, detections):
        with self.yolo_lock:
            frame = None if self.yolo_latest_frame is None \
                else self.yolo_latest_frame.copy()
        if frame is None:
            frame = np.zeros((1, 1, 3), dtype=np.uint8)
        boxed = draw_yolo_boxes(frame, detections, self.yolo_center_band_ratio)
        self.task_ledger.save_index += 1
        if event.kind == "street" and event.class_name in YOLO_STREET_MESSAGES:
            target_kind, _ = YOLO_STREET_MESSAGES[event.class_name]
            if target_kind == "people":
                count = self._people_count_for_event(event, detections)
                result = "%s%d个" % (event.display_name, count)
            else:
                result = event.display_name
        else:
            result = event.display_name
        filename = "%02d_%s_%s.jpg" % (
            self.task_ledger.save_index,
            safe_filename_text(event.area),
            safe_filename_text(result),
        )
        path = os.path.join(self.yolo_save_dir, filename)
        cv2.imwrite(path, boxed)
        return path

    def _report_yolo_task_event(self, detections):
        event = self.task_ledger.pending_event
        if event is None:
            return
        if event.kind == "street":
            target_kind, _ = YOLO_STREET_MESSAGES[event.class_name]
            if target_kind == "people":
                count = self._people_count_for_event(event, detections)
                rospy.loginfo(
                    "%s检测到人群：%s%d个",
                    event.area, event.display_name, count,
                )
            else:
                rospy.loginfo(
                    "%s检测到垃圾桶：%s",
                    event.area, event.display_name,
                )
        elif event.kind == "building":
            rospy.loginfo("%s检测到%s", event.area, event.display_name)
        self._save_yolo_event_image(event, detections)
        self.task_ledger.pending_event = None
```

- [ ] **Step 5: Replace duplicate debug drawing with helper**

In `draw_yolo_debug`, replace manual line/box loop with:

```python
        frame = draw_yolo_boxes(
            frame, detections,
            getattr(self, "yolo_center_band_ratio", YOLO_CENTER_BAND_RATIO),
        )
        status = "YOLO frame_interval={} detections={}".format(
            getattr(self, "yolo_frame_interval", YOLO_FRAME_INTERVAL),
            len(detections)
        )
```

- [ ] **Step 6: Run Task 2 tests and commit**

Run:

```bash
python3 -m unittest \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_prepare_yolo_save_dir_clears_previous_images \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_report_yolo_task_event_logs_and_saves_people_boxed_image \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_report_yolo_task_event_logs_trash_and_building \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_yolo_debug_window_draws_detection_box
```

Expected: all selected tests pass.

Commit:

```bash
git -C /home/zcy/eaibot add -f robocom_ws/src/line_cy_task.py robocom_ws/src/test_line_cy_new.py
git -C /home/zcy/eaibot commit -m "feat: save yolo task evidence"
```

---

### Task 3: Integrate Ledger With YOLO Stop And Route Pause

**Files:**
- Modify: `robocom_ws/src/line_cy_task.py`
- Modify: `robocom_ws/src/test_line_cy_new.py`

**Interfaces:**
- Consumes:
  - `yolo_route_context(task_index, state)`
  - `YoloTaskLedger.select_event(...)`
  - `LaneFollower._report_yolo_task_event(detections)`
- Produces:
  - `LaneFollower._current_yolo_context() -> dict`
  - `LaneFollower._yolo_inference_allowed() -> bool`
  - `LaneFollower._mark_yolo_segment_if_needed() -> None`
  - `LaneFollower._yolo_segment_has_fresh_result() -> bool`
  - `LaneFollower._wait_for_yolo_ready_if_needed() -> bool`

- [ ] **Step 1: Write failing integration tests**

Replace old tests `test_follow_dual_lane_yolo_target_enters_yolo_stop`, `test_yolo_people_stop_does_not_require_dual_lane_rows`, `test_building_stops_only_once_before_intersection_reset`, `test_people_stops_twice_with_three_second_cooldown`, and `test_intersection_completion_resets_yolo_stop_limits` with:

```python
    def test_street_task_detection_enters_yolo_stop_with_area_event(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.task_index = 1
        follower.task_ledger = line_task.YoloTaskLedger()
        detection = line_task.YoloDetection(
            5, "Medical population", 0.9,
            (80, 30, 120, 80), (100, 200, 3), 0.8
        )
        follower._poll_yolo_detections = lambda: (True, [detection])

        stopped = follower._maybe_enter_yolo_stop(
            types.SimpleNamespace(valid=True, dual_rows=0)
        )

        self.assertTrue(stopped)
        self.assertEqual(follower.state, "YOLO_STOP")
        self.assertEqual(follower.task_ledger.pending_event.area, "C区")
        self.assertEqual(
            follower.task_ledger.pending_event.class_name,
            "Medical population",
        )

    def test_building_task_detection_enters_yolo_stop_with_building_area(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.task_index = 4
        follower.task_ledger = line_task.YoloTaskLedger()
        detection = line_task.YoloDetection(
            2, "Fire Building", 0.9,
            (80, 30, 120, 80), (100, 200, 3), 0.8
        )
        follower._poll_yolo_detections = lambda: (True, [detection])

        stopped = follower._maybe_enter_yolo_stop(
            types.SimpleNamespace(valid=True, dual_rows=2)
        )

        self.assertTrue(stopped)
        self.assertEqual(follower.task_ledger.pending_event.area, "楼宇B")
        self.assertEqual(follower.task_ledger.pending_event.display_name, "火灾楼宇")

    def test_disabled_yolo_route_does_not_poll_or_stop(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.task_index = 3
        follower.task_ledger = line_task.YoloTaskLedger()
        calls = []
        follower._poll_yolo_detections = lambda: calls.append(True) or (True, [])

        stopped = follower._maybe_enter_yolo_stop(
            types.SimpleNamespace(valid=True, dual_rows=2)
        )

        self.assertFalse(stopped)
        self.assertEqual(calls, [])

    def test_yolo_worker_skips_inference_on_disabled_route(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.task_index = 3
        follower.state = "FOLLOW"

        self.assertFalse(follower._yolo_inference_allowed())

        follower.task_index = 4
        self.assertTrue(follower._yolo_inference_allowed())

    def test_enabled_yolo_route_waits_for_fresh_segment_result(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.task_index = 4
        follower.state = "FOLLOW"
        follower.yolo_ready = True
        follower.yolo_segment_key = None
        follower.yolo_segment_start_seq = 3
        with follower.yolo_lock:
            follower.yolo_latest_seq = 3

        ready = follower._wait_for_yolo_ready_if_needed()

        self.assertFalse(ready)
        self.assertEqual(follower.published[-1], (0, 0))

        with follower.yolo_lock:
            follower.yolo_latest_seq = 4
        self.assertTrue(follower._wait_for_yolo_ready_if_needed())
```

- [ ] **Step 2: Run integration tests and verify failure**

Run:

```bash
python3 -m unittest \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_street_task_detection_enters_yolo_stop_with_area_event \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_building_task_detection_enters_yolo_stop_with_building_area \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_disabled_yolo_route_does_not_poll_or_stop \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_yolo_worker_skips_inference_on_disabled_route \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_enabled_yolo_route_waits_for_fresh_segment_result
```

Expected: failure because integration helpers do not exist and `_maybe_enter_yolo_stop` still uses old stop counters.

- [ ] **Step 3: Initialize new segment state in `LaneFollower`**

In `LaneFollower.__init__`, remove old fields:

```python
        self.yolo_stop_event_type = None
        self.yolo_building_stop_count = 0
        self.yolo_people_stop_count = 0
        self.yolo_last_people_stop_time = None
```

Add:

```python
        self.yolo_segment_key = None
        self.yolo_segment_start_seq = 0
```

Update test `_follower` to set:

```python
        follower.task_ledger = line_task.YoloTaskLedger()
        follower.yolo_segment_key = None
        follower.yolo_segment_start_seq = 0
        follower.yolo_save_dir = tempfile.mkdtemp()
```

and remove old stop-limit fields from the fake follower.

- [ ] **Step 4: Implement route gating helpers**

Add methods near YOLO methods:

```python
    def _current_yolo_context(self):
        return yolo_route_context(
            getattr(self, "task_index", 0),
            getattr(self, "state", "FOLLOW"),
        )

    def _yolo_context_key(self, context):
        if context.get("kind") == "street":
            return ("street", tuple(context.get("areas", ())))
        if context.get("kind") == "building":
            return ("building", context.get("area"))
        return ("off", None)

    def _mark_yolo_segment_if_needed(self):
        context = self._current_yolo_context()
        key = self._yolo_context_key(context)
        if key != self.yolo_segment_key:
            self.yolo_segment_key = key
            self.yolo_segment_start_seq = self._latest_yolo_seq()
        return context

    def _yolo_inference_allowed(self):
        if not self.yolo_enabled:
            return False
        if getattr(self, "state", None) not in ("FOLLOW", "YOLO_STOP"):
            return False
        return self._current_yolo_context().get("kind") != "off" \
            or getattr(self, "state", None) == "YOLO_STOP"

    def _yolo_segment_has_fresh_result(self):
        if self._current_yolo_context().get("kind") == "off":
            return True
        return self._latest_yolo_seq() > self.yolo_segment_start_seq

    def _wait_for_yolo_ready_if_needed(self):
        context = self._mark_yolo_segment_if_needed()
        if context.get("kind") == "off" or not self.yolo_enabled:
            return True
        if self.yolo_ready and self._yolo_segment_has_fresh_result():
            return True
        self.publish(0, 0)
        return False
```

- [ ] **Step 5: Gate the YOLO worker**

In `_yolo_loop`, replace the state-only gate:

```python
            if getattr(self, "state", None) not in ("FOLLOW", "YOLO_STOP"):
                time.sleep(0.05)
                continue
```

with:

```python
            if not self._yolo_inference_allowed():
                time.sleep(0.05)
                continue
```

- [ ] **Step 6: Replace old YOLO stop selection with ledger selection**

Replace `_select_yolo_stop_event`, `_record_yolo_stop_event`, `_reset_yolo_stop_limits`, `_selected_yolo_target`, and `_log_yolo_stop_summary` with ledger-based behavior:

```python
    def _select_yolo_stop_event(self, detections, now=None):
        context = self._current_yolo_context()
        return self.task_ledger.select_event(
            context, detections, self.yolo_confidence
        )

    def _selected_yolo_target(self, detections):
        event = self._select_yolo_stop_event(detections)
        return None if event is None else event.detection
```

Update `_maybe_enter_yolo_stop`:

```python
    def _maybe_enter_yolo_stop(self, observation):
        if self.state != "FOLLOW" or not self.yolo_enabled:
            return False
        if not self._wait_for_yolo_ready_if_needed():
            return False
        sampled, detections = self._poll_yolo_detections()
        if not sampled:
            return False
        event = self._select_yolo_stop_event(detections)
        if event is None or not self.yolo_stop_enabled:
            return False
        self.task_ledger.accept(event)
        self.yolo_stop_detection = event.detection
        self.yolo_stop_reported = False
        self.yolo_stop_report_seq = self._latest_yolo_seq()
        self._set_state("YOLO_STOP")
        self.publish(0, 0)
        return True
```

Update `_handle_yolo_stop`:

```python
    def _handle_yolo_stop(self, now):
        if self.state != "YOLO_STOP":
            return False
        self.publish(0, 0)
        if not self.yolo_stop_reported:
            sampled, detections = self._poll_yolo_detections()
            if sampled and self.yolo_read_seq > self.yolo_stop_report_seq:
                self._report_yolo_task_event(detections)
                self.yolo_stop_reported = True
        if (self.yolo_stop_reported
                and float(now) - self.state_started >= self.yolo_stop_time):
            self._set_state("FOLLOW")
        return True
```

In `_complete_intersection`, remove the call to `_reset_yolo_stop_limits()`.

- [ ] **Step 7: Insert fresh-result waiting into `process`**

In the `FOLLOW` branch of `process`, after crosswalk entry priority has been handled and before YOLO stop polling or line control, add:

```python
            if self.state == "FOLLOW" and not self._wait_for_yolo_ready_if_needed():
                return
```

Keep crosswalk entry priority before this wait, so路口停车摆正 remains responsive.

- [ ] **Step 8: Run Task 3 tests and commit**

Run:

```bash
python3 -m unittest \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_street_task_detection_enters_yolo_stop_with_area_event \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_building_task_detection_enters_yolo_stop_with_building_area \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_disabled_yolo_route_does_not_poll_or_stop \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_yolo_worker_skips_inference_on_disabled_route \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_enabled_yolo_route_waits_for_fresh_segment_result \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_yolo_stop_waits_for_new_detection_before_resuming \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_yolo_stop_prints_new_detection_then_resumes_after_stop_time \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_crosswalk_entry_has_priority_over_yolo_stop
```

Expected: all selected tests pass.

Commit:

```bash
git -C /home/zcy/eaibot add -f robocom_ws/src/line_cy_task.py robocom_ws/src/test_line_cy_new.py
git -C /home/zcy/eaibot commit -m "feat: route yolo task stops through ledger"
```

---

### Task 4: Cleanup, Compatibility Tests, And Verification

**Files:**
- Modify: `robocom_ws/src/line_cy_task.py`
- Modify: `robocom_ws/src/test_line_cy_new.py`
- Modify: `docs/superpowers/plans/2026-07-19-yolo-task-ledger.md`

**Interfaces:**
- Consumes: all Task 1-3 interfaces.
- Produces: final verified implementation with obsolete stop-limit logic removed.

- [ ] **Step 1: Remove obsolete constants and fake-follower fields**

Remove from `line_cy_task.py`:

```python
YOLO_BUILDING_STOP_LIMIT = 1
YOLO_PEOPLE_STOP_LIMIT = 2
YOLO_PEOPLE_STOP_COOLDOWN = 6.0
```

Remove `rospy.get_param` reads for:

```python
~yolo_building_stop_limit
~yolo_people_stop_limit
~yolo_people_stop_cooldown
```

Remove fake follower setup fields with the same names from `test_line_cy_new.py`.

- [ ] **Step 2: Make YOLO class names parameter-overridable**

In `LaneFollower.__init__`, after reading YOLO params, add:

```python
        yolo_class_names = rospy.get_param("~yolo_class_names", None)
        yolo_class_profile = str(rospy.get_param(
            "~yolo_class_profile", "legacy"
        )).strip().lower()
        if yolo_class_names:
            if isinstance(yolo_class_names, str):
                yolo_class_names = [
                    item.strip() for item in yolo_class_names.split(",")
                    if item.strip()
                ]
            self.yolo_class_names = tuple(yolo_class_names)
        elif yolo_class_profile == "task":
            self.yolo_class_names = YOLO_TASK_CLASS_NAMES
        else:
            self.yolo_class_names = YOLO_CLASS_NAMES
```

Pass `class_names=self.yolo_class_names` into `YoloObstacleDetector(...)`.

Update `YoloObstacleDetector.__init__` signature:

```python
    def __init__(self, model_path, confidence=YOLO_CONFIDENCE,
                 center_band_ratio=YOLO_CENTER_BAND_RATIO,
                 image_size=YOLO_IMAGE_SIZE,
                 nms_threshold=YOLO_NMS_THRESHOLD,
                 class_names=YOLO_CLASS_NAMES):
```

and set:

```python
        self.names = self._normalize_names(class_names)
```

- [ ] **Step 3: Add parameter override test**

Add:

```python
    def test_yolo_detector_uses_custom_class_names(self):
        with tempfile.TemporaryDirectory() as root:
            model_path = os.path.join(root, "merge_yolov5n_320_best.onnx")
            open(model_path, "w").close()
            detector = line_task.YoloObstacleDetector(
                model_path,
                class_names=("custom_a", "custom_b"),
            )
            self.assertEqual(detector.names, {0: "custom_a", 1: "custom_b"})
```

Run:

```bash
python3 -m unittest \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests.test_yolo_detector_uses_custom_class_names
```

Expected: pass.

- [ ] **Step 4: Run focused YOLO and alignment tests**

Run:

```bash
python3 -m unittest \
  eaibot.robocom_ws.src.test_line_cy_new.TaskEntryAlignmentLockTests \
  eaibot.robocom_ws.src.test_line_cy_new.TaskYoloTests
```

Expected: all tests in these classes pass. If legacy tests in `TaskYoloTests` still expect removed old behavior, update them to assert the new ledger behavior or remove them when fully replaced by Task 3 tests.

- [ ] **Step 5: Compile edited files**

Run:

```bash
python3 -m py_compile \
  /home/zcy/eaibot/robocom_ws/src/line_cy_task.py \
  /home/zcy/eaibot/robocom_ws/src/test_line_cy_new.py
```

Expected: no output and exit code `0`.

- [ ] **Step 6: Inspect for obsolete code and noisy logs**

Run:

```bash
rg -n "YOLO_BUILDING_STOP_LIMIT|YOLO_PEOPLE_STOP_LIMIT|YOLO_PEOPLE_STOP_COOLDOWN|yolo_building_stop_count|yolo_people_stop_count|yolo_last_people_stop_time|YOLO no stop|repeat_skip|print_only" \
  /home/zcy/eaibot/robocom_ws/src/line_cy_task.py \
  /home/zcy/eaibot/robocom_ws/src/test_line_cy_new.py
```

Expected: no matches.

- [ ] **Step 7: Commit final cleanup**

Commit:

```bash
git -C /home/zcy/eaibot add -f \
  robocom_ws/src/line_cy_task.py \
  robocom_ws/src/test_line_cy_new.py \
  docs/superpowers/plans/2026-07-19-yolo-task-ledger.md
git -C /home/zcy/eaibot commit -m "test: verify yolo task ledger"
```

---

## Self-Review

Spec coverage:

- Route recognition map is covered by Task 1 route-context tests and Task 3 route gating.
- Street and building one-time class/area recognition is covered by Task 1 ledger tests.
- Boxed image saving and startup cleanup are covered by Task 2 tests.
- Area-aware terminal output is covered by Task 2 report tests.
- YOLO pause/resume and fresh inference waiting are covered by Task 3 tests.
- Old stop-limit logic cleanup is covered by Task 4 grep verification.

Placeholder scan:

- The plan intentionally contains no `TBD`, `TODO`, or open-ended implementation steps.
- Each code-changing step includes concrete code blocks or exact replacements.

Type consistency:

- `YoloTaskEvent.kind/area/class_name/display_name/detection` are used consistently by ledger, report, and tests.
- `YoloTaskLedger.select_event(context, detections, confidence)` returns an event or `None`.
- `YoloTaskLedger.accept(event)` records the event and sets `pending_event`.
