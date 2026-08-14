# zcy_last 循迹与联调 AI 交接

## 0. 首要事实：开发机与机器人不是同一路径

代码编辑发生在 Windows 开发机：

```text
F:\桌面\平安城市\国赛\code\eaibot
```

比赛实际运行在机器人：

```text
/home/eaibot
```

本文所有运行命令、模型、日志、配置和示教文件路径均使用机器人路径 `/home/eaibot/...`。

不要让用户在机器人终端执行 Windows 路径或旧 `/home/zcy/eaibot/...` 路径。开发机修改完成后必须把相关文件同步到机器人对应路径，不能只复制报错的单个文件。

## 1. 当前目标与完成情况

比赛主程序负责：

- 九路口循迹，路线为右、直、右、左、直、左、右、直、右；
- 每个入口摆正后等待红绿灯，只有稳定绿灯才通过；
- 第一圈识别人群和垃圾桶，后两圈识别楼宇；
- 比赛开始前执行 B 点有 Tag 抓取，可独立开关；
- 第 3 个路口完成后执行 A 点无 Tag 抓取，可独立开关；
- B 点和 A 点投递分别独立开关；
- 抓取期间管理 Astra、任务 YOLO 和 `/cmd_vel` 所有权；
- 投递失败只报警并继续，抓取失败进入 `PICK_FAILED` 停车。

已实现但仍需真机验收的最新改动：

1. B 点有 Tag 检测窗口 `tag_pick_detection`。
2. B 点抓取完成后不再寻找已越过的入口横条。
3. B 点抓取完成后先判断绿灯，再按独立时间执行第一次直行和右转。
4. A 点在第 3 个路口后加载无 Tag 模型，以固定零角速度直行并从 Astra 画面右侧等待物块出现，不恢复巡线修正。
5. A 点触发后先停车，再把 `/cmd_vel` 交给抓取子进程。
6. 正式比赛使用两终端：`launch.py` 常驻底盘、MoveIt 和手眼 TF，`main.py` 由用户决定是否启动，并按阶段托管 Astra、Tag 栈和 YOLO。
7. `launch.py` 不得自动调用 `main.py`；它必须保持前台运行，退出时自动清理自己启动的依赖。

不要把“单元测试通过”等同于“真机已验收”。

## 2. 权威运行入口

机器人终端一：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/robocom_ws/src
python3 -m zcy_last.launch
```

保持终端一运行。终端二：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/robocom_ws/src
python3 -m zcy_last.main \
  --tag-pick --tag-pick-count 4 --tag-delivery \
  --no-untagged-pick
```

任务命令固定对外提供五种组合：只循迹、B 抓取并投递、B 只抓取、A 抓取并投递、A 只抓取。权威命令见 `使用命令.md`，不再额外列出 A+B 同时抓取组合。

比赛命令速查以机器人文件为准：

```text
/home/eaibot/robocom_ws/src/zcy_last/使用命令.md
```

## 3. 代码分层与职责

机器人新版目录：

```text
/home/eaibot/robocom_ws/src/zcy_last/
├── launch.py
├── main.py
├── config.py
├── algorithms/
├── control/
├── task/
├── tests/
├── 使用命令.md
└── 循迹与联调AI交接.md
```

主要职责：

| 文件 | 责任 | 不应放入的内容 |
| --- | --- | --- |
| `launch.py` | 终端一启动、持有和清理底盘/MoveIt/手眼 TF | 比赛状态机、视觉算法 |
| `main.py` | 终端二任务入口；参数解析和组装状态机 | 底盘常驻启动、视觉算法 |
| `config.py` | 现场开关、设备号、模型路径、可调参数 | 运行流程代码 |
| `algorithms/vision.py` | 巡线、横条、补线等视觉算法 | ROS 进程编排 |
| `algorithms/traffic_light.py` | 红绿灯模型推理和画框 | 比赛路口状态 |
| `algorithms/yolo_task.py` | 人群、垃圾桶、楼宇推理与事件 | 机械臂动作 |
| `control/runtime.py` | 摄像头读取、PID 等运行控制 | 比赛路线 |
| `control/processes.py` | ROS 子进程启动、就绪检查、所有权和日志 | 视觉判定 |
| `control/grasp.py` | 有 Tag、无 Tag 抓取与投递子进程接口 | 巡线状态机 |
| `task/competition.py` | 九路口、红绿灯、任务 YOLO、抓取和投递状态机 | 底层机械臂动作实现 |

旧文件：

```text
/home/eaibot/robocom_ws/src/line_cy_task.py
/home/eaibot/robocom_ws/src/line_cy_new.py
```

它们是回退或单路口调试版本。新版不得运行时导入旧文件，也不得为了“保持一致”批量覆盖它们。

## 4. 机器人外部依赖文件

抓取流程还依赖机器人上的：

```text
/home/eaibot/handeye-calib/src/tag_chassis_align_pick_sequence.py
/home/eaibot/handeye-calib/src/mirobot_pick_test_tag.py
/home/eaibot/handeye-calib/src/block_pick_main.py
/home/eaibot/handeye-calib/src/mirobot_pick_test.py
/home/eaibot/handeye-calib/src/mirobot_delivery.py
/home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
```

模型：

```text
/home/eaibot/handeye-calib/src/model/yolov5/rub_roll_new_yolov5n_320_best.onnx
/home/eaibot/handeye-calib/src/model/yolov5/building_new_yolov5n_320_best.onnx
/home/eaibot/handeye-calib/src/model/yolov5/traffic_lights_yolov5n_320_best.onnx
/home/eaibot/handeye-calib/src/model/yolov5/block_occlusion_yolov5n_640_best.onnx
```

真机配置和示教结果：

```text
/home/eaibot/handeye-calib/config/astra_rgb_640x480.yaml
/home/eaibot/handeye-calib/config/tag_pick_place_presets.json
/home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
/home/eaibot/handeye-calib/config/delivery_presets.json
/home/eaibot/handeye-calib/config/untagged_delivery_presets.json
```

两种投递只共用 `delivery_presets.json` 的 `cargo_pick_joint_values_by_id`；
无 Tag 楼宇专用中转点和共享 P 读取 `untagged_delivery_presets.json`，
其中共享 P 保存在 ID1。

这些 JSON、相机内参和手眼标定结果属于真机数据。除非用户明确要求重新示教或标定，否则禁止覆盖。

## 5. 设备与进程所有权

当前配置：

```text
/dev/video4  巡线摄像头
/dev/video0  红绿灯摄像头，320x240
/dev/video2  任务 YOLO 当前 OpenCV 索引
Astra RGB    有 Tag 和无 Tag 抓取
```

注意：Astra 按 USB 设备由 ROS 驱动启动，不应只根据 `/dev/video2` 是否存在判断 Astra 是否正常。

正式两终端所有权：

```text
终端一 launch：启动或复用底盘
-> 临时启动 Astra
-> 启动或复用 MoveIt 和手眼 TF
-> 检查服务和 TF
-> 关闭临时 Astra
-> 保持终端一运行

终端二 main：直接使用常驻底盘/MoveIt/手眼 TF，不重复等待检查
-> 按阶段启停 Astra、Tag 栈、YOLO 和抓取子进程
-> 运行抓取、投递和九路口状态机
-> 退出时停车并只关闭终端二自己启动的进程
```

严禁增加 `detach()` 或“`launch.py` 退出、子进程继续常驻”的逻辑。常驻依赖由仍在前台运行的终端一持有；终端二复用它们但不关闭它们。

`/cmd_vel` 任何时刻只能有一个任务所有者：

- `velocity_owner="line"`：循迹状态机可以发布；
- `velocity_owner="grasp"`：循迹普通发布被禁止，抓取子进程控制底盘；
- 强制停车使用 `publish(0, 0, force=True)`。

不要新增第二套底盘发布器来“补偿”现有流程。

## 6. 主状态机

主要状态：

```text
B_PICK_PREPARE
B_PICKING
FOLLOW
YOLO_STOP
DELIVERING
APPROACH
ALIGN
TRAFFIC_WAIT
MANEUVER
EXIT_ALIGN
A_PICK_PREPARE
A_PICK_SEARCH
A_PICKING
PICK_RECOVER
PICK_FAILED
FINAL_EXIT
DONE
```

普通路口：

```text
FOLLOW
-> 识别入口横条
-> APPROACH
-> 横条进入停车区
-> ALIGN
-> 摆正
-> TRAFFIC_WAIT
-> 稳定绿灯
-> MANEUVER
-> 入口斑马线消失
-> 识别出口横条
-> EXIT_ALIGN
-> 下一个路口
```

不要把入口横条、出口横条和任务 YOLO 停车事件混为一个触发条件。

## 7. B 点有 Tag 抓取与第一次右转

开启 `--tag-pick` 时，主程序在创建 `LaneFollower` 前启动 Astra、机械臂公共栈、Tag 补白和 AprilTag。初始状态是 `B_PICK_PREPARE`。

流程：

```text
B_PICK_PREPARE
-> B_PICKING
-> 依次按画面从左到右抓取
-> 读取实际 completed_ids
-> 停止 Tag 栈并释放 Astra
-> 加载 street 任务 YOLO
-> TRAFFIC_WAIT
-> 稳定绿灯
-> MANEUVER/ENTRY
-> MANEUVER/TURN
-> MANEUVER/EXIT_STRAIGHT
-> 出口横条
-> EXIT_ALIGN
-> 完成第 1 个路口
```

车辆抓取时已经越过第 1 个入口横条。因此 B 点抓取成功后禁止改为 `FOLLOW` 或 `PICK_RECOVER`，否则程序会寻找不存在的入口横条。

第一次右转专用参数：

```python
TAG_PICK_FIRST_ENTRY_TIME = 5.5
TAG_PICK_FIRST_TURN_TIME = 4.0
```

仅当以下条件同时成立时使用：

```text
tag_pick_first_maneuver == True
task_index == 0
turn_cmd == "right"
```

第 1 个路口完成后必须清除 `tag_pick_first_maneuver`。后续路口继续使用 `TURN_ENTRY_TIME` 和 `TURN_TIME`。

## 8. A 点无 Tag 搜索与抓取

A 点触发点是第 3 个路口完成后，即第 3 个 `right` 与第 4 个 `left` 之间。

流程：

```text
第 3 个路口完成
-> 关闭任务 YOLO，释放共享相机
-> A_PICK_PREPARE，底盘停车
-> 启动 Astra、机械臂公共栈、无 Tag 父子进程
-> 子进程加载模型并进入搜索循环，写 ready 文件
-> A_PICK_SEARCH，固定零角速度直行，不使用 camera4 车道中心修正
-> Astra 只检测画面右侧搜索区
-> 目标稳定出现，子进程写 trigger 文件
-> 主状态机先强制停车
-> velocity_owner 改为 grasp
-> 主状态机写 release 文件
-> 子进程此时才创建 `/cmd_vel` 发布器并慢速对准
-> A_PICKING
-> 完成后关闭 Astra、恢复 building YOLO
-> 等待绿灯
-> 使用 A 点独立直行和左转时间通过第 4 个路口
-> 识别出口横条并摆正后恢复普通流程
```

搜索区：

```python
UNTAGGED_SEARCH_ROI = (0.60, 0.05, 0.98, 0.95)
UNTAGGED_SEARCH_STABLE_FRAMES = 3
UNTAGGED_SEARCH_POLL_HZ = 3.0
UNTAGGED_SEARCH_FORWARD_SPEED = 0.16
UNTAGGED_PICK_NEXT_ENTRY_TIME = 5.5
UNTAGGED_PICK_NEXT_TURN_TIME = 4.0
```

抓取区来自：

```text
/home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
grasp_roi_ratio: [0.06, 0, 0.24, 1]
```

右侧搜索区与左侧抓取区用途不同，禁止合并成一个 ROI。

如果已经到达第 4 个路口入口但 A 点仍未触发，当前设计进入 `PICK_FAILED`，防止跳过必须抓取的物资。不要擅自改成继续循迹。

## 9. 红绿灯逻辑

红绿灯模型只在 `TRAFFIC_WAIT` 加载，摄像头为 `/dev/video0`。

```python
TRAFFIC_LIGHT_CONFIDENCE = 0.55
TRAFFIC_GREEN_STABLE_FRAMES = 2
```

规则：

- 绿灯达到稳定帧数才进入 `MANEUVER`；
- 红灯、黄灯、无检测、模型加载失败或相机读取失败都保持停车；
- 离开 `TRAFFIC_WAIT` 时关闭模型和相机，避免占用算力；
- B 点抓取后的第一次右转也必须先经过 `TRAFFIC_WAIT`。

禁止增加“等待超时后默认放行”。

## 10. 任务 YOLO 与投递

第一圈使用 street 模型：

```text
普通人群 -> 有 Tag ID1
医疗人群 -> 有 Tag ID2
可回收垃圾 -> 有 Tag ID3
其他垃圾 -> 有 Tag ID4
```

后两圈使用 building 模型：

```text
电力故障楼宇 -> 无 Tag ID1
火灾楼宇 -> 无 Tag ID2
有毒气体楼宇 -> 无 Tag ID3
坍塌楼宇 -> 无 Tag ID4
```

投递只消费抓取进程返回的实际库存，不能根据检测类别伪造库存。

投递失败策略已由用户确定：终端报警、记录失败 ID、不自动重试、恢复循迹。不要改成永久停车。

抓取失败策略：进入 `PICK_FAILED` 并停车。不要改成静默跳过。

## 11. 调试窗口

```python
PICK_DEBUG_VIEW = True
DEBUG_VIEW = True
```

应出现：

- `tag_pick_detection`：B 点 Tag ID 框、中心点、红色对准 ROI；
- 无 Tag 抓取窗口：类别框、置信度、搜索或抓取 ROI；
- `line_cy_task_processed`：巡线二值图与横条；
- 任务 YOLO 窗口；
- 红绿灯窗口。

B 点窗口实现位于：

```text
/home/eaibot/handeye-calib/src/tag_chassis_align_pick_sequence.py
```

总任务通过 `control/grasp.py` 的 `--show-debug-window` 打开它。
正式比赛只显示这一个 B 点合成窗口：AprilTag 仍发布 `/tag_detections_image`，但 `show_image=false`；YOLO relay 不对送入 AprilTag 的图像画框。合成窗口在检测底图上再画 ID、中心点和红色 ROI，避免重复框和多窗口。

窗口不显示的排查顺序：

1. 确认机器人 `config.py` 中 `PICK_DEBUG_VIEW=True`；
2. 查看 `pick_tag.log` 中实际命令是否有 `--show-debug-window`；
3. 检查 `/tag_detections_image` 是否发布；
4. 检查 `DISPLAY` 和 OpenCV GUI；
5. 最后才修改代码。

不要用调高置信度或修改状态机解决“窗口没弹出”。

## 12. 日志与故障定位

机器人日志目录：

```text
/home/eaibot/logs/zcy_last/<启动时间>/
```

正式两终端流程会生成两个启动时间目录：终端一保存底盘、MoveIt 和手眼 TF 日志，终端二保存 Astra、Tag、抓取和任务日志。

常见日志：

```text
base.log
astra.log
moveit.log
handeye_tf.log
tag_relay.log
apriltag.log
pick_tag.log
pick_untagged.log
delivery.log
```

定位故障时必须收集：

1. 终端完整状态转换和时间戳；
2. 当前 `task/index/state/cmd/phase/side` 调试文字；
3. 对应检测窗口截图；
4. 对应子进程日志末尾；
5. 实际运行命令；
6. 机器人上相关文件版本或差异。

按故障层处理：

| 现象 | 先检查 |
| --- | --- |
| 一键启动失败 | `base.log`、`astra.log`、`moveit.log`、`handeye_tf.log` |
| B 点无窗口 | `pick_tag.log` 参数、`/tag_detections_image`、DISPLAY |
| Tag 有框但不移动 | 检测 JSON、ROI、`velocity_owner`、`/cmd_vel` 发布者 |
| 绿灯不放行 | 红绿灯窗口、camera0、置信度和稳定帧数 |
| 第一次右转时机不对 | 两个 `TAG_PICK_FIRST_*` 参数 |
| 普通路口转弯不对 | `TURN_ENTRY_TIME`、`TURN_TIME`、`TURN_ANGULAR` |
| A 点不触发 | `pick_untagged.log`、右侧搜索框、ready/trigger 文件 |
| A 点抓取后第 4 个左转时机不对 | 两个 `UNTAGGED_PICK_NEXT_*` 参数 |
| 抓取后任务 YOLO 不恢复 | Astra 是否释放、camera2 是否占用、模型加载日志 |
| 出口被当入口 | 先看状态和时间戳，不要直接增加保护时间 |

## 13. 修改纪律

后续 AI 必须遵守：

1. 先读取当前代码和日志，不根据旧对话猜实现。
2. 一次只解决一个可复现原因。
3. 优先调整已有参数；只有现有状态表达不了需求时才新增状态或字段。
4. 不得自行增加新的忽略时间、连续帧保护、超时放行、超时完成、开环补偿或自动恢复。
5. 任何新增保护必须先向用户说明：复现、触发条件、正常路径影响和可调参数，得到确认后再实现。
6. 不得用 `MANEUVER_MAX_TIME`、`FOLLOW` 兜底掩盖入口、出口、红绿灯或抓取失败。
7. 不得同时修改巡线 PD、横条识别和状态机来碰运气。
8. 不得覆盖真机示教 JSON、相机内参和手眼标定。
9. 不得重构已工作的机械臂动作源码，除非用户明确要求机械臂动作本身改变。
10. 不得批量同步旧版 `line_cy_task.py`、`line_cy_new.py` 与新版。
11. 不得回退工作区中不是本次产生的修改。
12. 状态机修改必须添加状态转换测试；命令接口修改必须测试实际参数列表。

## 14. 同步到机器人

开发机修改根目录：

```text
F:\桌面\平安城市\国赛\code\eaibot
```

机器人目标根目录：

```text
/home/eaibot
```

至少要整体同步新版目录：

```text
开发机：F:\桌面\平安城市\国赛\code\eaibot\robocom_ws\src\zcy_last\
机器人：/home/eaibot/robocom_ws/src/zcy_last/
```

如果改动涉及抓取子进程，还要同步对应 `handeye-calib/src` 文件。当前新流程依赖：

```text
tag_chassis_align_pick_sequence.py
block_pick_main.py
mirobot_pick_test.py
```

不要只复制 `competition.py`，否则 `config.py`、`grasp.py`、`processes.py`、抓取脚本接口可能版本不一致。

同步后在机器人上检查：

```bash
python3 -m py_compile \
  /home/eaibot/robocom_ws/src/zcy_last/config.py \
  /home/eaibot/robocom_ws/src/zcy_last/control/processes.py \
  /home/eaibot/robocom_ws/src/zcy_last/control/grasp.py \
  /home/eaibot/robocom_ws/src/zcy_last/task/competition.py \
  /home/eaibot/robocom_ws/src/zcy_last/main.py \
  /home/eaibot/robocom_ws/src/zcy_last/launch.py
```

## 15. 测试与真机验收

开发机是 Windows，当前默认使用 Python 3.9：

```powershell
$env:PYTHONPATH='F:\桌面\平安城市\国赛\code\eaibot\robocom_ws\src'
& 'C:\Users\zhuch\AppData\Local\Programs\Python\Python39\python.exe' -m pytest -q `
  'F:\桌面\平安城市\国赛\code\eaibot\robocom_ws\src\zcy_last\tests'
```

测试数量以本次实际输出为准，不要沿用旧文档中的数量。测试通过只证明纯逻辑和接口通过。

真机下一步必须依次验收：

1. 终端一 `python3 -m zcy_last.launch` 能就绪底盘、MoveIt 和 TF，关闭临时 Astra，但不启动任务；
2. 终端二 `python3 -m zcy_last.main` 只复用常驻依赖；终端二退出时常驻依赖不退出，终端一 `Ctrl+C` 后才安全关闭；
3. B 点 `tag_pick_detection` 窗口正常显示；
4. B 点抓取完成后绿灯前底盘保持停止；
5. 绿灯后第一次直行与右转时间可独立调节；
6. 第一次右转后能识别出口横条并进入第 2 个路口；
7. A 点搜索阶段只有总调度发布固定直行 `/cmd_vel`，不执行巡线修正；
8. A 点右侧触发后先停车再由抓取接管；
9. A 点抓取完成后 Astra 释放、楼宇 YOLO 恢复，等待绿灯并用独立时序完成第 4 个路口；
10. 投递成功更新库存，投递失败报警后继续。

未完成上述真机验收前，不要宣称整套比赛流程已经完成。
