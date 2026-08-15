# 九路口比赛任务

`zcy_last` 是九路口循迹、任务识别、红绿灯等待、机械臂抓取和物资投递的一键启动版本。旧版 `line_cy_task.py` 保留为比赛回退入口，新版不会在运行时导入旧文件。

## 目录

```text
zcy_last/
├── launch.py               # 终端一：启动并持有常驻 ROS 依赖
├── main.py                 # 终端二：用户按需启动的比赛任务
├── config.py               # 比赛开关、设备号、路径和调节参数
├── algorithms/             # 巡线、横条、补线、YOLO 和红绿灯算法
├── control/                # 摄像头、底盘、抓取协调和子进程所有权
├── task/                   # 九路口路线与比赛状态机
└── tests/                  # 新旧算法对比和抓取流程测试
```

## 正式启动（两个终端）

终端一启动 `launch.py`：启动或复用底盘，临时启动 Astra 建立相机坐标系，再启动 MoveIt 和手眼 TF。就绪后关闭临时 Astra，底盘、MoveIt 和手眼 TF 保持常驻。`launch.py` 不会启动比赛状态机：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/robocom_ws/src
python3 -m zcy_last.launch
```

用户决定是否在终端二启动 `main.py`。`main` 直接使用终端一的常驻依赖，不再重复等待底盘、MoveIt、机械臂服务和手眼 TF，只对 Astra、Tag 节点、YOLO 和抓取子进程进行按阶段托管：

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

终端二退出不会关闭终端一的常驻依赖。不再需要时，在终端一按 `Ctrl+C`，`launch.py` 会逆序关闭自己启动的进程。

## 启动任务

在机器人上先进入 ROS 和 `ww` 环境：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/robocom_ws/src
```

固定五种比赛组合：

```bash
# 1. 只循迹，A/B 都不抓取
python3 -m zcy_last.main \
  --no-tag-pick --no-untagged-pick

# 2. B 点抓取 4 个并投递
python3 -m zcy_last.main \
  --tag-pick --tag-pick-count 4 --tag-delivery \
  --no-untagged-pick

# 3. A 点抓取 4 个并投递
python3 -m zcy_last.main \
  --no-tag-pick \
  --untagged-pick --untagged-pick-count 4 --untagged-delivery

# 4. A/B 都抓取 4 个并投递
python3 -m zcy_last.main \
  --tag-pick --tag-pick-count 4 --tag-delivery \
  --untagged-pick --untagged-pick-count 4 --untagged-delivery

# 5. A/B 都抓取 4 个，都不投递
python3 -m zcy_last.main \
  --tag-pick --tag-pick-count 4 --no-tag-delivery \
  --untagged-pick --untagged-pick-count 4 --no-untagged-delivery
```

命令行开关优先于 `config.py` 顶部默认值。抓取数量范围为 `1..4`。B 点按当前画面从左到右尝试最多该数量；目标未出现、对准超时、Tag TF 超时或限位未接触时跳过该 ID，并按实际成功库存继续比赛，允许最终库存为空。A 点仍要求完成指定数量。

B 点有 Tag 和 A 点无 Tag 各有独立抓取、投递开关。临时联调单个点只抓取不投递时，分别使用 `--no-tag-delivery` 或 `--no-untagged-delivery`；这类调试写法不计入正式五种比赛组合。投递只有在对应抓取开启且库存中存在目标 ID 时才执行。

B 点投递映射：普通人群 `1`、医疗人群 `2`、可回收垃圾 `3`、其他垃圾 `4`。A 点投递映射：电力故障楼宇 `1`、火灾楼宇 `2`、有毒气体楼宇 `3`、坍塌楼宇 `4`。程序只使用抓取脚本返回的实际成功 ID，不使用计划抓取数量代替库存。

`python3 -m zcy_last.main` 是唯一任务入口。全部 ROS 依赖都由人工管理时可再加 `--external-ros`；此模式不托管、不检查也不关闭外部进程，只用于特殊联调。

调试 A 点抓取和后续楼宇投递时，可将车放在第 3 个右转完成、出口横条摆正后的正常
位置和朝向，再从终端二直接运行：

```bash
python3 -m zcy_last.main \
  --start-untagged-aligned \
  --untagged-pick-count 4 \
  --untagged-delivery
```

该开关自动跳过 B 点和前三个路口，从内部 `task_index=3`、
`A_PICK_PREPARE` 开始；A 点完成后仍按正常流程等待绿灯、执行第 4 个路口左转并继续
剩余路线。它不能与 `--tag-pick` 或 `--no-untagged-pick` 同时使用。启动后底盘会
真实运动，因此只用于已正确摆车的现场调试。

## 状态机

```text
STARTUP
  -> B_PICK_PREPARE -> B_PICKING -> TRAFFIC_WAIT -> MANEUVER
  -> FOLLOW -> YOLO_STOP -> DELIVERING -> FOLLOW                 # 街区
  -> FOLLOW -> YOLO_STOP -> DELIVERING -> FOLLOW                 # 楼宇
  -> APPROACH -> ALIGN -> TRAFFIC_WAIT
  -> MANEUVER -> EXIT_ALIGN
  -> A_PICK_PREPARE -> A_PICK_SEARCH -> A_PICKING
  -> TRAFFIC_WAIT -> MANEUVER -> EXIT_ALIGN
  -> FINAL_EXIT -> DONE
```

实际开启 B 点抓取时，`B_PICKING` 成功后不进入普通 `PICK_RECOVER`：车辆已经越过首个入口横条，会直接进入 `TRAFFIC_WAIT`，确认绿灯后按 `TAG_PICK_FIRST_ENTRY_TIME` 直行、按 `TAG_PICK_FIRST_TURN_TIME` 执行第一次右转，再沿用 `MANEUVER -> EXIT_ALIGN` 识别出口并完成第 1 个路口计数。

抓取关闭时会跳过对应状态。B 点只在循迹开始前执行一次；A 点只在路线第 3 个路口 `right` 完成后执行一次。开启 A 点抓取时，第 3 个右拐独立使用 `A_PICK_THIRD_RIGHT_ENTRY_TIME` 前进和 `A_PICK_THIRD_RIGHT_TURN_TIME` 右转，其他普通路口仍使用 `TURN_ENTRY_TIME/TURN_TIME`。A 点进入 `A_PICK_PREPARE` 后停车加载 Astra、模型、识别窗口和抓取子进程；窗口即使零检测也持续显示原始 RGB。全部就绪后切换到独立的 20Hz `A_PICK_SEARCH` 控制循环，从第一次实际发送普通 `FOLLOW_SPEED`、零角速度命令开始计满 `UNTAGGED_SEARCH_FORWARD_TIME`，再降至 `UNTAGGED_SEARCH_SPEED` 边走边搜索，不使用车道中心修正，也不因第 4 个路口横条停车。要求数量的不同无 Tag 目标在 Astra 全画面连续确认后，车辆先停车，再将底盘控制权交给抓取子进程低速逐个对准抓取 ROI。抓取成功后不进入 `PICK_RECOVER/FOLLOW`，而是等待绿灯，再按 `UNTAGGED_PICK_NEXT_ENTRY_TIME` 直行、按 `UNTAGGED_PICK_NEXT_TURN_TIME` 完成第 4 个路口左转，识别出口横条后恢复普通流程。第一圈街区识别消费 B 点库存，后两圈楼宇识别消费 A 点库存。街区仍按固定点直接投递；楼宇框中心进入原始320×240画面的红框 `x=54~173` 就按原 `YOLO_STOP` 停车，不旋转，也不根据距离二次移动底盘。标定和正式运行显示并使用同一红框；只用停车框的左右宽度模型估算真实毫米距离，上下裁切不影响，机械臂据此沿前探轴修正以 `450mm` 为参考示教的 P，再执行原有限位接触投递。

A/B 单目标失败（未出现、对准超时、TF/限位或该次机械臂动作失败）不会进入 `PICK_FAILED`，会停车并跳过当前目标、继续尝试剩余目标，最后按实际库存比赛。只有父进程自身无法继续运行、结果文件损坏或托管依赖退出等整批故障才停车。投递失败只输出警告并继续。A 点逐个慢速对齐时窗口显示所有剩余目标，控制器只对当前目标计算速度，成功后依次减少一个框。

## 摄像头切换

```text
/dev/video4  巡线，任务全程占用
/dev/video0  红绿灯，只在停止线摆正后打开模型
/dev/video2  任务 YOLO 当前配置的 OpenCV 索引
```

`launch.py` 的启动顺序为底盘 -> 临时 Astra -> MoveIt/手眼 TF -> 关闭临时 Astra，因为手眼发布器需要 Astra 先提供 `camera_link` 内部 TF。`main.py` 执行 B 点时只重新启动 Astra 并启动 Tag 补白/AprilTag，直接使用终端一的常驻机械臂公共栈。Astra 优先使用项目指定的 RGB 内参文件；文件不存在时自动按驱动默认方式启动，但仍要求实际发布的 `CameraInfo.K` 非空。B 点结束后关闭 Astra，再由任务 YOLO 打开 `/dev/video2`。第 3 个路口后先关闭任务 YOLO，再启动 Astra；搜索循环就绪后由 Astra 检查画面右侧的无 Tag 物块。抓取结束后关闭 Astra 并加载楼宇模型。

人偶多数类别默认连续确认 `7` 帧，垃圾桶类别默认连续确认 `2` 帧才停车，分别由 `YOLO_PEOPLE_STABLE_FRAMES` 和 `YOLO_TRASH_STABLE_FRAMES` 调整；楼宇不使用这两个帧数。

进程管理器会检查底盘节点、共享相机占用、Astra 内参、MoveIt 状态、机械臂服务和手眼 TF。日志保存在 `/home/eaibot/logs/zcy_last/<启动时间>/`。`launch` 退出时先停车，再逆序关闭本次启动的全部进程；检测到并复用的人工进程不属于本程序，不会被关闭。

## 故障处理

- `PICK_FAILED` 不自动重试，也不恢复循迹，需要人工排查后重启任务。
- 正式运行前确认 `/home/eaibot/handeye-calib/config/astra_rgb_640x480.yaml` 已安装且内参非零。
- 不要在一键启动期间手动运行抓取脚本或另一个 `/cmd_vel` 发布程序。
- 启动失败时先看终端错误，再查看对应的 `astra.log`、`moveit.log` 或 `pick_*.log`。

## 测试

```bash
cd /home/eaibot
PYTHONPATH=robocom_ws/src python3 -m pytest -q \
  robocom_ws/src/zcy_last/tests \
  handeye-calib/tests/test_tag_chassis_align_pick_sequence.py \
  handeye-calib/tests/test_mirobot_block_mono.py \
  handeye-calib/tests/test_block_pick_main.py
```

单元测试不代替真机验收。比赛前仍需验证 `/dev/video2` 的实际释放顺序、抓取期间 `/cmd_vel` 单一所有者，以及抓取后的车道恢复。
