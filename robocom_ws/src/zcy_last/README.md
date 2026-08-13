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

# 3. B 点只抓取 4 个，不投递
python3 -m zcy_last.main \
  --tag-pick --tag-pick-count 4 --no-tag-delivery \
  --no-untagged-pick

# 4. A 点抓取 3 个并投递
python3 -m zcy_last.main \
  --no-tag-pick \
  --untagged-pick --untagged-pick-count 3 --untagged-delivery

# 5. A 点只抓取 3 个，不投递
python3 -m zcy_last.main \
  --no-tag-pick \
  --untagged-pick --untagged-pick-count 3 --no-untagged-delivery
```

命令行开关优先于 `config.py` 顶部默认值。抓取数量范围为 `1..4`，表示必须成功完成的数量；目标按当前画面从左到右选择，目标不足或任一目标失败都会进入 `PICK_FAILED` 永久停车。

B 点有 Tag 和 A 点无 Tag 各有独立抓取、投递开关。临时只抓取不投递分别加 `--no-tag-delivery` 或 `--no-untagged-delivery`；投递只有在对应抓取开启且库存中存在目标 ID 时才执行。

B 点投递映射：普通人群 `1`、医疗人群 `2`、可回收垃圾 `3`、其他垃圾 `4`。A 点投递映射：电力故障楼宇 `1`、火灾楼宇 `2`、有毒气体楼宇 `3`、坍塌楼宇 `4`。程序只使用抓取脚本返回的实际成功 ID，不使用计划抓取数量代替库存。

`python3 -m zcy_last.main` 是唯一任务入口。全部 ROS 依赖都由人工管理时可再加 `--external-ros`；此模式不托管、不检查也不关闭外部进程，只用于特殊联调。

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

抓取关闭时会跳过对应状态。B 点只在循迹开始前执行一次；A 点只在路线第 3 个路口 `right` 完成后执行一次。A 点先在 `A_PICK_PREPARE` 停车加载 Astra、模型和抓取子进程；子进程进入检测循环后切换到 `A_PICK_SEARCH`，车辆以 `UNTAGGED_SEARCH_FORWARD_SPEED` 保持零角速度直行，不使用车道中心修正。无 Tag 目标在 Astra 画面右侧搜索区连续确认后，车辆先停车，再将底盘控制权交给抓取子进程低速对准左侧抓取 ROI。抓取成功后不进入 `PICK_RECOVER/FOLLOW`，而是等待绿灯，再按 `UNTAGGED_PICK_NEXT_ENTRY_TIME` 直行、按 `UNTAGGED_PICK_NEXT_TURN_TIME` 完成第 4 个路口左转，识别出口横条后恢复普通流程。第一圈街区识别消费 B 点库存，后两圈楼宇识别消费 A 点库存。街区仍按固定点直接投递；楼宇框中心进入原始320×240画面的红框 `x=54~173` 就按原 `YOLO_STOP` 停车，不旋转，也不根据距离二次移动底盘。标定和正式运行显示并使用同一红框；停车框的宽、高双模型只估算真实毫米距离，机械臂据此沿前探轴修正以 `450mm` 为参考示教的 P，再执行原有限位接触投递。

抓取、依赖进程或车道恢复失败会进入 `PICK_FAILED`。投递失败例外：终端输出警告、该 ID 不再自动重试，并继续循迹。

## 摄像头切换

```text
/dev/video4  巡线，任务全程占用
/dev/video0  红绿灯，只在停止线摆正后打开模型
/dev/video2  任务 YOLO 当前配置的 OpenCV 索引
```

`launch.py` 的启动顺序为底盘 -> 临时 Astra -> MoveIt/手眼 TF -> 关闭临时 Astra，因为手眼发布器需要 Astra 先提供 `camera_link` 内部 TF。`main.py` 执行 B 点时只重新启动 Astra 并启动 Tag 补白/AprilTag，直接使用终端一的常驻机械臂公共栈。Astra 优先使用项目指定的 RGB 内参文件；文件不存在时自动按驱动默认方式启动，但仍要求实际发布的 `CameraInfo.K` 非空。B 点结束后关闭 Astra，再由任务 YOLO 打开 `/dev/video2`。第 3 个路口后先关闭任务 YOLO，再启动 Astra；搜索循环就绪后由 Astra 检查画面右侧的无 Tag 物块。抓取结束后关闭 Astra 并加载楼宇模型。

人偶多数类别默认需要连续确认 `6` 帧才停车，由 `YOLO_PEOPLE_STABLE_FRAMES` 调整；垃圾桶和楼宇不使用该帧数。

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
