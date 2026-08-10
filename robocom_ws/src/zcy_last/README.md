# 九路口比赛任务

`zcy_last` 是九路口循迹、任务识别、红绿灯等待、机械臂抓取和物资投递的一键启动版本。旧版 `line_cy_task.py` 保留为比赛回退入口，新版不会在运行时导入旧文件。

## 目录

```text
zcy_last/
├── main.py                 # 比赛任务入口和抓取依赖进程托管
├── config.py               # 比赛开关、设备号、路径和调节参数
├── algorithms/             # 巡线、横条、补线、YOLO 和红绿灯算法
├── control/                # 摄像头、底盘、抓取协调和子进程所有权
├── task/                   # 九路口路线与比赛状态机
└── tests/                  # 新旧算法对比和抓取流程测试
```

## 启动底盘

底盘驱动必须在独立终端人工启动，`zcy_last` 不启动也不关闭它：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
roslaunch xpkg_bringup bringup_basic_ctrl.launch
```

任务启动时只检查 `/xnode_comm` 和 `/xnode_vehicle` 是否存在。

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

四种比赛组合：

```bash
# 纯循迹，不抓取
python3 -m zcy_last.main

# 仅开始前执行 B 点有 Tag 抓取，并在第一圈自动投递
python3 -m zcy_last.main --tag-pick --tag-pick-count 2

# 仅第 3 个路口完成后执行 A 点无 Tag 抓取，并在楼宇处自动投递
python3 -m zcy_last.main --untagged-pick --untagged-pick-count 3

# B 点和 A 点都抓取
python3 -m zcy_last.main \
  --tag-pick --tag-pick-count 2 \
  --untagged-pick --untagged-pick-count 3
```

命令行开关优先于 `config.py` 顶部默认值。抓取数量范围为 `1..4`，表示必须成功完成的数量；目标按当前画面从左到右选择，目标不足或任一目标失败都会进入 `PICK_FAILED` 永久停车。

B 点有 Tag 和 A 点无 Tag 各有独立抓取、投递开关。临时只抓取不投递分别加 `--no-tag-delivery` 或 `--no-untagged-delivery`；投递只有在对应抓取开启且库存中存在目标 ID 时才执行。

B 点投递映射：普通人群 `1`、医疗人群 `2`、可回收垃圾 `3`、其他垃圾 `4`。A 点投递映射：电力故障楼宇 `1`、火灾楼宇 `2`、有毒气体楼宇 `3`、坍塌楼宇 `4`。程序只使用抓取脚本返回的实际成功 ID，不使用计划抓取数量代替库存。

已经手动启动全部 ROS 依赖时，可在调试中增加 `--external-ros`。此模式不托管、不检查也不关闭外部进程，不用于正式比赛的一键启动。

## 状态机

```text
STARTUP
  -> B_PICK_PREPARE -> B_PICKING -> PICK_RECOVER
  -> FOLLOW -> YOLO_STOP -> DELIVERING -> FOLLOW
  -> APPROACH -> ALIGN -> TRAFFIC_WAIT
  -> MANEUVER -> EXIT_ALIGN
  -> A_PICK_PREPARE -> A_PICKING -> PICK_RECOVER
  -> FINAL_EXIT -> DONE
```

抓取关闭时会跳过对应状态。B 点只在循迹开始前执行一次；A 点只在路线第 3 个路口 `right` 完成后执行一次。抓取完成后必须连续稳定识别车道才能恢复 `FOLLOW`。第一圈街区识别消费 B 点库存，后两圈楼宇识别消费 A 点库存；投递时底盘停车，完成后恢复 `FOLLOW`。

抓取、依赖进程或车道恢复失败会进入 `PICK_FAILED`。投递失败例外：终端输出警告、该 ID 不再自动重试，并继续循迹。

## 摄像头切换

```text
/dev/video4  巡线，任务全程占用
/dev/video0  红绿灯，只在停止线摆正后打开模型
/dev/video2  Astra 与任务 YOLO 串行复用
```

B 点流程先由 Astra、Tag 补白和 AprilTag 使用 `/dev/video2`。B 点结束后释放相机，再加载人偶和垃圾桶模型。第 3 个路口后先关闭任务 YOLO，再启动 Astra 执行 A 点无 Tag 抓取。A 点结束后释放 Astra并加载楼宇模型。

进程管理器会检查外部底盘节点、共享相机占用、Astra 内参、MoveIt 状态、机械臂服务和手眼 TF。日志保存在 `/home/eaibot/logs/zcy_last/<启动时间>/`。程序退出时先停车，再逆序关闭本次启动的抓取相关进程，不关闭人工启动的底盘或其他外部进程。

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
