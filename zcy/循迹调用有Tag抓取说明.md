# 循迹程序调用机械臂

本文档供 AI 编写循迹联动代码时使用。

## 前置条件

Tag 抓取前，《机械臂操作.md》中的终端 0-5 必须正常运行。投递时至少保持终端 2 运行。

循迹程序的启动终端先执行：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
```

## 接口 1：底盘对准并抓取入仓

```bash
python2 /home/eaibot/handeye-calib/src/tag_chassis_align_pick_sequence.py \
  --sequence 1,2,3,4 \
  --tag-tf-wait-seconds 18.0 \
  --fail-on-skip
```

该命令会自动完成：底盘对准红框、Tag 抓取、放入对应载物仓、回 idle、启动回零。

`--sequence` 是允许处理的 ID 集合。默认每次抓当前画面中最靠左的剩余 ID，不强制按数字顺序。

## 接口 2：底盘已对准，单独抓取

```bash
python2 /home/eaibot/handeye-calib/src/mirobot_pick_test_tag.py \
  --mode run_taught_sequence \
  --sequence 3 \
  --preset-file /home/eaibot/handeye-calib/config/tag_pick_place_presets.json \
  --velocity-scale 0.2 \
  --acceleration-scale 0.2 \
  --tf-timeout 12.0 \
  --home-after-idle
```

将 `3` 替换为目标 ID。该命令不控制底盘。

## 接口 3：从载物仓投递

```bash
python2 /home/eaibot/handeye-calib/src/mirobot_delivery.py \
  --mode run_delivery \
  --sequence 1,2,3,4 \
  --delivery-file /home/eaibot/handeye-calib/config/delivery_presets.json \
  --tag-preset-file /home/eaibot/handeye-calib/config/tag_pick_place_presets.json
```

`--sequence` 必须只包含载物仓中实际存在的 ID。投递不使用视觉。

比赛任务会让有 Tag 抓取脚本通过 `--result-file` 返回实际成功入仓的 ID，
第一圈任务 YOLO 识别并停车后按下表投递：

| 识别目标 | Tag ID | 物资 |
| --- | ---: | --- |
| 普通人群 | 1 | 基本生活物资 |
| 医疗人群 | 2 | 医疗包 |
| 可回收垃圾 | 3 | 常规消杀剂 |
| 其他垃圾 | 4 | 生物危害专用消杀剂 |

只有对应 ID 确实在载物仓库存中才启动投递。成功后从库存移除该 ID；
投递失败时终端报警、不自动重试，并恢复循迹继续比赛。

## Python 阻塞调用

```python
import subprocess


def run_robot_task(command):
    return subprocess.call(command) == 0
```

调用前必须停止底盘，并暂停循迹程序对 `/cmd_vel` 的发布。子进程返回后：

- 返回码 `0`：任务成功，可恢复循迹。
- 单独调试脚本时返回码非 `0`：保持停车并人工处理。
- 由 `zcy_last.main` 托管时返回码非 `0`：输出投递失败警告，不自动重试，继续循迹。

## 必须遵守

- 循迹程序、键盘控制和 `tag_chassis_align_pick_sequence.py` 不得同时发布 `/cmd_vel`。
- 同一站点只触发一次机械臂任务，必须用状态标志防止重复调用。
- 机械臂任务未返回时，底盘保持停止。
- 抓取失败后不要盲目重试。
- 投递失败且吸泵仍开启时，不要自动关泵，避免物块在未知位置掉落。
- 比赛主任务虽会按配置继续循迹，但终端出现投递失败警告时应立即关注机械臂和吸泵状态。
