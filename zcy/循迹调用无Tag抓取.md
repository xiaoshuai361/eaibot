# 循迹调用无 Tag 抓取

循迹程序负责底盘，无 Tag 程序只执行单个物块的“抓取 → 中转 → 放置 → idle”。不要在循迹中调用 `--run-chassis-sequence`，否则两个程序会同时控制 `/cmd_vel`。

## 1. 启动环境

运行循迹前先加载完整环境，并确保 Astra RGB、Mirobot + MoveIt、手眼 TF 已启动：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/robocom_ws/src
```

目标编号：

```text
1=power  2=fire  3=gas  4=support
```

## 新版 `zcy_last` 自动抓取与投递

新版主任务会在第 3 个路口完成后独占底盘和 Astra，按画面从左到右抓取，并记录实际成功入仓的物资 ID。开启 A 点抓取和楼宇投递：

```bash
python3 -m zcy_last.main \
  --untagged-pick --untagged-pick-count 3 \
  --untagged-delivery
```

只抓取、不投递：

```bash
python3 -m zcy_last.main \
  --untagged-pick --untagged-pick-count 3 \
  --no-untagged-delivery
```

楼宇与物资 ID 的关系：电力故障 `1`、火灾 `2`、有毒气体 `3`、坍塌 `4`。只有实际库存中存在对应 ID 才启动投递。投递失败只在终端报警、不重试，并继续循迹。

A 点使用独立投递示教文件：

```text
/home/eaibot/handeye-calib/config/untagged_delivery_presets.json
```

比赛前依次示教载物仓抓取点、中转点和投递点：

```bash
python2 /home/eaibot/handeye-calib/src/mirobot_delivery.py \
  --mode teach_cargo_pick --sequence 1,2,3,4 \
  --delivery-file /home/eaibot/handeye-calib/config/untagged_delivery_presets.json \
  --tag-preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json

python2 /home/eaibot/handeye-calib/src/mirobot_delivery.py \
  --mode teach_transit \
  --delivery-file /home/eaibot/handeye-calib/config/untagged_delivery_presets.json \
  --tag-preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json

python2 /home/eaibot/handeye-calib/src/mirobot_delivery.py \
  --mode teach_release \
  --delivery-file /home/eaibot/handeye-calib/config/untagged_delivery_presets.json \
  --tag-preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

下面的单目标调用仅用于旧版或独立调试。运行 `zcy_last.main` 时不要同时手动启动这些抓取命令。

## 2. 单独测试调用

底盘停稳、目标中心已经进入红色 ROI 后，以 `1=power` 为例：

```bash
python3 /home/eaibot/handeye-calib/src/block_pick_main.py \
  --target 1 \
  --run-taught-block \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

返回码 `0` 表示抓放完成；非 `0` 表示失败，底盘必须保持停止，不能自动继续循迹。

## 3. 在 `line_cy_task.py` 中调用

文件顶部增加：

```python
import subprocess
import sys
```

在 `LaneFollower.__init__()` 增加：

```python
self.block_pick_process = None
self.block_pick_target = None
self.completed_block_picks = set()
```

在类中增加：

```python
def start_block_pick(self, target_id):
    target_id = int(target_id)
    if (self.block_pick_process is not None
            or target_id in self.completed_block_picks):
        return False
    self.publish(0, 0)
    command = [
        sys.executable,
        "/home/eaibot/handeye-calib/src/block_pick_main.py",
        "--target", str(target_id),
        "--run-taught-block",
        "--confidence", "0.5",
        "--config",
        "/home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml",
        "--preset-file",
        "/home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json",
    ]
    self.block_pick_target = target_id
    self.block_pick_process = subprocess.Popen(command)
    self._set_state("ARM_PICK")
    return True

def handle_block_pick(self):
    self.publish(0, 0)
    result = self.block_pick_process.poll()
    if result is None:
        return
    target = self.block_pick_target
    self.block_pick_process = None
    self.block_pick_target = None
    if result == 0:
        rospy.loginfo("无 Tag 目标 %d 抓放完成，恢复循迹", target)
        self.completed_block_picks.add(target)
        self.stop_hits = 0
        self._set_state("FOLLOW")
    else:
        rospy.logerr("无 Tag 目标 %d 抓放失败，底盘保持停止", target)
        self._set_state("PICK_FAILED")
```

在 `process()` 的状态分支中增加：

```python
elif self.state == "ARM_PICK":
    self.handle_block_pick()

elif self.state == "PICK_FAILED":
    self.publish(0, 0)
```

循迹到达物资点并确认底盘停稳后，只调用一次：

```python
if self.start_block_pick(1):  # 抓 power
    return
```

## 4. 必须满足

- 调用前先连续发布零速度，确认底盘完全停稳。
- 物块中心必须已经进入无 Tag 红色 ROI；该单目标入口不会移动底盘找物块。
- 四类抓取点、放置点、`carry_joint_values` 和 `idle_joint_values` 必须已经示教完成。
- 抓取期间循迹主循环继续运行，但 `ARM_PICK` 状态只能发布零速度。
- 同一个物资点要设置“已执行”标记，避免恢复 `FOLLOW` 后重复触发抓取。
