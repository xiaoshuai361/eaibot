# Git 使用说明

当前主仓库：

```bash
cd /home/zcy/eaibot
```

## 查看状态

```bash
git status
git log --oneline -5
```

## 保存当前状态

每次大改前或调通后执行：

```bash
git add -A
git commit -m "snapshot: 说明这次改了什么"
```

## 回退单个文件

例如只恢复 tag 抓取脚本：

```bash
git restore handeye-calib/src/mirobot_pick_test_tag.py
```

## 回退整个主仓库到上一次提交

先确认没有要保留的新文件：

```bash
git status
```

再恢复：

```bash
git restore .
```

如果还要删除未被 Git 管理的新文件：

```bash
git clean -fd
```

注意：`git clean -fd` 会删除未跟踪文件，慎用。

## 回到本次稳定快照

```bash
git reset --hard stable-workspace-baseline-20260724-104154
```

## 嵌套仓库

下面这些目录有自己的 Git 历史，根仓库不会直接管理它们内部修改：

```text
handeye-calib/src/aruco_ros
handeye-calib/src/easy_handeye
handeye-calib/src/vision_visp
mirobot_ws/src/apriltag_ros
mirobot_ws/src/astra_camera
robocom_ws/src/backward_ros
robocom_ws/src/astra_camera
```

如果要回退某个嵌套仓库：

```bash
cd /home/zcy/eaibot/mirobot_ws/src/apriltag_ros
git status
git reset --hard stable-workspace-baseline-20260724-104154
```

## 没有放进 Git 的东西

普通 Git 不管理这些大文件或临时文件：

- `*.pt`
- `*.onnx`
- `*.mp4`
- `build/`
- `devel/`
- `__pycache__/`
- `.pytest_cache/`

模型和视频需要单独备份。
