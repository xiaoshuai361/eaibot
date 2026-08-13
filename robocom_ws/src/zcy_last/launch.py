#!/usr/bin/env python3
# coding=utf-8
"""基础依赖启动器：常驻底盘、MoveIt 和手眼 TF，退出时自动清理。"""

import signal
import sys
import time

from . import config
from .control.processes import ProcessSupervisor


def _request_shutdown(signum, _frame):
    print("[zcy_last] 依赖启动器收到退出信号 %d，准备清理进程" % signum,
          flush=True)
    raise KeyboardInterrupt


def launch_dependencies(supervisor):
    """启动常驻依赖，并在手眼 TF 就绪后释放临时 Astra。"""
    supervisor.start_base()
    supervisor.start_astra()
    supervisor.start_arm_common()
    supervisor.stop_astra()
    print(
        "[zcy_last] 基础依赖已就绪：底盘、MoveIt 和手眼 TF 常驻，"
        "临时 Astra 已关闭",
        flush=True,
    )
    print(
        "[zcy_last] 可在另一终端按需运行 python3 -m zcy_last.main；"
        "本终端按 Ctrl+C 后自动清理",
        flush=True,
    )


def main():
    supervisor = ProcessSupervisor(
        enabled=config.MANAGE_ROS_PROCESSES,
        python3=sys.executable,
    )
    old_handlers = {}
    shutdown_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        shutdown_signals.append(signal.SIGHUP)
    for signum in shutdown_signals:
        old_handlers[signum] = signal.signal(signum, _request_shutdown)
    try:
        launch_dependencies(supervisor)
        while True:
            time.sleep(1.0)
    finally:
        print("[zcy_last] 正在关闭本启动器启动的常驻进程", flush=True)
        try:
            supervisor.shutdown()
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        print("[zcy_last] 基础依赖已清理完成", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
