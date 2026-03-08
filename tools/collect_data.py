#!/usr/bin/env python3
"""
FluxChi — 引导式 EMG 数据采集工具
=========================================================
用 WAVELETECH 手环采集带标签的 EMG 数据，用于训练活动分类模型。

活动类别：
  typing          - 打字
  mouse_use       - 使用鼠标
  idle            - 静止/发呆
  stretching      - 伸展/活动

采集协议：
  - 每个活动 3 分钟连续录制
  - 每个活动重复 3 次
  - 建议跨 3-5 天采集

用法：
  python tools/collect_data.py --port /dev/tty.usbserial-XXXX
  python tools/collect_data.py --port /dev/tty.usbserial-XXXX --session 2
  python tools/collect_data.py --list-ports
=========================================================
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stream import CHANNEL_COUNT, EMGSample, SerialEMGStream

ACTIVITIES = [
    ("typing", "打字（正常速度在键盘上打字）", 180),
    ("mouse_use", "使用鼠标（移动、点击、滚动）", 180),
    ("idle", "静止放松（手臂自然放在桌上不动）", 180),
    ("stretching", "伸展活动（手腕旋转、手指伸展、握拳）", 180),
]

DATA_DIR = Path("data")

_interrupted = False


def _signal_handler(sig, frame):
    global _interrupted
    _interrupted = True
    print("\n\n  [!] 收到中断信号，正在安全停止...")


def list_serial_ports() -> List[str]:
    """List available serial ports."""
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()]
    except Exception:
        import glob
        return glob.glob("/dev/tty.usbserial*") + glob.glob("/dev/tty.usbmodem*")


def record_activity(
    stream: SerialEMGStream,
    label: str,
    duration_sec: int,
    session_id: int,
    rep_id: int,
) -> Optional[Path]:
    """Record EMG data for a specified duration, save to CSV."""
    global _interrupted

    output_dir = DATA_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"s{session_id:02d}_r{rep_id:02d}_{timestamp}.csv"
    output_path = output_dir / filename

    buffer: List[List[float]] = []
    start_time = time.perf_counter()
    last_print = start_time
    sample_count = 0

    print(f"\n  ● 录制中... [{label}] 剩余 {duration_sec}s", end="", flush=True)

    while not _interrupted:
        elapsed = time.perf_counter() - start_time
        if elapsed >= duration_sec:
            break

        samples = stream.consume_samples(max_items=512)
        for sample in samples:
            rel_t = sample.timestamp - start_time
            row = [rel_t] + sample.values.tolist()
            buffer.append(row)
            sample_count += 1

        now = time.perf_counter()
        if now - last_print >= 1.0:
            remaining = max(0, duration_sec - int(elapsed))
            rate = sample_count / max(elapsed, 0.001)
            print(f"\r  ● 录制中... [{label}] 剩余 {remaining:3d}s  |  {sample_count} samples ({rate:.0f} Hz)   ", end="", flush=True)
            last_print = now

        time.sleep(0.002)

    print()

    if not buffer:
        print("  [!] 没有采集到数据")
        return None

    header = ["t"] + [f"ch{i+1}" for i in range(CHANNEL_COUNT)]
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(buffer)

    elapsed_total = time.perf_counter() - start_time
    rate = len(buffer) / max(elapsed_total, 0.001)
    print(f"  ✓ 保存: {output_path}")
    print(f"    {len(buffer)} samples, {elapsed_total:.1f}s, {rate:.0f} Hz")

    return output_path


def run_session(port: str, session_id: int, activities: list, reps: int = 3):
    """Run a complete data collection session."""
    global _interrupted
    _interrupted = False
    signal.signal(signal.SIGINT, _signal_handler)

    print("\n" + "=" * 60)
    print(f"  FluxChi — 数据采集会话 #{session_id}")
    print("=" * 60)
    print(f"\n  串口: {port}")
    print(f"  活动数: {len(activities)}")
    print(f"  每活动重复: {reps} 次")
    total_min = sum(dur for _, _, dur in activities) * reps / 60
    print(f"  预计总时长: {total_min:.0f} 分钟")

    print("\n  正在连接手环...")
    try:
        stream = SerialEMGStream(port)
        stream.start()
    except Exception as e:
        print(f"\n  [ERROR] 无法连接: {e}")
        print("  请检查:")
        print("    1. 手环已开机并处于配对状态")
        print("    2. 串口名称正确 (用 --list-ports 查看)")
        print("    3. 没有其他程序占用该串口")
        return

    time.sleep(1.0)
    test_samples = stream.consume_samples(max_items=100)
    if not test_samples:
        print("  [WARN] 1秒内未收到数据，手环可能未就绪")
        print("  等待 3 秒再试...")
        time.sleep(3.0)
        test_samples = stream.consume_samples(max_items=100)
        if not test_samples:
            print("  [ERROR] 仍未收到数据，请检查手环连接")
            stream.stop()
            return

    print(f"  ✓ 连接成功！收到 {len(test_samples)} 个测试样本")

    collected_files = []

    for act_idx, (label, description, duration) in enumerate(activities):
        if _interrupted:
            break

        for rep in range(1, reps + 1):
            if _interrupted:
                break

            print(f"\n{'─' * 60}")
            print(f"  活动 {act_idx+1}/{len(activities)}: {description}")
            print(f"  重复 {rep}/{reps}  |  时长 {duration}s")
            print(f"{'─' * 60}")

            stream.consume_samples(max_items=0)

            input(f"\n  准备好后按 Enter 开始录制 [{label}]...")

            countdown = 3
            for i in range(countdown, 0, -1):
                print(f"  {i}...", end=" ", flush=True)
                time.sleep(1)
            print("开始！\n")

            path = record_activity(stream, label, duration, session_id, rep)
            if path:
                collected_files.append(path)

            if rep < reps and not _interrupted:
                print("\n  休息 10 秒...")
                for i in range(10, 0, -1):
                    if _interrupted:
                        break
                    print(f"\r  休息中... {i}s ", end="", flush=True)
                    time.sleep(1)
                print()

        if act_idx < len(activities) - 1 and not _interrupted:
            print("\n  活动间休息 30 秒...")
            for i in range(30, 0, -1):
                if _interrupted:
                    break
                print(f"\r  活动间休息... {i}s ", end="", flush=True)
                time.sleep(1)
            print()

    stream.stop()

    print(f"\n{'=' * 60}")
    print(f"  会话 #{session_id} 完成！")
    print(f"{'=' * 60}")
    print(f"\n  采集文件数: {len(collected_files)}")
    for f in collected_files:
        size_kb = f.stat().st_size / 1024
        print(f"    {f}  ({size_kb:.1f} KB)")

    print(f"\n  下一步:")
    print(f"    python scripts/train_pipeline.py --source self")
    print()


def main():
    parser = argparse.ArgumentParser(description="FluxChi 引导式 EMG 数据采集")
    parser.add_argument("--port", help="串口名称 (e.g. /dev/tty.usbserial-XXXX)")
    parser.add_argument("--session", type=int, default=1, help="会话编号 (默认 1)")
    parser.add_argument("--reps", type=int, default=3, help="每活动重复次数 (默认 3)")
    parser.add_argument("--duration", type=int, default=None, help="覆盖每次录制时长 (秒)")
    parser.add_argument("--list-ports", action="store_true", help="列出可用串口")
    parser.add_argument(
        "--activities", nargs="+", default=None,
        help="指定要采集的活动 (e.g. typing idle)",
    )
    args = parser.parse_args()

    if args.list_ports:
        ports = list_serial_ports()
        if ports:
            print("可用串口:")
            for p in ports:
                print(f"  {p}")
        else:
            print("未检测到串口设备")
        return

    if not args.port:
        ports = list_serial_ports()
        if not ports:
            print("[ERROR] 未检测到串口。用 --port 手动指定。")
            sys.exit(1)
        usb_ports = [p for p in ports if "usbserial" in p or "usbmodem" in p]
        if len(usb_ports) == 1:
            args.port = usb_ports[0]
            print(f"  自动检测到串口: {args.port}")
        else:
            print("检测到多个串口:")
            for i, p in enumerate(ports):
                print(f"  [{i}] {p}")
            choice = input("选择序号: ").strip()
            try:
                args.port = ports[int(choice)]
            except (ValueError, IndexError):
                print("[ERROR] 无效选择")
                sys.exit(1)

    activities = ACTIVITIES
    if args.activities:
        valid = {a[0] for a in ACTIVITIES}
        activities = [a for a in ACTIVITIES if a[0] in args.activities]
        invalid = set(args.activities) - valid
        if invalid:
            print(f"  [WARN] 未知活动: {invalid}, 可用: {valid}")

    if args.duration:
        activities = [(label, desc, args.duration) for label, desc, _ in activities]

    run_session(args.port, args.session, activities, args.reps)


if __name__ == "__main__":
    main()
