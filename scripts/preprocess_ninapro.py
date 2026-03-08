"""
把 Ninapro DB5 的 .mat 文件转换成项目能用的 CSV 格式。

Ninapro DB5 使用双 Myo 臂环 (16通道, 200Hz)，我们取前 8 通道并重采样到 1000Hz，
使其与 WAVELETECH 传感器格式一致。

用法:
    1. 从 https://zenodo.org/records/1000116 下载 .mat 文件
    2. 放到 datasets/ninapro_db5/ 目录下
    3. 运行: python scripts/preprocess_ninapro.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample

# ---------- 配置 ----------
INPUT_DIR = Path("datasets/ninapro_db5")
OUTPUT_DIR = Path("data")
TARGET_FS = 1000  # 你的传感器采样率
SOURCE_FS = 200   # Ninapro DB5 采样率
CHANNELS = 8      # 取前 8 通道

# ---------- 标签映射 ----------
# Ninapro DB5 Exercise 1: 基本手指动作 (1-12)
# Exercise 2: 手腕动作和抓握 (13-29)
# Exercise 3: 功能性抓握 (30-41)
# 0 = 休息
#
# 对于"数字游民状态检测"，我们关心的分类：
LABEL_MAP = {
    0: "rest",             # 休息/无动作 -> 判断为"不在工作"
    1: "finger_movement",  # 手指动作 -> 类似打字
    2: "finger_movement",
    3: "finger_movement",
    4: "finger_movement",
    5: "finger_movement",
    13: "wrist_flex",      # 手腕弯曲
    14: "wrist_extend",    # 手腕伸展
    15: "wrist_movement",  # 手腕偏移
    16: "wrist_movement",
}

# 你也可以简化为两类：
# LABEL_MAP = {0: "not_working", 1: "working", 2: "working", ...}


def load_mat_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """加载 .mat 文件，返回 (emg, labels)"""
    # 先尝试 h5py (MATLAB v7.3 格式)
    try:
        import h5py
        with h5py.File(path, "r") as f:
            emg = np.array(f["emg"])
            # Ninapro 用 restimulus 作为标签（重复试验的标签）
            if "restimulus" in f:
                labels = np.array(f["restimulus"]).flatten()
            else:
                labels = np.array(f["stimulus"]).flatten()
        return emg, labels
    except Exception:
        pass

    # 回退到 scipy (MATLAB v5 格式)
    from scipy.io import loadmat
    data = loadmat(str(path))
    emg = data["emg"]
    labels = data.get("restimulus", data.get("stimulus")).flatten()
    return emg, labels


def process_subject(mat_path: Path) -> int:
    """处理一个受试者的数据，返回保存的段数"""
    print(f"处理: {mat_path.name}")
    emg, labels = load_mat_file(mat_path)

    # 只取前 CHANNELS 个通道
    if emg.shape[1] > CHANNELS:
        emg = emg[:, :CHANNELS]
    elif emg.shape[1] < CHANNELS:
        # 通道不够，补零
        pad = np.zeros((emg.shape[0], CHANNELS - emg.shape[1]))
        emg = np.hstack([emg, pad])

    # 按标签分段
    segments = _segment_by_label(labels)
    saved = 0
    subject_name = mat_path.stem

    for seg_label, seg_start, seg_end in segments:
        seg_label_int = int(seg_label)
        if seg_label_int not in LABEL_MAP:
            continue

        label_name = LABEL_MAP[seg_label_int]
        segment_data = emg[seg_start:seg_end]

        # 太短的段跳过（至少 0.5 秒）
        if len(segment_data) < SOURCE_FS // 2:
            continue

        # 重采样: SOURCE_FS -> TARGET_FS
        n_target = int(len(segment_data) * TARGET_FS / SOURCE_FS)
        resampled = resample(segment_data, n_target, axis=0)

        # 生成时间戳
        times = np.arange(n_target) / TARGET_FS

        # 保存为 CSV
        _save_csv(label_name, subject_name, seg_start, times, resampled)
        saved += 1

    print(f"  -> 保存了 {saved} 个有效段")
    return saved


def _segment_by_label(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """把连续相同标签的区域提取为 (label, start, end) 列表"""
    segments = []
    current = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((current, start, i))
            current = labels[i]
            start = i
    segments.append((current, start, len(labels)))
    return segments


def _save_csv(label: str, subject: str, seg_start: int,
              times: np.ndarray, data: np.ndarray) -> None:
    """保存为 recorder.py 兼容的 CSV 格式"""
    output_dir = OUTPUT_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{subject}_seg{seg_start}.csv"
    columns = ["t"] + [f"ch{i+1}" for i in range(CHANNELS)]
    df = pd.DataFrame(np.column_stack([times, data]), columns=columns)
    df.to_csv(output_dir / filename, index=False)


def main():
    if not INPUT_DIR.exists():
        print(f"数据目录不存在: {INPUT_DIR}")
        print(f"请先下载 Ninapro DB5:")
        print(f"  https://zenodo.org/records/1000116")
        print(f"并将 .mat 文件放到 {INPUT_DIR}/ 目录下")
        sys.exit(1)

    mat_files = sorted(INPUT_DIR.glob("*.mat"))
    if not mat_files:
        print(f"在 {INPUT_DIR} 下没找到 .mat 文件")
        sys.exit(1)

    print(f"找到 {len(mat_files)} 个文件")
    print(f"标签映射: {LABEL_MAP}")
    print(f"重采样: {SOURCE_FS}Hz -> {TARGET_FS}Hz")
    print(f"输出目录: {OUTPUT_DIR}\n")

    total = 0
    for path in mat_files:
        total += process_subject(path)

    print(f"\n预处理完成！共保存 {total} 个数据段")
    print(f"数据在 {OUTPUT_DIR}/ 目录下，可以用 Trainer 训练了")


if __name__ == "__main__":
    main()
