"""
用现有的 RandomForest pipeline 快速训练。
这是最快上手的方案，先跑通再考虑深度学习。

用法:
    python scripts/train_basic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保能 import src 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trainer import Trainer


def main():
    trainer = Trainer(
        data_dir="data",
        model_dir="model",
        sample_rate=1000,
        window_seconds=0.5,
        stride_seconds=0.1,
    )

    # 查看可用数据
    recordings = trainer.available_recordings()
    if not recordings:
        print("data/ 目录下没有数据！")
        print("请先运行预处理脚本（如 scripts/preprocess_ninapro.py）")
        print("或者连接传感器用 app.py 录制数据（按 R）")
        return

    print("可用数据:")
    total_files = 0
    for label, files in sorted(recordings.items()):
        print(f"  {label}: {len(files)} 个录制文件")
        total_files += len(files)
    print(f"  共计: {total_files} 个文件\n")

    if len(recordings) < 2:
        print("至少需要 2 个类别的数据才能训练！")
        return

    # 训练
    print("开始训练 RandomForest 模型...")
    metrics = trainer.train()

    if metrics:
        print("\n训练完成！")
        print(f"模型保存在: model/model.pkl")
        print(f"配置保存在: model/config.json")
        print("\n下一步: 连接传感器运行 app.py，按 I 开启实时推理")
    else:
        print("\n训练失败，请检查数据")


if __name__ == "__main__":
    main()
