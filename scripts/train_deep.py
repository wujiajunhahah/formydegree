"""
用 PyTorch 训练 1D-CNN 模型（Apple Silicon MPS 加速）。

相比 RandomForest:
- 直接用原始 EMG 信号，不需要手工特征
- 自动学习时间+空间模式
- 泛化能力更好

用法:
    python scripts/train_deep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# ---------- 配置 ----------
DATA_DIR = "data"
MODEL_DIR = "model"
SAMPLE_RATE = 1000
WINDOW_SEC = 0.5
STRIDE_SEC = 0.1
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)   # 500
STRIDE_SAMPLES = int(STRIDE_SEC * SAMPLE_RATE)    # 100
CHANNELS = 8
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001


# ============================================================
# 数据集
# ============================================================

class EMGWindowDataset(Dataset):
    """从 data/ 目录加载 CSV，切成固定长度的窗口"""

    def __init__(self, data_dir: str, window_samples: int, stride_samples: int):
        self.windows: list[np.ndarray] = []
        self.labels: list[str] = []

        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_path}")

        for label_dir in sorted(data_path.iterdir()):
            if not label_dir.is_dir() or label_dir.name.startswith("."):
                continue
            label = label_dir.name
            csv_files = sorted(label_dir.glob("*.csv"))
            for csv_file in csv_files:
                self._load_and_window(csv_file, label, window_samples, stride_samples)

        if not self.windows:
            raise ValueError("没有加载到任何数据窗口")

        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.num_classes = len(self.label_encoder.classes_)

        print(f"数据集加载完成:")
        print(f"  总窗口数: {len(self.windows)}")
        print(f"  类别数: {self.num_classes}")
        for cls in self.label_encoder.classes_:
            count = sum(1 for label in self.labels if label == cls)
            print(f"    {cls}: {count}")

    def _load_and_window(self, path: Path, label: str,
                         window_samples: int, stride_samples: int) -> None:
        df = pd.read_csv(path)
        ch_cols = [c for c in df.columns if c.startswith("ch")]
        if not ch_cols:
            return
        data = df[ch_cols].values.astype(np.float32)

        # 滑动窗口
        start = 0
        while start + window_samples <= len(data):
            self.windows.append(data[start:start + window_samples])
            self.labels.append(label)
            start += stride_samples

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        # Conv1d 需要 (channels, time) 格式
        window = torch.tensor(self.windows[idx], dtype=torch.float32).T
        label = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return window, label


# ============================================================
# 模型
# ============================================================

class EMG1DCNN(nn.Module):
    """
    轻量级 1D-CNN，适合手腕 EMG 分类。
    输入: (batch, 8, window_samples)
    输出: (batch, num_classes)
    """

    def __init__(self, in_channels: int = 8, num_classes: int = 3,
                 window_samples: int = 500):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ============================================================
# 训练
# ============================================================

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("使用 Apple Silicon MPS 加速")
        return torch.device("mps")
    print("使用 CPU（如果你在 Mac 上，确认 PyTorch 版本 >= 2.0）")
    return torch.device("cpu")


def train() -> None:
    device = get_device()

    # 加载数据
    dataset = EMGWindowDataset(DATA_DIR, WINDOW_SAMPLES, STRIDE_SAMPLES)

    if dataset.num_classes < 2:
        print("至少需要 2 个类别的数据！")
        return

    # 划分训练/测试
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=dataset.encoded_labels,
    )

    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )
    test_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    )

    # 创建模型
    model = EMG1DCNN(
        in_channels=CHANNELS,
        num_classes=dataset.num_classes,
        window_samples=WINDOW_SAMPLES,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {param_count:,}")
    print(f"训练集: {len(train_idx)} | 测试集: {len(test_idx)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    # 训练循环
    best_acc = 0.0
    print(f"\n开始训练 {EPOCHS} 轮...\n")

    for epoch in range(EPOCHS):
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == targets).sum().item()
            train_total += targets.size(0)

        # --- 测试 ---
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                test_correct += (outputs.argmax(1) == targets).sum().item()
                test_total += targets.size(0)

        train_acc = train_correct / max(train_total, 1)
        test_acc = test_correct / max(test_total, 1)
        avg_loss = train_loss / max(train_total, 1)
        scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train: {train_acc:.4f} | "
            f"Test: {test_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            _save_model(model, dataset, WINDOW_SAMPLES)
            print(f"  -> 最佳模型已保存 (test_acc={best_acc:.4f})")

    print(f"\n训练完成！最佳测试准确率: {best_acc:.4f}")
    print(f"模型: {MODEL_DIR}/model_cnn.pt")
    print(f"配置: {MODEL_DIR}/config_cnn.json")


def _save_model(model: nn.Module, dataset: EMGWindowDataset,
                window_samples: int) -> None:
    out = Path(MODEL_DIR)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "label_classes": dataset.label_encoder.classes_.tolist(),
            "num_classes": dataset.num_classes,
            "in_channels": CHANNELS,
            "window_samples": window_samples,
        },
        out / "model_cnn.pt",
    )

    config = {
        "model_type": "cnn",
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SEC,
        "stride_seconds": STRIDE_SEC,
        "window_samples": window_samples,
        "channels": CHANNELS,
        "classes": dataset.label_encoder.classes_.tolist(),
    }
    with (out / "config_cnn.json").open("w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    train()
