# 从数据集到模型：完整实操指南

> 目标：用公开数据集训练模型，部署到你的 WAVELETECH EMG 手环上，判断数字游民的工作状态和疲劳程度。
> 硬件：Mac Studio M1 Ultra 128GB / MacBook M1 Pro

---

## 0. 整体流程概览

```
下载数据集 → 数据预处理(对齐格式) → 特征提取 → 训练模型 → 导出模型 → 接入实时推理
```

每一步都是独立的 Python 脚本，你可以一步一步跑。

---

## 1. 环境准备

在你的 Mac Studio 上执行：

```bash
# 进入项目目录
cd /Users/wujiajun/Downloads/FluxChi/reference/harward-gesture

# 激活虚拟环境
source .venv/bin/activate

# 安装额外依赖（你已有 numpy, scikit-learn, joblib）
pip install torch torchvision torchaudio  # PyTorch (Apple Silicon 原生支持 MPS 加速)
pip install pandas                        # 数据处理
pip install h5py scipy                    # 读取 .mat/.h5 格式数据集
pip install mediapipe                     # 面部检测（模态2用）
pip install opencv-python                 # 摄像头采集（模态2用）
pip install onnxruntime                   # 模型导出和轻量推理
```

验证 PyTorch 是否能用 Apple Silicon GPU：

```python
import torch
print(torch.backends.mps.is_available())  # 应该输出 True
```

---

## 2. 数据集下载

### 2.1 模态1：手腕 EMG（优先下载这两个）

#### (A) emg2qwerty — 打字检测（最重要）

```bash
# 创建数据目录
mkdir -p datasets/emg2qwerty

# 从 Meta 官方 GitHub 获取
# 仓库地址: https://github.com/facebookresearch/emg2qwerty
# 数据集需要从论文页面下载，约 50GB
# 步骤：
# 1. 访问 https://github.com/facebookresearch/emg2qwerty
# 2. 按 README 指引下载数据（通常是 AWS S3 链接）
# 3. 解压到 datasets/emg2qwerty/
```

#### (B) Ninapro DB5 — 手势分类

```bash
mkdir -p datasets/ninapro_db5

# 从 Zenodo 下载
# 1. 访问 https://zenodo.org/records/1000116
# 2. 下载所有 .mat 文件
# 3. 放到 datasets/ninapro_db5/
```

#### (C) sEMG 肌肉疲劳数据集 — 疲劳检测

```bash
mkdir -p datasets/fatigue

# 从 MDPI 论文的补充材料下载
# 论文: https://www.mdpi.com/1424-8220/24/24/8081
# 数据通常在论文的 "Supplementary Materials" 或 "Data Availability" 部分有链接
```

### 2.2 模态2：面部表情

#### (D) FER2013 — 面部表情基础数据集

```bash
mkdir -p datasets/fer2013

# 从 Kaggle 下载（需要 Kaggle 账号）
# 1. pip install kaggle
# 2. 配置 ~/.kaggle/kaggle.json
# 3. kaggle datasets download -d msambare/fer2013 -p datasets/fer2013
# 或者直接浏览器下载: https://www.kaggle.com/datasets/msambare/fer2013
```

---

## 3. 数据预处理（最关键的一步）

公开数据集的格式和你的传感器格式不一样，必须做对齐。

### 3.1 理解你的硬件数据格式

你的 WAVELETECH 传感器输出（参考 src/stream.py）：

```
EMG: 8 通道, 24-bit ADC, 转换为微伏 (uV), ~1000 Hz
IMU: 3轴加速度 + 3轴陀螺仪
录制保存格式 (recorder.py): CSV, 列为 [t, ch1, ch2, ..., ch8]
```

### 3.2 Ninapro DB5 数据预处理示例

Ninapro DB5 是 .mat 格式（MATLAB），8通道 Myo 臂环，200Hz。
需要做的事：重采样到你的采样率，提取通道，保存为你的 CSV 格式。

```python
"""
scripts/preprocess_ninapro.py
把 Ninapro DB5 的 .mat 文件转换成项目能用的 CSV 格式
"""
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample

# ---------- 配置 ----------
INPUT_DIR = Path("datasets/ninapro_db5")
OUTPUT_DIR = Path("data")  # 项目的 data/ 目录，Trainer 会从这里读
TARGET_FS = 1000           # 你的传感器采样率
SOURCE_FS = 200            # Ninapro DB5 采样率
CHANNELS = 8               # 只取前8通道（第一个 Myo 臂环）

# ---------- 标签映射 ----------
# Ninapro 用数字标签，你需要映射成有意义的名称
# 0 = 休息, 1-12 = 各种手指动作, 13-17 = 手腕动作, ...
LABEL_MAP = {
    0: "rest",           # 休息/无动作
    1: "finger_flex",    # 手指弯曲（类似打字）
    13: "wrist_flex",    # 手腕弯曲
    14: "wrist_extend",  # 手腕伸展
    # 根据需要添加更多映射
}

def process_subject(mat_path: Path):
    """处理一个受试者的 .mat 文件"""
    print(f"处理: {mat_path.name}")

    # 尝试用 h5py 读取（Ninapro v7 格式）
    try:
        with h5py.File(mat_path, 'r') as f:
            emg = np.array(f['emg'])        # (samples, channels)
            label = np.array(f['restimulus']).flatten()  # (samples,)
    except Exception:
        # 旧版 .mat 用 scipy
        from scipy.io import loadmat
        data = loadmat(str(mat_path))
        emg = data['emg']
        label = data['restimulus'].flatten()

    # 只取前 CHANNELS 个通道
    emg = emg[:, :CHANNELS]

    # 按标签分段
    segments = []
    current_label = label[0]
    start = 0

    for i in range(1, len(label)):
        if label[i] != current_label:
            segments.append((current_label, start, i))
            current_label = label[i]
            start = i
    segments.append((current_label, start, len(label)))

    # 保存每个有效段
    subject_name = mat_path.stem
    for seg_label, seg_start, seg_end in segments:
        seg_label_int = int(seg_label)
        if seg_label_int not in LABEL_MAP:
            continue

        label_name = LABEL_MAP[seg_label_int]
        segment_data = emg[seg_start:seg_end]

        if len(segment_data) < SOURCE_FS:  # 太短的段跳过
            continue

        # 重采样: 200Hz -> 1000Hz
        n_target = int(len(segment_data) * TARGET_FS / SOURCE_FS)
        resampled = resample(segment_data, n_target, axis=0)

        # 生成时间戳
        times = np.arange(n_target) / TARGET_FS

        # 保存为 CSV（和 recorder.py 格式一致）
        output_dir = OUTPUT_DIR / label_name
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{subject_name}_seg{seg_start}.csv"
        columns = ["t"] + [f"ch{i+1}" for i in range(CHANNELS)]
        df = pd.DataFrame(
            np.column_stack([times, resampled]),
            columns=columns
        )
        df.to_csv(output_dir / filename, index=False)
        print(f"  保存: {label_name}/{filename} ({n_target} 样本)")


def main():
    mat_files = sorted(INPUT_DIR.glob("*.mat"))
    if not mat_files:
        print(f"在 {INPUT_DIR} 下没找到 .mat 文件")
        return
    print(f"找到 {len(mat_files)} 个文件")
    for path in mat_files:
        process_subject(path)
    print("预处理完成！")


if __name__ == "__main__":
    main()
```

### 3.3 emg2qwerty 数据预处理

emg2qwerty 的核心价值是区分"在打字"和"没打字"。

```python
"""
scripts/preprocess_emg2qwerty.py
把 emg2qwerty 数据转换为二分类: typing / not_typing
"""
import numpy as np
import pandas as pd
from pathlib import Path

INPUT_DIR = Path("datasets/emg2qwerty")
OUTPUT_DIR = Path("data")
TARGET_FS = 1000
WINDOW_SEC = 2.0  # 每个样本取2秒

def extract_segments(session_dir: Path):
    """
    emg2qwerty 的具体格式取决于你下载的版本。
    通常包含:
    - emg 信号文件 (HDF5 或 NPY)
    - 按键时间戳标注文件

    核心逻辑:
    - 有按键的时间段 -> "typing"
    - 没有按键的时间段 -> "not_typing"
    """
    # 这里需要根据实际下载的数据格式调整
    # 伪代码示意流程：
    #
    # emg_data = load_emg(session_dir / "emg.hdf5")
    # keypress_times = load_labels(session_dir / "keypresses.csv")
    #
    # 把有按键活动的窗口标记为 typing
    # 把无按键活动的窗口标记为 not_typing
    #
    # 保存到 data/typing/*.csv 和 data/not_typing/*.csv
    pass


if __name__ == "__main__":
    # 需要根据实际数据结构完善
    print("请先查看 emg2qwerty 的实际数据格式，再完善此脚本")
    print("参考: https://github.com/facebookresearch/emg2qwerty")
```

---

## 4. 特征提取

你的项目里 `src/features.py` 已经有完整的特征提取器。
预处理后的数据放到 `data/` 目录下就行，`Trainer` 会自动读取。

目录结构应该像这样：

```
data/
  rest/
    subject1_seg0.csv
    subject1_seg5000.csv
    ...
  typing/
    subject1_seg1200.csv
    ...
  wrist_flex/
    ...
```

每个子目录名就是标签名，里面的 CSV 就是录制数据。

你的 `src/features.py` 会对每个窗口提取这些特征：
- MAV (平均绝对值) - 肌肉活动强度
- RMS (均方根) - 信号能量
- WL (波形长度) - 信号复杂度
- ZC (过零率) - 频率相关
- SSC (斜率变化) - 频率相关
- 通道间相关性 - 空间模式

这些特征对于区分"打字/休息/手势"已经够用了。

---

## 5. 训练模型

### 5.1 用现有代码训练（最快上手）

数据放好后，直接用你项目里已有的 `Trainer`：

```python
"""
scripts/train_basic.py
用现有的 RandomForest pipeline 快速训练
"""
from src.trainer import Trainer

trainer = Trainer(
    data_dir="data",          # 预处理后的数据目录
    model_dir="model",        # 模型输出目录
    sample_rate=1000,         # 采样率
    window_seconds=0.5,       # 窗口大小
    stride_seconds=0.1,       # 步长
)

# 看看有哪些数据
recordings = trainer.available_recordings()
print("可用数据:")
for label, files in recordings.items():
    print(f"  {label}: {len(files)} 个录制文件")

# 训练！
metrics = trainer.train()
# 训练完成后模型保存在 model/model.pkl
# 配置保存在 model/config.json
```

运行：
```bash
cd /Users/wujiajun/Downloads/FluxChi/reference/harward-gesture
source .venv/bin/activate
python scripts/train_basic.py
```

### 5.2 进阶：用 PyTorch 训练深度学习模型

RandomForest 是 baseline，如果你想要更好的效果（特别是毕设需要对比实验），
可以训练一个 1D-CNN 或 LSTM 模型。Mac Studio M1 Ultra 跑这个绰绰有余。

```python
"""
scripts/train_deep.py
用 PyTorch 训练 1D-CNN 模型（在 Mac Studio 上用 MPS 加速）
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.features import FeatureExtractor

# ============================================================
# 第一步: 加载数据
# ============================================================

class EMGDataset(Dataset):
    """把 data/ 下的 CSV 文件加载为 PyTorch Dataset"""

    def __init__(self, data_dir="data", window_samples=500, stride_samples=100):
        self.windows = []  # (window_samples, 8) 的 numpy 数组
        self.labels = []   # 字符串标签

        data_path = Path(data_dir)
        for label_dir in sorted(data_path.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for csv_file in sorted(label_dir.glob("*.csv")):
                self._load_file(csv_file, label, window_samples, stride_samples)

        # 编码标签
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.num_classes = len(self.label_encoder.classes_)

        print(f"加载完成: {len(self.windows)} 个窗口, {self.num_classes} 个类别")
        for cls in self.label_encoder.classes_:
            count = sum(1 for l in self.labels if l == cls)
            print(f"  {cls}: {count} 个窗口")

    def _load_file(self, path, label, window_samples, stride_samples):
        import pandas as pd
        df = pd.read_csv(path)
        # 取 ch1-ch8 列
        ch_cols = [c for c in df.columns if c.startswith("ch")]
        data = df[ch_cols].values.astype(np.float32)

        # 滑动窗口切分
        start = 0
        while start + window_samples <= len(data):
            window = data[start:start + window_samples]
            self.windows.append(window)
            self.labels.append(label)
            start += stride_samples

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # 返回 (channels, time) 的格式给 1D-CNN
        window = torch.tensor(self.windows[idx], dtype=torch.float32).T  # (8, 500)
        label = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return window, label


# ============================================================
# 第二步: 定义模型
# ============================================================

class EMG1DCNN(nn.Module):
    """
    1D 卷积网络，输入 (batch, 8, window_samples)
    简单但有效，适合 EMG 分类
    """
    def __init__(self, in_channels=8, num_classes=3, window_samples=500):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化 -> (batch, 128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 第三步: 训练
# ============================================================

def train():
    # --- 配置 ---
    DATA_DIR = "data"
    MODEL_DIR = "model"
    WINDOW_SAMPLES = 500    # 0.5秒 x 1000Hz
    STRIDE_SAMPLES = 100    # 0.1秒步长
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 0.001

    # --- 选择设备 ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Mac Studio GPU 加速
        print("使用 Apple Silicon MPS 加速")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    # --- 加载数据 ---
    dataset = EMGDataset(DATA_DIR, WINDOW_SAMPLES, STRIDE_SAMPLES)

    if dataset.num_classes < 2:
        print("至少需要 2 个类别的数据！")
        return

    # 划分训练集/测试集
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=dataset.encoded_labels
    )

    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx)
    )
    test_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx)
    )

    # --- 创建模型 ---
    model = EMG1DCNN(
        in_channels=8,
        num_classes=dataset.num_classes,
        window_samples=WINDOW_SAMPLES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练集: {len(train_idx)} | 测试集: {len(test_idx)}")
    print(f"开始训练 {EPOCHS} 轮...\n")

    # --- 训练循环 ---
    best_acc = 0.0
    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        # 测试
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        train_acc = train_correct / train_total
        test_acc = test_correct / test_total if test_total > 0 else 0
        avg_loss = train_loss / train_total

        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'label_encoder_classes': dataset.label_encoder.classes_.tolist(),
                'num_classes': dataset.num_classes,
                'window_samples': WINDOW_SAMPLES,
                'in_channels': 8,
            }, Path(MODEL_DIR) / "model_cnn.pt")
            print(f"  -> 最佳模型已保存 (acc={best_acc:.4f})")

    print(f"\n训练完成！最佳测试准确率: {best_acc:.4f}")
    print(f"模型保存在: {MODEL_DIR}/model_cnn.pt")

    # 保存配置（兼容现有代码）
    config = {
        "model_type": "cnn",
        "sample_rate": 1000,
        "window_seconds": WINDOW_SAMPLES / 1000,
        "stride_seconds": STRIDE_SAMPLES / 1000,
        "classes": dataset.label_encoder.classes_.tolist(),
    }
    with open(Path(MODEL_DIR) / "config_cnn.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    train()
```

---

## 6. 模型使用/部署

训练完成后，模型文件在 `model/` 目录下：

```
model/
  model.pkl          # RandomForest 模型（方案A）
  config.json
  model_cnn.pt       # PyTorch CNN 模型（方案B）
  config_cnn.json
```

### 6.1 RandomForest 模型（已集成）

你的项目已经支持了。直接运行 app.py，按 I 开启推理即可。

### 6.2 CNN 模型推理（需要写个适配器）

```python
"""
加载训练好的 CNN 模型进行推理
可以替换 src/inference.py 中的 model
"""
import torch
import numpy as np
from pathlib import Path


class CNNPredictor:
    """兼容现有 GestureInference 接口的 CNN 推理器"""

    def __init__(self, model_dir="model"):
        checkpoint = torch.load(
            Path(model_dir) / "model_cnn.pt",
            map_location="cpu"
        )
        from scripts.train_deep import EMG1DCNN

        self.classes_ = np.array(checkpoint['label_encoder_classes'])
        self.model = EMG1DCNN(
            in_channels=checkpoint['in_channels'],
            num_classes=checkpoint['num_classes'],
            window_samples=checkpoint['window_samples']
        )
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def predict_proba(self, features_list):
        """兼容 sklearn 接口，输入是原始窗口数据"""
        # 注意: CNN 直接用原始信号，不需要手工特征
        # 这里需要适配，详见下方说明
        pass
```

---

## 7. 模态2：面部表情检测（网页摄像头）

这部分独立于 EMG，通过网页调用摄像头。

### 7.1 训练面部模型

```python
"""
scripts/train_face.py
用 FER2013 训练面部表情分类模型
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# FER2013 下载解压后的目录结构:
# datasets/fer2013/
#   train/
#     angry/
#     happy/
#     neutral/
#     sad/
#     surprise/
#     ...
#   test/
#     ...

def train_face_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_set = ImageFolder("datasets/fer2013/train", transform=transform)
    test_set = ImageFolder("datasets/fer2013/test", transform=transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128)

    # 简单的 CNN（你也可以用预训练的 ResNet）
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 7),  # 7种表情
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 测试
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.size(0)
        print(f"Epoch {epoch+1}: Test Acc = {correct/total:.4f}")

    torch.save(model.state_dict(), "model/face_model.pt")
    print("面部模型保存完成")

if __name__ == "__main__":
    train_face_model()
```

### 7.2 面部状态映射

对于你的场景，不需要区分 7 种情绪，只需要映射到工作状态：

```
neutral + 无微笑 + 长时间静止 = 可能在专注工作
happy/surprise = 可能在社交/休息
sad/angry + EMG疲劳信号 = 需要休息
```

---

## 8. 实操步骤总结（按顺序做）

### 阶段1：先跑通 baseline（1-2天）

```bash
# 1. 下载 Ninapro DB5（最小，~几百MB）
# 2. 运行预处理脚本
python scripts/preprocess_ninapro.py

# 3. 用现有的 RandomForest 训练
python scripts/train_basic.py

# 4. 连接传感器测试
python app.py --port /dev/cu.usbserial-0001 --baud 921600 --fs 1000
# 按 I 开启推理，看看效果
```

### 阶段2：深度学习模型（3-5天）

```bash
# 1. 下载 emg2qwerty（大数据集，在 Mac Studio 上跑）
# 2. 预处理
# 3. 训练 CNN 模型
python scripts/train_deep.py

# 4. 对比 RandomForest vs CNN 的准确率
```

### 阶段3：加入面部模态（3-5天）

```bash
# 1. 下载 FER2013
# 2. 训练面部模型
python scripts/train_face.py

# 3. 写网页摄像头采集 + 推理代码
# 4. 双模态融合决策
```

---

## 9. 常见问题

### Q: 公开数据集训练的模型能直接用在我的传感器上吗？

**不能直接用。** 不同传感器的通道数、增益、位置都不同。但可以：
1. 用公开数据集训练一个预训练模型
2. 再用你自己传感器采集少量数据做微调（fine-tune）
3. 这就是迁移学习，效果远好于从零开始

### Q: 我应该先下载哪个数据集？

按优先级：
1. **Ninapro DB5** — 最小、格式最规范、与你硬件最匹配
2. **emg2qwerty** — 最大、专门做打字检测
3. **FER2013** — 面部表情、Kaggle 一键下载

### Q: 训练要多久？

在 Mac Studio M1 Ultra 上：
- RandomForest: 几秒到几分钟
- CNN (PyTorch + MPS): 几分钟到半小时
- 面部模型: 10-30 分钟

### Q: 需要多少自采数据做微调？

每个类别 5-10 段录制（每段 3-5 秒），大约 15 分钟就能采集完。
用你的 app.py 按 R 录制即可。
