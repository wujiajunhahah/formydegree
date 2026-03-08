#!/bin/bash
# ============================================================
# FluxChi — Mac Studio 环境搭建 + 数据集下载
# ============================================================
# 在 Mac Studio 终端里执行：
#   chmod +x scripts/setup_and_download.sh
#   ./scripts/setup_and_download.sh
#
# 前提：macOS 系统，能联网
# 脚本会自动安装缺少的工具和依赖
# ============================================================

set -e  # 出错即停

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ============================================================
# 1. 检查并安装基础工具
# ============================================================
info "===== 步骤 1/5: 检查基础工具 ====="

# Xcode Command Line Tools (macOS 编译基础)
if ! xcode-select -p &>/dev/null; then
    info "安装 Xcode Command Line Tools..."
    xcode-select --install
    echo "请在弹出的窗口中点击「安装」，安装完成后重新运行此脚本。"
    exit 0
fi
info "✓ Xcode Command Line Tools 已安装"

# Homebrew
if ! command -v brew &>/dev/null; then
    info "安装 Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Apple Silicon Mac 的 brew 路径
    if [ -f /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    fi
fi
info "✓ Homebrew 已安装: $(brew --version | head -1)"

# wget (用于下载数据集)
if ! command -v wget &>/dev/null; then
    info "安装 wget..."
    brew install wget
fi
info "✓ wget 已安装"

# Python 3 (macOS 自带或通过 brew)
if ! command -v python3 &>/dev/null; then
    info "安装 Python 3..."
    brew install python@3.12
fi
PYTHON_VERSION=$(python3 --version)
info "✓ Python 已安装: $PYTHON_VERSION"

# ============================================================
# 2. 设置项目目录和虚拟环境
# ============================================================
info "===== 步骤 2/5: 设置 Python 虚拟环境 ====="

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
info "项目目录: $PROJECT_DIR"

if [ ! -d ".venv" ]; then
    info "创建虚拟环境..."
    python3 -m venv .venv
fi

source .venv/bin/activate
info "✓ 虚拟环境已激活: $(which python3)"

# 升级 pip
python3 -m pip install --upgrade pip --quiet

# ============================================================
# 3. 安装 Python 依赖
# ============================================================
info "===== 步骤 3/5: 安装 Python 依赖 ====="

# 项目已有的依赖
pip install --quiet -r requirements.txt 2>/dev/null || true

# 模型训练额外需要的依赖
pip install --quiet \
    numpy>=1.24 \
    scipy>=1.10 \
    pandas>=2.0 \
    scikit-learn>=1.3 \
    joblib>=1.3 \
    h5py>=3.8 \
    matplotlib>=3.7 \
    seaborn>=0.12 \
    onnx>=1.14 \
    skl2onnx>=1.16 \
    onnxruntime>=1.16 \
    pyserial>=3.5

info "✓ 所有 Python 依赖已安装"

# 验证关键包
python3 -c "
import numpy, scipy, pandas, sklearn, joblib, h5py, matplotlib, onnx, skl2onnx, onnxruntime
print('  numpy:', numpy.__version__)
print('  scipy:', scipy.__version__)
print('  pandas:', pandas.__version__)
print('  scikit-learn:', sklearn.__version__)
print('  h5py:', h5py.__version__)
print('  onnx:', onnx.__version__)
print('  onnxruntime:', onnxruntime.__version__)
" || error "依赖验证失败，请检查安装日志"

info "✓ 依赖验证通过"

# ============================================================
# 4. 下载数据集
# ============================================================
info "===== 步骤 4/5: 下载数据集 ====="

DATASETS_DIR="$PROJECT_DIR/datasets"
mkdir -p "$DATASETS_DIR"

# --------------------------------------------------
# 4.1 Ninapro DB5 — 方法论验证用（核心数据集）
#     10 名受试者，双 Myo 臂环 16ch，200Hz
#     52 种手势 + 休息，每种 6 次重复
#     用途：在公开数据集上验证你的特征提取 + 分类 pipeline
# --------------------------------------------------
NINAPRO_DIR="$DATASETS_DIR/ninapro_db5"
mkdir -p "$NINAPRO_DIR"

NINAPRO_BASE_URL="https://ninapro.hevs.ch/files/DB5_Preproc"

info "下载 Ninapro DB5（10 个受试者）..."
NINAPRO_DONE=true
for i in $(seq 1 10); do
    FILENAME="s${i}.zip"
    FILEPATH="$NINAPRO_DIR/$FILENAME"

    if [ -f "$FILEPATH" ]; then
        info "  s${i}.zip 已存在，跳过"
        continue
    fi

    info "  下载 s${i}.zip ..."
    if wget --quiet --show-progress -O "$FILEPATH" "${NINAPRO_BASE_URL}/${FILENAME}"; then
        info "  ✓ s${i}.zip 下载完成"
    else
        warn "  ✗ s${i}.zip 下载失败，稍后可手动下载: ${NINAPRO_BASE_URL}/${FILENAME}"
        rm -f "$FILEPATH"
        NINAPRO_DONE=false
    fi
done

# 解压
if [ "$NINAPRO_DONE" = true ] || ls "$NINAPRO_DIR"/*.zip &>/dev/null; then
    info "解压 Ninapro DB5 文件..."
    cd "$NINAPRO_DIR"
    for zipfile in *.zip; do
        if [ -f "$zipfile" ]; then
            unzip -o -q "$zipfile" 2>/dev/null || warn "解压 $zipfile 失败"
        fi
    done
    cd "$PROJECT_DIR"
    info "✓ Ninapro DB5 解压完成"

    # 统计文件
    MAT_COUNT=$(find "$NINAPRO_DIR" -name "*.mat" | wc -l | tr -d ' ')
    info "  共 $MAT_COUNT 个 .mat 文件"
fi

# --------------------------------------------------
# 4.2 MDPI sEMG 疲劳数据集（可选，如果可用）
#     论文: https://www.mdpi.com/1424-8220/24/24/8081
#     用途：验证疲劳检测算法
# --------------------------------------------------
info ""
info "注意：sEMG 疲劳数据集需要从论文页面手动获取。"
info "  论文链接: https://www.mdpi.com/1424-8220/24/24/8081"
info "  在论文的 'Supplementary Materials' 或 'Data Availability' 部分查找数据下载链接。"
info "  下载后放到: $DATASETS_DIR/fatigue/"
mkdir -p "$DATASETS_DIR/fatigue"

# --------------------------------------------------
# 4.3 emg2qwerty (Meta, NeurIPS 2024)（可选）
#     108 用户，346 小时，打字 EMG 数据
#     数据集很大（~50GB+），仅在需要时下载
# --------------------------------------------------
info ""
info "注意：emg2qwerty 数据集约 50GB+，仅在需要论文对比实验时下载。"
info "  GitHub: https://github.com/facebookresearch/emg2qwerty"
info "  如需下载，执行:"
info "    cd $DATASETS_DIR"
info "    git clone https://github.com/facebookresearch/emg2qwerty.git"
info "    # 按仓库 README 指引下载数据文件"

# ============================================================
# 5. 验证
# ============================================================
info ""
info "===== 步骤 5/5: 验证安装 ====="

python3 << 'PYEOF'
import sys
from pathlib import Path

print("Python 环境验证:")
print(f"  Python: {sys.version}")
print(f"  路径: {sys.executable}")

project = Path(__file__).parent if "__file__" in dir() else Path(".")

# 检查 Ninapro DB5
ninapro = Path("datasets/ninapro_db5")
mat_files = list(ninapro.glob("**/*.mat")) if ninapro.exists() else []
if mat_files:
    print(f"\n✓ Ninapro DB5: {len(mat_files)} 个 .mat 文件")

    # 尝试读取一个文件验证格式
    try:
        from scipy.io import loadmat
        data = loadmat(str(mat_files[0]))
        keys = [k for k in data.keys() if not k.startswith("_")]
        emg_shape = data.get("emg", [[]]).__class__.__name__
        print(f"  示例文件: {mat_files[0].name}")
        print(f"  数据字段: {', '.join(keys[:10])}")
        if "emg" in data:
            import numpy as np
            emg = np.array(data["emg"])
            print(f"  EMG 数据形状: {emg.shape} (样本数 × 通道数)")
            print(f"  EMG 值范围: [{emg.min():.4f}, {emg.max():.4f}]")
        if "restimulus" in data:
            labels = np.array(data["restimulus"]).flatten()
            unique_labels = np.unique(labels)
            print(f"  动作类别数: {len(unique_labels)} (包含 rest=0)")
    except Exception as e:
        print(f"  ⚠ 读取验证失败: {e}")
else:
    print("\n⚠ Ninapro DB5 未找到 .mat 文件")

# 检查 ONNX 导出能力
try:
    from skl2onnx import convert_sklearn
    print("\n✓ ONNX 导出工具可用")
except ImportError:
    print("\n⚠ skl2onnx 未安装，无法导出 ONNX 格式")

# 检查 src 模块
src = Path("src")
if src.exists():
    modules = [f.stem for f in src.glob("*.py") if f.stem != "__init__"]
    print(f"\n✓ 项目 src/ 模块: {', '.join(sorted(modules))}")

print("\n===== 验证完成 =====")
PYEOF

info ""
info "============================================"
info "  环境搭建完成！"
info "============================================"
info ""
info "数据集位置:"
info "  Ninapro DB5: $DATASETS_DIR/ninapro_db5/"
info ""
info "下一步:"
info "  1. 确认 Ninapro DB5 下载成功（上面应该显示 .mat 文件数量）"
info "  2. 等我写好训练管线代码，你 git pull 到 Mac Studio"
info "  3. 然后用 WAVELETECH 手环采集你自己的数据"
info ""
info "激活环境的命令（以后每次使用前执行）:"
info "  cd $PROJECT_DIR"
info "  source .venv/bin/activate"
info ""
