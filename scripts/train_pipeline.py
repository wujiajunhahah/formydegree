"""
FluxChi — 完整模型训练管线
=========================================================
支持两种数据源：
  1. Ninapro DB5 (.mat) — 公开数据集方法论验证
  2. 自采数据 (data/*.csv) — 部署到 WAVELETECH 手环

用法:
  # 在 Ninapro DB5 上验证方法论
  python scripts/train_pipeline.py --source ninapro

  # 用自采数据训练部署模型
  python scripts/train_pipeline.py --source self

  # 两者都跑（论文对比用）
  python scripts/train_pipeline.py --source both
=========================================================
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_RATE_SELF = 1000
SAMPLE_RATE_NINAPRO = 200
CHANNELS = 8
WINDOW_SEC = 0.25
OVERLAP = 0.5

NINAPRO_LABEL_MAP = {
    0: "rest",
    1: "finger_movement", 2: "finger_movement", 3: "finger_movement",
    4: "finger_movement", 5: "finger_movement",
    13: "wrist_flex", 14: "wrist_extend",
    15: "wrist_movement", 16: "wrist_movement",
}

RESULTS_DIR = Path("results")


# ============================================================
# Feature extraction (enhanced: time + frequency domain)
# ============================================================

def _zero_crossings(values: np.ndarray, threshold: float = 20.0) -> int:
    diffs = np.diff(np.sign(values))
    abs_diff = np.abs(np.diff(values))
    return int(np.sum((diffs != 0) & (abs_diff > threshold)))


def _slope_sign_changes(values: np.ndarray, threshold: float = 20.0) -> int:
    if len(values) < 3:
        return 0
    d1 = values[1:-1] - values[:-2]
    d2 = values[2:] - values[1:-1]
    return int(np.sum((d1 * d2 < 0) & (np.abs(d1) > threshold) & (np.abs(d2) > threshold)))


def _median_frequency(values: np.ndarray, sample_rate: int) -> float:
    """Median frequency (MDF) — the frequency dividing the PSD into equal halves."""
    if len(values) < 4:
        return 0.0
    freqs = np.fft.rfftfreq(len(values), d=1.0 / sample_rate)
    psd = np.abs(np.fft.rfft(values)) ** 2
    cumulative = np.cumsum(psd)
    total = cumulative[-1]
    if total < 1e-12:
        return 0.0
    idx = np.searchsorted(cumulative, total / 2.0)
    return float(freqs[min(idx, len(freqs) - 1)])


def _mean_frequency(values: np.ndarray, sample_rate: int) -> float:
    """Mean frequency (MNF)."""
    if len(values) < 4:
        return 0.0
    freqs = np.fft.rfftfreq(len(values), d=1.0 / sample_rate)
    psd = np.abs(np.fft.rfft(values)) ** 2
    total_power = np.sum(psd)
    if total_power < 1e-12:
        return 0.0
    return float(np.sum(freqs * psd) / total_power)


def extract_window_features(
    window: np.ndarray,
    sample_rate: int,
    n_channels: int = CHANNELS,
    threshold: float = 20.0,
) -> np.ndarray:
    """Extract 7 features per channel + inter-channel correlations from one window.

    Per-channel (7 each):
        MAV, RMS, WL, ZC, SSC, MNF, MDF

    Spatial:
        Pairwise Pearson correlations between channels
    """
    feats = []
    for ch in range(n_channels):
        sig = window[:, ch]
        mav = np.mean(np.abs(sig))
        rms = np.sqrt(np.mean(sig ** 2))
        wl = np.sum(np.abs(np.diff(sig))) if len(sig) > 1 else 0.0
        zc = _zero_crossings(sig, threshold)
        ssc = _slope_sign_changes(sig, threshold)
        mnf = _mean_frequency(sig, sample_rate)
        mdf = _median_frequency(sig, sample_rate)
        feats.extend([mav, rms, wl, zc, ssc, mnf, mdf])

    # inter-channel correlations
    if window.shape[0] >= 2:
        try:
            corr = np.corrcoef(window, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
        except Exception:
            corr = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                feats.append(corr[i, j])
    else:
        n_pairs = n_channels * (n_channels - 1) // 2
        feats.extend([0.0] * n_pairs)

    return np.array(feats, dtype=np.float64)


def feature_names(n_channels: int = CHANNELS) -> List[str]:
    names = []
    metrics = ["MAV", "RMS", "WL", "ZC", "SSC", "MNF", "MDF"]
    for ch in range(n_channels):
        for m in metrics:
            names.append(f"ch{ch+1}_{m}")
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            names.append(f"corr_{i+1}_{j+1}")
    return names


def sliding_window_features(
    data: np.ndarray,
    sample_rate: int,
    window_sec: float = WINDOW_SEC,
    overlap: float = OVERLAP,
    n_channels: int = CHANNELS,
) -> np.ndarray:
    """Apply sliding window and extract features from continuous EMG data."""
    window_samples = int(window_sec * sample_rate)
    stride = int(window_samples * (1 - overlap))
    if stride < 1:
        stride = 1

    n_samples = data.shape[0]
    features_list = []
    start = 0
    while start + window_samples <= n_samples:
        window = data[start:start + window_samples, :n_channels]
        features_list.append(extract_window_features(window, sample_rate, n_channels))
        start += stride

    if not features_list:
        return np.zeros((0, len(feature_names(n_channels))))
    return np.array(features_list)


# ============================================================
# Data loading — Ninapro DB5
# ============================================================

def load_ninapro_db5(
    data_dir: str = "datasets/ninapro_db5",
    label_map: Optional[Dict[int, str]] = None,
    max_subjects: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Ninapro DB5 .mat files, return (X_features, y_labels, groups).

    groups = subject IDs for Leave-One-Subject-Out CV.
    """
    if label_map is None:
        label_map = NINAPRO_LABEL_MAP

    data_path = Path(data_dir)
    mat_files = sorted(data_path.glob("**/*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {data_path}. Run setup_and_download.sh first.")

    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy required: pip install scipy")

    all_X, all_y, all_groups = [], [], []
    subjects_processed = 0

    for mat_path in mat_files:
        if subjects_processed >= max_subjects:
            break

        try:
            data = loadmat(str(mat_path), squeeze_me=True)
        except Exception as e:
            print(f"  [WARN] Cannot read {mat_path.name}: {e}")
            continue

        emg = data.get("emg")
        labels = data.get("restimulus", data.get("stimulus"))
        if emg is None or labels is None:
            continue

        emg = np.array(emg, dtype=np.float64)
        labels = np.array(labels, dtype=int).flatten()

        if emg.shape[1] > CHANNELS:
            emg = emg[:, :CHANNELS]
        elif emg.shape[1] < CHANNELS:
            pad = np.zeros((emg.shape[0], CHANNELS - emg.shape[1]))
            emg = np.hstack([emg, pad])

        subject_id = mat_path.stem
        subjects_processed += 1
        print(f"  Loading {mat_path.name}: {emg.shape[0]} samples, {len(np.unique(labels))} classes")

        segments = _segment_by_label(labels)
        for seg_label, seg_start, seg_end in segments:
            if seg_label not in label_map:
                continue
            segment_data = emg[seg_start:seg_end]
            min_samples = int(WINDOW_SEC * SAMPLE_RATE_NINAPRO)
            if len(segment_data) < min_samples:
                continue

            X_seg = sliding_window_features(
                segment_data, SAMPLE_RATE_NINAPRO, WINDOW_SEC, OVERLAP, CHANNELS
            )
            if X_seg.shape[0] == 0:
                continue

            label_name = label_map[seg_label]
            all_X.append(X_seg)
            all_y.extend([label_name] * X_seg.shape[0])
            all_groups.extend([subject_id] * X_seg.shape[0])

    if not all_X:
        raise ValueError("No valid data segments found in Ninapro DB5")

    X = np.vstack(all_X)
    y = np.array(all_y)
    groups = np.array(all_groups)
    print(f"\n  Ninapro DB5 loaded: {X.shape[0]} windows, {len(np.unique(y))} classes, {len(np.unique(groups))} subjects")
    return X, y, groups


def _segment_by_label(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    segments = []
    current = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((int(current), start, i))
            current = labels[i]
            start = i
    segments.append((int(current), start, len(labels)))
    return segments


# ============================================================
# Data loading — self-collected CSV
# ============================================================

def load_self_collected(
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load self-collected CSV data from data/<label>/*.csv.

    groups = file paths for Leave-One-Session-Out CV.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_path} not found. Collect data first.")

    all_X, all_y, all_groups = [], [], []

    for label_dir in sorted(data_path.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("."):
            continue
        label = label_dir.name
        csv_files = sorted(label_dir.glob("*.csv"))
        if not csv_files:
            continue

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            ch_cols = [c for c in df.columns if c.startswith("ch")]
            if not ch_cols:
                continue
            data = df[ch_cols].values.astype(np.float64)
            if data.shape[1] < CHANNELS:
                pad = np.zeros((data.shape[0], CHANNELS - data.shape[1]))
                data = np.hstack([data, pad])
            elif data.shape[1] > CHANNELS:
                data = data[:, :CHANNELS]

            X_file = sliding_window_features(
                data, SAMPLE_RATE_SELF, WINDOW_SEC, OVERLAP, CHANNELS
            )
            if X_file.shape[0] == 0:
                continue

            session_id = csv_path.stem
            all_X.append(X_file)
            all_y.extend([label] * X_file.shape[0])
            all_groups.extend([session_id] * X_file.shape[0])
            print(f"  {label}/{csv_path.name}: {X_file.shape[0]} windows")

    if not all_X:
        raise ValueError(f"No valid CSV data in {data_path}/")

    X = np.vstack(all_X)
    y = np.array(all_y)
    groups = np.array(all_groups)
    print(f"\n  Self-collected loaded: {X.shape[0]} windows, {len(np.unique(y))} classes, {len(np.unique(groups))} sessions")
    return X, y, groups


# ============================================================
# Training + Evaluation
# ============================================================

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    source_name: str = "data",
    n_estimators: int = 200,
) -> Tuple[RandomForestClassifier, Dict]:
    """Train RandomForest with Leave-One-Group-Out CV, return model + metrics."""

    print(f"\n{'='*60}")
    print(f"Training: {source_name}")
    print(f"{'='*60}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {np.unique(y).tolist()}")
    print(f"  Groups: {len(np.unique(groups))}")

    # Clean NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    unique_groups = np.unique(groups)
    classes = np.unique(y).tolist()

    results = {
        "source": source_name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": len(classes),
        "classes": classes,
        "n_groups": len(unique_groups),
    }

    # Leave-One-Group-Out cross-validation
    if len(unique_groups) >= 3:
        print(f"\n  Leave-One-Group-Out CV ({len(unique_groups)} folds)...")
        logo = LeaveOneGroupOut()
        all_preds = np.empty_like(y)
        fold_accs = []

        for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            fold_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            fold_model.fit(X[train_idx], y[train_idx])
            preds = fold_model.predict(X[test_idx])
            all_preds[test_idx] = preds
            fold_acc = accuracy_score(y[test_idx], preds)
            fold_accs.append(fold_acc)
            test_group = groups[test_idx[0]]
            print(f"    Fold {fold_i+1}: test={test_group}, acc={fold_acc:.4f}, "
                  f"n_train={len(train_idx)}, n_test={len(test_idx)}")

        overall_acc = accuracy_score(y, all_preds)
        overall_f1 = f1_score(y, all_preds, average="weighted")
        cm = confusion_matrix(y, all_preds, labels=classes)
        report = classification_report(y, all_preds, output_dict=True)
        report_str = classification_report(y, all_preds)

        print(f"\n  LOGO-CV Results:")
        print(f"    Overall Accuracy: {overall_acc:.4f}")
        print(f"    Weighted F1:      {overall_f1:.4f}")
        print(f"    Per-fold acc:     {[f'{a:.3f}' for a in fold_accs]}")
        print(f"    Mean fold acc:    {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
        print(f"\n{report_str}")

        results["cv_method"] = "Leave-One-Group-Out"
        results["overall_accuracy"] = float(overall_acc)
        results["weighted_f1"] = float(overall_f1)
        results["fold_accuracies"] = [float(a) for a in fold_accs]
        results["mean_fold_accuracy"] = float(np.mean(fold_accs))
        results["std_fold_accuracy"] = float(np.std(fold_accs))
        results["confusion_matrix"] = cm.tolist()
        results["classification_report"] = report
    else:
        print(f"\n  Only {len(unique_groups)} group(s), using 80/20 split...")
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            range(len(y)), test_size=0.2, random_state=42, stratify=y
        )
        fold_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        fold_model.fit(X[train_idx], y[train_idx])
        preds = fold_model.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], preds)
        report_str = classification_report(y[test_idx], preds)
        print(f"    Accuracy: {acc:.4f}")
        print(f"\n{report_str}")
        results["cv_method"] = "holdout_80_20"
        results["overall_accuracy"] = float(acc)

    # Train final model on ALL data
    print(f"\n  Training final model on all {X.shape[0]} samples...")
    final_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    final_model.fit(X, y)

    # Feature importances
    importances = final_model.feature_importances_
    names = feature_names(CHANNELS)
    top_k = min(15, len(names))
    top_idx = np.argsort(importances)[::-1][:top_k]
    print(f"\n  Top {top_k} features:")
    for rank, idx in enumerate(top_idx):
        print(f"    {rank+1}. {names[idx]}: {importances[idx]:.4f}")

    results["top_features"] = [
        {"name": names[int(idx)], "importance": float(importances[int(idx)])}
        for idx in top_idx
    ]

    return final_model, results


# ============================================================
# Model export
# ============================================================

def save_model(
    model: RandomForestClassifier,
    results: Dict,
    output_dir: str = "model",
    prefix: str = "activity_classifier",
) -> None:
    """Save model as .pkl + .onnx + config.json + evaluation report."""
    import joblib

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # .pkl
    pkl_path = out / f"{prefix}.pkl"
    joblib.dump(model, pkl_path)
    print(f"\n  Saved: {pkl_path}")

    # config.json
    config = {
        "model_type": "RandomForestClassifier",
        "n_estimators": model.n_estimators,
        "classes": model.classes_.tolist(),
        "n_features": model.n_features_in_,
        "feature_names": feature_names(CHANNELS),
        "window_seconds": WINDOW_SEC,
        "overlap": OVERLAP,
        "channels": CHANNELS,
        "source": results.get("source", "unknown"),
        "accuracy": results.get("overall_accuracy"),
        "weighted_f1": results.get("weighted_f1"),
        "cv_method": results.get("cv_method"),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = out / f"{prefix}_config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {config_path}")

    # ONNX
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("features", FloatTensorType([None, model.n_features_in_]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        onnx_path = out / f"{prefix}.onnx"
        with onnx_path.open("wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"  Saved: {onnx_path}")
    except Exception as e:
        print(f"  [WARN] ONNX export failed: {e}")

    # Evaluation report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"{prefix}_evaluation.json"
    with report_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {report_path}")


def save_confusion_matrix_plot(
    results: Dict,
    prefix: str = "activity_classifier",
) -> None:
    """Save confusion matrix as PNG."""
    cm = results.get("confusion_matrix")
    classes = results.get("classes")
    if cm is None or classes is None:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 6))
        cm_array = np.array(cm)
        sns.heatmap(
            cm_array, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {results.get('source', '')}\n"
                      f"Accuracy: {results.get('overall_accuracy', 0):.4f}  "
                      f"F1: {results.get('weighted_f1', 0):.4f}")
        plt.tight_layout()

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = RESULTS_DIR / f"{prefix}_confusion_matrix.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {plot_path}")
    except Exception as e:
        print(f"  [WARN] Confusion matrix plot failed: {e}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FluxChi Model Training Pipeline")
    parser.add_argument(
        "--source", choices=["ninapro", "self", "both"], default="ninapro",
        help="Data source: ninapro (public dataset validation), self (deploy model), both",
    )
    parser.add_argument("--ninapro-dir", default="datasets/ninapro_db5", help="Ninapro DB5 directory")
    parser.add_argument("--data-dir", default="data", help="Self-collected data directory")
    parser.add_argument("--model-dir", default="model", help="Output model directory")
    parser.add_argument("--n-estimators", type=int, default=200, help="RandomForest trees")
    args = parser.parse_args()

    print("=" * 60)
    print("FluxChi — Model Training Pipeline")
    print("=" * 60)

    if args.source in ("ninapro", "both"):
        print(f"\n[1] Loading Ninapro DB5 from {args.ninapro_dir}...")
        try:
            X, y, groups = load_ninapro_db5(args.ninapro_dir)
            model, results = train_and_evaluate(X, y, groups, source_name="ninapro_db5")
            save_model(model, results, args.model_dir, prefix="ninapro_classifier")
            save_confusion_matrix_plot(results, prefix="ninapro_classifier")
        except Exception as e:
            print(f"\n  [ERROR] Ninapro training failed: {e}")
            if args.source == "ninapro":
                sys.exit(1)

    if args.source in ("self", "both"):
        print(f"\n[2] Loading self-collected data from {args.data_dir}...")
        try:
            X, y, groups = load_self_collected(args.data_dir)
            model, results = train_and_evaluate(X, y, groups, source_name="self_collected")
            save_model(model, results, args.model_dir, prefix="activity_classifier")
            save_confusion_matrix_plot(results, prefix="activity_classifier")
        except Exception as e:
            print(f"\n  [ERROR] Self-collected training failed: {e}")
            if args.source == "self":
                sys.exit(1)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"  Models: {args.model_dir}/")
    print(f"  Reports: {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
