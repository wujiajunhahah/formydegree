"""
FluxChi — 疲劳估计器
=========================================================
基于 Median Frequency (MDF) 随时间的下降趋势来估计肌肉疲劳。

核心思路 (De Luca, 1997):
  肌肉疲劳 → 运动单元放电频率↓ → EMG频谱左移 → MDF↓

方法:
  1. 用 AR(3) 模型估计 PSD (比 FFT 在短窗口上更稳定)
  2. 计算 MDF (频谱面积二等分点)
  3. 对 MDF 时间序列做线性回归，斜率即疲劳指标
=========================================================
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np

from .stream import CHANNEL_COUNT


@dataclass
class FatigueReading:
    """Fatigue estimation output."""
    score: float          # 0-1, higher = more fatigued
    mdf_current: float    # current MDF in Hz
    mdf_baseline: float   # baseline MDF in Hz
    mdf_slope: float      # Hz/s, negative = increasing fatigue
    confidence: float     # 0-1, based on data sufficiency


class FatigueEstimator:
    """Estimate muscle fatigue from MDF trend using AR-based PSD."""

    def __init__(
        self,
        sample_rate: int = 1000,
        window_seconds: float = 0.5,
        history_seconds: float = 120.0,
        ar_order: int = 3,
        baseline_windows: int = 20,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = max(1, int(window_seconds * sample_rate))
        self.history_seconds = history_seconds
        self.ar_order = ar_order
        self.baseline_windows = baseline_windows

        self._mdf_history: Deque[Tuple[float, float]] = deque()
        self._mdf_baseline: Optional[float] = None
        self._baseline_buffer: list[float] = []
        self._latest: Optional[FatigueReading] = None

    def update(self, timestamp: float, window: np.ndarray) -> FatigueReading:
        """Process one EMG window and return fatigue reading.

        Args:
            timestamp: time in seconds
            window: shape (samples, channels) or (channels, samples)
        """
        if window.ndim == 2:
            if window.shape[0] == CHANNEL_COUNT and window.shape[1] != CHANNEL_COUNT:
                window = window.T
            n_ch = min(window.shape[1], CHANNEL_COUNT)
        else:
            window = window.reshape(-1, 1)
            n_ch = 1

        mdfs = []
        for ch in range(n_ch):
            sig = window[:, ch].astype(np.float64)
            mdf = self._ar_mdf(sig)
            if mdf > 0:
                mdfs.append(mdf)

        if not mdfs:
            return self._latest or FatigueReading(0.0, 0.0, 0.0, 0.0, 0.0)

        current_mdf = float(np.median(mdfs))

        if self._mdf_baseline is None:
            self._baseline_buffer.append(current_mdf)
            if len(self._baseline_buffer) >= self.baseline_windows:
                self._mdf_baseline = float(np.median(self._baseline_buffer))
                self._baseline_buffer.clear()

        self._mdf_history.append((timestamp, current_mdf))
        while (self._mdf_history and
               timestamp - self._mdf_history[0][0] > self.history_seconds):
            self._mdf_history.popleft()

        slope = self._compute_slope()
        score = self._compute_score(current_mdf, slope)
        confidence = self._compute_confidence()

        self._latest = FatigueReading(
            score=score,
            mdf_current=current_mdf,
            mdf_baseline=self._mdf_baseline or current_mdf,
            mdf_slope=slope,
            confidence=confidence,
        )
        return self._latest

    def reset(self) -> None:
        self._mdf_history.clear()
        self._mdf_baseline = None
        self._baseline_buffer.clear()
        self._latest = None

    @property
    def latest(self) -> Optional[FatigueReading]:
        return self._latest

    def _ar_mdf(self, signal: np.ndarray) -> float:
        """Compute MDF using AR model PSD estimation (Burg method)."""
        n = len(signal)
        if n < self.ar_order + 2:
            return 0.0

        signal = signal - np.mean(signal)
        norm = np.std(signal)
        if norm < 1e-10:
            return 0.0
        signal = signal / norm

        ar_coeffs, sigma2 = self._burg_ar(signal, self.ar_order)
        if ar_coeffs is None or sigma2 <= 0:
            return self._fft_mdf(signal * norm)

        n_fft = 256
        freqs = np.linspace(0, self.sample_rate / 2, n_fft)
        psd = np.zeros(n_fft)

        for i, f in enumerate(freqs):
            z = np.exp(-2j * np.pi * f / self.sample_rate)
            denom = 1.0
            for k in range(len(ar_coeffs)):
                denom += ar_coeffs[k] * (z ** (k + 1))
            psd[i] = sigma2 / (np.abs(denom) ** 2 + 1e-20)

        cumulative = np.cumsum(psd)
        total = cumulative[-1]
        if total < 1e-12:
            return 0.0
        idx = np.searchsorted(cumulative, total / 2.0)
        return float(freqs[min(idx, len(freqs) - 1)])

    def _burg_ar(
        self, signal: np.ndarray, order: int
    ) -> Tuple[Optional[np.ndarray], float]:
        """Burg method for AR parameter estimation — more stable than Yule-Walker."""
        n = len(signal)
        if n <= order:
            return None, 0.0

        ef = signal.copy()
        eb = signal.copy()
        a = np.zeros(order)
        sigma2 = float(np.mean(signal ** 2))

        for m in range(order):
            efm = ef[m + 1:]
            ebm = eb[m:-1] if m > 0 else eb[:-1]

            num = -2.0 * np.dot(efm, ebm)
            den = np.dot(efm, efm) + np.dot(ebm, ebm)

            if abs(den) < 1e-20:
                return None, 0.0

            km = num / den

            if abs(km) >= 1.0:
                km = np.clip(km, -0.999, 0.999)

            new_a = np.zeros(m + 1)
            new_a[m] = km
            for j in range(m):
                new_a[j] = a[j] + km * a[m - 1 - j]
            a[:m + 1] = new_a

            sigma2 *= (1.0 - km * km)

            new_ef = ef[1:] + km * eb[:-1]
            new_eb = eb[:-1] + km * ef[1:]
            ef = np.concatenate([[ef[0]], new_ef])
            eb = np.concatenate([new_eb, [eb[-1]]])

        return a[:order], sigma2

    def _fft_mdf(self, signal: np.ndarray) -> float:
        """Fallback: FFT-based MDF."""
        if len(signal) < 4:
            return 0.0
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.sample_rate)
        psd = np.abs(np.fft.rfft(signal)) ** 2
        cumulative = np.cumsum(psd)
        total = cumulative[-1]
        if total < 1e-12:
            return 0.0
        idx = np.searchsorted(cumulative, total / 2.0)
        return float(freqs[min(idx, len(freqs) - 1)])

    def _compute_slope(self) -> float:
        if len(self._mdf_history) < 3:
            return 0.0
        times = np.array([t for t, _ in self._mdf_history])
        values = np.array([v for _, v in self._mdf_history])
        times_norm = times - times[0]
        if np.allclose(times_norm, 0.0):
            return 0.0
        slope, _ = np.polyfit(times_norm, values, 1)
        return float(slope)

    def _compute_score(self, current_mdf: float, slope: float) -> float:
        if self._mdf_baseline is None or self._mdf_baseline < 1e-6:
            return 0.0

        drop_ratio = max(0.0, (self._mdf_baseline - current_mdf) / self._mdf_baseline)
        slope_term = max(0.0, -slope / (self._mdf_baseline + 1e-6))
        raw = 0.6 * drop_ratio + 0.4 * min(slope_term * 60, 1.0)
        return float(np.clip(raw, 0.0, 1.0))

    def _compute_confidence(self) -> float:
        n = len(self._mdf_history)
        if n < 3:
            return 0.1
        if self._mdf_baseline is None:
            return 0.2
        if n < 10:
            return 0.3 + 0.07 * n
        return min(1.0, 0.5 + 0.05 * n)
