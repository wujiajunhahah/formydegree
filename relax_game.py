#!/usr/bin/env python3
"""Relaxation mini-game driven by EMG gesture inference."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

from src.features import FeatureExtractor
from src.inference import GestureInference, PredictionResult
from src.stream import SerialEMGStream
from src.trainer import load_model


@dataclass
class GameFeedback:
    message: str
    color: str
    timestamp: float


class RelaxGame:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stream = SerialEMGStream(args.port, args.baud)
        self.inference, self.gestures = self._load_inference()
        self.inhale_label = args.inhale or self.gestures[0]
        self.exhale_label = args.exhale or self.gestures[1]
        if self.inhale_label not in self.gestures:
            raise SystemExit(f"[relax] Unknown inhale gesture '{self.inhale_label}'.")
        if self.exhale_label not in self.gestures:
            raise SystemExit(f"[relax] Unknown exhale gesture '{self.exhale_label}'.")
        self.phase = "inhale"
        self.phase_start = time.perf_counter()
        self.phase_success = False

        self.score = 0
        self.combo = 0
        self.calm = 0.5
        self.latest_prediction: Optional[PredictionResult] = None
        self.feedback: Optional[GameFeedback] = None

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self._configure_axes()
        self.ring = Circle((0, 0), 0.3, fill=False, linewidth=6, edgecolor="#5BC0EB")
        self.ax.add_patch(self.ring)
        self.target_ring = Circle((0, 0), 0.65, fill=False, linewidth=2, edgecolor="#7BD389")
        self.ax.add_patch(self.target_ring)
        self.calm_bar = Rectangle((-0.8, -0.95), 1.6 * self.calm, 0.08, color="#7BD389")
        self.ax.add_patch(self.calm_bar)

        self.title_text = self.ax.text(0, 0.88, "EMG Relax Game", ha="center", color="#F5F5F5")
        self.target_text = self.ax.text(0, 0.72, "", ha="center", color="#F5F5F5", fontsize=14)
        self.phase_text = self.ax.text(0, 0.6, "", ha="center", color="#9CC9FF")
        self.score_text = self.ax.text(0, -0.72, "", ha="center", color="#F5F5F5")
        self.prediction_text = self.ax.text(0, -0.82, "", ha="center", color="#F5F5F5", fontsize=10)
        self.feedback_text = self.ax.text(0, -0.6, "", ha="center", color="#7BD389", fontsize=14)
        self.calm_label = self.ax.text(-0.8, -0.88, "Calm", ha="left", color="#9CC9FF", fontsize=10)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _configure_axes(self) -> None:
        self.fig.patch.set_facecolor("#0F1B2D")
        self.ax.set_facecolor("#0F1B2D")
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def _load_inference(self) -> tuple[GestureInference, List[str]]:
        model, config = load_model(self.args.model_dir)
        if not model or not config:
            raise SystemExit("[relax] No trained model found. Train gestures first in app.py.")

        gestures = [str(label) for label in model.classes_]
        if len(gestures) < 2:
            raise SystemExit("[relax] Need at least two gesture classes for the game.")

        extractor = FeatureExtractor(
            sample_rate=int(config.get("sample_rate", self.args.fs)),
            window_seconds=float(config.get("window_seconds", self.args.window)),
            stride_seconds=float(config.get("stride_seconds", self.args.stride)),
        )
        inference = GestureInference(
            model=model,
            extractor=extractor,
            stable_windows=self.args.stable_windows,
            threshold=self.args.threshold,
        )
        return inference, gestures

    def run(self) -> None:
        print("[relax] Starting stream...")
        self.stream.start()
        self._print_instructions()
        self.animation = FuncAnimation(self.fig, self._update_frame, interval=40)
        try:
            plt.show()
        finally:
            self.stream.stop()

    def _print_instructions(self) -> None:
        print("[relax] Controls: Q to quit")
        print(f"[relax] Inhale gesture: {self.inhale_label}")
        print(f"[relax] Exhale gesture: {self.exhale_label}")
        if len(self.gestures) > 2:
            extra = ", ".join(self.gestures[2:])
            print(f"[relax] Extra gestures detected (ignored): {extra}")

    def _on_key(self, event) -> None:
        if (event.key or "").lower() == "q":
            plt.close(self.fig)

    def _on_close(self, event) -> None:
        self.stream.stop()

    def _update_frame(self, _frame) -> None:
        now = time.perf_counter()
        self._advance_phase(now)
        self._process_samples()
        self._update_visuals(now)

    def _advance_phase(self, now: float) -> None:
        duration = self.args.inhale_seconds if self.phase == "inhale" else self.args.exhale_seconds
        if now - self.phase_start >= duration:
            self.phase = "exhale" if self.phase == "inhale" else "inhale"
            self.phase_start = now
            self.phase_success = False

    def _process_samples(self) -> None:
        samples = self.stream.consume_samples()
        if not samples:
            return
        results = self.inference.process(samples)
        for result in results:
            self.latest_prediction = result
            if result.stable:
                self._handle_prediction(result)

    def _handle_prediction(self, result: PredictionResult) -> None:
        target = self._target_label()
        if result.label == target and not self.phase_success:
            self.phase_success = True
            self.combo += 1
            bonus = 2 * max(0, self.combo - 1)
            self.score += 10 + bonus
            self.calm = min(1.0, self.calm + 0.06)
            self.feedback = GameFeedback("Great timing!", "#7BD389", time.perf_counter())
        elif result.label != target:
            self.combo = 0
            self.calm = max(0.0, self.calm - 0.04)
            self.feedback = GameFeedback("Try the target gesture", "#F45B69", time.perf_counter())

    def _target_label(self) -> str:
        return self.inhale_label if self.phase == "inhale" else self.exhale_label

    def _phase_progress(self, now: float) -> float:
        duration = self.args.inhale_seconds if self.phase == "inhale" else self.args.exhale_seconds
        return min(1.0, max(0.0, (now - self.phase_start) / duration))

    def _update_visuals(self, now: float) -> None:
        progress = self._phase_progress(now)
        smooth = 0.5 - 0.5 * math.cos(math.pi * progress)
        min_radius = 0.28
        max_radius = 0.72
        if self.phase == "inhale":
            radius = min_radius + (max_radius - min_radius) * smooth
        else:
            radius = max_radius - (max_radius - min_radius) * smooth

        self.ring.set_radius(radius)
        self.ring.set_edgecolor("#7BD389" if self.phase_success else "#5BC0EB")
        self.target_ring.set_edgecolor("#9CC9FF" if self.phase == "inhale" else "#FAD02C")

        target = self._target_label()
        self.target_text.set_text(f"Target Gesture: {target}")
        phase_label = "Inhale – slow, steady" if self.phase == "inhale" else "Exhale – release"
        self.phase_text.set_text(phase_label)
        self.score_text.set_text(f"Score: {self.score}  |  Combo: {self.combo}")

        if self.latest_prediction:
            pred = f"Prediction: {self.latest_prediction.label} ({self.latest_prediction.confidence:.2f})"
        else:
            pred = "Prediction: --"
        self.prediction_text.set_text(pred)

        self.calm_bar.set_width(1.6 * self.calm)
        self.calm_bar.set_color("#7BD389" if self.calm > 0.4 else "#FAD02C")

        if self.feedback and now - self.feedback.timestamp < 1.2:
            self.feedback_text.set_text(self.feedback.message)
            self.feedback_text.set_color(self.feedback.color)
        else:
            self.feedback_text.set_text("")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EMG relaxation mini-game")
    parser.add_argument("--port", required=True, help="Serial port (e.g., /dev/cu.usbserial-0001)")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baudrate")
    parser.add_argument("--fs", type=int, default=1000, help="Sample rate (Hz)")
    parser.add_argument("--model-dir", default="model", help="Directory containing model.pkl")
    parser.add_argument("--window", type=float, default=0.5, help="Feature window in seconds")
    parser.add_argument("--stride", type=float, default=0.1, help="Feature stride in seconds")
    parser.add_argument("--stable-windows", type=int, default=3, help="Stable prediction windows")
    parser.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold")
    parser.add_argument("--inhale-seconds", type=float, default=4.0, help="Inhale duration")
    parser.add_argument("--exhale-seconds", type=float, default=6.0, help="Exhale duration")
    parser.add_argument("--inhale", help="Gesture label for inhale")
    parser.add_argument("--exhale", help="Gesture label for exhale")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    game = RelaxGame(args)
    game.run()


if __name__ == "__main__":
    main()
