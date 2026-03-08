#!/usr/bin/env python3
"""FPS-style HTML mini-game powered by EMG gestures."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import websockets

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.features import FeatureExtractor
from src.inference import GestureInference, PredictionResult
from src.stream import EMGSample, SerialEMGStream
from src.trainer import Trainer, load_model


@dataclass
class GestureProfile:
    name: str
    data_dir: Path
    model_dir: Path
    config_path: Path
    gesture_map: Dict[str, str]


def _record_window(stream: SerialEMGStream, seconds: float) -> List[EMGSample]:
    start = time.perf_counter()
    collected: List[EMGSample] = []
    while time.perf_counter() - start < seconds:
        elapsed = time.perf_counter() - start
        progress = min(1.0, elapsed / seconds)
        bar = "█" * int(progress * 20)
        pad = "·" * (20 - len(bar))
        print(f"\r采集中: [{bar}{pad}] {progress * 100:3.0f}%", end="", flush=True)
        samples = stream.consume_samples()
        if samples:
            collected.extend(samples)
        else:
            time.sleep(0.005)
    print("\r采集中: [████████████████████] 100%")
    return collected


def _save_recording(data_dir: Path, label: str, samples: Iterable[EMGSample]) -> Optional[Path]:
    rows = list(samples)
    if not rows:
        print(f"[calibrate] 没有采集到 {label} 的样本。")
        return None
    gesture_dir = data_dir / label
    gesture_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = gesture_dir / f"{timestamp}.csv"
    start = rows[0].timestamp
    header = ["t"] + [f"ch{idx+1}" for idx in range(len(rows[0].values))]
    with path.open("w") as fh:
        fh.write(",".join(header) + "\n")
        for sample in rows:
            rel_t = sample.timestamp - start
            values = [f"{rel_t:.6f}"] + [f"{val:.4f}" for val in sample.values.tolist()]
            fh.write(",".join(values) + "\n")
    print(f"[calibrate] 保存 {label} 到 {path}")
    return path


def _calibrate_profile(profile: GestureProfile, stream: SerialEMGStream, args: argparse.Namespace) -> None:
    print("\n=== 手势校准开始 ===")
    print("建议每个动作保持 2-3 秒，重复 2 次。\n")

    profile.gesture_map = {
        "shoot": args.shoot_label,
        "look_left": args.left_label,
        "look_right": args.right_label,
        "look_up": args.up_label,
        "look_down": args.down_label,
    }
    if args.switch_label:
        profile.gesture_map["switch"] = args.switch_label
    if args.reload_label:
        profile.gesture_map["reload"] = args.reload_label

    labels = list(profile.gesture_map.values())
    print("\n开始采集，请按提示动作。")
    for label in labels:
        for rep in range(1, args.reps + 1):
            print(f"\n准备动作 {label} (第 {rep}/{args.reps} 次) ...")
            time.sleep(args.countdown)
            samples = _record_window(stream, args.record_seconds)
            _save_recording(profile.data_dir, label, samples)

    profile.config_path.parent.mkdir(parents=True, exist_ok=True)
    with profile.config_path.open("w") as fh:
        json.dump({"gesture_map": profile.gesture_map}, fh, indent=2)
    print("\n[calibrate] 完成采集，开始训练模型...")

    trainer = Trainer(
        data_dir=str(profile.data_dir),
        model_dir=str(profile.model_dir),
        sample_rate=args.fs,
        window_seconds=args.window,
        stride_seconds=args.stride,
    )
    trainer.train()


def _load_profile(args: argparse.Namespace) -> GestureProfile:
    base = Path("profiles") / args.profile
    return GestureProfile(
        name=args.profile,
        data_dir=base / "data",
        model_dir=base / "model",
        config_path=base / "config.json",
        gesture_map={},
    )


def _load_gesture_map(profile: GestureProfile) -> None:
    if not profile.config_path.exists():
        return
    with profile.config_path.open() as fh:
        data = json.load(fh)
    profile.gesture_map = {k: str(v) for k, v in data.get("gesture_map", {}).items()}


def _build_inference(profile: GestureProfile, args: argparse.Namespace) -> GestureInference:
    model, config = load_model(str(profile.model_dir))
    if not model or not config:
        raise SystemExit("[fps] 未找到模型，请先校准并训练。")
    extractor = FeatureExtractor(
        sample_rate=int(config.get("sample_rate", args.fs)),
        window_seconds=float(config.get("window_seconds", args.window)),
        stride_seconds=float(config.get("stride_seconds", args.stride)),
    )
    return GestureInference(
        model=model,
        extractor=extractor,
        stable_windows=args.stable_windows,
        threshold=args.threshold,
    )


def _resolve_actions(profile: GestureProfile, inference: GestureInference) -> Dict[str, str]:
    available = {str(label) for label in inference.model.classes_}
    missing = [label for label in profile.gesture_map.values() if label not in available]
    if missing:
        print(f"[fps] 警告：模型缺少这些手势标签: {', '.join(missing)}")
        print("[fps] 建议重新校准，或者检查手势标签是否一致。")
    return profile.gesture_map


def _inference_worker(
    stream: SerialEMGStream,
    inference: GestureInference,
    actions: Dict[str, str],
    out_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    inverse_map = {label: action for action, label in actions.items()}
    last_prediction_time = 0.0
    last_action_time: Dict[str, float] = {}
    while not stop_event.is_set():
        samples = stream.consume_samples()
        if not samples:
            time.sleep(0.01)
            continue
        results = inference.process(samples)
        for result in results:
            if result.timestamp - last_prediction_time > 0.05:
                out_queue.put(
                    {
                        "type": "prediction",
                        "label": result.label,
                        "confidence": result.confidence,
                        "stable": result.stable,
                    }
                )
                last_prediction_time = result.timestamp
            if result.label in inverse_map and result.confidence >= inference.threshold:
                last_time = last_action_time.get(result.label, 0.0)
                if result.timestamp - last_time > 0.18:
                    last_action_time[result.label] = result.timestamp
                    out_queue.put({"type": "action", "action": inverse_map[result.label]})


async def _broadcast_loop(
    clients: set,
    out_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            message = out_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.02)
            continue
        payload = json.dumps(message)
        dead = []
        for ws in clients:
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)


async def _ws_handler(websocket, clients: set, greeting: dict) -> None:
    clients.add(websocket)
    await websocket.send(json.dumps(greeting))
    try:
        async for _ in websocket:
            pass
    finally:
        clients.discard(websocket)


def _start_ws_server(actions: Dict[str, str], out_queue: queue.Queue, stop_event: threading.Event) -> None:
    async def runner() -> None:
        clients: set = set()
        greeting = {
            "type": "hello",
            "actions": actions,
        }
        async with websockets.serve(
            lambda ws: _ws_handler(ws, clients, greeting),
            host="127.0.0.1",
            port=8765,
        ):
            await _broadcast_loop(clients, out_queue, stop_event)

    asyncio.run(runner())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EMG FPS gesture game server")
    parser.add_argument("--port", required=True, help="Serial port (e.g., /dev/cu.usbserial-0001)")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baudrate")
    parser.add_argument("--fs", type=int, default=1000, help="Sample rate (Hz)")
    parser.add_argument("--window", type=float, default=0.5, help="Feature window seconds")
    parser.add_argument("--stride", type=float, default=0.1, help="Feature stride seconds")
    parser.add_argument("--stable-windows", type=int, default=3, help="Stable prediction windows")
    parser.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold")
    parser.add_argument("--profile", default="default", help="Player profile name")
    parser.add_argument("--calibrate", action="store_true", help="Force recalibration")
    parser.add_argument("--reps", type=int, default=2, help="Repetitions per gesture")
    parser.add_argument("--record-seconds", type=float, default=3.0, help="Seconds per repetition")
    parser.add_argument("--countdown", type=float, default=1.0, help="Countdown before capture")
    parser.add_argument("--shoot-label", default="shoot", help="Gesture label for shoot")
    parser.add_argument("--left-label", default="left", help="Gesture label for look left")
    parser.add_argument("--right-label", default="right", help="Gesture label for look right")
    parser.add_argument("--up-label", default="up", help="Gesture label for look up")
    parser.add_argument("--down-label", default="down", help="Gesture label for look down")
    parser.add_argument("--switch-label", help="Gesture label for switch weapon")
    parser.add_argument("--reload-label", help="Gesture label for reload")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = _load_profile(args)
    _load_gesture_map(profile)

    stream = SerialEMGStream(args.port, args.baud)
    stream.start()
    try:
        if args.calibrate or not profile.config_path.exists():
            _calibrate_profile(profile, stream, args)
        inference = _build_inference(profile, args)
        actions = _resolve_actions(profile, inference)
        print("\n[fps] WebSocket 启动: ws://127.0.0.1:8765")
        print("[fps] 打开 game/fps_game.html 即可开始游戏。\n")

        out_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        worker = threading.Thread(
            target=_inference_worker,
            args=(stream, inference, actions, out_queue, stop_event),
            daemon=True,
        )
        worker.start()
        _start_ws_server(actions, out_queue, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
