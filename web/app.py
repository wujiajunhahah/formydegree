"""
FluxChi — Web 后端
=========================================================
FastAPI + WebSocket 实时推送 EMG 状态到前端仪表盘。

用法：
  python web/app.py --port /dev/tty.usbserial-XXXX
  python web/app.py --demo   # 无硬件演示模式

浏览器打开: http://localhost:8000
=========================================================
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.stream import CHANNEL_COUNT, BaseEMGStream, SerialEMGStream
from src.fatigue import FatigueEstimator, FatigueReading
from src.decision import DecisionEngine, DecisionOutput

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="FluxChi", version="1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── Global state ────────────────────────────────────────────
stream: Optional[BaseEMGStream] = None
model = None
model_config: Optional[dict] = None
fatigue_estimator: Optional[FatigueEstimator] = None
decision_engine: Optional[DecisionEngine] = None
feature_extractor = None
active_connections: Set[WebSocket] = set()
demo_mode = False
timeline: List[Dict] = []


# ─── Synthetic stream for demo mode ─────────────────────────
class DemoEMGStream(BaseEMGStream):
    """Generates synthetic EMG data that cycles through activities."""

    def __init__(self) -> None:
        from src.stream import EMGSample, FrameStats
        self._running = False
        self._stats = FrameStats()
        self._phase = 0.0
        self._activity_idx = 0
        self._activities = ["typing", "typing", "mouse_use", "idle", "stretching"]
        self._switch_interval = 30.0
        self._start_time = time.perf_counter()

    def start(self) -> None:
        self._running = True
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        self._running = False

    def consume_samples(self, max_items: int = 256) -> list:
        from src.stream import EMGSample
        if not self._running:
            return []
        elapsed = time.perf_counter() - self._start_time
        self._activity_idx = int(elapsed / self._switch_interval) % len(self._activities)
        activity = self._activities[self._activity_idx]

        samples = []
        now = time.perf_counter()
        for i in range(50):
            values = self._generate_emg(activity, now + i * 0.001)
            samples.append(EMGSample(timestamp=now + i * 0.001, values=values))
        self._stats.total_frames += len(samples)
        self._stats.emg_frames += len(samples)
        return samples

    def _generate_emg(self, activity: str, t: float) -> np.ndarray:
        rng = np.random.default_rng(int(t * 1000) % (2**31))
        base = rng.normal(0, 1, CHANNEL_COUNT)
        if activity == "typing":
            base[:4] *= 50
            base[4:] *= 10
            for i in range(4):
                base[i] += 30 * np.sin(2 * np.pi * 3 * t + i)
        elif activity == "mouse_use":
            base[:2] *= 40
            base[2:] *= 5
        elif activity == "stretching":
            base *= 60
            for i in range(CHANNEL_COUNT):
                base[i] += 20 * np.sin(2 * np.pi * 0.5 * t + i * 0.3)
        else:
            base *= 3
        return base

    def consume_imu(self, max_items: int = 256) -> list:
        return []

    def frame_stats(self):
        return self._stats


# ─── Demo classifier ────────────────────────────────────────
class DemoClassifier:
    """Fake classifier that returns the demo stream's current activity."""
    classes_ = np.array(["typing", "mouse_use", "idle", "stretching"])

    def predict_proba(self, X):
        idx = demo_stream_activity_idx()
        probs = np.full((len(X), 4), 0.05)
        probs[:, idx] = 0.85
        return probs

    @property
    def n_features_in_(self):
        return 84


def demo_stream_activity_idx() -> int:
    if stream and isinstance(stream, DemoEMGStream):
        return stream._activity_idx
    return 0


# ─── Routes ──────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def status():
    return {
        "connected": stream is not None and (hasattr(stream, '_running') and stream._running if hasattr(stream, '_running') else True),
        "model_loaded": model is not None,
        "demo_mode": demo_mode,
        "timeline_entries": len(timeline),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_connections.add(ws)
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            if data.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        active_connections.discard(ws)
    except Exception:
        active_connections.discard(ws)


async def broadcast(data: dict):
    dead = set()
    for ws in active_connections:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    active_connections.difference_update(dead)


# ─── Main processing loop ───────────────────────────────────
async def processing_loop():
    """Continuously read EMG, classify, estimate fatigue, decide."""
    global fatigue_estimator, decision_engine, feature_extractor

    from src.features import FeatureExtractor

    sample_rate = 1000 if not demo_mode else 1000
    fe = FeatureExtractor(
        sample_rate=sample_rate,
        window_seconds=0.25,
        stride_seconds=0.1,
    )
    feature_extractor = fe
    fatigue_estimator = FatigueEstimator(sample_rate=sample_rate)
    decision_engine = DecisionEngine()

    emg_buffer = np.zeros((0, CHANNEL_COUNT))
    last_push = time.time()

    while True:
        if stream is None:
            await asyncio.sleep(0.1)
            continue

        samples = stream.consume_samples(max_items=512)
        if not samples:
            await asyncio.sleep(0.02)
            continue

        new_data = np.array([s.values for s in samples])
        emg_buffer = np.vstack([emg_buffer, new_data]) if emg_buffer.size else new_data

        if emg_buffer.shape[0] > sample_rate * 10:
            emg_buffer = emg_buffer[-sample_rate * 5:]

        now = time.time()
        if now - last_push < 0.2:
            await asyncio.sleep(0.01)
            continue
        last_push = now

        activity = "idle"
        confidence = 0.0
        probabilities = {}

        if model is not None and emg_buffer.shape[0] >= fe.window_samples:
            window = emg_buffer[-fe.window_samples:]
            try:
                feats = fe.transform_window(window)

                expected = model.n_features_in_
                if len(feats) < expected:
                    feats.extend([0.0] * (expected - len(feats)))
                elif len(feats) > expected:
                    feats = feats[:expected]

                proba = model.predict_proba([feats])[0]
                best_idx = int(np.argmax(proba))
                activity = str(model.classes_[best_idx])
                confidence = float(proba[best_idx])
                probabilities = {
                    str(model.classes_[i]): float(proba[i])
                    for i in range(len(model.classes_))
                }
            except Exception as e:
                print(f"[inference] {e}")

        fatigue_reading = FatigueReading(0.0, 0.0, 0.0, 0.0, 0.0)
        if fatigue_estimator and emg_buffer.shape[0] >= fe.window_samples:
            try:
                window = emg_buffer[-fe.window_samples:]
                fatigue_reading = fatigue_estimator.update(now, window)
            except Exception:
                pass

        decision = None
        if decision_engine:
            try:
                decision = decision_engine.update(activity, fatigue_reading.score, now)
            except Exception:
                pass

        rms_values = np.sqrt(np.mean(emg_buffer[-min(250, len(emg_buffer)):] ** 2, axis=0)).tolist()

        payload = {
            "type": "state_update",
            "timestamp": now,
            "activity": activity,
            "confidence": confidence,
            "probabilities": probabilities,
            "fatigue": {
                "score": fatigue_reading.score,
                "mdf_current": fatigue_reading.mdf_current,
                "mdf_baseline": fatigue_reading.mdf_baseline,
                "mdf_slope": fatigue_reading.mdf_slope,
                "confidence": fatigue_reading.confidence,
            },
            "rms": rms_values,
            "emg_sample_count": len(emg_buffer),
        }

        if decision:
            payload["decision"] = {
                "state": decision.state.value,
                "recommendation": decision.recommendation.value,
                "urgency": decision.urgency,
                "reasons": decision.reasons,
                "continuous_work_min": decision.continuous_work_min,
                "total_work_min": decision.total_work_min,
            }
            if decision.urgency >= 0.5:
                timeline.append({
                    "time": now,
                    "type": "alert",
                    "message": decision.reasons[0] if decision.reasons else "",
                    "urgency": decision.urgency,
                })
                if len(timeline) > 200:
                    timeline[:] = timeline[-100:]

        await broadcast(payload)
        await asyncio.sleep(0.01)


@app.on_event("startup")
async def startup():
    asyncio.create_task(processing_loop())


# ─── Entrypoint ──────────────────────────────────────────────
def main():
    global stream, model, model_config, demo_mode

    parser = argparse.ArgumentParser(description="FluxChi Web Server")
    parser.add_argument("--port", help="Serial port for WAVELETECH wristband")
    parser.add_argument("--demo", action="store_true", help="Demo mode (no hardware)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8000)
    parser.add_argument("--model-dir", default="model")
    args = parser.parse_args()

    demo_mode = args.demo

    model_dir = Path(args.model_dir)
    for prefix in ("activity_classifier", "ninapro_classifier", "model"):
        pkl_path = model_dir / f"{prefix}.pkl"
        if pkl_path.exists():
            import joblib
            model = joblib.load(pkl_path)
            config_path = model_dir / f"{prefix}_config.json"
            if config_path.exists():
                model_config = json.loads(config_path.read_text())
            print(f"[web] Loaded model: {pkl_path}")
            break

    if demo_mode:
        print("[web] Running in DEMO mode (synthetic data)")
        stream = DemoEMGStream()
        stream.start()
        if model is None:
            model = DemoClassifier()
            print("[web] Using demo classifier")
    elif args.port:
        print(f"[web] Connecting to {args.port}...")
        stream = SerialEMGStream(args.port)
        stream.start()
    else:
        print("[web] No port specified and --demo not set.")
        print("[web] Use --port /dev/tty.usbserial-XXXX or --demo")
        sys.exit(1)

    import uvicorn
    print(f"\n  FluxChi Web: http://localhost:{args.web_port}")
    print(f"  Demo mode: {demo_mode}")
    print(f"  Model: {'loaded' if model else 'none'}\n")
    uvicorn.run(app, host=args.host, port=args.web_port, log_level="warning")


if __name__ == "__main__":
    main()
