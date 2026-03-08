# Project Flux – Developer Guide

This note explains how to restart, record, train, and extend Project Flux once everything is checked out from GitHub.

## 1. Environment Setup

```bash
cd harward-gesture
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Hardware Check

```bash
python3 -m tools.serial_probe --port /dev/cu.usbserial-0001 --seconds 5
```
- Confirms AA/BB frames, sequence numbers, and EMG throughput.

## 3. Run the App

```bash
python3 app.py --port /dev/cu.usbserial-0001 --baud 921600 --fs 1000
```

### UI Controls
- **Label Box**: Enter gesture name.
- **Record**: Start/stop recording with current label.
- **Train**: Trigger RandomForest training.
- **Inference**: Toggle real-time recognition.
- **Screenshot**: Save UI snapshot into `screenshots/`.
- **Quit**: Close the Matplotlib window.

Keyboard shortcuts (window must be focused): `R`, `T`, `I`, `S`, `Q`.

## 4. Workflow
1. Set label → click Record → hold gesture (5–10 s) → click Record again.
2. Repeat for every gesture (2–3 recordings each).
3. Click Train, wait for completion.
4. Click Inference to see predictions + fatigue/anxiety/focus states.

## 5. Saving & Version Control
- Data stored under `data/<gesture>/<timestamp>.csv`.
- Trained model saved as `model/model.pkl` + `model/config.json`.
- Git should ignore `.venv/`, `data/`, `model/`, `__pycache__/`, `.DS_Store`.
- Use a clean repo state before pushing:
  ```bash
  git status
  git add ...
  git commit -m "..."
  git push
  ```

## 6. Extending the Project
- `src/stream.py`: Serial parser + IMU/EMG scaling.
- `src/state.py`: Adjust fatigue/anxiety/focus logic or thresholds.
- `src/features.py`: LibEMG integration + fallback features.
- `src/ui.py`: Buttons, label text box, status displays.
- `src/bridge.py`: UDP bridge to LibEMG’s OnlineDataHandler.

When adding dependencies, update `requirements.txt`. For major changes, consider duplicating this guide with the new steps so future developers have a clear path.
