#!/bin/bash
set -euo pipefail

PORT=${1:-}
PROFILE=${2:-player1}

if [[ -z "${PORT}" ]]; then
  echo "Usage: ./game/start_fps_game.sh <serial_port> [profile] [extra flags]"
  echo "Example: ./game/start_fps_game.sh /dev/cu.usbserial-0001 player1 --calibrate"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -d ".venv" ]]; then
  echo "[start] .venv not found, please run: python3 -m venv .venv"
  exit 1
fi

source .venv/bin/activate
pip install -r requirements.txt

shift 2 || true
python3 game/fps_game.py --port "${PORT}" --profile "${PROFILE}" "$@"
