#!/bin/bash
# FluxChi — 答辩演示一键启动
# 用法:
#   ./scripts/demo.sh          # 演示模式（无需硬件）
#   ./scripts/demo.sh real     # 真实模式（需要手环）

set -e
cd "$(dirname "$0")/.."

MODE="${1:-demo}"

if [ "$MODE" = "demo" ]; then
    echo ""
    echo "  ╔══════════════════════════════════════╗"
    echo "  ║   FluxChi — 演示模式                  ║"
    echo "  ║   浏览器打开: http://localhost:8000   ║"
    echo "  ╚══════════════════════════════════════╝"
    echo ""
    python web/app.py --demo
elif [ "$MODE" = "real" ]; then
    PORT=$(python -c "
import serial.tools.list_ports
ports = [p.device for p in serial.tools.list_ports.comports() if 'usbserial' in p.device or 'usbmodem' in p.device]
if ports: print(ports[0])
else: print('')
" 2>/dev/null)

    if [ -z "$PORT" ]; then
        echo "[ERROR] 未检测到手环串口"
        echo "  请用: ./scripts/demo.sh real /dev/tty.usbserial-XXXX"
        exit 1
    fi

    echo ""
    echo "  ╔══════════════════════════════════════╗"
    echo "  ║   FluxChi — 真实模式                  ║"
    echo "  ║   串口: $PORT"
    echo "  ║   浏览器打开: http://localhost:8000   ║"
    echo "  ╚══════════════════════════════════════╝"
    echo ""
    python web/app.py --port "$PORT"
else
    SERIAL_PORT="$MODE"
    echo "  Using port: $SERIAL_PORT"
    python web/app.py --port "$SERIAL_PORT"
fi
