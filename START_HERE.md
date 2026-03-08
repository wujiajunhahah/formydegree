快速启动（复制下面每一行到终端执行，不要带任何符号）

cd /Users/wujiajun/Downloads/FluxChi/reference/harward-gesture
source .venv/bin/activate
python3 app.py --port /dev/cu.usbserial-0001 --baud 921600 --fs 1000

如果端口不是 usbserial-0001，请先查端口：
ls /dev/cu.usbserial*
