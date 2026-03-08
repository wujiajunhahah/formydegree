# EMG FPS 训练场（HTML 独立游戏）

这个模式是“手势校准 + 自动训练 + HTML 游戏”的完整流程：

1. 运行 `fps_game.py` → 引导你录制手势 → 自动训练模型。
2. 打开 `fps_game.html` → 用手势控制射击/瞄准/切枪。

## 1. 启动服务（包含校准+训练）

```bash
cd /Users/wujiajun/Downloads/FluxChi/reference/harward-gesture
source .venv/bin/activate
pip install -r requirements.txt

python3 game/fps_game.py --port /dev/cu.usbserial-0001 --profile player1 --calibrate
```

一键启动脚本（等价上面命令）：

```bash
./game/start_fps_game.sh /dev/cu.usbserial-0001 player1 --calibrate
```

如果你已经在 `game` 目录里，只需要：

```bash
./start_fps_game.sh /dev/cu.usbserial-0001 player1 --calibrate
```

首次运行会按默认标签采集样本（shoot/left/right/up/down），并显示进度条。
如需自定义标签，请加参数：

```bash
python3 game/fps_game.py --port /dev/cu.usbserial-0001 --profile player1 --calibrate \
  --shoot-label fire --left-label swipe_left --right-label swipe_right --up-label raise --down-label drop
```
校准完成后会自动训练模型并启动 WebSocket（`ws://127.0.0.1:8765`）。

下次直接运行即可（不重新校准）：

```bash
python3 game/fps_game.py --port /dev/cu.usbserial-0001 --profile player1
```

一键启动脚本：

```bash
./game/start_fps_game.sh /dev/cu.usbserial-0001 player1
```

## 2. 打开游戏

双击或用浏览器打开：

`game/fps_game.html`

如果想玩 Doom 风格版本：

`game/doom_clone/emg_doom.html`

## 3. 控制逻辑

- `look_left` / `look_right`：左右转头
- `look_up` / `look_down`：上下转头
- `shoot`：射击
- `switch`：切枪（可选）
- `reload`：装弹（可选）

若没配置 `reload`，弹夹会自动装满。

## 4. 怎么玩（简版）

1. 运行校准命令后，看到进度条时做对应手势，保持 2-3 秒。
2. 校准结束会自动训练，然后打印 WebSocket 地址。
3. 打开 `game/fps_game.html`，出现靶子后用手势射击。
4. 左/右手势移动准星，射击手势开火。

## 5. Doom 版玩法

1. 先跑 `fps_game.py` 或 `start_fps_game.sh`，保持 WebSocket 在线。
2. 推荐用本地小服务器打开（避免浏览器拦截）：

```bash
cd game/doom_clone
./start_emg_doom.sh 8000
```

然后访问：`http://localhost:8000/emg_doom.html`

3. 射击手势 = 开火，左右/上下手势 = 视角转动。
4. 键盘仍可用：W/A/S/D 移动，鼠标也可辅助视角。

## 4. 多用户/多手势

每个用户用不同 profile：

```bash
python3 game/fps_game.py --port /dev/cu.usbserial-0001 --profile alice --calibrate
python3 game/fps_game.py --port /dev/cu.usbserial-0001 --profile bob --calibrate
```

模型和手势配置会保存在：

`profiles/<profile>/config.json`
`profiles/<profile>/data/`
`profiles/<profile>/model/`
