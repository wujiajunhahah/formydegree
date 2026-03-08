# EMG 放松小游戏（独立窗口）

这个小游戏是一个单独脚本，不会和 `app.py` 一起跑。它使用你训练好的手势模型，
跟随呼吸节奏完成目标动作来得分。

## 快速启动（复制一行一行执行）

```bash
cd /Users/wujiajun/Downloads/FluxChi/reference/harward-gesture
source .venv/bin/activate
python3 relax_game.py --port /dev/cu.usbserial-0001 --baud 921600 --fs 1000
```

如果端口不是 `usbserial-0001`，先查端口：

```bash
ls /dev/cu.usbserial*
```

## 手势设置

脚本默认把“模型里的前两个手势”作为：
- 吸气（Inhale）
- 呼气（Exhale）

你可以手动指定：

```bash
python3 relax_game.py --port /dev/cu.usbserial-0001 --inhale fist --exhale open
```

## 玩法说明

- 屏幕中心圆环会按节奏变大/变小。
- 看到 “Target Gesture” 后，在该阶段做对应手势。
- 命中会加分并提升 Calm（放松条）。
- 误触会降低 Calm。

## 科学依据（简要）

- **节律呼吸**：4–6 次/分钟的慢呼吸可提升心率变异性（HRV）和副交感活性，
  有助于放松与压力缓解。
- **视觉节奏引导**：平滑的视觉节律可促进呼吸同步（entrainment），让人更容易
  进入稳定节奏。
- **目标反馈**：明确目标 + 即时反馈的轻量游戏化，有助于坚持练习与注意力集中。

> 以上为放松训练相关的常见研究结论，非医疗建议。
