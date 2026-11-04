# 在 cortiq 上运行 Kandinsky S2V

*语言：[English](KANDINSKY_S2V.md) · [Русский](KANDINSKY_S2V.ru.md) · **中文***

[Kandinsky S2V 1.0 Pro-distill](https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Pro-distill-Diffusers)
根据一张参考帧和一段语音生成数字人视频。动起来的是整个人物，而不只是头部。音频只作为条件：
模型不生成声音，`cortiq` 会把你的音轨写入输出文件。

## 运行

```bash
cargo install cortiq-cli
hf download infosave/Kandinsky-S2V-1.0-Pro-distill-cmf kandinsky-s2v-pro-q4tp.cmf --local-dir .
cortiq s2v --model kandinsky-s2v-pro-q4tp.cmf --image reference.png --audio speech.wav --out avatar.avi
```

输出为 AVI 封装的 MJPEG 和 PCM 音频，25 fps，无需安装其他软件。如需 mp4：

```bash
ffmpeg -i avatar.avi -c:v libx264 -crf 18 -c:a aac avatar.mp4
```

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--image` | 必填 | 参考帧（PNG、JPEG） |
| `--audio` | 必填 | 语音音轨：直接读取 WAV，其他格式通过 `ffmpeg` |
| `--out` | `avatar.avi` | 输出文件 |
| `--prompt` | 参考提示词 | 描述动作；自定义提示词会加载 7B 文本编码器 |
| `--seed` | 5 | 噪声种子 |
| `--steps` | 16 | 每个分块的 pi-Flow 段数 |
| `--chunks` | 覆盖音轨所需的数量 | 每块 5 秒，视频按音轨长度裁剪 |
| `--height`、`--width` | 最接近图像比例的 512 档 | 512×512、768×512 或 512×768 |
| `--quality` | 92 | 帧的 JPEG 质量 |

## 流程

```
语音 → 梅尔谱 → 音频塔 → 25 token/秒 ─┐
提示词 → Thinker + CLIP               ├─ DiT，每块 16 段 ─→ VAE 解码器 ─→ 帧
（默认提示词已存入文件）              │     ↑ 上一分块的 FramePack 记忆
参考帧 → VAE 编码器 ──────────────────┘
```

- **DiT。** 60 个 4096 维视觉块、4 个提示词块和 4 个语音块，共 21B 参数。512² 档的一个分块有
  36736 个 token：三个时间尺度的 FramePack 记忆、参考帧和 32 个潜在帧。
- **pi-Flow。** 一次前向在一段原始时间上预测 5 个 x0，然后对其积分 ODE，不再评估网络。引导已蒸馏进权重。
- **VAE。** HunyuanVideo 的因果 3-D VAE，按该类的默认方式在时间上分块：每 3 个潜在帧取 5 帧，接缝处混合 4 帧。

## 运行方式

权重以 f16 平面形式放不进 32 GB 显卡（42 GB）。容器中的量化字节原样留在显卡上，每个块运行前才展开它的平面，
因此 q4tp 的 DiT 约需 14 GB 显存。在 Metal 上，权重留在文件映射中，由 GEMM 读取时解量化；每次只映射一个块，因为驱动会把映射的全部长度计入工作集上限。
两个后端的 VAE 都常驻设备。设备拒绝时 `cortiq s2v` 回退到 CPU，在 36736 个 token 下每段需要数小时。

L40S 用 5.6 分钟渲染 5 秒 512² 视频。MacBook Air M4（10 核 GPU、24 GB、无风扇）每步 955–1047 秒，同样的片段需 4.45 小时，
统一内存峰值 14.6 GB。

## 一致性

`scripts/s2v/oracle.py` 按阶段导出参考结果（整个流程在 32 GB 显卡上无法放入一个进程），`cortiq s2v-check` 逐阶段比较：

```bash
python scripts/s2v/oracle.py enc --image i.png --audio a.wav --chunks 1 --out enc.safetensors
cortiq s2v-check --model kandinsky-s2v-pro-q8.cmf --stage enc --oracle enc.safetensors
```

全部测量记录见 `scripts/s2v/JOURNAL_S2V_2026-11-03.md`。

## 负责任地使用

该模型可以用一张照片和不一定属于本人的声音让真人动起来。请仅在获得本人同意时使用其参考帧，
不要把结果当作任何人的真实录像。
