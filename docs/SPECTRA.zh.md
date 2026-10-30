# Cortiq Spectra

*语言：[English](SPECTRA.md) · [Русский](SPECTRA.ru.md) · **中文***

`cortiq-spectra` 是双能 X 射线采集数据流式着色的可移植 CPU 实现。它是一个
确定性的、基于物理原理的参考处理器，而不是训练得到的模型，也不是医疗诊断
设备。

## 流处理架构

输入目录包含两个能量流和若干重复的校准行。处理开始之前，重复的 `gain_*` 和
`offset_*` 行被平均为四个探测器向量。对每一对行，处理器会：

1. 结合 gain/offset 对 high/low 计数做归一化，并保护分母；
2. 计算 `-ln` 衰减、low/high 比值、材料和置信度；
3. 把材料映射到确定性的 organic → mixed → metal 轴上，并应用有界的类双边
   3×3 滤波；
4. 在单行 halo 填满之后输出该行。到达 EOF 时复制边缘行，因此结果与分块边界
   无关。

`StreamingColorizer` 只保存三行相邻数据。`--chunk-rows` 的大小改变读取调度，
但不改变结果字节。`process_stream` API 和 CLI 为了写出 PNG 会物化所有行，因此
输出缓冲区本身自然是整帧大小；流式核心的状态仍以 halo 为界。

## 安全性与拒绝着色

不可信的像素不会得到看似合理的颜色。每个像素都保存一个拒绝位掩码（标志可以
组合）：

| 标志 | 原因 |
|---|---|
| `1` (`REFUSAL_INVALID_CALIBRATION`) | gain/offset 不是数值，或分母 `gain - offset` 不大于 epsilon |
| `2` (`REFUSAL_SATURATED`) | 原始计数等于 `u16::MAX` |
| `4` (`REFUSAL_NONFINITE`) | 归一化后的值不是有限数 |

任何非零标志都会传入 `StreamOutput.refusal`。这样的像素得到黑色 RGB，
material/confidence 为零，并被排除在滤波和 LUT 之外。`InvalidDenominator::Clip`
策略允许安全地计算中间值，但不会取消拒绝。在 `refusal.png` 中，被拒绝的像素
为白色，可信像素为黑色，以便在普通查看时就能看清可信边界。

## 校准配置：拟合与运行时

没有参考面板时，流输出使用 `ColorParameters` 调色板的物理基线。如果采集数据
旁边有 `reference/suitcase_colorized_declassify_bilateral_strong_rot.png`，
CLI 会：

1. 计算一次基线；
2. 把参考图从显示方向旋转回流顺序；
3. 只用可信的 RGB/参考像素对构建确定性的 `128³` RGB LUT。空的 bin 用可信 RGB
   的平均值填充。

LUT 是显示配置，而不是空间掩码：其中不包含绝对坐标、物体掩码或某个特定
行李箱的状态。运行时，LUT 在行滤波之后只应用于可信像素。

`save_profile` 写出可校验的 CMF 容器：

- `scanner.profile`：形状为 `[4, width]` 的 F32 张量（gain/offset high/low）；
- `scanner.color_lut`：可选的 U8 张量，形状为 `[size, size, size, 3]`；
- `spectra.v1` 溯源信息（provenance），包含 epsilon、拒绝策略和全部
  `ColorParameters`。

`load_profile` 检查 CMF 边界和哈希、溯源模式以及数值是否有限，然后返回可用的
`Calibration`。加载后的配置是自包含的：结果不依赖源代码中的常量。也可以运行
`cortiq verify scanner-profile.cmf` 来检查容器。

## CLI

需要 16 位单通道 TIFF 文件（文件名固定）：

| 文件 | 内容 |
|---|---|
| `data_high.tif`, `data_low.tif` | 尺寸相同的能量流 |
| `gain_high.tif`, `gain_low.tif` | 重复的 gain 行 |
| `offset_high.tif`, `offset_low.tif` | 重复的 offset 行 |
| `reference/suitcase_colorized_declassify_bilateral_strong_rot.png` | 用于拟合的可选显示参考图 |

在仓库根目录运行：

```sh
cargo run --release -p cortiq-spectra -- \
  --input-dir /path/to/capture \
  --chunk-rows 64 \
  --out-dir artifacts/spectra/chunk-64-release
```

参数：

- `--input-dir`——输入 TIFF 的目录（默认值是编译进二进制文件的开发者采集数据
  路径）；
- `--chunk-rows N`——正的分块大小，默认 `64`；
- `--out-dir DIR`——结果目录，默认 `artifacts/spectra/chunk-N`。

CLI 检查能量数据和校准数据的尺寸，保存配置，再带校验地重新加载，然后才处理
数据流。所选目录中会生成：

| 文件 | 含义 |
|---|---|
| `rgb.png` | 显示方向的最终 RGB（流帧旋转 90°） |
| `material.png` | 8 位材料灰度图 |
| `confidence.png` | 8 位置信度灰度图 |
| `refusal.png` | 被拒绝像素的白色掩码 |
| `scanner-profile.cmf` | CMF 校准配置，拟合时还包含 LUT |
| `summary.json` | 尺寸、分块、时间、拒绝数量、置信度和参考 MAE |

对于库形式的运行时，只需加载配置并把行传给 `process_stream`：

```rust,no_run
let calibration = cortiq_spectra::load_profile("scanner-profile.cmf")?;
let (rows, height) = cortiq_spectra::process_stream(
    calibration, &high, &low, 64,
)?;
assert_eq!(rows.len(), height);
```

## 测量结果与适用范围

验收用的采集数据尺寸为 576×3110（3,110 行），有 71,530 个被拒绝的像素。下面
的处理都是在同一份采集数据上的 release 运行；`processing` 时间不含 TIFF 解码，
`wall` 是完整运行时间：

| `chunk-rows` | processing, ms | wall, s |
|---:|---:|---:|
| 16 | 86.72 | 0.72 |
| 64 | 94.23 | 0.28 |
| 512 | 116.05 | 0.32 |

对分块 16/64/512，`rgb.png`、三张灰度 PNG 和 `scanner-profile.cmf` 都逐字节
一致。相对于提供的参考图，RGB MAE 为 `28.3267515556151/255`（不带 LUT 的基线
为 `33.47857568179111/255`）。MAE 是三个对齐的 uint8 通道的平均绝对误差；它
不是衡量材料真实性的指标。

整张图像都参与了拟合，因此 `28.33/255` 是单份采集数据的样本内（in-sample）
指标。单独的空间留出验证（200×100 的块，每第五条带不参与拟合）得到
`30.732184250584318/255`；这是泛化诊断，而不是对新扫描仪的独立验证。不能把
这份行李箱采集数据得到的 LUT 视为通用模型：新的扫描仪需要用它自己具有代表性
的参考面板训练出配置。
