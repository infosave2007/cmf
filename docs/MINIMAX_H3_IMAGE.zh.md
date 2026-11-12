# MiniMax-H3-Image

*语言：[English](MINIMAX_H3_IMAGE.md) · [Русский](MINIMAX_H3_IMAGE.ru.md) · **中文***

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) 用六个采样步生成和编辑 4 到 16
百万像素的图片。cortiq 从一个 24.6 GB 的 CMF 文件运行它：其中有扩散 Transformer（200 亿参数，4 位）、
带视觉塔的 Qwen3-VL 提示词编码器（4 位）和图像 VAE（8 位）。它在 NVIDIA 显卡上走原生 CUDA 后端，也可以
走 Vulkan，在 Apple 芯片上走 Metal；三种后端下，图片从 Transformer 的第一个块到最后一个块都留在 GPU 上。
在 CUDA 和 Vulkan 上，显存上限让它可以在小显卡上运行。

```bash
cargo install cortiq-cli --features cuda
# 文生图
CMF_GPU=cuda cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
# 图片编辑：提示词里的 <Picture 1> 就是 --image 给出的文件
CMF_GPU=cuda cortiq imagine minimax-h3-image-q4tp.cmf --image photo.png --prompt @edit.txt --out edit.png
# 6 GB 显卡
CMF_GPU=cuda CMF_GPU_VRAM_MB=6000 cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
# Vulkan 和 Metal：默认后端（显存上限在 Vulkan 上同样有效）
cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
```

文件在 Hugging Face：[infosave/MiniMax-H3-Image-cmf](https://huggingface.co/infosave/MiniMax-H3-Image-cmf)。模型采用 MiniMax-H3-Image Community License，
该许可限制了使用地区，详见页面。

## 文生图

| 选项 | 含义 |
|---|---|
| `--prompt TEXT` 或 `--prompt @FILE` | 描述；长描述放在文件里更方便 |
| `--width`、`--height` | 32 的倍数，默认 2048 x 2048 |
| `--min-area PIXELS` | 模型运行的最小画布，默认 4194304（2048 x 2048）；更小的尺寸会在同宽高比的这种画布上生成再缩小，与参考服务器一致。`--min-area 1024` 关闭此行为 |
| `--seed N` | 默认 42；`--num-images K` 用一次提示词编码渲染种子 N .. N + K - 1 |
| `--steps N` | 少于模型的六步：同一 sigma 网格，步幅更长 |
| `--out FILE` | 按扩展名输出 PNG、JPEG 或 PPM |

模型经过蒸馏：在自己的 sigma 网格上走六步，没有 classifier-free guidance，也没有负面提示词。工作范围约为
4 百万像素（2048 x 2048）到 20 百万像素，16 百万像素（4096 x 4096）附近效果最好。模型需要对整张图片的
详细英文描述。作者用提示词扩展器（H3-Image-Context-IR API 或本地的 MiniMax-H3-Image-PE 模型）把简短请求
扩展成这种描述；cortiq 不包含扩展器，描述需要自己写，或用任意语言模型扩展。

## 图片编辑

`--image FILE` 把运行变成编辑；可以重复给出多张图片。提示词按给出的顺序称它们为 `<Picture 1>`、
`<Picture 2>` 等，`<Picture 1>` 是被编辑的图片。不给 `--width` 和 `--height` 时，结果取第一张图片的尺寸，
短边限制在 1024 到 2048 像素之间。

提示词编码器通过 Qwen3-VL 视觉塔在短边 1024 像素下看到每张图片，Transformer 还会把每张图片的 VAE latent
作为序列中目标之前的行接收。因此在 2048 x 2048 上编辑一张 1024 x 1024 的图片，要处理 4096 个目标行、
1024 个参考行和提示词。

模型自己的编辑提示词是任务说明：列出每个对象，并说明保留什么（`The man on the left in <Picture 1> is
<Subject 1>. ... The target image shows ...`）。简短指令也可以用，但保留的内容更少：提示词没有提到的文字
往往会改变。

## 小显卡

`CMF_GPU_VRAM_MB` 是进程在显卡上占用内存的上限。放不进上限的 Transformer 块不留在显卡上：它们的权重
每一步都流经显卡。随图片增大的部分很小：面板按行分块计算，注意力不保存分数矩阵。

2048 x 2048，CUDA，NVIDIA RTX 2000 Ada, 16 GB：

| 上限 | 总计，秒 | 每步，秒 | 显存峰值，MiB |
|---|---:|---:|---:|
| 无 | 50.3 | 7.38 | 12 654 |
| `CMF_GPU_VRAM_MB=8000` | 53.5 | 8.05 | 7 634 |
| `CMF_GPU_VRAM_MB=6000` | 54.4 | 8.18 | 5 720 |
| `CMF_GPU_VRAM_MB=4000` | 60.2 | 9.15 | 3 710 |

4 GB 上限下得到的是同一张图片：与无上限运行相比 PSNR 为 49.6 dB。

在 Vulkan 上，同一个变量是权重的预算。放得下的块留在显卡上；其余的块每一步分批流经显卡，每批不超过
1 GB。2048 x 2048，Vulkan，NVIDIA RTX A5000, 24 GB：

| 上限 | 总计，秒 | 每步，秒 | 显存峰值，MiB |
|---|---:|---:|---:|
| 无 | 32.8 | 4.02 | 12 673 |
| `CMF_GPU_VRAM_MB=12000` | 32.2 | 4.02 | 11 962 |
| `CMF_GPU_VRAM_MB=8000` | 38.1 | 5.17 | 8 560 |
| `CMF_GPU_VRAM_MB=6000` | 41.6 | 5.71 | 6 647 |
| `CMF_GPU_VRAM_MB=4000` | 46.8 | 6.64 | 4 754 |

任何上限下的图片都与无上限时逐位相同。显存峰值超出上限不到 0.8 GB。

## 速度

NVIDIA RTX 2000 Ada, 16 GB，一张图片，种子 42，单次运行（2048 x 2048 为三次的中位数）。

| 运行 | 提示词，秒 | 六步，秒 | 每步，秒 | 解码，秒 | 总计，秒 | 显存峰值，MiB |
|---|---:|---:|---:|---:|---:|---:|
| CUDA，1024 x 1024（`--min-area 1024`） | 2.4 | 11.7 | 1.95 | 0.5 | **16.1** | 12 116 |
| CUDA，2048 x 2048 | 2.6 | 44.3 | 7.38 | 1.7 | **50.3** | 12 654 |
| CUDA，4096 x 4096 | 9.2 | 264.5 | 43.90 | 8.3 | **283.7** | 13 414 |
| CUDA，2048 x 2048 编辑，一张 1024 x 1024 参考图 | 9.8 | 67.9 | 11.37 | 1.7 | **80.8** | 12 854 |

NVIDIA RTX A5000, 24 GB，同一个文件在两个后端上，单次运行。

| 运行 | CUDA：每步，秒 | CUDA：总计，秒 | Vulkan：每步，秒 | Vulkan：总计，秒 | Vulkan：显存峰值，MiB |
|---|---:|---:|---:|---:|---:|
| 1024 x 1024（`--min-area 1024`） | 0.90 | 10.8 | 1.15 | **16.2** | 12 673 |
| 2048 x 2048 | 3.31 | 24.8 | 4.01 | **33.7** | 12 673 |
| 4096 x 4096 | 21.69 | 138.6 | 21.59 | **142.3** | 13 759 |
| 1024 x 1024 编辑，一张 1024 x 1024 参考图 | 2.34 | 25.4 | 2.85 | **30.6** | 13 824 |
| 2048 x 2048 编辑，一张 1024 x 1024 参考图 | 5.49 | 45.1 | 6.16 | **52.1** | 13 824 |

MacBook Air M4, 24 GB，Metal，单次运行。

| 运行 | 提示词，秒 | 六步，秒 | 每步，秒 | 解码，秒 | 总计，秒 | 进程内存峰值，MiB |
|---|---:|---:|---:|---:|---:|---:|
| 1024 x 1024（`--min-area 1024`） | 12.0 | 119.3 | 20.23 | 2.9 | **138.8** | 10 654 |
| 2048 x 2048 | 12.3 | 537.2 | 89.96 | 12.7 | **565.8** | 8 551 |
| 1024 x 1024 编辑，一张 1024 x 1024 参考图 | 55.0 | 366.2 | 61.17 | 2.9 | **431.6** | 12 604 |

MacBook Air 没有风扇，机身变热后会降速：1024 x 1024 的一步在冷机上是 18 秒，热机上是 20 到 22 秒。

## 在 CUDA 上如何运行

- **Transformer 是显卡上的一条链。** 残差流每步上传一次、取回一次。其间每个块都在显卡上完成：带 adaLN
  调制的 RMS 归一化转为 f16、qkv 乘积、按头的 q/k 归一化与旋转、注意力、输出投影、带门控的残差，然后是
  SwiGLU MLP。乘积由 cuBLAS 在张量核心上计算（f16 操作数，f32 累加）。
- **注意力不产生任何中间矩阵。** 一个 warp 负责一个头的 16 行查询，每次处理 16 个键：两次张量核心乘积
  （`mma.sync`），softmax 在寄存器里完成。键和值按线程读取的顺序存放。逐段关闭内核的测量表明，时间花在
  操作数的加载上，而不是乘积上；重新排列后耗时减半。
- **权重保持在自己的编码里。** 一个 q4tp 块在显卡上占 222 MB，只在它的乘积运行期间展开成稠密的 f16
  平面；整个堆栈共用一个平面缓冲区。
- **提示词编码器**（Qwen3-VL-32B 的 50 层）以同样方式运行，使用因果注意力内核。每个权重只用一次，所以
  不保留：编码器占用的平面不超过一层。
- **VAE** 把 256 像素的瓦片分批通过同一条链解码；编辑的参考图片由一条卷积链编码（im2col 面板，每个卷积
  一次乘积）。

## 在 Vulkan 和 Metal 上

同样的链，按每个后端自己的矩阵单元编写。

- **Vulkan。** 乘积直接在显卡的 16 x 16 矩阵单元上读取 4 位张量（`VK_KHR_cooperative_matrix`，f16 操作数，
  f32 累加）：在 RTX A5000 上达到 50 到 58 TFLOP/s，这张卡的纯矩阵单元上限是 103。注意力是直接读取 q/k/v
  面板的 flash 内核。提示词编码器以同样方式运行，使用因果的分组注意力；VAE 解码器每批处理最多 41 个瓦片；
  编辑的参考图片由卷积编码，卷积在乘积内部收集各个抽头，不需要 im2col 面板。一个阶段的权重以自己的编码
  留在显卡上，或者在上限下流经显卡。
- **Metal。** Transformer 的乘积在为 8 x 8 simdgroup 矩阵准备数据时解量化 4 位瓦片，并从文件映射中原地读取
  权重。提示词编码器的权重逐层经过一个缓冲区，VAE 解码器的权重放在自己的缓冲区里：在内存小于文件的 Mac
  上，原地读取会让 GPU 先把文件的一个 13 GB 窗口换入内存。

Vulkan 的链需要 16 x 16 半精度协作矩阵和 32 线程的子组，NVIDIA RTX 显卡具备这些条件。没有这些条件时，
或者设置了 `CMF_H3_WGPU=0`、`CMF_H3_METAL=0` 时，流水线使用 MiniMax-H3 视频路径的逐算子内核。

## 打包

```bash
cortiq animate-pack --out minimax-h3-image-q4tp.cmf \
  --in mmh3-turbo-fl2va-q4tp.cmf \
  --dit minimax_h3_image_pruned_bf16.safetensors \
  --image-vae minimax_h3_image_vae_fp16.safetensors --image-vae-quant q8_2f \
  --drop avae --quant q4tp
```

`--dit` 是 ComfyUI 的 pruned 检查点（`Comfy-Org/MiniMax-H3-Image`）：其中 adaLN 已经折叠成秩 8 的曲线，
输出层是六个按步的头组成的头库。提示词编码器与 MiniMax-H3 相同，所以 `--in` 从 MiniMax-H3 容器中带入它和
视觉塔；`--te` 则从 bf16 源文件打包。

## 做过的检查

- 链与逐算子路径（每个乘积是单独的 f32 GEMM）在 2048 x 2048 编辑上对比：图片间 PSNR 54.0 dB。
- RTX A5000 上 Vulkan 与 CUDA 对比，同一种子：1024 x 1024 为 45.9 dB，2048 x 2048 为 44.5 dB，
  4096 x 4096 为 40.3 dB，1024 x 1024 编辑为 43.8 dB。Metal 与这些 CUDA 图片对比：1024 x 1024 为 47.2 dB，
  2048 x 2048 为 43.7 dB。
- CUDA 上 4 GB 上限与无上限：49.6 dB。Vulkan 上 12、8、6 和 4 GB 上限下的图片与无上限时完全相同。
- Vulkan 和 Metal 上提示词编码器的输出与逐算子路径对比：相差不超过 4e-4（相对 RMS）。
- VAE 解码器的链与逐算子路径：CUDA 上 47.5 dB，Vulkan 上 65.6 dB，Metal 上 66.3 dB；与 f16 权重：48.5 dB。
- `CMF_H3_SAT=1` 统计面板中达到饱和值的 f16 数：文生图和编辑运行中都没有。
- 同一 Transformer 下换用 8 位提示词编码器：条件状态相差 3 %（相对 RMS），图片相差 17.7 到 25.4 dB，
  细节水平相同。文件保留 4 位。
- 未检查：没有在同一张显卡上运行参考实现，因此没有与它的逐像素对比。

## 开关

| 变量 | 作用 |
|---|---|
| `CMF_GPU=cuda` | 原生 CUDA 后端（用 `--features cuda` 构建） |
| `CMF_GPU_VRAM_MB=N` | 显存上限（CUDA 和 Vulkan） |
| `CMF_CUDA_PROF=1` | 链的各阶段耗时（同步，较慢） |
| `CMF_H3_SAT=1` | 统计链中面板的饱和 f16 数和最大幅值 |
| `CMF_MMH3_CHAIN=0`、`CMF_TE_CHAIN=0`、`CMF_VAE3D_CHAIN=0` | Transformer、编码器、VAE 走逐算子路径 |
| `CMF_H3_FLASH=0` | 注意力按块用两次批量乘积计算，不用 flash 内核 |
| `CMF_H3_WGPU=0`、`CMF_H3_METAL=0` | Vulkan、Metal 走逐算子路径 |
| `CMF_LATENT_OUT=FILE`、`CMF_INIT_LATENT=FILE` | 写出最终 latent；从给定噪声开始 |
| `CMF_TE_DUMP=FILE`、`CMF_TE_LOAD=FILE`、`CMF_TE_FROM=FILE.cmf` | 写出提示词条件；使用另一次运行的条件；从另一个容器取编码器 |

## 限制

- **在 2048 x 2048 上编辑细节密集、对比强烈的图片，结果会过度处理**（过饱和、色调分离）。同样的编辑在
  1024 x 1024 上（`--width 1024 --height 1024 --min-area 1024`）是干净的，较柔和的图片在 2048 x 2048 上
  的编辑也是干净的。8 位编码器、8 位 Transformer 和 f32 逐算子路径给出相同结果，因此排除了编码和链；
  原因尚未找到。
- Vulkan 的链需要 NVIDIA RTX 显卡（16 x 16 半精度协作矩阵，32 线程的子组）。其他显卡走逐算子路径，慢几倍。
- 在 24 GB 的 Mac 上，文件比内存大：每个阶段都要从磁盘重新读取自己的部分。
- 注意力内核需要支持 `mma.sync` 的张量核心（compute capability 8.0 及以上）；更老的显卡改用两次批量乘积。
- 语言模型 turbo 文件的远场（O(1)）注意力在这里不适用：在 2048 x 2048 上用模型自己的操作数评估，
  它会让注意力输出改变 12 % 到 65 %。
