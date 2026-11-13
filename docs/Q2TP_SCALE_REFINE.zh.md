# q2tp：通道缩放与码值精修

2 比特 `q2tp` 编码的改进配方。文件大小、格式和推理速度都不变：全部在 `cortiq quantize-gptq` 阶段完成。
在 MiniCPM5-2B 上测得，2026 年 11 月 13 日。

[English](Q2TP_SCALE_REFINE.md) · [Русская версия](Q2TP_SCALE_REFINE.ru.md)

## 结果

同一个 1 314 231 679 字节的文件，同一组 Hessian（WikiText-2 train 的 32 768 个 token），困惑度：

| 文本（每窗 512 token） | 之前 | 之后 | 变化 |
|---|---|---|---|
| WikiText-2 test（160 窗） | 28.431 | 26.426 | −7.05 % ± 1.20 |
| WikiText-2 validation（160 窗） | 28.337 | 26.384 | −6.89 % ± 1.43 |
| 英文文档（64 窗） | 69.185 | 60.445 | −12.6 % ± 2.5 |
| Rust 源码（64 窗） | 27.515 | 23.706 | −13.8 % ± 3.0 |
| WikiText-2 test，原生文件（12 窗） | 26.106 | 23.898 | −8.5 % |

区间为同一批窗口上成对差值的 95 % 区间。参数在 test 上选定；validation、文档和代码没有参与选择。
64 窗和 160 窗的行通过每个文件的 q8_2f 副本评分（见最后一节），12 窗的行直接在 q2tp 文件上评分。

## 原理

在常规 GPTQ 之上加两步，都不占用比特。

### 1. 折叠进归一化层的通道缩放

2 比特编码对每组 32 个相邻输入通道只有一个步长。组内通道的幅度相差很大时，小的通道会丢失。量化之前，
每个输入通道 `j` 乘以自己的系数，其倒数放进该层前面 RMSNorm 的增益里：

```
t_j = σ_j^0.75 / rms_j^0.25          σ_j   — 张量组所有行上第 j 列的 RMS
                                     rms_j — 校准数据上第 j 个输入的 RMS（√H_jj）
W' = W · diag(1/t)                   送入量化器的权重
γ' = γ ⊙ t                           写入输出文件的归一化增益
H' = diag(t) · H · diag(t)           新基下的 Hessian
```

`t` 归一化到几何平均为 1，并限制在 `[1e-3, 1e3]`；死通道取 `t = 1`。量化之前模型在数学上不变：
`W'·(t⊙x) = W·x`。

读取同一输入的张量共用一个缩放：

| 张量 | 接收 `t` 的归一化层 |
|---|---|
| `mlp.gate_proj` + `mlp.up_proj` | `post_attention_layernorm` |
| `self_attn.q_proj` + `k_proj` + `v_proj`（`--channel-scale-qkv`） | `input_layernorm` |

指数很关键：取 0.5 时收益消失。取 1.0（ridge 为 1.0）时结果不比 0.75 差：26.72 对 26.99。

### 2. GPTQ 之后的码值精修

GPTQ 从左到右为每个权重只选一次电平。之后，如果把单个权重移到相邻电平能降低该层的输出误差，就移动它。
缩放阶梯和各组的档位保持不变。

```
r   = Ŵ_i − W_i                      该行的残差
H_f = H + 0.3 · mean(diag H) · I
g   = H_f · r
把权重 j 移动 Δ = ±s_组，r·H_f·rᵀ 的变化为   2Δ·g_j + Δ²·H_f[j,j]
```

每一轮对一行只做收益最大的一次移动，然后 `g += Δ·H_f[:, j]`。最多 256 轮；没有可用移动时该行提前停止。
之后照常做 debias（该行 `lo` 的平移）。

ridge 已实测：0.1 和 0.3 结果相同，1.0 较差，3.0 比不精修还差。

## 命令

```bash
cortiq quantize-gptq MODEL.f16.cmf --codec q2tp \
  --calib CALIB.txt --calib-model MODEL.q8_2f.cmf \
  --tokens 32768 --window 2048 --lambda 0.01 \
  --tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f \
  --channel-scale 0.75 --channel-scale-qkv \
  --refine 256 --refine-ridge 0.3 \
  --output MODEL.q2tp.cmf
```

其余保持默认：`placed` 阶梯，debias 开启。校准 token 不少于 32 768。

## 在新模型上要检查什么

1. 折叠是恒等变换。同样的命令加上 `CMF_CHSCALE_FP=1`，会把所有线性层以 f16 写出，不量化，但带上折叠的缩放。
   它的困惑度应与不带 `--channel-scale` 的同样运行一致（MiniCPM5：16.903 和 16.902）。若不一致，说明该架构的
   归一化增益形式不同（例如 `1 + γ`），或者还有别的模块读取归一化的输出。
2. 折叠的数量。日志里每个被折叠的组有一行 `channel scale <归一化层名>`。名称或成员不匹配的组会被静默跳过：
   GDN 层（`linear_attn.*`）、MoE 专家。
3. 在同一组 Hessian 上与基线成对比较：同样的命令去掉四个新参数。看模型的困惑度，不要看层误差：在 2 比特下
   层误差不能预测困惑度。
4. 校准数据之外的文本，以及贪心生成。

## 限制

- 只在一个模型上测过（MiniCPM5-2B，稠密的 Llama 类结构，GQA 16/2），校准数据为 WikiText。
- 缩放不作用于 `down_proj` 和 `o_proj` 的输入，也不作用于 MoE 专家和 GDN 层。
- 新参数涉及的组在 CPU 上做折叠和扫描。其余的组留在显卡上，但这条混合路径没有在显卡上运行过：在 Mac 上全部走 CPU。
- 已有 `--hessians` 缓存时，所有的组都走 CPU。
- 精确的最小二乘增益（`--channel-refit`，NVG 编码器的输入通道增益）没有额外收益：−0.23 %。不属于本配方。
- 第 7 层的 cortiq 规则（`--q2tp-rule-tensor`）在这条流水线里没有帮助：−0.03 % ± 0.43。
- `--refine-q4tp` 在 q4tp 张量的 16 级网格上做同样的精修；不属于本配方。

## 代码与工具

`crates/cortiq-cli/src/gptq.rs`：`chscale_vector`、`chscale_norm_for`、`q2tp_refine_codes`、
`q4tp_refine_codes`、`channel_refit_gain`。

在 CPU 上给 q2tp/q4tp 文件评分很慢（2B 模型约 18 tok/s）。`cortiq requant FILE --quant tiled-q8_2f
--output COPY` 会写出一个 q8_2f 副本，评分快 7 倍，困惑度偏高 0.4–0.7 %，对所有方案一致。
