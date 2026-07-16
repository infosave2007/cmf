# CMF v2 — 格式规范

*语言：[English](CMF_V2_SPEC.md) · [Русский](CMF_V2_SPEC.ru.md) · **中文***

**Cortiq Model Format** — 一个文件即可承载稀疏、任务路由推理所需的一切：
量化权重、分词器、逐任务掩码、预计算的稀疏索引 —— 以及独一无二的、
共享同一骨干的**技能集群**（Patent 15）。

> 规范来源：本文档。参考实现：Rust 读取器/运行时
> （`crates/cortiq-core`、`crates/cortiq-engine`）、Python 写入器
> （`converter/`），以及一个无依赖的 Python 读取器
> （`python/cmf_reader.py`，仅需 numpy）。

三项要求，按优先级排序：

1. **正确。** 没有静默损坏模式：严格的魔数、版本、
   `required_features`、每个区段的边界检查、每个张量的 64 位哈希。
   文件要么有效，要么 open() 返回错误 —— 不存在
   第三种状态。
2. **快速。** 权重区段按页对齐以便 mmap，每个张量
   都按 64 字节对齐（零拷贝 SIMD），张量目录为二进制格式 ——
   无需解析即可读取。冷（被掩掉的）权重不占用 RSS。
3. **紧凑。** 掩码按位打包（每个神经元 1 比特），权重采用
   q4/q8/可变位宽，整个文件由一个 128 字节的
   信封寻址。

和谐并非来自特性数量，而是来自**单一正典**：每个层级只有一种
布局（信封、目录、量化块、掩码），在各自领域重叠之处
（张量目录、量化布局、`hash64`）与经过验证的 `.vmfc` v2 格式逐字节
兼容。永远不存在同一事物的两种
定义。

物理基础（VMF）：模型是一个真空凝聚态 𝒲；技能是它在临界密度之上的
规则核；任务掩码在不改变权重的情况下选取一个活跃
子集。格式承载了该物理的推论（双场 𝒲×θ 量化、Born 重要性、临界掩码
阈值）—— 但**仅限那些经测量确认的部分**。

---

## 1. 信封（固定 128 字节）

所有整数均为小端序。

```
[0x00 : 0x04]  magic              = b"CMF\x01"  (4 bytes)
[0x04 : 0x08]  version            : u32 = 2
[0x08 : 0x0C]  flags              : u32 (reserved, 0)
[0x0C : 0x10]  required_features  : u32 (bitmask, §1.1)
[0x10 : 0x18]  header_off         : u64 (= 128)
[0x18 : 0x20]  header_len         : u64   — JSON header (§2)
[0x20 : 0x28]  dir_off            : u64   — tensor directory (§3)
[0x28 : 0x30]  dir_len            : u64
[0x30 : 0x38]  data_off           : u64   — weight blob; multiple of 4096 (§4)
[0x38 : 0x40]  data_len           : u64
[0x40 : 0x48]  masks_off          : u64   — masks section (§5); 0 = absent
[0x48 : 0x50]  masks_len          : u64
[0x50 : 0x58]  vocab_off          : u64   — tokenizer (§6); 0 = absent
[0x58 : 0x60]  vocab_len          : u64
[0x60 : 0x68]  index_off          : u64   — sparse index (§7); 0 = absent
[0x68 : 0x70]  index_len          : u64
[0x70 : 0x80]  reserved           : 16 bytes (§8.1: header/dir hashes)
```

磁盘上的区段顺序：信封 → 头部 JSON → 目录 → **权重
blob（对齐到 4096）** → 掩码 → 词表 → 稀疏索引。读取器必须
仅通过信封来寻址区段，绝不假设其顺序。

### 1.1 `required_features`

读取器不认识的某一位 → `UnsupportedFeature` 错误（快速失败；
不做"尽力读取"）。

| bit | name           | 含义 |
|-----|----------------|---------|
| 0   | `TENSOR_DIR`   | 二进制张量目录（v2 中始终置位） |
| 1   | `BINARY_MASKS` | 存在掩码区段（§5） |
| 2   | `QUANT_2F`     | 目录包含 `q8_2f`/`vbit` 张量（双场 𝒲×θ 量化） |
| 3   | `DELTA_MASKS`  | 保留：来自父级的 XOR 掩码增量 |
| 4   | `HOT_PACKS`    | 保留：物化的稠密切片 |

未知的**头部 JSON** 字段被忽略（增量式演进）；
破坏性变更只通过特性位或 `version` 递增来引入。

### 1.2 校验规则（规范性）

在以下情况下，读取器必须返回错误（而非默认值、也非警告）：

- magic ≠ `CMF\x01` → `InvalidMagic`；
- `version` ≠ 2 → `UnsupportedVersion`（v1 已死：不存在真实的 v1
  文件，不会启动任何支持计划）；
- 置位了未知的 `required_features` 位 → `UnsupportedFeature`；
- 任何区段越过 EOF、`data_off` 不是 4096 的倍数、某张量的
  `off + nbytes` 超过 `data_len` → `Bounds`；
- 张量名非 UTF-8、dtype 未知、`ndim > 6` → `Parse`。

张量哈希校验按需进行（`cortiq verify`、加载器标志），
并非每次打开都做：mmap 页面按需惰性读取。

## 2. 头部 JSON

UTF-8 JSON，不对齐。机器关键数据存放于二进制区段；
JSON 承载架构与来源信息 —— 供人阅读的部分。

```jsonc
{
  "format": "cmf",
  "version": 2,
  "arch": {
    "arch_name": "qwen3.5",
    "hidden_size": 5120, "intermediate_size": 17408,
    "num_layers": 64, "num_attention_heads": 24, "num_kv_heads": 4,
    "head_dim": 256, "vocab_size": 248320,
    "layer_types": ["LinearAttention", "...", "FullAttention"],
    "rms_norm_eps": 1e-6,
    "norm_style": "qwen",            // "qwen": x̂·w | "gemma": x̂·(1+w)
    "rope_theta": 1000000.0,
    "tie_word_embeddings": false,
    "max_position_embeddings": 262144,
    "linear_conv_kernel_dim": 4,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48
  },
  "quant_type": "Q4_BLOCK",          // informational default; truth = per-tensor dtype in the directory
  "provenance": { "tool": "…", "source_model": "…" }   // optional, free-form
}
```

`norm_style` 对引擎而言是强制的：把 Gemma 风格的 `(1+w)` 应用到
Qwen 权重上，会在一次前向传播的全部约 130 次归一化中产生静默的垃圾结果。

能力分派由**张量存在与否驱动**：引擎根据目录中存在什么来
逐层决定算子（q/k 偏置、qk 归一化、由投影宽度决定的输出门、MoE 路由器、GDN 投影）
—— 而非通过匹配模型名称。已知家族的新模型可以在
零引擎改动的情况下加载。

### 2.1 MTP —— 多 token 预测（可选）

如果模型携带 MTP 头（DeepSeek/Qwen 风格），arch 声明：

```jsonc
"mtp": { "num_layers": 1, "share_lm_head": true, "share_embed": true }
```

MTP 张量是使用规范名称的普通目录条目
（`model.mtp.*`）：`enorm.weight`、`hnorm.weight`、
`eh_proj.weight [hidden, 2·hidden]`、`layers.{i}.*`（一个标准的
transformer 块）、`norm.weight`。

语义：`x = eh_proj·[enorm(embed(t_{p+1})); hnorm(h_p)]` —— 嵌入
在先（oracle 验证：反过来的顺序会得到恰好 0% 的接受率）
→ 块 → 共享 lm_head → 对 token `t_{p+2}` 的草稿。读取器无需
执行 MTP（元数据 + 普通张量，增量式
演进，无特性位）；CMF 运行时将该头用于
推测解码，并有严格保证：**输出与纯贪婪解码完全相等** ——
被拒绝的草稿会从 KV 中回滚。

### 2.2 MoE —— 专家混合 FFN（可选）

如果模型携带 MoE 层（Qwen2-MoE / Qwen3-MoE / Qwen3.5-MoE），
arch 声明：

```jsonc
"moe": {
  "num_experts": 256, "top_k": 8, "moe_intermediate_size": 512,
  "norm_topk_prob": true,                       // Qwen2-MoE: false
  "shared_expert_intermediate_size": 512        // absent if no shared expert
}
```

张量是使用 HF 名称的普通目录条目：

```
model.layers.{i}.mlp.gate.weight                    [num_experts, hidden]  router
model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight
model.layers.{i}.mlp.shared_expert.{gate,up,down}_proj.weight
model.layers.{i}.mlp.shared_expert_gate.weight      [1, hidden]
```

哪些层是 MoE，由目录中路由器的**存在与否**决定
（逐层而非逐模型）：Qwen2-MoE 的
`mlp_only_layers`/`decoder_sparse_step` 会产生混合模型，稠密
层保留普通的 `mlp.*_proj`。

执行语义（HF 对齐，由 `tests/moe_parity.sh` 在
四个家族上把关，包括融合的 AgentWorld 布局）：对全部
路由器 logits 做 softmax → top-k（平局时：取较低索引，torch.topk 顺序）→ 若
`norm_topk_prob`，对所选的 k 个重归一化 → Σwₑ·FFNₑ(x)；共享
专家总是以权重 `sigmoid(shared_expert_gate·x)` 加入。
专家在 mmap 中保持量化状态；每个 token 只触及所选
k 个的页面 —— 与技能相同的驻留机制。每个专家是一个
单独的目录条目，各有**自己的** dtype：这正是
逐专家位宽分配的载体（P15 claim 12）—— 已实现，由
`tests/moe_vbit.sh` 把关；B 场（通过 `--route-stats` 得到的路由器选择频率）
已在一个 35B 模型上端到端测量。

## 3. 张量目录

逐字节采用 `.vmfc` v2 布局（单一正典，共享参考
解析器）：

```
[0 : 8 ]  count    : u64
[8 : 16]  pool_off : u64            (name-pool offset from section start)
[16 : 16 + count·56]  56-byte records:
   name_off : u32   (relative to pool_off)
   name_len : u16
   dtype    : u8    (§3.1)
   ndim     : u8    (≤ 6)
   shape    : u32 × 6  (zero-padded tail)
   off      : u64   (RELATIVE to data_off; multiple of 64)
   nbytes   : u64
   hash     : u64   (hash64 of the tensor bytes, §8)
[pool_off : …]  UTF-8 name pool
```

张量名与源模型**一一对应**
（`model.layers.{i}.mlp.gate_proj.weight`、`model.embed_tokens.weight`、
`lm_head.weight`、…）。格式不规定张量集合：
目录是 blob 内容的唯一真相来源。
不存在"可计算布局"。

### 3.1 `dtype`

编号与 `.vmfc` 共享（id 从不重用）：

| id | name       | 在 CMF v2 中的状态 |
|----|-----------|------------------|
| 0  | `f32`     | ✅ 读/写 |
| 1  | `f16`     | ✅ 读/写（归一化及 1 维张量始终为 f16） |
| 2  | `bf16`    | ✅ 读/写 |
| 3  | `q8_row`  | ✅ 读/写 |
| 4  | `q4_block`| ✅ 读/写 |
| 5  | `mix8_4`  | 保留 |
| 6  | `u8`      | 保留 |
| 7  | `q4_col`  | 保留 |
| 8  | `vbit`    | ✅ 读/写（`QUANT_2F` 位），可变 3–8 位 |
| 9  | `q8_2f`   | ✅ 读/写（`QUANT_2F` 位），𝒲×θ |
| 10 | `vbit_ro` | ✅ 读/写——`vbit` + 文件内行偏移表（O(1) 行访问）；`--quant vbit` 的转换器默认 |
| 11 | `q4_tiled`| ✅ 读/写——交错平铺的 q4 `[f16 scale][16B nibbles]`（`--quant q4t`） |
| 12 | `q1`      | ✅ 读/写——1 位二值权重，仅用于按 1 位训练的模型（`--quant q1`）；每 32 组一个 `[f16 scale][4B 符号位]` 平铺，`w = s·(2·bit−1)` |

### 3.2 量化布局（正典 = `.vmfc`："先量化值，后 scale"）

- **`q8_row`**（仅限二维 `[out, in]`）：
  `[int8 : out·in][f16 : out]` —— 每行一个 scale，
  `w = q[o,i]·scale[o]`，`scale[o] = absmax(row_o)/127`。
- **`q4_block`**：在展平张量上按 32 分组，零填充；
  `[u8 : ceil(n/32)·16][f16 : ceil(n/32)]`。
  半字节：元素 `2k` 为低位，`2k+1` 为高位；`w = (q − 8)·scale`，
  `scale = absmax(group)/7`。
- **一维张量以及元素数 < 32 的张量始终为 `f16`**
  （在矩阵最大压缩下保持归一化精度）。
- **`q8_2f`**：`[int8][f16 row-scale][f16 col-field]`，
  `w = q·scale[o]·col[i]` —— 双场 Madelung 拆分 𝒲×θ，在
  vmfcore 中已验证（同尺寸下 +37%；在离群输入通道上恢复了 q8→f16 差距的约 75%）。
- **`vbit`**（仅限二维，`in % 32 == 0`；P13 FIG.3）：
  `[u8 bits: rows][f16 scales: rows·in/32][bit-packed rows, MSB-first,
  each row padded to a byte]`；`w = (u − L)·scale[r,g]`，
  `L = 2^{b−1}−1`，位宽 b ∈ {3,4,5,6,8}，下限 3（claim 13）。
  分配 b_r：在 log2 行幅度上朝张量的平均预算做注水；
  对于 MoE 专家，预算在**家族内共享**（层 × 投影）：偏移
  `ā_expert − ā_family` 等价于在所有专家的行上做联合注水 —— 一个响亮的
  专家获得更多位，一个安静的被钉在下限（P15
  claim 12；gate `tests/moe_vbit.sh`）。分配可选地
  与一个 B 场取积 —— 在标定时收集的路由器选择频率
  （`b ∝ log2(A·B)`，截断 Fisher）。

## 4. 权重 blob

`data_off` 是 4096 的倍数（按页对齐的 mmap）；其中每个张量
都从 64 字节边界开始（SIMD 加载、缓存行）。张量之间为零
填充。读取器只能通过目录来解释该 blob。

## 5. 掩码区段

任务掩码 = 在共享权重之上"哪些活跃"的位字段（权重不变
—— VMF 原则：技能选取凝聚态的一个
子集）。

```
[0 : 4]  n_masks  : u32
[4 : 8]  meta_len : u32
[8 : 8 + meta_len]  JSON meta (§5.1)
[…]      mask blobs, each aligned to 8 from the section start
```

一个掩码 blob（尺寸由 arch 推导，无内部头部）：

```
[n_layers × ffn_bytes]   FFN bitfields      ffn_bytes  = ceil(intermediate_size / 8)
[n_layers × head_bytes]  head bitfields     head_bytes = ceil(num_attention_heads / 8)
[gates_bytes]            layer_gates        gates_bytes = ceil(num_layers / 8)
```

位序为 LSB 优先：神经元 `i` = 字节 `i / 8` 的第 `i % 8` 位；置位
→ 活跃。**超出维度的尾部位必须为零**（否则
popcount 会看到幻影神经元/头）。

### 5.1 掩码 JSON meta

```jsonc
{
  "default_task": "general",
  "masks": [{
    "task_id": 0, "name": "general", "description": null,
    "sparsity": 0.62,
    "quality": {                    // null = NOT MEASURED (declaring 1.0 is forbidden)
      "metric": "heldout_ppl_ratio", "value": 0.97,
      "baseline_dense": 6.10, "n_samples": 512, "dataset_sha256": "…"
    },
    "parent": null, "priority": "Fallback", "has_hot_pack": false,
    "blob_off": 4096, "blob_len": 139328   // relative to section start
  }]
}
```

`quality` 是一份**留出集契约**，而非一个声明：没有实测指标的
转换器写入 `null`；运行时在切换到未测量掩码时会记录警告。

## 6. 分词器区段

HuggingFace `tokenizer.json` 的字节，逐字保留。模型
自包含：一个文件 = 一个分发单元。附带文件
仍作为调试回退。

### 6.1 聊天捆绑（`header.tokenizer_config`）

是文件 —— 而非运行时二进制 —— 定义了聊天行为。头部
携带一个可选块（增量式演进，无特性位）：

```json
"tokenizer_config": {
  "chat_template": "<Jinja template from chat_template.jinja or tokenizer_config.json>",
  "eos_token_ids": [248044, 248045],
  "bos_token_id": null,
  "pad_token_id": 248055
}
```

运行时以 HF 语义渲染模板（trim_blocks、
lstrip_blocks、循环控制、Python 字符串方法），并在
`eos_token_ids` 中的任一 id 处停止生成。Gate：
`tests/chat_template_parity.sh` —— 运行时渲染结果与参考
jinja2 逐字节相等。没有该块的文件获得 ChatML 回退。

## 7. 稀疏索引

一座预计算的"掩码 → 计算跳过"桥梁：逐 (task, layer) 对的
活跃 FFN 量化组（每组 32 个神经元）与头。

> 诚实的状态：引擎当前直接从
> 掩码位字段取活跃索引；索引会被 CLI 读取并显示，但尚未
> 用于执行。它在
> "掩码 × 量化 mmap"路径上会变为强制项。

```
[0 : 4]  n_entries : u32
[4 : 8]  reserved  : u32 (0)
entry (4-aligned):
   task_id   : u32
   layer_idx : u32
   n_groups  : u32
   n_heads   : u32
   [u16 × n_groups]  active FFN-group indices (sorted)
   [u8  × n_heads]   active head indices (sorted)
   zero padding to a multiple of 4
```

一个组只要包含至少一个活跃掩码位即为活跃。

## 8. `hash64`

张量字节的非加密 64 位哈希：在 64 位 LE 字上做 murmur3
`fmix64`，带位置盐 `i·0x9E3779B97F4A7C15`，XOR 折叠、
`xor len`、最终 `fmix64`。与
`vmfcore.hash64`（Python）及 `vmfcore::hash64`（Rust）逐位兼容 ——
`.cmf` 与 `.vmfc` 之间共享张量的哈希相符（跨技能文件的
骨干去重是免费的）。

用途：`cortiq verify`（损坏检测）、去重、缓存键。

### 8.1 区段哈希

元数据完整性（不仅是张量）：

- 信封保留区 `[0x70:0x78]` = hash64(header JSON)，`[0x78:0x80]` =
  hash64(directory)。零 = "缺失"（旧文件可通过）。
- 头部 JSON 携带 `section_hashes` —— masks/vocab/index 的十六进制
  hash64（u64 若作为 JSON 数字，超过 2^53 会损失
  精度）。信封中的头部哈希传递性地覆盖它们。
- 信封自身（前 0x70 字节）不被哈希：哈希无法
  保护自己；损坏的偏移量会被链条下游的边界/哈希捕获。
- `cortiq verify` 检查整条链；单个被翻转的头部字节
  就是一个错误。

## 反特性 —— 格式刻意不具备的东西

- **可计算权重布局** —— v1 的一号 bug 类别（写入器与读取器
  各自"计算"布局并发生分歧）。
- **静默回退** —— v1 会把任何垃圾文件解释为"一个 27B
  模型"；v2 必须失败。
- **用 JSON 存位数据** —— v1 掩码用 JSON 存储会膨胀 3–4 倍。
- **声明字段** —— 默认的 `quality_score: 1.0`、面积律
  "容量"、动力学中的 Born 乘子：隐喻在被测量之前
  不会成为格式字段。

## 9. 技能 —— 一个文件中的集群（Patent 15，claims 2/12/15）

一个共享骨干 + K 条逐技能记录；没有任何记录存储完整
模型。存储量按 |backbone| + Σ|deltas| 伸缩。

**替换张量**是命名为
`skill.{skill_id}.{name_of_replaced_tensor}` 的普通目录条目，例如
`skill.sql.model.layers.3.mlp.gate_proj.weight`。被替换张量的完整逻辑形状
（full-shape —— 非低秩、非差异列表、非
掩码），可采用 §3 的任意编码。逐技能增量索引（claim 2）由
目录物化：一次前缀过滤得到 技能 →
字节偏移；惰性分页 = mmap 精确访问那些偏移
（claim 12）。

**注册表** —— 头部 JSON，增量式：

```json
"skills": [{
  "id": "sql",
  "name": "SQL assistant",
  "layers": [3, 4, 5],
  "selection": {"metric": "mse", "phi_layer": 20,
                 "mean": "<f16 base64>", "basis": "<f16 base64>"},
  "input_mask_task": null,
  "quality": {"metric": "ppl", "backbone": 21.4, "overlaid": 17.9,
               "dataset_sha256": "…"}
}]
```

`selection` 保存用于 recon-argmin 路由的仿射子空间参数
（`E = ‖r − BBᵀr‖²/‖φ‖²`，选择 E 最小的技能）；
文件对于选择是自足的。`quality` 是诚实的 claim-16
契约（留出数据上的叠加态 vs 骨干）。

**执行语义（claims 1/3/18）**：张量来源间接寻址 —— 对于
每个张量，运行时读取**要么**骨干条目，**要么**存在时的
`skill.{active}.{name}`；是替换而非相加，从不组装
完整的逐技能模型（所有张量都是指向
同一 mmap 的指针）。软叠加（claim 14）：混合工作张量
`Σwᵢ·Tᵢ`，`wᵢ = softmax(−E/T)`。

**仅追加式增长（claim 11）**：添加一个技能 = 在文件尾部追加新
张量 + 在尾部重新发出 目录/头部/索引 + 就地
更新信封偏移量（偏移 0 固定）。已写入张量的字节与
偏移量永不改变；旧的 dir/header 字节
成为死区段尾（兼容：读取器只通过
信封导航）。压实（`converter/cmf_compact.py`）= 一次普通重写。

状态：完全实现并把关（容器 + 间接寻址、
生产配方、recon-argmin 路由、仅追加 + 压实、
软混合）；claim 16 由测量满足（运行时任务 PPL −24.9%）。

## 10. 分片 —— 一个模型分为 N 个文件

命名：`{base}-{no:05}-of-{count:05}.cmf`（在精神上与
safetensors 兼容）。用户可打开**任意**名称；运行时归一化到分片 1
并按模式拾取兄弟文件。

**每个分片都是一个独立的有效 .cmf**：完整信封、头部 JSON、
一份属于**自己**张量的目录、自己的数据 blob、自己的哈希
（`section_hashes` + 逐张量）。`cortiq verify` 可在没有兄弟文件的情况下
对任意单个分片工作。

每个分片的头部携带：

```json
"shard": { "no": 1, "count": 5 }
```

没有该块 = 一个普通的单文件（向后兼容：旧读取器把
分片 1 视为一个有效但不完整的模型，并在缺失张量处诚实地失败）。

**内容分布**：张量按规范顺序贪婪地拆分
（`--shard-max-gb` 阈值，粗略的 f32 尺寸）；masks/vocab/稀疏
索引区段、`tokenizer_config`（聊天捆绑）以及 `skills`
注册表**仅**存在于分片 1 中 —— 其余分片的这些区段为空，且
`tokenizer_config: null`。技能张量（`skill.{id}.*`）作为
普通目录条目分布 —— 分片 1 的注册表通过合并后的目录按名称
引用它们。

**加载**（`CmfModel::open_sharded`）：打开分片 1 → mmap 所有兄弟文件
→ 合并目录（每个条目记住其分片索引 —— 一个运行时
字段，绝不写入磁盘）→ 运行时随后如同处理单个
文件一般工作。错误：直接打开一个非首分片、缺失一个兄弟文件、
`count` 不匹配。

Gate（Qwen3.5-0.8B q8_2f，5 个分片 ≤ 0.6 GB）：分片化 PPL == 未分片化，
在同一二进制上逐字节相等；`verify` 在每个单独分片上皆为绿。

---

*相关：[COMPARISON.md](COMPARISON.md)（CMF 与其他模型格式的对比）、
[项目 README](../README.md)（概览与快速上手）、
`python/cmf_reader.py`（独立读取器：标准库 + numpy，读取所有
dtype、分片、技能、verify）。*
