# CMF v2 — Format Specification

*Languages: **English** · [Русский](CMF_V2_SPEC.ru.md) · [中文](CMF_V2_SPEC.zh.md)*

**Cortiq Model Format** — a single file carrying everything needed for
sparse, task-routed inference: quantized weights, tokenizer, per-task
masks, a precomputed sparse index — and, uniquely, a **swarm of skills**
sharing one backbone (Patent 15).

> Normative source: this document. Reference
> implementations: Rust reader/runtime (`crates/cortiq-core`,
> `crates/cortiq-engine`), Python writer (`converter/`), and a
> standalone Python reader (`python/cmf_reader.py`, stdlib + numpy).

Three requirements, in priority order:

1. **Correct.** No silent corruption modes: strict magic, version,
   `required_features`, bounds on every section, a 64-bit hash for every
   tensor. A file is either valid or open() returns an error — there is
   no third state.
2. **Fast.** The weight section is page-aligned for mmap, every tensor
   is 64-byte aligned (zero-copy SIMD), the tensor directory is binary —
   read without parsing. Cold (masked-out) weights cost no RSS.
3. **Compact.** Masks are bit-packed (1 bit per neuron), weights are
   q4/q8/variable-bit, the whole file is addressed by one 128-byte
   envelope.

Harmony comes not from feature count but from **a single canon**: one
layout per level (envelope, directory, quant block, mask), byte-for-byte
compatible with the validated `.vmfc` v2 format where the domains
overlap (tensor directory, quant layouts, `hash64`). Never two
definitions of the same thing.

Physical basis (VMF): the model is a vacuum condensate 𝒲; a skill is its
regular core above a critical density; a task mask selects an active
subset without changing weights. The format carries the consequences of
that physics (two-field 𝒲×θ quantization, Born importance, critical mask
threshold) — but **only those confirmed by measurement**.

---

## 1. Envelope (fixed 128 bytes)

All integers are little-endian.

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

Section order on disk: envelope → header JSON → directory → **weight
blob (aligned to 4096)** → masks → vocab → sparse index. A reader MUST
address sections ONLY through the envelope, never by assuming order.

### 1.1 `required_features`

A bit the reader does not know → `UnsupportedFeature` error (fail-fast;
no "read as best we can").

| bit | name           | meaning |
|-----|----------------|---------|
| 0   | `TENSOR_DIR`   | binary tensor directory (always set in v2) |
| 1   | `BINARY_MASKS` | masks section (§5) present |
| 2   | `QUANT_2F`     | directory contains `q8_2f`/`vbit` tensors (two-field 𝒲×θ quant) |
| 3   | `DELTA_MASKS`  | reserved: XOR mask deltas from a parent |
| 4   | `HOT_PACKS`    | reserved: materialized dense slices |

Unknown **header-JSON** fields are ignored (additive evolution);
breaking changes go only through feature bits or a `version` bump.

### 1.2 Validation rules (normative)

The reader MUST return an error (not a default, not a warning) when:

- magic ≠ `CMF\x01` → `InvalidMagic`;
- `version` ≠ 2 → `UnsupportedVersion` (v1 is dead: no real v1 files
  exist, no support program will be started);
- an unknown `required_features` bit is set → `UnsupportedFeature`;
- any section extends past EOF, `data_off` is not a multiple of 4096, a
  tensor's `off + nbytes` exceeds `data_len` → `Bounds`;
- a tensor name is not UTF-8, dtype is unknown, `ndim > 6` → `Parse`.

Tensor-hash verification is on demand (`cortiq verify`, a loader flag),
not on every open: mmap pages are read lazily.

## 2. Header JSON

UTF-8 JSON, unaligned. Machine-critical data lives in binary sections;
JSON carries architecture and provenance — the parts a human reads.

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

`norm_style` is mandatory for an engine: Gemma-style `(1+w)` applied to
Qwen weights is silent garbage across all ~130 normalizations of a
forward pass.

Capability dispatch is **tensor-presence driven**: an engine decides
per-layer operators by what exists in the directory (q/k biases,
qk-norms, output gate by projection width, MoE router, GDN projections)
— not by matching model names. New models of a known family load with
zero engine changes.

### 2.1 MTP — multi-token prediction (optional)

If the model carries an MTP head (DeepSeek/Qwen style), arch declares:

```jsonc
"mtp": { "num_layers": 1, "share_lm_head": true, "share_embed": true }
```

MTP tensors are ordinary directory entries under canonical names
(`model.mtp.*`): `enorm.weight`, `hnorm.weight`,
`eh_proj.weight [hidden, 2·hidden]`, `layers.{i}.*` (a standard
transformer block), `norm.weight`.

Semantics: `x = eh_proj·[enorm(embed(t_{p+1})); hnorm(h_p)]` — embedding
FIRST (oracle-verified: the reverse order yields exactly 0% acceptance)
→ block → shared lm_head → draft of token `t_{p+2}`. A reader is not
required to execute MTP (metadata + ordinary tensors, additive
evolution, no feature bit); the CMF runtime uses the head for
speculative decode with a strict guarantee: **output is exactly equal to
plain greedy** — a rejected draft is rolled back from KV.

### 2.2 MoE — mixture-of-experts FFN (optional)

If the model carries MoE layers (Qwen2-MoE / Qwen3-MoE / Qwen3.5-MoE),
arch declares:

```jsonc
"moe": {
  "num_experts": 256, "top_k": 8, "moe_intermediate_size": 512,
  "norm_topk_prob": true,                       // Qwen2-MoE: false
  "shared_expert_intermediate_size": 512        // absent if no shared expert
}
```

Tensors are ordinary directory entries under HF names:

```
model.layers.{i}.mlp.gate.weight                    [num_experts, hidden]  router
model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight
model.layers.{i}.mlp.shared_expert.{gate,up,down}_proj.weight
model.layers.{i}.mlp.shared_expert_gate.weight      [1, hidden]
```

Which layers are MoE is decided by the PRESENCE of the router in the
directory (per-layer, not per-model): Qwen2-MoE's
`mlp_only_layers`/`decoder_sparse_step` produce mixed models, and dense
layers keep ordinary `mlp.*_proj`.

Execution semantics (HF parity, gated by `tests/moe_parity.sh` across
four families including the fused AgentWorld layout): softmax over ALL
router logits → top-k (ties: lower index, torch.topk order) → if
`norm_topk_prob`, renormalize the selected k → Σwₑ·FFNₑ(x); the shared
expert is always added with weight `sigmoid(shared_expert_gate·x)`.
Experts stay quantized in mmap; per token only the pages of the selected
k are touched — the same residency story as skills. Each expert is a
separate directory entry with ITS OWN dtype: that is the carrier of
per-expert bit allocation (P15 claim 12) — implemented, gated by
`tests/moe_vbit.sh`; the B-field (router selection frequencies via
`--route-stats`) was measured end-to-end on a 35B model.

## 3. Tensor directory

Byte-for-byte the `.vmfc` v2 layout (single canon, shared reference
parser):

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

Tensor names are **1:1 with the source model**
(`model.layers.{i}.mlp.gate_proj.weight`, `model.embed_tokens.weight`,
`lm_head.weight`, …). The format does not prescribe a tensor set: the
directory is the single source of truth for what the blob contains.
There is no "computable layout".

### 3.1 `dtype`

Numbering shared with `.vmfc` (ids are never reused):

| id | name       | status in CMF v2 |
|----|-----------|------------------|
| 0  | `f32`     | ✅ read/write |
| 1  | `f16`     | ✅ read/write (norms and 1-D are always f16) |
| 2  | `bf16`    | ✅ read/write |
| 3  | `q8_row`  | ✅ read/write |
| 4  | `q4_block`| ✅ read/write |
| 5  | `mix8_4`  | reserved |
| 6  | `u8`      | reserved |
| 7  | `q4_col`  | reserved |
| 8  | `vbit`    | ✅ read/write (`QUANT_2F` bit), variable 3–8 bit |
| 9  | `q8_2f`   | ✅ read/write (`QUANT_2F` bit), 𝒲×θ |
| 10 | `vbit_ro` | ✅ read/write — `vbit` + in-file row-offset table (O(1) row access); converter default for `--quant vbit` |
| 11 | `q4_tiled`| ✅ read/write — q4 in interleaved `[f16 scale][16B nibbles]` tiles (`--quant q4t`) |
| 12 | `q1`      | ✅ read/write — 1-bit binary, for 1-bit-TRAINED models only (`--quant q1`) |

### 3.2 Quant layouts (canon = `.vmfc`: "quants first, then scales")

- **`q8_row`** (2-D `[out, in]` only):
  `[int8 : out·in][f16 : out]` — one scale per row,
  `w = q[o,i]·scale[o]`, `scale[o] = absmax(row_o)/127`.
- **`q4_block`**: groups of 32 over the flattened tensor, zero-padded;
  `[u8 : ceil(n/32)·16][f16 : ceil(n/32)]`.
  Nibbles: element `2k` low, `2k+1` high; `w = (q − 8)·scale`,
  `scale = absmax(group)/7`.
- **1-D tensors and tensors < 32 elements are always `f16`**
  (normalization precision at maximal matrix compression).
- **`q8_2f`**: `[int8][f16 row-scale][f16 col-field]`,
  `w = q·scale[o]·col[i]` — the two-field Madelung split 𝒲×θ, validated
  in vmfcore (+37% at equal size; recovers ~75% of the q8→f16 gap on
  outlier input channels).
- **`vbit`** (2-D only, `in % 32 == 0`; P13 FIG.3):
  `[u8 bits: rows][f16 scales: rows·in/32][bit-packed rows, MSB-first,
  each row padded to a byte]`; `w = (u − L)·scale[r,g]`,
  `L = 2^{b−1}−1`, levels b ∈ {3,4,5,6,8}, floor 3 (claim 13).
  Allocation b_r: water-filling over the log2 row amplitude toward the
  tensor's mean budget; for MoE experts the budget is SHARED across the
  family (layer × projection): the shift `ā_expert − ā_family` is
  equivalent to joint water-filling over all experts' rows — a loud
  expert gets more bits, a quiet one is pinned to the floor (P15
  claim 12; gate `tests/moe_vbit.sh`). Optionally the allocation takes
  the product with a B-field — router selection frequencies collected
  at calibration (`b ∝ log2(A·B)`, truncated Fisher).
- **`vbit_ro`** (2-D only, `in % 32 == 0`): the same bits/scales/packed
  encoding as `vbit`, plus `u32 row_offsets[rows+1]` (relative to the
  packed area) between the scales and the packed rows —
  `[u8 bits: rows][f16 scales: rows·in/32][u32 offsets: rows+1][packed]`.
  Readers get O(1) row access without a prefix scan over bit widths.
  The byte semantics of `vbit = 8` are untouched; new id on purpose.
- **`q4_tiled`** (2-D only, `in % 32 == 0`):
  `repeat per 32-group { [f16 scale][16B nibbles] }` — 18-byte tiles,
  one sequential memory stream instead of two distant ones. Values and
  nibble order are identical to `q4_block`; only the placement of the
  scale differs (kernel-measured ×1.66 ARM / ×1.13 AVX2 over split).
- **`q1`** (2-D only, `in % 32 == 0`):
  `repeat per 32-group { [f16 scale][4B sign bits] }` — 6-byte tiles,
  1.5 bits/weight. Bit k of byte j (LSB-first) is weight j·8+k of the
  group; `w = scale·(2·bit − 1) ∈ {−s, +s}`, `scale = mean|group|`
  (the L2-optimal binary level). Intended for 1-bit-TRAINED models
  (Bonsai / BitNet class), where per-group weights already sit on two
  levels and the encoding is lossless up to f16; as post-training
  quantization of a normal checkpoint it destroys quality, so
  converters expose it only as an explicit opt-in.

## 4. Weight blob

`data_off` is a multiple of 4096 (page-aligned mmap); every tensor
inside starts on a 64-byte boundary (SIMD loads, cache lines). Zero
padding between tensors. A reader interprets the blob only through the
directory.

## 5. Masks section

A task mask = bit fields of "what is active" over shared weights
(weights do not change — the VMF principle: a skill selects a subset of
the condensate).

```
[0 : 4]  n_masks  : u32
[4 : 8]  meta_len : u32
[8 : 8 + meta_len]  JSON meta (§5.1)
[…]      mask blobs, each aligned to 8 from the section start
```

One mask blob (sizes derived from arch, no internal headers):

```
[n_layers × ffn_bytes]   FFN bitfields      ffn_bytes  = ceil(intermediate_size / 8)
[n_layers × head_bytes]  head bitfields     head_bytes = ceil(num_attention_heads / 8)
[gates_bytes]            layer_gates        gates_bytes = ceil(num_layers / 8)
```

Bit order is LSB-first: neuron `i` = bit `i % 8` of byte `i / 8`; bit
set → active. **Tail bits beyond the dimension MUST be zero** (or
popcount sees phantom neurons/heads).

### 5.1 Mask JSON meta

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

`quality` is a **held-out contract**, not a declaration: a converter
without a measured metric writes `null`; the runtime logs a warning when
switching to an unmeasured mask.

## 6. Tokenizer section

The bytes of HuggingFace `tokenizer.json`, verbatim. The model is
self-contained: one file = one unit of distribution. A sidecar file
remains a debugging fallback.

### 6.1 Chat bundle (`header.tokenizer_config`)

The file — not the runtime binary — defines chat behavior. The header
carries an optional block (additive evolution, no feature bit):

```json
"tokenizer_config": {
  "chat_template": "<Jinja template from chat_template.jinja or tokenizer_config.json>",
  "eos_token_ids": [248044, 248045],
  "bos_token_id": null,
  "pad_token_id": 248055
}
```

The runtime renders the template with HF semantics (trim_blocks,
lstrip_blocks, loop controls, Python string methods) and stops
generation on any id in `eos_token_ids`. Gate:
`tests/chat_template_parity.sh` — the runtime render equals reference
jinja2 byte-for-byte. Files without the block get a ChatML fallback.

## 7. Sparse index

A precomputed bridge "mask → computation skip": active FFN quant groups
(32 neurons each) and heads, per (task, layer) pair.

> Honest status: the engine currently takes active indices directly from
> the mask bitfields; the index is read and displayed by the CLI but not
> used in execution yet. It becomes mandatory on the
> "masks × quantized mmap" path.

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

A group is active if it contains at least one active mask bit.

## 8. `hash64`

A non-cryptographic 64-bit hash of tensor bytes: murmur3 `fmix64` over
64-bit LE words with positional salt `i·0x9E3779B97F4A7C15`, XOR fold,
`xor len`, final `fmix64`. Bit-for-bit compatible with
`vmfcore.hash64` (Python) and `vmfcore::hash64` (Rust) — hashes of
shared tensors match between `.cmf` and `.vmfc` (backbone dedup across
skill files is free).

Uses: `cortiq verify` (corruption detection), dedup, cache keys.

### 8.1 Section hashes

Metadata integrity (not just tensors):

- Envelope reserve `[0x70:0x78]` = hash64(header JSON), `[0x78:0x80]` =
  hash64(directory). Zero = "absent" (older files pass).
- The header JSON carries `section_hashes` — hex hash64 of
  masks/vocab/index (u64 as a JSON number would lose precision past
  2^53). The header hash in the envelope transitively covers them.
- The envelope itself (first 0x70 bytes) is not hashed: a hash cannot
  protect itself; corrupted offsets are caught by bounds/hashes further
  down the chain.
- `cortiq verify` checks the whole chain; a single flipped header byte
  is an error.

## Anti-features — what the format deliberately does NOT have

- **A computable weight layout** — bug class #1 of v1 (writer and reader
  "computed" the layout independently and diverged).
- **Silent fallbacks** — v1 would interpret any garbage file as "a 27B
  model"; v2 must fail.
- **JSON for bit data** — v1 masks in JSON bloated 3–4×.
- **Declaration fields** — `quality_score: 1.0` by default, area-law
  "capacities", Born multipliers in dynamics: a metaphor does not become
  a format field until it is measured.

## 9. Skills — a swarm in one file (Patent 15, claims 2/12/15)

One shared backbone + K per-skill records; no record stores a full
model. Storage scales as |backbone| + Σ|deltas|.

**Replacement tensors** are ordinary directory entries named
`skill.{skill_id}.{name_of_replaced_tensor}`, e.g.
`skill.sql.model.layers.3.mlp.gate_proj.weight`. The full logical shape
of the replaced tensor (full-shape — NOT low-rank, NOT a diff list, NOT
a mask), in any encoding of §3. The per-skill delta index (claim 2) is
materialized by the directory: a prefix filter yields skill →
byte-offsets; lazy paging = mmap access to exactly those offsets
(claim 12).

**Registry** — header JSON, additive:

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

`selection` holds the affine-subspace parameters for recon-argmin
routing (`E = ‖r − BBᵀr‖²/‖φ‖²`, choose the skill with minimal E); the
file is self-sufficient for selection. `quality` is the honest claim-16
contract (overlaid vs backbone on held-out data).

**Execution semantics (claims 1/3/18)**: tensor-source indirection — for
every tensor the runtime reads EITHER the backbone entry OR
`skill.{active}.{name}` if present; replacement instead of addition, a
full per-skill model is never assembled (all tensors are pointers into
one mmap). Soft superposition (claim 14): blended working tensors
`Σwᵢ·Tᵢ`, `wᵢ = softmax(−E/T)`.

**Append-only growth (claim 11)**: adding a skill = appending new
tensors at the file tail + re-emitting directory/header/index at the
tail + updating envelope offsets in place (offset 0 is fixed). Bytes and
offsets of previously written tensors never change; old dir/header bytes
become dead section tails (compatible: readers navigate only through the
envelope). Compaction (`converter/cmf_compact.py`) = a plain rewrite.

Status: fully implemented and gated (container + indirection,
production recipes, recon-argmin routing, append-only + compaction,
soft-blend); claim 16 met by measurement (−24.9% task-PPL in the
runtime).

## 10. Sharding — a model in N files

Naming: `{base}-{no:05}-of-{count:05}.cmf` (spiritually compatible with
safetensors). The user opens ANY name; the runtime normalizes to shard 1
and picks up siblings by pattern.

**Every shard is a standalone valid .cmf**: full envelope, header JSON,
a directory of ITS OWN tensors, its own data blob, its own hashes
(`section_hashes` + per-tensor). `cortiq verify` works on any single
shard without its siblings.

Each shard's header carries:

```json
"shard": { "no": 1, "count": 5 }
```

No block = an ordinary single file (backward compatible: old readers see
shard 1 as a valid but incomplete model and fail honestly on the missing
tensor).

**Content distribution**: tensors are split greedily in canonical order
(`--shard-max-gb` threshold, rough f32 size); the masks/vocab/sparse
index sections, `tokenizer_config` (chat bundle) and the `skills`
registry live ONLY in shard 1 — the rest have empty sections and
`tokenizer_config: null`. Skill tensors (`skill.{id}.*`) are distributed
as ordinary directory entries — the shard-1 registry references them by
name through the merged directory.

**Loading** (`CmfModel::open_sharded`): open shard 1 → mmap all siblings
→ merge directories (each entry remembers its shard index — a runtime
field, never written to disk) → the runtime then works as with a single
file. Errors: opening a non-first shard directly, a missing sibling, a
`count` mismatch.

Gate (Qwen3.5-0.8B q8_2f, 5 shards ≤ 0.6 GB): sharded PPL == unsharded
byte-exactly on the same binary; `verify` green on every shard alone.

## 11. Defragmentation — physical pruning (Patent 2, claims 9/10)

A mask (§5) is **virtual sparsity**: pruned neurons are flagged but still
stored in full (all tasks share one backbone — you cannot physically cut
it until you commit to ONE task). Defragmentation turns virtual sparsity
into **physical compression**: pruned FFN neurons are dropped from the
file — they are **neither stored nor computed**. This is Factory-Hard →
defrag from Patent 2: "bake one mask into the weights" and emit a
standalone compact `.cmf`.

**Representation — no new feature bit, backward compatible.** Physical
pruning is expressed ONLY by smaller tensor shapes in the directory (§3
"no computable layout"; the directory is the sole shape authority). The
runtime derives the FFN size from the tensor shape (`gate_proj.rows()`),
not from `arch.intermediate_size`, so a defragged file is an ordinary
smaller dense model that existing readers load unchanged. The masks
section (§5) is **absent** in a defragged file (the mask is the identity
after pruning). `arch.intermediate_size` becomes nominal (= the per-layer
max); the true size lives in each tensor.

**Per-layer variance — better than the patent.** Because the directory
carries an arbitrary shape per tensor, each layer shrinks to its OWN
live-neuron count. The patent must truncate every layer to `max(active)`
(one bottleneck layer caps the ratio at 80.2% vs. 94% achievable) — CMF
has no such limit.

**Invariants (mandatory):**

- per-layer triple: `gate_proj.rows() == up_proj.rows() ==
  down_proj.cols() == inter'ₗ`, and `down_proj.rows() == hidden_size`.
  One keep-set indexes all three (gate row i, up row i, down col i are
  the same neuron);
- neuron axis: rows (axis 0) for `gate_proj`/`up_proj`, columns (axis 1)
  for `down_proj`;
- quant group of 32: the `down_proj` neuron axis is its COLUMNS, and
  `vbit`/`q4_block` require `in % 32 == 0`. A `down_proj` whose `inter'`
  is not a multiple of 32 is written as `q8_2f` (per-row scale — no
  column constraint; the converter downgrades automatically). `gate/up`
  drop rows, so their columns (= hidden) are unaffected;
- NOT a byte truncation: quant scales are per-group/per-row, so pruning
  is dequant → gather live neurons → **requant** at the smaller shape
  (the `q8_2f` col-field / `vbit` scales of `down_proj` regenerate for
  the shrunk column set); tensor hashes are recomputed;
- `hidden_size`, `embed_tokens`, `lm_head`, and norms are untouched
  (skill-selection subspaces depend on hidden).

**One task, standalone file.** Defrag is destructive: one `.cmf` bakes
exactly one task. Multi-task serving stays on masks (§5) or per-skill
replacement tensors (§9).

**Provenance (honest contract).** Header `provenance.defrag`:

```jsonc
"defrag": {
  "source_skill": "…/skill_ru",
  "pre_intermediate": 3072,
  "post_intermediate_max": 640,
  "kept_per_layer": [608, 640, 512, ...],
  "pruned_ratio": 0.803
}
```

Numerically the dense output of a defragged model is IDENTICAL to the
masked output before quantization (a dead neuron contributes `act·0`
under a mask and is simply absent after defrag); after quantization the
only difference comes from quantizing the smaller matrices.

**Scope:** FFN neurons only (dense). Attention-head pruning (the head
count is a global runtime scalar) and MoE-expert pruning are out of scope.

**Producing it (native Rust):**

```
cortiq convert --model <hf_dir_or_repo> --defrag <skill_dir> \
  --quant q8_2f --output model.cmf
```

`<skill_dir>` carries baked FFN overlays (`tensors/*.npy`) and, if
available, a keep-set `ffn_keep.npy` (bool `[n_layers, intermediate]`,
True = live) from the pruning pipeline. Without `ffn_keep.npy` the
keep-set is autodetected from all-zero `down_proj` columns (the
Factory-Hard bake). The mask-training / bake step lives in the private
research pipeline; the public tool only consumes its artifacts.

---

*Related: [COMPARISON.md](COMPARISON.md) (CMF vs. other model formats),
[project README](../README.md) (overview and quick start),
`python/cmf_reader.py` (standalone reader: stdlib + numpy, reads every
dtype, shards, skills, verify).*
