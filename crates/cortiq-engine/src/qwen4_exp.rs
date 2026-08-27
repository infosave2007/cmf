//! Exact text-only Qwen3.8-Flash-Next (`qwen4_exp`) stack.
//!
//! The implementation deliberately composes `QTensor` projections instead
//! of owning backend-specific buffers.  Consequently q4tp weights use the
//! same CPU/Vulkan/DX12/Metal kernels, resident arena and expert LRU as every
//! other CMF model; weights that do not fit VRAM remain mmap-backed in RAM.

use crate::linear_core::{GdnCfg, GdnWeights, gdn_forward};
use crate::loader::{Overlay, build_ffn_at, load_f32, load_matrix};
use crate::pipeline::{FfnKind, MoeFfn, moe_ffn};
use crate::pool::Pool;
use crate::qtensor::QTensor;
use cortiq_core::{CmfError, CmfModel, LayerType, ModelArch, Qwen4ExpConfig, TensorDtype};
use std::cmp::Ordering;
use std::sync::Arc;

const PRIME_1: u64 = 10_007;
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;

pub struct GatedResidual {
    norm: Vec<f32>,
    down: QTensor,
    up: QTensor,
    inject: Option<QTensor>,
}

pub struct QsaWeights {
    q_proj: QTensor,
    k_proj: QTensor,
    v_proj: QTensor,
    o_proj: QTensor,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    index_qk: QTensor,
    index_q_norm: Vec<f32>,
    index_k_norm: Vec<f32>,
}

pub struct PleWeights {
    shards: Vec<QTensor>,
    rows_per_shard: usize,
    row_dim: usize,
    key_proj: QTensor,
    value_proj: QTensor,
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
    conv: Vec<f32>,
    multipliers: Vec<i64>,
    vocab_sizes: Vec<i64>,
    offsets: Vec<i64>,
}

pub enum Mixer {
    Gdn(GdnWeights),
    Qsa(QsaWeights),
}

pub struct Layer {
    attn_hc: GatedResidual,
    mlp_hc: GatedResidual,
    mixer: Mixer,
    moe: MoeFfn,
    ple: Option<PleWeights>,
    /// Directory triples for the 512 routed experts.  The dynamic GPU cache
    /// binds by directory index; keeping the table beside the layer avoids
    /// rebuilding 24,576 triples on every token.
    expert_ids: Vec<(usize, usize, usize)>,
    /// Qwen carries one gated shared expert per layer. It has the same
    /// geometry as a routed expert but owns a pinned cache line, exactly as
    /// the established dynamic DSV4 pool does for its shared branch.
    shared_ids: Option<(usize, usize, usize)>,
}

pub struct Globals {
    embed: QTensor,
    lm_head: QTensor,
    head_hc: GatedResidual,
}

#[derive(Clone)]
pub struct Cfg {
    hidden: usize,
    hc: usize,
    eps: f64,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    index_heads: usize,
    index_kv_heads: usize,
    index_dim: usize,
    index_budget: usize,
    compress_ratio: usize,
    gdn: GdnCfg,
    ngram_size: usize,
    heads_per_ngram: usize,
    ple_kernel: usize,
    ple_dilation: usize,
    eos: u32,
}

#[derive(Default)]
struct QsaState {
    raw_keys: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Default)]
struct LayerState {
    gdn: Vec<f32>,
    qsa: QsaState,
    /// Chronological normalized PLE values, at most (kernel-1)*dilation rows.
    ple_history: Vec<f32>,
    ple_history_rows: usize,
}

pub struct State {
    hyper: Vec<f32>,
    layers: Vec<LayerState>,
    token_history: Vec<u32>,
    gpu_pool: Option<QwenGpuPool>,
    pub pos: usize,
}

struct QwenGpuPool {
    segment_slots: usize,
    floor: usize,
    n_experts: usize,
    owner: Vec<Option<(usize, usize)>>,
    /// Qwen has one fixed expert count on every layer.  A dense
    /// `[layer][expert]` map avoids hundreds of hash lookups per layer and
    /// makes clearing/rebuilding the cache allocation-free.
    slot_for: Vec<u32>,
    shared_slot: Vec<u32>,
    seen: Vec<u16>,
    seen_epoch: Vec<u32>,
    /// Reverse-filled so `pop()` preserves the old 0,1,2... allocation
    /// order while avoiding an O(capacity) `position(None)` scan per fill.
    free: Vec<usize>,
    occupancy: Vec<usize>,
    last: Vec<u64>,
    clock: u64,
}

impl State {
    pub fn new(n_layers: usize) -> Self {
        Self {
            hyper: Vec::new(),
            layers: (0..n_layers).map(|_| LayerState::default()).collect(),
            token_history: Vec::new(),
            gpu_pool: None,
            pos: 0,
        }
    }

    fn reset(&mut self) {
        self.hyper.clear();
        self.token_history.clear();
        self.pos = 0;
        for st in &mut self.layers {
            *st = LayerState::default();
        }
    }
}

#[cfg(feature = "gpu")]
impl QwenGpuPool {
    fn create(
        model: &Arc<CmfModel>,
        inter: usize,
        hidden: usize,
        n_layers: usize,
        n_experts: usize,
        gu_q2: bool,
    ) -> Option<Self> {
        if !crate::gpu_wgpu::dsv4_global_moe_supported() {
            return None;
        }
        let gu = cortiq_core::quant::expected_nbytes(
            if gu_q2 {
                TensorDtype::Q2TiledP
            } else {
                TensorDtype::Q4TiledP
            },
            &[inter, hidden],
        )?;
        let dn = cortiq_core::quant::expected_nbytes(
            cortiq_core::TensorDtype::Q4TiledP,
            &[hidden, inter],
        )?;
        let per = 2usize.checked_mul(gu)?.checked_add(dn)?;
        let budget = crate::gpu_wgpu::dsv4_vram_budget()? as usize;
        // The Q8_2f attention/GDN skeleton, f32 HyperConnection projections,
        // KV/state and the full-vocabulary head live next to this arena. The
        // global allocator subtracts another 2-4 GiB workspace below this
        // request. 75% therefore becomes a ~50% expert arena on 8 GiB, ~62%
        // on 16 GiB and ~70% on 80 GiB: enough Qwen locality without stealing
        // the geometry-independent driver/KV/frame reserve. The old 55%
        // request left only 6.7 GiB of experts under a 16 GiB budget and lost
        // 12-15% decode to avoidable cold completions.
        let pool_pct = std::env::var("CMF_QWEN_POOL_PCT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(75)
            .clamp(25, 85);
        let requested = budget.saturating_mul(pool_pct) / 100 / per.max(1);
        let requested = std::env::var("CMF_QWEN_EXPERT_SLOTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(requested)
            .max(8);
        let (capacity, segment_slots) =
            crate::gpu_wgpu::dsv4_global_moe_create(model, requested, inter, hidden, gu_q2)?;
        if std::env::var_os("CMF_QWEN_PROF").is_some() {
            eprintln!(
                "qwen-pool capacity={capacity} segment_slots={segment_slots} requested={requested} gu={}",
                if gu_q2 { "q2tp" } else { "q4tp" }
            );
        }
        Some(Self {
            segment_slots,
            floor: (capacity / n_layers.max(1)).max(2),
            n_experts,
            owner: vec![None; capacity],
            slot_for: vec![u32::MAX; n_layers.checked_mul(n_experts)?],
            shared_slot: vec![u32::MAX; n_layers],
            seen: vec![0; n_layers.checked_mul(n_experts)?],
            seen_epoch: vec![0; n_layers.checked_mul(n_experts)?],
            free: (0..capacity).rev().collect(),
            occupancy: vec![0; n_layers.max(1)],
            last: vec![0; capacity],
            clock: 0,
        })
    }

    fn ensure(
        &mut self,
        model: &Arc<CmfModel>,
        layer: usize,
        picks: &[usize],
        triples: &[(usize, usize, usize)],
        shared: Option<(usize, usize, usize)>,
    ) -> Option<(Vec<u32>, u32)> {
        self.clock = self.clock.saturating_add(1);
        let now = self.clock;
        let shared_slot = if let Some(triple) = shared {
            let slot = *self.shared_slot.get(layer)?;
            if slot != u32::MAX {
                self.last[slot as usize] = now;
                slot
            } else {
                let slot = self.free.pop().or_else(|| {
                    self.owner
                        .iter()
                        .enumerate()
                        // A shared expert is pinned for the model lifetime.
                        .filter(|(_, o)| o.is_some_and(|(_, e)| e != usize::MAX))
                        .min_by_key(|(slot, _)| self.last[*slot])
                        .map(|(slot, _)| slot)
                })?;
                if !crate::gpu_wgpu::dsv4_global_slot_fill(model, slot, triple) {
                    if self.owner[slot].is_none() {
                        self.free.push(slot);
                    }
                    return None;
                }
                if let Some(old) = self.owner[slot] {
                    if old.1 == usize::MAX {
                        self.shared_slot[old.0] = u32::MAX;
                    } else {
                        self.slot_for[old.0 * self.n_experts + old.1] = u32::MAX;
                        self.occupancy[old.0] = self.occupancy[old.0].saturating_sub(1);
                    }
                }
                self.owner[slot] = Some((layer, usize::MAX));
                self.shared_slot[layer] = slot as u32;
                self.occupancy[layer] += 1;
                self.last[slot] = now;
                slot as u32
            }
        } else {
            0
        };
        // The HashMap implementation decayed every observed key by scanning
        // the whole table each 64 layer calls. With a dense 48×512 table that
        // scan is unnecessary: apply the same shifts lazily when a key is
        // next touched. Admission only examines current picks, so behaviour is
        // identical while idle experts cost zero work.
        let epoch = (now / 64).min(u32::MAX as u64) as u32;
        let base = layer.checked_mul(self.n_experts)?;
        for &expert in picks {
            if expert >= self.n_experts {
                return None;
            }
            let key = base + expert;
            let delta = epoch.saturating_sub(self.seen_epoch[key]);
            self.seen[key] = if delta >= u16::BITS {
                0
            } else {
                self.seen[key] >> delta
            };
            self.seen_epoch[key] = epoch;
            self.seen[key] = self.seen[key].saturating_add(1);
            let slot = self.slot_for[key];
            if slot != u32::MAX {
                self.last[slot as usize] = now;
            }
        }
        let (auto_quota, auto_min_seen) = crate::gpu_wgpu::dsv4_fetch_defaults();
        let quota = std::env::var("CMF_QWEN_FETCH_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(auto_quota);
        let min_seen = std::env::var("CMF_QWEN_FETCH_MIN_SEEN")
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            // With 512 experts/layer, first sightings churn the cache even on
            // a fast PCIe link. Both 16 and 32 GiB A/B runs were faster at 2;
            // an explicit operator value still wins for unusual hardware.
            .unwrap_or(auto_min_seen.max(2));
        let mut fetched = 0usize;
        for &expert in picks {
            let key = base + expert;
            if self.slot_for[key] != u32::MAX || fetched >= quota || self.seen[key] < min_seen {
                continue;
            }
            let triple = *triples.get(expert)?;
            let eligible = |owner: (usize, usize)| {
                owner.1 != usize::MAX && (owner.0 != layer || !picks.contains(&owner.1))
            };
            let victim = self
                .free
                .pop()
                .or_else(|| {
                    // Preserve a per-layer working set. A plain global LRU
                    // collapses under the deterministic 0..47 layer sweep:
                    // late layers evict early ones immediately before their
                    // next visit (the same failure measured in DSV4).
                    self.owner
                        .iter()
                        .enumerate()
                        .filter_map(|(slot, &o)| {
                            o.filter(|&x| {
                                eligible(x)
                                    && self.occupancy.get(x.0).copied().unwrap_or(0) > self.floor
                            })
                            .map(|_| slot)
                        })
                        .min_by_key(|&slot| self.last[slot])
                })
                .or_else(|| {
                    self.owner
                        .iter()
                        .enumerate()
                        .filter_map(|(slot, &o)| {
                            o.filter(|&x| eligible(x) && x.0 == layer).map(|_| slot)
                        })
                        .min_by_key(|&slot| self.last[slot])
                })
                .or_else(|| {
                    self.owner
                        .iter()
                        .enumerate()
                        .filter_map(|(slot, &o)| o.filter(|&x| eligible(x)).map(|_| slot))
                        .min_by_key(|&slot| self.last[slot])
                })?;
            if !crate::gpu_wgpu::dsv4_global_slot_fill(model, victim, triple) {
                if self.owner[victim].is_none() {
                    self.free.push(victim);
                }
                return None;
            }
            if let Some(old) = self.owner[victim] {
                if old.1 == usize::MAX {
                    self.shared_slot[old.0] = u32::MAX;
                } else {
                    self.slot_for[old.0 * self.n_experts + old.1] = u32::MAX;
                    self.occupancy[old.0] = self.occupancy[old.0].saturating_sub(1);
                }
            }
            self.owner[victim] = Some((layer, expert));
            self.slot_for[key] = victim as u32;
            self.occupancy[layer] += 1;
            self.last[victim] = now;
            fetched += 1;
        }
        if triples.len() != self.n_experts {
            return None;
        }
        let remap = self.slot_for[base..base + self.n_experts].to_vec();
        Some((remap, shared_slot))
    }
}

fn err(s: impl Into<String>) -> CmfError {
    CmfError::Parse(format!("qwen4_exp: {}", s.into()))
}

fn f(model: &CmfModel, name: &str) -> Result<Vec<f32>, CmfError> {
    load_f32(model, name, &Overlay::None).map_err(err)
}

fn t(model: &Arc<CmfModel>, name: &str) -> Result<QTensor, CmfError> {
    load_matrix(model, name, false, &Overlay::None)
}

fn load_hc(model: &Arc<CmfModel>, prefix: &str, inject: bool) -> Result<GatedResidual, CmfError> {
    Ok(GatedResidual {
        norm: f(model, &format!("{prefix}hc_norm.weight"))?,
        down: t(model, &format!("{prefix}input_mix_weight_down.weight"))?,
        up: t(model, &format!("{prefix}input_mix_weight_up.weight"))?,
        inject: inject
            .then(|| t(model, &format!("{prefix}block_inject_weight.weight")))
            .transpose()?,
    })
}

fn load_gdn(model: &Arc<CmfModel>, prefix: &str) -> Result<GdnWeights, CmfError> {
    Ok(GdnWeights {
        in_proj_qkv: t(model, &format!("{prefix}in_proj_qkv.weight"))?,
        in_proj_z: t(model, &format!("{prefix}in_proj_z.weight"))?,
        in_proj_a: t(model, &format!("{prefix}in_proj_a.weight"))?,
        in_proj_b: t(model, &format!("{prefix}in_proj_b.weight"))?,
        conv1d: f(model, &format!("{prefix}conv1d.weight"))?,
        a_log: f(model, &format!("{prefix}A_log"))?,
        dt_bias: f(model, &format!("{prefix}dt_bias"))?,
        norm: f(model, &format!("{prefix}norm.weight"))?,
        out_proj: t(model, &format!("{prefix}out_proj.weight"))?,
    })
}

fn load_qsa(model: &Arc<CmfModel>, prefix: &str) -> Result<QsaWeights, CmfError> {
    Ok(QsaWeights {
        q_proj: t(model, &format!("{prefix}q_proj.weight"))?,
        k_proj: t(model, &format!("{prefix}k_proj.weight"))?,
        v_proj: t(model, &format!("{prefix}v_proj.weight"))?,
        o_proj: t(model, &format!("{prefix}o_proj.weight"))?,
        q_norm: f(model, &format!("{prefix}q_norm.weight"))?,
        k_norm: f(model, &format!("{prefix}k_norm.weight"))?,
        index_qk: t(model, &format!("{prefix}indexer.index_qk_proj.weight"))?,
        index_q_norm: f(model, &format!("{prefix}indexer.q_norm.weight"))
            .or_else(|_| f(model, &format!("{prefix}indexer.q_layernorm.weight")))?,
        index_k_norm: f(model, &format!("{prefix}indexer.k_norm.weight"))
            .or_else(|_| f(model, &format!("{prefix}indexer.k_layernorm.weight")))?,
    })
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(SPLITMIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2);
    value ^ (value >> 31)
}

fn is_prime(value: usize) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut d = 3usize;
    while d <= value / d {
        if value.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

fn nth_prime_after(start: usize, count: usize) -> usize {
    let mut p = start;
    for _ in 0..count {
        p += 1;
        while !is_prime(p) {
            p += 1;
        }
    }
    p
}

fn ple_tables(
    qc: &Qwen4ExpConfig,
    vocab: usize,
    ple_index: usize,
) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let max_mul = i64::MAX as u64 / vocab.max(1) as u64;
    let half_bound = (max_mul / 2).max(1);
    let base = qc.seed.wrapping_add(PRIME_1.wrapping_mul(ple_index as u64));
    let multipliers = (0..qc.ngram_size)
        .map(|i| {
            let v = base.wrapping_add(SPLITMIX_GAMMA.wrapping_mul((i + 1) as u64));
            (2 * (splitmix64(v) % half_bound) + 1) as i64
        })
        .collect();
    let heads = (qc.ngram_size - 1) * qc.heads_per_ngram;
    let mut sizes = Vec::with_capacity(heads);
    let mut offsets = Vec::with_capacity(heads);
    let mut off = 0i64;
    for head in 0..heads {
        let global = ple_index * heads + head;
        let sz = nth_prime_after(qc.ngram_vocab_size_base - 1, global + 1) as i64;
        sizes.push(sz);
        offsets.push(off);
        off += sz;
    }
    (multipliers, sizes, offsets)
}

fn load_ple(
    model: &Arc<CmfModel>,
    prefix: &str,
    qc: &Qwen4ExpConfig,
    vocab: usize,
    ple_index: usize,
) -> Result<PleWeights, CmfError> {
    let mut shards = Vec::with_capacity(qc.split_ngram_parts);
    for si in 0..qc.split_ngram_parts {
        shards.push(t(
            model,
            &format!("{prefix}ple_embedding.ngram_embedding.shard_{si}.weight"),
        )?);
    }
    let first = shards
        .first()
        .ok_or_else(|| err("PLE has no embedding shards"))?;
    let (rows_per_shard, row_dim) = (first.rows(), first.cols());
    if shards
        .iter()
        .any(|s| s.rows() != rows_per_shard || s.cols() != row_dim)
    {
        return Err(err("PLE embedding shard shapes disagree"));
    }
    let (multipliers, vocab_sizes, offsets) = ple_tables(qc, vocab, ple_index);
    Ok(PleWeights {
        shards,
        rows_per_shard,
        row_dim,
        key_proj: t(model, &format!("{prefix}key_proj.weight"))?,
        value_proj: t(model, &format!("{prefix}value_proj.weight"))?,
        norm_key: f(model, &format!("{prefix}norm_key.weight"))?,
        norm_query: f(model, &format!("{prefix}norm_query.weight"))?,
        norm_conv: f(model, &format!("{prefix}norm_conv.weight"))?,
        conv: f(model, &format!("{prefix}conv1d.weight"))?,
        multipliers,
        vocab_sizes,
        offsets,
    })
}

/// Load the dedicated stack. All large matrices remain mmap-backed QTensor
/// views, so loading does not allocate a second copy of the model.
pub fn load(
    model: &Arc<CmfModel>,
    arch: &ModelArch,
) -> Result<(Globals, Vec<Layer>, Cfg, State), CmfError> {
    let qc = arch
        .qwen4_exp
        .as_ref()
        .ok_or_else(|| err("missing qwen4_exp descriptor"))?;
    if qc.hc_count == 0 || qc.indexer_compress_ratio == 0 || qc.ngram_size < 2 {
        return Err(err("invalid zero geometry"));
    }
    let gdn = GdnCfg {
        num_v_heads: arch.linear_num_value_heads.unwrap_or(48),
        num_k_heads: arch.linear_num_key_heads.unwrap_or(16),
        key_head_dim: arch.linear_key_head_dim.unwrap_or(128),
        value_head_dim: arch.linear_value_head_dim.unwrap_or(128),
        conv_kernel: arch.linear_conv_kernel_dim.unwrap_or(4),
        hidden_size: arch.hidden_size,
        rms_eps: arch.rms_norm_eps,
        output_gate_sigmoid: true,
    };
    let eos = model
        .header
        .tokenizer_config
        .as_ref()
        // The generation bundle lists both <|im_end|> and <|endoftext|>.
        // PLE segmentation uses text_config.eos_token_id, which for this
        // release is the pad/BOS id (248044), not the first generation stop.
        .and_then(|tc| {
            tc.pad_token_id
                .or(tc.bos_token_id)
                .or_else(|| tc.eos_token_ids.last().copied())
        })
        .unwrap_or(248_044);
    let cfg = Cfg {
        hidden: arch.hidden_size,
        hc: qc.hc_count,
        eps: arch.rms_norm_eps,
        n_heads: arch.num_attention_heads,
        n_kv_heads: arch.num_kv_heads,
        head_dim: arch.head_dim,
        rotary_dim: ((arch.head_dim as f32 * arch.partial_rotary_factor) as usize).max(2),
        index_heads: qc.indexer_n_heads,
        index_kv_heads: qc.indexer_kv_heads,
        index_dim: qc.indexer_head_dim,
        index_budget: qc.indexer_budget,
        compress_ratio: qc.indexer_compress_ratio,
        gdn,
        ngram_size: qc.ngram_size,
        heads_per_ngram: qc.heads_per_ngram,
        ple_kernel: qc.ple_conv_kernel_size,
        ple_dilation: qc.ngram_size,
        eos,
    };
    if cfg.n_heads % cfg.n_kv_heads != 0 || cfg.index_kv_heads != 1 {
        return Err(err("unsupported QSA head grouping"));
    }

    let globals = Globals {
        embed: t(model, "model.embed_tokens.weight")?,
        lm_head: t(model, "lm_head.weight")?,
        head_hc: load_hc(model, "model.hyper_connection_mixer.", false)?,
    };
    let mut layers = Vec::with_capacity(arch.num_layers);
    let mut ple_index = 0usize;
    for li in 0..arch.num_layers {
        let p = format!("model.layers.{li}.");
        let mixer = match arch.layer_types.get(li) {
            Some(LayerType::LinearAttention) => {
                Mixer::Gdn(load_gdn(model, &format!("{p}linear_attn."))?)
            }
            _ => Mixer::Qsa(load_qsa(model, &format!("{p}self_attn."))?),
        };
        let ple = if qc.ple_layer_ids.contains(&(li + 1)) {
            let w = load_ple(model, &format!("{p}ple."), qc, arch.vocab_size, ple_index)?;
            ple_index += 1;
            Some(w)
        } else {
            None
        };
        let moe = match build_ffn_at(model, arch, &p, false, &Overlay::None)? {
            FfnKind::Moe(m) => m,
            _ => return Err(err(format!("layer {li} is not MoE"))),
        };
        let expert_ids: Vec<_> = moe
            .experts
            .iter()
            .map(|e| {
                Some((
                    e.gate_proj.model_idx()?,
                    e.up_proj.model_idx()?,
                    e.down_proj.model_idx()?,
                ))
            })
            .collect::<Option<_>>()
            .ok_or_else(|| err(format!("layer {li} experts are not mmap-backed")))?;
        let shared_ids = moe
            .shared
            .as_ref()
            .and_then(|(e, _)| {
                Some((
                    e.gate_proj.model_idx()?,
                    e.up_proj.model_idx()?,
                    e.down_proj.model_idx()?,
                ))
            })
            .ok_or_else(|| err(format!("layer {li} shared expert is not mmap-backed")))?;
        layers.push(Layer {
            attn_hc: load_hc(model, &format!("{p}attn_hyper_connection."), true)?,
            mlp_hc: load_hc(model, &format!("{p}mlp_hyper_connection."), true)?,
            mixer,
            moe,
            ple,
            expert_ids,
            shared_ids: Some(shared_ids),
        });
    }
    let state = State::new(layers.len());
    Ok((globals, layers, cfg, state))
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn group_rms_zero_into(x: &[f32], weight: &[f32], group: usize, eps: f64, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());
    debug_assert!(group > 0 && x.len().is_multiple_of(group));
    for (gi, chunk) in x.chunks(group).enumerate() {
        let ss = chunk.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
        let inv = (ss / group as f64 + eps).sqrt().recip() as f32;
        let off = gi * group;
        for j in 0..group {
            out[off + j] = chunk[j] * inv * (1.0 + weight[off + j]);
        }
    }
}

fn group_rms_zero(x: &[f32], weight: &[f32], group: usize, eps: f64) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    group_rms_zero_into(x, weight, group, eps, &mut out);
    out
}

fn hc_mix(
    w: &GatedResidual,
    hyper: &[f32],
    cfg: &Cfg,
    pool: Option<&Pool>,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let mut normed = crate::attention::take_buf(hyper.len());
    group_rms_zero_into(hyper, &w.norm, cfg.hidden, cfg.eps, &mut normed);
    let mut low = crate::attention::take_buf(w.down.rows());
    // `down` and the four-row injection gate read the same normalized
    // 4-stream state. Run their rows under one pool publication: on the
    // Qwen stack this removes 96 barriers per token while preserving each
    // row's exact dot-product order.
    let mut inject = w.inject.as_ref().map(|iw| vec![0.0f32; iw.rows()]);
    match (&w.inject, inject.as_mut()) {
        (Some(iw), Some(inj)) => {
            QTensor::matvec_many([&w.down, iw], &normed, [&mut low, inj], pool)
        }
        _ => w.down.matvec(&normed, &mut low, pool),
    }
    for v in &mut low {
        *v = silu(*v / cfg.hc as f32);
    }
    let mut mix = crate::attention::take_buf(hyper.len());
    w.up.matvec(&low, &mut mix, pool);
    let mut folded = vec![0.0f32; cfg.hidden];
    for stream in 0..cfg.hc {
        let off = stream * cfg.hidden;
        for d in 0..cfg.hidden {
            folded[d] += sigmoid(mix[off + d]) * normed[off + d] / cfg.hc as f32;
        }
    }
    let inject = inject.map(|mut v| {
        for x in &mut v {
            *x = 2.0 * sigmoid(*x / cfg.hc as f32);
        }
        v
    });
    crate::attention::recycle_buf(&mut mix);
    crate::attention::recycle_buf(&mut low);
    crate::attention::recycle_buf(&mut normed);
    (folded, inject)
}

fn inject(hyper: &mut [f32], block: &[f32], weights: &[f32], cfg: &Cfg) {
    for stream in 0..cfg.hc {
        let off = stream * cfg.hidden;
        for d in 0..cfg.hidden {
            hyper[off + d] += weights[stream] * block[d];
        }
    }
}

fn trace_stats(label: &str, li: usize, position: usize, values: &[f32]) {
    if std::env::var_os("CMF_QWEN_TRACE").is_none() {
        return;
    }
    let mut sumsq = 0.0f64;
    let mut max = 0.0f32;
    let mut finite = 0usize;
    for &v in values {
        if v.is_finite() {
            sumsq += (v as f64) * (v as f64);
            max = max.max(v.abs());
            finite += 1;
        }
    }
    let rms = if finite == 0 {
        f64::NAN
    } else {
        (sumsq / finite as f64).sqrt()
    };
    eprintln!(
        "qwen4_exp pos={position} layer={li:02} {label}: rms={rms:.6} max={max:.6} finite={finite}/{}",
        values.len()
    );
}

fn dump_values(label: &str, li: usize, position: usize, values: &[f32]) {
    let Some(root) = std::env::var_os("CMF_QWEN_DUMP") else {
        return;
    };
    if let Ok(wanted) = std::env::var("CMF_QWEN_DUMP_LAYER")
        && wanted.parse::<usize>().ok() != Some(li)
    {
        return;
    }
    let root = std::path::PathBuf::from(root);
    if std::fs::create_dir_all(&root).is_err() {
        return;
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    let _ = std::fs::write(
        root.join(format!("p{position:06}_l{li:02}_{label}.f32")),
        bytes,
    );
}

fn observe(label: &str, li: usize, position: usize, values: &[f32]) {
    trace_stats(label, li, position, values);
    dump_values(label, li, position, values);
}

fn rms_zero_head(v: &mut [f32], weight: &[f32], head_dim: usize, eps: f64) {
    for head in v.chunks_mut(head_dim) {
        let ss = head.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        let inv = (ss / head_dim as f64 + eps).sqrt().recip() as f32;
        for (x, &w) in head.iter_mut().zip(weight) {
            *x *= inv * (1.0 + w);
        }
    }
}

fn rope(v: &mut [f32], pos: usize, inv_freq: &[f32], rotary_dim: usize) {
    let rd = rotary_dim.min(v.len()).min(inv_freq.len() * 2);
    let half = rd / 2;
    for i in 0..half {
        let a = (pos as f32 * inv_freq[i]).cos();
        let b = (pos as f32 * inv_freq[i]).sin();
        let x1 = v[i];
        let x2 = v[i + half];
        v[i] = x1 * a - x2 * b;
        v[i + half] = x2 * a + x1 * b;
    }
}

fn selected_tokens(
    q: &[f32],
    raw_keys: &[f32],
    npos: usize,
    w: &QsaWeights,
    cfg: &Cfg,
    inv_freq: &[f32],
) -> Vec<usize> {
    let cr = cfg.compress_ratio;
    let complete = npos / cr;
    let mut scores = Vec::with_capacity(complete);
    for block in 0..complete {
        let mut k = vec![0.0f32; cfg.index_dim];
        for ti in block * cr..(block + 1) * cr {
            let src = &raw_keys[ti * cfg.index_dim..(ti + 1) * cfg.index_dim];
            for (d, &x) in src.iter().enumerate() {
                k[d] += x / cr as f32;
            }
        }
        rms_zero_head(&mut k, &w.index_k_norm, cfg.index_dim, cfg.eps);
        rope(
            &mut k,
            block * cr,
            inv_freq,
            cfg.rotary_dim.min(cfg.index_dim),
        );
        let mut s = 0.0f32;
        for h in 0..cfg.index_heads {
            let qh = &q[h * cfg.index_dim..(h + 1) * cfg.index_dim];
            let dot = qh.iter().zip(&k).map(|(&a, &b)| a * b).sum::<f32>();
            s += dot.max(0.0);
        }
        scores.push((block, s / (cfg.index_dim as f32).sqrt()));
    }
    let keep = (cfg.index_budget / cr).min(complete);
    if complete > keep {
        scores.select_nth_unstable_by(keep, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        });
        scores.truncate(keep);
    }
    let mut out = Vec::with_capacity(keep * cr + cr.saturating_sub(1));
    for (block, _) in scores {
        out.extend(block * cr..(block + 1) * cr);
    }
    out.extend(complete * cr..npos);
    out
}

fn qsa_forward(
    x: &[f32],
    w: &QsaWeights,
    cfg: &Cfg,
    st: &mut QsaState,
    position: usize,
    inv_freq: &[f32],
    pool: Option<&Pool>,
) -> Vec<f32> {
    let mut iqk =
        crate::attention::take_buf((cfg.index_heads + cfg.index_kv_heads) * cfg.index_dim);
    let mut qg = crate::attention::take_buf(cfg.n_heads * cfg.head_dim * 2);
    let mut k = crate::attention::take_buf(cfg.n_kv_heads * cfg.head_dim);
    let mut v = crate::attention::take_buf(cfg.n_kv_heads * cfg.head_dim);
    // Indexer QK and attention Q/K/V all read the same folded state. Their
    // q8-family rows are independent, so publish one virtual row range to
    // the pool instead of four back-to-back barriers.
    QTensor::matvec_many(
        [&w.index_qk, &w.q_proj, &w.k_proj, &w.v_proj],
        x,
        [&mut iqk, &mut qg, &mut k, &mut v],
        pool,
    );
    let qlen = cfg.index_heads * cfg.index_dim;
    let mut iq = crate::attention::take_buf(qlen);
    iq.copy_from_slice(&iqk[..qlen]);
    rms_zero_head(&mut iq, &w.index_q_norm, cfg.index_dim, cfg.eps);
    for h in 0..cfg.index_heads {
        rope(
            &mut iq[h * cfg.index_dim..(h + 1) * cfg.index_dim],
            position,
            inv_freq,
            cfg.rotary_dim.min(cfg.index_dim),
        );
    }
    st.raw_keys.extend_from_slice(&iqk[qlen..]);

    let mut q = crate::attention::take_buf(cfg.n_heads * cfg.head_dim);
    let mut gate = crate::attention::take_buf(q.len());
    for h in 0..cfg.n_heads {
        let src = h * cfg.head_dim * 2;
        let dst = h * cfg.head_dim;
        q[dst..dst + cfg.head_dim].copy_from_slice(&qg[src..src + cfg.head_dim]);
        gate[dst..dst + cfg.head_dim]
            .copy_from_slice(&qg[src + cfg.head_dim..src + cfg.head_dim * 2]);
    }
    rms_zero_head(&mut q, &w.q_norm, cfg.head_dim, cfg.eps);
    rms_zero_head(&mut k, &w.k_norm, cfg.head_dim, cfg.eps);
    for h in 0..cfg.n_heads {
        rope(
            &mut q[h * cfg.head_dim..(h + 1) * cfg.head_dim],
            position,
            inv_freq,
            cfg.rotary_dim,
        );
    }
    for h in 0..cfg.n_kv_heads {
        rope(
            &mut k[h * cfg.head_dim..(h + 1) * cfg.head_dim],
            position,
            inv_freq,
            cfg.rotary_dim,
        );
    }
    st.keys.extend_from_slice(&k);
    st.values.extend_from_slice(&v);
    let npos = position + 1;
    let selected = selected_tokens(&iq, &st.raw_keys, npos, w, cfg, inv_freq);
    let groups = cfg.n_heads / cfg.n_kv_heads;
    let scale = (cfg.head_dim as f32).sqrt().recip();
    let mut merged = crate::attention::take_buf(cfg.n_heads * cfg.head_dim);
    for qh in 0..cfg.n_heads {
        let kvh = qh / groups;
        let qs = &q[qh * cfg.head_dim..(qh + 1) * cfg.head_dim];
        let mut scores = Vec::with_capacity(selected.len());
        let mut max = f32::NEG_INFINITY;
        for &ti in &selected {
            let ko = (ti * cfg.n_kv_heads + kvh) * cfg.head_dim;
            let s = qs
                .iter()
                .zip(&st.keys[ko..ko + cfg.head_dim])
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
                * scale;
            max = max.max(s);
            scores.push(s);
        }
        let z = scores.iter().map(|&s| (s - max).exp()).sum::<f32>();
        let out = &mut merged[qh * cfg.head_dim..(qh + 1) * cfg.head_dim];
        for (&ti, &score) in selected.iter().zip(&scores) {
            let p = (score - max).exp() / z.max(f32::MIN_POSITIVE);
            let vo = (ti * cfg.n_kv_heads + kvh) * cfg.head_dim;
            for d in 0..cfg.head_dim {
                out[d] += p * st.values[vo + d];
            }
        }
        let go = qh * cfg.head_dim;
        for d in 0..cfg.head_dim {
            out[d] *= sigmoid(gate[go + d]);
        }
    }
    let mut out = vec![0.0f32; cfg.hidden];
    w.o_proj.matvec(&merged, &mut out, pool);
    crate::attention::recycle_buf(&mut merged);
    crate::attention::recycle_buf(&mut gate);
    crate::attention::recycle_buf(&mut q);
    crate::attention::recycle_buf(&mut iq);
    crate::attention::recycle_buf(&mut v);
    crate::attention::recycle_buf(&mut k);
    crate::attention::recycle_buf(&mut qg);
    crate::attention::recycle_buf(&mut iqk);
    out
}

fn shifted_token(history: &[u32], current: u32, shift: usize, eos: u32) -> u32 {
    if shift == 0 {
        return current;
    }
    if history.len() < shift {
        return eos;
    }
    let source = history.len() - shift;
    if history[source + 1..].contains(&eos) || history.last() == Some(&eos) {
        eos
    } else {
        history[source]
    }
}

fn ple_embedding(w: &PleWeights, cfg: &Cfg, history: &[u32], token: u32) -> Vec<f32> {
    let shifted: Vec<i64> = (0..cfg.ngram_size)
        .map(|s| shifted_token(history, token, s, cfg.eos) as i64)
        .collect();
    let mut ids = Vec::with_capacity((cfg.ngram_size - 1) * cfg.heads_per_ngram);
    for ngram in 2..=cfg.ngram_size {
        let mut mixed = shifted[0].wrapping_mul(w.multipliers[0]);
        for p in 1..ngram {
            mixed ^= shifted[p].wrapping_mul(w.multipliers[p]);
        }
        let h0 = (ngram - 2) * cfg.heads_per_ngram;
        for hi in h0..h0 + cfg.heads_per_ngram {
            ids.push(mixed.rem_euclid(w.vocab_sizes[hi]) + w.offsets[hi]);
        }
    }
    let mut out = vec![0.0f32; ids.len() * w.row_dim];
    let mut row = vec![0.0f32; w.row_dim];
    for (hi, &id) in ids.iter().enumerate() {
        let global = id as usize;
        let shard = global / w.rows_per_shard;
        let local = global % w.rows_per_shard;
        debug_assert!(shard < w.shards.len());
        w.shards[shard].row_f32(local, &mut row);
        out[hi * w.row_dim..(hi + 1) * w.row_dim].copy_from_slice(&row);
    }
    out
}

fn ple_forward(
    hyper: &[f32],
    token: u32,
    history: &[u32],
    w: &PleWeights,
    cfg: &Cfg,
    st: &mut LayerState,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let emb = ple_embedding(w, cfg, history, token);
    let mut key = vec![0.0f32; cfg.hc * cfg.hidden];
    let mut value = vec![0.0f32; cfg.hidden];
    w.key_proj.matvec(&emb, &mut key, pool);
    w.value_proj.matvec(&emb, &mut value, pool);
    let key = group_rms_zero(&key, &w.norm_key, cfg.hidden, cfg.eps);
    let query = group_rms_zero(hyper, &w.norm_query, cfg.hidden, cfg.eps);
    let mut gated = vec![0.0f32; cfg.hc * cfg.hidden];
    for stream in 0..cfg.hc {
        let off = stream * cfg.hidden;
        let dot = key[off..off + cfg.hidden]
            .iter()
            .zip(&query[off..off + cfg.hidden])
            .map(|(&a, &b)| a * b)
            .sum::<f32>()
            / (cfg.hidden as f32).sqrt();
        let signed_root = dot.signum() * dot.abs().max(1e-6).sqrt();
        let g = sigmoid(signed_root);
        for d in 0..cfg.hidden {
            gated[off + d] = g * value[d];
        }
    }
    let normed = group_rms_zero(&gated, &w.norm_conv, cfg.hidden, cfg.eps);
    let hist_cap = (cfg.ple_kernel - 1) * cfg.ple_dilation;
    let width = gated.len();
    let mut conv = vec![0.0f32; width];
    for channel in 0..width {
        let mut sum = w.conv[channel * cfg.ple_kernel + cfg.ple_kernel - 1] * normed[channel];
        for tap in 0..cfg.ple_kernel - 1 {
            let lag = (cfg.ple_kernel - 1 - tap) * cfg.ple_dilation;
            if lag <= st.ple_history_rows {
                let row = st.ple_history_rows - lag;
                sum +=
                    w.conv[channel * cfg.ple_kernel + tap] * st.ple_history[row * width + channel];
            }
        }
        conv[channel] = silu(sum);
    }
    if hist_cap > 0 {
        if st.ple_history_rows < hist_cap {
            st.ple_history.extend_from_slice(&normed);
            st.ple_history_rows += 1;
        } else {
            st.ple_history.copy_within(width.., 0);
            let off = (hist_cap - 1) * width;
            st.ple_history[off..off + width].copy_from_slice(&normed);
        }
    }
    for (o, &g) in conv.iter_mut().zip(&gated) {
        *o += g;
    }
    conv
}

#[cfg(feature = "gpu")]
fn dynamic_moe_gpu(
    layer: &Layer,
    li: usize,
    x: &[f32],
    state: &mut State,
    pool: Option<&Pool>,
) -> Option<(Vec<f32>, Vec<usize>)> {
    let m = &layer.moe;
    if !crate::gpu::enabled_here()
        || m.router_sigmoid
        || !m.norm_topk_prob
        || (m.routed_scaling - 1.0).abs() > 1e-9
        || m.per_expert_scale.is_some()
        || m.route_tau.is_some()
        || m.mask.is_some()
    {
        return None;
    }
    let model = m.experts.first()?.gate_proj.model_arc()?;
    let gu_q2 = m
        .experts
        .first()
        .is_some_and(|e| e.gate_proj.model_dtype() == Some(TensorDtype::Q2TiledP));
    let dynamic_mode = std::env::var("CMF_QWEN_DYNAMIC_MOE").ok();
    let dynamic_enabled = match dynamic_mode.as_deref() {
        Some("1") => true,
        Some("0") => false,
        // Q2 gate/up experts are small enough for a useful model-wide cache
        // on a 16 GB class card. Q4 moves twice as much cold data and measured
        // substantially slower than its CPU/GPU auto plan, so it remains
        // opt-in. Unsupported backends and smaller cards fall through to the
        // exact CPU path; this also makes the same artifact safe on Metal.
        None | Some("auto") => {
            gu_q2
                && crate::gpu_wgpu::dsv4_global_moe_supported()
                && crate::gpu_wgpu::dsv4_vram_budget().is_some_and(|b| b >= 14_000_000_000)
        }
        Some(_) => false,
    };
    if !dynamic_enabled {
        return None;
    }
    if state.gpu_pool.is_none() {
        state.gpu_pool = QwenGpuPool::create(
            &model,
            m.experts.first()?.gate_proj.rows(),
            x.len(),
            state.layers.len(),
            m.experts.len(),
            gu_q2,
        );
    }
    let mut logits = vec![0.0f32; m.experts.len()];
    m.router.matvec(x, &mut logits, pool);
    let (picks, probabilities, wsum) = crate::pipeline::moe_route(&logits, m, None);
    let (remap, shared_slot) =
        state
            .gpu_pool
            .as_mut()?
            .ensure(&model, li, &picks, &layer.expert_ids, layer.shared_ids)?;
    let shared_weight = m.shared.as_ref().map_or(1.0, |(_, gate)| {
        gate.as_ref().map_or(1.0, |gate| {
            let mut y = [0.0f32; 1];
            gate.matvec(x, &mut y, pool);
            sigmoid(y[0])
        })
    });
    let has_shared = m.shared.is_some();
    let mut mix_weights = vec![0.0f32; m.experts.len()];
    for &expert in &picks {
        mix_weights[expert] =
            probabilities[expert] / wsum * m.per_expert_scale.as_ref().map_or(1.0, |v| v[expert]);
    }
    let cold_ids: Vec<_> = picks
        .iter()
        .copied()
        .filter(|&expert| remap[expert] == u32::MAX)
        .collect();
    let cold_jobs: Vec<_> = cold_ids
        .iter()
        .copied()
        .map(|expert| (&m.experts[expert], mix_weights[expert]))
        .collect();
    let gp = state.gpu_pool.as_ref()?;
    let weights = crate::gpu_wgpu::Dsv4MoeW {
        router: &[],
        experts: &layer.expert_ids,
        logits: &logits,
        // With forced ids this is a weight table, not selection bias. The
        // preweighted flag keeps the shader from exponentiating/reducing it.
        bias: Some(&mix_weights),
        mask: None,
        forced: Some(&picks),
        remap: Some(&remap),
        global: Some(crate::gpu_wgpu::Dsv4GlobalMoe {
            pool_uid: model.uid(),
            shared_slot,
            segment_slots: gp.segment_slots as u32,
        }),
        has_shared,
        shared_weight,
        preweighted: true,
        qwen_softmax: true,
    };
    let geom = crate::gpu_wgpu::Dsv4MoeGeom {
        hidden: x.len(),
        inter: m.experts.first()?.gate_proj.rows(),
        top_k: m.top_k,
        route_scale: 1.0,
        swiglu_limit: 0.0,
        gu_q2,
    };
    let mut out = vec![0.0f32; x.len()];
    let mut cold = Vec::new();
    let mut cold_x = Vec::new();
    let (frame_ok, mut cold_cpu) = std::thread::scope(|scope| {
        let cpu = (!cold_jobs.is_empty())
            .then(|| scope.spawn(|| crate::pipeline::moe_cold_experts_cpu(&cold_jobs, x, pool)));
        let ok = crate::gpu_wgpu::dsv4_moe_frame(
            &model,
            &weights,
            geom,
            x,
            &mut cold,
            &mut cold_x,
            None,
            None,
            &mut out,
        );
        let early = cpu
            .map(|job| job.join().ok())
            .flatten()
            .unwrap_or_else(|| vec![0.0; x.len()]);
        (ok, early)
    });
    if !frame_ok
        || cold.len() != cold_jobs.len()
        || cold.iter().map(|&(e, _)| e).ne(cold_ids.iter().copied())
    {
        return None;
    }
    // The GPU returned the cold ids as a contract check; the CPU work used
    // the exact host route and ran concurrently with the resident kernels.
    for (o, c) in out.iter_mut().zip(&cold_cpu) {
        *o += c;
    }
    crate::attention::recycle_buf(&mut cold_cpu);
    Some((out, picks))
}

/// Decode one token and return full-vocabulary logits. The state is entirely
/// host-owned; GPU use is opportunistic per projection and therefore safe to
/// change between requests or even between layers.
pub fn forward_token(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    token_id: u32,
    position: usize,
    inv_freq: &[f32],
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
    want_logits: bool,
) {
    let prof = std::env::var_os("CMF_QWEN_PROF").is_some();
    #[cfg(target_arch = "x86_64")]
    if position == 0
        && std::env::var_os("CMF_POOL_SPIN").is_none()
        && let Some(workers) = pool
    {
        // Qwen4Exp publishes hundreds of short HC/router/q8 jobs per token.
        // On the 20-worker 4090 pod, letting workers park between them costs
        // 3.5 tok/s; 200k bounded spins sustains 7.8-7.9. Scale with the real
        // pool and cap it. ARM keeps the established 4k default (200k was a
        // measured regression on Apple silicon); an explicit env always wins.
        workers.set_spin_budget((workers.n_workers() * 10_000).clamp(30_000, 200_000));
    }
    #[cfg(feature = "gpu")]
    let gpu_moe_before = if prof {
        use std::sync::atomic::Ordering;
        Some((
            crate::gpu_wgpu::MOE_ENC_NS.load(Ordering::Relaxed),
            crate::gpu_wgpu::MOE_WAIT_NS.load(Ordering::Relaxed),
            crate::gpu_wgpu::MOE_GPU_NS[0].load(Ordering::Relaxed),
            crate::gpu_wgpu::MOE_GPU_N.load(Ordering::Relaxed),
            crate::gpu_wgpu::DSV4_FILLS.load(Ordering::Relaxed),
            crate::gpu_wgpu::DSV4_FILL_BYTES.load(Ordering::Relaxed),
        ))
    } else {
        None
    };
    let token_t0 = std::time::Instant::now();
    let mut ple_dt = std::time::Duration::ZERO;
    let mut attn_hc_dt = std::time::Duration::ZERO;
    let mut mixer_dt = std::time::Duration::ZERO;
    let mut mlp_hc_dt = std::time::Duration::ZERO;
    let mut moe_dt = std::time::Duration::ZERO;
    if position == 0 || state.pos != position {
        state.reset();
    }
    // Every token starts from its own embedding. Only recurrent/KV/PLE
    // caches cross token boundaries; carrying the prior token's final hyper
    // state here would turn the Transformer residual into an accidental RNN.
    let mut emb = vec![0.0f32; cfg.hidden];
    if (token_id as usize) < globals.embed.rows() {
        globals.embed.row_f32(token_id as usize, &mut emb);
    }
    observe("embedding", 0, position, &emb);
    state.hyper.clear();
    state.hyper.reserve(cfg.hc * cfg.hidden);
    for _ in 0..cfg.hc {
        state.hyper.extend_from_slice(&emb);
    }
    for (li, layer) in layers.iter().enumerate() {
        // Feed the logical layer into the shared residency manager. This lets
        // Vulkan/DX12/Metal keep the hottest projections in VRAM while older
        // layers fall back to the mmap-backed CPU representation.
        crate::gpu::set_layer(li as i64);
        let st = &mut state.layers[li];
        if let Some(ple) = &layer.ple
            && std::env::var_os("CMF_QWEN_NO_PLE").is_none()
        {
            let t0 = std::time::Instant::now();
            let side = ple_forward(
                &state.hyper,
                token_id,
                &state.token_history,
                ple,
                cfg,
                st,
                pool,
            );
            for (h, s) in state.hyper.iter_mut().zip(side) {
                *h += s;
            }
            ple_dt += t0.elapsed();
            observe("post_ple", li, position, &state.hyper);
        }
        let t0 = std::time::Instant::now();
        let (mixed, inject_w) = hc_mix(&layer.attn_hc, &state.hyper, cfg, pool);
        attn_hc_dt += t0.elapsed();
        observe("attn_in", li, position, &mixed);
        let t0 = std::time::Instant::now();
        let block = match &layer.mixer {
            Mixer::Gdn(w) => gdn_forward(&mixed, w, &cfg.gdn, &mut st.gdn, pool),
            Mixer::Qsa(w) => qsa_forward(&mixed, w, cfg, &mut st.qsa, position, inv_freq, pool),
        };
        mixer_dt += t0.elapsed();
        observe("attn_out", li, position, &block);
        inject(
            &mut state.hyper,
            &block,
            inject_w.as_deref().expect("layer HC has injection"),
            cfg,
        );
        observe("post_attn", li, position, &state.hyper);
        let t0 = std::time::Instant::now();
        let (mixed, inject_w) = hc_mix(&layer.mlp_hc, &state.hyper, cfg, pool);
        mlp_hc_dt += t0.elapsed();
        observe("moe_in", li, position, &mixed);
        let t0 = std::time::Instant::now();
        #[cfg(feature = "gpu")]
        let gpu_moe = dynamic_moe_gpu(layer, li, &mixed, state, pool);
        #[cfg(not(feature = "gpu"))]
        let gpu_moe: Option<(Vec<f32>, Vec<usize>)> = None;
        let block = match gpu_moe {
            Some((block, _)) => block,
            None => {
                // `moe_ffn` already routed once. The old implementation ran
                // the 512×hidden router a SECOND time merely to predict a
                // future cache fill, but the dynamic path computes the exact
                // current route before dispatch and never consumed that
                // prediction. Removing it saves 48 matrix passes per token.
                moe_ffn(&layer.moe, &mixed, pool, None)
            }
        };
        moe_dt += t0.elapsed();
        observe("moe_out", li, position, &block);
        inject(
            &mut state.hyper,
            &block,
            inject_w.as_deref().expect("layer HC has injection"),
            cfg,
        );
        observe("post_moe", li, position, &state.hyper);
    }
    crate::gpu::set_layer(-1);
    let head_t0 = std::time::Instant::now();
    if want_logits {
        let (hidden, _) = hc_mix(&globals.head_hc, &state.hyper, cfg, pool);
        observe("head_in", layers.len(), position, &hidden);
        logits.resize(globals.lm_head.rows(), 0.0);
        globals.lm_head.matvec(&hidden, logits, pool);
        observe("logits", layers.len(), position, logits);
    } else {
        logits.clear();
    }
    let head_dt = head_t0.elapsed();
    state.token_history.push(token_id);
    state.pos = position + 1;
    if prof {
        eprintln!(
            "qwen-prof pos={position} total={:.3}s ple={:.3}s attn_hc={:.3}s mixer={:.3}s mlp_hc={:.3}s moe={:.3}s head={:.3}s",
            token_t0.elapsed().as_secs_f64(),
            ple_dt.as_secs_f64(),
            attn_hc_dt.as_secs_f64(),
            mixer_dt.as_secs_f64(),
            mlp_hc_dt.as_secs_f64(),
            moe_dt.as_secs_f64(),
            head_dt.as_secs_f64(),
        );
        #[cfg(feature = "gpu")]
        if let Some((enc0, wait0, card0, calls0, fills0, fill_bytes0)) = gpu_moe_before {
            use std::sync::atomic::Ordering;
            let enc = crate::gpu_wgpu::MOE_ENC_NS.load(Ordering::Relaxed) - enc0;
            let wait = crate::gpu_wgpu::MOE_WAIT_NS.load(Ordering::Relaxed) - wait0;
            let card = crate::gpu_wgpu::MOE_GPU_NS[0].load(Ordering::Relaxed) - card0;
            let calls = crate::gpu_wgpu::MOE_GPU_N.load(Ordering::Relaxed) - calls0;
            let fills = crate::gpu_wgpu::DSV4_FILLS.load(Ordering::Relaxed) - fills0;
            let fill_bytes = crate::gpu_wgpu::DSV4_FILL_BYTES.load(Ordering::Relaxed) - fill_bytes0;
            eprintln!(
                "qwen-moe-gpu pos={position} calls={calls} fills={fills} fill={:.1}MB encode={:.1}ms wait={:.1}ms card={:.1}ms",
                fill_bytes as f64 / 1e6,
                enc as f64 / 1e6,
                wait as f64 / 1e6,
                card as f64 / 1e6,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_hash_tables_match_contract() {
        let q = Qwen4ExpConfig {
            hc_count: 4,
            hc_lowrank: 320,
            indexer_n_heads: 4,
            indexer_kv_heads: 1,
            indexer_head_dim: 128,
            indexer_budget: 2048,
            indexer_compress_ratio: 4,
            ple_layer_ids: vec![2],
            ple_embed_dim: 2560,
            ple_conv_kernel_size: 4,
            ngram_size: 3,
            heads_per_ngram: 8,
            ngram_vocab_size_base: 20_000_000,
            make_ngram_vocab_size_divisible_by: 128,
            split_ngram_parts: 128,
            seed: 1234,
        };
        let (m, sizes, offsets) = ple_tables(&q, 248_320, 0);
        assert_eq!(
            m,
            [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
        );
        assert!(m.iter().all(|x| x & 1 == 1));
        assert_eq!(sizes.len(), 16);
        assert!(sizes.iter().all(|&x| is_prime(x as usize)));
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], sizes[0]);
    }

    #[test]
    fn eos_breaks_ngram_context_only_after_it() {
        let eos = 99;
        assert_eq!(shifted_token(&[1, 2], 3, 1, eos), 2);
        assert_eq!(shifted_token(&[1, eos], 3, 1, eos), eos);
        assert_eq!(shifted_token(&[1, 2], eos, 1, eos), 2);
        assert_eq!(shifted_token(&[], 3, 2, eos), eos);
    }
}
