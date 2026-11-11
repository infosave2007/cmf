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
    pub(crate) norm: Vec<f32>,
    pub(crate) down: QTensor,
    pub(crate) up: QTensor,
    pub(crate) inject: Option<QTensor>,
    /// Directory indices of the same three matrices: the device frame binds
    /// the file bytes directly (f16/f32/q8_2f), the host path keeps the
    /// dequantized `QTensor` views above.
    down_idx: Option<usize>,
    up_idx: Option<usize>,
    inject_idx: Option<usize>,
}

/// Directory indices of every skeleton matrix the device frame binds for
/// one layer. `None` when a tensor is missing from the file: the device
/// path then stays off and the host path runs as before.
struct LayerIdx {
    /// qkv, z, a, b, out
    gdn: Option<[usize; 5]>,
    /// q, k, v, o, index_qk
    qsa: Option<[usize; 5]>,
    /// key_proj, value_proj
    ple: Option<(usize, usize)>,
    router: usize,
    shared_gate: Option<usize>,
}

pub struct QsaWeights {
    pub(crate) q_proj: QTensor,
    pub(crate) k_proj: QTensor,
    pub(crate) v_proj: QTensor,
    pub(crate) o_proj: QTensor,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) index_qk: QTensor,
    pub(crate) index_q_norm: Vec<f32>,
    pub(crate) index_k_norm: Vec<f32>,
}

pub struct PleWeights {
    shards: Vec<QTensor>,
    rows_per_shard: usize,
    row_dim: usize,
    pub(crate) key_proj: QTensor,
    pub(crate) value_proj: QTensor,
    pub(crate) norm_key: Vec<f32>,
    pub(crate) norm_query: Vec<f32>,
    pub(crate) norm_conv: Vec<f32>,
    pub(crate) conv: Vec<f32>,
    multipliers: Vec<i64>,
    vocab_sizes: Vec<i64>,
    offsets: Vec<i64>,
}

pub enum Mixer {
    Gdn(GdnWeights),
    Qsa(QsaWeights),
}

pub struct Layer {
    pub(crate) attn_hc: GatedResidual,
    pub(crate) mlp_hc: GatedResidual,
    pub(crate) mixer: Mixer,
    pub(crate) moe: MoeFfn,
    pub(crate) ple: Option<PleWeights>,
    /// Directory triples for the 512 routed experts.  The dynamic GPU cache
    /// binds by directory index; keeping the table beside the layer avoids
    /// rebuilding 24,576 triples on every token.
    expert_ids: Vec<(usize, usize, usize)>,
    /// Qwen carries one gated shared expert per layer. It has the same
    /// geometry as a routed expert but owns a pinned cache line, exactly as
    /// the established dynamic DSV4 pool does for its shared branch.
    shared_ids: Option<(usize, usize, usize)>,
    idx: Option<LayerIdx>,
}

impl Layer {
    /// (routed experts, intermediate width of one) — `(0, 0)` for a layer
    /// without experts.
    pub fn expert_geometry(&self) -> (usize, usize) {
        let inter = self.moe.experts.first().map_or(0, |d| d.gate_proj.rows());
        (self.moe.experts.len(), inter)
    }
}

pub struct Globals {
    pub(crate) embed: QTensor,
    pub(crate) lm_head: QTensor,
    pub(crate) head_hc: GatedResidual,
    lm_head_idx: Option<usize>,
    /// The embedding table, re-read on the card by the MTP draft chain.
    embed_idx: Option<usize>,
    /// Every skeleton matrix the device frame reads, for the one-time
    /// upload-and-pin before the expert arena takes the rest of the card.
    skeleton_idxs: Vec<usize>,
}

#[derive(Clone)]
pub struct Cfg {
    pub(crate) hidden: usize,
    pub(crate) hc: usize,
    pub(crate) eps: f64,
    pub(crate) n_heads: usize,
    pub(crate) n_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) index_heads: usize,
    pub(crate) index_kv_heads: usize,
    pub(crate) index_dim: usize,
    pub(crate) index_budget: usize,
    pub(crate) compress_ratio: usize,
    pub(crate) gdn: GdnCfg,
    ngram_size: usize,
    heads_per_ngram: usize,
    pub(crate) ple_kernel: usize,
    pub(crate) ple_dilation: usize,
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
    /// Device-resident token path (`gpu_wgpu::qwen4`): the hyper state,
    /// GDN/QSA/PLE caches and the frame scratch live on the card.
    #[cfg(feature = "gpu")]
    dev: Option<crate::gpu_wgpu::qwen4::Dev>,
    /// Routed winners per layer from the previous token: the arena admits
    /// experts one token late, because the route is decided on the card
    /// and only read back with the cold list.
    picks_prev: Vec<Vec<usize>>,
    /// The device path refused once (setup or mid-token); stay on the host.
    device_off: bool,
    /// Positions the host-side caches (`layers`) hold. The device path
    /// advances `pos` and `token_history` but keeps its caches on the card,
    /// so after it turns off mid-sequence this lags `pos` and the host path
    /// replays the history first.
    host_pos: usize,
    /// The CUDA device path (`gpu_cuda::qwen4`): the whole stack on the card
    /// through cuBLAS, taken instead of the wgpu one under `CMF_GPU=cuda`.
    #[cfg(feature = "cuda")]
    cuda: Option<crate::gpu_cuda::qwen4::Dev>,
    /// The CUDA path refused (setup or mid-sequence); stay off it.
    cuda_off: bool,
    /// Check mode (`CMF_QWEN_CUDA_CHECK=1`): the host path's own state over
    /// the same tokens, to compare every stage against.
    #[cfg(feature = "cuda")]
    cuda_ref: Option<Box<State>>,
    /// The copy cursor of the CUDA path's speculation rounds.
    #[cfg(feature = "cuda")]
    cuda_copy: CopyCur,
    #[cfg(feature = "gpu")]
    profile: Option<ExpertProfile>,
    /// The next device forward is a verify window: snapshot the recurrent
    /// state per token and return every token's logits in `window_rows`.
    verify_window: bool,
    window_rows: Option<WindowRows>,
    /// When the last speculative round ended (the host time between rounds, under the profiler).
    round_end: Option<std::time::Instant>,
    #[cfg(feature = "gpu")]
    mtp: Option<MtpHead>,
    mtp_tried: bool,
    /// Every routed expert of every layer holds an arena slot (a model cut
    /// to fit the card, `moe-defrag --keep`): no winner can be cold, so a
    /// whole token, prompt chunks included, goes out as one chained
    /// submission instead of one fenced frame per layer.
    #[cfg(feature = "gpu")]
    resident_all: bool,
    /// A prompt chunk still running on the card (resident mode): its
    /// readback is collected when the next chunk has been submitted, so the
    /// card works while the host encodes. Drained before anything reads
    /// device results.
    #[cfg(feature = "gpu")]
    inflight: Option<crate::gpu_wgpu::qwen4::Pending>,
    /// This sequence's routing so far, `[layer][expert]` winner counts: the prompt's own
    /// narrow bundle of experts, and the pack router's signal (`prompt_done`).
    #[cfg(feature = "gpu")]
    route_hist: Vec<u32>,
    #[cfg(feature = "gpu")]
    route_tokens: usize,
    /// `route_hist` when the φ dump last wrote a window (`CMF_QWEN_PHI_DUMP`).
    #[cfg(feature = "gpu")]
    phi_last: Vec<u32>,
    #[cfg(feature = "gpu")]
    phi_last_tokens: usize,
    /// `route_hist` / `route_tokens` at the last request-phase boundary
    /// (`CMF_QWEN_ROUTE_ROWS`); `row_gen`: the prompt is done, decode routing accrues.
    #[cfg(feature = "gpu")]
    row_snap: Vec<u32>,
    #[cfg(feature = "gpu")]
    row_snap_tokens: usize,
    #[cfg(feature = "gpu")]
    row_gen: bool,
    /// The expert core (`CMF_QWEN_CORE`) is pinned in the arena.
    #[cfg(feature = "gpu")]
    core_pinned: bool,
    /// The expert packs of `CMF_QWEN_PACKS`, read on first use (inner None: none).
    #[cfg(feature = "gpu")]
    packs: Option<Option<PackSet>>,
    /// Decode keeps its routing to the arena's experts (`CMF_QWEN_MASK=1`, set after a
    /// prompt by `prompt_done`): no cold winner, the whole token chained as on a resident
    /// model. The next prompt is exact again.
    #[cfg(feature = "gpu")]
    masked: bool,
    /// A copy round's frame (`CMF_QWEN_COPY_FRAME`): the next device forward may take more
    /// than `TMAX` tokens under the mask, through the prompt-frame kernels.
    #[cfg(feature = "gpu")]
    copy_frame: bool,
    /// That frame verifies drafts: checkpoint the recurrent state first and bring every
    /// row's argmax back in `frame_ids` (megabytes of rows stay on the card).
    #[cfg(feature = "gpu")]
    frame_check: bool,
    #[cfg(feature = "gpu")]
    frame_ids: Option<Vec<u32>>,
    /// Tokens the copy rounds in a row gave with every draft accepted: how far the copy has
    /// held is how far the next round trusts it (a frame's size).
    #[cfg(feature = "gpu")]
    copy_run: usize,
    /// Tokens the request still takes after the one in hand (`set_draft_budget`).
    #[cfg(feature = "gpu")]
    draft_budget: usize,
}

/// Summed wall time and count of expert uploads (every thread), for the
/// per-token profile line: against the admission wall time it says whether
/// the uploads overlap.
pub(crate) static FILL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static FILL_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Wall time of the parallel fill sections (against FILL_NS: their overlap).
pub(crate) static FILLS_WALL_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// An expert's (gate, up, down) bytes straight from the model's mapping.
#[cfg(feature = "gpu")]
fn expert_slices(model: &CmfModel, t: (usize, usize, usize)) -> Option<[&[u8]; 3]> {
    let bytes = model.primary_bytes();
    let part = |i: usize| -> Option<&[u8]> {
        let e = model.tensors.get(i)?;
        let abs = model.entry_abs_offset(e)?;
        bytes.get(abs..abs + e.nbytes as usize)
    };
    Some([part(t.0)?, part(t.1)?, part(t.2)?])
}

/// `CMF_QWEN_FILL_TRACE=1`: one line per admission (caller, layer, expert, slot).
fn fill_trace() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_QWEN_FILL_TRACE").as_deref() == Ok("1"))
}

pub(crate) struct QwenGpuPool {
    /// Layers whose remap is encoded in a frame that has not run yet: no
    /// admission may evict their experts until that frame is done.
    pub(crate) hold_layers: [Option<usize>; 2],
    /// The last `staging` slots never enter the LRU: a frame's cold winners
    /// are uploaded there (token slot × rank) and read by the next frame's
    /// cold pass, so no admission or eviction can touch them in between.
    staging_base: usize,
    staging: usize,
    /// The pinned staging ring admissions go through (flushed before every
    /// frame submit); None uploads directly.
    #[cfg(feature = "gpu")]
    stager: Option<crate::gpu_wgpu::qwen4::Stager>,
    /// The host-heap tier admissions copy from by DMA (`HostTier`); None
    /// uploads through `write_buffer` (or the staging ring).
    #[cfg(feature = "gpu")]
    pub(crate) tier: Option<Arc<crate::gpu_wgpu::qwen4::HostTier>>,
    /// Persistent workers for the admission copies (`CMF_QWEN_ADMIT_THREADS`,
    /// default 16; 0 spawns scoped threads per batch as before).
    admit_pool: Option<Arc<Pool>>,
    /// The host tiers behind this arena (RAM tier, then the file), set by
    /// the device path; None reads experts straight from the memory map.
    pub(crate) store: Option<Arc<crate::expert_store::ExpertStore>>,
    /// The bank's expert layout (q4tp, q2tp gate/up, or mxfp4).
    #[cfg(feature = "gpu")]
    pub(crate) codec: crate::gpu_wgpu::MoeBankCodec,
    pub(crate) segment_slots: usize,
    floor: usize,
    n_experts: usize,
    fetch_quota: usize,
    fetch_min_seen: u16,
    fetch_max_env: &'static str,
    fetch_min_env: &'static str,
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
    /// `[layer][expert]`: experts no admission may evict (the shared core, `set_pins`).
    pin: Vec<bool>,
    /// Slots `ensure_layer` admitted (a streamed prompt's whole-bank sweep), oldest first:
    /// the transient window the next layers of the sweep take their slots from, so the
    /// resident working set of the layers ahead survives the sweep.
    sweep_slots: Vec<usize>,
    /// Per slot: `layer * n_experts + expert + 1` of the sweep admission that owns it, 0 = none.
    sweep_of: Vec<u32>,
}

/// An admission whose serial bookkeeping is complete but whose reserved
/// expert fills have not run yet.
///
/// This is deliberately a narrow, consuming lease over [`QwenGpuPool`].  A
/// caller may use [`Self::cold_ids`] only to overlap discardable host work
/// with the fills.  It must call [`Self::finish`] before using any slot: a
/// failed upload invalidates the whole admission and returns `None`.  Dropping
/// an unfinished lease removes its unfilled reservations, so no stale logical
/// slot can escape an abandoned overlap attempt.
#[cfg(feature = "gpu")]
#[must_use = "a pending admission must be finished or deliberately dropped"]
pub(crate) struct QwenGpuPendingEnsure<'pool, 'input> {
    pool: Option<&'pool mut QwenGpuPool>,
    model: &'input Arc<CmfModel>,
    picks: &'input [usize],
    triples: &'input [(usize, usize, usize)],
    layer: usize,
    base: usize,
    shared_slot: u32,
    plan: Vec<(usize, usize)>,
}

#[cfg(feature = "gpu")]
impl QwenGpuPendingEnsure<'_, '_> {
    /// Routed picks which are definitely absent after reservation.  The
    /// order is the caller's route order, including any duplicate input, so
    /// it can be compared directly to the completed remap.
    pub(crate) fn cold_ids(&self) -> Option<Vec<usize>> {
        let pool = self.pool.as_deref()?;
        cold_picks_from_slots(&pool.slot_for, self.base, pool.n_experts, self.picks)
    }

    /// Complete the reserved fills and return the same remap contract as
    /// [`QwenGpuPool::ensure`].  This consumes the lease, ending its mutable
    /// borrow of the pool before the caller submits uploads or encodes a
    /// frame.
    pub(crate) fn finish(self) -> Option<(Vec<u32>, u32)> {
        self.finish_inner(false)
            .map(|(remap, shared_slot, _)| (remap, shared_slot))
    }

    /// Complete the fills and submit any staged uploads before returning the
    /// remap. This is for a caller whose next operation is a separate DSV4
    /// frame rather than the generic Qwen frame that normally flushes itself.
    /// The upload queue is flushed even when an admitted fill failed, matching
    /// MiniCPM tail's existing failed-admission cleanup contract.
    pub(crate) fn finish_and_flush(self) -> Option<(Vec<u32>, u32, usize)> {
        self.finish_inner(true)
    }

    fn finish_inner(mut self, flush_uploads: bool) -> Option<(Vec<u32>, u32, usize)> {
        // Taking the pool makes `Drop` a no-op. From this point this method
        // owns the same cleanup responsibility as the old monolithic
        // `ensure` implementation.
        let pool = self.pool.take()?;
        let model = self.model;
        let triples = self.triples;
        let layer = self.layer;
        let base = self.base;
        let shared_slot = self.shared_slot;
        let plan = std::mem::take(&mut self.plan);
        // phase 2: the uploads, in parallel (they were one after another:
        // at a 12 GB budget that was the larger half of every token)
        let ok = pool.run_fills(model, layer, &plan, triples, false);
        let mut failed = false;
        for (&(slot, e), ok) in plan.iter().zip(&ok) {
            if !ok {
                pool.owner[slot] = None;
                pool.slot_for[base + e] = u32::MAX;
                pool.occupancy[layer] = pool.occupancy[layer].saturating_sub(1);
                pool.free.push(slot);
                failed = true;
            }
        }
        let result = if failed || triples.len() != pool.n_experts {
            None
        } else {
            let remap = pool.slot_for[base..base + pool.n_experts].to_vec();
            Some((remap, shared_slot, pool.segment_slots))
        };
        if flush_uploads {
            pool.flush_uploads();
        }
        result
    }
}

#[cfg(feature = "gpu")]
impl Drop for QwenGpuPendingEnsure<'_, '_> {
    fn drop(&mut self) {
        let Some(pool) = self.pool.as_deref_mut() else {
            return;
        };
        // A lease can be discarded after its caller has started speculative
        // CPU work but before any GPU bytes were written. Clear only the
        // reservations still owned by this exact plan; an old evicted expert
        // remains absent rather than being incorrectly remapped to bytes it
        // no longer owns.
        for &(slot, expert) in &self.plan {
            let Some(key) = self.base.checked_add(expert) else {
                continue;
            };
            if pool.owner.get(slot).copied().flatten() == Some((self.layer, expert))
                && pool.slot_for.get(key) == Some(&(slot as u32))
            {
                pool.owner[slot] = None;
                pool.slot_for[key] = u32::MAX;
                if let Some(occupancy) = pool.occupancy.get_mut(self.layer) {
                    *occupancy = occupancy.saturating_sub(1);
                }
                pool.free.push(slot);
            }
        }
    }
}

/// Select misses from a post-reservation logical slot map in the exact route
/// order.  Kept pure so an overlap caller can validate its optimistic host
/// work without opening a GPU device.
#[cfg(feature = "gpu")]
fn cold_picks_from_slots(
    slot_for: &[u32],
    base: usize,
    n_experts: usize,
    picks: &[usize],
) -> Option<Vec<usize>> {
    let slots = slot_for.get(base..base.checked_add(n_experts)?)?;
    let mut cold = Vec::with_capacity(picks.len());
    for &expert in picks {
        let slot = *slots.get(expert)?;
        if slot == u32::MAX {
            cold.push(expert);
        }
    }
    Some(cold)
}

/// Parse a bounded pool percentage without allowing a malformed operator
/// knob to turn into an unbounded allocation.  Kept pure so the Qwen and GLM
/// policies can be regression-tested without initializing a GPU adapter.
fn bounded_pool_pct(raw: Option<&str>, default: usize, min: usize, max: usize) -> usize {
    raw.and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(min, max)
}

fn bounded_pool_slots(raw: Option<&str>, safe: usize, cap_override: bool) -> usize {
    let floor = if cap_override { safe.min(8) } else { 8 };
    raw.and_then(|v| v.parse::<usize>().ok())
        .map(|v| if cap_override { v.min(safe) } else { v })
        .unwrap_or(safe)
        .max(floor)
}

/// Divide a resident arena only over layers that can actually reach it.
///
/// `floor` counts every owner, including a layer's pinned shared expert.  Do
/// not round this up: when a deliberately small operator-provided arena has
/// fewer slots than active layers, a larger floor would promise more protected
/// slots than exist and turn the fallback into an accidental global policy.
/// The caller's normal own-layer/global fallbacks still make progress at zero.
fn active_layer_floor(capacity: usize, active_layers: usize) -> usize {
    capacity / active_layers.max(1)
}

impl State {
    pub fn new(n_layers: usize) -> Self {
        Self {
            hyper: Vec::new(),
            layers: (0..n_layers).map(|_| LayerState::default()).collect(),
            token_history: Vec::new(),
            gpu_pool: None,
            pos: 0,
            #[cfg(feature = "gpu")]
            dev: None,
            picks_prev: vec![Vec::new(); n_layers],
            device_off: false,
            host_pos: 0,
            #[cfg(feature = "cuda")]
            cuda: None,
            cuda_off: false,
            #[cfg(feature = "cuda")]
            cuda_ref: None,
            #[cfg(feature = "cuda")]
            cuda_copy: CopyCur::default(),
            #[cfg(feature = "gpu")]
            profile: None,
            verify_window: false,
            window_rows: None,
            round_end: None,
            #[cfg(feature = "gpu")]
            mtp: None,
            mtp_tried: false,
            #[cfg(feature = "gpu")]
            resident_all: false,
            #[cfg(feature = "gpu")]
            inflight: None,
            #[cfg(feature = "gpu")]
            route_hist: Vec::new(),
            #[cfg(feature = "gpu")]
            route_tokens: 0,
            #[cfg(feature = "gpu")]
            phi_last: Vec::new(),
            #[cfg(feature = "gpu")]
            phi_last_tokens: 0,
            #[cfg(feature = "gpu")]
            row_snap: Vec::new(),
            #[cfg(feature = "gpu")]
            row_snap_tokens: 0,
            #[cfg(feature = "gpu")]
            row_gen: false,
            #[cfg(feature = "gpu")]
            core_pinned: false,
            #[cfg(feature = "gpu")]
            packs: None,
            #[cfg(feature = "gpu")]
            masked: false,
            #[cfg(feature = "gpu")]
            copy_frame: false,
            #[cfg(feature = "gpu")]
            frame_check: false,
            #[cfg(feature = "gpu")]
            frame_ids: None,
            #[cfg(feature = "gpu")]
            copy_run: 0,
            #[cfg(feature = "gpu")]
            draft_budget: usize::MAX,
        }
    }

    /// How many more tokens the request takes after the one in hand: a copy frame drafts
    /// no further (a window's seven drafts overshoot by little, a frame's would not).
    #[cfg(feature = "gpu")]
    pub fn set_draft_budget(&mut self, n: usize) {
        self.draft_budget = n;
    }

    pub(crate) fn reset(&mut self) {
        #[cfg(feature = "gpu")]
        if let Some(p) = self.inflight.take() {
            let _ = p.wait();
        }
        self.hyper.clear();
        self.token_history.clear();
        self.pos = 0;
        self.host_pos = 0;
        #[cfg(feature = "cuda")]
        {
            self.cuda_copy = CopyCur::default();
        }
        #[cfg(feature = "gpu")]
        {
            if self.row_gen {
                route_row("gen", &self.route_hist, self.route_tokens, &mut self.row_snap, &mut self.row_snap_tokens);
                self.row_gen = false;
            }
            trace_flush();
            self.route_hist.iter_mut().for_each(|c| *c = 0);
            self.route_tokens = 0;
            self.phi_last.iter_mut().for_each(|c| *c = 0);
            self.phi_last_tokens = 0;
            self.row_snap.iter_mut().for_each(|c| *c = 0);
            self.row_snap_tokens = 0;
        }
        for st in &mut self.layers {
            *st = LayerState::default();
        }
    }
}

#[cfg(feature = "gpu")]
impl QwenGpuPool {
    pub(crate) fn create(
        model: &Arc<CmfModel>,
        inter: usize,
        hidden: usize,
        n_layers: usize,
        n_experts: usize,
        gu_q2: bool,
    ) -> Option<Self> {
        Self::create_with_policy(
            model,
            inter,
            hidden,
            n_layers,
            n_experts,
            crate::gpu_wgpu::MoeBankCodec::from_gu_q2(gu_q2),
            "CMF_QWEN_POOL_PCT",
            75,
            25,
            85,
            "CMF_QWEN_EXPERT_SLOTS",
            "CMF_QWEN_FETCH_MAX",
            "CMF_QWEN_FETCH_MIN_SEEN",
            None,
            0,
        )
    }

    /// The device token path sizes the arena itself: the skeleton is already
    /// resident and pinned, so the slot count is whatever the budget still
    /// holds after it and a cache reserve, not a percentage of the card.
    pub(crate) fn create_explicit(
        model: &Arc<CmfModel>,
        inter: usize,
        hidden: usize,
        n_layers: usize,
        n_experts: usize,
        gu_q2: bool,
        slots: usize,
        staging: usize,
    ) -> Option<Self> {
        Self::create_with_policy(
            model,
            inter,
            hidden,
            n_layers,
            n_experts,
            crate::gpu_wgpu::MoeBankCodec::from_gu_q2(gu_q2),
            "CMF_QWEN_POOL_PCT",
            75,
            25,
            85,
            "CMF_QWEN_EXPERT_SLOTS",
            "CMF_QWEN_FETCH_MAX",
            "CMF_QWEN_FETCH_MIN_SEEN",
            Some(slots),
            staging,
        )
        .map(|mut pool| {
            // Admissions go straight through the queue (`write_buffer`) by
            // default; CMF_QWEN_STAGE_MB=<n> puts them through a pinned
            // staging ring of n MB per buffer. Measured on an RTX 3090 with
            // the file in the page cache, the ring lost at every budget: the
            // card waited ~25 ms longer per 8-token prompt frame (prompt
            // ingest 48-49 -> 56-60 tok/s without it), and decode of a
            // 200-token story went 32.0/32.3 -> 33.6/34.0 tok/s at the full
            // card and 17.1/17.7 -> 19.5/19.4 at 12 GB (131 cold experts a
            // token). The ring was introduced on hosts whose page cache could
            // not hold the file; there it may still win.
            let stage_mb = std::env::var("CMF_QWEN_STAGE_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
            pool.stager = crate::gpu_wgpu::qwen4::Stager::new(stage_mb);
            // Spawning a thread per expert and batch (mmap + munmap of its
            // stack) contends with the copies' own page faults on the
            // address-space lock: a pool of parked workers instead.
            let admit_threads = std::env::var("CMF_QWEN_ADMIT_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(16);
            pool.admit_pool = (admit_threads > 0).then(|| Arc::new(Pool::new(admit_threads)));
            // A cold expert costs the host hundreds of microseconds; the
            // device path admits on the first miss and fetches at least a
            // handful per layer and token. Both stay operator-tunable.
            pool.fetch_min_seen = std::env::var("CMF_QWEN_FETCH_MIN_SEEN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            pool.fetch_quota = std::env::var("CMF_QWEN_FETCH_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(pool.fetch_quota.max(16));
            pool
        })
    }

    /// Apply MiniCPM's narrow CPU-tail policy after generic construction.
    /// Generic Qwen/device callers deliberately retain the constructor's
    /// all-layer policy; MiniCPM knows that only its graph-produced tail can
    /// touch this bank.
    ///
    /// The floor is policy-only: it never changes an admitted slot, router
    /// score, route, or remap.  It merely changes which already-resident
    /// routed expert is considered first for a future replacement.  The
    /// two-worker admission default is likewise placement-only.  An explicit
    /// `CMF_QWEN_ADMIT_THREADS` always wins, including an explicit `0`.
    pub(crate) fn configure_minicpm_tail(&mut self, active_layers: usize) {
        let layers = active_layers.clamp(1, self.occupancy.len().max(1));
        self.floor = active_layer_floor(self.staging_base, layers);
        if std::env::var_os("CMF_QWEN_ADMIT_THREADS").is_none() {
            // MiniCPM's top-8 tail has only a handful of 1.7-MB admissions
            // per layer.  The generic 16-worker Qwen default causes page
            // fault/copy contention here; two parked workers leave the CPU
            // cold-expert correction and the graph driver room to run.
            self.admit_pool = Some(Arc::new(Pool::new(2)));
        }
    }

    /// The MTP head's own bank: exactly `slots` slots (whole segments), no
    /// carve-out — it is sized before the main arena, which leaves the
    /// reserve.
    pub(crate) fn create_exact(
        model: &Arc<CmfModel>,
        inter: usize,
        hidden: usize,
        n_experts: usize,
        gu_q2: bool,
        slots: usize,
    ) -> Option<Self> {
        Self::create_with_policy(
            model,
            inter,
            hidden,
            1,
            n_experts,
            crate::gpu_wgpu::MoeBankCodec::from_gu_q2(gu_q2),
            "CMF_QWEN_MTP_BANK",
            75,
            25,
            85,
            "CMF_QWEN_MTP_BANK_SLOTS",
            "CMF_QWEN_FETCH_MAX",
            "CMF_QWEN_FETCH_MIN_SEEN",
            Some(slots),
            0,
        )
    }

    /// Bytes one routed expert occupies in the arena (gate + up + down).
    pub(crate) fn per_expert_bytes(inter: usize, hidden: usize, gu_q2: bool) -> Option<usize> {
        let gu = cortiq_core::quant::expected_nbytes(
            if gu_q2 {
                TensorDtype::Q2TiledP
            } else {
                TensorDtype::Q4TiledP
            },
            &[inter, hidden],
        )?;
        let dn = cortiq_core::quant::expected_nbytes(TensorDtype::Q4TiledP, &[hidden, inter])?;
        2usize.checked_mul(gu)?.checked_add(dn)
    }

    /// A free slot, else a slot of the streamed prompt's sweep window (`sweep_victim`), else
    /// the least recently used one (`lru_victim`); never a shared slot or one of `protect`
    /// (this layer's current winners).
    fn victim(&mut self, layer: usize, protect: &[usize]) -> Option<usize> {
        if let Some(s) = self.free.pop() {
            self.sweep_of[s] = 0;
            return Some(s);
        }
        let s = self.sweep_victim(layer, protect).or_else(|| self.lru_victim(layer, protect))?;
        self.sweep_of[s] = 0;
        Some(s)
    }

    /// The newest slot of the sweep window another layer's whole-bank admission still owns:
    /// a streamed prompt visits the layers 0..n in order, so the slots the previous layer
    /// just took are the ones needed again last. Without the window the layer floor sits
    /// above the average occupancy of a full arena and the plain LRU evicts the residents of
    /// the layers the sweep reaches next, which then come back from the file every sweep.
    fn sweep_victim(&mut self, layer: usize, protect: &[usize]) -> Option<usize> {
        let ne = self.n_experts;
        let mut i = self.sweep_slots.len();
        while i > 0 {
            i -= 1;
            let slot = self.sweep_slots[i];
            let live = match self.owner[slot] {
                Some((l, e)) if e != usize::MAX && self.sweep_of[slot] as usize == l * ne + e + 1 => Some((l, e)),
                _ => None,
            };
            let Some((l, e)) = live else {
                self.sweep_slots.remove(i);
                continue;
            };
            if self.pin.get(l * ne + e).copied().unwrap_or(false) {
                self.sweep_of[slot] = 0;
                self.sweep_slots.remove(i);
                continue;
            }
            if (l == layer && protect.contains(&e)) || self.hold_layers.contains(&Some(l)) {
                continue;
            }
            self.sweep_slots.remove(i);
            return Some(slot);
        }
        None
    }

    /// The sweep window is over (the prompt is read): admissions go back to the plain LRU.
    pub(crate) fn sweep_clear(&mut self) {
        self.sweep_slots.clear();
        self.sweep_of.fill(0);
    }

    /// The least recently used slot of a layer above its working-set floor, else one of
    /// this layer's own, else any.
    fn lru_victim(&self, layer: usize, protect: &[usize]) -> Option<usize> {
        let hold = self.hold_layers;
        let (pin, ne) = (&self.pin, self.n_experts);
        let eligible = |owner: (usize, usize)| {
            owner.1 != usize::MAX
                && !pin.get(owner.0 * ne + owner.1).copied().unwrap_or(false)
                && (owner.0 != layer || !protect.contains(&owner.1))
                // a frame already encoded against this layer's remap is in
                // flight: its slots must stay what the remap says
                && !hold.contains(&Some(owner.0))
        };
        // Preserve a per-layer working set. A plain global LRU collapses
        // under the deterministic 0..47 layer sweep: late layers evict early
        // ones immediately before their next visit (the same failure
        // measured in DSV4).
        self.owner
            .iter()
            .enumerate()
            .filter_map(|(slot, &o)| {
                o.filter(|&x| {
                    eligible(x) && self.occupancy.get(x.0).copied().unwrap_or(0) > self.floor
                })
                .map(|_| slot)
            })
            .min_by_key(|&slot| self.last[slot])
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
            })
    }

    /// Admit one expert right now (a routed winner the arena did not hold),
    /// regardless of the admission hysteresis. Returns its slot.
    pub(crate) fn admit_now(
        &mut self,
        model: &Arc<CmfModel>,
        layer: usize,
        expert: usize,
        triples: &[(usize, usize, usize)],
        protect: &[usize],
    ) -> Option<u32> {
        if expert >= self.n_experts || triples.len() != self.n_experts {
            return None;
        }
        let key = layer * self.n_experts + expert;
        if self.slot_for[key] != u32::MAX {
            return Some(self.slot_for[key]);
        }
        let victim = self.victim(layer, protect)?;
        if !self.fill_slot(model, victim, layer, expert, triples[expert]) {
            if self.owner[victim].is_none() {
                self.free.push(victim);
            }
            return None;
        }
        if let Some(old) = self.owner[victim] {
            self.dropped(old);
            if old.1 == usize::MAX {
                self.shared_slot[old.0] = u32::MAX;
            } else {
                self.slot_for[old.0 * self.n_experts + old.1] = u32::MAX;
                self.occupancy[old.0] = self.occupancy[old.0].saturating_sub(1);
            }
        }
        self.clock = self.clock.saturating_add(1);
        self.owner[victim] = Some((layer, expert));
        self.slot_for[key] = victim as u32;
        self.occupancy[layer] += 1;
        self.seen[key] = self.seen[key].max(1);
        self.last[victim] = self.clock;
        Some(victim as u32)
    }

    /// Upload one expert into `slot`: through the staging ring when there
    /// is one with room, else straight through the queue.
    #[cfg(feature = "gpu")]
    fn fill_slot(
        &self,
        model: &Arc<CmfModel>,
        slot: usize,
        layer: usize,
        expert: usize,
        triple: (usize, usize, usize),
    ) -> bool {
        let t0 = std::time::Instant::now();
        let key = layer * self.n_experts + expert;
        // the host-heap tier: a DMA copy when it holds the expert; else the
        // expert goes into the tier first (one memcpy into host RAM) when it
        // has a slot, and through the queue otherwise
        let tier = self.tier.as_ref();
        if tier.is_some_and(|t| t.hit(key, slot)) {
            if slot < self.staging_base
                && let Some(store) = self.store.as_ref()
            {
                store.note_vram(layer, expert, true);
            }
            return true;
        }
        let up = |parts: [&[u8]; 3]| -> bool {
            if tier.is_some_and(|t| t.fill(key, Some(slot), parts)) {
                return true;
            }
            crate::gpu_wgpu::qwen4::upload_expert_parts(self.stager.as_ref(), model, slot, parts)
        };
        let stored = self
            .store
            .as_ref()
            .and_then(|store| store.with_expert(layer, expert, true, up));
        let ok = match stored {
            Some(ok) => ok,
            None => {
                if tier.is_some()
                    && let Some(parts) = expert_slices(model, triple)
                {
                    up(parts)
                } else if let Some(st) = self.stager.as_ref()
                    && crate::gpu_wgpu::qwen4::stage_expert(st, model, slot, triple)
                {
                    true
                } else {
                    crate::gpu_wgpu::dsv4_global_slot_fill(model, slot, triple)
                }
            }
        };
        // an arena slot (not a frame's staging slot): the RAM tier's copy
        // of this expert is now redundant
        if ok && slot < self.staging_base
            && let Some(store) = self.store.as_ref()
        {
            store.note_vram(layer, expert, true);
        }
        FILL_NS.fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        FILL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ok
    }

    /// `fill_slot` for every `(slot, expert)` job of `layer`, in parallel:
    /// on the admission pool when there is one, else one scoped thread per
    /// job (`chunked`: at most 32 threads). One result per job.
    #[cfg(feature = "gpu")]
    fn run_fills(
        &self,
        model: &Arc<CmfModel>,
        layer: usize,
        jobs: &[(usize, usize)],
        triples: &[(usize, usize, usize)],
        chunked: bool,
    ) -> Vec<bool> {
        let t0 = std::time::Instant::now();
        let me = self;
        // `current_device` is thread-local.  Unlike the persistent `Pool`
        // path below (which propagates its descriptor's device), the scoped
        // admission fallback creates fresh threads.  Preserve the caller's
        // adapter there too: a fill on adapter 0 followed by a frame on
        // adapter N would otherwise make a valid logical slot name stale
        // physical bytes.
        let device = crate::gpu::current_device();
        let out: Vec<bool> = if jobs.len() <= 1 {
            jobs.iter()
                .map(|&(slot, e)| me.fill_slot(model, slot, layer, e, triples[e]))
                .collect()
        } else if let Some(pool) = self.admit_pool.as_ref() {
            let done: Vec<std::sync::atomic::AtomicBool> = jobs
                .iter()
                .map(|_| std::sync::atomic::AtomicBool::new(false))
                .collect();
            let next = std::sync::atomic::AtomicUsize::new(0);
            pool.run(&|_, _| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(slot, e)) = jobs.get(i) else { break };
                    let ok = me.fill_slot(model, slot, layer, e, triples[e]);
                    done[i].store(ok, std::sync::atomic::Ordering::Relaxed);
                }
            });
            done.iter()
                .map(|d| d.load(std::sync::atomic::Ordering::Relaxed))
                .collect()
        } else if chunked {
            const ADMIT_THREADS: usize = 32;
            let nth = jobs.len().min(ADMIT_THREADS);
            let per = jobs.len().div_ceil(nth);
            std::thread::scope(|scope| {
                let handles: Vec<_> = jobs
                    .chunks(per)
                    .map(|part| {
                        let model = model.clone();
                        let h = scope.spawn(move || {
                            crate::gpu::with_device(device, || {
                                part.iter()
                                    .map(|&(slot, e)| {
                                        me.fill_slot(&model, slot, layer, e, triples[e])
                                    })
                                    .collect::<Vec<bool>>()
                            })
                        });
                        (part.len(), h)
                    })
                    .collect();
                // a panicked thread fails its whole part, keeping the
                // results aligned with `jobs`
                handles
                    .into_iter()
                    .flat_map(|(n, h)| h.join().unwrap_or_else(|_| vec![false; n]))
                    .collect()
            })
        } else {
            std::thread::scope(|scope| {
                let hs: Vec<_> = jobs
                    .iter()
                    .map(|&(slot, e)| {
                        let model = model.clone();
                        let triple = triples[e];
                        scope.spawn(move || {
                            crate::gpu::with_device(device, || {
                                me.fill_slot(&model, slot, layer, e, triple)
                            })
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap_or(false)).collect()
            })
        };
        FILLS_WALL_NS.fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        out
    }

    /// An arena slot gave up `old`: the RAM tier wants it back.
    fn dropped(&self, old: (usize, usize)) {
        if old.1 != usize::MAX
            && let Some(store) = self.store.as_ref()
        {
            store.note_vram(old.0, old.1, false);
        }
    }

    /// Submit what the staging ring and the host tier hold ahead of the
    /// next frame.
    #[cfg(feature = "gpu")]
    pub(crate) fn flush_uploads(&mut self) {
        if let Some(st) = self.stager.as_mut() {
            st.flush();
        }
        if let Some(cb) = self.tier.as_ref().and_then(|t| t.take()) {
            crate::gpu_wgpu::qwen4::submit_cb(cb);
        }
    }

    /// The staged uploads as a command buffer to submit ahead of a frame
    /// in the same queue submission; `rearm_uploads` follows the submit.
    /// (With both a staging ring and a host tier, the ring's copies go out
    /// on their own submission first: they target other slots.)
    pub(crate) fn take_uploads(&mut self) -> Option<wgpu::CommandBuffer> {
        let staged = self.stager.as_mut().and_then(|st| st.take());
        let tiered = self.tier.as_ref().and_then(|t| t.take());
        match (staged, tiered) {
            (Some(a), Some(b)) => {
                crate::gpu_wgpu::qwen4::submit_cb(a);
                Some(b)
            }
            (a, b) => a.or(b),
        }
    }

    /// A submission that followed every copy `take_uploads` handed out has
    /// completed: the host tier may reuse the slots they read.
    pub(crate) fn uploads_done(&self) {
        if let Some(t) = self.tier.as_ref() {
            t.done();
        }
    }

    pub(crate) fn rearm_uploads(&mut self) {
        if let Some(st) = self.stager.as_mut() {
            st.rearm();
        }
    }

    /// Upload a frame's cold winners into the staging slots of token slot
    /// `tok` (rank `j` each), outside the LRU. Returns the slot per expert,
    /// `None` where the upload failed or no staging exists.
    pub(crate) fn stage_cold(
        &self,
        model: &Arc<CmfModel>,
        layer: usize,
        tok: usize,
        experts: &[usize],
        triples: &[(usize, usize, usize)],
        top_k: usize,
    ) -> Vec<Option<u32>> {
        if self.staging == 0 || (tok + 1) * top_k > self.staging || triples.len() != self.n_experts
        {
            return vec![None; experts.len()];
        }
        let base = self.staging_base + tok * top_k;
        if fill_trace() {
            eprintln!("fill staging tok={tok} experts={experts:?}");
        }
        let jobs: Vec<(usize, usize)> = experts
            .iter()
            .enumerate()
            .filter(|&(_, &e)| e < self.n_experts)
            .map(|(j, &e)| (base + j, e))
            .collect();
        let ok = self.run_fills(model, layer, &jobs, triples, false);
        let mut out = vec![None; experts.len()];
        let mut ji = 0;
        for (j, &e) in experts.iter().enumerate() {
            if e < self.n_experts {
                if ok[ji] {
                    out[j] = Some(jobs[ji].0 as u32);
                }
                ji += 1;
            }
        }
        out
    }

    /// `admit_now` for a layer's whole cold list: slots are reserved in
    /// order, the uploads (a 1.7 MB memcpy each) run on parallel threads.
    /// Returns one slot per expert, `None` where the arena could not take it.
    pub(crate) fn admit_many(
        &mut self,
        model: &Arc<CmfModel>,
        layer: usize,
        experts: &[usize],
        triples: &[(usize, usize, usize)],
        protect: &[usize],
    ) -> Vec<Option<u32>> {
        if triples.len() != self.n_experts {
            return vec![None; experts.len()];
        }
        let plan = self.reserve_many(layer, experts, protect);
        let uploads: Vec<(usize, usize)> = plan
            .iter()
            .filter_map(|(_, p)| p.filter(|&(_, e)| e != usize::MAX))
            .collect();
        let failed = self.upload_reserved(model, layer, &uploads, triples, false);
        Self::plan_slots(plan, &failed)
    }

    /// Phase 1 of an admission: a slot per expert, reserved in order (the
    /// bookkeeping is serial). `(expert, Some((slot, expert)))` for a new
    /// reservation that still needs its upload, `(expert, Some((slot,
    /// usize::MAX)))` for one already resident, `(expert, None)` where the
    /// arena could not take it.
    fn reserve_many(
        &mut self,
        layer: usize,
        experts: &[usize],
        protect: &[usize],
    ) -> Vec<(usize, Option<(usize, usize)>)> {
        let mut plan: Vec<(usize, Option<(usize, usize)>)> = Vec::with_capacity(experts.len());
        for &expert in experts {
            if expert >= self.n_experts {
                plan.push((expert, None));
                continue;
            }
            let key = layer * self.n_experts + expert;
            if self.slot_for[key] != u32::MAX {
                plan.push((expert, Some((self.slot_for[key] as usize, usize::MAX))));
                continue;
            }
            let Some(victim) = self.victim(layer, protect) else {
                plan.push((expert, None));
                continue;
            };
            // take the slot now so a later expert of this list cannot pick it
            let old = self.owner[victim].take();
            if let Some(old) = old {
                self.dropped(old);
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
            self.seen[key] = self.seen[key].max(1);
            self.clock = self.clock.saturating_add(1);
            self.last[victim] = self.clock;
            if fill_trace() {
                eprintln!("fill cold layer={layer} expert={expert} slot={victim}");
            }
            plan.push((expert, Some((victim, expert))));
        }
        plan
    }

    /// Phase 2: the reserved uploads (a 1.7 MB memcpy each) on parallel
    /// threads; a failed one rolls its reservation back. `pooled` spreads
    /// them over at most `ADMIT_THREADS` threads instead of one thread per
    /// expert. Returns the experts that failed.
    fn upload_reserved(
        &mut self,
        model: &Arc<CmfModel>,
        layer: usize,
        uploads: &[(usize, usize)],
        triples: &[(usize, usize, usize)],
        pooled: bool,
    ) -> Vec<usize> {
        if let Some(store) = self.store.as_ref() {
            let es: Vec<usize> = uploads.iter().map(|&(_, e)| e).collect();
            store.prefetch(layer, &es);
        }
        // host-tier hits are a queued copy each: no thread for those
        let mut hit = vec![false; uploads.len()];
        if let Some(t) = self.tier.as_ref() {
            for (i, &(slot, e)) in uploads.iter().enumerate() {
                if t.hit(layer * self.n_experts + e, slot) {
                    hit[i] = true;
                    if slot < self.staging_base
                        && let Some(store) = self.store.as_ref()
                    {
                        store.note_vram(layer, e, true);
                    }
                }
            }
        }
        let all = uploads;
        let rest: Vec<(usize, usize)> = all
            .iter()
            .zip(&hit)
            .filter(|(_, h)| !**h)
            .map(|(&u, _)| u)
            .collect();
        let results = self.run_fills(model, layer, &rest, triples, pooled);
        let mut rest_ok = results.into_iter();
        let results: Vec<bool> = hit
            .iter()
            .map(|&h| h || rest_ok.next().unwrap_or(false))
            .collect();
        let uploads = all;
        let mut failed: Vec<usize> = Vec::new();
        for (i, &(slot, e)) in uploads.iter().enumerate() {
            if !results.get(i).copied().unwrap_or(false) {
                // roll the reservation back: the slot holds nothing usable now
                self.owner[slot] = None;
                self.slot_for[layer * self.n_experts + e] = u32::MAX;
                self.occupancy[layer] = self.occupancy[layer].saturating_sub(1);
                self.free.push(slot);
                failed.push(e);
            }
        }
        failed
    }

    fn plan_slots(plan: Vec<(usize, Option<(usize, usize)>)>, failed: &[usize]) -> Vec<Option<u32>> {
        plan.into_iter()
            .map(|(expert, p)| match p {
                Some((slot, _)) if !failed.contains(&expert) => Some(slot as u32),
                _ => None,
            })
            .collect()
    }

    /// `admit_many` for every token of a frame at once: the slots are
    /// reserved token by token in order, exactly as consecutive
    /// `admit_many` calls reserve them (same victims, same slots), and all
    /// the frame's uploads then run in one parallel batch instead of one
    /// small batch per token. `reqs`: each token's cold experts and its
    /// protected picks. (`CMF_QWEN_ADMIT_BATCH=0`: per token.)
    pub(crate) fn admit_frame(
        &mut self,
        model: &Arc<CmfModel>,
        layer: usize,
        reqs: &[(Vec<usize>, Vec<usize>)],
        triples: &[(usize, usize, usize)],
    ) -> Vec<Vec<Option<u32>>> {
        if triples.len() != self.n_experts {
            return reqs.iter().map(|(e, _)| vec![None; e.len()]).collect();
        }
        let plans: Vec<_> = reqs
            .iter()
            .map(|(experts, protect)| self.reserve_many(layer, experts, protect))
            .collect();
        // A slot reserved twice in the frame (a later token evicted an
        // earlier token's fresh admission) ends up holding the later
        // expert, as with per-token admission: upload only that one.
        let mut uploads: Vec<(usize, usize)> = Vec::new();
        for plan in &plans {
            for &(_, p) in plan {
                if let Some((slot, e)) = p.filter(|&(_, e)| e != usize::MAX) {
                    uploads.retain(|&(s, _)| s != slot);
                    uploads.push((slot, e));
                }
            }
        }
        let failed = self.upload_reserved(model, layer, &uploads, triples, true);
        plans
            .into_iter()
            .map(|plan| Self::plan_slots(plan, &failed))
            .collect()
    }

    /// Fill free slots with the given `(layer, expert)` pairs in order, the
    /// hottest first, until the arena is full. Returns how many went in.
    pub(crate) fn prefill(
        &mut self,
        model: &Arc<CmfModel>,
        ranked: &[(usize, usize)],
        triples: &[Vec<(usize, usize, usize)>],
    ) -> usize {
        if let Some(store) = self.store.clone() {
            return self.prefill_parallel(model, ranked, triples, &store);
        }
        let mut n = 0;
        for &(layer, expert) in ranked {
            if self.free.is_empty() {
                break;
            }
            if layer >= triples.len() || expert >= self.n_experts {
                continue;
            }
            let key = layer * self.n_experts + expert;
            if self.slot_for[key] != u32::MAX {
                continue;
            }
            let Some(&triple) = triples[layer].get(expert) else {
                continue;
            };
            let Some(slot) = self.free.pop() else { break };
            if !crate::gpu_wgpu::dsv4_global_slot_fill(model, slot, triple) {
                self.free.push(slot);
                break;
            }
            self.owner[slot] = Some((layer, expert));
            self.slot_for[key] = slot as u32;
            self.occupancy[layer] += 1;
            self.seen[key] = self.seen[key].max(1);
            self.clock = self.clock.saturating_add(1);
            self.last[slot] = self.clock;
            n += 1;
        }
        n
    }

    /// `prefill` through the host tiers: the slots are planned serially, the
    /// reads and uploads run on many threads (a fast drive needs a deep
    /// queue), and the queue is flushed every few hundred experts so wgpu's
    /// write staging never holds the whole arena at once.
    fn prefill_parallel(
        &mut self,
        model: &Arc<CmfModel>,
        ranked: &[(usize, usize)],
        triples: &[Vec<(usize, usize, usize)>],
        store: &Arc<crate::expert_store::ExpertStore>,
    ) -> usize {
        let mut plan: Vec<(usize, usize, usize)> = Vec::new();
        for &(layer, expert) in ranked {
            if layer >= triples.len() || expert >= self.n_experts {
                continue;
            }
            let key = layer * self.n_experts + expert;
            if self.slot_for[key] != u32::MAX {
                continue;
            }
            let Some(slot) = self.free.pop() else { break };
            plan.push((slot, layer, expert));
        }
        let threads = std::thread::available_parallelism()
            .map_or(8, |n| n.get())
            .clamp(4, 32);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let done: Vec<std::sync::atomic::AtomicBool> = (0..plan.len())
            .map(|_| std::sync::atomic::AtomicBool::new(false))
            .collect();
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(slot, layer, expert)) = plan.get(i) else {
                            break;
                        };
                        // into VRAM only: a RAM copy of an arena expert is
                        // redundant (the background loader takes the rest)
                        let ok = store
                            .with_expert(layer, expert, false, |parts| {
                                crate::gpu_wgpu::qwen4::upload_expert_parts(
                                    None, model, slot, parts,
                                )
                            })
                            .unwrap_or(false);
                        done[i].store(ok, std::sync::atomic::Ordering::Relaxed);
                        if i % 384 == 383 {
                            crate::gpu_wgpu::qwen4::flush_writes();
                        }
                    }
                });
            }
        });
        crate::gpu_wgpu::qwen4::flush_writes();
        let mut n = 0;
        for (i, &(slot, layer, expert)) in plan.iter().enumerate() {
            if !done[i].load(std::sync::atomic::Ordering::Relaxed) {
                self.free.push(slot);
                continue;
            }
            let key = layer * self.n_experts + expert;
            self.owner[slot] = Some((layer, expert));
            self.slot_for[key] = slot as u32;
            self.occupancy[layer] += 1;
            self.seen[key] = self.seen[key].max(1);
            self.clock = self.clock.saturating_add(1);
            self.last[slot] = self.clock;
            store.note_vram(layer, expert, true);
            n += 1;
        }
        n
    }

    /// Every routed expert of `layer` resident at once (a streamed prompt chunk runs the
    /// layer's whole bank): the missing ones get slots — never one of this layer's own,
    /// a shared expert's or a held layer's — and are uploaded in one parallel batch.
    /// The layer's remap and shared slot; None when the arena cannot hold them all.
    #[cfg(feature = "gpu")]
    pub(crate) fn ensure_layer(
        &mut self,
        model: &Arc<CmfModel>,
        layer: usize,
        triples: &[(usize, usize, usize)],
        shared: Option<(usize, usize, usize)>,
    ) -> Option<(Vec<u32>, u32)> {
        let ne = self.n_experts;
        if triples.len() != ne || self.staging_base < ne + self.shared_slot.len() + 1 {
            return None;
        }
        let (_, shared_slot) = self.ensure(model, layer, &[], triples, shared)?;
        let all: Vec<usize> = (0..ne).collect();
        self.clock = self.clock.saturating_add(1);
        let now = self.clock;
        let base = layer * ne;
        if layer == 0 {
            // a new sweep: drop the window's dead and repeated entries (a slot keeps its newest place)
            let mut once = vec![false; self.owner.len()];
            let mut live: Vec<usize> = Vec::with_capacity(self.sweep_slots.len());
            for &slot in self.sweep_slots.iter().rev() {
                let tagged = matches!(self.owner[slot], Some((l, e)) if e != usize::MAX && self.sweep_of[slot] as usize == l * ne + e + 1);
                if tagged && !once[slot] {
                    once[slot] = true;
                    live.push(slot);
                }
            }
            live.reverse();
            self.sweep_slots = live;
        }
        let mut jobs: Vec<(usize, usize)> = Vec::new();
        for e in 0..ne {
            let s = self.slot_for[base + e];
            if s != u32::MAX {
                self.last[s as usize] = now;
                continue;
            }
            let Some(v) = self.victim(layer, &all) else {
                // undo the slots taken so far: nothing was uploaded yet
                for &(slot, e2) in &jobs {
                    self.owner[slot] = None;
                    self.slot_for[base + e2] = u32::MAX;
                    self.occupancy[layer] = self.occupancy[layer].saturating_sub(1);
                    self.free.push(slot);
                }
                return None;
            };
            if let Some(old) = self.owner[v] {
                self.dropped(old);
                if old.1 == usize::MAX {
                    self.shared_slot[old.0] = u32::MAX;
                } else {
                    self.slot_for[old.0 * ne + old.1] = u32::MAX;
                    self.occupancy[old.0] = self.occupancy[old.0].saturating_sub(1);
                }
            }
            self.owner[v] = Some((layer, e));
            self.slot_for[base + e] = v as u32;
            self.occupancy[layer] += 1;
            self.seen[base + e] = self.seen[base + e].max(1);
            self.last[v] = now;
            jobs.push((v, e));
        }
        let ok = self.run_fills(model, layer, &jobs, triples, true);
        if ok.iter().any(|&x| !x) {
            for (&(slot, e), &good) in jobs.iter().zip(&ok) {
                if !good {
                    self.owner[slot] = None;
                    self.slot_for[base + e] = u32::MAX;
                    self.occupancy[layer] = self.occupancy[layer].saturating_sub(1);
                    self.free.push(slot);
                }
            }
            return None;
        }
        for &(slot, e) in &jobs {
            self.sweep_of[slot] = (base + e + 1) as u32;
            self.sweep_slots.push(slot);
        }
        Some((self.slot_for[base..base + ne].to_vec(), shared_slot))
    }

    /// Make the arena hold `ranked` (hottest first) as far as it fits: routed experts
    /// outside the target give their slots up, then the target is admitted in order.
    /// (evicted, admitted).
    #[cfg(feature = "gpu")]
    pub(crate) fn repack(
        &mut self,
        model: &Arc<CmfModel>,
        ranked: &[(usize, usize)],
        triples: &[Vec<(usize, usize, usize)>],
    ) -> (usize, usize) {
        let ne = self.n_experts;
        let pinned = self.shared_slot.iter().filter(|&&s| s != u32::MAX).count();
        let room = self.staging_base.saturating_sub(pinned);
        let target: Vec<(usize, usize)> = ranked
            .iter()
            .copied()
            .filter(|&(l, e)| l < triples.len() && e < ne)
            .take(room)
            .collect();
        let mut want = vec![false; self.slot_for.len()];
        for &(l, e) in &target {
            want[l * ne + e] = true;
        }
        let mut evicted = 0;
        for slot in 0..self.staging_base {
            if let Some((l, e)) = self.owner[slot]
                && e != usize::MAX
                && !want[l * ne + e]
                && !self.pin.get(l * ne + e).copied().unwrap_or(false)
            {
                self.dropped((l, e));
                self.slot_for[l * ne + e] = u32::MAX;
                self.occupancy[l] = self.occupancy[l].saturating_sub(1);
                self.owner[slot] = None;
                self.free.push(slot);
                evicted += 1;
            }
        }
        // the target's missing experts into the freed slots, layer by layer in one
        // parallel batch each, through `fill_slot` (a DMA from the host tier when it
        // holds the expert; `prefill` would write every one through the queue)
        let now = self.clock.saturating_add(1);
        self.clock = now;
        let mut by_layer: Vec<Vec<(usize, usize)>> = vec![Vec::new(); triples.len()];
        for &(l, e) in &target {
            let key = l * ne + e;
            if self.slot_for[key] != u32::MAX {
                self.last[self.slot_for[key] as usize] = now;
                continue;
            }
            let Some(slot) = self.free.pop() else { break };
            self.owner[slot] = Some((l, e));
            self.slot_for[key] = slot as u32;
            self.occupancy[l] += 1;
            self.seen[key] = self.seen[key].max(1);
            self.last[slot] = now;
            by_layer[l].push((slot, e));
        }
        let mut admitted = 0;
        for (l, jobs) in by_layer.iter().enumerate() {
            if jobs.is_empty() {
                continue;
            }
            let ok = self.run_fills(model, l, jobs, &triples[l], true);
            for (&(slot, e), &good) in jobs.iter().zip(&ok) {
                if good {
                    admitted += 1;
                } else {
                    self.owner[slot] = None;
                    self.slot_for[l * ne + e] = u32::MAX;
                    self.occupancy[l] = self.occupancy[l].saturating_sub(1);
                    self.free.push(slot);
                }
            }
        }
        (evicted, admitted)
    }

    /// The live `(layer, expert) → slot` row and the layer's pinned shared
    /// slot, without admitting anything.
    pub(crate) fn remap_snapshot(&self, layer: usize) -> (Vec<u32>, u32) {
        let base = layer * self.n_experts;
        (
            self.slot_for[base..base + self.n_experts].to_vec(),
            self.shared_slot.get(layer).copied().unwrap_or(u32::MAX),
        )
    }

    /// DeepSeek V4.1 uses the same segmented, model-wide bank as Qwen's
    /// dynamic MoE path, but its routed experts are Q4TP in the production
    /// profile.  Keep the policy knob architecture-specific while sharing
    /// the allocator and LRU implementation.
    pub(crate) fn create_for_dsv41(
        model: &Arc<CmfModel>,
        inter: usize,
        hidden: usize,
        n_layers: usize,
        n_experts: usize,
        gu_q2: bool,
    ) -> Option<Self> {
        Self::create_with_policy(
            model,
            inter,
            hidden,
            n_layers,
            n_experts,
            crate::gpu_wgpu::MoeBankCodec::from_gu_q2(gu_q2),
            "CMF_DSV41_POOL_PCT",
            75,
            25,
            85,
            "CMF_DSV41_EXPERT_SLOTS",
            "CMF_DSV41_FETCH_MAX",
            "CMF_DSV41_FETCH_MIN_SEEN",
            None,
            0,
        )
    }

    /// [`create_for_dsv41`](Self::create_for_dsv41) for any bank codec —
    /// the mxfp4 experts of a `convert --quant fp4` file included.
    pub(crate) fn create_for_dsv41_codec(
        model: &Arc<CmfModel>,
        inter: usize,
        hidden: usize,
        n_layers: usize,
        n_experts: usize,
        codec: crate::gpu_wgpu::MoeBankCodec,
    ) -> Option<Self> {
        Self::create_with_policy(
            model,
            inter,
            hidden,
            n_layers,
            n_experts,
            codec,
            "CMF_DSV41_POOL_PCT",
            75,
            25,
            85,
            "CMF_DSV41_EXPERT_SLOTS",
            "CMF_DSV41_FETCH_MAX",
            "CMF_DSV41_FETCH_MIN_SEEN",
            None,
            0,
        )
    }

    /// GLM-5.3-Flash's Q2 expert arena shares the card with a much larger
    /// static attention/control footprint and with transient cold-expert
    /// staging allocations.  Keep that model-specific policy here while
    /// reusing the same LRU/cache machinery as Qwen.  The lower default is
    /// intentionally applied before the common allocator's workspace carve
    /// out; it treats the configured budget as a physical envelope rather
    /// than as a promise that the whole budget may become resident weights.
    pub(crate) fn create_for_glm(
        model: &Arc<CmfModel>,
        inter: usize,
        hidden: usize,
        n_layers: usize,
        n_experts: usize,
        gu_q2: bool,
    ) -> Option<Self> {
        Self::create_with_policy(
            model,
            inter,
            hidden,
            n_layers,
            n_experts,
            crate::gpu_wgpu::MoeBankCodec::from_gu_q2(gu_q2),
            "CMF_GLM_POOL_PCT",
            40,
            20,
            40,
            "CMF_GLM_EXPERT_SLOTS",
            "CMF_GLM_FETCH_MAX",
            "CMF_GLM_FETCH_MIN_SEEN",
            None,
            0,
        )
    }

    fn create_with_policy(
        model: &Arc<CmfModel>,
        inter: usize,
        hidden: usize,
        n_layers: usize,
        n_experts: usize,
        codec: crate::gpu_wgpu::MoeBankCodec,
        pct_env: &str,
        default_pct: usize,
        min_pct: usize,
        max_pct: usize,
        slots_env: &str,
        fetch_max_env: &'static str,
        fetch_min_env: &'static str,
        explicit_slots: Option<usize>,
        staging: usize,
    ) -> Option<Self> {
        if !crate::gpu_wgpu::dsv4_global_moe_supported() {
            return None;
        }
        let gu_q2 = codec.gu_q2();
        // Only the V4.1 allocator knows the mxfp4 bank.
        if codec == crate::gpu_wgpu::MoeBankCodec::Mxfp4 && pct_env != "CMF_DSV41_POOL_PCT" {
            return None;
        }
        let (gu, dn) = codec.lens(inter, hidden)?;
        let per = 2usize.checked_mul(gu)?.checked_add(dn)?;
        let budget = crate::gpu_wgpu::dsv4_vram_budget()? as usize;
        // The Q8_2f attention/GDN skeleton, f32 HyperConnection projections,
        // KV/state and the full-vocabulary head live next to this arena. The
        // common allocator subtracts another 2-4 GiB workspace below this
        // request. Qwen's 75% profile preserves locality. GLM has a separate
        // physical-card envelope: the measured RTX-3090 budget can use the
        // full 100% request (the allocator subtracts workspace below), while
        // the 16-GB compatibility profile stays at the proven 40% cap. The
        // global allocator still subtracts its workspace reserve and refuses
        // an unsafe allocation.
        let max_pct = if pct_env == "CMF_GLM_POOL_PCT" && budget >= 20_000_000_000 {
            // A 24-GiB card has room for the measured static trunk plus a
            // larger expert arena.  The allocator below still subtracts its
            // workspace reserve and rounds the result, while the 16-GiB
            // compatibility budget remains capped at the conservative 40%.
            100
        } else {
            max_pct
        };
        let default_pct = if pct_env == "CMF_GLM_POOL_PCT" && budget >= 20_000_000_000 {
            100
        } else {
            default_pct
        };
        let pool_pct = bounded_pool_pct(
            std::env::var(pct_env).ok().as_deref(),
            default_pct,
            min_pct,
            max_pct,
        );
        let safe_requested = budget.saturating_mul(pool_pct) / 100 / per.max(1);
        // GLM's explicit slot knob is still subject to the same
        // physical-envelope cap as its percentage policy.  This prevents
        // `CMF_GPU_VRAM_MB=16000` plus an oversized slot override from
        // recreating pod-7's physical OOM.  Qwen retains its established
        // operator-controlled slot override semantics.
        let bounded_override = pct_env == "CMF_GLM_POOL_PCT" || pct_env == "CMF_DSV41_POOL_PCT";
        let requested = match explicit_slots {
            Some(n) => n.max(1),
            None => bounded_pool_slots(
                std::env::var(slots_env).ok().as_deref(),
                safe_requested,
                bounded_override,
            ),
        };
        // V4.1: never more slots than the model has experts (each layer's
        // routed experts and its shared one), whole S16 segments, plus the
        // allocator's workspace carve-out (it subtracts that from the
        // request). A small V4.1 file (the test fixtures) otherwise sized
        // its bank from 75 % of the card: 147,168 slots for the five-layer
        // GPU fixture under a 4 GiB budget, and without a budget bound most
        // of a 96 GB card. No-op for the 1017 release (40 x 385 = 15,400
        // slots, far above any card's request).
        let requested = if pct_env == "CMF_DSV41_POOL_PCT" {
            let needed = n_layers
                .saturating_mul(n_experts.saturating_add(1))
                .div_ceil(16)
                .saturating_mul(16);
            requested.min(
                needed.saturating_add(
                    crate::gpu_wgpu::dsv4_global_moe_workspace_slots(per).unwrap_or(0),
                ),
            )
        } else {
            requested
        };
        let (capacity, segment_slots) = if pct_env == "CMF_DSV41_POOL_PCT" {
            crate::gpu_wgpu::dsv4_global_moe_create_for_dsv41_codec(
                model, requested, inter, hidden, codec,
            )?
        } else if explicit_slots.is_some() && pct_env == "CMF_QWEN_POOL_PCT" {
            // the device path: the caller already left its reserve and the
            // workspace (see the arena sizing in `forward_tokens_device`)
            crate::gpu_wgpu::dsv4_global_moe_create_s8_exact(
                model, requested, inter, hidden, gu_q2,
            )?
        } else if pct_env == "CMF_QWEN_MTP_BANK" {
            // the draft head's bank: exactly its experts, no workspace
            // carve-out (the main arena already left the card's reserve)
            crate::gpu_wgpu::dsv4_global_moe_create_slots(model, requested, inter, hidden, gu_q2)?
        } else {
            // Generic Qwen/GLM/DSV4 callers keep the established S8 bank.
            crate::gpu_wgpu::dsv4_global_moe_create(model, requested, inter, hidden, gu_q2)?
        };
        let (auto_quota, auto_min_seen) = crate::gpu_wgpu::dsv4_fetch_defaults();
        let fetch_quota = std::env::var(fetch_max_env)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(auto_quota);
        // GLM routes through a 45-layer sweep with a larger cold-expert
        // penalty. Q2's mixed profile waits for three observations so
        // one-shot routes do not trigger a synchronous upload; Q4 keeps the
        // first-recurrence policy because its larger rows make CPU misses
        // more expensive. Qwen retains its established hysteresis.
        let glm_policy = pct_env == "CMF_GLM_POOL_PCT";
        let fetch_min_seen = std::env::var(fetch_min_env)
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            // Repair-8's first-recurrence policy is the measured GLM
            // baseline. Repair-9's Q2-specific three-observation gate
            // regressed throughput and is intentionally removed here;
            // persistent-device scheduling must not be substituted by an
            // admission heuristic.
            .unwrap_or(if glm_policy || pct_env == "CMF_DSV41_POOL_PCT" {
                1
            } else {
                auto_min_seen.max(2)
            });
        if std::env::var_os("CMF_QWEN_PROF").is_some()
            || std::env::var_os("CMF_GLM_PROF").is_some()
            || std::env::var_os("CMF_DSV41_PROF").is_some()
        {
            eprintln!(
                "dynamic-pool capacity={capacity} segment_slots={segment_slots} requested={requested} pct={pool_pct} env={pct_env} fetch_quota={fetch_quota} min_seen={fetch_min_seen} gu={}",
                codec.label()
            );
        }
        let staging = staging.min(capacity / 2);
        let staging_base = capacity - staging;
        Some(Self {
            hold_layers: [None, None],
            staging_base,
            staging,
            #[cfg(feature = "gpu")]
            stager: None,
            #[cfg(feature = "gpu")]
            tier: None,
            admit_pool: None,
            store: None,
            codec,
            segment_slots,
            floor: (staging_base / n_layers.max(1)).max(2),
            n_experts,
            fetch_quota,
            fetch_min_seen,
            fetch_max_env,
            fetch_min_env,
            owner: vec![None; capacity],
            slot_for: vec![u32::MAX; n_layers.checked_mul(n_experts)?],
            shared_slot: vec![u32::MAX; n_layers],
            seen: vec![0; n_layers.checked_mul(n_experts)?],
            seen_epoch: vec![0; n_layers.checked_mul(n_experts)?],
            free: (0..staging_base).rev().collect(),
            occupancy: vec![0; n_layers.max(1)],
            last: vec![0; capacity],
            clock: 0,
            pin: Vec::new(),
            sweep_slots: Vec::new(),
            sweep_of: vec![0; capacity],
        })
    }

    /// Pin `core` (`[layer] -> experts`): no admission evicts a pinned expert.
    pub(crate) fn set_pins(&mut self, core: &[Vec<usize>]) {
        let ne = self.n_experts;
        self.pin = vec![false; self.slot_for.len()];
        for (l, row) in core.iter().enumerate() {
            for &e in row {
                if let Some(p) = self.pin.get_mut(l * ne + e) {
                    *p = true;
                }
            }
        }
    }

    /// Reserve slots for an admission without starting its potentially slow
    /// fills.  Normal callers should use [`Self::ensure`]; this split form is
    /// only for an exact caller that can overlap independent, discardable CPU
    /// work with the fills and validates the completed remap before use.
    pub(crate) fn begin_ensure<'pool, 'input>(
        &'pool mut self,
        model: &'input Arc<CmfModel>,
        layer: usize,
        picks: &'input [usize],
        triples: &'input [(usize, usize, usize)],
        shared: Option<(usize, usize, usize)>,
    ) -> Option<QwenGpuPendingEnsure<'pool, 'input>> {
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
                self.sweep_of[slot as usize] = 0;
            }
        }
        let quota = std::env::var(self.fetch_max_env)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(self.fetch_quota);
        let min_seen = std::env::var(self.fetch_min_env)
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(self.fetch_min_seen);
        // phase 1: reserve a slot per prefetch (the bookkeeping is serial)
        let mut plan: Vec<(usize, usize)> = Vec::new();
        for &expert in picks {
            let key = base + expert;
            if self.slot_for[key] != u32::MAX || plan.len() >= quota || self.seen[key] < min_seen {
                continue;
            }
            if triples.get(expert).is_none() {
                return None;
            }
            let victim = self.victim(layer, picks)?;
            if let Some(old) = self.owner[victim] {
                self.dropped(old);
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
            if fill_trace() {
                eprintln!(
                    "fill prefetch layer={layer} expert={expert} slot={victim} seen={}",
                    self.seen[key]
                );
            }
            plan.push((victim, expert));
        }
        Some(QwenGpuPendingEnsure {
            pool: Some(self),
            model,
            picks,
            triples,
            layer,
            base,
            shared_slot,
            plan,
        })
    }

    pub(crate) fn ensure(
        &mut self,
        model: &Arc<CmfModel>,
        layer: usize,
        picks: &[usize],
        triples: &[(usize, usize, usize)],
        shared: Option<(usize, usize, usize)>,
    ) -> Option<(Vec<u32>, u32)> {
        self.begin_ensure(model, layer, picks, triples, shared)?.finish()
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
        down_idx: model.tensor_index(&format!("{prefix}input_mix_weight_down.weight")),
        up_idx: model.tensor_index(&format!("{prefix}input_mix_weight_up.weight")),
        inject_idx: inject
            .then(|| model.tensor_index(&format!("{prefix}block_inject_weight.weight")))
            .flatten(),
    })
}

impl GatedResidual {
    fn idxs(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend(self.down_idx);
        v.extend(self.up_idx);
        v.extend(self.inject_idx);
        v
    }
}

/// Resolve the device frame's matrices for one layer by name.
fn layer_idx(model: &CmfModel, p: &str, is_gdn: bool, has_ple: bool) -> Option<LayerIdx> {
    let ix = |n: String| model.tensor_index(&n);
    let gdn = if is_gdn {
        let q = format!("{p}linear_attn.");
        Some([
            ix(format!("{q}in_proj_qkv.weight"))?,
            ix(format!("{q}in_proj_z.weight"))?,
            ix(format!("{q}in_proj_a.weight"))?,
            ix(format!("{q}in_proj_b.weight"))?,
            ix(format!("{q}out_proj.weight"))?,
        ])
    } else {
        None
    };
    let qsa = if !is_gdn {
        let q = format!("{p}self_attn.");
        Some([
            ix(format!("{q}q_proj.weight"))?,
            ix(format!("{q}k_proj.weight"))?,
            ix(format!("{q}v_proj.weight"))?,
            ix(format!("{q}o_proj.weight"))?,
            ix(format!("{q}indexer.index_qk_proj.weight"))?,
        ])
    } else {
        None
    };
    let ple = if has_ple {
        Some((
            ix(format!("{p}ple.key_proj.weight"))?,
            ix(format!("{p}ple.value_proj.weight"))?,
        ))
    } else {
        None
    };
    Some(LayerIdx {
        gdn,
        qsa,
        ple,
        router: ix(format!("{p}mlp.gate.weight"))?,
        shared_gate: ix(format!("{p}mlp.shared_expert_gate.weight")),
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

    let head_hc = load_hc(model, "model.hyper_connection_mixer.", false)?;
    let lm_head_idx = model.tensor_index("lm_head.weight");
    let embed_idx = model.tensor_index("model.embed_tokens.weight");
    let mut skeleton_idxs: Vec<usize> = head_hc.idxs();
    skeleton_idxs.extend(lm_head_idx);
    // with an MTP sidecar the draft chain re-embeds on the card: the table
    // joins the skeleton so the expert arena is sized around it
    if std::env::var("CMF_QWEN_MTP").as_deref() != Ok("0")
        && std::env::var("CMF_QWEN_MTP_DEVICE_DRAFT").as_deref() == Ok("1")
    {
        let side = cortiq_core::mtp_sidecar_path(&model.path);
        if mtp_embedded(model) || (side != model.path && side.exists()) {
            skeleton_idxs.extend(embed_idx);
        }
    }
    let globals_partial = (
        t(model, "model.embed_tokens.weight")?,
        t(model, "lm_head.weight")?,
        head_hc,
    );
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
        let attn_hc = load_hc(model, &format!("{p}attn_hyper_connection."), true)?;
        let mlp_hc = load_hc(model, &format!("{p}mlp_hyper_connection."), true)?;
        let idx = layer_idx(model, &p, matches!(mixer, Mixer::Gdn(_)), ple.is_some());
        skeleton_idxs.extend(attn_hc.idxs());
        skeleton_idxs.extend(mlp_hc.idxs());
        if let Some(ix) = &idx {
            skeleton_idxs.extend(ix.gdn.iter().flatten().copied());
            skeleton_idxs.extend(ix.qsa.iter().flatten().copied());
            if let Some((k, v)) = ix.ple {
                skeleton_idxs.extend([k, v]);
            }
            skeleton_idxs.push(ix.router);
            skeleton_idxs.extend(ix.shared_gate);
        }
        layers.push(Layer {
            attn_hc,
            mlp_hc,
            mixer,
            moe,
            ple,
            expert_ids,
            shared_ids: Some(shared_ids),
            idx,
        });
    }
    let globals = Globals {
        embed: globals_partial.0,
        lm_head: globals_partial.1,
        head_hc: globals_partial.2,
        lm_head_idx,
        embed_idx,
        skeleton_idxs,
    };
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

/// `CMF_QWEN_DEVICE_TAP=<layer>`: the host path's intermediates of that
/// layer, kept for the device path to compare against (check mode).
static TAP: std::sync::Mutex<Vec<(String, usize, usize, Vec<f32>)>> =
    std::sync::Mutex::new(Vec::new());

fn tap_layer() -> Option<usize> {
    std::env::var("CMF_QWEN_DEVICE_TAP").ok()?.parse().ok()
}

#[cfg(feature = "cuda")]
thread_local! {
    /// Check mode of the CUDA path: while armed, every stage the host path
    /// observes on this thread is kept (label, layer, values).
    static CUDA_TAP: std::cell::RefCell<Option<Vec<(String, usize, Vec<f32>)>>> = const { std::cell::RefCell::new(None) };
}

fn observe(label: &str, li: usize, position: usize, values: &[f32]) {
    #[cfg(feature = "cuda")]
    CUDA_TAP.with(|t| {
        if let Some(rows) = t.borrow_mut().as_mut() {
            rows.push((label.to_string(), li, values.to_vec()));
        }
    });
    if tap_layer() == Some(li) && position < 3 {
        TAP.lock()
            .unwrap()
            .push((label.to_string(), li, position, values.to_vec()));
    }
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
    let ids = ple_row_ids(w, cfg, history, token);
    let mut out = vec![0.0f32; ids.len() * w.row_dim];
    for (hi, &id) in ids.iter().enumerate() {
        ple_row(w, id, &mut out[hi * w.row_dim..(hi + 1) * w.row_dim]);
    }
    out
}

/// One n-gram table row, dequantized into `dst` (`row_dim` long).
fn ple_row(w: &PleWeights, id: i64, dst: &mut [f32]) {
    let global = id as usize;
    let shard = global / w.rows_per_shard;
    let local = global % w.rows_per_shard;
    debug_assert!(shard < w.shards.len());
    w.shards[shard].row_f32(local, dst);
}

/// The table rows a token's n-gram heads read, in head order.
fn ple_row_ids(w: &PleWeights, cfg: &Cfg, history: &[u32], token: u32) -> Vec<i64> {
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
    ids
}

/// The n-gram table in RAM for good (`CMF_QWEN_PLE_LOCK`, on by default on the CUDA path; 0 = off): a decode
/// step reads a few of its ~90-byte rows on three pages each of a ~26 GB mapping, and a host whose page cache
/// does not keep the table (a 62 GB container with a 67 GB file) waits on the disk inside the step (thousands of
/// major faults in a request). The table's pages are registered with the CUDA driver (pinned) from a background
/// thread when the whole file fits beside it: the available RAM (the container's limit, if lower) must exceed the
/// file by 8 GB. Pinning the table where it does not (a 62 GB container, a 67 GB file) evicts the pages a prompt
/// needs and made every prompt wait on the disk (TTFT 12-15 s instead of 1.5-3 s). `CMF_QWEN_PLE_LOCK=1` pins
/// whatever the memory, `0` never. Returns at once.
#[cfg(feature = "gpu")]
fn ple_lock(layers: &[Layer]) {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        static DONE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        let mode = std::env::var("CMF_QWEN_PLE_LOCK").ok().map(|v| v.trim().to_string());
        if DONE.set(()).is_err() || mode.as_deref() == Some("0") {
            return;
        }
        // SAFETY: plain sysconf query.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut file = 0u64;
        for l in layers {
            let Some(pw) = l.ple.as_ref() else { continue };
            if file == 0 {
                file = pw.shards.first().and_then(|s| s.model_arc()).and_then(|m| std::fs::metadata(&m.path).ok()).map_or(0, |md| md.len());
            }
            for sh in &pw.shards {
                if let Some(b) = sh.mapped_bytes() {
                    let (a, len) = (b.as_ptr() as usize, b.len());
                    let start = a / page * page;
                    spans.push((start, (a + len).div_ceil(page) * page - start));
                }
            }
        }
        spans.sort_unstable();
        spans.dedup();
        let bytes: usize = spans.iter().map(|s| s.1).sum();
        if bytes == 0 {
            return;
        }
        let avail = mem_available();
        let need = (file as usize).max(bytes) + (8 << 30);
        if mode.as_deref() != Some("1") && avail < need {
            eprintln!(
                "qwen4: n-gram table: {:.1} GB of a {:.1} GB file, {:.1} GB of RAM available: left to the page cache",
                bytes as f64 / 1e9,
                file as f64 / 1e9,
                avail as f64 / 1e9
            );
            return;
        }
        crate::gpu_cuda::qwen4::pin_host_for_good("n-gram table", spans);
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    let _ = layers;
}

/// Bytes of RAM this process may still take: MemAvailable, or the container's memory limit minus its anonymous
/// memory when lower.
#[cfg(all(feature = "gpu", any(target_os = "linux", target_os = "android")))]
fn mem_available() -> usize {
    let kb = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<usize>().ok()))
        .unwrap_or(0);
    let mut avail = kb * 1024;
    // the container's limit minus what it cannot give back (its page cache can be)
    let max = std::fs::read_to_string("/sys/fs/cgroup/memory.max").ok().and_then(|v| v.trim().parse::<usize>().ok());
    let anon = std::fs::read_to_string("/sys/fs/cgroup/memory.stat")
        .ok()
        .and_then(|m| m.lines().find(|l| l.starts_with("anon ")).and_then(|l| l.split_whitespace().nth(1)?.parse::<usize>().ok()));
    if let (Some(max), Some(anon)) = (max, anon) {
        avail = avail.min(max.saturating_sub(anon));
    }
    avail
}

/// Threads that ask the kernel for a prompt chunk's n-gram rows ahead of the
/// gather (`CMF_QWEN_PLE_PREFETCH`, default 32; 0 turns the read-ahead off).
#[cfg(feature = "gpu")]
fn ple_prefetch_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("CMF_QWEN_PLE_PREFETCH")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(32)
            .min(256)
    })
}

/// Read-ahead for a prompt chunk's n-gram rows. A row of the table is ~90
/// bytes on three pages of a ~26 GB mapping, so a host whose page cache does
/// not hold the table takes three disk reads per row: 160 thousand faults
/// for a 3.4K-token prompt, which the gather takes a few at a time, one per
/// pool worker. The row ids depend on the tokens alone, so the pages are
/// named before the first frame is built and handed to the kernel as
/// `MADV_WILLNEED` from detached threads: the reads then run at the disk's
/// queue depth while the first frame and layer 0 occupy the card, and the
/// gather finds the rows in the page cache. Advice only; returns the number
/// of rows asked for.
#[cfg(feature = "gpu")]
fn ple_prefetch(layers: &[Layer], cfg: &Cfg, history: &[u32], ids: &[u32]) -> usize {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    {
        let threads = ple_prefetch_threads();
        if threads == 0 || ids.len() < 64 {
            return 0;
        }
        let t0 = std::time::Instant::now();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut model = None;
        let mut hist: Vec<u32> = history.to_vec();
        for l in layers {
            let Some(pw) = l.ple.as_ref() else { continue };
            if model.is_none() {
                model = pw.shards.first().and_then(|s| s.model_arc());
            }
            hist.truncate(history.len());
            for &id in ids {
                for r in ple_row_ids(pw, cfg, &hist, id) {
                    let global = r as usize;
                    if let Some(s) = pw.shards.get(global / pw.rows_per_shard) {
                        s.row_spans(global % pw.rows_per_shard, &mut spans);
                    }
                }
                hist.push(id);
            }
        }
        let Some(model) = model else { return 0 };
        if spans.is_empty() {
            return 0;
        }
        let (rows, ntok) = (spans.len() / 3, ids.len());
        let prof = std::env::var_os("CMF_QWEN_PROF").is_some();
        // the page list, its sort and the advice itself stay off the request's thread
        let _ = std::thread::Builder::new().name("ple-readahead".into()).spawn(move || {
            // SAFETY: plain sysconf query.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;
            let mut pages: Vec<usize> = Vec::with_capacity(spans.len() + spans.len() / 8);
            for &(a, len) in &spans {
                pages.extend(a / page..=(a + len - 1) / page);
            }
            pages.sort_unstable();
            pages.dedup();
            // adjacent pages go to the kernel as one range
            let mut runs: Vec<(usize, usize)> = Vec::with_capacity(pages.len());
            for &p in &pages {
                match runs.last_mut() {
                    Some((s, n)) if *s + *n == p => *n += 1,
                    _ => runs.push((p, 1)),
                }
            }
            let threads = threads.min(runs.len().div_ceil(512)).max(1);
            let runs = &runs;
            std::thread::scope(|sc| {
                for t in 0..threads {
                    sc.spawn(move || {
                        for &(p, n) in runs.iter().skip(t).step_by(threads) {
                            // SAFETY: whole pages of our read-only mapping,
                            // which `model` keeps alive; advice only.
                            unsafe {
                                libc::madvise(
                                    (p * page) as *mut libc::c_void,
                                    n * page,
                                    libc::MADV_WILLNEED,
                                );
                            }
                        }
                    });
                }
            });
            if prof {
                eprintln!(
                    "qwen4-ple: read-ahead of {} pages in {} ranges for {ntok} tokens on {threads} threads, issued in {:.1}ms",
                    pages.len(),
                    runs.len(),
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            drop(model);
        });
        rows
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    {
        let _ = (layers, cfg, history, ids);
        0
    }
}

/// The n-gram rows of every PLE layer for every token of a frame, each
/// against its own history: `[layer][token]` → heads × `row_dim`. The rows
/// are scattered over a ~25 GB table, so on a host whose page cache does
/// not hold the file every row is a disk read; read one after another they
/// cost tens of milliseconds per token. They are independent, so the pool
/// reads them all at once.
fn ple_frame_rows(
    layers: &[Layer],
    cfg: &Cfg,
    history: &[u32],
    ids: &[u32],
    pool: Option<&Pool>,
) -> Vec<Vec<Vec<f32>>> {
    ple_rows_of(layers, cfg, history, ids, pool, None)
}

/// `ple_frame_rows`, for layer `only` alone when given (the others empty).
fn ple_rows_of(
    layers: &[Layer],
    cfg: &Cfg,
    history: &[u32],
    ids: &[u32],
    pool: Option<&Pool>,
    only: Option<usize>,
) -> Vec<Vec<Vec<f32>>> {
    let mut out: Vec<Vec<Vec<f32>>> = Vec::with_capacity(layers.len());
    // (layer, token, head, row id), and the destination of each row
    let mut jobs: Vec<(usize, usize, usize, i64)> = Vec::new();
    let mut hist: Vec<u32> = history.to_vec();
    for (li, l) in layers.iter().enumerate() {
        let Some(pw) = l.ple.as_ref().filter(|_| only.is_none_or(|o| o == li)) else {
            out.push(Vec::new());
            continue;
        };
        hist.truncate(history.len());
        let mut rows = Vec::with_capacity(ids.len());
        for (t, &id) in ids.iter().enumerate() {
            let rid = ple_row_ids(pw, cfg, &hist, id);
            rows.push(vec![0.0f32; rid.len() * pw.row_dim]);
            jobs.extend(rid.into_iter().enumerate().map(|(h, r)| (li, t, h, r)));
            hist.push(id);
        }
        out.push(rows);
    }
    // the tables alone: a layer also holds host-side caches that are not Sync
    let tables: Vec<Option<&PleWeights>> = layers.iter().map(|l| l.ple.as_ref()).collect();
    let dst: Vec<crate::pool::SendMut> = jobs
        .iter()
        .map(|&(li, t, h, _)| {
            let dim = tables[li].map_or(0, |w| w.row_dim);
            crate::pool::SendMut::new(out[li][t][h * dim..].as_mut_ptr())
        })
        .collect();
    let read = |s: usize, e: usize| {
        for j in s..e {
            let (li, _, _, id) = jobs[j];
            let Some(w) = tables[li] else {
                continue;
            };
            // SAFETY: every job owns a distinct `row_dim` slice of `out`,
            // which outlives the dispatch.
            let row = unsafe { std::slice::from_raw_parts_mut(dst[j].at(0), w.row_dim) };
            ple_row(w, id, row);
        }
    };
    match pool {
        Some(p) if jobs.len() > 1 => p.run_rows(jobs.len(), &read),
        _ => read(0, jobs.len()),
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
    // the expert pool below is the wgpu one
    #[cfg(feature = "cuda")]
    if crate::gpu_cuda::selected() {
        return None;
    }
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
        bf16: false,
    };
    let mut out = vec![0.0f32; x.len()];
    let mut cold = Vec::new();
    let mut cold_x = Vec::new();
    let (frame_ok, mut cold_cpu) = std::thread::scope(|scope| {
        // The resident kernels know no tubes: every winner's come below.
        let cpu = (!cold_jobs.is_empty())
            .then(|| scope.spawn(|| crate::pipeline::moe_cold_experts_cpu_with(&cold_jobs, x, pool, false)));
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
    // The experts' hot neurons (`hot-bake --experts`), resident or cold alike, and the
    // shared expert's tube (`--shared-q8`).
    for &e in &picks {
        crate::pipeline::expert_tubes_add(&m.experts[e], x, mix_weights[e], &mut out, pool);
    }
    if let Some((se, _)) = &m.shared {
        crate::pipeline::expert_tubes_add(se, x, shared_weight, &mut out, pool);
    }
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
    #[cfg(feature = "cuda")]
    if cuda_forward(globals, layers, cfg, state, &[token_id], position, inv_freq, pool, logits, want_logits) {
        return;
    }
    #[cfg(feature = "gpu")]
    if forward_token_device(
        globals,
        layers,
        cfg,
        state,
        token_id,
        position,
        inv_freq,
        pool,
        logits,
        want_logits,
    ) {
        return;
    }
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
    if state.host_pos != position && state.token_history.len() == position {
        // The device path ran positions host_pos..position and then turned
        // off (a refused frame): the host's K/V, GDN and PLE caches never
        // saw them, and attention would index keys that were never stored.
        // Rebuild them from the token history (slow, but the text stays
        // that of the device path).
        tracing::warn!(
            "qwen4: host caches rebuilt over {position} tokens after the device path turned off"
        );
        if std::env::var_os("CMF_QWEN_PROF").is_some() {
            eprintln!("qwen4: replaying {position} tokens on the host path");
        }
        let history = std::mem::take(&mut state.token_history);
        state.reset();
        for (p, &id) in history.iter().enumerate() {
            let mut lg = Vec::new();
            forward_token(
                globals, layers, cfg, state, id, p, inv_freq, pool, &mut lg, false,
            );
        }
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
    state.host_pos = position + 1;
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

#[cfg(feature = "gpu")]
fn device_geom(cfg: &Cfg, moe: &MoeFfn, gu_q2: bool) -> crate::gpu_wgpu::qwen4::Geom {
    use crate::gpu_wgpu::qwen4::{GdnGeom, Geom};
    Geom {
        hidden: cfg.hidden,
        hc: cfg.hc,
        eps: cfg.eps as f32,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        rotary_dim: cfg.rotary_dim,
        index_heads: cfg.index_heads,
        index_dim: cfg.index_dim,
        index_budget: cfg.index_budget,
        compress_ratio: cfg.compress_ratio,
        gdn: GdnGeom {
            nv: cfg.gdn.num_v_heads,
            nk: cfg.gdn.num_k_heads,
            dk: cfg.gdn.key_head_dim,
            dv: cfg.gdn.value_head_dim,
            kk: cfg.gdn.conv_kernel,
        },
        ple_kernel: cfg.ple_kernel,
        ple_dilation: cfg.ple_dilation,
        top_k: moe.top_k,
        n_experts: moe.experts.len(),
        inter: moe.experts.first().map_or(0, |e| e.gate_proj.rows()),
        gu_q2,
    }
}

#[cfg(feature = "gpu")]
fn device_layer_w<'a>(
    layer: &'a Layer,
    ix: &LayerIdx,
) -> Option<crate::gpu_wgpu::qwen4::LayerW<'a>> {
    use crate::gpu_wgpu::qwen4::{HcW, LayerW, MixerW, PleW};
    let hc = |g: &'a GatedResidual| -> Option<HcW<'a>> {
        Some(HcW {
            norm: &g.norm,
            down: g.down_idx?,
            up: g.up_idx?,
            inject: g.inject_idx,
        })
    };
    let mixer = match &layer.mixer {
        Mixer::Gdn(w) => {
            let [qkv, z, a, b, out] = ix.gdn?;
            MixerW::Gdn {
                qkv,
                z,
                a,
                b,
                out,
                conv1d: &w.conv1d,
                a_log: &w.a_log,
                dt_bias: &w.dt_bias,
                norm: &w.norm,
            }
        }
        Mixer::Qsa(w) => {
            let [q, k, v, o, index_qk] = ix.qsa?;
            MixerW::Qsa {
                q,
                k,
                v,
                o,
                index_qk,
                q_norm: &w.q_norm,
                k_norm: &w.k_norm,
                iq_norm: &w.index_q_norm,
                ik_norm: &w.index_k_norm,
            }
        }
    };
    let ple = match (&layer.ple, ix.ple) {
        (Some(w), Some((key_proj, value_proj))) => Some(PleW {
            key_proj,
            value_proj,
            norm_key: &w.norm_key,
            norm_query: &w.norm_query,
            norm_conv: &w.norm_conv,
            conv: &w.conv,
        }),
        (None, None) => None,
        _ => return None,
    };
    Some(LayerW {
        attn_hc: hc(&layer.attn_hc)?,
        mlp_hc: hc(&layer.mlp_hc)?,
        mixer,
        ple,
        router: ix.router,
        shared_gate: ix.shared_gate,
    })
}

#[cfg(feature = "gpu")]
#[allow(clippy::type_complexity)]
static DEV_TAP: std::sync::Mutex<Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>> =
    std::sync::Mutex::new(Vec::new());

#[cfg(feature = "gpu")]
fn tap_compare(label: &str, host: &[f32], dev: &[f32]) {
    let n = host.len().min(dev.len());
    let mut max_abs = 0.0f32;
    let mut at = 0usize;
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let d = (host[i] - dev[i]).abs();
        if d > max_abs {
            max_abs = d;
            at = i;
        }
        dot += host[i] as f64 * dev[i] as f64;
        na += host[i] as f64 * host[i] as f64;
        nb += dev[i] as f64 * dev[i] as f64;
    }
    eprintln!(
        "qwen4-tap {label:>10}: cos={:.6} max_abs={max_abs:.4} at {at} (host {:.4} dev {:.4}) |host|={:.3} |dev|={:.3} n={n}",
        dot / (na.sqrt() * nb.sqrt()).max(1e-30),
        host.get(at).copied().unwrap_or(0.0),
        dev.get(at).copied().unwrap_or(0.0),
        na.sqrt(),
        nb.sqrt()
    );
}

/// Routing hits per `(layer, expert)` observed by the device path, and the
/// sidecar that carries them between runs (`CMF_QWEN_PROFILE`): a warm arena
/// from the first token instead of a few hundred tokens of LRU learning.
#[cfg(feature = "gpu")]
struct ExpertProfile {
    n_layers: usize,
    n_experts: usize,
    counts: Vec<u32>,
    tokens: u64,
}

#[cfg(feature = "gpu")]
impl ExpertProfile {
    const MAGIC: &'static [u8; 8] = b"CMFQ4PF\0";

    fn new(n_layers: usize, n_experts: usize) -> Self {
        Self {
            n_layers,
            n_experts,
            counts: vec![0; n_layers * n_experts],
            tokens: 0,
        }
    }

    fn load(path: &str, n_layers: usize, n_experts: usize) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        if bytes.len() < 24 || &bytes[..8] != Self::MAGIC {
            return None;
        }
        let rd = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        let (nl, ne) = (rd(8) as usize, rd(16) as usize);
        if nl != n_layers || ne != n_experts || bytes.len() < 32 + nl * ne * 4 {
            return None;
        }
        let tokens = rd(24);
        let counts = bytes[32..32 + nl * ne * 4]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Some(Self {
            n_layers,
            n_experts,
            counts,
            tokens,
        })
    }

    fn save(&self, path: &str) {
        let mut out = Vec::with_capacity(32 + self.counts.len() * 4);
        out.extend_from_slice(Self::MAGIC);
        out.extend_from_slice(&(self.n_layers as u64).to_le_bytes());
        out.extend_from_slice(&(self.n_experts as u64).to_le_bytes());
        out.extend_from_slice(&self.tokens.to_le_bytes());
        for &c in &self.counts {
            out.extend_from_slice(&c.to_le_bytes());
        }
        if let Err(e) = std::fs::write(path, out) {
            tracing::warn!("expert profile not saved to {path}: {e}");
        }
    }

    fn note(&mut self, layer: usize, picks: &[usize]) {
        for &e in picks {
            if let Some(c) = self.counts.get_mut(layer * self.n_experts + e) {
                *c = c.saturating_add(1);
            }
        }
    }

    /// `(layer, expert)` pairs, hottest first, zero counts left out.
    fn ranked(&self) -> Vec<(usize, usize)> {
        let mut v: Vec<(u32, usize, usize)> = self
            .counts
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(i, &c)| (c, i / self.n_experts, i % self.n_experts))
            .collect();
        v.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        v.into_iter().map(|(_, l, e)| (l, e)).collect()
    }
}

/// `CMF_QWEN_ROUTE_WSTAT=1` (with `CMF_QWEN_LEAN=0`, so the route is read back): how light the routed winners'
/// tail is — the share of winners under 1/2/3/5/8/10 % of their token's routing and the routing they carry.
#[cfg(feature = "gpu")]
fn route_wstat() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("CMF_QWEN_ROUTE_WSTAT").as_deref() == Ok("1"))
}

/// One token's routed (expert, weight bits) pairs into the tail statistics; a line every 9600 token-layers.
#[cfg(feature = "gpu")]
fn note_route_weights(pairs: &[u32]) {
    const CUTS: [f32; 6] = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10];
    // token-layers, winners, winners under each cut, their routing
    static ACC: std::sync::Mutex<(u64, u64, [u64; 6], [f64; 6])> = std::sync::Mutex::new((0, 0, [0; 6], [0.0; 6]));
    let mut a = ACC.lock().unwrap();
    a.0 += 1;
    for pr in pairs.chunks_exact(2) {
        if pr[0] == u32::MAX {
            continue;
        }
        let w = f32::from_bits(pr[1]);
        a.1 += 1;
        for (i, &c) in CUTS.iter().enumerate() {
            if w < c {
                a.2[i] += 1;
                a.3[i] += f64::from(w);
            }
        }
    }
    if a.0 % 9600 == 0 {
        let share: Vec<String> = a.2.iter().map(|&n| format!("{:.1}", 100.0 * n as f64 / a.1.max(1) as f64)).collect();
        let mass: Vec<String> = a.3.iter().map(|&m| format!("{:.2}", 100.0 * m / a.0 as f64)).collect();
        eprintln!(
            "route weights: {} token-layers, {:.2} winners each; under 1/2/3/5/8/10 %: {} % of the winners, {} % of the routing",
            a.0,
            a.1 as f64 / a.0 as f64,
            share.join("/"),
            mass.join("/")
        );
    }
}

/// Vocabulary rows the device draft chain's head projects
/// (`CMF_QWEN_DRAFT_VOCAB`, 0 = the whole vocabulary).
#[cfg(feature = "gpu")]
fn draft_vocab_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("CMF_QWEN_DRAFT_VOCAB")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// The draft gate (`CMF_QWEN_DRAFT_MINP`, 0 = off): a draft joins the verify
/// window while the chain's cumulative confidence (the product of the
/// probabilities its head gave its own picks) stays at or above this. And
/// whether the chain reports its confidences at all (the gate, or
/// `CMF_QWEN_DRAFT_CONF=1` to print them with the gate off).
#[cfg(feature = "gpu")]
fn draft_gate() -> (f32, bool) {
    static V: std::sync::OnceLock<(f32, bool)> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let minp = std::env::var("CMF_QWEN_DRAFT_MINP")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|p| p.is_finite() && *p > 0.0)
            .map_or(0.0, |p| p.min(1.0));
        let report = minp > 0.0 || std::env::var("CMF_QWEN_DRAFT_CONF").as_deref() == Ok("1");
        (minp, report)
    })
}

/// How many of a chain's drafts the gate keeps: the longest prefix whose
/// cumulative confidence is at least `minp`, never fewer than one (the
/// round's bonus token rides the same window either way).
#[cfg(feature = "gpu")]
fn draft_gate_keep(conf: &[f32], minp: f32) -> usize {
    if minp <= 0.0 {
        return conf.len();
    }
    let mut cum = 1.0f32;
    let mut keep = 0;
    for &p in conf {
        cum *= p;
        // a NaN confidence ends the prefix too
        if cum.is_nan() || cum < minp {
            break;
        }
        keep += 1;
    }
    keep.max(1).min(conf.len())
}

/// Suffix drafts (`CMF_QWEN_SUFFIX=EXT:STRONG:MAX`, off by default): drafts
/// read off the context itself, at no card time. `strong`: a context tail
/// of at least this many tokens that occurred before is continued from
/// there with no draft chain at all; `ext`: the chain's drafts are extended
/// the same way when the tail with them on it (this many tokens) occurred
/// before; `max`: drafts a window carries at most. 0 turns a mode off.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, PartialEq)]
struct SuffixCfg {
    ext: usize,
    strong: usize,
    max: usize,
}

#[cfg(feature = "gpu")]
fn suffix_parse(raw: &str) -> Option<SuffixCfg> {
    let mut it = raw.split(':').map(|v| v.trim().parse::<usize>().ok());
    let ext = it.next()??;
    let strong = it.next().unwrap_or(Some(0))?;
    let max = it.next().unwrap_or(Some(usize::MAX))?;
    let max = max.clamp(1, crate::gpu_wgpu::qwen4::TMAX - 1);
    (ext > 0 || strong > 0).then_some(SuffixCfg { ext, strong, max })
}

#[cfg(feature = "gpu")]
fn suffix_cfg() -> Option<SuffixCfg> {
    static V: std::sync::OnceLock<Option<SuffixCfg>> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("CMF_QWEN_SUFFIX").ok().and_then(|v| suffix_parse(&v)))
}

/// `CMF_QWEN_COPY_FRAME=N[:F]`: a copy round (`CMF_QWEN_SUFFIX`'s strong match) that has
/// held for a while verifies up to N drafts as one frame of the prompt kernels (31, 63,
/// 127 or 255 drafts: 32 to 256 tokens) instead of a window of `TMAX`. A frame carries no
/// more drafts than F times the tokens the copy has given so far (default 2). Unset:
/// windows.
#[cfg(feature = "gpu")]
fn copy_frame_cfg() -> Option<(usize, f32)> {
    static V: std::sync::OnceLock<Option<(usize, f32)>> = std::sync::OnceLock::new();
    *V.get_or_init(|| copy_frame_parse(&std::env::var("CMF_QWEN_COPY_FRAME").ok()?))
}

/// `CMF_QWEN_FRAME_DEBUG=1`: a copy frame reads every row's logits back and prints the
/// card's row ids beside the host's argmax of the same rows. A probe, not a release path.
#[cfg(feature = "gpu")]
fn copy_frame_debug() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("CMF_QWEN_FRAME_DEBUG").as_deref() == Ok("1"))
}

#[cfg(feature = "gpu")]
fn copy_frame_parse(raw: &str) -> Option<(usize, f32)> {
    let mut it = raw.split(':');
    let n = it.next()?.trim().parse::<usize>().ok()?;
    let f = match it.next() {
        Some(v) => v.trim().parse::<f32>().ok().filter(|f| f.is_finite() && *f > 0.0)?,
        None => 2.0,
    };
    (n >= COPY_FRAMES[COPY_FRAMES.len() - 1]).then_some((n.min(crate::gpu_wgpu::qwen4::TROWS - 1), f))
}

/// The drafts a copy frame carries, largest first: a few fixed sizes, so that each builds
/// its bind groups once. A frame's launches cost what they cost whatever it carries (RTX
/// 5090: 84 ms for 32 tokens, 110 ms for 64, where a window of 8 takes 24 ms), so a frame
/// of 16 tokens (70 ms) would lose to two windows.
#[cfg(feature = "gpu")]
const COPY_FRAMES: [usize; 4] = [255, 127, 63, 31];

/// The drafts of the next copy frame: the largest size within the configured most, the
/// copy's run so far times the factor, and what the request still takes. None: a window.
#[cfg(feature = "gpu")]
fn copy_frame_drafts(most: usize, factor: f32, run: usize, budget: usize) -> Option<usize> {
    let lim = most.min(budget).min((run as f32 * factor) as usize);
    COPY_FRAMES.into_iter().find(|&k| k <= lim)
}

/// The longest earlier occurrence of the context's tail (at least `min_m`
/// tokens; the latest one on a tie): its length and the index of its last
/// token. Length 0: the tail did not occur before.
#[cfg(feature = "gpu")]
fn suffix_find(ids: &[u32], min_m: usize) -> (usize, usize) {
    // a match this long is as good as any longer one
    const CAP: usize = 32;
    let n = ids.len();
    if min_m == 0 || n <= min_m {
        return (0, 0);
    }
    let last = ids[n - 1];
    let (mut best_m, mut best_j) = (0usize, 0usize);
    for j in (min_m - 1..n - 1).rev() {
        if ids[j] != last {
            continue;
        }
        let mut m = 1;
        while m < CAP && m <= j && ids[j - m] == ids[n - 1 - m] {
            m += 1;
        }
        if m > best_m {
            (best_m, best_j) = (m, j);
            if m >= CAP {
                break;
            }
        }
    }
    if best_m < min_m { (0, 0) } else { (best_m, best_j) }
}

/// The longest earlier occurrence of the context's tail (at least `min_m`
/// tokens; the latest one on a tie) and the up to `max_out` tokens that
/// followed it there. An occurrence that runs into the tail itself (a
/// period shorter than the draft) continues through its own prediction.
/// `(0, [])` when the tail did not occur before.
#[cfg(feature = "gpu")]
fn suffix_draft(ids: &[u32], min_m: usize, max_out: usize) -> (usize, Vec<u32>) {
    let n = ids.len();
    if min_m == 0 || max_out == 0 || n <= min_m {
        return (0, Vec::new());
    }
    let (best_m, best_j) = suffix_find(ids, min_m);
    if best_m < min_m {
        return (0, Vec::new());
    }
    let mut out: Vec<u32> = Vec::with_capacity(max_out);
    for i in 0..max_out {
        let p = best_j + 1 + i;
        out.push(if p < n { ids[p] } else { out[p - n] });
    }
    (best_m, out)
}

/// Tokens a copy may be interrupted by before the cursor lets go of its source.
#[cfg(feature = "cuda")]
const GAP_MAX: usize = 12;

/// The copy rounds of the CUDA path. The model's answer often follows an
/// earlier text of the context token by token, with insertions at the same
/// places (an edit: the same comment line before every `def`). The cursor
/// keeps the source position the next token should come from, and learns the
/// insertions as rules `(token before, token after) -> inserted tokens`, so
/// that a draft carries the insertion across, not only the copy.
#[cfg(feature = "cuda")]
#[derive(Default)]
struct CopyCur {
    /// Tokens of the context already accounted for.
    seen: usize,
    /// The index of the context token the next token should repeat.
    at: Option<usize>,
    /// Tokens that came as expected since the cursor was last disturbed.
    run: usize,
    /// Tokens inserted since the last copied one.
    gap: Vec<u32>,
    last_gap: Vec<u32>,
    /// (token before, token after) -> (inserted tokens, times they came).
    rules: std::collections::HashMap<(u32, u32), (Vec<u32>, u8)>,
}

#[cfg(feature = "cuda")]
impl CopyCur {
    fn key(ids: &[u32], at: usize) -> Option<(u32, u32)> {
        (at >= 1 && at < ids.len()).then(|| (ids[at - 1], ids[at]))
    }

    fn start(&mut self, at: usize, run: usize) {
        self.at = Some(at);
        self.run = run;
        self.gap.clear();
    }

    fn drop_cursor(&mut self) {
        self.at = None;
        self.run = 0;
        self.gap.clear();
    }

    /// Account for the tokens of `ids` the cursor has not seen.
    fn account(&mut self, ids: &[u32]) {
        if self.seen > ids.len() {
            *self = Self::default();
        }
        for j in self.seen..ids.len() {
            let Some(at) = self.at else { break };
            if at >= j {
                self.drop_cursor();
                break;
            }
            let key = Self::key(ids, at);
            if ids[j] == ids[at] {
                if !self.gap.is_empty() {
                    let g = std::mem::take(&mut self.gap);
                    if let Some(key) = key {
                        match self.rules.get_mut(&key) {
                            Some((rg, hits)) if *rg == g => *hits = hits.saturating_add(1),
                            _ => {
                                self.rules.insert(key, (g.clone(), 1));
                            }
                        }
                    }
                    self.last_gap = g;
                } else if let Some(key) = key
                    && let Some(r) = self.rules.get_mut(&key)
                {
                    // an insertion was expected here and did not come
                    if r.1 <= 1 {
                        self.rules.remove(&key);
                    } else {
                        r.1 -= 1;
                    }
                }
                self.at = Some(at + 1);
                self.run += 1;
            } else {
                self.gap.push(ids[j]);
                let expected = key.and_then(|k| self.rules.get(&k)).is_some_and(|(rg, _)| rg.starts_with(&self.gap));
                if !expected {
                    self.run = 0;
                }
                if self.gap.len() > GAP_MAX {
                    self.drop_cursor();
                    break;
                }
            }
        }
        self.seen = ids.len();
    }

    /// Up to `most` tokens the answer should continue with: the rest of an
    /// insertion in progress, then the copy with the insertions the rules know
    /// (seen twice at least). Empty: the cursor cannot say what comes next.
    fn draft(&self, ids: &[u32], most: usize) -> Vec<u32> {
        let Some(at) = self.at else { return Vec::new() };
        let mut out: Vec<u32> = Vec::with_capacity(most);
        let mut skip = None;
        if !self.gap.is_empty() {
            let g = &self.gap;
            let rule = Self::key(ids, at).and_then(|k| self.rules.get(&k)).map(|(rg, _)| rg.as_slice());
            let rest: &[u32] = match rule {
                Some(rg) if rg.len() >= g.len() && rg[..g.len()] == g[..] => &rg[g.len()..],
                _ if self.last_gap.len() >= g.len() && self.last_gap[..g.len()] == g[..] => &self.last_gap[g.len()..],
                _ => return Vec::new(),
            };
            out.extend_from_slice(&rest[..rest.len().min(most)]);
            skip = Some(at);
        }
        let mut i = at;
        while out.len() < most && i < ids.len() {
            if i >= 1
                && skip != Some(i)
                && let Some((rg, hits)) = self.rules.get(&(ids[i - 1], ids[i]))
                && *hits >= 2
            {
                for &t in rg {
                    if out.len() < most {
                        out.push(t);
                    }
                }
                if out.len() >= most {
                    break;
                }
            }
            out.push(ids[i]);
            i += 1;
        }
        out
    }
}

/// The copy rounds of the CUDA path are on unless `CMF_QWEN_SUFFIX=0` (or
/// another value `suffix_parse` refuses); unset: a strong match of 8 tokens,
/// windows of 7 drafts.
#[cfg(feature = "cuda")]
fn cuda_copy_cfg() -> Option<SuffixCfg> {
    static V: std::sync::OnceLock<Option<SuffixCfg>> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("CMF_QWEN_SUFFIX") {
        Ok(v) => suffix_parse(&v),
        Err(_) => Some(SuffixCfg { ext: 4, strong: 8, max: 7 }),
    })
}

/// One speculation round on the CUDA card: the copy cursor drafts, the card
/// verifies the window (or a frame of up to 255 drafts with
/// `CMF_QWEN_COPY_FRAME`) in one pass and keeps the rows that stand.
/// `None`: the caller decodes one token plainly.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_spec_round(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    next_pos: usize,
    all_ids: &[u32],
    pool: Option<&Pool>,
) -> Option<SpecRound> {
    if state.cuda_off || next_pos == 0 || all_ids.len() != next_pos + 1 || state.pos != next_pos || state.token_history.len() != next_pos {
        return None;
    }
    let sc = cuda_copy_cfg();
    let tape = match state.cuda.as_ref() {
        Some(d) if d.pos == next_pos => d.tape(),
        _ => return None,
    };
    let budget = state.draft_budget;
    let mut drafts: Vec<u32> = Vec::new();
    if let Some(sc) = sc {
        state.cuda_copy.account(all_ids);
        if state.cuda_copy.at.is_none() && sc.strong > 0 {
            let (m, j) = suffix_find(all_ids, sc.strong);
            if m >= sc.strong {
                state.cuda_copy.start(j + 1, 1);
            }
        }
        if state.cuda_copy.at.is_some() {
            let run = state.cuda_copy.run;
            let mut most = (1 + 2 * run).min(sc.max);
            if let Some((fmax, factor)) = copy_frame_cfg() {
                let lim = fmax.min(budget).min((run as f32 * factor) as usize).min(tape.saturating_sub(1));
                if lim >= 63 {
                    most = lim;
                }
            }
            most = most.min(budget);
            if most > 0 {
                drafts = state.cuda_copy.draft(all_ids, most);
            }
        }
    }
    // no copy to follow: the MTP head drafts
    let copied = !drafts.is_empty();
    let t_draft0 = std::time::Instant::now();
    if drafts.is_empty() && budget > 0 {
        if let (Some(head), Some(dev)) = (state.mtp.as_ref(), state.cuda.as_mut()) {
            if dev.has_mtp() {
                let embed = |id: u32, row: &mut [f32]| {
                    if (id as usize) < globals.embed.rows() {
                        globals.embed.row_f32(id as usize, row);
                    }
                };
                let kmax = head.k.min(budget).min(tape.saturating_sub(1)).min(7);
                let mut out: Vec<u32> = Vec::new();
                match dev.draft(&head.layer, &embed, all_ids[next_pos], next_pos, kmax, &all_ids[all_ids.len().saturating_sub(256)..], &mut out) {
                    Ok(true) => drafts = out,
                    Ok(false) => {}
                    Err(e) => {
                        eprintln!("qwen4 cuda: the MTP head failed ({e}); it stays off");
                        dev.mtp_off();
                    }
                }
            }
        }
    }
    let t_draft = t_draft0.elapsed();
    let k = drafts.len();
    if k == 0 {
        return None;
    }
    // `CMF_QWEN_NCU=<n>`: round number n is the profiler's range
    static ROUNDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let ncu_round = std::env::var("CMF_QWEN_NCU").ok().and_then(|v| v.trim().parse::<usize>().ok());
    let this_round = ROUNDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let ncu_on = ncu_round == Some(this_round);
    let ids: Vec<u32> = std::iter::once(all_ids[next_pos]).chain(drafts.iter().copied()).collect();
    let mut dev = state.cuda.take()?;
    let h = cfg.hidden;
    let no_ple = std::env::var_os("CMF_QWEN_NO_PLE").is_some();
    let prof = std::env::var_os("CMF_QWEN_PROF").is_some();
    let t_round = std::time::Instant::now();
    if ncu_on {
        dev.profiler_range(true);
    }
    let res = (|| -> Result<(usize, Vec<f32>), String> {
        let mut emb = vec![0.0f32; ids.len() * h];
        for (row, &id) in emb.chunks_mut(h).zip(&ids) {
            if (id as usize) < globals.embed.rows() {
                globals.embed.row_f32(id as usize, row);
            }
        }
        let t_host = t_round.elapsed();
        let mut row_ids: Vec<u32> = Vec::new();
        let t_spec = std::time::Instant::now();
        let history = &state.token_history;
        let rows_of = |li: usize| {
            if no_ple { Vec::new() } else { ple_rows_of(layers, cfg, history, &ids, pool, Some(li)).into_iter().nth(li).unwrap_or_default() }
        };
        dev.spec(layers, &emb, &rows_of, next_pos, &mut row_ids)?;
        let t_spec = t_spec.elapsed();
        let t_settle = std::time::Instant::now();
        if row_ids.len() != ids.len() {
            return Err("qwen4 cuda: a window came back short".into());
        }
        let mut a = 0;
        while a < k && row_ids[a] == drafts[a] {
            a += 1;
        }
        let mut logits: Vec<f32> = Vec::new();
        let settled = dev.settle(a + 1, &mut logits)?;
        let t_settle = t_settle.elapsed();
        if !settled {
            // too long a window to settle in part: the rows that stand run again
            let part: Vec<Vec<Vec<f32>>> = if no_ple { Vec::new() } else { ple_frame_rows(layers, cfg, &state.token_history, &ids[..a + 1], pool) };
            dev.forward(layers, &emb[..(a + 1) * h], &part, next_pos, Some(&mut logits))?;
        }
        if prof {
            eprintln!(
                "qwen4 cuda: round of {k} drafts, {a} accepted: host rows {:.1} ms, whole {:.1} ms",
                t_host.as_secs_f64() * 1e3,
                t_round.elapsed().as_secs_f64() * 1e3
            );
        }
        if std::env::var_os("CMF_QWEN_ROUND_LOG").is_some() {
            eprintln!(
                "qwen4 cuda: round {} k={k} a={a} draft {:.1} ms verify {:.1} ms lean={} spec {:.1} ms (queued in {:.1}) settle {:.1} ms host {:.1} ms",
                if copied { "copy" } else { "mtp" },
                t_draft.as_secs_f64() * 1e3,
                t_round.elapsed().as_secs_f64() * 1e3,
                u8::from(dev.lean_was()),
                t_spec.as_secs_f64() * 1e3,
                dev.launch_ms,
                t_settle.as_secs_f64() * 1e3,
                t_host.as_secs_f64() * 1e3
            );
        }
        Ok((a, logits))
    })();
    if ncu_on {
        dev.profiler_range(false);
    }
    match res {
        Ok((a, logits)) => {
            state.token_history.extend_from_slice(&ids[..a + 1]);
            state.pos = next_pos + a + 1;
            state.cuda = Some(dev);
            Some(SpecRound { accepted: drafts[..a].to_vec(), logits, drafted: k })
        }
        Err(e) => {
            eprintln!("qwen4 cuda: {e}; the sequence continues on the host path");
            drop(dev);
            state.cuda_off = true;
            None
        }
    }
}

/// `CMF_QWEN_ADMIT_BATCH=0`: a multi-token frame admits its cold winners
/// token by token (the slots are the same either way; the batch only
/// runs the uploads of the whole frame in parallel).
#[cfg(feature = "gpu")]
fn admit_batch_env() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_QWEN_ADMIT_BATCH").as_deref() != Ok("0"))
}

/// Token-level profile of the device path (`CMF_QWEN_PROF`).
#[cfg(feature = "gpu")]
#[derive(Default)]
struct DevProf {
    encode: std::time::Duration,
    wait: std::time::Duration,
    cold_cpu: std::time::Duration,
    admit: std::time::Duration,
    head: std::time::Duration,
    cold_experts: usize,
    cold_dev: usize,
    layers_with_cold: usize,
    chains: usize,
    aborted: usize,
    /// finishing encoders into command buffers (encode-ahead included)
    finish: std::time::Duration,
    /// spin on the frame's fence only
    spin: std::time::Duration,
    /// from the fence to the next submit: cold parse, admissions, finalize
    post: std::time::Duration,
    /// gathering the frame's PLE n-gram rows from the mapped table
    ple: std::time::Duration,
    /// GPU stage times from the timestamp probe (`CMF_QWEN_TS=1`), ns,
    /// indexed by `TS_NAMES`
    gpu: [f64; 17],
    gpu_last_end: Option<f64>,
}

#[cfg(feature = "gpu")]
const TS_NAMES: [&str; 17] = [
    "pending", "ple", "attn_hc", "gdn_proj", "gdn_rec", "gdn_out", "qsa_proj", "qsa_attn",
    "qsa_out", "moe_hc", "router", "route", "gu", "dn", "tail", "head", "idle",
];

#[cfg(feature = "gpu")]
impl DevProf {
    /// Fold one frame's marks (`TS_MARKS` of them) into the stage sums.
    fn gpu_note(&mut self, fi: usize, n: usize, m: &[f64], gdn: Option<bool>) {
        if m.len() < 13 {
            return;
        }
        let d = |a: usize, b: usize| (m[b] - m[a]).max(0.0);
        if let Some(prev) = self.gpu_last_end {
            self.gpu[16] += (m[0] - prev).max(0.0);
        }
        self.gpu_last_end = Some(m[8]);
        self.gpu[0] += d(0, 1);
        if fi >= n {
            self.gpu[15] += d(7, 8);
            return;
        }
        self.gpu[1] += d(1, 2);
        self.gpu[2] += d(2, 3);
        let o = if gdn == Some(true) { 3 } else { 6 };
        self.gpu[o] += d(3, 9);
        self.gpu[o + 1] += d(9, 10);
        self.gpu[o + 2] += d(10, 4);
        self.gpu[9] += d(4, 5);
        self.gpu[10] += d(5, 11);
        self.gpu[11] += d(11, 6);
        self.gpu[12] += d(6, 12);
        self.gpu[13] += d(12, 7);
        self.gpu[14] += d(7, 8);
    }
}

/// The whole token on the card. Returns `false` when the device path is
/// not available (the caller then runs the host path); a refusal after
/// the first token of a sequence is reported once and turns the path off.
/// Tokens one device frame carries; the prompt is fed in chunks of this many.
#[cfg(feature = "gpu")]
pub fn prefill_chunk() -> usize {
    // CUDA runs a chunk as plain GEMMs over all of its tokens
    #[cfg(feature = "cuda")]
    if crate::gpu_cuda::selected() {
        return std::env::var("CMF_QWEN_CUDA_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(if crate::gpu_cuda::qwen4::lean_card() { 1024 } else { 4096 })
            .clamp(1, crate::gpu_cuda::qwen4::MAX_CHUNK);
    }
    // `forward_tokens` cuts a chunk into TROWS-token frames when every expert is
    // resident; otherwise it streams the chunk layer by layer (`forward_stream_device`,
    // several frames), or cuts it into eight-token frames when that is off
    let rows = crate::gpu_wgpu::qwen4::TROWS;
    let dflt = if stream_on() { rows * stream_frames() } else { rows };
    std::env::var("CMF_QWEN_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(dflt)
        .clamp(1, rows * 64)
}

/// The layer-major prompt for a card that cannot hold every expert
/// (`forward_stream_device`); `CMF_QWEN_STREAM=0` keeps the eight-token frames.
#[cfg(feature = "gpu")]
fn stream_on() -> bool {
    std::env::var("CMF_QWEN_STREAM").as_deref() != Ok("0")
}

/// Frames (TROWS tokens each) of a streamed prompt chunk: `CMF_QWEN_STREAM_TOKENS` tokens,
/// 4096, or 2048 under an 11 GiB budget (a 12 GB card; its bank takes ~52 KiB a token).
#[cfg(feature = "gpu")]
fn stream_frames() -> usize {
    let rows = crate::gpu_wgpu::qwen4::TROWS;
    std::env::var("CMF_QWEN_STREAM_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            if crate::gpu_wgpu::qwen4::vram_budget().is_some_and(|b| b < 11 << 30) { 2048 } else { 4096 }
        })
        .div_ceil(rows)
        .clamp(1, 64)
}

#[cfg(not(feature = "gpu"))]
pub fn prefill_chunk() -> usize {
    1
}

/// Several consecutive tokens (`ids` at `pos0..`), logits of the last one
/// when `want_logits`. The device path runs them in one frame per layer;
/// otherwise each token takes the host path.
#[allow(clippy::too_many_arguments)]
/// Debug (`CMF_QWEN_TOPLOG=1`): the five largest logits of the last row
/// after a chunked prompt, for comparing prompt-frame kernels.
fn top_log(logits: &[f32], vocab: usize) {
    if vocab == 0 || logits.len() < vocab {
        return;
    }
    let row = &logits[logits.len() - vocab..];
    let mut ix: Vec<usize> = (0..vocab).collect();
    ix.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
    let top: Vec<String> = ix[..5].iter().map(|&i| format!("{i}:{:.4}", row[i])).collect();
    eprintln!("[toplog] {}", top.join(" "));
}

/// `CMF_QWEN_CUDA_CHECK=1`: the host path runs the same tokens and every
/// stage of the chunk's last token is compared (slow: a host token each).
#[cfg(feature = "cuda")]
fn cuda_check_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_QWEN_CUDA_CHECK").as_deref() == Ok("1"))
}

/// `ids` at `pos0..` through the CUDA device, in chunks of its capacity;
/// `history` are the tokens before them (the n-gram tables hash it).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_run(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    dev: &mut crate::gpu_cuda::qwen4::Dev,
    history: &[u32],
    ids: &[u32],
    pos0: usize,
    pool: Option<&Pool>,
    mut logits: Option<&mut Vec<f32>>,
) -> Result<(), String> {
    let h = cfg.hidden;
    let chunk = dev.chunk().max(1);
    let no_ple = std::env::var_os("CMF_QWEN_NO_PLE").is_some();
    let prof = std::env::var_os("CMF_QWEN_PROF").is_some();
    // a prompt's n-gram rows leave the disk while the first chunk runs
    if !no_ple && ids.len() >= 64 {
        ple_prefetch(layers, cfg, history, ids);
    }
    // the history of a later chunk also holds the chunks before it
    let mut grown: Vec<u32> = Vec::new();
    let mut off = 0usize;
    while off < ids.len() {
        let end = (off + chunk).min(ids.len());
        let part = &ids[off..end];
        let t0 = std::time::Instant::now();
        let mut emb = vec![0.0f32; part.len() * h];
        for (row, &id) in emb.chunks_mut(h).zip(part) {
            if (id as usize) < globals.embed.rows() {
                globals.embed.row_f32(id as usize, row);
            }
        }
        // a decode step or a window asks for a layer's rows when the pass gets there
        let lazy = !no_ple && part.len() <= 8 && off == 0;
        let ple = if no_ple || lazy {
            Vec::new()
        } else if off == 0 {
            ple_frame_rows(layers, cfg, history, part, pool)
        } else {
            if grown.is_empty() {
                grown.extend_from_slice(history);
            }
            grown.extend_from_slice(&ids[grown.len() - history.len()..off]);
            ple_frame_rows(layers, cfg, &grown, part, pool)
        };
        if prof && part.len() > 1 {
            eprintln!("qwen4 cuda: host rows of {} tokens {:.1} ms", part.len(), t0.elapsed().as_secs_f64() * 1e3);
        }
        let lg = if end == ids.len() { logits.take() } else { None };
        if lazy {
            let rows_of = |li: usize| ple_rows_of(layers, cfg, history, part, pool, Some(li)).into_iter().nth(li).unwrap_or_default();
            dev.forward_lazy(layers, &emb, &rows_of, pos0 + off, lg)?;
        } else {
            dev.forward(layers, &emb, &ple, pos0 + off, lg)?;
        }
        off = end;
    }
    Ok(())
}

/// The prompt has run on the CUDA card: the arena is refilled with what this request needs
/// (see `prompt_done` of the wgpu path): the task's pack (`CMF_QWEN_PACKS`, named by
/// `CMF_QWEN_TASK_HINT`) blended with the prompt's own routing (`CMF_QWEN_PACK_MIX`, the
/// prompt's share, default 0.5; 1 without a pack); `CMF_QWEN_PACK_SWAP` bounds the experts
/// brought in (default 4096). Off unless `CMF_QWEN_SELF_PACK=1`: on the CUDA card the refill
/// from the prompt's own routing alone changed the decode speed by less than the run-to-run
/// spread, and the python pack on a writing task made it slower.
#[cfg(feature = "cuda")]
fn cuda_prompt_done(layers: &[Layer], state: &mut State) {
    if std::env::var("CMF_QWEN_SELF_PACK").as_deref() != Ok("1") {
        return;
    }
    let (n, ne) = (layers.len(), layers.first().map_or(0, |l| l.moe.experts.len()));
    if n == 0 || ne == 0 || state.cuda.is_none() {
        return;
    }
    if state.packs.is_none() {
        state.packs = Some(PackSet::load(n, ne));
    }
    let packs = state.packs.as_ref().and_then(|p| p.as_ref());
    // the pack: named from outside, or the router's choice on the prompt's routing
    let weighted: Vec<(usize, f32)> = match packs {
        Some(p) => match p.hint_weights() {
            Some(w) => w,
            None => state
                .cuda
                .as_ref()
                .and_then(|d| p.decide(d.route_hist(), ne))
                .map(|(i, _)| vec![(i, 1.0)])
                .unwrap_or_default(),
        },
        None => Vec::new(),
    };
    let pack: Option<Vec<f32>> = packs.filter(|_| !weighted.is_empty()).map(|p| {
        let mut pk = vec![0f32; n * ne];
        for &(i, w) in &weighted {
            for l in 0..n {
                let row = &p.profiles[i].counts[l * ne..(l + 1) * ne];
                let ps = row.iter().map(|&c| c as f32).sum::<f32>().max(1.0);
                for (o, &c) in pk[l * ne..(l + 1) * ne].iter_mut().zip(row) {
                    *o += w * c as f32 / ps;
                }
            }
        }
        pk
    });
    let mix: f32 = if pack.is_some() { std::env::var("CMF_QWEN_PACK_MIX").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5f32).clamp(0.0, 1.0) } else { 1.0 };
    let max_swap = std::env::var("CMF_QWEN_PACK_SWAP").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4096);
    let t0 = std::time::Instant::now();
    let name = weighted.first().and_then(|&(i, _)| packs.map(|p| p.names[i].clone())).unwrap_or_else(|| "prompt routing".into());
    let Some(dev) = state.cuda.as_mut() else { return };
    match dev.retarget(layers, pack.as_deref(), mix, max_swap) {
        Ok((out, inn)) => {
            if std::env::var_os("CMF_QWEN_PROF").is_some() || std::env::var_os("CMF_QWEN_PACK_LOG").is_some() {
                eprintln!("qwen4 cuda: pack {name} (prompt share {mix}): {out} experts out, {inn} in, {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        Err(e) => eprintln!("qwen4 cuda: pack: {e}"),
    }
}

/// The whole stack of `ids` at `position..` on the CUDA card
/// (`gpu_cuda::qwen4`). Returns `false` when that path is not available or
/// has refused: the caller then runs the tokens on the host path, which
/// replays the history into its own caches first.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_forward(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    ids: &[u32],
    position: usize,
    inv_freq: &[f32],
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
    want_logits: bool,
) -> bool {
    if state.cuda_off || ids.is_empty() || !crate::gpu_cuda::selected() {
        return false;
    }
    // the wgpu device path never runs beside this one
    state.device_off = true;
    if !crate::gpu::enabled_here() || std::env::var("CMF_QWEN_DEVICE").as_deref() == Ok("0") {
        return false;
    }
    if !crate::gpu_cuda::enabled() {
        state.cuda_off = true;
        return false;
    }
    if state.cuda.is_none() {
        ple_lock(layers);
        // the file's MTP draft head (`CMF_QWEN_MTP=0` leaves it out)
        if state.mtp.is_none() && !state.mtp_tried {
            state.mtp_tried = true;
            if let Some(model) = layers.first().and_then(|l| l.moe.experts.first()).and_then(|f| f.gate_proj.model_arc()) {
                state.mtp = MtpHead::load(&model, model.arch());
            }
        }
        let fcs = state.mtp.as_ref().and_then(|h| {
            Some((t(&h.side, "model.mtp.fc_embedding.weight").ok()?, t(&h.side, "model.mtp.fc_hidden.weight").ok()?))
        });
        let parts = state.mtp.as_ref().zip(fcs.as_ref()).map(|(h, (fe, fh))| crate::gpu_cuda::qwen4::MtpParts {
            layer: &h.layer,
            mixer: &h.mixer,
            enorm: &h.enorm,
            hnorm: &h.hnorm,
            fc_e: fe,
            fc_h: fh,
        });
        match crate::gpu_cuda::qwen4::Dev::new(globals, layers, cfg, inv_freq, prefill_chunk(), parts.as_ref()) {
            Ok(mut d) => {
                // the MTP head's experts stay in the arena
                if let Some(head) = state.mtp.as_ref() {
                    match d.warm_mtp(&head.layer) {
                        Ok(n) if n > 0 => {
                            if std::env::var_os("CMF_QWEN_PROF").is_some() {
                                eprintln!("qwen4 cuda: {n} experts of the MTP head resident");
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            eprintln!("qwen4 cuda: the MTP head's experts: {e}; it stays off");
                            d.mtp_off();
                        }
                    }
                }
                // `CMF_QWEN_PROFILE`: the experts earlier runs used most go in first
                let ne = layers.first().map_or(0, |l| l.moe.experts.len());
                if let Some(pr) = std::env::var("CMF_QWEN_PROFILE").ok().and_then(|path| ExpertProfile::load(&path, layers.len(), ne)) {
                    let t0 = std::time::Instant::now();
                    match d.prefill(layers, &pr.ranked()) {
                        Ok(n) => eprintln!(
                            "qwen4 cuda: arena prefilled with {n} of {} profiled experts ({} tokens of history) in {:.2}s",
                            pr.ranked().len(),
                            pr.tokens,
                            t0.elapsed().as_secs_f64()
                        ),
                        Err(e) => eprintln!("qwen4 cuda: prefill: {e}"),
                    }
                }
                state.cuda = Some(d)
            }
            Err(e) => {
                eprintln!("qwen4 cuda: {e}; staying on the host path");
                state.cuda_off = true;
                return false;
            }
        }
    }
    let Some(mut dev) = state.cuda.take() else {
        return false;
    };
    let check = cuda_check_on();
    dev.want_trace = check;
    let res = (|| -> Result<bool, String> {
        if position == 0 || state.pos != position {
            state.reset();
            dev.reset()?;
        }
        if dev.pos != position {
            // the host path ran these positions (or the sequence does not
            // start at its beginning, which only the host path can serve)
            if state.token_history.len() != position {
                return Ok(false);
            }
            dev.reset()?;
            cuda_run(globals, layers, cfg, &mut dev, &[], &state.token_history, 0, pool, None)?;
        }
        cuda_run(
            globals,
            layers,
            cfg,
            &mut dev,
            &state.token_history,
            ids,
            position,
            pool,
            if want_logits { Some(&mut *logits) } else { None },
        )?;
        Ok(true)
    })();
    match res {
        Ok(true) => {
            if check {
                cuda_check(globals, layers, cfg, state, &dev, ids, position, inv_freq, pool, want_logits);
            }
            state.token_history.extend_from_slice(ids);
            state.pos = position + ids.len();
            if !want_logits {
                logits.clear();
            }
            state.cuda = Some(dev);
            if want_logits && position == 0 && ids.len() >= 32 {
                cuda_prompt_done(layers, state);
            }
            true
        }
        Ok(false) => {
            state.cuda = Some(dev);
            false
        }
        Err(e) => {
            eprintln!("qwen4 cuda: {e}; the sequence continues on the host path");
            drop(dev);
            state.cuda_off = true;
            false
        }
    }
}

/// Check mode: run `ids` through the host path on its own state and compare
/// every stage of the last token with the device's trace.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_check(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    dev: &crate::gpu_cuda::qwen4::Dev,
    ids: &[u32],
    position: usize,
    inv_freq: &[f32],
    pool: Option<&Pool>,
    want_logits: bool,
) {
    if position == 0 || state.cuda_ref.is_none() {
        let mut r = Box::new(State::new(layers.len()));
        r.cuda_off = true;
        r.device_off = true;
        state.cuda_ref = Some(r);
    }
    let Some(host) = state.cuda_ref.as_mut() else {
        return;
    };
    if host.pos != position {
        eprintln!("qwen4 cuda check: the host reference is at {} and the chunk at {position}; not compared", host.pos);
        return;
    }
    let mut host_logits = Vec::new();
    let mut tap = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        let last = i + 1 == ids.len();
        if last {
            CUDA_TAP.with(|t| *t.borrow_mut() = Some(Vec::new()));
        }
        // exact float activations: the default int8-activation kernels differ
        // from any f32 path by more than the device does
        crate::qtensor::float_activations_scope(|| {
            crate::gpu::cpu_scope(|| {
                forward_token(
                    globals,
                    layers,
                    cfg,
                    host,
                    id,
                    position + i,
                    inv_freq,
                    pool,
                    &mut host_logits,
                    last && want_logits,
                );
            })
        });
        if last {
            tap = CUDA_TAP.with(|t| t.borrow_mut().take()).unwrap_or_default();
        }
    }
    // relative error of a stage: the largest difference over the host's largest value
    let rel = |a: &[f32], b: &[f32]| -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let diff = a.iter().zip(b).fold(0f32, |m, (x, y)| if (x - y).is_nan() { f32::INFINITY } else { m.max((x - y).abs()) });
        diff / scale
    };
    let at = position + ids.len() - 1;
    let mut worst = (0f32, "", 0usize);
    let mut lines: Vec<String> = Vec::new();
    let mut cur = usize::MAX;
    for (label, li, dv) in &dev.trace {
        let Some((_, _, hv)) = tap.iter().find(|(l, i, _)| l == label && i == li) else {
            continue;
        };
        let r = rel(hv, dv);
        if r > worst.0 || worst.1.is_empty() {
            worst = (r, label, *li);
        }
        if *label == "logits" {
            let arg = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m }).0;
            let (ha, da) = (arg(hv), arg(dv));
            lines.push(format!(
                "qwen4 cuda check pos {at} logits: rel {r:.2e}, argmax host {ha} cuda {da} ({})",
                if ha == da { "same" } else { "DIFFERENT" }
            ));
            continue;
        }
        if *li != cur {
            cur = *li;
            lines.push(format!("qwen4 cuda check pos {at} L{li}:"));
        }
        if let Some(line) = lines.last_mut() {
            line.push_str(&format!(" {label} {r:.1e}"));
        }
    }
    if ids.len() > 1 || std::env::var_os("CMF_QWEN_CUDA_CHECK_ALL").is_some() {
        for l in &lines {
            eprintln!("{l}");
        }
    } else if let Some(l) = lines.iter().rev().find(|l| l.contains(" logits: ")) {
        eprintln!("{l}");
    }
    eprintln!(
        "qwen4 cuda check pos {at}: {} stages compared, worst {} at layer {} rel {:.2e}",
        dev.trace.len(),
        worst.1,
        worst.2,
        worst.0
    );
}

#[allow(clippy::too_many_arguments)]
pub fn forward_tokens(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    ids: &[u32],
    pos0: usize,
    inv_freq: &[f32],
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
    want_logits: bool,
) {
    #[cfg(feature = "cuda")]
    if cuda_forward(globals, layers, cfg, state, ids, pos0, inv_freq, pool, logits, want_logits) {
        return;
    }
    #[cfg(feature = "gpu")]
    if ids.len() > 1 && state.masked {
        state.masked = false;
        if let Some(d) = state.dev.as_mut() {
            d.set_masked(false);
        }
    }
    // a new prompt after a decode: the decode's routing is one row (`CMF_QWEN_ROUTE_ROWS`)
    #[cfg(feature = "gpu")]
    if ids.len() > 1 && state.row_gen {
        route_row("gen", &state.route_hist, state.route_tokens, &mut state.row_snap, &mut state.row_snap_tokens);
        state.row_gen = false;
        trace_mark(1);
        trace_flush();
    } else if ids.len() > 1 && pos0 == 0 {
        trace_mark(1);
    }
    forward_tokens_stream(globals, layers, cfg, state, ids, pos0, inv_freq, pool, logits, want_logits);
    // the prompt's last chunk: the arena gets the narrow bundle this request needs
    #[cfg(feature = "gpu")]
    if want_logits && ids.len() > 1 {
        prompt_done(layers, state);
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_tokens_stream(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    ids: &[u32],
    pos0: usize,
    inv_freq: &[f32],
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
    want_logits: bool,
) {
    // A prompt chunk on a card that cannot hold every expert streams layer by layer:
    // the sequence's first frame builds (or resets) the device state the usual way, the
    // body streams, a last frame of at most TMAX tokens brings the logits.
    #[cfg(feature = "gpu")]
    {
        use crate::gpu_wgpu::qwen4 as q4;
        if stream_on() && !state.resident_all && !state.device_off && ids.len() > 3 * q4::TMAX {
            // the chunk's n-gram rows leave the disk while the first frame and layer 0 run
            let seen = pos0.min(state.token_history.len());
            ple_prefetch(layers, cfg, &state.token_history[..seen], ids);
            let head = if state.dev.is_none() || pos0 == 0 || state.pos != pos0 { q4::TMAX } else { 0 };
            if head > 0 {
                let mut lg = Vec::new();
                forward_tokens_split(globals, layers, cfg, state, &ids[..head], pos0, inv_freq, pool, &mut lg, false);
            }
            let tail = if want_logits { q4::TMAX } else { 0 };
            let (a, b) = (head, ids.len() - tail);
            if !state.resident_all
                && !state.device_off
                && b > a + q4::TMAX
                && forward_stream_device(globals, layers, cfg, state, &ids[a..b], pos0 + a, inv_freq, pool)
            {
                if tail > 0 {
                    forward_tokens_split(globals, layers, cfg, state, &ids[b..], pos0 + b, inv_freq, pool, logits, want_logits);
                } else {
                    logits.clear();
                }
                return;
            }
            forward_tokens_split(globals, layers, cfg, state, &ids[head..], pos0 + head, inv_freq, pool, logits, want_logits);
            return;
        }
    }
    forward_tokens_split(globals, layers, cfg, state, ids, pos0, inv_freq, pool, logits, want_logits)
}

#[allow(clippy::too_many_arguments)]
fn forward_tokens_split(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    ids: &[u32],
    pos0: usize,
    inv_freq: &[f32],
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
    want_logits: bool,
) {
    // Frames take TMAX tokens, or TROWS on a resident cut model (no cold pass);
    // a chunk asking for logits ends on a frame of at most TMAX tokens, so the
    // head runs over a small frame.
    #[cfg(feature = "gpu")]
    {
        use crate::gpu_wgpu::qwen4 as q4;
        let big = state.resident_all && !state.device_off;
        let cap = if big { q4::TROWS } else { q4::TMAX };
        if ids.len() > q4::TMAX && (ids.len() > cap || want_logits) {
            let mut at = 0usize;
            while at < ids.len() {
                let rem = ids.len() - at;
                let big = state.resident_all && !state.device_off;
                let n = if !big {
                    rem.min(q4::TMAX)
                } else if want_logits && rem > q4::TMAX {
                    (rem - q4::TMAX).min(q4::TROWS)
                } else {
                    rem.min(if want_logits { q4::TMAX } else { q4::TROWS })
                };
                let last = at + n == ids.len();
                let mut lg = Vec::new();
                forward_tokens_split(
                    globals,
                    layers,
                    cfg,
                    state,
                    &ids[at..at + n],
                    pos0 + at,
                    inv_freq,
                    pool,
                    &mut lg,
                    want_logits && last,
                );
                if last {
                    *logits = lg;
                }
                at += n;
            }
            if want_logits && std::env::var("CMF_QWEN_TOPLOG").as_deref() == Ok("1") {
                top_log(logits, globals.lm_head.rows());
            }
            return;
        }
    }
    #[cfg(feature = "gpu")]
    if !ids.is_empty()
        && (ids.len() <= crate::gpu_wgpu::qwen4::TMAX
            || (state.resident_all && !want_logits && ids.len() <= crate::gpu_wgpu::qwen4::TROWS))
        && forward_tokens_device(
            globals,
            layers,
            cfg,
            state,
            ids,
            pos0,
            inv_freq,
            pool,
            logits,
            want_logits,
        )
    {
        return;
    }
    for (i, &id) in ids.iter().enumerate() {
        let last = i + 1 == ids.len();
        let mut lg = Vec::new();
        forward_token(
            globals,
            layers,
            cfg,
            state,
            id,
            pos0 + i,
            inv_freq,
            pool,
            &mut lg,
            want_logits && last,
        );
        if last {
            *logits = lg;
        }
    }
}

/// GPTQ calibration capture: `ids` from position 0 through the device
/// frames, with the MoE input rows and router logits of layers `lo..hi`
/// copied out after routing. Returns per layer `[ids.len()][hidden + n_experts]`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn capture_window(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    ids: &[u32],
    lo: usize,
    hi: usize,
    inv_freq: &[f32],
    pool: Option<&Pool>,
) -> Result<Vec<Vec<f32>>, String> {
    use crate::gpu_wgpu::qwen4 as q4;
    let first = layers.first().ok_or("no layers")?;
    let hidden = first.moe.experts.first().ok_or("no experts")?.gate_proj.cols();
    let ne = first.moe.experts.len();
    let hi = hi.min(layers.len());
    if lo >= hi || !q4::capture_begin(lo, hi, hidden, ne) {
        return Err("capture unavailable".into());
    }
    let mut out = vec![Vec::with_capacity(ids.len() * (hidden + ne)); hi - lo];
    let chunk = prefill_chunk();
    let mut pos = 0;
    let res = (|| {
        while pos < ids.len() {
            let end = (pos + chunk).min(ids.len());
            let mut lg = Vec::new();
            forward_tokens(globals, layers, cfg, state, &ids[pos..end], pos, inv_freq, pool, &mut lg, false);
            let (rows, written) = q4::capture_read(end - pos).ok_or("capture readback")?;
            if written < hi - lo {
                return Err(format!("the device path ran {written} of {} captured layer frames (host fallback?)", hi - lo));
            }
            for (k, r) in rows.into_iter().enumerate() {
                out[k].extend(r);
            }
            pos = end;
        }
        Ok(())
    })();
    q4::capture_end();
    res.map(|_| out)
}

/// Directory triples (gate, up, down tensor indices) of layer `li`'s routed
/// experts, and of its shared expert when it has one.
pub fn expert_dirs(layers: &[Layer], li: usize) -> Option<(Vec<(usize, usize, usize)>, Option<(usize, usize, usize)>)> {
    let l = layers.get(li)?;
    Some((l.expert_ids.clone(), l.shared_ids))
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn forward_token_device(
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
) -> bool {
    forward_tokens_device(
        globals,
        layers,
        cfg,
        state,
        &[token_id],
        position,
        inv_freq,
        pool,
        logits,
        want_logits,
    )
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn forward_tokens_device(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    ids: &[u32],
    pos0: usize,
    inv_freq: &[f32],
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
    want_logits: bool,
) -> bool {
    use crate::gpu_wgpu::qwen4 as q4;
    let ntok = ids.len();
    // a copy frame: a decode forward past TMAX tokens, with or without logits (no far
    // field: its spec stash holds a window)
    let frame_ok = state.copy_frame
        && (state.resident_all || state.masked)
        && state.dev.as_ref().is_some_and(|d| d.o1.is_none());
    let cap = if (state.resident_all && !want_logits) || frame_ok { q4::TROWS } else { q4::TMAX };
    if ntok == 0 || ntok > cap {
        return false;
    }
    let position = pos0;
    let token_id = ids[0];
    if state.device_off
        || std::env::var("CMF_QWEN_DEVICE").as_deref() == Ok("0")
        || !crate::gpu::enabled_here()
    {
        return false;
    }
    fn off(state: &mut State, why: &str) -> bool {
        if !state.device_off {
            tracing::warn!("qwen4 device path off: {why}");
            if std::env::var_os("CMF_QWEN_PROF").is_some() {
                eprintln!("qwen4-device: off ({why})");
            }
        }
        state.device_off = true;
        false
    }
    if !q4::available() {
        return off(state, "kernels or the global expert arena unavailable");
    }
    let Some(first) = layers.first() else {
        return off(state, "no layers");
    };
    let Some(e0) = first.moe.experts.first() else {
        return off(state, "no experts");
    };
    let Some(model) = e0.gate_proj.model_arc() else {
        return off(state, "experts are not mmap-backed");
    };
    let gu_q2 = e0.gate_proj.model_dtype() == Some(TensorDtype::Q2TiledP);
    if !gu_q2 && e0.gate_proj.model_dtype() != Some(TensorDtype::Q4TiledP) {
        return off(state, "expert gate/up are neither q2tp nor q4tp");
    }
    if first.moe.router_sigmoid
        || !first.moe.norm_topk_prob
        || first.moe.route_tau.is_some()
        || first.moe.mask.is_some()
        || (first.moe.routed_scaling - 1.0).abs() > 1e-9
    {
        return off(state, "non-Qwen routing settings");
    }
    let g = device_geom(cfg, &first.moe, gu_q2);
    let prof = std::env::var_os("CMF_QWEN_PROF").is_some();
    let check = std::env::var("CMF_QWEN_DEVICE_CHECK").as_deref() == Ok("1");
    let t_token = std::time::Instant::now();

    // ── first use: device state, resident skeleton, the expert arena ──
    if state.dev.is_none() {
        if layers.iter().any(|l| l.idx.is_none()) || globals.lm_head_idx.is_none() {
            return off(state, "skeleton tensors missing from the directory");
        }
        let kinds: Vec<(bool, bool)> = layers
            .iter()
            .map(|l| (matches!(l.mixer, Mixer::Gdn(_)), l.ple.is_some()))
            .collect();
        let Some(mut dev) = q4::Dev::new(model.uid(), &g, &kinds) else {
            return off(state, "device state allocation refused");
        };
        // The attention layers' far field: the file's turbo header or `CMF_QWEN_O1`.
        if let Some(o1) = crate::qwen4_o1::Qwen4O1::from_model(&model) {
            match o1.check(g.head_dim) {
                Ok(()) => {
                    let (m, w, sink, seal) = (o1.o1.m, o1.o1.w, o1.o1.sink, o1.o1.seal);
                    let exact = o1.exact.as_ref().map_or(0, |x| x.budget);
                    if dev.set_o1(&g, o1) {
                        tracing::info!("qwen4 O(1): {m} landmarks, window {w}, {sink} sinks, {exact} exact rows, seal at {seal}");
                    }
                }
                Err(e) => tracing::warn!("{e}; the attention layers stay exact"),
            }
        }
        // The experts' hot neurons (`hot-bake --experts`): one q8_2f tube per routed
        // expert, resident on the card beside the skeleton.
        let tubes: Vec<Vec<Option<(usize, usize, usize)>>> = layers
            .iter()
            .map(|l| {
                l.moe
                    .experts
                    .iter()
                    .map(|e| match e.segs.as_slice() {
                        [s] => Some((s.gate.model_idx()?, s.up.model_idx()?, s.down.model_idx()?)),
                        _ => None,
                    })
                    .collect()
            })
            .collect();
        if layers.iter().any(|l| l.moe.experts.iter().any(|e| e.segs.len() > 1)) {
            return off(state, "an expert with more than one tube");
        }
        // The shared expert's tube (`hot-bake --shared-q8`: the whole expert at eight bits).
        let shared: Vec<Option<(usize, usize, usize)>> = layers
            .iter()
            .map(|l| match l.moe.shared.as_ref().map(|(se, _)| se.segs.as_slice()) {
                Some([s]) => Some((s.gate.model_idx()?, s.up.model_idx()?, s.down.model_idx()?)),
                _ => None,
            })
            .collect();
        if layers.iter().any(|l| l.moe.shared.as_ref().is_some_and(|(se, _)| se.segs.len() > 1)) {
            return off(state, "a shared expert with more than one tube");
        }
        let t0 = std::time::Instant::now();
        let Some(bytes) = q4::prewarm(&model, &globals.skeleton_idxs) else {
            return off(state, "skeleton does not fit the VRAM budget");
        };
        tracing::info!(
            "qwen4 device: {} skeleton tensors resident ({} MB) in {:.2}s",
            globals.skeleton_idxs.len(),
            bytes >> 20,
            t0.elapsed().as_secs_f64()
        );
        // The tubes after the skeleton, so the budget check sees what is up.
        if tubes.iter().flatten().any(Option::is_some) || shared.iter().any(Option::is_some) {
            if !dev.set_tubes(&model, &g, &tubes, &shared) {
                return off(state, "expert tubes do not load or do not fit the VRAM budget");
            }
            tracing::info!("qwen4 tubes: {} MB of expert tubes resident", dev.tube_bytes() >> 20);
        }
        if prof {
            eprintln!(
                "qwen4-device: skeleton {} MB resident in {:.2}s",
                bytes >> 20,
                t0.elapsed().as_secs_f64()
            );
        }
        state.dev = Some(dev);
        state.picks_prev = vec![Vec::new(); layers.len()];
    }
    if state.gpu_pool.is_none() {
        // The draft head's skeleton and its 512-expert bank come first, so
        // the main arena is sized from what is left.
        if state.mtp.is_none() && !state.mtp_tried {
            state.mtp_tried = true;
            state.mtp = MtpHead::load(&model, model.arch());
            if let Some(head) = state.mtp.as_mut()
                && let Err(why) = head.setup(&device_geom(cfg, &head.layer.moe, gu_q2), gu_q2)
            {
                tracing::warn!("qwen4 MTP off: {why}");
                state.mtp = None;
            }
        }
        let Some(budget) = q4::vram_budget() else {
            return off(state, "no VRAM budget");
        };
        let resident = q4::resident_bytes();
        let Some(per) = QwenGpuPool::per_expert_bytes(g.inter, g.hidden, gu_q2) else {
            return off(state, "expert geometry");
        };
        let reserve_mb = std::env::var("CMF_QWEN_KV_RESERVE_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            // 768 MiB beside the allocator's own driver reserve covers the
            // frame scratch and f32 K/V to ~16k context; raise it for longer
            // sequences (`CMF_QWEN_KV_RESERVE_MB`), lower it for more experts.
            .unwrap_or(768);
        // The generic bank allocator would carve another (budget/10)
        // clamped to 2-4 GiB below the request, a workspace for the DSV4/GLM
        // paths that this path never uses: its KV and frames live in the
        // reserve above. Below a 24 GiB budget the carve-out stays (slot
        // counts as before); on a 32 GB card it would idle ~2.8 GiB, about
        // 1,700 expert slots. `CMF_QWEN_WORKSPACE_MB` overrides.
        let workspace = std::env::var("CMF_QWEN_WORKSPACE_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|m| m << 20)
            .unwrap_or(if budget >= 24 << 30 {
                0
            } else {
                (budget / 10).clamp(2 << 30, 4 << 30)
            });
        // The verify window's recurrent-state snapshots (`ensure_snaps`,
        // one row per window position on every GDN and PLE layer) are
        // allocated later, out of the reserve. The 768 MiB default held the
        // four rows of the default k = 3 on the test cards; a longer window
        // (k up to 7, eight rows) reserves its extra rows on top.
        let window = state.mtp.as_ref().map_or(0, |h| h.k + 1);
        let snap_extra = {
            let kinds: Vec<(bool, bool)> = layers
                .iter()
                .map(|l| (matches!(l.mixer, Mixer::Gdn(_)), l.ple.is_some()))
                .collect();
            q4::Dev::snap_bytes(&g, &kinds, window.saturating_sub(MTP_MEASURED_WINDOW))
        };
        let free = budget
            .saturating_sub(resident)
            .saturating_sub(reserve_mb << 20)
            .saturating_sub(workspace)
            .saturating_sub(snap_extra);
        let mut slots = (free / per as u64) as usize;
        let n_layers = layers.len();
        // A cut model the card holds whole takes exactly its experts (every
        // routed one, the shared ones, a little slack), not the whole budget:
        // the rest of the card stays free (a 40-48 GB card runs it whole).
        let whole = n_layers * g.n_experts + n_layers + q4::TMAX * g.top_k + 64;
        if std::env::var("CMF_QWEN_RESIDENT").as_deref() != Ok("0") && slots >= whole {
            slots = whole;
        }
        // staging for the cold passes: one slot per token slot and rank
        let staging = q4::TMAX * g.top_k;
        let Some(arena) = QwenGpuPool::create_explicit(
            &model,
            g.inter,
            g.hidden,
            n_layers,
            g.n_experts,
            gu_q2,
            slots,
            staging,
        ) else {
            return off(state, "expert arena allocation refused");
        };
        if prof {
            eprintln!(
                "qwen4-device: budget {} MB, skeleton resident {} MB, reserve {} MB, workspace {} MB, window snapshots beyond {MTP_MEASURED_WINDOW} rows {} MB, arena request {} slots ({} MB), got {} slots",
                budget >> 20,
                resident >> 20,
                reserve_mb,
                workspace >> 20,
                snap_extra >> 20,
                slots,
                (slots as u64 * per as u64) >> 20,
                arena.owner.len()
            );
        }
        let (cap_slots, cap_staging) = (arena.staging_base, arena.staging);
        state.gpu_pool = Some(arena);
        let pinned = {
            let arena = state.gpu_pool.as_mut().unwrap();
            layers.iter().enumerate().all(|(li, l)| {
                arena
                    .ensure(&model, li, &[], &l.expert_ids, l.shared_ids)
                    .is_some_and(|(_, s)| s != u32::MAX)
            })
        };
        if !pinned {
            return off(state, "shared expert could not be pinned");
        }
        let n_experts = g.n_experts;
        let triples: Vec<Vec<(usize, usize, usize)>> =
            layers.iter().map(|l| l.expert_ids.clone()).collect();
        // The host tiers behind the arena: a RAM tier sized from free memory
        // and direct reads from the file (see `expert_store`).
        let store = (std::env::var("CMF_QWEN_STORE").as_deref() != Ok("0"))
            .then(|| {
                crate::expert_store::ExpertStore::new(&model, &triples, n_experts, cap_slots)
            })
            .flatten();
        // n-gram rows are random reads: no read-around on a fault
        let advised =
            crate::expert_store::advise_random(&model, |n| n.contains("ngram_embedding.shard_"));
        if let Some(store) = store.as_ref() {
            state.gpu_pool.as_mut().unwrap().store = Some(store.clone());
            if prof {
                eprintln!(
                    "qwen4-device: expert store io {:?}, RAM tier {} experts, n-gram table {} MiB advised random",
                    store.io_mode(),
                    store.tier_capacity(),
                    advised >> 20
                );
            }
        }
        let mut profile = std::env::var("CMF_QWEN_PROFILE")
            .ok()
            .and_then(|path| ExpertProfile::load(&path, layers.len(), n_experts));
        let mut warm_order: Vec<(usize, usize)> = Vec::new();
        if let Some(pr) = profile.as_ref() {
            let ranked = pr.ranked();
            warm_order = ranked.clone();
            let t0 = std::time::Instant::now();
            let n = state
                .gpu_pool
                .as_mut()
                .map_or(0, |a| a.prefill(&model, &ranked, &triples));
            tracing::info!(
                "qwen4 device: arena prefilled with {n} of {} profiled experts ({} tokens of history) in {:.2}s",
                ranked.len(),
                pr.tokens,
                t0.elapsed().as_secs_f64()
            );
            if prof {
                eprintln!(
                    "qwen4-device: arena prefilled with {n} of {} profiled experts in {:.2}s (capacity {} slots, {} staging)",
                    ranked.len(),
                    t0.elapsed().as_secs_f64(),
                    cap_slots,
                    cap_staging
                );
            }
        }
        // A cut model that fits the card whole: every routed expert gets its
        // slot now (profiled ones first, then the rest), and the frames chain
        // the whole token. `CMF_QWEN_RESIDENT=0` keeps the per-layer frames.
        if std::env::var("CMF_QWEN_RESIDENT").as_deref() != Ok("0")
            && cap_slots >= layers.len() * n_experts
        {
            let mut seen = vec![false; layers.len() * n_experts];
            let mut all: Vec<(usize, usize)> = Vec::with_capacity(seen.len());
            for &(l, e) in &warm_order {
                if l < layers.len() && e < n_experts && !std::mem::replace(&mut seen[l * n_experts + e], true) {
                    all.push((l, e));
                }
            }
            for l in 0..layers.len() {
                for e in 0..n_experts {
                    if !seen[l * n_experts + e] {
                        all.push((l, e));
                    }
                }
            }
            let t0 = std::time::Instant::now();
            let arena = state.gpu_pool.as_mut().unwrap();
            arena.prefill(&model, &all, &triples);
            let held = arena.slot_for.iter().take(layers.len() * n_experts).filter(|&&s| s != u32::MAX).count();
            state.resident_all = held == layers.len() * n_experts;
            tracing::info!(
                "qwen4 device: {held} of {} routed experts resident in {:.2}s{}",
                layers.len() * n_experts,
                t0.elapsed().as_secs_f64(),
                if state.resident_all { " — whole-token chained frames" } else { "" }
            );
        }
        // The host-heap tier behind the arena (Linux): admissions become DMA
        // copies (`HostTier`). On by default when the arena cannot hold every
        // expert (`CMF_QWEN_HTIER_MB=auto` sizes it for every expert the arena
        // cannot hold, =<MiB> to that size, =0 off): the streamed prompt admits
        // whole layer banks. Measured on the RTX 3090 (no resizable BAR, §8.7
        // of the device doc) with eight-token frames it added only 2-5 % of
        // steady decode and ingest, while pinning 28-37 GiB held up the frames
        // beside it (first token of a one-shot run +7-11 s) and squeezed the
        // page cache. Whatever
        // is asked, it never pins more than half of the memory this process
        // can still take (cgroup and MemAvailable) nor comes within 16 GiB
        // of it.
        {
            let n_keys = layers.len() * n_experts;
            let want_slots = n_keys.saturating_sub(cap_slots) + cap_slots / 8;
            let stride = (per as u64).div_ceil(256) * 256 + 1024;
            let want = want_slots as u64 * stride;
            // Unset: on when the arena cannot hold every expert — a streamed prompt
            // admits whole layer banks, which the queue's writes take at under 1 GB/s
            // and DMA from the tier at PCIe speed (16 GB card: prompt 110 → 356 tok/s).
            // `CMF_QWEN_HTIER_MB=0` keeps it off.
            let asked = match std::env::var("CMF_QWEN_HTIER_MB").ok().as_deref() {
                None | Some("") if cap_slots < n_keys => want,
                None | Some("") => 0,
                Some("auto") => want,
                Some(v) => v.parse::<u64>().map_or(0, |mb| mb << 20),
            };
            let cap_bytes = if asked == 0 {
                0
            } else {
                crate::expert_store::host_available_bytes().map_or(0, |avail| {
                    asked
                        .min(avail / 2)
                        .min(avail.saturating_sub(16 << 30))
                })
            };
            if cap_bytes > 0 {
                let t0 = std::time::Instant::now();
                let tier = q4::HostTier::new(&model, n_keys, cap_bytes);
                if prof {
                    match tier.as_ref() {
                        Some(t) => eprintln!(
                            "qwen4-device: host tier {} slots ({} MiB, first GiB in {:.0} ms, the rest in the background after the first frame)",
                            t.capacity(),
                            (t.capacity() as u64 * t.stride()) >> 20,
                            t0.elapsed().as_secs_f64() * 1e3
                        ),
                        None => eprintln!("qwen4-device: host tier unavailable"),
                    }
                }
                // CMF_QWEN_HTIER_FILL=<threads> (default 8, 0 = lazy only):
                // copy every expert the arena does not hold into the tier
                // in the background, the profiled ones first, so prompt
                // ingest admits by DMA alone.
                let fill_threads = std::env::var("CMF_QWEN_HTIER_FILL")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(8);
                if let Some(t) = tier.as_ref()
                    && fill_threads > 0
                {
                    let arena = state.gpu_pool.as_ref().unwrap();
                    let mut order: Vec<usize> = Vec::with_capacity(n_keys);
                    let mut seen = vec![false; n_keys];
                    let ranked = profile.as_ref().map(|p| p.ranked()).unwrap_or_default();
                    for &(l, e) in &ranked {
                        if l < layers.len() && e < n_experts {
                            let k = l * n_experts + e;
                            if !seen[k] && arena.slot_for[k] == u32::MAX {
                                seen[k] = true;
                                order.push(k);
                            }
                        }
                    }
                    for e in 0..n_experts {
                        for l in 0..layers.len() {
                            let k = l * n_experts + e;
                            if !seen[k] && arena.slot_for[k] == u32::MAX {
                                seen[k] = true;
                                order.push(k);
                            }
                        }
                    }
                    let flat: Vec<(usize, usize, usize)> =
                        triples.iter().flat_map(|r| r.iter().copied()).collect();
                    if flat.len() == n_keys {
                        t.start_fill(model.clone(), Arc::new(flat), order, fill_threads);
                    }
                }
                state.gpu_pool.as_mut().unwrap().tier = tier;
            }
        }
        if std::env::var_os("CMF_QWEN_PROFILE_SAVE").is_some() && profile.is_none() {
            profile = Some(ExpertProfile::new(layers.len(), n_experts));
        }
        // A loaded profile has done its work once it warmed the arena: it would only
        // count this run's routing for nobody (nothing saves it), and a live profile
        // keeps every layer frame copying its cold lists and MoE inputs back to the host
        // and every prompt chunk waiting for that readback.
        state.profile = if std::env::var_os("CMF_QWEN_PROFILE_SAVE").is_some() { profile } else { None };
        // Warm the RAM tier in the background: the profiled experts the
        // arena could not take, then every other expert (layer by layer
        // interleaved) while the tier has room for what VRAM lacks.
        if let Some(store) = store.as_ref() {
            let mut seen = vec![false; layers.len() * n_experts];
            for &(l, e) in &warm_order {
                if l < layers.len() && e < n_experts {
                    seen[l * n_experts + e] = true;
                }
            }
            for e in 0..n_experts {
                for l in 0..layers.len() {
                    if !seen[l * n_experts + e] {
                        warm_order.push((l, e));
                    }
                }
            }
            store.start_background(warm_order);
        }
    }
    if position == 0 || state.pos != position {
        state.reset();
        let ok = state.dev.as_mut().is_some_and(|d| d.reset());
        if !ok {
            return off(state, "device reset failed");
        }
    }
    // a masked decode (`CMF_QWEN_MASK`) has no cold winner either
    let frame_mode = state.copy_frame && ntok > q4::TMAX;
    let resident_all = state.resident_all || (state.masked && (ntok <= q4::TMAX || frame_mode));
    // the frame verifies drafts: every row's argmax comes back with the last row
    let frame_verify = frame_mode && want_logits && state.frame_check;
    let frame_dbg = frame_verify && copy_frame_debug();
    // A prompt chunk nobody reads anything back from (resident mode: no cold
    // winner to complete; no logits, no window, no profile) is left running
    // while the next one is encoded. Anything else first drains it.
    let defer_readback = resident_all
        && !want_logits
        && ntok > 1
        && !state.verify_window
        && state.profile.is_none()
        && std::env::var("CMF_QWEN_PIPELINE").as_deref() != Ok("0")
        && std::env::var_os("CMF_QWEN_DUMP_HYPER").is_none();
    if !defer_readback && let Some(p) = state.inflight.take() {
        if p.wait().is_none() {
            return off(state, "readback");
        }
    }
    let dev = state.dev.as_mut().unwrap();
    let arena = state.gpu_pool.as_mut().unwrap();

    // ── the tokens ──
    let mut pf = DevProf::default();
    let cold_on_device = std::env::var("CMF_QWEN_COLD_DEVICE").as_deref() != Ok("0");
    // Several tokens per frame rule the gated chain out (one frame per layer).
    // Sealed O(1) layers dispatch directly: a chain's aborted frame would still
    // run their state pushes, and a row pushed twice is a row evicted twice.
    // With every expert resident no winner is cold, so nothing gates: the
    // whole token (or prompt chunk) and the head go out as one submission.
    let chain_env = std::env::var("CMF_QWEN_CHAIN").ok().and_then(|v| v.parse::<usize>().ok());
    // (A sealed far field dispatches its layers directly, which only an
    // aborted chain frame could double; with every expert resident no frame
    // aborts, so the far field chains too.)
    let chain_len = if dev.o1.is_some() && !resident_all {
        1
    } else if resident_all {
        // A token (or a verify window) goes out in a few chains, each encoded
        // while the card runs the one before (`CMF_QWEN_CHAINS`, default 4):
        // only the first chain's host encode is exposed (PRO 6000, 4K, plain
        // decode: 1 chain 72.6 tok/s, 2–3 78.3, 4 79.7). A prompt chunk stays
        // one chain (its chunks already overlap the host, `defer_readback`).
        let chains = std::env::var("CMF_QWEN_CHAINS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        let split = want_logits && (ntok <= crate::gpu_wgpu::qwen4::TMAX || frame_mode);
        chain_env
            .unwrap_or(if split { (layers.len() + 1).div_ceil(chains) } else { layers.len() + 1 })
            .max(1)
    } else if ntok > 1 {
        1
    } else {
        std::env::var("CMF_QWEN_CHAIN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1)
    };
    let n = layers.len();
    let top_k = g.top_k;
    // a layer frame's readback: the cold lists of every token, then the
    // MoE inputs of every token
    let cs = q4::cold_stride(&g);
    let hidden_bytes = cfg.hidden * 4;
    let cold_part = (ntok * cs).div_ceil(16) * 16;
    // With every expert resident nothing in a layer frame's readback is acted
    // on (no cold winner, no profile, no probe): the frames copy nothing out
    // and only the head's logits come back.
    let lean = resident_all
        && state.profile.is_none()
        && !check
        && std::env::var_os("CMF_QWEN_DUMP_HYPER").is_none()
        && std::env::var("CMF_QWEN_LEAN").as_deref() != Ok("0");
    let frame_bytes = if lean { 0 } else { cold_part + (ntok * hidden_bytes).div_ceil(16) * 16 };
    let snapshot = state.verify_window;
    if snapshot {
        dev.ensure_snaps(&g, ntok);
    }
    let mut cold_vec: Vec<Option<Vec<f32>>> = vec![None; ntok];
    let mut cold_slots: Vec<Vec<(u32, f32)>> = vec![Vec::with_capacity(top_k); ntok];
    let mut failed: Option<&'static str> = None;
    let head_hc = device_hc_w(&globals.head_hc);
    let lm_head_idx = globals.lm_head_idx.unwrap_or(usize::MAX);
    // The readback stage of a chain holds its layer frames and, on the
    // head, the logits of the last token, or of every position of a verify
    // window. A 248k vocabulary is ~0.95 MiB a row: the initial 4 MiB stage
    // holds a window of four (MTP k = 3); five rows (k = 4) overran it.
    let logit_rows = match (want_logits, snapshot) {
        (false, _) => 0,
        // (a copy frame's ids ride behind its one row)
        (true, false) if frame_dbg => ntok + 1,
        (true, false) => 1 + usize::from(frame_verify),
        (true, true) => ntok,
    };
    let stage_need = q4::chain_stage_bytes(
        chain_len.min(n),
        frame_bytes,
        logit_rows,
        q4::head_stride(&model, lm_head_idx).unwrap_or(0),
    );
    if !dev.ensure_stage(stage_need) {
        failed = Some("readback stage allocation");
    }
    // a copy frame keeps no per-token snapshot: the state before it, whole
    if frame_verify && !dev.checkpoint(&g) {
        failed = Some("copy frame checkpoint");
    }
    for (t, &id) in ids.iter().enumerate() {
        let mut emb = vec![0.0f32; cfg.hidden];
        if (id as usize) < globals.embed.rows() {
            globals.embed.row_f32(id as usize, &mut emb);
        }
        dev.seed(t, &emb, cfg.hc);
    }
    // the PLE rows of every token of the chunk, against its own history
    let tp = std::time::Instant::now();
    let ple_rows = ple_frame_rows(layers, cfg, &state.token_history, ids, pool);
    pf.ple += tp.elapsed();
    // Frames 0..n are the layers, frame n the head. A chain of frames goes
    // out in one submission; its layers are gated on the card so that a
    // cold expert in one of them leaves every later frame unexecuted, and
    // the chain resumes after the miss once the arena admitted it.
    let frames_total = if want_logits { n + 1 } else { n };
    // nothing can miss when every expert is resident: plain dispatches
    dev.gated = chain_len > 1 && !resident_all;
    let mut fi = 0usize;
    let mut vocab_out = 0usize;
    let free_run = lean
        && resident_all
        && !dev.gated
        && !q4::ts_probe()
        && std::env::var("CMF_QWEN_FREE_RUN").as_deref() != Ok("0");
    // Single-frame chains are encoded one frame ahead, while the card runs
    // the current one: with one frame per chain the next chain always
    // starts at the next frame, miss or not.
    // It is finished into a command buffer right there: wgpu records the
    // real commands at finish, and doing it after the fence would put that
    // replay between one frame's fence and the next frame's submit.
    let mut prepared: Option<(wgpu::CommandBuffer, u64, Option<usize>, usize)> = None;
    let dump_hyper = std::env::var("CMF_QWEN_DUMP_HYPER").ok();
    let mut t_fence: Option<std::time::Instant> = None;
    // the row stride of the head's logits in the stage (bytes)
    let mut lstride_out = 0usize;
    while fi < frames_total && failed.is_none() {
        let hi = (fi + chain_len).min(frames_total);
        let t0 = std::time::Instant::now();
        let encode_chain = |lo: usize,
                            hi: usize,
                            dev: &mut q4::Dev,
                            arena: &mut QwenGpuPool,
                            picks_prev: &[Vec<usize>],
                            pf: &mut DevProf,
                            vocab_out: &mut usize|
         -> Result<
            (wgpu::CommandEncoder, u64, Option<usize>, usize),
            &'static str,
        > {
            let mut enc = q4::new_encoder("qwen4-chain").ok_or("encoder")?;
            // (the launch probe needs CMF_QWEN_CHAINS=1: the next chain's encode would clear this one's marks)
            q4::dts_reset();
            let merge = q4::merge_guard(&enc);
            q4::ts_mark(&mut enc, 0);
            // chain head: the previous frame's MoE output with its cold winners
            if !q4::encode_pending(&mut enc, dev, &g, ntok) {
                return Err("pending inject");
            }
            q4::pending_done(dev);
            q4::ts_mark(&mut enc, 1);
            let mut stage_off = 0u64;
            let mut logits_off = None;
            let mut lstride = 0usize;
            for f in lo..hi {
                let inject_prev = f > lo;
                if f == n {
                    for i in 2..8 {
                        q4::ts_mark(&mut enc, i);
                    }
                    // the last position's final hyper state, for the MTP's next cell
                    dev.keep_r(&mut enc, ntok - 1);
                    let hw = head_hc.as_ref().ok_or("head hyper-connection indices")?;
                    // logits of the chunk's last token; of every token in a
                    // verify window
                    let first = if snapshot || frame_dbg { 0 } else { ntok - 1 };
                    let (lb, vocab, ls) = q4::encode_head(
                        &mut enc,
                        dev,
                        &model,
                        &g,
                        hw,
                        lm_head_idx,
                        inject_prev,
                        ntok,
                    )
                    .ok_or("head frame declined")?;
                    let bytes = ((ntok - first) * ls) as u64;
                    if !q4::copy_to_stage(&mut enc, dev, &lb, (first * ls) as u64, stage_off, bytes)
                    {
                        return Err("logits overrun the readback stage");
                    }
                    logits_off = Some(stage_off as usize);
                    *vocab_out = vocab;
                    lstride = ls;
                    stage_off += bytes.div_ceil(16) * 16;
                    if frame_verify {
                        // every row's argmax, behind the last row
                        let idb = q4::encode_argmax_rows(&mut enc, dev, &lb, vocab, ls, ntok)
                            .ok_or("copy frame argmax")?;
                        if !q4::copy_to_stage(&mut enc, dev, &idb, 0, stage_off, (ntok * 4) as u64) {
                            return Err("copy frame ids overrun the readback stage");
                        }
                        stage_off += ((ntok * 4) as u64).div_ceil(16) * 16;
                    }
                    continue;
                }
                let layer = &layers[f];
                let ix = layer.idx.as_ref().ok_or("layer index")?;
                let w = device_layer_w(layer, ix).ok_or("layer weights")?;
                let ta = std::time::Instant::now();
                let (remap, shared_slot) = match arena.ensure(
                    &model,
                    f,
                    &picks_prev[f],
                    &layer.expert_ids,
                    layer.shared_ids,
                ) {
                    Some(r) => r,
                    None => arena.remap_snapshot(f),
                };
                pf.admit += ta.elapsed();
                if shared_slot == u32::MAX {
                    return Err("shared expert slot");
                }
                let out = q4::encode_layer(
                    &mut enc,
                    dev,
                    &model,
                    &g,
                    &w,
                    f,
                    pos0,
                    ntok,
                    inv_freq,
                    &remap,
                    shared_slot,
                    &ple_rows[f],
                    inject_prev,
                    snapshot,
                )
                .ok_or("layer frame declined")?;
                if lean {
                    continue;
                }
                if !q4::copy_to_stage(&mut enc, dev, &out.cold, 0, stage_off, (ntok * cs) as u64)
                    || !q4::copy_to_stage(
                        &mut enc,
                        dev,
                        &out.x2,
                        0,
                        stage_off + cold_part as u64,
                        (ntok * hidden_bytes) as u64,
                    )
                {
                    return Err("layer frame overruns the readback stage");
                }
                stage_off += frame_bytes as u64;
            }
            q4::ts_mark(&mut enc, 8);
            q4::ts_resolve(&mut enc);
            dev.arm_chain(lo, hi);
            drop(merge);
            Ok((enc, stage_off, logits_off, lstride))
        };
        let (cb, stage_off, logits_off, ls) = match prepared.take() {
            Some(p) => p,
            None => match encode_chain(
                fi,
                hi,
                dev,
                arena,
                &state.picks_prev,
                &mut pf,
                &mut vocab_out,
            ) {
                Ok((e, a, b, c)) => {
                    let tf = std::time::Instant::now();
                    let cb = q4::finish_frame(e);
                    pf.finish += tf.elapsed();
                    (cb, a, b, c)
                }
                Err(why) => {
                    failed = Some(why);
                    break;
                }
            },
        };
        if ls != 0 {
            lstride_out = ls;
        }
        pf.encode += t0.elapsed();
        // the chain head's pending inject gets the previous frame's cold data
        if !q4::finalize_pending(dev, &g, ntok, &cold_vec, &cold_slots) {
            failed = Some("finalize");
            break;
        }
        let t0 = std::time::Instant::now();
        // A layer chain of a lean frame with every expert resident brings nothing back (no frame can miss, nothing is
        // copied out): it is left running and the next chain follows it into the queue as soon as it is encoded; only
        // the chain with the head is waited on. A fence per chain cost the card an idle gap per chain — the next
        // chain's queue submission ran after it (`CMF_QWEN_FREE_RUN=0`: the fence per chain).
        if free_run && hi < frames_total && stage_off == 0 && arena.tier.is_none() {
            let uploads = arena.take_uploads();
            if q4::submit_indexed(uploads, cb).is_none() {
                failed = Some("submit");
                break;
            }
            arena.rearm_uploads();
            for f in fi..hi.min(n) {
                dev.commit(f);
            }
            pf.wait += t0.elapsed();
            pf.chains += 1;
            fi = hi;
            continue;
        }
        // staged admissions land before the frame that reads their slots:
        // their copies go first in the frame's own queue submission
        let uploads = arena.take_uploads();
        let Some(pend) = q4::submit_frame(dev, uploads, cb, stage_off) else {
            failed = Some("submit");
            break;
        };
        arena.rearm_uploads();
        if defer_readback && hi >= frames_total {
            // this chunk runs while the caller feeds the next; the previous
            // one's stage is collected now (it holds nothing to act on)
            if let Some(prev) = state.inflight.replace(pend)
                && prev.wait().is_none()
            {
                failed = Some("readback");
                break;
            }
            arena.uploads_done();
            pf.chains += 1;
            fi = hi;
            continue;
        }
        if let Some(tf) = t_fence.take() {
            pf.post += tf.elapsed();
        }
        // one frame per chain (or every chain with all experts resident: no
        // frame can miss, so the next chain always starts at `hi`): the next
        // chain is encoded while this one runs
        arena.hold_layers = [None, None];
        if (chain_len == 1 || resident_all) && hi < frames_total {
            let te = std::time::Instant::now();
            match encode_chain(
                hi,
                (hi + chain_len).min(frames_total),
                dev,
                arena,
                &state.picks_prev,
                &mut pf,
                &mut vocab_out,
            ) {
                Ok((e, a, b, c)) => {
                    let tf = std::time::Instant::now();
                    prepared = Some((q4::finish_frame(e), a, b, c));
                    pf.finish += tf.elapsed();
                }
                Err(why) => {
                    failed = Some(why);
                    break;
                }
            }
            pf.encode += te.elapsed();
            // its remap is now fixed: the admissions below must not evict
            // that layer's experts, nor this layer's own winners of the frame
            arena.hold_layers = [(hi < n).then_some(hi), (fi < n).then_some(fi)];
        }
        let ts = std::time::Instant::now();
        let Some(bytes) = pend.wait() else {
            failed = Some("readback");
            break;
        };
        arena.uploads_done();
        pf.spin += ts.elapsed();
        pf.wait += t0.elapsed();
        if let Some(m) = q4::ts_read() {
            pf.gpu_note(fi, n, &m, layers.get(fi).map(|l| matches!(l.mixer, Mixer::Gdn(_))));
        }
        t_fence = Some(std::time::Instant::now());
        pf.chains += 1;
        // the first frame of the chain that routed to a cold expert
        let mut miss_at: Option<usize> = None;
        let mut off = 0usize;
        'frames: for f in fi..hi {
            if f == n {
                break;
            }
            let layer = &layers[f];
            dev.commit(f);
            if lean {
                continue 'frames;
            }
            if let Some(dir) = dump_hyper.as_deref()
                && let Some(rows) = q4::read_hyper(dev, ntok, cfg.hc * cfg.hidden)
            {
                let _ = std::fs::create_dir_all(&dir);
                for (t, row) in rows.iter().enumerate() {
                    let path = std::path::Path::new(dir)
                        .join(format!("pos{:05}_layer{f:02}.f32", pos0 + t));
                    let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let _ = std::fs::write(path, bytes);
                }
            }
            let mut any_cold_in_frame = false;
            let frame_off = off;
            off += frame_bytes;
            // Several tokens: reserve every token's cold winners in token
            // order (the slots per-token admission would pick) and upload
            // them in one parallel batch.
            let mut pre_slots: Vec<Option<Vec<Option<u32>>>> = vec![None; ntok];
            if cold_on_device && ntok > 1 && admit_batch_env() {
                let ta = std::time::Instant::now();
                let mut toks = Vec::new();
                let mut reqs = Vec::new();
                for t in 0..ntok {
                    let w_off = frame_off + t * cs;
                    let word = |j: usize| {
                        let b = &bytes[w_off + 4 * j..w_off + 4 * j + 4];
                        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
                    };
                    let eids: Vec<usize> = (0..top_k)
                        .map(|j| word(2 * j))
                        .filter(|&e| e != u32::MAX && (e as usize) < layer.moe.experts.len())
                        .map(|e| e as usize)
                        .collect();
                    if eids.is_empty() {
                        continue;
                    }
                    let picks: Vec<usize> = (0..top_k)
                        .map(|j| word(2 * top_k + 2 * j))
                        .filter(|&e| e != u32::MAX && (e as usize) < g.n_experts)
                        .map(|e| e as usize)
                        .collect();
                    toks.push(t);
                    reqs.push((eids, picks));
                }
                if !reqs.is_empty() {
                    let got = arena.admit_frame(&model, f, &reqs, &layer.expert_ids);
                    for (t, sl) in toks.into_iter().zip(got) {
                        pre_slots[t] = Some(sl);
                    }
                }
                pf.admit += ta.elapsed();
            }
            let mut trace_row: Option<Vec<u16>> = (!check && trace_on()).then(|| Vec::with_capacity(ntok * top_k));
            for t in 0..ntok {
                let w_off = frame_off + t * cs;
                let words: Vec<u32> = bytes[w_off..w_off + 4 * top_k * 4]
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                let x_off = frame_off + cold_part + t * hidden_bytes;
                let picks: Vec<usize> = (0..top_k)
                    .map(|j| words[2 * top_k + 2 * j])
                    .filter(|&e| e != u32::MAX && (e as usize) < g.n_experts)
                    .map(|e| e as usize)
                    .collect();
                if let Some(pr) = state.profile.as_mut() {
                    pr.note(f, &picks);
                }
                note_route(&mut state.route_hist, n, g.n_experts, f, &picks);
                if route_wstat() {
                    note_route_weights(&words[2 * top_k..4 * top_k]);
                }
                if let Some(tr) = trace_row.as_mut() {
                    tr.extend((0..top_k).map(|j| picks.get(j).map_or(u16::MAX, |&e| e as u16)));
                }
                if check && ntok == 1 && tap_layer() == Some(f) && position < 3 && f + 1 == hi {
                    // the frame's scratch is still the layer's own: read it
                    let bufs = q4::tap_bufs(dev, &g);
                    let h = cfg.hidden;
                    let parts: Vec<(&wgpu::Buffer, u64)> = vec![
                        (&bufs[0], (h * 4) as u64),
                        (&bufs[1], (h * 4) as u64),
                        (&bufs[2], (h * 4) as u64),
                        (&bufs[3], (h * 4) as u64),
                        (&bufs[4], (cfg.hc * h * 4) as u64),
                    ];
                    if let Some(enc) = q4::new_encoder("qwen4-tap")
                        && let Some(tb) = q4::submit_readback(enc, &parts)
                    {
                        let fl = |o: usize, cnt: usize| -> Vec<f32> {
                            tb[o..o + cnt * 4]
                                .chunks_exact(4)
                                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                                .collect()
                        };
                        let al = |cnt: usize| ((cnt * 4) as u64).div_ceil(16) as usize * 16;
                        let mut o = 0;
                        let x = fl(o, h);
                        o += al(h);
                        let blk = fl(o, h);
                        o += al(h);
                        let x2v = fl(o, h);
                        o += al(h);
                        let mo = fl(o, h);
                        o += al(h);
                        let hyper = fl(o, cfg.hc * h);
                        DEV_TAP
                            .lock()
                            .unwrap()
                            .push((position, x, blk, x2v, mo, hyper));
                    }
                }
                let any_cold = (0..top_k).any(|j| words[2 * j] != u32::MAX);
                if t + 1 == ntok {
                    state.picks_prev[f] = picks.clone();
                }
                cold_slots[t].clear();
                cold_vec[t] = None;
                if !any_cold {
                    continue;
                }
                any_cold_in_frame = true;
                // Cold winners: admitted to the arena right away and computed
                // by the card at the head of the next chain; only what the
                // arena cannot take is completed on the host.
                let ta = std::time::Instant::now();
                let x2: Vec<f32> = bytes[x_off..x_off + cfg.hidden * 4]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                let mut jobs: Vec<(&crate::pipeline::DenseFfn, f32)> = Vec::new();
                let cold_list: Vec<(usize, f32)> = (0..top_k)
                    .filter_map(|j| {
                        let e = words[2 * j];
                        (e != u32::MAX && (e as usize) < layer.moe.experts.len())
                            .then(|| (e as usize, f32::from_bits(words[2 * j + 1])))
                    })
                    .collect();
                let eids: Vec<usize> = cold_list.iter().map(|&(e, _)| e).collect();
                // into the arena when it has room (the next token then finds
                // them warm), the rest into this token slot's staging slots
                let slots = if cold_on_device {
                    let mut s = match pre_slots[t].take() {
                        Some(s) if s.len() == eids.len() => s,
                        _ => arena.admit_many(&model, f, &eids, &layer.expert_ids, &picks),
                    };
                    if s.iter().any(Option::is_none) {
                        let rest: Vec<usize> = eids
                            .iter()
                            .zip(&s)
                            .filter(|(_, x)| x.is_none())
                            .map(|(&e, _)| e)
                            .collect();
                        let mut staged = arena
                            .stage_cold(&model, f, t, &rest, &layer.expert_ids, top_k)
                            .into_iter();
                        for x in s.iter_mut().filter(|x| x.is_none()) {
                            *x = staged.next().flatten();
                        }
                    }
                    s
                } else {
                    vec![None; eids.len()]
                };
                for (&(e, wgt), slot) in cold_list.iter().zip(slots) {
                    match slot {
                        Some(sl) => cold_slots[t].push((sl, wgt)),
                        None => jobs.push((&layer.moe.experts[e], wgt)),
                    }
                }
                pf.admit += ta.elapsed();
                pf.cold_experts += cold_slots[t].len() + jobs.len();
                pf.cold_dev += cold_slots[t].len();
                if !jobs.is_empty() {
                    let tc = std::time::Instant::now();
                    // the card adds every winner's tube (`q4t_tube_*`)
                    cold_vec[t] = Some(crate::pipeline::moe_cold_experts_cpu_with(&jobs, &x2, pool, false));
                    pf.cold_cpu += tc.elapsed();
                }
                if check
                    && ntok == 1
                    && tap_layer() == Some(f)
                    && position < 3
                    && let Some(cv) = cold_vec[t].as_ref()
                    && let Some(last) = DEV_TAP.lock().unwrap().last_mut()
                {
                    for (m, c) in last.4.iter_mut().zip(cv) {
                        *m += c;
                    }
                }
            }
            if let Some(tr) = trace_row {
                trace_frame(pos0, f, ntok, top_k, &tr);
            }
            if any_cold_in_frame {
                pf.layers_with_cold += 1;
                miss_at = Some(f);
                break 'frames;
            }
        }
        match miss_at {
            Some(j) => {
                // frames after the miss did not run: the chain resumes there
                dev.discard(j + 1, hi);
                pf.aborted += hi - (j + 1);
                fi = j + 1;
            }
            None => {
                if let Some(lo) = logits_off {
                    let stride = lstride_out;
                    let read = |o: usize| -> Vec<f32> {
                        bytes[o..o + vocab_out * 4]
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect()
                    };
                    if snapshot {
                        // the rows stay as the card wrote them: the round reads their argmax in place and turns
                        // only the row it samples from into floats
                        state.window_rows =
                            Some(WindowRows { bytes, lo, stride, vocab: vocab_out, rows: ntok });
                    } else {
                        let back = if frame_dbg { ntok } else { 1 };
                        *logits = read(lo + (back - 1) * stride);
                        if frame_verify {
                            let io = lo + (back * stride).div_ceil(16) * 16;
                            state.frame_ids = bytes.get(io..io + ntok * 4).map(|b| {
                                b.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect()
                            });
                        }
                        if frame_dbg {
                            // the host's argmax of every row (highest index on a tie, as the card's)
                            let host: Vec<(u32, f32, f32)> = (0..ntok)
                                .map(|r| {
                                    let row = read(lo + r * stride);
                                    let (mut bi, mut bv, mut second) = (0usize, f32::MIN, f32::MIN);
                                    for (i, &v) in row.iter().enumerate() {
                                        if v >= bv {
                                            second = bv;
                                            bv = v;
                                            bi = i;
                                        } else if v > second {
                                            second = v;
                                        }
                                    }
                                    (bi as u32, bv, bv - second)
                                })
                                .collect();
                            eprintln!(
                                "qwen4-frame-debug nt={ntok} card={:?} host={:?} margin={:?}",
                                state.frame_ids.as_deref().unwrap_or(&[]),
                                host.iter().map(|h| h.0).collect::<Vec<_>>(),
                                host.iter().map(|h| (h.2 * 100.0).round() / 100.0).collect::<Vec<_>>(),
                            );
                        }
                    }
                }
                fi = hi;
            }
        }
    }
    arena.flush_uploads();
    if let Some(why) = failed {
        return off(state, why);
    }
    // A token boundary never carries an injection: the next token reseeds
    // the hyper state from its embedding. (A chain that ends with the head
    // leaves the flag set by its layers; without this the next token's
    // first chain would inject this token's last MoE output.)
    dev.clear_pending();
    if !want_logits {
        logits.clear();
    }
    if !check {
        state.token_history.extend_from_slice(ids);
        state.pos = pos0 + ntok;
        if !lean {
            state.route_tokens += ntok;
            phi_dump(&state.route_hist, state.route_tokens, &mut state.phi_last, &mut state.phi_last_tokens);
        }
    }
    if let Some(d) = state.dev.as_mut() {
        d.pos = pos0 + ntok;
        // A prompt past the seal goes over to the far field (a verify window
        // waits for its accepted rows: `decode_spec_device` seals then).
        if !check && !snapshot && d.o1_seal_at().is_some_and(|at| pos0 + ntok >= at) {
            d.seal(&g, pos0 + ntok);
        }
    }
    if let (Some(pr), Ok(path)) = (
        state.profile.as_mut(),
        std::env::var("CMF_QWEN_PROFILE_SAVE"),
    ) {
        pr.tokens += ntok as u64;
        if pr.tokens % 32 < ntok as u64 {
            pr.save(&path);
        }
    }
    if prof {
        let store_line = arena
            .store
            .as_ref()
            .map(|s| s.report())
            .unwrap_or_default();
        let fill_ns = FILL_NS.swap(0, std::sync::atomic::Ordering::Relaxed);
        let fill_n = FILL_N.swap(0, std::sync::atomic::Ordering::Relaxed);
        let fills_wall = FILLS_WALL_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
        let store_line = match arena.tier.as_ref() {
            Some(t) => {
                use std::sync::atomic::Ordering::Relaxed;
                format!(
                    "{store_line} htier hits={} fills={} fill_sum={:.1}ms nofit={} online={} bg_fills={}",
                    t.hits.swap(0, Relaxed),
                    t.fills.swap(0, Relaxed),
                    t.fill_ns.swap(0, Relaxed) as f64 / 1e6,
                    t.misses.swap(0, Relaxed),
                    t.online(),
                    t.bg_fills.load(Relaxed)
                )
            }
            None => store_line,
        };
        eprintln!(
            "qwen4-device pos={position} ntok={ntok} total={:.1}ms encode={:.1}ms finish={:.1}ms wait={:.1}ms spin={:.1}ms post={:.1}ms ple={:.1}ms cold_cpu={:.1}ms admit={:.1}ms chains={} aborted_frames={} cold={} experts ({} on device) in {} layers fills={fill_n} fill_sum={:.1}ms fill_wall={fills_wall:.1}ms {store_line}",
            t_token.elapsed().as_secs_f64() * 1e3,
            pf.encode.as_secs_f64() * 1e3,
            pf.finish.as_secs_f64() * 1e3,
            pf.wait.as_secs_f64() * 1e3,
            pf.spin.as_secs_f64() * 1e3,
            pf.post.as_secs_f64() * 1e3,
            pf.ple.as_secs_f64() * 1e3,
            pf.cold_cpu.as_secs_f64() * 1e3,
            pf.admit.as_secs_f64() * 1e3,
            pf.chains,
            pf.aborted,
            pf.cold_experts,
            pf.cold_dev,
            pf.layers_with_cold,
            fill_ns as f64 / 1e6,
        );
    }
    if prof && pf.gpu_last_end.is_some() {
        let mut line = format!("qwen4-ts pos={position} ntok={ntok}");
        for (name, v) in TS_NAMES.iter().zip(pf.gpu) {
            line.push_str(&format!(" {name}={:.2}", v / 1e6));
        }
        line.push_str(&format!(" busy={:.2} ms", pf.gpu[..16].iter().sum::<f64>() / 1e6));
        eprintln!("{line}");
    }
    if check && ntok == 1 {
        // The host path keeps its own state; run it on every token (its
        // caches must see the whole sequence) and compare the logits where
        // both produced them.
        let dev_logits = logits.clone();
        let mut host = Vec::new();
        state.device_off = true;
        forward_token(
            globals,
            layers,
            cfg,
            state,
            token_id,
            position,
            inv_freq,
            pool,
            &mut host,
            want_logits,
        );
        state.device_off = false;
        if let Some(tl) = tap_layer() {
            let dev_taps = std::mem::take(&mut *DEV_TAP.lock().unwrap());
            let host_taps = std::mem::take(&mut *TAP.lock().unwrap());
            for (pos, x, blk, x2, mo, hyper) in &dev_taps {
                let find = |lab: &str| {
                    host_taps
                        .iter()
                        .find(|(l, li, p, _)| l == lab && *li == tl && p == pos)
                        .map(|(_, _, _, v)| v.as_slice())
                };
                eprintln!("qwen4-tap layer {tl} pos {pos}:");
                if let Some(h) = find("attn_in") {
                    tap_compare("attn_in", h, x);
                }
                if let Some(h) = find("attn_out") {
                    tap_compare("attn_out", h, blk);
                }
                if let Some(h) = find("post_attn") {
                    tap_compare("post_attn", h, hyper);
                }
                if let Some(h) = find("moe_in") {
                    tap_compare("moe_in", h, x2);
                }
                if let Some(h) = find("moe_out") {
                    tap_compare("moe_res", h, mo);
                }
            }
        }
        if !want_logits {
            return true;
        }
        let n = host.len().min(dev_logits.len());
        let mut max_abs = 0.0f32;
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            max_abs = max_abs.max((host[i] - dev_logits[i]).abs());
            dot += host[i] as f64 * dev_logits[i] as f64;
            na += host[i] as f64 * host[i] as f64;
            nb += dev_logits[i] as f64 * dev_logits[i] as f64;
        }
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| {
                    if x > a.1 { (i, x) } else { a }
                })
                .0
        };
        eprintln!(
            "qwen4-check pos={position} cos={:.6} max_abs={max_abs:.4} argmax host={} dev={}",
            dot / (na.sqrt() * nb.sqrt()).max(1e-30),
            argmax(&host),
            argmax(&dev_logits)
        );
        *logits = dev_logits;
    }
    true
}

/// A prompt chunk of several TROWS-token frames, layer by layer, on a card that cannot
/// hold every expert: each layer's whole bank is admitted once and runs every frame of the
/// chunk, the frames' cross-layer state (hyper rows, MoE output, inject weights) parked in
/// `Dev::stream_bank` between them. Nothing is read back and no winner is cold, so the
/// frames run the resident model's kernels; the card keeps one layer queued while the host
/// admits the next. False (nothing done) when the chunk does not qualify; a failure past
/// the first admission turns the device path off.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn forward_stream_device(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    ids: &[u32],
    pos0: usize,
    inv_freq: &[f32],
    pool: Option<&Pool>,
) -> bool {
    use crate::gpu_wgpu::qwen4 as q4;
    use std::time::{Duration, Instant};
    let ntok = ids.len();
    if ntok <= q4::TMAX
        || state.device_off
        || state.resident_all
        || state.verify_window
        || state.pos != pos0
        || state.profile.is_some()
        || state.dev.is_none()
        || state.gpu_pool.is_none()
        || q4::capture_on()
        || std::env::var_os("CMF_QWEN_DUMP_HYPER").is_some()
        || std::env::var("CMF_QWEN_DEVICE_CHECK").as_deref() == Ok("1")
    {
        return false;
    }
    let Some(first) = layers.first() else { return false };
    let Some(e0) = first.moe.experts.first() else { return false };
    let Some(model) = e0.gate_proj.model_arc() else { return false };
    let gu_q2 = e0.gate_proj.model_dtype() == Some(TensorDtype::Q2TiledP);
    let g = device_geom(cfg, &first.moe, gu_q2);
    let n = layers.len();
    let frames: Vec<(usize, usize)> =
        (0..ntok).step_by(q4::TROWS).map(|a| (a, (ntok - a).min(q4::TROWS))).collect();
    if state.gpu_pool.as_ref().is_some_and(|a| a.staging_base < g.n_experts + n + 1) {
        return false;
    }
    let cs = q4::cold_stride(&g);
    let cap_bytes = (n * frames.len() * q4::TROWS * cs) as u64;
    {
        let dev = state.dev.as_mut().unwrap();
        if dev.o1.is_some() || !dev.stream_bank(&g, frames.len()) || !dev.route_cap_ensure(cap_bytes) {
            return false;
        }
        dev.gated = false;
    }
    let prof = std::env::var_os("CMF_QWEN_PROF").is_some();
    let t_all = Instant::now();
    // a chunk the previous call left running
    if let Some(p) = state.inflight.take()
        && p.wait().is_none()
    {
        state.device_off = true;
        return false;
    }
    {
        let dev = state.dev.as_ref().unwrap();
        let mut emb = vec![0.0f32; cfg.hidden];
        for (k, &(a, nt)) in frames.iter().enumerate() {
            for t in 0..nt {
                emb.fill(0.0);
                let id = ids[a + t] as usize;
                if id < globals.embed.rows() {
                    globals.embed.row_f32(id, &mut emb);
                }
                dev.seed_bank(k, t, &emb, cfg.hc);
            }
        }
    }
    let (mut t_admit, mut t_ple, mut t_enc, mut t_wait) = (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let (mut t_sub, mut waits, mut quick_waits) = (Duration::ZERO, 0usize, 0usize);
    // probes: a host pause before each layer's wait (does the card run while the host works?)
    // and the launch profile of every layer's first frame
    let lag_ms = std::env::var("CMF_QWEN_STREAM_LAG_MS").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    let stream_dts = std::env::var("CMF_QWEN_STREAM_DTS").as_deref() == Ok("1") && q4::dts_on();
    if stream_dts {
        q4::dts_stream_line(None);
    }
    let fill_n0 = FILL_N.load(std::sync::atomic::Ordering::Relaxed);
    let mut prev: Option<wgpu::SubmissionIndex> = None;
    let mut failed: Option<&'static str> = None;
    'layers: for f in 0..n {
        let layer = &layers[f];
        let Some(ix) = layer.idx.as_ref() else {
            failed = Some("layer index");
            break;
        };
        let Some(w) = device_layer_w(layer, ix) else {
            failed = Some("layer weights");
            break;
        };
        let arena = state.gpu_pool.as_mut().unwrap();
        arena.hold_layers = [None, None];
        let ta = Instant::now();
        let Some((remap, shared_slot)) = arena.ensure_layer(&model, f, &layer.expert_ids, layer.shared_ids) else {
            failed = Some("the arena cannot hold a layer's experts");
            break;
        };
        let mut uploads = arena.take_uploads();
        t_admit += ta.elapsed();
        let tp = Instant::now();
        let rows = if layer.ple.is_some() {
            ple_rows_of(layers, cfg, &state.token_history, ids, pool, Some(f)).swap_remove(f)
        } else {
            Vec::new()
        };
        t_ple += tp.elapsed();
        let dev = state.dev.as_mut().unwrap();
        for (k, &(a, nt)) in frames.iter().enumerate() {
            let te = Instant::now();
            let Some(mut enc) = q4::new_encoder("qwen4-stream") else {
                failed = Some("encoder");
                break 'layers;
            };
            let dts = stream_dts && k == 0;
            if dts {
                q4::dts_reset();
            }
            if !dev.stream_swap(&mut enc, &g, k, nt, false, f > 0) {
                failed = Some("bank");
                break 'layers;
            }
            let ple: &[Vec<f32>] = if rows.is_empty() { &[] } else { &rows[a..a + nt] };
            if q4::encode_layer(&mut enc, dev, &model, &g, &w, f, pos0 + a, nt, inv_freq, &remap, shared_slot, ple, f > 0, false)
                .is_none()
            {
                failed = Some("layer frame declined");
                break 'layers;
            }
            // the next layer adds this MoE output itself (`inject_prev`)
            dev.clear_pending();
            dev.route_cap_copy(&mut enc, &g, ((f * frames.len() + k) * q4::TROWS * cs) as u64, nt);
            dev.stream_swap(&mut enc, &g, k, nt, true, true);
            if dts {
                q4::dts_flush(&mut enc);
            }
            let cb = q4::finish_frame(enc);
            t_enc += te.elapsed();
            let ts = Instant::now();
            let Some(idx) = q4::submit_indexed(uploads.take(), cb) else {
                failed = Some("submit");
                break 'layers;
            };
            dev.commit(f);
            t_sub += ts.elapsed();
            if dts {
                // the marks are read before the next frame reuses the query set
                if !q4::wait_submission(idx.clone()) {
                    failed = Some("wait");
                    break 'layers;
                }
                q4::dts_take();
            }
            if k + 1 == frames.len()
                && let Some(p) = prev.replace(idx)
            {
                if lag_ms > 0 {
                    std::thread::sleep(Duration::from_millis(lag_ms));
                }
                // the card holds this layer while the host admits the next
                let tw = Instant::now();
                if !q4::wait_submission(p) {
                    failed = Some("wait");
                    break 'layers;
                }
                let w = tw.elapsed();
                waits += 1;
                quick_waits += usize::from(w < Duration::from_micros(300));
                t_wait += w;
            }
        }
        // (the host tier gets its slots back only once the card ran every copy: at the end)
        state.gpu_pool.as_mut().unwrap().rearm_uploads();
    }
    if let Some(p) = prev.take() {
        let tw = Instant::now();
        if !q4::wait_submission(p) && failed.is_none() {
            failed = Some("wait");
        }
        t_wait += tw.elapsed();
    }
    if let Some(a) = state.gpu_pool.as_mut() {
        a.uploads_done();
        a.flush_uploads();
    }
    if stream_dts {
        q4::dts_stream_line(Some(&format!("qwen4-stream-dts pos={pos0} ntok={ntok}")));
    }
    if let Some(why) = failed {
        tracing::warn!("qwen4 device path off: streamed prompt: {why}");
        state.device_off = true;
        return false;
    }
    let dev = state.dev.as_mut().unwrap();
    dev.clear_pending();
    dev.pos = pos0 + ntok;
    // every frame's winners into the sequence's routing
    let tr = Instant::now();
    if let Some(bytes) = dev.route_cap_read(cap_bytes) {
        let k_top = g.top_k;
        for f in 0..n {
            for (k, &(_, nt)) in frames.iter().enumerate() {
                for t in 0..nt {
                    let base = ((f * frames.len() + k) * q4::TROWS + t) * cs;
                    let picks: Vec<usize> = (0..k_top)
                        .filter_map(|j| {
                            let o = base + 4 * (2 * k_top + 2 * j);
                            let e = u32::from_le_bytes(bytes.get(o..o + 4)?.try_into().ok()?);
                            (e != u32::MAX && (e as usize) < g.n_experts).then_some(e as usize)
                        })
                        .collect();
                    note_route(&mut state.route_hist, n, g.n_experts, f, &picks);
                }
            }
        }
        state.route_tokens += ntok;
        phi_dump(&state.route_hist, state.route_tokens, &mut state.phi_last, &mut state.phi_last_tokens);
    }
    let t_route = tr.elapsed();
    state.token_history.extend_from_slice(ids);
    state.pos = pos0 + ntok;
    if prof {
        eprintln!(
            "qwen4-stream routing read in {:.1}ms",
            t_route.as_secs_f64() * 1e3
        );
        eprintln!(
            "qwen4-stream pos={pos0} ntok={ntok} frames={} total={:.1}ms admit={:.1}ms ple={:.1}ms encode={:.1}ms wait={:.1}ms fills={} submit={:.1}ms quick_waits={quick_waits}/{waits}",
            frames.len(),
            t_all.elapsed().as_secs_f64() * 1e3,
            t_admit.as_secs_f64() * 1e3,
            t_ple.as_secs_f64() * 1e3,
            t_enc.as_secs_f64() * 1e3,
            t_wait.as_secs_f64() * 1e3,
            FILL_N.load(std::sync::atomic::Ordering::Relaxed) - fill_n0,
            t_sub.as_secs_f64() * 1e3,
        );
    }
    true
}

/// Count one token's winners of layer `f` into `hist` (`[layers][experts]`, sized on
/// first use).
#[cfg(feature = "gpu")]
fn note_route(hist: &mut Vec<u32>, n_layers: usize, ne: usize, f: usize, picks: &[usize]) {
    if hist.len() != n_layers * ne {
        *hist = vec![0; n_layers * ne];
    }
    for &e in picks {
        if let Some(c) = hist.get_mut(f * ne + e) {
            *c = c.saturating_add(1);
        }
    }
}

/// `CMF_QWEN_PHI_DUMP=<file>`: every `CMF_QWEN_PHI_WINDOW` tokens (default 128, at
/// frame granularity) append the window's routing counts, `[layers][experts]` u32 LE —
/// the rows `cortiq qwen-packs` fits a pack's topology on.
#[cfg(feature = "gpu")]
fn phi_dump(hist: &[u32], tokens: usize, last: &mut Vec<u32>, last_tokens: &mut usize) {
    let Ok(path) = std::env::var("CMF_QWEN_PHI_DUMP") else { return };
    let w = std::env::var("CMF_QWEN_PHI_WINDOW").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(128).max(1);
    if tokens < *last_tokens + w || hist.is_empty() {
        return;
    }
    if last.len() != hist.len() {
        *last = vec![0; hist.len()];
    }
    let mut row = Vec::with_capacity(hist.len() * 4);
    for (h, l) in hist.iter().zip(last.iter()) {
        row.extend_from_slice(&h.saturating_sub(*l).to_le_bytes());
    }
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = f.write_all(&row);
    }
    last.copy_from_slice(hist);
    *last_tokens = tokens;
}

/// `CMF_QWEN_ROUTE_ROWS=<prefix>`: one row per request phase, appended to `<prefix>.prompt`
/// (at the prompt's end) or `<prefix>.gen` (at the next prompt or reset): the phase's
/// token count, then its `[layers][experts]` winner counts, u32 LE.
#[cfg(feature = "gpu")]
fn route_row(kind: &str, hist: &[u32], tokens: usize, snap: &mut Vec<u32>, snap_tokens: &mut usize) {
    let Ok(prefix) = std::env::var("CMF_QWEN_ROUTE_ROWS") else { return };
    if hist.is_empty() {
        return;
    }
    if snap.len() != hist.len() {
        *snap = vec![0; hist.len()];
    }
    let mut row = Vec::with_capacity(hist.len() * 4 + 4);
    row.extend_from_slice(&(tokens.saturating_sub(*snap_tokens) as u32).to_le_bytes());
    for (h, s) in hist.iter().zip(snap.iter()) {
        row.extend_from_slice(&h.saturating_sub(*s).to_le_bytes());
    }
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(format!("{prefix}.{kind}")) {
        let _ = f.write_all(&row);
    }
    snap.copy_from_slice(hist);
    *snap_tokens = tokens;
}

/// `CMF_QWEN_ROUTE_TRACE=<file>`: every decode frame's winners (a frame = one layer of a
/// window of at most TMAX tokens): pos0 u32, layer u16, ntok u16, then ntok × top_k
/// expert ids u16 (u16::MAX: none), LE — what an arena policy is replayed on.
#[cfg(feature = "gpu")]
static ROUTE_TRACE: std::sync::Mutex<Option<std::io::BufWriter<std::fs::File>>> = std::sync::Mutex::new(None);

#[cfg(feature = "gpu")]
fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CMF_QWEN_ROUTE_TRACE").is_some())
}

#[cfg(feature = "gpu")]
fn trace_frame(pos0: usize, layer: usize, ntok: usize, top_k: usize, picks: &[u16]) {
    use std::io::Write;
    let mut g = ROUTE_TRACE.lock().unwrap();
    if g.is_none() {
        let Ok(path) = std::env::var("CMF_QWEN_ROUTE_TRACE") else { return };
        match std::fs::OpenOptions::new().create(true).append(true).open(path) {
            Ok(f) => *g = Some(std::io::BufWriter::with_capacity(1 << 20, f)),
            Err(_) => return,
        }
    }
    let w = g.as_mut().unwrap();
    let mut rec = Vec::with_capacity(8 + ntok * top_k * 2);
    rec.extend_from_slice(&(pos0 as u32).to_le_bytes());
    rec.extend_from_slice(&(layer as u16).to_le_bytes());
    rec.extend_from_slice(&(ntok as u16).to_le_bytes());
    for &p in picks.iter().take(ntok * top_k) {
        rec.extend_from_slice(&p.to_le_bytes());
    }
    let _ = w.write_all(&rec);
}

/// A request boundary in the trace: pos0 = u32::MAX, layer 1 = a prompt starts, 2 = its decode starts.
#[cfg(feature = "gpu")]
fn trace_mark(kind: u16) {
    if trace_on() {
        trace_frame(u32::MAX as usize, kind as usize, 0, 0, &[]);
    }
}

#[cfg(feature = "gpu")]
fn trace_flush() {
    use std::io::Write;
    if let Some(w) = ROUTE_TRACE.lock().unwrap().as_mut() {
        let _ = w.flush();
    }
}

/// Expert packs (`CMF_QWEN_PACKS=<dir>`): narrow bundles of experts, one per task
/// (rust, web, python, php, …), each a routing profile of its task's text, chosen per
/// request by the patent-1 resonance router: every pack keeps an isolated topology (a
/// mean and principal rows, `router::fit_descriptor`) of its own routing signature φ,
/// and the prompt goes to the pack whose topology reconstructs its φ best (argmin
/// reconstruction error). φ is the model's own routing of the prompt: the winner counts
/// of the first `phi_layers` layers, each layer a distribution, the whole unit-normed.
#[cfg(feature = "gpu")]
struct PackSet {
    phi_layers: usize,
    names: Vec<String>,
    subs: Vec<(Vec<f32>, Vec<f32>, usize)>,
    profiles: Vec<ExpertProfile>,
}

#[cfg(feature = "gpu")]
impl PackSet {
    fn load(n_layers: usize, ne: usize) -> Option<Self> {
        let dir = std::env::var("CMF_QWEN_PACKS").ok()?;
        let path = std::path::Path::new(&dir).join("router.json");
        let v: serde_json::Value = match std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()) {
            Some(v) => v,
            None => {
                // no topologies: the packs are chosen by name only (`CMF_QWEN_TASK_HINT`)
                let mut set = PackSet { phi_layers: 0, names: Vec::new(), subs: Vec::new(), profiles: Vec::new() };
                let mut files: Vec<_> = std::fs::read_dir(&dir).ok()?.flatten().map(|e| e.path()).collect();
                files.sort();
                for p in files {
                    let Some(name) = p.file_name().and_then(|s| s.to_str()).and_then(|s| s.strip_suffix(".profile")) else {
                        continue;
                    };
                    if let Some(pr) = ExpertProfile::load(&p.to_string_lossy(), n_layers, ne) {
                        set.names.push(name.to_string());
                        set.profiles.push(pr);
                    }
                }
                if set.names.is_empty() {
                    tracing::warn!("qwen4 packs: {} unreadable and no profiles", path.display());
                    return None;
                }
                return Some(set);
            }
        };
        let phi_layers = v["phi_layers"].as_u64()? as usize;
        let mut set = PackSet { phi_layers, names: Vec::new(), subs: Vec::new(), profiles: Vec::new() };
        for p in v["packs"].as_array()? {
            let name = p["name"].as_str()?.to_string();
            let d = &p["descriptor"];
            let rank = d["rank"].as_u64()? as usize;
            let mean = crate::router::decode_f16(d["mean"].as_str()?)?;
            let basis = crate::router::decode_f16(d["basis"].as_str()?)?;
            let dim = phi_layers * ne;
            let prof = ExpertProfile::load(&std::path::Path::new(&dir).join(format!("{name}.profile")).to_string_lossy(), n_layers, ne);
            match prof {
                Some(pr) if mean.len() == dim && basis.len() == rank * dim => {
                    set.names.push(name);
                    set.subs.push((mean, basis, rank));
                    set.profiles.push(pr);
                }
                _ => tracing::warn!("qwen4 packs: pack {name} skipped (profile or topology does not fit this model)"),
            }
        }
        (!set.names.is_empty()).then_some(set)
    }

    /// `CMF_QWEN_TASK_HINT` as `label=weight,…` (two or more known labels): the packs and
    /// their weights, summing to 1.
    fn hint_weights(&self) -> Option<Vec<(usize, f32)>> {
        let text = std::fs::read_to_string(std::env::var("CMF_QWEN_TASK_HINT").ok()?).ok()?;
        let mut w: Vec<(usize, f32)> = text
            .trim()
            .split(',')
            .filter_map(|kv| {
                let (k, v) = kv.split_once('=')?;
                let i = self.names.iter().position(|n| n == k.trim())?;
                Some((i, v.trim().parse::<f32>().ok()?.max(0.0)))
            })
            .collect();
        let s: f32 = w.iter().map(|x| x.1).sum();
        if w.len() < 2 || s <= 0.0 {
            return None;
        }
        w.iter_mut().for_each(|x| x.1 /= s);
        Some(w)
    }

    /// The pack whose topology reconstructs φ best, and every pack's error.
    fn decide(&self, hist: &[u32], ne: usize) -> Option<(usize, Vec<f32>)> {
        // a task named from outside (a text router: `CMF_QWEN_TASK_HINT=<file>` holds the label)
        if let Some(label) = std::env::var("CMF_QWEN_TASK_HINT").ok().and_then(|p| std::fs::read_to_string(p).ok())
            && let Some(i) = self.names.iter().position(|n| n == label.trim())
        {
            return Some((i, Vec::new()));
        }
        if let Some(w) = self.hint_weights()
            && let Some(&(i, _)) = w.iter().max_by(|a, b| a.1.total_cmp(&b.1))
        {
            return Some((i, Vec::new()));
        }
        if self.subs.is_empty() {
            return None;
        }
        let phi = route_phi(hist, ne, self.phi_layers)?;
        let errs: Vec<f32> = self.subs.iter().map(|(m, b, r)| crate::router::recon_error(&phi, m, b, *r)).collect();
        let best = errs.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1))?.0;
        Some((best, errs))
    }
}

/// φ of a routing histogram: the first `layers` layers' winner distributions, unit-normed.
fn route_phi(hist: &[u32], ne: usize, layers: usize) -> Option<Vec<f32>> {
    if hist.len() < layers * ne || layers == 0 {
        return None;
    }
    let mut phi = Vec::with_capacity(layers * ne);
    for l in 0..layers {
        let row = &hist[l * ne..(l + 1) * ne];
        let s = row.iter().map(|&c| c as f32).sum::<f32>().max(1.0);
        phi.extend(row.iter().map(|&c| c as f32 / s));
    }
    let n = phi.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n <= 0.0 {
        return None;
    }
    phi.iter_mut().for_each(|v| *v /= n);
    Some(phi)
}

/// The expert core of `CMF_QWEN_CORE=<profile>` (`[layer] -> experts` with a nonzero
/// count), read once.
#[cfg(feature = "gpu")]
fn core_set(n_layers: usize, ne: usize) -> Option<Vec<Vec<usize>>> {
    static CORE: std::sync::OnceLock<Option<Vec<Vec<usize>>>> = std::sync::OnceLock::new();
    CORE.get_or_init(|| {
        let path = std::env::var("CMF_QWEN_CORE").ok()?;
        let Some(p) = ExpertProfile::load(&path, n_layers, ne) else {
            tracing::warn!("qwen4 core: {path} does not fit this model");
            return None;
        };
        let core: Vec<Vec<usize>> =
            (0..n_layers).map(|l| (0..ne).filter(|&e| p.counts[l * ne + e] > 0).collect()).collect();
        eprintln!("qwen4-core: {} experts pinned ({} a layer on average)", core.iter().map(Vec::len).sum::<usize>(), core.iter().map(Vec::len).sum::<usize>() / n_layers.max(1));
        Some(core)
    })
    .clone()
}

/// The prompt has run: on a card that cannot hold every expert, the arena is refilled
/// with the narrow bundle this request needs — the chosen pack's experts blended with
/// the prompt's own routing (`CMF_QWEN_PACK_MIX`, the prompt's share, default 0.5; 1
/// without packs; `CMF_QWEN_SELF_PACK=0` keeps the arena as the prompt left it when
/// there are no packs). Each layer gets the same share of the arena, ranked by score.
#[cfg(feature = "gpu")]
fn prompt_done(layers: &[Layer], state: &mut State) {
    route_row("prompt", &state.route_hist, state.route_tokens, &mut state.row_snap, &mut state.row_snap_tokens);
    state.row_gen = true;
    trace_mark(2);
    if let Some(a) = state.gpu_pool.as_mut() {
        a.sweep_clear();
    }
    if state.resident_all || state.device_off || state.gpu_pool.is_none() || state.route_tokens == 0 {
        return;
    }
    let Some(model) = layers.first().and_then(|l| l.moe.experts.first()).and_then(|e| e.gate_proj.model_arc()) else {
        return;
    };
    let n = layers.len();
    let ne = state.gpu_pool.as_ref().unwrap().n_experts;
    if state.route_hist.len() != n * ne {
        return;
    }
    if state.packs.is_none() {
        state.packs = Some(PackSet::load(n, ne));
    }
    let t0 = std::time::Instant::now();
    let packs = state.packs.as_ref().unwrap().as_ref();
    let choice = packs.and_then(|p| p.decide(&state.route_hist, ne));
    if packs.is_none() && std::env::var("CMF_QWEN_SELF_PACK").as_deref() == Ok("0") {
        return;
    }
    let mix: f32 = match choice {
        None => 1.0,
        Some(_) => std::env::var("CMF_QWEN_PACK_MIX").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5f32).clamp(0.0, 1.0),
    };
    // the chosen pack, or several close tasks at once (`CMF_QWEN_TASK_HINT` holding
    // `label=weight,label=weight`: the text router's probabilities), each a per-layer share
    let weighted: Vec<(usize, f32)> = match packs.and_then(PackSet::hint_weights) {
        Some(w) => w,
        None => choice.as_ref().map(|(i, _)| vec![(*i, 1.0)]).unwrap_or_default(),
    };
    // per layer: experts by blended share, then round-robin over the layers by rank
    let mut per_layer: Vec<Vec<(f32, usize)>> = Vec::with_capacity(n);
    for l in 0..n {
        let h = &state.route_hist[l * ne..(l + 1) * ne];
        let hs = h.iter().map(|&c| c as f32).sum::<f32>().max(1.0);
        let mut pk = vec![0f32; ne];
        for &(i, w) in &weighted {
            let row = &packs.unwrap().profiles[i].counts[l * ne..(l + 1) * ne];
            let ps = row.iter().map(|&c| c as f32).sum::<f32>().max(1.0);
            for (p, &c) in pk.iter_mut().zip(row) {
                *p += w * c as f32 / ps;
            }
        }
        let mut v: Vec<(f32, usize)> = (0..ne)
            .map(|e| {
                let own = h[e] as f32 / hs;
                (mix * own + (1.0 - mix) * pk[e], e)
            })
            .filter(|&(sc, _)| sc > 0.0)
            .collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        per_layer.push(v);
    }
    // the expert core every task shares (`CMF_QWEN_CORE=<profile>`: its nonzero experts)
    // stays in the arena for good and leads the ranking; the request loads the rest
    let core = core_set(n, ne);
    let mut ranked: Vec<(usize, usize)> = Vec::new();
    if let Some(c) = core.as_ref() {
        for (l, row) in c.iter().enumerate() {
            ranked.extend(row.iter().map(|&e| (l, e)));
        }
    }
    let in_core = |l: usize, e: usize| core.as_ref().is_some_and(|c| c[l].contains(&e));
    let mut cursor = vec![0usize; n];
    for _ in 0..ne {
        for (l, v) in per_layer.iter().enumerate() {
            while let Some(&(_, e)) = v.get(cursor[l]) {
                cursor[l] += 1;
                if !in_core(l, e) {
                    ranked.push((l, e));
                    break;
                }
            }
        }
    }
    let triples: Vec<Vec<(usize, usize, usize)>> = layers.iter().map(|l| l.expert_ids.clone()).collect();
    let arena = state.gpu_pool.as_mut().unwrap();
    if let Some(c) = core.as_ref()
        && !state.core_pinned
    {
        arena.set_pins(c);
        state.core_pinned = true;
    }
    let (evicted, admitted) = arena.repack(&model, &ranked, &triples);
    arena.flush_uploads();
    // every layer must hold its top_k for a masked ranking
    let k_top = layers[0].moe.top_k;
    let enough = (0..n).all(|l| arena.occupancy.get(l).copied().unwrap_or(0) > k_top);
    if std::env::var("CMF_QWEN_MASK").as_deref() == Ok("1") && enough {
        state.masked = true;
        if let Some(d) = state.dev.as_mut() {
            d.set_masked(true);
        }
    }
    let name = choice.as_ref().map_or("prompt routing", |(i, _)| packs.unwrap().names[*i].as_str());
    tracing::info!(
        "qwen4 packs: {name} for this request ({} prompt tokens, prompt share {mix}), {evicted} experts out, {admitted} in, {:.0} ms",
        state.route_tokens,
        t0.elapsed().as_secs_f64() * 1e3
    );
    if std::env::var_os("CMF_QWEN_PROF").is_some() || std::env::var_os("CMF_QWEN_PACK_LOG").is_some() {
        let errs = choice.as_ref().map(|(_, e)| {
            packs.unwrap().names.iter().zip(e).map(|(nm, x)| format!("{nm}={x:.4}")).collect::<Vec<_>>().join(" ")
        });
        eprintln!(
            "qwen4-pack {name} tokens={} mix={mix} out={evicted} in={admitted} {:.0}ms errors: {}",
            state.route_tokens,
            t0.elapsed().as_secs_f64() * 1e3,
            errs.unwrap_or_default()
        );
    }
}

/// Fit the packs' router (`cortiq qwen-packs`): `dir` holds per pack `<name>.phi` (the
/// windows `CMF_QWEN_PHI_DUMP` wrote over that task's text) and `<name>.profile`. Each
/// pack's topology is fitted on its own windows alone (summed `blocks` at a time, so short
/// and long prompts are both seen), the last ones held out; `router.json` gets written
/// and the report holds the held-out argmin accuracy per pack.
pub fn fit_packs(dir: &str, phi_layers: usize, rank: usize, blocks: &[usize]) -> Result<String, String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| format!("{dir}: {e}"))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".phi")).map(str::to_string))
        .collect();
    names.sort();
    if names.is_empty() {
        return Err(format!("{dir}: no <name>.phi files"));
    }
    let mut report = String::new();
    let mut packs = Vec::new();
    let mut holds: Vec<(usize, Vec<Vec<f32>>)> = Vec::new();
    for (pi, name) in names.iter().enumerate() {
        let prof = std::fs::read(format!("{dir}/{name}.profile")).map_err(|e| format!("{name}.profile: {e}"))?;
        if prof.len() < 32 || &prof[..8] != b"CMFQ4PF\0" {
            return Err(format!("{name}.profile is not a routing profile"));
        }
        let rd = |o: usize| u64::from_le_bytes(prof[o..o + 8].try_into().unwrap()) as usize;
        let (nl, ne) = (rd(8), rd(16));
        let raw = std::fs::read(format!("{dir}/{name}.phi")).map_err(|e| format!("{name}.phi: {e}"))?;
        let w = nl * ne * 4;
        if raw.len() < w || raw.len() % w != 0 {
            return Err(format!("{name}.phi: not whole {nl}x{ne} rows"));
        }
        let rows: Vec<Vec<u32>> = raw
            .chunks_exact(w)
            .map(|r| r.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
            .collect();
        let mut phis: Vec<Vec<f32>> = Vec::new();
        for &b in blocks {
            for grp in rows.chunks(b.max(1)) {
                if grp.len() < b.max(1) {
                    continue;
                }
                let mut sum = vec![0u32; nl * ne];
                for r in grp {
                    for (s, &v) in sum.iter_mut().zip(r) {
                        *s += v;
                    }
                }
                if let Some(p) = route_phi(&sum, ne, phi_layers) {
                    phis.push(p);
                }
            }
        }
        let d = crate::router::fit_descriptor(&phis, phi_layers, rank).ok_or(format!("{name}: too few windows ({})", phis.len()))?;
        let hold = d.holdout.as_deref().and_then(crate::router::decode_f16).unwrap_or_default();
        let dim = phi_layers * ne;
        holds.push((pi, hold.chunks_exact(dim).map(|c| c.to_vec()).collect()));
        report.push_str(&format!("{name}: {} windows ({} raw), rank {}\n", phis.len(), rows.len(), d.rank));
        packs.push(serde_json::json!({ "name": name, "descriptor": d, "windows": phis.len() }));
    }
    // held-out argmin over every pack's topology, as decided at run time
    let subs: Vec<(Vec<f32>, Vec<f32>, usize)> = packs
        .iter()
        .map(|p| {
            let d = &p["descriptor"];
            (
                crate::router::decode_f16(d["mean"].as_str().unwrap_or("")).unwrap_or_default(),
                crate::router::decode_f16(d["basis"].as_str().unwrap_or("")).unwrap_or_default(),
                d["rank"].as_u64().unwrap_or(1) as usize,
            )
        })
        .collect();
    let (mut ok, mut tot) = (0usize, 0usize);
    for (pi, hs) in &holds {
        let mut right = 0;
        let mut confused: std::collections::BTreeMap<&str, usize> = Default::default();
        for h in hs {
            let errs: Vec<f32> = subs.iter().map(|(m, b, r)| crate::router::recon_error(h, m, b, *r)).collect();
            let best = errs.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).map(|x| x.0).unwrap_or(0);
            if best == *pi {
                right += 1;
            } else {
                *confused.entry(names[best].as_str()).or_default() += 1;
            }
        }
        ok += right;
        tot += hs.len();
        report.push_str(&format!("held-out {}: {right}/{} {:?}\n", names[*pi], hs.len(), confused));
    }
    report.push_str(&format!("held-out total: {ok}/{tot}\n"));
    let out = serde_json::json!({ "phi_layers": phi_layers, "rank": rank, "blocks": blocks, "packs": packs });
    std::fs::write(format!("{dir}/router.json"), serde_json::to_vec_pretty(&out).map_err(|e| e.to_string())?)
        .map_err(|e| format!("router.json: {e}"))?;
    Ok(report)
}

#[cfg(feature = "gpu")]
fn device_hc_w(g: &GatedResidual) -> Option<crate::gpu_wgpu::qwen4::HcW<'_>> {
    Some(crate::gpu_wgpu::qwen4::HcW {
        norm: &g.norm,
        down: g.down_idx?,
        up: g.up_idx?,
        inject: g.inject_idx,
    })
}

/// The verify window (k + 1 positions) the 768 MiB default of
/// `CMF_QWEN_KV_RESERVE_MB` was measured with: MTP k = 3.
#[cfg(feature = "gpu")]
const MTP_MEASURED_WINDOW: usize = 4;

/// Drafts per MTP round from `CMF_QWEN_MTP_K` (default 3), and whether the
/// value had to be clamped. The verify window, the last accepted token plus
/// k drafts, is one device frame, so k + 1 <= TMAX: k in 1..=7.
#[cfg(feature = "gpu")]
fn mtp_k(raw: Option<&str>) -> (usize, bool) {
    let max = crate::gpu_wgpu::qwen4::TMAX - 1;
    match raw.map(|v| v.trim().parse::<usize>()) {
        None => (3, false),
        Some(Ok(v)) => (v.clamp(1, max), !(1..=max).contains(&v)),
        Some(Err(_)) => (3, true),
    }
}

/// A verify window's logits as they came off the card: `rows` rows of `vocab` little-endian f32, row `t` at
/// `lo + t·stride` of `bytes`.
struct WindowRows {
    bytes: Vec<u8>,
    lo: usize,
    stride: usize,
    vocab: usize,
    rows: usize,
}

impl WindowRows {
    fn row(&self, t: usize) -> &[u8] {
        let o = self.lo + t * self.stride;
        &self.bytes[o..o + self.vocab * 4]
    }

    fn floats(&self, t: usize) -> Vec<f32> {
        self.row(t).chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }
}

/// The first index of the largest value in a row of little-endian f32; a NaN never wins and a row without a value
/// above −∞ answers 0 — what a scalar first-maximum fold answers, as two passes the compiler turns into vector code
/// (the fold's compare-and-carry chain ran a quarter of a millisecond per 248 320-entry row, four rows a round).
fn argmax_le(row: &[u8]) -> usize {
    const L: usize = 16;
    let val = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let mut m = [f32::NEG_INFINITY; L];
    let mut it = row.chunks_exact(4 * L);
    for c in &mut it {
        for (j, b) in c.chunks_exact(4).enumerate() {
            let x = val(b);
            m[j] = if x > m[j] { x } else { m[j] };
        }
    }
    let mut best = f32::NEG_INFINITY;
    for &x in &m {
        best = if x > best { x } else { best };
    }
    for b in it.remainder().chunks_exact(4) {
        let x = val(b);
        best = if x > best { x } else { best };
    }
    if best == f32::NEG_INFINITY {
        return 0;
    }
    for (ci, c) in row.chunks(4 * L).enumerate() {
        let mut hit = false;
        for b in c.chunks_exact(4) {
            hit |= val(b) == best;
        }
        if hit {
            return ci * L + c.chunks_exact(4).position(|b| val(b) == best).unwrap_or(0);
        }
    }
    0
}

/// One speculative round's outcome, in the shape the generation loop
/// consumes: the drafts the model confirmed, the logits the loop samples
/// its own next token from, how many were drafted.
#[cfg(feature = "gpu")]
pub struct SpecRound {
    pub accepted: Vec<u32>,
    pub logits: Vec<f32>,
    pub drafted: usize,
}

/// The Qwen3.8-Flash-Next multi-token-prediction head from the sidecar
/// `<stem>.mtp.cmf`: one hyper-connected hybrid layer (QSA attention, its
/// own 512 routed experts and shared expert) between a fused input and a
/// final mixer, sharing the main model's embedding and lm_head.
///
///   e = fc_embedding(rms(embed(tok))·(1+w_e))
///   h = fc_hidden(rms_{hc·hidden}(R)·(1+w_h)) per stream
///   R' = h + e → attention HC → QSA → MLP HC → MoE → inject → mixer → head
///
/// Cell i pairs the main model's final hyper state at position i with the
/// token at i+1 (rope position i) and predicts the token at i+2; its own
/// state feeds the next draft of a chain. Drafts only propose: the verify
/// window decides every emitted token.
#[cfg(feature = "gpu")]
pub struct MtpHead {
    side: Arc<CmfModel>,
    layer: Layer,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    fc_e: usize,
    fc_h: usize,
    mixer: GatedResidual,
    skeleton: Vec<usize>,
    dev: Option<crate::gpu_wgpu::qwen4::Dev>,
    arena: Option<QwenGpuPool>,
    pub k: usize,
    pub drafted: u64,
    pub accepted: u64,
    pub rounds: u64,
}

/// The main file carries the MTP draft head itself (`model.mtp.*`).
fn mtp_embedded(model: &CmfModel) -> bool {
    model.tensor("model.mtp.layers.0.self_attn.q_proj.weight").is_some()
}

#[cfg(feature = "gpu")]
impl MtpHead {
    /// Open the draft head: inside the main file, or the sidecar beside it.
    pub fn load(main: &Arc<CmfModel>, arch: &ModelArch) -> Option<Self> {
        if std::env::var("CMF_QWEN_MTP").as_deref() == Ok("0") {
            return None;
        }
        // The draft head rides inside the main file (`moe-defrag --merge`)
        // or in the `<stem>.mtp.cmf` sidecar beside it.
        let side = if mtp_embedded(main) {
            // A second handle on the same file: its own uid keys the head's
            // own expert bank and frame state apart from the backbone's
            // (one uid would hand the backbone the head's 520-slot pool).
            match CmfModel::open(&main.path) {
                Ok(m) => Arc::new(m),
                Err(e) => {
                    tracing::warn!("qwen4 MTP head in {}: {e}", main.path.display());
                    return None;
                }
            }
        } else {
            let path = cortiq_core::mtp_sidecar_path(&main.path);
            if path == main.path || !path.exists() {
                return None;
            }
            match CmfModel::open(&path) {
                Ok(m) => Arc::new(m),
                Err(e) => {
                    tracing::warn!("qwen4 MTP sidecar {}: {e}", path.display());
                    return None;
                }
            }
        };
        let path = side.path.clone();
        let sa = side.arch();
        if sa.arch_name != arch.arch_name
            || sa.hidden_size != arch.hidden_size
            || sa.vocab_size != arch.vocab_size
        {
            tracing::warn!(
                "qwen4 MTP sidecar {}: geometry differs from the backbone",
                path.display()
            );
            return None;
        }
        let p = "model.mtp.layers.0.";
        let build =
            || -> Result<(Layer, GatedResidual, Vec<f32>, Vec<f32>, usize, usize), CmfError> {
                let mixer = Mixer::Qsa(load_qsa(&side, &format!("{p}self_attn."))?);
                let moe = match build_ffn_at(&side, sa, p, false, &Overlay::None)? {
                    FfnKind::Moe(m) => m,
                    _ => return Err(err("MTP layer is not MoE")),
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
                    .ok_or_else(|| err("MTP experts are not mmap-backed"))?;
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
                    .ok_or_else(|| err("MTP shared expert is not mmap-backed"))?;
                let attn_hc = load_hc(&side, &format!("{p}attn_hyper_connection."), true)?;
                let mlp_hc = load_hc(&side, &format!("{p}mlp_hyper_connection."), true)?;
                let idx = layer_idx(&side, p, false, false);
                let layer = Layer {
                    attn_hc,
                    mlp_hc,
                    mixer,
                    moe,
                    ple: None,
                    expert_ids,
                    shared_ids: Some(shared_ids),
                    idx,
                };
                let mixer_hc = load_hc(&side, "model.mtp.hyper_connection_mixer.", false)?;
                let enorm = f(&side, "model.mtp.enorm.weight")?;
                let hnorm = f(&side, "model.mtp.hnorm.weight")?;
                let fc_e = side
                    .tensor_index("model.mtp.fc_embedding.weight")
                    .ok_or_else(|| err("fc_embedding"))?;
                let fc_h = side
                    .tensor_index("model.mtp.fc_hidden.weight")
                    .ok_or_else(|| err("fc_hidden"))?;
                Ok((layer, mixer_hc, enorm, hnorm, fc_e, fc_h))
            };
        let (layer, mixer, enorm, hnorm, fc_e, fc_h) = match build() {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("qwen4 MTP sidecar {}: {e}", path.display());
                return None;
            }
        };
        let ix = layer.idx.as_ref()?;
        let mut skeleton: Vec<usize> = Vec::new();
        skeleton.extend(layer.attn_hc.idxs());
        skeleton.extend(layer.mlp_hc.idxs());
        skeleton.extend(mixer.idxs());
        skeleton.extend(ix.qsa.iter().flatten().copied());
        skeleton.push(ix.router);
        skeleton.extend(ix.shared_gate);
        skeleton.extend([fc_e, fc_h]);
        let raw = std::env::var("CMF_QWEN_MTP_K").ok();
        let (k, clamped) = mtp_k(raw.as_deref());
        if clamped {
            tracing::warn!(
                "CMF_QWEN_MTP_K={} is outside 1..={}: the verify window (k + 1 positions) is one device frame of at most {} tokens; using k = {k}",
                raw.as_deref().unwrap_or(""),
                crate::gpu_wgpu::qwen4::TMAX - 1,
                crate::gpu_wgpu::qwen4::TMAX
            );
        }
        tracing::info!("qwen4 MTP: draft head from {} (k = {k})", path.display());
        Some(Self {
            side,
            layer,
            enorm,
            hnorm,
            fc_e,
            fc_h,
            mixer,
            skeleton,
            dev: None,
            arena: None,
            k,
            drafted: 0,
            accepted: 0,
            rounds: 0,
        })
    }

    /// Resident skeleton, all 512 experts in their own arena bank.
    fn setup(&mut self, g: &crate::gpu_wgpu::qwen4::Geom, gu_q2: bool) -> Result<(), &'static str> {
        use crate::gpu_wgpu::qwen4 as q4;
        if self.dev.is_some() && self.arena.is_some() {
            return Ok(());
        }
        let dev = q4::Dev::new(self.side.uid(), g, &[(false, false)]).ok_or("MTP device state")?;
        q4::prewarm(&self.side, &self.skeleton).ok_or("MTP skeleton does not fit")?;
        let n_exp = self.layer.moe.experts.len();
        // whole segments of 8: the shared expert needs one slot beyond the 512
        let mut arena =
            QwenGpuPool::create_exact(&self.side, g.inter, g.hidden, n_exp, gu_q2, n_exp + 8)
                .ok_or("MTP expert bank")?;
        arena
            .ensure(
                &self.side,
                0,
                &[],
                &self.layer.expert_ids,
                self.layer.shared_ids,
            )
            .filter(|(_, s)| *s != u32::MAX)
            .ok_or("MTP shared expert")?;
        let all: Vec<(usize, usize)> = (0..n_exp).map(|e| (0, e)).collect();
        let got = arena.prefill(
            &self.side,
            &all,
            std::slice::from_ref(&self.layer.expert_ids),
        );
        if got < n_exp {
            return Err("MTP experts do not all fit their bank");
        }
        self.dev = Some(dev);
        self.arena = Some(arena);
        Ok(())
    }
}

/// One speculative round: `k` greedy drafts from the MTP head, verified in
/// one batched forward of the main model (`all_ids[next_pos]` plus the
/// drafts at `next_pos..`). Returns `None` when the device path or the
/// head is unavailable; the caller then decodes plainly.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn spec_round(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    next_pos: usize,
    all_ids: &[u32],
    inv_freq: &[f32],
    pool: Option<&Pool>,
) -> Option<SpecRound> {
    use crate::gpu_wgpu::qwen4 as q4;
    #[cfg(feature = "cuda")]
    if crate::gpu_cuda::selected() {
        return cuda_spec_round(globals, layers, cfg, state, next_pos, all_ids, pool);
    }
    if state.device_off || state.dev.is_none() || next_pos == 0 || all_ids.len() != next_pos + 1 {
        return None;
    }
    if state.mtp.is_none() && !state.mtp_tried {
        state.mtp_tried = true;
        let model = layers.first()?.moe.experts.first()?.gate_proj.model_arc()?;
        state.mtp = MtpHead::load(&model, model.arch());
    }
    let first = layers.first()?;
    let e0 = first.moe.experts.first()?;
    let gu_q2 = e0.gate_proj.model_dtype() == Some(TensorDtype::Q2TiledP);
    let g = device_geom(cfg, &first.moe, gu_q2);
    let prof = std::env::var_os("CMF_QWEN_PROF").is_some();
    let t_round = std::time::Instant::now();
    // the host between two rounds: sampling the round's last row, detokenizing and streaming its tokens
    let t_gap = state.round_end.take().map(|t| t.elapsed()).unwrap_or_default();
    if prof && (t_gap.is_zero() || t_gap.as_millis() > 100) {
        // a request's first round: its context, for offline draft studies
        eprintln!("qwen4-ctx pos={next_pos} ids={:?}", all_ids);
    }
    // ── drafts ──
    let sfx = suffix_cfg();
    // The context repeating itself (an edit of a file it was shown, a test
    // after a test): what followed its tail before is the draft, and no
    // chain runs.
    // A copy that has held goes on as a frame (`CMF_QWEN_COPY_FRAME`).
    let frame_k = copy_frame_cfg()
        .filter(|_| {
            (state.resident_all || state.masked) && state.dev.as_ref().is_some_and(|d| d.o1.is_none())
        })
        .and_then(|(most, factor)| copy_frame_drafts(most, factor, state.copy_run, state.draft_budget));
    let copy = sfx.filter(|s| s.strong > 0 && state.mtp.is_some()).and_then(|s| {
        let (m, cont) = suffix_draft(all_ids, s.strong, frame_k.unwrap_or(s.max));
        (m > 0).then_some((m, cont))
    });
    let copied = copy.is_some();
    // a continuation that fits a window goes as a window, snapshots and all
    let framed = frame_k.is_some() && copy.as_ref().is_some_and(|(_, cont)| cont.len() + 1 > q4::TMAX);
    let mut sfx_note = String::new();
    let (k, drafts, conf) = if let Some((m, cont)) = copy {
        sfx_note = format!(" sfx={}:{m}", if framed { "frame" } else { "copy" });
        (cont.len(), cont, Vec::new())
    } else {
        let head = state.mtp.as_mut()?;
        // The draft head routes over its own expert set: a backbone cut to
        // K experts a layer (`moe-defrag --keep`) keeps the head's full 512,
        // so its frames take the head's geometry, not the backbone's.
        let g = device_geom(cfg, &head.layer.moe, gu_q2);
        if let Err(why) = head.setup(&g, gu_q2) {
            tracing::warn!("qwen4 MTP off: {why}");
            state.mtp = None;
            return None;
        }
        let k = head.k;
        let mdev = head.dev.as_mut()?;
        let arena = head.arena.as_mut()?;
        let (remap, shared_slot) = arena.remap_snapshot(0);
        let ix = head.layer.idx.as_ref()?;
        let w = device_layer_w(&head.layer, ix)?;
        let mixer_w = device_hc_w(&head.mixer)?;
        let main_dev = state.dev.as_ref()?;
        let main_model = e0.gate_proj.model_arc()?;
        let lm_head = globals.lm_head_idx?;
        // a host round trip reads one logits row per cell back
        if !mdev.ensure_stage(q4::chain_stage_bytes(
            0,
            0,
            1,
            q4::head_stride(&main_model, lm_head)?,
        )) {
            return None;
        }
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        // the chain's confidence in each draft (the device chain, when asked)
        let mut conf: Vec<f32> = Vec::new();
        let tok_in = all_ids[next_pos];
        // ── the whole chain in one submit: each cell's argmax is re-embedded
        // on the card for the next; the host reads k ids, not k×vocab logits ──
        // opt-in: it pins the 0.6 GB embedding table, which costs the arena
        // more than the host round trips cost the round
        let device_chain = std::env::var("CMF_QWEN_MTP_DEVICE_DRAFT").as_deref() == Ok("1");
        // The chain's head reads a vocabulary prefix (`CMF_QWEN_DRAFT_VOCAB`
        // rows, 0 = all): a draft outside it is a draft the window rejects,
        // never a wrong token. Text whose ids sit above the prefix (Cyrillic
        // and CJK live past 131072) takes the full head while it lasts.
        let draft_rows = {
            let cut = draft_vocab_rows();
            let guard = cut.max(131072) as u32;
            let tail = &all_ids[all_ids.len().saturating_sub(16)..];
            if cut > 0 && tail.iter().all(|&t| t < guard) { cut } else { 0 }
        };
        let mut chained = false;
        let (minp, report) = draft_gate();
        let conf_buf = if report { q4::draft_conf_buf() } else { None };
        if device_chain
            && let Some(embed_idx) = globals.embed_idx
            && let Some(ids_buf) = q4::draft_ids(tok_in)
        {
            let mut run = || -> Option<(Vec<u32>, Vec<f32>)> {
                let mut enc = q4::new_encoder("qwen4-mtp-draft")?;
                let merge = q4::merge_guard(&enc);
                for j in 0..k {
                    let cell = next_pos - 1 + j;
                    // the cells share one submit: each gets its own frame salt
                    // so its position uniforms and bind groups are its own
                    let _salt = q4::frame_salt(j);
                    let r = if j == 0 {
                        q4::whole(&main_dev.r_last)
                    } else {
                        mdev.hyper(0)
                    };
                    q4::encode_draft_gather(
                        &mut enc,
                        mdev,
                        &main_model,
                        &g,
                        embed_idx,
                        &ids_buf,
                        j,
                    )?;
                    if !q4::encode_mtp_input(
                        &mut enc,
                        mdev,
                        &head.side,
                        &g,
                        None,
                        &head.enorm,
                        &head.hnorm,
                        head.fc_e,
                        head.fc_h,
                        r,
                    ) {
                        return None;
                    }
                    q4::encode_layer(
                        &mut enc,
                        mdev,
                        &head.side,
                        &g,
                        &w,
                        0,
                        cell,
                        1,
                        inv_freq,
                        &remap,
                        shared_slot,
                        &[],
                        false,
                        false,
                    )?;
                    if !q4::encode_pending(&mut enc, mdev, &g, 1)
                        || !q4::finalize_pending(mdev, &g, 1, &[None], &[Vec::new()])
                    {
                        return None;
                    }
                    q4::pending_done(mdev);
                    let (lb, vocab, _) = q4::encode_head_with(
                        &mut enc,
                        mdev,
                        &head.side,
                        &main_model,
                        &g,
                        &mixer_w,
                        lm_head,
                        false,
                        1,
                        draft_rows,
                    )?;
                    q4::encode_argmax(&mut enc, mdev, &lb, vocab, &ids_buf, j + 1, draft_rows > 0)?;
                    if let Some(cb) = &conf_buf {
                        q4::encode_draft_conf(&mut enc, mdev, &lb, vocab, &ids_buf, cb, j + 1, draft_rows > 0)?;
                    }
                }
                if !q4::copy_to_stage(&mut enc, mdev, &ids_buf, 4, 0, (k * 4) as u64) {
                    return None;
                }
                let mut read = k * 4;
                if let Some(cb) = &conf_buf {
                    if !q4::copy_to_stage(&mut enc, mdev, cb, 4, read as u64, (k * 4) as u64) {
                        return None;
                    }
                    read += k * 4;
                }
                drop(merge);
                let bytes = q4::submit_chain(mdev, enc, read as u64)?.wait()?;
                let word = |b: &[u8]| [b[0], b[1], b[2], b[3]];
                Some((
                    bytes[..k * 4].chunks_exact(4).map(|b| u32::from_le_bytes(word(b))).collect(),
                    bytes[k * 4..read].chunks_exact(4).map(|b| f32::from_le_bytes(word(b))).collect(),
                ))
            };
            if let Some((d, c)) = run() {
                drafts = d;
                conf = c;
                chained = true;
            }
        }
        if !chained {
            // host round trip per cell (the embedding table is not on the card)
            let mut tok_in = tok_in;
            for j in 0..k {
                let cell = next_pos - 1 + j;
                let mut emb = vec![0.0f32; cfg.hidden];
                if (tok_in as usize) < globals.embed.rows() {
                    globals.embed.row_f32(tok_in as usize, &mut emb);
                }
                let r = if j == 0 {
                    q4::whole(&main_dev.r_last)
                } else {
                    mdev.hyper(0)
                };
                let mut enc = q4::new_encoder("qwen4-mtp-draft")?;
                let merge = q4::merge_guard(&enc);
                if !q4::encode_mtp_input(
                    &mut enc,
                    mdev,
                    &head.side,
                    &g,
                    Some(&emb),
                    &head.enorm,
                    &head.hnorm,
                    head.fc_e,
                    head.fc_h,
                    r,
                ) {
                    return None;
                }
                q4::encode_layer(
                    &mut enc,
                    mdev,
                    &head.side,
                    &g,
                    &w,
                    0,
                    cell,
                    1,
                    inv_freq,
                    &remap,
                    shared_slot,
                    &[],
                    false,
                    false,
                )?;
                if !q4::encode_pending(&mut enc, mdev, &g, 1)
                    || !q4::finalize_pending(mdev, &g, 1, &[None], &[Vec::new()])
                {
                    return None;
                }
                q4::pending_done(mdev);
                let (lb, vocab, _) = q4::encode_head_with(
                    &mut enc,
                    mdev,
                    &head.side,
                    &main_model,
                    &g,
                    &mixer_w,
                    lm_head,
                    false,
                    1,
                    0,
                )?;
                if !q4::copy_to_stage(&mut enc, mdev, &lb, 0, 0, (vocab * 4) as u64) {
                    return None;
                }
                drop(merge);
                let bytes = q4::submit_chain(mdev, enc, (vocab * 4) as u64)?.wait()?;
                let mut best = (0usize, f32::NEG_INFINITY);
                for (i, ch) in bytes[..vocab * 4].chunks_exact(4).enumerate() {
                    let v = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
                    if v > best.1 {
                        best = (i, v);
                    }
                }
                drafts.push(best.0 as u32);
                tok_in = best.0 as u32;
            }
        }
        // the gate: the window verifies the drafts the chain still believed in
        let kept = if conf.len() == k { draft_gate_keep(&conf, minp) } else { k };
        drafts.truncate(kept);
        // The chain's drafts continued by the context: where the tail with
        // the drafts on it occurred before, what followed it there rides the
        // same window (the chain agreeing with the old text is the evidence).
        if let Some(s) = sfx.filter(|s| s.ext > 0 && kept == k && k < s.max) {
            let mut ctx = all_ids.to_vec();
            ctx.extend_from_slice(&drafts);
            let (m, cont) = suffix_draft(&ctx, s.ext, s.max - k);
            if m > 0 {
                sfx_note = format!(" sfx=ext:{m}+{}", cont.len());
                drafts.extend(cont);
            }
        }
        (drafts.len(), drafts, conf)
    };
    let t_draft = t_round.elapsed();
    // ── verify: the window through the main model, every token's logits ──
    let mut ids = Vec::with_capacity(k + 1);
    ids.push(all_ids[next_pos]);
    ids.extend_from_slice(&drafts);
    let mut last_logits = Vec::new();
    let (a, logits, t_accept) = if framed {
        // ── a copy frame: the prompt kernels over the whole stretch, every row's argmax
        // taken on the card; the last row's logits come back with the ids ──
        state.copy_frame = true;
        state.frame_check = true;
        state.frame_ids = None;
        let ok = forward_tokens_device(
            globals,
            layers,
            cfg,
            state,
            &ids,
            next_pos,
            inv_freq,
            pool,
            &mut last_logits,
            true,
        );
        state.frame_check = false;
        state.copy_frame = false;
        let t_accept = std::time::Instant::now();
        let rows = state.frame_ids.take().filter(|r| ok && r.len() == k + 1)?;
        let mut a = 0usize;
        while a < k && rows[a] == drafts[a] {
            a += 1;
        }
        let t_accept = t_accept.elapsed();
        if a < k {
            // The frame ran past the copy's end and kept no per-token snapshot: back to
            // the state before it, then the accepted stretch again, on its own. Its last
            // row is the next token's logits.
            let dev = state.dev.as_mut()?;
            if !dev.restore_checkpoint(&g) {
                return None;
            }
            dev.pos = next_pos;
            let keep = state.token_history.len() - (k + 1);
            state.token_history.truncate(keep);
            state.pos = next_pos;
            state.copy_frame = a + 1 > q4::TMAX;
            let ok = forward_tokens_device(
                globals,
                layers,
                cfg,
                state,
                &ids[..a + 1],
                next_pos,
                inv_freq,
                pool,
                &mut last_logits,
                true,
            );
            state.copy_frame = false;
            if !ok {
                return None;
            }
            sfx_note.push_str(&format!(" redo={}", a + 1));
        }
        (a, last_logits, t_accept)
    } else {
        state.verify_window = true;
        let ok = forward_tokens_device(
            globals,
            layers,
            cfg,
            state,
            &ids,
            next_pos,
            inv_freq,
            pool,
            &mut last_logits,
            true,
        );
        state.verify_window = false;
        let t_accept = std::time::Instant::now();
        let Some(win) = state.window_rows.take().filter(|w| ok && w.rows == k + 1) else {
            return None;
        };
        let mut a = 0usize;
        while a < k && argmax_le(win.row(a)) == drafts[a] as usize {
            a += 1;
        }
        let logits = win.floats(a);
        drop(win);
        let t_accept = t_accept.elapsed();
        // Sealed O(1) layers: the accepted rows' keys and values go from the spec
        // stash into the ring and the far field; the rejected rows never happened.
        if let Some(uid) = state.dev.as_ref().filter(|d| d.o1_sealed()).map(|d| d.uid()) {
            if !crate::gpu_wgpu::o1_spec_commit(uid, next_pos, a + 1) {
                return None;
            }
        }
        // ── roll back what the rejected drafts changed ──
        if a < k {
            let dev = state.dev.as_mut()?;
            if !dev.restore(&g, a + 1) {
                return None;
            }
            let keep = state.token_history.len() - (k - a);
            state.token_history.truncate(keep);
            state.pos = next_pos + a + 1;
            dev.pos = state.pos;
            // the last accepted position's R for the next round's first cell
            if let Some(mut enc) = q4::new_encoder("qwen4-keep-r") {
                dev.keep_r(&mut enc, a);
                q4::submit_only(enc);
            }
        }
        (a, logits, t_accept)
    };
    // how long the copy has held: the next frame's size
    state.copy_run = if copied && a == k { state.copy_run + k + 1 } else { 0 };
    if let Some(d) = state.dev.as_mut() {
        if d.o1_seal_at().is_some_and(|at| state.pos >= at) {
            d.seal(&g, state.pos);
        }
    }
    if let Some(head) = state.mtp.as_mut() {
        head.rounds += 1;
        head.drafted += k as u64;
        head.accepted += a as u64;
        if prof {
            eprintln!(
                "qwen4-mtp round={} drafted={k} accepted={a} draft={:.1}ms total={:.1}ms accept={:.2}ms gap={:.2}ms (acceptance {:.1}%) ids={:?}{}{} tok={}",
                head.rounds,
                t_draft.as_secs_f64() * 1e3,
                t_round.elapsed().as_secs_f64() * 1e3,
                t_accept.as_secs_f64() * 1e3,
                t_gap.as_secs_f64() * 1e3,
                head.accepted as f64 * 100.0 / head.drafted.max(1) as f64,
                drafts,
                if conf.is_empty() {
                    String::new()
                } else {
                    format!(" conf={:?}", conf.iter().map(|p| (p * 1000.0).round() / 1000.0).collect::<Vec<_>>())
                },
                sfx_note,
                all_ids[next_pos]
            );
        }
    }
    state.round_end = Some(std::time::Instant::now());
    Some(SpecRound {
        accepted: drafts[..a].to_vec(),
        logits,
        drafted: k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[test]
    fn pending_admission_cold_picks_keep_route_order() {
        let slots = [3, u32::MAX, 7, u32::MAX, 11, u32::MAX, 13, u32::MAX];
        assert_eq!(
            cold_picks_from_slots(&slots, 4, 4, &[3, 1, 0, 1]),
            Some(vec![3, 1, 1])
        );
        assert_eq!(cold_picks_from_slots(&slots, 4, 4, &[4]), None);
        assert_eq!(cold_picks_from_slots(&slots, 5, 4, &[0]), None);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn draft_gate_keeps_the_confident_prefix() {
        // off: every draft stays
        assert_eq!(draft_gate_keep(&[0.1, 0.1, 0.1], 0.0), 3);
        // cumulative: 0.9, 0.72, 0.36
        assert_eq!(draft_gate_keep(&[0.9, 0.8, 0.5], 0.5), 2);
        assert_eq!(draft_gate_keep(&[0.9, 0.8, 0.5], 0.3), 3);
        // never fewer than one draft, and a later confident cell does not rescue a cut chain
        assert_eq!(draft_gate_keep(&[0.2, 0.99, 0.99], 0.5), 1);
        assert_eq!(draft_gate_keep(&[0.9, 0.3, 0.99], 0.5), 1);
        assert_eq!(draft_gate_keep(&[f32::NAN, 0.9], 0.5), 1);
        assert_eq!(draft_gate_keep(&[], 0.5), 0);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn copy_frame_grows_with_the_copy_and_stops_at_the_request() {
        assert_eq!(copy_frame_parse("63"), Some((63, 2.0)));
        assert_eq!(copy_frame_parse("31:1"), Some((31, 1.0)));
        assert_eq!(copy_frame_parse("999"), Some((255, 2.0)));
        // fewer drafts than the smallest frame, a factor that is none
        assert_eq!(copy_frame_parse("15"), None);
        assert_eq!(copy_frame_parse("63:0"), None);
        assert_eq!(copy_frame_parse("x"), None);
        // nothing copied yet: a window; two windows of eight held: 32 tokens, then 64,
        // 128, 256
        assert_eq!(copy_frame_drafts(255, 2.0, 0, usize::MAX), None);
        assert_eq!(copy_frame_drafts(255, 2.0, 8, usize::MAX), None);
        assert_eq!(copy_frame_drafts(255, 2.0, 16, usize::MAX), Some(31));
        assert_eq!(copy_frame_drafts(255, 2.0, 48, usize::MAX), Some(63));
        assert_eq!(copy_frame_drafts(255, 2.0, 112, usize::MAX), Some(127));
        assert_eq!(copy_frame_drafts(255, 2.0, 240, usize::MAX), Some(255));
        assert_eq!(copy_frame_drafts(255, 4.0, 8, usize::MAX), Some(31));
        assert_eq!(copy_frame_drafts(63, 1.0, 16, usize::MAX), None);
        assert_eq!(copy_frame_drafts(63, 1.0, 32, usize::MAX), Some(31));
        // the configured most and the request's end bound it
        assert_eq!(copy_frame_drafts(31, 2.0, 500, usize::MAX), Some(31));
        assert_eq!(copy_frame_drafts(63, 2.0, 500, 40), Some(31));
        assert_eq!(copy_frame_drafts(63, 2.0, 500, 30), None);
    }

    #[cfg(feature = "gpu")]
    #[cfg(feature = "cuda")]
    #[test]
    fn copy_cursor_copies_and_learns_insertions() {
        // a plain copy
        let mut ids: Vec<u32> = vec![5, 6, 7, 8, 9, 1, 5];
        let mut cur = CopyCur::default();
        cur.account(&ids);
        cur.start(1, 1);
        assert_eq!(cur.draft(&ids, 4), vec![6, 7, 8, 9]);
        ids.extend([6, 7]);
        cur.account(&ids);
        assert_eq!(cur.at, Some(3));
        assert_eq!(cur.draft(&ids, 3), vec![8, 9, 1]);

        // an insertion before every 4: seen once it is only remembered, seen twice it is drafted
        let p: Vec<u32> = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
        let mut ids = p.clone();
        ids.push(1);
        let mut cur = CopyCur::default();
        cur.account(&ids);
        cur.start(1, 1);
        ids.extend([2, 3, 100, 101, 4, 1, 2, 3]);
        cur.account(&ids);
        assert_eq!(cur.rules.get(&(3, 4)), Some(&(vec![100, 101], 1)));
        assert_eq!(cur.draft(&ids, 3), vec![4, 1, 2]);
        // the second time the insertion comes: inside it the rest of it is drafted, then the copy goes on
        ids.push(100);
        cur.account(&ids);
        assert_eq!(cur.draft(&ids, 3), vec![101, 4, 1]);
        ids.extend([101, 4, 1, 2, 3]);
        cur.account(&ids);
        assert_eq!(cur.rules.get(&(3, 4)).map(|r| r.1), Some(2));
        assert_eq!(&cur.draft(&ids, 4)[..3], &[100, 101, 4]);
    }

    #[test]
    fn suffix_draft_continues_the_latest_longest_occurrence() {
        // the tail [1, 2, 3] occurred at 0..3; 9, 8 followed it
        assert_eq!(suffix_draft(&[1, 2, 3, 9, 8, 7, 1, 2, 3], 3, 2), (3, vec![9, 8]));
        // too short a match, an empty context, nothing asked for
        assert_eq!(suffix_draft(&[1, 2, 3, 9, 8, 7, 1, 2, 3], 4, 2), (0, vec![]));
        assert_eq!(suffix_draft(&[], 1, 2), (0, vec![]));
        assert_eq!(suffix_draft(&[1, 2, 1, 2], 2, 0), (0, vec![]));
        // the longer of two occurrences wins over the later one
        assert_eq!(suffix_draft(&[5, 1, 2, 3, 4, 0, 2, 3, 6, 0, 1, 2, 3], 2, 1), (3, vec![4]));
        // equal length: the later one
        assert_eq!(suffix_draft(&[2, 3, 4, 0, 2, 3, 6, 0, 2, 3], 2, 1), (3, vec![6]));
        // a period shorter than the draft runs through its own prediction
        assert_eq!(suffix_draft(&[7, 1, 2, 1, 2, 1, 2], 2, 5), (4, vec![1, 2, 1, 2, 1]));
        assert_eq!(suffix_draft(&[4, 4, 4], 1, 3), (2, vec![4, 4, 4]));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn suffix_cfg_parses_and_clamps_to_the_frame() {
        let max = crate::gpu_wgpu::qwen4::TMAX - 1;
        assert_eq!(suffix_parse("6:12:5"), Some(SuffixCfg { ext: 6, strong: 12, max: 5 }));
        assert_eq!(suffix_parse("6"), Some(SuffixCfg { ext: 6, strong: 0, max }));
        assert_eq!(suffix_parse("0:8:99"), Some(SuffixCfg { ext: 0, strong: 8, max }));
        assert_eq!(suffix_parse("0:0"), None);
        assert_eq!(suffix_parse("x"), None);
        assert_eq!(suffix_parse(""), None);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn mtp_k_parses_and_clamps_to_the_frame() {
        assert_eq!(mtp_k(None), (3, false));
        for k in 1..=7 {
            assert_eq!(mtp_k(Some(&k.to_string())), (k, false));
        }
        assert_eq!(mtp_k(Some("0")), (1, true));
        assert_eq!(mtp_k(Some("8")), (7, true));
        assert_eq!(mtp_k(Some("four")), (3, true));
        // every accepted k leaves a window that fits one frame
        assert!(mtp_k(Some("99")).0 < crate::gpu_wgpu::qwen4::TMAX);
    }

    #[test]
    fn dynamic_pool_percent_is_bounded_per_policy() {
        // GLM's 40% default leaves a physical-VRAM reserve for static
        // attention/control tensors and transient cold-expert staging; the
        // operator may tune it, but never outside the policy envelope.
        assert_eq!(bounded_pool_pct(None, 40, 20, 60), 40);
        assert_eq!(bounded_pool_pct(Some("55"), 40, 20, 40), 40);
        assert_eq!(bounded_pool_pct(Some("999"), 40, 20, 40), 40);
        assert_eq!(bounded_pool_pct(Some("0"), 40, 20, 40), 20);
        assert_eq!(bounded_pool_pct(Some("bad"), 40, 20, 40), 40);
        // Qwen retains its established wider range independently.
        assert_eq!(bounded_pool_pct(Some("90"), 75, 25, 85), 85);
        assert_eq!(bounded_pool_slots(Some("9999"), 752, true), 752);
        assert_eq!(bounded_pool_slots(Some("9999"), 752, false), 9999);
    }

    #[test]
    fn active_tail_floor_uses_only_reachable_layers() {
        // The MiniCPM 35B graph leaves 13 of 40 MoE layers in its CPU tail.
        // A 512-slot arena therefore protects 39 owners per active layer,
        // not the generic 12-owner all-layer floor.
        assert_eq!(active_layer_floor(512, 40), 12);
        assert_eq!(active_layer_floor(512, 13), 39);
        assert!(active_layer_floor(512, 13) * 13 <= 512);
        // Never over-promise protected residency on an intentionally tiny
        // operator override.
        assert_eq!(active_layer_floor(16, 13), 1);
        assert_eq!(active_layer_floor(12, 13), 0);
    }

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
