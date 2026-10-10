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
    idx: Option<LayerIdx>,
}

pub struct Globals {
    embed: QTensor,
    lm_head: QTensor,
    head_hc: GatedResidual,
    lm_head_idx: Option<usize>,
    /// The embedding table, re-read on the card by the MTP draft chain.
    embed_idx: Option<usize>,
    /// Every skeleton matrix the device frame reads, for the one-time
    /// upload-and-pin before the expert arena takes the rest of the card.
    skeleton_idxs: Vec<usize>,
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
    #[cfg(feature = "gpu")]
    profile: Option<ExpertProfile>,
    /// The next device forward is a verify window: snapshot the recurrent
    /// state per token and return every token's logits in `window_logits`.
    verify_window: bool,
    window_logits: Vec<Vec<f32>>,
    #[cfg(feature = "gpu")]
    mtp: Option<MtpHead>,
    mtp_tried: bool,
}

/// Summed wall time and count of expert uploads (every thread), for the
/// per-token profile line: against the admission wall time it says whether
/// the uploads overlap.
pub(crate) static FILL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static FILL_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
    /// The host tiers behind this arena (RAM tier, then the file), set by
    /// the device path; None reads experts straight from the memory map.
    pub(crate) store: Option<Arc<crate::expert_store::ExpertStore>>,
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
            #[cfg(feature = "gpu")]
            profile: None,
            verify_window: false,
            window_logits: Vec::new(),
            #[cfg(feature = "gpu")]
            mtp: None,
            mtp_tried: false,
        }
    }

    fn reset(&mut self) {
        self.hyper.clear();
        self.token_history.clear();
        self.pos = 0;
        self.host_pos = 0;
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
            gu_q2,
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
            gu_q2,
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
            gu_q2,
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

    /// A free slot, else the least recently used slot of a layer above its
    /// working-set floor, else one of this layer's own, never a shared slot
    /// or one of `protect` (this layer's current winners).
    fn victim(&mut self, layer: usize, protect: &[usize]) -> Option<usize> {
        if let Some(s) = self.free.pop() {
            return Some(s);
        }
        let hold = self.hold_layers;
        let eligible = |owner: (usize, usize)| {
            owner.1 != usize::MAX
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
        let stored = self.store.as_ref().and_then(|store| {
            store.with_expert(layer, expert, true, |parts| {
                crate::gpu_wgpu::qwen4::upload_expert_parts(
                    self.stager.as_ref(),
                    model,
                    slot,
                    parts,
                )
            })
        });
        let ok = match stored {
            Some(ok) => ok,
            None => {
                if let Some(st) = self.stager.as_ref()
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

    /// An arena slot gave up `old`: the RAM tier wants it back.
    fn dropped(&self, old: (usize, usize)) {
        if old.1 != usize::MAX
            && let Some(store) = self.store.as_ref()
        {
            store.note_vram(old.0, old.1, false);
        }
    }

    /// Submit what the staging ring holds ahead of the next frame.
    #[cfg(feature = "gpu")]
    pub(crate) fn flush_uploads(&mut self) {
        if let Some(st) = self.stager.as_mut() {
            st.flush();
        }
    }

    /// The staged uploads as a command buffer to submit ahead of a frame
    /// in the same queue submission; `rearm_uploads` follows the submit.
    pub(crate) fn take_uploads(&mut self) -> Option<wgpu::CommandBuffer> {
        self.stager.as_mut().and_then(|st| st.take())
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
        let me = &*self;
        let ok: Vec<bool> = if jobs.len() <= 1 {
            jobs.iter()
                .map(|&(slot, e)| me.fill_slot(model, slot, layer, e, triples[e]))
                .collect()
        } else {
            std::thread::scope(|scope| {
                let hs: Vec<_> = jobs
                    .iter()
                    .map(|&(slot, e)| {
                        let model = model.clone();
                        let triple = triples[e];
                        scope.spawn(move || me.fill_slot(&model, slot, layer, e, triple))
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap_or(false)).collect()
            })
        };
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
        let me = &*self;
        let results: Vec<bool> = if uploads.len() <= 1 {
            uploads
                .iter()
                .map(|&(slot, e)| me.fill_slot(model, slot, layer, e, triples[e]))
                .collect()
        } else if pooled {
            const ADMIT_THREADS: usize = 32;
            let nth = uploads.len().min(ADMIT_THREADS);
            let per = uploads.len().div_ceil(nth);
            std::thread::scope(|scope| {
                let handles: Vec<_> = uploads
                    .chunks(per)
                    .map(|part| {
                        let model = model.clone();
                        let h = scope.spawn(move || {
                            part.iter()
                                .map(|&(slot, e)| me.fill_slot(&model, slot, layer, e, triples[e]))
                                .collect::<Vec<bool>>()
                        });
                        (part.len(), h)
                    })
                    .collect();
                // a panicked thread fails its whole part, keeping the
                // results aligned with `uploads`
                handles
                    .into_iter()
                    .flat_map(|(n, h)| h.join().unwrap_or_else(|_| vec![false; n]))
                    .collect()
            })
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = uploads
                    .iter()
                    .map(|&(slot, e)| {
                        let model = model.clone();
                        let triple = triples[e];
                        scope.spawn(move || me.fill_slot(&model, slot, layer, e, triple))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or(false))
                    .collect()
            })
        };
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
            gu_q2,
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
            gu_q2,
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
        gu_q2: bool,
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
        let (capacity, segment_slots) = if pct_env == "CMF_DSV41_POOL_PCT" {
            crate::gpu_wgpu::dsv4_global_moe_create_for_dsv41(
                model, requested, inter, hidden, gu_q2,
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
                if gu_q2 { "q2tp" } else { "q4tp" }
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
            store: None,
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
        // phase 2: the uploads, in parallel (they were one after another:
        // at a 12 GB budget that was the larger half of every token)
        let me = &*self;
        let ok: Vec<bool> = if plan.len() <= 1 {
            plan.iter()
                .map(|&(slot, e)| me.fill_slot(model, slot, layer, e, triples[e]))
                .collect()
        } else {
            std::thread::scope(|scope| {
                let hs: Vec<_> = plan
                    .iter()
                    .map(|&(slot, e)| {
                        let model = model.clone();
                        let triple = triples[e];
                        scope.spawn(move || me.fill_slot(&model, slot, layer, e, triple))
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap_or(false)).collect()
            })
        };
        let mut failed = false;
        for (&(slot, e), ok) in plan.iter().zip(&ok) {
            if !ok {
                self.owner[slot] = None;
                self.slot_for[base + e] = u32::MAX;
                self.occupancy[layer] = self.occupancy[layer].saturating_sub(1);
                self.free.push(slot);
                failed = true;
            }
        }
        if failed || triples.len() != self.n_experts {
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
        if side != model.path && side.exists() {
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

fn observe(label: &str, li: usize, position: usize, values: &[f32]) {
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
    let mut out: Vec<Vec<Vec<f32>>> = Vec::with_capacity(layers.len());
    // (layer, token, head, row id), and the destination of each row
    let mut jobs: Vec<(usize, usize, usize, i64)> = Vec::new();
    let mut hist: Vec<u32> = history.to_vec();
    for (li, l) in layers.iter().enumerate() {
        let Some(pw) = l.ple.as_ref() else {
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
    std::env::var("CMF_QWEN_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(crate::gpu_wgpu::qwen4::TMAX)
        .clamp(1, crate::gpu_wgpu::qwen4::TMAX)
}

#[cfg(not(feature = "gpu"))]
pub fn prefill_chunk() -> usize {
    1
}

/// Several consecutive tokens (`ids` at `pos0..`), logits of the last one
/// when `want_logits`. The device path runs them in one frame per layer;
/// otherwise each token takes the host path.
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
    #[cfg(feature = "gpu")]
    if !ids.is_empty()
        && ids.len() <= crate::gpu_wgpu::qwen4::TMAX
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
    if ntok == 0 || ntok > q4::TMAX {
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
        let Some(dev) = q4::Dev::new(model.uid(), &g, &kinds) else {
            return off(state, "device state allocation refused");
        };
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
                && let Err(why) = head.setup(&g, gu_q2)
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
        let slots = (free / per as u64) as usize;
        let n_layers = layers.len();
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
        if std::env::var_os("CMF_QWEN_PROFILE_SAVE").is_some() && profile.is_none() {
            profile = Some(ExpertProfile::new(layers.len(), n_experts));
        }
        state.profile = profile;
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
    let dev = state.dev.as_mut().unwrap();
    let arena = state.gpu_pool.as_mut().unwrap();

    // ── the tokens ──
    let mut pf = DevProf::default();
    let cold_on_device = std::env::var("CMF_QWEN_COLD_DEVICE").as_deref() != Ok("0");
    // Several tokens per frame rule the gated chain out (one frame per layer).
    let chain_len = if ntok > 1 {
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
    let frame_bytes = cold_part + (ntok * hidden_bytes).div_ceil(16) * 16;
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
        (true, false) => 1,
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
    dev.gated = chain_len > 1;
    let mut fi = 0usize;
    let mut vocab_out = 0usize;
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
                    let first = if snapshot { 0 } else { ntok - 1 };
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
        // staged admissions land before the frame that reads their slots:
        // their copies go first in the frame's own queue submission
        let uploads = arena.take_uploads();
        let Some(pend) = q4::submit_frame(dev, uploads, cb, stage_off) else {
            failed = Some("submit");
            break;
        };
        arena.rearm_uploads();
        if let Some(tf) = t_fence.take() {
            pf.post += tf.elapsed();
        }
        // one frame per chain: the next frame is encoded while this one runs
        arena.hold_layers = [None, None];
        if chain_len == 1 && hi < frames_total {
            let te = std::time::Instant::now();
            match encode_chain(
                hi,
                hi + 1,
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
                    cold_vec[t] = Some(crate::pipeline::moe_cold_experts_cpu(&jobs, &x2, pool));
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
                        state.window_logits = (0..ntok).map(|t| read(lo + t * stride)).collect();
                        *logits = state.window_logits.last().cloned().unwrap_or_default();
                    } else {
                        *logits = read(lo);
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
    }
    if let Some(d) = state.dev.as_mut() {
        d.pos = pos0 + ntok;
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
        eprintln!(
            "qwen4-device pos={position} ntok={ntok} total={:.1}ms encode={:.1}ms finish={:.1}ms wait={:.1}ms spin={:.1}ms post={:.1}ms ple={:.1}ms cold_cpu={:.1}ms admit={:.1}ms chains={} aborted_frames={} cold={} experts ({} on device) in {} layers fills={fill_n} fill_sum={:.1}ms {store_line}",
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

#[cfg(feature = "gpu")]
impl MtpHead {
    /// Open the sidecar beside the main file, when there is one that fits.
    pub fn load(main: &Arc<CmfModel>, arch: &ModelArch) -> Option<Self> {
        if std::env::var("CMF_QWEN_MTP").as_deref() == Ok("0") {
            return None;
        }
        let path = cortiq_core::mtp_sidecar_path(&main.path);
        if path == main.path || !path.exists() {
            return None;
        }
        let side = match CmfModel::open(&path) {
            Ok(m) => Arc::new(m),
            Err(e) => {
                tracing::warn!("qwen4 MTP sidecar {}: {e}", path.display());
                return None;
            }
        };
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
    // ── drafts ──
    let (k, drafts) = {
        let head = state.mtp.as_mut()?;
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
        let tok_in = all_ids[next_pos];
        // ── the whole chain in one submit: each cell's argmax is re-embedded
        // on the card for the next; the host reads k ids, not k×vocab logits ──
        // opt-in: it pins the 0.6 GB embedding table, which costs the arena
        // more than the host round trips cost the round
        let device_chain = std::env::var("CMF_QWEN_MTP_DEVICE_DRAFT").as_deref() == Ok("1");
        let mut chained = false;
        if device_chain
            && let Some(embed_idx) = globals.embed_idx
            && let Some(ids_buf) = q4::draft_ids(tok_in)
        {
            let mut run = || -> Option<Vec<u32>> {
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
                    )?;
                    q4::encode_argmax(&mut enc, mdev, &lb, vocab, &ids_buf, j + 1)?;
                }
                if !q4::copy_to_stage(&mut enc, mdev, &ids_buf, 4, 0, (k * 4) as u64) {
                    return None;
                }
                drop(merge);
                let bytes = q4::submit_chain(mdev, enc, (k * 4) as u64)?.wait()?;
                Some(
                    bytes[..k * 4]
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect(),
                )
            };
            if let Some(d) = run() {
                drafts = d;
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
        (k, drafts)
    };
    let t_draft = t_round.elapsed();
    // ── verify: the window through the main model, every token's logits ──
    let mut ids = Vec::with_capacity(k + 1);
    ids.push(all_ids[next_pos]);
    ids.extend_from_slice(&drafts);
    state.verify_window = true;
    let mut last_logits = Vec::new();
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
    if !ok || state.window_logits.len() != k + 1 {
        return None;
    }
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| {
                if x > a.1 { (i, x) } else { a }
            })
            .0
    };
    let mut a = 0usize;
    while a < k && argmax(&state.window_logits[a]) == drafts[a] as usize {
        a += 1;
    }
    let logits = std::mem::take(&mut state.window_logits).swap_remove(a);
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
    if let Some(head) = state.mtp.as_mut() {
        head.rounds += 1;
        head.drafted += k as u64;
        head.accepted += a as u64;
        if prof {
            eprintln!(
                "qwen4-mtp round={} drafted={k} accepted={a} draft={:.1}ms total={:.1}ms (acceptance {:.1}%) ids={:?}",
                head.rounds,
                t_draft.as_secs_f64() * 1e3,
                t_round.elapsed().as_secs_f64() * 1e3,
                head.accepted as f64 * 100.0 / head.drafted.max(1) as f64,
                drafts
            );
        }
    }
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
