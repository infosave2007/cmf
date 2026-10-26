//! Exact opt-in dynamic MoE cache for the CPU tail of MiniCPM-V 4.7.
//!
//! A MiniCPM-V-4.7 q4tp checkpoint can leave its attention/GDN prefix on a
//! WGPU device while the final layers deliberately execute on the host.  Letting
//! the ordinary per-op path re-enter WGPU for those layers streams a whole MoE
//! panel through the generic residency cache.  This controller instead keeps a
//! bounded, model-wide bank of *expert triples*: routed experts that are already
//! resident use the fused device frame, and cold routed experts are completed on
//! the CPU in parallel.  The route is intentionally supplied by the caller;
//! this module never predicts, substitutes, or reorders a MiniCPM route.
//!
//! It is deliberately narrower than the generic Qwen cache.  In particular it
//! accepts only the stock MiniCPM-V-4.7 q4tp top-8 softmax MoE with its shared
//! expert.  Every refusal returns `None`, so the caller executes its existing
//! exact CPU expert loop unchanged.

use crate::pipeline::{MoeFfn, MoeRoute};
use crate::pool::Pool;

#[cfg(feature = "gpu")]
use std::sync::{Arc, Mutex, Once, OnceLock};

/// Per-pipeline handle for the MiniCPM CPU-tail expert bank.
///
/// The handle intentionally owns no residency metadata.  WGPU's physical
/// DSV4 bank is process-global for a `(model uid, device)` pair, so a
/// per-pipeline slot map could point at weights another pipeline just
/// replaced.  [`SharedTailBank`] keeps the physical bank and its host-side LRU
/// map together instead.
///
/// Despite the historical name the bank serves two checked topologies:
/// MiniCPM-V-4.7's q4tp top-8 tail and MiniMax-M3.1's q2tp-mix (or q4tp)
/// top-4 tail, each behind its own env opt-in.
#[derive(Default)]
pub(crate) struct MiniCpmTailCache;

/// Short spelling for pipeline state declarations.
pub(crate) type MiniCpmTail = MiniCpmTailCache;

#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PoolKey {
    model_uid: u64,
    hidden: usize,
    inter: usize,
    experts: usize,
    layers: usize,
    /// Bank codec: gate/up are q2tp against a q4tp down (MiniMax q2tp-mix)
    /// when true; all-q4tp (MiniCPM, MiniMax q4tp) when false.
    gu_q2: bool,
    /// MiniMax-M3.1: the bank is sized from the FREE weight budget (its
    /// device prefix already claims most of the card) and reaches only its
    /// own top-4 prepared frame.
    minimax: bool,
}

/// The 35B-A3B graph leaves the final 13 of its 40 MoE layers to the host.
/// They are the only layers that can reach this bank, so its residency floor
/// must not be divided over the inactive device prefix.  Keep this exact
/// topology guard rather than applying a 35B cache policy to a future
/// `minicpmv4_7` variant with another layer layout.
#[cfg(feature = "gpu")]
const MINICPM_35B_TOTAL_LAYERS: usize = 40;
#[cfg(feature = "gpu")]
const MINICPM_35B_ACTIVE_TAIL_LAYERS: usize = 13;

#[cfg(feature = "gpu")]
fn active_tail_layers(total_layers: usize) -> Option<usize> {
    (total_layers == MINICPM_35B_TOTAL_LAYERS).then_some(MINICPM_35B_ACTIVE_TAIL_LAYERS)
}

#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TailBankKey {
    model_uid: u64,
    /// `gpu::current_device()` selects a separate WGPU context (and hence a
    /// separate physical DSV4 bank) for each adapter.
    device: usize,
}

#[cfg(feature = "gpu")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct LayerLayout {
    expert_ids: Arc<[(usize, usize, usize)]>,
    shared_ids: (usize, usize, usize),
}

/// One host LRU/map for exactly one physical WGPU DSV4 bank.  Its mutex stays
/// held from admission through the frame readback, so no other pipeline can
/// reuse a flat slot while that frame still names it.
#[cfg(feature = "gpu")]
struct SharedTailBank {
    gpu_pool: crate::qwen4_exp::QwenGpuPool,
    pool_key: PoolKey,
    /// Exact directory triples indexed by logical `(layer, expert)`.  A
    /// differing overlay resets this *shared* map under its mutex, so no stale
    /// logical slot survives to name newly overwritten physical bytes.
    layouts: Vec<Option<LayerLayout>>,
}

#[cfg(feature = "gpu")]
impl SharedTailBank {
    /// Record a layout, rebuilding the shared logical map when an overlay
    /// changes a layer's backing directory triples.  The global WGPU buffers
    /// are retained: `create_explicit` observes the already-created matching
    /// bank and creates fresh host-only LRU metadata for it.
    fn accepts_layout_or_resets(
        &mut self,
        model: &Arc<cortiq_core::CmfModel>,
        key: PoolKey,
        li: usize,
        candidate: LayerLayout,
    ) -> bool {
        if self.pool_key != key {
            return false;
        }
        let Some(slot) = self.layouts.get_mut(li) else {
            return false;
        };
        let same = match slot {
            Some(existing) => existing == &candidate,
            None => {
                *slot = Some(candidate);
                return true;
            }
        };
        if same {
            return true;
        }

        // Build all replacement state before modifying the live map.  A
        // failed WGPU allocation/admission leaves the old bank untouched and
        // the caller falls back to its exact CPU implementation.
        let Some(gpu_pool) = create_tail_gpu_pool(model, key) else {
            return false;
        };
        let mut layouts: Vec<Option<LayerLayout>> = (0..key.layers).map(|_| None).collect();
        layouts[li] = Some(candidate);
        self.gpu_pool = gpu_pool;
        self.layouts = layouts;
        true
    }
}

/// Build the generic physical DSV4 bank, then apply only the policy that is
/// justified by MiniCPM's known graph/CPU split.  The helper is shared by
/// initial allocation and overlay reset so a rebuilt host map cannot silently
/// fall back to the 40-layer generic floor.
#[cfg(feature = "gpu")]
fn create_tail_gpu_pool(
    model: &Arc<cortiq_core::CmfModel>,
    key: PoolKey,
) -> Option<crate::qwen4_exp::QwenGpuPool> {
    let slots = if key.minimax {
        minimax_tail_slots(&key)?
    } else {
        bounded_slots()
    };
    let mut gpu_pool = crate::qwen4_exp::QwenGpuPool::create_explicit(
        model,
        key.inter,
        key.hidden,
        key.layers,
        key.experts,
        key.gu_q2,
        slots,
        0,
    )?;
    if let Some(active_layers) = active_tail_layers(key.layers) {
        // At 512 slots this changes the protected floor from 12 (512 / 40)
        // to 39 (512 / 13). Shared experts count in both occupancies, so the
        // 13 floors consume at most 507 slots and leave five elastic slots.
        gpu_pool.configure_minicpm_tail(active_layers);
    }
    Some(gpu_pool)
}

/// MiniMax's arena comes out of the FREE weight budget, not a percentage of
/// the card: its device prefix has already claimed most of it (measured:
/// 80.4 GiB resident of a 91.6 GiB budget on the RTX PRO 6000), and the KV,
/// staging and driver overhead live out of the remainder.  `CMF_MINIMAX_TAIL_SLOTS`
/// overrides the estimate; a bank too small to hold one layer's route is
/// refused rather than allowed to churn on every token.
#[cfg(feature = "gpu")]
fn minimax_tail_slots(key: &PoolKey) -> Option<usize> {
    if let Some(v) = std::env::var("CMF_MINIMAX_TAIL_SLOTS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
    {
        return (v >= 64).then_some(v.min(4096));
    }
    let codec = crate::gpu_wgpu::MoeBankCodec::from_gu_q2(key.gu_q2);
    let (gu, dn) = codec.lens(key.inter, key.hidden)?;
    let per = (2usize.checked_mul(gu)? + dn) as u64;
    let budget = crate::gpu_wgpu::device_vram_budget();
    let resident = crate::gpu_wgpu::resident_bytes();
    let free = budget.saturating_sub(resident);
    // 55 % of the unclaimed budget: enough slots to cover a hot working set
    // across the tail while KV, the staging ring and the pipeline cache keep
    // their headroom. Below one layer's route (shared + top-4) the bank has
    // no protected floor left, so refuse and stay on the exact CPU path.
    let slots = ((free.saturating_mul(55) / 100) / per.max(1)) as usize;
    (slots >= 64).then_some(slots.min(4096))
}

#[cfg(feature = "gpu")]
type SharedTailBankRef = Arc<Mutex<SharedTailBank>>;

/// One logical registry entry per physical WGPU bank.  The WGPU context map
/// uses `current_device()` as its key too; omitting it here would let a
/// gpu-plan pipeline reuse adapter 0's host slots while dispatching on adapter
/// 1.
#[cfg(feature = "gpu")]
fn tail_banks() -> &'static Mutex<Vec<(TailBankKey, SharedTailBankRef)>> {
    static BANKS: OnceLock<Mutex<Vec<(TailBankKey, SharedTailBankRef)>>> = OnceLock::new();
    BANKS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Release process-global tail pools before WGPU contexts are dropped.  A
/// `QwenGpuPool` can own admission workers, so retaining the registry beyond
/// the WGPU lifetime would both leak those workers and leave stale logical
/// slots if a host deliberately reinitializes its GPU runtime.
#[cfg(feature = "gpu")]
pub(crate) fn shutdown_banks() {
    let Ok(mut registry) = tail_banks().lock() else {
        return;
    };
    let banks = std::mem::take(&mut *registry);
    drop(registry);
    drop(banks);
}

/// Forget host-side slot maps for one model whenever its model-owned WGPU
/// buffers are evicted.  The physical global MoE pool is keyed by model UID,
/// but an image-stage cache release may remove and later recreate that pool
/// without tearing down the device.  Keeping the old logical remap would
/// otherwise let a new pool be addressed as if it still contained the old
/// experts.  Dropping just this registry entry makes the next tail call build
/// a fresh map and refill only through its checked admission path.
#[cfg(feature = "gpu")]
pub(crate) fn release_model_bank(model_uid: u64) {
    let Ok(mut registry) = tail_banks().lock() else {
        return;
    };
    let mut retired = Vec::new();
    registry.retain(|(key, bank)| {
        if key.model_uid == model_uid {
            retired.push(bank.clone());
            false
        } else {
            true
        }
    });
    // A pool may own admission workers; do not join them under the global
    // registry mutex.
    drop(registry);
    drop(retired);
}

/// Get or initialize the single shared host map that corresponds to this
/// model's global DSV4 buffers on the current adapter.  A geometry mismatch is
/// never reinterpreted: the caller receives `None` and uses its exact CPU MoE
/// path instead.
#[cfg(feature = "gpu")]
fn acquire_tail_bank(
    model: &Arc<cortiq_core::CmfModel>,
    pool_key: PoolKey,
) -> Option<SharedTailBankRef> {
    let registry_key = TailBankKey {
        model_uid: pool_key.model_uid,
        device: crate::gpu::current_device(),
    };
    let mut registry = tail_banks().lock().ok()?;
    if let Some((_, bank)) = registry.iter().find(|(key, _)| *key == registry_key) {
        return Some(bank.clone());
    }

    let gpu_pool = create_tail_gpu_pool(model, pool_key)?;
    let bank = Arc::new(Mutex::new(SharedTailBank {
        gpu_pool,
        pool_key,
        layouts: (0..pool_key.layers).map(|_| None).collect(),
    }));
    registry.push((registry_key, bank.clone()));
    Some(bank)
}

/// The DSV4 frame intentionally consumes float activations so its resident
/// and cold branches agree.  It must therefore not replace the normal CPU
/// q4tp A8W8 path: that would be a precision-mode change rather than an
/// optimization, and can change greedy tokens.  The caller must explicitly
/// choose the exact CPU contract (`CMF_SDOT=0`) before opting into this tail.
#[cfg(feature = "gpu")]
fn dynamic_tail_precision_safe() -> bool {
    if std::env::var("CMF_MINICPM_DYNAMIC_TAIL").as_deref() != Ok("1") {
        return false;
    }
    if !crate::qtensor::a8w8_enabled() {
        return true;
    }
    static WARN: Once = Once::new();
    WARN.call_once(|| {
        tracing::warn!(
            "MiniCPM dynamic tail disabled: its float-activation DSV4 frame cannot replace the \
             normal A8W8 CPU q4tp path without changing precision; use CMF_SDOT=0 together with \
             CMF_MINICPM_DYNAMIC_TAIL=1 to opt into the parity-checked exact mode"
        );
    });
    false
}

/// MiniMax's opt-in dynamic tail debug: one line per decline reason so a
/// silent `None` back to the CPU path is attributable without a rebuild.
#[cfg(feature = "gpu")]
fn m3_tail_debug(msg: &str) {
    if std::env::var_os("CMF_M3_TAIL_DEBUG").is_some() {
        eprintln!("m3-tail decline: {msg}");
    }
}

/// MiniMax-M3.1's opt-in dynamic tail.  The prepared frame is f32-exact
/// between quantized matvecs while the plain host fallback may run the
/// engine's A8W8 activation approximation, so the switch is explicit — an
/// operator comparing tok/s across arms must be comparing a stable route.
#[cfg(feature = "gpu")]
fn minimax_tail_enabled() -> bool {
    std::env::var("CMF_MINIMAX_DYNAMIC_TAIL").as_deref() == Ok("1")
}

/// The prepared frame is the normal dynamic-tail arm: the host has already
/// formed the exact route, knows which picks are cold, and owns their input.
/// Keep a one-variable legacy arm for an honest same-binary A/B, and retain
/// the generic frame whenever its route diagnostic was explicitly requested.
#[cfg(feature = "gpu")]
fn prepared_frame_on() -> bool {
    std::env::var("CMF_MINICPM_PREPARED_FRAME").as_deref() != Ok("0")
        && std::env::var_os("CMF_DSV4_MOE_CHECK").is_none()
}

impl MiniCpmTailCache {
    /// True only for an explicitly requested MiniCPM-V-4.7 or MiniMax-M3.1
    /// pipeline.  Keep this inexpensive outer discriminator separate from
    /// `forward`: it lets the generic MoE match fall through to MiMo/ordinary
    /// executors for every other architecture rather than taking a tail arm
    /// and refusing late.
    #[cfg(feature = "gpu")]
    pub(crate) fn is_requested_target(m: &MoeFfn) -> bool {
        let Some(model) = m
            .experts
            .first()
            .and_then(|expert| expert.gate_proj.model_arc())
        else {
            return false;
        };
        match model.arch().arch_name.as_str() {
            "minicpmv4_7" => dynamic_tail_precision_safe(),
            "minimax_m3_vl" => {
                let on = minimax_tail_enabled();
                m3_tail_debug(&format!("is_requested_target minimax → {on}"));
                on
            }
            other => {
                m3_tail_debug(&format!("arch not a tail target: {other}"));
                false
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub(crate) fn is_requested_target(_m: &MoeFfn) -> bool {
        false
    }

    /// Execute an already-resolved MiniCPM route through the bounded dynamic
    /// bank.  `None` means *do not use this result*; callers must fall back to
    /// their normal exact CPU `moe_ffn_experts` path.
    ///
    /// This has no `gpu::enabled_here()` guard on purpose.  The production
    /// caller enters `gpu::cpu_scope()` for the tail so ordinary projections
    /// stay on the CPU, while this one explicitly-owned fused WGPU frame may
    /// still service the resident expert subset.
    #[cfg(feature = "gpu")]
    pub(crate) fn forward(
        &mut self,
        li: usize,
        m: &MoeFfn,
        x: &[f32],
        route: &MoeRoute,
        pool: Option<&Pool>,
        total_layers: usize,
    ) -> Option<Vec<f32>> {
        if !Self::is_requested_target(m)
            || !crate::gpu_wgpu::dsv4_global_moe_supported()
            || li >= total_layers
            || x.is_empty()
            || m.route_tau.is_some()
            || m.mask.is_some()
            || m.per_expert_scale.is_some()
            || m.router_input_norm
            || m.resonance.is_some()
            || !m.grown.is_empty()
        {
            m3_tail_debug(&format!(
                "outer gate: target={} supported={} li={li}/{total_layers} empty={} tau={} mask={} pscale={} rin={} res={} grown={} top_k={} sigmoid={} normtopk={} scaling={}",
                Self::is_requested_target(m),
                crate::gpu_wgpu::dsv4_global_moe_supported(),
                x.is_empty(),
                m.route_tau.is_some(),
                m.mask.is_some(),
                m.per_expert_scale.is_some(),
                m.router_input_norm,
                m.resonance.is_some(),
                !m.grown.is_empty(),
                m.top_k,
                m.router_sigmoid,
                m.norm_topk_prob,
                m.routed_scaling,
            ));
            return None;
        }

        let first = m.experts.first()?;
        let model = first.gate_proj.model_arc()?;
        // Per-architecture routing contract.  The prepared frame never
        // re-ranks, so what matters is the shape that formed `route` and the
        // semantics the host weights already carry.
        let minimax = match model.arch().arch_name.as_str() {
            "minicpmv4_7" => {
                if m.top_k != 8
                    || route.idx.len() != 8
                    || m.router_sigmoid
                    || !m.norm_topk_prob
                    || (m.routed_scaling - 1.0).abs() > 1e-9
                    // MiniCPM has no selection bias; keep its exact gate.
                    || m.expert_bias.is_some()
                {
                    return None;
                }
                false
            }
            "minimax_m3_vl" => {
                // MiniMax carries `mlp.expert_bias` (added to the top-k
                // CHOICE; gathered weights stay unbiased) — the host route
                // above already applied it, and the prepared frame never
                // re-ranks, so the bias is consumed before this point.
                if m.top_k != 4
                    || route.idx.len() != 4
                    || !m.router_sigmoid
                    || !m.norm_topk_prob
                {
                    return None;
                }
                true
            }
            _ => return None,
        };
        if std::env::var_os("CMF_M3_TAIL_DEBUG").is_some() {
            eprintln!(
                "m3-tail: arch={} minimax={minimax} top_k={} idx={} act={:?}",
                model.arch().arch_name,
                m.top_k,
                route.idx.len(),
                first.gate_proj.model_dtype()
            );
        }
        // MiniMax's frame exists only in the prepared arm: the legacy arm's
        // device routing re-derives the route through gu uniforms that carry
        // no activation code for this architecture.  Opting out of the
        // prepared frame therefore means opting out of the whole tail bank.
        if minimax && !prepared_frame_on() {
            return None;
        }
        let hidden = x.len();
        let inter = first.gate_proj.rows();
        if hidden % 32 != 0 || inter == 0 || inter % 32 != 0 {
            return None;
        }

        // Build directory triples from the actual mapped tensors rather than
        // names.  This preserves overlays/defrag layouts and refuses any
        // synthetic, transformed, or mixed-quantization expert before its raw
        // bytes could be bound as the bank's codec.  MiniMax's experts are
        // checked against the OAI activation contract (the gu shader's code 3
        // hardcodes alpha 1.702; `lim` rides the uniform) and against the
        // bank codec chosen from the gate/up dtype.
        let gu_q2 = if minimax {
            match first.gate_proj.model_dtype() {
                Some(cortiq_core::TensorDtype::Q2TiledP) => true,
                Some(cortiq_core::TensorDtype::Q4TiledP) => false,
                _ => return None,
            }
        } else {
            false
        };
        let triple = |expert: &crate::pipeline::DenseFfn| {
            if minimax {
                m3_triple(expert, &model, hidden, inter, gu_q2)
            } else {
                q4tp_triple(expert, &model, hidden, inter)
            }
        };
        let expert_ids: Vec<_> = match m.experts.iter().map(triple).collect::<Option<Vec<_>>>() {
            Some(ids) => ids,
            None => {
                m3_tail_debug("expert triple refused (act/dtype/geometry)");
                return None;
            }
        };
        let Some((shared, shared_gate)) = m.shared.as_ref() else {
            m3_tail_debug("no shared expert");
            return None;
        };
        let Some(shared_ids) = triple(shared) else {
            m3_tail_debug("shared triple refused");
            return None;
        };
        if let Some(gate) = shared_gate
            && (gate.rows() != 1 || gate.cols() != hidden)
        {
            return None;
        }

        let candidate = LayerLayout {
            expert_ids: expert_ids.into(),
            shared_ids,
        };
        if m.router.rows() != candidate.expert_ids.len()
            || m.router.cols() != hidden
            || route.logits.len() != candidate.expert_ids.len()
            || route.p.len() != candidate.expert_ids.len()
            || !route.wsum.is_finite()
            || route.wsum <= 0.0
            || route
                .idx
                .iter()
                .any(|&expert| expert >= candidate.expert_ids.len())
            || route
                .idx
                .iter()
                .enumerate()
                .any(|(i, &expert)| route.idx[..i].contains(&expert))
        {
            return None;
        }

        // `Dsv4MoeW::preweighted` consumes final host weights.  Thus the GPU
        // never re-runs a top-k / softmax reduction and cannot alter the
        // caller's exact route.  Reject NaNs rather than handing a malformed
        // uniform to the shader.
        let mut mix_weights = vec![0.0f32; candidate.expert_ids.len()];
        for &expert in &route.idx {
            let weight = route.p[expert] / route.wsum;
            if !weight.is_finite() {
                return None;
            }
            mix_weights[expert] = weight;
        }
        let shared_weight = shared_gate.as_ref().map_or(1.0, |gate| {
            let mut y = [0.0f32; 1];
            // The controller is normally invoked inside the tail's CPU
            // scope, but keep this tiny gate host-owned even if a future
            // caller forgets that outer guard.  Only the explicit DSV4 frame
            // below is allowed to use WGPU from this CPU-tail executor.
            crate::gpu::cpu_scope(|| gate.matvec(x, &mut y, pool));
            1.0 / (1.0 + (-y[0]).exp())
        });
        if !shared_weight.is_finite() {
            return None;
        }

        let key = PoolKey {
            model_uid: model.uid(),
            hidden,
            inter,
            experts: candidate.expert_ids.len(),
            layers: total_layers,
            gu_q2,
            minimax,
        };
        // The physical DSV4 buffers are global per `(model uid, device)`, so
        // their slot map has to be global on that same key.  Take the shared
        // lock before admitting picks and keep it through the frame readback;
        // otherwise a sibling pipeline could refill a slot named by `remap`.
        let shared = acquire_tail_bank(&model, key)?;
        let mut bank = shared.lock().ok()?;
        if !bank.accepts_layout_or_resets(&model, key, li, candidate) {
            // A geometry mismatch is never reinterpreted.  A compatible
            // directory-layout change was reset atomically above; this is the
            // remaining fail-closed path.
            return None;
        }
        // Retain the immutable directory ids outside the mutex after
        // admission.  An Arc clone is one atomic increment, not a per-token
        // copy of the full expert table, and lets the host cold work overlap
        // with other pipelines once the device frame has read its slots.
        let (expert_ids, shared_ids) = {
            let layout = bank.layouts.get(li)?.as_ref()?;
            (layout.expert_ids.clone(), layout.shared_ids)
        };
        // Split the pool admission at its serial reservation point. A pick
        // with no post-reservation slot is definitely cold if and only if all
        // reserved fills later succeed. The scope below starts exactly that
        // host work before it completes the fills; every failed or inconsistent
        // admission discards the speculative result and takes the existing
        // exact CPU fallback instead.
        let Some(pending) = bank.gpu_pool.begin_ensure(
            &model,
            li,
            &route.idx,
            expert_ids.as_ref(),
            Some(shared_ids),
        ) else {
            // Keep the old "flush even after failed admission" contract: an
            // earlier staged copy must not leak into a later frame.
            bank.gpu_pool.flush_uploads();
            return None;
        };
        let Some(planned_cold_ids) = pending.cold_ids() else {
            // This can only mean an internal slot-map invariant broke. Drop
            // the lease first so its unfilled reservations are cleared.
            drop(pending);
            bank.gpu_pool.flush_uploads();
            return None;
        };
        let planned_cold_jobs: Vec<_> = planned_cold_ids
            .iter()
            .copied()
            .map(|expert| (&m.experts[expert], mix_weights[expert]))
            .collect();
        let geom = crate::gpu_wgpu::Dsv4MoeGeom {
            hidden,
            inter,
            top_k: if minimax { 4 } else { 8 },
            route_scale: 1.0,
            // MiniMax's OAI limit rides the uniform (`lim`); the shader's
            // code 3 branch applies it with alpha 1.702. MiniCPM keeps the
            // historical zero: no clamp, plain SiLU with act code 0.
            swiglu_limit: if minimax { 7.0 } else { 0.0 },
            gu_q2,
            bf16: false,
        };
        let mut out = vec![0.0f32; hidden];
        let prepared = prepared_frame_on();
        let (frame_ok, route_ok, cpu_ok, mut cold_cpu) = std::thread::scope(|scope| {
            let cpu = (!planned_cold_jobs.is_empty()).then(|| {
                // The tail caller's CPU scope is thread-local.  Propagate it
                // into the overlap worker so a cold expert cannot accidentally
                // become a generic per-op GPU upload.
                scope.spawn(|| {
                    crate::qtensor::float_activations_scope(|| {
                        crate::gpu::cpu_scope(|| {
                            crate::pipeline::moe_cold_experts_cpu(&planned_cold_jobs, x, pool)
                        })
                    })
                })
            });
            // The tail's explicit frame is not the generic Qwen frame that
            // normally flushes admissions, so finish and submit the optional
            // staging ring as one consuming lease operation. It flushes even
            // after a failed fill, preventing an earlier staged copy from
            // leaking into a later successful admission.
            let ensured = pending.finish_and_flush();
            let Some((remap, shared_slot, segment_slots)) = ensured else {
                if let Some(job) = cpu
                    && let Ok(mut discarded) = job.join()
                {
                    crate::attention::recycle_buf(&mut discarded);
                }
                return (
                    false,
                    false,
                    false,
                    crate::attention::take_buf(hidden),
                );
            };
            if remap.len() != expert_ids.len() {
                if let Some(job) = cpu
                    && let Ok(mut discarded) = job.join()
                {
                    crate::attention::recycle_buf(&mut discarded);
                }
                return (
                    false,
                    false,
                    false,
                    crate::attention::take_buf(hidden),
                );
            }

            let cold_ids: Vec<_> = route
                .idx
                .iter()
                .copied()
                .filter(|&expert| remap[expert] == u32::MAX)
                .collect();
            // A completed successful admission must preserve the misses the
            // reservation exposed. If that invariant ever changes, do not
            // merge a speculative CPU sum with a different GPU frame.
            if cold_ids != planned_cold_ids {
                if let Some(job) = cpu
                    && let Ok(mut discarded) = job.join()
                {
                    crate::attention::recycle_buf(&mut discarded);
                }
                return (
                    false,
                    false,
                    false,
                    crate::attention::take_buf(hidden),
                );
            }
            // Keep resident picks in the caller's exact route order, then
            // append the pinned shared expert. Cold picks are deliberately
            // absent: their weights and activations are completed by the
            // already-running host jobs, while sending a dummy slot with
            // weight zero would still run a full GPU expert for each miss.
            let mut resident_slots = Vec::with_capacity(route.idx.len() + 1);
            let mut resident_weights = Vec::with_capacity(route.idx.len() + 1);
            for &expert in &route.idx {
                let slot = remap[expert];
                if slot != u32::MAX {
                    resident_slots.push(slot);
                    resident_weights.push(mix_weights[expert]);
                }
            }
            resident_slots.push(shared_slot);
            resident_weights.push(shared_weight);
            let global = crate::gpu_wgpu::Dsv4GlobalMoe {
                pool_uid: model.uid(),
                shared_slot,
                segment_slots: segment_slots as u32,
            };
            let (frame_ok, route_ok) = if prepared {
                // The host is the only route authority for MiniCPM.  This
                // skips the generic device router and its duplicated
                // cold-id/x readbacks, running only resident picks plus the
                // shared expert.  A malformed compact frame simply declines
                // and the outer MoE fallback remains exact.
                (
                    crate::gpu_wgpu::minicpm_tail_prepared_frame(
                        &model,
                        global,
                        geom,
                        // Activation code for the gu shader: 3 = SwiGLU-OAI
                        // (MiniMax), 0 = the historical SiLU arm.
                        u32::from(minimax) * 3,
                        x,
                        &resident_slots,
                        &resident_weights,
                        &mut out,
                    ),
                    true,
                )
            } else {
                // `CMF_MINICPM_PREPARED_FRAME=0` is the exact old frame for
                // same-binary A/B.  Its device-side cold list is retained as
                // a diagnostic oracle; the prepared arm needs no duplicate
                // route round trip because all of it is host-owned above.
                let mut cold_from_gpu = Vec::new();
                let mut cold_x = Vec::new();
                let weights = crate::gpu_wgpu::Dsv4MoeW {
                    router: &[],
                    experts: expert_ids.as_ref(),
                    logits: &route.logits,
                    bias: Some(&mix_weights),
                    mask: None,
                    forced: Some(&route.idx),
                    remap: Some(&remap),
                    global: Some(global),
                    has_shared: true,
                    shared_weight,
                    preweighted: true,
                    qwen_softmax: true,
                };
                let frame_ok = crate::gpu_wgpu::dsv4_moe_frame(
                    &model,
                    &weights,
                    geom,
                    x,
                    &mut cold_from_gpu,
                    &mut cold_x,
                    None,
                    None,
                    &mut out,
                );
                let route_ok = cold_from_gpu.len() == cold_ids.len()
                    && cold_from_gpu
                        .iter()
                        .map(|&(expert, _)| expert)
                        .eq(cold_ids.iter().copied());
                (frame_ok, route_ok)
            };
            // The frame has completed its output readback. Keep the shared
            // bank locked through the scoped worker join: `pending.finish()`
            // has released its field borrow, but retaining the physical map
            // here prevents another pipeline from replacing slots while this
            // frame and its host complement settle.
            let cpu_out = cpu.and_then(|job| job.join().ok());
            // A panicked cold-completion job is not equivalent to an empty
            // cold list.  Keep a recyclable placeholder for the common empty
            // case, but make the outer contract reject a failed job rather
            // than returning a partial MoE sum.
            let cpu_ok = planned_cold_jobs.is_empty() || cpu_out.is_some();
            let cold_cpu = cpu_out.unwrap_or_else(|| crate::attention::take_buf(hidden));
            (frame_ok, route_ok, cpu_ok, cold_cpu)
        });
        // The scoped worker has joined and `pending.finish()` consumed its
        // pool lease, so the exact fallback below never retains the shared
        // physical-bank mutex.
        drop(bank);
        // The legacy frame's device readback remains an A/B oracle.  The
        // prepared frame has no such duplicate readback: its slot list was
        // constructed directly from this same checked remap and the frame
        // itself fail-closes on any invalid flat slot.
        if !frame_ok || !route_ok || !cpu_ok {
            crate::attention::recycle_buf(&mut cold_cpu);
            return None;
        }
        for (dst, src) in out.iter_mut().zip(&cold_cpu) {
            *dst += src;
        }
        crate::attention::recycle_buf(&mut cold_cpu);
        Some(out)
    }

    /// CPU-only builds never create a WGPU arena.  Keeping the identical API
    /// makes a caller's fallback unconditional and prevents accidental feature
    /// coupling in the pipeline.
    #[cfg(not(feature = "gpu"))]
    pub(crate) fn forward(
        &mut self,
        _li: usize,
        _m: &MoeFfn,
        _x: &[f32],
        _route: &MoeRoute,
        _pool: Option<&Pool>,
        _total_layers: usize,
    ) -> Option<Vec<f32>> {
        None
    }
}

#[cfg(feature = "gpu")]
fn q4tp_triple(
    expert: &crate::pipeline::DenseFfn,
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    hidden: usize,
    inter: usize,
) -> Option<(usize, usize, usize)> {
    use cortiq_core::TensorDtype;

    if expert.act != crate::pipeline::Act::Silu || !expert.segs.is_empty() {
        return None;
    }
    let mapped_q4tp = |tensor: &crate::qtensor::QTensor| {
        tensor.model_dtype() == Some(TensorDtype::Q4TiledP)
            && tensor.model_arc().is_some_and(|owner| owner.uid() == model.uid())
    };
    if !mapped_q4tp(&expert.gate_proj)
        || !mapped_q4tp(&expert.up_proj)
        || !mapped_q4tp(&expert.down_proj)
        || expert.gate_proj.rows() != inter
        || expert.gate_proj.cols() != hidden
        || expert.up_proj.rows() != inter
        || expert.up_proj.cols() != hidden
        || expert.down_proj.rows() != hidden
        || expert.down_proj.cols() != inter
    {
        return None;
    }
    Some((
        expert.gate_proj.model_idx()?,
        expert.up_proj.model_idx()?,
        expert.down_proj.model_idx()?,
    ))
}

/// MiniMax-M3.1's expert triple: gate/up in the bank's codec (q2tp on the
/// `q2tp-mix` file, q4tp on the 4-bit file), down always q4tp, and the
/// checked SwiGLU-OAI activation — the prepared gu shader hardcodes alpha
/// 1.702 and takes the limit from the uniform, so any other contract must
/// fall back to the exact CPU path instead of borrowing this kernel.
#[cfg(feature = "gpu")]
fn m3_triple(
    expert: &crate::pipeline::DenseFfn,
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    hidden: usize,
    inter: usize,
    gu_q2: bool,
) -> Option<(usize, usize, usize)> {
    use cortiq_core::TensorDtype;

    match expert.act {
        crate::pipeline::Act::SwiGluOai { alpha, limit }
            if (alpha - 1.702).abs() < 1e-6 && (limit - 7.0).abs() < 1e-6 => {}
        _ => return None,
    }
    if !expert.segs.is_empty() {
        return None;
    }
    let gu_dt = if gu_q2 {
        TensorDtype::Q2TiledP
    } else {
        TensorDtype::Q4TiledP
    };
    let mapped = |t: &crate::qtensor::QTensor, dt: TensorDtype| {
        t.model_dtype() == Some(dt)
            && t.model_arc().is_some_and(|owner| owner.uid() == model.uid())
    };
    if !mapped(&expert.gate_proj, gu_dt)
        || !mapped(&expert.up_proj, gu_dt)
        || !mapped(&expert.down_proj, TensorDtype::Q4TiledP)
        || expert.gate_proj.rows() != inter
        || expert.gate_proj.cols() != hidden
        || expert.up_proj.rows() != inter
        || expert.up_proj.cols() != hidden
        || expert.down_proj.rows() != hidden
        || expert.down_proj.cols() != inter
    {
        return None;
    }
    Some((
        expert.gate_proj.model_idx()?,
        expert.up_proj.model_idx()?,
        expert.down_proj.model_idx()?,
    ))
}

/// `create_explicit` delegates segment rounding and physical-bank admission to
/// the global allocator.  We deliberately preserve a non-multiple override
/// here (for example 225): the allocator rounds it safely to S8 geometry,
/// rather than a second policy silently changing the operator's request.
#[cfg(feature = "gpu")]
const LARGE_TAIL_BUDGET: u64 = 20 * 1024 * 1024 * 1024;

/// The measured 512-slot bank is worthwhile on the 24 GiB Vulkan tier, but
/// reserving it by default on a smaller adapter needlessly crowds the graph
/// prefix.  The WGPU budget (rather than raw adapter RAM) is intentional: an
/// operator's `CMF_GPU_VRAM_MB` cap remains authoritative.
#[cfg(feature = "gpu")]
fn default_tail_slots(vram_budget: u64) -> usize {
    if vram_budget >= LARGE_TAIL_BUDGET {
        512
    } else {
        224
    }
}

#[cfg(feature = "gpu")]
fn bounded_slots() -> usize {
    std::env::var("CMF_MINICPM_TAIL_SLOTS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or_else(|| default_tail_slots(crate::gpu_wgpu::device_vram_budget()))
        .clamp(16, 512)
}

/// Shared resolution for the pooled token-graph arm: model, bank key, the
/// layer's routed triples and its shared triple, or `None` for any shape the
/// device route cannot serve exactly (the caller keeps its host path).
#[cfg(feature = "gpu")]
fn minimax_pool_ctx(
    m: &MoeFfn,
    li: usize,
    total_layers: usize,
) -> Option<(
    std::sync::Arc<cortiq_core::CmfModel>,
    PoolKey,
    Vec<(usize, usize, usize)>,
    (usize, usize, usize),
)> {
    if !minimax_tail_enabled() || !crate::gpu_wgpu::dsv4_global_moe_supported() {
        return None;
    }
    if li >= total_layers
        || m.top_k != 4
        || !m.router_sigmoid
        || !m.norm_topk_prob
        || m.route_tau.is_some()
        || m.mask.is_some()
        || m.per_expert_scale.is_some()
        || m.router_input_norm
        || m.resonance.is_some()
        || !m.grown.is_empty()
    {
        return None;
    }
    let first = m.experts.first()?;
    let model = first.gate_proj.model_arc()?;
    if model.arch().arch_name != "minimax_m3_vl" {
        return None;
    }
    let hidden = m.router.cols();
    let inter = first.gate_proj.rows();
    if hidden % 32 != 0 || inter % 32 != 0 || m.router.rows() != m.experts.len() {
        return None;
    }
    let gu_q2 = match first.gate_proj.model_dtype() {
        Some(cortiq_core::TensorDtype::Q2TiledP) => true,
        Some(cortiq_core::TensorDtype::Q4TiledP) => false,
        _ => return None,
    };
    let expert_ids: Vec<(usize, usize, usize)> = m
        .experts
        .iter()
        .map(|e| m3_triple(e, &model, hidden, inter, gu_q2))
        .collect::<Option<_>>()?;
    // The device route pins the shared expert at weight 1 (rflags bit3); a
    // GATED shared needs the prepared frame's host-side gate math instead.
    let (shared, shared_gate) = m.shared.as_ref()?;
    if shared_gate.is_some() {
        return None;
    }
    let shared_ids = m3_triple(shared, &model, hidden, inter, gu_q2)?;
    let key = PoolKey {
        model_uid: model.uid(),
        hidden,
        inter,
        experts: expert_ids.len(),
        layers: total_layers,
        gu_q2,
        minimax: true,
    };
    Some((model, key, expert_ids, shared_ids))
}

/// Arm the pooled token-graph frame for one MiniMax tail layer: resolve the
/// layout against the shared bank, admit anything the last token left cold
/// (nothing on the first call), and snapshot the remap + pinned shared slot
/// for [`crate::gpu_wgpu::pooled_frame_enter`].
#[cfg(feature = "gpu")]
pub(crate) fn minimax_pool_arm(
    m: &MoeFfn,
    li: usize,
    total_layers: usize,
    predicted: &[u32],
) -> Option<crate::gpu_wgpu::PooledFrame> {
    let (model, key, expert_ids, shared_ids) = minimax_pool_ctx(m, li, total_layers)?;
    let shared = acquire_tail_bank(&model, key)?;
    let mut bank = shared.lock().ok()?;
    let candidate = LayerLayout {
        expert_ids: expert_ids.clone().into(),
        shared_ids,
    };
    if !bank.accepts_layout_or_resets(&model, key, li, candidate) {
        return None;
    }
    // Admission PREDICTION: last token's winners for this layer are admitted
    // (and their LRU clocks refreshed) before this token's route runs, so a
    // stable greedy route finds them resident and `cold` collapses.
    let pred: Vec<usize> = predicted
        .iter()
        .map(|&e| e as usize)
        .filter(|&e| e < key.experts)
        .collect();
    let pending = bank
        .gpu_pool
        .begin_ensure(&model, li, &pred, &expert_ids, Some(shared_ids))?;
    let Some((remap, shared_slot, _)) = pending.finish_and_flush() else {
        bank.gpu_pool.flush_uploads();
        return None;
    };
    if remap.len() != expert_ids.len() {
        return None;
    }
    Some(crate::gpu_wgpu::PooledFrame {
        remap,
        shared_slot,
        pool_uid: model.uid(),
    })
}

/// Admit a drained cold list so the NEXT token finds those experts
/// resident. Uses the layout [`minimax_pool_arm`] already validated.
#[cfg(feature = "gpu")]
pub(crate) fn minimax_pool_ensure(m: &MoeFfn, li: usize, total_layers: usize, cold: &[usize]) -> bool {
    if cold.is_empty() {
        return true;
    }
    let (model, key, _, shared_ids) = match minimax_pool_ctx(m, li, total_layers) {
        Some(v) => v,
        None => return false,
    };
    if cold.iter().any(|&e| e >= key.experts) {
        return false;
    }
    let Some(shared) = acquire_tail_bank(&model, key) else {
        return false;
    };
    let Ok(mut bank) = shared.lock() else {
        return false;
    };
    let (expert_ids, layout_shared) = {
        let Some(layout) = bank.layouts.get(li).and_then(|l| l.as_ref()) else {
            return false;
        };
        (layout.expert_ids.clone(), layout.shared_ids)
    };
    debug_assert_eq!(layout_shared, shared_ids);
    let Some(pending) =
        bank.gpu_pool
            .begin_ensure(&model, li, cold, expert_ids.as_ref(), Some(layout_shared))
    else {
        bank.gpu_pool.flush_uploads();
        return false;
    };
    pending.finish_and_flush().is_some()
}

/// The cold routed experts of one MiniMax layer, computed ON THE DEVICE:
/// admit them into the bank (their fills queue before the frame's submit),
/// then run the prepared frame over exactly those slots. The shared expert
/// rides the frame at weight 0, so `out` holds the cold sum alone — the
/// resident part and the shared already live in the span's device output.
///
/// `false` means nothing was written to `out`; the caller computes the cold
/// experts on the host instead (the host copies stay authoritative).
#[cfg(feature = "gpu")]
pub(crate) fn minimax_cold_device(
    m: &MoeFfn,
    li: usize,
    total_layers: usize,
    cold: &[(usize, f32)],
    x: &[f32],
    out: &mut [f32],
) -> bool {
    if cold.is_empty() || cold.len() > 4 || out.len() < x.len() {
        return false;
    }
    let Some((model, key, expert_ids, shared_ids)) = minimax_pool_ctx(m, li, total_layers) else {
        return false;
    };
    let Some(shared) = acquire_tail_bank(&model, key) else {
        return false;
    };
    let Ok(mut bank) = shared.lock() else {
        return false;
    };
    let candidate = LayerLayout {
        expert_ids: expert_ids.clone().into(),
        shared_ids,
    };
    if !bank.accepts_layout_or_resets(&model, key, li, candidate) {
        return false;
    }
    let ids: Vec<usize> = cold.iter().map(|&(e, _)| e).collect();
    let Some(pending) = bank
        .gpu_pool
        .begin_ensure(&model, li, &ids, &expert_ids, Some(shared_ids))
    else {
        bank.gpu_pool.flush_uploads();
        return false;
    };
    let Some((remap, shared_slot, segment_slots)) = pending.finish_and_flush() else {
        return false;
    };
    if remap.len() != expert_ids.len() {
        return false;
    }
    let mut slots: Vec<u32> = Vec::with_capacity(cold.len() + 1);
    let mut weights: Vec<f32> = Vec::with_capacity(cold.len() + 1);
    for &(e, w) in cold {
        let slot = match remap.get(e) {
            Some(&s) if s != u32::MAX => s,
            _ => return false,
        };
        slots.push(slot);
        weights.push(w);
    }
    slots.push(shared_slot);
    weights.push(0.0);
    let hidden = x.len();
    let Some(inter) = m.experts.first().map(|e| e.gate_proj.rows()) else {
        return false;
    };
    let geom = crate::gpu_wgpu::Dsv4MoeGeom {
        hidden,
        inter,
        top_k: 4,
        route_scale: 1.0,
        // MiniMax's checked SwiGLU-OAI contract: alpha 1.702 lives in the
        // shader's code-3 arm, the limit rides the uniform.
        swiglu_limit: 7.0,
        gu_q2: key.gu_q2,
        bf16: false,
    };
    let global = crate::gpu_wgpu::Dsv4GlobalMoe {
        pool_uid: model.uid(),
        shared_slot,
        segment_slots: segment_slots as u32,
    };
    drop(bank);
    out[..hidden].fill(0.0);
    crate::gpu_wgpu::minicpm_tail_prepared_frame(
        &model,
        global,
        geom,
        3,
        x,
        &slots,
        &weights,
        &mut out[..hidden],
    )
}

#[cfg(not(feature = "gpu"))]
pub(crate) fn minimax_cold_device(
    _m: &MoeFfn,
    _li: usize,
    _total_layers: usize,
    _cold: &[(usize, f32)],
    _x: &[f32],
    _out: &mut [f32],
) -> bool {
    false
}

#[cfg(not(feature = "gpu"))]
pub(crate) fn minimax_pool_arm(
    _m: &MoeFfn,
    _li: usize,
    _total_layers: usize,
    _predicted: &[u32],
) -> Option<()> {
    None
}

#[cfg(not(feature = "gpu"))]
pub(crate) fn minimax_pool_ensure(
    _m: &MoeFfn,
    _li: usize,
    _total_layers: usize,
    _cold: &[usize],
) -> bool {
    false
}

/// Routed-expert count of the first MoE layer at or after `from` (the route
/// frequency profile is indexed by it).
pub(crate) fn minimax_n_experts(layers: &[crate::pipeline::LayerWeights], from: usize) -> usize {
    layers
        .iter()
        .skip(from)
        .find_map(|lw| match &lw.ffn {
            crate::pipeline::FfnKind::Moe(m) => Some(m.experts.len()),
            _ => None,
        })
        .unwrap_or(0)
}

/// Public gate for the pooled-tail feature (the pipeline's device head and
/// driver ask it without touching the module's internals).
pub(crate) fn minimax_pool_on() -> bool {
    #[cfg(feature = "gpu")]
    {
        minimax_tail_enabled()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::{
        LARGE_TAIL_BUDGET, MINICPM_35B_ACTIVE_TAIL_LAYERS,
        MINICPM_35B_TOTAL_LAYERS, active_tail_layers, default_tail_slots,
    };

    #[test]
    fn tail_slot_policy_is_bounded_before_allocator_rounding() {
        // Keep this test pure: opening a WGPU context from a parallel unit
        // test would turn a slot-policy assertion into an adapter probe.
        assert!((16..=512).contains(&default_tail_slots(0)));
    }

    #[test]
    fn high_budget_default_expands_without_exceeding_tail_cap() {
        assert_eq!(default_tail_slots(LARGE_TAIL_BUDGET - 1), 224);
        assert_eq!(default_tail_slots(LARGE_TAIL_BUDGET), 512);
        assert_eq!(default_tail_slots(u64::MAX), 512);
    }

    #[test]
    fn active_floor_is_only_enabled_for_the_known_35b_tail_topology() {
        assert_eq!(
            active_tail_layers(MINICPM_35B_TOTAL_LAYERS),
            Some(MINICPM_35B_ACTIVE_TAIL_LAYERS)
        );
        // The 1B text backbone is 24-layer dense; a future MiniCPM MoE
        // variant must opt into its own measured graph/tail split.
        assert_eq!(active_tail_layers(24), None);
    }
}
