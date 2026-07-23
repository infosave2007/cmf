//! Facade for GPU backends: a single call entry point for qtensor/pipeline/
//! linear_core. Job types and the threshold are canonical HERE; behind the
//! facade dispatch goes to a platform backend:
//!   - `gpu_metal` (Apple Silicon, unified memory + no-copy buffers);
//!   - `gpu_wgpu` (C1: Vulkan/DX12/Metal — NVIDIA/Radeon/Intel/Apple,
//!     weights resident in VRAM), available under `--features gpu`.
//!
//! Runtime selection via `CMF_GPU`: `1` — native Metal (macOS) or wgpu
//! (other OSes); `wgpu` — force wgpu (including for the local
//! Metal-via-wgpu parity test). Any backend refusal — `false` and the honest
//! CPU path, no partial results.

use cortiq_core::CmfModel;
use std::cell::Cell;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

thread_local! {
    /// Index of the current forward layer (−1 = outside a numbered layer:
    /// lm_head/embed — always allowed). The pipeline sets it before
    /// each layer so that the GPU/CPU layer-split works.
    static CUR_LAYER: Cell<i64> = const { Cell::new(-1) };
    /// Inside `cpu_scope` every GPU gate reports disabled: the timed CPU
    /// arm of a probe (and a class that lost its probe) must run PURE
    /// CPU, or inner per-op hooks would re-enter the GPU and poison the
    /// comparison.
    static CPU_ONLY: Cell<bool> = const { Cell::new(false) };
    /// "This op paid a one-off cost" (weight upload / first pipeline
    /// build): backends set it, `probe_record` discards the sample so
    /// only steady-state timings compete.
    static PROBE_COLD: Cell<bool> = const { Cell::new(false) };
}

/// Run `f` with the GPU gates off on this thread (pure-CPU arm).
pub fn cpu_scope<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            CPU_ONLY.with(|c| c.set(self.0));
        }
    }
    let previous = CPU_ONLY.with(|c| c.replace(true));
    let _restore = Restore(previous);
    f()
}

/// Backends: note a one-off cost (weight upload, buffer-cache fill) so
/// the probe discards this sample.
pub(crate) fn probe_note_cold() {
    PROBE_COLD.with(|c| c.set(true));
}

/// Pipeline: mark the current layer (or −1 outside layers) for layer-split.
pub fn set_layer(l: i64) {
    CUR_LAYER.with(|c| c.set(l));
}

/// The layer `set_layer` last marked on this thread (−1 outside layers).
pub fn cur_layer() -> i64 {
    CUR_LAYER.with(|c| c.get())
}

/// Parse `CMF_GPU_LAYERS` («0-19», «0,2,4», «0-9,30-39») once.
/// None = no restriction (all layers on GPU). Garbage → also no restriction.
fn layer_ranges() -> &'static Option<Vec<(i64, i64)>> {
    static R: OnceLock<Option<Vec<(i64, i64)>>> = OnceLock::new();
    R.get_or_init(|| {
        let s = std::env::var("CMF_GPU_LAYERS").ok()?;
        let mut v = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            match part.split_once('-') {
                Some((a, b)) => v.push((a.trim().parse().ok()?, b.trim().parse().ok()?)),
                None => {
                    let x: i64 = part.parse().ok()?;
                    v.push((x, x));
                }
            }
        }
        Some(v)
    })
}

fn layer_allowed() -> bool {
    match layer_ranges() {
        None => true,
        Some(ranges) => {
            let cur = CUR_LAYER.with(|c| c.get());
            cur < 0 || ranges.iter().any(|(a, b)| cur >= *a && cur <= *b)
        }
    }
}

/// GPU allowed FOR THE CURRENT LAYER: backend is initialized AND the layer
/// falls within `CMF_GPU_LAYERS` (GPU/CPU layer-split) AND we are not
/// inside a `cpu_scope`. Op gates call this.
pub fn enabled_here() -> bool {
    !CPU_ONLY.with(|c| c.get()) && enabled() && layer_allowed()
}

// ── Runtime GPU-vs-CPU probe ────────────────────────────────────────────
// CMF_GPU=1 does not TRUST that the device wins — it MEASURES. For each
// op class the first calls alternate arms: GPU timed vs pure-CPU timed
// (under cpu_scope). Cold GPU calls (weight upload / cache fill) are
// discarded; after PROBE_SAMPLES clean samples per arm the faster arm is
// chosen for the rest of the process. Rationale: submit+poll latency
// differs by an order of magnitude across driver stacks (Metal/PCIe
// ~3-4 ms, Vulkan/4090 ~0.3 ms) — a static threshold cannot know whether
// per-op offload pays off HERE. CMF_GPU_PROBE=0 → always trust the GPU.

/// GPU-eligible op classes, each with an independent probe.
#[derive(Clone, Copy)]
pub enum OpClass {
    /// Whole FFN chain in one submission (dense / MoE block).
    Ffn = 0,
    /// Large hybrid CPU∥GPU matvec (lm_head class).
    Matvec = 1,
    /// Prefill GEMM (matmat).
    Matmat = 2,
    /// Batched matvecs of one input (QKV).
    Batch = 3,
}

/// Probe verdict for one call.
pub enum ProbeArm {
    /// Run the GPU path (during probing: timed, recorded).
    Gpu,
    /// Probing: run the CPU path under `cpu_scope`, timed, recorded.
    CpuTimed,
    /// Decided: CPU won — run the CPU path (under `cpu_scope`).
    Cpu,
}

/// Clean samples per arm before a class decides.
const PROBE_SAMPLES: u32 = 6;

struct Probe {
    /// 0 = probing, 1 = GPU won, 2 = CPU won.
    state: AtomicU8,
    flip: AtomicU32,
    gpu_ns: AtomicU64,
    gpu_n: AtomicU32,
    cpu_ns: AtomicU64,
    cpu_n: AtomicU32,
}

impl Probe {
    const fn new() -> Self {
        Self {
            state: AtomicU8::new(0),
            flip: AtomicU32::new(0),
            gpu_ns: AtomicU64::new(0),
            gpu_n: AtomicU32::new(0),
            cpu_ns: AtomicU64::new(0),
            cpu_n: AtomicU32::new(0),
        }
    }
}

static PROBES: [Probe; 4] = [Probe::new(), Probe::new(), Probe::new(), Probe::new()];

fn probe_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CMF_GPU_PROBE")
            .map(|v| v != "0" && v != "off")
            .unwrap_or(true)
    })
}

/// q1 ops on the native Metal backend skip the probe entirely: the CPU
/// q1 kernel is load-port-bound, the GPU one wins warm — and probe
/// alternation itself cools the device between samples (measured: block
/// times 5.8 ms warm vs 8.8 ms mixed). Other backends keep probing.
pub fn q1_force() -> bool {
    #[cfg(target_os = "macos")]
    {
        backend() == Backend::Metal
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Which arm should this GPU-eligible call take? Consult AFTER the
/// eligibility gates (`enabled_here` / `min_rows`) so only real
/// candidates alternate.
pub fn probe_arm(c: OpClass) -> ProbeArm {
    if !probe_on() {
        return ProbeArm::Gpu;
    }
    let p = &PROBES[c as usize];
    match p.state.load(Ordering::Relaxed) {
        1 => ProbeArm::Gpu,
        2 => ProbeArm::Cpu,
        _ => {
            PROBE_COLD.with(|f| f.set(false));
            if p.flip.fetch_add(1, Ordering::Relaxed) % 2 == 0 {
                ProbeArm::Gpu
            } else {
                ProbeArm::CpuTimed
            }
        }
    }
}

/// Record a timed arm sample; on the `PROBE_SAMPLES`-th clean sample of
/// BOTH arms the class decides for the rest of the process.
pub fn probe_record(c: OpClass, gpu: bool, dur: std::time::Duration) {
    let p = &PROBES[c as usize];
    if p.state.load(Ordering::Relaxed) != 0 {
        return;
    }
    if gpu && PROBE_COLD.with(|f| f.replace(false)) {
        return; // one-off cost in this call — not a steady-state sample
    }
    let ns = dur.as_nanos().min(u64::MAX as u128) as u64;
    if gpu {
        p.gpu_ns.fetch_add(ns, Ordering::Relaxed);
        p.gpu_n.fetch_add(1, Ordering::Relaxed);
    } else {
        p.cpu_ns.fetch_add(ns, Ordering::Relaxed);
        p.cpu_n.fetch_add(1, Ordering::Relaxed);
    }
    let (gn, cn) = (
        p.gpu_n.load(Ordering::Relaxed),
        p.cpu_n.load(Ordering::Relaxed),
    );
    if gn >= 2 && cn >= 2 {
        let g = p.gpu_ns.load(Ordering::Relaxed) as f64 / gn as f64;
        let cp = p.cpu_ns.load(Ordering::Relaxed) as f64 / cn as f64;
        // Early verdict on a ≥3× gap — no reason to keep feeding the
        // losing arm; close races take the full sample count.
        if (gn < PROBE_SAMPLES || cn < PROBE_SAMPLES) && g < cp * 3.0 && cp < g * 3.0 {
            return;
        }
        let winner = if g <= cp { 1 } else { 2 };
        if p.state
            .compare_exchange(0, winner, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            tracing::info!(
                "gpu probe [{}]: gpu {:.2} ms vs cpu {:.2} ms per op → {}",
                ["ffn", "matvec", "matmat", "qkv-batch"][c as usize],
                g / 1e6,
                cp / 1e6,
                if winner == 1 { "gpu" } else { "cpu" },
            );
        }
    }
}

/// Is the class still collecting samples? (Call sites use this to route
/// cold-weight calls away from the GPU arm during probing.)
pub fn probe_deciding(c: OpClass) -> bool {
    probe_on() && PROBES[c as usize].state.load(Ordering::Relaxed) == 0
}

/// Probing helper: true — tensor `idx`'s quant weights are ALREADY
/// device-resident (a clean GPU sample is possible now); false — they
/// were not (the upload starts within the VRAM budget, so a later call
/// finds them warm) or the tensor cannot go to the GPU at all. Keeps the
/// probe from billing a full cold dispatch+readback to a sample it will
/// discard anyway. The verdict needs only a couple of warm tensors, so
/// probe-driven uploads are capped — the losing-GPU machine should not
/// pay for uploading the whole layer stack it will never use; if the GPU
/// wins, the rest uploads lazily on demand, in the same first-touch order.
#[allow(unused_variables)]
pub fn q8_resident_or_upload(model: &Arc<CmfModel>, idx: usize) -> bool {
    static PROBE_UPLOADS: AtomicU32 = AtomicU32::new(0);
    let may_upload = PROBE_UPLOADS.load(Ordering::Relaxed) < 4;
    let resident = match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::q8_resident_or_upload(model, idx, may_upload),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q8_resident_or_upload(model, idx, may_upload),
        Backend::None => false,
    };
    if !resident && may_upload {
        PROBE_UPLOADS.fetch_add(1, Ordering::Relaxed);
    }
    resident
}

/// Test hook: reset all probes to the undecided state.
#[cfg(test)]
pub(crate) fn probe_reset() {
    for p in &PROBES {
        p.state.store(0, Ordering::Relaxed);
        p.flip.store(0, Ordering::Relaxed);
        p.gpu_ns.store(0, Ordering::Relaxed);
        p.gpu_n.store(0, Ordering::Relaxed);
        p.cpu_ns.store(0, Ordering::Relaxed);
        p.cpu_n.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod probe_tests {
    use super::*;
    use std::time::Duration;

    // One test fn: PROBES is process-global and probe_reset touches all
    // classes — parallel test threads would race.
    #[test]
    fn probe_alternates_discards_cold_and_decides() {
        probe_reset();
        // Probing: arms alternate.
        assert!(matches!(probe_arm(OpClass::Ffn), ProbeArm::Gpu));
        assert!(matches!(probe_arm(OpClass::Ffn), ProbeArm::CpuTimed));

        // A cold GPU sample (upload noted) must be discarded: feed a
        // catastrophic cold sample, then clean fast-GPU samples — GPU
        // wins only if the cold one did not count.
        probe_note_cold();
        probe_record(OpClass::Ffn, true, Duration::from_secs(1000));
        for _ in 0..PROBE_SAMPLES {
            probe_record(OpClass::Ffn, true, Duration::from_millis(1));
            probe_record(OpClass::Ffn, false, Duration::from_millis(4));
        }
        assert!(matches!(probe_arm(OpClass::Ffn), ProbeArm::Gpu));

        // The reverse: a class where the CPU arm is faster decides CPU.
        for _ in 0..PROBE_SAMPLES {
            probe_record(OpClass::Matmat, true, Duration::from_millis(4));
            probe_record(OpClass::Matmat, false, Duration::from_millis(1));
        }
        assert!(matches!(probe_arm(OpClass::Matmat), ProbeArm::Cpu));

        // cpu_scope: gates off inside, restored after.
        cpu_scope(|| CPU_ONLY.with(|c| assert!(c.get())));
        CPU_ONLY.with(|c| assert!(!c.get()));
        cpu_scope(|| {
            cpu_scope(|| CPU_ONLY.with(|c| assert!(c.get())));
            CPU_ONLY.with(|c| assert!(c.get()));
        });
        let _ = std::panic::catch_unwind(|| cpu_scope(|| panic!("scope test")));
        CPU_ONLY.with(|c| assert!(!c.get()));
        probe_reset();
    }
}

/// Default row threshold: the GPU takes only larger matrices (lm_head
/// class). Below it, the dispatch/readback cost does not pay off on unified memory.
pub const GPU_MIN_ROWS: usize = 65_536;

/// Effective threshold: `CMF_GPU_MIN_ROWS` overrides. Defaults differ
/// by device class: on a DISCRETE card VRAM bandwidth pays off even for
/// FFN/QKV-class matrices (4096), on unified memory only lm_head-class
/// is worth the dispatch/readback (65536). Field case behind this: a
/// 35B model on an RTX 4090 saw ~0 offload because every layer matrix
/// sat below the old universal 65536.
pub fn min_rows() -> usize {
    if let Some(v) = std::env::var("CMF_GPU_MIN_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        return v;
    }
    if discrete() { 4096 } else { GPU_MIN_ROWS }
}

/// Is the active backend a discrete card (PCIe VRAM)?
pub fn discrete() -> bool {
    match backend() {
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::is_discrete(),
        #[cfg(target_os = "macos")]
        Backend::Metal => false, // UMA by the init() guard
        Backend::None => false,
    }
}

/// A single MoE-FFN job (an expert with its own weight), executed in one
/// submission: (rows, cols, idx, row_scale) for gate/up/down + prescaled
/// inputs + the down θ-field + the blending weight.
pub struct MoeJob<'a> {
    pub gate: (usize, usize, usize, &'a [f32]),
    pub up: (usize, usize, usize, &'a [f32]),
    pub down: (usize, usize, usize, &'a [f32]),
    pub xs_gate: Vec<f32>,
    pub xs_up: Vec<f32>,
    pub down_col: &'a [f32],
    pub w: f32,
    /// q1 trio: scales live inside the 6-byte tiles (row_scale slices
    /// empty, xs raw f32). Backends without a q1 kernel refuse the job.
    pub q1: bool,
}

/// A single independent batch matvec (GDN projections of one input).
pub struct BatchJob<'a> {
    pub idx: usize,
    pub rows: usize,
    pub cols: usize,
    pub row_scale: &'a [f32],
    pub xs: Vec<f32>,
    /// q1 tensor: tile-embedded scales, raw f32 xs (see `MoeJob::q1`).
    pub q1: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    None,
    #[cfg(target_os = "macos")]
    Metal,
    #[cfg(feature = "gpu")]
    Wgpu,
}

fn backend() -> Backend {
    #[cfg(feature = "gpu")]
    if crate::gpu_wgpu::selected() {
        return if crate::gpu_wgpu::enabled() {
            Backend::Wgpu
        } else {
            Backend::None
        };
    }
    #[cfg(target_os = "macos")]
    if crate::gpu_metal::enabled() {
        return Backend::Metal;
    }
    Backend::None
}

/// GPU enabled and initialized on the selected backend?
pub fn enabled() -> bool {
    backend() != Backend::None
}

/// q8_row/q8_2f matvec, rows [row0, row0+rows). `xs` — prescaled by the θ-field.
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn q8_matvec_range(
    model: &Arc<CmfModel>,
    idx: usize,
    row0: usize,
    row_scale: &[f32],
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => {
            crate::gpu_metal::q8_matvec_range(model, idx, row0, row_scale, xs, rows, cols, out)
        }
        #[cfg(feature = "gpu")]
        Backend::Wgpu => {
            crate::gpu_wgpu::q8_matvec_range(model, idx, row0, row_scale, xs, rows, cols, out)
        }
        Backend::None => false,
    }
}

/// GEMM of a prefill batch: `pre` — prescaled inputs row-major [b, cols],
/// out — row-major [b, rows].
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn q8_matmat(
    model: &Arc<CmfModel>,
    idx: usize,
    row_scale: &[f32],
    pre: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => {
            crate::gpu_metal::q8_matmat(model, idx, row_scale, pre, b, rows, cols, out)
        }
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q8_matmat(model, idx, row_scale, pre, b, rows, cols, out),
        Backend::None => false,
    }
}

/// q1 matvec: raw f32 activations, tile-embedded scales. Metal only
/// for now (wgpu q1 WGSL is queued); false = CPU fallback.
#[allow(unused_variables)]
pub fn q1_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::q1_matvec(model, idx, xs, rows, cols, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q1_matvec(model, idx, xs, rows, cols, out),
        Backend::None => false,
    }
}

/// Whole attention sub-block on the wgpu token graph (drop-in for
/// `qwen_attention`): normed hidden in, O-projection out, resident device
/// K/V mirror. false = refusal / not the wgpu backend → CPU path.
#[allow(clippy::too_many_arguments)]
pub fn attn_dropin(
    model: &Arc<CmfModel>,
    kv_id: u64,
    layer: usize,
    normed: &[f32],
    wq_idx: usize,
    wk_idx: usize,
    wv_idx: usize,
    wo_idx: usize,
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    invf: &[f32],
    nh: usize,
    nkv: usize,
    hd: usize,
    rd: usize,
    hidden: usize,
    pos: usize,
    cap: usize,
    gemma: bool,
    eps: f32,
    cpu_k: &[Vec<f32>],
    cpu_v: &[Vec<f32>],
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::attn_dropin_gpu(
            model, kv_id, layer, normed, wq_idx, wk_idx, wv_idx, wo_idx, q_norm, k_norm, invf, nh,
            nkv, hd, rd, hidden, pos, cap, gemma, eps, cpu_k, cpu_v, out,
        ),
        #[allow(unused_variables)]
        _ => false,
    }
}

/// One weight in the whole-token graph: tensor idx + a codec tag (0=q8_row,
/// 1=q1, 2=q4_tiled, 3=q1t, 4=f32) + per-row scales (q8_row only) + the raw f32
/// data (kind 4 only — small unquantized projections like GDN in_proj_a/b).
pub struct GraphW<'a> {
    pub idx: usize,
    pub kind: u8,
    pub row_scale: &'a [f32],
    pub data: &'a [f32],
}

/// A layer's token-mixing op: standard attention or a GDN (linear-attention)
/// block. The surrounding norms + SwiGLU FFN are common to both.
pub enum GraphAttn<'a> {
    Full {
        wq: GraphW<'a>,
        wk: GraphW<'a>,
        wv: GraphW<'a>,
        wo: GraphW<'a>,
        q_norm: Option<&'a [f32]>,
        k_norm: Option<&'a [f32]>,
        /// (bq, bk, bv) attention biases (Qwen2). None ⇒ no bias.
        bias: Option<(&'a [f32], &'a [f32], &'a [f32])>,
        /// Qwen3.5 gated attention: wq emits 2·nh·hd (q||gate per head), the
        /// attention output is scaled by sigmoid(gate) before the O projection.
        output_gate: bool,
        cpu_k: &'a [Vec<f32>],
        cpu_v: &'a [Vec<f32>],
    },
    Gdn {
        qkv: GraphW<'a>,
        z: GraphW<'a>,
        a: GraphW<'a>,
        b: GraphW<'a>,
        out: GraphW<'a>,
        conv1d: &'a [f32],
        a_log: &'a [f32],
        dt_bias: &'a [f32],
        norm: &'a [f32],
        nv: usize,
        nk: usize,
        dk: usize,
        dv: usize,
        kk: usize,
    },
}

/// Per-layer weights for the whole-token wgpu graph.
pub struct GraphLayer<'a> {
    pub input_norm: &'a [f32],
    pub attn: GraphAttn<'a>,
    pub post_norm: &'a [f32],
    pub gate: GraphW<'a>,
    pub up: GraphW<'a>,
    pub down: GraphW<'a>,
}

/// Whole-token decode graph on wgpu: the entire layer stack in ONE submit,
/// hidden resident, one readback. Updates `h` in place. false = refusal.
/// `loop_norm_at`: virtual layer indices after which `final_norm` is applied
/// (Looped Transformer mid-stack norm). Empty for standard models.
#[allow(clippy::too_many_arguments)]
pub fn forward_token_graph(
    model: &Arc<CmfModel>,
    kv_id: u64,
    layers: &[GraphLayer],
    invf: &[f32],
    h: &mut [f32],
    nh: usize,
    nkv: usize,
    hd: usize,
    rd: usize,
    hidden: usize,
    inter: usize,
    position: usize,
    cap: usize,
    gemma: bool,
    eps: f32,
    lm_head: Option<(&GraphW, usize)>,
    final_norm: &[f32],
    logits: &mut Vec<f32>,
    loop_norm_at: &[usize],
) -> bool {
    match backend() {
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::forward_token_graph(
            model,
            kv_id,
            layers,
            invf,
            h,
            nh,
            nkv,
            hd,
            rd,
            hidden,
            inter,
            position,
            cap,
            gemma,
            eps,
            lm_head,
            final_norm,
            logits,
            loop_norm_at,
        ),
        #[allow(unused_variables)]
        _ => {
            let _ = (lm_head, final_norm, logits, loop_norm_at);
            false
        }
    }
}

/// Batched prefill: k contiguous positions through the whole graph in one submit
/// (projections/FFN as GEMMs, attention/GDN looped over scratch). `h` is
/// [k·hidden] in/out; `positions` len k. wgpu only.
#[allow(clippy::too_many_arguments)]
pub fn forward_batch_graph(
    model: &Arc<CmfModel>,
    kv_id: u64,
    layers: &[GraphLayer],
    invf: &[f32],
    h: &mut [f32],
    nh: usize,
    nkv: usize,
    hd: usize,
    rd: usize,
    hidden: usize,
    inter: usize,
    positions: &[usize],
    cap: usize,
    gemma: bool,
    eps: f32,
    k: usize,
) -> bool {
    match backend() {
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::forward_batch_graph(
            model, kv_id, layers, invf, h, nh, nkv, hd, rd, hidden, inter, positions, cap, gemma,
            eps, k,
        ),
        _ => false,
    }
}

/// Drop the wgpu token graph's device K/V mirror for a pipeline.
pub fn graph_kv_reset(_kv_id: u64) {
    #[cfg(feature = "gpu")]
    if backend() == Backend::Wgpu {
        crate::gpu_wgpu::kv_mirror_reset(_kv_id);
    }
}

/// Ternary (q1t) BASE matvec on the GPU — fills `out` with the base dot; the
/// caller adds the sparse overlay on the CPU. Metal only for now (wgpu q1t not
/// yet written → CPU fallback).
pub fn q1t_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => {
            if metal_q1t_enabled() {
                crate::gpu_metal::q1t_matvec(model, idx, xs, rows, cols, out)
            } else {
                false
            }
        }
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q1t_matvec(model, idx, xs, rows, cols, out),
        Backend::None => false,
    }
}

/// q4_block matvec on the GPU — wgpu only (Metal drives q4_block through the
/// whole-token graph, not a standalone matvec).
#[allow(unused_variables)]
pub fn q4b_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => false,
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q4b_matvec(model, idx, xs, rows, cols, out),
        Backend::None => false,
    }
}

/// q1t batched GEMM (prefill) — base + overlay on-device (Metal simdgroup or
/// wgpu register-blocked).
pub fn q1t_matmat(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        // Batched prefill and single-token decode are both enabled. On the
        // real 14.8B Q1T model prefill PPL was within 0.3% of CPU (7.942 vs
        // 7.966), and the alignment-safe decode kernel reached 3.52e-6 max_rel.
        Backend::Metal => crate::gpu_metal::q1t_matmat(model, idx, xs, b, rows, cols, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q1t_matmat(model, idx, xs, b, rows, cols, out),
        Backend::None => false,
    }
}

/// Native Metal Q1T switch. Enabled by default after the byte-packed Q1T
/// fields were changed to alignment-safe loads; keep an explicit emergency
/// fallback for device/driver diagnostics.
#[cfg(target_os = "macos")]
pub(crate) fn metal_q1t_enabled() -> bool {
    std::env::var("CMF_METAL_Q1T")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("off"))
        .unwrap_or(true)
}

/// Batched q1 GEMM (prefill). wgpu only — Metal has its own block path.
pub fn q1_matmat(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q1_matmat(model, idx, xs, b, rows, cols, out),
        #[allow(unused_variables)]
        _ => false,
    }
}

/// Whole-block token-graph types re-exported from the Metal backend.
#[cfg(target_os = "macos")]
pub use crate::gpu_metal::{
    AttnDeviceParams, AttnGpuLayer, GdnGpuCfg, GdnGpuLayer, GraphDims, TokenGraph, kv_mirror_drop,
    kv_mirror_read_last, kv_mirror_take_imp,
};

/// A BLOCK of consecutive q1 GDN layers in one submission (Metal only).
#[cfg(target_os = "macos")]
pub fn gdn_block(
    model: &Arc<CmfModel>,
    layers: &[GdnGpuLayer],
    states: &mut [&mut [f32]],
    cfg: &GdnGpuCfg,
    h: &mut [f32],
) -> bool {
    match backend() {
        Backend::Metal => crate::gpu_metal::gdn_block(model, layers, states, cfg, h),
        _ => false,
    }
}

/// A layer's MoE-FFN in one submission (amortizing the dispatch cost).
#[allow(unused_variables)]
pub fn moe_block(model: &Arc<CmfModel>, jobs: &[MoeJob], out: &mut [f32]) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::moe_block(model, jobs, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::moe_block(model, jobs, out),
        Backend::None => false,
    }
}

/// Independent matvecs of one input in a single submission (GDN projections).
#[allow(unused_variables)]
pub fn matvec_batch(model: &Arc<CmfModel>, jobs: &[BatchJob], out: &mut [&mut [f32]]) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::matvec_batch(model, jobs, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::matvec_batch(model, jobs, out),
        Backend::None => false,
    }
}
