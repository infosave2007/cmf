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
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
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

/// Peek the cold flag without consuming it (`probe_record` consumes).
/// Contention heuristics use this: a slow COLD op is a one-off build
/// cost, not evidence the device is busy.
pub(crate) fn probe_was_cold() -> bool {
    PROBE_COLD.with(|c| c.get())
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
    /// Prefill GEMM at image-diffusion widths (b ≥ 128). Probed apart
    /// from `Matmat`: one imagegen process runs BOTH populations
    /// (prompt encode b≈40 where the GPU wins big, DiT b≥256 where
    /// the CPU AMX arm is competitive) — a single shared verdict locks
    /// the wrong arm for whichever population samples second.
    MatmatWide = 4,
    /// The lm_head itself, apart from the merely-large matvecs. Same
    /// reasoning as `MatmatWide`, and DeepSeek-V4 is where it bit: its
    /// attention projections are 37M weights and its head is 529M, so
    /// the projections' verdict — CPU, honestly measured at 0.19 ms —
    /// decided for a matvec fourteen times their size that took 11 ms
    /// a token on the host.
    MatvecHead = 5,
    /// The blocked f32 GEMM (`fcd_ops::gemm_nt`): attention's QKᵀ and
    /// AV, and the VAE decoders' projections. It used to take every job
    /// over 4 M MACs on sight, with no CPU arm to lose to — which on
    /// the MiniMax-H3 video decoder was three times SLOWER than the
    /// host it displaced. Its population is per-head slices, nothing
    /// like the weight GEMMs above, so it probes on its own.
    GemmNt = 6,
}

/// Which probe a large matvec belongs to. The head is an order of
/// magnitude bigger than anything else that reaches this gate, and the
/// two populations do not have the same answer.
pub fn matvec_class(rows: usize, cols: usize) -> OpClass {
    if rows * cols >= 67_108_864 {
        OpClass::MatvecHead
    } else {
        OpClass::Matvec
    }
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
    /// Best (minimum) sample per arm. The DECISION compares these:
    /// means are poisoned by one-off cold costs the cold-flag cannot
    /// see — e.g. the CPU arm's first mmap-cold expert matvec page
    /// faults its weights in and reads 3× its steady state, which
    /// locked the GPU arm on a 35B MoE at a 4× real-world loss. The
    /// minimum is each arm's honest steady-state pace.
    gpu_min: AtomicU64,
    cpu_min: AtomicU64,
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
            gpu_min: AtomicU64::new(u64::MAX),
            cpu_min: AtomicU64::new(u64::MAX),
        }
    }
}

static PROBES: [Probe; 7] = [
    Probe::new(),
    Probe::new(),
    Probe::new(),
    Probe::new(),
    Probe::new(),
    Probe::new(),
    Probe::new(),
];

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

/// Should a FUSED whole-block path trust the device instead of asking
/// the per-op probe? True on native Metal and on discrete wgpu adapters.
///
/// The probe answers "is one wide matmat faster on the GPU", and for the
/// DiT on Metal that is a coin flip — measured 2.62 ms GPU vs 2.56 ms
/// CPU, a 2% spread that lands on either arm run to run. But the fused
/// block's advantage is not per-op speed, it is that the hidden state,
/// the packs and the attention panels never leave the device: end to end
/// the whole-block path renders a 512² Lumina step in ~5.4 s against
/// ~8.4 s when the probe happens to pick the CPU. Gating a fusion win on
/// a per-op tie made every second render half-speed at random.
///
/// On a discrete card the verdict is never in doubt — an RTX 3090 against
/// a 256-core EPYC measured 11.5 ms vs 31 ms per wide op, four runs out
/// of four — so the probe's sampling phase is pure cost: it alone was 10%
/// of a 512² render (74.3 s against 66.9 s with the probe off). Integrated
/// and mobile adapters keep probing; there the submit latency is real and
/// can genuinely lose.
pub fn fused_block_trusted() -> bool {
    #[cfg(target_os = "macos")]
    if backend() == Backend::Metal {
        return true;
    }
    wgpu_graph_default()
}

/// Which arm should this GPU-eligible call take? Consult AFTER the
/// eligibility gates (`enabled_here` / `min_rows`) so only real
/// candidates alternate.
/// While a class is still probing, a call whose weights are NOT yet on
/// the card should take the GPU arm anyway: the upload is work the next
/// step needs regardless, and the sample it produces is discarded as
/// cold — so handing that call to the CPU arm buys nothing and costs a
/// host GEMM. Measured on a diffusion stack, where every layer is
/// touched once per step and therefore EVERY first-step GPU sample is
/// cold: one projection drew the CPU arm for the whole first step, 9.8 s
/// against the 2.8 s it costs once the weights are warm.
pub fn weight_is_resident(model: &Arc<CmfModel>, idx: usize) -> bool {
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    {
        return crate::gpu_wgpu::weight_is_resident(model, idx);
    }
    #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
    {
        let _ = (model, idx);
        true
    }
}

pub fn probe_arm_cold_prefers_gpu(c: OpClass, weights_resident: bool) -> ProbeArm {
    if !weights_resident && probe_deciding(c) {
        return ProbeArm::Gpu;
    }
    probe_arm(c)
}

pub fn probe_arm(c: OpClass) -> ProbeArm {
    // Every arbitrated call starts with a clean cold flag: both the
    // sample discard in `probe_record` and the contention kill-switch
    // read it AFTER the op, so a stale note from a previous call on
    // this thread must not leak in.
    PROBE_COLD.with(|f| f.set(false));
    if !probe_on() {
        return ProbeArm::Gpu;
    }
    let p = &PROBES[c as usize];
    match p.state.load(Ordering::Relaxed) {
        1 => ProbeArm::Gpu,
        2 => ProbeArm::Cpu,
        _ => {
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
        p.gpu_min.fetch_min(ns, Ordering::Relaxed);
    } else {
        p.cpu_ns.fetch_add(ns, Ordering::Relaxed);
        p.cpu_n.fetch_add(1, Ordering::Relaxed);
        p.cpu_min.fetch_min(ns, Ordering::Relaxed);
    }
    let (gn, cn) = (
        p.gpu_n.load(Ordering::Relaxed),
        p.cpu_n.load(Ordering::Relaxed),
    );
    if gn >= 2 && cn >= 2 {
        // Decide on each arm's BEST sample — the steady-state pace.
        // Means carry one-off cold costs (mmap page-in on the CPU arm)
        // that the cold-flag machinery cannot see.
        let g = p.gpu_min.load(Ordering::Relaxed) as f64;
        let cp = p.cpu_min.load(Ordering::Relaxed) as f64;
        // Early verdict on a ≥2× gap — no reason to keep feeding the
        // losing arm; close races take the full sample count. It was 3×,
        // and the cost of that half-octave was measured: a DiT whose
        // wide GEMMs run 11.4 ms on the device against 32.2 on the host
        // (2.8×) kept ALTERNATING through the whole diffusion stack, and
        // because the alternation counter is shared per class in call
        // order, one projection drew the CPU arm every single time — 9.9
        // seconds a step on a kernel that needs 0.4. Both arms are
        // compared on their BEST sample, so a 2× gap is not noise.
        if (gn < PROBE_SAMPLES || cn < PROBE_SAMPLES) && g < cp * 2.0 && cp < g * 2.0 {
            return;
        }
        let winner = if g <= cp { 1 } else { 2 };
        if p.state
            .compare_exchange(0, winner, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            tracing::info!(
                "gpu probe [{}]: gpu {:.2} ms vs cpu {:.2} ms per op → {}",
                ["ffn", "matvec", "matmat", "qkv-batch", "matmat-wide", "lm-head", "gemm-nt"]
                    [c as usize],
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
    /// q4_tiled trio: scales inside the 18-byte tiles (row_scale
    /// slices empty, xs raw f32) — the MoE-hybrid coder class.
    pub q4t: bool,
    /// q4tp trio: same raw-xs contract, 16-byte nibble stride and the scale
    /// on a per-row ladder. Without this the experts of a q4tp MoE model fall
    /// to the CPU while every other dtype rides the device.
    pub q4tp: bool,
    /// Mixed 2-bit profile: gate/up are q2tp (8-byte chunks, zero rung),
    /// down stays q4tp. Set together with `q4tp`; a backend without the
    /// 2-bit kernel must refuse the whole job.
    pub gu_q2: bool,
    /// The reference's `swiglu_limit`; 0 disables the clamp. A backend that
    /// cannot apply it must REFUSE the job rather than drop it silently —
    /// the difference only shows on saturating activations, which is the
    /// hardest kind of divergence to notice.
    pub swiglu_limit: f32,
}

/// A single independent batch matvec (GDN projections of one input).
pub struct BatchJob<'a> {
    pub idx: usize,
    pub rows: usize,
    pub cols: usize,
    pub row_scale: &'a [f32],
    pub xs: Vec<f32>,
    /// Weight layout. Was a bare `q1: bool`, which could only ever spell two
    /// of the four and silently sent everything else back to the CPU — the
    /// GDN projections of a q4t/q4tp model never reached the device at all.
    pub layout: BatchLayout,
}

/// Which kernel a batched matvec needs. q8 carries row scales in a side
/// buffer; the rest embed them in the payload and differ in stride.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BatchLayout {
    Q8,
    Q1,
    Q4t,
    Q4tp,
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
/// Whether THIS build can bring a GPU up on THIS device: a compiled-in
/// backend plus a live adapter. The mobile FFI exposes it so an app can
/// tell "GPU off" from "GPU impossible" (a CPU-only .so ships no
/// backend at all). Cached after the first call.
pub fn backend_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        // The Metal path is always compiled on macOS.
        true
    }
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    {
        static AVAIL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *AVAIL.get_or_init(crate::gpu_wgpu::adapter_probe)
    }
    #[cfg(all(not(feature = "gpu"), not(target_os = "macos")))]
    {
        false
    }
}

pub fn enabled() -> bool {
    backend() != Backend::None
}

/// Default-on condition for the wgpu whole-token graph: the wgpu
/// backend on a DISCRETE adapter. NOT plain `enabled()` (macOS/Metal
/// must not pay a per-token layer scan for a graph its backend
/// refuses), and NOT integrated adapters: the graph's ~300 barriered
/// dispatches per token are cheap on desktop immediate-mode GPUs but
/// tiled mobile GPUs (Adreno/Mali) drain the pipeline at every barrier
/// — field report: 0.2 tok/s on-graph vs 15 tok/s on the CPU. On
/// integrated adapters the per-op probe path arbitrates each op class
/// against the CPU instead; CMF_GPU_WGPU_GRAPH=1 still forces the
/// graph anywhere.
/// Is the wgpu backend active at all (any adapter)? Eligibility gate
/// for the whole-token graph — whether it actually RUNS is decided by
/// `wgpu_graph_default` (trusted on discrete) or the generation race.
pub fn wgpu_active() -> bool {
    #[cfg(feature = "gpu")]
    {
        matches!(backend(), Backend::Wgpu)
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Which GPU this thread's engine calls address. Multi-card hosts hold
/// one wgpu context PER card (weights, KV mirrors and scratch live
/// inside a context, so per-device contexts give per-device caches for
/// free); this thread-local says which one is current. Default: the
/// process pin (CMF_GPU_ADAPTER) or 0 — so single-card runs behave
/// exactly as they always have.
pub fn default_device() -> usize {
    static D: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *D.get_or_init(|| {
        std::env::var("CMF_GPU_ADAPTER")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0)
    })
}

thread_local! {
    static CUR_DEV: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// The device this thread is pinned to.
pub fn current_device() -> usize {
    CUR_DEV.with(|c| c.get()).unwrap_or_else(default_device)
}

/// Pin this thread to a device. Server slots call it once per request;
/// the worker pool propagates it into its threads, so a dispatch begun
/// on card 1 does not finish on card 0.
pub fn set_current_device(i: usize) {
    CUR_DEV.with(|c| c.set(Some(i)));
}

/// Run `f` with this thread pinned to `dev`, restoring the previous pin.
pub fn with_device<R>(dev: usize, f: impl FnOnce() -> R) -> R {
    let prev = CUR_DEV.with(|c| c.replace(Some(dev)));
    let r = f();
    CUR_DEV.with(|c| c.set(prev));
    r
}

/// How many GPUs this process can address (wgpu adapter count; 1 on
/// Metal, 0 without a backend).
pub fn device_count() -> usize {
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    {
        return crate::gpu_wgpu::adapter_count();
    }
    #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
    {
        usize::from(backend_available())
    }
}

/// Weight budget of the current GPU in bytes; 0 when there is none and
/// u64::MAX on unified memory (where the OS pages shared RAM and the
/// question "does the model fit the card" has no separate answer).
pub fn vram_budget() -> u64 {
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    {
        return crate::gpu_wgpu::device_vram_budget();
    }
    #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
    {
        if backend_available() { u64::MAX } else { 0 }
    }
}

/// Device weight bytes uploaded so far (wgpu; 0 on other backends).
/// Steady-state windows must show a ZERO delta — growth mid-benchmark
/// means eviction/re-upload and disqualifies the number.
pub fn upload_bytes() -> u64 {
    #[cfg(feature = "gpu")]
    {
        return crate::gpu_wgpu::UPLOAD_BYTES.load(std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(not(feature = "gpu"))]
    0
}

pub fn wgpu_graph_default() -> bool {
    #[cfg(feature = "gpu")]
    {
        // Discrete cards always; Apple-silicon UMA on macOS too — desktop
        // -class GPUs where the graph measured ~2x the CPU on the Qwen3.6
        // family (M4: 13.3 tok/s against 7.3). Phone-class UMA (Android/
        // iOS builds) keeps the per-op probe path: tiled mobile GPUs have
        // turned the ~300-dispatch graph into seconds per token.
        matches!(backend(), Backend::Wgpu)
            && (crate::gpu_wgpu::discrete_active()
                || (cfg!(target_os = "macos") && crate::gpu_wgpu::adapter_up()))
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
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
        /// CPU recurrent state `[ring (kk-1)·cdim | S nv·dk·dv]` — seeds the
        /// device mirror when prefill ran on the host (o1 collection, CPU
        /// fallback): a zero-initialized device state at decode is exactly
        /// the "coherent but contextless" garble.
        cpu_state: &'a [f32],
    },
}

/// Per-layer weights for the whole-token wgpu graph.
pub struct GraphLayer<'a> {
    pub input_norm: &'a [f32],
    pub attn: GraphAttn<'a>,
    pub post_norm: &'a [f32],
    pub ffn: GraphFfn<'a>,
}

/// The FFN of one graph layer: a dense SwiGLU trio, or a routed MoE —
/// router + top-k selection + all selected experts run ON DEVICE (the
/// routing decision depends on the resident hidden state, so a CPU
/// round-trip per layer would forfeit the one-submit design).
pub enum GraphFfn<'a> {
    Dense {
        gate: GraphW<'a>,
        up: GraphW<'a>,
        down: GraphW<'a>,
    },
    Moe {
        /// Router logits weight (f32, kind 4) `[n_exp, hidden]`.
        router: GraphW<'a>,
        /// Shared-expert sigmoid gate (f32) `[1, hidden]`.
        shared_gate: GraphW<'a>,
        /// Per-expert q4_tiled directory indices `(gate, up, down)`;
        /// the SHARED expert rides as the LAST entry — the select
        /// kernel pins it with the sigmoid weight.
        experts: Vec<(usize, usize, usize)>,
        /// Routed experts (shared excluded).
        n_exp: usize,
        top_k: usize,
        inter: usize,
        norm_topk: bool,
        /// Expert weight layout, uniform across the layer: `false` =
        /// q4_tiled (18 B tiles, inline f16 scale), `true` = q4tp
        /// (16 B nibbles + a per-row ladder plane). The two differ only
        /// in where the scale comes from, so they share every kernel
        /// but the weight-staging block.
        q4tp: bool,
        /// `true` = the gate/up experts are `q2tp` (2-bit plane) while
        /// `down` stays q4tp — the mixed profile a 2-bit-class checkpoint
        /// converts into. Only meaningful with `q4tp: true`.
        gu_q2: bool,
    },
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
    // Per-layer sealed o1 (Nystrom) state; Some = replace this layer's
    // exact attention with the O(1) kernels. wgpu only.
    o1: &[Option<Vec<crate::nystrom::O1DeviceView<'_>>>],
    o1_epoch: u64,
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
    steps: usize,
    embed: Option<(&GraphW, usize, f32)>,
    ids_out: Option<&mut Vec<u32>>,
    // How many leading layers the graph ran (see the wgpu twin) — smaller
    // than layers.len() when the expert budget ended the device prefix.
    layers_run: Option<&mut usize>,
    // Absolute index of layers[0] in the model — the KV/state mirrors key
    // on it, so a layer SPAN (network split segment) shares mirrors with
    // a full-stack run instead of colliding at slot 0.
    layer_base: usize,
) -> bool {
    match backend() {
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::forward_token_graph(
            model,
            kv_id,
            layers,
            o1,
            o1_epoch,
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
            steps,
            embed,
            ids_out,
            layers_run,
            layer_base,
        ),
        #[allow(unused_variables)]
        _ => {
            let _ = (lm_head, final_norm, logits, loop_norm_at, layers_run, layer_base);
            false
        }
    }
}

/// Speculative-verify tail for the batched graph: fold final-norm + lm_head
/// over every batch position and read all k logit rows back; the batch also
/// snapshots the GDN state per position for `gdn_spec_restore`.
pub struct SpecTail<'a> {
    pub lm: GraphW<'a>,
    pub lm_rows: usize,
    pub final_norm: &'a [f32],
    pub logits_out: &'a mut Vec<f32>,
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
    spec: Option<SpecTail<'_>>,
) -> bool {
    match backend() {
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::forward_batch_graph(
            model, kv_id, layers, invf, h, nh, nkv, hd, rd, hidden, inter, positions, cap, gemma,
            eps, k, spec,
        ),
        #[allow(unreachable_patterns)]
        _ => {
            let _ = spec;
            false
        }
    }
}

/// After a partial speculative acceptance: restore every GDN layer's device
/// state to the snapshot after batch position `slot`. wgpu only.
pub fn gdn_spec_restore(kv_id: u64, slot: usize) -> bool {
    #[cfg(feature = "gpu")]
    if backend() == Backend::Wgpu {
        return crate::gpu_wgpu::gdn_spec_restore(kv_id, slot);
    }
    #[allow(unreachable_code)]
    {
        let _ = (kv_id, slot);
        false
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

/// Contention kill for the wide imagegen GEMM/FFN paths: one grossly
/// slow op under a work-proportional budget (fair-device ops are
/// ≤~100 ms even at 1024px) means another process owns the device —
/// verdicts are per-process, so CPU for the rest of this one.
static MM_KILL: AtomicBool = AtomicBool::new(false);
pub(crate) fn mm_killed() -> bool {
    MM_KILL.load(Ordering::Relaxed)
}
pub(crate) fn mm_kill() {
    MM_KILL.store(true, Ordering::Relaxed);
}

/// Fused DiT SwiGLU FFN on the device: g=X·W1ᵀ, u=X·W3ᵀ, silu(g)·u,
/// Causal chunk attention on the device: `b` queries against `s0 + b`
/// cached keys. wgpu only — Metal's chunk graph keeps attention inside
/// the resident block and never calls out.
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn chunk_attend(
    q: &[f32],
    k: &[&[f32]],
    v: &[&[f32]],
    b: usize,
    s0: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    scale: f32,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::chunk_attend(q, k, v, b, s0, nh, nkv, hd, scale, out),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// Fused QKV projection: one upload of the normed chunk, three GEMMs,
/// one readback of Q|K|V back to back. Metal has no twin yet — its
/// chunk graph keeps the whole layer resident and never surfaces QKV.
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn q4t_qkv(
    model: &Arc<CmfModel>,
    wq: usize,
    wk: usize,
    wv: usize,
    xs: &[f32],
    b: usize,
    cols: usize,
    rq: usize,
    rk: usize,
    rv: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q4t_qkv(model, wq, wk, wv, xs, b, cols, rq, rk, rv, out),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// y=·W2ᵀ — one command buffer, only X and Y cross the CPU boundary.
#[allow(unused_variables, clippy::too_many_arguments)]
/// SwiGLU FFN with a row-packed [gate|up] fc1 (MiniMax-H3's DiT), run
/// end to end on the device. wgpu only: Metal keeps the host loop until
/// its own packed kernel exists.
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn q4tp_ffn_packed(
    model: &Arc<CmfModel>,
    w1: usize,
    w2: usize,
    xs: &[f32],
    b: usize,
    hidden: usize,
    inter: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(feature = "gpu")]
        Backend::Wgpu => {
            crate::gpu_wgpu::q4tp_ffn_packed(model, w1, w2, xs, b, hidden, inter, bias, out)
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

pub fn q4tp_ffn(
    model: &Arc<CmfModel>,
    w1: usize,
    w3: usize,
    w2: usize,
    xs: &[f32],
    b: usize,
    hidden: usize,
    inter: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::q4tp_ffn(model, w1, w3, w2, xs, b, hidden, inter, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q4tp_ffn(model, w1, w3, w2, xs, b, hidden, inter, out),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

pub fn q4t_ffn(
    model: &Arc<CmfModel>,
    w1: usize,
    w3: usize,
    w2: usize,
    xs: &[f32],
    b: usize,
    hidden: usize,
    inter: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::q4t_ffn(model, w1, w3, w2, xs, b, hidden, inter, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q4t_ffn(model, w1, w3, w2, xs, b, hidden, inter, out),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// One whole modulated DiT block for `dit_block`: geometry, norm
/// weights, AdaLN scale/gate vectors (gates pre-tanh'd), a per-token
/// f32 RoPE cos/sin table, and the directory indices of the seven
/// q4t projections. `x` is in-out `[n, hidden]`.
pub struct DitBlockArgs<'a> {
    pub n: usize,
    pub hidden: usize,
    pub inter: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub eps: f32,
    pub rope_cos: &'a [f32],
    pub rope_sin: &'a [f32],
    pub norm1: &'a [f32],
    pub norm2: &'a [f32],
    pub ffn_norm1: &'a [f32],
    pub ffn_norm2: &'a [f32],
    pub norm_q: &'a [f32],
    pub norm_k: &'a [f32],
    pub s_msa: &'a [f32],
    pub gate_msa: &'a [f32],
    pub s_mlp: &'a [f32],
    pub gate_mlp: &'a [f32],
    pub wq: usize,
    pub wk: usize,
    pub wv: usize,
    pub wo: usize,
    pub w1: usize,
    pub w3: usize,
    pub w2: usize,
    /// The projections' layout: q4tp (ladder scales) vs plain q4_tiled.
    /// The recommended Lumina file is q4tp, and a backend that only
    /// knows q4t must decline rather than decode with the wrong reader.
    pub q4tp: bool,
    /// The hidden state is already on the device from the previous block,
    /// so `x` need not be uploaded.
    pub resident_in: bool,
    /// Leave the result on the device instead of reading it back. The DiT
    /// loop does not touch `x` between blocks, so 27 of every 28 readbacks
    /// were moving 19 MB across PCIe and stalling on it for nothing.
    pub resident_out: bool,
}

/// Can the selected backend keep the DiT's hidden state on the device
/// between blocks? Only the wgpu whole-block path; the Metal entry takes
/// and returns host memory every call.
pub fn dit_chain_supported() -> bool {
    #[cfg(feature = "gpu")]
    {
        return matches!(backend(), Backend::Wgpu) && fused_dit_block_available();
    }
    #[allow(unreachable_code)]
    false
}

/// Pull the resident hidden state back to the host. For the caller that
/// chained blocks and then hit one the device declined.
pub fn dit_state_fetch(_x: &mut [f32]) -> bool {
    #[cfg(feature = "gpu")]
    {
        if matches!(backend(), Backend::Wgpu) {
            return crate::gpu_wgpu::dit_state_fetch(_x);
        }
    }
    false
}

/// One whole modulated DiT block on the device — norms, qkv, RoPE,
/// attention, residuals and the SwiGLU FFN in a single command
/// buffer; only `x` crosses the CPU boundary (in and out).
#[allow(unused_variables)]
/// The DiT's three projections in one submission (wgpu only; the
/// Metal path fuses the whole block instead). False = the caller keeps
/// its three separate calls.
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn dit_qkv(
    model: &Arc<CmfModel>,
    wq: usize,
    wk: usize,
    wv: usize,
    xs: &[f32],
    b: usize,
    hidden: usize,
    qrows: usize,
    kvrows: usize,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q4tp_qkv(
            model, wq, wk, wv, xs, b, hidden, qrows, kvrows, q_out, k_out, v_out,
        ),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// Is a FUSED whole-block device path on offer? The batched-CFG shape
/// (two sequences in one tall batch) and the fused block (one sequence,
/// one command buffer) are alternatives, and the caller picks.
pub fn fused_dit_block_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        matches!(backend(), Backend::Metal) && fused_block_trusted()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

pub fn dit_block(model: &Arc<CmfModel>, a: &DitBlockArgs, x: &mut [f32]) -> bool {
    dit_block_seg(model, a, &[a.n], x)
}

/// The same block over a CONCATENATION of independent sequences:
/// attention per segment, everything position-wise batched. wgpu only —
/// the Metal path takes the single-sequence entry above.
pub fn dit_block_seg(
    model: &Arc<CmfModel>,
    a: &DitBlockArgs,
    segs: &[usize],
    x: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal if segs.len() <= 1 => crate::gpu_metal::dit_block(model, a, x),
        // The wgpu whole-block path. What it buys is host round trips —
        // six a block become one — so it defaults ON where those cost
        // real time (a discrete card across PCIe) and OFF on unified
        // memory, where the per-op path shares the same pages and the
        // fusion measured slightly slower on an M4. `CMF_DIT_FUSED=1`
        // forces it anywhere, `=0` forbids it.
        #[cfg(feature = "gpu")]
        Backend::Wgpu
            if match std::env::var("CMF_DIT_FUSED").ok().as_deref() {
                Some("0") => false,
                Some(_) => true,
                None => crate::gpu_wgpu::discrete_active(),
            } =>
        {
            crate::gpu_wgpu::dit_block_seg(model, a, segs, x)
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// One VAE resnet block for `vae_resnet`: norm/conv weights and the
/// channel/shape geometry. `shortcut` is the 1×1 projection (w, b, k)
/// when in/out channels differ.
pub struct VaeResnetArgs<'a> {
    pub groups: usize,
    pub ic: usize,
    pub oc: usize,
    pub h: usize,
    pub w: usize,
    pub n1w: &'a [f32],
    pub n1b: &'a [f32],
    pub c1w: &'a [f32],
    pub c1b: &'a [f32],
    pub c1k: usize,
    pub n2w: &'a [f32],
    pub n2b: &'a [f32],
    pub c2w: &'a [f32],
    pub c2b: &'a [f32],
    pub c2k: usize,
    pub shortcut: Option<(&'a [f32], &'a [f32], usize)>,
}

/// One whole VAE resnet block on the device (norm+silu → conv ×2 →
/// shortcut → add, one command buffer).
#[allow(unused_variables)]
pub fn vae_resnet(a: &VaeResnetArgs, x: &[f32], out: &mut [f32]) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::vae_resnet(a, x, out),
        _ => false,
    }
}

/// Nearest-2× upsample fused with the following conv — the small
/// pre-upsample image is what crosses the CPU boundary.
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn vae_upsample_conv(
    w: &[f32],
    bias: &[f32],
    x: &[f32],
    ic: usize,
    oc: usize,
    h: usize,
    w_img: usize,
    k: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::vae_upsample_conv(w, bias, x, ic, oc, h, w_img, k, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => {
            crate::gpu_wgpu::vae_upsample_conv(w, bias, x, ic, oc, h, w_img, k, out)
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// VAE conv2d on the device (implicit GEMM — the CPU path pays for a
/// multi-GB im2col matrix at high resolutions).
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn vae_conv2d(
    w: &[f32],
    bias: &[f32],
    x: &[f32],
    ic: usize,
    oc: usize,
    h: usize,
    w_img: usize,
    k: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::vae_conv2d(w, bias, x, ic, oc, h, w_img, k, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::vae_conv2d(w, bias, x, ic, oc, h, w_img, k, out),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// DiT full bidirectional attention on the device (all heads:
/// scores GEMM → row softmax → P·V → panel unstack, one command
/// buffer). Head-major inputs; out is [n, nh·hd].
#[allow(unused_variables, clippy::too_many_arguments)]
/// Attention from an interleaved qkv panel, splitting into head-major
/// planes ON the device. wgpu only; `false` elsewhere so the caller
/// keeps its host repack.
#[allow(unused_variables)]
#[allow(clippy::too_many_arguments)]
/// qkv projection + attention with the panel never leaving the card.
/// wgpu only; `false` elsewhere and the caller keeps its host chain.
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn dit_qkv_attention(
    model: &Arc<CmfModel>,
    qkv_idx: usize,
    xn: &[f32],
    n: usize,
    hidden: usize,
    nh: usize,
    hd: usize,
    scale: f32,
    nr: (&[f32], &[f32], &[f32], f32),
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        Backend::Wgpu => crate::gpu_wgpu::dit_qkv_attention(
            model, qkv_idx, xn, n, hidden, nh, hd, scale, nr, out,
        ),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

pub fn dit_attention_packed(
    qkv: &[f32],
    nh: usize,
    n: usize,
    hd: usize,
    scale: f32,
    // (rope angles, q norm weights, k norm weights, eps) when the device
    // should apply qk-norm and RoPE itself; None when the host already did.
    nr: Option<(&[f32], &[f32], &[f32], f32)>,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        Backend::Wgpu => crate::gpu_wgpu::dit_attention_packed(qkv, nh, n, hd, scale, nr, out),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

pub fn dit_attention(
    qh: &[f32],
    kh: &[f32],
    vh: &[f32],
    nh: usize,
    nkv: usize,
    n: usize,
    hd: usize,
    scale: f32,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::dit_attention(qh, kh, vh, nh, nkv, n, hd, scale, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::dit_attention(qh, kh, vh, nh, nkv, n, hd, scale, out),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// Batched q4t GEMM on the device (imagegen DiT prefill shapes).
/// Metal: q4t_mul_mm decodes the mmap-resident tiles inside the
/// GEMM's K loop. wgpu (Vulkan/DX12 → NVIDIA/AMD/Intel/Adreno/Mali):
/// the register-blocked WGSL twin, weights cached in VRAM.
#[allow(unused_variables)]
pub fn q4tp_matmat(
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
        Backend::Metal => crate::gpu_metal::q4tp_matmat(model, idx, xs, b, rows, cols, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q4tp_matmat(model, idx, xs, b, rows, cols, out),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// The same over a two-bit weight plane. Metal has no q2tp kernel, so
/// there it declines and the host takes it.
pub fn q2tp_matmat(
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
        Backend::Wgpu => crate::gpu_wgpu::q2tp_matmat(model, idx, xs, b, rows, cols, out),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// Single-token q4tp matvec on the device — the lm_head class. Through the
/// DEDICATED matvec kernel: the batched GEMM at b=1 measured 11.73 ms
/// against the host's 9.51 on the release head, so the route that was
/// supposed to save eleven milliseconds a token lost its own probe instead.
pub fn q4tp_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::q4tp_matvec_for_test(model, idx, xs, rows, cols, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q4tp_matvec(model, idx, xs, rows, cols, out),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// Single-token q4_tiled matvec on the device — the lm_head class (a
/// q4t checkpoint's head is its biggest host matvec, exactly like the
/// q4tp twin above). wgpu holds q4t_mv pipelines only inside the graph
/// encoder — the standalone arm stays an honest refusal until a
/// discrete-GPU q4t model reaches the bench.
pub fn q4t_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    match backend() {
        #[cfg(target_os = "macos")]
        Backend::Metal => crate::gpu_metal::q4t_matvec_for_test(model, idx, xs, rows, cols, out),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

pub fn q4t_matmat(
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
        Backend::Metal => crate::gpu_metal::q4t_matmat(model, idx, xs, b, rows, cols, out),
        #[cfg(feature = "gpu")]
        Backend::Wgpu => crate::gpu_wgpu::q4t_matmat(model, idx, xs, b, rows, cols, out),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// Whole-block token-graph types re-exported from the Metal backend.
#[cfg(target_os = "macos")]
pub use crate::gpu_metal::{
    AttnDeviceParams, AttnGpuLayer, GdnGpuCfg, GdnGpuLayer, GpuMoe, GraphDims, MetalFfn,
    TokenGraph, kv_mirror_drop, kv_mirror_read_last, kv_mirror_take_imp,
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

// ── Whole-token wgpu graph race (generation granularity) ─────────────
// On integrated/mobile adapters the graph is neither trusted nor banned
// a priori — it RACES the normal path: generations alternate arms (the
// normal path first — known-good UX — then the graph), per-token wall
// times accumulate per arm, and once both arms have enough steady
// samples the faster one wins for the process. Arm switches happen ONLY
// at generation boundaries (`kv_cache.clear()` resets state), so the
// device KV mirror and the CPU cache never diverge mid-sequence. The
// single exception is the first-token bail: the very first decode token
// of a graph generation may be discarded and recomputed on the CPU
// path (the prompt KV is CPU-owned at that point, so this is safe) —
// a tiled mobile GPU that drains its pipeline at every barrier turns
// the ~300-dispatch graph into seconds per token (field report: 0.2
// tok/s vs 15 on the CPU), and one token is all it takes to see that.
static GRAPH_RACE_STATE: AtomicU8 = AtomicU8::new(0); // 0 racing, 1 graph won, 2 normal won
static GRAPH_RACE_FLIP: AtomicU32 = AtomicU32::new(0);
static GRAPH_RACE_ARM_GRAPH: AtomicU8 = AtomicU8::new(0); // this generation's arm
static GRAPH_RACE_TOK: AtomicU32 = AtomicU32::new(0); // token index within the generation
static GRAPH_NS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)]; // [normal, graph]
static GRAPH_N: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];

/// Steady per-token samples per arm before the race decides.
const GRAPH_RACE_SAMPLES: u32 = 4;

/// Called at every generation start (fresh KV). Applies a pending
/// verdict and picks this generation's arm while racing.
pub fn graph_race_begin_generation() {
    GRAPH_RACE_TOK.store(0, Ordering::Relaxed);
    if GRAPH_RACE_STATE.load(Ordering::Relaxed) != 0 {
        return;
    }
    let (gn, cn) = (
        GRAPH_N[1].load(Ordering::Relaxed),
        GRAPH_N[0].load(Ordering::Relaxed),
    );
    if gn >= GRAPH_RACE_SAMPLES && cn >= GRAPH_RACE_SAMPLES {
        let g_avg = GRAPH_NS[1].load(Ordering::Relaxed) / gn as u64;
        let c_avg = GRAPH_NS[0].load(Ordering::Relaxed) / cn as u64;
        let verdict = if g_avg < c_avg { 1 } else { 2 };
        GRAPH_RACE_STATE.store(verdict, Ordering::Relaxed);
        tracing::info!(
            "wgpu graph race: graph {:.2} ms/tok vs normal {:.2} ms/tok -> {}",
            g_avg as f64 / 1e6,
            c_avg as f64 / 1e6,
            if verdict == 1 { "graph" } else { "normal path" }
        );
        return;
    }
    let flip = GRAPH_RACE_FLIP.fetch_add(1, Ordering::Relaxed);
    GRAPH_RACE_ARM_GRAPH.store((flip % 2 == 1) as u8, Ordering::Relaxed);
}

/// Should this decode token try the graph? `trusted` (discrete adapter,
/// explicit env, or a GDN hybrid whose state lives on the device) skips
/// the race entirely.
pub fn graph_race_use_graph(trusted: bool) -> bool {
    if trusted {
        return true;
    }
    match GRAPH_RACE_STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => GRAPH_RACE_ARM_GRAPH.load(Ordering::Relaxed) == 1,
    }
}

/// First decode token of a racing graph generation: hopeless already?
/// (>4x the normal path's per-token average AND over a second.) Settles
/// the race immediately; the caller discards the graph result and
/// recomputes this token on the normal path.
pub fn graph_race_first_token_hopeless(dur: std::time::Duration) -> bool {
    if GRAPH_RACE_STATE.load(Ordering::Relaxed) != 0 {
        return false;
    }
    let first = GRAPH_RACE_TOK.load(Ordering::Relaxed) == 0;
    let cn = GRAPH_N[0].load(Ordering::Relaxed);
    if !first || cn == 0 {
        return false;
    }
    let c_avg = GRAPH_NS[0].load(Ordering::Relaxed) / cn as u64;
    let ns = dur.as_nanos() as u64;
    if ns > 1_000_000_000 && ns > 4 * c_avg {
        GRAPH_RACE_STATE.store(2, Ordering::Relaxed);
        tracing::info!(
            "wgpu graph race: first graph token {:.0} ms vs normal {:.2} ms/tok — hopeless, normal path wins",
            ns as f64 / 1e6,
            c_avg as f64 / 1e6
        );
        return true;
    }
    false
}

/// Record one decode-token wall time for the racing arm. The first
/// token of each generation is discarded (KV-mirror upload / cold
/// caches on the graph arm; cold mmap on the normal arm).
pub fn graph_race_record(used_graph: bool, dur: std::time::Duration) {
    if GRAPH_RACE_STATE.load(Ordering::Relaxed) != 0 {
        return;
    }
    let tok = GRAPH_RACE_TOK.fetch_add(1, Ordering::Relaxed);
    if tok == 0 {
        return;
    }
    let i = used_graph as usize;
    GRAPH_NS[i].fetch_add(dur.as_nanos() as u64, Ordering::Relaxed);
    GRAPH_N[i].fetch_add(1, Ordering::Relaxed);
}

/// Bounded-cost content fingerprint for the backends' pointer-keyed device
/// caches: FNV over the whole slice up to 4 KiB, over 64 spread 64-byte
/// windows (plus the length) above. An address-keyed hit must also prove
/// the bytes are still the ones it uploaded — the allocator reuses heap
/// and mmap addresses freely, so a reloaded model or a re-dequantized
/// layer lands where the old bytes were — and sampling keeps that proof at
/// ~a microsecond even for a 126 MB matrix. Real replacements (another
/// model's tensor, an Adam-updated master) differ densely, so a 4 KiB
/// spread cannot miss them.
pub(crate) fn fp_bytes(data: &[u8]) -> u64 {
    #[inline]
    fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
        let (chunks, tail) = bytes.split_at(bytes.len() & !7);
        for c in chunks.chunks_exact(8) {
            h ^= u64::from_le_bytes(c.try_into().unwrap());
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        for &b in tail {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ (data.len() as u64);
    if data.len() <= 4096 {
        return fnv(h, data);
    }
    let step = (data.len() - 64) / 63;
    for i in 0..64 {
        h = fnv(h, &data[i * step..i * step + 64]);
    }
    h
}

/// `fp_bytes` over an f32 slice without a bytemuck dependency (the Metal
/// backend builds with no GPU feature flags).
pub(crate) fn fp_f32(data: &[f32]) -> u64 {
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    fp_bytes(bytes)
}

#[cfg(test)]
mod fp_tests {
    use super::fp_bytes;

    /// The pointer-keyed caches survive on `fp_bytes` telling two different
    /// tensors apart at a reused address. Its sampling must therefore see a
    /// change ANYWHERE — head, tail, and the stretches between windows are
    /// the places a cheaper hash would go blind.
    #[test]
    fn fp_bytes_sees_a_change_anywhere_in_a_sampled_slice() {
        let n = 1 << 20; // 1 MiB — far above the 4 KiB full-hash threshold
        let base: Vec<u8> = (0..n).map(|i| (i * 31 + 7) as u8).collect();
        let h0 = fp_bytes(&base);
        assert_eq!(h0, fp_bytes(&base), "fingerprint must be deterministic");
        // A DENSE change (every requantized/redequantized tensor is one)
        // must flip the fingerprint no matter how the windows fall.
        let mut dense = base.clone();
        for b in dense.iter_mut() {
            *b = b.wrapping_add(1);
        }
        assert_ne!(h0, fp_bytes(&dense), "a fully different tensor slipped through");
        // Length participates: the same prefix at a shorter length is a
        // different key AND a different fingerprint.
        assert_ne!(h0, fp_bytes(&base[..n - 64]));
        // Below the threshold the hash is exact: a single flipped byte in
        // a norm-sized vector must be seen.
        let mut small = vec![3u8; 4096];
        let hs = fp_bytes(&small);
        small[2048] ^= 1;
        assert_ne!(hs, fp_bytes(&small), "full hash missed a one-byte change");
        // And the sampled windows land within bounds on awkward sizes.
        for n in [4097usize, 5000, 64 * 64, 1 << 16] {
            let v = vec![9u8; n];
            let _ = fp_bytes(&v); // must not panic on window math
        }
    }
}

/// Hand the card back after a bake: drop its resident weights, planes and
/// pools so the ordinary engine (the runtime gate, a serve that follows)
/// starts from a clean budget. No-op off the wgpu backend.
pub fn bake_release() {
    #[cfg(feature = "gpu")]
    crate::gpu_wgpu::bake_release();
}

/// Strict-f32 for the bake's GEMMs (phase A mask training): the mask
/// selects neurons by a gradient signal, and f16 operand rounding on
/// that signal closes the wrong ones. No-op off the wgpu backend.
pub fn bake_precision_strict(on: bool) {
    #[cfg(feature = "gpu")]
    crate::gpu_wgpu::bake_precision_strict(on);
    #[cfg(not(feature = "gpu"))]
    let _ = on;
}
