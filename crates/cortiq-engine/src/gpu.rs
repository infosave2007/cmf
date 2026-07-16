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
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
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
    CPU_ONLY.with(|c| c.set(true));
    let r = f();
    CPU_ONLY.with(|c| c.set(false));
    r
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
    let (gn, cn) = (p.gpu_n.load(Ordering::Relaxed), p.cpu_n.load(Ordering::Relaxed));
    if gn >= 2 && cn >= 2 {
        let g = p.gpu_ns.load(Ordering::Relaxed) as f64 / gn as f64;
        let cp = p.cpu_ns.load(Ordering::Relaxed) as f64 / cn as f64;
        // Early verdict on a ≥3× gap — no reason to keep feeding the
        // losing arm; close races take the full sample count.
        if (gn < PROBE_SAMPLES || cn < PROBE_SAMPLES) && g < cp * 3.0 && cp < g * 3.0 {
            return;
        }
        let winner = if g <= cp { 1 } else { 2 };
        if p
            .state
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
    if let Some(v) = std::env::var("CMF_GPU_MIN_ROWS").ok().and_then(|v| v.parse().ok()) {
        return v;
    }
    if discrete() {
        4096
    } else {
        GPU_MIN_ROWS
    }
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
}

/// A single independent batch matvec (GDN projections of one input).
pub struct BatchJob<'a> {
    pub idx: usize,
    pub rows: usize,
    pub cols: usize,
    pub row_scale: &'a [f32],
    pub xs: Vec<f32>,
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
        return if crate::gpu_wgpu::enabled() { Backend::Wgpu } else { Backend::None };
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
        Backend::Wgpu => {
            crate::gpu_wgpu::q8_matmat(model, idx, row_scale, pre, b, rows, cols, out)
        }
        Backend::None => false,
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
