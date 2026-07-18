//! Cross-platform GPU backend (C1): wgpu → Vulkan / DX12 / Metal
//! (NVIDIA, AMD Radeon, Intel Arc, Apple). Implements the same contract as
//! `gpu_metal.rs`, behind the `gpu.rs` facade — runtime call-sites do not change.
//!
//! Difference from the Metal path: a discrete card has no unified memory, so
//! the quantized weights are LOADED into VRAM ONCE (residency cache keyed by
//! tensor index) — that is where the win lives (VRAM bandwidth ×5–10 vs CPU). The math
//! is identical to CPU/Metal: y[o] = row_scale[o]·Σ q[o,i]·xs[i], where xs is already
//! prescaled by the θ field (the two-field q8_2f folds into the input prescale).
//!
//! Enabling: `CMF_GPU=wgpu` (or `=1` on non-macOS, where wgpu is the only backend).
//! Any init/limit failure — `false` and an honest CPU path.

use crate::gpu::{BatchJob, MoeJob};
use cortiq_core::CmfModel;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use wgpu::util::DeviceExt;

/// Workgroup limit per dimension (WebGPU minimum; lm_head has more
/// rows — we use grid-stride in the shader).
const MAX_WG: u32 = 65_535;

const WGSL: &str = r#"
struct Params { cols4: u32, rows: u32, row0_words: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       q  : array<u32>;   // 4×i8 packed into u32, row-major
@group(0) @binding(1) var<storage, read>       xs : array<f32>;   // cols, already prescaled by the θ field
@group(0) @binding(2) var<storage, read>       rs : array<f32>;   // row scales for the range
@group(0) @binding(3) var<storage, read_write> y  : array<f32>;   // output: rows
@group(0) @binding(4) var<uniform>             p  : Params;

var<workgroup> partial: array<f32, 64>;

// Exact unpack of 4 signed bytes from u32 (little-endian) — like char4→
// float4 on Metal, without snorm error.
fn i8x4(w: u32) -> vec4<f32> {
    let s = i32(w);
    let b0 = (s << 24u) >> 24u;
    let b1 = (s << 16u) >> 24u;
    let b2 = (s <<  8u) >> 24u;
    let b3 =  s          >> 24u;
    return vec4<f32>(f32(b0), f32(b1), f32(b2), f32(b3));
}

// Grid-stride over rows: the number of workgroups is capped at 65535/dimension,
// while rows (lm_head) number in the hundreds of thousands; one group processes rows
// wid.x, wid.x+nwg.x, … , reducing each with 64 threads.
@compute @workgroup_size(64)
fn q8_matvec(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    var row = wid.x;
    loop {
        if (row >= p.rows) { break; }
        let base = p.row0_words + row * p.cols4;
        var acc = 0.0;
        var i = lid;
        loop {
            if (i >= p.cols4) { break; }
            let v = i8x4(q[base + i]);
            let xi = i * 4u;
            let xv = vec4<f32>(xs[xi], xs[xi + 1u], xs[xi + 2u], xs[xi + 3u]);
            acc = acc + dot(v, xv);
            i = i + 64u;
        }
        partial[lid] = acc;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) { partial[lid] = partial[lid] + partial[lid + stride]; }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u) { y[row] = partial[0] * rs[row]; }
        workgroupBarrier(); // before partial is reused by the next row
        row = row + nwg.x;
    }
}

// GEMM of the prefill batch: y[bi, o] = rs[o]·Σ q[o,i]·xs[bi,i]. One workgroup
// per (row, position); the quant row stays hot in cache across bi.
struct MMParams { cols4: u32, rows: u32, nb: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       qm  : array<u32>;
@group(0) @binding(1) var<storage, read>       xsm : array<f32>;  // [nb, cols] row-major
@group(0) @binding(2) var<storage, read>       rsm : array<f32>;  // [rows]
@group(0) @binding(3) var<storage, read_write> ym  : array<f32>;  // [nb, rows] row-major
@group(0) @binding(4) var<uniform>             pm  : MMParams;

var<workgroup> partial_mm: array<f32, 64>;

@compute @workgroup_size(64)
fn q8_matmat(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    let bi = wid.y;
    if (bi >= pm.nb) { return; }
    let xb = bi * pm.cols4 * 4u;
    var row = wid.x;
    loop {
        if (row >= pm.rows) { break; }
        let qb = row * pm.cols4;
        var acc = 0.0;
        var i = lid;
        loop {
            if (i >= pm.cols4) { break; }
            let v = i8x4(qm[qb + i]);
            let xi = xb + i * 4u;
            let xv = vec4<f32>(xsm[xi], xsm[xi + 1u], xsm[xi + 2u], xsm[xi + 3u]);
            acc = acc + dot(v, xv);
            i = i + 64u;
        }
        partial_mm[lid] = acc;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) { partial_mm[lid] = partial_mm[lid] + partial_mm[lid + stride]; }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u) { ym[bi * pm.rows + row] = partial_mm[0] * rsm[row]; }
        workgroupBarrier();
        row = row + nwg.x;
    }
}

// q1: 6-byte tiles [f16 scale][4B sign bits] per 32-group; gpr is even,
// so a row is whole 12-byte tile PAIRS = 3 u32 each (same layout walk
// as the Metal kernel). Bit set → +x. Grid-stride over rows, 64
// threads reduce a row; np = gpr/2.
struct Q1Params { np: u32, rows: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read>       q1w : array<u32>;
@group(0) @binding(1) var<storage, read>       q1x : array<f32>;   // raw f32 activations
@group(0) @binding(2) var<storage, read_write> q1y : array<f32>;
@group(0) @binding(3) var<uniform>             q1p : Q1Params;

var<workgroup> partial_q1: array<f32, 64>;

fn q1_tile_sum(bits: u32, xbase: u32) -> f32 {
    var s = vec4<f32>(0.0);
    for (var j = 0u; j < 8u; j = j + 1u) {
        let nib = bits >> (j * 4u);
        let xi = xbase + j * 4u;
        let x = vec4<f32>(q1x[xi], q1x[xi + 1u], q1x[xi + 2u], q1x[xi + 3u]);
        s = s + select(-x, x,
            vec4<bool>((nib & 1u) != 0u, (nib & 2u) != 0u, (nib & 4u) != 0u, (nib & 8u) != 0u));
    }
    return s.x + s.y + s.z + s.w;
}

@compute @workgroup_size(64)
fn q1_matvec(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    var row = wid.x;
    loop {
        if (row >= q1p.rows) { break; }
        let base = row * q1p.np * 3u;
        var acc = 0.0;
        var pi = lid;
        loop {
            if (pi >= q1p.np) { break; }
            let a0 = q1w[base + pi * 3u];
            let a1 = q1w[base + pi * 3u + 1u];
            let a2 = q1w[base + pi * 3u + 2u];
            // pair words: [s0 | bits0.lo] [bits0.hi | s1] [bits1]
            let s0 = unpack2x16float(a0).x;
            let s1 = unpack2x16float(a1).y;
            let bits0 = (a0 >> 16u) | (a1 << 16u);
            let g = pi * 2u;
            acc = acc + s0 * q1_tile_sum(bits0, g * 32u)
                      + s1 * q1_tile_sum(a2, (g + 1u) * 32u);
            pi = pi + 64u;
        }
        partial_q1[lid] = acc;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) { partial_q1[lid] = partial_q1[lid] + partial_q1[lid + stride]; }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u) { q1y[row] = partial_q1[0]; }
        workgroupBarrier();
        row = row + nwg.x;
    }
}

// Tiled GEMM for wide prefill batches (the WGSL cousin of Metal's
// q8_mul_mm; WGSL has no subgroup matrices, so this is the classic
// register-blocked form): a 64(b)×64(rows) C-tile per 16×16 workgroup,
// each thread owning a 4×4 accumulator block; X and dequantized W stage
// through 8 KB of workgroup memory in K-steps of 16. The naive kernel
// above re-reads every W row per position — here W is read once per 64
// positions. Perf is hardware-dependent by design: the runtime probe
// decides per machine whether this beats the CPU, so a card where it
// loses simply keeps the CPU path.
var<workgroup> mm_at: array<f32, 64 * 16>;
var<workgroup> mm_wt: array<f32, 64 * 16>;

@compute @workgroup_size(16, 16)
fn q8_mul_mm(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let cols = pm.cols4 * 4u;
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let tid = lid.y * 16u + lid.x;
    var acc: array<array<f32, 4>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) { acc[i][j] = 0.0; }
    }
    var k0 = 0u;
    loop {
        if (k0 >= cols) { break; }
        // Stage X tile [64×16] (4 f32 per thread) and W tile [64×16]
        // (one u32 = 4 quants per thread per round).
        for (var t = tid; t < 64u * 4u; t = t + 256u) {
            let m = t / 4u;
            let k4 = t % 4u;
            var xv = vec4<f32>(0.0);
            if (m0 + m < pm.nb && (k0 / 4u) + k4 < pm.cols4) {
                let xi = (m0 + m) * cols + k0 + k4 * 4u;
                xv = vec4<f32>(xsm[xi], xsm[xi + 1u], xsm[xi + 2u], xsm[xi + 3u]);
            }
            let dst = m * 16u + k4 * 4u;
            mm_at[dst] = xv.x;
            mm_at[dst + 1u] = xv.y;
            mm_at[dst + 2u] = xv.z;
            mm_at[dst + 3u] = xv.w;
        }
        for (var t = tid; t < 64u * 4u; t = t + 256u) {
            let n = t / 4u;
            let k4 = t % 4u;
            var wv = vec4<f32>(0.0);
            if (n0 + n < pm.rows && (k0 / 4u) + k4 < pm.cols4) {
                wv = i8x4(qm[(n0 + n) * pm.cols4 + (k0 / 4u) + k4]);
            }
            let dst = n * 16u + k4 * 4u;
            mm_wt[dst] = wv.x;
            mm_wt[dst + 1u] = wv.y;
            mm_wt[dst + 2u] = wv.z;
            mm_wt[dst + 3u] = wv.w;
        }
        workgroupBarrier();
        // 4×4 outer-product accumulation over the 16 staged K values.
        for (var k = 0u; k < 16u; k = k + 1u) {
            var av: array<f32, 4>;
            var wv: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                av[i] = mm_at[(lid.y * 4u + i) * 16u + k];
                wv[i] = mm_wt[(lid.x * 4u + i) * 16u + k];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    acc[i][j] = acc[i][j] + av[i] * wv[j];
                }
            }
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = m0 + lid.y * 4u + i;
        if (m >= pm.nb) { continue; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let n = n0 + lid.x * 4u + j;
            if (n < pm.rows) {
                ym[m * pm.rows + n] = acc[i][j] * rsm[n];
            }
        }
    }
}

// ── Element-wise kernels of the MoE block (silu·mul·col, axpy, zeroing) ──
struct N1 { n: u32, f: u32, _b: u32, _c: u32 };

@group(0) @binding(0) var<storage, read>       sg   : array<f32>;
@group(0) @binding(1) var<storage, read>       su   : array<f32>;
@group(0) @binding(2) var<storage, read>       scol : array<f32>;
@group(0) @binding(3) var<storage, read_write> sact : array<f32>;
@group(0) @binding(4) var<uniform>             snp  : N1;
@compute @workgroup_size(256)
fn silu_mul_pre(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= snp.n) { return; }
    let gv = sg[i];
    var v = (gv / (1.0 + exp(-gv))) * su[i];
    if (snp.f == 1u) { v = v * scol[i]; }
    sact[i] = v;
}

struct AxpyP { w: f32, n: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       ad : array<f32>;
@group(0) @binding(1) var<storage, read_write> ay : array<f32>;
@group(0) @binding(2) var<uniform>             ap : AxpyP;
@compute @workgroup_size(256)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ap.n) { return; }
    ay[i] = ay[i] + ap.w * ad[i];
}

@group(0) @binding(0) var<storage, read_write> zy  : array<f32>;
@group(0) @binding(1) var<uniform>             znp : N1;
@compute @workgroup_size(256)
fn fill_zero(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < znp.n) { zy[i] = 0.0; }
}
"#;

struct Ctx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matvec: wgpu::ComputePipeline,
    matmat: wgpu::ComputePipeline,
    mul_mm: wgpu::ComputePipeline,
    silu: wgpu::ComputePipeline,
    axpy: wgpu::ComputePipeline,
    zero: wgpu::ComputePipeline,
    q1: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    layout_mm: wgpu::BindGroupLayout,
    layout_mmm: wgpu::BindGroupLayout,
    layout_silu: wgpu::BindGroupLayout,
    layout_axpy: wgpu::BindGroupLayout,
    layout_zero: wgpu::BindGroupLayout,
    layout_q1: wgpu::BindGroupLayout,
    /// Discrete card (PCIe VRAM) vs UMA — thresholds and budgets differ.
    discrete: bool,
    /// Weight-residency budget in bytes (CMF_GPU_VRAM_MB override). On a
    /// 24 GB card holding a 35 GB model, the first-touched tensors (=
    /// the first layers, decode touches them in order) stay resident and
    /// the rest honestly fall back to CPU — ngl-style offload without an
    /// explicit layer list, and no OOM.
    vram_budget: u64,
    /// Bytes currently resident in `weight_bufs`.
    resident: std::sync::atomic::AtomicU64,
    /// Pooled per-op scratch (grow-only): xs upload, y output, uniform
    /// params, readback staging. Every op used to CREATE all four (plus
    /// a bind group) and map_async-poll a fresh staging buffer — pure
    /// allocator traffic on the hot path. The lock is held across the
    /// whole op (encode → submit → poll): ops already serialize on the
    /// single queue.
    scratch: Mutex<Scratch>,
    /// Resident quant weights in VRAM — the WHOLE tensor is loaded once
    /// (key (base_ptr, idx)); ranges/batches address it by offset.
    weight_bufs: Mutex<HashMap<(usize, usize), wgpu::Buffer>>,
    /// row_scale buffer per (idx, row0) — small, cached.
    rs_bufs: Mutex<HashMap<(usize, usize), wgpu::Buffer>>,
}

#[derive(Default)]
struct Scratch {
    xs: Option<(wgpu::Buffer, u64)>,
    y: Option<(wgpu::Buffer, u64)>,
    stage: Option<(wgpu::Buffer, u64)>,
    params: Option<wgpu::Buffer>,
}

impl Scratch {
    /// Grow-only slot: reuse when big enough, else recreate.
    fn ensure(
        dev: &wgpu::Device,
        slot: &mut Option<(wgpu::Buffer, u64)>,
        need: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> wgpu::Buffer {
        match slot {
            Some((b, cap)) if *cap >= need => b.clone(),
            _ => {
                crate::gpu::probe_note_cold();
                let cap = need.next_power_of_two().max(4096);
                let b = dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: cap,
                    usage,
                    mapped_at_creation: false,
                });
                *slot = Some((b.clone(), cap));
                b
            }
        }
    }
}

static CTX: OnceLock<Option<Ctx>> = OnceLock::new();

/// Whether the wgpu path is selected by env (the facade asks before `enabled()`):
/// `CMF_GPU=wgpu` — always; `CMF_GPU=1` (≠0) — only on non-macOS, where
/// there is no native Metal (on macOS `=1` goes to Metal).
pub fn selected() -> bool {
    match std::env::var("CMF_GPU") {
        Ok(v) if v == "wgpu" => true,
        Ok(v) if v != "0" => !cfg!(target_os = "macos"),
        _ => false,
    }
}

fn ctx() -> Option<&'static Ctx> {
    CTX.get_or_init(|| {
        if !selected() {
            return None;
        }
        match init() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!("wgpu init failed — CPU fallback: {e}");
                None
            }
        }
    })
    .as_ref()
}

fn init() -> Result<Ctx, String> {
    // Backend selection is automatic (wgpu picks the platform's best:
    // DX12 on Windows, Vulkan on Linux, Metal on macOS), but the
    // standard WGPU_BACKEND env (vulkan|dx12|metal|gl) forces one.
    let backends = std::env::var("WGPU_BACKEND")
        .ok()
        .map(|v| match v.to_lowercase().as_str() {
            "vulkan" | "vk" => wgpu::Backends::VULKAN,
            "dx12" | "d3d12" => wgpu::Backends::DX12,
            "metal" | "mtl" => wgpu::Backends::METAL,
            "gl" | "gles" => wgpu::Backends::GL,
            _ => wgpu::Backends::all(),
        })
        .unwrap_or(wgpu::Backends::all());
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .map_err(|e| format!("no adapter: {e}"))?;

    // Take the card's maximum limits — large tensors (lm_head ≈ 254 MB
    // int8) require a raised storage buffer; a discrete card handles GB.
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cortiq-wgpu"),
        required_limits: limits,
        ..Default::default()
    }))
    .map_err(|e| format!("request_device: {e}"))?;

    let info = adapter.get_info();
    let discrete = info.device_type == wgpu::DeviceType::DiscreteGpu;
    let vram_budget = std::env::var("CMF_GPU_VRAM_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(if discrete {
            // Conservative default for unknown cards; 4090-class users
            // should set CMF_GPU_VRAM_MB=20000.
            8 * 1024 * 1024 * 1024
        } else {
            u64::MAX // UMA: the OS pages shared memory
        });
    tracing::info!(
        "wgpu GPU path: on ({} / {:?}, {}, weight budget {})",
        info.name,
        info.backend,
        if discrete { "discrete" } else { "uma" },
        if vram_budget == u64::MAX { "unlimited".to_string() } else { format!("{} MB", vram_budget / 1024 / 1024) },
    );

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("q8"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });
    // Auto layout: the bind group layout is inferred from the shader.
    let pipe = |ep: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: None, // auto: layout is inferred from the shader
            module: &module,
            entry_point: Some(ep),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let matvec = pipe("q8_matvec");
    let matmat = pipe("q8_matmat");
    let mul_mm = pipe("q8_mul_mm");
    let silu = pipe("silu_mul_pre");
    let axpy = pipe("axpy");
    let zero = pipe("fill_zero");
    let q1 = pipe("q1_matvec");
    let layout = matvec.get_bind_group_layout(0);
    let layout_q1 = q1.get_bind_group_layout(0);
    let layout_mm = matmat.get_bind_group_layout(0);
    let layout_mmm = mul_mm.get_bind_group_layout(0);
    let layout_silu = silu.get_bind_group_layout(0);
    let layout_axpy = axpy.get_bind_group_layout(0);
    let layout_zero = zero.get_bind_group_layout(0);

    Ok(Ctx {
        device,
        queue,
        matvec,
        matmat,
        mul_mm,
        silu,
        axpy,
        zero,
        q1,
        layout,
        layout_mm,
        layout_mmm,
        layout_silu,
        layout_axpy,
        layout_zero,
        layout_q1,
        discrete,
        vram_budget,
        resident: std::sync::atomic::AtomicU64::new(0),
        scratch: Mutex::new(Scratch::default()),
        weight_bufs: Mutex::new(HashMap::new()),
        rs_bufs: Mutex::new(HashMap::new()),
    })
}

/// Is the active adapter a discrete card? (facade: threshold policy)
pub fn is_discrete() -> bool {
    ctx().map(|c| c.discrete).unwrap_or(false)
}

/// Resident quant weights of the WHOLE tensor in VRAM (loaded once per
/// (file, idx)), guarded by the VRAM budget: once the budget is spent,
/// new tensors return None and their ops run on the CPU. Decode touches
/// layers in order, so the resident set is deterministically the first
/// layers — ngl-style offload without configuration.
fn weight_buffer(c: &Ctx, key: (usize, usize), full_quant: &[u8]) -> Option<wgpu::Buffer> {
    use std::sync::atomic::Ordering;
    let mut map = c.weight_bufs.lock().unwrap();
    if let Some(b) = map.get(&key) {
        return Some(b.clone());
    }
    let len = full_quant.len() as u64;
    if c.resident.load(Ordering::Relaxed) + len > c.vram_budget {
        return None; // budget spent — this tensor stays on the CPU
    }
    crate::gpu::probe_note_cold(); // first touch = upload, not a steady sample
    let buf = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("q8-weights"),
        contents: full_quant,
        usage: wgpu::BufferUsages::STORAGE,
    });
    c.resident.fetch_add(len, Ordering::Relaxed);
    map.insert(key, buf.clone());
    Some(buf)
}

/// GPU enabled and initialized?
pub fn enabled() -> bool {
    ctx().is_some()
}

/// Probe helper: true — tensor `idx`'s weights are already resident;
/// false — not yet (with `may_upload`, the upload happens NOW within the
/// budget, without a dispatch, so the next touch is warm) or the tensor
/// can't be resolved.
pub fn q8_resident_or_upload(model: &Arc<CmfModel>, idx: usize, may_upload: bool) -> bool {
    let Some(c) = ctx() else { return false };
    let entry = &model.tensors[idx];
    let rows_total = entry.shape.first().copied().unwrap_or(0);
    let cols = entry.shape.get(1).copied().unwrap_or(0);
    if rows_total == 0 || cols == 0 {
        return false;
    }
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    if abs + rows_total * cols > bytes.len() {
        return false;
    }
    let key = (bytes.as_ptr() as usize, idx);
    if c.weight_bufs.lock().unwrap().contains_key(&key) {
        return true;
    }
    if may_upload {
        let _ = weight_buffer(c, key, &bytes[abs..abs + rows_total * cols]);
    }
    false
}

/// q8_row/q8_2f matvec on the GPU, rows [row0, row0+rows). `xs` are already
/// prescaled activations. false = could not (the caller falls back to CPU).
#[allow(clippy::too_many_arguments)]
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
    let Some(c) = ctx() else { return false };
    if cols % 4 != 0 || rows == 0 {
        return false;
    }
    let entry = &model.tensors[idx];
    let rows_total = entry.shape.first().copied().unwrap_or(0);
    if rows_total < row0 + rows {
        return false;
    }
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false; // neighboring shard — different mapping; CPU
    };
    let bytes = model.primary_bytes();
    if abs + rows_total * cols > bytes.len() {
        return false;
    }
    let full_quant = &bytes[abs..abs + rows_total * cols];
    let key = (bytes.as_ptr() as usize, idx);
    dispatch_matvec(c, Some(key), full_quant, row0, row_scale, xs, rows, cols, out)
}

/// matvec kernel: resident weights of the WHOLE tensor + row0 offset, rs, xs,
/// dispatch, readback. `weight_key = None` — no cache (test).
#[allow(clippy::too_many_arguments)]
fn dispatch_matvec(
    c: &Ctx,
    weight_key: Option<(usize, usize)>,
    full_quant: &[u8],
    row0: usize,
    row_scale: &[f32],
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    if row_scale.len() < rows || xs.len() < cols || full_quant.len() < (row0 + rows) * cols {
        return false;
    }
    let q_buf = match weight_key {
        Some(k) => match weight_buffer(c, k, full_quant) {
            Some(b) => b,
            None => return false, // over VRAM budget — honest CPU path
        },
        None => c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q8-weights"),
            contents: full_quant,
            usage: wgpu::BufferUsages::STORAGE,
        }),
    };
    let make_rs = || {
        c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q8-rs"),
            contents: bytemuck::cast_slice(&row_scale[..rows]),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let rs_buf = match weight_key {
        Some((base, idx)) => c
            .rs_bufs
            .lock()
            .unwrap()
            .entry((base ^ idx.wrapping_mul(1_000_003), row0))
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                make_rs()
            })
            .clone(),
        None => make_rs(),
    };

    // Pooled scratch for the whole op (encode → submit → poll).
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "q8-xs",
    );
    c.queue.write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..cols]));
    let y_size = (rows * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "q8-y",
    );
    let params = [(cols / 4) as u32, rows as u32, (row0 * cols / 4) as u32, 0u32];
    let p_buf = match &sc.params {
        Some(b) => b.clone(),
        None => {
            let b = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("q8-params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            sc.params = Some(b.clone());
            b
        }
    };
    c.queue.write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "q8-stage",
    );

    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q8-bg"),
        layout: &c.layout,
        entries: &[
            bind_buf(0, &q_buf),
            bind_buf(1, &xs_buf),
            bind_buf(2, &rs_buf),
            bind_buf(3, &y_buf),
            bind_buf(4, &p_buf),
        ],
    });

    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("q8") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q8"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.matvec);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1); // grid-stride over rows
    }
    let ok = readback(c, enc, &y_buf, &stage_buf, y_size, &mut out[..rows]);
    drop(sc);
    ok
}

/// q1 matvec: raw f32 activations, tile-embedded scales (no rs buffer).
/// Weights resident under the same VRAM budget as q8; false = CPU path.
pub fn q1_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let gpr = cols / 32;
    if rows == 0 || cols % 32 != 0 || gpr % 2 != 0 || xs.len() < cols || out.len() < rows {
        return false;
    }
    let entry = &model.tensors[idx];
    if entry.shape.first().copied().unwrap_or(0) < rows {
        return false;
    }
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    let plen = rows * gpr * 6;
    if abs + plen > bytes.len() {
        return false;
    }
    dispatch_q1(c, Some((bytes.as_ptr() as usize, idx)), &bytes[abs..abs + plen], xs, rows, cols, out)
}

/// q1 kernel body (weight_key = None — no residency cache; test path).
fn dispatch_q1(
    c: &Ctx,
    weight_key: Option<(usize, usize)>,
    payload: &[u8],
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let gpr = cols / 32;
    let q_buf = match weight_key {
        Some(k) => match weight_buffer(c, k, payload) {
            Some(b) => b,
            None => return false, // over VRAM budget — honest CPU path
        },
        None => c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q1-weights"),
            contents: payload,
            usage: wgpu::BufferUsages::STORAGE,
        }),
    };
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "q1-xs",
    );
    c.queue.write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..cols]));
    let y_size = (rows * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "q1-y",
    );
    let params = [(gpr / 2) as u32, rows as u32, 0u32, 0u32];
    let p_buf = match &sc.params {
        Some(b) => b.clone(),
        None => {
            let b = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("q1-params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            sc.params = Some(b.clone());
            b
        }
    };
    c.queue.write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "q1-stage",
    );
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q1-bg"),
        layout: &c.layout_q1,
        entries: &[
            bind_buf(0, &q_buf),
            bind_buf(1, &xs_buf),
            bind_buf(2, &y_buf),
            bind_buf(3, &p_buf),
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("q1") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q1"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.q1);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
    }
    let ok = readback(c, enc, &y_buf, &stage_buf, y_size, &mut out[..rows]);
    drop(sc);
    ok
}

/// GEMM of the prefill batch: `pre` are prescaled inputs row-major [b, cols],
/// out — row-major [b, rows]. Weights are resident in VRAM. false = CPU path.
#[allow(clippy::too_many_arguments)]
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
    let Some(c) = ctx() else { return false };
    if cols % 4 != 0 || rows == 0 || b == 0 {
        return false;
    }
    let entry = &model.tensors[idx];
    if entry.shape.first().copied().unwrap_or(0) < rows {
        return false;
    }
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    if abs + rows * cols > bytes.len()
        || row_scale.len() < rows
        || pre.len() < b * cols
        || out.len() < b * rows
    {
        return false;
    }
    let full_quant = &bytes[abs..abs + rows * cols];
    dispatch_matmat(
        c,
        Some((bytes.as_ptr() as usize, idx)),
        full_quant,
        row_scale,
        pre,
        b,
        rows,
        cols,
        out,
    )
}

/// matmat kernel: resident weights + rs + batch of inputs, 2D dispatch, readback.
#[allow(clippy::too_many_arguments)]
fn dispatch_matmat(
    c: &Ctx,
    weight_key: Option<(usize, usize)>,
    full_quant: &[u8],
    row_scale: &[f32],
    pre: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    if full_quant.len() < rows * cols
        || row_scale.len() < rows
        || pre.len() < b * cols
        || out.len() < b * rows
    {
        return false;
    }
    let q_buf = match weight_key {
        Some(k) => match weight_buffer(c, k, full_quant) {
            Some(b) => b,
            None => return false, // over VRAM budget — honest CPU path
        },
        None => c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mm-weights"),
            contents: full_quant,
            usage: wgpu::BufferUsages::STORAGE,
        }),
    };
    // rs cached per tensor (row0 sentinel = full-tensor scales).
    let rs_buf = match weight_key {
        Some((base, idx)) => c
            .rs_bufs
            .lock()
            .unwrap()
            .entry((base ^ idx.wrapping_mul(1_000_003), usize::MAX))
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mm-rs"),
                    contents: bytemuck::cast_slice(&row_scale[..rows]),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .clone(),
        None => c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mm-rs"),
            contents: bytemuck::cast_slice(&row_scale[..rows]),
            usage: wgpu::BufferUsages::STORAGE,
        }),
    };
    // Pooled scratch for the whole op (encode → submit → poll).
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (b * cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "mm-xs",
    );
    c.queue.write_buffer(&xs_buf, 0, bytemuck::cast_slice(&pre[..b * cols]));
    let y_size = (b * rows * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "mm-y",
    );
    let params = [(cols / 4) as u32, rows as u32, b as u32, 0u32];
    let p_buf = match &sc.params {
        Some(bf) => bf.clone(),
        None => {
            let bf = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mm-params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            sc.params = Some(bf.clone());
            bf
        }
    };
    c.queue.write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "mm-stage",
    );
    let use_mm = b >= 32;
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mm-bg"),
        // Auto bind-group layouts are pipeline-exclusive in wgpu — pick
        // the layout of the pipeline this dispatch actually uses.
        layout: if use_mm { &c.layout_mmm } else { &c.layout_mm },
        entries: &[
            bind_buf(0, &q_buf),
            bind_buf(1, &xs_buf),
            bind_buf(2, &rs_buf),
            bind_buf(3, &y_buf),
            bind_buf(4, &p_buf),
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mm") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mm"),
            timestamp_writes: None,
        });
        if use_mm {
            pass.set_pipeline(&c.mul_mm);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(
                (rows as u32).div_ceil(64).min(MAX_WG),
                (b as u32).div_ceil(64),
                1,
            );
        } else {
            pass.set_pipeline(&c.matmat);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((rows as u32).min(MAX_WG), b as u32, 1);
        }
    }
    let ok = readback(c, enc, &y_buf, &stage_buf, y_size, &mut out[..b * rows]);
    drop(sc);
    ok
}

/// Copy the output buffer GPU→staging→CPU (map+poll). Single readback path
/// for matvec/matmat.
fn readback(
    c: &Ctx,
    mut enc: wgpu::CommandEncoder,
    y_buf: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    y_size: u64,
    out: &mut [f32],
) -> bool {
    enc.copy_buffer_to_buffer(y_buf, 0, staging, 0, y_size);
    c.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..y_size);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    {
        let Ok(data) = slice.get_mapped_range() else { return false };
        out.copy_from_slice(bytemuck::cast_slice(&data[..out.len() * 4]));
    }
    staging.unmap();
    true
}

fn bind_buf(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

fn storage_bytes(c: &Ctx, data: &[u8]) -> wgpu::Buffer {
    c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn uniform_u32x4(c: &Ctx, v: [u32; 4]) -> wgpu::Buffer {
    c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&v),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn rw_f32(c: &Ctx, n: usize, copy_src: bool) -> wgpu::Buffer {
    let usage = if copy_src {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
    } else {
        wgpu::BufferUsages::STORAGE
    };
    c.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n * 4) as u64,
        usage,
        mapped_at_creation: false,
    })
}

/// Resident quant weights of tensor `idx` (the whole tensor, cached by (file,idx)).
fn tensor_weight(c: &Ctx, model: &Arc<CmfModel>, idx: usize, rows: usize, cols: usize) -> Option<wgpu::Buffer> {
    let entry = &model.tensors[idx];
    if entry.shape.first().copied().unwrap_or(0) < rows {
        return None;
    }
    let abs = model.entry_abs_offset(entry)?;
    let bytes = model.primary_bytes();
    if abs + rows * cols > bytes.len() {
        return None;
    }
    weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + rows * cols])
}

/// Encodes q8-matvec (row0=0) into the given encoder, writes to `y`. The bind
/// group and uniform are ref-counted by the command buffer until submit.
fn encode_matvec(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    rs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) {
    let p_buf = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, 0, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.layout,
        entries: &[
            bind_buf(0, weight),
            bind_buf(1, xs),
            bind_buf(2, rs),
            bind_buf(3, y),
            bind_buf(4, &p_buf),
        ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&c.matvec);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
}

/// Layer MoE-FFN in a single submission: for each expert gate/up-matvec →
/// silu·mul·col_down → down-matvec → y += w·d. Intermediate buffers are
/// GPU-resident, one sync per layer.
pub fn moe_block(model: &Arc<CmfModel>, jobs: &[MoeJob], out: &mut [f32]) -> bool {
    if jobs.iter().any(|j| j.q1) {
        return false; // q1 WGSL kernel not implemented yet — honest CPU
    }
    let Some(c) = ctx() else { return false };
    if jobs.is_empty() {
        return false;
    }
    let inter = jobs[0].gate.1;
    let hidden = jobs[0].down.1;
    if out.len() != hidden {
        return false;
    }
    // Resident weights of all triples — validate first (fail → CPU entirely).
    let mut w3 = Vec::with_capacity(jobs.len());
    for j in jobs {
        let (gi, gr, gc, _) = j.gate;
        let (ui, ur, uc, _) = j.up;
        let (di, dr, dc, _) = j.down;
        if gc % 4 != 0 || uc % 4 != 0 || dc % 4 != 0 {
            return false;
        }
        let (Some(gw), Some(uw), Some(dw)) = (
            tensor_weight(c, model, gi, gr, gc),
            tensor_weight(c, model, ui, ur, uc),
            tensor_weight(c, model, di, dr, dc),
        ) else {
            return false;
        };
        w3.push((gw, uw, dw));
    }

    let g_buf = rw_f32(c, inter, false);
    let u_buf = rw_f32(c, inter, false);
    let a_buf = rw_f32(c, inter, false);
    let d_buf = rw_f32(c, hidden, false);
    let y_buf = rw_f32(c, hidden, true);

    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("moe") });

    // y = 0
    {
        let np = uniform_u32x4(c, [hidden as u32, 0, 0, 0]);
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &c.layout_zero,
            entries: &[bind_buf(0, &y_buf), bind_buf(1, &np)],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.zero);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((hidden as u32).div_ceil(256), 1, 1);
    }

    for (j, (gw, uw, dw)) in jobs.iter().zip(&w3) {
        let (_, gr, gc, grs) = &j.gate;
        let (_, ur, uc, urs) = &j.up;
        let (_, dr, dc, drs) = &j.down;
        // Per-tensor scale/col buffers are stable across tokens — cache
        // them like the matvec row-scales instead of re-uploading.
        let mut rs_map = c.rs_bufs.lock().unwrap();
        let mut cached = |tag: usize, idx: usize, data: &[f32]| -> wgpu::Buffer {
            rs_map
                .entry((idx.wrapping_mul(1_000_003) ^ tag, usize::MAX - 1))
                .or_insert_with(|| {
                    crate::gpu::probe_note_cold();
                    storage_bytes(c, bytemuck::cast_slice(data))
                })
                .clone()
        };
        let grs_b = cached(1, j.gate.0, grs);
        let urs_b = cached(2, j.up.0, urs);
        let drs_b = cached(3, j.down.0, drs);
        let has_col = !j.down_col.is_empty();
        let col_b = if has_col {
            cached(4, j.down.0, j.down_col)
        } else {
            cached(5, usize::MAX, &[0f32]) // dummy, gated by f=0
        };
        drop(rs_map);
        let xsg = storage_bytes(c, bytemuck::cast_slice(&j.xs_gate));
        let xsu = storage_bytes(c, bytemuck::cast_slice(&j.xs_up));

        encode_matvec(c, &mut enc, gw, &xsg, &grs_b, &g_buf, *gr, *gc);
        encode_matvec(c, &mut enc, uw, &xsu, &urs_b, &u_buf, *ur, *uc);
        // act = silu(g)·u·col_down
        {
            let np = uniform_u32x4(c, [inter as u32, has_col as u32, 0, 0]);
            let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &c.layout_silu,
                entries: &[
                    bind_buf(0, &g_buf),
                    bind_buf(1, &u_buf),
                    bind_buf(2, &col_b),
                    bind_buf(3, &a_buf),
                    bind_buf(4, &np),
                ],
            });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&c.silu);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((inter as u32).div_ceil(256), 1, 1);
        }
        encode_matvec(c, &mut enc, dw, &a_buf, &drs_b, &d_buf, *dr, *dc);
        // y += w·d
        {
            let wp = uniform_u32x4(c, [j.w.to_bits(), hidden as u32, 0, 0]);
            let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &c.layout_axpy,
                entries: &[bind_buf(0, &d_buf), bind_buf(1, &y_buf), bind_buf(2, &wp)],
            });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&c.axpy);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((hidden as u32).div_ceil(256), 1, 1);
        }
    }
    // Hold the scratch lock across the readback: with concurrent server
    // slots two ops must not share the staging buffer mid-flight.
    let mut sc = c.scratch.lock().unwrap();
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        (hidden * 4) as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "moe-stage",
    );
    let ok = readback(c, enc, &y_buf, &stage_buf, (hidden * 4) as u64, out);
    drop(sc);
    ok
}

/// N independent q8-matvec (GDN projections of one input) in a single submission.
pub fn matvec_batch(model: &Arc<CmfModel>, jobs: &[BatchJob], out: &mut [&mut [f32]]) -> bool {
    let Some(c) = ctx() else { return false };
    if jobs.is_empty() || jobs.len() != out.len() {
        return false;
    }
    let mut weights = Vec::with_capacity(jobs.len());
    for j in jobs {
        if j.cols % 4 != 0 {
            return false;
        }
        let Some(w) = tensor_weight(c, model, j.idx, j.rows, j.cols) else {
            return false;
        };
        weights.push(w);
    }
    let mut y_bufs = Vec::with_capacity(jobs.len());
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("batch") });
    for (j, w) in jobs.iter().zip(&weights) {
        let rs_b = storage_bytes(c, bytemuck::cast_slice(j.row_scale));
        let xs_b = storage_bytes(c, bytemuck::cast_slice(&j.xs));
        let y_b = rw_f32(c, j.rows, true);
        encode_matvec(c, &mut enc, w, &xs_b, &rs_b, &y_b, j.rows, j.cols);
        y_bufs.push(y_b);
    }
    // ONE pooled staging buffer for all outputs (per-job offsets),
    // one map — instead of N fresh MAP_READ allocations per call.
    let total: u64 = jobs.iter().map(|j| (j.rows * 4) as u64).sum();
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        total,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "batch-stage",
    );
    let mut off = 0u64;
    for (y_b, j) in y_bufs.iter().zip(jobs) {
        enc.copy_buffer_to_buffer(y_b, 0, &stage, off, (j.rows * 4) as u64);
        off += (j.rows * 4) as u64;
    }
    c.queue.submit(Some(enc.finish()));
    stage.slice(..total).map_async(wgpu::MapMode::Read, |_| {});
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    {
        let Ok(data) = stage.slice(..total).get_mapped_range() else { return false };
        let mut off = 0usize;
        for (j, o) in jobs.iter().zip(out.iter_mut()) {
            o[..j.rows]
                .copy_from_slice(bytemuck::cast_slice(&data[off..off + j.rows * 4]));
            off += j.rows * 4;
        }
    }
    stage.unmap();
    drop(sc);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_q8_matvec_matches_cpu_reference() {
        // Force the wgpu path on (Metal-via-wgpu locally; Vulkan on the server).
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping parity test");
            return;
        };
        let (rows, cols) = (256usize, 64usize); // cols % 4 == 0
        // Synthetic int8 weights + row scales + pre-scaled activations.
        let mut q = vec![0i8; rows * cols];
        for (i, v) in q.iter_mut().enumerate() {
            *v = (((i * 37 + 11) % 255) as i32 - 127) as i8;
        }
        let rs: Vec<f32> = (0..rows).map(|r| 0.01 + (r % 7) as f32 * 0.003).collect();
        let xs: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        // CPU reference: y[o] = rs[o] * Σ q[o,i]·xs[i].
        let mut want = vec![0f32; rows];
        for o in 0..rows {
            let mut acc = 0f32;
            for i in 0..cols {
                acc += q[o * cols + i] as f32 * xs[i];
            }
            want[o] = acc * rs[o];
        }

        let qbytes: &[u8] = bytemuck::cast_slice(&q);
        let mut got = vec![0f32; rows];
        assert!(dispatch_matvec(c, None, qbytes, 0, &rs, &xs, rows, cols, &mut got));

        let max_d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-3, "wgpu q8_matvec ≠ CPU: max|Δ| = {max_d}");

        // Also check the row0 offset: the range [rows/2, rows) of the full
        // tensor must match the tail of the reference.
        let r0 = rows / 2;
        let mut got2 = vec![0f32; rows - r0];
        assert!(dispatch_matvec(
            c, None, qbytes, r0, &rs[r0..], &xs, rows - r0, cols, &mut got2
        ));
        let max_d2 = want[r0..]
            .iter()
            .zip(&got2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d2 < 1e-3, "wgpu row0 offset ≠ CPU: max|Δ| = {max_d2}");
    }

    #[test]
    fn wgpu_q1_matvec_matches_cpu_reference() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping q1 parity test");
            return;
        };
        let (rows, cols) = (33usize, 256usize); // gpr = 8 (even), odd rows
        let gpr = cols / 32;
        let mut payload = Vec::new();
        for t in 0..rows * gpr {
            let sc = 0.005 + (t % 9) as f32 * 0.004;
            payload.extend_from_slice(&cortiq_core::quant::f32_to_f16(sc).to_le_bytes());
            for j in 0..4 {
                payload.push(((t * 41 + j * 71 + 13) % 253) as u8);
            }
        }
        let xs: Vec<f32> = (0..cols).map(|i| ((i * 7 + 3) % 29) as f32 / 29.0 - 0.5).collect();
        let mut w = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_q1(&payload, &mut w);
        let mut want = vec![0f32; rows];
        for o in 0..rows {
            want[o] = (0..cols).map(|i| w[o * cols + i] * xs[i]).sum();
        }
        let mut got = vec![0f32; rows];
        assert!(dispatch_q1(c, None, &payload, &xs, rows, cols, &mut got));
        let max_d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-3, "wgpu q1_matvec ≠ CPU: max|Δ| = {max_d}");
    }

    #[test]
    fn wgpu_q8_matmat_matches_cpu_reference() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping matmat test");
            return;
        };
        let (rows, cols, b) = (128usize, 64usize, 5usize);
        let mut q = vec![0i8; rows * cols];
        for (i, v) in q.iter_mut().enumerate() {
            *v = (((i * 53 + 3) % 255) as i32 - 127) as i8;
        }
        let rs: Vec<f32> = (0..rows).map(|r| 0.01 + (r % 5) as f32 * 0.004).collect();
        let pre: Vec<f32> = (0..b * cols).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        // CPU ref: out[bi, o] = rs[o]·Σ q[o,i]·pre[bi,i].
        let mut want = vec![0f32; b * rows];
        for bi in 0..b {
            for o in 0..rows {
                let mut acc = 0f32;
                for i in 0..cols {
                    acc += q[o * cols + i] as f32 * pre[bi * cols + i];
                }
                want[bi * rows + o] = acc * rs[o];
            }
        }
        let qbytes: &[u8] = bytemuck::cast_slice(&q);
        let mut got = vec![0f32; b * rows];
        assert!(dispatch_matmat(c, None, qbytes, &rs, &pre, b, rows, cols, &mut got));
        let max_d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-3, "wgpu q8_matmat ≠ CPU: max|Δ| = {max_d}");
    }

    /// The tiled kernel (b ≥ 32) on deliberately awkward shapes: rows
    /// not a multiple of the 64-tile, cols not a multiple of the K-step
    /// — every edge guard fires.
    #[test]
    fn wgpu_q8_mul_mm_matches_cpu_reference() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping mul_mm test");
            return;
        };
        let (rows, cols, b) = (100usize, 52usize, 70usize);
        let mut q = vec![0i8; rows * cols];
        for (i, v) in q.iter_mut().enumerate() {
            *v = (((i * 31 + 7) % 255) as i32 - 127) as i8;
        }
        let rs: Vec<f32> = (0..rows).map(|r| 0.01 + (r % 7) as f32 * 0.003).collect();
        let pre: Vec<f32> = (0..b * cols).map(|i| ((i % 19) as f32 - 9.0) * 0.04).collect();
        let mut want = vec![0f32; b * rows];
        for bi in 0..b {
            for o in 0..rows {
                let mut acc = 0f32;
                for i in 0..cols {
                    acc += q[o * cols + i] as f32 * pre[bi * cols + i];
                }
                want[bi * rows + o] = acc * rs[o];
            }
        }
        let qbytes: &[u8] = bytemuck::cast_slice(&q);
        let mut got = vec![0f32; b * rows];
        assert!(dispatch_matmat(c, None, qbytes, &rs, &pre, b, rows, cols, &mut got));
        let max_d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-3, "wgpu q8_mul_mm ≠ CPU: max|Δ| = {max_d}");
    }
}
