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

// ── Element-wise kernels of the MoE block (silu·mul·col, axpy, zeroing) ──
struct N1 { n: u32, _a: u32, _b: u32, _c: u32 };

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
    sact[i] = (gv / (1.0 + exp(-gv))) * su[i] * scol[i];
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
    silu: wgpu::ComputePipeline,
    axpy: wgpu::ComputePipeline,
    zero: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    layout_mm: wgpu::BindGroupLayout,
    layout_silu: wgpu::BindGroupLayout,
    layout_axpy: wgpu::BindGroupLayout,
    layout_zero: wgpu::BindGroupLayout,
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
    /// Resident quant weights in VRAM — the WHOLE tensor is loaded once
    /// (key (base_ptr, idx)); ranges/batches address it by offset.
    weight_bufs: Mutex<HashMap<(usize, usize), wgpu::Buffer>>,
    /// row_scale buffer per (idx, row0) — small, cached.
    rs_bufs: Mutex<HashMap<(usize, usize), wgpu::Buffer>>,
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
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
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
    let silu = pipe("silu_mul_pre");
    let axpy = pipe("axpy");
    let zero = pipe("fill_zero");
    let layout = matvec.get_bind_group_layout(0);
    let layout_mm = matmat.get_bind_group_layout(0);
    let layout_silu = silu.get_bind_group_layout(0);
    let layout_axpy = axpy.get_bind_group_layout(0);
    let layout_zero = zero.get_bind_group_layout(0);

    Ok(Ctx {
        device,
        queue,
        matvec,
        matmat,
        silu,
        axpy,
        zero,
        layout,
        layout_mm,
        layout_silu,
        layout_axpy,
        layout_zero,
        discrete,
        vram_budget,
        resident: std::sync::atomic::AtomicU64::new(0),
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
            .or_insert_with(make_rs)
            .clone(),
        None => make_rs(),
    };

    let xs_buf = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("q8-xs"),
        contents: bytemuck::cast_slice(&xs[..cols]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let y_size = (rows * 4) as u64;
    let y_buf = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("q8-y"),
        size: y_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = [(cols / 4) as u32, rows as u32, (row0 * cols / 4) as u32, 0u32];
    let p_buf = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("q8-params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

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
    readback(c, enc, &y_buf, y_size, &mut out[..rows])
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
    let rs_buf = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mm-rs"),
        contents: bytemuck::cast_slice(&row_scale[..rows]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let xs_buf = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mm-xs"),
        contents: bytemuck::cast_slice(&pre[..b * cols]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let y_size = (b * rows * 4) as u64;
    let y_buf = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mm-y"),
        size: y_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = [(cols / 4) as u32, rows as u32, b as u32, 0u32];
    let p_buf = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mm-params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mm-bg"),
        layout: &c.layout_mm,
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
        pass.set_pipeline(&c.matmat);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((rows as u32).min(MAX_WG), b as u32, 1);
    }
    readback(c, enc, &y_buf, y_size, &mut out[..b * rows])
}

/// Copy the output buffer GPU→staging→CPU (map+poll). Single readback path
/// for matvec/matmat.
fn readback(
    c: &Ctx,
    mut enc: wgpu::CommandEncoder,
    y_buf: &wgpu::Buffer,
    y_size: u64,
    out: &mut [f32],
) -> bool {
    let staging = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stg"),
        size: y_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(y_buf, 0, &staging, 0, y_size);
    c.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
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
        let grs_b = storage_bytes(c, bytemuck::cast_slice(grs));
        let urs_b = storage_bytes(c, bytemuck::cast_slice(urs));
        let drs_b = storage_bytes(c, bytemuck::cast_slice(drs));
        let col_b = storage_bytes(c, bytemuck::cast_slice(j.down_col));
        let xsg = storage_bytes(c, bytemuck::cast_slice(&j.xs_gate));
        let xsu = storage_bytes(c, bytemuck::cast_slice(&j.xs_up));

        encode_matvec(c, &mut enc, gw, &xsg, &grs_b, &g_buf, *gr, *gc);
        encode_matvec(c, &mut enc, uw, &xsu, &urs_b, &u_buf, *ur, *uc);
        // act = silu(g)·u·col_down
        {
            let np = uniform_u32x4(c, [inter as u32, 0, 0, 0]);
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
    readback(c, enc, &y_buf, (hidden * 4) as u64, out)
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
    // Copy all outputs into staging (one submit, one poll).
    let stagings: Vec<wgpu::Buffer> = jobs
        .iter()
        .map(|j| {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (j.rows * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect();
    for ((y_b, j), st) in y_bufs.iter().zip(jobs).zip(&stagings) {
        enc.copy_buffer_to_buffer(y_b, 0, st, 0, (j.rows * 4) as u64);
    }
    c.queue.submit(Some(enc.finish()));
    for st in &stagings {
        st.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    }
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    for ((st, j), o) in stagings.iter().zip(jobs).zip(out.iter_mut()) {
        let Ok(data) = st.slice(..).get_mapped_range() else { return false };
        o[..j.rows].copy_from_slice(bytemuck::cast_slice(&data[..j.rows * 4]));
    }
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
}
