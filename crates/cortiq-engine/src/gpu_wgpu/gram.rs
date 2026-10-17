//! GPTQ calibration Hessians on the device: `H += XᵀX` for every staged
//! batch of a linear's inputs, accumulated in f32 in one device buffer per
//! distinct input and read back once, as f64, when the pass ends.
//!
//! The host fold (`gptq_capture::gram_upper_add`) is what a 27B calibration
//! spent most of its time in: ~250 M multiply-adds per token for one 16-layer
//! chunk, every one of them on the CPU while the card sat idle. Here each
//! batch is one dispatch of 64×64 output tiles over the upper triangle
//! (16×16 threads, 4×4 outputs each, 16 tokens per shared-memory step).
//!
//! Precision: every dispatch sums its batch (≤ a few thousand tokens) in f32
//! registers, then adds that once into the stored f32 entry, so an entry sees
//! one rounding per batch, not one per token — about 1e-6 relative after a
//! 32k-token pass, far below the λ = 1 % damping the fold adds.

use super::{Ctx, ctx, submit};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const GRAM_SRC: &str = r#"
struct GramP { b: u32, cols: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> gx: array<f32>;
@group(0) @binding(1) var<storage, read_write> gh: array<f32>;
@group(0) @binding(2) var<uniform> gp: GramP;
var<workgroup> ta: array<f32, 1024>;
var<workgroup> tb: array<f32, 1024>;

// H[i][j] += sum_t x[t][i] * x[t][j] for j >= i, stored as the packed upper
// triangle (the host mirrors it). x is row-major [b][cols].
@compute @workgroup_size(16, 16, 1)
fn gram_upper(@builtin(workgroup_id) wg: vec3<u32>,
              @builtin(local_invocation_index) li: u32,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let ti = wg.y;
    let tj = wg.x;
    if (tj < ti) { return; }
    let n = gp.cols;
    let i0 = ti * 64u;
    let j0 = tj * 64u;
    var acc: array<vec4<f32>, 4>;
    for (var r = 0u; r < 4u; r = r + 1u) { acc[r] = vec4<f32>(0.0); }
    var t0 = 0u;
    loop {
        if (t0 >= gp.b) { break; }
        for (var e = 0u; e < 4u; e = e + 1u) {
            let idx = li + e * 256u;
            let k = idx / 64u;
            let c = idx % 64u;
            let t = t0 + k;
            var va = 0.0;
            var vb = 0.0;
            if (t < gp.b) {
                if (i0 + c < n) { va = gx[t * n + i0 + c]; }
                if (j0 + c < n) { vb = gx[t * n + j0 + c]; }
            }
            ta[idx] = va;
            tb[idx] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let ra = k * 64u + lid.y * 4u;
            let rb = k * 64u + lid.x * 4u;
            let a = vec4<f32>(ta[ra], ta[ra + 1u], ta[ra + 2u], ta[ra + 3u]);
            let bv = vec4<f32>(tb[rb], tb[rb + 1u], tb[rb + 2u], tb[rb + 3u]);
            acc[0] = fma(vec4<f32>(a.x), bv, acc[0]);
            acc[1] = fma(vec4<f32>(a.y), bv, acc[1]);
            acc[2] = fma(vec4<f32>(a.z), bv, acc[2]);
            acc[3] = fma(vec4<f32>(a.w), bv, acc[3]);
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }
    // Packed upper triangle, row-major: row r starts at r*(2n-r+1)/2.
    for (var r = 0u; r < 4u; r = r + 1u) {
        let row = i0 + lid.y * 4u + r;
        if (row < n) {
            let base = (row * (2u * n - row + 1u)) / 2u;
            for (var c = 0u; c < 4u; c = c + 1u) {
                let col = j0 + lid.x * 4u + c;
                if (col < n && col >= row) {
                    let k = base + (col - row);
                    gh[k] = gh[k] + acc[r][c];
                }
            }
        }
    }
}
"#;

static PIPE: OnceLock<Option<wgpu::ComputePipeline>> = OnceLock::new();

/// Entries of a packed upper triangle (`n(n+1)/2`).
fn packed_len(n: usize) -> u64 {
    (n as u64) * (n as u64 + 1) / 2
}

/// Start of row `r` in the packed upper triangle.
#[inline]
fn row_off(r: usize, n: usize) -> usize {
    r * (2 * n - r + 1) / 2
}

/// Device bytes a `cols`-wide accumulator takes.
pub fn bytes(cols: usize) -> u64 {
    packed_len(cols) * 4
}

fn pipe(c: &Ctx) -> Option<&'static wgpu::ComputePipeline> {
    PIPE.get_or_init(|| {
        let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gram"),
            source: wgpu::ShaderSource::Wgsl(GRAM_SRC.into()),
        });
        if let Some(e) = pollster::block_on(sc.pop()) {
            tracing::warn!("gram module rejected: {e}");
            return None;
        }
        let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gram_upper"),
            layout: None,
            module: &m,
            entry_point: Some("gram_upper"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(sc.pop()) {
            tracing::warn!("gram pipeline rejected: {e}");
            return None;
        }
        Some(p)
    })
    .as_ref()
}

struct Reg {
    next: u64,
    slots: HashMap<u64, (wgpu::Buffer, usize)>,
    x: Option<(wgpu::Buffer, u64)>,
    prm: Option<wgpu::Buffer>,
    /// Submissions since the last blocking wait: the queue's staging copies
    /// of the uploads stay alive until their submission retires.
    pending: usize,
}

static REG: Mutex<Option<Reg>> = Mutex::new(None);

/// A zeroed `cols × cols` f32 accumulator on the device, or None (no
/// device, no pipeline, or the buffer does not fit).
pub fn alloc(cols: usize) -> Option<u64> {
    let c = ctx()?;
    pipe(c)?;
    let size = packed_len(cols) * 4;
    let lim = c.device.limits();
    if size > lim.max_buffer_size || size > lim.max_storage_buffer_binding_size as u64 {
        return None;
    }
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    // wgpu zero-initializes every buffer before its first use.
    let buf = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gram_h"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if pollster::block_on(sc.pop()).is_some() {
        return None;
    }
    let mut g = REG.lock().unwrap();
    let r = g.get_or_insert_with(|| Reg { next: 1, slots: HashMap::new(), x: None, prm: None, pending: 0 });
    let id = r.next;
    r.next += 1;
    r.slots.insert(id, (buf, cols));
    Some(id)
}

/// `H[id] += XᵀX` (upper tiles) for `b` rows of `xs` (`[b·cols]`).
pub fn add(id: u64, xs: &[f32], b: usize) -> bool {
    let Some(c) = ctx() else { return false };
    let Some(p) = pipe(c) else { return false };
    let mut g = REG.lock().unwrap();
    let Some(r) = g.as_mut() else { return false };
    let Some((h, cols)) = r.slots.get(&id) else { return false };
    let cols = *cols;
    if b == 0 || xs.len() != b * cols {
        return false;
    }
    let bytes = (xs.len() * 4) as u64;
    if r.x.as_ref().is_none_or(|(_, cap)| *cap < bytes) {
        let cap = bytes.next_power_of_two();
        r.x = Some((
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gram_x"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            cap,
        ));
    }
    let prm = r.prm.get_or_insert_with(|| {
        c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gram_p"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });
    let xb = &r.x.as_ref().unwrap().0;
    c.queue.write_buffer(xb, 0, bytemuck::cast_slice(xs));
    c.queue.write_buffer(prm, 0, bytemuck::cast_slice(&[b as u32, cols as u32, 0, 0]));
    let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gram_bg"),
        layout: &p.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: xb, offset: 0, size: std::num::NonZeroU64::new(bytes) }) },
            wgpu::BindGroupEntry { binding: 1, resource: h.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: prm.as_entire_binding() },
        ],
    });
    let nt = cols.div_ceil(64) as u32;
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gram") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gram"), timestamp_writes: None });
        pass.set_pipeline(p);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(nt, nt, 1);
    }
    submit(c, enc.finish());
    r.pending += 1;
    if r.pending >= 16 {
        r.pending = 0;
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
    }
    true
}

/// `H[id] += XᵀX` for `b` rows of a device buffer `x` (`[b·cols]` f32).
pub fn add_dev(id: u64, x: &wgpu::Buffer, b: usize) -> bool {
    let Some(c) = ctx() else { return false };
    let Some(p) = pipe(c) else { return false };
    let mut g = REG.lock().unwrap();
    let Some(r) = g.as_mut() else { return false };
    let Some((h, cols)) = r.slots.get(&id) else { return false };
    let cols = *cols;
    if b == 0 {
        return true;
    }
    let prm = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gram_p"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    c.queue.write_buffer(&prm, 0, bytemuck::cast_slice(&[b as u32, cols as u32, 0, 0]));
    let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gram_bg"),
        layout: &p.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: x, offset: 0, size: std::num::NonZeroU64::new((b * cols * 4) as u64) }) },
            wgpu::BindGroupEntry { binding: 1, resource: h.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: prm.as_entire_binding() },
        ],
    });
    let nt = cols.div_ceil(64) as u32;
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gram") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gram"), timestamp_writes: None });
        pass.set_pipeline(p);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(nt, nt, 1);
    }
    submit(c, enc.finish());
    true
}

/// Read `H[id]` back as a full symmetric f64 matrix and free the buffer.
pub fn take(id: u64) -> Option<Vec<f64>> {
    let out = read(id)?;
    free(id);
    Some(out)
}

/// Read `H[id]` back as a full symmetric f64 matrix; the buffer stays.
pub fn read(id: u64) -> Option<Vec<f64>> {
    let c = ctx()?;
    let (h, cols) = buffer(id)?;
    let n = cols;
    let total = packed_len(n) * 4;
    const CHUNK: u64 = 256 << 20;
    let staging = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gram_rb"),
        size: CHUNK.min(total),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let t0 = std::time::Instant::now();
    let mut upper = vec![0f32; packed_len(n) as usize];
    let (mut t_dma, mut t_cp) = (0f64, 0f64);
    let mut off = 0u64;
    while off < total {
        let ta = std::time::Instant::now();
        let len = CHUNK.min(total - off);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gram_rb") });
        enc.copy_buffer_to_buffer(&h, off, &staging, 0, len);
        submit(c, enc.finish());
        let slice = staging.slice(..len);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
            return None;
        }
        t_dma += ta.elapsed().as_secs_f64();
        let tb = std::time::Instant::now();
        {
            let Ok(data) = slice.get_mapped_range() else {
                staging.unmap();
                return None;
            };
            let dst = &mut upper[(off / 4) as usize..((off + len) / 4) as usize];
            super::par_copy(dst, bytemuck::cast_slice(&data[..len as usize]));
        }
        staging.unmap();
        t_cp += tb.elapsed().as_secs_f64();
        off += len;
    }
    drop(h);
    let t_mirror = std::time::Instant::now();
    // Expand the packed triangle in 64-row bands: a band reads its own
    // packed rows left to right and, for its lower half, a 64-wide strip of
    // every packed row above it (contiguous), never one element per line.
    let mut out = vec![0f64; n * n];
    const B: usize = 64;
    let bands: Vec<(usize, &mut [f64])> = out.chunks_mut(B * n).enumerate().map(|(k, c)| (k * B, c)).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let bands = Mutex::new(bands.into_iter().map(Some).collect::<Vec<_>>());
    let nth = std::thread::available_parallelism().map_or(8, |p| p.get());
    std::thread::scope(|s| {
        for _ in 0..nth {
            s.spawn(|| {
                let upper = &upper;
                loop {
                    let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((r0, band)) = bands.lock().unwrap().get_mut(k).and_then(Option::take) else { break };
                    let r1 = (r0 + B).min(n);
                    // j >= i: straight copy of each row's packed part.
                    for i in r0..r1 {
                        let row = &mut band[(i - r0) * n..(i - r0 + 1) * n];
                        let src = &upper[row_off(i, n)..row_off(i, n) + (n - i)];
                        for (v, &u) in row[i..].iter_mut().zip(src) {
                            *v = u as f64;
                        }
                    }
                    // j < i: H[i][j] = H[j][i], read from packed row j.
                    for j in 0..r1.min(n) {
                        let src = &upper[row_off(j, n)..row_off(j, n) + (n - j)];
                        for i in r0.max(j + 1)..r1 {
                            band[(i - r0) * n + j] = src[i - j] as f64;
                        }
                    }
                }
            });
        }
    });
    if std::env::var("CMF_GRAM_TRACE").as_deref() == Ok("1") {
        eprintln!(
            "gram take {n}²: alloc+dma {:.2}s copy {:.2}s mirror {:.2}s total {:.2}s",
            t_dma,
            t_cp,
            t_mirror.elapsed().as_secs_f64(),
            t0.elapsed().as_secs_f64()
        );
    }
    Some(out)
}

/// The packed accumulator `id` for another device stage (the GPTQ fold),
/// left in the registry: (buffer handle, cols).
pub(super) fn buffer(id: u64) -> Option<(wgpu::Buffer, usize)> {
    REG.lock().unwrap().as_ref()?.slots.get(&id).map(|(b, n)| (b.clone(), *n))
}

/// Free accumulator `id` without reading it.
pub fn free(id: u64) {
    if let Some(r) = REG.lock().unwrap().as_mut() {
        r.slots.remove(&id);
    }
}

/// Drop every accumulator (a pass that failed half way).
pub fn clear() {
    if let Some(r) = REG.lock().unwrap().as_mut() {
        r.slots.clear();
    }
}
