//! Binary low-rank (`blr`) on wgpu: the two ±1 stages as two dispatches in
//! one submission (docs/BLR_NVG.md §3).
//!
//! * `blr_v`: one workgroup per latent row `q = p·r + k` and block of `TB`
//!   tokens: `T[t, q] = ℓ_p[k] · Σ_j V_p(k, j) · g_p[j] · x[t, j]`, the signs
//!   read 16 at a time from the packed row.
//! * `blr_u`: one thread per output row, one workgroup row per token:
//!   `y[t, i] = Σ_p h_p[i] · Σ_k U_p(i, k) · T[t, p·r + k]`, with `T[t]`
//!   staged in workgroup memory.
//!
//! The payload is uploaded once as-is (resident weight cache); every section
//! offset is recomputed in the shader from `(d_out, d_in, r)`, and f16 scales
//! are read with `unpack2x16float`. `b = 1` is the decode GEMV, `b > 1` the
//! batched (prefill) form of the same two kernels.

use super::*;
use std::sync::Arc;

const TB: usize = 4;
/// Workgroup staging of one token's latents (`P·r` floats).
const MAX_PR: usize = 2048;

const COMMON: &str = r#"
struct P0 { d_out: u32, d_in: u32, r: u32, np: u32 };
struct P1 { b: u32, a1: u32, a2: u32, a3: u32 };
fn path_base(p: u32) -> u32 {
  let s = 2u * (p0.d_out + p0.d_in + p0.r) + (p0.r / 8u) * (p0.d_out + p0.d_in);
  return 16u + p * s;
}
fn f16at(byte_off: u32) -> f32 {
  let v = unpack2x16float(w[byte_off >> 2u]);
  return select(v.x, v.y, (byte_off & 2u) != 0u);
}
"#;

const V_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> t: array<f32>;
@group(0) @binding(3) var<uniform> p0: P0;
@group(0) @binding(4) var<uniform> p1: P1;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(64)
fn blr_v(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let q = wid.x;
  let p = q / p0.r;
  let k = q % p0.r;
  let t0 = wid.y * 4u;
  let base = path_base(p);
  let goff = base + 2u * p0.d_out;
  let loff = goff + 2u * p0.d_in;
  let uoff = loff + 2u * p0.r;
  let voff = uoff + p0.d_out * (p0.r / 8u);
  let vrow = (voff >> 2u) + k * (p0.d_in / 32u);
  let gw0 = goff >> 2u;
  var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  let nh = p0.d_in / 16u;
  let row4 = p0.d_in / 4u;
  for (var hw = lid.x; hw < nh; hw += 64u) {
    let word = w[vrow + (hw >> 1u)];
    let bits = (word >> ((hw & 1u) * 16u)) & 0xffffu;
    let j0 = hw * 16u;
    let gw = gw0 + (j0 >> 1u);
    for (var c = 0u; c < 4u; c++) {
      let ga = unpack2x16float(w[gw + 2u * c]);
      let gb = unpack2x16float(w[gw + 2u * c + 1u]);
      let sb = (bits >> (4u * c)) & 0xfu;
      let s4 = vec4<f32>(f32(sb & 1u), f32((sb >> 1u) & 1u), f32((sb >> 2u) & 1u), f32((sb >> 3u) & 1u)) * 2.0 - vec4<f32>(1.0);
      let gs = vec4<f32>(ga.x, ga.y, gb.x, gb.y) * s4;
      for (var tt = 0u; tt < 4u; tt++) {
        let tok = t0 + tt;
        if (tok < p1.b) {
          acc[tt] += dot(x[tok * row4 + (j0 >> 2u) + c], gs);
        }
      }
    }
  }
  for (var tt = 0u; tt < 4u; tt++) {
    red[tt * 64u + lid.x] = acc[tt];
  }
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      for (var tt = 0u; tt < 4u; tt++) {
        red[tt * 64u + lid.x] += red[tt * 64u + lid.x + s];
      }
    }
    workgroupBarrier();
  }
  if (lid.x < 4u) {
    let tok = t0 + lid.x;
    if (tok < p1.b) {
      t[tok * (p0.np * p0.r) + q] = red[lid.x * 64u] * f16at(loff + 2u * k);
    }
  }
}
"#;

const U_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> t: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p0: P0;
@group(0) @binding(4) var<uniform> p1: P1;
var<workgroup> ts: array<f32, 2048>;

@compute @workgroup_size(64)
fn blr_u(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tok = wid.y;
  // Workgroup-uniform guard; it also keeps binding 4 in the derived layout
  // (an unused binding is dropped from it and the 5-entry bind group fails).
  if (tok >= p1.b) {
    return;
  }
  let pr = p0.np * p0.r;
  for (var i = lid.x; i < pr; i += 64u) {
    ts[i] = t[tok * pr + i];
  }
  workgroupBarrier();
  let row = wid.x * 64u + lid.x;
  if (row >= p0.d_out) {
    return;
  }
  let rb = p0.r / 8u;
  var yv = 0.0;
  for (var p = 0u; p < p0.np; p++) {
    let base = path_base(p);
    let uoff = base + 2u * (p0.d_out + p0.d_in + p0.r);
    let a0 = uoff + row * rb;
    var s = 0.0;
    for (var kb = 0u; kb < rb; kb++) {
      let a = a0 + kb;
      let byte = (w[a >> 2u] >> ((a & 3u) * 8u)) & 0xffu;
      let tb = p * p0.r + kb * 8u;
      for (var bt = 0u; bt < 8u; bt++) {
        let tv = ts[tb + bt];
        s += select(-tv, tv, ((byte >> bt) & 1u) != 0u);
      }
    }
    yv += f16at(base + 2u * row) * s;
  }
  y[tok * p0.d_out + row] = yv;
}
"#;

struct BlrPipes {
    v: wgpu::ComputePipeline,
    u: wgpu::ComputePipeline,
}

/// Pipelines and the latent scratch, per device context.
struct BlrState {
    pipes: Arc<BlrPipes>,
    t: Option<(wgpu::Buffer, u64)>,
}

fn states() -> &'static std::sync::Mutex<std::collections::HashMap<usize, BlrState>> {
    static S: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<usize, BlrState>>> =
        std::sync::OnceLock::new();
    S.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn build_pipes(c: &Ctx) -> BlrPipes {
    let mk = |label: &str, src: &str, ep: &str| {
        let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(format!("{src}\n{COMMON}").into()),
        });
        c.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &m,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            })
    };
    BlrPipes {
        v: mk("blr_v", V_SRC, "blr_v"),
        u: mk("blr_u", U_SRC, "blr_u"),
    }
}

/// The kernels' shape contract: whole 32-column sign words, 4-row f16
/// alignment of the scale sections, one token's latents in workgroup memory.
fn shape_ok(rows: usize, cols: usize, r: usize, np: usize, b: usize) -> bool {
    b > 0 && b <= 65_535 && cols % 32 == 0 && rows % 4 == 0 && np * r <= MAX_PR
}

/// `out[b·rows] = xs[b·cols] · Wᵀ` for a `blr` tensor on the device. The
/// payload goes into the resident weight cache on first use.
pub fn blr_matmat(t: &crate::blr::BlrTensor, xs: &[f32], b: usize, out: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    let _op = c.op_gate.enter();
    let core = &t.core;
    let (rows, cols, r, np) = (core.rows, core.cols, core.rank, core.paths);
    if !shape_ok(rows, cols, r, np, b) || xs.len() < b * cols || out.len() < b * rows {
        return false;
    }
    let model = &t.model;
    let entry = &model.tensors[t.idx];
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    let plen = entry.nbytes as usize;
    if plen % 4 != 0 || abs + plen > bytes.len() {
        return false;
    }
    let Some(q_buf) = weight_buffer_l(
        c,
        (model.uid() as usize, t.idx),
        &bytes[abs..abs + plen],
        layer_of_name(&entry.name),
    ) else {
        return false;
    };
    dispatch_blr(c, &q_buf, (rows, cols, r, np), xs, b, out)
}

/// Both stages over an uploaded payload `q_buf`; `dims = (rows, cols, r, P)`.
/// The caller holds the op gate.
fn dispatch_blr(
    c: &Ctx,
    q_buf: &wgpu::Buffer,
    dims: (usize, usize, usize, usize),
    xs: &[f32],
    b: usize,
    out: &mut [f32],
) -> bool {
    let (rows, cols, r, np) = dims;
    let key = c as *const Ctx as usize;
    let (pipes, t_buf) = {
        let mut st = states().lock().unwrap();
        let s = st.entry(key).or_insert_with(|| BlrState {
            pipes: Arc::new(build_pipes(c)),
            t: None,
        });
        let need = (b * np * r * 4) as u64;
        let tb = Scratch::ensure(
            &c.device,
            &mut s.t,
            need,
            wgpu::BufferUsages::STORAGE,
            "blr-t",
        );
        (s.pipes.clone(), tb)
    };
    let p0 = uniform_u32x4(c, [rows as u32, cols as u32, r as u32, np as u32]);
    let p1 = uniform_u32x4(c, [b as u32, 0, 0, 0]);
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (b * cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "blr-xs",
    );
    c.queue
        .write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..b * cols]));
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        (b * rows * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "blr-y",
    );
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        (b * rows * 4) as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "blr-stage",
    );
    let bind_v = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blr-v"),
        layout: &pipes.v.get_bind_group_layout(0),
        entries: &[
            bind_buf(0, q_buf),
            bind_buf(1, &xs_buf),
            bind_buf(2, &t_buf),
            bind_buf(3, &p0),
            bind_buf(4, &p1),
        ],
    });
    let bind_u = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blr-u"),
        layout: &pipes.u.get_bind_group_layout(0),
        entries: &[
            bind_buf(0, q_buf),
            bind_buf(1, &t_buf),
            bind_buf(2, &y_buf),
            bind_buf(3, &p0),
            bind_buf(4, &p1),
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("blr") });
    {
        let mut pass = begin_pass(&mut enc);
        pass.set_pipeline(&pipes.v);
        pass.set_bind_group(0, &bind_v, &[]);
        pass.dispatch_workgroups((np * r) as u32, b.div_ceil(TB) as u32, 1);
    }
    {
        let mut pass = begin_pass(&mut enc);
        pass.set_pipeline(&pipes.u);
        pass.set_bind_group(0, &bind_u, &[]);
        pass.dispatch_workgroups(rows.div_ceil(64) as u32, b as u32, 1);
    }
    readback(
        c,
        enc,
        &y_buf,
        &stage_buf,
        (b * rows * 4) as u64,
        &mut out[..b * rows],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::DeviceExt;

    /// The device kernels against the CPU kernels (`crate::blr`) on the same
    /// payload: decode (b = 1) and batched (b = 3, 4, 17), P = 1 and P = 2,
    /// the r = 8 edge, and a non-multiple-of-64 row count.
    #[test]
    fn wgpu_blr_matches_cpu_kernels() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping blr parity test");
            return;
        };
        let _op = c.op_gate.enter();
        let mut rng = crate::blr::tests::Rng(23);
        for &(d_out, d_in, r, np) in &[
            (64usize, 96usize, 8usize, 1usize),
            (100, 64, 24, 2),
            (256, 512, 40, 2),
            (1024, 256, 64, 1),
        ] {
            let bytes = crate::blr::tests::make(&mut rng, d_out, d_in, r, np);
            let core = crate::blr::BlrCore::parse(&[d_out, d_in], &bytes).unwrap();
            let q_buf = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("blr-test"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
            for &b in &[1usize, 3, 4, 17] {
                let xs: Vec<f32> = (0..b * d_in).map(|_| rng.f()).collect();
                let mut want = vec![0f32; b * d_out];
                core.matmat(&bytes, &xs, b, &mut want, None);
                let mut got = vec![0f32; b * d_out];
                assert!(
                    dispatch_blr(c, &q_buf, (d_out, d_in, r, np), &xs, b, &mut got),
                    "{d_out}x{d_in} r{r} P{np} b{b}: dispatch refused"
                );
                let scale = want.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-30);
                let err = want
                    .iter()
                    .zip(&got)
                    .map(|(a, g)| (a - g) * (a - g))
                    .sum::<f32>()
                    .sqrt();
                assert!(
                    err / scale < 1e-5,
                    "{d_out}x{d_in} r{r} P{np} b{b}: wgpu vs CPU rel {}",
                    err / scale
                );
            }
        }
        // Outside the contract the device declines (the CPU kernel runs).
        assert!(!shape_ok(64, 48, 8, 1, 1), "cols % 32");
        assert!(!shape_ok(66, 64, 8, 1, 1), "rows % 4");
        assert!(!shape_ok(64, 4096, 1032, 2, 1), "P·r > workgroup staging");
    }
}
