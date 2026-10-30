//! Device encoder for the Nemotron 3.5 Transcribe audio towers (the ASR
//! transformer and the Sortformer diarization transformer).
//!
//! One submit per encoder call: feature-stacking projection, the embedding
//! LayerNorm, every pre-norm block (LayerNorm, fused q/k/v GEMM, per-head
//! q/k LayerNorm + RoPE, masked bidirectional attention, output GEMM with
//! bias and residual, LayerNorm, GELU MLP with residual) and the final
//! LayerNorm. Weights stay bf16 (the release's own values) and every GEMM
//! accumulates in f32, which is the arithmetic of the fp32 reference tower.
//! The RoPE angles come from the host (`f32` sin/cos of `pos · inv_freq`,
//! as the reference computes them) instead of the device's approximations.

use super::*;

const AE_WGSL: &str = r#"
// ── Y[n·m] (=|+=) X[n·k] · W[m·k]ᵀ (+bias) (gelu); W bf16 pairs ──
struct GmP { n: u32, m: u32, k: u32, flags: u32 };
@group(0) @binding(0) var<storage, read>       gm_x : array<f32>;
@group(0) @binding(1) var<storage, read>       gm_w : array<u32>;
@group(0) @binding(2) var<storage, read>       gm_b : array<f32>;
@group(0) @binding(3) var<storage, read_write> gm_y : array<f32>;
@group(0) @binding(4) var<uniform>             gm_p : GmP;
var<workgroup> gx: array<f32, 1024>;
var<workgroup> gw: array<f32, 1024>;
fn ae_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t + 0.2548296) * t) * exp(-a * a);
    return select(y, -y, x < 0.0);
}
fn ae_put(r: u32, c: u32, v0: f32, m: u32) {
    var v = v0;
    if ((gm_p.flags & 1u) != 0u) { v = v + gm_b[c]; }
    if ((gm_p.flags & 2u) != 0u) { v = 0.5 * v * (1.0 + ae_erf(v * 0.70710678)); }
    if ((gm_p.flags & 4u) != 0u) { v = v + gm_y[r * m + c]; }
    gm_y[r * m + c] = v;
}
fn ae_store(r: u32, c: u32, v: vec4<f32>, n: u32, m: u32) {
    if (r >= n) { return; }
    if (c < m) { ae_put(r, c, v.x, m); }
    if (c + 1u < m) { ae_put(r, c + 1u, v.y, m); }
    if (c + 2u < m) { ae_put(r, c + 2u, v.z, m); }
    if (c + 3u < m) { ae_put(r, c + 3u, v.w, m); }
}
// 128×128 output tile, k step 8; thread (tx, ty) owns rows ty·4+i, 64+ty·4+i
// and columns tx·4+j, 64+tx·4+j: 16 vec4 accumulators in registers (no
// private arrays: they would spill), vec4 loads from [k][128] shared tiles.
@compute @workgroup_size(256)
fn ae_gemm(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) t: u32) {
    let n = gm_p.n;
    let m = gm_p.m;
    let kk = gm_p.k;
    let row0 = wid.y * 128u;
    let col0 = wid.x * 128u;
    let tx = t % 16u;
    let ty = t / 16u;
    var c00 = vec4<f32>(0.0);
    var c01 = vec4<f32>(0.0);
    var c10 = vec4<f32>(0.0);
    var c11 = vec4<f32>(0.0);
    var c20 = vec4<f32>(0.0);
    var c21 = vec4<f32>(0.0);
    var c30 = vec4<f32>(0.0);
    var c31 = vec4<f32>(0.0);
    var c40 = vec4<f32>(0.0);
    var c41 = vec4<f32>(0.0);
    var c50 = vec4<f32>(0.0);
    var c51 = vec4<f32>(0.0);
    var c60 = vec4<f32>(0.0);
    var c61 = vec4<f32>(0.0);
    var c70 = vec4<f32>(0.0);
    var c71 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < kk; k0 = k0 + 8u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let e = t + i * 256u;
            let r = e / 8u;
            let c = e % 8u;
            let gr = row0 + r;
            var v = 0.0;
            if (gr < n) { v = gm_x[gr * kk + k0 + c]; }
            gx[c * 128u + r] = v;
        }
        for (var i = 0u; i < 2u; i = i + 1u) {
            let e = t + i * 256u;
            let r = e / 4u;
            let c2 = e % 4u;
            let gc = col0 + r;
            var w = 0u;
            if (gc < m) { w = gm_w[(gc * kk + k0) / 2u + c2]; }
            gw[(c2 * 2u) * 128u + r] = bitcast<f32>(w << 16u);
            gw[(c2 * 2u + 1u) * 128u + r] = bitcast<f32>(w & 0xFFFF0000u);
        }
        workgroupBarrier();
        for (var q = 0u; q < 8u; q = q + 1u) {
            let a0 = vec4<f32>(gx[q * 128u + ty * 4u], gx[q * 128u + ty * 4u + 1u], gx[q * 128u + ty * 4u + 2u], gx[q * 128u + ty * 4u + 3u]);
            let a1 = vec4<f32>(gx[q * 128u + 64u + ty * 4u], gx[q * 128u + 64u + ty * 4u + 1u], gx[q * 128u + 64u + ty * 4u + 2u], gx[q * 128u + 64u + ty * 4u + 3u]);
            let b0 = vec4<f32>(gw[q * 128u + tx * 4u], gw[q * 128u + tx * 4u + 1u], gw[q * 128u + tx * 4u + 2u], gw[q * 128u + tx * 4u + 3u]);
            let b1 = vec4<f32>(gw[q * 128u + 64u + tx * 4u], gw[q * 128u + 64u + tx * 4u + 1u], gw[q * 128u + 64u + tx * 4u + 2u], gw[q * 128u + 64u + tx * 4u + 3u]);
            c00 = c00 + a0.x * b0;
            c01 = c01 + a0.x * b1;
            c10 = c10 + a0.y * b0;
            c11 = c11 + a0.y * b1;
            c20 = c20 + a0.z * b0;
            c21 = c21 + a0.z * b1;
            c30 = c30 + a0.w * b0;
            c31 = c31 + a0.w * b1;
            c40 = c40 + a1.x * b0;
            c41 = c41 + a1.x * b1;
            c50 = c50 + a1.y * b0;
            c51 = c51 + a1.y * b1;
            c60 = c60 + a1.z * b0;
            c61 = c61 + a1.z * b1;
            c70 = c70 + a1.w * b0;
            c71 = c71 + a1.w * b1;
        }
        workgroupBarrier();
    }
    let base_r = row0 + ty * 4u;
    let base_c = col0 + tx * 4u;
    ae_store(base_r + 0u, base_c + 0u, c00, n, m);
    ae_store(base_r + 0u, base_c + 64u, c01, n, m);
    ae_store(base_r + 1u, base_c + 0u, c10, n, m);
    ae_store(base_r + 1u, base_c + 64u, c11, n, m);
    ae_store(base_r + 2u, base_c + 0u, c20, n, m);
    ae_store(base_r + 2u, base_c + 64u, c21, n, m);
    ae_store(base_r + 3u, base_c + 0u, c30, n, m);
    ae_store(base_r + 3u, base_c + 64u, c31, n, m);
    ae_store(base_r + 64u, base_c + 0u, c40, n, m);
    ae_store(base_r + 64u, base_c + 64u, c41, n, m);
    ae_store(base_r + 65u, base_c + 0u, c50, n, m);
    ae_store(base_r + 65u, base_c + 64u, c51, n, m);
    ae_store(base_r + 66u, base_c + 0u, c60, n, m);
    ae_store(base_r + 66u, base_c + 64u, c61, n, m);
    ae_store(base_r + 67u, base_c + 0u, c70, n, m);
    ae_store(base_r + 67u, base_c + 64u, c71, n, m);
}

// ── row LayerNorm: y = (x − μ)/σ · w + b, eps 1e-5 ──
struct LnP { d: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       ln_x : array<f32>;
@group(0) @binding(1) var<storage, read>       ln_w : array<f32>;
@group(0) @binding(2) var<storage, read_write> ln_y : array<f32>;
@group(0) @binding(3) var<uniform>             ln_p : LnP;
var<workgroup> ln_s: array<f32, 256>;
@compute @workgroup_size(256)
fn ae_ln(@builtin(workgroup_id) wid: vec3<u32>,
         @builtin(local_invocation_index) lid: u32) {
    let d = ln_p.d;
    let base = wid.x * d;
    var acc = 0.0;
    for (var i = lid; i < d; i = i + 256u) { acc = acc + ln_x[base + i]; }
    ln_s[lid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { ln_s[lid] = ln_s[lid] + ln_s[lid + s]; }
        workgroupBarrier();
    }
    let mean = ln_s[0] / f32(d);
    workgroupBarrier();
    acc = 0.0;
    for (var i = lid; i < d; i = i + 256u) {
        let c = ln_x[base + i] - mean;
        acc = acc + c * c;
    }
    ln_s[lid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { ln_s[lid] = ln_s[lid] + ln_s[lid + s]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(ln_s[0] / f32(d) + 1e-5);
    for (var i = lid; i < d; i = i + 256u) {
        ln_y[base + i] = (ln_x[base + i] - mean) * inv * ln_w[i] + ln_w[d + i];
    }
}

// ── per-head q/k LayerNorm (qk != 0) + RoPE on the first `rot` dims,
// in place in the packed [q | k | v] rows; cs = [cos | sin] [n · rot/2] ──
struct QrP { n: u32, heads: u32, hd: u32, rot: u32, qk: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> qr_x  : array<f32>;
@group(0) @binding(1) var<storage, read>       qr_qn : array<f32>;
@group(0) @binding(2) var<storage, read>       qr_kn : array<f32>;
@group(0) @binding(3) var<storage, read>       qr_cs : array<f32>;
@group(0) @binding(4) var<uniform>             qr_p  : QrP;
var<workgroup> qr_s: array<f32, 128>;
var<workgroup> qr_v: array<f32, 128>;
@compute @workgroup_size(128)
fn ae_qk(@builtin(workgroup_id) wid: vec3<u32>,
         @builtin(local_invocation_index) i: u32) {
    let t = wid.x;
    let h = wid.y;
    let which = wid.z;
    let hd = qr_p.hd;
    let d = qr_p.heads * hd;
    let base = t * 3u * d + which * d + h * hd;
    var v = 0.0;
    if (i < hd) { v = qr_x[base + i]; }
    if (qr_p.qk != 0u) {
        qr_s[i] = v;
        workgroupBarrier();
        for (var s = 64u; s > 0u; s = s >> 1u) {
            if (i < s) { qr_s[i] = qr_s[i] + qr_s[i + s]; }
            workgroupBarrier();
        }
        let mean = qr_s[0] / f32(hd);
        workgroupBarrier();
        let c = select(0.0, v - mean, i < hd);
        qr_s[i] = c * c;
        workgroupBarrier();
        for (var s = 64u; s > 0u; s = s >> 1u) {
            if (i < s) { qr_s[i] = qr_s[i] + qr_s[i + s]; }
            workgroupBarrier();
        }
        let inv = inverseSqrt(qr_s[0] / f32(hd) + 1e-5);
        if (i < hd) {
            if (which == 0u) {
                v = (v - mean) * inv * qr_qn[i] + qr_qn[hd + i];
            } else {
                v = (v - mean) * inv * qr_kn[i] + qr_kn[hd + i];
            }
        }
    }
    qr_v[i] = v;
    workgroupBarrier();
    let half = qr_p.rot / 2u;
    if (i < hd) {
        var o = v;
        if (i < half) {
            let c = qr_cs[t * half + i];
            let s = qr_cs[qr_p.n * half + t * half + i];
            o = v * c - qr_v[i + half] * s;
        } else if (i < qr_p.rot) {
            let j = i - half;
            let c = qr_cs[t * half + j];
            let s = qr_cs[qr_p.n * half + t * half + j];
            o = v * c + qr_v[j] * s;
        }
        qr_x[base + i] = o;
    }
}

// ── masked bidirectional attention, workgroup per (query, head) ──
struct EaP { n: u32, heads: u32, hd: u32, scale: f32 };
@group(0) @binding(0) var<storage, read>       ea_x     : array<f32>;
@group(0) @binding(1) var<storage, read>       ea_valid : array<u32>;
@group(0) @binding(2) var<storage, read_write> ea_o     : array<f32>;
@group(0) @binding(3) var<uniform>             ea_p     : EaP;
var<workgroup> ea_s: array<f32, 1024>;
var<workgroup> ea_r: array<f32, 128>;
var<workgroup> ea_q: array<f32, 128>;
@compute @workgroup_size(128)
fn ae_attn(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    let t = wid.x;
    let h = wid.y;
    let n = ea_p.n;
    let hd = ea_p.hd;
    let d = ea_p.heads * hd;
    if (lid < hd) { ea_q[lid] = ea_x[t * 3u * d + h * hd + lid]; }
    workgroupBarrier();
    var mx = -3.0e38;
    for (var j = lid; j < n; j = j + 128u) {
        var s = -3.0e38;
        if (ea_valid[j] != 0u) {
            let kb = j * 3u * d + d + h * hd;
            var acc = 0.0;
            for (var c = 0u; c < hd; c = c + 1u) { acc = acc + ea_q[c] * ea_x[kb + c]; }
            s = acc * ea_p.scale;
        }
        ea_s[j] = s;
        mx = max(mx, s);
    }
    ea_r[lid] = mx;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (lid < s) { ea_r[lid] = max(ea_r[lid], ea_r[lid + s]); }
        workgroupBarrier();
    }
    let gmax = ea_r[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var j = lid; j < n; j = j + 128u) {
        var e = 0.0;
        if (ea_valid[j] != 0u) { e = exp(ea_s[j] - gmax); }
        ea_s[j] = e;
        sum = sum + e;
    }
    ea_r[lid] = sum;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (lid < s) { ea_r[lid] = ea_r[lid] + ea_r[lid + s]; }
        workgroupBarrier();
    }
    let inv = 1.0 / ea_r[0];
    if (lid < hd) {
        var acc = 0.0;
        for (var j = 0u; j < n; j = j + 1u) {
            acc = acc + ea_s[j] * ea_x[j * 3u * d + 2u * d + h * hd + lid];
        }
        ea_o[t * d + h * hd + lid] = acc * inv;
    }
}
"#;

struct Pipes {
    gemm: wgpu::ComputePipeline,
    ln: wgpu::ComputePipeline,
    qk: wgpu::ComputePipeline,
    attn: wgpu::ComputePipeline,
}

static PIPES: std::sync::OnceLock<Option<Pipes>> = std::sync::OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cmf-nemotron-audio"),
                source: wgpu::ShaderSource::Wgsl(AE_WGSL.into()),
            });
            let pipe = |ep: &str| {
                c.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(ep),
                        layout: None,
                        module: &module,
                        entry_point: Some(ep),
                        compilation_options: pco(),
                        cache: c.pipeline_cache.as_ref(),
                    })
            };
            let p = Pipes {
                gemm: pipe("ae_gemm"),
                ln: pipe("ae_ln"),
                qk: pipe("ae_qk"),
                attn: pipe("ae_attn"),
            };
            if let Some(e) = pollster::block_on(scope.pop()) {
                tracing::warn!("nemotron audio encoder: module rejected ({e})");
                return None;
            }
            Some(p)
        })
        .as_ref()
}

/// One encoder block's weights for [`AudioEncoder::build`]: bf16 matrices
/// (raw payloads), LayerNorms as `[w | b]` f32.
pub struct AeLayerDesc<'a> {
    pub ln1: Vec<f32>,
    pub qkv: &'a [u8],
    pub qn: Option<Vec<f32>>,
    pub kn: Option<Vec<f32>>,
    pub out: &'a [u8],
    pub out_b: Vec<f32>,
    pub ln2: Vec<f32>,
    pub fc1: &'a [u8],
    pub fc1_b: Vec<f32>,
    pub fc2: &'a [u8],
    pub fc2_b: Vec<f32>,
}

pub struct AeDesc<'a> {
    pub d: usize,
    pub heads: usize,
    pub rot: usize,
    pub inter: usize,
    /// stacked-feature width of the input projection
    pub stack_in: usize,
    pub proj: &'a [u8],
    pub embed_norm: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub layers: Vec<AeLayerDesc<'a>>,
}

struct Layer {
    ln1: wgpu::Buffer,
    qkv: wgpu::Buffer,
    qn: wgpu::Buffer,
    kn: wgpu::Buffer,
    qk: bool,
    out: wgpu::Buffer,
    out_b: wgpu::Buffer,
    ln2: wgpu::Buffer,
    fc1: wgpu::Buffer,
    fc1_b: wgpu::Buffer,
    fc2: wgpu::Buffer,
    fc2_b: wgpu::Buffer,
}

pub struct AudioEncoder {
    c: &'static Ctx,
    p: &'static Pipes,
    d: usize,
    heads: usize,
    rot: usize,
    inter: usize,
    stack_in: usize,
    proj: wgpu::Buffer,
    embed_norm: wgpu::Buffer,
    final_norm: wgpu::Buffer,
    layers: Vec<Layer>,
    dummy: wgpu::Buffer,
    pub bytes: u64,
}

impl AudioEncoder {
    pub fn build(d: &AeDesc) -> Option<Self> {
        let c = ctx()?;
        let p = pipes(c)?;
        let hd = d.d / d.heads;
        if hd > 128 || d.d % 8 != 0 || d.inter % 8 != 0 || d.stack_in % 8 != 0 {
            return None;
        }
        let total = std::cell::Cell::new(0u64);
        let up = |label: &str, data: &[u8]| -> wgpu::Buffer {
            total.set(total.get() + data.len() as u64);
            let mut v = data.to_vec();
            v.resize(data.len().div_ceil(16) * 16 + 16, 0);
            c.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &v,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let upf = |label: &str, v: &[f32]| up(label, bytemuck::cast_slice(v));
        let mut layers = Vec::with_capacity(d.layers.len());
        for l in &d.layers {
            layers.push(Layer {
                ln1: upf("ae-ln1", &l.ln1),
                qkv: up("ae-qkv", l.qkv),
                qn: upf("ae-qn", l.qn.as_deref().unwrap_or(&[0.0; 4])),
                kn: upf("ae-kn", l.kn.as_deref().unwrap_or(&[0.0; 4])),
                qk: l.qn.is_some(),
                out: up("ae-out", l.out),
                out_b: upf("ae-outb", &l.out_b),
                ln2: upf("ae-ln2", &l.ln2),
                fc1: up("ae-fc1", l.fc1),
                fc1_b: upf("ae-fc1b", &l.fc1_b),
                fc2: up("ae-fc2", l.fc2),
                fc2_b: upf("ae-fc2b", &l.fc2_b),
            });
        }
        Some(Self {
            c,
            p,
            d: d.d,
            heads: d.heads,
            rot: d.rot,
            inter: d.inter,
            stack_in: d.stack_in,
            proj: up("ae-proj", d.proj),
            embed_norm: upf("ae-en", &d.embed_norm),
            final_norm: upf("ae-fn", &d.final_norm),
            layers,
            dummy: upf("ae-dummy", &[0.0; 4]),
            bytes: total.get(),
        })
    }

    fn run_pass(
        &self,
        pass: &mut wgpu::ComputePass,
        pipe: &wgpu::ComputePipeline,
        bufs: &[&wgpu::Buffer],
        u: [u32; 8],
        g: (u32, u32, u32),
    ) {
        let ub = uniform_u32x8(self.c, u);
        let layout = pipe.get_bind_group_layout(0);
        let mut entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| bind_buf(i as u32, b))
            .collect();
        entries.push(bind_buf(bufs.len() as u32, &ub));
        let bg = self.c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(g.0, g.1, g.2);
    }

    fn gemm(
        &self,
        pass: &mut wgpu::ComputePass,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        b: Option<&wgpu::Buffer>,
        y: &wgpu::Buffer,
        n: usize,
        m: usize,
        k: usize,
        flags: u32,
    ) {
        let flags = flags | b.is_some() as u32;
        self.run_pass(
            pass,
            &self.p.gemm,
            &[x, w, b.unwrap_or(&self.dummy), y],
            [n as u32, m as u32, k as u32, flags, 0, 0, 0, 0],
            (m.div_ceil(128) as u32, n.div_ceil(128) as u32, 1),
        );
    }

    /// Run the encoder over `n` rows: stacked features `[n · stack_in]`
    /// (`embedded = false`) or embedded rows `[n · d]` (`true`); `valid`
    /// masks keys. Positions restart at 0. Returns `[n · d]`.
    pub fn run(&self, x: &[f32], n: usize, embedded: bool, valid: &[bool]) -> Option<Vec<f32>> {
        if n == 0 || n > 1024 || valid.len() != n {
            return None;
        }
        let c = self.c;
        let d = self.d;
        let hd = d / self.heads;
        let buf = |label: &str, floats: usize| {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (floats.max(4) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let width_in = if embedded { d } else { self.stack_in };
        if x.len() != n * width_in {
            return None;
        }
        let xin = buf("ae-x", n * width_in);
        c.queue.write_buffer(&xin, 0, bytemuck::cast_slice(x));
        let h = buf("ae-h", n * d);
        let a = buf("ae-a", n * d);
        let qkv = buf("ae-qkv", n * 3 * d);
        let att = buf("ae-att", n * d);
        let f = buf("ae-f", n * self.inter);
        let vmask: Vec<u32> = valid.iter().map(|&v| v as u32).collect();
        let vb = buf("ae-valid", n);
        c.queue.write_buffer(&vb, 0, bytemuck::cast_slice(&vmask));
        // RoPE table [cos n·half | sin n·half], as the host computes it
        let half = self.rot / 2;
        let mut cs = vec![0f32; 2 * n * half.max(1)];
        for t in 0..n {
            for i in 0..half {
                let inv = 1.0 / 10000f32.powf((2 * i) as f32 / self.rot as f32);
                let ang = t as f32 * inv;
                cs[t * half + i] = ang.cos();
                cs[n * half + t * half + i] = ang.sin();
            }
        }
        let csb = buf("ae-cs", cs.len());
        c.queue.write_buffer(&csb, 0, bytemuck::cast_slice(&cs));
        let stage = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ae-stage"),
            size: (n * d * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ae") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ae"),
                timestamp_writes: None,
            });
            let p = self.p;
            let src = if embedded {
                &xin
            } else {
                self.gemm(
                    &mut pass,
                    &xin,
                    &self.proj,
                    None,
                    &a,
                    n,
                    d,
                    self.stack_in,
                    0,
                );
                &a
            };
            self.run_pass(
                &mut pass,
                &p.ln,
                &[src, &self.embed_norm, &h],
                [d as u32, 0, 0, 0, 0, 0, 0, 0],
                (n as u32, 1, 1),
            );
            let scale = (1.0 / (hd as f32).sqrt()).to_bits();
            for l in &self.layers {
                self.run_pass(
                    &mut pass,
                    &p.ln,
                    &[&h, &l.ln1, &a],
                    [d as u32, 0, 0, 0, 0, 0, 0, 0],
                    (n as u32, 1, 1),
                );
                self.gemm(&mut pass, &a, &l.qkv, None, &qkv, n, 3 * d, d, 0);
                self.run_pass(
                    &mut pass,
                    &p.qk,
                    &[&qkv, &l.qn, &l.kn, &csb],
                    [
                        n as u32,
                        self.heads as u32,
                        hd as u32,
                        self.rot as u32,
                        l.qk as u32,
                        0,
                        0,
                        0,
                    ],
                    (n as u32, self.heads as u32, 2),
                );
                self.run_pass(
                    &mut pass,
                    &p.attn,
                    &[&qkv, &vb, &att],
                    [n as u32, self.heads as u32, hd as u32, scale, 0, 0, 0, 0],
                    (n as u32, self.heads as u32, 1),
                );
                self.gemm(&mut pass, &att, &l.out, Some(&l.out_b), &h, n, d, d, 4);
                self.run_pass(
                    &mut pass,
                    &p.ln,
                    &[&h, &l.ln2, &a],
                    [d as u32, 0, 0, 0, 0, 0, 0, 0],
                    (n as u32, 1, 1),
                );
                self.gemm(
                    &mut pass,
                    &a,
                    &l.fc1,
                    Some(&l.fc1_b),
                    &f,
                    n,
                    self.inter,
                    d,
                    2,
                );
                self.gemm(
                    &mut pass,
                    &f,
                    &l.fc2,
                    Some(&l.fc2_b),
                    &h,
                    n,
                    d,
                    self.inter,
                    4,
                );
            }
            self.run_pass(
                &mut pass,
                &p.ln,
                &[&h, &self.final_norm, &a],
                [d as u32, 0, 0, 0, 0, 0, 0, 0],
                (n as u32, 1, 1),
            );
        }
        enc.copy_buffer_to_buffer(&a, 0, &stage, 0, (n * d * 4) as u64);
        c.queue.submit(Some(enc.finish()));
        let slice = stage.slice(..);
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let d2 = done.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            if r.is_ok() {
                d2.store(true, std::sync::atomic::Ordering::Release);
            }
        });
        if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err()
            || !done.load(std::sync::atomic::Ordering::Acquire)
        {
            return None;
        }
        let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range().ok()?).to_vec();
        stage.unmap();
        Some(out)
    }
}

#[cfg(test)]
mod shader_tests {
    /// The Nemotron frame, prefill and audio modules parse and validate
    /// (naga) without a device.
    #[test]
    fn nemotron_shaders_validate() {
        let frame = super::super::nemotron::nemo_wgsl();
        let verify = super::super::nemotron_verify::ver_wgsl();
        for (name, src) in [
            ("nemotron-audio", super::AE_WGSL),
            ("nemotron-frame", frame.as_str()),
            ("nemotron-verify", verify.as_str()),
            ("nemotron-prefill", super::super::nemotron_prefill::PB_WGSL),
        ] {
            let module = wgpu::naga::front::wgsl::parse_str(src)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
