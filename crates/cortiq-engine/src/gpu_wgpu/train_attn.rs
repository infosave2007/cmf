//! Causal GQA attention of the trainer on the matrix units.
//!
//! The window is short (T = 512 for the 27B student), so the scores are
//! materialized per (window, head) instead of streamed: every product is a
//! batched f16 GEMM with f32 accumulators (`att_mm`), the softmax and the
//! score gradient run as row kernels in f32.
//!
//! ```text
//! forward   Q16 = f16(q·sc)  K16 = f16(k)  V16 = f16(v)
//!           S = Q16·K16ᵀ (f32)   P = softmax(S) (masked)   LSE
//!           O = P16·V16
//! backward  dO16 = f16(dO·gs)   dP = dO16·V16ᵀ/gs   D = Σ dO⊙O
//!           dS = exp(S − LSE)⊙(dP − D)   dS16 = f16(dS·gs)
//!           dQ = dS16·K16·sc/gs   dK = Σ_h dS16ᵀ·Q16/gs   dV = Σ_h P16ᵀ·dO16/gs
//! ```
//!
//! `sc` = 1/√hd (a power of two for hd = 256: exact), `gs` the trainer's
//! power-of-two gradient scale that keeps f16 operands out of the
//! subnormals. Tiles above the diagonal are skipped; P and dS are written
//! as zeros past the diagonal inside the diagonal tiles, so the K loops of
//! O, dQ (keys ≤ the tile's last query) and dK, dV (queries ≥ the tile's
//! first key) read only zeros beyond the mask.

use super::Ctx;
use super::train_mm::{self as tm, Op};
use super::train_ops::{AttnGeom, Disp, wg_fold};
use std::sync::OnceLock;

/// Output tile of `att_mm` (rows × columns), K slice.
const BM: usize = 64;
const BN: usize = 64;
const BK: u32 = 32;
const LDS: u32 = BK + 8;

/// One `att_mm` entry point: operand layouts (k-contiguous or not) and the
/// causal rule (1: skip tiles above the diagonal; 2: K ≤ the M tile's end;
/// 3: K ≥ the M tile's start).
const VARIANTS: [(&str, bool, bool, u32); 3] = [("att_mm_s", true, true, 1), ("att_mm_o", true, false, 2), ("att_mm_d", false, false, 3)];

/// Side of the cooperative fragments: NVIDIA's 16×16, or Apple's 8×8
/// simdgroup matrices on Metal (each subgroup still owns a 32×32 block).
fn frag() -> u32 {
    tm::frag_side()
}

fn src() -> String {
    use std::fmt::Write;
    let fr = frag();
    let nf = 32 / fr;
    let mut s = String::new();
    let _ = writeln!(s, "enable f16;\nenable wgpu_cooperative_matrix;\ndiagnostic(off, derivative_uniformity);");
    let _ = writeln!(
        s,
        "struct AP {{
    m: u32, n: u32, k: u32, nz: u32,
    a_ld: u32, a_sb: u32, a_sh: u32, a_hdiv: u32,
    b_ld: u32, b_sb: u32, b_sh: u32, b_hdiv: u32,
    c_ld: u32, c_sb: u32, c_sh: u32, c_hdiv: u32,
    nchunk: u32, a_ck: u32, b_ck: u32, oscale: f32,
    t: u32, nh: u32, gs: f32, pad: u32,
}}
@group(0) @binding(0) var<uniform> p: AP;
@group(0) @binding(1) var<storage, read> AV: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> BV: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> CO: array<f32>;
@group(0) @binding(4) var<storage, read_write> SS: array<f32>;     // [B·nh][T][T] scores
@group(0) @binding(5) var<storage, read_write> P16: array<f16>;    // [B·nh][T][T] P or dS (f16)
@group(0) @binding(6) var<storage, read_write> LSE: array<f32>;    // [B·T][nh]
@group(0) @binding(7) var<storage, read_write> DSUM: array<f32>;   // [B·T][nh]
@group(0) @binding(8) var<storage, read_write> DP: array<f32>;     // [B·nh][T][T]
var<workgroup> sa: array<f16, {}>;
var<workgroup> sb: array<f16, {}>;",
        BM as u32 * LDS,
        BN as u32 * LDS
    );
    for (name, a_kc, b_kc, causal) in VARIANTS {
        let _ = writeln!(
            s,
            "@compute @workgroup_size(128)
fn {name}(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32, @builtin(subgroup_id) sg: u32) {{
    let z = wid.z;
    let zb = z / p.nz;
    let zh = z % p.nz;
    let m0 = wid.y * {BM}u;
    let n0 = wid.x * {BN}u;
    let wy = sg / 2u;
    let wx = sg % 2u;"
        );
        if causal == 1 {
            let _ = writeln!(s, "    if (n0 > m0 + {}u) {{ return; }}", BM - 1);
        }
        for i in 0..nf {
            for j in 0..nf {
                let _ = writeln!(s, "    var c{i}{j}: coop_mat{fr}x{fr}<f32, C>;");
            }
        }
        let _ = writeln!(
            s,
            "    for (var ch = 0u; ch < p.nchunk; ch = ch + 1u) {{
        let aoff = zb * p.a_sb + (zh / p.a_hdiv) * p.a_sh + ch * p.a_ck;
        let boff = zb * p.b_sb + (zh / p.b_hdiv) * p.b_sh + ch * p.b_ck;"
        );
        match causal {
            2 => {
                let _ = writeln!(s, "        let klo = 0u;\n        let khi = min(p.k, m0 + {BM}u);");
            }
            3 => {
                let _ = writeln!(s, "        let klo = m0;\n        let khi = p.k;");
            }
            _ => {
                let _ = writeln!(s, "        let klo = 0u;\n        let khi = p.k;");
            }
        }
        let _ = writeln!(s, "        for (var k0 = klo; k0 < khi; k0 = k0 + {BK}u) {{");
        // A tile [64 m][32 k] -> sa[m][LDS]
        if a_kc {
            let _ = writeln!(
                s,
                "            for (var e = 0u; e < 4u; e = e + 1u) {{
                let v = tid + 128u * e;
                let row = v / 8u;
                let kq = v % 8u;
                let x = AV[(aoff + (m0 + row) * p.a_ld + k0 + kq * 4u) / 4u];
                let d = row * {LDS}u + kq * 4u;
                sa[d] = x.x; sa[d + 1u] = x.y; sa[d + 2u] = x.z; sa[d + 3u] = x.w;
            }}"
            );
        } else {
            let _ = writeln!(
                s,
                "            for (var e = 0u; e < 4u; e = e + 1u) {{
                let v = tid + 128u * e;
                let kk = v / 16u;
                let mq = v % 16u;
                let x = AV[(aoff + (k0 + kk) * p.a_ld + m0 + mq * 4u) / 4u];
                let d = mq * 4u * {LDS}u + kk;
                sa[d] = x.x; sa[d + {LDS}u] = x.y; sa[d + {}u] = x.z; sa[d + {}u] = x.w;
            }}",
                2 * LDS,
                3 * LDS
            );
        }
        // B tile [64 n][32 k] -> sb[n][LDS]
        if b_kc {
            let _ = writeln!(
                s,
                "            for (var e = 0u; e < 4u; e = e + 1u) {{
                let v = tid + 128u * e;
                let row = v / 8u;
                let kq = v % 8u;
                let x = BV[(boff + (n0 + row) * p.b_ld + k0 + kq * 4u) / 4u];
                let d = row * {LDS}u + kq * 4u;
                sb[d] = x.x; sb[d + 1u] = x.y; sb[d + 2u] = x.z; sb[d + 3u] = x.w;
            }}"
            );
        } else {
            let _ = writeln!(
                s,
                "            for (var e = 0u; e < 4u; e = e + 1u) {{
                let v = tid + 128u * e;
                let kk = v / 16u;
                let nq = v % 16u;
                let x = BV[(boff + (k0 + kk) * p.b_ld + n0 + nq * 4u) / 4u];
                let d = nq * 4u * {LDS}u + kk;
                sb[d] = x.x; sb[d + {LDS}u] = x.y; sb[d + {}u] = x.z; sb[d + {}u] = x.w;
            }}",
                2 * LDS,
                3 * LDS
            );
        }
        let _ = writeln!(s, "            workgroupBarrier();");
        for kk in 0..BK / fr {
            let ko = kk * fr;
            let _ = writeln!(s, "            {{");
            for i in 0..nf {
                let _ = writeln!(s, "                let a{i} = coopLoadT<coop_mat{fr}x{fr}<f16, A>>(&sa[(wy * 32u + {}u) * {LDS}u + {ko}u], {LDS}u);", i * fr);
            }
            for j in 0..nf {
                let _ = writeln!(s, "                let b{j} = coopLoad<coop_mat{fr}x{fr}<f16, B>>(&sb[(wx * 32u + {}u) * {LDS}u + {ko}u], {LDS}u);", j * fr);
            }
            for i in 0..nf {
                for j in 0..nf {
                    let _ = writeln!(s, "                c{i}{j} = coopMultiplyAdd(a{i}, b{j}, c{i}{j});");
                }
            }
            let _ = writeln!(s, "            }}");
        }
        let _ = writeln!(
            s,
            "            workgroupBarrier();
        }}
    }}
    let ldc = p.c_ld;
    let osc = p.oscale;
    let coff = zb * p.c_sb + (zh / p.c_hdiv) * p.c_sh;
    let orow = m0 + wy * 32u;
    let ocol = n0 + wx * 32u;"
        );
        for i in 0..nf {
            for j in 0..nf {
                if fr == 16 {
                    let _ = writeln!(s, "    {{ let oi = coff + (orow + {}u) * ldc + ocol + {}u; let cv = c{i}{j} * osc; coopStoreT(cv, &CO[oi], ldc); }}", i * fr, j * fr);
                } else {
                    let _ = writeln!(s, "    {{ let oi = coff + (orow + {}u) * ldc + ocol + {}u; coopStoreT(c{i}{j}, &CO[oi], ldc); }}", i * fr, j * fr);
                }
            }
        }
        if fr != 16 {
            // Metal's simdgroup matrices take no scalar product: the lanes
            // rescale the subgroup's 32×32 block after the store
            let _ = writeln!(
                s,
                "    if (osc != 1.0) {{
        storageBarrier();
        let lane = tid - sg * 32u;
        for (var e = lane; e < 1024u; e = e + 32u) {{ let oi = coff + (orow + e / 32u) * ldc + ocol + e % 32u; CO[oi] = CO[oi] * osc; }}
    }}"
            );
        }
        let _ = writeln!(s, "}}");
    }
    // Row kernels: one subgroup (32 lanes) per score row r = z·T + i,
    // eight rows per workgroup; columns j = lane + 32e (j ≤ i live), zeros
    // written up to the end of the row's diagonal tile.
    let _ = writeln!(
        s,
        "fn bmax(v: f32) -> f32 {{
    var x = max(v, subgroupShuffleXor(v, 1u));
    x = max(x, subgroupShuffleXor(x, 2u));
    x = max(x, subgroupShuffleXor(x, 4u));
    x = max(x, subgroupShuffleXor(x, 8u));
    return max(x, subgroupShuffleXor(x, 16u));
}}
fn bsum(v: f32) -> f32 {{
    var x = v + subgroupShuffleXor(v, 1u);
    x = x + subgroupShuffleXor(x, 2u);
    x = x + subgroupShuffleXor(x, 4u);
    x = x + subgroupShuffleXor(x, 8u);
    return x + subgroupShuffleXor(x, 16u);
}}

// P = softmax over j ≤ i of S (already scaled), as f16; LSE.
@compute @workgroup_size(256)
fn att_softmax(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) lane: u32) {{
    let row = (wid.x + wid.y * 65535u) * 8u + lid / 32u;
    let nrow = p.nz * p.t;
    let live = row < nrow;
    let r = select(0u, row, live);
    let z = r / p.t;
    let i = r % p.t;
    let jend = min(p.t, (i / {BM}u + 1u) * {BM}u);
    let base = r * p.t;
    // loop bounds uniform (p.t): the shuffles below need uniform control flow
    var mx = -3.0e38;
    for (var e = 0u; e < p.t; e = e + 32u) {{ let j = e + lane; if (j <= i) {{ mx = max(mx, SS[base + j]); }} }}
    mx = bmax(mx);
    var l = 0.0;
    for (var e = 0u; e < p.t; e = e + 32u) {{ let j = e + lane; if (j <= i) {{ l = l + exp(SS[base + j] - mx); }} }}
    l = bsum(l);
    let il = 1.0 / l;
    if (live) {{
        for (var j = lane; j < jend; j = j + 32u) {{
            var v = 0.0;
            if (j <= i) {{ v = exp(SS[base + j] - mx) * il; }}
            P16[base + j] = f16(v);
        }}
        if (lane == 0u) {{
            let b = z / p.nh;
            let h = z % p.nh;
            LSE[(b * p.t + i) * p.nh + h] = mx + log(l);
        }}
    }}
}}

// dS = exp(S − LSE)·(dP − D) for j ≤ i, as f16(dS·gs).
@compute @workgroup_size(256)
fn att_ds(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let row = (wid.x + wid.y * 65535u) * 8u + lid / 32u;
    let lane = lid % 32u;
    if (row >= p.nz * p.t) {{ return; }}
    let z = row / p.t;
    let i = row % p.t;
    let b = z / p.nh;
    let h = z % p.nh;
    let lse = LSE[(b * p.t + i) * p.nh + h];
    let dd = DSUM[(b * p.t + i) * p.nh + h];
    let jend = min(p.t, (i / {BM}u + 1u) * {BM}u);
    let base = row * p.t;
    for (var j = lane; j < jend; j = j + 32u) {{
        var v = 0.0;
        if (j <= i) {{ v = exp(SS[base + j] - lse) * (DP[base + j] - dd) * p.gs; }}
        P16[base + j] = f16(v);
    }}
}}"
    );
    s
}

struct Pipes {
    p: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            // Apple's SIMD groups are 32 wide; wgpu reports Metal's range as 4..64
            let sg32 = (c.adapter_info.subgroup_min_size == 32 && c.adapter_info.subgroup_max_size == 32) || c.adapter_info.backend == wgpu::Backend::Metal;
            if !tm::matrix_units()
                || !c.device.features().contains(wgpu::Features::SUBGROUP)
                || !sg32
                || std::env::var("CMF_ATT_TC").as_deref() == Ok("0")
            {
                return None;
            }
            let src = src();
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let m = super::train_ops::trainer_module(c, "train_attn", &src);
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("train_attn module rejected: {e}");
                return None;
            }
            let mut p = std::collections::HashMap::new();
            for ep in ["att_mm_s", "att_mm_o", "att_mm_d", "att_softmax", "att_ds"] {
                let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
                let pipe = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(ep),
                    layout: None,
                    module: &m,
                    entry_point: Some(ep),
                    compilation_options: crate::gpu_wgpu::pco(),
                    cache: None,
                });
                if let Some(e) = pollster::block_on(sc.pop()) {
                    tracing::warn!("train_attn/{ep} rejected: {e}");
                    return None;
                }
                p.insert(ep, pipe);
            }
            // named where they stay (the profiler keys pipelines by address)
            for (ep, pipe) in &p {
                super::train_prof::name_pipe(pipe, ep);
            }
            Some(Pipes { p })
        })
        .as_ref()
}

/// Whether the matrix-unit attention runs for this geometry.
pub fn usable(c: &Ctx, g: &AttnGeom) -> bool {
    g.t % BM == 0 && g.hd % BN == 0 && g.nh % g.nkv == 0 && pipes(c).is_some()
}

/// Device scratch of the matrix-unit attention (one layer at a time).
pub struct AttScratch {
    q16: wgpu::Buffer,
    k16: wgpu::Buffer,
    v16: wgpu::Buffer,
    do16: wgpu::Buffer,
    s: wgpu::Buffer,
    dp: wgpu::Buffer,
    p16: wgpu::Buffer,
    ds16: wgpu::Buffer,
}

impl AttScratch {
    pub fn new(c: &Ctx, g: &AttnGeom) -> Self {
        let n = g.nb * g.t;
        let (qd, kd) = (g.nh * g.hd, g.nkv * g.hd);
        let ss = g.nb * g.nh * g.t * g.t;
        AttScratch {
            q16: tm::f16_buf(c, tm::pad_rows(n) * qd),
            k16: tm::f16_buf(c, tm::pad_rows(n) * kd),
            v16: tm::f16_buf(c, tm::pad_rows(n) * kd),
            do16: tm::f16_buf(c, tm::pad_rows(n) * qd),
            s: super::train_ops::zeros_u32(c, ss),
            dp: super::train_ops::zeros_u32(c, ss),
            p16: tm::f16_buf(c, ss),
            ds16: tm::f16_buf(c, ss),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn uni(m: usize, n: usize, k: usize, nz: usize, a: [usize; 4], b: [usize; 4], cc: [usize; 4], chunk: [usize; 3], oscale: f32, g: &AttnGeom, gs: f32) -> Vec<u32> {
    let u = |x: usize| x as u32;
    vec![
        u(m), u(n), u(k), u(nz),
        u(a[0]), u(a[1]), u(a[2]), u(a[3]),
        u(b[0]), u(b[1]), u(b[2]), u(b[3]),
        u(cc[0]), u(cc[1]), u(cc[2]), u(cc[3]),
        u(chunk[0]), u(chunk[1]), u(chunk[2]), oscale.to_bits(),
        u(g.t), u(g.nh), gs.to_bits(), 0,
    ]
}

fn row_grid(rows: usize) -> (u32, u32, u32) {
    wg_fold(rows.div_ceil(8))
}

/// Forward ops: (q, k, v f32) → o f32 and LSE; S and P stay in the scratch
/// for [`bwd_ops`].
#[allow(clippy::too_many_arguments)]
pub fn fwd_ops<'a>(c: &'a Ctx, g: AttnGeom, sc: &'a AttScratch, q: &'a wgpu::Buffer, k: &'a wgpu::Buffer, v: &'a wgpu::Buffer, o: &'a wgpu::Buffer, lse: &'a wgpu::Buffer) -> Option<Vec<Op<'a>>> {
    let pp = pipes(c)?;
    let (t, hd, nh) = (g.t, g.hd, g.nh);
    let rep = g.nh / g.nkv;
    let n = g.nb * t;
    let (qd, kd) = (nh * hd, g.nkv * hd);
    let scale = 1.0 / (hd as f32).sqrt();
    let z = g.nb * nh;
    let mut ops = vec![
        Op::D(tm::cvt(c, q, None, &sc.q16, n, qd, scale)?),
        Op::D(tm::cvt(c, k, None, &sc.k16, n, kd, 1.0)?),
        Op::D(tm::cvt(c, v, None, &sc.v16, n, kd, 1.0)?),
    ];
    // S = Q16·K16ᵀ
    ops.push(Op::D(Disp {
        pipe: &pp.p["att_mm_s"],
        uni: uni(t, t, hd, nh, [qd, t * qd, hd, 1], [kd, t * kd, hd, rep], [t, nh * t * t, t * t, 1], [1, 0, 0], 1.0, &g, 1.0),
        bufs: vec![(1, &sc.q16), (2, &sc.k16), (3, &sc.s)],
        grid: ((t / BN) as u32, (t / BM) as u32, z as u32),
    }));
    ops.push(Op::D(Disp {
        pipe: &pp.p["att_softmax"],
        uni: uni(0, 0, 0, z, [0; 4], [0; 4], [0; 4], [0; 3], 1.0, &g, 1.0),
        bufs: vec![(4, &sc.s), (5, &sc.p16), (6, lse)],
        grid: row_grid(z * t),
    }));
    // O = P16·V16
    ops.push(Op::D(Disp {
        pipe: &pp.p["att_mm_o"],
        uni: uni(t, hd, t, nh, [t, nh * t * t, t * t, 1], [kd, t * kd, hd, rep], [qd, t * qd, hd, 1], [1, 0, 0], 1.0, &g, 1.0),
        bufs: vec![(1, &sc.p16), (2, &sc.v16), (3, o)],
        grid: ((hd / BN) as u32, (t / BM) as u32, z as u32),
    }));
    Some(ops)
}

/// Backward ops of the last [`fwd_ops`] on this scratch: `dout` (f32) and
/// `dsum` = Σ dO⊙O (f32, filled by the caller) → dq, dk, dv (f32,
/// overwritten). `gs`: the power-of-two scale of the f16 gradients.
#[allow(clippy::too_many_arguments)]
pub fn bwd_ops<'a>(c: &'a Ctx, g: AttnGeom, sc: &'a AttScratch, lse: &'a wgpu::Buffer, dout: &'a wgpu::Buffer, dsum: &'a wgpu::Buffer, dq: &'a wgpu::Buffer, dk: &'a wgpu::Buffer, dv: &'a wgpu::Buffer, gs: f32) -> Option<Vec<Op<'a>>> {
    let pp = pipes(c)?;
    let (t, hd, nh, nkv) = (g.t, g.hd, g.nh, g.nkv);
    let rep = nh / nkv;
    let n = g.nb * t;
    let (qd, kd) = (nh * hd, nkv * hd);
    let scale = 1.0 / (hd as f32).sqrt();
    let z = g.nb * nh;
    let mut ops = vec![Op::D(tm::cvt(c, dout, None, &sc.do16, n, qd, gs)?)];
    // dP = dO16·V16ᵀ / gs
    ops.push(Op::D(Disp {
        pipe: &pp.p["att_mm_s"],
        uni: uni(t, t, hd, nh, [qd, t * qd, hd, 1], [kd, t * kd, hd, rep], [t, nh * t * t, t * t, 1], [1, 0, 0], 1.0 / gs, &g, gs),
        bufs: vec![(1, &sc.do16), (2, &sc.v16), (3, &sc.dp)],
        grid: ((t / BN) as u32, (t / BM) as u32, z as u32),
    }));
    // dS16
    ops.push(Op::D(Disp {
        pipe: &pp.p["att_ds"],
        uni: uni(0, 0, 0, z, [0; 4], [0; 4], [0; 4], [0; 3], 1.0, &g, gs),
        bufs: vec![(4, &sc.s), (5, &sc.ds16), (6, lse), (7, dsum), (8, &sc.dp)],
        grid: row_grid(z * t),
    }));
    // dQ = dS16·K16 · sc/gs
    ops.push(Op::D(Disp {
        pipe: &pp.p["att_mm_o"],
        uni: uni(t, hd, t, nh, [t, nh * t * t, t * t, 1], [kd, t * kd, hd, rep], [qd, t * qd, hd, 1], [1, 0, 0], scale / gs, &g, gs),
        bufs: vec![(1, &sc.ds16), (2, &sc.k16), (3, dq)],
        grid: ((hd / BN) as u32, (t / BM) as u32, z as u32),
    }));
    // dK = Σ_h dS16ᵀ·Q16 / gs ; dV = Σ_h P16ᵀ·dO16 / gs  (batch over (window, kv-head))
    let zk = g.nb * nkv;
    for (a, b, out) in [(&sc.ds16, &sc.q16, dk), (&sc.p16, &sc.do16, dv)] {
        ops.push(Op::D(Disp {
            pipe: &pp.p["att_mm_d"],
            uni: uni(t, hd, t, nkv, [t, nh * t * t, rep * t * t, 1], [qd, t * qd, rep * hd, 1], [kd, t * kd, hd, 1], [rep, t * t, hd], 1.0 / gs, &g, gs),
            bufs: vec![(1, a), (2, b), (3, out)],
            grid: ((hd / BN) as u32, (t / BM) as u32, zk as u32),
        }));
    }
    Some(ops)
}

#[cfg(test)]
mod tests {
    use super::super::train_ops::{attn_bwd, attn_fwd, dev, download, upload, zeros};
    use super::*;

    fn rnd(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32 * scale
            })
            .collect()
    }

    fn rel(name: &str, got: &[f32], want: &[f32], tol: f64) {
        let num: f64 = got.iter().zip(want).map(|(g, w)| (*g as f64 - *w as f64).powi(2)).sum();
        let den: f64 = want.iter().map(|w| (*w as f64).powi(2)).sum::<f64>().max(1e-30);
        let r = (num / den).sqrt();
        eprintln!("{name}: relative difference {r:e}");
        assert!(r < tol, "{name}: relative difference {r:e}");
    }

    /// The matrix-unit attention against the exact f32 kernels (f16
    /// operands: ~1e-3 relative).
    #[test]
    fn attention_tc_matches_f32() {
        let Some(c) = dev() else { return };
        let g = AttnGeom { nb: 2, t: 128, nh: 6, nkv: 2, hd: 256 };
        if !usable(c, &g) {
            eprintln!("no matrix units / subgroups — skipped");
            return;
        }
        let (n, qs, ks) = (g.nb * g.t, g.nh * g.hd, g.nkv * g.hd);
        let q = rnd(n * qs, 21, 1.0);
        let k = rnd(n * ks, 22, 1.0);
        let v = rnd(n * ks, 23, 1.0);
        let dout = rnd(n * qs, 24, 1e-3);
        let (bq, bk, bv, bdo) = (upload(c, &q), upload(c, &k), upload(c, &v), upload(c, &dout));
        // reference: the f32 kernels
        let (ro, rlse, rds, rdq, rdk, rdv) = (zeros(c, n * qs), zeros(c, n * g.nh), zeros(c, n * g.nh), zeros(c, n * qs), zeros(c, n * ks), zeros(c, n * ks));
        super::super::train_ops::run(c, &[attn_fwd(c, g, &bq, &bk, &bv, &ro, &rlse).unwrap()]);
        super::super::train_ops::run(c, &attn_bwd(c, g, &bq, &bk, &bv, &ro, &rlse, &bdo, &rds, &rdq, &rdk, &rdv).unwrap());
        // matrix units
        let sc = AttScratch::new(c, &g);
        let (o, lse, dsum, dq, dk, dv) = (zeros(c, n * qs), zeros(c, n * g.nh), zeros(c, n * g.nh), zeros(c, n * qs), zeros(c, n * ks), zeros(c, n * ks));
        tm::run(c, &fwd_ops(c, g, &sc, &bq, &bk, &bv, &o, &lse).unwrap());
        // D = Σ dO⊙O from the reference kernel's dsum pass on this O
        let d = attn_bwd(c, g, &bq, &bk, &bv, &o, &lse, &bdo, &dsum, &rdq, &rdk, &rdv).unwrap();
        super::super::train_ops::run(c, &d[..1]);
        tm::run(c, &bwd_ops(c, g, &sc, &lse, &bdo, &dsum, &dq, &dk, &dv, 1024.0).unwrap());
        let dl = |b: &wgpu::Buffer, len: usize| download(c, b, len).unwrap();
        rel("o", &dl(&o, n * qs), &dl(&ro, n * qs), 3e-3);
        rel("lse", &dl(&lse, n * g.nh), &dl(&rlse, n * g.nh), 1e-4);
        rel("dq", &dl(&dq, n * qs), &dl(&rdq, n * qs), 1e-2);
        rel("dk", &dl(&dk, n * ks), &dl(&rdk, n * ks), 1e-2);
        rel("dv", &dl(&dv, n * ks), &dl(&rdv, n * ks), 3e-3);
    }
}
