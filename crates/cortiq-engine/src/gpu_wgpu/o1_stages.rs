//! A sealed layer's attention for one row, in short threads.
//!
//! `o1_attend` holds one 256-lane workgroup per query head, and a workgroup stays on one
//! SM: the card sits mostly idle while each head walks its near keys. These stages take
//! the same arithmetic apart, as Dory's graph does: near scores with a thread per key and
//! all the group's query heads at once, the largest score, the far part per head, the
//! weights, the value sum in pieces, and the pieces added.

use super::*;

/// Pieces of the value sum.
pub(crate) const CH: usize = 8;

/// Query heads per KV head the generated kernels are built for: one accumulator a head
/// (Qwen3.8-27B has 6, Ripple-65B 10).
pub(crate) const MAX_HPG: usize = 16;

fn source(hpg: usize) -> String {
    let each = |f: &dyn Fn(usize) -> String| (0..hpg).map(f).collect::<Vec<_>>().join("\n");
    let zero = each(&|h| format!("    var a{h} = vec4<f32>(0.0);"));
    let dots_rk = each(&|h| format!("            a{h} += k * sq[q0 + {h}u * e4 + e];"));
    let writes = each(&|h| format!("    ssc[at + {h}u] = hs(a{h}) * u0.scale;"));
    let lwrites = each(&|h| format!("    lfs[at + {h}u] = hs(a{h}) * u0.scale;"));
    let ldots = each(&|h| format!("        a{h} += k * lq[q0 + {h}u * e4 + e];"));
    let o_zero = each(&|h| format!("    var o{h} = vec4<f32>(0.0);"));
    let o_add = each(&|h| format!("        o{h} += usc[(sb + i) * {hpg}u + {h}u] * v;"));
    let o_write = each(&|h| format!("    upz[((g * 8u + ch) * {hpg}u + {h}u) * e4 + d] = o{h};"));
    format!(
        r#"
struct OU {{ hpg: u32, m: u32, w: u32, ns: u32, d: u32, g: u32, scale: f32, ncap: u32 }};
@group(0) @binding(0) var<uniform> u0: OU;
fn hs(v: vec4<f32>) -> f32 {{ return v.x + v.y + v.z + v.w; }}

// Near scores: a thread per key of the sink and the window.
@group(0) @binding(1) var<storage, read> smeta: array<u32>;
@group(0) @binding(2) var<storage, read> sq: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> srk: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> ssk: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> ssc: array<f32>;
@compute @workgroup_size(32)
fn o1s_score(@builtin(global_invocation_id) id: vec3<u32>) {{
    let s = id.x;
    let g = id.y;
    if (s >= u0.ns + smeta[g * 4u]) {{ return; }}
    let e4 = u0.d / 4u;
    let q0 = g * {hpg}u * e4;
{zero}
    if (s < u0.ns) {{
        let kb = (g * u0.ns + s) * e4;
        for (var e = 0u; e < e4; e += 1u) {{
            let k = ssk[kb + e];
{dots_rk}
        }}
    }} else {{
        let kb = (g * u0.w + s - u0.ns) * e4;
        for (var e = 0u; e < e4; e += 1u) {{
            let k = srk[kb + e];
{dots_rk}
        }}
    }}
    let at = (g * u0.ncap + s) * {hpg}u;
{writes}
}}

// Landmark scores, the same way.
@group(0) @binding(1) var<storage, read> lq: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> lkt: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> lfs: array<f32>;
@compute @workgroup_size(32)
fn o1s_land(@builtin(global_invocation_id) id: vec3<u32>) {{
    let a = id.x;
    let g = id.y;
    if (a >= u0.m) {{ return; }}
    let e4 = u0.d / 4u;
    let q0 = g * {hpg}u * e4;
    let kb = (g * u0.m + a) * e4;
{zero}
    for (var e = 0u; e < e4; e += 1u) {{
        let k = lkt[kb + e];
{ldots}
    }}
    let at = (g * u0.m + a) * {hpg}u;
{lwrites}
}}

var<workgroup> r_a: array<f32, 64>;
fn rmax1(lid: u32, v: f32) -> f32 {{
    r_a[lid] = v;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st = st >> 1u) {{
        if (lid < st) {{ r_a[lid] = max(r_a[lid], r_a[lid + st]); }}
        workgroupBarrier();
    }}
    let r = r_a[0];
    workgroupBarrier();
    return r;
}}
fn rsum1(lid: u32, v: f32) -> f32 {{
    r_a[lid] = v;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st = st >> 1u) {{
        if (lid < st) {{ r_a[lid] = r_a[lid] + r_a[lid + st]; }}
        workgroupBarrier();
    }}
    let r = r_a[0];
    workgroupBarrier();
    return r;
}}

// The largest near score of a query head.
@group(0) @binding(1) var<storage, read> mmeta: array<u32>;
@group(0) @binding(2) var<storage, read> msc: array<f32>;
@group(0) @binding(3) var<storage, read_write> mnm: array<f32>;
@compute @workgroup_size(64)
fn o1s_max(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let gh = w.x;
    let g = gh / {hpg}u;
    let h = gh % {hpg}u;
    let n = u0.ns + mmeta[g * 4u];
    let sb = g * u0.ncap;
    var m = -1.0e30;
    for (var i = lid; i < n; i += 64u) {{ m = max(m, msc[(sb + i) * {hpg}u + h]); }}
    let mx = rmax1(lid, m);
    if (lid == 0u) {{ mnm[gh] = mx; }}
}}

// A query head's far part: the landmark weights through its inverse, one gain per landmark,
// the far mass, and the common exponent the near scores will use.
@group(0) @binding(1) var<storage, read> fmeta: array<u32>;
@group(0) @binding(2) var<storage, read> ffs: array<f32>;
@group(0) @binding(3) var<storage, read> fnm: array<f32>;
@group(0) @binding(4) var<storage, read> fmu: array<f32>;
@group(0) @binding(5) var<storage, read> fmz: array<f32>;
@group(0) @binding(6) var<storage, read_write> fhd: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> fgn: array<f32>;
@compute @workgroup_size(8)
fn o1s_far(@builtin(global_invocation_id) id: vec3<u32>) {{
    let gh = id.x;
    if (gh >= u0.g * {hpg}u) {{ return; }}
    let g = gh / {hpg}u;
    let h = gh % {hpg}u;
    let m = u0.m;
    var c_all = fnm[gh];
    var far_den = 0.0;
    var have = 0.0;
    if (fmeta[g * 4u + 2u] > 0u) {{
        var f = -3.0e38;
        var fe: array<f32, 32>;
        var u: array<f32, 32>;
        for (var a = 0u; a < m; a += 1u) {{
            let v = ffs[(g * m + a) * {hpg}u + h];
            fe[a] = v;
            f = max(f, v);
        }}
        for (var a = 0u; a < m; a += 1u) {{
            let e = exp(fe[a] - f);
            let row = (gh * m + a) * m;
            for (var b = 0u; b < m; b += 1u) {{ u[b] += e * fmu[row + b]; }}
        }}
        let mzb = gh * 2u * m;
        for (var b = 0u; b < m; b += 1u) {{ c_all = max(c_all, f + fmz[mzb + b]); }}
        for (var b = 0u; b < m; b += 1u) {{
            let gain = u[b] * exp(f + fmz[mzb + b] - c_all);
            fgn[gh * m + b] = gain;
            far_den += gain * fmz[mzb + m + b];
        }}
        if (far_den >= 0.0) {{ have = 1.0; }} else {{ far_den = 0.0; }}
    }}
    fhd[gh] = vec4<f32>(c_all, far_den, have, 0.0);
}}

// The near weights under that exponent, and 1/Σ with the far mass in it.
@group(0) @binding(1) var<storage, read> tmeta: array<u32>;
@group(0) @binding(2) var<storage, read_write> tsc: array<f32>;
@group(0) @binding(3) var<storage, read> thd: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> tinv: array<f32>;
@compute @workgroup_size(64)
fn o1s_soft(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let gh = w.x;
    let g = gh / {hpg}u;
    let h = gh % {hpg}u;
    let n = u0.ns + tmeta[g * 4u];
    let sb = g * u0.ncap;
    let hd = thd[gh];
    var den = 0.0;
    for (var i = lid; i < n; i += 64u) {{
        let at = (sb + i) * {hpg}u + h;
        let e = exp(tsc[at] - hd.x);
        tsc[at] = e;
        den += e;
    }}
    let total = rsum1(lid, den) + hd.y;
    if (lid == 0u) {{ tinv[gh] = 1.0 / max(total, 1.0e-30); }}
}}

// The near value sum in eight pieces: a thread per (KV head, piece, vec4 of the head).
@group(0) @binding(1) var<storage, read> umeta: array<u32>;
@group(0) @binding(2) var<storage, read> usc: array<f32>;
@group(0) @binding(3) var<storage, read> urv: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> usv: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> upz: array<vec4<f32>>;
@compute @workgroup_size(32)
fn o1s_sum(@builtin(global_invocation_id) id: vec3<u32>) {{
    let e4 = u0.d / 4u;
    let t = id.x;
    let d = t % e4;
    let ch = (t / e4) % 8u;
    let g = t / (e4 * 8u);
    if (g >= u0.g) {{ return; }}
    let n = u0.ns + umeta[g * 4u];
    let per = (n + 7u) / 8u;
    let sb = g * u0.ncap;
{o_zero}
    let last = min((ch + 1u) * per, n);
    for (var i = ch * per; i < last; i += 1u) {{
        var v: vec4<f32>;
        if (i < u0.ns) {{ v = usv[(g * u0.ns + i) * e4 + d]; }} else {{ v = urv[(g * u0.w + i - u0.ns) * e4 + d]; }}
{o_add}
    }}
{o_write}
}}

// The pieces, the far values, 1/Σ.
@group(0) @binding(1) var<storage, read> apz: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> ahd: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> agn: array<f32>;
@group(0) @binding(4) var<storage, read> ath: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> ainv: array<f32>;
@group(0) @binding(6) var<storage, read_write> aout: array<vec4<f32>>;
@compute @workgroup_size(32)
fn o1s_add(@builtin(global_invocation_id) id: vec3<u32>) {{
    let e = id.x;
    let e4 = u0.d / 4u;
    if (e >= u0.g * {hpg}u * e4) {{ return; }}
    let d = e % e4;
    let gh = e / e4;
    let g = gh / {hpg}u;
    let h = gh % {hpg}u;
    var acc = vec4<f32>(0.0);
    for (var ch = 0u; ch < 8u; ch += 1u) {{
        acc += apz[((g * 8u + ch) * {hpg}u + h) * e4 + d];
    }}
    if (ahd[gh].z > 0.5) {{
        for (var b = 0u; b < u0.m; b += 1u) {{
            acc += agn[gh * u0.m + b] * ath[(gh * u0.m + b) * e4 + d];
        }}
    }}
    aout[e] = acc * ainv[gh];
}}
"#
    )
}

pub(crate) struct Pipe {
    pub(crate) p: wgpu::ComputePipeline,
    pub(crate) l: wgpu::BindGroupLayout,
}

pub(crate) struct Pipes {
    pub(crate) hpg: usize,
    pub(crate) score: Pipe,
    pub(crate) land: Pipe,
    pub(crate) max: Pipe,
    pub(crate) far: Pipe,
    pub(crate) soft: Pipe,
    pub(crate) sum: Pipe,
    pub(crate) add: Pipe,
}

/// One layer's scratch for the stages and their uniform.
pub(crate) struct Scratch {
    pub(crate) u: wgpu::Buffer,
    pub(crate) sc: wgpu::Buffer,
    pub(crate) fs: wgpu::Buffer,
    pub(crate) nm: wgpu::Buffer,
    pub(crate) hd: wgpu::Buffer,
    pub(crate) gn: wgpu::Buffer,
    pub(crate) inv: wgpu::Buffer,
    pub(crate) pz: wgpu::Buffer,
    /// A speculative chunk's keys and values until its accepted part is known, and the
    /// commit's scratch.
    pub(crate) spec_k: wgpu::Buffer,
    pub(crate) spec_v: wgpu::Buffer,
    pub(crate) spec_s: wgpu::Buffer,
}

static PIPES: std::sync::OnceLock<Option<Pipes>> = std::sync::OnceLock::new();

/// The stages for `hpg` query heads per KV head. A second geometry in one process keeps
/// `o1_attend`, as does `CMF_O1_STAGES=0`.
pub(crate) fn pipes(c: &Ctx, hpg: usize) -> Option<&'static Pipes> {
    if std::env::var("CMF_O1_STAGES").as_deref() == Ok("0") {
        return None;
    }
    PIPES
        .get_or_init(|| {
            if !(1..=MAX_HPG).contains(&hpg) {
                return None;
            }
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("o1-stages"),
                source: wgpu::ShaderSource::Wgsl(source(hpg).into()),
            });
            let mk = |entry: &str| {
                let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &m,
                    entry_point: Some(entry),
                    compilation_options: pco(),
                    cache: None,
                });
                let l = p.get_bind_group_layout(0);
                Pipe { p, l }
            };
            let out = Pipes {
                hpg,
                score: mk("o1s_score"),
                land: mk("o1s_land"),
                max: mk("o1s_max"),
                far: mk("o1s_far"),
                soft: mk("o1s_soft"),
                sum: mk("o1s_sum"),
                add: mk("o1s_add"),
            };
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("o1 stages rejected: {e}");
                return None;
            }
            Some(out)
        })
        .as_ref()
        .filter(|p| p.hpg == hpg)
}

/// The scratch of a state of this geometry.
pub(crate) fn scratch(c: &Ctx, g: usize, hpg: usize, m: usize, w: usize, ns: usize, d: usize, scale: f32) -> Scratch {
    let store = |bytes: usize, label: &'static str| {
        c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let (nh, ncap) = (g * hpg, w + ns);
    let u = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("o1s-u"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let words = [hpg as u32, m as u32, w as u32, ns as u32, d as u32, g as u32, scale.to_bits(), ncap as u32];
    c.queue.write_buffer(&u, 0, bytemuck::cast_slice(&words));
    Scratch {
        u,
        sc: store(g * ncap * hpg * 4, "o1s-sc"),
        fs: store(g * m * hpg * 4, "o1s-fs"),
        nm: store(nh * 4, "o1s-nm"),
        hd: store(nh * 16, "o1s-hd"),
        gn: store(nh * m * 4, "o1s-gn"),
        inv: store(nh * 4, "o1s-inv"),
        pz: store(g * CH * hpg * d * 4, "o1s-pz"),
        spec_k: {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("o1s-spec-k"),
                size: (o1_batch::OB_K * g * d * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        },
        spec_v: {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("o1s-spec-v"),
                size: (o1_batch::OB_K * g * d * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        },
        spec_s: store(o1_batch::Geom { g, hpg, m, w, ns, rect_fm: false, d, dv: d, scale }.scratch_len() * 4, "o1s-spec-s"),
    }
}

/// The same stages for a chunk of rows, in place of `o1b_attend`: a verify of drafts and a
/// sealed prompt chunk. Far terms come from `o1b_far_e`'s scratch, as there, and every
/// plane has a row dimension, in one set of buffers for all layers of this geometry.
fn chunk_source(hpg: usize) -> String {
    let each = |f: &dyn Fn(usize) -> String| (0..hpg).map(f).collect::<Vec<_>>().join("\n");
    let zero = each(&|h| format!("    var a{h} = vec4<f32>(0.0);"));
    let dots = |buf: &str| each(&|h| format!("            a{h} += sk_ * sq[q0 + {h}u * e4 + e];")).replace("sk_", buf);
    let (dots_sink, dots_ring, dots_chunk) = (dots("ssk[kb + e]"), dots("srk[kb + e]"), dots("skb[kb + e]"));
    let writes = each(&|h| format!("    ssc[at + {h}u] = hs(a{h}) * c0.scale;"));
    let lwrites = each(&|h| format!("    lfs[at + {h}u] = hs(a{h}) * c0.scale;"));
    let ldots = each(&|h| format!("        a{h} += k * lq[q0 + {h}u * e4 + e];"));
    let o_zero = each(&|h| format!("    var o{h} = vec4<f32>(0.0);"));
    let o_add = each(&|h| format!("        o{h} += usc[(sb + j) * {hpg}u + {h}u] * v;"));
    let o_write = each(&|h| format!("    upz[(((i * c0.ng + g) * 8u + ch) * {hpg}u + {h}u) * e4 + d] = o{h};"));
    format!(
        r#"
struct OC {{ hpg: u32, m: u32, w: u32, nsrect: u32, d: u32, dv: u32, scale: f32, k: u32, nh: u32, ng: u32, r0: u32, p_: u32 }};
@group(0) @binding(0) var<uniform> c0: OC;
fn hs(v: vec4<f32>) -> f32 {{ return v.x + v.y + v.z + v.w; }}
fn nsr() -> u32 {{ return c0.nsrect & 0xFFu; }}
fn cap() -> u32 {{ return c0.w + nsr(); }}
fn wgf(gw: u32) -> u32 {{ return select(c0.w, gw, gw != 0u); }}

// Near scores: a thread per (key, group, row). A row's window runs over the ring and the
// chunk's rows up to its own.
@group(0) @binding(1) var<storage, read> smeta: array<u32>;
@group(0) @binding(2) var<storage, read> sq: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> srk: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> ssk: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> skb: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> ssc: array<f32>;
@compute @workgroup_size(32)
fn o1c_score(@builtin(global_invocation_id) id: vec3<u32>) {{
    let s = id.x;
    let g = id.y;
    let i = id.z;
    let ns = nsr();
    let len0 = smeta[g * 4u];
    let wg = wgf(smeta[g * 4u + 3u]);
    let win = min(wg, len0 + i + 1u);
    if (s >= ns + win) {{ return; }}
    let e4 = c0.d / 4u;
    let q0 = ((c0.r0 + i) * c0.nh + g * {hpg}u) * e4;
{zero}
    if (s < ns) {{
        let kb = (g * ns + s) * e4;
        for (var e = 0u; e < e4; e += 1u) {{
{dots_sink}
        }}
    }} else {{
        let x = len0 + i + 1u - win + s - ns;
        if (x < len0) {{
            let kb = (g * c0.w + (smeta[g * 4u + 1u] + x) % wg) * e4;
            for (var e = 0u; e < e4; e += 1u) {{
{dots_ring}
            }}
        }} else {{
            let kb = ((c0.r0 + x - len0) * c0.ng + g) * e4;
            for (var e = 0u; e < e4; e += 1u) {{
{dots_chunk}
            }}
        }}
    }}
    let at = ((i * c0.ng + g) * cap() + s) * {hpg}u;
{writes}
}}

// Landmark scores.
@group(0) @binding(1) var<storage, read> lq: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> lkt: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> lfs: array<f32>;
@compute @workgroup_size(32)
fn o1c_land(@builtin(global_invocation_id) id: vec3<u32>) {{
    let a = id.x;
    let g = id.y;
    let i = id.z;
    if (a >= c0.m) {{ return; }}
    let e4 = c0.d / 4u;
    let q0 = ((c0.r0 + i) * c0.nh + g * {hpg}u) * e4;
    let kb = (g * c0.m + a) * e4;
{zero}
    for (var e = 0u; e < e4; e += 1u) {{
        let k = lkt[kb + e];
{ldots}
    }}
    let at = ((i * c0.ng + g) * c0.m + a) * {hpg}u;
{lwrites}
}}

var<workgroup> r_a: array<f32, 64>;
fn rmax1(lid: u32, v: f32) -> f32 {{
    r_a[lid] = v;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st = st >> 1u) {{
        if (lid < st) {{ r_a[lid] = max(r_a[lid], r_a[lid + st]); }}
        workgroupBarrier();
    }}
    let r = r_a[0];
    workgroupBarrier();
    return r;
}}
fn rsum1(lid: u32, v: f32) -> f32 {{
    r_a[lid] = v;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st = st >> 1u) {{
        if (lid < st) {{ r_a[lid] = r_a[lid] + r_a[lid + st]; }}
        workgroupBarrier();
    }}
    let r = r_a[0];
    workgroupBarrier();
    return r;
}}

// The largest near score of a (row, query head).
@group(0) @binding(1) var<storage, read> mmeta: array<u32>;
@group(0) @binding(2) var<storage, read> msc: array<f32>;
@group(0) @binding(3) var<storage, read_write> mnm: array<f32>;
@compute @workgroup_size(64)
fn o1c_max(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let gh = w.x;
    let i = w.y;
    let g = gh / {hpg}u;
    let h = gh % {hpg}u;
    let n = nsr() + min(wgf(mmeta[g * 4u + 3u]), mmeta[g * 4u] + i + 1u);
    let sb = (i * c0.ng + g) * cap();
    var m = -1.0e30;
    for (var j = lid; j < n; j += 64u) {{ m = max(m, msc[(sb + j) * {hpg}u + h]); }}
    let mx = rmax1(lid, m);
    if (lid == 0u) {{ mnm[i * c0.nh + gh] = mx; }}
}}

// The far part of a (row, query head), with the far state as of this row: the mass up to it
// and the weight of each row the chunk has evicted so far.
@group(0) @binding(1) var<storage, read> fmeta: array<u32>;
@group(0) @binding(2) var<storage, read> ffs: array<f32>;
@group(0) @binding(3) var<storage, read> fnm: array<f32>;
@group(0) @binding(4) var<storage, read> fmu: array<f32>;
@group(0) @binding(5) var<storage, read> fsc: array<f32>;
@group(0) @binding(6) var<storage, read_write> fhd: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> fgn: array<f32>;
@group(0) @binding(8) var<storage, read_write> fcf: array<f32>;
@compute @workgroup_size(8)
fn o1c_far(@builtin(global_invocation_id) id: vec3<u32>) {{
    let gh = id.x;
    if (gh >= c0.nh) {{ return; }}
    let i = id.y;
    let g = gh / {hpg}u;
    let h = gh % {hpg}u;
    let m = c0.m;
    let j0 = wgf(fmeta[g * 4u + 3u]) - fmeta[g * 4u];
    var farl = fmeta[g * 4u + 2u];
    if (i >= j0) {{ farl += i - j0 + 1u; }}
    let at0 = i * c0.nh + gh;
    var c_all = fnm[at0];
    var far_den = 0.0;
    var have = 0.0;
    if (farl > 0u) {{
        var f = -3.0e38;
        var fe: array<f32, 32>;
        var u: array<f32, 32>;
        var gain: array<f32, 32>;
        for (var a = 0u; a < m; a += 1u) {{
            let v = ffs[((i * c0.ng + g) * m + a) * {hpg}u + h];
            fe[a] = v;
            f = max(f, v);
        }}
        for (var a = 0u; a < m; a += 1u) {{
            let e = exp(fe[a] - f);
            let row = (gh * m + a) * m;
            for (var b = 0u; b < m; b += 1u) {{ u[b] += e * fmu[row + b]; }}
        }}
        let ghm = c0.nh * m;
        let mb = 2u * ghm * 64u;
        for (var b = 0u; b < m; b += 1u) {{ c_all = max(c_all, f + fsc[mb + gh * m + b]); }}
        for (var b = 0u; b < m; b += 1u) {{
            gain[b] = u[b] * exp(f + fsc[mb + gh * m + b] - c_all);
            far_den += gain[b] * fsc[(ghm + gh * m + b) * 64u + i];
            fgn[at0 * m + b] = gain[b] * fsc[mb + ghm + gh * m + b];
        }}
        if (far_den >= 0.0) {{ have = 1.0; }} else {{ far_den = 0.0; }}
        if (have > 0.5 && i >= j0) {{
            for (var j = j0; j <= i; j += 1u) {{
                var cf = 0.0;
                for (var b = 0u; b < m; b += 1u) {{ cf += gain[b] * fsc[(gh * m + b) * 64u + j]; }}
                fcf[at0 * 64u + j] = cf;
            }}
        }}
    }}
    fhd[at0] = vec4<f32>(c_all, far_den, have, 0.0);
}}

// The near weights and 1/Σ.
@group(0) @binding(1) var<storage, read> tmeta: array<u32>;
@group(0) @binding(2) var<storage, read_write> tsc: array<f32>;
@group(0) @binding(3) var<storage, read> thd: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> tinv: array<f32>;
@compute @workgroup_size(64)
fn o1c_soft(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let gh = w.x;
    let i = w.y;
    let g = gh / {hpg}u;
    let h = gh % {hpg}u;
    let n = nsr() + min(wgf(tmeta[g * 4u + 3u]), tmeta[g * 4u] + i + 1u);
    let sb = (i * c0.ng + g) * cap();
    let hd = thd[i * c0.nh + gh];
    var den = 0.0;
    for (var j = lid; j < n; j += 64u) {{
        let at = (sb + j) * {hpg}u + h;
        let e = exp(tsc[at] - hd.x);
        tsc[at] = e;
        den += e;
    }}
    let total = rsum1(lid, den) + hd.y;
    if (lid == 0u) {{ tinv[i * c0.nh + gh] = 1.0 / max(total, 1.0e-30); }}
}}

// The near value sum in eight pieces: a thread per (row, KV head, piece, vec4 of the head).
@group(0) @binding(1) var<storage, read> umeta: array<u32>;
@group(0) @binding(2) var<storage, read> usc: array<f32>;
@group(0) @binding(3) var<storage, read> urv: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> usv: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> uvb: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> upz: array<vec4<f32>>;
@compute @workgroup_size(32)
fn o1c_sum(@builtin(global_invocation_id) id: vec3<u32>) {{
    let e4 = c0.d / 4u;
    let t = id.x;
    let i = id.y;
    let d = t % e4;
    let ch = (t / e4) % 8u;
    let g = t / (e4 * 8u);
    if (g >= c0.ng) {{ return; }}
    let ns = nsr();
    let len0 = umeta[g * 4u];
    let head0 = umeta[g * 4u + 1u];
    let wg = wgf(umeta[g * 4u + 3u]);
    let win = min(wg, len0 + i + 1u);
    let lo = len0 + i + 1u - win;
    let n = ns + win;
    let per = (n + 7u) / 8u;
    let sb = (i * c0.ng + g) * cap();
{o_zero}
    let last = min((ch + 1u) * per, n);
    for (var j = ch * per; j < last; j += 1u) {{
        var v: vec4<f32>;
        if (j < ns) {{
            v = usv[(g * ns + j) * e4 + d];
        }} else {{
            let x = lo + j - ns;
            if (x < len0) {{
                v = urv[(g * c0.w + (head0 + x) % wg) * e4 + d];
            }} else {{
                v = uvb[((c0.r0 + x - len0) * c0.ng + g) * e4 + d];
            }}
        }}
{o_add}
    }}
{o_write}
}}

// The pieces, the far values, the rows evicted so far, 1/Σ.
@group(0) @binding(1) var<storage, read> ameta: array<u32>;
@group(0) @binding(2) var<storage, read> apz: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> ahd: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> agn: array<f32>;
@group(0) @binding(5) var<storage, read> ath: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> ainv: array<f32>;
@group(0) @binding(7) var<storage, read> acf: array<f32>;
@group(0) @binding(8) var<storage, read> arv: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> aout: array<vec4<f32>>;
@compute @workgroup_size(32)
fn o1c_add(@builtin(global_invocation_id) id: vec3<u32>) {{
    let e = id.x;
    let i = id.y;
    let e4 = c0.d / 4u;
    if (e >= c0.nh * e4) {{ return; }}
    let d = e % e4;
    let gh = e / e4;
    let g = gh / {hpg}u;
    let h = gh % {hpg}u;
    let at0 = i * c0.nh + gh;
    var acc = vec4<f32>(0.0);
    for (var ch = 0u; ch < 8u; ch += 1u) {{
        acc += apz[(((i * c0.ng + g) * 8u + ch) * {hpg}u + h) * e4 + d];
    }}
    if (ahd[at0].z > 0.5) {{
        for (var b = 0u; b < c0.m; b += 1u) {{
            acc += agn[at0 * c0.m + b] * ath[(gh * c0.m + b) * e4 + d];
        }}
        let wg = wgf(ameta[g * 4u + 3u]);
        let j0 = wg - ameta[g * 4u];
        if (i >= j0) {{
            let head0 = ameta[g * 4u + 1u];
            for (var j = j0; j <= i; j += 1u) {{
                acc += acf[at0 * 64u + j] * arv[(g * c0.w + (head0 + j - j0) % wg) * e4 + d];
            }}
        }}
    }}
    aout[((c0.r0 + i) * c0.nh + gh) * e4 + d] = acc * ainv[at0];
}}
"#
    )
}

static CHUNK_PIPES: [std::sync::OnceLock<Option<Pipes>>; MAX_HPG + 1] = [const { std::sync::OnceLock::new() }; MAX_HPG + 1];

/// The chunk stages for `hpg` query heads per KV head, on the terms of [`pipes`].
pub(crate) fn chunk_pipes(c: &Ctx, hpg: usize) -> Option<&'static Pipes> {
    if std::env::var("CMF_O1_STAGES").as_deref() == Ok("0") {
        return None;
    }
    if !(1..=MAX_HPG).contains(&hpg) {
        return None;
    }
    CHUNK_PIPES[hpg]
        .get_or_init(|| {
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("o1-chunk-stages"),
                source: wgpu::ShaderSource::Wgsl(chunk_source(hpg).into()),
            });
            let mk = |entry: &str| {
                let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &m,
                    entry_point: Some(entry),
                    compilation_options: pco(),
                    cache: None,
                });
                let l = p.get_bind_group_layout(0);
                Pipe { p, l }
            };
            let out = Pipes {
                hpg,
                score: mk("o1c_score"),
                land: mk("o1c_land"),
                max: mk("o1c_max"),
                far: mk("o1c_far"),
                soft: mk("o1c_soft"),
                sum: mk("o1c_sum"),
                add: mk("o1c_add"),
            };
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("o1 chunk stages rejected: {e}");
                return None;
            }
            Some(out)
        })
        .as_ref()
}

/// One chunk's planes. Layers run one after another, so all of them share one set.
pub(crate) struct ChunkScratch {
    key: (usize, usize, usize, usize, usize),
    sc: wgpu::Buffer,
    fs: wgpu::Buffer,
    nm: wgpu::Buffer,
    hd: wgpu::Buffer,
    gn: wgpu::Buffer,
    cf: wgpu::Buffer,
    inv: wgpu::Buffer,
    pz: wgpu::Buffer,
}

static CHUNK_SCRATCH: std::sync::Mutex<Option<std::sync::Arc<ChunkScratch>>> = std::sync::Mutex::new(None);

fn chunk_scratch(c: &Ctx, geom: &o1_batch::Geom) -> std::sync::Arc<ChunkScratch> {
    let k = o1_batch::OB_K;
    let (g, hpg, m, d, ncap) = (geom.g, geom.hpg, geom.m, geom.d, geom.w + geom.ns);
    let key = (g, hpg, m, d, ncap);
    let mut slot = CHUNK_SCRATCH.lock().unwrap();
    if let Some(s) = slot.as_ref().filter(|s| s.key == key) {
        return s.clone();
    }
    let store = |bytes: usize, label: &'static str| {
        c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let nh = g * hpg;
    let s = std::sync::Arc::new(ChunkScratch {
        key,
        sc: store(k * nh * ncap * 4, "o1c-sc"),
        fs: store(k * nh * m * 4, "o1c-fs"),
        nm: store(k * nh * 4, "o1c-nm"),
        hd: store(k * nh * 16, "o1c-hd"),
        gn: store(k * nh * m * 4, "o1c-gn"),
        cf: store(k * nh * k * 4, "o1c-cf"),
        inv: store(k * nh * 4, "o1c-inv"),
        pz: store(k * nh * CH * d * 4, "o1c-pz"),
    });
    *slot = Some(s.clone());
    s
}

/// `o1b_attend` for `kk` rows from `r0`, in stages. `u` is the chunk's `OBP` uniform and
/// `far_s` the scratch `o1b_far_e` has just filled.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_chunk(
    c: &Ctx,
    pass: &mut wgpu::ComputePass<'_>,
    p: &Pipes,
    st: &o1_batch::State<'_>,
    geom: o1_batch::Geom,
    q: &wgpu::Buffer,
    kb: &wgpu::Buffer,
    vb: &wgpu::Buffer,
    far_s: &wgpu::Buffer,
    out: &wgpu::Buffer,
    u: &wgpu::Buffer,
    kk: usize,
) {
    let s = chunk_scratch(c, &geom);
    let bind = |l: &wgpu::BindGroupLayout, e: &[&wgpu::Buffer]| {
        let entries: Vec<_> = e.iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: l, entries: &entries })
    };
    let (g, nh, ncap, m) = (geom.g as u32, (geom.g * geom.hpg) as u32, (geom.w + geom.ns) as u32, geom.m as u32);
    let e4 = (geom.d / 4) as u32;
    let k = kk as u32;
    let steps: [(&Pipe, wgpu::BindGroup, [u32; 3]); 7] = [
        (&p.score, bind(&p.score.l, &[u, st.meta, q, st.ring_k, st.sink_k, kb, &s.sc]), [ncap.div_ceil(32), g, k]),
        (&p.land, bind(&p.land.l, &[u, q, st.k_tilde, &s.fs]), [m.div_ceil(32), g, k]),
        (&p.max, bind(&p.max.l, &[u, st.meta, &s.sc, &s.nm]), [nh, k, 1]),
        (
            &p.far,
            bind(&p.far.l, &[u, st.meta, &s.fs, &s.nm, st.mu, far_s, &s.hd, &s.gn, &s.cf]),
            [nh.div_ceil(8), k, 1],
        ),
        (&p.soft, bind(&p.soft.l, &[u, st.meta, &s.sc, &s.hd, &s.inv]), [nh, k, 1]),
        (
            &p.sum,
            bind(&p.sum.l, &[u, st.meta, &s.sc, st.ring_v, st.sink_v, vb, &s.pz]),
            [(e4 * 8 * g).div_ceil(32), k, 1],
        ),
        (
            &p.add,
            bind(&p.add.l, &[u, st.meta, &s.pz, &s.hd, &s.gn, st.that, &s.inv, &s.cf, st.ring_v, out]),
            [(nh * e4).div_ceil(32), k, 1],
        ),
    ];
    for (pipe, bg, [x, y, z]) in &steps {
        pass.set_pipeline(&pipe.p);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(*x, *y, *z);
    }
}
