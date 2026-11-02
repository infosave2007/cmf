//! Unlimited-OCR's DASH attention on the wgpu token graph: the router and the
//! block-sparse read set of one position, resident (`uocr` has the semantics).
//!
//! The graph's Full arm still projects q/k/v, rotates them and appends K/V to
//! the layer's mirror. These kernels replace its attend:
//!
//! * `dash_route`: one workgroup holding the whole router of a position, so it
//!   adds no pipeline barrier (each one costs tens of microseconds on NVIDIA,
//!   which dwarfed the arithmetic when these were seven dispatches): the q and
//!   k rows against `n1`, RMSNorm and rotation, the key into the open block,
//!   the scores of the candidate blocks the host listed (so a long document's
//!   other pages cost nothing), entmax-1.5 over the best 16, the best 12 blocks
//!   and their centered log priors;
//! * `dash_aw`, `dash_aw_soft`, `dash_bar`: a closing block's summary, every
//!   32 tokens;
//! * `dash_attend`: one workgroup per head over those blocks and the local
//!   window, into the graph's `attn`.

use super::*;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Rows a query reads, its head stride in the score buffer.
const LIST_MAX: usize = 512;
const BLOCKS_MAX: usize = 4096;
const CHUNK_MAX: usize = crate::gpu::DASH_CHUNK_MAX;

const SRC: &str = r#"
struct DP {
    hidden: u32, d: u32, pos: u32, block: u32,
    ncand: u32, nblk: u32, k: u32, m: u32,
    eps: f32, inv_sqrt: f32, idx_scale: f32, floor: f32,
    sigma: f32, window: u32, nh: u32, hd: u32,
    cap: u32, scale: f32, rows: u32, close: u32,
};
@group(0) @binding(20) var<uniform> p: DP;
// A closing block: the keys against head_cls, their softmax, the mean key.
const ACT: f32 = 10000.0;
@group(0) @binding(6) var<storage, read> co: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> hc: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> aw: array<f32>;
@compute @workgroup_size(32)
fn dash_aw(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = g.x;
    if (t >= p.block) { return; }
    let n = p.d / 4u;
    var acc = vec4<f32>(0.0);
    let at = (p.close % 3u) * p.block;
    for (var e = 0u; e < n; e = e + 1u) { acc = acc + co[(at + t) * n + e] * hc[e]; }
    aw[t] = clamp((acc.x + acc.y + acc.z + acc.w) * p.inv_sqrt, -ACT, ACT);
}
@compute @workgroup_size(1)
fn dash_aw_soft() {
    var mx = -1.0e30;
    for (var t = 0u; t < p.block; t = t + 1u) { mx = max(mx, aw[t]); }
    var den = 0.0;
    for (var t = 0u; t < p.block; t = t + 1u) {
        let v = exp(aw[t] - mx);
        aw[t] = v;
        den = den + v;
    }
    den = 1.0 / max(den, 1.0e-6);
    for (var t = 0u; t < p.block; t = t + 1u) { aw[t] = aw[t] * den; }
}
@group(0) @binding(9) var<storage, read> awr: array<f32>;
@group(0) @binding(10) var<storage, read_write> kbarw: array<vec4<f32>>;
@compute @workgroup_size(32)
fn dash_bar(@builtin(global_invocation_id) g: vec3<u32>) {
    let e = g.x;
    let n = p.d / 4u;
    if (e >= n) { return; }
    var acc = vec4<f32>(0.0);
    let at = (p.close % 3u) * p.block;
    for (var t = 0u; t < p.block; t = t + 1u) { acc = acc + awr[t] * co[(at + t) * n + e]; }
    kbarw[p.close * n + e] = acc;
}

// The router of one position in one workgroup, so it needs no barrier of its
// own: the q and k rows against n1, RMSNorm and rotation, the key into its slot
// of the open block, the candidate scores, entmax-1.5 over the best 16, the read
// set and its centered log priors. 256 lanes, index_dim 128.
const ENTMAX_NEG: f32 = -10000.0;
const NONE: u32 = 0xffffffffu;
@group(0) @binding(0) var<storage, read> pr_n1: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> pr_w: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> qn: array<f32>;
@group(0) @binding(3) var<storage, read> kn: array<f32>;
@group(0) @binding(4) var<storage, read> ang: array<f32>;
@group(0) @binding(5) var<storage, read_write> open: array<f32>;
@group(0) @binding(11) var<storage, read> sk: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read> cands: array<u32>;
@group(0) @binding(14) var<storage, read_write> sc: array<f32>;
@group(0) @binding(12) var<storage, read> coff: array<u32>;
@group(0) @binding(16) var<storage, read_write> sel: array<u32>;
@group(0) @binding(17) var<storage, read_write> sbias: array<f32>;
var<workgroup> rr: array<f32, 256>;
var<workgroup> rs: array<f32, 256>;
var<workgroup> qi: array<vec4<f32>, 32>;
var<workgroup> qs: array<f32, 128>;
@compute @workgroup_size(256)
fn dash_route(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    let pos = p.pos + row;
    let d = p.d;
    let half = d / 2u;
    let n = p.hidden / 4u;
    var v = 0.0;
    {
        var acc = vec4<f32>(0.0);
        let base = lid * n;
        let xb = row * n;
        for (var c = 0u; c < n; c = c + 1u) { acc = acc + pr_w[base + c] * pr_n1[xb + c]; }
        v = acc.x + acc.y + acc.z + acc.w;
    }
    rr[lid] = v;
    rs[lid] = v * v;
    workgroupBarrier();
    // The two sums of squares: lanes 0..128 and 128..256.
    var st = 64u;
    loop {
        if (st == 0u) { break; }
        let l2 = lid % 128u;
        if (l2 < st) { rs[lid] = rs[lid] + rs[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let inv_q = inverseSqrt(rs[0] / f32(d) + p.eps);
    let inv_k = inverseSqrt(rs[128] / f32(d) + p.eps);
    workgroupBarrier();
    if (lid < half) {
        let j = lid;
        let c = cos(ang[row * half + j]);
        let s = sin(ang[row * half + j]);
        let a = clamp(clamp(rr[j] * inv_q, -ACT, ACT) * (1.0 + qn[j]), -ACT, ACT);
        let b = clamp(clamp(rr[half + j] * inv_q, -ACT, ACT) * (1.0 + qn[half + j]), -ACT, ACT);
        qs[j] = a * c - b * s;
        qs[half + j] = b * c + a * s;
    } else if (lid >= d && lid < d + half) {
        let j = lid - d;
        let c = cos(ang[row * half + j]);
        let s = sin(ang[row * half + j]);
        let a = clamp(clamp(rr[d + j] * inv_k, -ACT, ACT) * (1.0 + kn[j]), -ACT, ACT);
        let b = clamp(clamp(rr[d + half + j] * inv_k, -ACT, ACT) * (1.0 + kn[half + j]), -ACT, ACT);
        let at = (((pos / p.block) % 3u) * p.block + pos % p.block) * d;
        open[at + j] = a * c - b * s;
        open[at + half + j] = b * c + a * s;
    }
    workgroupBarrier();
    if (lid < 32u) {
        qi[lid] = vec4<f32>(qs[lid * 4u], qs[lid * 4u + 1u], qs[lid * 4u + 2u], qs[lid * 4u + 3u]);
    }
    workgroupBarrier();
    let lanes = d / 4u;
    let c0 = coff[row];
    let ncand = coff[row + 1u] - c0;
    var i = lid;
    loop {
        if (i >= ncand) { break; }
        let at = cands[c0 + i] * lanes;
        var acc = vec4<f32>(0.0);
        for (var e = 0u; e < lanes; e = e + 1u) { acc = acc + qi[e] * sk[at + e]; }
        sc[c0 + i] = clamp((acc.x + acc.y + acc.z + acc.w) * p.inv_sqrt, -ACT, ACT);
        i = i + 256u;
    }
    storageBarrier();
    if (lid != 0u) { return; }
    var vals: array<f32, 16>;
    var ids: array<u32, 16>;
    var taus: array<f32, 16>;
    let nblk = (select(p.nblk, (pos + p.block) / p.block, p.nblk == 0u));
    let m = clamp(min(p.m, nblk), 1u, 16u);
    for (var t = 0u; t < m; t = t + 1u) { vals[t] = ENTMAX_NEG; ids[t] = NONE; }
    var cnt = 0u;
    for (var t = 0u; t < ncand; t = t + 1u) {
        let x = sc[c0 + t] * p.idx_scale;
        if (cnt == m && x <= vals[m - 1u]) { continue; }
        var j = min(cnt, m - 1u);
        loop {
            if (j == 0u || vals[j - 1u] >= x) { break; }
            vals[j] = vals[j - 1u];
            ids[j] = ids[j - 1u];
            j = j - 1u;
        }
        vals[j] = x;
        ids[j] = cands[c0 + t];
        cnt = min(cnt + 1u, m);
    }
    var count = 0u;
    if (cnt > 0u) {
        let top = vals[0];
        var sum = 0.0;
        var sum2 = 0.0;
        var support = 0u;
        for (var t = 0u; t < m; t = t + 1u) {
            let y = (vals[t] - top) / 2.0;
            sum = sum + y;
            sum2 = sum2 + y * y;
            let rho = f32(t + 1u);
            let mean = sum / rho;
            let ss = rho * (sum2 / rho - mean * mean);
            let delta = max((1.0 - ss) / rho, 0.0);
            let tau = mean - sqrt(delta);
            taus[t] = tau;
            if (tau <= y) { support = support + 1u; }
        }
        let tau_star = taus[max(support, 1u) - 1u];
        let k = clamp(min(p.k, m), 1u, 16u);
        var logp: array<f32, 16>;
        var mu = 0.0;
        var valid = 0u;
        for (var t = 0u; t < k; t = t + 1u) {
            let y = (vals[t] - top) / 2.0 - tau_star;
            logp[t] = log(max(select(0.0, y * y, y > 0.0), p.floor));
            if (ids[t] != NONE) { mu = mu + logp[t]; valid = valid + 1u; }
        }
        mu = mu / f32(max(valid, 1u));
        for (var t = 0u; t < k; t = t + 1u) {
            if (ids[t] == NONE) { continue; }
            sel[row * 17u + 1u + count] = ids[t];
            sbias[row * 16u + count] = (logp[t] - mu) / p.sigma;
            count = count + 1u;
        }
    }
    sel[row * 17u] = count;
}

// Attention over the read set, one workgroup a head.
@group(0) @binding(18) var<storage, read> aq: array<vec4<f32>>;
@group(0) @binding(19) var<storage, read> ak: array<vec4<f32>>;
@group(0) @binding(21) var<storage, read> av: array<vec4<f32>>;
@group(0) @binding(22) var<storage, read_write> ao: array<vec4<f32>>;
@group(0) @binding(23) var<storage, read> asel: array<u32>;
@group(0) @binding(24) var<storage, read> abias: array<f32>;
var<workgroup> wsc: array<f32, 512>;
var<workgroup> wrow: array<u32, 512>;
var<workgroup> red: array<f32, 256>;
var<workgroup> vsum: array<vec4<f32>, 256>;
fn reduce_max(lid: u32) {
    var st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { red[lid] = max(red[lid], red[lid + st]); }
        workgroupBarrier();
        st = st >> 1u;
    }
}
fn reduce_sum(lid: u32) {
    var st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
}
@compute @workgroup_size(256)
fn dash_attend(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let h = w.x;
    let row_q = w.y;
    if (h >= p.nh || row_q >= p.rows) { return; }
    let pos = p.pos + row_q;
    let lanes = p.hd / 4u;
    let sb = row_q * 17u;
    let remote = asel[sb] * p.block;
    let total = remote + p.window;
    let lo = u32(max(i32(pos) + 1 - i32(p.window), 0));
    let qb = (row_q * p.nh + h) * lanes;
    var mym = -1.0e30;
    for (var part = 0u; part < 2u; part = part + 1u) {
        let i = lid + part * 256u;
        var row = NONE;
        var bias = 0.0;
        if (i < remote) {
            let s = i / p.block;
            let r = asel[sb + 1u + s] * p.block + i % p.block;
            // The local window wins on overlap.
            if (r < lo) { row = r; bias = abias[row_q * 16u + s]; }
        } else if (i < total) {
            let r = lo + i - remote;
            if (r <= pos) { row = r; }
        }
        wrow[i] = row;
        if (row != NONE) {
            let kb = (h * p.cap + row) * lanes;
            var acc = vec4<f32>(0.0);
            for (var e = 0u; e < lanes; e = e + 1u) { acc = acc + aq[qb + e] * ak[kb + e]; }
            let v = (acc.x + acc.y + acc.z + acc.w) * p.scale + bias;
            wsc[i] = v;
            mym = max(mym, v);
        }
    }
    red[lid] = mym;
    workgroupBarrier();
    reduce_max(lid);
    let mx = red[0];
    workgroupBarrier();
    var s = 0.0;
    for (var part = 0u; part < 2u; part = part + 1u) {
        let i = lid + part * 256u;
        if (wrow[i] != NONE) {
            let e = exp(wsc[i] - mx);
            wsc[i] = e;
            s = s + e;
        }
    }
    red[lid] = s;
    workgroupBarrier();
    reduce_sum(lid);
    let inv = 1.0 / red[0];
    workgroupBarrier();
    // A loop over the rows on one warp stalls on every load, so the rows split
    // over the eight warps: lane = (warp lid >> 5, vec4 lane lid & 31), then a
    // tree over the warps.
    let quad = lid & 31u;
    let warp = lid >> 5u;
    var acc = vec4<f32>(0.0);
    if (quad < lanes) {
        var i = warp;
        loop {
            if (i >= total) { break; }
            let r = wrow[i];
            if (r != NONE) { acc = acc + wsc[i] * av[(h * p.cap + r) * lanes + quad]; }
            i = i + 8u;
        }
    }
    vsum[lid] = acc;
    workgroupBarrier();
    var wst = 4u;
    loop {
        if (wst == 0u) { break; }
        if (warp < wst) { vsum[lid] = vsum[lid] + vsum[lid + wst * 32u]; }
        workgroupBarrier();
        wst = wst >> 1u;
    }
    if (warp == 0u && quad < lanes) {
        ao[qb + quad] = vsum[lid] * inv;
    }
}

// A prompt chunk's rotation and mirror append: every row by its own angles.
@group(0) @binding(25) var<storage, read> br_q: array<f32>;
@group(0) @binding(26) var<storage, read> br_k: array<f32>;
@group(0) @binding(27) var<storage, read> br_v: array<f32>;
@group(0) @binding(28) var<storage, read> br_ang: array<f32>;
@group(0) @binding(29) var<storage, read_write> br_qo: array<f32>;
@group(0) @binding(30) var<storage, read_write> br_kc: array<f32>;
@group(0) @binding(31) var<storage, read_write> br_vc: array<f32>;
@compute @workgroup_size(64)
fn dash_rope_b(@builtin(global_invocation_id) g: vec3<u32>) {
    let hd = p.hd;
    let half = hd / 2u;
    let per = p.nh * half;
    let row = g.x / per;
    if (row >= p.rows) { return; }
    let h = (g.x / half) % p.nh;
    let j = g.x % half;
    let a = br_ang[row * half + j];
    let c = cos(a);
    let s = sin(a);
    let base = (row * p.nh + h) * hd;
    let q0 = br_q[base + j];
    let q1 = br_q[base + half + j];
    br_qo[base + j] = q0 * c - q1 * s;
    br_qo[base + half + j] = q1 * c + q0 * s;
    let pos = p.pos + row;
    let mb = (h * p.cap + pos) * hd;
    let k0 = br_k[base + j];
    let k1 = br_k[base + half + j];
    br_kc[mb + j] = k0 * c - k1 * s;
    br_kc[mb + half + j] = k1 * c + k0 * s;
    br_vc[mb + j] = br_v[base + j];
    br_vc[mb + half + j] = br_v[base + half + j];
}
"#;

struct Pipe {
    p: wgpu::ComputePipeline,
    l: wgpu::BindGroupLayout,
}

struct Pipes {
    route: Pipe,
    aw: Pipe,
    aw_soft: Pipe,
    bar: Pipe,
    attend: Pipe,
    rope_b: Pipe,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("dash"),
                source: wgpu::ShaderSource::Wgsl(SRC.into()),
            });
            let mk = |entry: &str| {
                let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &m,
                    entry_point: Some(entry),
                    compilation_options: crate::gpu_wgpu::pco(),
                    cache: None,
                });
                let l = p.get_bind_group_layout(0);
                Pipe { p, l }
            };
            let out = Pipes {
                route: mk("dash_route"),
                aw: mk("dash_aw"),
                aw_soft: mk("dash_aw_soft"),
                bar: mk("dash_bar"),
                attend: mk("dash_attend"),
                rope_b: mk("dash_rope_b"),
            };
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("dash kernels rejected: {e}");
                return None;
            }
            Some(out)
        })
        .as_ref()
}

/// One layer's router on the device.
struct LayerDev {
    w: wgpu::Buffer,
    qn: wgpu::Buffer,
    kn: wgpu::Buffer,
    hc: wgpu::Buffer,
    open: wgpu::Buffer,
    kbar: wgpu::Buffer,
    aw: wgpu::Buffer,
    sc: wgpu::Buffer,
    sel: wgpu::Buffer,
    bias: wgpu::Buffer,
    sc_cap: usize,
    u: wgpu::Buffer,
    /// The uniform of up to two blocks closing in one chunk.
    uc: [wgpu::Buffer; 2],
    /// Bind groups over the graph's buffers, while those stay the same.
    bgs: Option<(Vec<wgpu::Buffer>, LayerBgs)>,
}

struct LayerBgs {
    route: wgpu::BindGroup,
    aw: [wgpu::BindGroup; 2],
    aw_soft: wgpu::BindGroup,
    bar: [wgpu::BindGroup; 2],
    attend: wgpu::BindGroup,
}

/// Candidate lists of the chunk, shared by the layers.
struct Shared {
    cands: wgpu::Buffer,
    cands_cap: usize,
    coff: wgpu::Buffer,
    /// Same `kv_id`/position already written this frame.
    written: (u64, usize, usize),
}

static STATE: Mutex<Option<(HashMap<(u64, usize), LayerDev>, Option<Shared>)>> = Mutex::new(None);

fn buf(c: &Ctx, label: &str, size: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    c.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(4) as u64, usage, mapped_at_creation: false })
}

fn stor_f32(c: &Ctx, label: &str, data: &[f32]) -> wgpu::Buffer {
    let b = buf(c, label, data.len() * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
    b
}

/// Refuse a geometry the kernels cannot hold, before any device state moves.
pub(crate) fn admits(d: &crate::gpu::GraphDash<'_>, hidden: usize, nh: usize, hd: usize) -> bool {
    let idim = d.q_norm.len();
    hidden % 4 == 0
        && hd % 4 == 0
        && idim == 128
        && d.block <= 64
        && d.k <= 16
        && hd / 4 <= 32
        && d.block * d.k + d.block * d.local_blocks <= LIST_MAX
        && d.qk_proj.len() == 2 * idim * hidden
        && nh > 0
}

/// Everything an encode needs from the graph.
pub(crate) struct DashEnc<'a> {
    pub d: &'a crate::gpu::GraphDash<'a>,
    pub key: (u64, usize),
    pub n1: &'a wgpu::Buffer,
    pub qout: &'a wgpu::Buffer,
    pub kbuf: &'a wgpu::Buffer,
    pub vbuf: &'a wgpu::Buffer,
    pub invf: &'a wgpu::Buffer,
    pub attn: &'a wgpu::Buffer,
    pub cap: usize,
    pub pos0: usize,
    pub k: usize,
    pub nh: usize,
    pub hd: usize,
    pub hidden: usize,
    pub scale: f32,
    pub eps: f32,
}

/// The chunk's raw projections, for the rotation pass of a prompt chunk.
pub(crate) struct DashRaw<'a> {
    pub q: &'a wgpu::Buffer,
    pub k: &'a wgpu::Buffer,
    pub v: &'a wgpu::Buffer,
}

/// Encode the router and attend of `e.k` rows at `e.pos0`. With `raw` (a
/// prompt chunk) this also rotates the rows and appends them to the mirror;
/// without (the token graph) the graph's own rope and append ran already.
pub(crate) fn encode_layer(c: &Ctx, enc: &mut wgpu::CommandEncoder, e: &DashEnc<'_>, raw: Option<&DashRaw<'_>>) -> bool {
    let Some(p) = pipes(c) else { return false };
    let d = e.d;
    let idim = d.q_norm.len();
    let (k, nh, hd) = (e.k, e.nh, e.hd);
    let end = e.pos0 + k;
    if k == 0 || k > CHUNK_MAX || (end - 1) / d.block >= BLOCKS_MAX {
        return false;
    }
    let mut g = STATE.lock().unwrap();
    let (layers, shared) = g.get_or_insert_with(|| (HashMap::new(), None));
    let st = wgpu::BufferUsages::STORAGE;
    let sh = shared.get_or_insert_with(|| Shared {
        cands: buf(c, "dash-cands", BLOCKS_MAX * 4, st | wgpu::BufferUsages::COPY_DST),
        cands_cap: BLOCKS_MAX,
        coff: buf(c, "dash-coff", (CHUNK_MAX + 1) * 4, st | wgpu::BufferUsages::COPY_DST),
        written: (u64::MAX, 0, 0),
    });
    if d.cands.len() > sh.cands_cap {
        sh.cands_cap = d.cands.len().next_power_of_two();
        sh.cands = buf(c, "dash-cands", sh.cands_cap * 4, st | wgpu::BufferUsages::COPY_DST);
    }
    let frame = (e.key.0, e.pos0, k);
    if sh.written != frame || e.key.1 == 0 {
        if !d.cands.is_empty() {
            c.queue.write_buffer(&sh.cands, 0, bytemuck::cast_slice(d.cands));
        }
        c.queue.write_buffer(&sh.coff, 0, bytemuck::cast_slice(d.coff));
        sh.written = frame;
    }
    let sc_cap = sh.cands_cap;
    let l = layers.entry(e.key).or_insert_with(|| LayerDev {
        w: stor_f32(c, "dash-w", d.qk_proj),
        qn: stor_f32(c, "dash-qn", d.q_norm),
        kn: stor_f32(c, "dash-kn", d.k_norm),
        hc: stor_f32(c, "dash-hc", d.head_cls),
        open: buf(c, "dash-open", 3 * d.block * idim * 4, st),
        kbar: buf(c, "dash-kbar", BLOCKS_MAX * idim * 4, st),
        aw: buf(c, "dash-aw", d.block * 4, st),
        sc: buf(c, "dash-sc", sc_cap * 4, st),
        sc_cap,
        sel: buf(c, "dash-sel", CHUNK_MAX * 17 * 4, st),
        bias: buf(c, "dash-bias", CHUNK_MAX * 16 * 4, st),
        u: buf(c, "dash-u", 80, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
        uc: [
            buf(c, "dash-uc", 80, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            buf(c, "dash-uc", 80, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
        ],
        bgs: None,
    });
    if l.sc_cap < sc_cap {
        l.sc = buf(c, "dash-sc", sc_cap * 4, st);
        l.sc_cap = sc_cap;
        l.bgs = None;
    }
    let mut words: [u32; 20] = [
        e.hidden as u32,
        idim as u32,
        e.pos0 as u32,
        d.block as u32,
        0,
        d.nblk as u32,
        d.k as u32,
        16,
        e.eps.to_bits(),
        (idim as f32).sqrt().recip().to_bits(),
        d.idx_scale.to_bits(),
        d.floor.to_bits(),
        d.sigma.to_bits(),
        (d.block * d.local_blocks) as u32,
        nh as u32,
        hd as u32,
        e.cap as u32,
        e.scale.to_bits(),
        k as u32,
        0,
    ];
    c.queue.write_buffer(&l.u, 0, bytemuck::cast_slice(&words));
    let graph_bufs: [&wgpu::Buffer; 8] = [e.n1, e.qout, e.kbuf, e.vbuf, e.invf, e.attn, &sh.cands, &sh.coff];
    let fresh = l.bgs.as_ref().is_none_or(|(ks, _)| ks.iter().zip(graph_bufs).any(|(a, b)| a != b));
    if fresh {
        let bgs = LayerBgs {
            route: bind_pairs(
                c,
                &p.route.l,
                &[
                    (0, e.n1),
                    (1, &l.w),
                    (2, &l.qn),
                    (3, &l.kn),
                    (4, e.invf),
                    (5, &l.open),
                    (11, &l.kbar),
                    (12, &sh.coff),
                    (13, &sh.cands),
                    (14, &l.sc),
                    (16, &l.sel),
                    (17, &l.bias),
                    (20, &l.u),
                ],
            ),
            aw: [
                bind_pairs(c, &p.aw.l, &[(6, &l.open), (7, &l.hc), (8, &l.aw), (20, &l.uc[0])]),
                bind_pairs(c, &p.aw.l, &[(6, &l.open), (7, &l.hc), (8, &l.aw), (20, &l.uc[1])]),
            ],
            aw_soft: bind_pairs(c, &p.aw_soft.l, &[(8, &l.aw), (20, &l.u)]),
            bar: [
                bind_pairs(c, &p.bar.l, &[(6, &l.open), (9, &l.aw), (10, &l.kbar), (20, &l.uc[0])]),
                bind_pairs(c, &p.bar.l, &[(6, &l.open), (9, &l.aw), (10, &l.kbar), (20, &l.uc[1])]),
            ],
            attend: bind_pairs(
                c,
                &p.attend.l,
                &[(18, e.qout), (19, e.kbuf), (21, e.vbuf), (22, e.attn), (23, &l.sel), (24, &l.bias), (20, &l.u)],
            ),
        };
        l.bgs = Some((graph_bufs.iter().map(|b| (*b).clone()).collect(), bgs));
    }
    let bgs = &l.bgs.as_ref().expect("set above").1;
    // A chunk's raw buffers come from a scratch pool; bind them per call.
    let rope_bg = raw.map(|r| {
        bind_pairs(
            c,
            &p.rope_b.l,
            &[
                (25, r.q),
                (26, r.k),
                (27, r.v),
                (28, e.invf),
                (29, e.qout),
                (30, e.kbuf),
                (31, e.vbuf),
                (20, &l.u),
            ],
        )
    });
    let mut pass = begin_pass(enc);
    if let Some(b) = &rope_bg {
        pass.set_pipeline(&p.rope_b.p);
        pass.set_bind_group(0, b, &[]);
        pass.dispatch_workgroups(((k * nh * hd / 2) as u32).div_ceil(64), 1, 1);
    }
    pass.set_pipeline(&p.route.p);
    pass.set_bind_group(0, &bgs.route, &[]);
    pass.dispatch_workgroups(k as u32, 1, 1);
    // Blocks closing in this chunk. The rows above read older blocks only:
    // a remote block is at least `local_blocks` behind, and a chunk spans
    // fewer than that.
    let mut slot = 0usize;
    for b in e.pos0 / d.block..=(end - 1) / d.block {
        if (b + 1) * d.block > end {
            break;
        }
        words[19] = b as u32;
        c.queue.write_buffer(&l.uc[slot], 0, bytemuck::cast_slice(&words));
        pass.set_pipeline(&p.aw.p);
        pass.set_bind_group(0, &bgs.aw[slot], &[]);
        pass.dispatch_workgroups((d.block as u32).div_ceil(32), 1, 1);
        pass.set_pipeline(&p.aw_soft.p);
        pass.set_bind_group(0, &bgs.aw_soft, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&p.bar.p);
        pass.set_bind_group(0, &bgs.bar[slot], &[]);
        pass.dispatch_workgroups(((idim / 4) as u32).div_ceil(32), 1, 1);
        slot += 1;
        if slot == l.uc.len() {
            break;
        }
    }
    pass.set_pipeline(&p.attend.p);
    pass.set_bind_group(0, &bgs.attend, &[]);
    pass.dispatch_workgroups(nh as u32, k as u32, 1);
    true
}
