//! GLM-5.3's DSA on the wgpu token graph (the CUDA hybrid arm's kernels,
//! `gpu_cuda/hyb.cu` `hy_dsa_*`). Past `index_topk` positions a DSA layer
//! attends only to the positions its indexer selects. Every position adds its
//! normed MLA latent, its layer-normed index key and its pool gate to the
//! layer's caches; every `kp` positions make a pool whose key is the
//! gate-softmax mixture of its keys. A row scores every complete pool, keeps
//! the topk / kp best (score, then the lower pool) and the incomplete tail,
//! and attends over those positions in the absorbed form: the query through
//! kv_b's key part into the latent space, the softmax over the selected
//! latents, the mixture back through kv_b's value part.

use super::*;
use std::sync::{Mutex, OnceLock};

const SRC: &str = r#"
struct P { lora: u32, id: u32, ih: u32, kp: u32, pos: u32, np: u32, keep: u32, npos: u32,
           nh: u32, nope: u32, vd: u32, hd: u32, p0: u32, eps: f32, scale: f32, pad: u32 };
@group(0) @binding(0) var<storage, read>       kva   : array<f32>;
@group(0) @binding(1) var<storage, read>       lw    : array<f32>;
@group(0) @binding(2) var<storage, read>       xk    : array<f32>;
@group(0) @binding(3) var<storage, read>       knw   : array<f32>;
@group(0) @binding(4) var<storage, read>       knb   : array<f32>;
@group(0) @binding(5) var<storage, read_write> lat   : array<f32>;
@group(0) @binding(6) var<storage, read_write> ikc   : array<f32>;
@group(0) @binding(7) var<storage, read_write> gtc   : array<f32>;
@group(0) @binding(8) var<storage, read_write> pooled: array<f32>;
@group(0) @binding(9) var<storage, read>       ape   : array<f32>;
@group(0) @binding(10) var<storage, read_write> sc   : array<f32>;
@group(0) @binding(11) var<storage, read>      iq    : array<f32>;
@group(0) @binding(12) var<storage, read_write> sel  : array<u32>;
@group(0) @binding(13) var<storage, read>      q     : array<f32>;
@group(0) @binding(14) var<storage, read>      wkv   : array<f32>;
@group(0) @binding(15) var<storage, read_write> qlat : array<f32>;
@group(0) @binding(16) var<storage, read_write> olat : array<f32>;
@group(0) @binding(17) var<storage, read_write> outv : array<f32>;
@group(0) @binding(18) var<uniform>             p    : P;
// a prompt chunk's rows of index keys and pool gates (`dsa_put_b`)
@group(0) @binding(19) var<storage, read>      ikr   : array<f32>;
@group(0) @binding(20) var<storage, read>      gtr   : array<f32>;
var<workgroup> red: array<f32, 256>;
var<workgroup> redu: array<u32, 256>;
var<workgroup> shs: array<f32, 2560>;
var<workgroup> cnt: atomic<u32>;
var<workgroup> wv1: f32;
var<workgroup> wv2: f32;

fn sum256(t: u32, v: f32) -> f32 {
    red[t] = v;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (t < st) { red[t] = red[t] + red[t + st]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

// the row's normed latent, layer-normed index key (eps 1e-6) and pool gate
// at its position; xk = [index key | pool gate | head weights]
@compute @workgroup_size(256)
fn dsa_put(@builtin(local_invocation_index) t: u32) {
    let pos = p.pos;
    var s = 0.0;
    for (var i = t; i < p.lora; i = i + 256u) { s = s + kva[i] * kva[i]; }
    let tot = sum256(t, s);
    let inv = 1.0 / sqrt(tot / f32(p.lora) + p.eps);
    for (var i = t; i < p.lora; i = i + 256u) { lat[pos * p.lora + i] = kva[i] * inv * lw[i]; }
    var m1 = 0.0;
    for (var i = t; i < p.id; i = i + 256u) { m1 = m1 + xk[i]; }
    let mean = sum256(t, m1) / f32(p.id);
    var m2 = 0.0;
    for (var i = t; i < p.id; i = i + 256u) {
        let d = xk[i] - mean;
        m2 = m2 + d * d;
    }
    let var_ = sum256(t, m2) / f32(p.id);
    let kinv = 1.0 / sqrt(var_ + 1e-6);
    for (var i = t; i < p.id; i = i + 256u) {
        ikc[pos * p.id + i] = (xk[i] - mean) * kinv * knw[i] + knb[i];
        gtc[pos * p.id + i] = xk[p.id + i];
    }
}

// a prompt chunk's rows (a workgroup a row r at position pos + r): the
// latent (kva rows `hd` apart — the kva row stride rides `hd`), the index key
// and gate rows `id` apart
@compute @workgroup_size(256)
fn dsa_put_b(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) w: vec3<u32>) {
    let r = w.x;
    let pos = p.pos + r;
    let kb = r * p.hd;
    var s = 0.0;
    for (var i = t; i < p.lora; i = i + 256u) { s = s + kva[kb + i] * kva[kb + i]; }
    let tot = sum256(t, s);
    let inv = 1.0 / sqrt(tot / f32(p.lora) + p.eps);
    for (var i = t; i < p.lora; i = i + 256u) { lat[pos * p.lora + i] = kva[kb + i] * inv * lw[i]; }
    let ib = r * p.id;
    var m1 = 0.0;
    for (var i = t; i < p.id; i = i + 256u) { m1 = m1 + ikr[ib + i]; }
    let mean = sum256(t, m1) / f32(p.id);
    var m2 = 0.0;
    for (var i = t; i < p.id; i = i + 256u) {
        let d = ikr[ib + i] - mean;
        m2 = m2 + d * d;
    }
    let var_ = sum256(t, m2) / f32(p.id);
    let kinv = 1.0 / sqrt(var_ + 1e-6);
    for (var i = t; i < p.id; i = i + 256u) {
        ikc[pos * p.id + i] = (ikr[ib + i] - mean) * kinv * knw[i] + knb[i];
        gtc[pos * p.id + i] = gtr[ib + i];
    }
}

// pools [p0, p0 + np): a thread a (pool, channel)
@compute @workgroup_size(256)
fn dsa_pool(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= p.np * p.id) { return; }
    let d = i % p.id;
    let pp = p.p0 + i / p.id;
    var mx = -3.0e38;
    for (var k = 0u; k < p.kp; k = k + 1u) { mx = max(mx, gtc[(pp * p.kp + k) * p.id + d] + ape[k * p.id + d]); }
    var z = 0.0;
    var v = 0.0;
    for (var k = 0u; k < p.kp; k = k + 1u) {
        let e = exp(gtc[(pp * p.kp + k) * p.id + d] + ape[k * p.id + d] - mx);
        z = z + e;
        v = v + e * ikc[(pp * p.kp + k) * p.id + d];
    }
    pooled[pp * p.id + d] = v / max(z, 1.17549435e-38);
}

// a thread a pool: its score
@compute @workgroup_size(256)
fn dsa_score(@builtin(global_invocation_id) g: vec3<u32>) {
    let pp = g.x;
    if (pp >= p.np) { return; }
    let iscale = 1.0 / sqrt(f32(p.id));
    let hscale = 1.0 / sqrt(f32(p.ih));
    var score = 0.0;
    for (var h = 0u; h < p.ih; h = h + 1u) {
        var d = 0.0;
        for (var e = 0u; e < p.id; e = e + 1u) { d = d + iq[h * p.id + e] * pooled[pp * p.id + e]; }
        score = score + xk[2u * p.id + h] * hscale * max(d * iscale, 0.0);
    }
    sc[pp] = score;
}

fn okey(f: f32) -> u32 {
    let b = bitcast<u32>(f);
    return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

// one workgroup: the selected positions in pool order, then the tail;
// sel[keep kp + kp] holds the count
@compute @workgroup_size(256)
fn dsa_select(@builtin(local_invocation_index) t: u32) {
    let np = p.np;
    let keep = p.keep;
    let all = np <= keep;
    var th = 0u;
    var need = keep;
    if (!all) {
        var lo = 0u;
        var hi = 0xFFFFFFFFu;
        loop {
            if (lo >= hi) { break; }
            let mid = lo + ((hi - lo) >> 1u) + 1u;
            var c = 0u;
            for (var q = t; q < np; q = q + 256u) { c = c + select(0u, 1u, okey(sc[q]) >= mid); }
            redu[t] = c;
            workgroupBarrier();
            for (var st = 128u; st > 0u; st = st >> 1u) {
                if (t < st) { redu[t] = redu[t] + redu[t + st]; }
                workgroupBarrier();
            }
            let tot = redu[0];
            workgroupBarrier();
            if (tot >= keep) { lo = mid; } else { hi = mid - 1u; }
        }
        th = lo;
        var gtc_ = 0u;
        for (var q = t; q < np; q = q + 256u) { gtc_ = gtc_ + select(0u, 1u, okey(sc[q]) > th); }
        redu[t] = gtc_;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st = st >> 1u) {
            if (t < st) { redu[t] = redu[t] + redu[t + st]; }
            workgroupBarrier();
        }
        need = keep - redu[0];
        workgroupBarrier();
    }
    let per = (np + 255u) / 256u;
    let a = t * per;
    let b = min(np, a + per);
    var gt = 0u;
    var eq = 0u;
    for (var q = a; q < b; q = q + 1u) {
        if (all) { gt = gt + 1u; continue; }
        let k = okey(sc[q]);
        gt = gt + select(0u, 1u, k > th);
        eq = eq + select(0u, 1u, k == th);
    }
    // exclusive prefix of the ties
    redu[t] = eq;
    workgroupBarrier();
    for (var o = 1u; o < 256u; o = o << 1u) {
        var v = 0u;
        if (t >= o) { v = redu[t - o]; }
        workgroupBarrier();
        redu[t] = redu[t] + v;
        workgroupBarrier();
    }
    let eq_before = redu[t] - eq;
    workgroupBarrier();
    var take_eq = 0u;
    if (need > eq_before) { take_eq = min(eq, need - eq_before); }
    let mine = gt + take_eq;
    redu[t] = mine;
    workgroupBarrier();
    for (var o = 1u; o < 256u; o = o << 1u) {
        var v = 0u;
        if (t >= o) { v = redu[t - o]; }
        workgroupBarrier();
        redu[t] = redu[t] + v;
        workgroupBarrier();
    }
    var at = (redu[t] - mine) * p.kp;
    var eq_seen = eq_before;
    for (var q = a; q < b; q = q + 1u) {
        var take = all;
        if (!all) {
            let k = okey(sc[q]);
            if (k > th) { take = true; } else if (k == th) { take = eq_seen < need; eq_seen = eq_seen + 1u; }
        }
        if (take) {
            for (var j = 0u; j < p.kp; j = j + 1u) { sel[at + j] = q * p.kp + j; }
            at = at + p.kp;
        }
    }
    let chosen = redu[255];
    workgroupBarrier();
    let tail0 = np * p.kp;
    for (var q = tail0 + t; q < p.npos; q = q + 256u) { sel[chosen * p.kp + q - tail0] = q; }
    if (t == 0u) { sel[keep * p.kp + p.kp] = chosen * p.kp + (p.npos - tail0); }
}

// the query's heads through kv_b's key part (a thread a latent channel,
// workgroup row = head)
@compute @workgroup_size(256)
fn dsa_qlat(@builtin(global_invocation_id) g: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let h = w.y;
    let l = g.x;
    if (l >= p.lora) { return; }
    let qb = h * p.hd + (p.hd - p.nope);
    let wb = h * (p.nope + p.vd) * p.lora;
    var a = 0.0;
    for (var d = 0u; d < p.nope; d = d + 1u) { a = a + q[qb + d] * wkv[wb + d * p.lora + l]; }
    qlat[h * p.lora + l] = a;
}

// a workgroup a head: the scores over the selected latents, the softmax,
// the latents' mixture
@compute @workgroup_size(256)
fn dsa_attn(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) w: vec3<u32>) {
    let h = w.x;
    let ns = sel[p.keep * p.kp + p.kp];
    var mx = -3.0e38;
    for (var j = t; j < ns; j = j + 256u) {
        let row = sel[j] * p.lora;
        var d = 0.0;
        for (var e = 0u; e < p.lora; e = e + 1u) { d = d + qlat[h * p.lora + e] * lat[row + e]; }
        d = d * p.scale;
        shs[j] = d;
        mx = max(mx, d);
    }
    red[t] = mx;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (t < st) { red[t] = max(red[t], red[t + st]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var z = 0.0;
    for (var j = t; j < ns; j = j + 256u) {
        let e = exp(shs[j] - m);
        shs[j] = e;
        z = z + e;
    }
    let zt = sum256(t, z);
    let iz = 1.0 / max(zt, 1.17549435e-38);
    for (var l = t; l < p.lora; l = l + 256u) {
        var a = 0.0;
        for (var j = 0u; j < ns; j = j + 1u) { a = a + shs[j] * iz * lat[sel[j] * p.lora + l]; }
        olat[h * p.lora + l] = a;
    }
}

// a thread an output (head h, channel d): kv_b's value part over the head's
// mixture, into the attention's output row
@compute @workgroup_size(256)
fn dsa_out(@builtin(global_invocation_id) g: vec3<u32>) {
    let o = g.x;
    if (o >= p.nh * p.vd) { return; }
    let h = o / p.vd;
    let d = o % p.vd;
    let wb = (h * (p.nope + p.vd) + p.nope + d) * p.lora;
    var a = 0.0;
    for (var l = 0u; l < p.lora; l = l + 1u) { a = a + wkv[wb + l] * olat[h * p.lora + l]; }
    outv[o] = a;
}
"#;

pub(crate) struct Pipes {
    put: wgpu::ComputePipeline,
    put_b: wgpu::ComputePipeline,
    pool: wgpu::ComputePipeline,
    score: wgpu::ComputePipeline,
    select: wgpu::ComputePipeline,
    qlat: wgpu::ComputePipeline,
    attn: wgpu::ComputePipeline,
    out: wgpu::ComputePipeline,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

pub(crate) fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            if c.device.limits().max_storage_buffers_per_shader_stage < 18 || c.device.limits().max_compute_workgroup_storage_size < 14_000 {
                return None;
            }
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("dsa"), source: wgpu::ShaderSource::Wgsl(SRC.into()) });
            let mk = |e: &str| {
                c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e),
                    layout: None,
                    module: &m,
                    entry_point: Some(e),
                    compilation_options: crate::gpu_wgpu::pco(),
                    cache: None,
                })
            };
            let p = Pipes {
                put: mk("dsa_put"),
                put_b: mk("dsa_put_b"),
                pool: mk("dsa_pool"),
                score: mk("dsa_score"),
                select: mk("dsa_select"),
                qlat: mk("dsa_qlat"),
                attn: mk("dsa_attn"),
                out: mk("dsa_out"),
            };
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("DSA kernels rejected: {e}");
                return None;
            }
            Some(p)
        })
        .as_ref()
}

/// A layer's caches by (sequence, layer): latents, index keys, gates, pool
/// keys and pool scores for `cap` positions; grown by doubling.
struct Caches {
    lat: wgpu::Buffer,
    ikc: wgpu::Buffer,
    gtc: wgpu::Buffer,
    pooled: wgpu::Buffer,
    sc: wgpu::Buffer,
    cap: usize,
}

fn caches() -> &'static Mutex<HashMap<(u64, usize), Caches>> {
    static M: OnceLock<Mutex<HashMap<(u64, usize), Caches>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

/// Drop a sequence's caches (`kv_mirror_reset`).
pub(crate) fn reset(kv_id: u64) {
    caches().lock().unwrap().retain(|k, _| k.0 != kv_id);
}

/// A DSA layer's small weights on the card (its four projections ride the
/// graph's own matrices).
pub(crate) struct Weights {
    pub knw: wgpu::Buffer,
    pub knb: wgpu::Buffer,
    pub ape: wgpu::Buffer,
    pub wkv: wgpu::Buffer,
    pub ih: usize,
    pub id: usize,
    pub topk: usize,
    pub kp: usize,
}

/// A layer's small weights on the card, made once per tensor set (kv_b in
/// f32 is 67 MB a GLM layer: never per token).
pub(crate) fn weights(c: &Ctx, d: &crate::gpu::GraphDsa<'_>) -> std::sync::Arc<Weights> {
    static M: OnceLock<Mutex<HashMap<usize, std::sync::Arc<Weights>>>> = OnceLock::new();
    let key = d.kv_b.as_ptr() as usize;
    let m = M.get_or_init(Default::default);
    if let Some(w) = m.lock().unwrap().get(&key) {
        return w.clone();
    }
    let stor = |v: &[f32], label: &'static str| {
        c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let w = std::sync::Arc::new(Weights {
        knw: stor(d.k_norm, "dsa-knw"),
        knb: stor(d.k_bias, "dsa-knb"),
        ape: stor(d.ape, "dsa-ape"),
        wkv: stor(d.kv_b, "dsa-wkv"),
        ih: d.ih,
        id: d.id,
        topk: d.topk,
        kp: d.kpool,
    });
    c.resident.fetch_add((d.kv_b.len() * 4) as u64, std::sync::atomic::Ordering::Relaxed);
    m.lock().unwrap().insert(key, w.clone());
    w
}

/// The layer's caches holding `need` positions (grown with the rows so far
/// copied, `have` of them).
fn ensure(c: &Ctx, key: (u64, usize), lora: usize, id: usize, kp: usize, need: usize, have: usize) -> Option<[wgpu::Buffer; 5]> {
    let mut m = caches().lock().unwrap();
    if let Some(x) = m.get(&key) {
        if x.cap >= need {
            return Some([x.lat.clone(), x.ikc.clone(), x.gtc.clone(), x.pooled.clone(), x.sc.clone()]);
        }
    }
    let old = m.remove(&key);
    let cap = need.max(old.as_ref().map_or(0, |o| 2 * o.cap)).max(4096).next_multiple_of(kp * 64);
    let mk = |label: &'static str, words: usize| {
        c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (words.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let n = Caches {
        lat: mk("dsa-lat", cap * lora),
        ikc: mk("dsa-ikc", cap * id),
        gtc: mk("dsa-gtc", cap * id),
        pooled: mk("dsa-pool", cap / kp * id),
        sc: mk("dsa-sc", cap / kp),
        cap,
    };
    if let Some(o) = old {
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dsa-grow") });
        for (src, dst, w) in [(&o.lat, &n.lat, have * lora), (&o.ikc, &n.ikc, have * id), (&o.gtc, &n.gtc, have * id), (&o.pooled, &n.pooled, have / kp * id)] {
            if w > 0 {
                enc.copy_buffer_to_buffer(src, 0, dst, 0, (w * 4) as u64);
            }
        }
        c.queue.submit(Some(enc.finish()));
    }
    let r = [n.lat.clone(), n.ikc.clone(), n.gtc.clone(), n.pooled.clone(), n.sc.clone()];
    m.insert(key, n);
    Some(r)
}

/// Encode a token's DSA steps for layer `key` at `pos`: the indexer's rows
/// (`xk`: key | gate | weights, `iqb`: the index queries, both projected by
/// the caller), the caches, the pool it completes and, past `topk`, the
/// selection and the absorbed attention into `outv` (nh·vd). False: no
/// kernels, or the caches refused.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    key: (u64, usize),
    w: &Weights,
    dims: (usize, usize, usize, usize, usize, f32, f32),
    pos: usize,
    kva: &wgpu::Buffer,
    kv_a_norm: &wgpu::Buffer,
    xk: &wgpu::Buffer,
    iqb: &wgpu::Buffer,
    q: &wgpu::Buffer,
    outv: &wgpu::Buffer,
) -> bool {
    let Some(pp) = pipes(c) else { return false };
    let (nh, lora, nope, vd, hd, eps, scale) = dims;
    let Some([lat, ikc, gtc, pooled, sc]) = ensure(c, key, lora, w.id, w.kp, pos + 1, pos) else { return false };
    let npos = pos + 1;
    let np = npos / w.kp;
    let keep = w.topk / w.kp;
    let words = |p0: usize, n: usize| -> [u32; 16] {
        [
            lora as u32,
            w.id as u32,
            w.ih as u32,
            w.kp as u32,
            pos as u32,
            n as u32,
            keep as u32,
            npos as u32,
            nh as u32,
            nope as u32,
            vd as u32,
            hd as u32,
            p0 as u32,
            eps.to_bits(),
            scale.to_bits(),
            0,
        ]
    };
    let mk_u = |v: [u32; 16]| c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("dsa-u"), contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::UNIFORM });
    let scratch = |name: &'static str, n: usize| kimi::scratch(c, name, n);
    let qlat = scratch("dsa-qlat", nh * lora);
    let olat = scratch("dsa-olat", nh * lora);
    let sel = scratch("dsa-sel", w.topk + w.kp + 1);
    let all: [(u32, &wgpu::Buffer); 18] = [
        (0, kva),
        (1, kv_a_norm),
        (2, xk),
        (3, &w.knw),
        (4, &w.knb),
        (5, &lat),
        (6, &ikc),
        (7, &gtc),
        (8, &pooled),
        (9, &w.ape),
        (10, &sc),
        (11, iqb),
        (12, &sel),
        (13, q),
        (14, &w.wkv),
        (15, &qlat),
        (16, &olat),
        (17, outv),
    ];
    let bind = |pipe: &wgpu::ComputePipeline, u: &wgpu::Buffer, want: &[u32]| {
        let layout = pipe.get_bind_group_layout(0);
        let mut e: Vec<wgpu::BindGroupEntry> =
            all.iter().filter(|(i, _)| want.contains(i)).map(|(i, b)| wgpu::BindGroupEntry { binding: *i, resource: b.as_entire_binding() }).collect();
        e.push(wgpu::BindGroupEntry { binding: 18, resource: u.as_entire_binding() });
        c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("dsa"), layout: &layout, entries: &e })
    };
    let u0 = mk_u(words(0, np));
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&pp.put);
    pass.set_bind_group(0, &bind(&pp.put, &u0, &[0, 1, 2, 3, 4, 5, 6, 7]), &[]);
    pass.dispatch_workgroups(1, 1, 1);
    if npos % w.kp == 0 {
        let up = mk_u(words(npos / w.kp - 1, 1));
        pass.set_pipeline(&pp.pool);
        pass.set_bind_group(0, &bind(&pp.pool, &up, &[6, 7, 8, 9]), &[]);
        pass.dispatch_workgroups((w.id as u32).div_ceil(256), 1, 1);
    }
    if npos <= w.topk {
        return true;
    }
    if np > 0 {
        pass.set_pipeline(&pp.score);
        pass.set_bind_group(0, &bind(&pp.score, &u0, &[2, 8, 10, 11]), &[]);
        pass.dispatch_workgroups((np as u32).div_ceil(256), 1, 1);
    }
    pass.set_pipeline(&pp.select);
    pass.set_bind_group(0, &bind(&pp.select, &u0, &[10, 12]), &[]);
    pass.dispatch_workgroups(1, 1, 1);
    pass.set_pipeline(&pp.qlat);
    pass.set_bind_group(0, &bind(&pp.qlat, &u0, &[13, 14, 15]), &[]);
    pass.dispatch_workgroups((lora as u32).div_ceil(256), nh as u32, 1);
    pass.set_pipeline(&pp.attn);
    pass.set_bind_group(0, &bind(&pp.attn, &u0, &[5, 12, 15, 16]), &[]);
    pass.dispatch_workgroups(nh as u32, 1, 1);
    pass.set_pipeline(&pp.out);
    pass.set_bind_group(0, &bind(&pp.out, &u0, &[14, 16, 17]), &[]);
    pass.dispatch_workgroups(((nh * vd) as u32).div_ceil(256), 1, 1);
    true
}

/// Encode a prompt chunk's DSA cache rows (`k` rows from `pos0`): the
/// latents of `kva` (rows `kstride` apart), the index keys and gates of
/// `ikb` / `gtb` (rows `id` apart), and the pools the chunk completes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rows(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    key: (u64, usize),
    w: &Weights,
    lora: usize,
    kstride: usize,
    eps: f32,
    pos0: usize,
    k: usize,
    kva: &wgpu::Buffer,
    kv_a_norm: &wgpu::Buffer,
    ikb: &wgpu::Buffer,
    gtb: &wgpu::Buffer,
) -> bool {
    let Some(pp) = pipes(c) else { return false };
    let Some([lat, ikc, gtc, pooled, _]) = ensure(c, key, lora, w.id, w.kp, pos0 + k, pos0) else { return false };
    let u = |pos: usize, p0: usize, np: usize| {
        let v: [u32; 16] = [lora as u32, w.id as u32, w.ih as u32, w.kp as u32, pos as u32, np as u32, 0, 0, 0, 0, 0, kstride as u32, p0 as u32, eps.to_bits(), 0, 0];
        c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("dsa-u"), contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::UNIFORM })
    };
    let bind = |pipe: &wgpu::ComputePipeline, ub: &wgpu::Buffer, pairs: &[(u32, &wgpu::Buffer)]| {
        let layout = pipe.get_bind_group_layout(0);
        let mut e: Vec<wgpu::BindGroupEntry> = pairs.iter().map(|(i, b)| wgpu::BindGroupEntry { binding: *i, resource: b.as_entire_binding() }).collect();
        e.push(wgpu::BindGroupEntry { binding: 18, resource: ub.as_entire_binding() });
        c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("dsa-b"), layout: &layout, entries: &e })
    };
    let ub = u(pos0, 0, 0);
    let (p0, p1) = (pos0 / w.kp, (pos0 + k) / w.kp);
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&pp.put_b);
    pass.set_bind_group(0, &bind(&pp.put_b, &ub, &[(0, kva), (1, kv_a_norm), (3, &w.knw), (4, &w.knb), (5, &lat), (6, &ikc), (7, &gtc), (19, ikb), (20, gtb)]), &[]);
    pass.dispatch_workgroups(k as u32, 1, 1);
    if p1 > p0 {
        let up = u(pos0, p0, p1 - p0);
        pass.set_pipeline(&pp.pool);
        pass.set_bind_group(0, &bind(&pp.pool, &up, &[(6, &ikc), (7, &gtc), (8, &pooled), (9, &w.ape)]), &[]);
        pass.dispatch_workgroups((((p1 - p0) * w.id) as u32).div_ceil(256), 1, 1);
    }
    true
}

#[cfg(test)]
mod tests {
    #[test]
    fn dsa_wgsl_validates() {
        let module = wgpu::naga::front::wgsl::parse_str(super::SRC).expect("DSA WGSL must parse");
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
            .validate(&module)
            .expect("DSA WGSL must validate");
    }
}
