//! Batch-graph kernels for sealed O(1) layers (the post-seal prompt).
//!
//! The batched prompt graph ran a sealed O(1) layer as a loop of six
//! dispatches per row (rope, far insert, ring push, attend, gate, blit),
//! each on groups × heads workgroups and each depending on the previous
//! row's state: a 64-row chunk of Qwen3.5-0.8B through wgpu/Metal took
//! 600 ms against 165 ms for the same chunk with exact attention.
//!
//! These kernels take the whole chunk at once (spec: `nystrom.rs`
//! `step_group` once per row, which is what the loop dispatches):
//!
//! 1. `o1b_far_e` — every key the chunk evicts from the window against
//!    every landmark query. The far max moves straight to its post-chunk
//!    value M, and each row gets its prefix of the far mass. A mid-chunk
//!    row then reads its far terms relative to M instead of the running
//!    max; that scales its far and near terms by one positive factor, so
//!    the attention output is the serial one up to rounding.
//! 2. `o1b_attend` — one workgroup per (row, group, head): sinks, the
//!    row's window (ring rows not yet evicted, then chunk rows up to it)
//!    and the far readout, whose in-chunk part is a causal sum over the
//!    rows evicted so far.
//! 3. `o1b_far_commit`, `o1b_ring_commit`, `o1b_meta` — the state after
//!    the chunk, which the next chunk or the token graph continues from.
//!
//! A chunk longer than [`OB_K`] rows runs as consecutive sub-chunks.
//! `CMF_O1_BATCH=0` keeps the per-row loop.

use super::*;

/// Rows per sub-chunk: the lanes of `o1b_far_e` and the in-chunk far
/// coefficients of `o1b_attend`.
pub(crate) const OB_K: usize = 64;

pub(crate) const O1_BATCH_WGSL: &str = r#"
const OB_K: u32 = 64u;
struct OBP {
    hpg: u32, m: u32, w: u32, nsrect: u32,
    d: u32, dv: u32, scale: f32, k: u32,
    nh: u32, ng: u32, r0: u32, _p: u32,
};
@group(0) @binding(0)  var<storage, read_write> ob_meta : array<u32>;
@group(0) @binding(1)  var<storage, read_write> ob_rk   : array<f32>;
@group(0) @binding(2)  var<storage, read_write> ob_rv   : array<f32>;
@group(0) @binding(3)  var<storage, read>       ob_sk   : array<f32>;
@group(0) @binding(4)  var<storage, read>       ob_sv   : array<f32>;
@group(0) @binding(5)  var<storage, read>       ob_kt   : array<f32>;
@group(0) @binding(6)  var<storage, read>       ob_qt   : array<f32>;
@group(0) @binding(7)  var<storage, read>       ob_mu   : array<f32>;
@group(0) @binding(8)  var<storage, read_write> ob_mz   : array<f32>;
@group(0) @binding(9)  var<storage, read_write> ob_th   : array<f32>;
@group(0) @binding(10) var<storage, read>       ob_q    : array<f32>;
@group(0) @binding(11) var<storage, read>       ob_kb   : array<f32>;
@group(0) @binding(12) var<storage, read>       ob_vb   : array<f32>;
// E [ghm][OB_K] | far-mass prefix [ghm][OB_K] | M [ghm] | rescale R [ghm]
@group(0) @binding(13) var<storage, read_write> ob_s    : array<f32>;
@group(0) @binding(14) var<storage, read_write> ob_out  : array<f32>;
@group(0) @binding(15) var<uniform>             ob_p    : OBP;

// Group window (meta slot 3; 0 = the layer's). ob_p.w is the ring stride.
fn ob_wg(g: u32) -> u32 {
    let gw = ob_meta[g * 4u + 3u];
    return select(ob_p.w, gw, gw != 0u);
}
fn ob_ghm() -> u32 { return ob_p.ng * ob_p.hpg * ob_p.m; }

var<workgroup> fe_l: array<f32, 64>;

// One workgroup per (group, head, landmark); lane j = chunk row j. Row j
// evicts the oldest window key once the ring is full (j >= wg - len0);
// that key is ring logical index j - j0, i.e. slot (head0 + j - j0) % wg.
@compute @workgroup_size(64)
fn o1b_far_e(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let hm = ob_p.hpg * ob_p.m;
    let g = wid.x / hm;
    let rr = wid.x % hm;
    let h = rr / ob_p.m;
    let a = rr % ob_p.m;
    let len0 = ob_meta[g * 4u];
    let head0 = ob_meta[g * 4u + 1u];
    let wg = ob_wg(g);
    let j0 = wg - len0;
    let d = ob_p.d;
    let k = ob_p.k;
    let gha = (g * ob_p.hpg + h) * ob_p.m + a;
    var l = -3.0e38;
    if (lid < k && lid >= j0) {
        let qb = gha * d;
        let kb = (g * ob_p.w + (head0 + lid - j0) % wg) * d;
        var acc = 0.0;
        for (var t = 0u; t < d; t = t + 1u) { acc = acc + ob_qt[qb + t] * ob_rk[kb + t]; }
        l = acc * ob_p.scale;
    }
    fe_l[lid] = l;
    workgroupBarrier();
    if (lid == 0u) {
        let mzb = (g * ob_p.hpg + h) * 2u * ob_p.m;
        let m_old = ob_mz[mzb + a];
        var mx = m_old;
        for (var j = j0; j < k; j = j + 1u) { mx = max(mx, fe_l[j]); }
        // An empty far field starts at -inf: no eviction keeps it (R = 1),
        // the first one rescales the zero mass by exp(-inf) = 0.
        let r = select(exp(m_old - mx), 1.0, mx == m_old);
        var z = ob_mz[mzb + ob_p.m + a] * r;
        let ghm = ob_ghm();
        for (var j = 0u; j < k; j = j + 1u) {
            var e = 0.0;
            if (j >= j0) { e = exp(fe_l[j] - mx); }
            z = z + e;
            ob_s[gha * OB_K + j] = e;
            ob_s[(ghm + gha) * OB_K + j] = z;
        }
        ob_s[2u * ghm * OB_K + gha] = mx;
        ob_s[2u * ghm * OB_K + ghm + gha] = r;
    }
}

var<workgroup> oba_qs:  array<f32, 256>;
// sink + window up to 2052 rows, as o1_attend.
var<workgroup> oba_scr: array<f32, 2052>;
var<workgroup> oba_f:   array<f32, 32>;
var<workgroup> oba_u:   array<f32, 32>;
var<workgroup> oba_g:   array<f32, 32>;
var<workgroup> oba_cf:  array<f32, 64>;
var<workgroup> oba_red: array<f32, 256>;
var<workgroup> oba_sc:  array<f32, 4>;

fn oba_key_dot(g: u32, s: u32, ns: u32, lo: u32, len0: u32, head0: u32, wg: u32) -> f32 {
    let d = ob_p.d;
    var acc = 0.0;
    if (s < ns) {
        let kb = (g * ns + s) * d;
        for (var j = 0u; j < d; j = j + 1u) { acc = acc + oba_qs[j] * ob_sk[kb + j]; }
        return acc;
    }
    let x = lo + (s - ns);
    if (x < len0) {
        let kb = (g * ob_p.w + (head0 + x) % wg) * d;
        for (var j = 0u; j < d; j = j + 1u) { acc = acc + oba_qs[j] * ob_rk[kb + j]; }
    } else {
        let kb = ((ob_p.r0 + x - len0) * ob_p.ng + g) * d;
        for (var j = 0u; j < d; j = j + 1u) { acc = acc + oba_qs[j] * ob_kb[kb + j]; }
    }
    return acc;
}

fn oba_val(g: u32, s: u32, ns: u32, lo: u32, len0: u32, head0: u32, wg: u32, t: u32) -> f32 {
    let dv = ob_p.dv;
    if (s < ns) { return ob_sv[(g * ns + s) * dv + t]; }
    let x = lo + (s - ns);
    if (x < len0) { return ob_rv[(g * ob_p.w + (head0 + x) % wg) * dv + t]; }
    return ob_vb[((ob_p.r0 + x - len0) * ob_p.ng + g) * dv + t];
}

// One workgroup per (group·head, row): o1_attend for row i of the chunk.
@compute @workgroup_size(256)
fn o1b_attend(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let g = wid.x / ob_p.hpg;
    let h = wid.x % ob_p.hpg;
    let i = wid.y;
    let d = ob_p.d;
    let dv = ob_p.dv;
    let m = ob_p.m;
    let ns = ob_p.nsrect & 0xFFu;
    let rect_fm = (ob_p.nsrect >> 8u) != 0u;
    let len0 = ob_meta[g * 4u];
    let head0 = ob_meta[g * 4u + 1u];
    let farl0 = ob_meta[g * 4u + 2u];
    let wg = ob_wg(g);
    let j0 = wg - len0;
    // Row i's window: the last min(wg, len0+i+1) of [ring ++ chunk rows].
    let win = min(wg, len0 + i + 1u);
    let lo = len0 + i + 1u - win;
    let n = ns + win;
    var farl = farl0;
    if (i >= j0) { farl = farl + (i - j0 + 1u); }
    let gh = g * ob_p.hpg + h;
    let row = ob_p.r0 + i;
    let ghm = ob_ghm();
    var t = lid;
    loop {
        if (t >= d) { break; }
        oba_qs[t] = ob_q[(row * ob_p.nh + gh) * d + t];
        t = t + 256u;
    }
    workgroupBarrier();
    var s = lid;
    loop {
        if (s >= n) { break; }
        oba_scr[s] = oba_key_dot(g, s, ns, lo, len0, head0, wg) * ob_p.scale;
        s = s + 256u;
    }
    if (lid < m && farl > 0u) {
        var acc = 0.0;
        let ktb = (g * m + lid) * d;
        for (var j = 0u; j < d; j = j + 1u) { acc = acc + oba_qs[j] * ob_kt[ktb + j]; }
        oba_f[lid] = acc * ob_p.scale;
    }
    workgroupBarrier();
    var c = -3.0e38;
    s = lid;
    loop {
        if (s >= n) { break; }
        c = max(c, oba_scr[s]);
        s = s + 256u;
    }
    oba_red[lid] = c;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { oba_red[lid] = max(oba_red[lid], oba_red[lid + stride]); }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) {
        oba_sc[0] = oba_red[0];
        oba_sc[1] = 0.0;
        if (farl > 0u) {
            var f = -3.0e38;
            for (var a = 0u; a < m; a = a + 1u) { f = max(f, oba_f[a]); }
            for (var a = 0u; a < m; a = a + 1u) { oba_f[a] = exp(oba_f[a] - f); }
            oba_sc[1] = f;
        }
    }
    workgroupBarrier();
    if (lid < m && farl > 0u) {
        let b = lid;
        var uacc = 0.0;
        for (var a = 0u; a < m; a = a + 1u) {
            uacc = uacc + oba_f[a] * ob_mu[(gh * m + a) * m + b];
        }
        if (rect_fm) { uacc = max(uacc, 0.0); }
        oba_u[b] = uacc;
    }
    workgroupBarrier();
    if (lid == 0u) {
        let c0 = oba_sc[0];
        let f = oba_sc[1];
        var c_all = c0;
        var far_den = 0.0;
        var have_far = 0.0;
        if (farl > 0u) {
            let mb = 2u * ghm * OB_K;
            for (var b = 0u; b < m; b = b + 1u) {
                c_all = max(c_all, f + ob_s[mb + gh * m + b]);
            }
            for (var b = 0u; b < m; b = b + 1u) {
                let gain = oba_u[b] * exp(f + ob_s[mb + gh * m + b] - c_all);
                oba_u[b] = gain;
                oba_g[b] = gain * ob_s[mb + ghm + gh * m + b];
                far_den = far_den + gain * ob_s[(ghm + gh * m + b) * OB_K + i];
            }
            if (far_den >= 0.0) { have_far = 1.0; } else { far_den = 0.0; }
        }
        oba_sc[0] = c_all;
        oba_sc[1] = far_den;
        oba_sc[3] = have_far;
    }
    workgroupBarrier();
    let c_all = oba_sc[0];
    let have_far = oba_sc[3] > 0.5;
    var part = 0.0;
    s = lid;
    loop {
        if (s >= n) { break; }
        let pv = exp(oba_scr[s] - c_all);
        oba_scr[s] = pv;
        part = part + pv;
        s = s + 256u;
    }
    oba_red[lid] = part;
    // In-chunk far rows j0..i: their weight is sum_b gain_b E_bj.
    if (lid < OB_K) {
        var cf = 0.0;
        if (have_far && lid >= j0 && lid <= i) {
            for (var b = 0u; b < m; b = b + 1u) {
                cf = cf + oba_u[b] * ob_s[(gh * m + b) * OB_K + lid];
            }
        }
        oba_cf[lid] = cf;
    }
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { oba_red[lid] = oba_red[lid] + oba_red[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let den = max(oba_sc[1] + oba_red[0], 1e-30);
    t = lid;
    loop {
        if (t >= dv) { break; }
        var acc = 0.0;
        if (have_far) {
            for (var b = 0u; b < m; b = b + 1u) {
                acc = acc + oba_g[b] * ob_th[(gh * m + b) * dv + t];
            }
            if (i >= j0) {
                for (var j = j0; j <= i; j = j + 1u) {
                    acc = acc + oba_cf[j] * ob_rv[(g * ob_p.w + (head0 + j - j0) % wg) * dv + t];
                }
            }
        }
        for (var sx = 0u; sx < n; sx = sx + 1u) {
            acc = acc + oba_scr[sx] * oba_val(g, sx, ns, lo, len0, head0, wg, t);
        }
        ob_out[(row * ob_p.nh + gh) * dv + t] = acc / den;
        t = t + 256u;
    }
}

// One workgroup per (group, head, landmark): the far numerator/mass after
// the chunk — rescaled to M, plus every evicted row. Runs after o1b_attend
// (which reads the pre-chunk numerator) and before o1b_ring_commit (which
// overwrites the evicted rows).
@compute @workgroup_size(256)
fn o1b_far_commit(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let hm = ob_p.hpg * ob_p.m;
    let g = wid.x / hm;
    let rr = wid.x % hm;
    let h = rr / ob_p.m;
    let b = rr % ob_p.m;
    let len0 = ob_meta[g * 4u];
    let head0 = ob_meta[g * 4u + 1u];
    let wg = ob_wg(g);
    let j0 = wg - len0;
    let k = ob_p.k;
    if (j0 >= k) { return; }
    let dv = ob_p.dv;
    let ghb = (g * ob_p.hpg + h) * ob_p.m + b;
    let ghm = ob_ghm();
    let r = ob_s[2u * ghm * OB_K + ghm + ghb];
    var t = lid;
    loop {
        if (t >= dv) { break; }
        var acc = ob_th[ghb * dv + t] * r;
        for (var j = j0; j < k; j = j + 1u) {
            acc = acc + ob_s[ghb * OB_K + j] * ob_rv[(g * ob_p.w + (head0 + j - j0) % wg) * dv + t];
        }
        ob_th[ghb * dv + t] = acc;
        t = t + 256u;
    }
    if (lid == 0u) {
        let mzb = (g * ob_p.hpg + h) * 2u * ob_p.m;
        ob_mz[mzb + b] = ob_s[2u * ghm * OB_K + ghb];
        ob_mz[mzb + ob_p.m + b] = ob_s[(ghm + ghb) * OB_K + k - 1u];
    }
}

// One workgroup per (group, row): the chunk's rotated K and V into the
// ring. Row j lands where row j's eviction was: slot (head0 + len0 + j) % wg.
@compute @workgroup_size(256)
fn o1b_ring_commit(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let g = wid.x;
    let j = wid.y;
    let len0 = ob_meta[g * 4u];
    let head0 = ob_meta[g * 4u + 1u];
    let wg = ob_wg(g);
    let slot = (head0 + len0 + j) % wg;
    let row = ob_p.r0 + j;
    var t = lid;
    loop {
        if (t >= ob_p.d) { break; }
        ob_rk[(g * ob_p.w + slot) * ob_p.d + t] = ob_kb[(row * ob_p.ng + g) * ob_p.d + t];
        t = t + 256u;
    }
    t = lid;
    loop {
        if (t >= ob_p.dv) { break; }
        ob_rv[(g * ob_p.w + slot) * ob_p.dv + t] = ob_vb[(row * ob_p.ng + g) * ob_p.dv + t];
        t = t + 256u;
    }
}

// Ring length, head and far count after k pushes (o1_push k times).
@compute @workgroup_size(64)
fn o1b_meta(@builtin(local_invocation_index) lid: u32) {
    let g = lid;
    if (g >= ob_p.ng) { return; }
    let len0 = ob_meta[g * 4u];
    let head0 = ob_meta[g * 4u + 1u];
    let wg = ob_wg(g);
    let tot = len0 + ob_p.k;
    if (tot <= wg) {
        ob_meta[g * 4u] = tot;
    } else {
        let ev = tot - wg;
        ob_meta[g * 4u] = wg;
        ob_meta[g * 4u + 1u] = (head0 + ev) % wg;
        ob_meta[g * 4u + 2u] = ob_meta[g * 4u + 2u] + ev;
    }
}
"#;

pub(crate) struct Pipes {
    far_e: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    attend: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    far_commit: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    ring_commit: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    meta: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
}

fn build(c: &Ctx) -> Option<Pipes> {
    if std::env::var("CMF_O1_BATCH").as_deref() == Ok("0") {
        return None;
    }
    let lim = c.device.limits();
    if lim.max_storage_buffers_per_shader_stage < 15
        || lim.max_compute_workgroup_storage_size < 2800 * 4
    {
        tracing::warn!(
            "o1 batch kernels need 15 storage buffers and 11 KB of workgroup memory: per-row loop"
        );
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-o1-batch"),
        source: wgpu::ShaderSource::Wgsl(O1_BATCH_WGSL.into()),
    });
    let pipe = |ep: &str| {
        let p = c
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: None,
                module: &module,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: c.pipeline_cache.as_ref(),
            });
        let l = p.get_bind_group_layout(0);
        (p, l)
    };
    let pipes = Pipes {
        far_e: pipe("o1b_far_e"),
        attend: pipe("o1b_attend"),
        far_commit: pipe("o1b_far_commit"),
        ring_commit: pipe("o1b_ring_commit"),
        meta: pipe("o1b_meta"),
    };
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!(
            "cmf-o1-batch module rejected ({e}): sealed O(1) layers keep the per-row loop"
        );
        return None;
    }
    Some(pipes)
}

/// The batch kernels, when the module came up.
pub(crate) fn pipes(c: &Ctx) -> Option<&Pipes> {
    c.o1_batch_pipes.get_or_init(|| build(c)).as_ref()
}

/// One layer's sealed device state (the `O1Dev` mirror's buffers).
pub(crate) struct State<'a> {
    pub(crate) meta: &'a wgpu::Buffer,
    pub(crate) ring_k: &'a wgpu::Buffer,
    pub(crate) ring_v: &'a wgpu::Buffer,
    pub(crate) sink_k: &'a wgpu::Buffer,
    pub(crate) sink_v: &'a wgpu::Buffer,
    pub(crate) k_tilde: &'a wgpu::Buffer,
    pub(crate) qt: &'a wgpu::Buffer,
    pub(crate) mu: &'a wgpu::Buffer,
    pub(crate) mz: &'a wgpu::Buffer,
    pub(crate) that: &'a wgpu::Buffer,
}

/// Geometry of the layer: `g` KV groups × `hpg` query heads, `m`
/// landmarks, ring stride `w` (the widest group's window), `ns` sinks.
#[derive(Clone, Copy)]
pub(crate) struct Geom {
    pub(crate) g: usize,
    pub(crate) hpg: usize,
    pub(crate) m: usize,
    pub(crate) w: usize,
    pub(crate) ns: usize,
    pub(crate) rect_fm: bool,
    pub(crate) d: usize,
    pub(crate) dv: usize,
    pub(crate) scale: f32,
}

impl Geom {
    /// Floats of the per-call scratch (`ob_s`).
    pub(crate) fn scratch_len(&self) -> usize {
        let ghm = self.g * self.hpg * self.m;
        2 * ghm * OB_K + 2 * ghm
    }
}

/// Smallest group window of a sealed layer: a sub-chunk evicts at most
/// this many rows, all of them still in the ring at its start.
pub(crate) fn min_window(views: &[crate::nystrom::O1DeviceView<'_>]) -> usize {
    views.iter().map(|v| v.w).min().unwrap_or(0)
}

/// Encode `rows` consecutive prompt rows of one sealed layer: attention
/// rows into `out` ([rows][g·hpg][dv]) and the state advanced by `rows`
/// pushes. `q` is the rotated Q ([rows][g·hpg][d]), `kb`/`vb` the rotated
/// K and V rows ([rows][g][d|dv]); `scratch` holds `geom.scratch_len()`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    p: &Pipes,
    st: &State<'_>,
    geom: Geom,
    q: &wgpu::Buffer,
    kb: &wgpu::Buffer,
    vb: &wgpu::Buffer,
    scratch: &wgpu::Buffer,
    out: &wgpu::Buffer,
    rows: usize,
) {
    let bind = |l: &wgpu::BindGroupLayout, e: &[(u32, &wgpu::Buffer)]| {
        let entries: Vec<_> = e.iter().map(|(i, b)| bind_buf(*i, b)).collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: l,
            entries: &entries,
        })
    };
    let ghm = geom.g * geom.hpg * geom.m;
    let mut pass = begin_pass(enc);
    let mut r0 = 0;
    while r0 < rows {
        let kk = OB_K.min(rows - r0);
        let u = uniform_rope(
            c,
            [
                geom.hpg as u32,
                geom.m as u32,
                geom.w as u32,
                (geom.ns as u32) | (u32::from(geom.rect_fm) << 8),
                geom.d as u32,
                geom.dv as u32,
                geom.scale.to_bits(),
                kk as u32,
                (geom.g * geom.hpg) as u32,
                geom.g as u32,
                r0 as u32,
                0,
            ],
        );
        let bg_far = bind(
            &p.far_e.1,
            &[
                (0, st.meta),
                (1, st.ring_k),
                (6, st.qt),
                (8, st.mz),
                (13, scratch),
                (15, &u),
            ],
        );
        let bg_att = bind(
            &p.attend.1,
            &[
                (0, st.meta),
                (1, st.ring_k),
                (2, st.ring_v),
                (3, st.sink_k),
                (4, st.sink_v),
                (5, st.k_tilde),
                (7, st.mu),
                (9, st.that),
                (10, q),
                (11, kb),
                (12, vb),
                (13, scratch),
                (14, out),
                (15, &u),
            ],
        );
        let bg_fc = bind(
            &p.far_commit.1,
            &[
                (0, st.meta),
                (2, st.ring_v),
                (8, st.mz),
                (9, st.that),
                (13, scratch),
                (15, &u),
            ],
        );
        let bg_rc = bind(
            &p.ring_commit.1,
            &[
                (0, st.meta),
                (1, st.ring_k),
                (2, st.ring_v),
                (11, kb),
                (12, vb),
                (15, &u),
            ],
        );
        let bg_meta = bind(&p.meta.1, &[(0, st.meta), (15, &u)]);
        pass.set_pipeline(&p.far_e.0);
        pass.set_bind_group(0, &bg_far, &[]);
        pass.dispatch_workgroups(ghm as u32, 1, 1);
        pass.set_pipeline(&p.attend.0);
        pass.set_bind_group(0, &bg_att, &[]);
        pass.dispatch_workgroups((geom.g * geom.hpg) as u32, kk as u32, 1);
        pass.set_pipeline(&p.far_commit.0);
        pass.set_bind_group(0, &bg_fc, &[]);
        pass.dispatch_workgroups(ghm as u32, 1, 1);
        pass.set_pipeline(&p.ring_commit.0);
        pass.set_bind_group(0, &bg_rc, &[]);
        pass.dispatch_workgroups(geom.g as u32, kk as u32, 1);
        pass.set_pipeline(&p.meta.0);
        pass.set_bind_group(0, &bg_meta, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        r0 += kk;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The batch kernels against the REAL NystromState stepped one row at
    /// a time: two groups (the second with its own window), a chunk longer
    /// than one sub-chunk, then a second chunk from the committed state —
    /// ring, meta and far skeleton all have to be what the serial steps
    /// leave behind.
    #[test]
    fn o1_batch_matches_serial_steps() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping o1 batch test");
            return;
        };
        let _op = c.op_gate.enter();
        let Some(p) = pipes(c) else {
            eprintln!("o1 batch kernels unavailable — skipping");
            return;
        };
        // (d, m, w, sink, hpg, prompt t, second group's window, chunks)
        for (d, m, w, sink, hpg, t, w2, chunks) in [
            // Both rings full at the start, 70 rows = two sub-chunks.
            (
                8usize,
                4usize,
                64usize,
                2usize,
                2usize,
                100usize,
                64usize,
                [70usize, 9],
            ),
            // Wide group fills during the chunk: its first eviction meets
            // an empty far field (max -inf).
            (8, 4, 64, 2, 2, 100, 128, [70, 40]),
            // Production head geometry (Qwen3.8: d 256, 6 heads a group, 32
            // landmarks) with a wide group whose prompt still fits its ring.
            (256, 32, 128, 4, 6, 300, 512, [64, 64]),
        ] {
            let jit = |a: usize, b: usize| ((a * 37 + b * 13 + 3) % 83) as f32 / 83.0 - 0.5;
            let mut states = Vec::new();
            for (gi, wg) in [(0usize, w), (1, w2)] {
                let ks: Vec<f32> = (0..t * d).map(|i| jit(i, 1 + 20 * gi)).collect();
                let vs: Vec<f32> = (0..t * d).map(|i| jit(i, 2 + 20 * gi)).collect();
                let qs: Vec<Vec<f32>> = (0..hpg)
                    .map(|h| (0..t * d).map(|i| jit(i, 3 + h + 20 * gi)).collect())
                    .collect();
                let refs: Vec<&[f32]> = qs.iter().map(|v| v.as_slice()).collect();
                let mut st =
                    crate::nystrom::NystromState::new_group(m, w, sink, hpg).with_window(wg);
                st.prefill_group(&refs, &ks, &vs, t, d, d);
                states.push(st);
            }
            let (ng, nh) = (2usize, 2 * hpg);
            let views: Vec<_> = states.iter().map(|s| s.device_view()).collect();
            assert!(!views[0].exact_only, "t must exceed w+8 for this test");
            let key = (u64::MAX - 1, usize::MAX - 1);
            o1_ensure(c, key.0, key.1, &views, 1).expect("o1 upload");
            let mv = views[0].m_eff;
            let geom = Geom {
                g: ng,
                hpg,
                m: mv,
                w: w.max(w2),
                ns: sink,
                rect_fm: views[0].heads[0].rect_fm,
                d,
                dv: d,
                scale: views[0].scale,
            };
            drop(views);
            let stor = |data: &[f32]| {
                use wgpu::util::DeviceExt;
                c.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    })
            };
            let mut base = 0usize;
            for (ci, &rows) in chunks.iter().enumerate() {
                let q: Vec<f32> = (0..rows * nh * d)
                    .map(|i| jit(i + base * nh * d, 5))
                    .collect();
                let k: Vec<f32> = (0..rows * ng * d)
                    .map(|i| jit(i + base * ng * d, 6))
                    .collect();
                let v: Vec<f32> = (0..rows * ng * d)
                    .map(|i| jit(i + base * ng * d, 7))
                    .collect();
                base += rows;
                let mut want = vec![0f32; rows * nh * d];
                for r in 0..rows {
                    for (g, st) in states.iter_mut().enumerate() {
                        let qo = (r * nh + g * hpg) * d;
                        let ko = (r * ng + g) * d;
                        st.step_group(
                            &q[qo..qo + hpg * d],
                            &k[ko..ko + d],
                            &v[ko..ko + d],
                            &mut want[qo..qo + hpg * d],
                        );
                    }
                }
                let (qb, kb, vb) = (stor(&q), stor(&k), stor(&v));
                let sb = stor(&vec![0f32; geom.scratch_len()]);
                let ob = stor(&vec![0f32; rows * nh * d]);
                let staging = c.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (rows * nh * d * 4) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut enc = c
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let map = c.o1m.lock().unwrap();
                    let dv = map.get(&key).unwrap();
                    let st = State {
                        meta: &dv.meta,
                        ring_k: &dv.ring_k,
                        ring_v: &dv.ring_v,
                        sink_k: &dv.sink_k,
                        sink_v: &dv.sink_v,
                        k_tilde: &dv.k_tilde,
                        qt: &dv.qt,
                        mu: &dv.mu,
                        mz: &dv.mz,
                        that: &dv.that,
                    };
                    encode(c, &mut enc, p, &st, geom, &qb, &kb, &vb, &sb, &ob, rows);
                }
                let mut got = vec![0f32; rows * nh * d];
                assert!(readback(
                    c,
                    enc,
                    &ob,
                    &staging,
                    (rows * nh * d * 4) as u64,
                    &mut got
                ));
                assert!(got.iter().all(|x| x.is_finite()), "non-finite batch output");
                let md = want
                    .iter()
                    .zip(&got)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "o1 batch d={d} m={m} w={w}/{w2} hpg={hpg} chunk {ci} ({rows} rows): max|Δ| {md:.2e}"
                );
                assert!(
                    md < 1e-3,
                    "o1 batch ≠ serial steps (d={d} w={w}/{w2} chunk {ci}): max|Δ| = {md}"
                );
            }
            c.o1m.lock().unwrap().remove(&key);
        }
    }
}
