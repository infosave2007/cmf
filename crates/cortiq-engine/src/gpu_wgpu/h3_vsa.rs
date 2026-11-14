//! Video sparse attention (VSA) in the wgpu MiniMax-H3 chain: FastVideo's
//! VSA-H3 as its FastH3 students were trained with it (see `mmh3::VsaPlan`).
//!
//! The panel keeps its rows in sequence order; the plan's tiles address them
//! through `rowof` (slot → row, `NONE` for a tile's pad slots). Per block:
//!
//! - `vsa_qpad`: the queries (normed and rotated by `h3_qkrope`) copied into
//!   tile-padded order, the operand the matrix units load 16 rows of at once;
//! - `vsa_pool`: per head the tile means of q, k (and v) in f32;
//! - `vsa_scores`: tile against tile, `q̄ · k̄ · scale`;
//! - `vsa_select`: a video query tile's key tiles — every non-video tile,
//!   then its top-k video tiles (a radix select, the list in ascending order);
//! - `vsa_flash`: `qi_flash` over a query tile's key tiles, the keys and
//!   values read from the panel rows the tiles name (a non-video query tile
//!   walks every tile), the output into the rows' own places;
//! - `vsa_compress` and `vsa_gate_add`: the gated compression branch,
//!   `gate · softmax(tile scores) · v̄` added to every row of the tile.
//!
//! The values carry the panel's guard (×2^-qkv) through the means, and the
//! gate is a GEMM at the true scale, so the branch lands at the output's.

use std::sync::Arc;

use super::super::Ctx;
use super::super::zimage::{self as zi, Call, FlashCfg};
use crate::mmh3::{VSA_TILE, VsaPlan};

pub(super) const NONE: u32 = u32::MAX;

/// The queries into tile-padded order (pad slots zero): a workgroup a slot.
const QPAD_SRC: &str = r#"
struct QP { np: u32, ld: u32, inner: u32, _a: u32 };
@group(0) @binding(0) var<storage, read> panel: array<u32>;
@group(0) @binding(1) var<storage, read> rowof: array<u32>;
@group(0) @binding(2) var<storage, read_write> qpad: array<u32>;
@group(0) @binding(3) var<uniform> qp: QP;

@compute @workgroup_size(256)
fn vsa_qpad(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let s = wid.y * 65535u + wid.x;
    if (s >= qp.np) { return; }
    let row = rowof[s];
    for (var i = lid; i < qp.inner; i = i + 256u) {
        var v = 0u;
        if (row != 0xFFFFFFFFu) { v = panel[row * qp.ld + i]; }
        qpad[s * qp.inner + i] = v;
    }
}
"#;

/// Tile means of q, k (and v): a workgroup per (tile, head), a lane a dim.
const POOL_SRC: &str = r#"
enable f16;
struct PP { t: u32, ld: u32, inner: u32, wantv: u32 };
@group(0) @binding(0) var<storage, read> panel: array<f16>;
@group(0) @binding(1) var<storage, read> rowof: array<u32>;
@group(0) @binding(2) var<storage, read> size: array<u32>;
@group(0) @binding(3) var<storage, read_write> pq: array<f32>;
@group(0) @binding(4) var<storage, read_write> pk: array<f32>;
@group(0) @binding(5) var<storage, read_write> pv: array<f32>;
@group(0) @binding(6) var<uniform> pp: PP;

@compute @workgroup_size(128)
fn vsa_pool(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) d: u32) {
    let t = wid.x;
    let h = wid.y;
    let sz = size[t];
    var aq = 0.0;
    var ak = 0.0;
    var av = 0.0;
    for (var i = 0u; i < sz; i = i + 1u) {
        let b = rowof[t * 64u + i] * pp.ld + h * 128u + d;
        aq = aq + f32(panel[b]);
        ak = ak + f32(panel[b + pp.inner]);
        if (pp.wantv != 0u) { av = av + f32(panel[b + 2u * pp.inner]); }
    }
    let inv = 1.0 / f32(sz);
    let o = (h * pp.t + t) * 128u + d;
    pq[o] = aq * inv;
    pk[o] = ak * inv;
    if (pp.wantv != 0u) { pv[o] = av * inv; }
}
"#;

/// Tile scores, 16×16 a workgroup, the dims through workgroup memory.
const SCORES_SRC: &str = r#"
struct SP { t: u32, scale: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> pq: array<f32>;
@group(0) @binding(1) var<storage, read> pk: array<f32>;
@group(0) @binding(2) var<storage, read_write> s: array<f32>;
@group(0) @binding(3) var<uniform> sp: SP;
var<workgroup> a: array<f32, 2064>;
var<workgroup> b: array<f32, 2064>;

@compute @workgroup_size(256)
fn vsa_scores(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let i0 = wid.x * 16u;
    let j0 = wid.y * 16u;
    let h = wid.z;
    for (var c = tid; c < 2048u; c = c + 256u) {
        let r = c / 128u;
        let d = c % 128u;
        var x = 0.0;
        var y = 0.0;
        if (i0 + r < sp.t) { x = pq[(h * sp.t + i0 + r) * 128u + d]; }
        if (j0 + r < sp.t) { y = pk[(h * sp.t + j0 + r) * 128u + d]; }
        a[r * 129u + d] = x;
        b[r * 129u + d] = y;
    }
    workgroupBarrier();
    let tx = tid % 16u;
    let ty = tid / 16u;
    var acc = 0.0;
    for (var d = 0u; d < 128u; d = d + 1u) { acc = acc + a[ty * 129u + d] * b[tx * 129u + d]; }
    let i = i0 + ty;
    let j = j0 + tx;
    if (i < sp.t && j < sp.t) { s[(h * sp.t + i) * sp.t + j] = acc * sp.scale; }
}
"#;

/// A video query tile's key tiles: the non-video ones, then the top-k video
/// tiles by score (radix select of the k-th largest, then a block scan that
/// writes them in ascending order — the same list every run).
const SELECT_SRC: &str = r#"
struct LP { t: u32, p: u32, k: u32, _a: u32 };
@group(0) @binding(0) var<storage, read> s: array<f32>;
@group(0) @binding(1) var<storage, read_write> list: array<u32>;
@group(0) @binding(2) var<uniform> lp: LP;
var<workgroup> hist: array<atomic<u32>, 256>;
var<workgroup> scan: array<u32, 256>;
var<workgroup> sh: array<u32, 2>;

fn okey(f: f32) -> u32 {
    let u = bitcast<u32>(f);
    return select(u | 0x80000000u, ~u, (u & 0x80000000u) != 0u);
}

@compute @workgroup_size(256)
fn vsa_select(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let i = lp.p + wid.x;
    let h = wid.y;
    let rb = (h * lp.t + i) * lp.t + lp.p;
    let nv = lp.t - lp.p;
    let ob = (h * lp.t + i) * (lp.p + lp.k);
    for (var c = tid; c < lp.p; c = c + 256u) { list[ob + c] = c; }
    var prefix = 0u;
    var mask = 0u;
    var left = lp.k;
    for (var ps = 0u; ps < 4u; ps = ps + 1u) {
        let shift = 24u - 8u * ps;
        atomicStore(&hist[tid], 0u);
        workgroupBarrier();
        for (var c = tid; c < nv; c = c + 256u) {
            let key = okey(s[rb + c]);
            if ((key & mask) == prefix) { atomicAdd(&hist[(key >> shift) & 255u], 1u); }
        }
        workgroupBarrier();
        if (tid == 0u) {
            var cum = 0u;
            var digit = 0u;
            for (var bb = 255i; bb >= 0i; bb = bb - 1i) {
                let hv = atomicLoad(&hist[u32(bb)]);
                if (cum + hv >= left) {
                    digit = u32(bb);
                    break;
                }
                cum = cum + hv;
            }
            sh[0] = prefix | (digit << shift);
            sh[1] = left - cum;
        }
        prefix = workgroupUniformLoad(&sh[0]);
        left = workgroupUniformLoad(&sh[1]);
        mask = mask | (255u << shift);
    }
    let gt_total = lp.k - left;
    let per = (nv + 255u) / 256u;
    let c0 = tid * per;
    let c1 = min(nv, c0 + per);
    for (var which = 0u; which < 2u; which = which + 1u) {
        var cnt = 0u;
        for (var c = c0; c < c1; c = c + 1u) {
            let key = okey(s[rb + c]);
            if (which == 0u) { cnt = cnt + select(0u, 1u, key > prefix); } else { cnt = cnt + select(0u, 1u, key == prefix); }
        }
        scan[tid] = cnt;
        workgroupBarrier();
        for (var o = 1u; o < 256u; o = o << 1u) {
            var v = 0u;
            if (tid >= o) { v = scan[tid - o]; }
            workgroupBarrier();
            scan[tid] = scan[tid] + v;
            workgroupBarrier();
        }
        var pos = scan[tid] - cnt;
        for (var c = c0; c < c1; c = c + 1u) {
            let key = okey(s[rb + c]);
            if (which == 0u && key > prefix) {
                list[ob + lp.p + pos] = lp.p + c;
                pos = pos + 1u;
            } else if (which == 1u && key == prefix) {
                if (pos < left) { list[ob + lp.p + gt_total + pos] = lp.p + c; }
                pos = pos + 1u;
            }
        }
        workgroupBarrier();
    }
}
"#;

/// The compression branch's tile output: softmax over a tile-score row times
/// the tile means of v. A workgroup per (tile, head), a lane a dim.
const COMPRESS_SRC: &str = r#"
struct CP { t: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> s: array<f32>;
@group(0) @binding(1) var<storage, read> pv: array<f32>;
@group(0) @binding(2) var<storage, read_write> oc: array<f32>;
@group(0) @binding(3) var<uniform> cp: CP;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn vsa_compress(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) d: u32) {
    let i = wid.x;
    let h = wid.y;
    let rb = (h * cp.t + i) * cp.t;
    var m = -3.0e38;
    for (var j = d; j < cp.t; j = j + 128u) { m = max(m, s[rb + j]); }
    red[d] = m;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st = st >> 1u) {
        if (d < st) { red[d] = max(red[d], red[d + st]); }
        workgroupBarrier();
    }
    let mx = red[0];
    workgroupBarrier();
    var z = 0.0;
    for (var j = d; j < cp.t; j = j + 128u) { z = z + exp(s[rb + j] - mx); }
    red[d] = z;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st = st >> 1u) {
        if (d < st) { red[d] = red[d] + red[d + st]; }
        workgroupBarrier();
    }
    let inv = 1.0 / red[0];
    var acc = 0.0;
    for (var j = 0u; j < cp.t; j = j + 1u) { acc = acc + exp(s[rb + j] - mx) * pv[(h * cp.t + j) * 128u + d]; }
    oc[(h * cp.t + i) * 128u + d] = acc * inv;
}
"#;

/// `att[r] += gate[r] · oc[tile of r]`, an element a lane.
const GATE_ADD_SRC: &str = r#"
enable f16;
struct GP { n: u32, inner: u32, t: u32, _a: u32 };
@group(0) @binding(0) var<storage, read_write> att: array<f16>;
@group(0) @binding(1) var<storage, read> gate: array<f16>;
@group(0) @binding(2) var<storage, read> oc: array<f32>;
@group(0) @binding(3) var<storage, read> slot: array<u32>;
@group(0) @binding(4) var<uniform> gp: GP;

@compute @workgroup_size(256)
fn vsa_gate_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.y * (65535u * 256u) + gid.x;
    if (e >= gp.n * gp.inner) { return; }
    let r = e / gp.inner;
    let c = e - r * gp.inner;
    let h = c / 128u;
    let d = c - h * 128u;
    let t = slot[r] / 64u;
    att[e] = f16(f32(att[e]) + f32(gate[e]) * oc[(h * gp.t + t) * 128u + d]);
}
"#;

/// The flash tile of `vsa_flash`: four subgroups of 16 queries (one 64-slot
/// query tile a workgroup) and `bc` keys a block, `bc` dividing 64.
pub(super) fn flash_cfg(c: &Ctx) -> Option<FlashCfg> {
    let lim = c.device.limits().max_compute_workgroup_storage_size;
    [32u32, 16].into_iter().map(|bc| FlashCfg { nw: 4, bc }).find(|&f| super::super::qi21::flash_shared(f) <= lim && flash_src(f).is_some())
}

/// WGSL of `vsa_flash` (hd 128): `qi_flash`'s schedule — Q·Kᵀ and P·V on the
/// matrix units in f16, the online softmax in f32 with the lazy,
/// workgroup-uniform rescale — over a query tile's key tiles. Queries from
/// `qpad` (tile-padded), keys and values from the panel rows `rowof` names,
/// a key past its tile's size masked, the output into the query's own row.
pub(super) fn flash_src(f: FlashCfg) -> Option<String> {
    use std::fmt::Write;
    let (nw, bc) = (f.nw, f.bc);
    if nw != 4 || bc == 0 || bc % 16 != 0 || 64 % bc != 0 {
        return None;
    }
    let nt = nw * 32;
    let ldk = 136u32;
    let ldp = bc + 8;
    let nkf = bc / 16;
    let vk = bc * 32;
    if vk % nt != 0 || 8 % nkf != 0 || (bc / 2) % 8 != 0 {
        return None;
    }
    let per = vk / nt;
    let half = bc / 2;
    let rounds = 8 / nkf;
    let cpt = 64 / bc;
    let thr = zi::flash_thr();
    let mut s = String::new();
    let _ = writeln!(s, "enable f16;\nenable wgpu_cooperative_matrix;\ndiagnostic(off, derivative_uniformity);");
    let _ = writeln!(
        s,
        "struct VP {{ ld: u32, k_col: u32, v_col: u32, o_ld: u32, scl: f32, oscale: f32, t: u32, p: u32, k: u32, inner: u32, _a: u32, _b: u32 }};"
    );
    let _ = writeln!(s, "@group(0) @binding(0) var<storage, read> qh: array<f16>;");
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> qv: array<vec4<f16>>;");
    let _ = writeln!(s, "@group(0) @binding(2) var<storage, read_write> oh: array<vec4<u32>>;");
    let _ = writeln!(s, "@group(0) @binding(3) var<storage, read> rowof: array<u32>;");
    let _ = writeln!(s, "@group(0) @binding(4) var<storage, read> size: array<u32>;");
    let _ = writeln!(s, "@group(0) @binding(5) var<storage, read> list: array<u32>;");
    let _ = writeln!(s, "@group(0) @binding(6) var<uniform> p: VP;");
    let _ = writeln!(s, "var<workgroup> sk: array<f16, {}>;", bc * ldk);
    let _ = writeln!(s, "var<workgroup> sv: array<f16, {}>;", bc * ldk);
    let _ = writeln!(s, "var<workgroup> ss: array<f32, {}>;", nw * 16 * bc);
    let _ = writeln!(s, "var<workgroup> sp: array<f16, {}>;", nw * 16 * ldp);
    let _ = writeln!(s, "var<workgroup> smx: array<f32, {nt}>;");
    let _ = writeln!(s, "var<workgroup> sfl: array<u32, 4>;");
    let _ = writeln!(
        s,
        "@compute @workgroup_size({nt})\nfn vsa_flash(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32, @builtin(subgroup_id) sg: u32) {{"
    );
    let _ = writeln!(s, "  let h = wid.y;");
    let _ = writeln!(s, "  let qt = wid.x;");
    let _ = writeln!(s, "  let q0 = qt * 64u + sg * 16u;");
    let _ = writeln!(s, "  let lane = tid - sg * 32u;");
    let _ = writeln!(s, "  let r = lane & 15u; let hf = lane >> 4u;");
    let _ = writeln!(s, "  let ldq = p.ld / 4u;");
    let _ = writeln!(s, "  let kq4 = (p.k_col + h * 128u) / 4u; let vq4 = (p.v_col + h * 128u) / 4u;");
    for d in 0..8 {
        let _ = writeln!(s, "  let qa{d} = coopLoadT<coop_mat16x16<f16, A>>(&qh[q0 * p.inner + h * 128u + {}u], p.inner);", d * 16);
    }
    for j in 0..8 {
        let _ = writeln!(s, "  var o{j}: coop_mat16x16<f32, C>;");
    }
    let _ = writeln!(s, "  var zc: coop_mat16x16<f32, C>;");
    let _ = writeln!(s, "  var mu = -1.0e30; var ls = 0.0;");
    let _ = writeln!(s, "  let qs = q0 + r;");
    let _ = writeln!(s, "  let live = qs < qt * 64u + size[qt];");
    // the key tiles: every tile for a non-video query tile, else its list
    let _ = writeln!(s, "  let dense = qt < p.p;");
    let _ = writeln!(s, "  let lb = (h * p.t + qt) * (p.p + p.k);");
    let _ = writeln!(s, "  let nkb = select(p.p + p.k, p.t, dense) * {cpt}u;");
    let _ = writeln!(s, "  if (tid < 2u) {{ sfl[tid] = 0u; }}");
    let load = |s: &mut String, kbx: &str, ind: &str| {
        let _ = writeln!(s, "{ind}{{ let kbl = {kbx}; let ktl = select(list[lb + kbl / {cpt}u], kbl / {cpt}u, dense);");
        for t in 0..per {
            let _ = writeln!(
                s,
                "{ind}  {{ let idx = tid + {o}u; let row = rowof[ktl * 64u + (kbl % {cpt}u) * {bc}u + idx / 32u]; let c4 = idx % 32u;",
                o = t * nt
            );
            let _ = writeln!(
                s,
                "{ind}    if (row != 0xFFFFFFFFu) {{ rk{t} = qv[row * ldq + kq4 + c4]; rv{t} = qv[row * ldq + vq4 + c4]; }} else {{ rk{t} = vec4<f16>(0.0); rv{t} = vec4<f16>(0.0); }} }}"
            );
        }
        let _ = writeln!(s, "{ind}}}");
    };
    for t in 0..per {
        let _ = writeln!(s, "  var rk{t}: vec4<f16>; var rv{t}: vec4<f16>;");
    }
    load(&mut s, "0u", "  ");
    let _ = writeln!(s, "  for (var kb = 0u; kb < nkb; kb = kb + 1u) {{");
    for t in 0..per {
        let _ = writeln!(
            s,
            "    {{ let idx = tid + {o}u; let d = (idx / 32u) * {ldk}u + (idx % 32u) * 4u; sk[d] = rk{t}.x; sk[d + 1u] = rk{t}.y; sk[d + 2u] = rk{t}.z; sk[d + 3u] = rk{t}.w; sv[d] = rv{t}.x; sv[d + 1u] = rv{t}.y; sv[d + 2u] = rv{t}.z; sv[d + 3u] = rv{t}.w; }}",
            o = t * nt
        );
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "    let kt = select(list[lb + kb / {cpt}u], kb / {cpt}u, dense);");
    let _ = writeln!(s, "    let klim = size[kt];");
    let _ = writeln!(s, "    if (kb + 1u < nkb) {{");
    load(&mut s, "(kb + 1u)", "      ");
    let _ = writeln!(s, "    }}");
    for j in 0..nkf {
        let _ = writeln!(s, "    let kf{j}_0 = coopLoad<coop_mat16x16<f16, B>>(&sk[{}u], {ldk}u);", j * 16 * ldk);
        let _ = writeln!(s, "    var s{j} = coopMultiplyAdd(qa0, kf{j}_0, zc);");
        for d in 1..8 {
            let _ = writeln!(s, "    let kf{j}_{d} = coopLoad<coop_mat16x16<f16, B>>(&sk[{}u], {ldk}u);", j * 16 * ldk + d * 16);
            let _ = writeln!(s, "    s{j} = coopMultiplyAdd(qa{d}, kf{j}_{d}, s{j});");
        }
        let _ = writeln!(s, "    coopStoreT(s{j}, &ss[sg * {}u + {}u], {bc}u);", 16 * bc, j * 16);
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "    let sbase = sg * {}u + r * {bc}u + hf * {half}u;", 16 * bc);
    let _ = writeln!(s, "    let kbase = (kb % {cpt}u) * {bc}u + hf * {half}u;");
    for e in 0..half {
        let _ = writeln!(s, "    let x{e} = select(-1.0e30, ss[sbase + {e}u] * p.scl, kbase + {e}u < klim);");
    }
    let _ = writeln!(s, "    var pm = x0;");
    for e in 1..half {
        let _ = writeln!(s, "    pm = max(pm, x{e});");
    }
    let _ = writeln!(s, "    smx[tid] = pm;");
    let _ = writeln!(s, "    if (pm > mu + {thr} && live) {{ sfl[kb & 1u] = 1u; }}");
    let _ = writeln!(s, "    let need = workgroupUniformLoad(&sfl[kb & 1u]);");
    let _ = writeln!(s, "    if (tid == 0u) {{ sfl[(kb + 1u) & 1u] = 0u; }}");
    let _ = writeln!(s, "    let rmax = max(pm, smx[tid ^ 16u]);");
    let _ = writeln!(s, "    if (need != 0u) {{");
    let _ = writeln!(s, "      let mn = max(mu, rmax);");
    let _ = writeln!(s, "      let alpha = exp2(mu - mn);");
    let _ = writeln!(s, "      ls = ls * alpha; mu = mn;");
    let _ = writeln!(s, "      if (kb > 0u) {{");
    for rd in 0..rounds {
        for q in 0..nkf {
            let _ = writeln!(s, "        coopStoreT(o{}, &ss[sg * {}u + {}u], {bc}u);", rd * nkf + q, 16 * bc, q * 16);
        }
        let _ = writeln!(s, "        workgroupBarrier();");
        for e in 0..half {
            let _ = writeln!(s, "        ss[sbase + {e}u] = ss[sbase + {e}u] * alpha;");
        }
        let _ = writeln!(s, "        workgroupBarrier();");
        for q in 0..nkf {
            let _ = writeln!(
                s,
                "        o{} = coopLoadT<coop_mat16x16<f32, C>>(&ss[sg * {}u + {}u], {bc}u);",
                rd * nkf + q,
                16 * bc,
                q * 16
            );
        }
        let _ = writeln!(s, "        workgroupBarrier();");
    }
    let _ = writeln!(s, "      }}");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    let pbase = sg * {}u + r * {ldp}u + hf * {half}u;", 16 * ldp);
    for e in 0..half {
        let _ = writeln!(s, "    let pv{e} = select(0.0, exp2(x{e} - mu), kbase + {e}u < klim);");
    }
    for e in 0..half {
        let _ = writeln!(s, "    ls = ls + pv{e}; sp[pbase + {e}u] = f16(pv{e});");
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    for kk in 0..nkf {
        let _ = writeln!(s, "    {{ let pa = coopLoadT<coop_mat16x16<f16, A>>(&sp[sg * {}u + {}u], {ldp}u);", 16 * ldp, kk * 16);
        for j in 0..8 {
            let _ = writeln!(s, "      let vf{j} = coopLoadT<coop_mat16x16<f16, B>>(&sv[{}u], {ldk}u);", kk * 16 * ldk + j * 16);
            let _ = writeln!(s, "      o{j} = coopMultiplyAdd(pa, vf{j}, o{j});");
        }
        let _ = writeln!(s, "    }}");
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "  }}");
    let _ = writeln!(s, "  smx[tid] = ls;");
    let _ = writeln!(s, "  workgroupBarrier();");
    let _ = writeln!(s, "  let lt = ls + smx[tid ^ 16u];");
    let _ = writeln!(s, "  let inv = select(0.0, p.oscale / lt, lt > 0.0);");
    let _ = writeln!(s, "  let orow = rowof[min(qs, arrayLength(&rowof) - 1u)];");
    let _ = writeln!(s, "  let sbase = sg * {}u + r * {bc}u + hf * {half}u;", 16 * bc);
    for rd in 0..rounds {
        for q in 0..nkf {
            let _ = writeln!(s, "  coopStoreT(o{}, &ss[sg * {}u + {}u], {bc}u);", rd * nkf + q, 16 * bc, q * 16);
        }
        let _ = writeln!(s, "  workgroupBarrier();");
        let _ = writeln!(s, "  if (live) {{");
        for c8 in 0..half / 8 {
            let _ = writeln!(
                s,
                "    {{ let b = sbase + {}u; oh[(orow * p.o_ld + h * 128u + {}u + hf * {half}u + {}u) / 8u] = vec4<u32>(pack2x16float(vec2<f32>(ss[b], ss[b + 1u]) * inv), pack2x16float(vec2<f32>(ss[b + 2u], ss[b + 3u]) * inv), pack2x16float(vec2<f32>(ss[b + 4u], ss[b + 5u]) * inv), pack2x16float(vec2<f32>(ss[b + 6u], ss[b + 7u]) * inv)); }}",
                c8 * 8,
                rd * bc,
                c8 * 8
            );
        }
        let _ = writeln!(s, "  }}");
        let _ = writeln!(s, "  workgroupBarrier();");
    }
    let _ = writeln!(s, "}}");
    Some(s)
}

/// The plan and the working buffers of one chain call.
pub(super) struct VsaDev {
    pub(super) n: usize,
    pub(super) np: usize,
    pub(super) t: usize,
    pub(super) p: usize,
    pub(super) k: usize,
    rowof: wgpu::Buffer,
    slot: wgpu::Buffer,
    size: wgpu::Buffer,
    qpad: wgpu::Buffer,
    pq: wgpu::Buffer,
    pk: wgpu::Buffer,
    pv: wgpu::Buffer,
    s: wgpu::Buffer,
    oc: wgpu::Buffer,
    list: wgpu::Buffer,
    /// `[mp][inner]` f16: the gate GEMM's output (`mp` rows: the tiles'
    /// over-reads).
    pub(super) gate: wgpu::Buffer,
    pipes: [Arc<wgpu::ComputePipeline>; 7],
}

impl VsaDev {
    /// Bytes on the card for `plan` at `inner` wide heads (`nh` of them).
    pub(super) fn bytes(plan: &VsaPlan, nh: usize, inner: usize, mp: usize) -> u64 {
        let (t, np) = (plan.tiles, plan.tiles * VSA_TILE);
        let lw = plan.prefix_tiles + plan.topk;
        (np * inner * 2 + mp * inner * 2 + 4 * nh * t * 128 * 4 + nh * t * t * 4 + nh * t * lw * 4 + (np + plan.n + t) * 4) as u64
    }

    pub(super) fn new(c: &Ctx, plan: &VsaPlan, nh: usize, inner: usize, mp: usize) -> Option<VsaDev> {
        let fc = flash_cfg(c)?;
        let (t, np) = (plan.tiles, plan.tiles * VSA_TILE);
        let lw = plan.prefix_tiles + plan.topk;
        let mut rowof = vec![NONE; np];
        for (r, &s) in plan.slot.iter().enumerate() {
            rowof[s as usize] = r as u32;
        }
        let pipes = [
            zi::pipeline(c, "vsa_qpad", QPAD_SRC, "vsa_qpad")?,
            zi::pipeline(c, "vsa_pool", POOL_SRC, "vsa_pool")?,
            zi::pipeline(c, "vsa_scores", SCORES_SRC, "vsa_scores")?,
            zi::pipeline(c, "vsa_select", SELECT_SRC, "vsa_select")?,
            zi::pipeline(c, &format!("vsa_flash_{}_{}_{}", fc.nw, fc.bc, zi::flash_thr()), &flash_src(fc)?, "vsa_flash")?,
            zi::pipeline(c, "vsa_compress", COMPRESS_SRC, "vsa_compress")?,
            zi::pipeline(c, "vsa_gate_add", GATE_ADD_SRC, "vsa_gate_add")?,
        ];
        let b = |bytes: usize, label: &str| zi::sbuf(c, bytes as u64, label);
        Some(VsaDev {
            n: plan.n,
            np,
            t,
            p: plan.prefix_tiles,
            k: plan.topk,
            rowof: zi::sbuf_init(c, bytemuck::cast_slice(&rowof), "vsa_rowof"),
            slot: zi::sbuf_init(c, bytemuck::cast_slice(&plan.slot), "vsa_slot"),
            size: zi::sbuf_init(c, bytemuck::cast_slice(&plan.size), "vsa_size"),
            qpad: b(np * inner * 2, "vsa_qpad"),
            pq: b(nh * t * 128 * 4, "vsa_pq"),
            pk: b(nh * t * 128 * 4, "vsa_pk"),
            pv: b(nh * t * 128 * 4, "vsa_pv"),
            s: b(nh * t * t * 4, "vsa_s"),
            oc: b(nh * t * 128 * 4, "vsa_oc"),
            list: b(nh * t * lw.max(1) * 4, "vsa_list"),
            gate: b(mp * inner * 2, "vsa_gate"),
            pipes,
        })
    }

    /// One block's sparse attention over the normed panel into `att` (and
    /// the gated branch when `gated`): the calls in order.
    pub(super) fn calls(&self, c: &Ctx, panel: &wgpu::Buffer, att: &wgpu::Buffer, nh: usize, inner: usize, gated: bool) -> Vec<Call> {
        let (t, np, n) = (self.t as u32, self.np as u32, self.n as u32);
        let ld = 3 * inner as u32;
        let mut out = Vec::with_capacity(7);
        let call = |pipe: &Arc<wgpu::ComputePipeline>, bufs: &[&wgpu::Buffer], grid: (u32, u32, u32)| Call {
            pipe: pipe.clone(),
            bg: zi::bg(c, pipe, bufs),
            grid,
        };
        let u = zi::ubuf(c, &[np, ld / 2, inner as u32 / 2, 0]);
        out.push(call(&self.pipes[0], &[panel, &self.rowof, &self.qpad, &u], (np.min(65535), np.div_ceil(65535), 1)));
        let u = zi::ubuf(c, &[t, ld, inner as u32, gated as u32]);
        out.push(call(&self.pipes[1], &[panel, &self.rowof, &self.size, &self.pq, &self.pk, &self.pv, &u], (t, nh as u32, 1)));
        let scale = 1.0f32 / (128f32).sqrt();
        let u = zi::ubuf(c, &[t, scale.to_bits(), 0, 0]);
        out.push(call(&self.pipes[2], &[&self.pq, &self.pk, &self.s, &u], (t.div_ceil(16), t.div_ceil(16), nh as u32)));
        if self.t > self.p {
            let u = zi::ubuf(c, &[t, self.p as u32, self.k as u32, 0]);
            out.push(call(&self.pipes[3], &[&self.s, &self.list, &u], (t - self.p as u32, nh as u32, 1)));
        }
        let scl = std::f32::consts::LOG2_E / (128f32).sqrt();
        let u = zi::ubuf(
            c,
            &[ld, inner as u32, 2 * inner as u32, inner as u32, scl.to_bits(), 1.0f32.to_bits(), t, self.p as u32, self.k as u32, inner as u32, 0, 0],
        );
        out.push(call(&self.pipes[4], &[&self.qpad, panel, att, &self.rowof, &self.size, &self.list, &u], (t, nh as u32, 1)));
        if gated {
            let u = zi::ubuf(c, &[t, 0, 0, 0]);
            out.push(call(&self.pipes[5], &[&self.s, &self.pv, &self.oc, &u], (t, nh as u32, 1)));
            let tot = n * inner as u32;
            let wg = tot.div_ceil(256);
            let u = zi::ubuf(c, &[n, inner as u32, t, 0]);
            out.push(call(&self.pipes[6], &[att, &self.gate, &self.oc, &self.slot, &u], (wg.min(65535), wg.div_ceil(65535), 1)));
        }
        out
    }

}

#[cfg(test)]
mod tests {
    /// The VSA kernels are valid WGSL for naga.
    #[test]
    fn vsa_kernels_validate() {
        use wgpu::naga;
        let check = |name: &str, src: &str| {
            let m = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&m)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
        };
        check("vsa_qpad", super::QPAD_SRC);
        check("vsa_pool", super::POOL_SRC);
        check("vsa_scores", super::SCORES_SRC);
        check("vsa_select", super::SELECT_SRC);
        check("vsa_compress", super::COMPRESS_SRC);
        check("vsa_gate_add", super::GATE_ADD_SRC);
        for bc in [16, 32, 64] {
            let src = super::flash_src(super::super::super::zimage::FlashCfg { nw: 4, bc }).expect("flash source");
            check(&format!("vsa_flash bc {bc}"), &src);
        }
    }
}
