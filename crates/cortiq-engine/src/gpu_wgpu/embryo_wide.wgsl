// Resident Embryo token on packed weights: the matrices stay in the file's form on the
// device, f16 or q8_2f, read in 16-byte loads through the row kernels of
// `embryo_wide_f16.wgsl` / `embryo_wide_q8.wgsl`. Activation and residual ride in the
// matrix stage, so they need no dispatch of their own. A dispatch boundary costs more
// than most of a kernel, about 8 µs on an M4 and 5 µs on an RTX 2000 Ada, so stages fuse
// wherever a workgroup can hold the reduction. A store to one component of a vec4 cell
// may be a read-modify-write of the cell: a thread either owns the whole cell or stores
// through a plain f32 view. A GPU only fills up with many short threads: a thread walks
// at most `CHUNK` values of a row, its partial sum lands in a plane of `S_PARTS` and the
// norm stage that follows adds the planes up.
struct Params {
    position: u32,
    max_seq: u32,
    vocab: u32,
    flags: u32,
    eps: f32,
    phase_mass: f32,
    layer: u32,
    chunk: u32,
};

@group(0) @binding(0) var<storage, read> small4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> md: array<u32>;
@group(0) @binding(2) var<storage, read> head_mat: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read_write> state1: array<f32>;
@group(0) @binding(9) var<storage, read_write> l1: array<f32>;
@group(0) @binding(9) var<storage, read_write> l4: array<vec4<f32>>;
@group(0) @binding(10) var<uniform> p: Params;
@group(0) @binding(11) var<storage, read_write> s4: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> mats: array<vec4<u32>>;
// The scratch image again: a thread that owns one row stores a plain f32 here and
// reads its input through `s4`.
@group(0) @binding(13) var<storage, read_write> s1: array<f32>;

const HEADER: u32 = 32u;
const REC: u32 = 64u;
// Planes of EMBRYO_CORE_SRC, in f32 words.
const S_NORM: u32 = 0u;
const S_GDN_Z: u32 = 2048u;
const S_GDN_A: u32 = 5120u;
const S_GDN_B: u32 = 5632u;
const S_GDN_O: u32 = 6144u;
const S_G: u32 = 5120u;
const S_UP: u32 = 8192u;
const S_GDN_QKV: u32 = 9216u;
const S_GDN_CQ: u32 = 11264u;
const S_SEL: u32 = 13312u;
const S_PARTS: u32 = 45056u;
const PART_STRIDE: u32 = 1024u;
const CHUNK: u32 = 128u;
const S_ROPE: u32 = 53248u;
const CP_Q: u32 = 106496u;
const CP_ATT: u32 = 368640u;

fn rb() -> u32 { return HEADER + p.layer * REC; }
fn sm(i: u32) -> f32 { return small4[i >> 2u][i & 3u]; }
fn hsum(v: vec4<f32>) -> f32 { return v.x + v.y + v.z + v.w; }
fn silu(v: f32) -> f32 { return v / (1.0 + exp(-v)); }
fn chunks(cols: u32) -> u32 { return (cols + CHUNK - 1u) / CHUNK; }
// Stages with a reduction are the lanes of one workgroup: their shared memory is on
// the chip, and a barrier there costs less than a dispatch boundary.
@group(0) @binding(3) var<storage, read> clusters4: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> final4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> state4: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> hb4: array<vec4<f32>>;

var<workgroup> wred: array<f32, 256>;
var<workgroup> winv: f32;
// The normed token for the lanes that read it in the same workgroup: a barrier here
// orders workgroup memory, not a store to the scratch image.
var<workgroup> wx: array<vec4<f32>, 256>;

// RMSNorm into the scratch plane: a lane per four values.
fn norm_lanes(lid: u32, from_final: bool, wbase: u32, parts: u32) -> vec4<f32> {
    let hidden = md[0u];
    let n = hidden >> 2u;
    var v = vec4<f32>(0.0);
    if (lid < n) {
        v = hb4[lid];
        for (var c = 0u; c < parts; c += 1u) {
            v += s4[((S_PARTS + c * PART_STRIDE) >> 2u) + lid];
        }
        hb4[lid] = v;
        wred[lid] = dot(v, v);
    }
    workgroupBarrier();
    if (lid == 0u) {
        var t = 0.0;
        for (var j = 0u; j < n; j += 1u) { t += wred[j]; }
        winv = inverseSqrt(t / f32(hidden) + p.eps);
    }
    workgroupBarrier();
    if (lid < n) {
        var w = vec4<f32>(0.0);
        if (from_final) { w = final4[lid]; } else { w = small4[(wbase >> 2u) + lid]; }
        if ((p.flags & 1u) != 0u) { w = vec4<f32>(1.0) + w; }
        v *= winv * w;
        s4[lid] = v;
    }
    return v;
}

// Each matrix that reads the normed token gets its input from the norm stage itself:
// `fold_lane` is free on f16 and multiplies in the column field of a q8_2f matrix.
@compute @workgroup_size(256)
fn w_norm(@builtin(local_invocation_index) lid: u32) {
    let r = rb();
    // The first layer starts from the token embedding, the others from an FFN.
    let v = norm_lanes(lid, false, md[r + 1u], select(chunks(md[1u]), 0u, p.layer == 0u));
    if (lid >= (md[0u] >> 2u)) { return; }
    if (md[r] == 4u) {
        fold_lane(0u, md[r + 56u], lid, v);
        fold_lane(1u, md[r + 57u], lid, v);
        fold_lane(2u, md[r + 58u], lid, v);
        fold_lane(3u, md[r + 59u], lid, v);
    } else {
        fold_lane(0u, md[r + 11u], lid, v);
        fold_lane(1u, md[r + 12u], lid, v);
        fold_lane(2u, md[r + 13u], lid, v);
    }
}

@compute @workgroup_size(256)
fn w_post_norm(@builtin(local_invocation_index) lid: u32) {
    let r = rb();
    let cols = select(md[7u] * md[9u], md[24u] * md[27u], md[r] == 4u);
    let v = norm_lanes(lid, false, md[r + 2u], chunks(cols));
    if (lid >= (md[0u] >> 2u)) { return; }
    fold_lane(0u, md[r + 21u], lid, v);
    fold_lane(1u, md[r + 22u], lid, v);
}

// The final norm and the cluster scores, as log probabilities, for the gate of the head.
@compute @workgroup_size(256)
fn w_final_norm(@builtin(local_invocation_index) lid: u32) {
    let v = norm_lanes(lid, true, 0u, chunks(md[1u]));
    let n = md[0u] >> 2u;
    if (lid < n) {
        wx[lid] = v;
        fold_lane(0u, md[30u], lid, v);
    }
    workgroupBarrier();
    let nc = md[11u];
    if (lid < nc) {
        var z = vec4<f32>(0.0);
        for (var j = 0u; j < n; j += 1u) { z += clusters4[lid * n + j] * wx[j]; }
        wred[lid] = hsum(z);
    }
    workgroupBarrier();
    if (lid == 0u) {
        var mx = -3.402823e+38;
        for (var j = 0u; j < nc; j += 1u) { mx = max(mx, wred[j]); }
        var se = 0.0;
        for (var j = 0u; j < nc; j += 1u) { se += exp(wred[j] - mx); }
        winv = mx + log(se);
    }
    workgroupBarrier();
    if (lid < nc) { l1[p.vocab + lid] = wred[lid] - winv; }
}

// GDN projections of the normed token. q/k/v go through the depthwise causal
// conv and the ring shift in the same thread: a channel depends on nothing else.
@compute @workgroup_size(32)
fn w_gdn_project(@builtin(global_invocation_id) g: vec3<u32>) {
    let r = rb();
    let hidden = md[0u];
    let nv = md[24u];
    let vd = nv * md[27u];
    let cdim = md[29u];
    var id = g.x;
    if (id < cdim) {
        let x = mv(md[r + 56u], id, hidden, x_in(0u));
        let kk = md[28u];
        let ring = md[r + 25u];
        let taps = md[r + 60u] + id * kk;
        var acc = x * sm(taps + kk - 1u);
        for (var j = 0u; j + 1u < kk; j += 1u) {
            acc += state1[ring + j * cdim + id] * sm(taps + j);
        }
        s1[S_GDN_CQ + id] = acc / (1.0 + exp(-acc));
        for (var j = 0u; j + 2u < kk; j += 1u) {
            state1[ring + j * cdim + id] = state1[ring + (j + 1u) * cdim + id];
        }
        if (kk > 1u) { state1[ring + (kk - 2u) * cdim + id] = x; }
        return;
    }
    id -= cdim;
    if (id < vd) {
        s1[S_GDN_Z + id] = mv(md[r + 57u], id, hidden, x_in(1u));
        return;
    }
    id -= vd;
    if (id < nv) {
        let ab = mv2(md[r + 58u], md[r + 59u], id, hidden, x_in(2u), x_in(3u));
        s1[S_GDN_A + id] = ab.x;
        s1[S_GDN_B + id] = ab.y;
    }
}

@compute @workgroup_size(32)
fn w_gdn_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vd = md[24u] * md[27u];
    let k = chunks(vd);
    let id = gid.x / k;
    let ch = gid.x % k;
    if (id >= md[0u]) { return; }
    s1[S_PARTS + ch * PART_STRIDE + id] = mv_part(md[rb() + 29u], id, ch, vd, S_GDN_O);
}

@compute @workgroup_size(32)
fn w_anchor_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cols = md[7u] * md[9u];
    let k = chunks(cols);
    let id = gid.x / k;
    let ch = gid.x % k;
    if (id >= md[0u]) { return; }
    s1[S_PARTS + ch * PART_STRIDE + id] = mv_part(md[rb() + 14u], id, ch, cols, CP_ATT);
}

// Gate and up of the selected expert and of the shared FFN, with the activation.
@compute @workgroup_size(32)
fn w_ffn_up(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = rb();
    let hidden = md[0u];
    let inter = md[1u];
    var id = gid.x;
    if (md[r + 15u] == 1u) {
        if (id < inter) {
            let sel = u32(s4[S_SEL >> 2u].x);
            let gu = mv2(md[r + 32u + sel * 3u], md[r + 33u + sel * 3u], id, hidden, x_in(2u), x_in(3u));
            s1[S_G + id] = fold1(md[r + 34u + sel * 3u], id, silu(gu.x) * gu.y);
            return;
        }
        id -= inter;
    }
    if (id < inter) {
        let gu = mv2(md[r + 21u], md[r + 22u], id, hidden, x_in(0u), x_in(1u));
        s1[S_UP + id] = fold1(md[r + 23u], id, silu(gu.x) * gu.y);
    }
}

// Down of both.
@compute @workgroup_size(32)
fn w_ffn_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let inter = md[1u];
    let k = chunks(inter);
    let id = gid.x / k;
    let ch = gid.x % k;
    if (id >= md[0u]) { return; }
    let r = rb();
    var y = mv_part(md[r + 23u], id, ch, inter, S_UP);
    if (md[r + 15u] == 1u) {
        let sel = u32(s4[S_SEL >> 2u].x);
        y += mv_part(md[r + 34u + sel * 3u], id, ch, inter, S_G);
    }
    s1[S_PARTS + ch * PART_STRIDE + id] = y;
}

// The head is hierarchical: 128 clusters, then the rows inside one. A workgroup is one
// cluster. The cluster scores come first, and a cluster below the gate (`p.phase_mass`,
// a log probability) reads no rows: nothing it holds can reach the top_p mass, and its
// rows stay off the bus, marked `DROPPED`. A gate of −∞ computes every row. An open
// cluster gets the log-sum-exp of its rows and both corrections.
const DROPPED: f32 = -1e30;

@compute @workgroup_size(256)
fn w_head(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let cl = wid.x;
    let cs = md[12u];
    if (cl >= md[11u]) { return; }
    let start = cl * cs;
    let g = l1[p.vocab + cl];
    if (g < p.phase_mass) {
        if (lid < cs) { l1[start + lid] = DROPPED; }
        return;
    }
    if (lid < cs) { wred[lid] = mv_head(start + lid); }
    workgroupBarrier();
    if (lid == 0u) {
        var m = -3.402823e+38;
        for (var j = 0u; j < cs; j += 1u) { m = max(m, wred[j]); }
        var se = 0.0;
        for (var j = 0u; j < cs; j += 1u) { se += exp(wred[j] - m); }
        winv = g - (m + log(se));
    }
    workgroupBarrier();
    if (lid < cs) { l1[start + lid] = wred[lid] + winv; }
}

// Bounded anchor on the ring alone: the token's key and value go straight into
// their slot (position mod W) before the attention reads them, and no thread
// waits on another: scores per four keys, the softmax per head, the sum per four
// columns. Scores and weights live in the plane the f32 path keeps for chunk gates.
@group(0) @binding(8) var<storage, read_write> kv4: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> kv1: array<f32>;
const S_SC: u32 = 303104u;
const UMAX: u32 = 0xffffffffu;

fn group_window(kh: u32) -> u32 {
    if (md[22u] != UMAX) { return u32(sm(md[22u] + kh)); }
    return md[18u];
}

fn score_stride() -> u32 { return (md[18u] + md[20u] + 3u) & ~3u; }
// The ring row of the key `d` steps back from the row `pm` of the current position.
fn ring_slot(pm: u32, d: u32, w: u32) -> u32 { return select(pm + w - d, pm - d, pm >= d); }

// The rotation of a key and of a query goes on the absolute position, which the host
// writes per token: (R(t)q)·(R(j)k) = (R(t−j)q)·k, so the ring holds rotated keys and
// a score is a plain dot product. A thread makes both rows of a rotated pair.
@compute @workgroup_size(32)
fn w_anchor_project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = rb();
    let hidden = md[0u];
    let hd = md[9u];
    let qrows = md[7u] * hd;
    let krows = md[8u] * hd;
    let half = md[16u] >> 1u;
    var id = gid.x;
    if (id < qrows) {
        let rel = id % hd;
        if (rel < half) {
            let q = mv2r(md[r + 11u], id, md[r + 11u], id + half, hidden, x_in(0u), x_in(0u));
            let c = s1[S_ROPE + rel];
            let sn = s1[S_ROPE + half + rel];
            s1[CP_Q + id] = q.x * c - q.y * sn;
            s1[CP_Q + id + half] = q.x * sn + q.y * c;
        } else if (rel >= 2u * half) {
            s1[CP_Q + id] = mv(md[r + 11u], id, hidden, x_in(0u));
        }
        return;
    }
    id -= qrows;
    let w = md[18u];
    if (id < krows) {
        let rel = id % hd;
        let o = md[r + 26u] + ((id / hd) * w + p.position % w) * hd + rel;
        if (rel < half) {
            let k = mv2r(md[r + 12u], id, md[r + 12u], id + half, hidden, x_in(1u), x_in(1u));
            let c = s1[S_ROPE + rel];
            let sn = s1[S_ROPE + half + rel];
            kv1[o] = k.x * c - k.y * sn;
            kv1[o + half] = k.x * sn + k.y * c;
        } else if (rel >= 2u * half) {
            kv1[o] = mv(md[r + 12u], id, hidden, x_in(1u));
        }
        return;
    }
    id -= krows;
    if (id >= krows) { return; }
    let o = md[r + 26u] + ((id / hd) * w + p.position % w) * hd + id % hd;
    kv1[o + md[8u] * w * hd] = mv(md[r + 13u], id, hidden, x_in(2u));
}

fn attend_score_one(hq: u32, kh: u32, u: u32) -> f32 {
    let hd = md[9u];
    let w = md[18u];
    let sink = md[20u];
    let r = rb();
    let n = hd >> 2u;
    let qb = (CP_Q + hq * hd) >> 2u;
    var dv = vec4<f32>(0.0);
    if (u < sink) {
        // A sink key has no position: the query goes back to its plain form.
        let k = (md[r + 27u] + (kh * sink + u) * hd) >> 2u;
        let half = md[16u] >> 1u;
        let h4 = half >> 2u;
        let co = S_ROPE >> 2u;
        for (var i = 0u; i < h4; i += 1u) {
            let q0 = s4[qb + i];
            let q1 = s4[qb + h4 + i];
            let c = s4[co + i];
            let sn = s4[co + h4 + i];
            dv += (q0 * c + q1 * sn) * small4[k + i] + (q1 * c - q0 * sn) * small4[k + h4 + i];
        }
        for (var j = 2u * h4; j < n; j += 1u) {
            dv += s4[qb + j] * small4[k + j];
        }
    } else {
        let ko = (md[r + 26u] + (kh * w + ring_slot(p.position % w, u - sink, w)) * hd) >> 2u;
        for (var j = 0u; j < n; j += 1u) {
            dv += s4[qb + j] * kv4[ko + j];
        }
    }
    return hsum(dv) * inverseSqrt(f32(hd));
}

@compute @workgroup_size(32)
fn w_attend_score(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qheads = md[7u];
    let sink = md[20u];
    let w = md[18u];
    let per = (sink + min(p.position + 1u, w) + 3u) >> 2u;
    let hq = gid.x / per;
    let u4 = gid.x % per;
    if (hq >= qheads) { return; }
    let kh = hq / (qheads / md[8u]);
    let total = sink + min(p.position + 1u, group_window(kh));
    if (u4 * 4u >= total) { return; }
    let cell = ((S_SC + 256u + hq * score_stride()) >> 2u) + u4;
    let first = u4 * 4u;
    // A cell past the head's last key stays below every score.
    if (first < sink || first + 4u > total) {
        var out = vec4<f32>(-3.402823e+38);
        for (var c = 0u; c < 4u; c += 1u) {
            let u = first + c;
            if (u < total) { out[c] = attend_score_one(hq, kh, u); }
        }
        s4[cell] = out;
        return;
    }
    // Four ring rows against one read of the query.
    let hd = md[9u];
    let n = hd >> 2u;
    let qb = (CP_Q + hq * hd) >> 2u;
    let ring = md[rb() + 26u] + kh * w * hd;
    let pm = p.position % w;
    let d = first - sink;
    let k0 = (ring + ring_slot(pm, d, w) * hd) >> 2u;
    let k1 = (ring + ring_slot(pm, d + 1u, w) * hd) >> 2u;
    let k2 = (ring + ring_slot(pm, d + 2u, w) * hd) >> 2u;
    let k3 = (ring + ring_slot(pm, d + 3u, w) * hd) >> 2u;
    var z0 = vec4<f32>(0.0);
    var z1 = vec4<f32>(0.0);
    var z2 = vec4<f32>(0.0);
    var z3 = vec4<f32>(0.0);
    for (var j = 0u; j < n; j += 1u) {
        let q = s4[qb + j];
        z0 += q * kv4[k0 + j];
        z1 += q * kv4[k1 + j];
        z2 += q * kv4[k2 + j];
        z3 += q * kv4[k3 + j];
    }
    s4[cell] = vec4<f32>(hsum(z0), hsum(z1), hsum(z2), hsum(z3)) * inverseSqrt(f32(hd));
}

// The softmax of one head. A key whose weight is below e^-16 of the largest one cannot
// reach a sum that holds the largest: the list that the value sum walks leaves it out.
const WEIGHT_FLOOR: f32 = -16.0;

@compute @workgroup_size(1)
fn w_attend_weights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hq = gid.x;
    let qheads = md[7u];
    if (hq >= qheads) { return; }
    let kh = hq / (qheads / md[8u]);
    let total = md[20u] + min(p.position + 1u, group_window(kh));
    let stride = score_stride();
    let b4 = (S_SC + 256u + hq * stride) >> 2u;
    let wo = S_SC + 256u + (qheads + hq) * stride;
    let uo = S_SC + 256u + (2u * qheads + hq) * stride;
    let n4 = (total + 3u) >> 2u;
    var mx4 = vec4<f32>(-3.402823e+38);
    for (var u = 0u; u < n4; u += 1u) {
        mx4 = max(mx4, s4[b4 + u]);
    }
    let mx = vec4<f32>(max(max(mx4.x, mx4.y), max(mx4.z, mx4.w)));
    var l = 0.0;
    var count = 0u;
    for (var u = 0u; u < n4; u += 1u) {
        let d = s4[b4 + u] - mx;
        if (any(d >= vec4<f32>(WEIGHT_FLOOR))) {
            let v = exp(d);
            for (var c = 0u; c < 4u; c += 1u) {
                if (d[c] >= WEIGHT_FLOOR) {
                    l += v[c];
                    s1[wo + count] = v[c];
                    s1[uo + count] = f32(u * 4u + c);
                    count += 1u;
                }
            }
        }
    }
    s4[(S_SC >> 2u) + hq] = vec4<f32>(1.0 / max(l, 1e-20), f32(count), 0.0, 0.0);
}

// The value sum goes by blocks: a thread adds 32 keys of the list over 8 cells of a row,
// so it stays short and reads whole cache lines of the ring; the next stage adds the
// blocks up.
const KEY_BLOCK: u32 = 32u;
const COL_BLOCK: u32 = 8u;
const S_PART: u32 = 434176u;

fn part_at(hq: u32, b: u32) -> u32 {
    return (S_PART + (hq * ((md[18u] + md[20u] + KEY_BLOCK - 1u) / KEY_BLOCK) + b) * md[9u]) >> 2u;
}

@compute @workgroup_size(32)
fn w_attend_part(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qheads = md[7u];
    let hd = md[9u];
    let w = md[18u];
    let sink = md[20u];
    let n = hd >> 2u;
    let cols = (n + COL_BLOCK - 1u) / COL_BLOCK;
    let blocks = (sink + min(p.position + 1u, w) + KEY_BLOCK - 1u) / KEY_BLOCK;
    let hq = gid.x / (blocks * cols);
    if (hq >= qheads) { return; }
    let rest = gid.x % (blocks * cols);
    let b = rest / cols;
    let c = rest % cols;
    let info = s4[(S_SC >> 2u) + hq];
    let count = u32(info.y);
    let first = b * KEY_BLOCK;
    if (first >= count) { return; }
    let r = rb();
    let kh = hq / (qheads / md[8u]);
    let stride = score_stride();
    let wo = S_SC + 256u + (qheads + hq) * stride;
    let uo = S_SC + 256u + (2u * qheads + hq) * stride;
    let vb = md[r + 26u] + md[8u] * w * hd + kh * w * hd;
    let pm = p.position % w;
    let c0 = c * COL_BLOCK;
    let m = min(COL_BLOCK, n - c0);
    var acc: array<vec4<f32>, 8>;
    for (var i = first; i < min(first + KEY_BLOCK, count); i += 1u) {
        let t = s1[wo + i];
        let u = u32(s1[uo + i]);
        if (u < sink) {
            let row = ((md[r + 28u] + (kh * sink + u) * hd) >> 2u) + c0;
            for (var j = 0u; j < m; j += 1u) { acc[j] += t * small4[row + j]; }
        } else {
            let row = ((vb + ring_slot(pm, u - sink, w) * hd) >> 2u) + c0;
            for (var j = 0u; j < m; j += 1u) { acc[j] += t * kv4[row + j]; }
        }
    }
    let o = part_at(hq, b) + c0;
    for (var j = 0u; j < m; j += 1u) { s4[o + j] = acc[j]; }
}

@compute @workgroup_size(32)
fn w_attend_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qheads = md[7u];
    let hd = md[9u];
    let n = hd >> 2u;
    let hq = gid.x / n;
    let d4 = gid.x % n;
    if (hq >= qheads) { return; }
    let info = s4[(S_SC >> 2u) + hq];
    let blocks = (u32(info.y) + KEY_BLOCK - 1u) / KEY_BLOCK;
    var acc = vec4<f32>(0.0);
    for (var b = 0u; b < blocks; b += 1u) {
        acc += s4[part_at(hq, b) + d4];
    }
    s4[((CP_ATT + hq * hd) >> 2u) + d4] = fold4(md[rb() + 14u], ((hq * hd) >> 2u) + d4, acc * info.x);
}

// Delta-rule state step and gated RMSNorm per head, one workgroup per head. A lane owns
// four value columns over a block of key rows: kv = kfᵀS_old, S ← g·S + kf ⊗ β(v − g·kv),
// o = qfᵀS, then x̂ · norm · silu(z). The state is device-only, and a column's rows lie
// in one run: [head][col/4][row].
var<workgroup> wo: array<vec4<f32>, 128>;
var<workgroup> wp: array<vec4<f32>, 128>;
var<workgroup> wnq: f32;
var<workgroup> wnk: f32;

fn gdn_softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}

@compute @workgroup_size(128)
fn w_gdn_step(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let h = wid.x;
    let nv = md[24u];
    if (h >= nv) { return; }
    let nk = md[25u];
    let dk = md[26u];
    let dv = md[27u];
    let kk = md[28u];
    let cdim = md[29u];
    let dv4 = dv >> 2u;
    let dk4 = dk >> 2u;
    let r = rb();
    let ko = h / (nv / nk);
    let kd = nk * dk;
    let qb = (S_GDN_CQ + ko * dk) >> 2u;
    let kb = (S_GDN_CQ + kd + ko * dk) >> 2u;
    // The norms of the query and key rows of this head.
    var a = vec2<f32>(0.0);
    if (lid < dk4) {
        let q = s4[qb + lid];
        let k = s4[kb + lid];
        a = vec2<f32>(dot(q, q), dot(k, k));
    }
    wo[lid] = vec4<f32>(a, 0.0, 0.0);
    workgroupBarrier();
    if (lid == 0u) {
        var t = vec2<f32>(0.0);
        for (var j = 0u; j < dk4; j += 1u) { t += wo[j].xy; }
        wnq = t.x;
        wnk = t.y;
    }
    workgroupBarrier();
    let blocks = 128u / dv4;
    let col = lid % dv4;
    let blk = lid / dv4;
    let live = blk < blocks;
    let first = (blk * dk4) / blocks;
    let end = ((blk + 1u) * dk4) / blocks;
    let base = ((md[r + 25u] + (kk - 1u) * cdim + h * dk * dv) >> 2u) + col * dk;
    var kv = vec4<f32>(0.0);
    if (live) {
        for (var c4 = first; c4 < end; c4 += 1u) {
            let kf = s4[kb + c4];
            let b4 = base + 4u * c4;
            kv += state4[b4] * kf.x + state4[b4 + 1u] * kf.y + state4[b4 + 2u] * kf.z + state4[b4 + 3u] * kf.w;
        }
    }
    wo[lid] = kv;
    workgroupBarrier();
    let invq = 1.0 / (sqrt(wnq + 1e-6) * sqrt(f32(dk)));
    let invk = 1.0 / sqrt(wnk + 1e-6);
    var o = vec4<f32>(0.0);
    if (live) {
        kv = vec4<f32>(0.0);
        for (var b = 0u; b < blocks; b += 1u) { kv += wo[col + b * dv4]; }
        let g = exp(-exp(sm(md[r + 61u] + h)) * gdn_softplus(s1[S_GDN_A + h] + sm(md[r + 62u] + h)));
        let beta = 1.0 / (1.0 + exp(-s1[S_GDN_B + h]));
        let vt = s4[((S_GDN_CQ + 2u * kd + h * dv) >> 2u) + col];
        let delta = (vt - g * kv * invk) * beta * invk;
        for (var c4 = first; c4 < end; c4 += 1u) {
            let kf = s4[kb + c4];
            let qf = s4[qb + c4];
            let b4 = base + 4u * c4;
            let s0 = g * state4[b4] + kf.x * delta;
            let s1_ = g * state4[b4 + 1u] + kf.y * delta;
            let s2 = g * state4[b4 + 2u] + kf.z * delta;
            let s3 = g * state4[b4 + 3u] + kf.w * delta;
            state4[b4] = s0;
            state4[b4 + 1u] = s1_;
            state4[b4 + 2u] = s2;
            state4[b4 + 3u] = s3;
            o += s0 * qf.x + s1_ * qf.y + s2 * qf.z + s3 * qf.w;
        }
    }
    wp[lid] = o;
    workgroupBarrier();
    if (lid < dv4) {
        o = vec4<f32>(0.0);
        for (var b = 0u; b < blocks; b += 1u) { o += wp[col + b * dv4]; }
        o *= invq;
        wo[lid] = o;
        wred[lid] = dot(o, o);
    }
    workgroupBarrier();
    if (lid == 0u) {
        var t = 0.0;
        for (var j = 0u; j < dv4; j += 1u) { t += wred[j]; }
        winv = 1.0 / sqrt(t / f32(dv) + p.eps);
    }
    workgroupBarrier();
    if (lid < dv4) {
        let cell = ((h * dv) >> 2u) + lid;
        let z = s4[(S_GDN_Z >> 2u) + cell];
        let gate = z / (vec4<f32>(1.0) + exp(-z));
        s4[(S_GDN_O >> 2u) + cell] = fold4(md[r + 29u], cell, wo[lid] * winv * small4[(md[r + 63u] >> 2u) + lid] * gate);
    }
}

const S_ROUTE_REDUCED: u32 = 32768u;

// The reconstruction error of the resonance router, a thread per (expert, row, chunk):
// row 0 is the squared distance to the expert's centre, rows 1..=rank its projections.
@compute @workgroup_size(32)
fn w_route_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = rb();
    let hidden = md[0u];
    let rank = md[r + 20u];
    let comps = rank + 1u;
    let k = chunks(hidden);
    let ec = gid.x / k;
    let ch = gid.x % k;
    if (ec >= md[r + 16u] * comps) { return; }
    let expert = ec / comps;
    let component = ec % comps;
    let first = ch * (CHUNK >> 2u);
    let end = min(first + (CHUNK >> 2u), hidden >> 2u);
    let mu = (md[r + 17u] + expert * hidden) >> 2u;
    var z = vec4<f32>(0.0);
    if (component == 0u) {
        for (var j = first; j < end; j += 1u) {
            let d = s4[j] - small4[mu + j];
            z += d * d;
        }
    } else {
        let u = (md[r + 18u] + (expert * rank + component - 1u) * hidden) >> 2u;
        for (var j = first; j < end; j += 1u) {
            z += (s4[j] - small4[mu + j]) * small4[u + j];
        }
    }
    s1[S_ROUTE_REDUCED + gid.x] = hsum(z);
}

// Score and pick in one workgroup: the top score wins, the lower expert id wins an
// exact tie like the CPU loop. A growth shell (record word 30: E shells then one −∞
// sentinel) masks an expert whose error exceeds it, as the host `Resonance::scores`
// does; trunk rows carry +inf and are never masked. Then the input of the chosen
// expert's gate and up.
var<workgroup> wsel: u32;

@compute @workgroup_size(256)
fn w_route_final(@builtin(local_invocation_index) lid: u32) {
    let r = rb();
    let ne = md[r + 16u];
    let rank = md[r + 20u];
    if (lid < ne) {
        let k = chunks(md[0u]);
        let base = S_ROUTE_REDUCED + lid * (rank + 1u) * k;
        var d2 = 0.0;
        for (var ch = 0u; ch < k; ch += 1u) { d2 += s1[base + ch]; }
        var pn = 0.0;
        for (var c = 1u; c <= rank; c += 1u) {
            var v = 0.0;
            for (var ch = 0u; ch < k; ch += 1u) { v += s1[base + c * k + ch]; }
            pn += v * v;
        }
        let err = d2 - pn;
        var score = sm(md[r + 19u] + lid) - err;
        let so = md[r + 30u];
        if (so != UMAX && err > sm(so + lid)) { score = sm(so + ne); }
        wred[lid] = score;
    }
    workgroupBarrier();
    if (lid == 0u) {
        var best = 0u;
        var top = wred[0];
        for (var e = 1u; e < ne; e += 1u) {
            if (wred[e] > top) {
                top = wred[e];
                best = e;
            }
        }
        wsel = best;
        s1[S_SEL] = f32(best);
    }
    workgroupBarrier();
    if (lid < (md[0u] >> 2u)) {
        let sel = wsel;
        let v = s4[lid];
        fold_lane(2u, md[r + 32u + sel * 3u], lid, v);
        fold_lane(3u, md[r + 33u + sel * 3u], lid, v);
    }
}
