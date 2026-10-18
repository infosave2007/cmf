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

// ── Whole-token graph kernels (decode, b = 1) ─────────────────────────────
//
// `blr_gv` (stage V) and `blr_gu` / `blr_gu_act` (stage U) run inside the
// resident token graph: no upload, no readback, the latents stay in a graph
// scratch. One dispatch serves up to three tensors of one input (q|k|v,
// qkv|z, gate|up), so a group of `blr` projections costs two dispatches.
//
// * The latents of tensor j sit at `toff_j` in `t`, path p at `p·rpad`,
//   `rpad = ⌈r/32⌉·32`, the pad entries written as 0 — stage U then reads
//   whole sign words without masking (the bits past `r` meet zeros).
// * Stage V: a 32-lane subgroup owns 4 latent rows, the lanes walk the row's
//   32-bit sign words (coalesced), `z = x ⊙ g_p` is formed per word in
//   registers and shared by the 4 rows, and a subgroup sum (or, without
//   32-wide subgroups, a workgroup-memory sum) closes each row.
// * Stage U: one thread per output row, 32 rows a workgroup. The rows'
//   sign bits are staged through workgroup memory in chunks of ≤ 32 words
//   (row-major U rows are byte-aligned, so a lane funnel-shifts two words;
//   the loads of one chunk row are 32 consecutive words), then every thread
//   walks its own row against the latents, which all threads read at the
//   same address (a broadcast). `blr_gu_act` computes gate and up of the
//   same rows and writes act(gate)·up.
//
// Sign rule: bit 1 → +v, bit 0 → −v, applied by flipping the f32 sign bit
// (`~bits` shifted to bit 31, xor), three instructions a bit.
const GCOMMON: &str = r#"
fn bsd4(nb: u32, sh: u32, z: vec4<f32>) -> f32 {
  let s = (vec4<u32>(nb) << vec4<u32>(31u - sh, 30u - sh, 29u - sh, 28u - sh)) & vec4<u32>(0x80000000u);
  let v = bitcast<vec4<f32>>(bitcast<vec4<u32>>(z) ^ s);
  return (v.x + v.y) + (v.z + v.w);
}
fn bsd32(b: u32, z0: vec4<f32>, z1: vec4<f32>, z2: vec4<f32>, z3: vec4<f32>,
         z4: vec4<f32>, z5: vec4<f32>, z6: vec4<f32>, z7: vec4<f32>) -> f32 {
  let nb = ~b;
  return ((bsd4(nb, 0u, z0) + bsd4(nb, 4u, z1)) + (bsd4(nb, 8u, z2) + bsd4(nb, 12u, z3)))
       + ((bsd4(nb, 16u, z4) + bsd4(nb, 20u, z5)) + (bsd4(nb, 24u, z6) + bsd4(nb, 28u, z7)));
}
"#;

/// Stage V, the body for one tensor slot (`WB` its payload, `MJ` its
/// (r, P, toff, d_out) word, `J` the slot).
const GV_BODY: &str = r#"
fn gv_f16_J(byte_off: u32) -> f32 {
  let v = unpack2x16float(WB[byte_off >> 2u]);
  return select(v.x, v.y, (byte_off & 2u) != 0u);
}
fn gv_run_J(blk: u32, warp: u32, lane: u32, lid: u32) {
  let m = MJ;
  let r = m.x;
  let np = m.y;
  let toff = m.z;
  let d_out = m.w;
  let cols = gvp.cols;
  let nw = cols >> 5u;
  let rpad = (r + 31u) & ~31u;
  let nrows = np * rpad;
  var q0 = (blk * GV_WARPSu + warp) * 4u;
  let ok = q0 < nrows;
  q0 = min(q0, nrows - 4u);
  let p = q0 / rpad;
  let k0 = q0 - p * rpad;
  let s = 2u * (d_out + cols + r) + (r >> 3u) * (d_out + cols);
  let base = 16u + p * s;
  let gw = (base + 2u * d_out) >> 2u;
  let lb = base + 2u * (d_out + cols);
  let vw = (lb + 2u * r + d_out * (r >> 3u)) >> 2u;
  let ra = vw + min(k0, r - 1u) * nw;
  let rb = vw + min(k0 + 1u, r - 1u) * nw;
  let rc = vw + min(k0 + 2u, r - 1u) * nw;
  let rd = vw + min(k0 + 3u, r - 1u) * nw;
  var a0 = 0.0;
  var a1 = 0.0;
  var a2 = 0.0;
  var a3 = 0.0;
  for (var w = lane; w < nw; w = w + 32u) {
    let xo = w * 8u;
    let go = gw + w * 16u;
    let ba = WB[ra + w];
    let bb = WB[rb + w];
    let bc = WB[rc + w];
    let bd = WB[rd + w];
    let z0 = gvx[xo] * vec4<f32>(unpack2x16float(WB[go]), unpack2x16float(WB[go + 1u]));
    let z1 = gvx[xo + 1u] * vec4<f32>(unpack2x16float(WB[go + 2u]), unpack2x16float(WB[go + 3u]));
    let z2 = gvx[xo + 2u] * vec4<f32>(unpack2x16float(WB[go + 4u]), unpack2x16float(WB[go + 5u]));
    let z3 = gvx[xo + 3u] * vec4<f32>(unpack2x16float(WB[go + 6u]), unpack2x16float(WB[go + 7u]));
    let z4 = gvx[xo + 4u] * vec4<f32>(unpack2x16float(WB[go + 8u]), unpack2x16float(WB[go + 9u]));
    let z5 = gvx[xo + 5u] * vec4<f32>(unpack2x16float(WB[go + 10u]), unpack2x16float(WB[go + 11u]));
    let z6 = gvx[xo + 6u] * vec4<f32>(unpack2x16float(WB[go + 12u]), unpack2x16float(WB[go + 13u]));
    let z7 = gvx[xo + 7u] * vec4<f32>(unpack2x16float(WB[go + 14u]), unpack2x16float(WB[go + 15u]));
    a0 = a0 + bsd32(ba, z0, z1, z2, z3, z4, z5, z6, z7);
    a1 = a1 + bsd32(bb, z0, z1, z2, z3, z4, z5, z6, z7);
    a2 = a2 + bsd32(bc, z0, z1, z2, z3, z4, z5, z6, z7);
    a3 = a3 + bsd32(bd, z0, z1, z2, z3, z4, z5, z6, z7);
  }
GV_REDUCE
  if (ok && lane == 0u) {
    let o = toff + q0;
    gvt[o] = select(0.0, s0 * gv_f16_J(lb + 2u * k0), k0 < r);
    gvt[o + 1u] = select(0.0, s1 * gv_f16_J(lb + 2u * k0 + 2u), k0 + 1u < r);
    gvt[o + 2u] = select(0.0, s2 * gv_f16_J(lb + 2u * k0 + 4u), k0 + 2u < r);
    gvt[o + 3u] = select(0.0, s3 * gv_f16_J(lb + 2u * k0 + 6u), k0 + 3u < r);
  }
}
"#;

const GV_REDUCE_SG: &str = r#"
  let s0 = subgroupAdd(a0);
  let s1 = subgroupAdd(a1);
  let s2 = subgroupAdd(a2);
  let s3 = subgroupAdd(a3);
"#;

const GV_REDUCE_WG: &str = r#"
  gvred[lid] = vec4<f32>(a0, a1, a2, a3);
  workgroupBarrier();
  var sv = vec4<f32>(0.0);
  if (lane == 0u) {
    for (var i = 0u; i < 32u; i = i + 1u) {
      sv = sv + gvred[warp * 32u + i];
    }
  }
  let s0 = sv.x;
  let s1 = sv.y;
  let s2 = sv.z;
  let s3 = sv.w;
"#;

const GV_HEAD: &str = r#"
struct GvP { cols: u32, e0: u32, e1: u32, nm: u32, m0: vec4<u32>, m1: vec4<u32>, m2: vec4<u32> };
@group(0) @binding(0) var<storage, read> gvw0: array<u32>;
@group(0) @binding(1) var<storage, read> gvw1: array<u32>;
@group(0) @binding(2) var<storage, read> gvw2: array<u32>;
@group(0) @binding(3) var<storage, read> gvx: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> gvt: array<f32>;
@group(0) @binding(5) var<uniform> gvp: GvP;
"#;

const GV_ENTRY_SG: &str = r#"
@compute @workgroup_size(GV_THREADS)
fn blr_gv(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_index) lid: u32,
          @builtin(subgroup_invocation_id) lane: u32) {
  let g = wid.x;
  let warp = lid >> 5u;
  if (g < gvp.e0) {
    gv_run_0(g, warp, lane, lid);
  } else if (g < gvp.e1) {
    gv_run_1(g - gvp.e0, warp, lane, lid);
  } else {
    gv_run_2(g - gvp.e1, warp, lane, lid);
  }
}
"#;

const GV_ENTRY_WG: &str = r#"
var<workgroup> gvred: array<vec4<f32>, GV_THREADS>;
@compute @workgroup_size(GV_THREADS)
fn blr_gv(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_index) lid: u32) {
  let g = wid.x;
  let warp = lid >> 5u;
  let lane = lid & 31u;
  if (g < gvp.e0) {
    gv_run_0(g, warp, lane, lid);
  } else if (g < gvp.e1) {
    gv_run_1(g - gvp.e0, warp, lane, lid);
  } else {
    gv_run_2(g - gvp.e1, warp, lane, lid);
  }
}
"#;

/// Stage U, the per-row dot of one tensor slot.
const GU_BODY: &str = r#"
fn gu_f16_J(byte_off: u32) -> f32 {
  let v = unpack2x16float(WB[byte_off >> 2u]);
  return select(v.x, v.y, (byte_off & 2u) != 0u);
}
fn gu_dot_J(i0: u32, lid: u32) -> f32 {
  let m = MJ;
  let r = m.x;
  let np = m.y;
  let toff = m.z;
  let d_out = m.w;
  let cols = gup.cols;
  let rpad = (r + 31u) & ~31u;
  let s = 2u * (d_out + cols + r) + (r >> 3u) * (d_out + cols);
  let i = min(i0 + lid, d_out - 1u);
  let nwr = rpad >> 5u;
  let nch = (nwr + 31u) >> 5u;
  let cw = (nwr + nch - 1u) / nch;
  var y = 0.0;
  for (var p = 0u; p < np; p = p + 1u) {
    let base = 16u + p * s;
    let ubit = (base + 2u * (d_out + cols + r)) * 8u;
    var acc = 0.0;
    for (var c = 0u; c < nch; c = c + 1u) {
      let w0 = c * cw;
      let nwc = min(cw, nwr - w0);
      if (lid < nwc) {
        let cb = ubit + (w0 + lid) * 32u;
        for (var rr = 0u; rr < 32u; rr = rr + 1u) {
          let bit = cb + min(i0 + rr, d_out - 1u) * r;
          let wi = bit >> 5u;
          let sh = bit & 31u;
          let lo = WB[wi];
          let hi = WB[wi + 1u];
          gusb[rr * 33u + lid] = select((lo >> sh) | (hi << ((32u - sh) & 31u)), lo, sh == 0u);
        }
      }
      workgroupBarrier();
      let tb = (toff + p * rpad + w0 * 32u) >> 2u;
      for (var k = 0u; k < nwc; k = k + 1u) {
        let b = gusb[lid * 33u + k];
        let t = tb + k * 8u;
        acc = acc + bsd32(b, gut[t], gut[t + 1u], gut[t + 2u], gut[t + 3u],
                          gut[t + 4u], gut[t + 5u], gut[t + 6u], gut[t + 7u]);
      }
      workgroupBarrier();
    }
    y = y + gu_f16_J(base + 2u * i) * acc;
  }
  return y;
}
"#;

const GU_HEAD: &str = r#"
struct GuP { cols: u32, e0: u32, e1: u32, act: u32, m0: vec4<u32>, m1: vec4<u32>, m2: vec4<u32> };
@group(0) @binding(0) var<storage, read> guw0: array<u32>;
@group(0) @binding(1) var<storage, read> guw1: array<u32>;
@group(0) @binding(2) var<storage, read> guw2: array<u32>;
@group(0) @binding(3) var<storage, read> gut: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> guy0: array<f32>;
@group(0) @binding(5) var<storage, read_write> guy1: array<f32>;
@group(0) @binding(6) var<storage, read_write> guy2: array<f32>;
@group(0) @binding(7) var<uniform> gup: GuP;
var<workgroup> gusb: array<u32, 1056>;

fn gu_erf(x: f32) -> f32 {
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0
      - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t
          + 0.2548296)
          * t)
          * exp(-a * a);
  return select(y, -y, x < 0.0);
}

@compute @workgroup_size(32)
fn blr_gu(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_index) lid: u32) {
  let g = wid.x;
  if (g < gup.e0) {
    let i0 = g * 32u;
    let y = gu_dot_0(i0, lid);
    if (i0 + lid < gup.m0.w) { guy0[i0 + lid] = y; }
  } else if (g < gup.e1) {
    let i0 = (g - gup.e0) * 32u;
    let y = gu_dot_1(i0, lid);
    if (i0 + lid < gup.m1.w) { guy1[i0 + lid] = y; }
  } else {
    let i0 = (g - gup.e1) * 32u;
    let y = gu_dot_2(i0, lid);
    if (i0 + lid < gup.m2.w) { guy2[i0 + lid] = y; }
  }
}

// act(gate)·up of the same rows: slot 0 gate, slot 1 up, out in y0.
// `act` = `GraphAct::code`: 0 SiLU, 1 exact GELU, 2 tanh GELU (the formulas
// of `silu_mul_pre`).
@compute @workgroup_size(32)
fn blr_gu_act(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
  let i0 = wid.x * 32u;
  let gv = gu_dot_0(i0, lid);
  let uv = gu_dot_1(i0, lid);
  var v = (gv / (1.0 + exp(-gv))) * uv;
  if (gup.act == 1u) { v = 0.5 * gv * (1.0 + gu_erf(gv * 0.70710678)) * uv; }
  if (gup.act == 2u) { v = 0.5 * gv * (1.0 + tanh(0.7978846 * (gv + 0.044715 * gv * gv * gv))) * uv; }
  if (i0 + lid < gup.m0.w) { guy0[i0 + lid] = v; }
  // Never taken: keeps bindings 2, 5 and 6 in the derived layout, so both
  // stage-U entry points bind the same eight buffers in order.
  if (gup.act == 0xFFFFFFFFu) {
    guy1[0] = f32(guw2[0]);
    guy2[0] = 0.0;
  }
}
"#;

/// Warps (32 lanes) per stage-V workgroup; each owns 4 latent rows.
const GV_WARPS: usize = 4;
/// Latent rows per stage-V workgroup.
const GV_ROWS_WG: usize = 4 * GV_WARPS;
/// Output rows per stage-U workgroup.
const GU_ROWS_WG: usize = 32;

fn gv_source(sg: bool) -> String {
    let mut s = String::new();
    s.push_str(GV_HEAD);
    s.push_str(GCOMMON);
    for j in 0..3 {
        s.push_str(
            &GV_BODY
                .replace("GV_REDUCE", if sg { GV_REDUCE_SG } else { GV_REDUCE_WG })
                .replace("WB", &format!("gvw{j}"))
                .replace("MJ", &format!("gvp.m{j}"))
                .replace("_J", &format!("_{j}")),
        );
    }
    s.push_str(if sg { GV_ENTRY_SG } else { GV_ENTRY_WG });
    s.replace("GV_WARPSu", &format!("{GV_WARPS}u"))
        .replace("GV_THREADS", &format!("{}", 32 * GV_WARPS))
}

fn gu_source() -> String {
    let mut s = String::new();
    s.push_str(GU_HEAD);
    s.push_str(GCOMMON);
    for j in 0..3 {
        s.push_str(
            &GU_BODY
                .replace("WB", &format!("guw{j}"))
                .replace("MJ", &format!("gup.m{j}"))
                .replace("_J", &format!("_{j}")),
        );
    }
    s
}

/// The graph kernels of one device context.
pub(super) struct BlrGraph {
    v: wgpu::ComputePipeline,
    u: wgpu::ComputePipeline,
    ua: wgpu::ComputePipeline,
    v_l: wgpu::BindGroupLayout,
    u_l: wgpu::BindGroupLayout,
    ua_l: wgpu::BindGroupLayout,
    /// Never-written output for the unused stage-U slots.
    dummy_y: wgpu::Buffer,
    /// Stage V reduces with subgroup sums (32-wide subgroups).
    pub(super) sg: bool,
}

fn compile_graph(c: &Ctx) -> Option<BlrGraph> {
    let sg = c.device.features().contains(wgpu::Features::SUBGROUP)
        && c.adapter_info.subgroup_min_size == 32
        && c.adapter_info.subgroup_max_size == 32
        && std::env::var("CMF_BLR_SG").as_deref() != Ok("0");
    let mk = |label: &str, src: &str, ep: &str| -> Option<wgpu::ComputePipeline> {
        let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let p = c
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: None,
                module: &m,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: c.pipeline_cache.as_ref(),
            });
        if let Some(e) = pollster::block_on(scope.pop()) {
            tracing::warn!("{label} module rejected ({e}): blr keeps the per-op path");
            return None;
        }
        Some(p)
    };
    let gu = gu_source();
    let v = mk("blr-gv", &gv_source(sg), "blr_gv")?;
    let u = mk("blr-gu", &gu, "blr_gu")?;
    let ua = mk("blr-gu-act", &gu, "blr_gu_act")?;
    Some(BlrGraph {
        v_l: v.get_bind_group_layout(0),
        u_l: u.get_bind_group_layout(0),
        ua_l: ua.get_bind_group_layout(0),
        v,
        u,
        ua,
        dummy_y: c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blr-dummy-y"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }),
        sg,
    })
}

/// Geometry of a `blr` payload (from its validated header).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlrDims {
    pub rows: usize,
    pub cols: usize,
    pub r: usize,
    pub np: usize,
}

impl BlrDims {
    /// Latents per path as stored in the graph scratch (whole sign words).
    pub fn rpad(&self) -> usize {
        self.r.div_ceil(32) * 32
    }
    /// Scratch floats this tensor's latents take.
    pub fn latents(&self) -> usize {
        self.np * self.rpad()
    }
}

/// The graph kernels' contract on a `blr` tensor: whole 32-column sign
/// words for stage V, word-aligned sections (`rows % 4`), a word-sized
/// payload, and offsets that stay inside u32 bit addressing.
pub(crate) fn graph_dims(model: &cortiq_core::CmfModel, idx: usize) -> Option<BlrDims> {
    let e = model.tensors.get(idx)?;
    if e.dtype != cortiq_core::TensorDtype::BinaryLowRank {
        return None;
    }
    let bytes = model.entry_bytes(e);
    let h = cortiq_core::blr::validate_blr(&e.shape, bytes).ok()?;
    let d = BlrDims {
        rows: h.d_out,
        cols: h.d_in,
        r: h.rank,
        np: h.paths,
    };
    let ok = d.cols % 32 == 0
        && d.rows % 4 == 0
        && d.rows >= 4
        && bytes.len() % 4 == 0
        && (bytes.len() as u64) * 8 < u32::MAX as u64;
    ok.then_some(d)
}

/// The graph kernels of `c`, compiled on first use; None when the device
/// rejected them (`CMF_BLR_GRAPH=0` also answers None).
pub(super) fn graph_pipes(c: &Ctx) -> Option<Arc<BlrGraph>> {
    if std::env::var("CMF_BLR_GRAPH").as_deref() == Ok("0") {
        return None;
    }
    let key = c as *const Ctx as usize;
    let mut st = states().lock().unwrap();
    let s = st.entry(key).or_insert_with(|| BlrState {
        pipes: Arc::new(build_pipes(c)),
        t: None,
        graph: None,
        u16: std::collections::HashMap::new(),
    });
    if s.graph.is_none() {
        s.graph = Some(compile_graph(c).map(Arc::new));
    }
    s.graph.clone().flatten()
}

/// A content-cached 16-word uniform (the bind-group cache needs stable
/// buffers for repeating parameters).
fn uniform16(c: &Ctx, v: [u32; 16]) -> wgpu::Buffer {
    let key = c as *const Ctx as usize;
    let mut st = states().lock().unwrap();
    let s = st.entry(key).or_insert_with(|| BlrState {
        pipes: Arc::new(build_pipes(c)),
        t: None,
        graph: None,
        u16: std::collections::HashMap::new(),
    });
    s.u16
        .entry(v)
        .or_insert_with(|| {
            use wgpu::util::DeviceExt;
            c.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("blr-u16"),
                    contents: bytemuck::cast_slice(&v),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        })
        .clone()
}

/// One tensor of a graph group: its resident payload, geometry and output.
pub(super) struct GroupMat<'a> {
    pub buf: &'a wgpu::Buffer,
    pub dims: BlrDims,
    pub y: &'a wgpu::Buffer,
}

/// Scratch floats a group needs (all of its tensors' latents).
pub(super) fn group_latents(mats: &[GroupMat<'_>]) -> usize {
    mats.iter().map(|m| m.dims.latents()).sum()
}

/// Encode stage V + stage U for up to three `blr` tensors of ONE input `x`
/// (`cols` wide) into the open pass of `enc`. `act = Some(code)` takes
/// exactly two tensors (gate, up) of equal rows and writes act(gate)·up
/// into the first tensor's `y`. `bind(stage, layout, buffers)` makes (or
/// fetches) the bind group: stage 0 = V, 1 = U. False = outside the
/// kernels' contract (nothing was encoded).
pub(super) fn encode_group(
    c: &Ctx,
    g: &BlrGraph,
    enc: &mut wgpu::CommandEncoder,
    mats: &[GroupMat<'_>],
    x: &wgpu::Buffer,
    t: &wgpu::Buffer,
    act: Option<u32>,
    bind: &dyn Fn(u32, &wgpu::BindGroupLayout, &[&wgpu::Buffer]) -> wgpu::BindGroup,
) -> bool {
    if mats.is_empty() || mats.len() > 3 {
        return false;
    }
    let cols = mats[0].dims.cols;
    if mats.iter().any(|m| m.dims.cols != cols) {
        return false;
    }
    if act.is_some() && (mats.len() != 2 || mats[0].dims.rows != mats[1].dims.rows) {
        return false;
    }
    // (r, P, toff, d_out) per slot; unused slots repeat slot 0 with empty
    // workgroup ranges.
    let mut mw = [0u32; 12];
    let mut toff = 0usize;
    let mut ev = [0u32; 3];
    let mut eu = [0u32; 3];
    let (mut accv, mut accu) = (0usize, 0usize);
    for j in 0..3 {
        let m = mats.get(j).unwrap_or(&mats[0]);
        let d = m.dims;
        mw[4 * j] = d.r as u32;
        mw[4 * j + 1] = d.np as u32;
        mw[4 * j + 2] = toff as u32;
        mw[4 * j + 3] = d.rows as u32;
        if j < mats.len() {
            toff += d.latents();
            accv += d.latents().div_ceil(GV_ROWS_WG);
            accu += d.rows.div_ceil(GU_ROWS_WG);
        }
        ev[j] = accv as u32;
        eu[j] = accu as u32;
    }
    if accv as u32 > 65_535 || accu as u32 > 65_535 {
        return false;
    }
    let pv = uniform16(
        c,
        [
            cols as u32, ev[0], ev[1], mats.len() as u32, mw[0], mw[1], mw[2], mw[3], mw[4],
            mw[5], mw[6], mw[7], mw[8], mw[9], mw[10], mw[11],
        ],
    );
    let w = |j: usize| mats.get(j).unwrap_or(&mats[0]).buf;
    let bv = bind(0, &g.v_l, &[w(0), w(1), w(2), x, t, &pv]);
    let (pipe_u, bu, groups_u) = match act {
        Some(code) => {
            let pu = uniform16(
                c,
                [
                    cols as u32, eu[0], eu[0], code, mw[0], mw[1], mw[2], mw[3], mw[4], mw[5],
                    mw[6], mw[7], mw[8], mw[9], mw[10], mw[11],
                ],
            );
            (
                &g.ua,
                bind(
                    1,
                    &g.ua_l,
                    &[w(0), w(1), w(0), t, mats[0].y, &g.dummy_y, &g.dummy_y, &pu],
                ),
                eu[0],
            )
        }
        None => {
            let pu = uniform16(
                c,
                [
                    cols as u32, eu[0], eu[1], 0, mw[0], mw[1], mw[2], mw[3], mw[4], mw[5], mw[6],
                    mw[7], mw[8], mw[9], mw[10], mw[11],
                ],
            );
            let y = |j: usize| mats.get(j).map_or(&g.dummy_y, |m| m.y);
            (
                &g.u,
                bind(1, &g.u_l, &[w(0), w(1), w(2), t, y(0), y(1), y(2), &pu]),
                eu[2],
            )
        }
    };
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&g.v);
    pass.set_bind_group(0, &bv, &[]);
    pass.dispatch_workgroups(accv as u32, 1, 1);
    pass.set_pipeline(pipe_u);
    pass.set_bind_group(0, &bu, &[]);
    pass.dispatch_workgroups(groups_u, 1, 1);
    true
}

/// A bind group over `bufs` at bindings 0.. in order (the uncached maker).
pub(super) fn plain_bind(
    c: &Ctx,
    layout: &wgpu::BindGroupLayout,
    bufs: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let e: Vec<_> = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| bind_buf(i as u32, b))
        .collect();
    c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blr-graph"),
        layout,
        entries: &e,
    })
}

struct BlrPipes {
    v: wgpu::ComputePipeline,
    u: wgpu::ComputePipeline,
}

/// Pipelines and the latent scratch, per device context.
struct BlrState {
    pipes: Arc<BlrPipes>,
    t: Option<(wgpu::Buffer, u64)>,
    /// The graph kernels (None inside = the device rejected them).
    graph: Option<Option<Arc<BlrGraph>>>,
    u16: std::collections::HashMap<[u32; 16], wgpu::Buffer>,
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
            graph: None,
            u16: std::collections::HashMap::new(),
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

    fn read_f32(c: &Ctx, src: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let stage = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blr-test-stage"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut out = vec![0f32; n];
        assert!(readback(c, enc, src, &stage, (n * 4) as u64, &mut out));
        out
    }

    /// Dense reference `y = W x` from the codec's own dequant, in f64.
    fn dense_ref(bytes: &[u8], d_out: usize, d_in: usize, x: &[f32]) -> Vec<f64> {
        let mut w = vec![0f32; d_out * d_in];
        cortiq_core::blr::dequant_blr(bytes, d_out, d_in, &mut w).unwrap();
        (0..d_out)
            .map(|i| {
                (0..d_in)
                    .map(|j| w[i * d_in + j] as f64 * x[j] as f64)
                    .sum::<f64>()
            })
            .collect()
    }

    fn rel_err(got: &[f32], want: &[f64]) -> f64 {
        let scale = want.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        got.iter()
            .zip(want)
            .map(|(g, w)| (*g as f64 - w).powi(2))
            .sum::<f64>()
            .sqrt()
            / scale
    }

    /// The graph kernels (`encode_group`) against the codec's dequant: one,
    /// two and three tensors of one input, ranks that are not whole sign
    /// words (r % 32 ≠ 0, the stage-U funnel shift and the zeroed latent
    /// pads), P = 1 and 2, row counts that are not whole workgroups, a
    /// latent row longer than one 32-word chunk, and the fused act(gate)·up
    /// for all three activation codes. Both reductions of stage V (subgroup
    /// and workgroup memory) when the device has 32-wide subgroups.
    #[test]
    fn wgpu_blr_graph_kernels_match_dequant() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping blr graph kernel test");
            return;
        };
        let _op = c.op_gate.enter();
        let Some(g_main) = graph_pipes(c) else {
            panic!("blr graph kernels failed to compile");
        };
        let mut variants = vec![g_main.clone()];
        if g_main.sg {
            // The workgroup-memory reduction as well.
            unsafe { std::env::set_var("CMF_BLR_SG", "0") };
            variants.push(Arc::new(compile_graph(c).expect("wg variant")));
            unsafe { std::env::remove_var("CMF_BLR_SG") };
            assert!(!variants[1].sg);
        }
        let mut rng = crate::blr::tests::Rng(41);
        // (d_out, r, P) per tensor; all share d_in.
        let groups: &[(usize, &[(usize, usize, usize)])] = &[
            (64, &[(64, 8, 1)]),
            (96, &[(100, 24, 2)]),
            (256, &[(36, 32, 2), (68, 40, 1), (132, 128, 2)]),
            (512, &[(1028, 216, 2), (1028, 504, 1)]),
            (2048, &[(1092, 1064, 2), (1092, 1088, 1)]),
        ];
        for g in &variants {
            for &(d_in, specs) in groups {
                let x: Vec<f32> = (0..d_in).map(|_| rng.f()).collect();
                let x_buf = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("blr-test-x"),
                    contents: bytemuck::cast_slice(&x),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let payloads: Vec<Vec<u8>> = specs
                    .iter()
                    .map(|&(d_out, r, np)| crate::blr::tests::make(&mut rng, d_out, d_in, r, np))
                    .collect();
                let bufs: Vec<wgpu::Buffer> = payloads
                    .iter()
                    .map(|p| {
                        c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("blr-test-w"),
                            contents: p,
                            usage: wgpu::BufferUsages::STORAGE,
                        })
                    })
                    .collect();
                let ys: Vec<wgpu::Buffer> = specs
                    .iter()
                    .map(|&(d_out, _, _)| {
                        c.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("blr-test-y"),
                            size: (d_out * 4) as u64,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        })
                    })
                    .collect();
                let mats: Vec<GroupMat<'_>> = specs
                    .iter()
                    .enumerate()
                    .map(|(j, &(d_out, r, np))| GroupMat {
                        buf: &bufs[j],
                        dims: BlrDims {
                            rows: d_out,
                            cols: d_in,
                            r,
                            np,
                        },
                        y: &ys[j],
                    })
                    .collect();
                let t_buf = c.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("blr-test-t"),
                    // Room for the group and for a slot-0 tensor taken twice
                    // (the single-tensor act case).
                    size: (group_latents(&mats).max(2 * mats[0].dims.latents()) * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                let mut enc = c
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                assert!(encode_group(
                    c,
                    g,
                    &mut enc,
                    &mats,
                    &x_buf,
                    &t_buf,
                    None,
                    &|_, l, b| plain_bind(c, l, b),
                ));
                c.queue.submit([enc.finish()]);
                for (j, &(d_out, r, np)) in specs.iter().enumerate() {
                    let want = dense_ref(&payloads[j], d_out, d_in, &x);
                    let got = read_f32(c, &ys[j], d_out);
                    let e = rel_err(&got, &want);
                    assert!(
                        e < 1e-5,
                        "sg={} {d_out}x{d_in} r{r} P{np} (slot {j} of {}): rel {e}",
                        g.sg,
                        specs.len()
                    );
                }
                // Fused act(gate)·up over two tensors of equal rows.
                if specs.len() >= 2 && specs[0].0 == specs[1].0 || specs.len() == 1 {
                    let (gi, ui) = if specs.len() == 1 { (0, 0) } else { (0, 1) };
                    let d_out = specs[gi].0;
                    let wg = dense_ref(&payloads[gi], d_out, d_in, &x);
                    let wu = dense_ref(&payloads[ui], d_out, d_in, &x);
                    let pair = [
                        GroupMat {
                            buf: &bufs[gi],
                            dims: mats[gi].dims,
                            y: &ys[gi],
                        },
                        GroupMat {
                            buf: &bufs[ui],
                            dims: mats[ui].dims,
                            y: &ys[ui],
                        },
                    ];
                    for code in 0..3u32 {
                        let mut enc = c
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                        assert!(encode_group(
                            c,
                            g,
                            &mut enc,
                            &pair,
                            &x_buf,
                            &t_buf,
                            Some(code),
                            &|_, l, b| plain_bind(c, l, b),
                        ));
                        c.queue.submit([enc.finish()]);
                        let got = read_f32(c, &ys[gi], d_out);
                        let want: Vec<f64> = wg
                            .iter()
                            .zip(&wu)
                            .map(|(&a, &u)| {
                                let a = a as f32;
                                let v = match code {
                                    0 => a / (1.0 + (-a).exp()),
                                    1 => 0.5 * a * (1.0 + crate::inference::erf_f32(a * 0.70710678)),
                                    _ => 0.5 * a * (1.0 + (0.7978846 * (a + 0.044715 * a * a * a)).tanh()),
                                };
                                v as f64 * u
                            })
                            .collect();
                        let e = rel_err(&got, &want);
                        assert!(
                            e < 2e-5,
                            "sg={} act {code} {d_out}x{d_in}: rel {e} got {:?} want {:?}",
                            g.sg,
                            &got[..4],
                            &want[..4]
                        );
                    }
                }
            }
        }
    }
}
