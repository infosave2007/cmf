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
// Decode reads every sign bit once a token, so both stages are written for
// memory-level parallelism: coalesced 32-bit sign words, two words of every
// row a lane in flight before the first add, and no dependent load chains.
//
// * Latents. Stage U reads a row's sign words where they lie: a U row
//   starts at bit `i·r` of its section, `r % 8 == 0`, so the start sits
//   `σ = i·r mod 32 ∈ {0, 8, 16, 24}` bits into a word. Instead of
//   funnel-shifting the bits, the latents are stored four times, copy
//   `c` shifted by `8c` (`t_c[m] = t[m − 8c]`, zero outside `[8c, r + 8c)`):
//   the bits of the neighbouring rows that share a word meet zeros. Tensor
//   j's copies sit at `toff_j`, path p, copy c at `(4p + c)·(rpad + 32)`,
//   `rpad = ⌈r/32⌉·32`, each copy stored word-transposed (vec4 c of word w
//   at `c·(rpad/32 + 1) + w`: lanes that walk words read consecutive vec4s).
//   Rows `i ≡ k (mod 4)` share σ, so a subgroup that owns rows `k, k+4,
//   k+8, k+12` reads one copy for all four.
// * Stage V: a set of `LV` lanes (32: one subgroup, or 128: a workgroup,
//   chosen by the input width) owns 8 latent rows; the lanes walk the rows'
//   sign words four at a time (coalesced), `z = x ⊙ g_p` is formed per word in
//   registers (g resident as word-transposed f32) and shared by the 8 rows;
//   subgroup sums (or, without 32-wide subgroups, workgroup-memory sums)
//   close each row, which is written to its four copies (and the copies'
//   zero borders by the k = 0 set).
// * Stage U: a subgroup owns 4 output rows of one tensor (`blr_gu_act`: the
//   same 4 rows of gate and of up). Its lanes walk the rows' words over all
//   paths as one flattened sequence, four words a lane in flight; the latent
//   words a lane needs are the same for the 4 rows. A subgroup sum per (row,
//   path), then `y = Σ_p h_p·s_p`; `blr_gu_act` writes act(gate)·up.
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

const GV_HEAD: &str = r#"
struct GvP { cols: u32, e0: u32, e1: u32, nm: u32, m0: vec4<u32>, m1: vec4<u32>, m2: vec4<u32> };
@group(0) @binding(0) var<storage, read> gvw0: array<u32>;
@group(0) @binding(1) var<storage, read> gvw1: array<u32>;
@group(0) @binding(2) var<storage, read> gvw2: array<u32>;
@group(0) @binding(3) var<storage, read> gvx: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> gvt: array<f32>;
@group(0) @binding(5) var<uniform> gvp: GvP;
@group(0) @binding(6) var<storage, read> gvg0: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> gvg1: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> gvg2: array<vec4<f32>>;
var<workgroup> gvred: array<vec4<f32>, GV_RED>;
// Position of latent slot m in a copy stored word-transposed: the c-th vec4
// of every word is contiguous across words (stage U's lanes walk words and
// read the same vec4 of consecutive words — one coalesced access).
fn gv_tpos(m: u32, nwt: u32) -> u32 {
  let w = m >> 5u;
  let j = m & 31u;
  return ((j >> 2u) * nwt + w) * 4u + (j & 3u);
}
"#;

/// Stage V, the body for one tensor slot (`WB` its payload, `MJ` its
/// (r, P, toff, d_out) word, `_J` the slot suffix).
/// Stage V, the helpers of one tensor slot (`WB` its payload, `GB` its f32
/// g, `_J` the slot suffix). The row walk itself is generated
/// ([`gv_run_src`]) for 8 or 16 rows a set.
const GV_HELPERS: &str = r#"
fn gv_f16_J(byte_off: u32) -> f32 {
  let v = unpack2x16float(WB[byte_off >> 2u]);
  return select(v.x, v.y, (byte_off & 2u) != 0u);
}
// z = x ⊙ g for vec4 c of word w; g is stored word-transposed (vec4 c of
// word w at c·nw + w), so a subgroup reading consecutive words reads
// consecutive g vec4s. x keeps the producer's layout.
fn gv_z_J(xo: u32, go: u32, c: u32, nw: u32) -> vec4<f32> {
  return gvx[xo + c] * GB[go + c * nw];
}
// The sign words of 4 rows at word w (none past the row).
fn gv_w4_J(rows: vec4<u32>, w: u32, nw: u32) -> vec4<u32> {
  if (w >= nw) { return vec4<u32>(0u); }
  return vec4<u32>(WB[rows.x + w], WB[rows.y + w], WB[rows.z + w], WB[rows.w + w]);
}
fn gv_d4_J(b: vec4<u32>, z0: vec4<f32>, z1: vec4<f32>, z2: vec4<f32>, z3: vec4<f32>,
           z4: vec4<f32>, z5: vec4<f32>, z6: vec4<f32>, z7: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(bsd32(b.x, z0, z1, z2, z3, z4, z5, z6, z7),
                   bsd32(b.y, z0, z1, z2, z3, z4, z5, z6, z7),
                   bsd32(b.z, z0, z1, z2, z3, z4, z5, z6, z7),
                   bsd32(b.w, z0, z1, z2, z3, z4, z5, z6, z7));
}
fn gv_l4_J(lb: u32, k: u32, r: u32, s: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(select(0.0, s.x * gv_f16_J(lb + 2u * k), k < r),
                   select(0.0, s.y * gv_f16_J(lb + 2u * k + 2u), k + 1u < r),
                   select(0.0, s.z * gv_f16_J(lb + 2u * k + 4u), k + 2u < r),
                   select(0.0, s.w * gv_f16_J(lb + 2u * k + 6u), k + 3u < r));
}
"#;

/// The stage-V row walk of one slot for `rg` groups of 4 rows: `ns` words a
/// lane in flight per row group and round, z formed once per word for all
/// `4·rg` rows, then the row sums (`reduce`: "sg32" one subgroup a set,
/// "sgn" several subgroups a set, "wg" workgroup memory) and the writes of
/// the four shifted latent copies.
fn gv_run_src(rg: usize, ns: usize, reduce: &str) -> String {
    let mut o = String::new();
    let g = |f: &dyn Fn(usize) -> String, sep: &str| -> String {
        (0..rg).map(f).collect::<Vec<_>>().join(sep)
    };
    // The dot of word w for every row group (z formed once).
    o.push_str(&format!(
        "struct GvD_J {{ {} }};\n",
        g(&|i| format!("d{i}: vec4<f32>"), ", ")
    ));
    o.push_str(&format!(
        "fn gv_dot_J(w: u32, nw: u32, gw: u32, {}) -> GvD_J {{\n",
        g(&|i| format!("b{i}: vec4<u32>"), ", ")
    ));
    o.push_str(&format!(
        "  if (w >= nw) {{ return GvD_J({}); }}\n",
        g(&|_| "vec4<f32>(0.0)".to_string(), ", ")
    ));
    o.push_str("  let xo = w * 8u;\n  let go = gw + w;\n");
    for c in 0..8 {
        o.push_str(&format!("  let z{c} = gv_z_J(xo, go, {c}u, nw);\n"));
    }
    o.push_str(&format!(
        "  return GvD_J({});\n}}\n",
        g(&|i| format!("gv_d4_J(b{i}, z0, z1, z2, z3, z4, z5, z6, z7)"), ", ")
    ));
    let r = 4 * rg;
    o.push_str(&format!(
        r#"fn gv_run_J(blk: u32, tid: u32, lane: u32, warp: u32) {{
  let m = MJ;
  let r = m.x;
  let np = m.y;
  let toff = m.z;
  let d_out = m.w;
  let cols = gvp.cols;
  let nw = cols >> 5u;
  let rpad = (r + 31u) & ~31u;
  let ts = rpad + 32u;
  let nrows = np * rpad;
  let vset = tid / GV_LVu;
  let li = tid - vset * GV_LVu;
  // {r} latent rows a set (rpad % 32 == 0: never across a path).
  var q0 = (blk * GV_SETSu + vset) * {r}u;
  let ok = q0 < nrows;
  q0 = min(q0, nrows - {r}u);
  let p = q0 / rpad;
  let k0 = q0 - p * rpad;
  let s = 2u * (d_out + cols + r) + (r >> 3u) * (d_out + cols);
  let base = 16u + p * s;
  let gw = p * (cols >> 2u);
  let lb = base + 2u * (d_out + cols);
  let vw = (lb + 2u * r + d_out * (r >> 3u)) >> 2u;
  let rk = r - 1u;
"#
    ));
    for i in 0..rg {
        let k = 4 * i;
        o.push_str(&format!(
            "  let row{i} = vec4<u32>(vw + min(k0 + {k}u, rk) * nw, vw + min(k0 + {}u, rk) * nw, vw + min(k0 + {}u, rk) * nw, vw + min(k0 + {}u, rk) * nw);\n",
            k + 1,
            k + 2,
            k + 3
        ));
        o.push_str(&format!("  var acc{i} = vec4<f32>(0.0);\n"));
    }
    o.push_str("  var w = li;\n  loop {\n    if (w >= nw) { break; }\n");
    o.push_str("    // Every sign word of the round in flight before the first add.\n");
    for sl in 0..ns {
        for i in 0..rg {
            o.push_str(&format!(
                "    let b{sl}_{i} = gv_w4_J(row{i}, w + {sl}u * GV_LVu, nw);\n"
            ));
        }
    }
    for sl in 0..ns {
        o.push_str(&format!(
            "    let d{sl} = gv_dot_J(w + {sl}u * GV_LVu, nw, gw, {});\n",
            g(&|i| format!("b{sl}_{i}"), ", ")
        ));
    }
    for i in 0..rg {
        let sum = (0..ns)
            .map(|sl| format!("d{sl}.d{i}"))
            .collect::<Vec<_>>()
            .join(" + ");
        o.push_str(&format!("    acc{i} = acc{i} + ({sum});\n"));
    }
    o.push_str(&format!("    w = w + {ns}u * GV_LVu;\n  }}\n"));
    match reduce {
        "sg32" => {
            for i in 0..rg {
                o.push_str(&format!("  let s{i} = subgroupAdd(acc{i});\n"));
            }
        }
        "sgn" => {
            for i in 0..rg {
                o.push_str(&format!("  let sq{i} = subgroupAdd(acc{i});\n"));
            }
            o.push_str("  if (lane == 0u) {\n");
            for i in 0..rg {
                o.push_str(&format!("    gvred[{rg}u * warp + {i}u] = sq{i};\n"));
            }
            o.push_str("  }\n  workgroupBarrier();\n");
            for i in 0..rg {
                o.push_str(&format!("  var s{i} = vec4<f32>(0.0);\n"));
            }
            o.push_str("  for (var i = 0u; i < GV_LVu / 32u; i = i + 1u) {\n");
            for i in 0..rg {
                o.push_str(&format!("    s{i} = s{i} + gvred[{rg}u * i + {i}u];\n"));
            }
            o.push_str("  }\n");
        }
        _ => {
            for i in 0..rg {
                o.push_str(&format!("  gvred[{rg}u * tid + {i}u] = acc{i};\n"));
            }
            o.push_str("  workgroupBarrier();\n");
            for i in 0..rg {
                o.push_str(&format!("  var s{i} = vec4<f32>(0.0);\n"));
            }
            o.push_str("  if (li == 0u) {\n    for (var i = 0u; i < GV_LVu; i = i + 1u) {\n");
            for i in 0..rg {
                o.push_str(&format!(
                    "      s{i} = s{i} + gvred[{rg}u * (vset * GV_LVu + i) + {i}u];\n"
                ));
            }
            o.push_str("    }\n  }\n");
        }
    }
    o.push_str(
        "  let tp = toff + p * 4u * ts;\n  let nwt = ts >> 5u;\n  if (ok && li == 0u) {\n",
    );
    for i in 0..rg {
        let k = 4 * i;
        o.push_str(&format!(
            "    let l{i} = gv_l4_J(lb, k0 + {k}u, r, s{i});\n"
        ));
    }
    o.push_str(
        "    // k0 % 8 == 0 and 8c % 4 == 0: each run of four latents is one vec4 slot.\n    for (var c = 0u; c < 4u; c = c + 1u) {\n",
    );
    for i in 0..rg {
        let k = 4 * i;
        o.push_str(&format!(
            "      let o{i} = tp + c * ts + gv_tpos(k0 + {k}u + 8u * c, nwt);\n      gvt[o{i}] = l{i}.x;\n      gvt[o{i} + 1u] = l{i}.y;\n      gvt[o{i} + 2u] = l{i}.z;\n      gvt[o{i} + 3u] = l{i}.w;\n"
        ));
    }
    o.push_str(
        r#"    }
  }
  // The copies' zero borders, [0, 8c) and [rpad + 8c, rpad + 32).
  if (ok && k0 == 0u && li < 32u) {
    for (var c = 0u; c < 4u; c = c + 1u) {
      let o = tp + c * ts;
      let mm = select(rpad + li, li, li < 8u * c);
      gvt[o + gv_tpos(mm, nwt)] = 0.0;
    }
  }
}
"#,
    );
    o
}

const GV_ENTRY_SG: &str = r#"
@compute @workgroup_size(GV_WG)
fn blr_gv(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_index) tid: u32,
          @builtin(subgroup_invocation_id) lane: u32) {
  let g = wid.x;
  let warp = tid >> 5u;
  if (g < gvp.e0) {
    gv_run_0(g, tid, lane, warp);
  } else if (g < gvp.e1) {
    gv_run_1(g - gvp.e0, tid, lane, warp);
  } else {
    gv_run_2(g - gvp.e1, tid, lane, warp);
  }
}
"#;

const GV_ENTRY_WG: &str = r#"
@compute @workgroup_size(GV_WG)
fn blr_gv(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_index) tid: u32) {
  let g = wid.x;
  let warp = tid >> 5u;
  let lane = tid & 31u;
  if (g < gvp.e0) {
    gv_run_0(g, tid, lane, warp);
  } else if (g < gvp.e1) {
    gv_run_1(g - gvp.e0, tid, lane, warp);
  } else {
    gv_run_2(g - gvp.e1, tid, lane, warp);
  }
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
var<workgroup> gured: array<vec4<f32>, 128>;
var<workgroup> gured2: array<vec4<f32>, 4>;

// One stream walk of a subgroup's 4 rows: the paths' word rows (each
// `rpad/32 + 1` words: the shifted latent copy needs the spill word)
// flattened into `ntot` words.
struct GuGeo {
  np: u32, toff: u32, ts: u32, nwp: u32,
  ntot: u32, sidx: u32, uw0: u32, ps: u32,
  ro: vec4<u32>,
  i: vec4<u32>,
};
// Four word slots (f, f+32, f+64, f+96) of the 4 rows.
struct GuW { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, d: vec4<u32> };

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

// The 32 latents a word slot multiplies, dotted with the 4 rows' words,
// accumulated into its path (nothing past the walk).
fn gu_acc(g: GuGeo, f: u32, b: vec4<u32>, a0: ptr<function, vec4<f32>>, a1: ptr<function, vec4<f32>>) {
  if (f >= g.ntot) { return; }
  let p = select(0u, 1u, f >= g.nwp);
  let w = f - p * g.nwp;
  // Word-transposed copy: vec4 c of word w at c·nwp + w.
  let tb = ((g.toff + (p * 4u + g.sidx) * g.ts) >> 2u) + w;
  let n = g.nwp;
  let x0 = gut[tb];
  let x1 = gut[tb + n];
  let x2 = gut[tb + 2u * n];
  let x3 = gut[tb + 3u * n];
  let x4 = gut[tb + 4u * n];
  let x5 = gut[tb + 5u * n];
  let x6 = gut[tb + 6u * n];
  let x7 = gut[tb + 7u * n];
  let d = vec4<f32>(bsd32(b.x, x0, x1, x2, x3, x4, x5, x6, x7),
                    bsd32(b.y, x0, x1, x2, x3, x4, x5, x6, x7),
                    bsd32(b.z, x0, x1, x2, x3, x4, x5, x6, x7),
                    bsd32(b.w, x0, x1, x2, x3, x4, x5, x6, x7));
  if (p == 0u) {
    *a0 = *a0 + d;
  } else {
    *a1 = *a1 + d;
  }
}
"#;

/// Stage U, the 4 rows `rbase + 4ρ` of one tensor slot: geometry, word
/// fetch, and the h-scaled sum.
const GU_BODY: &str = r#"
fn gu_f16_J(byte_off: u32) -> f32 {
  let v = unpack2x16float(WB[byte_off >> 2u]);
  return select(v.x, v.y, (byte_off & 2u) != 0u);
}
fn gu_geo_J(rbase: u32) -> GuGeo {
  let m = MJ;
  let r = m.x;
  let d_out = m.w;
  let cols = gup.cols;
  let rpad = (r + 31u) & ~31u;
  let nwp = (rpad >> 5u) + 1u;
  let i = vec4<u32>(min(rbase, d_out - 1u), min(rbase + 4u, d_out - 1u),
                    min(rbase + 8u, d_out - 1u), min(rbase + 12u, d_out - 1u));
  var g: GuGeo;
  g.np = m.y;
  g.toff = m.z;
  g.ts = rpad + 32u;
  g.nwp = nwp;
  g.ntot = m.y * nwp;
  g.sidx = ((i.x * r) & 31u) >> 3u;
  g.uw0 = (16u + 2u * (d_out + cols + r)) >> 2u;
  g.ps = 2u * (d_out + cols + r) + (r >> 3u) * (d_out + cols);
  g.ro = (i * r) >> vec4<u32>(5u);
  g.i = i;
  return g;
}
fn gu_w4_J(g: GuGeo, f: u32) -> vec4<u32> {
  if (f >= g.ntot) { return vec4<u32>(0u); }
  let p = select(0u, 1u, f >= g.nwp);
  let ub = g.uw0 + ((p * g.ps) >> 2u) + f - p * g.nwp;
  return vec4<u32>(WB[ub + g.ro.x], WB[ub + g.ro.y], WB[ub + g.ro.z], WB[ub + g.ro.w]);
}
fn gu_fetch_J(g: GuGeo, f: u32) -> GuW {
  return GuW(gu_w4_J(g, f), gu_w4_J(g, f + 32u), gu_w4_J(g, f + 64u), gu_w4_J(g, f + 96u));
}
fn gu_h_J(g: GuGeo, s0: vec4<f32>, s1: vec4<f32>) -> vec4<f32> {
  var y = vec4<f32>(gu_f16_J(16u + 2u * g.i.x), gu_f16_J(16u + 2u * g.i.y),
                    gu_f16_J(16u + 2u * g.i.z), gu_f16_J(16u + 2u * g.i.w)) * s0;
  if (g.np > 1u) {
    let h1 = 16u + g.ps;
    y = y + vec4<f32>(gu_f16_J(h1 + 2u * g.i.x), gu_f16_J(h1 + 2u * g.i.y),
                      gu_f16_J(h1 + 2u * g.i.z), gu_f16_J(h1 + 2u * g.i.w)) * s1;
  }
  return y;
}
fn gu_rows_J(rbase: u32, lane: u32, warp: u32, tid: u32) -> vec4<f32> {
  let g = gu_geo_J(rbase);
  var a0 = vec4<f32>(0.0);
  var a1 = vec4<f32>(0.0);
  var f = lane;
  loop {
    if (f >= g.ntot) { break; }
    let w = gu_fetch_J(g, f);
    gu_acc(g, f, w.a, &a0, &a1);
    gu_acc(g, f + 32u, w.b, &a0, &a1);
    gu_acc(g, f + 64u, w.c, &a0, &a1);
    gu_acc(g, f + 96u, w.d, &a0, &a1);
    f = f + 128u;
  }
  let s0 = gu_red(a0, lane, warp, tid);
  let s1 = gu_red(a1, lane, warp, tid);
  return gu_h_J(g, s0, s1);
}
"#;

const GU_REDUCE_SG: &str = r#"
fn gu_red(v: vec4<f32>, lane: u32, warp: u32, tid: u32) -> vec4<f32> {
  return subgroupAdd(v);
}
"#;

const GU_REDUCE_WG: &str = r#"
fn gu_red(v: vec4<f32>, lane: u32, warp: u32, tid: u32) -> vec4<f32> {
  gured[tid] = v;
  workgroupBarrier();
  if (lane == 0u) {
    var s = vec4<f32>(0.0);
    for (var i = 0u; i < 32u; i = i + 1u) {
      s = s + gured[warp * 32u + i];
    }
    gured2[warp] = s;
  }
  workgroupBarrier();
  let r = gured2[warp];
  workgroupBarrier();
  return r;
}
"#;

const GU_ENTRY: &str = r#"
@compute @workgroup_size(128)
fn blr_gu(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_index) tid: uGU_LANE_ARG) {
  let warp = tid >> 5u;
GU_LANE_LET
  let g = wid.x;
  if (g < gup.e0) {
    let rb = g * 16u + warp;
    let y = gu_rows_0(rb, lane, warp, tid);
    if (lane < 4u && rb + 4u * lane < gup.m0.w) { guy0[rb + 4u * lane] = y[lane]; }
  } else if (g < gup.e1) {
    let rb = (g - gup.e0) * 16u + warp;
    let y = gu_rows_1(rb, lane, warp, tid);
    if (lane < 4u && rb + 4u * lane < gup.m1.w) { guy1[rb + 4u * lane] = y[lane]; }
  } else {
    let rb = (g - gup.e1) * 16u + warp;
    let y = gu_rows_2(rb, lane, warp, tid);
    if (lane < 4u && rb + 4u * lane < gup.m2.w) { guy2[rb + 4u * lane] = y[lane]; }
  }
}

// act(gate)·up of the same rows: slot 0 gate, slot 1 up, out in y0. The
// two tensors' sign words of a round are fetched together.
// `act` = `GraphAct::code`: 0 SiLU, 1 exact GELU, 2 tanh GELU (the formulas
// of `silu_mul_pre`). The tanh argument is clamped to ±10, exact in f32
// (tanh rounds to ±1 from ±9); Metal's tanh returns NaN for large arguments.
@compute @workgroup_size(128)
fn blr_gu_act(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) tid: uGU_LANE_ARG) {
  let warp = tid >> 5u;
GU_LANE_LET
  let rb = wid.x * 16u + warp;
  let gg = gu_geo_0(rb);
  let gu = gu_geo_1(rb);
  var ga0 = vec4<f32>(0.0);
  var ga1 = vec4<f32>(0.0);
  var ua0 = vec4<f32>(0.0);
  var ua1 = vec4<f32>(0.0);
  let nt = max(gg.ntot, gu.ntot);
  var f = lane;
  loop {
    if (f >= nt) { break; }
    let wg = gu_fetch_0(gg, f);
    let wu = gu_fetch_1(gu, f);
    gu_acc(gg, f, wg.a, &ga0, &ga1);
    gu_acc(gg, f + 32u, wg.b, &ga0, &ga1);
    gu_acc(gg, f + 64u, wg.c, &ga0, &ga1);
    gu_acc(gg, f + 96u, wg.d, &ga0, &ga1);
    gu_acc(gu, f, wu.a, &ua0, &ua1);
    gu_acc(gu, f + 32u, wu.b, &ua0, &ua1);
    gu_acc(gu, f + 64u, wu.c, &ua0, &ua1);
    gu_acc(gu, f + 96u, wu.d, &ua0, &ua1);
    f = f + 128u;
  }
  let gy = gu_h_0(gg, gu_red(ga0, lane, warp, tid), gu_red(ga1, lane, warp, tid));
  let uy = gu_h_1(gu, gu_red(ua0, lane, warp, tid), gu_red(ua1, lane, warp, tid));
  if (lane < 4u) {
    let gv = gy[lane];
    let uv = uy[lane];
    var v = (gv / (1.0 + exp(-gv))) * uv;
    if (gup.act == 1u) { v = 0.5 * gv * (1.0 + gu_erf(gv * 0.70710678)) * uv; }
    if (gup.act == 2u) { v = 0.5 * gv * (1.0 + tanh(clamp(0.7978846 * (gv + 0.044715 * gv * gv * gv), -10.0, 10.0))) * uv; }
    let row = rb + 4u * lane;
    if (row < gup.m0.w) { guy0[row] = v; }
  }
  // Never taken: keeps bindings 2, 5 and 6 in the derived layout, so both
  // stage-U entry points bind the same eight buffers in order.
  if (gup.act == 0xFFFFFFFFu) {
    guy1[0] = f32(guw2[0]);
    guy2[0] = 0.0;
  }
}
"#;

/// Output rows per stage-U workgroup (4 subgroups × 4 rows).
const GU_ROWS_WG: usize = 16;

/// Stage-V variants with a pipeline each: (lanes a set, row groups of 4).
const GV_VARIANTS: [(usize, usize); 7] =
    [(32, 2), (128, 2), (256, 2), (512, 2), (128, 4), (256, 4), (512, 4)];

/// Stage-V variant for an input `cols` wide: 32 lanes (one subgroup, 4 sets
/// a workgroup) below 2048 columns, else 128 lanes (one set a workgroup);
/// 8 rows a set. The 16-row and wider sets were measured at the 27B shapes
/// (device timestamps, DRAM regime, V+U pairs as the graph runs them): no
/// gain, the 17408-wide FFN down included (35.1 us with 128x8 against
/// 36.7 / 37.3 with 512x16 / 256x16; stage V alone looked 1.7x faster with
/// 512x16 only because back-to-back V dispatches overlap). `CMF_BLR_LV=32|
/// 128|256|512` and `CMF_BLR_VROWS=8|16` pin it (a pair without a pipeline
/// falls back to the default).
fn gv_variant(cols: usize) -> (usize, usize) {
    let lv = if cols >= 2048 { 128 } else { 32 };
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
    let pinned = (
        env("CMF_BLR_LV").unwrap_or(lv),
        env("CMF_BLR_VROWS").map(|r| r / 4).unwrap_or(2),
    );
    if GV_VARIANTS.contains(&pinned) {
        pinned
    } else {
        (lv, 2)
    }
}

/// Threads of a stage-V workgroup for lane-set width `lv`.
fn gv_wg(lv: usize) -> usize {
    lv.max(128)
}

fn gv_source(sg: bool, lv: usize, rg: usize) -> String {
    let reduce = match (sg, lv) {
        (true, 32) => "sg32",
        (true, _) => "sgn",
        (false, _) => "wg",
    };
    // Words a lane fetches a round per row group: 4 (8 rows) or 2 (16 rows).
    let ns = if rg >= 4 { 2 } else { 4 };
    let run = gv_run_src(rg, ns, reduce);
    let mut s = String::new();
    s.push_str(GV_HEAD);
    s.push_str(GCOMMON);
    for j in 0..3 {
        s.push_str(
            &format!("{GV_HELPERS}\n{run}")
                .replace("WB", &format!("gvw{j}"))
                .replace("GB", &format!("gvg{j}"))
                .replace("MJ", &format!("gvp.m{j}"))
                .replace("_J", &format!("_{j}")),
        );
    }
    s.push_str(if sg { GV_ENTRY_SG } else { GV_ENTRY_WG });
    s.replace("GV_LVu", &format!("{lv}u"))
        .replace("GV_SETSu", &format!("{}u", gv_wg(lv) / lv))
        .replace("GV_RED", &format!("{}", rg * gv_wg(lv)))
        .replace("GV_WG", &format!("{}", gv_wg(lv)))
}

fn gu_source(sg: bool) -> String {
    let mut s = String::new();
    s.push_str(GU_HEAD);
    s.push_str(GCOMMON);
    s.push_str(if sg { GU_REDUCE_SG } else { GU_REDUCE_WG });
    for j in 0..3 {
        s.push_str(
            &GU_BODY
                .replace("WB", &format!("guw{j}"))
                .replace("MJ", &format!("gup.m{j}"))
                .replace("_J", &format!("_{j}")),
        );
    }
    let (arg, let_) = if sg {
        (",\n          @builtin(subgroup_invocation_id) lane: u32", "")
    } else {
        ("", "  let lane = tid & 31u;")
    };
    s.push_str(
        &GU_ENTRY
            .replace("uGU_LANE_ARG", &format!("u32{arg}"))
            .replace("GU_LANE_LET", let_),
    );
    s
}

/// The graph kernels of one device context.
pub(super) struct BlrGraph {
    /// Stage V per variant ([`GV_VARIANTS`]) with its layout: the defaults
    /// compiled up front, a pinned variant on first use.
    v: std::sync::Mutex<
        std::collections::HashMap<(usize, usize), (wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    >,
    u: wgpu::ComputePipeline,
    ua: wgpu::ComputePipeline,
    u_l: wgpu::BindGroupLayout,
    ua_l: wgpu::BindGroupLayout,
    /// Never-written output for the unused stage-U slots.
    dummy_y: wgpu::Buffer,
    /// The row sums are subgroup sums (32-wide subgroups).
    pub(super) sg: bool,
}

fn compile_pipe(c: &Ctx, label: &str, src: &str, ep: &str) -> Option<wgpu::ComputePipeline> {
    {
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
                compilation_options: crate::gpu_wgpu::pco(),
                cache: c.pipeline_cache.as_ref(),
            });
        if let Some(e) = pollster::block_on(scope.pop()) {
            tracing::warn!("{label} module rejected ({e}): blr keeps the per-op path");
            if cfg!(test) || std::env::var("CMF_GPU_DEBUG").is_ok() {
                eprintln!("{label} module rejected: {e}");
            }
            return None;
        }
        Some(p)
    }
}

impl BlrGraph {
    /// The stage-V pipeline and layout of variant (lanes, row groups).
    fn v_pipe(
        &self,
        c: &Ctx,
        lv: usize,
        rg: usize,
    ) -> Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
        let mut m = self.v.lock().unwrap();
        if let Some(e) = m.get(&(lv, rg)) {
            return Some(e.clone());
        }
        let p = compile_pipe(
            c,
            &format!("blr-gv{lv}x{}", 4 * rg),
            &gv_source(self.sg, lv, rg),
            "blr_gv",
        )?;
        let l = p.get_bind_group_layout(0);
        m.insert((lv, rg), (p.clone(), l.clone()));
        Some((p, l))
    }
}

fn compile_graph(c: &Ctx) -> Option<BlrGraph> {
    let sg = c.device.features().contains(wgpu::Features::SUBGROUP)
        && c.adapter_info.subgroup_min_size == 32
        && c.adapter_info.subgroup_max_size == 32
        && std::env::var("CMF_BLR_SG").as_deref() != Ok("0");
    let gu = gu_source(sg);
    let u = compile_pipe(c, "blr-gu", &gu, "blr_gu")?;
    let ua = compile_pipe(c, "blr-gu-act", &gu, "blr_gu_act")?;
    let g = BlrGraph {
        u_l: u.get_bind_group_layout(0),
        ua_l: ua.get_bind_group_layout(0),
        v: std::sync::Mutex::new(std::collections::HashMap::new()),
        u,
        ua,
        dummy_y: c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blr-dummy-y"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }),
        sg,
    };
    // The default stage-V variants up front: a device that rejects them
    // keeps the per-op path from the start.
    g.v_pipe(c, 32, 2)?;
    g.v_pipe(c, 128, 2)?;
    Some(g)
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
    /// Latents per path rounded to whole sign words.
    pub fn rpad(&self) -> usize {
        self.r.div_ceil(32) * 32
    }
    /// Scratch floats this tensor's latents take: four shifted copies of
    /// `rpad + 32` per path.
    pub fn latents(&self) -> usize {
        self.np * 4 * (self.rpad() + 32)
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
        gbufs: std::collections::HashMap::new(),
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
        gbufs: std::collections::HashMap::new(),
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

/// The input scales of every path of a `blr` tensor as f32 (path p at
/// `p·cols`), resident once per (model, tensor): stage V multiplies them
/// into `x` per sign word, and an f16 unpack there costs as much as the
/// sign walk itself (measured: stage V without g took half the time).
pub(super) fn g_f32(c: &Ctx, model: &cortiq_core::CmfModel, idx: usize) -> Option<wgpu::Buffer> {
    let key = c as *const Ctx as usize;
    let mkey = (model.uid() as usize, idx);
    {
        let st = states().lock().unwrap();
        if let Some(b) = st.get(&key).and_then(|s| s.gbufs.get(&mkey)) {
            return Some(b.clone());
        }
    }
    let e = model.tensors.get(idx)?;
    let b = g_f32_upload(c, &e.shape, model.entry_bytes(e))?;
    crate::gpu::probe_note_cold();
    let mut st = states().lock().unwrap();
    let s = st.entry(key).or_insert_with(|| BlrState {
        pipes: Arc::new(build_pipes(c)),
        t: None,
        graph: None,
        u16: std::collections::HashMap::new(),
        gbufs: std::collections::HashMap::new(),
    });
    s.gbufs.insert(mkey, b.clone());
    Some(b)
}

/// [`g_f32`]'s buffer from a payload (uncached).
pub(super) fn g_f32_upload(c: &Ctx, shape: &[usize], bytes: &[u8]) -> Option<wgpu::Buffer> {
    let h = cortiq_core::blr::validate_blr(shape, bytes).ok()?;
    // Word-transposed per path (as the latent copies): vec4 c of sign word
    // w at `p·d_in/4 + c·(d_in/32) + w`, so stage V's lanes, which walk
    // words, read consecutive vec4s.
    let nw = h.d_in / 32;
    let mut g = vec![0f32; h.paths * h.d_in];
    for p in 0..h.paths {
        let o = cortiq_core::blr::blr_path_offsets(&h, p);
        let gp = cortiq_core::blr::blr_f16s(bytes, o.g, h.d_in);
        for (j, v) in gp.into_iter().enumerate() {
            let (w, jj) = (j / 32, j % 32);
            g[p * h.d_in + ((jj / 4) * nw + w) * 4 + jj % 4] = v;
        }
    }
    use wgpu::util::DeviceExt;
    Some(
        c.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("blr-g-f32"),
                contents: bytemuck::cast_slice(&g),
                usage: wgpu::BufferUsages::STORAGE,
            }),
    )
}

/// One tensor of a graph group: its resident payload, geometry and output.
pub(super) struct GroupMat<'a> {
    pub buf: &'a wgpu::Buffer,
    /// The paths' input scales g as f32, path p at `p·cols` ([`g_f32`]).
    pub g: &'a wgpu::Buffer,
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
    let Some([v, u]) = prep_group(c, g, mats, x, t, act, bind) else {
        return false;
    };
    let mut pass = begin_pass(enc);
    for (p, b, n) in [&v, &u] {
        pass.set_pipeline(p.as_ref());
        pass.set_bind_group(0, b, &[]);
        pass.dispatch_workgroups(*n, 1, 1);
    }
    true
}

/// One stage dispatch: pipeline, bind group, workgroups.
pub(super) type StageDispatch<'a> = (std::borrow::Cow<'a, wgpu::ComputePipeline>, wgpu::BindGroup, u32);

/// The two dispatches of [`encode_group`] (stage V, stage U) without
/// encoding them. None = outside the kernels' contract.
#[allow(clippy::too_many_arguments)]
pub(super) fn prep_group<'g>(
    c: &Ctx,
    g: &'g BlrGraph,
    mats: &[GroupMat<'_>],
    x: &wgpu::Buffer,
    t: &wgpu::Buffer,
    act: Option<u32>,
    bind: &dyn Fn(u32, &wgpu::BindGroupLayout, &[&wgpu::Buffer]) -> wgpu::BindGroup,
) -> Option<[StageDispatch<'g>; 2]> {
    if mats.is_empty() || mats.len() > 3 {
        return None;
    }
    let cols = mats[0].dims.cols;
    if mats.iter().any(|m| m.dims.cols != cols) {
        return None;
    }
    if act.is_some() && (mats.len() != 2 || mats[0].dims.rows != mats[1].dims.rows) {
        return None;
    }
    // (r, P, toff, d_out) per slot; unused slots repeat slot 0 with empty
    // workgroup ranges.
    let (lv, rg) = gv_variant(cols);
    // Latent rows a stage-V workgroup covers: 4·rg per set.
    let v_rows_wg = 4 * rg * (gv_wg(lv) / lv);
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
            accv += (d.np * d.rpad()).div_ceil(v_rows_wg);
            accu += d.rows.div_ceil(GU_ROWS_WG);
        }
        ev[j] = accv as u32;
        eu[j] = accu as u32;
    }
    if accv as u32 > 65_535 || accu as u32 > 65_535 {
        return None;
    }
    let pv = uniform16(
        c,
        [
            cols as u32, ev[0], ev[1], mats.len() as u32, mw[0], mw[1], mw[2], mw[3], mw[4],
            mw[5], mw[6], mw[7], mw[8], mw[9], mw[10], mw[11],
        ],
    );
    let w = |j: usize| mats.get(j).unwrap_or(&mats[0]).buf;
    let (pipe_v, layout_v) = g.v_pipe(c, lv, rg)?;
    let gb = |j: usize| mats.get(j).unwrap_or(&mats[0]).g;
    let bv = bind(0, &layout_v, &[w(0), w(1), w(2), x, t, &pv, gb(0), gb(1), gb(2)]);
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
    Some([
        (std::borrow::Cow::Owned(pipe_v), bv, accv as u32),
        (std::borrow::Cow::Borrowed(pipe_u), bu, groups_u),
    ])
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
    /// f32 input scales per (model uid, tensor idx), see [`g_f32`].
    gbufs: std::collections::HashMap<(usize, usize), wgpu::Buffer>,
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
                compilation_options: crate::gpu_wgpu::pco(),
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
            gbufs: std::collections::HashMap::new(),
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

    /// Throughput of the graph kernels at Qwen3.8-27B's NVG 0.55 bpw shapes
    /// (random payloads of the file's ranks), in the DRAM regime the token
    /// graph sees: each group is uploaded `BLR_BENCH_COPIES` times (default:
    /// enough copies for ~512 MB, past any L2) and one submit runs every
    /// copy's V+U pair once; device timestamps around the block, best of
    /// `BLR_BENCH_TRIES` (12). µs per pair and the payload's effective read
    /// bandwidth. `BLR_BENCH_STAGE=v|u` times one stage alone.
    /// `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn wgpu_blr_graph_throughput_27b_shapes() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping");
            return;
        };
        let _op = c.op_gate.enter();
        let g = graph_pipes(c).expect("graph kernels");
        let mut rng = crate::blr::tests::Rng(7);
        let stage = match std::env::var("BLR_BENCH_STAGE").as_deref() {
            Ok("v") => Some(0usize),
            Ok("u") => Some(1usize),
            _ => None,
        };
        let env_n = |k: &str, d: usize| -> usize {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let tries = env_n("BLR_BENCH_TRIES", 12);
        // (label, d_in, [(d_out, r, P)], act)
        let groups: &[(&str, usize, &[(usize, usize, usize)], bool)] = &[
            ("gdn qkv|z", 5120, &[(10240, 920, 2), (6144, 744, 2)], false),
            ("gdn out", 6144, &[(5120, 744, 2)], false),
            ("attn q|k|v", 5120, &[(12288, 976, 2), (1024, 216, 2), (1024, 216, 2)], false),
            ("ffn gate|up act", 5120, &[(17408, 1064, 2), (17408, 1064, 2)], true),
            ("ffn down P2", 17408, &[(5120, 1064, 2)], false),
            ("ffn down P1", 17408, &[(5120, 2152, 1)], false),
            ("0.8B gate|up act", 1024, &[(3584, 376, 2), (3584, 376, 2)], true),
            ("0.8B down", 3584, &[(1024, 776, 1)], false),
        ];
        let qs = c
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
            .then(|| {
                c.device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("blr-bench-ts"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                })
            });
        let qres = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blr-bench-qres"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let qstage = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blr-bench-qstage"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for &(label, d_in, specs, act) in groups {
            let payloads: Vec<Vec<u8>> = specs
                .iter()
                .map(|&(d_out, r, np)| crate::blr::tests::make(&mut rng, d_out, d_in, r, np))
                .collect();
            let bytes: usize = payloads.iter().map(|p| p.len()).sum();
            let copies = env_n("BLR_BENCH_COPIES", (512usize << 20).div_ceil(bytes).clamp(1, 96));
            // Per copy: the payloads and their f32 g.
            let bufs: Vec<Vec<(wgpu::Buffer, wgpu::Buffer)>> = (0..copies)
                .map(|_| {
                    specs
                        .iter()
                        .zip(&payloads)
                        .map(|(&(d_out, _, _), p)| {
                            (
                                c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("blr-bench-w"),
                                    contents: p,
                                    usage: wgpu::BufferUsages::STORAGE,
                                }),
                                g_f32_upload(c, &[d_out, d_in], p).unwrap(),
                            )
                        })
                        .collect()
                })
                .collect();
            let x: Vec<f32> = (0..d_in).map(|_| rng.f()).collect();
            let x_buf = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("blr-bench-x"),
                contents: bytemuck::cast_slice(&x),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let ys: Vec<wgpu::Buffer> = specs
                .iter()
                .map(|&(d_out, _, _)| {
                    c.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("blr-bench-y"),
                        size: (d_out * 4) as u64,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    })
                })
                .collect();
            let dims = |j: usize| BlrDims {
                rows: specs[j].0,
                cols: d_in,
                r: specs[j].1,
                np: specs[j].2,
            };
            let lat: usize = (0..specs.len()).map(|j| dims(j).latents()).sum();
            let t_buf = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("blr-bench-t"),
                size: (lat * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let code = act.then_some(0u32);
            let prepped: Vec<[StageDispatch<'_>; 2]> = bufs
                .iter()
                .map(|cp| {
                    let mats: Vec<GroupMat<'_>> = cp
                        .iter()
                        .enumerate()
                        .map(|(j, (w, gb))| GroupMat {
                            buf: w,
                            g: gb,
                            dims: dims(j),
                            y: &ys[j],
                        })
                        .collect();
                    prep_group(c, &g, &mats, &x_buf, &t_buf, code, &|_, l, b| plain_bind(c, l, b))
                        .expect("contract")
                })
                .collect();
            let mut best = f64::MAX;
            for _ in 0..tries {
                let mut enc = c
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                if let Some(q) = &qs {
                    enc.write_timestamp(q, 0);
                }
                {
                    let mut pass = begin_pass(&mut enc);
                    for vu in &prepped {
                        for (si, (p, b, n)) in vu.iter().enumerate() {
                            if stage.is_some_and(|s| s != si) {
                                continue;
                            }
                            pass.set_pipeline(p.as_ref());
                            pass.set_bind_group(0, b, &[]);
                            pass.dispatch_workgroups(*n, 1, 1);
                        }
                    }
                }
                if let Some(q) = &qs {
                    enc.write_timestamp(q, 1);
                    enc.resolve_query_set(q, 0..2, &qres, 0);
                    enc.copy_buffer_to_buffer(&qres, 0, &qstage, 0, 16);
                }
                let t0 = std::time::Instant::now();
                c.queue.submit([enc.finish()]);
                let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
                let host = t0.elapsed().as_secs_f64();
                let dt = if qs.is_some() {
                    let (tx, rx) = std::sync::mpsc::channel();
                    qstage.map_async(wgpu::MapMode::Read, ..16, move |r| {
                        let _ = tx.send(r);
                    });
                    let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
                    rx.recv().unwrap().unwrap();
                    let t: Vec<u64> = {
                        let raw = qstage.get_mapped_range(..16).unwrap();
                        bytemuck::cast_slice::<u8, u64>(&raw).to_vec()
                    };
                    qstage.unmap();
                    t[1].saturating_sub(t[0]) as f64 * c.queue.get_timestamp_period() as f64 * 1e-9
                } else {
                    host
                };
                best = best.min(dt);
            }
            let us = best * 1e6 / copies as f64;
            eprintln!(
                "blr-bench {label:18} {:7.2} MB x{copies:<3} {us:8.1} us/pair  {:7.1} GB/s",
                bytes as f64 / 1e6,
                bytes as f64 / (us * 1e-6) / 1e9
            );
        }
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
            (4096, &[(36, 24, 2), (36, 32, 1)]),
        ];
        // Every stage-V lane-set width on every group (the default picks
        // the width from the row length).
        for (g, (lvn, rgn)) in variants
            .iter()
            .flat_map(|g| GV_VARIANTS.iter().map(move |v| (g, *v)))
        {
            let lv = format!("{lvn}x{}", 4 * rgn);
            unsafe { std::env::set_var("CMF_BLR_LV", lvn.to_string()) };
            unsafe { std::env::set_var("CMF_BLR_VROWS", (4 * rgn).to_string()) };
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
                let gbufs: Vec<wgpu::Buffer> = specs
                    .iter()
                    .zip(&payloads)
                    .map(|(&(d_out, _, _), p)| g_f32_upload(c, &[d_out, d_in], p).unwrap())
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
                        g: &gbufs[j],
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
                        "sg={} lv={lv} {d_out}x{d_in} r{r} P{np} (slot {j} of {}): rel {e}",
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
                            g: &gbufs[gi],
                            dims: mats[gi].dims,
                            y: &ys[gi],
                        },
                        GroupMat {
                            buf: &bufs[ui],
                            g: &gbufs[ui],
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
                                    1 => 0.5 * a * (1.0 + crate::inference::erf_f32(a * std::f32::consts::FRAC_1_SQRT_2)),
                                    _ => 0.5 * a * (1.0 + (0.7978846 * (a + 0.044715 * a * a * a)).tanh()),
                                };
                                v as f64 * u
                            })
                            .collect();
                        let e = rel_err(&got, &want);
                        assert!(
                            e < 2e-5,
                            "sg={} lv={lv} act {code} {d_out}x{d_in}: rel {e} got {:?} want {:?}",
                            g.sg,
                            &got[..4],
                            &want[..4]
                        );
                    }
                }
            }
        }
        unsafe { std::env::remove_var("CMF_BLR_LV") };
        unsafe { std::env::remove_var("CMF_BLR_VROWS") };
    }
}
