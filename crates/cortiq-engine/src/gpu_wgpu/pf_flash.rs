//! Causal flash attention for the batch graph's 256-wide GQA layers
//! (Qwen3.5 / Qwen3.8 full attention), on the matrix units.
//!
//! The query-tiled split (`gqa_attend_part_xc2` + merge) ran the prompt's
//! attention at ~7 TFLOP/s: 109 of a 512-row chunk's 215 ms at 15 000
//! positions on an RTX A5000 (Qwen3.5-4B). This is `zimage::flash_src`
//! (51–54 TF bidirectional, 128-wide heads) reshaped for the prompt:
//!
//! - a workgroup is one KV group × a tile of 64 query rows (16 tokens × 4
//!   heads, the `q_f16_plane` row order), eight subgroups: row block
//!   `sg & 3` (16 rows) × V half `sg >> 2` (128 of the 256 columns). Both
//!   subgroups of a row block hold its 16 Q fragments and compute the same
//!   S (the QK product twice) so each keeps only 8 O fragments in registers;
//! - keys walk in blocks of 16 straight from the f16 mirror (or the K/V
//!   planes made from an f32 one), staged with a register prefetch, rows
//!   past the run read as zero;
//! - online softmax in f32 with the lazy rescale (threshold 2⁸, exact), the
//!   decision workgroup-uniform; a key past a row's own position is masked,
//!   blocks past the tile's last token are never read;
//! - the output is normalized and written as f32 rows of the batch's
//!   attention buffer (`[token][head][256]`), no partials, no merge.
//!
//! Same math as the split up to f16 rounding of P and the order of sums.
//! `CMF_PF_FLASH=0` keeps the split.

use super::*;

/// Keys per block.
const BC: u32 = 16;
/// Padded row of the K/V staging tiles (halves): 256 + 8, a multiple of 8.
const LDK: u32 = 264;
/// Padded row of a subgroup's P tile (halves).
const LDP: u32 = 24;
/// Keys per block of the split-S kernel (`src32`).
const BC2: u32 = 32;
/// Padded row of a row block's 16×32 P tile (halves).
const LDP2: u32 = 40;

/// The kernel `pipe` builds: 32-key blocks (`src32`) unless
/// `CMF_PF_FLASH_BC=16`.
fn wide() -> bool {
    std::env::var("CMF_PF_FLASH_BC").as_deref() != Ok("16")
}

/// Workgroup bytes of the chosen kernel.
fn shared_bytes(wide: bool) -> u32 {
    if wide {
        2 * BC2 * LDK * 2 + 8 * 256 * 4 + 4 * 16 * LDP2 * 2 + 1024 + 16
    } else {
        2 * BC * LDK * 2 + 8 * 256 * 4 + 8 * 16 * LDP * 2 + 1024 + 16
    }
}

/// [`src`] with 32-key blocks and the S product split between the two
/// subgroups of a row block: subgroup `dh` multiplies its 16 Q rows by keys
/// `16·dh..16·dh+16` of the block only (the 16-key kernel had both compute
/// the same S), the 16×32 S tile meets in workgroup memory, four lanes (two
/// per subgroup) share a row's softmax, and each subgroup then adds
/// P (16×32) · V (32 × its 128 columns). Per 32 keys a subgroup issues 32
/// MMAs instead of 48, and half the barriers per key. Same math.
pub(crate) fn src32() -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let thr = "8.0";
    let _ = writeln!(s, "enable f16;\nenable wgpu_cooperative_matrix;\ndiagnostic(off, derivative_uniformity);");
    let _ = writeln!(
        s,
        "struct FP {{ nh: u32, hpk: u32, cap: u32, pos0: u32, k: u32, kpad: u32, scl: f32, nkv: u32 }};"
    );
    let _ = writeln!(s, "@group(0) @binding(0) var<storage, read> qh: array<f16>;");
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> kq: array<vec4<f16>>;");
    let _ = writeln!(s, "@group(0) @binding(2) var<storage, read> vq: array<vec4<f16>>;");
    let _ = writeln!(s, "@group(0) @binding(3) var<storage, read_write> ob: array<f32>;");
    let _ = writeln!(s, "@group(0) @binding(4) var<uniform> p: FP;");
    let _ = writeln!(s, "var<workgroup> sk: array<f16, {}>;", BC2 * LDK);
    let _ = writeln!(s, "var<workgroup> sv: array<f16, {}>;", BC2 * LDK);
    // S tiles (4 row blocks × 16 × 32), later each subgroup's 16×16 O scratch.
    let _ = writeln!(s, "var<workgroup> ss: array<f32, {}>;", 8 * 256);
    let _ = writeln!(s, "var<workgroup> sp: array<f16, {}>;", 4 * 16 * LDP2);
    let _ = writeln!(s, "var<workgroup> smx: array<f32, 256>;");
    let _ = writeln!(s, "var<workgroup> sfl: array<u32, 4>;");
    let _ = writeln!(
        s,
        "@compute @workgroup_size(256)\nfn pf_flash(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32, @builtin(subgroup_id) sg: u32) {{"
    );
    let _ = writeln!(s, "  let g = wid.y;");
    let _ = writeln!(s, "  let tpt = 64u / p.hpk;");
    let _ = writeln!(s, "  let t0 = wid.x * tpt;");
    let _ = writeln!(s, "  let rb = sg & 3u; let dh = sg >> 2u;");
    let _ = writeln!(s, "  let lane = tid - sg * 32u;");
    let _ = writeln!(s, "  let r = lane & 15u; let hf = lane >> 4u;");
    // This lane's 8 keys of the block: part q of 4.
    let _ = writeln!(s, "  let qp = dh * 2u + hf;");
    let _ = writeln!(s, "  let rr = rb * 16u + r;");
    let _ = writeln!(s, "  let tl = t0 + rr / p.hpk;");
    let _ = writeln!(s, "  let hh = rr % p.hpk;");
    let _ = writeln!(s, "  let qpos = p.pos0 + tl;");
    let _ = writeln!(s, "  let live = tl < p.k;");
    let _ = writeln!(s, "  let qrow = (g * p.kpad + t0) * p.hpk + rb * 16u;");
    for d in 0..16 {
        let _ = writeln!(s, "  let qa{d} = coopLoadT<coop_mat16x16<f16, A>>(&qh[qrow * 256u + {}u], 256u);", d * 16);
    }
    for j in 0..8 {
        let _ = writeln!(s, "  var o{j}: coop_mat16x16<f32, C>;");
    }
    let _ = writeln!(s, "  var zc: coop_mat16x16<f32, C>;");
    let _ = writeln!(s, "  var mu = -1.0e30; var ls = 0.0;");
    let _ = writeln!(s, "  if (tid < 4u) {{ sfl[tid] = 0u; }}");
    let _ = writeln!(s, "  let tlast = min(t0 + tpt, p.k) - 1u;");
    let _ = writeln!(s, "  let kend = p.pos0 + tlast + 1u;");
    let _ = writeln!(s, "  let nkb = (kend + {}u) / {BC2}u;", BC2 - 1);
    let _ = writeln!(s, "  let kvb = g * p.cap * 64u;");
    // Prefetch: 32 keys × 64 vec4 = 2048 a tile, 8 a thread.
    let npf = (BC2 * 64 / 256) as usize;
    for t in 0..npf {
        let _ = writeln!(
            s,
            "  var rk{t} = vec4<f16>(0.0h); var rv{t} = vec4<f16>(0.0h);\n  {{ let idx = tid + {o}u; let key = idx / 64u; if (key < kend) {{ rk{t} = kq[kvb + key * 64u + idx % 64u]; rv{t} = vq[kvb + key * 64u + idx % 64u]; }} }}",
            o = t * 256
        );
    }
    let _ = writeln!(s, "  for (var kb = 0u; kb < nkb; kb = kb + 1u) {{");
    for t in 0..npf {
        let _ = writeln!(
            s,
            "    {{ let idx = tid + {o}u; let d = (idx / 64u) * {LDK}u + (idx % 64u) * 4u; sk[d] = rk{t}.x; sk[d + 1u] = rk{t}.y; sk[d + 2u] = rk{t}.z; sk[d + 3u] = rk{t}.w; sv[d] = rv{t}.x; sv[d + 1u] = rv{t}.y; sv[d + 2u] = rv{t}.z; sv[d + 3u] = rv{t}.w; }}",
            o = t * 256
        );
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "    if (kb + 1u < nkb) {{");
    for t in 0..npf {
        let _ = writeln!(
            s,
            "      {{ let idx = tid + {o}u; let key = (kb + 1u) * {BC2}u + idx / 64u; var a = vec4<f16>(0.0h); var b = vec4<f16>(0.0h); if (key < kend) {{ a = kq[kvb + key * 64u + idx % 64u]; b = vq[kvb + key * 64u + idx % 64u]; }} rk{t} = a; rv{t} = b; }}",
            o = t * 256
        );
    }
    let _ = writeln!(s, "    }}");
    // S half: this subgroup's 16 keys of the block.
    let _ = writeln!(s, "    let kofs = dh * {}u;", 16 * LDK);
    let _ = writeln!(s, "    let kf0 = coopLoad<coop_mat16x16<f16, B>>(&sk[kofs], {LDK}u);");
    let _ = writeln!(s, "    var s0 = coopMultiplyAdd(qa0, kf0, zc);");
    for d in 1..16 {
        let _ = writeln!(s, "    let kf{d} = coopLoad<coop_mat16x16<f16, B>>(&sk[kofs + {}u], {LDK}u);", d * 16);
        let _ = writeln!(s, "    s0 = coopMultiplyAdd(qa{d}, kf{d}, s0);");
    }
    let _ = writeln!(s, "    coopStoreT(s0, &ss[rb * 512u + dh * 16u], 32u);");
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "    let sbase = rb * 512u + r * 32u + qp * 8u;");
    let _ = writeln!(s, "    let kbase = kb * {BC2}u + qp * 8u;");
    let _ = writeln!(s, "    var pm = -1.0e30;");
    for e in 0..8 {
        let _ = writeln!(
            s,
            "    var sv{e} = -1.0e30; if (kbase + {e}u <= qpos && live) {{ sv{e} = ss[sbase + {e}u] * p.scl; }} pm = max(pm, sv{e});"
        );
    }
    let _ = writeln!(s, "    smx[tid] = pm;");
    let _ = writeln!(s, "    if (pm > mu + {thr} && live) {{ sfl[kb & 1u] = 1u; }}");
    let _ = writeln!(s, "    let need = workgroupUniformLoad(&sfl[kb & 1u]);");
    let _ = writeln!(s, "    if (tid == 0u) {{ sfl[(kb + 1u) & 1u] = 0u; }}");
    // The row's four lanes: tid ^ 16 (other half), tid ^ 128 (other subgroup).
    let _ = writeln!(s, "    let rmax = max(max(pm, smx[tid ^ 16u]), max(smx[tid ^ 128u], smx[tid ^ 144u]));");
    let _ = writeln!(s, "    if (need != 0u) {{");
    let _ = writeln!(s, "      let mn = max(mu, rmax);");
    let _ = writeln!(s, "      let alpha = exp2(mu - mn);");
    let _ = writeln!(s, "      ls = ls * alpha; mu = mn;");
    let _ = writeln!(s, "      if (kb > 0u) {{");
    let _ = writeln!(s, "        let obs = sg * 256u + r * 16u + hf * 8u;");
    for j in 0..8 {
        let _ = writeln!(s, "        coopStoreT(o{j}, &ss[sg * 256u], 16u);");
        let _ = writeln!(s, "        workgroupBarrier();");
        for e in 0..8 {
            let _ = writeln!(s, "        ss[obs + {e}u] = ss[obs + {e}u] * alpha;");
        }
        let _ = writeln!(s, "        workgroupBarrier();");
        let _ = writeln!(s, "        o{j} = coopLoadT<coop_mat16x16<f32, C>>(&ss[sg * 256u], 16u);");
        let _ = writeln!(s, "        workgroupBarrier();");
    }
    let _ = writeln!(s, "      }}");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    let pbase = rb * {}u + r * {LDP2}u + qp * 8u;", 16 * LDP2);
    for e in 0..8 {
        let _ = writeln!(s, "    let pv{e} = select(0.0, exp2(sv{e} - mu), sv{e} > -1.0e29);");
        let _ = writeln!(s, "    ls = ls + pv{e}; sp[pbase + {e}u] = f16(pv{e});");
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "    let pa0 = coopLoadT<coop_mat16x16<f16, A>>(&sp[rb * {}u], {LDP2}u);", 16 * LDP2);
    let _ = writeln!(s, "    let pa1 = coopLoadT<coop_mat16x16<f16, A>>(&sp[rb * {}u + 16u], {LDP2}u);", 16 * LDP2);
    for j in 0..8 {
        let _ = writeln!(
            s,
            "    let vf{j} = coopLoadT<coop_mat16x16<f16, B>>(&sv[dh * 128u + {}u], {LDK}u);",
            j * 16
        );
        let _ = writeln!(s, "    o{j} = coopMultiplyAdd(pa0, vf{j}, o{j});");
        let _ = writeln!(
            s,
            "    let wf{j} = coopLoadT<coop_mat16x16<f16, B>>(&sv[{}u + dh * 128u + {}u], {LDK}u);",
            16 * LDK,
            j * 16
        );
        let _ = writeln!(s, "    o{j} = coopMultiplyAdd(pa1, wf{j}, o{j});");
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "  }}");
    let _ = writeln!(s, "  smx[tid] = ls;");
    let _ = writeln!(s, "  workgroupBarrier();");
    let _ = writeln!(s, "  let lt = (ls + smx[tid ^ 16u]) + (smx[tid ^ 128u] + smx[tid ^ 144u]);");
    let _ = writeln!(s, "  let inv = select(0.0, 1.0 / lt, lt > 0.0);");
    let _ = writeln!(s, "  let obase = (tl * p.nh + g * p.hpk + hh) * 256u + dh * 128u + hf * 8u;");
    let _ = writeln!(s, "  let obs = sg * 256u + r * 16u + hf * 8u;");
    for j in 0..8 {
        let _ = writeln!(s, "  coopStoreT(o{j}, &ss[sg * 256u], 16u);");
        let _ = writeln!(s, "  workgroupBarrier();");
        let _ = writeln!(s, "  if (live) {{");
        for e in 0..8 {
            let _ = writeln!(s, "    ob[obase + {}u] = ss[obs + {e}u] * inv;", j * 16 + e);
        }
        let _ = writeln!(s, "  }}");
        let _ = writeln!(s, "  workgroupBarrier();");
    }
    let _ = writeln!(s, "}}");
    s
}

pub(crate) fn src() -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let thr = "8.0";
    let _ = writeln!(s, "enable f16;\nenable wgpu_cooperative_matrix;\ndiagnostic(off, derivative_uniformity);");
    let _ = writeln!(
        s,
        "struct FP {{ nh: u32, hpk: u32, cap: u32, pos0: u32, k: u32, kpad: u32, scl: f32, nkv: u32 }};"
    );
    let _ = writeln!(s, "@group(0) @binding(0) var<storage, read> qh: array<f16>;");
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> kq: array<vec4<f16>>;");
    let _ = writeln!(s, "@group(0) @binding(2) var<storage, read> vq: array<vec4<f16>>;");
    let _ = writeln!(s, "@group(0) @binding(3) var<storage, read_write> ob: array<f32>;");
    let _ = writeln!(s, "@group(0) @binding(4) var<uniform> p: FP;");
    // Every workgroup array a multiple of 16 bytes (`zimage::flash_src`:
    // an 8-byte array shifted the cooperative loads after it).
    let _ = writeln!(s, "var<workgroup> sk: array<f16, {}>;", BC * LDK);
    let _ = writeln!(s, "var<workgroup> sv: array<f16, {}>;", BC * LDK);
    let _ = writeln!(s, "var<workgroup> ss: array<f32, {}>;", 8 * 256);
    let _ = writeln!(s, "var<workgroup> sp: array<f16, {}>;", 8 * 16 * LDP);
    let _ = writeln!(s, "var<workgroup> smx: array<f32, 256>;");
    let _ = writeln!(s, "var<workgroup> sfl: array<u32, 4>;");
    let _ = writeln!(
        s,
        "@compute @workgroup_size(256)\nfn pf_flash(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32, @builtin(subgroup_id) sg: u32) {{"
    );
    let _ = writeln!(s, "  let g = wid.y;");
    let _ = writeln!(s, "  let tpt = 64u / p.hpk;");
    let _ = writeln!(s, "  let t0 = wid.x * tpt;");
    let _ = writeln!(s, "  let rb = sg & 3u; let dh = sg >> 2u;");
    let _ = writeln!(s, "  let lane = tid - sg * 32u;");
    let _ = writeln!(s, "  let r = lane & 15u; let hf = lane >> 4u;");
    let _ = writeln!(s, "  let rr = rb * 16u + r;");
    let _ = writeln!(s, "  let tl = t0 + rr / p.hpk;");
    let _ = writeln!(s, "  let hh = rr % p.hpk;");
    let _ = writeln!(s, "  let qpos = p.pos0 + tl;");
    let _ = writeln!(s, "  let live = tl < p.k;");
    let _ = writeln!(s, "  let qrow = (g * p.kpad + t0) * p.hpk + rb * 16u;");
    for d in 0..16 {
        let _ = writeln!(s, "  let qa{d} = coopLoadT<coop_mat16x16<f16, A>>(&qh[qrow * 256u + {}u], 256u);", d * 16);
    }
    for j in 0..8 {
        let _ = writeln!(s, "  var o{j}: coop_mat16x16<f32, C>;");
    }
    // Never assigned: the zero each block's S starts from.
    let _ = writeln!(s, "  var zc: coop_mat16x16<f32, C>;");
    let _ = writeln!(s, "  var mu = -1.0e30; var ls = 0.0;");
    let _ = writeln!(s, "  if (tid < 4u) {{ sfl[tid] = 0u; }}");
    // Keys the tile can see: up to its last live token's position.
    let _ = writeln!(s, "  let tlast = min(t0 + tpt, p.k) - 1u;");
    let _ = writeln!(s, "  let kend = p.pos0 + tlast + 1u;");
    let _ = writeln!(s, "  let nkb = (kend + {}u) / {BC}u;", BC - 1);
    let _ = writeln!(s, "  let kvb = g * p.cap * 64u;");
    // Prefetch registers: 16 keys × 64 vec4 = 1024 a tile, 4 a thread.
    for t in 0..4 {
        let _ = writeln!(
            s,
            "  var rk{t} = vec4<f16>(0.0h); var rv{t} = vec4<f16>(0.0h);\n  {{ let idx = tid + {o}u; let key = idx / 64u; if (key < kend) {{ rk{t} = kq[kvb + key * 64u + idx % 64u]; rv{t} = vq[kvb + key * 64u + idx % 64u]; }} }}",
            o = t * 256
        );
    }
    let _ = writeln!(s, "  for (var kb = 0u; kb < nkb; kb = kb + 1u) {{");
    for t in 0..4 {
        let _ = writeln!(
            s,
            "    {{ let idx = tid + {o}u; let d = (idx / 64u) * {LDK}u + (idx % 64u) * 4u; sk[d] = rk{t}.x; sk[d + 1u] = rk{t}.y; sk[d + 2u] = rk{t}.z; sk[d + 3u] = rk{t}.w; sv[d] = rv{t}.x; sv[d + 1u] = rv{t}.y; sv[d + 2u] = rv{t}.z; sv[d + 3u] = rv{t}.w; }}",
            o = t * 256
        );
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "    if (kb + 1u < nkb) {{");
    for t in 0..4 {
        let _ = writeln!(
            s,
            "      {{ let idx = tid + {o}u; let key = (kb + 1u) * {BC}u + idx / 64u; var a = vec4<f16>(0.0h); var b = vec4<f16>(0.0h); if (key < kend) {{ a = kq[kvb + key * 64u + idx % 64u]; b = vq[kvb + key * 64u + idx % 64u]; }} rk{t} = a; rv{t} = b; }}",
            o = t * 256
        );
    }
    let _ = writeln!(s, "    }}");
    // S = Q·Kᵀ over the 256 dims.
    let _ = writeln!(s, "    let kf0 = coopLoad<coop_mat16x16<f16, B>>(&sk[0u], {LDK}u);");
    let _ = writeln!(s, "    var s0 = coopMultiplyAdd(qa0, kf0, zc);");
    for d in 1..16 {
        let _ = writeln!(s, "    let kf{d} = coopLoad<coop_mat16x16<f16, B>>(&sk[{}u], {LDK}u);", d * 16);
        let _ = writeln!(s, "    s0 = coopMultiplyAdd(qa{d}, kf{d}, s0);");
    }
    let _ = writeln!(s, "    coopStoreT(s0, &ss[sg * 256u], 16u);");
    let _ = writeln!(s, "    workgroupBarrier();");
    // This lane's 8 keys of row r: scaled scores, masked past its position.
    let _ = writeln!(s, "    let sbase = sg * 256u + r * 16u + hf * 8u;");
    let _ = writeln!(s, "    let kbase = kb * {BC}u + hf * 8u;");
    let _ = writeln!(s, "    var pm = -1.0e30;");
    for e in 0..8 {
        let _ = writeln!(
            s,
            "    var sv{e} = -1.0e30; if (kbase + {e}u <= qpos && live) {{ sv{e} = ss[sbase + {e}u] * p.scl; }} pm = max(pm, sv{e});"
        );
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
    for j in 0..8 {
        let _ = writeln!(s, "        coopStoreT(o{j}, &ss[sg * 256u], 16u);");
        let _ = writeln!(s, "        workgroupBarrier();");
        for e in 0..8 {
            let _ = writeln!(s, "        ss[sbase + {e}u] = ss[sbase + {e}u] * alpha;");
        }
        let _ = writeln!(s, "        workgroupBarrier();");
        let _ = writeln!(s, "        o{j} = coopLoadT<coop_mat16x16<f32, C>>(&ss[sg * 256u], 16u);");
        let _ = writeln!(s, "        workgroupBarrier();");
    }
    let _ = writeln!(s, "      }}");
    let _ = writeln!(s, "    }}");
    // P = 2^(s − mu) → f16, row sums.
    let _ = writeln!(s, "    let pbase = sg * {}u + r * {LDP}u + hf * 8u;", 16 * LDP);
    for e in 0..8 {
        let _ = writeln!(s, "    let pv{e} = select(0.0, exp2(sv{e} - mu), sv{e} > -1.0e29);");
        let _ = writeln!(s, "    ls = ls + pv{e}; sp[pbase + {e}u] = f16(pv{e});");
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    // O += P·V over this subgroup's 128 columns.
    let _ = writeln!(s, "    let pa = coopLoadT<coop_mat16x16<f16, A>>(&sp[sg * {}u], {LDP}u);", 16 * LDP);
    for j in 0..8 {
        let _ = writeln!(
            s,
            "    let vf{j} = coopLoadT<coop_mat16x16<f16, B>>(&sv[dh * 128u + {}u], {LDK}u);",
            j * 16
        );
        let _ = writeln!(s, "    o{j} = coopMultiplyAdd(pa, vf{j}, o{j});");
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "  }}");
    // Row sums of the two half lanes, then O / l as f32 rows.
    let _ = writeln!(s, "  smx[tid] = ls;");
    let _ = writeln!(s, "  workgroupBarrier();");
    let _ = writeln!(s, "  let lt = ls + smx[tid ^ 16u];");
    let _ = writeln!(s, "  let inv = select(0.0, 1.0 / lt, lt > 0.0);");
    let _ = writeln!(s, "  let obase = (tl * p.nh + g * p.hpk + hh) * 256u + dh * 128u + hf * 8u;");
    let _ = writeln!(s, "  let sbase = sg * 256u + r * 16u + hf * 8u;");
    for j in 0..8 {
        let _ = writeln!(s, "  coopStoreT(o{j}, &ss[sg * 256u], 16u);");
        let _ = writeln!(s, "  workgroupBarrier();");
        let _ = writeln!(s, "  if (live) {{");
        for e in 0..8 {
            let _ = writeln!(s, "    ob[obase + {}u] = ss[sbase + {e}u] * inv;", j * 16 + e);
        }
        let _ = writeln!(s, "  }}");
        let _ = writeln!(s, "  workgroupBarrier();");
    }
    let _ = writeln!(s, "}}");
    s
}

pub(crate) type Pipe = (wgpu::ComputePipeline, wgpu::BindGroupLayout);

/// The pipeline, once per context; None without the matrix units or f16,
/// on rejection, or under `CMF_PF_FLASH=0`.
pub(crate) fn pipe(c: &Ctx) -> Option<&'static Pipe> {
    static P: std::sync::OnceLock<Option<Pipe>> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        if std::env::var("CMF_PF_FLASH").as_deref() == Ok("0") {
            return None;
        }
        zimage::zctx()?;
        let mut w = wide();
        if c.device.limits().max_compute_workgroup_storage_size < shared_bytes(w) {
            w = false;
        }
        if c.device.limits().max_compute_workgroup_storage_size < shared_bytes(w) {
            return None;
        }
        let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cmf-pf-flash"),
            source: wgpu::ShaderSource::Wgsl(if w { src32() } else { src() }.into()),
        });
        let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pf_flash"),
            layout: None,
            module: &module,
            entry_point: Some("pf_flash"),
            compilation_options: crate::gpu_wgpu::pco(),
            cache: c.pipeline_cache.as_ref(),
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            tracing::warn!("cmf-pf-flash rejected ({e}): the prompt keeps the split attend");
            return None;
        }
        let l = p.get_bind_group_layout(0);
        Some((p, l))
    })
    .as_ref()
}

#[cfg(test)]
mod tests {
    #[test]
    fn source_validates() {
        for s in [super::src(), super::src32()] {
            let module = wgpu::naga::front::wgsl::parse_str(&s).unwrap_or_else(|e| panic!("{e}"));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        }
        assert!(super::shared_bytes(true) <= 49152);
    }
}
