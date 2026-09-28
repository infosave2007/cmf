//! Qwen-Image-2.1 denoiser on wgpu/Vulkan.
//!
//! Implements the `gpu::qi21_*` contract (the same one `gpu_metal/qi21.rs`
//! serves): [`prefill`] runs the prompt prefix (text and condition-image
//! rows, t = 0 modulation, block-causal mask) through every block once
//! and keeps each layer's post-RoPE keys and values on the device;
//! [`step`] runs the target rows against `[cached prefix, own rows]` —
//! the pipeline's KV cache.
//!
//! Built from the Z-Image chain (`zimage.rs`), which is a close relative:
//! - weights are f16 planes built once per model by
//!   [`ZBlockDev::from_model_all`] (q8_2f/q8_row streamed through `zi_dq8`,
//!   q4tp through the parent's `q4tp_dq_f16`, F16/Bf16/F32 converted; a
//!   block may mix codecs) — 13.96 GB for the 32 blocks;
//! - the four GEMM sites run `zi_mm` (tensor cores, f32 accumulation):
//!   qkv with the f16 epilogue, O and down with the f32 epilogue, gate/up
//!   on the interleaved plane with the fused SwiGLU epilogue
//!   ([`Epi::SwiGluIn`]: the FFN input's range guard is undone before the
//!   nonlinearity);
//! - `zi_qkrope` (per-head RMSNorm·w, complex-interleaved RoPE, in place on
//!   the f16 qkv panel), `zi_embed` (img_in, no bias) and `zi_final`
//!   (LN·fs → proj_out, no bias) as they are.
//!
//! What differs from Z-Image and lives here:
//! - `qi_rowop`: the gated residual `x += tanh(g)·y` (no post-norm) and the
//!   next modulated affine-free LayerNorm `LN(x)·(1+s)` → f16, one pass
//!   over the row; the one shared modulation is uploaded once per step
//!   with its two gate chunks already `tanh`-ed on the host (the CPU
//!   path's own f32 `tanh`);
//! - `qi_flash`: flash attention whose keys come from two places — the
//!   per-layer prefix cache `[lp][2H]` (K then V) for keys `< n_pre`, the
//!   qkv panel for the rest — with an optional visibility mask (query `i`
//!   sees keys `[0, vis[i])`, non-decreasing: the prefill's block-causal
//!   mask; a workgroup walks only the key blocks its last live query
//!   sees) and an output multiplier (the attention-output range guard);
//! - `qi_kvcopy`: the prefill's K|V panel columns → the layer's cache.
//!
//! Range guards (the Metal module's sites and defaults): the attention
//! input ×2⁻², the qkv panel ×2⁻⁴ (the qk-norm is scale-free, v carries it),
//! the attention output ×2⁻⁴, the FFN input ×2⁻², the SwiGLU hidden ×2⁻⁸;
//! each consumer multiplies its factor back in an f32 epilogue.
//!
//! Knobs: `CMF_QI21_WGPU=0` (device path off), `CMF_QI21_WGPU_PROF=1`
//! (per-class device time after every step: each class alone, 3 reps),
//! `CMF_QI21_AMAX=1` (largest |value| at each f16 site per pass),
//! `CMF_QI21_FLASH=nw,bc` (flash tile), `CMF_QI21_{ATTN,QKV,AO,FFN,HID}_SHIFT`
//! (the guards), and the Z-Image GEMM knobs (`CMF_ZI_TILE`).

use crate::gpu::{Qi21Geom, Qi21PrefillArgs, ZBlockRef};
use cortiq_core::CmfModel;
use std::sync::{Arc, Mutex};

use super::zimage::{self as zi, Call, Class, Epi, FlashCfg, MmArgs, MmCall, MmCfg, ZBlockDev, ZCalls, ZDims};
use super::Ctx;

fn enabled() -> bool {
    std::env::var("CMF_QI21_WGPU").as_deref() != Ok("0")
}

fn decline(reason: &str) -> bool {
    static SAID: Mutex<Vec<String>> = Mutex::new(Vec::new());
    if let Ok(mut v) = SAID.lock() {
        if !v.iter().any(|r| r == reason) {
            eprintln!("qwen-image-2.1: wgpu device path declined: {reason}");
            v.push(reason.to_string());
        }
    }
    false
}

fn prof_on() -> bool {
    std::env::var("CMF_QI21_PROF").is_ok_and(|v| v != "0")
}

/// Rows one dispatch may address (`zi_embed` / `zi_final` / the row op
/// index rows by `workgroup_id.x`).
const MAX_ROWS: usize = 65_535;

/// Programs kept at once (the positive and the negative prompt).
const MAX_PROGS: usize = 2;

// ───────────────────────────── guards ─────────────────────────────

/// Power-of-two guards of the f16 sites (stored value = x·2^-s).
#[derive(Clone, Copy, Debug)]
struct Guards {
    attn: i32,
    qkv: i32,
    ao: i32,
    ffn: i32,
    hid: i32,
}

impl Guards {
    fn from_env() -> Guards {
        let e = |k: &str, d: i32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d).clamp(0, 14);
        Guards {
            attn: e("CMF_QI21_ATTN_SHIFT", 2),
            qkv: e("CMF_QI21_QKV_SHIFT", 4),
            ao: e("CMF_QI21_AO_SHIFT", 4),
            ffn: e("CMF_QI21_FFN_SHIFT", 2),
            hid: e("CMF_QI21_HID_SHIFT", 8),
        }
    }
}

fn p2(e: i32) -> f32 {
    (2.0f32).powi(e)
}

// ───────────────────────────── kernels ─────────────────────────────

/// mode bit 0: `x += g ⊙ y` (g = the tanh-ed gate chunk of `mods`);
/// mode bit 1: `xn = f16(LN(x)·(1 + s)·oscale)` (affine-free LayerNorm,
/// biased variance, two passes in f32). One workgroup per row, h ≤ 4096.
const ROWOP_SRC: &str = r#"
struct RP { h: u32, mode: u32, g_off: u32, s_off: u32, eps: f32, oscale: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> xn: array<u32>;
@group(0) @binding(3) var<storage, read> mods: array<f32>;
@group(0) @binding(4) var<uniform> rp: RP;
var<workgroup> red: array<f32, 256>;

fn wsum(v: f32, lid: u32) -> f32 {
    red[lid] = v;
    workgroupBarrier();
    var st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn qi_rowop(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let np = rp.h / 2u;
    let base = row * rp.h;
    var xv: array<vec2<f32>, 8>;
    for (var j = 0u; j < 8u; j = j + 1u) {
        let pi = lid + j * 256u;
        if (pi < np) {
            let c = base + 2u * pi;
            var v = vec2<f32>(x[c], x[c + 1u]);
            if ((rp.mode & 1u) != 0u) {
                let g = vec2<f32>(mods[rp.g_off + 2u * pi], mods[rp.g_off + 2u * pi + 1u]);
                v = v + g * vec2<f32>(y[c], y[c + 1u]);
                x[c] = v.x;
                x[c + 1u] = v.y;
            }
            xv[j] = v;
        }
    }
    if ((rp.mode & 2u) == 0u) { return; }
    var sm = 0.0;
    for (var j = 0u; j < 8u; j = j + 1u) {
        if (lid + j * 256u < np) { sm = sm + xv[j].x + xv[j].y; }
    }
    let mean = wsum(sm, lid) / f32(rp.h);
    var q = 0.0;
    for (var j = 0u; j < 8u; j = j + 1u) {
        if (lid + j * 256u < np) { let d = xv[j] - vec2<f32>(mean); q = q + d.x * d.x + d.y * d.y; }
    }
    let inv = inverseSqrt(wsum(q, lid) / f32(rp.h) + rp.eps);
    for (var j = 0u; j < 8u; j = j + 1u) {
        let pi = lid + j * 256u;
        if (pi < np) {
            let c = 2u * pi;
            let s = vec2<f32>(1.0) + vec2<f32>(mods[rp.s_off + c], mods[rp.s_off + c + 1u]);
            xn[row * np + pi] = pack2x16float((xv[j] - vec2<f32>(mean)) * inv * s * rp.oscale);
        }
    }
}
"#;

/// The K|V columns of the prefill's qkv panel → the layer's cache
/// `[rows][2H]`, 8 halves per thread.
const KVCOPY_SRC: &str = r#"
struct KP { ldp8: u32, ldkv8: u32, off8: u32, n: u32 };
@group(0) @binding(0) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> kp: KP;
@compute @workgroup_size(256)
fn qi_kvcopy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let c = gid.x;
    if (row >= kp.n || c >= kp.ldkv8) { return; }
    dst[row * kp.ldkv8 + c] = src[row * kp.ldp8 + kp.off8 + c];
}
"#;

/// The flash tile (`nw` subgroups × 16 queries, `bc` keys per block).
/// Per-step flash time on the RTX PRO 4000 at 1024² (4096 queries, 4114
/// keys, 32 layers): 8×16 149 ms, 4×32 165, 16×16 184, 8×32 185, 4×16
/// (Z-Image's tile) 213, 2×16 284; at 512² 8×16 is also the fastest.
/// `CMF_QI21_FLASH=nw,bc` overrides.
fn flash_cfg() -> FlashCfg {
    if let Ok(s) = std::env::var("CMF_QI21_FLASH") {
        let v: Vec<u32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if v.len() == 2 && flash_src(FlashCfg { nw: v[0], bc: v[1] }).is_some() {
            return FlashCfg { nw: v[0], bc: v[1] };
        }
    }
    FlashCfg { nw: 8, bc: 16 }
}

fn flash_key(f: FlashCfg) -> String {
    format!("qi_flash_{}_{}_{}", f.nw, f.bc, zi::flash_thr())
}

/// Workgroup-shared bytes of `qi_flash`.
fn flash_shared(f: FlashCfg) -> u32 {
    2 * f.bc * 136 * 2 + f.nw * 16 * f.bc * 4 + f.nw * 16 * (f.bc + 8) * 2 + f.nw * 32 * 4 + 16
}

/// WGSL of `qi_flash` (hd 128). The schedule is `zi_flash`'s: Q·Kᵀ and P·V
/// on the matrix units in f16, the online softmax in f32 with the lazy,
/// workgroup-uniform rescale (exact; P ≤ 2^thr in f16). Differences: keys
/// j < `n_pre` come from the prefix cache (binding 2, `[j][ldkv]` with V
/// at `kv_v`), the rest from the panel rows j − n_pre; key j is visible
/// to query i iff j < lim(i) (`vis[i]` under `use_vis`, else all keys),
/// masked scores give P = 0 exactly; the lane's scores are kept in
/// registers, so the O rescale may reuse the S scratch; the output is
/// multiplied by `oscale`. Bindings: 0 panel (f16), 1 panel (vec4<f16>),
/// 2 prefix cache (vec4<f16>), 3 output (vec4<u32>), 4 vis, 5 `FP`.
fn flash_src(f: FlashCfg) -> Option<String> {
    use std::fmt::Write;
    let (nw, bc) = (f.nw, f.bc);
    if nw == 0 || bc == 0 || bc % 16 != 0 || nw * 32 > 1024 {
        return None;
    }
    let nt = nw * 32;
    let ldk = 136u32;
    let ldp = bc + 8;
    let nkf = bc / 16;
    let vk = bc * 32; // vec4<f16> per K (or V) tile: bc rows × 32
    if vk % nt != 0 || 8 % nkf != 0 || (bc / 2) % 8 != 0 {
        return None;
    }
    let per = vk / nt;
    let half = bc / 2;
    let qw = nw * 16;
    let rounds = 8 / nkf;
    let thr = zi::flash_thr();
    let mut s = String::new();
    let _ = writeln!(s, "enable f16;\nenable wgpu_cooperative_matrix;\ndiagnostic(off, derivative_uniformity);");
    let _ = writeln!(
        s,
        "struct FP {{ nq: u32, n_pre: u32, n_own: u32, ld: u32, k_col: u32, v_col: u32, o_ld: u32, scl: f32, use_vis: u32, ldkv: u32, kv_v: u32, oscale: f32 }};"
    );
    let _ = writeln!(s, "@group(0) @binding(0) var<storage, read> qh: array<f16>;");
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> qv: array<vec4<f16>>;");
    let _ = writeln!(s, "@group(0) @binding(2) var<storage, read> kv: array<vec4<f16>>;");
    let _ = writeln!(s, "@group(0) @binding(3) var<storage, read_write> oh: array<vec4<u32>>;");
    let _ = writeln!(s, "@group(0) @binding(4) var<storage, read> vis: array<u32>;");
    let _ = writeln!(s, "@group(0) @binding(5) var<uniform> p: FP;");
    // Every workgroup array is a multiple of 16 bytes (see the zimage.rs
    // trap: cooperative loads ignore the low address bits).
    let _ = writeln!(s, "var<workgroup> sk: array<f16, {}>;", bc * ldk);
    let _ = writeln!(s, "var<workgroup> sv: array<f16, {}>;", bc * ldk);
    let _ = writeln!(s, "var<workgroup> ss: array<f32, {}>;", nw * 16 * bc);
    let _ = writeln!(s, "var<workgroup> sp: array<f16, {}>;", nw * 16 * ldp);
    let _ = writeln!(s, "var<workgroup> smx: array<f32, {nt}>;");
    let _ = writeln!(s, "var<workgroup> sfl: array<u32, 4>;");
    let _ = writeln!(
        s,
        "@compute @workgroup_size({nt})\nfn qi_flash(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32, @builtin(subgroup_id) sg: u32) {{"
    );
    let _ = writeln!(s, "  let h = wid.y;");
    let _ = writeln!(s, "  let q0 = wid.x * {qw}u + sg * 16u;");
    let _ = writeln!(s, "  let lane = tid - sg * 32u;");
    let _ = writeln!(s, "  let r = lane & 15u; let hf = lane >> 4u;");
    let _ = writeln!(s, "  let ldq = p.ld / 4u; let ldkv4 = p.ldkv / 4u; let kvv4 = p.kv_v / 4u;");
    let _ = writeln!(s, "  let kq4 = (p.k_col + h * 128u) / 4u; let vq4 = (p.v_col + h * 128u) / 4u;");
    let _ = writeln!(s, "  let ntot = p.n_pre + p.n_own;");
    for d in 0..8 {
        let _ = writeln!(s, "  let qa{d} = coopLoadT<coop_mat16x16<f16, A>>(&qh[q0 * p.ld + h * 128u + {}u], p.ld);", d * 16);
    }
    for j in 0..8 {
        let _ = writeln!(s, "  var o{j}: coop_mat16x16<f32, C>;");
    }
    // Never assigned: the zero every block's S accumulation starts from.
    let _ = writeln!(s, "  var zc: coop_mat16x16<f32, C>;");
    let _ = writeln!(s, "  var mu = -1.0e30; var ls = 0.0;");
    let _ = writeln!(s, "  let qi = q0 + r;");
    let _ = writeln!(s, "  let live = qi < p.nq;");
    let _ = writeln!(s, "  var lim = ntot;");
    let _ = writeln!(s, "  if (p.use_vis != 0u) {{ lim = min(ntot, vis[min(qi, p.nq - 1u)]); }}");
    let _ = writeln!(s, "  if (tid < 2u) {{ sfl[tid] = 0u; }}");
    // the key blocks this workgroup walks: its last live query's visibility
    let _ = writeln!(
        s,
        "  if (tid == 0u) {{ var kl = ntot; if (p.use_vis != 0u) {{ kl = min(kl, vis[min(wid.x * {qw}u + {qw}u, p.nq) - 1u]); }} sfl[2] = (kl + {}u) / {bc}u; }}",
        bc - 1
    );
    let _ = writeln!(s, "  let nkb = workgroupUniformLoad(&sfl[2]);");
    let load = |s: &mut String, kbx: &str, ind: &str| {
        for t in 0..per {
            let _ = writeln!(
                s,
                "{ind}{{ let idx = tid + {o}u; let j = min({kbx} * {bc}u + idx / 32u, ntot - 1u); let c4 = idx % 32u;",
                o = t * nt
            );
            let _ = writeln!(s, "{ind}  if (j < p.n_pre) {{ let b = j * ldkv4 + h * 32u + c4; rk{t} = kv[b]; rv{t} = kv[b + kvv4]; }}");
            let _ = writeln!(
                s,
                "{ind}  else {{ let row = j - p.n_pre; rk{t} = qv[row * ldq + kq4 + c4]; rv{t} = qv[row * ldq + vq4 + c4]; }} }}"
            );
        }
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
    let _ = writeln!(s, "    if (kb + 1u < nkb) {{");
    load(&mut s, "(kb + 1u)", "      ");
    let _ = writeln!(s, "    }}");
    // S = Q Kᵀ
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
    // this lane's scores (scaled, masked) into registers
    let _ = writeln!(s, "    let sbase = sg * {}u + r * {bc}u + hf * {half}u;", 16 * bc);
    let _ = writeln!(s, "    let kbase = kb * {bc}u + hf * {half}u;");
    for e in 0..half {
        let _ = writeln!(s, "    let x{e} = select(-1.0e30, ss[sbase + {e}u] * p.scl, kbase + {e}u < lim);");
    }
    let _ = writeln!(s, "    var pm = x0;");
    for e in 1..half {
        let _ = writeln!(s, "    pm = max(pm, x{e});");
    }
    let _ = writeln!(s, "    smx[tid] = pm;");
    // Only live rows vote for the workgroup-wide rescale (zimage.rs, vk2).
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
    // P = 2^(s − mu) → f16 (masked keys exactly 0), row sums
    let _ = writeln!(s, "    let pbase = sg * {}u + r * {ldp}u + hf * {half}u;", 16 * ldp);
    for e in 0..half {
        let _ = writeln!(s, "    let pv{e} = select(0.0, exp2(x{e} - mu), kbase + {e}u < lim);");
    }
    for e in 0..half {
        let _ = writeln!(s, "    ls = ls + pv{e}; sp[pbase + {e}u] = f16(pv{e});");
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    // O += P V
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
    // finalize: O / l · oscale → f16
    let _ = writeln!(s, "  smx[tid] = ls;");
    let _ = writeln!(s, "  workgroupBarrier();");
    let _ = writeln!(s, "  let lt = ls + smx[tid ^ 16u];");
    let _ = writeln!(s, "  let inv = select(0.0, p.oscale / lt, lt > 0.0);");
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
                "    {{ let b = sbase + {}u; oh[(qi * p.o_ld + h * 128u + {}u + hf * {half}u + {}u) / 8u] = vec4<u32>(pack2x16float(vec2<f32>(ss[b], ss[b + 1u]) * inv), pack2x16float(vec2<f32>(ss[b + 2u], ss[b + 3u]) * inv), pack2x16float(vec2<f32>(ss[b + 4u], ss[b + 5u]) * inv), pack2x16float(vec2<f32>(ss[b + 6u], ss[b + 7u]) * inv)); }}",
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

// ───────────────────────────── state ─────────────────────────────

/// Activations of one pass (prefill or step), `mp = pad_rows(rows)` rows
/// each so the unchecked kernels' tile over-reads stay inside.
struct Acts {
    rows: usize,
    x: wgpu::Buffer,     // f32 [mp][H]
    xn: wgpu::Buffer,    // f16 [mp][H]
    panel: wgpu::Buffer, // f16 [mp][3H]
    att: wgpu::Buffer,   // f16 [mp][H]
    y: wgpu::Buffer,     // f32 [mp][H]
    hid: wgpu::Buffer,   // f16 [mp][inter]
}

fn acts_bytes(rows: usize, g: &Qi21Geom) -> u64 {
    (zi::pad_rows(rows) * (18 * g.hidden + 2 * g.inter)) as u64
}

impl Acts {
    fn new(c: &Ctx, rows: usize, g: &Qi21Geom) -> Acts {
        let mp = zi::pad_rows(rows) as u64;
        let (h, i) = (g.hidden as u64, g.inter as u64);
        Acts {
            rows,
            x: zi::sbuf(c, mp * h * 4, "qi_x"),
            xn: zi::sbuf(c, mp * h * 2, "qi_xn"),
            panel: zi::sbuf(c, mp * 3 * h * 2, "qi_panel"),
            att: zi::sbuf(c, mp * h * 2, "qi_att"),
            y: zi::sbuf(c, mp * h * 4, "qi_y"),
            hid: zi::sbuf(c, mp * i * 2, "qi_hid"),
        }
    }
}

/// One prompt's step program: its prefix cache and the prebuilt dispatches
/// of a whole denoiser call (a step writes tok/mods/fs and replays them).
struct Prog {
    key: u64,
    lp: usize,
    n: usize,
    acts: Arc<Acts>,
    /// per layer `[lp][2H]` f16: post-RoPE K, then V (×2^-qkv); read
    /// through the step's bind groups
    _pkv: Vec<wgpu::Buffer>,
    tok: wgpu::Buffer,
    mods: wgpu::Buffer,
    fs: wgpu::Buffer,
    out: wgpu::Buffer,
    calls: ZCalls,
    amax: Option<wgpu::Buffer>,
    _rope_t: (wgpu::Buffer, wgpu::Buffer),
}

fn pkv_bytes(lp: usize, g: &Qi21Geom, nb: usize) -> u64 {
    (nb * lp * 2 * g.hidden * 2) as u64
}

struct Dev {
    uid: u64,
    _model: Arc<CmfModel>,
    geom: Qi21Geom,
    guards: Guards,
    blocks: Vec<ZBlockDev>,
    /// img_in `[H][64]` / proj_out `[64][H]`, f32
    emb_w: wgpu::Buffer,
    fin_w: wgpu::Buffer,
    /// zeros `[H]`: the bias (and pad row) inputs of `zi_embed`/`zi_final`
    zeros: wgpu::Buffer,
    /// the activations a new pass reuses when they are large enough
    acts: Option<Arc<Acts>>,
    progs: Vec<Prog>,
}

impl Dev {
    fn acts_for(&mut self, c: &Ctx, rows: usize) -> Arc<Acts> {
        if let Some(a) = self.acts.as_ref().filter(|a| a.rows >= rows) {
            return a.clone();
        }
        self.acts = None; // released here unless a live program holds it
        let a = Arc::new(Acts::new(c, rows, &self.geom));
        self.acts = Some(a.clone());
        a
    }

    /// Device bytes held beyond the planes (activations, prefix caches).
    fn extra_bytes(&self) -> u64 {
        let nb = self.blocks.len();
        let mut acts: Vec<*const Acts> = Vec::new();
        let mut total = 0u64;
        for a in self.progs.iter().map(|p| &p.acts).chain(self.acts.iter()) {
            if !acts.contains(&Arc::as_ptr(a)) {
                acts.push(Arc::as_ptr(a));
                total += acts_bytes(a.rows, &self.geom);
            }
        }
        total + self.progs.iter().map(|p| pkv_bytes(p.lp, &self.geom, nb)).sum::<u64>()
    }
}

static STATE: Mutex<Option<Dev>> = Mutex::new(None);

fn geom_ok(g: &Qi21Geom) -> Result<(), String> {
    if g.hd != 128 || g.nh * g.hd != g.hidden || g.hidden % 128 != 0 || g.hidden > 4096 {
        return Err(format!("geometry {g:?} (the kernels need hd 128, hidden % 128 == 0, ≤ 4096)"));
    }
    if g.inter % 64 != 0 || g.in_ch != 64 {
        return Err(format!("geometry {g:?} (inter % 64, 64 latent channels)"));
    }
    Ok(())
}

fn dims(g: &Qi21Geom) -> ZDims {
    ZDims {
        h: g.hidden,
        nh: g.nh,
        inter: g.inter,
        eps: g.eps,
        final_eps: g.eps,
        pd: 64,
    }
}

/// `[s1 | tanh g1 | s2 | tanh g2]` — the gate chunks with the CPU path's
/// own f32 `tanh`.
fn mods_dev(mods: &[f32], h: usize) -> Vec<f32> {
    mods[..4 * h]
        .iter()
        .enumerate()
        .map(|(i, &v)| if (h..2 * h).contains(&i) || i >= 3 * h { v.tanh() } else { v })
        .collect()
}

// ───────────────────────────── dispatches ─────────────────────────────

/// `zi_mm` with the uniform's last word set (`Epi::SwiGluIn`'s input
/// multiplier; 0 for the other epilogues, as `zimage.rs` writes it).
fn mm(c: &Ctx, g: MmCfg, a: &MmArgs, extra: u32, plane: &wgpu::Buffer, act: &wgpu::Buffer, out: &wgpu::Buffer) -> Option<MmCall> {
    if a.n % g.bn != 0 || a.k % g.bk != 0 {
        return None;
    }
    let pipe = zi::mm_pipe(c, g)?;
    let u = zi::ubuf(c, &[a.m, a.n, a.k, a.ldo, a.ocol, a.arow, a.oscale.to_bits(), extra]);
    let bg = zi::bg(c, &pipe, &[plane, act, out, &u]);
    Some(MmCall {
        pipe,
        bg,
        grid: (a.n / g.bn, a.m.div_ceil(g.bm)),
    })
}

fn margs(m: usize, n: usize, k: usize, ldo: usize, oscale: f32) -> MmArgs {
    MmArgs {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ldo: ldo as u32,
        ocol: 0,
        arow: 0,
        oscale,
        conv: [0; 3],
    }
}

/// The gate/up GEMM tile: 256×128×32 with 4×2 subgroups (`zimage_gemmbench
/// mm` on the RTX PRO 4000, SwiGLU epilogue: 60 TF at M 4224 against 54
/// for the 128×128 default; the other sites keep the default, 65–76 TF).
/// `CMF_QI21_TILE_W13=bm,bn,bk,wm,wn` overrides.
fn w13_cfg() -> MmCfg {
    if let Ok(s) = std::env::var("CMF_QI21_TILE_W13") {
        let v: Vec<u32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if v.len() == 5 {
            let c = MmCfg::new(v[0], v[1], v[2], v[3], v[4], Epi::SwiGluIn);
            if c.valid() {
                return c;
            }
        }
    }
    MmCfg::new(256, 128, 32, 4, 2, Epi::SwiGluIn)
}

const ROW_GATE: u32 = 1;
const ROW_NORM: u32 = 2;

#[allow(clippy::too_many_arguments)]
fn rowop(c: &Ctx, g: &Qi21Geom, a: &Acts, mods: &wgpu::Buffer, rows: usize, mode: u32, g_off: usize, s_off: usize, oscale: f32) -> Option<Call> {
    let pipe = zi::pipeline(c, "qi_rowop", ROWOP_SRC, "qi_rowop")?;
    let u = zi::ubuf(
        c,
        &[g.hidden as u32, mode, g_off as u32, s_off as u32, g.eps.to_bits(), oscale.to_bits(), 0, 0],
    );
    let bg = zi::bg(c, &pipe, &[&a.y, &a.x, &a.xn, mods, &u]);
    Some(Call { pipe, bg, grid: (rows as u32, 1, 1) })
}

/// One pass over the blocks (prefill or step) on `rows` rows of `a`.
struct Pass<'b> {
    rows: usize,
    mods: &'b wgpu::Buffer,
    rope: (&'b wgpu::Buffer, &'b wgpu::Buffer),
    /// prefill: the visibility buffer (and the caches are written);
    /// step: None (the caches are read, `lp` keys each)
    vis: Option<&'b wgpu::Buffer>,
    pkv: &'b [wgpu::Buffer],
    lp: usize,
    /// dummy binding for `vis` in a step
    dummy: &'b wgpu::Buffer,
    amax: Option<&'b wgpu::Buffer>,
}

/// Probe sites of `CMF_QI21_AMAX=1` (the Metal module's order).
const AMAX_SITES: [&str; 5] = ["attn-in", "qkv", "attn-out", "ffn-in", "hidden"];

/// The dispatches of every block over the pass. `xn` must already hold
/// block 0's modulated LayerNorm. A prefill stops after the last layer's
/// cache write (its attention and FFN would feed nothing).
fn blocks(c: &Ctx, d: &Dev, a: &Acts, s: &Pass, calls: &mut ZCalls) -> Option<()> {
    let g = d.geom;
    let gd = d.guards;
    let (h, inter, nh) = (g.hidden, g.inter, g.nh);
    let n = s.rows;
    let t = (zi::default_cfg(Epi::F16), zi::default_cfg(Epi::F32), w13_cfg());
    let fc = flash_cfg();
    if flash_shared(fc) > c.device.limits().max_compute_workgroup_storage_size {
        return None;
    }
    let fpipe = zi::pipeline(c, &flash_key(fc), &flash_src(fc)?, "qi_flash")?;
    let qkpipe = zi::pipeline(c, "zi_qkrope", zi::QKROPE_SRC, "zi_qkrope")?;
    let kvpipe = zi::pipeline(c, "qi_kvcopy", KVCOPY_SRC, "qi_kvcopy")?;
    let prefill = s.vis.is_some();
    let probe = |calls: &mut ZCalls, buf: &wgpu::Buffer, halves: usize, site: u32| -> Option<()> {
        if let Some(ab) = s.amax {
            calls.push(Class::Io, zi::amax_call(c, buf, halves / 2, false, ab, site)?);
        }
        Some(())
    };
    let nb = d.blocks.len();
    for (l, blk) in d.blocks.iter().enumerate() {
        let last = l + 1 == nb;
        probe(calls, &a.xn, n * h, 0)?;
        // qkv → panel (f16, ×2^-qkv)
        let aq = margs(n, 3 * h, h, 3 * h, p2(gd.attn - gd.qkv));
        calls.push_mm(Class::MmQkv, mm(c, t.0, &aq, 0, &blk.qkv, &a.xn, &a.panel)?);
        probe(calls, &a.panel, n * 3 * h, 1)?;
        {
            // q/k arrive ×2^-qkv: eps ×2^-2qkv keeps the RMSNorm exact
            let eps = g.eps * p2(-2 * gd.qkv);
            let u = zi::ubuf(c, &[3 * h as u32, nh as u32, eps.to_bits(), 0]);
            let b = zi::bg(c, &qkpipe, &[&a.panel, &blk.norm_q, &blk.norm_k, s.rope.0, s.rope.1, &u]);
            calls.push(Class::Rows, Call { pipe: qkpipe.clone(), bg: b, grid: (2 * nh as u32, n as u32, 1) });
        }
        if prefill {
            let u = zi::ubuf(c, &[(3 * h / 8) as u32, (2 * h / 8) as u32, (h / 8) as u32, n as u32]);
            let b = zi::bg(c, &kvpipe, &[&a.panel, &s.pkv[l], &u]);
            calls.push(Class::Rows, Call { pipe: kvpipe.clone(), bg: b, grid: ((2 * h / 8).div_ceil(256) as u32, n as u32, 1) });
            if last {
                break;
            }
        }
        {
            let (n_pre, kv) = if prefill { (0, &a.panel) } else { (s.lp, &s.pkv[l]) };
            let scl = std::f32::consts::LOG2_E / (g.hd as f32).sqrt();
            let u = zi::ubuf(
                c,
                &[
                    n as u32,
                    n_pre as u32,
                    n as u32,
                    (3 * h) as u32,
                    h as u32,
                    (2 * h) as u32,
                    h as u32,
                    scl.to_bits(),
                    prefill as u32,
                    (2 * h) as u32,
                    h as u32,
                    p2(gd.qkv - gd.ao).to_bits(),
                ],
            );
            let b = zi::bg(c, &fpipe, &[&a.panel, &a.panel, kv, &a.att, s.vis.unwrap_or(s.dummy), &u]);
            calls.push(
                Class::Flash,
                Call { pipe: fpipe.clone(), bg: b, grid: ((n as u32).div_ceil(fc.nw * 16), nh as u32, 1) },
            );
        }
        probe(calls, &a.att, n * h, 2)?;
        let ao = margs(n, h, h, h, p2(gd.ao));
        calls.push_mm(Class::MmO, mm(c, t.1, &ao, 0, &blk.o, &a.att, &a.y)?);
        calls.push(Class::Rows, rowop(c, &g, a, s.mods, n, ROW_GATE | ROW_NORM, h, 2 * h, p2(-gd.ffn))?);
        probe(calls, &a.xn, n * h, 3)?;
        let a13 = margs(n, 2 * inter, h, inter, p2(-gd.hid));
        calls.push_mm(Class::MmW13, mm(c, t.2, &a13, p2(gd.ffn).to_bits(), &blk.w13, &a.xn, &a.hid)?);
        probe(calls, &a.hid, n * inter, 4)?;
        let a2 = margs(n, h, inter, h, p2(gd.hid));
        calls.push_mm(Class::MmW2, mm(c, t.1, &a2, 0, &blk.w2, &a.hid, &a.y)?);
        if last {
            calls.push(Class::Rows, rowop(c, &g, a, s.mods, n, ROW_GATE, 3 * h, 0, 1.0)?);
        } else {
            calls.push(Class::Rows, rowop(c, &g, a, s.mods, n, ROW_GATE | ROW_NORM, 3 * h, 0, p2(-gd.attn))?);
        }
    }
    Some(())
}

/// Run a recorded list once (submit + wait) under a validation scope.
fn run_calls(c: &Ctx, calls: &ZCalls) -> bool {
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut enc = c.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        calls.record(&mut pass, None);
    }
    c.queue.submit(Some(enc.finish()));
    zi::wait(c);
    match pollster::block_on(sc.pop()) {
        Some(e) => {
            eprintln!("qwen-image-2.1 wgpu: {e}");
            false
        }
        None => true,
    }
}

fn read_amax(c: &Ctx, b: &wgpu::Buffer, what: &str, gd: Guards) {
    let Some(raw) = zi::read_bytes(c, b, 16 * 4) else { return };
    c.queue.write_buffer(b, 0, &[0u8; 64]);
    let v: &[f32] = bytemuck::cast_slice(&raw);
    let sh = [gd.attn, gd.qkv, gd.ao, gd.ffn, gd.hid];
    let s: Vec<String> = AMAX_SITES
        .iter()
        .enumerate()
        .map(|(i, n)| format!("{n} {:.1} (×2^{} = {:.3e})", v[i], sh[i], v[i] * p2(sh[i])))
        .collect();
    eprintln!("qi21 wgpu {what} amax (stored): {}", s.join(" · "));
}

/// Allocate under an out-of-memory scope: `None` when the device refused.
fn alloc_scoped<T>(c: &Ctx, f: impl FnOnce() -> Option<T>) -> Option<T> {
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let v = f();
    match pollster::block_on(sc.pop()) {
        Some(e) => {
            eprintln!("qwen-image-2.1 wgpu: {e}");
            None
        }
        None => v,
    }
}

fn build_dev(c: &Ctx, a: &Qi21PrefillArgs) -> Option<Dev> {
    let g = a.geom;
    let h = g.hidden;
    let ones = vec![1f32; h];
    let zrefs: Vec<ZBlockRef> = a
        .blocks
        .iter()
        .map(|b| ZBlockRef {
            wq: b.w[0],
            wk: b.w[1],
            wv: b.w[2],
            wo: b.w[3],
            w1: b.w[4],
            w3: b.w[5],
            w2: b.w[6],
            norm1: &ones,
            norm2: &ones,
            ffn_norm1: &ones,
            ffn_norm2: &ones,
            norm_q: b.norm_q,
            norm_k: b.norm_k,
        })
        .collect();
    let refs: Vec<&ZBlockRef> = zrefs.iter().collect();
    let blocks = ZBlockDev::from_model_all(a.model, &dims(&g), &refs)?;
    Some(Dev {
        uid: a.model.uid(),
        _model: a.model.clone(),
        geom: g,
        guards: Guards::from_env(),
        blocks,
        emb_w: zi::sbuf_init(c, bytemuck::cast_slice(&a.img_in[..h * 64]), "qi_emb"),
        fin_w: zi::sbuf_init(c, bytemuck::cast_slice(&a.proj_out[..64 * h]), "qi_fin"),
        zeros: zi::sbuf(c, (h * 4) as u64, "qi_zeros"),
        acts: None,
        progs: Vec::new(),
    })
}

/// The step program of a prepared prefix (`pkv` filled by the prefill).
fn build_prog(c: &Ctx, d: &Dev, a: &Qi21PrefillArgs, acts: Arc<Acts>, pkv: Vec<wgpu::Buffer>) -> Option<Prog> {
    let g = d.geom;
    let h = g.hidden;
    let n = a.n;
    let tok = zi::sbuf(c, (n * 64 * 4) as u64, "qi_tok");
    let mods = zi::sbuf(c, (4 * h * 4) as u64, "qi_mods");
    let fs = zi::sbuf(c, (h * 4) as u64, "qi_fs");
    let out = zi::sbuf(c, (n * 64 * 4) as u64, "qi_out");
    let rope_t = (
        zi::sbuf_init(c, bytemuck::cast_slice(&a.rope_t.0[..n * 64]), "qi_rc"),
        zi::sbuf_init(c, bytemuck::cast_slice(&a.rope_t.1[..n * 64]), "qi_rs"),
    );
    let amax = (std::env::var("CMF_QI21_AMAX").as_deref() == Ok("1")).then(|| zi::sbuf(c, 64, "qi_amax"));
    let mut calls = ZCalls::default();
    {
        let pipe = zi::pipeline(c, "zi_embed", zi::EMBED_SRC, "zi_embed")?;
        let u = zi::ubuf(c, &[h as u32, n as u32, 1u32 << 30, 64]);
        let b = zi::bg(c, &pipe, &[&tok, &d.emb_w, &d.zeros, &d.zeros, &acts.x, &u]);
        calls.push(Class::Io, Call { pipe, bg: b, grid: (n as u32, 1, 1) });
    }
    calls.push(Class::Rows, rowop(c, &g, &acts, &mods, n, ROW_NORM, 0, 0, p2(-d.guards.attn))?);
    blocks(
        c,
        d,
        &acts,
        &Pass {
            rows: n,
            mods: &mods,
            rope: (&rope_t.0, &rope_t.1),
            vis: None,
            pkv: &pkv,
            lp: a.lp,
            dummy: &d.zeros,
            amax: amax.as_ref(),
        },
        &mut calls,
    )?;
    {
        let pipe = zi::pipeline(c, "zi_final", zi::FINAL_SRC, "zi_final")?;
        let u = zi::ubuf(c, &[h as u32, 64, g.eps.to_bits(), n as u32, 1u32 << 30, 0, 0, 0]);
        let b = zi::bg(c, &pipe, &[&acts.x, &fs, &d.fin_w, &d.zeros, &out, &u]);
        calls.push(Class::Io, Call { pipe, bg: b, grid: (n as u32, 1, 1) });
    }
    Some(Prog {
        key: a.key,
        lp: a.lp,
        n,
        acts,
        _pkv: pkv,
        tok,
        mods,
        fs,
        out,
        calls,
        amax,
        _rope_t: rope_t,
    })
}

// ───────────────────────────── contract ─────────────────────────────

/// Build the program for `a.key` (and, once per model, the weight planes)
/// and run its prefix through every block, keeping each layer's keys and
/// values.
pub(crate) fn prefill(a: &Qi21PrefillArgs) -> bool {
    if !enabled() {
        return false;
    }
    let g = a.geom;
    if let Err(e) = geom_ok(&g) {
        return decline(&e);
    }
    let h = g.hidden;
    let (lp, n) = (a.lp, a.n);
    if lp == 0
        || n == 0
        || a.blocks.is_empty()
        || a.x.len() != lp * h
        || a.rope_p.0.len() != lp * 64
        || a.rope_p.1.len() != lp * 64
        || a.rope_t.0.len() != n * 64
        || a.rope_t.1.len() != n * 64
        || a.vis.len() != lp
        || a.mods0.len() != 4 * h
        || a.img_in.len() != h * 64
        || a.proj_out.len() != 64 * h
        || a.blocks.iter().any(|b| b.norm_q.len() != 128 || b.norm_k.len() != 128)
    {
        return decline("prefill args do not match the lengths");
    }
    // the mask the kernel walks: a leading run, non-decreasing, ≥ 1 key
    if a.vis.iter().zip(a.vis.iter().skip(1)).any(|(p, q)| q < p) || a.vis.iter().any(|&v| v == 0 || v as usize > lp) {
        return decline("the visibility mask is not a non-decreasing leading run");
    }
    if lp.max(n) > MAX_ROWS {
        return decline(&format!("{} rows exceed the device path's limit of {MAX_ROWS}", lp.max(n)));
    }
    let Some(c) = zi::zctx() else {
        return decline("no adapter with 16×16 f16 cooperative matrices, f16 shaders and 32-wide subgroups");
    };
    let Ok(mut st) = STATE.lock() else { return false };
    if st
        .as_ref()
        .is_some_and(|d| d.uid != a.model.uid() || d.geom != g || d.blocks.len() != a.blocks.len())
    {
        *st = None; // another model: free its planes first
    }
    if let Some(d) = st.as_mut() {
        d.progs.retain(|p| p.key != a.key);
        while d.progs.len() >= MAX_PROGS {
            d.progs.remove(0);
        }
    }
    // Budget: the planes (once), the activations, the prefix caches,
    // against the adapter's weight budget minus what the residency cache
    // holds; wgpu's out-of-memory error would otherwise panic the thread.
    let nb = a.blocks.len();
    let need = |st: &Option<Dev>| -> u64 {
        let planes = zi::plane_bytes(&dims(&g), nb);
        let rows = lp.max(n);
        let small = 64 << 20;
        match st {
            None => planes + acts_bytes(rows, &g) + pkv_bytes(lp, &g, nb) + small,
            Some(d) => {
                let new_acts = if d.acts.as_ref().is_some_and(|x| x.rows >= rows) { 0 } else { acts_bytes(rows, &g) };
                planes + d.extra_bytes() + new_acts + pkv_bytes(lp, &g, nb) + small
            }
        }
    };
    let budget = super::device_vram_budget();
    if budget > 0 && need(&*st) + super::resident_bytes() > budget {
        // The text encoder's resident weights (this container) are done.
        super::release_idle_model_buffers(a.model.uid());
        let held = super::resident_bytes();
        if need(&*st) + held > budget {
            return decline(&format!(
                "the denoiser needs about {:.1} GB on the device (+{:.1} GB held by other weights) and the adapter's budget is {:.1} GB",
                need(&*st) as f64 / 1e9,
                held as f64 / 1e9,
                budget as f64 / 1e9
            ));
        }
    }
    if st.is_none() {
        let t0 = std::time::Instant::now();
        match alloc_scoped(c, || build_dev(c, a)) {
            Some(d) => *st = Some(d),
            None => return decline("a weight codec the plane builder cannot expand, or the planes did not fit"),
        }
        if prof_on() {
            eprintln!("qi21 wgpu: planes {} blocks {:.2}s", nb, t0.elapsed().as_secs_f64());
        }
    }
    let d = st.as_mut().unwrap();
    let t0 = std::time::Instant::now();
    let Some((acts, pkv)) = alloc_scoped(c, || {
        let acts = d.acts_for(c, lp.max(n));
        let pkv: Vec<wgpu::Buffer> = (0..nb).map(|_| zi::sbuf(c, (lp * 2 * h * 2) as u64, "qi_pkv")).collect();
        Some((acts, pkv))
    }) else {
        d.acts = None;
        return decline("the activations or the prefix cache did not fit");
    };
    // the prefix pass
    let d_ref: &Dev = d;
    let ok = {
        let mods0 = zi::sbuf_init(c, bytemuck::cast_slice(&mods_dev(a.mods0, h)), "qi_mods0");
        let rope_p = (
            zi::sbuf_init(c, bytemuck::cast_slice(a.rope_p.0), "qi_rpc"),
            zi::sbuf_init(c, bytemuck::cast_slice(a.rope_p.1), "qi_rps"),
        );
        let vis = zi::sbuf_init(c, bytemuck::cast_slice(a.vis), "qi_vis");
        let amax = (std::env::var("CMF_QI21_AMAX").as_deref() == Ok("1")).then(|| zi::sbuf(c, 64, "qi_amax"));
        c.queue.write_buffer(&acts.x, 0, bytemuck::cast_slice(a.x));
        let mut calls = ZCalls::default();
        let built = rowop(c, &g, &acts, &mods0, lp, ROW_NORM, 0, 0, p2(-d_ref.guards.attn)).and_then(|r| {
            calls.push(Class::Rows, r);
            blocks(
                c,
                d_ref,
                &acts,
                &Pass {
                    rows: lp,
                    mods: &mods0,
                    rope: (&rope_p.0, &rope_p.1),
                    vis: Some(&vis),
                    pkv: &pkv,
                    lp,
                    dummy: &d_ref.zeros,
                    amax: amax.as_ref(),
                },
                &mut calls,
            )
        });
        let ok = built.is_some() && run_calls(c, &calls);
        if let (true, Some(b)) = (ok, amax.as_ref()) {
            read_amax(c, b, "prefill", d_ref.guards);
        }
        ok
    };
    if !ok {
        return decline("the prefill program could not be built or run");
    }
    let t_pre = t0.elapsed().as_secs_f64();
    let Some(prog) = build_prog(c, d_ref, a, acts, pkv) else {
        return decline("the step program could not be built");
    };
    // the chain's pipelines were compiled at first use: keep them for the
    // next process (once per process)
    static FLUSHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !FLUSHED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        super::pipeline_cache_flush();
    }
    if prof_on() {
        eprintln!(
            "qi21 wgpu: prefill {lp} rows {:.3}s, step program {:.3}s ({} dispatches)",
            t_pre,
            t0.elapsed().as_secs_f64() - t_pre,
            prog.calls.len()
        );
    }
    d.progs.push(prog);
    true
}

/// One denoiser call for the prepared `key`: latent tokens `[n, 64]` →
/// velocity `[n, 64]`. `mods` = the step's `[s1|g1|s2|g2]`, `fs` = 1 + the
/// final norm's scale.
pub(crate) fn step(key: u64, xtok: &[f32], mods: &[f32], fs: &[f32], out: &mut [f32]) -> bool {
    if !enabled() {
        return false;
    }
    let Some(c) = zi::zctx() else { return false };
    let Ok(st) = STATE.lock() else { return false };
    let Some(d) = st.as_ref() else { return false };
    let Some(p) = d.progs.iter().find(|p| p.key == key) else { return false };
    let h = d.geom.hidden;
    let n = p.n;
    if xtok.len() != n * 64 || out.len() != n * 64 || mods.len() != 4 * h || fs.len() != h {
        return decline("step args do not match the program");
    }
    c.queue.write_buffer(&p.tok, 0, bytemuck::cast_slice(xtok));
    c.queue.write_buffer(&p.mods, 0, bytemuck::cast_slice(&mods_dev(mods, h)));
    c.queue.write_buffer(&p.fs, 0, bytemuck::cast_slice(fs));
    if !run_calls(c, &p.calls) {
        return decline("a step command buffer failed");
    }
    let Some(raw) = zi::read_bytes(c, &p.out, (n * 64 * 4) as u64) else {
        return decline("the step readback failed");
    };
    let v: &[f32] = bytemuck::cast_slice(&raw);
    if let Some(b) = p.amax.as_ref() {
        read_amax(c, b, "step", d.guards);
    }
    if std::env::var("CMF_QI21_WGPU_PROF").as_deref() == Ok("1") {
        class_times(c, p);
    }
    if v.iter().any(|x| !x.is_finite()) {
        return decline("the device step produced a non-finite velocity (an f16 range guard overflowed?)");
    }
    out.copy_from_slice(v);
    true
}

/// Per-class device time of one step: each class recorded alone, 3 reps,
/// one submit (the results are meaningless; the next step recomputes
/// everything from `tok`, and the prefix caches are only read).
fn class_times(c: &Ctx, p: &Prog) {
    let mut line = format!("qi21 wgpu step classes ({} rows, prefix {}):", p.n, p.lp);
    let mut tot = 0.0;
    for cl in Class::ALL {
        let t = std::time::Instant::now();
        let mut enc = c.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for _ in 0..3 {
                p.calls.record(&mut pass, Some(cl));
            }
        }
        c.queue.submit(Some(enc.finish()));
        zi::wait(c);
        let ms = t.elapsed().as_secs_f64() * 1e3 / 3.0;
        tot += ms;
        line += &format!(" {cl:?} {ms:.1}");
    }
    eprintln!("{line} · sum {tot:.1} ms");
}

/// Drop one program (a prompt is done).
pub(crate) fn release_key(key: u64) {
    if let Ok(mut st) = STATE.lock() {
        if let Some(d) = st.as_mut() {
            d.progs.retain(|q| q.key != key);
        }
    }
}

/// Drop everything (planes, programs). Touches module-local state only:
/// never brings a device up.
pub(crate) fn release() {
    if let Ok(mut st) = STATE.lock() {
        *st = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_variants_generate() {
        for (nw, bc) in [(4, 16), (4, 32), (8, 16), (2, 16)] {
            let f = FlashCfg { nw, bc };
            let s = flash_src(f).expect("flash source");
            assert!(s.contains("fn qi_flash"));
            assert!(flash_shared(f) < 48 * 1024);
        }
        assert!(flash_src(FlashCfg { nw: 4, bc: 24 }).is_none());
    }

    #[test]
    fn gate_chunks_are_tanh_ed() {
        let h = 3;
        let m: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let d = mods_dev(&m, h);
        assert_eq!(&d[..3], &m[..3]);
        assert_eq!(d[4], (0.4f32).tanh());
        assert_eq!(&d[6..9], &m[6..9]);
        assert_eq!(d[11], (1.1f32).tanh());
    }
}
