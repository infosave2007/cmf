//! SAM3's ViT trunk over Vulkan (a child of `ltx.rs`): the patch rows go up
//! once and the trunk features come back once. The patch embedding is a
//! `zi_mm` product over the rows padded to 608 columns; a layer is LayerNorm
//! into an f16 product input (`sm_ln`), the fused q|k|v product in f32, its
//! bias and axial rotation into a `[q | k | v]` f16 panel in window order
//! (`sm_pack`: the 24 x 24 windows as `vit_flash`'s batch of 576-row
//! sequences, the four global layers one sequence of 5184), `vit_flash` at
//! head width 64, the output product added back in raster order with its
//! bias (`sm_addrows`), then LayerNorm, the erf-GELU MLP (`sm_gelu`) and its
//! residual. The int8 weights stay on the card in their codec and pass
//! through an f16 plane (`zi_dq8`) before each product. `CMF_SAM3_WGPU=0`
//! keeps the host trunk.

use super::*;
use crate::sam3 as s3;

/// Columns of the padded patch rows (588 = 3 x 14 x 14, to the products' 32).
const KP: usize = 608;

const LN_SRC: &str = r#"
enable f16;
struct LP { w: u32, wo: u32, bo: u32, half: u32, eps: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@group(0) @binding(2) var<storage, read> cw: array<f32>;
@group(0) @binding(3) var<uniform> p: LP;
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

// LayerNorm of row wid.x of x with the weight and bias at wo / bo: into o
// (f16) under `half`, else back into x.
@compute @workgroup_size(256)
fn sm_ln(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let base = wid.x * p.w;
    var s = 0.0;
    for (var i = lid; i < p.w; i = i + 256u) { s = s + x[base + i]; }
    let mean = wsum(s, lid) / f32(p.w);
    var q = 0.0;
    for (var i = lid; i < p.w; i = i + 256u) {
        let d = x[base + i] - mean;
        q = q + d * d;
    }
    let inv = inverseSqrt(wsum(q, lid) / f32(p.w) + p.eps);
    for (var i = lid; i < p.w; i = i + 256u) {
        let v = (x[base + i] - mean) * inv * cw[p.wo + i] + cw[p.bo + i];
        if (p.half != 0u) { o[base + i] = f16(v); } else { x[base + i] = v; }
    }
}
"#;

const PACK_SRC: &str = r#"
enable f16;
struct PP { tokens: u32, win: u32, bo: u32, ro: u32, _a: u32, _b: u32, _c: u32, _d: u32 };
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> pan: array<f16>;
@group(0) @binding(2) var<storage, read> cw: array<f32>;
@group(0) @binding(3) var<uniform> p: PP;

// Token t of the 72 x 72 grid, pair j of head h: q|k|v rows [t][3072] plus
// the bias into the panel at the token's row in window order (its row in
// the 24 x 24 window under `win`), q and k rotated by the table's pair at
// that row.
@compute @workgroup_size(256)
fn sm_pack(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 256u) + gid.x;
    if (i >= p.tokens * 512u) { return; }
    let j = i % 32u;
    let h = (i / 32u) % 16u;
    let t = i / 512u;
    var row = t;
    var pr = t;
    if (p.win != 0u) {
        let yy = t / 72u;
        let xx = t - yy * 72u;
        let rr = (yy % 24u) * 24u + xx % 24u;
        row = ((yy / 24u) * 3u + xx / 24u) * 576u + rr;
        pr = rr;
    }
    let d0 = h * 64u + 2u * j;
    let src = t * 3072u;
    let c = cw[p.ro + (pr * 32u + j) * 2u];
    let s = cw[p.ro + (pr * 32u + j) * 2u + 1u];
    let dst = row * 3072u;
    let qa = y[src + d0] + cw[p.bo + d0];
    let qb = y[src + d0 + 1u] + cw[p.bo + d0 + 1u];
    pan[dst + d0] = f16(qa * c - qb * s);
    pan[dst + d0 + 1u] = f16(qb * c + qa * s);
    let ka = y[src + 1024u + d0] + cw[p.bo + 1024u + d0];
    let kb = y[src + 1024u + d0 + 1u] + cw[p.bo + 1024u + d0 + 1u];
    pan[dst + 1024u + d0] = f16(ka * c - kb * s);
    pan[dst + 1024u + d0 + 1u] = f16(kb * c + ka * s);
    pan[dst + 2048u + d0] = f16(y[src + 2048u + d0] + cw[p.bo + 2048u + d0]);
    pan[dst + 2048u + d0 + 1u] = f16(y[src + 2048u + d0 + 1u] + cw[p.bo + 2048u + d0 + 1u]);
}
"#;

const ADD_SRC: &str = r#"
struct AP { tokens: u32, w: u32, bo: u32, win: u32, _a: u32, _b: u32, _c: u32, _d: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> cw: array<f32>;
@group(0) @binding(3) var<uniform> p: AP;

// x[t] += y[row of t] (+ the bias at bo) over the stream in raster order; y's
// rows are in window order under `win`.
@compute @workgroup_size(256)
fn sm_addrows(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 256u) + gid.x;
    if (i >= p.tokens * p.w) { return; }
    let e = i % p.w;
    let t = i / p.w;
    var row = t;
    if (p.win != 0u) {
        let yy = t / 72u;
        let xx = t - yy * 72u;
        row = ((yy / 24u) * 3u + xx / 24u) * 576u + (yy % 24u) * 24u + xx % 24u;
    }
    var v = y[row * p.w + e];
    if (p.bo != 0xFFFFFFFFu) { v = v + cw[p.bo + e]; }
    x[i] = x[i] + v;
}
"#;

const GELU_SRC: &str = r#"
enable f16;
struct GP { n: u32, w: u32, bo: u32, _a: u32, _b: u32, _c: u32, _d: u32, _e: u32 };
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> act: array<f16>;
@group(0) @binding(2) var<storage, read> cw: array<f32>;
@group(0) @binding(3) var<uniform> p: GP;

// erf (Abramowitz-Stegun 7.1.26, |error| < 1.5e-7)
fn erf1(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return sign(x) * (1.0 - poly * exp(-a * a));
}

@compute @workgroup_size(256)
fn sm_gelu(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 256u) + gid.x;
    if (i >= p.n) { return; }
    let v = y[i] + cw[p.bo + i % p.w];
    act[i] = f16(clamp(0.5 * v * (1.0 + erf1(v * 0.70710678118)), -65504.0, 65504.0));
}
"#;

/// The trunk features `[5184][1024]` of the patch rows `pv` (`[5184][588]`),
/// or `None` (said once) for the host trunk.
pub(crate) fn sam3_trunk(model: &Arc<CmfModel>, pv: &[f32], pos_skip_cls: bool) -> Option<Vec<f32>> {
    if !ltx_enabled() || std::env::var("CMF_SAM3_WGPU").as_deref() == Ok("0") {
        return None;
    }
    let (td, ti, th, grid) = (s3::TD, s3::TI, s3::TH, s3::GRID);
    let np = grid * grid;
    let kc = 3 * s3::PATCH * s3::PATCH;
    if pv.len() != np * kc || td != 1024 || th != 16 || !ti.is_multiple_of(128) || kc > KP {
        not_mine("a SAM3 trunk outside the chain's shapes");
        return None;
    }
    let c = zi::zctx()?;
    let vec_of = |n: &str| s3::load_vec(model, n).ok();
    const MPER: usize = 4;
    let mut spans = Vec::with_capacity(s3::TL * MPER);
    let mut q8s: Vec<Q8> = Vec::with_capacity(s3::TL * MPER);
    let mut plane = 0usize;
    let mut cw = Cw::default();
    // per layer: norm1 w, b, qkv bias, rope, proj bias, norm2 w, b, fc1 bias, fc2 bias
    let mut lay: Vec<[u32; 9]> = Vec::with_capacity(s3::TL);
    for l in 0..s3::TL {
        let b = format!("trunk.blocks.{l}.");
        for (n, rows, cols) in [("attn.qkv.weight", 3 * td, td), ("attn.proj.weight", td, td), ("mlp.fc1.weight", ti, td), ("mlp.fc2.weight", td, ti)] {
            let ix = model.tensor_index(&format!("{b}{n}"))?;
            if let Some(sp) = span(model, ix, rows, cols) {
                spans.push(sp);
                q8s.push(None);
            } else if let Some((sp, q8)) = span8(model, ix, rows, cols) {
                spans.push(sp);
                q8s.push(q8);
                plane = plane.max(rows * cols);
            } else {
                not_mine("a SAM3 trunk weight that is neither q4tp nor int8");
                return None;
            }
        }
        let rows = if s3::GLOBAL.contains(&l) { np } else { s3::WIN * s3::WIN };
        let rope = vec_of(&format!("{b}attn.freqs_cis"))?;
        if rope.len() != rows * 64 {
            not_mine("a SAM3 rotation table of the wrong size");
            return None;
        }
        let ro = cw.put(&rope);
        let mut put = |n: &str| -> Option<u32> { Some(cw.put(&vec_of(&format!("{b}{n}"))?)) };
        lay.push([
            put("norm1.weight")?,
            put("norm1.bias")?,
            put("attn.qkv.bias")?,
            ro,
            put("attn.proj.bias")?,
            put("norm2.weight")?,
            put("norm2.bias")?,
            put("mlp.fc1.bias")?,
            put("mlp.fc2.bias")?,
        ]);
    }
    let lnp = [cw.put(&vec_of("trunk.ln_pre.weight")?), cw.put(&vec_of("trunk.ln_pre.bias")?)];
    // the patch weight [1024][588] as an f16 plane padded to KP columns
    let pw = vec_of("trunk.patch_embed.proj.weight")?;
    if pw.len() != td * kc {
        not_mine("a SAM3 patch embedding of the wrong size");
        return None;
    }
    let mut pw16 = vec![0u16; td * KP];
    for r in 0..td {
        for k in 0..kc {
            pw16[r * KP + k] = cortiq_core::quant::f32_to_f16(pw[r * kc + k]);
        }
    }
    // the position table [1, 577, 1024] (Meta's cls row first) tiled 24 x 24 -> 72 x 72
    let pt = vec_of("trunk.pos_embed")?;
    let base = if pos_skip_cls { 1 } else { 0 };
    if pt.len() < (base + 576) * td {
        return None;
    }
    let mut pos = vec![0f32; np * td];
    for p in 0..np {
        let (py, px) = (p / grid, p % grid);
        let src = (base + (py % 24) * 24 + px % 24) * td;
        pos[p * td..(p + 1) * td].copy_from_slice(&pt[src..src + td]);
    }
    let rp = zi::pad_rows(np);
    let mut pv16 = vec![0u16; rp * KP];
    for p in 0..np {
        for k in 0..kc {
            pv16[p * KP + k] = cortiq_core::quant::f32_to_f16(pv[p * kc + k]);
        }
    }
    let lim = c.device.limits();
    if (cw.0.len() * 4) as u64 > lim.max_storage_buffer_binding_size || (2 * plane) as u64 > lim.max_storage_buffer_binding_size {
        decline("a SAM3 constant table or plane passes the largest storage binding");
        return None;
    }

    let _op = c.op_gate.enter();
    let _gate = c.mm_gate.lock().unwrap();
    let ld = 3 * td;
    let acts: u64 = (4 * np * td + 2 * rp * ti + 4 * rp * ti + 2 * rp * ld + 2 * rp * td + 2 * plane + 4 * np * td + 2 * rp * KP + 2 * td * KP) as u64
        + (cw.0.len() * 4) as u64;
    let plan = match plan(c, model, &spans, MPER, acts) {
        Ok(p) => p,
        Err(_) => return None,
    };
    let sc_oom = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    // rows past the grid catch the products' padded tiles
    let xb = zi::sbuf(c, (rp * td * 4) as u64, "sm_x");
    let xn = zi::sbuf(c, (rp * ti * 2) as u64, "sm_xn");
    let y = zi::sbuf(c, (rp * ti * 4) as u64, "sm_y");
    let pan = zi::sbuf(c, (rp * ld * 2) as u64, "sm_panel");
    let att = zi::sbuf(c, (rp * td * 2) as u64, "sm_att");
    let slot = zi::sbuf(c, (2 * plane).max(16) as u64, "sm_plane");
    let cwb = zi::sbuf_init(c, bytemuck::cast_slice(&cw.0), "sm_cw");
    let posb = zi::sbuf_init(c, bytemuck::cast_slice(&pos), "sm_pos");
    let pvb = zi::sbuf_init(c, bytemuck::cast_slice(&pv16), "sm_pv");
    let pwb = zi::sbuf_init(c, bytemuck::cast_slice(&pw16), "sm_pw");
    if let Some(e) = pollster::block_on(sc_oom.pop()) {
        eprintln!("sam3 wgpu: {e}");
        decline("the SAM3 trunk's working set did not fit the card");
        return None;
    }
    let t = tile();
    let (f16c, f32c) = (site_cfg(t, Epi::F16), site_cfg(t, Epi::F32));
    let f32p = MmCfg::new(t.0, t.1, t.2, t.3, t.4, Epi::F32);
    let _ = f16c;
    let fc = flash_cfg();
    let fkey = format!("vit_flash_{}_{}_64_{}", fc.nw, fc.bc, zi::flash_thr());
    let pipes = || -> Option<_> {
        Some((
            zi::pipeline(c, "zi_dq8", zi::DQ8_SRC, "zi_dq8")?,
            zi::pipeline(c, "sm_ln", LN_SRC, "sm_ln")?,
            zi::pipeline(c, "sm_pack", PACK_SRC, "sm_pack")?,
            zi::pipeline(c, &fkey, &vit_flash_src(fc, 64)?, "vit_flash")?,
            zi::pipeline(c, "sm_addrows", ADD_SRC, "sm_addrows")?,
            zi::pipeline(c, "sm_gelu", GELU_SRC, "sm_gelu")?,
        ))
    };
    let Some((pdq8, pln, ppack, pfl, padd, pgelu)) = pipes() else {
        decline("a kernel of the SAM3 trunk was rejected by the device");
        return None;
    };
    // tensor `ti` of the stack (its buffer `w`) through a product into f32
    let prod = |calls: &mut ZCalls, cls: Class, tix: usize, nn: usize, k: usize, w: &wgpu::Buffer, act: &wgpu::Buffer, out: &wgpu::Buffer| -> Option<()> {
        let args = margs(np, nn, k, nn, 0, 1.0);
        match q8s[tix] {
            None => calls.push_mm(cls, mm(c, f32c, &args, 0, w, act, out)?),
            Some((rs, cs, has_col)) => {
                let u = zi::ubuf(c, &[k as u32, nn as u32, has_col as u32, 0, 0, rs, cs, 0]);
                let bg = zi::bg(c, &pdq8, &[w, &slot, &u]);
                calls.push(Class::Rows, Call { pipe: pdq8.clone(), bg, grid: grid1(nn * k / 4) });
                calls.push_mm(cls, mm(c, f32p, &args, 0, &slot, act, out)?);
            }
        }
        Some(())
    };
    let ln = |calls: &mut ZCalls, wo: u32, bo: u32, half: bool| {
        let u = zi::ubuf(c, &[td as u32, wo, bo, half as u32, s3::T_EPS.to_bits(), 0, 0, 0]);
        let bg = zi::bg(c, &pln, &[&xb, &xn, &cwb, &u]);
        calls.push(Class::Rows, Call { pipe: pln.clone(), bg, grid: (np as u32, 1, 1) });
    };
    let addrows = |calls: &mut ZCalls, src: &wgpu::Buffer, bo: u32, win: bool| {
        let u = zi::ubuf(c, &[np as u32, td as u32, bo, win as u32, 0, 0, 0, 0]);
        let bg = zi::bg(c, &padd, &[&xb, src, &cwb, &u]);
        calls.push(Class::Rows, Call { pipe: padd.clone(), bg, grid: grid1(np * td) });
    };
    let build = |layers: std::ops::Range<usize>, wb: &[wgpu::Buffer], first: bool| -> Option<ZCalls> {
        let mut calls = ZCalls::default();
        if first {
            // x = pv · patchᵀ + positions, then ln_pre in place
            let args = margs(np, td, KP, td, 0, 1.0);
            calls.push_mm(Class::MmO, mm(c, f32p, &args, 0, &pwb, &pvb, &xb)?);
            addrows(&mut calls, &posb, NONE, false);
            ln(&mut calls, lnp[0], lnp[1], false);
        }
        for (j, l) in layers.clone().enumerate() {
            let (o, w) = (&lay[l], &wb[j * MPER..(j + 1) * MPER]);
            let tix = l * MPER;
            let global = s3::GLOBAL.contains(&l);
            ln(&mut calls, o[0], o[1], true);
            prod(&mut calls, Class::MmQkv, tix, ld, td, &w[0], &xn, &y)?;
            {
                let u = zi::ubuf(c, &[np as u32, (!global) as u32, o[2], o[3], 0, 0, 0, 0]);
                let bg = zi::bg(c, &ppack, &[&y, &pan, &cwb, &u]);
                calls.push(Class::Rows, Call { pipe: ppack.clone(), bg, grid: grid1(np * 512) });
            }
            {
                let (nseq, groups) = if global { (np, 1u32) } else { (s3::WIN * s3::WIN, 9u32) };
                let scl = std::f32::consts::LOG2_E / 8.0;
                let u = zi::ubuf(c, &[nseq as u32, nseq as u32, ld as u32, td as u32, 2 * td as u32, td as u32, scl.to_bits(), 1.0f32.to_bits()]);
                let bg = zi::bg(c, &pfl, &[&pan, &pan, &att, &u]);
                calls.push(Class::Flash, Call { pipe: pfl.clone(), bg, grid: ((nseq as u32).div_ceil(fc.nw * 16), th as u32, groups) });
            }
            prod(&mut calls, Class::MmO, tix + 1, td, td, &w[1], &att, &y)?;
            addrows(&mut calls, &y, o[4], !global);
            ln(&mut calls, o[5], o[6], true);
            prod(&mut calls, Class::MmW13, tix + 2, ti, td, &w[2], &xn, &y)?;
            {
                let u = zi::ubuf(c, &[(np * ti) as u32, ti as u32, o[7], 0, 0, 0, 0, 0]);
                let bg = zi::bg(c, &pgelu, &[&y, &xn, &cwb, &u]);
                calls.push(Class::Rows, Call { pipe: pgelu.clone(), bg, grid: grid1(np * ti) });
            }
            prod(&mut calls, Class::MmW2, tix + 3, td, ti, &w[3], &xn, &y)?;
            addrows(&mut calls, &y, o[8], false);
        }
        Some(calls)
    };

    let uid = model.uid() as usize;
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut submitted = false;
    let mut first = true;
    for (kept, layers) in plan.runs() {
        let mine = &spans[layers.start * MPER..layers.end * MPER];
        if !kept {
            if submitted {
                zi::wait(c);
            }
            let stale: Vec<(usize, usize)> = spans
                .iter()
                .enumerate()
                .filter(|(i, _)| !plan.keep.contains(&(i / MPER)) && !layers.contains(&(i / MPER)))
                .map(|(_, (idx, _))| (uid, *idx))
                .collect();
            super::super::super::weight_evict(c, &stale);
        }
        let wb = match fetch(c, model, mine) {
            Ok(wb) => wb,
            Err(_) => {
                let _ = pollster::block_on(sc.pop());
                return None;
            }
        };
        let mut start = layers.start;
        while start < layers.end {
            let end = (start + 8).min(layers.end);
            let Some(calls) = build(start..end, &wb[(start - layers.start) * MPER..(end - layers.start) * MPER], first) else {
                let _ = pollster::block_on(sc.pop());
                decline("a kernel of the SAM3 trunk was rejected by the device");
                return None;
            };
            first = false;
            let mut enc = c.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                calls.record(&mut pass, None);
            }
            c.queue.submit(Some(enc.finish()));
            submitted = true;
            start = end;
        }
    }
    let rx = zi::read_bytes(c, &xb, (np * td * 4) as u64);
    if let Some(e) = pollster::block_on(sc.pop()) {
        decline(&format!("the SAM3 trunk's command buffer failed: {e}"));
        return None;
    }
    let rx = rx?;
    let out: Vec<f32> = bytemuck::cast_slice(&rx).to_vec();
    if out.iter().any(|v| !v.is_finite()) {
        decline("the SAM3 trunk produced a value that is not finite");
        return None;
    }
    Some(out)
}
