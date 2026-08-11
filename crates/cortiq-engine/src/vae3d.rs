//! MiniMax-H3's video VAE decoder: a ViT3D, not a conv stack.
//!
//! 36 transformer blocks over the latent grid, each latent cell one
//! token, and a single linear that expands every token into a
//! 4×16×16×3 block of pixels. The encoder half is a 3-D causal CNN and
//! is not packed — text-to-video never runs it.
//!
//! ## Tiling is not an optimization here
//!
//! The reference decodes in 256-pixel spatial tiles and 17-frame
//! temporal clips ALWAYS — `tiling=True` is the constructor default and
//! `decode_tiled` just forwards to `decode`. Because the decoder is
//! global attention, a tile sees a different context than the whole
//! frame would, so the tiling is part of the model's output, not a
//! memory strategy layered on top of it. Both schedules are reproduced
//! exactly: `split_tiles` down to the last overlap unit, and the
//! clip/token-drop bookkeeping that makes a chunk emit 17 frames and
//! carry 5 over.

use std::sync::atomic::{AtomicU64, Ordering};

/// Where a VAE decode actually goes. The DiT had a profiler for months
/// while this stage — the LARGER half of a render at low step counts,
/// 37.8 s against the denoiser's 19.2 — had none, so every hour of
/// tuning went to the half that was already instrumented.
pub static VAE3D_PROF: [AtomicU64; 8] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];

pub(crate) fn vae3d_prof_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_VAE3D_PROF").is_ok())
}

fn vprof(slot: usize, t: std::time::Instant) {
    if vae3d_prof_on() {
        VAE3D_PROF[slot].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

/// One line per phase, sorted by cost.
pub fn vae3d_prof_report() -> Option<String> {
    if !vae3d_prof_on() {
        return None;
    }
    const NAMES: [&str; 8] = [
        "norm rows", "qkv gemm", "repack+rope", "attention", "out gemm", "ffn",
        "head+proj", "pixel shuffle",
    ];
    let mut v: Vec<(u64, &str)> = VAE3D_PROF
        .iter()
        .map(|a| a.load(Ordering::Relaxed))
        .zip(NAMES)
        .collect();
    let total: u64 = v.iter().map(|(u, _)| u).sum();
    if total == 0 {
        return None;
    }
    v.sort_by(|a, b| b.0.cmp(&a.0));
    let mut out = format!("vae3d phases (total {:.1} s):\n", total as f64 / 1e6);
    for (us, name) in v {
        out.push_str(&format!(
            "  {name:<16} {:>6.1} s  {:>5.1}%\n",
            us as f64 / 1e6,
            100.0 * us as f64 / total as f64
        ));
    }
    Some(out)
}

use crate::dit::Proj;
use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Pixel tile edge and the smallest overlap `split_tiles` will accept.
const TILE: usize = 256;
const TILE_OVERLAP_MIN: usize = 64;
/// Temporal clip in frames, and the tokens dropped off the encoder's
/// tail that the decoder must therefore re-manufacture.
const CLIP_LENGTH: usize = 17;
const TOKEN_DROP: usize = 3;

struct Block {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    scale1: Vec<f32>,
    scale2: Vec<f32>,
    qkv: Proj, // [3·dim, dim], PER HEAD interleaved
    qkv_b: Vec<f32>,
    out: Proj,
    out_b: Vec<f32>,
    w1: Proj, // [2·4·dim, dim]
    w1_b: Vec<f32>,
    w2: Proj, // [dim, 4·dim]
    w2_b: Vec<f32>,
}

pub struct VideoVae {
    post_quant: Proj,
    post_quant_b: Vec<f32>,
    x_embed: Proj,
    x_embed_b: Vec<f32>,
    registers: Vec<f32>, // [n_reg, dim]
    blocks: Vec<Block>,
    norm_out_w: Vec<f32>,
    norm_out_b: Vec<f32>,
    proj_out: Proj,
    proj_out_b: Vec<f32>,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    pool: Option<Arc<Pool>>,
    dim: usize,
    heads: usize,
    head_dim: usize,
    z_channels: usize,
    patch: usize,
    patch_t: usize,
    n_reg: usize,
    rope_theta: f32,
    rope_dim: usize,
    eps: f64,
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f64, dst: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for (((d, &v), &g), &bb) in dst.iter_mut().zip(x).zip(w).zip(b) {
        *d = ((v as f64 - mean) * inv) as f32 * g + bb;
    }
}

/// RMSNorm with a weight, no bias.
fn rms_norm_into(x: &[f32], w: &[f32], eps: f64, dst: &mut [f32]) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for ((d, &v), &g) in dst.iter_mut().zip(x).zip(w) {
        *d = (v as f64 * inv) as f32 * g;
    }
}

/// RMSNorm with NO affine — the decoder's q/k norms are weightless.
thread_local! {
    /// The check's reference arm re-enters `attention`; without this it
    /// checks its own check, forever.
    static CHECKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn rms_norm_plain(x: &mut [f32], eps: f64) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for v in x.iter_mut() {
        *v = (*v as f64 * inv) as f32;
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Rows across the pool, or straight through when there is none. The
/// DiT has had this since the start; this decoder was still walking its
/// norms and residuals on one thread — 3.3 s of a 38 s stage.
fn rows_par(pool: Option<&crate::pool::Pool>, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(pl) => pl.run_rows(n, f),
        None => f(0, n),
    }
}

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// SAFETY: caller guarantees disjoint `[off, off+len)` per worker.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

fn softmax_inplace(row: &mut [f32]) {
    let mx = row.iter().cloned().fold(f32::MIN, f32::max);
    let mut den = 0f32;
    for r in row.iter_mut() {
        *r = (*r - mx).exp();
        den += *r;
    }
    if den > 0.0 {
        let inv = 1.0 / den;
        for r in row.iter_mut() {
            *r *= inv;
        }
    }
}

/// `(starts, lens, overlaps)` for one axis — the reference's schedule,
/// which grows the overlaps rather than the tile count so every tile is
/// exactly `TILE` wide.
fn split_tiles(input_len: usize, ratio: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    if TILE >= input_len {
        return (vec![0], vec![input_len], Vec::new());
    }
    let mut n = input_len.div_ceil(TILE);
    let (mut overlaps, mut remaining);
    loop {
        overlaps = vec![TILE_OVERLAP_MIN; n - 1];
        let total: usize = overlaps.iter().sum();
        if TILE * n < total + input_len {
            n += 1;
            continue;
        }
        remaining = TILE * n - total - input_len;
        break;
    }
    for i in 0..remaining / ratio {
        overlaps[i % (n - 1)] += ratio;
    }
    let mut starts = vec![0usize];
    for i in 0..n - 1 {
        starts.push(starts[i] + TILE - overlaps[i]);
    }
    (starts, vec![TILE; n], overlaps)
}

impl VideoVae {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model.tensor_bytes("vvae.config_json").map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("vvae.config_json: {e}"))?;
        let u = |k: &str, d: usize| cfg[k].as_u64().map(|v| v as usize).unwrap_or(d);
        let f32v = |n: &str| crate::dit::cmf_f32(model, n);
        let n = u("num_layers", 36);
        let mut blocks = Vec::with_capacity(n);
        for l in 0..n {
            let p = format!("vvae.blocks.{l}");
            blocks.push(Block {
                norm1: f32v(&format!("{p}.norm1"))?,
                norm2: f32v(&format!("{p}.norm2"))?,
                scale1: f32v(&format!("{p}.scale1"))?,
                scale2: f32v(&format!("{p}.scale2"))?,
                qkv: Proj::from_model(model, &format!("{p}.attn.to_qkv.weight"))?,
                qkv_b: f32v(&format!("{p}.attn.to_qkv.bias"))?,
                out: Proj::from_model(model, &format!("{p}.attn.to_out.weight"))?,
                out_b: f32v(&format!("{p}.attn.to_out.bias"))?,
                w1: Proj::from_model(model, &format!("{p}.ff.w1.weight"))?,
                w1_b: f32v(&format!("{p}.ff.w1.bias"))?,
                w2: Proj::from_model(model, &format!("{p}.ff.w2.weight"))?,
                w2_b: f32v(&format!("{p}.ff.w2.bias"))?,
            });
        }
        let dim = u("dim", 2048);
        let heads = u("heads", 32);
        Ok(Self {
            post_quant: Proj::from_model(model, "vvae.post_quant_conv.weight")?,
            post_quant_b: f32v("vvae.post_quant_conv.bias")?,
            x_embed: Proj::from_model(model, "vvae.x_embedder.weight")?,
            x_embed_b: f32v("vvae.x_embedder.bias")?,
            registers: f32v("vvae.register_tokens")?,
            blocks,
            norm_out_w: f32v("vvae.norm_out.weight")?,
            norm_out_b: f32v("vvae.norm_out.bias")?,
            proj_out: Proj::from_model(model, "vvae.proj_out.weight")?,
            proj_out_b: f32v("vvae.proj_out.bias")?,
            latents_mean: f32v("vvae.latents_mean")?,
            latents_std: f32v("vvae.latents_std")?,
            pool: Pool::from_env(),
            dim,
            heads,
            head_dim: dim / heads,
            z_channels: u("z_channels", 24),
            patch: u("patch_size", 16),
            patch_t: u("patch_size_t", 4),
            n_reg: u("num_register_tokens", 4),
            rope_theta: cfg["rope_theta"].as_f64().unwrap_or(100.0) as f32,
            rope_dim: ((dim / heads) as f64 * cfg["rope_dim_ratio"].as_f64().unwrap_or(0.75))
                as usize,
            eps: cfg["eps"].as_f64().unwrap_or(1e-5),
        })
    }

    /// `arange(0.5, n)/n · 2 − 1` — the normalized cell centre of one
    /// axis, so a tile's coordinates depend on the TILE's extent and
    /// not on the frame's.
    fn axis(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 2.0 * ((i as f32 + 0.5) / n as f32) - 1.0)
            .collect()
    }

    /// `[S, rope_dim/2]` angles: three axes × `rope_dim/6` frequencies,
    /// scaled by 2π. Suffix tokens sit at the origin and get zeros.
    fn rope_angles(&self, t: usize, h: usize, w: usize, suffix: usize) -> Vec<f32> {
        let k = self.rope_dim / 6; // frequencies per axis
        let inv: Vec<f32> = (0..k)
            .map(|i| 1.0 / self.rope_theta.powf(i as f32 * 2.0 * 3.0 / self.rope_dim as f32))
            .collect();
        let (ta, ha, wa) = (Self::axis(t), Self::axis(h), Self::axis(w));
        let tau = 2.0 * std::f32::consts::PI;
        let mut out = Vec::with_capacity((t * h * w + suffix) * 3 * k);
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    for &c in &[ta[ti], ha[hi], wa[wi]] {
                        for &f in &inv {
                            out.push(tau * c * f);
                        }
                    }
                }
            }
        }
        out.extend(std::iter::repeat_n(0.0, suffix * 3 * k));
        out
    }

    fn attention(&self, qkv: &[f32], n: usize, attn: &mut [f32], angles: &[f32]) {
        let (nh, hd, dim) = (self.heads, self.head_dim, self.dim);
        let pairs = angles.len() / n; // = rope_dim / 2
        let scale = 1.0 / (hd as f32).sqrt();
        let pool = self.pool.as_deref();
        // Device path: the same `dit_qk`/`dit_softmax`/`dit_pv` chain the
        // DiT rides, so the n×n score plane stays on the card. This VAE
        // is the larger half of a render's wall (201 s of 475 on an RTX
        // 5090), and it was materializing those scores per head on the
        // host. CMF_VAE3D_ATTN=cpu forces the loop below.
        if std::env::var("CMF_VAE3D_ATTN").as_deref() != Ok("cpu")
            && crate::gpu::enabled_here()
            && n >= 256
        {
            // A CMF_VAE3D_CHECK harness lived here and was WRONG: its
            // reference arm re-entered this very function (so it
            // recursed through its own check) and its device arm was
            // never verified to have written anything, so its verdict
            // — "both layouts identical, 7.2765 from the host" — was
            // just max|host| against two buffers of zeros. It measured
            // itself. A replacement must assert the device arm wrote
            // before it compares, and must call the host repack
            // DIRECTLY rather than through the dispatcher.
            // A sound version of the harness the previous one failed to
            // be: a re-entry guard (its reference arm calls back into
            // this function), and a sentinel that proves the device arm
            // WROTE before its numbers are believed.
            //
            // VERDICT, and it moves the search a long way: the device
            // arm writes EVERY element (wrote=3680256/3680256) and it
            // writes ZEROS — `l1-host` comes back exactly equal to
            // `host|max|`, which is the distance from an all-zero
            // buffer. So the whole device attention is empty for this
            // decoder, and the split's addressing, its bias and its
            // layout are all downstream of a stage that has already
            // produced nothing. That also explains why every layout
            // experiment agreed: zero does not depend on addressing.
            //
            // Already eliminated, so nobody re-walks them:
            //  * the split/qk-norm encoder IS submitted before attention
            //    reads the planes (`c.queue.submit` at the end of that
            //    block) — the planes are not merely un-flushed;
            //  * the scratch slots are grow-only and the DiT ran first
            //    at LARGER shapes, so nothing here is undersized;
            //  * the split's dispatch grid is 14376 groups in x, well
            //    under the 65535 bound, so it is not rounding to zero;
            //  * `dit_attention_packed_src` returned true, so no stage
            //    refused — the chain ran and produced zeros.
            //
            // What is left: the three planes the split writes are read
            // back as zeros while the SAME kernels, fed the same three
            // planes uploaded from the host (`resident=false`, the
            // shipped VAE path), are correct at this very hd=64. So the
            // difference is the hand-off of the planes themselves, not
            // the attention math. Next: read `sc.dq` back right after
            // the split and compare it against the host repack — one
            // buffer, one dispatch, no attention involved.
            if std::env::var("CMF_VAE3D_CHECK").as_deref() == Ok("1")
                && !CHECKING.with(|c| c.get())
            {
                CHECKING.with(|c| c.set(true));
                const SENT: f32 = -12345.0;
                let mut d1 = vec![SENT; attn.len()];
                let mut d0 = vec![SENT; attn.len()];
                let ok1 = crate::gpu::vae_attention_packed_layout(
                    qkv, nh, n, hd, scale, angles, self.eps as f32, &mut d1, 1,
                );
                let ok0 = crate::gpu::vae_attention_packed_layout(
                    qkv, nh, n, hd, scale, angles, self.eps as f32, &mut d0, 0,
                );
                let wrote = |v: &[f32]| v.iter().filter(|&&x| x != SENT).count();
                let mut host = vec![0f32; attn.len()];
                let saved = std::env::var("CMF_VAE3D_ATTN").ok();
                // SAFETY: single-threaded diagnostic, guarded above.
                unsafe { std::env::set_var("CMF_VAE3D_ATTN", "cpu") };
                self.attention(qkv, n, &mut host, angles);
                match &saved {
                    Some(v) => unsafe { std::env::set_var("CMF_VAE3D_ATTN", v) },
                    None => unsafe { std::env::remove_var("CMF_VAE3D_ATTN") },
                }
                // The split ALONE: no norm, no RoPE, no attention. If
                // this mismatches, the hand-off is the whole story.
                let mut sq = vec![SENT; nh * n * hd];
                let oks = crate::gpu::dit_split_only(qkv, nh, n, hd, 1, None, &mut sq);
                let mut sqn = vec![SENT; nh * n * hd];
                let okn = crate::gpu::dit_split_only(
                    qkv, nh, n, hd, 1,
                    Some((angles, self.eps as f32)),
                    &mut sqn,
                );
                let mut hq = vec![0f32; nh * n * hd];
                for p in 0..n {
                    for h in 0..nh {
                        let b = p * 3 * dim + h * 3 * hd;
                        hq[(h * n + p) * hd..(h * n + p) * hd + hd]
                            .copy_from_slice(&qkv[b..b + hd]);
                    }
                }
                let sq_wrote = sq.iter().filter(|&&x| x != SENT).count();
                let sq_diff = sq
                    .iter()
                    .zip(&hq)
                    .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                // The same q plane after norm+RoPE, against the host's
                // own loop on a copy of it.
                let mut hqn = hq.clone();
                let pairs = angles.len() / n;
                for p in 0..n {
                    for h in 0..nh {
                        let r = &mut hqn[(h * n + p) * hd..(h * n + p) * hd + hd];
                        rms_norm_plain(r, self.eps);
                        for j in 0..pairs {
                            let (sn, cs) = angles[p * pairs + j].sin_cos();
                            let (a, b) = (r[j], r[j + pairs]);
                            r[j] = a * cs - b * sn;
                            r[j + pairs] = a * sn + b * cs;
                        }
                    }
                }
                let n_wrote = sqn.iter().filter(|&&x| x != SENT).count();
                let n_diff = sqn
                    .iter()
                    .zip(&hqn)
                    .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                eprintln!(
                    "vae split-only: ok={oks} wrote={sq_wrote}/{} maxdiff={sq_diff:.4e} host|max|={:.4e} || +norm: ok={okn} wrote={n_wrote} maxdiff={n_diff:.4e} host|max|={:.4e}",
                    sq.len(),
                    hq.iter().fold(0f32, |m, v| m.max(v.abs())),
                    hqn.iter().fold(0f32, |m, v| m.max(v.abs())),
                );
                let md = |a: &[f32], b: &[f32]| {
                    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
                };
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    eprintln!(
                        "vae attn check: ok={ok1}/{ok0} wrote={}/{} of {} | host|max| {:.4e} | l1-host {:.4e} | l0-host {:.4e} | l1-l0 {:.4e}",
                        wrote(&d1), wrote(&d0), attn.len(),
                        host.iter().fold(0f32, |m, v| m.max(v.abs())),
                        md(&d1, &host), md(&d0, &host), md(&d1, &d0),
                    )
                });
                CHECKING.with(|c| c.set(false));
            }
            if std::env::var("CMF_VAE3D_SPLIT").as_deref() == Ok("1")
                && crate::gpu::vae_attention_packed(
                    qkv, nh, n, hd, scale, angles, self.eps as f32, attn,
                )
            {
                return;
            }
            let t_rp = std::time::Instant::now();
            let mut qa = vec![0f32; nh * n * hd];
            let mut ka = vec![0f32; nh * n * hd];
            let mut va = vec![0f32; nh * n * hd];
            {
                let (pq, pk, pv) = (
                    SendPtr(qa.as_mut_ptr()),
                    SendPtr(ka.as_mut_ptr()),
                    SendPtr(va.as_mut_ptr()),
                );
                let fill = |lo: usize, hi: usize| {
                    for p in lo..hi {
                        for h in 0..nh {
                            let base = p * 3 * dim + h * 3 * hd;
                            let dst = (h * n + p) * hd;
                            // SAFETY: disjoint token ranges per worker,
                            // and each token owns its own head slots.
                            let (q, k, v) = unsafe {
                                (pq.row(dst, hd), pk.row(dst, hd), pv.row(dst, hd))
                            };
                            q.copy_from_slice(&qkv[base..base + hd]);
                            k.copy_from_slice(&qkv[base + hd..base + 2 * hd]);
                            v.copy_from_slice(&qkv[base + 2 * hd..base + 3 * hd]);
                            // q and k carry the same norm + rotation the
                            // host loop applies; v is untouched.
                            for x in [&mut *q, &mut *k] {
                                rms_norm_plain(x, self.eps);
                                for j in 0..pairs {
                                    let (sn, cs) = angles[p * pairs + j].sin_cos();
                                    let (a, b) = (x[j], x[j + pairs]);
                                    x[j] = a * cs - b * sn;
                                    x[j + pairs] = a * sn + b * cs;
                                }
                            }
                        }
                    }
                };
                match pool {
                    Some(pl) => pl.run_rows(n, &fill),
                    None => fill(0, n),
                }
            }
            vprof(2, t_rp);
            let t_at = std::time::Instant::now();
            if crate::gpu::dit_attention(&qa, &ka, &va, nh, nh, n, hd, scale, attn) {
                vprof(3, t_at);
                return;
            }
        }
        let mut qh = vec![0f32; n * hd];
        let mut kh = vec![0f32; n * hd];
        let mut vt = vec![0f32; hd * n];
        let mut scores = vec![0f32; n * n];
        let mut oh = vec![0f32; n * hd];
        for h in 0..nh {
            for p in 0..n {
                // to_qkv is viewed [.., heads, 3·hd] then chunked, so a
                // head's q, k and v are adjacent — not three planes.
                let base = p * 3 * dim + h * 3 * hd;
                qh[p * hd..(p + 1) * hd].copy_from_slice(&qkv[base..base + hd]);
                kh[p * hd..(p + 1) * hd].copy_from_slice(&qkv[base + hd..base + 2 * hd]);
                for d in 0..hd {
                    vt[d * n + p] = qkv[base + 2 * hd + d];
                }
            }
            for (buf, _) in [(&mut qh, 0), (&mut kh, 1)] {
                for p in 0..n {
                    let x = &mut buf[p * hd..(p + 1) * hd];
                    rms_norm_plain(x, self.eps);
                    for j in 0..pairs {
                        let (s, c) = angles[p * pairs + j].sin_cos();
                        let (a, b) = (x[j], x[j + pairs]);
                        x[j] = a * c - b * s;
                        x[j + pairs] = a * s + b * c;
                    }
                }
            }
            for v in qh.iter_mut() {
                *v *= scale;
            }
            crate::fcd_ops::gemm_nt(&qh, &kh, &mut scores, n, hd, n, pool);
            let sp = SendPtr(scores.as_mut_ptr());
            let soft = |lo: usize, hi: usize| {
                for r in lo..hi {
                    // SAFETY: workers own disjoint score rows.
                    softmax_inplace(unsafe { sp.row(r * n, n) });
                }
            };
            match pool {
                Some(pl) => pl.run_rows(n, &soft),
                None => soft(0, n),
            }
            crate::fcd_ops::gemm_nt(&scores, &vt, &mut oh, n, n, hd, pool);
            for p in 0..n {
                attn[p * dim + h * hd..p * dim + (h + 1) * hd]
                    .copy_from_slice(&oh[p * hd..(p + 1) * hd]);
            }
        }
    }

    /// One tile-clip: latents `[z_channels, t, h, w]` → pixels
    /// `[3, t·4, h·16, w·16]`, both channel-major.
    fn decode_tile(&self, z: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
        let pool = self.pool.as_deref();
        let dim = self.dim;
        let np = t * h * w;
        let suffix = 1 + self.n_reg;
        let n = np + suffix;

        // [C, T, H, W] → [T·H·W, C], then the 1×1×1 post-quant conv and
        // the embedder, which are both plain linears at this point.
        let zc = self.z_channels;
        let mut rows = vec![0f32; np * zc];
        for c in 0..zc {
            for i in 0..np {
                rows[i * zc + c] = z[c * np + i];
            }
        }
        let mut pq = vec![0f32; np * zc];
        self.post_quant.matmat(&rows, np, &mut pq, pool);
        for r in pq.chunks_exact_mut(zc) {
            for (v, &b) in r.iter_mut().zip(&self.post_quant_b) {
                *v += b;
            }
        }
        let mut x = vec![0f32; n * dim];
        self.x_embed.matmat(&pq, np, &mut x[..np * dim], pool);
        for r in x[..np * dim].chunks_exact_mut(dim) {
            for (v, &b) in r.iter_mut().zip(&self.x_embed_b) {
                *v += b;
            }
        }
        // register tokens, then one all-zero token
        x[np * dim..(np + self.n_reg) * dim].copy_from_slice(&self.registers);

        let angles = self.rope_angles(t, h, w, suffix);
        let mut xn = vec![0f32; n * dim];
        let mut qkv = vec![0f32; n * 3 * dim];
        let mut attn = vec![0f32; n * dim];
        let mut proj = vec![0f32; n * dim];
        let inner = 4 * dim;
        for blk in &self.blocks {
            let t = std::time::Instant::now();
            {
                let px = SendPtr(xn.as_mut_ptr());
                rows_par(pool, n, &|lo, hi| {
                    for p in lo..hi {
                        // SAFETY: workers own disjoint token rows.
                        rms_norm_into(&x[p * dim..(p + 1) * dim], &blk.norm1, self.eps, unsafe {
                            px.row(p * dim, dim)
                        });
                    }
                });
            }
            vprof(0, t);
            let t = std::time::Instant::now();
            // The whole attention half on the card: qkv GEMM, bias, the
            // weightless q/k norm, RoPE, attention, output projection —
            // nothing crossing the bus between them. The DiT's chain cut
            // its step 29%; this decoder spent 18.2 s of 26.1 on the
            // same six steps. `CMF_VAE3D_FUSE=0` restores the host chain.
            if std::env::var("CMF_GPU_DEBUG").is_ok() {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    eprintln!(
                        "vae3d shape: n={n} dim={dim} heads={} head_dim={} pairs={}",
                        self.heads,
                        self.head_dim,
                        angles.len() / n.max(1)
                    )
                });
            }
            let mut fused = false;
            // Default since the split kernel's encoder started reaching
            // the queue: this stage 26.1 s → 16.6 s and a 2-step render
            // 57.1 s → 46.9 s, at 2.33e-3 rel rms from the host chain
            // (max 8/255 on 3% of pixels) — four times inside the gate
            // the DiT's own device path is held to. `CMF_VAE3D_FUSE=0`
            // restores the host chain.
            //
            // It read 0.296 rel rms and 97% of pixels for most of a day,
            // and none of that was this code: the split never ran, so V
            // was zeros and P·V was zero. Zero does not depend on
            // addressing, which is why every layout experiment agreed
            // with every other and sent the search the wrong way.
            //
            // Narrowed to four stages, with everything else measured
            // and cleared (`CMF_VAE3D_CHECK=1` runs the harness):
            //   * the device attention returns ZEROS — proven with a
            //     sentinel fill, so "it wrote" is a fact, not a guess;
            //   * the split reproduces the host repack EXACTLY at this
            //     head-interleaved layout (maxdiff 0.0000e0);
            //   * qk-norm + RoPE match the host loop to 3.8e-6 against
            //     a magnitude of 6.87 — f32 rounding;
            //   * the plane pickup is sound: with an empty slice
            //     `Scratch::ensure` asks for 0 bytes, its `cap >= need`
            //     arm holds for any live slot, and both sides name the
            //     same slots with the same usage.
            // So the zeros are made in QK, softmax, PV or the unstack,
            // at nh=32 hd=64 n=1797 — shapes at which that same code is
            // correct when the planes arrive by upload instead
            // (`resident=false`, the shipped path). Give each of those
            // four private buffers, one at a time; the first that comes
            // back zero is the answer. Worth 26.1 s → 16.1 s on this stage
            // and 57.6 s → 47.2 s on a 2-step render once it is right.
            //
            // What is known: the error survives with both GEMMs on the
            // host (`CMF_VAE3D_SPLIT=1` reproduces it byte for byte), so
            // it lives in the split or the qk-norm kernel, not in the
            // resident hand-off. Teaching BOTH kernels this panel's
            // layout (head-interleaved, mode=1) and its bias changed the
            // output by nothing — which should be impossible at nh=32,
            // where the two addressings differ for every head above the
            // first. That contradiction is the thread to pull: the
            // one-shot probe in `packed_src` fires on the DiT's first
            // call and never reaches the VAE's, so instrument per layout.
            if std::env::var("CMF_VAE3D_FUSE").as_deref() != Ok("0")
                && crate::gpu::enabled_here()
                && n >= 256
            {
                if let (Proj::Q(q), Proj::Q(o)) = (&blk.qkv, &blk.out) {
                    if let (Some((m, i)), Some((_, oi))) = (q.mapped_q4tp(), o.mapped_q4tp()) {
                        fused = crate::gpu::vae_qkv_attn_out(
                            m,
                            i,
                            oi,
                            &xn,
                            n,
                            dim,
                            self.heads,
                            self.head_dim,
                            1.0 / (self.head_dim as f32).sqrt(),
                            &angles,
                            self.eps as f32,
                            &blk.qkv_b,
                            &mut proj,
                        );
                    }
                }
            }
            if fused {
                vprof(1, t);
            }
            if !fused {
            blk.qkv.matmat(&xn, n, &mut qkv, pool);
            {
                let pq = SendPtr(qkv.as_mut_ptr());
                rows_par(pool, n, &|lo, hi| {
                    for p in lo..hi {
                        // SAFETY: disjoint token rows.
                        for (v, &b) in unsafe { pq.row(p * 3 * dim, 3 * dim) }
                            .iter_mut()
                            .zip(&blk.qkv_b)
                        {
                            *v += b;
                        }
                    }
                });
            }
            vprof(1, t);
            self.attention(&qkv, n, &mut attn, &angles);
            let t = std::time::Instant::now();
            blk.out.matmat(&attn, n, &mut proj, pool);
            vprof(4, t);
            }
            {
                let pxx = SendPtr(x.as_mut_ptr());
                rows_par(pool, n, &|lo, hi| {
                    for p in lo..hi {
                        let r = &proj[p * dim..(p + 1) * dim];
                        // SAFETY: workers own disjoint token rows.
                        let dst = unsafe { pxx.row(p * dim, dim) };
                        for (i, v) in dst.iter_mut().enumerate() {
                            *v += (r[i] + blk.out_b[i]) * blk.scale1[i];
                        }
                    }
                });
            }
            let t = std::time::Instant::now();
            {
                let px = SendPtr(xn.as_mut_ptr());
                rows_par(pool, n, &|lo, hi| {
                    for p in lo..hi {
                        // SAFETY: workers own disjoint token rows.
                        rms_norm_into(&x[p * dim..(p + 1) * dim], &blk.norm2, self.eps, unsafe {
                            px.row(p * dim, dim)
                        });
                    }
                });
            }
            vprof(0, t);
            let t_ffn = std::time::Instant::now();
            // Device-resident FFN when both weights are q4tp: fc1 →
            // SwiGLU (with this VAE's gate/up bias) → fc2 without the
            // intermediate panel crossing the bus twice per block.
            // CMF_VAE3D_FFN=cpu forces the host chain.
            if std::env::var("CMF_VAE3D_FFN").as_deref() != Ok("cpu")
                && crate::gpu::enabled_here()
                && n >= 64
            {
                if let (Proj::Q(q1), Proj::Q(q2)) = (&blk.w1, &blk.w2) {
                    if let (Some((m, i1)), Some((_, i2))) =
                        (q1.mapped_q4tp(), q2.mapped_q4tp())
                    {
                        let mut fout = vec![0f32; n * dim];
                        if crate::gpu::q4tp_ffn_packed(
                            m,
                            i1,
                            i2,
                            &xn,
                            n,
                            dim,
                            inner,
                            Some(&blk.w1_b),
                            &mut fout,
                        ) {
                            {
                                let pxx = SendPtr(x.as_mut_ptr());
                                rows_par(pool, n, &|lo, hi| {
                                    for p in lo..hi {
                                        let r = &fout[p * dim..(p + 1) * dim];
                                        // SAFETY: disjoint token rows.
                                        let dst = unsafe { pxx.row(p * dim, dim) };
                                        for (i, v) in dst.iter_mut().enumerate() {
                                            *v += (r[i] + blk.w2_b[i]) * blk.scale2[i];
                                        }
                                    }
                                });
                            }
                            vprof(5, t_ffn);
                            continue;
                        }
                    }
                }
            }
            let mut gu = vec![0f32; n * 2 * inner];
            blk.w1.matmat(&xn, n, &mut gu, pool);
            let mut act = vec![0f32; n * inner];
            for p in 0..n {
                let r = &gu[p * 2 * inner..(p + 1) * 2 * inner];
                for i in 0..inner {
                    act[p * inner + i] =
                        silu(r[i] + blk.w1_b[i]) * (r[inner + i] + blk.w1_b[inner + i]);
                }
            }
            blk.w2.matmat(&act, n, &mut proj, pool);
            {
                let pxx = SendPtr(x.as_mut_ptr());
                rows_par(pool, n, &|lo, hi| {
                    for p in lo..hi {
                        let r = &proj[p * dim..(p + 1) * dim];
                        // SAFETY: disjoint token rows.
                        let dst = unsafe { pxx.row(p * dim, dim) };
                        for (i, v) in dst.iter_mut().enumerate() {
                            *v += (r[i] + blk.w2_b[i]) * blk.scale2[i];
                        }
                    }
                });
            }
            vprof(5, t_ffn);
        }

        let (pt, ps) = (self.patch_t, self.patch);
        let od = 3 * pt * ps * ps;
        let t_head = std::time::Instant::now();
        let mut head = vec![0f32; np * dim];
        {
            let ph = SendPtr(head.as_mut_ptr());
            rows_par(pool, np, &|lo, hi| {
                for p in lo..hi {
                    // SAFETY: disjoint token rows.
                    layer_norm(
                        &x[p * dim..(p + 1) * dim],
                        &self.norm_out_w,
                        &self.norm_out_b,
                        self.eps,
                        unsafe { ph.row(p * dim, dim) },
                    );
                }
            });
        }
        let mut out = vec![0f32; np * od];
        self.proj_out.matmat(&head, np, &mut out, pool);
        for r in out.chunks_exact_mut(od) {
            for (v, &b) in r.iter_mut().zip(&self.proj_out_b) {
                *v += b;
            }
        }

        vprof(6, t_head);
        let t_px = std::time::Instant::now();
        // [T,H,W, 3,pt,ps,ps] → [3, T·pt, H·ps, W·ps]
        let (ot, oh, ow) = (t * pt, h * ps, w * ps);
        let mut px = vec![0f32; 3 * ot * oh * ow];
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    let src = &out[((ti * h + hi) * w + wi) * od..((ti * h + hi) * w + wi + 1) * od];
                    for c in 0..3 {
                        for a in 0..pt {
                            for b in 0..ps {
                                for d in 0..ps {
                                    px[((c * ot + ti * pt + a) * oh + hi * ps + b) * ow
                                        + wi * ps + d] =
                                        src[((c * pt + a) * ps + b) * ps + d];
                                }
                            }
                        }
                    }
                }
            }
        }
        vprof(7, t_px);
        px
    }

    /// Spatially tiled decode of one temporal clip. `z` is
    /// `[z_channels, t, zh, zw]`; the result is `[3, t·4, zh·16, zw·16]`.
    fn decode_clip(&self, z: &[f32], t: usize, zh: usize, zw: usize) -> Vec<f32> {
        let r = self.patch;
        let (height, width) = (zh * r, zw * r);
        let (ys, yl, yo) = split_tiles(height, r);
        let (xs, xl, xo) = split_tiles(width, r);
        let frames = t * self.patch_t;
        let mut canvas = vec![0f32; 3 * frames * height * width];

        // The reference blends each tile into its predecessor's tail and
        // then trims the overlap off, so a pixel is written once.
        let mut row_tails: Vec<Vec<f32>> = Vec::new();
        let mut out_y = 0usize;
        for (i, (&ip, &il)) in ys.iter().zip(&yl).enumerate() {
            let (zi, zl) = (ip / r, il / r);
            let mut new_tails: Vec<Vec<f32>> = Vec::new();
            let mut left_tail: Option<Vec<f32>> = None;
            let mut out_x = 0usize;
            let mut row_h = 0usize;
            for (j, (&jp, &jl)) in xs.iter().zip(&xl).enumerate() {
                let (zj, zw_t) = (jp / r, jl / r);
                let mut sub = vec![0f32; self.z_channels * t * zl * zw_t];
                for c in 0..self.z_channels {
                    for ti in 0..t {
                        for hh in 0..zl {
                            for ww in 0..zw_t {
                                sub[((c * t + ti) * zl + hh) * zw_t + ww] =
                                    z[((c * t + ti) * zh + zi + hh) * zw + zj + ww];
                            }
                        }
                    }
                }
                let mut tile = self.decode_tile(&sub, t, zl, zw_t);
                let (mut th, mut tw) = (zl * r, zw_t * r);
                if i + 1 < ys.len() {
                    new_tails.push(crop(&tile, frames, th, tw, th - yo[i], th, 0, tw));
                }
                let next_left = if j + 1 < xs.len() {
                    Some(crop(&tile, frames, th, tw, 0, th, tw - xo[j], tw))
                } else {
                    None
                };
                if i > 0 {
                    tile = blend(&row_tails[j], &tile, frames, th, tw, yo[i - 1], 2);
                }
                if j > 0 {
                    let lt = left_tail.as_ref().unwrap();
                    tile = blend(lt, &tile, frames, th, tw, xo[j - 1], 3);
                }
                left_tail = next_left;
                if i + 1 < ys.len() {
                    tile = crop(&tile, frames, th, tw, 0, th - yo[i], 0, tw);
                    th -= yo[i];
                }
                if j + 1 < xs.len() {
                    tile = crop(&tile, frames, th, tw, 0, th, 0, tw - xo[j]);
                    tw -= xo[j];
                }
                for c in 0..3 {
                    for f in 0..frames {
                        for hh in 0..th {
                            let dst = ((c * frames + f) * height + out_y + hh) * width + out_x;
                            let src = ((c * frames + f) * th + hh) * tw;
                            canvas[dst..dst + tw].copy_from_slice(&tile[src..src + tw]);
                        }
                    }
                }
                out_x += tw;
                row_h = th;
            }
            row_tails = new_tails;
            out_y += row_h;
        }
        canvas
    }

    /// Normalized latents `[z_channels, t_lat, zh, zw]` → RGB in [0, 1],
    /// `[3, frames, zh·16, zw·16]`.
    pub fn decode(&self, z: &[f32], t_lat: usize, zh: usize, zw: usize) -> (Vec<f32>, usize) {
        let zc = self.z_channels;
        let np = t_lat * zh * zw;
        let mut zz = vec![0f32; zc * np];
        for c in 0..zc {
            let (m, s) = (self.latents_mean[c], self.latents_std[c]);
            for i in 0..np {
                zz[c * np + i] = z[c * np + i] * s + m;
            }
        }

        let ratio_t = self.patch_t;
        let chunk_tokens = CLIP_LENGTH.div_ceil(ratio_t); // 5
        let token_overlap = (chunk_tokens - TOKEN_DROP % chunk_tokens) % chunk_tokens; // 2
        let frame_pre_pad = (ratio_t - CLIP_LENGTH % ratio_t) % ratio_t; // 3
        let frame_overlap = (token_overlap * ratio_t).saturating_sub(frame_pre_pad); // 5
        let chunk_dec = chunk_tokens * ratio_t; // 20

        // The encoder dropped TOKEN_DROP tokens off the tail, so the
        // decoder plans against a longer pseudo-sequence and pads the
        // real one out with a repeat of its last token.
        let mut pseudo = t_lat + TOKEN_DROP;
        let mut pad = 0usize;
        if pseudo % chunk_tokens != 0 {
            pad = chunk_tokens - pseudo % chunk_tokens;
            pseudo += pad;
        }
        let mut chunks = pseudo / chunk_tokens - usize::from(TOKEN_DROP > 0);
        if chunks < 1 {
            pad += chunk_tokens;
            chunks += 1;
        }
        let t_pad = t_lat + pad;
        if pad > 0 {
            let mut grown = vec![0f32; zc * t_pad * zh * zw];
            for c in 0..zc {
                for ti in 0..t_pad {
                    let src = ti.min(t_lat - 1);
                    let a = (c * t_lat + src) * zh * zw;
                    let b = (c * t_pad + ti) * zh * zw;
                    grown[b..b + zh * zw].copy_from_slice(&zz[a..a + zh * zw]);
                }
            }
            zz = grown;
        }

        let (h, w) = (zh * self.patch, zw * self.patch);
        let mut out: Vec<f32> = Vec::new();
        let mut carry: Option<(Vec<f32>, usize)> = None;
        for i in 0..chunks {
            let a = i * chunk_tokens;
            let b = (a + chunk_tokens + token_overlap).min(t_pad);
            let n = b.saturating_sub(a.min(t_pad));
            if n == 0 {
                continue;
            }
            let mut sub = vec![0f32; zc * n * zh * zw];
            for c in 0..zc {
                let src = (c * t_pad + a) * zh * zw;
                let dst = c * n * zh * zw;
                sub[dst..dst + n * zh * zw].copy_from_slice(&zz[src..src + n * zh * zw]);
            }
            let dec = self.decode_clip(&sub, n, zh, zw);
            let dec_frames = n * ratio_t;
            for j in 0..2 {
                let fa = j * chunk_dec;
                let fb = (fa + chunk_dec).min(dec_frames);
                if fb <= fa + frame_pre_pad {
                    continue;
                }
                let mut part = frames_of(&dec, dec_frames, h, w, fa + frame_pre_pad, fb);
                let mut pn = fb - fa - frame_pre_pad;
                if j == 0 {
                    if let Some((tail, tn)) = carry.take() {
                        part = blend_frames(&tail, tn, &part, pn, h, w, frame_overlap);
                        pn = part.len() / (3 * h * w);
                    }
                    append_frames(&mut out, &part, pn, h, w);
                } else {
                    carry = Some((part, pn));
                }
            }
            if i + 1 == chunks {
                if let Some((tail, tn)) = carry.take() {
                    append_frames(&mut out, &tail, tn, h, w);
                }
            }
        }

        // Undo the ImageNet pixel normalization and clamp; the reference
        // then maps to [-1, 1] and every caller maps straight back, so
        // stop at [0, 1].
        let frames = out.len() / (3 * h * w);
        let want = t_lat * ratio_t - pad_frames(t_lat, pad, chunk_tokens, ratio_t);
        for c in 0..3 {
            let base = c * frames * h * w;
            for v in out[base..base + frames * h * w].iter_mut() {
                *v = (*v * IMAGENET_STD[c] + IMAGENET_MEAN[c]).clamp(0.0, 1.0);
            }
        }
        let keep = want.min(frames);
        if keep < frames {
            let mut trimmed = vec![0f32; 3 * keep * h * w];
            for c in 0..3 {
                let src = c * frames * h * w;
                let dst = c * keep * h * w;
                trimmed[dst..dst + keep * h * w].copy_from_slice(&out[src..src + keep * h * w]);
            }
            out = trimmed;
        }
        (out, keep)
    }

    pub fn spatial_ratio(&self) -> usize {
        self.patch
    }
    pub fn temporal_ratio(&self) -> usize {
        self.patch_t
    }
}

/// Frames the padding tokens manufactured, which the reference trims.
fn pad_frames(t_lat: usize, pad: usize, chunk_tokens: usize, ratio_t: usize) -> usize {
    if pad == 0 {
        return 0;
    }
    let intra = CLIP_LENGTH % ratio_t;
    if intra == 0 {
        return pad * ratio_t;
    }
    (0..pad)
        .map(|k| {
            if (t_lat + k) % chunk_tokens == 0 {
                intra
            } else {
                ratio_t
            }
        })
        .sum()
}

/// `[3, f, h, w]` sub-rectangle.
#[allow(clippy::too_many_arguments)]
fn crop(x: &[f32], f: usize, h: usize, w: usize, y0: usize, y1: usize, x0: usize, x1: usize) -> Vec<f32> {
    let (nh, nw) = (y1 - y0, x1 - x0);
    let mut out = vec![0f32; 3 * f * nh * nw];
    for c in 0..3 {
        for fi in 0..f {
            for yy in 0..nh {
                let s = ((c * f + fi) * h + y0 + yy) * w + x0;
                let d = ((c * f + fi) * nh + yy) * nw;
                out[d..d + nw].copy_from_slice(&x[s..s + nw]);
            }
        }
    }
    out
}

/// Linear cross-fade of `a`'s tail into `b`'s head along `dim` (2 = y,
/// 3 = x); the result is `b` with its first `extent` rows/columns
/// replaced by the blend.
#[allow(clippy::too_many_arguments)]
fn blend(a: &[f32], b: &[f32], f: usize, h: usize, w: usize, extent: usize, dim: usize) -> Vec<f32> {
    let ah = a.len() / (3 * f * w);
    let aw = a.len() / (3 * f * h);
    let mut out = b.to_vec();
    let e = if dim == 2 {
        extent.min(ah).min(h)
    } else {
        extent.min(aw).min(w)
    };
    for c in 0..3 {
        for fi in 0..f {
            for k in 0..e {
                let wb = k as f32 / e as f32;
                let wa = 1.0 - wb;
                if dim == 2 {
                    let sa = ((c * f + fi) * ah + ah - e + k) * w;
                    let sb = ((c * f + fi) * h + k) * w;
                    for x in 0..w {
                        out[sb + x] = a[sa + x] * wa + b[sb + x] * wb;
                    }
                } else {
                    for y in 0..h {
                        let sa = ((c * f + fi) * h + y) * aw + aw - e + k;
                        let sb = ((c * f + fi) * h + y) * w + k;
                        out[sb] = a[sa] * wa + b[sb] * wb;
                    }
                }
            }
        }
    }
    out
}

/// Frames `[a, b)` of a `[3, f, h, w]` clip.
fn frames_of(x: &[f32], f: usize, h: usize, w: usize, a: usize, b: usize) -> Vec<f32> {
    let n = b - a;
    let mut out = vec![0f32; 3 * n * h * w];
    for c in 0..3 {
        let s = (c * f + a) * h * w;
        let d = c * n * h * w;
        out[d..d + n * h * w].copy_from_slice(&x[s..s + n * h * w]);
    }
    out
}

/// Cross-fade `tail` into the head of `part` along the frame axis.
#[allow(clippy::too_many_arguments)]
fn blend_frames(
    tail: &[f32],
    tn: usize,
    part: &[f32],
    pn: usize,
    h: usize,
    w: usize,
    extent: usize,
) -> Vec<f32> {
    let e = extent.min(tn).min(pn);
    let mut out = part.to_vec();
    for c in 0..3 {
        for k in 0..e {
            let wb = k as f32 / e as f32;
            let wa = 1.0 - wb;
            let s = ((c * tn) + tn - e + k) * h * w;
            let d = ((c * pn) + k) * h * w;
            for i in 0..h * w {
                out[d + i] = tail[s + i] * wa + part[d + i] * wb;
            }
        }
    }
    out
}

/// Append a `[3, n, h, w]` clip to a growing `[3, ?, h, w]` buffer.
fn append_frames(out: &mut Vec<f32>, part: &[f32], n: usize, h: usize, w: usize) {
    let old = out.len() / (3 * h * w);
    let total = old + n;
    let mut grown = vec![0f32; 3 * total * h * w];
    for c in 0..3 {
        if old > 0 {
            let s = c * old * h * w;
            let d = c * total * h * w;
            grown[d..d + old * h * w].copy_from_slice(&out[s..s + old * h * w]);
        }
        let s = c * n * h * w;
        let d = (c * total + old) * h * w;
        grown[d..d + n * h * w].copy_from_slice(&part[s..s + n * h * w]);
    }
    *out = grown;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_schedule_matches_the_reference() {
        // 288 tall: two 256 tiles, the overlap grown to swallow the slack.
        let (s, l, o) = split_tiles(288, 16);
        assert_eq!(s, vec![0, 32]);
        assert_eq!(l, vec![256, 256]);
        assert_eq!(o, vec![224]);
        // 512 wide does not fit in two tiles at the minimum overlap.
        let (s, l, o) = split_tiles(512, 16);
        assert_eq!(s, vec![0, 128, 256]);
        assert_eq!(l, vec![256; 3]);
        assert_eq!(o, vec![128, 128]);
        // Anything at or under one tile is one tile.
        assert_eq!(split_tiles(256, 16).0, vec![0]);
        assert_eq!(split_tiles(128, 16).1, vec![128]);
    }

    #[test]
    fn temporal_constants_come_out_as_the_reference_computes_them() {
        let ratio_t = 4usize;
        let chunk = CLIP_LENGTH.div_ceil(ratio_t);
        assert_eq!(chunk, 5);
        assert_eq!((chunk - TOKEN_DROP % chunk) % chunk, 2);
        assert_eq!((ratio_t - CLIP_LENGTH % ratio_t) % ratio_t, 3);
    }
}

// ── the encoder, for keyframes only ─────────────────────────────────
//
// `fl2va` conditions on a first and/or last frame, and a frame is ONE
// frame. That collapses the whole 3-D causal encoder to a 2-D one:
// causal padding fills the front with zeros, and the reference's own
// `autopad="causal_zero"` therefore trims the kernel to
// `weight[:, :, -T:]` — at T = 1, the last temporal tap and nothing
// else. So the packer stores that tap alone (a third of the bytes) and
// this runs plain 2-D convolutions over it. Encoding real video would
// need the other two taps back; keyframes never reach them.

/// A 2-D convolution with the reference's padding policy: reflect on
/// H and W, applied by hand because the checkpoint's convolutions
/// carry no padding of their own.
struct EncConv {
    w: Vec<f32>, // [out, in, kh, kw]
    b: Vec<f32>,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

impl EncConv {
    fn load(model: &Arc<CmfModel>, name: &str, stride: usize, pad: usize) -> Result<Self, String> {
        let e = model
            .tensor(&format!("{name}.weight"))
            .ok_or_else(|| format!("missing {name}.weight"))?;
        // [out, in, kh, kw] — the packer already dropped the temporal axis.
        let (out_ch, in_ch, k) = (e.shape[0], e.shape[1], e.shape[2]);
        Ok(Self {
            w: crate::dit::cmf_f32(model, &format!("{name}.weight"))?,
            b: crate::dit::cmf_f32(model, &format!("{name}.bias"))?,
            out_ch,
            in_ch,
            k,
            stride,
            pad,
        })
    }

    /// `x` is `[in_ch, h, w]`; reflect-padded by `self.pad` on each side.
    fn apply(&self, x: &[f32], h: usize, w: usize, pool: Option<&Pool>) -> (Vec<f32>, usize, usize) {
        let (ph, pw) = (h + 2 * self.pad, w + 2 * self.pad);
        let oh = (ph - self.k) / self.stride + 1;
        let ow = (pw - self.k) / self.stride + 1;
        // Reflect padding: index |p| mirrored about the edges, which for
        // pad 1 on a plane of at least 2 is just the neighbour row.
        let refl = |i: isize, n: usize| -> usize {
            let n = n as isize;
            let mut i = i;
            while i < 0 || i >= n {
                if i < 0 {
                    i = -i;
                }
                if i >= n {
                    i = 2 * (n - 1) - i;
                }
            }
            i as usize
        };
        let mut out = vec![0f32; self.out_ch * oh * ow];
        let ptr = SendPtr(out.as_mut_ptr());
        let work = |lo: usize, hi: usize| {
            for o in lo..hi {
                // SAFETY: workers own disjoint output channels.
                let dst = unsafe { ptr.row(o * oh * ow, oh * ow) };
                dst.fill(self.b[o]);
                for i in 0..self.in_ch {
                    let ker = &self.w[(o * self.in_ch + i) * self.k * self.k
                        ..(o * self.in_ch + i + 1) * self.k * self.k];
                    let src = &x[i * h * w..(i + 1) * h * w];
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let mut acc = 0f32;
                            for ky in 0..self.k {
                                let sy = (oy * self.stride + ky) as isize - self.pad as isize;
                                let sy = refl(sy, h);
                                for kx in 0..self.k {
                                    let sx = (ox * self.stride + kx) as isize - self.pad as isize;
                                    acc += ker[ky * self.k + kx] * src[sy * w + refl(sx, w)];
                                }
                            }
                            dst[oy * ow + ox] += acc;
                        }
                    }
                }
            }
        };
        match pool {
            Some(p) => p.run_rows(self.out_ch, &work),
            None => work(0, self.out_ch),
        }
        (out, oh, ow)
    }
}

/// GroupNorm(32) with affine parameters, over `[ch, h, w]`.
struct GroupNorm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl GroupNorm {
    fn load(model: &Arc<CmfModel>, name: &str) -> Result<Self, String> {
        Ok(Self {
            w: crate::dit::cmf_f32(model, &format!("{name}.weight"))?,
            b: crate::dit::cmf_f32(model, &format!("{name}.bias"))?,
        })
    }

    fn apply(&self, x: &mut [f32], ch: usize, hw: usize) {
        let groups = 32;
        let per = ch / groups;
        for g in 0..groups {
            let seg = &mut x[g * per * hw..(g + 1) * per * hw];
            let n = seg.len() as f64;
            let mean = seg.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = seg.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            let inv = 1.0 / (var + 1e-6).sqrt();
            for (i, v) in seg.iter_mut().enumerate() {
                let c = g * per + i / hw;
                *v = ((*v as f64 - mean) * inv) as f32 * self.w[c] + self.b[c];
            }
        }
    }
}

struct ResBlock {
    norm1: GroupNorm,
    norm2: GroupNorm,
    conv1: EncConv,
    conv2: EncConv,
    shortcut: Option<EncConv>,
}

/// The encoder half: `[3, h, w]` in [-1, 1] → normalized latents
/// `[24, h/16, w/16]`.
pub struct VideoVaeEncoder {
    conv_in: EncConv,
    levels: Vec<(Vec<ResBlock>, Option<EncConv>)>,
    norm_out: GroupNorm,
    conv_out: EncConv,
    quant: Vec<f32>, // [48, 48] 1x1x1
    quant_b: Vec<f32>,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    pool: Option<Arc<Pool>>,
    z_channels: usize,
    ratio: usize,
}

impl VideoVaeEncoder {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model
                .tensor_bytes("vvae.config_json")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("vvae.config_json: {e}"))?;
        let space_down: Vec<usize> = cfg["space_down"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect())
            .unwrap_or_else(|| vec![2, 2, 2, 2, 1, 1]);
        let n_res = cfg["num_res_blocks"].as_u64().unwrap_or(2) as usize;
        let mut levels = Vec::new();
        for (i, &sd) in space_down.iter().enumerate() {
            let mut blocks = Vec::new();
            for j in 0..n_res {
                let p = format!("vvae.enc.down.{i}.block.{j}");
                let shortcut = model
                    .tensor(&format!("{p}.nin_shortcut.weight"))
                    .map(|_| EncConv::load(model, &format!("{p}.nin_shortcut"), 1, 0))
                    .transpose()?;
                blocks.push(ResBlock {
                    norm1: GroupNorm::load(model, &format!("{p}.norm1"))?,
                    norm2: GroupNorm::load(model, &format!("{p}.norm2"))?,
                    conv1: EncConv::load(model, &format!("{p}.conv1"), 1, 1)?,
                    conv2: EncConv::load(model, &format!("{p}.conv2"), 1, 1)?,
                    shortcut,
                });
            }
            // The downsample's own convolution pads nothing: the caller
            // reflect-pads one row and column on the far side instead.
            let down = if sd > 1 {
                Some(EncConv::load(
                    model,
                    &format!("vvae.enc.down.{i}.downsample.conv"),
                    sd,
                    0,
                )?)
            } else {
                None
            };
            levels.push((blocks, down));
        }
        Ok(Self {
            conv_in: EncConv::load(model, "vvae.enc.conv_in", 1, 1)?,
            levels,
            norm_out: GroupNorm::load(model, "vvae.enc.norm_out")?,
            conv_out: EncConv::load(model, "vvae.enc.conv_out", 1, 1)?,
            quant: crate::dit::cmf_f32(model, "vvae.quant_conv.weight")?,
            quant_b: crate::dit::cmf_f32(model, "vvae.quant_conv.bias")?,
            latents_mean: crate::dit::cmf_f32(model, "vvae.latents_mean")?,
            latents_std: crate::dit::cmf_f32(model, "vvae.latents_std")?,
            pool: Pool::from_env(),
            z_channels: cfg["z_channels"].as_u64().unwrap_or(24) as usize,
            ratio: cfg["patch_size"].as_u64().unwrap_or(16) as usize,
        })
    }

    /// One tile, unpadded: `[3, h, w]` → moments `[2·z, h/16, w/16]`.
    fn encode_tile(&self, x: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let pool = self.pool.as_deref();
        let (mut cur, mut ch, mut cw) = self.conv_in.apply(x, h, w, pool);
        let mut c = self.conv_in.out_ch;
        for (blocks, down) in &self.levels {
            for b in blocks {
                let mut hh = cur.clone();
                self.norm_out_like(&b.norm1, &mut hh, c, ch * cw);
                for v in hh.iter_mut() {
                    *v = silu(*v);
                }
                let (mut t, th, tw) = b.conv1.apply(&hh, ch, cw, pool);
                let oc = b.conv1.out_ch;
                self.norm_out_like(&b.norm2, &mut t, oc, th * tw);
                for v in t.iter_mut() {
                    *v = silu(*v);
                }
                let (t2, th2, tw2) = b.conv2.apply(&t, th, tw, pool);
                let skip = match &b.shortcut {
                    Some(s) => s.apply(&cur, ch, cw, pool).0,
                    None => cur.clone(),
                };
                cur = t2.iter().zip(&skip).map(|(&a, &b)| a + b).collect();
                (ch, cw, c) = (th2, tw2, oc);
            }
            if let Some(d) = down {
                // reflect one row and column on the far side, as the
                // reference does before its unpadded stride-2 kernel
                let (padded, ph, pw) = reflect_pad_far(&cur, c, ch, cw);
                let (t, th, tw) = d.apply(&padded, ph, pw, pool);
                cur = t;
                (ch, cw, c) = (th, tw, d.out_ch);
            }
        }
        self.norm_out_like(&self.norm_out, &mut cur, c, ch * cw);
        for v in cur.iter_mut() {
            *v = silu(*v);
        }
        let (moments, mh, mw) = self.conv_out.apply(&cur, ch, cw, pool);
        // quant_conv is 1x1x1: a per-position linear.
        let n = 2 * self.z_channels;
        let mut out = vec![0f32; n * mh * mw];
        for o in 0..n {
            for p in 0..mh * mw {
                let mut acc = self.quant_b[o];
                for i in 0..n {
                    acc += self.quant[o * n + i] * moments[i * mh * mw + p];
                }
                out[o * mh * mw + p] = acc;
            }
        }
        (out, mh, mw)
    }

    fn norm_out_like(&self, g: &GroupNorm, x: &mut [f32], ch: usize, hw: usize) {
        g.apply(x, ch, hw);
    }

    /// A single frame in [-1, 1], `[3, h, w]` → normalized latents
    /// `[z, h/16, w/16]`, spatially tiled exactly as the reference does.
    pub fn encode_frame(&self, rgb: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        // [-1, 1] → [0, 1] → ImageNet-normalized, the reference's order.
        let mut x = vec![0f32; 3 * h * w];
        for c in 0..3 {
            for p in 0..h * w {
                let v = (rgb[c * h * w + p] + 1.0) * 0.5;
                x[c * h * w + p] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
        let r = self.ratio;
        let (ys, yl, yo) = split_tiles(h, r);
        let (xs, xl, xo) = split_tiles(w, r);
        let (zh, zw) = (h / r, w / r);
        let zc = 2 * self.z_channels;
        let mut rows: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut dims: Vec<Vec<(usize, usize)>> = Vec::new();
        for (&ip, &il) in ys.iter().zip(&yl) {
            let mut row = Vec::new();
            let mut rd = Vec::new();
            for (&jp, &jl) in xs.iter().zip(&xl) {
                let mut sub = vec![0f32; 3 * il * jl];
                for c in 0..3 {
                    for yy in 0..il {
                        let s = (c * h + ip + yy) * w + jp;
                        let d = (c * il + yy) * jl;
                        sub[d..d + jl].copy_from_slice(&x[s..s + jl]);
                    }
                }
                let (t, th, tw) = self.encode_tile(&sub, il, jl);
                row.push(t);
                rd.push((th, tw));
            }
            rows.push(row);
            dims.push(rd);
        }
        // Latent-space blend then trim, mirroring `tiled_encode`.
        let mut canvas = vec![0f32; zc * zh * zw];
        let mut out_y = 0usize;
        for i in 0..rows.len() {
            let mut out_x = 0usize;
            let mut row_h = 0usize;
            for j in 0..rows[i].len() {
                let (th, tw) = dims[i][j];
                let mut tile = rows[i][j].clone();
                let (mut ch_, mut cw_) = (th, tw);
                if i > 0 {
                    tile = blend_plane(&rows[i - 1][j], &tile, zc, dims[i - 1][j], (ch_, cw_), yo[i - 1] / r, 0);
                }
                if j > 0 {
                    tile = blend_plane(&rows[i][j - 1], &tile, zc, dims[i][j - 1], (ch_, cw_), xo[j - 1] / r, 1);
                }
                if i + 1 < rows.len() {
                    tile = crop_plane(&tile, zc, ch_, cw_, 0, ch_ - yo[i] / r, 0, cw_);
                    ch_ -= yo[i] / r;
                }
                if j + 1 < rows[i].len() {
                    tile = crop_plane(&tile, zc, ch_, cw_, 0, ch_, 0, cw_ - xo[j] / r);
                    cw_ -= xo[j] / r;
                }
                for c in 0..zc {
                    for yy in 0..ch_ {
                        let d = (c * zh + out_y + yy) * zw + out_x;
                        let s = (c * ch_ + yy) * cw_;
                        canvas[d..d + cw_].copy_from_slice(&tile[s..s + cw_]);
                    }
                }
                out_x += cw_;
                row_h = ch_;
            }
            out_y += row_h;
        }
        // The posterior MEAN is the first z channels; no sampling.
        let mut z = vec![0f32; self.z_channels * zh * zw];
        for c in 0..self.z_channels {
            let (m, s) = (self.latents_mean[c], self.latents_std[c]);
            for p in 0..zh * zw {
                z[c * zh * zw + p] = (canvas[c * zh * zw + p] - m) / s;
            }
        }
        (z, zh, zw)
    }
}

/// Reflect one row and one column onto the far edge — `F.pad(x, (0,1,0,1))`.
fn reflect_pad_far(x: &[f32], c: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (ph, pw) = (h + 1, w + 1);
    let mut out = vec![0f32; c * ph * pw];
    for ci in 0..c {
        for y in 0..ph {
            let sy = if y < h { y } else { h - 2 };
            for x2 in 0..pw {
                let sx = if x2 < w { x2 } else { w - 2 };
                out[(ci * ph + y) * pw + x2] = x[(ci * h + sy) * w + sx];
            }
        }
    }
    (out, ph, pw)
}

/// Cross-fade `a`'s tail into `b`'s head over `extent`, `dim` 0 = y, 1 = x.
fn blend_plane(
    a: &[f32],
    b: &[f32],
    c: usize,
    ad: (usize, usize),
    bd: (usize, usize),
    extent: usize,
    dim: usize,
) -> Vec<f32> {
    let (ah, aw) = ad;
    let (bh, bw) = bd;
    let mut out = b.to_vec();
    let e = if dim == 0 { extent.min(ah).min(bh) } else { extent.min(aw).min(bw) };
    if e == 0 {
        return out;
    }
    for ci in 0..c {
        for k in 0..e {
            let wb = k as f32 / e as f32;
            let wa = 1.0 - wb;
            if dim == 0 {
                for x in 0..bw.min(aw) {
                    let sa = (ci * ah + ah - e + k) * aw + x;
                    let sb = (ci * bh + k) * bw + x;
                    out[sb] = a[sa] * wa + b[sb] * wb;
                }
            } else {
                for y in 0..bh.min(ah) {
                    let sa = (ci * ah + y) * aw + aw - e + k;
                    let sb = (ci * bh + y) * bw + k;
                    out[sb] = a[sa] * wa + b[sb] * wb;
                }
            }
        }
    }
    out
}

/// `[c, h, w]` sub-rectangle.
#[allow(clippy::too_many_arguments)]
fn crop_plane(x: &[f32], c: usize, h: usize, w: usize, y0: usize, y1: usize, x0: usize, x1: usize) -> Vec<f32> {
    let (nh, nw) = (y1 - y0, x1 - x0);
    let mut out = vec![0f32; c * nh * nw];
    for ci in 0..c {
        for y in 0..nh {
            let s = (ci * h + y0 + y) * w + x0;
            let d = (ci * nh + y) * nw;
            out[d..d + nw].copy_from_slice(&x[s..s + nw]);
        }
    }
    out
}
