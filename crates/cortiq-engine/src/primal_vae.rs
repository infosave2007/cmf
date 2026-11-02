//! Primal VAE (`ostris/primal_vae`): a deterministic convolutional
//! autoencoder, RGB in [-1, 1] ↔ a 1- or 2-channel latent at 2× spatial
//! compression, already unit normal (no scaling or shift factor).
//!
//! Both released models are the same network:
//!
//! * `pixel_unshuffle(2)` of the image (12 channels at half size) into the
//!   encoder; the decoder's 12 channels `pixel_shuffle(2)` back to RGB;
//! * encoder and decoder are mirrored U-Nets ([`UStack`]): `conv_in`, per
//!   level residual blocks then `pixel_unshuffle(2)` + a 3×3 conv down, the
//!   bottom blocks, per level up a 3×3 conv + `pixel_shuffle(2)` + the
//!   skip, residual blocks, then RMSNorm → SiLU → `conv_out`;
//! * a residual block is `x + conv2(silu(norm2(conv1(silu(norm1(x))))))`,
//!   `conv1` dilated by 1, 2, 4 cyclically along the level;
//! * every norm is an RMSNorm over the channels of one pixel (ε = 1e-6,
//!   a weight, no bias) — no spatial statistics, no attention.
//!
//! The architecture is read from the tensor shapes, as the reference
//! `PrimalVAE.config_from_shapes` does. The forward runs as a step list
//! ([`PvStep`]) that the device backends execute whole (`gpu::primal_vae_run`:
//! one upload, one command buffer, one readback); the host executes the
//! same list when no device takes it.
//!
//! Parity: `scripts/four/primal_vae_ref.py` dumps the diffusers model's
//! latents and reconstructions in float32.

use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

/// Dilation of block `j`'s first conv within a level.
const DILATION_CYCLE: [u32; 3] = [1, 2, 4];
const EPS: f32 = 1e-6;

/// One 3×3 convolution: weight `[cout][cin·9]` (the source `[cout, cin, 3,
/// 3]` flattened), bias `[cout]`.
pub struct PvConv {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
}

/// All weights of one model, in the order the step lists index them.
#[derive(Default)]
pub struct PvWeights {
    pub convs: Vec<PvConv>,
    pub gammas: Vec<Vec<f32>>,
}

/// One device step. Buffers are numbered; `h`×`w` is the spatial size of
/// the step's SOURCE, NCHW f32 throughout.
#[derive(Clone, Copy, Debug)]
pub enum PvStep {
    /// `dst = conv(src) + b (+ res)`, 3×3, dilation = padding = `dil`.
    Conv { wi: usize, src: u8, dst: u8, res: Option<u8>, dil: u32, h: usize, w: usize },
    /// `dst = silu(rmsnorm_c(src) · γ)`.
    NormSilu { gi: usize, src: u8, dst: u8, c: usize, h: usize, w: usize },
    /// `dst = pixel_unshuffle(src, 2)`: `[c, h, w] → [4c, h/2, w/2]`.
    Unshuffle { src: u8, dst: u8, c: usize, h: usize, w: usize },
    /// `dst = pixel_shuffle(src, 2) (+ add)`: `[c, h, w] → [c/4, 2h, 2w]`.
    Shuffle { src: u8, dst: u8, add: Option<u8>, c: usize, h: usize, w: usize },
}

/// A compiled forward: steps, buffer sizes (elements), input / output ids.
pub struct PvPlan {
    pub steps: Vec<PvStep>,
    pub buf_elems: Vec<usize>,
    pub input: u8,
    pub output: u8,
    pub out_elems: usize,
}

struct ResBlock {
    norm1: usize,
    conv1: usize,
    dil: u32,
    norm2: usize,
    conv2: usize,
}

/// One U-Net (encoder or decoder), as indices into [`PvWeights`].
struct UStack {
    widths: Vec<usize>,
    conv_in: usize,
    down_blocks: Vec<Vec<ResBlock>>,
    downs: Vec<usize>,
    ups: Vec<usize>,
    up_blocks: Vec<Vec<ResBlock>>,
    norm_out: usize,
    conv_out: usize,
}

/// The model.
pub struct PrimalVae {
    pub compression: usize,
    pub latent_channels: usize,
    weights: PvWeights,
    encoder: UStack,
    decoder: UStack,
    /// Process-unique key for the device weight cache.
    key: u64,
    pool: Option<Arc<Pool>>,
}

fn load_f32(model: &Arc<CmfModel>, name: &str) -> Result<Vec<f32>, String> {
    crate::dit::cmf_f32(model, name)
}

fn shape(model: &CmfModel, name: &str) -> Option<Vec<usize>> {
    model.tensor(name).map(|t| t.shape.clone())
}

fn push_conv(model: &Arc<CmfModel>, name: &str, cin_cout: Option<usize>, wts: &mut PvWeights) -> Result<usize, String> {
    let s = shape(model, &format!("{name}.weight")).ok_or_else(|| format!("primal_vae: missing {name}.weight"))?;
    if s.len() != 4 || s[2] != 3 || s[3] != 3 {
        return Err(format!("primal_vae: {name} is not a 3×3 conv ({s:?})"));
    }
    if cin_cout.is_some_and(|c| s[0] != c || s[1] != c) {
        return Err(format!("primal_vae: {name} is not a square conv of the level width"));
    }
    let w = load_f32(model, &format!("{name}.weight"))?;
    let b = load_f32(model, &format!("{name}.bias"))?;
    if b.len() != s[0] || w.len() != s[0] * s[1] * 9 {
        return Err(format!("primal_vae: {name} shapes disagree"));
    }
    wts.convs.push(PvConv { w, b, cin: s[1], cout: s[0] });
    Ok(wts.convs.len() - 1)
}

fn push_gamma(model: &Arc<CmfModel>, name: &str, c: usize, wts: &mut PvWeights) -> Result<usize, String> {
    let g = load_f32(model, &format!("{name}.weight"))?;
    if g.len() != c {
        return Err(format!("primal_vae: {name} has {} channels, expected {c}", g.len()));
    }
    wts.gammas.push(g);
    Ok(wts.gammas.len() - 1)
}

fn load_blocks(model: &Arc<CmfModel>, pre: &str, c: usize, wts: &mut PvWeights) -> Result<Vec<ResBlock>, String> {
    let mut out = Vec::new();
    for j in 0.. {
        let p = format!("{pre}.{j}");
        if model.tensor(&format!("{p}.conv1.weight")).is_none() {
            break;
        }
        out.push(ResBlock {
            norm1: push_gamma(model, &format!("{p}.norm1"), c, wts)?,
            conv1: push_conv(model, &format!("{p}.conv1"), Some(c), wts)?,
            dil: DILATION_CYCLE[j % DILATION_CYCLE.len()],
            norm2: push_gamma(model, &format!("{p}.norm2"), c, wts)?,
            conv2: push_conv(model, &format!("{p}.conv2"), Some(c), wts)?,
        });
    }
    Ok(out)
}

impl UStack {
    fn load(model: &Arc<CmfModel>, pre: &str, wts: &mut PvWeights) -> Result<Self, String> {
        let conv_in = push_conv(model, &format!("{pre}.conv_in"), None, wts)?;
        let mut widths = vec![wts.convs[conv_in].cout];
        let mut downs = Vec::new();
        while model.tensor(&format!("{pre}.downs.{}.weight", widths.len() - 1)).is_some() {
            let i = widths.len() - 1;
            let d = push_conv(model, &format!("{pre}.downs.{i}"), None, wts)?;
            if wts.convs[d].cin != 4 * widths[i] {
                return Err(format!("primal_vae: {pre}.downs.{i} does not take the unshuffled level"));
            }
            widths.push(wts.convs[d].cout);
            downs.push(d);
        }
        let levels = widths.len();
        let mut ups = Vec::new();
        for i in 0..levels - 1 {
            let u = push_conv(model, &format!("{pre}.ups.{i}"), None, wts)?;
            if wts.convs[u].cin != widths[i + 1] || wts.convs[u].cout != 4 * widths[i] {
                return Err(format!("primal_vae: {pre}.ups.{i} does not feed the shuffled level"));
            }
            ups.push(u);
        }
        let mut down_blocks = Vec::new();
        for (i, &c) in widths.iter().enumerate() {
            down_blocks.push(load_blocks(model, &format!("{pre}.down_blocks.{i}"), c, wts)?);
        }
        let mut up_blocks = Vec::new();
        for (i, &c) in widths.iter().take(levels - 1).enumerate() {
            up_blocks.push(load_blocks(model, &format!("{pre}.up_blocks.{i}"), c, wts)?);
        }
        let norm_out = push_gamma(model, &format!("{pre}.norm_out"), widths[0], wts)?;
        let conv_out = push_conv(model, &format!("{pre}.conv_out"), None, wts)?;
        Ok(UStack { widths, conv_in, down_blocks, downs, ups, up_blocks, norm_out, conv_out })
    }

    /// Append this U-Net's steps: `src` (`[cin, h, w]`) → `dst`. Buffers:
    /// 0 holds the bottom level's stream, 3+i level i's (which is also the
    /// level's skip: the up path adds into it in place), 1 and 2 are
    /// temporaries.
    fn plan(&self, wts: &PvWeights, steps: &mut Vec<PvStep>, sizes: &mut Vec<usize>, src: u8, dst: u8, h: usize, w: usize) {
        let need = |sizes: &mut Vec<usize>, b: u8, n: usize| {
            if sizes.len() <= b as usize {
                sizes.resize(b as usize + 1, 0);
            }
            sizes[b as usize] = sizes[b as usize].max(n);
        };
        let levels = self.widths.len();
        let xb = |i: usize| if i == levels - 1 { 0u8 } else { 3 + i as u8 };
        let (t, hb) = (1u8, 2u8);
        let block = |steps: &mut Vec<PvStep>, sizes: &mut Vec<usize>, x: u8, b: &ResBlock, c: usize, h: usize, w: usize| {
            need(sizes, t, c * h * w);
            need(sizes, hb, c * h * w);
            steps.push(PvStep::NormSilu { gi: b.norm1, src: x, dst: t, c, h, w });
            steps.push(PvStep::Conv { wi: b.conv1, src: t, dst: hb, res: None, dil: b.dil, h, w });
            steps.push(PvStep::NormSilu { gi: b.norm2, src: hb, dst: t, c, h, w });
            steps.push(PvStep::Conv { wi: b.conv2, src: t, dst: x, res: Some(x), dil: 1, h, w });
        };
        let (mut h, mut w) = (h, w);
        steps.push(PvStep::Conv { wi: self.conv_in, src, dst: xb(0), res: None, dil: 1, h, w });
        need(sizes, xb(0), self.widths[0] * h * w);
        for i in 0..levels {
            for b in &self.down_blocks[i] {
                block(steps, sizes, xb(i), b, self.widths[i], h, w);
            }
            if i < levels - 1 {
                let c = self.widths[i];
                steps.push(PvStep::Unshuffle { src: xb(i), dst: t, c, h, w });
                need(sizes, t, c * h * w);
                steps.push(PvStep::Conv { wi: self.downs[i], src: t, dst: xb(i + 1), res: None, dil: 1, h: h / 2, w: w / 2 });
                h /= 2;
                w /= 2;
                need(sizes, xb(i + 1), self.widths[i + 1] * h * w);
            }
        }
        for i in (0..levels - 1).rev() {
            let up = &wts.convs[self.ups[i]];
            steps.push(PvStep::Conv { wi: self.ups[i], src: xb(i + 1), dst: t, res: None, dil: 1, h, w });
            need(sizes, t, up.cout * h * w);
            steps.push(PvStep::Shuffle { src: t, dst: xb(i), add: Some(xb(i)), c: up.cout, h, w });
            h *= 2;
            w *= 2;
            for b in &self.up_blocks[i] {
                block(steps, sizes, xb(i), b, self.widths[i], h, w);
            }
        }
        steps.push(PvStep::NormSilu { gi: self.norm_out, src: xb(0), dst: t, c: self.widths[0], h, w });
        steps.push(PvStep::Conv { wi: self.conv_out, src: t, dst, res: None, dil: 1, h, w });
        need(sizes, dst, wts.convs[self.conv_out].cout * h * w);
    }
}

static NEXT_KEY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Whether `model` is a Primal VAE file.
pub fn is_primal_vae(model: &CmfModel) -> bool {
    model.header.arch.arch_name == "primal_vae"
}

impl PrimalVae {
    pub fn from_model(model: &Arc<CmfModel>, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        if !is_primal_vae(model) {
            return Err("not a Primal VAE file (arch_name primal_vae)".into());
        }
        let mut weights = PvWeights::default();
        let encoder = UStack::load(model, "encoder", &mut weights)?;
        let decoder = UStack::load(model, "decoder", &mut weights)?;
        let cin = weights.convs[encoder.conv_in].cin;
        let compression = ((cin / 3) as f64).sqrt().round() as usize;
        if compression * compression * 3 != cin || compression == 0 {
            return Err(format!("primal_vae: conv_in takes {cin} channels, not 3·c²"));
        }
        let latent_channels = weights.convs[encoder.conv_out].cout;
        if weights.convs[decoder.conv_in].cin != latent_channels || weights.convs[decoder.conv_out].cout != cin {
            return Err("primal_vae: encoder and decoder disagree on the latent or pixel channels".into());
        }
        Ok(PrimalVae {
            compression,
            latent_channels,
            weights,
            encoder,
            decoder,
            key: NEXT_KEY.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            pool,
        })
    }

    /// Image sides must be multiples of this (16 for the released models).
    pub fn image_multiple(&self) -> usize {
        self.compression * (1usize << (self.encoder.widths.len() - 1)).max(4)
    }

    fn plan(&self, encode: bool, h: usize, w: usize) -> PvPlan {
        let mut steps = Vec::new();
        let mut sizes = Vec::new();
        let c2 = self.compression;
        let (lh, lw) = (h / c2, w / c2);
        // Buffers: 0..=2 work, 3..3+L-1 skips, then IN and OUT.
        let levels = self.encoder.widths.len();
        let (inp, mid, out) = (3 + levels as u8, 4 + levels as u8, 5 + levels as u8);
        if encode {
            // the image arrives in IN; unshuffle by the compression into MID
            steps.push(PvStep::Unshuffle { src: inp, dst: mid, c: 3, h, w });
            if c2 != 2 {
                unreachable!("released models compress by 2");
            }
            self.encoder.plan(&self.weights, &mut steps, &mut sizes, mid, out, lh, lw);
            let out_elems = self.latent_channels * lh * lw;
            sizes.resize(sizes.len().max(out as usize + 1), 0);
            sizes[inp as usize] = 3 * h * w;
            sizes[mid as usize] = sizes[mid as usize].max(3 * h * w);
            sizes[out as usize] = sizes[out as usize].max(out_elems);
            PvPlan { steps, buf_elems: sizes, input: inp, output: out, out_elems }
        } else {
            self.decoder.plan(&self.weights, &mut steps, &mut sizes, inp, mid, lh, lw);
            steps.push(PvStep::Shuffle { src: mid, dst: out, add: None, c: 3 * c2 * c2, h: lh, w: lw });
            sizes.resize(sizes.len().max(out as usize + 1), 0);
            sizes[inp as usize] = self.latent_channels * lh * lw;
            sizes[out as usize] = 3 * h * w;
            PvPlan { steps, buf_elems: sizes, input: inp, output: out, out_elems: 3 * h * w }
        }
    }

    /// Run a plan on the device, else on the host.
    fn run(&self, plan: &PvPlan, input: &[f32]) -> Result<Vec<f32>, String> {
        let mut out = vec![0f32; plan.out_elems];
        if device_on() && crate::gpu::primal_vae_run(self.key, &self.weights, plan, input, &mut out) {
            return Ok(out);
        }
        let mut bufs: Vec<Vec<f32>> = plan.buf_elems.iter().map(|&n| vec![0f32; n]).collect();
        bufs[plan.input as usize][..input.len()].copy_from_slice(input);
        for s in &plan.steps {
            host_step(&self.weights, s, &mut bufs, self.pool.as_deref());
        }
        out.copy_from_slice(&bufs[plan.output as usize][..plan.out_elems]);
        Ok(out)
    }

    fn check(&self, h: usize, w: usize) -> Result<(), String> {
        let m = self.image_multiple();
        if h == 0 || w == 0 || h % m != 0 || w % m != 0 {
            return Err(format!("image sides must be multiples of {m}, got {w}x{h}"));
        }
        Ok(())
    }

    /// `[3, h, w]` in [-1, 1] → `[latent_channels, h/c, w/c]`.
    pub fn encode(&self, img: &[f32], h: usize, w: usize) -> Result<Vec<f32>, String> {
        self.check(h, w)?;
        if img.len() != 3 * h * w {
            return Err("primal_vae: image is not [3, h, w]".into());
        }
        self.run(&self.plan(true, h, w), img)
    }

    /// `[latent_channels, h/c, w/c]` → `[3, h, w]` (nominally [-1, 1], not
    /// clamped). `h`, `w` are the IMAGE sides.
    pub fn decode(&self, z: &[f32], h: usize, w: usize) -> Result<Vec<f32>, String> {
        self.check(h, w)?;
        let c = self.compression;
        if z.len() != self.latent_channels * (h / c) * (w / c) {
            return Err("primal_vae: latent does not match the image size".into());
        }
        self.run(&self.plan(false, h, w), z)
    }
}

/// `CMF_PRIMAL_VAE_GPU=0` keeps the forward on the host.
fn device_on() -> bool {
    std::env::var("CMF_PRIMAL_VAE_GPU").as_deref() != Ok("0") && crate::gpu::enabled_here()
}

/// Rows of output pixels per host conv band (bounds the im2col panel).
const BAND_PIXELS: usize = 8192;

fn host_conv(cv: &PvConv, src: &[f32], dst: &mut [f32], res: Option<&[f32]>, dil: usize, h: usize, w: usize, pool: Option<&Pool>) {
    let (cin, cout) = (cv.cin, cv.cout);
    let hw = h * w;
    let k = cin * 9;
    let mut panel = vec![0f32; BAND_PIXELS.min(hw) * k];
    let mut yb = vec![0f32; BAND_PIXELS.min(hw) * cout];
    let mut p0 = 0usize;
    while p0 < hw {
        let np = BAND_PIXELS.min(hw - p0);
        let pp = crate::pool::SendMut::new(panel.as_mut_ptr());
        let fill = |lo: usize, hi: usize| {
            for i in lo..hi {
                let p = p0 + i;
                let (y, x) = ((p / w) as isize, (p % w) as isize);
                // SAFETY: disjoint panel rows per worker.
                let row = unsafe { std::slice::from_raw_parts_mut(pp.at(i * k), k) };
                for ci in 0..cin {
                    let plane = &src[ci * hw..(ci + 1) * hw];
                    for tap in 0..9 {
                        let sy = y + (tap / 3) as isize * dil as isize - dil as isize;
                        let sx = x + (tap % 3) as isize * dil as isize - dil as isize;
                        row[ci * 9 + tap] = if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize {
                            0.0
                        } else {
                            plane[sy as usize * w + sx as usize]
                        };
                    }
                }
            }
        };
        match pool {
            Some(p) => p.run_rows(np, &fill),
            None => fill(0, np),
        }
        crate::fcd_ops::gemm_nt_host(&panel[..np * k], &cv.w, &mut yb[..np * cout], np, k, cout, pool);
        for i in 0..np {
            let p = p0 + i;
            for o in 0..cout {
                let r = res.map_or(0.0, |r| r[o * hw + p]);
                dst[o * hw + p] = yb[i * cout + o] + cv.b[o] + r;
            }
        }
        p0 += np;
    }
}

fn host_step(wts: &PvWeights, s: &PvStep, bufs: &mut [Vec<f32>], pool: Option<&Pool>) {
    // Steps never alias src and dst except a conv's residual (read before
    // the write of the same element).
    match *s {
        PvStep::Conv { wi, src, dst, res, dil, h, w } => {
            let cv = &wts.convs[wi];
            let srcv = std::mem::take(&mut bufs[src as usize]);
            let resv = res.map(|r| if r == src { srcv.clone() } else { bufs[r as usize].clone() });
            let mut d = std::mem::take(&mut bufs[dst as usize]);
            if src == dst {
                d = vec![0f32; srcv.len().max(cv.cout * h * w)];
            }
            host_conv(cv, &srcv, &mut d, resv.as_deref(), dil as usize, h, w, pool);
            if src != dst {
                bufs[src as usize] = srcv;
            }
            bufs[dst as usize] = d;
        }
        PvStep::NormSilu { gi, src, dst, c, h, w } => {
            let g = &wts.gammas[gi];
            let hw = h * w;
            let srcv = std::mem::take(&mut bufs[src as usize]);
            let d = &mut bufs[dst as usize];
            for p in 0..hw {
                let mut ss = 0f32;
                for ch in 0..c {
                    let v = srcv[ch * hw + p];
                    ss += v * v;
                }
                let inv = 1.0 / (ss / c as f32 + EPS).sqrt();
                for ch in 0..c {
                    let v = srcv[ch * hw + p] * inv * g[ch];
                    d[ch * hw + p] = v / (1.0 + (-v).exp());
                }
            }
            bufs[src as usize] = srcv;
        }
        PvStep::Unshuffle { src, dst, c, h, w } => {
            let srcv = std::mem::take(&mut bufs[src as usize]);
            let d = &mut bufs[dst as usize];
            let (oh, ow) = (h / 2, w / 2);
            for ch in 0..c {
                for i in 0..2 {
                    for j in 0..2 {
                        let oc = ch * 4 + i * 2 + j;
                        for y in 0..oh {
                            for x in 0..ow {
                                d[(oc * oh + y) * ow + x] = srcv[(ch * h + 2 * y + i) * w + 2 * x + j];
                            }
                        }
                    }
                }
            }
            bufs[src as usize] = srcv;
        }
        PvStep::Shuffle { src, dst, add, c, h, w } => {
            let srcv = std::mem::take(&mut bufs[src as usize]);
            let addv = add.map(|a| bufs[a as usize].clone());
            let d = &mut bufs[dst as usize];
            let (oh, ow, oc) = (2 * h, 2 * w, c / 4);
            for ch in 0..oc {
                for i in 0..2 {
                    for j in 0..2 {
                        let ic = ch * 4 + i * 2 + j;
                        for y in 0..h {
                            for x in 0..w {
                                let o = (ch * oh + 2 * y + i) * ow + 2 * x + j;
                                d[o] = srcv[(ic * h + y) * w + x] + addv.as_ref().map_or(0.0, |a| a[o]);
                            }
                        }
                    }
                }
            }
            bufs[src as usize] = srcv;
        }
    }
}

/// RGB8 `[h, w, 3]` → `[3, h, w]` in [-1, 1].
pub fn rgb8_to_chw(data: &[u8], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * h * w];
    for (i, px) in data.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * h * w + i] = px[c] as f32 / 127.5 - 1.0;
        }
    }
    out
}

/// `[3, h, w]` in [-1, 1] → RGB8 `[h, w, 3]` (clamped, rounded).
pub fn chw_to_rgb8(x: &[f32], h: usize, w: usize) -> Vec<u8> {
    let mut out = vec![0u8; 3 * h * w];
    for i in 0..h * w {
        for c in 0..3 {
            let v = ((x[c * h * w + i] + 1.0) * 127.5).round().clamp(0.0, 255.0);
            out[i * 3 + c] = v as u8;
        }
    }
    out
}

/// PSNR in dB between two `[3, h, w]` images in [-1, 1] (on the 0..255
/// scale after clamping, as an 8-bit comparison).
pub fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mut se = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (((x + 1.0) * 127.5).clamp(0.0, 255.0) as f64, ((y + 1.0) * 127.5).clamp(0.0, 255.0) as f64);
        se += (x - y) * (x - y);
    }
    let mse = se / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bufs_of(sizes: &[usize]) -> Vec<Vec<f32>> {
        sizes.iter().map(|&n| vec![0f32; n]).collect()
    }

    /// `pixel_unshuffle(2)` then `pixel_shuffle(2)` is the identity, and the
    /// channel order is `c·4 + i·2 + j` as in PyTorch.
    #[test]
    fn shuffles_invert_each_other_in_torch_order() {
        let (c, h, w) = (2usize, 4usize, 6usize);
        let x: Vec<f32> = (0..c * h * w).map(|v| v as f32).collect();
        let mut bufs = bufs_of(&[c * h * w, c * h * w, c * h * w]);
        bufs[0].copy_from_slice(&x);
        let wts = PvWeights::default();
        host_step(&wts, &PvStep::Unshuffle { src: 0, dst: 1, c, h, w }, &mut bufs, None);
        // channel 1 of the output = (c 0, i 0, j 1): x[0][2y][2x+1]
        assert_eq!(bufs[1][(1 * 2) * 3], x[1]);
        host_step(&wts, &PvStep::Shuffle { src: 1, dst: 2, add: None, c: 4 * c, h: h / 2, w: w / 2 }, &mut bufs, None);
        assert_eq!(bufs[2], x);
    }

    /// A dilated 3×3 conv with a centre-only kernel copies the input, and
    /// an off-centre tap reads `dil` pixels away with zero padding.
    #[test]
    fn dilated_conv_taps_and_padding() {
        let (h, w) = (5usize, 5usize);
        let x: Vec<f32> = (0..h * w).map(|v| v as f32 + 1.0).collect();
        let mut wt = vec![0f32; 9];
        wt[5] = 1.0; // tap (ky 1, kx 2): x + dil
        let cv = PvConv { w: wt, b: vec![0.5], cin: 1, cout: 1 };
        let mut out = vec![0f32; h * w];
        host_conv(&cv, &x, &mut out, None, 2, h, w, None);
        assert_eq!(out[0], x[2] + 0.5);
        assert_eq!(out[3], 0.5); // x + 2 is outside: zero padding
        let res = vec![1f32; h * w];
        host_conv(&cv, &x, &mut out, Some(&res), 2, h, w, None);
        assert_eq!(out[0], x[2] + 1.5);
    }

    /// Channel RMSNorm × γ then SiLU, per pixel.
    #[test]
    fn norm_silu_is_per_pixel() {
        let (c, h, w) = (2usize, 1usize, 2usize);
        let mut bufs = bufs_of(&[c * h * w, c * h * w]);
        bufs[0].copy_from_slice(&[3.0, 1.0, 4.0, 1.0]); // pixel 0 = (3, 4), pixel 1 = (1, 1)
        let wts = PvWeights { convs: vec![], gammas: vec![vec![1.0, 2.0]] };
        host_step(&wts, &PvStep::NormSilu { gi: 0, src: 0, dst: 1, c, h, w }, &mut bufs, None);
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let inv0 = 1.0 / ((9.0f32 + 16.0) / 2.0 + EPS).sqrt();
        assert!((bufs[1][0] - silu(3.0 * inv0)).abs() < 1e-6);
        assert!((bufs[1][2] - silu(4.0 * inv0 * 2.0)).abs() < 1e-6);
    }
}
