//! SAM3 image inference on CPU (f32) over the sections written by
//! `cortiq-cli sam3-pack` (`trunk.*`, `fpn.*`, `text.*`, `det.*`).
//!
//! Semantics follow transformers 5.20.2 `Sam3Model` (fixtures: run
//! `tools/sam3_ref.py` on the pod): CLIP-L text tower + resizer, ViT trunk
//! with windowed axial RoPE (per-layer baked freqs_cis), SAM2-style FPN
//! neck, DETR encoder/decoder (presence token + 200 queries + log
//! box-RPB, POST-norm decoder layers) and the mask decoder. Geometry
//! prompts are not part of the image path. Batch size 1; `pixels` is a
//! preprocessed [3,1008,1008] tensor.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use cortiq_core::quant::dequant_tensor;
use cortiq_core::{CmfModel, TensorDtype};

use crate::inference::gelu_erf;
use crate::pool::Pool;
use crate::qtensor::QTensor;

/// Additive mask for "key is padded": saturates softmax to 0.
const NEG: f32 = -1e30;

const PIX: usize = 1008;
const PATCH: usize = 14;
const GRID: usize = PIX / PATCH; // 72
const WIN: usize = 24;
const TD: usize = 1024;
const TH: usize = 16;
const TL: usize = 32;
const TI: usize = 4736;
const T_EPS: f32 = 1e-6;
const GLOBAL: [usize; 4] = [7, 15, 23, 31];

const SS: usize = 32;
const SD: usize = 1024;
const SL: usize = 24;
const SI: usize = 4096;
const S_EPS: f32 = 1e-5;
const PROJ: usize = 512;

const FD: usize = 256;
const FI: usize = 2048;
const FH: usize = 8;
const D_EPS: f32 = 1e-6;
const NL: usize = 6;
const NQ: usize = 200;
const MH: usize = 288;

/// One forward's outputs (batch 1), flat row-major f32.
pub struct Sam3Out {
    /// CLIP text projection of the pooled token, [512].
    pub text_embeds: Vec<f32>,
    /// Resizer output fed to the DETR stack, [32 * 256].
    pub text_feats: Vec<f32>,
    /// Trunk features, [5184 * 1024].
    pub trunk_feats: Vec<f32>,
    /// FPN levels 288/144/72/36, each [256 * H * W].
    pub fpn: Vec<Vec<f32>>,
    /// DETR encoder output, [5184 * 256].
    pub enc_hidden: Vec<f32>,
    /// Stacked decoder layer outputs, [6 * 200 * 256].
    pub dec_hidden: Vec<f32>,
    /// Final boxes (xyxy), [200 * 4].
    pub pred_boxes: Vec<f32>,
    /// Instance masks, [200 * 288 * 288].
    pub pred_masks: Vec<f32>,
}

/// Additive attention masks (added to scores before softmax).
#[derive(Clone, Copy)]
enum Mask<'a> {
    None,
    /// One value per key, shared by all queries/heads (padding).
    Key(&'a [f32]),
    /// [n_q, n_k] matrix, shared across heads (CLIP causal).
    Add(&'a [f32]),
    /// [heads, n_q, n_k] per-head bias (box RPB).
    Head(&'a [f32]),
}

pub struct Sam3 {
    model: Arc<CmfModel>,
    pool: Option<Arc<Pool>>,
    /// `fpn.convs` (image detector neck) or `fpn.sam2_convs` (SAM3_FPN).
    fpn_prefix: &'static str,
    /// `trunk.pos_embed` is [1,577,1024]: Meta's leading cls row is
    /// dropped by the HF conversion (SAM3_POS=first keeps rows 0..576).
    pos_skip_cls: bool,
}

impl Sam3 {
    pub fn open(path: &str) -> Result<Self> {
        let model = CmfModel::open(path)
            .with_context(|| format!("open {path}"))?
            .into();
        let fpn_prefix = match std::env::var("SAM3_FPN").as_deref() {
            Ok("sam2_convs") => "fpn.sam2_convs",
            _ => "fpn.convs",
        };
        let pos_skip_cls = std::env::var("SAM3_POS").as_deref() != Ok("first");
        Ok(Self { model, pool: Pool::from_env(), fpn_prefix, pos_skip_cls })
    }

    /// Debug: `"name dtype shape"` lines for every tensor in the CMF
    /// directory (the parity tests filter them by section).
    pub fn directory(&self) -> Vec<String> {
        self.model
            .tensors
            .iter()
            .map(|t| format!("{} {:?} {:?}", t.name, t.dtype, t.shape))
            .collect()
    }

    /// Debug: raw payload bytes of one directory tensor.
    pub fn raw_tensor(&self, name: &str) -> Option<Vec<u8>> {
        let e = self.model.tensor(name)?;
        Some(self.model.entry_bytes(e).to_vec())
    }

    /// Debug: every `q8_row` row scale and every `f16` payload element
    /// whose bit pattern is non-finite, as `"name@idx bad=0x…"` lines.
    pub fn bad_scales(&self) -> Vec<String> {
        let f16_bad = |h: u16| h & 0x7C00 == 0x7C00;
        let mut out = Vec::new();
        for t in self.model.tensors.iter() {
            let bytes = self.model.entry_bytes(t);
            match t.dtype {
                TensorDtype::Q8Row if t.shape.len() == 2 => {
                    let (rows, cols) = (t.shape[0], t.shape[1]);
                    let off = rows * cols;
                    if bytes.len() < off + rows * 2 {
                        out.push(format!("{}@payload short", t.name));
                        continue;
                    }
                    for r in 0..rows {
                        let h = u16::from_le_bytes([bytes[off + r * 2], bytes[off + r * 2 + 1]]);
                        if f16_bad(h) {
                            out.push(format!("{}@row{r} scale_f16=0x{h:04x}", t.name));
                        }
                    }
                }
                TensorDtype::F16 => {
                    for (i, c) in bytes.chunks_exact(2).enumerate() {
                        let h = u16::from_le_bytes([c[0], c[1]]);
                        if f16_bad(h) {
                            out.push(format!("{}@{i} val_f16=0x{h:04x}", t.name));
                            if out.len() > 200 { return out; }
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// `pixels` [3,1008,1008] preprocessed; `ids`/`attn` length 32
    /// (Sam3Processor, padding to max_length=32).
    ///
    /// Debug knobs (unset in normal runs): `SAM3_ONLY=trunk,...` runs only
    /// the listed stages and returns empty tensors for the rest (the
    /// parity test skips while it is set); `SAM3_DUMP_DIR` makes `trunk`
    /// write its stage outputs (`trunk_patch.f32`, `trunk_pre.f32`,
    /// `trunk_b{l}.f32`) for reference bisection; `SAM3_TEXT_DEBUG=1`
    /// prints per-stage finiteness stats of the text tower.
    pub fn forward(&mut self, pixels: &[f32], ids: &[u32], attn: &[u8]) -> Result<Sam3Out> {
        if pixels.len() != 3 * PIX * PIX {
            return Err(anyhow!("pixels {} != 3*{}*{}", pixels.len(), PIX, PIX));
        }
        if ids.len() != SS || attn.len() != SS {
            return Err(anyhow!("ids/attn must be {SS} long"));
        }
        let only = std::env::var("SAM3_ONLY").ok();
        let want = |k: &str| {
            only
                .as_deref()
                .map_or(true, |o| o.split(',').any(|s| s.trim() == k))
        };
        let pool = self.pool.as_deref();
        let (text_embeds, text_feats) = self.text(ids, attn, pool)?;
        let blank = |te: Vec<f32>, tf: Vec<f32>| Sam3Out {
            text_embeds: te,
            text_feats: tf,
            trunk_feats: Vec::new(),
            fpn: Vec::new(),
            enc_hidden: Vec::new(),
            dec_hidden: Vec::new(),
            pred_boxes: Vec::new(),
            pred_masks: Vec::new(),
        };
        if !want("trunk") {
            return Ok(blank(text_embeds, text_feats));
        }
        let trunk_feats = self.trunk(pixels, pool)?;
        if !want("fpn") {
            let mut o = blank(text_embeds, text_feats);
            o.trunk_feats = trunk_feats;
            return Ok(o);
        }
        let fpn = self.fpn(&trunk_feats, pool)?;
        let pos = sine_pos(GRID, GRID, FD / 2); // level-2 (72) encoding, rows
        let vision = chw_to_rows(&fpn[2], GRID, GRID);
        let enc_hidden = self.detr_encoder(&vision, &pos, &text_feats, attn, pool)?;
        if !want("dec") {
            let mut o = blank(text_embeds, text_feats);
            o.trunk_feats = trunk_feats;
            o.fpn = fpn;
            o.enc_hidden = enc_hidden;
            return Ok(o);
        }
        let (dec_hidden, final_refs) = self.detr_decoder(&enc_hidden, &pos, &text_feats, attn, pool)?;
        if !want("masks") {
            let mut o = blank(text_embeds, text_feats);
            o.trunk_feats = trunk_feats;
            o.fpn = fpn;
            o.enc_hidden = enc_hidden;
            o.dec_hidden = dec_hidden.clone();
            o.pred_boxes = xyxy(&final_refs);
            return Ok(o);
        }
        let pred_boxes = xyxy(&final_refs);
        let pred_masks = self.mask_decoder(
            &dec_hidden[(NL - 1) * NQ * FD..],
            &enc_hidden,
            &fpn,
            &text_feats,
            attn,
            pool,
        )?;
        Ok(Sam3Out {
            text_embeds,
            text_feats,
            trunk_feats,
            fpn,
            enc_hidden,
            dec_hidden,
            pred_boxes,
            pred_masks,
        })
    }

    // ── text tower (CLIP-L, pre-norm transformer) ───────────────────

    fn text(&self, ids: &[u32], attn: &[u8], pool: Option<&Pool>) -> Result<(Vec<f32>, Vec<f32>)> {
        let m = &self.model;
        let tok = lin(m, "text.encoder.token_embedding.weight")?;
        let pos = load_vec(m, "text.encoder.positional_embedding")?;
        let mut x = vec![0f32; SS * SD];
        let mut row = vec![0f32; SD];
        for (i, &id) in ids.iter().enumerate() {
            tok.row_f32(id as usize, &mut row);
            let r = &mut x[i * SD..(i + 1) * SD];
            r.copy_from_slice(&row);
            for (v, p) in r.iter_mut().zip(&pos[i * SD..(i + 1) * SD]) {
                *v += *p;
            }
        }
        dbg_stats("text.emb", &x);
        // causal + padding additive mask [SS, SS]
        let mut cm = vec![0f32; SS * SS];
        for q in 0..SS {
            for k in 0..=q {
                if attn[k] == 0 {
                    cm[q * SS + k] = NEG;
                }
            }
            for k in (q + 1)..SS {
                cm[q * SS + k] = NEG;
            }
        }
        for l in 0..SL {
            let b = format!("text.encoder.transformer.resblocks.{l}.");
            let (lw, lb) = (load_vec(m, &format!("{b}ln_1.weight"))?, load_vec(m, &format!("{b}ln_1.bias"))?);
            // Pre-norm: branch reads a normalized CLONE, x keeps the residual.
            let mut h = x.clone();
            layer_norm_rows(&mut h, &lw, &lb, S_EPS);
            let qkv = lin(m, &format!("{b}attn.in_proj_weight"))?;
            let ib = load_vec(m, &format!("{b}attn.in_proj_bias"))?;
            let op = lin(m, &format!("{b}attn.out_proj.weight"))?;
            let ob = load_vec(m, &format!("{b}attn.out_proj.bias"))?;
            let mut ao = vec![0f32; SS * SD];
            run_attn(
                &qkv, &ib, &op, &ob,
                &h, &h, &h, SS, SS, SD, 16, None, Mask::Add(&cm), pool, &mut ao,
            );
            for (a, c) in x.iter_mut().zip(&ao) { *a += *c; }
            let (lw, lb) = (load_vec(m, &format!("{b}ln_2.weight"))?, load_vec(m, &format!("{b}ln_2.bias"))?);
            let mut h = x.clone();
            layer_norm_rows(&mut h, &lw, &lb, S_EPS);
            let f1 = lin(m, &format!("{b}mlp.c_fc.weight"))?;
            let f1b = load_vec(m, &format!("{b}mlp.c_fc.bias"))?;
            let f2 = lin(m, &format!("{b}mlp.c_proj.weight"))?;
            let f2b = load_vec(m, &format!("{b}mlp.c_proj.bias"))?;
            let mut ff = vec![0f32; SS * SI];
            f1.matmat(&h, SS, &mut ff, pool);
            add_bias(&mut ff, SI, &f1b);
            ff.iter_mut().for_each(|v| *v = gelu_erf(*v));
            let mut fo = vec![0f32; SS * SD];
            f2.matmat(&ff, SS, &mut fo, pool);
            add_bias(&mut fo, SD, &f2b);
            for (a, c) in x.iter_mut().zip(&fo) { *a += *c; }
            dbg_stats(&format!("text.l{l}"), &x);
        }
        let (lw, lb) = (load_vec(m, "text.encoder.ln_final.weight")?, load_vec(m, "text.encoder.ln_final.bias")?);
        layer_norm_rows(&mut x, &lw, &lb, S_EPS);
        dbg_stats("text.ln_final", &x);
        if let Some(d) = std::env::var("SAM3_DUMP_DIR").ok() {
            dump_f32(&d, "text_lh", &x);
        }
        // CLIP pools at the first eos token: (ids == eos).argmax()
        let p = ids.iter().position(|&v| v == 49407).ok_or_else(|| anyhow!("no eos token"))?;
        let pooled = &x[p * SD..(p + 1) * SD];
        // text_embeds = pooled @ projection; projection is stored [1024, 512]
        let proj = load_vec(m, "text.encoder.text_projection")?;
        dbg_stats("text.proj", &proj);
        let mut embeds = vec![0f32; PROJ];
        for j in 0..PROJ {
            let mut acc = 0f32;
            for i in 0..SD { acc += pooled[i] * proj[i * PROJ + j]; }
            embeds[j] = acc;
        }
        dbg_stats("text.pooled", pooled);
        dbg_stats("text.embeds", &embeds);
        // DETR text features: Linear(1024 -> 256) over the full sequence
        let rz = lin(m, "text.resizer.weight")?;
        let rb = load_vec(m, "text.resizer.bias")?;
        let mut feats = vec![0f32; SS * FD];
        rz.matmat(&x, SS, &mut feats, pool);
        add_bias(&mut feats, FD, &rb);
        dbg_stats("text.feats", &feats);
        Ok((embeds, feats))
    }

    // ── trunk (ViT, pre-norm layers, windowed/global axial RoPE) ────

    fn trunk(&self, pixels: &[f32], pool: Option<&Pool>) -> Result<Vec<f32>> {
        let m = &self.model;
        let np = GRID * GRID;
        let kc = 3 * PATCH * PATCH;
        let dump = std::env::var("SAM3_DUMP_DIR").ok();
        let dlayers: Option<Vec<usize>> = std::env::var("SAM3_DUMP_LAYERS")
            .ok()
            .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect());
        let w = load_vec(m, "trunk.patch_embed.proj.weight")?; // [1024,3,14,14]
        let mut pv = vec![0f32; np * kc];
        let pvb = pv.as_mut_ptr() as usize;
        let fill = |s: usize, e: usize| {
            let pv = unsafe { std::slice::from_raw_parts_mut(pvb as *mut f32, np * kc) };
            for pt in s..e {
                let (py, px) = (pt / GRID, pt % GRID);
                let d = &mut pv[pt * kc..(pt + 1) * kc];
                let mut o = 0;
                for c in 0..3 {
                    for ky in 0..PATCH {
                        let y = py * PATCH + ky;
                        for kx in 0..PATCH {
                            d[o] = pixels[c * PIX * PIX + y * PIX + px * PATCH + kx];
                            o += 1;
                        }
                    }
                }
            }
        };
        match pool { Some(p) => p.run_rows(np, &fill), None => fill(0, np) };
        let mut x = vec![0f32; np * TD];
        let xb = x.as_mut_ptr() as usize;
        let gemm = |s: usize, e: usize| {
            let x = unsafe { std::slice::from_raw_parts_mut(xb as *mut f32, np * TD) };
            for ch in s..e {
                let wr = &w[ch * kc..(ch + 1) * kc];
                for pt in 0..np {
                    let xv = &pv[pt * kc..(pt + 1) * kc];
                    let mut acc = 0f32;
                    for k in 0..kc { acc += wr[k] * xv[k]; }
                    x[pt * TD + ch] = acc;
                }
            }
        };
        match pool { Some(p) => p.run_rows(TD, &gemm), None => gemm(0, TD) };
        if let Some(d) = &dump { dump_f32(d, "trunk_patch", &x); }
        // position embeddings [1,577,1024]: drop Meta's cls row, tile 24x24 -> 72x72
        let pos = load_vec(m, "trunk.pos_embed")?;
        let base = if self.pos_skip_cls { 1 } else { 0 };
        if pos.len() < (base + 576) * TD { return Err(anyhow!("pos_embed too small")); }
        for pt in 0..np {
            let (py, px) = (pt / GRID, pt % GRID);
            let src = (base + (py % 24) * 24 + px % 24) * TD;
            for d in 0..TD { x[pt * TD + d] += pos[src + d]; }
        }
        let (lw, lb) = (load_vec(m, "trunk.ln_pre.weight")?, load_vec(m, "trunk.ln_pre.bias")?);
        layer_norm_rows(&mut x, &lw, &lb, T_EPS);
        if let Some(d) = &dump { dump_f32(d, "trunk_pre", &x); }

        for l in 0..TL {
            let b = format!("trunk.blocks.{l}.");
            let (nw, nb) = (load_vec(m, &format!("{b}norm1.weight"))?, load_vec(m, &format!("{b}norm1.bias"))?);
            let mut h = x.clone();
            layer_norm_rows(&mut h, &nw, &nb, T_EPS);
            let dbg0 = dump.is_some() && l == 0;
            if let (Some(d), true) = (&dump, dbg0) { dump_f32(d, "b0_ln1", &h); }
            let qkv = lin(m, &format!("{b}attn.qkv.weight"))?;
            let qb = load_vec(m, &format!("{b}attn.qkv.bias"))?;
            let proj = lin(m, &format!("{b}attn.proj.weight"))?;
            let pb = load_vec(m, &format!("{b}attn.proj.bias"))?;
            let fc = load_vec(m, &format!("{b}attn.freqs_cis"))?;
            let attn_out = if GLOBAL.contains(&l) {
                let mut o = vec![0f32; np * TD];
                run_attn(&qkv, &qb, &proj, &pb, &h, &h, &h, np, np, TD, TH, Some(&fc), Mask::None, pool, &mut o);
                o
            } else {
                self.vit_windows(&h, &qkv, &qb, &proj, &pb, &fc, pool)?
            };
            if let (Some(d), true) = (&dump, dbg0) { dump_f32(d, "b0_attn", &attn_out); }
            for (a, c) in x.iter_mut().zip(&attn_out) { *a += *c; }
            if let (Some(d), true) = (&dump, dbg0) { dump_f32(d, "b0_x2", &x); }
            // pre-norm MLP branch: norm a CLONE — x keeps the residual
            // (norming x in place made the block add mlp(ln2(x)) instead
            // of x + mlp(ln2(x))).
            let (nw, nb) = (load_vec(m, &format!("{b}norm2.weight"))?, load_vec(m, &format!("{b}norm2.bias"))?);
            let mut h = x.clone();
            layer_norm_rows(&mut h, &nw, &nb, T_EPS);
            if let (Some(d), true) = (&dump, dbg0) { dump_f32(d, "b0_ln2", &h); }
            let f1 = lin(m, &format!("{b}mlp.fc1.weight"))?;
            let f1b = load_vec(m, &format!("{b}mlp.fc1.bias"))?;
            let f2 = lin(m, &format!("{b}mlp.fc2.weight"))?;
            let f2b = load_vec(m, &format!("{b}mlp.fc2.bias"))?;
            let mut ff = vec![0f32; np * TI];
            f1.matmat(&h, np, &mut ff, pool);
            add_bias(&mut ff, TI, &f1b);
            ff.iter_mut().for_each(|v| *v = gelu_erf(*v));
            let mut fo = vec![0f32; np * TD];
            f2.matmat(&ff, np, &mut fo, pool);
            add_bias(&mut fo, TD, &f2b);
            if let (Some(d), true) = (&dump, dbg0) { dump_f32(d, "b0_mlp", &fo); }
            for (a, c) in x.iter_mut().zip(&fo) { *a += *c; }
            if let (Some(d), true) = (&dump, dlayers.as_ref().map_or(true, |ls| ls.contains(&l))) {
                dump_f32(d, &format!("trunk_b{l}"), &x);
            }
        }
        Ok(x)
    }

    /// Windowed trunk attention: 3x3 windows of 24x24 tokens (freqs [576]).
    fn vit_windows(
        &self,
        h: &[f32],
        qkv: &QTensor,
        qb: &[f32],
        proj: &QTensor,
        pb: &[f32],
        fc: &[f32],
        pool: Option<&Pool>,
    ) -> Result<Vec<f32>> {
        let np = GRID * GRID;
        let nw = GRID / WIN;
        let wl = WIN * WIN;
        let mut wh = vec![0f32; wl * TD];
        let mut ao = vec![0f32; wl * TD];
        let mut out = vec![0f32; np * TD];
        for wy in 0..nw {
            for wx in 0..nw {
                for y in 0..WIN {
                    for x2 in 0..WIN {
                        let s = ((wy * WIN + y) * GRID + wx * WIN + x2) * TD;
                        let d = (y * WIN + x2) * TD;
                        wh[d..d + TD].copy_from_slice(&h[s..s + TD]);
                    }
                }
                run_attn(qkv, qb, proj, pb, &wh, &wh, &wh, wl, wl, TD, TH, Some(fc), Mask::None, pool, &mut ao);
                for y in 0..WIN {
                    for x2 in 0..WIN {
                        let d = ((wy * WIN + y) * GRID + wx * WIN + x2) * TD;
                        let s = (y * WIN + x2) * TD;
                        out[d..d + TD].copy_from_slice(&ao[s..s + TD]);
                    }
                }
            }
        }
        Ok(out)
    }

    // ── FPN neck ────────────────────────────────────────────────────

    /// Four pyramid levels from the trunk map: 288/144/72/36, each
    /// [256, H, W] channels-first.
    fn fpn(&self, trunk: &[f32], pool: Option<&Pool>) -> Result<Vec<Vec<f32>>> {
        let m = &self.model;
        let hidden = rows_to_chw(trunk, GRID, GRID); // [1024,72,72]
        let p = self.fpn_prefix;
        let scales = [4.0f32, 2.0, 1.0, 0.5];
        let mut levels = Vec::with_capacity(4);
        for (i, &s) in scales.iter().enumerate() {
            let (mut ci, mut ch, mut cw) = (TD, GRID, GRID);
            let mut cur = hidden.clone();
            match s {
                4.0 => {
                    let w0 = load_vec(m, &format!("{p}.{i}.dconv_2x2_0.weight"))?;
                    let b0 = load_vec(m, &format!("{p}.{i}.dconv_2x2_0.bias"))?;
                    cur = convt2x2(&w0, &b0, &cur, ci, ch, cw, pool);
                    (ch, cw, ci) = (ch * 2, cw * 2, b0.len());
                    cur.iter_mut().for_each(|v| *v = gelu_erf(*v));
                    let w1 = load_vec(m, &format!("{p}.{i}.dconv_2x2_1.weight"))?;
                    let b1 = load_vec(m, &format!("{p}.{i}.dconv_2x2_1.bias"))?;
                    cur = convt2x2(&w1, &b1, &cur, ci, ch, cw, pool);
                    (ch, cw, ci) = (ch * 2, cw * 2, b1.len());
                }
                2.0 => {
                    let w0 = load_vec(m, &format!("{p}.{i}.dconv_2x2.weight"))?;
                    let b0 = load_vec(m, &format!("{p}.{i}.dconv_2x2.bias"))?;
                    cur = convt2x2(&w0, &b0, &cur, ci, ch, cw, pool);
                    (ch, cw, ci) = (ch * 2, cw * 2, b0.len());
                }
                1.0 => {}
                _ => {
                    cur = maxpool2(&cur, ci, ch, cw);
                    (ch, cw) = (ch / 2, cw / 2);
                }
            }
            let w1 = load_vec(m, &format!("{p}.{i}.conv_1x1.weight"))?;
            let b1 = load_vec(m, &format!("{p}.{i}.conv_1x1.bias"))?;
            cur = conv2d(&w1, &b1, &cur, ci, ch, cw, 1, 1, 1, 0, pool);
            ci = b1.len();
            let w3 = load_vec(m, &format!("{p}.{i}.conv_3x3.weight"))?;
            let b3 = load_vec(m, &format!("{p}.{i}.conv_3x3.bias"))?;
            cur = conv2d(&w3, &b3, &cur, ci, ch, cw, 3, 3, 1, 1, pool);
            levels.push(cur);
        }
        Ok(levels)
    }

    // ── DETR encoder (pre-norm layers) ──────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn detr_encoder(
        &self,
        vision: &[f32],
        pos: &[f32],
        text: &[f32],
        am: &[u8],
        pool: Option<&Pool>,
    ) -> Result<Vec<f32>> {
        let m = &self.model;
        let n = vision.len() / FD;
        let mut xs = vision.to_vec();
        let mut km = vec![0f32; SS];
        for (k, &v) in am.iter().enumerate() { km[k] = if v == 0 { NEG } else { 0.0 }; }
        for l in 0..NL {
            let b = format!("det.transformer.encoder.layers.{l}.");
            // pre-norm self attention: q=k=normed+pos, v=normed
            let (nw, nb) = (load_vec(m, &format!("{b}norm1.weight"))?, load_vec(m, &format!("{b}norm1.bias"))?);
            let mut h = xs.clone();
            layer_norm_rows(&mut h, &nw, &nb, D_EPS);
            let hp: Vec<f32> = h
                .chunks_exact(FD)
                .zip(pos.chunks_exact(FD))
                .flat_map(|(a, c)| a.iter().zip(c).map(|(x, y)| x + y))
                .collect();
            let qkv = lin(m, &format!("{b}self_attn.in_proj_weight"))?;
            let ib = load_vec(m, &format!("{b}self_attn.in_proj_bias"))?;
            let op = lin(m, &format!("{b}self_attn.out_proj.weight"))?;
            let ob = load_vec(m, &format!("{b}self_attn.out_proj.bias"))?;
            let mut ao = vec![0f32; n * FD];
            run_attn(&qkv, &ib, &op, &ob, &hp, &hp, &h, n, n, FD, FH, None, Mask::None, pool, &mut ao);
            for (a, c) in xs.iter_mut().zip(&ao) { *a += *c; }
            // cross attention to the text features (query has no pos add)
            let (nw, nb) = (load_vec(m, &format!("{b}norm2.weight"))?, load_vec(m, &format!("{b}norm2.bias"))?);
            let mut h = xs.clone();
            layer_norm_rows(&mut h, &nw, &nb, D_EPS);
            let qkv = lin(m, &format!("{b}cross_attn_image.in_proj_weight"))?;
            let ib = load_vec(m, &format!("{b}cross_attn_image.in_proj_bias"))?;
            let op = lin(m, &format!("{b}cross_attn_image.out_proj.weight"))?;
            let ob = load_vec(m, &format!("{b}cross_attn_image.out_proj.bias"))?;
            let mut ao = vec![0f32; n * FD];
            run_attn(&qkv, &ib, &op, &ob, &h, text, text, n, SS, FD, FH, None, Mask::Key(&km), pool, &mut ao);
            for (a, c) in xs.iter_mut().zip(&ao) { *a += *c; }
            // relu MLP
            let (nw, nb) = (load_vec(m, &format!("{b}norm3.weight"))?, load_vec(m, &format!("{b}norm3.bias"))?);
            let mut h = xs.clone();
            layer_norm_rows(&mut h, &nw, &nb, D_EPS);
            let l1 = lin(m, &format!("{b}linear1.weight"))?;
            let l1b = load_vec(m, &format!("{b}linear1.bias"))?;
            let l2 = lin(m, &format!("{b}linear2.weight"))?;
            let l2b = load_vec(m, &format!("{b}linear2.bias"))?;
            let mut mid = vec![0f32; n * FI];
            l1.matmat(&h, n, &mut mid, pool);
            add_bias(&mut mid, FI, &l1b);
            mid.iter_mut().for_each(|v| *v = v.max(0.0));
            let mut fo = vec![0f32; n * FD];
            l2.matmat(&mid, n, &mut fo, pool);
            add_bias(&mut fo, FD, &l2b);
            for (a, c) in xs.iter_mut().zip(&fo) { *a += *c; }
        }
        Ok(xs)
    }

    // ── DETR decoder (POST-norm layers) ─────────────────────────────

    /// Returns (stacked layer outputs [6*200*256], final reference boxes
    /// in cxcywh — the box_head refinement of the last layer, which
    /// equals Sam3Model's `all_pred_boxes[-1]` pre-xyxy conversion).
    fn detr_decoder(
        &self,
        enc: &[f32],
        pos: &[f32],
        text: &[f32],
        am: &[u8],
        pool: Option<&Pool>,
    ) -> Result<(Vec<f32>, Vec<[f32; 4]>)> {
        let m = &self.model;
        let n = enc.len() / FD;
        let db = "det.transformer.decoder.";
        let query = load_vec(m, &format!("{db}query_embed.weight"))?;
        let presence = load_vec(m, &format!("{db}presence_token.weight"))?;
        let ref0 = load_vec(m, &format!("{db}reference_points.weight"))?;
        let mut hidden = vec![0f32; (NQ + 1) * FD];
        hidden[..FD].copy_from_slice(&presence);
        hidden[FD..].copy_from_slice(&query);
        let mut refs: Vec<[f32; 4]> = ref0
            .chunks_exact(4)
            .map(|c| [sigm(c[0]), sigm(c[1]), sigm(c[2]), sigm(c[3])])
            .collect();
        let mut km = vec![0f32; SS];
        for (k, &v) in am.iter().enumerate() { km[k] = if v == 0 { NEG } else { 0.0 }; }
        // heads (all loaded once, biases included)
        let rh0 = lin(m, &format!("{db}ref_point_head.layers.0.weight"))?;
        let rh0b = load_vec(m, &format!("{db}ref_point_head.layers.0.bias"))?;
        let rh1 = lin(m, &format!("{db}ref_point_head.layers.1.weight"))?;
        let rh1b = load_vec(m, &format!("{db}ref_point_head.layers.1.bias"))?;
        let rx = DecoderMlp::load(m, &format!("{db}boxRPB_embed_x."))?;
        let ry = DecoderMlp::load(m, &format!("{db}boxRPB_embed_y."))?;
        let bbox = DecoderMlp3::load3(m, &format!("{db}bbox_embed."))?;
        let olnw = load_vec(m, &format!("{db}norm.weight"))?;
        let olnb = load_vec(m, &format!("{db}norm.bias"))?;
        let key = add_rows(enc, pos); // enc + pos, [n, FD]
        let mut inter: Vec<Vec<f32>> = Vec::with_capacity(NL);
        for l in 0..NL {
            let b = format!("{db}layers.{l}.");
            // query position from the sine encoding of the reference boxes
            let sine = sine_boxes(&refs);
            let mut qp = vec![0f32; NQ * FD];
            rh0.matmat(&sine, NQ, &mut qp, pool);
            add_bias(&mut qp, FD, &rh0b);
            qp.iter_mut().for_each(|v| *v = v.max(0.0)); // DecoderMLP: relu between
            let mut qp2 = vec![0f32; NQ * FD];
            rh1.matmat(&qp, NQ, &mut qp2, pool);
            add_bias(&mut qp2, FD, &rh1b);
            let qp = qp2;
            let mut qp_full = vec![0f32; (NQ + 1) * FD];
            qp_full[FD..].copy_from_slice(&qp);
            let rpb = rpb_bias(&refs, &rx, &ry, n, pool)?;
            // self attention: q=k=h+qp (pad row stays qp=0), v=h ; POST-norm
            let hqp = add_rows(&hidden, &qp_full);
            let qkv = lin(m, &format!("{b}self_attn.in_proj_weight"))?;
            let ib = load_vec(m, &format!("{b}self_attn.in_proj_bias"))?;
            let op = lin(m, &format!("{b}self_attn.out_proj.weight"))?;
            let ob = load_vec(m, &format!("{b}self_attn.out_proj.bias"))?;
            let mut ao = vec![0f32; (NQ + 1) * FD];
            run_attn(&qkv, &ib, &op, &ob, &hqp, &hqp, &hidden, NQ + 1, NQ + 1, FD, FH, None, Mask::None, pool, &mut ao);
            for (a, c) in hidden.iter_mut().zip(&ao) { *a += *c; }
            let (nw, nb) = (load_vec(m, &format!("{b}norm1.weight"))?, load_vec(m, &format!("{b}norm1.bias"))?);
            layer_norm_rows(&mut hidden, &nw, &nb, D_EPS);
            // text cross attention
            let hqp = add_rows(&hidden, &qp_full);
            let qkv = lin(m, &format!("{b}ca_text.in_proj_weight"))?;
            let ib = load_vec(m, &format!("{b}ca_text.in_proj_bias"))?;
            let op = lin(m, &format!("{b}ca_text.out_proj.weight"))?;
            let ob = load_vec(m, &format!("{b}ca_text.out_proj.bias"))?;
            let mut ao = vec![0f32; (NQ + 1) * FD];
            run_attn(&qkv, &ib, &op, &ob, &hqp, text, text, NQ + 1, SS, FD, FH, None, Mask::Key(&km), pool, &mut ao);
            for (a, c) in hidden.iter_mut().zip(&ao) { *a += *c; }
            let (nw, nb) = (load_vec(m, &format!("{b}catext_norm.weight"))?, load_vec(m, &format!("{b}catext_norm.bias"))?);
            layer_norm_rows(&mut hidden, &nw, &nb, D_EPS);
            // vision cross attention (keys = enc + pos, values = enc, RPB bias)
            let hqp = add_rows(&hidden, &qp_full);
            let qkv = lin(m, &format!("{b}cross_attn.in_proj_weight"))?;
            let ib = load_vec(m, &format!("{b}cross_attn.in_proj_bias"))?;
            let op = lin(m, &format!("{b}cross_attn.out_proj.weight"))?;
            let ob = load_vec(m, &format!("{b}cross_attn.out_proj.bias"))?;
            let mut ao = vec![0f32; (NQ + 1) * FD];
            run_attn(&qkv, &ib, &op, &ob, &hqp, &key, enc, NQ + 1, n, FD, FH, None, Mask::Head(&rpb), pool, &mut ao);
            for (a, c) in hidden.iter_mut().zip(&ao) { *a += *c; }
            let (nw, nb) = (load_vec(m, &format!("{b}norm2.weight"))?, load_vec(m, &format!("{b}norm2.bias"))?);
            layer_norm_rows(&mut hidden, &nw, &nb, D_EPS);
            // relu MLP (Sam3MLP)
            let l1 = lin(m, &format!("{b}linear1.weight"))?;
            let l1b = load_vec(m, &format!("{b}linear1.bias"))?;
            let l2 = lin(m, &format!("{b}linear2.weight"))?;
            let l2b = load_vec(m, &format!("{b}linear2.bias"))?;
            let mut mid = vec![0f32; (NQ + 1) * FI];
            l1.matmat(&hidden, NQ + 1, &mut mid, pool);
            add_bias(&mut mid, FI, &l1b);
            mid.iter_mut().for_each(|v| *v = v.max(0.0));
            let mut fo = vec![0f32; (NQ + 1) * FD];
            l2.matmat(&mid, NQ + 1, &mut fo, pool);
            add_bias(&mut fo, FD, &l2b);
            for (a, c) in hidden.iter_mut().zip(&fo) { *a += *c; }
            let (nw, nb) = (load_vec(m, &format!("{b}norm3.weight"))?, load_vec(m, &format!("{b}norm3.bias"))?);
            layer_norm_rows(&mut hidden, &nw, &nb, D_EPS);
            // box refine + intermediate (query rows only, output_layer_norm)
            let mut qh = hidden[FD..].to_vec();
            layer_norm_rows(&mut qh, &olnw, &olnb, D_EPS);
            inter.push(qh.clone());
            let d = bbox.forward(&qh, NQ, pool);
            for (r, dd) in refs.iter_mut().zip(d.chunks_exact(4)) {
                for k in 0..4 { r[k] = sigm(inv_sigmoid(r[k]) + dd[k]); }
            }
        }
        let dec = inter.iter().flat_map(|v| v.iter().copied()).collect();
        Ok((dec, refs))
    }

    // ── mask decoder ────────────────────────────────────────────────

    /// Prompt cross-attention over the encoder states, pixel decoder
    /// (nearest upsample + sum + conv3x3 + GroupNorm + ReLU), then
    /// pred_masks[q] = einsum(mask_embed[q], instance_embed).
    #[allow(clippy::too_many_arguments)]
    fn mask_decoder(
        &self,
        queries: &[f32],
        enc: &[f32],
        fpn: &[Vec<f32>],
        text: &[f32],
        am: &[u8],
        pool: Option<&Pool>,
    ) -> Result<Vec<f32>> {
        let m = &self.model;
        let db = "det.seghead.";
        let n = enc.len() / FD;
        let mut km = vec![0f32; SS];
        for (k, &v) in am.iter().enumerate() { km[k] = if v == 0 { NEG } else { 0.0 }; }
        let nw = load_vec(m, &format!("{db}cross_attn_norm.weight"))?;
        let nb = load_vec(m, &format!("{db}cross_attn_norm.bias"))?;
        let mut h = enc.to_vec();
        layer_norm_rows(&mut h, &nw, &nb, D_EPS);
        let qkv = lin(m, &format!("{db}cross_attend_prompt.in_proj_weight"))?;
        let ib = load_vec(m, &format!("{db}cross_attend_prompt.in_proj_bias"))?;
        let op = lin(m, &format!("{db}cross_attend_prompt.out_proj.weight"))?;
        let ob = load_vec(m, &format!("{db}cross_attend_prompt.out_proj.bias"))?;
        let mut e = enc.to_vec();
        let mut ao = vec![0f32; n * FD];
        run_attn(&qkv, &ib, &op, &ob, &h, text, text, n, SS, FD, FH, None, Mask::Key(&km), pool, &mut ao);
        for (a, c) in e.iter_mut().zip(&ao) { *a += *c; }
        // pixel decoder: encoder states replace level 2 (72), then upsample
        let l72 = rows_to_chw(&e, GRID, GRID);
        let sizes = [(144usize, 144usize), (288, 288)];
        let mut prev = l72;
        for (li, &(sh, sw)) in sizes.iter().enumerate() {
            prev = upsample_nearest(&prev, FD, sh / 2, sh / 2, sh, sw);
            let fw = &fpn[1 - li]; // reversed(fpn[:-1]): 144 then 288
            for (a, b) in prev.iter_mut().zip(fw.iter()) { *a += *b; }
            let cw = load_vec(m, &format!("{db}pixel_decoder.conv_layers.{li}.weight"))?;
            let cb = load_vec(m, &format!("{db}pixel_decoder.conv_layers.{li}.bias"))?;
            prev = conv2d(&cw, &cb, &prev, FD, sh, sw, 3, 3, 1, 1, pool);
            let gw = load_vec(m, &format!("{db}pixel_decoder.norms.{li}.weight"))?;
            let gb = load_vec(m, &format!("{db}pixel_decoder.norms.{li}.bias"))?;
            group_norm(&mut prev, 8, &gw, &gb, 1e-5);
            prev.iter_mut().for_each(|v| *v = v.max(0.0));
        }
        let iw = load_vec(m, &format!("{db}instance_seg_head.weight"))?;
        let ib2 = load_vec(m, &format!("{db}instance_seg_head.bias"))?;
        let inst = conv2d(&iw, &ib2, &prev, FD, MH, MH, 1, 1, 1, 0, pool);
        // mask embeddings from the (already output-normed) queries
        let mut q = queries.to_vec();
        for i in 0..3 {
            let w = lin(m, &format!("{db}mask_predictor.mask_embed.layers.{i}.weight"))?;
            let b = load_vec(m, &format!("{db}mask_predictor.mask_embed.layers.{i}.bias"))?;
            let mut o = vec![0f32; NQ * FD];
            w.matmat(&q, NQ, &mut o, pool);
            add_bias(&mut o, FD, &b);
            if i < 2 { o.iter_mut().for_each(|v| *v = v.max(0.0)); }
            q = o;
        }
        let hw = MH * MH;
        let mut out = vec![0f32; NQ * hw];
        let ob2 = out.as_mut_ptr() as usize;
        let run = |s: usize, e: usize| {
            let o = unsafe { std::slice::from_raw_parts_mut(ob2 as *mut f32, NQ * hw) };
            for qi in s..e {
                let qq = &q[qi * FD..(qi + 1) * FD];
                let dst = &mut o[qi * hw..(qi + 1) * hw];
                for c in 0..FD {
                    let wgt = qq[c];
                    if wgt == 0.0 { continue; }
                    let ic = &inst[c * hw..(c + 1) * hw];
                    for p in 0..hw { dst[p] += wgt * ic[p]; }
                }
            }
        };
        match pool { Some(p) => p.run_rows(NQ, &run), None => run(0, NQ) }
        Ok(out)
    }
}

// ───────────────────── model pieces ─────────────────────

/// Sam3DecoderMLP, 2 layers: relu(l1), l2 (with biases).
struct DecoderMlp {
    w0: QTensor,
    b0: Vec<f32>,
    w1: QTensor,
    b1: Vec<f32>,
}

impl DecoderMlp {
    fn load(m: &Arc<CmfModel>, prefix: &str) -> Result<Self> {
        Ok(Self {
            w0: lin(m, &format!("{prefix}layers.0.weight"))?,
            b0: load_vec(m, &format!("{prefix}layers.0.bias"))?,
            w1: lin(m, &format!("{prefix}layers.1.weight"))?,
            b1: load_vec(m, &format!("{prefix}layers.1.bias"))?,
        })
    }
    fn forward(&self, x: &[f32], rows: usize, pool: Option<&Pool>, out_w: usize) -> Vec<f32> {
        let mut h = vec![0f32; rows * self.b0.len()];
        self.w0.matmat(x, rows, &mut h, pool);
        add_bias(&mut h, self.b0.len(), &self.b0);
        h.iter_mut().for_each(|v| *v = v.max(0.0));
        let mut o = vec![0f32; rows * out_w];
        self.w1.matmat(&h, rows, &mut o, pool);
        add_bias(&mut o, out_w, &self.b1);
        o
    }
}

/// Sam3DecoderMLP, 3 layers: relu(l1), relu(l2), l3 (box head).
struct DecoderMlp3 {
    w0: QTensor,
    b0: Vec<f32>,
    w1: QTensor,
    b1: Vec<f32>,
    w2: QTensor,
    b2: Vec<f32>,
}

impl DecoderMlp3 {
    fn load3(m: &Arc<CmfModel>, prefix: &str) -> Result<Self> {
        Ok(Self {
            w0: lin(m, &format!("{prefix}layers.0.weight"))?,
            b0: load_vec(m, &format!("{prefix}layers.0.bias"))?,
            w1: lin(m, &format!("{prefix}layers.1.weight"))?,
            b1: load_vec(m, &format!("{prefix}layers.1.bias"))?,
            w2: lin(m, &format!("{prefix}layers.2.weight"))?,
            b2: load_vec(m, &format!("{prefix}layers.2.bias"))?,
        })
    }
    fn forward(&self, x: &[f32], rows: usize, pool: Option<&Pool>) -> Vec<f32> {
        let mut d = vec![0f32; rows * self.b0.len()];
        self.w0.matmat(x, rows, &mut d, pool);
        add_bias(&mut d, self.b0.len(), &self.b0);
        d.iter_mut().for_each(|v| *v = v.max(0.0));
        let mut d2 = vec![0f32; rows * self.b1.len()];
        self.w1.matmat(&d, rows, &mut d2, pool);
        add_bias(&mut d2, self.b1.len(), &self.b1);
        d2.iter_mut().for_each(|v| *v = v.max(0.0));
        let mut o = vec![0f32; rows * self.b2.len()];
        self.w2.matmat(&d2, rows, &mut o, pool);
        add_bias(&mut o, self.b2.len(), &self.b2);
        o
    }
}

fn sigm(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn inv_sigmoid(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    ((x.max(1e-3)) / ((1.0 - x).max(1e-3))).ln()
}

fn xyxy(boxes: &[[f32; 4]]) -> Vec<f32> {
    let mut out = vec![0f32; boxes.len() * 4];
    for (i, b) in boxes.iter().enumerate() {
        out[i * 4] = b[0] - 0.5 * b[2];
        out[i * 4 + 1] = b[1] - 0.5 * b[3];
        out[i * 4 + 2] = b[0] + 0.5 * b[2];
        out[i * 4 + 3] = b[1] + 0.5 * b[3];
    }
    out
}

fn lin(m: &Arc<CmfModel>, name: &str) -> Result<QTensor> {
    QTensor::from_model(m, name).map_err(|e| anyhow!(e))
}

/// Dequantize a tensor to flat f32 (norms, biases, tables, convs).
fn load_vec(m: &Arc<CmfModel>, name: &str) -> Result<Vec<f32>> {
    let e = m.tensor(name).ok_or_else(|| anyhow!("tensor '{name}' not found"))?;
    if matches!(e.dtype, TensorDtype::U8) { return Err(anyhow!("'{name}' is u8")); }
    let bytes = m.entry_bytes(e);
    let mut dst = vec![0f32; e.n_elems()];
    dequant_tensor(e, bytes, &mut dst).map_err(|er| anyhow!("{name}: {er}"))?;
    Ok(dst)
}

/// Row-major [rows, cols] += bias (broadcast over rows).
fn add_bias(out: &mut [f32], cols: usize, bias: &[f32]) {
    debug_assert_eq!(bias.len(), cols);
    for r in out.chunks_exact_mut(cols) {
        for (v, b) in r.iter_mut().zip(bias) { *v += *b; }
    }
}

/// LayerNorm over rows of `x` (row length = w.len()).
fn layer_norm_rows(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let d = w.len();
    for r in x.chunks_exact_mut(d) {
        let mean = r.iter().sum::<f32>() / d as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (v, (&ww, &bb)) in r.iter_mut().zip(w.iter().zip(b.iter())) {
            *v = (*v - mean) * inv * ww + bb;
        }
    }
}

/// Axial 2D RoPE in place on a fused QKV block: rotates the q part
/// (dims [0, dim)) and the k part ([dim, 2*dim)); v untouched. freqs_cis
/// = [n, 32, 2] = (cos, sin) per pair; pairs 0..15 from w-position,
/// 16..31 from h-position; per head block of 64 dims (16 heads share the
/// same position-dependent table).
fn apply_rope(buf: &mut [f32], n: usize, stride: usize, width: usize, fc: &[f32]) {
    let npairs = 32;
    debug_assert_eq!(fc.len(), n * npairs * 2);
    let blocks = width / 64;
    for i in 0..n {
        let f = &fc[i * npairs * 2..(i + 1) * npairs * 2];
        let row = &mut buf[i * stride..i * stride + width];
        for b in 0..blocks {
            let base = b * 64;
            for j in 0..npairs {
                let (c, s) = (f[j * 2], f[j * 2 + 1]);
                let (a, d) = (row[base + j * 2], row[base + j * 2 + 1]);
                row[base + j * 2] = a * c - d * s;
                row[base + j * 2 + 1] = d * c + a * s;
            }
        }
    }
}

/// Multi-head attention for one sequence: fused QKV projection
/// (`w_qkv` rows = 3*dim laid out [q | k | v] per row, `qb` its bias),
/// optional axial RoPE on q/k, additive mask, softmax, output projection
/// (`w_o` + `ob`). `out` is [n_q, dim]; scale = head_dim^-0.5.
#[allow(clippy::too_many_arguments)]
fn run_attn(
    w_qkv: &QTensor,
    qb: &[f32],
    w_o: &QTensor,
    ob: &[f32],
    xs_q: &[f32],
    xs_k: &[f32],
    xs_v: &[f32],
    n_q: usize,
    n_k: usize,
    dim: usize,
    heads: usize,
    rope: Option<&[f32]>,
    mask: Mask,
    pool: Option<&Pool>,
    out: &mut [f32],
) {
    let h3 = dim * 3;
    let hd = dim / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut qkv_q = vec![0f32; n_q * h3];
    w_qkv.matmat(xs_q, n_q, &mut qkv_q, pool);
    add_bias(&mut qkv_q, h3, qb);
    let same_k = xs_k.as_ptr() == xs_q.as_ptr();
    let mut kbuf = Vec::new();
    if !same_k {
        kbuf = vec![0f32; n_k * h3];
        w_qkv.matmat(xs_k, n_k, &mut kbuf, pool);
        add_bias(&mut kbuf, h3, qb);
    }
    let same_v = xs_v.as_ptr() == xs_q.as_ptr();
    let mut vbuf = Vec::new();
    if !same_v {
        vbuf = vec![0f32; n_k * h3];
        w_qkv.matmat(xs_v, n_k, &mut vbuf, pool);
        add_bias(&mut vbuf, h3, qb);
    }
    if let Some(fc) = rope {
        // rotate q (and k when they share the buffer) over the first 2*dim dims
        apply_rope(&mut qkv_q, n_q, h3, dim * 2, fc);
        if !same_k {
            apply_rope(&mut kbuf, n_k, h3, dim * 2, fc);
        }
    }
    let qkv_k: &[f32] = if same_k { &qkv_q } else { &kbuf };
    let qkv_v: &[f32] = if same_v { &qkv_q } else { &vbuf };
    let mut attn = vec![0f32; n_q * dim];
    let ab = attn.as_mut_ptr() as usize;
    let run = |s: usize, e: usize| {
        let a = unsafe { std::slice::from_raw_parts_mut(ab as *mut f32, n_q * dim) };
        let mut scores = vec![0f32; n_k];
        for i in s..e {
            let qrow = &qkv_q[i * h3..i * h3 + dim];
            for t in 0..heads {
                let qs = &qrow[t * hd..(t + 1) * hd];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..n_k {
                    let kr = &qkv_k[j * h3 + dim + t * hd..j * h3 + dim + (t + 1) * hd];
                    let mut v = qs.iter().zip(kr).fold(0.0f32, |c, (a, b)| c + a * b) * scale;
                    v += match mask {
                        Mask::None => 0.0,
                        Mask::Key(km) => km[j],
                        Mask::Add(am) => am[i * n_k + j],
                        Mask::Head(hm) => hm[(t * n_q + i) * n_k + j],
                    };
                    scores[j] = v;
                    if v > mx { mx = v; }
                }
                // Stable softmax: the trunk's ViT scores reach ~155, so a
                // raw exp() overflows to inf for the row-max and the sum
                // turns inf → inv = 0 → the whole attention row silently
                // collapses to zeros. Subtract the row max (`mx`).
                let mut sum = 0.0f32;
                for sv in scores.iter_mut().take(n_k) {
                    *sv = (*sv - mx).exp();
                    sum += *sv;
                }
                let inv = 1.0 / sum;
                let orow = &mut a[i * dim + t * hd..i * dim + (t + 1) * hd];
                for j in 0..n_k {
                    let wgt = scores[j] * inv;
                    if wgt == 0.0 { continue; }
                    let vr = &qkv_v[j * h3 + 2 * dim + t * hd..j * h3 + 2 * dim + (t + 1) * hd];
                    for (o, vv) in orow.iter_mut().zip(vr) { *o += wgt * vv; }
                }
            }
        }
    };
    match pool { Some(p) => p.run_rows(n_q, &run), None => run(0, n_q) }
    w_o.matmat(&attn, n_q, out, pool);
    add_bias(out, dim, ob);
}

// ─────────── convolutions (NCHW f32, explicit shapes) ───────────

fn conv2d(
    w: &[f32],
    b: &[f32],
    x: &[f32],
    ci: usize,
    h: usize,
    wid: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    pool: Option<&Pool>,
) -> Vec<f32> {
    let co = b.len();
    debug_assert_eq!(w.len(), co * ci * kh * kw);
    debug_assert_eq!(x.len(), ci * h * wid);
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (wid + 2 * pad - kw) / stride + 1;
    let mut out = vec![0f32; co * oh * ow];
    let ob = out.as_mut_ptr() as usize;
    let run = |s: usize, e: usize| {
        let o = unsafe { std::slice::from_raw_parts_mut(ob as *mut f32, co * oh * ow) };
        for idx in s..e {
            let c_o = idx / (oh * ow);
            let rem = idx % (oh * ow);
            let (oy, ox) = (rem / ow, rem % ow);
            let mut acc = b[c_o];
            for c in 0..ci {
                for dy in 0..kh {
                    let iy = oy * stride + dy;
                    if iy < pad || iy - pad >= h { continue; }
                    for dx in 0..kw {
                        let ix = ox * stride + dx;
                        if ix < pad || ix - pad >= wid { continue; }
                        acc += w[((c_o * ci + c) * kh + dy) * kw + dx]
                            * x[(c * h + iy - pad) * wid + (ix - pad)];
                    }
                }
            }
            o[idx] = acc;
        }
    };
    match pool { Some(p) => p.run_rows(co * oh * ow, &run), None => run(0, co * oh * ow) }
    out
}

/// ConvTranspose2d(k=2, s=2): w is [ci, co, 2, 2]; x is [ci, h, wid].
fn convt2x2(w: &[f32], b: &[f32], x: &[f32], ci: usize, h: usize, wid: usize, pool: Option<&Pool>) -> Vec<f32> {
    let co = b.len();
    debug_assert_eq!(w.len(), ci * co * 4);
    let (oh, ow) = (h * 2, wid * 2);
    let mut out = vec![0f32; co * oh * ow];
    let ob = out.as_mut_ptr() as usize;
    let run = |s: usize, e: usize| {
        let o = unsafe { std::slice::from_raw_parts_mut(ob as *mut f32, co * oh * ow) };
        for idx in s..e {
            let c_o = idx / (oh * ow);
            let rem = idx % (oh * ow);
            let (oy, ox) = (rem / ow, rem % ow);
            let (kh, kx) = (oy & 1, ox & 1);
            let (iy, ix) = (oy >> 1, ox >> 1);
            let mut acc = b[c_o];
            for c in 0..ci {
                acc += w[((c * co + c_o) * 2 + kh) * 2 + kx] * x[(c * h + iy) * wid + ix];
            }
            o[idx] = acc;
        }
    };
    match pool { Some(p) => p.run_rows(co * oh * ow, &run), None => run(0, co * oh * ow) }
    out
}

fn maxpool2(x: &[f32], ci: usize, h: usize, wid: usize) -> Vec<f32> {
    let (oh, ow) = (h / 2, wid / 2);
    let mut out = vec![0f32; ci * oh * ow];
    for c in 0..ci {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut mx = f32::NEG_INFINITY;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let v = x[(c * h + oy * 2 + dy) * wid + ox * 2 + dx];
                        if v > mx { mx = v; }
                    }
                }
                out[(c * oh + oy) * ow + ox] = mx;
            }
        }
    }
    out
}

/// Nearest-neighbour upsample of [ci, h, wid] to [ci, sh, sw].
fn upsample_nearest(x: &[f32], ci: usize, h: usize, wid: usize, sh: usize, sw: usize) -> Vec<f32> {
    let mut out = vec![0f32; ci * sh * sw];
    for c in 0..ci {
        for oy in 0..sh {
            let iy = oy * h / sh;
            for ox in 0..sw {
                let ix = ox * wid / sw;
                out[(c * sh + oy) * sw + ox] = x[(c * h + iy) * wid + ix];
            }
        }
    }
    out
}

/// GroupNorm(eps=1e-5) over channels of [c, h, w].
fn group_norm(x: &mut [f32], groups: usize, w: &[f32], b: &[f32], eps: f32) {
    let co = w.len();
    let gsz = co / groups;
    let rest = x.len() / co;
    for g in 0..groups {
        let s = g * gsz * rest;
        let e = s + gsz * rest;
        let mean = x[s..e].iter().sum::<f32>() / (e - s) as f32;
        let var = x[s..e].iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (e - s) as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, v) in x[s..e].iter_mut().enumerate() {
            let ch = g * gsz + (i / rest);
            *v = (*v - mean) * inv * w[ch] + b[ch];
        }
    }
}

// ─────────── layout helpers ───────────

fn rows_to_chw(rows: &[f32], h: usize, w: usize) -> Vec<f32> {
    let n = h * w;
    let c = rows.len() / n;
    let mut out = vec![0f32; c * h * w];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w { out[(ch * h + y) * w + x] = rows[(y * w + x) * c + ch]; }
        }
    }
    out
}

fn chw_to_rows(chw: &[f32], h: usize, w: usize) -> Vec<f32> {
    let c = chw.len() / (h * w);
    let mut out = vec![0f32; h * w * c];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w { out[(y * w + x) * c + ch] = chw[(ch * h + y) * w + x]; }
        }
    }
    out
}

/// Normalized sine position encoding for an FPN level: rows [h*w, 2n]
/// (pos_y then pos_x), scale 2π, temperature 10000.
fn sine_pos(h: usize, w: usize, n: usize) -> Vec<f32> {
    let scale = std::f32::consts::TAU;
    let mut dim_t = vec![0f32; n];
    for i in 0..n { dim_t[i] = 10000f32.powf((2 * (i / 2)) as f32 / n as f32); }
    let mut out = vec![0f32; h * w * n * 2];
    let (hn, wn) = (h as f32, w as f32);
    for y in 0..h {
        let yy = (y + 1) as f32 / (hn + 1e-6) * scale;
        for x in 0..w {
            let xx = (x + 1) as f32 / (wn + 1e-6) * scale;
            let row = (y * w + x) * n * 2;
            for i in 0..n {
                let (py, px) = (yy / dim_t[i], xx / dim_t[i]);
                if i % 2 == 0 {
                    out[row + i] = py.sin();
                    out[row + n + i] = px.sin();
                } else {
                    out[row + i] = py.cos();
                    out[row + n + i] = px.cos();
                }
            }
        }
    }
    out
}

/// Sine box embedding (decoder query_pos input): [N, 4*128] =
/// cat(pos_y, pos_x, pos_w, pos_h) for boxes in (cx, cy, w, h),
/// scale 2π (Sam3SinePositionEmbedding.encode_boxes).
fn sine_boxes(boxes: &[[f32; 4]]) -> Vec<f32> {
    let nf = 128;
    let scale = std::f32::consts::TAU;
    let mut dim_t = vec![0f32; nf];
    for i in 0..nf { dim_t[i] = 10000f32.powf((2 * (i / 2)) as f32 / nf as f32); }
    let mut out = vec![0f32; boxes.len() * 4 * nf];
    for (q, b) in boxes.iter().enumerate() {
        let base = q * 4 * nf;
        for (ci, v) in [b[1], b[0], b[2], b[3]].iter().enumerate() {
            for i in 0..nf {
                let val = v * scale / dim_t[i];
                out[base + ci * nf + i] = if i % 2 == 0 { val.sin() } else { val.cos() };
            }
        }
    }
    out
}

/// Per-head log box-RPB additive bias: [heads, NQ+1, n]. Row 0 (the
/// presence token) is zero — HF pads the query dim from NQ to NQ+1.
fn rpb_bias(refs: &[[f32; 4]], rx: &DecoderMlp, ry: &DecoderMlp, n: usize, pool: Option<&Pool>) -> Result<Vec<f32>> {
    let gh = GRID;
    debug_assert_eq!(n, gh * gh);
    let l2_8 = 8.0f32.ln();
    let mut dy = vec![0f32; refs.len() * gh * 2];
    let mut dx = vec![0f32; refs.len() * gh * 2];
    for (q, r) in refs.iter().enumerate() {
        let x1 = r[0] - 0.5 * r[2];
        let y1 = r[1] - 0.5 * r[3];
        let x2 = r[0] + 0.5 * r[2];
        let y2 = r[1] + 0.5 * r[3];
        for hi in 0..gh {
            let c = hi as f32 / gh as f32;
            dy[(q * gh + hi) * 2] = log_compress(c - y1, l2_8);
            dy[(q * gh + hi) * 2 + 1] = log_compress(c - y2, l2_8);
        }
        for wi in 0..gh {
            let c = wi as f32 / gh as f32;
            dx[(q * gh + wi) * 2] = log_compress(c - x1, l2_8);
            dx[(q * gh + wi) * 2 + 1] = log_compress(c - x2, l2_8);
        }
    }
    let py = ry.forward(&dy, refs.len() * gh, pool, FH);
    let px = rx.forward(&dx, refs.len() * gh, pool, FH);
    let mut out = vec![0f32; FH * (NQ + 1) * n];
    for t in 0..FH {
        for q in 0..refs.len() {
            for hi in 0..gh {
                for wi in 0..gh {
                    out[(t * (NQ + 1) + (q + 1)) * n + hi * gh + wi] =
                        py[(q * gh + hi) * FH + t] + px[(q * gh + wi) * FH + t];
                }
            }
        }
    }
    Ok(out)
}

/// sign(d) * log2(|8d| + 1) / log2(8)  (box_rpb_mode = "log").
fn log_compress(d: f32, l2_8: f32) -> f32 {
    d.signum() * ((8.0 * d).abs() + 1.0).log2() / l2_8
}

/// elementwise a+b over equal-length buffers
fn add_rows(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Finiteness/range stats for one debug stage (env `SAM3_TEXT_DEBUG`).
#[allow(dead_code)]
fn dbg_stats(tag: &str, v: &[f32]) {
    if std::env::var("SAM3_TEXT_DEBUG").is_err() {
        return;
    }
    let (mut bad, mut mn, mut mx) = (0usize, f32::INFINITY, f32::NEG_INFINITY);
    for &x in v {
        if !x.is_finite() {
            bad += 1;
            continue;
        }
        if x < mn { mn = x; }
        if x > mx { mx = x; }
    }
    eprintln!(
        "[sam3] {tag} n={} nonfinite={} min={:.4e} max={:.4e}",
        v.len(), bad, mn, mx
    );
}

/// Raw f32 dump for reference bisection (env `SAM3_DUMP_DIR`).
#[allow(dead_code)]
fn dump_f32(dir: &str, name: &str, v: &[f32]) {
    use std::io::Write;
    let p = std::path::Path::new(dir).join(format!("{name}.f32"));
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    match std::fs::File::create(&p).and_then(|mut f| f.write_all(&b)) {
        Ok(()) => eprintln!("[sam3] dumped {} ({} elems)", p.display(), v.len()),
        Err(e) => eprintln!("[sam3] dump {} failed: {e}", p.display()),
    }
}