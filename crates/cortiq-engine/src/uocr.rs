//! Unlimited-OCR-2 (`unlimited-ocr`): the DeepEncoder (SAM ViT-B feeding
//! CLIP-L), the DeepSeek-V2 style MoE decoder and its DASH block-sparse
//! attention (SR-SWA), which reads a fixed set of references per query.
//!
//! ## The prompt
//!
//! `[BOS] (<PAGE> image tokens)×pages instruction`, no chat template. One
//! page is tiled ("gundam"): 640² tiles and a padded 1024² global view; several
//! pages take the global view only. See [`prepare`].
//!
//! ## The tower
//!
//! SAM: 16×16 patches over an absolute position table, 12 blocks at 768 (window
//! 14, global at 2, 5, 8, 11) with decomposed relative position bias, a conv
//! neck and two strided convs down to 1024 channels at 1/64. CLIP-L takes those
//! channels as its patch embeddings: class token + position table, then 24
//! blocks at 1024. Each token is `clip ‖ sam` projected to the decoder's 1280.
//!
//! ## DASH attention (`dash_attention.py`, the vLLM-parity inference path)
//!
//! Every layer has a router: `q_proj`, `k_proj` to 128, a Gemma-style RMSNorm
//! and the same two-axis rotary as the main branch. Keys are summarized per
//! completed 32-token block through a learned query (`head_cls`). For each
//! query:
//!
//! 1. score every block at least `local_blocks` behind the query,
//! 2. restrict to the candidates of the query's page, when it has any,
//! 3. entmax-1.5 over the 16 best scores, keep the 12 best blocks,
//! 4. bias = centered log prior,
//! 5. attend over those blocks (with the bias) and the last 128 tokens (none).
//!
//! The read set is at most 512 tokens however long the document is.
//!
//! The rotary has two axes (`page2d`): frequency pairs 0..63 follow the token's
//! running count inside its page, pair 63 the page number's parity.
//! [`PageRope`] keeps the coordinates, prefill and decode alike.
//!
//! Parity: `scripts/ocr/ref_unlimited_ocr.py` (Transformers 4.52.1).

use crate::Pipeline;
use crate::media::{self, RgbFrame};
use crate::onevision2::{Mat, gelu_rows, layer_norm_rows};
use crate::pool::Pool;
use cortiq_core::CmfModel;
use serde_json::Value;
use std::sync::Arc;

pub const ARCH: &str = "unlimited-ocr";
const BASE_SIZE: usize = 1024;
const IMAGE_SIZE: usize = 640;
const PATCH: usize = 16;
const DOWNSAMPLE: usize = 4;
const MIN_CROPS: usize = 2;
const MAX_CROPS: usize = 16;
const BOS_ID: u32 = 0;
/// `_ACT_CLAMP`, `_ENTMAX_NEG` in the reference.
const ACT_CLAMP: f32 = 1.0e4;
const ENTMAX_NEG: f32 = -1.0e4;
/// `DASH_INFER_TOPM`: the window entmax runs over.
const TOP_M: usize = 16;

/// Whether `model` is an Unlimited-OCR file.
pub fn is_uocr(model: &CmfModel) -> bool {
    model.header.arch.arch_name == ARCH
}

// ───────────────────────── page coordinates ─────────────────────────

/// The two-axis rotary coordinates and page bookkeeping of one sequence.
#[derive(Clone, Debug, Default)]
pub struct PageRope {
    pub image_id: u32,
    pub page_id: u32,
    /// `(tok, page)` of every prompt position.
    coords: Vec<(u32, u32)>,
    /// Image page of every prompt position, -1 elsewhere (raw markers − 1).
    image_page: Vec<i32>,
    /// The page of every prompt query, -1 for none.
    query_page: Vec<i32>,
    /// Prefill rows per page bucket (`pf_cnt`).
    pf_cnt: Vec<u32>,
    /// Decode state machine.
    page: u32,
    tok: u32,
    /// Response markers so far plus the prompt's, and the prompt's.
    mcum: i32,
    pp: i32,
    /// Per block `[min, max]` image page of the prompt, -1 without one.
    bmin: Vec<i32>,
    bmax: Vec<i32>,
    block: usize,
}

impl PageRope {
    /// `_build_page_rope_pos`, `_build_page_aware_ids` and the prefill half of
    /// `_uocr_dash_decode_allow` over the whole prompt.
    pub fn begin(ids: &[u32], image_id: u32, page_id: u32, block: usize) -> Result<Self, String> {
        let s = ids.len();
        let last_img = ids.iter().rposition(|&i| i == image_id).ok_or("unlimited-ocr: the prompt has no image")?;
        let mut cum = vec![0i32; s];
        let mut c = 0i32;
        for (i, &id) in ids.iter().enumerate() {
            c += i32::from(id == page_id);
            cum[i] = c;
        }
        let n_pages = cum[last_img];
        if n_pages == 0 {
            return Err("unlimited-ocr: the prompt carries no <PAGE> marker".into());
        }
        let page_of = |i: usize| -> u32 {
            let p = if i <= last_img { cum[i] } else { cum[i] - n_pages };
            p.max(0) as u32
        };
        let n_pg = (0..s).map(page_of).max().unwrap_or(0) as usize + 1;
        let mut run = vec![0u32; n_pg];
        let mut pf_cnt = vec![0u32; n_pg];
        let mut coords = Vec::with_capacity(s);
        for i in 0..s {
            let p = page_of(i) as usize;
            coords.push((run[p], p as u32));
            run[p] += 1;
            if i <= last_img {
                pf_cnt[p] += 1;
            }
        }
        let image_page: Vec<i32> =
            (0..s).map(|i| if ids[i] == image_id { cum[i] - 1 } else { -1 }).collect();
        let query_page: Vec<i32> = (0..s)
            .map(|i| {
                let p = if i <= last_img { cum[i] - 1 } else { cum[i] - 1 - n_pages };
                p.max(-1)
            })
            .collect();
        let nblk = s.div_ceil(block);
        let (mut bmin, mut bmax) = (vec![-1i32; nblk], vec![-1i32; nblk]);
        for (i, &pg) in image_page.iter().enumerate() {
            if pg >= 0 {
                let b = i / block;
                bmax[b] = bmax[b].max(pg);
                bmin[b] = if bmin[b] < 0 { pg } else { bmin[b].min(pg) };
            }
        }
        let (tok, page) = coords[s - 1];
        Ok(Self {
            image_id,
            page_id,
            coords,
            image_page,
            query_page,
            pf_cnt,
            page,
            tok,
            mcum: cum[s - 1],
            pp: n_pages,
            bmin,
            bmax,
            block,
        })
    }

    pub fn prompt_len(&self) -> usize {
        self.coords.len()
    }

    /// `(tok, page)` of prompt position `pos`.
    pub fn prompt_coords(&self, pos: usize) -> (u32, u32) {
        self.coords[pos]
    }

    /// Advance the decode state over token `id`; returns its coordinates.
    pub fn decode_step(&mut self, id: u32) -> (u32, u32) {
        if id == self.page_id {
            self.page = (self.page + 1).min(self.pf_cnt.len() as u32 - 1);
            self.tok = self.pf_cnt[self.page as usize];
            self.mcum += 1;
        } else {
            self.tok += 1;
        }
        (self.tok, self.page)
    }

    /// The candidate blocks of a prompt query (`_build_kl_candidate_mask`):
    /// the blocks with an image token of its page; everything when it has no
    /// page or the page has no block.
    pub fn prefill_allow(&self, pos: usize, nblk: usize, out: &mut Vec<bool>) {
        out.clear();
        out.resize(nblk, false);
        let qp = self.query_page[pos];
        let mut any = false;
        if qp >= 0 {
            // Blocks hold a page's tokens contiguously: `bmin..=bmax` of a
            // block spans the pages present in it.
            for b in 0..nblk.min(self.bmax.len()) {
                let hit = self.bmin[b] >= 0
                    && self.bmin[b] <= qp
                    && self.bmax[b] >= qp
                    && self.block_has_page(b, qp);
                out[b] = hit;
                any |= hit;
            }
        }
        if !any {
            out.iter_mut().for_each(|v| *v = true);
        }
    }

    fn block_has_page(&self, b: usize, page: i32) -> bool {
        let at = b * self.block;
        let end = (at + self.block).min(self.image_page.len());
        self.image_page[at..end].iter().any(|&p| p == page)
    }

    /// The candidate blocks of a decode query (`_uocr_dash_decode_allow`), the
    /// state already advanced over its token.
    pub fn decode_allow(&self, nblk: usize, out: &mut Vec<bool>) {
        out.clear();
        out.resize(nblk, false);
        let qp = self.mcum - 1 - self.pp;
        let qp = if qp >= 0 { qp } else { -1 };
        let mut any = false;
        if qp >= 0 {
            for b in 0..nblk.min(self.bmax.len()) {
                let hit = self.bmin[b] >= 0 && self.bmin[b] <= qp && self.bmax[b] >= qp;
                out[b] = hit;
                any |= hit;
            }
        }
        if !any {
            out.iter_mut().for_each(|v| *v = true);
        }
    }
}

/// The rotary angles of `(tok, page)` over `half` frequency pairs: all but the
/// last on the token axis (`1/θ^(j/n)`, θ = 10⁴), the last the page's parity
/// (`sparse_rope_page_mod = 2`, one pair at θ = 256, whose only frequency is 1).
pub fn rope_angles(tok: u32, page: u32, inv_tok: &[f32], out: &mut [f32]) {
    let t = tok as f32;
    for (o, &f) in out.iter_mut().zip(inv_tok) {
        *o = t * f;
    }
    *out.last_mut().expect("at least one pair") = (page % 2) as f32;
}

/// `inv_tok` of [`rope_angles`].
pub fn inv_tok(half: usize, theta: f32) -> Vec<f32> {
    let n = half - 1;
    (0..n).map(|j| 1.0 / theta.powf(j as f32 / n as f32)).collect()
}

/// `rotate_half`: `x[j]`, `x[j+half]` turned by `angles[j]`.
pub fn rotate(x: &mut [f32], cs: &[(f32, f32)]) {
    let half = cs.len();
    let (a, b) = x.split_at_mut(half);
    for j in 0..half {
        let (s, c) = cs[j];
        let (p, q) = (a[j], b[j]);
        a[j] = p * c - q * s;
        b[j] = q * c + p * s;
    }
}

// ───────────────────────── entmax routing ─────────────────────────

/// `_Entmax15.forward` over `x`, sorted descending, into `y`.
pub fn entmax15_sorted(x: &[f32], y: &mut [f32]) {
    let d = x.len();
    let top = x[0];
    let (mut sum, mut sum2) = (0f32, 0f32);
    let mut tau_star = f32::NAN;
    let mut support = 0usize;
    let mut taus = [0f32; TOP_M];
    for i in 0..d {
        let v = (x[i] - top) / 2.0;
        sum += v;
        sum2 += v * v;
        let rho = (i + 1) as f32;
        let mean = sum / rho;
        let mean_sq = sum2 / rho;
        let ss = rho * (mean_sq - mean * mean);
        let delta = ((1.0 - ss) / rho).max(0.0);
        let tau = mean - delta.sqrt();
        taus[i] = tau;
        if tau <= v {
            support += 1;
        }
    }
    if support > 0 {
        tau_star = taus[support - 1];
    } else if d > 0 {
        tau_star = taus[0];
    }
    for i in 0..d {
        let v = (x[i] - top) / 2.0 - tau_star;
        y[i] = if v > 0.0 { v * v } else { 0.0 };
    }
}

/// The routing parameters of one layer.
#[derive(Clone, Copy, Debug)]
pub struct RouteParams {
    /// `dash_scale_per_layer / √index_dim`.
    pub idx_scale: f32,
    pub floor: f32,
    pub sigma: f32,
    pub k: usize,
    pub top_m: usize,
}

/// `_infer_route_select` for one query. `scores[b]` is a block's score, None
/// when it is not a remote block; `allow` the page candidates. Fills `pick`
/// with `(block, bias)` of the read set.
pub fn route_select(
    scores: &[Option<f32>],
    allow: &[bool],
    p: &RouteParams,
    window: &mut Vec<(f32, u32, bool)>,
    pick: &mut Vec<(u32, f32)>,
) {
    pick.clear();
    let nblk = scores.len();
    if nblk == 0 {
        return;
    }
    let any_cand = scores.iter().zip(allow).any(|(s, &a)| s.is_some() && a);
    let is_cand = |b: usize| scores[b].is_some() && (!any_cand || allow[b]);
    window.clear();
    for b in 0..nblk {
        let c = is_cand(b);
        let v = if c { scores[b].unwrap_or(ENTMAX_NEG) * p.idx_scale } else { ENTMAX_NEG };
        let v = if v.is_finite() { v } else { ENTMAX_NEG };
        window.push((v, b as u32, c));
    }
    let m = p.top_m.min(nblk).max(1);
    // torch.topk: the largest values, descending; ties keep no contract that a
    // read set depends on (only non-candidates can tie at ENTMAX_NEG).
    window.select_nth_unstable_by(m - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    window.truncate(m);
    window.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let xv: Vec<f32> = window.iter().map(|w| w.0).collect();
    let mut prior = [0f32; TOP_M];
    entmax15_sorted(&xv, &mut prior[..m]);
    let k = p.k.min(m).max(1);
    let mut sum = 0f32;
    let mut cnt = 0usize;
    let mut logp = [0f32; TOP_M];
    for i in 0..k {
        logp[i] = prior[i].max(p.floor).ln();
        if window[i].2 {
            sum += logp[i];
            cnt += 1;
        }
    }
    let mu = sum / cnt.max(1) as f32;
    for i in 0..k {
        if window[i].2 {
            pick.push((window[i].1, (logp[i] - mu) / p.sigma));
        }
    }
}

// ───────────────────────── the router state ─────────────────────────

/// One layer's router weights.
pub struct DashLayer {
    /// `[2·index_dim, hidden]`: the q rows, then the k rows.
    pub qk_proj: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub head_cls: Vec<f32>,
    pub route: RouteParams,
}

/// The router: weights, geometry and the sequence state on the host.
pub struct Dash {
    pub layers: Vec<DashLayer>,
    pub block: usize,
    pub local_blocks: usize,
    pub local_window: usize,
    pub index_dim: usize,
    pub head_dim: usize,
    pub inv_tok: Vec<f32>,
    pub image_id: u32,
    pub page_id: u32,
    pub rope: Option<PageRope>,
    /// The answer's page bookkeeping (`_uocr_page_count_guard`).
    pub guard: Option<PageGuard>,
    /// `(sin, cos)` of the current position, one per frequency pair.
    pub cs: Vec<(f32, f32)>,
    /// The angles of the current rows, one `half` per row.
    pub angles: Vec<f32>,
    scratch_angles: Vec<f32>,
    /// Candidate blocks of the current query.
    pub allow: Vec<bool>,
    /// The blocks the device scores per row: remote and candidates, or every
    /// remote block when none is both. `coff[row]..coff[row + 1]`.
    pub cands: Vec<u32>,
    pub coff: Vec<u32>,
    /// Per layer: completed block summaries `[n, index_dim]`.
    pub k_bar: Vec<Vec<f32>>,
    /// Per layer: the keys of the open block.
    pub open: Vec<Vec<f32>>,
    /// Whether the graph holds the router state instead of the host.
    pub on_device: bool,
    // Scratch.
    scores: Vec<Option<f32>>,
    window: Vec<(f32, u32, bool)>,
    pick: Vec<(u32, f32)>,
    row: Vec<f32>,
}

fn sp_usize(sp: &Value, key: &str) -> Result<usize, String> {
    sp.get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("unlimited-ocr: sparse_attention_config has no '{key}'"))
}

fn sp_list(sp: &Value, key: &str, layers: usize) -> Result<Vec<f32>, String> {
    let a = sp
        .get(key)
        .and_then(Value::as_array)
        .filter(|a| a.len() == layers)
        .ok_or_else(|| format!("unlimited-ocr: sparse_attention_config.{key} must list {layers} layers"))?;
    Ok(a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
}

impl Dash {
    pub fn from_model(model: &CmfModel) -> Result<Self, String> {
        let cfg = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.pointer("/uocr/config"))
            .ok_or("unlimited-ocr: the file carries no provenance.uocr (reconvert it)")?;
        let sp = cfg.get("sparse_attention_config").ok_or("unlimited-ocr: config has no sparse_attention_config")?;
        let layers_n = cfg
            .get("num_hidden_layers")
            .and_then(Value::as_u64)
            .ok_or("unlimited-ocr: config has no num_hidden_layers")? as usize;
        let block = sp_usize(sp, "sparse_block_size")?;
        let local_blocks = sp_usize(sp, "sparse_local_block")?;
        let index_dim = sp_usize(sp, "sparse_index_dim")?;
        let (hidden, heads) = (model.header.arch.hidden_size, model.header.arch.num_attention_heads);
        let head_dim = hidden / heads;
        if index_dim != head_dim {
            return Err("unlimited-ocr: an index width other than the head width is not supported".into());
        }
        let on = |k: &str, want: &str| sp.get(k).and_then(Value::as_str) == Some(want);
        if !on("training_stage", "dash") || !on("dash_stage0", "head_cls") || !on("sparse_main_rope_mode", "page2d") {
            return Err("unlimited-ocr: only the dash head_cls page2d router is supported".into());
        }
        if sp.get("sparse_rope_page_pairs").and_then(Value::as_u64) != Some(1)
            || sp.get("sparse_rope_page_mod").and_then(Value::as_u64) != Some(2)
        {
            return Err("unlimited-ocr: only one page pair at parity is supported".into());
        }
        let scales = sp_list(sp, "dash_scale_per_layer", layers_n)?;
        let floors = sp_list(sp, "dash_floor_per_layer", layers_n)?;
        let sigma = sp.get("dash_sigma").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        let k = sp_usize(sp, "dash_topk_blocks")?;
        let f = |n: String, len: usize| -> Result<Vec<f32>, String> {
            let v = crate::dit::cmf_f32(model, &n)?;
            if v.len() != len {
                return Err(format!("tensor '{n}' has {} values, expected {len}", v.len()));
            }
            Ok(v)
        };
        let mut layers = Vec::with_capacity(layers_n);
        for li in 0..layers_n {
            let p = format!("model.layers.{li}.self_attn.dash_router");
            layers.push(DashLayer {
                qk_proj: {
                    let mut w = f(format!("{p}.q_proj.weight"), index_dim * hidden)?;
                    w.extend(f(format!("{p}.k_proj.weight"), index_dim * hidden)?);
                    w
                },
                q_norm: f(format!("{p}.q_norm.weight"), index_dim)?,
                k_norm: f(format!("{p}.k_norm.weight"), index_dim)?,
                head_cls: f(format!("{p}.head_cls"), index_dim)?,
                route: RouteParams {
                    idx_scale: scales[li] / (index_dim as f32).sqrt(),
                    floor: floors[li],
                    sigma,
                    k,
                    top_m: TOP_M,
                },
            });
        }
        let half = head_dim / 2;
        let theta = sp.get("sparse_rope_tok_theta").and_then(Value::as_f64).unwrap_or(10000.0) as f32;
        Ok(Self {
            block,
            local_blocks,
            local_window: local_blocks * block,
            index_dim,
            head_dim,
            inv_tok: inv_tok(half, theta),
            image_id: sp_usize(sp, "image_token_id")? as u32,
            page_id: sp_usize(sp, "page_token_id")? as u32,
            rope: None,
            guard: None,
            cs: vec![(0.0, 1.0); half],
            angles: vec![0.0; half],
            scratch_angles: vec![0.0; half],
            allow: Vec::new(),
            cands: Vec::new(),
            coff: Vec::new(),
            k_bar: vec![Vec::new(); layers_n],
            open: vec![Vec::new(); layers_n],
            on_device: false,
            scores: Vec::new(),
            window: Vec::new(),
            pick: Vec::new(),
            row: vec![0.0; index_dim],
            layers,
        })
    }

    /// A new sequence over the prompt `ids`.
    pub fn begin(&mut self, ids: &[u32], eos_id: u32) -> Result<(), String> {
        self.rope = Some(PageRope::begin(ids, self.image_id, self.page_id, self.block)?);
        let pages = ids.iter().filter(|&&i| i == self.page_id).count();
        self.guard = Some(PageGuard::new(pages, self.page_id, eos_id));
        for v in self.k_bar.iter_mut().chain(self.open.iter_mut()) {
            v.clear();
        }
        Ok(())
    }

    /// Set the coordinates and candidates of a prompt position.
    pub fn at_prompt(&mut self, pos: usize) {
        self.at_prompt_chunk(pos, 1);
        let rope = self.rope.as_ref().expect("begin ran");
        let (tok, page) = rope.prompt_coords(pos);
        self.set_cs(tok, page);
    }

    /// The angles and candidates of the prompt rows `pos..pos + k`. `allow` is
    /// left at the last row.
    pub fn at_prompt_chunk(&mut self, pos: usize, k: usize) {
        let half = self.head_dim / 2;
        let mut allow = std::mem::take(&mut self.allow);
        self.angles.clear();
        self.angles.resize(k * half, 0.0);
        self.cands.clear();
        self.coff.clear();
        self.coff.push(0);
        let rope = self.rope.as_ref().expect("begin ran");
        let nblk = rope.prompt_len().div_ceil(self.block);
        for i in 0..k {
            let p = pos + i;
            let (tok, page) = rope.prompt_coords(p);
            rope_angles(tok, page, &self.inv_tok, &mut self.angles[i * half..(i + 1) * half]);
            rope.prefill_allow(p, nblk, &mut allow);
            Self::push_cands(&mut self.cands, &allow, p, self.block, self.local_blocks);
            self.coff.push(self.cands.len() as u32);
        }
        self.allow = allow;
    }

    /// Advance over decoded token `id` at `pos` and set its candidates.
    pub fn at_decode(&mut self, id: u32, pos: usize) {
        let half = self.head_dim / 2;
        let rope = self.rope.as_mut().expect("begin ran");
        let (tok, page) = rope.decode_step(id);
        let nblk = (pos + 1).div_ceil(self.block);
        let mut allow = std::mem::take(&mut self.allow);
        self.rope.as_ref().expect("begin ran").decode_allow(nblk, &mut allow);
        self.cands.clear();
        Self::push_cands(&mut self.cands, &allow, pos, self.block, self.local_blocks);
        self.coff.clear();
        self.coff.extend([0, self.cands.len() as u32]);
        self.allow = allow;
        self.angles.clear();
        self.angles.resize(half, 0.0);
        rope_angles(tok, page, &self.inv_tok, &mut self.angles);
        self.set_cs(tok, page);
    }

    fn set_cs(&mut self, tok: u32, page: u32) {
        rope_angles(tok, page, &self.inv_tok, &mut self.scratch_angles);
        for (o, a) in self.cs.iter_mut().zip(&self.scratch_angles) {
            *o = a.sin_cos();
        }
    }

    /// Row `pos`'s candidates: the remote blocks it may read.
    fn push_cands(out: &mut Vec<u32>, allow: &[bool], pos: usize, block: usize, local: usize) {
        let nrem = (pos / block + 1).saturating_sub(local);
        let at = out.len();
        out.extend((0..nrem.min(allow.len())).filter(|&b| allow[b]).map(|b| b as u32));
        if out.len() == at {
            out.extend(0..nrem as u32);
        }
    }

    /// `_GemmaRMSNorm` plus the rotation: a router row in place.
    fn norm_rotate(x: &mut [f32], w: &[f32], eps: f32, cs: &[(f32, f32)]) {
        let d = x.len();
        let mean = x.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = (mean + eps).sqrt().recip();
        for (v, &wv) in x.iter_mut().zip(w) {
            let n = (*v * inv).clamp(-ACT_CLAMP, ACT_CLAMP);
            *v = (n * (1.0 + wv)).clamp(-ACT_CLAMP, ACT_CLAMP);
        }
        rotate(x, cs);
    }

    /// One position through layer `li`'s router on the host: appends its key
    /// (closing a block when full) and returns the read set in `self.pick`.
    fn route_host(&mut self, li: usize, n1: &[f32], pos: usize, eps: f32) {
        let d = self.index_dim;
        let hs = n1.len();
        let l = &self.layers[li];
        let mut q = vec![0f32; d];
        let mut k = std::mem::take(&mut self.row);
        for r in 0..d {
            let wq = &l.qk_proj[r * hs..(r + 1) * hs];
            let wk = &l.qk_proj[(d + r) * hs..(d + r + 1) * hs];
            let (mut a, mut b) = (0f32, 0f32);
            for c in 0..hs {
                a += wq[c] * n1[c];
                b += wk[c] * n1[c];
            }
            q[r] = a;
            k[r] = b;
        }
        Self::norm_rotate(&mut q, &l.q_norm, eps, &self.cs);
        Self::norm_rotate(&mut k, &l.k_norm, eps, &self.cs);
        // Queries score against the blocks completed before this position's
        // key: every remote block is complete already.
        let inv_sqrt = (d as f32).sqrt().recip();
        let qblk = pos / self.block;
        let done = self.k_bar[li].len() / d;
        let nblk = self.allow.len().max(done);
        self.scores.clear();
        self.scores.resize(nblk, None);
        for b in 0..done.min((qblk + 1).saturating_sub(self.local_blocks)) {
            let kb = &self.k_bar[li][b * d..(b + 1) * d];
            let z: f32 = q.iter().zip(kb).map(|(x, y)| x * y).sum::<f32>() * inv_sqrt;
            self.scores[b] = Some(z.clamp(-ACT_CLAMP, ACT_CLAMP));
        }
        if self.allow.len() < nblk {
            self.allow.resize(nblk, false);
        }
        let (scores, allow, window, pick) = (&self.scores, &self.allow, &mut self.window, &mut self.pick);
        route_select(scores, allow, &l.route, window, pick);
        self.open[li].extend_from_slice(&k);
        self.row = k;
        if self.open[li].len() == self.block * d {
            let mut kb = vec![0f32; d];
            summarize_block(&self.open[li], &self.layers[li].head_cls, &mut kb);
            self.k_bar[li].extend_from_slice(&kb);
            self.open[li].clear();
        }
    }
}

/// The learned chunk summary (`Stage 0`): softmax of the keys against
/// `head_cls`, the weighted mean key.
pub fn summarize_block(keys: &[f32], hc: &[f32], out: &mut [f32]) {
    let d = hc.len();
    let n = keys.len() / d;
    let inv_sqrt = (d as f32).sqrt().recip();
    let mut aw = vec![0f32; n];
    let mut mx = f32::NEG_INFINITY;
    for (i, a) in aw.iter_mut().enumerate() {
        let k = &keys[i * d..(i + 1) * d];
        *a = (k.iter().zip(hc).map(|(x, y)| x * y).sum::<f32>() * inv_sqrt).clamp(-ACT_CLAMP, ACT_CLAMP);
        mx = mx.max(*a);
    }
    let mut den = 0f32;
    for a in aw.iter_mut() {
        *a = (*a - mx).exp();
        den += *a;
    }
    let den = den.max(1e-6);
    out.iter_mut().for_each(|v| *v = 0.0);
    for (i, a) in aw.iter().enumerate() {
        let w = a / den;
        for (o, &v) in out.iter_mut().zip(&keys[i * d..(i + 1) * d]) {
            *o += w * v;
        }
    }
}

/// DASH attention for one position on the host: K/V appended to the layer's
/// cache, the read set from the router. `out` is `[heads·head_dim]`, before
/// `o_proj`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_host(
    dash: &mut Dash,
    li: usize,
    n1: &[f32],
    wq: &crate::qtensor::QTensor,
    wk: &crate::qtensor::QTensor,
    wv: &crate::qtensor::QTensor,
    cache: &mut crate::kv_cache::LayerKvCache,
    position: usize,
    nh: usize,
    hd: usize,
    eps: f32,
    pool: Option<&Pool>,
    out: &mut [f32],
) {
    dash.route_host(li, n1, position, eps);
    let (mut q, mut k, mut v) = (vec![0f32; nh * hd], vec![0f32; nh * hd], vec![0f32; nh * hd]);
    wq.matvec(n1, &mut q, pool);
    wk.matvec(n1, &mut k, pool);
    wv.matvec(n1, &mut v, pool);
    for h in 0..nh {
        rotate(&mut q[h * hd..(h + 1) * hd], &dash.cs);
        rotate(&mut k[h * hd..(h + 1) * hd], &dash.cs);
    }
    cache.append(&k, &v, &[]);
    let scale = (hd as f32).sqrt().recip();
    let (block, window) = (dash.block, dash.local_window);
    let lo = (position + 1).saturating_sub(window);
    let pick = &dash.pick;
    let (keys, vals) = (cache.k_heads(), cache.v_heads());
    let attend = |h: usize, o: &mut [f32]| {
        let qh = &q[h * hd..(h + 1) * hd];
        let (kh, vh) = (&keys[h], &vals[h]);
        let mut sc: Vec<(u32, f32)> = Vec::with_capacity(pick.len() * block + window);
        let dot = |p: usize| -> f32 {
            let kr = &kh[p * hd..(p + 1) * hd];
            qh.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() * scale
        };
        for &(b, bias) in pick {
            let at = b as usize * block;
            // The local window wins on overlap.
            for p in at..(at + block).min(lo) {
                sc.push((p as u32, dot(p) + bias));
            }
        }
        for p in lo..=position {
            sc.push((p as u32, dot(p)));
        }
        let mx = sc.iter().fold(f32::NEG_INFINITY, |m, s| m.max(s.1));
        let mut den = 0f32;
        for s in sc.iter_mut() {
            s.1 = (s.1 - mx).exp();
            den += s.1;
        }
        let inv = den.recip();
        o.iter_mut().for_each(|x| *x = 0.0);
        for &(p, w) in &sc {
            let w = w * inv;
            let vr = &vh[p as usize * hd..(p as usize + 1) * hd];
            for (x, &y) in o.iter_mut().zip(vr) {
                *x += w * y;
            }
        }
    };
    let oo = crate::pool::SendMut::new(out.as_mut_ptr());
    let run = |lo_h: usize, hi_h: usize| {
        for h in lo_h..hi_h {
            // SAFETY: each worker owns whole heads of `out`.
            let o = unsafe { std::slice::from_raw_parts_mut(oo.at(h * hd), hd) };
            attend(h, o);
        }
    };
    match pool {
        Some(p) => p.run_rows(nh, &run),
        None => run(0, nh),
    }
}

// ───────────────────────── image layout ─────────────────────────

fn round_half_even(v: f64) -> i64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 { (r - v.signum()) as i64 } else { r as i64 }
}

/// One axis of PIL's fixed-point bicubic (`precompute_coeffs`): per output
/// sample its first input and 22-bit weights.
fn pil_taps(inp: usize, out: usize) -> Vec<(usize, Vec<i32>)> {
    let bic = |x: f64| -> f64 {
        const A: f64 = -0.5;
        let x = x.abs();
        if x < 1.0 {
            ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * A
        } else {
            0.0
        }
    };
    let scale = inp as f64 / out as f64;
    let fs = scale.max(1.0);
    let support = 2.0 * fs;
    (0..out)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = ((center + support + 0.5) as i64).min(inp as i64) as usize;
            let w: Vec<f64> = (lo..hi).map(|x| bic((x as f64 - center + 0.5) / fs)).collect();
            let sum: f64 = w.iter().sum();
            let k = w
                .iter()
                .map(|v| {
                    let v = if sum != 0.0 { v / sum } else { *v } * (1u32 << 22) as f64;
                    (if v < 0.0 { v - 0.5 } else { v + 0.5 }) as i32
                })
                .collect();
            (lo, k)
        })
        .collect()
}

/// `Image.resize(size)`, whose default is BICUBIC: horizontal, then vertical,
/// each pass rounded to 8 bits.
pub fn pil_bicubic(img: &RgbFrame, height: usize, width: usize) -> RgbFrame {
    let clip = |v: i64| -> u8 { (v >> 22).clamp(0, 255) as u8 };
    let round = 1i64 << 21;
    let (h0, w0) = (img.height, img.width);
    let mut cur = img.data.clone();
    if width != w0 {
        let taps = pil_taps(w0, width);
        let mut out = vec![0u8; h0 * width * 3];
        for y in 0..h0 {
            let src = &cur[y * w0 * 3..(y + 1) * w0 * 3];
            let dst = &mut out[y * width * 3..(y + 1) * width * 3];
            for (x, (lo, k)) in taps.iter().enumerate() {
                for c in 0..3 {
                    let mut ss = round;
                    for (j, &w) in k.iter().enumerate() {
                        ss += src[(lo + j) * 3 + c] as i64 * w as i64;
                    }
                    dst[x * 3 + c] = clip(ss);
                }
            }
        }
        cur = out;
    }
    if height != h0 {
        let taps = pil_taps(h0, height);
        let mut out = vec![0u8; height * width * 3];
        for (y, (lo, k)) in taps.iter().enumerate() {
            for x in 0..width * 3 {
                let mut ss = round;
                for (j, &w) in k.iter().enumerate() {
                    ss += cur[(lo + j) * width * 3 + x] as i64 * w as i64;
                }
                out[y * width * 3 + x] = clip(ss);
            }
        }
        cur = out;
    }
    RgbFrame { width, height, data: cur }
}

/// `ImageOps.pad(image, (size, size), color=127)`: BICUBIC into the square,
/// centered on mid gray.
pub fn pad_square(img: &RgbFrame, size: usize) -> RgbFrame {
    let (w, h) = (img.width as f64, img.height as f64);
    let (mut nw, mut nh) = (size, size);
    if img.width != img.height {
        if img.width > img.height {
            nh = round_half_even(h / w * size as f64) as usize;
        } else {
            nw = round_half_even(w / h * size as f64) as usize;
        }
    }
    let r = if (nh, nw) == (img.height, img.width) {
        img.clone()
    } else {
        pil_bicubic(img, nh, nw)
    };
    if nw == size && nh == size {
        return r;
    }
    let mut data = vec![127u8; size * size * 3];
    let x0 = round_half_even((size - nw) as f64 * 0.5) as usize;
    let y0 = round_half_even((size - nh) as f64 * 0.5) as usize;
    for y in 0..nh {
        let d = ((y0 + y) * size + x0) * 3;
        let s = y * nw * 3;
        data[d..d + nw * 3].copy_from_slice(&r.data[s..s + nw * 3]);
    }
    RgbFrame { width: size, height: size, data }
}

/// `find_closest_aspect_ratio` + the target list of `dynamic_preprocess`:
/// `(cols, rows)`.
fn closest_grid(w: usize, h: usize, tile: usize) -> (usize, usize) {
    let mut ratios: Vec<(usize, usize)> = Vec::new();
    for i in 1..=MAX_CROPS {
        for j in 1..=MAX_CROPS {
            if i * j >= MIN_CROPS && i * j <= MAX_CROPS {
                ratios.push((i, j));
            }
        }
    }
    // `sorted(set, key=i·j)`: stable over a set's iteration order, which for
    // these small int tuples is what CPython's tuple hash gives. Ties only
    // matter at exactly equal aspect ratios with equal areas, which cannot be
    // two different (i, j); the order among equal areas is irrelevant.
    ratios.sort_by_key(|r| r.0 * r.1);
    let aspect = w as f64 / h as f64;
    let area = (w * h) as f64;
    let mut best = (1usize, 1usize);
    let mut best_diff = f64::INFINITY;
    for &(i, j) in &ratios {
        let diff = (aspect - i as f64 / j as f64).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (i, j);
        } else if diff == best_diff && area > 0.5 * (tile * tile * i * j) as f64 {
            best = (i, j);
        }
    }
    best
}

/// One image's views: the global `[3, 1024, 1024]` and the tiles
/// `[3, 640, 640]` with the grid `(cols, rows)`.
pub struct PageViews {
    pub global: Vec<f32>,
    pub tiles: Vec<Vec<f32>>,
    pub cols: usize,
    pub rows: usize,
}

fn to_chw(img: &RgbFrame, top: usize, left: usize, side: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * side * side];
    for c in 0..3 {
        for y in 0..side {
            let line = ((top + y) * img.width + left) * 3 + c;
            for x in 0..side {
                // ToTensor (/255) then Normalize(0.5, 0.5), both f32.
                let v = img.data[line + x * 3] as f32 / 255.0;
                out[(c * side + y) * side + x] = (v - 0.5) / 0.5;
            }
        }
    }
    out
}

/// `_uocr_build_inputs` for one image.
pub fn page_views(img: &RgbFrame, tiled: bool) -> PageViews {
    let global = to_chw(&pad_square(img, BASE_SIZE), 0, 0, BASE_SIZE);
    if !tiled || (img.width <= IMAGE_SIZE && img.height <= IMAGE_SIZE) {
        return PageViews { global, tiles: Vec::new(), cols: 1, rows: 1 };
    }
    let (cols, rows) = closest_grid(img.width, img.height, IMAGE_SIZE);
    let resized = pil_bicubic(img, IMAGE_SIZE * rows, IMAGE_SIZE * cols);
    let mut tiles = Vec::with_capacity(cols * rows);
    for y in 0..rows {
        for x in 0..cols {
            tiles.push(to_chw(&resized, y * IMAGE_SIZE, x * IMAGE_SIZE, IMAGE_SIZE));
        }
    }
    PageViews { global, tiles, cols, rows }
}

/// Image tokens of a page.
pub fn page_tokens(cols: usize, rows: usize) -> usize {
    let nq = (IMAGE_SIZE / PATCH).div_ceil(DOWNSAMPLE);
    let nqb = (BASE_SIZE / PATCH).div_ceil(DOWNSAMPLE);
    let mut n = nqb * (nqb + 1) + 1;
    if cols > 1 || rows > 1 {
        n += (nq * cols + 1) * (nq * rows);
    }
    n
}

// ───────────────────────── the tower ─────────────────────────

/// The nearest-free antialiased bicubic of a square position table
/// (`F.interpolate(mode="bicubic", antialias=True)`): `[src², d]` → `[dst², d]`.
fn resize_table(t: &[f32], src: usize, dst: usize, d: usize) -> Vec<f32> {
    if src == dst {
        return t.to_vec();
    }
    let weights = |src: usize, dst: usize| -> Vec<(usize, Vec<f32>)> {
        // torch's antialiased bicubic follows PIL: support 2·scale, a = -0.5.
        let scale = src as f64 / dst as f64;
        let fs = scale.max(1.0);
        let support = 2.0 * fs;
        (0..dst)
            .map(|i| {
                let center = (i as f64 + 0.5) * scale;
                let lo = ((center - support + 0.5) as i64).max(0) as usize;
                let hi = ((center + support + 0.5) as i64).min(src as i64) as usize;
                let bic = |x: f64| -> f64 {
                    let a = -0.5;
                    let x = x.abs();
                    if x < 1.0 {
                        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
                    } else if x < 2.0 {
                        (((x - 5.0) * x + 8.0) * x - 4.0) * a
                    } else {
                        0.0
                    }
                };
                let w: Vec<f64> = (lo..hi).map(|j| bic((j as f64 + 0.5 - center) / fs)).collect();
                let s: f64 = w.iter().sum();
                (lo, w.iter().map(|v| (v / s) as f32).collect())
            })
            .collect()
    };
    let w = weights(src, dst);
    // Horizontal, then vertical.
    let mut h = vec![0f32; src * dst * d];
    for y in 0..src {
        for x in 0..dst {
            let (lo, ws) = &w[x];
            for c in 0..d {
                let mut v = 0f32;
                for (j, &wt) in ws.iter().enumerate() {
                    v += wt * t[(y * src + lo + j) * d + c];
                }
                h[(y * dst + x) * d + c] = v;
            }
        }
    }
    let mut out = vec![0f32; dst * dst * d];
    for y in 0..dst {
        let (lo, ws) = &w[y];
        for x in 0..dst {
            for c in 0..d {
                let mut v = 0f32;
                for (j, &wt) in ws.iter().enumerate() {
                    v += wt * h[((lo + j) * dst + x) * d + c];
                }
                out[(y * dst + x) * d + c] = v;
            }
        }
    }
    out
}

fn vec_of(model: &Arc<CmfModel>, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let v = crate::dit::cmf_f32(model, name)?;
    if v.len() != len {
        return Err(format!("tensor '{name}' has {} values, expected {len}", v.len()));
    }
    Ok(v)
}

struct SamBlock {
    n1_w: Vec<f32>,
    n1_b: Vec<f32>,
    qkv: Mat,
    qkv_b: Vec<f32>,
    proj: Mat,
    proj_b: Vec<f32>,
    rel_h: Vec<f32>,
    rel_w: Vec<f32>,
    /// `rel_h` then `rel_w`, for the device.
    rel_cat: Vec<f32>,
    rel_len: usize,
    n2_w: Vec<f32>,
    n2_b: Vec<f32>,
    fc1: Mat,
    fc1_b: Vec<f32>,
    fc2: Mat,
    fc2_b: Vec<f32>,
    window: usize,
}

struct ClipBlock {
    n1_w: Vec<f32>,
    n1_b: Vec<f32>,
    qkv: Mat,
    qkv_b: Vec<f32>,
    out: Mat,
    out_b: Vec<f32>,
    n2_w: Vec<f32>,
    n2_b: Vec<f32>,
    fc1: Mat,
    fc1_b: Vec<f32>,
    fc2: Mat,
    fc2_b: Vec<f32>,
}

/// The DeepEncoder and its projector.
pub struct UocrTower {
    patch: Mat,
    patch_b: Vec<f32>,
    pos: Vec<f32>,
    sam: Vec<SamBlock>,
    neck0: Vec<f32>,
    neck1_w: Vec<f32>,
    neck1_b: Vec<f32>,
    neck2: Vec<f32>,
    neck3_w: Vec<f32>,
    neck3_b: Vec<f32>,
    net2: Vec<f32>,
    net3: Vec<f32>,
    /// The four convs as `[co, k·k, ci]`, for the device.
    conv_w: [Vec<f32>; 4],
    cls: Vec<f32>,
    clip_pos: Vec<f32>,
    pre_w: Vec<f32>,
    pre_b: Vec<f32>,
    clip: Vec<ClipBlock>,
    proj: Mat,
    proj_b: Vec<f32>,
    pub newline: Vec<f32>,
    pub view_sep: Vec<f32>,
    pool: Option<Arc<Pool>>,
    /// Interpolated tables and relative positions per input grid.
    cache: std::sync::Mutex<std::collections::HashMap<usize, Arc<GridTables>>>,
}

struct GridTables {
    pos: Vec<f32>,
    clip_pos: Vec<f32>,
    /// Per global block: `[2g-1, 64]` relative positions, h and w.
    rel_global: Vec<(Vec<f32>, Vec<f32>)>,
}

const SAM_D: usize = 768;
const SAM_HEADS: usize = 12;
const SAM_HD: usize = 64;
const SAM_INTER: usize = 3072;
const SAM_GLOBAL: [usize; 4] = [2, 5, 8, 11];
const SAM_WINDOW: usize = 14;
const SAM_SIDE: usize = 64;
const SAM_EPS: f64 = 1e-6;
const CLIP_D: usize = 1024;
const CLIP_HEADS: usize = 16;
const CLIP_HD: usize = 64;
const CLIP_INTER: usize = 4096;
const CLIP_LAYERS: usize = 24;
const CLIP_SIDE: usize = 16;
const CLIP_EPS: f64 = 1e-5;
const OUT_D: usize = 1280;

/// `F.interpolate(mode="linear")` of a `[l, c]` table to `[n, c]`.
fn resize_rel(t: &[f32], c: usize, n: usize) -> Vec<f32> {
    let l = t.len() / c;
    if l == n {
        return t.to_vec();
    }
    let mut out = vec![0f32; n * c];
    let scale = l as f64 / n as f64;
    for i in 0..n {
        let x = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
        let x0 = x.floor() as usize;
        let x1 = (x0 + 1).min(l - 1);
        let f = (x - x0 as f64) as f32;
        for k in 0..c {
            out[i * c + k] = t[x0 * c + k] * (1.0 - f) + t[x1 * c + k] * f;
        }
    }
    out
}

/// A bias-free conv over an HWC grid with weights `[co, ci, k, k]`:
/// `(output, output side)`.
#[allow(clippy::too_many_arguments)]
fn conv_same(x: &[f32], g: usize, ci: usize, w: &[f32], co: usize, k: usize, stride: usize, pad: usize) -> (Vec<f32>, usize) {
    let g2 = (g + 2 * pad - k) / stride + 1;
    let mut out = vec![0f32; g2 * g2 * co];
    let row = |oy: usize, ox: usize, o: &mut [f32]| {
        for ky in 0..k {
            let Some(iy) = (oy * stride + ky).checked_sub(pad).filter(|&v| v < g) else { continue };
            for kx in 0..k {
                let Some(ix) = (ox * stride + kx).checked_sub(pad).filter(|&v| v < g) else { continue };
                let src = &x[(iy * g + ix) * ci..(iy * g + ix + 1) * ci];
                for (co_i, ov) in o.iter_mut().enumerate() {
                    let base = co_i * ci * k * k + ky * k + kx;
                    let mut acc = 0f32;
                    for (ci_i, &s) in src.iter().enumerate() {
                        acc += w[base + ci_i * k * k] * s;
                    }
                    *ov += acc;
                }
            }
        }
    };
    let p = Pool::from_env();
    let oo = crate::pool::SendMut::new(out.as_mut_ptr());
    let f = |lo: usize, hi: usize| {
        for r in lo..hi {
            let (oy, ox) = (r / g2, r % g2);
            // SAFETY: disjoint output cells per worker.
            let o = unsafe { std::slice::from_raw_parts_mut(oo.at(r * co), co) };
            row(oy, ox, o);
        }
    };
    match p.as_deref() {
        Some(pool) => pool.run_rows(g2 * g2, &f),
        None => f(0, g2 * g2),
    }
    (out, g2)
}

/// `LayerNorm2d` over the channels of an HWC grid.
fn norm2d(x: &mut [f32], c: usize, w: &[f32], b: &[f32]) {
    for row in x.chunks_exact_mut(c) {
        let u = row.iter().sum::<f32>() / c as f32;
        let s = row.iter().map(|v| (v - u) * (v - u)).sum::<f32>() / c as f32;
        let inv = (s + 1e-6).sqrt().recip();
        for ((v, &ww), &bb) in row.iter_mut().zip(w).zip(b) {
            *v = ww * ((*v - u) * inv) + bb;
        }
    }
}

impl UocrTower {
    pub fn from_model(model: &Arc<CmfModel>, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let mut sam = Vec::with_capacity(12);
        for i in 0..12 {
            let p = format!("vis.sam.blocks.{i}");
            let global = SAM_GLOBAL.contains(&i);
            let rel_len = if global { 2 * SAM_SIDE - 1 } else { 2 * SAM_WINDOW - 1 };
            let d = SAM_D;
            sam.push(SamBlock {
                n1_w: vec_of(model, &format!("{p}.norm1.weight"), d)?,
                n1_b: vec_of(model, &format!("{p}.norm1.bias"), d)?,
                qkv: Mat::load(model, &format!("{p}.attn.qkv.weight"), 3 * d, d)?,
                qkv_b: vec_of(model, &format!("{p}.attn.qkv.bias"), 3 * d)?,
                proj: Mat::load(model, &format!("{p}.attn.proj.weight"), d, d)?,
                proj_b: vec_of(model, &format!("{p}.attn.proj.bias"), d)?,
                rel_h: vec_of(model, &format!("{p}.attn.rel_pos_h"), rel_len * SAM_HD)?,
                rel_w: vec_of(model, &format!("{p}.attn.rel_pos_w"), rel_len * SAM_HD)?,
                rel_cat: {
                    let mut v = vec_of(model, &format!("{p}.attn.rel_pos_h"), rel_len * SAM_HD)?;
                    v.extend(vec_of(model, &format!("{p}.attn.rel_pos_w"), rel_len * SAM_HD)?);
                    v
                },
                rel_len,
                n2_w: vec_of(model, &format!("{p}.norm2.weight"), d)?,
                n2_b: vec_of(model, &format!("{p}.norm2.bias"), d)?,
                fc1: Mat::load(model, &format!("{p}.mlp.lin1.weight"), SAM_INTER, d)?,
                fc1_b: vec_of(model, &format!("{p}.mlp.lin1.bias"), SAM_INTER)?,
                fc2: Mat::load(model, &format!("{p}.mlp.lin2.weight"), d, SAM_INTER)?,
                fc2_b: vec_of(model, &format!("{p}.mlp.lin2.bias"), d)?,
                window: if global { 0 } else { SAM_WINDOW },
            });
        }
        let mut clip = Vec::with_capacity(CLIP_LAYERS);
        for i in 0..CLIP_LAYERS {
            let p = format!("vis.clip.transformer.layers.{i}");
            let d = CLIP_D;
            clip.push(ClipBlock {
                n1_w: vec_of(model, &format!("{p}.layer_norm1.weight"), d)?,
                n1_b: vec_of(model, &format!("{p}.layer_norm1.bias"), d)?,
                qkv: Mat::load(model, &format!("{p}.self_attn.qkv_proj.weight"), 3 * d, d)?,
                qkv_b: vec_of(model, &format!("{p}.self_attn.qkv_proj.bias"), 3 * d)?,
                out: Mat::load(model, &format!("{p}.self_attn.out_proj.weight"), d, d)?,
                out_b: vec_of(model, &format!("{p}.self_attn.out_proj.bias"), d)?,
                n2_w: vec_of(model, &format!("{p}.layer_norm2.weight"), d)?,
                n2_b: vec_of(model, &format!("{p}.layer_norm2.bias"), d)?,
                fc1: Mat::load(model, &format!("{p}.mlp.fc1.weight"), CLIP_INTER, d)?,
                fc1_b: vec_of(model, &format!("{p}.mlp.fc1.bias"), CLIP_INTER)?,
                fc2: Mat::load(model, &format!("{p}.mlp.fc2.weight"), d, CLIP_INTER)?,
                fc2_b: vec_of(model, &format!("{p}.mlp.fc2.bias"), d)?,
            });
        }
        // `[co, ci, k, k]` → `[co, k·k, ci]`.
        let reorder = |w: &[f32], ci: usize, co: usize, k: usize| -> Vec<f32> {
            let mut out = vec![0f32; w.len()];
            for c in 0..co {
                for t in 0..k * k {
                    for i in 0..ci {
                        out[(c * k * k + t) * ci + i] = w[(c * ci + i) * k * k + t];
                    }
                }
            }
            out
        };
        let neck0 = vec_of(model, "vis.sam.neck.0.weight", 256 * SAM_D)?;
        let neck2 = vec_of(model, "vis.sam.neck.2.weight", 256 * 256 * 9)?;
        let net2 = vec_of(model, "vis.sam.net_2.weight", 512 * 256 * 9)?;
        let net3 = vec_of(model, "vis.sam.net_3.weight", 1024 * 512 * 9)?;
        let conv_w = [
            reorder(&neck0, SAM_D, 256, 1),
            reorder(&neck2, 256, 256, 3),
            reorder(&net2, 256, 512, 3),
            reorder(&net3, 512, 1024, 3),
        ];
        Ok(Self {
            patch: Mat::load(model, "vis.sam.patch_embed.proj.weight", SAM_D, 3 * PATCH * PATCH)?,
            patch_b: vec_of(model, "vis.sam.patch_embed.proj.bias", SAM_D)?,
            pos: vec_of(model, "vis.sam.pos_embed", SAM_SIDE * SAM_SIDE * SAM_D)?,
            sam,
            neck0,
            neck1_w: vec_of(model, "vis.sam.neck.1.weight", 256)?,
            neck1_b: vec_of(model, "vis.sam.neck.1.bias", 256)?,
            neck2,
            neck3_w: vec_of(model, "vis.sam.neck.3.weight", 256)?,
            neck3_b: vec_of(model, "vis.sam.neck.3.bias", 256)?,
            net2,
            net3,
            conv_w,
            cls: vec_of(model, "vis.clip.embeddings.class_embedding", CLIP_D)?,
            clip_pos: vec_of(model, "vis.clip.embeddings.position_embedding.weight", (CLIP_SIDE * CLIP_SIDE + 1) * CLIP_D)?,
            pre_w: vec_of(model, "vis.clip.pre_layrnorm.weight", CLIP_D)?,
            pre_b: vec_of(model, "vis.clip.pre_layrnorm.bias", CLIP_D)?,
            clip,
            proj: Mat::load(model, "vis.projector.layers.weight", OUT_D, 2 * CLIP_D)?,
            proj_b: vec_of(model, "vis.projector.layers.bias", OUT_D)?,
            newline: vec_of(model, "vis.image_newline", OUT_D)?,
            view_sep: vec_of(model, "vis.view_seperator", OUT_D)?,
            pool,
            cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    fn tables(&self, g: usize) -> Arc<GridTables> {
        let mut c = self.cache.lock().unwrap();
        if let Some(t) = c.get(&g) {
            return t.clone();
        }
        let cg = g / 4;
        let mut clip_pos = self.clip_pos[..CLIP_D].to_vec();
        clip_pos.extend(resize_table(&self.clip_pos[CLIP_D..], CLIP_SIDE, cg, CLIP_D));
        let t = Arc::new(GridTables {
            pos: resize_table(&self.pos, SAM_SIDE, g, SAM_D),
            clip_pos,
            rel_global: SAM_GLOBAL
                .iter()
                .map(|&i| {
                    let b = &self.sam[i];
                    (resize_rel(&b.rel_h, SAM_HD, 2 * g - 1), resize_rel(&b.rel_w, SAM_HD, 2 * g - 1))
                })
                .collect(),
        });
        c.insert(g, t.clone());
        t
    }

    /// A SAM attention over groups of `n` tokens on an `s`×`s` grid with
    /// decomposed relative positions; `x`/`out` hold `groups·n` rows.
    #[allow(clippy::too_many_arguments)]
    fn sam_attention(&self, b: &SamBlock, x: &[f32], groups: usize, s: usize, rel_h: &[f32], rel_w: &[f32], out: &mut [f32]) {
        let pool = self.pool.as_deref();
        let n = s * s;
        let rows = groups * n;
        let (d, nh, hd) = (SAM_D, SAM_HEADS, SAM_HD);
        let mut qkv = vec![0f32; rows * 3 * d];
        b.qkv.matmat(x, rows, Some(&b.qkv_b), &mut qkv, pool);
        let scale = (hd as f32).sqrt().recip();
        let center = s - 1;
        let mut att = vec![0f32; rows * d];
        let aa = crate::pool::SendMut::new(att.as_mut_ptr());
        // qkv is [row, 3, heads, hd]; the score of (qi, kj) is
        // q·k·scale + q·Rh[qy-ky] + q·Rw[qx-kx].
        let one = |gi: usize, h: usize, qi: usize, sc: &mut [f32], rh: &mut [f32], rw: &mut [f32]| {
            let at = |r: usize, part: usize| &qkv[(gi * n + r) * 3 * d + part * d + h * hd..(gi * n + r) * 3 * d + part * d + (h + 1) * hd];
            let q = at(qi, 0);
            let (qy, qx) = (qi / s, qi % s);
            for ky in 0..s {
                let r = &rel_h[(qy + center - ky) * hd..(qy + center - ky + 1) * hd];
                rh[ky] = q.iter().zip(r).map(|(a, b)| a * b).sum();
            }
            for kx in 0..s {
                let r = &rel_w[(qx + center - kx) * hd..(qx + center - kx + 1) * hd];
                rw[kx] = q.iter().zip(r).map(|(a, b)| a * b).sum();
            }
            let mut mx = f32::NEG_INFINITY;
            for (kj, v) in sc.iter_mut().enumerate() {
                let k = at(kj, 1);
                let dot: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
                *v = dot * scale + rh[kj / s] + rw[kj % s];
                mx = mx.max(*v);
            }
            let mut den = 0f32;
            for v in sc.iter_mut() {
                *v = (*v - mx).exp();
                den += *v;
            }
            let inv = den.recip();
            // SAFETY: one (row, head) slot per task.
            let o = unsafe { std::slice::from_raw_parts_mut(aa.at((gi * n + qi) * d + h * hd), hd) };
            o.iter_mut().for_each(|v| *v = 0.0);
            for (kj, &w) in sc.iter().enumerate() {
                let w = w * inv;
                for (a, &v) in o.iter_mut().zip(at(kj, 2)) {
                    *a += w * v;
                }
            }
        };
        let total = rows * nh;
        let f = |lo: usize, hi: usize| {
            let (mut sc, mut rh, mut rw) = (vec![0f32; n], vec![0f32; s], vec![0f32; s]);
            for t in lo..hi {
                let (r, h) = (t / nh, t % nh);
                one(r / n, h, r % n, &mut sc, &mut rh, &mut rw);
            }
        };
        match pool {
            Some(p) => p.run_rows(total, &f),
            None => f(0, total),
        }
        b.proj.matmat(&att, rows, Some(&b.proj_b), out, pool);
    }

    /// `[3, side, side]` pixels → the SAM channels `[g/4², 1024]` (HWC) and
    /// the CLIP tokens without the class token, `[g/4², 1024]`.
    fn encode(&self, px: &[f32], side: usize) -> (Vec<f32>, Vec<f32>) {
        let pool = self.pool.as_deref();
        let g = side / PATCH;
        let n = g * g;
        let d = SAM_D;
        let tb = self.tables(g);
        let prof = std::env::var_os("CMF_OCR_PROF").is_some();
        let t0 = std::time::Instant::now();
        let mut rows = vec![0f32; n * 3 * PATCH * PATCH];
        for y in 0..g {
            for x in 0..g {
                let o = &mut rows[(y * g + x) * 3 * PATCH * PATCH..(y * g + x + 1) * 3 * PATCH * PATCH];
                let mut k = 0;
                for c in 0..3 {
                    for py in 0..PATCH {
                        let s = (c * side + y * PATCH + py) * side + x * PATCH;
                        o[k..k + PATCH].copy_from_slice(&px[s..s + PATCH]);
                        k += PATCH;
                    }
                }
            }
        }
        let mut h = vec![0f32; n * d];
        self.patch.matmat(&rows, n, Some(&self.patch_b), &mut h, pool);
        for (a, &p) in h.iter_mut().zip(&tb.pos) {
            *a += p;
        }
        let mut dev_out = Vec::new();
        let device_sam = crate::gpu::enabled_here()
            && std::env::var("CMF_OCR_GPU").as_deref() != Ok("0")
            && self.sam_device(&h, g, &tb, &mut dev_out);
        if device_sam {
            let t_sam = t0.elapsed().as_secs_f64();
            let clip = self.clip_forward(&dev_out, g / 4, &tb, pool);
            if prof {
                eprintln!("uocr tower {side}² (device): sam+neck {t_sam:.3} s, clip {:.3} s", t0.elapsed().as_secs_f64() - t_sam);
            }
            return (dev_out, clip);
        }
        let mut hn = vec![0f32; n * d];
        let mut res = vec![0f32; n * d];
        let mut ff = vec![0f32; n * SAM_INTER];
        let mut gi = 0usize;
        for b in &self.sam {
            layer_norm_rows(&h, &b.n1_w, &b.n1_b, SAM_EPS, &mut hn, pool);
            if b.window == 0 {
                let (rh, rw) = &tb.rel_global[gi];
                gi += 1;
                self.sam_attention(b, &hn, 1, g, rh, rw, &mut res);
            } else {
                // window_partition with bottom/right zero padding.
                let w = b.window;
                let gp = g.div_ceil(w) * w;
                let wn = gp / w;
                let mut win = vec![0f32; wn * wn * w * w * d];
                for wy in 0..wn {
                    for wx in 0..wn {
                        for iy in 0..w {
                            let y = wy * w + iy;
                            if y >= g {
                                continue;
                            }
                            for ix in 0..w {
                                let x = wx * w + ix;
                                if x >= g {
                                    continue;
                                }
                                let dst = ((wy * wn + wx) * w * w + iy * w + ix) * d;
                                win[dst..dst + d].copy_from_slice(&hn[(y * g + x) * d..(y * g + x + 1) * d]);
                            }
                        }
                    }
                }
                let mut wo = vec![0f32; win.len()];
                self.sam_attention(b, &win, wn * wn, w, &b.rel_h, &b.rel_w, &mut wo);
                debug_assert_eq!(b.rel_len, 2 * w - 1);
                for wy in 0..wn {
                    for wx in 0..wn {
                        for iy in 0..w {
                            let y = wy * w + iy;
                            if y >= g {
                                continue;
                            }
                            for ix in 0..w {
                                let x = wx * w + ix;
                                if x >= g {
                                    continue;
                                }
                                let src = ((wy * wn + wx) * w * w + iy * w + ix) * d;
                                res[(y * g + x) * d..(y * g + x + 1) * d].copy_from_slice(&wo[src..src + d]);
                            }
                        }
                    }
                }
            }
            for (a, &r) in h.iter_mut().zip(&res) {
                *a += r;
            }
            layer_norm_rows(&h, &b.n2_w, &b.n2_b, SAM_EPS, &mut hn, pool);
            b.fc1.matmat(&hn, n, Some(&b.fc1_b), &mut ff, pool);
            gelu_rows(&mut ff, pool);
            b.fc2.matmat(&ff, n, Some(&b.fc2_b), &mut res, pool);
            for (a, &r) in h.iter_mut().zip(&res) {
                *a += r;
            }
        }
        let t_sam = t0.elapsed().as_secs_f64();
        // The neck: 1×1 and 3×3 convs with LayerNorm2d, then two strided 3×3.
        let (mut c1, _) = conv_same(&h, g, d, &self.neck0, 256, 1, 1, 0);
        norm2d(&mut c1, 256, &self.neck1_w, &self.neck1_b);
        let (mut c2, _) = conv_same(&c1, g, 256, &self.neck2, 256, 3, 1, 1);
        norm2d(&mut c2, 256, &self.neck3_w, &self.neck3_b);
        let (c3, g2) = conv_same(&c2, g, 256, &self.net2, 512, 3, 2, 1);
        let (sam_out, g3) = conv_same(&c3, g2, 512, &self.net3, 1024, 3, 2, 1);
        let t_neck = t0.elapsed().as_secs_f64();
        let clip = self.clip_forward(&sam_out, g3, &tb, pool);
        if prof {
            eprintln!(
                "uocr tower {side}²: sam {t_sam:.3} s, neck {:.3} s, clip {:.3} s",
                t_neck - t_sam,
                t0.elapsed().as_secs_f64() - t_neck
            );
        }
        (sam_out, clip)
    }

    /// SAM on the device: `false` = refused.
    fn sam_device(&self, h: &[f32], g: usize, tb: &GridTables, out: &mut Vec<f32>) -> bool {
        let layers = self
            .sam
            .iter()
            .map(|b| crate::gpu::SamLayer {
                ln1_w: &b.n1_w,
                ln1_b: &b.n1_b,
                qkv: b.qkv.weights(),
                qkv_b: &b.qkv_b,
                proj: b.proj.weights(),
                proj_b: &b.proj_b,
                rel: &b.rel_cat,
                ln2_w: &b.n2_w,
                ln2_b: &b.n2_b,
                fc1: b.fc1.weights(),
                fc1_b: &b.fc1_b,
                fc2: b.fc2.weights(),
                fc2_b: &b.fc2_b,
                window: b.window,
            })
            .collect();
        let conv = |w: &'_ [f32], ci: usize, co: usize, k: usize, stride: usize, pad: usize, norm: bool| crate::gpu::SamConv {
            w: unsafe { std::slice::from_raw_parts(w.as_ptr(), w.len()) },
            ci,
            co,
            k,
            stride,
            pad,
            norm,
        };
        let args = crate::gpu::SamArgs {
            key: self as *const Self as u64,
            layers,
            convs: vec![
                conv(&self.conv_w[0], SAM_D, 256, 1, 1, 0, true),
                conv(&self.conv_w[1], 256, 256, 3, 1, 1, true),
                conv(&self.conv_w[2], 256, 512, 3, 2, 1, false),
                conv(&self.conv_w[3], 512, 1024, 3, 2, 1, false),
            ],
            norms: vec![&self.neck1_w, &self.neck1_b, &self.neck3_w, &self.neck3_b],
            rel_global: tb
                .rel_global
                .iter()
                .map(|(rh, rw)| {
                    let mut v = rh.clone();
                    v.extend_from_slice(rw);
                    v
                })
                .collect(),
            g,
            d: SAM_D,
            heads: SAM_HEADS,
            head_dim: SAM_HD,
            inter: SAM_INTER,
            eps: SAM_EPS,
        };
        crate::gpu::sam_encode(&args, h, out)
    }

    /// CLIP over SAM's channels `[cg², 1024]`, the tokens without the class.
    fn clip_forward(&self, sam_out: &[f32], cg: usize, tb: &GridTables, pool: Option<&Pool>) -> Vec<f32> {
        let nc = cg * cg + 1;
        let cd = CLIP_D;
        let mut x = vec![0f32; nc * cd];
        x[..cd].copy_from_slice(&self.cls);
        x[cd..].copy_from_slice(sam_out);
        for (a, &p) in x.iter_mut().zip(&tb.clip_pos) {
            *a += p;
        }
        let mut xn = vec![0f32; nc * cd];
        layer_norm_rows(&x, &self.pre_w, &self.pre_b, CLIP_EPS, &mut xn, pool);
        std::mem::swap(&mut x, &mut xn);
        let resident = crate::gpu::enabled_here() && std::env::var("CMF_OCR_GPU").as_deref() != Ok("0") && {
            let (cos, sin) = (vec![1f32; nc * CLIP_HD], vec![0f32; nc * CLIP_HD]);
            let layers = self
                .clip
                .iter()
                .map(|l| crate::gpu::VitLayer {
                    ln1_w: &l.n1_w,
                    ln1_b: &l.n1_b,
                    qkv: l.qkv.weights(),
                    qkv_b: &l.qkv_b,
                    proj: l.out.weights(),
                    proj_b: &l.out_b,
                    ln2_w: &l.n2_w,
                    ln2_b: &l.n2_b,
                    fc1: l.fc1.weights(),
                    fc1_b: &l.fc1_b,
                    fc2: l.fc2.weights(),
                    fc2_b: &l.fc2_b,
                })
                .collect();
            let args = crate::gpu::VitArgs {
                key: &self.clip as *const _ as u64,
                layers,
                n: nc,
                d: cd,
                heads: CLIP_HEADS,
                head_dim: CLIP_HD,
                inter: CLIP_INTER,
                eps: CLIP_EPS,
                gelu_tanh: false,
                quick_gelu: true,
                rope_halves: true,
                cos: &cos,
                sin: &sin,
                prof: false,
            };
            crate::gpu::vit_encode(&args, &mut x)
        };
        if resident {
            return x[cd..].to_vec();
        }
        let mut qkv = vec![0f32; nc * 3 * cd];
        let (mut qh, mut kh, mut vh) = (vec![0f32; nc * cd], vec![0f32; nc * cd], vec![0f32; nc * cd]);
        let mut att = vec![0f32; nc * cd];
        let mut tmp = vec![0f32; nc * cd];
        let mut cf = vec![0f32; nc * CLIP_INTER];
        let scale = (CLIP_HD as f32).sqrt().recip();
        for l in &self.clip {
            layer_norm_rows(&x, &l.n1_w, &l.n1_b, CLIP_EPS, &mut xn, pool);
            l.qkv.matmat(&xn, nc, Some(&l.qkv_b), &mut qkv, pool);
            // `xqkv.view(bsz, seqlen, 3, heads, head)`: q, k, v per token.
            for p in 0..nc {
                let row = &qkv[p * 3 * cd..(p + 1) * 3 * cd];
                for hh in 0..CLIP_HEADS {
                    let at = (hh * nc + p) * CLIP_HD;
                    qh[at..at + CLIP_HD].copy_from_slice(&row[hh * CLIP_HD..(hh + 1) * CLIP_HD]);
                    kh[at..at + CLIP_HD].copy_from_slice(&row[cd + hh * CLIP_HD..cd + (hh + 1) * CLIP_HD]);
                    vh[at..at + CLIP_HD].copy_from_slice(&row[2 * cd + hh * CLIP_HD..2 * cd + (hh + 1) * CLIP_HD]);
                }
            }
            crate::onevision2::attend_host(&qh, &kh, &vh, CLIP_HEADS, CLIP_HD, nc, scale, &mut att, pool);
            l.out.matmat(&att, nc, Some(&l.out_b), &mut tmp, pool);
            for (a, &b) in x.iter_mut().zip(&tmp) {
                *a += b;
            }
            layer_norm_rows(&x, &l.n2_w, &l.n2_b, CLIP_EPS, &mut xn, pool);
            l.fc1.matmat(&xn, nc, Some(&l.fc1_b), &mut cf, pool);
            // quick_gelu: x·σ(1.702·x)
            for v in cf.iter_mut() {
                *v /= 1.0 + (-1.702 * *v).exp();
            }
            l.fc2.matmat(&cf, nc, Some(&l.fc2_b), &mut tmp, pool);
            for (a, &b) in x.iter_mut().zip(&tmp) {
                *a += b;
            }
        }
        x[cd..].to_vec()
    }

    /// `[3, side, side]` → projected rows `[g², 1280]`.
    fn view_rows(&self, px: &[f32], side: usize) -> Vec<f32> {
        let (sam, clip) = self.encode(px, side);
        let n = sam.len() / CLIP_D;
        let mut joined = vec![0f32; n * 2 * CLIP_D];
        for i in 0..n {
            joined[i * 2 * CLIP_D..i * 2 * CLIP_D + CLIP_D].copy_from_slice(&clip[i * CLIP_D..(i + 1) * CLIP_D]);
            joined[i * 2 * CLIP_D + CLIP_D..(i + 1) * 2 * CLIP_D].copy_from_slice(&sam[i * CLIP_D..(i + 1) * CLIP_D]);
        }
        let mut out = vec![0f32; n * OUT_D];
        self.proj.matmat(&joined, n, Some(&self.proj_b), &mut out, self.pool.as_deref());
        out
    }

    /// One page's feature rows in prompt order: tiles (a newline after every
    /// row), the global view (same), the view separator.
    pub fn page_rows(&self, v: &PageViews) -> Vec<f32> {
        let d = OUT_D;
        let mut out = Vec::new();
        if !v.tiles.is_empty() {
            let tiles: Vec<Vec<f32>> = v.tiles.iter().map(|t| self.view_rows(t, IMAGE_SIZE)).collect();
            let tg = IMAGE_SIZE / PATCH / DOWNSAMPLE;
            for ty in 0..v.rows {
                for iy in 0..tg {
                    for tx in 0..v.cols {
                        let t = &tiles[ty * v.cols + tx];
                        out.extend_from_slice(&t[iy * tg * d..(iy + 1) * tg * d]);
                    }
                    out.extend_from_slice(&self.newline);
                }
            }
        }
        let g = self.view_rows(&v.global, BASE_SIZE);
        let gg = BASE_SIZE / PATCH / DOWNSAMPLE;
        for y in 0..gg {
            out.extend_from_slice(&g[y * gg * d..(y + 1) * gg * d]);
            out.extend_from_slice(&self.newline);
        }
        out.extend_from_slice(&self.view_sep);
        out
    }
}

fn tower_for(model: &Arc<CmfModel>) -> Result<Arc<UocrTower>, String> {
    static TOWER: std::sync::Mutex<Option<(usize, Arc<UocrTower>)>> = std::sync::Mutex::new(None);
    let key = Arc::as_ptr(model) as usize;
    let mut g = TOWER.lock().unwrap();
    if let Some((k, t)) = g.as_ref() {
        if *k == key {
            return Ok(t.clone());
        }
    }
    let t = Arc::new(UocrTower::from_model(model, Pool::from_env())?);
    *g = Some((key, t.clone()));
    Ok(t)
}

// ───────────────────────── the prompt ─────────────────────────

/// The prepared prompt.
pub struct PreparedUocrPrompt {
    pub token_ids: Vec<u32>,
    pub rows: Vec<f32>,
    /// `<PAGE>` markers the answer must repeat.
    pub pages: usize,
}

/// [`prepare`] on the pipeline's own file.
pub fn prepare_for_pipeline(pipeline: &Pipeline, sources: &[Value], prompt: Option<&str>) -> Result<PreparedUocrPrompt, String> {
    let model = pipeline.model.as_ref().ok_or("Unlimited-OCR requires a file-backed pipeline")?;
    prepare(model, pipeline, sources, prompt)
}

/// `_uocr_build_inputs` with the trained layout: `<PAGE><image>` per page,
/// then the instruction. One page is tiled, several take the global view.
pub fn prepare(
    model: &Arc<CmfModel>,
    pipeline: &Pipeline,
    sources: &[Value],
    prompt: Option<&str>,
) -> Result<PreparedUocrPrompt, String> {
    if sources.is_empty() {
        return Err("Unlimited-OCR reads images: pass at least one".into());
    }
    let tower = tower_for(model)?;
    let dash = pipeline.dash.as_ref().ok_or("Unlimited-OCR: the pipeline has no router")?;
    let (image_id, page_id) = (dash.image_id, dash.page_id);
    let tiled = sources.len() == 1;
    let tok = &pipeline.tokenizer;
    let mut ids = vec![BOS_ID];
    let mut media: Vec<Vec<f32>> = Vec::with_capacity(sources.len());
    let dump = std::env::var_os("CMF_OCR_DUMP").map(std::path::PathBuf::from);
    for (pi, source) in sources.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let frame = media::decode_rgb(&media::load_image_bytes(source)?)?;
        let v = page_views(&frame, tiled);
        let t_pre = t0.elapsed().as_secs_f64();
        let rows = tower.page_rows(&v);
        let n = page_tokens(v.cols, v.rows);
        if rows.len() != n * OUT_D {
            return Err(format!("Unlimited-OCR: {} rows for {n} image tokens", rows.len() / OUT_D));
        }
        if let Some(dir) = &dump {
            let write = |name: String, data: &[f32]| {
                let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(name), bytes).map_err(|e| format!("CMF_OCR_DUMP: {e}"))
            };
            write(format!("global_{pi}.f32"), &v.global)?;
            for (ti, t) in v.tiles.iter().enumerate() {
                write(format!("tile_{pi}_{ti}.f32"), t)?;
            }
            write(format!("rows_{pi}.f32"), &rows)?;
        }
        if dump.is_some() || std::env::var_os("CMF_OCR_PROF").is_some() {
            eprintln!(
                "uocr: page {pi} {}x{}, grid {}x{}, {n} tokens, preprocess {t_pre:.3} s, total {:.2} s",
                frame.width,
                frame.height,
                v.cols,
                v.rows,
                t0.elapsed().as_secs_f64()
            );
        }
        ids.push(page_id);
        ids.extend(std::iter::repeat_n(image_id, n));
        media.push(rows);
    }
    let instruction = prompt.filter(|p| !p.trim().is_empty()).unwrap_or("Multi page parsing.");
    let tail = tok.encode(instruction);
    ids.extend(tail);
    let h = pipeline.hidden_size;
    let mut rows = Vec::with_capacity(ids.len() * h);
    let mut flat = media.iter().flatten().copied();
    for &id in &ids {
        if id == image_id {
            rows.extend((&mut flat).take(h));
        } else {
            rows.extend(pipeline.embed_id(id));
        }
    }
    if rows.len() != ids.len() * h {
        return Err("Unlimited-OCR: placeholder and image row counts differ".into());
    }
    Ok(PreparedUocrPrompt { token_ids: ids, rows, pages: sources.len() })
}

/// The images and the instruction of a chat request: every image part in order,
/// the last text part. Video and audio are refused.
pub fn request_parts(messages: &[Value]) -> Result<(Vec<Value>, Option<String>), String> {
    let mut sources = Vec::new();
    let mut text = None;
    for message in messages {
        match message.get("content") {
            Some(Value::String(t)) => text = Some(t.clone()),
            Some(Value::Array(parts)) => {
                for part in parts {
                    let typ = part.get("type").and_then(Value::as_str).unwrap_or("");
                    if typ == "image" || typ == "image_url" || part.get("image").is_some() || part.get("image_url").is_some() {
                        let source = part.get("image_url").or_else(|| part.get("image")).unwrap_or(part);
                        sources.push(match source.as_str() {
                            Some(url) => serde_json::json!({"url": url}),
                            None => source.clone(),
                        });
                    } else if let Some(t) = part.get("text").and_then(Value::as_str) {
                        text = Some(t.to_string());
                    } else if typ != "text" {
                        return Err(format!("Unlimited-OCR: unsupported content part '{typ}' (text and images only)"));
                    }
                }
            }
            _ => {}
        }
    }
    Ok((sources, text))
}

/// `_uocr_page_count_guard`: in place on the logits, over the generated ids.
pub struct PageGuard {
    pub expected: usize,
    pub page_id: u32,
    pub eos_id: u32,
    pub max_page_tokens: usize,
    marker_cum: usize,
    scanned: usize,
    page_len: usize,
}

impl PageGuard {
    pub fn new(expected: usize, page_id: u32, eos_id: u32) -> Self {
        Self { expected, page_id, eos_id, max_page_tokens: 20000, marker_cum: 0, scanned: 0, page_len: 0 }
    }

    pub fn apply(&mut self, logits: &mut [f32], generated: &[u32]) {
        if self.expected == 0 {
            return;
        }
        if generated.len() > self.scanned {
            let tail = &generated[self.scanned..];
            let m = tail.iter().filter(|&&t| t == self.page_id).count();
            if m > 0 {
                self.marker_cum += m;
                self.page_len = tail.iter().rev().position(|&t| t == self.page_id).unwrap_or(0);
            } else {
                self.page_len += tail.len();
            }
            self.scanned = generated.len();
        }
        let only = |logits: &mut [f32], t: u32| {
            let keep = logits[t as usize];
            logits.iter_mut().for_each(|v| *v = f32::NEG_INFINITY);
            logits[t as usize] = keep;
        };
        if self.marker_cum > self.expected {
            only(logits, self.eos_id);
        } else if self.max_page_tokens > 0 && self.page_len >= self.max_page_tokens {
            let t = if self.marker_cum < self.expected { self.page_id } else { self.eos_id };
            only(logits, t);
        } else if self.marker_cum < self.expected {
            logits[self.eos_id as usize] = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_tokens_match_the_trained_layout() {
        assert_eq!(page_tokens(1, 1), 273);
        // (10·3 + 1) · (10·4) local rows on top of the global view
        assert_eq!(page_tokens(3, 4), 273 + 31 * 40);
    }

    #[test]
    fn page_coordinates_follow_the_reference() {
        // BOS, <PAGE>, 3 image, <PAGE>, 2 image, 2 instruction
        let (bos, pg, im, t) = (0u32, 9u32, 7u32, 5u32);
        let ids = [bos, pg, im, im, im, pg, im, im, t, t];
        let mut r = PageRope::begin(&ids, im, pg, 4).unwrap();
        let c: Vec<_> = (0..ids.len()).map(|i| r.prompt_coords(i)).collect();
        assert_eq!(c[0], (0, 0));
        assert_eq!(c[1], (0, 1));
        assert_eq!(c[4], (3, 1));
        assert_eq!(c[5], (0, 2));
        assert_eq!(c[7], (2, 2));
        // the instruction shares the reserved page 0 and keeps its count
        assert_eq!(c[8], (1, 0));
        assert_eq!(c[9], (2, 0));
        // first response marker opens page 1 at that page's prefill length
        assert_eq!(r.decode_step(pg), (4, 1));
        assert_eq!(r.decode_step(t), (5, 1));
        assert_eq!(r.decode_step(pg), (3, 2));
    }

    #[test]
    fn decode_candidates_follow_the_response_page() {
        let (bos, pg, im, t) = (0u32, 9u32, 7u32, 5u32);
        let mut ids = vec![bos, pg];
        ids.extend(std::iter::repeat_n(im, 6));
        ids.push(pg);
        ids.extend(std::iter::repeat_n(im, 7));
        ids.push(t);
        let mut r = PageRope::begin(&ids, im, pg, 4).unwrap();
        let mut allow = Vec::new();
        // before the first response marker every block is open
        r.decode_allow(5, &mut allow);
        assert!(allow.iter().all(|&a| a));
        r.decode_step(pg);
        r.decode_allow(5, &mut allow);
        assert_eq!(allow, vec![true, true, false, false, false]);
        r.decode_step(pg);
        r.decode_allow(5, &mut allow);
        assert_eq!(allow, vec![false, false, true, true, false]);
    }

    #[test]
    fn entmax_matches_a_hand_case() {
        // Two equal scores share the mass; a far one gets none.
        let mut y = [0f32; 3];
        entmax15_sorted(&[0.0, 0.0, -50.0], &mut y);
        assert!((y[0] - 0.5).abs() < 1e-6, "{y:?}");
        assert!((y[1] - 0.5).abs() < 1e-6);
        assert_eq!(y[2], 0.0);
        // Sums to one on a mixed row.
        let mut z = [0f32; 4];
        entmax15_sorted(&[3.0, 2.5, 1.0, 0.0], &mut z);
        assert!((z.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn route_centers_the_bias_on_the_read_set() {
        let p = RouteParams { idx_scale: 1.0, floor: 1e-4, sigma: 1.0, k: 2, top_m: 16 };
        let scores = [Some(1.0), Some(5.0), None, Some(4.0)];
        let allow = [true, true, true, true];
        let (mut w, mut pick) = (Vec::new(), Vec::new());
        route_select(&scores, &allow, &p, &mut w, &mut pick);
        assert_eq!(pick.len(), 2);
        assert_eq!(pick[0].0, 1);
        assert_eq!(pick[1].0, 3);
        assert!((pick[0].1 + pick[1].1).abs() < 1e-5);
        // A page with candidates shuts the rest out.
        let allow = [true, false, false, false];
        route_select(&scores, &allow, &p, &mut w, &mut pick);
        assert_eq!(pick.len(), 1);
        assert_eq!(pick[0].0, 0);
    }

    #[test]
    fn closest_grid_picks_the_page_aspect() {
        // Reference `dynamic_preprocess` for a 1191×1684 page.
        let (c, r) = closest_grid(1191, 1684, 640);
        assert!((c as f64 / r as f64 - 1191.0 / 1684.0).abs() < 0.05, "{c}x{r}");
    }

    #[test]
    fn conv_reads_kernels_as_channel_out_in_y_x() {
        // 1 in, 1 out, 3×3 kernel: a centre-only kernel is the identity.
        let mut w = vec![0f32; 9];
        w[4] = 2.0;
        let x = [1.0, 2.0, 3.0, 4.0];
        let (y, g) = conv_same(&x, 2, 1, &w, 1, 3, 1, 1);
        assert_eq!(g, 2);
        assert_eq!(y, vec![2.0, 4.0, 6.0, 8.0]);
    }
}
