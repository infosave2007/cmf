//! SenseNova-Loop (`sensenova_loop`): a Qwen3.5 hybrid (Gated DeltaNet +
//! gated full attention) run as a LOOPED stack with manifold-constrained
//! hyper-connections (mHC) inside the loop, plus the Qwen3-VL style vision
//! tower (`qwen3vis`).
//!
//! ## Execution plan
//!
//! `loop_settings = {start, end, eval_count}`: physical layers
//! `[0, start)`, then `[start, end)` visited `eval_count` times, then
//! `[end, layers)`. Every visit is its own VIRTUAL layer with its own KV
//! cache / recurrent state (the reference maps each visit to a separate
//! cache slot); the weights are shared. SenseNova-Loop-9B: 4 + 24·2 + 4 =
//! 56 virtual layers over 32 physical ones.
//!
//! ## mHC (`mhc = {start, end, streams, iterations, pre_scale}`)
//!
//! Inside `[start, end)` the residual is `streams` copies of the hidden.
//! At the first mHC layer of a visit the incoming hidden is copied into
//! every stream; at the last one the streams are averaged back. Each
//! sub-layer (attention, FFN) has its own FP32 router:
//!
//! ```text
//! x̂ = flat(R) / rms(flat(R))                     (no weight, ε 1e-6)
//! l = W·x̂ + b                                    (2S + S² logits)
//! pre = σ(l[0:S]); post = 2σ(l[S:2S]); mix = Sinkhorn(l[2S:], iters)
//! branch = Σ_s pre_s·R_s (· pre_scale)           → norm → sub-layer → y
//! R_i ← Σ_j mix_ij·R_j + post_i·y
//! ```
//!
//! Sinkhorn runs in log space: each iteration subtracts the row
//! logsumexp, then the column logsumexp; the result is exponentiated.
//!
//! ## Images
//!
//! The tower is `qwen3vis` (27 blocks at 1152, no deepstack). The text
//! model uses interleaved MRoPE (sections 11/11/10 over the 32 rotary
//! frequencies): image tokens carry (t, h, w) positions. They are applied
//! in the prompt prefill only, shifted so that every token after the last
//! image sits at its plain sequence index — attention depends on position
//! DIFFERENCES only, so decode needs no offset (see [`mrope_positions`]).

use cortiq_core::CmfModel;
use serde_json::Value;
use std::sync::Arc;

/// `loop_settings`: physical `[start, end)` visited `count` times.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoopSpan {
    pub start: usize,
    pub end: usize,
    pub count: usize,
}

/// `mhc` settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MhcSettings {
    pub start: usize,
    pub end: usize,
    pub streams: usize,
    pub iterations: usize,
    pub pre_scale: f32,
    pub fixed_identity: bool,
}

/// Virtual → physical layer order: `[0, s)`, `[s, e)`×count, `[e, n)`.
pub fn execution_plan(layers: usize, span: LoopSpan) -> Vec<usize> {
    let mut v: Vec<usize> = (0..span.start).collect();
    for _ in 0..span.count {
        v.extend(span.start..span.end);
    }
    v.extend(span.end..layers);
    v
}

fn need(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| format!("sensenova_loop: missing '{key}'"))
}

/// The plan and mHC settings from `provenance.sensenova_loop.config`.
pub fn settings(model: &CmfModel) -> Result<Option<(LoopSpan, Option<MhcSettings>)>, String> {
    let Some(cfg) = model
        .header
        .provenance
        .as_ref()
        .and_then(|p| p.get("sensenova_loop"))
        .and_then(|p| p.get("config"))
    else {
        return Ok(None);
    };
    let ls = cfg.get("loop_settings").ok_or("sensenova_loop: no loop_settings")?;
    let count = need(ls, "eval_count")?;
    if ls.get("counts").and_then(Value::as_array).map(|a| a.len()) != Some(1) {
        return Err("sensenova_loop: only fixed-count loop checkpoints are supported".into());
    }
    let span = LoopSpan { start: need(ls, "start")?, end: need(ls, "end")?, count };
    let layers = model.header.arch.num_layers;
    if !(span.start < span.end && span.end <= layers && count >= 1) {
        return Err("sensenova_loop: invalid loop interval".into());
    }
    let mhc = match cfg.get("mhc") {
        None | Some(Value::Null) => None,
        Some(m) => {
            let s = MhcSettings {
                start: need(m, "start")?,
                end: need(m, "end")?,
                streams: need(m, "streams")?,
                iterations: need(m, "iterations")?,
                pre_scale: m.get("pre_scale").and_then(Value::as_f64).unwrap_or(1.0) as f32,
                fixed_identity: m.get("fixed_identity").and_then(Value::as_bool).unwrap_or(false),
            };
            if !(s.start < s.end && s.end <= layers && (s.streams == 2 || s.streams == 4) && s.iterations >= 1) {
                return Err("sensenova_loop: invalid mhc settings".into());
            }
            // The reference forbids an mHC range across a loop phase.
            let phases = [0, span.start, span.end, layers];
            if !phases.windows(2).any(|w| w[0] <= s.start && s.end <= w[1]) {
                return Err("sensenova_loop: mhc range crosses a loop phase".into());
            }
            Some(s)
        }
    };
    Ok(Some((span, mhc)))
}

/// One FP32 router: `W [2S + S², S·dim]`, `b [2S + S²]`.
pub struct MhcRouter {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

/// The routers of every mHC layer and the stream geometry.
pub struct Mhc {
    pub s: MhcSettings,
    pub dim: usize,
    /// Per physical layer: (attention router, FFN router).
    pub routers: Vec<Option<(MhcRouter, MhcRouter)>>,
}

/// One sub-layer's routing result: the residual update `post` and `mix`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MhcCoef {
    pub post: [f32; 4],
    pub mix: [[f32; 4]; 4],
}

impl Mhc {
    pub fn from_model(model: &Arc<CmfModel>, s: MhcSettings, layers: usize, dim: usize) -> Result<Self, String> {
        let out = 2 * s.streams + if s.fixed_identity { 0 } else { s.streams * s.streams };
        let mut routers = Vec::with_capacity(layers);
        for p in 0..layers {
            if p < s.start || p >= s.end {
                routers.push(None);
                continue;
            }
            let load = |which: &str| -> Result<MhcRouter, String> {
                let w = crate::dit::cmf_f32(model, &format!("model.layers.{p}.{which}.weight"))?;
                let b = crate::dit::cmf_f32(model, &format!("model.layers.{p}.{which}.bias"))?;
                if w.len() != out * s.streams * dim || b.len() != out {
                    return Err(format!("sensenova_loop: layer {p} {which} is not [{out}, {}]", s.streams * dim));
                }
                Ok(MhcRouter { w, b })
            };
            routers.push(Some((load("mhc_attn")?, load("mhc_ffn")?)));
        }
        Ok(Mhc { s, dim, routers })
    }

    /// Whether physical layer `p` carries mHC routers.
    #[inline]
    pub fn on(&self, p: usize) -> bool {
        self.routers.get(p).is_some_and(Option::is_some)
    }

    #[inline]
    pub fn enters(&self, p: usize) -> bool {
        p == self.s.start
    }

    #[inline]
    pub fn leaves(&self, p: usize) -> bool {
        p + 1 == self.s.end
    }

    fn router(&self, p: usize, ffn: bool) -> &MhcRouter {
        let (a, f) = self.routers[p].as_ref().expect("mhc layer");
        if ffn { f } else { a }
    }

    /// Route one row: `r` is `[S·dim]` (stream-major); writes the branch
    /// (`[dim]`) and returns the merge coefficients.
    pub fn route(&self, p: usize, ffn: bool, r: &[f32], branch: &mut [f32]) -> MhcCoef {
        let (ns, d) = (self.s.streams, self.dim);
        let rt = self.router(p, ffn);
        let n = ns * d;
        let ss: f64 = r[..n].iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv = (1.0 / (ss / n as f64 + 1e-6).sqrt()) as f32;
        let out = rt.b.len();
        let mut l = [0f32; 24];
        for (k, lk) in l.iter_mut().enumerate().take(out) {
            let row = &rt.w[k * n..(k + 1) * n];
            let mut acc = 0f64;
            for (&a, &x) in row.iter().zip(&r[..n]) {
                acc += (a * (x * inv)) as f64;
            }
            *lk = acc as f32 + rt.b[k];
        }
        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        let mut c = MhcCoef::default();
        let mut pre = [0f32; 4];
        for s in 0..ns {
            pre[s] = sig(l[s]);
            c.post[s] = 2.0 * sig(l[ns + s]);
        }
        if self.s.fixed_identity {
            for i in 0..ns {
                c.mix[i][i] = 1.0;
            }
        } else {
            let mut m = [[0f64; 4]; 4];
            for i in 0..ns {
                for j in 0..ns {
                    m[i][j] = l[2 * ns + i * ns + j] as f64;
                }
            }
            for _ in 0..self.s.iterations {
                for row in m.iter_mut().take(ns) {
                    let mx = row[..ns].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let lse = mx + row[..ns].iter().map(|&v| (v - mx).exp()).sum::<f64>().ln();
                    for v in row[..ns].iter_mut() {
                        *v -= lse;
                    }
                }
                for j in 0..ns {
                    let mx = (0..ns).fold(f64::NEG_INFINITY, |a, i| a.max(m[i][j]));
                    let lse = mx + (0..ns).map(|i| (m[i][j] - mx).exp()).sum::<f64>().ln();
                    for row in m.iter_mut().take(ns) {
                        row[j] -= lse;
                    }
                }
            }
            for i in 0..ns {
                for j in 0..ns {
                    c.mix[i][j] = m[i][j].exp() as f32;
                }
            }
        }
        let scale = self.s.pre_scale;
        for (e, b) in branch[..d].iter_mut().enumerate() {
            let mut v = 0f32;
            for s in 0..ns {
                v += pre[s] * r[s * d + e];
            }
            *b = v * scale;
        }
        c
    }

    /// `R_i ← Σ_j mix_ij·R_j + post_i·y` for one row.
    pub fn merge(&self, r: &mut [f32], y: &[f32], c: &MhcCoef) {
        let (ns, d) = (self.s.streams, self.dim);
        let mut tmp = [0f32; 4];
        for e in 0..d {
            for (i, t) in tmp.iter_mut().enumerate().take(ns) {
                let mut v = c.post[i] * y[e];
                for j in 0..ns {
                    v += c.mix[i][j] * r[j * d + e];
                }
                *t = v;
            }
            for i in 0..ns {
                r[i * d + e] = tmp[i];
            }
        }
    }

    /// The streams of a row entering the mHC range: `S` copies of `h`.
    pub fn enter(&self, r: &mut [f32], h: &[f32]) {
        let d = self.dim;
        for s in 0..self.s.streams {
            r[s * d..(s + 1) * d].copy_from_slice(&h[..d]);
        }
    }

    /// The hidden leaving the mHC range: the streams' mean.
    pub fn leave(&self, r: &[f32], h: &mut [f32]) {
        let (ns, d) = (self.s.streams, self.dim);
        for e in 0..d {
            let mut v = 0f32;
            for s in 0..ns {
                v += r[s * d + e];
            }
            h[e] = v / ns as f32;
        }
    }
}

/// The device-graph descriptor of physical layer `p`'s mHC (None outside
/// the range or without mHC).
pub fn graph_mhc(m: Option<&Mhc>, p: usize) -> Option<crate::gpu::GraphMhc<'_>> {
    let m = m?;
    let (a, f) = m.routers.get(p)?.as_ref()?;
    Some(crate::gpu::GraphMhc {
        attn_w: &a.w,
        attn_b: &a.b,
        ffn_w: &f.w,
        ffn_b: &f.b,
        enter: m.enters(p),
        leave: m.leaves(p),
        iters: m.s.iterations,
        pre_scale: m.s.pre_scale,
    })
}

/// Interleaved MRoPE axis of rotary frequency `i` for sections `[t, h, w]`
/// (Qwen3-VL `apply_interleaved_mrope`): h at `1, 4, …` below `3·sec_h`,
/// w at `2, 5, …` below `3·sec_w`, t everywhere else.
pub fn mrope_axis(i: usize, sections: [usize; 3]) -> usize {
    if i % 3 == 1 && i < 3 * sections[1] {
        1
    } else if i % 3 == 2 && i < 3 * sections[2] {
        2
    } else {
        0
    }
}

/// Qwen3.5 `get_rope_index` for one sequence: per token `(t, h, w)`
/// positions. `images` lists `(first_token, grid_t, grid_h, grid_w)` of each
/// image's merged-token span (grid in merged units). Text advances all
/// three axes together; an image starts at the running position and lays
/// out `t·h·w`; the next text resumes at max + 1. The result is shifted so
/// the tokens after the last image sit at their plain index (the decode
/// position), keeping every difference — RoPE sees only differences.
pub fn mrope_positions(n: usize, images: &[(usize, usize, usize, usize)]) -> Vec<[i64; 3]> {
    let mut out = Vec::with_capacity(n);
    let mut next = 0i64;
    let mut s = 0usize;
    for &(first, gt, gh, gw) in images {
        while s < first && s < n {
            out.push([next; 3]);
            next += 1;
            s += 1;
        }
        let base = next;
        for t in 0..gt {
            for y in 0..gh {
                for x in 0..gw {
                    out.push([base + t as i64, base + y as i64, base + x as i64]);
                }
            }
        }
        s += gt * gh * gw;
        next = base + gt.max(gh).max(gw) as i64;
    }
    while s < n {
        out.push([next; 3]);
        next += 1;
        s += 1;
    }
    out.truncate(n);
    // Shift: the last token's position becomes n − 1.
    if let Some(last) = out.last().map(|p| p[0]) {
        let shift = (n as i64 - 1) - last;
        for p in &mut out {
            for v in p.iter_mut() {
                *v += shift;
            }
        }
    }
    out
}

/// Whether `model` is a SenseNova-Loop file with its vision tower.
pub fn has_vision(model: &CmfModel) -> bool {
    model.header.arch.arch_name == "sensenova_loop" && model.tensor("vis.patch_embed.weight").is_some()
}

/// The prepared prompt: ids, one row per position, and the rows' MRoPE
/// positions (None for a text-only prompt).
pub struct PreparedSnPrompt {
    pub token_ids: Vec<u32>,
    pub rows: Option<Vec<f32>>,
    pub mrope: Option<Vec<[i64; 3]>>,
}

/// The image processor settings (`Qwen2VLImageProcessor`).
struct SnProcessor {
    min_pixels: usize,
    max_pixels: usize,
    mean: [f32; 3],
    std: [f32; 3],
    rescale: f32,
}

fn processor(model: &CmfModel) -> Result<SnProcessor, String> {
    let p = model
        .header
        .provenance
        .as_ref()
        .and_then(|p| p.pointer("/sensenova_loop/processor"))
        .filter(|p| !p.is_null())
        .ok_or("sensenova_loop: the file carries no image processor config (reconvert it)")?;
    let px = |k: &str, alt: &str| {
        p.get(k)
            .and_then(Value::as_u64)
            .or_else(|| p.pointer(alt).and_then(Value::as_u64))
            .map(|v| v as usize)
            .ok_or_else(|| format!("sensenova_loop processor: missing {k}"))
    };
    let tri = |k: &str| -> Result<[f32; 3], String> {
        let a = p.get(k).and_then(Value::as_array).filter(|a| a.len() == 3).ok_or_else(|| format!("sensenova_loop processor: {k}"))?;
        Ok([0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0) as f32))
    };
    Ok(SnProcessor {
        min_pixels: px("min_pixels", "/size/shortest_edge")?,
        max_pixels: px("max_pixels", "/size/longest_edge")?,
        mean: tri("image_mean")?,
        std: tri("image_std")?,
        rescale: p.get("rescale_factor").and_then(Value::as_f64).unwrap_or(1.0 / 255.0) as f32,
    })
}

/// Replace image parts with `{"type": "image"}` and collect the sources.
fn normalize(messages: &[Value]) -> Result<(Vec<Value>, Vec<Value>), String> {
    let mut messages = messages.to_vec();
    let mut items = Vec::new();
    for message in &mut messages {
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for part in parts {
            if part.is_string() {
                continue;
            }
            let typ = part.get("type").and_then(Value::as_str).unwrap_or("");
            if typ == "image" || typ == "image_url" || part.get("image").is_some() || part.get("image_url").is_some() {
                let source = part.get("image_url").or_else(|| part.get("image")).unwrap_or(part);
                items.push(match source.as_str() {
                    Some(url) => serde_json::json!({"url": url}),
                    None => source.clone(),
                });
                *part = serde_json::json!({"type": "image"});
            } else if typ != "text" && part.get("text").is_none() {
                return Err(format!("SenseNova-Loop: unsupported content part '{typ}' (text and images only)"));
            }
        }
    }
    Ok((messages, items))
}

/// Whether any message carries an image part.
pub fn has_media(messages: &[Value]) -> Result<bool, String> {
    Ok(!normalize(messages)?.1.is_empty())
}

/// The tower of `model`, built once and kept for the next request.
fn tower_for(model: &Arc<CmfModel>) -> Result<Arc<crate::qwen3vis::VisionTower>, String> {
    static TOWER: std::sync::Mutex<Option<(usize, Arc<crate::qwen3vis::VisionTower>)>> = std::sync::Mutex::new(None);
    let key = Arc::as_ptr(model) as usize;
    let mut g = TOWER.lock().unwrap();
    if let Some((k, t)) = g.as_ref() {
        if *k == key {
            return Ok(t.clone());
        }
    }
    let t = Arc::new(crate::qwen3vis::VisionTower::from_cmf_device(model)?);
    *g = Some((key, t.clone()));
    Ok(t)
}

/// [`prepare`] on the pipeline's own file.
pub fn prepare_for_pipeline(
    pipeline: &crate::Pipeline,
    messages: &[Value],
    tools: Option<&[Value]>,
    thinking: Option<bool>,
    image_max_pixels: Option<usize>,
) -> Result<PreparedSnPrompt, String> {
    let model = pipeline.model.as_ref().ok_or("SenseNova-Loop images require a file-backed pipeline")?;
    prepare(model, pipeline, messages, tools, thinking, image_max_pixels)
}

/// Render `messages`, run each image through the tower, expand every
/// `<|image_pad|>` to the image's merged-token count, and assemble the rows
/// and their MRoPE positions.
pub fn prepare(
    model: &Arc<CmfModel>,
    pipeline: &crate::Pipeline,
    messages: &[Value],
    tools: Option<&[Value]>,
    thinking: Option<bool>,
    image_max_pixels: Option<usize>,
) -> Result<PreparedSnPrompt, String> {
    let (messages, items) = normalize(messages)?;
    let tok = &pipeline.tokenizer;
    if items.is_empty() {
        let ids = tok.try_apply_chat_template_json(&messages, tools, thinking)?;
        return Ok(PreparedSnPrompt { token_ids: ids, rows: None, mrope: None });
    }
    let text = tok
        .render_chat_json(&messages, tools, thinking)
        .ok_or("SenseNova-Loop: the chat template failed to render")?;
    let tower = tower_for(model)?;
    let proc = processor(model)?;
    if tower.out_hidden != pipeline.hidden_size {
        return Err("SenseNova-Loop: the merger width differs from the text model".into());
    }
    let image_id = model
        .header
        .provenance
        .as_ref()
        .and_then(|p| p.pointer("/sensenova_loop/config/image_token_id"))
        .and_then(Value::as_u64)
        .ok_or("sensenova_loop: config has no image_token_id")? as u32;
    let dump = std::env::var_os("CMF_SN_DUMP").map(std::path::PathBuf::from);
    const PAD: &str = "<|image_pad|>";
    let (p, m) = (tower.patch_size, tower.merge);
    let factor = p * m;
    let mut sources = items.into_iter();
    let mut expanded = String::with_capacity(text.len());
    let mut media_rows: Vec<f32> = Vec::new();
    let mut grids: Vec<(usize, usize)> = Vec::new();
    let mut rest = text.as_str();
    let mut k = 0usize;
    while let Some(at) = rest.find(PAD) {
        expanded.push_str(&rest[..at]);
        rest = &rest[at + PAD.len()..];
        let source = sources.next().ok_or("SenseNova-Loop: more <|image_pad|> placeholders than images")?;
        let t0 = std::time::Instant::now();
        let frame = crate::media::decode_rgb(&crate::media::load_image_bytes(&source)?)?;
        let max_px = image_max_pixels.map_or(proc.max_pixels, |v| v.min(proc.max_pixels));
        let (hb, wb) = crate::lfm3_vl::smart_resize(frame.height, frame.width, factor, proc.min_pixels, max_px)?;
        let resized = if (hb, wb) == (frame.height, frame.width) {
            frame.clone()
        } else {
            crate::egemma2_vision::resize_bicubic_aa(&frame, hb, wb)
        };
        let plane = hb * wb;
        let mut chw = vec![0f32; 3 * plane];
        for c in 0..3 {
            let mean = proc.mean[c] / proc.rescale;
            let sd = proc.std[c] / proc.rescale;
            for (i, px) in resized.data.chunks_exact(3).enumerate() {
                chw[c * plane + i] = (px[c] as f32 - mean) / sd;
            }
        }
        let cfg = crate::mimo_vision::MimoProcessorConfig {
            patch_size: p,
            merge_size: m,
            temporal_patch_size: tower.temporal_patch,
            ..Default::default()
        };
        let frames: Vec<&[f32]> = (0..tower.temporal_patch).map(|_| chw.as_slice()).collect();
        let (patches, _gt, gh, gw) = crate::mimo_vision::patchify(&frames, hb, wb, &cfg)?;
        let (rows, _deep) = tower.forward(&patches, gh, gw);
        let tokens = gh * gw / (m * m);
        if let Some(dir) = &dump {
            let write = |name: String, v: &[f32]| {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(name), bytes).map_err(|e| format!("CMF_SN_DUMP: {e}"))
            };
            write(format!("pixels_{k}.f32"), &patches)?;
            write(format!("rows_{k}.f32"), &rows)?;
        }
        if dump.is_some() || std::env::var_os("CMF_SN_PROF").is_some() {
            eprintln!(
                "sensenova: image {k} {}x{} resized to {wb}x{hb}, grid {gh}x{gw}, {tokens} tokens, tower {:.2} s",
                frame.width,
                frame.height,
                t0.elapsed().as_secs_f64()
            );
        }
        expanded.push_str(&PAD.repeat(tokens));
        media_rows.extend(rows);
        grids.push((gh / m, gw / m));
        k += 1;
    }
    expanded.push_str(rest);
    if sources.next().is_some() {
        return Err("SenseNova-Loop: the template rendered fewer placeholders than images".into());
    }
    let ids = tok.with_bos(tok.encode(&expanded));
    let h = pipeline.hidden_size;
    let mut rows = Vec::with_capacity(ids.len() * h);
    let mut spans = Vec::new();
    let mut at = 0usize;
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == image_id {
            let (gh, gw) = *grids.get(spans.len()).ok_or("SenseNova-Loop: more image tokens than images")?;
            spans.push((i, 1, gh, gw));
            for _ in 0..gh * gw {
                if ids.get(i) != Some(&image_id) {
                    return Err("SenseNova-Loop: an image span is shorter than its grid".into());
                }
                let row = media_rows.get(at..at + h).ok_or("SenseNova-Loop: placeholder and image row counts differ")?;
                rows.extend_from_slice(row);
                at += h;
                i += 1;
            }
        } else {
            rows.extend(pipeline.embed_id(ids[i]));
            i += 1;
        }
    }
    if at != media_rows.len() {
        return Err("SenseNova-Loop: unused image rows".into());
    }
    let mrope = mrope_positions(ids.len(), &spans);
    Ok(PreparedSnPrompt { token_ids: ids, rows: Some(rows), mrope: Some(mrope) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_visits_the_loop_twice() {
        let p = execution_plan(32, LoopSpan { start: 4, end: 28, count: 2 });
        assert_eq!(p.len(), 56);
        assert_eq!(&p[..6], &[0, 1, 2, 3, 4, 5]);
        assert_eq!(p[27], 27);
        assert_eq!(p[28], 4);
        assert_eq!(&p[52..], &[28, 29, 30, 31]);
    }

    #[test]
    fn mrope_axes_follow_the_interleave() {
        let sec = [11, 11, 10];
        let axes: Vec<usize> = (0..32).map(|i| mrope_axis(i, sec)).collect();
        assert_eq!(axes.iter().filter(|&&a| a == 1).count(), 11);
        assert_eq!(axes.iter().filter(|&&a| a == 2).count(), 10);
        assert_eq!(&axes[..7], &[0, 1, 2, 0, 1, 2, 0]);
        assert_eq!(axes[30], 0);
        assert_eq!(axes[31], 1);
    }

    #[test]
    fn mrope_positions_resume_after_an_image_and_end_plain() {
        // 2 text, image 1×2×3 (6 tokens), 2 text
        let p = mrope_positions(10, &[(2, 1, 2, 3)]);
        // reference: 0,1 | base 2: t=2, h 2..3, w 2..4 | next = 2+3 = 5: 5, 6
        // shift so the last is 9: +3
        assert_eq!(p[0], [3, 3, 3]);
        assert_eq!(p[2], [5, 5, 5]);
        assert_eq!(p[7], [5, 6, 7]);
        assert_eq!(p[8], [8, 8, 8]);
        assert_eq!(p[9], [9, 9, 9]);
    }

    #[test]
    fn sinkhorn_makes_mix_doubly_stochastic() {
        let s = MhcSettings { start: 0, end: 1, streams: 2, iterations: 20, pre_scale: 1.0, fixed_identity: false };
        let d = 3;
        let out = 2 * 2 + 4;
        let mut w = vec![0f32; out * 2 * d];
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i * 7919) % 13) as f32 * 0.1 - 0.6;
        }
        let b = vec![0.1, -0.2, 0.3, 0.0, 1.0, -1.0, 0.5, 0.2];
        let m = Mhc { s, dim: d, routers: vec![Some((MhcRouter { w: w.clone(), b: b.clone() }, MhcRouter { w, b }))] };
        let r = vec![1.0, -2.0, 0.5, 0.3, 0.7, -1.1];
        let mut br = vec![0f32; d];
        let c = m.route(0, false, &r, &mut br);
        for i in 0..2 {
            let row: f32 = (0..2).map(|j| c.mix[i][j]).sum();
            let col: f32 = (0..2).map(|j| c.mix[j][i]).sum();
            assert!((row - 1.0).abs() < 1e-4 && (col - 1.0).abs() < 1e-4);
        }
        assert!(c.post.iter().take(2).all(|&p| p > 0.0 && p < 2.0));
    }
}
