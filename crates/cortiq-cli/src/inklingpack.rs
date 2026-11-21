//! Pack thinkingmachines/Inkling-2-Mini (`model_type: inkling2_loop`) into one `.cmf`.
//!
//! `cortiq convert --model <dir with config.json> --output x.cmf --quant P`
//! detects the model type and comes here; the generic LLM walk is never
//! touched. The runtime is `cortiq_engine::inkling`.
//!
//! The source is the release checkpoint: routed experts in NVFP4 (layers
//! whose `w13_weight` is U8: E2M1 pairs, one E4M3 scale per 16 values, a
//! per-expert factor `amax / outer_scale_denominator`) or MXFP8 (F8_E4M3
//! values, one E8M0 scale per 32), every other matrix BF16. The experts are
//! decoded as transformers' `Inkling2LoopDequantizeExperts` does:
//! `(E2M1 · scale) · factor`, and `w13` holds gate and up rows interleaved
//! (even rows gate, odd rows up).
//!
//! ## Layout
//!
//! Globals: `model.embed_tokens.weight` (the unpadded vocabulary's rows),
//! `model.embed_tokens.norm.weight`, `model.norm.weight`,
//! `model.final_input.query` `[hidden]`, `model.message_input.query`
//! `[loop layers, 4, hidden]` (the q, k, v and MLP message queries of each
//! loop layer, in that order), `lm_head.weight` (unpadded rows).
//!
//! Per layer `model.layers.{i}.`:
//! * `attention_input.query` `[1 | 3, hidden]` (not on layer 0; three on the
//!   full-attention layers: q, k, v), `mlp_input.query` `[1, hidden]`;
//! * `input_layernorm.weight` (sliding layers) or `{q,k,v}_input_layernorm.weight`
//!   (full layers), `post_attention_layernorm.weight`;
//! * `self_attn.{q,k,v,o,gate,rel}_proj.weight`, `self_attn.sink_proj.weight`
//!   (f32), `self_attn.q_scale` `[heads, head_dim]`, `self_attn.rel_scale`
//!   `[heads, d_rel]`, `self_attn.rel_table` `[d_rel, extent]`;
//! * `self_attn.{k,v}_conv.taps` `[channels, kernel]` with tap 0 on the
//!   current position (the sliding layers' conv1d reversed, the full layers'
//!   `static_w` transposed) and, on full layers, `self_attn.{k,v}_conv.proj`
//!   `[kv_heads · kernel, hidden]` (f32): the per-token taps;
//! * `post_attention_conv.taps`, `post_mlp_conv.taps` `[hidden, kernel]`;
//! * `mlp.gate.weight` (f32 router), `mlp.gate.e_score_correction_bias` (f32),
//!   `mlp.experts.{e}.{gate,up,down}_proj.weight`;
//! * loop layers: `{q,k,v,mlp}_message_gate.{down,forget,gain}` (f32).
//!
//! The vision tower and the MTP drafter (`mtp.safetensors`) are left out.
//!
//! ## Profiles (`--quant`)
//!
//! * `q4tp` (default, `auto`): routed experts q4tp, every other matrix q8_2f.
//! * `q8_2f`: every matrix q8_2f.
//! * `nvfp4`: the NVFP4 experts byte for byte (`nvfp4` dtype), the MXFP8
//!   experts q8_2f, the rest q8_2f — the checkpoint's own numbers, for the
//!   reference on the CPU.
//!
//! The embedding is f16 in every profile (read a row a token on the host).

use crate::convert::{Quant, SafeTensors, TensorMeta, fp8_e4m3_to_f32, open_safetensors, quantize_2d, tensor_quant_override, to_f32};
use cortiq_core::format::{CmfHeader, CmfStreamWriter, TokenizerBundle};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use std::collections::HashMap;
use std::path::Path;

pub const ARCH_NAME: &str = "inkling2_loop";

/// Is `config` an Inkling 2 looped checkpoint?
pub fn detect(config: &serde_json::Value) -> bool {
    config.get("model_type").and_then(|v| v.as_str()) == Some(ARCH_NAME)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Profile {
    Q8,
    Q4,
    Nv,
}

fn parse_profile(s: &str) -> anyhow::Result<Profile> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "auto" | "q4tp" | "q4" => Profile::Q4,
        "q8_2f" | "q82f" | "q8" => Profile::Q8,
        "nvfp4" => Profile::Nv,
        other => anyhow::bail!("inkling2_loop: unknown --quant '{other}' (q4tp | q8_2f | nvfp4 | auto)"),
    })
}

fn profile_name(p: Profile) -> &'static str {
    match p {
        Profile::Q8 => "q8_2f",
        Profile::Q4 => "q4tp",
        Profile::Nv => "nvfp4",
    }
}

/// How one output matrix is stored under `profile` (canonical name).
fn matrix_codec(name: &str, profile: Profile) -> Quant {
    if let Some(q) = tensor_quant_override(name) {
        return q;
    }
    if name == "model.embed_tokens.weight" {
        return Quant::F16;
    }
    if name.contains(".mlp.experts.") && profile == Profile::Q4 {
        return Quant::Q4TiledP;
    }
    Quant::Q8_2f
}

/// Weights kept exact (f32) whatever the profile: small or precision-critical.
fn keep_f32(name: &str) -> bool {
    name.ends_with(".mlp.gate.weight")
        || name.ends_with(".sink_proj.weight")
        || name.contains("_message_gate.")
        || name.ends_with("_conv.proj")
}

struct Sources {
    files: Vec<SafeTensors>,
    by_name: HashMap<String, (usize, usize)>,
    used: std::collections::HashSet<String>,
}

impl Sources {
    fn open(dir: &Path) -> anyhow::Result<Self> {
        let mut files = Vec::new();
        let mut by_name = HashMap::new();
        let mut names: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            // the MTP drafter rides in its own file outside the index
            .filter(|p| p.file_name().is_some_and(|n| n != "mtp.safetensors"))
            .collect();
        names.sort();
        anyhow::ensure!(!names.is_empty(), "{}: no .safetensors shards", dir.display());
        for p in names {
            let st = open_safetensors(&p)?;
            let fi = files.len();
            for (ti, m) in st.tensors.iter().enumerate() {
                by_name.insert(m.name.clone(), (fi, ti));
            }
            files.push(st);
        }
        Ok(Self { files, by_name, used: Default::default() })
    }

    fn has(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    fn need(&mut self, name: &str) -> anyhow::Result<(&SafeTensors, &TensorMeta)> {
        self.used.insert(name.to_string());
        let &(f, t) = self.by_name.get(name).ok_or_else(|| anyhow::anyhow!("inkling2_loop: tensor {name} missing"))?;
        Ok((&self.files[f], &self.files[f].tensors[t]))
    }

    /// A float tensor in f32 with its shape.
    fn f32(&mut self, name: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let (f, m) = self.need(name)?;
        Ok((to_f32(&m.dtype, f.bytes(m))?, m.shape.clone()))
    }

    /// A 2-D weight in f32.
    fn matrix(&mut self, name: &str) -> anyhow::Result<(Vec<f32>, usize, usize)> {
        let (v, sh) = self.f32(name)?;
        anyhow::ensure!(sh.len() == 2, "{name}: shape {sh:?} is not a matrix");
        Ok((v, sh[0], sh[1]))
    }
}

struct Out {
    w: CmfStreamWriter,
    census: std::collections::BTreeMap<&'static str, (usize, u64, u64)>,
}

impl Out {
    fn push(&mut self, group: &'static str, name: &str, dt: TensorDtype, shape: &[usize], bytes: &[u8]) -> anyhow::Result<()> {
        self.w.push(name, dt, shape, bytes).map_err(|e| anyhow::anyhow!("write {name}: {e}"))?;
        let e = self.census.entry(group).or_default();
        e.0 += 1;
        e.1 += shape.iter().product::<usize>() as u64;
        e.2 += bytes.len() as u64;
        Ok(())
    }

    fn f32(&mut self, group: &'static str, name: &str, shape: &[usize], v: &[f32]) -> anyhow::Result<()> {
        anyhow::ensure!(shape.iter().product::<usize>() == v.len(), "{name}: {} values for shape {shape:?}", v.len());
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.push(group, name, TensorDtype::F32, shape, &b)
    }

    fn matrix(&mut self, group: &'static str, name: &str, q: Quant, v: &[f32], rows: usize, cols: usize) -> anyhow::Result<()> {
        if keep_f32(name) {
            return self.f32(group, name, &[rows, cols], v);
        }
        let (dt, bytes) = quantize_2d(q, v, rows, cols);
        self.push(group, name, dt, &[rows, cols], &bytes)
    }
}

fn read_json(dir: &Path, rel: &str) -> Option<serde_json::Value> {
    std::fs::read(dir.join(rel)).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

/// E2M1 code → value (low nibble first in a byte), as transformers' table.
const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// One expert's matrix of a layer, the source's bytes: NVFP4 (`codes` `[rows,
/// cols/2]`, `scales` E4M3 `[rows, cols/16]`, `factor`) or MXFP8 (`codes`
/// E4M3 `[rows, cols]`, `scales` E8M0 `[rows, cols/32]`).
#[derive(Clone, Copy)]
struct ExpertSrc<'a> {
    codes: &'a [u8],
    scales: &'a [u8],
    factor: Option<f32>,
    rows: usize,
    cols: usize,
}

impl ExpertSrc<'_> {
    /// Rows `[r0, r0 + n)` with stride `step` (2 for one half of `w13`) in f32.
    fn decode_rows(&self, first: usize, step: usize, n: usize) -> anyhow::Result<Vec<f32>> {
        let cols = self.cols;
        let mut out = vec![0f32; n * cols];
        for (i, row) in out.chunks_mut(cols).enumerate() {
            let r = first + i * step;
            match self.factor {
                Some(g) => {
                    let (cpr, spr) = (cols / 2, cols / 16);
                    for b in 0..spr {
                        let s = fp8_e4m3_to_f32(self.scales[r * spr + b]);
                        anyhow::ensure!(s.is_finite(), "nvfp4: NaN block scale at row {r} block {b}");
                        for j in 0..8 {
                            let byte = self.codes[r * cpr + b * 8 + j];
                            row[b * 16 + 2 * j] = (E2M1[(byte & 0x0F) as usize] * s) * g;
                            row[b * 16 + 2 * j + 1] = (E2M1[(byte >> 4) as usize] * s) * g;
                        }
                    }
                }
                None => {
                    let spr = cols / 32;
                    for b in 0..spr {
                        let s = (self.scales[r * spr + b] as f32 - 127.0).exp2();
                        for j in 0..32 {
                            let v = fp8_e4m3_to_f32(self.codes[r * cols + b * 32 + j]);
                            anyhow::ensure!(v.is_finite(), "mxfp8: NaN value at row {r} column {}", b * 32 + j);
                            row[b * 32 + j] = v * s;
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// The same rows as a CMF `nvfp4` payload (codes, UE4M3 scales, factor):
    /// the source's bytes, no rounding. NVFP4 sources only.
    fn nvfp4_rows(&self, first: usize, step: usize, n: usize) -> Vec<u8> {
        let g = self.factor.expect("nvfp4 source");
        let (cpr, spr) = (self.cols / 2, self.cols / 16);
        let mut b = Vec::with_capacity(n * (cpr + spr) + 4);
        for i in 0..n {
            let r = first + i * step;
            b.extend_from_slice(&self.codes[r * cpr..(r + 1) * cpr]);
        }
        for i in 0..n {
            let r = first + i * step;
            b.extend_from_slice(&self.scales[r * spr..(r + 1) * spr]);
        }
        b.extend_from_slice(&g.to_le_bytes());
        b
    }
}

/// A short convolution's taps as `[channels, kernel]`, tap 0 on the current
/// position. `conv1d` `[channels, 1, kernel]` multiplies its LAST tap with the
/// current position; `static_w` `[kernel, channels]` is already by lag.
fn conv_taps(v: &[f32], shape: &[usize], by_lag: bool) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    if by_lag {
        anyhow::ensure!(shape.len() == 2, "static taps shape {shape:?}");
        let (k, c) = (shape[0], shape[1]);
        let mut t = vec![0f32; c * k];
        for lag in 0..k {
            for ch in 0..c {
                t[ch * k + lag] = v[lag * c + ch];
            }
        }
        Ok((t, c, k))
    } else {
        anyhow::ensure!(shape.len() == 3 && shape[1] == 1, "conv1d shape {shape:?}");
        let (c, k) = (shape[0], shape[2]);
        let mut t = vec![0f32; c * k];
        for ch in 0..c {
            for lag in 0..k {
                t[ch * k + lag] = v[ch * k + (k - 1 - lag)];
            }
        }
        Ok((t, c, k))
    }
}

/// Convert the checkpoint in `dir` into `output` under the `quant` profile.
pub fn convert(dir: &Path, quant: &str, output: &str) -> anyhow::Result<()> {
    let profile = parse_profile(quant)?;
    let t0 = std::time::Instant::now();
    let config = read_json(dir, "config.json").ok_or_else(|| anyhow::anyhow!("{}: no readable config.json", dir.display()))?;
    anyhow::ensure!(detect(&config), "{}: not an Inkling 2 looped checkpoint", dir.display());
    let tc = config.get("text_config").cloned().ok_or_else(|| anyhow::anyhow!("inkling2_loop: config has no text_config"))?;
    let g = |k: &str| -> anyhow::Result<usize> {
        tc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| anyhow::anyhow!("inkling2_loop: text_config has no {k}"))
    };
    let gf = |k: &str| -> anyhow::Result<f64> {
        tc.get(k).and_then(|v| v.as_f64()).ok_or_else(|| anyhow::anyhow!("inkling2_loop: text_config has no {k}"))
    };
    let hidden = g("hidden_size")?;
    let n_layers = g("num_hidden_layers")?;
    let nh = g("num_attention_heads")?;
    let nkv = g("num_key_value_heads")?;
    let hd = g("head_dim")?;
    let n_exp = g("n_routed_experts")?;
    let top_k = g("num_experts_per_tok")?;
    let inter = g("intermediate_size")?;
    let d_rel = g("d_rel")?;
    let kernel = g("sconv_kernel_size")?;
    let block = g("n_layers_per_attn_res_block")?;
    let vocab_padded = g("vocab_size")?;
    let vocab = g("unpadded_vocab_size")?;
    let window = g("sliding_window_size")?;
    let rel_extent = g("rel_extent")?;
    anyhow::ensure!(g("swa_num_attention_heads")? == nh && g("swa_num_key_value_heads")? == nkv && g("swa_head_dim")? == hd, "inkling2_loop: sliding layers with their own head geometry are not supported");
    anyhow::ensure!(tc.get("n_shared_experts").and_then(|v| v.as_u64()).unwrap_or(0) == 0, "inkling2_loop: shared experts are not supported");
    anyhow::ensure!(tc.get("q_bias").and_then(|v| v.as_bool()) != Some(true) && tc.get("o_bias").and_then(|v| v.as_bool()) != Some(true), "inkling2_loop: attention biases are not supported");
    anyhow::ensure!(tc.get("use_value_norm_no_gamma").and_then(|v| v.as_bool()) != Some(false), "inkling2_loop: a weighted value norm is not supported");
    let local: Vec<usize> = tc.get("local_layer_ids").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect()).unwrap_or_default();
    let lc = tc.get("loop_config").cloned().unwrap_or(serde_json::Value::Null);
    let lg = |k: &str| -> anyhow::Result<usize> { lc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| anyhow::anyhow!("inkling2_loop: loop_config has no {k}")) };
    let (core_start, core_end, passes, gate_rank) = (lg("core_start")?, lg("core_end")?, lg("num_passes")?, lg("gate_rank")?);
    anyhow::ensure!(lc.get("is_varloop").and_then(|v| v.as_bool()) != Some(true), "inkling2_loop: a variable loop is not supported");
    anyhow::ensure!(core_start > 0 && core_start < core_end && core_end <= n_layers && core_start % block == 0 && core_end % block == 0 && passes >= 1, "inkling2_loop: loop core [{core_start}, {core_end}) x {passes} with blocks of {block}");
    let n_loop = core_end - core_start;
    let is_sliding = |li: usize| local.contains(&li);

    let mut src = Sources::open(dir)?;
    let tokenizer_config = read_json(dir, "tokenizer_config.json").unwrap_or(serde_json::Value::Null);
    let generation = read_json(dir, "generation_config.json").unwrap_or(serde_json::Value::Null);
    let vocab_bytes = std::fs::read(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{}: tokenizer.json: {e}", dir.display()))?;
    let chat_template = std::fs::read_to_string(dir.join("chat_template.jinja"))
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| tokenizer_config.get("chat_template").and_then(|v| v.as_str()).map(String::from));

    let n_out_estimate = n_layers * (3 * n_exp + 48) + 64;
    let gap = CmfStreamWriter::head_reserve_for(n_out_estimate, 64);
    let tmp = format!("{output}.part");
    let w = CmfStreamWriter::new(&tmp, gap).map_err(|e| anyhow::anyhow!("create {tmp}: {e}"))?;
    let mut out = Out { w, census: Default::default() };
    let l = "model.llm.";

    // ---- globals: the padded rows of the embedding and the head are dropped
    for (from, to) in [("embed.weight", "model.embed_tokens.weight"), ("unembed.weight", "lm_head.weight")] {
        let (v, rows, cols) = src.matrix(&format!("{l}{from}"))?;
        anyhow::ensure!(rows == vocab_padded && cols == hidden, "{from}: {rows}x{cols}, expected {vocab_padded}x{hidden}");
        out.matrix("skeleton", to, matrix_codec(to, profile), &v[..vocab * hidden], vocab, hidden)?;
    }
    for (from, to) in [("embed_norm.weight", "model.embed_tokens.norm.weight"), ("norm.weight", "model.norm.weight"), ("attn_res.output_query", "model.final_input.query")] {
        let (v, _) = src.f32(&format!("{l}{from}"))?;
        anyhow::ensure!(v.len() == hidden, "{from}: {} values, expected {hidden}", v.len());
        out.f32("skeleton", to, &[hidden], &v)?;
    }
    {
        // (2·loop, hidden) [query, mlp] pairs and (2·loop, hidden) [key, value] pairs
        // → (loop, 4, hidden) [query, key, value, mlp]
        let (q, qs) = src.f32(&format!("{l}middle_cycle_cross_pass_queries"))?;
        let (kv, ks) = src.f32(&format!("{l}middle_cycle_cross_pass_kv_queries"))?;
        anyhow::ensure!(qs == vec![2 * n_loop, hidden] && ks == vec![2 * n_loop, hidden], "message queries {qs:?} / {ks:?}");
        let mut m = vec![0f32; n_loop * 4 * hidden];
        for i in 0..n_loop {
            let row = |v: &[f32], r: usize| v[r * hidden..(r + 1) * hidden].to_vec();
            for (slot, vals) in [(0, row(&q, 2 * i)), (1, row(&kv, 2 * i)), (2, row(&kv, 2 * i + 1)), (3, row(&q, 2 * i + 1))] {
                m[(i * 4 + slot) * hidden..(i * 4 + slot + 1) * hidden].copy_from_slice(&vals);
            }
        }
        out.f32("skeleton", "model.message_input.query", &[n_loop, 4, hidden], &m)?;
    }

    for li in 0..n_layers {
        let s = format!("{l}layers.{li}.");
        let p = format!("model.layers.{li}.");
        let sliding = is_sliding(li);
        // ---- block attention residual queries: attn_queries.{i} belongs to layer i + 1
        if li > 0 {
            let (v, sh) = src.f32(&format!("{l}attn_res.attn_queries.{}", li - 1))?;
            let nq = if sliding { 1 } else { 3 };
            anyhow::ensure!(v.len() == nq * hidden, "attn_queries.{}: shape {sh:?}, expected {nq} x {hidden}", li - 1);
            out.f32("skeleton", &format!("{p}attention_input.query"), &[nq, hidden], &v)?;
        }
        let (v, _) = src.f32(&format!("{l}attn_res.mlp_queries.{li}"))?;
        anyhow::ensure!(v.len() == hidden, "mlp_queries.{li}: {} values", v.len());
        out.f32("skeleton", &format!("{p}mlp_input.query"), &[1, hidden], &v)?;

        // ---- norms
        let norms: &[(&str, &str)] = if sliding {
            &[("attn_norm.weight", "input_layernorm.weight"), ("mlp_norm.weight", "post_attention_layernorm.weight")]
        } else {
            &[
                ("q_input_norm.weight", "q_input_layernorm.weight"),
                ("k_input_norm.weight", "k_input_layernorm.weight"),
                ("v_input_norm.weight", "v_input_layernorm.weight"),
                ("mlp_norm.weight", "post_attention_layernorm.weight"),
            ]
        };
        for (from, to) in norms {
            let (v, _) = src.f32(&format!("{s}{from}"))?;
            anyhow::ensure!(v.len() == hidden, "{s}{from}: {} values", v.len());
            out.f32("skeleton", &format!("{p}{to}"), &[hidden], &v)?;
        }

        // ---- attention
        let a = format!("{s}attn.");
        let o = format!("{p}self_attn.");
        for (from, to, rows, cols) in [
            ("wq_du", "q_proj", nh * hd, hidden),
            ("wk_dv", "k_proj", nkv * hd, hidden),
            ("wv_dv", "v_proj", nkv * hd, hidden),
            ("wo_ud", "o_proj", hidden, nh * hd),
            ("gate_attn", "gate_proj", nh * hd, hidden),
            ("wr_du", "rel_proj", nh * d_rel, hidden),
            ("attn_sink_proj", "sink_proj", nh, hidden),
        ] {
            let (v, r, c) = src.matrix(&format!("{a}{from}.weight"))?;
            anyhow::ensure!(r == rows && c == cols, "{a}{from}: {r}x{c}, expected {rows}x{cols}");
            let n = format!("{o}{to}.weight");
            out.matrix("skeleton", &n, matrix_codec(&n, profile), &v, r, c)?;
        }
        let (v, sh) = src.f32(&format!("{a}q_l2_clip.scale"))?;
        anyhow::ensure!(sh == vec![nh, hd], "q_l2_clip.scale {sh:?}");
        out.f32("skeleton", &format!("{o}q_scale"), &sh, &v)?;
        let (v, sh) = src.f32(&format!("{a}rel_l2_clip.scale"))?;
        anyhow::ensure!(sh == vec![nh, d_rel], "rel_l2_clip.scale {sh:?}");
        out.f32("skeleton", &format!("{o}rel_scale"), &sh, &v)?;
        let (v, sh) = src.f32(&format!("{a}rel_logits_proj.proj"))?;
        let extent = if sliding { window } else { rel_extent };
        anyhow::ensure!(sh == vec![d_rel, extent], "rel_logits_proj.proj {sh:?}, expected [{d_rel}, {extent}]");
        out.f32("skeleton", &format!("{o}rel_table"), &sh, &v)?;
        for kv in ["k", "v"] {
            if sliding {
                let (v, sh) = src.f32(&format!("{a}{kv}_sconv.weight"))?;
                let (t, c, k) = conv_taps(&v, &sh, false)?;
                anyhow::ensure!(c == nkv * hd && k == kernel, "{kv}_sconv {sh:?}");
                out.f32("skeleton", &format!("{o}{kv}_conv.taps"), &[c, k], &t)?;
            } else {
                let (v, sh) = src.f32(&format!("{a}{kv}_sconv.static_w"))?;
                let (t, c, k) = conv_taps(&v, &sh, true)?;
                anyhow::ensure!(c == nkv * hd && k == kernel, "{kv}_sconv.static_w {sh:?}");
                out.f32("skeleton", &format!("{o}{kv}_conv.taps"), &[c, k], &t)?;
                let (v, r, c) = src.matrix(&format!("{a}{kv}_sconv.weight_proj"))?;
                anyhow::ensure!(r == nkv * kernel && c == hidden, "{kv}_sconv.weight_proj {r}x{c}");
                out.f32("skeleton", &format!("{o}{kv}_conv.proj"), &[r, c], &v)?;
            }
        }
        for (from, to) in [("attn_sconv.weight", "post_attention_conv.taps"), ("mlp_sconv.weight", "post_mlp_conv.taps")] {
            let (v, sh) = src.f32(&format!("{s}{from}"))?;
            let (t, c, k) = conv_taps(&v, &sh, false)?;
            anyhow::ensure!(c == hidden && k == kernel, "{from} {sh:?}");
            out.f32("skeleton", &format!("{p}{to}"), &[c, k], &t)?;
        }

        // ---- loop controllers (message gates)
        if (core_start..core_end).contains(&li) {
            for (from, to) in [("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "mlp")] {
                for (wf, wt, rows, cols) in [("down_weight", "down", gate_rank, 2 * hidden), ("forget_weight", "forget", hidden, gate_rank), ("gain_weight", "gain", hidden, gate_rank)] {
                    let (v, r, c) = src.matrix(&format!("{s}controllers.{from}.{wf}"))?;
                    anyhow::ensure!(r == rows && c == cols, "controllers.{from}.{wf}: {r}x{c}");
                    out.f32("skeleton", &format!("{p}{to}_message_gate.{wt}"), &[r, c], &v)?;
                }
            }
        }

        // ---- MoE: router, selection bias, experts
        let m = format!("{p}mlp.");
        let (v, r, c) = src.matrix(&format!("{s}mlp.gate.weight"))?;
        anyhow::ensure!(r == n_exp && c == hidden, "{s}mlp.gate: {r}x{c}");
        out.f32("skeleton", &format!("{m}gate.weight"), &[r, c], &v)?;
        let (bias, _) = {
            let (f, t) = src.need(&format!("{s}mlp.gate.bias"))?;
            // the spread between experts is below a bf16 step: the router needs it in f32
            anyhow::ensure!(t.dtype == "F32", "{s}mlp.gate.bias is {}", t.dtype);
            (to_f32(&t.dtype, f.bytes(t))?, ())
        };
        anyhow::ensure!(bias.len() == n_exp, "{s}mlp.gate.bias has {} values", bias.len());
        out.f32("skeleton", &format!("{m}gate.e_score_correction_bias"), &[n_exp], &bias)?;
        pack_experts(&mut src, &mut out, &s, &m, profile, n_exp, hidden, inter)?;
        eprintln!("  layer {li} ({}) done, {:.0}s", if sliding { "sliding" } else { "full" }, t0.elapsed().as_secs_f64());
    }

    // every source tensor is either packed or deliberately left out (the vision tower)
    let mut left: Vec<&String> = src.by_name.keys().filter(|k| !src.used.contains(*k) && !k.starts_with("model.visual.")).collect();
    left.sort();
    anyhow::ensure!(left.is_empty(), "inkling2_loop: source tensors left unpacked: {left:?}");
    for (grp, (n, wts, b)) in &out.census {
        eprintln!("  {grp}: {n} tensors, {:.2} B weights → {:.2} GB", *wts as f64 / 1e9, *b as f64 / 1e9);
    }

    let mut eos: Vec<u32> = match generation.get("eos_token_id") {
        Some(serde_json::Value::Array(a)) => a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect(),
        Some(v) => v.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
        None => Vec::new(),
    };
    if eos.is_empty() {
        eos.push(config.get("eos_token_id").and_then(|v| v.as_u64()).unwrap_or(200006) as u32);
    }
    let layer_types: Vec<&str> = (0..n_layers).map(|li| if is_sliding(li) { "SlidingAttention" } else { "FullAttention" }).collect();
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": ARCH_NAME,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_layers": n_layers,
        "num_attention_heads": nh,
        "num_kv_heads": nkv,
        "head_dim": hd,
        "vocab_size": vocab,
        "layer_types": layer_types,
        "rms_norm_eps": gf("rms_norm_eps")?,
        "max_position_embeddings": g("model_max_length").unwrap_or(1 << 20),
        "linear_conv_kernel_dim": null,
        "linear_num_key_heads": null,
        "linear_num_value_heads": null,
        "moe": {
            "num_experts": n_exp,
            "top_k": top_k,
            "moe_intermediate_size": inter,
            "norm_topk_prob": true,
            "router_sigmoid": true,
            "routed_scaling_factor": gf("route_scale")?,
        },
    }))?;
    let mut text = tc.clone();
    if let Some(o) = text.as_object_mut() {
        for k in ["torch_dtype", "transformers_version"] {
            o.remove(k);
        }
    }
    let provenance = serde_json::json!({
        "source": dir.display().to_string(),
        "base_model": "thinkingmachines/Inkling-2-Mini",
        "quant": profile_name(profile),
        "license": "apache-2.0",
        // generation_config.json: temperature 1.0, sampled; `--sampling` names others
        "sampling": crate::convert::sampling_defaults().unwrap_or_else(|| serde_json::json!({ "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0 })),
        "inkling2_loop": { "config": text, "image_token_id": config.get("image_token_id").cloned().unwrap_or(serde_json::json!(200054)) },
    });
    let bundle = TokenizerBundle {
        chat_template,
        eos_token_ids: eos,
        bos_token_id: None,
        pad_token_id: generation.get("pad_token_id").and_then(|v| v.as_u64()).map(|n| n as u32),
    };
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch,
        quant_type: match profile {
            Profile::Q8 | Profile::Nv => QuantType::Q8_2f,
            Profile::Q4 => QuantType::Q4Block,
        },
        provenance: Some(provenance),
        tokenizer_config: Some(bundle),
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
        genome: None,
        lineage: Vec::new(),
        router: None,
        segments: Vec::new(),
    };
    let count = out.w.tensor_count();
    out.w.finish(&header, None, Some(&vocab_bytes)).map_err(|e| anyhow::anyhow!("finish {tmp}: {e}"))?;
    std::fs::rename(&tmp, output)?;
    let size = std::fs::metadata(output)?.len();
    eprintln!("{output}: {count} tensors, {:.2} GB ({}) in {:.0}s", size as f64 / 1e9, profile_name(profile), t0.elapsed().as_secs_f64());
    Ok(())
}

/// A layer's routed experts: decoded from NVFP4 / MXFP8 and encoded a batch
/// at a time on all cores (or, under `nvfp4`, the NVFP4 bytes as they are).
#[allow(clippy::too_many_arguments)]
fn pack_experts(src: &mut Sources, out: &mut Out, s: &str, m: &str, profile: Profile, n_exp: usize, hidden: usize, inter: usize) -> anyhow::Result<()> {
    let e = format!("{s}mlp.experts.");
    let nv = src.has(&format!("{e}w13_weight_amax"));
    let mut factors: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    if nv {
        for (i, w) in ["w13", "w2"].iter().enumerate() {
            let (amax, _) = src.f32(&format!("{e}{w}_weight_amax"))?;
            let (den, _) = src.f32(&format!("{e}{w}_weight_outer_scale_denominator"))?;
            anyhow::ensure!(amax.len() == n_exp && den.len() == 1, "{e}{w}: amax {} / denominator {}", amax.len(), den.len());
            factors[i] = amax.iter().map(|a| a / den[0]).collect();
        }
    }
    let (w13_codes, w13_scales, w2_codes, w2_scales) = {
        let mut take = |n: &str, dt: &str, shape: Vec<usize>| -> anyhow::Result<Vec<u8>> {
            let (f, t) = src.need(n)?;
            anyhow::ensure!(t.dtype == dt && t.shape == shape, "{n}: {} {:?}, expected {dt} {shape:?}", t.dtype, t.shape);
            Ok(f.bytes(t).to_vec())
        };
        if nv {
            (
                take(&format!("{e}w13_weight"), "U8", vec![n_exp, 2 * inter, hidden / 2])?,
                take(&format!("{e}w13_weight_scale"), "F8_E4M3", vec![n_exp, 2 * inter, hidden / 16])?,
                take(&format!("{e}w2_weight"), "U8", vec![n_exp, hidden, inter / 2])?,
                take(&format!("{e}w2_weight_scale"), "F8_E4M3", vec![n_exp, hidden, inter / 16])?,
            )
        } else {
            (
                take(&format!("{e}w13_weight"), "F8_E4M3", vec![n_exp, 2 * inter, hidden])?,
                take(&format!("{e}w13_weight_scale"), "U8", vec![n_exp, 2 * inter, hidden / 32])?,
                take(&format!("{e}w2_weight"), "F8_E4M3", vec![n_exp, hidden, inter])?,
                take(&format!("{e}w2_weight_scale"), "U8", vec![n_exp, hidden, inter / 32])?,
            )
        }
    };
    let expert = |x: usize, w2: bool| -> ExpertSrc<'_> {
        let (codes, scales, rows, cols) = if w2 { (&w2_codes, &w2_scales, hidden, inter) } else { (&w13_codes, &w13_scales, 2 * inter, hidden) };
        let (cb, sb) = if nv { (rows * cols / 2, rows * cols / 16) } else { (rows * cols, rows * cols / 32) };
        ExpertSrc {
            codes: &codes[x * cb..(x + 1) * cb],
            scales: &scales[x * sb..(x + 1) * sb],
            factor: nv.then(|| factors[w2 as usize][x]),
            rows,
            cols,
        }
    };
    // (name, source, first row, row step, rows)
    let mats = |x: usize| -> [(String, ExpertSrc<'_>, usize, usize, usize); 3] {
        [
            (format!("{m}experts.{x}.gate_proj.weight"), expert(x, false), 0, 2, inter),
            (format!("{m}experts.{x}.up_proj.weight"), expert(x, false), 1, 2, inter),
            (format!("{m}experts.{x}.down_proj.weight"), expert(x, true), 0, 1, hidden),
        ]
    };
    if profile == Profile::Nv && nv {
        for x in 0..n_exp {
            for (n, es, first, step, rows) in mats(x) {
                let b = es.nvfp4_rows(first, step, rows);
                out.push("experts", &n, TensorDtype::Nvfp4, &[rows, es.cols], &b)?;
            }
        }
        return Ok(());
    }
    let threads = crate::convert::encode_threads().max(1);
    let all: Vec<_> = (0..n_exp).flat_map(mats).collect();
    for batch in all.chunks(threads) {
        let enc: Vec<anyhow::Result<(TensorDtype, Vec<u8>)>> = std::thread::scope(|sc| {
            let hs: Vec<_> = batch
                .iter()
                .map(|(n, es, first, step, rows)| {
                    let q = matrix_codec(n, profile);
                    sc.spawn(move || -> anyhow::Result<(TensorDtype, Vec<u8>)> {
                        let v = es.decode_rows(*first, *step, *rows)?;
                        Ok(quantize_2d(q, &v, *rows, es.cols))
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("encoder thread")).collect()
        });
        for ((n, es, _, _, rows), r) in batch.iter().zip(enc) {
            let (dt, bytes) = r?;
            out.push("experts", n, dt, &[*rows, es.cols], &bytes)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_codecs() {
        let gu = "model.layers.3.mlp.experts.7.gate_proj.weight";
        assert_eq!(matrix_codec(gu, Profile::Q4), Quant::Q4TiledP);
        assert_eq!(matrix_codec(gu, Profile::Q8), Quant::Q8_2f);
        assert_eq!(matrix_codec("model.layers.3.self_attn.o_proj.weight", Profile::Q4), Quant::Q8_2f);
        assert_eq!(matrix_codec("lm_head.weight", Profile::Q4), Quant::Q8_2f);
        assert_eq!(matrix_codec("model.embed_tokens.weight", Profile::Q4), Quant::F16);
        assert!(keep_f32("model.layers.3.mlp.gate.weight"));
        assert!(keep_f32("model.layers.3.self_attn.sink_proj.weight"));
        assert!(keep_f32("model.layers.7.k_message_gate.down"));
        assert!(keep_f32("model.layers.5.self_attn.k_conv.proj"));
        assert!(!keep_f32("model.layers.3.self_attn.q_proj.weight"));
    }

    #[test]
    fn detects_the_model_type() {
        assert!(detect(&serde_json::json!({"model_type": "inkling2_loop"})));
        assert!(!detect(&serde_json::json!({"model_type": "inkling2"})));
    }

    #[test]
    fn conv1d_taps_are_reversed_static_taps_transposed() {
        // conv1d [2 channels, 1, 3 taps]: the last tap multiplies the current position
        let (t, c, k) = conv_taps(&[1., 2., 3., 4., 5., 6.], &[2, 1, 3], false).unwrap();
        assert_eq!((c, k), (2, 3));
        assert_eq!(t, vec![3., 2., 1., 6., 5., 4.]);
        // static_w [3 lags, 2 channels]
        let (t, c, k) = conv_taps(&[1., 2., 3., 4., 5., 6.], &[3, 2], true).unwrap();
        assert_eq!((c, k), (2, 3));
        assert_eq!(t, vec![1., 3., 5., 2., 4., 6.]);
    }

    #[test]
    fn nvfp4_decode_matches_the_reference_order() {
        // one row of 16: codes 0x21 → (0.5, 1.0) …, scale E4M3 0x38 = 1.0, factor 0.5
        let codes = [0x21u8, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F];
        let scales = [0x38u8];
        let e = ExpertSrc { codes: &codes, scales: &scales, factor: Some(0.5), rows: 1, cols: 16 };
        let v = e.decode_rows(0, 1, 1).unwrap();
        let want: Vec<f32> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0].iter().map(|&c| E2M1[c] * 0.5).collect();
        assert_eq!(v, want);
        let b = e.nvfp4_rows(0, 1, 1);
        assert_eq!(&b[..8], &codes);
        assert_eq!(b[8], 0x38);
        assert_eq!(f32::from_le_bytes(b[9..13].try_into().unwrap()), 0.5);
    }

    #[test]
    fn mxfp8_decode_scales_by_a_power_of_two() {
        let mut codes = [0u8; 32];
        codes[0] = 0x38; // 1.0
        codes[1] = 0xC0; // -2.0
        let scales = [128u8]; // 2^1
        let e = ExpertSrc { codes: &codes, scales: &scales, factor: None, rows: 1, cols: 32 };
        let v = e.decode_rows(0, 1, 1).unwrap();
        assert_eq!(v[0], 2.0);
        assert_eq!(v[1], -4.0);
        assert!(v[2..].iter().all(|&x| x == 0.0));
    }
}
