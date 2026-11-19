//! Pack tencent/Hy4-flash (`model_type: hy_v4_flash`) into one `.cmf`.
//!
//! `cortiq convert --model <dir with config.json> --output x.cmf --quant P`
//! detects the model type and comes here; the generic LLM walk is never
//! touched. The runtime is `cortiq_engine::hy4`.
//!
//! The source is the released FP8 checkpoint (`tencent/Hy4-flash-FP8`: the
//! routed experts and most attention projections in e4m3 with one scale per
//! 128x128 block, everything else bf16) or the bf16 one. `CMF_HY4_BF16=<dir>`
//! names a directory holding some bf16 shards of `tencent/Hy4-flash`: every
//! tensor found there wins over the same tensor in the main source, so the
//! attention can come from bf16 while the 150B of experts come from FP8.
//!
//! ## Layout
//!
//! Per layer `model.layers.{i}.`:
//! * `attn_hc.{fn,base,scale}`, `ffn_hc.{fn,base,scale}` — the identity
//!   hyper-connections (f32, `[2·hc, hc·hidden]`, `[2·hc]`, `[2]`);
//! * `input_layernorm.weight`, `post_attention_layernorm.weight`;
//! * `self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,
//!   kv_a_layernorm,kv_b_proj,gate_proj,o_proj,post_proj_norm}.weight`,
//!   `self_attn.sinks` — gated MLA in the engine's MLA layout: every
//!   `q_b_proj` head `[rope | nope]`, `kv_a_proj_with_mqa` `[latent | rope]`,
//!   the rotary rows permuted from the interleaved pairs the model was
//!   trained with to the half-split pairs the engine rotates;
//! * sparse layers: `self_attn.indexer.{query_projection,
//!   head_weight_projection,token_key_projection,token_key_norm,
//!   key_value_projection,key_gate_projection,key_norm}.weight`,
//!   `self_attn.indexer.intra_group_ape` — the compressed indexer, its rotary
//!   channels permuted the same way;
//! * layer 0: `mlp.{gate,up,down}_proj.weight`, `mlp.post_fc2_norm.weight`;
//! * MoE layers: `mlp.gate.weight` (f32 router), `mlp.gate.e_score_correction_bias`,
//!   `mlp.latent_down.weight` `[latent, hidden]`, `mlp.latent_up.weight`
//!   `[hidden, latent]`, `mlp.experts.{e}.{gate,up,down}_proj.weight`
//!   (`[inter, latent]`, `[inter, latent]`, `[latent, inter]`),
//!   `mlp.shared_experts.{gate,up,down}_proj.weight`, `mlp.combine_output_norm.weight`;
//! * Engram layers (1, 11, 21, 31): `engram.table` U8 `[rows, head_dim]` —
//!   each row e4m3 with its own scale `engram.table_scale` (bf16 `[rows]`):
//!   the per-vector fp8 round trip the model was trained to read, done once
//!   here, so the stored rows ARE the values the reference computes with;
//!   `engram.lookup` U32 (the normalized-vocabulary map),
//!   `engram.head_norms` `[heads, head_dim]`, `engram.key_projs.{j}.{weight,bias}`,
//!   `engram.value_proj.{weight,bias}`, `engram.norm_key.{j}.weight`,
//!   `engram.norm_query.{j}.weight`, `engram.pre_gamma`, `engram.hc_scales`
//!   `[hc, hidden]`.
//!
//! Globals: `model.embed_tokens.weight`, `lm_head.weight`, `model.norm.weight`,
//! `model.hc_head.{fn,base,scale}`, and the vision tower `vit.*` verbatim.
//! The DFly2 drafter is left out.
//!
//! ## Profiles (`--quant`)
//!
//! * `q4tp` (default, `auto`): routed experts q4tp; every other matrix q8_2f.
//! * `q2tp`: routed experts' gate/up q2tp (v2 encoder), down q4tp; every other
//!   matrix q4tp but the embedding and the head (q8_2f) — the cards of 12-16 GB.
//! * `q8_2f`: every matrix q8_2f.
//!
//! Norms, routers, hyper-connection weights, the indexer's pooling matrices
//! and other small tensors stay f32. `--tensor-quant PATTERN=QUANT` overrides
//! any matrix. The hyper-parameters the runtime needs (hash multipliers
//! included) ride in the header provenance under `hy4_flash`.

use crate::convert::{
    Quant, fp8_e4m3_to_f32, open_safetensors, quantize_2d, tensor_quant_override, to_f32,
    unpack_fp8_scale_inv, SafeTensors, TensorMeta,
};
use cortiq_core::format::{CmfHeader, CmfStreamWriter, TokenizerBundle};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use std::collections::HashMap;
use std::path::Path;

pub const ARCH_NAME: &str = "hy_v4_flash";

/// Is `config` a Hy4-flash checkpoint?
pub fn detect(config: &serde_json::Value) -> bool {
    config.get("model_type").and_then(|v| v.as_str()) == Some(ARCH_NAME)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Profile {
    Q8,
    Q4,
    Q2,
}

fn parse_profile(s: &str) -> anyhow::Result<Profile> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "q8_2f" | "q82f" | "q8" => Profile::Q8,
        "auto" | "q4tp" | "q4" => Profile::Q4,
        "q2tp" | "q2" => Profile::Q2,
        other => anyhow::bail!("hy_v4_flash: unknown --quant '{other}' (q4tp | q2tp | q8_2f | auto)"),
    })
}

fn profile_name(p: Profile) -> &'static str {
    match p {
        Profile::Q8 => "q8_2f",
        Profile::Q4 => "q4tp",
        Profile::Q2 => "q2tp",
    }
}

/// How one output matrix is stored under `profile` (canonical name).
fn matrix_codec(name: &str, profile: Profile) -> Quant {
    if let Some(q) = tensor_quant_override(name) {
        return q;
    }
    if name.contains(".mlp.experts.") {
        return match profile {
            Profile::Q8 => Quant::Q8_2f,
            Profile::Q4 => Quant::Q4TiledP,
            Profile::Q2 if name.ends_with(".down_proj.weight") => Quant::Q4TiledP,
            Profile::Q2 => Quant::Q2TiledP,
        };
    }
    // the small profile (cards of 12-16 GB): the always-active matrices q4tp
    // too, the vocabulary edges q8_2f
    if profile == Profile::Q2 && !(name == "model.embed_tokens.weight" || name == "lm_head.weight") {
        return Quant::Q4TiledP;
    }
    Quant::Q8_2f
}

/// Weights kept exact (f32) whatever the profile: small or precision-critical.
fn keep_f32(name: &str) -> bool {
    name.ends_with(".mlp.gate.weight")
        || name.contains("_hc.")
        || name.contains("hc_head.")
        || name.ends_with(".head_weight_projection.weight")
        || name.ends_with(".token_key_projection.weight")
        || name.ends_with(".key_value_projection.weight")
        || name.ends_with(".key_gate_projection.weight")
}

/// The interleaved-pair → half-split row order of one `rope`-wide rotary
/// slice: half-split row `r` takes interleaved row `src(r)`.
fn rope_src(r: usize, rope: usize) -> usize {
    if r < rope / 2 { 2 * r } else { 2 * (r - rope / 2) + 1 }
}

/// Permute the rotary rows `[off, off + rope)` of every `stride`-row block.
fn permute_rope_rows(vals: &mut [f32], cols: usize, stride: usize, off: usize, rope: usize) {
    let rows = vals.len() / cols;
    let mut tmp = vec![0f32; rope * cols];
    let mut b = 0;
    while b + stride <= rows {
        let base = (b + off) * cols;
        tmp.copy_from_slice(&vals[base..base + rope * cols]);
        for r in 0..rope {
            let s = rope_src(r, rope);
            vals[base + r * cols..base + (r + 1) * cols].copy_from_slice(&tmp[s * cols..(s + 1) * cols]);
        }
        b += stride;
    }
}

/// The engine's MLA query layout: every head `[rope | nope]` with the rotary
/// rows half-split (the source heads are `[nope | rope]`, rope interleaved).
fn rope_first_heads(vals: &[f32], cols: usize, nope: usize, rope: usize) -> Vec<f32> {
    let hd = nope + rope;
    let nh = vals.len() / cols / hd;
    let mut w = vec![0f32; vals.len()];
    for h in 0..nh {
        for r in 0..rope {
            let s = h * hd + nope + rope_src(r, rope);
            w[(h * hd + r) * cols..(h * hd + r + 1) * cols].copy_from_slice(&vals[s * cols..(s + 1) * cols]);
        }
        for r in 0..nope {
            let s = h * hd + r;
            w[(h * hd + rope + r) * cols..(h * hd + rope + r + 1) * cols].copy_from_slice(&vals[s * cols..(s + 1) * cols]);
        }
    }
    w
}

/// f32 → bf16 bits, round to nearest even.
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) as u16) | 0x40;
    }
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_round(x: f32) -> f32 {
    f32::from_bits((bf16_bits(x) as u32) << 16)
}

/// The 126 positive finite e4m3 values, by code 1..=126.
fn e4m3_table() -> &'static [f32; 127] {
    static T: std::sync::OnceLock<[f32; 127]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 127];
        for (c, v) in t.iter_mut().enumerate() {
            *v = fp8_e4m3_to_f32(c as u8);
        }
        t
    })
}

/// f32 → e4m3fn, round to nearest even, saturating at ±448 (torch's cast
/// after the reference's clamp).
fn e4m3_encode(x: f32) -> u8 {
    let t = e4m3_table();
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs().min(448.0);
    // codes are monotonic in value: find the first code whose value >= a
    let hi = t.partition_point(|&v| v < a);
    let code = if hi == 0 {
        0
    } else if hi >= 127 {
        126
    } else {
        let (lo, hv, lv) = (hi - 1, t[hi], t[hi - 1]);
        let (dl, dh) = (a - lv, hv - a);
        if dl < dh {
            lo
        } else if dh < dl {
            hi
        } else if lo % 2 == 0 {
            lo
        } else {
            hi
        }
    };
    if code == 0 { 0 } else { sign | code as u8 }
}

/// One Engram row through the reference's fp8 round trip (bf16 arithmetic):
/// `s = bf16(bf16(amax) / 448)`, codes `e4m3(bf16(x / s))`. Returns the codes
/// and `s`; the value of code `c` is `bf16(e4m3(c) · s)`.
fn engram_row(x: &[f32], codes: &mut [u8]) -> f32 {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let s = bf16_round(bf16_round(amax.max(1e-12)) / 448.0);
    for (c, &v) in codes.iter_mut().zip(x) {
        *c = e4m3_encode(bf16_round(v / s).clamp(-448.0, 448.0));
    }
    s
}

/// The tensor sources: one map from name to (file, meta index), the bf16
/// override shards winning.
pub(crate) struct Sources {
    files: Vec<SafeTensors>,
    by_name: HashMap<String, (usize, usize)>,
    used: std::sync::Mutex<std::collections::HashSet<String>>,
}

impl Sources {
    pub(crate) fn open(dirs: &[&Path]) -> anyhow::Result<Self> {
        let mut files = Vec::new();
        let mut by_name = HashMap::new();
        for d in dirs {
            let mut names: Vec<_> = std::fs::read_dir(d)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                .collect();
            names.sort();
            for p in names {
                let st = open_safetensors(&p)?;
                let fi = files.len();
                for (ti, m) in st.tensors.iter().enumerate() {
                    by_name.insert(m.name.clone(), (fi, ti));
                }
                files.push(st);
            }
        }
        Ok(Self { files, by_name, used: Default::default() })
    }

    fn meta(&self, name: &str) -> Option<(&SafeTensors, &TensorMeta)> {
        self.by_name.get(name).map(|&(f, t)| (&self.files[f], &self.files[f].tensors[t]))
    }

    fn need(&self, name: &str) -> anyhow::Result<(&SafeTensors, &TensorMeta)> {
        self.used.lock().unwrap().insert(name.to_string());
        self.meta(name).ok_or_else(|| anyhow::anyhow!("hy_v4_flash: tensor {name} missing"))
    }

    /// A 2-D weight in f32: bf16/f32 widened, or e4m3 times its block scales.
    pub(crate) fn matrix(&self, name: &str, block: usize) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let (f, m) = self.need(name)?;
        if m.dtype == "F8_E4M3" {
            anyhow::ensure!(m.shape.len() == 2, "{name}: FP8 tensor {:?} is not 2-D", m.shape);
            let sname = format!("{name}_scale_inv");
            let (sf, sm) = self.need(&sname)?;
            let scales = to_f32(&sm.dtype, sf.bytes(sm))?;
            let v = unpack_fp8_scale_inv(f.bytes(m), &scales, m.shape[0], m.shape[1], block)?;
            return Ok((v, m.shape.clone()));
        }
        Ok((to_f32(&m.dtype, f.bytes(m))?, m.shape.clone()))
    }

    /// Expert `e` of a 3-D bank `[E, rows, cols]` in f32.
    pub(crate) fn bank_slice(&self, name: &str, e: usize, block: usize) -> anyhow::Result<(Vec<f32>, usize, usize)> {
        let (f, m) = self.need(name)?;
        anyhow::ensure!(m.shape.len() == 3, "{name}: bank {:?} is not 3-D", m.shape);
        let (rows, cols) = (m.shape[1], m.shape[2]);
        let raw = f.bytes(m);
        if m.dtype == "F8_E4M3" {
            let per = rows * cols;
            let (sf, sm) = self.need(&format!("{name}_scale_inv"))?;
            anyhow::ensure!(sm.shape.len() == 3 && sm.shape[0] == m.shape[0], "{name}: scale shape {:?}", sm.shape);
            let sper = sm.shape[1] * sm.shape[2];
            let elem = match sm.dtype.as_str() {
                "BF16" | "F16" => 2,
                "F32" => 4,
                d => anyhow::bail!("{name}: scale dtype {d}"),
            };
            let sraw = &sf.bytes(sm)[e * sper * elem..(e + 1) * sper * elem];
            let scales = to_f32(&sm.dtype, sraw)?;
            let v = unpack_fp8_scale_inv(&raw[e * per..(e + 1) * per], &scales, rows, cols, block)?;
            return Ok((v, rows, cols));
        }
        let elem = match m.dtype.as_str() {
            "BF16" | "F16" => 2,
            "F32" => 4,
            d => anyhow::bail!("{name}: bank dtype {d}"),
        };
        let per = rows * cols * elem;
        Ok((to_f32(&m.dtype, &raw[e * per..(e + 1) * per])?, rows, cols))
    }
}

struct Out {
    w: CmfStreamWriter,
    census: std::collections::BTreeMap<&'static str, (usize, u64, u64)>,
    t0: std::time::Instant,
    n: usize,
}

impl Out {
    fn push(&mut self, group: &'static str, name: &str, dt: TensorDtype, shape: &[usize], bytes: &[u8]) -> anyhow::Result<()> {
        self.w.push(name, dt, shape, bytes).map_err(|e| anyhow::anyhow!("write {name}: {e}"))?;
        let e = self.census.entry(group).or_default();
        e.0 += 1;
        e.1 += shape.iter().product::<usize>() as u64;
        e.2 += bytes.len() as u64;
        self.n += 1;
        if self.n % 500 == 0 {
            eprintln!("  {} tensors, {:.0}s", self.n, self.t0.elapsed().as_secs_f64());
        }
        Ok(())
    }

    fn f32(&mut self, group: &'static str, name: &str, shape: &[usize], v: &[f32]) -> anyhow::Result<()> {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.push(group, name, TensorDtype::F32, shape, &b)
    }

    fn matrix(&mut self, group: &'static str, name: &str, q: Quant, v: &[f32], rows: usize, cols: usize) -> anyhow::Result<()> {
        if keep_f32(name) || matches!(q, Quant::F16) {
            return self.f32(group, name, &[rows, cols], v);
        }
        let (dt, bytes) = quantize_2d(q, v, rows, cols);
        self.push(group, name, dt, &[rows, cols], &bytes)
    }
}

fn read_json(dir: &Path, rel: &str) -> Option<serde_json::Value> {
    std::fs::read(dir.join(rel)).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

/// The Engram hash multipliers of the release: NumPy's
/// `default_rng(engram_seed + 10007 · layer_id).integers(0, ⌊i64::MAX / V⌋ / 2,
/// size=max_ngram) · 2 + 1` with `V` the compressed vocabulary (the lookup
/// map's max + 1), as `HYV4FlashEngramLayer.draw_hash_multipliers` draws
/// them. Drawn once (`numpy` 2.x PCG64) for the released seed 0 and V 92620.
fn release_multipliers(layer_id: usize, seed: u64, vocab: usize, max_ngram: usize) -> Option<Vec<u64>> {
    if seed != 0 || vocab != 92_620 || max_ngram != 3 {
        return None;
    }
    Some(match layer_id {
        2 => vec![81_853_575_799_317, 39_690_085_502_883, 24_229_915_018_165],
        12 => vec![19_457_948_045_719, 44_729_556_510_419, 84_779_352_922_315],
        22 => vec![72_974_230_057_307, 21_530_602_121_221, 55_998_057_243_097],
        32 => vec![29_128_509_562_787, 16_745_155_128_523, 22_474_588_302_733],
        _ => return None,
    })
}

/// Convert the checkpoint in `dir` into `output` under the `quant` profile.
pub fn convert(dir: &Path, quant: &str, output: &str) -> anyhow::Result<()> {
    let profile = parse_profile(quant)?;
    let t0 = std::time::Instant::now();
    let config = read_json(dir, "config.json").ok_or_else(|| anyhow::anyhow!("{}: no readable config.json", dir.display()))?;
    anyhow::ensure!(detect(&config), "{}: not a Hy4-flash checkpoint", dir.display());
    let tc = config.get("text_config").cloned().ok_or_else(|| anyhow::anyhow!("hy_v4_flash: no text_config"))?;
    let g = |k: &str| -> anyhow::Result<usize> {
        tc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| anyhow::anyhow!("hy_v4_flash: text_config has no {k}"))
    };
    let hidden = g("hidden_size")?;
    let n_layers = g("num_hidden_layers")?;
    let n_exp = g("n_routed_experts")?;
    let hc = g("hc_mult")?;
    let rope = g("qk_rope_head_dim")?;
    let nope = g("qk_nope_head_dim")?;
    let swa_rope = g("swa_qk_rope_head_dim")?;
    let swa_nope = g("swa_qk_nope_head_dim")?;
    anyhow::ensure!(rope == swa_rope && nope == swa_nope, "hy_v4_flash: sliding and sparse heads differ in width");
    let ix_hd = g("compressed_indexer_head_dim")?;
    let ix_nh = g("compressed_indexer_num_attention_heads")?;
    let latent = g("routed_expert_hidden_size")?;
    let moe_inter = g("moe_intermediate_size")?;
    let layer_types: Vec<String> = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("hy_v4_flash: no layer_types"))?
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();
    let mlp_types: Vec<String> = tc
        .get("mlp_layer_types")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("hy_v4_flash: no mlp_layer_types"))?
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();
    anyhow::ensure!(layer_types.len() == n_layers && mlp_types.len() == n_layers, "hy_v4_flash: layer schedule length");
    let sliding: Vec<bool> = layer_types
        .iter()
        .map(|t| match t.as_str() {
            "sliding_attention" => Ok(true),
            "deepseek_sparse_attention" | "indexed_attention" => Ok(false),
            o => Err(anyhow::anyhow!("hy_v4_flash: unknown layer type {o}")),
        })
        .collect::<anyhow::Result<_>>()?;
    let engram_ids: Vec<usize> = tc
        .get("engram_layer_ids")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
        .unwrap_or_default();
    let block = config
        .get("quantization_config")
        .or_else(|| tc.get("quantization_config"))
        .and_then(|q| q.get("weight_block_size"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(128);

    let bf16_dir = std::env::var("CMF_HY4_BF16").ok();
    let mut dirs: Vec<&Path> = vec![dir];
    if let Some(d) = bf16_dir.as_deref() {
        dirs.push(Path::new(d));
        eprintln!("  bf16 overrides from {d}");
    }
    let src = Sources::open(&dirs)?;

    let tokenizer_config = read_json(dir, "tokenizer_config.json").unwrap_or(serde_json::Value::Null);
    let generation = read_json(dir, "generation_config.json").unwrap_or(serde_json::Value::Null);
    let vocab = std::fs::read(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{}: tokenizer.json: {e}", dir.display()))?;
    let chat_template = std::fs::read_to_string(dir.join("chat_template.jinja"))
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| tokenizer_config.get("chat_template").and_then(|v| v.as_str()).map(String::from));

    let n_out_estimate = n_layers * (3 * n_exp + 64) + 512;
    let gap = CmfStreamWriter::head_reserve_for(n_out_estimate, 64);
    let tmp = format!("{output}.part");
    let w = CmfStreamWriter::new(&tmp, gap).map_err(|e| anyhow::anyhow!("create {tmp}: {e}"))?;
    let mut out = Out { w, census: Default::default(), t0, n: 0 };

    // ---- globals
    for (srcn, dst, group) in [
        ("model.embed_tokens.weight", "model.embed_tokens.weight", "skeleton"),
        ("lm_head.weight", "lm_head.weight", "skeleton"),
    ] {
        let (v, sh) = src.matrix(srcn, block)?;
        out.matrix(group, dst, matrix_codec(dst, profile), &v, sh[0], sh[1])?;
    }
    for (srcn, dst) in [
        ("model.norm.weight", "model.norm.weight"),
        ("model.hc_head.hc_head_fn", "model.hc_head.fn"),
        ("model.hc_head.hc_head_base", "model.hc_head.base"),
        ("model.hc_head.hc_head_scale", "model.hc_head.scale"),
    ] {
        let (v, sh) = src.matrix(srcn, block)?;
        out.f32("skeleton", dst, &sh, &v)?;
    }

    let mut engram_meta = serde_json::Map::new();
    // CMF_HY4_ENGRAM_FROM=<file.cmf>: the Engram tables come from another
    // profile's output (they are the same in every profile), so the 127 GB
    // of bf16 tables need not stay on disk for a second conversion.
    let engram_from = match std::env::var("CMF_HY4_ENGRAM_FROM") {
        Ok(p) => {
            eprintln!("  Engram tables from {p}");
            Some(cortiq_core::format::CmfModel::open(&p).map_err(|e| anyhow::anyhow!("open {p}: {e}"))?)
        }
        Err(_) => None,
    };
    for li in 0..n_layers {
        let p = format!("model.layers.{li}.");
        // ---- Engram (before the attention hyper-connection)
        if engram_ids.contains(&(li + 1)) {
            let e = format!("{p}engram_layer.");
            let d = format!("{p}engram.");
            let (rows, hd);
            if let Some(prev) = engram_from.as_ref() {
                // the same e4m3 rows another profile of this checkpoint wrote
                let t = prev.tensor(&format!("{d}table")).ok_or_else(|| anyhow::anyhow!("{li}: no Engram table in CMF_HY4_ENGRAM_FROM"))?;
                anyhow::ensure!(t.dtype == TensorDtype::U8 && t.shape.len() == 2, "{li}: Engram table {:?} {:?}", t.dtype, t.shape);
                rows = t.shape[0];
                hd = t.shape[1];
                let b = prev.tensor_bytes(&format!("{d}table")).map_err(|e| anyhow::anyhow!("{li}: {e}"))?;
                out.push("engram", &format!("{d}table"), TensorDtype::U8, &[rows, hd], b)?;
                let b = prev.tensor_bytes(&format!("{d}table_scale")).map_err(|e| anyhow::anyhow!("{li}: {e}"))?;
                out.push("engram", &format!("{d}table_scale"), TensorDtype::Bf16, &[rows], b)?;
            } else {
            let (tf, tm) = src.need(&format!("{e}multi_head_embedding.embedding.weight"))?;
            anyhow::ensure!(tm.dtype == "BF16" && tm.shape.len() == 2, "{li}: Engram table {} {:?}", tm.dtype, tm.shape);
            hd = tm.shape[1];
            rows = tm.shape[0];
            let raw = tf.bytes(tm);
            let mut codes = vec![0u8; rows * hd];
            let mut scales = vec![0u8; rows * 2];
            let threads = crate::convert::encode_threads().max(1);
            let per = rows.div_ceil(threads);
            std::thread::scope(|s| {
                for ((cc, sc), r0) in codes.chunks_mut(per * hd).zip(scales.chunks_mut(per * 2)).zip((0..).step_by(per)) {
                    s.spawn(move || {
                        let mut x = vec![0f32; hd];
                        for (k, (crow, srow)) in cc.chunks_mut(hd).zip(sc.chunks_mut(2)).enumerate() {
                            let r = r0 + k;
                            for (j, v) in x.iter_mut().enumerate() {
                                let o = (r * hd + j) * 2;
                                *v = f32::from_bits((u16::from_le_bytes([raw[o], raw[o + 1]]) as u32) << 16);
                            }
                            let s = engram_row(&x, crow);
                            srow.copy_from_slice(&bf16_bits(s).to_le_bytes());
                        }
                    });
                }
            });
            out.push("engram", &format!("{d}table"), TensorDtype::U8, &[rows, hd], &codes)?;
            drop(codes);
            out.push("engram", &format!("{d}table_scale"), TensorDtype::Bf16, &[rows], &scales)?;
            drop(scales);
            }
            let (lt, lm) = src.need(&format!("{e}lookup_table"))?;
            anyhow::ensure!(lm.dtype == "I32", "{li}: lookup table dtype {}", lm.dtype);
            let lookup: Vec<u32> = lt.bytes(lm).chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()).max(0) as u32).collect();
            let compressed = lookup.iter().copied().max().unwrap_or(0) as usize + 1;
            let lb: Vec<u8> = lookup.iter().flat_map(|v| v.to_le_bytes()).collect();
            out.push("engram", &format!("{d}lookup"), TensorDtype::U32, &[lookup.len()], &lb)?;
            let n_heads = tc["engram_layer_head_vocab_sizes"][format!("{}", li + 1)]
                .as_array()
                .map(|o| o.iter().map(|a| a.as_array().map_or(0, |a| a.len())).sum::<usize>())
                .unwrap_or(0);
            let mut hn = Vec::with_capacity(n_heads * hd);
            for h in 0..n_heads {
                hn.extend(src.matrix(&format!("{e}lookup_per_head_norms.{h}.weight"), block)?.0);
            }
            out.f32("engram", &format!("{d}head_norms"), &[n_heads, hd], &hn)?;
            for j in 0..hc {
                let (v, sh) = src.matrix(&format!("{e}key_projs.{j}.weight"), block)?;
                let n = format!("{d}key_projs.{j}.weight");
                out.matrix("engram", &n, matrix_codec(&n, profile), &v, sh[0], sh[1])?;
                let (b, sh) = src.matrix(&format!("{e}key_projs.{j}.bias"), block)?;
                out.f32("engram", &format!("{d}key_projs.{j}.bias"), &sh, &b)?;
                for k in ["norm_key", "norm_query"] {
                    let (v, sh) = src.matrix(&format!("{e}{k}.{j}.weight"), block)?;
                    out.f32("engram", &format!("{d}{k}.{j}.weight"), &sh, &v)?;
                }
            }
            let (v, sh) = src.matrix(&format!("{e}value_proj.weight"), block)?;
            let n = format!("{d}value_proj.weight");
            out.matrix("engram", &n, matrix_codec(&n, profile), &v, sh[0], sh[1])?;
            let (b, sh) = src.matrix(&format!("{e}value_proj.bias"), block)?;
            out.f32("engram", &format!("{d}value_proj.bias"), &sh, &b)?;
            let (v, sh) = src.matrix(&format!("{e}unified_pre_gamma"), block)?;
            out.f32("engram", &format!("{d}pre_gamma"), &sh, &v)?;
            let mut hs = Vec::with_capacity(hc * hidden);
            for j in 0..hc {
                hs.extend(src.matrix(&format!("{e}unified_per_hc_scales.{j}"), block)?.0);
            }
            out.f32("engram", &format!("{d}hc_scales"), &[hc, hidden], &hs)?;
            let seed = tc.get("engram_seed").and_then(|v| v.as_u64()).unwrap_or(0);
            let max_ngram = tc.get("engram_max_ngram_size").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
            let mult = release_multipliers(li + 1, seed, compressed, max_ngram).ok_or_else(|| {
                anyhow::anyhow!(
                    "hy_v4_flash: no Engram hash multipliers for layer id {} (seed {seed}, compressed vocabulary {compressed}); \
                     this checkpoint is not the released one",
                    li + 1
                )
            })?;
            engram_meta.insert(
                format!("{li}"),
                serde_json::json!({ "layer_id": li + 1, "multipliers": mult, "compressed_vocab": compressed, "rows": rows }),
            );
        }

        // ---- hyper-connections and norms
        for (kind, dst) in [("hc_attn_layer", "attn_hc"), ("hc_mlp_layer", "ffn_hc")] {
            for part in ["fn", "base", "scale"] {
                let (v, sh) = src.matrix(&format!("{p}{kind}.hc_pre.hc_{part}"), block)?;
                out.f32("skeleton", &format!("{p}{dst}.{part}"), &sh, &v)?;
            }
        }
        for n in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            let (v, sh) = src.matrix(&format!("{p}{n}"), block)?;
            out.f32("skeleton", &format!("{p}{n}"), &sh, &v)?;
        }

        // ---- attention
        let a = format!("{p}self_attn.");
        let hd = rope + nope;
        for n in ["q_a_layernorm.weight", "kv_a_layernorm.weight", "post_proj_norm.weight"] {
            let (v, sh) = src.matrix(&format!("{a}{n}"), block)?;
            out.f32("skeleton", &format!("{a}{n}"), &sh, &v)?;
        }
        let (v, sh) = src.matrix(&format!("{a}learnable_sink_param"), block)?;
        out.f32("skeleton", &format!("{a}sinks"), &sh, &v)?;
        for (srcn, dst) in [
            ("q_a_proj.weight", "q_a_proj.weight"),
            ("q_b_proj.weight", "q_b_proj.weight"),
            ("kv_a_proj_with_mqa.weight", "kv_a_proj_with_mqa.weight"),
            ("kv_b_proj.weight", "kv_b_proj.weight"),
            ("linear_gate.weight", "gate_proj.weight"),
            ("o_proj.weight", "o_proj.weight"),
        ] {
            let (mut v, sh) = src.matrix(&format!("{a}{srcn}"), block)?;
            let (rows, cols) = (sh[0], sh[1]);
            if srcn == "q_b_proj.weight" {
                anyhow::ensure!(rows % hd == 0, "{a}{srcn}: rows {rows} not a multiple of {hd}");
                v = rope_first_heads(&v, cols, nope, rope);
            }
            if srcn == "kv_a_proj_with_mqa.weight" {
                permute_rope_rows(&mut v, cols, rows, rows - rope, rope);
            }
            let n = format!("{a}{dst}");
            out.matrix("skeleton", &n, matrix_codec(&n, profile), &v, rows, cols)?;
        }
        if !sliding[li] {
            let i = format!("{a}compressed_indexer.");
            let o = format!("{a}indexer.");
            // rotary channels of the index heads and the pooled key: the last
            // `rope` of each `ix_hd`-wide head
            let (mut v, sh) = src.matrix(&format!("{i}query_projection.weight"), block)?;
            anyhow::ensure!(sh[0] == ix_nh * ix_hd, "{i}query_projection {:?}", sh);
            permute_rope_rows(&mut v, sh[1], ix_hd, ix_hd - rope, rope);
            let n = format!("{o}query_projection.weight");
            out.matrix("skeleton", &n, matrix_codec(&n, profile), &v, sh[0], sh[1])?;
            for k in ["head_weight_projection.weight", "token_key_projection.weight", "token_key_norm.weight"] {
                let (v, sh) = src.matrix(&format!("{i}{k}"), block)?;
                out.f32("skeleton", &format!("{o}{k}"), &sh, &v)?;
            }
            for k in ["key_value_projection.weight", "key_gate_projection.weight"] {
                let (mut v, sh) = src.matrix(&format!("{i}{k}"), block)?;
                permute_rope_rows(&mut v, sh[1], ix_hd, ix_hd - rope, rope);
                out.f32("skeleton", &format!("{o}{k}"), &sh, &v)?;
            }
            // per-channel vectors: the same permutation on their last axis
            let (mut v, sh) = src.matrix(&format!("{i}key_norm.weight"), block)?;
            permute_rope_rows(&mut v, 1, ix_hd, ix_hd - rope, rope);
            out.f32("skeleton", &format!("{o}key_norm.weight"), &sh, &v)?;
            let (ape, sh) = src.matrix(&format!("{i}intra_group_ape"), block)?;
            let mut t = vec![0f32; ape.len()];
            let (slots, w) = (sh[0], sh[1]);
            for s in 0..slots {
                for c in 0..w {
                    let sc = if c >= w - rope { (w - rope) + rope_src(c - (w - rope), rope) } else { c };
                    t[s * w + c] = ape[s * w + sc];
                }
            }
            out.f32("skeleton", &format!("{o}intra_group_ape"), &sh, &t)?;
        }

        // ---- FFN
        let m = format!("{p}mlp.");
        if mlp_types[li] == "dense" {
            for k in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
                let (v, sh) = src.matrix(&format!("{m}{k}"), block)?;
                let n = format!("{m}{k}");
                out.matrix("skeleton", &n, matrix_codec(&n, profile), &v, sh[0], sh[1])?;
            }
            let (v, sh) = src.matrix(&format!("{m}post_fc2_norm.weight"), block)?;
            out.f32("skeleton", &format!("{m}post_fc2_norm.weight"), &sh, &v)?;
            continue;
        }
        let (v, sh) = src.matrix(&format!("{m}gate.weight"), block)?;
        out.f32("skeleton", &format!("{m}gate.weight"), &sh, &v)?;
        let (v, sh) = src.matrix(&format!("{m}gate.e_score_correction_bias"), block)?;
        out.f32("skeleton", &format!("{m}gate.e_score_correction_bias"), &sh, &v)?;
        let (v, sh) = src.matrix(&format!("{m}combine_output_norm.weight"), block)?;
        out.f32("skeleton", &format!("{m}combine_output_norm.weight"), &sh, &v)?;
        for (srcn, dst) in [
            ("routed_expert_down_proj.weight", "latent_down.weight"),
            ("routed_expert_up_proj.weight", "latent_up.weight"),
            ("shared_experts.gate_proj.weight", "shared_experts.gate_proj.weight"),
            ("shared_experts.up_proj.weight", "shared_experts.up_proj.weight"),
            ("shared_experts.down_proj.weight", "shared_experts.down_proj.weight"),
        ] {
            let (v, sh) = src.matrix(&format!("{m}{srcn}"), block)?;
            let n = format!("{m}{dst}");
            out.matrix("skeleton", &n, matrix_codec(&n, profile), &v, sh[0], sh[1])?;
        }
        let gu = format!("{m}experts.gate_up_proj");
        let dn = format!("{m}experts.down_proj");
        for e in 0..n_exp {
            let (v, rows, cols) = src.bank_slice(&gu, e, block)?;
            anyhow::ensure!(rows == 2 * moe_inter && cols == latent, "{gu}: expert shape {rows}x{cols}");
            let half = moe_inter * cols;
            for (k, part) in [("gate_proj", &v[..half]), ("up_proj", &v[half..])] {
                let n = format!("{m}experts.{e}.{k}.weight");
                out.matrix("experts", &n, matrix_codec(&n, profile), part, moe_inter, cols)?;
            }
            let (v, rows, cols) = src.bank_slice(&dn, e, block)?;
            anyhow::ensure!(rows == latent && cols == moe_inter, "{dn}: expert shape {rows}x{cols}");
            let n = format!("{m}experts.{e}.down_proj.weight");
            out.matrix("experts", &n, matrix_codec(&n, profile), &v, rows, cols)?;
        }
        eprintln!("  layer {li} done, {:.0}s", t0.elapsed().as_secs_f64());
    }

    // ---- vision tower, verbatim
    let mut vit: Vec<&String> = src.by_name.keys().filter(|k| k.starts_with("vit.")).collect();
    vit.sort();
    for k in vit {
        let (f, m) = src.need(k)?;
        let dt = match m.dtype.as_str() {
            "BF16" => TensorDtype::Bf16,
            "F16" => TensorDtype::F16,
            "F32" => TensorDtype::F32,
            "F8_E4M3" => {
                let (v, sh) = src.matrix(k, block)?;
                out.f32("vision", k, &sh, &v)?;
                continue;
            }
            d => anyhow::bail!("{k}: vision dtype {d}"),
        };
        out.push("vision", k, dt, &m.shape, f.bytes(m))?;
    }
    // every source tensor is either packed or deliberately left out
    let used = src.used.lock().unwrap();
    let mut left: Vec<&String> = src
        .by_name
        .keys()
        .filter(|k| !used.contains(*k) && !k.contains("dfly_layers") && !k.ends_with("_scale_inv"))
        .collect();
    left.sort();
    anyhow::ensure!(left.is_empty(), "hy_v4_flash: source tensors left unpacked: {left:?}");
    drop(used);
    for (g, (n, wts, b)) in &out.census {
        eprintln!("  {g}: {n} tensors, {:.2} B weights → {:.2} GB", *wts as f64 / 1e9, *b as f64 / 1e9);
    }

    let eos = generation.get("eos_token_id").and_then(|v| v.as_u64()).or_else(|| tc.get("eos_token_id").and_then(|v| v.as_u64())).unwrap_or(120025) as u32;
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": ARCH_NAME,
        "hidden_size": hidden,
        "intermediate_size": g("intermediate_size")?,
        "num_layers": n_layers,
        "num_attention_heads": g("num_attention_heads")?,
        "num_kv_heads": g("num_attention_heads")?,
        "head_dim": rope + nope,
        "v_head_dim": g("v_head_dim")?,
        "vocab_size": g("vocab_size")?,
        "layer_types": vec!["FullAttention"; n_layers],
        "rms_norm_eps": tc.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5),
        "rope_theta": tc["rope_parameters"]["deepseek_sparse_attention"]["rope_theta"].as_f64().unwrap_or(1e7),
        "rope_local_base_freq": tc["rope_parameters"]["sliding_attention"]["rope_theta"].as_f64().unwrap_or(1e4),
        "sliding_window": g("sliding_window")?,
        "attention_sink": true,
        "attention_heads_per_layer": sliding.iter().map(|&s| if s { g("swa_num_attention_heads") } else { g("num_attention_heads") }).collect::<anyhow::Result<Vec<_>>>()?,
        "max_position_embeddings": g("max_position_embeddings")?,
        "linear_conv_kernel_dim": null,
        "linear_num_key_heads": null,
        "linear_num_value_heads": null,
        "mla": {
            "kv_lora_rank": g("kv_lora_rank")?,
            "qk_rope_head_dim": rope,
            "qk_nope_head_dim": nope,
            "v_head_dim": g("v_head_dim")?,
            "q_lora_rank": g("q_lora_rank")?,
        },
        "moe": {
            "num_experts": n_exp,
            "top_k": g("num_experts_per_tok")?,
            "moe_intermediate_size": moe_inter,
            "norm_topk_prob": tc.get("norm_topk_prob").and_then(|v| v.as_bool()).unwrap_or(true),
            "shared_expert_intermediate_size": moe_inter * g("n_shared_experts")?,
            "router_sigmoid": true,
            "routed_scaling_factor": tc.get("routed_scaling_factor").and_then(|v| v.as_f64()).unwrap_or(1.0),
        },
    }))?;
    let mut text = tc.clone();
    if let Some(o) = text.as_object_mut() {
        // the hash moduli are kept (the runtime needs them); nothing else is dropped
        o.insert("engram".into(), serde_json::Value::Object(engram_meta));
    }
    let provenance = serde_json::json!({
        "source": dir.display().to_string(),
        "base_model": "tencent/Hy4-flash",
        "quant": profile_name(profile),
        "license": "tencent-hy-license",
        "hy4_flash": {
            "text_config": text,
            "vision_config": config.get("vision_config").cloned(),
            "image_token_id": config.get("image_token_id").cloned(),
            "video_token_id": config.get("video_token_id").cloned(),
            "image_start_token_id": config.get("image_start_token_id").cloned(),
            "image_end_token_id": config.get("image_end_token_id").cloned(),
            "image_newline_token_id": config.get("image_newline_token_id").cloned(),
            "rope_layout": "half_split",
            "processor": read_json(dir, "processor_config.json"),
        },
    });
    let bundle = TokenizerBundle {
        chat_template,
        eos_token_ids: vec![eos],
        bos_token_id: tc.get("bos_token_id").and_then(|v| v.as_u64()).map(|n| n as u32),
        pad_token_id: tc.get("pad_token_id").and_then(|v| v.as_u64()).map(|n| n as u32),
    };
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch,
        quant_type: match profile {
            Profile::Q8 => QuantType::Q8_2f,
            Profile::Q4 | Profile::Q2 => QuantType::Q4Block,
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
    out.w.finish(&header, None, Some(&vocab)).map_err(|e| anyhow::anyhow!("finish {tmp}: {e}"))?;
    std::fs::rename(&tmp, output)?;
    let size = std::fs::metadata(output)?.len();
    eprintln!(
        "{output}: {count} tensors, {:.2} GB ({}) in {:.0}s",
        size as f64 / 1e9,
        profile_name(profile),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_permutation_maps_pairs_to_halves() {
        // interleaved rows (x0 y0 x1 y1) → half-split (x0 x1 y0 y1)
        let mut v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        permute_rope_rows(&mut v, 1, 8, 4, 4);
        assert_eq!(v, vec![0., 1., 2., 3., 4., 6., 5., 7.]);
        let mut w: Vec<f32> = (0..12).map(|i| i as f32).collect();
        // two heads of 6 rows, rope = last 4 of each
        permute_rope_rows(&mut w, 1, 6, 2, 4);
        assert_eq!(w, vec![0., 1., 2., 4., 3., 5., 6., 7., 8., 10., 9., 11.]);
    }

    #[test]
    fn query_heads_go_rope_first() {
        // one head: nope [0, 1], rope interleaved [2, 3, 4, 5] (x0 y0 x1 y1)
        let v: Vec<f32> = (0..6).map(|i| i as f32).collect();
        assert_eq!(rope_first_heads(&v, 1, 2, 4), vec![2., 4., 3., 5., 0., 1.]);
    }

    #[test]
    fn e4m3_encode_round_trips_and_rounds_to_even() {
        for c in 0u8..=126 {
            let v = fp8_e4m3_to_f32(c);
            assert_eq!(e4m3_encode(v), c, "code {c}");
            assert_eq!(e4m3_encode(-v), if c == 0 { 0 } else { c | 0x80 });
        }
        // halfway between 1.0 (0x38) and 1.125 (0x39): even code wins
        assert_eq!(e4m3_encode(1.0625), 0x38);
        // halfway between 1.125 (0x39) and 1.25 (0x3a)
        assert_eq!(e4m3_encode(1.1875), 0x3a);
        assert_eq!(e4m3_encode(1000.0), 126);
        assert_eq!(e4m3_encode(1e-9), 0);
    }

    #[test]
    fn engram_row_matches_the_reference_round_trip() {
        let x = [0.5f32, -0.25, 0.125, 3.0, -1e-3, 0.0];
        let mut c = [0u8; 6];
        let s = engram_row(&x, &mut c);
        assert_eq!(s, bf16_round(3.0 / 448.0));
        // amax maps to 448 exactly
        assert_eq!(fp8_e4m3_to_f32(c[3]), 448.0);
        for (k, &v) in x.iter().enumerate() {
            let back = bf16_round(fp8_e4m3_to_f32(c[k]) * s);
            assert!((back - v).abs() <= v.abs() / 8.0 + s * 0.01, "{v} -> {back}");
        }
    }

    #[test]
    fn profile_codecs() {
        let gu = "model.layers.3.mlp.experts.7.gate_proj.weight";
        let dn = "model.layers.3.mlp.experts.7.down_proj.weight";
        assert_eq!(matrix_codec(gu, Profile::Q4), Quant::Q4TiledP);
        assert_eq!(matrix_codec(gu, Profile::Q2), Quant::Q2TiledP);
        assert_eq!(matrix_codec(dn, Profile::Q2), Quant::Q4TiledP);
        assert_eq!(matrix_codec("model.layers.3.self_attn.o_proj.weight", Profile::Q4), Quant::Q8_2f);
        assert_eq!(matrix_codec("model.layers.3.self_attn.o_proj.weight", Profile::Q2), Quant::Q4TiledP);
        assert_eq!(matrix_codec("lm_head.weight", Profile::Q2), Quant::Q8_2f);
        assert!(keep_f32("model.layers.3.mlp.gate.weight"));
        assert!(keep_f32("model.layers.3.attn_hc.fn"));
    }
}
