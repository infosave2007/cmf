//! Pack maxwell-model/Maxwell-V1-Flash (`model_type: maxwell_moe`) into one `.cmf`.
//!
//! `cortiq convert --model <dir with config.json> --output x.cmf --quant P`
//! detects the model type and comes here; the generic LLM walk is never
//! touched. The runtime is `cortiq_engine::maxwell`.
//!
//! ## Layout
//!
//! Per layer `model.layers.{i}.`:
//! * `attn_hc.{fn,base,scale}`, `ffn_hc.{fn,base,scale}` — the manifold
//!   hyper-connections (f32): `fn` `[2·hc + hc², hc·hidden]`
//!   (`mapping_proj.weight`), `base` its bias, `scale` the three gating
//!   factors `[alpha_pre, alpha_post, alpha_res]`;
//! * `input_layernorm.weight`, `post_attention_layernorm.weight`;
//! * `self_attn.{q,k,v,o}_proj.weight`, `self_attn.{q,k}_norm.weight` — GQA
//!   with a per-head RMSNorm on q and k before the rotation;
//! * `self_attn.indexer.{q,k}_proj.weight`, `self_attn.indexer.{q,k}_norm.weight`
//!   — the block router of the sparse attention (`[groups·id, hidden]`,
//!   `[id, hidden]`);
//! * dense layers (`mlp_only_layers`): `mlp.{gate,up,down}_proj.weight`
//!   (`w1`, `w3`, `w2` of the source);
//! * MoE layers: `mlp.gate.weight` (f32 router), `mlp.gate.e_score_correction_bias`
//!   (the selection bias, f32 as the reference insists),
//!   `mlp.experts.{e}.{gate,up,down}_proj.weight`,
//!   `mlp.shared_experts.{gate,up,down}_proj.weight`.
//!
//! Globals: `model.embed_tokens.weight`, `lm_head.weight`, `model.norm.weight`,
//! `model.hc_head.{fn,base,scale}`. The MTP layer (`mtp_layers.*`) is left out.
//!
//! ## Profiles (`--quant`)
//!
//! * `q8_2f` (default, `auto`): every matrix q8_2f.
//! * `q4tp`: routed experts q4tp; every other matrix q8_2f.
//! * `q2tp`: routed experts' gate/up q2tp, down q4tp; every other matrix q4tp
//!   but the embedding and the head (q8_2f).
//!
//! The embedding is f16 in every profile (a row a token, read on the host).
//! Norms, routers, hyper-connection weights and the indexer's projections
//! stay f32. `--tensor-quant PATTERN=QUANT` overrides any matrix. The
//! hyper-parameters the runtime needs ride in the header provenance under
//! `maxwell_moe`; `sampling` there holds the release's settings (temperature
//! 0.6, top-p 0.95, top-k 20) unless `--sampling` names others.

use crate::convert::{Quant, SafeTensors, TensorMeta, open_safetensors, quantize_2d, tensor_quant_override, to_f32};
use cortiq_core::format::{CmfHeader, CmfStreamWriter, TokenizerBundle};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use std::collections::HashMap;
use std::path::Path;

pub const ARCH_NAME: &str = "maxwell_moe";

/// Is `config` a Maxwell MoE checkpoint?
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
        "auto" | "q8_2f" | "q82f" | "q8" => Profile::Q8,
        "q4tp" | "q4" => Profile::Q4,
        "q2tp" | "q2" => Profile::Q2,
        other => anyhow::bail!("maxwell_moe: unknown --quant '{other}' (q8_2f | q4tp | q2tp | auto)"),
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
    // the embedding rows of the frequent tokens carry outliers (max / rms of 15):
    // an 8-bit row loses 3.5 % of them, 2 % of the stream after the first layer.
    // The table is read a row a token on the host, so f16 costs no card memory.
    if name == "model.embed_tokens.weight" {
        return Quant::F16;
    }
    if profile == Profile::Q2 && name != "lm_head.weight" {
        return Quant::Q4TiledP;
    }
    Quant::Q8_2f
}

/// Weights kept exact (f32) whatever the profile: small or precision-critical.
fn keep_f32(name: &str) -> bool {
    name.ends_with(".mlp.gate.weight") || name.contains("_hc.") || name.contains("hc_head.") || name.contains(".indexer.")
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

    fn need(&mut self, name: &str) -> anyhow::Result<(&SafeTensors, &TensorMeta)> {
        self.used.insert(name.to_string());
        let &(f, t) = self.by_name.get(name).ok_or_else(|| anyhow::anyhow!("maxwell_moe: tensor {name} missing"))?;
        Ok((&self.files[f], &self.files[f].tensors[t]))
    }

    /// A tensor in f32 with its shape.
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
        Ok(())
    }

    fn f32(&mut self, group: &'static str, name: &str, shape: &[usize], v: &[f32]) -> anyhow::Result<()> {
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

/// The sampling the Maxwell team runs the model with (the model card's usage
/// section): temperature 0.6, top-p 0.95, top-k 20, no penalties.
fn release_sampling() -> serde_json::Value {
    serde_json::json!({ "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0 })
}

/// One hyper-connection: the mapping, its bias and the three gating factors.
fn pack_hc(src: &mut Sources, out: &mut Out, from: &str, to: &str, hc: usize, hidden: usize) -> anyhow::Result<()> {
    let mix = 2 * hc + hc * hc;
    let (w, rows, cols) = src.matrix(&format!("{from}.mapping_proj.weight"))?;
    anyhow::ensure!(rows == mix && cols == hc * hidden, "{from}.mapping_proj {rows}x{cols}, expected {mix}x{}", hc * hidden);
    out.f32("skeleton", &format!("{to}.fn"), &[rows, cols], &w)?;
    let (b, _) = src.f32(&format!("{from}.bias"))?;
    anyhow::ensure!(b.len() == mix, "{from}.bias has {} values, expected {mix}", b.len());
    out.f32("skeleton", &format!("{to}.base"), &[mix], &b)?;
    let mut scale = Vec::with_capacity(3);
    for k in ["alpha_pre", "alpha_post", "alpha_res"] {
        let (v, _) = src.f32(&format!("{from}.{k}"))?;
        anyhow::ensure!(v.len() == 1, "{from}.{k} has {} values", v.len());
        scale.push(v[0]);
    }
    out.f32("skeleton", &format!("{to}.scale"), &[3], &scale)
}

/// Convert the checkpoint in `dir` into `output` under the `quant` profile.
pub fn convert(dir: &Path, quant: &str, output: &str) -> anyhow::Result<()> {
    let profile = parse_profile(quant)?;
    let t0 = std::time::Instant::now();
    let config = read_json(dir, "config.json").ok_or_else(|| anyhow::anyhow!("{}: no readable config.json", dir.display()))?;
    anyhow::ensure!(detect(&config), "{}: not a Maxwell MoE checkpoint", dir.display());
    let g = |k: &str| -> anyhow::Result<usize> {
        config.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| anyhow::anyhow!("maxwell_moe: config has no {k}"))
    };
    let hidden = g("hidden_size")?;
    let n_layers = g("num_hidden_layers")?;
    let n_exp = g("num_experts")?;
    let hc = g("num_residual_streams")?;
    let nh = g("num_attention_heads")?;
    let nkv = g("num_key_value_heads")?;
    let hd = config.get("attention_head_size").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(hidden / nh.max(1));
    let moe_inter = g("moe_intermediate_size")?;
    let shared_inter = g("shared_expert_intermediate_size")?;
    let inter = g("intermediate_size")?;
    let ix_nh = g("block_sparse_gqa_indexer_n_heads")?;
    let ix_hd = g("block_sparse_gqa_indexer_head_dim")?;
    anyhow::ensure!(config.get("attention_variant").and_then(|v| v.as_str()) == Some("block_sparse_gqa"), "maxwell_moe: attention_variant is not block_sparse_gqa");
    anyhow::ensure!(config.get("enable_hyper_connections").and_then(|v| v.as_bool()) == Some(true), "maxwell_moe: a checkpoint without hyper-connections is not supported");
    anyhow::ensure!(config.get("qk_layernorm").and_then(|v| v.as_bool()) == Some(true), "maxwell_moe: a checkpoint without the q/k norms is not supported");
    anyhow::ensure!(config.get("router_score_function").and_then(|v| v.as_str()) == Some("sigmoid"), "maxwell_moe: only the sigmoid router is supported");
    anyhow::ensure!(config.get("use_shared_expert_gate").and_then(|v| v.as_bool()) != Some(true), "maxwell_moe: a gated shared expert is not supported");
    anyhow::ensure!(config.get("rope_scaling").is_none_or(|v| v.is_null()), "maxwell_moe: rope_scaling is not supported");
    anyhow::ensure!(ix_nh == nkv && ix_hd == hd, "maxwell_moe: the indexer's heads must match the KV groups");
    anyhow::ensure!((1..=4).contains(&hc), "maxwell_moe: {hc} residual streams");
    let dense: Vec<usize> = config.get("mlp_only_layers").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect()).unwrap_or_default();
    let bias_on = config.get("moe_router_enable_expert_bias").and_then(|v| v.as_bool()).unwrap_or(false);

    let mut src = Sources::open(dir)?;
    let tokenizer_config = read_json(dir, "tokenizer_config.json").unwrap_or(serde_json::Value::Null);
    let generation = read_json(dir, "generation_config.json").unwrap_or(serde_json::Value::Null);
    let vocab = std::fs::read(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{}: tokenizer.json: {e}", dir.display()))?;
    let chat_template = std::fs::read_to_string(dir.join("chat_template.jinja"))
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| tokenizer_config.get("chat_template").and_then(|v| v.as_str()).map(String::from));

    let n_out_estimate = n_layers * (3 * n_exp + 32) + 64;
    let gap = CmfStreamWriter::head_reserve_for(n_out_estimate, 64);
    let tmp = format!("{output}.part");
    let w = CmfStreamWriter::new(&tmp, gap).map_err(|e| anyhow::anyhow!("create {tmp}: {e}"))?;
    let mut out = Out { w, census: Default::default(), n: 0 };

    // ---- globals
    for name in ["model.embed_tokens.weight", "lm_head.weight"] {
        let (v, rows, cols) = src.matrix(name)?;
        out.matrix("skeleton", name, matrix_codec(name, profile), &v, rows, cols)?;
    }
    let (v, sh) = src.f32("model.norm.weight")?;
    out.f32("skeleton", "model.norm.weight", &sh, &v)?;
    for (from, to) in [("hc_head_fn", "fn"), ("hc_head_base", "base"), ("hc_head_scale", "scale")] {
        let (v, sh) = src.f32(&format!("model.hc_head.{from}"))?;
        out.f32("skeleton", &format!("model.hc_head.{to}"), &sh, &v)?;
    }

    for li in 0..n_layers {
        let p = format!("model.layers.{li}.");
        pack_hc(&mut src, &mut out, &format!("{p}self_attn_hyper_connection"), &format!("{p}attn_hc"), hc, hidden)?;
        pack_hc(&mut src, &mut out, &format!("{p}mlp_hyper_connection"), &format!("{p}ffn_hc"), hc, hidden)?;
        for n in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            let (v, sh) = src.f32(&format!("{p}{n}"))?;
            out.f32("skeleton", &format!("{p}{n}"), &sh, &v)?;
        }

        // ---- attention
        let a = format!("{p}self_attn.");
        for (k, rows, cols) in [("q_proj", nh * hd, hidden), ("k_proj", nkv * hd, hidden), ("v_proj", nkv * hd, hidden), ("o_proj", hidden, nh * hd)] {
            let n = format!("{a}{k}.weight");
            let (v, r, c) = src.matrix(&n)?;
            anyhow::ensure!(r == rows && c == cols, "{n}: {r}x{c}, expected {rows}x{cols}");
            out.matrix("skeleton", &n, matrix_codec(&n, profile), &v, r, c)?;
        }
        for (from, to) in [("q_layernorm", "q_norm"), ("k_layernorm", "k_norm")] {
            let (v, sh) = src.f32(&format!("{a}{from}.weight"))?;
            anyhow::ensure!(v.len() == hd, "{a}{from}: {} values, expected {hd}", v.len());
            out.f32("skeleton", &format!("{a}{to}.weight"), &sh, &v)?;
            let (v, sh) = src.f32(&format!("{a}indexer.{from}.weight"))?;
            anyhow::ensure!(v.len() == ix_hd, "{a}indexer.{from}: {} values, expected {ix_hd}", v.len());
            out.f32("skeleton", &format!("{a}indexer.{to}.weight"), &sh, &v)?;
        }
        for (from, to, rows) in [("linear_q_proj", "q_proj", ix_nh * ix_hd), ("linear_k_proj", "k_proj", ix_hd)] {
            let (v, r, c) = src.matrix(&format!("{a}indexer.{from}.weight"))?;
            anyhow::ensure!(r == rows && c == hidden, "{a}indexer.{from}: {r}x{c}, expected {rows}x{hidden}");
            out.f32("skeleton", &format!("{a}indexer.{to}.weight"), &[r, c], &v)?;
        }

        // ---- FFN: the source's SwiGLU is w2(act(w1 x) · w3 x)
        let s = format!("{p}block_sparse_moe.");
        let m = format!("{p}mlp.");
        const TRIO: [(&str, &str); 3] = [("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")];
        if dense.contains(&li) {
            for (from, to) in TRIO {
                let (v, r, c) = src.matrix(&format!("{s}{from}.weight"))?;
                anyhow::ensure!(if to == "down_proj" { r == hidden && c == inter } else { r == inter && c == hidden }, "{s}{from}: {r}x{c}");
                let n = format!("{m}{to}.weight");
                out.matrix("skeleton", &n, matrix_codec(&n, profile), &v, r, c)?;
            }
            eprintln!("  layer {li} (dense) done, {:.0}s", t0.elapsed().as_secs_f64());
            continue;
        }
        let (v, r, c) = src.matrix(&format!("{s}gate.weight"))?;
        anyhow::ensure!(r == n_exp && c == hidden, "{s}gate: {r}x{c}");
        out.f32("skeleton", &format!("{m}gate.weight"), &[r, c], &v)?;
        let bias = if bias_on {
            let (f, t) = src.need(&format!("{s}expert_bias"))?;
            // the spread between experts is below a bf16 step: a half-precision bias collapses the routing
            anyhow::ensure!(t.dtype == "F32", "{s}expert_bias is {} in the checkpoint; the router needs it in f32", t.dtype);
            to_f32(&t.dtype, f.bytes(t))?
        } else {
            vec![0f32; n_exp]
        };
        anyhow::ensure!(bias.len() == n_exp, "{s}expert_bias has {} values", bias.len());
        out.f32("skeleton", &format!("{m}gate.e_score_correction_bias"), &[n_exp], &bias)?;
        for (from, to) in TRIO {
            let (v, r, c) = src.matrix(&format!("{s}shared_expert.{from}.weight"))?;
            anyhow::ensure!(if to == "down_proj" { r == hidden && c == shared_inter } else { r == shared_inter && c == hidden }, "{s}shared_expert.{from}: {r}x{c}");
            let n = format!("{m}shared_experts.{to}.weight");
            out.matrix("skeleton", &n, matrix_codec(&n, profile), &v, r, c)?;
        }
        // the routed experts: read and encode a batch at a time on all cores
        let threads = crate::convert::encode_threads().max(1);
        let mut jobs: Vec<(String, Vec<f32>, usize, usize)> = Vec::new();
        let flush = |jobs: &mut Vec<(String, Vec<f32>, usize, usize)>, out: &mut Out| -> anyhow::Result<()> {
            let enc: Vec<(TensorDtype, Vec<u8>)> = std::thread::scope(|sc| {
                let hs: Vec<_> = jobs
                    .iter()
                    .map(|(n, v, r, c)| {
                        let q = matrix_codec(n, profile);
                        sc.spawn(move || quantize_2d(q, v, *r, *c))
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().expect("encoder thread")).collect()
            });
            for ((n, _, r, c), (dt, bytes)) in jobs.iter().zip(&enc) {
                out.push("experts", n, *dt, &[*r, *c], bytes)?;
            }
            jobs.clear();
            Ok(())
        };
        for e in 0..n_exp {
            for (from, to) in TRIO {
                let (v, r, c) = src.matrix(&format!("{s}experts.{e}.{from}.weight"))?;
                anyhow::ensure!(if to == "down_proj" { r == hidden && c == moe_inter } else { r == moe_inter && c == hidden }, "{s}experts.{e}.{from}: {r}x{c}");
                jobs.push((format!("{m}experts.{e}.{to}.weight"), v, r, c));
            }
            if jobs.len() >= threads * 3 {
                flush(&mut jobs, &mut out)?;
            }
        }
        flush(&mut jobs, &mut out)?;
        eprintln!("  layer {li} done, {:.0}s", t0.elapsed().as_secs_f64());
    }

    // every source tensor is either packed or deliberately left out (the MTP layer)
    let mut left: Vec<&String> = src.by_name.keys().filter(|k| !src.used.contains(*k) && !k.starts_with("mtp_layers.")).collect();
    left.sort();
    anyhow::ensure!(left.is_empty(), "maxwell_moe: source tensors left unpacked: {left:?}");
    for (grp, (n, wts, b)) in &out.census {
        eprintln!("  {grp}: {n} tensors, {:.2} B weights → {:.2} GB", *wts as f64 / 1e9, *b as f64 / 1e9);
    }

    let mut eos: Vec<u32> = match generation.get("eos_token_id") {
        Some(serde_json::Value::Array(a)) => a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect(),
        Some(v) => v.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
        None => Vec::new(),
    };
    if eos.is_empty() {
        eos.push(config.get("eos_token_id").and_then(|v| v.as_u64()).unwrap_or(2) as u32);
    }
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": ARCH_NAME,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_layers": n_layers,
        "num_attention_heads": nh,
        "num_kv_heads": nkv,
        "head_dim": hd,
        "vocab_size": g("vocab_size")?,
        "layer_types": vec!["FullAttention"; n_layers],
        "rms_norm_eps": config.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6),
        "rope_theta": config.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(2e7),
        "max_position_embeddings": g("max_position_embeddings")?,
        "linear_conv_kernel_dim": null,
        "linear_num_key_heads": null,
        "linear_num_value_heads": null,
        "moe": {
            "num_experts": n_exp,
            "top_k": g("num_experts_per_tok")?,
            "moe_intermediate_size": moe_inter,
            "norm_topk_prob": true,
            "shared_expert_intermediate_size": shared_inter,
            "router_sigmoid": true,
        },
    }))?;
    let mut text = config.clone();
    if let Some(o) = text.as_object_mut() {
        for k in ["auto_map", "architectures", "block_sparse_gqa_layer_freq", "transformers_version", "torch_dtype"] {
            o.remove(k);
        }
    }
    let provenance = serde_json::json!({
        "source": dir.display().to_string(),
        "base_model": "maxwell-model/Maxwell-V1-Flash",
        "quant": profile_name(profile),
        "license": "apache-2.0",
        // the release's own settings (its model card), unless `--sampling` names others:
        // `run` and `serve` read them, flags and request fields still override
        "sampling": crate::convert::sampling_defaults().unwrap_or_else(release_sampling),
        "maxwell_moe": { "config": text },
    });
    let bundle = TokenizerBundle {
        chat_template,
        eos_token_ids: eos,
        bos_token_id: config.get("bos_token_id").and_then(|v| v.as_u64()).map(|n| n as u32),
        pad_token_id: config.get("pad_token_id").and_then(|v| v.as_u64()).map(|n| n as u32),
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
    eprintln!("{output}: {count} tensors, {:.2} GB ({}) in {:.0}s", size as f64 / 1e9, profile_name(profile), t0.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_codecs() {
        let gu = "model.layers.3.mlp.experts.7.gate_proj.weight";
        let dn = "model.layers.3.mlp.experts.7.down_proj.weight";
        assert_eq!(matrix_codec(gu, Profile::Q8), Quant::Q8_2f);
        assert_eq!(matrix_codec(gu, Profile::Q4), Quant::Q4TiledP);
        assert_eq!(matrix_codec(gu, Profile::Q2), Quant::Q2TiledP);
        assert_eq!(matrix_codec(dn, Profile::Q2), Quant::Q4TiledP);
        assert_eq!(matrix_codec("model.layers.3.self_attn.o_proj.weight", Profile::Q4), Quant::Q8_2f);
        assert_eq!(matrix_codec("model.layers.3.self_attn.o_proj.weight", Profile::Q2), Quant::Q4TiledP);
        assert_eq!(matrix_codec("model.layers.3.mlp.shared_experts.up_proj.weight", Profile::Q4), Quant::Q8_2f);
        assert_eq!(matrix_codec("lm_head.weight", Profile::Q2), Quant::Q8_2f);
        assert_eq!(matrix_codec("model.embed_tokens.weight", Profile::Q8), Quant::F16);
        assert_eq!(matrix_codec("model.embed_tokens.weight", Profile::Q2), Quant::F16);
        assert!(keep_f32("model.layers.3.mlp.gate.weight"));
        assert!(keep_f32("model.layers.3.attn_hc.fn"));
        assert!(keep_f32("model.layers.3.self_attn.indexer.q_proj.weight"));
        assert!(!keep_f32("model.layers.3.self_attn.q_proj.weight"));
    }

    #[test]
    fn detects_the_model_type() {
        assert!(detect(&serde_json::json!({"model_type": "maxwell_moe"})));
        assert!(!detect(&serde_json::json!({"model_type": "hy_v4_flash"})));
    }
}
