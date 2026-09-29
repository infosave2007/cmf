//! Pack a diffusers Qwen-Image-2.1 directory (`QwenImage21Pipeline`) into
//! ONE .cmf that `cortiq imagine file.cmf --prompt …` runs with no other
//! flags (text-to-image, RGBA, and editing with `--image`).
//!
//! Layout of the container:
//! - `dit.*`  every transformer tensor under its diffusers name. The seven
//!   block projections carry `--quant` (default q4tp); `--dit-keep` names
//!   the ones kept at q8_2f instead (a block `blocks.N`, a projection kind
//!   `to_out`/`out`/…, or one tensor `blocks.N.attn.to_q`). The text
//!   projection, the shared modulation, the time MLP, the final norm's
//!   linear, `img_in` and `proj_out` stay at the source bf16 (they run
//!   once per prompt or step on the host); norms are f32.
//! - `te.*`   Qwen3-VL-8B's language model, all 36 layers, no final norm
//!   (the pipeline reads the last layer before it); projections at
//!   `--te-quant` (default q8 = q8_2f: q4tp moves the last hidden state 40 %
//!   against fp32, q8_2f 11 %), `embed_tokens` q8_row, norms f32; `--te-keep` names the
//!   projections kept at the source bf16.
//! - `vis.*`  the Qwen3-VL vision tower + mergers (for condition images),
//!   projections at `--vis-quant` (default q8_2f). `--no-vision` omits it
//!   (text-to-image only).
//! - `vae.*`  decoder, encoder and the quant convs at `--vae-quant`
//!   (default f16; norms and biases f32). The temporal convs never run on
//!   a single frame and are not packed.
//! - `qi21.config_json` the defaults (steps, size, cfg, system prompt) and
//!   `qi21.scheduler_json` the source scheduler.
//! - VOCAB = `processor/tokenizer.json`.

use crate::zimagepack::{self as zp, Codec};
use crate::{convert, gguf};
use anyhow::{ensure, Context};
use cortiq_core::format::{CmfHeader, CmfStreamWriter};
use cortiq_core::types::{ModelArch, TensorDtype};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub(crate) const DEFAULT_DIT_CODEC: &str = "q4tp";
pub(crate) const DEFAULT_TE_CODEC: &str = "q8";
pub(crate) const DEFAULT_VIS_CODEC: &str = "q8";
pub(crate) const DEFAULT_VAE_CODEC: &str = "f16";

pub(crate) struct PackOpts {
    pub dit: Codec,
    pub te: Codec,
    pub vis: Codec,
    pub vae: Codec,
    /// DiT projections kept at q8_2f (see the module docs).
    pub dit_keep: Vec<String>,
    /// Text-encoder projections kept at the source bf16 (`layers.N` = a
    /// whole layer, `layers.N.<suffix>`, or a bare suffix for every layer).
    pub te_keep: Vec<String>,
    pub vision: bool,
    pub source_sha: bool,
}

/// Is `root` a diffusers Qwen-Image-2.1 pipeline directory?
pub(crate) fn is_qi21_root(root: &Path) -> bool {
    std::fs::read(root.join("model_index.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["_class_name"] == "QwenImage21Pipeline")
}

const BLOCK_PROJ: [&str; 7] = [
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "img_mlp.gate_layer.weight",
    "img_mlp.proj.weight",
    "img_mlp.out.weight",
];

/// `--dit-keep` entry match against a block projection name
/// `transformer_blocks.N.<proj>.weight`.
fn dit_keep_match(name: &str, key: &str) -> bool {
    let Some(rest) = name.strip_prefix("transformer_blocks.") else {
        return false;
    };
    let (l, proj) = rest.split_once('.').unwrap_or((rest, ""));
    let proj = proj.trim_end_matches(".weight");
    let short = proj.rsplit('.').next().unwrap_or(proj);
    let short = if short == "0" { "to_out" } else { short };
    if let Some(k) = key.strip_prefix("blocks.") {
        match k.split_once('.') {
            None => k == l,
            Some((kl, kp)) => kl == l && (kp == proj || kp == short),
        }
    } else {
        key == short || key == proj
    }
}

fn te_keep_match(name: &str, key: &str) -> bool {
    if key.starts_with("layers.") {
        name.starts_with(&format!("{key}.")) || name == key
    } else {
        name.contains(&format!(".{key}."))
    }
}

pub(crate) fn pack(root: &Path, out: &str, o: &PackOpts) -> anyhow::Result<()> {
    ensure!(
        is_qi21_root(root),
        "{}: not a diffusers QwenImage21Pipeline directory",
        root.display()
    );
    let t_all = std::time::Instant::now();
    let read = |rel: &str| -> anyhow::Result<Vec<u8>> {
        std::fs::read(root.join(rel)).with_context(|| format!("{}/{rel}", root.display()))
    };
    let dit_cfg_raw = read("transformer/config.json")?;
    let dit_cfg: serde_json::Value = serde_json::from_slice(&dit_cfg_raw)?;
    ensure!(
        dit_cfg["_class_name"] == "QwenImage21Transformer2DModel",
        "transformer is not QwenImage21Transformer2DModel"
    );
    let te_full: serde_json::Value = serde_json::from_slice(&read("text_encoder/config.json")?)?;
    let mut te_cfg = te_full["text_config"].clone();
    ensure!(te_cfg.is_object(), "text_encoder/config.json has no text_config");
    let te_layers = te_cfg["num_hidden_layers"].as_u64().context("num_hidden_layers")? as usize;
    te_cfg["final_norm"] = serde_json::json!(false);
    te_cfg["cortiq_tap"] = serde_json::json!("hidden_states[-1] before the final norm");
    let vis_cfg = te_full["vision_config"].clone();
    let vae_cfg_raw = read("vae/config.json")?;
    let sched_raw = read("scheduler/scheduler_config.json")?;
    let defaults = serde_json::json!({
        "steps": 40, "height": 1024, "width": 1024, "cfg": 1.0,
        "output_resolution": 1024,
        "system_prompt": "Comprehend and analyze the provided prompt.",
        "vae_scale": 16, "latent_channels": dit_cfg["in_channels"],
    });
    let vocab = read("processor/tokenizer.json")?;

    let dit_files = zp::shard_files(
        &root.join("transformer"),
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
    )?;
    let te_files = zp::shard_files(&root.join("text_encoder"), "model.safetensors.index.json", "model.safetensors")?;
    let vae_files = zp::shard_files(
        &root.join("vae"),
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
    )?;

    let mut source_sha = serde_json::Map::new();
    if o.source_sha {
        let files: Vec<PathBuf> = dit_files.iter().chain(&te_files).chain(&vae_files).cloned().collect();
        let hashes: Vec<anyhow::Result<String>> = std::thread::scope(|s| {
            let hs: Vec<_> = files.iter().map(|f| s.spawn(|| zp::sha256_file(f))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (f, h) in files.iter().zip(hashes) {
            let rel = f.strip_prefix(root).unwrap_or(f).display().to_string();
            source_sha.insert(rel, serde_json::json!(h?));
        }
    }

    struct Item {
        out: String,
        file: usize,
        t: usize,
        codec: Codec,
        shape: Option<Vec<usize>>,
    }
    let q8 = Codec::Q(convert::Quant::Q8_2f);
    let mut files: Vec<(PathBuf, Vec<zp::StTensor>, usize)> = Vec::new();
    let mut items: Vec<Item> = Vec::new();
    let mut n_kept = 0usize;
    for f in &dit_files {
        let (ts, size) = zp::st_header(f)?;
        let fi = files.len();
        for (ti, t) in ts.iter().enumerate() {
            let n = &t.name;
            let is_proj = n.starts_with("transformer_blocks.") && BLOCK_PROJ.iter().any(|p| n.ends_with(p));
            let codec = if is_proj {
                if o.dit_keep.iter().any(|k| dit_keep_match(n, k)) {
                    n_kept += 1;
                    q8
                } else {
                    o.dit
                }
            } else if t.shape.len() == 2 {
                Codec::Raw
            } else {
                Codec::F32
            };
            items.push(Item {
                out: format!("dit.{n}"),
                file: fi,
                t: ti,
                codec,
                shape: None,
            });
        }
        files.push((f.clone(), ts, size));
    }
    let n_dit = items.len();
    let mut n_vis = 0usize;
    for f in &te_files {
        let (ts, size) = zp::st_header(f)?;
        let fi = files.len();
        for (ti, t) in ts.iter().enumerate() {
            let name = t.name.as_str();
            if let Some(n) = name.strip_prefix("model.language_model.") {
                if n == "norm.weight" {
                    continue;
                }
                let codec = if n == "embed_tokens.weight" {
                    match o.te {
                        Codec::Q(_) => Codec::Q(convert::Quant::Q8Row),
                        c => c,
                    }
                } else if t.shape.len() == 2 && n.ends_with("_proj.weight") {
                    if o.te_keep.iter().any(|k| te_keep_match(n, k)) {
                        Codec::Raw
                    } else {
                        o.te
                    }
                } else {
                    Codec::F32
                };
                items.push(Item {
                    out: format!("te.{n}"),
                    file: fi,
                    t: ti,
                    codec,
                    shape: None,
                });
            } else if let Some(n) = name.strip_prefix("model.visual.") {
                if !o.vision {
                    continue;
                }
                n_vis += 1;
                let (out, codec, shape) = if n == "patch_embed.proj.weight" {
                    let flat: usize = t.shape[1..].iter().product();
                    ("vis.patch_embed.weight".to_string(), Codec::Raw, Some(vec![t.shape[0], flat]))
                } else if n == "patch_embed.proj.bias" {
                    ("vis.patch_embed.bias".to_string(), Codec::F32, None)
                } else if n == "pos_embed.weight" {
                    ("vis.pos_embed.weight".to_string(), Codec::F32, None)
                } else {
                    let out = format!(
                        "vis.{}",
                        n.replace("deepstack_merger_list.", "deepstack.")
                    );
                    let codec = if t.shape.len() == 2 && n.ends_with(".weight") && !n.contains("norm") {
                        o.vis
                    } else {
                        Codec::F32
                    };
                    (out, codec, None)
                };
                items.push(Item {
                    out,
                    file: fi,
                    t: ti,
                    codec,
                    shape,
                });
            }
            // lm_head and the final norm are never run
        }
        files.push((f.clone(), ts, size));
    }
    let n_te = items.len() - n_dit - n_vis;
    for f in &vae_files {
        let (ts, size) = zp::st_header(f)?;
        let fi = files.len();
        for (ti, t) in ts.iter().enumerate() {
            let n = &t.name;
            if n.contains(".time_conv.") {
                continue; // single frame: never runs
            }
            let codec = if n.ends_with(".weight") && t.shape.len() >= 2 {
                o.vae
            } else {
                Codec::F32
            };
            items.push(Item {
                out: format!("vae.{n}"),
                file: fi,
                t: ti,
                codec,
                shape: None,
            });
        }
        files.push((f.clone(), ts, size));
    }
    ensure!(n_dit > 0 && n_te > 0, "no transformer or text-encoder tensors found");
    let mut extras: Vec<(&str, Vec<u8>)> = vec![
        ("dit.config_json", dit_cfg_raw.clone()),
        ("te.config_json", serde_json::to_vec_pretty(&te_cfg)?),
        ("vae.config_json", vae_cfg_raw.clone()),
        ("qi21.config_json", serde_json::to_vec_pretty(&defaults)?),
        ("qi21.scheduler_json", sched_raw.clone()),
    ];
    if o.vision {
        extras.push(("vis.config_json", serde_json::to_vec_pretty(&vis_cfg)?));
    }
    let count = items.len() + extras.len();
    eprintln!(
        "qwen-image-2.1 pack: {n_dit} dit ({}, {n_kept} kept q8_2f), {n_te} te ({}), {n_vis} vision ({}), {} vae ({})",
        zp::codec_name(o.dit),
        zp::codec_name(o.te),
        zp::codec_name(o.vis),
        items.len() - n_dit - n_te - n_vis,
        zp::codec_name(o.vae),
    );

    let out_path = PathBuf::from(out);
    let temp = gguf::qwen_image_temp_path(&out_path)?;
    let threads: usize = std::env::var("CMF_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .max(1);
    let mut dtype_bytes: BTreeMap<String, u64> = BTreeMap::new();
    let result = (|| -> anyhow::Result<()> {
        let mut w = CmfStreamWriter::new(&temp, CmfStreamWriter::head_reserve_for(count, 64))?;
        for (n, b) in &extras {
            w.push(n, TensorDtype::U8, &[b.len()], b)?;
        }
        let mut done = 0usize;
        let mut maps: Vec<Option<memmap2::Mmap>> = (0..files.len()).map(|_| None).collect();
        for chunk in items.chunks(threads.max(2)) {
            let need: std::collections::BTreeSet<usize> = chunk.iter().map(|it| it.file).collect();
            for (i, m) in maps.iter_mut().enumerate() {
                if !need.contains(&i) {
                    *m = None;
                }
            }
            for it in chunk {
                if maps[it.file].is_none() {
                    let (p, _, size) = &files[it.file];
                    let fh = std::fs::File::open(p)?;
                    let m = unsafe { memmap2::Mmap::map(&fh)? };
                    ensure!(m.len() == *size, "{}: size changed", p.display());
                    maps[it.file] = Some(m);
                }
            }
            let encoded: Vec<anyhow::Result<(TensorDtype, Vec<u8>)>> = std::thread::scope(|s| {
                let hs: Vec<_> = chunk
                    .iter()
                    .map(|it| {
                        let t = &files[it.file].1[it.t];
                        let raw = &maps[it.file].as_ref().unwrap()[t.range.clone()];
                        s.spawn(move || zp::encode(t, raw, it.codec))
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for (it, enc) in chunk.iter().zip(encoded) {
                let (dt, bytes) = enc.with_context(|| it.out.clone())?;
                let t = &files[it.file].1[it.t];
                *dtype_bytes.entry(format!("{dt:?}")).or_default() += bytes.len() as u64;
                w.push(&it.out, dt, it.shape.as_deref().unwrap_or(&t.shape), &bytes)?;
                done += 1;
            }
            if done % 64 < chunk.len() || done == items.len() {
                eprintln!("  {done}/{} tensors ({:.0}s)", items.len(), t_all.elapsed().as_secs_f64());
            }
        }
        drop(maps);
        let arch: ModelArch = serde_json::from_value(serde_json::json!({
            "arch_name": cortiq_engine::qwen_image21::ARCH_NAME,
            "hidden_size": te_cfg["hidden_size"],
            "intermediate_size": te_cfg["intermediate_size"],
            "num_layers": te_layers,
            "num_attention_heads": te_cfg["num_attention_heads"],
            "num_kv_heads": te_cfg["num_key_value_heads"],
            "head_dim": te_cfg["head_dim"],
            "vocab_size": te_cfg["vocab_size"],
            "layer_types": vec!["FullAttention"; te_layers],
            "rms_norm_eps": te_cfg["rms_norm_eps"],
            "max_position_embeddings": te_cfg["max_position_embeddings"],
            "linear_conv_kernel_dim": 0, "linear_num_key_heads": 0, "linear_num_value_heads": 0,
        }))?;
        let quant_type = match o.dit {
            Codec::Q(q) => gguf::quant_type_for(q),
            _ => cortiq_core::types::QuantType::F16,
        };
        let header = CmfHeader {
            format: "cmf".into(),
            version: cortiq_core::CMF_VERSION,
            arch,
            quant_type,
            provenance: Some(serde_json::json!({
                "tool": "cortiq imagine-pack",
                "pipeline": "qwen-image-2.1",
                "components": {"te": "qwen3-vl-8b language model (36 layers, pre-norm output)",
                               "vis": if o.vision { "qwen3-vl vision tower + deepstack mergers" } else { "omitted" },
                               "dit": "QwenImage21Transformer2DModel", "vae": "AutoencoderKLQwenImage21 (2-D)"},
                // the directory name only: a local path does not belong in a
                // published file, and the same source must give the same bytes
                "packed_from": root.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
                "dit_codec": zp::codec_name(o.dit),
                "dit_keep_q8_2f": o.dit_keep,
                "te_codec": zp::codec_name(o.te),
                "te_keep_bf16": o.te_keep,
                "vis_codec": zp::codec_name(o.vis),
                "vae_codec": zp::codec_name(o.vae),
                "defaults": defaults,
                "source_sha256": source_sha,
                "tensor_name_policy": "diffusers names under dit./vae., HF names under te./vis.",
            })),
            tokenizer_config: None,
            section_hashes: None,
            skills: vec![],
            shard: None,
            calibration: None,
            routing: None,
            genome: None,
            lineage: Vec::new(),
            router: None,
            segments: Vec::new(),
        };
        w.finish(&header, None, Some(&vocab))?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&temp);
        return Err(e);
    }
    std::fs::rename(&temp, &out_path)?;
    gguf::qwen_image_sync_parent(&out_path)?;
    let sha = zp::sha256_file(&out_path)?;
    let fname = out_path.file_name().and_then(|s| s.to_str()).unwrap_or(out).to_string();
    std::fs::write(format!("{out}.sha256"), format!("{sha}  {fname}\n"))?;
    let size = std::fs::metadata(&out_path)?.len();
    println!(
        "{out}: {count} tensors, {:.3} GB ({:.3} GiB), sha256 {sha}, {:.0}s",
        size as f64 / 1e9,
        size as f64 / (1u64 << 30) as f64,
        t_all.elapsed().as_secs_f64()
    );
    for (k, v) in dtype_bytes {
        println!("  {k}: {:.3} GB", v as f64 / 1e9);
    }
    Ok(())
}

pub(crate) fn parse_keep(v: &[String]) -> Vec<String> {
    v.iter().filter(|k| k.as_str() != "none").cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dit_keep_matches_blocks_kinds_and_single_tensors() {
        let n = "transformer_blocks.3.attn.to_out.0.weight";
        assert!(dit_keep_match(n, "blocks.3"));
        assert!(!dit_keep_match(n, "blocks.31"));
        assert!(dit_keep_match(n, "to_out"));
        assert!(dit_keep_match(n, "blocks.3.to_out"));
        assert!(dit_keep_match(n, "blocks.3.attn.to_out.0"));
        let m = "transformer_blocks.0.img_mlp.out.weight";
        assert!(dit_keep_match(m, "out"));
        assert!(dit_keep_match(m, "blocks.0.out"));
        assert!(!dit_keep_match(m, "proj"));
        assert!(dit_keep_match("transformer_blocks.7.img_mlp.proj.weight", "proj"));
    }
}
