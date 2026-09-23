//! Pack a diffusers Z-Image / Z-Image-Turbo directory (`ZImagePipeline`)
//! into ONE .cmf that `cortiq imagine file.cmf --prompt …` runs with no
//! other flags.
//!
//! Layout of the container:
//! - `dit.*`  every transformer tensor under its diffusers name. The seven
//!   block projections and `cap_embedder.1.weight` carry `--quant`;
//!   `adaLN_modulation.0.weight` stays 16-bit (bf16 source → bf16, exact;
//!   f32 source → f16) because it only feeds a host precompute; the
//!   t-embedder, x-embedder, final layer, pad tokens, norms and biases are
//!   f32. `dit.config_json` = the transformer config.
//! - `te.*`   Qwen3-4B layers 0..34 only (`hidden_states[-2]` is the raw
//!   output of layer 34 — layer 35 and the final norm are never run), the
//!   projections at `--te-quant`, `embed_tokens` q8_row, norms f32;
//!   `te.config_json` with `num_hidden_layers: 35` and no final norm.
//! - `vae.*`  the Flux VAE decoder only, f32 (exact; it is 0.2 GB);
//!   `vae.config_json`.
//! - `zimage.config_json` the per-model defaults (variant, steps, guidance,
//!   shift, resolution, cfg options, negative prompt, max tokens) and
//!   `zimage.scheduler_json` the source scheduler config.
//! - VOCAB = `tokenizer/tokenizer.json`.
//!
//! Streaming: every shard is mmapped and converted one tensor at a time
//! (a bounded window encodes in parallel and is written in source order),
//! so a 24.6 GB fp32 transformer never sits in RAM. Writes `<out>.sha256`.

use crate::{convert, gguf};
use anyhow::{anyhow, ensure, Context};
use cortiq_core::format::{CmfHeader, CmfStreamWriter};
use cortiq_core::types::{ModelArch, TensorDtype};
use sha2::Digest;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Default DiT projection codec (measured; see the codec table in the
/// Z-Image docs).
pub(crate) const DEFAULT_DIT_CODEC: &str = "q8";
/// Default text-encoder projection codec.
pub(crate) const DEFAULT_TE_CODEC: &str = "q8";

/// Codec of one family of weights.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Codec {
    /// Keep the source dtype bytes as they are (dev / parity containers).
    Raw,
    F32,
    Bf16,
    F16,
    /// A `convert::Quant` 2-D codec (q8_2f, q8_row, q4tp, q4t…).
    Q(convert::Quant),
}

pub(crate) fn parse_codec(s: &str) -> anyhow::Result<Codec> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "raw" | "source" => Codec::Raw,
        "f32" | "fp32" => Codec::F32,
        "bf16" => Codec::Bf16,
        "f16" | "fp16" => Codec::F16,
        // "q8" means the two-field q8 here (the better 8-bit codec at equal size).
        "q8" => Codec::Q(convert::Quant::Q8_2f),
        other => Codec::Q(convert::parse_quant(other)?),
    })
}

fn codec_name(c: Codec) -> String {
    match c {
        Codec::Raw => "raw".into(),
        Codec::F32 => "f32".into(),
        Codec::Bf16 => "bf16".into(),
        Codec::F16 => "f16".into(),
        Codec::Q(q) => convert::quant_name(q).into(),
    }
}

struct StTensor {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    range: std::ops::Range<usize>,
}

/// Parse one safetensors header (local file) → tensors in file order.
fn st_header(path: &Path) -> anyhow::Result<(Vec<StTensor>, usize)> {
    let mut f = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    let size = f.metadata()?.len() as usize;
    let mut pre = [0u8; 8];
    f.read_exact(&mut pre)?;
    let len = u64::from_le_bytes(pre) as usize;
    ensure!(len > 0 && len + 8 <= size, "{}: bad header", path.display());
    let mut raw = vec![0u8; len];
    f.read_exact(&mut raw)?;
    let json: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&raw)?;
    let base = 8 + len;
    let mut out = Vec::new();
    for (name, v) in json {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"].as_str().context("dtype")?.to_string();
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .context("shape")?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as usize)
            .collect();
        let o = v["data_offsets"].as_array().context("offsets")?;
        let (s, e) = (
            o[0].as_u64().context("off")? as usize + base,
            o[1].as_u64().context("off")? as usize + base,
        );
        ensure!(e <= size && s <= e, "{name}: range past end of file");
        let item = match dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            d => return Err(anyhow!("{name}: unsupported dtype {d}")),
        };
        ensure!(
            e - s == item * shape.iter().product::<usize>().max(1),
            "{name}: size mismatch"
        );
        out.push(StTensor {
            name,
            dtype,
            shape,
            range: s..e,
        });
    }
    out.sort_by_key(|t| t.range.start);
    Ok((out, size))
}

fn shard_files(dir: &Path, index: &str, single: &str) -> anyhow::Result<Vec<PathBuf>> {
    let ip = dir.join(index);
    if ip.exists() {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&ip)?)?;
        let mut s: Vec<String> = v["weight_map"]
            .as_object()
            .context("weight_map")?
            .values()
            .filter_map(|x| x.as_str().map(String::from))
            .collect();
        s.sort();
        s.dedup();
        Ok(s.into_iter().map(|f| dir.join(f)).collect())
    } else {
        Ok(vec![dir.join(single)])
    }
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = sha2::Sha256::new();
    let mut buf = vec![0u8; 16 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Encode one source tensor to (dtype, bytes) under `codec`.
fn encode(t: &StTensor, raw: &[u8], codec: Codec) -> anyhow::Result<(TensorDtype, Vec<u8>)> {
    let src_dtype = match t.dtype.as_str() {
        "F32" => TensorDtype::F32,
        "F16" => TensorDtype::F16,
        _ => TensorDtype::Bf16,
    };
    Ok(match codec {
        Codec::Raw => (src_dtype, raw.to_vec()),
        Codec::F32 if src_dtype == TensorDtype::F32 => (TensorDtype::F32, raw.to_vec()),
        Codec::F32 => (TensorDtype::F32, f32_bytes(&convert::to_f32(&t.dtype, raw)?)),
        Codec::Bf16 if src_dtype == TensorDtype::Bf16 => (TensorDtype::Bf16, raw.to_vec()),
        Codec::Bf16 => {
            let v = convert::to_f32(&t.dtype, raw)?;
            let b: Vec<u8> = v
                .iter()
                .flat_map(|x| {
                    // round-to-nearest-even f32 → bf16
                    let bits = x.to_bits();
                    let r = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
                    ((r >> 16) as u16).to_le_bytes()
                })
                .collect();
            (TensorDtype::Bf16, b)
        }
        Codec::F16 if src_dtype == TensorDtype::F16 => (TensorDtype::F16, raw.to_vec()),
        Codec::F16 => (
            TensorDtype::F16,
            convert::encode_f16(&convert::to_f32(&t.dtype, raw)?),
        ),
        Codec::Q(q) => {
            ensure!(t.shape.len() == 2, "{}: a quantized codec needs 2-D", t.name);
            let v = convert::to_f32(&t.dtype, raw)?;
            convert::quantize_2d(q, &v, t.shape[0], t.shape[1])
        }
    })
}

/// 16-bit keep for weights that only feed host precomputes: bf16 sources
/// stay bf16 (exact), wider sources go to f16 unless a value overflows it.
fn keep16(t: &StTensor, raw: &[u8]) -> anyhow::Result<Codec> {
    Ok(match t.dtype.as_str() {
        "BF16" | "F16" => Codec::Raw,
        _ => {
            let v = convert::to_f32(&t.dtype, raw)?;
            if v.iter().all(|x| x.abs() < 65000.0) {
                Codec::F16
            } else {
                Codec::F32
            }
        }
    })
}

pub(crate) struct PackOpts {
    pub dit: Codec,
    pub te: Codec,
    /// Dev: keep only the first N main DiT layers (config says so).
    pub layers: Option<usize>,
    /// "turbo" | "base" | None = from the scheduler shift.
    pub variant: Option<String>,
    pub source_sha: bool,
}

/// Is `root` a diffusers Z-Image pipeline directory?
pub(crate) fn is_zimage_root(root: &Path) -> bool {
    std::fs::read(root.join("model_index.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["_class_name"] == "ZImagePipeline")
}

pub(crate) fn pack(root: &Path, out: &str, o: &PackOpts) -> anyhow::Result<()> {
    ensure!(
        is_zimage_root(root),
        "{}: not a diffusers ZImagePipeline directory",
        root.display()
    );
    let t_all = std::time::Instant::now();
    let read = |rel: &str| -> anyhow::Result<Vec<u8>> {
        std::fs::read(root.join(rel)).with_context(|| format!("{}/{rel}", root.display()))
    };
    let dit_cfg_raw = read("transformer/config.json")?;
    let mut dit_cfg: serde_json::Value = serde_json::from_slice(&dit_cfg_raw)?;
    ensure!(
        dit_cfg["_class_name"] == "ZImageTransformer2DModel",
        "transformer is not ZImageTransformer2DModel"
    );
    let n_layers = dit_cfg["n_layers"].as_u64().context("n_layers")? as usize;
    let keep_layers = o.layers.unwrap_or(n_layers).min(n_layers);
    if keep_layers != n_layers {
        dit_cfg["n_layers"] = serde_json::json!(keep_layers);
        dit_cfg["cortiq_dev_truncated_from"] = serde_json::json!(n_layers);
    }
    let te_cfg_raw = read("text_encoder/config.json")?;
    let mut te_cfg: serde_json::Value = serde_json::from_slice(&te_cfg_raw)?;
    let te_layers_src = te_cfg["num_hidden_layers"].as_u64().context("num_hidden_layers")? as usize;
    ensure!(te_layers_src >= 2, "text encoder too shallow");
    // hidden_states[-2] = the raw output of the second-to-last layer.
    let te_keep = te_layers_src - 1;
    te_cfg["num_hidden_layers"] = serde_json::json!(te_keep);
    te_cfg["final_norm"] = serde_json::json!(false);
    te_cfg["cortiq_tap"] = serde_json::json!("hidden_states[-2] of the source (no final norm)");
    let vae_cfg_raw = read("vae/config.json")?;
    let sched_raw = read("scheduler/scheduler_config.json")?;
    let sched: serde_json::Value = serde_json::from_slice(&sched_raw)?;
    let shift = sched["shift"].as_f64().unwrap_or(3.0);
    ensure!(
        !sched["use_dynamic_shifting"].as_bool().unwrap_or(false),
        "dynamic shifting is not supported (the Z-Image schedulers use a static shift)"
    );
    let variant = o
        .variant
        .clone()
        .unwrap_or_else(|| if shift <= 3.5 { "turbo" } else { "base" }.into());
    let defaults = match variant.as_str() {
        // Model card + diffusers docs: 8 DiT forwards, guidance 0.
        "turbo" => serde_json::json!({
            "variant": "turbo", "steps": 8, "guidance": 0.0, "shift": shift,
            "height": 1024, "width": 1024, "cfg_normalization": 0.0, "cfg_truncation": 1.0,
            "negative_prompt": "", "max_sequence_length": 512,
            "template": "qwen3-chat-user-assistant (enable_thinking=True)", "vae_scale": 8
        }),
        // Model card: guidance 3-5, 28-50 steps, negative prompts, CFG.
        "base" => serde_json::json!({
            "variant": "base", "steps": 28, "guidance": 4.0, "shift": shift,
            "height": 1024, "width": 1024, "cfg_normalization": 0.0, "cfg_truncation": 1.0,
            "negative_prompt": "", "max_sequence_length": 512,
            "template": "qwen3-chat-user-assistant (enable_thinking=True)", "vae_scale": 8
        }),
        v => return Err(anyhow!("--variant {v}: expected turbo or base")),
    };
    let vocab = read("tokenizer/tokenizer.json")?;

    let dit_files = shard_files(
        &root.join("transformer"),
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
    )?;
    let te_files = shard_files(
        &root.join("text_encoder"),
        "model.safetensors.index.json",
        "model.safetensors",
    )?;
    let vae_files = shard_files(
        &root.join("vae"),
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
    )?;

    // Source provenance: sha256 of every weight shard (hashed in parallel).
    let mut source_sha = serde_json::Map::new();
    if o.source_sha {
        let files: Vec<PathBuf> = dit_files
            .iter()
            .chain(&te_files)
            .chain(&vae_files)
            .cloned()
            .collect();
        let hashes: Vec<anyhow::Result<String>> = std::thread::scope(|s| {
            let hs: Vec<_> = files.iter().map(|f| s.spawn(|| sha256_file(f))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (f, h) in files.iter().zip(hashes) {
            let rel = f.strip_prefix(root).unwrap_or(f).display().to_string();
            source_sha.insert(rel, serde_json::json!(h?));
        }
    }

    // ── the tensor plan: (name out, source file, tensor, codec rule) ──
    #[derive(Clone, Copy)]
    enum Rule {
        Fixed(Codec),
        Keep16,
    }
    struct Item {
        out: String,
        file: usize,
        t: usize,
        rule: Rule,
    }
    let mut files: Vec<(PathBuf, Vec<StTensor>, usize)> = Vec::new();
    let mut items: Vec<Item> = Vec::new();
    let is_dit_proj = |n: &str| {
        (n.contains(".attention.to_") && n.ends_with(".weight") && !n.contains("norm"))
            || n.contains(".feed_forward.w")
            || n == "cap_embedder.1.weight"
    };
    let layer_of = |n: &str| -> Option<usize> {
        n.strip_prefix("layers.")
            .and_then(|r| r.split('.').next())
            .and_then(|x| x.parse().ok())
    };
    for f in &dit_files {
        let (ts, size) = st_header(f)?;
        let fi = files.len();
        for (ti, t) in ts.iter().enumerate() {
            if layer_of(&t.name).is_some_and(|l| l >= keep_layers) {
                continue;
            }
            if t.name.starts_with("siglip") {
                continue; // omni-mode only
            }
            let rule = if t.name.ends_with("adaLN_modulation.0.weight") {
                Rule::Keep16
            } else if t.shape.len() == 2 && is_dit_proj(&t.name) {
                Rule::Fixed(o.dit)
            } else {
                Rule::Fixed(Codec::F32)
            };
            items.push(Item {
                out: format!("dit.{}", t.name),
                file: fi,
                t: ti,
                rule,
            });
        }
        files.push((f.clone(), ts, size));
    }
    let n_dit = items.len();
    for f in &te_files {
        let (ts, size) = st_header(f)?;
        let fi = files.len();
        for (ti, t) in ts.iter().enumerate() {
            let n = t.name.strip_prefix("model.").unwrap_or(&t.name);
            if n == "lm_head.weight" || n == "norm.weight" {
                continue;
            }
            if let Some(l) = n
                .strip_prefix("layers.")
                .and_then(|r| r.split('.').next())
                .and_then(|x| x.parse::<usize>().ok())
            {
                if l >= te_keep {
                    continue;
                }
            }
            let rule = if n == "embed_tokens.weight" {
                // A per-token lookup table: never 4-bit (LTX lesson).
                Rule::Fixed(match o.te {
                    Codec::Raw | Codec::F32 | Codec::Bf16 | Codec::F16 => o.te,
                    Codec::Q(_) => Codec::Q(convert::Quant::Q8Row),
                })
            } else if t.shape.len() == 2 && n.ends_with("_proj.weight") {
                Rule::Fixed(o.te)
            } else {
                Rule::Fixed(Codec::F32)
            };
            items.push(Item {
                out: format!("te.{n}"),
                file: fi,
                t: ti,
                rule,
            });
        }
        files.push((f.clone(), ts, size));
    }
    let n_te = items.len() - n_dit;
    for f in &vae_files {
        let (ts, size) = st_header(f)?;
        let fi = files.len();
        for (ti, t) in ts.iter().enumerate() {
            if !t.name.starts_with("decoder.") {
                continue; // text-to-image needs the decoder only
            }
            items.push(Item {
                out: format!("vae.{}", t.name),
                file: fi,
                t: ti,
                rule: Rule::Fixed(Codec::F32),
            });
        }
        files.push((f.clone(), ts, size));
    }
    ensure!(n_dit > 0 && n_te > 0, "no transformer or text-encoder tensors found");
    let extras: Vec<(&str, Vec<u8>)> = vec![
        ("dit.config_json", serde_json::to_vec_pretty(&dit_cfg)?),
        ("te.config_json", serde_json::to_vec_pretty(&te_cfg)?),
        ("vae.config_json", vae_cfg_raw.clone()),
        ("zimage.config_json", serde_json::to_vec_pretty(&defaults)?),
        ("zimage.scheduler_json", sched_raw.clone()),
    ];
    let count = items.len() + extras.len();
    eprintln!(
        "z-image pack ({variant}): {n_dit} dit ({}), {n_te} te ({}), {} vae tensors",
        codec_name(o.dit),
        codec_name(o.te),
        items.len() - n_dit - n_te
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
            // Map only the shards this window reads; drop the rest.
            let need: std::collections::BTreeSet<usize> =
                chunk.iter().map(|it| it.file).collect();
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
                        s.spawn(move || {
                            let codec = match it.rule {
                                Rule::Fixed(c) => c,
                                Rule::Keep16 => keep16(t, raw)?,
                            };
                            encode(t, raw, codec)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for (it, enc) in chunk.iter().zip(encoded) {
                let (dt, bytes) = enc.with_context(|| it.out.clone())?;
                let t = &files[it.file].1[it.t];
                *dtype_bytes.entry(format!("{:?}", dt)).or_default() += bytes.len() as u64;
                w.push(&it.out, dt, &t.shape, &bytes)?;
                done += 1;
            }
            if done % 64 < chunk.len() || done == items.len() {
                eprintln!(
                    "  {done}/{} tensors ({:.0}s)",
                    items.len(),
                    t_all.elapsed().as_secs_f64()
                );
            }
        }
        drop(maps);
        let arch: ModelArch = serde_json::from_value(serde_json::json!({
            "arch_name": "z_image",
            "hidden_size": te_cfg["hidden_size"],
            "intermediate_size": te_cfg["intermediate_size"],
            "num_layers": te_keep,
            "num_attention_heads": te_cfg["num_attention_heads"],
            "num_kv_heads": te_cfg["num_key_value_heads"],
            "head_dim": te_cfg["head_dim"],
            "vocab_size": te_cfg["vocab_size"],
            "layer_types": vec!["FullAttention"; te_keep],
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
                "pipeline": if variant == "turbo" { "z-image-turbo" } else { "z-image" },
                "components": {"te": "qwen3-4b encoder, layers 0..34 (hidden_states[-2])",
                               "dit": "ZImageTransformer2DModel", "vae": "flux vae decoder"},
                "packed_from": root.display().to_string(),
                "dit_codec": codec_name(o.dit),
                "te_codec": codec_name(o.te),
                "dit_layers": keep_layers,
                "defaults": defaults,
                "source_sha256": source_sha,
                "tensor_name_policy": "diffusers names under dit./te./vae.",
            })),
            tokenizer_config: None,
            section_hashes: None,
            skills: vec![],
            shard: None,
            calibration: None,
            routing: None,
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
    let sha = sha256_file(&out_path)?;
    let fname = out_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(out)
        .to_string();
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
