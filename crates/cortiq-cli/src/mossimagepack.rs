//! Pack the official MOSS-Image-Base diffusers pipeline into one CMF.
//!
//! MOSS is intentionally a separate image family rather than an alias for
//! Z-Image: it combines a Qwen3.5 GatedDeltaNet text encoder, a 42-block
//! joint MOSS DiT, and an AutoencoderKLFlux2 decoder.  The source components
//! are streamed one safetensors tensor at a time, so packing never expands a
//! whole 25+ GiB checkpoint in RAM.
//!
//! The output layout is stable and purpose-built for the native runtime:
//! - `model.*`: canonical Qwen3.5 language-tower names for the source Base
//!   string-prompt text-to-image route,
//! - `moss.*`: source MOSS transformer suffixes verbatim,
//! - `vae.*`: Flux2 decoder, post-quant convolution and BN statistics,
//! - `moss.*_json`: source component/scheduler/tokenizer metadata.
//!
//! The source Base `MOSSImagePipeline.__call__` samples fresh latents and only
//! calls `vae.decode`, so its supported string-prompt text-to-image route never
//! reaches `encoder.*` or `quant_conv.*`.  It likewise supplies no image pixels
//! to Qwen3.5, so that route cannot invoke Qwen's visual tower.  The packer
//! deliberately omits those unreachable tensors to reduce residency and file
//! size; this is not a claim that they are useless to a different multimodal or
//! future reference-image pipeline.

use crate::{convert, gguf, zimagepack};
use anyhow::{Context, anyhow, ensure};
use cortiq_core::format::{CmfHeader, CmfStreamWriter, TokenizerBundle};
use cortiq_core::types::{ModelArch, TensorDtype};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Default codec for the MOSS DiT's projection matrices.  `q8` maps to the
/// two-field q8_2f representation, which keeps the first release quality
/// oriented.  Pass `--quant q4tp` for the lower-footprint profile and
/// `--dit-keep layers.N,...` for an explicit, provenance-recorded q8 lift.
pub(crate) const DEFAULT_DIT_CODEC: &str = "q8";
/// Default codec for the dense Qwen3.5 text encoder's projections.
pub(crate) const DEFAULT_TE_CODEC: &str = "q8";

/// MOSS's text encoder is a normal Qwen3.5 text stack wrapped under
/// `language_model.`.  Reuse the converter's canonical naming policy, but
/// refuse any accidental vision or language head payload instead of hiding it.
fn text_name(source: &str) -> Option<String> {
    let out = convert::canon_name(source)?;
    if out == "lm_head.weight" || out == "lm_head.bias" {
        return None;
    }
    Some(out)
}

/// Is `root` an official/local MOSS image pipeline snapshot?
pub(crate) fn is_moss_image_root(root: &Path) -> bool {
    std::fs::read(root.join("model_index.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["_class_name"] == "MOSSImagePipeline")
}

#[derive(Clone)]
pub(crate) struct PackOpts {
    pub dit: zimagepack::Codec,
    /// MOSS DiT matrices intentionally lifted to q8_2f over `dit`.  This is
    /// a transparent mixed-precision quality profile, not a name for an
    /// unmeasured q4 refit.
    pub dit_keep: Vec<String>,
    pub te: zimagepack::Codec,
    /// Optional embedding codec.  Quantized text models default to q8_row;
    /// a 4-bit embedding table is intentionally never selected implicitly.
    pub te_embed: Option<zimagepack::Codec>,
    /// Qwen text projections to retain at source precision.  An entry may be
    /// a full `layers.N.foo` prefix or a suffix such as `down_proj`.
    pub te_keep: Vec<String>,
    /// Hash all source weight shards into provenance.  Disabling this is only
    /// a speed trade-off; the source revision/path remains recorded.
    pub source_sha: bool,
}

#[derive(Clone, Copy)]
enum Rule {
    Fixed(zimagepack::Codec),
    /// Source is float; write f32 controls/norms/biases for deterministic
    /// host math and to avoid quantizing scale/normalization vectors.
    F32,
}

struct Item {
    out: String,
    file: usize,
    tensor: usize,
    rule: Rule,
}

fn add_item(
    items: &mut Vec<Item>,
    output_names: &mut BTreeSet<String>,
    out: String,
    file: usize,
    tensor: usize,
    rule: Rule,
) -> anyhow::Result<()> {
    ensure!(
        output_names.insert(out.clone()),
        "duplicate output tensor {out}"
    );
    items.push(Item {
        out,
        file,
        tensor,
        rule,
    });
    Ok(())
}

fn read(root: &Path, rel: &str) -> anyhow::Result<Vec<u8>> {
    std::fs::read(root.join(rel)).with_context(|| format!("{}/{rel}", root.display()))
}

fn read_optional(root: &Path, rel: &str) -> anyhow::Result<Option<Vec<u8>>> {
    match std::fs::read(root.join(rel)) {
        Ok(v) => Ok(Some(v)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("{}/{rel}", root.display())),
    }
}

fn te_keep_match(name: &str, key: &str) -> bool {
    if key.starts_with("layers.") {
        name.starts_with(&format!("{key}.")) || name == key
    } else {
        name.contains(&format!(".{key}."))
    }
}

/// Does a source MOSS DiT matrix match a `--dit-keep` selector?  A whole
/// `layers.N` keeps that block's q/k/v/o and MLP matrices at q8_2f; a bare
/// `q`, `o`, `ffn_up_gate`, or `ffn_down` selects that projection in every
/// layer; `layers.N.<projection>` selects one.  All source matrices live
/// under `layers.N.proj.*`, so matching is intentionally explicit rather than
/// accepting arbitrary substring selectors.
fn dit_keep_match(name: &str, key: &str) -> bool {
    let Some(rest) = name.strip_prefix("layers.") else {
        return false;
    };
    let Some((layer, projection)) = rest.split_once('.') else {
        return false;
    };
    let projection = projection
        .strip_prefix("proj.")
        .unwrap_or(projection)
        .trim_end_matches(".weight");
    let key = key.strip_prefix("layers.").unwrap_or(key);
    match key.split_once('.') {
        None => key == layer || key == projection,
        Some((want_layer, want_projection)) => want_layer == layer && want_projection == projection,
    }
}

fn text_matrix_rule(name: &str, t: &zimagepack::StTensor, o: &PackOpts) -> Rule {
    if name == "model.embed_tokens.weight" {
        return Rule::Fixed(match (o.te_embed, o.te) {
            (Some(c), _) => c,
            (
                None,
                zimagepack::Codec::Raw
                | zimagepack::Codec::F32
                | zimagepack::Codec::Bf16
                | zimagepack::Codec::F16,
            ) => o.te,
            (None, zimagepack::Codec::Q(_)) => zimagepack::Codec::Q(convert::Quant::Q8Row),
        });
    }
    if t.shape.len() == 2 && name.ends_with(".weight") {
        if o.te_keep.iter().any(|key| te_keep_match(name, key)) {
            return Rule::Fixed(zimagepack::Codec::Raw);
        }
        return Rule::Fixed(o.te);
    }
    Rule::F32
}

/// DiT matrices that take a sequence/latent activation.  Timesteps and the
/// adaptive modulation networks have only one or a few rows per pass, so
/// their tiny weights stay 16-bit/f32 rather than paying a q4/q8 error at a
/// host-side scalar boundary.
fn moss_matrix_rule(name: &str, t: &zimagepack::StTensor, o: &PackOpts) -> Rule {
    if t.shape.len() != 2 || !name.ends_with(".weight") {
        return Rule::F32;
    }
    let control = name.starts_with("time_embedding.")
        || name.starts_with("shared_block_modulation.")
        || name.starts_with("video_head_modulation.");
    if control {
        Rule::Fixed(zimagepack::Codec::Raw)
    } else if o.dit_keep.iter().any(|key| dit_keep_match(name, key)) {
        Rule::Fixed(zimagepack::Codec::Q(convert::Quant::Q8_2f))
    } else {
        Rule::Fixed(o.dit)
    }
}

/// The source Base text-to-image `__call__` samples a fresh unpatched
/// 32-channel latent then invokes `AutoencoderKLFlux2.decode()`, which uses
/// only post_quant_conv + decoder.  MOSS normalizes with the saved BatchNorm
/// running statistics before that call.  This predicate is intentionally
/// limited to that supported decode-only route.
fn keep_vae_tensor(name: &str) -> bool {
    name.starts_with("decoder.")
        || name.starts_with("post_quant_conv.")
        || name == "bn.running_mean"
        || name == "bn.running_var"
}

fn build_arch(te_outer: &serde_json::Value) -> anyhow::Result<ModelArch> {
    let te = te_outer
        .get("text_config")
        .filter(|v| v.is_object())
        .unwrap_or(te_outer);
    let get_usize = |key: &str| -> anyhow::Result<usize> {
        te.get(key)
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| anyhow!("text_encoder config: missing integer {key}"))
    };
    let model_type = te
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    ensure!(
        model_type.to_ascii_lowercase().starts_with("qwen3_5"),
        "text_encoder model_type {model_type:?} is not Qwen3.5"
    );
    let layer_types = te
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .context("text_encoder config: missing layer_types")?
        .iter()
        .map(|v| {
            Ok(match v.as_str() {
                Some("linear_attention") => "LinearAttention",
                Some("full_attention") => "FullAttention",
                other => return Err(anyhow!("unsupported MOSS Qwen3.5 layer type {other:?}")),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let n_layers = get_usize("num_hidden_layers")?;
    ensure!(
        layer_types.len() == n_layers,
        "layer_types length != num_hidden_layers"
    );
    let linear_heads = get_usize("linear_num_value_heads")?;
    let linear_v_dim = get_usize("linear_value_head_dim")?;
    serde_json::from_value(serde_json::json!({
        "arch_name": "moss_image",
        "hidden_size": get_usize("hidden_size")?,
        "intermediate_size": get_usize("intermediate_size")?,
        "num_layers": n_layers,
        "num_attention_heads": get_usize("num_attention_heads")?,
        "num_kv_heads": get_usize("num_key_value_heads")?,
        "head_dim": get_usize("head_dim")?,
        "vocab_size": get_usize("vocab_size")?,
        "layer_types": layer_types,
        "rms_norm_eps": te.get("rms_norm_eps").and_then(serde_json::Value::as_f64).unwrap_or(1e-6),
        // HF Qwen3.5's RMSNorm uses the zero-centered/Gemma form.
        "norm_style": "gemma",
        "rope_theta": te.get("rope_parameters").and_then(|r| r.get("rope_theta")).and_then(serde_json::Value::as_f64).unwrap_or(10_000_000.0),
        "partial_rotary_factor": te.get("partial_rotary_factor").and_then(serde_json::Value::as_f64).unwrap_or(0.25),
        "tie_word_embeddings": false,
        "hidden_act": te.get("hidden_act").and_then(serde_json::Value::as_str).unwrap_or("silu"),
        "max_position_embeddings": get_usize("max_position_embeddings")?,
        "linear_core": {
            "kind": "gated_delta_net", "num_heads": linear_heads,
            "nphase": null, "value_head_dim": linear_v_dim, "phase_delta_layers": null
        },
        "linear_conv_kernel_dim": get_usize("linear_conv_kernel_dim")?,
        "linear_num_key_heads": get_usize("linear_num_key_heads")?,
        "linear_num_value_heads": linear_heads,
        "linear_key_head_dim": get_usize("linear_key_head_dim")?,
        "linear_value_head_dim": linear_v_dim
    }))
    .context("construct MOSS Qwen3.5 architecture")
}

/// Pack an already-downloaded MOSS source snapshot into a single CMF.
pub(crate) fn pack(root: &Path, out: &str, o: &PackOpts) -> anyhow::Result<()> {
    ensure!(
        is_moss_image_root(root),
        "{}: not a diffusers MOSSImagePipeline directory",
        root.display()
    );
    let began = std::time::Instant::now();
    let model_index: serde_json::Value = serde_json::from_slice(&read(root, "model_index.json")?)?;
    ensure!(
        model_index["_class_name"] == "MOSSImagePipeline",
        "wrong pipeline class"
    );

    let moss_cfg_raw = read(root, "transformer/config.json")?;
    let moss_cfg: serde_json::Value = serde_json::from_slice(&moss_cfg_raw)?;
    ensure!(
        moss_cfg["_class_name"] == "MOSSImageTransformer",
        "transformer is not MOSSImageTransformer"
    );
    for key in [
        "dim",
        "ffn_dim",
        "freq_dim",
        "latent_channels",
        "num_heads",
        "num_layers",
        "text_dim",
    ] {
        ensure!(
            moss_cfg
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .is_some(),
            "MOSS transformer config missing {key}"
        );
    }

    let te_cfg_raw = read(root, "text_encoder/config.json")?;
    let te_cfg: serde_json::Value = serde_json::from_slice(&te_cfg_raw)?;
    let arch = build_arch(&te_cfg)?;

    let vae_cfg_raw = read(root, "vae/config.json")?;
    let vae_cfg: serde_json::Value = serde_json::from_slice(&vae_cfg_raw)?;
    ensure!(
        vae_cfg["_class_name"] == "AutoencoderKLFlux2",
        "VAE is not AutoencoderKLFlux2"
    );
    ensure!(
        vae_cfg["latent_channels"] == 32,
        "expected Flux2 latent_channels=32"
    );
    ensure!(
        vae_cfg["patch_size"] == serde_json::json!([2, 2]),
        "expected Flux2 patch_size=[2,2]"
    );

    let scheduler_raw = read(root, "scheduler/scheduler_config.json")?;
    let scheduler: serde_json::Value = serde_json::from_slice(&scheduler_raw)?;
    ensure!(
        scheduler["_class_name"] == "MOSSImageScheduler",
        "scheduler is not MOSSImageScheduler"
    );
    ensure!(
        !scheduler["use_dynamic_shifting"].as_bool().unwrap_or(false),
        "dynamic MOSS shifting is not supported by the portable recipe"
    );

    let vocab = read(root, "tokenizer/tokenizer.json")?;
    let tokenizer_cfg_raw = read(root, "tokenizer/tokenizer_config.json")?;
    let tokenizer_cfg: serde_json::Value = serde_json::from_slice(&tokenizer_cfg_raw)?;
    let chat_template = read_optional(root, "tokenizer/chat_template.jinja")?
        .map(|b| String::from_utf8(b).context("tokenizer chat_template.jinja is not UTF-8"))
        .transpose()?
        .or_else(|| {
            tokenizer_cfg
                .get("chat_template")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .context("MOSS tokenizer has no chat_template.jinja or tokenizer_config.chat_template")?;
    ensure!(
        chat_template.contains("<|im_start|>assistant"),
        "MOSS tokenizer chat template lacks its assistant generation suffix"
    );

    let dit_files = zimagepack::shard_files(
        &root.join("transformer"),
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
    )?;
    let te_files = zimagepack::shard_files(
        &root.join("text_encoder"),
        "model.safetensors.index.json",
        "model.safetensors",
    )?;
    let vae_files = zimagepack::shard_files(
        &root.join("vae"),
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
    )?;

    let mut source_sha = serde_json::Map::new();
    if o.source_sha {
        let source_files: Vec<PathBuf> = dit_files
            .iter()
            .chain(&te_files)
            .chain(&vae_files)
            .cloned()
            .collect();
        let hashes: Vec<anyhow::Result<String>> = std::thread::scope(|scope| {
            let jobs: Vec<_> = source_files
                .iter()
                .map(|path| scope.spawn(|| zimagepack::sha256_file(path)))
                .collect();
            jobs.into_iter().map(|j| j.join().unwrap()).collect()
        });
        for (path, hash) in source_files.iter().zip(hashes) {
            source_sha.insert(
                path.strip_prefix(root)
                    .unwrap_or(path)
                    .display()
                    .to_string(),
                serde_json::json!(hash?),
            );
        }
    }

    // A file descriptor/header is retained, while mmap mappings are opened
    // only for the current small encoding window.  This caps resident source
    // bytes even on four multi-gigabyte Qwen shards.
    let mut files: Vec<(PathBuf, Vec<zimagepack::StTensor>, usize)> = Vec::new();
    let mut items = Vec::<Item>::new();
    let mut output_names = BTreeSet::<String>::new();
    let before_dit = items.len();
    for path in &dit_files {
        let (tensors, size) = zimagepack::st_header(path)?;
        let file = files.len();
        for (tensor, source) in tensors.iter().enumerate() {
            // The one source rope tensor is a non-matrix control and remains
            // f32.  Nothing in MOSS's DiT is an unused visual tower.
            let rule = moss_matrix_rule(&source.name, source, o);
            add_item(
                &mut items,
                &mut output_names,
                format!("moss.{}", source.name),
                file,
                tensor,
                rule,
            )?;
        }
        files.push((path.clone(), tensors, size));
    }
    let n_moss = items.len() - before_dit;

    let before_te = items.len();
    for path in &te_files {
        let (tensors, size) = zimagepack::st_header(path)?;
        let file = files.len();
        for (tensor, source) in tensors.iter().enumerate() {
            let Some(out) = text_name(&source.name) else {
                continue;
            };
            let rule = text_matrix_rule(&out, source, o);
            add_item(&mut items, &mut output_names, out, file, tensor, rule)?;
        }
        files.push((path.clone(), tensors, size));
    }
    let n_te = items.len() - before_te;

    let before_vae = items.len();
    for path in &vae_files {
        let (tensors, size) = zimagepack::st_header(path)?;
        let file = files.len();
        for (tensor, source) in tensors.iter().enumerate() {
            if !keep_vae_tensor(&source.name) {
                continue;
            }
            add_item(
                &mut items,
                &mut output_names,
                format!("vae.{}", source.name),
                file,
                tensor,
                Rule::F32,
            )?;
        }
        files.push((path.clone(), tensors, size));
    }
    let n_vae = items.len() - before_vae;
    ensure!(
        n_moss > 0 && n_te > 0 && n_vae > 0,
        "MOSS pack has an empty component"
    );
    for required in [
        "moss.video_embedder.weight",
        "moss.final_linear_video.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "vae.bn.running_mean",
        "vae.bn.running_var",
        "vae.post_quant_conv.weight",
        "vae.decoder.conv_in.weight",
    ] {
        ensure!(
            output_names.contains(required),
            "source missing required MOSS tensor {required}"
        );
    }

    let defaults = serde_json::json!({
        "height": 1024,
        "width": 1024,
        "steps": 50,
        "cfg": 3.0,
        "shift": scheduler["shift"].as_f64().unwrap_or(5.0),
        "negative_prompt": "",
        "max_sequence_length": 4096,
        "uncond_skip_layers": [22],
        "template": "Qwen3.5 chat template, add_generation_prompt=true, enable_thinking=true",
        "latent_channels_before_unpatch": 128,
        "latent_channels": 32,
        "vae_scale": 16
    });
    let extras: Vec<(&str, Vec<u8>)> = vec![
        ("moss.config_json", moss_cfg_raw),
        ("moss.scheduler_json", scheduler_raw),
        ("moss.defaults_json", serde_json::to_vec_pretty(&defaults)?),
        ("te.config_json", te_cfg_raw),
        ("vae.config_json", vae_cfg_raw),
        ("moss.tokenizer_config_json", tokenizer_cfg_raw),
        (
            "moss.chat_template_jinja",
            chat_template.as_bytes().to_vec(),
        ),
    ];
    for (name, _) in &extras {
        ensure!(
            output_names.insert((*name).into()),
            "extra clashes with a tensor: {name}"
        );
    }
    let count = items.len() + extras.len();
    eprintln!(
        "moss-image pack: {n_moss} DiT ({}; {} q8-kept selector(s)) + {n_te} Qwen3.5 text ({}) + {n_vae} Flux2 decode tensors; visual tower and VAE encode tensors are unreachable in source Base string-prompt text-to-image and omitted",
        zimagepack::codec_name(o.dit),
        o.dit_keep.len(),
        zimagepack::codec_name(o.te),
    );

    let out_path = PathBuf::from(out);
    let temp = gguf::qwen_image_temp_path(&out_path)?;
    let threads = std::env::var("CMF_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8)
        .clamp(1, 32);
    let mut dtype_bytes = BTreeMap::<String, u64>::new();
    let result = (|| -> anyhow::Result<()> {
        let mut writer = CmfStreamWriter::new(&temp, CmfStreamWriter::head_reserve_for(count, 96))?;
        for (name, bytes) in &extras {
            writer.push(name, TensorDtype::U8, &[bytes.len()], bytes)?;
        }
        let mut maps: Vec<Option<memmap2::Mmap>> = (0..files.len()).map(|_| None).collect();
        let mut done = 0usize;
        for chunk in items.chunks(threads.max(2)) {
            let needed: BTreeSet<usize> = chunk.iter().map(|i| i.file).collect();
            for (index, map) in maps.iter_mut().enumerate() {
                if !needed.contains(&index) {
                    *map = None;
                }
            }
            for item in chunk {
                if maps[item.file].is_none() {
                    let (path, _, size) = &files[item.file];
                    let file = std::fs::File::open(path)?;
                    let map = unsafe { memmap2::Mmap::map(&file)? };
                    ensure!(
                        map.len() == *size,
                        "{} changed while packing",
                        path.display()
                    );
                    maps[item.file] = Some(map);
                }
            }
            let encoded: Vec<anyhow::Result<(TensorDtype, Vec<u8>)>> =
                std::thread::scope(|scope| {
                    let jobs: Vec<_> = chunk
                        .iter()
                        .map(|item| {
                            let source = &files[item.file].1[item.tensor];
                            let raw =
                                &maps[item.file].as_ref().expect("mapped")[source.range.clone()];
                            scope.spawn(move || match item.rule {
                                Rule::Fixed(codec) => zimagepack::encode(source, raw, codec),
                                Rule::F32 => {
                                    zimagepack::encode(source, raw, zimagepack::Codec::F32)
                                }
                            })
                        })
                        .collect();
                    jobs.into_iter().map(|job| job.join().unwrap()).collect()
                });
            for (item, encoded) in chunk.iter().zip(encoded) {
                let (dtype, bytes) = encoded.with_context(|| item.out.clone())?;
                let source = &files[item.file].1[item.tensor];
                *dtype_bytes.entry(format!("{:?}", dtype)).or_default() += bytes.len() as u64;
                writer.push(&item.out, dtype, &source.shape, &bytes)?;
                done += 1;
            }
            if done % 64 < chunk.len() || done == items.len() {
                eprintln!(
                    "  {done}/{} tensors ({:.0}s)",
                    items.len(),
                    began.elapsed().as_secs_f64()
                );
            }
        }
        drop(maps);
        let quant_type = match o.dit {
            zimagepack::Codec::Q(q) => gguf::quant_type_for(q),
            zimagepack::Codec::Raw | zimagepack::Codec::F16 => cortiq_core::types::QuantType::F16,
            zimagepack::Codec::Bf16 => cortiq_core::types::QuantType::BF16,
            zimagepack::Codec::F32 => cortiq_core::types::QuantType::F32,
        };
        let eos = te_cfg
            .get("text_config")
            .unwrap_or(&te_cfg)
            .get("eos_token_id")
            .and_then(serde_json::Value::as_u64)
            .map(|id| id as u32)
            .into_iter()
            .collect();
        let header = CmfHeader {
            format: "cmf".into(),
            version: cortiq_core::CMF_VERSION,
            arch,
            quant_type,
            provenance: Some(serde_json::json!({
                "tool": "cortiq imagine-pack",
                "pipeline": "moss-image-base",
                "source_pipeline_class": "MOSSImagePipeline",
                "components": {
                    "text_encoder": "Qwen3.5 language tower; visual tower is unreachable in source Base string-prompt text-to-image",
                    "transformer": "MOSSImageTransformer",
                    "vae": "AutoencoderKLFlux2 decode path: decoder + post_quant_conv + BN statistics; encode-only tensors are unreachable in source Base text-to-image"
                },
                "packed_from": root.display().to_string(),
                "moss_codec": zimagepack::codec_name(o.dit),
                "moss_dit_keep_q8_2f": o.dit_keep,
                "text_encoder_codec": zimagepack::codec_name(o.te),
                "text_embed_codec": o.te_embed.map(zimagepack::codec_name),
                "text_keep": o.te_keep,
                "recipe": defaults,
                "tensor_name_policy": "MOSS transformer source suffixes under moss.; Flux2 retained decode tensors under vae.; language_model.* canonically mapped to model.*",
                "source_sha256": source_sha,
            })),
            tokenizer_config: Some(TokenizerBundle {
                chat_template: Some(chat_template),
                eos_token_ids: eos,
                bos_token_id: None,
                pad_token_id: None,
            }),
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
        writer.finish(&header, None, Some(&vocab))?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temp);
        return Err(error);
    }
    std::fs::rename(&temp, &out_path)?;
    gguf::qwen_image_sync_parent(&out_path)?;
    let sha = zimagepack::sha256_file(&out_path)?;
    let filename = out_path.file_name().and_then(|s| s.to_str()).unwrap_or(out);
    std::fs::write(format!("{out}.sha256"), format!("{sha}  {filename}\n"))?;
    let size = std::fs::metadata(&out_path)?.len();
    println!(
        "{out}: {count} tensors, {:.3} GB ({:.3} GiB), sha256 {sha}, {:.0}s",
        size as f64 / 1e9,
        size as f64 / (1u64 << 30) as f64,
        began.elapsed().as_secs_f64(),
    );
    for (dtype, bytes) in dtype_bytes {
        println!("  {dtype}: {:.3} GB", bytes as f64 / 1e9);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_only_the_text_tower() {
        assert_eq!(
            text_name("language_model.layers.3.mlp.down_proj.weight").as_deref(),
            Some("model.layers.3.mlp.down_proj.weight")
        );
        assert_eq!(text_name("visual.blocks.0.attn.qkv.weight"), None);
    }

    #[test]
    fn dit_keep_matches_whole_blocks_and_explicit_moss_projections() {
        let q = "layers.36.proj.q.weight";
        let up = "layers.36.proj.ffn_up_gate.weight";
        assert!(dit_keep_match(q, "layers.36"));
        assert!(!dit_keep_match(q, "layers.35"));
        assert!(dit_keep_match(q, "q"));
        assert!(dit_keep_match(q, "layers.36.q"));
        assert!(!dit_keep_match(q, "layers.36.o"));
        assert!(dit_keep_match(up, "ffn_up_gate"));
        assert!(dit_keep_match(up, "layers.36.ffn_up_gate"));
        assert!(!dit_keep_match("video_embedder.weight", "layers.36"));
    }

    #[test]
    fn keeps_exact_decode_subset() {
        assert!(keep_vae_tensor("decoder.conv_in.weight"));
        assert!(keep_vae_tensor("post_quant_conv.weight"));
        assert!(keep_vae_tensor("bn.running_var"));
        assert!(!keep_vae_tensor("bn.num_batches_tracked"));
        assert!(!keep_vae_tensor("encoder.conv_in.weight"));
        assert!(!keep_vae_tensor("quant_conv.weight"));
    }

    /// The normal-CI policy gate for the deliberately narrow Base package.
    /// The source-pinned, opt-in integration test below checks the same policy
    /// against the actual snapshot; this one keeps accidental selection-policy
    /// drift visible even in a weight-free checkout.
    #[test]
    fn base_text_to_image_payload_contract() {
        // MOSSImagePipeline.encode_prompt passes tokenizer text tensors to
        // Qwen, while __call__ samples a latent and invokes VAE decode only.
        assert_eq!(
            text_name("language_model.layers.3.mlp.down_proj.weight").as_deref(),
            Some("model.layers.3.mlp.down_proj.weight")
        );
        assert_eq!(text_name("visual.blocks.0.attn.qkv.weight"), None);
        for name in [
            "decoder.conv_in.weight",
            "post_quant_conv.weight",
            "post_quant_conv.bias",
            "bn.running_mean",
            "bn.running_var",
        ] {
            assert!(keep_vae_tensor(name), "required decode tensor {name}");
        }
        for name in [
            "encoder.conv_in.weight",
            "quant_conv.weight",
            "quant_conv.bias",
            "bn.num_batches_tracked",
        ] {
            assert!(!keep_vae_tensor(name), "encode/training-only tensor {name}");
        }
    }

    /// Release-source contract for the pinned upstream Base snapshot.  The
    /// heavyweight source checkout is intentionally optional so standard CI
    /// remains weight-free; release validation supplies `CMF_MOSS_SOURCE_DIR`.
    #[test]
    fn source_base_tti_contract_when_snapshot_supplied() {
        let Ok(root) = std::env::var("CMF_MOSS_SOURCE_DIR") else {
            return;
        };
        let root = std::path::PathBuf::from(root);
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("model_index.json")).expect("read MOSS model_index"),
        )
        .expect("parse MOSS model_index");
        assert_eq!(index["_class_name"], "MOSSImagePipeline");
        let components = index
            .as_object()
            .expect("MOSS model_index object")
            .keys()
            .filter(|key| !key.starts_with('_'))
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            components,
            std::collections::BTreeSet::from([
                "scheduler",
                "text_encoder",
                "tokenizer",
                "transformer",
                "vae",
            ])
        );

        let pipeline = std::fs::read_to_string(root.join("pipeline_moss_image.py"))
            .expect("read MOSS custom pipeline");
        for required in [
            "output = self.text_encoder(**inputs, use_cache=False, return_dict=True)",
            "if latents is None:",
            "images = self.vae.decode(z, return_dict=False)[0].clamp(-1, 1)",
        ] {
            assert!(
                pipeline.contains(required),
                "pinned Base pipeline lost required text-to-image route: {required}"
            );
        }
        assert!(
            !pipeline.contains("self.vae.encode("),
            "Base text-to-image pipeline unexpectedly invokes VAE encode"
        );

        let names = |dir: &str, index: &str, single: &str| {
            zimagepack::shard_files(&root.join(dir), index, single)
                .expect("source shard list")
                .into_iter()
                .flat_map(|path| {
                    zimagepack::st_header(&path)
                        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
                        .0
                        .into_iter()
                        .map(|tensor| tensor.name)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let text = names(
            "text_encoder",
            "model.safetensors.index.json",
            "model.safetensors",
        );
        assert_eq!(
            text.iter()
                .filter(|name| name.starts_with("visual."))
                .count(),
            333,
            "pinned Qwen visual inventory"
        );
        assert_eq!(
            text.iter()
                .filter(|name| name.starts_with("language_model."))
                .count(),
            426,
            "pinned Qwen language inventory"
        );
        assert_eq!(
            text.iter().filter_map(|name| text_name(name)).count(),
            426,
            "all and only language tensors are retained"
        );

        let vae = names(
            "vae",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors",
        );
        assert_eq!(
            vae.iter()
                .filter(|name| name.starts_with("decoder."))
                .count(),
            138,
            "pinned Flux2 decoder inventory"
        );
        assert_eq!(
            vae.iter()
                .filter(|name| name.starts_with("post_quant_conv."))
                .count(),
            2,
            "pinned Flux2 post-quant inventory"
        );
        assert_eq!(
            vae.iter().filter(|name| keep_vae_tensor(name)).count(),
            142,
            "all and only Base decode tensors are retained"
        );
        assert_eq!(
            vae.iter()
                .filter(|name| name.starts_with("encoder."))
                .count(),
            106,
            "pinned Flux2 encode inventory"
        );
        assert_eq!(
            vae.iter()
                .filter(|name| name.starts_with("quant_conv."))
                .count(),
            2,
            "pinned Flux2 quant-conv inventory"
        );
        assert_eq!(
            vae.iter()
                .filter(|name| name.as_str() == "bn.num_batches_tracked")
                .count(),
            1,
            "pinned Flux2 BatchNorm training counter"
        );
    }
}
