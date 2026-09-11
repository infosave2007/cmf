//! Stream official Qwen Image companion components into standalone CMFs.
//! Remote sources are read in bounded parallel ranges; no source checkpoint
//! copy or expanded whole-model weight array is needed.
use crate::{convert, gguf, http_range};
use anyhow::{anyhow, ensure, Context};
use cortiq_core::format::{CmfHeader, CmfModel, CmfStreamWriter};
use cortiq_core::types::{ModelArch, TensorDtype};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

const SMALL_SOURCE_LIMIT: usize = 32 * 1024 * 1024;
const SAFETENSORS_HEADER_LIMIT: usize = 16 * 1024 * 1024;

struct Source {
    root: String,
    remote: bool,
    agent: ureq::Agent,
    token: Option<String>,
}
impl Source {
    fn new(root: &str) -> anyhow::Result<Self> {
        let remote = root.starts_with("https://") || root.starts_with("http://");
        Ok(Self {
            root: root.trim_end_matches('/').into(),
            remote,
            agent: ureq::AgentBuilder::new()
                .timeout_connect(std::time::Duration::from_secs(30))
                .timeout_read(std::time::Duration::from_secs(120))
                .build(),
            token: std::env::var("HF_TOKEN").ok(),
        })
    }
    fn location(&self, relative: &str) -> anyhow::Result<String> {
        ensure!(
            !relative.is_empty()
                && !relative.contains('\\')
                && Path::new(relative).is_relative()
                && Path::new(relative)
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
            "invalid component path"
        );
        Ok(format!("{}/{relative}", self.root))
    }
    fn small(&self, relative: &str, optional: bool) -> anyhow::Result<Option<Vec<u8>>> {
        let location = self.location(relative)?;
        if !self.remote {
            let mut file = match std::fs::File::open(&location) {
                Ok(file) => file,
                Err(e) if optional && e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(e) => return Err(e.into()),
            };
            let size = usize::try_from(file.metadata()?.len())?;
            ensure!(
                size <= SMALL_SOURCE_LIMIT,
                "component metadata exceeds 32 MiB"
            );
            let mut bytes = Vec::with_capacity(size);
            file.take(SMALL_SOURCE_LIMIT as u64 + 1)
                .read_to_end(&mut bytes)?;
            ensure!(
                bytes.len() <= SMALL_SOURCE_LIMIT,
                "component metadata exceeds 32 MiB"
            );
            return Ok(Some(bytes));
        }
        let mut req = self.agent.get(&location);
        if location.starts_with("https://huggingface.co/") {
            if let Some(t) = &self.token {
                req = req.set("Authorization", &format!("Bearer {t}"));
            }
        }
        let response = match req.call() {
            Ok(r) => r,
            Err(ureq::Error::Status(404, _)) if optional => return Ok(None),
            Err(e) => return Err(anyhow!("read {relative}: {e}")),
        };
        let mut bytes = Vec::new();
        response
            .into_reader()
            .take(SMALL_SOURCE_LIMIT as u64 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= SMALL_SOURCE_LIMIT,
            "component metadata exceeds 32 MiB"
        );
        Ok(Some(bytes))
    }
    fn required(&self, relative: &str) -> anyhow::Result<Vec<u8>> {
        self.small(relative, false)?
            .ok_or_else(|| anyhow!("missing {relative}"))
    }
}
struct Tensor {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    range: std::ops::Range<usize>,
}
struct Shard {
    relative: String,
    size: usize,
    tensors: Vec<Tensor>,
}
fn parse_header(
    relative: String,
    raw: &[u8],
    data_start: usize,
    size: usize,
) -> anyhow::Result<Shard> {
    ensure!(
        data_start <= size,
        "safetensors data starts past source end"
    );
    let json: BTreeMap<String, serde_json::Value> = serde_json::from_slice(raw)?;
    let mut tensors = Vec::new();
    for (name, value) in json {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value["dtype"]
            .as_str()
            .context("safetensors dtype")?
            .to_owned();
        let item_bytes = match dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            _ => return Err(anyhow!("unsupported companion dtype {dtype}")),
        };
        let shape = value["shape"]
            .as_array()
            .context("safetensors shape")?
            .iter()
            .map(|v| {
                usize::try_from(v.as_u64().context("invalid tensor dimension")?)
                    .map_err(anyhow::Error::from)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        ensure!(
            !shape.is_empty() && shape.len() <= 8 && shape.iter().all(|&d| d > 0),
            "invalid shape for {name}"
        );
        let bytes = shape
            .iter()
            .try_fold(item_bytes, |a: usize, &b| a.checked_mul(b))
            .context("tensor size overflow")?;
        let offsets = value["data_offsets"]
            .as_array()
            .context("safetensors offsets")?;
        ensure!(offsets.len() == 2, "invalid offsets for {name}");
        let start = usize::try_from(offsets[0].as_u64().context("invalid tensor start")?)?
            .checked_add(data_start)
            .context("offset overflow")?;
        let end = usize::try_from(offsets[1].as_u64().context("invalid tensor end")?)?
            .checked_add(data_start)
            .context("offset overflow")?;
        ensure!(
            start >= data_start && end <= size && end.checked_sub(start) == Some(bytes),
            "invalid tensor range for {name}"
        );
        tensors.push(Tensor {
            name,
            dtype,
            shape,
            range: start..end,
        });
    }
    ensure!(!tensors.is_empty(), "empty companion shard");
    // Source byte order gives predictable sequential disk reads and HTTP prefetch.
    tensors.sort_by_key(|t| t.range.start);
    ensure!(
        tensors
            .windows(2)
            .all(|p| p[0].range.end <= p[1].range.start),
        "overlapping safetensors ranges"
    );
    Ok(Shard {
        relative,
        size,
        tensors,
    })
}
fn shard_header(source: &Source, relative: String) -> anyhow::Result<Shard> {
    let location = source.location(&relative)?;
    if source.remote {
        let (prefix, size) = http_range::read_http_range(
            &source.agent,
            &location,
            source.token.as_deref(),
            0,
            8,
            None,
        )?;
        ensure!(prefix.len() == 8, "truncated safetensors");
        let len = usize::try_from(u64::from_le_bytes(prefix.try_into().unwrap()))?;
        let data_start = len
            .checked_add(8)
            .ok_or_else(|| anyhow!("safetensors header length overflows"))?;
        ensure!(
            len > 0 && len <= SAFETENSORS_HEADER_LIMIT && data_start <= size,
            "invalid safetensors header length"
        );
        let (header, _) = http_range::read_http_range(
            &source.agent,
            &location,
            source.token.as_deref(),
            8,
            8 + len,
            Some(size),
        )?;
        parse_header(relative, &header, data_start, size)
    } else {
        let mut file = std::fs::File::open(location)?;
        let size = usize::try_from(file.metadata()?.len())?;
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)?;
        let len = usize::try_from(u64::from_le_bytes(prefix))?;
        let data_start = len
            .checked_add(8)
            .ok_or_else(|| anyhow!("safetensors header length overflows"))?;
        ensure!(
            len > 0 && len <= SAFETENSORS_HEADER_LIMIT && data_start <= size,
            "invalid safetensors header length"
        );
        let mut header = vec![0; len];
        file.read_exact(&mut header)?;
        parse_header(relative, &header, data_start, size)
    }
}

pub(crate) fn pack(root: &str, component: &str, quant: &str, output: &str) -> anyhow::Result<()> {
    let (folder, class, index, single) = match component {
        "qwen-text-encoder" => (
            "text_encoder",
            "Qwen2_5_VLForConditionalGeneration",
            "model.safetensors.index.json",
            "model.safetensors",
        ),
        "qwen-vae" => (
            "vae",
            "AutoencoderKLQwenImage",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors",
        ),
        _ => return Err(anyhow!("--component expects qwen-text-encoder or qwen-vae")),
    };
    let source = Source::new(root)?;
    let quant = convert::parse_quant(quant)?;
    ensure!(
        matches!(
            quant,
            convert::Quant::F16
                | convert::Quant::Q8Row
                | convert::Quant::Q8_2f
                | convert::Quant::Q4Tiled
                | convert::Quant::Q4TiledP
        ),
        "Qwen companion codec must be f16, q8, q8_2f, q4t or q4tp"
    );
    let quant = if quant == convert::Quant::Q8Row {
        convert::Quant::Q8_2f
    } else {
        quant
    };
    let config = source.required(&format!("{folder}/config.json"))?;
    let cfg: serde_json::Value = serde_json::from_slice(&config)?;
    if folder == "vae" {
        ensure!(
            cfg["_class_name"] == class,
            "expected Qwen Image VAE config"
        );
    } else {
        ensure!(
            cfg["model_type"] == "qwen2_5_vl",
            "expected Qwen2.5-VL encoder config"
        );
    }
    let mut extras = vec![("image.config_json", config.clone())];
    if folder == "text_encoder" {
        extras.push((
            "image.tokenizer_json",
            source.required("processor/tokenizer.json")?,
        ));
        extras.push((
            "image.processor_config_json",
            source.required("processor/preprocessor_config.json")?,
        ));
        if let Some(b) = source.small("processor/tokenizer_config.json", true)? {
            extras.push(("image.tokenizer_config_json", b));
        }
    }
    let mut files = BTreeSet::new();
    if let Some(index_bytes) = source.small(&format!("{folder}/{index}"), true)? {
        let index_json: serde_json::Value = serde_json::from_slice(&index_bytes)?;
        for filename in index_json["weight_map"]
            .as_object()
            .context("component weight_map")?
            .values()
        {
            files.insert(format!(
                "{folder}/{}",
                filename.as_str().context("shard filename")?
            ));
        }
    } else {
        files.insert(format!("{folder}/{single}"));
    }
    ensure!(!files.is_empty(), "component weight_map is empty");
    let shards = files
        .into_iter()
        .map(|f| shard_header(&source, f))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut seen: BTreeSet<String> = extras.iter().map(|(s, _)| s.to_string()).collect();
    for t in shards.iter().flat_map(|s| &s.tensors) {
        ensure!(seen.insert(t.name.clone()), "duplicate tensor {}", t.name);
    }
    let count = seen.len();
    let output_path = PathBuf::from(output);
    let temp = gguf::qwen_image_temp_path(&output_path)?;
    let result = (|| -> anyhow::Result<()> {
        let mut writer =
            CmfStreamWriter::new(&temp, CmfStreamWriter::head_reserve_for(count, 100))?;
        for (name, bytes) in &extras {
            writer.push(name, TensorDtype::U8, &[bytes.len()], bytes)?;
        }
        let mut written = extras.len();
        for shard in &shards {
            eprintln!(
                "Packing {} ({} tensors)",
                shard.relative,
                shard.tensors.len()
            );
            let mut push = |t: &Tensor, raw: &[u8]| -> anyhow::Result<()> {
                ensure!(raw.len() == t.range.len(), "source tensor size changed");
                if folder == "text_encoder"
                    && t.shape.len() == 2
                    && t.name.ends_with(".weight")
                    && quant != convert::Quant::F16
                {
                    let vals = convert::to_f32(&t.dtype, raw)?;
                    let (dtype, bytes) = convert::quantize_2d(quant, &vals, t.shape[0], t.shape[1]);
                    writer.push(&t.name, dtype, &t.shape, &bytes)?;
                } else {
                    let dtype = match t.dtype.as_str() {
                        "F32" => TensorDtype::F32,
                        "F16" => TensorDtype::F16,
                        _ => TensorDtype::Bf16,
                    };
                    writer.push(&t.name, dtype, &t.shape, raw)?;
                }
                written += 1;
                if written % 50 == 0 || written == count {
                    eprintln!("{component}: {written}/{count} tensors");
                }
                Ok(())
            };
            let location = source.location(&shard.relative)?;
            if source.remote {
                http_range::with_http_ranges(
                    &source.agent,
                    &location,
                    source.token.as_deref(),
                    shard.size,
                    shard.tensors.iter().map(|t| t.range.clone()).collect(),
                    |read| {
                        for tensor in &shard.tensors {
                            push(tensor, &read()?)?;
                        }
                        Ok(())
                    },
                )?;
            } else {
                let file = std::fs::File::open(location)?;
                let mapping = unsafe { memmap2::Mmap::map(&file)? };
                ensure!(
                    mapping.len() == shard.size,
                    "source size changed while packing"
                );
                for tensor in &shard.tensors {
                    push(tensor, &mapping[tensor.range.clone()])?;
                }
            }
        }
        let arch: ModelArch = serde_json::from_value(serde_json::json!({
            "arch_name": format!("qwen_image_{folder}"), "hidden_size": cfg["hidden_size"].as_u64().unwrap_or(0),
            "intermediate_size": 0, "num_layers": 0, "num_attention_heads": 0, "num_kv_heads": 0,
            "head_dim": 0, "vocab_size": cfg["vocab_size"].as_u64().unwrap_or(0), "layer_types": [],
            "rms_norm_eps": 1e-6, "max_position_embeddings": 0,
            "linear_conv_kernel_dim": 0, "linear_num_key_heads": 0, "linear_num_value_heads": 0
        }))?;
        let header = CmfHeader {
            format: "cmf".into(),
            version: cortiq_core::CMF_VERSION,
            arch,
            quant_type: gguf::quant_type_for(quant),
            provenance: Some(serde_json::json!({
                "tool": "cortiq imagine-pack", "source": root, "component": folder,
                "component_class": class, "tensor_name_policy": "source_names_unchanged",
                "quant": if folder == "vae" { "source_float" } else { convert::quant_name(quant) }
            })),
            tokenizer_config: None,
            section_hashes: None,
            skills: vec![],
            shard: None,
            calibration: None,
            routing: None,
        };
        writer.finish(&header, None, None)?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&temp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&temp, &output_path) {
        let _ = std::fs::remove_file(&temp);
        return Err(e.into());
    }
    gguf::qwen_image_sync_parent(&output_path)?;
    println!(
        "{output}: {count} tensors, {:.3} GiB",
        std::fs::metadata(output)?.len() as f64 / 1073741824.0
    );
    Ok(())
}

/// The three files in a ready Qwen Image bundle share one CMF directory.  The
/// payload names remain the names consumed by the native loaders; only the
/// colliding component config is given an explicit alias.
#[derive(Clone, Copy, Debug)]
enum BundlePart {
    Transformer,
    TextEncoder,
    Vae,
}

impl BundlePart {
    fn label(self) -> &'static str {
        match self {
            Self::Transformer => "transformer",
            Self::TextEncoder => "text_encoder",
            Self::Vae => "vae",
        }
    }
}

fn bundle_entry_name(part: BundlePart, name: &str) -> String {
    match (part, name) {
        (BundlePart::TextEncoder, "image.config_json") => "image.text_encoder.config_json".into(),
        (BundlePart::Vae, "image.config_json") => "image.vae.config_json".into(),
        _ => name.to_string(),
    }
}

fn u8_json<'a>(model: &'a CmfModel, name: &str, label: &str) -> anyhow::Result<&'a [u8]> {
    let entry = model
        .tensor(name)
        .ok_or_else(|| anyhow!("{label}: missing {name}"))?;
    ensure!(
        entry.dtype == TensorDtype::U8 && entry.shape.len() == 1,
        "{label}: {name} must be a one-dimensional U8 blob"
    );
    ensure!(
        entry.shape[0] == entry.n_elems(),
        "{label}: {name} has an invalid byte length"
    );
    Ok(model.entry_bytes(entry))
}

/// Validate a source component and return the names needed for a bounded
/// preflight.  The component is unmapped before the next one is opened, so a
/// bundle operation never needs all three large source mappings live together.
fn inspect_bundle_part(path: &Path, part: BundlePart) -> anyhow::Result<(CmfHeader, Vec<String>)> {
    let model = CmfModel::open_sharded(path)
        .map_err(|e| anyhow!("{} component {}: {e}", part.label(), path.display()))?;
    let config = u8_json(&model, "image.config_json", part.label())?;
    let config_json: serde_json::Value = serde_json::from_slice(config)
        .with_context(|| format!("{}: invalid image.config_json", part.label()))?;
    match part {
        BundlePart::Transformer => {
            ensure!(
                model.header.arch.arch_name == "qwen_image",
                "transformer component has architecture '{}', expected qwen_image",
                model.header.arch.arch_name
            );
            for key in [
                "patch_size",
                "in_channels",
                "out_channels",
                "num_attention_heads",
                "attention_head_dim",
                "num_layers",
            ] {
                ensure!(
                    config_json
                        .get(key)
                        .and_then(serde_json::Value::as_u64)
                        .is_some(),
                    "transformer image.config_json missing numeric {key}"
                );
            }
        }
        BundlePart::TextEncoder => {
            ensure!(
                config_json["model_type"] == "qwen2_5_vl",
                "text encoder image.config_json is not qwen2_5_vl"
            );
            u8_json(&model, "image.tokenizer_json", "text_encoder")?;
            u8_json(&model, "image.processor_config_json", "text_encoder")?;
        }
        BundlePart::Vae => {
            ensure!(
                config_json["_class_name"] == "AutoencoderKLQwenImage",
                "VAE image.config_json is not AutoencoderKLQwenImage"
            );
            ensure!(
                config_json
                    .get("z_dim")
                    .and_then(serde_json::Value::as_u64)
                    .is_some(),
                "VAE image.config_json missing numeric z_dim"
            );
        }
    }
    let names = model
        .tensors
        .iter()
        .map(|entry| bundle_entry_name(part, &entry.name))
        .collect();
    Ok((model.header.clone(), names))
}

fn copy_bundle_part(
    writer: &mut CmfStreamWriter,
    path: &Path,
    part: BundlePart,
) -> anyhow::Result<usize> {
    let model = CmfModel::open_sharded(path)
        .map_err(|e| anyhow!("{} component {}: {e}", part.label(), path.display()))?;
    let mut copied = 0usize;
    for entry in &model.tensors {
        let name = bundle_entry_name(part, &entry.name);
        writer.push_bounded(
            &name,
            entry.dtype,
            &entry.shape,
            model.entry_bytes(entry),
            16 * 1024 * 1024,
        )?;
        copied += 1;
        if copied % 100 == 0 || copied == model.tensors.len() {
            eprintln!("{}: {copied}/{} tensors", part.label(), model.tensors.len());
        }
    }
    Ok(copied)
}

fn read_bundle_scheduler(path: &Path) -> anyhow::Result<Vec<u8>> {
    let bytes =
        std::fs::read(path).with_context(|| format!("read scheduler config {}", path.display()))?;
    ensure!(
        bytes.len() <= SMALL_SOURCE_LIMIT,
        "scheduler config exceeds 32 MiB"
    );
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid scheduler config {}", path.display()))?;
    ensure!(value.is_object(), "scheduler config must be a JSON object");
    Ok(bytes)
}

/// Merge the retained standalone Qwen components into one mmap-served CMF.
/// This copies encoded payloads byte-for-byte; it does not requantize or hold
/// the three source mappings at the same time.
pub(crate) fn bundle(root: &str, output: &str) -> anyhow::Result<()> {
    let root_path = Path::new(root);
    ensure!(
        root_path.is_dir(),
        "Qwen bundle root must be a component directory"
    );
    let transformer_path = root_path.join("transformer.cmf");
    let text_encoder_path = root_path.join("text_encoder.cmf");
    let vae_path = root_path.join("vae.cmf");
    let scheduler_path = root_path.join("scheduler_config.json");
    for path in [
        &transformer_path,
        &text_encoder_path,
        &vae_path,
        &scheduler_path,
    ] {
        ensure!(
            path.is_file(),
            "missing Qwen bundle input {}",
            path.display()
        );
    }
    let output_path = PathBuf::from(output);
    let output_abs = std::fs::canonicalize(&output_path).ok();
    for path in [
        &transformer_path,
        &text_encoder_path,
        &vae_path,
        &scheduler_path,
    ] {
        if output_abs.as_ref() == std::fs::canonicalize(path).ok().as_ref() {
            return Err(anyhow!(
                "bundle output must differ from input {}",
                path.display()
            ));
        }
    }

    let (transformer_header, transformer_names) =
        inspect_bundle_part(&transformer_path, BundlePart::Transformer)?;
    let (_, text_encoder_names) = inspect_bundle_part(&text_encoder_path, BundlePart::TextEncoder)?;
    let (_, vae_names) = inspect_bundle_part(&vae_path, BundlePart::Vae)?;
    let scheduler = read_bundle_scheduler(&scheduler_path)?;

    let manifest = serde_json::to_vec(&serde_json::json!({
        "version": 1,
        "pipeline": "qwen-image-edit-2509",
        "components": {
            "transformer": "transformer.cmf",
            "text_encoder": "text_encoder.cmf",
            "vae": "vae.cmf"
        },
        "embedded_assets": [
            "image.text_encoder.config_json",
            "image.vae.config_json",
            "image.tokenizer_json",
            "image.processor_config_json",
            "image.scheduler_config_json"
        ],
        "tensor_payloads": "standalone bytes copied without requantization"
    }))?;

    let mut names = BTreeSet::new();
    for name in transformer_names
        .into_iter()
        .chain(text_encoder_names)
        .chain(vae_names)
    {
        ensure!(
            name != "image.bundle_config_json" && name != "image.scheduler_config_json",
            "reserved bundle asset name already present: {name}"
        );
        ensure!(
            names.insert(name.clone()),
            "duplicate bundle tensor name {name}"
        );
    }
    let count = names
        .len()
        .checked_add(2)
        .ok_or_else(|| anyhow!("bundle tensor count overflows"))?;
    let temp = gguf::qwen_image_temp_path(&output_path)?;
    let result = (|| -> anyhow::Result<()> {
        let mut writer =
            CmfStreamWriter::new(&temp, CmfStreamWriter::head_reserve_for(count, 128))?;
        let copied_transformer =
            copy_bundle_part(&mut writer, &transformer_path, BundlePart::Transformer)?;
        let copied_text_encoder =
            copy_bundle_part(&mut writer, &text_encoder_path, BundlePart::TextEncoder)?;
        let copied_vae = copy_bundle_part(&mut writer, &vae_path, BundlePart::Vae)?;
        ensure!(
            copied_transformer
                .checked_add(copied_text_encoder)
                .and_then(|n| n.checked_add(copied_vae))
                .and_then(|n| n.checked_add(2))
                == Some(count),
            "bundle source changed during preflight"
        );
        writer.push_bounded(
            "image.scheduler_config_json",
            TensorDtype::U8,
            &[scheduler.len()],
            &scheduler,
            16 * 1024 * 1024,
        )?;
        writer.push_bounded(
            "image.bundle_config_json",
            TensorDtype::U8,
            &[manifest.len()],
            &manifest,
            16 * 1024 * 1024,
        )?;
        let mut header = transformer_header.clone();
        header.arch.arch_name = "qwen_image_bundle".into();
        header.provenance = Some(serde_json::json!({
            "tool": "cortiq imagine-pack",
            "pipeline": "qwen-image-edit-2509",
            "bundle": true,
            "components": ["transformer.cmf", "text_encoder.cmf", "vae.cmf"],
            "tensor_payloads": "standalone bytes copied without requantization"
        }));
        header.tokenizer_config = None;
        header.section_hashes = None;
        header.skills.clear();
        header.shard = None;
        header.calibration = None;
        header.routing = None;
        writer.finish(&header, None, None)?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&temp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&temp, &output_path) {
        let _ = std::fs::remove_file(&temp);
        return Err(e.into());
    }
    gguf::qwen_image_sync_parent(&output_path)?;
    println!(
        "{output}: {count} tensors, {:.3} GiB",
        std::fs::metadata(&output_path)?.len() as f64 / 1073741824.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::Arc;
    use std::thread::JoinHandle;

    fn unique_path(label: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cortiq-qwen-pack-{label}-{}-{stamp}",
            std::process::id()
        ))
    }

    fn safetensors(entries: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        let mut offset = 0usize;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, bytes) in entries {
            let end = offset + bytes.len();
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, end]
                }),
            );
            offset = end;
        }
        header.insert(
            "__metadata__".into(),
            serde_json::json!({"format": "pt", "fixture": "qwen-image"}),
        );
        let header = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::with_capacity(8 + header.len() + offset);
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        for (_, _, _, bytes) in entries {
            out.extend_from_slice(bytes);
        }
        out
    }

    fn write_component_source(root: &Path, component: &str) -> BTreeMap<String, Vec<u8>> {
        let mut files = BTreeMap::new();
        let folder = root.join(component);
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::create_dir_all(root.join("processor")).unwrap();
        if component == "vae" {
            let config = br#"{"_class_name":"AutoencoderKLQwenImage","base_dim":8,"z_dim":16}"#;
            std::fs::write(folder.join("config.json"), config).unwrap();
            files.insert("vae/config.json".into(), config.to_vec());
            let values: Vec<u8> = (0..64)
                .flat_map(|i| (i as f32 * 0.25).to_le_bytes())
                .collect();
            let shard = safetensors(&[("encoder.conv_in.weight", "F32", &[2, 32], &values)]);
            std::fs::write(folder.join("diffusion_pytorch_model.safetensors"), &shard).unwrap();
            files.insert("vae/diffusion_pytorch_model.safetensors".into(), shard);
            let index = br#"{"weight_map":{"encoder.conv_in.weight":"diffusion_pytorch_model.safetensors"}}"#;
            std::fs::write(
                folder.join("diffusion_pytorch_model.safetensors.index.json"),
                index,
            )
            .unwrap();
            files.insert(
                "vae/diffusion_pytorch_model.safetensors.index.json".into(),
                index.to_vec(),
            );
        } else {
            let config = br#"{"model_type":"qwen2_5_vl","hidden_size":32,"vocab_size":4}"#;
            std::fs::write(folder.join("config.json"), config).unwrap();
            files.insert("text_encoder/config.json".into(), config.to_vec());
            let tokenizer = br#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}"#;
            let processor = br#"{"image_processor_type":"Qwen2VLImageProcessor","min_pixels":1,"max_pixels":4}"#;
            let tokenizer_config = br#"{"tokenizer_class":"Qwen2VLProcessor"}"#;
            for (relative, bytes) in [
                ("processor/tokenizer.json", tokenizer.as_slice()),
                ("processor/preprocessor_config.json", processor.as_slice()),
                (
                    "processor/tokenizer_config.json",
                    tokenizer_config.as_slice(),
                ),
            ] {
                std::fs::write(root.join(relative), bytes).unwrap();
                files.insert(relative.into(), bytes.to_vec());
            }
            let values: Vec<u8> = (0..(2 * 32))
                .flat_map(|i| ((i as f32 - 16.0) / 16.0).to_le_bytes())
                .collect();
            let bias: Vec<u8> = (0..2).flat_map(|i| (i as f32).to_le_bytes()).collect();
            let shard = safetensors(&[
                ("model.embed_tokens.weight", "F32", &[2, 32], &values),
                ("model.layers.0.input_layernorm.weight", "F32", &[2], &bias),
            ]);
            std::fs::write(folder.join("model.safetensors"), &shard).unwrap();
            files.insert("text_encoder/model.safetensors".into(), shard);
            let index = br#"{"weight_map":{"model.embed_tokens.weight":"model.safetensors","model.layers.0.input_layernorm.weight":"model.safetensors"}}"#;
            std::fs::write(folder.join("model.safetensors.index.json"), index).unwrap();
            files.insert(
                "text_encoder/model.safetensors.index.json".into(),
                index.to_vec(),
            );
        }
        files
    }

    fn serve_files(
        files: Arc<BTreeMap<String, Vec<u8>>>,
        expected_requests: usize,
    ) -> (String, JoinHandle<anyhow::Result<()>>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let handle = std::thread::spawn(move || {
            for _ in 0..expected_requests {
                let (mut stream, _) = listener.accept()?;
                handle_request(&mut stream, &files)?;
            }
            Ok(())
        });
        (address, handle)
    }

    fn handle_request(
        stream: &mut TcpStream,
        files: &BTreeMap<String, Vec<u8>>,
    ) -> anyhow::Result<()> {
        let mut request = Vec::new();
        let mut buf = [0u8; 1024];
        while !request.windows(4).any(|w| w == b"\r\n\r\n") {
            let n = stream.read(&mut buf)?;
            ensure!(n > 0, "truncated HTTP request");
            request.extend_from_slice(&buf[..n]);
            ensure!(request.len() <= 64 * 1024, "HTTP request too large");
        }
        let request = String::from_utf8_lossy(&request);
        let mut lines = request.lines();
        let target = lines
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .ok_or_else(|| anyhow!("malformed HTTP request"))?;
        let relative = target.trim_start_matches('/');
        let Some(body) = files.get(relative) else {
            stream.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )?;
            return Ok(());
        };
        let range = lines.find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.eq_ignore_ascii_case("range")).then_some(value.trim().to_string())
        });
        let (status, start, end) = if let Some(range) = range {
            let span = range
                .strip_prefix("bytes=")
                .ok_or_else(|| anyhow!("malformed test range"))?;
            let (start, end) = span
                .split_once('-')
                .ok_or_else(|| anyhow!("malformed test range"))?;
            let start = start.parse::<usize>()?;
            let end = end.parse::<usize>()?.min(body.len().saturating_sub(1));
            ensure!(start <= end && start < body.len(), "invalid test range");
            ("206 Partial Content", start, end)
        } else {
            ("200 OK", 0, body.len().saturating_sub(1))
        };
        let payload = if body.is_empty() {
            &[][..]
        } else {
            &body[start..=end]
        };
        let header = if status.starts_with("206") {
            format!(
                "HTTP/1.1 {status}\r\nContent-Range: bytes {start}-{end}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len(),
                payload.len()
            )
        } else {
            format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            )
        };
        stream.write_all(header.as_bytes())?;
        stream.write_all(payload)?;
        Ok(())
    }

    fn assert_same_payloads(local: &cortiq_core::CmfModel, remote: &cortiq_core::CmfModel) {
        assert_eq!(
            serde_json::to_value(&local.header.arch).unwrap(),
            serde_json::to_value(&remote.header.arch).unwrap()
        );
        assert_eq!(local.header.quant_type, remote.header.quant_type);
        assert_eq!(local.tensors.len(), remote.tensors.len());
        for (left, right) in local.tensors.iter().zip(&remote.tensors) {
            assert_eq!(
                (&left.name, left.dtype, &left.shape),
                (&right.name, right.dtype, &right.shape)
            );
            assert_eq!(
                local.entry_bytes(left),
                remote.entry_bytes(right),
                "{}",
                left.name
            );
        }
        assert_eq!(local.vocab, remote.vocab);
        for name in local.tensors.iter().map(|t| t.name.as_str()) {
            if name.starts_with("image.") {
                assert_eq!(
                    local.tensor_bytes(name).unwrap(),
                    remote.tensor_bytes(name).unwrap()
                );
            }
        }
    }

    fn bundle_header(arch_name: &str) -> CmfHeader {
        let arch: ModelArch = serde_json::from_value(serde_json::json!({
            "arch_name": arch_name,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_layers": 1,
            "num_attention_heads": 1,
            "num_kv_heads": 1,
            "head_dim": 32,
            "vocab_size": 4,
            "layer_types": [],
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128,
            "linear_conv_kernel_dim": 0,
            "linear_num_key_heads": 0,
            "linear_num_value_heads": 0
        }))
        .unwrap();
        CmfHeader {
            format: "cmf".into(),
            version: cortiq_core::CMF_VERSION,
            arch,
            quant_type: gguf::quant_type_for(convert::Quant::F16),
            provenance: None,
            tokenizer_config: None,
            section_hashes: None,
            skills: vec![],
            shard: None,
            calibration: None,
            routing: None,
        }
    }

    fn write_tiny_component(
        path: &Path,
        arch_name: &str,
        config: &[u8],
        extras: &[(&str, &[u8])],
        tensors: &[(&str, &[usize], &[u8])],
    ) {
        let mut specs = Vec::with_capacity(1 + extras.len() + tensors.len());
        specs.push(cortiq_core::format::TensorSpec {
            name: "image.config_json".into(),
            dtype: TensorDtype::U8,
            shape: vec![config.len()],
            data: config.to_vec(),
        });
        specs.extend(
            extras
                .iter()
                .map(|(name, data)| cortiq_core::format::TensorSpec {
                    name: (*name).into(),
                    dtype: TensorDtype::U8,
                    shape: vec![data.len()],
                    data: (*data).to_vec(),
                }),
        );
        specs.extend(
            tensors
                .iter()
                .map(|(name, shape, data)| cortiq_core::format::TensorSpec {
                    name: (*name).into(),
                    dtype: TensorDtype::F32,
                    shape: shape.to_vec(),
                    data: (*data).to_vec(),
                }),
        );
        cortiq_core::CmfModel::write(path, &bundle_header(arch_name), &specs, None, None).unwrap();
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn qwen_bundle_copies_components_and_embeds_assets() {
        let root = unique_path("bundle");
        std::fs::create_dir_all(&root).unwrap();
        let transformer_config = br#"{"patch_size":2,"in_channels":64,"out_channels":64,"num_attention_heads":1,"attention_head_dim":32,"num_layers":1,"joint_attention_dim":32}"#;
        let transformer_weight = [1.0f32, -2.0, 3.0, -4.0];
        let transformer_bytes = f32_bytes(&transformer_weight);
        write_tiny_component(
            &root.join("transformer.cmf"),
            "qwen_image",
            transformer_config,
            &[],
            &[("img_in.weight", &[2, 2], &transformer_bytes)],
        );
        let text_config = br#"{"model_type":"qwen2_5_vl","hidden_size":32,"vocab_size":4}"#;
        let tokenizer = br#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}"#;
        let processor =
            br#"{"image_processor_type":"Qwen2VLImageProcessor","min_pixels":1,"max_pixels":4}"#;
        let tokenizer_config = br#"{"tokenizer_class":"Qwen2VLProcessor"}"#;
        let text_weight = [5.0f32, 6.0, 7.0, 8.0];
        let text_bytes = f32_bytes(&text_weight);
        write_tiny_component(
            &root.join("text_encoder.cmf"),
            "qwen_image_text_encoder",
            text_config,
            &[
                ("image.tokenizer_json", tokenizer),
                ("image.processor_config_json", processor),
                ("image.tokenizer_config_json", tokenizer_config),
            ],
            &[("model.embed_tokens.weight", &[2, 2], &text_bytes)],
        );
        let vae_config = br#"{"_class_name":"AutoencoderKLQwenImage","base_dim":8,"z_dim":16}"#;
        let vae_weight = [9.0f32, 10.0, 11.0, 12.0];
        let vae_bytes = f32_bytes(&vae_weight);
        write_tiny_component(
            &root.join("vae.cmf"),
            "qwen_image_vae",
            vae_config,
            &[],
            &[("encoder.conv_in.weight", &[2, 2], &vae_bytes)],
        );
        let scheduler =
            br#"{"base_image_seq_len":256,"max_image_seq_len":8192,"num_train_timesteps":1000}"#;
        std::fs::write(root.join("scheduler_config.json"), scheduler).unwrap();
        let output = unique_path("bundle-output.cmf");

        bundle(root.to_str().unwrap(), output.to_str().unwrap()).unwrap();
        let model = cortiq_core::CmfModel::open(&output).unwrap();
        assert!(model.verify().is_empty());
        assert_eq!(model.header.arch.arch_name, "qwen_image_bundle");
        for name in [
            "image.config_json",
            "image.text_encoder.config_json",
            "image.vae.config_json",
            "image.tokenizer_json",
            "image.processor_config_json",
            "image.tokenizer_config_json",
            "image.scheduler_config_json",
            "image.bundle_config_json",
            "img_in.weight",
            "model.embed_tokens.weight",
            "encoder.conv_in.weight",
        ] {
            assert!(model.tensor(name).is_some(), "missing {name}");
        }
        assert_eq!(
            model.tensor_bytes("image.config_json").unwrap(),
            transformer_config
        );
        assert_eq!(
            model
                .tensor_bytes("image.text_encoder.config_json")
                .unwrap(),
            text_config
        );
        assert_eq!(
            model.tensor_bytes("image.vae.config_json").unwrap(),
            vae_config
        );
        assert_eq!(
            model.tensor_bytes("image.scheduler_config_json").unwrap(),
            scheduler
        );
        assert_eq!(
            model.tensor_bytes("img_in.weight").unwrap(),
            transformer_bytes
        );
        assert_eq!(
            model.tensor_bytes("model.embed_tokens.weight").unwrap(),
            text_bytes
        );
        assert_eq!(
            model.tensor_bytes("encoder.conv_in.weight").unwrap(),
            vae_bytes
        );
        let manifest: serde_json::Value =
            serde_json::from_slice(model.tensor_bytes("image.bundle_config_json").unwrap())
                .unwrap();
        assert_eq!(manifest["pipeline"], "qwen-image-edit-2509");
        drop(model);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_file(output);
    }

    #[test]
    fn qwen_bundle_rejects_missing_or_malformed_scheduler() {
        let root = unique_path("bundle-invalid");
        std::fs::create_dir_all(&root).unwrap();
        let transformer_config = br#"{"patch_size":2,"in_channels":64,"out_channels":64,"num_attention_heads":1,"attention_head_dim":32,"num_layers":1}"#;
        let text_config = br#"{"model_type":"qwen2_5_vl","hidden_size":32,"vocab_size":4}"#;
        let vae_config = br#"{"_class_name":"AutoencoderKLQwenImage","z_dim":16}"#;
        let bytes = [0u8; 16];
        let tok = br#"{}"#;
        let proc = br#"{}"#;
        write_tiny_component(
            &root.join("transformer.cmf"),
            "qwen_image",
            transformer_config,
            &[],
            &[("img_in.weight", &[2, 2], &bytes)],
        );
        write_tiny_component(
            &root.join("text_encoder.cmf"),
            "qwen_image_text_encoder",
            text_config,
            &[
                ("image.tokenizer_json", tok),
                ("image.processor_config_json", proc),
            ],
            &[("model.embed_tokens.weight", &[2, 2], &bytes)],
        );
        write_tiny_component(
            &root.join("vae.cmf"),
            "qwen_image_vae",
            vae_config,
            &[],
            &[("encoder.conv_in.weight", &[2, 2], &bytes)],
        );
        let output = unique_path("bundle-invalid-output.cmf");
        assert!(bundle(root.to_str().unwrap(), output.to_str().unwrap()).is_err());
        std::fs::write(root.join("scheduler_config.json"), b"[]").unwrap();
        assert!(bundle(root.to_str().unwrap(), output.to_str().unwrap()).is_err());
        assert!(!output.exists());
        std::fs::write(root.join("scheduler_config.json"), b"{}").unwrap();
        let scheduler_before = std::fs::read(root.join("scheduler_config.json")).unwrap();
        assert!(
            bundle(
                root.to_str().unwrap(),
                root.join("scheduler_config.json").to_str().unwrap()
            )
            .is_err()
        );
        assert_eq!(
            std::fs::read(root.join("scheduler_config.json")).unwrap(),
            scheduler_before
        );
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_file(output);
    }

    #[test]
    fn qwen_vae_pack_local_remote_payload_and_metadata_match() {
        let root = unique_path("vae");
        let files = write_component_source(&root, "vae");
        let local_out = unique_path("vae-local.cmf");
        let remote_out = unique_path("vae-remote.cmf");
        pack(
            root.to_str().unwrap(),
            "qwen-vae",
            "q4tp",
            local_out.to_str().unwrap(),
        )
        .unwrap();
        // config + index + shard prefix/header + one tensor range.
        let (url, server) = serve_files(Arc::new(files), 5);
        pack(&url, "qwen-vae", "q4tp", remote_out.to_str().unwrap()).unwrap();
        server.join().unwrap().unwrap();
        let local = cortiq_core::CmfModel::open(&local_out).unwrap();
        let remote = cortiq_core::CmfModel::open(&remote_out).unwrap();
        assert!(local.verify().is_empty());
        assert!(remote.verify().is_empty());
        assert_same_payloads(&local, &remote);
        drop((local, remote));
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_file(local_out);
        let _ = std::fs::remove_file(remote_out);
    }

    #[test]
    fn qwen_text_encoder_pack_local_remote_preserves_original_names() {
        let root = unique_path("text");
        let files = write_component_source(&root, "text_encoder");
        let local_out = unique_path("text-local.cmf");
        let remote_out = unique_path("text-remote.cmf");
        pack(
            root.to_str().unwrap(),
            "qwen-text-encoder",
            "q4tp",
            local_out.to_str().unwrap(),
        )
        .unwrap();
        // config + three processor/index files + shard prefix/header + two
        // tensor ranges.
        let (url, server) = serve_files(Arc::new(files), 9);
        pack(
            &url,
            "qwen-text-encoder",
            "q4tp",
            remote_out.to_str().unwrap(),
        )
        .unwrap();
        server.join().unwrap().unwrap();
        let local = cortiq_core::CmfModel::open(&local_out).unwrap();
        let remote = cortiq_core::CmfModel::open(&remote_out).unwrap();
        assert!(local.verify().is_empty());
        assert!(remote.verify().is_empty());
        assert_same_payloads(&local, &remote);
        assert!(local.tensor("model.embed_tokens.weight").is_some());
        assert!(local
            .tensor("model.layers.0.input_layernorm.weight")
            .is_some());
        assert!(local.tensor("image.tokenizer_json").is_some());
        assert!(local.tensor("image.processor_config_json").is_some());
        assert!(local.tensor("image.tokenizer_config_json").is_some());
        drop((local, remote));
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_file(local_out);
        let _ = std::fs::remove_file(remote_out);
    }
}
