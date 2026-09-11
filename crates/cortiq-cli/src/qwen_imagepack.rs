//! Stream official Qwen Image companion components into standalone CMFs.
//! Remote sources are read in bounded parallel ranges; no source checkpoint
//! copy or expanded whole-model weight array is needed.
use crate::{convert, gguf, http_range};
use anyhow::{anyhow, ensure, Context};
use cortiq_core::format::{CmfHeader, CmfStreamWriter};
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
