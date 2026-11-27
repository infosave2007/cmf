//! Pack ai-sage/GigaAM-v4 (`model_type: gigaam`) into one `.cmf`: the
//! Zapformer encoder, both recognition heads (`raw` — lower-case words,
//! `e2e` — case, punctuation and digits), the mel front end's filter bank
//! and window, and the voice-activity model that cuts long recordings.
//!
//! `cortiq convert --model <dir with config.json> --output x.cmf --quant P`
//! detects the model type and comes here. The directory is one revision of
//! the upstream repository (`rnnt` — the default branch — or `ctc`):
//! `config.json`, `model.safetensors`, and, when present,
//! `vad.safetensors` + `vad_config.json` and the two SentencePiece models
//! of the RNN-T heads. The `ssl` revision carries no recognition head and
//! is refused.
//!
//! ## Profiles (`--quant`)
//!
//! * `f16` (alias `exact`) — the encoder as released (half precision), the
//!   heads and the voice-activity model in float32. The parity file.
//! * `q8_2f` (default, also `auto` / `q8`) and `q4tp` — the encoder's 2-D
//!   matrices on the requested codec. What stays exact: every 1-D tensor,
//!   the front end (its convolutions and its 960 → 768 projection, which
//!   every frame enters through), the depthwise convolutions, the heads
//!   (2.7 M weights) and the voice-activity model (1.3 M).
//!   Under `q4tp` the stack projections stay at q8_2f: a stack's whole
//!   output leaves through one of them.
//!   `--tensor-quant PATTERN=QUANT` overrides any 2-D encoder matrix.
//!
//! ## What the packer derives
//!
//! * `…encoders.N.proj.weight` is stored with its `in^-½ / 0.3` scale
//!   applied, and `…encoders.N.proj_t.weight` is its transpose: the
//!   checkpoint uses one matrix in both directions, the runtime wants two
//!   ordinary products.
//! * Of the four copies of the rotary table (one module, serialized once
//!   per stack) only the encoder's own is kept.
//! * The SentencePiece vocabularies become piece lists in the header
//!   (`gigaam.tokenizers`), so the file needs no sidecar.

use crate::convert::{
    Quant, SafeTensors, open_safetensors, quant_name, quantize_2d, tensor_quant_override, to_f32,
};
use cortiq_core::CmfModel;
use cortiq_core::format::{CmfHeader, TensorSpecRef};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use std::path::Path;

/// The architecture name written into the header.
pub const ARCH_NAME: &str = cortiq_engine::gigaam::ARCH_NAME;

/// Is this `config.json` a GigaAM checkpoint?
pub(crate) fn detect(config: &serde_json::Value) -> bool {
    config.get("model_type").and_then(|v| v.as_str()) == Some("gigaam")
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Profile {
    /// the released values: half-precision encoder, float32 heads
    Exact,
    /// everything widened to f32
    F32,
    /// a quantized encoder
    Quant(Quant),
}

pub(crate) fn parse_profile(s: &str) -> anyhow::Result<Profile> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "auto" | "q8_2f" | "q82f" | "q8f" | "q8" => Profile::Quant(Quant::Q8_2f),
        "q4tp" | "q4" => Profile::Quant(Quant::Q4TiledP),
        "f16" | "fp16" | "exact" => Profile::Exact,
        "f32" | "fp32" => Profile::F32,
        other => anyhow::bail!("gigaam: unknown --quant '{other}' (f16 | f32 | q8_2f | q4tp | auto)"),
    })
}

fn profile_name(p: Profile) -> &'static str {
    match p {
        Profile::Exact => "f16",
        Profile::F32 => "f32",
        Profile::Quant(q) => quant_name(q),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Codec {
    /// the source values in the source precision
    Exact,
    /// widened to f32
    F32,
    /// a 2-D matrix on a quantizing codec
    Matrix(Quant),
}

/// Is `name` one of the encoder's products (a 2-D matrix a frame batch is
/// multiplied by)? The heads, the front end and the voice-activity model
/// are not.
fn is_encoder_matrix(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2 && shape[0] >= 2 && name.starts_with("model.encoder.encoder.encoders.")
}

fn is_stack_projection(name: &str) -> bool {
    name.ends_with(".proj.weight") || name.ends_with(".proj_t.weight")
}

/// The codec of one tensor under `profile` (first rule that applies).
fn codec_for(name: &str, shape: &[usize], profile: Profile) -> Codec {
    let base = match profile {
        Profile::Exact => return Codec::Exact,
        Profile::F32 => return Codec::F32,
        Profile::Quant(q) => q,
    };
    if !is_encoder_matrix(name, shape) {
        return Codec::Exact;
    }
    if let Some(q) = tensor_quant_override(name) {
        return match q {
            Quant::F16 => Codec::Exact,
            q => Codec::Matrix(q),
        };
    }
    if is_stack_projection(name) && !matches!(base, Quant::Q8_2f | Quant::Q8Row) {
        return Codec::Matrix(Quant::Q8_2f);
    }
    Codec::Matrix(base)
}

fn exact_dtype(src: &str) -> anyhow::Result<TensorDtype> {
    Ok(match src {
        "BF16" => TensorDtype::Bf16,
        "F16" => TensorDtype::F16,
        "F32" => TensorDtype::F32,
        other => anyhow::bail!("gigaam: unexpected source dtype {other}"),
    })
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn read_json(path: &Path) -> anyhow::Result<serde_json::Value> {
    let b = std::fs::read(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    serde_json::from_slice(&b).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

/// The pieces of a SentencePiece model file, in id order (the `pieces`
/// field of its `ModelProto`: field 1, each a message whose field 1 is the
/// piece text).
pub(crate) fn sentencepiece_pieces(bytes: &[u8]) -> anyhow::Result<Vec<String>> {
    fn varint(b: &[u8], i: &mut usize) -> anyhow::Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *b.get(*i).ok_or_else(|| anyhow::anyhow!("SentencePiece model: truncated varint"))?;
            *i += 1;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
        }
        anyhow::bail!("SentencePiece model: varint too long")
    }
    /// Skip or return one field: `(field number, payload of a length-delimited field)`.
    fn field<'a>(b: &'a [u8], i: &mut usize) -> anyhow::Result<(u64, Option<&'a [u8]>)> {
        let key = varint(b, i)?;
        let (num, wire) = (key >> 3, key & 7);
        match wire {
            0 => {
                varint(b, i)?;
                Ok((num, None))
            }
            1 => {
                *i += 8;
                Ok((num, None))
            }
            5 => {
                *i += 4;
                Ok((num, None))
            }
            2 => {
                let len = varint(b, i)? as usize;
                let end = i.checked_add(len).filter(|e| *e <= b.len());
                let end = end.ok_or_else(|| anyhow::anyhow!("SentencePiece model: truncated field"))?;
                let s = &b[*i..end];
                *i = end;
                Ok((num, Some(s)))
            }
            w => anyhow::bail!("SentencePiece model: unsupported wire type {w}"),
        }
    }
    let mut pieces = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let (num, payload) = field(bytes, &mut i)?;
        anyhow::ensure!(i <= bytes.len(), "SentencePiece model: truncated");
        if let (1, Some(msg)) = (num, payload) {
            let mut j = 0;
            let mut text = None;
            while j < msg.len() {
                let (n, p) = field(msg, &mut j)?;
                if let (1, Some(p)) = (n, p) {
                    text = Some(String::from_utf8(p.to_vec()).map_err(|_| anyhow::anyhow!("SentencePiece model: a piece is not UTF-8"))?);
                }
            }
            pieces.push(text.ok_or_else(|| anyhow::anyhow!("SentencePiece model: a piece has no text"))?);
        }
    }
    anyhow::ensure!(!pieces.is_empty(), "SentencePiece model: no pieces");
    Ok(pieces)
}

/// One planned output tensor.
struct Planned {
    name: String,
    dtype: TensorDtype,
    shape: Vec<usize>,
    /// `None`: the bytes are borrowed from source `(file, tensor)`.
    owned: Option<Vec<u8>>,
    src: (usize, usize),
}

/// Convert the checkpoint in `dir` into `output` under the `quant` profile.
pub fn convert(dir: &Path, quant: &str, output: &str) -> anyhow::Result<()> {
    let profile = parse_profile(quant)?;
    let t0 = std::time::Instant::now();
    let config = read_json(&dir.join("config.json"))?;
    anyhow::ensure!(detect(&config), "{}: config.json is not a GigaAM checkpoint", dir.display());
    let cfg = config.get("cfg").ok_or_else(|| anyhow::anyhow!("gigaam config: no cfg"))?;
    let kind = cfg.get("model_name").and_then(|v| v.as_str()).unwrap_or("");
    anyhow::ensure!(
        matches!(kind, "rnnt" | "ctc") && cfg.get("heads").is_some(),
        "gigaam: revision '{kind}' has no recognition head (the ssl revision is an encoder only) — \
         download the rnnt or the ctc revision"
    );
    let enc = cfg.get("encoder").ok_or_else(|| anyhow::anyhow!("gigaam config: no encoder"))?;
    anyhow::ensure!(
        enc.get("_target_").and_then(|v| v.as_str()).is_some_and(|t| t.ends_with("ZapformerEncoder")),
        "gigaam: only the Zapformer encoder (GigaAM-v4) is supported"
    );
    anyhow::ensure!(
        cfg.get("language_conditioning").is_none_or(|v| v.is_null()),
        "gigaam: language-conditioned checkpoints are not supported"
    );

    // tokenizers of the RNN-T heads: SentencePiece files named by the config
    let mut tokenizers = serde_json::Map::new();
    if let Some(decodings) = cfg.get("decodings").and_then(|v| v.as_object()) {
        for (head, d) in decodings {
            if let Some(file) = d.get("model_path").and_then(|v| v.as_str()) {
                let name = Path::new(file).file_name().and_then(|s| s.to_str()).unwrap_or(file);
                let bytes = std::fs::read(dir.join(name))
                    .map_err(|e| anyhow::anyhow!("{}: {name}: {e} (the head '{head}' needs it)", dir.display()))?;
                let pieces = sentencepiece_pieces(&bytes).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
                tokenizers.insert(head.clone(), serde_json::json!(pieces));
            }
        }
    }

    let mut files: Vec<SafeTensors> = vec![open_safetensors(&dir.join("model.safetensors"))?];
    let vad_cfg = match (dir.join("vad.safetensors"), dir.join("vad_config.json")) {
        (w, c) if w.is_file() && c.is_file() => {
            files.push(open_safetensors(&w)?);
            Some(read_json(&c)?)
        }
        _ => {
            eprintln!("  no vad.safetensors + vad_config.json: the file will not transcribe recordings over 30 s");
            None
        }
    };

    let mut order: Vec<(usize, usize)> = Vec::new();
    for (fi, f) in files.iter().enumerate() {
        for ti in 0..f.tensors.len() {
            order.push((fi, ti));
        }
    }
    let out_name = |fi: usize, name: &str| if fi == 0 { name.to_string() } else { format!("vad.{name}") };
    order.sort_by(|a, b| {
        out_name(a.0, &files[a.0].tensors[a.1].name).cmp(&out_name(b.0, &files[b.0].tensors[b.1].name))
    });

    let mut planned: Vec<Planned> = Vec::new();
    let mut census: std::collections::BTreeMap<String, (usize, u64, u64)> = Default::default();
    let mut encode = |name: String, shape: Vec<usize>, src_dtype: &str, vals: Option<Vec<f32>>, src: (usize, usize), raw_len: usize| -> anyhow::Result<Planned> {
        let codec = codec_for(&name, &shape, profile);
        // a derived tensor (`vals` given) has no source bytes to borrow
        let (dtype, owned) = match (codec, vals) {
            (Codec::Exact, None) => (exact_dtype(src_dtype)?, None),
            (Codec::Exact | Codec::F32, Some(v)) => (TensorDtype::F32, Some(f32_bytes(&v))),
            (Codec::F32, None) => {
                let f = &files[src.0];
                (TensorDtype::F32, Some(f32_bytes(&to_f32(src_dtype, f.bytes(&f.tensors[src.1]))?)))
            }
            (Codec::Matrix(q), v) => {
                let v = match v {
                    Some(v) => v,
                    None => {
                        let f = &files[src.0];
                        to_f32(src_dtype, f.bytes(&f.tensors[src.1]))?
                    }
                };
                let (dt, bytes) = quantize_2d(q, &v, shape[0], shape[1]);
                (dt, Some(bytes))
            }
        };
        let nbytes = owned.as_ref().map_or(raw_len, |b| b.len()) as u64;
        let group = match name.as_str() {
            n if n.starts_with("vad.") => "voice activity",
            n if n.starts_with("model.heads.") => "heads",
            n if n.starts_with("model.encoder.encoder_embed.") || n.starts_with("model.preprocessor.") => "front end",
            _ => "encoder",
        };
        let e = census.entry(group.to_string()).or_default();
        e.0 += 1;
        e.1 += shape.iter().product::<usize>().max(1) as u64;
        e.2 += nbytes;
        Ok(Planned { name, dtype, shape, owned, src })
    };
    for &(fi, ti) in &order {
        let m = &files[fi].tensors[ti];
        // the rotary table is one module seen from every stack: keep the encoder's own copy
        if m.name.contains(".rope_bank.") && m.name.contains(".encoders.") {
            continue;
        }
        let name = out_name(fi, &m.name);
        let raw = files[fi].bytes(m);
        if fi == 0 && m.name.starts_with("model.encoder.encoder.encoders.") && m.name.ends_with(".proj.weight") {
            // one matrix used in both directions: store both, scaled
            anyhow::ensure!(m.shape.len() == 2, "{}: expected a matrix", m.name);
            let (rows, cols) = (m.shape[0], m.shape[1]);
            let scale = (cols as f64).powf(-0.5) as f32 / 0.3f32;
            let w: Vec<f32> = to_f32(&m.dtype, raw)?.iter().map(|v| v * scale).collect();
            let mut wt = vec![0f32; w.len()];
            for r in 0..rows {
                for c in 0..cols {
                    wt[c * rows + r] = w[r * cols + c];
                }
            }
            planned.push(encode(name.clone(), vec![rows, cols], &m.dtype, Some(w), (fi, ti), 0)?);
            let t_name = format!("{}proj_t.weight", name.strip_suffix("proj.weight").unwrap());
            planned.push(encode(t_name, vec![cols, rows], &m.dtype, Some(wt), (fi, ti), 0)?);
            continue;
        }
        planned.push(encode(name, m.shape.clone(), &m.dtype, None, (fi, ti), raw.len())?);
    }
    planned.sort_by(|a, b| a.name.cmp(&b.name));
    anyhow::ensure!(
        planned.iter().any(|p| p.name == "model.encoder.encoder_embed.out.weight"),
        "gigaam: model.encoder.encoder_embed.out.weight missing — not a GigaAM-v4 checkpoint"
    );
    for (g, (n, w, b)) in &census {
        eprintln!("  {g}: {n} tensors, {:.1} M weights → {:.1} MB", *w as f64 / 1e6, *b as f64 / 1e6);
    }

    let refs: Vec<TensorSpecRef> = planned
        .iter()
        .map(|p| TensorSpecRef {
            name: p.name.clone(),
            dtype: p.dtype,
            shape: p.shape.clone(),
            data: match &p.owned {
                Some(b) => b.as_slice(),
                None => files[p.src.0].bytes(&files[p.src.0].tensors[p.src.1]),
            },
        })
        .collect();

    let tuple = |key: &str| -> Vec<usize> {
        match enc.get(key) {
            Some(serde_json::Value::String(s)) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
            Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| vec![v as usize]).unwrap_or_default(),
            _ => Vec::new(),
        }
    };
    let (dims, layers, heads) = (tuple("encoder_dim"), tuple("num_encoder_layers"), tuple("num_heads"));
    anyhow::ensure!(!dims.is_empty() && dims.len() == layers.len(), "gigaam config: encoder_dim / num_encoder_layers");
    let n_layers: usize = layers.iter().sum();
    let widest = dims.iter().copied().max().unwrap_or(768);
    let vocab_size = cfg
        .pointer("/heads/e2e/joint/num_classes")
        .or_else(|| cfg.pointer("/heads/e2e/num_classes"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": ARCH_NAME,
        "hidden_size": widest,
        "intermediate_size": widest * tuple("feedforward_multiple").first().copied().unwrap_or(4),
        "num_layers": n_layers,
        "num_attention_heads": heads.iter().copied().max().unwrap_or(16),
        "num_kv_heads": heads.iter().copied().max().unwrap_or(16),
        "head_dim": tuple("query_head_dim").first().copied().unwrap_or(32),
        "vocab_size": vocab_size,
        "layer_types": vec!["FullAttention"; n_layers],
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 10000,
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
    }))?;
    // the upstream repository and revision, never the local path: the same checkpoint packs to the same bytes anywhere
    let provenance = serde_json::json!({
        "source": format!("ai-sage/GigaAM-v4@{kind}"),
        "base_model": "ai-sage/GigaAM-v4",
        "quant": profile_name(profile),
        "gigaam": {
            "revision": kind,
            "config": config,
            "vad_config": vad_cfg,
            "tokenizers": tokenizers,
        },
    });
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch,
        quant_type: match profile {
            Profile::Exact => QuantType::F16,
            Profile::F32 => QuantType::F32,
            Profile::Quant(Quant::Q4TiledP) => QuantType::Q4Block,
            Profile::Quant(_) => QuantType::Q8_2f,
        },
        provenance: Some(provenance),
        tokenizer_config: None,
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
    CmfModel::write_ref(output, &header, &refs, None, None).map_err(|e| anyhow::anyhow!("write {output}: {e}"))?;
    let size = std::fs::metadata(output)?.len();
    eprintln!(
        "{output}: GigaAM-v4 {kind}, {} tensors, {:.1} MB ({}) in {:.1}s",
        refs.len(),
        size as f64 / 1e6,
        profile_name(profile),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_parse() {
        assert_eq!(parse_profile("auto").unwrap(), Profile::Quant(Quant::Q8_2f));
        assert_eq!(parse_profile("q4tp").unwrap(), Profile::Quant(Quant::Q4TiledP));
        assert_eq!(parse_profile("exact").unwrap(), Profile::Exact);
        assert!(parse_profile("q1").is_err());
    }

    #[test]
    fn only_encoder_matrices_quantize() {
        let q8 = Profile::Quant(Quant::Q8_2f);
        let q4 = Profile::Quant(Quant::Q4TiledP);
        let ff = "model.encoder.encoder.encoders.1.layers.3.feed_forward1.in_proj.weight";
        assert_eq!(codec_for(ff, &[4096, 1024], q8), Codec::Matrix(Quant::Q8_2f));
        assert_eq!(codec_for(ff, &[4096, 1024], q4), Codec::Matrix(Quant::Q4TiledP));
        assert_eq!(codec_for(ff, &[4096, 1024], Profile::Exact), Codec::Exact);
        // a stack's projections stay at eight bits under the four-bit profile
        let proj = "model.encoder.encoder.encoders.1.proj_t.weight";
        assert_eq!(codec_for(proj, &[1536, 1024], q4), Codec::Matrix(Quant::Q8_2f));
        // the front end, the convolutions, the heads and the VAD stay exact
        for (name, shape) in [
            ("model.encoder.encoder_embed.out.weight", vec![768, 960]),
            ("model.encoder.encoder.encoders.0.layers.0.conv_module.depthwise_conv.weight", vec![768, 1, 5]),
            ("model.encoder.encoder.encoders.0.layers.0.norm.weight", vec![768]),
            ("model.heads.e2e.joint.enc.weight", vec![320, 768]),
            ("model.heads.raw.decoder.lstm.weight_hh_l0", vec![1280, 320]),
            ("vad.encoder.layers.0.self_attn.linear_q.weight", vec![128, 128]),
            ("model.preprocessor.featurizer.0.mel_scale.fb", vec![161, 64]),
        ] {
            assert_eq!(codec_for(name, &shape, q4), Codec::Exact, "{name}");
        }
    }

    #[test]
    fn sentencepiece_pieces_are_read_in_id_order() {
        // ModelProto { pieces: [{piece:"<unk>", score:0, type:2}, {piece:"▁a", score:-1.5}] , trainer_spec (field 2) }
        fn piece(text: &str, extra: &[u8]) -> Vec<u8> {
            let mut m = vec![0x0a, text.len() as u8];
            m.extend_from_slice(text.as_bytes());
            m.extend_from_slice(extra);
            let mut out = vec![0x0a, m.len() as u8];
            out.extend(m);
            out
        }
        let mut b = piece("<unk>", &[0x15, 0, 0, 0, 0, 0x18, 2]);
        b.extend(piece("▁a", &[0x15, 0, 0, 0xc0, 0xbf]));
        b.extend([0x12, 0x02, 0x08, 0x01]);
        assert_eq!(sentencepiece_pieces(&b).unwrap(), vec!["<unk>".to_string(), "▁a".to_string()]);
        assert!(sentencepiece_pieces(&[0x0a, 0x7f, 0x01]).is_err());
    }
}
