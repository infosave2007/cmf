//! Pack LTX-2.5 — the 21 B audio-video DiT (`AVTransformer3DModel`), the
//! Gemma-4 12 B prompt encoder with its aggregate projections, the video
//! and audio VAEs, the latent upscalers and the duration head — into ONE
//! q4tp `.cmf`.
//!
//! The release is 70 GB of bf16 across six files; the pack is one
//! memory-mapped container of ~22 GB, and every pass can run on a stand
//! whose disk holds less than the sum of its inputs: `--in` carries the
//! previous pass through byte for byte, so a component is packed, its
//! source deleted, and the next one downloaded.
//!
//! ## What decides a tensor's codec
//!
//! * **2-D, `cols % 32 == 0`, at least `--min-q4tp` weights → q4tp.**
//!   That is every projection of the DiT and of the encoder: 4.16 bits a
//!   weight with the per-row scale ladder.
//! * **The adaLN tables stay exact.** `scale_shift_table` and the
//!   per-block prompt/audio tables ship as F32 in the release and are
//!   read once per step to modulate everything else; four bits there
//!   would move every block's normalization. 19 MB in total — free.
//! * **Convolutions stay f16.** Both VAEs are 3-D/2-D convs whose planes
//!   are not 2-D matrices; `--vae-quant q4tp` reshapes them to
//!   `[out, in·k·k·k]` and quantizes anyway, which is offered but not
//!   the default: the decoder is what the eye sees.
//! * **Norms and biases: exact if the source is F32, f16 otherwise.**
//!
//! The encoder's own `tokenizer_json` (32 MB) becomes the container's
//! VOCAB section, and its `hf_asset__*` blobs ride as U8 tensors, so the
//! file needs no sidecar to tokenize a prompt.

use anyhow::{Context, anyhow};
use cortiq_core::CmfModel;
use cortiq_core::format::{CmfHeader, TensorSpec, TensorSpecRef};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::Path;

use crate::convert::{encode_f16, encode_q4tp, encode_q8_row};

// ── a memory-mapped safetensors file ────────────────────────────────

struct StEntry {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

struct StFile {
    map: Mmap,
    dir: HashMap<String, StEntry>,
    order: Vec<String>,
    metadata: serde_json::Value,
}

impl StFile {
    fn open(path: &Path) -> anyhow::Result<Self> {
        let f = std::fs::File::open(path).with_context(|| path.display().to_string())?;
        // SAFETY: the packer is the only reader; a truncation under us is
        // a torn read, same as any mmap.
        let map = unsafe { Mmap::map(&f) }.with_context(|| path.display().to_string())?;
        if map.len() < 8 {
            return Err(anyhow!("{}: truncated safetensors", path.display()));
        }
        let hlen = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(
            map.get(8..8 + hlen)
                .ok_or_else(|| anyhow!("{}: header past EOF", path.display()))?,
        )?;
        let base = 8 + hlen;
        let mut dir = HashMap::new();
        let mut order = Vec::new();
        let mut metadata = serde_json::Value::Null;
        for (name, meta) in header.as_object().ok_or_else(|| anyhow!("header"))? {
            if name == "__metadata__" {
                metadata = meta.clone();
                continue;
            }
            let offs = meta["data_offsets"]
                .as_array()
                .ok_or_else(|| anyhow!("{name}: data_offsets"))?;
            dir.insert(
                name.clone(),
                StEntry {
                    dtype: meta["dtype"].as_str().unwrap_or("F32").to_string(),
                    shape: meta["shape"]
                        .as_array()
                        .ok_or_else(|| anyhow!("{name}: shape"))?
                        .iter()
                        .map(|v| v.as_u64().unwrap_or(0) as usize)
                        .collect(),
                    start: offs[0].as_u64().unwrap_or(0) as usize + base,
                    end: offs[1].as_u64().unwrap_or(0) as usize + base,
                },
            );
            order.push(name.clone());
        }
        order.sort();
        Ok(Self {
            map,
            dir,
            order,
            metadata,
        })
    }

    fn raw(&self, name: &str) -> Option<&[u8]> {
        let e = self.dir.get(name)?;
        self.map.get(e.start..e.end)
    }

    /// One tensor as f32. bf16/f16/f32 only — every LTX-2.5 release file
    /// is one of those (the int8/nvfp4 builds are not sources here).
    fn get(&self, name: &str) -> anyhow::Result<Vec<f32>> {
        let e = self
            .dir
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))?;
        let raw = self
            .map
            .get(e.start..e.end)
            .ok_or_else(|| anyhow!("{name}: span past EOF"))?;
        Ok(match e.dtype.as_str() {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| cortiq_core::quant::f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16))
                .collect(),
            other => return Err(anyhow!("{name}: unsupported dtype {other}")),
        })
    }
}

// ── codec policy ────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
enum Level {
    Q4tp,
    Q8,
    F16,
    F32,
}

fn parse_level(s: &str) -> anyhow::Result<Level> {
    Ok(match s {
        "q4tp" | "q4" => Level::Q4tp,
        "q8" => Level::Q8,
        "f16" => Level::F16,
        "f32" => Level::F32,
        other => return Err(anyhow!("--quant {other}: expected q4tp, q8, f16 or f32")),
    })
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Encode one tensor at `level`, falling back where the codec's shape
/// constraints are not met (q4tp needs 2-D and `cols % 32 == 0`).
fn spec(name: String, vals: &[f32], shape: Vec<usize>, level: Level) -> TensorSpec {
    let two_d = shape.len() == 2;
    let (dtype, data) = match level {
        Level::Q4tp if two_d && shape[1] % 32 == 0 => {
            (TensorDtype::Q4TiledP, encode_q4tp(vals, shape[0], shape[1]))
        }
        Level::Q4tp | Level::Q8 if two_d => {
            (TensorDtype::Q8Row, encode_q8_row(vals, shape[0], shape[1]))
        }
        Level::F32 => (TensorDtype::F32, f32_bytes(vals)),
        _ => (TensorDtype::F16, encode_f16(vals)),
    };
    TensorSpec {
        name,
        dtype,
        shape,
        data,
    }
}

/// The adaLN modulation tables and the tiny learned scalars: read once a
/// step, applied to everything downstream. Kept exact.
fn is_exact_plane(name: &str) -> bool {
    name.contains("scale_shift_table")
        || name.contains("prompt_scale_shift")
        || name.contains("audio_scale_shift")
        || name.ends_with("learnable_registers")
        || name.ends_with("keyframes_abs_pos_embedding")
        || name.contains("per_channel_statistics")
        || name.ends_with("layer_scalar")
        // The adaLN-single stacks. Their output is not a residual: it is the
        // scale and shift every block applies to every token, so a codec
        // error here is not averaged away by anything downstream — it is
        // multiplied into the whole stream. Measured against the reference,
        // quantizing these three projections put 3.6e-2 of relative error
        // into the very first normalization of block 0, while the attention
        // and feed-forward planes around them cost 3e-3. They are 420 M of
        // 21 B weights: 0.6 GB to keep exact, and worth every byte.
        || name.contains("adaln_single")
}

struct Policy {
    /// codec for the big 2-D planes
    level: Level,
    /// below this many weights a 2-D plane stays f16 (gates, patchify,
    /// projections in and out — a few MB in total, and every one of them
    /// is read at full sequence length)
    min_q4tp: usize,
    /// convolutions: f16 by default, q4tp by reshaping to [out, in·k…]
    conv_level: Level,
}

/// Pack one safetensors component under `dst` (`dit.`, `te.`, …).
/// `strip` is removed from every source name first. Returns
/// (tensors, weights, bytes written).
fn pack_component(
    specs: &mut Vec<TensorSpec>,
    path: &Path,
    strip: &str,
    dst: &str,
    pol: &Policy,
    vocab: &mut Option<Vec<u8>>,
) -> anyhow::Result<(usize, u64, u64, serde_json::Value)> {
    let st = StFile::open(path)?;
    let mut n = 0usize;
    let (mut weights, mut bytes) = (0u64, 0u64);
    let names = st.order.clone();
    for name in &names {
        let e = &st.dir[name];
        let numel: usize = e.shape.iter().product::<usize>().max(1);
        // The encoder carries its own tokenizer and HF assets as U8 blobs.
        if e.dtype == "U8" {
            let raw = st.raw(name).unwrap_or(&[]);
            if name == "tokenizer_json" {
                if vocab.is_none() {
                    *vocab = Some(raw.to_vec());
                }
                continue;
            }
            let short = name.trim_start_matches("hf_asset__");
            specs.push(TensorSpec {
                name: format!("{dst}asset.{short}"),
                dtype: TensorDtype::U8,
                shape: vec![raw.len()],
                data: raw.to_vec(),
            });
            n += 1;
            bytes += raw.len() as u64;
            continue;
        }
        let short = name.strip_prefix(strip).unwrap_or(name);
        let out_name = format!("{dst}{short}");
        let vals = st.get(name)?;
        let conv = e.shape.len() >= 4;
        let level = if is_exact_plane(short) {
            // exact means exact: F32 sources stay F32, bf16 ones go f16
            if e.dtype == "F32" {
                Level::F32
            } else {
                Level::F16
            }
        } else if short.ends_with("embed_tokens.weight") {
            // The token table is the one plane whose error nothing dilutes:
            // it *is* the residual stream at layer zero, and it carries
            // straight through forty-eight residual additions. Measured
            // against the reference on a real prompt, q4tp put 11% into the
            // first hidden state and every one after it; q8 puts 0.5% there
            // for half a gigabyte more on a 22 GB file.
            Level::Q8
        } else if conv {
            pol.conv_level
        } else if e.shape.len() == 2 && numel >= pol.min_q4tp {
            pol.level
        } else if e.dtype == "F32" {
            Level::F32
        } else {
            Level::F16
        };
        // A convolution asked to quantize is folded to [out, in·k·k·k]:
        // the plane the codec wants, the same numbers, and the runtime
        // reads the kernel extent from the component's config.
        let (shape, level) = if conv && level == Level::Q4tp {
            let rows = e.shape[0];
            let cols = numel / rows.max(1);
            if cols % 32 == 0 {
                (vec![rows, cols], Level::Q4tp)
            } else {
                (e.shape.clone(), Level::F16)
            }
        } else {
            (e.shape.clone(), level)
        };
        let s = spec(out_name, &vals, shape, level);
        weights += numel as u64;
        bytes += s.data.len() as u64;
        specs.push(s);
        n += 1;
    }
    Ok((n, weights, bytes, st.metadata))
}

fn config_spec(prefix: &str, cfg: &serde_json::Value) -> TensorSpec {
    let raw = serde_json::to_vec(cfg).expect("config json");
    TensorSpec {
        name: format!("{prefix}.config_json"),
        dtype: TensorDtype::U8,
        shape: vec![raw.len()],
        data: raw,
    }
}

// ── driver ──────────────────────────────────────────────────────────

pub struct LtxPackArgs<'a> {
    pub out: &'a str,
    pub carry: Option<&'a str>,
    pub dit: Option<&'a str>,
    pub te: Option<&'a str>,
    pub video_vae: Option<&'a str>,
    pub audio_vae: Option<&'a str>,
    pub spatial_upscaler: Option<&'a str>,
    pub temporal_upscaler: Option<&'a str>,
    pub duration_head: Option<&'a str>,
    pub quant: &'a str,
    pub vae_quant: &'a str,
    pub min_q4tp: usize,
}

/// (source prefix to strip, destination prefix) of every component.
const COMPONENTS: [(&str, &str); 6] = [
    ("model.diffusion_model.", "dit."),
    ("", "te."),
    ("", "vvae."),
    ("audio_vae.", "avae."),
    ("", "ups."),
    ("", "upt."),
];

pub fn cmd_ltx_pack(args: LtxPackArgs<'_>) -> anyhow::Result<()> {
    let level = parse_level(args.quant)?;
    let conv_level = parse_level(args.vae_quant)?;
    let pol = Policy {
        level,
        min_q4tp: args.min_q4tp,
        conv_level,
    };
    let t0 = std::time::Instant::now();
    let mut specs: Vec<TensorSpec> = Vec::new();
    let mut prov = serde_json::Map::new();
    let mut vocab: Option<Vec<u8>> = None;
    let mut ltx_cfg = serde_json::Value::Null;

    // One component pass: pack, report, record provenance.
    fn run(
        specs: &mut Vec<TensorSpec>,
        path: Option<&str>,
        idx: usize,
        tag: &str,
        vocab: &mut Option<Vec<u8>>,
        prov: &mut serde_json::Map<String, serde_json::Value>,
        pol: &Policy,
        t0: std::time::Instant,
    ) -> anyhow::Result<Option<serde_json::Value>> {
        let Some(p) = path else { return Ok(None) };
        let (strip, dst) = COMPONENTS[idx];
        let (n, w, b, meta) = pack_component(specs, Path::new(p), strip, dst, pol, vocab)?;
        eprintln!(
            "{tag}: {n} tensors, {:.3} B weights → {:.2} GB ({:.1}s)",
            w as f64 / 1e9,
            b as f64 / (1u64 << 30) as f64,
            t0.elapsed().as_secs_f64()
        );
        prov.insert(
            tag.to_string(),
            serde_json::json!({"source": p, "tensors": n, "weights": w}),
        );
        Ok(Some(meta))
    }

    if let Some(meta) = run(
        &mut specs, args.dit, 0, "dit", &mut vocab, &mut prov, &pol, t0,
    )? {
        // The single-file release carries the whole pipeline config in the
        // safetensors metadata; it is the spec the runtime is written
        // against, so it rides in the container verbatim.
        if let Some(cfg) = meta.get("config").and_then(|c| c.as_str()) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(cfg) {
                ltx_cfg = v;
                specs.push(config_spec("ltx", &ltx_cfg));
            }
        }
        if let Some(v) = meta.get("model_version").and_then(|c| c.as_str()) {
            prov.insert("model_version".into(), serde_json::json!(v));
        }
    }
    run(
        &mut specs, args.te, 1, "te", &mut vocab, &mut prov, &pol, t0,
    )?;
    if let Some(meta) = run(
        &mut specs,
        args.video_vae,
        2,
        "vvae",
        &mut vocab,
        &mut prov,
        &pol,
        t0,
    )? {
        // the VAE file carries its own `config.vae` block; a VAE-only pass
        // must still leave the runtime a config to build the decoder from
        if let Some(cfg) = meta.get("config").and_then(|c| c.as_str()) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(cfg) {
                specs.push(config_spec("vvae", &v));
            }
        } else if meta.get("config").is_some() {
            specs.push(config_spec("vvae", &meta["config"]));
        }
    }
    if let Some(meta) = run(
        &mut specs,
        args.audio_vae,
        3,
        "avae",
        &mut vocab,
        &mut prov,
        &pol,
        t0,
    )? {
        // the audio VAE carries the vocoder's own geometry — upsample rates,
        // kernel sizes, sampling rates — which the runtime cannot infer
        if let Some(cfg) = meta.get("config").and_then(|c| c.as_str()) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(cfg) {
                specs.push(config_spec("avae", &v));
            }
        } else if meta.get("config").is_some() {
            specs.push(config_spec("avae", &meta["config"]));
        }
    }
    run(
        &mut specs,
        args.spatial_upscaler,
        4,
        "ups",
        &mut vocab,
        &mut prov,
        &pol,
        t0,
    )?;
    run(
        &mut specs,
        args.temporal_upscaler,
        5,
        "upt",
        &mut vocab,
        &mut prov,
        &pol,
        t0,
    )?;
    if let Some(p) = args.duration_head {
        let (n, w, b, _) = pack_component(
            &mut specs,
            Path::new(p),
            "",
            "dhead.",
            &Policy {
                level: Level::F16,
                min_q4tp: usize::MAX,
                conv_level: Level::F16,
            },
            &mut vocab,
        )?;
        eprintln!("dhead: {n} tensors, {w} weights → {b} bytes");
        prov.insert(
            "dhead".into(),
            serde_json::json!({"source": p, "tensors": n}),
        );
    }

    // ── carry the previous pass through, byte for byte ──
    let carried = match args.carry {
        Some(p) => Some(CmfModel::open(p).map_err(|e| anyhow!("open {p}: {e}"))?),
        None => None,
    };
    let mut refs: Vec<TensorSpecRef> = Vec::with_capacity(specs.len() + 4096);
    for s in &specs {
        refs.push(TensorSpecRef {
            name: s.name.clone(),
            dtype: s.dtype,
            shape: s.shape.clone(),
            data: &s.data,
        });
    }
    // A re-packed component replaces its WHOLE namespace: a 48-layer
    // encoder packed over a carried 48-layer one is fine name by name,
    // but a shorter build would leave dead tail layers in the file.
    let mut drop_pre: Vec<&str> = Vec::new();
    for (on, pre) in [
        (args.dit.is_some(), "dit."),
        (args.te.is_some(), "te."),
        (args.video_vae.is_some(), "vvae."),
        (args.audio_vae.is_some(), "avae."),
        (args.spatial_upscaler.is_some(), "ups."),
        (args.temporal_upscaler.is_some(), "upt."),
        (args.duration_head.is_some(), "dhead."),
    ] {
        if on {
            drop_pre.push(pre);
        }
    }
    let mut carried_cfg: Option<serde_json::Value> = None;
    let mut carried_prov: Option<serde_json::Value> = None;
    if let Some(m) = &carried {
        let names: std::collections::HashSet<&str> =
            specs.iter().map(|s| s.name.as_str()).collect();
        for e in &m.tensors {
            if names.contains(e.name.as_str()) || drop_pre.iter().any(|p| e.name.starts_with(p)) {
                continue;
            }
            if e.name == "ltx.config_json" && ltx_cfg.is_null() {
                carried_cfg = serde_json::from_slice(m.entry_bytes(e)).ok();
            }
            refs.push(TensorSpecRef {
                name: e.name.clone(),
                dtype: e.dtype,
                shape: e.shape.clone(),
                data: m.entry_bytes(e),
            });
        }
        if vocab.is_none() {
            vocab = m.vocab.clone();
        }
        carried_prov = m.header.provenance.clone();
    }
    refs.sort_by(|a, b| a.name.cmp(&b.name));
    if ltx_cfg.is_null() {
        if let Some(c) = carried_cfg {
            ltx_cfg = c;
        }
    }

    // ── header ──
    // The arch block describes the DiT: it is the stack that defines the
    // pipeline, and `cortiq info` is read by people deciding whether the
    // file fits their card.
    let t = &ltx_cfg["transformer"];
    let g = |k: &str, d: u64| t[k].as_u64().unwrap_or(d);
    let layers = g("num_layers", 48) as usize;
    let heads = g("num_attention_heads", 32) as usize;
    let head_dim = g("attention_head_dim", 128) as usize;
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "ltx-2.5-av",
        "hidden_size": heads * head_dim,
        "intermediate_size": 0,
        "num_layers": layers,
        "num_attention_heads": heads,
        "num_kv_heads": heads,
        "head_dim": head_dim,
        "vocab_size": 262144,
        "layer_types": vec!["FullAttention"; layers],
        "rms_norm_eps": t["norm_eps"].as_f64().unwrap_or(1e-6),
        "rope_theta": t["positional_embedding_theta"].as_f64().unwrap_or(10000.0),
        "max_position_embeddings": 262144,
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
    }))?;
    let mut provenance = serde_json::json!({
        "pipeline": "ltx-2.5",
        "components": {
            "dit": "ltx-2.5 22b audio-video dit (AVTransformer3DModel, 48 blocks, video 4096 / audio 2048, embeddings connectors)",
            "te": "gemma-4 12b prompt encoder with the video/audio aggregate projections and the vision tower",
            "vvae": "ltx-2.5 video vae (3-D conv, 32x spatial / 8x temporal)",
            "avae": "ltx-2.5 audio vae",
            "ups": "latent spatial upscaler x2",
            "upt": "latent temporal upscaler x2",
            "dhead": "duration head",
        },
        "quant": {"weights": args.quant, "conv": args.vae_quant, "min_q4tp": args.min_q4tp},
        "sources": prov,
    });
    // Passes accumulate: a later pass keeps what earlier ones recorded.
    if let Some(serde_json::Value::Object(old)) = carried_prov {
        if let (Some(oldsrc), Some(newsrc)) = (
            old.get("sources").and_then(|s| s.as_object()),
            provenance["sources"].as_object_mut(),
        ) {
            for (k, v) in oldsrc {
                newsrc.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    }
    if !ltx_cfg.is_null() {
        provenance["ltx_config"] = serde_json::json!("see the ltx.config_json tensor");
    }
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch,
        quant_type: match level {
            Level::Q8 => QuantType::Q8Row,
            Level::F16 => QuantType::F16,
            Level::F32 => QuantType::F32,
            Level::Q4tp => QuantType::Q4Block,
        },
        provenance: Some(provenance),
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
    };
    CmfModel::write_ref(args.out, &header, &refs, None, vocab.as_deref())
        .map_err(|e| anyhow!("write {}: {e}", args.out))?;
    let size = std::fs::metadata(args.out)?.len();
    println!(
        "{}: {} tensors, {:.2} GB in {:.1}s",
        args.out,
        refs.len(),
        size as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
