//! Pack Meta SAM 3 (`facebook/sam3`) exports into ONE quantized CMF.
//!
//! Input is the export directory produced from the upstream gated `sam3.pt`
//! (torch state dict): `detector.safetensors` + `tracker.safetensors` plus the
//! vendor `config.json`, `detector_config.json`, `tracker_config.json`,
//! `processor_config.json` and `tokenizer.json` (CLIP BPE, 49 408 tokens).
//!
//! The file carries five sections, one per runtime stage:
//!
//! - `trunk.*`  — the SAM3 vision trunk (ViT blocks with windowed RoPE);
//! - `fpn.*`    — the SAM2-style feature pyramid (`convs` + `sam2_convs`);
//! - `text.*`   — the CLIP text tower and its resizer to width 256;
//! - `det.*`    — DETR encoder/decoder, geometry encoder (points/boxes),
//!                segmentation head and concept scoring;
//! - `trk.*`    — the video tracker (memory bank, SAM mask decoder, prompts).
//!
//! Quantization policy mirrors the image packers: chosen codec for 2-D
//! projections, f16 for 4-D conv kernels, exact f32 for norms/biases and the
//! small geometry tables, q8 for the query/label/presence embeddings. The
//! whole vendor configs ride along as `cfg.*` U8 tensors, the tokenizer in
//! VOCAB.

use anyhow::{Context, anyhow};
use cortiq_core::CmfModel;
use cortiq_core::format::{CmfHeader, TensorSpec};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use std::path::Path;

use crate::convert::{encode_f16, encode_q4_tiled, encode_q8_row};

#[derive(Clone, Copy, PartialEq)]
enum Level {
    Q8,
    Q4t,
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Encode one 2-D projection at `level` (falls back q4t→q8 when the column
/// constraint is not met).
fn encode_2d(name: String, vals: &[f32], rows: usize, cols: usize, level: Level) -> TensorSpec {
    let (dtype, data) = if level == Level::Q4t && cols % 32 == 0 {
        (TensorDtype::Q4Tiled, encode_q4_tiled(vals, rows, cols))
    } else {
        (TensorDtype::Q8Row, encode_q8_row(vals, rows, cols))
    };
    TensorSpec {
        name,
        dtype,
        shape: vec![rows, cols],
        data,
    }
}

fn spec_f32(name: String, shape: Vec<usize>, vals: &[f32]) -> TensorSpec {
    TensorSpec {
        name,
        dtype: TensorDtype::F32,
        shape,
        data: f32_bytes(vals),
    }
}

fn u8_spec(name: &str, raw: Vec<u8>) -> TensorSpec {
    let shape = vec![raw.len()];
    TensorSpec {
        name: name.into(),
        dtype: TensorDtype::U8,
        shape,
        data: raw,
    }
}

/// Map a vendor tensor name onto its CMF section.
fn section(name: &str) -> Option<String> {
    const RULES: [(&str, &str); 9] = [
        ("detector.backbone.vision_backbone.trunk.", "trunk."),
        ("detector.backbone.vision_backbone.convs.", "fpn.convs."),
        ("detector.backbone.vision_backbone.sam2_convs.", "fpn.sam2_convs."),
        ("detector.backbone.language_backbone.", "text."),
        ("detector.transformer.", "det.transformer."),
        ("detector.geometry_encoder.", "det.geometry."),
        ("detector.segmentation_head.", "det.seghead."),
        ("detector.dot_prod_scoring.", "det.scoring."),
        ("tracker.", "trk."),
    ];
    for (from, to) in RULES {
        if let Some(rest) = name.strip_prefix(from) {
            return Some(format!("{to}{rest}"));
        }
    }
    None
}

/// Small, precision-sensitive tensors that never take the projection codec.
fn is_exact_table(name: &str) -> bool {
    const TABLES: [&str; 9] = [
        "pos_embed",
        "positional_embedding",
        "query_embed",
        "reference_points",
        "label_embed",
        "cls_embed",
        "presence_token",
        "no_obj_",
        "freqs_cis",
    ];
    TABLES.iter().any(|t| name.contains(t))
}

/// Stream one exported safetensors component into `specs`.
fn pack_component(
    path: &Path,
    level: Level,
    specs: &mut Vec<TensorSpec>,
) -> anyhow::Result<usize> {
    let mut n = 0usize;
    cortiq_engine::vae::read_safetensors_each(path, &mut |vendor, shape, data| {
        let name = section(vendor)
            .ok_or_else(|| format!("unmapped vendor tensor: {vendor}"))?;
        // Exact-f32 rules: 1-D (norms, biases, gains) and the small tables.
        if shape.len() == 1
            || vendor.ends_with(".bias")
            || vendor.contains("norm")
            || is_exact_table(vendor)
        {
            specs.push(spec_f32(name, shape.clone(), &data));
        } else if shape.len() == 4 {
            specs.push(TensorSpec {
                name,
                dtype: TensorDtype::F32, // promoted to f16 in the post-pass
                shape: shape.clone(),
                data: f32_bytes(&data),
            });
        } else if shape.len() == 2 {
            let (rows, cols) = (shape[0], shape[1]);
            specs.push(encode_2d(name, &data, rows, cols, level));
        } else {
            // Scalars and odd ranks stay exact.
            specs.push(spec_f32(name, shape.clone(), &data));
        }
        n += 1;
        Ok(())
    })
    .map_err(|e| anyhow!("{}: {e}", path.display()))?;
    Ok(n)
}

fn read_json(src: &Path, file: &str) -> anyhow::Result<Vec<u8>> {
    std::fs::read(src.join(file)).with_context(|| format!("{file}"))
}

/// Pack the export directory into one CMF.
pub fn pack(src: &str, quant: &str, out: &str) -> anyhow::Result<()> {
    let src = Path::new(src);
    let level = match quant {
        "q8" | "q8_2f" | "q8f" => Level::Q8,
        "q4t" | "q4tp" | "q4" => Level::Q4t,
        other => return Err(anyhow!("--quant {other}: expected q8_2f or q4tp")),
    };
    let t0 = std::time::Instant::now();
    let mut specs: Vec<TensorSpec> = Vec::new();

    let det = src.join("detector.safetensors");
    ensure_exists(&det)?;
    let n_det = pack_component(&det, level, &mut specs)?;
    eprintln!("detector packed ({n_det} tensors, {:.1}s)", t0.elapsed().as_secs_f64());

    let mut n_trk = 0usize;
    let trk = src.join("tracker.safetensors");
    if trk.exists() {
        n_trk = pack_component(&trk, level, &mut specs)?;
        eprintln!("tracker packed ({n_trk} tensors, {:.1}s)", t0.elapsed().as_secs_f64());
    }

    // f16 for the 4-D conv kernels (trunk MLPs stay 2-D; FPN kernels are 4-D).
    for s in specs.iter_mut() {
        if s.shape.len() == 4 && s.dtype == TensorDtype::F32 {
            let vals: Vec<f32> = s
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            s.data = encode_f16(&vals);
            s.dtype = TensorDtype::F16;
        }
    }

    // Vendor configs ride along for the runtime (preprocess + thresholds).
    for (file, name) in [
        ("config.json", "cfg.root"),
        ("detector_config.json", "cfg.detector"),
        ("tracker_config.json", "cfg.tracker"),
        ("processor_config.json", "cfg.processor"),
    ] {
        let p = src.join(file);
        if p.exists() {
            specs.push(u8_spec(name, read_json(src, file)?));
        }
    }

    let vocab = read_json(src, "tokenizer.json").context("tokenizer.json")?;

    // Honest header numbers, read from the vendor detector config when present.
    let det_cfg: serde_json::Value = serde_json::from_slice(&read_json(src, "detector_config.json")?)
        .unwrap_or_else(|_| serde_json::json!({}));
    let dec = &det_cfg["detr_decoder_config"];
    let txt = &det_cfg["text_config"];
    let hidden = dec["hidden_size"].as_u64().unwrap_or(256) as usize;
    let inter = dec["ffn_dim"].as_u64()
        .or_else(|| dec["intermediate_size"].as_u64())
        .unwrap_or(2048) as usize;
    let layers = dec["num_hidden_layers"].as_u64().unwrap_or(6) as usize;
    let heads = dec["num_attention_heads"].as_u64().unwrap_or(8) as usize;
    let vocab_size = txt["vocab_size"].as_u64().unwrap_or(49_408) as usize;
    let max_pos = txt["max_position_embeddings"].as_u64().unwrap_or(32) as usize;
    let eps = dec["layer_norm_eps"].as_f64().unwrap_or(1e-6);

    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "sam3",
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_layers": layers,
        "num_attention_heads": heads,
        "num_kv_heads": heads,
        "head_dim": (hidden / heads.max(1)).max(1),
        "vocab_size": vocab_size,
        "layer_types": vec!["FullAttention"; layers],
        "rms_norm_eps": eps,
        "max_position_embeddings": max_pos,
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
    }))?;

    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch,
        quant_type: match level {
            Level::Q8 => QuantType::Q8Row,
            Level::Q4t => QuantType::Q4Block,
        },
        provenance: Some(serde_json::json!({
            "pipeline": "sam3",
            "source": "facebook/sam3 (SAM License — a copy of the agreement ships beside this file)",
            "components": {
                "trunk": "SAM3 vision trunk (ViT + windowed RoPE)",
                "fpn": "SAM2-style feature pyramid",
                "text": "CLIP text tower -> resizer(1024 -> 256)",
                "det": "DETR encoder/decoder + geometry + mask head + concept scoring",
                "trk": "video tracker (memory bank + SAM mask decoder)",
            },
            "packed_from": src.display().to_string(),
            "quant": quant,
            "tensors": {"detector": n_det, "tracker": n_trk},
        })),
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

    CmfModel::write(out, &header, &specs, None, Some(&vocab))
        .map_err(|e| anyhow!("write {out}: {e}"))?;
    let size = std::fs::metadata(out)?.len();
    println!(
        "{out}: {} tensors, {:.2} GB in {:.1}s",
        specs.len(),
        size as f64 / (1 << 30) as f64,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn ensure_exists(p: &Path) -> anyhow::Result<()> {
    if p.exists() {
        Ok(())
    } else {
        Err(anyhow!(
            "{}: not found — export facebook/sam3's sam3.pt to safetensors first",
            p.display()
        ))
    }
}