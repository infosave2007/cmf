//! Assemble already-qualified tower groups without requantization or touching
//! the 164-GB text backbone. BASE supplies inventory/config; VISION supplies
//! visual.* and AUDIO supplies audio_encoder.*, speech_embeddings.* and
//! audio_tokenizer.encoder.*. Writes a NEW file only, with source provenance.
use cortiq_core::{CmfModel, format::TensorSpec};
use cortiq_engine::mimo_mm::{MimoMm, MimoTowerGroup, is_codebook};
use serde_json::json;
use std::{collections::BTreeMap, path::Path, sync::Arc};

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 5,
        "usage: mimo_mm_assemble BASE VISION AUDIO OUTPUT"
    );
    anyhow::ensure!(
        !Path::new(&args[4]).exists(),
        "refusing to replace an existing output"
    );
    let base = Arc::new(CmfModel::open(&args[1])?);
    MimoMm::from_model(&base).map_err(anyhow::Error::msg)?;
    let vision = CmfModel::open(&args[2])?;
    let audio = CmfModel::open(&args[3])?;
    let mut specs = Vec::with_capacity(base.tensors.len());
    let mut codecs: BTreeMap<String, BTreeMap<String, BTreeMap<String, usize>>> = BTreeMap::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut bytes: BTreeMap<String, u64> = BTreeMap::new();
    for e in &base.tensors {
        let group = MimoTowerGroup::of(&e.name);
        let src = match group {
            Some(MimoTowerGroup::Vision) => &vision,
            Some(_) => &audio,
            None => &*base,
        };
        let si = src
            .tensor_index(&e.name)
            .ok_or_else(|| anyhow::anyhow!("missing source tensor {}", e.name))?;
        let se = &src.tensors[si];
        anyhow::ensure!(se.shape == e.shape, "source shape mismatch for {}", e.name);
        let data = src.tensor_bytes(&e.name)?.to_vec();
        if let Some(g) = group {
            let cat = if is_codebook(&e.name) {
                "codebooks"
            } else if g == MimoTowerGroup::SpeechEmbeddings {
                "tables"
            } else if e.shape.len() == 2 {
                "matrices"
            } else {
                "other"
            };
            *codecs
                .entry(g.label().into())
                .or_default()
                .entry(cat.into())
                .or_default()
                .entry(se.dtype.name().into())
                .or_default() += 1;
            *counts.entry(g.label().into()).or_default() += 1;
            *bytes.entry(g.label().into()).or_default() += data.len() as u64;
        }
        specs.push(TensorSpec {
            name: e.name.clone(),
            dtype: se.dtype,
            shape: e.shape.clone(),
            data,
        });
    }
    let mut header = base.header.clone();
    let provenance = header.provenance.get_or_insert_with(|| json!({}));
    let mm = &mut provenance["mimo_mm"];
    // Same compact codec form as the converter.
    let mut codec_json = json!({});
    for (group, categories) in codecs {
        codec_json[&group] = json!({});
        for (category, hist) in categories {
            codec_json[&group][&category] = if hist.len() == 1 {
                json!(hist.keys().next().unwrap())
            } else {
                json!(hist)
            };
        }
    }
    mm["codec"] = codec_json;
    mm["tensor_counts"] = json!(counts);
    mm["payload_bytes"] = json!(bytes);
    mm["gates"] = json!({}); // Combined end-to-end gates must be rerun, not inherited.
    mm["assembly"] = json!({"method":"byte-for-byte group replacement; no requantization",
        "vision_source":Path::new(&args[2]).file_name().unwrap().to_string_lossy(),
        "audio_source":Path::new(&args[3]).file_name().unwrap().to_string_lossy(),
        "vision_provenance":vision.header.provenance,"audio_provenance":audio.header.provenance});
    CmfModel::write(&args[4], &header, &specs, None, None)?;
    let assembled = MimoMm::open(&args[4]).map_err(anyhow::Error::msg)?;
    println!(
        "validated {} ({} tensors): {:?}",
        args[4],
        specs.len(),
        assembled.source
    );
    Ok(())
}
