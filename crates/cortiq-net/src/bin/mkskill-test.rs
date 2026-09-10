//! Build a PARITY-ORACLE skill: `skill.test.*` tensors that are byte
//! copies of the layer-0 FFN trio. Running `--skill test` must produce
//! output IDENTICAL to the backbone — which makes it the exact oracle
//! for validating skill plumbing (network split, overlay resolution)
//! without training anything.
//!
//! Usage: mkskill-test <in.cmf> <out.cmf>

use cortiq_core::{CmfModel, SkillRecord, TensorSpec};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let inp = args.next().expect("usage: mkskill-test <in.cmf> <out.cmf>");
    let out = args.next().expect("usage: mkskill-test <in.cmf> <out.cmf>");

    let model = CmfModel::open_sharded(&inp)?;
    let ffn0: Vec<String> = model
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("model.layers.0.") && t.name.contains("mlp."))
        .map(|t| t.name.clone())
        .collect();
    if ffn0.is_empty() {
        return Err("no model.layers.0.*mlp.* tensors — wrong arch naming?".into());
    }
    println!(
        "oracle skill 'test' over {} layer-0 FFN tensors:",
        ffn0.len()
    );
    for n in &ffn0 {
        println!("  {n}");
    }

    let mut tensors: Vec<TensorSpec> = Vec::with_capacity(model.tensors.len() + ffn0.len());
    for t in &model.tensors {
        if t.name.starts_with("skill.test.") {
            continue; // idempotent re-run
        }
        tensors.push(TensorSpec {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: model.tensor_bytes(&t.name)?.to_vec(),
        });
    }
    for n in &ffn0 {
        let e = model
            .tensors
            .iter()
            .find(|t| &t.name == n)
            .expect("listed above");
        tensors.push(TensorSpec {
            name: format!("skill.test.{n}"),
            dtype: e.dtype,
            shape: e.shape.clone(),
            data: model.tensor_bytes(n)?.to_vec(),
        });
    }

    let mut header = model.header.clone();
    header.skills.retain(|s| s.id != "test");
    header.skills.push(SkillRecord {
        id: "test".to_string(),
        name: Some("parity oracle (byte-copy of layer-0 FFN)".to_string()),
        layers: vec![0],
        selection: None,
        input_mask_task: None,
        quality: None,
        base_dir_hash: None,
        base_arch: None,
        task: None,
        provenance: Some(serde_json::json!({
            "recipe": "mkskill-test: byte-copy oracle, output must equal backbone",
        })),
    });

    CmfModel::write(
        &out,
        &header,
        &tensors,
        if model.masks.masks.is_empty() {
            None
        } else {
            Some(&model.masks)
        },
        model.vocab.as_deref(),
    )?;
    let check = CmfModel::open(&out)?;
    assert!(check.skill_tensors("test").count() == ffn0.len());
    println!("✓ wrote {out} (dir_hash {:016x})", check.dir_hash());
    Ok(())
}
