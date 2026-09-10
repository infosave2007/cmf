//! Compare the isolated V4.1 Engram path with the released component oracle.
//!
//! The remote fixture directory is produced from the pinned official
//! safetensors by `export_engram_component_raw.py`. It contains only the
//! Engram tensors and the official per-case input/output arrays; the helper
//! writes a small CMF once, then exercises the same mmap-backed row lookup
//! used by the full loader.

use cortiq_core::{CmfModel, TensorDtype, TensorSpec};
use cortiq_engine::dsv41::{Dsv41Cfg, Dsv41Engram, RawFp8Rows, dsv41_apply_engram_for_test};
use cortiq_engine::qtensor::QTensor;
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn read_bf16(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err(format!("{} has odd BF16 byte length", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect())
}

fn write_fixture(dir: &Path, cmf: &Path) -> Result<(), Box<dyn Error>> {
    if cmf.exists() {
        return Ok(());
    }
    let base = CmfModel::open(dir.parent().unwrap().join("../tiny-reference-f16.cmf"))?;
    let layer = dir
        .file_name()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_prefix("raw-layer-"))
        .ok_or("fixture directory must be named raw-layer-N")?;
    let prefix = format!("model.layers.{layer}.engram");
    let mut tensors = Vec::new();
    let push = |tensors: &mut Vec<TensorSpec>,
                name: &str,
                dtype: TensorDtype,
                shape: &[usize],
                file: &str| {
        tensors.push(TensorSpec {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            data: fs::read(dir.join(file)).expect("raw Engram fixture is readable"),
        });
    };
    push(
        &mut tensors,
        &format!("{prefix}.embed.weight"),
        TensorDtype::U8,
        &[664, 256],
        "embed_weight.bin",
    );
    push(
        &mut tensors,
        &format!("{prefix}.embed.scale"),
        TensorDtype::U8,
        &[664, 8],
        "embed_scale.bin",
    );
    push(
        &mut tensors,
        &format!("{prefix}.wkv.weight"),
        TensorDtype::F32,
        &[25600, 6144],
        "wkv_weight.bin",
    );
    push(
        &mut tensors,
        &format!("{prefix}.q_weight"),
        TensorDtype::F32,
        &[4, 5120],
        "q_weight.bin",
    );
    push(
        &mut tensors,
        &format!("{prefix}.k_weight"),
        TensorDtype::F32,
        &[4, 5120],
        "k_weight.bin",
    );
    CmfModel::write(cmf, &base.header, &tensors, None, None)?;
    Ok(())
}

fn f32_tensor(model: &CmfModel, name: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let entry = model
        .tensor(name)
        .ok_or_else(|| format!("missing {name}"))?;
    let bytes = model.entry_bytes(entry);
    if bytes.len() % 4 != 0 {
        return Err(format!("{name} is not F32 bytes").into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn cfg() -> Dsv41Cfg {
    Dsv41Cfg {
        dim: 5120,
        n_heads: 20,
        head_dim: 128,
        rope_head_dim: 64,
        q_lora_rank: 1536,
        o_lora_rank: 512,
        o_groups: 4,
        hc_mult: 4,
        hc_sinkhorn_iters: 20,
        hc_eps: 1e-6,
        norm_eps: 1e-20,
        n_routed_experts: 256,
        top_k: 8,
        moe_inter: 1536,
        gate_temp: 1.0,
        norm_topk_prob: true,
        route_scale: 2.5,
        swiglu_limit: 7.0,
        window: 128,
        rope_theta: 10_000.0,
        compress_rope_theta: 160_000.0,
        rope_factor: 1.0,
        original_seq_len: 65_536,
        beta_fast: 32.0,
        beta_slow: 1.0,
        index_heads: 32,
        index_head_dim: 64,
        index_topk: 64,
        candidate_source: 3,
        candidate_topk_blocks: 2,
        candidate_block_size: 8,
        kv_sources: vec![1, 3],
        index_sources: vec![1, 3],
        compress_ratios: vec![0, 2, 2, 1, 1],
        engram_layers: vec![1, 14],
        engram_vocab: 16_000_000,
        engram_embeddings: vec![16_000_000, 16_000_000],
        engram_max_ngram: 4,
        engram_heads: 8,
        engram_head_dim: 256,
        engram_compressed_vocab: 99_092,
        engram_pad_id: 2,
        vocab: 129_280,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let dir = PathBuf::from(args.next().ok_or("missing raw fixture directory")?);
    let layer = dir
        .file_name()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_prefix("raw-layer-"))
        .ok_or("fixture directory must be named raw-layer-N")?;
    let prefix = format!("model.layers.{layer}.engram");
    let cmf = dir.join("engram-component.cmf");
    write_fixture(&dir, &cmf)?;
    let model = Arc::new(CmfModel::open(&cmf)?);
    let embed = RawFp8Rows::from_model(
        &model,
        &format!("{prefix}.embed.weight"),
        &format!("{prefix}.embed.scale"),
    )?;
    let wkv = QTensor::from_model(&model, &format!("{prefix}.wkv.weight"))?;
    let q_weight = f32_tensor(&model, &format!("{prefix}.q_weight"))?;
    let k_weight = f32_tensor(&model, &format!("{prefix}.k_weight"))?;
    let engram = Dsv41Engram {
        embed,
        wkv,
        q_weight,
        k_weight,
    };
    let cfg = cfg();
    let meta: Value = serde_json::from_slice(&fs::read(dir.join("manifest.json"))?)?;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut count = 0usize;
    let mut cases = 0usize;
    for case in 0..3 {
        let prefix = format!("case{case}");
        let input_shape = meta[&format!("{prefix}.input")]["shape"]
            .as_array()
            .unwrap();
        let seq = input_shape[1].as_u64().unwrap() as usize;
        let input = read_bf16(&dir.join(format!("{prefix}_input.bin")))?;
        let expected = read_bf16(&dir.join(format!("{prefix}_output.bin")))?;
        let index_bytes = fs::read(dir.join(format!("{prefix}_indices.bin")))?;
        let mut indices = Vec::with_capacity(index_bytes.len() / 8);
        for b in index_bytes.chunks_exact(8) {
            indices.push(u64::from_le_bytes(b.try_into().unwrap()) as usize);
        }
        let token_mask: Vec<bool> = if case == 1 {
            vec![
                true, true, true, true, true, false, false, false, true, true, true, true, true,
                true, true, true, true, true, true, true,
            ]
        } else {
            vec![true; seq]
        };
        for token in 0..seq {
            let mut h = input[token * 4 * 5120..(token + 1) * 4 * 5120].to_vec();
            let hashes = &indices[token * 24..(token + 1) * 24];
            dsv41_apply_engram_for_test(&engram, &mut h, hashes, &cfg, token_mask[token]);
            for (&got, &want) in h
                .iter()
                .zip(&expected[token * 4 * 5120..(token + 1) * 4 * 5120])
            {
                let d = (got - want).abs();
                max_abs = max_abs.max(d);
                sum_abs += d as f64;
                count += 1;
            }
        }
        cases += 1;
        println!("case={case} seq={seq} cumulative_max_abs={max_abs:.8e}",);
    }
    println!(
        "summary cases={cases} values={count} max_abs={max_abs:.8e} mean_abs={:.8e}",
        sum_abs / count.max(1) as f64
    );
    Ok(())
}
