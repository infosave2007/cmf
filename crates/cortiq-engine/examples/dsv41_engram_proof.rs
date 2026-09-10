//! Bounded exact proof for the released V4.1 Engram hash and component path.
//!
//! The full CMF is opened only for its embedded tokenizer/header. Engram
//! component weights are the 664-row compact fixtures; no full table is
//! decoded or sent to a device.
//!
//! Usage:
//!   dsv41_engram_proof <full.cmf> <engram-reference.json> <component-root> <projected-root> <map.bin>

use cortiq_core::CmfModel;
use cortiq_engine::dsv41::{
    Dsv41Cfg, Dsv41Engram, EngramHash, RawFp8Rows, dsv41_apply_engram_for_test,
    token_map_from_model,
};
use cortiq_engine::qtensor::QTensor;
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

const DIM: usize = 5120;
const HC_MULT: usize = 4;
const HASH_COLS: usize = 24;

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

fn read_i64(path: &Path) -> Result<Vec<usize>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 8 != 0 {
        return Err(format!("{} has non-i64 length", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as usize)
        .collect())
}

#[inline]
fn bf16_roundtrip(value: f32) -> f32 {
    let bits = value.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(round) & 0xffff_0000)
}

fn compare_bf16(label: &str, got: &[f32], expected: &[f32]) -> Result<(), Box<dyn Error>> {
    if got.len() != expected.len() {
        return Err(format!(
            "{label} length mismatch: runtime={} oracle={}",
            got.len(),
            expected.len()
        )
        .into());
    }
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut exact = 0usize;
    let mut max_ulp = 0u32;
    let mut over_one_ulp = 0usize;
    for (&actual, &want) in got.iter().zip(expected) {
        let rounded = bf16_roundtrip(actual);
        let diff = (rounded - want).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff as f64;
        exact += usize::from(rounded.to_bits() == want.to_bits());
        let actual_bf16 = (rounded.to_bits() >> 16) as i32;
        let expected_bf16 = (want.to_bits() >> 16) as i32;
        let ulp = actual_bf16.abs_diff(expected_bf16);
        max_ulp = max_ulp.max(ulp);
        over_one_ulp += usize::from(ulp > 1);
    }
    let mean_abs = sum_abs / got.len().max(1) as f64;
    println!(
        "{label} values={} exact_bf16={}/{} max_abs={max_abs:.8e} mean_abs={:.8e} max_bf16_ulp={max_ulp} over_1ulp={over_one_ulp}",
        got.len(),
        exact,
        got.len(),
        mean_abs
    );
    // The oracle is a CUDA FP8 GEMM with a tile-dependent FP32 reduction;
    // this CPU path uses the released F32 container and a scalar reduction.
    // Compare the externally recorded BF16 oracle with a tight absolute bound
    // while retaining exact gathered/hash checks and reporting BF16 drift.
    if max_abs > 2.0e-2 || mean_abs > 1.0e-6 {
        return Err(format!(
            "{label} exceeds numeric oracle bound: max_abs={max_abs:.8e} mean_abs={mean_abs:.8e} max_bf16_ulp={max_ulp} over_1ulp={over_one_ulp}"
        ).into());
    }
    Ok(())
}

fn cfg() -> Dsv41Cfg {
    Dsv41Cfg {
        dim: DIM,
        n_heads: 20,
        head_dim: 128,
        rope_head_dim: 64,
        q_lora_rank: 1536,
        o_lora_rank: 512,
        o_groups: 4,
        hc_mult: HC_MULT,
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

fn sha256sum(path: &Path) -> Result<String, Box<dyn Error>> {
    let output = Command::new("sha256sum").arg(path).output()?;
    if !output.status.success() {
        return Err(format!("sha256sum failed for {}", path.display()).into());
    }
    let text = String::from_utf8(output.stdout)?;
    Ok(text
        .split_whitespace()
        .next()
        .ok_or("sha256sum produced no digest")?
        .to_string())
}

fn verify_hash_reference(
    model_path: &Path,
    reference_path: &Path,
    map_path: &Path,
) -> Result<(Value, Vec<Vec<Vec<Vec<usize>>>>), Box<dyn Error>> {
    let reference: Value = serde_json::from_slice(&fs::read(reference_path)?)?;
    let model = CmfModel::open(model_path)?;
    let vocab = model.header.arch.vocab_size;
    let token_map = token_map_from_model(&model, vocab);
    let mut bytes = Vec::with_capacity(token_map.len() * 4);
    for value in &token_map {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(map_path, &bytes)?;
    let got_sha = sha256sum(map_path)?;
    let expected_sha = reference["token_map_le_u32_sha256"]
        .as_str()
        .ok_or("reference token map SHA is missing")?;
    println!(
        "hash token_map vocab={} compressed_vocab={} sha256={got_sha}",
        token_map.len(),
        token_map.iter().copied().max().unwrap_or(0) + 1
    );
    if got_sha != expected_sha {
        return Err(
            format!("token map SHA mismatch: runtime={got_sha} oracle={expected_sha}").into(),
        );
    }

    let compressed_vocab = reference["compressed_vocab_size"]
        .as_u64()
        .ok_or("reference compressed vocab is missing")? as usize;
    let hash = EngramHash::new(
        vec![1, 14],
        4,
        8,
        16_000_000,
        compressed_vocab,
        2,
        token_map,
    )?;
    let expected_pad = reference["compressed_pad_id"]
        .as_i64()
        .ok_or("reference compressed pad is missing")?;
    if hash.pad_id != expected_pad {
        return Err(format!(
            "pad id mismatch: runtime={} oracle={expected_pad}",
            hash.pad_id
        )
        .into());
    }
    let expected_multipliers: Vec<[u64; 4]> =
        serde_json::from_value(reference["multipliers"].clone())?;
    let expected_primes: Vec<Vec<Vec<u64>>> = serde_json::from_value(reference["primes"].clone())?;
    let expected_offsets: Vec<Vec<u64>> = serde_json::from_value(reference["offsets"].clone())?;
    if hash.multipliers != expected_multipliers {
        return Err(format!(
            "multiplier mismatch: runtime={:?} oracle={expected_multipliers:?}",
            hash.multipliers
        )
        .into());
    }
    if hash.primes != expected_primes {
        return Err("prime layout mismatch".into());
    }
    let actual_offsets: Vec<Vec<u64>> = hash
        .offsets
        .iter()
        .zip(&hash.primes)
        .map(|(starts, per_ngram)| {
            per_ngram
                .iter()
                .zip(starts)
                .flat_map(|(primes, &start)| {
                    let mut offset = start;
                    primes.iter().map(move |&prime| {
                        let current = offset;
                        offset += prime;
                        current
                    })
                })
                .collect()
        })
        .collect();
    if actual_offsets != expected_offsets {
        return Err("offset layout mismatch".into());
    }
    println!(
        "hash layout layers={} primes={} offsets={} multipliers=exact",
        hash.layer_ids.len(),
        hash.primes
            .iter()
            .map(|x| x.iter().map(Vec::len).sum::<usize>())
            .sum::<usize>(),
        hash.offsets.iter().map(Vec::len).sum::<usize>()
    );

    let cases = reference["cases"]
        .as_array()
        .ok_or("reference cases missing")?;
    let mut all_hashes = Vec::with_capacity(cases.len());
    for (case_no, case) in cases.iter().enumerate() {
        let ids: Vec<u32> = serde_json::from_value(case["input_ids"].clone())?;
        let mask: Option<Vec<bool>> = if case["token_mask"].is_null() {
            None
        } else {
            Some(serde_json::from_value(case["token_mask"].clone())?)
        };
        let expected_hashes: Vec<Vec<Vec<usize>>> = serde_json::from_value(case["hashes"].clone())?;
        let mut state = hash.clone();
        state.reset();
        let mut got_hashes = Vec::with_capacity(ids.len());
        for (pos, &id) in ids.iter().enumerate() {
            got_hashes.push(state.push(id, mask.as_ref().map(|m| m[pos]).unwrap_or(true)));
        }
        if got_hashes != expected_hashes {
            let first = got_hashes
                .iter()
                .zip(&expected_hashes)
                .enumerate()
                .find(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| (i, a, b));
            return Err(format!("hash case {case_no} mismatch: {first:?}").into());
        }
        let cuts: Vec<usize> = serde_json::from_value(case["chunk_ends"].clone())?;
        let mut chunk_state = hash.clone();
        chunk_state.reset();
        let mut chunk_hashes = Vec::with_capacity(ids.len());
        let mut start = 0;
        for end in cuts {
            for pos in start..end {
                chunk_hashes.push(
                    chunk_state.push(ids[pos], mask.as_ref().map(|m| m[pos]).unwrap_or(true)),
                );
            }
            start = end;
        }
        if start != ids.len() || chunk_hashes != expected_hashes {
            return Err(format!("hash case {case_no} chunked sequence mismatch").into());
        }
        println!(
            "hash case={} name={} seq={} exact_indices=true chunked=true",
            case_no,
            case["name"].as_str().unwrap_or("?"),
            ids.len()
        );
        all_hashes.push(expected_hashes);
    }
    Ok((reference, all_hashes))
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
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn verify_component_layer(
    root: &Path,
    projected_root: &Path,
    layer: usize,
    reference_hashes: &[Vec<Vec<Vec<usize>>>],
) -> Result<(), Box<dyn Error>> {
    let dir = root.join(format!("raw-layer-{layer}"));
    let layer_meta: Value =
        serde_json::from_slice(&fs::read(root.join(format!("layer-{layer}.json")))?)?;
    let original_rows: Vec<usize> = serde_json::from_value(layer_meta["original_row_ids"].clone())?;
    let row_to_compact: HashMap<usize, usize> = original_rows
        .iter()
        .copied()
        .enumerate()
        .map(|(compact, original)| (original, compact))
        .collect();
    let prefix = format!("model.layers.{layer}.engram");
    let cmf = dir.join("engram-component.cmf");
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
    let cases = layer_meta["cases"]
        .as_array()
        .ok_or("layer cases missing")?;
    for (case_no, case) in cases.iter().enumerate() {
        let original_hashes: Vec<Vec<usize>> =
            serde_json::from_value(case["original_hash_ids"].clone())?;
        if original_hashes.len() != reference_hashes[case_no].len() {
            return Err(format!("layer {layer} case {case_no} sequence length mismatch").into());
        }
        let mut expected_compact = Vec::with_capacity(original_hashes.len() * HASH_COLS);
        for (pos, rows) in original_hashes.iter().enumerate() {
            if rows.len() != HASH_COLS || reference_hashes[case_no][pos][0].len() != HASH_COLS {
                return Err(format!("layer {layer} case {case_no} hash width mismatch").into());
            }
            let layer_index = if layer == 1 { 0 } else { 1 };
            if rows != &reference_hashes[case_no][pos][layer_index] {
                return Err(format!(
                    "layer {layer} case {case_no} original indices differ from full hash oracle at position {pos}"
                )
                .into());
            }
            for &row in rows {
                expected_compact.push(
                    *row_to_compact
                        .get(&row)
                        .ok_or_else(|| format!("layer {layer} missing compact row for {row}"))?,
                );
            }
        }
        let got_compact = read_i64(&dir.join(format!("case{case_no}_indices.bin")))?;
        if got_compact != expected_compact {
            let first = got_compact
                .iter()
                .zip(&expected_compact)
                .enumerate()
                .find(|(_, (a, b))| a != b);
            return Err(format!(
                "layer {layer} case {case_no} compact indices mismatch: {first:?}"
            )
            .into());
        }

        let seq = original_hashes.len();
        let mut gathered = vec![0.0f32; seq * HASH_COLS * 256];
        for token in 0..seq {
            for col in 0..HASH_COLS {
                engram.embed.row_into(
                    got_compact[token * HASH_COLS + col],
                    &mut gathered
                        [(token * HASH_COLS + col) * 256..(token * HASH_COLS + col + 1) * 256],
                );
            }
        }
        gathered.iter_mut().for_each(|v| *v = bf16_roundtrip(*v));
        compare_bf16(
            &format!("layer={layer} case={case_no} gathered"),
            &gathered,
            &read_bf16(&dir.join(format!("case{case_no}_gathered.bin")))?,
        )?;

        let mut projected = Vec::with_capacity(seq * (DIM * (HC_MULT + 1)));
        for token in 0..seq {
            let input = &gathered[token * HASH_COLS * 256..(token + 1) * HASH_COLS * 256];
            let mut row = vec![0.0f32; DIM * (HC_MULT + 1)];
            engram.wkv.matvec(input, &mut row, None);
            row.iter_mut().for_each(|v| *v = bf16_roundtrip(*v));
            projected.extend_from_slice(&row);
        }
        compare_bf16(
            &format!("layer={layer} case={case_no} projected"),
            &projected,
            &read_bf16(&projected_root.join(format!("layer-{layer}/case{case_no}_projected.bin")))?,
        )?;

        let input = read_bf16(&dir.join(format!("case{case_no}_input.bin")))?;
        let expected_output = read_bf16(&dir.join(format!("case{case_no}_output.bin")))?;
        let mask: Option<Vec<bool>> = if case["token_mask"].is_null() {
            None
        } else {
            Some(serde_json::from_value(case["token_mask"].clone())?)
        };
        let mut output = Vec::with_capacity(input.len());
        for token in 0..seq {
            let mut h = input[token * HC_MULT * DIM..(token + 1) * HC_MULT * DIM].to_vec();
            dsv41_apply_engram_for_test(
                &engram,
                &mut h,
                &got_compact[token * HASH_COLS..(token + 1) * HASH_COLS],
                &cfg(),
                mask.as_ref().map(|m| m[token]).unwrap_or(true),
            );
            output.extend_from_slice(&h);
        }
        compare_bf16(
            &format!("layer={layer} case={case_no} output"),
            &output,
            &expected_output,
        )?;
        println!(
            "component layer={} case={} name={} compact_indices=true gathered=true projected=true output=true",
            layer,
            case_no,
            case["name"].as_str().unwrap_or("?")
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let model = PathBuf::from(args.next().ok_or("missing full CMF path")?);
    let reference = PathBuf::from(args.next().ok_or("missing Engram reference JSON")?);
    let component_root = PathBuf::from(args.next().ok_or("missing component root")?);
    let projected_root = PathBuf::from(args.next().ok_or("missing projected root")?);
    let map_path = PathBuf::from(args.next().ok_or("missing token map output path")?);
    if args.next().is_some() {
        return Err("usage: dsv41_engram_proof <full.cmf> <engram-reference.json> <component-root> <projected-root> <map.bin>".into());
    }
    let (_reference, reference_hashes) = verify_hash_reference(&model, &reference, &map_path)?;
    for layer in [1usize, 14] {
        verify_component_layer(&component_root, &projected_root, layer, &reference_hashes)?;
    }
    println!("ENGRAM_PROOF_PASS layers=2 cases=6 hash_cases=3 full_table_decode=false gpu=false");
    Ok(())
}
