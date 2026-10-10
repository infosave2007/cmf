//! q8_2f routed experts on native Metal (`gpu_metal::moe_q8`): the decode
//! graph's jobs kernels and the MoE chunk stage against a scalar reference
//! read straight from the q8_2f payload (int8 · row scale, x · input field).
//!
//! Every expert carries its OWN input field, so a kernel that applied one
//! job's x·col to another (the 0.8.14 per-op bug) or read the fields at the
//! wrong blob offset fails these by orders of magnitude.

#![cfg(target_os = "macos")]

use cortiq_core::format::{CmfHeader, TensorSpec};
use cortiq_core::quant::{f16_to_f32, f32_to_f16};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use cortiq_core::{CMF_VERSION, CmfModel};
use std::sync::Arc;

const EXPERTS: usize = 8;
// Mellum-like shapes, scaled down; both cross the r4 kernels' unrolled
// 128-float4 loop and leave a tail.
const HIDDEN: usize = 576;
const INTER: usize = 640;

/// [rows×cols i8][rows f16 row scales][cols f16 input field] and the decoded
/// (row scales, input field).
fn q8f_payload(rows: usize, cols: usize, seed: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut out = Vec::with_capacity(rows * cols + rows * 2 + cols * 2);
    for r in 0..rows {
        for c in 0..cols {
            out.push((((seed * 31 + r * 13 + c * 7) % 255) as i32 - 127) as i8 as u8);
        }
    }
    let mut rs = Vec::with_capacity(rows);
    for r in 0..rows {
        let h = f32_to_f16(0.002 + ((seed + r) % 7) as f32 * 0.0005);
        out.extend_from_slice(&h.to_le_bytes());
        rs.push(f16_to_f32(h));
    }
    let mut col = Vec::with_capacity(cols);
    for c in 0..cols {
        let h = f32_to_f16(0.5 + ((seed * 11 + c * 3) % 17) as f32 / 16.0);
        out.extend_from_slice(&h.to_le_bytes());
        col.push(f16_to_f32(h));
    }
    (out, rs, col)
}

fn header() -> CmfHeader {
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "moe-q8f-metal-test",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_layers": 1,
        "num_attention_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 4,
        "vocab_size": 8,
        "layer_types": ["FullAttention"],
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 8,
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
    }))
    .unwrap();
    CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        arch,
        quant_type: QuantType::Q8_2f,
        provenance: None,
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
    }
}

/// One q8_2f tensor's reference: the int8 rows, row scales and input field.
struct RefW {
    q: Vec<i8>,
    rs: Vec<f32>,
    col: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl RefW {
    /// y[r] = rs[r] · Σ_c q[r,c] · (x[c] · col[c]) in f64.
    fn mv(&self, x: &[f64]) -> Vec<f64> {
        (0..self.rows)
            .map(|r| {
                let row = &self.q[r * self.cols..(r + 1) * self.cols];
                let s: f64 = row
                    .iter()
                    .zip(x)
                    .zip(&self.col)
                    .map(|((&q, &xv), &c)| q as f64 * (xv * c as f64))
                    .sum();
                s * self.rs[r] as f64
            })
            .collect()
    }
}

struct Fixture {
    model: Arc<CmfModel>,
    trios: Vec<(usize, usize, usize)>,
    refs: Vec<[RefW; 3]>,
    dir: std::path::PathBuf,
}

fn fixture(tag: &str) -> Fixture {
    let mut specs = Vec::new();
    let mut refs = Vec::new();
    for e in 0..EXPERTS {
        let mut trio: Vec<RefW> = Vec::new();
        for (k, (part, rows, cols)) in [
            ("gate", INTER, HIDDEN),
            ("up", INTER, HIDDEN),
            ("down", HIDDEN, INTER),
        ]
        .into_iter()
        .enumerate()
        {
            let (data, rs, col) = q8f_payload(rows, cols, e * 5 + k + 1);
            trio.push(RefW {
                q: data[..rows * cols].iter().map(|&b| b as i8).collect(),
                rs,
                col,
                rows,
                cols,
            });
            specs.push(TensorSpec {
                name: format!("experts.{e}.{part}"),
                dtype: TensorDtype::Q8_2f,
                shape: vec![rows, cols],
                data,
            });
        }
        let [g, u, d]: [RefW; 3] = trio.try_into().ok().unwrap();
        refs.push([g, u, d]);
    }
    // Keep the last expert away from the end-of-file page edge (the no-copy
    // weight window is page-truncated).
    specs.push(TensorSpec {
        name: "pad".into(),
        dtype: TensorDtype::F32,
        shape: vec![8192, 2],
        data: vec![0; 8192 * 8],
    });
    let dir = std::env::temp_dir().join(format!("cmf-metal-moe-q8-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("q8f.cmf");
    CmfModel::write(&path, &header(), &specs, None, None).unwrap();
    let model = Arc::new(CmfModel::open(&path).unwrap());
    let trios = (0..EXPERTS)
        .map(|e| {
            let i = |p: &str| model.tensor_index(&format!("experts.{e}.{p}")).unwrap();
            (i("gate"), i("up"), i("down"))
        })
        .collect();
    Fixture {
        model,
        trios,
        refs,
        dir,
    }
}

fn expert_ref(r: &[RefW; 3], x: &[f64]) -> Vec<f64> {
    let g = r[0].mv(x);
    let u = r[1].mv(x);
    let a: Vec<f64> = g
        .iter()
        .zip(&u)
        .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
        .collect();
    r[2].mv(&a)
}

fn rel_rms(got: &[f32], want: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(want) {
        assert!(a.is_finite(), "non-finite GPU output");
        num += (a as f64 - b).powi(2);
        den += b.powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn metal_on() -> bool {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    cortiq_engine::gpu_metal::enabled()
}

#[test]
fn q8f_decode_jobs_match_scalar_reference() {
    if !metal_on() {
        eprintln!("skipped: no native Metal adapter");
        return;
    }
    let f = fixture("decode");
    let x: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i * 19 + 3) % 97) as f32 / 97.0 - 0.5)
        .collect();
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    // Non-contiguous winners, as a router picks them.
    let pick = [7usize, 2, 5, 0];
    let w = [0.4f32, 0.3, 0.2, 0.1];
    let trios: Vec<_> = pick.iter().map(|&e| f.trios[e]).collect();
    let mut want = vec![0.0f64; HIDDEN];
    for (&e, &we) in pick.iter().zip(&w) {
        for (dst, v) in want.iter_mut().zip(expert_ref(&f.refs[e], &x64)) {
            *dst += we as f64 * v;
        }
    }
    let run = |gu: bool, dn: bool| {
        cortiq_engine::gpu_metal::moe_q8::decode_block_for_test(
            &f.model, &trios, &x, &w, HIDDEN, INTER, gu, dn,
        )
        .expect("q8_2f decode kernels refused a well-formed block")
    };
    let jobs = run(false, false);
    let fused_gu = run(true, false);
    let fused_all = run(true, true);
    for (name, got) in [
        ("jobs", &jobs),
        ("fused gu", &fused_gu),
        ("fused gu+down", &fused_all),
    ] {
        let rel = rel_rms(got, &want);
        eprintln!("q8_2f decode {name}: relative RMS error {rel:.3e}");
        assert!(
            rel < 1e-5,
            "q8_2f decode {name} drifted from the reference: {rel:.3e}"
        );
    }
    drop(f.model);
    std::fs::remove_dir_all(f.dir).ok();
}

#[test]
fn q8f_chunk_stage_matches_scalar_reference() {
    if !metal_on() {
        eprintln!("skipped: no native Metal adapter");
        return;
    }
    let f = fixture("chunk");
    let b = 40usize;
    let top_k = 2usize;
    let xs: Vec<f32> = (0..b * HIDDEN)
        .map(|i| ((i * 23 + 7) % 89) as f32 / 89.0 - 0.5)
        .collect();
    // A router whose logits are far apart for every row (no near-ties
    // between the device's f32 sums and the reference).
    let router: Vec<f32> = (0..EXPERTS * HIDDEN)
        .map(|i| {
            let (e, c) = (i / HIDDEN, i % HIDDEN);
            ((e * 7 + c * 13) % 31) as f32 / 31.0 - 0.5
        })
        .collect();
    let route = |lg: &[f32]| -> (Vec<usize>, Vec<f32>) {
        let mut idx: Vec<usize> = (0..lg.len()).collect();
        idx.sort_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(2);
        let mx = lg[idx[0]];
        let p: Vec<f32> = idx.iter().map(|&e| (lg[e] - mx).exp()).collect();
        let s: f32 = p.iter().sum();
        (idx, p.iter().map(|v| v / s).collect())
    };
    let got = cortiq_engine::gpu_metal::moe_q8::chunk_block_for_test(
        &f.model,
        &router,
        &f.trios,
        Box::new(route),
        &xs,
        b,
        HIDDEN,
        INTER,
    )
    .expect("q8_2f chunk stage refused a well-formed block");
    let mut want = vec![0.0f64; b * HIDDEN];
    for bi in 0..b {
        let x = &xs[bi * HIDDEN..(bi + 1) * HIDDEN];
        let lg: Vec<f32> = (0..EXPERTS)
            .map(|e| {
                router[e * HIDDEN..(e + 1) * HIDDEN]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| (*a as f64 * *b as f64) as f32)
                    .sum()
            })
            .collect();
        let (idx, w) = route(&lg);
        assert_eq!(idx.len(), top_k);
        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        for (&e, &we) in idx.iter().zip(&w) {
            for (dst, v) in want[bi * HIDDEN..(bi + 1) * HIDDEN]
                .iter_mut()
                .zip(expert_ref(&f.refs[e], &x64))
            {
                *dst += we as f64 * v;
            }
        }
    }
    let rel = rel_rms(&got, &want);
    eprintln!("q8_2f chunk stage: relative RMS error {rel:.3e}");
    assert!(
        rel < 1e-4,
        "q8_2f chunk stage drifted from the reference: {rel:.3e}"
    );
    drop(f.model);
    std::fs::remove_dir_all(f.dir).ok();
}
