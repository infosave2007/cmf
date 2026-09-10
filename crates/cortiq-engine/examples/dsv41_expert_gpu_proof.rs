//! Bounded Vulkan proof for the DSV4.1 global dynamic MoE frame.
//!
//! The fixture has four Q4TP experts.  This example installs their directory
//! triples in the same segmented global bank used by the runtime, executes
//! 24 inputs across four deterministic top-2 route cases (96 frames), and
//! compares the resident GPU result with an independent CPU BF16-boundary
//! implementation.  It deliberately reports residuals instead of inventing
//! a quality threshold.

use anyhow::{Context, ensure};
use cortiq_core::{CmfModel, TensorDtype};
use cortiq_engine::gpu;
use cortiq_engine::gpu_wgpu;
use cortiq_engine::pool::Pool;
use cortiq_engine::qtensor::QTensor;
use cortiq_engine::vae::{StTensor, read_safetensors};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;

const EXPERT_IDS: [usize; 4] = [0, 1, 10, 100];
const TOKENS: usize = 24;

#[derive(Clone, Copy, Debug)]
struct Metrics {
    n: usize,
    max_abs: f32,
    mean_abs: f64,
    rmse: f64,
    nrmse: f64,
    cosine: f64,
    finite_got: bool,
    finite_expected: bool,
}

fn measure(got: &[f32], expected: &[f32]) -> Metrics {
    let mut max_abs = 0.0f32;
    let mut abs = 0.0f64;
    let mut sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut got_sq = 0.0f64;
    let mut expected_sq = 0.0f64;
    for (&a, &b) in got.iter().zip(expected) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        abs += d as f64;
        sq += (d as f64) * (d as f64);
        dot += a as f64 * b as f64;
        got_sq += a as f64 * a as f64;
        expected_sq += b as f64 * b as f64;
    }
    let n = got.len().max(1) as f64;
    let den = (got_sq * expected_sq).sqrt();
    let expected_rms = (expected_sq / n).sqrt();
    Metrics {
        n: got.len(),
        max_abs,
        mean_abs: abs / n,
        rmse: (sq / n).sqrt(),
        nrmse: if expected_rms > 0.0 {
            (sq / n).sqrt() / expected_rms
        } else {
            f64::NAN
        },
        cosine: if den > 0.0 { dot / den } else { f64::NAN },
        finite_got: got.iter().all(|v| v.is_finite()),
        finite_expected: expected.iter().all(|v| v.is_finite()),
    }
}

#[inline]
fn bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(round) & 0xffff_0000)
}

fn bf16_slice(values: &mut [f32]) {
    for value in values {
        *value = bf16(*value);
    }
}

/// Exact V4.1 expert boundaries: gate/up matvecs become BF16 before the
/// limited SwiGLU, down becomes BF16 before routed/shared accumulation, and
/// the final sum becomes BF16.  `clamps` counts exercised limited-SwiGLU
/// components and is evidence that the branch was actually covered.
fn cpu_expert(
    gate: &QTensor,
    up: &QTensor,
    down: &QTensor,
    x: &[f32],
    weight: f32,
    inter: usize,
    limit: f32,
    pool: &Pool,
) -> (Vec<f32>, usize) {
    let mut g = vec![0.0f32; inter];
    let mut u = vec![0.0f32; inter];
    gpu::cpu_scope(|| {
        gate.matvec(x, &mut g, Some(pool));
        up.matvec(x, &mut u, Some(pool));
    });
    bf16_slice(&mut g);
    bf16_slice(&mut u);
    let mut clamps = 0usize;
    for i in 0..inter {
        if limit > 0.0 {
            let old_g = g[i];
            let old_u = u[i];
            g[i] = g[i].min(limit);
            u[i] = u[i].clamp(-limit, limit);
            clamps += usize::from(g[i] != old_g) + usize::from(u[i] != old_u);
        }
        g[i] = g[i] / (1.0 + (-g[i]).exp()) * u[i] * weight;
    }
    // Official Expert.forward uses x.to(dtype) before w2; for this BF16
    // fixture that means the weighted SwiGLU activation is rounded here.
    bf16_slice(&mut g);
    let mut out = vec![0.0f32; down.rows()];
    gpu::cpu_scope(|| down.matvec(&g, &mut out, Some(pool)));
    bf16_slice(&mut out);
    (out, clamps)
}

fn add(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

fn fixture_tensor<'a>(
    oracle: &'a std::collections::HashMap<String, StTensor>,
    name: &str,
) -> anyhow::Result<&'a StTensor> {
    oracle
        .get(name)
        .with_context(|| format!("oracle is missing {name}"))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    ensure!(
        args.len() == 3,
        "usage: dsv41_expert_gpu_proof <probe.cmf> <oracle.safetensors> <oracle.json>"
    );
    let model_path = &args[0];
    let oracle_path = &args[1];
    let report_path = &args[2];

    // The caller supplies the GPU policy.  Keep these explicit so a proof
    // cannot quietly become a CPU-only run or a timed backend probe.
    unsafe {
        std::env::set_var("CMF_GPU", "1");
        std::env::set_var("CMF_GPU_PROBE", "0");
        std::env::set_var("CMF_DSV4_GLOBAL_POOL", "1");
    }
    ensure!(gpu_wgpu::enabled(), "wgpu backend did not initialize");
    ensure!(
        gpu_wgpu::dsv4_global_moe_supported(),
        "segmented global MoE is unavailable"
    );

    let report: Value = serde_json::from_reader(
        std::fs::File::open(report_path)
            .with_context(|| format!("open {}", report_path.display()))?,
    )?;
    let dim = report["dim"].as_u64().context("oracle report has no dim")? as usize;
    let inter = report["inter_dim"]
        .as_u64()
        .context("oracle report has no inter_dim")? as usize;
    let limit = report["swiglu_limit"]
        .as_f64()
        .context("oracle report has no swiglu_limit")? as f32;
    ensure!(dim > 0 && inter > 0 && limit.is_finite() && dim % 32 == 0 && inter % 32 == 0);

    let oracle = read_safetensors(oracle_path).map_err(anyhow::Error::msg)?;
    let input = fixture_tensor(&oracle, "input")?;
    ensure!(input.shape == [TOKENS, dim] && input.data.len() == TOKENS * dim);
    ensure!(
        input.data.iter().all(|v| v.is_finite()),
        "fixture input is non-finite"
    );

    let model = Arc::new(CmfModel::open(model_path)?);
    let integrity = model.verify();
    ensure!(
        integrity.is_empty(),
        "probe CMF integrity errors: {integrity:?}"
    );
    let mut q = Vec::with_capacity(EXPERT_IDS.len());
    let mut triples = Vec::with_capacity(EXPERT_IDS.len());
    for &expert in &EXPERT_IDS {
        let prefix = format!("model.layers.0.mlp.experts.{expert}");
        let gate = QTensor::from_model(&model, &format!("{prefix}.gate_proj.weight"))
            .map_err(anyhow::Error::msg)?;
        let up = QTensor::from_model(&model, &format!("{prefix}.up_proj.weight"))
            .map_err(anyhow::Error::msg)?;
        let down = QTensor::from_model(&model, &format!("{prefix}.down_proj.weight"))
            .map_err(anyhow::Error::msg)?;
        ensure!(gate.model_dtype() == Some(TensorDtype::Q4TiledP));
        ensure!(up.model_dtype() == Some(TensorDtype::Q4TiledP));
        ensure!(down.model_dtype() == Some(TensorDtype::Q4TiledP));
        ensure!(gate.rows() == inter && gate.cols() == dim);
        ensure!(up.rows() == inter && up.cols() == dim);
        ensure!(down.rows() == dim && down.cols() == inter);
        triples.push((
            gate.model_idx().context("gate has no model index")?,
            up.model_idx().context("up has no model index")?,
            down.model_idx().context("down has no model index")?,
        ));
        q.push((gate, up, down));
    }

    // Reserve only a handful of tiny fixture slots.  The full runtime uses
    // this exact API with its bounded model-wide Q4TP bank.
    // `dsv4_global_moe_create` reserves a fixed workspace from its request,
    // so a literal request of 16 is rounded away on a large card.  Ask for
    // exactly that reserve plus sixteen fixture slots; the resulting bank is
    // still tiny and cannot consume the production pool budget.
    let gib = 1024u64 * 1024 * 1024;
    let budget = gpu_wgpu::dsv4_vram_budget().context("GPU VRAM budget unavailable")?;
    let workspace = (budget / 10).clamp(2 * gib, 4 * gib);
    let gu_len = cortiq_core::quant::expected_nbytes(TensorDtype::Q4TiledP, &[inter, dim])
        .context("Q4TP gate geometry has no byte size")?;
    let down_len = cortiq_core::quant::expected_nbytes(TensorDtype::Q4TiledP, &[dim, inter])
        .context("Q4TP down geometry has no byte size")?;
    let per_expert = 2usize
        .checked_mul(gu_len)
        .and_then(|n| n.checked_add(down_len))
        .context("fixture expert size overflow")?;
    let request = (workspace / per_expert as u64) as usize + 16;
    let (capacity, segment_slots) =
        gpu_wgpu::dsv4_global_moe_create(&model, request, inter, dim, false)
            .context("failed to create tiny global MoE bank")?;
    ensure!(capacity >= 8 && segment_slots > 0);
    for (slot, triple) in triples.iter().copied().enumerate() {
        ensure!(
            gpu_wgpu::dsv4_global_slot_fill(&model, slot, triple),
            "fill route slot {slot} failed"
        );
    }
    // Shared semantics are kept explicit.  The small fixture has no separate
    // shared tensor, so use expert 10's exact directory triple as a shared
    // branch while retaining its own pinned slot and unit weight.
    let shared_slot = triples.len();
    ensure!(
        gpu_wgpu::dsv4_global_slot_fill(&model, shared_slot, triples[2]),
        "fill shared slot failed"
    );

    let route_cases: [([usize; 2], [f32; 2]); 4] = [
        ([0, 1], [0.25, 0.75]),
        ([1, 2], [0.50, 0.50]),
        ([2, 3], [0.90, 0.10]),
        ([3, 0], [1.00, 0.00]),
    ];
    let pool = Pool::new(3);
    let mut got_all = Vec::with_capacity(TOKENS * route_cases.len() * dim);
    let mut expected_all = Vec::with_capacity(got_all.capacity());
    let mut per_case_got: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(TOKENS * dim));
    let mut per_case_expected: [Vec<f32>; 4] =
        std::array::from_fn(|_| Vec::with_capacity(TOKENS * dim));
    let mut total_clamps = 0usize;
    let mut frame_count = 0usize;
    let mut cold_frames = 0usize;
    let fills_before = gpu_wgpu::DSV4_FILLS.load(std::sync::atomic::Ordering::Relaxed);
    let gpu_before = gpu_wgpu::MOE_GPU_N.load(std::sync::atomic::Ordering::Relaxed);
    for token in 0..TOKENS {
        let x = &input.data[token * dim..(token + 1) * dim];
        for (case_index, (picks, route_weights)) in route_cases.iter().copied().enumerate() {
            let mut weights = vec![0.0f32; EXPERT_IDS.len()];
            weights[picks[0]] = route_weights[0];
            weights[picks[1]] = route_weights[1];
            let mut logits = vec![0.0f32; EXPERT_IDS.len()];
            logits[picks[0]] = 1.0;
            logits[picks[1]] = 0.5;
            let remap = vec![0u32, 1, 2, 3];
            let w = gpu_wgpu::Dsv4MoeW {
                router: &[],
                experts: &triples,
                logits: &logits,
                bias: Some(&weights),
                mask: None,
                forced: Some(&picks),
                remap: Some(&remap),
                global: Some(gpu_wgpu::Dsv4GlobalMoe {
                    pool_uid: model.uid(),
                    shared_slot: shared_slot as u32,
                    segment_slots: segment_slots as u32,
                }),
                has_shared: true,
                shared_weight: 1.0,
                preweighted: true,
                qwen_softmax: false,
            };
            let geom = gpu_wgpu::Dsv4MoeGeom {
                hidden: dim,
                inter,
                top_k: 2,
                route_scale: 1.0,
                swiglu_limit: limit,
                gu_q2: false,
                bf16: true,
            };
            let mut got = vec![0.0f32; dim];
            let mut cold = Vec::new();
            let mut cold_x = Vec::new();
            ensure!(
                gpu_wgpu::dsv4_moe_frame(
                    &model,
                    &w,
                    geom,
                    x,
                    &mut cold,
                    &mut cold_x,
                    None,
                    None,
                    &mut got
                ),
                "GPU frame failed at token {token} picks={picks:?}"
            );
            ensure!(
                cold.is_empty(),
                "resident proof unexpectedly returned cold experts: {cold:?}"
            );
            // The frame exports its normalized activation alongside the
            // cold list unconditionally; the runtime only consumes it when
            // `cold` is non-empty.  An empty cold list is the no-fallback
            // proof, while a correctly sized activation confirms the readback
            // contract remained intact.
            ensure!(
                cold_x.len() == dim,
                "GPU frame returned malformed activation: {}",
                cold_x.len()
            );
            let mut expected = vec![0.0f32; dim];
            for (&expert, &weight) in picks.iter().zip(route_weights.iter()) {
                let (value, clamps) = cpu_expert(
                    &q[expert].0,
                    &q[expert].1,
                    &q[expert].2,
                    x,
                    weight,
                    inter,
                    limit,
                    &pool,
                );
                total_clamps += clamps;
                add(&mut expected, &value);
            }
            let (shared, clamps) =
                cpu_expert(&q[2].0, &q[2].1, &q[2].2, x, 1.0, inter, limit, &pool);
            total_clamps += clamps;
            add(&mut expected, &shared);
            bf16_slice(&mut expected);
            got_all.extend_from_slice(&got);
            expected_all.extend_from_slice(&expected);
            per_case_got[case_index].extend_from_slice(&got);
            per_case_expected[case_index].extend_from_slice(&expected);
            frame_count += 1;
            cold_frames += usize::from(!cold.is_empty());
        }
    }
    let m = measure(&got_all, &expected_all);
    let fills_after = gpu_wgpu::DSV4_FILLS.load(std::sync::atomic::Ordering::Relaxed);
    let gpu_after = gpu_wgpu::MOE_GPU_N.load(std::sync::atomic::Ordering::Relaxed);
    ensure!(
        m.finite_got && m.finite_expected,
        "non-finite expert proof output"
    );
    ensure!(frame_count == 96 && cold_frames == 0);
    for (case_index, (picks, _)) in route_cases.iter().enumerate() {
        let case = measure(&per_case_got[case_index], &per_case_expected[case_index]);
        ensure!(
            case.finite_got && case.finite_expected,
            "non-finite case {case_index}"
        );
        println!(
            "case={} picks={:?} frames={} max_abs={:.8e} rmse={:.8e} nrmse={:.8e} cosine={:.9} finite={}",
            case_index,
            picks,
            TOKENS,
            case.max_abs,
            case.rmse,
            case.nrmse,
            case.cosine,
            case.finite_got && case.finite_expected,
        );
    }
    // Exercise the mixed path once per route shape.  Mark the second winner
    // cold in the remap even though its bytes were filled earlier; this is a
    // deterministic way to prove the frame exports the selected cold id and
    // that host completion can be added to the resident GPU result.  The
    // completion is explicitly inside cpu_scope, matching production's fixed
    // cold-expert closure and preventing a tiny matvec from re-entering GPU.
    let mixed_gpu_before = gpu_wgpu::MOE_GPU_N.load(std::sync::atomic::Ordering::Relaxed);
    let mut mixed_got = Vec::with_capacity(route_cases.len() * dim);
    let mut mixed_expected = Vec::with_capacity(route_cases.len() * dim);
    let mut mixed_cold = 0usize;
    for (case_index, (picks, route_weights)) in route_cases.iter().copied().enumerate() {
        let x = &input.data[case_index * dim..(case_index + 1) * dim];
        let mut weights = vec![0.0f32; EXPERT_IDS.len()];
        weights[picks[0]] = route_weights[0];
        weights[picks[1]] = route_weights[1];
        let mut logits = vec![0.0f32; EXPERT_IDS.len()];
        logits[picks[0]] = 1.0;
        logits[picks[1]] = 0.5;
        let cold_expert = picks[1];
        let mut remap = vec![0u32, 1, 2, 3];
        remap[cold_expert] = u32::MAX;
        let w = gpu_wgpu::Dsv4MoeW {
            router: &[],
            experts: &triples,
            logits: &logits,
            bias: Some(&weights),
            mask: None,
            forced: Some(&picks),
            remap: Some(&remap),
            global: Some(gpu_wgpu::Dsv4GlobalMoe {
                pool_uid: model.uid(),
                shared_slot: shared_slot as u32,
                segment_slots: segment_slots as u32,
            }),
            has_shared: true,
            shared_weight: 1.0,
            preweighted: true,
            qwen_softmax: false,
        };
        let geom = gpu_wgpu::Dsv4MoeGeom {
            hidden: dim,
            inter,
            top_k: 2,
            route_scale: 1.0,
            swiglu_limit: limit,
            gu_q2: false,
            bf16: true,
        };
        let mut got = vec![0.0f32; dim];
        let mut cold = Vec::new();
        let mut cold_x = Vec::new();
        ensure!(
            gpu_wgpu::dsv4_moe_frame(
                &model,
                &w,
                geom,
                x,
                &mut cold,
                &mut cold_x,
                None,
                None,
                &mut got
            ),
            "mixed GPU frame failed at case {case_index}"
        );
        ensure!(
            cold.len() == 1 && cold[0].0 == cold_expert,
            "mixed frame cold ids: {cold:?}"
        );
        ensure!(
            cold_x.len() == dim,
            "mixed frame activation length {}",
            cold_x.len()
        );
        mixed_cold += cold.len();
        let (cold_value, _) = gpu::cpu_scope(|| {
            cpu_expert(
                &q[cold_expert].0,
                &q[cold_expert].1,
                &q[cold_expert].2,
                x,
                route_weights[1],
                inter,
                limit,
                &pool,
            )
        });
        add(&mut got, &cold_value);
        bf16_slice(&mut got);
        let mut expected = vec![0.0f32; dim];
        for (&expert, &weight) in picks.iter().zip(route_weights.iter()) {
            let (value, _) = cpu_expert(
                &q[expert].0,
                &q[expert].1,
                &q[expert].2,
                x,
                weight,
                inter,
                limit,
                &pool,
            );
            add(&mut expected, &value);
        }
        let (shared, _) = cpu_expert(&q[2].0, &q[2].1, &q[2].2, x, 1.0, inter, limit, &pool);
        add(&mut expected, &shared);
        bf16_slice(&mut expected);
        mixed_got.extend_from_slice(&got);
        mixed_expected.extend_from_slice(&expected);
    }
    let mixed = measure(&mixed_got, &mixed_expected);
    let mixed_gpu_after = gpu_wgpu::MOE_GPU_N.load(std::sync::atomic::Ordering::Relaxed);
    ensure!(mixed.finite_got && mixed.finite_expected && mixed_cold == route_cases.len());
    println!(
        "proof=mixed-resident-cold frames={} cold_returns={} elements={} max_abs={:.8e} mean_abs={:.8e} rmse={:.8e} nrmse={:.8e} cosine={:.9} finite={} gpu_frames_delta={}",
        route_cases.len(),
        mixed_cold,
        mixed.n,
        mixed.max_abs,
        mixed.mean_abs,
        mixed.rmse,
        mixed.nrmse,
        mixed.cosine,
        mixed.finite_got && mixed.finite_expected,
        mixed_gpu_after.saturating_sub(mixed_gpu_before),
    );
    println!(
        "proof=resident-global-dsv41 frames={} elements={} max_abs={:.8e} mean_abs={:.8e} rmse={:.8e} nrmse={:.8e} cosine={:.9} finite={} clamps={} fills_delta={} gpu_frames_delta={} capacity={} segment_slots={}",
        frame_count,
        m.n,
        m.max_abs,
        m.mean_abs,
        m.rmse,
        m.nrmse,
        m.cosine,
        m.finite_got && m.finite_expected,
        total_clamps,
        fills_after.saturating_sub(fills_before),
        gpu_after.saturating_sub(gpu_before),
        capacity,
        segment_slots,
    );
    println!(
        "gpu_timing_ns enc={} wait={} pass={} gpu={}",
        gpu_wgpu::MOE_ENC_NS.load(std::sync::atomic::Ordering::Relaxed),
        gpu_wgpu::MOE_WAIT_NS.load(std::sync::atomic::Ordering::Relaxed),
        gpu_wgpu::MOE_PASS_NS.load(std::sync::atomic::Ordering::Relaxed),
        gpu_wgpu::MOE_GPU_NS[0].load(std::sync::atomic::Ordering::Relaxed),
    );
    Ok(())
}
