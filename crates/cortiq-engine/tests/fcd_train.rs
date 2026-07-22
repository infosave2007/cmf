//! FCD trainer end-to-end checks on a tiny synthetic .cmf:
//! 1. block-level gradcheck — finite differences over TRAINABLE weights
//!    through the WHOLE training graph (teacher + student + CE/KL),
//!    which transitively exercises every through-grad (attention, rope,
//!    qk-norm, GQA reduce, residuals, final norm, tied head);
//! 2. training smoke — loss decreases over 20 steps, the best
//!    checkpoint is restored, the polished file loads and forwards.

use cortiq_core::format::TensorSpec;
use cortiq_core::{CmfHeader, CmfModel, LayerType, ModelArch, NormStyle, QuantType, TensorDtype};
use cortiq_engine::Pipeline;
use cortiq_engine::SamplerConfig;
use cortiq_engine::fcd::{FcdHyper, FcdModel, TrainState, run_polish};
use cortiq_engine::nystrom::O1Cfg;
use std::sync::Arc;

const H: usize = 16;
const NH: usize = 2;
const NKV: usize = 1; // rep = 2 → the GQA sum-reduce path is live
const HD: usize = 8;
const NL: usize = 2;
const INTER: usize = 24;
const VOCAB: usize = 48;
const SEQ: usize = 16; // > w + sink + 8 = 13 → skeleton active

fn synth_f32(n: usize, salt: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(salt.wrapping_mul(1442695040888963407) ^ 0x9E3779B97F4A7C15);
            let x = (x ^ (x >> 31)).wrapping_mul(0xBF58476D1CE4E5B9);
            (((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32) * scale
        })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn tiny_arch() -> ModelArch {
    ModelArch {
        arch_name: "qwen3".into(),
        hidden_size: H,
        intermediate_size: INTER,
        num_layers: NL,
        num_attention_heads: NH,
        num_kv_heads: NKV,
        head_dim: HD,
        vocab_size: VOCAB,
        layer_types: vec![LayerType::FullAttention; NL],
        rms_norm_eps: 1e-6,
        norm_style: NormStyle::Qwen,
        rope_theta: 10_000.0,
        tie_word_embeddings: true,
        partial_rotary_factor: 1.0,
        yarn: None,
        attention_heads_per_layer: None,
        local_partial_rotary_factor: None,
        mtp: None,
        moe: None,
        linear_core: None,
        max_position_embeddings: 4096,
        linear_conv_kernel_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        hidden_act: "silu".into(),
        embed_multiplier: 1.0,
        query_pre_attn_scalar: None,
        sliding_window: None,
        sliding_window_pattern: None,
        rope_local_base_freq: None,
        global_head_dim: None,
        num_global_kv_heads: None,
        global_partial_rotary_factor: None,
        final_logit_softcapping: None,
        attn_v_norm: false,
    }
}

/// GDN geometry of the tiny hybrid fixture (rep = nv/nk = 2).
const G_NV: usize = 2;
const G_NK: usize = 1;
const G_DK: usize = 4;
const G_DV: usize = 4;
const G_KK: usize = 3;

/// Write a tiny all-f32 model (with per-head qk-norms — that path must
/// be exercised) and return its path. `gate` adds the Qwen3.5 output
/// gate (q_proj rows = 2·nh·hd); `hybrid` makes layer 1 GatedDeltaNet.
fn write_tiny_model_variant(dir: &std::path::Path, name: &str, gate: bool, hybrid: bool) -> std::path::PathBuf {
    let mut specs: Vec<TensorSpec> = Vec::new();
    let mut push = |name: &str, shape: Vec<usize>, data: Vec<f32>| {
        specs.push(TensorSpec {
            name: name.into(),
            dtype: TensorDtype::F32,
            shape,
            data: f32_bytes(&data),
        });
    };
    push("model.embed_tokens.weight", vec![VOCAB, H], synth_f32(VOCAB * H, 100, 0.6));
    let norm1 = |n: usize, salt: u64| -> Vec<f32> { synth_f32(n, salt, 0.2).iter().map(|v| 1.0 + v).collect() };
    push("model.norm.weight", vec![H], norm1(H, 101));
    for li in 0..NL {
        let p = format!("model.layers.{li}.");
        let s = li as u64 * 20;
        push(&format!("{p}input_layernorm.weight"), vec![H], norm1(H, 102 + s));
        push(&format!("{p}post_attention_layernorm.weight"), vec![H], norm1(H, 103 + s));
        if hybrid && li == 1 {
            // GatedDeltaNet layer (frozen in FCD; through-backward only).
            let c_dim = 2 * G_NK * G_DK + G_NV * G_DV;
            let vd = G_NV * G_DV;
            let la = format!("{p}linear_attn.");
            push(&format!("{la}in_proj_qkv.weight"), vec![c_dim, H], synth_f32(c_dim * H, 130 + s, 0.5));
            push(&format!("{la}in_proj_z.weight"), vec![vd, H], synth_f32(vd * H, 131 + s, 0.5));
            push(&format!("{la}in_proj_a.weight"), vec![G_NV, H], synth_f32(G_NV * H, 132 + s, 0.5));
            push(&format!("{la}in_proj_b.weight"), vec![G_NV, H], synth_f32(G_NV * H, 133 + s, 0.5));
            push(&format!("{la}conv1d.weight"), vec![c_dim, G_KK], synth_f32(c_dim * G_KK, 134 + s, 0.6));
            push(&format!("{la}A_log"), vec![G_NV], synth_f32(G_NV, 135 + s, 0.8));
            push(&format!("{la}dt_bias"), vec![G_NV], synth_f32(G_NV, 136 + s, 0.8));
            push(&format!("{la}norm.weight"), vec![G_DV], norm1(G_DV, 137 + s));
            push(&format!("{la}out_proj.weight"), vec![H, vd], synth_f32(H * vd, 138 + s, 0.5));
            push(&format!("{p}mlp.gate_proj.weight"), vec![INTER, H], synth_f32(INTER * H, 110 + s, 0.5));
            push(&format!("{p}mlp.up_proj.weight"), vec![INTER, H], synth_f32(INTER * H, 111 + s, 0.5));
            push(&format!("{p}mlp.down_proj.weight"), vec![H, INTER], synth_f32(H * INTER, 112 + s, 0.5));
            continue;
        }
        let q_rows = if gate { 2 * NH * HD } else { NH * HD };
        push(&format!("{p}self_attn.q_proj.weight"), vec![q_rows, H], synth_f32(q_rows * H, 104 + s, 0.5));
        push(&format!("{p}self_attn.k_proj.weight"), vec![NKV * HD, H], synth_f32(NKV * HD * H, 105 + s, 0.5));
        push(&format!("{p}self_attn.v_proj.weight"), vec![NKV * HD, H], synth_f32(NKV * HD * H, 106 + s, 0.5));
        push(&format!("{p}self_attn.o_proj.weight"), vec![H, NH * HD], synth_f32(H * NH * HD, 107 + s, 0.5));
        push(&format!("{p}self_attn.q_norm.weight"), vec![HD], norm1(HD, 108 + s));
        push(&format!("{p}self_attn.k_norm.weight"), vec![HD], norm1(HD, 109 + s));
        push(&format!("{p}mlp.gate_proj.weight"), vec![INTER, H], synth_f32(INTER * H, 110 + s, 0.5));
        push(&format!("{p}mlp.up_proj.weight"), vec![INTER, H], synth_f32(INTER * H, 111 + s, 0.5));
        push(&format!("{p}mlp.down_proj.weight"), vec![H, INTER], synth_f32(H * INTER, 112 + s, 0.5));
    }
    let mut arch = tiny_arch();
    if hybrid {
        arch.layer_types = vec![LayerType::FullAttention, LayerType::LinearAttention];
        arch.linear_core = Some(cortiq_core::types::LinearCoreConfig {
            kind: "gated_delta_net".into(),
            num_heads: G_NV,
            nphase: None,
            value_head_dim: G_DV,
        });
        arch.linear_num_key_heads = Some(G_NK);
        arch.linear_num_value_heads = Some(G_NV);
        arch.linear_key_head_dim = Some(G_DK);
        arch.linear_conv_kernel_dim = Some(G_KK);
    }
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::format::CMF_VERSION,
        arch,
        quant_type: QuantType::F32,
        provenance: None,
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
    };
    let path = dir.join(format!("{name}.cmf"));
    CmfModel::write(&path, &header, &specs, None, None).expect("write tiny model");
    path
}

fn write_tiny_model(dir: &std::path::Path) -> std::path::PathBuf {
    write_tiny_model_variant(dir, "tiny_fcd", false, false)
}

fn tiny_o1() -> O1Cfg {
    O1Cfg::from_spec("all", Some(4), Some(4), Some(1), None).unwrap()
}

fn rand_ids(n: usize, salt: u64) -> Vec<u32> {
    synth_f32(n, salt, 1.0).iter().map(|v| (((v + 0.5) * VOCAB as f32) as u32).min(VOCAB as u32 - 1)).collect()
}

/// FD over top-|g| trainable weights through the whole training graph.
/// Returns (per-tensor worst rel err). `only` filters which tensor
/// slots are asserted; the rest are printed.
fn block_fd(fm: &FcdModel, ts: &mut TrainState, ids: &[u32], tgt: &[u32], t: usize, tol: f64, only: &dyn Fn(usize) -> bool) -> f64 {
    let kl_w = 0.7;
    let l0 = fm.loss_and_grads_for_test(ids, tgt, 1, t, ts, kl_w);
    assert!(l0.is_finite());
    let analytic: Vec<Vec<f32>> = ts.grads().to_vec();
    let h = 1e-3f32;
    let mut worst: f64 = 0.0;
    for (pi, g) in analytic.iter().enumerate() {
        let mut order: Vec<usize> = (0..g.len()).collect();
        order.sort_by(|&a, &b| g[b].abs().partial_cmp(&g[a].abs()).unwrap());
        for &i in order.iter().take(3) {
            let ga = g[i] as f64;
            let orig = ts.data[pi][i];
            ts.data[pi][i] = orig + h;
            let lp = fm.loss_and_grads_for_test(ids, tgt, 1, t, ts, kl_w);
            ts.data[pi][i] = orig - h;
            let lm = fm.loss_and_grads_for_test(ids, tgt, 1, t, ts, kl_w);
            ts.data[pi][i] = orig;
            let fd = (lp - lm) / (2.0 * h as f64);
            let rel = (ga - fd).abs() / ga.abs().max(fd.abs()).max(1e-8);
            if only(pi) {
                worst = worst.max(rel);
                assert!(rel < tol, "tensor {pi} idx {i}: analytic {ga:.5e} vs fd {fd:.5e} (rel {rel:.2e})");
            } else {
                println!("  (unasserted, M-gap zone) tensor {pi} idx {i}: rel {rel:.2e}");
            }
        }
    }
    worst
}

/// Full-graph FD with the skeleton INACTIVE (w large → the kernel's
/// exact fallback): there is no frozen M anywhere, so EVERY trainable
/// tensor must match tightly. This exercises the whole assembly —
/// checkpointed recompute, projections, qk-norm, rope, GQA reduce,
/// residuals, final norm, tied head, CE+KL.
#[test]
fn block_gradcheck_full_graph_exact_mode() {
    let dir = std::env::temp_dir().join(format!("fcd_block_ex_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = write_tiny_model(&dir);
    let model = CmfModel::open(&path).unwrap();
    let o1 = O1Cfg::from_spec("all", Some(4), Some(64), Some(0), None).unwrap(); // w ≥ t → exact
    let fm = FcdModel::from_cmf(&model, &o1).unwrap();
    let mut ts = TrainState::new(&fm);
    let seqs = rand_ids(SEQ + 1, 7);
    let worst = block_fd(&fm, &mut ts, &seqs[..SEQ], &seqs[1..], SEQ, 3e-2, &|_| true);
    println!("block gradcheck exact-mode (all 30 FD points): worst rel err {worst:.2e}");
    std::fs::remove_dir_all(&dir).ok();
}

/// Full-graph FD with the skeleton ACTIVE. The training gradient
/// deliberately freezes M (no-grad pinv — the certified convention), so
/// FD only matches tightly on tensors with NO attention above them:
/// the LAST layer's post-attention half (pln/gate/up/down). Everything
/// upstream moves M under perturbation — printed, not asserted (the
/// op-level frozen-M gradcheck covers those chains exactly).
#[test]
fn block_gradcheck_skeleton_active_last_layer_tight() {
    let dir = std::env::temp_dir().join(format!("fcd_block_ny_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = write_tiny_model(&dir);
    let model = CmfModel::open(&path).unwrap();
    let fm = FcdModel::from_cmf(&model, &tiny_o1()).unwrap(); // w=4 → skeleton on
    let mut ts = TrainState::new(&fm);
    let seqs = rand_ids(SEQ + 1, 7);
    // Trainable slots: layer-major, 5 per layer; the last layer's
    // pln/gate/up/down are slots (NL-1)*5 + 1..5.
    let base = (NL - 1) * 5;
    let worst = block_fd(&fm, &mut ts, &seqs[..SEQ], &seqs[1..], SEQ, 3e-2, &move |pi| pi > base);
    println!("block gradcheck skeleton-active (last layer post-attn tight): worst rel err {worst:.2e}");
    std::fs::remove_dir_all(&dir).ok();
}

/// E2E smoke: 20 steps on a tiny model — loss decreases, the best
/// checkpoint restore happens, the polished file loads and forwards.
#[test]
fn training_smoke_loss_decreases_and_output_loads() {
    let dir = std::env::temp_dir().join(format!("fcd_smoke_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = write_tiny_model(&dir);
    let model = Arc::new(CmfModel::open(&path).unwrap());
    let tr = rand_ids(4000, 11);
    let va = rand_ids(800, 13);
    let hp = FcdHyper {
        steps: 20,
        lr: 1e-3, // tiny random model: visible motion beats recipe-lr here
        kl_w: 0.7,
        eval_every: 5,
        bs: 2,
        seq: SEQ,
        seed: 0,
    };
    let out = dir.join("tiny_fcd.fcd.cmf");
    let report = run_polish(&model, &tiny_o1(), &hp, &tr, &va, &out, None).expect("polish");

    assert_eq!(report.losses.len(), 20, "one (ce, kl) per step");
    let total = |(ce, kl): &(f64, f64)| 0.3 * ce + 0.7 * kl;
    let first: f64 = report.losses[..5].iter().map(total).sum::<f64>() / 5.0;
    let last: f64 = report.losses[15..].iter().map(total).sum::<f64>() / 5.0;
    println!(
        "smoke: loss first5 {first:.4} → last5 {last:.4} | ppl start {:.2} final {:.2} \
         | best step {}",
        report.ppl_start, report.ppl_final, report.best_step
    );
    assert!(last < first, "loss must decrease over 20 steps ({first:.4} → {last:.4})");
    assert!(report.best_step > 0 && report.best_step % 5 == 0, "best from an eval point");
    assert!(report.ppl_final.is_finite() && report.ppl_start.is_finite());

    // The polished container: intact, trainables replaced as F32, o1 +
    // fcd provenance present, loadable and forwardable by the runtime.
    let polished = CmfModel::open(&out).expect("open polished");
    assert!(polished.verify().is_empty(), "hashes intact");
    let prov = polished.header.provenance.as_ref().expect("provenance");
    assert!(prov.get("o1_attn").is_some() && prov.get("fcd").is_some());
    let t0 = polished.tensor("model.layers.0.mlp.gate_proj.weight").unwrap();
    assert_eq!(t0.dtype, TensorDtype::F32);
    // Weights actually moved.
    let orig = model.tensor_bytes("model.layers.0.mlp.gate_proj.weight").unwrap();
    let new = polished.tensor_bytes("model.layers.0.mlp.gate_proj.weight").unwrap();
    assert_ne!(orig, new, "polish must change the trained tensors");
    // Untouched tensors byte-identical.
    assert_eq!(model.tensor_bytes("model.embed_tokens.weight").unwrap(), polished.tensor_bytes("model.embed_tokens.weight").unwrap());

    let polished = Arc::new(polished);
    let mut p = Pipeline::from_model(&polished, SamplerConfig::default()).expect("pipeline");
    let logits = p.forward_ids(&va[..32], None).expect("forward");
    assert!(logits.iter().all(|v| v.is_finite()), "finite logits from the polished file");
    std::fs::remove_dir_all(&dir).ok();
}

/// Hybrid graph FD: layer 0 = Full attention (converted, TRAINABLE),
/// layer 1 = GatedDeltaNet (frozen). FD over layer-0 weights flows
/// through the ENTIRE GDN BPTT above it (conv+SiLU, l2-norms, gates,
/// delta-rule state chain, gated RMSNorm) — the dispatch integration
/// check the GDN op-level gradcheck cannot give. Exact attention mode
/// (w ≥ t) so the only new machinery under test is the GDN arm.
#[test]
fn block_gradcheck_hybrid_gdn_above_trainable() {
    let dir = std::env::temp_dir().join(format!("fcd_block_gdn_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = write_tiny_model_variant(&dir, "tiny_hybrid", false, true);
    let model = CmfModel::open(&path).unwrap();
    // "0" alone means OFF in the spec grammar — build the cfg directly.
    let o1 = O1Cfg {
        layers: cortiq_engine::nystrom::O1Layers::List(vec![0]),
        m: 4,
        w: 64,
        sink: 0,
        rect: cortiq_engine::nystrom::O1_DEFAULT_RECT,
    };
    let fm = FcdModel::from_cmf(&model, &o1).unwrap();
    let mut ts = TrainState::new(&fm);
    assert_eq!(ts.layers, vec![0], "only the Full layer is convertible");
    let seqs = rand_ids(SEQ + 1, 17);
    let worst = block_fd(&fm, &mut ts, &seqs[..SEQ], &seqs[1..], SEQ, 3e-2, &|_| true);
    println!("block gradcheck hybrid (through GDN BPTT): worst rel err {worst:.2e}");
    std::fs::remove_dir_all(&dir).ok();
}

/// Output-gate variant: both layers Full with the Qwen3.5 per-head
/// [q; gate] projection; exact mode; every trainable FD-checked tight
/// (validates the gate split forward + σ-gate backward + re-interleave).
#[test]
fn block_gradcheck_output_gate() {
    let dir = std::env::temp_dir().join(format!("fcd_block_gate_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = write_tiny_model_variant(&dir, "tiny_gate", true, false);
    let model = CmfModel::open(&path).unwrap();
    let o1 = O1Cfg::from_spec("all", Some(4), Some(64), Some(0), None).unwrap();
    let fm = FcdModel::from_cmf(&model, &o1).unwrap();
    let mut ts = TrainState::new(&fm);
    let seqs = rand_ids(SEQ + 1, 19);
    let worst = block_fd(&fm, &mut ts, &seqs[..SEQ], &seqs[1..], SEQ, 3e-2, &|_| true);
    println!("block gradcheck output-gate: worst rel err {worst:.2e}");
    std::fs::remove_dir_all(&dir).ok();
}
