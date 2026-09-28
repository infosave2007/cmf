//! Natively bounded anchor `swa_sink_v1` — runtime gates (plan S3,
//! contract `docs/EMBRYO_BOUNDED_ANCHOR.md` §3-4).
//!
//! A tiny synthetic CMF (layer 0 = GatedDeltaNet mixer, layer 1 =
//! BoundedAttention) is written with `CmfModel::write` in two geometries:
//! `SMALL` (W=8, S=2) for the long-context gates and `REAL` (the Embryo
//! anchor: hd=128, kvh=2, W=128, S=4) for the operator oracle and the
//! ms/token gate.
//!
//! Gates: f64 reference of the operator (≤ 1e-5), per-token == chunked
//! prefill, state bytes identical after 64 / 4096 / 32768 generated
//! tokens (Δ = 0 B), 20-turn prefix reuse prefilling only the new tokens
//! with a bounded reuse record, wire round trip (f32 + f16 ring) →
//! next-64 logits Δ = 0, `--o1`/`CMF_O1*` refused on a bounded file while
//! a legacy file keeps its o1 path, ring rollback bit-for-bit.
//!
//! Every test that loads a file takes `ENV_LOCK`: the loader reads
//! `CMF_O1*`, which one test sets on purpose.

use cortiq_core::format::TensorSpec;
use cortiq_core::{
    AnchorCoreConfig, CmfHeader, CmfModel, LayerType, ModelArch, NormStyle, QuantType, TensorDtype,
};
use cortiq_engine::bounded::{BoundedRope, BoundedState};
use cortiq_engine::nystrom::{O1Cfg, O1Layers, O1_DEFAULT_RECT};
use cortiq_engine::pipeline::KV_PREFIX_TAIL;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::sync::{Arc, Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_guard() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Clone, Copy)]
struct Geom {
    hidden: usize,
    inter: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    vocab: usize,
    window: usize,
    sink: usize,
    /// Projection weight scale (keeps scores O(1) for either head_dim).
    wscale: f32,
}

/// Fast geometry for the long-context gates.
const SMALL: Geom = Geom {
    hidden: 32,
    inter: 48,
    nh: 4,
    nkv: 2,
    hd: 16,
    vocab: 256,
    window: 8,
    sink: 2,
    wscale: 0.5,
};

/// The Embryo anchor geometry (hd 128, kvh 2, W 128, S 4).
const REAL: Geom = Geom {
    hidden: 64,
    inter: 96,
    nh: 8,
    nkv: 2,
    hd: 128,
    vocab: 256,
    window: 128,
    sink: 4,
    wscale: 0.2,
};

/// GDN mixer geometry of layer 0 (rep = nv/nk = 2).
const G_NV: usize = 2;
const G_NK: usize = 1;
const G_DK: usize = 4;
const G_DV: usize = 4;
const G_KK: usize = 3;

fn synth_f32(n: usize, salt: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(salt.wrapping_mul(1442695040888963407) ^ 0x9E3779B97F4A7C15);
            let x = (x ^ (x >> 31)).wrapping_mul(0xBF58476D1CE4E5B9);
            (((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32) * scale
        })
        .collect()
}

fn ids(n: usize, salt: u64, vocab: usize) -> Vec<u32> {
    synth_f32(n, salt, 1.0)
        .iter()
        .map(|x| (((x + 0.5) * vocab as f32) as usize % vocab) as u32)
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn arch(g: Geom, bounded: bool) -> ModelArch {
    ModelArch {
        arch_name: "cortiq_embryo".into(),
        hidden_size: g.hidden,
        intermediate_size: g.inter,
        num_layers: 2,
        num_attention_heads: g.nh,
        num_kv_heads: g.nkv,
        head_dim: g.hd,
        vocab_size: g.vocab,
        layer_types: vec![
            LayerType::LinearAttention,
            if bounded {
                LayerType::BoundedAttention
            } else {
                LayerType::FullAttention
            },
        ],
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
        qwen4_exp: None,
        deepseek_v41: None,
        anchor_core: bounded.then(|| AnchorCoreConfig {
            kind: "swa_sink_v1".into(),
            window: g.window,
            sink: g.sink,
            rope: "relative_in_window".into(),
            sink_scores: "nope".into(),
            train_windows: vec![g.window / 2, g.window],
            far: None,
        }),
        linear_core: Some(cortiq_core::types::LinearCoreConfig {
            kind: "gated_delta_net".into(),
            num_heads: G_NV,
            nphase: None,
            value_head_dim: G_DV,
            phase_delta_layers: None,
        }),
        head_clusters: None,
        // Far beyond anything a test walks: the bounded operator has no
        // position, so the cap must never be the thing that stops it.
        max_position_embeddings: 1 << 20,
        linear_conv_kernel_dim: Some(G_KK),
        linear_num_key_heads: Some(G_NK),
        linear_num_value_heads: Some(G_NV),
        linear_key_head_dim: Some(G_DK),
        linear_value_head_dim: Some(G_DV),
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
        attn_logit_softcapping: None,
        mla: None,
        activation_situ_beta: None,
        activation_situ_linear_beta: None,
        attn_v_norm: false,
        num_loops: 1,
        kda_gate_lower_bound: None,
        g3n: None,
        rope_freq_factors: None,
        logit_multiplier: None,
        loop_final_norm: false,
        kv_heads_per_layer: None,
        v_head_dim: None,
        qk_norm_after_rope: false,
        prism_hadamard: None,
    }
}

/// Write the two-layer file: layer 0 GatedDeltaNet, layer 1 bounded
/// anchor (or a legacy full-attention layer when `bounded == false`).
fn write_model(
    dir: &std::path::Path,
    name: &str,
    g: Geom,
    bounded: bool,
    provenance: Option<serde_json::Value>,
) -> std::path::PathBuf {
    let (h, inter) = (g.hidden, g.inter);
    let mut specs: Vec<TensorSpec> = Vec::new();
    let mut push = |name: &str, shape: Vec<usize>, data: Vec<f32>| {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}");
        specs.push(TensorSpec {
            name: name.into(),
            dtype: TensorDtype::F32,
            shape,
            data: f32_bytes(&data),
        });
    };
    let norm1 = |n: usize, salt: u64| -> Vec<f32> {
        synth_f32(n, salt, 0.2).iter().map(|v| 1.0 + v).collect()
    };
    push(
        "model.embed_tokens.weight",
        vec![g.vocab, h],
        synth_f32(g.vocab * h, 100, 0.6),
    );
    push("model.norm.weight", vec![h], norm1(h, 101));
    for li in 0..2 {
        let p = format!("model.layers.{li}.");
        let s = li as u64 * 40;
        push(
            &format!("{p}input_layernorm.weight"),
            vec![h],
            norm1(h, 102 + s),
        );
        push(
            &format!("{p}post_attention_layernorm.weight"),
            vec![h],
            norm1(h, 103 + s),
        );
        if li == 0 {
            let c_dim = 2 * G_NK * G_DK + G_NV * G_DV;
            let vd = G_NV * G_DV;
            let la = format!("{p}linear_attn.");
            push(
                &format!("{la}in_proj_qkv.weight"),
                vec![c_dim, h],
                synth_f32(c_dim * h, 130 + s, 0.5),
            );
            push(
                &format!("{la}in_proj_z.weight"),
                vec![vd, h],
                synth_f32(vd * h, 131 + s, 0.5),
            );
            push(
                &format!("{la}in_proj_a.weight"),
                vec![G_NV, h],
                synth_f32(G_NV * h, 132 + s, 0.5),
            );
            push(
                &format!("{la}in_proj_b.weight"),
                vec![G_NV, h],
                synth_f32(G_NV * h, 133 + s, 0.5),
            );
            push(
                &format!("{la}conv1d.weight"),
                vec![c_dim, G_KK],
                synth_f32(c_dim * G_KK, 134 + s, 0.6),
            );
            push(
                &format!("{la}A_log"),
                vec![G_NV],
                synth_f32(G_NV, 135 + s, 0.8),
            );
            push(
                &format!("{la}dt_bias"),
                vec![G_NV],
                synth_f32(G_NV, 136 + s, 0.8),
            );
            push(
                &format!("{la}norm.weight"),
                vec![G_DV],
                norm1(G_DV, 137 + s),
            );
            push(
                &format!("{la}out_proj.weight"),
                vec![h, vd],
                synth_f32(h * vd, 138 + s, 0.5),
            );
        } else {
            let (nh, nkv, hd) = (g.nh, g.nkv, g.hd);
            push(
                &format!("{p}self_attn.q_proj.weight"),
                vec![nh * hd, h],
                synth_f32(nh * hd * h, 104 + s, g.wscale),
            );
            push(
                &format!("{p}self_attn.k_proj.weight"),
                vec![nkv * hd, h],
                synth_f32(nkv * hd * h, 105 + s, g.wscale),
            );
            push(
                &format!("{p}self_attn.v_proj.weight"),
                vec![nkv * hd, h],
                synth_f32(nkv * hd * h, 106 + s, g.wscale),
            );
            push(
                &format!("{p}self_attn.o_proj.weight"),
                vec![h, nh * hd],
                synth_f32(h * nh * hd, 107 + s, g.wscale),
            );
            if bounded {
                push(
                    &format!("{p}self_attn.sink_k.weight"),
                    vec![nkv, g.sink, hd],
                    synth_f32(nkv * g.sink * hd, 108 + s, 0.4),
                );
                push(
                    &format!("{p}self_attn.sink_v.weight"),
                    vec![nkv, g.sink, hd],
                    synth_f32(nkv * g.sink * hd, 109 + s, 0.4),
                );
            }
        }
        push(
            &format!("{p}mlp.gate_proj.weight"),
            vec![inter, h],
            synth_f32(inter * h, 110 + s, 0.5),
        );
        push(
            &format!("{p}mlp.up_proj.weight"),
            vec![inter, h],
            synth_f32(inter * h, 111 + s, 0.5),
        );
        push(
            &format!("{p}mlp.down_proj.weight"),
            vec![h, inter],
            synth_f32(h * inter, 112 + s, 0.5),
        );
    }
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::format::CMF_VERSION,
        arch: arch(g, bounded),
        quant_type: QuantType::F32,
        provenance,
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
    let path = dir.join(format!("{name}.cmf"));
    CmfModel::write(&path, &header, &specs, None, None).expect("write tiny model");
    path
}

fn greedy() -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        min_p: 0.0,
        seed: Some(0),
        ..Default::default()
    }
}

fn load(path: &std::path::Path) -> Pipeline {
    let m = Arc::new(CmfModel::open(path).expect("open"));
    Pipeline::from_model(&m, greedy()).expect("bounded file loads")
}

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("cortiq-bounded-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Per-position logits by the per-token path (`forward_span`, one
/// position at a time — the decode path) from a cleared cache.
fn logits_per_token(p: &mut Pipeline, ids: &[u32]) -> Vec<Vec<f32>> {
    p.kv_cache.clear();
    p.clear_history();
    let mut out = Vec::with_capacity(ids.len());
    for (pos, &id) in ids.iter().enumerate() {
        let e = p.embed_id(id);
        let h = p.forward_span(&e, pos, 0, 1, None).unwrap();
        out.push(p.logits_from_hidden(&h));
    }
    out
}

/// Per-position logits by the chunked prefill path (`prefill_span_ids`
/// in chunks of `chunk`) from a cleared cache.
fn logits_chunked(p: &mut Pipeline, ids: &[u32], chunk: usize) -> Vec<Vec<f32>> {
    p.kv_cache.clear();
    p.clear_history();
    let hs = p.hidden_size;
    let mut out = Vec::with_capacity(ids.len());
    let mut pos = 0;
    while pos < ids.len() {
        let end = (pos + chunk).min(ids.len());
        let hb = p.prefill_span_ids(&ids[pos..end], pos, 1, None).unwrap();
        for k in 0..end - pos {
            out.push(p.logits_from_hidden(&hb[k * hs..(k + 1) * hs]));
        }
        pos = end;
    }
    out
}

fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut m = 0f64;
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.len(), y.len());
        for (u, v) in x.iter().zip(y) {
            m = m.max(((*u as f64) - (*v as f64)).abs());
        }
    }
    m
}

fn state_report(p: &Pipeline) -> Vec<usize> {
    let mut v: Vec<usize> = p.kv_cache.layers.iter().map(|l| l.memory_bytes()).collect();
    v.push(p.kv_cache.total_memory_bytes());
    v.push(p.kv_cache.attention_state_bytes());
    v.push(p.kv_cache.recurrent_state_bytes());
    v.push(p.kv_cache.bounded_state_bytes());
    v
}

// ── (a) f64 reference of the operator ──────────────────────────────────

/// The served operator against a straightforward f64 implementation of
/// the contract (§1): sinks on raw q̂, window keys j ∈ (t−W, t] with q̂
/// rotated by t−j (absolute-angle f64 rotation), one softmax.
fn f64_reference(
    q: &[f32],
    keys: &[Vec<f32>],
    vals: &[Vec<f32>],
    t: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    w: usize,
    sink_k: &[f32],
    sink_v: &[f32],
    sink: usize,
    inv_freq: &[f32],
) -> Vec<f64> {
    let hpk = nh / nkv;
    let scale = 1.0 / (hd as f64).sqrt();
    let half = inv_freq.len();
    let mut out = vec![0f64; nh * hd];
    let lo = (t + 1).saturating_sub(w);
    for h in 0..nh {
        let g = h / hpk;
        let qh: Vec<f64> = q[h * hd..(h + 1) * hd].iter().map(|&x| x as f64).collect();
        let mut scores = Vec::with_capacity(sink + w);
        for s in 0..sink {
            let kr = &sink_k[(g * sink + s) * hd..(g * sink + s + 1) * hd];
            scores.push(qh.iter().zip(kr).map(|(a, &b)| a * b as f64).sum::<f64>() * scale);
        }
        for j in lo..=t {
            let delta = (t - j) as f64;
            let mut qr = qh.clone();
            for i in 0..half {
                let ang = delta * inv_freq[i] as f64;
                let (sn, cs) = ang.sin_cos();
                let (x0, x1) = (qh[i], qh[i + half]);
                qr[i] = x0 * cs - x1 * sn;
                qr[i + half] = x0 * sn + x1 * cs;
            }
            let kr = &keys[j][g * hd..(g + 1) * hd];
            scores.push(qr.iter().zip(kr).map(|(a, &b)| a * b as f64).sum::<f64>() * scale);
        }
        let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
        let z: f64 = e.iter().sum();
        let oh = &mut out[h * hd..(h + 1) * hd];
        for s in 0..sink {
            let vr = &sink_v[(g * sink + s) * hd..(g * sink + s + 1) * hd];
            for d in 0..hd {
                oh[d] += e[s] / z * vr[d] as f64;
            }
        }
        for (n, j) in (lo..=t).enumerate() {
            let vr = &vals[j][g * hd..(g + 1) * hd];
            for d in 0..hd {
                oh[d] += e[sink + n] / z * vr[d] as f64;
            }
        }
    }
    out
}

#[test]
fn operator_matches_f64_reference() {
    for (tag, g) in [("small", SMALL), ("real", REAL)] {
        let (nh, nkv, hd, w, sink) = (g.nh, g.nkv, g.hd, g.window, g.sink);
        let inv = cortiq_engine::attention::rope_inv_freq(hd, 10_000.0);
        let rope = BoundedRope::new(w, &inv, 1.0);
        let mut st = BoundedState::new(nkv, hd, w);
        let bytes0 = st.state_bytes();
        // Scores of order one: entries ~U(-1.5, 1.5) → |q·k|/√hd ≈ 0.75.
        let sink_k = synth_f32(nkv * sink * hd, 7, 3.0);
        let sink_v = synth_f32(nkv * sink * hd, 8, 3.0);
        let t_total = 3 * w + 5;
        let (mut keys, mut vals) = (Vec::new(), Vec::new());
        let mut worst = 0f64;
        for t in 0..t_total {
            let q = synth_f32(nh * hd, 1000 + t as u64, 3.0);
            let k = synth_f32(nkv * hd, 2000 + t as u64, 3.0);
            let v = synth_f32(nkv * hd, 3000 + t as u64, 3.0);
            st.insert(&k, &v);
            let mut out = vec![0f32; nh * hd];
            st.attend(&q, nh, &sink_k, &sink_v, sink, &rope, 1.0 / (hd as f32).sqrt(), &mut out);
            keys.push(k);
            vals.push(v);
            let reference = f64_reference(
                &q, &keys, &vals, t, nh, nkv, hd, w, &sink_k, &sink_v, sink, &inv,
            );
            for d in 0..nh * hd {
                worst = worst.max((out[d] as f64 - reference[d]).abs());
            }
            assert_eq!(st.len(), (t + 1).min(w));
            assert_eq!(st.state_bytes(), bytes0, "ring never grows");
        }
        eprintln!("{tag}: max |Δ out| vs f64 over {t_total} tokens = {worst:.3e}");
        assert!(worst <= 1e-5, "{tag}: operator drifts from the f64 reference: {worst:.3e}");
    }
}

// ── (a') per-token == chunked prefill, ppl through the same path ───────

#[test]
fn per_token_equals_chunked_prefill_and_ppl() {
    let _g = env_guard();
    let dir = tmpdir("paths");
    let path = write_model(&dir, "small", SMALL, true, None);
    let mut p = load(&path);
    let seq = ids(300, 11, SMALL.vocab);
    let per_token = logits_per_token(&mut p, &seq);
    let chunked = logits_chunked(&mut p, &seq, 128);
    let d = max_abs_diff(&per_token, &chunked);
    eprintln!("per-token vs chunked(128): max |Δ logit| = {d:.3e}");
    assert!(d <= 1e-6, "per-token and chunked prefill must agree: {d:.3e}");
    // A different chunking is the same operator too (ring + chunk only).
    let chunked2 = logits_chunked(&mut p, &seq, 37);
    let d2 = max_abs_diff(&chunked, &chunked2);
    assert!(d2 <= 1e-6, "chunk 128 vs 37: {d2:.3e}");
    // `ppl` scores through the normal path: exp(mean NLL) of the chunked
    // logits equals what the pipeline reports.
    let mut nll = 0f64;
    for (pos, lg) in chunked.iter().enumerate().take(seq.len() - 1) {
        let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = lg.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln() + mx;
        nll += lse - lg[seq[pos + 1] as usize] as f64;
    }
    let want = (nll / (seq.len() - 1) as f64).exp();
    let got = p.ppl_ids(&seq).expect("ppl scoring");
    eprintln!("ppl: pipeline {got:.6} vs recomputed {want:.6}");
    assert!(got.is_finite() && ((got - want) / want).abs() < 1e-6, "{got} vs {want}");
    // No anchor layer stores anything per position.
    assert_eq!(p.kv_cache.layers[1].head_keys(0).len(), 0);
    assert!(p.kv_cache.layers[1].bounded.is_some());
    assert!(!p.o1_active());
}

// ── (b) strictly O(1): state bytes identical after 64 / 4096 / 32768 ──

#[test]
fn state_bytes_identical_after_64_4096_32768_tokens() {
    let _g = env_guard();
    let dir = tmpdir("o1");
    let path = write_model(&dir, "small", SMALL, true, None);
    let mut p = load(&path);
    let prompt = ids(23, 5, SMALL.vocab);
    let mut reports = Vec::new();
    for &n in &[64usize, 4096, 32768] {
        let t0 = std::time::Instant::now();
        let r = p.generate_from_ids(&prompt, n, None, None).unwrap();
        assert_eq!(r.tokens_generated, n, "no EOS on the byte-level fallback");
        let rep = state_report(&p);
        eprintln!(
            "after {n} tokens: per-layer {:?} total {} attn {} rec {} ring {} — {:.1} ms",
            &rep[..2],
            rep[2],
            rep[3],
            rep[4],
            rep[5],
            t0.elapsed().as_secs_f64() * 1e3
        );
        assert_eq!(p.kv_cache.layers[1].head_keys(0).len(), 0);
        assert_eq!(p.kv_cache.layers[1].seq_len, 23 + n - 1, "honest context depth");
        assert!(!p.kv_cache.needs_eviction(), "an O(1) model has no eviction cliff");
        assert!(p.kv_history.is_empty(), "bounded file keeps no literal history");
        assert!(p.kv_prefix.tail_len() <= KV_PREFIX_TAIL);
        assert_eq!(p.kv_prefix.len(), 23 + n - 1);
        reports.push(rep);
    }
    assert_eq!(reports[0], reports[1], "Δ state bytes 64 → 4096 must be 0");
    assert_eq!(reports[1], reports[2], "Δ state bytes 4096 → 32768 must be 0");
    let ring = 2 * SMALL.nkv * SMALL.window * SMALL.hd * 4 + 8;
    assert_eq!(p.kv_cache.bounded_state_bytes(), ring);
    // The header alone predicts the same fixed record.
    let m = CmfModel::open(&path).unwrap();
    let st = cortiq_engine::loader::per_sequence_state_bytes(&m).unwrap();
    assert_eq!(st.growing_layers, 0);
    assert_eq!(st.bounded_bytes + 8, ring);
    assert_eq!(st.recurrent_bytes, p.kv_cache.recurrent_state_bytes());
}

/// ms/token flat in context on the REAL anchor geometry: the first 64
/// decode steps against the last 64 of a 32768-token generation.
/// Debug builds crawl here — run it as
/// `CMF_GPU=0 cargo test --release -p cortiq-engine --test bounded_runtime ms_per_token -- --ignored --nocapture`.
#[test]
#[ignore]
fn ms_per_token_flat_from_64_to_32768() {
    let _g = env_guard();
    let dir = tmpdir("speed");
    let path = write_model(&dir, "real", REAL, true, None);
    let mut p = load(&path);
    let prompt = ids(23, 5, REAL.vocab);
    let n = 32768usize;
    let stamps: Arc<Mutex<Vec<std::time::Instant>>> = Arc::new(Mutex::new(Vec::with_capacity(n)));
    let s2 = stamps.clone();
    let cb: cortiq_engine::pipeline::TokenCallback = Box::new(move |_t| {
        s2.lock().unwrap().push(std::time::Instant::now());
        true
    });
    let r = p.generate_from_ids(&prompt, n, None, Some(cb)).unwrap();
    assert_eq!(r.tokens_generated, n);
    let st = stamps.lock().unwrap();
    let ms = |a: usize, b: usize| (st[b] - st[a]).as_secs_f64() * 1e3 / (b - a) as f64;
    // The first W decode steps attend over a ring that is still FILLING
    // (fewer than W keys) — cheaper by construction, not a context
    // effect. The O(1) claim compares windows after the ring is full.
    let w = REAL.window;
    let fill = ms(0, 64);
    let early = ms(w + 64, w + 128);
    let mid = ms(4032, 4096);
    let late = ms(n - 65, n - 1);
    let all = ms(0, n - 1);
    eprintln!(
        "ms/token: fill(first 64) = {fill:.4}, early(ring full, {}..{}) = {early:.4}, at 4096 = {mid:.4}, \
         last 64 = {late:.4}, overall = {all:.4}; late/early = {:.3}, mid/early = {:.3}",
        w + 64,
        w + 128,
        late / early,
        mid / early
    );
    let rep = state_report(&p);
    eprintln!("state after {n}: {rep:?}");
    assert!(late / early <= 1.10, "ms/token grew with context: {late:.4} vs {early:.4}");
    assert!(mid / early <= 1.10, "ms/token grew with context: {mid:.4} vs {early:.4}");
}

// ── (c) multi-turn: prefill only the new tokens, bounded bookkeeping ───

#[test]
fn multi_turn_prefills_only_new_tokens() {
    let _g = env_guard();
    let dir = tmpdir("turns");
    let path = write_model(&dir, "small", SMALL, true, None);
    let mut p = load(&path);
    let mut history = ids(23, 5, SMALL.vocab);
    let mut reports = Vec::new();
    let mut turn2_check: Option<(Vec<u32>, Vec<u32>)> = None;
    for turn in 1..=20usize {
        history.extend(ids(512, 700 + turn as u64, SMALL.vocab));
        let r = p.generate_from_ids(&history, 16, None, None).unwrap();
        let want = if turn == 1 { history.len() } else { 512 + 1 };
        assert_eq!(
            p.last_prefill_tokens, want,
            "turn {turn}: only the new tokens (+ the unforwarded tip) are prefilled"
        );
        assert!(p.kv_history.is_empty());
        assert!(p.kv_prefix.tail_len() <= KV_PREFIX_TAIL);
        assert_eq!(p.kv_prefix.len(), history.len() + 16 - 1);
        history.extend(r.token_ids.iter().copied());
        reports.push(state_report(&p));
        if turn == 2 {
            turn2_check = Some((history.clone(), r.token_ids.clone()));
        }
    }
    for t in 1..reports.len() {
        assert_eq!(reports[t], reports[1], "turn {} state bytes drifted", t + 1);
    }
    // Extension-only reuse is EXACT: turn 2 with a reused prefix equals a
    // fresh pipeline prefilling the whole history.
    let (h2, reused) = turn2_check.unwrap();
    let prompt2 = &h2[..h2.len() - reused.len()];
    let mut fresh = load(&path);
    let rf = fresh.generate_from_ids(prompt2, 16, None, None).unwrap();
    assert_eq!(rf.token_ids, reused, "reused-prefix generation must equal fresh prefill");
    assert_eq!(fresh.last_prefill_tokens, prompt2.len());
    // A prompt that is NOT an extension starts fresh.
    let mut other = ids(40, 999, SMALL.vocab);
    other.push(1);
    let _ = p.generate_from_ids(&other, 4, None, None).unwrap();
    assert_eq!(p.last_prefill_tokens, other.len());
}

// ── (d) wire v2: import(export(state)) → next 64 logits Δ = 0 ──────────

#[test]
fn wire_round_trip_reproduces_next_64_logits() {
    let _g = env_guard();
    let dir = tmpdir("wire");
    let path = write_model(&dir, "small", SMALL, true, None);
    let prompt = ids(23, 5, SMALL.vocab);
    let cont = ids(64, 77, SMALL.vocab);
    for f16 in [false, true] {
        let mut a = load(&path);
        let _ = a.generate_from_ids(&prompt, 100, None, None).unwrap();
        let mut b = load(&path);
        for li in 0..2 {
            let bytes = a.kv_cache.layers[li].export_wire(f16).unwrap();
            assert_eq!(&bytes[..4], b"CMFS");
            if f16 {
                // Both sides then hold the SAME f16-rounded ring.
                a.kv_cache.layers[li].import_wire(&bytes).unwrap();
            }
            b.kv_cache.layers[li].import_wire(&bytes).unwrap();
        }
        assert_eq!(b.kv_cache.seq_len(), a.kv_cache.seq_len());
        assert_eq!(state_report(&a), state_report(&b));
        let pos0 = a.kv_cache.seq_len();
        for (i, &id) in cont.iter().enumerate() {
            let ea = a.embed_id(id);
            let eb = b.embed_id(id);
            let ha = a.forward_span(&ea, pos0 + i, 0, 1, None).unwrap();
            let hb = b.forward_span(&eb, pos0 + i, 0, 1, None).unwrap();
            let la = a.logits_from_hidden(&ha);
            let lb = b.logits_from_hidden(&hb);
            assert_eq!(la, lb, "f16={f16}: logits differ at continuation token {i}");
        }
        eprintln!("wire f16={f16}: 64 continuation logits identical");
    }
}

// ── (e) --o1 / CMF_O1* refused on a bounded file; legacy keeps its o1 ──

#[test]
fn o1_refused_on_bounded_file_legacy_unchanged() {
    let _g = env_guard();
    let dir = tmpdir("o1refuse");
    let path = write_model(&dir, "small", SMALL, true, None);
    let cfg = O1Cfg {
        layers: O1Layers::All,
        m: 4,
        w: 8,
        sink: 2,
        rect: O1_DEFAULT_RECT,
    };
    let mut p = load(&path);
    let err = p.try_set_o1(Some(cfg.clone())).unwrap_err();
    assert!(err.contains("native bounded"), "{err}");
    p.set_o1(Some(cfg.clone()));
    assert!(!p.o1_active(), "set_o1 must not arm o1 on a bounded file");
    let m = Arc::new(CmfModel::open(&path).unwrap());
    for (k, v) in [("CMF_O1", "all"), ("CMF_O1_M", "8"), ("CMF_O1_WINDOW", "64")] {
        unsafe { std::env::set_var(k, v) };
        let r = Pipeline::from_model(&m, greedy());
        unsafe { std::env::remove_var(k) };
        let e = format!("{}", r.err().expect("bounded file must refuse {k}"));
        assert!(e.contains("native bounded") && e.contains(k), "{e}");
    }
    unsafe { std::env::set_var("CMF_O1", "off") };
    let r = Pipeline::from_model(&m, greedy());
    unsafe { std::env::remove_var("CMF_O1") };
    assert!(r.is_ok(), "CMF_O1=off is not an override");
    // A converter hint in provenance is refused too.
    let hinted = write_model(
        &dir,
        "hinted",
        SMALL,
        true,
        Some(serde_json::json!({"o1_attn": {"layers": "all", "m": 8}})),
    );
    let mh = Arc::new(CmfModel::open(&hinted).unwrap());
    let e = format!("{}", Pipeline::from_model(&mh, greedy()).err().unwrap());
    assert!(e.contains("o1_attn"), "{e}");
    // Legacy FullAttention file: loads, o1 arms, generation runs.
    let legacy = write_model(&dir, "legacy", SMALL, false, None);
    let mut l = load(&legacy);
    assert!(!l.bounded_native());
    assert!(l.kv_cache.layers[1].bounded.is_none());
    l.set_o1(Some(cfg));
    assert!(l.o1_active(), "legacy file keeps the post-hoc o1 path");
    let r = l.generate_from_ids(&ids(23, 5, SMALL.vocab), 12, None, None).unwrap();
    assert_eq!(r.tokens_generated, 12);
    assert!(l.kv_cache.layers[1].o1_sealed());
    // …and its reuse bookkeeping is the legacy vector, untouched.
    assert!(!l.kv_history.is_empty() || l.o1_active());
}

// ── (f) snapshot / restore: truncate_last rolls the ring back ─────────

#[test]
fn truncate_last_restores_ring_bit_for_bit() {
    let _g = env_guard();
    let dir = tmpdir("rollback");
    let path = write_model(&dir, "small", SMALL, true, None);
    let mut p = load(&path);
    let prompt = ids(23, 5, SMALL.vocab);
    let _ = p.generate_from_ids(&prompt, 40, None, None).unwrap();
    let snap = p.kv_cache.layers[1].bounded_snapshot().unwrap();
    let seq0 = p.kv_cache.layers[1].seq_len;
    let before = p.kv_cache.layers[1].bounded.clone().unwrap();
    let pos0 = p.kv_cache.seq_len();
    for (i, &id) in ids(5, 9, SMALL.vocab).iter().enumerate() {
        let e = p.embed_id(id);
        let _ = p.forward_span(&e, pos0 + i, 0, 1, None).unwrap();
    }
    assert!(!p.kv_cache.layers[1].bounded.as_ref().unwrap().same_state(&before));
    p.kv_cache.layers[1].truncate_last(5);
    assert!(
        p.kv_cache.layers[1].bounded.as_ref().unwrap().same_state(&before),
        "truncate_last must restore the ring bit for bit"
    );
    assert_eq!(p.kv_cache.layers[1].seq_len, seq0);
    // The explicit snapshot API restores the same state.
    for (i, &id) in ids(3, 10, SMALL.vocab).iter().enumerate() {
        let e = p.embed_id(id);
        let _ = p.forward_span(&e, pos0 + i, 0, 1, None).unwrap();
    }
    p.kv_cache.layers[1].bounded_restore(&snap);
    assert!(p.kv_cache.layers[1].bounded.as_ref().unwrap().same_state(&before));
    assert_eq!(p.kv_cache.layers[1].seq_len, seq0);
    // Eviction never touches a bounded layer.
    p.kv_cache.evict(4);
    assert!(p.kv_cache.layers[1].bounded.as_ref().unwrap().same_state(&before));
}
