//! MiMo-V2 tower companion (`<stem>.mm.cmf`) at load time — gate G1.3.
//!
//! Synthetic files (always run): a complete tiny companion loads on its
//! own; attaching it to a text model with a different hidden size, or with
//! a tokenizer that does not map the nine special tokens to the pinned ids,
//! is a hard error; so are a missing tensor, a quantized codebook, a wrong
//! base arch, and two candidate companions next to one text file.
//!
//! Real files (opt-in): `CMF_MIMO_MM=/path/x.mm.cmf` loads the converted
//! companion and prints its inventory; with `CMF_MIMO_TEXT=/path/text.cmf`
//! it is also attached to that text model.
//!
//!   CMF_MIMO_MM=/root/mimo/out/mimo-mm-conv/MiMo-V2.6-Flash.mm.cmf \
//!     cargo test --release -p cortiq-engine --test mimo_mm_companion -- --nocapture

use std::path::{Path, PathBuf};
use std::sync::Arc;

use cortiq_core::CMF_VERSION;
use cortiq_core::format::{CmfHeader, CmfModel, TensorSpec};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use cortiq_engine::mimo_mm::{
    AUDIO_TOKENIZER_CONFIG_BLOB, MIMO_SPECIAL_TOKENS, MM_CONFIG_BLOB, MimoMm, MimoMmSource,
    MimoTowerGroup, is_codebook, mimo_tower_inventory,
};
use serde_json::{Value, json};

fn tiny_configs() -> (Value, Value) {
    let cfg = json!({
        "model_type": "mimo_v2",
        "hidden_size": 64,
        "vocab_size": 151680,
        "vision_start_token_id": 151652, "vision_end_token_id": 151653,
        "image_token_id": 151655, "video_token_id": 151656,
        "audio_token_id": 151669, "audio_start_token_id": 151673, "audio_end_token_id": 151674,
        "processor_config": {"video_start_token_id": 151670, "video_end_token_id": 151671},
        "vision_config": {
            "depth": 3, "hidden_size": 64, "intermediate_size": 96, "num_heads": 4,
            "num_key_value_heads": 2, "qk_channels": 16, "out_hidden_size": 64,
            "in_chans": 3, "patch_size": 4, "temporal_patch_size": 2,
            "spatial_merge_size": 2, "fullatt_block_indexes": [0],
            "vit_window_attn_types": [-1, 0, 1], "visual_token_window_size": 4,
            "use_sink": true
        },
        "audio_config": {
            "audio_channels": 3, "group_size": 4, "input_local_dim": 32,
            "input_local_layers": 2, "input_local_attn_heads": 2, "input_local_head_dim": 16,
            "input_local_intermediate_size": 64, "out_hidden_size": 64,
            "projection_layers": 2, "rope_theta": 640000, "add_post_norm": true,
            "speech_vocab_size": "40", "speech_zeroemb_idx": "32"
        }
    });
    let at = json!({
        "d_model": 32, "encoder_layers": 2, "encoder_attention_heads": 2,
        "encoder_ffn_dim": 64, "n_mels": 8, "kernel_size": 3, "stride_size": 2,
        "avg_pooler": 2, "encoder_skip_layer_id": 1, "encoder_causal": true,
        "encoder_attn_window_size": [4, 0], "hybrid_attention": true, "swa_per_block": 2,
        "rope_theta": 10000, "num_quantizers": 3, "codebook_size": [32, 16, 8],
        "sampling_rate": 24000, "hop_length": 240, "nfft": 960, "window_size": 960,
        "ln_type": "LayerNorm", "scale_embedding": false
    });
    (cfg, at)
}

fn arch(name: &str, hidden: usize, vocab: usize) -> ModelArch {
    serde_json::from_value(json!({
        "arch_name": name, "hidden_size": hidden, "intermediate_size": 0, "num_layers": 0,
        "num_attention_heads": 0, "num_kv_heads": 0, "head_dim": 0, "vocab_size": vocab,
        "layer_types": [], "rms_norm_eps": 1e-6, "max_position_embeddings": 0
    }))
    .unwrap()
}

fn header(arch: ModelArch, provenance: Option<Value>) -> CmfHeader {
    CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        arch,
        quant_type: QuantType::F16,
        provenance,
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
    }
}

fn blob(name: &str, v: &Value) -> TensorSpec {
    let b = serde_json::to_vec(v).unwrap();
    TensorSpec {
        name: name.into(),
        dtype: TensorDtype::U8,
        shape: vec![b.len()],
        data: b,
    }
}

/// Every inventory tensor (F16 zeros, F32 codebooks) plus both blobs.
fn tower_tensors(cfg: &Value, at: &Value) -> Vec<TensorSpec> {
    let mut out = vec![
        blob(MM_CONFIG_BLOB, cfg),
        blob(AUDIO_TOKENIZER_CONFIG_BLOB, at),
    ];
    for (name, shape) in mimo_tower_inventory(cfg, at).unwrap() {
        let n: usize = shape.iter().product();
        let (dtype, data) = if is_codebook(&name) {
            (TensorDtype::F32, vec![0u8; n * 4])
        } else {
            (TensorDtype::F16, vec![0u8; n * 2])
        };
        out.push(TensorSpec {
            name,
            dtype,
            shape,
            data,
        });
    }
    out
}

fn companion_prov(base: &str) -> Value {
    json!({"tool": "test", "mimo_mm": {"base_arch": base, "codec": {"visual": {"matrices": "f16"}}}})
}

fn write_companion(path: &Path, edit: impl FnOnce(&mut Vec<TensorSpec>), base: &str) {
    let (cfg, at) = tiny_configs();
    let mut t = tower_tensors(&cfg, &at);
    edit(&mut t);
    CmfModel::write(
        path,
        &header(arch("mimo_v2_mm", 64, 151680), Some(companion_prov(base))),
        &t,
        None,
        None,
    )
    .unwrap();
}

/// A tokenizer.json with the nine special tokens (`remap` moves one id).
fn tokenizer_json(remap: Option<(&str, u32)>) -> Vec<u8> {
    let added: Vec<Value> = MIMO_SPECIAL_TOKENS
        .iter()
        .map(|(s, id)| {
            let id = match remap {
                Some((r, to)) if r == *s => to,
                _ => *id,
            };
            json!({"id": id, "content": s, "special": true})
        })
        .collect();
    serde_json::to_vec(&json!({
        "model": {"vocab": {"a": 0, "b": 1}, "merges": []},
        "added_tokens": added,
    }))
    .unwrap()
}

/// A text-side `mimo_v2` file (towers are validated only against its
/// header and tokenizer, so one embedding tensor is enough).
fn write_text(path: &Path, hidden: usize, vocab: Option<Vec<u8>>) {
    let emb = TensorSpec {
        name: "model.embed_tokens.weight".into(),
        dtype: TensorDtype::F16,
        shape: vec![4, hidden],
        data: vec![0u8; 4 * hidden * 2],
    };
    CmfModel::write(
        path,
        &header(arch("mimo_v2", hidden, 151680), None),
        &[emb],
        None,
        vocab.as_deref(),
    )
    .unwrap();
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("cortiq-mimo-mm-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn open(p: &Path) -> Arc<CmfModel> {
    Arc::new(CmfModel::open(p).unwrap())
}

#[test]
fn companion_loads_alone_and_attaches_to_its_text_model() {
    let d = tmp("ok");
    let comp = d.join("m.mm.cmf");
    write_companion(&comp, |_| {}, "mimo_v2");
    let mm = MimoMm::open(&comp).unwrap();
    assert_eq!(mm.source, MimoMmSource::Companion);
    assert_eq!(mm.hidden_size, 64);
    assert_eq!(mm.vision.qkv_rows(), (4 + 2 * 2) * 16);
    assert!(mm.vision.has_sink(1) && !mm.vision.has_sink(0));
    assert_eq!(mm.audio.speech_vocab, vec![40; 3]);
    assert_eq!(mm.audio_tokenizer.codebook_sizes, vec![32, 16, 8]);
    assert_eq!(mm.group_codec(MimoTowerGroup::Vision), Some("f16"));
    // Typed access.
    let q = mm
        .linear("visual.blocks.2.attn.qkv.weight", 128, 64)
        .unwrap();
    assert_eq!((q.rows(), q.cols()), (128, 64));
    assert!(
        mm.linear("visual.blocks.2.attn.qkv.weight", 64, 64)
            .is_err()
    );
    assert_eq!(
        mm.f32_shaped(
            "audio_tokenizer.encoder.quantizer.vq.layers.1._codebook.embed",
            &[16, 32]
        )
        .unwrap()
        .len(),
        16 * 32
    );

    // Sibling discovery: `<stem>.mm.cmf` is not needed — the only
    // *.mm.cmf in the directory is found; then it is validated.
    let text = d.join("m-q4tp.cmf");
    write_text(&text, 64, Some(tokenizer_json(None)));
    let t = open(&text);
    let found = MimoMm::discover(&text, None).unwrap();
    assert_eq!(found.as_deref(), Some(comp.as_path()));
    let attached = MimoMm::attach(&t, None).unwrap().expect("companion found");
    assert_eq!(attached.model().path, comp);
    // The exact stem wins over other candidates.
    let exact = d.join("m-q4tp.mm.cmf");
    std::fs::copy(&comp, &exact).unwrap();
    assert_eq!(MimoMm::discover(&text, None).unwrap(), Some(exact.clone()));
    std::fs::remove_file(&exact).unwrap();
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn wrong_hidden_size_is_a_hard_error() {
    let d = tmp("hidden");
    let comp = d.join("m.mm.cmf");
    write_companion(&comp, |_| {}, "mimo_v2");
    let text = d.join("m.cmf");
    write_text(&text, 32, Some(tokenizer_json(None)));
    let err = MimoMm::attach(&open(&text), None).unwrap_err();
    assert!(err.contains("hidden_size"), "{err}");
    let err = MimoMm::attach(&open(&text), Some(&comp)).unwrap_err();
    assert!(err.contains("hidden_size"), "{err}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn tokenizer_that_moves_a_special_token_is_a_hard_error() {
    let d = tmp("tok");
    let comp = d.join("m.mm.cmf");
    write_companion(&comp, |_| {}, "mimo_v2");
    let mm = MimoMm::open(&comp).unwrap();
    for (tok, to) in [("<|image_pad|>", 151654u32), ("<|mimo_audio_end|>", 151675)] {
        let text = d.join("m.cmf");
        write_text(&text, 64, Some(tokenizer_json(Some((tok, to)))));
        let err = mm.validate_text_model(&open(&text)).unwrap_err();
        assert!(err.contains(tok), "{err}");
    }
    // No embedded tokenizer: the ids cannot be checked → refused.
    let text = d.join("m.cmf");
    write_text(&text, 64, None);
    let err = mm.validate_text_model(&open(&text)).unwrap_err();
    assert!(err.contains("tokenizer"), "{err}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn incomplete_or_foreign_companions_are_refused() {
    let d = tmp("bad");
    let p = d.join("x.mm.cmf");
    write_companion(
        &p,
        |t| t.retain(|s| s.name != "visual.blocks.1.attn.sinks"),
        "mimo_v2",
    );
    let err = MimoMm::open(&p).unwrap_err();
    assert!(err.contains("missing"), "{err}");

    write_companion(
        &p,
        |t| {
            for s in t.iter_mut().filter(|s| is_codebook(&s.name)) {
                s.dtype = TensorDtype::F16;
                s.data.truncate(s.data.len() / 2);
            }
        },
        "mimo_v2",
    );
    let err = MimoMm::open(&p).unwrap_err();
    assert!(err.contains("F32"), "{err}");

    write_companion(
        &p,
        |t| {
            t.push(TensorSpec {
                name: "visual.blocks.9.norm1.weight".into(),
                dtype: TensorDtype::F16,
                shape: vec![64],
                data: vec![0; 128],
            })
        },
        "mimo_v2",
    );
    let err = MimoMm::open(&p).unwrap_err();
    assert!(err.contains("unexpected"), "{err}");

    write_companion(&p, |_| {}, "qwen3");
    let err = MimoMm::open(&p).unwrap_err();
    assert!(err.contains("base_arch"), "{err}");

    // A config whose special ids differ from the pinned layout.
    write_companion(
        &p,
        |t| {
            let (mut cfg, _) = tiny_configs();
            cfg["image_token_id"] = json!(151654);
            t[0] = blob(MM_CONFIG_BLOB, &cfg);
        },
        "mimo_v2",
    );
    let err = MimoMm::open(&p).unwrap_err();
    assert!(err.contains("<|image_pad|>"), "{err}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn discovery_is_explicit_or_unambiguous() {
    let d = tmp("disc");
    let text = d.join("m-00001-of-00002.cmf");
    write_text(&text, 64, Some(tokenizer_json(None)));
    assert_eq!(MimoMm::discover(&text, None).unwrap(), None);
    assert!(MimoMm::attach(&open(&text), None).unwrap().is_none());
    let a = d.join("a.mm.cmf");
    let b = d.join("b.mm.cmf");
    write_companion(&a, |_| {}, "mimo_v2");
    std::fs::copy(&a, &b).unwrap();
    let err = MimoMm::discover(&text, None).unwrap_err();
    assert!(err.contains("--mm"), "{err}");
    // The shard suffix is not part of the stem.
    let m = d.join("m.mm.cmf");
    std::fs::copy(&a, &m).unwrap();
    assert_eq!(MimoMm::discover(&text, None).unwrap(), Some(m));
    assert_eq!(MimoMm::discover(&text, Some(&b)).unwrap(), Some(b.clone()));
    assert!(MimoMm::discover(&text, Some(&d.join("nope.mm.cmf"))).is_err());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn single_file_multimodal_is_detected_by_its_config_blob() {
    let d = tmp("single");
    let (cfg, at) = tiny_configs();
    let p = d.join("full.cmf");
    let mut t = tower_tensors(&cfg, &at);
    t.push(TensorSpec {
        name: "model.embed_tokens.weight".into(),
        dtype: TensorDtype::F16,
        shape: vec![4, 64],
        data: vec![0; 512],
    });
    CmfModel::write(
        &p,
        &header(arch("mimo_v2", 64, 151680), None),
        &t,
        None,
        Some(&tokenizer_json(None)),
    )
    .unwrap();
    let m = open(&p);
    let mm = MimoMm::attach(&m, None).unwrap().expect("own towers");
    assert_eq!(mm.source, MimoMmSource::SingleFile);
    // A text-only mimo_v2 file is not a tower source by itself.
    let text = d.join("text.cmf");
    write_text(&text, 64, Some(tokenizer_json(None)));
    assert!(MimoMm::open(&text).unwrap_err().contains("text-only"));
    let _ = std::fs::remove_dir_all(&d);
}

/// Opt-in: the real converted companion (and optionally its text model).
#[test]
fn real_companion_from_env() {
    let Ok(path) = std::env::var("CMF_MIMO_MM") else {
        eprintln!("CMF_MIMO_MM unset — skipping the real-file check");
        return;
    };
    let t0 = std::time::Instant::now();
    let mm = MimoMm::open(&path).unwrap();
    let m = mm.model();
    let mut by_group: std::collections::BTreeMap<String, std::collections::BTreeMap<&str, usize>> =
        Default::default();
    for e in &m.tensors {
        let g = MimoTowerGroup::of(&e.name)
            .map(|g| g.label().to_string())
            .unwrap_or_else(|| "blob".into());
        *by_group
            .entry(g)
            .or_default()
            .entry(e.dtype.name())
            .or_default() += 1;
    }
    eprintln!(
        "{path}: {:?}, hidden {}, {} tensors, opened+validated in {:.2} s",
        mm.source,
        mm.hidden_size,
        m.tensors.len(),
        t0.elapsed().as_secs_f64()
    );
    for (g, h) in &by_group {
        eprintln!("  {g}: {h:?}");
    }
    eprintln!(
        "  codec: {}",
        mm.provenance
            .as_ref()
            .map(|p| p["codec"].to_string())
            .unwrap_or_default()
    );
    // The release inventory: 364 + 95 + 389 tower tensors + 2 blobs.
    assert_eq!(m.tensors.len(), 364 + 95 + 389 + 2);
    assert!(m.verify().is_empty(), "{:?}", m.verify());
    // Weight-level codec error against an exact (F16) companion.
    if let Ok(reference) = std::env::var("CMF_MIMO_MM_REF") {
        let r = MimoMm::open(&reference).unwrap();
        // group → (tensors, identical, min cos, Σ rel err, max rel err, worst name)
        let mut acc: std::collections::BTreeMap<&str, (usize, usize, f64, f64, f64, String)> =
            Default::default();
        for e in &m.tensors {
            let Some(g) = MimoTowerGroup::of(&e.name) else {
                continue;
            };
            let a = mm.f32(&e.name).unwrap();
            let b = r.f32(&e.name).unwrap();
            assert_eq!(a.len(), b.len(), "{}", e.name);
            let (mut dot, mut na, mut nb, mut nd) = (0f64, 0f64, 0f64, 0f64);
            for (x, y) in a.iter().zip(&b) {
                let (x, y) = (*x as f64, *y as f64);
                dot += x * y;
                na += x * x;
                nb += y * y;
                nd += (x - y) * (x - y);
            }
            let cos = if na > 0.0 && nb > 0.0 {
                dot / (na * nb).sqrt()
            } else {
                1.0
            };
            let rel = if nb > 0.0 {
                (nd / nb).sqrt()
            } else {
                nd.sqrt()
            };
            let slot = acc
                .entry(g.label())
                .or_insert((0, 0, 1.0, 0.0, 0.0, String::new()));
            slot.0 += 1;
            slot.1 += (nd == 0.0) as usize;
            slot.2 = slot.2.min(cos);
            slot.3 += rel;
            if rel > slot.4 {
                slot.4 = rel;
                slot.5 = e.name.clone();
            }
        }
        eprintln!("  vs {reference}:");
        for (g, (n, same, min_cos, sum_rel, max_rel, worst)) in &acc {
            eprintln!(
                "    {g}: {n} tensors, {same} identical, min cos {min_cos:.6}, \
                 mean rel err {:.4}, max rel err {max_rel:.4} ({worst})",
                sum_rel / *n as f64
            );
        }
    }
    // CMF_MIMO_TEXT: attach to that text model; with CMF_MIMO_TEXT_EXPECT_ERR
    // the attach must instead fail with a message containing that text.
    if let Ok(text) = std::env::var("CMF_MIMO_TEXT") {
        let t = Arc::new(CmfModel::open(&text).unwrap());
        let r = mm.validate_text_model(&t);
        match std::env::var("CMF_MIMO_TEXT_EXPECT_ERR") {
            Ok(want) => {
                let err = r.expect_err("attach must fail");
                assert!(err.contains(&want), "{err}");
                eprintln!("  refused {text} as expected: {err}");
            }
            Err(_) => {
                r.unwrap();
                eprintln!(
                    "  attached to {text} (arch {}, hidden {}): 9 special ids OK",
                    t.arch().arch_name,
                    t.arch().hidden_size
                );
            }
        }
    }
}
