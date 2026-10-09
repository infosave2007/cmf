//! EmbeddingGemma 2 audio parity against the float32 transformers /
//! sentence-transformers reference.
//!
//! Needs two env vars (the test passes trivially without them):
//! * `CORTIQ_EGEMMA2_MODEL` — a `.cmf` from `cortiq convert --model
//!   <google/embeddinggemma-2 dir>` with the audio tower;
//! * `CORTIQ_EGEMMA2_REF` — the reference dir: `manifest.json` (cases with
//!   `modality: "audio"`, `path`, `input_ids`, `row`) and `embeddings.npy`.
//!
//! Optional `CORTIQ_EGEMMA2_AUDIO_DEBUG`: a dir of per-stage `.npy` dumps
//! named `<case>__<key>.npy` (`input_features`, `audio_subsample_out`,
//! `audio_layer_NN_out`, `audio_tower_out`, `embed_audio_out`); each stage
//! present is compared and its max relative error printed.
//!
//! The cosine floor is 0.99999 for an exact (bf16 / f32) file and 0.995 for
//! a quantized one; `CORTIQ_EGEMMA2_MIN_COS` overrides it.
//!
//! When `<ref>/audio_extra/manifest.json` exists (`make_ref_audio_extra.py`)
//! its cases run too: text with one or two `<|audio|>` placeholders (with and
//! without a task prompt), a clip past the 30 s cut, a 0.3 s clip — all under
//! the same floor — and WAVs at 8 / 44.1 / 48 kHz, stereo and float. Those
//! the reference loads with librosa (soxr) while we resample with our own
//! Kaiser sinc, so they get a floor of their own, 0.9999 (measured on the
//! exact file: 1 − cos ≤ 8e-7).
//!
//! `cargo test --release -p cortiq-engine --test egemma2_audio_parity -- --nocapture`

use cortiq_core::CmfModel;
use cortiq_engine::egemma2::{EmbeddingGemma2, MATRYOSHKA_DIMS, TextInput, cosine, matryoshka};
use cortiq_engine::egemma2_audio::{self as ea, AudioInput, AudioTower};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A float32 C-order .npy as (shape, data).
fn read_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&b[..6], b"\x93NUMPY", "not a .npy");
    let (hl, off) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[off..off + hl]).unwrap();
    assert!(header.contains("'<f4'"), "expected float32: {header}");
    let shape_s = header
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let shape: Vec<usize> = shape_s
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data = b[off + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (shape, data)
}

fn max_rel(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "length");
    let scale = b
        .iter()
        .fold(0f64, |m, &v| m.max((v as f64).abs()))
        .max(1e-30);
    a.iter()
        .zip(b)
        .fold(0f64, |m, (&x, &y)| m.max((x as f64 - y as f64).abs()))
        / scale
}

#[test]
fn egemma2_audio_parity() {
    let (Some(model), Some(refdir)) = (
        std::env::var_os("CORTIQ_EGEMMA2_MODEL").map(PathBuf::from),
        std::env::var_os("CORTIQ_EGEMMA2_REF").map(PathBuf::from),
    ) else {
        eprintln!("egemma2_audio_parity: set CORTIQ_EGEMMA2_MODEL and CORTIQ_EGEMMA2_REF to run");
        return;
    };
    let dbg = std::env::var_os("CORTIQ_EGEMMA2_AUDIO_DEBUG").map(PathBuf::from);
    let cmf = Arc::new(CmfModel::open(&model).expect("open model"));
    let pool = cortiq_engine::pool::Pool::from_env();
    let enc = EmbeddingGemma2::load(&cmf, pool.clone()).expect("load text");
    let t0 = std::time::Instant::now();
    let tower = AudioTower::load(&cmf, pool).expect("load audio tower");
    eprintln!("audio tower loaded in {:.2}s", t0.elapsed().as_secs_f64());
    let floor: f64 = std::env::var("CORTIQ_EGEMMA2_MIN_COS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if matches!(enc.quant.as_str(), "bf16" | "f32") {
            0.99999
        } else {
            0.995
        });
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(refdir.join("manifest.json")).unwrap()).unwrap();
    let (shape, refs) = read_npy_f32(&refdir.join("embeddings.npy"));
    let d = shape[1];
    let cases: Vec<&serde_json::Value> = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["modality"] == "audio")
        .collect();
    assert!(!cases.is_empty(), "no audio cases in the manifest");

    let mut inputs = Vec::new();
    for c in &cases {
        let id = c["id"].as_str().unwrap();
        let wave = ea::read_audio(&refdir.join(c["path"].as_str().unwrap())).expect("read wav");
        assert_eq!(wave.len() as u64, c["num_samples"].as_u64().unwrap());
        // token ids are exact
        let want: Vec<u32> = c["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = ea::input_ids(
            &enc,
            (tower.audio_token, tower.boa_token, tower.eoa_token),
            &AudioInput {
                text: None,
                clips: vec![wave.clone()],
            },
            &[ea::num_tokens(wave.len())],
        )
        .unwrap();
        assert_eq!(got, want, "token ids of {id}");
        // per-stage checks against the debug dumps
        if let Some(dir) = &dbg {
            let load = |k: &str| {
                let p = dir.join(format!("{id}__{k}.npy"));
                p.exists().then(|| read_npy_f32(&p).1)
            };
            let (mel, frames) = tower.log_mel(&wave);
            if let Some(r) = load("input_features") {
                let r = &r[..frames * ea::N_MELS];
                let maxabs = mel
                    .iter()
                    .zip(r)
                    .fold(0f64, |m, (&a, &b)| m.max((a as f64 - b as f64).abs()));
                eprintln!("{id}: input_features ({frames} frames) max |Δ| {maxabs:.3e}");
            }
            // the tower on the reference's own features, then on ours
            let feed = load("input_features").map(|r| r[..frames * ea::N_MELS].to_vec());
            for (label, m) in [("ref mel", feed.as_ref()), ("our mel", Some(&mel))] {
                let Some(m) = m else { continue };
                let mut trace = Vec::new();
                let soft = tower.encode_mels(&[(&m[..], frames)], Some(&mut trace));
                let mut stages = vec!["audio_subsample_out".to_string()];
                for i in 0..trace.len() - 2 {
                    stages.push(format!("audio_layer_{i:02}_out"));
                }
                stages.push("audio_tower_out".into());
                for (k, got) in stages.iter().zip(&trace) {
                    if let Some(r) = load(k) {
                        eprintln!("{id} [{label}] {k}: max rel {:.3e}", max_rel(got, &r));
                    }
                }
                if let Some(r) = load("embed_audio_out") {
                    eprintln!(
                        "{id} [{label}] embed_audio_out: max rel {:.3e}",
                        max_rel(&soft[0], &r)
                    );
                }
            }
        }
        inputs.push((
            c,
            AudioInput {
                text: None,
                clips: vec![wave],
            },
        ));
    }
    // all clips in one call
    let all: Vec<AudioInput> = inputs.iter().map(|(_, i)| i.clone()).collect();
    let t0 = std::time::Instant::now();
    let got = ea::embed_audio_inputs(&enc, &tower, &all).expect("embed");
    eprintln!(
        "embedded {} clips in {:.3}s",
        all.len(),
        t0.elapsed().as_secs_f64()
    );
    let mut worst = (1.0f64, String::new());
    for ((c, _), g) in inputs.iter().zip(&got) {
        let row = c["row"].as_u64().unwrap() as usize;
        let r = &refs[row * d..(row + 1) * d];
        for dim in MATRYOSHKA_DIMS {
            let a = matryoshka(g, dim).unwrap();
            let cs = cosine(&a, &r[..dim]);
            if cs < worst.0 {
                worst = (cs, format!("{} @{dim}", c["id"]));
            }
            assert!(
                cs >= floor,
                "{} dim {dim}: cosine {cs:.7} < {floor}",
                c["id"]
            );
        }
        let cs = cosine(g, r);
        eprintln!("{} cos {cs:.9} (1 - cos = {:.2e})", c["id"], 1.0 - cs);
    }
    eprintln!("worst {:.7} ({}), floor {floor}", worst.0, worst.1);
    // a clip embeds alone as it does in a batch
    let alone = ea::embed_audio_inputs(&enc, &tower, &all[..1]).unwrap();
    let cs = cosine(&alone[0], &got[0]);
    assert!(cs > 0.999_999, "batch vs single: {cs}");
    // the same clip through a text placeholder (no prompt) is the same input
    let via_text = ea::embed_audio_inputs(
        &enc,
        &tower,
        &[AudioInput {
            text: Some(TextInput::plain(ea::PLACEHOLDER)),
            clips: all[0].clips.clone(),
        }],
    )
    .unwrap();
    let cs = cosine(&via_text[0], &got[0]);
    assert!(cs > 0.999_999, "placeholder text vs audio-only: {cs}");

    // ── the extra cases
    let xdir = refdir.join("audio_extra");
    let Ok(xm) = std::fs::read(xdir.join("manifest.json")) else {
        eprintln!("no {}: extra cases skipped", xdir.display());
        return;
    };
    let xm: serde_json::Value = serde_json::from_slice(&xm).unwrap();
    let (_, xrefs) = read_npy_f32(&xdir.join("embeddings.npy"));
    let mut ins = Vec::new();
    let mut resampled = Vec::new();
    for c in xm["cases"].as_array().unwrap() {
        let o = &c["ours"];
        let clips: Vec<Vec<f32>> = o["audio"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| ea::read_audio(&xdir.join(p.as_str().unwrap())).unwrap())
            .collect();
        let text = o["text"].as_str().map(|t| TextInput {
            text: t.to_string(),
            prompt_name: o["prompt_name"].as_str().map(str::to_string),
            ..Default::default()
        });
        let inp = AudioInput { text, clips };
        let n: Vec<usize> = inp.clips.iter().map(|c| ea::num_tokens(c.len())).collect();
        let ids = ea::input_ids(
            &enc,
            (tower.audio_token, tower.boa_token, tower.eoa_token),
            &inp,
            &n,
        )
        .unwrap();
        let want: Vec<u32> = c["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(ids, want, "token ids of {}", c["id"]);
        let wav_rate = o["audio"].as_array().unwrap().iter().any(|p| {
            let b = std::fs::read(xdir.join(p.as_str().unwrap())).unwrap();
            u32::from_le_bytes([b[24], b[25], b[26], b[27]]) != ea::SAMPLE_RATE
                || u16::from_le_bytes([b[22], b[23]]) != 1
        });
        resampled.push(wav_rate);
        ins.push(inp);
    }
    let t0 = std::time::Instant::now();
    let got = ea::embed_audio_inputs(&enc, &tower, &ins).expect("embed extra");
    eprintln!(
        "embedded {} extra inputs in {:.3}s",
        ins.len(),
        t0.elapsed().as_secs_f64()
    );
    for ((c, g), &rs) in xm["cases"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&got)
        .zip(&resampled)
    {
        let row = c["row"].as_u64().unwrap() as usize;
        let r = &xrefs[row * d..(row + 1) * d];
        let fl = if rs { floor.min(0.9999) } else { floor };
        for dim in MATRYOSHKA_DIMS {
            let cs = cosine(&matryoshka(g, dim).unwrap(), &r[..dim]);
            assert!(cs >= fl, "{} dim {dim}: cosine {cs:.7} < {fl}", c["id"]);
        }
        let cs = cosine(g, r);
        eprintln!(
            "{} cos {cs:.9} (1 - cos = {:.2e}){}",
            c["id"],
            1.0 - cs,
            if rs { " [resampled]" } else { "" }
        );
    }
}
