//! Toy BERT (spec §6.1): 2 layers, hidden 32, 4 heads, intermediate 48, from
//! `tools/mk_decision_toy.py` in the exact export format of
//! `tools/decision_export_encoder.py`.
//!
//! * the native tokenizer reproduces the HF `tokenizers` ids of the toy
//!   tokenizer, the native forward the numpy float64 `last_hidden_state` and φ_P
//!   (max |Δ| ≤ 1e-5);
//! * `cortiq decision init` tensors → a decision file → the loaded encoder
//!   passes its golden check bit for bit and encodes like the export;
//! * the result of the batch encoder does not depend on the thread count;
//! * the encoder never touches a GPU backend: no event from
//!   `cortiq_engine::gpu*` (Metal and wgpu log their initialisation), the
//!   GPU-capable `gemm_nt` is never entered (`fcd::prof::GEMM_CALLS`), no Metal
//!   command buffer is submitted. The check is meaningful when `CMF_GPU` is
//!   unset (the default selects Metal on macOS); with `CMF_GPU=0` no backend
//!   can come up at all.

use cortiq_decision::bert::{EncoderExport, GOLDEN_MAX_ABS};
use cortiq_decision::container::{DecisionModel, FileBuilder, Verify};
use cortiq_decision::manifest::{
    DEFAULT_ENCODER_GOLDEN_TEXTS, DEFAULT_MODEL_ID, DEFAULT_NAME, ENCODER_GOLDEN_COUNT,
    GOLDEN_TENSOR,
};
use cortiq_decision::signal::{self, SignalEncoder};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::Ordering;

fn toy_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy")
}

fn export() -> EncoderExport {
    EncoderExport::read(toy_dir().join("encoder")).expect("toy export")
}

fn golden() -> Value {
    serde_json::from_slice(&std::fs::read(toy_dir().join("encoder_golden.json")).unwrap()).unwrap()
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn toy_forward_matches_numpy_f64() {
    let ex = export();
    let enc = ex.encoder().unwrap();
    let g = golden();
    let tol = g["tolerance"].as_f64().unwrap();
    assert!(tol <= 1e-5);
    let dim = enc.dim();
    assert_eq!(dim, 32);
    let (mut worst_h, mut worst_p) = (0.0f64, 0.0f64);
    let rows = g["rows"].as_array().unwrap();
    assert!(rows.len() >= 10);
    let mut truncated = false;
    for r in rows {
        let text = r["text"].as_str().unwrap();
        let ids: Vec<u32> = r["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(enc.tokenize(text), ids, "toy tokenizer ids of {text:?}");
        truncated |= ids.len() == g["max_length"].as_u64().unwrap() as usize;
        let hidden = enc.model().forward(&ids);
        let want_h = f64s(&r["hidden"]);
        assert_eq!(hidden.len(), want_h.len());
        for (a, b) in hidden.iter().zip(&want_h) {
            worst_h = worst_h.max((*a as f64 - b).abs());
        }
        let phi = enc.encode(text);
        let want_p = f64s(&r["phi_p"]);
        assert_eq!(phi.len(), dim);
        for (a, b) in phi.iter().zip(&want_p) {
            worst_p = worst_p.max((*a as f64 - b).abs());
        }
        let n = phi.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
        assert!((n - 1.0).abs() < 1e-6, "φ_P is unit length");
    }
    assert!(truncated, "one toy text is truncated at max_length");
    eprintln!("toy BERT vs numpy f64: max |Δhidden| {worst_h:.3e}, max |Δφ_P| {worst_p:.3e}");
    assert!(
        worst_h <= tol,
        "last_hidden_state max |Δ| {worst_h:e} > {tol:e}"
    );
    assert!(worst_p <= tol, "φ_P max |Δ| {worst_p:e} > {tol:e}");
}

#[test]
fn toy_decision_file_passes_the_golden_check() {
    let ex = export();
    let init = ex.init_tensors(&DEFAULT_ENCODER_GOLDEN_TEXTS).unwrap();
    assert_eq!(init.golden.len(), ENCODER_GOLDEN_COUNT * 32);
    let golden_tensor = init
        .tensors
        .iter()
        .find(|t| t.name == GOLDEN_TENSOR)
        .expect("golden tensor");
    assert_eq!(golden_tensor.shape, vec![ENCODER_GOLDEN_COUNT, 32]);
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("toy-encoder.cmf");
    let report = FileBuilder::new(
        DEFAULT_MODEL_ID,
        DEFAULT_NAME,
        init.record.clone(),
        init.tensors,
    )
    .unwrap()
    .write(&out)
    .unwrap();
    assert!(report.bytes > 0);
    let model = DecisionModel::open(&out, Verify::Full).unwrap();
    assert_eq!(model.encoder_dim(), 32);
    assert_eq!(model.signal_dim(), 32 + 4096);
    let (se, golden) = SignalEncoder::from_model(&model).unwrap();
    assert_eq!(golden.rows, ENCODER_GOLDEN_COUNT);
    assert_eq!(golden.bit_exact_rows, ENCODER_GOLDEN_COUNT);
    assert_eq!(golden.max_abs, 0.0);
    assert!(se.encoder().warnings().is_empty());
    assert_eq!(se.dim(), 32 + 4096);
    // The file's encoder encodes exactly like the export's.
    let enc = ex.encoder().unwrap();
    for t in ["How do I top up my card?", "", "野口 [MASK] ΣΑΣ"] {
        let f = se.features(t);
        assert_eq!(
            f.phi_p.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            enc.encode(t)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(f.phi_h, signal::phi_h(t));
        let x = se.signal(t);
        assert_eq!(x.len(), 32 + 4096);
        assert_eq!(&x[..32], f.phi_p.as_slice());
        for (a, b) in x[32..].iter().zip(&f.phi_h) {
            assert_eq!(a.to_bits(), (0.5 * b).to_bits());
        }
    }
    // A stored golden that is off by more than the tolerance is refused.
    let stored = model.encoder_golden().unwrap().into_owned();
    let mut bad = stored.clone();
    bad[5] += 10.0 * GOLDEN_MAX_ABS;
    let texts = &model.representation().encoder.golden_texts;
    assert!(se.encoder().check_golden(texts, &stored).is_ok());
    let err = se.encoder().check_golden(texts, &bad).unwrap_err();
    assert!(err.to_string().contains("golden mismatch"), "{err}");
}

#[test]
fn batch_encoding_does_not_depend_on_threads() {
    let enc = export().encoder().unwrap();
    let se = SignalEncoder::new(enc);
    let texts: Vec<String> = (0..37)
        .map(|i| {
            format!(
                "text {i}: top up card {} café {}",
                i * 7,
                "x ".repeat(i % 9)
            )
        })
        .collect();
    let one = se.features_batch(&texts, 1);
    for threads in [2, 3, 8] {
        let many = se.features_batch(&texts, threads);
        assert_eq!(one.len(), many.len());
        for (a, b) in one.iter().zip(&many) {
            assert!(a.bit_eq(b), "threads {threads}");
        }
    }
    let flat = se.signals_batch(&texts, 4);
    assert_eq!(flat.len(), texts.len() * se.dim());
    for (i, f) in one.iter().enumerate() {
        let row = &flat[i * se.dim()..(i + 1) * se.dim()];
        assert!(
            row.iter()
                .zip(f.signal())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
    // Two runs in one process are bit-identical.
    let again = se.features_batch(&texts, 1);
    assert!(one.iter().zip(&again).all(|(a, b)| a.bit_eq(b)));
}

#[test]
fn export_directory_is_checked() {
    let src = toy_dir().join("encoder");
    let dir = tempfile::tempdir().unwrap();
    let copy = |dst: &std::path::Path| {
        std::fs::create_dir(dst).unwrap();
        for e in std::fs::read_dir(&src).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), dst.join(e.file_name())).unwrap();
        }
    };
    // A tampered weight file.
    let d1 = dir.path().join("tampered");
    copy(&d1);
    let p = d1.join("layer.1.output.dense.weight.npy");
    let mut b = std::fs::read(&p).unwrap();
    let last = b.len() - 1;
    b[last] ^= 1;
    std::fs::write(&p, b).unwrap();
    let err = EncoderExport::read(&d1).unwrap_err();
    assert!(format!("{err:#}").contains("sha256"), "{err:#}");
    // A changed vocab.
    let d2 = dir.path().join("vocab");
    copy(&d2);
    let mut v = std::fs::read(d2.join("vocab.txt")).unwrap();
    v.extend(b"extra\n");
    std::fs::write(d2.join("vocab.txt"), v).unwrap();
    assert!(EncoderExport::read(&d2).is_err());
    // A missing weight.
    let d3 = dir.path().join("missing");
    copy(&d3);
    std::fs::remove_file(d3.join("layer.0.attention.self.key.bias.npy")).unwrap();
    assert!(EncoderExport::read(&d3).is_err());
    // Wrong golden count.
    assert!(export().init_tensors(&["only one"]).is_err());
}

// ------------------------------------------------------------------ the GPU is never touched

static GPU_EVENTS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Records the target of every event a `cortiq_engine::gpu*` module emits.
struct GpuWatch;

impl tracing::Subscriber for GpuWatch {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let target = event.metadata().target();
        if target.starts_with("cortiq_engine::gpu") {
            GPU_EVENTS.lock().unwrap().push(target.to_string());
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

#[test]
fn encoder_never_touches_the_gpu() {
    let _ = tracing::subscriber::set_global_default(GpuWatch);
    let gemm_calls = cortiq_engine::fcd::prof::GEMM_CALLS.load(Ordering::Relaxed);
    #[cfg(target_os = "macos")]
    let submits = cortiq_engine::gpu_metal::METAL_SUBMITS.load(Ordering::Relaxed);
    let ex = export();
    let se = SignalEncoder::new(ex.encoder().unwrap());
    let long = "a b c d e f g h i j k l m n o p q r s t u v w x y z";
    for t in ["How do I top up my card?", long, ""] {
        let v = se.signal(t);
        assert!(v.iter().all(|x| x.is_finite()));
    }
    let _ = se.features_batch(&["one", "two", "three", long], 3);
    let init = ex.init_tensors(&DEFAULT_ENCODER_GOLDEN_TEXTS).unwrap();
    assert_eq!(init.golden.len(), ENCODER_GOLDEN_COUNT * se.dim_p());
    assert_eq!(
        cortiq_engine::fcd::prof::GEMM_CALLS.load(Ordering::Relaxed),
        gemm_calls,
        "the encoder entered the GPU-capable gemm_nt"
    );
    #[cfg(target_os = "macos")]
    {
        assert_eq!(
            cortiq_engine::gpu_metal::METAL_SUBMITS.load(Ordering::Relaxed),
            submits,
            "a Metal command buffer was submitted"
        );
        assert!(cortiq_engine::gpu_metal::initialization_error().is_none());
    }
    let events = GPU_EVENTS.lock().unwrap().clone();
    assert!(
        events.is_empty(),
        "a GPU backend was initialised or used: {events:?}"
    );
    eprintln!(
        "GPU untouched (CMF_GPU={:?}): 0 gpu events, gemm_nt calls unchanged",
        std::env::var("CMF_GPU").ok()
    );
}
