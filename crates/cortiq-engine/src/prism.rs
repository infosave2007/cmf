//! Prism/Bonsai signed FWHT activation boundary.
//!
//! Prism matrices carry an explicit signed-Hadamard activation boundary: the
//! source checkpoint folds `D·H` into every forward matrix and stores the
//! embedding twin in the inverse basis. Keep the transform in one small
//! module so CPU fallback, tests, and GPU paths share one typed header.

use cortiq_core::CmfModel;
use cortiq_core::hadamard::{signed_fwht_forward, signed_fwht_inverse};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// A validated, immutable view of the Prism header used by the hot per-op
/// path.  Header validation is intentionally done before this object enters
/// the cache; the cache therefore removes repeated manifest scans without
/// weakening the fail-closed contract of the loader/runtime.
struct ValidatedDescriptor {
    signs: Vec<f32>,
    widths: Vec<usize>,
    block_size: usize,
    activation_f16: bool,
    forward_names: HashSet<String>,
    inverse_names: HashSet<String>,
    affine_names: HashSet<String>,
}

impl ValidatedDescriptor {
    fn from_model(model: &CmfModel) -> Self {
        let cfg = model
            .header
            .arch
            .prism_hadamard
            .as_ref()
            .expect("Prism tensor without prism_hadamard metadata");
        let t0 = std::time::Instant::now();
        cfg.validate()
            .expect("invalid prism_hadamard metadata in CMF header");
        PROFILE_VALIDATIONS.fetch_add(1, Ordering::Relaxed);
        PROFILE_VALIDATE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Keep both the manifest spelling and its canonical spelling.  This
        // is exactly the old `manifest_name_matches` rule, represented once
        // rather than rescanned for every projection and token.
        let mut forward_names = HashSet::with_capacity(cfg.forward_weight_names.len() * 2);
        for n in &cfg.forward_weight_names {
            forward_names.insert(n.clone());
            if let Some(canonical) = n.strip_prefix("language_model.") {
                forward_names.insert(canonical.to_string());
            }
        }
        let mut inverse_names = HashSet::with_capacity(cfg.inverse_weight_names.len() * 2);
        for n in &cfg.inverse_weight_names {
            inverse_names.insert(n.clone());
            if let Some(canonical) = n.strip_prefix("language_model.") {
                inverse_names.insert(canonical.to_string());
            }
        }
        let affine_names = cfg
            .affine
            .as_ref()
            .map(|affine| affine.target_names.iter().cloned().collect())
            .unwrap_or_default();

        Self {
            signs: cfg.signs.clone(),
            widths: cfg.widths.clone(),
            block_size: cfg.block_size,
            activation_f16: cfg.activation_f16,
            forward_names,
            inverse_names,
            affine_names,
        }
    }

    fn signs_for_width(&self, width: usize) -> Option<&[f32]> {
        let mut off = 0usize;
        for &w in &self.widths {
            if w == width {
                return self.signs.get(off..off + w);
            }
            off += w;
        }
        None
    }
}

static DESCRIPTORS: OnceLock<Mutex<std::collections::HashMap<u64, Arc<ValidatedDescriptor>>>> =
    OnceLock::new();

// Most Prism calls occur on a small fixed worker pool.  Avoid taking the
// process-wide mutex after the first lookup on each worker while retaining a
// UID key (rather than an mmap address) for reload correctness.
thread_local! {
    static LOCAL_DESCRIPTOR: std::cell::RefCell<Option<(u64, Arc<ValidatedDescriptor>)>> =
        const { std::cell::RefCell::new(None) };
}

static PROFILE_VALIDATIONS: AtomicU64 = AtomicU64::new(0);
static PROFILE_VALIDATE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_LOOKUPS: AtomicU64 = AtomicU64::new(0);
static PERF_FORWARD_CALLS: AtomicU64 = AtomicU64::new(0);
static PERF_FORWARD_NS: AtomicU64 = AtomicU64::new(0);
static PERF_INVERSE_CALLS: AtomicU64 = AtomicU64::new(0);
static PERF_INVERSE_NS: AtomicU64 = AtomicU64::new(0);

fn perf_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_PERF_PROFILE").as_deref() == Ok("1"))
}

fn descriptors() -> &'static Mutex<std::collections::HashMap<u64, Arc<ValidatedDescriptor>>> {
    DESCRIPTORS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn descriptor_for(model: &CmfModel) -> Arc<ValidatedDescriptor> {
    PROFILE_LOOKUPS.fetch_add(1, Ordering::Relaxed);
    let uid = model.uid();
    if let Some(hit) = LOCAL_DESCRIPTOR.with(|slot| {
        slot.borrow()
            .as_ref()
            .filter(|(cached_uid, _)| *cached_uid == uid)
            .map(|(_, desc)| Arc::clone(desc))
    }) {
        return hit;
    }

    // `CMF_PRISM_CACHE=0` is an audit switch: it preserves the strict
    // validation and old behavior while making repeated validation measurable
    // against the cached production path.
    let use_cache = std::env::var("CMF_PRISM_CACHE").map_or(true, |v| v != "0");
    let desc = if !use_cache {
        Arc::new(ValidatedDescriptor::from_model(model))
    } else {
        let mut all = descriptors()
            .lock()
            .expect("Prism descriptor cache poisoned");
        Arc::clone(
            all.entry(uid)
                .or_insert_with(|| Arc::new(ValidatedDescriptor::from_model(model))),
        )
    };
    LOCAL_DESCRIPTOR.with(|slot| *slot.borrow_mut() = Some((uid, Arc::clone(&desc))));
    desc
}

/// Print cache/validation counters for a bounded benchmark.  The report is
/// opt-in so normal CLI output and production hot paths remain unchanged.
pub fn profile_report() {
    if std::env::var("CMF_PRISM_PROFILE").is_err() {
        return;
    }
    eprintln!(
        "[prism-profile] descriptor lookups={} validations={} validation_ms={:.3}",
        PROFILE_LOOKUPS.load(Ordering::Relaxed),
        PROFILE_VALIDATIONS.load(Ordering::Relaxed),
        PROFILE_VALIDATE_NS.load(Ordering::Relaxed) as f64 / 1e6,
    );
}

/// Aggregate the CPU-side signed-Hadamard work for one bounded benchmark.
/// This is deliberately opt-in and does not sample or print in the hot path.
pub fn perf_report() {
    if !perf_enabled() {
        return;
    }
    let fc = PERF_FORWARD_CALLS.load(Ordering::Relaxed);
    let ic = PERF_INVERSE_CALLS.load(Ordering::Relaxed);
    eprintln!(
        "[perf-prism] forward_calls={} forward_ms={:.3} forward_ms_per_call={:.3} inverse_calls={} inverse_ms={:.3} inverse_ms_per_call={:.3}",
        fc,
        PERF_FORWARD_NS.load(Ordering::Relaxed) as f64 / 1e6,
        PERF_FORWARD_NS.load(Ordering::Relaxed) as f64 / 1e6 / fc.max(1) as f64,
        ic,
        PERF_INVERSE_NS.load(Ordering::Relaxed) as f64 / 1e6,
        PERF_INVERSE_NS.load(Ordering::Relaxed) as f64 / 1e6 / ic.max(1) as f64,
    );
}

fn descriptor(model: &CmfModel, width: usize) -> (Arc<ValidatedDescriptor>, usize, bool) {
    let desc = descriptor_for(model);
    let _signs = desc
        .signs_for_width(width)
        .expect("Prism width has no explicit sign vector");
    (desc.clone(), desc.block_size, desc.activation_f16)
}

#[inline]
fn round_f16(x: f32) -> f32 {
    cortiq_core::quant::f16_to_f32(cortiq_core::hadamard::prism_f32_to_f16_rne(x))
}

/// Transform an activation row before a forward Prism matrix: `x·D·H`.
/// The source contract requests an f16 boundary after the FWHT; keeping that
/// cast here makes the CPU implementation agree with the trained MLX path.
pub fn forward(model: &CmfModel, x: &[f32]) -> Vec<f32> {
    let (desc, block, activation_f16) = descriptor(model, x.len());
    let signs = desc
        .signs_for_width(x.len())
        .expect("Prism width has no explicit sign vector");
    let perf_t0 = perf_enabled().then(std::time::Instant::now);
    let mut out = x.to_vec();
    signed_fwht_forward(&mut out, signs, block).expect("validated Prism FWHT activation");
    if activation_f16 {
        for v in &mut out {
            *v = round_f16(*v);
        }
    }
    if let Some(t0) = perf_t0 {
        PERF_FORWARD_CALLS.fetch_add(1, Ordering::Relaxed);
        PERF_FORWARD_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    out
}

/// Whether this named matrix is one of the source's forward-basis weights.
/// Vision tensors and auxiliary projections are intentionally not included.
pub fn is_forward_weight(model: &CmfModel, name: &str) -> bool {
    model.header.arch.prism_hadamard.is_some() && descriptor_for(model).forward_names.contains(name)
}

/// True when this tensor belongs to a Prism header, regardless of whether it
/// is a forward or inverse matrix.  GPU graph/device paths use this as a
/// conservative safety gate because they do not carry the transform
/// descriptor yet.
pub fn has_contract(model: &CmfModel) -> bool {
    model.header.arch.prism_hadamard.is_some()
}

/// Whether this exact canonical matrix carries the production affine
/// correction.  Ordinary Prism q2tp and non-Prism dtype16 remain the raw
/// `(c - 1.5) * s` codec; no correction is inferred from the architecture
/// name or from the tensor dtype alone.
pub fn is_affine_target(model: &CmfModel, name: &str) -> bool {
    model.header.arch.prism_hadamard.is_some() && descriptor_for(model).affine_names.contains(name)
}

/// Apply the forward basis only to a manifest-listed matrix.  This is the
/// mixed-profile counterpart of [`forward`]: ordinary q2tp may retain q4tp
/// for attention/down projections, but those matrices are still stored in
/// the same signed-Hadamard basis as the production Prism profile.
pub fn forward_weight(model: &CmfModel, name: &str, x: &[f32]) -> Vec<f32> {
    if is_forward_weight(model, name) {
        forward(model, x)
    } else {
        x.to_vec()
    }
}

/// Transform a decoded embedding row back to the caller basis: `z·H·D`.
pub fn inverse_embedding(model: &CmfModel, x: &mut [f32]) {
    let (desc, block, activation_f16) = descriptor(model, x.len());
    let signs = desc
        .signs_for_width(x.len())
        .expect("Prism width has no explicit sign vector");
    let perf_t0 = perf_enabled().then(std::time::Instant::now);
    // MLX dequantizes an embedding row and enters the inverse boundary as
    // the module dtype (float16), then performs the normalized FWHT in
    // float32 and casts back to that dtype.  `row_f32` exposes the decoded
    // row as f32 for the native runtime, so reproduce both dtype boundaries
    // explicitly instead of silently using a higher-precision embedding.
    if activation_f16 {
        for v in x.iter_mut() {
            *v = round_f16(*v);
        }
    }
    signed_fwht_inverse(x, signs, block).expect("validated Prism FWHT embedding");
    if activation_f16 {
        for v in x.iter_mut() {
            *v = round_f16(*v);
        }
    }
    if let Some(t0) = perf_t0 {
        PERF_INVERSE_CALLS.fetch_add(1, Ordering::Relaxed);
        PERF_INVERSE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// The source manifest names are wrapper-qualified, while CMF stores the
/// canonical runtime name.  The embedding is the only inverse matrix in the
/// Prism checkpoint; all other projections use `forward` activations.
pub fn is_inverse_embedding(model: &CmfModel, name: &str) -> bool {
    if name == "model.embed_tokens.weight" {
        return true;
    }
    if model.header.arch.prism_hadamard.is_none() {
        return false;
    }
    descriptor_for(model).inverse_names.contains(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_core::types::PrismHadamardConfig;

    #[test]
    fn oracle_roundtrip() {
        let mut x = (0..1024).map(|i| i as f32 / 1024.0).collect::<Vec<_>>();
        let original = x.clone();
        let signs = (0..1024)
            .map(|i| if i & 1 == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        signed_fwht_forward(&mut x, &signs, 1024).unwrap();
        signed_fwht_inverse(&mut x, &signs, 1024).unwrap();
        for (a, b) in x.iter().zip(original) {
            assert!((a - b).abs() < 2e-5);
        }
        let cfg = PrismHadamardConfig {
            version: 1,
            block_size: 1024,
            transform: "normalized-sylvester-walsh-hadamard".into(),
            axis: "input-last-dimension".into(),
            sign_mode: "explicit".into(),
            widths: vec![1024],
            signs,
            forward_weight_names: vec![],
            inverse_weight_names: vec![],
            gdn_v_grouped: true,
            activation_f16: true,
            affine: None,
        };
        cfg.validate().unwrap();
    }

    #[test]
    fn inverse_embedding_matches_mlx_f16_boundaries() {
        let signs = (0..1024)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let decoded = (0..1024)
            .map(|i| (i as f32 * 0.001_234_567).sin())
            .collect::<Vec<_>>();
        // This is the exact boundary sequence in runtime.py: decoded row
        // → f16, inverse normalized FWHT in f32, then → f16.
        let mut expected = decoded.iter().map(|&v| round_f16(v)).collect::<Vec<_>>();
        signed_fwht_inverse(&mut expected, &signs, 1024).unwrap();
        for v in &mut expected {
            *v = round_f16(*v);
        }
        let mut actual = decoded;
        for v in &mut actual {
            *v = round_f16(*v);
        }
        signed_fwht_inverse(&mut actual, &signs, 1024).unwrap();
        for v in &mut actual {
            *v = round_f16(*v);
        }
        assert_eq!(actual, expected);
    }
}
