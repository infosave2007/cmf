//! The organism's "day" side (docs/NATIVE_MODEL_TECH.ru.md §2): every
//! request touches an idle marker, and requests the skill router finds
//! out-of-distribution (min recon error > τ, or no skills yet) go to an
//! append-only buffer the sleep daemon (`cortiq-embryo sleep`) bakes new
//! skills from during idle time. Enabled by `CMF_OOD_DIR`; `CMF_OOD_TAU`
//! (default 0.30) sets the OOD threshold.

use cortiq_core::CmfModel;
use cortiq_engine::pipeline::Pipeline;
use std::io::Write;
use std::path::PathBuf;

pub fn ood_dir() -> Option<PathBuf> {
    std::env::var("CMF_OOD_DIR").ok().map(PathBuf::from)
}

fn tau() -> f32 {
    std::env::var("CMF_OOD_TAU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.30)
}

/// Touch the idle marker (mtime = last request).
pub fn touch_last_request() {
    let Some(dir) = ood_dir() else { return };
    let _ = std::fs::create_dir_all(&dir);
    let p = dir.join("last_request");
    let _ = std::fs::write(&p, format!("{}\n", unix_now()));
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Route the prompt through the file's skill descriptors; if it is OOD
/// (or the file has no routable skill), append it to the buffer.
/// Returns (best skill, its E) when routable.
pub fn record_if_ood(
    model: &CmfModel,
    pipe: &mut Pipeline,
    prompt_ids: &[u32],
    prompt_text: &str,
) -> Option<(String, f32)> {
    let dir = ood_dir()?;
    // calibrated files decide by the novelty ensemble (cortiq-router recipe);
    // uncalibrated ones by E_min > τ
    let r = cortiq_engine::router::route_full(model, pipe, prompt_ids, tau());
    let best = r.scores.first().map(|s| (s.id.clone(), s.error));
    let ood = r.is_novel;
    if ood {
        let _ = std::fs::create_dir_all(&dir);
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("buffer.jsonl"))
        {
            let rec = serde_json::json!({
                "ts": unix_now(),
                "e_min": best.as_ref().map(|(_, e)| *e),
                "nearest": best.as_ref().map(|(id, _)| id.clone()),
                "novelty": if r.novelty.is_finite() { Some(r.novelty) } else { None },
                "confidence": r.confidence,
                "calibrated": r.calibrated,
                "tokens": prompt_ids.len(),
                "text": prompt_text,
            });
            let _ = writeln!(f, "{}", rec);
        }
    }
    best
}

/// Router-v2 twin of [`record_if_ood`]: the decision was already made on
/// the dedicated probe pipeline (no second φ prefill, no reset of the
/// generation slot's state); a NOVEL request goes to the buffer.
pub fn record_decision(
    d: &cortiq_engine::router::RouteDecision,
    prompt_ids: &[u32],
    prompt_text: &str,
) {
    let Some(dir) = ood_dir() else { return };
    if !d.routing.is_novel {
        return;
    }
    let _ = std::fs::create_dir_all(&dir);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("buffer.jsonl"))
    {
        let mut rec = d.summary_json();
        rec["ts"] = serde_json::json!(unix_now());
        rec["nearest"] = serde_json::json!(d.nearest_skill());
        rec["calibrated"] = serde_json::json!(d.routing.calibrated);
        rec["tokens"] = serde_json::json!(prompt_ids.len());
        rec["text"] = serde_json::json!(prompt_text);
        let _ = writeln!(f, "{}", rec);
    }
}
