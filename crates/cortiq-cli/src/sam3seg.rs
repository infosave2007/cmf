//! `cortiq segment` — SAM3 image inference: a PNG + text prompt → a
//! mask-overlay PNG and JSON with normalized boxes and scores.
//!
//! The pipeline is the reference one, end to end:
//!
//! 1. decode the image → [`sam3::preprocess`] (resize 1008², bilinear,
//!    mean/std 0.5 — the vendor `cfg.processor`, the parity fixtures'
//!    `pixels.f32` recipe);
//! 2. tokenize the prompt with the CMF's embedded CLIP tokenizer and
//!    frame it exactly as `Sam3Processor` does: `<|startoftext|>` …
//!    `endftext`, padded to 32 with `endftext`, attention mask over the
//!    real tokens only;
//! 3. one CPU forward: `Sam3::forward`;
//! 4. score every query `sigmoid(logit) · sigmoid(presence)` (HF
//!    `post_process_object_detection`), keep `score ≥ --thr`, sort,
//!    cut at `--topk`;
//! 5. draw the kept masks (sigmoid ≥ `--mask-thr`, upsampled from the
//!    head's 288² logits) over the original and emit the JSON.
//!
//! The e2e test at the bottom runs the WHOLE thing on the synthetic
//! parity image (white + red square + blue stripe, prompt
//! "red square") and gates the box, the score and the mask IoU against
//! the checked-in torch fixtures.

use anyhow::{Context, Result};
use cortiq_core::CmfModel;
use cortiq_engine::media::{self, RgbFrame};
use cortiq_engine::sam3::{self, Sam3};
use cortiq_engine::tokenizer::Tokenizer;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// SAM3 sequence length (the text tower's positions; Sam3Processor
/// pads/truncates to `max_length = 32`).
const MAX_LEN: usize = 32;
/// The mask head's logit resolution (both `cfg.processor.mask_size`
/// and the fixtures' `pred_masks` are 288×288).
const MASK_SIDE: usize = 288;
/// Instance colors for the overlay (fixed order → stable between runs).
/// First is a vivid teal: the top-scored object must stay visible on
/// content of any hue (red-on-red would be near-invisible).
const PALETTE: [[u8; 3]; 12] = [
    [42, 226, 180],
    [230, 57, 70],
    [244, 162, 97],
    [69, 123, 157],
    [231, 111, 81],
    [29, 53, 87],
    [233, 196, 106],
    [140, 122, 230],
    [222, 93, 176],
    [111, 201, 152],
    [88, 101, 42],
    [15, 133, 84],
];

/// The `cortiq segment` arguments (the clap fields live in `main.rs`).
pub struct SegArgs {
    /// Path to the SAM3 .cmf.
    pub model: String,
    /// Input image (anything `media::decode_rgb` reads: PNG/JPEG/…).
    pub image: String,
    /// Text prompt, e.g. `"red square"`.
    pub prompt: String,
    /// Overlay PNG path; `None` = `<image stem>_seg.png` beside the input.
    pub out: Option<String>,
    /// Write the JSON here instead of stdout.
    pub json: Option<String>,
    /// Keep at most this many objects (sorted by score).
    pub topk: usize,
    /// Score threshold; default 0.5 = the vendor's
    /// `score_threshold_detection`.
    pub thr: f32,
    /// Mask binarization threshold on the sigmoid (HF default 0.5).
    pub mask_thr: f32,
}

/// One kept detection.
pub struct Detection {
    /// The decoder query it came from (for cross-checking fixtures).
    pub query: usize,
    /// `sigmoid(pred_logits[q]) * sigmoid(presence_logit)`.
    pub score: f32,
    /// `xyxy`, normalized to the input image (as the model predicts).
    pub box_xyxy: [f32; 4],
    /// Binarized mask, `MASK_SIDE²` row-major.
    pub mask: Vec<u8>,
}

/// Everything one `segment` run produced.
pub struct SegOutput {
    pub detections: Vec<Detection>,
    pub image_size: (usize, usize),
    pub preprocess_ms: f64,
    pub forward_ms: f64,
    pub total_ms: f64,
    pub overlay_path: PathBuf,
    pub json: serde_json::Value,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Tokenize `prompt` into the SAM3 frame: `[cls] + ids + [eot]` padded
/// to [`MAX_LEN`] with `eot`, plus the attention mask (`1` = real
/// token, `0` = padding). Matches `Sam3Processor`'s
/// `tokenizer(text, padding="max_length", max_length=32)`.
pub fn tokenize_prompt(tok: &Tokenizer, prompt: &str) -> (Vec<u32>, Vec<u8>) {
    let cls = tok.bos_token_id.unwrap_or(49_406);
    let eot = tok.eos_token_id.unwrap_or(49_407);
    let mut words = tok.encode(prompt);
    if words.len() + 2 > MAX_LEN {
        // The frame (cls + eot) is mandatory; long prompts lose the tail.
        words.truncate(MAX_LEN - 2);
    }
    let mut ids = Vec::with_capacity(MAX_LEN);
    ids.push(cls);
    ids.extend_from_slice(&words);
    ids.push(eot);
    let n_real = ids.len();
    ids.resize(MAX_LEN, eot);
    let mut attn = vec![1u8; n_real];
    attn.resize(MAX_LEN, 0);
    (ids, attn)
}

/// The overlay path when `--out` is absent: `photo.png` → `photo_seg.png`.
fn default_out(image: &str) -> PathBuf {
    let p = Path::new(image);
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("segment");
    p.with_file_name(format!("{stem}_seg.png"))
}

/// Run the whole image-inference pipeline (used by the CLI and the e2e
/// test alike). Writes the overlay PNG; the caller decides where the
/// JSON goes (`args.json`), the value is returned either way.
pub fn run(args: &SegArgs) -> Result<SegOutput> {
    let t_all = Instant::now();

    // ── image → [3,1008,1008] f32 ─────────────────────────────────
    let frame = media::read_rgb(Path::new(&args.image))
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("read image {}", args.image))?;
    let t0 = Instant::now();
    let pixels = sam3::preprocess(&frame);
    let preprocess_ms = t0.elapsed().as_secs_f64() * 1e3;

    // ── prompt → ids/attn ─────────────────────────────────────────
    let cm = CmfModel::open(&args.model).with_context(|| format!("open {}", args.model))?;
    let vocab = cm
        .vocab
        .clone()
        .context("the file carries no embedded tokenizer (VOCAB section)")?;
    let tok = Tokenizer::from_bytes(&vocab).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let (ids, attn) = tokenize_prompt(&tok, &args.prompt);

    // ── forward ───────────────────────────────────────────────────
    let mut m = Sam3::open(&args.model)?;
    let t1 = Instant::now();
    let out = m.forward(&pixels, &ids, &attn)?;
    let forward_ms = t1.elapsed().as_secs_f64() * 1e3;

    // ── scores → keep / sort / cut ────────────────────────────────
    anyhow::ensure!(
        !out.pred_boxes.is_empty() && out.pred_logits.len() == out.pred_boxes.len() / 4,
        "forward returned no scores/masks (SAM3_ONLY debug env set?)"
    );
    let presence = sigmoid(out.presence_logit);
    let mut cands: Vec<(usize, f32)> = out
        .pred_logits
        .iter()
        .enumerate()
        .map(|(q, &logit)| (q, sigmoid(logit) * presence))
        .collect();
    cands.retain(|&(_, s)| s > args.thr);
    cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    cands.truncate(args.topk.max(1));

    // ── masks ─────────────────────────────────────────────────────
    let nq = out.pred_boxes.len() / 4;
    let hw = out.pred_masks.len() / nq.max(1);
    anyhow::ensure!(hw == MASK_SIDE * MASK_SIDE, "unexpected mask shape {hw} (nq {nq})");
    let logit_thr = {
        let t = args.mask_thr.clamp(1e-6, 1.0 - 1e-6);
        (t / (1.0 - t)).ln()
    };
    let mut detections = Vec::with_capacity(cands.len());
    for (query, score) in cands {
        let base = query * hw;
        let mask: Vec<u8> = out.pred_masks[base..base + hw]
            .iter()
            .map(|&v| u8::from(v > logit_thr))
            .collect();
        detections.push(Detection {
            query,
            score,
            box_xyxy: [
                out.pred_boxes[query * 4],
                out.pred_boxes[query * 4 + 1],
                out.pred_boxes[query * 4 + 2],
                out.pred_boxes[query * 4 + 3],
            ],
            mask,
        });
    }

    // ── overlay PNG ───────────────────────────────────────────────
    let overlay_path = args.out.as_ref().map(PathBuf::from).unwrap_or_else(|| default_out(&args.image));
    let overlay = draw_overlay(&frame, &detections);
    media::save_png(&overlay_path, &overlay).map_err(anyhow::Error::msg)?;

    // ── JSON ──────────────────────────────────────────────────────
    let total_ms = t_all.elapsed().as_secs_f64() * 1e3;
    let objects: Vec<serde_json::Value> = detections
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let [x0, y0, x1, y1] = d.box_xyxy;
            json!({
                "rank": i,
                "box": [x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0)],
                "score": d.score,
                "label": args.prompt,
                "query": d.query,
            })
        })
        .collect();
    let value = json!({
        "model": args.model,
        "image": args.image,
        "prompt": args.prompt,
        "image_size": [frame.width, frame.height],
        "num_objects": objects.len(),
        "objects": objects,
        "params": {
            "threshold": args.thr,
            "topk": args.topk,
            "mask_threshold": args.mask_thr,
            "presence_score": presence,
        },
        "timing_ms": {
            "preprocess": (preprocess_ms * 100.0).round() / 100.0,
            "forward": (forward_ms * 100.0).round() / 100.0,
            "total": (total_ms * 100.0).round() / 100.0,
        },
        "overlay": overlay_path.display().to_string(),
    });

    if let Some(path) = &args.json {
        std::fs::write(path, serde_json::to_vec_pretty(&value)?)
            .with_context(|| format!("write {path}"))?;
    }

    Ok(SegOutput {
        detections,
        image_size: (frame.width, frame.height),
        preprocess_ms,
        forward_ms,
        total_ms,
        overlay_path,
        json: value,
    })
}

/// Composite every kept mask over the original image at 45 % opacity,
/// one stable palette color per instance. The 288² mask covers the
/// (resized) input, so it maps straight back onto the original pixels.
fn draw_overlay(frame: &RgbFrame, dets: &[Detection]) -> RgbFrame {
    let (w, h) = (frame.width, frame.height);
    let mut data = frame.data.clone();
    for (i, d) in dets.iter().enumerate() {
        let color = PALETTE[i % PALETTE.len()];
        for y in 0..MASK_SIDE {
            let sy = ((y * h) / MASK_SIDE).min(h - 1);
            for x in 0..MASK_SIDE {
                if d.mask[y * MASK_SIDE + x] == 0 {
                    continue;
                }
                let sx = ((x * w) / MASK_SIDE).min(w - 1);
                let o = (sy * w + sx) * 3;
                for c in 0..3 {
                    let base = data[o + c] as f32;
                    data[o + c] = (base * 0.55 + color[c] as f32 * 0.45).round() as u8;
                }
            }
        }
    }
    RgbFrame::new(w, h, data).expect("overlay buffer matches its frame")
}

/// Write a finished run's JSON where the CLI wants it: stdout when
/// `dest` is `None` (the file case is already written by [`run`]).
pub fn print_json_if_stdio(value: &serde_json::Value, dest: Option<&str>) -> Result<()> {
    if dest.is_none() {
        println!("{}", serde_json::to_string_pretty(value)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The synthetic parity image — byte-for-byte what
    /// `tools/sam3_ref.py::make_image` builds (384×384, white, a red
    /// square at rows/cols 64..192, a blue stripe at rows 300..340).
    fn synthetic_image() -> RgbFrame {
        const S: usize = 384;
        let mut data = vec![255u8; S * S * 3];
        for y in 64..192 {
            for x in 64..192 {
                let o = (y * S + x) * 3;
                data[o..o + 3].copy_from_slice(&[200, 30, 30]);
            }
        }
        for y in 300..340 {
            for x in 0..S {
                let o = (y * S + x) * 3;
                data[o..o + 3].copy_from_slice(&[30, 60, 200]);
            }
        }
        RgbFrame::new(S, S, data).unwrap()
    }

    fn f32s(p: &Path) -> Vec<f32> {
        let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn fixture_paths() -> (PathBuf, PathBuf) {
        let fx = std::env::var("SAM3_FX")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../cortiq-engine/tests/fixtures/sam3")
            });
        let cmf = std::env::var("SAM3_CMF").map(PathBuf::from).unwrap_or_else(|_| {
            ["/workspace/models/sam3-q8_2f.cmf", "/home/hermes/models/sam3-q8_2f.cmf"]
                .iter()
                .map(PathBuf::from)
                .find(|p| p.exists())
                .unwrap_or_else(|| PathBuf::from("/nonexistent/sam3.cmf"))
        });
        (fx, cmf)
    }

    /// IoU of two 288² binary masks.
    fn mask_iou(a: &[u8], b: &[u8]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut inter = 0u32;
        let mut union = 0u32;
        for (&x, &y) in a.iter().zip(b) {
            let x = x != 0;
            let y = y != 0;
            inter += u32::from(x && y);
            union += u32::from(x || y);
        }
        if union == 0 { 1.0 } else { inter as f32 / union as f32 }
    }

    /// The embedded CLIP tokenizer framed exactly as `Sam3Processor`
    /// frames a prompt — ids verified against the reference
    /// `tokenizers` implementation. Fast: no forward.
    #[test]
    fn clip_tokenizer_frame() {
        let (_, cmf) = fixture_paths();
        if !cmf.exists() {
            eprintln!("SKIP clip_tokenizer_frame: cmf not found: {}", cmf.display());
            return;
        }
        let cm = CmfModel::open(cmf.to_str().unwrap()).expect("open cmf");
        let tok = Tokenizer::from_bytes(cm.vocab.as_deref().expect("embedded vocab"))
            .expect("load embedded tokenizer");
        let frame = |p: &str| tokenize_prompt(&tok, p);
        let padded = |body: &[u32]| -> Vec<u32> {
            let mut v = vec![49_406];
            v.extend_from_slice(body);
            v.push(49_407);
            v.resize(MAX_LEN, 49_407);
            v
        };
        // (prompt, body ids) — from `tokenizers.Tokenizer.encode` minus
        // its own cls/sep frame.
        for (prompt, body) in [
            ("red square", vec![736u32, 3999]),
            // Lowercase normalization: "Red Square!" tokenizes as "red square!"
            ("Red Square!", vec![736, 3999, 256]),
            ("a cat and a dog", vec![320, 2368, 537, 320, 1929]),
            ("2+2=4", vec![273, 266, 273, 284, 275]),
            ("hello, world", vec![3306, 267, 1002]),
            ("blue stripe", vec![1746, 19368]),
        ] {
            let (ids, attn) = frame(prompt);
            assert_eq!(ids, padded(&body), "framed ids for {prompt:?}");
            let n_real = body.len() + 2;
            assert_eq!(
                attn.iter().filter(|&&a| a == 1).count(),
                n_real,
                "attention mask for {prompt:?}"
            );
        }
    }

    /// End-to-end: PNG + "red square" → box, score, mask — gated
    /// against the checked-in torch fixtures. One command:
    /// `cargo test --release -p cortiq-cli --bin cortiq sam3_segment_e2e -- --nocapture`
    #[test]
    fn sam3_segment_e2e() {
        let (fx, cmf) = fixture_paths();
        if !fx.join("pred_masks.f32").exists() {
            eprintln!("SKIP sam3_segment_e2e: fixtures not found at {}", fx.display());
            return;
        }
        if !cmf.exists() {
            eprintln!("SKIP sam3_segment_e2e: cmf not found (set SAM3_CMF): {}", cmf.display());
            return;
        }
        let tmp = std::env::temp_dir().join(format!("sam3_seg_e2e_{}.png", std::process::id()));
        media::save_png(&tmp, &synthetic_image()).expect("write synthetic png");

        // ── preprocess parity: our pixels vs the fixture's ────────
        let mine = sam3::preprocess(&synthetic_image());
        let want = f32s(&fx.join("pixels.f32"));
        assert_eq!(mine.len(), want.len());
        let mut max_d = 0f32;
        let mut sum_d = 0f64;
        let mut big = 0usize;
        for (&g, &w) in mine.iter().zip(&want) {
            let d = (g - w).abs();
            max_d = max_d.max(d);
            sum_d += d as f64;
            if d > 0.1 {
                big += 1;
            }
        }
        let mean_d = sum_d / want.len() as f64;
        eprintln!(
            "preprocess vs pixels.f32: max|Δ|={max_d:.4} mean|Δ|={mean_d:.5} pixels>0.1: {big}/{}",
            want.len()
        );
        // Bilinear is `resample: 2` in cfg.processor; our kernel and
        // torchvision's may disagree inside sub-pixel edge bands only.
        assert!(mean_d < 0.005, "preprocess mean|Δ| {mean_d:.5} too far from the fixture");
        assert!(
            (big as f64) < 0.02 * want.len() as f64,
            "preprocess: {big} pixels differ by >0.1"
        );

        // ── tokenizer: Sam3Processor("red square") ids ────────────
        let cm = CmfModel::open(cmf.to_str().unwrap()).expect("open cmf");
        let tok = Tokenizer::from_bytes(cm.vocab.as_deref().expect("embedded vocab"))
            .expect("load embedded tokenizer");
        let (ids, attn) = tokenize_prompt(&tok, "red square");
        let expect: Vec<u32> = std::iter::once(49_406)
            .chain([736, 3999])
            .chain(std::iter::repeat(49_407).take(29))
            .collect();
        assert_eq!(ids, expect, "tokenization drifted from Sam3Processor");
        assert_eq!(attn, [1u8, 1, 1, 1].iter().copied().chain(std::iter::repeat(0).take(28)).collect::<Vec<_>>());

        // ── the run itself ────────────────────────────────────────
        let out_png = tmp.with_file_name("sam3_seg_e2e_out.png");
        let out_json = tmp.with_file_name("sam3_seg_e2e_out.json");
        let args = SegArgs {
            model: cmf.display().to_string(),
            image: tmp.display().to_string(),
            prompt: "red square".into(),
            out: out_png.display().to_string().into(),
            json: out_json.display().to_string().into(),
            topk: 10,
            thr: 0.5,
            mask_thr: 0.5,
        };
        let res = run(&args).expect("segment run");
        eprintln!(
            "segment: {} object(s), forward {:.0} ms, total {:.0} ms -> {}",
            res.detections.len(),
            res.forward_ms,
            res.total_ms,
            res.overlay_path.display()
        );
        assert!(res.forward_ms > 0.0 && res.forward_ms < 600_000.0, "forward timing recorded");
        assert!(out_png.exists(), "overlay PNG written");
        assert!(out_json.exists(), "JSON written");
        assert!(
            res.json["timing_ms"]["forward"].as_f64().unwrap() > 0.0,
            "timing_ms.forward in the JSON"
        );
        assert!(!res.detections.is_empty(), "the red square must be detected");

        // ── box: [64,64,192,192] px ±10 % of 384 ──────────────────
        let (iw, ih) = (res.image_size.0 as f32, res.image_size.1 as f32);
        let top = &res.detections[0];
        let px: Vec<f32> = top
            .box_xyxy
            .iter()
            .zip([iw, ih, iw, ih])
            .map(|(&v, s)| v * s)
            .collect();
        eprintln!(
            "top box px [{:.0},{:.0},{:.0},{:.1}] score {:.4} (query {})",
            px[0], px[1], px[2], px[3], top.score, top.query
        );
        let gt = [64f32, 64f32, 192f32, 192f32];
        let tol = 0.10 * 384.0;
        for (i, (&got, &want)) in px.iter().zip(&gt).enumerate() {
            assert!(
                (got - want).abs() <= tol,
                "box corner {i}: got {got:.1}, want {want:.1} ±{tol:.1}"
            );
        }
        assert!(top.score >= 0.5, "score must be high, got {:.4}", top.score);

        // ── mask vs the fixture's pred_masks (IoU ≥ 0.5) ──────────
        let fm = f32s(&fx.join("pred_masks.f32"));
        let hw = MASK_SIDE * MASK_SIDE;
        assert_eq!(fm.len(), 200 * hw, "fixture pred_masks shape");
        let mut best = 0f32;
        let mut best_q = usize::MAX;
        let mut same_q = f32::NAN;
        for q in 0..fm.len() / hw {
            let bin: Vec<u8> = fm[q * hw..(q + 1) * hw]
                .iter()
                .map(|&v| u8::from(v > 0.0))
                .collect();
            let iou = mask_iou(&top.mask, &bin);
            if q == top.query {
                same_q = iou;
            }
            if iou > best {
                best = iou;
                best_q = q;
            }
        }
        eprintln!("mask IoU vs fixture: best {best:.4} (query {best_q}), same query {same_q:.4}");
        assert!(best >= 0.5, "mask IoU vs fixture {best:.4} < 0.5");
        assert!(
            best_q != usize::MAX && best >= 0.5,
            "no fixture mask matches the CLI mask"
        );

        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(&out_png);
        let _ = std::fs::remove_file(&out_json);
    }
}