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
    /// Input: an image (anything `media::decode_rgb` reads), a directory of
    /// PNG frames, or a video file (mp4/mkv/webm/… via ffmpeg).
    pub input: String,
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
    let frame = media::read_rgb(Path::new(&args.input))
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("read image {}", args.input))?;
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
    let overlay_path = args.out.as_ref().map(PathBuf::from).unwrap_or_else(|| default_out(&args.input));
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
        "input": args.input,
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

// ───────────────────── video mode ─────────────────────
//
// `cortiq segment` widens to video input: a directory of PNG frames (no
// decoder needed) or a video file (frames pulled with ffmpeg). Every frame
// goes through the detector AND the memory-bank tracker, so objects keep one
// stable `track_id` across the clip.

use cortiq_engine::sam3::{self as sam3mod};
use cortiq_engine::sam3vid::{self, FrameOut as VidFrame, Sam3Video, Session};

/// Video file extensions the CLI routes through ffmpeg.
const VIDEO_EXTS: [&str; 6] = ["mp4", "mkv", "webm", "mov", "avi", "m4v"];

/// `true` for a directory of PNG frames or a video file.
pub fn is_video_input(input: &str) -> bool {
    let p = Path::new(input);
    if p.is_dir() {
        return true;
    }
    match p.extension().and_then(|e| e.to_str()) {
        Some(e) => VIDEO_EXTS.iter().any(|x| e.eq_ignore_ascii_case(x)),
        None => false,
    }
}

/// PNG frames of a directory, sorted by name.
fn list_frames(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e.eq_ignore_ascii_case("png")))
        .collect();
    v.sort();
    anyhow::ensure!(!v.is_empty(), "no PNG frames in {}", dir.display());
    Ok(v)
}

/// ffmpeg: `$FFMPEG`, then `$PATH`, then the bundled tool path.
fn find_ffmpeg() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("FFMPEG") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    if let Ok(path) = std::env::var("PATH") {
        for d in path.split(':') {
            let c = Path::new(d).join("ffmpeg");
            if c.exists() {
                return Some(c);
            }
        }
    }
    const KNOWN: &str = "/home/hermes/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg";
    if Path::new(KNOWN).exists() {
        Some(PathBuf::from(KNOWN))
    } else {
        None
    }
}

/// Pull every frame of a video into `dst` as PNGs.
fn extract_frames(video: &Path, dst: &Path) -> Result<Vec<PathBuf>> {
    let ff = find_ffmpeg().ok_or_else(|| anyhow::anyhow!("ffmpeg not found — set $FFMPEG"))?;
    std::fs::create_dir_all(dst).with_context(|| format!("mkdir {}", dst.display()))?;
    let status = std::process::Command::new(&ff)
        .arg("-y")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(video)
        .arg(dst.join("frame_%05d.png"))
        .status()
        .with_context(|| format!("spawn {}", ff.display()))?;
    anyhow::ensure!(status.success(), "ffmpeg exited with {status}");
    list_frames(dst)
}

/// One video run: overlays + JSON tracks.
pub struct VidOutput {
    pub frames: usize,
    /// Per-track JSON records (see [`build_tracks`]).
    pub tracks: Vec<serde_json::Value>,
    pub overlay_dir: PathBuf,
    pub mean_frame_ms: f64,
    pub json: serde_json::Value,
    pub per_frame: Vec<VidFrame>,
}

/// Default overlay folder: `frames/` → `frames_seg/`, `clip.mp4` → `clip_seg/`.
fn default_overlay_dir(input: &str) -> PathBuf {
    let p = Path::new(input);
    let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("video");
    let base = if p.is_dir() { name } else { p.file_stem().and_then(|s| s.to_str()).unwrap_or(name) };
    p.parent().map(|d| d.join(format!("{base}_seg"))).unwrap_or_else(|| PathBuf::from(format!("{base}_seg")))
}

/// Group per-frame observations into tracks: per-frame boxes/scores plus the
/// temporal id-stability stats (one id for the whole track, mask IoU between
/// consecutive frames).
fn build_tracks(per_frame: &[VidFrame], prompt: &str) -> Vec<serde_json::Value> {
    let mut by_id: std::collections::BTreeMap<i64, Vec<(usize, &sam3vid::TrackObs)>> = Default::default();
    for f in per_frame {
        for t in &f.tracks {
            by_id.entry(t.id).or_default().push((f.frame, t));
        }
    }
    by_id
        .into_iter()
        .map(|(id, obs)| {
            let mut ids_seen: std::collections::BTreeSet<i64> = Default::default();
            let mut iou_sum = 0f32;
            let mut iou_n = 0usize;
            let mut switches = 0usize;
            let mut prev: Option<&sam3vid::TrackObs> = None;
            for (_, t) in &obs {
                ids_seen.insert(t.id);
                if let Some(p) = prev {
                    if p.id != t.id {
                        switches += 1;
                    }
                    iou_sum += sam3vid::mask_iou(&p.mask, &t.mask);
                    iou_n += 1;
                }
                prev = Some(t);
            }
            let frames: Vec<serde_json::Value> = obs
                .iter()
                .map(|(fi, t)| {
                    let [x0, y0, x1, y1] = t.box_px;
                    json!({
                        "frame": fi,
                        "track_id": t.id,
                        "box": [x0, y0, (x1 - x0 + 1.0).max(0.0), (y1 - y0 + 1.0).max(0.0)],
                        "box_xyxy": [x0, y0, x1, y1],
                        "score": t.score,
                        "source": if t.from_detection { "detection" } else { "tracker" },
                    })
                })
                .collect();
            let last = obs.last().unwrap().1;
            json!({
                "track_id": id,
                "label": prompt,
                "score": last.score,
                "first_frame": obs.first().unwrap().0,
                "last_frame": obs.last().unwrap().0,
                "frames_present": obs.len(),
                "ids": ids_seen.iter().collect::<Vec<_>>(),
                "id_stable": switches == 0 && ids_seen.len() == 1,
                "id_switches": switches,
                "mean_mask_iou_prev": if iou_n > 0 { (iou_sum / iou_n as f32 * 1e4).round() / 1e4 } else { 1.0 },
                "frames": frames,
            })
        })
        .collect()
}

/// Run the whole video: detect + track every frame, write overlays and JSON.
pub fn run_video(args: &SegArgs) -> Result<VidOutput> {
    let t_all = Instant::now();
    let input = Path::new(&args.input);
    anyhow::ensure!(input.exists(), "input {} does not exist", args.input);

    // ── frames: PNG directory or ffmpeg extraction ────────────────
    let tmp_dir;
    let frames: Vec<PathBuf> = if input.is_dir() {
        list_frames(input)?
    } else {
        tmp_dir = std::env::temp_dir().join(format!("cortiq_seg_{}", std::process::id()));
        extract_frames(input, &tmp_dir)?
    };
    let n = frames.len();

    // ── overlays + output paths ───────────────────────────────────
    let overlay_dir = args.out.clone().map(PathBuf::from).unwrap_or_else(|| default_overlay_dir(&args.input));
    std::fs::create_dir_all(&overlay_dir)
        .with_context(|| format!("mkdir {}", overlay_dir.display()))?;

    // ── model, prompt, session ────────────────────────────────────
    let cm = CmfModel::open(&args.model).with_context(|| format!("open {}", args.model))?;
    let vocab = cm
        .vocab
        .clone()
        .context("the file carries no embedded tokenizer (VOCAB section)")?;
    let tok = Tokenizer::from_bytes(&vocab).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let (ids, attn) = tokenize_prompt(&tok, &args.prompt);

    let mut model = Sam3Video::open(&args.model)?;
    model.cfg.score_threshold_detection = args.thr;
    let mut session = Session::new();
    session.num_frames = n;

    // ── frame loop ────────────────────────────────────────────────
    let mut per_frame: Vec<VidFrame> = Vec::with_capacity(n);
    for (i, png) in frames.iter().enumerate() {
        let frame = media::read_rgb(png).map_err(anyhow::Error::msg).with_context(|| {
            format!("read frame {}", png.display())
        })?;
        let pixels = sam3mod::preprocess(&frame);
        let out = model.forward(&mut session, i, &pixels, &ids, &attn, (frame.width, frame.height))?;

        let stem = png.file_stem().and_then(|s| s.to_str()).unwrap_or("frame");
        let overlay_path = overlay_dir.join(format!("{stem}_seg.png"));
        let overlay = draw_video_overlay(&frame, &out.tracks);
        media::save_png(&overlay_path, &overlay).map_err(anyhow::Error::msg)?;
        eprintln!(
            "frame {i}: {} track(s), {} detection(s), forward {:.0} ms (det {:.0} + trk {:.0}) -> {}",
            out.tracks.len(),
            out.detections,
            out.total_ms,
            out.det_ms,
            out.trk_ms,
            overlay_path.display()
        );
        per_frame.push(out);
    }

    // ── JSON ──────────────────────────────────────────────────────
    let tracks = build_tracks(&per_frame, &args.prompt);
    let frames_json: Vec<serde_json::Value> = per_frame
        .iter()
        .map(|f| {
            json!({
                "frame": f.frame,
                "detections": f.detections,
                "track_ids": f.tracks.iter().map(|t| t.id).collect::<Vec<_>>(),
                "forward_ms": (f.total_ms * 100.0).round() / 100.0,
            })
        })
        .collect();
    let mean = per_frame.iter().map(|f| f.total_ms).sum::<f64>() / per_frame.len().max(1) as f64;
    let mean_det = per_frame.iter().map(|f| f.det_ms).sum::<f64>() / per_frame.len().max(1) as f64;
    let mean_trk = per_frame.iter().map(|f| f.trk_ms).sum::<f64>() / per_frame.len().max(1) as f64;
    let stable = tracks.iter().all(|t| t["id_stable"].as_bool().unwrap_or(false));
    let value = json!({
        "model": args.model,
        "input": args.input,
        "prompt": args.prompt,
        "mode": "video",
        "num_frames": n,
        "num_tracks": tracks.len(),
        "tracks": tracks,
        "frames": frames_json,
        "id_stability": {
            "tracks_with_single_id": tracks.iter().filter(|t| t["ids"].as_array().map_or(false, |a| a.len() == 1)).count(),
            "total_id_switches": tracks.iter().filter_map(|t| t["id_switches"].as_u64()).sum::<u64>(),
            "stable": stable,
        },
        "params": {
            "threshold": model.cfg.score_threshold_detection,
            "assoc_iou": model.cfg.assoc_iou_thresh,
            "trk_assoc_iou": model.cfg.trk_assoc_iou_thresh,
            "new_det_thresh": model.cfg.new_det_thresh,
        },
        "timing_ms": {
            "total": (t_all.elapsed().as_secs_f64() * 1e3 * 100.0).round() / 100.0,
            "mean_frame": (mean * 100.0).round() / 100.0,
            "mean_detector": (mean_det * 100.0).round() / 100.0,
            "mean_tracker": (mean_trk * 100.0).round() / 100.0,
        },
        "overlays": overlay_dir.display().to_string(),
    });

    if let Some(path) = &args.json {
        std::fs::write(path, serde_json::to_vec_pretty(&value)?)
            .with_context(|| format!("write {path}"))?;
    }

    Ok(VidOutput {
        frames: n,
        tracks,
        overlay_dir,
        mean_frame_ms: mean,
        json: value,
        per_frame,
    })
}

/// 3×5 digit glyphs (rows of 3 bits, MSB left) for the track-id label.
const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b010, 0b110, 0b010, 0b010, 0b111],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b001, 0b001, 0b001],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b111],
];

/// Mask (45 %) + 1-px box + the track id, drawn at 3× over the frame.
fn draw_video_overlay(frame: &RgbFrame, tracks: &[sam3vid::TrackObs]) -> RgbFrame {
    let (w, h) = (frame.width, frame.height);
    let mut data = frame.data.clone();
    let mut put = |x: usize, y: usize, color: [u8; 3], alpha: f32| {
        if x >= w || y >= h {
            return;
        }
        let o = (y * w + x) * 3;
        for c in 0..3 {
            let base = data[o + c] as f32;
            data[o + c] = (base * (1.0 - alpha) + color[c] as f32 * alpha).round() as u8;
        }
    };
    for t in tracks {
        let color = PALETTE[(t.id as usize).rem_euclid(PALETTE.len())];
        for y in 0..h {
            for x in 0..w {
                if t.mask[y * w + x] != 0 {
                    put(x, y, color, 0.45);
                }
            }
        }
        let [x0, y0, x1, y1] = t.box_px;
        let (x0, y0) = (x0 as usize, y0 as usize);
        let (x1, y1) = ((x1 as usize).min(w.saturating_sub(1)), (y1 as usize).min(h.saturating_sub(1)));
        for x in x0..=x1 {
            put(x, y0, color, 1.0);
            put(x, y1, color, 1.0);
        }
        for y in y0..=y1 {
            put(x0, y, color, 1.0);
            put(x1, y, color, 1.0);
        }
        // track id label, top-left of the box, on a dark plate
        let mut cx = x0 + 2;
        let cy = y0 + 2;
        for ch in t.id.max(0).to_string().chars() {
            let d = ch.to_digit(10).unwrap_or(0) as usize;
            for (r, bits) in DIGITS[d].iter().enumerate() {
                for c in 0..3 {
                    if bits >> (2 - c) & 1 == 0 {
                        continue;
                    }
                    for dy in 0..3 {
                        for dx in 0..3 {
                            put(cx + c * 3 + dx, cy + r * 3 + dy, color, 1.0);
                        }
                    }
                }
            }
            cx += 12;
        }
    }
    RgbFrame::new(w, h, data).expect("overlay buffer matches its frame")
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
            input: tmp.display().to_string(),
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

    // ── video mode ────────────────────────────────────────────────

    /// Clip shape: 8 frames of 384², a 128² red square moving 16 px right
    /// and 8 px down per frame, a static blue stripe — byte-identical to
    /// `tools/sam3_video_ref.py` (the reference fixtures come from it).
    const V_FRAMES: usize = 8;
    const V_SIZE: usize = 384;

    fn v_square(i: usize) -> [usize; 4] {
        let x0 = 64 + 16 * i;
        let y0 = 64 + 8 * i;
        [x0, y0, x0 + 127, y0 + 127] // inclusive, as masks_to_boxes reports
    }

    fn synthetic_frame(i: usize) -> RgbFrame {
        let s = V_SIZE;
        let mut data = vec![255u8; s * s * 3];
        for y in 300..340 {
            for x in 0..s {
                let o = (y * s + x) * 3;
                data[o..o + 3].copy_from_slice(&[30, 60, 200]);
            }
        }
        let [x0, y0, x1, y1] = v_square(i);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let o = (y * s + x) * 3;
                data[o..o + 3].copy_from_slice(&[200, 30, 30]);
            }
        }
        RgbFrame::new(s, s, data).unwrap()
    }

    fn gt_mask(i: usize) -> Vec<u8> {
        let s = V_SIZE;
        let mut m = vec![0u8; s * s];
        let [x0, y0, x1, y1] = v_square(i);
        for y in y0..=y1 {
            for x in x0..=x1 {
                m[y * s + x] = 1;
            }
        }
        m
    }

    /// Input routing: directories and video extensions go to the tracker,
    /// images do not. The ffmpeg leg pulls real frames out of an mp4.
    #[test]
    fn segment_video_input_paths() {
        let tmp = std::env::temp_dir().join(format!("sam3_vid_paths_{}", std::process::id()));
        let frames = tmp.join("frames");
        std::fs::create_dir_all(&frames).unwrap();
        for i in 0..3 {
            media::save_png(&frames.join(format!("f{i:02}.png")), &synthetic_frame(i)).unwrap();
        }
        assert!(is_video_input(frames.to_str().unwrap()), "PNG directory must route to video mode");
        assert!(is_video_input("clip.MP4"), "video extensions are case-insensitive");
        assert!(is_video_input("/tmp/missing_dir_xyz") == false || Path::new("/tmp/missing_dir_xyz").is_dir());
        assert!(!is_video_input("photo.png"), "a PNG file stays image mode");
        assert!(!is_video_input("photo.jpg"), "a JPEG stays image mode");
        assert_eq!(list_frames(&frames).unwrap().len(), 3, "all PNG frames are picked up");

        // ffmpeg: PNG dir → mp4 → PNG dir
        let Some(ff) = find_ffmpeg() else {
            eprintln!("SKIP ffmpeg leg: ffmpeg not on $PATH/$FFMPEG");
            let _ = std::fs::remove_dir_all(&tmp);
            return;
        };
        let mp4 = tmp.join("clip.mp4");
        let status = std::process::Command::new(&ff)
            .args(["-y", "-loglevel", "error", "-framerate", "1", "-i"])
            .arg(frames.join("f%02d.png"))
            .args(["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p"])
            .arg(&mp4)
            .status()
            .unwrap();
        assert!(status.success(), "ffmpeg encode failed");
        assert!(is_video_input(mp4.to_str().unwrap()), "mp4 routes to video mode");
        let back = tmp.join("back");
        let got = extract_frames(&mp4, &back).expect("ffmpeg extraction");
        assert_eq!(got.len(), 3, "lossless round trip keeps every frame");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// End-to-end video: the moving red square must be found in EVERY frame
    /// (box ±10 px of the ground-truth motion), keep ONE stable track id for
    /// the whole clip, and hold mask IoU ≥ 0.5 against the GT square each
    /// frame. One command:
    /// `cargo test --release -p cortiq-cli --bin cortiq sam3_video_e2e -- --nocapture`
    #[test]
    fn sam3_video_e2e() {
        let (_, cmf) = fixture_paths();
        if !cmf.exists() {
            eprintln!("SKIP sam3_video_e2e: cmf not found (set SAM3_CMF): {}", cmf.display());
            return;
        }
        let tmp = std::env::temp_dir().join(format!("sam3_video_e2e_{}", std::process::id()));
        // A shorter clip is useful for local tracker debugging; the default
        // remains the full eight-frame acceptance run.
        let n_frames = std::env::var("SAM3_VIDEO_FRAMES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(V_FRAMES)
            .clamp(1, V_FRAMES);
        let frames = tmp.join("frames");
        std::fs::create_dir_all(&frames).unwrap();
        for i in 0..n_frames {
            media::save_png(&frames.join(format!("frame_{i:02}.png")), &synthetic_frame(i))
                .expect("write frame");
        }
        let out_dir = tmp.join("seg");
        let json_path = tmp.join("tracks.json");
        let args = SegArgs {
            model: cmf.display().to_string(),
            input: frames.display().to_string(),
            prompt: "red square".into(),
            out: out_dir.display().to_string().into(),
            json: json_path.display().to_string().into(),
            topk: 10,
            thr: 0.5,
            mask_thr: 0.5,
        };
        let res = run_video(&args).expect("video run");
        assert_eq!(res.frames, n_frames, "every frame was processed");
        assert!(out_dir.exists() && json_path.exists(), "overlays + JSON written");

        let mut ids: Vec<i64> = Vec::new();
        for i in 0..n_frames {
            let f = &res.per_frame[i];
            assert!(!f.tracks.is_empty(), "frame {i}: the square was not tracked");
            let gt = gt_mask(i);
            let best = f
                .tracks
                .iter()
                .max_by(|a, b| {
                    sam3vid::mask_iou(&a.mask, &gt)
                        .partial_cmp(&sam3vid::mask_iou(&b.mask, &gt))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            let iou = sam3vid::mask_iou(&best.mask, &gt);
            assert!(iou >= 0.5, "frame {i}: mask IoU {iou:.3} < 0.5");
            let gtb = v_square(i).map(|v| v as f32);
            for (c, (got, want)) in best.box_px.iter().zip(gtb).enumerate() {
                assert!(
                    (got - want).abs() <= 10.0,
                    "frame {i}: box corner {c} got {got:.1}, want {want:.1} ±10"
                );
            }
            ids.push(best.id);
            eprintln!(
                "video frame {i}: id {} box [{:.0},{:.0},{:.0},{:.0}] score {:.4} IoU {:.3} ({:.0} ms)",
                best.id, best.box_px[0], best.box_px[1], best.box_px[2], best.box_px[3], best.score, iou, f.total_ms
            );
        }
        assert!(
            ids.iter().all(|x| *x == ids[0]),
            "track id must not flicker across frames: {ids:?}"
        );
        // ── parity with the reference (tools/sam3_video_ref.py) ──────
        // Same clip, transformers 5.20.2 + facebook/sam3 weights: our
        // tracker masks for the first three frames must line up with the
        // checked-in reference `pred_masks`.
        if let Some(fx) = fixture_paths().0.to_str() {
            let ref_dir = Path::new(fx).parent().map(|d| d.join("sam3_video"));
            if let Some(rd) = ref_dir.filter(|d| d.exists()) {
                for i in 0..3.min(n_frames) {
                    let mut cand = Vec::new();
                    for key in ["cond_frame_outputs", "non_cond_frame_outputs"] {
                        let p = rd.join(format!("f{i}/obj0/{key}/pred_masks.f32"));
                        if let Ok(bytes) = std::fs::read(p) {
                            cand = bytes
                                .chunks_exact(4)
                                .map(|c| u8::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]]) > 0.0))
                                .collect();
                            break;
                        }
                    }
                    if cand.len() != MASK_SIDE * MASK_SIDE {
                        continue;
                    }
                    let gt_ref = gt_mask(i);
                    // our track mask is at frame resolution; the reference is
                    // 288² — compare both after resizing to 288²
                    let mine = &res.per_frame[i]
                        .tracks
                        .iter()
                        .max_by(|a, b| {
                            sam3vid::mask_iou(&a.mask, &gt_ref)
                                .partial_cmp(&sam3vid::mask_iou(&b.mask, &gt_ref))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap()
                        .mask;
                    let mut down = vec![0u8; MASK_SIDE * MASK_SIDE];
                    for y in 0..MASK_SIDE {
                        for x in 0..MASK_SIDE {
                            let sy = (y * V_SIZE / MASK_SIDE).min(V_SIZE - 1);
                            let sx = (x * V_SIZE / MASK_SIDE).min(V_SIZE - 1);
                            down[y * MASK_SIDE + x] = mine[sy * V_SIZE + sx];
                        }
                    }
                    let iou = sam3vid::mask_iou(&down, &cand);
                    eprintln!("frame {i}: mask IoU vs reference transformers = {iou:.4}");
                    assert!(iou >= 0.8, "frame {i}: reference IoU {iou:.4} < 0.8");
                }
            } else {
                eprintln!("reference fixtures not found — skipping the parity cross-check");
            }
        }

        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&json_path).expect("read tracks.json")).unwrap();
        assert_eq!(json["mode"], "video");
        assert_eq!(json["num_frames"].as_u64().unwrap(), n_frames as u64);
        assert_eq!(json["id_stability"]["stable"], serde_json::Value::Bool(true), "id stability report");
        assert!(json["num_tracks"].as_u64().unwrap() >= 1, "at least one track");
        eprintln!(
            "video: {} track(s), stable={}, {:.0} ms/frame",
            json["num_tracks"], json["id_stability"]["stable"], res.mean_frame_ms
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}