//! MiMo-V2.6 vision parity against `tools/mimo_vis_ref.py` fixtures.
//!
//! Every test is skipped unless its inputs exist:
//!   CMF_MIMO_VIS_FX   fixture dir written by `mimo_vis_ref.py fixtures` (+ `vit`)
//!   CMF_MIMO_VIS_TOK  dir with tokenizer.json + chat_template.jinja (prompt ids)
//!   CMF_MIMO_VIS_CMF  exact tower (BF16/F16 `visual.*` + mm.config_json)
//!   CMF_MIMO_VIS_Q4   the same tower quantized (q4tp for G4.1; any codec compares)
//!   CMF_MIMO_VIS_EXTRA_IMAGES  optional comma list of extra images for G4.1
//! The toy tower is `$CMF_MIMO_VIS_FX/toy.cmf` (packed by the
//! `mimo_vis_devpack` example from the oracle's toy.safetensors).
//!
//! Gates (numbers are printed; run with --nocapture):
//!   G2.1 smart_resize == oracle; G2.2 pixel rows max|Δ| ≤ 1e-4;
//!   G2.3/G7.2 prompt ids byte-equal; G7.1 frame count/indices/timestamps;
//!   G3.1 toy rel ≤ 1e-5; G3.2 real rel ≤ 1e-3 and row cos ≥ 0.9999;
//!   G7.3 8-frame video rel ≤ 1e-3; G4.1 q4tp vs exact mean cos ≥ 0.995, min ≥ 0.98.

use cortiq_core::CmfModel;
use cortiq_engine::media;
use cortiq_engine::mimo_vision::{
    self as mv, MimoProcessorConfig, MimoVit, VisualInput, VisualKind,
};
use cortiq_engine::tokenizer::Tokenizer;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn fx() -> Option<(PathBuf, Value)> {
    let dir = PathBuf::from(std::env::var("CMF_MIMO_VIS_FX").ok()?);
    let man: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).ok()?).ok()?;
    Some((dir, man))
}

fn read_f32(p: &Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn max_abs(a: &[f32]) -> f32 {
    a.iter().map(|x| x.abs()).fold(0.0, f32::max)
}

/// (mean, min) cosine over rows of width `d`.
fn row_cos(a: &[f32], b: &[f32], d: usize) -> (f64, f64) {
    let mut sum = 0.0;
    let mut mn = f64::INFINITY;
    let rows = a.len() / d;
    for r in 0..rows {
        let (x, y) = (&a[r * d..(r + 1) * d], &b[r * d..(r + 1) * d]);
        let dot: f64 = x.iter().zip(y).map(|(p, q)| *p as f64 * *q as f64).sum();
        let nx: f64 = x.iter().map(|p| (*p as f64).powi(2)).sum::<f64>().sqrt();
        let ny: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum::<f64>().sqrt();
        let c = dot / (nx * ny).max(1e-30);
        sum += c;
        mn = mn.min(c);
    }
    (sum / rows as f64, mn)
}

/// Diagnostics: the five rows with the lowest cosine, with their token
/// (row, col) on the merged grid and their norm against the median norm.
fn print_worst_rows(a: &[f32], b: &[f32], d: usize, merged_w: usize) {
    let rows = a.len() / d;
    let norm = |x: &[f32]| x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let mut norms: Vec<f64> = (0..rows).map(|r| norm(&b[r * d..(r + 1) * d])).collect();
    let mut cs: Vec<(f64, usize)> = (0..rows)
        .map(|r| {
            (
                row_cos(&a[r * d..(r + 1) * d], &b[r * d..(r + 1) * d], d).0,
                r,
            )
        })
        .collect();
    cs.sort_by(|x, y| x.0.total_cmp(&y.0));
    let n_exact = norms.clone();
    norms.sort_by(f64::total_cmp);
    let median = norms[rows / 2];
    for &(c, r) in cs.iter().take(5) {
        println!(
            "    worst row {r} (merged y {} x {}): cos {c:.4}, |exact| {:.1} = {:.2}× median",
            r / merged_w.max(1),
            r % merged_w.max(1),
            n_exact[r],
            n_exact[r] / median
        );
    }
}

fn backend_label() -> String {
    format!(
        "CMF_GPU={} CMF_COOP={}",
        std::env::var("CMF_GPU").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("CMF_COOP").unwrap_or_else(|_| "<unset>".into())
    )
}

#[test]
fn g2_1_smart_resize_matches_oracle() {
    let Some((_, man)) = fx() else {
        eprintln!("skipped: set CMF_MIMO_VIS_FX");
        return;
    };
    let cases = man["resize_cases"].as_array().unwrap();
    let mut bad = 0;
    let mut zero_side = 0;
    for c in cases {
        let (h, w) = (
            c["h"].as_u64().unwrap() as usize,
            c["w"].as_u64().unwrap() as usize,
        );
        let (mn, mx) = (
            c["min"].as_u64().unwrap() as usize,
            c["max"].as_u64().unwrap() as usize,
        );
        let got = mv::smart_resize(h, w, 32, mn, mx);
        // A zero side from the reference smart_resize is an error one step
        // later (F.interpolate rejects an empty size); the engine errors at once.
        let collapsed = c.get("out").is_some_and(|o| usizes(o).contains(&0));
        zero_side += collapsed as usize;
        let ok = match (c.get("error").is_some() || collapsed, &got) {
            (true, Err(_)) => true,
            (false, Ok((gh, gw))) => usizes(&c["out"]) == vec![*gh, *gw],
            _ => false,
        };
        if !ok {
            bad += 1;
            eprintln!("G2.1 MISMATCH {h}x{w} [{mn},{mx}]: got {got:?}, oracle {c}");
        }
    }
    println!(
        "G2.1 smart_resize: {}/{} cases equal ({zero_side} reference zero-side outputs counted as errors)",
        cases.len() - bad,
        cases.len()
    );
    assert_eq!(bad, 0);
}

#[test]
fn g2_2_pixel_rows_match_oracle() {
    let Some((dir, man)) = fx() else {
        eprintln!("skipped: set CMF_MIMO_VIS_FX");
        return;
    };
    let cfg = MimoProcessorConfig::default();
    let mut worst = 0f32;
    for img in man["images"].as_array().unwrap() {
        let name = img["name"].as_str().unwrap();
        let frame = media::read_rgb(&dir.join(img["png"].as_str().unwrap())).unwrap();
        let got = mv::prepare_image(&frame, &cfg, None).unwrap();
        let grid = usizes(&img["grid"]);
        assert_eq!(
            vec![got.grid_t, got.grid_h, got.grid_w],
            grid,
            "{name} grid"
        );
        let want = read_f32(&dir.join(img["rows"].as_str().unwrap()));
        let d = max_abs_diff(&got.rows, &want);
        println!(
            "G2.2 {name}: grid {grid:?}, rows {}, max|Δ| = {d:.3e}",
            got.patches()
        );
        worst = worst.max(d);
    }
    assert!(worst <= 1e-4, "G2.2 max|Δ| {worst} > 1e-4");
}

fn tokenizer() -> Option<Tokenizer> {
    let dir = PathBuf::from(std::env::var("CMF_MIMO_VIS_TOK").ok()?);
    let mut tok = Tokenizer::from_file(dir.join("tokenizer.json")).ok()?;
    tok.chat_template = Some(std::fs::read_to_string(dir.join("chat_template.jinja")).ok()?);
    Some(tok)
}

fn stub(kind: VisualKind, grid: &[usize], timestamps: Vec<f32>) -> VisualInput {
    VisualInput {
        kind,
        rows: Vec::new(),
        grid_t: grid[0],
        grid_h: grid[1],
        grid_w: grid[2],
        patch_dim: 1536,
        merge_size: 2,
        timestamps,
        frame_indices: Vec::new(),
        resized: (grid[1] * 16, grid[2] * 16),
    }
}

#[test]
fn g2_3_and_g7_2_prompt_ids_match_oracle() {
    let (Some((_, man)), Some(tok)) = (fx(), tokenizer()) else {
        eprintln!("skipped: set CMF_MIMO_VIS_FX and CMF_MIMO_VIS_TOK");
        return;
    };
    let grids: std::collections::HashMap<String, Vec<usize>> = man["images"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| (i["name"].as_str().unwrap().to_string(), usizes(&i["grid"])))
        .collect();
    let mut bad = 0;
    for c in man["prompt_cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let msgs: Vec<Value> = c["messages"].as_array().unwrap().clone();
        let raw = tok
            .try_apply_chat_template_json(&msgs, None, None)
            .expect("strict render");
        let want_raw: Vec<u32> = c["raw_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let images: Vec<VisualInput> = c["images"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| stub(VisualKind::Image, &grids[n.as_str().unwrap()], Vec::new()))
            .collect();
        let videos: Vec<VisualInput> = c["videos"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let ts = v["timestamps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect();
                stub(VisualKind::Video, &usizes(&v["grid"]), ts)
            })
            .collect();
        let irefs: Vec<&VisualInput> = images.iter().collect();
        let vrefs: Vec<&VisualInput> = videos.iter().collect();
        let got = mv::expand_prompt_ids(&raw, &irefs, &vrefs, &[], &tok).expect("expand");
        let want: Vec<u32> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let same = got == want;
        println!(
            "G2.3/G7.2 {name}: {} ids, oracle {} → {} (rendered+tokenized once == HF: {})",
            got.len(),
            want.len(),
            if same { "byte-equal" } else { "MISMATCH" },
            raw == want_raw
        );
        if !same {
            bad += 1;
            let at = got.iter().zip(&want).position(|(a, b)| a != b);
            eprintln!(
                "first diff at {at:?}; text:\n{}",
                c["text"].as_str().unwrap()
            );
        }
    }
    assert_eq!(bad, 0);
}

#[test]
fn g7_1_frame_sampling_matches_oracle() {
    let Some((_, man)) = fx() else {
        eprintln!("skipped: set CMF_MIMO_VIS_FX");
        return;
    };
    let cfg = MimoProcessorConfig::default();
    let mut bad = 0;
    let cases = man["frame_cases"].as_array().unwrap();
    for c in cases {
        let total = c["total"].as_u64().unwrap() as usize;
        let fps = c["fps"].as_f64().unwrap();
        let got = mv::sample_frames(total, fps, &cfg);
        let ok = if c.get("error").is_some() {
            got.is_err()
        } else {
            let (idx, ts) = got.as_ref().unwrap();
            let want_idx = usizes(&c["indices"]);
            let mut ts_pad = ts.clone();
            if ts_pad.len() % 2 == 1 {
                ts_pad.push(*ts.last().unwrap());
            }
            let want_ts: Vec<f32> = c["timestamps"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let labels: Vec<String> = ts_pad
                .iter()
                .step_by(2)
                .map(|&t| mv::format_timestamp(t))
                .collect();
            let want_labels: Vec<String> = c["labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap().to_string())
                .collect();
            let mp = mv::video_max_pixels(idx.len(), &cfg);
            let ok = *idx == want_idx
                && ts_pad == want_ts
                && labels == want_labels
                && idx.len() == c["n"].as_u64().unwrap() as usize
                && mp == c["max_pixels"].as_u64().unwrap() as usize;
            println!(
                "G7.1 N={total} fps={fps:.4}: n={} first/last idx {:?}/{:?} labels {}..{} max_px {mp} {}",
                idx.len(),
                idx.first(),
                idx.last(),
                labels.first().map(String::as_str).unwrap_or(""),
                labels.last().map(String::as_str).unwrap_or(""),
                if ok { "equal" } else { "MISMATCH" }
            );
            ok
        };
        if !ok {
            bad += 1;
            eprintln!("G7.1 MISMATCH N={total} fps={fps}: got {got:?}");
        }
    }
    println!("G7.1: {}/{} cases equal", cases.len() - bad, cases.len());
    assert_eq!(bad, 0);
}

fn open(p: &Path) -> Arc<CmfModel> {
    Arc::new(CmfModel::open(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
}

#[test]
fn g3_1_toy_tower_matches_hf() {
    let Some((dir, man)) = fx() else {
        eprintln!("skipped: set CMF_MIMO_VIS_FX");
        return;
    };
    let toy = &man["toy"];
    let path = dir.join("toy.cmf");
    if !path.exists() {
        eprintln!(
            "skipped: {} missing (pack toy.safetensors with mimo_vis_devpack)",
            path.display()
        );
        return;
    }
    let model = open(&path);
    let vit = MimoVit::from_model(&model).expect("load toy");
    let grid = usizes(&toy["grid"]);
    let rows = read_f32(&dir.join(toy["rows"].as_str().unwrap()));
    let input = VisualInput {
        rows,
        ..stub(VisualKind::Video, &grid, Vec::new())
    };
    let input = VisualInput {
        patch_dim: input.rows.len() / (grid[0] * grid[1] * grid[2]),
        ..input
    };
    let got = vit.forward(&input).expect("forward");
    let want = read_f32(&dir.join(toy["out"].as_str().unwrap()));
    let rel = max_abs_diff(&got, &want) / max_abs(&want);
    let (cm, cmin) = row_cos(&got, &want, *usizes(&toy["out_shape"]).last().unwrap());
    println!(
        "G3.1 toy [{}]: rel {rel:.3e}, row cos mean {cm:.8} min {cmin:.8}",
        backend_label()
    );
    assert!(rel <= 1e-5, "G3.1 rel {rel} > 1e-5");
}

fn exact_tower() -> Option<Arc<CmfModel>> {
    let p = PathBuf::from(std::env::var("CMF_MIMO_VIS_CMF").ok()?);
    Some(open(&p))
}

#[test]
fn g3_2_and_g7_3_real_tower_matches_hf() {
    let (Some((dir, man)), Some(model)) = (fx(), exact_tower()) else {
        eprintln!("skipped: set CMF_MIMO_VIS_FX and CMF_MIMO_VIS_CMF");
        return;
    };
    let Some(vit_res) = man.get("vit") else {
        eprintln!("skipped: run `mimo_vis_ref.py vit` first");
        return;
    };
    let t0 = std::time::Instant::now();
    let vit = MimoVit::from_model(&model).expect("load tower");
    println!(
        "tower loaded in {:.1} s [{}]",
        t0.elapsed().as_secs_f64(),
        backend_label()
    );
    let cfg = MimoProcessorConfig::default();
    let mut fails = Vec::new();
    for img in man["images"].as_array().unwrap() {
        let name = img["name"].as_str().unwrap();
        let Some(r) = vit_res.get(name) else { continue };
        let frame = media::read_rgb(&dir.join(img["png"].as_str().unwrap())).unwrap();
        let input = mv::prepare_image(&frame, &cfg, None).unwrap();
        let t = std::time::Instant::now();
        let got = vit.forward(&input).unwrap();
        let dt = t.elapsed().as_secs_f64();
        let want = read_f32(&dir.join(r["out"].as_str().unwrap()));
        let rel = max_abs_diff(&got, &want) / max_abs(&want);
        let (cm, cmin) = row_cos(&got, &want, 4096);
        println!(
            "G3.2 {name}: {} patches → {} tokens in {dt:.2} s, rel {rel:.3e}, row cos mean {cm:.7} min {cmin:.7}; \
             G3.3 LN-vs-RMS merger cos mean {:.5} min {:.5}",
            input.patches(),
            input.tokens(),
            r["g3_3_ln_vs_rms_row_cos_mean"].as_f64().unwrap(),
            r["g3_3_ln_vs_rms_row_cos_min"].as_f64().unwrap()
        );
        if rel > 1e-3 || cmin < 0.9999 {
            fails.push(name.to_string());
        }
    }
    // G7.3: the 8-frame clip through the frame-directory source.
    if let Some(r) = vit_res.get("video8") {
        let v = &man["video"];
        let src = mv::VideoSource::frame_dir(
            &dir.join(v["dir"].as_str().unwrap()),
            v["fps"].as_f64().unwrap(),
        )
        .unwrap();
        let input = mv::prepare_video(&src, &cfg, None).unwrap();
        assert_eq!(
            vec![input.grid_t, input.grid_h, input.grid_w],
            usizes(&v["grid"]),
            "video grid"
        );
        assert_eq!(input.frame_indices, usizes(&v["indices"]), "video indices");
        assert_eq!(
            input.timestamp_labels(),
            v["labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        );
        let rows_want = read_f32(&dir.join(v["rows"].as_str().unwrap()));
        let drows = max_abs_diff(&input.rows, &rows_want);
        let t = std::time::Instant::now();
        let got = vit.forward(&input).unwrap();
        let dt = t.elapsed().as_secs_f64();
        let want = read_f32(&dir.join(r["out"].as_str().unwrap()));
        let rel = max_abs_diff(&got, &want) / max_abs(&want);
        let (cm, cmin) = row_cos(&got, &want, 4096);
        println!(
            "G7.3 video8: grid {:?} rows max|Δ| {drows:.3e}; {} tokens in {dt:.2} s, rel {rel:.3e}, row cos mean {cm:.7} min {cmin:.7}",
            usizes(&v["grid"]),
            input.tokens()
        );
        if rel > 1e-3 || drows > 1e-4 {
            fails.push("video8".into());
        }
    }
    println!(
        "full-attention chunks on the device so far: {}",
        mv::gpu_attention_dispatches()
    );
    assert!(fails.is_empty(), "failed: {fails:?}");
}

#[test]
fn g4_1_q4tp_tower_tracks_exact() {
    let (Some((dir, man)), Some(exact)) = (fx(), exact_tower()) else {
        eprintln!("skipped: set CMF_MIMO_VIS_FX, CMF_MIMO_VIS_CMF and CMF_MIMO_VIS_Q4");
        return;
    };
    let Ok(q4) = std::env::var("CMF_MIMO_VIS_Q4") else {
        eprintln!("skipped: set CMF_MIMO_VIS_Q4");
        return;
    };
    let q4 = open(Path::new(&q4));
    let vx = MimoVit::from_model(&exact).unwrap();
    let vq = MimoVit::from_model(&q4).unwrap();
    let cfg = MimoProcessorConfig::default();
    let mut paths: Vec<PathBuf> = man["images"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|i| i["name"] != "tiny20x300")
        .map(|i| dir.join(i["png"].as_str().unwrap()))
        .collect();
    if let Ok(extra) = std::env::var("CMF_MIMO_VIS_EXTRA_IMAGES") {
        paths.extend(
            extra
                .split(',')
                .filter(|s| !s.is_empty())
                .map(PathBuf::from),
        );
    }
    let (mut means, mut mins) = (Vec::new(), Vec::new());
    let mut all_rows: Vec<f64> = Vec::new();
    for p in &paths {
        let frame = media::read_rgb(p).unwrap();
        let input = mv::prepare_image(&frame, &cfg, None).unwrap();
        let a = vx.forward(&input).unwrap();
        let t = std::time::Instant::now();
        let b = vq.forward(&input).unwrap();
        let dt = t.elapsed().as_secs_f64();
        let (cm, cmin) = row_cos(&b, &a, 4096);
        if std::env::var("CMF_MIMO_VIS_WORST").is_ok() {
            print_worst_rows(&b, &a, 4096, input.grid_w / 2);
        }
        println!(
            "G4.1 {} ({}x{} → {} tokens): quantized vs exact row cos mean {cm:.5} min {cmin:.5} (quantized forward {dt:.2} s) [{}]",
            p.file_name().unwrap().to_string_lossy(),
            frame.width,
            frame.height,
            input.tokens(),
            backend_label()
        );
        // The reference's own noise floor, when the oracle ran with --bf16:
        // HF in bf16 (the serving numerics) against HF in fp32.
        let stem = p.file_stem().unwrap().to_string_lossy().to_string();
        if let Some(r) = man.get("vit").and_then(|v| v.get(&stem)) {
            if let (Some(m), Some(n)) = (
                r.get("bf16_vs_fp32_row_cos_mean").and_then(Value::as_f64),
                r.get("bf16_vs_fp32_row_cos_min").and_then(Value::as_f64),
            ) {
                println!(
                    "    reference noise floor (HF bf16 vs HF fp32): row cos mean {m:.5} min {n:.5}"
                );
            }
        }
        means.push(cm);
        mins.push(cmin);
        all_rows.extend((0..a.len() / 4096).map(|r| {
            row_cos(
                &b[r * 4096..(r + 1) * 4096],
                &a[r * 4096..(r + 1) * 4096],
                4096,
            )
            .0
        }));
    }
    let mean = means.iter().sum::<f64>() / means.len() as f64;
    let min = mins.iter().copied().fold(f64::INFINITY, f64::min);
    all_rows.sort_by(f64::total_cmp);
    let p1 = all_rows[all_rows.len() / 100];
    let below = all_rows.iter().filter(|&&c| c < 0.98).count();
    println!(
        "G4.1 over {} images: mean of row-cos means {mean:.5}, min row cos {min:.5}; \
         1st-percentile row cos {p1:.5}, rows below 0.98: {below}/{}",
        paths.len(),
        all_rows.len()
    );
    assert!(
        mean >= 0.995 && min >= 0.98,
        "G4.1 failed: mean {mean} min {min}"
    );
}
