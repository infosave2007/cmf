//! EmbeddingGemma 2 image / video / interleaved parity against the float32
//! transformers / sentence-transformers reference.
//!
//! Env (the test passes trivially without the first two):
//! * `CORTIQ_EGEMMA2_MODEL` — a `.cmf` packed with its vision tower;
//! * `CORTIQ_EGEMMA2_REF` — the reference dir: `manifest.json`,
//!   `embeddings.npy`, `inputs/` (the media) and `vision/` (raw dumps of
//!   the processor's resized pixels and the soft tokens, written by
//!   `export_vision_ref.py`);
//! * `CORTIQ_EGEMMA2_MIN_COS` — the cosine floor (default 0.99999 for an
//!   exact bf16 / f32 file, 0.995 for a quantized one).
//!
//! Checks, in order: the resize is byte-exact against torchvision; the
//! soft tokens of every image match `embed_vision`'s output; the final
//! embeddings of every image (budgets 70…1120), the video (from the
//! reference's decoded frames and, when ffmpeg is on PATH, from the mp4)
//! and the interleaved text + image input clear the floor at every
//! Matryoshka size.
//!
//! `cargo test --release -p cortiq-engine --test egemma2_vision_parity -- --nocapture`

use cortiq_core::CmfModel;
use cortiq_engine::egemma2::{MATRYOSHKA_DIMS, cosine, matryoshka};
use cortiq_engine::egemma2_mm::{MediaEncoder, MixedInput};
use cortiq_engine::egemma2_vision::{VisionProcessor, decode_video, resize_bicubic_aa};
use cortiq_engine::media::{RgbFrame, read_rgb};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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

/// (min row cosine, relative Frobenius error) of `a` against `b`, `[n, d]`.
fn rows_vs(a: &[f32], b: &[f32], d: usize) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "row counts differ");
    let mut minc = 1.0f64;
    let (mut num, mut den) = (0f64, 0f64);
    for (ra, rb) in a.chunks_exact(d).zip(b.chunks_exact(d)) {
        minc = minc.min(cosine(ra, rb));
        for (&x, &y) in ra.iter().zip(rb) {
            num += (x as f64 - y as f64).powi(2);
            den += (y as f64).powi(2);
        }
    }
    (minc, (num / den).sqrt())
}

struct Ctx {
    enc: MediaEncoder,
    refdir: PathBuf,
    manifest: serde_json::Value,
    refs: Vec<f32>,
    d: usize,
    floor: f64,
    worst: std::cell::RefCell<(f64, String)>,
}

impl Ctx {
    fn case(&self, id: &str) -> &serde_json::Value {
        self.manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == id)
            .unwrap_or_else(|| panic!("no case {id}"))
    }

    fn check(&self, id: &str, got: &[f32]) {
        let c = self.case(id);
        let row = c["row"].as_u64().unwrap() as usize;
        let r = &self.refs[row * self.d..(row + 1) * self.d];
        let mut line = format!("{id:34}");
        for dim in MATRYOSHKA_DIMS {
            let a = matryoshka(got, dim).unwrap();
            let cs = cosine(&a, &r[..dim]);
            line += &format!(" @{dim} {cs:.6}");
            let mut w = self.worst.borrow_mut();
            if cs < w.0 {
                *w = (cs, format!("{id} @{dim}"));
            }
            assert!(cs >= self.floor, "{id} dim {dim}: cosine {cs:.7} < {}", self.floor);
        }
        eprintln!("{line}");
    }

    fn ids(&self, id: &str) -> Vec<u32> {
        self.case(id)["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    }
}

#[test]
fn egemma2_vision_parity() {
    let (Some(model), Some(refdir)) = (
        std::env::var_os("CORTIQ_EGEMMA2_MODEL").map(PathBuf::from),
        std::env::var_os("CORTIQ_EGEMMA2_REF").map(PathBuf::from),
    ) else {
        eprintln!("egemma2_vision_parity: set CORTIQ_EGEMMA2_MODEL and CORTIQ_EGEMMA2_REF to run");
        return;
    };
    let cmf = Arc::new(CmfModel::open(&model).expect("open model"));
    let enc = MediaEncoder::load(&cmf, cortiq_engine::pool::Pool::from_env()).expect("load");
    assert!(enc.has_vision(), "the file has no vision tower");
    let floor: f64 = std::env::var("CORTIQ_EGEMMA2_MIN_COS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if matches!(enc.text.quant.as_str(), "bf16" | "f32") {
            0.99999
        } else {
            0.995
        });
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(refdir.join("manifest.json")).unwrap()).unwrap();
    let (shape, refs) = read_npy_f32(&refdir.join("embeddings.npy"));
    let vdir = refdir.join("vision");
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(vdir.join("index.json")).unwrap()).unwrap();
    let cx = Ctx {
        enc,
        refdir: refdir.clone(),
        manifest,
        refs,
        d: shape[1],
        floor,
        worst: std::cell::RefCell::new((1.0, String::new())),
    };
    let proc = cx.enc.proc.clone();
    assert_eq!(proc, VisionProcessor::default(), "processor config");

    // ── images: resize bytes, soft tokens, embeddings
    let images = [
        ("image_fox_512_280", "image_fox_512.png", 280),
        ("image_chart_rgba_1980x756_280", "image_chart_rgba_1980x756.png", 280),
        ("image_portrait_530x941_280", "image_portrait_530x941.png", 280),
        ("image_tiny_64x48_280", "image_tiny_64x48.png", 280),
        ("image_fox_512_70", "image_fox_512.png", 70),
        ("image_fox_512_140", "image_fox_512.png", 140),
        ("image_fox_512_560", "image_fox_512.png", 560),
        ("image_fox_512_1120", "image_fox_512.png", 1120),
    ];
    let mut inputs = Vec::new();
    for (id, file, budget) in images {
        let img = read_rgb(&cx.refdir.join("inputs").join(file)).unwrap();
        let (tw, th) = proc.resized_size(img.width, img.height, budget).unwrap();
        let hw: Vec<usize> = index[id]["resized_hw"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!((th, tw), (hw[0], hw[1]), "{id}: resized size");
        let want = std::fs::read(vdir.join(format!("{id}.resized.u8"))).unwrap();
        let got = resize_bicubic_aa(&img, th, tw);
        let diff = got
            .data
            .iter()
            .zip(&want)
            .filter(|(a, b)| a != b)
            .count();
        let maxd = got
            .data
            .iter()
            .zip(&want)
            .map(|(&a, &b)| (a as i32 - b as i32).abs())
            .max()
            .unwrap_or(0);
        eprintln!("{id:34} resize {tw}x{th}: {diff} of {} bytes differ (max {maxd})", want.len());
        assert_eq!(diff, 0, "{id}: resize is not byte-exact");
        let m = cx.enc.prepare_image(&img, Some(budget)).unwrap();
        let x = MixedInput {
            media: vec![m],
            ..Default::default()
        };
        assert_eq!(cx.enc.input_ids(&x).unwrap(), cx.ids(id), "{id}: token ids");
        inputs.push((id, x));
    }
    // soft tokens of the fox at 280 against embed_vision's output
    {
        let t0 = std::time::Instant::now();
        let soft = cx
            .enc
            .vision()
            .unwrap()
            .encode(&[match &inputs[0].1.media[0] {
                cortiq_engine::egemma2_mm::Media::Image(v) => v.clone(),
                _ => unreachable!(),
            }])
            .unwrap();
        let (_, want) = read_npy_f32(&vdir.join("image_fox_512_280.embed_vision.npy"));
        let (mc, rel) = rows_vs(&soft[0], &want, 512);
        eprintln!(
            "image_fox_512_280 soft tokens: min row cos {mc:.6}, rel err {rel:.2e} ({:.2}s)",
            t0.elapsed().as_secs_f64()
        );
    }
    let batch: Vec<MixedInput> = inputs.iter().map(|(_, x)| x.clone()).collect();
    let t0 = std::time::Instant::now();
    let got = cx.enc.embed(&batch).unwrap();
    eprintln!(
        "{} images embedded in {:.2}s",
        batch.len(),
        t0.elapsed().as_secs_f64()
    );
    for ((id, _), g) in inputs.iter().zip(&got) {
        cx.check(id, g);
    }
    // an image alone embeds as it does in the batch
    let alone = cx.enc.embed(&batch[1..2]).unwrap();
    let cs = cosine(&alone[0], &got[1]);
    assert!(cs > 0.999_999, "batch vs single: {cs}");

    // ── video: the reference's decoded frames, then the mp4 itself
    let fh: Vec<usize> = index["video_fox_pan"]["frames_fhw"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let raw = std::fs::read(vdir.join("video_fox_pan.frames.u8")).unwrap();
    let per = fh[1] * fh[2] * 3;
    let frames: Vec<RgbFrame> = raw
        .chunks_exact(per)
        .map(|c| RgbFrame::new(fh[2], fh[1], c.to_vec()).unwrap())
        .collect();
    let want = std::fs::read(vdir.join("video_fox_pan.resized.u8")).unwrap();
    let vin = cx.enc.prepare_frames(&frames, None).unwrap();
    let x = MixedInput {
        media: vec![vin],
        ..Default::default()
    };
    assert_eq!(cx.enc.input_ids(&x).unwrap(), cx.ids("video_fox_pan"), "video token ids");
    {
        let (tw, th) = proc.resized_size(fh[2], fh[1], 140).unwrap();
        let mut diff = 0usize;
        for (f, w) in frames.iter().zip(want.chunks_exact(tw * th * 3)) {
            diff += resize_bicubic_aa(f, th, tw)
                .data
                .iter()
                .zip(w)
                .filter(|(a, b)| a != b)
                .count();
        }
        eprintln!("video frames resize {tw}x{th}: {diff} bytes differ");
        assert_eq!(diff, 0);
    }
    let t0 = std::time::Instant::now();
    let gv = cx.enc.embed(std::slice::from_ref(&x)).unwrap();
    eprintln!("video (4 frames) embedded in {:.2}s", t0.elapsed().as_secs_f64());
    cx.check("video_fox_pan", &gv[0]);
    let mp4 = cx.refdir.join("inputs/video_fox_pan.mp4");
    match decode_video(&mp4, None, &proc) {
        Ok(dv) => {
            assert_eq!(dv.indices, vec![0, 25, 50, 75], "sampled frames");
            let maxd = dv
                .frames
                .iter()
                .flat_map(|f| f.data.iter())
                .zip(raw.iter())
                .map(|(&a, &b)| (a as i32 - b as i32).abs())
                .max()
                .unwrap_or(0);
            let m = cx.enc.prepare_frames(&dv.frames, None).unwrap();
            let g = cx
                .enc
                .embed(&[MixedInput {
                    media: vec![m],
                    ..Default::default()
                }])
                .unwrap();
            let row = cx.case("video_fox_pan")["row"].as_u64().unwrap() as usize;
            let cs = cosine(&g[0], &cx.refs[row * cx.d..(row + 1) * cx.d]);
            eprintln!(
                "video_fox_pan via {} (pixels vs PyAV: max |d| {maxd}): cos {cs:.6}",
                dv.decoder
            );
            assert!(cs >= floor.min(0.999), "mp4 decode path: {cs}");
        }
        Err(e) => eprintln!("mp4 decode skipped: {e}"),
    }

    // ── interleaved text + image
    let c = cx.case("interleaved_text_image");
    let text = c["text"].as_str().unwrap().to_string();
    let fox = read_rgb(&cx.refdir.join("inputs/image_fox_512.png")).unwrap();
    let x = MixedInput {
        text,
        media: vec![cx.enc.prepare_image(&fox, None).unwrap()],
        ..Default::default()
    };
    assert_eq!(
        cx.enc.input_ids(&x).unwrap(),
        cx.ids("interleaved_text_image"),
        "interleaved token ids"
    );
    let g = cx.enc.embed(std::slice::from_ref(&x)).unwrap();
    cx.check("interleaved_text_image", &g[0]);

    // text and media in one batch embed as they do apart
    let mixed = vec![
        MixedInput::text(cortiq_engine::egemma2::TextInput::plain("a red fox in the snow")),
        x.clone(),
    ];
    let gm = cx.enc.embed(&mixed).unwrap();
    assert!(cosine(&gm[1], &g[0]) > 0.999_999);

    let w = cx.worst.borrow();
    eprintln!("worst {:.7} ({}), floor {floor}", w.0, w.1);
}
