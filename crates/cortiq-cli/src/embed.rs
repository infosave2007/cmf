//! `cortiq embed` — embeddings from an EmbeddingGemma 2 `.cmf`: texts,
//! images, videos, and interleaved text + media, all in one 768-d space.
//!
//! Every input is one embedding: unit length, 768-d, or a Matryoshka prefix
//! (`--dim 512|256|128`, re-normalized). `--prompt-name` applies one of the
//! model's task prompts (`SearchQuery`, `Document`, `QuestionAnswering`, …;
//! `--list-prompts` prints them) to the text inputs; `--title` fills the
//! Document prompt. `--jsonl` gives each line its own prompt/title/media.
//!
//! Media: `--image PATH|URL` and `--video PATH` (repeatable) are inputs of
//! their own. A video is an mp4/webm/mov/… (decoded with the `ffmpeg`
//! executable), a `.y4m`, or a directory of frames (`--video-fps` gives
//! their rate; without it they are taken as already sampled at 1 fps).
//! `--interleave` makes ONE input of the text and the media instead: the
//! text's `<|image|>` / `<|video|>` placeholders take them in order (or,
//! with no placeholders, the media come first). `--image-tokens` /
//! `--video-tokens` set the soft-token budget (70 | 140 | 280 | 560 | 1120;
//! defaults 280 per image, 140 per frame).

use anyhow::{Context, Result, anyhow};
use cortiq_core::CmfModel;
use cortiq_engine::egemma2::{cosine, matryoshka};
use cortiq_engine::egemma2_mm::{Media, MediaEncoder, MixedInput};
use cortiq_engine::egemma2_vision::decode_video;
use std::io::Write;
use std::sync::Arc;

pub struct EmbedArgs {
    pub model: String,
    pub texts: Vec<String>,
    pub file: Option<String>,
    pub jsonl: Option<String>,
    pub prompt_name: Option<String>,
    pub title: Option<String>,
    pub prompt: Option<String>,
    pub dim: usize,
    pub json: bool,
    pub npy: Option<String>,
    pub show_tokens: bool,
    pub list_prompts: bool,
    /// run the forward this many times and report each (in-process timing)
    pub repeat: usize,
    pub images: Vec<String>,
    pub videos: Vec<String>,
    pub video_fps: Option<f64>,
    pub image_tokens: Option<usize>,
    pub video_tokens: Option<usize>,
    pub interleave: bool,
}

/// One input as given, before its media are decoded.
#[derive(Clone, Debug, Default, PartialEq)]
struct Spec {
    text: String,
    images: Vec<String>,
    videos: Vec<String>,
    prompt_name: Option<String>,
    title: Option<String>,
    prompt: Option<String>,
    image_tokens: Option<usize>,
    video_tokens: Option<usize>,
}

fn str_or_list(v: Option<&serde_json::Value>) -> Result<Vec<String>> {
    match v {
        None | Some(serde_json::Value::Null) => Ok(Vec::new()),
        Some(serde_json::Value::String(s)) => Ok(vec![s.clone()]),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("media lists hold strings (paths or URLs)"))
            })
            .collect(),
        Some(_) => Err(anyhow!("media must be a string or a list of strings")),
    }
}

/// One JSON Lines input: a bare string, or an object
/// `{"text", "image", "video", "prompt_name"|"task", "title", "prompt",
/// "image_tokens", "video_tokens"}`.
fn parse_jsonl_line(line: &str, defaults: &Spec) -> Result<Spec> {
    let v: serde_json::Value = serde_json::from_str(line)?;
    if let Some(s) = v.as_str() {
        return Ok(Spec {
            text: s.to_string(),
            ..defaults.clone()
        });
    }
    let images = str_or_list(v.get("image").or_else(|| v.get("images")))?;
    let videos = str_or_list(v.get("video").or_else(|| v.get("videos")))?;
    let text = v
        .get("text")
        .or_else(|| v.get("input"))
        .and_then(|t| t.as_str());
    if text.is_none() && images.is_empty() && videos.is_empty() {
        return Err(anyhow!(
            "expected a string or an object with \"text\", \"image\" or \"video\""
        ));
    }
    let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
    let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
    let has = |k: &str| v.get(k).is_some();
    // sentence-transformers prompts text-only inputs: a line with media
    // takes a prompt only from itself
    let media = !images.is_empty() || !videos.is_empty();
    let inherit = |own: Option<String>, k: bool, d: &Option<String>| {
        if k {
            own
        } else if media {
            None
        } else {
            d.clone()
        }
    };
    Ok(Spec {
        text: text.unwrap_or("").to_string(),
        prompt_name: inherit(
            s("prompt_name").or_else(|| s("task")),
            has("prompt_name") || has("task"),
            &defaults.prompt_name,
        ),
        title: inherit(s("title"), has("title"), &defaults.title),
        prompt: inherit(s("prompt"), has("prompt"), &defaults.prompt),
        images,
        videos,
        image_tokens: u("image_tokens").or(defaults.image_tokens),
        video_tokens: u("video_tokens").or(defaults.video_tokens),
    })
}

fn write_npy(path: &str, rows: &[Vec<f32>]) -> Result<()> {
    let n = rows.len();
    let d = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({n}, {d}), }}");
    // magic(6) + version(2) + len(2) + header + '\n' aligned to 64
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + n * d * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for r in rows {
        for v in r {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, out).with_context(|| format!("write {path}"))
}

/// Read an image from a path, a `file://`, `http(s)://` or `data:` URL.
fn load_image(src: &str) -> Result<cortiq_engine::media::RgbFrame> {
    let bytes = cortiq_engine::media::load_image_bytes(&serde_json::json!({ "url": src }))
        .map_err(|e| anyhow!("{src}: {e}"))?;
    cortiq_engine::media::decode_rgb(&bytes).map_err(|e| anyhow!("{src}: {e}"))
}

/// Decode a spec's media and build the input.
fn build(enc: &MediaEncoder, s: &Spec, fps: Option<f64>) -> Result<(MixedInput, String)> {
    let mut media = Vec::with_capacity(s.images.len() + s.videos.len());
    let mut notes = Vec::new();
    // images first, then videos: each kind fills its own placeholders in order
    for p in &s.images {
        let img = load_image(p)?;
        let m = enc
            .prepare_image(&img, s.image_tokens)
            .map_err(|e| anyhow!("{p}: {e}"))?;
        if let Media::Image(v) = &m {
            notes.push(format!(
                "image {}x{} → {}x{} px, {} tokens",
                img.width,
                img.height,
                v.pw * enc.proc.patch,
                v.ph * enc.proc.patch,
                v.n_soft(enc.proc.pool_k)
            ));
        }
        media.push(m);
    }
    for p in &s.videos {
        let dv = decode_video(std::path::Path::new(p), fps, &enc.proc)
            .map_err(|e| anyhow!("{p}: {e}"))?;
        let m = enc
            .prepare_frames(&dv.frames, s.video_tokens)
            .map_err(|e| anyhow!("{p}: {e}"))?;
        if let Media::Video(f) = &m {
            notes.push(format!(
                "video {} frames{} via {} → {} sampled {:?}, {} tokens each",
                dv.total_frames,
                dv.src_fps
                    .map(|f| format!(" @ {f:.3} fps"))
                    .unwrap_or_default(),
                dv.decoder,
                f.len(),
                dv.indices,
                f.first().map(|x| x.n_soft(enc.proc.pool_k)).unwrap_or(0)
            ));
        }
        media.push(m);
    }
    Ok((
        MixedInput {
            text: s.text.clone(),
            media,
            prompt_name: s.prompt_name.clone(),
            title: s.title.clone(),
            prompt: s.prompt.clone(),
        },
        notes.join("; "),
    ))
}

pub fn run(args: EmbedArgs) -> Result<()> {
    let t_load = std::time::Instant::now();
    let model = Arc::new(CmfModel::open(&args.model).with_context(|| args.model.clone())?);
    let pool = cortiq_engine::pool::Pool::from_env();
    let enc = MediaEncoder::load(&model, pool).map_err(anyhow::Error::msg)?;
    let load_s = t_load.elapsed().as_secs_f64();
    if args.list_prompts {
        for name in enc.text.prompts().names() {
            println!("{name}\t{:?}", enc.text.prompts().get(name).unwrap_or(""));
        }
        return Ok(());
    }
    if !cortiq_engine::egemma2::MATRYOSHKA_DIMS.contains(&args.dim) {
        return Err(anyhow!("--dim {}: use 768, 512, 256 or 128", args.dim));
    }
    let defaults = Spec {
        prompt_name: args.prompt_name.clone(),
        title: args.title.clone(),
        prompt: args.prompt.clone(),
        image_tokens: args.image_tokens,
        video_tokens: args.video_tokens,
        ..Default::default()
    };
    let mut specs: Vec<Spec> = Vec::new();
    let text_spec = |t: &str| Spec {
        text: t.to_string(),
        ..defaults.clone()
    };
    if args.interleave {
        if args.texts.len() > 1 || args.file.is_some() || args.jsonl.is_some() {
            return Err(anyhow!(
                "--interleave makes one input of ONE text and the --image/--video media \
                 (use --jsonl for several interleaved inputs)"
            ));
        }
        specs.push(Spec {
            text: args.texts.first().cloned().unwrap_or_default(),
            images: args.images.clone(),
            videos: args.videos.clone(),
            ..defaults.clone()
        });
    } else {
        specs.extend(args.texts.iter().map(|t| text_spec(t)));
        if let Some(f) = &args.file {
            let body = std::fs::read_to_string(f).with_context(|| f.clone())?;
            for line in body.lines().filter(|l| !l.trim().is_empty()) {
                specs.push(text_spec(line));
            }
        }
        if let Some(f) = &args.jsonl {
            let body = std::fs::read_to_string(f).with_context(|| f.clone())?;
            for (i, line) in body.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                specs.push(
                    parse_jsonl_line(line, &defaults).with_context(|| format!("{f}:{}", i + 1))?,
                );
            }
        }
        // media alone take no prompt (sentence-transformers)
        let bare = Spec {
            image_tokens: args.image_tokens,
            video_tokens: args.video_tokens,
            ..Default::default()
        };
        for p in &args.images {
            specs.push(Spec {
                images: vec![p.clone()],
                ..bare.clone()
            });
        }
        for p in &args.videos {
            specs.push(Spec {
                videos: vec![p.clone()],
                ..bare.clone()
            });
        }
    }
    if specs.is_empty() {
        return Err(anyhow!(
            "nothing to embed: give texts (positional or --text), --file, --jsonl, --image or --video"
        ));
    }
    let t_prep = std::time::Instant::now();
    let mut inputs = Vec::with_capacity(specs.len());
    let mut notes = Vec::with_capacity(specs.len());
    for (i, s) in specs.iter().enumerate() {
        let (x, note) = build(&enc, s, args.video_fps).with_context(|| format!("input {i}"))?;
        inputs.push(x);
        notes.push(note);
    }
    let prep_s = t_prep.elapsed().as_secs_f64();
    let ids: Vec<Vec<u32>> = inputs
        .iter()
        .enumerate()
        .map(|(i, x)| enc.input_ids(x).map_err(|e| anyhow!("input {i}: {e}")))
        .collect::<Result<_>>()?;
    let tokens: usize = ids.iter().map(|s| s.len()).sum();
    if args.show_tokens {
        for (i, s) in ids.iter().enumerate() {
            eprintln!("[{i}] {} tokens: {s:?}", s.len());
        }
    }
    let has_media = inputs.iter().any(|x| !x.media.is_empty());
    if has_media {
        // the tower's one-time load is not part of the timed forward
        let t = std::time::Instant::now();
        enc.warm_vision().map_err(anyhow::Error::msg)?;
        eprintln!("vision tower loaded in {:.2}s", t.elapsed().as_secs_f64());
    }
    let mut times = Vec::new();
    let mut full = Vec::new();
    for _ in 0..args.repeat.max(1) {
        let t0 = std::time::Instant::now();
        full = enc.embed(&inputs).map_err(anyhow::Error::msg)?;
        times.push(t0.elapsed().as_secs_f64());
    }
    if times.len() > 1 {
        let ms: Vec<String> = times.iter().map(|t| format!("{:.1}", t * 1e3)).collect();
        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        eprintln!(
            "repeat {}: [{}] ms; median {:.1} ms = {:.0} tok/s",
            times.len(),
            ms.join(", "),
            sorted[sorted.len() / 2] * 1e3,
            tokens as f64 / sorted[sorted.len() / 2]
        );
    }
    let secs = times[0];
    let vecs: Vec<Vec<f32>> = full
        .iter()
        .map(|v| matryoshka(v, args.dim))
        .collect::<std::result::Result<_, _>>()
        .map_err(anyhow::Error::msg)?;
    eprintln!(
        "embedded {} input(s), {tokens} tokens in {:.3}s ({:.0} tok/s; load {:.2}s{}, {})",
        vecs.len(),
        secs,
        tokens as f64 / secs.max(1e-9),
        load_s,
        if has_media {
            format!(", media decode+resize {prep_s:.2}s")
        } else {
            String::new()
        },
        if enc.text.quant.is_empty() {
            "?"
        } else {
            &enc.text.quant
        }
    );
    if let Some(p) = &args.npy {
        write_npy(p, &vecs)?;
        eprintln!("wrote {p} ({} x {})", vecs.len(), args.dim);
    }
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    if args.json {
        let data: Vec<serde_json::Value> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| serde_json::json!({"object": "embedding", "index": i, "embedding": v}))
            .collect();
        let body = serde_json::json!({
            "object": "list",
            "model": args.model,
            "data": data,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        });
        writeln!(out, "{body}")?;
    } else if args.npy.is_none() {
        for (i, v) in vecs.iter().enumerate() {
            let head: Vec<String> = v.iter().take(6).map(|x| format!("{x:+.5}")).collect();
            let what = if notes[i].is_empty() {
                String::new()
            } else {
                format!(" ({})", notes[i])
            };
            writeln!(
                out,
                "[{i}] {} tokens, dim {}{what}: {} …",
                ids[i].len(),
                v.len(),
                head.join(" ")
            )?;
        }
        if (2..=8).contains(&vecs.len()) {
            writeln!(out, "cosine similarity:")?;
            for a in &vecs {
                let row: Vec<String> = vecs
                    .iter()
                    .map(|b| format!("{:.4}", cosine(a, b)))
                    .collect();
                writeln!(out, "  {}", row.join("  "))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_lines_override_defaults() {
        let d = Spec {
            prompt_name: Some("SearchQuery".into()),
            ..Default::default()
        };
        let a = parse_jsonl_line("\"hello\"", &d).unwrap();
        assert_eq!(a.text, "hello");
        assert_eq!(a.prompt_name.as_deref(), Some("SearchQuery"));
        let b =
            parse_jsonl_line(r#"{"text":"doc","prompt_name":"Document","title":"T"}"#, &d).unwrap();
        assert_eq!(b.prompt_name.as_deref(), Some("Document"));
        assert_eq!(b.title.as_deref(), Some("T"));
        let c = parse_jsonl_line(r#"{"text":"raw","prompt_name":null}"#, &d).unwrap();
        assert_eq!(c.prompt_name, None);
        let t = parse_jsonl_line(r#"{"input":"x","task":"Clustering"}"#, &d).unwrap();
        assert_eq!(t.prompt_name.as_deref(), Some("Clustering"));
        assert!(parse_jsonl_line("{\"nope\":1}", &d).is_err());
    }

    #[test]
    fn jsonl_media_lines() {
        let d = Spec {
            prompt_name: Some("SearchQuery".into()),
            image_tokens: Some(560),
            ..Default::default()
        };
        let a = parse_jsonl_line(r#"{"image":"a.png"}"#, &d).unwrap();
        assert_eq!(a.images, vec!["a.png".to_string()]);
        // media inputs do not inherit the default prompt
        assert_eq!(a.prompt_name, None);
        assert_eq!(a.image_tokens, Some(560));
        let b = parse_jsonl_line(
            r#"{"text":"A fox: <|image|> and <|video|>","image":["f.png"],"video":"v.mp4","video_tokens":70}"#,
            &d,
        )
        .unwrap();
        assert_eq!(b.videos, vec!["v.mp4".to_string()]);
        assert_eq!(b.video_tokens, Some(70));
        let c = parse_jsonl_line(r#"{"image":"a.png","task":"Document"}"#, &d).unwrap();
        assert_eq!(c.prompt_name.as_deref(), Some("Document"));
        assert!(parse_jsonl_line(r#"{"image":5}"#, &d).is_err());
    }

    #[test]
    fn npy_header_is_aligned() {
        let dir = std::env::temp_dir().join(format!("egemma2-npy-{}", std::process::id()));
        let p = dir.with_extension("npy");
        write_npy(p.to_str().unwrap(), &[vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let b = std::fs::read(&p).unwrap();
        let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
        assert_eq!((10 + hl) % 64, 0);
        assert_eq!(b.len(), 10 + hl + 16);
        let _ = std::fs::remove_file(&p);
    }
}
