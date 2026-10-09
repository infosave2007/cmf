//! `cortiq embed` — text and audio embeddings from an EmbeddingGemma 2
//! `.cmf`.
//!
//! Every input is one embedding: unit length, 768-d, or a Matryoshka prefix
//! (`--dim 512|256|128`, re-normalized). `--prompt-name` applies one of the
//! model's task prompts (`SearchQuery`, `Document`, `QuestionAnswering`, …;
//! `--list-prompts` prints them); `--title` fills the Document prompt.
//! `--jsonl` gives each line its own prompt/title.
//!
//! `--audio PATH` (repeatable) embeds a clip on its own, unprompted: WAV
//! natively, other formats through `ffmpeg`; any rate or channel count is
//! mixed down to mono and resampled to 16 kHz; clips are cut at 30 s. A
//! `--jsonl` line `{"text": "… <|audio|> …", "audio": ["a.wav"]}` interleaves
//! text and audio in one input (one `<|audio|>` per clip, in order);
//! `{"audio": "a.wav"}` is an audio-only line.

use anyhow::{Context, Result, anyhow};
use cortiq_core::CmfModel;
use cortiq_engine::egemma2::{EmbeddingGemma2, TextInput, cosine, matryoshka};
use cortiq_engine::egemma2_audio::{self as ea, AudioInput, AudioTower};
use std::io::Write;
use std::sync::Arc;

pub struct EmbedArgs {
    pub model: String,
    pub texts: Vec<String>,
    /// audio files, one audio-only input each
    pub audio: Vec<String>,
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
}

/// One JSON Lines input: a bare string, or an object.
fn parse_jsonl_line(line: &str, defaults: &TextInput) -> Result<TextInput> {
    let v: serde_json::Value = serde_json::from_str(line)?;
    if let Some(s) = v.as_str() {
        return Ok(TextInput {
            text: s.to_string(),
            ..defaults.clone()
        });
    }
    let text = v
        .get("text")
        .or_else(|| v.get("input"))
        .and_then(|t| t.as_str())
        .ok_or_else(|| anyhow!("expected a string or an object with \"text\""))?;
    let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
    let has = |k: &str| v.get(k).is_some();
    Ok(TextInput {
        text: text.to_string(),
        // a line that names its own prompt (even `null`) overrides the flag
        prompt_name: if has("prompt_name") || has("task") {
            s("prompt_name").or_else(|| s("task"))
        } else {
            defaults.prompt_name.clone()
        },
        title: if has("title") {
            s("title")
        } else {
            defaults.title.clone()
        },
        prompt: if has("prompt") {
            s("prompt")
        } else {
            defaults.prompt.clone()
        },
    })
}

/// One input of the run, in output order.
enum Job {
    Text(TextInput),
    /// audio clips (paths, for the listing) with optional placeholder text
    Audio(Vec<String>, AudioInput),
}

/// The `audio` field of a JSON Lines object: a path or a list of paths.
fn audio_paths(v: &serde_json::Value) -> Result<Option<Vec<String>>> {
    match v.get("audio") {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(p)) => Ok(Some(vec![p.clone()])),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("\"audio\" must hold file paths"))
            })
            .collect::<Result<Vec<_>>>()
            .map(Some),
        Some(_) => Err(anyhow!("\"audio\" must be a path or a list of paths")),
    }
}

/// A JSON Lines input: text (as [`parse_jsonl_line`]) or, with an `audio`
/// field, audio clips with optional `<|audio|>` text.
fn parse_jsonl_job(line: &str, defaults: &TextInput) -> Result<Job> {
    let v: serde_json::Value = serde_json::from_str(line)?;
    let Some(paths) = audio_paths(&v)? else {
        return Ok(Job::Text(parse_jsonl_line(line, defaults)?));
    };
    if paths.is_empty() {
        return Err(anyhow!("\"audio\" is empty"));
    }
    let has_text = v
        .get("text")
        .or_else(|| v.get("input"))
        .is_some_and(|t| t.is_string());
    let text = if has_text {
        Some(parse_jsonl_line(line, defaults)?)
    } else {
        None
    };
    let clips = paths
        .iter()
        .map(|p| ea::read_audio(std::path::Path::new(p)).map_err(anyhow::Error::msg))
        .collect::<Result<Vec<_>>>()?;
    Ok(Job::Audio(paths, AudioInput { text, clips }))
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

pub fn run(args: EmbedArgs) -> Result<()> {
    let t_load = std::time::Instant::now();
    let model = Arc::new(CmfModel::open(&args.model).with_context(|| args.model.clone())?);
    let pool = cortiq_engine::pool::Pool::from_env();
    let enc = EmbeddingGemma2::load(&model, pool).map_err(anyhow::Error::msg)?;
    let load_s = t_load.elapsed().as_secs_f64();
    if args.list_prompts {
        for name in enc.prompts().names() {
            println!("{name}\t{:?}", enc.prompts().get(name).unwrap_or(""));
        }
        return Ok(());
    }
    if !cortiq_engine::egemma2::MATRYOSHKA_DIMS.contains(&args.dim) {
        return Err(anyhow!("--dim {}: use 768, 512, 256 or 128", args.dim));
    }
    let defaults = TextInput {
        text: String::new(),
        prompt_name: args.prompt_name.clone(),
        title: args.title.clone(),
        prompt: args.prompt.clone(),
    };
    let mut jobs: Vec<Job> = args
        .texts
        .iter()
        .map(|t| {
            Job::Text(TextInput {
                text: t.clone(),
                ..defaults.clone()
            })
        })
        .collect();
    if let Some(f) = &args.file {
        let body = std::fs::read_to_string(f).with_context(|| f.clone())?;
        for line in body.lines().filter(|l| !l.trim().is_empty()) {
            jobs.push(Job::Text(TextInput {
                text: line.to_string(),
                ..defaults.clone()
            }));
        }
    }
    if let Some(f) = &args.jsonl {
        let body = std::fs::read_to_string(f).with_context(|| f.clone())?;
        for (i, line) in body.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            jobs.push(parse_jsonl_job(line, &defaults).with_context(|| format!("{f}:{}", i + 1))?);
        }
    }
    for p in &args.audio {
        let clip = ea::read_audio(std::path::Path::new(p)).map_err(anyhow::Error::msg)?;
        jobs.push(Job::Audio(
            vec![p.clone()],
            AudioInput {
                text: None,
                clips: vec![clip],
            },
        ));
    }
    if jobs.is_empty() {
        return Err(anyhow!(
            "nothing to embed: give texts (positional or --text), --file, --jsonl or --audio"
        ));
    }
    let mut text_idx = Vec::new();
    let mut text_ids: Vec<Vec<u32>> = Vec::new();
    let mut audio_idx = Vec::new();
    let mut audio_in: Vec<AudioInput> = Vec::new();
    for (i, j) in jobs.into_iter().enumerate() {
        match j {
            Job::Text(x) => {
                text_ids.push(enc.input_ids(&x).map_err(|e| anyhow!("input {i}: {e}"))?);
                text_idx.push(i);
            }
            Job::Audio(paths, a) => {
                for (p, c) in paths.iter().zip(&a.clips) {
                    eprintln!(
                        "[{i}] audio {p}: {:.2} s → {} tokens{}",
                        c.len() as f64 / ea::SAMPLE_RATE as f64,
                        ea::num_tokens(c.len()),
                        if c.len() > ea::MAX_SAMPLES {
                            " (cut at 30 s)"
                        } else {
                            ""
                        }
                    );
                }
                audio_in.push(a);
                audio_idx.push(i);
            }
        }
    }
    let n_inputs = text_idx.len() + audio_idx.len();
    let tower = if audio_in.is_empty() {
        None
    } else {
        let t0 = std::time::Instant::now();
        let t = AudioTower::load(&model, cortiq_engine::pool::Pool::from_env())
            .map_err(anyhow::Error::msg)?;
        eprintln!("audio tower loaded in {:.2}s", t0.elapsed().as_secs_f64());
        Some(t)
    };
    // token ids of every input, in output order
    let mut ids: Vec<Vec<u32>> = vec![Vec::new(); n_inputs];
    for (k, &i) in text_idx.iter().enumerate() {
        ids[i] = text_ids[k].clone();
    }
    if let Some(t) = &tower {
        for (k, &i) in audio_idx.iter().enumerate() {
            let a = &audio_in[k];
            let n: Vec<usize> = a.clips.iter().map(|c| ea::num_tokens(c.len())).collect();
            ids[i] = ea::input_ids(&enc, (t.audio_token, t.boa_token, t.eoa_token), a, &n)
                .map_err(|e| anyhow!("input {i}: {e}"))?;
        }
    }
    let tokens: usize = ids.iter().map(|s| s.len()).sum();
    if args.show_tokens {
        for (i, s) in ids.iter().enumerate() {
            eprintln!("[{i}] {} tokens: {s:?}", s.len());
        }
    }
    let mut times = Vec::new();
    let mut full: Vec<Vec<f32>> = Vec::new();
    for _ in 0..args.repeat.max(1) {
        let t0 = std::time::Instant::now();
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); n_inputs];
        if !text_ids.is_empty() {
            for (v, &i) in enc
                .embed_ids(&text_ids)
                .map_err(anyhow::Error::msg)?
                .into_iter()
                .zip(&text_idx)
            {
                out[i] = v;
            }
        }
        if let Some(t) = &tower {
            for (v, &i) in ea::embed_audio_inputs(&enc, t, &audio_in)
                .map_err(anyhow::Error::msg)?
                .into_iter()
                .zip(&audio_idx)
            {
                out[i] = v;
            }
        }
        full = out;
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
        "embedded {} input(s), {tokens} tokens in {:.3}s ({:.0} tok/s; load {:.2}s, {})",
        vecs.len(),
        secs,
        tokens as f64 / secs.max(1e-9),
        load_s,
        if enc.quant.is_empty() {
            "?"
        } else {
            &enc.quant
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
            writeln!(
                out,
                "[{i}] {} tokens, dim {}: {} …",
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
        let d = TextInput {
            text: String::new(),
            prompt_name: Some("SearchQuery".into()),
            title: None,
            prompt: None,
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
    fn jsonl_audio_lines() {
        let v = |s: &str| serde_json::from_str::<serde_json::Value>(s).unwrap();
        assert_eq!(audio_paths(&v(r#"{"text":"x"}"#)).unwrap(), None);
        assert_eq!(
            audio_paths(&v(r#"{"audio":"a.wav"}"#)).unwrap(),
            Some(vec!["a.wav".to_string()])
        );
        assert_eq!(
            audio_paths(&v(r#"{"audio":["a.wav","b.flac"]}"#))
                .unwrap()
                .unwrap()
                .len(),
            2
        );
        assert!(audio_paths(&v(r#"{"audio":[1]}"#)).is_err());
        assert!(audio_paths(&v(r#"{"audio":3}"#)).is_err());
        let d = TextInput::default();
        assert!(parse_jsonl_job(r#"{"audio":[]}"#, &d).is_err());
        assert!(parse_jsonl_job(r#"{"audio":"/nonexistent/x.wav"}"#, &d).is_err());
        assert!(matches!(
            parse_jsonl_job(r#"{"text":"plain"}"#, &d).unwrap(),
            Job::Text(_)
        ));
        // a real clip: an audio-only line and one with placeholder text
        let wav = std::env::temp_dir().join(format!("egemma2-jsonl-{}.wav", std::process::id()));
        let mut b = Vec::new();
        let data: Vec<u8> = (0..1600i16).flat_map(|i| (i * 7).to_le_bytes()).collect();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&16000u32.to_le_bytes());
        b.extend_from_slice(&32000u32.to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(&data);
        std::fs::write(&wav, &b).unwrap();
        let p = wav.to_str().unwrap();
        let Job::Audio(paths, a) = parse_jsonl_job(&format!(r#"{{"audio":"{p}"}}"#), &d).unwrap()
        else {
            panic!()
        };
        assert_eq!(paths, vec![p.to_string()]);
        assert!(a.text.is_none());
        assert_eq!(a.clips[0].len(), 1600);
        let Job::Audio(_, a) = parse_jsonl_job(
            &format!(r#"{{"text":"Said: <|audio|>","audio":["{p}"],"task":"Clustering"}}"#),
            &d,
        )
        .unwrap() else {
            panic!()
        };
        let t = a.text.unwrap();
        assert_eq!(t.text, "Said: <|audio|>");
        assert_eq!(t.prompt_name.as_deref(), Some("Clustering"));
        let _ = std::fs::remove_file(&wav);
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
