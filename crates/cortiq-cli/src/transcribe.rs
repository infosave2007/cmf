use anyhow::{Context, Result};

pub fn run(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    audio: &str,
    language: &str,
    task: &str,
    max_new_tokens: usize,
) -> Result<String> {
    cortiq_engine::whisper::transcribe(
        model,
        std::path::Path::new(audio),
        language,
        task,
        max_new_tokens,
    )
    .map_err(anyhow::Error::msg)
    .with_context(|| format!("transcribing {audio}"))
}

/// Nemotron 3.5 Transcribe: stream the transcript of every file to stdout
/// (one load for all of them; with several files each line is prefixed by
/// the file name).
pub fn run_nemotron(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    audio: &[String],
    prompt: Option<&str>,
    max_new_tokens: usize,
    stats: bool,
    timestamps: bool,
) -> Result<()> {
    use std::io::Write;
    let t0 = std::time::Instant::now();
    let tr = cortiq_engine::nemotron_asr::Transcriber::load(model).map_err(anyhow::Error::msg)?;
    let load_s = t0.elapsed().as_secs_f64();
    if stats {
        eprintln!("load {load_s:.1}s");
    }
    let pool = cortiq_engine::pool::Pool::from_env();
    let many = audio.len() > 1;
    anyhow::ensure!(
        !timestamps || tr.words.is_some(),
        "this file has no word timestamp head (repack it with cortiq 0.8.45 or later)"
    );
    for path in audio {
        let samples = cortiq_engine::nemotron_asr::read_audio(std::path::Path::new(path))
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("reading {path}"))?;
        anyhow::ensure!(!samples.is_empty(), "{path}: no samples");
        // Parity tooling: CMF_NEMO_PARITY_DIR=<dir> CMF_NEMO_FORCE_IDS=<file of
        // whitespace-separated reference ids> dumps the intermediate tensors and
        // the teacher-forced logits instead of transcribing.
        if let Ok(dir) = std::env::var("CMF_NEMO_PARITY_DIR") {
            let ids: Vec<u32> = match std::env::var("CMF_NEMO_FORCE_IDS") {
                Ok(p) => std::fs::read_to_string(&p)?
                    .split_whitespace()
                    .map(|s| s.parse::<u32>())
                    .collect::<std::result::Result<_, _>>()?,
                Err(_) => Vec::new(),
            };
            tr.dump_parity(&samples, prompt, &ids, std::path::Path::new(&dir), pool.as_deref())
                .map_err(anyhow::Error::msg)?;
            eprintln!("parity dump written to {dir}");
            continue;
        }
        let mut stdout = std::io::stdout();
        if many {
            let name = std::path::Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or(path);
            let _ = write!(stdout, "{name}: ");
        }
        let mut first = true;
        let out = tr
            .transcribe(&samples, prompt, max_new_tokens, timestamps, pool.as_deref(), |piece| {
                let piece = if first { piece.trim_start() } else { piece };
                first = false;
                let _ = stdout.write_all(piece.as_bytes());
                let _ = stdout.flush();
            })
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("transcribing {path}"))?;
        println!();
        match &out.words {
            Some(Ok(words)) => {
                for w in words {
                    match w.speaker {
                        Some(s) if out.text.contains("<spk:") => {
                            println!("  {:9.3} {:9.3}  spk:{s}  {}", w.start, w.end, w.word)
                        }
                        _ => println!("  {:9.3} {:9.3}  {}", w.start, w.end, w.word),
                    }
                }
            }
            Some(Err(e)) => eprintln!("{path}: word timestamps failed: {e}"),
            None => {}
        }
        if stats {
            let n_gen = out.ids.len();
            eprintln!(
                "{path}: audio {:.2}s | audio encoder {:.2}s | prefill {} tokens ({} audio) {:.2}s = {:.0} tok/s | decode {} tokens {:.2}s = {:.1} tok/s{} | real-time factor {:.3}",
                out.audio_s,
                out.encode_s,
                out.prompt_tokens,
                out.audio_tokens,
                out.prefill_s,
                out.prompt_tokens as f64 / out.prefill_s.max(1e-9),
                n_gen,
                out.decode_s,
                n_gen as f64 / out.decode_s.max(1e-9),
                if out.words.is_some() { format!(" | word timestamps {:.2}s", out.words_s) } else { String::new() },
                (out.encode_s + out.prefill_s + out.decode_s + out.words_s) / out.audio_s.max(1e-9),
            );
        }
    }
    Ok(())
}

/// GigaAM-v4: transcribe every file (one load for all of them; with several
/// files each line is prefixed by the file name).
#[allow(clippy::too_many_arguments)]
pub fn run_gigaam(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    audio: &[String],
    head: &str,
    stats: bool,
    timestamps: bool,
    json: bool,
    vad: bool,
) -> Result<()> {
    use cortiq_engine::gigaam::{self, Heads};
    let heads = match head {
        "e2e" | "text" => Heads::E2e,
        "raw" => Heads::Raw,
        "both" => Heads::Both,
        other => anyhow::bail!("--head {other}: use e2e (formatted text), raw (lower-case words) or both"),
    };
    let t0 = std::time::Instant::now();
    let tr = gigaam::Transcriber::load(model).map_err(anyhow::Error::msg)?;
    if stats {
        eprintln!("load {:.2}s ({}, device {})", t0.elapsed().as_secs_f64(), tr.revision, cortiq_engine::gpu::backend_name());
    }
    let pool = cortiq_engine::pool::Pool::from_env();
    let opts = gigaam::Options { heads, words: timestamps, force_vad: vad, ..Default::default() };
    let many = audio.len() > 1;
    for path in audio {
        let t1 = std::time::Instant::now();
        let samples = gigaam::read_audio(std::path::Path::new(path)).map_err(anyhow::Error::msg)?;
        let read_s = t1.elapsed().as_secs_f64();
        // Parity tooling: CMF_GIGAAM_TRACE_DIR=<dir> writes the intermediate
        // tensors of a clip of at most 30 s as .npy instead of transcribing.
        if let Ok(dir) = std::env::var("CMF_GIGAAM_TRACE_DIR") {
            std::fs::create_dir_all(&dir)?;
            for (name, data, width) in tr.trace(&samples, pool.as_deref()).map_err(anyhow::Error::msg)? {
                write_npy(&std::path::Path::new(&dir).join(format!("{name}.npy")), &data, width)?;
            }
            eprintln!("trace written to {dir}");
            continue;
        }
        let out = tr.transcribe(&samples, &opts, pool.as_deref()).map_err(anyhow::Error::msg).with_context(|| format!("transcribing {path}"))?;
        if json {
            let part = |t: &Option<gigaam::Transcript>| {
                t.as_ref().map(|t| {
                    serde_json::json!({
                        "text": t.text,
                        "segments": t.segments.iter().map(|s| serde_json::json!({
                            "start": s.start, "end": s.end, "text": s.text,
                            "words": s.words.iter().map(|w| serde_json::json!({"word": w.text, "start": w.start, "end": w.end})).collect::<Vec<_>>(),
                        })).collect::<Vec<_>>(),
                    })
                })
            };
            println!(
                "{}",
                serde_json::json!({
                    "file": path, "duration": out.audio_s, "e2e": part(&out.e2e), "raw": part(&out.raw),
                    "seconds": {"vad": out.vad_s, "features": out.features_s, "encoder": out.encoder_s, "decode": out.decode_s, "total": out.total_s()},
                })
            );
        } else {
            let name = std::path::Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or(path);
            for (label, t) in [("", &out.e2e), ("raw: ", &out.raw)] {
                let Some(t) = t else { continue };
                let label = if heads == Heads::Both { label } else { "" };
                if many {
                    println!("{name}: {label}{}", t.text);
                } else {
                    println!("{label}{}", t.text);
                }
                if timestamps {
                    for w in t.segments.iter().flat_map(|s| &s.words) {
                        println!("  {:9.3} {:9.3}  {}", w.start, w.end, w.text);
                    }
                }
            }
        }
        if stats {
            let total = out.total_s();
            eprintln!(
                "{path}: audio {:.2}s (speech {:.2}s, {} segments) | read {:.3}s | vad {:.3}s | features {:.3}s | encoder {:.3}s | decode {:.3}s | total {:.3}s = {:.1}x real time (RTF {:.4})",
                out.audio_s,
                out.speech_s,
                out.e2e.as_ref().or(out.raw.as_ref()).map_or(0, |t| t.segments.len()),
                read_s,
                out.vad_s,
                out.features_s,
                out.encoder_s,
                out.decode_s,
                total,
                out.audio_s / total.max(1e-9),
                total / out.audio_s.max(1e-9),
            );
        }
    }
    Ok(())
}

/// A `[rows, width]` float32 array as a NumPy `.npy` file.
fn write_npy(path: &std::path::Path, data: &[f32], width: usize) -> Result<()> {
    use std::io::Write;
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}", data.len() / width.max(1), width);
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
