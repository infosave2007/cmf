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
