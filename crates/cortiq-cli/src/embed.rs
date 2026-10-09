//! `cortiq embed` — text embeddings from an EmbeddingGemma 2 `.cmf`.
//!
//! Every input is one embedding: unit length, 768-d, or a Matryoshka prefix
//! (`--dim 512|256|128`, re-normalized). `--prompt-name` applies one of the
//! model's task prompts (`SearchQuery`, `Document`, `QuestionAnswering`, …;
//! `--list-prompts` prints them); `--title` fills the Document prompt.
//! `--jsonl` gives each line its own prompt/title.

use anyhow::{Context, Result, anyhow};
use cortiq_core::CmfModel;
use cortiq_engine::egemma2::{EmbeddingGemma2, TextInput, cosine, matryoshka};
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
    let mut inputs: Vec<TextInput> = args
        .texts
        .iter()
        .map(|t| TextInput {
            text: t.clone(),
            ..defaults.clone()
        })
        .collect();
    if let Some(f) = &args.file {
        let body = std::fs::read_to_string(f).with_context(|| f.clone())?;
        for line in body.lines().filter(|l| !l.trim().is_empty()) {
            inputs.push(TextInput {
                text: line.to_string(),
                ..defaults.clone()
            });
        }
    }
    if let Some(f) = &args.jsonl {
        let body = std::fs::read_to_string(f).with_context(|| f.clone())?;
        for (i, line) in body.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            inputs
                .push(parse_jsonl_line(line, &defaults).with_context(|| format!("{f}:{}", i + 1))?);
        }
    }
    if inputs.is_empty() {
        return Err(anyhow!(
            "nothing to embed: give texts (positional or --text), --file or --jsonl"
        ));
    }
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
    let mut times = Vec::new();
    let mut full = Vec::new();
    for _ in 0..args.repeat.max(1) {
        let t0 = std::time::Instant::now();
        full = enc.embed_ids(&ids).map_err(anyhow::Error::msg)?;
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
