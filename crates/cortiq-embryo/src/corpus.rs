//! Tokenizer + corpus commands (portable; no GPU).

use std::path::PathBuf;
use std::time::Instant;

/// Train our byte-level BPE on up to `sample_bytes` of text from `inputs`.
pub fn train_tokenizer(inputs: &[PathBuf], out: &std::path::Path, vocab: usize, sample_bytes: usize) {
    use crate::tokenizer::{SPLIT, count_words, train};
    let re = fancy_regex::Regex::new(SPLIT).unwrap();
    let mut counts = std::collections::HashMap::new();
    // the byte budget is split equally across the inputs (a mixed corpus
    // should shape the vocabulary in proportion, not in file order)
    let per_input = sample_bytes / inputs.len().max(1);
    let mut seen_total = 0usize;
    let t0 = Instant::now();
    for p in inputs {
        let mut seen = 0usize;
        let n = crate::data::for_each_doc(p, |text| {
            if seen >= per_input {
                return;
            }
            seen += text.len();
            count_words(text, &re, &mut counts);
        })
        .expect("read corpus");
        seen_total += seen;
        eprintln!("{}: {n} docs read, {:.1} MB sampled ({:.1} MB total), {} word types [{:.0} s]", p.display(), seen as f64 / 1e6, seen_total as f64 / 1e6, counts.len(), t0.elapsed().as_secs_f64());
    }
    let bpe = train(&counts, vocab, true);
    bpe.save(out).expect("write tokenizer.json");
    println!("tokenizer: {} tokens, {} merges → {} [{:.0} s]", bpe.vocab_size(), bpe.merges.len(), out.display(), t0.elapsed().as_secs_f64());
}

/// Encode corpus files into one u16 shard (documents separated by EOT),
/// multi-threaded over document batches.
pub fn shard(tok: &std::path::Path, inputs: &[PathBuf], out: &std::path::Path, max_tokens: usize) {
    use crate::tokenizer::{Bpe, EOT};
    let bpe = std::sync::Arc::new(Bpe::load(tok).expect("load tokenizer"));
    let eot = bpe.special_id(EOT).expect("EOT id") as u16;
    assert!(bpe.vocab_size() <= 65536);
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut all: Vec<u16> = Vec::new();
    let t0 = Instant::now();
    'outer: for p in inputs {
        let mut batch: Vec<String> = Vec::new();
        let mut batch_bytes = 0usize;
        let mut flush = |batch: &mut Vec<String>, all: &mut Vec<u16>| {
            if batch.is_empty() {
                return;
            }
            let docs = std::mem::take(batch);
            let chunk = docs.len().div_ceil(nthreads).max(1);
            let parts: Vec<Vec<u16>> = std::thread::scope(|s| {
                let hs: Vec<_> = docs
                    .chunks(chunk)
                    .map(|ds| {
                        let bpe = bpe.clone();
                        s.spawn(move || {
                            let mut cache = std::collections::HashMap::new();
                            let mut ids = Vec::new();
                            let mut out = Vec::new();
                            for d in ds {
                                ids.clear();
                                bpe.encode(d, &mut cache, &mut ids);
                                out.extend(ids.iter().map(|&i| i as u16));
                                out.push(eot);
                            }
                            out
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for p in parts {
                all.extend_from_slice(&p);
            }
        };
        crate::data::for_each_doc(p, |text| {
            if all.len() >= max_tokens {
                return;
            }
            batch.push(text.to_string());
            batch_bytes += text.len();
            if batch_bytes >= 64 << 20 {
                flush(&mut batch, &mut all);
                batch_bytes = 0;
                eprintln!("  {:.1} M tokens [{:.0} s]", all.len() as f64 / 1e6, t0.elapsed().as_secs_f64());
            }
        })
        .expect("read corpus");
        flush(&mut batch, &mut all);
        eprintln!("{}: {:.1} M tokens total [{:.0} s]", p.display(), all.len() as f64 / 1e6, t0.elapsed().as_secs_f64());
        if all.len() >= max_tokens {
            break 'outer;
        }
    }
    all.truncate(max_tokens);
    crate::train::Shard { tokens: all }.save(out).expect("write shard");
    println!("shard: {} tokens → {}", std::fs::metadata(out).map(|m| m.len() / 2).unwrap_or(0), out.display());
}

/// Download URLs into `dir` with curl (resumable). Skips existing files.
pub fn fetch(urls: &[String], dir: &std::path::Path) {
    std::fs::create_dir_all(dir).expect("mkdir");
    for u in urls {
        let name = u.rsplit('/').next().unwrap_or("file").split('?').next().unwrap().to_string();
        let dst = dir.join(&name);
        if dst.exists() && !dst.with_extension("part").exists() {
            println!("have {}", dst.display());
            continue;
        }
        println!("fetch {u}");
        let st = std::process::Command::new("curl")
            .args(["-L", "-C", "-", "--retry", "5", "-o"])
            .arg(&dst)
            .arg(u)
            .status()
            .expect("curl");
        if !st.success() {
            eprintln!("curl failed for {u}");
        }
    }
}

/// Dump `docs` documents of a corpus file after skipping `skip` (held-out
/// text for `cortiq ppl` gates), each followed by a blank line.
pub fn sample_text(input: &std::path::Path, skip: usize, docs: usize, out: &std::path::Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(out).expect("create");
    let mut i = 0usize;
    let mut written = 0usize;
    crate::data::for_each_doc(input, |text| {
        if i >= skip && written < docs {
            let _ = f.write_all(text.as_bytes());
            let _ = f.write_all(b"\n\n");
            written += 1;
        }
        i += 1;
    })
    .expect("read");
    println!("{written} docs (skipped {skip} of {i}) → {}", out.display());
}
