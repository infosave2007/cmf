//! MiMo audio-tower dump for the parity gates (tools/mimo_audio_ref.py cmp).
//!
//! ```text
//! mimo_audio_dump decode   --wav F --out F.npy
//! mimo_audio_dump frontend --wav F --out DIR
//!     dec.npy [C,N], chan24k.npy [C,N'], wave24k.npy, mel.npy [M,128]
//! mimo_audio_dump tower --src (HF_DIR | X.cmf) --out DIR (--wav F | --mel M.npy) [--codes C.npy]
//!     feats.npy, codes_exact.npy, codes_bf16books.npy, embeds.npy (from --codes,
//!     else from codes_bf16books), embeds_own.npy, tower.json (timings)
//! mimo_audio_dump calib --src X.cmf --wav-dir D --out H.bin
//!     GPTQ input Hessians of every tower linear over the clips in D, in the
//!     `cortiq quantize-gptq --codec q4tp --hessians H.bin` cache format
//! ```
//!
//! The array names match the oracle's, so `mimo_audio_ref.py cmp --ref R
//! --eng E` lines them up.

use cortiq_engine::mimo_audio::{self, MimoAudio};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

fn npy_write(path: &Path, descr: &str, shape: &[usize], bytes: &[u8]) {
    let shape_s = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut hdr = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_s}, }}");
    let total = 10 + hdr.len() + 1;
    hdr.push_str(&" ".repeat((64 - total % 64) % 64));
    hdr.push('\n');
    let mut out = Vec::with_capacity(10 + hdr.len() + bytes.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(hdr.len() as u16).to_le_bytes());
    out.extend_from_slice(hdr.as_bytes());
    out.extend_from_slice(bytes);
    std::fs::write(path, out).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

fn save_f32(path: &Path, shape: &[usize], v: &[f32]) {
    assert_eq!(
        shape.iter().product::<usize>(),
        v.len(),
        "{}",
        path.display()
    );
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    npy_write(path, "<f4", shape, &bytes);
}

fn save_i32(path: &Path, shape: &[usize], v: &[u32]) {
    assert_eq!(shape.iter().product::<usize>(), v.len());
    let bytes: Vec<u8> = v.iter().flat_map(|x| (*x as i32).to_le_bytes()).collect();
    npy_write(path, "<i4", shape, &bytes);
}

/// `(descr, shape, payload)` of a C-order .npy.
fn npy_read(path: &Path) -> (String, Vec<usize>, Vec<u8>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&b[..6], b"\x93NUMPY", "{}: not .npy", path.display());
    let (hlen, start) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let hdr = std::str::from_utf8(&b[start..start + hlen]).unwrap();
    assert!(!hdr.contains("'fortran_order': True"), "fortran order");
    let descr = hdr
        .split("'descr':")
        .nth(1)
        .unwrap()
        .split('\'')
        .nth(1)
        .unwrap()
        .to_string();
    let shape_s = hdr.split("'shape':").nth(1).unwrap();
    let shape_s = &shape_s[shape_s.find('(').unwrap() + 1..shape_s.find(')').unwrap()];
    let shape = shape_s
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    (descr, shape, b[start + hlen..].to_vec())
}

fn load_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let (d, shape, p) = npy_read(path);
    assert_eq!(d, "<f4", "{}", path.display());
    (
        shape,
        p.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn load_codes(path: &Path) -> (Vec<usize>, Vec<u32>) {
    let (d, shape, p) = npy_read(path);
    let v = match d.as_str() {
        "<i4" => p
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
            .collect(),
        "<i8" => p
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32)
            .collect(),
        other => panic!("{}: codes dtype {other}", path.display()),
    };
    (shape, v)
}

/// The `cortiq quantize-gptq --hessians` cache (`CMFHESS1`, the layout of
/// cortiq-cli's `save_hessians`): identical Hessians are stored once under
/// all their names, each as its upper triangle.
fn save_hessians(
    path: &Path,
    hess: &std::collections::HashMap<String, cortiq_engine::gptq_capture::HessianAcc>,
) {
    use std::io::Write;
    let mut names: Vec<&String> = hess.keys().collect();
    names.sort();
    let mut uniq: Vec<(Vec<&String>, &cortiq_engine::gptq_capture::HessianAcc)> = Vec::new();
    for n in names {
        let a = &hess[n];
        if let Some(u) = uniq.iter_mut().find(|(_, b)| {
            b.cols == a.cols && b.count == a.count && b.sumsq == a.sumsq && b.h == a.h
        }) {
            u.0.push(n);
        } else {
            uniq.push((vec![n], a));
        }
    }
    let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(path).unwrap());
    f.write_all(b"CMFHESS1").unwrap();
    f.write_all(&(uniq.len() as u64).to_le_bytes()).unwrap();
    for (ns, a) in &uniq {
        f.write_all(&(ns.len() as u32).to_le_bytes()).unwrap();
        for n in ns {
            f.write_all(&(n.len() as u32).to_le_bytes()).unwrap();
            f.write_all(n.as_bytes()).unwrap();
        }
        f.write_all(&(a.cols as u64).to_le_bytes()).unwrap();
        f.write_all(&(a.count as u64).to_le_bytes()).unwrap();
        f.write_all(&(a.h.len() as u64).to_le_bytes()).unwrap();
        for v in &a.sumsq {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        let n = a.cols;
        if a.h.len() == n * n {
            for i in 0..n {
                for v in &a.h[i * n + i..i * n + n] {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
            }
        }
    }
    f.flush().unwrap();
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    let out = PathBuf::from(arg(&args, "--out").expect("--out"));
    match cmd {
        "decode" => {
            let wav = std::fs::read(arg(&args, "--wav").expect("--wav")).unwrap();
            let w = mimo_audio::decode_wav(&wav).unwrap();
            let flat: Vec<f32> = w.channels.concat();
            save_f32(&out, &[w.channels.len(), w.frames()], &flat);
            println!(
                "rate {} channels {} frames {}",
                w.sample_rate,
                w.channels.len(),
                w.frames()
            );
        }
        "frontend" => {
            std::fs::create_dir_all(&out).unwrap();
            let wav = std::fs::read(arg(&args, "--wav").expect("--wav")).unwrap();
            let t0 = Instant::now();
            let w = mimo_audio::decode_wav(&wav).unwrap();
            save_f32(
                &out.join("dec.npy"),
                &[w.channels.len(), w.frames()],
                &w.channels.concat(),
            );
            let chans: Vec<Vec<f32>> = w
                .channels
                .iter()
                .map(|c| mimo_audio::resample_sinc(c, w.sample_rate, mimo_audio::SAMPLE_RATE))
                .collect();
            save_f32(
                &out.join("chan24k.npy"),
                &[chans.len(), chans[0].len()],
                &chans.concat(),
            );
            let mono = mimo_audio::wav_to_mono_24k(&w).unwrap();
            save_f32(&out.join("wave24k.npy"), &[mono.len()], &mono);
            let pool = cortiq_engine::pool::Pool::from_env();
            let (mel, m) = mimo_audio::log_mel(&mono, pool.as_deref()).unwrap();
            save_f32(&out.join("mel.npy"), &[m, mimo_audio::N_MELS], &mel);
            println!(
                "rate {} channels {} frames {} -> {} samples, {m} mel frames, K {} ({:.3}s)",
                w.sample_rate,
                w.channels.len(),
                w.frames(),
                mono.len(),
                mimo_audio::audio_token_count(m, 4),
                t0.elapsed().as_secs_f64()
            );
        }
        "tower" => {
            std::fs::create_dir_all(&out).unwrap();
            let src = PathBuf::from(arg(&args, "--src").expect("--src"));
            let t0 = Instant::now();
            let audio = if src.extension().is_some_and(|e| e == "cmf") {
                let model = Arc::new(cortiq_core::CmfModel::open(&src).expect("open cmf"));
                MimoAudio::from_model(&model).expect("load towers")
            } else {
                MimoAudio::from_hf_dir(&src).expect("load towers")
            };
            let t_load = t0.elapsed().as_secs_f64();
            let (mel, m) = if let Some(mp) = arg(&args, "--mel") {
                let (shape, v) = load_f32(Path::new(&mp));
                assert_eq!(shape[1], mimo_audio::N_MELS);
                (v, shape[0])
            } else {
                let wav = std::fs::read(arg(&args, "--wav").expect("--wav or --mel")).unwrap();
                audio.wav_to_mel(&wav).unwrap()
            };
            let t1 = Instant::now();
            let feats = audio.features(&mel, m).unwrap();
            let t_feats = t1.elapsed().as_secs_f64();
            let d = audio.tokenizer.cfg.d_model;
            let rows = feats.len() / d;
            save_f32(&out.join("feats.npy"), &[rows, d], &feats);
            let t2 = Instant::now();
            let exact = audio.tokenizer.quantize(&feats, rows, false, audio.pool());
            let t_rvq = t2.elapsed().as_secs_f64();
            let rounded = audio.tokenizer.quantize(&feats, rows, true, audio.pool());
            let levels = exact.len() / rows;
            save_i32(&out.join("codes_exact.npy"), &[rows, levels], &exact);
            save_i32(&out.join("codes_bf16books.npy"), &[rows, levels], &rounded);
            let own = mimo_audio::AudioCodes {
                frames: rows,
                levels,
                codes: if audio.bf16_codebooks {
                    rounded.clone()
                } else {
                    exact.clone()
                },
            };
            let t3 = Instant::now();
            let emb_own = audio.embed_codes(&own).unwrap();
            let t_enc = t3.elapsed().as_secs_f64();
            save_f32(
                &out.join("embeds_own.npy"),
                &[emb_own.n_tokens, emb_own.dim],
                &emb_own.rows,
            );
            let fixed = match arg(&args, "--codes") {
                Some(cp) => {
                    let (shape, v) = load_codes(Path::new(&cp));
                    mimo_audio::AudioCodes {
                        frames: shape[0],
                        levels: shape[1],
                        codes: v,
                    }
                }
                None => own.clone(),
            };
            let emb = audio.embed_codes(&fixed).unwrap();
            save_f32(&out.join("embeds.npy"), &[emb.n_tokens, emb.dim], &emb.rows);
            let k = mimo_audio::audio_token_count(m, audio.encoder.cfg.group);
            let meta = serde_json::json!({
                "src": src.display().to_string(),
                "mel_frames": m,
                "segments": mimo_audio::segment_lengths(m),
                "codes": rows,
                "placeholder_count_K": k,
                "embed_rows_own": emb_own.n_tokens,
                "embed_rows_fixed": emb.n_tokens,
                "load_s": t_load,
                "features_s": t_feats,
                "rvq_s": t_rvq,
                "encoder_s": t_enc,
                "bf16_codebooks_default": audio.bf16_codebooks,
                "threads": cortiq_engine::pool::Pool::effective_threads(),
            });
            std::fs::write(
                out.join("tower.json"),
                serde_json::to_string_pretty(&meta).unwrap(),
            )
            .unwrap();
            println!("{meta}");
            assert_eq!(emb_own.n_tokens, k, "placeholder count != encoder rows");
        }
        "calib" => {
            let src = PathBuf::from(arg(&args, "--src").expect("--src"));
            let model = Arc::new(cortiq_core::CmfModel::open(&src).expect("open cmf"));
            let audio = MimoAudio::from_model(&model).expect("load towers");
            let dir = PathBuf::from(arg(&args, "--wav-dir").expect("--wav-dir"));
            let mut wavs: Vec<PathBuf> = std::fs::read_dir(&dir)
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "wav"))
                .collect();
            wavs.sort();
            let t0 = Instant::now();
            cortiq_engine::gptq_capture::begin(true);
            let mut frames = 0usize;
            for w in &wavs {
                let emb = audio.embed_wav(&std::fs::read(w).unwrap()).unwrap();
                frames += emb.n_tokens;
                eprintln!(
                    "  {} -> {} rows ({:.0}s)",
                    w.display(),
                    emb.n_tokens,
                    t0.elapsed().as_secs_f64()
                );
            }
            let hess = cortiq_engine::gptq_capture::end();
            save_hessians(&out, &hess);
            println!(
                "{} clips, {frames} LLM rows, {} linears -> {} ({:.0}s)",
                wavs.len(),
                hess.len(),
                out.display(),
                t0.elapsed().as_secs_f64()
            );
        }
        _ => {
            eprintln!(
                "usage: mimo_audio_dump (decode|frontend|tower|calib) --out ... (see the source header)"
            );
            std::process::exit(2);
        }
    }
}
