//! MiMo audio towers against the oracle dumps of `tools/mimo_audio_ref.py`.
//!
//! Skipped unless `MIMO_AUDIO_FIXTURES` names a directory laid out as the
//! gate script writes it:
//!
//! ```text
//! $MIMO_AUDIO_FIXTURES/wavs/*.wav            test clips (oracle `wavs` + the Mac `say`/afconvert set)
//! $MIMO_AUDIO_FIXTURES/ref/<stem>/dec.npy    oracle numpy decode, [C, N]
//! $MIMO_AUDIO_FIXTURES/ref/<stem>/wave24k_f64.npy, mel_f64.npy     oracle frontend (fp64 twin)
//! $MIMO_AUDIO_FIXTURES/ref/<stem>/feats.npy, codes_exact.npy, embeds.npy (tower oracle, optional)
//! ```
//!
//! The tower part also needs `MIMO_AUDIO_SRC` (HF checkpoint dir or a
//! companion `.cmf`). Run with `CMF_GPU=0` for the exact CPU leg.

use cortiq_engine::mimo_audio::{self, MimoAudio};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn npy(path: &Path) -> Option<(Vec<usize>, Vec<u8>, String)> {
    let b = std::fs::read(path).ok()?;
    assert_eq!(&b[..6], b"\x93NUMPY");
    let (hlen, start) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let hdr = std::str::from_utf8(&b[start..start + hlen]).unwrap();
    let descr = hdr
        .split("'descr':")
        .nth(1)?
        .split('\'')
        .nth(1)?
        .to_string();
    let s = hdr.split("'shape':").nth(1)?;
    let s = &s[s.find('(')? + 1..s.find(')')?];
    let shape = s
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(|x| x.parse().unwrap())
        .collect();
    Some((shape, b[start + hlen..].to_vec(), descr))
}

fn f32s(path: &Path) -> Option<(Vec<usize>, Vec<f32>)> {
    let (shape, p, d) = npy(path)?;
    assert_eq!(d, "<f4", "{}", path.display());
    Some((
        shape,
        p.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    ))
}

fn i32s(path: &Path) -> Option<(Vec<usize>, Vec<u32>)> {
    let (shape, p, d) = npy(path)?;
    let v = match d.as_str() {
        "<i4" => p
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
            .collect(),
        "<i8" => p
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32)
            .collect(),
        other => panic!("{}: {other}", path.display()),
    };
    Some((shape, v))
}

fn fixtures() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("MIMO_AUDIO_FIXTURES").ok()?);
    d.join("wavs").is_dir().then_some(d)
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

fn peak(a: &[f32]) -> f64 {
    a.iter().map(|x| (*x as f64).abs()).fold(0.0, f64::max)
}

fn wavs(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir.join("wavs"))
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    v.sort();
    v
}

/// G8.1: every WAV decodes to the oracle's numpy samples, bit for bit.
#[test]
fn g8_1_wav_decode_matches_numpy() {
    let Some(dir) = fixtures() else { return };
    let mut n = 0;
    for wav in wavs(&dir) {
        let stem = wav.file_stem().unwrap().to_str().unwrap();
        let Some((shape, want)) = f32s(&dir.join("ref").join(stem).join("dec.npy")) else {
            continue;
        };
        let got = mimo_audio::decode_wav(&std::fs::read(&wav).unwrap()).unwrap();
        assert_eq!(vec![got.channels.len(), got.frames()], shape, "{stem}");
        let flat = got.channels.concat();
        assert_eq!(max_abs(&flat, &want), 0.0, "{stem}");
        n += 1;
    }
    assert!(n > 0, "no decode fixtures");
}

/// G8.2 and G8.3: resampled waveform within 1e-6 of peak and log-mel within
/// 1e-4 of the oracle's fp64 twin of torchaudio's math; M = 1 + N/240.
#[test]
fn g8_2_g8_3_frontend_matches_torchaudio_port() {
    let Some(dir) = fixtures() else { return };
    let mut n = 0;
    for wav in wavs(&dir) {
        let stem = wav.file_stem().unwrap().to_str().unwrap();
        let r = dir.join("ref").join(stem);
        let (Some((_, w_ref)), Some((mshape, m_ref))) = (
            f32s(&r.join("wave24k_f64.npy")),
            f32s(&r.join("mel_f64.npy")),
        ) else {
            continue;
        };
        let w = mimo_audio::decode_wav(&std::fs::read(&wav).unwrap()).unwrap();
        let mono = mimo_audio::wav_to_mono_24k(&w).unwrap();
        assert_eq!(mono.len(), w_ref.len(), "{stem}: resampled length");
        assert_eq!(
            mono.len(),
            mimo_audio::resampled_len(w.frames(), w.sample_rate, mimo_audio::SAMPLE_RATE),
            "{stem}"
        );
        let e = max_abs(&mono, &w_ref) / peak(&w_ref);
        assert!(e <= 1e-6, "{stem}: resample max|Δ|/peak {e:.3e}");
        let (mel, m) = mimo_audio::log_mel(&mono, None).unwrap();
        assert_eq!(m, mimo_audio::mel_frames(mono.len()), "{stem}");
        assert_eq!(vec![m, mimo_audio::N_MELS], mshape, "{stem}");
        let e = max_abs(&mel, &m_ref);
        assert!(e <= 1e-4, "{stem}: log-mel max|Δ| {e:.3e}");
        n += 1;
    }
    assert!(n > 0, "no frontend fixtures");
}

/// G9.1 / G9.2 / G10.1 on the clips that have tower dumps.
#[test]
fn g9_g10_towers_match_hf_fp32() {
    let Some(dir) = fixtures() else { return };
    let Ok(src) = std::env::var("MIMO_AUDIO_SRC") else {
        return;
    };
    let src = PathBuf::from(src);
    let audio = if src.extension().is_some_and(|e| e == "cmf") {
        MimoAudio::from_model(&Arc::new(cortiq_core::CmfModel::open(&src).unwrap())).unwrap()
    } else {
        MimoAudio::from_hf_dir(&src).unwrap()
    };
    let mut n = 0;
    for entry in std::fs::read_dir(dir.join("ref")).unwrap() {
        let r = entry.unwrap().path();
        let (Some((ms, mel)), Some((fs, f_ref)), Some((_, c_ref))) = (
            f32s(&r.join("mel.npy")),
            f32s(&r.join("feats.npy")),
            i32s(&r.join("codes_exact.npy")),
        ) else {
            continue;
        };
        if ms[0] > mimo_audio::SEGMENT_FRAMES {
            continue; // the long clip is gated by the dump script (G9.3)
        }
        let feats = audio.features(&mel, ms[0]).unwrap();
        assert_eq!(feats.len(), fs[0] * fs[1]);
        let rel = max_abs(&feats, &f_ref) / peak(&f_ref);
        let d = fs[1];
        let min_cos = feats
            .chunks_exact(d)
            .zip(f_ref.chunks_exact(d))
            .map(|(a, b)| {
                let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
                for (x, y) in a.iter().zip(b) {
                    ab += *x as f64 * *y as f64;
                    aa += *x as f64 * *x as f64;
                    bb += *y as f64 * *y as f64;
                }
                ab / (aa.sqrt() * bb.sqrt())
            })
            .fold(1.0, f64::min);
        assert!(
            rel <= 1e-3 && min_cos >= 0.9999,
            "{}: G9.1 rel {rel:.3e} cos {min_cos:.6}",
            r.display()
        );
        let codes = audio.tokenizer.quantize(&feats, fs[0], false, audio.pool());
        let lv = c_ref.len() / fs[0];
        let flips = codes.iter().zip(&c_ref).filter(|(a, b)| a != b).count();
        let first3 = (0..fs[0]).all(|i| (0..3).all(|l| codes[i * lv + l] == c_ref[i * lv + l]));
        assert!(
            first3 && flips as f64 <= 0.002 * codes.len() as f64,
            "{}: G9.2 flips {flips}/{} levels0-2 identical {first3}",
            r.display(),
            codes.len()
        );
        if let Some((es, e_ref)) = f32s(&r.join("embeds.npy")) {
            let got = audio
                .embed_codes(&mimo_audio::AudioCodes {
                    frames: fs[0],
                    levels: lv,
                    codes: c_ref.clone(),
                })
                .unwrap();
            assert_eq!(vec![got.n_tokens, got.dim], es);
            let rel = max_abs(&got.rows, &e_ref) / peak(&e_ref);
            assert!(rel <= 1e-3, "{}: G10.1 rel {rel:.3e}", r.display());
        }
        n += 1;
    }
    assert!(n > 0, "no tower fixtures");
}
