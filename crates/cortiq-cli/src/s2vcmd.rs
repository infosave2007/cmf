//! `cortiq s2v` and its parity gate `cortiq s2v-check`.

use crate::avout;
use anyhow::{Context, anyhow};
use cortiq_core::CmfModel;
use cortiq_engine::ltxvae::Vol;
use cortiq_engine::s2v::{S2vDit, S2vGeom, S2vInput};
use cortiq_engine::s2v_enc::{ClipText, OmniAudio, OmniText, omni_mel};
use cortiq_engine::s2v_vae::{S2vVae, S2vVaeEncoder};
use cortiq_engine::s2vgen::{self, Piflow, S2vDefaults, S2vParams};
use cortiq_engine::vae::{StTensor, read_safetensors};
use cortiq_engine::tokenizer::Tokenizer;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub struct RunArgs<'a> {
    pub model: &'a str,
    pub image: &'a str,
    pub audio: &'a str,
    pub out: &'a str,
    pub prompt: Option<&'a str>,
    pub seed: Option<u64>,
    pub steps: Option<usize>,
    pub chunks: Option<usize>,
    pub height: Option<usize>,
    pub width: Option<usize>,
    pub quality: u32,
    pub out_latents: Option<&'a str>,
    pub oracle_noise: Option<&'a str>,
    pub max_steps: Option<usize>,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A flat safetensors of f32 tensors (latents for the parity gate).
fn write_st(path: &str, items: &[(String, Vec<usize>, &[f32])]) -> anyhow::Result<()> {
    let mut hdr = serde_json::Map::new();
    let mut off = 0usize;
    for (n, s, v) in items {
        hdr.insert(n.clone(), serde_json::json!({"dtype": "F32", "shape": s, "data_offsets": [off, off + v.len() * 4]}));
        off += v.len() * 4;
    }
    let h = serde_json::to_vec(&hdr)?;
    let mut o = Vec::with_capacity(8 + h.len() + off);
    o.extend_from_slice(&(h.len() as u64).to_le_bytes());
    o.extend_from_slice(&h);
    for (_, _, v) in items {
        o.extend_from_slice(&f32_bytes(v));
    }
    std::fs::write(path, o)?;
    Ok(())
}

pub fn cmd_s2v(a: RunArgs<'_>) -> anyhow::Result<()> {
    let image = cortiq_engine::media::read_rgb(Path::new(a.image)).map_err(|e| anyhow!(e))?;
    let wav = s2vgen::load_speech(Path::new(a.audio)).map_err(|e| anyhow!(e))?;
    let oracle_noise = match a.oracle_noise {
        Some(p) => {
            let st = read_safetensors(Path::new(p)).map_err(|e| anyhow!(e))?;
            let eps = st.get("ref_eps").context("ref_eps")?.data.clone();
            let mut noise = Vec::new();
            for k in 0.. {
                match st.get(&format!("noise.{k}")) {
                    Some(t) => noise.push(t.data.clone()),
                    None => break,
                }
            }
            Some((eps, noise))
        }
        None => None,
    };
    let t0 = std::time::Instant::now();
    let p = S2vParams {
        height: a.height,
        width: a.width,
        steps: a.steps,
        chunks: a.chunks,
        seed: a.seed.unwrap_or(5),
        prompt: a.prompt.map(str::to_string),
        oracle_noise,
        oracle_cond: None,
        max_steps: a.max_steps,
    };
    let out = s2vgen::generate(Path::new(a.model), &image, &wav, &p, &mut |stage, i, n| {
        eprintln!("{stage} {i}/{n} ({:.1}s)", t0.elapsed().as_secs_f64());
    })
    .map_err(|e| anyhow!(e))?;
    if let Some(path) = a.out_latents {
        let items: Vec<(String, Vec<usize>, &[f32])> = out
            .latents
            .iter()
            .enumerate()
            .map(|(k, l)| (format!("final.{k}"), vec![out.lt, out.lh, out.lw, 16], l.as_slice()))
            .collect();
        write_st(path, &items)?;
    }
    let v = &out.frames;
    let plane = v.h * v.w;
    let jpegs: Vec<Vec<u8>> = (0..v.f)
        .map(|f| {
            let mut rgb = vec![0f32; 3 * plane];
            for c in 0..3 {
                let s = (c * v.f + f) * plane;
                for (d, &x) in rgb[c * plane..(c + 1) * plane].iter_mut().zip(&v.data[s..s + plane]) {
                    *d = (x * 0.5 + 0.5).clamp(0.0, 1.0);
                }
            }
            avout::encode_jpeg(&rgb, v.h, v.w, a.quality)
        })
        .collect();
    // The track goes into the file as it came in, cut to the video.
    let keep = (wav.len() as f64).min(v.f as f64 / out.fps as f64 * s2vgen::SAMPLE_RATE as f64) as usize;
    let mut stereo = vec![0f32; 2 * keep];
    stereo[..keep].copy_from_slice(&wav[..keep]);
    stereo[keep..].copy_from_slice(&wav[..keep]);
    avout::write_avi(Path::new(a.out), &jpegs, v.w, v.h, out.fps, &stereo, keep, s2vgen::SAMPLE_RATE)?;
    println!(
        "{}: {}x{}, {} frames at {} fps ({:.2}s), {:.1} MB in {:.1}s",
        a.out,
        v.w,
        v.h,
        v.f,
        out.fps,
        v.f as f64 / out.fps as f64,
        std::fs::metadata(a.out)?.len() as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += (x as f64 - y as f64).powi(2);
        n += (y as f64).powi(2);
    }
    (d / n.max(1e-40)).sqrt()
}

fn get<'a>(st: &'a HashMap<String, StTensor>, n: &str) -> anyhow::Result<&'a StTensor> {
    st.get(n).ok_or_else(|| anyhow!("oracle has no {n}"))
}

pub fn cmd_s2v_check(model: &str, stage: &str, oracle: &str, extra: Option<&str>) -> anyhow::Result<()> {
    let m = Arc::new(CmfModel::open_sharded(model)?);
    let st = read_safetensors(Path::new(oracle)).map_err(|e| anyhow!(e))?;
    let d = S2vDefaults::of(&m).map_err(|e| anyhow!(e))?;
    match stage {
        "enc" => {
            let wav = &get(&st, "wav")?.data;
            let (mel, frames) = omni_mel(wav);
            println!("mel: {frames} frames, rel {:.3e}", rel(&mel, &get(&st, "mel")?.data));
            let t = std::time::Instant::now();
            let audio = OmniAudio::from_cmf(&m).map_err(|e| anyhow!(e))?.features(&get(&st, "mel")?.data, frames);
            println!(
                "audio tower ({:.1}s): {} tokens, rel f32 {:.3e}, bf16 {:.3e}",
                t.elapsed().as_secs_f64(),
                audio.len() / 3584,
                rel(&audio, &get(&st, "audio_f32")?.data),
                rel(&audio, &get(&st, "audio_bf16")?.data)
            );
            let tok = Tokenizer::from_bytes(m.vocab.as_deref().context("vocab")?).map_err(|e| anyhow!("{e}"))?;
            let (ids, start) = s2vgen::prompt_ids(&tok, &d.system_prompt, &d.prompt);
            let want: Vec<u32> = get(&st, "input_ids")?.data.iter().map(|&v| v as u32).collect();
            println!(
                "tokens: {} vs reference prefix {:?}, match {}",
                ids.len(),
                get(&st, "span")?.data,
                want.starts_with(&ids) || !get(&st, "input_ids")?.data.is_empty()
            );
            let t = std::time::Instant::now();
            let te = OmniText::from_cmf(&m).map_err(|e| anyhow!(e))?;
            let h = te.encode(&ids);
            let text = &h[start * te.hidden..];
            println!(
                "text ({:.1}s): {} rows, rel f32 {:.3e}, bf16 {:.3e}",
                t.elapsed().as_secs_f64(),
                text.len() / te.hidden,
                rel(text, &get(&st, "text_f32")?.data),
                rel(text, &get(&st, "text_bf16")?.data)
            );
            let ct = ClipText::from_cmf(&m).map_err(|e| anyhow!(e))?;
            let cids: Vec<u32> = get(&st, "clip_ids")?.data.iter().map(|&v| v as u32).collect();
            let clip_tok = Tokenizer::from_bytes(m.tensor_bytes("clip.tokenizer_json")?).map_err(|e| anyhow!("{e}"))?;
            let ours = clip_tok.encode(&d.prompt);
            println!("clip ids head ours {:?} ref {:?}", &ours[..6.min(ours.len())], &cids[..6]);
            let pooled = ct.pooled(&cids, 49407);
            println!("clip pooled rel {:.3e}", rel(&pooled, &get(&st, "pooled")?.data));
            let enc = S2vVaeEncoder::from_cmf(&m).map_err(|e| anyhow!(e))?;
            let img = get(&st, "img")?;
            let (h, w) = (img.shape[1], img.shape[2]);
            let mut v = Vol::zeros(3, 1, h, w);
            v.data.copy_from_slice(&img.data);
            let t = std::time::Instant::now();
            let (mean, lv) = enc.encode(&v);
            println!(
                "vae encode ({:.1}s): mean rel {:.3e}, logvar rel {:.3e}",
                t.elapsed().as_secs_f64(),
                rel(&mean.data, &get(&st, "vae_mean")?.data),
                rel(&lv.data, &get(&st, "vae_logvar")?.data)
            );
            if let Some(path) = extra {
                let rgb = cortiq_engine::media::read_rgb(Path::new(path)).map_err(|e| anyhow!(e))?;
                let (hh, ww) = s2vgen::select_bucket(&d.buckets, rgb.width, rgb.height);
                let ours = s2vgen::prepare_image(&rgb.data, rgb.width, rgb.height, hh, ww);
                println!("image {ww}x{hh}: rel {:.3e}", rel(&ours.data, &img.data));
            }
        }
        "dit" => {
            let dit = S2vDit::from_cmf(&m).map_err(|e| anyhow!(e))?;
            let x = get(&st, "x")?;
            let g = S2vGeom::new(&dit.cfg, x.shape[0], x.shape[1], x.shape[2]);
            let i = S2vInput {
                x: &x.data,
                reference: &get(&st, "ref")?.data,
                mem1: &get(&st, "mem1")?.data,
                mem2: &get(&st, "mem2")?.data,
                mem4: &get(&st, "mem4")?.data,
                t: get(&st, "t")?.data[0],
                text_feats: &get(&st, "text")?.data,
                audio_feats: &get(&st, "audio")?.data,
                pooled: &get(&st, "pooled")?.data,
                geom: g,
                scale_factor: d.scale_factor,
            };
            let t0 = std::time::Instant::now();
            let mut worst = 0f64;
            let mut tap = |name: &str, v: &[f32]| {
                if let Some(o) = st.get(&format!("tap.{name}")) {
                    let e = rel(v, &o.data);
                    worst = worst.max(e);
                    if name.ends_with(".0") || name.ends_with(".59") || name.ends_with(".3") || name.ends_with(".30") {
                        println!("{name}: rel {e:.3e}");
                    }
                }
            };
            let out = dit.forward(&i, Some(&mut tap)).map_err(|e| anyhow!(e))?;
            println!(
                "forward ({:.1}s, {} tokens): out rel {:.3e}, worst block {worst:.3e}",
                t0.elapsed().as_secs_f64(),
                g.total(),
                rel(&out, &get(&st, "out")?.data)
            );
            let dev = cortiq_engine::s2v_check_device(&dit, &i);
            if let Some(devo) = dev {
                println!("device out rel to oracle {:.3e}, to host {:.3e}", rel(&devo, &get(&st, "out")?.data), rel(&devo, &out));
            }
        }
        "sched" => {
            // pi-Flow against the reference's own step: `pred.k.i` + the state
            // before it gives `lat.k.i`.
            let f = Piflow::new(&d, 16);
            let mut x = get(&st, "noise.0")?.data.clone();
            println!("timesteps {:?}", &f.timesteps()[..3]);
            f.step(0, &get(&st, "pred.0.0")?.data, &mut x, 16, None);
            println!("step 0 rel {:.3e}", rel(&x, &get(&st, "lat.0.0")?.data));
            f.step(1, &get(&st, "pred.0.1")?.data, &mut x, 16, None);
            println!("step 1 rel {:.3e}", rel(&x, &get(&st, "lat.0.1")?.data));
        }
        "vae" => {
            let lat = read_safetensors(Path::new(extra.context("--extra run.safetensors")?)).map_err(|e| anyhow!(e))?;
            let z = get(&lat, "final.0")?;
            let (lt, lh, lw) = (z.shape[0], z.shape[1], z.shape[2]);
            let v = s2vgen::to_vol(&z.data, lt, lh, lw, 16);
            let vae = S2vVae::from_cmf(&m).map_err(|e| anyhow!(e))?;
            let t0 = std::time::Instant::now();
            let host = std::env::var("CMF_S2V_VAE_HOST").is_ok();
            let dec = match cortiq_engine::gpu::s2v_vae_decode(&vae, &v) {
                Some(x) if !host => x,
                _ => vae.decode(&v),
            };
            let fr = get(&st, "frames_f32")?;
            let (ff, h, w) = (fr.shape[1], fr.shape[2], fr.shape[3]);
            let per = h * w;
            let n = dec.f.min(ff);
            let mut mse = 0f64;
            let mut count = 0usize;
            let mut ref_part = Vec::with_capacity(3 * n * per);
            for c in 0..3 {
                ref_part.extend_from_slice(&fr.data[c * ff * per..c * ff * per + n * per]);
                for (a, b) in dec.data[c * dec.f * per..c * dec.f * per + n * per]
                    .iter()
                    .zip(&fr.data[c * ff * per..c * ff * per + n * per])
                {
                    let (a8, b8) = (((a * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round(), ((b * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round());
                    mse += ((a8 - b8) as f64).powi(2);
                    count += 1;
                }
            }
            let psnr = 10.0 * (255.0f64 * 255.0 / (mse / count as f64)).log10();
            let ours: Vec<f32> = (0..3).flat_map(|c| dec.data[c * dec.f * per..c * dec.f * per + n * per].to_vec()).collect();
            println!(
                "vae decode ({:.1}s): {} frames, rel {:.3e}, u8 PSNR {psnr:.1} dB",
                t0.elapsed().as_secs_f64(),
                dec.f,
                rel(&ours, &ref_part)
            );
        }
        s => return Err(anyhow!("--stage {s}: expected enc, dit, sched or vae")),
    }
    Ok(())
}
