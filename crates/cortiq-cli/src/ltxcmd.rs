//! Run the LTX-2.5 stacks that are ported: today the video VAE decoder.
//!
//! `ltx-decode` reads latents from a safetensors file (`latent`, shaped
//! `[1, 128, F, H, W]` or `[128, F, H, W]`), decodes them through the
//! `vvae.*` half of an `ltx-2.5-av` container, and writes the frames as
//! PPM stills and/or a safetensors tensor. With `--gate` it also compares
//! against a reference `frames` tensor in the same file and reports the
//! difference — the numeric contract the port is held to.

use anyhow::{Context, anyhow};
use cortiq_core::CmfModel;
use cortiq_engine::ltxdit::{LtxDit, StreamInput};
use cortiq_engine::ltxvae::{ConvVaeDecoder, Vol};
use cortiq_engine::pool::Pool;
use memmap2::Mmap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

/// Minimal safetensors reader: (name → (dtype, shape, bytes)).
struct St {
    map: Mmap,
    dir: serde_json::Value,
    base: usize,
}

impl St {
    fn open(p: &Path) -> anyhow::Result<St> {
        let f = std::fs::File::open(p).with_context(|| p.display().to_string())?;
        // SAFETY: read-only view of a file this process does not write.
        let map = unsafe { Mmap::map(&f) }?;
        let hlen = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let dir: serde_json::Value = serde_json::from_slice(&map[8..8 + hlen])?;
        Ok(St { map, dir, base: 8 + hlen })
    }
    fn get(&self, name: &str) -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
        let e = self
            .dir
            .get(name)
            .ok_or_else(|| anyhow!("tensor '{name}' not in file"))?;
        let shape: Vec<usize> = e["shape"]
            .as_array()
            .ok_or_else(|| anyhow!("{name}: shape"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let off = e["data_offsets"].as_array().ok_or_else(|| anyhow!("{name}: offsets"))?;
        let (s, t) = (
            off[0].as_u64().unwrap_or(0) as usize + self.base,
            off[1].as_u64().unwrap_or(0) as usize + self.base,
        );
        let raw = &self.map[s..t];
        let vals = match e["dtype"].as_str().unwrap_or("F32") {
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| cortiq_core::quant::f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16))
                .collect(),
            d => return Err(anyhow!("{name}: dtype {d}")),
        };
        Ok((shape, vals))
    }
}

fn write_ppm(path: &Path, frame: &[f32], h: usize, w: usize) -> anyhow::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "P6\n{w} {h}\n255")?;
    let mut buf = Vec::with_capacity(h * w * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                // the decoder emits [-1, 1]
                let v = frame[(c * h + y) * w + x];
                buf.push((((v.clamp(-1.0, 1.0) + 1.0) * 0.5) * 255.0).round() as u8);
            }
        }
    }
    f.write_all(&buf)?;
    Ok(())
}

pub struct DecodeArgs<'a> {
    pub model: &'a str,
    pub latent: &'a str,
    pub out_dir: Option<&'a str>,
    pub out_tensors: Option<&'a str>,
    pub gate: bool,
    pub dump_stages: Option<&'a str>,
}

pub fn cmd_ltx_decode(a: DecodeArgs<'_>) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(a.model)?);
    let pool = Pool::from_env();
    let t0 = std::time::Instant::now();
    let dec = ConvVaeDecoder::from_cmf(&model, pool.as_deref()).map_err(|e| anyhow!(e))?;
    eprintln!("vae decoder loaded in {:.1}s", t0.elapsed().as_secs_f64());
    let st = St::open(Path::new(a.latent))?;
    let (shape, vals) = st.get("latent")?;
    let s: Vec<usize> = if shape.len() == 5 { shape[1..].to_vec() } else { shape.clone() };
    anyhow::ensure!(s.len() == 4, "latent must be [C,F,H,W] or [1,C,F,H,W], got {shape:?}");
    let lat = Vol { c: s[0], f: s[1], h: s[2], w: s[3], data: vals };
    eprintln!("latent [{}, {}, {}, {}]", lat.c, lat.f, lat.h, lat.w);
    let t1 = std::time::Instant::now();
    let out = dec.decode(&lat, pool.as_deref());
    let secs = t1.elapsed().as_secs_f64();
    println!(
        "decoded [{}, {}, {}, {}] in {:.1}s ({:.2} s/frame)",
        out.c,
        out.f,
        out.h,
        out.w,
        secs,
        secs / out.f as f64
    );
    if a.gate {
        // compare every stage the reference recorded, in pipeline order
        let mut stages: Vec<(String, (Vec<usize>, Vec<f32>))> = Vec::new();
        for name in ["after_conv_in"]
            .iter()
            .map(|s| s.to_string())
            .chain((0..16).map(|i| format!("after_block_{i}")))
            .chain(["after_conv_out".to_string(), "frames".to_string()])
        {
            if let Ok(t) = st.get(&name) {
                stages.push((name, t));
            }
        }
        let mut ours: std::collections::HashMap<String, Vol> = std::collections::HashMap::new();
        let _ = dec.decode_traced(&lat, pool.as_deref(), &mut |n, v| {
            ours.insert(n.to_string(), v.clone());
        });
        let mut first_bad: Option<String> = None;
        for (name, (rs, rv)) in &stages {
            let Some(o) = ours.get(name) else { continue };
            let r: Vec<usize> = if rs.len() == 5 { rs[1..].to_vec() } else { rs.clone() };
            let shape_ok = r == vec![o.c, o.f, o.h, o.w];
            let (mut worst, mut sum, mut rsum) = (0f64, 0f64, 0f64);
            if shape_ok {
                for (a, b) in o.data.iter().zip(rv) {
                    let d = (*a as f64 - *b as f64).abs();
                    worst = worst.max(d);
                    sum += d * d;
                    rsum += (*b as f64) * (*b as f64);
                }
            }
            let n = o.data.len() as f64;
            let ratio = (sum / rsum.max(1e-30)).sqrt();
            println!(
                "{name:<16} {} ours [{},{},{},{}] ref {r:?}  worst {:.2e} rel {:.2e}",
                if shape_ok { "ok  " } else { "SHAPE" },
                o.c,
                o.f,
                o.h,
                o.w,
                worst,
                ratio
            );
            let _ = n;
            if first_bad.is_none() && (!shape_ok || ratio > 1e-3) {
                first_bad = Some(name.clone());
            }
        }
        if let Some(b) = first_bad {
            println!("first stage that diverges: {b}");
        } else {
            println!("all stages match");
        }
        return Ok(());
    }
    if let Some(p) = a.dump_stages {
        // every traced stage into one safetensors, for a side-by-side diff
        let mut names: Vec<String> = Vec::new();
        let mut shapes: Vec<[usize; 4]> = Vec::new();
        let mut datas: Vec<Vec<f32>> = Vec::new();
        let _ = dec.decode_traced(&lat, pool.as_deref(), &mut |n, v| {
            names.push(n.to_string());
            shapes.push([v.c, v.f, v.h, v.w]);
            datas.push(v.data.clone());
        });
        let mut hdr = serde_json::Map::new();
        let mut off = 0usize;
        for ((n, s), d) in names.iter().zip(&shapes).zip(&datas) {
            let len = d.len() * 4;
            hdr.insert(
                n.clone(),
                serde_json::json!({"dtype": "F32", "shape": s, "data_offsets": [off, off + len]}),
            );
            off += len;
        }
        let hs = serde_json::to_vec(&serde_json::Value::Object(hdr))?;
        let mut f = std::fs::File::create(p)?;
        f.write_all(&(hs.len() as u64).to_le_bytes())?;
        f.write_all(&hs)?;
        for d in &datas {
            let mut bytes = Vec::with_capacity(d.len() * 4);
            for v in d {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            f.write_all(&bytes)?;
        }
        println!("{} stages → {p}", names.len());
    }
    if let Some(d) = a.out_dir {
        std::fs::create_dir_all(d)?;
        for f in 0..out.f {
            let frame: Vec<f32> = (0..3 * out.h * out.w)
                .map(|i| {
                    let c = i / (out.h * out.w);
                    let r = i % (out.h * out.w);
                    out.data[((c * out.f + f) * out.h) * out.w + r]
                })
                .collect();
            write_ppm(&Path::new(d).join(format!("frame_{f:04}.ppm")), &frame, out.h, out.w)?;
        }
        println!("{} frames → {d}/frame_*.ppm", out.f);
    }
    if let Some(p) = a.out_tensors {
        let mut bytes = Vec::with_capacity(out.data.len() * 4);
        for v in &out.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let hdr = serde_json::json!({
            "frames": {"dtype": "F32", "shape": [out.c, out.f, out.h, out.w], "data_offsets": [0, bytes.len()]}
        });
        let hs = serde_json::to_vec(&hdr)?;
        let mut f = std::fs::File::create(p)?;
        f.write_all(&(hs.len() as u64).to_le_bytes())?;
        f.write_all(&hs)?;
        f.write_all(&bytes)?;
        println!("frames tensor → {p}");
    }
    Ok(())
}

// ---------------------------------------------------------------- ltx-dit

pub struct DitArgs<'a> {
    pub model: &'a str,
    pub oracle: &'a str,
    pub gate: bool,
    pub dump: Option<&'a str>,
}

/// One `[1, T, D]` (or `[1, T, 1]`) oracle tensor as a flat f32 row-major
/// buffer plus its token count.
fn oracle_2d(st: &St, name: &str) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    let (s, v) = st.get(name)?;
    let dims: Vec<usize> = s.iter().copied().filter(|&d| d > 0).collect();
    anyhow::ensure!(dims.len() >= 2, "{name}: rank {dims:?}");
    let d = *dims.last().unwrap();
    let t = dims[dims.len() - 2];
    Ok((v, t, d))
}

/// The `[1, axes, T, 2]` position grid as per-token patch midpoints —
/// `use_middle_indices_grid`, the reference's default.
fn oracle_positions(st: &St, name: &str) -> anyhow::Result<Vec<Vec<f64>>> {
    let (s, v) = st.get(name)?;
    let d: Vec<usize> = s.iter().copied().filter(|&x| x > 0).collect();
    anyhow::ensure!(d.len() == 4 && d[3] == 2, "{name}: expected [1, axes, T, 2], got {s:?}");
    let (axes, t) = (d[1], d[2]);
    Ok((0..t)
        .map(|i| {
            (0..axes)
                .map(|a| {
                    let o = (a * t + i) * 2;
                    (v[o] as f64 + v[o + 1] as f64) / 2.0
                })
                .collect()
        })
        .collect())
}

pub fn cmd_ltx_dit(a: DitArgs<'_>) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(a.model)?);
    let pool = Pool::from_env();
    let t0 = std::time::Instant::now();
    let dit = LtxDit::from_cmf(&model).map_err(|e| anyhow!(e))?;
    eprintln!("dit loaded in {:.1}s ({} blocks)", t0.elapsed().as_secs_f64(), dit.blocks());
    let st = St::open(Path::new(a.oracle))?;

    let stream = |tag: &str| -> anyhow::Result<StreamInput> {
        let (latent, tokens, _) = oracle_2d(&st, &format!("{tag}.latent"))?;
        let (timesteps, _, _) = oracle_2d(&st, &format!("{tag}.timesteps"))?;
        let (context, ctx_len, _) = oracle_2d(&st, &format!("{tag}.context"))?;
        let positions = oracle_positions(&st, &format!("{tag}.positions"))?;
        let sigma = st.get(&format!("{tag}.sigma"))?.1[0];
        let keyframes = st
            .get(&format!("{tag}.keyframes_mask"))
            .map(|(_, v)| v)
            .unwrap_or_default();
        let context_mask = st
            .get(&format!("{tag}.args.context_mask"))
            .map(|(s, v)| v[v.len() - s[s.len() - 1]..].to_vec())
            .unwrap_or_default();
        Ok(StreamInput {
            latent,
            tokens,
            timesteps,
            positions,
            context,
            ctx_len,
            context_mask,
            keyframes,
            sigma,
        })
    };
    let video = stream("v")?;
    let audio = stream("a")?;
    eprintln!(
        "video {} tokens, audio {} tokens, prompt {} tokens, sigma v={} a={}",
        video.tokens, audio.tokens, video.ctx_len, video.sigma, audio.sigma
    );

    let mut ours: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    let mut want: Vec<String> = ["v.args.x", "a.args.x", "v.out", "a.out"]
        .iter()
        .map(|s| s.to_string())
        .chain([0usize, 1, dit.blocks() - 1].iter().flat_map(|&i| {
            [format!("v.block{i}"), format!("a.block{i}")]
        }))
        .collect();
    // the block-0 dissection, when the oracle carries it
    for m in ["v", "a"] {
        for st in ["sa", "ca", "ff", if m == "v" { "a2v" } else { "v2a" }] {
            for part in ["in", "ctx", "out"] {
                want.push(format!("{m}.b0.{st}.{part}"));
            }
        }
    }
    let t1 = std::time::Instant::now();
    dit.forward_traced(&video, &audio, pool.as_deref(), &mut |n, v| {
        if want.iter().any(|w| w == n) || a.dump.is_some() {
            ours.insert(n.to_string(), v.to_vec());
        }
    });
    println!("forward in {:.1}s", t1.elapsed().as_secs_f64());

    if a.gate {
        let mut first_bad: Option<String> = None;
        for name in &want {
            let Some(o) = ours.get(name) else { continue };
            let Ok((_, r)) = st.get(name) else { continue };
            let ok = r.len() == o.len();
            let (mut worst, mut sum, mut rsum) = (0f64, 0f64, 0f64);
            if ok {
                for (x, y) in o.iter().zip(&r) {
                    let d = (*x as f64 - *y as f64).abs();
                    worst = worst.max(d);
                    sum += d * d;
                    rsum += (*y as f64) * (*y as f64);
                }
            }
            let rel = (sum / rsum.max(1e-30)).sqrt();
            println!(
                "{name:<14} {} n {} vs {}  worst {:.2e} rel {:.2e}",
                if ok { "ok  " } else { "SHAPE" },
                o.len(),
                r.len(),
                worst,
                rel
            );
            if first_bad.is_none() && (!ok || rel > 5e-3) {
                first_bad = Some(name.clone());
            }
        }
        match first_bad {
            Some(b) => println!("first stage that diverges: {b}"),
            None => println!("all stages match"),
        }
    }

    if let Some(p) = a.dump {
        let mut hdr = serde_json::Map::new();
        let mut off = 0usize;
        let mut names: Vec<&String> = ours.keys().collect();
        names.sort();
        for n in &names {
            let len = ours[*n].len() * 4;
            hdr.insert(
                (*n).clone(),
                serde_json::json!({"dtype":"F32","shape":[ours[*n].len()],"data_offsets":[off, off+len]}),
            );
            off += len;
        }
        let hs = serde_json::to_vec(&serde_json::Value::Object(hdr))?;
        let mut f = std::fs::File::create(p)?;
        f.write_all(&(hs.len() as u64).to_le_bytes())?;
        f.write_all(&hs)?;
        for n in &names {
            let mut bytes = Vec::with_capacity(ours[*n].len() * 4);
            for v in &ours[*n] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            f.write_all(&bytes)?;
        }
        println!("{} tensors → {p}", names.len());
    }
    Ok(())
}

// ------------------------------------------------------------- ltx-render

pub struct RenderArgs<'a> {
    pub model: &'a str,
    pub context: &'a str,
    pub height: usize,
    pub width: usize,
    pub frames: usize,
    pub fps: f64,
    pub seed: u64,
    pub out_dir: Option<&'a str>,
    pub out_y4m: Option<&'a str>,
    pub out_latent: Option<&'a str>,
    pub skip_decode: bool,
}

/// Frames as one YUV4MPEG2 stream — raw, seekable and exactly what `ffmpeg
/// -i out.y4m out.mp4` wants, so the renderer needs no video encoder.
fn write_y4m(path: &Path, frames: &Vol, fps: f64) -> anyhow::Result<()> {
    let (h, w) = (frames.h, frames.w);
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let num = (fps * 1000.0).round() as u64;
    writeln!(f, "YUV4MPEG2 W{w} H{h} F{num}:1000 Ip A1:1 C420jpeg")?;
    let px = |c: usize, t: usize, y: usize, x: usize| -> f32 {
        let v = frames.data[((c * frames.f + t) * h + y) * w + x];
        ((v.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)
    };
    for t in 0..frames.f {
        writeln!(f, "FRAME")?;
        let mut yp = vec![0u8; h * w];
        let (mut up, mut vp) = (vec![0u8; h * w / 4], vec![0u8; h * w / 4]);
        for y in 0..h {
            for x in 0..w {
                let (r, g, b) = (px(0, t, y, x), px(1, t, y, x), px(2, t, y, x));
                yp[y * w + x] = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
            }
        }
        for y in (0..h).step_by(2) {
            for x in (0..w).step_by(2) {
                let (mut r, mut g, mut b) = (0f32, 0f32, 0f32);
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (yy, xx) = ((y + dy).min(h - 1), (x + dx).min(w - 1));
                        r += px(0, t, yy, xx);
                        g += px(1, t, yy, xx);
                        b += px(2, t, yy, xx);
                    }
                }
                let (r, g, b) = (r / 4.0, g / 4.0, b / 4.0);
                let i = (y / 2) * (w / 2) + x / 2;
                up[i] = (-0.168736 * r - 0.331264 * g + 0.5 * b + 128.0).round().clamp(0.0, 255.0) as u8;
                vp[i] = (0.5 * r - 0.418688 * g - 0.081312 * b + 128.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        f.write_all(&yp)?;
        f.write_all(&up)?;
        f.write_all(&vp)?;
    }
    Ok(())
}

pub fn cmd_ltx_render(a: RenderArgs<'_>) -> anyhow::Result<()> {
    use cortiq_engine::ltxpipe::{Geometry, Rng, Stage, run_stage, unpatchify_video};
    let model = Arc::new(CmfModel::open_sharded(a.model)?);
    let pool = Pool::from_env();
    let t0 = std::time::Instant::now();
    let dit = LtxDit::from_cmf(&model).map_err(|e| anyhow!(e))?;
    eprintln!("dit loaded in {:.1}s ({} blocks)", t0.elapsed().as_secs_f64(), dit.blocks());

    let st = St::open(Path::new(a.context))?;
    let (vshape, vctx) = st
        .get("enc.video")
        .or_else(|_| st.get("v.context"))
        .context("context file needs enc.video (or v.context)")?;
    let (_, actx) = st.get("enc.audio").or_else(|_| st.get("a.context"))?;
    let ctx_len = vshape[vshape.len() - 2];
    eprintln!("prompt context: {ctx_len} tokens");

    let geo = Geometry::new(a.frames, a.height, a.width, a.fps);
    eprintln!(
        "latent {}x{}x{} ({} video tokens), audio {} tokens",
        geo.lf,
        geo.lh,
        geo.lw,
        geo.video_tokens(),
        geo.af
    );
    let mut rng = Rng::new(a.seed);
    let stage = Stage::stage1();
    let total = std::time::Instant::now();
    let lat = run_stage(
        &dit,
        &geo,
        &stage,
        &vctx,
        &actx,
        ctx_len,
        None,
        &mut rng,
        pool.as_deref(),
        &mut |i, n, secs| println!("step {i}/{n}  {secs:.1}s"),
    );
    println!("denoised in {:.1}s", total.elapsed().as_secs_f64());

    let vol = unpatchify_video(&lat.video, &geo);
    if let Some(p) = a.out_latent {
        let mut bytes = Vec::with_capacity(vol.len() * 4);
        for v in &vol {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let hdr = serde_json::json!({
            "latent": {"dtype":"F32","shape":[1, 128, geo.lf, geo.lh, geo.lw],"data_offsets":[0, bytes.len()]}
        });
        let hs = serde_json::to_vec(&hdr)?;
        let mut f = std::fs::File::create(p)?;
        f.write_all(&(hs.len() as u64).to_le_bytes())?;
        f.write_all(&hs)?;
        f.write_all(&bytes)?;
        println!("latent → {p}");
    }
    if a.skip_decode {
        return Ok(());
    }

    let dec = ConvVaeDecoder::from_cmf(&model, pool.as_deref()).map_err(|e| anyhow!(e))?;
    let latent = Vol { c: 128, f: geo.lf, h: geo.lh, w: geo.lw, data: vol };
    let t1 = std::time::Instant::now();
    let out = dec.decode(&latent, pool.as_deref());
    println!(
        "decoded [{}, {}, {}, {}] in {:.1}s",
        out.c,
        out.f,
        out.h,
        out.w,
        t1.elapsed().as_secs_f64()
    );
    if let Some(p) = a.out_y4m {
        write_y4m(Path::new(p), &out, a.fps)?;
        println!("{} frames → {p}  (ffmpeg -i {p} -pix_fmt yuv420p out.mp4)", out.f);
    }
    if let Some(d) = a.out_dir {
        std::fs::create_dir_all(d)?;
        for f in 0..out.f {
            let frame: Vec<f32> = (0..3 * out.h * out.w)
                .map(|i| {
                    let c = i / (out.h * out.w);
                    let r = i % (out.h * out.w);
                    out.data[((c * out.f + f) * out.h) * out.w + r]
                })
                .collect();
            write_ppm(&Path::new(d).join(format!("frame_{f:04}.ppm")), &frame, out.h, out.w)?;
        }
        println!("{} frames → {d}/frame_*.ppm", out.f);
    }
    Ok(())
}

// ------------------------------------------------------------- ltx-encode

pub struct EncodeArgs<'a> {
    pub model: &'a str,
    pub prompt: &'a str,
    pub oracle: Option<&'a str>,
    pub out: Option<&'a str>,
}

pub fn cmd_ltx_encode(a: EncodeArgs<'_>) -> anyhow::Result<()> {
    use cortiq_engine::ltxte::LtxTextEncoder;
    use cortiq_engine::tokenizer::Tokenizer;
    let model = Arc::new(CmfModel::open_sharded(a.model)?);
    let pool = Pool::from_env();
    let vocab = model.vocab.as_ref().context("container carries no tokenizer")?;
    let tok = Tokenizer::from_bytes(vocab).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let t0 = std::time::Instant::now();
    let te = LtxTextEncoder::from_cmf(&model).map_err(|e| anyhow!(e))?;
    eprintln!("prompt encoder loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let ids = tok.encode(a.prompt.trim());
    let (ids, mask) = te.pad_ids(&ids);
    let valid = mask.iter().filter(|&&m| m != 0.0).count();
    eprintln!("{valid} prompt tokens (of {} window)", ids.len());
    eprintln!("ids {:?}", &ids[ids.len() - valid..]);

    let t1 = std::time::Instant::now();
    let hs = te.hidden_states(&ids, &mask, pool.as_deref());
    eprintln!("gemma forward in {:.1}s ({} hidden states)", t1.elapsed().as_secs_f64(), hs.len());

    if let Some(p) = a.oracle {
        let st = St::open(Path::new(p))?;
        // Only the valid rows are compared: the prompt is left-padded, and a
        // pad position attends to nothing at all. The reference's masked
        // softmax leaves those rows a uniform average of the values while
        // ours leaves them zero — both are dead weight, masked out by the
        // feature extractor and overwritten by the connector's registers,
        // and comparing them would drown the signal.
        let cmp = |name: &str, ours: &[f32]| {
            let Ok((_, r)) = st.get(name) else { return };
            if r.len() != ours.len() {
                println!("{name:<18} SHAPE ours {} ref {}", ours.len(), r.len());
                return;
            }
            let skip = ours.len() - ours.len() / ids.len() * valid;
            let (mut worst, mut s, mut rs) = (0f64, 0f64, 0f64);
            for (x, y) in ours.iter().skip(skip).zip(r.iter().skip(skip)) {
                let d = (*x as f64 - *y as f64).abs();
                worst = worst.max(d);
                s += d * d;
                rs += (*y as f64) * (*y as f64);
            }
            println!(
                "{name:<18} worst {:.2e} rel {:.2e}  ({} valid rows)",
                worst,
                (s / rs.max(1e-30)).sqrt(),
                valid
            );
        };
        for (i, h) in hs.iter().enumerate() {
            if i == 0 || i == 1 || i == 5 || i == 6 || i == 24 || i == hs.len() - 1 {
                cmp(&format!("gemma.h{i}"), h);
            }
        }
    }

    let t2 = std::time::Instant::now();
    let (v, au, n) = te.encode_ids(&ids, &mask, pool.as_deref());
    eprintln!("projections + connectors in {:.1}s", t2.elapsed().as_secs_f64());
    if let Some(p) = a.oracle {
        let st = St::open(Path::new(p))?;
        for (name, ours) in [("enc.video", &v), ("enc.audio", &au)] {
            let Ok((_, r)) = st.get(name) else { continue };
            let (mut worst, mut s, mut rs) = (0f64, 0f64, 0f64);
            for (x, y) in ours.iter().zip(&r) {
                let d = (*x as f64 - *y as f64).abs();
                worst = worst.max(d);
                s += d * d;
                rs += (*y as f64) * (*y as f64);
            }
            println!(
                "{name:<18} worst {:.2e} rel {:.2e}",
                worst,
                (s / rs.max(1e-30)).sqrt()
            );
        }
    }
    if let Some(p) = a.out {
        let mut hdr = serde_json::Map::new();
        let mut off = 0usize;
        let mut blob: Vec<u8> = Vec::new();
        for (name, data, dim) in [("enc.video", &v, 4096usize), ("enc.audio", &au, 2048)] {
            let len = data.len() * 4;
            hdr.insert(
                name.to_string(),
                serde_json::json!({"dtype":"F32","shape":[1, n, dim],"data_offsets":[off, off+len]}),
            );
            off += len;
            for x in data.iter() {
                blob.extend_from_slice(&x.to_le_bytes());
            }
        }
        let hs = serde_json::to_vec(&serde_json::Value::Object(hdr))?;
        let mut f = std::fs::File::create(p)?;
        f.write_all(&(hs.len() as u64).to_le_bytes())?;
        f.write_all(&hs)?;
        f.write_all(&blob)?;
        println!("context → {p}");
    }
    Ok(())
}

// -------------------------------------------------------------- ltx-video

pub struct VideoArgs<'a> {
    pub model: &'a str,
    pub two_stage: bool,
    pub prompt: &'a str,
    pub height: usize,
    pub width: usize,
    pub frames: usize,
    pub fps: f64,
    pub seed: u64,
    pub out: Option<&'a str>,
    pub out_dir: Option<&'a str>,
    pub out_latent: Option<&'a str>,
    pub out_audio: Option<&'a str>,
    /// A still to start from (PPM), encoded into the first latent frame
    pub image: Option<&'a str>,
    /// A directory of PPM frames to condition on
    pub video: Option<&'a str>,
    /// Hold the whole picture fixed and write only the soundtrack
    pub video_to_audio: bool,
    /// A soundtrack to start from (16-bit WAV): audio-to-video
    pub audio_in: Option<&'a str>,
    /// How far to re-noise a `--video` clip before denoising it again
    pub video_strength: f32,
    /// Encode the prompt every time instead of reusing a cached context
    pub no_context_cache: bool,
    /// Denoising steps. The default 8 is the distilled ladder itself; other
    /// counts resample it and put the model on sigmas it was not distilled
    /// for, which usually softens the frame rather than sharpening it.
    pub steps: Option<usize>,
    /// Refinement steps in the second stage (`--two-stage`), default 3
    pub steps2: Option<usize>,
}

/// Prompt in, video out — the whole pipeline in one process and one file.
pub fn cmd_ltx_video(a: VideoArgs<'_>) -> anyhow::Result<()> {
    use cortiq_engine::ltxpipe::{Geometry, Rng, Stage, run_stage, unpatchify_video};
    use cortiq_engine::ltxte::LtxTextEncoder;
    use cortiq_engine::tokenizer::Tokenizer;

    anyhow::ensure!(
        a.height % 32 == 0 && a.width % 32 == 0,
        "height and width must be multiples of 32 (the video VAE's spatial stride)"
    );
    anyhow::ensure!(
        a.frames % 8 == 1,
        "frames must be 8k+1 (the VAE's temporal stride, plus the standalone first frame)"
    );

    let model = Arc::new(CmfModel::open_sharded(a.model)?);
    let pool = Pool::from_env();
    let whole = std::time::Instant::now();

    let vocab = model.vocab.as_ref().context("container carries no tokenizer")?;
    let tok = Tokenizer::from_bytes(vocab).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let te = LtxTextEncoder::from_cmf(&model).map_err(|e| anyhow!(e))?;
    let (ids, mask) = te.pad_ids(&tok.encode(a.prompt.trim()));
    let valid = mask.iter().filter(|&&m| m != 0.0).count();
    // The prompt encoder is a 12 B forward that depends on nothing but the
    // token ids and the container — half a minute that a second render of
    // the same prompt has no reason to pay again.
    let cache = (!a.no_context_cache)
        .then(|| context_cache_path(a.model, &ids))
        .flatten();
    let t0 = std::time::Instant::now();
    let cached = cache.as_ref().and_then(|p| read_context(p).ok());
    let (vctx, actx, ctx_len) = match cached {
        Some(c) => {
            println!(
                "prompt: {valid} tokens → {}-token context from cache in {:.2}s",
                c.2,
                t0.elapsed().as_secs_f64()
            );
            c
        }
        None => {
            let c = te.encode_ids(&ids, &mask, pool.as_deref());
            println!(
                "prompt: {valid} tokens → {}-token context in {:.1}s",
                c.2,
                t0.elapsed().as_secs_f64()
            );
            if let Some(p) = &cache {
                if let Err(e) = write_context(p, &c.0, &c.1, c.2) {
                    eprintln!("context cache: {e}");
                }
            }
            c
        }
    };
    drop(te);
    // The 12 B prompt encoder ran its once-per-render pass. Dropping its
    // Rust side frees nothing that matters — the weights are page cache
    // over the mmap, and on a 24 GB Mac the container is 20.5 GiB, so
    // 6.8 GiB of dead encoder competes with the DiT's 10.8 GiB for the
    // rest of the render and the machine answers with the compressor.
    // Clean file-backed pages refault from disk if anything wants them.
    release_pages(&model, "te.");

    let dit = LtxDit::from_cmf(&model).map_err(|e| anyhow!(e))?;
    // Two-stage is how the distilled model was trained to be sampled: eight
    // ancestral steps at half resolution, a learned latent upscale, then
    // three deterministic steps that refine the detail the upscale invented.
    let (h1, w1) = if a.two_stage { (a.height / 2, a.width / 2) } else { (a.height, a.width) };
    let geo = Geometry::new(a.frames, h1, w1, a.fps);
    println!(
        "stage 1 latent {}x{}x{} ({} video tokens), audio {} tokens",
        geo.lf,
        geo.lh,
        geo.lw,
        geo.video_tokens(),
        geo.af
    );
    let mut rng = Rng::new(a.seed);
    // Conditioning: whatever picture the caller supplied is encoded into the
    // model's own latent space and frozen there, and everything else is
    // generated around it.
    let (cond, init) = build_conditioning(
        &model,
        &geo,
        a.image,
        a.video,
        a.audio_in,
        a.video_to_audio,
        a.video_strength,
        pool.as_deref(),
    )?;
    // A clip that covers the whole render is *re-noised*, not frozen: freezing
    // it would hand back exactly what was given. The schedule starts at the
    // strength asked for, which is also what the latent is mixed to.
    let stage1 = match init {
        Some(_) => Stage::from_strength(a.video_strength),
        None => match a.steps {
            Some(n) => Stage::stage1_steps(n),
            None => Stage::stage1(),
        },
    };
    if init.is_some() {
        println!(
            "video-to-video at strength {:.2} — {} steps",
            a.video_strength,
            stage1.sigmas.len() - 1
        );
    }
    let t1 = std::time::Instant::now();
    let lat = cortiq_engine::ltxpipe::run_stage_cond(
        &dit,
        &geo,
        &stage1,
        &vctx,
        &actx,
        ctx_len,
        init,
        cond.as_ref(),
        &mut rng,
        pool.as_deref(),
        &mut |i, n, secs| println!("  step {i}/{n}  {secs:.1}s"),
    );
    println!("stage 1 denoised in {:.1}s", t1.elapsed().as_secs_f64());

    let (geo, vol, audio) = if a.two_stage {
        use cortiq_engine::ltxpipe::patchify_video;
        use cortiq_engine::ltxups::LatentUpscaler;
        let ups = LatentUpscaler::from_cmf(&model).map_err(|e| anyhow!(e))?;
        let small = Vol { c: 128, f: geo.lf, h: geo.lh, w: geo.lw, data: unpatchify_video(&lat.video, &geo) };
        let t = std::time::Instant::now();
        let big = ups.upscale(&small, pool.as_deref());
        println!(
            "upscaled to {}x{} in {:.1}s",
            big.w,
            big.h,
            t.elapsed().as_secs_f64()
        );
        let geo2 = Geometry::new(a.frames, a.height, a.width, a.fps);
        anyhow::ensure!(big.h == geo2.lh && big.w == geo2.lw, "upscaler shape mismatch");
        let init = cortiq_engine::ltxpipe::Latents {
            video: patchify_video(&big.data, &geo2),
            audio: lat.audio.clone(),
        };
        let t2 = std::time::Instant::now();
        let lat2 = run_stage(
            &dit,
            &geo2,
            &match a.steps2 {
                Some(n) => Stage::stage2_steps(n),
                None => Stage::stage2(),
            },
            &vctx,
            &actx,
            ctx_len,
            Some(init),
            &mut rng,
            pool.as_deref(),
            &mut |i, n, secs| println!("  step {i}/{n}  {secs:.1}s"),
        );
        println!("stage 2 denoised in {:.1}s", t2.elapsed().as_secs_f64());
        let v = unpatchify_video(&lat2.video, &geo2);
        (geo2, v, lat2.audio)
    } else {
        let v = unpatchify_video(&lat.video, &geo);
        (geo, v, lat.audio)
    };
    drop(dit);
    // Same again in the other direction: every denoising stage is done, so
    // the DiT's 10.8 GiB stops earning its RAM and the two VAEs and the
    // vocoder get the machine to themselves.
    release_pages(&model, "dit.");
    if let Some(p) = a.out_latent {
        write_latent(p, &vol, &geo, &audio)?;
    }
    // The soundtrack came out of the same 48 blocks as the picture; the
    // audio VAE and its vocoder turn it into a waveform.
    if let Some(p) = a.out_audio {
        use cortiq_engine::ltxaudio::{AudioStack, Grid, write_wav};
        use cortiq_engine::ltxpipe::unpatchify_audio;
        let t = std::time::Instant::now();
        let stack = AudioStack::from_cmf(&model).map_err(|e| anyhow!(e))?;
        let al = unpatchify_audio(&audio, geo.af);
        let wave = stack.decode(&Grid { c: 8, h: geo.af, w: 16, data: al }, pool.as_deref());
        write_wav(Path::new(p), &wave, stack.out_rate)?;
        println!(
            "{:.1}s of {} Hz stereo → {p} ({:.1}s)",
            wave.t as f64 / stack.out_rate as f64,
            stack.out_rate,
            t.elapsed().as_secs_f64()
        );
    }
    let dec = ConvVaeDecoder::from_cmf(&model, pool.as_deref()).map_err(|e| anyhow!(e))?;
    let latent = Vol { c: 128, f: geo.lf, h: geo.lh, w: geo.lw, data: vol };
    let t2 = std::time::Instant::now();
    let out = dec.decode(&latent, pool.as_deref());
    println!(
        "decoded {} frames of {}x{} in {:.1}s",
        out.f,
        out.w,
        out.h,
        t2.elapsed().as_secs_f64()
    );
    if let Some(p) = a.out {
        write_y4m(Path::new(p), &out, a.fps)?;
        println!("{p}  (ffmpeg -i {p} -pix_fmt yuv420p out.mp4)");
    }
    if let Some(d) = a.out_dir {
        write_frames(d, &out)?;
    }
    println!("total {:.1}s", whole.elapsed().as_secs_f64());
    // `CMF_MM_AB=1` ran both arms of every eligible q4tp GEMM back to back
    // on the same data; the table is the only device-vs-host comparison a
    // laptop that drifts between runs can be trusted to give.
    if cortiq_engine::mm_ab::on() {
        eprintln!("{}", cortiq_engine::mm_ab::report());
    }
    Ok(())
}

fn write_latent(
    p: &str,
    vol: &[f32],
    geo: &cortiq_engine::ltxpipe::Geometry,
    audio: &[f32],
) -> anyhow::Result<()> {
    let mut bytes = Vec::with_capacity(vol.len() * 4 + audio.len() * 4);
    for v in vol {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let split = bytes.len();
    for v in audio {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let hdr = serde_json::json!({
        "latent": {"dtype":"F32","shape":[1, 128, geo.lf, geo.lh, geo.lw],"data_offsets":[0, split]},
        "audio_latent": {"dtype":"F32","shape":[1, geo.af, 128],"data_offsets":[split, bytes.len()]}
    });
    let hs = serde_json::to_vec(&hdr)?;
    let mut f = std::fs::File::create(p)?;
    f.write_all(&(hs.len() as u64).to_le_bytes())?;
    f.write_all(&hs)?;
    f.write_all(&bytes)?;
    println!("latent → {p}");
    Ok(())
}

fn write_frames(d: &str, out: &Vol) -> anyhow::Result<()> {
    std::fs::create_dir_all(d)?;
    for f in 0..out.f {
        let frame: Vec<f32> = (0..3 * out.h * out.w)
            .map(|i| {
                let c = i / (out.h * out.w);
                let r = i % (out.h * out.w);
                out.data[((c * out.f + f) * out.h) * out.w + r]
            })
            .collect();
        write_ppm(&Path::new(d).join(format!("frame_{f:04}.ppm")), &frame, out.h, out.w)?;
    }
    println!("{} frames → {d}/frame_*.ppm", out.f);
    Ok(())
}

// -------------------------------------------------------------- ltx-audio

pub struct AudioArgs<'a> {
    pub model: &'a str,
    pub latent: &'a str,
    pub out: &'a str,
    pub stats: bool,
    pub oracle: Option<&'a str>,
}

/// Decode a saved audio latent on its own — the fast loop for the audio
/// tail, which is seconds of work behind minutes of denoising.
pub fn cmd_ltx_audio(a: AudioArgs<'_>) -> anyhow::Result<()> {
    use cortiq_engine::ltxaudio::{AudioStack, Grid, write_wav};
    use cortiq_engine::ltxpipe::unpatchify_audio;
    let model = Arc::new(CmfModel::open_sharded(a.model)?);
    let pool = Pool::from_env();
    let st = St::open(Path::new(a.latent))?;
    // an oracle file carries the reference's own latent, so the comparison
    // is of this stack against that stack on identical input
    let (shape, vals) = st.get("audio_latent")?;
    let frames = if shape.len() == 4 { shape[2] } else { shape[shape.len() - 2] };
    let vals = if shape.len() == 4 {
        // the reference dumps [B, 8, T, 16]; ours is the patchified [B, T, 128]
        let (c, t, m) = (shape[1], shape[2], shape[3]);
        let mut p = vec![0f32; t * c * m];
        for ch in 0..c {
            for ti in 0..t {
                for mi in 0..m {
                    p[ti * c * m + ch * m + mi] = vals[(ch * t + ti) * m + mi];
                }
            }
        }
        p
    } else {
        vals
    };
    let stack = AudioStack::from_cmf(&model).map_err(|e| anyhow!(e))?;
    let grid = Grid { c: 8, h: frames, w: 16, data: unpatchify_audio(&vals, frames) };
    let describe = |name: &str, v: &[f32]| {
        let n = v.len().max(1) as f64;
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
        let rms = (v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / n).sqrt();
        let (lo, hi) = v.iter().fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
        println!("{name:<12} n {:<9} mean {:+.4} rms {:.4} range [{:+.3}, {:+.3}]", v.len(), mean, rms, lo, hi);
    };
    if a.stats {
        describe("latent", &grid.data);
        let mel = stack.decoder.decode(&grid, pool.as_deref());
        describe("log-mel", &mel.data);
        // per-frame energy, to see whether anything happens over time
        let step = (mel.h / 12).max(1);
        let e: Vec<String> = (0..mel.h)
            .step_by(step)
            .map(|f| {
                let mut s = 0f64;
                for c in 0..mel.c {
                    for m in 0..mel.w {
                        s += mel.data[(c * mel.h + f) * mel.w + m] as f64;
                    }
                }
                format!("{:.1}", s / (mel.c * mel.w) as f64)
            })
            .collect();
        println!("mel over time: {}", e.join(" "));
    }
    if let Some(p) = a.oracle {
        let o = St::open(Path::new(p))?;
        let mel = stack.decoder.decode(&grid, pool.as_deref());
        let cmp = |name: &str, ours: &[f32], r: &[f32]| {
            let n = ours.len().min(r.len());
            let (mut worst, mut s, mut rs) = (0f64, 0f64, 0f64);
            for (x, y) in ours[..n].iter().zip(&r[..n]) {
                let d = (*x as f64 - *y as f64).abs();
                worst = worst.max(d);
                s += d * d;
                rs += (*y as f64) * (*y as f64);
            }
            println!(
                "{name:<10} ours {} ref {}  worst {:.2e} rel {:.2e}",
                ours.len(),
                r.len(),
                worst,
                (s / rs.max(1e-30)).sqrt()
            );
        };
        if let Ok((_, r)) = o.get("mel") {
            cmp("mel", &mel.data, &r);
        }
        if let Ok((_, r)) = o.get("waveform") {
            let w = stack.decode(&grid, pool.as_deref());
            cmp("waveform", &w.data, &r);
            // The vocoder is a GAN: it invents phase, and phase is where a
            // small change in its input goes. Run it on the *reference's*
            // own mel to separate our vocoder from our spectrogram.
            if let Ok((ms, mv)) = o.get("mel") {
                let d: Vec<usize> = ms.iter().copied().filter(|&x| x > 0).collect();
                if d.len() >= 3 {
                    let mel = cortiq_engine::ltxaudio::Grid {
                        c: d[d.len() - 3],
                        h: d[d.len() - 2],
                        w: d[d.len() - 1],
                        data: mv,
                    };
                    let w2 = stack.decode_from_mel(&mel, pool.as_deref());
                    cmp("waveform*", &w2.data, &r);
                }
            }
        }
    }
    let t = std::time::Instant::now();
    let wave = stack.decode(&grid, pool.as_deref());
    if a.stats {
        describe("waveform", &wave.data);
        let blk = (wave.t / 12).max(1);
        let env: Vec<String> = (0..wave.t)
            .step_by(blk)
            .map(|i| {
                let hi = (i + blk).min(wave.t);
                let s: f64 = wave.data[i..hi].iter().map(|&x| (x as f64) * (x as f64)).sum();
                format!("{:.3}", (s / (hi - i) as f64).sqrt())
            })
            .collect();
        println!("envelope: {}", env.join(" "));
    }
    write_wav(Path::new(a.out), &wave, stack.out_rate)?;
    println!(
        "{:.2}s of {} Hz stereo → {} ({:.1}s)",
        wave.t as f64 / stack.out_rate as f64,
        stack.out_rate,
        a.out,
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Encode the caller's picture into the latent space and mark which tokens
/// the sampler must leave alone.
#[allow(clippy::too_many_arguments)]
type Conditioned = (
    Option<cortiq_engine::ltxpipe::Conditioning>,
    Option<cortiq_engine::ltxpipe::Latents>,
);

fn build_conditioning(
    model: &Arc<CmfModel>,
    geo: &cortiq_engine::ltxpipe::Geometry,
    image: Option<&str>,
    video: Option<&str>,
    audio_in: Option<&str>,
    video_to_audio: bool,
    _video_strength: f32,
    pool: Option<&cortiq_engine::pool::Pool>,
) -> anyhow::Result<Conditioned> {
    use cortiq_engine::ltxenc::VideoEncoder;
    use cortiq_engine::ltxpipe::{Conditioning, patchify_video};
    // A soundtrack to build a picture around: the wav is resampled to the
    // VAE's 16 kHz, turned into the same log-mel its encoder was trained on,
    // and frozen in the latent the transformer denoises.
    let audio_cond = match audio_in {
        None => None,
        Some(p) => {
            use cortiq_engine::ltxaudio::{AudioVaeEncoder, read_wav, resample, waveform_to_mel};
            let t = std::time::Instant::now();
            let (wave, rate) = read_wav(Path::new(p)).map_err(|e| anyhow!(e))?;
            let wave = resample(&wave, rate, 16000);
            let mel = waveform_to_mel(&wave, 16000, 1024, 160, 64);
            let enc = AudioVaeEncoder::from_cmf(model).map_err(|e| anyhow!(e))?;
            let lat = enc.encode(&mel, pool);
            // patchified layout, cropped or zero-padded to the render's length
            let mut clean = vec![0f32; geo.af * 128];
            for t in 0..geo.af.min(lat.h) {
                for c in 0..lat.c {
                    for m in 0..lat.w {
                        clean[t * 128 + c * lat.w + m] = lat.data[(c * lat.h + t) * lat.w + m];
                    }
                }
            }
            println!(
                "conditioned on {:.2}s of audio → latent [{}, {}, {}] in {:.1}s",
                wave.t as f64 / 16000.0,
                lat.c,
                lat.h,
                lat.w,
                t.elapsed().as_secs_f64()
            );
            Some(clean)
        }
    };
    if image.is_none() && video.is_none() {
        return Ok((audio_cond.map(|a| Conditioning::default().with_audio_all(geo, &a)), None));
    }
    let enc = VideoEncoder::from_cmf(model).map_err(|e| anyhow!(e))?;
    let t = std::time::Instant::now();
    let frames = match (image, video) {
        (Some(p), _) => read_frames_one(p)?,
        (_, Some(d)) => read_frames_dir(d)?,
        _ => unreachable!(),
    };
    anyhow::ensure!(
        frames.h == geo.height && frames.w == geo.width,
        "conditioning is {}x{} but the render is {}x{}",
        frames.w,
        frames.h,
        geo.width,
        geo.height
    );
    let lat = enc.encode(&frames, pool);
    println!(
        "conditioned on {} frame(s) → latent [{}, {}, {}, {}] in {:.1}s",
        frames.f,
        lat.c,
        lat.f,
        lat.h,
        lat.w,
        t.elapsed().as_secs_f64()
    );
    let clean = patchify_video(&lat.data, &cortiq_engine::ltxpipe::Geometry {
        lf: lat.f,
        lh: lat.h,
        lw: lat.w,
        ..*geo
    });
    // pad or crop the encoded prefix into the target token grid
    let mut full = vec![0f32; geo.video_tokens() * 128];
    let take = clean.len().min(full.len());
    full[..take].copy_from_slice(&clean[..take]);
    // Three different things, and the difference matters:
    //  * `--video-to-audio` freezes the picture entirely — the soundtrack is
    //    what is being written;
    //  * a clip that covers the whole render is *re-noised* and denoised
    //    again, because freezing it would return it unchanged;
    //  * a clip (or a still) shorter than the render freezes what it covers
    //    and generates the rest — a continuation.
    let covers_all = lat.f >= geo.lf && video.is_some();
    let (cond, init) = if video_to_audio {
        (Some(Conditioning::video_all(geo, &full)), None)
    } else if covers_all {
        (
            None,
            Some(cortiq_engine::ltxpipe::Latents {
                video: full.clone(),
                audio: vec![0f32; geo.af * 128],
            }),
        )
    } else {
        (Some(Conditioning::video_prefix(geo, &full, lat.f)), None)
    };
    let cond = match (cond, audio_cond) {
        (Some(c), Some(a)) => Some(c.with_audio_all(geo, &a)),
        (Some(c), None) => Some(c),
        (None, Some(a)) => Some(Conditioning::default().with_audio_all(geo, &a)),
        (None, None) => None,
    };
    Ok((cond, init))
}

/// One PPM as a single-frame volume in `[-1, 1]`.
fn read_frames_one(path: &str) -> anyhow::Result<Vol> {
    let (rgb, h, w) = read_ppm_f32(path)?;
    Ok(Vol { c: 3, f: 1, h, w, data: rgb })
}

/// A directory of `frame_*.ppm` as one volume.
fn read_frames_dir(dir: &str) -> anyhow::Result<Vol> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("ppm"))
        .collect();
    paths.sort();
    anyhow::ensure!(!paths.is_empty(), "{dir}: no .ppm frames");
    let mut data: Vec<f32> = Vec::new();
    let (mut h, mut w) = (0usize, 0usize);
    let mut per: Vec<Vec<f32>> = Vec::new();
    for p in &paths {
        let (rgb, ph, pw) = read_ppm_f32(p.to_str().unwrap())?;
        if h == 0 {
            (h, w) = (ph, pw);
        }
        anyhow::ensure!(ph == h && pw == w, "frames differ in size");
        per.push(rgb);
    }
    let f = per.len();
    // planes are channel-major: [C, F, H, W]
    for c in 0..3 {
        for frame in &per {
            data.extend_from_slice(&frame[c * h * w..(c + 1) * h * w]);
        }
    }
    Ok(Vol { c: 3, f, h, w, data })
}

/// A binary PPM (P6) into planar RGB in `[-1, 1]`.
fn read_ppm_f32(path: &str) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    let raw = std::fs::read(path).with_context(|| path.to_string())?;
    anyhow::ensure!(raw.starts_with(b"P6"), "{path}: not a binary PPM");
    let mut fields: Vec<usize> = Vec::new();
    let mut i = 2usize;
    while fields.len() < 3 && i < raw.len() {
        while i < raw.len() && (raw[i] as char).is_whitespace() {
            i += 1;
        }
        if raw[i] == b'#' {
            while i < raw.len() && raw[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        while i < raw.len() && (raw[i] as char).is_ascii_digit() {
            i += 1;
        }
        fields.push(std::str::from_utf8(&raw[start..i])?.parse()?);
    }
    i += 1; // the single whitespace after maxval
    let (w, h) = (fields[0], fields[1]);
    let px = &raw[i..];
    anyhow::ensure!(px.len() >= w * h * 3, "{path}: short pixel data");
    let mut out = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                out[(c * h + y) * w + x] = px[(y * w + x) * 3 + c] as f32 / 127.5 - 1.0;
            }
        }
    }
    Ok((out, h, w))
}

/// Where a prompt's encoded context lives: keyed by the token ids and by
/// which container produced it, since a different pack is a different
/// encoder.
/// Best-effort page-cache release for a finished stage's weights. Names are
/// prefixed by component in the LTX container (`te.`, `dit.`, `vvae.`,
/// `avae.`), and a stage that has run is never read again in one render.
fn release_pages(model: &std::sync::Arc<cortiq_core::CmfModel>, prefix: &str) {
    let dropped = model.advise_done(|n| n.starts_with(prefix));
    if dropped > 0 {
        tracing::info!("{prefix}* pages released: {} MB", dropped / (1024 * 1024));
    }
}

fn context_cache_path(model: &str, ids: &[u32]) -> Option<std::path::PathBuf> {
    use std::hash::{Hash, Hasher};
    let meta = std::fs::metadata(model).ok()?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ids.hash(&mut h);
    meta.len().hash(&mut h);
    if let Ok(t) = meta.modified() {
        if let Ok(d) = t.duration_since(std::time::UNIX_EPOCH) {
            d.as_secs().hash(&mut h);
        }
    }
    model.hash(&mut h);
    let key = h.finish();
    let dir = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?
        .join("cortiq")
        .join("ltx-context");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join(format!("{key:016x}.safetensors")))
}

fn read_context(p: &std::path::Path) -> anyhow::Result<(Vec<f32>, Vec<f32>, usize)> {
    let st = St::open(p)?;
    let (vs, v) = st.get("enc.video")?;
    let (_, a) = st.get("enc.audio")?;
    let n = vs[vs.len() - 2];
    Ok((v, a, n))
}

fn write_context(p: &std::path::Path, v: &[f32], a: &[f32], n: usize) -> anyhow::Result<()> {
    let mut blob: Vec<u8> = Vec::with_capacity((v.len() + a.len()) * 4);
    for x in v {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    let split = blob.len();
    for x in a {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    let hdr = serde_json::json!({
        "enc.video": {"dtype":"F32","shape":[1, n, v.len() / n.max(1)],"data_offsets":[0, split]},
        "enc.audio": {"dtype":"F32","shape":[1, n, a.len() / n.max(1)],"data_offsets":[split, blob.len()]},
    });
    let hs = serde_json::to_vec(&hdr)?;
    let tmp = p.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&(hs.len() as u64).to_le_bytes())?;
        f.write_all(&hs)?;
        f.write_all(&blob)?;
    }
    std::fs::rename(&tmp, p)?;
    Ok(())
}
