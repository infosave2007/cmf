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
