//! The resident wgpu VAE (`gpu::vae_decode_chain`) against the fp32 oracle.
//!
//! zimage_vaecheck <container.cmf> <oracle vae_*.safetensors> [reps]
//!
//! Decodes the oracle's `z` with `VaeDecoder::decode_fast` and prints rel
//! against the oracle `img` (fp32 diffusers VAE), the u8 PSNR, and the
//! in-process time of every repetition (the first one builds the weights
//! and compiles the kernels).
use std::collections::HashMap;

fn read_st(path: &str) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
    let mut out = HashMap::new();
    for (k, v) in h.as_object().unwrap() {
        if v["dtype"].as_str() != Some("F32") {
            continue;
        }
        let o = v["data_offsets"].as_array().unwrap();
        let raw = &b[8 + n + o[0].as_u64().unwrap() as usize..8 + n + o[1].as_u64().unwrap() as usize];
        let shape = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        out.insert(k.clone(), (shape, raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()));
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = cortiq_core::CmfModel::open(&a[1]).unwrap();
    let o = read_st(&a[2]);
    let reps: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(3);
    let (zs, z) = &o["z"];
    let (_, img) = &o["img"];
    let (h, w) = (zs[zs.len() - 2], zs[zs.len() - 1]);
    let vae = cortiq_engine::vae::VaeDecoder::from_cmf(&model).unwrap();
    for r in 0..reps {
        let t = std::time::Instant::now();
        let got = vae.decode_fast(z, h, w);
        let dt = t.elapsed().as_secs_f64();
        let (mut d, mut n, mut se) = (0f64, 0f64, 0f64);
        for (x, y) in got.iter().zip(img) {
            d += (*x as f64 - *y as f64).powi(2);
            n += (*y as f64).powi(2);
            let q = |v: f32| ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round_ties_even() as f64;
            se += (q(*x) - q(*y)).powi(2);
        }
        let psnr = 10.0 * (255f64.powi(2) / (se / got.len() as f64).max(1e-12)).log10();
        println!("rep {r}: {:.3} s   img rel {:.3e}   u8 PSNR {:.2} dB", dt, (d / n).sqrt(), psnr);
    }
}
