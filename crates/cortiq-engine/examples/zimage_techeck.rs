//! The Z-Image text encoder of a container against the fp32 oracle.
//!
//! zimage_techeck <container.cmf> <oracle dir> [p0,p1,...] [reps]
//!
//! For each prompt key, encodes the oracle's `ids` (te_{p}_fp32.safetensors)
//! and prints `h_m2` rel / cos / maxabs against the oracle plus the
//! in-process time of each repetition. The device is paused (the pipeline's
//! CPU text encoder) unless `ZC_TE_GPU=1`; `CMF_SDOT=0` selects the exact
//! weight-only kernels.
use std::collections::HashMap;
use std::sync::Arc;

fn read_st(path: &str) -> HashMap<String, (Vec<usize>, Vec<f64>)> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
    let mut out = HashMap::new();
    for (k, v) in h.as_object().unwrap() {
        let Some(dt) = v["dtype"].as_str() else { continue };
        let o = v["data_offsets"].as_array().unwrap();
        let raw = &b[8 + n + o[0].as_u64().unwrap() as usize..8 + n + o[1].as_u64().unwrap() as usize];
        let shape = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let vals: Vec<f64> = match dt {
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            "I64" => raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            _ => continue,
        };
        out.insert(k.clone(), (shape, vals));
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = Arc::new(cortiq_core::CmfModel::open(&a[1]).unwrap());
    let keys = a.get(3).cloned().unwrap_or("p0,p1".into());
    let reps: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(2);
    // `ZC_TE_GPU=1`: leave the device on (the engine's per-op device path,
    // the arm the pipeline used before B2).
    let _p = (std::env::var("ZC_TE_GPU").as_deref() != Ok("1")).then(cortiq_engine::gpu::pause_gpu);
    let t = std::time::Instant::now();
    let enc = cortiq_engine::qwen3te::Qwen3Encoder::from_cmf(&model).unwrap();
    println!("load {:.3} s", t.elapsed().as_secs_f64());
    for k in keys.split(',') {
        let o = read_st(&format!("{}/te_{k}_fp32.safetensors", a[2]));
        let ids: Vec<u32> = o["ids"].1.iter().map(|&v| v as u32).collect();
        let want = &o["h_m2"].1;
        for r in 0..reps {
            let t = std::time::Instant::now();
            let got = enc.encode(&ids);
            let dt = t.elapsed().as_secs_f64();
            let (mut d, mut nn, mut dot, mut ng, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for (x, y) in got.iter().zip(want) {
                let x = *x as f64;
                d += (x - y) * (x - y);
                nn += y * y;
                dot += x * y;
                ng += x * x;
                mx = mx.max((x - y).abs());
            }
            println!(
                "{k} (L {}) rep {r}: {:.3} s   h_m2 rel {:.3e}  cos {:.9}  maxabs {:.3e}",
                ids.len(),
                dt,
                (d / nn).sqrt(),
                dot / (nn * ng).sqrt(),
                mx
            );
        }
    }
}
