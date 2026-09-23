//! B1 device path vs the core's CPU path on one oracle forward, tap by tap.
//! zimage_devcheck <container.cmf> <oracle dir> <case, e.g. r512_p0_t8_i0>
use cortiq_engine::zimage::{self, ZImageDit, ZShape};
use std::collections::HashMap;
use std::sync::Arc;

fn read_st(path: &std::path::Path) -> (HashMap<String, Vec<f32>>, serde_json::Value) {
    let b = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
    let base = 8 + n;
    let mut out = HashMap::new();
    let mut meta = serde_json::Value::Null;
    for (k, v) in h.as_object().unwrap() {
        if k == "__metadata__" {
            meta = v.clone();
            continue;
        }
        if v["dtype"].as_str() != Some("F32") {
            continue;
        }
        let o = v["data_offsets"].as_array().unwrap();
        let raw = &b[base + o[0].as_u64().unwrap() as usize..base + o[1].as_u64().unwrap() as usize];
        out.insert(k.clone(), raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
    }
    (out, meta)
}

fn meta_usize(meta: &serde_json::Value, k: &str) -> usize {
    meta[k].as_str().unwrap().trim_matches('"').parse().unwrap()
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut d, mut r) = (0f64, 0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        d += (x - y) * (x - y);
        r += y * y;
    }
    (d / r.max(1e-300)).sqrt()
}

fn main() {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    let a: Vec<String> = std::env::args().collect();
    let model = Arc::new(cortiq_core::CmfModel::open(&a[1]).unwrap());
    let dit = ZImageDit::from_cmf(&model).unwrap();
    let od = std::path::Path::new(&a[2]);
    let case = &a[3];
    let (o, meta) = read_st(&od.join(format!("dit_{case}_fp32.safetensors")));
    let (hh, ww, l) = (meta_usize(&meta, "H"), meta_usize(&meta, "W"), meta_usize(&meta, "L"));
    let t = o["t_model"][0];
    let cap = &o["cap"];
    let x_in = &o["x_in"];
    let shape = ZShape::new(hh, ww, l);
    let mods = dit.mods_for_steps(&[t]);
    let fs = dit.final_scale_for_steps(&[t]);
    let rope = zimage::ids_and_rope(shape.grid, l, dit.cfg.rope_theta, dit.cfg.axes_dims);
    // caption: device-refined (prepare) vs CPU-refined
    let prep = dit.prepare(cap, shape, 1, None).unwrap();
    println!("device prepared: {}", prep.device);
    let mut cap_cpu = dit.embed_caption(cap, l);
    dit.refine_caption_cpu(&mut cap_cpu, (&rope.cap.0, &rope.cap.1));
    println!("cr1_out  dev vs cpu {:.3e}   cpu vs oracle {:.3e}   dev vs oracle {:.3e}",
        rel(&prep.cap, &cap_cpu), rel(&cap_cpu, &o["cr1_out"]), rel(&prep.cap, &o["cr1_out"]));
    let (c, lh, lw) = (dit.cfg.in_channels, hh / 8, ww / 8);
    let x_tok = zimage::pad_rows_repeat_last(&zimage::patchify(x_in, c, lh, lw), shape.n_img, shape.n_img_p, dit.geom().patch_dim);
    let tapdir = std::env::var("CMF_ZI_TAPS").unwrap_or("/root/zb/taps".into());
    let v_dev = dit.step(&prep, 0, &x_tok, &mods, &fs);
    // Replays must be stateless: the same inputs again, and a different
    // step's mods in between.
    let mods2 = dit.mods_for_steps(&[t * 0.5 + 0.25]);
    let fs2 = dit.final_scale_for_steps(&[t * 0.5 + 0.25]);
    let _ = dit.step(&prep, 1, &x_tok, &mods2, &fs2);
    let v_dev2 = dit.step(&prep, 2, &x_tok, &mods, &fs);
    println!("replay   dev2 vs dev1 {:.3e}", rel(&v_dev2, &v_dev));
    let mut taps: Vec<(String, Vec<f32>)> = Vec::new();
    // the CPU path must see the SAME (device-refined) caption to isolate the DiT
    let mut prep_cpu = prep;
    prep_cpu.device = false;
    let v_cpu = dit.step_cpu_taps(&prep_cpu, &x_tok, &mods, &fs, &mut |n, v| taps.push((n.to_string(), v.to_vec())));
    println!("v        dev vs cpu {:.3e}   cpu vs oracle(final_out) {:.3e}", rel(&v_dev, &v_cpu), rel(&v_cpu, &o["final_out"]));
    for (n, v) in &taps {
        let p = std::path::Path::new(&tapdir).join("step0").join(format!("{n}.f32"));
        if let Ok(b) = std::fs::read(&p) {
            let d: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let or = o.get(n.as_str()).map(|w| format!("{:.3e}", rel(v, w))).unwrap_or_default();
            println!("{n:10} dev vs cpu {:.3e}   (cpu vs oracle {or})", rel(&d, v));
        }
    }
}
