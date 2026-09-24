//! One Z-Image DiT forward on the device against the CPU path on the same
//! container and against the diffusers fp32 oracle (plan WP3 parity gate).
//!
//! zimage_metal_check <container.cmf> <oracle dir> <case, e.g. r512_p0_t8_i0> [reps] [neg case]
//!
//! Prints: refined caption dev vs CPU; `v` device vs CPU (same caption),
//! device vs oracle fp32, CPU vs oracle fp32, oracle bf16 vs fp32 (the
//! diffusers-bf16 floor, when the bf16 dump is present); the in-process
//! step times. `ZC_NEG=1` with a CFG oracle case: the batch-2 pair
//! (positive = the oracle caption, negative = the same caption) against
//! two single forwards. `ZC_NOCPU=1` skips the CPU forward.
use cortiq_engine::zimage::{self, ZImageDit, ZShape};
use std::collections::HashMap;
use std::sync::Arc;

fn read_st(path: &std::path::Path) -> (HashMap<String, Vec<f32>>, serde_json::Value) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
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
        let o = v["data_offsets"].as_array().unwrap();
        let raw = &b[base + o[0].as_u64().unwrap() as usize..base + o[1].as_u64().unwrap() as usize];
        let data: Vec<f32> = match v["dtype"].as_str() {
            Some("F32") => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            Some("BF16") => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            _ => continue,
        };
        out.insert(k.clone(), data);
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
    let a: Vec<String> = std::env::args().collect();
    let model = Arc::new(cortiq_core::CmfModel::open(&a[1]).unwrap());
    let dit = ZImageDit::from_cmf(&model).unwrap();
    let od = std::path::Path::new(&a[2]);
    let case = &a[3];
    let reps: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(2);
    let (o, meta) = read_st(&od.join(format!("dit_{case}_fp32.safetensors")));
    let bf = od.join(format!("dit_{case}_bf16.safetensors"));
    let (hh, ww, l) = (meta_usize(&meta, "H"), meta_usize(&meta, "W"), meta_usize(&meta, "L"));
    let t = o["t_model"][0];
    let cap = &o["cap"];
    let x_in = &o["x_in"];
    let shape = ZShape::new(hh, ww, l);
    let mods = dit.mods_for_steps(&[t]);
    let fs = dit.final_scale_for_steps(&[t]);
    let rope = zimage::ids_and_rope(shape.grid, l, dit.cfg.rope_theta, dit.cfg.axes_dims);
    let t0 = std::time::Instant::now();
    let prep = dit.prepare(cap, shape, 1, None).unwrap();
    println!("device prepared: {} ({:.3}s incl. caption refine)", prep.device, t0.elapsed().as_secs_f64());
    let mut cap_cpu = dit.embed_caption(cap, l);
    dit.refine_caption_cpu(&mut cap_cpu, (&rope.cap.0, &rope.cap.1));
    if let Some(cr) = o.get("cr1_out") {
        println!(
            "caption  dev vs cpu {:.3e}   cpu vs oracle {:.3e}   dev vs oracle {:.3e}",
            rel(&prep.cap, &cap_cpu),
            rel(&cap_cpu, cr),
            rel(&prep.cap, cr)
        );
    }
    let (c, lh, lw) = (dit.cfg.in_channels, hh / 8, ww / 8);
    let x_tok = zimage::pad_rows_repeat_last(&zimage::patchify(x_in, c, lh, lw), shape.n_img, shape.n_img_p, dit.geom().patch_dim);
    let mut v_dev = Vec::new();
    for r in 0..reps {
        let ts = std::time::Instant::now();
        let v = dit.step(&prep, 0, &x_tok, &mods, &fs);
        println!("device step {r}: {:.3}s", ts.elapsed().as_secs_f64());
        if r > 0 {
            println!("replay   {:.3e}", rel(&v, &v_dev));
        }
        v_dev = v;
    }
    let orc = &o["final_out"];
    println!("v        dev vs oracle fp32 {:.3e}", rel(&v_dev, orc));
    if bf.exists() {
        let (ob, _) = read_st(&bf);
        if let Some(vb) = ob.get("final_out") {
            println!("v        oracle bf16 vs fp32 {:.3e} (the diffusers-bf16 floor)", rel(vb, orc));
        }
    }
    if std::env::var("ZC_NOCPU").as_deref() != Ok("1") {
        let mut prep_cpu = dit.prepare_with(cap, shape, 2, None, false).unwrap();
        prep_cpu.cap = prep.cap.clone();
        let ts = std::time::Instant::now();
        let v_cpu = dit.step_cpu(&prep_cpu, &x_tok, &mods, &fs);
        println!("cpu step: {:.3}s", ts.elapsed().as_secs_f64());
        println!(
            "v        dev vs cpu {:.3e}   cpu vs oracle fp32 {:.3e}",
            rel(&v_dev, &v_cpu),
            rel(&v_cpu, orc)
        );
    }
    if std::env::var("ZC_NEG").as_deref() == Ok("1") {
        // batch-2 pair: item 1 = the same caption at a different padded
        // length is not available here, so use the same caption; the pair
        // must reproduce the single forward exactly per item
        let mut p2 = dit.prepare(cap, shape, 7, None).unwrap();
        p2.device = false;
        let pair_key = 9;
        let okp = dit.attach_device_pair(&prep, &p2, pair_key, None);
        println!("pair prepared: {okp}");
        if okp {
            let ts = std::time::Instant::now();
            let r = dit.step_pair_device(pair_key, shape.n_img, 0, &x_tok, &mods, &fs).unwrap();
            println!("pair step: {:.3}s", ts.elapsed().as_secs_f64());
            println!("pair     item0 vs single {:.3e}   item1 vs single {:.3e}", rel(&r.0, &v_dev), rel(&r.1, &v_dev));
        }
    }
}
