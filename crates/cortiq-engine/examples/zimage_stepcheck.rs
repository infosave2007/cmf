//! One mid-trajectory DiT step, device vs CPU, from traced latents.
//!
//! zimage_stepcheck <container.cmf> <prompt> <H> <W> <steps> <shift> <step i> <trace A> [trace B]
//!
//! Encodes the prompt (CPU text encoder), prepares the caption, then runs
//! step i on `lat_i` of trace A (noise when i = 0 needs CMF_INIT_LATENT)
//! on the device and on the CPU and prints rel(dev, cpu) and rel against
//! trace A's `v_i`. With trace B it also runs the device on B's `lat_i`
//! and prints how far the two device outputs are apart against how far
//! the two inputs are — the trajectory's local sensitivity.
use cortiq_engine::tokenizer::Tokenizer;
use cortiq_engine::zimage::{self, ZImageDit, ZShape};
use std::sync::Arc;

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut r) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64 - *y as f64).powi(2);
        r += (*y as f64).powi(2);
    }
    (d / r.max(1e-300)).sqrt()
}

fn read_f32(p: &str) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{p}: {e}"))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = Arc::new(cortiq_core::CmfModel::open(&a[1]).unwrap());
    let (prompt, hh, ww) = (&a[2], a[3].parse::<usize>().unwrap(), a[4].parse::<usize>().unwrap());
    let (steps, shift, i) = (a[5].parse::<usize>().unwrap(), a[6].parse::<f32>().unwrap(), a[7].parse::<usize>().unwrap());
    let tok = Tokenizer::from_bytes(model.vocab.as_deref().unwrap()).unwrap();
    let ids = cortiq_engine::zimagegen::prompt_ids(&tok, prompt, 512);
    let cap = {
        let _p = cortiq_engine::gpu::pause_gpu();
        cortiq_engine::qwen3te::Qwen3Encoder::from_cmf(&model).unwrap().encode(&ids)
    };
    let dit = ZImageDit::from_cmf(&model).unwrap();
    let sig = zimage::sigmas_torch_f32(steps, shift);
    let t = zimage::t_model(sig[i]);
    let mods = dit.mods_for_steps(&[t]);
    let fs = dit.final_scale_for_steps(&[t]);
    let shape = ZShape::new(hh, ww, ids.len());
    let prep = dit.prepare(&cap, shape, 1, None).unwrap();
    println!("device prepared: {}", prep.device);
    let (c, lh, lw) = (dit.cfg.in_channels, hh / 8, ww / 8);
    let lat = |dir: &str| -> Vec<f32> {
        if i == 0 {
            read_f32(&std::env::var("CMF_INIT_LATENT").unwrap())
        } else {
            read_f32(&format!("{dir}/lat_{i}.f32"))
        }
    };
    let xa = lat(&a[8]);
    // `ZC_NEG=<negative prompt>`: the CFG pair as one batch-2 device
    // forward against the two items stepped one by one.
    if let Ok(neg) = std::env::var("ZC_NEG") {
        let nids = cortiq_engine::zimagegen::prompt_ids(&tok, &neg, 512);
        let ncap = {
            let _p = cortiq_engine::gpu::pause_gpu();
            cortiq_engine::qwen3te::Qwen3Encoder::from_cmf(&model).unwrap().encode(&nids)
        };
        let mut np = dit.prepare(&ncap, ZShape::new(hh, ww, nids.len()), 2, None).unwrap();
        let tok_a = dit.tokens(&xa, &shape);
        let vp = dit.step(&prep, i, &tok_a, &mods, &fs);
        let vn = dit.step(&np, i, &tok_a, &mods, &fs);
        np.device = false;
        let vn_cpu = dit.step(&np, i, &tok_a, &mods, &fs);
        let ok = dit.attach_device_pair(&prep, &np, 3, None);
        println!("pair prepared: {ok}  (neg L = {})", nids.len());
        if let Some((pp, pn)) = dit.step_pair_device(3, shape.n_img, i, &tok_a, &mods, &fs) {
            let nan = pp.iter().chain(&pn).filter(|v| !v.is_finite()).count();
            println!("pair vs singles: pos {:.3e}  neg {:.3e}  non-finite {nan}   single neg dev vs cpu {:.3e}",
                rel(&pp, &vp), rel(&pn, &vn), rel(&vn, &vn_cpu));
        }
        return;
    }
    let tok_a = dit.tokens(&xa, &shape);
    let va_dev = zimage::unpatchify(&dit.step(&prep, i, &tok_a, &mods, &fs), c, lh, lw);
    let mut hp = zimage::ZPrepared { device: false, ..prep };
    let va_cpu = zimage::unpatchify(&dit.step(&hp, i, &tok_a, &mods, &fs), c, lh, lw);
    let va_tr = read_f32(&format!("{}/v_{i}.f32", a[8]));
    println!("step {i} on A's lat: dev vs cpu {:.3e}   cpu vs A's v {:.3e}   dev vs A's v {:.3e}",
        rel(&va_dev, &va_cpu), rel(&va_cpu, &va_tr), rel(&va_dev, &va_tr));
    if let Some(b) = a.get(9) {
        hp.device = true;
        let xb = lat(b);
        let vb_dev = zimage::unpatchify(&dit.step(&hp, i, &dit.tokens(&xb, &shape), &mods, &fs), c, lh, lw);
        println!("inputs A vs B {:.3e}   device outputs {:.3e}   (B's own v {:.3e})",
            rel(&xb, &xa), rel(&vb_dev, &va_dev), rel(&read_f32(&format!("{b}/v_{i}.f32")), &va_dev));
    }
}
