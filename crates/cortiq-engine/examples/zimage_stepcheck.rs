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
        // `ZC_TAPS=<dir>`: every block's residual stream of the three
        // forwards (single pos, single neg, pair) into <dir>/{pos,neg,pair},
        // then compared block by block (the pair's item rows vs the single).
        let taps = std::env::var("ZC_TAPS").ok();
        let set_taps = |sub: &str| {
            if let Some(d) = &taps {
                unsafe { std::env::set_var("CMF_ZI_TAPS", format!("{d}/{sub}")) };
            }
        };
        set_taps("pos");
        let vp = dit.step(&prep, i, &tok_a, &mods, &fs);
        set_taps("neg");
        let vn = dit.step(&np, i, &tok_a, &mods, &fs);
        unsafe { std::env::remove_var("CMF_ZI_TAPS") };
        np.device = false;
        let vn_cpu = dit.step(&np, i, &tok_a, &mods, &fs);
        np.device = true;
        // `ZC_SWAP=1`: the pair as (neg, pos) — tells a positional cause
        // (item 0 vs item 1) from a content one (the caption length).
        let swap = std::env::var("ZC_SWAP").as_deref() == Ok("1");
        let (first, second) = if swap { (&np, &prep) } else { (&prep, &np) };
        let ok = dit.attach_device_pair(first, second, 3, None);
        println!(
            "pair prepared: {ok}  (pos L = {}, neg L = {}, n_img_p {}, order {})",
            ids.len(),
            nids.len(),
            shape.n_img_p,
            if swap { "neg,pos" } else { "pos,neg" }
        );
        set_taps("pair");
        if let Some((p0, p1)) = dit.step_pair_device(3, shape.n_img, i, &tok_a, &mods, &fs) {
            let (pp, pn) = if swap { (p1, p0) } else { (p0, p1) };
            let nan = pp.iter().chain(&pn).filter(|v| !v.is_finite()).count();
            println!("pair vs singles: pos {:.3e}  neg {:.3e}  non-finite {nan}   single neg dev vs cpu {:.3e}",
                rel(&pp, &vp), rel(&pn, &vn), rel(&vn, &vn_cpu));
        }
        unsafe { std::env::remove_var("CMF_ZI_TAPS") };
        if let Some(d) = &taps {
            // Row layout of the pair: image stage [item][n_img_p]; joint
            // stage [item0: n_img_p + cp0][item1: n_img_p + cp1].
            let h = dit.cfg.dim;
            let cp = |l: usize| l.div_ceil(32) * 32;
            let (cp_pos, cp_neg) = (cp(ids.len()), cp(nids.len()));
            let (cp0, cp1) = if swap { (cp_neg, cp_pos) } else { (cp_pos, cp_neg) };
            let nip = shape.n_img_p;
            let rd = |p: String| -> Option<Vec<f32>> {
                std::fs::read(&p).ok().map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
            };
            let mut names: Vec<String> = (0..2).map(|k| format!("nr{k}_out")).collect();
            names.extend((0..dit.cfg.n_layers).map(|k| format!("l{k}_out")));
            for n in names {
                let (Some(sp), Some(sn), Some(pr)) = (
                    rd(format!("{d}/pos/step{i}/{n}.f32")),
                    rd(format!("{d}/neg/step{i}/{n}.f32")),
                    rd(format!("{d}/pair/step{i}/{n}.f32")),
                ) else {
                    continue;
                };
                let (r0, r1) = if n.starts_with("nr") {
                    ((0, nip), (nip, nip))
                } else {
                    ((0, nip + cp0), (nip + cp0, nip + cp1))
                };
                let item = |r: (usize, usize)| &pr[r.0 * h..(r.0 + r.1) * h];
                let (ip, ineg) = if swap { (item(r1), item(r0)) } else { (item(r0), item(r1)) };
                // image rows and caption rows separately
                let split = |v: &[f32]| (v[..nip * h].to_vec(), v[nip * h..].to_vec());
                let (pi, pc) = split(ip);
                let (spi, spc) = split(&sp[..ip.len().min(sp.len())]);
                println!(
                    "{n:9} pos: img {:.3e} cap {:.3e}   neg: all {:.3e}",
                    rel(&pi, &spi),
                    if pc.is_empty() { 0.0 } else { rel(&pc, &spc) },
                    rel(ineg, &sn[..ineg.len().min(sn.len())])
                );
            }
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
