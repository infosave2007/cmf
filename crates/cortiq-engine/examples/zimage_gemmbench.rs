//! Z-Image fast-path measurements on the wgpu device (plan WP2, B1):
//! the card's real ceilings, the existing GEMM arms at Z-Image shapes, the
//! new `zi_mm` / `zi_flash` kernels (speed + precision against an f64 host
//! reference), and the synthetic whole-step benchmark.
//!
//! ```text
//! export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
//! flock /root/gpu.lock target/release/examples/zimage_gemmbench info
//! flock /root/gpu.lock target/release/examples/zimage_gemmbench existing
//! flock /root/gpu.lock target/release/examples/zimage_gemmbench mm [bm,bn,bk,wm,wn …]
//! flock /root/gpu.lock target/release/examples/zimage_gemmbench prec
//! flock /root/gpu.lock target/release/examples/zimage_gemmbench flash [nw,bc …]
//! flock /root/gpu.lock target/release/examples/zimage_gemmbench step <512|1024> <batch> [reps]
//! flock /root/gpu.lock target/release/examples/zimage_gemmbench lumina <512|1024> <batch> [reps]
//! ```
//! Every time is in-process (device submit → fence), median of rounds.

#[cfg(feature = "gpu")]
mod imp {
    use cortiq_engine::gpu_wgpu::zimage::{self as zi, bench, Epi, FlashCfg, MmCfg};

    /// Z-Image GEMM sites: (name, N plane rows, K, epilogue).
    const SITES: [(&str, usize, usize, Epi); 4] = [
        ("qkv", 11520, 3840, Epi::F16),
        ("o", 3840, 3840, Epi::F32),
        ("w13", 20480, 3840, Epi::SwiGlu),
        ("w2", 3840, 10240, Epi::F32),
    ];
    /// Token counts: 512² (cap 32), 1024² (cap 128), and batch 2 of each.
    const MS: [usize; 4] = [1056, 2112, 4224, 8448];

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn uni(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
        /// Roughly N(0,1) (sum of 4 uniforms, scaled).
        fn gauss(&mut self) -> f32 {
            (self.uni() + self.uni() + self.uni() + self.uni()) * 0.866
        }
    }

    fn f16(x: f32) -> u16 {
        cortiq_core::quant::f32_to_f16(x)
    }
    fn f32h(h: u16) -> f32 {
        cortiq_core::quant::f16_to_f32(h)
    }

    fn tflops(m: usize, n: usize, k: usize, s: f64) -> f64 {
        2.0 * m as f64 * n as f64 * k as f64 / s / 1e12
    }

    fn parse_cfgs(args: &[String], epi: Epi) -> Vec<MmCfg> {
        let mut v: Vec<MmCfg> = args
            .iter()
            .filter_map(|a| {
                let direct = a.ends_with('d');
                let t: Vec<u32> = a.trim_end_matches('d').split(',').filter_map(|x| x.parse().ok()).collect();
                (t.len() == 5).then(|| MmCfg { direct, ..MmCfg::new(t[0], t[1], t[2], t[3], t[4], epi) })
            })
            .collect();
        if v.is_empty() {
            v.push(zi::default_cfg(epi));
        }
        v.into_iter().map(|c| MmCfg { epi, ..c }).collect()
    }

    fn cmd_info() {
        match bench::info() {
            Some(s) => print!("{s}"),
            None => {
                println!("no wgpu device");
                return;
            }
        }
        for acc16 in [false, true] {
            match bench::peak(acc16) {
                Some(t) => println!("pure MMA peak, acc {}: {t:.1} TFLOPS", if acc16 { "f16" } else { "f32" }),
                None => println!("pure MMA peak, acc {}: pipeline rejected", if acc16 { "f16" } else { "f32" }),
            }
        }
    }

    fn q4tp_len(rows: usize, cols: usize) -> usize {
        let groups = cols / 32;
        rows * groups * 16 + rows * 4 + rows * (groups * 5).div_ceil(8)
    }

    fn cmd_existing(args: &[String]) {
        let arms: Vec<&str> = if args.is_empty() {
            vec!["scalar", "coop_q4", "coop_f16"]
        } else {
            args.iter().map(|s| s.as_str()).collect()
        };
        let mut rng = Rng(0x9e3779b97f4a7c15);
        for &(name, n, k, _) in &SITES {
            // Random bytes: the kernels' speed does not depend on the values.
            let q: Vec<u8> = (0..q4tp_len(n, k)).map(|_| (rng.next() >> 32) as u8).collect();
            for &m in &MS {
                let mut line = format!("{name:>4} M={m:<5} N={n:<5} K={k:<5}");
                for arm in &arms {
                    match bench::existing(arm, m, k, n, &q) {
                        Some(s) => line += &format!("  {arm} {:.2} ms {:.1} TF", s * 1e3, tflops(m, n, k, s)),
                        None => line += &format!("  {arm} n/a"),
                    }
                }
                println!("{line}");
            }
        }
    }

    fn cmd_mm(args: &[String]) {
        for &(name, n, k, epi) in &SITES {
            for cfg in parse_cfgs(args, epi) {
                let mut line = format!("{name:>4} N={n:<5} K={k:<5} {:?}", (cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn, cfg.direct));
                for &m in &MS {
                    match bench::mm_time(cfg, m, k, n) {
                        Some(s) => line += &format!("  M{m} {:.2}ms {:.1}TF", s * 1e3, tflops(m, n, k, s)),
                        None => line += &format!("  M{m} n/a"),
                    }
                }
                println!("{line}");
            }
        }
    }

    /// zi_mm vs an f64 host reference on f16-rounded operands (so the number is
    /// the kernel's accumulation error, plus the f16 output rounding for the
    /// f16 epilogues). Rows sampled to keep the host side cheap.
    fn cmd_prec(args: &[String]) {
        let m = 256usize;
        let mut rng = Rng(12345);
        for &(name, n, k, epi) in &SITES {
            for cfg in parse_cfgs(args, epi) {
                let act: Vec<u16> = (0..m * k).map(|_| f16(rng.gauss())).collect();
                let amp = 1.7 / (k as f32).sqrt();
                let pl: Vec<u16> = (0..n * k).map(|_| f16(rng.uni() * amp)).collect();
                let Some(out) = bench::mm_run(cfg, m, k, n, &act, &pl) else {
                    println!("{name}: n/a");
                    continue;
                };
                let a: Vec<f64> = act.iter().map(|&h| f32h(h) as f64).collect();
                let w: Vec<f64> = pl.iter().map(|&h| f32h(h) as f64).collect();
                let dot = |r: usize, row: usize| -> f64 {
                    let (x, y) = (&a[r * k..(r + 1) * k], &w[row * k..(row + 1) * k]);
                    x.iter().zip(y).map(|(p, q)| p * q).sum()
                };
                let ncol = if epi == Epi::SwiGlu { n / 2 } else { n };
                let (mut e2, mut r2, mut mx) = (0f64, 0f64, 0f64);
                let rows: Vec<usize> = (0..m).step_by(7).collect();
                let refs: Vec<(usize, Vec<f64>)> = std::thread::scope(|sc| {
                    let hs: Vec<_> = rows
                        .chunks(rows.len().div_ceil(24))
                        .map(|chunk| {
                            let chunk = chunk.to_vec();
                            let dot = &dot;
                            sc.spawn(move || {
                                chunk
                                    .into_iter()
                                    .map(|r| {
                                        let v: Vec<f64> = (0..ncol)
                                            .map(|j| match epi {
                                                Epi::SwiGlu => {
                                                    let (q, t) = (j / 16, j % 16);
                                                    let g = dot(r, 32 * q + t);
                                                    let u = dot(r, 32 * q + 16 + t);
                                                    g / (1.0 + (-g).exp()) * u
                                                }
                                                _ => dot(r, j),
                                            })
                                            .collect();
                                        (r, v)
                                    })
                                    .collect::<Vec<_>>()
                            })
                        })
                        .collect();
                    hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
                });
                for (r, v) in refs {
                    for (j, &rf) in v.iter().enumerate() {
                        let d = out[r * ncol + j] as f64 - rf;
                        e2 += d * d;
                        r2 += rf * rf;
                        mx = mx.max(d.abs());
                    }
                }
                println!(
                    "{name:>4} K={k:<5} {:?} {:?}: rel {:.2e}  maxabs {:.2e}  (ref rms {:.3})",
                    epi,
                    (cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn, cfg.direct),
                    (e2 / r2).sqrt(),
                    mx,
                    (r2 / (rows.len() * ncol) as f64).sqrt()
                );
            }
        }
    }

    fn flash_cfgs(args: &[String]) -> Vec<FlashCfg> {
        let mut v: Vec<FlashCfg> = args
            .iter()
            .filter_map(|a| {
                let t: Vec<u32> = a.split(',').filter_map(|x| x.parse().ok()).collect();
                (t.len() == 2).then(|| FlashCfg { nw: t[0], bc: t[1] })
            })
            .collect();
        if v.is_empty() {
            v.push(zi::default_flash());
        }
        v
    }

    /// Host reference attention for (segment, head) on the f16 panel.
    fn attn_ref(qkv: &[u16], nh: usize, seg: (usize, usize), h: usize, qi: usize) -> Vec<f64> {
        let ld = 3 * nh * 128;
        let g = |row: usize, col: usize| f32h(qkv[row * ld + col]) as f64;
        let (off, len) = seg;
        let q: Vec<f64> = (0..128).map(|d| g(off + qi, h * 128 + d)).collect();
        let sc = 1.0 / (128f64).sqrt();
        let s: Vec<f64> = (0..len)
            .map(|j| (0..128).map(|d| q[d] * g(off + j, nh * 128 + h * 128 + d)).sum::<f64>() * sc)
            .collect();
        let mx = s.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
        let l: f64 = e.iter().sum();
        (0..128)
            .map(|d| (0..len).map(|j| e[j] * g(off + j, 2 * nh * 128 + h * 128 + d)).sum::<f64>() / l)
            .collect()
    }

    fn cmd_flash(args: &[String]) {
        let nh = 30usize;
        // Correctness: two segments (batch 2, unequal caption lengths), with a
        // score scale that forces lazy rescales (q·k grows along the keys).
        let segs = [(0usize, 1056usize), (1056, 1024)];
        let m = 2080;
        let ld = 3 * nh * 128;
        let mut rng = Rng(777);
        let mut qkv = vec![0u16; m * ld];
        for row in 0..m {
            for col in 0..ld {
                let base = rng.gauss();
                // k grows with the row index inside the segment → the running
                // max climbs block after block (exercises the rescale path).
                let v = if (nh * 128..2 * nh * 128).contains(&col) { base * (1.0 + (row % 1056) as f32 / 300.0) } else { base };
                qkv[row * ld + col] = f16(v * 1.5);
            }
        }
        for f in flash_cfgs(args) {
            let Some(out) = bench::flash_run(f, nh, &qkv, &segs) else {
                println!("flash {f:?}: n/a (shared {} B)", f.shared_bytes());
                continue;
            };
            let (mut e2, mut r2, mut mx) = (0f64, 0f64, 0f64);
            for (si, &seg) in segs.iter().enumerate() {
                for &h in &[0usize, 13, 29] {
                    for &qi in &[0usize, 1, 17, 63, 64, 500, seg.1 - 1] {
                        let rf = attn_ref(&qkv, nh, seg, h, qi);
                        for d in 0..128 {
                            let got = out[(seg.0 + qi) * nh * 128 + h * 128 + d] as f64;
                            let dd = got - rf[d];
                            e2 += dd * dd;
                            r2 += rf[d] * rf[d];
                            mx = mx.max(dd.abs());
                        }
                    }
                }
                let _ = si;
            }
            println!("flash {f:?}: rel {:.2e} maxabs {:.2e}", (e2 / r2).sqrt(), mx);
            for (label, segs) in [
                ("512² b1", vec![(0usize, 1056usize)]),
                ("512² b2", vec![(0, 1056), (1056, 1056)]),
                ("1024² b1", vec![(0, 4224)]),
                ("1024² b2", vec![(0, 4224), (4224, 4224)]),
            ] {
                match bench::flash_time(f, nh, &segs) {
                    Some(s) => {
                        let fl: f64 = segs.iter().map(|&(_, l)| 4.0 * (l * l) as f64 * (nh * 128) as f64).sum();
                        println!("  {label}: {:.3} ms  {:.1} TF  (×30 layers = {:.1} ms)", s * 1e3, fl / s / 1e12, s * 30e3);
                    }
                    None => println!("  {label}: n/a"),
                }
            }
        }
    }


    // ── one block on the device vs an f64 host block (small dims) ─────────

    fn rms(v: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
        let r = 1.0 / ((v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64) + eps).sqrt();
        v.iter().zip(w).map(|(x, w)| x * r * w).collect()
    }

    fn matvec(x: &[f64], w: &[f64], rows: usize) -> Vec<f64> {
        let k = x.len();
        (0..rows).map(|r| (0..k).map(|j| x[j] * w[r * k + j]).sum()).collect()
    }

    /// Device block (zi_rowop → qkv → qkrope → flash → o → gres+pre →
    /// w13 SwiGLU → w2 → gres) against the diffusers block math in f64.
    pub fn cmd_blockcheck(args: &[String]) {
        let nh: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(2);
        let h = nh * 128;
        let inter = 4 * h;
        let d = zi::ZDims { h, nh, inter, eps: 1e-5, final_eps: 1e-6, pd: 64 };
        let segs = [(0usize, 96usize), (96, 64)];
        let m = 160;
        let mut rng = Rng(4242);
        let mut wv = |rows: usize, cols: usize| -> Vec<u16> {
            let a = 1.7 / (cols as f32).sqrt();
            (0..rows * cols).map(|_| f16(rng.uni() * a)).collect()
        };
        let (wq, wk, wvv, wo) = (wv(h, h), wv(h, h), wv(h, h), wv(h, h));
        let (w1, w3, w2) = (wv(inter, h), wv(inter, h), wv(h, inter));
        let mut nv = |n: usize| -> Vec<f32> { (0..n).map(|_| 1.0 + 0.2 * rng.uni()).collect() };
        let norms: Vec<Vec<f32>> = vec![nv(h), nv(h), nv(h), nv(h), nv(128), nv(128)];
        let mods: Vec<f32> = (0..4 * h).map(|_| 0.5 * rng.gauss()).collect();
        let x0: Vec<f32> = (0..m * h).map(|_| 2.0 * rng.gauss()).collect();
        // rope: segment-local positions on 3 axes, θ = 256 (spec §2.6)
        let (mut rc, mut rs) = (vec![0f32; m * 64], vec![0f32; m * 64]);
        for &(off, len) in &segs {
            for t in 0..len {
                let pos = [1 + t / 40, (t / 8) % 5, t % 8];
                let mut j = 0;
                for (ax, dim) in [(0usize, 32usize), (1, 48), (2, 48)] {
                    for pp in 0..dim / 2 {
                        let f = 1.0 / 256f64.powf(2.0 * pp as f64 / dim as f64);
                        let ang = (pos[ax] as f64 * f) as f32;
                        rc[(off + t) * 64 + j] = ang.cos();
                        rs[(off + t) * 64 + j] = ang.sin();
                        j += 1;
                    }
                }
            }
        }
        let Some(blk) = zi::ZBlockDev::from_host(
            &d, &wq, &wk, &wvv, &wo, &w1, &w3, &w2,
            [&norms[0], &norms[1], &norms[2], &norms[3], &norms[4], &norms[5]],
        ) else {
            println!("no device");
            return;
        };
        let seq = zi::ZSeq::new(&d, &segs).unwrap();
        seq.set_rope(&rc, &rs);
        seq.write_x(&x0);
        let mb = zi::upload_f32(&mods).unwrap();
        let t = zi::ZTiles::default();
        let calls = zi::block_calls(&d, &t, &seq, &blk, &mb, 0, true, None, true).expect("block calls");
        calls.run().unwrap();
        let got = seq.read_x(&d).unwrap();
        // host f64
        let g = |v: &[u16]| -> Vec<f64> { v.iter().map(|&x| f32h(x) as f64).collect() };
        let (wq, wk, wvv, wo, w1, w3, w2) = (g(&wq), g(&wk), g(&wvv), g(&wo), g(&w1), g(&w3), g(&w2));
        let nf: Vec<Vec<f64>> = norms.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
        let md: Vec<f64> = mods.iter().map(|&x| x as f64).collect();
        let (s_msa, g_msa, s_mlp, g_mlp) = (&md[0..h], &md[h..2 * h], &md[2 * h..3 * h], &md[3 * h..4 * h]);
        let mut x: Vec<Vec<f64>> = (0..m).map(|r| x0[r * h..(r + 1) * h].iter().map(|&v| v as f64).collect()).collect();
        let mut q = vec![vec![0f64; h]; m];
        let mut k = vec![vec![0f64; h]; m];
        let mut v = vec![vec![0f64; h]; m];
        for r in 0..m {
            let xn: Vec<f64> = rms(&x[r], &nf[0], 1e-5).iter().zip(s_msa).map(|(a, s)| a * (1.0 + s)).collect();
            q[r] = matvec(&xn, &wq, h);
            k[r] = matvec(&xn, &wk, h);
            v[r] = matvec(&xn, &wvv, h);
            for hh in 0..nh {
                for (buf, w) in [(&mut q[r], &nf[4]), (&mut k[r], &nf[5])] {
                    let seg = &buf[hh * 128..(hh + 1) * 128];
                    let n = rms(seg, w, 1e-5);
                    for p in 0..64 {
                        let (c, s) = (rc[r * 64 + p] as f64, rs[r * 64 + p] as f64);
                        let (a, b) = (n[2 * p], n[2 * p + 1]);
                        buf[hh * 128 + 2 * p] = a * c - b * s;
                        buf[hh * 128 + 2 * p + 1] = a * s + b * c;
                    }
                }
            }
        }
        let mut att = vec![vec![0f64; h]; m];
        for &(off, len) in &segs {
            for i in 0..len {
                for hh in 0..nh {
                    let sc: Vec<f64> = (0..len)
                        .map(|j| (0..128).map(|dd| q[off + i][hh * 128 + dd] * k[off + j][hh * 128 + dd]).sum::<f64>() / (128f64).sqrt())
                        .collect();
                    let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
                    let e: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let l: f64 = e.iter().sum();
                    for dd in 0..128 {
                        att[off + i][hh * 128 + dd] = (0..len).map(|j| e[j] * v[off + j][hh * 128 + dd]).sum::<f64>() / l;
                    }
                }
            }
        }
        for r in 0..m {
            let o = matvec(&att[r], &wo, h);
            let on = rms(&o, &nf[1], 1e-5);
            for c in 0..h {
                x[r][c] += g_msa[c].tanh() * on[c];
            }
            let xn2: Vec<f64> = rms(&x[r], &nf[2], 1e-5).iter().zip(s_mlp).map(|(a, s)| a * (1.0 + s)).collect();
            let a1 = matvec(&xn2, &w1, inter);
            let a3 = matvec(&xn2, &w3, inter);
            let hid: Vec<f64> = a1.iter().zip(&a3).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
            let y = matvec(&hid, &w2, h);
            let yn = rms(&y, &nf[3], 1e-5);
            for c in 0..h {
                x[r][c] += g_mlp[c].tanh() * yn[c];
            }
        }
        let (mut e2, mut r2, mut d2) = (0f64, 0f64, 0f64);
        for r in 0..m {
            for c in 0..h {
                let dd = got[r * h + c] as f64 - x[r][c];
                e2 += dd * dd;
                r2 += x[r][c] * x[r][c];
                let delta = x[r][c] - x0[r * h + c] as f64;
                d2 += delta * delta;
            }
        }
        println!(
            "block nh={nh} h={h}: x rel {:.2e}; relative to the block's update ‖Δx‖: {:.2e}",
            (e2 / r2).sqrt(),
            (e2 / d2).sqrt()
        );
    }

    // ── synthetic whole step ─────────────────────────────────────────────

    fn step_flops(n_img_p: usize, caps: &[usize], d: &zi::ZDims) -> f64 {
        let (h, i) = (d.h as f64, d.inter as f64);
        let lin = |t: f64| 2.0 * t * (3.0 * h * h + h * h + 2.0 * i * h + h * i);
        let att = |s: f64| 4.0 * s * s * h;
        let mut f = 0.0;
        for &cp in caps {
            f += 2.0 * (lin(n_img_p as f64) + att(n_img_p as f64));
            f += 30.0 * (lin((n_img_p + cp) as f64) + att((n_img_p + cp) as f64));
        }
        f
    }

    fn geometry(res: usize, batch: usize) -> (usize, Vec<usize>) {
        let n_img = (res / 16) * (res / 16);
        let cap = if res <= 512 { 32 } else { 128 };
        (n_img, vec![cap; batch])
    }

    pub fn cmd_step(args: &[String]) {
        let res: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(1024);
        let batch: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
        let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
        let d = zi::ZDims::TURBO;
        let (n_img, caps) = geometry(res, batch);
        let n_img_p = n_img.div_ceil(32) * 32;
        let t0 = std::time::Instant::now();
        let blocks: Vec<zi::ZBlockDev> = (0..32).map(|s| zi::ZBlockDev::synthetic(&d, s as u32 + 1).expect("planes")).collect();
        let mut rng = Rng(99);
        let ew: Vec<f32> = (0..d.h * 64).map(|_| rng.uni() * 0.2).collect();
        let eb: Vec<f32> = (0..d.h).map(|_| rng.uni() * 0.1).collect();
        let ep: Vec<f32> = (0..d.h).map(|_| rng.uni()).collect();
        let fw: Vec<f32> = (0..64 * d.h).map(|_| rng.uni() * 0.02).collect();
        let fb: Vec<f32> = (0..64).map(|_| rng.uni() * 0.1).collect();
        let io = zi::ZIo { x_emb_w: &ew, x_emb_b: &eb, x_pad: &ep, final_w: &fw, final_b: &fb };
        let cap: Vec<f32> = (0..caps.iter().sum::<usize>() * d.h).map(|_| rng.gauss()).collect();
        let t = zi::ZTiles::default();
        let Some(sd) = zi::ZStepDev::new(d, &t, &blocks[..2], &blocks[2..], &io, n_img, &caps, &cap) else {
            println!("step program: n/a");
            return;
        };
        // rope tables (the host builds them once per prompt/resolution)
        let mk_rope = |segs: &[(usize, usize)], m: usize, img_only: bool| -> (Vec<f32>, Vec<f32>) {
            let (mut c, mut s) = (vec![0f32; m * 64], vec![0f32; m * 64]);
            let side = res / 16;
            for (bi, &(off, len)) in segs.iter().enumerate() {
                for tkn in 0..len {
                    let pos = if tkn < n_img {
                        [caps[bi] + 1, tkn / side, tkn % side]
                    } else if tkn < n_img_p || img_only {
                        [0, 0, 0]
                    } else {
                        [1 + tkn - n_img_p, 0, 0]
                    };
                    let mut j = 0;
                    for (ax, dim) in [(0usize, 32usize), (1, 48), (2, 48)] {
                        for pp in 0..dim / 2 {
                            let f = 1.0 / 256f64.powf(2.0 * pp as f64 / dim as f64);
                            let ang = (pos[ax] as f64 * f) as f32;
                            c[(off + tkn) * 64 + j] = ang.cos();
                            s[(off + tkn) * 64 + j] = ang.sin();
                            j += 1;
                        }
                    }
                }
            }
            (c, s)
        };
        let (c1, s1) = mk_rope(&sd.img.segs, sd.img.m, true);
        sd.img.set_rope(&c1, &s1);
        let (c2, s2) = mk_rope(&sd.joint.segs, sd.joint.m, false);
        sd.joint.set_rope(&c2, &s2);
        let x_tok: Vec<f32> = (0..batch * n_img_p * 64).map(|_| rng.gauss()).collect();
        let mods: Vec<f32> = (0..32 * 4 * d.h).map(|_| 0.3 * rng.gauss()).collect();
        let fscale: Vec<f32> = (0..d.h).map(|_| 1.0 + 0.1 * rng.uni()).collect();
        sd.upload(&x_tok, &mods, &fscale);
        let mut out = vec![0f32; batch * n_img * 64];
        sd.run(&mut out).expect("run");
        let finite = out.iter().all(|v| v.is_finite());
        let rmsv = (out.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / out.len() as f64).sqrt();
        println!(
            "zi step {res}² batch {batch}: S={:?} dispatches {} setup {:.1} s  output finite {finite} rms {rmsv:.3}",
            sd.joint.segs,
            sd.dispatches(),
            t0.elapsed().as_secs_f64()
        );
        let fl = step_flops(n_img_p, &caps, &d);
        let tt = sd.time(reps, 3, None).unwrap();
        println!("  step {:.3} s  ({:.1} TFLOPS effective, {:.2} TF/step)", tt, fl / tt / 1e12, fl / 1e12);
        if std::env::var("ZB_PROFILE").as_deref() != Ok("0") {
            let mut sum = 0.0;
            for cl in zi::Class::ALL {
                let tc = sd.time(reps.min(2), 3, Some(cl)).unwrap();
                sum += tc;
                println!("    {cl:?}: {:.1} ms", tc * 1e3);
            }
            println!("    sum of classes {:.1} ms", sum * 1e3);
        }
    }

    // ── Lumina-path baseline: gpu::dit_block_seg as is ──────────────────

    pub fn cmd_lumina(args: &[String]) {
        use cortiq_core::{CmfHeader, CmfModel, TensorDtype, TensorSpec, CMF_VERSION};
        use std::sync::Arc;
        let res: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(1024);
        let batch: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
        let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
        let resident = std::env::var("ZB_RESIDENT").as_deref() == Ok("1");
        let d = zi::ZDims::TURBO;
        let (h, inter, nh) = (d.h, d.inter, d.nh);
        let path = std::path::PathBuf::from(std::env::var("ZB_TMP").unwrap_or("/root/zb/tmp".into())).join("lumina_q4tp_block.cmf");
        if !path.exists() {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut tensors = Vec::new();
            for (name, rows, cols) in [
                ("b.q", h, h),
                ("b.k", h, h),
                ("b.v", h, h),
                ("b.o", h, h),
                ("b.w1", inter, h),
                ("b.w3", inter, h),
                ("b.w2", h, inter),
            ] {
                // Constant nibbles/params: finite values, speed is value-blind.
                tensors.push(TensorSpec { name: name.into(), dtype: TensorDtype::Q4TiledP, shape: vec![rows, cols], data: vec![0x11u8; q4tp_len(rows, cols)] });
            }
            let hdr: CmfHeader = serde_json::from_value(serde_json::json!({
                "version": CMF_VERSION,
                "arch": { "arch_name": "zimage-lumina-baseline", "hidden_size": h, "intermediate_size": inter,
                          "num_layers": 1, "num_attention_heads": nh, "num_kv_heads": nh, "head_dim": 128,
                          "vocab_size": 0, "layer_types": [], "rms_norm_eps": 1e-5, "max_position_embeddings": 8192 },
                "quant_type": "F32"
            }))
            .unwrap();
            CmfModel::write(&path, &hdr, &tensors, None, None).unwrap();
        }
        let model = Arc::new(CmfModel::open(&path).unwrap());
        let (n_img, caps) = geometry(res, batch);
        let s = n_img + caps[0];
        let n = s * batch;
        let segs = vec![s; batch];
        let mut rng = Rng(5);
        let mut x: Vec<f32> = (0..n * h).map(|_| rng.gauss()).collect();
        let rc: Vec<f32> = (0..n * 64).map(|i| ((i % 97) as f32 * 0.01).cos()).collect();
        let rs: Vec<f32> = (0..n * 64).map(|i| ((i % 97) as f32 * 0.01).sin()).collect();
        let ones = vec![1.0f32; h];
        let ones_hd = vec![1.0f32; 128];
        let sm: Vec<f32> = (0..h).map(|_| 0.1 * rng.uni()).collect();
        let gm: Vec<f32> = (0..h).map(|_| 0.1 * rng.uni()).collect();
        let idx = |nm: &str| model.tensors.iter().position(|t| t.name == nm).unwrap();
        let mut times = Vec::new();
        for step in 0..=steps {
            let t = std::time::Instant::now();
            for bi in 0..32 {
                let a = cortiq_engine::gpu::DitBlockArgs {
                    n, hidden: h, inter, nh, nkv: nh, hd: 128, eps: 1e-5,
                    rope_cos: &rc, rope_sin: &rs,
                    norm1: &ones, norm2: &ones, ffn_norm1: &ones, ffn_norm2: &ones,
                    norm_q: &ones_hd, norm_k: &ones_hd,
                    s_msa: &sm, gate_msa: &gm, s_mlp: &sm, gate_mlp: &gm,
                    wq: idx("b.q"), wk: idx("b.k"), wv: idx("b.v"), wo: idx("b.o"),
                    w1: idx("b.w1"), w3: idx("b.w3"), w2: idx("b.w2"),
                    q4tp: true,
                    resident_in: resident && bi > 0,
                    resident_out: resident && bi < 31,
                };
                if !cortiq_engine::gpu::dit_block_seg(&model, &a, &segs, &mut x) {
                    println!("lumina dit_block_seg declined at {res}² batch {batch} (block {bi})");
                    return;
                }
            }
            let e = t.elapsed().as_secs_f64();
            println!("  lumina step {step}: {e:.3} s{}", if step == 0 { " (warm-up)" } else { "" });
            if step > 0 {
                times.push(e);
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n_img_p = n_img.div_ceil(32) * 32;
        let fl = step_flops(n_img_p, &caps, &d);
        let med = times[times.len() / 2];
        println!(
            "lumina-path baseline {res}² batch {batch} (32 blocks, resident={resident}): median {med:.3} s/step, {:.1} TFLOPS effective",
            fl / med / 1e12
        );
    }

    pub fn main() {
        // SAFETY: set before any thread or GPU init.
        unsafe {
            std::env::set_var("CMF_GPU", "1");
        }
        let args: Vec<String> = std::env::args().skip(1).collect();
        let rest = if args.len() > 1 { &args[1..] } else { &[][..] };
        match args.first().map(|s| s.as_str()) {
            Some("info") | None => cmd_info(),
            Some("existing") => cmd_existing(rest),
            Some("mm") => cmd_mm(rest),
            Some("prec") => cmd_prec(rest),
            Some("flash") => cmd_flash(rest),
            Some("blockcheck") => cmd_blockcheck(rest),
            Some("step") => cmd_step(rest),
            Some("lumina") => cmd_lumina(rest),
            Some(x) => eprintln!("unknown subcommand {x}"),
        }
    }

}

fn main() {
    #[cfg(feature = "gpu")]
    imp::main();
    #[cfg(not(feature = "gpu"))]
    eprintln!("zimage_gemmbench needs --features gpu");
}
