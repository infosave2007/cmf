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
