//! Z-Image wgpu fast-path kernels against f64 host references (plan WP2).
//! Needs a Vulkan device with cooperative matrices; skips (passes) without.
//!
//! ```text
//! XDG_RUNTIME_DIR=/tmp CMF_GPU=1 flock /root/gpu.lock \
//!   cargo test --release -p cortiq-engine --features gpu --test zimage_wgpu
//! ```
#![cfg(feature = "gpu")]

use cortiq_core::quant::{f16_to_f32, f32_to_f16};
use cortiq_engine::gpu_wgpu::zimage::{self as zi, bench, Epi, FlashCfg, MmCfg};

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn device() -> bool {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    bench::info().is_some_and(|s| s.contains("zi fast path usable: true"))
}

fn rel(got: &[f32], want: &[f64]) -> f64 {
    let (mut e, mut r) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        e += (*g as f64 - w).powi(2);
        r += w * w;
    }
    (e / r).sqrt()
}

#[test]
fn zi_mm_epilogues_match_f64() {
    if !device() {
        eprintln!("skip: no coop device");
        return;
    }
    let (m, k, n) = (96usize, 512usize, 256usize);
    let mut rng = Rng(7);
    let act: Vec<u16> = (0..m * k).map(|_| f32_to_f16(rng.uni() * 2.0)).collect();
    let pl: Vec<u16> = (0..n * k).map(|_| f32_to_f16(rng.uni() * 0.08)).collect();
    let a: Vec<f64> = act.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let w: Vec<f64> = pl.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let dot = |r: usize, row: usize| -> f64 { (0..k).map(|j| a[r * k + j] * w[row * k + j]).sum() };
    for epi in [Epi::F32, Epi::F16, Epi::SwiGlu] {
        let cfg = zi::default_cfg(epi);
        let out = bench::mm_run(cfg, m, k, n, &act, &pl).expect("zi_mm");
        let ncol = if epi == Epi::SwiGlu { n / 2 } else { n };
        let want: Vec<f64> = (0..m)
            .flat_map(|r| {
                let dot = &dot;
                (0..ncol).map(move |j| match epi {
                    Epi::SwiGlu => {
                        let g = dot(r, zi::w13_plane_row(j, false));
                        let u = dot(r, zi::w13_plane_row(j, true));
                        g / (1.0 + (-g).exp()) * u
                    }
                    _ => dot(r, j),
                })
            })
            .collect();
        let e = rel(&out[..m * ncol], &want);
        let bar = if epi == Epi::F32 { 1e-4 } else { 1e-3 };
        assert!(e < bar, "{epi:?}: rel {e:.2e} ≥ {bar:.0e}");
    }
    let _ = MmCfg::new(128, 128, 32, 2, 2, Epi::F32);
}

#[test]
fn zi_flash_matches_f64_with_segments_and_rescales() {
    if !device() {
        eprintln!("skip: no coop device");
        return;
    }
    let nh = 2usize;
    let ld = 3 * nh * 128;
    let segs = [(0usize, 96usize), (96, 64)];
    let m = 160;
    let mut rng = Rng(11);
    // keys grow along the segment: the running max climbs → lazy rescales run
    let qkv: Vec<u16> = (0..m * ld)
        .map(|i| {
            let (row, col) = (i / ld, i % ld);
            let grow = if (nh * 128..2 * nh * 128).contains(&col) { 1.0 + (row % 96) as f32 / 12.0 } else { 1.0 };
            f32_to_f16(rng.uni() * 1.7 * grow)
        })
        .collect();
    let g = |row: usize, col: usize| f16_to_f32(qkv[row * ld + col]) as f64;
    for f in [zi::default_flash(), FlashCfg { nw: 4, bc: 32 }] {
        let out = bench::flash_run(f, nh, &qkv, &segs).expect("zi_flash");
        let mut got = Vec::new();
        let mut want = Vec::new();
        for &(off, len) in &segs {
            for h in 0..nh {
                for qi in [0, 1, 15, 16, 33, len - 1] {
                    let sc: Vec<f64> = (0..len)
                        .map(|j| (0..128).map(|d| g(off + qi, h * 128 + d) * g(off + j, nh * 128 + h * 128 + d)).sum::<f64>() / (128f64).sqrt())
                        .collect();
                    let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
                    let e: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let l: f64 = e.iter().sum();
                    for d in 0..128 {
                        want.push((0..len).map(|j| e[j] * g(off + j, 2 * nh * 128 + h * 128 + d)).sum::<f64>() / l);
                        got.push(out[(off + qi) * nh * 128 + h * 128 + d]);
                    }
                }
            }
        }
        let e = rel(&got, &want);
        assert!(e < 1e-3, "{f:?}: rel {e:.2e}");
    }
}

/// The flushed f16-accumulate arm (`MmCfg::acc16`, off by default: slower
/// and 20–50× less precise than the f32 arm on the 3090, vk2) stays
/// correct: every epilogue within the f16-accumulation error.
#[test]
fn zi_mm_acc16_arm_is_close() {
    if !device() {
        eprintln!("skip: no coop device");
        return;
    }
    let (m, k, n) = (96usize, 512usize, 256usize);
    let mut rng = Rng(9);
    let act: Vec<u16> = (0..m * k).map(|_| f32_to_f16(rng.uni() * 2.0)).collect();
    let pl: Vec<u16> = (0..n * k).map(|_| f32_to_f16(rng.uni() * 0.08)).collect();
    let a: Vec<f64> = act.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let w: Vec<f64> = pl.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let dot = |r: usize, row: usize| -> f64 { (0..k).map(|j| a[r * k + j] * w[row * k + j]).sum() };
    for epi in [Epi::F32, Epi::F16, Epi::SwiGlu] {
        for (tile, flush) in [((128, 128, 64, 2, 2), 1), ((128, 64, 64, 2, 2), 1), ((128, 128, 32, 2, 2), 2)] {
            let cfg = MmCfg { acc16: flush, ..MmCfg::new(tile.0, tile.1, tile.2, tile.3, tile.4, epi) };
            assert!(cfg.valid(), "{cfg:?}");
            let out = bench::mm_run(cfg, m, k, n, &act, &pl).expect("zi_mm acc16");
            let ncol = if epi == Epi::SwiGlu { n / 2 } else { n };
            let want: Vec<f64> = (0..m)
                .flat_map(|r| {
                    let dot = &dot;
                    (0..ncol).map(move |j| match epi {
                        Epi::SwiGlu => {
                            let g = dot(r, zi::w13_plane_row(j, false));
                            let u = dot(r, zi::w13_plane_row(j, true));
                            g / (1.0 + (-g).exp()) * u
                        }
                        _ => dot(r, j),
                    })
                })
                .collect();
            let e = rel(&out[..m * ncol], &want);
            assert!(e < 2e-3, "{epi:?} {cfg:?}: rel {e:.2e}");
        }
    }
}

/// A segment's rows do not depend on what lies past its end. The last
/// query block of a segment whose length is not a multiple of nw·16 also
/// holds rows past the end (the next CFG item, or pad rows); before vk2
/// those rows voted in the workgroup-wide lazy-rescale decision and moved
/// the live rows' f16 rounding (a CFG pair's first item 2.2e-4 away from
/// its single forward). Bit-exact now.
#[test]
fn zi_flash_rows_past_a_segment_do_not_vote() {
    if !device() {
        eprintln!("skip: no coop device");
        return;
    }
    let nh = 2usize;
    let ld = 3 * nh * 128;
    let (l0, l1) = (96usize, 64usize);
    let m = l0 + l1;
    let mut rng = Rng(21);
    // item 1: queries 16× larger and keys that keep growing, so its rows
    // raise their running max by far more than the rescale threshold
    let full: Vec<u16> = (0..m * ld)
        .map(|i| {
            let (row, col) = (i / ld, i % ld);
            let s = if col < nh * 128 && row >= l0 {
                16.0
            } else if (nh * 128..2 * nh * 128).contains(&col) {
                1.0 + (row % l0) as f32 / 8.0
            } else {
                1.0
            };
            f32_to_f16(rng.uni() * 1.7 * s)
        })
        .collect();
    let mut alone = full.clone();
    for v in &mut alone[l0 * ld..] {
        *v = 0;
    }
    for f in [zi::default_flash(), FlashCfg { nw: 4, bc: 32 }] {
        let a = bench::flash_run(f, nh, &alone, &[(0, l0)]).expect("zi_flash single");
        let b = bench::flash_run(f, nh, &full, &[(0, l0), (l0, l1)]).expect("zi_flash pair");
        let w = nh * 128;
        let diff = (0..l0 * w).filter(|&i| a[i].to_bits() != b[i].to_bits()).count();
        assert_eq!(diff, 0, "{f:?}: {diff} of {} item-0 outputs moved with item 1 beside it", l0 * w);
    }
}
