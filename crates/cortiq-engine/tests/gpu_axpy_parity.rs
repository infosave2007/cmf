//! `axpy` on the device against the two lines of arithmetic it stands for.
//!
//! This kernel had no test. Its uniform was written `[n, 0, 0, w]` while the
//! shader declared `{ w: f32, n: u32, … }`, so the kernel read `n` as zero,
//! every invocation returned at the bounds check, and the op did NOTHING
//! wherever the device ran it — the indexer's score scaling and the
//! overlapping compressor's position bias among them. Nothing caught it,
//! because both sides of the usual comparison run the same GPU path.

#![cfg(feature = "gpu")]

use cortiq_engine::gpu_wgpu;

fn near(got: &[f32], want: &[f32], what: &str) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5 * w.abs().max(1.0),
            "{what}: [{i}] GPU {g} vs CPU {w}"
        );
    }
}

#[test]
fn device_axpy_matches_the_cpu() {
    unsafe { std::env::set_var("CMF_GPU", "wgpu") };
    if !gpu_wgpu::enabled() {
        eprintln!("пропуск: нет адаптера wgpu");
        return;
    }
    // Not a round count: the kernel is 256 wide and the tail is where a
    // bounds check either works or reads past the end.
    let n = 700usize;
    let x: Vec<f32> = (0..n + 64).map(|i| ((i * 17 % 23) as f32) * 0.125 - 1.0).collect();
    let y0: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32) * 0.25 - 1.5).collect();
    let w = 0.375f32;

    // accumulate
    let mut got = y0.clone();
    assert!(gpu_wgpu::axpy_for_test(&x, &mut got, w, false, 0), "устройство отказало");
    let want: Vec<f32> = y0.iter().zip(&x).map(|(y, xv)| y + w * xv).collect();
    near(&got, &want, "y += w·x");

    // assign, which is what replaces a zero-fill plus an accumulate
    let mut got = y0.clone();
    assert!(gpu_wgpu::axpy_for_test(&x, &mut got, w, true, 0));
    let want: Vec<f32> = x[..n].iter().map(|xv| w * xv).collect();
    near(&got, &want, "y = w·x");

    // strided source, which is what replaces a copy of one window slot
    let soff = 37usize;
    let mut got = y0.clone();
    assert!(gpu_wgpu::axpy_for_test(&x, &mut got, 1.0, false, soff));
    let want: Vec<f32> = y0
        .iter()
        .enumerate()
        .map(|(i, y)| y + x[soff + i])
        .collect();
    near(&got, &want, "y += x[soff..]");
}
