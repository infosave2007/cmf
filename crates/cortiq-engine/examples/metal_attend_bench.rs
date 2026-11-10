//! GPU time of one decode attend on Metal for a head layout, by kernel:
//! 0 per-head `gqa_attend`, 1 split-K `gqa_attend_blk` + combine, 2 the
//! per-key `gqa_attend_kb` (f32 mirror) — the decode graph's choices.
//! cargo run --release -p cortiq-engine --example metal_attend_bench [-- NH NKV HD]
fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|v| v.parse().ok()).collect();
    let (nh, nkv, hd) = (*a.first().unwrap_or(&14), *a.get(1).unwrap_or(&2), *a.get(2).unwrap_or(&64));
    let _ = cortiq_engine::gpu_metal::enabled();
    println!("heads {nh}/{nkv}, head dim {hd}: GPU us per attend");
    println!("{:>6} {:>10} {:>10} {:>10}", "pos", "per-head", "split-K", "per-key");
    for n in [32usize, 64, 128, 256, 512, 1024, 2048] {
        let t: Vec<String> = (0..3u32)
            .map(|w| {
                // The GPU clocks up under load: warm it, then the best of 9 runs.
                for _ in 0..20 {
                    let _ = cortiq_engine::gpu_metal::attend_kernel_bench(nh, nkv, hd, n, w, 256);
                }
                let best = (0..9)
                    .filter_map(|_| cortiq_engine::gpu_metal::attend_kernel_bench(nh, nkv, hd, n, w, 256))
                    .fold(f64::MAX, f64::min);
                if best < f64::MAX { format!("{best:10.1}") } else { format!("{:>10}", "-") }
            })
            .collect();
        println!("{n:6} {}", t.join(" "));
    }
}
