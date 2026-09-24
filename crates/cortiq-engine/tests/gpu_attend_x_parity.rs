//! The per-layer-geometry attention kernels (MiMo-V2 on the wgpu graphs),
//! device against a direct CPU softmax: V narrower than the head, grouped
//! KV heads, a sliding window over a ring mirror that has wrapped several
//! times, learned sink logits, and the split-K part/merge past one chunk.
//!
//!     CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu --release \
//!         --test gpu_attend_x_parity -- --nocapture

#[cfg(feature = "gpu")]
fn val(i: usize, salt: usize) -> f32 {
    (((i * 7919 + salt * 104_729) % 1009) as f32 / 1009.0 - 0.5) * 1.6
}

/// Softmax over positions [first, npos) of kv head h / hpk, plus the sink
/// column (max includes it, the denominator gets exp(sink − max), no value).
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn cpu_attend(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sink: Option<&[f32]>,
    window: Option<usize>,
    nh: usize,
    nkv: usize,
    hd: usize,
    dv: usize,
    scale: f32,
) -> Vec<f32> {
    let npos = k.len() / (nkv * hd);
    let first = window.map_or(0, |w| npos.saturating_sub(w));
    let hpk = nh / nkv;
    let mut out = vec![0f32; nh * dv];
    for h in 0..nh {
        let g = h / hpk;
        let scores: Vec<f64> = (first..npos)
            .map(|p| {
                let kr = &k[(p * nkv + g) * hd..(p * nkv + g + 1) * hd];
                let qr = &q[h * hd..(h + 1) * hd];
                qr.iter()
                    .zip(kr)
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum::<f64>()
                    * scale as f64
            })
            .collect();
        let s = sink.map(|s| s[h] as f64);
        let m = scores
            .iter()
            .cloned()
            .chain(s)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut l = s.map_or(0.0, |s| (s - m).exp());
        let mut acc = vec![0f64; dv];
        for (j, p) in (first..npos).enumerate() {
            let w = (scores[j] - m).exp();
            l += w;
            let vr = &v[(p * nkv + g) * dv..(p * nkv + g + 1) * dv];
            for d in 0..dv {
                acc[d] += w * vr[d] as f64;
            }
        }
        for d in 0..dv {
            out[h * dv + d] = (acc[d] / l) as f32;
        }
    }
    out
}

#[cfg(feature = "gpu")]
#[test]
fn attend_x_matches_the_cpu_softmax_needs_cmf_gpu() {
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            cortiq_engine::gpu_wgpu::skip_or_fail(module_path!());
            return;
        }
        Some(false) => panic!("wgpu selected but the context did not come up"),
        Some(true) => {}
    }
    // (nh, nkv, hd, dv, npos, window, sink, split)
    let cases: &[(usize, usize, usize, usize, usize, Option<usize>, bool, bool)] = &[
        // MiMo-V2 sliding layer: 8 KV heads, 192/128, window 128, ring of
        // 256 rows wrapped by 700 positions, sinks.
        (16, 8, 192, 128, 700, Some(128), true, false),
        // the toy's sliding layer: window 8 (ring 16), 37 positions
        (8, 4, 48, 32, 37, Some(8), true, false),
        // a sliding layer before its window fills
        (8, 4, 48, 32, 5, Some(8), true, false),
        // MiMo-V2 full layer: 4 KV heads under 64 Q heads, one chunk
        (64, 4, 192, 128, 200, None, false, false),
        // full layer past one chunk: split-K part + merge, and the single
        // workgroup walking three chunks, on the same rows
        (64, 4, 192, 128, 600, None, false, true),
        (64, 4, 192, 128, 600, None, false, false),
        // sinks through the merge (not a MiMo shape: coverage)
        (8, 2, 64, 64, 520, None, true, true),
    ];
    for &(nh, nkv, hd, dv, npos, window, with_sink, split) in cases {
        let q: Vec<f32> = (0..nh * hd).map(|i| val(i, 1) * 0.5).collect();
        let k: Vec<f32> = (0..npos * nkv * hd).map(|i| val(i, 2)).collect();
        let v: Vec<f32> = (0..npos * nkv * dv).map(|i| val(i, 3)).collect();
        let sink: Vec<f32> = (0..nh).map(|h| val(h, 4) * 4.0).collect();
        let sink = with_sink.then_some(sink.as_slice());
        let scale = (hd as f32).powf(-0.5);
        let want = cpu_attend(&q, &k, &v, sink, window, nh, nkv, hd, dv, scale);
        let mut got = vec![0f32; nh * dv];
        assert!(
            cortiq_engine::gpu_wgpu::attend_x_for_test(
                &q, &k, &v, sink, window, nh, nkv, hd, dv, scale, split, &mut got
            ),
            "device or attend-x module missing"
        );
        let num: f64 = got
            .iter()
            .zip(&want)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum();
        let den: f64 = want
            .iter()
            .map(|a| (*a as f64).powi(2))
            .sum::<f64>()
            .max(1e-30);
        let rel = (num / den).sqrt();
        println!(
            "nh {nh} nkv {nkv} hd {hd} dv {dv} npos {npos} window {window:?} sink {with_sink} split {split}: rel {rel:.3e}"
        );
        assert!(
            rel < 1e-5,
            "device attend-x diverged from the CPU: {rel:.3e}"
        );
    }
}
