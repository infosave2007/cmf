//! Gate for the MiniMax-H3 latent upscaler port: our forward against the
//! node's own torch module on the same weights and the same input.
//!
//! ```sh
//! python3 tools/mmh3_ups_oracle.py ups3d.safetensors oracle.safetensors
//! cargo run --release --example mmh3_ups_parity -- ups3d.safetensors oracle.safetensors
//! ```
//!
//! The oracle carries `x` (the normalized input the reference was handed)
//! and `y` (what it produced), both `[c, t, h, w]`. The port normalizes
//! internally, so the comparison feeds `x` back through the inverse of that
//! normalization — otherwise the two are measuring different functions.

use cortiq_engine::mmh3ups::{LATENT_MEAN, LATENT_STD, LatentUpscaler, Vol};
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let weights = args
        .next()
        .expect("usage: mmh3_ups_parity <weights> <oracle>");
    let oracle = args
        .next()
        .expect("usage: mmh3_ups_parity <weights> <oracle>");

    let ups = LatentUpscaler::load(Path::new(&weights)).expect("load upscaler");
    let st = cortiq_engine::mmh3ups::read_oracle(Path::new(&oracle)).expect("load oracle");
    let (xs, xd) = st.get("x").expect("oracle has no x").clone();
    let (ys, yd) = st.get("y").expect("oracle has no y").clone();
    assert_eq!(xs.len(), 4, "x is [c, t, h, w]");
    let (c, t, h, w) = (xs[0], xs[1], xs[2], xs[3]);
    let (oh, ow) = (ys[2], ys[3]);

    // The oracle's `x` is already normalized; `upscale` normalizes what it
    // is given, so undo it first and let the port do its own.
    let n = t * h * w;
    let mut raw = xd.clone();
    for ci in 0..c {
        for i in 0..n {
            raw[ci * n + i] = raw[ci * n + i] * LATENT_STD[ci] + LATENT_MEAN[ci];
        }
    }
    let z = Vol {
        c,
        t,
        h,
        w,
        data: raw,
    };
    let started = std::time::Instant::now();
    let got = ups.upscale(&z, oh, ow, None);
    let secs = started.elapsed().as_secs_f64();

    assert_eq!(
        got.data.len(),
        yd.len(),
        "shape mismatch: {:?} vs {ys:?}",
        (got.c, got.t, got.h, got.w)
    );
    // The oracle is the bare network: it neither normalizes its input nor
    // denormalizes its output. `upscale` does both, because every caller
    // holds a raw sampler latent — so put ours back on the network's own
    // scale before diffing, or the comparison measures the statistics
    // rather than the port.
    let mut got = got;
    let on = got.t * got.h * got.w;
    for ci in 0..got.c {
        for i in 0..on {
            got.data[ci * on + i] = (got.data[ci * on + i] - LATENT_MEAN[ci]) / LATENT_STD[ci];
        }
    }
    let mut worst = 0f64;
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, r) in got.data.iter().zip(&yd) {
        let d = (*g as f64 - *r as f64).abs();
        worst = worst.max(d);
        num += d * d;
        den += (*r as f64) * (*r as f64);
    }
    let rel = (num / den.max(1e-30)).sqrt();
    println!(
        "upscaler parity  {}x{} -> {}x{}  worst {:.3e}  rel rms {:.3e}  in {:.1}s",
        h, w, oh, ow, worst, rel, secs
    );
    // The reference runs in f32 here, so anything above a loose float
    // tolerance is a port bug, not precision.
    assert!(
        rel < 2e-3,
        "relative rms {rel:.3e} is a port bug, not rounding"
    );
}
