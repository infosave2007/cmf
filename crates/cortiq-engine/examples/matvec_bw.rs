//! Weight-path bandwidth probe (scratch diagnostic, not a product).
//!
//! Two modes, both isolating the weight path from GDN/attention/sampling:
//!
//! - `one <tensor>`: repeat ONE tensor. A single dispatch amortized over a
//!   large matrix — measures the kernel's ceiling. NOTE: tensors small
//!   enough to sit in the SLC report inflated GB/s; only the ~636 MB head
//!   is a trustworthy DRAM number here.
//! - `sweep` (default): walk EVERY 2-D q8_2f tensor once, in directory
//!   order — the real decode access pattern (cold weights, one dispatch
//!   per tensor, ~200 dispatches). This is the honest weight-path number.
//!
//! Usage: cargo run --release --example matvec_bw -- <model.cmf> [sweep|one <tensor>]

use cortiq_engine::pool::Pool;
use cortiq_engine::qtensor::QTensor;
use cortiq_core::TensorDtype;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: matvec_bw <model.cmf> [sweep|one <tensor>]");
    let mode = args.next().unwrap_or_else(|| "sweep".to_string());
    let model = Arc::new(cortiq_core::CmfModel::open(&path).expect("open model"));

    // Real LM activations carry a few heavy channels (>8·rms); measured
    // mean on this model is ~3.7. NOUT models that distribution.
    let nout: usize = std::env::var("NOUT").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let mk_x = |cols: usize| -> Vec<f32> {
        let mut x: Vec<f32> = (0..cols).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
        for k in 0..nout {
            x[k * 37 % cols] = 40.0;
        }
        x
    };

    if mode == "one" {
        let name = args.next().unwrap_or_else(|| "model.embed_tokens.weight".to_string());
        let entry = model.tensor(&name).expect("tensor not found");
        let (rows, cols) = (entry.shape[0], entry.shape[1]);
        let nbytes = entry.nbytes as f64;
        println!("tensor {name}: {rows}x{cols} {:?} = {:.1} MB, NOUT={nout}", entry.dtype, nbytes / 1e6);
        let t = QTensor::from_model(&model, &name).expect("wrap");
        let x = mk_x(cols);
        let mut out = vec![0f32; rows];
        t.matvec(&x, &mut out, None);
        for nt in [1usize, 2, 4, 6, 8, 10] {
            let pool = if nt == 1 { None } else { Some(Pool::new(nt)) };
            let iters = 8;
            t.matvec(&x, &mut out, pool.as_ref());
            let t0 = Instant::now();
            for _ in 0..iters {
                t.matvec(&x, &mut out, pool.as_ref());
            }
            let el = t0.elapsed().as_secs_f64();
            println!("threads={nt:2}  {:6.2} ms/matvec  {:6.1} GB/s (sink {:.3})",
                el / iters as f64 * 1e3, nbytes * iters as f64 / el / 1e9, out[0]);
        }
        return;
    }

    // sweep: every 2-D q8_2f tensor once = one decode's worth of weights.
    let names: Vec<String> = model
        .tensors
        .iter()
        .filter(|t| t.dtype == TensorDtype::Q8_2f && t.shape.len() == 2)
        .map(|t| t.name.clone())
        .collect();
    let total_bytes: f64 = model
        .tensors
        .iter()
        .filter(|t| t.dtype == TensorDtype::Q8_2f && t.shape.len() == 2)
        .map(|t| t.nbytes as f64)
        .sum();
    println!(
        "sweep: {} q8_2f tensors, {:.2} GB total (= weights streamed per decode token), NOUT={nout}",
        names.len(),
        total_bytes / 1e9
    );

    let tensors: Vec<(QTensor, Vec<f32>, Vec<f32>)> = names
        .iter()
        .map(|n| {
            let e = model.tensor(n).unwrap();
            let (rows, cols) = (e.shape[0], e.shape[1]);
            (QTensor::from_model(&model, n).expect("wrap"), mk_x(cols), vec![0f32; rows])
        })
        .collect();
    let mut tensors = tensors;

    // Whole-model residency pass BEFORE any timing: the first touch of a
    // 4.2 GB mmap faults ~260k pages, which would otherwise be charged to
    // whichever thread count happens to run first. REVERSE=1 flips the
    // order as a check that no first-touch cost is left in the table.
    for _ in 0..2 {
        for (t, x, out) in tensors.iter_mut() {
            t.matvec(x, out, None);
        }
    }
    let mut counts = vec![1usize, 2, 4, 6, 8, 10];
    if std::env::var("REVERSE").is_ok() {
        counts.reverse();
    }
    for nt in counts {
        let pool = if nt == 1 { None } else { Some(Pool::new(nt)) };
        // one warm pass, then two measured
        for (t, x, out) in tensors.iter_mut() {
            t.matvec(x, out, pool.as_ref());
        }
        let iters = 2;
        let t0 = Instant::now();
        for _ in 0..iters {
            for (t, x, out) in tensors.iter_mut() {
                t.matvec(x, out, pool.as_ref());
            }
        }
        let el = t0.elapsed().as_secs_f64();
        let per_tok = el / iters as f64;
        println!(
            "threads={nt:2}  {:7.1} ms/sweep  {:6.1} GB/s  -> weight-path-only ceiling {:5.1} tok/s",
            per_tok * 1e3,
            total_bytes * iters as f64 / el / 1e9,
            1.0 / per_tok
        );
    }
}
