//! Does the fused Metal path compute the same thing as the host path?
//!
//! The branch rides in the base GEMM's command buffer, over the activation
//! buffer that GEMM already pre-scaled, and accumulates into the output it
//! already wrote. Three places to get wrong, none of which show up as a
//! crash — so they get an oracle instead of an argument.
//!
//!   cargo run --release -p cortiq-engine --example lora_fused_parity -- model.cmf

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("lora_fused_parity: the fused branch is a Metal path");
}

#[cfg(target_os = "macos")]
fn main() {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    let path = std::env::args()
        .nth(1)
        .expect("usage: lora_fused_parity <model.cmf>");
    let model = std::sync::Arc::new(cortiq_core::CmfModel::open_sharded(&path).expect("open"));

    // A real q4tp projection of the DiT, at a batch the gate accepts.
    let (idx, rows, cols) = model
        .tensors
        .iter()
        .enumerate()
        .find_map(|(i, e)| {
            (e.dtype == cortiq_core::types::TensorDtype::Q4TiledP
                && e.shape.len() == 2
                && e.shape[0] == 4096
                && e.shape[1] == 4096
                && e.name.contains("transformer_blocks."))
            .then(|| (i, e.shape[0], e.shape[1]))
        })
        .expect("no 4096x4096 q4tp projection in this file");
    println!("tensor: {}", model.tensors[idx].name);

    let (n, rank) = (384usize, 128usize);
    let x: Vec<f32> = (0..n * cols)
        .map(|i| ((i % 977) as f32 * 0.013).sin() * 0.7)
        .collect();
    let a: Vec<f32> = (0..rank * cols)
        .map(|i| ((i % 613) as f32 * 0.021).cos() * 0.05)
        .collect();
    let b: Vec<f32> = (0..rows * rank)
        .map(|i| ((i % 419) as f32 * 0.017).sin() * 0.05)
        .collect();
    let scale = 0.8f32;

    // Host reference: base GEMM on the CPU, then the branch on the CPU.
    let wbytes = model.entry_bytes(&model.tensors[idx]);
    let mut w = vec![0f32; rows * cols];
    cortiq_core::quant::dequant_q4tp(wbytes, rows, cols, &mut w);
    let mut want = vec![0f32; n * rows];
    cortiq_engine::gpu::cpu_scope(|| {
        cortiq_engine::fcd_ops::gemm_nt(&x, &w, &mut want, n, cols, rows, None);
        let mut h = vec![0f32; n * rank];
        cortiq_engine::fcd_ops::gemm_nt(&x, &a, &mut h, n, cols, rank, None);
        let mut d = vec![0f32; n * rows];
        cortiq_engine::fcd_ops::gemm_nt(&h, &b, &mut d, n, rank, rows, None);
        for (w, dv) in want.iter_mut().zip(&d) {
            *w += scale * dv;
        }
    });

    // Fused: one submission, base and branch together.
    let mut got = vec![0f32; n * rows];
    let side = cortiq_engine::gpu_metal::LoraSide {
        a: &a,
        b: &b,
        rank,
        scale,
        id: 7_000_001,
    };
    let ok =
        cortiq_engine::gpu_metal::q4tp_matmat_lora(&model, idx, &x, n, rows, cols, &mut got, &side);
    assert!(ok, "the device declined the shape");

    // Scale the difference by the row's own magnitude: a quiet row and a loud
    // one are not comparable in absolute terms.
    let mut worst = 0f32;
    let mut at = (0usize, 0usize);
    for t in 0..n {
        let mag = want[t * rows..(t + 1) * rows]
            .iter()
            .fold(0f32, |m, v| m.max(v.abs()))
            .max(1e-6);
        for o in 0..rows {
            let rel = (got[t * rows + o] - want[t * rows + o]).abs() / mag;
            if rel > worst {
                worst = rel;
                at = (t, o);
            }
        }
    }
    println!(
        "fused vs host: worst relative {worst:.3e} at row {} col {}  (got {:.5}, want {:.5})",
        at.0,
        at.1,
        got[at.0 * rows + at.1],
        want[at.0 * rows + at.1]
    );

    // The base GEMM alone, to show how much of that residue is the codec's
    // own half-precision staging rather than anything the branch did.
    let mut base_dev = vec![0f32; n * rows];
    cortiq_engine::gpu::q4tp_matmat(&model, idx, &x, n, rows, cols, &mut base_dev);
    let mut base_host = vec![0f32; n * rows];
    cortiq_engine::gpu::cpu_scope(|| {
        cortiq_engine::fcd_ops::gemm_nt(&x, &w, &mut base_host, n, cols, rows, None);
    });
    let mut base_worst = 0f32;
    for t in 0..n {
        let mag = base_host[t * rows..(t + 1) * rows]
            .iter()
            .fold(0f32, |m, v| m.max(v.abs()))
            .max(1e-6);
        for o in 0..rows {
            base_worst =
                base_worst.max((base_dev[t * rows + o] - base_host[t * rows + o]).abs() / mag);
        }
    }
    println!("base GEMM alone, device vs host: worst relative {base_worst:.3e}");
    assert!(
        worst <= base_worst * 3.0 + 1e-3,
        "the branch adds error beyond the base GEMM's own staging"
    );
    println!("OK");
}
