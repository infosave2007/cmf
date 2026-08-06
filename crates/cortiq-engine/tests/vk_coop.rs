//! The native Vulkan lane against the format's own definition.
//!
//! The kernel runs on the card's matrix units and works in f16, so it is
//! NOT bit-equal to the scalar path — the point of the test is that it is
//! the same arithmetic within what half precision costs, and that it never
//! quietly returns something plausible instead. A wrong GEMM here would
//! look like a model that answers a question it was not asked.
//!
//! Skips itself, loudly, wherever the extension is not there:
//! `cargo test -p cortiq-engine --release --features gpu --test vk_coop -- --nocapture`

#![cfg(all(
    feature = "gpu",
    any(target_os = "linux", target_os = "windows", target_os = "android")
))]

use cortiq_core::quant::{
    GROUP_SIZE, dequant_q4tp, f32_to_f16, q4tp_code_stride, q4tp_put_code, q4tp_sections,
};

/// Random nibbles plus a per-row ladder whose span varies row to row, so
/// the 5-bit codes cover their range rather than clustering on one rung.
fn synth(rows: usize, cols: usize) -> Vec<u8> {
    let gpr = cols / GROUP_SIZE;
    let stride = q4tp_code_stride(gpr);
    let (params_off, codes_off, _) = q4tp_sections(rows, cols);
    let mut b = vec![0u8; codes_off + rows * stride];
    for r in 0..rows {
        for g in 0..gpr {
            let t = (r * gpr + g) * 16;
            for k in 0..16 {
                b[t + k] = ((r * 31 + g * 7 + k * 13) % 251) as u8;
            }
        }
        let p = params_off + r * 4;
        b[p..p + 2].copy_from_slice(&f32_to_f16(-6.0 - 0.03 * (r % 17) as f32).to_le_bytes());
        b[p + 2..p + 4].copy_from_slice(&f32_to_f16(0.01 + 0.004 * (r % 11) as f32).to_le_bytes());
        let crow = &mut b[codes_off + r * stride..codes_off + (r + 1) * stride];
        for g in 0..gpr {
            q4tp_put_code(crow, g, (r * 5 + g * 3) % 32);
        }
    }
    b
}

#[test]
fn vk_coop_matmat_matches_dequant_reference() {
    if !cortiq_engine::vulkan::available() {
        eprintln!("no f16 16x16 cooperative matrices here — skipping");
        return;
    }
    // 320 rows and 132 tokens are deliberately not multiples of the 64-wide
    // tile: an edge the kernel masks wrongly is exactly the bug that hides
    // until someone renders at an unusual size.
    for &(rows, cols, b) in &[(256usize, 512usize, 48usize), (320, 256, 132)] {
        let payload = synth(rows, cols);
        let mut w = vec![0f32; rows * cols];
        dequant_q4tp(&payload, rows, cols, &mut w);
        let xs: Vec<f32> = (0..b * cols)
            .map(|i| ((i * 29 + 13) % 103) as f32 / 103.0 - 0.5)
            .collect();
        let mut got = vec![0f32; b * rows];
        assert!(
            cortiq_engine::vulkan::q4tp_matmat(
                (0, rows * 1000 + cols),
                &payload,
                &xs,
                b,
                rows,
                cols,
                &mut got,
            ),
            "the lane reported itself available and then refused a well-formed GEMM"
        );
        let (mut worst, mut at) = (0f32, 0usize);
        for t in 0..b {
            let x = &xs[t * cols..(t + 1) * cols];
            for r in 0..rows {
                let want: f32 = (0..cols).map(|c| w[r * cols + c] * x[c]).sum();
                let mag: f32 = (0..cols).map(|c| (w[r * cols + c] * x[c]).abs()).sum();
                let e = (got[t * rows + r] - want).abs() / mag.max(1e-6);
                if e > worst {
                    worst = e;
                    at = t * rows + r;
                }
            }
        }
        // f16 carries eleven bits of mantissa and the operands are rounded
        // to it before the multiply; the accumulator is f32. Against the
        // sum of magnitudes that lands near 1e-3, where the scalar fp32
        // path sits near 1e-7 — the price of the matrix units, and small
        // against 4-bit weights.
        println!("{rows}x{cols} b={b}: worst relative error {worst:.3e} at cell {at}");
        assert!(
            worst < 4e-3,
            "{rows}x{cols} b={b}: cooperative GEMM off by {worst:.3e} of the row's magnitude"
        );
    }
}
