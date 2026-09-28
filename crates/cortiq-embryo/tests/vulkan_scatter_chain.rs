//! S6c item 2: the tied-embedding scatter-add (WGSL op 10) as one owner per
//! (first-occurrence row, column) walking a device-built token chain (op 58)
//! instead of a single invocation over rows·d. Gate: dE bit-identical to the
//! legacy serial loop (repeated and masked tokens included) and the measured
//! time at the B8/T512 geometry (rows 4096, hidden 384).
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use cortiq_embryo::metal::{Cmd, GBuf, ctx};
use cortiq_embryo::ops::lcg_vec;

fn tokens(seed: u64, rows: usize, vocab: usize, masked_every: usize) -> Vec<u32> {
    lcg_vec(seed, rows)
        .iter()
        .enumerate()
        .map(|(i, x)| {
            if masked_every > 0 && i % masked_every == 3 {
                u32::MAX
            } else {
                ((x * 0.5 + 0.5) * vocab as f32) as u32 % vocab as u32
            }
        })
        .collect()
}

/// Returns (serial dE, chain dE, serial ms, chain ms) — medians of `reps`.
fn run(rows: usize, d: usize, vocab: usize, masked_every: usize, deoff: usize, seed: u64, reps: usize) -> (Vec<f32>, Vec<f32>, f64, f64) {
    let c = ctx().expect("native Vulkan adapter is required");
    let tok = tokens(seed, rows, vocab, masked_every);
    let dx = lcg_vec(seed + 1, rows * d);
    let de0 = lcg_vec(seed + 2, deoff + vocab * d);
    let gtok = GBuf::from_u32(c, &tok);
    let gdx = GBuf::from_slice(c, &dx);
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let mut t_serial = Vec::new();
    let mut out_serial = Vec::new();
    for _ in 0..reps {
        let gde = GBuf::from_slice(c, &de0);
        let cmd = Cmd::new(c);
        cmd.embed_scatter_add_serial(&gde, deoff, &gtok, &gdx, rows, d);
        t_serial.push(cmd.commit());
        out_serial = gde.to_vec();
    }
    let mut t_chain = Vec::new();
    let mut out_chain = Vec::new();
    for _ in 0..reps {
        let gde = GBuf::from_slice(c, &de0);
        let cmd = Cmd::new(c);
        cmd.embed_scatter_add(&gde, deoff, &gtok, &gdx, rows, d);
        t_chain.push(cmd.commit());
        out_chain = gde.to_vec();
    }
    // CPU reference of the serial semantics (ascending rows, f32)
    let mut want = de0.clone();
    for r in 0..rows {
        if tok[r] != u32::MAX {
            for col in 0..d {
                let i = deoff + tok[r] as usize * d + col;
                want[i] += dx[r * d + col];
            }
        }
    }
    assert_eq!(out_serial, want, "legacy serial scatter drifted from the CPU order");
    (out_serial, out_chain, med(&mut t_serial), med(&mut t_chain))
}

#[test]
fn chain_scatter_is_bit_identical_to_serial_on_tiny() {
    // rows 64, hidden 16, vocab 8 → heavy repeats, every 5th row masked,
    // a nonzero destination offset
    let (a, b, _, _) = run(64, 16, 8, 5, 7, 900, 1);
    assert_eq!(a, b, "chain scatter differs from the serial loop (tiny)");
    // one token everywhere: a single chain of 64 rows
    let (a, b, _, _) = run(64, 16, 1, 0, 0, 901, 1);
    assert_eq!(a, b, "chain scatter differs (single chain)");
    // tiny genome geometry: rows 128, hidden 64, vocab 4096
    let (a, b, _, _) = run(128, 64, 4096, 0, 0, 902, 1);
    assert_eq!(a, b, "chain scatter differs (tiny genome)");
    eprintln!("tiny: chain scatter bit-identical to the serial loop (3 cases)");
}

#[test]
fn chain_scatter_at_b8_t512_is_fast_and_identical() {
    // B8/T512 with the Embryo-0 hidden and vocab
    let (a, b, ts, tc) = run(4096, 384, 32768, 0, 0, 903, 3);
    assert_eq!(a, b, "chain scatter differs (B8/T512)");
    eprintln!("B8/T512 rows 4096 × 384, vocab 32768: serial {ts:.2} ms, chain {tc:.2} ms");
    // pathological repeats: vocab 16 → chains of ~256 rows
    let (a, b, ts2, tc2) = run(4096, 384, 16, 0, 0, 904, 3);
    assert_eq!(a, b, "chain scatter differs (vocab 16)");
    eprintln!("B8/T512 rows 4096 × 384, vocab 16: serial {ts2:.2} ms, chain {tc2:.2} ms");
    assert!(tc <= 10.0, "chain scatter {tc:.2} ms > 10 ms at B8/T512");
}
