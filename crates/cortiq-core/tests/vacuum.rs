//! Known-answer tests for the quantizers — payloads whose decoded value is
//! known exactly, digit for digit, before anything runs.
//!
//! `roundtrip.rs` next door measures dequantization ACCURACY: encode
//! random weights, decode, assert the error is small. That catches a
//! broken codec but not a shifted one — an off-by-one in a scale ladder
//! moves every weight by one geometric step, which is a small relative
//! error everywhere and passes an accuracy bound comfortably.
//!
//! These assert equality instead, on inputs where equality is derivable.

use cortiq_core::quant::{
    GROUP_SIZE, Q2TP_CHUNK, dequant_q2tp, q2tp_ladder, q2tp_sections, q4tp_ladder,
};

/// f16 bit patterns for a ladder with a base of 2^-3 and a step of 2^0.25
/// — an ordinary-looking ladder, chosen so the rungs are all distinct and
/// none of them is accidentally 1.0.
const LO_M3: [u8; 2] = [0x00, 0xC2]; // -3.0
const ST_Q: [u8; 2] = [0x00, 0x34]; //  0.25

fn params() -> Vec<u8> {
    let mut p = Vec::with_capacity(4);
    p.extend_from_slice(&LO_M3);
    p.extend_from_slice(&ST_Q);
    p
}

/// The two-bit ladder is the four-bit ladder shifted down one rung, with
/// rung 0 spent on the exact zero.
///
/// This is the whole design of `q2tp` in one sentence, and it is the
/// sentence a future edit is most likely to break: the ±0.5/±1.5 code
/// grid cannot spell zero, so a pruned or masked group would come back
/// as noise unless the scale itself is allowed to be zero. Asserting the
/// shift by equality means an edit that renumbers the rungs fails here
/// rather than in a render three stacks downstream, where it looks like
/// a quality regression.
#[test]
fn the_two_bit_ladder_is_the_four_bit_one_shifted_one_rung() {
    let p = params();
    let four = q4tp_ladder(&p, 0);
    let two = q2tp_ladder(&p, 0);

    assert_eq!(two[0], 0.0, "rung 0 of the two-bit ladder is not zero");
    assert_eq!(
        two[0].to_bits(),
        0,
        "rung 0 is a negative or subnormal zero"
    );
    for c in 0..31 {
        assert_eq!(
            two[c + 1],
            four[c],
            "rung {} of the two-bit ladder is not rung {c} of the four-bit one",
            c + 1
        );
    }
    // And the four-bit ladder itself is geometric from its stated base:
    // t[0] = 2^lo, t[c] = t[c-1]·2^step. Derivable, so assert it.
    assert!(
        (four[0] - 0.125).abs() < 1e-7,
        "base rung is {} not 2^-3",
        four[0]
    );
    let ratio = 2f32.powf(0.25);
    for c in 1..32 {
        let want = four[c - 1] * ratio;
        assert!(
            (four[c] - want).abs() <= want * 1e-5,
            "rung {c}: {} is not {want} (ratio broke)",
            four[c]
        );
    }
}

/// A group sitting on rung 0 decodes to exact zero — every weight, every
/// code, bit-identical.
///
/// Not "small". Zero. A masked expert or a pruned block that comes back
/// as ±1e-9 is still wrong: it defeats the sparsity it was quantized to
/// express, and nothing downstream will ever tell you, because 1e-9
/// looks exactly like a well-quantized zero.
#[test]
fn a_group_on_rung_zero_decodes_to_exact_zero() {
    let (rows, cols) = (1usize, 2 * GROUP_SIZE);
    let gpr = cols / GROUP_SIZE;
    let (params_off, codes_off, stride) = q2tp_sections(rows, cols);
    let mut bytes = vec![0u8; codes_off + rows * stride];

    // 2-bit codes: 0b11_10_01_00 puts one of each code in every byte, so
    // the zero below has to survive all four, not just the code that
    // happens to sit at the grid's centre (there isn't one).
    for b in bytes[..rows * gpr * Q2TP_CHUNK].iter_mut() {
        *b = 0xE4;
    }
    bytes[params_off..params_off + 4].copy_from_slice(&params());
    // Group 0 → scale code 0 (the exact zero); group 1 → code 7.
    // Codes are 5 bits, LSB-first: g=0 occupies bits 0..5, g=1 bits 5..10.
    bytes[codes_off] = 7 << 5;
    bytes[codes_off + 1] = 7 >> 3;

    let mut dst = vec![f32::NAN; rows * cols];
    dequant_q2tp(&bytes, rows, cols, &mut dst);

    for (i, &v) in dst[..GROUP_SIZE].iter().enumerate() {
        assert_eq!(v, 0.0, "weight {i} of the rung-0 group decoded to {v}");
        assert!(v.is_finite(), "weight {i} of the rung-0 group is {v}");
    }

    // Recorded rather than required: the decode is `(code − 1.5)·s`, so
    // with s = 0 the two codes below the grid's centre produce NEGATIVE
    // zero. It compares equal to +0.0 and adds and multiplies
    // identically, so nothing downstream can tell — but it is a fact
    // about the payload nobody had written down, and pinning it here
    // means a future edit that changes the sign pattern is a decision
    // somebody made rather than one that happened. If a consumer ever
    // needs +0.0 (raw-bit hashing, sparsity detection by bit pattern),
    // this is the line that says where it comes from.
    for (i, &v) in dst[..GROUP_SIZE].iter().enumerate() {
        let want_negative = i % 4 < 2;
        assert_eq!(
            v.is_sign_negative(),
            want_negative,
            "weight {i} of the rung-0 group has the wrong signed zero"
        );
    }

    // The neighbouring group is on a live rung, so its 32 weights are
    // the code grid times that rung — also exactly predictable, which
    // rules out "everything decoded to zero" passing the test above.
    let s = q2tp_ladder(&params(), 0)[7];
    let grid = [-1.5f32, -0.5, 0.5, 1.5];
    for (i, &v) in dst[GROUP_SIZE..].iter().enumerate() {
        let want = grid[i % 4] * s;
        assert_eq!(v, want, "weight {i} of the live group: {v} ≠ {want}");
    }
}

/// Two rows, one zeroed and one not, decode independently.
///
/// Rows carry their own params and their own code slice precisely so
/// they can be decoded in parallel; a stride computed once and reused
/// across rows is the classic way that breaks, and it shows up as the
/// second row wearing the first row's scales.
#[test]
fn a_zeroed_row_does_not_disturb_its_neighbour() {
    let (rows, cols) = (2usize, 2 * GROUP_SIZE);
    let gpr = cols / GROUP_SIZE;
    let (params_off, codes_off, stride) = q2tp_sections(rows, cols);
    let mut bytes = vec![0u8; codes_off + rows * stride];
    for b in bytes[..rows * gpr * Q2TP_CHUNK].iter_mut() {
        *b = 0xE4;
    }
    for r in 0..rows {
        bytes[params_off + r * 4..params_off + r * 4 + 4].copy_from_slice(&params());
    }
    // Row 0: both groups on rung 0. Row 1: both on rung 12.
    bytes[codes_off] = 0;
    bytes[codes_off + 1] = 0;
    bytes[codes_off + stride] = 12 | (12 << 5);
    bytes[codes_off + stride + 1] = 12 >> 3;

    let mut dst = vec![f32::NAN; rows * cols];
    dequant_q2tp(&bytes, rows, cols, &mut dst);

    assert!(
        dst[..cols].iter().all(|&v| v == 0.0 && v.is_finite()),
        "the zeroed row is not zero"
    );
    // Both rows carry identical params, so row 0's ladder is row 1's.
    let s = q2tp_ladder(&params(), 0)[12];
    let grid = [-1.5f32, -0.5, 0.5, 1.5];
    for (i, &v) in dst[cols..].iter().enumerate() {
        assert_eq!(v, grid[i % 4] * s, "row 1 weight {i} took the wrong scale");
    }
}
