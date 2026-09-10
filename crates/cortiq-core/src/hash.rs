//! Per-tensor 64-bit hash — corruption detection for mmap'd weights and
//! backbone-tensor dedup between skill files.
//!
//! Bit-for-bit identical to `vmfcore.hash64` (Python/numpy) and
//! `vmfcore::hash64` (Rust): hashes of shared tensors match across
//! `.cmf` and `.vmfc`. Not cryptographic.

#[inline]
fn fmix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51AFD7ED558CCD);
    x ^= x >> 33;
    x = x.wrapping_mul(0xC4CEB9FE1A85EC53);
    x ^= x >> 33;
    x
}

/// murmur3-fmix over 64-bit LE words with positional salt, XOR fold.
pub fn hash64(b: &[u8]) -> u64 {
    let full = b.len() / 8;
    let mut acc = 0u64;
    for i in 0..full {
        let w = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
        acc ^= fmix64(w) ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
    }
    let rem = b.len() % 8;
    if rem != 0 {
        let mut w8 = [0u8; 8];
        w8[..rem].copy_from_slice(&b[full * 8..]);
        acc ^= fmix64(u64::from_le_bytes(w8)) ^ (full as u64).wrapping_mul(0x9E3779B97F4A7C15);
    }
    acc ^= b.len() as u64;
    fmix64(acc)
}
