//! The GigaAM encoder resident on wgpu (`gpu::gigaam_encode`).

/// The encoder stacks over one segment `x` (`[t, width]`) in place;
/// `false` = declined, the caller runs its own path.
pub fn encode(_a: &crate::gpu::GigaamArgs, _x: &mut [f32], _t: usize) -> bool {
    false
}
