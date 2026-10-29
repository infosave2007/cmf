//! `act_quant(o, 32, "ue8m0", inplace=True)` of the DeepSeek-V4.1 attention
//! output on the device (reference commit 4dd75a8b): per 32-value block an
//! E8M0 scale `s = 2^fast_log2_ceil(max(amax, 1e-4) * f32(1/448))`, the
//! values rounded to E4M3FN (nearest-even, saturating at 448) and multiplied
//! back by `s`, then the BF16 store. The host twin is
//! `dsv41::fp8_activation_inplace`; the kernel is bit-identical to it.
//!
//! WGSL has no FP8 type, so the kernel works on the f32 BITS (the buffer is
//! bound as `array<u32>`): the block maximum is an integer max of the
//! magnitudes, `x / s` is an exponent subtraction, the E4M3 rounding is a
//! round-to-nearest-even on the mantissa (three kept bits for normals, the
//! 2^-9 grid below 2^-6), and `q * s` is an exponent addition. The one
//! floating-point operation is `amax * f32(1/448)`, a single correctly
//! rounded multiply on every backend, of a normal number (amax >= 1e-4).
//! No step depends on the device's denormal or fast-math behaviour.
//!
//! Sign rule (the host's): an exact zero, of either sign, becomes +0; a
//! nonzero value that rounds to zero keeps its sign, unless the f32 quotient
//! `x / s` already underflowed to zero on the host. Finite inputs only (the
//! attention output after its BF16 boundary; NaN/inf are not matched).
//!
//! One invocation per 32-value block: the release output is 1024 blocks a
//! layer, so the dispatch is 16 workgroups of serial 32-value loops, far
//! below the cost of the grouped projection that follows. Its own lazily
//! built module, so models that never set `CMF_DSV41_O_FP8` compile nothing.

use super::*;

/// `f32(1/448)` and `1e-4f32` as the WGSL source spells them.
#[cfg(test)]
pub(crate) const FP8_MAX_INV_BITS: u32 = 0x3B12_4925;
#[cfg(test)]
pub(crate) const AMAX_FLOOR_BITS: u32 = 0x38D1_B717;

const SRC: &str = r#"
struct P { blocks: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> fx : array<u32>;
@group(0) @binding(1) var<uniform> fp : P;

const FP8_MAX_INV_BITS: u32 = 0x3B124925u; // f32(1/448), the kernel's constant
const AMAX_FLOOR_BITS: u32 = 0x38D1B717u;  // 1e-4f32
const E4M3_MAX_BITS: u32 = 0x43E00000u;    // 448
const E4M3_MIN_NORMAL_BITS: u32 = 0x3C800000u; // 2^-6

// RNE to BF16 of a finite value (the frame's bf16_round, on bits).
fn fp8_bf16(b: u32) -> u32 {
    return (b + 0x7FFFu + ((b >> 16u) & 1u)) & 0xFFFF0000u;
}

// One value of a block whose scale is 2^e.
fn fp8_one(b: u32, e: i32) -> u32 {
    let sgn = b & 0x80000000u;
    let ab = b & 0x7FFFFFFFu;
    if (ab == 0u) {
        return 0u;
    }
    let normal = ab >= 0x00800000u;
    // |x| = m * 2^(ex - 150); the quotient |x| / 2^e has the biased
    // exponent qf when x is normal.
    let m = select(ab, (ab & 0x7FFFFFu) | 0x800000u, normal);
    let ex = select(1, i32(ab >> 23u), normal);
    let qf = ex - e;
    if (!normal || qf <= 0) {
        // The quotient is below 2^-104 (a subnormal x: |x| < 2^-126 and
        // e >= -22) or below 2^-126 (a normal x with qf <= 0), far under
        // E4M3's smallest half step 2^-10, so E4M3 rounds it to zero. The
        // host's f32 division returns an exact zero, whose sign it drops,
        // when the quotient is at most half of 2^-149: m * 2^qf <= 1, which
        // needs qf <= 0 (with qf >= 1 the quotient is at least |x|).
        var exact_zero = false;
        if (qf <= 0) {
            let k = u32(-qf);
            exact_zero = k >= 24u || m <= (1u << k);
        }
        return select(sgn, 0u, exact_zero);
    }
    // |x| / s exactly (a normal f32), clamped to 448.
    let vb = min((u32(qf) << 23u) | (ab & 0x7FFFFFu), E4M3_MAX_BITS);
    var qb = 0u;
    if (vb < E4M3_MIN_NORMAL_BITS) {
        // The subnormal grid: n = round(|v| * 2^9), ties to even; n = 8 is
        // the carry into the smallest normal.
        let ve = i32(vb >> 23u) - 127;
        let mm = (vb & 0x7FFFFFu) | 0x800000u;
        let r = u32(14 - ve); // |v| * 2^9 = mm * 2^-r, r >= 21
        var n = 0u;
        if (r <= 24u) {
            n = mm >> r;
            let rem = mm & ((1u << r) - 1u);
            let half = 1u << (r - 1u);
            if (rem > half || (rem == half && (n & 1u) == 1u)) {
                n = n + 1u;
            }
        }
        if (n == 0u) {
            return sgn;
        }
        // n * 2^-9 as f32 bits.
        let lead = firstLeadingBit(n);
        qb = ((lead + 118u) << 23u) | ((n << (23u - lead)) & 0x7FFFFFu);
    } else {
        // Three mantissa bits, nearest-even; a carry moves into the exponent
        // (at most 448, which is representable).
        qb = (vb + 0x7FFFFu + ((vb >> 20u) & 1u)) & 0xFFF00000u;
    }
    // q * 2^e by exponent addition: q >= 2^-9 and e >= -22 stay normal.
    let out = (qb + (u32(e + 127) << 23u)) - (127u << 23u);
    return sgn | fp8_bf16(out);
}

@compute @workgroup_size(64)
fn fp8_act_quant32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let blk = gid.x;
    if (blk >= fp.blocks) {
        return;
    }
    let base = blk * 32u;
    var mx = AMAX_FLOOR_BITS;
    for (var i = 0u; i < 32u; i = i + 1u) {
        mx = max(mx, fx[base + i] & 0x7FFFFFFFu);
    }
    let t = bitcast<u32>(bitcast<f32>(mx) * bitcast<f32>(FP8_MAX_INV_BITS));
    let e = i32((t >> 23u) & 0xFFu) - 127 + select(0, 1, (t & 0x7FFFFFu) != 0u);
    for (var i = 0u; i < 32u; i = i + 1u) {
        fx[base + i] = fp8_one(fx[base + i], e);
    }
}
"#;

pub(crate) type Pipe = (wgpu::ComputePipeline, wgpu::BindGroupLayout);

fn build(c: &Ctx) -> Option<Pipe> {
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-fp8-act"),
        source: wgpu::ShaderSource::Wgsl(SRC.into()),
    });
    let p = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fp8_act_quant32"),
            layout: None,
            module: &module,
            entry_point: Some("fp8_act_quant32"),
            compilation_options: crate::gpu_wgpu::pco(),
            cache: c.pipeline_cache.as_ref(),
        });
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-fp8-act module rejected ({e}): act_quant(o) stays on the host tail");
        return None;
    }
    let l = p.get_bind_group_layout(0);
    Some((p, l))
}

/// The kernel, built on first use; `None` when the device rejected it (the
/// caller then declines the frame and the host tail quantizes).
pub(crate) fn pipe(c: &Ctx) -> Option<&Pipe> {
    c.fp8_act_pipe.get_or_init(|| build(c)).as_ref()
}

/// Encodes of the kernel by `encode` (the attention frame), cumulative for
/// the process.
static ENCODES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many times the attention frame encoded `fp8_act_quant32`, cumulative
/// for the process. For tests that must prove the frame quantized: its
/// numeric output alone cannot tell a skipped act_quant(o) from the device's
/// own rounding.
#[doc(hidden)]
pub fn frame_encodes() -> u64 {
    ENCODES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Quantize `n` floats of `x` in place (`n` a multiple of 32). The caller
/// has checked `pipe(c)`.
pub(crate) fn encode(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    n: usize,
    bkey: (u8, u64, usize),
) {
    let Some((pipeline, layout)) = pipe(c) else {
        return;
    };
    let blocks = (n / 32) as u32;
    let bind = cached_bind(c, bkey, || {
        let p = uniform_u32x4(c, [blocks, 0, 0, 0]);
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[bind_buf(0, x), bind_buf(1, &p)],
        })
    });
    let mut pass = begin_pass(enc);
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(blocks.div_ceil(64), 1, 1);
    ENCODES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The kernel alone on `x` (a multiple of 32 values), for the bit-exactness
/// test against the host quantizer. `None` without an adapter or when the
/// module was rejected.
pub fn act_quant32_for_test(x: &[f32]) -> Option<Vec<f32>> {
    let c = ctx()?;
    let _op = c.op_gate.enter();
    if x.is_empty() || x.len() % 32 != 0 {
        return None;
    }
    use wgpu::util::DeviceExt;
    let (pipeline, layout) = pipe(c)?;
    let buf = c
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fp8-act-test"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
    let blocks = (x.len() / 32) as u32;
    let p = uniform_u32x4(c, [blocks, 0, 0, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[bind_buf(0, &buf), bind_buf(1, &p)],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fp8-act-test"),
        });
    {
        let mut pass = begin_pass(&mut enc);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(blocks.div_ceil(64), 1, 1);
    }
    let stage = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fp8-act-test-stage"),
        size: (x.len() * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut out = vec![0.0f32; x.len()];
    readback(c, enc, &buf, &stage, (x.len() * 4) as u64, &mut out).then_some(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn wgsl_constants_are_the_host_f32_values() {
        assert_eq!((1.0f32 / 448.0).to_bits(), super::FP8_MAX_INV_BITS);
        assert_eq!(1e-4f32.to_bits(), super::AMAX_FLOOR_BITS);
        assert!(super::SRC.contains(&format!("0x{:08X}u", super::FP8_MAX_INV_BITS)));
        assert!(super::SRC.contains(&format!("0x{:08X}u", super::AMAX_FLOOR_BITS)));
    }
}
