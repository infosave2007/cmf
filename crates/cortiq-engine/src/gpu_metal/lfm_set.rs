//! d1's one-pass forward on Metal (`pipeline_masked::forward_masked`): every
//! layer of an LFM3 stack over the whole packed row in ONE command buffer —
//! the batched GEMMs of the chunk prefill (q4tp, and `moe_q8`'s blob GEMM
//! for q8_2f, which reads the weight's input field itself), the short
//! convolution walked over the positions from an empty ring, Q/K norm +
//! RoPE per row, the `set` attention (or causal within a sliding window)
//! and the per-head sigmoid gate. The host path ran each projection as its
//! own submission: 290 ms for a 43-token question on an M4.
//!
//! The kernels live in `lfm_set_msl.metal`, compiled the first time a d1
//! request asks for them; `CMF_D1_DEVICE=0` keeps the host path.

use super::{Ctx, GROUP_SIZE, MmKind, ctx, enc_mul_mm, file_buffer, io_buf_grow, moe_q8};
use crate::gpu::{SetDims, SetHead, SetLayer, SetMixer, SetW};
use cortiq_core::{CmfModel, TensorDtype};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

const LMSL: &str = include_str!("lfm_set_msl.metal");

struct Pipes {
    rms: ComputePipelineState,
    norm_add: ComputePipelineState,
    sconv: ComputePipelineState,
    qk_rope: ComputePipelineState,
    attn: ComputePipelineState,
    head_gate: ComputePipelineState,
    silu_mul: ComputePipelineState,
    gather: ComputePipelineState,
}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    let r = PIPES.get_or_init(|| {
        let opts = metal::CompileOptions::new();
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);
        let lib = c
            ._device
            .new_library_with_source(LMSL, &opts)
            .map_err(|e| format!("lfm_set MSL compile: {e}"))?;
        let pso = |name: &str| -> Result<ComputePipelineState, String> {
            let f = lib
                .get_function(name, None)
                .map_err(|e| format!("kernel {name}: {e}"))?;
            c._device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("pipeline {name}: {e}"))
        };
        Ok(Pipes {
            rms: pso("ls_rms_rows")?,
            norm_add: pso("ls_norm_add_rows")?,
            sconv: pso("ls_sconv")?,
            qk_rope: pso("ls_qk_rope")?,
            attn: pso("ls_attn")?,
            head_gate: pso("ls_head_gate")?,
            silu_mul: pso("ls_silu_mul")?,
            gather: pso("ls_gather_rows")?,
        })
    });
    match r {
        Ok(p) => Some(p),
        Err(e) => {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!("d1 Metal kernels unavailable ({e}) — the host path answers");
            }
            None
        }
    }
}

fn set_u32(enc: &ComputeCommandEncoderRef, slot: u64, v: u32) {
    enc.set_bytes(slot, 4, &v as *const u32 as *const c_void);
}

fn set_f32(enc: &ComputeCommandEncoderRef, slot: u64, v: f32) {
    enc.set_bytes(slot, 4, &v as *const f32 as *const c_void);
}

/// A resolved projection: its offset in the file arena and its GEMM.
#[derive(Clone, Copy)]
struct Proj {
    abs: usize,
    rows: usize,
    cols: usize,
    q8: bool,
}

fn resolve(model: &CmfModel, safe_len: usize, w: SetW, q8_ok: bool) -> Option<Proj> {
    let (idx, rows, cols) = w;
    let e = model.tensors.get(idx)?;
    if e.shape.as_slice() != [rows, cols] || cols % GROUP_SIZE != 0 {
        return None;
    }
    let abs = model.entry_abs_offset(e)?;
    let (bytes, q8) = match e.dtype {
        TensorDtype::Q4TiledP => (
            cortiq_core::quant::expected_nbytes(TensorDtype::Q4TiledP, &[rows, cols])?,
            false,
        ),
        // The blob GEMM tiles K by 32 and reads the payload, the row
        // scales and the input field from the mapping.
        TensorDtype::Q8_2f if q8_ok && cols % 32 == 0 && abs % 4 == 0 => {
            (moe_q8::tensor_bytes(rows, cols)?, true)
        }
        _ => return None,
    };
    (abs + bytes <= safe_len && e.nbytes as usize >= bytes).then_some(Proj {
        abs,
        rows,
        cols,
        q8,
    })
}

/// The final-normed hidden rows `[t, hidden]` of `rows` (embedding rows
/// `[t, hidden]`) under d1's mask (`state` leading rows, `group[r]` the
/// question of every later row) and, with `head`, the logits of its rows;
/// None when the stack or a weight is outside these kernels (the caller
/// keeps the host path).
#[allow(clippy::too_many_arguments)]
pub fn forward(
    model: &Arc<CmfModel>,
    layers: &[SetLayer],
    d: &SetDims,
    final_norm: &[f32],
    rows: &[f32],
    t: usize,
    state: usize,
    group: &[i32],
    head: Option<&SetHead>,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let c = ctx()?;
    let p = pipes(c)?;
    let hs = d.hidden;
    // Eight rows: the GEMM tiles' measured floor (the chunk prefill's).
    if t < 8
        || rows.len() != t * hs
        || group.len() != t
        || state > t
        || hs % 4 != 0
        || d.hd % 4 != 0
        || d.hd > 256
        || d.nh == 0
    {
        return None;
    }
    let (fbuf, safe_len) = file_buffer(c, model)?;
    let q8p = moe_q8::pipes(c);
    let res = |w: SetW| resolve(model, safe_len, w, q8p.is_some());
    // Validate the whole stack before anything is encoded.
    struct Prep {
        mix: [Proj; 4],
        ffn: [Proj; 3],
    }
    let mut preps = Vec::with_capacity(layers.len());
    let (mut wide, mut inter_max) = (3 * hs, 0usize);
    for l in layers {
        let ffn = [res(l.gate)?, res(l.up)?, res(l.down)?];
        let inter = ffn[0].rows;
        if ffn[1].rows != inter || ffn[2].cols != inter || ffn[0].cols != hs || ffn[2].rows != hs
        {
            return None;
        }
        inter_max = inter_max.max(inter);
        let mix = match &l.mixer {
            SetMixer::Conv {
                in_proj,
                out_proj,
                taps,
                kernel,
            } => {
                let (a, b) = (res(*in_proj)?, res(*out_proj)?);
                if !(1..=16).contains(kernel)
                    || taps.len() != hs * kernel
                    || a.rows != 3 * hs
                    || a.cols != hs
                    || b.rows != hs
                    || b.cols != hs
                {
                    return None;
                }
                [a, b, a, a]
            }
            SetMixer::Attn {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                head_gate,
                inv_freq,
                rd,
                nkv,
                window,
            } => {
                let w = [res(*wq)?, res(*wk)?, res(*wv)?, res(*wo)?];
                let (qd, kvd) = (d.nh * d.hd, nkv * d.hd);
                if *nkv == 0
                    || d.nh % nkv != 0
                    || w[0].rows != qd
                    || w[1].rows != kvd
                    || w[2].rows != kvd
                    || w[3].cols != qd
                    || w[3].rows != hs
                    || w.iter().take(3).any(|x| x.cols != hs)
                    || *rd > d.hd
                    || rd % 2 != 0
                    || inv_freq.len() < rd / 2
                    || q_norm.is_some_and(|n| n.len() != d.hd)
                    || k_norm.is_some_and(|n| n.len() != d.hd)
                    || head_gate.is_some_and(|g| g.len() != d.nh * hs)
                    || *window == Some(0)
                {
                    return None;
                }
                wide = wide.max(qd);
                w
            }
        };
        preps.push(Prep { mix, ffn });
    }
    // The head over the answer slots: at least eight rows (the GEMM's
    // floor), the first slot repeated.
    let head = match head {
        Some(hd) if !hd.rows.is_empty() => {
            let w = res(hd.w)?;
            if w.cols != hs || hd.rows.iter().any(|&r| r >= t) || hd.rows.len().max(8) > t {
                return None;
            }
            let mut idx: Vec<u32> = hd.rows.iter().map(|&r| r as u32).collect();
            idx.resize(idx.len().max(8), idx[0]);
            Some((w, idx, hd.rows.len()))
        }
        _ => None,
    };
    let kv_max = layers
        .iter()
        .map(|l| match l.mixer {
            SetMixer::Attn { nkv, .. } => nkv * d.hd,
            SetMixer::Conv { .. } => 0,
        })
        .max()
        .unwrap_or(0)
        .max(4);
    let wide = wide.max(inter_max);

    // Scratch, grown per call (`io_buf_grow`: one slot per key).
    let buf = |key: usize, floats: usize| io_buf_grow(c, key, floats.max(1) * 4);
    let h_b = buf(91_000_000_001, t * hs);
    let n_b = buf(91_000_000_003, t * hs);
    let d_b = buf(91_000_000_007, t * hs);
    let x_b = buf(91_000_000_009, t * wide);
    let u_b = buf(91_000_000_013, t * inter_max.max(d.nh * d.hd));
    let k_b = buf(91_000_000_019, t * kv_max);
    let v_b = buf(91_000_000_021, t * kv_max);
    let a_b = buf(91_000_000_027, t * wide);
    let g_b = io_buf_grow(c, 91_000_000_031, t * 4);
    // SAFETY: fresh shared buffers at least this large; no command buffer
    // holds them (this call commits and waits before returning).
    unsafe {
        std::ptr::copy_nonoverlapping(rows.as_ptr(), h_b.contents() as *mut f32, t * hs);
        std::ptr::copy_nonoverlapping(group.as_ptr(), g_b.contents() as *mut i32, t);
    }
    let cb = |data: &[f32]| super::const_buf(c, data);
    let dummy = cb(&[0f32; 4]);
    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let rows_tg = |enc: &ComputeCommandEncoderRef| {
        enc.dispatch_thread_groups(MTLSize::new(t as u64, 1, 1), MTLSize::new(256, 1, 1));
    };
    let rms = |x: &Buffer, w: &[f32], o: &Buffer| {
        enc.set_compute_pipeline_state(&p.rms);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(&cb(w)), 0);
        enc.set_buffer(2, Some(o), 0);
        set_u32(enc, 3, hs as u32);
        set_u32(enc, 4, d.gemma as u32);
        set_f32(enc, 5, d.eps);
        rows_tg(enc);
    };
    let norm_add = |dl: &Buffer, w: Option<&[f32]>| {
        enc.set_compute_pipeline_state(&p.norm_add);
        enc.set_buffer(0, Some(&h_b), 0);
        enc.set_buffer(1, Some(dl), 0);
        enc.set_buffer(2, Some(&w.map(cb).unwrap_or_else(|| dummy.clone())), 0);
        set_u32(enc, 3, hs as u32);
        set_u32(enc, 4, d.gemma as u32);
        set_f32(enc, 5, d.eps);
        set_u32(enc, 6, w.is_some() as u32);
        rows_tg(enc);
    };
    let mm_b = |w: &Proj, xs: &Buffer, y: &Buffer, t: usize| {
        if w.q8 {
            moe_q8::enc_mm_rows(
                q8p.expect("checked in resolve"),
                enc,
                &fbuf,
                w.abs,
                xs,
                0,
                y,
                0,
                t,
                w.rows,
                w.cols,
                true,
            );
        } else {
            enc_mul_mm(c, enc, &fbuf, w.abs, &dummy, MmKind::Q4tp, xs, y, t, w.rows, w.cols);
        }
    };
    let mm = |w: &Proj, xs: &Buffer, y: &Buffer| mm_b(w, xs, y, t);
    for (l, pr) in layers.iter().zip(&preps) {
        rms(&h_b, l.input_norm, &n_b);
        match &l.mixer {
            SetMixer::Conv { taps, kernel, .. } => {
                mm(&pr.mix[0], &n_b, &x_b);
                enc.set_compute_pipeline_state(&p.sconv);
                enc.set_buffer(0, Some(&x_b), 0);
                enc.set_buffer(1, Some(&cb(*taps)), 0);
                enc.set_buffer(2, Some(&a_b), 0);
                set_u32(enc, 3, hs as u32);
                set_u32(enc, 4, *kernel as u32);
                set_u32(enc, 5, t as u32);
                enc.dispatch_threads(MTLSize::new(hs as u64, 1, 1), MTLSize::new(256, 1, 1));
                mm(&pr.mix[1], &a_b, &d_b);
            }
            SetMixer::Attn {
                q_norm,
                k_norm,
                head_gate,
                inv_freq,
                rd,
                nkv,
                window,
                ..
            } => {
                mm(&pr.mix[0], &n_b, &x_b);
                mm(&pr.mix[1], &n_b, &k_b);
                mm(&pr.mix[2], &n_b, &v_b);
                let invf = cb(&inv_freq[..rd / 2]);
                for (buf, norm, heads) in [(&x_b, q_norm, d.nh), (&k_b, k_norm, *nkv)] {
                    enc.set_compute_pipeline_state(&p.qk_rope);
                    enc.set_buffer(0, Some(buf), 0);
                    enc.set_buffer(1, Some(&norm.map(cb).unwrap_or_else(|| dummy.clone())), 0);
                    enc.set_buffer(2, Some(&invf), 0);
                    set_u32(enc, 3, heads as u32);
                    set_u32(enc, 4, d.hd as u32);
                    set_u32(enc, 5, *rd as u32);
                    set_u32(enc, 6, norm.is_some() as u32);
                    set_u32(enc, 7, d.gemma as u32);
                    set_f32(enc, 8, d.eps);
                    enc.dispatch_thread_groups(
                        MTLSize::new(heads as u64, t as u64, 1),
                        MTLSize::new(32, 1, 1),
                    );
                }
                enc.set_compute_pipeline_state(&p.attn);
                enc.set_buffer(0, Some(&x_b), 0);
                enc.set_buffer(1, Some(&k_b), 0);
                enc.set_buffer(2, Some(&v_b), 0);
                enc.set_buffer(3, Some(&a_b), 0);
                enc.set_buffer(4, Some(&g_b), 0);
                set_u32(enc, 5, t as u32);
                set_u32(enc, 6, d.nh as u32);
                set_u32(enc, 7, *nkv as u32);
                set_u32(enc, 8, d.hd as u32);
                set_u32(enc, 9, state as u32);
                set_u32(enc, 10, window.unwrap_or(0) as u32);
                set_f32(enc, 11, d.scale);
                enc.dispatch_thread_groups(
                    MTLSize::new(t as u64, d.nh as u64, 1),
                    MTLSize::new(128, 1, 1),
                );
                if let Some(gw) = head_gate {
                    enc.set_compute_pipeline_state(&p.head_gate);
                    enc.set_buffer(0, Some(&a_b), 0);
                    enc.set_buffer(1, Some(&n_b), 0);
                    enc.set_buffer(2, Some(&cb(*gw)), 0);
                    set_u32(enc, 3, d.nh as u32);
                    set_u32(enc, 4, d.hd as u32);
                    set_u32(enc, 5, hs as u32);
                    enc.dispatch_thread_groups(
                        MTLSize::new(d.nh as u64, t as u64, 1),
                        MTLSize::new(256, 1, 1),
                    );
                }
                mm(&pr.mix[3], &a_b, &d_b);
            }
        }
        norm_add(&d_b, l.attn_out_norm);
        rms(&h_b, l.post_norm, &n_b);
        let inter = pr.ffn[0].rows;
        mm(&pr.ffn[0], &n_b, &x_b);
        mm(&pr.ffn[1], &n_b, &u_b);
        enc.set_compute_pipeline_state(&p.silu_mul);
        enc.set_buffer(0, Some(&x_b), 0);
        enc.set_buffer(1, Some(&u_b), 0);
        enc.set_buffer(2, Some(&a_b), 0);
        set_u32(enc, 3, (t * inter) as u32);
        enc.dispatch_threads(MTLSize::new((t * inter) as u64, 1, 1), MTLSize::new(256, 1, 1));
        mm(&pr.ffn[2], &a_b, &d_b);
        norm_add(&d_b, l.ffn_out_norm);
    }
    rms(&h_b, final_norm, &n_b);
    let logits_b = head.as_ref().map(|(w, idx, _)| {
        let nb = idx.len();
        let ib = c._device.new_buffer_with_data(
            idx.as_ptr() as *const c_void,
            (nb * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        enc.set_compute_pipeline_state(&p.gather);
        enc.set_buffer(0, Some(&n_b), 0);
        enc.set_buffer(1, Some(&ib), 0);
        enc.set_buffer(2, Some(&d_b), 0);
        set_u32(enc, 3, hs as u32);
        enc.dispatch_threads(MTLSize::new(hs as u64, nb as u64, 1), MTLSize::new(256, 1, 1));
        let lb = buf(91_000_000_037, nb * w.rows);
        mm_b(w, &d_b, &lb, nb);
        lb
    });
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    if cmd.status() != metal::MTLCommandBufferStatus::Completed {
        tracing::warn!("d1 Metal forward: command buffer {:?}", cmd.status());
        return None;
    }
    let mut out = vec![0f32; t * hs];
    // SAFETY: the command buffer completed; n_b holds t·hs floats.
    unsafe {
        std::ptr::copy_nonoverlapping(n_b.contents() as *const f32, out.as_mut_ptr(), t * hs);
    }
    let mut logits = Vec::new();
    if let (Some((w, _, n)), Some(lb)) = (&head, &logits_b) {
        logits = vec![0f32; n * w.rows];
        // SAFETY: as above; the head wrote at least n·vocab floats.
        unsafe {
            std::ptr::copy_nonoverlapping(lb.contents() as *const f32, logits.as_mut_ptr(), logits.len());
        }
    }
    Some((out, logits))
}
