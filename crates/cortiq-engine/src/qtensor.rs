//! QTensor — weight tensor with pluggable storage.
//!
//! Two backings, one interface:
//! - `F32`   — owned dense floats (small models, tests). Every operation
//!   is bit-identical to the historical `&[f32]` code paths.
//! - `Mapped` — quantized bytes zero-copy from the CMF mmap (`q8_row` /
//!   `q8_2f`). The matvec is fused: int8 rows × f32 activations, the
//!   q8_2f column field folds into a pre-scale of the input
//!   (`x'[i] = col[i]·x[i]`), so the inner loop is the same i8 dot as
//!   q8_row. This is what lets a 15B file run in a few GB of RSS.
//!
//! Extension point: new dtypes = new match arm here, nothing else moves.

use crate::pool::{matvec_rows, matvec_rows2, Pool};
use cortiq_core::quant::{f16_to_f32, GROUP_SIZE};
use cortiq_core::{CmfModel, TensorDtype};
use std::sync::Arc;

pub enum QTensor {
    F32 {
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    Mapped {
        model: Arc<CmfModel>,
        /// Index into the model's tensor directory.
        idx: usize,
        dtype: TensorDtype,
        rows: usize,
        cols: usize,
        /// Per-row scales, dequantized to f32 up front (tiny).
        row_scale: Vec<f32>,
        /// q8_2f column field (θ), dequantized up front; empty for q8_row.
        col_field: Vec<f32>,
    },
}

impl QTensor {
    pub fn from_f32(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self::F32 { data, rows, cols }
    }

    /// Wrap a directory tensor without dequantizing the payload.
    /// Falls back to dequantized f32 for dtypes without a fused kernel.
    pub fn from_model(model: &Arc<CmfModel>, name: &str) -> Result<Self, String> {
        let idx = model
            .tensors
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| format!("tensor '{name}' not found in CMF directory"))?;
        let entry = &model.tensors[idx];
        if entry.shape.len() != 2 {
            return Err(format!("QTensor::from_model needs 2-D, got '{name}'"));
        }
        let (rows, cols) = (entry.shape[0], entry.shape[1]);
        let bytes = model.entry_bytes(entry);

        match entry.dtype {
            TensorDtype::Q8Row | TensorDtype::Q8_2f => {
                let n = rows * cols;
                let scales_off = n;
                let row_scale: Vec<f32> = (0..rows)
                    .map(|o| {
                        f16_to_f32(u16::from_le_bytes([
                            bytes[scales_off + o * 2],
                            bytes[scales_off + o * 2 + 1],
                        ]))
                    })
                    .collect();
                let col_field: Vec<f32> = if entry.dtype == TensorDtype::Q8_2f {
                    let col_off = n + rows * 2;
                    (0..cols)
                        .map(|i| {
                            f16_to_f32(u16::from_le_bytes([
                                bytes[col_off + i * 2],
                                bytes[col_off + i * 2 + 1],
                            ]))
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                Ok(Self::Mapped {
                    model: model.clone(),
                    idx,
                    dtype: entry.dtype,
                    rows,
                    cols,
                    row_scale,
                    col_field,
                })
            }
            // vbit: fused kernel unpacks variable-bit rows from mmap.
            TensorDtype::Vbit if cols % GROUP_SIZE == 0 => Ok(Self::Mapped {
                model: model.clone(),
                idx,
                dtype: entry.dtype,
                rows,
                cols,
                row_scale: Vec::new(),
                col_field: Vec::new(),
            }),
            // q4_block: fused kernel reads nibbles straight from mmap —
            // a 14B q4 file no longer explodes into ×8 f32 RAM.
            TensorDtype::Q4Block if cols % GROUP_SIZE == 0 => Ok(Self::Mapped {
                model: model.clone(),
                idx,
                dtype: entry.dtype,
                rows,
                cols,
                row_scale: Vec::new(),
                col_field: Vec::new(),
            }),
            // No fused kernel yet → dequantize once (correct, more RAM).
            _ => {
                let mut data = vec![0.0f32; rows * cols];
                cortiq_core::quant::dequant_tensor(entry, bytes, &mut data)?;
                Ok(Self::from_f32(data, rows, cols))
            }
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::F32 { rows, .. } | Self::Mapped { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::F32 { cols, .. } | Self::Mapped { cols, .. } => *cols,
        }
    }

    /// Dense f32 view — only for owned tensors. Masked/sparse execution
    /// paths require it; quantized weights don't support masks yet.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32 { data, .. } => Some(data),
            Self::Mapped { .. } => None,
        }
    }

    fn quant_bytes(&self) -> &[u8] {
        match self {
            Self::Mapped { model, idx, .. } => model.entry_bytes(&model.tensors[*idx]),
            Self::F32 { .. } => unreachable!("quant_bytes on F32"),
        }
    }

    /// Dequantize one row into `dst` (embedding lookup).
    pub fn row_f32(&self, r: usize, dst: &mut [f32]) {
        let cols = self.cols();
        debug_assert_eq!(dst.len(), cols);
        match self {
            Self::F32 { data, .. } => dst.copy_from_slice(&data[r * cols..(r + 1) * cols]),
            Self::Mapped {
                dtype,
                row_scale,
                col_field,
                ..
            } => {
                if *dtype == TensorDtype::Q4Block {
                    let (packed, scales) = q4_split(self.quant_bytes(), self.rows(), cols);
                    let gpr = cols / GROUP_SIZE;
                    for gi in 0..gpr {
                        let g = r * gpr + gi;
                        let s = f16_to_f32(u16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]));
                        for (k, &b) in packed[g * 16..(g + 1) * 16].iter().enumerate() {
                            dst[gi * GROUP_SIZE + k * 2] = ((b & 0x0F) as f32 - 8.0) * s;
                            dst[gi * GROUP_SIZE + k * 2 + 1] = (((b >> 4) & 0x0F) as f32 - 8.0) * s;
                        }
                    }
                    return;
                }
                if *dtype == TensorDtype::Vbit {
                    let bytes = self.quant_bytes();
                    let rows = self.rows();
                    let ng = cols / GROUP_SIZE;
                    let bits = &bytes[..rows];
                    let sc_off = rows;
                    let mut off = sc_off + rows * ng * 2;
                    for rr in 0..r {
                        off += (cols * bits[rr] as usize + 7) / 8;
                    }
                    let b = bits[r] as usize;
                    let l = ((1usize << (b - 1)) - 1) as f32;
                    let data = &bytes[off..];
                    let (mut acc, mut nbits, mut idx) = (0u64, 0usize, 0usize);
                    for (i, d) in dst.iter_mut().enumerate() {
                        while nbits < b {
                            acc = (acc << 8) | data[idx] as u64;
                            idx += 1;
                            nbits += 8;
                        }
                        let u = ((acc >> (nbits - b)) & ((1u64 << b) - 1)) as f32;
                        nbits -= b;
                        let so = (r * ng + i / GROUP_SIZE) * 2;
                        let sv = f16_to_f32(u16::from_le_bytes([
                            bytes[sc_off + so],
                            bytes[sc_off + so + 1],
                        ]));
                        *d = (u - l) * sv;
                    }
                    return;
                }
                let q = &self.quant_bytes()[r * cols..(r + 1) * cols];
                let s = row_scale[r];
                match dtype {
                    TensorDtype::Q8Row => {
                        for (d, &b) in dst.iter_mut().zip(q) {
                            *d = (b as i8) as f32 * s;
                        }
                    }
                    TensorDtype::Q8_2f => {
                        for (i, (d, &b)) in dst.iter_mut().zip(q).enumerate() {
                            *d = (b as i8) as f32 * s * col_field[i];
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Can this tensor's columns be read cheaply (for sparse down_proj)?
    /// True for F32/Q8Row/Q8_2f (per-row scale, direct strided access);
    /// false for group-packed q4/vbit (column access would unpack whole
    /// groups — sparse execution falls back to f32 for those).
    pub fn sparse_col_ok(&self) -> bool {
        match self {
            Self::F32 { .. } => true,
            Self::Mapped { dtype, .. } => {
                matches!(dtype, TensorDtype::Q8Row | TensorDtype::Q8_2f)
            }
        }
    }

    /// down_proj [hidden, inter]: accumulate `w · col(c)` into `out`
    /// [hidden] — reads ONLY column `c` (one neuron) from the mmap,
    /// no full-matrix dequant. `out[k] += w · down[k, c]`.
    pub fn add_col_scaled(&self, c: usize, w: f32, out: &mut [f32]) {
        let inter = self.cols();
        let hidden = self.rows();
        debug_assert_eq!(out.len(), hidden);
        match self {
            Self::F32 { data, .. } => {
                for (k, o) in out.iter_mut().enumerate() {
                    *o += w * data[k * inter + c];
                }
            }
            Self::Mapped {
                dtype, row_scale, col_field, ..
            } => {
                let q = self.quant_bytes();
                let colf = if *dtype == TensorDtype::Q8_2f {
                    col_field[c]
                } else {
                    1.0
                };
                let wc = w * colf;
                for (k, o) in out.iter_mut().enumerate() {
                    let b = q[k * inter + c] as i8 as f32;
                    *o += wc * b * row_scale[k];
                }
            }
        }
    }

    /// Dot of row `r` with `x` (gate/up active-neuron path). Reads only
    /// row `r` from the mmap — no full dequant. q4/vbit dequant the row
    /// into `scratch` first (rare for active-FFN weights).
    pub fn row_dot(&self, r: usize, x: &[f32], scratch: &mut [f32]) -> f32 {
        let cols = self.cols();
        match self {
            Self::F32 { data, .. } => {
                let row = &data[r * cols..(r + 1) * cols];
                row.iter().zip(x).map(|(w, v)| w * v).sum()
            }
            Self::Mapped { dtype, row_scale, col_field, .. } => match dtype {
                TensorDtype::Q8Row => {
                    let q = &self.quant_bytes()[r * cols..(r + 1) * cols];
                    let s: f32 = q.iter().zip(x).map(|(&b, v)| (b as i8) as f32 * v).sum();
                    s * row_scale[r]
                }
                TensorDtype::Q8_2f => {
                    let q = &self.quant_bytes()[r * cols..(r + 1) * cols];
                    let s: f32 = q
                        .iter()
                        .zip(x)
                        .zip(col_field)
                        .map(|((&b, v), c)| (b as i8) as f32 * v * c)
                        .sum();
                    s * row_scale[r]
                }
                _ => {
                    self.row_f32(r, scratch);
                    scratch.iter().zip(x).map(|(w, v)| w * v).sum()
                }
            },
        }
    }

    /// `out = W · x` (row-major). F32 delegates to the historical
    /// bit-exact path; Mapped runs the fused int8 kernel.
    pub fn matvec(&self, x: &[f32], out: &mut [f32], pool: Option<&Pool>) {
        match self {
            Self::F32 { data, .. } => matvec_rows(pool, data, x, out),
            Self::Mapped {
                model,
                idx,
                dtype,
                rows,
                cols,
                row_scale,
                col_field,
            } => {
                let _ = (model, idx);
                if *dtype == TensorDtype::Q4Block {
                    q4matvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                    return;
                }
                if *dtype == TensorDtype::Vbit {
                    vbitmatvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                    return;
                }
                let xs = prescale(x, col_field, *dtype);
                // D5: large q8 matrices (lm_head-class) — hybrid
                // CPU∥GPU: split the rows, both sides compute
                // SIMULTANEOUSLY (same math, shared prescale).
                // GPU share: CMF_GPU_SPLIT (0..1, default 0.5).
                if *rows >= crate::gpu::min_rows()
                    && matches!(dtype, TensorDtype::Q8Row | TensorDtype::Q8_2f)
                    && std::env::var("CMF_GPU_LMHEAD").map(|v| v != "0").unwrap_or(true)
                    && crate::gpu::enabled_here()
                {
                    let frac = std::env::var("CMF_GPU_SPLIT")
                        .ok()
                        .and_then(|v| v.parse::<f32>().ok())
                        .unwrap_or(0.5)
                        .clamp(0.0, 1.0);
                    let cpu_rows = ((*rows as f32) * (1.0 - frac)) as usize;
                    let (out_cpu, out_gpu) = out.split_at_mut(cpu_rows);
                    let bytes = self.quant_bytes();
                    let ok = std::thread::scope(|sc| {
                        let g = sc.spawn(|| {
                            crate::gpu::q8_matvec_range(
                                model,
                                *idx,
                                cpu_rows,
                                &row_scale[cpu_rows..],
                                &xs,
                                *rows - cpu_rows,
                                *cols,
                                out_gpu,
                            )
                        });
                        if cpu_rows > 0 {
                            qmatvec(
                                &bytes[..cpu_rows * *cols],
                                &row_scale[..cpu_rows],
                                &xs,
                                cpu_rows,
                                *cols,
                                out_cpu,
                                pool,
                            );
                        }
                        g.join().unwrap_or(false)
                    });
                    if ok {
                        return;
                    }
                    // GPU failed — CPU finishes its half.
                    qmatvec(
                        &bytes[cpu_rows * *cols..(*rows) * *cols],
                        &row_scale[cpu_rows..],
                        &xs,
                        *rows - cpu_rows,
                        *cols,
                        out_gpu,
                        pool,
                    );
                    return;
                }
                qmatvec(self.quant_bytes(), row_scale, &xs, *rows, *cols, out, pool);
            }
        }
    }

    /// Fused two-input matvec (MTP verify pair): weights streamed once.
    pub fn matvec2(&self, x1: &[f32], x2: &[f32], o1: &mut [f32], o2: &mut [f32], pool: Option<&Pool>) {
        match self {
            Self::F32 { data, .. } => matvec_rows2(pool, data, x1, x2, o1, o2),
            Self::Mapped {
                dtype,
                rows,
                cols,
                row_scale,
                col_field,
                ..
            } => {
                if *dtype == TensorDtype::Q4Block {
                    q4matvec(self.quant_bytes(), x1, *rows, *cols, o1, pool);
                    q4matvec(self.quant_bytes(), x2, *rows, *cols, o2, pool);
                    return;
                }
                if *dtype == TensorDtype::Vbit {
                    vbitmatvec(self.quant_bytes(), x1, *rows, *cols, o1, pool);
                    vbitmatvec(self.quant_bytes(), x2, *rows, *cols, o2, pool);
                    return;
                }
                let x1s = prescale(x1, col_field, *dtype);
                let x2s = prescale(x2, col_field, *dtype);
                qmatvec2(self.quant_bytes(), row_scale, &x1s, &x2s, *rows, *cols, o1, o2, pool);
            }
        }
    }
}

impl QTensor {
    /// Batched matvec (prefill-GEMM): xs — row-major [b, cols],
    /// out — row-major [b, rows]. Element-wise semantics are IDENTICAL
    /// to b matvec calls (same dot kernels in the same order); the win —
    /// the weight row streams from DRAM once per batch, not b times.
    pub fn matmat(&self, xs_all: &[f32], b: usize, out: &mut [f32], pool: Option<&Pool>) {
        let cols = self.cols();
        let rows = self.rows();
        debug_assert_eq!(xs_all.len(), b * cols);
        debug_assert_eq!(out.len(), b * rows);
        match self {
            Self::F32 { data, .. } => {
                let out_addr = SendMut(out.as_mut_ptr());
                let run = |start: usize, end: usize| {
                    for o in start..end {
                        let row = &data[o * cols..(o + 1) * cols];
                        for bi in 0..b {
                            let x = &xs_all[bi * cols..(bi + 1) * cols];
                            let mut acc = 0f32;
                            for j in 0..cols {
                                acc += row[j] * x[j];
                            }
                            unsafe { *out_addr.at(bi * rows + o) = acc };
                        }
                    }
                };
                dispatch_rows(pool, rows, &run);
            }
            Self::Mapped {
                dtype,
                row_scale,
                col_field,
                ..
            } => {
                if matches!(dtype, TensorDtype::Q4Block | TensorDtype::Vbit) {
                    // No batch kernel — honest element-wise fallback.
                    for bi in 0..b {
                        let x = &xs_all[bi * cols..(bi + 1) * cols];
                        let (head, tail) = out[bi * rows..].split_at_mut(rows);
                        let _ = tail;
                        self.matvec(x, head, pool);
                    }
                    return;
                }
                let pre: Vec<Vec<f32>> = (0..b)
                    .map(|bi| {
                        prescale(&xs_all[bi * cols..(bi + 1) * cols], col_field, *dtype)
                    })
                    .collect();
                // D5: large prefill-batch GEMMs — on the GPU (threshold by
                // work volume: submission carries b×rows×cols MACs).
                if b >= 8
                    && b * rows * cols >= 128_000_000
                    && crate::gpu::enabled_here()
                {
                    if let Self::Mapped { model, idx, .. } = self {
                        let flat: Vec<f32> =
                            pre.iter().flat_map(|v| v.iter().copied()).collect();
                        if crate::gpu::q8_matmat(
                            model, *idx, row_scale, &flat, b, rows, cols, out)
                        {
                            return;
                        }
                    }
                }
                let q = self.quant_bytes();
                qmatmat(q, row_scale, &pre, rows, cols, out, pool);
            }
        }
    }
}

/// Batched q8 kernel: same math as qmatvec, the row makes a single
/// pass from memory for the whole batch.
fn qmatmat(
    q: &[u8],
    row_scale: &[f32],
    pre: &[Vec<f32>],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    let b = pre.len();
    debug_assert_eq!(out.len(), b * rows);
    #[cfg(target_arch = "aarch64")]
    if sdot_enabled() {
        let acts: Vec<SplitAct> = pre.iter().map(|x| split_act(x)).collect();
        let out_addr = SendMut(out.as_mut_ptr());
        let run = |start: usize, end: usize| {
            for o in start..end {
                let row = &q[o * cols..(o + 1) * cols];
                for (bi, act) in acts.iter().enumerate() {
                    let v = row_dot_sdot(row, act) * row_scale[o];
                    unsafe { *out_addr.at(bi * rows + o) = v };
                }
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let out_addr = SendMut(out.as_mut_ptr());
    let run = |start: usize, end: usize| {
        for o in start..end {
            let row = &q[o * cols..(o + 1) * cols];
            for (bi, x) in pre.iter().enumerate() {
                let mut acc = 0f32;
                for j in 0..cols {
                    acc += (row[j] as i8) as f32 * x[j];
                }
                unsafe { *out_addr.at(bi * rows + o) = acc * row_scale[o] };
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// Split rows across pool workers (shared qmatvec pattern).
fn dispatch_rows(pool: Option<&Pool>, rows: usize, run: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(pool) if rows >= 256 => {
            pool.run(&|widx, n| {
                let chunk = rows.div_ceil(n);
                let start = widx * chunk;
                let end = (start + chunk).min(rows);
                if start < end {
                    run(start, end);
                }
            });
        }
        _ => run(0, rows),
    }
}

/// Split a q4_block blob into (packed nibbles, f16 group scales).
fn q4_split(bytes: &[u8], rows: usize, cols: usize) -> (&[u8], &[u8]) {
    let groups = rows * cols / GROUP_SIZE;
    bytes.split_at(groups * 16)
}

/// Unpack 8 MSB-first B-bit values from exactly B bytes (fixed shifts —
/// no serial bit-buffer, auto-vectorizable). Every 32-value group starts
/// byte-aligned (32·B/8 is integral for B∈3..8), so groups decompose
/// into 4 such blocks.
#[inline(always)]
fn unpack8<const B: usize>(data: &[u8]) -> [i32; 8] {
    let mut acc = 0u64;
    for i in 0..B {
        acc = (acc << 8) | data[i] as u64;
    }
    let mask = (1u64 << B) - 1;
    let mut out = [0i32; 8];
    for (k, o) in out.iter_mut().enumerate() {
        *o = ((acc >> ((7 - k) * B)) & mask) as i32;
    }
    out
}

/// Fused vbit matvec straight from the mapped bytes (spec §3, P13
/// FIG.3): [u8 bits: rows][f16 scales: rows·cols/32][bit-packed rows,
/// MSB-first, byte-padded]. Row data offsets are a prefix sum over
/// bits — computed once per call, then rows split across the pool.
fn vbitmatvec(bytes: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut [f32], pool: Option<&Pool>) {
    debug_assert_eq!(out.len(), rows);
    let ng = cols / GROUP_SIZE;
    let bits = &bytes[..rows];
    let sc_off = rows;
    let data_off = sc_off + rows * ng * 2;
    let mut offsets = Vec::with_capacity(rows + 1);
    let mut off = data_off;
    for r in 0..rows {
        offsets.push(off);
        off += (cols * bits[r] as usize + 7) / 8;
    }
    offsets.push(off);

    let offsets = &offsets;

    // SDOT path: unpack the row to centered i8 once, then per-group
    // int8 dot against the quantized activations — same A8W8 contract
    // as q8 (bounded noise; CMF_SDOT=0 keeps the exact scalar path).
    #[cfg(target_arch = "aarch64")]
    if sdot_enabled() {
        let act = split_act(x);
        let act = &act;
        let row_dot = move |r: usize| -> f32 {
            let b = bits[r] as usize;
            let l = ((1i32 << (b - 1)) - 1) as i32;
            let mask = (1u64 << b) - 1;
            let data = &bytes[offsets[r]..offsets[r + 1]];
            if b == 8 {
                // u−L reaches 128 → does not fit i8; exact f32 path.
                let (mut acc, mut nbits, mut idx) = (0u64, 0usize, 0usize);
                let mut dot = 0f32;
                for g in 0..ng {
                    let so = (r * ng + g) * 2;
                    let sgf = f16_to_f32(u16::from_le_bytes([
                        bytes[sc_off + so],
                        bytes[sc_off + so + 1],
                    ]));
                    let xg = &x[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
                    let mut gd = 0f32;
                    for &xv in xg.iter() {
                        if nbits < 8 {
                            acc = (acc << 8) | data[idx] as u64;
                            idx += 1;
                            nbits += 8;
                        }
                        let u = ((acc >> (nbits - 8)) & 0xFF) as i32;
                        nbits -= 8;
                        gd += (u - l) as f32 * xv;
                    }
                    dot += gd * sgf;
                }
                return dot;
            }
            let mut buf = vec![0u8; cols]; // centered i8 stored as u8 bits
            #[inline(always)]
            fn fill<const B: usize>(data: &[u8], l: i32, buf: &mut [u8]) {
                for (blk, chunk) in buf.chunks_exact_mut(8).enumerate() {
                    let u = unpack8::<B>(&data[blk * B..]);
                    for k in 0..8 {
                        chunk[k] = (u[k] - l) as i8 as u8;
                    }
                }
            }
            let _ = mask;
            match b {
                3 => fill::<3>(data, l, &mut buf),
                4 => fill::<4>(data, l, &mut buf),
                5 => fill::<5>(data, l, &mut buf),
                6 => fill::<6>(data, l, &mut buf),
                _ => unreachable!(),
            }
            let mut dot = 0f32;
            for g in 0..ng {
                let so = (r * ng + g) * 2;
                let s = f16_to_f32(u16::from_le_bytes([
                    bytes[sc_off + so],
                    bytes[sc_off + so + 1],
                ]));
                let d = unsafe {
                    dot_i8_sdot(
                        &buf[g * GROUP_SIZE..(g + 1) * GROUP_SIZE],
                        &act.xq[g * GROUP_SIZE..(g + 1) * GROUP_SIZE],
                    )
                } as f32
                    * act.sx;
                dot += d * s;
            }
            for &(j, xv) in &act.outliers {
                let so = (r * ng + j / GROUP_SIZE) * 2;
                let s = f16_to_f32(u16::from_le_bytes([
                    bytes[sc_off + so],
                    bytes[sc_off + so + 1],
                ]));
                // xq is zeroed at outlier slots — add the exact term.
                dot += (buf[j] as i8) as f32 * s * xv;
            }
            dot
        };
        match pool {
            Some(pool) if rows >= 256 => {
                let out_addr = SendMut(out.as_mut_ptr());
                pool.run(&move |widx, n| {
                    let chunk = rows.div_ceil(n);
                    let start = widx * chunk;
                    let end = (start + chunk).min(rows);
                    for r in start..end {
                        // SAFETY: disjoint row ranges per worker.
                        unsafe { *out_addr.at(r) = row_dot(r) };
                    }
                });
            }
            _ => {
                for (r, dst) in out.iter_mut().enumerate() {
                    *dst = row_dot(r);
                }
            }
        }
        return;
    }

    // Per-bit-width specialized inner loops: the compiler unrolls the
    // constant shifts (the generic bit-buffer loop was branch-bound —
    // 5.6 vs 13.2 tok/s q4 on the 0.8B).
    #[inline(always)]
    fn dot_row<const B: usize>(
        data: &[u8],
        bytes: &[u8],
        sc_off: usize,
        r: usize,
        ng: usize,
        x: &[f32],
    ) -> f32 {
        let l = ((1i32 << (B - 1)) - 1) as f32;
        let gbytes = GROUP_SIZE * B / 8;
        let mut dot = 0f32;
        for g in 0..ng {
            let so = (r * ng + g) * 2;
            let s = f16_to_f32(u16::from_le_bytes([bytes[sc_off + so], bytes[sc_off + so + 1]]));
            let xg = &x[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
            let gd0 = &data[g * gbytes..(g + 1) * gbytes];
            let mut gd = 0f32;
            for blk in 0..GROUP_SIZE / 8 {
                let u = unpack8::<B>(&gd0[blk * B..]);
                let xb = &xg[blk * 8..blk * 8 + 8];
                for k in 0..8 {
                    gd += (u[k] as f32 - l) * xb[k];
                }
            }
            dot += gd * s;
        }
        dot
    }
    let row_dot = move |r: usize| -> f32 {
        let data = &bytes[offsets[r]..offsets[r + 1]];
        match bits[r] {
            3 => dot_row::<3>(data, bytes, sc_off, r, ng, x),
            4 => dot_row::<4>(data, bytes, sc_off, r, ng, x),
            5 => dot_row::<5>(data, bytes, sc_off, r, ng, x),
            6 => dot_row::<6>(data, bytes, sc_off, r, ng, x),
            8 => dot_row::<8>(data, bytes, sc_off, r, ng, x),
            b => unreachable!("vbit bit-width {b} (validated at load)"),
        }
    };
    match pool {
        Some(pool) if rows >= 256 => {
            let out_addr = SendMut(out.as_mut_ptr());
            pool.run(&move |widx, n| {
                let chunk = rows.div_ceil(n);
                let start = widx * chunk;
                let end = (start + chunk).min(rows);
                for r in start..end {
                    // SAFETY: disjoint row ranges per worker.
                    unsafe { *out_addr.at(r) = row_dot(r) };
                }
            });
        }
        _ => {
            for (r, dst) in out.iter_mut().enumerate() {
                *dst = row_dot(r);
            }
        }
    }
}

/// Fused q4_block matvec straight from the mapped bytes. Scalar inner
/// loop for now — correctness (and the ×8 RAM fix) first; a nibble
/// SDOT port (vmfcore: +23%) is the follow-up optimization.
fn q4matvec(bytes: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut [f32], pool: Option<&Pool>) {
    debug_assert_eq!(out.len(), rows);
    let (packed, scales) = q4_split(bytes, rows, cols);
    let gpr = cols / GROUP_SIZE;
    let row_dot = |r: usize| -> f32 {
        let mut acc = 0f32;
        for gi in 0..gpr {
            let g = r * gpr + gi;
            let s = f16_to_f32(u16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]));
            let pk = &packed[g * 16..(g + 1) * 16];
            let xg = &x[gi * GROUP_SIZE..(gi + 1) * GROUP_SIZE];
            let mut ga = 0f32;
            for (k, &b) in pk.iter().enumerate() {
                ga += ((b & 0x0F) as f32 - 8.0) * xg[k * 2]
                    + (((b >> 4) & 0x0F) as f32 - 8.0) * xg[k * 2 + 1];
            }
            acc += ga * s;
        }
        acc
    };
    match pool {
        Some(pool) if rows >= 256 => {
            let out_addr = SendMut(out.as_mut_ptr());
            pool.run(&move |widx, n| {
                let chunk = rows.div_ceil(n);
                let start = widx * chunk;
                let end = (start + chunk).min(rows);
                for r in start..end {
                    // SAFETY: disjoint row ranges per worker.
                    unsafe { *out_addr.at(r) = row_dot(r) };
                }
            });
        }
        _ => {
            for (r, dst) in out.iter_mut().enumerate() {
                *dst = row_dot(r);
            }
        }
    }
}

pub(crate) fn prescale(x: &[f32], col_field: &[f32], dtype: TensorDtype) -> Vec<f32> {
    if dtype == TensorDtype::Q8_2f {
        x.iter().zip(col_field).map(|(a, c)| a * c).collect()
    } else {
        x.to_vec()
    }
}

// ───────────────────── A8W8 SDOT path (port of vmfcore, ×1.78 decode) ─────────────────────

/// SDOT enabled? Default ON when the CPU has ARMv8.2 dotprod;
/// `CMF_SDOT=0` disables (falls back to i8×f32 NEON).
fn sdot_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let want = std::env::var("CMF_SDOT").map(|v| v != "0").unwrap_or(true);
        #[cfg(target_arch = "aarch64")]
        {
            want && std::arch::is_aarch64_feature_detected!("dotprod")
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = want;
            false
        }
    })
}

/// Two-field activation split (≡ vmfcore `q8_split_prep`): outlier
/// channels (>8·rms) are computed exactly in f32; the bulk (outliers
/// zeroed → clean absmax) goes through int8 SDOT. Computed ONCE per
/// matvec, shared by all rows/workers.
struct SplitAct {
    xq: Vec<i8>,
    sx: f32,
    outliers: Vec<(usize, f32)>,
}

fn split_act(x: &[f32]) -> SplitAct {
    let n = x.len();
    let rms = (x.iter().map(|&v| (v * v) as f64).sum::<f64>() / n.max(1) as f64).sqrt() as f32;
    let thr = 8.0 * rms;
    let outliers: Vec<(usize, f32)> = (0..n)
        .filter(|&j| x[j].abs() > thr)
        .map(|j| (j, x[j]))
        .collect();
    let mut xb = x.to_vec();
    for &(j, _) in &outliers {
        xb[j] = 0.0;
    }
    let amax = xb.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let sx = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let inv = 1.0 / sx;
    let xq = xb
        .iter()
        .map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8)
        .collect();
    SplitAct { xq, sx, outliers }
}

/// int8(weight)·int8(activation) → i32 via `sdot` (inline asm — the
/// vdotq intrinsic is unstable; port of vmfcore `dot_i8_sdot`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_i8_sdot(w: &[u8], xq: &[i8]) -> i32 {
    // SAFETY: callers uphold slice-length contracts (see call sites).
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let wp = w.as_ptr() as *const i8;
        let n = w.len();
        let (mut a0, mut a1, mut a2, mut a3) =
            (vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0));
        let mut i = 0;
        while i + 64 <= n {
            let (w0, x0) = (vld1q_s8(wp.add(i)), vld1q_s8(xq.as_ptr().add(i)));
            let (w1, x1) = (vld1q_s8(wp.add(i + 16)), vld1q_s8(xq.as_ptr().add(i + 16)));
            let (w2, x2) = (vld1q_s8(wp.add(i + 32)), vld1q_s8(xq.as_ptr().add(i + 32)));
            let (w3, x3) = (vld1q_s8(wp.add(i + 48)), vld1q_s8(xq.as_ptr().add(i + 48)));
            asm!(
                "sdot {a0:v}.4s, {w0:v}.16b, {x0:v}.16b",
                "sdot {a1:v}.4s, {w1:v}.16b, {x1:v}.16b",
                "sdot {a2:v}.4s, {w2:v}.16b, {x2:v}.16b",
                "sdot {a3:v}.4s, {w3:v}.16b, {x3:v}.16b",
                a0 = inout(vreg) a0, a1 = inout(vreg) a1, a2 = inout(vreg) a2, a3 = inout(vreg) a3,
                w0 = in(vreg) w0, x0 = in(vreg) x0, w1 = in(vreg) w1, x1 = in(vreg) x1,
                w2 = in(vreg) w2, x2 = in(vreg) x2, w3 = in(vreg) w3, x3 = in(vreg) x3,
                options(pure, nomem, nostack),
            );
            i += 64;
        }
        while i + 16 <= n {
            let (wv, xv) = (vld1q_s8(wp.add(i)), vld1q_s8(xq.as_ptr().add(i)));
            asm!("sdot {a:v}.4s, {w:v}.16b, {x:v}.16b",
                 a = inout(vreg) a0, w = in(vreg) wv, x = in(vreg) xv, options(pure, nomem, nostack));
            i += 16;
        }
        let mut s = vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)));
        while i < n {
            s += (*wp.add(i)) as i32 * xq[i] as i32;
            i += 1;
        }
        s
}
}

/// Row-blocked SDOT: 4 output rows per pass — the activation chunk is
/// loaded once and reused, 4 independent accumulators hide sdot latency
/// (port of vmfcore `dot_i8_sdot_4rows`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_i8_sdot_4rows(w0: &[u8], w1: &[u8], w2: &[u8], w3: &[u8], xq: &[i8]) -> [i32; 4] {
    // SAFETY: callers uphold slice-length contracts (see call sites).
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let n = xq.len();
        let px = xq.as_ptr();
        let (p0, p1, p2, p3) = (
            w0.as_ptr() as *const i8,
            w1.as_ptr() as *const i8,
            w2.as_ptr() as *const i8,
            w3.as_ptr() as *const i8,
        );
        let (mut a0, mut a1, mut a2, mut a3) =
            (vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0));
        let mut i = 0;
        while i + 16 <= n {
            let x = vld1q_s8(px.add(i));
            let v0 = vld1q_s8(p0.add(i));
            let v1 = vld1q_s8(p1.add(i));
            let v2 = vld1q_s8(p2.add(i));
            let v3 = vld1q_s8(p3.add(i));
            asm!(
                "sdot {a0:v}.4s, {v0:v}.16b, {x:v}.16b",
                "sdot {a1:v}.4s, {v1:v}.16b, {x:v}.16b",
                "sdot {a2:v}.4s, {v2:v}.16b, {x:v}.16b",
                "sdot {a3:v}.4s, {v3:v}.16b, {x:v}.16b",
                a0 = inout(vreg) a0, a1 = inout(vreg) a1, a2 = inout(vreg) a2, a3 = inout(vreg) a3,
                v0 = in(vreg) v0, v1 = in(vreg) v1, v2 = in(vreg) v2, v3 = in(vreg) v3, x = in(vreg) x,
                options(pure, nomem, nostack),
            );
            i += 16;
        }
        let mut r = [vaddvq_s32(a0), vaddvq_s32(a1), vaddvq_s32(a2), vaddvq_s32(a3)];
        while i < n {
            let xi = *px.add(i) as i32;
            r[0] += (*p0.add(i)) as i32 * xi;
            r[1] += (*p1.add(i)) as i32 * xi;
            r[2] += (*p2.add(i)) as i32 * xi;
            r[3] += (*p3.add(i)) as i32 * xi;
            i += 1;
        }
        r
}
}

/// SDOT row dot with exact outlier correction:
/// `dot = sdot(w, xq)·sx + Σ_outl w[j]·x[j]` (then × row_scale by caller).
#[cfg(target_arch = "aarch64")]
#[inline]
fn row_dot_sdot(row: &[u8], act: &SplitAct) -> f32 {
    let mut acc = unsafe { dot_i8_sdot(row, &act.xq) } as f32 * act.sx;
    for &(j, xv) in &act.outliers {
        acc += (row[j] as i8) as f32 * xv;
    }
    acc
}

// ───────────────────── fused int8 kernels ─────────────────────

/// i8 row · f32 x. NEON on aarch64 (ported from vmfcore `dot_i8_f32_neon`,
/// ≈9× scalar), scalar elsewhere.
#[inline]
fn dot_i8_f32(w: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_i8_f32_neon(w, x);
    }
    #[allow(unreachable_code)]
    {
        let mut sum = 0.0f32;
        for (j, &b) in w.iter().enumerate() {
            sum += (b as i8) as f32 * x[j];
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_f32_neon(w: &[u8], x: &[f32]) -> f32 {
    // SAFETY: callers uphold slice-length contracts (see call sites).
    unsafe {
        use core::arch::aarch64::*;
        let n = x.len();
        let wp = w.as_ptr() as *const i8;
        let xp = x.as_ptr();
        let (mut a0, mut a1, mut a2, mut a3) =
            (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let mut j = 0usize;
        while j + 16 <= n {
            let wb = vld1q_s8(wp.add(j));
            let lo = vmovl_s8(vget_low_s8(wb));
            let hi = vmovl_s8(vget_high_s8(wb));
            let w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
            let w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
            let w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
            let w3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
            a0 = vfmaq_f32(a0, w0, vld1q_f32(xp.add(j)));
            a1 = vfmaq_f32(a1, w1, vld1q_f32(xp.add(j + 4)));
            a2 = vfmaq_f32(a2, w2, vld1q_f32(xp.add(j + 8)));
            a3 = vfmaq_f32(a3, w3, vld1q_f32(xp.add(j + 12)));
            j += 16;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        while j < n {
            sum += (*wp.add(j)) as f32 * *xp.add(j);
            j += 1;
        }
        sum
}
}

#[allow(clippy::too_many_arguments)]
fn qmatvec(
    q: &[u8],
    row_scale: &[f32],
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), rows);

    #[cfg(target_arch = "aarch64")]
    if sdot_enabled() {
        let act = split_act(xs);
        let out_addr = SendMut(out.as_mut_ptr());
        let run_range = |start: usize, end: usize| {
            let mut o = start;
            while o + 4 <= end {
                let r = unsafe {
                    dot_i8_sdot_4rows(
                        &q[o * cols..(o + 1) * cols],
                        &q[(o + 1) * cols..(o + 2) * cols],
                        &q[(o + 2) * cols..(o + 3) * cols],
                        &q[(o + 3) * cols..(o + 4) * cols],
                        &act.xq,
                    )
                };
                for k in 0..4 {
                    let mut acc = r[k] as f32 * act.sx;
                    for &(j, xv) in &act.outliers {
                        acc += (q[(o + k) * cols + j] as i8) as f32 * xv;
                    }
                    // SAFETY: disjoint row ranges per worker.
                    unsafe { *out_addr.at(o + k) = acc * row_scale[o + k] };
                }
                o += 4;
            }
            while o < end {
                let v = row_dot_sdot(&q[o * cols..(o + 1) * cols], &act) * row_scale[o];
                unsafe { *out_addr.at(o) = v };
                o += 1;
            }
        };
        match pool {
            Some(pool) if rows >= 256 => {
                pool.run(&|widx, n| {
                    let chunk = rows.div_ceil(n);
                    let start = widx * chunk;
                    run_range(start, (start + chunk).min(rows));
                });
            }
            _ => run_range(0, rows),
        }
        return;
    }

    let row_dot = |o: usize| -> f32 { dot_i8_f32(&q[o * cols..(o + 1) * cols], xs) * row_scale[o] };
    match pool {
        Some(pool) if rows >= 256 => {
            let out_addr = SendMut(out.as_mut_ptr());
            pool.run(&move |widx, n| {
                let chunk = rows.div_ceil(n);
                let start = widx * chunk;
                let end = (start + chunk).min(rows);
                for o in start..end {
                    // SAFETY: disjoint row ranges per worker.
                    unsafe { *out_addr.at(o) = row_dot(o) };
                }
            });
        }
        _ => {
            for (o, dst) in out.iter_mut().enumerate() {
                *dst = row_dot(o);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn qmatvec2(
    q: &[u8],
    row_scale: &[f32],
    x1: &[f32],
    x2: &[f32],
    rows: usize,
    cols: usize,
    o1: &mut [f32],
    o2: &mut [f32],
    pool: Option<&Pool>,
) {
    #[cfg(target_arch = "aarch64")]
    if sdot_enabled() {
        let a1s = split_act(x1);
        let a2s = split_act(x2);
        let p1 = SendMut(o1.as_mut_ptr());
        let p2 = SendMut(o2.as_mut_ptr());
        let run_range = |start: usize, end: usize| {
            for o in start..end {
                let row = &q[o * cols..(o + 1) * cols];
                // SAFETY: disjoint row ranges per worker.
                unsafe {
                    *p1.at(o) = row_dot_sdot(row, &a1s) * row_scale[o];
                    *p2.at(o) = row_dot_sdot(row, &a2s) * row_scale[o];
                }
            }
        };
        match pool {
            Some(pool) if rows >= 256 => {
                pool.run(&|widx, n| {
                    let chunk = rows.div_ceil(n);
                    let start = widx * chunk;
                    run_range(start, (start + chunk).min(rows));
                });
            }
            _ => run_range(0, rows),
        }
        return;
    }

    let row_dots = |o: usize| -> (f32, f32) {
        let row = &q[o * cols..(o + 1) * cols];
        (dot_i8_f32(row, x1) * row_scale[o], dot_i8_f32(row, x2) * row_scale[o])
    };
    match pool {
        Some(pool) if rows >= 256 => {
            let p1 = SendMut(o1.as_mut_ptr());
            let p2 = SendMut(o2.as_mut_ptr());
            pool.run(&move |widx, n| {
                let chunk = rows.div_ceil(n);
                let start = widx * chunk;
                let end = (start + chunk).min(rows);
                for o in start..end {
                    let (s1, s2) = row_dots(o);
                    // SAFETY: disjoint row ranges per worker.
                    unsafe {
                        *p1.at(o) = s1;
                        *p2.at(o) = s2;
                    }
                }
            });
        }
        _ => {
            for o in 0..rows {
                let (s1, s2) = row_dots(o);
                o1[o] = s1;
                o2[o] = s2;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct SendMut(*mut f32);
unsafe impl Send for SendMut {}
unsafe impl Sync for SendMut {}

impl SendMut {
    #[inline]
    fn at(self, i: usize) -> *mut f32 {
        unsafe { self.0.add(i) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_matvec_matches_matvec_rows_bitexact() {
        let (rows, cols) = (300, 40);
        let w: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.017).sin()).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.05).cos()).collect();
        let qt = QTensor::from_f32(w.clone(), rows, cols);

        let mut a = vec![0.0f32; rows];
        matvec_rows(None, &w, &x, &mut a);
        let mut b = vec![0.0f32; rows];
        qt.matvec(&x, &mut b, None);
        assert_eq!(a, b);
    }

    #[test]
    fn sdot_kernel_exact_on_grid() {
        // Activations already on the i8 grid (±1 with amax=1 → sx=1/127,
        // xq=±127 dequantizes EXACTLY) → the SDOT path must match the
        // exact f32 dot to float rounding. This isolates kernel
        // correctness from quantization noise.
        eprintln!("sdot_enabled = {}", sdot_enabled());
        let (rows, cols) = (9, 80); // odd rows → exercises 4-row + tail
        let w: Vec<u8> = (0..rows * cols)
            .map(|i| (((i * 37) % 251) as i32 - 125) as i8 as u8)
            .collect();
        let scales: Vec<f32> = (0..rows).map(|o| 0.005 + o as f32 * 0.001).collect();
        let x: Vec<f32> = (0..cols)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        let mut a = vec![0.0f32; rows];
        qmatvec(&w, &scales, &x, rows, cols, &mut a, None);
        for o in 0..rows {
            let mut acc = 0.0f32;
            for j in 0..cols {
                acc += (w[o * cols + j] as i8) as f32 * x[j];
            }
            let expect = acc * scales[o];
            assert!(
                (a[o] - expect).abs() < 1e-3 * expect.abs().max(1e-3),
                "row {o}: {} vs {expect}",
                a[o]
            );
        }
    }

    #[test]
    fn sdot_a8w8_noise_is_bounded() {
        // Off-grid activations: A8 quantization noise must stay small in
        // relative L2 over the whole output (realistic accuracy contract;
        // vmfcore measured argmax-identical decode on real models).
        let (rows, cols) = (16, 512);
        let w: Vec<u8> = (0..rows * cols)
            .map(|i| (((i * 37) % 251) as i32 - 125) as i8 as u8)
            .collect();
        let scales = vec![0.01f32; rows];
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.21).sin()).collect();
        let mut a = vec![0.0f32; rows];
        qmatvec(&w, &scales, &x, rows, cols, &mut a, None);
        let (mut num, mut den) = (0f64, 0f64);
        for o in 0..rows {
            let mut acc = 0.0f32;
            for j in 0..cols {
                acc += (w[o * cols + j] as i8) as f32 * x[j];
            }
            let expect = acc * scales[o];
            num += ((a[o] - expect) as f64).powi(2);
            den += (expect as f64).powi(2);
        }
        let rel = (num / den.max(1e-12)).sqrt();
        assert!(rel < 0.05, "A8W8 relative L2 error too high: {rel}");
    }

    #[test]
    fn i8_dot_neon_matches_scalar() {
        let n = 100;
        let w: Vec<u8> = (0..n).map(|i| ((i * 37 + 11) % 251) as u8).collect();
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin()).collect();
        let mut scalar = 0.0f32;
        for j in 0..n {
            scalar += (w[j] as i8) as f32 * x[j];
        }
        let fast = dot_i8_f32(&w, &x);
        assert!((scalar - fast).abs() < 1e-3 * scalar.abs().max(1.0));
    }

    /// Fused vbit matvec must match full dequant_vbit + dense matvec.
    #[test]
    fn vbitmatvec_matches_full_dequant() {
        let (rows, cols) = (6, 64);
        let ng = cols / GROUP_SIZE;
        // Hand-craft: bits per row, f16 scales, packed rows.
        let bits: Vec<u8> = vec![3, 4, 5, 6, 8, 4];
        let mut bytes = bits.clone();
        for g in 0..rows * ng {
            let s = 0.02 + 0.001 * g as f32;
            bytes.extend_from_slice(&cortiq_core::quant::f32_to_f16(s).to_le_bytes());
        }
        for r in 0..rows {
            let b = bits[r] as usize;
            let (mut acc, mut nb) = (0u64, 0usize);
            let mut rowbytes = Vec::new();
            for i in 0..cols {
                let v = ((i * 7 + r * 13) % (1 << b)) as u64;
                acc = (acc << b) | v;
                nb += b;
                while nb >= 8 {
                    nb -= 8;
                    rowbytes.push(((acc >> nb) & 0xFF) as u8);
                }
            }
            if nb > 0 {
                rowbytes.push(((acc << (8 - nb)) & 0xFF) as u8);
            }
            bytes.extend_from_slice(&rowbytes);
        }
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.19).sin()).collect();

        let mut reference = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_vbit(&bytes, rows, cols, &mut reference).unwrap();
        let mut expect = vec![0f32; rows];
        for r in 0..rows {
            expect[r] = reference[r * cols..(r + 1) * cols]
                .iter()
                .zip(&x)
                .map(|(w, xv)| w * xv)
                .sum();
        }
        let mut got = vec![0f32; rows];
        vbitmatvec(&bytes, &x, rows, cols, &mut got, None);
        // SDOT path quantizes activations to i8 (A8W8): bounded noise,
        // same contract as q8 (exact path is pinned by CMF_SDOT=0 in
        // the golden-parity gate).
        let tol = if sdot_enabled() { 6e-2 } else { 1e-4 };
        let scale = expect.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        for r in 0..rows {
            assert!(
                (got[r] - expect[r]).abs() < tol * scale,
                "row {r}: {} vs {}",
                got[r],
                expect[r]
            );
        }
    }

    /// Fused q4 matvec must match the reference full-dequant + dense
    /// matvec bit-for-bit in structure (same f32 math, group order).
    #[test]
    fn q4matvec_matches_full_dequant() {
        let (rows, cols) = (8, 64);
        let groups = rows * cols / GROUP_SIZE;
        // Hand-craft a q4_block blob: nibbles then f16 scales.
        let mut bytes = Vec::with_capacity(groups * 16 + groups * 2);
        for i in 0..groups * 16 {
            bytes.push((((i * 7 + 3) % 256) & 0xFF) as u8);
        }
        for g in 0..groups {
            let s = 0.01 + 0.003 * g as f32;
            bytes.extend_from_slice(&cortiq_core::quant::f32_to_f16(s).to_le_bytes());
        }
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.17).sin()).collect();

        let mut reference = vec![0.0f32; rows * cols];
        cortiq_core::quant::dequant_q4_block(&bytes, &mut reference);
        let mut expect = vec![0.0f32; rows];
        for r in 0..rows {
            expect[r] = reference[r * cols..(r + 1) * cols]
                .iter()
                .zip(&x)
                .map(|(w, xv)| w * xv)
                .sum();
        }

        let mut got = vec![0.0f32; rows];
        q4matvec(&bytes, &x, rows, cols, &mut got, None);
        for r in 0..rows {
            assert!(
                (got[r] - expect[r]).abs() < 1e-4 * expect[r].abs().max(1.0),
                "row {r}: {} vs {}",
                got[r],
                expect[r]
            );
        }
    }
}
