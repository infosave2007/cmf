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
        /// Vbit only: byte offset of each row's packed data within the
        /// tensor blob (`[rows + 1]`, computed once at load — the per-
        /// matvec prefix scan over row bit-widths was O(rows) each call).
        vbit_offsets: Vec<usize>,
    },
}

/// Prefix-sum of vbit row payload offsets (absolute within the tensor
/// bytes). `offsets[r]..offsets[r+1]` is row r's packed data.
fn vbit_row_offsets(bytes: &[u8], rows: usize, cols: usize) -> Vec<usize> {
    let ng = cols / GROUP_SIZE;
    let bits = &bytes[..rows];
    let mut offsets = Vec::with_capacity(rows + 1);
    let mut off = rows + rows * ng * 2;
    for r in 0..rows {
        offsets.push(off);
        off += (cols * bits[r] as usize + 7) / 8;
    }
    offsets.push(off);
    offsets
}

impl QTensor {
    pub fn from_f32(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self::F32 { data, rows, cols }
    }

    /// Wrap a directory tensor without dequantizing the payload.
    /// Falls back to dequantized f32 for dtypes without a fused kernel.
    pub fn from_model(model: &Arc<CmfModel>, name: &str) -> Result<Self, String> {
        // Indexed lookup: the linear directory scan made pipeline build
        // O(N²) on MoE/skills files with thousands of tensors.
        let idx = model
            .tensor_index(name)
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
                    vbit_offsets: Vec::new(),
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
                vbit_offsets: vbit_row_offsets(bytes, rows, cols),
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
                vbit_offsets: Vec::new(),
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
                vbit_offsets,
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
                    // Precomputed at load — embedding lookup used to scan
                    // the bit-widths of every preceding row (O(token_id)).
                    let off = vbit_offsets[r];
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
                    dot_i8_f32(q, x) * row_scale[r]
                }
                TensorDtype::Q8_2f => {
                    let q = &self.quant_bytes()[r * cols..(r + 1) * cols];
                    dot_i8_col_f32(q, x, col_field) * row_scale[r]
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
                vbit_offsets,
            } => {
                let _ = (model, idx);
                if *dtype == TensorDtype::Q4Block {
                    q4matvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                    return;
                }
                if *dtype == TensorDtype::Vbit {
                    vbitmatvec(self.quant_bytes(), vbit_offsets, x, *rows, *cols, out, pool);
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
                vbit_offsets,
                ..
            } => {
                if *dtype == TensorDtype::Q4Block {
                    q4matvec2(self.quant_bytes(), x1, x2, *rows, *cols, o1, o2, pool);
                    return;
                }
                if *dtype == TensorDtype::Vbit {
                    vbitmatvec2(self.quant_bytes(), vbit_offsets, x1, x2, *rows, *cols, o1, o2, pool);
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
                vbit_offsets,
                ..
            } => {
                if *dtype == TensorDtype::Q4Block {
                    q4matmat(self.quant_bytes(), xs_all, b, rows, cols, out, pool);
                    return;
                }
                if *dtype == TensorDtype::Vbit {
                    vbitmatmat(self.quant_bytes(), vbit_offsets, xs_all, b, rows, cols, out, pool);
                    return;
                }
                let pre: Vec<std::borrow::Cow<'_, [f32]>> = (0..b)
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
    pre: &[std::borrow::Cow<'_, [f32]>],
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

/// Split rows across pool workers (shared qmatvec pattern). Self-balancing
/// — see `Pool::run_rows` for why a static 1/n split is wrong here.
fn dispatch_rows(pool: Option<&Pool>, rows: usize, run: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(pool) if rows >= 256 => pool.run_rows(rows, run),
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
/// MSB-first, byte-padded]. Row data offsets are precomputed at load
/// (`vbit_row_offsets`) — the per-call prefix scan was O(rows) pure
/// overhead on every matvec.
#[allow(clippy::too_many_arguments)]
fn vbitmatvec(
    bytes: &[u8],
    offsets: &[usize],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), rows);
    debug_assert_eq!(offsets.len(), rows + 1);
    let ng = cols / GROUP_SIZE;
    let bits = &bytes[..rows];
    let sc_off = rows;

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
            // Per-worker scratch: this closure runs for every row of the
            // tensor (lm_head ≈ 150k rows/token) — a heap allocation per
            // row was measurable pure overhead.
            thread_local! {
                static VBIT_SCRATCH: std::cell::RefCell<Vec<u8>> =
                    const { std::cell::RefCell::new(Vec::new()) };
            }
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
            VBIT_SCRATCH.with(|scratch| {
                let mut buf = scratch.borrow_mut();
                buf.resize(cols, 0);
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
            })
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

/// Fused two-input vbit matvec: each row is unpacked from the mmap ONCE
/// and dotted against BOTH activations (MTP verify / pair prefill used
/// to run two full matvecs — double weight traffic and double unpack).
/// Per-input math is identical to `vbitmatvec` → same accuracy contract.
#[allow(clippy::too_many_arguments)]
fn vbitmatvec2(
    bytes: &[u8],
    offsets: &[usize],
    x1: &[f32],
    x2: &[f32],
    rows: usize,
    cols: usize,
    o1: &mut [f32],
    o2: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(o1.len(), rows);
    debug_assert_eq!(o2.len(), rows);
    let ng = cols / GROUP_SIZE;
    let bits = &bytes[..rows];
    let sc_off = rows;

    #[cfg(target_arch = "aarch64")]
    if sdot_enabled() {
        let a1 = split_act(x1);
        let a2 = split_act(x2);
        let (a1, a2) = (&a1, &a2);
        let row_dots = move |r: usize| -> (f32, f32) {
            let b = bits[r] as usize;
            let l = (1i32 << (b - 1)) - 1;
            let data = &bytes[offsets[r]..offsets[r + 1]];
            if b == 8 {
                // u−L reaches 128 → does not fit i8; exact f32 path,
                // bits still streamed once for both lanes.
                let (mut acc, mut nbits, mut idx) = (0u64, 0usize, 0usize);
                let (mut d1, mut d2) = (0f32, 0f32);
                for g in 0..ng {
                    let so = (r * ng + g) * 2;
                    let sgf = f16_to_f32(u16::from_le_bytes([
                        bytes[sc_off + so],
                        bytes[sc_off + so + 1],
                    ]));
                    let (mut g1, mut g2) = (0f32, 0f32);
                    for k in 0..GROUP_SIZE {
                        if nbits < 8 {
                            acc = (acc << 8) | data[idx] as u64;
                            idx += 1;
                            nbits += 8;
                        }
                        let u = ((acc >> (nbits - 8)) & 0xFF) as i32;
                        nbits -= 8;
                        let w = (u - l) as f32;
                        g1 += w * x1[g * GROUP_SIZE + k];
                        g2 += w * x2[g * GROUP_SIZE + k];
                    }
                    d1 += g1 * sgf;
                    d2 += g2 * sgf;
                }
                return (d1, d2);
            }
            thread_local! {
                static VBIT_SCRATCH2: std::cell::RefCell<Vec<u8>> =
                    const { std::cell::RefCell::new(Vec::new()) };
            }
            #[inline(always)]
            fn fill<const B: usize>(data: &[u8], l: i32, buf: &mut [u8]) {
                for (blk, chunk) in buf.chunks_exact_mut(8).enumerate() {
                    let u = unpack8::<B>(&data[blk * B..]);
                    for k in 0..8 {
                        chunk[k] = (u[k] - l) as i8 as u8;
                    }
                }
            }
            VBIT_SCRATCH2.with(|scratch| {
                let mut buf = scratch.borrow_mut();
                buf.resize(cols, 0);
                match b {
                    3 => fill::<3>(data, l, &mut buf),
                    4 => fill::<4>(data, l, &mut buf),
                    5 => fill::<5>(data, l, &mut buf),
                    6 => fill::<6>(data, l, &mut buf),
                    _ => unreachable!(),
                }
                let (mut d1, mut d2) = (0f32, 0f32);
                for g in 0..ng {
                    let so = (r * ng + g) * 2;
                    let s = f16_to_f32(u16::from_le_bytes([
                        bytes[sc_off + so],
                        bytes[sc_off + so + 1],
                    ]));
                    let wg = &buf[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
                    let v1 = unsafe {
                        dot_i8_sdot(wg, &a1.xq[g * GROUP_SIZE..(g + 1) * GROUP_SIZE])
                    } as f32
                        * a1.sx;
                    let v2 = unsafe {
                        dot_i8_sdot(wg, &a2.xq[g * GROUP_SIZE..(g + 1) * GROUP_SIZE])
                    } as f32
                        * a2.sx;
                    d1 += v1 * s;
                    d2 += v2 * s;
                }
                for &(j, xv) in &a1.outliers {
                    let so = (r * ng + j / GROUP_SIZE) * 2;
                    let s = f16_to_f32(u16::from_le_bytes([
                        bytes[sc_off + so],
                        bytes[sc_off + so + 1],
                    ]));
                    d1 += (buf[j] as i8) as f32 * s * xv;
                }
                for &(j, xv) in &a2.outliers {
                    let so = (r * ng + j / GROUP_SIZE) * 2;
                    let s = f16_to_f32(u16::from_le_bytes([
                        bytes[sc_off + so],
                        bytes[sc_off + so + 1],
                    ]));
                    d2 += (buf[j] as i8) as f32 * s * xv;
                }
                (d1, d2)
            })
        };
        let p1 = SendMut(o1.as_mut_ptr());
        let p2 = SendMut(o2.as_mut_ptr());
        let run = move |start: usize, end: usize| {
            for r in start..end {
                let (v1, v2) = row_dots(r);
                // SAFETY: disjoint row ranges per worker.
                unsafe {
                    *p1.at(r) = v1;
                    *p2.at(r) = v2;
                }
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }

    // Scalar path: per-bit-width specialized, two accumulators per row —
    // per-lane accumulation order matches `vbitmatvec` exactly.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn dot_row2<const B: usize>(
        data: &[u8],
        bytes: &[u8],
        sc_off: usize,
        r: usize,
        ng: usize,
        x1: &[f32],
        x2: &[f32],
    ) -> (f32, f32) {
        let l = ((1i32 << (B - 1)) - 1) as f32;
        let gbytes = GROUP_SIZE * B / 8;
        let (mut d1, mut d2) = (0f32, 0f32);
        for g in 0..ng {
            let so = (r * ng + g) * 2;
            let s = f16_to_f32(u16::from_le_bytes([bytes[sc_off + so], bytes[sc_off + so + 1]]));
            let x1g = &x1[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
            let x2g = &x2[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
            let gd0 = &data[g * gbytes..(g + 1) * gbytes];
            let (mut g1, mut g2) = (0f32, 0f32);
            for blk in 0..GROUP_SIZE / 8 {
                let u = unpack8::<B>(&gd0[blk * B..]);
                for k in 0..8 {
                    let w = u[k] as f32 - l;
                    g1 += w * x1g[blk * 8 + k];
                    g2 += w * x2g[blk * 8 + k];
                }
            }
            d1 += g1 * s;
            d2 += g2 * s;
        }
        (d1, d2)
    }
    let row_dots = move |r: usize| -> (f32, f32) {
        let data = &bytes[offsets[r]..offsets[r + 1]];
        match bits[r] {
            3 => dot_row2::<3>(data, bytes, sc_off, r, ng, x1, x2),
            4 => dot_row2::<4>(data, bytes, sc_off, r, ng, x1, x2),
            5 => dot_row2::<5>(data, bytes, sc_off, r, ng, x1, x2),
            6 => dot_row2::<6>(data, bytes, sc_off, r, ng, x1, x2),
            8 => dot_row2::<8>(data, bytes, sc_off, r, ng, x1, x2),
            b => unreachable!("vbit bit-width {b} (validated at load)"),
        }
    };
    let p1 = SendMut(o1.as_mut_ptr());
    let p2 = SendMut(o2.as_mut_ptr());
    let run = move |start: usize, end: usize| {
        for r in start..end {
            let (v1, v2) = row_dots(r);
            // SAFETY: disjoint row ranges per worker.
            unsafe {
                *p1.at(r) = v1;
                *p2.at(r) = v2;
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// Fused q4_block matvec straight from the mapped bytes. SDOT path when
/// dotprod is available (port of vmfcore `dot_q4_block_sdot`, measured
/// +23% on q4 decode): nibbles → centered i8, int8×int8 `sdot` per
/// 32-group, exact outlier correction — the same A8W8 contract as q8.
/// `CMF_SDOT=0` keeps the exact scalar path.
fn q4matvec(bytes: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut [f32], pool: Option<&Pool>) {
    debug_assert_eq!(out.len(), rows);
    let (packed, scales) = q4_split(bytes, rows, cols);
    let gpr = cols / GROUP_SIZE;

    #[cfg(target_arch = "aarch64")]
    if sdot_enabled() {
        let act = split_act(x);
        let act = &act;
        let row_dot = move |r: usize| -> f32 {
            let mut acc =
                unsafe { dot_q4_row_sdot(packed, scales, r * gpr, gpr, &act.xq) } * act.sx;
            // xq is zeroed at outlier slots — add the exact terms.
            for &(j, xv) in &act.outliers {
                let flat = r * cols + j;
                let byte = packed[flat / 2];
                let nib = if flat & 1 == 0 { byte & 0x0F } else { byte >> 4 };
                let s = f16_to_f32(u16::from_le_bytes([
                    scales[(flat / GROUP_SIZE) * 2],
                    scales[(flat / GROUP_SIZE) * 2 + 1],
                ]));
                acc += ((nib as i32 - 8) as f32) * s * xv;
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
        return;
    }

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

/// Fused two-input q4 matvec: nibbles are unpacked ONCE per group and
/// dotted against both activations (was: two full matvecs — double
/// weight traffic). Per-lane math matches `q4matvec` exactly.
#[allow(clippy::too_many_arguments)]
fn q4matvec2(
    bytes: &[u8],
    x1: &[f32],
    x2: &[f32],
    rows: usize,
    cols: usize,
    o1: &mut [f32],
    o2: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(o1.len(), rows);
    debug_assert_eq!(o2.len(), rows);
    let (packed, scales) = q4_split(bytes, rows, cols);
    let gpr = cols / GROUP_SIZE;

    #[cfg(target_arch = "aarch64")]
    if sdot_enabled() {
        let a1 = split_act(x1);
        let a2 = split_act(x2);
        let (a1, a2) = (&a1, &a2);
        let row_dots = move |r: usize| -> (f32, f32) {
            let (s1, s2) =
                unsafe { dot_q4_row_sdot2(packed, scales, r * gpr, gpr, &a1.xq, &a2.xq) };
            let mut acc1 = s1 * a1.sx;
            let mut acc2 = s2 * a2.sx;
            // xq is zeroed at outlier slots — add the exact terms.
            let fix = |outliers: &[(usize, f32)], acc: &mut f32| {
                for &(j, xv) in outliers {
                    let flat = r * cols + j;
                    let byte = packed[flat / 2];
                    let nib = if flat & 1 == 0 { byte & 0x0F } else { byte >> 4 };
                    let s = f16_to_f32(u16::from_le_bytes([
                        scales[(flat / GROUP_SIZE) * 2],
                        scales[(flat / GROUP_SIZE) * 2 + 1],
                    ]));
                    *acc += ((nib as i32 - 8) as f32) * s * xv;
                }
            };
            fix(&a1.outliers, &mut acc1);
            fix(&a2.outliers, &mut acc2);
            (acc1, acc2)
        };
        let p1 = SendMut(o1.as_mut_ptr());
        let p2 = SendMut(o2.as_mut_ptr());
        let run = move |start: usize, end: usize| {
            for r in start..end {
                let (v1, v2) = row_dots(r);
                // SAFETY: disjoint row ranges per worker.
                unsafe {
                    *p1.at(r) = v1;
                    *p2.at(r) = v2;
                }
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }

    let row_dots = |r: usize| -> (f32, f32) {
        let (mut acc1, mut acc2) = (0f32, 0f32);
        for gi in 0..gpr {
            let g = r * gpr + gi;
            let s = f16_to_f32(u16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]));
            let pk = &packed[g * 16..(g + 1) * 16];
            let x1g = &x1[gi * GROUP_SIZE..(gi + 1) * GROUP_SIZE];
            let x2g = &x2[gi * GROUP_SIZE..(gi + 1) * GROUP_SIZE];
            let (mut g1, mut g2) = (0f32, 0f32);
            for (k, &b) in pk.iter().enumerate() {
                let wl = (b & 0x0F) as f32 - 8.0;
                let wh = ((b >> 4) & 0x0F) as f32 - 8.0;
                g1 += wl * x1g[k * 2] + wh * x1g[k * 2 + 1];
                g2 += wl * x2g[k * 2] + wh * x2g[k * 2 + 1];
            }
            acc1 += g1 * s;
            acc2 += g2 * s;
        }
        (acc1, acc2)
    };
    let p1 = SendMut(o1.as_mut_ptr());
    let p2 = SendMut(o2.as_mut_ptr());
    let run = move |start: usize, end: usize| {
        for r in start..end {
            let (v1, v2) = row_dots(r);
            // SAFETY: disjoint row ranges per worker.
            unsafe {
                *p1.at(r) = v1;
                *p2.at(r) = v2;
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

thread_local! {
    /// Per-worker decoded-row scratch for the batched q4/vbit kernels
    /// (centered i8 for SDOT, f32 for the exact/scalar paths).
    static ROW_I8: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    static ROW_F32: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Batched q4 matmat: each weight row is unpacked from the mmap ONCE
/// and dotted against ALL b activations (prefill used to fall back to b
/// full matvecs — b× weight traffic and b× nibble decode). Per-position
/// math matches `q4matvec` exactly: same group order, same accumulation.
/// `out` is row-major [b, rows] like `qmatmat`.
#[allow(clippy::too_many_arguments)]
fn q4matmat(
    bytes: &[u8],
    xs_all: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(xs_all.len(), b * cols);
    debug_assert_eq!(out.len(), b * rows);
    let (packed, scales) = q4_split(bytes, rows, cols);
    let gpr = cols / GROUP_SIZE;
    let gscale = |g: usize| f16_to_f32(u16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]));

    #[cfg(target_arch = "aarch64")]
    if sdot_enabled() {
        let acts: Vec<SplitAct> =
            (0..b).map(|bi| split_act(&xs_all[bi * cols..(bi + 1) * cols])).collect();
        let acts = &acts;
        let out_addr = SendMut(out.as_mut_ptr());
        let run = move |start: usize, end: usize| {
            ROW_I8.with(|rb| {
                let mut buf = rb.borrow_mut();
                buf.resize(cols, 0);
                for r in start..end {
                    // Unpack the row's nibbles to centered i8 once
                    // (element 2k = low nibble, 2k+1 = high — flat order,
                    // same as dot_q4_row_sdot's zip).
                    for gi in 0..gpr {
                        let g = r * gpr + gi;
                        for (k, &bt) in packed[g * 16..(g + 1) * 16].iter().enumerate() {
                            buf[gi * GROUP_SIZE + k * 2] = ((bt & 0x0F) as i32 - 8) as i8 as u8;
                            buf[gi * GROUP_SIZE + k * 2 + 1] =
                                (((bt >> 4) & 0x0F) as i32 - 8) as i8 as u8;
                        }
                    }
                    for (bi, act) in acts.iter().enumerate() {
                        let mut acc = 0f32;
                        for gi in 0..gpr {
                            let d = unsafe {
                                dot_i8_sdot(
                                    &buf[gi * GROUP_SIZE..(gi + 1) * GROUP_SIZE],
                                    &act.xq[gi * GROUP_SIZE..(gi + 1) * GROUP_SIZE],
                                )
                            };
                            acc += d as f32 * gscale(r * gpr + gi);
                        }
                        acc *= act.sx;
                        // xq is zeroed at outlier slots — exact terms.
                        for &(j, xv) in &act.outliers {
                            acc += (buf[j] as i8) as f32 * gscale((r * cols + j) / GROUP_SIZE) * xv;
                        }
                        // SAFETY: disjoint (bi, r) cells per worker row range.
                        unsafe { *out_addr.at(bi * rows + r) = acc };
                    }
                }
            })
        };
        dispatch_rows(pool, rows, &run);
        return;
    }

    let out_addr = SendMut(out.as_mut_ptr());
    let run = move |start: usize, end: usize| {
        ROW_F32.with(|rb| {
            let mut buf = rb.borrow_mut();
            buf.resize(cols, 0.0);
            for r in start..end {
                // Decode raw (nib − 8) values once; scales stay per-group
                // so the accumulation order matches q4matvec bit-for-bit.
                for gi in 0..gpr {
                    let g = r * gpr + gi;
                    for (k, &bt) in packed[g * 16..(g + 1) * 16].iter().enumerate() {
                        buf[gi * GROUP_SIZE + k * 2] = (bt & 0x0F) as f32 - 8.0;
                        buf[gi * GROUP_SIZE + k * 2 + 1] = ((bt >> 4) & 0x0F) as f32 - 8.0;
                    }
                }
                for bi in 0..b {
                    let x = &xs_all[bi * cols..(bi + 1) * cols];
                    let mut acc = 0f32;
                    for gi in 0..gpr {
                        let mut ga = 0f32;
                        for k in 0..GROUP_SIZE {
                            ga += buf[gi * GROUP_SIZE + k] * x[gi * GROUP_SIZE + k];
                        }
                        acc += ga * gscale(r * gpr + gi);
                    }
                    // SAFETY: disjoint (bi, r) cells per worker row range.
                    unsafe { *out_addr.at(bi * rows + r) = acc };
                }
            }
        })
    };
    dispatch_rows(pool, rows, &run);
}

/// Batched vbit matmat: each variable-bit row is decoded from the mmap
/// ONCE for the whole microbatch. Same per-position math as
/// `vbitmatvec` (SDOT A8W8 with exact outliers / exact f32 for b=8 rows
/// and the scalar path).
#[allow(clippy::too_many_arguments)]
fn vbitmatmat(
    bytes: &[u8],
    offsets: &[usize],
    xs_all: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(xs_all.len(), b * cols);
    debug_assert_eq!(out.len(), b * rows);
    debug_assert_eq!(offsets.len(), rows + 1);
    let ng = cols / GROUP_SIZE;
    let bits = &bytes[..rows];
    let sc_off = rows;
    let gscale = |r: usize, g: usize| {
        let so = (r * ng + g) * 2;
        f16_to_f32(u16::from_le_bytes([bytes[sc_off + so], bytes[sc_off + so + 1]]))
    };

    // Decode row r's raw (u − L) values into `dst` (f32, unscaled).
    let decode_f32 = |r: usize, dst: &mut [f32]| {
        let bw = bits[r] as usize;
        let l = ((1i32 << (bw - 1)) - 1) as f32;
        let data = &bytes[offsets[r]..offsets[r + 1]];
        let (mut acc, mut nbits, mut idx) = (0u64, 0usize, 0usize);
        for d in dst.iter_mut() {
            while nbits < bw {
                acc = (acc << 8) | data[idx] as u64;
                idx += 1;
                nbits += 8;
            }
            let u = ((acc >> (nbits - bw)) & ((1u64 << bw) - 1)) as f32;
            nbits -= bw;
            *d = u - l;
        }
    };

    #[cfg(target_arch = "aarch64")]
    if sdot_enabled() {
        let acts: Vec<SplitAct> =
            (0..b).map(|bi| split_act(&xs_all[bi * cols..(bi + 1) * cols])).collect();
        let acts = &acts;
        let out_addr = SendMut(out.as_mut_ptr());
        let run = move |start: usize, end: usize| {
            for r in start..end {
                let bw = bits[r] as usize;
                if bw == 8 {
                    // u−L reaches 128 → no i8 path; decode once, exact
                    // f32 dots for every position (same as vbitmatvec).
                    ROW_F32.with(|rb| {
                        let mut buf = rb.borrow_mut();
                        buf.resize(cols, 0.0);
                        decode_f32(r, &mut buf);
                        for bi in 0..b {
                            let x = &xs_all[bi * cols..(bi + 1) * cols];
                            let mut dot = 0f32;
                            for g in 0..ng {
                                let mut gd = 0f32;
                                for k in 0..GROUP_SIZE {
                                    gd += buf[g * GROUP_SIZE + k] * x[g * GROUP_SIZE + k];
                                }
                                dot += gd * gscale(r, g);
                            }
                            // SAFETY: disjoint (bi, r) cells per worker range.
                            unsafe { *out_addr.at(bi * rows + r) = dot };
                        }
                    });
                    continue;
                }
                let l = (1i32 << (bw - 1)) - 1;
                let data = &bytes[offsets[r]..offsets[r + 1]];
                ROW_I8.with(|rb| {
                    let mut buf = rb.borrow_mut();
                    buf.resize(cols, 0);
                    #[inline(always)]
                    fn fill<const B: usize>(data: &[u8], l: i32, buf: &mut [u8]) {
                        for (blk, chunk) in buf.chunks_exact_mut(8).enumerate() {
                            let u = unpack8::<B>(&data[blk * B..]);
                            for k in 0..8 {
                                chunk[k] = (u[k] - l) as i8 as u8;
                            }
                        }
                    }
                    match bw {
                        3 => fill::<3>(data, l, &mut buf),
                        4 => fill::<4>(data, l, &mut buf),
                        5 => fill::<5>(data, l, &mut buf),
                        6 => fill::<6>(data, l, &mut buf),
                        _ => unreachable!("vbit bit-width {bw} (validated at load)"),
                    }
                    for (bi, act) in acts.iter().enumerate() {
                        let mut dot = 0f32;
                        for g in 0..ng {
                            let d = unsafe {
                                dot_i8_sdot(
                                    &buf[g * GROUP_SIZE..(g + 1) * GROUP_SIZE],
                                    &act.xq[g * GROUP_SIZE..(g + 1) * GROUP_SIZE],
                                )
                            } as f32
                                * act.sx;
                            dot += d * gscale(r, g);
                        }
                        for &(j, xv) in &act.outliers {
                            dot += (buf[j] as i8) as f32 * gscale(r, j / GROUP_SIZE) * xv;
                        }
                        // SAFETY: disjoint (bi, r) cells per worker range.
                        unsafe { *out_addr.at(bi * rows + r) = dot };
                    }
                });
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }

    let out_addr = SendMut(out.as_mut_ptr());
    let run = move |start: usize, end: usize| {
        ROW_F32.with(|rb| {
            let mut buf = rb.borrow_mut();
            buf.resize(cols, 0.0);
            for r in start..end {
                decode_f32(r, &mut buf);
                for bi in 0..b {
                    let x = &xs_all[bi * cols..(bi + 1) * cols];
                    let mut dot = 0f32;
                    for g in 0..ng {
                        let mut gd = 0f32;
                        for k in 0..GROUP_SIZE {
                            gd += buf[g * GROUP_SIZE + k] * x[g * GROUP_SIZE + k];
                        }
                        dot += gd * gscale(r, g);
                    }
                    // SAFETY: disjoint (bi, r) cells per worker range.
                    unsafe { *out_addr.at(bi * rows + r) = dot };
                }
            }
        })
    };
    dispatch_rows(pool, rows, &run);
}

/// θ col-field fold for q8_2f activations. Borrowed pass-through for
/// every other dtype — the old unconditional `x.to_vec()` was a pure
/// per-matvec allocation on the q8_row hot path.
pub(crate) fn prescale<'a>(
    x: &'a [f32],
    col_field: &[f32],
    dtype: TensorDtype,
) -> std::borrow::Cow<'a, [f32]> {
    if dtype == TensorDtype::Q8_2f {
        x.iter().zip(col_field).map(|(a, c)| a * c).collect()
    } else {
        std::borrow::Cow::Borrowed(x)
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

thread_local! {
    /// Recycled xq buffers: split_act runs for every matvec (~200/token)
    /// and its hidden-size allocation was steady-state heap churn.
    static XQ_FREE: std::cell::RefCell<Vec<Vec<i8>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

impl Drop for SplitAct {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.xq);
        if buf.capacity() > 0 {
            XQ_FREE.with(|f| {
                let mut f = f.borrow_mut();
                if f.len() < 16 {
                    f.push(buf);
                }
            });
        }
    }
}

fn split_act(x: &[f32]) -> SplitAct {
    let n = x.len();
    let rms = (x.iter().map(|&v| (v * v) as f64).sum::<f64>() / n.max(1) as f64).sqrt() as f32;
    let thr = 8.0 * rms;
    // One pass: collect outliers and the bulk absmax (outliers excluded —
    // identical to the old zero-then-fold over a copied buffer, minus the
    // full-vector copy).
    let mut outliers: Vec<(usize, f32)> = Vec::new();
    let mut amax = 0f32;
    for (j, &v) in x.iter().enumerate() {
        let a = v.abs();
        if a > thr {
            outliers.push((j, v));
        } else if a > amax {
            amax = a;
        }
    }
    let sx = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let inv = 1.0 / sx;
    let mut xq = XQ_FREE.with(|f| f.borrow_mut().pop()).unwrap_or_default();
    xq.clear();
    xq.reserve(n);
    if outliers.is_empty() {
        xq.extend(x.iter().map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8));
    } else {
        // Outlier slots quantize to 0 (their exact term is added later).
        xq.extend(x.iter().map(|&v| {
            if v.abs() > thr {
                0
            } else {
                (v * inv).round().clamp(-127.0, 127.0) as i8
            }
        }));
    }
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

/// One q4 row via SDOT: each 32-group's nibbles unpack to centered i8
/// (nib−8 ∈ [−8,7]), int8×int8 `sdot` against the pre-quantized
/// activation group, × the group's f16 scale. Returns Σ_g dot_g·s_g;
/// the caller multiplies by the activation scale and adds the exact
/// outlier terms (port of vmfcore `dot_q4_block_sdot`, +23% measured).
/// Nibble order matches the writer: element 2k = low nibble, 2k+1 = high
/// → zip(lo,hi) restores flat order.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q4_row_sdot(packed: &[u8], scales: &[u8], g0: usize, gpr: usize, xq: &[i8]) -> f32 {
    // SAFETY: callers uphold slice-length contracts (16 packed bytes and
    // 2 scale bytes per group; xq.len() == gpr·GROUP_SIZE).
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let lomask = vdupq_n_u8(0x0F);
        let eight = vdupq_n_s8(8);
        let mut acc = 0f32;
        for gi in 0..gpr {
            let g = g0 + gi;
            let s = f16_to_f32(u16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]));
            let b = vld1q_u8(packed.as_ptr().add(g * 16));
            let lo = vandq_u8(b, lomask);
            let hi = vshrq_n_u8::<4>(b);
            let e0 = vsubq_s8(vreinterpretq_s8_u8(vzip1q_u8(lo, hi)), eight);
            let e1 = vsubq_s8(vreinterpretq_s8_u8(vzip2q_u8(lo, hi)), eight);
            let x0 = vld1q_s8(xq.as_ptr().add(gi * GROUP_SIZE));
            let x1 = vld1q_s8(xq.as_ptr().add(gi * GROUP_SIZE + 16));
            let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
            asm!(
                "sdot {a0:v}.4s, {e0:v}.16b, {x0:v}.16b",
                "sdot {a1:v}.4s, {e1:v}.16b, {x1:v}.16b",
                a0 = inout(vreg) a0, a1 = inout(vreg) a1,
                e0 = in(vreg) e0, x0 = in(vreg) x0, e1 = in(vreg) e1, x1 = in(vreg) x1,
                options(pure, nomem, nostack),
            );
            acc += vaddvq_s32(vaddq_s32(a0, a1)) as f32 * s;
        }
        acc
    }
}

/// Two-activation q4 row via SDOT: the nibble unpack (the expensive
/// part) happens ONCE per group; both pre-quantized activations are
/// dotted against the same centered i8 registers. Per-lane math matches
/// `dot_q4_row_sdot` exactly.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q4_row_sdot2(
    packed: &[u8],
    scales: &[u8],
    g0: usize,
    gpr: usize,
    xq1: &[i8],
    xq2: &[i8],
) -> (f32, f32) {
    // SAFETY: callers uphold slice-length contracts (16 packed bytes and
    // 2 scale bytes per group; xq*.len() == gpr·GROUP_SIZE).
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let lomask = vdupq_n_u8(0x0F);
        let eight = vdupq_n_s8(8);
        let (mut acc1, mut acc2) = (0f32, 0f32);
        for gi in 0..gpr {
            let g = g0 + gi;
            let s = f16_to_f32(u16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]));
            let b = vld1q_u8(packed.as_ptr().add(g * 16));
            let lo = vandq_u8(b, lomask);
            let hi = vshrq_n_u8::<4>(b);
            let e0 = vsubq_s8(vreinterpretq_s8_u8(vzip1q_u8(lo, hi)), eight);
            let e1 = vsubq_s8(vreinterpretq_s8_u8(vzip2q_u8(lo, hi)), eight);
            let x10 = vld1q_s8(xq1.as_ptr().add(gi * GROUP_SIZE));
            let x11 = vld1q_s8(xq1.as_ptr().add(gi * GROUP_SIZE + 16));
            let x20 = vld1q_s8(xq2.as_ptr().add(gi * GROUP_SIZE));
            let x21 = vld1q_s8(xq2.as_ptr().add(gi * GROUP_SIZE + 16));
            let (mut a0, mut a1, mut b0, mut b1) =
                (vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0));
            asm!(
                "sdot {a0:v}.4s, {e0:v}.16b, {x10:v}.16b",
                "sdot {a1:v}.4s, {e1:v}.16b, {x11:v}.16b",
                "sdot {b0:v}.4s, {e0:v}.16b, {x20:v}.16b",
                "sdot {b1:v}.4s, {e1:v}.16b, {x21:v}.16b",
                a0 = inout(vreg) a0, a1 = inout(vreg) a1,
                b0 = inout(vreg) b0, b1 = inout(vreg) b1,
                e0 = in(vreg) e0, e1 = in(vreg) e1,
                x10 = in(vreg) x10, x11 = in(vreg) x11,
                x20 = in(vreg) x20, x21 = in(vreg) x21,
                options(pure, nomem, nostack),
            );
            acc1 += vaddvq_s32(vaddq_s32(a0, a1)) as f32 * s;
            acc2 += vaddvq_s32(vaddq_s32(b0, b1)) as f32 * s;
        }
        (acc1, acc2)
    }
}

// ───────────────────── fused int8 kernels ─────────────────────

/// `acc += w · row` where the row is centered i8 — NEON widen+fma on
/// aarch64, scalar elsewhere. The KV-cache q8 value path rides on this.
#[inline]
pub(crate) fn axpy_i8_f32(acc: &mut [f32], row: &[i8], w: f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return axpy_i8_f32_neon(acc, row, w);
    }
    #[allow(unreachable_code)]
    {
        for (a, &b) in acc.iter_mut().zip(row) {
            *a += w * b as f32;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_i8_f32_neon(acc: &mut [f32], row: &[i8], w: f32) {
    // SAFETY: callers uphold slice-length contracts (see call sites).
    unsafe {
        use core::arch::aarch64::*;
        let n = acc.len().min(row.len());
        let ap = acc.as_mut_ptr();
        let rp = row.as_ptr();
        let wv = vdupq_n_f32(w);
        let mut j = 0usize;
        while j + 16 <= n {
            let rb = vld1q_s8(rp.add(j));
            let lo = vmovl_s8(vget_low_s8(rb));
            let hi = vmovl_s8(vget_high_s8(rb));
            for (off, half) in [(0, lo), (8, hi)] {
                let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(half)));
                let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(half)));
                let o = j + off;
                vst1q_f32(ap.add(o), vfmaq_f32(vld1q_f32(ap.add(o)), wv, f0));
                vst1q_f32(ap.add(o + 4), vfmaq_f32(vld1q_f32(ap.add(o + 4)), wv, f1));
            }
            j += 16;
        }
        while j < n {
            *ap.add(j) += w * (*rp.add(j)) as f32;
            j += 1;
        }
}
}

/// i8 row · f32 x. NEON on aarch64 (ported from vmfcore `dot_i8_f32_neon`,
/// ≈9× scalar), scalar elsewhere.
#[inline]
pub(crate) fn dot_i8_f32(w: &[u8], x: &[f32]) -> f32 {
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

/// i8 row · (x ⊙ col_field) — the q8_2f row dot with the θ col-field
/// folded into the product (no prescaled copy of x). NEON on aarch64,
/// scalar elsewhere. Used by the active-neuron path `row_dot`.
#[inline]
fn dot_i8_col_f32(w: &[u8], x: &[f32], col: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_i8_col_f32_neon(w, x, col);
    }
    #[allow(unreachable_code)]
    {
        let mut sum = 0.0f32;
        for (j, &b) in w.iter().enumerate() {
            sum += (b as i8) as f32 * x[j] * col[j];
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_col_f32_neon(w: &[u8], x: &[f32], col: &[f32]) -> f32 {
    // SAFETY: callers uphold slice-length contracts (see call sites).
    unsafe {
        use core::arch::aarch64::*;
        let n = x.len();
        let wp = w.as_ptr() as *const i8;
        let xp = x.as_ptr();
        let cp = col.as_ptr();
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
            a0 = vfmaq_f32(a0, w0, vmulq_f32(vld1q_f32(xp.add(j)), vld1q_f32(cp.add(j))));
            a1 = vfmaq_f32(a1, w1, vmulq_f32(vld1q_f32(xp.add(j + 4)), vld1q_f32(cp.add(j + 4))));
            a2 = vfmaq_f32(a2, w2, vmulq_f32(vld1q_f32(xp.add(j + 8)), vld1q_f32(cp.add(j + 8))));
            a3 = vfmaq_f32(a3, w3, vmulq_f32(vld1q_f32(xp.add(j + 12)), vld1q_f32(cp.add(j + 12))));
            j += 16;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        while j < n {
            sum += (*wp.add(j)) as f32 * *xp.add(j) * *cp.add(j);
            j += 1;
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
            Some(pool) if rows >= 256 => pool.run_rows(rows, &run_range),
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
        let offsets = vbit_row_offsets(&bytes, rows, cols);
        vbitmatvec(&bytes, &offsets, &x, rows, cols, &mut got, None);
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
        // SDOT path quantizes activations to i8 (A8W8): bounded noise,
        // same contract as q8/vbit (exact path is pinned by CMF_SDOT=0
        // in the golden-parity gate).
        let tol = if sdot_enabled() { 6e-2 } else { 1e-4 };
        let scale = expect.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
        for r in 0..rows {
            assert!(
                (got[r] - expect[r]).abs() < tol * scale,
                "row {r}: {} vs {}",
                got[r],
                expect[r]
            );
        }
    }

    /// Fused two-input vbit matvec must equal two single matvecs exactly
    /// (same per-lane accumulation order on both scalar and SDOT paths).
    #[test]
    fn vbitmatvec2_equals_two_singles() {
        let (rows, cols) = (6, 64);
        let ng = cols / GROUP_SIZE;
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
        let x1: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.19).sin()).collect();
        let x2: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.11).cos()).collect();
        let offsets = vbit_row_offsets(&bytes, rows, cols);

        let (mut a1, mut a2) = (vec![0f32; rows], vec![0f32; rows]);
        vbitmatvec(&bytes, &offsets, &x1, rows, cols, &mut a1, None);
        vbitmatvec(&bytes, &offsets, &x2, rows, cols, &mut a2, None);
        let (mut b1, mut b2) = (vec![0f32; rows], vec![0f32; rows]);
        vbitmatvec2(&bytes, &offsets, &x1, &x2, rows, cols, &mut b1, &mut b2, None);
        assert_eq!(a1, b1, "fused vbit lane 1 must be bit-identical");
        assert_eq!(a2, b2, "fused vbit lane 2 must be bit-identical");
    }

    /// Fused two-input q4 matvec must equal two single matvecs exactly.
    #[test]
    fn q4matvec2_equals_two_singles() {
        let (rows, cols) = (8, 128);
        let groups = rows * cols / GROUP_SIZE;
        let mut bytes = Vec::with_capacity(groups * 16 + groups * 2);
        for i in 0..groups * 16 {
            bytes.push((((i * 7 + 3) % 256) & 0xFF) as u8);
        }
        for g in 0..groups {
            let s = 0.01 + 0.003 * g as f32;
            bytes.extend_from_slice(&cortiq_core::quant::f32_to_f16(s).to_le_bytes());
        }
        // Include an outlier channel so the SDOT correction path is
        // exercised in the pair kernel too.
        let mut x1: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.17).sin()).collect();
        x1[9] = 250.0;
        let x2: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.23).cos()).collect();

        let (mut a1, mut a2) = (vec![0f32; rows], vec![0f32; rows]);
        q4matvec(&bytes, &x1, rows, cols, &mut a1, None);
        q4matvec(&bytes, &x2, rows, cols, &mut a2, None);
        let (mut b1, mut b2) = (vec![0f32; rows], vec![0f32; rows]);
        q4matvec2(&bytes, &x1, &x2, rows, cols, &mut b1, &mut b2, None);
        assert_eq!(a1, b1, "fused q4 lane 1 must be bit-identical");
        assert_eq!(a2, b2, "fused q4 lane 2 must be bit-identical");
    }

    /// Batched q4/vbit matmat must equal per-position matvec calls
    /// exactly (the fallback it replaced) — same kernels, same order.
    #[test]
    fn batched_matmat_equals_per_position_matvec() {
        let (rows, cols, b) = (8, 64, 5);
        // q4 blob.
        let groups = rows * cols / GROUP_SIZE;
        let mut q4 = Vec::new();
        for i in 0..groups * 16 {
            q4.push((((i * 7 + 3) % 256) & 0xFF) as u8);
        }
        for g in 0..groups {
            q4.extend_from_slice(
                &cortiq_core::quant::f32_to_f16(0.01 + 0.003 * g as f32).to_le_bytes(),
            );
        }
        // vbit blob (mixed widths incl. 8).
        let ng = cols / GROUP_SIZE;
        let bits: Vec<u8> = vec![3, 4, 5, 6, 8, 4, 5, 3];
        let mut vb = bits.clone();
        for g in 0..rows * ng {
            vb.extend_from_slice(
                &cortiq_core::quant::f32_to_f16(0.02 + 0.001 * g as f32).to_le_bytes(),
            );
        }
        for r in 0..rows {
            let bw = bits[r] as usize;
            let (mut acc, mut nb) = (0u64, 0usize);
            let mut rowbytes = Vec::new();
            for i in 0..cols {
                let v = ((i * 7 + r * 13) % (1 << bw)) as u64;
                acc = (acc << bw) | v;
                nb += bw;
                while nb >= 8 {
                    nb -= 8;
                    rowbytes.push(((acc >> nb) & 0xFF) as u8);
                }
            }
            if nb > 0 {
                rowbytes.push(((acc << (8 - nb)) & 0xFF) as u8);
            }
            vb.extend_from_slice(&rowbytes);
        }
        let offsets = vbit_row_offsets(&vb, rows, cols);

        let xs: Vec<f32> = (0..b * cols).map(|i| (i as f32 * 0.13).sin()).collect();

        // q4: batch vs singles.
        let mut got = vec![0f32; b * rows];
        q4matmat(&q4, &xs, b, rows, cols, &mut got, None);
        for bi in 0..b {
            let mut expect = vec![0f32; rows];
            q4matvec(&q4, &xs[bi * cols..(bi + 1) * cols], rows, cols, &mut expect, None);
            assert_eq!(&got[bi * rows..(bi + 1) * rows], &expect[..], "q4 batch pos {bi}");
        }

        // vbit: batch vs singles.
        let mut got = vec![0f32; b * rows];
        vbitmatmat(&vb, &offsets, &xs, b, rows, cols, &mut got, None);
        for bi in 0..b {
            let mut expect = vec![0f32; rows];
            vbitmatvec(
                &vb, &offsets, &xs[bi * cols..(bi + 1) * cols], rows, cols, &mut expect, None,
            );
            assert_eq!(&got[bi * rows..(bi + 1) * rows], &expect[..], "vbit batch pos {bi}");
        }
    }

    /// q4 SDOT outlier correction: a single huge activation channel
    /// (>8·rms → outlier, zeroed in xq) must still contribute its EXACT
    /// term. On-grid bulk (±1/0 → xq dequantizes exactly) isolates the
    /// correction from A8W8 noise. cols must exceed 64: at n=64 the
    /// 8·rms threshold equals sqrt(v²+rest) ≥ v, so a single outlier
    /// can never qualify (8² = n).
    #[test]
    fn q4matvec_sdot_outlier_exact() {
        let (rows, cols) = (4, 128);
        let groups = rows * cols / GROUP_SIZE;
        let mut bytes = Vec::with_capacity(groups * 16 + groups * 2);
        for i in 0..groups * 16 {
            bytes.push(((i * 11 + 5) % 256) as u8);
        }
        for g in 0..groups {
            let s = 0.02 + 0.002 * g as f32;
            bytes.extend_from_slice(&cortiq_core::quant::f32_to_f16(s).to_le_bytes());
        }
        let mut x: Vec<f32> = (0..cols)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        x[17] = 300.0; // ≫ 8·rms → outlier channel

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
        let scale = expect.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
        for r in 0..rows {
            assert!(
                (got[r] - expect[r]).abs() < 2e-3 * scale,
                "row {r}: {} vs {} (outlier term must be exact)",
                got[r],
                expect[r]
            );
        }
    }
}
