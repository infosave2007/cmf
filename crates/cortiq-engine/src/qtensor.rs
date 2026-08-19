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

use crate::pool::{Pool, matvec_rows, matvec_rows2};
use cortiq_core::quant::{
    GROUP_SIZE, Q1_TILE, Q2TP_CHUNK, Q4_TILE, Q4TP_NIB, f16_to_f32, q2tp_ladder, q2tp_sections,
    q4tp_code, q4tp_ladder, q4tp_sections,
};
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
        /// q8-family decode repack (load-time, optional): rows in groups
        /// of 4, interleaved in 16-byte units — one 64-byte line per
        /// iteration feeds all 4 sdot lanes, ONE sequential weight
        /// stream per worker instead of four (this is where llama.cpp's
        /// repacked Q8 kernels get their bandwidth). Empty = off
        /// (CMF_REPACK=0, non-SDOT arch, or an ineligible shape). Trades
        /// an anonymous copy of the quants for mmap pages that go cold.
        repack: Vec<u8>,
    },
}

/// Load-time q8 repack gate (see `Mapped::repack`). OPT-IN
/// (`CMF_REPACK=1`): the single-stream hypothesis LOST on Apple Silicon
/// (M4, interleaved A/B: decode 101 vs 94 tok/s — four adjacent row
/// streams per worker feed the prefetcher MORE memory-level parallelism
/// than one); kept as an experiment flag for x86, where the tradeoff
/// may land differently.
fn repack_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CMF_REPACK")
            .map(|v| v == "1")
            .unwrap_or(cfg!(target_os = "android"))
    })
}

/// Interleave q8 rows for the decode kernel: group g holds rows
/// 4g..4g+4 as [r0[c], r1[c], r2[c], r3[c]] per 16-byte chunk c. Only
/// full groups are packed — tail rows keep reading the mmap layout.
fn q8_repack(bytes: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    #[cfg(target_arch = "aarch64")]
    let arch_ok = sdot_enabled();
    #[cfg(not(target_arch = "aarch64"))]
    let arch_ok = false;
    if !arch_ok || !repack_enabled() || rows < 256 || cols % 16 != 0 {
        return Vec::new();
    }
    q8_repack_layout(bytes, rows, cols)
}

/// The pure layout transform behind `q8_repack` (tested directly —
/// the gate depends on arch and env).
fn q8_repack_layout(bytes: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let groups = rows / 4;
    let mut rep = vec![0u8; groups * 4 * cols];
    for g in 0..groups {
        let dst = &mut rep[g * 4 * cols..(g + 1) * 4 * cols];
        for c in 0..cols / 16 {
            for lane in 0..4 {
                let src = (g * 4 + lane) * cols + c * 16;
                dst[c * 64 + lane * 16..c * 64 + lane * 16 + 16]
                    .copy_from_slice(&bytes[src..src + 16]);
            }
        }
    }
    rep
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
        off += (cols * bits[r] as usize).div_ceil(8);
    }
    offsets.push(off);
    offsets
}

/// `CMF_X86_BLOCKED` / `CMF_GPU_LMHEAD` / `CMF_GPU_SPLIT`, read once. They
/// used to be read from the environment on every large matvec and on every
/// matmat in six places — microseconds each, but also a knob that could
/// change under a running process, which is not a thing a kernel choice
/// should be able to do mid-sequence.
fn blocked_enabled() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match BLOCKED_OVERRIDE.load(Relaxed) {
        1 => false,
        2 => true,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("CMF_X86_BLOCKED")
                    .map(|v| v != "0")
                    .unwrap_or(true)
            })
        }
    }
}

static BLOCKED_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Force the blocked GEMM on or off, ignoring the environment; `None`
/// restores it. For tests that need to run BOTH paths and compare them:
/// `blocked_enabled` caches its answer for the life of the process, which
/// is right when the environment is the only input, but leaves a test that
/// flips `CMF_X86_BLOCKED` between two calls comparing a path against
/// itself — or against whatever a test running in parallel latched first.
pub fn set_blocked_override(on: Option<bool>) {
    let v = match on {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    BLOCKED_OVERRIDE.store(v, std::sync::atomic::Ordering::Relaxed);
}

fn gpu_lmhead_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CMF_GPU_LMHEAD")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

fn gpu_split_frac() -> f32 {
    static F: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("CMF_GPU_SPLIT")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.5)
            .clamp(0.0, 1.0)
    })
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
                    repack: q8_repack(bytes, rows, cols),
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
                repack: Vec::new(),
            }),
            // vbit_ro (§4.2): the offset table comes straight from the
            // file — no load-time prefix scan; kernels are shared with
            // legacy vbit (they consume absolute offsets either way).
            TensorDtype::VbitRo if cols % GROUP_SIZE == 0 => {
                let (_, off_off, packed_off) = cortiq_core::quant::vbit_ro_sections(rows, cols);
                let offsets: Vec<usize> = (0..=rows)
                    .map(|r| packed_off + cortiq_core::quant::vbit_ro_offset(bytes, off_off, r))
                    .collect();
                Ok(Self::Mapped {
                    model: model.clone(),
                    idx,
                    dtype: entry.dtype,
                    rows,
                    cols,
                    row_scale: Vec::new(),
                    col_field: Vec::new(),
                    vbit_offsets: offsets,
                    repack: Vec::new(),
                })
            }
            // q4_block: fused kernel reads nibbles straight from mmap —
            // a 14B q4 file no longer explodes into ×8 f32 RAM.
            // q4_tiled (§4.3): interleaved [scale][nibbles] tiles — one
            // sequential memory stream (measured ×1.66 ARM / ×1.13 AVX2
            // at kernel level over the split layout).
            TensorDtype::Q4Tiled if cols % GROUP_SIZE == 0 => Ok(Self::Mapped {
                model: model.clone(),
                idx,
                dtype: entry.dtype,
                rows,
                cols,
                row_scale: Vec::new(),
                col_field: Vec::new(),
                vbit_offsets: Vec::new(),
                repack: Vec::new(),
            }),
            // q4tp (§4.10): nibbles from mmap, scale from the row ladder —
            // 7.3% less file than q4t at the same 4-bit grid.
            TensorDtype::Q4TiledP if cols % GROUP_SIZE == 0 => Ok(Self::Mapped {
                model: model.clone(),
                idx,
                dtype: entry.dtype,
                rows,
                cols,
                row_scale: Vec::new(),
                col_field: Vec::new(),
                vbit_offsets: Vec::new(),
                repack: Vec::new(),
            }),
            // q2tp: 2-bit chunks from mmap, scale from the same row ladder.
            TensorDtype::Q2TiledP if cols % GROUP_SIZE == 0 => Ok(Self::Mapped {
                model: model.clone(),
                idx,
                dtype: entry.dtype,
                rows,
                cols,
                row_scale: Vec::new(),
                col_field: Vec::new(),
                vbit_offsets: Vec::new(),
                repack: Vec::new(),
            }),
            TensorDtype::Q4Block if cols % GROUP_SIZE == 0 => Ok(Self::Mapped {
                model: model.clone(),
                idx,
                dtype: entry.dtype,
                rows,
                cols,
                row_scale: Vec::new(),
                col_field: Vec::new(),
                vbit_offsets: Vec::new(),
                repack: Vec::new(),
            }),
            // q1: binary sign-bit tiles from mmap (1-bit-trained models).
            TensorDtype::Q1 if cols % GROUP_SIZE == 0 => Ok(Self::Mapped {
                model: model.clone(),
                idx,
                dtype: entry.dtype,
                rows,
                cols,
                row_scale: Vec::new(),
                col_field: Vec::new(),
                vbit_offsets: Vec::new(),
                repack: Vec::new(),
            }),
            // q1t (ternary + outlier overlay): fused per-row dequant kernel
            // reads straight from mmap — a 12B q1t stays ~its file size in
            // RAM instead of dequantizing to ~48 GB of f32.
            TensorDtype::Q1T if cols % GROUP_SIZE == 0 => Ok(Self::Mapped {
                model: model.clone(),
                idx,
                dtype: entry.dtype,
                rows,
                cols,
                row_scale: Vec::new(),
                col_field: Vec::new(),
                vbit_offsets: Vec::new(),
                repack: Vec::new(),
            }),
            // No fused kernel yet → dequantize once (correct, more RAM).
            _ => {
                let mut data = vec![0.0f32; rows * cols];
                cortiq_core::quant::dequant_tensor(entry, bytes, &mut data)?;
                Ok(Self::from_f32(data, rows, cols))
            }
        }
    }

    /// q1-mapped tensor? (GPU gates: the q1 CPU kernel is
    /// compute-bound, so offload pays at much smaller shapes than q8.)
    pub(crate) fn is_q1(&self) -> bool {
        matches!(
            self,
            Self::Mapped {
                dtype: TensorDtype::Q1,
                ..
            }
        )
    }

    /// Owned-f32 view (data, rows, cols) — the GDN a/b gate projections
    /// arrive dequantized (force-f16 in the converter → F32 in RAM).
    pub(crate) fn f32_parts(&self) -> Option<(&[f32], usize, usize)> {
        match self {
            Self::F32 { data, rows, cols } => Some((data, *rows, *cols)),
            _ => None,
        }
    }

    /// (directory idx, rows, cols) of a q1-mapped tensor — the
    /// whole-block GPU path resolves offsets itself.
    /// (idx, rows, cols) of a mapped tensor the whole-token GPU graph can drive
    /// — Q1, Q1T or Q4-block (it resolves the offset and picks the kernel by
    /// dtype). Q4-block lets a precise down_proj/lm_head stay on-device.
    /// Named `q1_parts` for historical reasons.
    pub(crate) fn q1_parts(&self) -> Option<(usize, usize, usize)> {
        match self {
            #[cfg(target_os = "macos")]
            Self::Mapped {
                dtype: TensorDtype::Q1T,
                ..
            } if !crate::gpu::metal_q1t_enabled() => None,
            Self::Mapped {
                idx,
                dtype:
                    TensorDtype::Q1
                    | TensorDtype::Q1T
                    | TensorDtype::Q4Block
                    | TensorDtype::Q4Tiled
                    // Q2TiledP deliberately absent: the Metal graph has no
                    // q2tp kernel, and advertising it here made the block
                    // plan truncate mid-run at the first q2tp layer.
                    | TensorDtype::Q4TiledP
                    | TensorDtype::Q8Row
                    | TensorDtype::Q8_2f,
                rows,
                cols,
                ..
            } => Some((*idx, *rows, *cols)),
            _ => None,
        }
    }

    /// (directory idx, rows, cols) of a q4_tiled mapped tensor. The
    /// chunk-prefill graph takes it in the same 4-tuple slot as
    /// `q8_row_parts` with an EMPTY row_scale — q4t carries its scales
    /// inside the 18-byte tiles, and the empty slice is what tells the
    /// encoder to reach for the q4t kernels.
    pub(crate) fn q4t_parts(&self) -> Option<(usize, usize, usize)> {
        match self {
            Self::Mapped {
                idx,
                dtype: TensorDtype::Q4Tiled,
                rows,
                cols,
                ..
            } => Some((*idx, *rows, *cols)),
            _ => None,
        }
    }

    /// (directory idx, rows, cols) of a q4tp mapped tensor. Same empty-scale
    /// slot as `q4t_parts` in the chunk graph — the encoder tells the two
    /// apart by the tensor's dtype, not by the slot.
    pub(crate) fn q4tp_parts(&self) -> Option<(usize, usize, usize)> {
        match self {
            Self::Mapped {
                idx,
                dtype: TensorDtype::Q4TiledP,
                rows,
                cols,
                ..
            } => Some((*idx, *rows, *cols)),
            _ => None,
        }
    }

    /// (directory idx, rows, cols, row_scale) of a plain q8_row mapped
    /// tensor — the chunk-prefill GPU graph resolves offsets itself.
    /// q8_2f is excluded on purpose: its column field would need a
    /// prescale stage on the device.
    pub(crate) fn q8_row_parts(&self) -> Option<(usize, usize, usize, &[f32])> {
        match self {
            Self::Mapped {
                idx,
                dtype: TensorDtype::Q8Row,
                rows,
                cols,
                row_scale,
                col_field,
                ..
            } if col_field.is_empty() => Some((*idx, *rows, *cols, row_scale)),
            _ => None,
        }
    }

    /// The layout this tensor is stored in, when it is mapped from a model.
    /// The frames branch on it — a q2tp gate against a q4tp down is a real
    /// combination in the 2-bit profile and needs a different kernel.
    pub fn model_dtype(&self) -> Option<cortiq_core::TensorDtype> {
        match self {
            Self::Mapped { dtype, .. } => Some(*dtype),
            _ => None,
        }
    }

    /// The tensor's index in the model directory, when it is mapped from one.
    /// The GPU frames bind by index rather than by name — a name lookup per
    /// layer per token is not free, and the index is what the device cache is
    /// keyed on anyway.
    pub fn model_idx(&self) -> Option<usize> {
        match self {
            Self::Mapped { idx, .. } => Some(*idx),
            _ => None,
        }
    }

    /// The model this tensor is mapped from, when it is mapped at all. The
    /// GPU frames need the container to reach the bytes; a QTensor already
    /// holds it, and threading a second handle down every call site to say
    /// the same thing invites the two to disagree.
    pub fn model_arc(&self) -> Option<std::sync::Arc<cortiq_core::CmfModel>> {
        match self {
            Self::Mapped { model, .. } => Some(model.clone()),
            _ => None,
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::F32 { rows, .. } | Self::Mapped { rows, .. } => *rows,
        }
    }

    /// Mapped q4t handle (model + directory index) — the fused GPU FFN
    /// needs the raw file coordinates of its three projections.
    pub(crate) fn mapped_q4t(&self) -> Option<(&Arc<CmfModel>, usize)> {
        match self {
            Self::Mapped {
                model,
                idx,
                dtype: TensorDtype::Q4Tiled,
                ..
            } => Some((model, *idx)),
            _ => None,
        }
    }

    /// Same slot as `mapped_q4t` for a q4tp tensor — the fused DiT FFN picks
    /// its kernels by which of the two answers.
    pub fn mapped_q4tp(&self) -> Option<(&Arc<CmfModel>, usize)> {
        match self {
            Self::Mapped {
                model,
                idx,
                dtype: TensorDtype::Q4TiledP,
                ..
            } => Some((model, *idx)),
            _ => None,
        }
    }

    /// (model, tensor idx) for a q2tp mapped weight — the 2-bit twin of
    /// `mapped_q4tp`, used by the mixed MoE profile.
    pub fn mapped_q2tp(&self) -> Option<(&Arc<CmfModel>, usize)> {
        match self {
            Self::Mapped {
                model,
                idx,
                dtype: TensorDtype::Q2TiledP,
                ..
            } => Some((model, *idx)),
            _ => None,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::F32 { cols, .. } | Self::Mapped { cols, .. } => *cols,
        }
    }

    /// (model, tensor idx) for a q1 mapped weight — the wgpu token graph
    /// keys its resident VRAM cache by idx. None for any other dtype/kind.
    pub fn mapped_q1(&self) -> Option<(&std::sync::Arc<CmfModel>, usize)> {
        match self {
            Self::Mapped {
                model,
                idx,
                dtype: TensorDtype::Q1,
                ..
            } => Some((model, *idx)),
            _ => None,
        }
    }

    /// (model, idx, kind, row_scale) for a graph-capable mapped weight.
    /// kind: 0=q8_row (per-row scales), 1=q1, 2=q4_block, 3=q1t
    /// (tile-embedded, no rs), 5=q4_tiled, 6=q4tp, 7=q8_2f (both scale
    /// planes live inside the tensor). None only for `vbit`.
    ///
    /// The old comment here claimed q4_block was unhandled while the arm
    /// right below mapped it, and it named q8_2f as unhandled after that
    /// stopped being true — a stale comment on this function is how a
    /// model silently loses the graph, so it is worth keeping honest.
    pub fn graph_weight(&self) -> Option<(&std::sync::Arc<CmfModel>, usize, u8, &[f32])> {
        match self {
            Self::Mapped {
                model,
                idx,
                dtype: TensorDtype::Q8Row,
                row_scale,
                ..
            } => Some((model, *idx, 0, row_scale.as_slice())),
            Self::Mapped {
                model,
                idx,
                dtype: TensorDtype::Q1,
                ..
            } => Some((model, *idx, 1, &[])),
            // Q4Tiled is kind 5, NOT 2: both carried 2 historically, and
            // the wgpu token graph fed 18B interleaved tiles to the
            // split-layout q4b kernel — garbage output on q4t models
            // (caught by an end-to-end answer check on real Vulkan).
            Self::Mapped {
                model,
                idx,
                dtype: TensorDtype::Q4Tiled,
                ..
            } => Some((model, *idx, 5, &[])),
            // Kind 6, not 5: q4tp's nibble stride and scale planes differ,
            // and feeding them to the q4t kernel is exactly the mistake that
            // produced garbage when Q4Tiled shared kind 2 with Q4Block.
            Self::Mapped {
                model,
                idx,
                dtype: TensorDtype::Q4TiledP,
                ..
            } => Some((model, *idx, 6, &[])),
            Self::Mapped {
                model,
                idx,
                dtype: TensorDtype::Q4Block,
                ..
            } => Some((model, *idx, 2, &[])),
            // q8_2f carries BOTH scale planes after the int8 body (rows
            // f16, then cols f16), so the graph takes the whole tensor
            // and the kernel reads them where they lie — no host-side
            // prescale, which is what the per-op path does instead.
            Self::Mapped {
                model,
                idx,
                dtype: TensorDtype::Q8_2f,
                ..
            } => Some((model, *idx, 7, &[])),
            Self::Mapped {
                model,
                idx,
                dtype: TensorDtype::Q1T,
                ..
            } => Some((model, *idx, 3, &[])),
            // Kind 9: the 2-bit plane on the q4tp ladder (dense FFN gate/up
            // of the q2tp profile). Its own kernel — 8 bytes a group where
            // q4tp has 16, and rung 0 is the exact zero.
            Self::Mapped {
                model,
                idx,
                dtype: TensorDtype::Q2TiledP,
                ..
            } => Some((model, *idx, 9, &[])),
            _ => None,
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
                if *dtype == TensorDtype::Q4Tiled {
                    let bytes = self.quant_bytes();
                    let gpr = cols / GROUP_SIZE;
                    for gi in 0..gpr {
                        let tile = &bytes[(r * gpr + gi) * Q4_TILE..(r * gpr + gi + 1) * Q4_TILE];
                        let s = f16_to_f32(u16::from_le_bytes([tile[0], tile[1]]));
                        for (k, &b) in tile[2..].iter().enumerate() {
                            dst[gi * GROUP_SIZE + k * 2] = ((b & 0x0F) as f32 - 8.0) * s;
                            dst[gi * GROUP_SIZE + k * 2 + 1] = (((b >> 4) & 0x0F) as f32 - 8.0) * s;
                        }
                    }
                    return;
                }
                if *dtype == TensorDtype::Q4TiledP {
                    let bytes = self.quant_bytes();
                    let gpr = cols / GROUP_SIZE;
                    let v = Q4tpView::new(bytes, self.rows(), cols);
                    let mut sc = vec![0f32; gpr];
                    v.scales_into(r, gpr, &mut sc);
                    for gi in 0..gpr {
                        let tile = &v.nib[(r * gpr + gi) * Q4TP_NIB..(r * gpr + gi + 1) * Q4TP_NIB];
                        let s = sc[gi];
                        for (k, &b) in tile.iter().enumerate() {
                            dst[gi * GROUP_SIZE + k * 2] = ((b & 0x0F) as f32 - 8.0) * s;
                            dst[gi * GROUP_SIZE + k * 2 + 1] = (((b >> 4) & 0x0F) as f32 - 8.0) * s;
                        }
                    }
                    return;
                }
                if *dtype == TensorDtype::Q2TiledP {
                    let bytes = self.quant_bytes();
                    let gpr = cols / GROUP_SIZE;
                    let v = Q4tpView::new_q2(bytes, self.rows(), cols);
                    let mut sc = vec![0f32; gpr];
                    v.scales_into(r, gpr, &mut sc);
                    for gi in 0..gpr {
                        let ch =
                            &v.nib[(r * gpr + gi) * Q2TP_CHUNK..(r * gpr + gi + 1) * Q2TP_CHUNK];
                        let s = sc[gi];
                        for (k, &b) in ch.iter().enumerate() {
                            for j in 0..4 {
                                dst[gi * GROUP_SIZE + k * 4 + j] =
                                    (((b >> (2 * j)) & 3) as f32 - 1.5) * s;
                            }
                        }
                    }
                    return;
                }
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
                if *dtype == TensorDtype::Q1 {
                    let bytes = self.quant_bytes();
                    let gpr = cols / GROUP_SIZE;
                    for gi in 0..gpr {
                        let tile = &bytes[(r * gpr + gi) * Q1_TILE..(r * gpr + gi + 1) * Q1_TILE];
                        let s = f16_to_f32(u16::from_le_bytes([tile[0], tile[1]]));
                        for (j, &b) in tile[2..].iter().enumerate() {
                            for k in 0..8 {
                                dst[gi * GROUP_SIZE + j * 8 + k] =
                                    (((b >> k) & 1) as f32 * 2.0 - 1.0) * s;
                            }
                        }
                    }
                    return;
                }
                if *dtype == TensorDtype::Q1T {
                    let bytes = self.quant_bytes();
                    let gpr = cols / GROUP_SIZE;
                    let base_len = self.rows() * gpr * cortiq_core::quant::Q1T_TILE;
                    for gi in 0..gpr {
                        let off = (r * gpr + gi) * cortiq_core::quant::Q1T_TILE;
                        let s = cortiq_core::quant::f16_to_f32(u16::from_le_bytes([
                            bytes[off],
                            bytes[off + 1],
                        ]));
                        let codes = &bytes[off + 2..off + cortiq_core::quant::Q1T_TILE];
                        for k in 0..GROUP_SIZE {
                            dst[gi * GROUP_SIZE + k] = match cortiq_core::quant::q1t_code(codes, k)
                            {
                                1 => s,
                                2 => -s,
                                _ => 0.0,
                            };
                        }
                    }
                    // Overlay
                    let rows = self.rows();
                    let entries = base_len + (rows + 1) * 4;
                    if entries <= bytes.len() {
                        let ptrs = &bytes[base_len..base_len + (rows + 1) * 4];
                        let r0 = u32::from_le_bytes([
                            ptrs[r * 4],
                            ptrs[r * 4 + 1],
                            ptrs[r * 4 + 2],
                            ptrs[r * 4 + 3],
                        ]) as usize;
                        let r1 = u32::from_le_bytes([
                            ptrs[(r + 1) * 4],
                            ptrs[(r + 1) * 4 + 1],
                            ptrs[(r + 1) * 4 + 2],
                            ptrs[(r + 1) * 4 + 3],
                        ]) as usize;
                        let off = entries + r0 * 4;
                        for i in 0..r1 - r0 {
                            let item = &bytes[off + i * 4..off + i * 4 + 4];
                            let c = u16::from_le_bytes([item[0], item[1]]) as usize;
                            let v = cortiq_core::quant::f16_to_f32(u16::from_le_bytes([
                                item[2], item[3],
                            ]));
                            if c < cols {
                                dst[c] = v;
                            }
                        }
                    }
                    return;
                }
                if matches!(dtype, TensorDtype::Vbit | TensorDtype::VbitRo) {
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
                dtype,
                row_scale,
                col_field,
                ..
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
            Self::Mapped {
                dtype,
                row_scale,
                col_field,
                ..
            } => match dtype {
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
            // NOTE: `out.len()` DRIVES this arm — it computes that many rows,
            // and `x.len()` is the stride. A short `out` is legitimate here,
            // which is why the check below lives in the Mapped arm only.
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
                repack,
            } => {
                let _ = (model, idx);
                // Every kernel below writes `rows` entries through a raw
                // pointer, so a short `out` is an out-of-bounds WRITE, not a
                // wrong answer: it scribbles on the allocator's metadata and
                // the process aborts much later, somewhere innocent
                // (`double free or corruption`, `corrupted double-linked
                // list`). The debug_assert two of the kernels carried is
                // compiled out of the release — exactly the build where it
                // matters. Fail here instead, while the caller is still on
                // the stack to be named.
                assert!(
                    out.len() >= *rows && x.len() >= *cols,
                    "matvec {rows}x{cols}: out {} (need {rows}), x {} (need {cols})",
                    out.len(),
                    x.len(),
                );
                if *dtype == TensorDtype::Q4Block {
                    // GPU route (wgpu q4b kernel) for large q4_block matvecs —
                    // gives NVIDIA/AMD/Intel q4 models a GPU path. Probe keeps
                    // the winner; Metal returns false → the CPU kernel below.
                    if *rows * *cols >= 8_388_608 && crate::gpu::enabled_here() {
                        let t0 = std::time::Instant::now();
                        match crate::gpu::probe_arm(crate::gpu::OpClass::Matvec) {
                            crate::gpu::ProbeArm::Gpu => {
                                if crate::gpu::q4b_matvec(model, *idx, x, *rows, *cols, out) {
                                    crate::gpu::probe_record(
                                        crate::gpu::OpClass::Matvec,
                                        true,
                                        t0.elapsed(),
                                    );
                                    return;
                                }
                            }
                            crate::gpu::ProbeArm::CpuTimed => {
                                q4matvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                                crate::gpu::probe_record(
                                    crate::gpu::OpClass::Matvec,
                                    false,
                                    t0.elapsed(),
                                );
                                return;
                            }
                            crate::gpu::ProbeArm::Cpu => {}
                        }
                    }
                    q4matvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                    return;
                }
                if *dtype == TensorDtype::Q4Tiled {
                    // GPU route for large q4t matvecs — the lm_head class,
                    // same shape as the q4tp arm below. The probe keeps the
                    // winner; a backend without the kernel refuses and the
                    // CPU path stays.
                    if *rows * *cols >= 8_388_608 && crate::gpu::enabled_here() {
                        let t0 = std::time::Instant::now();
                        let cls = crate::gpu::matvec_class(*rows, *cols);
                        match crate::gpu::probe_arm(cls) {
                            crate::gpu::ProbeArm::Gpu => {
                                if crate::gpu::q4t_matvec(model, *idx, x, *rows, *cols, out) {
                                    crate::gpu::probe_record(cls, true, t0.elapsed());
                                    return;
                                }
                            }
                            crate::gpu::ProbeArm::CpuTimed => {
                                q4t_matvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                                crate::gpu::probe_record(cls, false, t0.elapsed());
                                return;
                            }
                            crate::gpu::ProbeArm::Cpu => {}
                        }
                    }
                    q4t_matvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                    return;
                }
                if *dtype == TensorDtype::Q4TiledP {
                    // GPU route for large q4tp matvecs — the lm_head class.
                    // On a q4tp checkpoint the head is the biggest single
                    // host matvec left in the decode step, and the batched
                    // kernel at b=1 already exists on both backends. Probe
                    // keeps the winner, same as q4_block above.
                    if *rows * *cols >= 8_388_608 && crate::gpu::enabled_here() {
                        let t0 = std::time::Instant::now();
                        let cls = crate::gpu::matvec_class(*rows, *cols);
                        match crate::gpu::probe_arm(cls) {
                            crate::gpu::ProbeArm::Gpu => {
                                if crate::gpu::q4tp_matvec(model, *idx, x, *rows, *cols, out) {
                                    crate::gpu::probe_record(cls, true, t0.elapsed());
                                    return;
                                }
                            }
                            crate::gpu::ProbeArm::CpuTimed => {
                                q4tp_matvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                                crate::gpu::probe_record(cls, false, t0.elapsed());
                                return;
                            }
                            crate::gpu::ProbeArm::Cpu => {}
                        }
                    }
                    q4tp_matvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                    return;
                }
                if *dtype == TensorDtype::Q2TiledP {
                    q2tp_matvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                    return;
                }
                if *dtype == TensorDtype::Q1 {
                    // GPU route for large q1 matvecs (out_proj / lm_head
                    // class): the CPU q1 kernel is load-port-bound at
                    // ~4 GB/s/core, the GPU one is bandwidth-bound — the
                    // probe measures both arms and keeps the winner.
                    if *rows * *cols >= 8_388_608 && crate::gpu::enabled_here() {
                        let t0 = std::time::Instant::now();
                        let arm = if crate::gpu::q1_force() {
                            crate::gpu::ProbeArm::Gpu
                        } else {
                            crate::gpu::probe_arm(crate::gpu::OpClass::Matvec)
                        };
                        match arm {
                            crate::gpu::ProbeArm::Gpu => {
                                if crate::gpu::q1_matvec(model, *idx, x, *rows, *cols, out) {
                                    crate::gpu::probe_record(
                                        crate::gpu::OpClass::Matvec,
                                        true,
                                        t0.elapsed(),
                                    );
                                    return;
                                }
                            }
                            crate::gpu::ProbeArm::CpuTimed => {
                                q1_matvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                                crate::gpu::probe_record(
                                    crate::gpu::OpClass::Matvec,
                                    false,
                                    t0.elapsed(),
                                );
                                return;
                            }
                            crate::gpu::ProbeArm::Cpu => {}
                        }
                    }
                    q1_matvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                    return;
                }
                if *dtype == TensorDtype::Q1T {
                    // GPU route for large q1t matvecs: the ternary BASE dot runs
                    // on the GPU (load-port-bound on CPU, like q1), then the
                    // sparse overlay is added on the CPU. Probe keeps the winner.
                    if *rows * *cols >= 8_388_608 && crate::gpu::enabled_here() {
                        let t0 = std::time::Instant::now();
                        match crate::gpu::probe_arm(crate::gpu::OpClass::Matvec) {
                            crate::gpu::ProbeArm::Gpu => {
                                if crate::gpu::q1t_matvec(model, *idx, x, *rows, *cols, out) {
                                    q1t_add_overlay(self.quant_bytes(), x, *rows, *cols, out, pool);
                                    crate::gpu::probe_record(
                                        crate::gpu::OpClass::Matvec,
                                        true,
                                        t0.elapsed(),
                                    );
                                    return;
                                }
                            }
                            crate::gpu::ProbeArm::CpuTimed => {
                                q1t_matvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                                crate::gpu::probe_record(
                                    crate::gpu::OpClass::Matvec,
                                    false,
                                    t0.elapsed(),
                                );
                                return;
                            }
                            crate::gpu::ProbeArm::Cpu => {}
                        }
                    }
                    q1t_matvec(self.quant_bytes(), x, *rows, *cols, out, pool);
                    return;
                }
                if matches!(dtype, TensorDtype::Vbit | TensorDtype::VbitRo) {
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
                    && gpu_lmhead_enabled()
                    && crate::gpu::enabled_here()
                {
                    // Runtime probe: alternate the hybrid against the
                    // pure-CPU matvec, keep whichever is faster HERE.
                    let t0 = std::time::Instant::now();
                    match crate::gpu::probe_arm(crate::gpu::OpClass::Matvec) {
                        crate::gpu::ProbeArm::Gpu => {}
                        crate::gpu::ProbeArm::CpuTimed => {
                            qmatvec(
                                self.quant_bytes(),
                                repack,
                                row_scale,
                                x,
                                col_field,
                                *dtype,
                                *rows,
                                *cols,
                                out,
                                pool,
                            );
                            crate::gpu::probe_record(
                                crate::gpu::OpClass::Matvec,
                                false,
                                t0.elapsed(),
                            );
                            return;
                        }
                        crate::gpu::ProbeArm::Cpu => {
                            qmatvec(
                                self.quant_bytes(),
                                repack,
                                row_scale,
                                x,
                                col_field,
                                *dtype,
                                *rows,
                                *cols,
                                out,
                                pool,
                            );
                            return;
                        }
                    }
                    let frac = gpu_split_frac();
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
                            // Repack prefix covers the full groups of the
                            // CPU half (the split starts at row 0).
                            let rep_cpu = if repack.is_empty() {
                                &[][..]
                            } else {
                                &repack[..(cpu_rows / 4) * 4 * *cols]
                            };
                            qmatvec(
                                &bytes[..cpu_rows * *cols],
                                rep_cpu,
                                &row_scale[..cpu_rows],
                                x,
                                col_field,
                                *dtype,
                                cpu_rows,
                                *cols,
                                out_cpu,
                                pool,
                            );
                        }
                        g.join().unwrap_or(false)
                    });
                    if ok {
                        crate::gpu::probe_record(crate::gpu::OpClass::Matvec, true, t0.elapsed());
                        return;
                    }
                    // GPU failed — CPU finishes its half (rows rebased —
                    // group offsets don't line up, mmap layout only).
                    qmatvec(
                        &bytes[cpu_rows * *cols..(*rows) * *cols],
                        &[],
                        &row_scale[cpu_rows..],
                        x,
                        col_field,
                        *dtype,
                        *rows - cpu_rows,
                        *cols,
                        out_gpu,
                        pool,
                    );
                    return;
                }
                qmatvec(
                    self.quant_bytes(),
                    repack,
                    row_scale,
                    x,
                    col_field,
                    *dtype,
                    *rows,
                    *cols,
                    out,
                    pool,
                );
            }
        }
    }

    /// Fused two-input matvec (MTP verify pair): weights streamed once.
    pub fn matvec2(
        &self,
        x1: &[f32],
        x2: &[f32],
        o1: &mut [f32],
        o2: &mut [f32],
        pool: Option<&Pool>,
    ) {
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
                if *dtype == TensorDtype::Q4Tiled {
                    q4t_matvec2(self.quant_bytes(), x1, x2, *rows, *cols, o1, o2, pool);
                    return;
                }
                if *dtype == TensorDtype::Q4TiledP {
                    q4tp_matvec2(self.quant_bytes(), x1, x2, *rows, *cols, o1, o2, pool);
                    return;
                }
                if *dtype == TensorDtype::Q2TiledP {
                    q2tp_matvec2(self.quant_bytes(), x1, x2, *rows, *cols, o1, o2, pool);
                    return;
                }
                if *dtype == TensorDtype::Q1 {
                    q1_matvec2(self.quant_bytes(), x1, x2, *rows, *cols, o1, o2, pool);
                    return;
                }
                if *dtype == TensorDtype::Q1T {
                    // Fused ternary pair: one row pass, the register
                    // unpack shared across both streams on ARM. (Q1T
                    // lacks a row_scale array — scales live inline in
                    // the tiles — so it must not fall through to the
                    // q8 qmatvec2 below.)
                    q1t_matvec2(self.quant_bytes(), x1, x2, *rows, *cols, o1, o2, pool);
                    return;
                }
                if matches!(dtype, TensorDtype::Vbit | TensorDtype::VbitRo) {
                    vbitmatvec2(
                        self.quant_bytes(),
                        vbit_offsets,
                        x1,
                        x2,
                        *rows,
                        *cols,
                        o1,
                        o2,
                        pool,
                    );
                    return;
                }
                qmatvec2(
                    self.quant_bytes(),
                    row_scale,
                    x1,
                    x2,
                    col_field,
                    *dtype,
                    *rows,
                    *cols,
                    o1,
                    o2,
                    pool,
                );
            }
        }
    }
}

impl QTensor {
    /// Batched matvec (prefill-GEMM): xs — row-major [b, cols],
    /// out — row-major [b, rows]. Element-wise semantics are IDENTICAL
    /// to b matvec calls (same dot kernels in the same order); the win —
    /// the weight row streams from DRAM once per batch, not b times.
    /// `(model, index)` when this is a memory-mapped q4tp tensor — the
    /// identity a device-resident chain needs to hand `tp_matmat` the
    /// weight without going through this struct's own dispatch.
    pub fn q4tp_mapped(&self) -> Option<(&std::sync::Arc<CmfModel>, usize)> {
        match self {
            Self::Mapped {
                model, idx, dtype, ..
            } if *dtype == TensorDtype::Q4TiledP => Some((model, *idx)),
            _ => None,
        }
    }

    pub fn matmat(&self, xs_all: &[f32], b: usize, out: &mut [f32], pool: Option<&Pool>) {
        let cols = self.cols();
        let rows = self.rows();
        debug_assert_eq!(xs_all.len(), b * cols);
        debug_assert_eq!(out.len(), b * rows);
        // GPTQ calibration: fold this layer's inputs into its Hessian. Only
        // Mapped tensors carry a directory name; the check is a relaxed
        // atomic load, free when not calibrating.
        if crate::gptq_capture::capturing() {
            if let Self::Mapped { model, idx, .. } = self {
                crate::gptq_capture::accumulate(&model.tensors[*idx].name, xs_all, b, cols);
            }
        }
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
                if *dtype == TensorDtype::Q4TiledP {
                    // GPU batched q4tp GEMM (dequant + f32nt mul_mm on the
                    // device); the probe keeps whichever beats the CPU arm.
                    // Narrow (prompt-encode) and wide (DiT) batches probe
                    // as separate classes — the regimes have opposite
                    // winners and one shared verdict locked the wrong arm.
                    // Kill switch (gpu::mm_kill): one grossly slow GPU op
                    // (a fair-condition op is ≤~100 ms even at 1024px)
                    // means the device is contended by another process
                    // (e.g. a simulator) — verdicts are per-process, so
                    // without the bail the whole render crawls behind
                    // someone else's queue.
                    if b >= 32
                        && b * rows * cols >= 128_000_000
                        && cols % 32 == 0
                        && !crate::gpu::mm_killed()
                        && crate::gpu::enabled_here()
                    {
                        let class = if b >= 128 {
                            crate::gpu::OpClass::MatmatWide
                        } else {
                            crate::gpu::OpClass::Matmat
                        };
                        if let Self::Mapped { model, idx, .. } = self {
                            // In-process A/B (`CMF_MM_AB=1`). Three
                            // wall-clock A/Bs on a shared stand disagreed
                            // with each other by 25% on the same change,
                            // because the machine drifts between processes
                            // and interleaving whole renders does not fix
                            // that. Here both arms run back to back on the
                            // SAME data inside one call, so whatever the
                            // machine is doing, it does to both — and the
                            // disagreement between their outputs falls out
                            // for free. Doubles the work; a diagnostic,
                            // not a mode.
                            if crate::mm_ab::on() {
                                let mut g = vec![0f32; b * rows];
                                let t = std::time::Instant::now();
                                let took = crate::gpu::q4tp_matmat(
                                    model, *idx, xs_all, b, rows, cols, &mut g,
                                );
                                let dg = t.elapsed();
                                let t = std::time::Instant::now();
                                q4tp_matmat(self.quant_bytes(), xs_all, b, rows, cols, out, pool);
                                let dc = t.elapsed();
                                crate::mm_ab::record(b, rows, cols, took, dg, dc, &g, out);
                                return;
                            }
                            let t0 = std::time::Instant::now();
                            // A cold call takes the device arm: its sample
                            // is discarded either way, and the upload is
                            // what the next step needs.
                            let resident = crate::gpu::weight_is_resident(model, *idx);
                            match crate::gpu::probe_arm_cold_prefers_gpu(class, resident) {
                                crate::gpu::ProbeArm::Gpu => {
                                    if crate::gpu::q4tp_matmat(
                                        model, *idx, xs_all, b, rows, cols, out,
                                    ) {
                                        let el = t0.elapsed();
                                        // Work-proportional budget: ~8× the
                                        // fair-device estimate (+20 ms slack).
                                        // An absolute cap missed the worst
                                        // case — contended ops sit at
                                        // 100–240 ms each and still bury a
                                        // render whose fair op is 3–9 ms.
                                        // Cold ops (first PSO build, buffer
                                        // alloc) are exempt: a one-off
                                        // ~50 ms compile is not contention.
                                        let flops = 2.0 * b as f64 * rows as f64 * cols as f64;
                                        let budget = std::time::Duration::from_secs_f64(
                                            flops / 1.5e12 * 8.0 + 0.020,
                                        );
                                        crate::gpu::mm_budget_check(
                                            "q4tp matmat",
                                            el,
                                            budget,
                                            crate::gpu::probe_was_cold() || !resident,
                                        );
                                        crate::gpu::probe_record(class, true, el);
                                        return;
                                    }
                                }
                                crate::gpu::ProbeArm::CpuTimed => {
                                    q4tp_matmat(
                                        self.quant_bytes(),
                                        xs_all,
                                        b,
                                        rows,
                                        cols,
                                        out,
                                        pool,
                                    );
                                    crate::gpu::probe_record(class, false, t0.elapsed());
                                    return;
                                }
                                crate::gpu::ProbeArm::Cpu => {}
                            }
                        }
                    }
                    q4tp_matmat(self.quant_bytes(), xs_all, b, rows, cols, out, pool);
                    return;
                }
                if *dtype == TensorDtype::Q2TiledP {
                    // Same device arm as q4tp, behind the same probe:
                    // the planes differ, the dispatch does not. Without
                    // this a q2tp file ran its widest projections on the
                    // host while the 4-bit one had the card, which is a
                    // codec paying for its size twice.
                    if b >= 32
                        && b * rows * cols >= 128_000_000
                        && cols % 32 == 0
                        && !crate::gpu::mm_killed()
                        && crate::gpu::enabled_here()
                    {
                        let class = if b >= 128 {
                            crate::gpu::OpClass::MatmatWide
                        } else {
                            crate::gpu::OpClass::Matmat
                        };
                        if let Self::Mapped { model, idx, .. } = self {
                            let t0 = std::time::Instant::now();
                            match crate::gpu::probe_arm(class) {
                                crate::gpu::ProbeArm::Gpu => {
                                    if crate::gpu::q2tp_matmat(
                                        model, *idx, xs_all, b, rows, cols, out,
                                    ) {
                                        crate::gpu::probe_record(class, true, t0.elapsed());
                                        return;
                                    }
                                }
                                crate::gpu::ProbeArm::CpuTimed => {
                                    q2tp_matmat(
                                        self.quant_bytes(),
                                        xs_all,
                                        b,
                                        rows,
                                        cols,
                                        out,
                                        pool,
                                    );
                                    crate::gpu::probe_record(class, false, t0.elapsed());
                                    return;
                                }
                                crate::gpu::ProbeArm::Cpu => {}
                            }
                        }
                    }
                    // Without a host arm a q2tp tensor falls through to
                    // the q8 fallback, which reads it at one BYTE per
                    // weight — a 2x overrun that killed pool workers
                    // mid-prefill while the dispatcher waited forever.
                    q2tp_matmat(self.quant_bytes(), xs_all, b, rows, cols, out, pool);
                    return;
                }
                if *dtype == TensorDtype::Q4Tiled {
                    // GPU batched q4t GEMM (dequant + f32nt mul_mm on the
                    // device); the probe keeps whichever beats the CPU arm.
                    // Narrow (prompt-encode) and wide (DiT) batches probe
                    // as separate classes — the regimes have opposite
                    // winners and one shared verdict locked the wrong arm.
                    // Kill switch (gpu::mm_kill): one grossly slow GPU op
                    // (a fair-condition op is ≤~100 ms even at 1024px)
                    // means the device is contended by another process
                    // (e.g. a simulator) — verdicts are per-process, so
                    // without the bail the whole render crawls behind
                    // someone else's queue.
                    if b >= 32
                        && b * rows * cols >= 128_000_000
                        && cols % 32 == 0
                        && !crate::gpu::mm_killed()
                        && crate::gpu::enabled_here()
                    {
                        let class = if b >= 128 {
                            crate::gpu::OpClass::MatmatWide
                        } else {
                            crate::gpu::OpClass::Matmat
                        };
                        if let Self::Mapped { model, idx, .. } = self {
                            let t0 = std::time::Instant::now();
                            match crate::gpu::probe_arm(class) {
                                crate::gpu::ProbeArm::Gpu => {
                                    if crate::gpu::q4t_matmat(
                                        model, *idx, xs_all, b, rows, cols, out,
                                    ) {
                                        let el = t0.elapsed();
                                        // Work-proportional budget: ~8× the
                                        // fair-device estimate (+20 ms slack).
                                        // An absolute cap missed the worst
                                        // case — contended ops sit at
                                        // 100–240 ms each and still bury a
                                        // render whose fair op is 3–9 ms.
                                        // Cold ops (first PSO build, buffer
                                        // alloc) are exempt: a one-off
                                        // ~50 ms compile is not contention.
                                        let flops = 2.0 * b as f64 * rows as f64 * cols as f64;
                                        let budget = std::time::Duration::from_secs_f64(
                                            flops / 1.5e12 * 8.0 + 0.020,
                                        );
                                        crate::gpu::mm_budget_check(
                                            "q4t matmat",
                                            el,
                                            budget,
                                            crate::gpu::probe_was_cold(),
                                        );
                                        crate::gpu::probe_record(class, true, el);
                                        return;
                                    }
                                }
                                crate::gpu::ProbeArm::CpuTimed => {
                                    q4t_matmat(
                                        self.quant_bytes(),
                                        xs_all,
                                        b,
                                        rows,
                                        cols,
                                        out,
                                        pool,
                                    );
                                    crate::gpu::probe_record(class, false, t0.elapsed());
                                    return;
                                }
                                crate::gpu::ProbeArm::Cpu => {}
                            }
                        }
                    }
                    q4t_matmat(self.quant_bytes(), xs_all, b, rows, cols, out, pool);
                    return;
                }
                if *dtype == TensorDtype::Q1 {
                    // GPU batched q1 GEMM for wide prefill (q1_mul_mm on the
                    // device); the probe keeps whichever beats the CPU matmat.
                    if b >= 32
                        && b * rows * cols >= 128_000_000
                        && cols % 64 == 0
                        && crate::gpu::enabled_here()
                    {
                        if let Self::Mapped { model, idx, .. } = self {
                            let t0 = std::time::Instant::now();
                            match crate::gpu::probe_arm(crate::gpu::OpClass::Matmat) {
                                crate::gpu::ProbeArm::Gpu => {
                                    if crate::gpu::q1_matmat(
                                        model, *idx, xs_all, b, rows, cols, out,
                                    ) {
                                        crate::gpu::probe_record(
                                            crate::gpu::OpClass::Matmat,
                                            true,
                                            t0.elapsed(),
                                        );
                                        return;
                                    }
                                }
                                crate::gpu::ProbeArm::CpuTimed => {
                                    q1_matmat(self.quant_bytes(), xs_all, b, rows, cols, out, pool);
                                    crate::gpu::probe_record(
                                        crate::gpu::OpClass::Matmat,
                                        false,
                                        t0.elapsed(),
                                    );
                                    return;
                                }
                                crate::gpu::ProbeArm::Cpu => {}
                            }
                        }
                    }
                    q1_matmat(self.quant_bytes(), xs_all, b, rows, cols, out, pool);
                    return;
                }
                if *dtype == TensorDtype::Q1T {
                    // GPU batched GEMM for wide prefill (base + overlay on the
                    // device); probe keeps the winner vs the CPU matmat.
                    if b >= 32 && b * rows * cols >= 128_000_000 && crate::gpu::enabled_here() {
                        if let Self::Mapped { model, idx, .. } = self {
                            let t0 = std::time::Instant::now();
                            match crate::gpu::probe_arm(crate::gpu::OpClass::Matmat) {
                                crate::gpu::ProbeArm::Gpu => {
                                    if crate::gpu::q1t_matmat(
                                        model, *idx, xs_all, b, rows, cols, out,
                                    ) {
                                        crate::gpu::probe_record(
                                            crate::gpu::OpClass::Matmat,
                                            true,
                                            t0.elapsed(),
                                        );
                                        return;
                                    }
                                }
                                crate::gpu::ProbeArm::CpuTimed => {
                                    q1t_matmat(
                                        self.quant_bytes(),
                                        xs_all,
                                        b,
                                        rows,
                                        cols,
                                        out,
                                        pool,
                                    );
                                    crate::gpu::probe_record(
                                        crate::gpu::OpClass::Matmat,
                                        false,
                                        t0.elapsed(),
                                    );
                                    return;
                                }
                                crate::gpu::ProbeArm::Cpu => {}
                            }
                        }
                    }
                    q1t_matmat(self.quant_bytes(), xs_all, b, rows, cols, out, pool);
                    return;
                }
                if matches!(dtype, TensorDtype::Vbit | TensorDtype::VbitRo) {
                    vbitmatmat(
                        self.quant_bytes(),
                        vbit_offsets,
                        xs_all,
                        b,
                        rows,
                        cols,
                        out,
                        pool,
                    );
                    return;
                }
                let pre: Vec<std::borrow::Cow<'_, [f32]>> = (0..b)
                    .map(|bi| prescale(&xs_all[bi * cols..(bi + 1) * cols], col_field, *dtype))
                    .collect();
                // D5: large prefill-batch GEMMs — on the GPU (threshold by
                // work volume: submission carries b×rows×cols MACs).
                // Runtime probe: the naive GEMM shader + sync readback
                // lose to the CPU GEMM on slow driver stacks — alternate
                // both arms and keep the winner.
                if b >= 8 && b * rows * cols >= 128_000_000 && crate::gpu::enabled_here() {
                    if let Self::Mapped { model, idx, .. } = self {
                        let t0 = std::time::Instant::now();
                        match crate::gpu::probe_arm(crate::gpu::OpClass::Matmat) {
                            crate::gpu::ProbeArm::Gpu
                                if crate::gpu::probe_deciding(crate::gpu::OpClass::Matmat)
                                    && !crate::gpu::q8_resident_or_upload(model, *idx) =>
                            {
                                // Cold weights during probing: the upload
                                // has started, the count runs on the CPU —
                                // the GPU arm samples on the next touch.
                                let q = self.quant_bytes();
                                qmatmat(q, row_scale, &pre, rows, cols, out, pool);
                                return;
                            }
                            crate::gpu::ProbeArm::Gpu => {
                                let flat: Vec<f32> =
                                    pre.iter().flat_map(|v| v.iter().copied()).collect();
                                if crate::gpu::q8_matmat(
                                    model, *idx, row_scale, &flat, b, rows, cols, out,
                                ) {
                                    crate::gpu::probe_record(
                                        crate::gpu::OpClass::Matmat,
                                        true,
                                        t0.elapsed(),
                                    );
                                    return;
                                }
                            }
                            crate::gpu::ProbeArm::CpuTimed => {
                                let q = self.quant_bytes();
                                qmatmat(q, row_scale, &pre, rows, cols, out, pool);
                                crate::gpu::probe_record(
                                    crate::gpu::OpClass::Matmat,
                                    false,
                                    t0.elapsed(),
                                );
                                return;
                            }
                            crate::gpu::ProbeArm::Cpu => {}
                        }
                    }
                }
                let q = self.quant_bytes();
                qmatmat(q, row_scale, &pre, rows, cols, out, pool);
            }
        }
    }
}

impl QTensor {
    /// The device GEMM this tensor would take, run once on the caller's
    /// data — the startup parity probe's arm, and the one place that knows
    /// which entry point each codec has.
    ///
    /// It exists because the probe used to look for a `q4tp` weight by
    /// name AND dtype, and a container packed any other way was declared
    /// "host path" for the whole render even though its codec had a device
    /// GEMM of its own. A gate that only recognizes one codec is a gate
    /// that silently downgrades every other one.
    pub fn device_matmat(&self, xs: &[f32], b: usize, out: &mut [f32]) -> bool {
        let (rows, cols) = (self.rows(), self.cols());
        let Self::Mapped { model, idx, dtype, row_scale, col_field, .. } = self else {
            return false;
        };
        match *dtype {
            TensorDtype::Q4TiledP => {
                crate::gpu::q4tp_matmat(model, *idx, xs, b, rows, cols, out)
            }
            // The two-field codec folds its column field into the
            // activation, which leaves a plain per-row int8 GEMM — the
            // same kernel `q8_row` uses, on both backends.
            TensorDtype::Q8Row | TensorDtype::Q8_2f => {
                let flat: Vec<f32> = (0..b)
                    .flat_map(|bi| {
                        prescale(&xs[bi * cols..(bi + 1) * cols], col_field, *dtype).into_owned()
                    })
                    .collect();
                crate::gpu::q8_matmat(model, *idx, row_scale, &flat, b, rows, cols, out)
            }
            _ => false,
        }
    }

    /// Multi-matrix job (roadmap §3 P0): N tensors sharing one input
    /// run under a SINGLE pool dispatch — QKV or gate+up cost one
    /// barrier instead of N. Per-row math is the exact same kernel as
    /// `matvec` (bit-identical outputs); only the dispatch is fused.
    /// Falls back to N sequential matvecs when the set is not a uniform
    /// q8-family/F32 group or there is no pool.
    pub fn matvec_many<const N: usize>(
        ts: [&QTensor; N],
        x: &[f32],
        mut outs: [&mut [f32]; N],
        pool: Option<&Pool>,
    ) {
        let total_rows: usize = ts.iter().map(|t| t.rows()).sum();
        let uniform_q8 = ts.iter().all(|t| {
            matches!(
                t,
                Self::Mapped {
                    dtype: TensorDtype::Q8Row | TensorDtype::Q8_2f,
                    ..
                }
            )
        });
        let uniform_f32 = ts.iter().all(|t| matches!(t, Self::F32 { .. }));
        let uniform_q4 = ts.iter().all(|t| {
            matches!(
                t,
                Self::Mapped {
                    dtype: TensorDtype::Q4Block,
                    ..
                }
            )
        });
        let uniform_vbit = ts.iter().all(|t| {
            matches!(
                t,
                Self::Mapped {
                    dtype: TensorDtype::Vbit | TensorDtype::VbitRo,
                    ..
                }
            )
        });
        let uniform_q1 = ts.iter().all(|t| {
            matches!(
                t,
                Self::Mapped {
                    dtype: TensorDtype::Q1,
                    ..
                }
            )
        });
        let uniform_q1t = ts.iter().all(|t| {
            matches!(
                t,
                Self::Mapped {
                    dtype: TensorDtype::Q1T,
                    ..
                }
            )
        });
        // q4tp is the skeleton dtype of the big MoE files, and without an arm
        // here every projection that shares an input paid its own pool
        // barrier: DeepSeek-V4's attention step alone hands this function
        // wq_a, wkv and both compressors' pairs off the same hidden state.
        let uniform_q4tp = ts.iter().all(|t| {
            matches!(
                t,
                Self::Mapped {
                    dtype: TensorDtype::Q4TiledP,
                    ..
                }
            )
        }) && ts
            .iter()
            .all(|t| t.cols() == ts[0].cols() && t.cols() % GROUP_SIZE == 0);
        let Some(pool) = pool else {
            for (t, o) in ts.iter().zip(outs.iter_mut()) {
                t.matvec(x, o, None);
            }
            return;
        };
        if total_rows < 256
            || !(uniform_q8
                || uniform_f32
                || uniform_q4
                || uniform_vbit
                || uniform_q1
                || uniform_q1t
                || uniform_q4tp)
        {
            for (t, o) in ts.iter().zip(outs.iter_mut()) {
                t.matvec(x, o, Some(pool));
            }
            return;
        }

        if uniform_q4tp {
            // Every tensor's rows laid end to end in one virtual row space,
            // so the whole set is ONE dispatch. The per-row body is the
            // `q4tp_matvec` arm verbatim — same activation split, same
            // accumulation order — so the outputs are bit-identical to the
            // sequential calls this replaces.
            let cols = ts[0].cols();
            let gpr = cols / GROUP_SIZE;
            let views: Vec<Q4tpView> = ts
                .iter()
                .map(|t| Q4tpView::new(t.quant_bytes(), t.rows(), cols))
                .collect();
            let rows_of: Vec<usize> = ts.iter().map(|t| t.rows()).collect();
            let outs_addr: [SendMut; N] = std::array::from_fn(|i| SendMut(outs[i].as_mut_ptr()));
            // flat index -> (which tensor, which of its rows)
            let locate = |flat: usize| -> (usize, usize) {
                let mut acc = 0;
                for (i, &r) in rows_of.iter().enumerate() {
                    if flat < acc + r {
                        return (i, flat - acc);
                    }
                    acc += r;
                }
                (rows_of.len() - 1, 0)
            };
            let (views, outs_addr) = (&views, &outs_addr);
            if a8w8_enabled() {
                let act = split_act(x);
                let act = &act;
                let run = |start: usize, end: usize| {
                    let mut sc = vec![0f32; gpr];
                    for flat in start..end {
                        let (t, r) = locate(flat);
                        let v = &views[t];
                        v.scales_into(r, gpr, &mut sc);
                        let mut acc = dot_q4tp_row_i8(v.nib, r, gpr, &act.xq, &sc) * act.sx;
                        for &(j, xv) in &act.outliers {
                            let (w, s) = q4tp_outlier(v.nib, r, gpr, j, &sc);
                            acc += w * s * xv;
                        }
                        // SAFETY: one worker owns each (tensor, row) pair.
                        unsafe { *outs_addr[t].at(r) = acc };
                    }
                };
                pool.run_rows(total_rows, &run);
            } else {
                let run = |start: usize, end: usize| {
                    let mut sc = vec![0f32; gpr];
                    for flat in start..end {
                        let (t, r) = locate(flat);
                        let v = &views[t];
                        v.scales_into(r, gpr, &mut sc);
                        // SAFETY: one worker owns each (tensor, row) pair.
                        unsafe { *outs_addr[t].at(r) = q4tp_row_exact(v.nib, r, gpr, x, &sc) };
                    }
                };
                pool.run_rows(total_rows, &run);
            }
            return;
        }

        if uniform_q1 {
            // One shared activation split + group sums (q1 has no col
            // field; the same input feeds every tensor).
            let outs_addr: [SendMut; N] = std::array::from_fn(|i| SendMut(outs[i].as_mut_ptr()));
            if a8w8_enabled() {
                let act = split_act(x);
                let gsum = q1_group_sums(&act.xq, ts[0].cols() / GROUP_SIZE);
                let (act, gsum) = (&act, &gsum);
                let closures: [_; N] = std::array::from_fn(|i| {
                    let (bytes, gpr, out) =
                        (ts[i].quant_bytes(), ts[i].cols() / GROUP_SIZE, outs_addr[i]);
                    move |s: usize, e: usize| q1_range_a8w8(bytes, gpr, act, gsum, out, s, e)
                });
                let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                    std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
                pool.run_many(&parts);
            } else {
                let closures: [_; N] = std::array::from_fn(|i| {
                    let (bytes, gpr, out) =
                        (ts[i].quant_bytes(), ts[i].cols() / GROUP_SIZE, outs_addr[i]);
                    move |s: usize, e: usize| q1_range_f32(bytes, gpr, x, out, s, e)
                });
                let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                    std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
                pool.run_many(&parts);
            }
            return;
        }

        if uniform_q1t {
            // Q1T batched: one shared activation split + overlay decode,
            // all tensors' rows in ONE pool dispatch (saves N−1 dispatches
            // and N−1 redundant split_act calls per layer).
            let outs_addr: [SendMut; N] = std::array::from_fn(|i| SendMut(outs[i].as_mut_ptr()));
            const TILE: usize = cortiq_core::quant::Q1T_TILE;
            if a8w8_enabled() {
                let act = split_act(x);
                let act = &act;
                let x_ref = x;
                let closures: [_; N] = std::array::from_fn(|i| {
                    let bytes = ts[i].quant_bytes();
                    let (rows, cols) = (ts[i].rows(), ts[i].cols());
                    let gpr = cols / GROUP_SIZE;
                    let (rp_off, ent_off, has_ov) = q1t_overlay(bytes, rows * gpr * TILE, rows);
                    let out = outs_addr[i];
                    move |s: usize, e: usize| {
                        q1t_range_a8w8(bytes, gpr, rp_off, ent_off, has_ov, act, x_ref, out, s, e)
                    }
                });
                let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                    std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
                pool.run_many(&parts);
            } else {
                let x_ref = x;
                let closures: [_; N] = std::array::from_fn(|i| {
                    let bytes = ts[i].quant_bytes();
                    let (rows, cols) = (ts[i].rows(), ts[i].cols());
                    let gpr = cols / GROUP_SIZE;
                    let (rp_off, ent_off, has_ov) = q1t_overlay(bytes, rows * gpr * TILE, rows);
                    let out = outs_addr[i];
                    move |s: usize, e: usize| {
                        q1t_range_f32_batch(bytes, gpr, rp_off, ent_off, has_ov, x_ref, out, s, e)
                    }
                });
                let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                    std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
                pool.run_many(&parts);
            }
            return;
        }

        if uniform_q4 || uniform_vbit {
            let outs_addr: [SendMut; N] = std::array::from_fn(|i| SendMut(outs[i].as_mut_ptr()));
            // q4/vbit share one activation split — no per-tensor col field.
            if a8w8_enabled() {
                let act = split_act(x);
                let act = &act;
                if uniform_q4 {
                    let closures: [_; N] = std::array::from_fn(|i| {
                        let (packed, scales) =
                            q4_split(ts[i].quant_bytes(), ts[i].rows(), ts[i].cols());
                        let (gpr, cols, out) =
                            (ts[i].cols() / GROUP_SIZE, ts[i].cols(), outs_addr[i]);
                        move |s: usize, e: usize| {
                            q4_range_a8w8(packed, scales, gpr, cols, act, out, s, e)
                        }
                    });
                    let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                        std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
                    pool.run_many(&parts);
                } else {
                    let closures: [_; N] = std::array::from_fn(|i| {
                        let Self::Mapped { vbit_offsets, .. } = ts[i] else {
                            unreachable!()
                        };
                        let (bytes, rows, cols, out) = (
                            ts[i].quant_bytes(),
                            ts[i].rows(),
                            ts[i].cols(),
                            outs_addr[i],
                        );
                        move |s: usize, e: usize| {
                            vbit_range_a8w8(bytes, vbit_offsets, x, act, rows, cols, out, s, e)
                        }
                    });
                    let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                        std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
                    pool.run_many(&parts);
                }
                return;
            }
            if uniform_q4 {
                let closures: [_; N] = std::array::from_fn(|i| {
                    let (packed, scales) =
                        q4_split(ts[i].quant_bytes(), ts[i].rows(), ts[i].cols());
                    let (gpr, out) = (ts[i].cols() / GROUP_SIZE, outs_addr[i]);
                    move |s: usize, e: usize| q4_range_f32(packed, scales, gpr, x, out, s, e)
                });
                let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                    std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
                pool.run_many(&parts);
            } else {
                let closures: [_; N] = std::array::from_fn(|i| {
                    let Self::Mapped { vbit_offsets, .. } = ts[i] else {
                        unreachable!()
                    };
                    let (bytes, rows, cols, out) = (
                        ts[i].quant_bytes(),
                        ts[i].rows(),
                        ts[i].cols(),
                        outs_addr[i],
                    );
                    move |s: usize, e: usize| {
                        vbit_range_f32(bytes, vbit_offsets, x, rows, cols, out, s, e)
                    }
                });
                let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                    std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
                pool.run_many(&parts);
            }
            return;
        }

        if uniform_f32 {
            let outs_addr: [SendMut; N] = std::array::from_fn(|i| SendMut(outs[i].as_mut_ptr()));
            let closures: [_; N] = std::array::from_fn(|i| {
                let Self::F32 { data, cols, .. } = ts[i] else {
                    unreachable!()
                };
                let out = outs_addr[i];
                move |start: usize, end: usize| {
                    for o in start..end {
                        let row = &data[o * cols..(o + 1) * cols];
                        let mut sum = 0.0f32;
                        for j in 0..*cols {
                            sum += row[j] * x[j];
                        }
                        // SAFETY: disjoint (tensor, row) cells per worker.
                        unsafe { *out.at(o) = sum };
                    }
                }
            });
            let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
            pool.run_many(&parts);
            return;
        }

        // Uniform q8-family: per-tensor prescale (q8_2f col fields
        // differ per tensor) + the shared range kernels.
        struct Ctx<'a> {
            bytes: &'a [u8],
            #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
            rep: &'a [u8],
            row_scale: &'a [f32],
            cols: usize,
            xs: std::borrow::Cow<'a, [f32]>,
        }
        let ctxs: [Ctx<'_>; N] = std::array::from_fn(|i| {
            let Self::Mapped {
                dtype,
                cols,
                row_scale,
                col_field,
                repack,
                ..
            } = ts[i]
            else {
                unreachable!()
            };
            Ctx {
                bytes: ts[i].quant_bytes(),
                rep: repack,
                row_scale,
                cols: *cols,
                xs: prescale(x, col_field, *dtype),
            }
        });
        let outs_addr: [SendMut; N] = std::array::from_fn(|i| SendMut(outs[i].as_mut_ptr()));
        #[cfg(target_arch = "aarch64")]
        if sdot_enabled() {
            let acts: [SplitAct; N] = std::array::from_fn(|i| split_act(&ctxs[i].xs));
            let closures: [_; N] = std::array::from_fn(|i| {
                let (c, act, out) = (&ctxs[i], &acts[i], outs_addr[i]);
                move |start: usize, end: usize| {
                    q8_range_sdot(c.bytes, c.rep, c.row_scale, act, c.cols, out, start, end)
                }
            });
            let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
            pool.run_many(&parts);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if avx2_a8w8_enabled() {
            let acts: [SplitAct; N] = std::array::from_fn(|i| split_act(&ctxs[i].xs));
            let closures: [_; N] = std::array::from_fn(|i| {
                let (c, act, out) = (&ctxs[i], &acts[i], outs_addr[i]);
                move |start: usize, end: usize| {
                    q8_range_avx2(c.bytes, c.row_scale, act, c.cols, out, start, end)
                }
            });
            let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
            pool.run_many(&parts);
            return;
        }
        let closures: [_; N] = std::array::from_fn(|i| {
            let (c, out) = (&ctxs[i], outs_addr[i]);
            move |start: usize, end: usize| {
                q8_range_f32(c.bytes, c.row_scale, &c.xs, c.cols, out, start, end)
            }
        });
        let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
            std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
        pool.run_many(&parts);
    }
}

impl QTensor {
    /// Pair-input multi-matrix job: N tensors × 2 shared inputs under a
    /// single pool dispatch — the MTP/pair decode path publishes one job
    /// for Q/K/V (and one for gate+up) instead of one per tensor.
    /// Per-row math is exactly `matvec2`'s kernels; bit-identical.
    #[allow(clippy::needless_range_loop)]
    pub fn matvec2_many<const N: usize>(
        ts: [&QTensor; N],
        x1: &[f32],
        x2: &[f32],
        mut o1s: [&mut [f32]; N],
        mut o2s: [&mut [f32]; N],
        pool: Option<&Pool>,
    ) {
        let total_rows: usize = ts.iter().map(|t| t.rows()).sum();
        let uniform_q8 = ts.iter().all(|t| {
            matches!(
                t,
                Self::Mapped {
                    dtype: TensorDtype::Q8Row | TensorDtype::Q8_2f,
                    ..
                }
            )
        });
        let uniform_f32 = ts.iter().all(|t| matches!(t, Self::F32 { .. }));
        let uniform_q4 = ts.iter().all(|t| {
            matches!(
                t,
                Self::Mapped {
                    dtype: TensorDtype::Q4Block,
                    ..
                }
            )
        });
        let uniform_vbit = ts.iter().all(|t| {
            matches!(
                t,
                Self::Mapped {
                    dtype: TensorDtype::Vbit | TensorDtype::VbitRo,
                    ..
                }
            )
        });
        let fusable = pool.is_some()
            && total_rows >= 256
            && (uniform_q8 || uniform_f32 || uniform_q4 || uniform_vbit);
        if !fusable {
            for i in 0..N {
                ts[i].matvec2(x1, x2, o1s[i], o2s[i], pool);
            }
            return;
        }
        let pool = pool.unwrap();

        if uniform_q4 || uniform_vbit {
            let p1: [SendMut; N] = std::array::from_fn(|i| SendMut(o1s[i].as_mut_ptr()));
            let p2: [SendMut; N] = std::array::from_fn(|i| SendMut(o2s[i].as_mut_ptr()));
            // q4/vbit share activation splits — no per-tensor col field.
            if a8w8_enabled() {
                let a1 = split_act(x1);
                let a2 = split_act(x2);
                let (a1, a2) = (&a1, &a2);
                if uniform_q4 {
                    let closures: [_; N] = std::array::from_fn(|i| {
                        let (packed, scales) =
                            q4_split(ts[i].quant_bytes(), ts[i].rows(), ts[i].cols());
                        let (gpr, cols, o1, o2) =
                            (ts[i].cols() / GROUP_SIZE, ts[i].cols(), p1[i], p2[i]);
                        move |s: usize, e: usize| {
                            q4_range2_a8w8(packed, scales, gpr, cols, a1, a2, o1, o2, s, e)
                        }
                    });
                    let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                        std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
                    pool.run_many(&parts);
                } else {
                    let closures: [_; N] = std::array::from_fn(|i| {
                        let Self::Mapped { vbit_offsets, .. } = ts[i] else {
                            unreachable!()
                        };
                        let (bytes, rows, cols, o1, o2) = (
                            ts[i].quant_bytes(),
                            ts[i].rows(),
                            ts[i].cols(),
                            p1[i],
                            p2[i],
                        );
                        move |s: usize, e: usize| {
                            vbit_range2_a8w8(
                                bytes,
                                vbit_offsets,
                                x1,
                                x2,
                                a1,
                                a2,
                                rows,
                                cols,
                                o1,
                                o2,
                                s,
                                e,
                            )
                        }
                    });
                    let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                        std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
                    pool.run_many(&parts);
                }
                return;
            }
            if uniform_q4 {
                let closures: [_; N] = std::array::from_fn(|i| {
                    let (packed, scales) =
                        q4_split(ts[i].quant_bytes(), ts[i].rows(), ts[i].cols());
                    let (gpr, o1, o2) = (ts[i].cols() / GROUP_SIZE, p1[i], p2[i]);
                    move |s: usize, e: usize| {
                        q4_range2_f32(packed, scales, gpr, x1, x2, o1, o2, s, e)
                    }
                });
                let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                    std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
                pool.run_many(&parts);
            } else {
                let closures: [_; N] = std::array::from_fn(|i| {
                    let Self::Mapped { vbit_offsets, .. } = ts[i] else {
                        unreachable!()
                    };
                    let (bytes, rows, cols, o1, o2) = (
                        ts[i].quant_bytes(),
                        ts[i].rows(),
                        ts[i].cols(),
                        p1[i],
                        p2[i],
                    );
                    move |s: usize, e: usize| {
                        vbit_range2_f32(bytes, vbit_offsets, x1, x2, rows, cols, o1, o2, s, e)
                    }
                });
                let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                    std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
                pool.run_many(&parts);
            }
            return;
        }

        if uniform_f32 {
            let p1: [SendMut; N] = std::array::from_fn(|i| SendMut(o1s[i].as_mut_ptr()));
            let p2: [SendMut; N] = std::array::from_fn(|i| SendMut(o2s[i].as_mut_ptr()));
            let closures: [_; N] = std::array::from_fn(|i| {
                let Self::F32 { data, cols, .. } = ts[i] else {
                    unreachable!()
                };
                let (o1, o2) = (p1[i], p2[i]);
                move |start: usize, end: usize| {
                    for o in start..end {
                        let row = &data[o * cols..(o + 1) * cols];
                        let (mut s1, mut s2) = (0.0f32, 0.0f32);
                        for j in 0..*cols {
                            s1 += row[j] * x1[j];
                            s2 += row[j] * x2[j];
                        }
                        // SAFETY: disjoint (tensor, row) cells per worker.
                        unsafe {
                            *o1.at(o) = s1;
                            *o2.at(o) = s2;
                        }
                    }
                }
            });
            let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
            pool.run_many(&parts);
            return;
        }

        struct Ctx<'a> {
            bytes: &'a [u8],
            row_scale: &'a [f32],
            cols: usize,
            xs1: std::borrow::Cow<'a, [f32]>,
            xs2: std::borrow::Cow<'a, [f32]>,
        }
        let ctxs: [Ctx<'_>; N] = std::array::from_fn(|i| {
            let Self::Mapped {
                dtype,
                cols,
                row_scale,
                col_field,
                ..
            } = ts[i]
            else {
                unreachable!()
            };
            Ctx {
                bytes: ts[i].quant_bytes(),
                row_scale,
                cols: *cols,
                xs1: prescale(x1, col_field, *dtype),
                xs2: prescale(x2, col_field, *dtype),
            }
        });
        let p1: [SendMut; N] = std::array::from_fn(|i| SendMut(o1s[i].as_mut_ptr()));
        let p2: [SendMut; N] = std::array::from_fn(|i| SendMut(o2s[i].as_mut_ptr()));
        #[cfg(target_arch = "aarch64")]
        if sdot_enabled() {
            let acts: [(SplitAct, SplitAct); N] =
                std::array::from_fn(|i| (split_act(&ctxs[i].xs1), split_act(&ctxs[i].xs2)));
            let closures: [_; N] = std::array::from_fn(|i| {
                let (c, a, o1, o2) = (&ctxs[i], &acts[i], p1[i], p2[i]);
                move |start: usize, end: usize| {
                    q8_range2_sdot(c.bytes, c.row_scale, &a.0, &a.1, c.cols, o1, o2, start, end)
                }
            });
            let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
            pool.run_many(&parts);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if avx2_a8w8_enabled() {
            let acts: [(SplitAct, SplitAct); N] =
                std::array::from_fn(|i| (split_act(&ctxs[i].xs1), split_act(&ctxs[i].xs2)));
            let closures: [_; N] = std::array::from_fn(|i| {
                let (c, a, o1, o2) = (&ctxs[i], &acts[i], p1[i], p2[i]);
                move |start: usize, end: usize| {
                    q8_range2_avx2(c.bytes, c.row_scale, &a.0, &a.1, c.cols, o1, o2, start, end)
                }
            });
            let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
                std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
            pool.run_many(&parts);
            return;
        }
        let closures: [_; N] = std::array::from_fn(|i| {
            let (c, o1, o2) = (&ctxs[i], p1[i], p2[i]);
            move |start: usize, end: usize| {
                q8_range2_f32(
                    c.bytes,
                    c.row_scale,
                    &c.xs1,
                    &c.xs2,
                    c.cols,
                    o1,
                    o2,
                    start,
                    end,
                )
            }
        });
        let parts: [(usize, &(dyn Fn(usize, usize) + Sync)); N] =
            std::array::from_fn(|i| (ts[i].rows(), &closures[i] as _));
        pool.run_many(&parts);
    }

    /// Fused gate+up matvec with SiLU·mul: for each row r, computes
    /// `silu(gate·x) * (up·x)` and writes to `out[r]`. ONE pool dispatch,
    /// no intermediate g/u buffers, no separate silu pass. Falls back
    /// (returns false) for unsupported dtype combos.
    pub fn matvec_silu_mul(
        gate: &QTensor,
        up: &QTensor,
        x: &[f32],
        out: &mut [f32],
        pool: Option<&Pool>,
    ) -> bool {
        let inter = gate.rows();
        debug_assert_eq!(up.rows(), inter);
        debug_assert_eq!(out.len(), inter);
        debug_assert_eq!(gate.cols(), up.cols());
        if !a8w8_enabled() {
            return false;
        }
        let act = split_act(x);
        let act = &act;
        let x_ref = x;
        let out_addr = SendMut(out.as_mut_ptr());

        match (gate, up) {
            // Q4Block gate + Q4Block up (most common mobile q4 models)
            (
                Self::Mapped {
                    dtype: TensorDtype::Q4Block,
                    ..
                },
                Self::Mapped {
                    dtype: TensorDtype::Q4Block,
                    ..
                },
            ) => {
                let (gp, gs) = q4_split(gate.quant_bytes(), gate.rows(), gate.cols());
                let (up_p, up_s) = q4_split(up.quant_bytes(), up.rows(), up.cols());
                let gpr = gate.cols() / GROUP_SIZE;
                let cols = gate.cols();
                let run = move |start: usize, end: usize| {
                    for r in start..end {
                        let mut gv = dot_q4_row_i8(gp, gs, r * gpr, gpr, &act.xq) * act.sx;
                        let mut uv = dot_q4_row_i8(up_p, up_s, r * gpr, gpr, &act.xq) * act.sx;
                        for &(j, xv) in &act.outliers {
                            let flat = r * cols + j;
                            let gb = gp[flat / 2];
                            let gn = if flat & 1 == 0 { gb & 0x0F } else { gb >> 4 };
                            let gsc = f16_to_f32(u16::from_le_bytes([
                                gs[(flat / GROUP_SIZE) * 2],
                                gs[(flat / GROUP_SIZE) * 2 + 1],
                            ]));
                            gv += ((gn as i32 - 8) as f32) * gsc * xv;
                            let ub = up_p[flat / 2];
                            let un = if flat & 1 == 0 { ub & 0x0F } else { ub >> 4 };
                            let usc = f16_to_f32(u16::from_le_bytes([
                                up_s[(flat / GROUP_SIZE) * 2],
                                up_s[(flat / GROUP_SIZE) * 2 + 1],
                            ]));
                            uv += ((un as i32 - 8) as f32) * usc * xv;
                        }
                        let silu_g = gv / (1.0 + (-gv).exp());
                        // SAFETY: disjoint row ranges per worker.
                        unsafe { *out_addr.at(r) = silu_g * uv };
                    }
                };
                dispatch_rows(pool, inter, &run);
                true
            }
            // Q4Tiled gate + Q4Tiled up — one row pass, both tile
            // streams sequential, silu·mul fused (same per-row math as
            // `q4t_matvec`).
            (
                Self::Mapped {
                    dtype: TensorDtype::Q4Tiled,
                    ..
                },
                Self::Mapped {
                    dtype: TensorDtype::Q4Tiled,
                    ..
                },
            ) => {
                let g_bytes = gate.quant_bytes();
                let u_bytes = up.quant_bytes();
                let gpr = gate.cols() / GROUP_SIZE;
                let run = move |start: usize, end: usize| {
                    for r in start..end {
                        let mut gv = dot_q4t_row_i8(g_bytes, r, gpr, &act.xq) * act.sx;
                        let mut uv = dot_q4t_row_i8(u_bytes, r, gpr, &act.xq) * act.sx;
                        for &(j, xv) in &act.outliers {
                            let (w, s) = q4t_outlier(g_bytes, r, gpr, j);
                            gv += w * s * xv;
                            let (w, s) = q4t_outlier(u_bytes, r, gpr, j);
                            uv += w * s * xv;
                        }
                        let silu_g = gv / (1.0 + (-gv).exp());
                        // SAFETY: disjoint row ranges per worker.
                        unsafe { *out_addr.at(r) = silu_g * uv };
                    }
                };
                dispatch_rows(pool, inter, &run);
                true
            }
            // Q4TiledP gate + Q4TiledP up — the same fused row pass, with
            // each row's two ladders built once and spent on both streams.
            (
                Self::Mapped {
                    dtype: TensorDtype::Q4TiledP,
                    ..
                },
                Self::Mapped {
                    dtype: TensorDtype::Q4TiledP,
                    ..
                },
            ) => {
                let cols = gate.cols();
                let gpr = cols / GROUP_SIZE;
                let gv_view = Q4tpView::new(gate.quant_bytes(), inter, cols);
                let uv_view = Q4tpView::new(up.quant_bytes(), inter, cols);
                let run = |start: usize, end: usize| {
                    let (mut gsc, mut usc) = (vec![0f32; gpr], vec![0f32; gpr]);
                    for r in start..end {
                        gv_view.scales_into(r, gpr, &mut gsc);
                        uv_view.scales_into(r, gpr, &mut usc);
                        let mut gv = dot_q4tp_row_i8(gv_view.nib, r, gpr, &act.xq, &gsc) * act.sx;
                        let mut uv = dot_q4tp_row_i8(uv_view.nib, r, gpr, &act.xq, &usc) * act.sx;
                        for &(j, xv) in &act.outliers {
                            let (w, s) = q4tp_outlier(gv_view.nib, r, gpr, j, &gsc);
                            gv += w * s * xv;
                            let (w, s) = q4tp_outlier(uv_view.nib, r, gpr, j, &usc);
                            uv += w * s * xv;
                        }
                        let silu_g = gv / (1.0 + (-gv).exp());
                        // SAFETY: disjoint row ranges per worker.
                        unsafe { *out_addr.at(r) = silu_g * uv };
                    }
                };
                dispatch_rows(pool, inter, &run);
                true
            }
            // Q1 gate + Q1 up — one row pass over both sign streams,
            // silu·mul fused (the per-row math of `q1_range_a8w8`); the
            // activation group sums are shared by both streams. Without
            // this arm a q1 dense FFN paid two dispatches + a combine
            // loop — the exact barrier this function exists to remove.
            (
                Self::Mapped {
                    dtype: TensorDtype::Q1,
                    ..
                },
                Self::Mapped {
                    dtype: TensorDtype::Q1,
                    ..
                },
            ) => {
                let g_bytes = gate.quant_bytes();
                let u_bytes = up.quant_bytes();
                let gpr = gate.cols() / GROUP_SIZE;
                let gsum = q1_group_sums(&act.xq, gpr);
                let gsum = &gsum;
                let run = move |start: usize, end: usize| {
                    for r in start..end {
                        let mut gv = dot_q1_row_i8(g_bytes, r, gpr, &act.xq, gsum) * act.sx;
                        let mut uv = dot_q1_row_i8(u_bytes, r, gpr, &act.xq, gsum) * act.sx;
                        for &(j, xv) in &act.outliers {
                            let (w, s) = q1_outlier(g_bytes, r, gpr, j);
                            gv += w * s * xv;
                            let (w, s) = q1_outlier(u_bytes, r, gpr, j);
                            uv += w * s * xv;
                        }
                        let silu_g = gv / (1.0 + (-gv).exp());
                        // SAFETY: disjoint row ranges per worker.
                        unsafe { *out_addr.at(r) = silu_g * uv };
                    }
                };
                dispatch_rows(pool, inter, &run);
                true
            }
            // Q2TiledP gate + Q2TiledP up — the 2-bit expert pair (MoE
            // FFNs of the W2 class): one row pass, both ladders built
            // once, integer code dots with shared group sums.
            (
                Self::Mapped {
                    dtype: TensorDtype::Q2TiledP,
                    ..
                },
                Self::Mapped {
                    dtype: TensorDtype::Q2TiledP,
                    ..
                },
            ) => {
                let cols = gate.cols();
                let gpr = cols / GROUP_SIZE;
                let gv_view = Q4tpView::new_q2(gate.quant_bytes(), inter, cols);
                let uv_view = Q4tpView::new_q2(up.quant_bytes(), inter, cols);
                let gsum = q1_group_sums(&act.xq, gpr);
                let gsum = &gsum;
                let run = move |start: usize, end: usize| {
                    let (mut gsc, mut usc) = (vec![0f32; gpr], vec![0f32; gpr]);
                    for r in start..end {
                        gv_view.scales_into(r, gpr, &mut gsc);
                        uv_view.scales_into(r, gpr, &mut usc);
                        let mut gv =
                            dot_q2tp_row_i8(gv_view.nib, r, gpr, &act.xq, gsum, &gsc) * act.sx;
                        let mut uv =
                            dot_q2tp_row_i8(uv_view.nib, r, gpr, &act.xq, gsum, &usc) * act.sx;
                        for &(j, xv) in &act.outliers {
                            let (w, s) = q2tp_outlier(gv_view.nib, r, gpr, j, &gsc);
                            gv += w * s * xv;
                            let (w, s) = q2tp_outlier(uv_view.nib, r, gpr, j, &usc);
                            uv += w * s * xv;
                        }
                        let silu_g = gv / (1.0 + (-gv).exp());
                        // SAFETY: disjoint row ranges per worker.
                        unsafe { *out_addr.at(r) = silu_g * uv };
                    }
                };
                dispatch_rows(pool, inter, &run);
                true
            }
            // Q8Row gate + Q8Row up — one row pass over both i8 streams.
            // Q8_2f stays out on purpose: its column field prescales the
            // activations PER TENSOR, which breaks this fn's shared
            // split_act contract — it keeps the two-dispatch path.
            (
                Self::Mapped {
                    dtype: TensorDtype::Q8Row,
                    row_scale: g_rs,
                    ..
                },
                Self::Mapped {
                    dtype: TensorDtype::Q8Row,
                    row_scale: u_rs,
                    ..
                },
            ) => {
                let g_bytes = gate.quant_bytes();
                let u_bytes = up.quant_bytes();
                let cols = gate.cols();
                let run = move |start: usize, end: usize| {
                    for r in start..end {
                        let gv = q8_row_dot(&g_bytes[r * cols..(r + 1) * cols], act) * g_rs[r];
                        let uv = q8_row_dot(&u_bytes[r * cols..(r + 1) * cols], act) * u_rs[r];
                        let silu_g = gv / (1.0 + (-gv).exp());
                        // SAFETY: disjoint row ranges per worker.
                        unsafe { *out_addr.at(r) = silu_g * uv };
                    }
                };
                dispatch_rows(pool, inter, &run);
                true
            }
            // Q1T gate + Q1T up
            (
                Self::Mapped {
                    dtype: TensorDtype::Q1T,
                    ..
                },
                Self::Mapped {
                    dtype: TensorDtype::Q1T,
                    ..
                },
            ) => {
                const TILE: usize = cortiq_core::quant::Q1T_TILE;
                let g_bytes = gate.quant_bytes();
                let u_bytes = up.quant_bytes();
                let gpr = gate.cols() / GROUP_SIZE;
                let (g_rp, g_ent, g_ov) = q1t_overlay(g_bytes, inter * gpr * TILE, inter);
                let (u_rp, u_ent, u_ov) = q1t_overlay(u_bytes, inter * gpr * TILE, inter);
                let run = move |start: usize, end: usize| {
                    for r in start..end {
                        let mut gv = q1t_dot_row_i8(g_bytes, r, gpr, &act.xq) * act.sx;
                        let mut uv = q1t_dot_row_i8(u_bytes, r, gpr, &act.xq) * act.sx;
                        for &(j, xv) in &act.outliers {
                            gv += q1t_base_weight(g_bytes, r, gpr, j) * xv;
                            uv += q1t_base_weight(u_bytes, r, gpr, j) * xv;
                        }
                        gv += q1t_row_outlier_correction(g_bytes, r, g_rp, g_ent, g_ov, x_ref);
                        uv += q1t_row_outlier_correction(u_bytes, r, u_rp, u_ent, u_ov, x_ref);
                        let silu_g = gv / (1.0 + (-gv).exp());
                        // SAFETY: disjoint row ranges per worker.
                        unsafe { *out_addr.at(r) = silu_g * uv };
                    }
                };
                dispatch_rows(pool, inter, &run);
                true
            }
            _ => false,
        }
    }

    /// Every routed expert's fused gate/up/SiLU under ONE pool dispatch.
    ///
    /// The per-expert path pays a pool barrier per expert per stage: at 9
    /// experts over 40 layers that is ~720 barriers a token, and a decode
    /// profile of Qwen3.6-35B-A3B showed the pool parked in
    /// `psynch_cvwait` about twice as long as it spent computing. Laying
    /// every expert's rows end-to-end in one virtual row space collapses
    /// the stage to a single dispatch. The per-row body is the
    /// single-expert q4tp arm verbatim, so outputs are bit-identical.
    ///
    /// `false` = something is outside the fused q4tp kernel (dtype, shape,
    /// or the `CMF_SDOT=0` exact contract); the caller walks the ordinary
    /// per-expert path.
    pub fn moe_gate_up_many(
        pairs: &[(&QTensor, &QTensor)],
        x: &[f32],
        outs: &mut [Vec<f32>],
        pool: Option<&Pool>,
    ) -> bool {
        if pairs.is_empty() || pairs.len() != outs.len() || !a8w8_enabled() {
            return false;
        }
        let inter = pairs[0].0.rows();
        let cols = pairs[0].0.cols();
        if cols % GROUP_SIZE != 0 {
            return false;
        }
        let gpr = cols / GROUP_SIZE;
        // Uniform layout across every routed pair: q4tp, or the 2-bit
        // profile's q2tp gate/up (the W2 class). Mixed sets refuse.
        let q2 = matches!(
            pairs[0].0,
            Self::Mapped {
                dtype: TensorDtype::Q2TiledP,
                ..
            }
        );
        let want = if q2 {
            TensorDtype::Q2TiledP
        } else {
            TensorDtype::Q4TiledP
        };
        let mut views = Vec::with_capacity(pairs.len() * 2);
        for ((g, u), o) in pairs.iter().zip(outs.iter()) {
            let both = matches!(g, Self::Mapped { dtype, .. } if *dtype == want)
                && matches!(u, Self::Mapped { dtype, .. } if *dtype == want);
            if !both
                || g.rows() != inter
                || u.rows() != inter
                || g.cols() != cols
                || u.cols() != cols
                || o.len() != inter
            {
                return false;
            }
            let mk = if q2 { Q4tpView::new_q2 } else { Q4tpView::new };
            views.push(mk(g.quant_bytes(), inter, cols));
            views.push(mk(u.quant_bytes(), inter, cols));
        }
        let act = split_act(x);
        let gsum = if q2 {
            q1_group_sums(&act.xq, gpr)
        } else {
            Vec::new()
        };
        let (act, gsum) = (&act, &gsum);
        let ptrs: Vec<SendMut> = outs.iter_mut().map(|o| SendMut(o.as_mut_ptr())).collect();
        let (views, ptrs) = (&views, &ptrs);
        let run = |start: usize, end: usize| {
            let (mut gsc, mut usc) = (vec![0f32; gpr], vec![0f32; gpr]);
            for flat in start..end {
                let (e, r) = (flat / inter, flat % inter);
                let gv_view = &views[e * 2];
                let uv_view = &views[e * 2 + 1];
                gv_view.scales_into(r, gpr, &mut gsc);
                uv_view.scales_into(r, gpr, &mut usc);
                let (mut gv, mut uv) = if q2 {
                    (
                        dot_q2tp_row_i8(gv_view.nib, r, gpr, &act.xq, gsum, &gsc) * act.sx,
                        dot_q2tp_row_i8(uv_view.nib, r, gpr, &act.xq, gsum, &usc) * act.sx,
                    )
                } else {
                    (
                        dot_q4tp_row_i8(gv_view.nib, r, gpr, &act.xq, &gsc) * act.sx,
                        dot_q4tp_row_i8(uv_view.nib, r, gpr, &act.xq, &usc) * act.sx,
                    )
                };
                for &(j, xv) in &act.outliers {
                    let (og, ou) = if q2 {
                        (
                            q2tp_outlier(gv_view.nib, r, gpr, j, &gsc),
                            q2tp_outlier(uv_view.nib, r, gpr, j, &usc),
                        )
                    } else {
                        (
                            q4tp_outlier(gv_view.nib, r, gpr, j, &gsc),
                            q4tp_outlier(uv_view.nib, r, gpr, j, &usc),
                        )
                    };
                    gv += og.0 * og.1 * xv;
                    uv += ou.0 * ou.1 * xv;
                }
                let silu_g = gv / (1.0 + (-gv).exp());
                // SAFETY: one worker owns each (expert, row) pair.
                unsafe { *ptrs[e].at(r) = silu_g * uv };
            }
        };
        dispatch_rows(pool, pairs.len() * inter, &run);
        true
    }

    /// Every routed expert's down projection, weighted and summed into
    /// `out`, under ONE pool dispatch.
    ///
    /// Partitioned by OUTPUT row rather than by expert: each row is owned
    /// by a single worker, so the experts are summed in the caller's order
    /// — the same sequence of f32 adds the serial `out[i] += w·eo[i]` loop
    /// performs, hence bit-identical. Partitioning by expert instead would
    /// race on the shared accumulator.
    pub fn moe_down_many(
        downs: &[&QTensor],
        gs: &[Vec<f32>],
        weights: &[f32],
        out: &mut [f32],
        pool: Option<&Pool>,
    ) -> bool {
        if downs.is_empty()
            || downs.len() != gs.len()
            || downs.len() != weights.len()
            || !a8w8_enabled()
        {
            return false;
        }
        let rows = out.len();
        let cols = downs[0].cols();
        if cols % GROUP_SIZE != 0 {
            return false;
        }
        let gpr = cols / GROUP_SIZE;
        let mut views = Vec::with_capacity(downs.len());
        for (d, g) in downs.iter().zip(gs.iter()) {
            if !matches!(
                d,
                Self::Mapped {
                    dtype: TensorDtype::Q4TiledP,
                    ..
                }
            ) || d.rows() != rows
                || d.cols() != cols
                || g.len() != cols
            {
                return false;
            }
            views.push(Q4tpView::new(d.quant_bytes(), rows, cols));
        }
        // One int8 split per expert — the activation vectors differ.
        let acts: Vec<SplitAct> = gs.iter().map(|g| split_act(g)).collect();
        // Partitioned by OUTPUT row, with the experts folded inside: each
        // row is owned by one worker, so they are summed in the caller's
        // order — the same f32 sequence the serial `out[i] += w·eo[i]`
        // loop produces. Partitioning by expert instead would either race
        // on the accumulator or need a scratch plane and a second pass;
        // measured, that variant was a wash, so this keeps the simpler
        // shape.
        let out_addr = SendMut(out.as_mut_ptr());
        let (views, acts, weights) = (&views, &acts, &weights);
        let run = |start: usize, end: usize| {
            let mut sc = vec![0f32; gpr];
            for r in start..end {
                let mut acc = 0f32;
                for (e, v) in views.iter().enumerate() {
                    v.scales_into(r, gpr, &mut sc);
                    let a = &acts[e];
                    let mut d = dot_q4tp_row_i8(v.nib, r, gpr, &a.xq, &sc) * a.sx;
                    for &(j, xv) in &a.outliers {
                        let (w, s) = q4tp_outlier(v.nib, r, gpr, j, &sc);
                        d += w * s * xv;
                    }
                    acc += weights[e] * d;
                }
                // SAFETY: disjoint row ranges per worker.
                unsafe { *out_addr.at(r) = acc };
            }
        };
        dispatch_rows(pool, rows, &run);
        true
    }
}

/// Batched q8 kernel: same math as qmatvec, the row makes a single
/// pass from memory for the whole batch.
/// Accelerate CBLAS — the Apple AMX matrix units, the same engine
/// llama.cpp's `-ngl 0` prefill rides via ggml-blas.
#[cfg(target_os = "macos")]
mod accel_blas {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn accel_gemm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_ACCEL").map(|v| v != "0").unwrap_or(true))
}

/// Off macOS the "accel" GEMM is the portable NEON micro-kernel below —
/// same entry point, so the batched-attention path opens on mobile.
#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
pub(crate) fn accel_gemm_enabled() -> bool {
    true
}

/// Portable NEON f32 GEMM (row-major, optional Bᵀ): a 4×8 fmla
/// micro-kernel with A broadcast against B panels — the mobile stand-in
/// for Accelerate in the batched causal attention (QKᵀ and P·V). Not a
/// BLAS: shapes here are the attention panels (m ≤ heads·chunk,
/// k = head_dim or context), and the goal is removing the per-position
/// quadratic wall, not peak GEMM.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn neon_gemm_rm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b_mat: &[f32],
    ldb: usize,
    b_rows_are_n: bool,
    c: &mut [f32],
    ldc: usize,
) {
    debug_assert!(a.len() >= (m - 1) * lda + k);
    debug_assert!(c.len() >= (m - 1) * ldc + n);
    // SAFETY: bounds asserted above; NEON is baseline on aarch64.
    unsafe {
        use core::arch::aarch64::*;
        let mut i = 0usize;
        while i < m {
            let mi = (m - i).min(4);
            let mut j = 0usize;
            while j < n {
                let nj = (n - j).min(8);
                if mi == 4 && nj == 8 {
                    let (mut c0a, mut c0b) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
                    let (mut c1a, mut c1b) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
                    let (mut c2a, mut c2b) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
                    let (mut c3a, mut c3b) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
                    for p in 0..k {
                        let (b0, b1) = if b_rows_are_n {
                            // B is [n, k]: column p of Bᵀ = element p of
                            // eight consecutive B rows — gathered.
                            let base = b_mat.as_ptr().add(j * ldb + p);
                            let g = |o: usize| *base.add(o * ldb);
                            ([g(0), g(1), g(2), g(3)], [g(4), g(5), g(6), g(7)])
                        } else {
                            let base = b_mat.as_ptr().add(p * ldb + j);
                            (
                                [*base, *base.add(1), *base.add(2), *base.add(3)],
                                [*base.add(4), *base.add(5), *base.add(6), *base.add(7)],
                            )
                        };
                        let bv0 = vld1q_f32(b0.as_ptr());
                        let bv1 = vld1q_f32(b1.as_ptr());
                        let a0 = vdupq_n_f32(*a.as_ptr().add(i * lda + p));
                        let a1 = vdupq_n_f32(*a.as_ptr().add((i + 1) * lda + p));
                        let a2 = vdupq_n_f32(*a.as_ptr().add((i + 2) * lda + p));
                        let a3 = vdupq_n_f32(*a.as_ptr().add((i + 3) * lda + p));
                        c0a = vfmaq_f32(c0a, a0, bv0);
                        c0b = vfmaq_f32(c0b, a0, bv1);
                        c1a = vfmaq_f32(c1a, a1, bv0);
                        c1b = vfmaq_f32(c1b, a1, bv1);
                        c2a = vfmaq_f32(c2a, a2, bv0);
                        c2b = vfmaq_f32(c2b, a2, bv1);
                        c3a = vfmaq_f32(c3a, a3, bv0);
                        c3b = vfmaq_f32(c3b, a3, bv1);
                    }
                    let al = vdupq_n_f32(alpha);
                    for (r, (ca, cb)) in [(c0a, c0b), (c1a, c1b), (c2a, c2b), (c3a, c3b)]
                        .iter()
                        .enumerate()
                    {
                        let dst = c.as_mut_ptr().add((i + r) * ldc + j);
                        vst1q_f32(dst, vmulq_f32(*ca, al));
                        vst1q_f32(dst.add(4), vmulq_f32(*cb, al));
                    }
                } else {
                    for r in 0..mi {
                        for q in 0..nj {
                            let mut acc = 0f32;
                            for p in 0..k {
                                let bv = if b_rows_are_n {
                                    b_mat[(j + q) * ldb + p]
                                } else {
                                    b_mat[p * ldb + j + q]
                                };
                                acc += a[(i + r) * lda + p] * bv;
                            }
                            c[(i + r) * ldc + j + q] = acc * alpha;
                        }
                    }
                }
                j += nj;
            }
            i += mi;
        }
    }
}

/// Off-macOS aarch64: the batched attention rides the NEON micro-GEMM.
#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm_rm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b_mat: &[f32],
    ldb: usize,
    b_rows_are_n: bool,
    c: &mut [f32],
    ldc: usize,
) {
    neon_gemm_rm(m, n, k, alpha, a, lda, b_mat, ldb, b_rows_are_n, c, ldc);
}

/// Row-major f32 GEMM, exposed for offline tools (the AWNP pass builds a
/// per-layer projection and applies it to every expert; a naive triple loop
/// would turn a two-minute job into half an hour).
#[allow(clippy::too_many_arguments)]
pub fn sgemm_public(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b_mat: &[f32],
    ldb: usize,
    b_rows_are_n: bool,
    c: &mut [f32],
    ldc: usize,
) {
    #[cfg(any(target_os = "macos", target_arch = "aarch64"))]
    {
        sgemm_rm(m, n, k, alpha, a, lda, b_mat, ldb, b_rows_are_n, c, ldc);
    }
    // x86 without Accelerate has no sgemm_rm: the specialized paths there are
    // quantized kernels, not an f32 GEMM. Only the offline AWNP pass reaches
    // this, so correctness matters and throughput does not — a triple loop is
    // the honest fallback rather than a reason to make the tool macOS-only.
    #[cfg(not(any(target_os = "macos", target_arch = "aarch64")))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    let bv = if b_rows_are_n {
                        b_mat[j * ldb + p]
                    } else {
                        b_mat[p * ldb + j]
                    };
                    acc += a[i * lda + p] * bv;
                }
                c[i * ldc + j] = alpha * acc;
            }
        }
    }
}

/// Row-major f32 GEMM on Accelerate: C[m,n] = alpha·A[m,k] × B(ᵀ).
/// `b_rows_are_n` = true multiplies by Bᵀ where B is stored [n, k].
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm_rm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b_mat: &[f32],
    ldb: usize,
    b_rows_are_n: bool,
    c: &mut [f32],
    ldc: usize,
) {
    debug_assert!(a.len() >= (m - 1) * lda + k);
    debug_assert!(c.len() >= (m - 1) * ldc + n);
    // Test hook: route the attention GEMMs through the portable NEON
    // micro-kernel ON APPLE SILICON — how the mobile batched attend is
    // measured without a phone in the loop. (Intel macOS has no NEON —
    // the hook is a no-op there, Accelerate continues below.)
    #[cfg(target_arch = "aarch64")]
    if std::env::var("CMF_FORCE_NEON_GEMM")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return neon_gemm_rm(m, n, k, alpha, a, lda, b_mat, ldb, b_rows_are_n, c, ldc);
    }
    unsafe {
        accel_blas::cblas_sgemm(
            101, // RowMajor
            111, // NoTrans A
            if b_rows_are_n { 112 } else { 111 },
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.as_ptr(),
            lda as i32,
            b_mat.as_ptr(),
            ldb as i32,
            0.0,
            c.as_mut_ptr(),
            ldc as i32,
        );
    }
}

/// Prefill GEMM through Accelerate (macOS): dequantize q8 rows into
/// f32 tiles (scale folded in, pool-parallel) and multiply each tile
/// on the AMX with one row-major sgemm. Tiles live in cache, weights
/// stream once. Numerics are f32-GEMM (not the int8 dot): prefill
/// logits shift within f32 rounding — tolerance-class, like every
/// reduction-order change; decode (M=1) never takes this path.
#[cfg(target_os = "macos")]
fn qmatmat_accel(
    q: &[u8],
    row_scale: &[f32],
    pre: &[std::borrow::Cow<'_, [f32]>],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    // NOTE: double-buffering the dequant against the sgemm (a scoped
    // thread driving the pool on tile k+1 while the caller multiplies
    // tile k) was tried and LOST ~6%: Accelerate's sgemm is itself
    // multithreaded, and the dequant workers just steal its cores.
    const TR: usize = 2048;
    let b = pre.len();
    thread_local! {
        static XPANEL: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        static WTILE: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    XPANEL.with(|xp| {
        WTILE.with(|wt| {
            let mut xpanel = xp.borrow_mut();
            xpanel.clear();
            for x in pre {
                xpanel.extend_from_slice(x);
            }
            let mut wtile = wt.borrow_mut();
            wtile.resize(TR * cols, 0.0);
            let mut r0 = 0usize;
            while r0 < rows {
                let tr = TR.min(rows - r0);
                // Dequant the tile (scale folded) — pool-parallel.
                let wt_addr = SendMut(wtile.as_mut_ptr());
                let run = |start: usize, end: usize| {
                    for r in start..end {
                        let row = &q[(r0 + r) * cols..(r0 + r + 1) * cols];
                        let s = row_scale[r0 + r];
                        // SAFETY: workers cover disjoint r ranges.
                        let dst =
                            unsafe { std::slice::from_raw_parts_mut(wt_addr.at(r * cols), cols) };
                        for (d, &v) in dst.iter_mut().zip(row) {
                            *d = (v as i8) as f32 * s;
                        }
                    }
                };
                dispatch_rows(pool, tr, &run);
                // C[b, tr] (at column r0 of out[b, rows]) = X · Wtileᵀ
                unsafe {
                    accel_blas::cblas_sgemm(
                        101, // RowMajor
                        111, // NoTrans A
                        112, // Trans B
                        b as i32,
                        tr as i32,
                        cols as i32,
                        1.0,
                        xpanel.as_ptr(),
                        cols as i32,
                        wtile.as_ptr(),
                        cols as i32,
                        0.0,
                        out.as_mut_ptr().add(r0),
                        rows as i32,
                    );
                }
                r0 += tr;
            }
        })
    });
}

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
    // Big prefill batches ride the AMX (roadmap PR3): the row×batch
    // SDOT loop below peaks near the CPU's dot throughput, an order
    // below the matrix units. Small tensors and tiny test models stay
    // on the exact integer path.
    #[cfg(target_os = "macos")]
    if b >= 8 && rows * cols >= 500_000 && accel_gemm_enabled() {
        qmatmat_accel(q, row_scale, pre, rows, cols, out, pool);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if sdot_enabled() {
        let acts: Vec<SplitAct> = pre.iter().map(|x| split_act(x)).collect();
        let out_addr = SendMut(out.as_mut_ptr());
        // Blocked 2×4 (mobile prefill: no AMX to fall back on — this
        // path IS the ARM prefill GEMM off Apple silicon).
        let blocked_ok = blocked_enabled();
        let use_i8mm = i8mm_enabled();
        if blocked_ok {
            let run = |start: usize, end: usize| {
                let mut o = start;
                while o < end {
                    if o + 2 <= end {
                        let r0 = &q[o * cols..(o + 1) * cols];
                        let r1 = &q[(o + 1) * cols..(o + 2) * cols];
                        let mut bi = 0usize;
                        while bi + 4 <= acts.len() {
                            let xs = [
                                acts[bi].xq.as_slice(),
                                acts[bi + 1].xq.as_slice(),
                                acts[bi + 2].xq.as_slice(),
                                acts[bi + 3].xq.as_slice(),
                            ];
                            let d = if use_i8mm {
                                unsafe { dot_i8_smmla_2x4(r0, r1, xs) }
                            } else {
                                unsafe { dot_i8_sdot_2x4(r0, r1, xs) }
                            };
                            for (r, row) in [r0, r1].into_iter().enumerate() {
                                for k in 0..4 {
                                    let act = &acts[bi + k];
                                    let mut v = d[r][k] as f32 * act.sx;
                                    for &(j, xv) in &act.outliers {
                                        v += (row[j] as i8) as f32 * xv;
                                    }
                                    unsafe {
                                        *out_addr.at((bi + k) * rows + o + r) = v * row_scale[o + r]
                                    };
                                }
                            }
                            bi += 4;
                        }
                        while bi < acts.len() {
                            for (r, row) in [r0, r1].into_iter().enumerate() {
                                let v = row_dot_sdot(row, &acts[bi]) * row_scale[o + r];
                                unsafe { *out_addr.at(bi * rows + o + r) = v };
                            }
                            bi += 1;
                        }
                        o += 2;
                    } else {
                        let row = &q[o * cols..(o + 1) * cols];
                        for (bi, act) in acts.iter().enumerate() {
                            let v = row_dot_sdot(row, act) * row_scale[o];
                            unsafe { *out_addr.at(bi * rows + o) = v };
                        }
                        o += 1;
                    }
                }
            };
            dispatch_rows(pool, rows, &run);
            return;
        }
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
    // x86 A8W8 batch. Non-VNNI parts take the BLOCKED 2×4 kernel
    // (roadmap P0: two weight rows' abs() stay in registers across four
    // activation streams); VNNI machines keep the per-row bias-trick
    // dot, which is already throughput-bound there.
    #[cfg(target_arch = "x86_64")]
    if avx2_a8w8_enabled() {
        let acts: Vec<SplitAct> = pre.iter().map(|x| split_act(x)).collect();
        let out_addr = SendMut(out.as_mut_ptr());
        // CMF_X86_BLOCKED=0 forces the per-row path (paired in-process
        // A/B on noisy shared-vCPU hosts).
        let blocked_ok = blocked_enabled();
        if !avx512vnni_enabled() && blocked_ok {
            let run = |start: usize, end: usize| {
                let mut o = start;
                while o < end {
                    if o + 2 <= end {
                        let r0 = &q[o * cols..(o + 1) * cols];
                        let r1 = &q[(o + 1) * cols..(o + 2) * cols];
                        let mut bi = 0usize;
                        while bi + 4 <= acts.len() {
                            let xs = [
                                acts[bi].xq.as_slice(),
                                acts[bi + 1].xq.as_slice(),
                                acts[bi + 2].xq.as_slice(),
                                acts[bi + 3].xq.as_slice(),
                            ];
                            let d = unsafe { dot_i8_i8_avx2_2x4(r0, r1, xs) };
                            for (r, row) in [r0, r1].into_iter().enumerate() {
                                for k in 0..4 {
                                    let act = &acts[bi + k];
                                    let mut v = d[r][k] as f32 * act.sx;
                                    for &(j, xv) in &act.outliers {
                                        v += (row[j] as i8) as f32 * xv;
                                    }
                                    unsafe {
                                        *out_addr.at((bi + k) * rows + o + r) = v * row_scale[o + r]
                                    };
                                }
                            }
                            bi += 4;
                        }
                        while bi < acts.len() {
                            for (r, row) in [r0, r1].into_iter().enumerate() {
                                let v = row_dot_avx2(row, &acts[bi]) * row_scale[o + r];
                                unsafe { *out_addr.at(bi * rows + o + r) = v };
                            }
                            bi += 1;
                        }
                        o += 2;
                    } else {
                        let row = &q[o * cols..(o + 1) * cols];
                        for (bi, act) in acts.iter().enumerate() {
                            let v = row_dot_avx2(row, act) * row_scale[o];
                            unsafe { *out_addr.at(bi * rows + o) = v };
                        }
                        o += 1;
                    }
                }
            };
            dispatch_rows(pool, rows, &run);
            return;
        }
        let run = |start: usize, end: usize| {
            for o in start..end {
                let row = &q[o * cols..(o + 1) * cols];
                for (bi, act) in acts.iter().enumerate() {
                    let v = row_dot_avx2(row, act) * row_scale[o];
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

/// SIMD unpack for the dominant vbit width B=4 (94% of rows on the
/// log2-shape calibration): 16 packed bytes -> 32 centered i8 values.
/// vbit packs MSB-first, so the HIGH nibble is the even element
/// (opposite of q4_block's lo-first interleave). Centering is u-7.
#[inline]
fn vbit_fill4(data: &[u8], buf: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return vbit_fill4_neon(data, buf);
    }
    #[cfg(target_arch = "x86_64")]
    if avx2_enabled() {
        return unsafe { vbit_fill4_avx2(data, buf) };
    }
    #[allow(unreachable_code)]
    for (blk, chunk) in buf.chunks_exact_mut(8).enumerate() {
        let u = unpack8::<4>(&data[blk * 4..]);
        for k in 0..8 {
            chunk[k] = (u[k] - 7) as i8 as u8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vbit_fill4_neon(data: &[u8], buf: &mut [u8]) {
    // SAFETY: buf.len() is a multiple of GROUP_SIZE=32; data holds
    // buf.len()/2 packed bytes (validated at load).
    unsafe {
        use core::arch::aarch64::*;
        let n = buf.len();
        let mask = vdupq_n_u8(0x0F);
        let seven = vdupq_n_s8(7);
        let mut g = 0usize;
        while g * 32 + 32 <= n {
            let b = vld1q_u8(data.as_ptr().add(g * 16));
            let hi = vshrq_n_u8::<4>(b);
            let lo = vandq_u8(b, mask);
            let z0 = vsubq_s8(vreinterpretq_s8_u8(vzip1q_u8(hi, lo)), seven);
            let z1 = vsubq_s8(vreinterpretq_s8_u8(vzip2q_u8(hi, lo)), seven);
            vst1q_u8(buf.as_mut_ptr().add(g * 32), vreinterpretq_u8_s8(z0));
            vst1q_u8(buf.as_mut_ptr().add(g * 32 + 16), vreinterpretq_u8_s8(z1));
            g += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vbit_fill4_avx2(data: &[u8], buf: &mut [u8]) {
    // SAFETY: see vbit_fill4_neon.
    unsafe {
        use core::arch::x86_64::*;
        let n = buf.len();
        let mask = _mm_set1_epi8(0x0F);
        let seven = _mm256_set1_epi8(7);
        let mut g = 0usize;
        while g * 32 + 32 <= n {
            let b = _mm_loadu_si128(data.as_ptr().add(g * 16) as *const __m128i);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(b), mask);
            let lo = _mm_and_si128(b, mask);
            let z = _mm256_sub_epi8(
                _mm256_set_m128i(_mm_unpackhi_epi8(hi, lo), _mm_unpacklo_epi8(hi, lo)),
                seven,
            );
            _mm256_storeu_si256(buf.as_mut_ptr().add(g * 32) as *mut __m256i, z);
            g += 1;
        }
    }
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

    // SDOT path: unpack the row to centered i8 once, then per-group
    // int8 dot against the quantized activations — same A8W8 contract
    // as q8 (bounded noise; CMF_SDOT=0 keeps the exact scalar path).
    if a8w8_enabled() {
        let act = split_act(x);
        let out_addr = SendMut(out.as_mut_ptr());
        let run = move |start: usize, end: usize| {
            vbit_range_a8w8(bytes, offsets, x, &act, rows, cols, out_addr, start, end)
        };
        dispatch_rows(pool, rows, &run);
        return;
    }

    let out_addr = SendMut(out.as_mut_ptr());
    let run = move |start: usize, end: usize| {
        vbit_range_f32(bytes, offsets, x, rows, cols, out_addr, start, end)
    };
    dispatch_rows(pool, rows, &run);
}

/// One vbit row range via the A8W8 int8 path — kernel body of
/// `vbitmatvec`, extracted so multi-matrix jobs can drive it for
/// several tensors in one dispatch (b=8 rows go exact f32).
#[allow(clippy::too_many_arguments)]
fn vbit_range_a8w8(
    bytes: &[u8],
    offsets: &[usize],
    x: &[f32],
    act: &SplitAct,
    rows: usize,
    cols: usize,
    out: SendMut,
    start: usize,
    end: usize,
) {
    let ng = cols / GROUP_SIZE;
    let bits = &bytes[..rows];
    let sc_off = rows;
    let row_dot = |r: usize| -> f32 {
        let b = bits[r] as usize;
        let l = (1i32 << (b - 1)) - 1;
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
                4 => vbit_fill4(data, &mut buf),
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
                let d = dot_i8_i8(
                    &buf[g * GROUP_SIZE..(g + 1) * GROUP_SIZE],
                    &act.xq[g * GROUP_SIZE..(g + 1) * GROUP_SIZE],
                ) as f32
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
    for r in start..end {
        // SAFETY: disjoint row ranges per worker.
        unsafe { *out.at(r) = row_dot(r) };
    }
}

/// Exact scalar vbit row range (same extraction, non-SDOT path).
#[allow(clippy::too_many_arguments)]
fn vbit_range_f32(
    bytes: &[u8],
    offsets: &[usize],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: SendMut,
    start: usize,
    end: usize,
) {
    let ng = cols / GROUP_SIZE;
    let bits = &bytes[..rows];
    let sc_off = rows;
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
            let s = f16_to_f32(u16::from_le_bytes([
                bytes[sc_off + so],
                bytes[sc_off + so + 1],
            ]));
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
    for r in start..end {
        let data = &bytes[offsets[r]..offsets[r + 1]];
        let v = match bits[r] {
            3 => dot_row::<3>(data, bytes, sc_off, r, ng, x),
            4 => dot_row::<4>(data, bytes, sc_off, r, ng, x),
            5 => dot_row::<5>(data, bytes, sc_off, r, ng, x),
            6 => dot_row::<6>(data, bytes, sc_off, r, ng, x),
            8 => dot_row::<8>(data, bytes, sc_off, r, ng, x),
            b => unreachable!("vbit bit-width {b} (validated at load)"),
        };
        // SAFETY: disjoint row ranges per worker.
        unsafe { *out.at(r) = v };
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

    if a8w8_enabled() {
        let a1 = split_act(x1);
        let a2 = split_act(x2);
        let p1 = SendMut(o1.as_mut_ptr());
        let p2 = SendMut(o2.as_mut_ptr());
        let run = move |start: usize, end: usize| {
            vbit_range2_a8w8(
                bytes, offsets, x1, x2, &a1, &a2, rows, cols, p1, p2, start, end,
            )
        };
        dispatch_rows(pool, rows, &run);
        return;
    }

    let p1 = SendMut(o1.as_mut_ptr());
    let p2 = SendMut(o2.as_mut_ptr());
    let run = move |start: usize, end: usize| {
        vbit_range2_f32(bytes, offsets, x1, x2, rows, cols, p1, p2, start, end)
    };
    dispatch_rows(pool, rows, &run);
}

/// Two-input vbit row range via the A8W8 int8 path — kernel body of
/// `vbitmatvec2`, extracted for pair multi-matrix jobs (b=8 rows go
/// exact f32 for both lanes, bits streamed once).
#[allow(clippy::too_many_arguments)]
fn vbit_range2_a8w8(
    bytes: &[u8],
    offsets: &[usize],
    x1: &[f32],
    x2: &[f32],
    a1: &SplitAct,
    a2: &SplitAct,
    rows: usize,
    cols: usize,
    p1: SendMut,
    p2: SendMut,
    start: usize,
    end: usize,
) {
    let ng = cols / GROUP_SIZE;
    let bits = &bytes[..rows];
    let sc_off = rows;
    let row_dots = |r: usize| -> (f32, f32) {
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
                4 => vbit_fill4(data, &mut buf),
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
                let v1 = dot_i8_i8(wg, &a1.xq[g * GROUP_SIZE..(g + 1) * GROUP_SIZE]) as f32 * a1.sx;
                let v2 = dot_i8_i8(wg, &a2.xq[g * GROUP_SIZE..(g + 1) * GROUP_SIZE]) as f32 * a2.sx;
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
    for r in start..end {
        let (v1, v2) = row_dots(r);
        // SAFETY: disjoint row ranges per worker.
        unsafe {
            *p1.at(r) = v1;
            *p2.at(r) = v2;
        }
    }
}

/// Two-input exact scalar vbit row range (same extraction) —
/// per-bit-width specialized, two accumulators per row; per-lane
/// accumulation order matches `vbitmatvec` exactly.
#[allow(clippy::too_many_arguments)]
fn vbit_range2_f32(
    bytes: &[u8],
    offsets: &[usize],
    x1: &[f32],
    x2: &[f32],
    rows: usize,
    cols: usize,
    p1: SendMut,
    p2: SendMut,
    start: usize,
    end: usize,
) {
    let ng = cols / GROUP_SIZE;
    let bits = &bytes[..rows];
    let sc_off = rows;
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
            let s = f16_to_f32(u16::from_le_bytes([
                bytes[sc_off + so],
                bytes[sc_off + so + 1],
            ]));
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
    for r in start..end {
        let data = &bytes[offsets[r]..offsets[r + 1]];
        let (v1, v2) = match bits[r] {
            3 => dot_row2::<3>(data, bytes, sc_off, r, ng, x1, x2),
            4 => dot_row2::<4>(data, bytes, sc_off, r, ng, x1, x2),
            5 => dot_row2::<5>(data, bytes, sc_off, r, ng, x1, x2),
            6 => dot_row2::<6>(data, bytes, sc_off, r, ng, x1, x2),
            8 => dot_row2::<8>(data, bytes, sc_off, r, ng, x1, x2),
            b => unreachable!("vbit bit-width {b} (validated at load)"),
        };
        // SAFETY: disjoint row ranges per worker.
        unsafe {
            *p1.at(r) = v1;
            *p2.at(r) = v2;
        }
    }
}

// ───────────────────── q4_tiled kernels (§4.3) ─────────────────────

/// One q4_tiled row dot on the A8W8 int8 path: per 32-group the tile
/// is ONE sequential read — [f16 scale][16B nibbles] — versus the two
/// distant streams of the split layout. Values/order identical to the
/// split kernels.
#[inline]
#[allow(unreachable_code)]
fn dot_q4t_row_i8(bytes: &[u8], r: usize, gpr: usize, xq: &[i8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_q4t_row_sdot(bytes, r, gpr, xq);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if vnni_tiles_enabled() {
            return dot_q4t_row_vnni(bytes, r, gpr, xq);
        }
        return dot_q4t_row_avx2(bytes, r, gpr, xq);
    }
    let mut acc = 0f32;
    for gi in 0..gpr {
        let tile = &bytes[(r * gpr + gi) * Q4_TILE..(r * gpr + gi + 1) * Q4_TILE];
        let s = f16_to_f32(u16::from_le_bytes([tile[0], tile[1]]));
        let mut d = 0i32;
        for (k, &b) in tile[2..].iter().enumerate() {
            d += ((b & 0x0F) as i32 - 8) * xq[gi * GROUP_SIZE + k * 2] as i32
                + (((b >> 4) & 0x0F) as i32 - 8) * xq[gi * GROUP_SIZE + k * 2 + 1] as i32;
        }
        acc += d as f32 * s;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q4t_row_sdot(bytes: &[u8], r: usize, gpr: usize, xq: &[i8]) -> f32 {
    // SAFETY: callers uphold slice-length contracts (18B tile per group,
    // xq.len() == gpr·GROUP_SIZE).
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let lomask = vdupq_n_u8(0x0F);
        let eight = vdupq_n_s8(8);
        let mut acc = 0f32;
        for gi in 0..gpr {
            let t = bytes.as_ptr().add((r * gpr + gi) * Q4_TILE);
            let s = f16_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
            let b = vld1q_u8(t.add(2));
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4t_row_avx2(bytes: &[u8], r: usize, gpr: usize, xq: &[i8]) -> f32 {
    // SAFETY: see dot_q4t_row_sdot.
    unsafe {
        use core::arch::x86_64::*;
        let lomask = _mm_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let ones = _mm256_set1_epi16(1);
        let mut acc = 0f32;
        for gi in 0..gpr {
            let t = bytes.as_ptr().add((r * gpr + gi) * Q4_TILE);
            let s = f16_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
            let b = _mm_loadu_si128(t.add(2) as *const __m128i);
            let lo = _mm_and_si128(b, lomask);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(b), lomask);
            let w = _mm256_sub_epi8(
                _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)),
                eight,
            );
            let x = _mm256_loadu_si256(xq.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
            let d = _mm256_madd_epi16(p16, ones);
            let hi128 = _mm256_extracti128_si256::<1>(d);
            let s128 = _mm_add_epi32(_mm256_castsi256_si128(d), hi128);
            let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
            let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
            acc += _mm_cvtsi128_si32(s32) as f32 * s;
        }
        acc
    }
}

/// VNNI twin of `dot_q4t_row_avx2`: same unpack, `vpdpbusd` replaces
/// the maddubs+madd pair (see `dpbusd_hsum` — sums are bit-identical).
/// 256-bit VL encoding, so the VEX `vpsignb` stays usable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn dot_q4t_row_vnni(bytes: &[u8], r: usize, gpr: usize, xq: &[i8]) -> f32 {
    // SAFETY: see dot_q4t_row_sdot.
    unsafe {
        use core::arch::x86_64::*;
        let lomask = _mm_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let mut acc = 0f32;
        for gi in 0..gpr {
            let t = bytes.as_ptr().add((r * gpr + gi) * Q4_TILE);
            let s = f16_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
            let b = _mm_loadu_si128(t.add(2) as *const __m128i);
            let lo = _mm_and_si128(b, lomask);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(b), lomask);
            let w = _mm256_sub_epi8(
                _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)),
                eight,
            );
            let x = _mm256_loadu_si256(xq.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
            let d = dpbusd_hsum(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
            acc += d as f32 * s;
        }
        acc
    }
}

/// One q4_tiled row against FOUR activation streams: the nibble unpack
/// and abs() happen once per group instead of once per (group,
/// activation) — the unpack is the dominant per-element cost of the
/// tiled format (roadmap P0 portable blocking, q4t leg).
#[cfg(target_arch = "x86_64")]
// `fma` is NOT implied by `avx2`: without it LLVM lowers _mm256_fmadd_ps
// to a libm call per lane — measured 2x slower than the reduction this
// kernel replaces. The runtime gate (`avx2_enabled`) already requires
// both features, so declaring it here is safe.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4t_row_1x4_avx2(bytes: &[u8], r: usize, gpr: usize, xs: [&[i8]; 4]) -> [f32; 4] {
    // SAFETY: callers uphold the 18B-tile and xq-length contracts.
    unsafe {
        use core::arch::x86_64::*;
        let lomask = _mm_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let ones = _mm256_set1_epi16(1);
        // One f32 accumulator VECTOR per activation, reduced once at the
        // end. Folding each group's i32 lanes to a scalar inside the loop
        // costs an extracti128 + three shift/add + a movd — a cross-lane
        // dependency chain per (group, activation), 288 of them per row at
        // cols=2304. The per-group scale is what forces a float
        // accumulator; it does not force a horizontal sum.
        //
        // The four accumulators are NAMED, not an array: as `[__m256; 4]`
        // indexed by a loop variable LLVM keeps them in memory and every
        // group pays four 32-byte loads and stores. That alone made this
        // kernel 2x SLOWER than the per-group reduction it replaces
        // (measured on the EPYC box: 150 s vs 71 s for two 256² steps).
        let mut f0 = _mm256_setzero_ps();
        let mut f1 = _mm256_setzero_ps();
        let mut f2 = _mm256_setzero_ps();
        let mut f3 = _mm256_setzero_ps();
        for gi in 0..gpr {
            let t = bytes.as_ptr().add((r * gpr + gi) * Q4_TILE);
            let s = f16_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
            let sv = _mm256_set1_ps(s);
            let bb = _mm_loadu_si128(t.add(2) as *const __m128i);
            let lo = _mm_and_si128(bb, lomask);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(bb), lomask);
            let w = _mm256_sub_epi8(
                _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)),
                eight,
            );
            let aw = _mm256_abs_epi8(w);
            let off = gi * GROUP_SIZE;
            let dot = |xq: &[i8]| {
                let x = _mm256_loadu_si256(xq.as_ptr().add(off) as *const __m256i);
                let p16 = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w));
                _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones))
            };
            f0 = _mm256_fmadd_ps(dot(xs[0]), sv, f0);
            f1 = _mm256_fmadd_ps(dot(xs[1]), sv, f1);
            f2 = _mm256_fmadd_ps(dot(xs[2]), sv, f2);
            f3 = _mm256_fmadd_ps(dot(xs[3]), sv, f3);
        }
        [
            hsum256_ps(f0),
            hsum256_ps(f1),
            hsum256_ps(f2),
            hsum256_ps(f3),
        ]
    }
}

/// Horizontal sum of eight f32 lanes — the one cross-lane reduction the
/// blocked kernels pay, once per row instead of once per group.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: core::arch::x86_64::__m256) -> f32 {
    // SAFETY: pure register arithmetic on the caller's vector.
    unsafe {
        use core::arch::x86_64::*;
        let hi = _mm256_extractf128_ps::<1>(v);
        let s = _mm_add_ps(_mm256_castps256_ps128(v), hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
        _mm_cvtss_f32(s)
    }
}

/// VNNI twin of `dot_q4t_row_1x4_avx2` (see `dpbusd_hsum`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn dot_q4t_row_1x4_vnni(bytes: &[u8], r: usize, gpr: usize, xs: [&[i8]; 4]) -> [f32; 4] {
    // SAFETY: callers uphold the 18B-tile and xq-length contracts.
    unsafe {
        use core::arch::x86_64::*;
        let lomask = _mm_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        // Same shape as the AVX2 twin: accumulate in f32 vectors and pay
        // one cross-lane reduction per row, not per (group, activation).
        let mut f0 = _mm256_setzero_ps();
        let mut f1 = _mm256_setzero_ps();
        let mut f2 = _mm256_setzero_ps();
        let mut f3 = _mm256_setzero_ps();
        for gi in 0..gpr {
            let t = bytes.as_ptr().add((r * gpr + gi) * Q4_TILE);
            let s = f16_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
            let sv = _mm256_set1_ps(s);
            let bb = _mm_loadu_si128(t.add(2) as *const __m128i);
            let lo = _mm_and_si128(bb, lomask);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(bb), lomask);
            let w = _mm256_sub_epi8(
                _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)),
                eight,
            );
            let aw = _mm256_abs_epi8(w);
            let off = gi * GROUP_SIZE;
            let dot = |xq: &[i8]| {
                let x = _mm256_loadu_si256(xq.as_ptr().add(off) as *const __m256i);
                _mm256_cvtepi32_ps(_mm256_dpbusd_epi32(
                    _mm256_setzero_si256(),
                    aw,
                    _mm256_sign_epi8(x, w),
                ))
            };
            f0 = _mm256_fmadd_ps(dot(xs[0]), sv, f0);
            f1 = _mm256_fmadd_ps(dot(xs[1]), sv, f1);
            f2 = _mm256_fmadd_ps(dot(xs[2]), sv, f2);
            f3 = _mm256_fmadd_ps(dot(xs[3]), sv, f3);
        }
        let acc = [
            hsum256_ps(f0),
            hsum256_ps(f1),
            hsum256_ps(f2),
            hsum256_ps(f3),
        ];
        acc
    }
}

/// ARM twin of `dot_q4t_row_1x4_avx2`: one nibble unpack per group
/// serves FOUR activation streams. Per stream the group order and f32
/// accumulation match `dot_q4t_row_sdot` exactly — batch == matvec
/// bit-for-bit.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q4t_row_1x4_sdot(bytes: &[u8], r: usize, gpr: usize, xs: [&[i8]; 4]) -> [f32; 4] {
    // SAFETY: callers uphold the 18B-tile and xq-length contracts.
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let lomask = vdupq_n_u8(0x0F);
        let eight = vdupq_n_s8(8);
        let mut acc = [0f32; 4];
        for gi in 0..gpr {
            let t = bytes.as_ptr().add((r * gpr + gi) * Q4_TILE);
            let s = f16_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
            let b = vld1q_u8(t.add(2));
            let lo = vandq_u8(b, lomask);
            let hi = vshrq_n_u8::<4>(b);
            let e0 = vsubq_s8(vreinterpretq_s8_u8(vzip1q_u8(lo, hi)), eight);
            let e1 = vsubq_s8(vreinterpretq_s8_u8(vzip2q_u8(lo, hi)), eight);
            for (k, xq) in xs.iter().enumerate() {
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
                acc[k] += vaddvq_s32(vaddq_s32(a0, a1)) as f32 * s;
            }
        }
        acc
    }
}

/// Exact-term correction for A8W8 outliers on a tiled row.
#[inline]
fn q4t_outlier(bytes: &[u8], r: usize, gpr: usize, j: usize) -> (f32, f32) {
    let gi = j / GROUP_SIZE;
    let k = j % GROUP_SIZE;
    let tile = &bytes[(r * gpr + gi) * Q4_TILE..(r * gpr + gi + 1) * Q4_TILE];
    let s = f16_to_f32(u16::from_le_bytes([tile[0], tile[1]]));
    let byte = tile[2 + k / 2];
    let nib = if k & 1 == 0 { byte & 0x0F } else { byte >> 4 };
    ((nib as i32 - 8) as f32, s)
}

/// Exact scalar q4_tiled row (CMF_SDOT=0 contract) — same pairwise
/// accumulation shape as `q4_range_f32`.
#[inline]
fn q4t_row_exact(bytes: &[u8], r: usize, gpr: usize, x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for gi in 0..gpr {
        let tile = &bytes[(r * gpr + gi) * Q4_TILE..(r * gpr + gi + 1) * Q4_TILE];
        let s = f16_to_f32(u16::from_le_bytes([tile[0], tile[1]]));
        let xg = &x[gi * GROUP_SIZE..(gi + 1) * GROUP_SIZE];
        let mut ga = 0f32;
        for (k, &b) in tile[2..].iter().enumerate() {
            ga += ((b & 0x0F) as f32 - 8.0) * xg[k * 2]
                + (((b >> 4) & 0x0F) as f32 - 8.0) * xg[k * 2 + 1];
        }
        acc += ga * s;
    }
    acc
}

/// Split view of a `q4tp` payload. The three planes are resolved once per
/// matvec instead of per row — `q4tp_sections` is cheap, but doing it inside
/// the row loop would put a division on the hot path for nothing.
struct Q4tpView<'a> {
    nib: &'a [u8],
    params: &'a [u8],
    codes: &'a [u8],
    stride: usize,
    /// q2tp reads the ladder with rung 0 = exact zero.
    zero_rung: bool,
}

impl<'a> Q4tpView<'a> {
    fn new(bytes: &'a [u8], rows: usize, cols: usize) -> Self {
        let (params_off, codes_off, stride) = q4tp_sections(rows, cols);
        Self {
            nib: &bytes[..params_off],
            params: &bytes[params_off..codes_off],
            codes: &bytes[codes_off..],
            stride,
            zero_rung: false,
        }
    }

    /// The q2tp view: identical params/codes planes, 8 B weight chunks.
    fn new_q2(bytes: &'a [u8], rows: usize, cols: usize) -> Self {
        let (params_off, codes_off, stride) = q2tp_sections(rows, cols);
        Self {
            nib: &bytes[..params_off],
            params: &bytes[params_off..codes_off],
            codes: &bytes[codes_off..],
            stride,
            zero_rung: true,
        }
    }

    /// Expand row `r`'s per-tile scales into `out` (length `gpr`).
    ///
    /// Doing this once per row — rather than decoding a 5-bit code inside the
    /// tile loop — is what makes the format free at runtime. Random access to
    /// a packed 5-bit field costs a division, two bounds checks and a branch;
    /// the tile's actual work is two `sdot`s, so per-tile decoding dominated
    /// the kernel and cost 5x (measured: 1.4 vs 6.9 tok/s on Nanbeige-3B).
    /// Walking the plane sequentially with a bit accumulator is ~3 ops.
    /// Eight 5-bit codes are exactly five bytes, so a whole group of
    /// eight decodes from one little-endian word at fixed shifts. The
    /// bit-accumulator this replaces carried a data-dependent `while
    /// have < 5` refill whose branch sat in the innermost loop of every
    /// q4tp row; a decode profile put this function above the dot
    /// products it feeds. Same bitstream, same codes — just no branch
    /// and eight independent extractions.
    #[inline]
    fn scales_into(&self, r: usize, gpr: usize, out: &mut [f32]) {
        let tab = if self.zero_rung {
            q2tp_ladder(self.params, r)
        } else {
            q4tp_ladder(self.params, r)
        };
        let codes = &self.codes[r * self.stride..(r + 1) * self.stride];
        let out = &mut out[..gpr];
        let mut chunks = out.chunks_exact_mut(8);
        let mut ci = 0usize;
        for c in &mut chunks {
            let w = u64::from(codes[ci])
                | u64::from(codes[ci + 1]) << 8
                | u64::from(codes[ci + 2]) << 16
                | u64::from(codes[ci + 3]) << 24
                | u64::from(codes[ci + 4]) << 32;
            for (k, o) in c.iter_mut().enumerate() {
                *o = tab[((w >> (5 * k)) & 31) as usize];
            }
            ci += 5;
        }
        // Fewer than eight codes left: the shared total accessor, which
        // tolerates a 5-bit field whose spill byte is past the stride.
        let tail = &codes[ci..];
        for (k, o) in chunks.into_remainder().iter_mut().enumerate() {
            *o = tab[q4tp_code(tail, k)];
        }
    }
}

#[inline]
fn dot_q4tp_row_i8(nib: &[u8], r: usize, gpr: usize, xq: &[i8], scales: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_q4tp_row_sdot(nib, r, gpr, xq, scales);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if vnni_tiles_enabled() {
            return dot_q4tp_row_vnni(nib, r, gpr, xq, scales);
        }
        return dot_q4tp_row_avx2(nib, r, gpr, xq, scales);
    }
    #[allow(unreachable_code)]
    {
        let mut acc = 0f32;
        for gi in 0..gpr {
            let tile = &nib[(r * gpr + gi) * Q4TP_NIB..(r * gpr + gi + 1) * Q4TP_NIB];
            let s = scales[gi];
            let mut d = 0i32;
            for (k, &b) in tile.iter().enumerate() {
                d += ((b & 0x0F) as i32 - 8) * xq[gi * GROUP_SIZE + k * 2] as i32
                    + (((b >> 4) & 0x0F) as i32 - 8) * xq[gi * GROUP_SIZE + k * 2 + 1] as i32;
            }
            acc += d as f32 * s;
        }
        acc
    }
}

/// q4tp twin of `dot_q4t_row_sdot`: identical nibble math, but the tile
/// stride is 16 B (no inline scale) and the scale is a ladder lookup.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q4tp_row_sdot(nib: &[u8], r: usize, gpr: usize, xq: &[i8], scales: &[f32]) -> f32 {
    // SAFETY: callers uphold slice-length contracts (16B tile per group,
    // xq.len() == gpr·GROUP_SIZE, codes covering gpr 5-bit fields).
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let lomask = vdupq_n_u8(0x0F);
        let eight = vdupq_n_s8(8);
        let mut acc = 0f32;
        for gi in 0..gpr {
            let t = nib.as_ptr().add((r * gpr + gi) * Q4TP_NIB);
            let s = *scales.get_unchecked(gi);
            let b = vld1q_u8(t);
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4tp_row_avx2(nib: &[u8], r: usize, gpr: usize, xq: &[i8], scales: &[f32]) -> f32 {
    // SAFETY: see dot_q4tp_row_sdot.
    unsafe {
        use core::arch::x86_64::*;
        let lomask = _mm_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let ones = _mm256_set1_epi16(1);
        let mut acc = 0f32;
        for gi in 0..gpr {
            let t = nib.as_ptr().add((r * gpr + gi) * Q4TP_NIB);
            let s = *scales.get_unchecked(gi);
            let b = _mm_loadu_si128(t as *const __m128i);
            let lo = _mm_and_si128(b, lomask);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(b), lomask);
            let w = _mm256_sub_epi8(
                _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)),
                eight,
            );
            let x = _mm256_loadu_si256(xq.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
            let d = _mm256_madd_epi16(p16, ones);
            let hi128 = _mm256_extracti128_si256::<1>(d);
            let s128 = _mm_add_epi32(_mm256_castsi256_si128(d), hi128);
            let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
            let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
            acc += _mm_cvtsi128_si32(s32) as f32 * s;
        }
        acc
    }
}

/// VNNI twin of `dot_q4tp_row_avx2` (see `dot_q4t_row_vnni` for why the
/// 256-bit VL encoding is the one to use here).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn dot_q4tp_row_vnni(nib: &[u8], r: usize, gpr: usize, xq: &[i8], scales: &[f32]) -> f32 {
    // SAFETY: see dot_q4tp_row_sdot.
    unsafe {
        use core::arch::x86_64::*;
        let lomask = _mm_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let mut acc = 0f32;
        for gi in 0..gpr {
            let t = nib.as_ptr().add((r * gpr + gi) * Q4TP_NIB);
            let s = *scales.get_unchecked(gi);
            let b = _mm_loadu_si128(t as *const __m128i);
            let lo = _mm_and_si128(b, lomask);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(b), lomask);
            let w = _mm256_sub_epi8(
                _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)),
                eight,
            );
            let x = _mm256_loadu_si256(xq.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
            acc += dpbusd_hsum(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w)) as f32 * s;
        }
        acc
    }
}

/// Exact scalar q4tp row — the `CMF_SDOT=0` contract, same pairwise
/// accumulation shape as `q4t_row_exact`.
#[inline]
fn q4tp_row_exact(nib: &[u8], r: usize, gpr: usize, x: &[f32], scales: &[f32]) -> f32 {
    let mut acc = 0f32;
    for gi in 0..gpr {
        let tile = &nib[(r * gpr + gi) * Q4TP_NIB..(r * gpr + gi + 1) * Q4TP_NIB];
        let s = scales[gi];
        let xg = &x[gi * GROUP_SIZE..(gi + 1) * GROUP_SIZE];
        let mut ga = 0f32;
        for (k, &b) in tile.iter().enumerate() {
            ga += ((b & 0x0F) as f32 - 8.0) * xg[k * 2]
                + (((b >> 4) & 0x0F) as f32 - 8.0) * xg[k * 2 + 1];
        }
        acc += ga * s;
    }
    acc
}

/// Single weight of a q4tp tensor — the a8w8 outlier path, which restores
/// activation outliers at full precision after the int8 pass.
#[inline]
fn q4tp_outlier(nib: &[u8], r: usize, gpr: usize, j: usize, scales: &[f32]) -> (f32, f32) {
    let (gi, k) = (j / GROUP_SIZE, j % GROUP_SIZE);
    let byte = nib[(r * gpr + gi) * Q4TP_NIB + k / 2];
    let n = if k & 1 == 0 { byte & 0x0F } else { byte >> 4 };
    ((n as i32 - 8) as f32, scales[gi])
}

/// Fused q4tp matvec (dispatch mirrors `q4t_matvec`).
fn q4tp_matvec(
    bytes: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), rows);
    let gpr = cols / GROUP_SIZE;
    let v = Q4tpView::new(bytes, rows, cols);
    let out_addr = SendMut(out.as_mut_ptr());
    if a8w8_enabled() {
        let act = split_act(x);
        let run = |start: usize, end: usize| {
            // One scratch row of scales per worker — borrowed, not minted.
            with_krow(gpr, |sc| {
                for r in start..end {
                    v.scales_into(r, gpr, sc);
                    let mut acc = dot_q4tp_row_i8(v.nib, r, gpr, &act.xq, sc) * act.sx;
                    for &(j, xv) in &act.outliers {
                        let (w, s) = q4tp_outlier(v.nib, r, gpr, j, sc);
                        acc += w * s * xv;
                    }
                    // SAFETY: disjoint row ranges per worker.
                    unsafe { *out_addr.at(r) = acc };
                }
            })
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = |start: usize, end: usize| {
        with_krow(gpr, |sc| {
            for r in start..end {
                v.scales_into(r, gpr, sc);
                // SAFETY: disjoint row ranges per worker.
                unsafe { *out_addr.at(r) = q4tp_row_exact(v.nib, r, gpr, x, sc) };
            }
        })
    };
    dispatch_rows(pool, rows, &run);
}

/// Fused two-input q4tp matvec — the SwiGLU gate/up pair. Weights and the
/// row ladder are read once and spent on both activation streams.
#[allow(clippy::too_many_arguments)]
fn q4tp_matvec2(
    bytes: &[u8],
    x1: &[f32],
    x2: &[f32],
    rows: usize,
    cols: usize,
    o1: &mut [f32],
    o2: &mut [f32],
    pool: Option<&Pool>,
) {
    let gpr = cols / GROUP_SIZE;
    let v = Q4tpView::new(bytes, rows, cols);
    let (p1, p2) = (SendMut(o1.as_mut_ptr()), SendMut(o2.as_mut_ptr()));
    let run = |start: usize, end: usize| {
        let mut sc = vec![0f32; gpr];
        for r in start..end {
            v.scales_into(r, gpr, &mut sc);
            // SAFETY: disjoint row ranges per worker.
            unsafe {
                *p1.at(r) = q4tp_row_exact(v.nib, r, gpr, x1, &sc);
                *p2.at(r) = q4tp_row_exact(v.nib, r, gpr, x2, &sc);
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// One q2tp outlier weight at column `j` of row `r`: the 2-bit code and
/// its group scale, mirrored on `q4tp_outlier`.
#[inline]
fn q2tp_outlier(chunks: &[u8], r: usize, gpr: usize, j: usize, scales: &[f32]) -> (f32, f32) {
    let (gi, k) = (j / GROUP_SIZE, j % GROUP_SIZE);
    let byte = chunks[(r * gpr + gi) * Q2TP_CHUNK + k / 4];
    let c = (byte >> (2 * (k % 4))) & 3;
    (c as f32 - 1.5, scales[gi])
}

/// Integer dot of one q2tp row against pre-quantized activations:
/// Σ_g s_g · (Σ c·xq − 1.5·Σ xq). The half-integer grid (c − 1.5)
/// becomes exact integer math through the group sums — the same trick
/// every a8w8 kernel in this file rides. The codes decode into a
/// 32-byte scratch in natural order and the dot itself is the shared
/// SDOT primitive; elsewhere a scalar integer loop.
#[inline]
fn dot_q2tp_row_i8(
    chunks: &[u8],
    r: usize,
    gpr: usize,
    xq: &[i8],
    gsum: &[i32],
    scales: &[f32],
) -> f32 {
    let mut acc = 0f32;
    let base = r * gpr * Q2TP_CHUNK;
    #[cfg(not(target_arch = "aarch64"))]
    let mut codes = [0i8; GROUP_SIZE];
    for gi in 0..gpr {
        let ch = &chunks[base + gi * Q2TP_CHUNK..base + (gi + 1) * Q2TP_CHUNK];
        let xg = &xq[gi * GROUP_SIZE..(gi + 1) * GROUP_SIZE];
        #[cfg(target_arch = "aarch64")]
        // NEON: the byte's four 2-bit fields land in four lane vectors
        // (shift+mask), vld4 de-interleaves xq to match (xj[k] =
        // xq[4k+j]), widening MACs accumulate exactly in i32. A scalar
        // decode here cost as much as the dot it fed — the profile put
        // it at the top of the whole W2 decode.
        let dot = unsafe {
            use core::arch::aarch64::*;
            let b = vld1_u8(ch.as_ptr());
            let three = vdup_n_u8(3);
            let c0 = vreinterpret_s8_u8(vand_u8(b, three));
            let c1 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(b, 2), three));
            let c2 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(b, 4), three));
            let c3 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(b, 6), three));
            let x4 = vld4_s8(xg.as_ptr());
            let mut acc4 = vdupq_n_s32(0);
            acc4 = vpadalq_s16(acc4, vmull_s8(c0, x4.0));
            acc4 = vpadalq_s16(acc4, vmull_s8(c1, x4.1));
            acc4 = vpadalq_s16(acc4, vmull_s8(c2, x4.2));
            acc4 = vpadalq_s16(acc4, vmull_s8(c3, x4.3));
            vaddvq_s32(acc4)
        };
        #[cfg(not(target_arch = "aarch64"))]
        let dot: i32 = {
            for (k, &b) in ch.iter().enumerate() {
                codes[k * 4] = (b & 3) as i8;
                codes[k * 4 + 1] = ((b >> 2) & 3) as i8;
                codes[k * 4 + 2] = ((b >> 4) & 3) as i8;
                codes[k * 4 + 3] = ((b >> 6) & 3) as i8;
            }
            codes
                .iter()
                .zip(xg)
                .map(|(&c, &x)| c as i32 * x as i32)
                .sum()
        };
        acc += scales[gi] * (dot as f32 - 1.5 * gsum[gi] as f32);
    }
    acc
}

/// Exact f32 dot of one q2tp row: 2-bit fields LSB-first, (c − 1.5)·s.
/// Scalar on purpose — the 2-bit class targets the GPU graph; the CPU
/// path exists for parity gates and small-machine fallback.
fn q2tp_row_exact(chunks: &[u8], r: usize, gpr: usize, x: &[f32], scales: &[f32]) -> f32 {
    let mut acc = 0f32;
    for gi in 0..gpr {
        let ch = &chunks[(r * gpr + gi) * Q2TP_CHUNK..(r * gpr + gi + 1) * Q2TP_CHUNK];
        let s = scales[gi];
        let xb = &x[gi * GROUP_SIZE..(gi + 1) * GROUP_SIZE];
        let mut g = 0f32;
        for (k, &b) in ch.iter().enumerate() {
            g += ((b & 3) as f32 - 1.5) * xb[k * 4]
                + (((b >> 2) & 3) as f32 - 1.5) * xb[k * 4 + 1]
                + (((b >> 4) & 3) as f32 - 1.5) * xb[k * 4 + 2]
                + (((b >> 6) & 3) as f32 - 1.5) * xb[k * 4 + 3];
        }
        acc += s * g;
    }
    acc
}

fn q2tp_matvec(
    bytes: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), rows);
    let gpr = cols / GROUP_SIZE;
    let v = Q4tpView::new_q2(bytes, rows, cols);
    let out_addr = SendMut(out.as_mut_ptr());
    // a8w8 fast path (CMF_SDOT=0 keeps the exact scalar walk): integer
    // code dots + group sums, exact outlier correction — the same
    // contract as every sibling kernel; measured 2-bit rows were the
    // only scalar holdout in the family.
    if a8w8_enabled() {
        let act = split_act(x);
        let gsum = q1_group_sums(&act.xq, gpr);
        let (act, gsum) = (&act, &gsum);
        let run = move |start: usize, end: usize| {
            with_krow(gpr, |sc| {
                for r in start..end {
                    v.scales_into(r, gpr, sc);
                    let mut acc = dot_q2tp_row_i8(v.nib, r, gpr, &act.xq, gsum, sc) * act.sx;
                    for &(j, xv) in &act.outliers {
                        let (w, s) = q2tp_outlier(v.nib, r, gpr, j, sc);
                        acc += w * s * xv;
                    }
                    // SAFETY: disjoint row ranges per worker.
                    unsafe { *out_addr.at(r) = acc };
                }
            })
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = |start: usize, end: usize| {
        with_krow(gpr, |sc| {
            for r in start..end {
                v.scales_into(r, gpr, sc);
                // SAFETY: disjoint row ranges per worker.
                unsafe { *out_addr.at(r) = q2tp_row_exact(v.nib, r, gpr, x, sc) };
            }
        })
    };
    dispatch_rows(pool, rows, &run);
}

/// Fused two-input q2tp matvec — the SwiGLU gate/up pair.
#[allow(clippy::too_many_arguments)]
fn q2tp_matvec2(
    bytes: &[u8],
    x1: &[f32],
    x2: &[f32],
    rows: usize,
    cols: usize,
    o1: &mut [f32],
    o2: &mut [f32],
    pool: Option<&Pool>,
) {
    let gpr = cols / GROUP_SIZE;
    let v = Q4tpView::new_q2(bytes, rows, cols);
    let (p1, p2) = (SendMut(o1.as_mut_ptr()), SendMut(o2.as_mut_ptr()));
    let run = |start: usize, end: usize| {
        let mut sc = vec![0f32; gpr];
        for r in start..end {
            v.scales_into(r, gpr, &mut sc);
            // SAFETY: disjoint row ranges per worker.
            unsafe {
                *p1.at(r) = q2tp_row_exact(v.nib, r, gpr, x1, &sc);
                *p2.at(r) = q2tp_row_exact(v.nib, r, gpr, x2, &sc);
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// Batched q2tp matmat: scalar row kernel over every batch column. CPU
/// prefill only — decode rides the graph, so plain and correct beats
/// clever here.
/// Test doors into the host 2-bit kernels: the stand's heap corruption
/// pointed at down-shaped tensors, and the private fns need a way to be
/// held to a reference without a model file around them.
pub fn q2tp_matvec_for_test(bytes: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    // The facade IS the reference: encoder oracles hold requant output
    // to the exact scalar walk. The production dispatch may take the i8
    // fast path, whose error scale is the ACTIVATIONS' — a different
    // claim than the encoder correctness these tests pin.
    let gpr = cols / GROUP_SIZE;
    let v = Q4tpView::new_q2(bytes, rows, cols);
    with_krow(gpr, |sc| {
        for r in 0..rows {
            v.scales_into(r, gpr, sc);
            out[r] = q2tp_row_exact(v.nib, r, gpr, x, sc);
        }
    });
}

pub fn q2tp_matmat_for_test(
    bytes: &[u8],
    xs_all: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    q2tp_matmat(bytes, xs_all, b, rows, cols, out, None);
}

fn q2tp_matmat(
    bytes: &[u8],
    xs_all: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), b * rows);
    let gpr = cols / GROUP_SIZE;
    let v = Q4tpView::new_q2(bytes, rows, cols);
    let out_addr = SendMut(out.as_mut_ptr());
    let run = |start: usize, end: usize| {
        let mut sc = vec![0f32; gpr];
        for r in start..end {
            v.scales_into(r, gpr, &mut sc);
            for bi in 0..b {
                let x = &xs_all[bi * cols..(bi + 1) * cols];
                // SAFETY: disjoint row ranges per worker.
                unsafe { *out_addr.at(bi * rows + r) = q2tp_row_exact(v.nib, r, gpr, x, &sc) };
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// The pre-vectorised shape, kept for A/B (`CMF_Q4TP_V1=1`): the
/// horizontal add lands once per group per column instead of once per
/// row. Same weights, same activations — only the reduction differs.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q4tp_row_1x4_sdot_v1(
    nib: &[u8],
    r: usize,
    gpr: usize,
    xs: [&[i8]; 4],
    scales: &[f32],
) -> [f32; 4] {
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let lomask = vdupq_n_u8(0x0F);
        let eight = vdupq_n_s8(8);
        let (mut f0, mut f1, mut f2, mut f3) = (0f32, 0f32, 0f32, 0f32);
        for gi in 0..gpr {
            let t = nib.as_ptr().add((r * gpr + gi) * Q4TP_NIB);
            let s = *scales.get_unchecked(gi);
            let bb = vld1q_u8(t);
            let lo = vandq_u8(bb, lomask);
            let hi = vshrq_n_u8::<4>(bb);
            let e0 = vsubq_s8(vreinterpretq_s8_u8(vzip1q_u8(lo, hi)), eight);
            let e1 = vsubq_s8(vreinterpretq_s8_u8(vzip2q_u8(lo, hi)), eight);
            let mut d = [0f32; 4];
            for (k, dk) in d.iter_mut().enumerate() {
                let x0 = vld1q_s8(xs[k].as_ptr().add(gi * GROUP_SIZE));
                let x1 = vld1q_s8(xs[k].as_ptr().add(gi * GROUP_SIZE + 16));
                let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
                asm!(
                    "sdot {a0:v}.4s, {e0:v}.16b, {x0:v}.16b",
                    "sdot {a1:v}.4s, {e1:v}.16b, {x1:v}.16b",
                    a0 = inout(vreg) a0, a1 = inout(vreg) a1,
                    e0 = in(vreg) e0, x0 = in(vreg) x0, e1 = in(vreg) e1, x1 = in(vreg) x1,
                    options(pure, nomem, nostack),
                );
                *dk = vaddvq_s32(vaddq_s32(a0, a1)) as f32 * s;
            }
            f0 += d[0];
            f1 += d[1];
            f2 += d[2];
            f3 += d[3];
        }
        [f0, f1, f2, f3]
    }
}

/// Which q4tp batch kernel to run: 1 = the previous one, 2 = the tuned
/// one, 0 = decide from the CPU. An atomic rather than a `OnceLock` so a
/// benchmark can alternate the two inside one process, where the machine's
/// mood — a shared box drifts ±25% between runs — is the same for both.
/// What the two mean is per-architecture: on x86 the blocked AVX-512 path
/// against the per-column one, on ARM the two reduction shapes.
#[allow(dead_code)]
static Q4TP_ALT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Blocking pays on x86 only with 512-bit VNNI. With AVX2 alone, four
/// columns sharing an unpack still measured slower than the per-column
/// path (23.2 ms against 19.4 on a 48-thread EPYC), because that path
/// already dequantizes the row once — so the blocked kernel bought a
/// second unpack-free pass at the price of half the vector width.
#[cfg(target_arch = "x86_64")]
fn q4tp_blocked_x86() -> bool {
    match Q4TP_ALT.load(std::sync::atomic::Ordering::Relaxed) {
        1 => false,
        // A forced ON still asks the CPU. The switch exists so a bench can
        // pick a kernel, not so it can promise instructions the machine
        // does not have — CI caught that as a SIGILL on a runner without
        // AVX-512, where the parity test had turned the path on by hand.
        2 => avx512vnni_enabled(),
        // Deliberately not cached back into the switch: both gates below
        // hold their own `OnceLock`, and latching their answer here would
        // make a test's override outlive the test that set it.
        _ => blocked_enabled() && avx512vnni_enabled(),
    }
}

/// `CMF_Q4TP_V1=1` picks the old reduction shape (A/B only).
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn q4tp_v1() -> bool {
    match Q4TP_ALT.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("CMF_Q4TP_V1").is_ok_and(|v| v != "0"))
        }
    }
}

/// Two weight rows against eight columns. The activation load is the
/// same for both rows, so it is paid once for twice the arithmetic, and
/// sixteen accumulator chains run where eight did — which is what a kernel
/// retiring 0.29 instructions a cycle is short of. Register pressure is
/// the limit: sixteen `zmm` accumulators, two weight tiles, one
/// activation, of thirty-two.
///
/// Four rows by four columns spends the same sixteen accumulators the
/// other way and measured worse — 1488 GFLOP/s against 1644 — so the
/// unpack, which four rows pay twice as often, costs more than the extra
/// sharing of one activation load buys.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot_q4tp_2x8_avx512(
    nib: &[u8],
    r0: usize,
    gpr: usize,
    xs: [&[i8]; 8],
    sc0: &[f32],
    sc1: &[f32],
) -> [[f32; 8]; 2] {
    // SAFETY: as dot_q4tp_row_1x8_avx512, two adjacent rows at once; the
    // caller guarantees r0 + 1 < rows and the ISA.
    unsafe {
        use core::arch::x86_64::*;
        let lomask = _mm256_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let zero = _mm512_setzero_si512();
        let mut v0 = [_mm512_setzero_ps(); 8];
        let mut v1 = [_mm512_setzero_ps(); 8];
        let pairs = gpr / 2;
        let unpack = |r: usize, gi: usize| -> (__m512i, __mmask64) {
            let t = nib.as_ptr().add((r * gpr + gi) * Q4TP_NIB);
            let bb = _mm256_loadu_si256(t as *const __m256i);
            let lo = _mm256_and_si256(bb, lomask);
            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(bb), lomask);
            let ul = _mm256_sub_epi8(_mm256_unpacklo_epi8(lo, hi), eight);
            let uh = _mm256_sub_epi8(_mm256_unpackhi_epi8(lo, hi), eight);
            let cat = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(ul), uh);
            let w = _mm512_shuffle_i64x2::<0b11_01_10_00>(cat, cat);
            (_mm512_abs_epi8(w), _mm512_movepi8_mask(w))
        };
        for gp in 0..pairs {
            let gi = gp * 2;
            let (wa0, neg0) = unpack(r0, gi);
            let (wa1, neg1) = unpack(r0 + 1, gi);
            let off = gi * GROUP_SIZE;
            let sv = |sc: &[f32]| {
                _mm512_insertf32x8::<1>(
                    _mm512_castps256_ps512(_mm256_set1_ps(*sc.get_unchecked(gi))),
                    _mm256_set1_ps(*sc.get_unchecked(gi + 1)),
                )
            };
            let s0 = sv(sc0);
            let s1 = sv(sc1);
            for k in 0..8 {
                let xv = _mm512_loadu_si512(xs[k].as_ptr().add(off) as *const __m512i);
                let d0 = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(
                    zero,
                    wa0,
                    _mm512_mask_sub_epi8(xv, neg0, zero, xv),
                ));
                let d1 = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(
                    zero,
                    wa1,
                    _mm512_mask_sub_epi8(xv, neg1, zero, xv),
                ));
                v0[k] = _mm512_fmadd_ps(d0, s0, v0[k]);
                v1[k] = _mm512_fmadd_ps(d1, s1, v1[k]);
            }
        }
        let mut acc = [[0f32; 8]; 2];
        for k in 0..8 {
            acc[0][k] = _mm512_reduce_add_ps(v0[k]);
            acc[1][k] = _mm512_reduce_add_ps(v1[k]);
        }
        if gpr % 2 == 1 {
            let off = (gpr - 1) * GROUP_SIZE;
            for j in off..off + GROUP_SIZE {
                let (w0, sa) = q4tp_outlier(nib, r0, gpr, j, sc0);
                let (w1, sb) = q4tp_outlier(nib, r0 + 1, gpr, j, sc1);
                for k in 0..8 {
                    let x = *xs[k].get_unchecked(j) as f32;
                    acc[0][k] += w0 * sa * x;
                    acc[1][k] += w1 * sb * x;
                }
            }
        }
        acc
    }
}

/// The same, eight columns at a time. One unpack then feeds twice as many
/// activation streams, so a wide batch reads the weight tile half as
/// often; the price is eight accumulators live at once. Measured 9.0 ->
/// 8.3 ms at 9216x2304, b=296 on a 48-thread EPYC 9B45.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot_q4tp_row_1x8_avx512(
    nib: &[u8],
    r: usize,
    gpr: usize,
    xs: [&[i8]; 8],
    scales: &[f32],
) -> [f32; 8] {
    // SAFETY: as dot_q4tp_row_1x4_avx2; caller guarantees the ISA.
    unsafe {
        use core::arch::x86_64::*;
        let lomask = _mm256_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let zero = _mm512_setzero_si512();
        let (mut v0, mut v1, mut v2, mut v3) = (
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
        );
        let (mut v4, mut v5, mut v6, mut v7) = (
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
        );
        let pairs = gpr / 2;
        for gp in 0..pairs {
            let gi = gp * 2;
            let t = nib.as_ptr().add((r * gpr + gi) * Q4TP_NIB);
            let bb = _mm256_loadu_si256(t as *const __m256i);
            let lo = _mm256_and_si256(bb, lomask);
            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(bb), lomask);
            // `unpack` works per 128-bit lane, so the halves come out as
            // [A.lo, B.lo] and [A.hi, B.hi]; the shuffle reorders the four
            // 128-bit lanes into the weights' natural order, which is what
            // the straight activation load expects.
            let ul = _mm256_sub_epi8(_mm256_unpacklo_epi8(lo, hi), eight);
            let uh = _mm256_sub_epi8(_mm256_unpackhi_epi8(lo, hi), eight);
            let cat = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(ul), uh);
            let w = _mm512_shuffle_i64x2::<0b11_01_10_00>(cat, cat);
            let wabs = _mm512_abs_epi8(w);
            let neg = _mm512_movepi8_mask(w);
            let off = gi * GROUP_SIZE;
            let sv = _mm512_insertf32x8::<1>(
                _mm512_castps256_ps512(_mm256_set1_ps(*scales.get_unchecked(gi))),
                _mm256_set1_ps(*scales.get_unchecked(gi + 1)),
            );
            let dot = |x: &[i8]| -> __m512 {
                let xv = _mm512_loadu_si512(x.as_ptr().add(off) as *const __m512i);
                let sx = _mm512_mask_sub_epi8(xv, neg, zero, xv);
                _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(zero, wabs, sx))
            };
            v0 = _mm512_fmadd_ps(dot(xs[0]), sv, v0);
            v1 = _mm512_fmadd_ps(dot(xs[1]), sv, v1);
            v2 = _mm512_fmadd_ps(dot(xs[2]), sv, v2);
            v3 = _mm512_fmadd_ps(dot(xs[3]), sv, v3);
            v4 = _mm512_fmadd_ps(dot(xs[4]), sv, v4);
            v5 = _mm512_fmadd_ps(dot(xs[5]), sv, v5);
            v6 = _mm512_fmadd_ps(dot(xs[6]), sv, v6);
            v7 = _mm512_fmadd_ps(dot(xs[7]), sv, v7);
        }
        let mut acc = [
            _mm512_reduce_add_ps(v0),
            _mm512_reduce_add_ps(v1),
            _mm512_reduce_add_ps(v2),
            _mm512_reduce_add_ps(v3),
            _mm512_reduce_add_ps(v4),
            _mm512_reduce_add_ps(v5),
            _mm512_reduce_add_ps(v6),
            _mm512_reduce_add_ps(v7),
        ];
        // An odd group count leaves one group over; the narrow kernel
        // finishes it rather than the tail being a special case here.
        if gpr % 2 == 1 {
            let off = (gpr - 1) * GROUP_SIZE;
            for j in off..off + GROUP_SIZE {
                let (w, s) = q4tp_outlier(nib, r, gpr, j, scales);
                let ws = w * s;
                for k in 0..8 {
                    acc[k] += ws * *xs[k].get_unchecked(j) as f32;
                }
            }
        }
        acc
    }
}

/// The same four columns, 512 bits wide. Two groups (64 weights) ride one
/// unpack and one `vpdpbusd`, where AVX2 needs two unpacks and four
/// `maddubs`/`madd` pairs — about 2.3x fewer instructions for the same
/// arithmetic. The two groups carry different scales, so the fma takes a
/// vector whose halves hold each group's scale rather than a broadcast.
///
/// There is no 512-bit `vpsignb`, so the activation's sign is applied by
/// negating under a mask taken from the weight's sign bits. That mask is
/// per-tile, so it is hoisted out of the column loop and the per-column
/// cost stays exactly one instruction, as with `sign_epi8`. Weights of
/// zero are not zeroed by the mask trick and do not need to be: their
/// magnitude is zero, so the product is.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot_q4tp_row_1x4_avx512(
    nib: &[u8],
    r: usize,
    gpr: usize,
    xs: [&[i8]; 4],
    scales: &[f32],
) -> [f32; 4] {
    // SAFETY: as dot_q4tp_row_1x4_avx2; caller guarantees the ISA.
    unsafe {
        use core::arch::x86_64::*;
        let lomask = _mm256_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let zero = _mm512_setzero_si512();
        let (mut v0, mut v1, mut v2, mut v3) = (
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
        );
        let pairs = gpr / 2;
        for gp in 0..pairs {
            let gi = gp * 2;
            let t = nib.as_ptr().add((r * gpr + gi) * Q4TP_NIB);
            let bb = _mm256_loadu_si256(t as *const __m256i);
            let lo = _mm256_and_si256(bb, lomask);
            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(bb), lomask);
            // `unpack` works per 128-bit lane, so the halves come out as
            // [A.lo, B.lo] and [A.hi, B.hi]; the shuffle reorders the four
            // 128-bit lanes into the weights' natural order, which is what
            // the straight activation load expects.
            let ul = _mm256_sub_epi8(_mm256_unpacklo_epi8(lo, hi), eight);
            let uh = _mm256_sub_epi8(_mm256_unpackhi_epi8(lo, hi), eight);
            let cat = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(ul), uh);
            let w = _mm512_shuffle_i64x2::<0b11_01_10_00>(cat, cat);
            let wabs = _mm512_abs_epi8(w);
            let neg = _mm512_movepi8_mask(w);
            let off = gi * GROUP_SIZE;
            let sv = _mm512_insertf32x8::<1>(
                _mm512_castps256_ps512(_mm256_set1_ps(*scales.get_unchecked(gi))),
                _mm256_set1_ps(*scales.get_unchecked(gi + 1)),
            );
            let dot = |x: &[i8]| -> __m512 {
                let xv = _mm512_loadu_si512(x.as_ptr().add(off) as *const __m512i);
                let sx = _mm512_mask_sub_epi8(xv, neg, zero, xv);
                _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(zero, wabs, sx))
            };
            v0 = _mm512_fmadd_ps(dot(xs[0]), sv, v0);
            v1 = _mm512_fmadd_ps(dot(xs[1]), sv, v1);
            v2 = _mm512_fmadd_ps(dot(xs[2]), sv, v2);
            v3 = _mm512_fmadd_ps(dot(xs[3]), sv, v3);
        }
        let mut acc = [
            _mm512_reduce_add_ps(v0),
            _mm512_reduce_add_ps(v1),
            _mm512_reduce_add_ps(v2),
            _mm512_reduce_add_ps(v3),
        ];
        // An odd group count leaves one group over; the narrow kernel
        // finishes it rather than the tail being a special case here.
        if gpr % 2 == 1 {
            let off = (gpr - 1) * GROUP_SIZE;
            for j in off..off + GROUP_SIZE {
                let (w, s) = q4tp_outlier(nib, r, gpr, j, scales);
                let ws = w * s;
                for k in 0..4 {
                    acc[k] += ws * *xs[k].get_unchecked(j) as f32;
                }
            }
        }
        acc
    }
}

/// Four batch columns against one q4tp row: the tile is unpacked ONCE and
/// spent on four activation streams, which is where a prefill batch stops
/// being weight-bandwidth-bound. Twin of `dot_q4t_row_1x4_sdot`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q4tp_row_1x4_sdot(
    nib: &[u8],
    r: usize,
    gpr: usize,
    xs: [&[i8]; 4],
    scales: &[f32],
) -> [f32; 4] {
    // SAFETY: see dot_q4tp_row_sdot; every xs[k] is gpr·GROUP_SIZE long.
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let lomask = vdupq_n_u8(0x0F);
        let eight = vdupq_n_s8(8);
        // Named accumulators, NOT an array indexed by a loop variable: the
        // latter does not stay in registers (the same defect cost 2x in the
        // AVX2 q4t kernel and again in WGSL).
        //
        // They are VECTORS, and the horizontal add happens once at the end
        // instead of once per group per column. `vaddvq` is a cross-lane
        // reduction — with 72 groups and four columns the old shape paid
        // 288 of them per row, each one a dependency stall the pipeline
        // cannot hide, to save four float adds. The group's scale now
        // rides an fma into the lane accumulators, so the arithmetic per
        // group is one convert and one fma. Summation order changes (the
        // lanes carry independent partial sums), which is the same
        // round-off class the SDOT path already lives in — the strict
        // kernel (`CMF_SDOT=0`, what `cortiq ppl` runs) is unchanged and
        // stays the reference.
        let (mut v0, mut v1, mut v2, mut v3) = (
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
        );
        for gi in 0..gpr {
            let t = nib.as_ptr().add((r * gpr + gi) * Q4TP_NIB);
            let s = *scales.get_unchecked(gi);
            let bb = vld1q_u8(t);
            let lo = vandq_u8(bb, lomask);
            let hi = vshrq_n_u8::<4>(bb);
            let e0 = vsubq_s8(vreinterpretq_s8_u8(vzip1q_u8(lo, hi)), eight);
            let e1 = vsubq_s8(vreinterpretq_s8_u8(vzip2q_u8(lo, hi)), eight);
            let off = gi * GROUP_SIZE;
            let dot4 = |x: &[i8]| -> int32x4_t {
                let x0 = vld1q_s8(x.as_ptr().add(off));
                let x1 = vld1q_s8(x.as_ptr().add(off + 16));
                let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
                asm!(
                    "sdot {a0:v}.4s, {e0:v}.16b, {x0:v}.16b",
                    "sdot {a1:v}.4s, {e1:v}.16b, {x1:v}.16b",
                    a0 = inout(vreg) a0, a1 = inout(vreg) a1,
                    e0 = in(vreg) e0, x0 = in(vreg) x0, e1 = in(vreg) e1, x1 = in(vreg) x1,
                    options(pure, nomem, nostack),
                );
                vaddq_s32(a0, a1)
            };
            v0 = vfmaq_n_f32(v0, vcvtq_f32_s32(dot4(xs[0])), s);
            v1 = vfmaq_n_f32(v1, vcvtq_f32_s32(dot4(xs[1])), s);
            v2 = vfmaq_n_f32(v2, vcvtq_f32_s32(dot4(xs[2])), s);
            v3 = vfmaq_n_f32(v3, vcvtq_f32_s32(dot4(xs[3])), s);
        }
        [
            vaddvq_f32(v0),
            vaddvq_f32(v1),
            vaddvq_f32(v2),
            vaddvq_f32(v3),
        ]
    }
}

/// Fused q4tp matmat — the same three arms `q4t_matmat` has. Shipping only
/// the scalar one made Nanbeige-3B decode at 1.2 tok/s against q4t's 5.9:
/// the format was fine, the missing arms were the whole regression.
fn q4tp_matmat(
    bytes: &[u8],
    xs_all: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), b * rows);
    let gpr = cols / GROUP_SIZE;
    let v = Q4tpView::new(bytes, rows, cols);

    // Wide batches ride the AMX through a dequant-tile sgemm, as in q4t.
    #[cfg(target_os = "macos")]
    if b >= 8 && rows * cols >= 500_000 && accel_gemm_enabled() {
        dequant_matmat_accel(
            &|r, dst| {
                let mut sc = [0f32; 32];
                let mut scv;
                let s: &[f32] = if gpr <= 32 {
                    v.scales_into(r, gpr, &mut sc);
                    &sc[..gpr]
                } else {
                    scv = vec![0f32; gpr];
                    v.scales_into(r, gpr, &mut scv);
                    &scv
                };
                for gi in 0..gpr {
                    let tile = &v.nib[(r * gpr + gi) * Q4TP_NIB..(r * gpr + gi + 1) * Q4TP_NIB];
                    for (k, &bb) in tile.iter().enumerate() {
                        dst[gi * GROUP_SIZE + k * 2] = ((bb & 0x0F) as f32 - 8.0) * s[gi];
                        dst[gi * GROUP_SIZE + k * 2 + 1] =
                            (((bb >> 4) & 0x0F) as f32 - 8.0) * s[gi];
                    }
                }
            },
            xs_all,
            b,
            rows,
            cols,
            out,
            pool,
        );
        return;
    }

    let out_addr = SendMut(out.as_mut_ptr());
    if a8w8_enabled() {
        let acts: Vec<SplitAct> = (0..b)
            .map(|bi| split_act(&xs_all[bi * cols..(bi + 1) * cols]))
            .collect();
        let acts = &acts;
        #[cfg(target_arch = "aarch64")]
        let blocked_ok = sdot_enabled() && blocked_enabled();
        // x86 gets the same blocking: one tile unpack spent on four
        // columns. Without it every column re-decoded the row, which is
        // why a 48-core EPYC measured a sixth of an M4's per-core rate.
        // The gate is `avx2_enabled`, as in q4t — `sdot_enabled` answers
        // for ARM's dotprod and is hard-wired false everywhere else, so
        // asking it here left the whole blocked path unreachable on x86.
        #[cfg(target_arch = "x86_64")]
        let blocked_ok = q4tp_blocked_x86();
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let blocked_ok = false;
        // Columns are swept in panels that fit L2. Without this a
        // row-pair walks every activation in the batch — 4.8 MB at
        // 512x512 — and does it again for the next pair, so the whole
        // batch streams out of the shared cache once per row. Measured
        // 800 GB/s of it, flat across batch sizes, which is the signature
        // of a loop bound by traffic rather than by arithmetic. A panel of
        // 256 columns is 590 KB beside 221 KB of this worker's weights:
        // both stay resident and the batch crosses L3 once instead of
        // once per row.
        let panel_cols: usize = std::env::var("CMF_Q4TP_PANEL")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(256);
        let run = |start: usize, end: usize| {
            for abase in (0..acts.len()).step_by(panel_cols) {
                let alen = (acts.len() - abase).min(panel_cols);
                let mut sc = vec![0f32; gpr];
                #[cfg(target_arch = "x86_64")]
                let mut r_lo = start;
                #[cfg(target_arch = "x86_64")]
                if blocked_ok && alen >= 8 {
                    let mut sc1 = vec![0f32; gpr];
                    while r_lo + 2 <= end {
                        v.scales_into(r_lo, gpr, &mut sc);
                        v.scales_into(r_lo + 1, gpr, &mut sc1);
                        let mut bi = 0usize;
                        while bi + 8 <= alen {
                            let xs = [
                                acts[abase + bi].xq.as_slice(),
                                acts[abase + bi + 1].xq.as_slice(),
                                acts[abase + bi + 2].xq.as_slice(),
                                acts[abase + bi + 3].xq.as_slice(),
                                acts[abase + bi + 4].xq.as_slice(),
                                acts[abase + bi + 5].xq.as_slice(),
                                acts[abase + bi + 6].xq.as_slice(),
                                acts[abase + bi + 7].xq.as_slice(),
                            ];
                            let d = unsafe { dot_q4tp_2x8_avx512(v.nib, r_lo, gpr, xs, &sc, &sc1) };
                            for (row, dr, scr) in [(r_lo, &d[0], &sc), (r_lo + 1, &d[1], &sc1)] {
                                for k in 0..8 {
                                    let act = &acts[abase + bi + k];
                                    let mut acc = dr[k] * act.sx;
                                    for &(j, xv) in &act.outliers {
                                        let (w, s) = q4tp_outlier(v.nib, row, gpr, j, scr);
                                        acc += w * s * xv;
                                    }
                                    // SAFETY: disjoint (bi, r) cells per worker.
                                    unsafe { *out_addr.at((abase + bi + k) * rows + row) = acc };
                                }
                            }
                            bi += 8;
                        }
                        // Columns past the last group of eight, both rows —
                        // the same single-row kernel the tail below uses.
                        for row in [r_lo, r_lo + 1] {
                            let scr: &[f32] = if row == r_lo { &sc } else { &sc1 };
                            for b2 in bi..alen {
                                let act = &acts[abase + b2];
                                let xs4 = [
                                    act.xq.as_slice(),
                                    act.xq.as_slice(),
                                    act.xq.as_slice(),
                                    act.xq.as_slice(),
                                ];
                                let d =
                                    unsafe { dot_q4tp_row_1x4_avx512(v.nib, row, gpr, xs4, scr) };
                                let mut acc = d[0] * act.sx;
                                for &(j, xv) in &act.outliers {
                                    let (w, s) = q4tp_outlier(v.nib, row, gpr, j, scr);
                                    acc += w * s * xv;
                                }
                                // SAFETY: disjoint (bi, r) cells per worker.
                                unsafe { *out_addr.at((abase + b2) * rows + row) = acc };
                            }
                        }
                        r_lo += 2;
                    }
                }
                #[cfg(target_arch = "x86_64")]
                let row_start = r_lo;
                #[cfg(not(target_arch = "x86_64"))]
                let row_start = start;
                for r in row_start..end {
                    v.scales_into(r, gpr, &mut sc);
                    let mut bi = 0usize;
                    #[cfg(target_arch = "x86_64")]
                    if blocked_ok {
                        while bi + 8 <= alen {
                            let xs = [
                                acts[abase + bi].xq.as_slice(),
                                acts[abase + bi + 1].xq.as_slice(),
                                acts[abase + bi + 2].xq.as_slice(),
                                acts[abase + bi + 3].xq.as_slice(),
                                acts[abase + bi + 4].xq.as_slice(),
                                acts[abase + bi + 5].xq.as_slice(),
                                acts[abase + bi + 6].xq.as_slice(),
                                acts[abase + bi + 7].xq.as_slice(),
                            ];
                            let d = unsafe { dot_q4tp_row_1x8_avx512(v.nib, r, gpr, xs, &sc) };
                            for k in 0..8 {
                                let act = &acts[abase + bi + k];
                                let mut acc = d[k] * act.sx;
                                for &(j, xv) in &act.outliers {
                                    let (w, s) = q4tp_outlier(v.nib, r, gpr, j, &sc);
                                    acc += w * s * xv;
                                }
                                // SAFETY: disjoint (bi, r) cells per worker.
                                unsafe { *out_addr.at((abase + bi + k) * rows + r) = acc };
                            }
                            bi += 8;
                        }
                        while bi + 4 <= alen {
                            let xs = [
                                acts[abase + bi].xq.as_slice(),
                                acts[abase + bi + 1].xq.as_slice(),
                                acts[abase + bi + 2].xq.as_slice(),
                                acts[abase + bi + 3].xq.as_slice(),
                            ];
                            let d = unsafe { dot_q4tp_row_1x4_avx512(v.nib, r, gpr, xs, &sc) };
                            for k in 0..4 {
                                let act = &acts[abase + bi + k];
                                let mut acc = d[k] * act.sx;
                                for &(j, xv) in &act.outliers {
                                    let (w, s) = q4tp_outlier(v.nib, r, gpr, j, &sc);
                                    acc += w * s * xv;
                                }
                                // SAFETY: disjoint (bi, r) cells per worker.
                                unsafe { *out_addr.at((abase + bi + k) * rows + r) = acc };
                            }
                            bi += 4;
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    if blocked_ok {
                        while bi + 4 <= alen {
                            let xs = [
                                acts[abase + bi].xq.as_slice(),
                                acts[abase + bi + 1].xq.as_slice(),
                                acts[abase + bi + 2].xq.as_slice(),
                                acts[abase + bi + 3].xq.as_slice(),
                            ];
                            let d = unsafe {
                                if q4tp_v1() {
                                    dot_q4tp_row_1x4_sdot_v1(v.nib, r, gpr, xs, &sc)
                                } else {
                                    dot_q4tp_row_1x4_sdot(v.nib, r, gpr, xs, &sc)
                                }
                            };
                            for k in 0..4 {
                                let act = &acts[abase + bi + k];
                                let mut acc = d[k] * act.sx;
                                for &(j, xv) in &act.outliers {
                                    let (w, s) = q4tp_outlier(v.nib, r, gpr, j, &sc);
                                    acc += w * s * xv;
                                }
                                // SAFETY: disjoint (bi, r) cells per worker.
                                unsafe { *out_addr.at((abase + bi + k) * rows + r) = acc };
                            }
                            bi += 4;
                        }
                    }
                    let _ = blocked_ok;
                    while bi < alen {
                        let act = &acts[abase + bi];
                        let mut acc = dot_q4tp_row_i8(v.nib, r, gpr, &act.xq, &sc) * act.sx;
                        for &(j, xv) in &act.outliers {
                            let (w, s) = q4tp_outlier(v.nib, r, gpr, j, &sc);
                            acc += w * s * xv;
                        }
                        // SAFETY: disjoint (bi, r) cells per worker range.
                        unsafe { *out_addr.at((abase + bi) * rows + r) = acc };
                        bi += 1;
                    }
                }
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }

    let run = |start: usize, end: usize| {
        let mut sc = vec![0f32; gpr];
        for r in start..end {
            v.scales_into(r, gpr, &mut sc);
            for bi in 0..b {
                let x = &xs_all[bi * cols..(bi + 1) * cols];
                // SAFETY: disjoint (bi, r) cells per worker range.
                unsafe { *out_addr.at(bi * rows + r) = q4tp_row_exact(v.nib, r, gpr, x, &sc) };
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// Fused q4_tiled matvec (dispatch mirrors `q4matvec`).
fn q4t_matvec(
    bytes: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), rows);
    let gpr = cols / GROUP_SIZE;
    let out_addr = SendMut(out.as_mut_ptr());
    if a8w8_enabled() {
        let act = split_act(x);
        let run = move |start: usize, end: usize| {
            for r in start..end {
                let mut acc = dot_q4t_row_i8(bytes, r, gpr, &act.xq) * act.sx;
                for &(j, xv) in &act.outliers {
                    let (w, s) = q4t_outlier(bytes, r, gpr, j);
                    acc += w * s * xv;
                }
                // SAFETY: disjoint row ranges per worker.
                unsafe { *out_addr.at(r) = acc };
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = move |start: usize, end: usize| {
        for r in start..end {
            // SAFETY: disjoint row ranges per worker.
            unsafe { *out_addr.at(r) = q4t_row_exact(bytes, r, gpr, x) };
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// Fused two-input q4_tiled matvec (weights read once per pair).
#[allow(clippy::too_many_arguments)]
fn q4t_matvec2(
    bytes: &[u8],
    x1: &[f32],
    x2: &[f32],
    rows: usize,
    cols: usize,
    o1: &mut [f32],
    o2: &mut [f32],
    pool: Option<&Pool>,
) {
    let gpr = cols / GROUP_SIZE;
    let p1 = SendMut(o1.as_mut_ptr());
    let p2 = SendMut(o2.as_mut_ptr());
    if a8w8_enabled() {
        let a1 = split_act(x1);
        let a2 = split_act(x2);
        let run = move |start: usize, end: usize| {
            for r in start..end {
                let mut v1 = dot_q4t_row_i8(bytes, r, gpr, &a1.xq) * a1.sx;
                let mut v2 = dot_q4t_row_i8(bytes, r, gpr, &a2.xq) * a2.sx;
                for &(j, xv) in &a1.outliers {
                    let (w, s) = q4t_outlier(bytes, r, gpr, j);
                    v1 += w * s * xv;
                }
                for &(j, xv) in &a2.outliers {
                    let (w, s) = q4t_outlier(bytes, r, gpr, j);
                    v2 += w * s * xv;
                }
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
    let run = move |start: usize, end: usize| {
        for r in start..end {
            // SAFETY: disjoint row ranges per worker.
            unsafe {
                *p1.at(r) = q4t_row_exact(bytes, r, gpr, x1);
                *p2.at(r) = q4t_row_exact(bytes, r, gpr, x2);
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// Batched q4_tiled matmat: each row's tiles stream once per microbatch.
#[allow(clippy::too_many_arguments)]
/// Prefill GEMM through Accelerate for group-quantized codecs: a
/// caller-supplied row dequantizer fills f32 tiles (pool-parallel) and
/// each tile rides the AMX with one sgemm — the generic sibling of
/// `qmatmat_accel` (q8). Numerics are f32-GEMM (tolerance class);
/// decode (b=1) never takes this path.
#[cfg(target_os = "macos")]
fn dequant_matmat_accel(
    dequant_row: &(dyn Fn(usize, &mut [f32]) + Sync),
    xs_all: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    const TR: usize = 2048;
    thread_local! {
        static WTILE: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    WTILE.with(|wt| {
        let mut wtile = wt.borrow_mut();
        wtile.resize(TR * cols, 0.0);
        let mut r0 = 0usize;
        while r0 < rows {
            let tr = TR.min(rows - r0);
            let wt_addr = SendMut(wtile.as_mut_ptr());
            let run = |start: usize, end: usize| {
                for r in start..end {
                    // SAFETY: workers cover disjoint r ranges.
                    let dst = unsafe { std::slice::from_raw_parts_mut(wt_addr.at(r * cols), cols) };
                    dequant_row(r0 + r, dst);
                }
            };
            dispatch_rows(pool, tr, &run);
            unsafe {
                accel_blas::cblas_sgemm(
                    101, // RowMajor
                    111, // NoTrans A
                    112, // Trans B
                    b as i32,
                    tr as i32,
                    cols as i32,
                    1.0,
                    xs_all.as_ptr(),
                    cols as i32,
                    wtile.as_ptr(),
                    cols as i32,
                    0.0,
                    out.as_mut_ptr().add(r0),
                    rows as i32,
                );
            }
            r0 += tr;
        }
    });
}

fn q4t_matmat(
    bytes: &[u8],
    xs_all: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), b * rows);
    let gpr = cols / GROUP_SIZE;
    // Wide batches ride the AMX like q8's qmatmat: on Apple silicon
    // the dequant-tile sgemm is an order above the SDOT row loop for
    // prefill shapes (imagegen DiT forwards are exactly this).
    #[cfg(target_os = "macos")]
    if b >= 8 && rows * cols >= 500_000 && accel_gemm_enabled() {
        dequant_matmat_accel(
            &|r, dst| {
                for gi in 0..gpr {
                    let tile = &bytes[(r * gpr + gi) * Q4_TILE..(r * gpr + gi + 1) * Q4_TILE];
                    let s = f16_to_f32(u16::from_le_bytes([tile[0], tile[1]]));
                    for (k, &bb) in tile[2..].iter().enumerate() {
                        dst[gi * GROUP_SIZE + k * 2] = ((bb & 0x0F) as f32 - 8.0) * s;
                        dst[gi * GROUP_SIZE + k * 2 + 1] = (((bb >> 4) & 0x0F) as f32 - 8.0) * s;
                    }
                }
            },
            xs_all,
            b,
            rows,
            cols,
            out,
            pool,
        );
        return;
    }
    let out_addr = SendMut(out.as_mut_ptr());
    if a8w8_enabled() {
        let acts: Vec<SplitAct> = (0..b)
            .map(|bi| split_act(&xs_all[bi * cols..(bi + 1) * cols]))
            .collect();
        let acts = &acts;
        #[cfg(target_arch = "x86_64")]
        let blocked_ok = avx2_enabled() && blocked_enabled();
        #[cfg(target_arch = "aarch64")]
        let blocked_ok = sdot_enabled() && blocked_enabled();
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let blocked_ok = false;
        let run = move |start: usize, end: usize| {
            for r in start..end {
                let mut bi = 0usize;
                #[cfg(target_arch = "aarch64")]
                if blocked_ok {
                    while bi + 4 <= acts.len() {
                        let xs = [
                            acts[bi].xq.as_slice(),
                            acts[bi + 1].xq.as_slice(),
                            acts[bi + 2].xq.as_slice(),
                            acts[bi + 3].xq.as_slice(),
                        ];
                        let d = unsafe { dot_q4t_row_1x4_sdot(bytes, r, gpr, xs) };
                        for k in 0..4 {
                            let act = &acts[bi + k];
                            let mut acc = d[k] * act.sx;
                            for &(j, xv) in &act.outliers {
                                let (w, sc) = q4t_outlier(bytes, r, gpr, j);
                                acc += w * sc * xv;
                            }
                            // SAFETY: disjoint (bi, r) cells per worker.
                            unsafe { *out_addr.at((bi + k) * rows + r) = acc };
                        }
                        bi += 4;
                    }
                }
                #[cfg(target_arch = "x86_64")]
                if blocked_ok {
                    while bi + 4 <= acts.len() {
                        let xs = [
                            acts[bi].xq.as_slice(),
                            acts[bi + 1].xq.as_slice(),
                            acts[bi + 2].xq.as_slice(),
                            acts[bi + 3].xq.as_slice(),
                        ];
                        let d = unsafe {
                            if vnni_tiles_enabled() {
                                dot_q4t_row_1x4_vnni(bytes, r, gpr, xs)
                            } else {
                                dot_q4t_row_1x4_avx2(bytes, r, gpr, xs)
                            }
                        };
                        for k in 0..4 {
                            let act = &acts[bi + k];
                            let mut acc = d[k] * act.sx;
                            for &(j, xv) in &act.outliers {
                                let (w, sc) = q4t_outlier(bytes, r, gpr, j);
                                acc += w * sc * xv;
                            }
                            // SAFETY: disjoint (bi, r) cells per worker.
                            unsafe { *out_addr.at((bi + k) * rows + r) = acc };
                        }
                        bi += 4;
                    }
                }
                let _ = blocked_ok;
                while bi < acts.len() {
                    let act = &acts[bi];
                    let mut acc = dot_q4t_row_i8(bytes, r, gpr, &act.xq) * act.sx;
                    for &(j, xv) in &act.outliers {
                        let (w, s) = q4t_outlier(bytes, r, gpr, j);
                        acc += w * s * xv;
                    }
                    // SAFETY: disjoint (bi, r) cells per worker range.
                    unsafe { *out_addr.at(bi * rows + r) = acc };
                    bi += 1;
                }
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = move |start: usize, end: usize| {
        for r in start..end {
            for bi in 0..b {
                let x = &xs_all[bi * cols..(bi + 1) * cols];
                // SAFETY: disjoint (bi, r) cells per worker range.
                unsafe { *out_addr.at(bi * rows + r) = q4t_row_exact(bytes, r, gpr, x) };
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

// ── q1 (dtype 12): binary weights, [f16 scale][4B sign bits] per
// 32-group tile. The kernel family mirrors q4_tiled: one sequential
// stream of 6-byte tiles, per-tile integer dot × scale, exact outlier
// correction (A8W8 contract), exact scalar path under CMF_SDOT=0. ──

/// Per-32-group sums of the quantized activation — the ±1 identity's
/// shared half: `dot = −2·sdot(mask, x) − gsum[g]`, computed ONCE per
/// matvec and reused by every row.
fn q1_group_sums(xq: &[i8], gpr: usize) -> Vec<i32> {
    (0..gpr)
        .map(|gi| {
            xq[gi * GROUP_SIZE..(gi + 1) * GROUP_SIZE]
                .iter()
                .map(|&v| v as i32)
                .sum()
        })
        .collect()
}

/// One q1 row via the A8W8 int8 path — mask-SDOT on ARM (no ±1
/// expansion at all), scalar bit loop elsewhere (AVX2 queued with the
/// x86 pass).
#[inline]
#[allow(unreachable_code)]
/// AVX2 q1 row via the same ±1 identity as the ARM sdot kernel: the
/// sign bits expand to a {0, −1} byte mask through shuffle+cmpeq, the
/// masked activation sums through maddubs(1, x&mask), and
/// `dot = −(2·masked_sum + Σx_group)` — bit-identical integer math.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q1_row_avx2(bytes: &[u8], r: usize, gpr: usize, xq: &[i8], gsum: &[i32]) -> f32 {
    // SAFETY: callers uphold the 6B-tile and xq/gsum length contracts.
    unsafe {
        use core::arch::x86_64::*;
        // Byte j of the mask must replicate bits-byte j/8.
        let expand = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3,
        );
        let bitsel = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64,
            -128, 1, 2, 4, 8, 16, 32, 64, -128,
        );
        let ones8 = _mm256_set1_epi8(1);
        let ones16 = _mm256_set1_epi16(1);
        let mut acc = 0f32;
        for gi in 0..gpr {
            let t = bytes.as_ptr().add((r * gpr + gi) * Q1_TILE);
            let s = f16_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
            let bits = u32::from_le_bytes([*t.add(2), *t.add(3), *t.add(4), *t.add(5)]);
            let bc = _mm256_shuffle_epi8(_mm256_set1_epi32(bits as i32), expand);
            let mask = _mm256_cmpeq_epi8(_mm256_and_si256(bc, bitsel), bitsel);
            let x = _mm256_loadu_si256(xq.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
            let sel = _mm256_and_si256(x, mask);
            // Σ of selected i8 lanes: maddubs(1u8, sel_i8) pairs → madd.
            let p16 = _mm256_maddubs_epi16(ones8, sel);
            let d32 = _mm256_madd_epi16(p16, ones16);
            let hi128 = _mm256_extracti128_si256::<1>(d32);
            let s128 = _mm_add_epi32(_mm256_castsi256_si128(d32), hi128);
            let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
            let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
            let msum = _mm_cvtsi128_si32(s32);
            // The and-select keeps x UN-negated (unlike ARM's −1-mask
            // sdot): d = Σ_set − Σ_unset = 2·Σ_set − Σ_all.
            let d = 2 * msum - gsum[gi];
            acc += d as f32 * s;
        }
        acc
    }
}

/// VNNI twin of `dot_q1_row_avx2`: the masked-select sum goes through
/// one `vpdpbusd(1u8, sel)` (see `dpbusd_hsum` — bit-identical).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn dot_q1_row_vnni(bytes: &[u8], r: usize, gpr: usize, xq: &[i8], gsum: &[i32]) -> f32 {
    // SAFETY: callers uphold the 6B-tile and xq/gsum length contracts.
    unsafe {
        use core::arch::x86_64::*;
        let expand = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3,
        );
        let bitsel = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64,
            -128, 1, 2, 4, 8, 16, 32, 64, -128,
        );
        let ones8 = _mm256_set1_epi8(1);
        let mut acc = 0f32;
        for gi in 0..gpr {
            let t = bytes.as_ptr().add((r * gpr + gi) * Q1_TILE);
            let s = f16_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
            let bits = u32::from_le_bytes([*t.add(2), *t.add(3), *t.add(4), *t.add(5)]);
            let bc = _mm256_shuffle_epi8(_mm256_set1_epi32(bits as i32), expand);
            let mask = _mm256_cmpeq_epi8(_mm256_and_si256(bc, bitsel), bitsel);
            let x = _mm256_loadu_si256(xq.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
            let msum = dpbusd_hsum(ones8, _mm256_and_si256(x, mask));
            let d = 2 * msum - gsum[gi];
            acc += d as f32 * s;
        }
        acc
    }
}

/// VNNI twin of `dot_q1_row_1x4_avx2` (see `dpbusd_hsum`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn dot_q1_row_1x4_vnni(
    bytes: &[u8],
    r: usize,
    gpr: usize,
    xs: [&[i8]; 4],
    gsums: [&[i32]; 4],
) -> [f32; 4] {
    // SAFETY: callers uphold the 6B-tile and xq/gsum length contracts.
    unsafe {
        use core::arch::x86_64::*;
        let expand = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3,
        );
        let bitsel = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64,
            -128, 1, 2, 4, 8, 16, 32, 64, -128,
        );
        let ones8 = _mm256_set1_epi8(1);
        let mut acc = [0f32; 4];
        for gi in 0..gpr {
            let t = bytes.as_ptr().add((r * gpr + gi) * Q1_TILE);
            let s = f16_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
            let bits = u32::from_le_bytes([*t.add(2), *t.add(3), *t.add(4), *t.add(5)]);
            let bc = _mm256_shuffle_epi8(_mm256_set1_epi32(bits as i32), expand);
            let mask = _mm256_cmpeq_epi8(_mm256_and_si256(bc, bitsel), bitsel);
            for (k, xq) in xs.iter().enumerate() {
                let x = _mm256_loadu_si256(xq.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
                let msum = dpbusd_hsum(ones8, _mm256_and_si256(x, mask));
                let d = 2 * msum - gsums[k][gi];
                acc[k] += d as f32 * s;
            }
        }
        acc
    }
}

/// The blocked 1×4 flavor: the expanded bit mask serves four activation
/// streams per group (mask build once, four select+reduce chains).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q1_row_1x4_avx2(
    bytes: &[u8],
    r: usize,
    gpr: usize,
    xs: [&[i8]; 4],
    gsums: [&[i32]; 4],
) -> [f32; 4] {
    // SAFETY: callers uphold the 6B-tile and xq/gsum length contracts.
    unsafe {
        use core::arch::x86_64::*;
        let expand = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3,
        );
        let bitsel = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64,
            -128, 1, 2, 4, 8, 16, 32, 64, -128,
        );
        let ones8 = _mm256_set1_epi8(1);
        let ones16 = _mm256_set1_epi16(1);
        let mut acc = [0f32; 4];
        for gi in 0..gpr {
            let t = bytes.as_ptr().add((r * gpr + gi) * Q1_TILE);
            let s = f16_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
            let bits = u32::from_le_bytes([*t.add(2), *t.add(3), *t.add(4), *t.add(5)]);
            let bc = _mm256_shuffle_epi8(_mm256_set1_epi32(bits as i32), expand);
            let mask = _mm256_cmpeq_epi8(_mm256_and_si256(bc, bitsel), bitsel);
            for (k, xq) in xs.iter().enumerate() {
                let x = _mm256_loadu_si256(xq.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
                let sel = _mm256_and_si256(x, mask);
                let p16 = _mm256_maddubs_epi16(ones8, sel);
                let d32 = _mm256_madd_epi16(p16, ones16);
                let hi128 = _mm256_extracti128_si256::<1>(d32);
                let s128 = _mm_add_epi32(_mm256_castsi256_si128(d32), hi128);
                let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
                let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
                let msum = _mm_cvtsi128_si32(s32);
                let d = 2 * msum - gsums[k][gi];
                acc[k] += d as f32 * s;
            }
        }
        acc
    }
}

#[allow(unreachable_code)]
fn dot_q1_row_i8(bytes: &[u8], r: usize, gpr: usize, xq: &[i8], gsum: &[i32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_q1_row_sdot(bytes, r, gpr, xq, gsum);
    }
    #[cfg(target_arch = "x86_64")]
    if avx2_enabled() {
        unsafe {
            if vnni_tiles_enabled() {
                return dot_q1_row_vnni(bytes, r, gpr, xq, gsum);
            }
            return dot_q1_row_avx2(bytes, r, gpr, xq, gsum);
        }
    }
    let _ = gsum;
    let mut acc = 0f32;
    for gi in 0..gpr {
        let tile = &bytes[(r * gpr + gi) * Q1_TILE..(r * gpr + gi + 1) * Q1_TILE];
        let s = f16_to_f32(u16::from_le_bytes([tile[0], tile[1]]));
        let mut d = 0i32;
        for (j, &b) in tile[2..].iter().enumerate() {
            for k in 0..8 {
                let w = ((b >> k) & 1) as i32 * 2 - 1;
                d += w * xq[gi * GROUP_SIZE + j * 8 + k] as i32;
            }
        }
        acc += d as f32 * s;
    }
    acc
}

/// SDOT q1 row via the ±1 identity: the vtst mask (0xFF where the bit
/// is set, i.e. −1 as i8) feeds `sdot` DIRECTLY — no expansion to ±1
/// lanes at all — and `dot = −(2·sdot(mask, x) + Σx_group)`, with the
/// per-group activation sums shared across every row of the matvec.
/// Four tiles (128 weights) per iteration: integer dots reduce through
/// a vpaddq tree into ONE i32x4 that meets its four scales in a single
/// fused f32 multiply-add. Integer math throughout — bit-identical to
/// the scalar ±1 reference.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q1_row_sdot(bytes: &[u8], r: usize, gpr: usize, xq: &[i8], gsum: &[i32]) -> f32 {
    // SAFETY: callers uphold slice-length contracts (6B tile per group,
    // xq.len() == gpr·GROUP_SIZE, gsum.len() == gpr).
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        const MASKS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let m = vld1q_u8(MASKS.as_ptr());
        // One tile's −Σ_set(x) as an UNREDUCED i32x4 (two mask-sdots).
        macro_rules! tile_dot {
            ($t:expr, $x:expr) => {{
                let v0 = vcombine_u8(vdup_n_u8(*$t.add(2)), vdup_n_u8(*$t.add(3)));
                let v1 = vcombine_u8(vdup_n_u8(*$t.add(4)), vdup_n_u8(*$t.add(5)));
                let w0 = vreinterpretq_s8_u8(vtstq_u8(v0, m));
                let w1 = vreinterpretq_s8_u8(vtstq_u8(v1, m));
                let x0 = vld1q_s8($x);
                let x1 = vld1q_s8($x.add(16));
                let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
                asm!(
                    "sdot {a0:v}.4s, {w0:v}.16b, {x0:v}.16b",
                    "sdot {a1:v}.4s, {w1:v}.16b, {x1:v}.16b",
                    a0 = inout(vreg) a0, a1 = inout(vreg) a1,
                    w0 = in(vreg) w0, x0 = in(vreg) x0, w1 = in(vreg) w1, x1 = in(vreg) x1,
                    options(pure, nomem, nostack),
                );
                vaddq_s32(a0, a1)
            }};
        }
        // TBL unpack over PAIR loads: one vld1q covers two 6B tiles
        // ([s s b b b b][s s b b b b] + 4B slack), TBL replicates each
        // bit-byte across 8 lanes for vtst, and the four scales gather
        // through tbl2 into one fcvtl — the 16 ld1r broadcast loads and
        // 4 branchy software f16 conversions per 128 weights (the
        // measured load-port wall of this kernel) become 2 vector
        // loads + 9 table lookups. Integer math order is unchanged —
        // bit-identical results (FCVTL is exact on every f16).
        const IW00: [u8; 16] = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3];
        const IW01: [u8; 16] = [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5];
        const IW10: [u8; 16] = [8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9];
        const IW11: [u8; 16] = [
            10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        ];
        const ISC: [u8; 8] = [0, 1, 6, 7, 16, 17, 22, 23];
        let (iw00, iw01) = (vld1q_u8(IW00.as_ptr()), vld1q_u8(IW01.as_ptr()));
        let (iw10, iw11) = (vld1q_u8(IW10.as_ptr()), vld1q_u8(IW11.as_ptr()));
        let isc = vld1_u8(ISC.as_ptr());
        // One tile's −Σ_set(x) from a TBL-unpacked pair load.
        macro_rules! tile_dot_tbl {
            ($ld:expr, $i0:expr, $i1:expr, $x:expr) => {{
                let w0 = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8($ld, $i0), m));
                let w1 = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8($ld, $i1), m));
                let x0 = vld1q_s8($x);
                let x1 = vld1q_s8($x.add(16));
                let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
                asm!(
                    "sdot {a0:v}.4s, {w0:v}.16b, {x0:v}.16b",
                    "sdot {a1:v}.4s, {w1:v}.16b, {x1:v}.16b",
                    a0 = inout(vreg) a0, a1 = inout(vreg) a1,
                    w0 = in(vreg) w0, x0 = in(vreg) x0, w1 = in(vreg) w1, x1 = in(vreg) x1,
                    options(pure, nomem, nostack),
                );
                vaddq_s32(a0, a1)
            }};
        }
        let base = bytes.as_ptr().add(r * gpr * Q1_TILE);
        let row_base = r * gpr * Q1_TILE;
        let abs_end = bytes.len();
        let xp = xq.as_ptr();
        let gp = gsum.as_ptr();
        let mut accv = vdupq_n_f32(0.0);
        let mut gi = 0;
        // The second pair load reads 4B past tile gi+3 — stay inside
        // the payload slice (only the file's final tiles fall back).
        while gi + 4 <= gpr && row_base + (gi + 4) * Q1_TILE + 4 <= abs_end {
            let t0 = base.add(gi * Q1_TILE);
            let ld_a = vld1q_u8(t0);
            let ld_b = vld1q_u8(t0.add(2 * Q1_TILE));
            let d0 = tile_dot_tbl!(ld_a, iw00, iw01, xp.add(gi * GROUP_SIZE));
            let d1 = tile_dot_tbl!(ld_a, iw10, iw11, xp.add((gi + 1) * GROUP_SIZE));
            let d2 = tile_dot_tbl!(ld_b, iw00, iw01, xp.add((gi + 2) * GROUP_SIZE));
            let d3 = tile_dot_tbl!(ld_b, iw10, iw11, xp.add((gi + 3) * GROUP_SIZE));
            // [−Σ0, −Σ1, −Σ2, −Σ3] → dots = −(2·Σset_neg + gsum)
            let neg = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
            let g = vld1q_s32(gp.add(gi));
            let dots = vnegq_s32(vaddq_s32(vshlq_n_s32::<1>(neg), g));
            let sc16 = vqtbl2_u8(uint8x16x2_t(ld_a, ld_b), isc);
            let scf: float32x4_t;
            asm!(
                "fcvtl {o:v}.4s, {i:v}.4h",
                o = out(vreg) scf, i = in(vreg) sc16,
                options(pure, nomem, nostack),
            );
            accv = vfmaq_f32(accv, vcvtq_f32_s32(dots), scf);
            gi += 4;
        }
        let mut acc = vaddvq_f32(accv);
        while gi < gpr {
            let t = base.add(gi * Q1_TILE);
            let s = f16_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
            let d = vaddvq_s32(tile_dot!(t, xp.add(gi * GROUP_SIZE)));
            acc += (-(2 * d + *gp.add(gi))) as f32 * s;
            gi += 1;
        }
        acc
    }
}

/// Blocked q1 1×4: one TBL unpack of the tile pair serves FOUR
/// activation streams (prefill amortization — the same idea as the
/// AVX2 twin; per stream the group order, fma order and tail match the
/// single-row kernel exactly, so batch == matvec bit-for-bit).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q1_row_1x4_sdot(
    bytes: &[u8],
    r: usize,
    gpr: usize,
    xs: [&[i8]; 4],
    gs: [&[i32]; 4],
) -> [f32; 4] {
    // SAFETY: same slice-length contracts as `dot_q1_row_sdot`, ×4.
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        const MASKS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        const IW00: [u8; 16] = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3];
        const IW01: [u8; 16] = [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5];
        const IW10: [u8; 16] = [8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9];
        const IW11: [u8; 16] = [
            10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        ];
        const ISC: [u8; 8] = [0, 1, 6, 7, 16, 17, 22, 23];
        let m = vld1q_u8(MASKS.as_ptr());
        let (iw00, iw01) = (vld1q_u8(IW00.as_ptr()), vld1q_u8(IW01.as_ptr()));
        let (iw10, iw11) = (vld1q_u8(IW10.as_ptr()), vld1q_u8(IW11.as_ptr()));
        let isc = vld1_u8(ISC.as_ptr());
        macro_rules! sdot2 {
            ($w0:expr, $w1:expr, $x:expr) => {{
                let x0 = vld1q_s8($x);
                let x1 = vld1q_s8($x.add(16));
                let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
                asm!(
                    "sdot {a0:v}.4s, {w0:v}.16b, {x0:v}.16b",
                    "sdot {a1:v}.4s, {w1:v}.16b, {x1:v}.16b",
                    a0 = inout(vreg) a0, a1 = inout(vreg) a1,
                    w0 = in(vreg) $w0, x0 = in(vreg) x0, w1 = in(vreg) $w1, x1 = in(vreg) x1,
                    options(pure, nomem, nostack),
                );
                vaddq_s32(a0, a1)
            }};
        }
        let base = bytes.as_ptr().add(r * gpr * Q1_TILE);
        let row_base = r * gpr * Q1_TILE;
        let abs_end = bytes.len();
        let mut accv = [vdupq_n_f32(0.0); 4];
        let mut gi = 0;
        while gi + 4 <= gpr && row_base + (gi + 4) * Q1_TILE + 4 <= abs_end {
            let t0 = base.add(gi * Q1_TILE);
            let ld_a = vld1q_u8(t0);
            let ld_b = vld1q_u8(t0.add(2 * Q1_TILE));
            // Unpack ONCE — eight ±mask vectors serve all four streams.
            let w00 = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(ld_a, iw00), m));
            let w01 = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(ld_a, iw01), m));
            let w10 = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(ld_a, iw10), m));
            let w11 = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(ld_a, iw11), m));
            let w20 = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(ld_b, iw00), m));
            let w21 = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(ld_b, iw01), m));
            let w30 = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(ld_b, iw10), m));
            let w31 = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(ld_b, iw11), m));
            let sc16 = vqtbl2_u8(uint8x16x2_t(ld_a, ld_b), isc);
            let scf: float32x4_t;
            asm!(
                "fcvtl {o:v}.4s, {i:v}.4h",
                o = out(vreg) scf, i = in(vreg) sc16,
                options(pure, nomem, nostack),
            );
            for k in 0..4 {
                let xp = xs[k].as_ptr();
                let d0 = sdot2!(w00, w01, xp.add(gi * GROUP_SIZE));
                let d1 = sdot2!(w10, w11, xp.add((gi + 1) * GROUP_SIZE));
                let d2 = sdot2!(w20, w21, xp.add((gi + 2) * GROUP_SIZE));
                let d3 = sdot2!(w30, w31, xp.add((gi + 3) * GROUP_SIZE));
                let neg = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
                let g = vld1q_s32(gs[k].as_ptr().add(gi));
                let dots = vnegq_s32(vaddq_s32(vshlq_n_s32::<1>(neg), g));
                accv[k] = vfmaq_f32(accv[k], vcvtq_f32_s32(dots), scf);
            }
            gi += 4;
        }
        let mut acc = [
            vaddvq_f32(accv[0]),
            vaddvq_f32(accv[1]),
            vaddvq_f32(accv[2]),
            vaddvq_f32(accv[3]),
        ];
        while gi < gpr {
            let t = base.add(gi * Q1_TILE);
            let sc = f16_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
            let v0 = vcombine_u8(vdup_n_u8(*t.add(2)), vdup_n_u8(*t.add(3)));
            let v1 = vcombine_u8(vdup_n_u8(*t.add(4)), vdup_n_u8(*t.add(5)));
            let w0 = vreinterpretq_s8_u8(vtstq_u8(v0, m));
            let w1 = vreinterpretq_s8_u8(vtstq_u8(v1, m));
            for k in 0..4 {
                let d = vaddvq_s32(sdot2!(w0, w1, xs[k].as_ptr().add(gi * GROUP_SIZE)));
                acc[k] += (-(2 * d + *gs[k].as_ptr().add(gi))) as f32 * sc;
            }
            gi += 1;
        }
        acc
    }
}

/// (weight ±1, scale) of one q1 element — the exact outlier term.
#[inline]
fn q1_outlier(bytes: &[u8], r: usize, gpr: usize, j: usize) -> (f32, f32) {
    let gi = j / GROUP_SIZE;
    let k = j % GROUP_SIZE;
    let tile = &bytes[(r * gpr + gi) * Q1_TILE..(r * gpr + gi + 1) * Q1_TILE];
    let s = f16_to_f32(u16::from_le_bytes([tile[0], tile[1]]));
    let bit = (tile[2 + k / 8] >> (k % 8)) & 1;
    ((bit as i32 * 2 - 1) as f32, s)
}

/// Exact scalar q1 row (CMF_SDOT=0 contract).
#[inline]
fn q1_row_exact(bytes: &[u8], r: usize, gpr: usize, x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for gi in 0..gpr {
        let tile = &bytes[(r * gpr + gi) * Q1_TILE..(r * gpr + gi + 1) * Q1_TILE];
        let s = f16_to_f32(u16::from_le_bytes([tile[0], tile[1]]));
        let xg = &x[gi * GROUP_SIZE..(gi + 1) * GROUP_SIZE];
        let mut ga = 0f32;
        for (j, &b) in tile[2..].iter().enumerate() {
            for k in 0..8 {
                ga += (((b >> k) & 1) as f32 * 2.0 - 1.0) * xg[j * 8 + k];
            }
        }
        acc += ga * s;
    }
    acc
}

/// One q1 row range via A8W8 (the body of `q1_matvec`'s hot loop,
/// extracted so multi-matrix jobs drive the same kernel).
#[allow(clippy::too_many_arguments)]
fn q1_range_a8w8(
    bytes: &[u8],
    gpr: usize,
    act: &SplitAct,
    gsum: &[i32],
    out: SendMut,
    start: usize,
    end: usize,
) {
    for r in start..end {
        let mut acc = dot_q1_row_i8(bytes, r, gpr, &act.xq, gsum) * act.sx;
        for &(j, xv) in &act.outliers {
            let (w, s) = q1_outlier(bytes, r, gpr, j);
            acc += w * s * xv;
        }
        // SAFETY: disjoint row ranges per worker.
        unsafe { *out.at(r) = acc };
    }
}

/// Exact-scalar q1 row range (CMF_SDOT=0 contract).
fn q1_range_f32(bytes: &[u8], gpr: usize, x: &[f32], out: SendMut, start: usize, end: usize) {
    for r in start..end {
        // SAFETY: disjoint row ranges per worker.
        unsafe { *out.at(r) = q1_row_exact(bytes, r, gpr, x) };
    }
}

/// q1t per-row overlay locator. After the base (`base_len`) come
/// `[u32 row_ptr[rows+1]]` then `[(u16 col, f16 val)]` grouped by row (row
/// `r`'s entries are `[row_ptr[r], row_ptr[r+1])`). Returns
/// `(row_ptr offset, entries offset, present)`.
fn q1t_overlay(bytes: &[u8], base_len: usize, rows: usize) -> (usize, usize, bool) {
    let entries = base_len + (rows + 1) * 4;
    (base_len, entries, entries <= bytes.len())
}

/// Read `row_ptr[r]` from the overlay's prefix-sum table.
#[inline]
fn q1t_rowptr(bytes: &[u8], rp_off: usize, r: usize) -> usize {
    let o = rp_off + r * 4;
    u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize
}

/// Byte → the 5 ternary signs it packs `{−1,0,+1}` as f32, precomputed so
/// decoding a q1t code is a table load, not the base-3 divide/modulo per
/// weight (division is ~20–40× the cost of a load). Built at compile time.
const SIGN5: [[f32; 5]; 256] = {
    let mut lut = [[0.0f32; 5]; 256];
    let pow3 = [1u16, 3, 9, 27, 81];
    let mut byte = 0usize;
    while byte < 256 {
        let mut i = 0usize;
        while i < 5 {
            let code = (byte as u16 / pow3[i]) % 3;
            lut[byte][i] = if code == 1 {
                1.0
            } else if code == 2 {
                -1.0
            } else {
                0.0
            };
            i += 1;
        }
        byte += 1;
    }
    lut
};

/// Same table, as i8 signs — the operand for the int8 SDOT base kernel.
const SIGN5_I8: [[i8; 5]; 256] = {
    let mut lut = [[0i8; 5]; 256];
    let pow3 = [1u16, 3, 9, 27, 81];
    let mut byte = 0usize;
    while byte < 256 {
        let mut i = 0usize;
        while i < 5 {
            let code = (byte as u16 / pow3[i]) % 3;
            lut[byte][i] = if code == 1 {
                1
            } else if code == 2 {
                -1
            } else {
                0
            };
            i += 1;
        }
        byte += 1;
    }
    lut
};

/// The same 5 i8 signs packed into a u64 (`[s0 s1 s2 s3 s4 0 0 0]`, LE) so the
/// group unpack is 7 unaligned u64 stores at offsets 0,5,10,…,30 instead of
/// six 5-byte copies + LUT indexing — each store's trailing zeros are fixed by
/// the next store, and the last one runs 6 B past the 32nd weight (the unpack
/// buffer is padded to 40). This is the decode/prefill hot inner op.
const SIGN5_U64: [u64; 256] = {
    let mut lut = [0u64; 256];
    let pow3 = [1u16, 3, 9, 27, 81];
    let mut byte = 0usize;
    while byte < 256 {
        let mut v = 0u64;
        let mut i = 0usize;
        while i < 5 {
            let code = (byte as u16 / pow3[i]) % 3;
            let s: u8 = if code == 1 {
                1
            } else if code == 2 {
                0xFF
            } else {
                0
            };
            v |= (s as u64) << (i * 8);
            i += 1;
        }
        lut[byte] = v;
        byte += 1;
    }
    lut
};

/// Ternary base weight at `(row r, col j)` = `sign(code)·s_group`. Used to add
/// back activation-outlier columns, whose `x` was zeroed for the int8 bulk dot
/// (`split_act`). At a weight-outlier position the code is 0, so this is 0 and
/// the overlay correction owns that column — no double counting.
#[inline]
fn q1t_base_weight(bytes: &[u8], r: usize, gpr: usize, j: usize) -> f32 {
    const TILE: usize = cortiq_core::quant::Q1T_TILE;
    let off = (r * gpr + j / GROUP_SIZE) * TILE;
    let s = f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
    let within = j % GROUP_SIZE;
    SIGN5[bytes[off + 2 + within / 5] as usize][within % 5] * s
}

/// One 32-group int8 dot via two SDOTs. Bit-exact vs the scalar i8 sum
/// (integer accumulation is order-independent).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn sdot32_i8(w: *const i8, x: *const i8) -> i32 {
    // SAFETY: caller guarantees 32 readable i8 at each pointer.
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let w0 = vld1q_s8(w);
        let w1 = vld1q_s8(w.add(16));
        let x0 = vld1q_s8(x);
        let x1 = vld1q_s8(x.add(16));
        let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
        asm!(
            "sdot {a0:v}.4s, {w0:v}.16b, {x0:v}.16b",
            "sdot {a1:v}.4s, {w1:v}.16b, {x1:v}.16b",
            a0 = inout(vreg) a0, a1 = inout(vreg) a1,
            w0 = in(vreg) w0, x0 = in(vreg) x0, w1 = in(vreg) w1, x1 = in(vreg) x1,
            options(pure, nomem, nostack),
        );
        vaddvq_s32(vaddq_s32(a0, a1))
    }
}

/// One 32-group int8 dot via AVX2: signed·signed as `maddubs(|w|, sign(x,w))`
/// then `madd` and a horizontal reduce (the same idiom as `dot_q4t_row_avx2`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn i8dot32_avx2(w: *const i8, x: *const i8) -> i32 {
    // SAFETY: caller guarantees 32 readable i8 at each pointer.
    unsafe {
        use core::arch::x86_64::*;
        let wv = _mm256_loadu_si256(w as *const __m256i);
        let xv = _mm256_loadu_si256(x as *const __m256i);
        let p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(xv, wv));
        let d = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
        let hi128 = _mm256_extracti128_si256::<1>(d);
        let s128 = _mm_add_epi32(_mm256_castsi256_si128(d), hi128);
        let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
        let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
        _mm_cvtsi128_si32(s32)
    }
}

/// Unpack one q1t group's base-3 codes into 32 i8 signs via 7 unaligned u64
/// stores (see `SIGN5_U64`). `dst` MUST have ≥ 40 bytes: the 7th store writes
/// `dst[30..38]`. Stores go in order so each one's trailing zeros are
/// overwritten by the next; the final 6 padding bytes are unused by the dot.
#[inline]
fn q1t_unpack_group_i8(codes: *const u8, dst: &mut [i8]) {
    debug_assert!(dst.len() >= 40);
    // SAFETY: codes points at 7 readable bytes; dst has ≥ 40 bytes so every
    // 8-byte store at offset bi*5 (bi ≤ 6 → ≤ 30) stays in bounds.
    unsafe {
        let p = dst.as_mut_ptr();
        for bi in 0..7 {
            core::ptr::write_unaligned(
                p.add(bi * 5) as *mut u64,
                SIGN5_U64[*codes.add(bi) as usize],
            );
        }
    }
}

/// One 32-group int8 dot, arch-dispatched (the matmat inner loop, where the
/// row's signs are unpacked once and dotted against every batch input).
/// Callers are gated by `a8w8_enabled()`, so the target-feature arms are
/// reachable; the scalar arm is a non-SIMD-arch fallback.
#[inline]
fn q1t_i8dot32(w: *const i8, x: *const i8) -> i32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return sdot32_i8(w, x);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return i8dot32_avx2(w, x);
    }
    #[allow(unreachable_code)]
    unsafe {
        let mut s = 0i32;
        for k in 0..GROUP_SIZE {
            s += *w.add(k) as i32 * *x.add(k) as i32;
        }
        s
    }
}

#[inline]
unsafe fn q1t_unpack_reg_u64s(codes: *const u8) -> (u64, u64, u64, u64) {
    let (s0, s1, s2, s3, s4, s5, s6) = unsafe {
        (
            SIGN5_U64[*codes as usize],
            SIGN5_U64[*codes.add(1) as usize],
            SIGN5_U64[*codes.add(2) as usize],
            SIGN5_U64[*codes.add(3) as usize],
            SIGN5_U64[*codes.add(4) as usize],
            SIGN5_U64[*codes.add(5) as usize],
            SIGN5_U64[*codes.add(6) as usize],
        )
    };

    let u0 = s0 | (s1 << 40);
    let u1 = (s1 >> 24) | (s2 << 16) | (s3 << 56);
    let u2 = (s3 >> 8) | (s4 << 32);
    let u3 = (s4 >> 32) | (s5 << 8) | (s6 << 48);

    (u0, u1, u2, u3)
}

/// One q1t row's int8 base dot: `Σ_group s·dot(signs, xq)` (before the shared
/// `sx`). Direct register unpacking (zero stack stores/loads, no STLF stalls).
/// ARM SDOT.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q1t_dot_row_sdot(bytes: &[u8], r: usize, gpr: usize, xq: &[i8]) -> f32 {
    use core::arch::aarch64::*;
    use core::arch::asm;
    unsafe {
        const TILE: usize = cortiq_core::quant::Q1T_TILE;
        let mut acc = 0f32;
        let bytes_ptr = bytes.as_ptr();
        let xq_ptr = xq.as_ptr();
        let row_off = r * gpr * TILE;

        let gpr2 = gpr & !1;
        let mut gi = 0;
        while gi < gpr2 {
            let off0 = row_off + gi * TILE;
            let off1 = off0 + TILE;
            let s0 = f16_to_f32(u16::from_le_bytes([
                *bytes_ptr.add(off0),
                *bytes_ptr.add(off0 + 1),
            ]));
            let s1 = f16_to_f32(u16::from_le_bytes([
                *bytes_ptr.add(off1),
                *bytes_ptr.add(off1 + 1),
            ]));

            let (u0_0, u1_0, u2_0, u3_0) = q1t_unpack_reg_u64s(bytes_ptr.add(off0 + 2));
            let (u0_1, u1_1, u2_1, u3_1) = q1t_unpack_reg_u64s(bytes_ptr.add(off1 + 2));

            let w0_0 = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(u0_0), vcreate_u64(u1_0)));
            let w1_0 = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(u2_0), vcreate_u64(u3_0)));
            let w0_1 = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(u0_1), vcreate_u64(u1_1)));
            let w1_1 = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(u2_1), vcreate_u64(u3_1)));

            let x0_0 = vld1q_s8(xq_ptr.add(gi * GROUP_SIZE));
            let x1_0 = vld1q_s8(xq_ptr.add(gi * GROUP_SIZE + 16));
            let x0_1 = vld1q_s8(xq_ptr.add((gi + 1) * GROUP_SIZE));
            let x1_1 = vld1q_s8(xq_ptr.add((gi + 1) * GROUP_SIZE + 16));

            let (mut a0_0, mut a1_0) = (vdupq_n_s32(0), vdupq_n_s32(0));
            let (mut a0_1, mut a1_1) = (vdupq_n_s32(0), vdupq_n_s32(0));
            asm!(
                "sdot {a0_0:v}.4s, {w0_0:v}.16b, {x0_0:v}.16b",
                "sdot {a1_0:v}.4s, {w1_0:v}.16b, {x1_0:v}.16b",
                "sdot {a0_1:v}.4s, {w0_1:v}.16b, {x0_1:v}.16b",
                "sdot {a1_1:v}.4s, {w1_1:v}.16b, {x1_1:v}.16b",
                a0_0 = inout(vreg) a0_0, a1_0 = inout(vreg) a1_0,
                a0_1 = inout(vreg) a0_1, a1_1 = inout(vreg) a1_1,
                w0_0 = in(vreg) w0_0, x0_0 = in(vreg) x0_0, w1_0 = in(vreg) w1_0, x1_0 = in(vreg) x1_0,
                w0_1 = in(vreg) w0_1, x0_1 = in(vreg) x0_1, w1_1 = in(vreg) w1_1, x1_1 = in(vreg) x1_1,
                options(pure, nomem, nostack),
            );
            let d0 = vaddvq_s32(vaddq_s32(a0_0, a1_0));
            let d1 = vaddvq_s32(vaddq_s32(a0_1, a1_1));
            acc += d0 as f32 * s0 + d1 as f32 * s1;
            gi += 2;
        }

        if gi < gpr {
            let off = row_off + gi * TILE;
            let s = f16_to_f32(u16::from_le_bytes([
                *bytes_ptr.add(off),
                *bytes_ptr.add(off + 1),
            ]));
            let (u0, u1, u2, u3) = q1t_unpack_reg_u64s(bytes_ptr.add(off + 2));
            let w0 = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(u0), vcreate_u64(u1)));
            let w1 = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(u2), vcreate_u64(u3)));
            let x0 = vld1q_s8(xq_ptr.add(gi * GROUP_SIZE));
            let x1 = vld1q_s8(xq_ptr.add(gi * GROUP_SIZE + 16));
            let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
            asm!(
                "sdot {a0:v}.4s, {w0:v}.16b, {x0:v}.16b",
                "sdot {a1:v}.4s, {w1:v}.16b, {x1:v}.16b",
                a0 = inout(vreg) a0, a1 = inout(vreg) a1,
                w0 = in(vreg) w0, x0 = in(vreg) x0, w1 = in(vreg) w1, x1 = in(vreg) x1,
                options(pure, nomem, nostack),
            );
            let d = vaddvq_s32(vaddq_s32(a0, a1));
            acc += d as f32 * s;
        }
        acc
    }
}

/// x86 AVX2 mirror of `q1t_dot_row_sdot` (maddubs int8 dot per group).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q1t_dot_row_avx2(bytes: &[u8], r: usize, gpr: usize, xq: &[i8]) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        const TILE: usize = cortiq_core::quant::Q1T_TILE;
        let mut acc = 0f32;
        let bytes_ptr = bytes.as_ptr();
        let xq_ptr = xq.as_ptr();
        let row_off = r * gpr * TILE;

        let ones = _mm256_set1_epi16(1);
        for gi in 0..gpr {
            let off = row_off + gi * TILE;
            let s = f16_to_f32(u16::from_le_bytes([
                *bytes_ptr.add(off),
                *bytes_ptr.add(off + 1),
            ]));
            let (u0, u1, u2, u3) = q1t_unpack_reg_u64s(bytes_ptr.add(off + 2));
            let wv = _mm256_set_epi64x(u3 as i64, u2 as i64, u1 as i64, u0 as i64);
            let xv = _mm256_loadu_si256(xq_ptr.add(gi * GROUP_SIZE) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(xv, wv));
            let d256 = _mm256_madd_epi16(p16, ones);
            let d128 = _mm_add_epi32(
                _mm256_castsi256_si128(d256),
                _mm256_extracti128_si256(d256, 1),
            );
            let d64 = _mm_add_epi32(d128, _mm_shuffle_epi32(d128, 0xee));
            let d32 = _mm_cvtsi128_si32(_mm_add_epi32(d64, _mm_shuffle_epi32(d64, 0x55)));
            acc += d32 as f32 * s;
        }
        acc
    }
}

/// VNNI twin of `q1t_dot_row_avx2` (see `dpbusd_hsum`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn q1t_dot_row_vnni(bytes: &[u8], r: usize, gpr: usize, xq: &[i8]) -> f32 {
    use core::arch::x86_64::*;
    // SAFETY: same tile/xq contracts as `q1t_dot_row_avx2`.
    unsafe {
        const TILE: usize = cortiq_core::quant::Q1T_TILE;
        let mut acc = 0f32;
        let bytes_ptr = bytes.as_ptr();
        let xq_ptr = xq.as_ptr();
        let row_off = r * gpr * TILE;
        for gi in 0..gpr {
            let off = row_off + gi * TILE;
            let s = f16_to_f32(u16::from_le_bytes([
                *bytes_ptr.add(off),
                *bytes_ptr.add(off + 1),
            ]));
            let (u0, u1, u2, u3) = q1t_unpack_reg_u64s(bytes_ptr.add(off + 2));
            let wv = _mm256_set_epi64x(u3 as i64, u2 as i64, u1 as i64, u0 as i64);
            let xv = _mm256_loadu_si256(xq_ptr.add(gi * GROUP_SIZE) as *const __m256i);
            let d = dpbusd_hsum(_mm256_abs_epi8(wv), _mm256_sign_epi8(xv, wv));
            acc += d as f32 * s;
        }
        acc
    }
}

/// Per-row int8 base dot, dispatched once per row (matvec decode hot path).
/// Callers are gated by `a8w8_enabled()`, so the target-feature kernels are
/// reachable.
#[inline]
fn q1t_dot_row_i8(bytes: &[u8], r: usize, gpr: usize, xq: &[i8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return q1t_dot_row_sdot(bytes, r, gpr, xq);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if vnni_tiles_enabled() {
            return q1t_dot_row_vnni(bytes, r, gpr, xq);
        }
        return q1t_dot_row_avx2(bytes, r, gpr, xq);
    }
    #[allow(unreachable_code)]
    {
        const TILE: usize = cortiq_core::quant::Q1T_TILE;
        let mut acc = 0f32;
        let mut sg = [0i8; GROUP_SIZE + 8]; // +8 slack for the u64-store unpack
        for gi in 0..gpr {
            let off = (r * gpr + gi) * TILE;
            let s = f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
            q1t_unpack_group_i8(bytes.as_ptr().wrapping_add(off + 2), &mut sg);
            let mut d = 0i32;
            for k in 0..GROUP_SIZE {
                d += sg[k] as i32 * xq[gi * GROUP_SIZE + k] as i32;
            }
            acc += d as f32 * s;
        }
        acc
    }
}

/// Σ over a row's outliers of `value·x[col]` — the correction that adds the
/// overlay's exact weights on top of the base dot. INVARIANT: the encoder
/// writes ternary code 0 at every outlier position (`quantize_q1t`), so the
/// base contributes nothing there and this is a plain `value·x`, not
/// `(value − base)·x` — no scattered per-outlier scale read. Row `r`'s entries
/// are the contiguous slice `[row_ptr[r], row_ptr[r+1])`, so no binary search.
fn q1t_row_outlier_correction(
    bytes: &[u8],
    r: usize,
    rp_off: usize,
    entries_off: usize,
    has_ov: bool,
    x: &[f32],
) -> f32 {
    if !has_ov {
        return 0.0;
    }
    let (c0, c1) = (
        q1t_rowptr(bytes, rp_off, r),
        q1t_rowptr(bytes, rp_off, r + 1),
    );
    let mut corr = 0f32;
    for p in c0..c1 {
        let e = entries_off + p * 4;
        let col = u16::from_le_bytes([bytes[e], bytes[e + 1]]) as usize;
        let val = f16_to_f32(u16::from_le_bytes([bytes[e + 2], bytes[e + 3]]));
        corr += val * x[col];
    }
    corr
}

/// Dequantize one q1t row into `buf[..cols]` via the sign LUT (no division),
/// then apply the row's outliers (its `[row_ptr[r], row_ptr[r+1])` slice).
/// Used by the batched (prefill) path where the decode amortizes over the batch.
fn q1t_dequant_row(
    bytes: &[u8],
    r: usize,
    gpr: usize,
    rp_off: usize,
    entries_off: usize,
    has_ov: bool,
    buf: &mut [f32],
) {
    const TILE: usize = cortiq_core::quant::Q1T_TILE;
    for g in 0..gpr {
        let off = (r * gpr + g) * TILE;
        let s = f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
        let codes = &bytes[off + 2..off + TILE];
        let bc = g * GROUP_SIZE;
        // 6 full bytes (30 codes) + a 7th byte holding the last 2.
        for bi in 0..6 {
            let lut = &SIGN5[codes[bi] as usize];
            let d = &mut buf[bc + bi * 5..bc + bi * 5 + 5];
            for i in 0..5 {
                d[i] = lut[i] * s;
            }
        }
        let lut = &SIGN5[codes[6] as usize];
        buf[bc + 30] = lut[0] * s;
        buf[bc + 31] = lut[1] * s;
    }
    if !has_ov {
        return;
    }
    let (c0, c1) = (
        q1t_rowptr(bytes, rp_off, r),
        q1t_rowptr(bytes, rp_off, r + 1),
    );
    for p in c0..c1 {
        let e = entries_off + p * 4;
        let col = u16::from_le_bytes([bytes[e], bytes[e + 1]]) as usize;
        buf[col] = f16_to_f32(u16::from_le_bytes([bytes[e + 2], bytes[e + 3]]));
    }
}

/// Add the sparse outlier overlay onto a base dot already in `out` (the GPU
/// computes the ternary base; the overlay stays on the CPU — its entries are
/// few and its per-row gather doesn't vectorize on the GPU). Row-parallel.
fn q1t_add_overlay(
    bytes: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    const TILE: usize = cortiq_core::quant::Q1T_TILE;
    let gpr = cols / GROUP_SIZE;
    let (rp_off, ent_off, has_ov) = q1t_overlay(bytes, rows * gpr * TILE, rows);
    if !has_ov {
        return;
    }
    let out_addr = SendMut(out.as_mut_ptr());
    let run = move |start: usize, end: usize| {
        for r in start..end {
            let corr = q1t_row_outlier_correction(bytes, r, rp_off, ent_off, has_ov, x);
            // SAFETY: disjoint rows; add onto the base the GPU already wrote.
            unsafe { *out_addr.at(r) += corr };
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// Q1T row range via the A8W8 int8 path — shared activation split,
/// per-row: base SDOT dot + outlier correction + overlay.
#[allow(clippy::too_many_arguments)]
fn q1t_range_a8w8(
    bytes: &[u8],
    gpr: usize,
    rp_off: usize,
    ent_off: usize,
    has_ov: bool,
    act: &SplitAct,
    x: &[f32],
    out: SendMut,
    start: usize,
    end: usize,
) {
    for r in start..end {
        let mut acc = q1t_dot_row_i8(bytes, r, gpr, &act.xq) * act.sx;
        for &(j, xv) in &act.outliers {
            acc += q1t_base_weight(bytes, r, gpr, j) * xv;
        }
        acc += q1t_row_outlier_correction(bytes, r, rp_off, ent_off, has_ov, x);
        // SAFETY: disjoint row ranges per worker.
        unsafe { *out.at(r) = acc };
    }
}

/// Q1T row range via the f32 path (no SDOT) — for matvec_many batched
/// dispatch when a8w8 is unavailable.
#[allow(clippy::too_many_arguments)]
fn q1t_range_f32_batch(
    bytes: &[u8],
    gpr: usize,
    rp_off: usize,
    ent_off: usize,
    has_ov: bool,
    x: &[f32],
    out: SendMut,
    start: usize,
    end: usize,
) {
    const TILE: usize = cortiq_core::quant::Q1T_TILE;
    let mut sg = [0f32; GROUP_SIZE];
    for r in start..end {
        let mut acc = 0f32;
        for g in 0..gpr {
            let off = (r * gpr + g) * TILE;
            let s = f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
            let codes = &bytes[off + 2..off + TILE];
            let xg = &x[g * GROUP_SIZE..g * GROUP_SIZE + GROUP_SIZE];
            for bi in 0..6 {
                sg[bi * 5..bi * 5 + 5].copy_from_slice(&SIGN5[codes[bi] as usize]);
            }
            let lut = &SIGN5[codes[6] as usize];
            sg[30] = lut[0];
            sg[31] = lut[1];
            let mut gsum = 0f32;
            for k in 0..GROUP_SIZE {
                gsum += sg[k] * xg[k];
            }
            acc += s * gsum;
        }
        acc += q1t_row_outlier_correction(bytes, r, rp_off, ent_off, has_ov, x);
        // SAFETY: disjoint row ranges per worker.
        unsafe { *out.at(r) = acc };
    }
}

/// Ternary (q1t) matvec — decode+dot straight from mmap, one group at a time:
/// no per-ROW buffer, no division (the sign LUT), and a tiny per-group sign
/// buffer so the 32-wide dot vectorizes. This is the decode hot path.
fn q1t_matvec(
    bytes: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), rows);
    const TILE: usize = cortiq_core::quant::Q1T_TILE;
    let gpr = cols / GROUP_SIZE;
    let (rp_off, ent_off, has_ov) = q1t_overlay(bytes, rows * gpr * TILE, rows);
    let out_addr = SendMut(out.as_mut_ptr());
    // int8 SDOT base dot (ARM dotprod): ~4× the f32 arithmetic. x → i8 once
    // (`split_act`), activation outliers added back exactly in f32, weight
    // overlay on top. ARM SDOT / x86 AVX2; CMF_SDOT=0 keeps the exact f32 path.
    if a8w8_enabled() {
        let act = split_act(x);
        let act = &act;
        let run = move |start: usize, end: usize| {
            for r in start..end {
                let mut acc = q1t_dot_row_i8(bytes, r, gpr, &act.xq) * act.sx;
                for &(j, xv) in &act.outliers {
                    acc += q1t_base_weight(bytes, r, gpr, j) * xv;
                }
                acc += q1t_row_outlier_correction(bytes, r, rp_off, ent_off, has_ov, x);
                // SAFETY: disjoint row ranges per worker.
                unsafe { *out_addr.at(r) = acc };
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = move |start: usize, end: usize| {
        // Per-group signs, unpacked contiguously so the dot below is a clean
        // 32-wide reduction the autovectorizer turns into f32x4 FMAs — the
        // 5-values-per-byte base-3 layout won't SIMD in place.
        let mut sg = [0f32; GROUP_SIZE];
        for r in start..end {
            let mut acc = 0f32;
            for g in 0..gpr {
                let off = (r * gpr + g) * TILE;
                let s = f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
                let codes = &bytes[off + 2..off + TILE];
                let xg = &x[g * GROUP_SIZE..g * GROUP_SIZE + GROUP_SIZE];
                for bi in 0..6 {
                    sg[bi * 5..bi * 5 + 5].copy_from_slice(&SIGN5[codes[bi] as usize]);
                }
                let lut = &SIGN5[codes[6] as usize];
                sg[30] = lut[0];
                sg[31] = lut[1];
                let mut gsum = 0f32;
                for k in 0..GROUP_SIZE {
                    gsum += sg[k] * xg[k];
                }
                acc += s * gsum;
            }
            acc += q1t_row_outlier_correction(bytes, r, rp_off, ent_off, has_ov, x);
            unsafe { *out_addr.at(r) = acc };
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// Fused-pair twin of `q1t_dot_row_sdot`: ONE register unpack of the
/// ternary codes serves BOTH activation streams (the unpack chain is
/// the dominant per-row cost — MTP verify pairs paid it twice). Per
/// stream the group order and f32 accumulation match the single-row
/// kernel exactly, so pair == 2×matvec bit-for-bit.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q1t_dot_row_sdot2(bytes: &[u8], r: usize, gpr: usize, xa: &[i8], xb: &[i8]) -> [f32; 2] {
    use core::arch::aarch64::*;
    use core::arch::asm;
    // SAFETY: same slice-length contracts as `q1t_dot_row_sdot`, ×2.
    unsafe {
        const TILE: usize = cortiq_core::quant::Q1T_TILE;
        let bytes_ptr = bytes.as_ptr();
        let row_off = r * gpr * TILE;
        let xp = [xa.as_ptr(), xb.as_ptr()];
        let mut acc = [0f32; 2];
        macro_rules! sdot2 {
            ($w0:expr, $w1:expr, $x:expr) => {{
                let x0 = vld1q_s8($x);
                let x1 = vld1q_s8($x.add(16));
                let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
                asm!(
                    "sdot {a0:v}.4s, {w0:v}.16b, {x0:v}.16b",
                    "sdot {a1:v}.4s, {w1:v}.16b, {x1:v}.16b",
                    a0 = inout(vreg) a0, a1 = inout(vreg) a1,
                    w0 = in(vreg) $w0, x0 = in(vreg) x0, w1 = in(vreg) $w1, x1 = in(vreg) x1,
                    options(pure, nomem, nostack),
                );
                vaddvq_s32(vaddq_s32(a0, a1))
            }};
        }
        let gpr2 = gpr & !1;
        let mut gi = 0;
        while gi < gpr2 {
            let off0 = row_off + gi * TILE;
            let off1 = off0 + TILE;
            let s0 = f16_to_f32(u16::from_le_bytes([
                *bytes_ptr.add(off0),
                *bytes_ptr.add(off0 + 1),
            ]));
            let s1 = f16_to_f32(u16::from_le_bytes([
                *bytes_ptr.add(off1),
                *bytes_ptr.add(off1 + 1),
            ]));
            let (u0_0, u1_0, u2_0, u3_0) = q1t_unpack_reg_u64s(bytes_ptr.add(off0 + 2));
            let (u0_1, u1_1, u2_1, u3_1) = q1t_unpack_reg_u64s(bytes_ptr.add(off1 + 2));
            let w0_0 = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(u0_0), vcreate_u64(u1_0)));
            let w1_0 = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(u2_0), vcreate_u64(u3_0)));
            let w0_1 = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(u0_1), vcreate_u64(u1_1)));
            let w1_1 = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(u2_1), vcreate_u64(u3_1)));
            for k in 0..2 {
                let d0 = sdot2!(w0_0, w1_0, xp[k].add(gi * GROUP_SIZE));
                let d1 = sdot2!(w0_1, w1_1, xp[k].add((gi + 1) * GROUP_SIZE));
                acc[k] += d0 as f32 * s0 + d1 as f32 * s1;
            }
            gi += 2;
        }
        if gi < gpr {
            let off = row_off + gi * TILE;
            let s = f16_to_f32(u16::from_le_bytes([
                *bytes_ptr.add(off),
                *bytes_ptr.add(off + 1),
            ]));
            let (u0, u1, u2, u3) = q1t_unpack_reg_u64s(bytes_ptr.add(off + 2));
            let w0 = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(u0), vcreate_u64(u1)));
            let w1 = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(u2), vcreate_u64(u3)));
            for k in 0..2 {
                let d = sdot2!(w0, w1, xp[k].add(gi * GROUP_SIZE));
                acc[k] += d as f32 * s;
            }
        }
        acc
    }
}

/// Fused Q1T pair matvec: ONE pass over the rows serves both
/// activation streams — on ARM the ternary register unpack happens
/// once per tile pair (`q1t_dot_row_sdot2`); elsewhere the second dot
/// rides the row's L1-warm tile bytes. Per stream the math matches
/// `q1t_matvec` exactly.
fn q1t_matvec2(
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
    const TILE: usize = cortiq_core::quant::Q1T_TILE;
    let gpr = cols / GROUP_SIZE;
    let (rp_off, ent_off, has_ov) = q1t_overlay(bytes, rows * gpr * TILE, rows);
    let out1 = SendMut(o1.as_mut_ptr());
    let out2 = SendMut(o2.as_mut_ptr());
    if a8w8_enabled() {
        let a1 = split_act(x1);
        let a2 = split_act(x2);
        let (a1, a2) = (&a1, &a2);
        let run = move |start: usize, end: usize| {
            for r in start..end {
                #[cfg(target_arch = "aarch64")]
                // a8w8 on aarch64 ⇔ sdot_enabled(), so the kernel's
                // target features are present.
                let ds = unsafe { q1t_dot_row_sdot2(bytes, r, gpr, &a1.xq, &a2.xq) };
                #[cfg(not(target_arch = "aarch64"))]
                let ds = [
                    q1t_dot_row_i8(bytes, r, gpr, &a1.xq),
                    q1t_dot_row_i8(bytes, r, gpr, &a2.xq),
                ];
                let mut acc1 = ds[0] * a1.sx;
                for &(j, xv) in &a1.outliers {
                    acc1 += q1t_base_weight(bytes, r, gpr, j) * xv;
                }
                acc1 += q1t_row_outlier_correction(bytes, r, rp_off, ent_off, has_ov, x1);
                let mut acc2 = ds[1] * a2.sx;
                for &(j, xv) in &a2.outliers {
                    acc2 += q1t_base_weight(bytes, r, gpr, j) * xv;
                }
                acc2 += q1t_row_outlier_correction(bytes, r, rp_off, ent_off, has_ov, x2);
                // SAFETY: disjoint row ranges per worker.
                unsafe {
                    *out1.at(r) = acc1;
                    *out2.at(r) = acc2;
                }
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = move |start: usize, end: usize| {
        // Exact path (CMF_SDOT=0): unpack the sign LUT once per group,
        // dot both streams — same op order per stream as `q1t_matvec`.
        let mut sg = [0f32; GROUP_SIZE];
        for r in start..end {
            let mut acc1 = 0f32;
            let mut acc2 = 0f32;
            for g in 0..gpr {
                let off = (r * gpr + g) * TILE;
                let s = f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
                let codes = &bytes[off + 2..off + TILE];
                for bi in 0..6 {
                    sg[bi * 5..bi * 5 + 5].copy_from_slice(&SIGN5[codes[bi] as usize]);
                }
                let lut = &SIGN5[codes[6] as usize];
                sg[30] = lut[0];
                sg[31] = lut[1];
                let xg1 = &x1[g * GROUP_SIZE..g * GROUP_SIZE + GROUP_SIZE];
                let xg2 = &x2[g * GROUP_SIZE..g * GROUP_SIZE + GROUP_SIZE];
                let mut gsum1 = 0f32;
                for k in 0..GROUP_SIZE {
                    gsum1 += sg[k] * xg1[k];
                }
                acc1 += s * gsum1;
                let mut gsum2 = 0f32;
                for k in 0..GROUP_SIZE {
                    gsum2 += sg[k] * xg2[k];
                }
                acc2 += s * gsum2;
            }
            acc1 += q1t_row_outlier_correction(bytes, r, rp_off, ent_off, has_ov, x1);
            acc2 += q1t_row_outlier_correction(bytes, r, rp_off, ent_off, has_ov, x2);
            // SAFETY: disjoint row ranges per worker.
            unsafe {
                *out1.at(r) = acc1;
                *out2.at(r) = acc2;
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// Ternary (q1t) matmat (prefill) — dequant each row once, dot the whole
/// batch against it (amortizes the per-row decode).
fn q1t_matmat(
    bytes: &[u8],
    xs: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), b * rows);
    const TILE: usize = cortiq_core::quant::Q1T_TILE;
    let gpr = cols / GROUP_SIZE;
    let (rp_off, ent_off, has_ov) = q1t_overlay(bytes, rows * gpr * TILE, rows);
    let out_addr = SendMut(out.as_mut_ptr());
    // int8 prefill (ARM SDOT / x86 AVX2): quantize the B inputs once, unpack
    // each weight row's signs to i8 ONCE, then int8-dot against every input —
    // the row sign-decode amortizes over the whole batch. CMF_SDOT=0 → f32.
    if a8w8_enabled() {
        let acts: Vec<SplitAct> = (0..b)
            .map(|bi| split_act(&xs[bi * cols..(bi + 1) * cols]))
            .collect();
        let acts = &acts;
        let run = move |start: usize, end: usize| {
            let mut sg = vec![0i8; cols + 8]; // row signs, i8 (+8 unpack slack)
            let mut sc = vec![0f32; gpr]; // per-group scales
            let mut accs = vec![0f32; b]; // per-batch accumulators, reused per row
            for r in start..end {
                for g in 0..gpr {
                    let off = (r * gpr + g) * TILE;
                    sc[g] = f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
                    q1t_unpack_group_i8(
                        bytes.as_ptr().wrapping_add(off + 2),
                        &mut sg[g * GROUP_SIZE..],
                    );
                }
                for bi in 0..b {
                    let act = &acts[bi];
                    let mut isum = 0f32;
                    for g in 0..gpr {
                        let d = q1t_i8dot32(
                            sg.as_ptr().wrapping_add(g * GROUP_SIZE),
                            act.xq.as_ptr().wrapping_add(g * GROUP_SIZE),
                        );
                        isum += d as f32 * sc[g];
                    }
                    let mut acc = isum * act.sx;
                    for &(j, xv) in &act.outliers {
                        acc += q1t_base_weight(bytes, r, gpr, j) * xv;
                    }
                    accs[bi] = acc;
                }
                // Overlay ONCE per row for the whole batch: read each (col, val)
                // from mmap a single time (was b× — the re-read dominated prefill)
                // and fan it out over the batch via the cached inputs.
                if has_ov {
                    let (c0, c1) = (
                        q1t_rowptr(bytes, rp_off, r),
                        q1t_rowptr(bytes, rp_off, r + 1),
                    );
                    for p in c0..c1 {
                        let e = ent_off + p * 4;
                        let col = u16::from_le_bytes([bytes[e], bytes[e + 1]]) as usize;
                        let val = f16_to_f32(u16::from_le_bytes([bytes[e + 2], bytes[e + 3]]));
                        for bi in 0..b {
                            accs[bi] += val * xs[bi * cols + col];
                        }
                    }
                }
                for bi in 0..b {
                    unsafe { *out_addr.at(bi * rows + r) = accs[bi] };
                }
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = move |start: usize, end: usize| {
        let mut buf = vec![0f32; cols];
        for r in start..end {
            q1t_dequant_row(bytes, r, gpr, rp_off, ent_off, has_ov, &mut buf);
            for bi in 0..b {
                let xr = &xs[bi * cols..(bi + 1) * cols];
                let mut acc = 0f32;
                for j in 0..cols {
                    acc += buf[j] * xr[j];
                }
                unsafe { *out_addr.at(bi * rows + r) = acc };
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

fn q1_matvec(
    bytes: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), rows);
    let gpr = cols / GROUP_SIZE;
    let out_addr = SendMut(out.as_mut_ptr());
    if a8w8_enabled() {
        let act = split_act(x);
        let gsum = q1_group_sums(&act.xq, gpr);
        let (act, gsum) = (&act, &gsum);
        let run = move |start: usize, end: usize| {
            q1_range_a8w8(bytes, gpr, act, gsum, out_addr, start, end)
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = move |start: usize, end: usize| q1_range_f32(bytes, gpr, x, out_addr, start, end);
    dispatch_rows(pool, rows, &run);
}

/// Fused two-input q1 matvec (weights read once per pair).
#[allow(clippy::too_many_arguments)]
fn q1_matvec2(
    bytes: &[u8],
    x1: &[f32],
    x2: &[f32],
    rows: usize,
    cols: usize,
    o1: &mut [f32],
    o2: &mut [f32],
    pool: Option<&Pool>,
) {
    let gpr = cols / GROUP_SIZE;
    let p1 = SendMut(o1.as_mut_ptr());
    let p2 = SendMut(o2.as_mut_ptr());
    if a8w8_enabled() {
        let a1 = split_act(x1);
        let a2 = split_act(x2);
        let g1 = q1_group_sums(&a1.xq, gpr);
        let g2 = q1_group_sums(&a2.xq, gpr);
        let (a1, a2, g1, g2) = (&a1, &a2, &g1, &g2);
        let run = move |start: usize, end: usize| {
            for r in start..end {
                let mut v1 = dot_q1_row_i8(bytes, r, gpr, &a1.xq, g1) * a1.sx;
                let mut v2 = dot_q1_row_i8(bytes, r, gpr, &a2.xq, g2) * a2.sx;
                for &(j, xv) in &a1.outliers {
                    let (w, s) = q1_outlier(bytes, r, gpr, j);
                    v1 += w * s * xv;
                }
                for &(j, xv) in &a2.outliers {
                    let (w, s) = q1_outlier(bytes, r, gpr, j);
                    v2 += w * s * xv;
                }
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
    let run = move |start: usize, end: usize| {
        for r in start..end {
            // SAFETY: disjoint row ranges per worker.
            unsafe {
                *p1.at(r) = q1_row_exact(bytes, r, gpr, x1);
                *p2.at(r) = q1_row_exact(bytes, r, gpr, x2);
            }
        }
    };
    dispatch_rows(pool, rows, &run);
}

/// Batched q1 matmat: each row's tiles stream once per microbatch.
#[allow(clippy::too_many_arguments)]
fn q1_matmat(
    bytes: &[u8],
    xs_all: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), b * rows);
    let gpr = cols / GROUP_SIZE;
    let out_addr = SendMut(out.as_mut_ptr());
    if a8w8_enabled() {
        let acts: Vec<(SplitAct, Vec<i32>)> = (0..b)
            .map(|bi| {
                let act = split_act(&xs_all[bi * cols..(bi + 1) * cols]);
                let gsum = q1_group_sums(&act.xq, gpr);
                (act, gsum)
            })
            .collect();
        let acts = &acts;
        #[cfg(target_arch = "x86_64")]
        let blocked_ok = avx2_enabled() && blocked_enabled();
        #[cfg(target_arch = "aarch64")]
        let blocked_ok = sdot_enabled() && blocked_enabled();
        let run = move |start: usize, end: usize| {
            for r in start..end {
                let mut bi = 0usize;
                // Blocked 1×4: the unpacked bit mask serves four
                // activation streams per group.
                #[cfg(target_arch = "aarch64")]
                if blocked_ok {
                    while bi + 4 <= acts.len() {
                        let xs = [
                            acts[bi].0.xq.as_slice(),
                            acts[bi + 1].0.xq.as_slice(),
                            acts[bi + 2].0.xq.as_slice(),
                            acts[bi + 3].0.xq.as_slice(),
                        ];
                        let gs = [
                            acts[bi].1.as_slice(),
                            acts[bi + 1].1.as_slice(),
                            acts[bi + 2].1.as_slice(),
                            acts[bi + 3].1.as_slice(),
                        ];
                        let d = unsafe { dot_q1_row_1x4_sdot(bytes, r, gpr, xs, gs) };
                        for k in 0..4 {
                            let (act, _) = &acts[bi + k];
                            let mut acc = d[k] * act.sx;
                            for &(j, xv) in &act.outliers {
                                let (w, sc) = q1_outlier(bytes, r, gpr, j);
                                acc += w * sc * xv;
                            }
                            // SAFETY: disjoint (bi, r) cells per worker.
                            unsafe { *out_addr.at((bi + k) * rows + r) = acc };
                        }
                        bi += 4;
                    }
                }
                #[cfg(target_arch = "x86_64")]
                if blocked_ok {
                    while bi + 4 <= acts.len() {
                        let xs = [
                            acts[bi].0.xq.as_slice(),
                            acts[bi + 1].0.xq.as_slice(),
                            acts[bi + 2].0.xq.as_slice(),
                            acts[bi + 3].0.xq.as_slice(),
                        ];
                        let gs = [
                            acts[bi].1.as_slice(),
                            acts[bi + 1].1.as_slice(),
                            acts[bi + 2].1.as_slice(),
                            acts[bi + 3].1.as_slice(),
                        ];
                        let d = unsafe {
                            if vnni_tiles_enabled() {
                                dot_q1_row_1x4_vnni(bytes, r, gpr, xs, gs)
                            } else {
                                dot_q1_row_1x4_avx2(bytes, r, gpr, xs, gs)
                            }
                        };
                        for k in 0..4 {
                            let (act, _) = &acts[bi + k];
                            let mut acc = d[k] * act.sx;
                            for &(j, xv) in &act.outliers {
                                let (w, sc) = q1_outlier(bytes, r, gpr, j);
                                acc += w * sc * xv;
                            }
                            // SAFETY: disjoint (bi, r) cells per worker.
                            unsafe { *out_addr.at((bi + k) * rows + r) = acc };
                        }
                        bi += 4;
                    }
                }
                while bi < acts.len() {
                    let (act, gsum) = &acts[bi];
                    let mut acc = dot_q1_row_i8(bytes, r, gpr, &act.xq, gsum) * act.sx;
                    for &(j, xv) in &act.outliers {
                        let (w, s) = q1_outlier(bytes, r, gpr, j);
                        acc += w * s * xv;
                    }
                    // SAFETY: disjoint (bi, r) cells per worker range.
                    unsafe { *out_addr.at(bi * rows + r) = acc };
                    bi += 1;
                }
            }
        };
        dispatch_rows(pool, rows, &run);
        return;
    }
    let run = move |start: usize, end: usize| {
        for r in start..end {
            for bi in 0..b {
                let x = &xs_all[bi * cols..(bi + 1) * cols];
                // SAFETY: disjoint (bi, r) cells per worker range.
                unsafe { *out_addr.at(bi * rows + r) = q1_row_exact(bytes, r, gpr, x) };
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
fn q4matvec(
    bytes: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), rows);
    let (packed, scales) = q4_split(bytes, rows, cols);
    let gpr = cols / GROUP_SIZE;
    let out_addr = SendMut(out.as_mut_ptr());

    if a8w8_enabled() {
        let act = split_act(x);
        let run = move |start: usize, end: usize| {
            q4_range_a8w8(packed, scales, gpr, cols, &act, out_addr, start, end)
        };
        dispatch_rows(pool, rows, &run);
        return;
    }

    let run =
        move |start: usize, end: usize| q4_range_f32(packed, scales, gpr, x, out_addr, start, end);
    dispatch_rows(pool, rows, &run);
}

/// One q4 row via the A8W8 int8 path — SDOT on ARM, AVX2 maddubs on
/// x86 (scalar fallback is unreachable: callers gate on a8w8_enabled).
#[inline]
#[allow(unreachable_code)]
/// One UNPACKED q4 row (centered i8 in `buf`) against four activation
/// streams: the 32-byte weight chunk and its abs() load once per group,
/// the per-group f16 scale decodes once — four maddubs+reduce chains
/// instead of four full (load, abs, dot) rounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4b_row_1x4_avx2(
    buf: &[u8],
    scales: &[u8],
    g0: usize,
    gpr: usize,
    xs: [&[i8]; 4],
) -> [f32; 4] {
    // SAFETY: callers uphold buffer contracts (buf.len() == gpr·32).
    unsafe {
        use core::arch::x86_64::*;
        let ones = _mm256_set1_epi16(1);
        let mut acc = [0f32; 4];
        for gi in 0..gpr {
            let s = f16_to_f32(u16::from_le_bytes([
                scales[(g0 + gi) * 2],
                scales[(g0 + gi) * 2 + 1],
            ]));
            let w = _mm256_loadu_si256(buf.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
            let aw = _mm256_abs_epi8(w);
            for (k, xq) in xs.iter().enumerate() {
                let x = _mm256_loadu_si256(xq.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
                let p16 = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w));
                let d = _mm256_madd_epi16(p16, ones);
                let hi128 = _mm256_extracti128_si256::<1>(d);
                let s128 = _mm_add_epi32(_mm256_castsi256_si128(d), hi128);
                let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
                let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
                acc[k] += _mm_cvtsi128_si32(s32) as f32 * s;
            }
        }
        acc
    }
}

/// VNNI twin of `dot_q4b_row_1x4_avx2` (see `dpbusd_hsum`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn dot_q4b_row_1x4_vnni(
    buf: &[u8],
    scales: &[u8],
    g0: usize,
    gpr: usize,
    xs: [&[i8]; 4],
) -> [f32; 4] {
    // SAFETY: callers uphold buffer contracts (buf.len() == gpr·32).
    unsafe {
        use core::arch::x86_64::*;
        let mut acc = [0f32; 4];
        for gi in 0..gpr {
            let s = f16_to_f32(u16::from_le_bytes([
                scales[(g0 + gi) * 2],
                scales[(g0 + gi) * 2 + 1],
            ]));
            let w = _mm256_loadu_si256(buf.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
            let aw = _mm256_abs_epi8(w);
            for (k, xq) in xs.iter().enumerate() {
                let x = _mm256_loadu_si256(xq.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
                let d = dpbusd_hsum(aw, _mm256_sign_epi8(x, w));
                acc[k] += d as f32 * s;
            }
        }
        acc
    }
}

/// The vbit flavor of the blocked 1×4: the per-activation A8W8 scale
/// folds in PER GROUP as `(d·sx)·s` — bit-matching the single-matvec
/// accumulation order (the q4_block flavor applies sx once at the end,
/// matching ITS single path; the two conventions are historical and
/// each blocked leg must mirror its own).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4b_row_1x4_sx_avx2(
    buf: &[u8],
    scales: &[u8],
    g0: usize,
    gpr: usize,
    xs: [&[i8]; 4],
    sxs: [f32; 4],
) -> [f32; 4] {
    // SAFETY: callers uphold buffer contracts (buf.len() == gpr·32).
    unsafe {
        use core::arch::x86_64::*;
        let ones = _mm256_set1_epi16(1);
        let mut acc = [0f32; 4];
        for gi in 0..gpr {
            let s = f16_to_f32(u16::from_le_bytes([
                scales[(g0 + gi) * 2],
                scales[(g0 + gi) * 2 + 1],
            ]));
            let w = _mm256_loadu_si256(buf.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
            let aw = _mm256_abs_epi8(w);
            for (k, xq) in xs.iter().enumerate() {
                let x = _mm256_loadu_si256(xq.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
                let p16 = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w));
                let d = _mm256_madd_epi16(p16, ones);
                let hi128 = _mm256_extracti128_si256::<1>(d);
                let s128 = _mm_add_epi32(_mm256_castsi256_si128(d), hi128);
                let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
                let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
                acc[k] += (_mm_cvtsi128_si32(s32) as f32 * sxs[k]) * s;
            }
        }
        acc
    }
}

/// VNNI twin of `dot_q4b_row_1x4_sx_avx2` (see `dpbusd_hsum`; the
/// per-group `(d·sx)·s` fold mirrors the vbit single path).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn dot_q4b_row_1x4_sx_vnni(
    buf: &[u8],
    scales: &[u8],
    g0: usize,
    gpr: usize,
    xs: [&[i8]; 4],
    sxs: [f32; 4],
) -> [f32; 4] {
    // SAFETY: callers uphold buffer contracts (buf.len() == gpr·32).
    unsafe {
        use core::arch::x86_64::*;
        let mut acc = [0f32; 4];
        for gi in 0..gpr {
            let s = f16_to_f32(u16::from_le_bytes([
                scales[(g0 + gi) * 2],
                scales[(g0 + gi) * 2 + 1],
            ]));
            let w = _mm256_loadu_si256(buf.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
            let aw = _mm256_abs_epi8(w);
            for (k, xq) in xs.iter().enumerate() {
                let x = _mm256_loadu_si256(xq.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
                let d = dpbusd_hsum(aw, _mm256_sign_epi8(x, w));
                acc[k] += (d as f32 * sxs[k]) * s;
            }
        }
        acc
    }
}

#[allow(unreachable_code)]
fn dot_q4_row_i8(packed: &[u8], scales: &[u8], g0: usize, gpr: usize, xq: &[i8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_q4_row_sdot(packed, scales, g0, gpr, xq);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return dot_q4_row_avx2(packed, scales, g0, gpr, xq);
    }
    let mut acc = 0f32;
    for gi in 0..gpr {
        let g = g0 + gi;
        let s = f16_to_f32(u16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]));
        let mut d = 0i32;
        for (k, &b) in packed[g * 16..(g + 1) * 16].iter().enumerate() {
            d += ((b & 0x0F) as i32 - 8) * xq[gi * GROUP_SIZE + k * 2] as i32
                + (((b >> 4) & 0x0F) as i32 - 8) * xq[gi * GROUP_SIZE + k * 2 + 1] as i32;
        }
        acc += d as f32 * s;
    }
    acc
}

/// Two-activation q4 row via the A8W8 int8 path (see `dot_q4_row_i8`).
#[inline]
#[allow(unreachable_code)]
fn dot_q4_row_i8_2(
    packed: &[u8],
    scales: &[u8],
    g0: usize,
    gpr: usize,
    xq1: &[i8],
    xq2: &[i8],
) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_q4_row_sdot2(packed, scales, g0, gpr, xq1, xq2);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return dot_q4_row_avx2_2(packed, scales, g0, gpr, xq1, xq2);
    }
    (
        dot_q4_row_i8(packed, scales, g0, gpr, xq1),
        dot_q4_row_i8(packed, scales, g0, gpr, xq2),
    )
}

/// One q4 row range via SDOT (kernel body of `q4matvec`, extracted so
/// multi-matrix jobs can drive it for several tensors in one dispatch).
#[allow(clippy::too_many_arguments)]
fn q4_range_a8w8(
    packed: &[u8],
    scales: &[u8],
    gpr: usize,
    cols: usize,
    act: &SplitAct,
    out: SendMut,
    start: usize,
    end: usize,
) {
    for r in start..end {
        let mut acc = dot_q4_row_i8(packed, scales, r * gpr, gpr, &act.xq) * act.sx;
        // xq is zeroed at outlier slots — add the exact terms.
        for &(j, xv) in &act.outliers {
            let flat = r * cols + j;
            let byte = packed[flat / 2];
            let nib = if flat & 1 == 0 {
                byte & 0x0F
            } else {
                byte >> 4
            };
            let s = f16_to_f32(u16::from_le_bytes([
                scales[(flat / GROUP_SIZE) * 2],
                scales[(flat / GROUP_SIZE) * 2 + 1],
            ]));
            acc += ((nib as i32 - 8) as f32) * s * xv;
        }
        // SAFETY: disjoint row ranges per worker.
        unsafe { *out.at(r) = acc };
    }
}

/// Two-input q4 row range via the A8W8 int8 path — kernel body of
/// `q4matvec2`, extracted for pair multi-matrix jobs.
#[allow(clippy::too_many_arguments)]
fn q4_range2_a8w8(
    packed: &[u8],
    scales: &[u8],
    gpr: usize,
    cols: usize,
    a1: &SplitAct,
    a2: &SplitAct,
    p1: SendMut,
    p2: SendMut,
    start: usize,
    end: usize,
) {
    for r in start..end {
        let (s1, s2) = dot_q4_row_i8_2(packed, scales, r * gpr, gpr, &a1.xq, &a2.xq);
        let mut acc1 = s1 * a1.sx;
        let mut acc2 = s2 * a2.sx;
        // xq is zeroed at outlier slots — add the exact terms.
        let fix = |outliers: &[(usize, f32)], acc: &mut f32| {
            for &(j, xv) in outliers {
                let flat = r * cols + j;
                let byte = packed[flat / 2];
                let nib = if flat & 1 == 0 {
                    byte & 0x0F
                } else {
                    byte >> 4
                };
                let s = f16_to_f32(u16::from_le_bytes([
                    scales[(flat / GROUP_SIZE) * 2],
                    scales[(flat / GROUP_SIZE) * 2 + 1],
                ]));
                *acc += ((nib as i32 - 8) as f32) * s * xv;
            }
        };
        fix(&a1.outliers, &mut acc1);
        fix(&a2.outliers, &mut acc2);
        // SAFETY: disjoint row ranges per worker.
        unsafe {
            *p1.at(r) = acc1;
            *p2.at(r) = acc2;
        }
    }
}

/// Exact scalar q4 row range (same extraction, non-SDOT path).
fn q4_range_f32(
    packed: &[u8],
    scales: &[u8],
    gpr: usize,
    x: &[f32],
    out: SendMut,
    start: usize,
    end: usize,
) {
    for r in start..end {
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
        // SAFETY: disjoint row ranges per worker.
        unsafe { *out.at(r) = acc };
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

    if a8w8_enabled() {
        let a1 = split_act(x1);
        let a2 = split_act(x2);
        let p1 = SendMut(o1.as_mut_ptr());
        let p2 = SendMut(o2.as_mut_ptr());
        let run = move |start: usize, end: usize| {
            q4_range2_a8w8(packed, scales, gpr, cols, &a1, &a2, p1, p2, start, end)
        };
        dispatch_rows(pool, rows, &run);
        return;
    }

    let p1 = SendMut(o1.as_mut_ptr());
    let p2 = SendMut(o2.as_mut_ptr());
    let run = move |start: usize, end: usize| {
        q4_range2_f32(packed, scales, gpr, x1, x2, p1, p2, start, end)
    };
    dispatch_rows(pool, rows, &run);
}

/// Two-input exact scalar q4 row range (same extraction).
#[allow(clippy::too_many_arguments)]
fn q4_range2_f32(
    packed: &[u8],
    scales: &[u8],
    gpr: usize,
    x1: &[f32],
    x2: &[f32],
    p1: SendMut,
    p2: SendMut,
    start: usize,
    end: usize,
) {
    for r in start..end {
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
        // SAFETY: disjoint row ranges per worker.
        unsafe {
            *p1.at(r) = acc1;
            *p2.at(r) = acc2;
        }
    }
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

    if a8w8_enabled() {
        let acts: Vec<SplitAct> = (0..b)
            .map(|bi| split_act(&xs_all[bi * cols..(bi + 1) * cols]))
            .collect();
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
                    let mut bi = 0usize;
                    #[cfg(target_arch = "x86_64")]
                    if avx2_enabled() && blocked_enabled() {
                        while bi + 4 <= acts.len() {
                            let xs = [
                                acts[bi].xq.as_slice(),
                                acts[bi + 1].xq.as_slice(),
                                acts[bi + 2].xq.as_slice(),
                                acts[bi + 3].xq.as_slice(),
                            ];
                            let d = unsafe {
                                if vnni_tiles_enabled() {
                                    dot_q4b_row_1x4_vnni(&buf, scales, r * gpr, gpr, xs)
                                } else {
                                    dot_q4b_row_1x4_avx2(&buf, scales, r * gpr, gpr, xs)
                                }
                            };
                            for k in 0..4 {
                                let act = &acts[bi + k];
                                let mut acc = d[k] * act.sx;
                                for &(j, xv) in &act.outliers {
                                    acc += (buf[j] as i8) as f32
                                        * gscale((r * cols + j) / GROUP_SIZE)
                                        * xv;
                                }
                                // SAFETY: disjoint (bi, r) cells per worker.
                                unsafe { *out_addr.at((bi + k) * rows + r) = acc };
                            }
                            bi += 4;
                        }
                    }
                    while bi < acts.len() {
                        let act = &acts[bi];
                        let mut acc = 0f32;
                        for gi in 0..gpr {
                            let d = dot_i8_i8(
                                &buf[gi * GROUP_SIZE..(gi + 1) * GROUP_SIZE],
                                &act.xq[gi * GROUP_SIZE..(gi + 1) * GROUP_SIZE],
                            );
                            acc += d as f32 * gscale(r * gpr + gi);
                        }
                        acc *= act.sx;
                        // xq is zeroed at outlier slots — exact terms.
                        for &(j, xv) in &act.outliers {
                            acc += (buf[j] as i8) as f32 * gscale((r * cols + j) / GROUP_SIZE) * xv;
                        }
                        // SAFETY: disjoint (bi, r) cells per worker row range.
                        unsafe { *out_addr.at(bi * rows + r) = acc };
                        bi += 1;
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
                        // Pairwise (lo + hi) addition, matching
                        // q4matvec's `ga += lo·x + hi·x` shape exactly —
                        // a flat one-per-element loop rounds differently
                        // and broke bit-parity on the scalar (x86) path.
                        for k in 0..GROUP_SIZE / 2 {
                            let e = gi * GROUP_SIZE + k * 2;
                            ga += buf[e] * x[e] + buf[e + 1] * x[e + 1];
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
        f16_to_f32(u16::from_le_bytes([
            bytes[sc_off + so],
            bytes[sc_off + so + 1],
        ]))
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

    if a8w8_enabled() {
        let acts: Vec<SplitAct> = (0..b)
            .map(|bi| split_act(&xs_all[bi * cols..(bi + 1) * cols]))
            .collect();
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
                        4 => vbit_fill4(data, &mut buf),
                        5 => fill::<5>(data, l, &mut buf),
                        6 => fill::<6>(data, l, &mut buf),
                        _ => unreachable!("vbit bit-width {bw} (validated at load)"),
                    }
                    let mut bi = 0usize;
                    // The vbit scale table shares q4_block's layout
                    // (contiguous f16 per (row·ng + g)), so the same
                    // blocked 1×4 kernel serves the decoded row.
                    #[cfg(target_arch = "x86_64")]
                    if avx2_enabled() && blocked_enabled() {
                        while bi + 4 <= acts.len() {
                            let xs = [
                                acts[bi].xq.as_slice(),
                                acts[bi + 1].xq.as_slice(),
                                acts[bi + 2].xq.as_slice(),
                                acts[bi + 3].xq.as_slice(),
                            ];
                            let sxs = [
                                acts[bi].sx,
                                acts[bi + 1].sx,
                                acts[bi + 2].sx,
                                acts[bi + 3].sx,
                            ];
                            let d = unsafe {
                                if vnni_tiles_enabled() {
                                    dot_q4b_row_1x4_sx_vnni(
                                        &buf,
                                        &bytes[sc_off..],
                                        r * ng,
                                        ng,
                                        xs,
                                        sxs,
                                    )
                                } else {
                                    dot_q4b_row_1x4_sx_avx2(
                                        &buf,
                                        &bytes[sc_off..],
                                        r * ng,
                                        ng,
                                        xs,
                                        sxs,
                                    )
                                }
                            };
                            for k in 0..4 {
                                let act = &acts[bi + k];
                                let mut dot = d[k];
                                for &(j, xv) in &act.outliers {
                                    dot += (buf[j] as i8) as f32 * gscale(r, j / GROUP_SIZE) * xv;
                                }
                                // SAFETY: disjoint (bi, r) cells per worker.
                                unsafe { *out_addr.at((bi + k) * rows + r) = dot };
                            }
                            bi += 4;
                        }
                    }
                    while bi < acts.len() {
                        let act = &acts[bi];
                        let mut dot = 0f32;
                        for g in 0..ng {
                            let d = dot_i8_i8(
                                &buf[g * GROUP_SIZE..(g + 1) * GROUP_SIZE],
                                &act.xq[g * GROUP_SIZE..(g + 1) * GROUP_SIZE],
                            ) as f32
                                * act.sx;
                            dot += d * gscale(r, g);
                        }
                        for &(j, xv) in &act.outliers {
                            dot += (buf[j] as i8) as f32 * gscale(r, j / GROUP_SIZE) * xv;
                        }
                        // SAFETY: disjoint (bi, r) cells per worker range.
                        unsafe { *out_addr.at(bi * rows + r) = dot };
                        bi += 1;
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

/// Build a GPU batch job for a q8-family mapped tensor (primary
/// shard): prescaled input + directory coordinates. None → not
/// GPU-eligible, caller stays on the CPU.
pub(crate) fn gpu_batch_job<'a>(
    t: &'a QTensor,
    x: &[f32],
) -> Option<(std::sync::Arc<CmfModel>, crate::gpu::BatchJob<'a>)> {
    match t {
        QTensor::Mapped {
            model,
            idx,
            dtype: dt @ (TensorDtype::Q8Row | TensorDtype::Q8_2f),
            rows,
            cols,
            row_scale,
            col_field,
            ..
        } => Some((
            model.clone(),
            crate::gpu::BatchJob {
                idx: *idx,
                rows: *rows,
                cols: *cols,
                row_scale,
                xs: prescale(x, col_field, *dt).into_owned(),
                layout: crate::gpu::BatchLayout::Q8,
            },
        )),
        // q1: raw f32 activations, tile-embedded scales.
        QTensor::Mapped {
            model,
            idx,
            dtype: TensorDtype::Q1,
            rows,
            cols,
            ..
        } => Some((
            model.clone(),
            crate::gpu::BatchJob {
                idx: *idx,
                rows: *rows,
                cols: *cols,
                row_scale: &[],
                xs: x.to_vec(),
                layout: crate::gpu::BatchLayout::Q1,
            },
        )),
        // q4_tiled / q4tp: raw f32 activations; the scales live in the
        // payload (inline tiles / row ladder), so row_scale stays empty.
        // The GDN projection batch already runs these layouts on Metal —
        // this arm lets the attention QKV batch reach the same kernels.
        QTensor::Mapped {
            model,
            idx,
            dtype: dt @ (TensorDtype::Q4Tiled | TensorDtype::Q4TiledP),
            rows,
            cols,
            ..
        } => Some((
            model.clone(),
            crate::gpu::BatchJob {
                idx: *idx,
                rows: *rows,
                cols: *cols,
                row_scale: &[],
                xs: x.to_vec(),
                layout: if *dt == TensorDtype::Q4Tiled {
                    crate::gpu::BatchLayout::Q4t
                } else {
                    crate::gpu::BatchLayout::Q4tp
                },
            },
        )),
        _ => None,
    }
}

thread_local! {
    static PRESCALE_BUF1: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    static PRESCALE_BUF2: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

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

/// θ col-field fold for q8_2f activations. Borrowed pass-through for
/// every other dtype, using thread-local buffers to eliminate per-matvec allocations.
pub(crate) fn prescale_with<R, F: FnOnce(&[f32]) -> R>(
    x: &[f32],
    col_field: &[f32],
    dtype: TensorDtype,
    buf_id: u8,
    f: F,
) -> R {
    if dtype == TensorDtype::Q8_2f {
        if buf_id == 1 {
            PRESCALE_BUF1.with(|b| {
                let mut buf = b.borrow_mut();
                buf.clear();
                buf.extend(x.iter().zip(col_field).map(|(a, c)| a * c));
                f(&buf)
            })
        } else {
            PRESCALE_BUF2.with(|b| {
                let mut buf = b.borrow_mut();
                buf.clear();
                buf.extend(x.iter().zip(col_field).map(|(a, c)| a * c));
                f(&buf)
            })
        }
    } else {
        f(x)
    }
}

// ───────────────────── x86-64 AVX2 kernels (roadmap этап 2) ─────────────────────

/// AVX2+FMA available? Default ON when the CPU supports both;
/// `CMF_AVX2=0` disables (falls back to the autovectorized loops).
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CMF_AVX2").map(|v| v != "0").unwrap_or(true)
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    })
}

/// AVX2 A8W8 allowed? The quantized-activation contract is switched by
/// the SAME env as the ARM SDOT path: `CMF_SDOT=0` keeps exact kernels
/// (the golden-parity exact gate relies on it) — AVX2 f32 kernels stay
/// active either way, they are exact (regrouped sums only).
#[cfg(target_arch = "x86_64")]
fn avx2_a8w8_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        avx2_enabled() && std::env::var("CMF_SDOT").map(|v| v != "0").unwrap_or(true)
    })
}

/// A8W8 quantized-activation path available on THIS machine? One
/// switch across architectures: ARM dotprod (CMF_SDOT) or x86 AVX2
/// (CMF_AVX2 + the same CMF_SDOT exact-contract override).
#[inline]
pub(crate) fn a8w8_enabled() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        sdot_enabled()
    }
    #[cfg(target_arch = "x86_64")]
    {
        avx2_a8w8_enabled()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

/// int8·int8 dot dispatch: SDOT on ARM; AVX-512 VNNI (vpdpbusd) or AVX2
/// maddubs on x86. Callers are gated by `a8w8_enabled()`.
#[inline]
#[allow(unreachable_code)]
fn dot_i8_i8(w: &[u8], xq: &[i8]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_i8_sdot(w, xq);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if avx512vnni_enabled() {
            return dot_i8_i8_vnni(w, xq);
        }
        return dot_i8_i8_avx2(w, xq);
    }
    w.iter()
        .zip(xq)
        .map(|(&a, &b)| (a as i8) as i32 * b as i32)
        .sum()
}

/// AVX-512 VNNI available? (F+BW+VL+VNNI; `CMF_AVX512=0` falls back to
/// AVX2.) VL matters: short 32-byte groups (q4/vbit) ride the 256-bit
/// `vpdpbusd` encoding.
#[cfg(target_arch = "x86_64")]
fn avx512vnni_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CMF_AVX512")
            .map(|v| v != "0")
            .unwrap_or(true)
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512vnni")
    })
}

/// Grouped-codec VNNI arms (the q4t/q4b/q1/q1t tile kernels): default
/// ON where AVX-512 VNNI exists (`CMF_VNNI_TILES=0` opt-out). Measured
/// on Ryzen 7950X (Zen4, 3 alternating process pairs, blocked GEMM
/// 4864×896 b=256): q4t 63→68 GF/s (+8%), q1 53→56 (+6%), q4b 72→75
/// (+4%) — consistent, no leg regressed. The tile kernels keep a
/// horizontal reduce per 32-weight group, so the `vpdpbusd` saving is
/// smaller than the long-dot q8 win (+13%), but it is real and free.
#[cfg(target_arch = "x86_64")]
fn vnni_tiles_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CMF_VNNI_TILES")
            .map(|v| v != "0")
            .unwrap_or(true)
            && avx512vnni_enabled()
    })
}

/// One 256-bit u8×i8 dot → i32 via `vpdpbusd` into a fresh accumulator
/// plus the same horizontal reduce the AVX2 kernels use. Products are
/// bounded (|w| ≤ 8 or ≤ 1), so maddubs never saturated — the i32 sum
/// is bit-identical to the maddubs+madd pair it replaces.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn dpbusd_hsum(aw: core::arch::x86_64::__m256i, xs: core::arch::x86_64::__m256i) -> i32 {
    // SAFETY: pure register math.
    unsafe {
        use core::arch::x86_64::*;
        let d = _mm256_dpbusd_epi32(_mm256_setzero_si256(), aw, xs);
        let hi128 = _mm256_extracti128_si256::<1>(d);
        let s128 = _mm_add_epi32(_mm256_castsi256_si128(d), hi128);
        let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
        let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
        _mm_cvtsi128_si32(s32)
    }
}

/// int8·int8 via AVX-512 VNNI: `vpdpbusd` fuses the maddubs+madd+add
/// triple into one u8×i8 dot-accumulate. AVX-512 has no vpsignb, so the
/// |w|·sign(x,w) trick becomes |w| × (x negated where w<0) via a mask
/// subtract — w==0 lanes contribute 0 through |w|=0 either way.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn dot_i8_i8_vnni(w: &[u8], xq: &[i8]) -> i32 {
    // SAFETY: callers uphold slice-length contracts (see call sites).
    unsafe {
        use core::arch::x86_64::*;
        let n = w.len();
        let mut j = 0usize;
        let mut total: i32;
        // 4 independent accumulators: vpdpbusd is its own loop-carried
        // dependency (~5-cycle latency) — a single-acc loop runs
        // latency-bound and LOSES to the AVX2 maddubs kernel, measured
        // on Granite Rapids.
        {
            #[inline(always)]
            unsafe fn step(
                w: *const u8,
                x: *const i8,
                acc: core::arch::x86_64::__m512i,
            ) -> core::arch::x86_64::__m512i {
                unsafe {
                    use core::arch::x86_64::*;
                    let wv = _mm512_loadu_si512(w as *const _);
                    let xv = _mm512_loadu_si512(x as *const _);
                    let aw = _mm512_abs_epi8(wv);
                    let neg = _mm512_movepi8_mask(wv);
                    let sx = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
                    _mm512_dpbusd_epi32(acc, aw, sx)
                }
            }
            let (mut a0, mut a1, mut a2, mut a3) = (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            );
            while j + 256 <= n {
                a0 = step(w.as_ptr().add(j), xq.as_ptr().add(j), a0);
                a1 = step(w.as_ptr().add(j + 64), xq.as_ptr().add(j + 64), a1);
                a2 = step(w.as_ptr().add(j + 128), xq.as_ptr().add(j + 128), a2);
                a3 = step(w.as_ptr().add(j + 192), xq.as_ptr().add(j + 192), a3);
                j += 256;
            }
            while j + 64 <= n {
                a0 = step(w.as_ptr().add(j), xq.as_ptr().add(j), a0);
                j += 64;
            }
            let s01 = _mm512_add_epi32(a0, a1);
            let s23 = _mm512_add_epi32(a2, a3);
            total = _mm512_reduce_add_epi32(_mm512_add_epi32(s01, s23));
        }
        // 32-wide (q4/vbit groups are exactly 32 bytes).
        if j + 32 <= n {
            let wv = _mm256_loadu_si256(w.as_ptr().add(j) as *const __m256i);
            let xv = _mm256_loadu_si256(xq.as_ptr().add(j) as *const __m256i);
            let d = _mm256_dpbusd_epi32(
                _mm256_setzero_si256(),
                _mm256_abs_epi8(wv),
                _mm256_sign_epi8(xv, wv),
            );
            let hi128 = _mm256_extracti128_si256::<1>(d);
            let s128 = _mm_add_epi32(_mm256_castsi256_si128(d), hi128);
            let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
            let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
            total += _mm_cvtsi128_si32(s32);
            j += 32;
        }
        while j < n {
            total += (w[j] as i8) as i32 * xq[j] as i32;
            j += 1;
        }
        total
    }
}

/// i8 row · f32 x via AVX2/FMA (x86 mirror of `dot_i8_f32_neon`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_i8_f32_avx2(w: &[u8], x: &[f32]) -> f32 {
    // SAFETY: callers uphold slice-length contracts (see call sites).
    unsafe {
        use core::arch::x86_64::*;
        let n = x.len();
        let wp = w.as_ptr();
        let xp = x.as_ptr();
        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut j = 0usize;
        while j + 16 <= n {
            let wb = _mm_loadu_si128(wp.add(j) as *const __m128i);
            let lo = _mm256_cvtepi8_epi32(wb);
            let hi = _mm256_cvtepi8_epi32(_mm_srli_si128::<8>(wb));
            a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(lo), _mm256_loadu_ps(xp.add(j)), a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(hi), _mm256_loadu_ps(xp.add(j + 8)), a1);
            j += 16;
        }
        let acc = _mm256_add_ps(a0, a1);
        let hi128 = _mm256_extractf128_ps::<1>(acc);
        let s128 = _mm_add_ps(_mm256_castps256_ps128(acc), hi128);
        let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
        let s32 = _mm_add_ss(s64, _mm_shuffle_ps::<1>(s64, s64));
        let mut sum = _mm_cvtss_f32(s32);
        while j < n {
            sum += (*wp.add(j) as i8) as f32 * *xp.add(j);
            j += 1;
        }
        sum
    }
}

/// int8(weight)·int8(activation) → i32 via AVX2 maddubs — the x86
/// analogue of the SDOT A8W8 path. `maddubs` takes u8×i8, so the
/// standard sign trick applies: |w| × sign(x, w) ≡ w × x per lane.
/// Pair saturation is safe: |w|≤128, |x|≤127 → 2·128·127 < 32767.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_i8_avx2(w: &[u8], xq: &[i8]) -> i32 {
    // SAFETY: callers uphold slice-length contracts (see call sites).
    unsafe {
        use core::arch::x86_64::*;
        let n = w.len();
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_si256();
        let mut j = 0usize;
        while j + 32 <= n {
            let wv = _mm256_loadu_si256(w.as_ptr().add(j) as *const __m256i);
            let xv = _mm256_loadu_si256(xq.as_ptr().add(j) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(xv, wv));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones));
            j += 32;
        }
        let hi128 = _mm256_extracti128_si256::<1>(acc);
        let s128 = _mm_add_epi32(_mm256_castsi256_si128(acc), hi128);
        let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
        let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
        let mut s = _mm_cvtsi128_si32(s32);
        while j < n {
            s += (w[j] as i8) as i32 * xq[j] as i32;
            j += 1;
        }
        s
    }
}

/// smmla 2×4: one instruction covers a 2-row × 2-activation × 8-deep
/// tile (32 MACs vs sdot's 16) — the weight pair loads once per 8-k
/// slice as a combined 2×8 register and meets two activation pairs.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn dot_i8_smmla_2x4(w0: &[u8], w1: &[u8], xs: [&[i8]; 4]) -> [[i32; 4]; 2] {
    // SAFETY: callers uphold slice-length contracts.
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let n = w0.len();
        let w0p = w0.as_ptr() as *const i8;
        let w1p = w1.as_ptr() as *const i8;
        // acc01 holds [c(r0,x0) c(r0,x1) c(r1,x0) c(r1,x1)]; acc23 the
        // same for x2/x3.
        let mut acc01 = vdupq_n_s32(0);
        let mut acc23 = vdupq_n_s32(0);
        let mut i = 0usize;
        while i + 8 <= n {
            let wa = vcombine_s8(vld1_s8(w0p.add(i)), vld1_s8(w1p.add(i)));
            let xb01 = vcombine_s8(
                vld1_s8(xs[0].as_ptr().add(i)),
                vld1_s8(xs[1].as_ptr().add(i)),
            );
            let xb23 = vcombine_s8(
                vld1_s8(xs[2].as_ptr().add(i)),
                vld1_s8(xs[3].as_ptr().add(i)),
            );
            asm!(
                "smmla {a01:v}.4s, {w:v}.16b, {x01:v}.16b",
                "smmla {a23:v}.4s, {w:v}.16b, {x23:v}.16b",
                a01 = inout(vreg) acc01, a23 = inout(vreg) acc23,
                w = in(vreg) wa, x01 = in(vreg) xb01, x23 = in(vreg) xb23,
                options(pure, nomem, nostack),
            );
            i += 8;
        }
        let mut out = [[0i32; 4]; 2];
        let a01: [i32; 4] = core::mem::transmute(acc01);
        let a23: [i32; 4] = core::mem::transmute(acc23);
        out[0][0] = a01[0];
        out[0][1] = a01[1];
        out[1][0] = a01[2];
        out[1][1] = a01[3];
        out[0][2] = a23[0];
        out[0][3] = a23[1];
        out[1][2] = a23[2];
        out[1][3] = a23[3];
        if i < n {
            for (k, x) in xs.iter().enumerate() {
                for j in i..n {
                    out[0][k] += (w0[j] as i8) as i32 * x[j] as i32;
                    out[1][k] += (w1[j] as i8) as i32 * x[j] as i32;
                }
            }
        }
        out
    }
}

/// ARM twin of the x86 blocked prefill GEMM: two weight rows stay in
/// registers across four activation streams, eight sdot accumulators.
/// (The per-row form re-read each W row once per activation.)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_i8_sdot_2x4(w0: &[u8], w1: &[u8], xs: [&[i8]; 4]) -> [[i32; 4]; 2] {
    // SAFETY: callers uphold slice-length contracts.
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let n = w0.len();
        let w0p = w0.as_ptr() as *const i8;
        let w1p = w1.as_ptr() as *const i8;
        let mut acc = [[vdupq_n_s32(0); 4]; 2];
        let mut i = 0usize;
        while i + 16 <= n {
            let wv0 = vld1q_s8(w0p.add(i));
            let wv1 = vld1q_s8(w1p.add(i));
            for (k, x) in xs.iter().enumerate() {
                let xv = vld1q_s8(x.as_ptr().add(i));
                let (mut a0, mut a1) = (acc[0][k], acc[1][k]);
                asm!(
                    "sdot {a0:v}.4s, {w0:v}.16b, {x:v}.16b",
                    "sdot {a1:v}.4s, {w1:v}.16b, {x:v}.16b",
                    a0 = inout(vreg) a0, a1 = inout(vreg) a1,
                    w0 = in(vreg) wv0, w1 = in(vreg) wv1, x = in(vreg) xv,
                    options(pure, nomem, nostack),
                );
                acc[0][k] = a0;
                acc[1][k] = a1;
            }
            i += 16;
        }
        let mut out = [[0i32; 4]; 2];
        for r in 0..2 {
            for k in 0..4 {
                out[r][k] = vaddvq_s32(acc[r][k]);
            }
        }
        if i < n {
            for (k, x) in xs.iter().enumerate() {
                for j in i..n {
                    out[0][k] += (w0[j] as i8) as i32 * x[j] as i32;
                    out[1][k] += (w1[j] as i8) as i32 * x[j] as i32;
                }
            }
        }
        out
    }
}

/// Blocked 2 weight rows × 4 activations for the prefill GEMM
/// (roadmap P0: packed panels + multi-row accumulators). The two rows'
/// abs() live in registers across all four activation streams; the
/// sign-fixup is recomputed per pair (the price of the maddubs trick).
/// Returns raw i8·i8 dots; the caller applies scales and outliers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_i8_avx2_2x4(w0: &[u8], w1: &[u8], xs: [&[i8]; 4]) -> [[i32; 4]; 2] {
    // SAFETY: callers uphold slice-length contracts.
    unsafe {
        use core::arch::x86_64::*;
        let n = w0.len();
        let ones = _mm256_set1_epi16(1);
        let mut acc = [[_mm256_setzero_si256(); 4]; 2];
        let mut j = 0usize;
        while j + 32 <= n {
            let wv0 = _mm256_loadu_si256(w0.as_ptr().add(j) as *const __m256i);
            let wv1 = _mm256_loadu_si256(w1.as_ptr().add(j) as *const __m256i);
            let aw0 = _mm256_abs_epi8(wv0);
            let aw1 = _mm256_abs_epi8(wv1);
            for (k, x) in xs.iter().enumerate() {
                let xv = _mm256_loadu_si256(x.as_ptr().add(j) as *const __m256i);
                let p0 = _mm256_maddubs_epi16(aw0, _mm256_sign_epi8(xv, wv0));
                acc[0][k] = _mm256_add_epi32(acc[0][k], _mm256_madd_epi16(p0, ones));
                let p1 = _mm256_maddubs_epi16(aw1, _mm256_sign_epi8(xv, wv1));
                acc[1][k] = _mm256_add_epi32(acc[1][k], _mm256_madd_epi16(p1, ones));
            }
            j += 32;
        }
        let mut out = [[0i32; 4]; 2];
        for r in 0..2 {
            for k in 0..4 {
                let a = acc[r][k];
                let hi128 = _mm256_extracti128_si256::<1>(a);
                let s128 = _mm_add_epi32(_mm256_castsi256_si128(a), hi128);
                let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
                let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
                out[r][k] = _mm_cvtsi128_si32(s32);
            }
        }
        if j < n {
            for (k, x) in xs.iter().enumerate() {
                for i in j..n {
                    out[0][k] += (w0[i] as i8) as i32 * x[i] as i32;
                    out[1][k] += (w1[i] as i8) as i32 * x[i] as i32;
                }
            }
        }
        out
    }
}

/// AVX2/VNNI q8 row dot with exact outlier correction (x86 mirror of
/// `row_dot_sdot` — same A8W8 contract). With AVX-512 VNNI the row goes
/// through the bias trick: Σ(w+128)·x via pure `vpdpbusd` (no per-lane
/// sign fixups), corrected by −128·Σx with Σx precomputed per split.
#[cfg(target_arch = "x86_64")]
#[inline]
fn row_dot_avx2(row: &[u8], act: &SplitAct) -> f32 {
    let dot = if avx512vnni_enabled() && row.len() >= 64 {
        (unsafe { dot_u8p128_i8_vnni(row, &act.xq) }) - 128 * act.xsum
    } else {
        unsafe { dot_i8_i8_avx2(row, &act.xq) }
    };
    let mut acc = dot as f32 * act.sx;
    for &(j, xv) in &act.outliers {
        acc += (row[j] as i8) as f32 * xv;
    }
    acc
}

/// Σ (w[i]+128)·x[i] via pure `vpdpbusd` — the caller subtracts
/// 128·Σx. Four independent accumulators (dpbusd is ~5-cycle latency;
/// a single-acc loop runs latency-bound, measured on Granite Rapids).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn dot_u8p128_i8_vnni(w: &[u8], xq: &[i8]) -> i32 {
    // SAFETY: callers uphold slice-length contracts (see call sites).
    unsafe {
        use core::arch::x86_64::*;
        let n = w.len();
        let flip = _mm512_set1_epi8(-128); // XOR 0x80: i8 w → u8 (w+128)
        #[inline(always)]
        unsafe fn step(
            w: *const u8,
            x: *const i8,
            flip: core::arch::x86_64::__m512i,
            acc: core::arch::x86_64::__m512i,
        ) -> core::arch::x86_64::__m512i {
            unsafe {
                use core::arch::x86_64::*;
                let wv = _mm512_xor_si512(_mm512_loadu_si512(w as *const _), flip);
                _mm512_dpbusd_epi32(acc, wv, _mm512_loadu_si512(x as *const _))
            }
        }
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm512_setzero_si512(),
            _mm512_setzero_si512(),
            _mm512_setzero_si512(),
            _mm512_setzero_si512(),
        );
        let mut j = 0usize;
        while j + 256 <= n {
            a0 = step(w.as_ptr().add(j), xq.as_ptr().add(j), flip, a0);
            a1 = step(w.as_ptr().add(j + 64), xq.as_ptr().add(j + 64), flip, a1);
            a2 = step(w.as_ptr().add(j + 128), xq.as_ptr().add(j + 128), flip, a2);
            a3 = step(w.as_ptr().add(j + 192), xq.as_ptr().add(j + 192), flip, a3);
            j += 256;
        }
        while j + 64 <= n {
            a0 = step(w.as_ptr().add(j), xq.as_ptr().add(j), flip, a0);
            j += 64;
        }
        let mut total = _mm512_reduce_add_epi32(_mm512_add_epi32(
            _mm512_add_epi32(a0, a1),
            _mm512_add_epi32(a2, a3),
        ));
        // Scalar tail: (w as i8) + 128 ≡ (w as u8) ^ 0x80.
        while j < n {
            total += ((w[j] ^ 0x80) as i32) * xq[j] as i32;
            j += 1;
        }
        total
    }
}

/// One q4 row via AVX2: nibbles → centered i8 (unpacklo/hi restores the
/// writer's flat order, same as the NEON vzip pair), maddubs against
/// the pre-quantized activation group, × the group's f16 scale. Pair
/// saturation safe: |w|≤8, |x|≤127 → 2·8·127 ≪ 32767. Mirror of
/// `dot_q4_row_sdot`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4_row_avx2(packed: &[u8], scales: &[u8], g0: usize, gpr: usize, xq: &[i8]) -> f32 {
    // SAFETY: callers uphold slice-length contracts (16 packed bytes and
    // 2 scale bytes per group; xq.len() == gpr·GROUP_SIZE).
    unsafe {
        use core::arch::x86_64::*;
        let lomask = _mm_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let ones = _mm256_set1_epi16(1);
        let mut acc = 0f32;
        for gi in 0..gpr {
            let g = g0 + gi;
            let s = f16_to_f32(u16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]));
            let b = _mm_loadu_si128(packed.as_ptr().add(g * 16) as *const __m128i);
            let lo = _mm_and_si128(b, lomask);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(b), lomask);
            let w = _mm256_sub_epi8(
                _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)),
                eight,
            );
            let x = _mm256_loadu_si256(xq.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
            let d = _mm256_madd_epi16(p16, ones);
            let hi128 = _mm256_extracti128_si256::<1>(d);
            let s128 = _mm_add_epi32(_mm256_castsi256_si128(d), hi128);
            let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
            let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
            acc += _mm_cvtsi128_si32(s32) as f32 * s;
        }
        acc
    }
}

/// Two-activation q4 row via AVX2: nibbles unpacked ONCE per group,
/// both activations dotted against the same centered i8 register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4_row_avx2_2(
    packed: &[u8],
    scales: &[u8],
    g0: usize,
    gpr: usize,
    xq1: &[i8],
    xq2: &[i8],
) -> (f32, f32) {
    // SAFETY: callers uphold slice-length contracts (see dot_q4_row_avx2).
    unsafe {
        use core::arch::x86_64::*;
        let lomask = _mm_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let ones = _mm256_set1_epi16(1);
        let (mut acc1, mut acc2) = (0f32, 0f32);
        #[inline(always)]
        unsafe fn hsum(d: core::arch::x86_64::__m256i) -> i32 {
            unsafe {
                use core::arch::x86_64::*;
                let hi128 = _mm256_extracti128_si256::<1>(d);
                let s128 = _mm_add_epi32(_mm256_castsi256_si128(d), hi128);
                let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
                let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
                _mm_cvtsi128_si32(s32)
            }
        }
        for gi in 0..gpr {
            let g = g0 + gi;
            let s = f16_to_f32(u16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]));
            let b = _mm_loadu_si128(packed.as_ptr().add(g * 16) as *const __m128i);
            let lo = _mm_and_si128(b, lomask);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(b), lomask);
            let w = _mm256_sub_epi8(
                _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)),
                eight,
            );
            let aw = _mm256_abs_epi8(w);
            let x1 = _mm256_loadu_si256(xq1.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
            let x2 = _mm256_loadu_si256(xq2.as_ptr().add(gi * GROUP_SIZE) as *const __m256i);
            let d1 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(x1, w)), ones);
            let d2 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(x2, w)), ones);
            acc1 += hsum(d1) as f32 * s;
            acc2 += hsum(d2) as f32 * s;
        }
        (acc1, acc2)
    }
}

/// One q8 row range via AVX2 (x86 mirror of `q8_range_sdot`).
#[cfg(target_arch = "x86_64")]
fn q8_range_avx2(
    q: &[u8],
    row_scale: &[f32],
    act: &SplitAct,
    cols: usize,
    out_addr: SendMut,
    start: usize,
    end: usize,
) {
    for o in start..end {
        let v = row_dot_avx2(&q[o * cols..(o + 1) * cols], act) * row_scale[o];
        // SAFETY: disjoint row ranges per worker.
        unsafe { *out_addr.at(o) = v };
    }
}

/// Two-input q8 row range via AVX2 (x86 mirror of `q8_range2_sdot`).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn q8_range2_avx2(
    q: &[u8],
    row_scale: &[f32],
    a1: &SplitAct,
    a2: &SplitAct,
    cols: usize,
    p1: SendMut,
    p2: SendMut,
    start: usize,
    end: usize,
) {
    for o in start..end {
        let row = &q[o * cols..(o + 1) * cols];
        // SAFETY: disjoint row ranges per worker.
        unsafe {
            *p1.at(o) = row_dot_avx2(row, a1) * row_scale[o];
            *p2.at(o) = row_dot_avx2(row, a2) * row_scale[o];
        }
    }
}

// ───────────────────── A8W8 SDOT path (port of vmfcore, ×1.78 decode) ─────────────────────

/// ARMv8.6 i8mm (smmla): 32 int8 MACs per instruction vs sdot's 16 —
/// yet MEASURED 2.4× SLOWER than the blocked sdot on Apple silicon
/// (108 vs 264 GF/s): the on-the-fly vcombine packing and the two-
/// accumulator dependency chain swamp the MAC advantage, and Apple's
/// four SIMD pipes already keep sdot fed. OPT-IN (CMF_I8MM=1) for
/// field trials on Cortex-A710/X-class parts with two pipes, where the
/// balance may differ; a pre-interleaved weight layout (repack infra)
/// is the known path if it ever earns its keep.
#[cfg(target_arch = "aarch64")]
fn i8mm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CMF_I8MM").map(|v| v == "1").unwrap_or(false)
            && std::arch::is_aarch64_feature_detected!("i8mm")
    })
}

/// SDOT enabled? Default ON when the CPU has ARMv8.2 dotprod;
/// `CMF_SDOT=0` disables (falls back to i8×f32 NEON).
/// (On non-ARM release builds only the test tolerance switch calls it.)
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn sdot_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let want = std::env::var("CMF_SDOT").map(|v| v != "0").unwrap_or(true);
        if !want {
            return false;
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                return true;
            }
            #[cfg(target_os = "android")]
            {
                if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
                    if cpuinfo.lines().any(|l| {
                        (l.starts_with("Features") || l.starts_with("features"))
                            && l.contains("asimddp")
                    }) {
                        return true;
                    }
                }
            }
            false
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
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
    /// Σ xq — the VNNI bias-trick correction (`(w+128)·x` sums need
    /// `−128·Σx`); one i32 per split, computed once per matvec.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    xsum: i32,
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

thread_local! {
    /// One scratch row per WORKER, kept for the life of the thread.
    ///
    /// The kernels take a row of group scales per dispatch, and a fresh
    /// `vec![0f32; gpr]` inside the closure is one allocation per worker per
    /// dispatch — on the release checkpoint about six thousand a token, a
    /// quarter of everything the benchmark counts.
    static KROW: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Borrow `n` floats of the calling worker's scratch. Nothing inside a
/// kernel body borrows it again, which is what keeps the RefCell honest.
#[inline]
fn with_krow<R>(n: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    KROW.with(|s| {
        let mut b = s.borrow_mut();
        if b.len() < n {
            b.resize(n, 0.0);
        }
        f(&mut b[..n])
    })
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
        xq.extend(
            x.iter()
                .map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8),
        );
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
    let xsum = xq.iter().map(|&v| v as i32).sum();
    SplitAct {
        xq,
        sx,
        outliers,
        xsum,
    }
}

fn split_act_q8_2f(x: &[f32], col: &[f32]) -> SplitAct {
    let n = x.len();
    let rms = (x
        .iter()
        .zip(col)
        .map(|(&a, &c)| {
            let v = a * c;
            (v * v) as f64
        })
        .sum::<f64>()
        / n.max(1) as f64)
        .sqrt() as f32;
    let thr = 8.0 * rms;

    let mut outliers = Vec::new();
    let mut amax = 0f32;
    for (j, (&a, &c)) in x.iter().zip(col).enumerate() {
        let v = a * c;
        let s = v.abs();
        if s > thr {
            outliers.push((j, v));
        } else if s > amax {
            amax = s;
        }
    }

    let sx = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let inv = 1.0 / sx;
    let mut xq = XQ_FREE.with(|f| f.borrow_mut().pop()).unwrap_or_default();
    xq.clear();
    xq.reserve(n);
    if outliers.is_empty() {
        xq.extend(
            x.iter()
                .zip(col)
                .map(|(&a, &c)| ((a * c) * inv).round().clamp(-127.0, 127.0) as i8),
        );
    } else {
        xq.extend(x.iter().zip(col).map(|(&a, &c)| {
            let v = a * c;
            if v.abs() > thr {
                0
            } else {
                (v * inv).round().clamp(-127.0, 127.0) as i8
            }
        }));
    }
    let xsum = xq.iter().map(|&v| v as i32).sum();
    SplitAct {
        xq,
        sx,
        outliers,
        xsum,
    }
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
        let (mut a0, mut a1, mut a2, mut a3) = (
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
        );
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
        let (mut a0, mut a1, mut a2, mut a3) = (
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
        );
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
        let mut r = [
            vaddvq_s32(a0),
            vaddvq_s32(a1),
            vaddvq_s32(a2),
            vaddvq_s32(a3),
        ];
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

/// 4 interleaved rows in one pass: the repacked group is [r0[c], r1[c],
/// r2[c], r3[c]] per 16-byte chunk, so each iteration reads ONE 64-byte
/// line plus the shared activation chunk — a single sequential weight
/// stream per worker. Per-row accumulation is the same one-accumulator
/// scheme as `dot_i8_sdot_4rows`; integer sums are exact, so outputs
/// are bit-identical to the mmap-layout kernel.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_i8_sdot_4rows_il(g: &[u8], xq: &[i8]) -> [i32; 4] {
    // SAFETY: callers uphold slice-length contracts (g.len() == 4·n,
    // n % 16 == 0 — guaranteed by the repack gate).
    unsafe {
        use core::arch::aarch64::*;
        use core::arch::asm;
        let n = xq.len();
        let px = xq.as_ptr();
        let pg = g.as_ptr() as *const i8;
        let (mut a0, mut a1, mut a2, mut a3) = (
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
        );
        let mut i = 0;
        while i + 16 <= n {
            let x = vld1q_s8(px.add(i));
            let base = pg.add(4 * i);
            let v0 = vld1q_s8(base);
            let v1 = vld1q_s8(base.add(16));
            let v2 = vld1q_s8(base.add(32));
            let v3 = vld1q_s8(base.add(48));
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
        [
            vaddvq_s32(a0),
            vaddvq_s32(a1),
            vaddvq_s32(a2),
            vaddvq_s32(a3),
        ]
    }
}

/// One q8 row range via SDOT (4-row blocks + tail) — the body of
/// `qmatvec`'s hot loop, extracted so multi-matrix jobs can drive the
/// SAME kernel for several tensors under one pool dispatch. `rep` — the
/// load-time interleaved repack (empty = mmap layout only); rows outside
/// full 4-row groups always come from the mmap layout.
#[cfg(target_arch = "aarch64")]
fn q8_range_sdot(
    q: &[u8],
    rep: &[u8],
    row_scale: &[f32],
    act: &SplitAct,
    cols: usize,
    out_addr: SendMut,
    start: usize,
    end: usize,
) {
    let mut o = start;
    // Leading rows to the group boundary (repack path only): the pool
    // splits row ranges arbitrarily, groups are absolute.
    if !rep.is_empty() {
        while o < end && o % 4 != 0 {
            let v = row_dot_sdot(&q[o * cols..(o + 1) * cols], act) * row_scale[o];
            unsafe { *out_addr.at(o) = v };
            o += 1;
        }
    }
    while o + 4 <= end {
        let r = if rep.is_empty() {
            unsafe {
                dot_i8_sdot_4rows(
                    &q[o * cols..(o + 1) * cols],
                    &q[(o + 1) * cols..(o + 2) * cols],
                    &q[(o + 2) * cols..(o + 3) * cols],
                    &q[(o + 3) * cols..(o + 4) * cols],
                    &act.xq,
                )
            }
        } else {
            unsafe { dot_i8_sdot_4rows_il(&rep[o * cols..(o + 4) * cols], &act.xq) }
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
        let v = row_dot_sdot(&q[o * cols..(o + 1) * cols], act) * row_scale[o];
        unsafe { *out_addr.at(o) = v };
        o += 1;
    }
}

/// Two-input q8 row range via SDOT — `qmatvec2`'s hot loop, extracted
/// for the fused pair multi-matrix job (`matvec2_many`).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn q8_range2_sdot(
    q: &[u8],
    row_scale: &[f32],
    a1: &SplitAct,
    a2: &SplitAct,
    cols: usize,
    p1: SendMut,
    p2: SendMut,
    start: usize,
    end: usize,
) {
    for o in start..end {
        let row = &q[o * cols..(o + 1) * cols];
        // SAFETY: disjoint row ranges per worker.
        unsafe {
            *p1.at(o) = row_dot_sdot(row, a1) * row_scale[o];
            *p2.at(o) = row_dot_sdot(row, a2) * row_scale[o];
        }
    }
}

/// Two-input q8 row range, f32 kernel (non-SDOT) — same extraction.
#[allow(clippy::too_many_arguments)]
fn q8_range2_f32(
    q: &[u8],
    row_scale: &[f32],
    x1: &[f32],
    x2: &[f32],
    cols: usize,
    p1: SendMut,
    p2: SendMut,
    start: usize,
    end: usize,
) {
    for o in start..end {
        let row = &q[o * cols..(o + 1) * cols];
        // SAFETY: disjoint row ranges per worker.
        unsafe {
            *p1.at(o) = dot_i8_f32(row, x1) * row_scale[o];
            *p2.at(o) = dot_i8_f32(row, x2) * row_scale[o];
        }
    }
}

/// Scalar/NEON-f32 q8 row range (non-SDOT platforms) — same extraction.
fn q8_range_f32(
    q: &[u8],
    row_scale: &[f32],
    xs: &[f32],
    cols: usize,
    out_addr: SendMut,
    start: usize,
    end: usize,
) {
    for o in start..end {
        let v = dot_i8_f32(&q[o * cols..(o + 1) * cols], xs) * row_scale[o];
        // SAFETY: disjoint row ranges per worker.
        unsafe { *out_addr.at(o) = v };
    }
}

/// One q8 row against a split activation, portable: the per-arch fast
/// dots where they exist, the exact scalar loop elsewhere. The scalar
/// arm is also the test oracle for both fast arms.
#[inline]
fn q8_row_dot(row: &[u8], act: &SplitAct) -> f32 {
    #[cfg(target_arch = "aarch64")]
    return row_dot_sdot(row, act);
    #[cfg(target_arch = "x86_64")]
    return row_dot_avx2(row, act);
    #[allow(unreachable_code)]
    q8_row_dot_scalar(row, act)
}

#[allow(dead_code)]
fn q8_row_dot_scalar(row: &[u8], act: &SplitAct) -> f32 {
    let mut acc = 0i32;
    for (k, &b) in row.iter().enumerate() {
        acc += (b as i8) as i32 * act.xq[k] as i32;
    }
    let mut acc = acc as f32 * act.sx;
    for &(j, xv) in &act.outliers {
        acc += (row[j] as i8) as f32 * xv;
    }
    acc
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
            let (mut a0, mut a1, mut b0, mut b1) = (
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
            );
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
    #[cfg(target_arch = "x86_64")]
    if avx2_enabled() {
        return unsafe { axpy_i8_f32_avx2(acc, row, w) };
    }
    #[allow(unreachable_code)]
    {
        for (a, &b) in acc.iter_mut().zip(row) {
            *a += w * b as f32;
        }
    }
}

/// i8→f32 axpy via AVX2/FMA (x86 mirror of `axpy_i8_f32_neon`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_i8_f32_avx2(acc: &mut [f32], row: &[i8], w: f32) {
    // SAFETY: callers uphold slice-length contracts (see call sites).
    unsafe {
        use core::arch::x86_64::*;
        let n = acc.len().min(row.len());
        let ap = acc.as_mut_ptr();
        let rp = row.as_ptr();
        let wv = _mm256_set1_ps(w);
        let mut j = 0usize;
        while j + 16 <= n {
            let rb = _mm_loadu_si128(rp.add(j) as *const __m128i);
            let lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(rb));
            let hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(rb)));
            let v0 = _mm256_fmadd_ps(wv, lo, _mm256_loadu_ps(ap.add(j)));
            let v1 = _mm256_fmadd_ps(wv, hi, _mm256_loadu_ps(ap.add(j + 8)));
            _mm256_storeu_ps(ap.add(j), v0);
            _mm256_storeu_ps(ap.add(j + 8), v1);
            j += 16;
        }
        while j < n {
            *ap.add(j) += w * (*rp.add(j)) as f32;
            j += 1;
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
    #[cfg(target_arch = "x86_64")]
    if avx2_enabled() {
        return unsafe { dot_i8_f32_avx2(w, x) };
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
        let (mut a0, mut a1, mut a2, mut a3) = (
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
        );
        let mut j = 0usize;
        while j + 16 <= n {
            let wb = vld1q_s8(wp.add(j));
            let lo = vmovl_s8(vget_low_s8(wb));
            let hi = vmovl_s8(vget_high_s8(wb));
            let w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
            let w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
            let w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
            let w3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
            a0 = vfmaq_f32(
                a0,
                w0,
                vmulq_f32(vld1q_f32(xp.add(j)), vld1q_f32(cp.add(j))),
            );
            a1 = vfmaq_f32(
                a1,
                w1,
                vmulq_f32(vld1q_f32(xp.add(j + 4)), vld1q_f32(cp.add(j + 4))),
            );
            a2 = vfmaq_f32(
                a2,
                w2,
                vmulq_f32(vld1q_f32(xp.add(j + 8)), vld1q_f32(cp.add(j + 8))),
            );
            a3 = vfmaq_f32(
                a3,
                w3,
                vmulq_f32(vld1q_f32(xp.add(j + 12)), vld1q_f32(cp.add(j + 12))),
            );
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
        let (mut a0, mut a1, mut a2, mut a3) = (
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
        );
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
    rep: &[u8],
    row_scale: &[f32],
    x: &[f32],
    col_field: &[f32],
    dtype: TensorDtype,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    debug_assert_eq!(out.len(), rows);
    #[cfg(not(target_arch = "aarch64"))]
    let _ = rep;

    #[cfg(target_arch = "aarch64")]
    if sdot_enabled() {
        let act = if dtype == TensorDtype::Q8_2f {
            split_act_q8_2f(x, col_field)
        } else {
            split_act(x)
        };
        let out_addr = SendMut(out.as_mut_ptr());
        let run_range = |start: usize, end: usize| {
            q8_range_sdot(q, rep, row_scale, &act, cols, out_addr, start, end)
        };
        match pool {
            Some(pool) if rows >= 256 => pool.run_rows(rows, &run_range),
            _ => run_range(0, rows),
        }
        return;
    }
    // x86 A8W8 via AVX2 maddubs — same quantized-activation contract as
    // the SDOT path (CMF_AVX2=0 keeps the exact i8×f32 loop).
    #[cfg(target_arch = "x86_64")]
    if avx2_a8w8_enabled() {
        let act = if dtype == TensorDtype::Q8_2f {
            split_act_q8_2f(x, col_field)
        } else {
            split_act(x)
        };
        let out_addr = SendMut(out.as_mut_ptr());
        let run_range = |start: usize, end: usize| {
            q8_range_avx2(q, row_scale, &act, cols, out_addr, start, end)
        };
        match pool {
            Some(pool) if rows >= 256 => pool.run_rows(rows, &run_range),
            _ => run_range(0, rows),
        }
        return;
    }

    prescale_with(x, col_field, dtype, 1, |xs| {
        let out_addr = SendMut(out.as_mut_ptr());
        let run_range = move |start: usize, end: usize| {
            for o in start..end {
                let v = dot_i8_f32(&q[o * cols..(o + 1) * cols], xs) * row_scale[o];
                // SAFETY: disjoint row ranges per worker.
                unsafe { *out_addr.at(o) = v };
            }
        };
        match pool {
            Some(pool) if rows >= 256 => pool.run_rows(rows, &run_range),
            _ => run_range(0, rows),
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn qmatvec2(
    q: &[u8],
    row_scale: &[f32],
    x1: &[f32],
    x2: &[f32],
    col_field: &[f32],
    dtype: TensorDtype,
    rows: usize,
    cols: usize,
    o1: &mut [f32],
    o2: &mut [f32],
    pool: Option<&Pool>,
) {
    #[cfg(target_arch = "aarch64")]
    if sdot_enabled() {
        let a1s = if dtype == TensorDtype::Q8_2f {
            split_act_q8_2f(x1, col_field)
        } else {
            split_act(x1)
        };
        let a2s = if dtype == TensorDtype::Q8_2f {
            split_act_q8_2f(x2, col_field)
        } else {
            split_act(x2)
        };
        let p1 = SendMut(o1.as_mut_ptr());
        let p2 = SendMut(o2.as_mut_ptr());
        let run_range = |start: usize, end: usize| {
            q8_range2_sdot(q, row_scale, &a1s, &a2s, cols, p1, p2, start, end)
        };
        match pool {
            Some(pool) if rows >= 256 => pool.run_rows(rows, &run_range),
            _ => run_range(0, rows),
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if avx2_a8w8_enabled() {
        let a1s = if dtype == TensorDtype::Q8_2f {
            split_act_q8_2f(x1, col_field)
        } else {
            split_act(x1)
        };
        let a2s = if dtype == TensorDtype::Q8_2f {
            split_act_q8_2f(x2, col_field)
        } else {
            split_act(x2)
        };
        let p1 = SendMut(o1.as_mut_ptr());
        let p2 = SendMut(o2.as_mut_ptr());
        let run_range = |start: usize, end: usize| {
            q8_range2_avx2(q, row_scale, &a1s, &a2s, cols, p1, p2, start, end)
        };
        match pool {
            Some(pool) if rows >= 256 => pool.run_rows(rows, &run_range),
            _ => run_range(0, rows),
        }
        return;
    }

    prescale_with(x1, col_field, dtype, 1, |x1s| {
        prescale_with(x2, col_field, dtype, 2, |x2s| {
            let p1 = SendMut(o1.as_mut_ptr());
            let p2 = SendMut(o2.as_mut_ptr());
            let run_range = move |start: usize, end: usize| {
                for o in start..end {
                    let row = &q[o * cols..(o + 1) * cols];
                    let s1 = dot_i8_f32(row, x1s) * row_scale[o];
                    let s2 = dot_i8_f32(row, x2s) * row_scale[o];
                    // SAFETY: disjoint row ranges per worker.
                    unsafe {
                        *p1.at(o) = s1;
                        *p2.at(o) = s2;
                    }
                }
            };
            match pool {
                Some(pool) if rows >= 256 => pool.run_rows(rows, &run_range),
                _ => run_range(0, rows),
            }
        });
    });
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
    fn q2tp_i8_dot_matches_exact_on_grid() {
        // On-grid activations (±1 → sx=1/127, xq=±127 dequantizes
        // exactly, no outliers) must make the integer path agree with
        // the exact scalar walk to f32 rounding.
        let (rows, cols) = (5, 64);
        let gpr = cols / GROUP_SIZE;
        // Synthetic codes plane + a flat ladder: scales_into is not under
        // test here, so drive dot_q2tp_row_i8 / q2tp_row_exact directly
        // with hand-made scales.
        let chunks: Vec<u8> = (0..rows * gpr * Q2TP_CHUNK)
            .map(|i| (i as u32).wrapping_mul(2654435761) as u8)
            .collect();
        let scales: Vec<f32> = (0..gpr).map(|g| 0.5 + g as f32 * 0.25).collect();
        let x: Vec<f32> = (0..cols)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let act = split_act(&x);
        assert!(
            act.outliers.is_empty(),
            "on-grid input must have no outliers"
        );
        let gsum = q1_group_sums(&act.xq, gpr);
        for r in 0..rows {
            let exact = q2tp_row_exact(&chunks, r, gpr, &x, &scales);
            let fast = dot_q2tp_row_i8(&chunks, r, gpr, &act.xq, &gsum, &scales) * act.sx;
            assert!(
                (exact - fast).abs() <= exact.abs() * 1e-5 + 1e-5,
                "row {r}: exact {exact} vs i8 {fast}"
            );
        }
    }

    #[test]
    fn q8_row_dot_fast_matches_scalar() {
        // The per-arch fast dot must agree with the exact scalar oracle
        // (same contract the fused q8 FFN arm rides on).
        let cols = 96;
        let row: Vec<u8> = (0..cols)
            .map(|i| ((i * 37 % 251) - 125) as i8 as u8)
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.13).sin()).collect();
        let act = split_act(&x);
        let fast = q8_row_dot(&row, &act);
        let scalar = q8_row_dot_scalar(&row, &act);
        assert!(
            (fast - scalar).abs() <= scalar.abs() * 1e-5 + 1e-5,
            "fast {fast} vs scalar {scalar}"
        );
    }

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
        qmatvec(
            &w,
            &[],
            &scales,
            &x,
            &[],
            TensorDtype::Q8Row,
            rows,
            cols,
            &mut a,
            None,
        );
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
    fn q1_tbl_fast_path_matches_reference() {
        // gpr = 8 exercises the TBL pair-load fast loop, and the LAST
        // row's final 4-tile window trips the 4B-overread guard (the
        // payload ends exactly at the last tile) — both paths must
        // agree with the dequant reference.
        let (rows, cols) = (5, 256);
        let gpr = cols / GROUP_SIZE;
        let mut bytes = Vec::new();
        for t in 0..rows * gpr {
            let s = 0.007 + (t % 11) as f32 * 0.004;
            bytes.extend_from_slice(&cortiq_core::quant::f32_to_f16(s).to_le_bytes());
            for j in 0..4 {
                bytes.push(((t * 53 + j * 89 + 7) % 249) as u8);
            }
        }
        let x: Vec<f32> = (0..cols)
            .map(|i| if (i * 5) % 7 < 3 { 1.0 } else { -1.0 })
            .collect();
        let mut w = vec![0.0f32; rows * cols];
        cortiq_core::quant::dequant_q1(&bytes, &mut w);
        let mut got = vec![0.0f32; rows];
        q1_matvec(&bytes, &x, rows, cols, &mut got, None);
        for o in 0..rows {
            let expect: f32 = (0..cols).map(|j| w[o * cols + j] * x[j]).sum();
            assert!(
                (got[o] - expect).abs() < 1e-3 * expect.abs().max(1e-3),
                "row {o}: {} vs {expect}",
                got[o]
            );
        }
        // Blocked 1×4 batch (b=5: one quad + remainder) must equal the
        // single-matvec path bit-for-bit.
        let b = 5usize;
        let mut xs_all = Vec::new();
        for bi in 0..b {
            xs_all.extend(x.iter().map(|v| if bi % 2 == 0 { *v } else { -*v }));
        }
        let mut mm = vec![0.0f32; b * rows];
        q1_matmat(&bytes, &xs_all, b, rows, cols, &mut mm, None);
        for bi in 0..b {
            let mut single = vec![0.0f32; rows];
            q1_matvec(
                &bytes,
                &xs_all[bi * cols..(bi + 1) * cols],
                rows,
                cols,
                &mut single,
                None,
            );
            assert_eq!(&mm[bi * rows..(bi + 1) * rows], &single[..], "stream {bi}");
        }
    }

    #[test]
    fn q1_kernels_match_exact_reference() {
        // Synthetic q1 payload: 6-byte tiles [f16 scale][4B bits].
        let (rows, cols) = (7, 96);
        let gpr = cols / GROUP_SIZE;
        let mut bytes = Vec::new();
        for t in 0..rows * gpr {
            let s = 0.01 + (t % 13) as f32 * 0.003;
            bytes.extend_from_slice(&cortiq_core::quant::f32_to_f16(s).to_le_bytes());
            for j in 0..4 {
                bytes.push(((t * 31 + j * 97) % 251) as u8);
            }
        }
        // On-grid activations (±1, amax 1) → the SDOT path is exact.
        let x: Vec<f32> = (0..cols)
            .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        // Reference through the core dequant.
        let mut w = vec![0.0f32; rows * cols];
        cortiq_core::quant::dequant_q1(&bytes, &mut w);
        let mut expect = vec![0.0f32; rows];
        for o in 0..rows {
            expect[o] = (0..cols).map(|j| w[o * cols + j] * x[j]).sum();
        }
        let mut got = vec![0.0f32; rows];
        q1_matvec(&bytes, &x, rows, cols, &mut got, None);
        for o in 0..rows {
            assert!(
                (got[o] - expect[o]).abs() < 1e-3 * expect[o].abs().max(1e-3),
                "row {o}: {} vs {}",
                got[o],
                expect[o]
            );
        }
        // Pair and batch paths agree with the single path.
        let x2: Vec<f32> = x.iter().map(|v| -v).collect();
        let (mut a1, mut a2) = (vec![0.0f32; rows], vec![0.0f32; rows]);
        q1_matvec2(&bytes, &x, &x2, rows, cols, &mut a1, &mut a2, None);
        assert_eq!(a1, got);
        let mut xs = x.clone();
        xs.extend_from_slice(&x2);
        let mut mm = vec![0.0f32; 2 * rows];
        q1_matmat(&bytes, &xs, 2, rows, cols, &mut mm, None);
        assert_eq!(&mm[..rows], got.as_slice());
        assert_eq!(&mm[rows..], a2.as_slice());
    }

    #[test]
    fn repack_is_bit_identical() {
        // The interleaved-repack kernel must produce EXACTLY the same
        // bits as the mmap-layout kernel: integer accumulation is order-
        // exact, the f32 epilogue is identical. Odd rows exercise the
        // tail; direct range calls exercise unaligned pool splits.
        let (rows, cols) = (267, 96); // 66 groups + 3 tail rows, cols % 16 == 0
        let w: Vec<u8> = (0..rows * cols)
            .map(|i| (((i * 89) % 253) as i32 - 126) as i8 as u8)
            .collect();
        let scales: Vec<f32> = (0..rows).map(|o| 0.003 + o as f32 * 0.0007).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.37).sin() * 2.0).collect();
        let rep = q8_repack_layout(&w, rows, cols);
        // Group interleave round-trips.
        for g in 0..rows / 4 {
            for c in 0..cols / 16 {
                for lane in 0..4 {
                    assert_eq!(
                        &rep[g * 4 * cols + c * 64 + lane * 16
                            ..g * 4 * cols + c * 64 + lane * 16 + 16],
                        &w[(g * 4 + lane) * cols + c * 16..(g * 4 + lane) * cols + c * 16 + 16],
                    );
                }
            }
        }
        let mut a = vec![0.0f32; rows];
        qmatvec(
            &w,
            &[],
            &scales,
            &x,
            &[],
            TensorDtype::Q8Row,
            rows,
            cols,
            &mut a,
            None,
        );
        let mut b = vec![0.0f32; rows];
        qmatvec(
            &w,
            &rep,
            &scales,
            &x,
            &[],
            TensorDtype::Q8Row,
            rows,
            cols,
            &mut b,
            None,
        );
        assert_eq!(a, b, "full-range repack output diverged");

        #[cfg(target_arch = "aarch64")]
        if sdot_enabled() {
            // Unaligned range split (pool workers get arbitrary bounds).
            let act = split_act(&x);
            let mut c1 = vec![0.0f32; rows];
            let mut c2 = vec![0.0f32; rows];
            q8_range_sdot(
                &w,
                &[],
                &scales,
                &act,
                cols,
                SendMut(c1.as_mut_ptr()),
                3,
                rows - 2,
            );
            q8_range_sdot(
                &w,
                &rep,
                &scales,
                &act,
                cols,
                SendMut(c2.as_mut_ptr()),
                3,
                rows - 2,
            );
            assert_eq!(c1, c2, "unaligned-range repack output diverged");
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
        qmatvec(
            &w,
            &[],
            &scales,
            &x,
            &[],
            TensorDtype::Q8Row,
            rows,
            cols,
            &mut a,
            None,
        );
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
        let tol = if a8w8_enabled() { 6e-2 } else { 1e-4 };
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
    /// vbit matmat: the blocked 1×4 leg must match the per-row path
    /// (paired env toggle; larger shape so both code paths engage).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn vbit_matmat_blocked_matches_per_row() {
        let (rows, cols, b) = (64usize, 128usize, 9usize);
        let ng = cols / GROUP_SIZE;
        let bits: Vec<u8> = (0..rows).map(|r| [3u8, 4, 5, 6][r % 4]).collect();
        let mut bytes = bits.clone();
        for g in 0..rows * ng {
            let sc = 0.02 + 0.0005 * g as f32;
            bytes.extend_from_slice(&cortiq_core::quant::f32_to_f16(sc).to_le_bytes());
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
            bytes.extend_from_slice(&rowbytes);
        }
        let x: Vec<f32> = (0..b * cols)
            .map(|i| ((i * 13 + 7) % 97) as f32 / 97.0 - 0.5)
            .collect();
        let offsets = vbit_row_offsets(&bytes, rows, cols);
        let mut y_a = vec![0f32; b * rows];
        let mut y_b = vec![0f32; b * rows];
        unsafe { std::env::set_var("CMF_X86_BLOCKED", "1") };
        vbitmatmat(&bytes, &offsets, &x, b, rows, cols, &mut y_a, None);
        unsafe { std::env::set_var("CMF_X86_BLOCKED", "0") };
        vbitmatmat(&bytes, &offsets, &x, b, rows, cols, &mut y_b, None);
        unsafe { std::env::remove_var("CMF_X86_BLOCKED") };
        let max_d = y_a
            .iter()
            .zip(&y_b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-4, "vbit blocked ≠ per-row: max|Δ| = {max_d}");
    }

    /// q4t blocked 1×4 (SDOT on ARM, AVX2 on x86) must equal the
    /// per-row path exactly: same nibble unpack, same group order,
    /// same f32 accumulation — batch == matvec bit-for-bit. b=9 covers
    /// two full 1×4 blocks plus a remainder through the single-row
    /// kernel. (Both paths produce identical output, so the shared
    /// CMF_X86_BLOCKED env var racing with other tests cannot flip
    /// the verdict — worst case both sides take the same path.)
    #[test]
    fn q4t_matmat_blocked_matches_per_row() {
        let (rows, cols, b) = (16usize, 64usize, 9usize);
        let gpr = cols / GROUP_SIZE;
        let mut bytes = vec![0u8; rows * gpr * Q4_TILE];
        for r in 0..rows {
            for g in 0..gpr {
                let t = (r * gpr + g) * Q4_TILE;
                let sc = 0.02 + 0.001 * (r * gpr + g) as f32;
                bytes[t..t + 2].copy_from_slice(&cortiq_core::quant::f32_to_f16(sc).to_le_bytes());
                for k in 0..16 {
                    bytes[t + 2 + k] = ((r * 31 + g * 7 + k * 13) % 251) as u8;
                }
            }
        }
        let x: Vec<f32> = (0..b * cols)
            .map(|i| ((i * 13 + 7) % 97) as f32 / 97.0 - 0.5)
            .collect();
        let mut y_blk = vec![0f32; b * rows];
        let mut y_row = vec![0f32; b * rows];
        unsafe { std::env::set_var("CMF_X86_BLOCKED", "1") };
        q4t_matmat(&bytes, &x, b, rows, cols, &mut y_blk, None);
        unsafe { std::env::set_var("CMF_X86_BLOCKED", "0") };
        q4t_matmat(&bytes, &x, b, rows, cols, &mut y_row, None);
        unsafe { std::env::remove_var("CMF_X86_BLOCKED") };
        assert_eq!(y_blk, y_row, "q4t blocked 1x4 ≠ per-row");
    }

    /// The wide-batch Accelerate arm of q4t_matmat vs a brute-force
    /// f32 dequant matmul: both are f32 GEMMs, so only reduction
    /// order differs — tight tolerance.
    /// A synthetic q4tp payload: random nibbles plus a per-row ladder whose
    /// span varies row to row, so the codes actually exercise the full 0..31
    /// range rather than clustering on one rung.
    fn synth_q4tp(rows: usize, cols: usize) -> Vec<u8> {
        use cortiq_core::quant::{f32_to_f16, q4tp_code_stride, q4tp_put_code};
        let gpr = cols / GROUP_SIZE;
        let stride = q4tp_code_stride(gpr);
        let (params_off, codes_off, _) = q4tp_sections(rows, cols);
        let mut b = vec![0u8; codes_off + rows * stride];
        for r in 0..rows {
            for g in 0..gpr {
                let t = (r * gpr + g) * Q4TP_NIB;
                for k in 0..16 {
                    b[t + k] = ((r * 31 + g * 7 + k * 13) % 251) as u8;
                }
            }
            let lo = -6.0 - 0.03 * (r % 17) as f32;
            let step = 0.01 + 0.004 * (r % 11) as f32;
            let p = params_off + r * 4;
            b[p..p + 2].copy_from_slice(&f32_to_f16(lo).to_le_bytes());
            b[p + 2..p + 4].copy_from_slice(&f32_to_f16(step).to_le_bytes());
            let crow = &mut b[codes_off + r * stride..codes_off + (r + 1) * stride];
            for g in 0..gpr {
                q4tp_put_code(crow, g, (r * 5 + g * 3) % 32);
            }
        }
        b
    }

    /// The same weights re-expressed as q4_tiled, so the proven kernel can
    /// be the reference: each tile stores the ladder scale its code selects.
    /// Only the f16 rounding of that scale separates the two payloads.
    fn q4tp_as_q4t(bytes: &[u8], rows: usize, cols: usize) -> Vec<u8> {
        let gpr = cols / GROUP_SIZE;
        let v = Q4tpView::new(bytes, rows, cols);
        let mut out = vec![0u8; rows * gpr * Q4_TILE];
        let mut sc = vec![0f32; gpr];
        for r in 0..rows {
            v.scales_into(r, gpr, &mut sc);
            for g in 0..gpr {
                let t = (r * gpr + g) * Q4_TILE;
                let s = sc[g];
                out[t..t + 2].copy_from_slice(&cortiq_core::quant::f32_to_f16(s).to_le_bytes());
                let src = (r * gpr + g) * Q4TP_NIB;
                out[t + 2..t + Q4_TILE].copy_from_slice(&v.nib[src..src + Q4TP_NIB]);
            }
        }
        out
    }

    /// The exact (`CMF_SDOT=0`) path must reproduce `dequant_q4tp` to f32
    /// rounding — that scalar routine is the format's definition, and the
    /// kernels re-derive the scale from the ladder independently. Call the
    /// row kernel directly: `matmat` picks the int8 arm when a8w8 is on,
    /// so routing through it would test the other path by accident.
    #[test]
    fn q4tp_exact_path_matches_dequant_reference() {
        let (rows, cols) = (256usize, 512usize);
        let gpr = cols / GROUP_SIZE;
        let bytes = synth_q4tp(rows, cols);
        let mut w = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_q4tp(&bytes, rows, cols, &mut w);

        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 13 + 7) % 97) as f32 / 97.0 - 0.5)
            .collect();
        let v = Q4tpView::new(&bytes, rows, cols);
        let mut sc = vec![0f32; gpr];
        for r in 0..rows {
            v.scales_into(r, gpr, &mut sc);
            let got = q4tp_row_exact(v.nib, r, gpr, &x, &sc);
            let want: f32 = (0..cols).map(|c| w[r * cols + c] * x[c]).sum();
            // These dot products cancel down to ~1e-3 from terms of ~5e-2, so
            // the meaningful yardstick is the summed magnitude, not the result:
            // against the result any reordering of a 512-term f32 sum "fails".
            let mag: f32 = (0..cols).map(|c| (w[r * cols + c] * x[c]).abs()).sum();
            assert!(
                (got - want).abs() <= 1e-5 * mag,
                "row {r}: kernel {got} vs dequant {want}"
            );
        }
    }

    /// The int8 (a8w8) path can't be checked against an f32 reference — the
    /// activation quantization dominates. Check it against the q4t kernel it
    /// was ported from instead, on payloads holding the same weights: that
    /// isolates exactly what the port could break (16 B stride, ladder
    /// lookup, nibble unpack) from what it deliberately shares.
    #[test]
    fn q4tp_matvec_matches_the_q4t_kernel_it_was_ported_from() {
        let (rows, cols) = (256usize, 512usize);
        let bytes = synth_q4tp(rows, cols);
        let twin = q4tp_as_q4t(&bytes, rows, cols);
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 13 + 7) % 97) as f32 / 97.0 - 0.5)
            .collect();

        let mut got = vec![0f32; rows];
        q4tp_matvec(&bytes, &x, rows, cols, &mut got, None);
        let mut want = vec![0f32; rows];
        q4t_matvec(&twin, &x, rows, cols, &mut want, None);

        // Scale is f16 in the twin and f32 here, so allow that rounding on
        // top of the summed magnitude (same cancellation argument as above).
        let mut w = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_q4tp(&bytes, rows, cols, &mut w);
        for r in 0..rows {
            let mag: f32 = (0..cols).map(|c| (w[r * cols + c] * x[c]).abs()).sum();
            assert!(
                (got[r] - want[r]).abs() <= 1e-3 * mag,
                "row {r}: q4tp {} vs q4t {}",
                got[r],
                want[r]
            );
        }
    }

    /// `matmat` carries three arms (Accelerate, blocked int8 1x4, scalar).
    /// Batch 5 crosses the blocked kernel's stride, so this exercises the
    /// 1x4 path AND its scalar tail in one run — the blocked kernel is new
    /// code and its four accumulators are exactly what tends to go wrong.
    #[test]
    fn q4tp_matmat_matches_the_q4t_kernel_it_was_ported_from() {
        let (rows, cols, b) = (256usize, 512usize, 5usize);
        let bytes = synth_q4tp(rows, cols);
        let twin = q4tp_as_q4t(&bytes, rows, cols);
        let xs: Vec<f32> = (0..b * cols)
            .map(|i| ((i * 29 + 11) % 89) as f32 / 89.0 - 0.5)
            .collect();

        let mut got = vec![0f32; b * rows];
        q4tp_matmat(&bytes, &xs, b, rows, cols, &mut got, None);
        let mut want = vec![0f32; b * rows];
        q4t_matmat(&twin, &xs, b, rows, cols, &mut want, None);

        let mut w = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_q4tp(&bytes, rows, cols, &mut w);
        for t in 0..b {
            for r in 0..rows {
                let mag: f32 = (0..cols)
                    .map(|c| (w[r * cols + c] * xs[t * cols + c]).abs())
                    .sum();
                let (g, wa) = (got[t * rows + r], want[t * rows + r]);
                assert!(
                    (g - wa).abs() <= 1e-3 * mag,
                    "batch {t} row {r}: q4tp {g} vs q4t {wa}"
                );
            }
        }
    }

    #[test]
    fn q4tp_matvec2_matches_the_single_stream_kernel() {
        let (rows, cols) = (128usize, 256usize);
        let gpr = cols / GROUP_SIZE;
        let bytes = synth_q4tp(rows, cols);
        let xs: Vec<f32> = (0..2 * cols)
            .map(|i| ((i * 29 + 11) % 89) as f32 / 89.0 - 0.5)
            .collect();

        let (mut o1, mut o2) = (vec![0f32; rows], vec![0f32; rows]);
        q4tp_matvec2(
            &bytes,
            &xs[..cols],
            &xs[cols..],
            rows,
            cols,
            &mut o1,
            &mut o2,
            None,
        );

        // matvec2 takes the exact path for both streams, so the single-row
        // kernel is an exact reference — no tolerance for path differences.
        let v = Q4tpView::new(&bytes, rows, cols);
        let mut sc = vec![0f32; gpr];
        for r in 0..rows {
            v.scales_into(r, gpr, &mut sc);
            assert_eq!(o1[r], q4tp_row_exact(v.nib, r, gpr, &xs[..cols], &sc));
            assert_eq!(o2[r], q4tp_row_exact(v.nib, r, gpr, &xs[cols..], &sc));
        }
    }

    /// q4tp must not COST speed — it exists to save bytes, and a format that
    /// trades 7% of a file for a slower model is a bad trade. This guard is
    /// here because correctness tests happily passed while `q4tp_matmat` was
    /// missing its int8 and Accelerate arms and the model ran 5x slower.
    /// Measured on M-series: 0.97-1.04x, i.e. parity (16 B tiles are better
    /// aligned than q4t's 18 B, which pays for the scale indirection).
    #[test]
    fn q4tp_matvec_keeps_pace_with_q4t() {
        let (rows, cols) = (4096usize, 3072usize);
        let bytes = synth_q4tp(rows, cols);
        let twin = q4tp_as_q4t(&bytes, rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| (i % 97) as f32 / 97.0 - 0.5).collect();
        let mut o = vec![0f32; rows];
        let n = 12;
        let mut best = (f64::MAX, f64::MAX);
        // Interleaved A/B, minimum statistic: this machine throttles, and a
        // mean over a thermal ramp reliably indicts whichever ran second.
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                q4t_matvec(&twin, &x, rows, cols, &mut o, None);
            }
            best.0 = best.0.min(t0.elapsed().as_secs_f64());
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                q4tp_matvec(&bytes, &x, rows, cols, &mut o, None);
            }
            best.1 = best.1.min(t0.elapsed().as_secs_f64());
        }
        let ratio = best.1 / best.0;
        println!(
            "q4t {:.3} ms | q4tp {:.3} ms | {ratio:.2}x",
            best.0 * 1e3 / n as f64,
            best.1 * 1e3 / n as f64
        );
        assert!(ratio < 2.0, "q4tp matvec {ratio:.2}x slower than q4t");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn q4t_matmat_accel_matches_dequant_reference() {
        if !accel_gemm_enabled() {
            return; // CMF_ACCEL=0
        }
        let (rows, cols, b) = (512usize, 1024usize, 8usize); // ≥500K → accel arm
        let gpr = cols / GROUP_SIZE;
        let mut bytes = vec![0u8; rows * gpr * Q4_TILE];
        for r in 0..rows {
            for g in 0..gpr {
                let t = (r * gpr + g) * Q4_TILE;
                let sc = 0.02 + 0.0005 * ((r * gpr + g) % 64) as f32;
                bytes[t..t + 2].copy_from_slice(&cortiq_core::quant::f32_to_f16(sc).to_le_bytes());
                for k in 0..16 {
                    bytes[t + 2 + k] = ((r * 31 + g * 7 + k * 13) % 251) as u8;
                }
            }
        }
        let x: Vec<f32> = (0..b * cols)
            .map(|i| ((i * 13 + 7) % 97) as f32 / 97.0 - 0.5)
            .collect();
        let mut got = vec![0f32; b * rows];
        q4t_matmat(&bytes, &x, b, rows, cols, &mut got, None);
        // Brute-force reference off the same tiles.
        let mut w = vec![0f32; rows * cols];
        for r in 0..rows {
            for g in 0..gpr {
                let t = (r * gpr + g) * Q4_TILE;
                let s = f16_to_f32(u16::from_le_bytes([bytes[t], bytes[t + 1]]));
                for (k, &bb) in bytes[t + 2..t + Q4_TILE].iter().enumerate() {
                    w[r * cols + g * GROUP_SIZE + k * 2] = ((bb & 0x0F) as f32 - 8.0) * s;
                    w[r * cols + g * GROUP_SIZE + k * 2 + 1] =
                        (((bb >> 4) & 0x0F) as f32 - 8.0) * s;
                }
            }
        }
        for bi in 0..b {
            for r in 0..rows {
                let want: f32 = (0..cols).map(|j| x[bi * cols + j] * w[r * cols + j]).sum();
                let d = (got[bi * rows + r] - want).abs();
                assert!(
                    d <= want.abs().max(1.0) * 1e-4,
                    "accel q4t GEMM diverged at ({bi},{r}): {} vs {want}",
                    got[bi * rows + r]
                );
            }
        }
    }

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
        let tol = if a8w8_enabled() { 6e-2 } else { 1e-4 };
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
        vbitmatvec2(
            &bytes, &offsets, &x1, &x2, rows, cols, &mut b1, &mut b2, None,
        );
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

    /// Multi-matrix job must equal separate matvecs exactly — same
    /// kernels, only the dispatch is fused.
    #[test]
    fn matvec_many_equals_separate_matvecs() {
        use crate::pool::Pool;
        let (r1, r2, cols) = (300, 200, 64);
        let mk = |salt: usize, rows: usize| {
            QTensor::from_f32(
                (0..rows * cols)
                    .map(|i| ((i * 7 + salt) % 97) as f32 / 97.0 - 0.5)
                    .collect(),
                rows,
                cols,
            )
        };
        let (a, b) = (mk(1, r1), mk(5, r2));
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.11).sin()).collect();
        let pool = Pool::new(3);

        let (mut ea, mut eb) = (vec![0f32; r1], vec![0f32; r2]);
        a.matvec(&x, &mut ea, Some(&pool));
        b.matvec(&x, &mut eb, Some(&pool));
        let (mut ga, mut gb) = (vec![0f32; r1], vec![0f32; r2]);
        QTensor::matvec_many([&a, &b], &x, [&mut ga, &mut gb], Some(&pool));
        assert_eq!(ea, ga, "fused multi-matrix lane 1 must be bit-identical");
        assert_eq!(eb, gb, "fused multi-matrix lane 2 must be bit-identical");
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
            q4matvec(
                &q4,
                &xs[bi * cols..(bi + 1) * cols],
                rows,
                cols,
                &mut expect,
                None,
            );
            assert_eq!(
                &got[bi * rows..(bi + 1) * rows],
                &expect[..],
                "q4 batch pos {bi}"
            );
        }

        // vbit: batch vs singles.
        let mut got = vec![0f32; b * rows];
        vbitmatmat(&vb, &offsets, &xs, b, rows, cols, &mut got, None);
        for bi in 0..b {
            let mut expect = vec![0f32; rows];
            vbitmatvec(
                &vb,
                &offsets,
                &xs[bi * cols..(bi + 1) * cols],
                rows,
                cols,
                &mut expect,
                None,
            );
            assert_eq!(
                &got[bi * rows..(bi + 1) * rows],
                &expect[..],
                "vbit batch pos {bi}"
            );
        }
    }

    /// q4_tiled kernels must produce BIT-identical outputs to the q4
    /// split kernels on the same values (same ints, same order — only
    /// the byte placement differs).
    #[test]
    fn q4_tiled_matches_q4_block_bitexact() {
        let (rows, cols, b) = (8usize, 128usize, 3usize);
        let groups = rows * cols / GROUP_SIZE;
        let mut split = Vec::with_capacity(groups * 18);
        for i in 0..groups * 16 {
            split.push((((i * 7 + 3) % 256) & 0xFF) as u8);
        }
        for g in 0..groups {
            split.extend_from_slice(
                &cortiq_core::quant::f32_to_f16(0.01 + 0.003 * g as f32).to_le_bytes(),
            );
        }
        // Re-tile: [scale][nibbles] per group.
        let (packed, scales) = split.split_at(groups * 16);
        let mut tiled = Vec::with_capacity(groups * Q4_TILE);
        for g in 0..groups {
            tiled.extend_from_slice(&scales[g * 2..g * 2 + 2]);
            tiled.extend_from_slice(&packed[g * 16..(g + 1) * 16]);
        }

        let mut x1: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.17).sin()).collect();
        x1[9] = 250.0; // exercise the outlier path
        let x2: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.23).cos()).collect();

        let (mut a, mut t) = (vec![0f32; rows], vec![0f32; rows]);
        q4matvec(&split, &x1, rows, cols, &mut a, None);
        q4t_matvec(&tiled, &x1, rows, cols, &mut t, None);
        assert_eq!(a, t, "q4t matvec must match q4 bit-for-bit");

        let (mut a1, mut a2) = (vec![0f32; rows], vec![0f32; rows]);
        let (mut t1, mut t2) = (vec![0f32; rows], vec![0f32; rows]);
        q4matvec2(&split, &x1, &x2, rows, cols, &mut a1, &mut a2, None);
        q4t_matvec2(&tiled, &x1, &x2, rows, cols, &mut t1, &mut t2, None);
        assert_eq!(a1, t1);
        assert_eq!(a2, t2);

        let xs: Vec<f32> = (0..b * cols).map(|i| (i as f32 * 0.13).sin()).collect();
        let (mut am, mut tm) = (vec![0f32; b * rows], vec![0f32; b * rows]);
        q4matmat(&split, &xs, b, rows, cols, &mut am, None);
        q4t_matmat(&tiled, &xs, b, rows, cols, &mut tm, None);
        assert_eq!(am, tm, "q4t matmat must match q4 bit-for-bit");
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

    /// The fused q1t matvec must equal the reference (dequant_q1t → dot),
    /// including the ternary zero level and the binary-searched outlier
    /// overlay. Guards the mmap kernel that makes a 12B q1t runnable.
    #[test]
    fn q1t_matvec_matches_reference() {
        use cortiq_core::quant::{dequant_q1t, f32_to_f16};
        let (rows, cols) = (3usize, 64usize); // gpr = 2
        let gpr = cols / GROUP_SIZE;
        let scales = [0.5f32, 0.3, 0.7, 0.2, 0.6, 0.15];
        // Overlay (must be sorted by flat index): a few spikes across rows.
        let outliers: [(u32, f32); 3] = [(5, 9.0), (70, -4.5), (150, 3.25)];
        let is_out = |flat: usize| outliers.iter().any(|&(i, _)| i as usize == flat);
        let mut bytes = Vec::new();
        for r in 0..rows {
            for g in 0..gpr {
                bytes.extend_from_slice(&f32_to_f16(scales[r * gpr + g]).to_le_bytes());
                let mut c = [0u8; 7];
                for k in 0..GROUP_SIZE {
                    // Encoder invariant: code 0 at outlier positions.
                    let code = if is_out(r * cols + g * GROUP_SIZE + k) {
                        0
                    } else {
                        ((k + r * 3 + g) % 3) as u8 // 0,1,2
                    };
                    cortiq_core::quant::q1t_pack(&mut c, k, code);
                }
                bytes.extend_from_slice(&c);
            }
        }
        // Per-row overlay: [u32 row_ptr[rows+1]] then [(u16 col, f16 val)] by
        // row (outliers are sorted by flat index → already grouped by row).
        let mut row_ptr = vec![0u32; rows + 1];
        for &(idx, _) in &outliers {
            row_ptr[idx as usize / cols + 1] += 1;
        }
        for r in 0..rows {
            row_ptr[r + 1] += row_ptr[r];
        }
        for &p in &row_ptr {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
        for &(idx, v) in &outliers {
            bytes.extend_from_slice(&((idx as usize % cols) as u16).to_le_bytes());
            bytes.extend_from_slice(&f32_to_f16(v).to_le_bytes());
        }

        let mut refw = vec![0f32; rows * cols];
        dequant_q1t(&bytes, rows, cols, &mut refw);
        // On-grid activations (±1, amax 1) so the int8 SDOT path reconstructs
        // x exactly and matches the f32 reference (same trick as the q1 test).
        let x: Vec<f32> = (0..cols)
            .map(|j| if j % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut expect = vec![0f32; rows];
        for r in 0..rows {
            let mut a = 0.0f32;
            for j in 0..cols {
                a += refw[r * cols + j] * x[j];
            }
            expect[r] = a;
        }
        let tol = |e: f32| 1e-3 * e.abs().max(1e-3);
        let mut got = vec![0f32; rows];
        q1t_matvec(&bytes, &x, rows, cols, &mut got, None);
        for r in 0..rows {
            assert!(
                (got[r] - expect[r]).abs() < tol(expect[r]),
                "row {r}: {} vs {}",
                got[r],
                expect[r]
            );
        }
        // matmat (b=2, f32 decode path) must agree too.
        let x2: Vec<f32> = x.iter().chain(x.iter()).copied().collect();
        let mut gm = vec![0f32; 2 * rows];
        q1t_matmat(&bytes, &x2, 2, rows, cols, &mut gm, None);
        for r in 0..rows {
            assert!((gm[r] - expect[r]).abs() < tol(expect[r]));
            assert!((gm[rows + r] - expect[r]).abs() < tol(expect[r]));
        }
        // Fused pair (q1t_matvec2) must equal two single matvecs
        // bit-for-bit: same unpack, same group order, same f32
        // accumulation per stream. Distinct x2 exercises both lanes.
        let xb: Vec<f32> = (0..cols)
            .map(|j| if j % 5 == 0 { -1.0 } else { 1.0 })
            .collect();
        let (mut s1, mut s2) = (vec![0f32; rows], vec![0f32; rows]);
        q1t_matvec(&bytes, &x, rows, cols, &mut s1, None);
        q1t_matvec(&bytes, &xb, rows, cols, &mut s2, None);
        let (mut p1, mut p2) = (vec![0f32; rows], vec![0f32; rows]);
        q1t_matvec2(&bytes, &x, &xb, rows, cols, &mut p1, &mut p2, None);
        assert_eq!(p1, s1, "q1t pair lane 1 ≠ single matvec");
        assert_eq!(p2, s2, "q1t pair lane 2 ≠ single matvec");
    }

    /// Pair == 2×matvec with an ODD group count (the kernel's tail
    /// group) and no overlay section.
    #[test]
    fn q1t_matvec2_odd_gpr_matches_singles() {
        use cortiq_core::quant::{Q1T_TILE, f32_to_f16, q1t_pack};
        let (rows, cols) = (5usize, 96usize); // gpr = 3 → paired + tail
        let gpr = cols / GROUP_SIZE;
        let mut bytes = Vec::with_capacity(rows * gpr * Q1T_TILE);
        for r in 0..rows {
            for g in 0..gpr {
                bytes.extend_from_slice(&f32_to_f16(0.1 + 0.05 * (r + g) as f32).to_le_bytes());
                let mut c = [0u8; 7];
                for k in 0..GROUP_SIZE {
                    q1t_pack(&mut c, k, ((k * 7 + r * 5 + g * 3) % 3) as u8);
                }
                bytes.extend_from_slice(&c);
            }
        }
        let x1: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.31).sin()).collect();
        let x2: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.17).cos()).collect();
        let (mut s1, mut s2) = (vec![0f32; rows], vec![0f32; rows]);
        q1t_matvec(&bytes, &x1, rows, cols, &mut s1, None);
        q1t_matvec(&bytes, &x2, rows, cols, &mut s2, None);
        let (mut p1, mut p2) = (vec![0f32; rows], vec![0f32; rows]);
        q1t_matvec2(&bytes, &x1, &x2, rows, cols, &mut p1, &mut p2, None);
        assert_eq!(p1, s1, "odd-gpr pair lane 1 ≠ single");
        assert_eq!(p2, s2, "odd-gpr pair lane 2 ≠ single");
    }

    // Speed A/B: fused pair (one unpack, two streams) vs two single
    // matvecs. Single-threaded, FFN-sized, min-of paired in-process.
    //   cargo test -p cortiq-engine --release q1t_matvec2_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn q1t_matvec2_speed() {
        use cortiq_core::quant::{Q1T_TILE, f32_to_f16, q1t_pack};
        use std::time::Instant;
        let (rows, cols) = (8192usize, 4096usize);
        let gpr = cols / GROUP_SIZE;
        let mut bytes = Vec::with_capacity(rows * gpr * Q1T_TILE);
        for r in 0..rows {
            for g in 0..gpr {
                let s = 0.1 + ((r + g) % 7) as f32 * 0.01;
                bytes.extend_from_slice(&f32_to_f16(s).to_le_bytes());
                let mut c = [0u8; 7];
                for k in 0..GROUP_SIZE {
                    q1t_pack(&mut c, k, ((k * 7 + r + g) % 3) as u8);
                }
                bytes.extend_from_slice(&c);
            }
        }
        let x1: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.31).sin()).collect();
        let x2: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.17).cos()).collect();
        let (mut s1, mut s2) = (vec![0f32; rows], vec![0f32; rows]);
        let (mut p1, mut p2) = (vec![0f32; rows], vec![0f32; rows]);
        // Warm both paths once.
        q1t_matvec(&bytes, &x1, rows, cols, &mut s1, None);
        q1t_matvec2(&bytes, &x1, &x2, rows, cols, &mut p1, &mut p2, None);
        let (mut t_pair, mut t_two) = (f64::MAX, f64::MAX);
        for _ in 0..8 {
            let t0 = Instant::now();
            q1t_matvec2(&bytes, &x1, &x2, rows, cols, &mut p1, &mut p2, None);
            t_pair = t_pair.min(t0.elapsed().as_secs_f64() * 1000.0);
            let t1 = Instant::now();
            q1t_matvec(&bytes, &x1, rows, cols, &mut s1, None);
            q1t_matvec(&bytes, &x2, rows, cols, &mut s2, None);
            t_two = t_two.min(t1.elapsed().as_secs_f64() * 1000.0);
        }
        assert_eq!(p1, s1);
        assert_eq!(p2, s2);
        println!("q1t pair {rows}x{cols}: fused {t_pair:.2} ms | two singles {t_two:.2} ms");
    }

    // Speed A/B: the base-3-division decode (what the packing commit left in
    // place) vs the fused sign-LUT matvec. Both single-threaded, same bytes.
    //   cargo test -p cortiq-engine q1t_matvec_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn q1t_matvec_speed() {
        use cortiq_core::quant::{Q1T_TILE, f32_to_f16, q1t_code, q1t_pack};
        use std::time::Instant;
        let (rows, cols) = (8192usize, 4096usize); // FFN-sized
        let gpr = cols / GROUP_SIZE;
        let mut bytes = Vec::with_capacity(rows * gpr * Q1T_TILE + 16);
        for r in 0..rows {
            for g in 0..gpr {
                let s = 0.1 + ((r + g) % 7) as f32 * 0.01;
                bytes.extend_from_slice(&f32_to_f16(s).to_le_bytes());
                let mut c = [0u8; 7];
                for k in 0..GROUP_SIZE {
                    q1t_pack(&mut c, k, ((k * 7 + r + g) % 3) as u8);
                }
                bytes.extend_from_slice(&c);
            }
        }
        let (n, stride) = (rows * cols, 40usize); // ~2.5% outliers, per-row overlay
        let mut row_ptr = vec![0u32; rows + 1];
        let mut idx = 0usize;
        while idx < n {
            row_ptr[idx / cols + 1] += 1;
            idx += stride;
        }
        for r in 0..rows {
            row_ptr[r + 1] += row_ptr[r];
        }
        for &p in &row_ptr {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
        let mut idx = 0usize;
        while idx < n {
            bytes.extend_from_slice(&((idx % cols) as u16).to_le_bytes());
            bytes.extend_from_slice(&f32_to_f16((idx % 13) as f32 * 0.1 - 0.6).to_le_bytes());
            idx += stride;
        }
        // On-grid ±1 so the fast path's int8 SDOT is exact vs the f32 "slow"
        // reference (the A/B is a timing check; values must still agree).
        let x: Vec<f32> = (0..cols)
            .map(|j| if j % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        let (rp_off, ent_off, has_ov) = q1t_overlay(&bytes, rows * gpr * Q1T_TILE, rows);

        // "before": base-3 division decode into a buffer, then dot.
        let slow = |out: &mut [f32]| {
            let mut buf = vec![0f32; cols];
            for r in 0..rows {
                for g in 0..gpr {
                    let off = (r * gpr + g) * Q1T_TILE;
                    let s = f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
                    let codes = &bytes[off + 2..off + Q1T_TILE];
                    for k in 0..GROUP_SIZE {
                        buf[g * GROUP_SIZE + k] = match q1t_code(codes, k) {
                            1 => s,
                            2 => -s,
                            _ => 0.0,
                        };
                    }
                }
                out[r] = q1t_row_outlier_correction(&bytes, r, rp_off, ent_off, has_ov, &x)
                    + (0..cols).map(|j| buf[j] * x[j]).sum::<f32>();
            }
        };
        let iters = 5;
        let mut a = vec![0f32; rows];
        slow(&mut a); // warm
        let t = Instant::now();
        for _ in 0..iters {
            slow(&mut a);
        }
        let slow_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let mut b = vec![0f32; rows];
        q1t_matvec(&bytes, &x, rows, cols, &mut b, None); // warm
        let t = Instant::now();
        for _ in 0..iters {
            q1t_matvec(&bytes, &x, rows, cols, &mut b, None);
        }
        let fast_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        for r in 0..rows {
            assert!((a[r] - b[r]).abs() < 1e-2, "mismatch row {r}");
        }
        println!(
            "q1t matvec {rows}x{cols} (1 thread): div-decode {slow_ms:.2} ms  fused-LUT {fast_ms:.2} ms  => {:.2}x",
            slow_ms / fast_ms
        );
    }
}

#[cfg(test)]
mod gemm_bench {
    /// `cargo test -p cortiq-engine --release q4tp_matmat_throughput -- --ignored --nocapture`
    /// Times the batched q4tp GEMM at the shapes the image DiT runs
    /// (b=296 tokens, 2304 -> 9216), on synthetic bytes: no model, no
    /// mmap, no thermal drift over minutes — a kernel change shows up
    /// here in seconds where a full render hides it in noise.
    ///
    /// On macOS add `CMF_ACCEL=0`: this shape is over the 500k-cell mark
    /// where the matmat hands off to Accelerate's dequant sgemm, and
    /// without the opt-out both rows below measure the AMX, not the
    /// kernel under test.
    #[test]
    #[ignore]
    fn q4tp_matmat_throughput() {
        // 296 is a prompt-encode batch; the image DiT runs 2085 at
        // 512x512, where the activation panel stops fitting L2 and the
        // loop's shape starts to matter more than its instructions.
        let b: usize = std::env::var("CMF_BENCH_B")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(296);
        let (rows, cols) = (9216usize, 2304usize);
        let (_, _, _) = (rows, cols, b);
        let total =
            cortiq_core::quant::expected_nbytes(cortiq_core::TensorDtype::Q4TiledP, &[rows, cols])
                .unwrap();
        // Random nibbles are fine, but the row params are f16 (lo, step)
        // of a geometric ladder: garbage there gives exp2 of a huge
        // exponent, the scales come back inf, and the whole bench times
        // NaN arithmetic instead of the kernel.
        let (params_off, codes_off, _) = cortiq_core::quant::q4tp_sections(rows, cols);
        let mut bytes: Vec<u8> = (0..total).map(|i| (i * 37 % 251) as u8).collect();
        let lo = cortiq_core::quant::f32_to_f16(-4.0);
        let step = cortiq_core::quant::f32_to_f16(0.1);
        for r in 0..rows {
            let o = params_off + r * 4;
            bytes[o..o + 2].copy_from_slice(&lo.to_le_bytes());
            bytes[o + 2..o + 4].copy_from_slice(&step.to_le_bytes());
        }
        let _ = codes_off;
        let xs: Vec<f32> = (0..b * cols)
            .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
            .collect();
        let mut out = vec![0f32; b * rows];
        let pool = crate::pool::Pool::from_env();
        // A shared 48-core stand drifts ±25% run to run, which is wider
        // than any kernel change worth making. So: alternate the two
        // kernels inside one process and keep the BEST time for
        // each. Interleaving makes both see the same interference, and a
        // minimum is the one statistic another tenant cannot inflate.
        super::q4tp_matmat(&bytes, &xs, b, rows, cols, &mut out, pool.as_deref());
        let reps: usize = std::env::var("CMF_BENCH_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let mut best = [f64::MAX; 2];
        let mut sums = [0f32; 2];
        for _ in 0..reps {
            for (k, w) in [(0usize, 1u8), (1usize, 2u8)] {
                super::Q4TP_ALT.store(w, std::sync::atomic::Ordering::Relaxed);
                let t = std::time::Instant::now();
                super::q4tp_matmat(&bytes, &xs, b, rows, cols, &mut out, pool.as_deref());
                best[k] = best[k].min(t.elapsed().as_secs_f64());
                sums[k] = out.iter().take(64).sum::<f32>();
            }
        }
        let flops = 2.0 * b as f64 * rows as f64 * cols as f64;
        for (k, name) in ["previous", "tuned   "].iter().enumerate() {
            println!(
                "q4tp matmat {rows}x{cols} b={b} {name}: {:.1} ms  {:.1} GFLOP/s  (checksum {:.3})",
                best[k] * 1e3,
                flops / best[k] / 1e9,
                sums[k]
            );
        }
        assert!(
            (sums[0] - sums[1]).abs() < 1e-2,
            "the tuned kernel changed the result: {} vs {}",
            sums[0],
            sums[1]
        );
    }

    /// The blocked kernel must agree with the per-column path exactly —
    /// same weights, same activation split, only a different instruction
    /// mix. Shapes are chosen to hit the awkward cases: a column count
    /// that leaves an odd group (the 512-bit kernel does two at a time),
    /// and a batch that does not divide by four.
    #[test]
    fn q4tp_matmat_blocked_matches_scalar() {
        use std::sync::atomic::Ordering::Relaxed;
        // The last shape carries the image DiT's column count — 2304, so
        // 72 groups of accumulation, which is where a reordered sum can
        // actually drift — and runs through the thread pool, since the
        // blocked path splits rows across workers. Its row count stays
        // under 500k cells on purpose: above that, macOS diverts the whole
        // matmat to the Accelerate/AMX dequant sgemm and neither kernel
        // here would run.
        for &(rows, cols, b) in &[
            (64usize, 128usize, 7usize),
            (33, 96, 4),
            (16, 256, 9),
            (192, 2304, 37),
        ] {
            let total = cortiq_core::quant::expected_nbytes(
                cortiq_core::TensorDtype::Q4TiledP,
                &[rows, cols],
            )
            .unwrap();
            let (params_off, _, _) = cortiq_core::quant::q4tp_sections(rows, cols);
            let mut bytes: Vec<u8> = (0..total).map(|i| (i * 61 % 251) as u8).collect();
            let lo = cortiq_core::quant::f32_to_f16(-4.0);
            let step = cortiq_core::quant::f32_to_f16(0.1);
            for r in 0..rows {
                let o = params_off + r * 4;
                bytes[o..o + 2].copy_from_slice(&lo.to_le_bytes());
                bytes[o + 2..o + 4].copy_from_slice(&step.to_le_bytes());
            }
            let xs: Vec<f32> = (0..b * cols)
                .map(|i| ((i % 89) as f32 - 44.0) / 44.0)
                .collect();
            let mut got = vec![0f32; b * rows];
            let mut want = vec![0f32; b * rows];
            let gpr = cols / 32;
            let view = super::Q4tpView::new(&bytes, rows, cols);
            let pool = crate::pool::Pool::from_env();
            super::Q4TP_ALT.store(2, Relaxed);
            super::q4tp_matmat(&bytes, &xs, b, rows, cols, &mut got, pool.as_deref());
            super::Q4TP_ALT.store(1, Relaxed);
            super::q4tp_matmat(&bytes, &xs, b, rows, cols, &mut want, pool.as_deref());
            super::Q4TP_ALT.store(0, Relaxed);
            // Measured against the output's scale, not cell by cell: a
            // dot product of 2304 terms lands near zero wherever the row
            // and the activation nearly cancel, and there a per-cell
            // ratio reports 1e-3 for an absolute error of 5e-6 — f32's
            // own rounding, reordered. What must stay small is the error
            // relative to what the layer actually outputs.
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            let (mut worst, mut at) = (0f32, 0usize);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                if (g - w).abs() > worst {
                    worst = (g - w).abs();
                    at = i;
                }
            }
            assert!(
                worst <= 1e-4 * scale,
                "{rows}x{cols} b={b}: blocked and scalar disagree by {worst:.3e} \
                 (scale {scale:.3e}) at cell {at}: {} vs {}",
                got[at],
                want[at]
            );

            // "Same speed, no quality loss" is a claim about which answer
            // is RIGHT, not about which two agree. Both paths sum the same
            // 2304 products in different orders, so f64 decides: the
            // blocked kernel keeps sixteen partial sums and folds them at
            // the end, which is a shallower addition tree than the
            // per-column path's running scalar, and it must not be worse.
            let (mut e_blocked, mut e_scalar) = (0f64, 0f64);
            for bi in 0..b {
                let act = super::split_act(&xs[bi * cols..(bi + 1) * cols]);
                for r in 0..rows {
                    let mut sc = vec![0f32; gpr];
                    view.scales_into(r, gpr, &mut sc);
                    let mut exact = 0f64;
                    for j in 0..cols {
                        let (w, sq) = super::q4tp_outlier(view.nib, r, gpr, j, &sc);
                        exact += w as f64 * sq as f64 * act.xq[j] as f64;
                    }
                    exact *= act.sx as f64;
                    for &(j, xv) in &act.outliers {
                        let (w, sq) = super::q4tp_outlier(view.nib, r, gpr, j, &sc);
                        exact += w as f64 * sq as f64 * xv as f64;
                    }
                    let i = bi * rows + r;
                    e_blocked = e_blocked.max((got[i] as f64 - exact).abs());
                    e_scalar = e_scalar.max((want[i] as f64 - exact).abs());
                }
            }
            println!(
                "{rows}x{cols} b={b}: worst error vs f64 — blocked {e_blocked:.3e}, \
                 per-column {e_scalar:.3e}"
            );
            // An absolute bar, not a race between the two: at these
            // magnitudes both sit in f32's last bits, and on a small shape
            // whichever one happens to round the unluckiest cell "wins" by
            // a factor the next seed reverses.
            assert!(
                e_blocked <= 1e-5 * scale as f64 && e_scalar <= 1e-5 * scale as f64,
                "{rows}x{cols} b={b}: error against f64 too large — blocked \
                 {e_blocked:.3e}, per-column {e_scalar:.3e}, scale {scale:.3e}"
            );
        }
    }

    /// The q4t twin of the throughput bench, same shape and rules, so the
    /// two quantisations' batch kernels can be read against each other.
    /// `cargo test -p cortiq-engine --release q4t_matmat_throughput -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn q4t_matmat_throughput() {
        let (rows, cols, b) = (9216usize, 2304usize, 296usize);
        let total =
            cortiq_core::quant::expected_nbytes(cortiq_core::TensorDtype::Q4Tiled, &[rows, cols])
                .unwrap();
        // q4t carries a per-group f16 scale in the tile's first two bytes;
        // random bytes there decode to inf and the bench would time NaNs.
        let mut bytes: Vec<u8> = (0..total).map(|i| (i * 37 % 251) as u8).collect();
        let sc = cortiq_core::quant::f32_to_f16(0.02);
        for t in bytes.chunks_mut(super::Q4_TILE) {
            t[..2].copy_from_slice(&sc.to_le_bytes());
        }
        let xs: Vec<f32> = (0..b * cols)
            .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
            .collect();
        let mut out = vec![0f32; b * rows];
        let pool = crate::pool::Pool::from_env();
        super::q4t_matmat(&bytes, &xs, b, rows, cols, &mut out, pool.as_deref());
        let reps: usize = std::env::var("CMF_BENCH_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            super::q4t_matmat(&bytes, &xs, b, rows, cols, &mut out, pool.as_deref());
            best = best.min(t.elapsed().as_secs_f64());
        }
        let flops = 2.0 * b as f64 * rows as f64 * cols as f64;
        println!(
            "q4t matmat {rows}x{cols} b={b}: {:.1} ms  {:.1} GFLOP/s  (checksum {:.3})",
            best * 1e3,
            flops / best / 1e9,
            out.iter().take(64).sum::<f32>()
        );
    }
}
