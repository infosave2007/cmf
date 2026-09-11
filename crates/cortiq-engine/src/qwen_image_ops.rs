//! Bounded weight operations used by the native Qwen Image transformer.
//!
//! Qwen Image keeps its large projection matrices in the CMF mapping.  A
//! generic `QTensor::from_model` call is deliberately not used for F16/BF16
//! matrices: that path owns a full F32 copy for dtypes without a fused
//! quantized kernel.  `Linear::Mapped` below decodes only one bounded row at
//! a time, while quantized tensors borrow the existing `Proj` implementation.

use crate::dit::Proj;
use crate::pool::Pool;
use cortiq_core::{CmfModel, TensorDtype};
use std::sync::Arc;

/// Maximum number of dense rows materialized by one call to a mapped linear.
/// The implementation currently reuses one row buffer; keeping the constant
/// here documents the intentional bound and gives future device tiling a
/// stable knob without changing the public API.
pub const DENSE_ROW_TILE: usize = 32;

/// A row-major `y = x · Wᵀ` projection.
///
/// Quantized projections stay in the existing mmap-backed `Proj`/`QTensor`
/// path, including Q8_2f and Q4TP.  Native F32/F16/BF16 projections retain an
/// Arc to the CMF mapping and are decoded into a single bounded scratch row.
pub struct Linear {
    repr: LinearRepr,
}

enum LinearRepr {
    Mapped {
        model: Arc<CmfModel>,
        idx: usize,
        dtype: TensorDtype,
        rows: usize,
        cols: usize,
    },
    Quant(Proj),
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("rows", &self.rows())
            .field("cols", &self.cols())
            .field("dtype", &self.dtype())
            .finish()
    }
}

impl Linear {
    /// Load a rank-2 projection without materializing a whole F16/BF16
    /// matrix.  The CMF shape is semantic `[out_features, in_features]`.
    pub fn load(model: &Arc<CmfModel>, name: &str) -> Result<Self, String> {
        let idx = model
            .tensor_index(name)
            .ok_or_else(|| format!("missing linear tensor '{name}'"))?;
        let entry = &model.tensors[idx];
        if entry.shape.len() != 2 {
            return Err(format!(
                "linear tensor '{name}' must be rank 2, got shape {:?}",
                entry.shape
            ));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        match entry.dtype {
            TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16 => Ok(Self {
                repr: LinearRepr::Mapped {
                    model: model.clone(),
                    idx,
                    dtype: entry.dtype,
                    rows,
                    cols,
                },
            }),
            TensorDtype::Q8Row
            | TensorDtype::Q8_2f
            | TensorDtype::Q4Block
            | TensorDtype::Q4Tiled
            | TensorDtype::Q4TiledP
            | TensorDtype::Q2TiledP
            | TensorDtype::Vbit
            | TensorDtype::VbitRo
            | TensorDtype::Q1
            | TensorDtype::Q1S
            | TensorDtype::Q1T => Ok(Self {
                repr: LinearRepr::Quant(Proj::from_model(model, name)?),
            }),
            other => Err(format!(
                "linear tensor '{name}' has unsupported dtype {}",
                other.name()
            )),
        }
    }

    /// Construct an owned F32 projection for a focused unit test. Production
    /// model loading should use [`Self::load`] so large matrices remain
    /// mapped or quantized.
    #[cfg(test)]
    pub(crate) fn from_f32_for_test(weights: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(weights.len(), rows * cols);
        Self {
            repr: LinearRepr::Quant(Proj::f32(weights, cols)),
        }
    }

    pub fn rows(&self) -> usize {
        match &self.repr {
            LinearRepr::Mapped { rows, .. } => *rows,
            LinearRepr::Quant(p) => p.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match &self.repr {
            LinearRepr::Mapped { cols, .. } => *cols,
            LinearRepr::Quant(p) => p.cols(),
        }
    }

    pub fn dtype(&self) -> Option<TensorDtype> {
        match &self.repr {
            LinearRepr::Mapped { dtype, .. } => Some(*dtype),
            LinearRepr::Quant(p) => match p {
                Proj::F32 { .. } => Some(TensorDtype::F32),
                Proj::Q(q) => q.model_dtype(),
            },
        }
    }

    /// Return the mapped CMF identity for quantized device GEMMs.
    pub(crate) fn mapped_device_gemm(&self) -> Option<(&Arc<CmfModel>, usize)> {
        match &self.repr {
            LinearRepr::Mapped { .. } => None,
            LinearRepr::Quant(Proj::Q(q)) => q.mapped_device_gemm(),
            LinearRepr::Quant(Proj::F32 { .. }) => None,
        }
    }

    /// Compute `out[b, rows] = x[b, cols] · Wᵀ`.
    ///
    /// The output and activation buffers are caller-owned.  Mapped dense
    /// weights are decoded in bounded row tiles and sent through the existing
    /// GEMM path, so a 3,072×3,072 BF16 projection never becomes a 36 MiB
    /// temporary.  Quantized projections use the existing pooled CPU/GPU
    /// dispatch and preserve Q8_2f/Q4TP semantics.
    pub fn forward(
        &self,
        x: &[f32],
        batch: usize,
        out: &mut [f32],
        pool: Option<&Pool>,
    ) -> Result<(), String> {
        let rows = self.rows();
        let cols = self.cols();
        if x.len() != batch.saturating_mul(cols) {
            return Err(format!(
                "linear input length {} != batch {batch} × cols {cols}",
                x.len()
            ));
        }
        if out.len() != batch.saturating_mul(rows) {
            return Err(format!(
                "linear output length {} != batch {batch} × rows {rows}",
                out.len()
            ));
        }
        match &self.repr {
            LinearRepr::Quant(p) => {
                p.matmat(x, batch, out, pool);
                Ok(())
            }
            LinearRepr::Mapped {
                model,
                idx,
                dtype,
                rows,
                cols,
            } => {
                let entry = &model.tensors[*idx];
                let bytes = model.entry_bytes(entry);
                let elem_bytes = match dtype {
                    TensorDtype::F32 => 4,
                    TensorDtype::F16 | TensorDtype::Bf16 => 2,
                    _ => unreachable!("mapped linear dtype is dense"),
                };
                let expected = rows
                    .checked_mul(*cols)
                    .and_then(|n| n.checked_mul(elem_bytes))
                    .ok_or_else(|| "linear byte-size overflow".to_string())?;
                if bytes.len() != expected {
                    return Err(format!(
                        "linear tensor '{}' has {} bytes, expected {}",
                        entry.name,
                        bytes.len(),
                        expected
                    ));
                }
                let mut tile = vec![0.0f32; DENSE_ROW_TILE * *cols];
                let mut tile_out = vec![0.0f32; batch * DENSE_ROW_TILE];
                for base in (0..*rows).step_by(DENSE_ROW_TILE) {
                    let tile_rows = (*rows - base).min(DENSE_ROW_TILE);
                    for r in 0..tile_rows {
                        decode_dense_row(
                            bytes,
                            *dtype,
                            base + r,
                            *cols,
                            &mut tile[r * *cols..(r + 1) * *cols],
                        );
                    }
                    crate::fcd_ops::gemm_nt(
                        x,
                        &tile[..tile_rows * *cols],
                        &mut tile_out[..batch * tile_rows],
                        batch,
                        *cols,
                        tile_rows,
                        pool,
                    );
                    for b in 0..batch {
                        for r in 0..tile_rows {
                            out[b * *rows + base + r] = tile_out[b * tile_rows + r];
                        }
                    }
                }
                Ok(())
            }
        }
    }

    /// Compute one row, useful for small control projections and fixtures.
    pub fn forward_one(
        &self,
        x: &[f32],
        out: &mut [f32],
        pool: Option<&Pool>,
    ) -> Result<(), String> {
        self.forward(x, 1, out, pool)
    }
}

fn decode_dense_row(bytes: &[u8], dtype: TensorDtype, row: usize, cols: usize, dst: &mut [f32]) {
    debug_assert_eq!(dst.len(), cols);
    let width = match dtype {
        TensorDtype::F32 => 4,
        TensorDtype::F16 | TensorDtype::Bf16 => 2,
        _ => unreachable!("decode_dense_row only accepts dense dtypes"),
    };
    let start = row * cols * width;
    let src = &bytes[start..start + cols * width];
    for i in 0..cols {
        let off = i * width;
        dst[i] = match dtype {
            TensorDtype::F32 => f32::from_le_bytes(src[off..off + 4].try_into().unwrap()),
            TensorDtype::F16 => {
                cortiq_core::quant::f16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]))
            }
            TensorDtype::Bf16 => {
                cortiq_core::quant::bf16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]))
            }
            _ => unreachable!(),
        };
    }
}

/// Add a bias vector to `batch` row-major projections.
pub(crate) fn add_bias(rows: &mut [f32], batch: usize, bias: &[f32]) -> Result<(), String> {
    if rows.len() != batch.saturating_mul(bias.len()) {
        return Err(format!(
            "bias add length {} != batch {batch} × bias {}",
            rows.len(),
            bias.len()
        ));
    }
    for row in rows.chunks_exact_mut(bias.len()) {
        for (v, &b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Linear;

    #[test]
    fn f32_fixture_projection_is_bounded_api_compatible() {
        let p = Linear::from_f32_for_test(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let mut out = [0.0; 2];
        p.forward_one(&[2.0, -1.0], &mut out, None).unwrap();
        assert_eq!(out, [0.0, 2.0]);
    }

    #[test]
    fn bias_add_rejects_wrong_batch_shape() {
        assert!(super::add_bias(&mut [0.0; 3], 2, &[1.0, 2.0]).is_err());
    }
}
