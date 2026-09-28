//! Packed resonance scoring, bit-exact with [`crate::resonance::reference_error`]
//! (spec §3.5; port of `tools/cortiq-decision/src/packed.rs`).
//!
//! SIMD runs across [`LANES`] independent topologies, never across coordinates:
//! every lane performs exactly the reference's f32 operations in the reference's
//! order, so the result does not depend on the orthogonality of a basis, on the
//! rank of a task or on the dimension tail. Two changes against the reference
//! keep that property and only save memory passes: the block width is 8 lanes
//! (two independent dependency chains per 128-bit register pair) and the three
//! sweeps of a basis row are fused — at coordinate `i` the residual is first
//! updated with the previous row's coefficient, then added into the next
//! coefficient, which is the same value at the same position of the same
//! sequential sum. Lanes beyond a task's rank see zero basis rows: their
//! coefficient is `+0.0` and `r − (+0.0)·0.0 = r` exactly.

use crate::resonance::{TaskView, check_input};
use anyhow::{Result, ensure};

/// Topologies per SIMD block.
pub const LANES: usize = 8;
type Lane = [f32; LANES];

#[cfg(target_os = "macos")]
#[path = "packed_metal.rs"]
pub(crate) mod metal_backend;
#[cfg(feature = "vulkan")]
#[path = "packed_vulkan.rs"]
pub(crate) mod vulkan_backend;

struct Block {
    /// `[dim]` lanes of the means.
    mean: Vec<Lane>,
    /// `[rank × dim]` lanes of the basis rows (zero rows beyond a task's rank).
    basis: Vec<Lane>,
    rank: usize,
    /// Real tasks in this block (`≤ LANES`).
    count: usize,
}

/// The topologies of a skill, interleaved for scoring. Immutable after
/// construction (`Sync`): concurrent callers pass their own scratch residual.
pub struct Packed {
    #[cfg(target_os = "macos")]
    pub(crate) metal: Option<parking_lot::Mutex<metal_backend::MetalScorer>>,
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan: Option<parking_lot::Mutex<vulkan_backend::VulkanScorer>>,
    blocks: Vec<Block>,
    dim: usize,
    tasks: usize,
    ranks: Vec<usize>,
}

impl Packed {
    /// Interleave `tasks` (all of the same dimension; any ranks, including 0).
    pub fn new(tasks: &[TaskView<'_>]) -> Result<Self> {
        let dim = tasks.first().map_or(0, |t| t.dim());
        for t in tasks {
            t.check()?;
            ensure!(
                t.dim() == dim,
                "topologies of one skill must share the dimension ({} vs {dim})",
                t.dim()
            );
        }
        let mut blocks = Vec::with_capacity(tasks.len().div_ceil(LANES));
        for ts in tasks.chunks(LANES) {
            let rank = ts.iter().map(|t| t.rank()).max().unwrap_or(0);
            let mut mean = vec![[0.0f32; LANES]; dim];
            let mut basis = vec![[0.0f32; LANES]; dim * rank];
            for (lane, t) in ts.iter().enumerate() {
                for (m, &v) in mean.iter_mut().zip(t.mean) {
                    m[lane] = v;
                }
                for k in 0..t.rank() {
                    let row = t.row(k);
                    for (b, &v) in basis[k * dim..(k + 1) * dim].iter_mut().zip(row) {
                        b[lane] = v;
                    }
                }
            }
            blocks.push(Block {
                mean,
                basis,
                rank,
                count: ts.len(),
            });
        }
        Ok(Self {
            #[cfg(target_os = "macos")]
            metal: None,
            #[cfg(feature = "vulkan")]
            vulkan: None,
            blocks,
            dim,
            tasks: tasks.len(),
            ranks: tasks.iter().map(|t| t.rank()).collect(),
        })
    }

    /// Opt-in FP32 Metal reductions; CPU remains bit-exact with the reference.
    #[allow(unused_mut)]
    pub fn with_device(mut self, device: crate::bert::EncoderDevice) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            self.metal = match device {
                crate::bert::EncoderDevice::Cpu | crate::bert::EncoderDevice::Vulkan => None,
                crate::bert::EncoderDevice::Metal if self.tasks > 0 => Some(
                    parking_lot::Mutex::new(metal_backend::MetalScorer::new(&self)?),
                ),
                crate::bert::EncoderDevice::Metal => None,
            };
        }
        #[cfg(not(target_os = "macos"))]
        ensure!(
            device != crate::bert::EncoderDevice::Metal,
            "decision Metal requires macOS on Apple Silicon"
        );
        #[cfg(feature = "vulkan")]
        {
            self.vulkan = if device == crate::bert::EncoderDevice::Vulkan && self.tasks > 0 {
                Some(parking_lot::Mutex::new(vulkan_backend::VulkanScorer::new(
                    &self,
                )?))
            } else {
                None
            };
        }
        #[cfg(not(feature = "vulkan"))]
        ensure!(
            device != crate::bert::EncoderDevice::Vulkan,
            "rebuild cortiq-decision with --features vulkan"
        );
        Ok(self)
    }
    /// Actual completed GPU commands, used by acceptance benchmarks.
    pub fn metal_submissions(&self) -> u64 {
        #[cfg(target_os = "macos")]
        if let Some(m) = &self.metal {
            return m.lock().submissions();
        }
        0
    }

    pub fn vulkan_submissions(&self) -> u64 {
        #[cfg(feature = "vulkan")]
        if let Some(v) = &self.vulkan {
            return v.lock().submissions();
        }
        0
    }
    pub fn gpu_submissions(&self) -> u64 {
        self.metal_submissions() + self.vulkan_submissions()
    }

    /// Number of topologies (the length of every error vector).
    pub fn tasks(&self) -> usize {
        self.tasks
    }

    /// Signal dimension (0 when there are no topologies).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Rank of each topology, in order.
    pub fn ranks(&self) -> &[usize] {
        &self.ranks
    }

    /// A scratch residual for [`Packed::errors_with`].
    pub fn scratch(&self) -> Vec<[f32; LANES]> {
        // GPU residuals are resident on the device; do not zero a CPU buffer
        // on every standalone scoring call only to discard it immediately.
        #[cfg(target_os = "macos")]
        if self.metal.is_some() {
            return Vec::new();
        }
        #[cfg(feature = "vulkan")]
        if self.vulkan.is_some() {
            return Vec::new();
        }
        vec![[0.0; LANES]; self.dim]
    }

    /// The errors of `x` against every topology, in order.
    pub fn errors(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0; self.tasks];
        self.errors_into(x, &mut out)?;
        Ok(out)
    }

    /// [`Packed::errors`] into `out` (`len == tasks()`).
    pub fn errors_into(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let mut scratch = self.scratch();
        self.errors_with(x, out, &mut scratch)
    }

    /// [`Packed::errors_into`] with a caller-owned scratch residual.
    pub fn errors_with(
        &self,
        x: &[f32],
        out: &mut [f32],
        scratch: &mut Vec<[f32; LANES]>,
    ) -> Result<()> {
        ensure!(
            out.len() == self.tasks,
            "output has {} slots for {} topologies",
            out.len(),
            self.tasks
        );
        if self.tasks == 0 {
            return Ok(());
        }
        check_input(x, self.dim)?;
        #[cfg(target_os = "macos")]
        if let Some(metal) = &self.metal {
            return metal.lock().errors(x, out);
        }
        #[cfg(feature = "vulkan")]
        if let Some(v) = &self.vulkan {
            return v.lock().errors(x, out);
        }
        scratch.resize(self.dim, [0.0; LANES]);
        let r = &mut scratch[..self.dim];
        for (bi, b) in self.blocks.iter().enumerate() {
            let e = block_errors(x, b, r, self.dim);
            out[bi * LANES..bi * LANES + b.count].copy_from_slice(&e[..b.count]);
        }
        ensure!(out.iter().all(|v| v.is_finite()), "numeric overflow");
        Ok(())
    }
}

/// The errors of one block: per lane, `reference_error` with fused sweeps.
#[allow(clippy::needless_range_loop)] // fixed lane loops expose SIMD across tasks
fn block_errors(x: &[f32], b: &Block, r: &mut [Lane], dim: usize) -> Lane {
    let x = &x[..dim];
    let r = &mut r[..dim];
    let mean = &b.mean[..dim];
    let mut e = [0.0f32; LANES];
    if b.rank == 0 {
        for ((ri, &xi), m) in r.iter_mut().zip(x).zip(mean) {
            for l in 0..LANES {
                ri[l] = xi - m[l];
                e[l] += ri[l] * ri[l];
            }
        }
        return e;
    }
    // r = x − μ, fused with the first coefficient c_0 = Σ r_i w0_i.
    let mut c = [0.0f32; LANES];
    for (((ri, &xi), m), w) in r.iter_mut().zip(x).zip(mean).zip(&b.basis[..dim]) {
        for l in 0..LANES {
            ri[l] = xi - m[l];
            c[l] += ri[l] * w[l];
        }
    }
    // r −= c_{k−1} w_{k−1}, fused with c_k = Σ r_i w_k,i.
    for k in 1..b.rank {
        let prev = &b.basis[(k - 1) * dim..k * dim];
        let cur = &b.basis[k * dim..(k + 1) * dim];
        let mut next = [0.0f32; LANES];
        for ((ri, p), w) in r.iter_mut().zip(prev).zip(cur) {
            for l in 0..LANES {
                ri[l] -= c[l] * p[l];
                next[l] += ri[l] * w[l];
            }
        }
        c = next;
    }
    // The last update, fused with E = Σ r_i².
    let last = &b.basis[(b.rank - 1) * dim..b.rank * dim];
    for (ri, p) in r.iter_mut().zip(last) {
        for l in 0..LANES {
            ri[l] -= c[l] * p[l];
            e[l] += ri[l] * ri[l];
        }
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resonance::reference_error;

    #[test]
    fn empty_packed_scores_nothing() {
        let p = Packed::new(&[]).unwrap();
        assert_eq!(p.tasks(), 0);
        assert!(p.errors(&[]).unwrap().is_empty());
    }

    #[test]
    fn rank_zero_is_squared_distance() {
        let mean = [0.5f32, -0.25, 1.0];
        let t = TaskView::new(&mean, &[]).unwrap();
        let p = Packed::new(&[t]).unwrap();
        let x = [1.0f32, 1.0, 1.0];
        assert_eq!(p.errors(&x).unwrap()[0], reference_error(&x, &mean, &[]));
    }
}
