//! O(1) far field for the attention layers of the Qwen3.8-Flash-Next device path
//! (`gpu_wgpu::qwen4`), after `dory`: a prompt runs exact up to `seal` rows, then its
//! keys, values and a strided trace of its queries leave the card, `nystrom` builds a
//! sink + window + landmark state per KV head on the host, and the sealed layers go on
//! with the engine's O(1) kernels (`o1_batch`): every later row attends the sinks, the
//! window and the far field, and the attention's footprint stays where the seal left it.
//! The GDN layers are bounded already; the routed experts do not see the change.
//!
//! The far-turbo header (`o1_turbo` in the provenance, `cortiq far-turbo`) switches it
//! on; `CMF_QWEN_O1=m,w,sink,seal` runs any file so, `CMF_QWEN_O1=off` keeps a turbo
//! file exact.

use crate::dory::{DoryO1, ExactRows, MAX_NEAR, QT_STRIDE};
use cortiq_core::CmfModel;
use std::sync::Arc;

/// Rows the near field may hold on the batch kernels (`o1_batch`): sinks, the window
/// and the exact rows together.
pub const BATCH_NEAR: usize = 2052;

/// The layer's far-field settings and the rows kept exact.
#[derive(Clone)]
pub struct Qwen4O1 {
    pub o1: DoryO1,
    pub exact: Option<ExactRows>,
}

impl Qwen4O1 {
    /// `CMF_QWEN_O1=m,w,sink,seal` first, then the file's turbo header; `off` for none.
    pub fn from_model(model: &CmfModel) -> Option<Self> {
        match std::env::var("CMF_QWEN_O1").ok().as_deref() {
            Some("off") | Some("0") => return None,
            Some(v) => {
                let n: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if n.len() == 4 {
                    return Some(Self { o1: DoryO1 { m: n[0], w: n[1], sink: n[2], seal: n[3].max(n[1] + n[2] + 16) }, exact: None });
                }
                tracing::warn!("CMF_QWEN_O1={v}: want m,w,sink,seal");
                return None;
            }
            None => {}
        }
        let h = model.header.provenance.as_ref()?.get("o1_turbo")?;
        let field = |k: &str| h.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
        let (m, w, sink, seal) = (field("m")?, field("w")?, field("sink")?, field("seal")?);
        let budget = field("exact_rows").unwrap_or(0);
        let exact = (budget > 0)
            .then(|| crate::loader::load_f32(model, "o1.exact.weights", &crate::loader::Overlay::None).ok())
            .flatten()
            .map(|w| ExactRows { weights: Arc::new(w), budget });
        Some(Self { o1: DoryO1 { m, w, sink, seal: seal.max(w + sink + 16) }, exact })
    }

    /// Rows the near field holds past the seal: sinks, the window, the exact rows.
    pub fn near(&self) -> usize {
        self.o1.w + self.o1.sink + self.exact.as_ref().map_or(0, |x| x.budget)
    }

    /// What the device kernels take: landmarks to 32, a head to 256 lanes, a window the
    /// batch kernels can evict from, and a near field they can hold.
    pub fn check(&self, head_dim: usize) -> Result<(), String> {
        let o = &self.o1;
        if o.m == 0 || o.m > 32 {
            return Err(format!("qwen4 O(1): {} landmarks (want 1..=32)", o.m));
        }
        if head_dim > 256 || head_dim % 4 != 0 {
            return Err(format!("qwen4 O(1): head_dim {head_dim} (want a multiple of 4 up to 256)"));
        }
        if o.w < 64 {
            return Err(format!("qwen4 O(1): window {} (want at least 64)", o.w));
        }
        let near = self.near();
        if near > BATCH_NEAR.min(MAX_NEAR) {
            return Err(format!("qwen4 O(1): window {} + sinks {} + exact rows {} = {near} go over the kernels' {BATCH_NEAR}", o.w, o.sink, near - o.w - o.sink));
        }
        Ok(())
    }
}

/// A device cache `[nkv][cap][hd]` as host rows `[row][nkv][hd]` for `rows` positions.
pub fn rows_from_heads(src: &[f32], nkv: usize, cap: usize, hd: usize, rows: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * nkv * hd];
    for h in 0..nkv {
        for r in 0..rows {
            let s = (h * cap + r) * hd;
            let d = (r * nkv + h) * hd;
            out[d..d + hd].copy_from_slice(&src[s..s + hd]);
        }
    }
    out
}

/// The sealed prompt's far states, one per KV head: `ks`/`vs` `[row][nkv][hd]`, `qs`
/// the query trace `[row / QT_STRIDE][nh][hd]`, `order` the rows' order into the state
/// (`dory::seal_order`), `kept` the exact rows among the sinks.
#[allow(clippy::too_many_arguments)]
pub fn far_states(nh: usize, nkv: usize, hd: usize, o1: DoryO1, ks: &[f32], vs: &[f32], qs: &[f32], rows: usize, order: &[u32], kept: usize) -> Vec<crate::nystrom::NystromState> {
    crate::dory::far_states_geom(nh, nkv, hd, o1, ks, vs, qs, QT_STRIDE, rows, order, kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_from_heads_transposes_the_cache() {
        let (nkv, cap, hd, rows) = (2, 4, 3, 3);
        let src: Vec<f32> = (0..nkv * cap * hd).map(|i| i as f32).collect();
        let out = rows_from_heads(&src, nkv, cap, hd, rows);
        for r in 0..rows {
            for h in 0..nkv {
                for d in 0..hd {
                    assert_eq!(out[(r * nkv + h) * hd + d], src[(h * cap + r) * hd + d]);
                }
            }
        }
    }

    #[test]
    fn check_rejects_what_the_kernels_cannot_hold() {
        let mk = |m, w, sink| Qwen4O1 { o1: DoryO1 { m, w, sink, seal: 4096 }, exact: None };
        assert!(mk(16, 1024, 4).check(256).is_ok());
        assert!(mk(64, 1024, 4).check(256).is_err());
        assert!(mk(16, 32, 4).check(256).is_err());
        assert!(mk(16, 2048, 8).check(256).is_err());
        assert!(mk(16, 1024, 4).check(300).is_err());
    }
}
