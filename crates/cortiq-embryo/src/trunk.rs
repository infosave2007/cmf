//! Function-preserving trunk growth (`cortiq-embryo grow-trunk`): a wider
//! and/or deeper genome that computes exactly what its source computes, so
//! training continues from everything the small genome has learned. No
//! foreign weights anywhere: every new parameter is a copy, a zero or noise.
//!
//! Width (hidden `H → H'`, FFN `inter → inter'`): the residual stream
//! becomes `[s·x, 0]` with `s = √(H'/H)`. An RMSNorm over `H'` then sees the
//! same mean square (eps included), so every pre-norm weight is divided by
//! `s` and every write into the stream (embedding, mixer and FFN/expert
//! output rows) is multiplied by `s`. The head reads the tied embedding
//! (now `s·E`), which `final_norm / s²` compensates; the cluster table is
//! multiplied by `s`. New residual channels start at zero: their output
//! rows are zero and their input columns are random — a zero column would
//! never receive a gradient, and neither would the channel. New FFN units
//! get random gate/up rows and zero down columns. Expert descriptors are
//! padded with zeros (μ, U), which leaves every routing score unchanged.
//!
//! Depth (`double_depth`): after every block of layers that ends at an
//! anchor, a copy of that block is inserted with its output projections
//! (mixer `o`, FFN and expert `down`) zeroed — an identity on the stream
//! that starts from the source block's features. 8 layers with anchors 3, 7
//! become 16 with anchors 3, 7, 11, 15.

use crate::model::{EmbryoCfg, Layout, Mixer};
use crate::train::Checkpoint;
use anyhow::{Context, bail, ensure};
use std::collections::HashMap;

/// Target geometry of a trunk growth.
#[derive(Clone, Debug)]
pub struct TrunkGrowth {
    pub hidden: usize,
    pub inter: usize,
    pub double_depth: bool,
    pub seed: u64,
}

/// New layer index → (source layer, inserted copy).
pub fn depth_map(cfg: &EmbryoCfg, double: bool) -> Vec<(usize, bool)> {
    if !double {
        return (0..cfg.layers).map(|l| (l, false)).collect();
    }
    let mut out = Vec::with_capacity(2 * cfg.layers);
    let mut block = Vec::new();
    for l in 0..cfg.layers {
        block.push(l);
        if cfg.is_anchor(l) || l + 1 == cfg.layers {
            out.extend(block.iter().map(|&s| (s, false)));
            out.extend(block.iter().map(|&s| (s, true)));
            block.clear();
        }
    }
    out
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn gauss(&mut self) -> f32 {
        let u1 = ((self.next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}

fn rms(x: &[f32]) -> f32 {
    ((x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len().max(1) as f64).sqrt()) as f32
}

/// Copy `[rows, cols]` into the top-left of `[rows2, cols2]`, times `scale`.
fn place(dst: &mut [f32], cols2: usize, src: &[f32], cols: usize, scale: f32) {
    for (r, row) in src.chunks_exact(cols).enumerate() {
        for (d, s) in dst[r * cols2..r * cols2 + cols].iter_mut().zip(row) {
            *d = s * scale;
        }
    }
}

/// The grown checkpoint (no optimizer moments: the trainer restarts them).
pub fn grow_trunk(ck: &Checkpoint, g: &TrunkGrowth) -> anyhow::Result<Checkpoint> {
    let c0 = &ck.cfg;
    ensure!(c0.mixer == Mixer::Gdn, "grow-trunk: only the GDN genome grows (mixer {:?})", c0.mixer);
    ensure!(
        !c0.gdn_lane && !c0.gqa_lane && !c0.phase_delta && c0.phase_delta_layers.is_none() && c0.phase_delta_layer.is_none(),
        "grow-trunk: lanes and phase-delta layers are not supported"
    );
    ensure!(g.hidden >= c0.hidden && g.inter >= c0.inter, "grow-trunk: a genome only grows");
    ensure!(
        g.hidden.is_multiple_of(64) && g.inter.is_multiple_of(64),
        "grow-trunk: hidden and inter must be multiples of 64 (GEMM tiles)"
    );
    let map = depth_map(c0, g.double_depth);
    let mut c1 = c0.clone();
    c1.hidden = g.hidden;
    c1.inter = g.inter;
    c1.layers = map.len();
    c1.anchor_layers = Some((0..map.len()).filter(|&l| c0.is_anchor(map[l].0)).collect());
    let (l0, l1) = (Layout::new(c0), Layout::new(&c1));
    ensure!(ck.params.len() == l0.total, "grow-trunk: checkpoint/layout mismatch");
    let (h0, h1, i0, i1) = (c0.hidden, c1.hidden, c0.inter, c1.inter);
    let s = (h1 as f32 / h0 as f32).sqrt();
    let old: HashMap<&str, (usize, usize)> = l0.names.iter().map(|(n, o, len)| (n.as_str(), (*o, *len))).collect();
    let mut rng = Rng(g.seed ^ 0x4752_4f57_5452_554e);
    let mut p = vec![0.0f32; l1.total];
    for (name, off, len) in &l1.names {
        let (src_name, copy) = match name.strip_prefix("layers.") {
            Some(rest) => {
                let (l, suffix) = rest.split_once('.').context("layer tensor name")?;
                let (sl, copy) = map[l.parse::<usize>()?];
                (format!("layers.{sl}.{suffix}"), copy)
            }
            None => (name.clone(), false),
        };
        let &(so, slen) = old
            .get(src_name.as_str())
            .with_context(|| format!("grow-trunk: no source tensor for {name}"))?;
        let src = &ck.params[so..so + slen];
        let dst = &mut p[*off..*off + len];
        let suffix = src_name.rsplit_once('.').map_or(src_name.as_str(), |x| x.1);
        let kind = if src_name.starts_with("layers.") {
            src_name.splitn(3, '.').nth(2).unwrap_or("")
        } else {
            src_name.as_str()
        };
        match kind {
            "embed" | "head.clusters" => {
                place(dst, h1, src, h0, s);
            }
            "final_norm" | "ln1" | "ln2" => {
                let k = if kind == "final_norm" { 1.0 / (s * s) } else { 1.0 / s };
                let fill = src.iter().map(|x| x.abs()).sum::<f32>() / h0 as f32 * k;
                for (j, d) in dst.iter_mut().enumerate() {
                    *d = if j < h0 { src[j] * k } else { fill };
                }
            }
            "attn.o" | "gdn.o" | "ffn.down" => {
                // [H, cols]; cols = inter for the FFN
                let cols = slen / h0;
                if !copy {
                    place(dst, len / h1, src, cols, s);
                }
            }
            "attn.q" | "attn.k" | "attn.v" | "gdn.in_qkv" | "gdn.in_z" | "gdn.in_a" | "gdn.in_b" | "ffn.gate"
            | "ffn.up" => {
                in_proj(dst, src, h0, h1, &mut rng);
            }
            "gdn.conv" | "gdn.alog" | "gdn.dt_bias" | "gdn.norm" | "attn.sink_k" | "attn.sink_v" => {
                ensure!(slen == *len, "grow-trunk: {name} changed size");
                dst.copy_from_slice(src);
            }
            _ if kind.starts_with("experts.") && (suffix == "gate" || suffix == "up") => {
                in_proj(dst, src, h0, h1, &mut rng);
            }
            _ if kind.starts_with("experts.") && suffix == "down" => {
                if !copy {
                    place(dst, i1, src, i0, s);
                }
            }
            _ => bail!("grow-trunk: no growth rule for tensor {name}"),
        }
    }
    let mut extras = Vec::new();
    let (e, n0, n1) = (c0.experts, c0.layers, c1.layers);
    for (n, x) in &ck.extras {
        let grown = match n.as_str() {
            "desc.mu" | "desc.u" => {
                // [L, E, K, H] (K = 1 for μ)
                ensure!(x.len() % (n0 * e * h0) == 0, "grow-trunk: {n} has an odd size");
                let k = x.len() / (n0 * e * h0);
                let mut y = vec![0.0f32; n1 * e * k * h1];
                for (l, &(sl, _)) in map.iter().enumerate() {
                    let src = &x[sl * e * k * h0..(sl + 1) * e * k * h0];
                    place(&mut y[l * e * k * h1..(l + 1) * e * k * h1], h1, src, h0, 1.0);
                }
                y
            }
            "desc.bias" => {
                ensure!(x.len() == n0 * e, "grow-trunk: desc.bias has an odd size");
                map.iter().flat_map(|&(sl, _)| x[sl * e..(sl + 1) * e].to_vec()).collect()
            }
            _ => bail!("grow-trunk: no growth rule for extra {n}"),
        };
        extras.push((n.clone(), grown));
    }
    Ok(Checkpoint {
        cfg: c1,
        step: ck.step,
        params: p,
        m: None,
        v: None,
        extras,
    })
}

/// `[rows, H]` → `[rows', H']`: old block as is; new columns of live rows and
/// whole new rows are noise at the source's RMS (rows that are zero in the
/// source — padding — stay zero).
fn in_proj(dst: &mut [f32], src: &[f32], h0: usize, h1: usize, rng: &mut Rng) {
    let sigma = rms(src);
    let rows0 = src.len() / h0;
    for (r, row) in dst.chunks_exact_mut(h1).enumerate() {
        if r < rows0 {
            let old = &src[r * h0..(r + 1) * h0];
            row[..h0].copy_from_slice(old);
            if old.iter().any(|v| *v != 0.0) {
                for d in &mut row[h0..] {
                    *d = sigma * rng.gauss();
                }
            }
        } else {
            for d in row.iter_mut() {
                *d = sigma * rng.gauss();
            }
        }
    }
}
