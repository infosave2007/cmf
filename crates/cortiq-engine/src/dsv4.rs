//! DeepSeek-V4 blocks that no other supported architecture has.
//!
//! Transcribed from the reference `inference/model.py` + `inference/kernel.py`
//! shipped with the checkpoint, not inferred from tensor names — the pieces
//! below have enough hidden structure (a second normalization on the heads, a
//! bias that steers selection but not weights, a mixing matrix normalized by
//! Sinkhorn) that guessing produces a model which *almost* answers.
//!
//! Each function is the smallest unit the reference defines, so it can be
//! checked on its own. The forward that stitches them together comes after
//! the attention and compressor land.

/// Hyper-connections. The hidden state of this model is not a vector: it is
/// `hc` copies of one (`hc_mult`, 4 in the release). A block folds them to
/// one, runs attention or the FFN, then expands back — there is no ordinary
/// residual anywhere in the stack.
///
/// `mixes` is the per-token projection `F.linear(x.flatten(), hc_fn) * rsqrt`
/// of length `(2 + hc) * hc`; it splits into three parts:
///   * `pre[j]`  — how much of copy `j` goes into the folded vector,
///   * `post[j]` — how much of the block's output returns to copy `j`,
///   * `comb`    — an `hc x hc` matrix mixing the old copies into the new.
///
/// `comb` is made doubly stochastic by Sinkhorn: a row softmax, then
/// alternating row/column normalization. The reference runs the column step
/// once before the loop and `iters - 1` times inside it, which is why the
/// loop below starts from the column-normalized matrix.
pub fn hc_split_sinkhorn(
    mixes: &[f32],
    hc_scale: &[f32; 3],
    hc_base: &[f32],
    hc: usize,
    iters: usize,
    eps: f32,
    pre: &mut [f32],
    post: &mut [f32],
    comb: &mut [f32],
) {
    debug_assert_eq!(mixes.len(), (2 + hc) * hc);
    debug_assert_eq!(comb.len(), hc * hc);
    for j in 0..hc {
        pre[j] = sigmoid(mixes[j] * hc_scale[0] + hc_base[j]) + eps;
        // The post weights carry a factor 2 in the reference — with a
        // sigmoid alone the block's output could never exceed the residual.
        post[j] = 2.0 * sigmoid(mixes[j + hc] * hc_scale[1] + hc_base[j + hc]);
    }
    for j in 0..hc {
        for k in 0..hc {
            let idx = j * hc + k + hc * 2;
            comb[j * hc + k] = mixes[idx] * hc_scale[2] + hc_base[idx];
        }
    }
    // row softmax + eps
    for j in 0..hc {
        let row = &mut comb[j * hc..(j + 1) * hc];
        let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v = *v / sum + eps;
        }
    }
    // one column normalization, then (iters - 1) row/column rounds
    normalize_cols(comb, hc, eps);
    for _ in 0..iters.saturating_sub(1) {
        normalize_rows(comb, hc, eps);
        normalize_cols(comb, hc, eps);
    }
}

fn normalize_rows(m: &mut [f32], n: usize, eps: f32) {
    for j in 0..n {
        let s: f32 = m[j * n..(j + 1) * n].iter().sum::<f32>() + eps;
        for v in m[j * n..(j + 1) * n].iter_mut() {
            *v /= s;
        }
    }
}

fn normalize_cols(m: &mut [f32], n: usize, eps: f32) {
    for k in 0..n {
        let mut s = eps;
        for j in 0..n {
            s += m[j * n + k];
        }
        for j in 0..n {
            m[j * n + k] /= s;
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The projection feeding `hc_split_sinkhorn`: the `hc` copies are flattened
/// to one `hc*dim` vector, RMS-scaled (no learned weight — the reference uses
/// a bare rsqrt of the mean square), and projected by `hc_fn` `[mix_hc, hc*dim]`.
pub fn hc_mixes(x_flat: &[f32], hc_fn: &[f32], mix_hc: usize, eps: f32, out: &mut [f32]) {
    let n = x_flat.len();
    debug_assert_eq!(hc_fn.len(), mix_hc * n);
    debug_assert_eq!(out.len(), mix_hc);
    let ms = x_flat.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rsqrt = 1.0 / (ms + eps).sqrt();
    for (i, o) in out.iter_mut().enumerate() {
        let row = &hc_fn[i * n..(i + 1) * n];
        *o = row.iter().zip(x_flat).map(|(a, b)| a * b).sum::<f32>() * rsqrt;
    }
}

/// Fold `hc` copies into one vector: `y = Σ_j pre[j] · x[j]`.
pub fn hc_fold(x: &[f32], pre: &[f32], hc: usize, dim: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), hc * dim);
    out.fill(0.0);
    for j in 0..hc {
        let w = pre[j];
        let src = &x[j * dim..(j + 1) * dim];
        for (o, v) in out.iter_mut().zip(src) {
            *o += w * v;
        }
    }
}

/// Expand the block's output back into `hc` copies:
/// `y[j] = post[j] · out + Σ_k comb[k][j] · residual[k]`.
///
/// Note the transpose: the reference sums over the SECOND-to-last axis of
/// `comb.unsqueeze(-1) * residual.unsqueeze(-2)`, i.e. copy `k` of the
/// residual contributes to new copy `j` with weight `comb[k][j]`.
pub fn hc_expand(
    block_out: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    hc: usize,
    dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(residual.len(), hc * dim);
    debug_assert_eq!(out.len(), hc * dim);
    for j in 0..hc {
        let dst = &mut out[j * dim..(j + 1) * dim];
        let p = post[j];
        for (d, o) in dst.iter_mut().enumerate() {
            *o = p * block_out[d];
        }
        for k in 0..hc {
            let w = comb[k * hc + j];
            let src = &residual[k * dim..(k + 1) * dim];
            for (o, v) in dst.iter_mut().zip(src) {
                *o += w * v;
            }
        }
    }
}

/// The head fold, run once after the last layer: same shape as `hc_fold`'s
/// weights but WITHOUT Sinkhorn — a plain sigmoid gate per copy.
pub fn hc_head_pre(mixes: &[f32], scale: f32, base: &[f32], hc: usize, eps: f32, pre: &mut [f32]) {
    for j in 0..hc {
        pre[j] = sigmoid(mixes[j] * scale + base[j]) + eps;
    }
}

/// MoE routing. Three details decide whether this model answers or merely
/// produces fluent text:
///   * the score is `sqrt(softplus(x))`, not a softmax or a sigmoid;
///   * the selection bias shifts WHICH experts win but never the weights —
///     those come from the pre-bias scores;
///   * the weights are renormalized over the chosen experts, then scaled.
///
/// `bias` is `None` on the hash layers, where `indices` come from a
/// token-id table instead (see `hash_route`).
pub fn route(
    scores_in: &[f32],
    bias: Option<&[f32]>,
    top_k: usize,
    route_scale: f32,
    indices: &mut Vec<usize>,
    weights: &mut Vec<f32>,
) {
    let n = scores_in.len();
    let mut scores = Vec::with_capacity(n);
    for &s in scores_in {
        // softplus, guarded like the reference's F.softplus (linear past 20)
        let sp = if s > 20.0 { s } else { (1.0 + s.exp()).ln() };
        scores.push(sp.sqrt());
    }
    indices.clear();
    weights.clear();
    let mut shifted: Vec<f32> = match bias {
        Some(b) => scores.iter().zip(b).map(|(s, b)| s + b).collect(),
        None => scores.clone(),
    };
    for _ in 0..top_k.min(n) {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in shifted.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        indices.push(best);
        // the WEIGHT is the pre-bias score
        weights.push(scores[best]);
        shifted[best] = f32::NEG_INFINITY;
    }
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in weights.iter_mut() {
            *w = *w / sum * route_scale;
        }
    }
}

/// Hash layers: the experts of token `tid` are a row of the `tid2eid` table,
/// and the router does not run at all. Their weights still come from the
/// scored path (the reference gathers `original_scores` at those indices).
pub fn hash_route(tid2eid: &[f32], vocab: usize, top_k: usize, tid: u32) -> Vec<usize> {
    let row = (tid as usize).min(vocab.saturating_sub(1)) * top_k;
    (0..top_k)
        .map(|k| tid2eid.get(row + k).copied().unwrap_or(0.0) as usize)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sinkhorn's whole point is a doubly stochastic matrix: every row and
    /// every column sums to one. If the alternating normalization is wrong
    /// (or the loop count is off by one) the sums drift, and the residual
    /// mixing quietly gains or loses mass on every layer.
    #[test]
    fn sinkhorn_leaves_the_mixing_matrix_doubly_stochastic() {
        let hc = 4;
        let mix_hc = (2 + hc) * hc;
        // a deliberately lopsided projection
        let mixes: Vec<f32> = (0..mix_hc).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let base: Vec<f32> = (0..mix_hc).map(|i| (i as f32 * 0.11).cos()).collect();
        let (mut pre, mut post, mut comb) = (vec![0.0; hc], vec![0.0; hc], vec![0.0; hc * hc]);
        hc_split_sinkhorn(
            &mixes,
            &[1.0, 1.0, 1.0],
            &base,
            hc,
            20,
            1e-6,
            &mut pre,
            &mut post,
            &mut comb,
        );
        for j in 0..hc {
            let r: f32 = comb[j * hc..(j + 1) * hc].iter().sum();
            assert!((r - 1.0).abs() < 2e-3, "row {j} sums to {r}");
            let c: f32 = (0..hc).map(|k| comb[k * hc + j]).sum();
            assert!((c - 1.0).abs() < 2e-3, "col {j} sums to {c}");
        }
        // pre is a gate in (eps, 1+eps); post carries the factor 2
        assert!(pre.iter().all(|&v| v > 0.0 && v < 1.001));
        assert!(post.iter().all(|&v| v >= 0.0 && v <= 2.0));
    }

    /// Folding four copies and expanding them back must preserve a constant
    /// state exactly when the block contributes nothing: with post = 0 the
    /// expansion is a doubly stochastic mix of identical copies, i.e. itself.
    #[test]
    fn expand_of_identical_copies_is_a_fixed_point() {
        let (hc, dim) = (4usize, 3usize);
        let residual: Vec<f32> = std::iter::repeat([1.5f32, -2.0, 0.25])
            .take(hc)
            .flatten()
            .collect();
        let comb = {
            // exactly doubly stochastic: uniform
            vec![0.25f32; hc * hc]
        };
        let post = vec![0.0f32; hc];
        let mut out = vec![0.0f32; hc * dim];
        hc_expand(&[0.0; 3], &residual, &post, &comb, hc, dim, &mut out);
        for (o, r) in out.iter().zip(&residual) {
            assert!((o - r).abs() < 1e-6, "{o} vs {r}");
        }
    }

    /// The bias must move the SELECTION without touching the weights: with a
    /// large bias on a low-scoring expert it gets picked, but its weight is
    /// still its own (small) score, renormalized.
    #[test]
    fn selection_bias_steers_the_choice_but_not_the_weights() {
        let scores = [3.0f32, 0.1, 2.0, 0.05];
        let bias = [0.0f32, 10.0, 0.0, 0.0];
        let (mut idx, mut w) = (Vec::new(), Vec::new());
        route(&scores, Some(&bias), 2, 1.5, &mut idx, &mut w);
        assert_eq!(idx[0], 1, "the biased expert must win selection");
        assert_eq!(idx[1], 0);
        // weights come from sqrt(softplus(score)) BEFORE the bias, so the
        // biased expert's share must be the smaller of the two
        assert!(w[0] < w[1], "biased expert kept its own (small) weight");
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.5).abs() < 1e-5, "weights renormalize then scale");
    }

    #[test]
    fn hash_route_reads_the_table_row() {
        // vocab 3, top_k 2
        let table = [7.0f32, 9.0, 1.0, 2.0, 5.0, 6.0];
        assert_eq!(hash_route(&table, 3, 2, 0), vec![7, 9]);
        assert_eq!(hash_route(&table, 3, 2, 2), vec![5, 6]);
        // out-of-range ids clamp instead of panicking
        assert_eq!(hash_route(&table, 3, 2, 99), vec![5, 6]);
    }
}
