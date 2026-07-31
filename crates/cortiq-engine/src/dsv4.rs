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

/// Rotary on the LAST `rd` dims only — the rest of the head carries no
/// position. `inverse` runs the rotation backwards, which the reference
/// applies to the attention OUTPUT before the o-projection (the value
/// stream carries the same rope-tagged tail as the keys, and it has to be
/// untagged again). Missing that step leaves a model that reads fluently
/// and attends to the wrong offsets.
pub fn rope_tail(v: &mut [f32], inv_freq: &[f32], pos: usize, rd: usize, inverse: bool) {
    let n = v.len();
    debug_assert!(rd <= n && rd % 2 == 0);
    let base = n - rd;
    let half = rd / 2;
    for i in 0..half {
        let theta = pos as f32 * inv_freq[i];
        let (s, c) = (theta.sin(), theta.cos());
        let s = if inverse { -s } else { s };
        let a = v[base + i];
        let b = v[base + i + half];
        v[base + i] = a * c - b * s;
        v[base + i + half] = a * s + b * c;
    }
}

/// RMS normalize in place with no learned weight — the reference applies
/// this to each attention head AFTER `wq_b`, on top of the `q_norm` that
/// already normalized the LoRA rank. Two normalizations, not one.
pub fn rms_inplace(v: &mut [f32], eps: f32) {
    let ms = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Attention over an explicit position LIST (window ⊕ compressed), with a
/// learned per-head sink. The sink is an extra logit with no value vector:
/// it lets a head attend to "nothing", so its softmax denominator carries
/// `exp(sink - max)` while contributing no output. Index `usize::MAX`
/// marks a masked slot (the reference writes -1 into topk_idxs).
pub fn sparse_attend(
    q: &[f32],
    kv: &[f32],
    idxs: &[usize],
    sink: f32,
    scale: f32,
    head_dim: usize,
    out: &mut [f32],
) {
    let mut m = sink;
    let mut scores = Vec::with_capacity(idxs.len());
    for &p in idxs {
        if p == usize::MAX {
            scores.push(f32::NEG_INFINITY);
            continue;
        }
        let k = &kv[p * head_dim..(p + 1) * head_dim];
        let dot: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
        m = m.max(dot);
        scores.push(dot);
    }
    let mut denom = (sink - m).exp();
    out.fill(0.0);
    for (&p, &s) in idxs.iter().zip(&scores) {
        if p == usize::MAX {
            continue;
        }
        let w = (s - m).exp();
        denom += w;
        let v = &kv[p * head_dim..(p + 1) * head_dim];
        for (o, x) in out.iter_mut().zip(v) {
            *o += w * x;
        }
    }
    let inv = 1.0 / denom;
    for o in out.iter_mut() {
        *o *= inv;
    }
}

/// The grouped low-rank output projection: heads are split into `groups`,
/// each group's slice is projected to `lora` by its own block of `wo_a`,
/// and the concatenation goes through `wo_b`. `wo_a` is stored
/// `[groups, lora, per_group]`.
pub fn o_project(
    attn: &[f32],
    wo_a: &[f32],
    wo_b: &[f32],
    groups: usize,
    lora: usize,
    dim: usize,
    out: &mut [f32],
) {
    let per_group = attn.len() / groups;
    let mut mid = vec![0.0f32; groups * lora];
    for g in 0..groups {
        let src = &attn[g * per_group..(g + 1) * per_group];
        let blk = &wo_a[g * lora * per_group..(g + 1) * lora * per_group];
        for r in 0..lora {
            let row = &blk[r * per_group..(r + 1) * per_group];
            mid[g * lora + r] = row.iter().zip(src).map(|(a, b)| a * b).sum();
        }
    }
    debug_assert_eq!(wo_b.len(), dim * mid.len());
    for d in 0..dim {
        let row = &wo_b[d * mid.len()..(d + 1) * mid.len()];
        out[d] = row.iter().zip(&mid).map(|(a, b)| a * b).sum();
    }
}

/// The KV compressor: `ratio` consecutive tokens collapse into one through
/// a softmax over the window, with a learned in-window position bias added
/// to the scores BEFORE the softmax. Both streams come from the same
/// hidden state through `wkv` and `wgate`.
pub fn compress_window(
    kv: &[f32],
    score: &[f32],
    ape: &[f32],
    ratio: usize,
    width: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(kv.len(), ratio * width);
    debug_assert_eq!(ape.len(), ratio * width);
    out.fill(0.0);
    for d in 0..width {
        let mut m = f32::NEG_INFINITY;
        for t in 0..ratio {
            m = m.max(score[t * width + d] + ape[t * width + d]);
        }
        let mut denom = 0.0;
        for t in 0..ratio {
            denom += (score[t * width + d] + ape[t * width + d] - m).exp();
        }
        for t in 0..ratio {
            let w = (score[t * width + d] + ape[t * width + d] - m).exp() / denom;
            out[d] += w * kv[t * width + d];
        }
    }
}

/// The sparse indexer's scoring pass. For each query it ranks the
/// compressed positions and keeps the best `topk`.
///
/// Three details from the reference that a shape-only reading misses:
///   * the query comes from the SHARED LoRA output `qr` (the output of
///     `q_norm(wq_a(x))`, before attention's own `wq_b`), through the
///     indexer's own `wq_b` — not from attention's queries;
///   * scores are **relu'd** before the per-head weighting, so a head can
///     only ever vote for a position, never against it;
///   * the per-head weights are a projection of the hidden state scaled by
///     `head_dim^-0.5 * n_heads^-0.5`.
///
/// `causal_limit` is the number of compressed positions this query may see
/// (`(pos + 1) / ratio`); anything at or past it is masked.
pub fn index_scores(
    q_heads: &[f32],
    kv: &[f32],
    head_weights: &[f32],
    n_heads: usize,
    head_dim: usize,
    n_pos: usize,
    causal_limit: usize,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.resize(n_pos, 0.0);
    for t in 0..n_pos {
        if t >= causal_limit {
            out[t] = f32::NEG_INFINITY;
            continue;
        }
        let k = &kv[t * head_dim..(t + 1) * head_dim];
        let mut acc = 0.0;
        for h in 0..n_heads {
            let q = &q_heads[h * head_dim..(h + 1) * head_dim];
            let dot: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
            // relu BEFORE weighting: a head votes for a position or abstains
            acc += dot.max(0.0) * head_weights[h];
        }
        out[t] = acc;
    }
}

/// Top-`k` positions by score, ties broken by the lower index so the choice
/// is deterministic across backends. Masked slots (-inf) never win, and a
/// short history simply returns fewer than `k`.
pub fn top_k_positions(scores: &[f32], k: usize, out: &mut Vec<usize>) {
    out.clear();
    let mut taken = vec![false; scores.len()];
    for _ in 0..k.min(scores.len()) {
        let mut best = usize::MAX;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in scores.iter().enumerate() {
            if !taken[i] && v > bv && v.is_finite() {
                bv = v;
                best = i;
            }
        }
        if best == usize::MAX {
            break;
        }
        taken[best] = true;
        out.push(best);
    }
    out.sort_unstable();
}

/// SwiGLU expert: `w2(silu(w1(x)) * w3(x))`, with the routing weight folded
/// in before the down projection exactly as the reference does.
pub fn expert_swiglu(
    x: &[f32],
    w1: &dyn Fn(&[f32], &mut [f32]),
    w3: &dyn Fn(&[f32], &mut [f32]),
    w2: &dyn Fn(&[f32], &mut [f32]),
    inter: usize,
    weight: f32,
    out: &mut [f32],
) {
    let mut gate = vec![0.0f32; inter];
    let mut up = vec![0.0f32; inter];
    w1(x, &mut gate);
    w3(x, &mut up);
    for (g, u) in gate.iter_mut().zip(&up) {
        let silu = *g / (1.0 + (-*g).exp());
        *g = silu * u * weight;
    }
    w2(&gate, out);
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

    /// The sink is an extra logit with no value: it must lower every
    /// weight without adding output. With a huge sink the head should
    /// attend to almost nothing.
    #[test]
    fn attention_sink_drains_weight_without_contributing_output() {
        let hd = 2;
        let q = [1.0f32, 0.0];
        let kv = [1.0f32, 0.0, 0.0, 1.0];
        let mut out = vec![0.0f32; hd];
        sparse_attend(&q, &kv, &[0, 1], f32::NEG_INFINITY, 1.0, hd, &mut out);
        let plain = out.clone();
        assert!(plain[0] > plain[1], "the aligned key must dominate");
        sparse_attend(&q, &kv, &[0, 1], 20.0, 1.0, hd, &mut out);
        assert!(
            out[0] < plain[0] * 0.01 && out[1] < plain[1] * 0.01,
            "a large sink must drain nearly all the mass: {out:?}"
        );
    }

    /// A masked slot must be ignored entirely — not folded in as a zero
    /// key, which would still add exp(0) to the denominator.
    #[test]
    fn masked_positions_leave_the_denominator_alone() {
        let hd = 2;
        let q = [1.0f32, 0.0];
        let kv = [1.0f32, 0.0, 0.0, 1.0];
        let (mut a, mut b) = (vec![0.0f32; hd], vec![0.0f32; hd]);
        sparse_attend(&q, &kv, &[0], f32::NEG_INFINITY, 1.0, hd, &mut a);
        sparse_attend(&q, &kv, &[0, usize::MAX], f32::NEG_INFINITY, 1.0, hd, &mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
    }

    /// Forward then inverse rotation is the identity — the property the
    /// output path depends on.
    #[test]
    fn rope_tail_inverts_itself() {
        let inv_freq = [1.0f32, 0.5];
        let orig = [9.0f32, 8.0, 1.0, 2.0, 3.0, 4.0];
        let mut v = orig;
        rope_tail(&mut v, &inv_freq, 7, 4, false);
        assert!(v[..2] == orig[..2], "the non-rope head must not move");
        assert!(v[2..] != orig[2..], "the tail must actually rotate");
        rope_tail(&mut v, &inv_freq, 7, 4, true);
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    /// The window pooling is a softmax per DIMENSION over the ratio, with
    /// the position bias inside the exponent.
    #[test]
    fn compressor_pools_the_window_per_dimension() {
        let (ratio, width) = (2usize, 2usize);
        let kv = [1.0f32, 10.0, 3.0, 20.0];
        // dim 0: equal scores → mean; dim 1: second token wins by a mile
        let score = [0.0f32, 0.0, 0.0, 50.0];
        let ape = vec![0.0f32; ratio * width];
        let mut out = vec![0.0f32; width];
        compress_window(&kv, &score, &ape, ratio, width, &mut out);
        assert!((out[0] - 2.0).abs() < 1e-5, "equal scores average: {}", out[0]);
        assert!((out[1] - 20.0).abs() < 1e-3, "a dominant score wins: {}", out[1]);
    }

    /// A negative dot product must not drag a position down: the relu
    /// means heads abstain rather than veto.
    #[test]
    fn index_scores_relu_before_weighting() {
        let (nh, hd) = (2usize, 2usize);
        // head 0 aligns with position 0, head 1 anti-aligns with it
        let q = [1.0f32, 0.0, -1.0, 0.0];
        let kv = [1.0f32, 0.0, 0.0, 1.0];
        let w = [1.0f32, 1.0];
        let mut sc = Vec::new();
        index_scores(&q, &kv, &w, nh, hd, 2, 2, &mut sc);
        // without the relu the anti-aligned head would cancel head 0 to zero
        assert!(sc[0] > 0.9, "abstention, not veto: {:?}", sc);
    }

    #[test]
    fn index_scores_mask_the_future() {
        let (nh, hd) = (1usize, 2usize);
        let q = [1.0f32, 0.0];
        let kv = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0];
        let w = [1.0f32];
        let mut sc = Vec::new();
        index_scores(&q, &kv, &w, nh, hd, 3, 2, &mut sc);
        assert!(sc[0].is_finite() && sc[1].is_finite());
        assert!(sc[2] == f32::NEG_INFINITY, "position 2 is in the future");
        let mut idx = Vec::new();
        top_k_positions(&sc, 3, &mut idx);
        assert_eq!(idx, vec![0, 1], "a masked slot never wins a slot");
    }

    #[test]
    fn top_k_is_deterministic_on_ties() {
        let sc = [1.0f32, 1.0, 1.0, 0.0];
        let mut idx = Vec::new();
        top_k_positions(&sc, 2, &mut idx);
        assert_eq!(idx, vec![0, 1], "ties resolve to the lower index");
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
