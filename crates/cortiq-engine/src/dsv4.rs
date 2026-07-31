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
/// `forced` fixes the chosen experts (the hash layers' token-id table). They
/// have to be known here rather than swapped in afterwards: the weights are
/// the scores gathered at whichever indices win, so substituting the indices
/// later leaves every weight attached to a different expert.
pub fn route(
    scores_in: &[f32],
    bias: Option<&[f32]>,
    top_k: usize,
    route_scale: f32,
    forced: Option<&[usize]>,
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
    match forced {
        Some(f) => indices.extend(f.iter().copied()),
        None => {
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
                shifted[best] = f32::NEG_INFINITY;
            }
        }
    }
    // The weight is always the PRE-bias score of the chosen expert.
    for &i in indices.iter() {
        weights.push(scores.get(i).copied().unwrap_or(0.0));
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
/// `wo_a_row` is `(row, x) -> dot`, reading one row of `wo_a` against the
/// slice of `attn` its group owns; `wo_b` is the plain projection of the
/// concatenated groups. Both arrive as closures so the caller can serve them
/// straight from quantized tensors.
pub fn o_project(
    attn: &[f32],
    wo_a_row: &(dyn Fn(usize, &[f32], &mut [f32]) -> f32 + Sync),
    scratch_len: usize,
    wo_b: &dyn Fn(&[f32], &mut [f32]),
    groups: usize,
    lora: usize,
    pool: Option<&crate::pool::Pool>,
    out: &mut [f32],
) {
    let per_group = attn.len() / groups;
    let mut mid = vec![0.0f32; groups * lora];
    let slice_of = |i: usize| {
        let g = i / lora;
        &attn[g * per_group..(g + 1) * per_group]
    };
    match pool {
        // Each row of `mid` is one dot product against its group's slice —
        // independent, so the rows split cleanly. This is the largest
        // single-threaded cost in the decode otherwise: on the release
        // checkpoint wo_a is 33M weights, read once per layer per token.
        Some(p) if mid.len() >= 256 => {
            let addr = crate::pool::SendMut::new(mid.as_mut_ptr());
            p.run_rows(mid.len(), &|start, end| {
                let mut sc = vec![0.0f32; scratch_len];
                for i in start..end {
                    let v = wo_a_row(i, slice_of(i), &mut sc);
                    unsafe { *addr.at(i) = v };
                }
            });
        }
        _ => {
            let mut sc = vec![0.0f32; scratch_len];
            for (i, m) in mid.iter_mut().enumerate() {
                *m = wo_a_row(i, slice_of(i), &mut sc);
            }
        }
    }
    wo_b(&mid, out);
}

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
    let biased: Vec<f32> = score.iter().zip(ape).map(|(s, a)| s + a).collect();
    pool_by_score(kv, &biased, ratio, width, out);
}

/// Softmax over the `slots` axis, per dimension, then the weighted sum —
/// the pooling both the plain and the overlapping compressor end in.
/// `-inf` scores are how an absent slot votes for nothing, so the
/// max-subtraction has to survive a whole column of them.
pub fn pool_by_score(kv: &[f32], score: &[f32], slots: usize, width: usize, out: &mut [f32]) {
    debug_assert_eq!(kv.len(), slots * width);
    debug_assert_eq!(score.len(), slots * width);
    out.fill(0.0);
    for d in 0..width {
        let mut m = f32::NEG_INFINITY;
        for t in 0..slots {
            m = m.max(score[t * width + d]);
        }
        if !m.is_finite() {
            continue;
        }
        let mut denom = 0.0;
        for t in 0..slots {
            denom += (score[t * width + d] - m).exp();
        }
        if denom <= 0.0 {
            continue;
        }
        for t in 0..slots {
            out[d] += ((score[t * width + d] - m).exp() / denom) * kv[t * width + d];
        }
    }
}

/// The overlapping compressor (the release uses it wherever the ratio is 4).
///
/// Each token contributes `2*d` values: the first half belongs to the window
/// that started half a stride earlier, the second half to the current one.
/// At fold time the reference pools `2*ratio` entries of width `d` — the
/// PREVIOUS window's slots taking their first half, the current window's
/// slots taking their second half — then the current window becomes the
/// previous one. An absent previous window votes with `-inf`.
#[allow(clippy::too_many_arguments)]
pub fn compress_window_overlap(
    prev_kv: &[f32],
    prev_score: &[f32],
    cur_kv: &[f32],
    cur_score: &[f32],
    ratio: usize,
    d: usize,
    out: &mut [f32],
) {
    let slots = 2 * ratio;
    let mut kv = vec![0.0f32; slots * d];
    let mut sc = vec![f32::NEG_INFINITY; slots * d];
    let have_prev = prev_kv.len() == ratio * 2 * d;
    for t in 0..ratio {
        if have_prev {
            // the previous window's slots, first half of the dimensions
            kv[t * d..(t + 1) * d].copy_from_slice(&prev_kv[t * 2 * d..t * 2 * d + d]);
            sc[t * d..(t + 1) * d].copy_from_slice(&prev_score[t * 2 * d..t * 2 * d + d]);
        }
        // the current window's slots, second half
        let src = t * 2 * d + d;
        let dst = (ratio + t) * d;
        kv[dst..dst + d].copy_from_slice(&cur_kv[src..src + d]);
        sc[dst..dst + d].copy_from_slice(&cur_score[src..src + d]);
    }
    pool_by_score(&kv, &sc, slots, d, out);
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
///
/// `limit` is the reference's `swiglu_limit` (10.0 in the release), and its
/// asymmetry is not a typo: `up` is clamped on BOTH sides, `gate` only from
/// above — the reference leaves silu's negative tail alone. A limit of 0
/// disables the clamp, which is also what the reference does.
#[allow(clippy::too_many_arguments)]
pub fn expert_swiglu(
    x: &[f32],
    w1: &dyn Fn(&[f32], &mut [f32]),
    w3: &dyn Fn(&[f32], &mut [f32]),
    w2: &dyn Fn(&[f32], &mut [f32]),
    inter: usize,
    weight: f32,
    limit: f32,
    out: &mut [f32],
) {
    let mut gate = vec![0.0f32; inter];
    let mut up = vec![0.0f32; inter];
    w1(x, &mut gate);
    w3(x, &mut up);
    if limit > 0.0 {
        for u in up.iter_mut() {
            *u = u.clamp(-limit, limit);
        }
        for g in gate.iter_mut() {
            *g = g.min(limit);
        }
    }
    for (g, u) in gate.iter_mut().zip(&up) {
        let silu = *g / (1.0 + (-*g).exp());
        *g = silu * u * weight;
    }
    w2(&gate, out);
}

/// Everything one layer needs that is not a plain matrix: the shapes and
/// scalars the reference reads out of `ModelArgs`.
#[derive(Debug, Clone, Copy)]
pub struct Dsv4Cfg {
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub norm_eps: f32,
    pub n_routed_experts: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    pub route_scale: f32,
    /// The reference's `swiglu_limit`; 0 disables the clamp.
    pub swiglu_limit: f32,
    /// Sliding-window size (`window_size`, 128 in the release).
    pub window: usize,
    pub index_topk: usize,
    pub vocab: usize,
}

/// The per-block hyper-connection cycle, which is the same shape around
/// attention and around the FFN: fold the copies, normalize, run the
/// block, expand back. `block` sees a plain `dim`-vector and knows nothing
/// about the copies — that separation is what keeps attention and the MoE
/// free of hyper-connection bookkeeping.
///
/// `hc_fn` is `[mix_hc, hc*dim]`, `hc_base` is `[mix_hc]`, `hc_scale` is 3.
#[allow(clippy::too_many_arguments)]
pub fn hc_block<F: FnMut(&[f32], &mut [f32])>(
    state: &mut [f32],
    hc_fn: &[f32],
    hc_scale: &[f32; 3],
    hc_base: &[f32],
    norm_w: &[f32],
    cfg: &Dsv4Cfg,
    scratch: &mut HcScratch,
    mut block: F,
) {
    let (hc, dim) = (cfg.hc_mult, cfg.dim);
    let mix_hc = (2 + hc) * hc;
    hc_mixes(state, hc_fn, mix_hc, cfg.norm_eps, &mut scratch.mixes);
    hc_split_sinkhorn(
        &scratch.mixes,
        hc_scale,
        hc_base,
        hc,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
        &mut scratch.pre,
        &mut scratch.post,
        &mut scratch.comb,
    );
    hc_fold(state, &scratch.pre, hc, dim, &mut scratch.folded);
    // RMSNorm with the layer's learned weight, on the folded vector.
    let ms = scratch.folded.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv = 1.0 / (ms + cfg.norm_eps).sqrt();
    for (v, w) in scratch.folded.iter_mut().zip(norm_w) {
        *v = *v * inv * w;
    }
    block(&scratch.folded, &mut scratch.block_out);
    scratch.residual.copy_from_slice(state);
    hc_expand(
        &scratch.block_out,
        &scratch.residual,
        &scratch.post,
        &scratch.comb,
        hc,
        dim,
        state,
    );
}

/// Reusable buffers for `hc_block` — one allocation per pipeline, not per
/// layer per token.
pub struct HcScratch {
    pub mixes: Vec<f32>,
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    pub comb: Vec<f32>,
    pub folded: Vec<f32>,
    pub block_out: Vec<f32>,
    pub residual: Vec<f32>,
}

impl HcScratch {
    pub fn new(cfg: &Dsv4Cfg) -> Self {
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        Self {
            mixes: vec![0.0; (2 + hc) * hc],
            pre: vec![0.0; hc],
            post: vec![0.0; hc],
            comb: vec![0.0; hc * hc],
            folded: vec![0.0; dim],
            block_out: vec![0.0; dim],
            residual: vec![0.0; hc * dim],
        }
    }
}

/// The final fold, after the last layer: `hc` copies to one vector, with a
/// plain sigmoid gate (no Sinkhorn), then the model's output norm.
pub fn hc_head_fold(
    state: &[f32],
    hc_fn: &[f32],
    hc_scale: f32,
    hc_base: &[f32],
    cfg: &Dsv4Cfg,
    out: &mut [f32],
) {
    let (hc, dim) = (cfg.hc_mult, cfg.dim);
    let mut mixes = vec![0.0f32; hc];
    hc_mixes(state, hc_fn, hc, cfg.norm_eps, &mut mixes);
    let mut pre = vec![0.0f32; hc];
    hc_head_pre(&mixes, hc_scale, hc_base, hc, cfg.hc_eps, &mut pre);
    hc_fold(state, &pre, hc, dim, out);
}

/// One layer's weights. Everything quantized rides as `QTensor` so the
/// existing kernels (and the mmap) serve them; the small fp32 pieces —
/// norms, the hyper-connection projections, the sink, the compressor's
/// position bias — are plain vectors, exactly as the reference keeps them
/// in fp32 regardless of the checkpoint's storage dtype.
pub struct Dsv4Layer {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    // attention: the double LoRA, the compressed KV, the grouped output
    pub wq_a: crate::qtensor::QTensor,
    pub q_norm: Vec<f32>,
    pub wq_b: crate::qtensor::QTensor,
    pub wkv: crate::qtensor::QTensor,
    pub kv_norm: Vec<f32>,
    pub wo_a: crate::qtensor::QTensor,
    pub wo_b: crate::qtensor::QTensor,
    pub attn_sink: Vec<f32>,
    /// `None` on the pure sliding-window layers (`compress_ratio == 0`).
    pub compressor: Option<Dsv4Compressor>,
    /// Only on the layers whose ratio is 4.
    pub indexer: Option<Dsv4Indexer>,
    // hyper-connections, one set for the attention half and one for the FFN
    pub hc_attn_fn: Vec<f32>,
    pub hc_attn_base: Vec<f32>,
    pub hc_attn_scale: [f32; 3],
    pub hc_ffn_fn: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub hc_ffn_scale: [f32; 3],
    // MoE
    pub gate: crate::qtensor::QTensor,
    /// noaux_tc selection bias — `None` on the hash layers.
    pub gate_bias: Option<Vec<f32>>,
    /// Token-id → expert table on the hash layers, `None` elsewhere.
    pub tid2eid: Option<Vec<f32>>,
    pub experts: Vec<Dsv4Expert>,
    pub shared: Dsv4Expert,
}

pub struct Dsv4Expert {
    pub w1: crate::qtensor::QTensor,
    pub w2: crate::qtensor::QTensor,
    pub w3: crate::qtensor::QTensor,
}

pub struct Dsv4Compressor {
    pub wkv: crate::qtensor::QTensor,
    pub wgate: crate::qtensor::QTensor,
    pub norm: Vec<f32>,
    /// `[ratio, coff*head_dim]` — the in-window position bias.
    pub ape: Vec<f32>,
    pub ratio: usize,
    /// Overlapping windows (the reference sets this when ratio == 4), which
    /// doubles the projection width.
    pub overlap: bool,
}

pub struct Dsv4Indexer {
    pub wq_b: crate::qtensor::QTensor,
    pub weights_proj: crate::qtensor::QTensor,
    pub compressor: Dsv4Compressor,
}

/// Model-global pieces: the embedding, the output head and the final
/// hyper-connection fold.
pub struct Dsv4Globals {
    pub embed: crate::qtensor::QTensor,
    pub norm: Vec<f32>,
    pub head: crate::qtensor::QTensor,
    pub hc_head_fn: Vec<f32>,
    pub hc_head_base: Vec<f32>,
    pub hc_head_scale: f32,
}

/// Per-sequence state. The compressor and the indexer each keep their own
/// compressed cache and a partial window, so decode picks up mid-window
/// exactly where prefill left off.
pub struct Dsv4State {
    /// Sliding-window KV per layer, `[window, head_dim]` ring.
    pub window: Vec<Vec<f32>>,
    /// Compressed KV per layer, appended once per `ratio` tokens.
    pub compressed: Vec<Vec<f32>>,
    /// The indexer's own compressed cache per layer.
    pub index_kv: Vec<Vec<f32>>,
    /// Partial window being accumulated, per layer: kv and score streams.
    pub pending_kv: Vec<Vec<f32>>,
    pub pending_score: Vec<Vec<f32>>,
    /// The window before it, kept only by the overlapping compressor —
    /// its fold reads half its dimensions from the previous stride.
    pub prev_kv: Vec<Vec<f32>>,
    pub prev_score: Vec<Vec<f32>>,
    pub pos: usize,
}

impl Dsv4State {
    pub fn new(layers: usize) -> Self {
        Self {
            window: vec![Vec::new(); layers],
            compressed: vec![Vec::new(); layers],
            index_kv: vec![Vec::new(); layers],
            pending_kv: vec![Vec::new(); layers],
            pending_score: vec![Vec::new(); layers],
            prev_kv: vec![Vec::new(); layers],
            prev_score: vec![Vec::new(); layers],
            pos: 0,
        }
    }
}

/// One attention block for a single position. `hidden` is the folded,
/// normalized vector `hc_block` hands over; the result goes back to it.
///
/// The order matters and is the reference's: q through the LoRA pair with
/// a normalization at each end, kv compressed to one head's width, rope on
/// the tails, the window and the compressed positions concatenated into
/// one index list, sparse attention with the sink, the INVERSE rope on the
/// output, then the grouped low-rank projection.
#[allow(clippy::too_many_arguments)]
pub fn attention_step(
    hidden: &[f32],
    l: &Dsv4Layer,
    cfg: &Dsv4Cfg,
    st: &mut Dsv4State,
    li: usize,
    inv_freq: &[f32],
    pool: Option<&crate::pool::Pool>,
    out: &mut [f32],
) {
    let (hd, rd) = (cfg.head_dim, cfg.rope_head_dim);
    let pos = st.pos;

    // ── q: wq_a → q_norm → wq_b → per-head norm → rope tail ──
    let mut qr = vec![0.0f32; cfg.q_lora_rank];
    l.wq_a.matvec(hidden, &mut qr, pool);
    rms_weighted(&mut qr, &l.q_norm, cfg.norm_eps);
    let mut q = vec![0.0f32; cfg.n_heads * hd];
    l.wq_b.matvec(&qr, &mut q, pool);
    for h in 0..cfg.n_heads {
        let head = &mut q[h * hd..(h + 1) * hd];
        rms_inplace(head, cfg.norm_eps);
        rope_tail(head, inv_freq, pos, rd, false);
    }

    // ── kv: one head's width, shared by every query head ──
    let mut kv = vec![0.0f32; hd];
    l.wkv.matvec(hidden, &mut kv, pool);
    rms_weighted(&mut kv, &l.kv_norm, cfg.norm_eps);
    rope_tail(&mut kv, inv_freq, pos, rd, false);

    // ── the compressor: accumulate `ratio` tokens, then fold them into
    // one compressed entry. The reference fires when (pos+1) % ratio == 0,
    // so a partial window simply waits — which is why the state carries
    // the pending streams across tokens.
    if let Some(cp) = &l.compressor {
        let width = cp.wkv.rows();
        // With overlapping windows the projection is twice the entry width:
        // half the dimensions belong to the previous stride, half to this one.
        let ew = if cp.overlap { width / 2 } else { width };
        let mut ckv = vec![0.0f32; width];
        let mut cscore = vec![0.0f32; width];
        cp.wkv.matvec(hidden, &mut ckv, pool);
        cp.wgate.matvec(hidden, &mut cscore, pool);
        if cp.overlap {
            // The reference biases the score as the token arrives and keeps
            // it biased across the shift, so ape is added ONCE, here.
            let slot = pos % cp.ratio;
            for (c, a) in cscore
                .iter_mut()
                .zip(&cp.ape[slot * width..(slot + 1) * width])
            {
                *c += a;
            }
        }
        st.pending_kv[li].extend_from_slice(&ckv);
        st.pending_score[li].extend_from_slice(&cscore);
        if st.pending_kv[li].len() / width >= cp.ratio {
            let mut folded = vec![0.0f32; ew];
            if cp.overlap {
                compress_window_overlap(
                    &st.prev_kv[li],
                    &st.prev_score[li],
                    &st.pending_kv[li],
                    &st.pending_score[li],
                    cp.ratio,
                    ew,
                    &mut folded,
                );
                // this window becomes the previous one
                st.prev_kv[li] = std::mem::take(&mut st.pending_kv[li]);
                st.prev_score[li] = std::mem::take(&mut st.pending_score[li]);
            } else {
                compress_window(
                    &st.pending_kv[li],
                    &st.pending_score[li],
                    &cp.ape,
                    cp.ratio,
                    width,
                    &mut folded,
                );
            }
            rms_weighted(&mut folded, &cp.norm, cfg.norm_eps);
            // The compressed entry carries the same rope-tagged tail as a
            // window key, at the position of the window's first token.
            let cpos = pos + 1 - cp.ratio;
            rope_tail(&mut folded, inv_freq, cpos, rd, false);
            // It lives in the attention cache at head width; a compressor
            // whose entry width differs (the indexer's) keeps its own store.
            if ew == hd {
                st.compressed[li].extend_from_slice(&folded);
            } else {
                st.index_kv[li].extend_from_slice(&folded);
            }
            st.pending_kv[li].clear();
            st.pending_score[li].clear();
        }
    }

    st.window[li].extend_from_slice(&kv);
    // The reference keeps the window in a ring of `window_size`; holding the
    // last N in order is the same set, and without this the "window" grows
    // for the whole generation — wrong attention AND unbounded memory.
    let cap = cfg.window * hd;
    if st.window[li].len() > cap {
        let drop = st.window[li].len() - cap;
        st.window[li].drain(..drop);
    }
    let win_len = st.window[li].len() / hd;
    let mut cache: Vec<f32> = st.window[li].clone();
    cache.extend_from_slice(&st.compressed[li]);
    let n_pos = cache.len() / hd;

    // Index list: every window position, plus whatever the indexer picked
    // (or, without an indexer, every compressed position).
    let mut idxs: Vec<usize> = (0..win_len).collect();
    if !st.compressed[li].is_empty() {
        let n_comp = st.compressed[li].len() / hd;
        match &l.indexer {
            Some(ix) => {
                // The indexer scores from the SHARED LoRA output through
                // its own wq_b — not from attention's queries — and its
                // per-head weights are a projection of the hidden state,
                // scaled by head_dim^-0.5 * n_heads^-0.5 as the reference
                // folds into `weights_proj`'s output.
                let ih = ix.weights_proj.rows();
                let idim = ix.wq_b.rows() / ih.max(1);
                let mut qi = vec![0.0f32; ix.wq_b.rows()];
                ix.wq_b.matvec(&qr, &mut qi, pool);
                for h in 0..ih {
                    rope_tail(&mut qi[h * idim..(h + 1) * idim], inv_freq, pos, rd, false);
                }
                let mut hw = vec![0.0f32; ih];
                ix.weights_proj.matvec(hidden, &mut hw, pool);
                let sc_factor = (idim as f32).powf(-0.5) * (ih as f32).powf(-0.5);
                for w in hw.iter_mut() {
                    *w *= sc_factor;
                }
                let n_ix = st.index_kv[li].len() / idim.max(1);
                let mut sc = Vec::new();
                index_scores(
                    &qi,
                    &st.index_kv[li],
                    &hw,
                    ih,
                    idim,
                    n_ix.min(n_comp),
                    n_ix.min(n_comp),
                    &mut sc,
                );
                let mut picked = Vec::new();
                top_k_positions(&sc, cfg.index_topk, &mut picked);
                idxs.extend(picked.into_iter().map(|p| win_len + p));
            }
            None => idxs.extend((0..n_comp).map(|p| win_len + p)),
        }
    }
    debug_assert!(idxs.iter().all(|&p| p < n_pos));

    // ── sparse attention per head, then the inverse rope ──
    let scale = (hd as f32).powf(-0.5);
    let mut attn = vec![0.0f32; cfg.n_heads * hd];
    for h in 0..cfg.n_heads {
        let qh = &q[h * hd..(h + 1) * hd];
        let mut oh = vec![0.0f32; hd];
        sparse_attend(qh, &cache, &idxs, l.attn_sink[h], scale, hd, &mut oh);
        rope_tail(&mut oh, inv_freq, pos, rd, true);
        attn[h * hd..(h + 1) * hd].copy_from_slice(&oh);
    }

    // ── grouped low-rank output ──
    // Read the two blocks through the quantized readers. Materializing them
    // here instead costs ~270 MB of dequantization per layer per token on
    // the release checkpoint (wo_a and wo_b are 33M weights each), which is
    // the difference between decoding and not.
    o_project(
        &attn,
        &|r, x, sc| l.wo_a.row_dot(r, x, sc),
        l.wo_a.cols(),
        &|mid, dst| l.wo_b.matvec(mid, dst, pool),
        cfg.o_groups,
        cfg.o_lora_rank,
        pool,
        out,
    );
}

/// RMSNorm with a learned weight, in place.
pub fn rms_weighted(v: &mut [f32], w: &[f32], eps: f32) {
    let ms = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for (x, g) in v.iter_mut().zip(w) {
        *x = *x * inv * g;
    }
}

/// The MoE half of a block: route, run the chosen experts plus the shared
/// one, and sum. `token_id` is only read on the hash layers.
pub fn moe_step(
    hidden: &[f32],
    l: &Dsv4Layer,
    cfg: &Dsv4Cfg,
    token_id: u32,
    pool: Option<&crate::pool::Pool>,
    out: &mut [f32],
) {
    let mut logits = vec![0.0f32; cfg.n_routed_experts];
    l.gate.matvec(hidden, &mut logits, pool);
    let (mut idx, mut w) = (Vec::new(), Vec::new());
    route(
        &logits,
        l.gate_bias.as_deref(),
        cfg.top_k,
        cfg.route_scale,
        l.tid2eid
            .as_ref()
            .map(|tbl| hash_route(tbl, cfg.vocab, cfg.top_k, token_id))
            .as_deref(),
        &mut idx,
        &mut w,
    );
    out.fill(0.0);
    let mut acc = vec![0.0f32; cfg.dim];
    for (e, &ei) in idx.iter().enumerate() {
        let Some(exp) = l.experts.get(ei) else { continue };
        run_expert(hidden, exp, cfg, w.get(e).copied().unwrap_or(0.0), pool, &mut acc);
        for (o, a) in out.iter_mut().zip(&acc) {
            *o += a;
        }
    }
    // The shared expert always runs, at weight 1.
    run_expert(hidden, &l.shared, cfg, 1.0, pool, &mut acc);
    for (o, a) in out.iter_mut().zip(&acc) {
        *o += a;
    }
}

/// The routed and shared experts both come through here, so the clamp and
/// the weight folding have exactly one implementation — `expert_swiglu`.
fn run_expert(
    x: &[f32],
    e: &Dsv4Expert,
    cfg: &Dsv4Cfg,
    weight: f32,
    pool: Option<&crate::pool::Pool>,
    out: &mut [f32],
) {
    expert_swiglu(
        x,
        &|src, dst| e.w1.matvec(src, dst, pool),
        &|src, dst| e.w3.matvec(src, dst, pool),
        &|src, dst| e.w2.matvec(src, dst, pool),
        cfg.moe_inter,
        weight,
        cfg.swiglu_limit,
        out,
    );
}

/// One token through the whole stack.
///
/// The hidden state is `hc_mult` copies of a `dim`-vector from the very
/// first line to the very last: the embedding is replicated, every layer
/// folds/expands around its two halves, and only `hc_head_fold` collapses
/// it before the output norm and the head. There is no point in this
/// function where an ordinary residual would fit.
#[allow(clippy::too_many_arguments)]
pub fn forward_token(
    g: &Dsv4Globals,
    layers: &[Dsv4Layer],
    cfg: &Dsv4Cfg,
    st: &mut Dsv4State,
    token_id: u32,
    inv_freq: &[f32],
    pool: Option<&crate::pool::Pool>,
    logits: &mut Vec<f32>,
) {
    let (hc, dim) = (cfg.hc_mult, cfg.dim);

    // Embedding, replicated into the copies.
    let mut emb = vec![0.0f32; dim];
    g.embed.row_f32(token_id as usize, &mut emb);
    let mut state = vec![0.0f32; hc * dim];
    for j in 0..hc {
        state[j * dim..(j + 1) * dim].copy_from_slice(&emb);
    }

    let mut scratch = HcScratch::new(cfg);
    for (li, l) in layers.iter().enumerate() {
        // attention half
        hc_block(
            &mut state,
            &l.hc_attn_fn,
            &l.hc_attn_scale,
            &l.hc_attn_base,
            &l.attn_norm,
            cfg,
            &mut scratch,
            |folded, out| attention_step(folded, l, cfg, st, li, inv_freq, pool, out),
        );
        // FFN half
        hc_block(
            &mut state,
            &l.hc_ffn_fn,
            &l.hc_ffn_scale,
            &l.hc_ffn_base,
            &l.ffn_norm,
            cfg,
            &mut scratch,
            |folded, out| moe_step(folded, l, cfg, token_id, pool, out),
        );
    }
    st.pos += 1;

    // Collapse the copies, normalize, project to the vocabulary.
    let mut h = vec![0.0f32; dim];
    hc_head_fold(&state, &g.hc_head_fn, g.hc_head_scale, &g.hc_head_base, cfg, &mut h);
    rms_weighted(&mut h, &g.norm, cfg.norm_eps);
    logits.clear();
    logits.resize(g.head.rows(), 0.0);
    g.head.matvec(&h, logits, pool);
}

/// Build the runtime weights from a converted `.cmf`.
///
/// Names are the converter's output (see `canon_name`'s deepseek_v4 arm):
/// attention keeps DeepSeek's own spelling under `self_attn.`, the MoE is
/// rewritten into the layout every other MoE here uses, and the hyper-
/// connection tensors ride under the layer prefix.
pub fn load(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    cfg: &Dsv4Cfg,
    n_layers: usize,
) -> Result<(Dsv4Globals, Vec<Dsv4Layer>), String> {
    let q = |name: &str| -> Result<crate::qtensor::QTensor, String> {
        crate::qtensor::QTensor::from_model(model, name)
    };
    let f = |name: &str| -> Result<Vec<f32>, String> {
        let t = q(name)?;
        let (r, c) = (t.rows(), t.cols());
        let mut v = vec![0.0f32; r * c];
        for i in 0..r {
            t.row_f32(i, &mut v[i * c..(i + 1) * c]);
        }
        Ok(v)
    };
    let opt_f = |name: &str| -> Option<Vec<f32>> { f(name).ok() };

    let globals = Dsv4Globals {
        embed: q("model.embed_tokens.weight")?,
        norm: f("model.norm.weight")?,
        head: q("lm_head.weight")?,
        hc_head_fn: f("model.hc_head_fn")?,
        hc_head_base: f("model.hc_head_base")?,
        hc_head_scale: *f("model.hc_head_scale")?
            .first()
            .ok_or("dsv4: empty hc_head_scale")?,
    };

    let mut layers = Vec::with_capacity(n_layers);
    for li in 0..n_layers {
        let p = format!("model.layers.{li}");
        let scale3 = |name: &str| -> Result<[f32; 3], String> {
            let v = f(name)?;
            if v.len() < 3 {
                return Err(format!("{name}: expected 3 scales, got {}", v.len()));
            }
            Ok([v[0], v[1], v[2]])
        };
        // The compressor exists on every layer whose ratio is non-zero;
        // its presence in the file is the only signal we need.
        let compressor = match q(&format!("{p}.self_attn.compressor.wkv.weight")) {
            Ok(wkv) => {
                let ape = f(&format!("{p}.self_attn.compressor.ape"))?;
                // ape is [ratio, coff*head_dim]; coff is 2 when the windows
                // overlap, which the release does at ratio 4.
                let width = wkv.rows();
                let ratio = (ape.len() / width.max(1)).max(1);
                Some(Dsv4Compressor {
                    wkv,
                    wgate: q(&format!("{p}.self_attn.compressor.wgate.weight"))?,
                    norm: f(&format!("{p}.self_attn.compressor.norm.weight"))?,
                    ape,
                    ratio,
                    overlap: ratio == 4,
                })
            }
            Err(_) => None,
        };
        let indexer = match q(&format!("{p}.self_attn.indexer.wq_b.weight")) {
            Ok(wq_b) => {
                let ape = f(&format!("{p}.self_attn.indexer.compressor.ape"))?;
                let cwkv = q(&format!("{p}.self_attn.indexer.compressor.wkv.weight"))?;
                let width = cwkv.rows();
                let ratio = (ape.len() / width.max(1)).max(1);
                Some(Dsv4Indexer {
                    wq_b,
                    weights_proj: q(&format!("{p}.self_attn.indexer.weights_proj.weight"))?,
                    compressor: Dsv4Compressor {
                        wkv: cwkv,
                        wgate: q(&format!("{p}.self_attn.indexer.compressor.wgate.weight"))?,
                        norm: f(&format!("{p}.self_attn.indexer.compressor.norm.weight"))?,
                        ape,
                        ratio,
                        overlap: ratio == 4,
                    },
                })
            }
            Err(_) => None,
        };

        let mut experts = Vec::with_capacity(cfg.n_routed_experts);
        for e in 0..cfg.n_routed_experts {
            let ep = format!("{p}.mlp.experts.{e}");
            experts.push(Dsv4Expert {
                w1: q(&format!("{ep}.gate_proj.weight"))?,
                w2: q(&format!("{ep}.down_proj.weight"))?,
                w3: q(&format!("{ep}.up_proj.weight"))?,
            });
        }

        layers.push(Dsv4Layer {
            attn_norm: f(&format!("{p}.input_layernorm.weight"))?,
            ffn_norm: f(&format!("{p}.post_attention_layernorm.weight"))?,
            wq_a: q(&format!("{p}.self_attn.wq_a.weight"))?,
            q_norm: f(&format!("{p}.self_attn.q_norm.weight"))?,
            wq_b: q(&format!("{p}.self_attn.wq_b.weight"))?,
            wkv: q(&format!("{p}.self_attn.wkv.weight"))?,
            kv_norm: f(&format!("{p}.self_attn.kv_norm.weight"))?,
            wo_a: q(&format!("{p}.self_attn.wo_a.weight"))?,
            wo_b: q(&format!("{p}.self_attn.wo_b.weight"))?,
            attn_sink: f(&format!("{p}.self_attn.attn_sink"))?,
            compressor,
            indexer,
            hc_attn_fn: f(&format!("{p}.hc_attn_fn"))?,
            hc_attn_base: f(&format!("{p}.hc_attn_base"))?,
            hc_attn_scale: scale3(&format!("{p}.hc_attn_scale"))?,
            hc_ffn_fn: f(&format!("{p}.hc_ffn_fn"))?,
            hc_ffn_base: f(&format!("{p}.hc_ffn_base"))?,
            hc_ffn_scale: scale3(&format!("{p}.hc_ffn_scale"))?,
            gate: q(&format!("{p}.mlp.gate.weight"))?,
            // The bias is absent exactly on the hash layers, and the table
            // is present exactly there — the file itself says which is which.
            gate_bias: opt_f(&format!("{p}.mlp.expert_bias")),
            tid2eid: opt_f(&format!("{p}.mlp.tid2eid")),
            experts,
            shared: Dsv4Expert {
                w1: q(&format!("{p}.mlp.shared_expert.gate_proj.weight"))?,
                w2: q(&format!("{p}.mlp.shared_expert.down_proj.weight"))?,
                w3: q(&format!("{p}.mlp.shared_expert.up_proj.weight"))?,
            },
        });
    }
    Ok((globals, layers))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A whole model, small enough to reason about: 2 layers, 4 heads, 8
    // experts. Weights are deterministic and tiny, which is the point —
    // this test is about shapes, indexing and cache bookkeeping, the things
    // that a 138 GB file would surface only after an hour of loading.
    fn toy() -> (Dsv4Globals, Vec<Dsv4Layer>, Dsv4Cfg) {
        use crate::qtensor::QTensor;
        let cfg = Dsv4Cfg {
            dim: 32,
            n_heads: 4,
            head_dim: 8,
            rope_head_dim: 4,
            q_lora_rank: 16,
            o_lora_rank: 16,
            o_groups: 2,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
            norm_eps: 1e-6,
            n_routed_experts: 8,
            top_k: 2,
            moe_inter: 16,
            route_scale: 1.0,
            swiglu_limit: 10.0,
            window: 6,
            index_topk: 8,
            vocab: 24,
        };
        // Deterministic pseudo-random in a narrow band: big enough to move
        // the state, small enough that nothing saturates.
        let w = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 7 + seed * 13) % 101) as f32 / 101.0 - 0.5) * 0.3)
                .collect()
        };
        let t = |rows: usize, cols: usize, seed: usize| QTensor::from_f32(w(rows * cols, seed), rows, cols);
        let ones = |n: usize| vec![1.0f32; n];

        let (dim, hc) = (cfg.dim, cfg.hc_mult);
        // q is n_heads*head_dim wide and kv is one head wide; rope rides the
        // tail of each rather than widening anything.
        let q_width = cfg.n_heads * cfg.head_dim;
        let kv_width = cfg.head_dim;
        let o_per_group = q_width / cfg.o_groups;
        let mut layers = Vec::new();
        for li in 0..2 {
            let experts: Vec<Dsv4Expert> = (0..cfg.n_routed_experts)
                .map(|e| Dsv4Expert {
                    w1: t(cfg.moe_inter, dim, 40 + e + li * 8),
                    w2: t(dim, cfg.moe_inter, 60 + e + li * 8),
                    w3: t(cfg.moe_inter, dim, 80 + e + li * 8),
                })
                .collect();
            // Layer 0 is a hash layer (table-routed); layer 1 routes normally
            // and carries the compressor — both paths get exercised.
            layers.push(Dsv4Layer {
                attn_norm: ones(dim),
                ffn_norm: ones(dim),
                wq_a: t(cfg.q_lora_rank, dim, 1 + li),
                q_norm: ones(cfg.q_lora_rank),
                wq_b: t(q_width, cfg.q_lora_rank, 3 + li),
                wkv: t(kv_width, dim, 5 + li),
                kv_norm: ones(kv_width),
                wo_a: t(cfg.o_groups * cfg.o_lora_rank, o_per_group, 7 + li),
                wo_b: t(dim, cfg.o_groups * cfg.o_lora_rank, 9 + li),
                attn_sink: vec![0.1; cfg.n_heads],
                // Layer 1 carries the OVERLAPPING compressor, as the release
                // does at ratio 4: the projection is twice the entry width.
                compressor: if li == 1 {
                    Some(Dsv4Compressor {
                        wkv: t(2 * kv_width, dim, 11),
                        wgate: t(2 * kv_width, dim, 13),
                        norm: ones(kv_width),
                        ape: vec![0.01; 4 * 2 * kv_width],
                        ratio: 4,
                        overlap: true,
                    })
                } else {
                    None
                },
                indexer: None,
                hc_attn_fn: w((2 + hc) * hc * hc * dim, 15 + li),
                hc_attn_base: w((2 + hc) * hc, 17 + li),
                hc_attn_scale: [1.0, 1.0, 1.0],
                hc_ffn_fn: w((2 + hc) * hc * hc * dim, 19 + li),
                hc_ffn_base: w((2 + hc) * hc, 21 + li),
                hc_ffn_scale: [1.0, 1.0, 1.0],
                gate: t(cfg.n_routed_experts, dim, 23 + li),
                gate_bias: if li == 1 {
                    Some(vec![0.0; cfg.n_routed_experts])
                } else {
                    None
                },
                tid2eid: if li == 0 {
                    Some(
                        (0..cfg.vocab * cfg.top_k)
                            .map(|i| (i % cfg.n_routed_experts) as f32)
                            .collect(),
                    )
                } else {
                    None
                },
                experts,
                shared: Dsv4Expert {
                    w1: t(cfg.moe_inter, dim, 25 + li),
                    w2: t(dim, cfg.moe_inter, 27 + li),
                    w3: t(cfg.moe_inter, dim, 29 + li),
                },
            });
        }
        let g = Dsv4Globals {
            embed: t(cfg.vocab, dim, 31),
            norm: ones(dim),
            head: t(cfg.vocab, dim, 33),
            hc_head_fn: w(hc * hc * dim, 35),
            hc_head_base: w(hc, 37),
            hc_head_scale: 1.0,
        };
        (g, layers, cfg)
    }

    /// The whole stack, decoding a sequence. Every block is on the path:
    /// hyper-connections, the double-LoRA attention with its sink, the KV
    /// compressor firing on its ratio boundary, hash routing on one layer
    /// and score routing on the other.
    #[test]
    fn forward_token_decodes_a_sequence_without_falling_over() {
        let (g, layers, cfg) = toy();
        let mut st = Dsv4State::new(layers.len());
        let inv_freq: Vec<f32> = (0..cfg.rope_head_dim / 2)
            .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / cfg.rope_head_dim as f32))
            .collect();
        let mut logits = Vec::new();

        // Ten tokens: more than twice the compressor's ratio, so the
        // compressed cache is written on a boundary and read afterwards.
        let mut first: Option<Vec<f32>> = None;
        for (step, tok) in [3u32, 7, 1, 9, 4, 2, 8, 5, 6, 0].into_iter().enumerate() {
            forward_token(&g, &layers, &cfg, &mut st, tok, &inv_freq, None, &mut logits);
            assert_eq!(logits.len(), cfg.vocab, "step {step}: logit count");
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "step {step}: non-finite logit — {logits:?}"
            );
            // A model that has collapsed returns the same distribution
            // regardless of input; that is the failure this catches.
            let spread = logits.iter().cloned().fold(f32::MIN, f32::max)
                - logits.iter().cloned().fold(f32::MAX, f32::min);
            assert!(spread > 1e-6, "step {step}: logits are flat ({spread})");
            if step == 0 {
                first = Some(logits.clone());
            }
            assert_eq!(st.pos, step + 1, "position bookkeeping");
        }

        // The cache has to have grown, and the compressor layer must have
        // emitted compressed entries (10 tokens / ratio 4 = 2 windows).
        assert!(!st.window[0].is_empty(), "sliding window never filled");
        // Ten tokens through a window of six: it must have slid, not grown.
        for (li, w) in st.window.iter().enumerate() {
            assert!(
                w.len() / cfg.head_dim <= cfg.window,
                "layer {li}: window holds {} positions, cap is {}",
                w.len() / cfg.head_dim,
                cfg.window
            );
        }
        assert!(
            !st.compressed[1].is_empty(),
            "compressor layer produced no compressed KV in 10 tokens"
        );
        // Ten tokens at ratio 4 fold twice, and the entries must be one head
        // wide — the overlapping projection is 2x that, so a width mistake
        // shows up here rather than as quiet nonsense.
        assert_eq!(
            st.compressed[1].len() / cfg.head_dim,
            2,
            "expected two folds in ten tokens at ratio 4"
        );
        assert!(
            !st.prev_kv[1].is_empty(),
            "the overlapping compressor never kept a previous window"
        );

        // Context must matter: the same token at position 0 of a fresh state
        // and at the end of a filled one cannot give identical logits.
        let mut fresh = Dsv4State::new(layers.len());
        let mut relogits = Vec::new();
        forward_token(&g, &layers, &cfg, &mut fresh, 3, &inv_freq, None, &mut relogits);
        assert_eq!(
            relogits,
            first.unwrap(),
            "the same token from a fresh state must reproduce exactly"
        );
    }

    /// The reference clamps `up` on both sides but `gate` only from above.
    /// Getting that symmetric would quietly change every expert's output on
    /// the tokens that saturate, which is the hardest kind of bug to see.
    #[test]
    fn swiglu_limit_clamps_up_both_ways_and_gate_only_from_above() {
        let inter = 4;
        // gate = [-50, 50, 1, -1], up = [50, -50, 1, -1]
        let gate_src = [-50.0f32, 50.0, 1.0, -1.0];
        let up_src = [50.0f32, -50.0, 1.0, -1.0];
        let limit = 10.0f32;
        let mut got = vec![0.0f32; inter];
        expert_swiglu(
            &[0.0],
            &|_, d| d.copy_from_slice(&gate_src),
            &|_, d| d.copy_from_slice(&up_src),
            &|src, d| d.copy_from_slice(src),
            inter,
            1.0,
            limit,
            &mut got,
        );
        let silu = |g: f32| g / (1.0 + (-g).exp());
        // gate: only the +50 is cut, the -50 rides through silu untouched.
        let want = [
            silu(-50.0) * limit,
            silu(limit) * -limit,
            silu(1.0) * 1.0,
            silu(-1.0) * -1.0,
        ];
        for (i, w) in want.iter().enumerate() {
            assert!(
                (got[i] - w).abs() < 1e-5,
                "lane {i}: got {} want {w}",
                got[i]
            );
        }
        // And with the clamp off nothing is touched.
        let mut raw = vec![0.0f32; inter];
        expert_swiglu(
            &[0.0],
            &|_, d| d.copy_from_slice(&gate_src),
            &|_, d| d.copy_from_slice(&up_src),
            &|src, d| d.copy_from_slice(src),
            inter,
            1.0,
            0.0,
            &mut raw,
        );
        assert!((raw[1] - silu(50.0) * -50.0).abs() < 1e-3, "limit 0 must not clamp");
    }

    /// The grouped projection writes its intermediate from several threads
    /// at once. Disjoint indices are the whole argument for that being safe,
    /// so the pooled result has to equal the serial one exactly — a race
    /// here would show up as occasional wrong tokens, not as a crash.
    #[test]
    fn grouped_projection_is_identical_with_and_without_a_pool() {
        let (groups, lora, per_group, dim) = (4usize, 128usize, 64usize, 32usize);
        let attn: Vec<f32> = (0..groups * per_group)
            .map(|i| ((i * 13) as f32 * 0.021).sin())
            .collect();
        let wo_a: Vec<f32> = (0..groups * lora * per_group)
            .map(|i| ((i * 7) as f32 * 0.011).cos())
            .collect();
        let wo_b: Vec<f32> = (0..dim * groups * lora)
            .map(|i| ((i * 5) as f32 * 0.009).sin())
            .collect();
        let row = |r: usize, x: &[f32], _sc: &mut [f32]| -> f32 {
            wo_a[r * per_group..(r + 1) * per_group]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        };
        let project = |mid: &[f32], dst: &mut [f32]| {
            for (d, o) in dst.iter_mut().enumerate() {
                *o = wo_b[d * mid.len()..(d + 1) * mid.len()]
                    .iter()
                    .zip(mid)
                    .map(|(a, b)| a * b)
                    .sum();
            }
        };

        let mut serial = vec![0.0f32; dim];
        o_project(&attn, &row, per_group, &project, groups, lora, None, &mut serial);

        let pool = crate::pool::Pool::new(4);
        let mut pooled = vec![0.0f32; dim];
        o_project(
            &attn,
            &row,
            per_group,
            &project,
            groups,
            lora,
            Some(&pool),
            &mut pooled,
        );
        assert_eq!(serial, pooled, "the pooled projection diverged");
        assert!(serial.iter().any(|v| v.abs() > 1e-6), "test data is degenerate");
    }

    /// The overlapping compressor folds 2*ratio slots, not ratio: the
    /// previous window contributes its first half of dimensions and the
    /// current one its second half. Treating it as a plain compressor makes
    /// the entry twice as wide as the cache expects, which lands the whole
    /// thing in the wrong store rather than raising anything.
    #[test]
    fn overlapping_compressor_folds_both_windows() {
        let (ratio, d) = (2usize, 3usize);
        // Current window: two tokens, 2*d wide each. Second half is what the
        // current window contributes.
        let cur_kv: Vec<f32> = vec![
            1.0, 1.0, 1.0, /*|*/ 10.0, 20.0, 30.0, // token 0
            2.0, 2.0, 2.0, /*|*/ 40.0, 50.0, 60.0, // token 1
        ];
        // Make the current window's second-half scores dominate everywhere.
        let cur_sc: Vec<f32> = vec![
            0.0, 0.0, 0.0, /*|*/ 0.0, 0.0, 100.0, //
            0.0, 0.0, 0.0, /*|*/ 100.0, 100.0, 0.0,
        ];
        // Previous window: its FIRST half is what it contributes.
        let prev_kv: Vec<f32> = vec![
            7.0, 8.0, 9.0, /*|*/ 0.0, 0.0, 0.0, //
            5.0, 6.0, 7.0, /*|*/ 0.0, 0.0, 0.0,
        ];
        let prev_sc = vec![0.0f32; ratio * 2 * d];

        let mut out = vec![0.0f32; d];
        compress_window_overlap(&prev_kv, &prev_sc, &cur_kv, &cur_sc, ratio, d, &mut out);
        // dim 0 and 1: token 1's second half wins (score 100)
        assert!((out[0] - 40.0).abs() < 1e-3, "dim0 = {}", out[0]);
        assert!((out[1] - 50.0).abs() < 1e-3, "dim1 = {}", out[1]);
        // dim 2: token 0's second half wins
        assert!((out[2] - 30.0).abs() < 1e-3, "dim2 = {}", out[2]);

        // With no previous window the fold still works and uses only the
        // current one — this is the very first window of a generation.
        let mut first = vec![0.0f32; d];
        compress_window_overlap(&[], &[], &cur_kv, &cur_sc, ratio, d, &mut first);
        assert!(first.iter().all(|v| v.is_finite()), "first window: {first:?}");
        assert!((first[0] - 40.0).abs() < 1e-3, "first dim0 = {}", first[0]);

        // And a previous window with real scores does pull the result.
        let mut both = vec![0.0f32; d];
        let strong_prev = vec![100.0f32; ratio * 2 * d];
        compress_window_overlap(&prev_kv, &strong_prev, &cur_kv, &cur_sc, ratio, d, &mut both);
        assert!(
            (both[0] - 40.0).abs() > 1.0,
            "a scored previous window must move the fold, got {}",
            both[0]
        );
    }

    /// Numerical parity with the reference. The vectors below come from
    /// running `kernel.py::hc_split_sinkhorn`'s own formula on a fixed
    /// input; matching them pins the exponent order, the eps placement and
    /// the off-by-one in the iteration count all at once — a property test
    /// alone would pass with any of those wrong.
    #[test]
    fn sinkhorn_matches_the_reference_numbers() {
        let hc = 4;
        let mixes: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let base: Vec<f32> = (0..24).map(|i| (i as f32 * 0.11).cos()).collect();
        let (mut pre, mut post, mut comb) = (vec![0.0; hc], vec![0.0; hc], vec![0.0; hc * hc]);
        hc_split_sinkhorn(
            &mixes, &[1.0, 1.0, 1.0], &base, hc, 20, 1e-6, &mut pre, &mut post, &mut comb,
        );
        let want_pre = [0.7310596, 0.8888268, 0.9525191, 0.97424865];
        let want_post = [1.9600224, 1.9534285, 1.9201256, 1.8160983];
        let want_comb = [
            0.5996052, 0.28253591, 0.09218107, 0.025676856,
            0.17564717, 0.22228767, 0.27174541, 0.33031881,
            0.029528176, 0.12206022, 0.32619134, 0.5222193,
            0.19521846, 0.37311527, 0.30988118, 0.12178412,
        ];
        for (i, w) in want_pre.iter().enumerate() {
            assert!((pre[i] - w).abs() < 1e-5, "pre[{i}]: {} vs {w}", pre[i]);
        }
        for (i, w) in want_post.iter().enumerate() {
            assert!((post[i] - w).abs() < 1e-5, "post[{i}]: {} vs {w}", post[i]);
        }
        for (i, w) in want_comb.iter().enumerate() {
            assert!((comb[i] - w).abs() < 1e-4, "comb[{i}]: {} vs {w}", comb[i]);
        }
    }

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
        route(&scores, Some(&bias), 2, 1.5, None, &mut idx, &mut w);
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

    /// The block cycle must leave the state's SHAPE intact (hc copies in,
    /// hc copies out) and must actually route the block's output back in:
    /// a block that writes a constant has to move every copy.
    #[test]
    fn hc_block_preserves_the_copy_structure_and_applies_the_block() {
        let cfg = Dsv4Cfg {
            dim: 4,
            n_heads: 1,
            head_dim: 4,
            rope_head_dim: 2,
            q_lora_rank: 4,
            o_lora_rank: 2,
            o_groups: 1,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
            norm_eps: 1e-6,
            n_routed_experts: 2,
            top_k: 1,
            moe_inter: 4,
            route_scale: 1.0,
            swiglu_limit: 10.0,
            window: 128,
            index_topk: 4,
            vocab: 8,
        };
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        let mix_hc = (2 + hc) * hc;
        let hc_fn: Vec<f32> = (0..mix_hc * hc * dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
            .collect();
        let hc_base: Vec<f32> = (0..mix_hc).map(|i| (i as f32 * 0.2).sin()).collect();
        let norm_w = vec![1.0f32; dim];
        let mut state: Vec<f32> = (0..hc * dim).map(|i| (i as f32 * 0.3).cos()).collect();
        let before = state.clone();
        let mut scratch = HcScratch::new(&cfg);
        hc_block(
            &mut state,
            &hc_fn,
            &[1.0, 1.0, 1.0],
            &hc_base,
            &norm_w,
            &cfg,
            &mut scratch,
            |_folded, out| out.iter_mut().for_each(|o| *o = 1.0),
        );
        assert_eq!(state.len(), before.len(), "copy structure must survive");
        assert!(state.iter().all(|v| v.is_finite()), "{state:?}");
        assert!(
            state.iter().zip(&before).any(|(a, b)| (a - b).abs() > 1e-4),
            "the block's output has to reach the state"
        );
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

    /// On a hash layer the reference gathers the scores AT THE TABLE's
    /// experts. Choosing top-k first and swapping the indices afterwards
    /// leaves every weight attached to a different expert than the one it
    /// scales — silently, since both lists are the right length.
    #[test]
    fn hash_layers_weight_the_experts_the_table_names() {
        // Expert 3 scores highest, expert 0 lowest; the table names 0 and 1.
        let scores = [0.1f32, 0.4, 0.2, 5.0];
        let table = vec![0.0f32, 1.0];
        let idx_forced = hash_route(&table, 1, 2, 0);
        assert_eq!(idx_forced, vec![0, 1]);

        let (mut idx, mut w) = (Vec::new(), Vec::new());
        route(&scores, None, 2, 1.0, Some(&idx_forced), &mut idx, &mut w);
        assert_eq!(idx, vec![0, 1], "the table must decide the experts");

        // The weights must be the table experts' own scores, normalized.
        let sp = |x: f32| (1.0 + x.exp()).ln().sqrt();
        let (s0, s1) = (sp(scores[0]), sp(scores[1]));
        let tot = s0 + s1;
        assert!((w[0] - s0 / tot).abs() < 1e-6, "w[0]={} want {}", w[0], s0 / tot);
        assert!((w[1] - s1 / tot).abs() < 1e-6, "w[1]={} want {}", w[1], s1 / tot);

        // And the top-k path is untouched: expert 3 still wins there.
        let (mut idx2, mut w2) = (Vec::new(), Vec::new());
        route(&scores, None, 2, 1.0, None, &mut idx2, &mut w2);
        assert_eq!(idx2[0], 3, "without a table the highest score still wins");
    }
}
