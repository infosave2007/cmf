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
pub fn hc_mixes(
    x_flat: &[f32],
    hc_fn: &[f32],
    mix_hc: usize,
    eps: f32,
    pool: Option<&crate::pool::Pool>,
    out: &mut [f32],
) {
    let n = x_flat.len();
    debug_assert_eq!(hc_fn.len(), mix_hc * n);
    debug_assert_eq!(out.len(), mix_hc);
    let ms = x_flat.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rsqrt = 1.0 / (ms + eps).sqrt();
    // A dense f32 matvec of mix_hc rows over hc*dim — 1.6 MB read per call on
    // the release, and TWO calls per layer, so 135 MB a token. It ran on one
    // thread and cost more than the whole attention block.
    match pool {
        Some(p) if n >= 4096 => {
            let addr = crate::pool::SendMut::new(out.as_mut_ptr());
            p.run_rows(mix_hc, &|start, end| {
                for i in start..end {
                    let row = &hc_fn[i * n..(i + 1) * n];
                    let v = row.iter().zip(x_flat).map(|(a, b)| a * b).sum::<f32>() * rsqrt;
                    unsafe { *addr.at(i) = v };
                }
            });
        }
        _ => {
            for (i, o) in out.iter_mut().enumerate() {
                let row = &hc_fn[i * n..(i + 1) * n];
                *o = row.iter().zip(x_flat).map(|(a, b)| a * b).sum::<f32>() * rsqrt;
            }
        }
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
    mask: Option<&[bool]>,
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
            if let Some(m) = mask {
                for (i, s) in shifted.iter_mut().enumerate() {
                    if !m.get(i).copied().unwrap_or(true) {
                        *s = f32::NEG_INFINITY;
                    }
                }
            }
            for _ in 0..top_k.min(n) {
                let mut best = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for (i, &v) in shifted.iter().enumerate() {
                    if v > bv {
                        bv = v;
                        best = i;
                    }
                }
                if !bv.is_finite() {
                    break;
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
    debug_assert!(
        rd <= n && rd % 2 == 0,
        "rope tail {rd} wider than the vector {n}"
    );
    // A tail wider than the vector is a configuration mistake, and `n - rd`
    // would wrap into an index in the billions rather than say so.
    let rd = rd.min(n) & !1;
    let base = n - rd;
    // ADJACENT pairs, not halves. The reference forms its complex numbers
    // with `unflatten(-1, (-1, 2))` + `view_as_complex`, i.e. (x0,x1),
    // (x2,x3), … — the interleaved convention. Half-split pairing agrees
    // with it exactly at position 0, where the rotation is the identity,
    // and disagrees everywhere else. That is why short answers came out
    // right and everything longer drifted, repeated itself and could not
    // count: every position past the first was rotated into the wrong
    // basis.
    for i in 0..rd / 2 {
        let theta = pos as f32 * inv_freq[i];
        let (s, c) = (theta.sin(), theta.cos());
        let s = if inverse { -s } else { s };
        let a = v[base + 2 * i];
        let b = v[base + 2 * i + 1];
        v[base + 2 * i] = a * c - b * s;
        v[base + 2 * i + 1] = a * s + b * c;
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
    if std::env::var("CMF_ATTN_DEBUG").is_ok() {
        eprintln!(
            "    [порт] позиций={} score={:?} sink={sink:.4} denom={denom:.4} |q|={:.3}",
            idxs.iter().filter(|&&p| p != usize::MAX).count(),
            scores
                .iter()
                .map(|x| (x * 10000.0).round() / 10000.0)
                .collect::<Vec<_>>(),
            q.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
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
#[allow(clippy::too_many_arguments)]
pub fn index_scores(
    q_heads: &[f32],
    kv: &[f32],
    head_weights: &[f32],
    n_heads: usize,
    head_dim: usize,
    n_pos: usize,
    causal_limit: usize,
    pool: Option<&crate::pool::Pool>,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.resize(n_pos, 0.0);
    let score_at = |t: usize| -> f32 {
        if t >= causal_limit {
            return f32::NEG_INFINITY;
        }
        let k = &kv[t * head_dim..(t + 1) * head_dim];
        let mut acc = 0.0;
        for h in 0..n_heads {
            let q = &q_heads[h * head_dim..(h + 1) * head_dim];
            let dot: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
            // relu BEFORE weighting: a head votes for a position or abstains
            acc += dot.max(0.0) * head_weights[h];
        }
        acc
    };
    // Positions are independent, and their number grows with the context —
    // this was the one loop in the attention step still walking the whole
    // compressed axis on one thread.
    match pool {
        Some(p) if n_pos >= 64 => {
            let addr = crate::pool::SendMut::new(out.as_mut_ptr());
            p.run_rows(n_pos, &|start, end| {
                for t in start..end {
                    unsafe { *addr.at(t) = score_at(t) };
                }
            });
        }
        _ => {
            for (t, o) in out.iter_mut().enumerate() {
                *o = score_at(t);
            }
        }
    }
}

/// Top-`k` positions by score, ties broken by the lower index so the choice
/// is deterministic across backends. Masked slots (-inf) never win, and a
/// short history simply returns fewer than `k`.
pub fn top_k_positions(scores: &[f32], k: usize, out: &mut Vec<usize>) {
    out.clear();
    // When k reaches the whole list there is nothing to choose: every finite
    // position wins, and they come out in index order anyway. The general
    // path is k rounds of argmax — O(k·n) — and at index_topk = 512 against a
    // compressed axis that is still shorter than that, it was doing 160k
    // comparisons a layer to arrive at "all of them". This grows with the
    // context, which is exactly when it hurts.
    if k >= scores.len() {
        out.extend(
            scores
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(i, _)| i),
        );
        return;
    }
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
#[allow(clippy::too_many_arguments)]
pub fn hc_block<F: FnMut(&[f32], &mut [f32])>(
    state: &mut [f32],
    hc_fn: &[f32],
    hc_scale: &[f32; 3],
    hc_base: &[f32],
    norm_w: &[f32],
    cfg: &Dsv4Cfg,
    scratch: &mut HcScratch,
    pool: Option<&crate::pool::Pool>,
    mut block: F,
) {
    let (hc, dim) = (cfg.hc_mult, cfg.dim);
    let mix_hc = (2 + hc) * hc;
    hc_mixes(state, hc_fn, mix_hc, cfg.norm_eps, pool, &mut scratch.mixes);
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
    pool: Option<&crate::pool::Pool>,
    out: &mut [f32],
) {
    let (hc, dim) = (cfg.hc_mult, cfg.dim);
    let mut mixes = vec![0.0f32; hc];
    hc_mixes(state, hc_fn, hc, cfg.norm_eps, pool, &mut mixes);
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
    /// Task-conditional restriction over the routed experts
    /// (`CMF_MOE_MASK` + `CMF_MOE_MASK_COVER`): `false` experts are not
    /// selectable and the weights renormalize over what remains. `None` on
    /// the hash layers — their table names specific experts, so masking
    /// there would silently reroute rather than restrict.
    pub mask: Option<Vec<bool>>,
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
    /// RoPE frequencies for the layers that carry a KV compressor: base
    /// `compress_rope_theta` (160 000 in the release) WITH YaRN.
    pub inv_freq_compress: Vec<f32>,
    /// …and for the pure sliding-window layers: base `rope_theta` (10 000)
    /// with YaRN OFF. The reference picks per layer:
    ///   if compress_ratio { original_seq_len, compress_rope_theta }
    ///   else              { 0, rope_theta }   // "disable YaRN"
    /// One shared table gets both groups wrong — the model still retrieves
    /// facts, because attention still attends, but every position is rotated
    /// by the wrong angle, so it repeats itself and cannot count.
    pub inv_freq_window: Vec<f32>,
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
    /// The indexer's compressor runs alongside the attention one and keeps
    /// its own window — same shape, different width and different weights.
    pub pending_ix_kv: Vec<Vec<f32>>,
    pub pending_ix_score: Vec<Vec<f32>>,
    pub prev_ix_kv: Vec<Vec<f32>>,
    pub prev_ix_score: Vec<Vec<f32>>,
    pub pos: usize,
    /// Identifies this sequence's caches on the device. A fresh state gets a
    /// fresh id, so a device buffer left over from the previous conversation
    /// can never be read as if it belonged to this one.
    pub kv_id: u64,
    /// When the token graph owns a layer's caches, the CONTENTS live on the
    /// card and only these counts stay here — how much of the window is
    /// filled, and how many compressed entries each cache holds. All three
    /// follow from the position, so keeping them costs nothing and reading
    /// them back would cost a round trip.
    pub dev_filled: Vec<usize>,
    pub dev_n_comp: Vec<usize>,
    pub dev_n_ix: Vec<usize>,
    /// True once this sequence has run a layer on the card with the device
    /// owning its state. The host copies above are stale from then on, so
    /// the CPU path must not be used for that layer again.
    pub dev_owned: bool,
    /// The device-layer set of the FIRST chained token. If it ever differs,
    /// some layer's caches are on the wrong side and the answer would be
    /// quietly wrong — the loop refuses instead.
    pub dev_set: Vec<bool>,
}

impl Dsv4State {
    pub fn new(layers: usize) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self {
            kv_id: NEXT.fetch_add(1, Ordering::Relaxed),
            dev_filled: vec![0; layers],
            dev_n_comp: vec![0; layers],
            dev_n_ix: vec![0; layers],
            dev_owned: false,
            dev_set: Vec::new(),
            window: vec![Vec::new(); layers],
            compressed: vec![Vec::new(); layers],
            index_kv: vec![Vec::new(); layers],
            pending_kv: vec![Vec::new(); layers],
            pending_score: vec![Vec::new(); layers],
            prev_kv: vec![Vec::new(); layers],
            prev_score: vec![Vec::new(); layers],
            pending_ix_kv: vec![Vec::new(); layers],
            pending_ix_score: vec![Vec::new(); layers],
            prev_ix_kv: vec![Vec::new(); layers],
            prev_ix_score: vec![Vec::new(); layers],
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
/// Advance one compressor by a token and return its folded entry when the
/// window closes. Both the attention compressor and the indexer's own run
/// through here — the indexer's was simply never called, so its cache stayed
/// empty and every layer that has an indexer selected ZERO compressed
/// positions, discarding a correctly-built long-range memory.
#[allow(clippy::too_many_arguments)]
fn compressor_step(
    cp: &Dsv4Compressor,
    hidden: &[f32],
    pos: usize,
    rd: usize,
    norm_eps: f32,
    inv_freq: &[f32],
    pool: Option<&crate::pool::Pool>,
    pending_kv: &mut Vec<f32>,
    pending_score: &mut Vec<f32>,
    prev_kv: &mut Vec<f32>,
    prev_score: &mut Vec<f32>,
) -> Option<Vec<f32>> {
    let width = cp.wkv.rows();
    let ew = if cp.overlap { width / 2 } else { width };
    let mut ckv = vec![0.0f32; width];
    let mut cscore = vec![0.0f32; width];
    // Same input, so one dispatch instead of two — and this runs twice a
    // layer (the compressor and the indexer's own), 43 layers a token.
    crate::qtensor::QTensor::matvec_many(
        [&cp.wkv, &cp.wgate],
        hidden,
        [&mut ckv, &mut cscore],
        pool,
    );
    if cp.overlap {
        // The reference biases the score as the token arrives and keeps it
        // biased across the shift, so ape is added ONCE, here.
        let slot = pos % cp.ratio;
        for (c, a) in cscore
            .iter_mut()
            .zip(&cp.ape[slot * width..(slot + 1) * width])
        {
            *c += a;
        }
    }
    pending_kv.extend_from_slice(&ckv);
    pending_score.extend_from_slice(&cscore);
    if pending_kv.len() / width < cp.ratio {
        return None;
    }
    let mut folded = vec![0.0f32; ew];
    if cp.overlap {
        compress_window_overlap(
            prev_kv,
            prev_score,
            pending_kv,
            pending_score,
            cp.ratio,
            ew,
            &mut folded,
        );
        *prev_kv = std::mem::take(pending_kv);
        *prev_score = std::mem::take(pending_score);
    } else {
        compress_window(
            pending_kv,
            pending_score,
            &cp.ape,
            cp.ratio,
            width,
            &mut folded,
        );
    }
    rms_weighted(&mut folded, &cp.norm, norm_eps);
    // The entry carries the same rope-tagged tail as a window key, at the
    // position of the window's first token.
    rope_tail(&mut folded, inv_freq, pos + 1 - cp.ratio, rd, false);
    pending_kv.clear();
    pending_score.clear();
    Some(folded)
}

/// `CMF_DSV4_PROFILE=1` accumulates wall time per stage and prints the split
/// when the process ends. Guessing which half of a layer costs what is how
/// one ends up optimising the cheap one: the fused attention block came out a
/// wash on the release checkpoint, and no amount of reasoning about MAC
/// counts settles whether that is because attention was already cheap or
/// because the device arm was slow.
pub(crate) mod prof {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    pub static ATTN_NS: AtomicU64 = AtomicU64::new(0);
    pub static MOE_NS: AtomicU64 = AtomicU64::new(0);
    pub static CALLS: AtomicU64 = AtomicU64::new(0);
    /// Everything in a layer that is neither attention nor the experts: the
    /// hyper-connection fold and expand, the two norms, the residual.
    pub static HC_NS: AtomicU64 = AtomicU64::new(0);
    /// The head: final norm plus lm_head over 129280 rows.
    pub static HEAD_NS: AtomicU64 = AtomicU64::new(0);
    /// The whole forward, so the buckets can be checked against a total
    /// instead of against a guess. 78 ms of measured work in a 108 ms token
    /// left 30 ms that no counter had ever looked at.
    pub static ALL_NS: AtomicU64 = AtomicU64::new(0);
    pub static TOKENS: AtomicU64 = AtomicU64::new(0);

    /// One token = one visit to layer zero. Counting `moe_step` calls instead
    /// counts layers.
    pub fn note_layer(li: usize) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        if li == 0 {
            // The first token pays for the whole expert set reaching the card
            // — tens of seconds of it. Left in, that one-time cost is divided
            // by every later call and reads as a per-call price: it is what
            // made "the host encodes for 4.45 ms a layer" out of an upload
            // that happens once. Everything measured before the SECOND token
            // starts is therefore thrown away, and the report describes
            // steady state, which is the only thing worth optimising.
            // `swap` and not a TOKENS comparison: resetting TOKENS to 1 made
            // the test true again on every later token, so the report
            // described one token instead of the run.
            if TOKENS.fetch_add(1, Ordering::Relaxed) == 1 && !ZEROED.swap(true, Ordering::Relaxed)
            {
                for a in [&ATTN_NS, &MOE_NS, &HC_NS, &HEAD_NS, &ALL_NS, &CALLS] {
                    a.store(0, Ordering::Relaxed);
                }
                TOKENS.store(1, Ordering::Relaxed);
                #[cfg(feature = "gpu")]
                for a in [
                    &crate::gpu_wgpu::MOE_ENC_NS,
                    &crate::gpu_wgpu::MOE_WAIT_NS,
                    &crate::gpu_wgpu::MOE_BUFS_NS,
                    &crate::gpu_wgpu::MOE_UP_NS,
                    &crate::gpu_wgpu::MOE_PASS_NS,
                    &crate::gpu_wgpu::ATT_ENC_NS,
                    &crate::gpu_wgpu::ATT_WAIT_NS,
                    &crate::gpu_wgpu::CHAIN_ENC_NS,
                    &crate::gpu_wgpu::CHAIN_WAIT_NS,
                    &crate::gpu_wgpu::CHAIN_LAYERS,
                    &crate::gpu_wgpu::CHAIN_RUNS,
                    &crate::gpu_wgpu::SUBMITS,
                    &crate::gpu_wgpu::PASSES,
                ] {
                    a.store(0, Ordering::Relaxed);
                }
            }
        }
    }
    static REPORT: AtomicBool = AtomicBool::new(false);
    /// The one-time "drop the first token's numbers" latch.
    static ZEROED: AtomicBool = AtomicBool::new(false);

    pub fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("CMF_DSV4_PROFILE").is_ok_and(|v| v != "0"))
    }

    /// Print once, from wherever the last caller happens to be — a process
    /// that exits through several paths would otherwise report zero or twice.
    pub fn report() {
        if !on() || REPORT.swap(true, Ordering::Relaxed) {
            return;
        }
        // CALLS counts layer visits, not tokens — dividing by it and calling
        // the result "per token" is off by the layer count, which is 43 on
        // the release and reads as a plausible number either way.
        let calls = CALLS.load(Ordering::Relaxed).max(1);
        let toks = TOKENS.load(Ordering::Relaxed).max(1);
        let (a, m) = (
            ATTN_NS.load(Ordering::Relaxed) as f64 / 1e6,
            MOE_NS.load(Ordering::Relaxed) as f64 / 1e6,
        );
        let all = ALL_NS.load(Ordering::Relaxed) as f64 / 1e6;
        // HC_NS wraps the FFN half's hc_block WHOLE, and moe_step runs
        // inside that block — so the raw counter double-counts every MoE
        // millisecond as hyper-connection time. Reported as the difference:
        // the glue alone. (This inflation is what made moving the
        // hyper-connections to the card look like a 19 ms win when the glue
        // is ~4.)
        let hc = (HC_NS.load(Ordering::Relaxed) as f64 / 1e6
            - MOE_NS.load(Ordering::Relaxed) as f64 / 1e6)
            .max(0.0);
        let hd = HEAD_NS.load(Ordering::Relaxed) as f64 / 1e6;
        eprintln!(
            "[dsv4-профиль] {calls} вызовов слоя за {toks} токенов | \
             на токен: внимание {:.0} мс, MoE {:.0} мс, гипер-связи+нормы {:.0} мс, \
             голова {:.0} мс | на вызов: внимание {:.2}, MoE {:.2}, связи {:.2}",
            a / toks as f64,
            m / toks as f64,
            hc / toks as f64,
            hd / toks as f64,
            a / calls as f64,
            m / calls as f64,
            hc / calls as f64,
        );
        eprintln!(
            "[dsv4-профиль] весь проход {:.0} мс на токен; вне счётчиков {:.0} мс",
            all / toks as f64,
            (all - a - m - hd) / toks as f64,
        );
        #[cfg(feature = "gpu")]
        {
            let ae = crate::gpu_wgpu::ATT_ENC_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let aw = crate::gpu_wgpu::ATT_WAIT_NS.load(Ordering::Relaxed) as f64 / 1e6;
            if ae + aw > 0.0 {
                eprintln!(
                    "[dsv4-профиль] кадр внимания на вызов: кодирование {:.2} мс, \
                     отправка и ожидание {:.2} мс",
                    ae / calls as f64,
                    aw / calls as f64,
                );
            }
            // At the OUTER level on purpose: this used to sit inside the MoE
            // frame's own report, and the chain does not use the MoE frame —
            // so the one number that says where a chained token goes was
            // printed only when the chain was not running.
            let ub = crate::gpu_wgpu::UPLOAD_BYTES.load(Ordering::Relaxed);
            let un = crate::gpu_wgpu::UPLOAD_NS.load(Ordering::Relaxed);
            if ub > 0 && un > 0 {
                eprintln!(
                    "[dsv4-профиль] ЗАЛИВКА весов: {:.1} ГБ за {:.1} с ({:.0} МБ/с)",
                    ub as f64 / 1e9,
                    un as f64 / 1e9,
                    ub as f64 / (un as f64 / 1e9) / 1e6,
                );
            }
            let sub = crate::gpu_wgpu::SUBMITS.load(Ordering::Relaxed);
            if sub > 0 {
                eprintln!(
                    "[dsv4-профиль] ОТПРАВОК на карту: {:.1} на токен, ПРОХОДОВ {:.0} \
                     ({:.1} на слой)",
                    sub as f64 / toks as f64,
                    crate::gpu_wgpu::PASSES.load(Ordering::Relaxed) as f64 / toks as f64,
                    crate::gpu_wgpu::PASSES.load(Ordering::Relaxed) as f64 / calls as f64,
                );
            }
            let cl = crate::gpu_wgpu::CHAIN_LAYERS.load(Ordering::Relaxed);
            if cl > 0 {
                let toks2 = toks.max(1) as f64;
                eprintln!(
                    "[dsv4-профиль] ЦЕПОЧКА на токен: кодирование {:.2} мс, \
                     ожидание {:.2} мс ({} слоёв, {} отправок)",
                    crate::gpu_wgpu::CHAIN_ENC_NS.load(Ordering::Relaxed) as f64 / 1e6 / toks2,
                    crate::gpu_wgpu::CHAIN_WAIT_NS.load(Ordering::Relaxed) as f64 / 1e6 / toks2,
                    cl / toks.max(1),
                    crate::gpu_wgpu::CHAIN_RUNS.load(Ordering::Relaxed) / toks.max(1),
                );
            }
            let e = crate::gpu_wgpu::MOE_ENC_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let wt = crate::gpu_wgpu::MOE_WAIT_NS.load(Ordering::Relaxed) as f64 / 1e6;
            if e + wt > 0.0 {
                let ns = |a: &std::sync::atomic::AtomicU64| {
                    a.load(Ordering::Relaxed) as f64 / 1e6 / calls as f64
                };
                eprintln!(
                    "[dsv4-профиль] кадр MoE на вызов: кодирование {:.2} мс, \
                     отправка и ожидание {:.2} мс",
                    e / calls as f64,
                    wt / calls as f64,
                );
                let an = crate::gpu_wgpu::ATT_GPU_N.load(Ordering::Relaxed);
                if an > 0 {
                    let g = |i: usize| {
                        crate::gpu_wgpu::ATT_GPU_NS[i].load(Ordering::Relaxed) as f64
                            / 1e6 / an as f64
                    };
                    eprintln!(
                        "[dsv4-профиль]   ВНИМАНИЕ НА КАРТЕ на вызов: одиночное {:.3} мс, \
                         оценки {:.3} мс, применение {:.3} мс",
                        g(0), g(1), g(2),
                    );
                }
                let gn = crate::gpu_wgpu::MOE_GPU_N.load(Ordering::Relaxed);
                let gns = crate::gpu_wgpu::MOE_GPU_NS[0].load(Ordering::Relaxed);
                if gn > 0 && gns > 0 {
                    eprintln!(
                        "[dsv4-профиль]   MoE НА КАРТЕ: {:.3} мс на вызов ({gn} замеров)",
                        gns as f64 / 1e6 / gn as f64,
                    );
                } else if gn > 0 {
                    // Zero across thousands of samples is a broken query, not
                    // an instant kernel, and printing it as a time is how a
                    // profile starts lying.
                    eprintln!(
                        "[dsv4-профиль]   MoE НА КАРТЕ: метки вернули НОЛЬ на {gn} замерах — \
                         запрос времени не сработал, число не использовать"
                    );
                }
                eprintln!(
                    "[dsv4-профиль]   из кодирования: буферы экспертов {:.2} мс, \
                     загрузки {:.2} мс, проходы {:.2} мс",
                    ns(&crate::gpu_wgpu::MOE_BUFS_NS),
                    ns(&crate::gpu_wgpu::MOE_UP_NS),
                    ns(&crate::gpu_wgpu::MOE_PASS_NS),
                );
            }
        }
    }
}

/// Print the per-token split, if `CMF_DSV4_PROFILE` asked for one.
pub fn profile_report() {
    prof::report();
}

/// `CMF_DSV4_GPU_ATTN=1` moves the attention block onto the device as one
/// submission. Off by default: it needs every attention weight in q4tp and a
/// working wgpu context, and a frame that declines mid-layer after the state
/// has been advanced would be worse than one that never ran.
fn gpu_attn_enabled() -> bool {
    #[cfg(feature = "gpu")]
    {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            let want = std::env::var("CMF_DSV4_GPU_ATTN")
                .map(|v| v != "0")
                .unwrap_or(false);
            let have = want && crate::gpu::backend_available();
            if want && !have {
                tracing::warn!(
                    "CMF_DSV4_GPU_ATTN задан, но устройства нет — блок внимания                      остаётся на CPU. Проверьте CMF_GPU=wgpu и Vulkan-ICD."
                );
            }
            if std::env::var("CMF_DSV4_FRAME_DEBUG").is_ok() {
                eprintln!("кадр dsv4: запрошен={want} доступен={have}");
            }
            have
        })
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// The device half of `attention_step`. Returns false — having changed
/// nothing — whenever it cannot do the whole block, so the caller's CPU path
/// is still correct to run.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn attn_frame(
    l: &Dsv4Layer,
    cfg: &Dsv4Cfg,
    st: &Dsv4State,
    li: usize,
    qn: &[f32],
    idxs: &[usize],
    inv_freq: &[f32],
    pos: usize,
    win_len: usize,
    scale: f32,
    // Present: the frame also does this layer's hyper-connection handover
    // and leaves the MoE half's input on the card. `out` may then be empty.
    hc: Option<&crate::gpu_wgpu::Dsv4HcTail>,
    out: &mut [f32],
) -> bool {
    let hd = cfg.head_dim;
    let (Some(wq_a), Some(wq_b), Some(wo_a), Some(wo_b)) = (
        l.wq_a.model_idx(),
        l.wq_b.model_idx(),
        l.wo_a.model_idx(),
        l.wo_b.model_idx(),
    ) else {
        return false;
    };
    let Some(model) = l.wq_b.model_arc() else {
        return false;
    };
    // Fixed window region, then the compressed tail — so a token writes one
    // window slot's worth of movement and whatever the compressor just added,
    // not the whole cache. `cap` has to cover the longest run this sequence
    // will reach; the compressed axis grows by one entry per `ratio` tokens.
    let n_comp = st.compressed[li].len() / hd;
    let cap = (cfg.window + n_comp.next_power_of_two().max(64)) * hd;
    let kv_id = st.kv_id;
    // The window is rewritten whole. A ring would write one slot instead of
    // 128 — 2 KB against 256 — and was tried: it bought NOTHING (the cost is
    // per-dispatch driver bookkeeping, not the copy) and moved perplexity by
    // 6e-5 because the attended positions arrive in a different order and the
    // softmax accumulates differently. Not a trade worth making.
    if !crate::gpu_wgpu::dsv4_cache_write(kv_id, li, 0, &st.window[li], cap) {
        return false;
    }
    // The compressed axis only ever grows, so write the TAIL. Rewriting it
    // whole was 22 MB a token at 1024 positions — the cache write, not the
    // arithmetic, was what the attention block had left to pay.
    // The compressed tail is written WHOLE every token. Writing only the new
    // part was tried and gave nothing measurable, and the bookkeeping it
    // needs — a per-layer tail count invalidated by every buffer growth — is
    // exactly the kind of state that drifts silently and shows up as a model
    // that stops early. Not worth carrying for zero.
    if n_comp > 0
        && !crate::gpu_wgpu::dsv4_cache_write(
            kv_id,
            li,
            cfg.window * hd,
            &st.compressed[li],
            cap,
        )
    {
        return false;
    }
    let idx32: Vec<u32> = idxs
        .iter()
        .map(|&p| {
            if p < win_len {
                p as u32
            } else {
                (cfg.window + (p - win_len)) as u32
            }
        })
        .collect();
    let w = crate::gpu_wgpu::Dsv4AttnW {
        wq_a,
        wq_b,
        wo_a,
        wo_b,
        q_norm: &l.q_norm,
        sink: &l.attn_sink,
    };
    let g = crate::gpu_wgpu::Dsv4AttnGeom {
        dim: cfg.dim,
        nh: cfg.n_heads,
        hd,
        rd: cfg.rope_head_dim,
        q_lora: cfg.q_lora_rank,
        o_lora: cfg.o_lora_rank,
        o_groups: cfg.o_groups,
        eps: cfg.norm_eps,
        scale,
    };
    crate::gpu_wgpu::dsv4_attn_frame(
        &model, &w, g, &[], Some(qn), kv_id, li, &idx32, inv_freq, pos, hc, out,
    )
}

/// What the host still owes the device before a layer frame can run: the
/// shared LoRA vector the indexer reads, and the attended position list.
#[derive(Default)]
pub struct AttnPrep {
    pub qr: Vec<f32>,
    pub idxs: Vec<usize>,
    pub win_len: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn attention_step(
    hidden: &[f32],
    l: &Dsv4Layer,
    cfg: &Dsv4Cfg,
    st: &mut Dsv4State,
    li: usize,
    // Chosen by the caller from the layer's kind — see Dsv4Globals.
    inv_freq: &[f32],
    pool: Option<&crate::pool::Pool>,
    // When set, stop once the caches are advanced and the index list is
    // built, and hand those back instead of running attention: the layer
    // frame does the rest on the device.
    prep_out: Option<&mut AttnPrep>,
    out: &mut [f32],
) {
    let _t0 = prof::on().then(std::time::Instant::now);
    let _guard = scopeguard_attn(_t0);
    let (hd, rd) = (cfg.head_dim, cfg.rope_head_dim);
    let pos = st.pos;
    if std::env::var("CMF_FREQ_DEBUG").is_ok() && li == 0 && pos == 0 {
        eprintln!(
            "    [порт] rd={rd} частот={} inv_freq[0..4]={:?}",
            inv_freq.len(),
            &inv_freq[..4.min(inv_freq.len())]
        );
    }

    // ── q and kv: both read the same hidden state, so they go out as ONE
    // dispatch. The norms after them differ, and they stay separate.
    // (q: wq_a → q_norm → wq_b → per-head norm → rope tail;
    //  kv: one head's width, shared by every query head.)
    let mut qr = vec![0.0f32; cfg.q_lora_rank];
    let mut kv = vec![0.0f32; hd];
    crate::qtensor::QTensor::matvec_many([&l.wq_a, &l.wkv], hidden, [&mut qr, &mut kv], pool);
    rms_weighted(&mut qr, &l.q_norm, cfg.norm_eps);
    // The queries are built further down, after the frame has had its chance
    // at the whole block. `qr` is needed either way: the indexer reads it.
    let on_gpu = gpu_attn_enabled();

    rms_weighted(&mut kv, &l.kv_norm, cfg.norm_eps);
    rope_tail(&mut kv, inv_freq, pos, rd, false);

    // ── the compressor: accumulate `ratio` tokens, then fold them into
    // one compressed entry. The reference fires when (pos+1) % ratio == 0,
    // so a partial window simply waits — which is why the state carries
    // the pending streams across tokens.
    if let Some(cp) = &l.compressor {
        let mut pk = std::mem::take(&mut st.pending_kv[li]);
        let mut ps = std::mem::take(&mut st.pending_score[li]);
        let mut qk = std::mem::take(&mut st.prev_kv[li]);
        let mut qs = std::mem::take(&mut st.prev_score[li]);
        let entry = compressor_step(
            cp,
            hidden,
            pos,
            rd,
            cfg.norm_eps,
            inv_freq,
            pool,
            &mut pk,
            &mut ps,
            &mut qk,
            &mut qs,
        );
        st.pending_kv[li] = pk;
        st.pending_score[li] = ps;
        st.prev_kv[li] = qk;
        st.prev_score[li] = qs;
        if let Some(e) = entry {
            st.compressed[li].extend_from_slice(&e);
        }
    }
    // The indexer scores against ITS OWN compressed cache, built by its own
    // compressor. Without this the cache is empty, `n_ix` is zero, and every
    // indexer layer picks no compressed positions at all — the long-range
    // memory is built and then never read.
    if let Some(ix) = &l.indexer {
        let mut pk = std::mem::take(&mut st.pending_ix_kv[li]);
        let mut ps = std::mem::take(&mut st.pending_ix_score[li]);
        let mut qk = std::mem::take(&mut st.prev_ix_kv[li]);
        let mut qs = std::mem::take(&mut st.prev_ix_score[li]);
        let entry = compressor_step(
            &ix.compressor,
            hidden,
            pos,
            rd,
            cfg.norm_eps,
            inv_freq,
            pool,
            &mut pk,
            &mut ps,
            &mut qk,
            &mut qs,
        );
        st.pending_ix_kv[li] = pk;
        st.pending_ix_score[li] = ps;
        st.prev_ix_kv[li] = qk;
        st.prev_ix_score[li] = qs;
        if let Some(e) = entry {
            st.index_kv[li].extend_from_slice(&e);
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
    let n_pos = win_len + st.compressed[li].len() / hd;

    // Index list: every window position, plus whatever the indexer picked
    // (or, without an indexer, every compressed position).
    //
    // CMF_DSV4_NO_COMPRESSED=1 attends to the sliding window ALONE. That is
    // not a mode anyone should serve — it drops the model's long-range
    // memory — but it separates two failure modes that look identical from
    // the outside: output that degrades because the compressed path is
    // wrong, and output that degrades because the weights are too coarse.
    let mut idxs: Vec<usize> = (0..win_len).collect();
    if !st.compressed[li].is_empty() && !no_compressed() {
        let n_comp = st.compressed[li].len() / hd;
        match &l.indexer {
            Some(ix) => {
                // The indexer scores from the SHARED LoRA output through
                // its own wq_b — not from attention's queries — and its
                // per-head weights are a projection of the hidden state,
                // scaled by head_dim^-0.5 * n_heads^-0.5 as the reference
                // folds into `weights_proj`'s output.
                //
                // The reference also applies a randomized Hadamard rotation
                // to the queries here and to the keys in the indexer's
                // compressor, then simulates FP4 on both. That transform is
                // orthogonal (`hadamard_transform` scaled by d^-0.5) and it
                // hits BOTH sides of the same dot product, so it cancels:
                // its purpose is to condition the FP4 quantization, which we
                // do not do either. Omitting the pair is exact, and keeping
                // f32 is strictly more accurate than the reference — not an
                // approximation to be fixed later.
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
                    pool,
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
    if let Some(p) = prep_out {
        p.qr = qr;
        p.idxs = idxs;
        p.win_len = win_len;
        return;
    }

    // ── the whole block on the device, or nothing ──
    let scale = (hd as f32).powf(-0.5);
    #[cfg(feature = "gpu")]
    if on_gpu
        && attn_frame(
            l, cfg, st, li, &qr, &idxs, inv_freq, pos, win_len, scale, None, out,
        )
    {
        return;
    }

    // ── queries: wq_b, then a norm and the rope tail per head ──
    let mut q = vec![0.0f32; cfg.n_heads * hd];
    l.wq_b.matvec(&qr, &mut q, pool);
    for h in 0..cfg.n_heads {
        let head = &mut q[h * hd..(h + 1) * hd];
        rms_inplace(head, cfg.norm_eps);
        rope_tail(head, inv_freq, pos, rd, false);
    }
    let mut cache: Vec<f32> = st.window[li].clone();
    cache.extend_from_slice(&st.compressed[li]);

    // ── sparse attention per head, then the inverse rope ──
    let mut attn = vec![0.0f32; cfg.n_heads * hd];
    for h in 0..cfg.n_heads {
        let qh = &q[h * hd..(h + 1) * hd];
        // Straight into this head's slice of the output: the scratch vector
        // that used to sit here was an allocation and a copy per head, so 64
        // of each per layer per token, for a value that was never read
        // anywhere else.
        let oh = &mut attn[h * hd..(h + 1) * hd];
        sparse_attend(qh, &cache, &idxs, l.attn_sink[h], scale, hd, oh);
        rope_tail(oh, inv_freq, pos, rd, true);
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
/// Per-layer expert-selection counts, the routing field a task-conditional
/// expert set is derived from (`CMF_MOE_STATS`). The generic MoE path keeps
/// these on its `MoeFfn`; this architecture has its own experts and never
/// touches that struct, so without this the field cannot be recorded for
/// DeepSeek-V4 at all — and its hash layers already make defrag useless, so
/// the only interesting question is what the OTHER forty layers do.
///
/// Decode drives this from one thread; the pool parallelizes inside the
/// matvecs, below this point.
thread_local! {
    static ROUTE_COUNTS: std::cell::RefCell<Vec<Vec<u64>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn route_stats_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_MOE_STATS").is_ok())
}

fn record_route(li: usize, n_layers_hint: usize, n_experts: usize, idx: &[usize]) {
    ROUTE_COUNTS.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() <= li.max(n_layers_hint) {
            c.resize(li.max(n_layers_hint) + 1, Vec::new());
        }
        let row = &mut c[li];
        if row.len() < n_experts {
            row.resize(n_experts, 0);
        }
        for &e in idx {
            if e < row.len() {
                row[e] += 1;
            }
        }
    });
}

/// Take the recorded routing field, leaving the counters empty.
pub fn take_route_counts() -> Vec<Vec<u64>> {
    ROUTE_COUNTS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

/// Charge elapsed time to a counter when it goes out of scope — the two
/// steps have several early returns each, and a timer that only stops on the
/// long path measures the short one as free.
struct Charge(Option<std::time::Instant>, &'static std::sync::atomic::AtomicU64);
impl Drop for Charge {
    fn drop(&mut self) {
        if let Some(t) = self.0 {
            self.1.fetch_add(
                t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }
}
fn scopeguard_attn(t: Option<std::time::Instant>) -> Charge {
    Charge(t, &prof::ATTN_NS)
}
fn scopeguard_moe(t: Option<std::time::Instant>, li: usize) -> Charge {
    if t.is_some() {
        prof::note_layer(li);
    }
    Charge(t, &prof::MOE_NS)
}

/// The whole token, one submission per layer. Returns false having changed
/// nothing if the device declines any layer — the caller's loop is then still
/// correct to run.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn dsv4_layer_loop(
    state: &mut [f32],
    layers: &[Dsv4Layer],
    g: &Dsv4Globals,
    cfg: &Dsv4Cfg,
    st: &mut Dsv4State,
    token_id: u32,
    inv_freq: &[f32],
    pool: Option<&crate::pool::Pool>,
    scratch: &mut HcScratch,
) -> bool {
    let dim = cfg.dim;
    let freqs_of = |l: &Dsv4Layer| -> &[f32] {
        let f = if l.compressor.is_some() {
            &g.inv_freq_compress
        } else {
            &g.inv_freq_window
        };
        if f.is_empty() { inv_freq } else { f.as_slice() }
    };
    // PRE-FLIGHT. The prep inside the loop advances the window and the
    // compressor caches, so a refusal halfway leaves state that the CPU
    // fallback would advance a SECOND time — which is not a slow answer but a
    // wrong one. Everything that can decline is therefore asked before the
    // first byte of state moves. The expert upload happens here too, which is
    // where it belonged anyway.
    // The head goes to the card BEFORE the experts ask for room. It is the
    // single most-used tensor in the file — every token reads all of it —
    // and it is a rounding error next to the expert stack: 265 MB against
    // ninety-odd gigabytes on the release. Uploaded in first-touch order it
    // arrived last, after the budget was gone, and stayed on the host for
    // the life of the process.
    {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            if let (Some(idx), Some(model)) = (g.head.model_idx(), g.head.model_arc()) {
                let ok = crate::gpu_wgpu::dsv4_weight_ready(&model, idx);
                tracing::info!("dsv4: голова на карте: {}", if ok { "да" } else { "нет" });
            }
        }
    }
    let mut on_dev = vec![false; layers.len()];
    for (li, l) in layers.iter().enumerate() {
        let Some(pk) = pack_for(l, cfg, li) else {
            return false;
        };
        if l.wq_a.model_idx().is_none()
            || l.wq_b.model_idx().is_none()
            || l.wo_a.model_idx().is_none()
            || l.wo_b.model_idx().is_none()
        {
            return false;
        }
        let Some(model) = l.experts.first().and_then(|e| e.w1.model_arc()) else {
            return false;
        };
        let gu_q2 = l
            .experts
            .first()
            .is_some_and(|e| e.w1.model_dtype() == Some(cortiq_core::TensorDtype::Q2TiledP));
        // A layer whose experts do not fit is not a reason to abandon the
        // token: 100 GB of experts against a 98 GB card means SOME layer will
        // always miss. Those run on the host, with the state fetched and put
        // back around them — two transfers for the few that need it.
        // The attention weights have to be asked for too. Experts fill the
        // card first, and a wo_b that misses at layer 11 used to surface as a
        // mid-loop refusal — after the caches had advanced, which the CPU
        // fallback then advanced again.
        // …and, when the layer is to prepare itself, everything that
        // preparation reads: the KV projection, both compressors and the
        // indexer. Leaving them out is how the chain came to refuse ninety
        // times a token on the release — the experts had taken the card by
        // the time `dsv4_encode_prep` asked, and it declined silently into a
        // fallback that looked like "the chain simply does not help".
        let mut want = vec![
            l.wq_a.model_idx(),
            l.wq_b.model_idx(),
            l.wo_a.model_idx(),
            l.wo_b.model_idx(),
        ];
        if chain_enabled() {
            want.push(l.wkv.model_idx());
            if let Some(cp) = &l.compressor {
                want.push(cp.wkv.model_idx());
                want.push(cp.wgate.model_idx());
            }
            if let Some(ix) = &l.indexer {
                want.push(ix.wq_b.model_idx());
                want.push(ix.weights_proj.model_idx());
                want.push(ix.compressor.wkv.model_idx());
                want.push(ix.compressor.wgate.model_idx());
            }
        }
        let attn_ok = want
            .into_iter()
            .flatten()
            .all(|i| crate::gpu_wgpu::dsv4_weight_ready(&model, i));
        // The layer frame's router has no remap yet, so a PARTIAL packing
        // there would silently become a mask — the one thing that is known to
        // wreck this model. Such a layer goes to the host until the frame
        // learns to hand cold picks back the way the two-frame path does.
        on_dev[li] = attn_ok
            && pk.globals.len() == cfg.n_routed_experts
            && crate::gpu_wgpu::dsv4_experts_ready(&model, &pk.tensors, cfg.moe_inter, dim, gu_q2);
    }
    if !on_dev.iter().any(|&x| x) {
        return false;
    }

    // Which layers the card actually took, said once. A layer that falls to
    // the host costs an order of magnitude more than one that does not, and
    // "the GPU path is on" hid the difference between all of them and most.
    {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let host: Vec<usize> = on_dev
                .iter()
                .enumerate()
                .filter(|&(_, d)| !*d)
                .map(|(i, _)| i)
                .collect();
            if host.is_empty() {
                tracing::info!("dsv4: все {} слоёв на карте", on_dev.len());
            } else {
                tracing::info!(
                    "dsv4: {} из {} слоёв на карте; на хосте остались {:?}",
                    on_dev.len() - host.len(),
                    on_dev.len(),
                    host,
                );
            }
        }
    }

    // Layer zero's opening fold has no frame before it to have prepared it.
    let (mut folded, post0, comb0) = hc_fold_norm(
        state,
        &layers[0].hc_attn_fn,
        &layers[0].hc_attn_scale,
        &layers[0].hc_attn_base,
        &layers[0].attn_norm,
        cfg,
        pool,
    );
    if !crate::gpu_wgpu::dsv4_state_write(state)
        || !crate::gpu_wgpu::dsv4_hc_write(&post0, &comb0)
    {
        return false;
    }
    // The device-owned set must not move once a token has run on it.
    if st.dev_owned && st.dev_set != on_dev {
        tracing::warn!("состав слоёв на карте изменился — кеши на разных сторонах");
        return false;
    }
    let chain = chain_enabled();
    // CMF_DSV4_LAYERS_PROBE=N — TIMING ONLY, the answer is garbage. Runs the
    // first N layers and leaves the rest alone. Decode time against N is a
    // line whose SLOPE is the per-layer cost and whose intercept is
    // everything that happens once a token. Unlike the skip probe it does
    // not change what a layer does — which on a MoE model is the difference
    // between a measurement and an artefact, because dropping any stage
    // changes the routing and the routing changes what the experts cost.
    let layer_cap = {
        static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *N.get_or_init(|| {
            std::env::var("CMF_DSV4_LAYERS_PROBE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(usize::MAX)
        })
    };
    let mut run: Vec<usize> = Vec::new();
    let mut sink_out = vec![0.0f32; dim];
    for (li, l) in layers.iter().enumerate() {
        if li >= layer_cap {
            break;
        }
        // The device path never ticked the profiler, so every per-token
        // number it printed described the two host-path tokens at the start
        // of a run — the ones that also pay for the upload. Ticking here is
        // what makes the chain's encode-and-wait split a per-token figure at
        // all.
        if prof::on() {
            prof::note_layer(li);
        }
        if chain && on_dev[li] {
            // Hash layers used to break the run in two: their forced expert
            // list changes per token, went through the (tag, len) upload
            // pool, and every layer of a submission shared one buffer. The
            // list has a per-layer slot now, so they chain like the rest.
            run.push(li);
            // CMF_DSV4_CHAIN_MAX=N caps a run's length. Diagnostic, not a
            // tuning knob: length-1 runs put ONE layer per submission, which
            // separates "the layer frame is wrong" from "layers in one
            // encoder contaminate each other" in a single ppl run.
            if run.len() >= chain_max() {
                let need_qn = run[0] == 0 || !on_dev[run[0] - 1];
                if !dsv4_chain_run(
                    layers, &run, cfg, g, st, token_id, &mut folded, None, need_qn, pool,
                ) {
                    return false;
                }
                run.clear();
            }
            continue;
        }
        if chain
            && !run.is_empty()
            && !dsv4_chain_run(
                layers, &run, cfg, g, st, token_id, &mut folded, None,
                run[0] == 0 || !on_dev[run[0] - 1], pool,
            )
        {
            return false;
        }
        run.clear();
        if !on_dev[li] {
            if !crate::gpu_wgpu::dsv4_state_read(state) {
                return false;
            }
            let freqs = freqs_of(l);
            hc_block(
                state,
                &l.hc_attn_fn,
                &l.hc_attn_scale,
                &l.hc_attn_base,
                &l.attn_norm,
                cfg,
                scratch,
                pool,
                |f, o| attention_step(f, l, cfg, st, li, freqs, pool, None, o),
            );
            hc_block(
                state,
                &l.hc_ffn_fn,
                &l.hc_ffn_scale,
                &l.hc_ffn_base,
                &l.ffn_norm,
                cfg,
                scratch,
                pool,
                // The layer the card had no room for. Its experts are
                // reached one matvec at a time and the probe sends each to
                // the device — right per op, and a fence per op: this one
                // layer is why a token that submits ONCE for 42 layers
                // submits 13 times. CMF_DSV4_HOST_CPU_MOE=1 keeps them on
                // the host instead, trading arithmetic for round trips.
                |f, o| {
                    if host_cpu_moe() {
                        crate::gpu::cpu_scope(|| moe_step(f, l, cfg, token_id, li, pool, o))
                    } else {
                        moe_step(f, l, cfg, token_id, li, pool, o)
                    }
                },
            );
            if let Some(n) = layers.get(li + 1) {
                let (f, p2, c2) = hc_fold_norm(
                    state,
                    &n.hc_attn_fn,
                    &n.hc_attn_scale,
                    &n.hc_attn_base,
                    &n.attn_norm,
                    cfg,
                    pool,
                );
                folded = f;
                if !crate::gpu_wgpu::dsv4_hc_write(&p2, &c2) {
                    return false;
                }
            }
            if !crate::gpu_wgpu::dsv4_state_write(state) {
                return false;
            }
            continue;
        }
        let mut prep = AttnPrep::default();
        attention_step(
            &folded,
            l,
            cfg,
            st,
            li,
            freqs_of(l),
            pool,
            Some(&mut prep),
            &mut sink_out,
        );
        // The caches the frame will read.
        let hd = cfg.head_dim;
        let n_comp = st.compressed[li].len() / hd;
        let cap = (cfg.window + n_comp.next_power_of_two().max(64)) * hd;
        let kv_id = st.kv_id;
        if !crate::gpu_wgpu::dsv4_cache_write(kv_id, li, 0, &st.window[li], cap)
            || (n_comp > 0
                && !crate::gpu_wgpu::dsv4_cache_write(
                    kv_id,
                    li,
                    cfg.window * hd,
                    &st.compressed[li],
                    cap,
                ))
        {
            return false;
        }
        let idx32: Vec<u32> = prep
            .idxs
            .iter()
            .map(|&p| {
                if p < prep.win_len {
                    p as u32
                } else {
                    (cfg.window + (p - prep.win_len)) as u32
                }
            })
            .collect();
        let Some(pk) = pack_for(l, cfg, li) else {
            return false;
        };
        let (Some(wq_a), Some(wq_b), Some(wo_a), Some(wo_b)) = (
            l.wq_a.model_idx(),
            l.wq_b.model_idx(),
            l.wo_a.model_idx(),
            l.wo_b.model_idx(),
        ) else {
            return false;
        };
        let Some(model) = l.experts.first().and_then(|e| e.w1.model_arc()) else {
            return false;
        };
        let forced: Option<Vec<usize>> = l.tid2eid.as_ref().and_then(|tbl| {
            let v: Vec<usize> = hash_route(tbl, cfg.vocab, cfg.top_k, token_id)
                .into_iter()
                .map(|gi| pk.to_slot[gi])
                .collect();
            if v.iter().any(|&x| x == usize::MAX) {
                None
            } else {
                Some(v)
            }
        });
        if l.tid2eid.is_some() && forced.is_none() {
            return false;
        }
        let nxt = layers.get(li + 1);
        let w = crate::gpu_wgpu::Dsv4LayerW {
            attn: crate::gpu_wgpu::Dsv4AttnW {
                wq_a,
                wq_b,
                wo_a,
                wo_b,
                q_norm: &l.q_norm,
                sink: &l.attn_sink,
            },
            moe: crate::gpu_wgpu::Dsv4MoeW {
                router: &[],
                experts: &pk.tensors,
                logits: &[],
                // The PACK's bias, whose address outlives the process: the
                // frame's const cache is keyed on it, and a per-layer Vec
                // here handed every layer the first layer's — the exact
                // transient-Vec trap the const_buf war story describes,
                // reintroduced by this session and caught because the OFF
                // baseline moved.
                bias: pk.bias.as_deref(),
                forced: forced.as_deref(),
                remap: None,
            },
            hc_ffn_fn: &l.hc_ffn_fn,
            hc_ffn_scale: &l.hc_ffn_scale,
            hc_ffn_base: &l.hc_ffn_base,
            hc_next_fn: nxt.map(|n| n.hc_attn_fn.as_slice()),
            hc_next_scale: nxt.map_or(&l.hc_attn_scale, |n| &n.hc_attn_scale),
            hc_next_base: nxt.map_or(&l.hc_attn_base, |n| n.hc_attn_base.as_slice()),
            ffn_norm: &l.ffn_norm,
            next_norm: nxt.map_or(&l.attn_norm, |n| n.attn_norm.as_slice()),
            next_q_norm: nxt.map_or(&l.q_norm, |n| n.q_norm.as_slice()),
            next_wq_a: nxt.and_then(|n| n.wq_a.model_idx()),
            router: &pk.router,
        };
        let geom = crate::gpu_wgpu::Dsv4LayerGeom {
            attn: crate::gpu_wgpu::Dsv4AttnGeom {
                dim,
                nh: cfg.n_heads,
                hd,
                rd: cfg.rope_head_dim,
                q_lora: cfg.q_lora_rank,
                o_lora: cfg.o_lora_rank,
                o_groups: cfg.o_groups,
                eps: cfg.norm_eps,
                scale: (hd as f32).powf(-0.5),
            },
            moe: crate::gpu_wgpu::Dsv4MoeGeom {
                hidden: dim,
                inter: cfg.moe_inter,
                top_k: cfg.top_k,
                route_scale: cfg.route_scale,
                swiglu_limit: cfg.swiglu_limit,
                gu_q2: l.experts.first().is_some_and(|e| {
                    e.w1.model_dtype() == Some(cortiq_core::TensorDtype::Q2TiledP)
                }),
            },
            hc: cfg.hc_mult,
            hc_eps: cfg.hc_eps,
            sinkhorn_iters: cfg.hc_sinkhorn_iters,
        };
        let mut next = vec![0.0f32; dim];
        if !crate::gpu_wgpu::dsv4_layer_frame(
            &model,
            &w,
            geom,
            kv_id,
            li,
            Some(&prep.qr),
            &idx32,
            freqs_of(l),
            st.pos,
            &mut next,
        ) {
            return false;
        }
        folded = next;
    }
    let mut state_home = false;
    if chain {
        if !run.is_empty() {
            // The token's LAST run brings the state back with it. Only the
            // last: an earlier run's state is one the layers after it still
            // change.
            let need_qn = run[0] == 0 || !on_dev[run[0] - 1];
            let last_on_dev = *on_dev.last().unwrap_or(&false);
            let carry = last_on_dev && run.last() == Some(&(layers.len() - 1));
            let ok = if carry {
                let r = dsv4_chain_run(
                    layers, &run, cfg, g, st, token_id, &mut folded,
                    Some(state), need_qn, pool,
                );
                state_home = r;
                r
            } else {
                dsv4_chain_run(
                    layers, &run, cfg, g, st, token_id, &mut folded, None, need_qn, pool,
                )
            };
            if !ok {
                return false;
            }
        }
        if st.dev_set.is_empty() {
            st.dev_set = on_dev.clone();
        }
    }
    if state_home {
        return true;
    }
    crate::gpu_wgpu::dsv4_state_read(state)
}

#[cfg(feature = "gpu")]
fn chain_max() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("CMF_DSV4_CHAIN_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX)
    })
}

/// `CMF_DSV4_CHAIN=1`: put a run of consecutive device-capable layers in ONE
/// submission. Off by default until it has been measured on a real card.
#[cfg(feature = "gpu")]
/// `CMF_DSV4_HOST_CPU_MOE=1`: a layer that fell off the card runs its MoE on
/// the host WITHOUT the per-op device route — one fence a token instead of
/// one a matvec. Whether that wins is a measurement.
fn host_cpu_moe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_DSV4_HOST_CPU_MOE").is_ok_and(|v| v != "0"))
}

fn chain_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_DSV4_CHAIN").is_ok_and(|v| v != "0"))
}

/// Encode a maximal run of consecutive device-capable layers and submit it
/// ONCE. Every layer in the run builds its own attention inputs on the card,
/// so nothing comes back between them — that is the whole saving.
///
/// The run's state belongs to the device from here on: `st.window`,
/// `st.compressed` and the compressor streams for these layers are stale on
/// the host afterwards, and only the counts in `st.dev_*` are kept. A layer
/// that has ever been in a run must therefore never be handed to the CPU
/// path again, which `dev_owned` records.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn dsv4_chain_run(
    layers: &[Dsv4Layer],
    run: &[usize],
    cfg: &Dsv4Cfg,
    g: &Dsv4Globals,
    st: &mut Dsv4State,
    token_id: u32,
    // In AND out: the run reads the fold it starts from and MUST leave the
    // fold it produced, because whatever follows — a host layer, or the next
    // run after a cap — seeds from this. Passing it read-only left every
    // later segment starting from a stale fold: exact with one unbroken run,
    // release-scale garbage the moment anything splits the chain.
    folded: &mut Vec<f32>,
    // When present, the hyper-connection state rides home in the run's own
    // submission instead of costing a second fence afterwards. Only the
    // token's LAST run passes it — an earlier one would read a state the
    // layers after it still change.
    state_out: Option<&mut [f32]>,
    // Whether the device's qn buffer is stale: true at layer zero and after
    // a host layer. When the previous layer was chained, its frame's tail
    // already left THIS layer's LoRA vector on the card, and recomputing it
    // here was a full wq_a matvec on the CPU per run — at CHAIN_MAX=1 that
    // is one per LAYER, which is how a 43-fence path measured slower than
    // an 86-fence one.
    need_qn: bool,
    pool: Option<&crate::pool::Pool>,
) -> bool {
    if run.is_empty() {
        return true;
    }
    let (dim, hd) = (cfg.dim, cfg.head_dim);
    let first = run[0];
    let Some(model) = layers[first].experts.first().and_then(|e| e.w1.model_arc()) else {
        return false;
    };
    if need_qn {
        let mut qn0 = vec![0.0f32; cfg.q_lora_rank];
        layers[first].wq_a.matvec(folded, &mut qn0, pool);
        rms_weighted(&mut qn0, &layers[first].q_norm, cfg.norm_eps);
        if !crate::gpu_wgpu::dsv4_chain_seed(folded, &qn0) {
            return false;
        }
    } else if !crate::gpu_wgpu::dsv4_chain_seed_fold(folded) {
        return false;
    }

    // Held apart from the borrowing structs below, which point into them.
    let mut packs = Vec::with_capacity(run.len());
    let mut forceds: Vec<Option<Vec<usize>>> = Vec::with_capacity(run.len());
    for &li in run {
        let Some(pk) = pack_for(&layers[li], cfg, li) else {
            return false;
        };
        let forced: Option<Vec<usize>> = layers[li].tid2eid.as_ref().and_then(|tbl| {
            let v: Vec<usize> = hash_route(tbl, cfg.vocab, cfg.top_k, token_id)
                .into_iter()
                .map(|gi| pk.to_slot[gi])
                .collect();
            if v.iter().any(|&x| x == usize::MAX) { None } else { Some(v) }
        });
        if layers[li].tid2eid.is_some() && forced.is_none() {
            return false;
        }
        forceds.push(forced);
        packs.push(pk);
    }

    let mut items = Vec::with_capacity(run.len());
    let mut freqs = Vec::with_capacity(run.len());
    for (i, &li) in run.iter().enumerate() {
        let l = &layers[li];
        let (Some(wq_a), Some(wq_b), Some(wo_a), Some(wo_b), Some(wkv)) = (
            l.wq_a.model_idx(),
            l.wq_b.model_idx(),
            l.wo_a.model_idx(),
            l.wo_b.model_idx(),
            l.wkv.model_idx(),
        ) else {
            return false;
        };
        let comp = match &l.compressor {
            None => None,
            Some(cp) => {
                let (Some(a), Some(b)) = (cp.wkv.model_idx(), cp.wgate.model_idx()) else {
                    return false;
                };
                Some((
                    crate::gpu_wgpu::Dsv4CompW { wkv: a, wgate: b, norm: &cp.norm, ape: &cp.ape },
                    crate::gpu_wgpu::Dsv4CompGeom {
                        width: cp.wkv.rows(),
                        hidden: dim,
                        ratio: cp.ratio,
                        overlap: cp.overlap,
                        rope_dim: cfg.rope_head_dim,
                        eps: cfg.norm_eps,
                    },
                ))
            }
        };
        let ix = match &l.indexer {
            None => None,
            Some(ixr) => {
                let cp = &ixr.compressor;
                let (Some(a), Some(b), Some(qb), Some(wp)) = (
                    cp.wkv.model_idx(),
                    cp.wgate.model_idx(),
                    ixr.wq_b.model_idx(),
                    ixr.weights_proj.model_idx(),
                ) else {
                    return false;
                };
                let ih = ixr.weights_proj.rows();
                Some((
                    crate::gpu_wgpu::Dsv4CompW { wkv: a, wgate: b, norm: &cp.norm, ape: &cp.ape },
                    crate::gpu_wgpu::Dsv4CompGeom {
                        width: cp.wkv.rows(),
                        hidden: dim,
                        ratio: cp.ratio,
                        overlap: cp.overlap,
                        rope_dim: cfg.rope_head_dim,
                        eps: cfg.norm_eps,
                    },
                    crate::gpu_wgpu::Dsv4IxW { wq_b: qb, weights_proj: wp },
                    crate::gpu_wgpu::Dsv4IxGeom {
                        ih,
                        idim: ixr.wq_b.rows() / ih.max(1),
                        q_lora: cfg.q_lora_rank,
                        hidden: dim,
                        rope_dim: cfg.rope_head_dim,
                        eps: cfg.norm_eps,
                        top_k: cfg.index_topk,
                        window: cfg.window,
                    },
                ))
            }
        };
        // The cache has to be big enough BEFORE the frame appends into it:
        // a chained layer never calls dsv4_cache_write, which is what used
        // to create and grow it.
        let ew_c0 = l.compressor.as_ref().map_or(0, |cp| {
            if cp.overlap { cp.wkv.rows() / 2 } else { cp.wkv.rows() }
        });
        let need = cfg.window * hd + (st.dev_n_comp[li] + 2) * ew_c0.max(1);
        if !crate::gpu_wgpu::dsv4_cache_ensure(st.kv_id, li, need.next_power_of_two()) {
            return false;
        }
        let ew_c = comp.as_ref().map_or(0, |(_, cg)| {
            if cg.overlap { cg.width / 2 } else { cg.width }
        });
        let ew_i = ix.as_ref().map_or(0, |(_, cg, _, _)| {
            if cg.overlap { cg.width / 2 } else { cg.width }
        });
        let prep = crate::gpu_wgpu::Dsv4Prep {
            wkv,
            kv_norm: &l.kv_norm,
            comp,
            ix,
            filled: st.dev_filled[li],
            window: cfg.window,
            n_comp: st.dev_n_comp[li],
            n_ix: st.dev_n_ix[li],
            comp_dst_off: cfg.window * hd + st.dev_n_comp[li] * ew_c,
            ix_dst_off: st.dev_n_ix[li] * ew_i,
            idx_cap: cfg.window
                + if l.indexer.is_some() {
                    cfg.index_topk
                } else {
                    st.dev_n_comp[li] + 2
                },
        };
        let nxt = layers.get(li + 1);
        let w = crate::gpu_wgpu::Dsv4LayerW {
            attn: crate::gpu_wgpu::Dsv4AttnW {
                wq_a, wq_b, wo_a, wo_b,
                q_norm: &l.q_norm,
                sink: &l.attn_sink,
            },
            moe: crate::gpu_wgpu::Dsv4MoeW {
                router: &packs[i].router,
                experts: &packs[i].tensors,
                logits: &[],
                // The PACK's slice, not a per-run Vec: the address stability
                // is the whole point (see Pack::bias).
                bias: packs[i].bias.as_deref(),
                forced: forceds[i].as_deref(),
                remap: None,
            },
            hc_ffn_fn: &l.hc_ffn_fn,
            hc_ffn_scale: &l.hc_ffn_scale,
            hc_ffn_base: &l.hc_ffn_base,
            hc_next_fn: nxt.map(|n| n.hc_attn_fn.as_slice()),
            hc_next_scale: nxt.map_or(&l.hc_attn_scale, |n| &n.hc_attn_scale),
            hc_next_base: nxt.map_or(&l.hc_attn_base, |n| n.hc_attn_base.as_slice()),
            ffn_norm: &l.ffn_norm,
            next_norm: nxt.map_or(&l.attn_norm, |n| n.attn_norm.as_slice()),
            next_q_norm: nxt.map_or(&l.q_norm, |n| n.q_norm.as_slice()),
            next_wq_a: nxt.and_then(|n| n.wq_a.model_idx()),
            router: &packs[i].router,
        };
        let geom = crate::gpu_wgpu::Dsv4LayerGeom {
            attn: crate::gpu_wgpu::Dsv4AttnGeom {
                dim,
                nh: cfg.n_heads,
                hd,
                rd: cfg.rope_head_dim,
                q_lora: cfg.q_lora_rank,
                o_lora: cfg.o_lora_rank,
                o_groups: cfg.o_groups,
                eps: cfg.norm_eps,
                scale: (hd as f32).powf(-0.5),
            },
            moe: crate::gpu_wgpu::Dsv4MoeGeom {
                hidden: dim,
                inter: cfg.moe_inter,
                top_k: cfg.top_k,
                route_scale: cfg.route_scale,
                swiglu_limit: cfg.swiglu_limit,
                gu_q2: l.experts.first().is_some_and(|e| {
                    e.w1.model_dtype() == Some(cortiq_core::TensorDtype::Q2TiledP)
                }),
            },
            hc: cfg.hc_mult,
            hc_eps: cfg.hc_eps,
            sinkhorn_iters: cfg.hc_sinkhorn_iters,
        };
        freqs.push(if l.compressor.is_some() {
            g.inv_freq_compress.as_slice()
        } else {
            g.inv_freq_window.as_slice()
        });
        items.push((w, geom, prep));
    }

    let mut out = vec![0.0f32; dim];
    if !crate::gpu_wgpu::dsv4_layer_chain(
        &model, &items, st.kv_id, first, &freqs, st.pos, &mut out, state_out,
    ) {
        return false;
    }
    *folded = out;
    // The device advanced these; the host keeps only the arithmetic.
    for (i, &li) in run.iter().enumerate() {
        st.dev_filled[li] = (st.dev_filled[li] + 1).min(cfg.window);
        if let Some((_, cg, ..)) = items[i].2.ix.as_ref() {
            if (st.pos + 1) % cg.ratio == 0 {
                st.dev_n_ix[li] += 1;
            }
        }
        if let Some((_, cg)) = items[i].2.comp.as_ref() {
            if (st.pos + 1) % cg.ratio == 0 {
                st.dev_n_comp[li] += 1;
            }
        }
    }
    st.dev_owned = true;
    true
}

/// `CMF_DSV4_HC_DEV=0` puts the hyper-connections back on the host.
#[cfg(feature = "gpu")]
fn hc_on_device() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        // OPT-IN. On the release checkpoint this path reads 3.234 against
        // the CPU's 3.282 — divergent — and the speed is unchanged, so there
        // is no trade to weigh: it must not be the default until it is
        // exact. The toy's near-agreement (129.787 vs 129.792) hid a real
        // fault the release exposes.
        std::env::var("CMF_DSV4_HC_DEV").is_ok_and(|v| v != "0")
            && crate::gpu::backend_available()
    })
}

/// The two-frame path with the hyper-connections on the card.
///
/// The host still prepares each layer's attention inputs — the compressor,
/// the indexer and the window, which are exact there — but it no longer
/// folds, Sinkhorns or norms, and it no longer carries the MoE half's input
/// between the halves: the attention frame leaves it on the device and the
/// MoE frame reads it from there. One readback a layer instead of two, and
/// 19 ms of host arithmetic a token gone.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn dsv4_two_frame_loop(
    state: &mut [f32],
    layers: &[Dsv4Layer],
    g: &Dsv4Globals,
    cfg: &Dsv4Cfg,
    st: &mut Dsv4State,
    token_id: u32,
    inv_freq: &[f32],
    pool: Option<&crate::pool::Pool>,
    scratch: &mut HcScratch,
) -> bool {
    let dim = cfg.dim;
    let freqs_of = |l: &Dsv4Layer| -> &[f32] {
        let f = if l.compressor.is_some() {
            &g.inv_freq_compress
        } else {
            &g.inv_freq_window
        };
        if f.is_empty() { inv_freq } else { f.as_slice() }
    };
    // Layer zero's fold has no frame before it, exactly as in the layer path.
    let (mut folded, post0, comb0) = hc_fold_norm(
        state,
        &layers[0].hc_attn_fn,
        &layers[0].hc_attn_scale,
        &layers[0].hc_attn_base,
        &layers[0].attn_norm,
        cfg,
        pool,
    );
    if !crate::gpu_wgpu::dsv4_state_write(state)
        || !crate::gpu_wgpu::dsv4_hc_write(&post0, &comb0)
    {
        return false;
    }
    // PRE-FLIGHT, before the first byte of state moves: a mid-loop refusal
    // would hand the token back to the ordinary loop AFTER these caches
    // advanced, and the second advance is not a slow answer but a wrong one.
    // The same discipline the layer loop states in the same words.
    let mut on_dev = vec![false; layers.len()];
    for (li, l) in layers.iter().enumerate() {
        let Some(pk) = pack_for(l, cfg, li) else {
            return false;
        };
        let Some(model) = l.experts.first().and_then(|e| e.w1.model_arc()) else {
            return false;
        };
        let gu_q2 = l
            .experts
            .first()
            .is_some_and(|e| e.w1.model_dtype() == Some(cortiq_core::TensorDtype::Q2TiledP));
        let attn_ok = [
            l.wq_a.model_idx(),
            l.wq_b.model_idx(),
            l.wo_a.model_idx(),
            l.wo_b.model_idx(),
        ]
        .into_iter()
        .flatten()
        .all(|i| crate::gpu_wgpu::dsv4_weight_ready(&model, i));
        on_dev[li] = attn_ok
            && pk.globals.len() == cfg.n_routed_experts
            && crate::gpu_wgpu::dsv4_experts_ready(&model, &pk.tensors, cfg.moe_inter, dim, gu_q2);
    }
    if !on_dev.iter().any(|&x| x) {
        return false;
    }
    let mut sink = vec![0.0f32; dim];
    for (li, l) in layers.iter().enumerate() {
        // A layer the card cannot hold runs on the host WHOLE, with the
        // state fetched and put back around it — the mixed ownership the
        // layer loop already proved out.
        if !on_dev[li] {
            if !crate::gpu_wgpu::dsv4_state_read(state) {
                return false;
            }
            let freqs = freqs_of(l);
            hc_block(
                state,
                &l.hc_attn_fn,
                &l.hc_attn_scale,
                &l.hc_attn_base,
                &l.attn_norm,
                cfg,
                scratch,
                pool,
                |f, o| attention_step(f, l, cfg, st, li, freqs, pool, None, o),
            );
            hc_block(
                state,
                &l.hc_ffn_fn,
                &l.hc_ffn_scale,
                &l.hc_ffn_base,
                &l.ffn_norm,
                cfg,
                scratch,
                pool,
                |f, o| moe_step(f, l, cfg, token_id, li, pool, o),
            );
            let nref = layers.get(li + 1).unwrap_or(l);
            let (f, p2, c2) = hc_fold_norm(
                state,
                &nref.hc_attn_fn,
                &nref.hc_attn_scale,
                &nref.hc_attn_base,
                &nref.attn_norm,
                cfg,
                pool,
            );
            folded = f;
            if !crate::gpu_wgpu::dsv4_hc_write(&p2, &c2)
                || !crate::gpu_wgpu::dsv4_state_write(state)
            {
                return false;
            }
            continue;
        }
        // The host's half: the caches and the attended list, untouched.
        let mut prep = AttnPrep::default();
        attention_step(
            &folded, l, cfg, st, li, freqs_of(l), pool, Some(&mut prep), &mut sink,
        );
        let hd = cfg.head_dim;
        let n_comp = st.compressed[li].len() / hd;
        let cap = (cfg.window + n_comp.next_power_of_two().max(64)) * hd;
        if !crate::gpu_wgpu::dsv4_cache_write(st.kv_id, li, 0, &st.window[li], cap)
            || (n_comp > 0
                && !crate::gpu_wgpu::dsv4_cache_write(
                    st.kv_id, li, cfg.window * hd, &st.compressed[li], cap,
                ))
        {
            return false;
        }
        let idx32: Vec<u32> = prep
            .idxs
            .iter()
            .map(|&p| {
                if p < prep.win_len {
                    p as u32
                } else {
                    (cfg.window + (p - prep.win_len)) as u32
                }
            })
            .collect();
        let nxt = layers.get(li + 1);
        let a_tail = crate::gpu_wgpu::Dsv4HcTail {
            fn_: &l.hc_ffn_fn,
            scale: &l.hc_ffn_scale,
            base: &l.hc_ffn_base,
            norm: &l.ffn_norm,
            hc: cfg.hc_mult,
            sinkhorn_iters: cfg.hc_sinkhorn_iters,
            hc_eps: cfg.hc_eps,
            eps: cfg.norm_eps,
        };
        let scale = (cfg.head_dim as f32).powf(-0.5);
        if !attn_frame(
            l, cfg, st, li, &prep.qr, &prep.idxs, freqs_of(l), st.pos, prep.win_len,
            scale, Some(&a_tail), &mut [],
        ) {
            return false;
        }
        let m_tail = nxt.map(|n| crate::gpu_wgpu::Dsv4HcTail {
            fn_: &n.hc_attn_fn,
            scale: &n.hc_attn_scale,
            base: &n.hc_attn_base,
            norm: &n.attn_norm,
            hc: cfg.hc_mult,
            sinkhorn_iters: cfg.hc_sinkhorn_iters,
            hc_eps: cfg.hc_eps,
            eps: cfg.norm_eps,
        });
        let mut next = vec![0.0f32; dim];
        let pair = m_tail
            .as_ref()
            .zip(nxt)
            .map(|(t, n)| (t, n.attn_norm.as_slice()));
        let forced = l
            .tid2eid
            .as_ref()
            .map(|tbl| hash_route(tbl, cfg.vocab, cfg.top_k, token_id));
        if !moe_frame(&[], l, cfg, li, &[], forced.as_deref(), pool, Some(&a_tail), pair, &mut next) {
            return false;
        }
        folded = next;
    }
    let _ = scratch;
    crate::gpu_wgpu::dsv4_state_read(state)
}

/// The host half of one hyper-connection block: mixes, Sinkhorn, fold, norm.
/// The device does this for every layer but the first, whose state it has not
/// seen yet.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn hc_fold_norm(
    state: &[f32],
    hc_fn: &[f32],
    hc_scale: &[f32; 3],
    hc_base: &[f32],
    norm_w: &[f32],
    cfg: &Dsv4Cfg,
    pool: Option<&crate::pool::Pool>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (hc, dim) = (cfg.hc_mult, cfg.dim);
    let mix_hc = (2 + hc) * hc;
    let mut mixes = vec![0.0f32; mix_hc];
    hc_mixes(state, hc_fn, mix_hc, cfg.norm_eps, pool, &mut mixes);
    let mut pre = vec![0.0f32; hc];
    let mut post = vec![0.0f32; hc];
    let mut comb = vec![0.0f32; hc * hc];
    hc_split_sinkhorn(
        &mixes,
        hc_scale,
        hc_base,
        hc,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
        &mut pre,
        &mut post,
        &mut comb,
    );
    let mut folded = vec![0.0f32; dim];
    hc_fold(state, &pre, hc, dim, &mut folded);
    rms_weighted(&mut folded, norm_w, cfg.norm_eps);
    // post and comb travel with the fold: the frame's opening expand needs
    // exactly those, and they are not recoverable from the state alone.
    (folded, post, comb)
}

/// `CMF_DSV4_GPU_LAYER=1`: one submission per layer instead of two, with the
/// hyper-connection glue and the router on the device.
///
/// CORRECT — perplexity 5.211 against the CPU's 5.211 on the release, 128.576
/// against 128.576 on the toy — and SLOWER on this hardware: 6.0 tok/s where
/// the two-frame path gets 9.3. The reason is not the frame, it is the
/// all-or-nothing granularity underneath it. A layer whose experts miss VRAM
/// runs entirely on the host, attention included (6.5 ms a call against 0.9),
/// and with 100 GB of experts against a 98 GB card a fifth of the layers
/// miss. The two-frame path only loses the MoE half of those layers.
///
/// So the barrier it saves is real and the fallback it forces costs more. The
/// fix is the granularity: pack the experts that FIT, route over all of them
/// anyway, and run the few cold picks of a token on the host — per EXPERT,
/// not per layer. Then no layer ever leaves the device and this frame wins by
/// the 15 ms a token it was built to save.
#[cfg(feature = "gpu")]
fn gpu_layer_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CMF_DSV4_GPU_LAYER").is_ok_and(|v| v != "0")
            && crate::gpu::backend_available()
    })
}

/// The packed expert set of one layer: which globals made it in, and their
/// directory indices in packing order with the shared expert last. Built once
/// — the mask does not change during a run — and keyed by layer.
#[cfg(feature = "gpu")]
struct Pack {
    /// The router as dense f32, expanded once. It is 4 MB a layer against a
    /// 112 GB model, it lives as long as the process — so the address-keyed
    /// device cache is sound for it, unlike anything built per call.
    router: Vec<f32>,
    /// global expert id -> packed slot, `usize::MAX` for the ones left out.
    to_slot: Vec<usize>,
    /// The same, as the u32 table the router reads.
    remap: Vec<u32>,
    /// packed order, globals only (shared is not in here).
    globals: Vec<usize>,
    tensors: Vec<(usize, usize, usize)>,
    /// The noaux_tc bias in PACKED order, kept here because it is the same
    /// every token and the pack lives as long as the process: a stable
    /// address means a stable device buffer, and a stable device buffer is
    /// what lets many layers share one submission. A bias uploaded through
    /// the per-call pool is written by every layer of a run BEFORE the run's
    /// single submit — queue writes do not interleave with passes — so every
    /// layer routed with the LAST layer's bias. On the release every scored
    /// layer carries one, which is the 50.280.
    bias: Option<Vec<f32>>,
}

#[cfg(feature = "gpu")]
fn pack_for(l: &Dsv4Layer, cfg: &Dsv4Cfg, li: usize) -> Option<std::sync::Arc<Pack>> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<usize, Option<Arc<Pack>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(v) = cache.lock().unwrap().get(&li) {
        return v.clone();
    }
    let build = || -> Option<Arc<Pack>> {
        let mut to_slot = vec![usize::MAX; cfg.n_routed_experts];
        let mut globals = Vec::new();
        let mut tensors = Vec::new();
        let idx3 = |e: &Dsv4Expert| -> Option<(usize, usize, usize)> {
            Some((e.w1.model_idx()?, e.w3.model_idx()?, e.w2.model_idx()?))
        };
        // How many experts the card still has room for, minus one for the
        // shared expert, which always rides. Everything past that stays on the
        // host and is reached through the remap — the router still ranges over
        // all of them, so this costs speed and not a single bit of quality.
        let gu_q2 = l
            .experts
            .first()
            .is_some_and(|e| e.w1.model_dtype() == Some(cortiq_core::TensorDtype::Q2TiledP));
        // `CMF_DSV4_COLD_CPU=1` packs only what fits and leaves the rest to
        // the host. Exact on the toy at any budget, still 5.379 against the
        // CPU's 5.211 on the release — so it is opt-in until that is closed.
        // Off, a layer that does not fit whole declines and runs on the host,
        // which is slower and right.
        // `CMF_DSV4_PACK_MAX=N` caps the packing directly, so a toy can
        // reproduce the subset path without needing a card that runs out.
        if let Some(n) = std::env::var("CMF_DSV4_PACK_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            let mut to_slot = vec![usize::MAX; cfg.n_routed_experts];
            let mut globals = Vec::new();
            let mut tensors = Vec::new();
            for (gi, e) in l.experts.iter().enumerate().take(n) {
                to_slot[gi] = globals.len();
                globals.push(gi);
                tensors.push(idx3(e)?);
            }
            tensors.push(idx3(&l.shared)?);
            let (rows, cols) = (l.gate.rows(), l.gate.cols());
            let mut router = vec![0.0f32; rows * cols];
            for r in 0..rows {
                l.gate.row_f32(r, &mut router[r * cols..(r + 1) * cols]);
            }
            let remap: Vec<u32> = to_slot
                .iter()
                .map(|&sl| if sl == usize::MAX { u32::MAX } else { sl as u32 })
                .collect();
            return Some(Arc::new(Pack {
                bias: l.gate_bias.as_deref().map(|b| {
                    globals.iter().map(|&g| b[g]).collect()
                }),
                router,
                to_slot,
                remap,
                globals,
                tensors,
            }));
        }
        let room = if std::env::var("CMF_DSV4_COLD_CPU").is_ok_and(|v| v != "0") {
            crate::gpu_wgpu::dsv4_experts_fit(cfg.moe_inter, cfg.dim, gu_q2).saturating_sub(1)
        } else {
            usize::MAX
        };
        for (gi, e) in l.experts.iter().enumerate() {
            if l.mask.as_deref().is_some_and(|m| !m.get(gi).copied().unwrap_or(true)) {
                continue;
            }
            if globals.len() >= room {
                break;
            }
            to_slot[gi] = globals.len();
            globals.push(gi);
            match idx3(e) {
                Some(t) => tensors.push(t),
                None => {
                    if std::env::var("CMF_DSV4_FRAME_DEBUG").is_ok() {
                        eprintln!("слой {li}: эксперт {gi} без индексов в каталоге");
                    }
                    return None;
                }
            }
        }
        if globals.is_empty() {
            // Two very different causes, and blaming the mask for the other
            // one sent a reader looking for a mask that was never set: an
            // actual empty mask, or a VRAM budget with no room left for even
            // one expert (`room` is 0, which is what a nearly-full card does
            // to the last layers).
            if room == 0 {
                tracing::warn!(
                    "слой {li}: в бюджете VRAM не осталось места ни под одного эксперта — \
                     слой уходит на CPU. Поднимите CMF_GPU_VRAM_MB"
                );
            } else {
                tracing::warn!("слой {li}: маска не оставила ни одного эксперта");
            }
            return None;
        }
        tensors.push(idx3(&l.shared)?); // shared rides last, as the kernels expect
        let (rows, cols) = (l.gate.rows(), l.gate.cols());
        let mut router = vec![0.0f32; rows * cols];
        for r in 0..rows {
            l.gate.row_f32(r, &mut router[r * cols..(r + 1) * cols]);
        }
        let remap: Vec<u32> = to_slot
            .iter()
            .map(|&sl| if sl == usize::MAX { u32::MAX } else { sl as u32 })
            .collect();
        Some(Arc::new(Pack {
            bias: l.gate_bias.as_deref().map(|b| {
                globals.iter().map(|&g| b[g]).collect()
            }),
            router,
            to_slot,
            remap,
            globals,
            tensors,
        }))
    };
    let v = build();
    cache.lock().unwrap().insert(li, v.clone());
    v
}

/// `CMF_DSV4_GPU_MOE2=1`: the whole MoE block in one submission, experts
/// resident. Returns false having changed nothing if it cannot.
///
/// NOT CORRECT YET — off by default and it must stay off. On the release
/// checkpoint it diverges from the CPU by up to 0.44 relative on most MoE
/// layers (`CMF_DSV4_MOE_CHECK=1` prints them), and perplexity lands at 5.162
/// against the CPU's 5.211 on the exact contract.
///
/// Ruled out so far, so the next attempt need not redo it:
///   * the routing kernel — `gpu_route_parity` matches exactly at 256 experts
///     with bias, mask, a hash row and an all-closed mask;
///   * the 2-bit expert layout — a q2tp toy (gate/up Q2TiledP, down Q4TiledP,
///     confirmed identical to the release by `expert_dtypes`) agrees bit for
///     bit, PPL 143.512 both ways;
///   * missing storage barriers in `moe_route` — added, no change.
///
/// So it is scale-dependent: the toy runs 8 experts, top_k 2, inter 64,
/// hidden 128; the release runs 256 / 6 / 2048 / 4096. The next thing to try
/// is a toy at release proportions, which will either reproduce it or narrow
/// it to something only the real weights do.
#[cfg(feature = "gpu")]
fn moe_frame(
    hidden: &[f32],
    l: &Dsv4Layer,
    cfg: &Dsv4Cfg,
    li: usize,
    logits: &[f32],
    forced: Option<&[usize]>,
    pool: Option<&crate::pool::Pool>,
    // The state handover: expand always when the device owns the state,
    // fold only when there is a next layer.
    hc_cur: Option<&crate::gpu_wgpu::Dsv4HcTail>,
    hc_next: Option<(&crate::gpu_wgpu::Dsv4HcTail, &[f32])>,
    out: &mut [f32],
) -> bool {
    macro_rules! no {
        ($($t:tt)*) => {{
            if std::env::var("CMF_DSV4_FRAME_DEBUG").is_ok() {
                eprintln!("кадр MoE отклонён: {}", format_args!($($t)*));
            }
            return false;
        }};
    }
    let Some(pk) = pack_for(l, cfg, li) else {
        no!("слой {li}: упаковка экспертов не построена");
    };
    // The router is a small f32 tensor and is usually NOT mapped; the handle
    // has to come from something that is.
    let Some(model) = l.experts.first().and_then(|e| e.w1.model_arc()) else {
        no!("слой {li}: эксперты не отображены из файла");
    };
    // A forced expert outside the packing has nowhere to go; the hash layers
    // name specific experts and a mask that drops one of them is a mask this
    // layer cannot use.
    let fpack: Option<Vec<usize>> = match forced {
        Some(f) => {
            let v: Vec<usize> = f.iter().map(|&g| pk.to_slot[g]).collect();
            if v.iter().any(|&s| s == usize::MAX) {
                no!("слой {li}: хеш-слой называет эксперта вне упаковки");
            }
            Some(v)
        }
        None => None,
    };
    // Routing ranges over EVERY expert; the remap turns a winner into a slot
    // or marks it cold. Nothing is masked, so nothing is lost.
    let subset = pk.globals.len() < cfg.n_routed_experts;
    // Empty logits are the device-scored case: the frame computes them from
    // pk.router, whose rows are already in global order, so there is nothing
    // to reorder — and indexing an empty slice is how this line greeted the
    // first engaged run.
    let lg: Vec<f32> = if logits.is_empty() || subset {
        logits.to_vec()
    } else {
        pk.globals.iter().map(|&g| logits[g]).collect()
    };
    let bias: Option<Vec<f32>> = l.gate_bias.as_deref().map(|b| {
        if subset {
            b.to_vec()
        } else {
            pk.globals.iter().map(|&g| b[g]).collect()
        }
    });
    let w = crate::gpu_wgpu::Dsv4MoeW {
        router: &pk.router,
        experts: &pk.tensors,
        logits: &lg,
        bias: bias.as_deref(),
        forced: fpack.as_deref(),
        remap: if subset { Some(&pk.remap) } else { None },
    };
    let g = crate::gpu_wgpu::Dsv4MoeGeom {
        hidden: cfg.dim,
        inter: cfg.moe_inter,
        top_k: cfg.top_k,
        route_scale: cfg.route_scale,
        swiglu_limit: cfg.swiglu_limit,
        gu_q2: l.experts.first().is_some_and(|e| {
            e.w1.model_dtype() == Some(cortiq_core::TensorDtype::Q2TiledP)
        }),
    };
    let mut cold = Vec::new();
    if !crate::gpu_wgpu::dsv4_moe_frame(&model, &w, g, hidden, &mut cold, hc_cur, hc_next, out) {
        return false;
    }
    // The picks the card had no room for, finished here and added in. Their
    // weights already carry the top-k normalisation the device applied.
    if std::env::var("CMF_DSV4_MOE_CHECK").is_ok() {
        let csum: f32 = cold.iter().map(|c| c.1).sum();
        eprintln!(
            "[холодные] слой {li}: вернулось {} из {} | сумма холодных {csum:.4} | \
             route_scale {:.4} | {:?}",
            cold.len(),
            cfg.top_k,
            cfg.route_scale,
            &cold[..cold.len().min(3)]
        );
    }
    let mut acc = vec![0.0f32; cfg.dim];
    for &(gi, wt) in &cold {
        let Some(exp) = l.experts.get(gi) else { continue };
        run_expert(hidden, exp, cfg, wt, pool, &mut acc);
        for (o, a) in out.iter_mut().zip(&acc) {
            *o += a;
        }
    }
    true
}

/// How much of each layer's compressed cache already sits on the card. ONE
/// map: a reader and a writer with a `static` each are two maps, and the
/// reader would never see a thing the writer put down.
/// The reallocation counter as of the last successful tail write. Any change
/// means some buffer was rebuilt and every tail count is stale.
#[cfg(feature = "gpu")]
fn last_grew(now: u64) -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEEN: AtomicU64 = AtomicU64::new(0);
    let was = SEEN.load(Ordering::Relaxed);
    if was != now {
        SEEN.store(now, Ordering::Relaxed);
        compressed_map().lock().unwrap().clear();
        return u64::MAX; // force a full write this round
    }
    now
}

#[cfg(feature = "gpu")]
fn compressed_map() -> &'static std::sync::Mutex<std::collections::HashMap<(u64, usize), usize>> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static W: OnceLock<Mutex<HashMap<(u64, usize), usize>>> = OnceLock::new();
    W.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(feature = "gpu")]
fn compressed_written(kv_id: u64, li: usize) -> usize {
    compressed_map()
        .lock()
        .unwrap()
        .get(&(kv_id, li))
        .copied()
        .unwrap_or(0)
}

/// Reset to zero whenever a write fails or the buffer grows — a grown buffer
/// keeps none of its contents.
#[cfg(feature = "gpu")]
fn note_compressed(kv_id: u64, li: usize, n: usize) {
    compressed_map().lock().unwrap().insert((kv_id, li), n);
}

#[cfg(feature = "gpu")]
fn gpu_moe2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CMF_DSV4_GPU_MOE2").is_ok_and(|v| v != "0")
            && crate::gpu::backend_available()
    })
}

pub fn moe_step(
    hidden: &[f32],
    l: &Dsv4Layer,
    cfg: &Dsv4Cfg,
    token_id: u32,
    // Layer index — only used to bucket routing statistics.
    li: usize,
    pool: Option<&crate::pool::Pool>,
    out: &mut [f32],
) {
    let _t0 = prof::on().then(std::time::Instant::now);
    let _guard = scopeguard_moe(_t0, li);
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
        l.mask.as_deref(),
        &mut idx,
        &mut w,
    );
    if route_stats_on() {
        record_route(li, 0, cfg.n_routed_experts, &idx);
    }
    // The whole block on the device, in one submission, or nothing. Routing
    // happens there too — the logits above are what it starts from, so the
    // CPU's own choice is discarded rather than second-guessed.
    #[cfg(feature = "gpu")]
    if gpu_moe2_enabled() {
        let forced = l
            .tid2eid
            .as_ref()
            .map(|tbl| hash_route(tbl, cfg.vocab, cfg.top_k, token_id));
        if moe_frame(hidden, l, cfg, li, &logits, forced.as_deref(), pool, None, None, out) {
            // CMF_DSV4_MOE_CHECK=1 recomputes the same block on the CPU and
            // reports where they part. A wrong MoE does not fail — it answers
            // differently — and the toy agreed bit for bit while the release
            // did not, so the difference lives in something the toy has no
            // instance of. Only a per-layer number will say which.
            if std::env::var("CMF_DSV4_MOE_CHECK").is_ok() {
                let mut want = vec![0.0f32; out.len()];
                let mut acc = vec![0.0f32; cfg.dim];
                for (e, &ei) in idx.iter().enumerate() {
                    let Some(exp) = l.experts.get(ei) else { continue };
                    run_expert(hidden, exp, cfg, w.get(e).copied().unwrap_or(0.0), pool, &mut acc);
                    for (o, a) in want.iter_mut().zip(&acc) {
                        *o += a;
                    }
                }
                run_expert(hidden, &l.shared, cfg, 1.0, pool, &mut acc);
                for (o, a) in want.iter_mut().zip(&acc) {
                    *o += a;
                }
                let num: f32 = want.iter().zip(out.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                let den: f32 = want.iter().map(|a| a * a).sum::<f32>().max(1e-20);
                let rel = (num / den).sqrt();
                if rel > 1e-3 {
                    let packed = pack_for(l, cfg, li).map_or(0, |p| p.globals.len());
                    eprintln!(
                        "[кадр MoE] слой {li}: расхождение {rel:.3e} | выбрано {} | \
                         упаковано {packed} из {} | хеш={} | смещение={}",
                        idx.len(),
                        cfg.n_routed_experts,
                        l.tid2eid.is_some(),
                        l.gate_bias.is_some()
                    );
                }
            }
            return;
        }
    }
    if dump_path().is_some() {
        PICKED.with(|p| {
            let mut p = p.borrow_mut();
            if p.len() <= li {
                p.resize(li + 1, Vec::new());
            }
            p[li] = idx.clone();
        });
    }
    // One submission for the whole block — the chosen experts plus the
    // shared one. Per-expert dispatches are what made MoE slow elsewhere,
    // and the device keeps the weights across tokens, so the cost is the
    // arithmetic rather than the traffic. A refusal (missing kernel, mixed
    // layouts, weights that do not fit the budget) falls to the CPU whole,
    // never half.
    // CORRECT but SLOWER, so off by default. Parity holds on real weights
    // (perplexity 6.808 → 6.839 at 64 tokens, 5.102 → 5.146 at 200), and an
    // honest alternating A/B says 1.0 tok/s against the CPU's 2.2. The first
    // measurement claimed the opposite — 0.7 → 2.0 — because the CPU arm ran
    // first and paged in 158 GB for the GPU arm to inherit.
    //
    // The cost is not arithmetic, it is round trips: this submits and reads
    // back once per layer, forty-three times a token, and a discrete card
    // charges milliseconds for each. Fixing it means one submission per
    // token — the whole-token graph — not a faster kernel.
    //
    // `CMF_DSV4_GPU_MOE=1` opts in.
    fn gpu_moe_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("CMF_DSV4_GPU_MOE").is_ok_and(|v| v != "0"))
    }
    if gpu_moe_on() && crate::gpu::enabled_here() {
        let mut jobs = Vec::with_capacity(idx.len() + 1);
        let mut model_ref = None;
        let mut ok = true;
        for (e, &ei) in idx.iter().enumerate() {
            let Some(exp) = l.experts.get(ei) else { continue };
            ok &= crate::pipeline::moe_push_job_parts(
                &exp.w1,
                &exp.w3,
                &exp.w2,
                hidden,
                w.get(e).copied().unwrap_or(0.0),
                cfg.swiglu_limit,
                &mut jobs,
                &mut model_ref,
            )
            .is_some();
        }
        ok &= crate::pipeline::moe_push_job_parts(
            &l.shared.w1,
            &l.shared.w3,
            &l.shared.w2,
            hidden,
            1.0,
            cfg.swiglu_limit,
            &mut jobs,
            &mut model_ref,
        )
        .is_some();
        if ok {
            if let Some(m) = model_ref.as_ref() {
                if crate::gpu::moe_block(m, &jobs, out) {
                    // CMF_DSV4_GPU_CHECK=1 recomputes the same block on the
                    // CPU and reports the divergence. A GPU MoE that is wrong
                    // does not fail — it answers differently — so the only way
                    // to know is to ask both.
                    if std::env::var("CMF_DSV4_GPU_CHECK").is_ok() {
                        let mut want = vec![0.0f32; out.len()];
                        let mut acc = vec![0.0f32; cfg.dim];
                        for (e, &ei) in idx.iter().enumerate() {
                            let Some(exp) = l.experts.get(ei) else { continue };
                            run_expert(
                                hidden, exp, cfg,
                                w.get(e).copied().unwrap_or(0.0), pool, &mut acc,
                            );
                            for (o, a) in want.iter_mut().zip(&acc) {
                                *o += a;
                            }
                        }
                        run_expert(hidden, &l.shared, cfg, 1.0, pool, &mut acc);
                        for (o, a) in want.iter_mut().zip(&acc) {
                            *o += a;
                        }
                        let num: f32 = want
                            .iter()
                            .zip(out.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum();
                        let den: f32 = want.iter().map(|a| a * a).sum::<f32>().max(1e-20);
                        eprintln!(
                            "[dsv4-gpu] слой {li}: расхождение {:.3e} | |CPU|={:.5} |GPU|={:.5} | экспертов {}",
                            (num / den).sqrt(),
                            den.sqrt(),
                            out.iter().map(|x| x * x).sum::<f32>().sqrt(),
                            jobs.len()
                        );
                    }
                    return;
                }
            }
        }
    }
    out.fill(0.0);
    let mut acc = vec![0.0f32; cfg.dim];
    for (e, &ei) in idx.iter().enumerate() {
        let Some(exp) = l.experts.get(ei) else {
            continue;
        };
        run_expert(
            hidden,
            exp,
            cfg,
            w.get(e).copied().unwrap_or(0.0),
            pool,
            &mut acc,
        );
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

/// `CMF_DSV4_TRACE=1` prints the hidden state's RMS after each half-block and
/// the logits' shape at the end. A 300B model that decodes nonsense gives no
/// other handle: this says whether the state grew, collapsed or went
/// non-finite, and at which layer — before anyone reaches for a debugger on a
/// hundred-gigabyte file.
fn no_compressed() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("CMF_DSV4_NO_COMPRESSED").is_ok_and(|v| v != "0"))
}

fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_DSV4_TRACE").is_ok_and(|v| v != "0"))
}

fn rms_of(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

/// `CMF_DSV4_DUMP=<path>` appends one JSON line per token: the embedding, the
/// hyper-connection state after every layer, the folded-and-normed head input
/// and the logits. It exists to be diffed against the reference forward on
/// the same weights — the numerical parity this port has never had, which at
/// toy scale is a few thousand floats and entirely tractable.
thread_local! {
    /// The attention body's input and output per layer, interleaved.
    static BODY: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Experts chosen per layer for the token being decoded — the dump needs
    /// them, because two implementations that pick DIFFERENT experts diverge
    /// hugely for a reason that is not a bug in either.
    static PICKED: std::cell::RefCell<Vec<Vec<usize>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn dump_path() -> Option<&'static str> {
    static P: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    P.get_or_init(|| std::env::var("CMF_DSV4_DUMP").ok())
        .as_deref()
}

fn dump_line(json: &str) {
    if let Some(p) = dump_path() {
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
        {
            let _ = writeln!(f, "{json}");
        }
    }
}

fn vec_json(v: &[f32]) -> String {
    let mut s = String::with_capacity(v.len() * 9);
    s.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{x:.6e}"));
    }
    s.push(']');
    s
}

/// One token through the whole stack.
///
/// The hidden state is `hc_mult` copies of a `dim`-vector from the very
/// first line to the very last: the embedding is replicated, every layer
/// folds/expands around its two halves, and only `hc_head_fold` collapses
/// it before the output norm and the head. There is no point in this
/// function where an ordinary residual would fit.
#[allow(clippy::too_many_arguments)]
/// A chunk of prompt tokens. Stage one of the batched prefill (see
/// docs/DSV4_PREFILL.md): the walk itself, with the head skipped for every
/// token but the last.
///
/// Prefill costs `len × per-token` today, and on a 2500-token prompt that is
/// a minute and a half before the first word. The stages that follow batch
/// the weight reads — which is where the nine-fold gap to the bandwidth
/// floor lives — but this one is the scaffolding they hang on, and it
/// already stops computing 129 280 logits for tokens nobody asks about.
#[allow(clippy::too_many_arguments)]
pub fn forward_chunk(
    g: &Dsv4Globals,
    layers: &[Dsv4Layer],
    cfg: &Dsv4Cfg,
    st: &mut Dsv4State,
    ids: &[u32],
    pos0: usize,
    inv_freq: &[f32],
    pool: Option<&crate::pool::Pool>,
    logits: &mut Vec<f32>,
) {
    for (i, &id) in ids.iter().enumerate() {
        st.pos = pos0 + i;
        let last = i + 1 == ids.len();
        forward_token_inner(g, layers, cfg, st, id, inv_freq, pool, logits, last);
    }
}

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
    forward_token_inner(g, layers, cfg, st, token_id, inv_freq, pool, logits, true);
}

#[allow(clippy::too_many_arguments)]
fn forward_token_inner(
    g: &Dsv4Globals,
    layers: &[Dsv4Layer],
    cfg: &Dsv4Cfg,
    st: &mut Dsv4State,
    token_id: u32,
    inv_freq: &[f32],
    pool: Option<&crate::pool::Pool>,
    logits: &mut Vec<f32>,
    // Prompt tokens other than the last one have their logits thrown away.
    want_logits: bool,
) {
    let _t_all = prof::on().then(std::time::Instant::now);
    let _all_guard = Charge(_t_all, &prof::ALL_NS);
    let (hc, dim) = (cfg.hc_mult, cfg.dim);

    // Embedding, replicated into the copies.
    let mut emb = vec![0.0f32; dim];
    g.embed.row_f32(token_id as usize, &mut emb);
    let mut state = vec![0.0f32; hc * dim];
    for j in 0..hc {
        state[j * dim..(j + 1) * dim].copy_from_slice(&emb);
    }

    let mut scratch = HcScratch::new(cfg);
    let mut dump: Vec<String> = Vec::new();
    if dump_path().is_some() {
        dump.push(format!("\"embed\":{}", vec_json(&emb)));
        PICKED.with(|p| p.borrow_mut().clear());
        BODY.with(|b| b.borrow_mut().clear());
        dump.push(",\"layers\":[".into());
    }
    if trace_on() {
        eprintln!(
            "[dsv4] tok={token_id} pos={} embed rms={:.5}",
            st.pos,
            rms_of(&emb)
        );
    }
    // ── one submission per layer, when the device will take it ──
    #[cfg(feature = "gpu")]
    let layer_frames = gpu_layer_enabled()
        && dsv4_layer_loop(
            &mut state, layers, g, cfg, st, token_id, inv_freq, pool, &mut scratch,
        );
    #[cfg(not(feature = "gpu"))]
    let layer_frames = false;

    // ── the fast two-frame path: hyper-connections on the card ──
    // Measured on the release, the fold, the Sinkhorn and the norms cost 19
    // ms of a 57 ms token on the host and hundredths of one on the device.
    // With both frames doing their own, the host carries nothing between a
    // layer's halves and the MoE half's input never leaves the card — one
    // readback a layer instead of two.
    #[cfg(feature = "gpu")]
    let hc_dev = hc_on_device()
        && !layer_frames
        && gpu_attn_enabled()
        && gpu_moe2_enabled()
        && dump_path().is_none();
    #[cfg(not(feature = "gpu"))]
    let hc_dev = false;
    // The device loop's verdict as a VALUE, not as a cfg-gated `if`. It used
    // to be the latter, with the CPU loop in the `else` arm — so a build
    // without the gpu feature compiled no layer loop at all and every token
    // passed through untouched. The window test said so ("sliding window
    // never filled") and only in the CPU-only build, which is the one
    // configuration the gate was not running.
    #[cfg(feature = "gpu")]
    let two_frame_done = hc_dev
        && dsv4_two_frame_loop(
            &mut state, layers, g, cfg, st, token_id, inv_freq, pool, &mut scratch,
        );
    #[cfg(not(feature = "gpu"))]
    let two_frame_done = false;
    if !two_frame_done {

    for (li, l) in layers.iter().enumerate() {
        if layer_frames {
            break;
        }
        // attention half
        hc_block(
            &mut state,
            &l.hc_attn_fn,
            &l.hc_attn_scale,
            &l.hc_attn_base,
            &l.attn_norm,
            cfg,
            &mut scratch,
            pool,
            |folded, out| {
                if dump_path().is_some() {
                    // The body's own input and output, so the reference can be
                    // fed the port's input: then only the body can differ.
                    BODY.with(|b| b.borrow_mut().push(vec_json(folded)));
                }
                // The layer's kind decides its frequencies, not the model's.
                let freqs = if l.compressor.is_some() {
                    &g.inv_freq_compress
                } else {
                    &g.inv_freq_window
                };
                let freqs = if freqs.is_empty() {
                    inv_freq
                } else {
                    freqs.as_slice()
                };
                attention_step(folded, l, cfg, st, li, freqs, pool, None, out);
                if dump_path().is_some() {
                    BODY.with(|b| b.borrow_mut().push(vec_json(out)));
                }
            },
        );
        if dump_path().is_some() {
            // After the attention half only — this is what separates an
            // attention discrepancy from an expert one.
            dump.push(format!(
                "{}{}",
                if li == 0 { "" } else { "," },
                vec_json(&state)
            ));
        }
        // FFN half
        let _t_hc2 = prof::on().then(std::time::Instant::now);
        hc_block(
            &mut state,
            &l.hc_ffn_fn,
            &l.hc_ffn_scale,
            &l.hc_ffn_base,
            &l.ffn_norm,
            cfg,
            &mut scratch,
            pool,
            |folded, out| moe_step(folded, l, cfg, token_id, li, pool, out),
        );
        if let Some(t) = _t_hc2 {
            // The block's own time minus the expert step inside it — what the
            // fold, the norm and the expand cost on their own.
            prof::HC_NS.fetch_add(
                t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        if dump_path().is_some() {
            dump.push(format!(",{}", vec_json(&state)));
        }
        if trace_on() && (st.pos % 64 == 0 || st.pos == 199) {
            eprintln!(
                "[dsv4]  кеши слоя {li}: окно={} сжатых={} индекс={} (ratio={:?})",
                st.window[li].len() / cfg.head_dim.max(1),
                st.compressed[li].len() / cfg.head_dim.max(1),
                st.index_kv[li].len().max(1) / 128,
                l.compressor.as_ref().map(|c| (c.ratio, c.overlap)),
            );
        }
        if trace_on() {
            let bad = state.iter().filter(|v| !v.is_finite()).count();
            eprintln!(
                "[dsv4]  layer {li:>2}: rms={:.5}{}",
                rms_of(&state),
                if bad > 0 {
                    format!("  NON-FINITE x{bad}")
                } else {
                    String::new()
                }
            );
        }
    }
    }
    st.pos += 1;

    // Collapse the copies, normalize, project to the vocabulary.
    let mut h = vec![0.0f32; dim];
    hc_head_fold(
        &state,
        &g.hc_head_fn,
        g.hc_head_scale,
        &g.hc_head_base,
        cfg,
        pool,
        &mut h,
    );
    if !want_logits {
        logits.clear();
        return;
    }
    let _t_head = prof::on().then(std::time::Instant::now);
    rms_weighted(&mut h, &g.norm, cfg.norm_eps);
    logits.clear();
    logits.resize(g.head.rows(), 0.0);
    g.head.matvec(&h, logits, pool);
    if let Some(t) = _t_head {
        prof::HEAD_NS.fetch_add(
            t.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    if dump_path().is_some() {
        dump.push("]".into());
        let picked = PICKED.with(|p| {
            p.borrow()
                .iter()
                .map(|v| {
                    format!(
                        "[{}]",
                        v.iter()
                            .map(|e| e.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        });
        dump.push(format!(",\"experts\":[{picked}]"));
        let body = BODY.with(|b| b.borrow().join(","));
        dump.push(format!(",\"attn_io\":[{body}]"));
        dump_line(&format!(
            "{{\"tok\":{token_id},\"pos\":{},{},\"head\":{},\"logits\":{}}}",
            st.pos - 1,
            dump.join(""),
            vec_json(&h),
            vec_json(logits)
        ));
    }
    if trace_on() {
        let (mut top, mut best) = (0usize, f32::NEG_INFINITY);
        for (i, &v) in logits.iter().enumerate() {
            if v > best {
                best = v;
                top = i;
            }
        }
        let lo = logits.iter().cloned().fold(f32::MAX, f32::min);
        eprintln!(
            "[dsv4]  head: rms={:.5} logits[{}..{:.3}] argmax={top}",
            rms_of(&h),
            format_args!("{lo:.3}"),
            best
        );
    }
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
    // The small pieces — norms, the sink, ape, the hyper-connection
    // projections — are read as plain f32. They are not all 2-D (a norm is a
    // vector), so this cannot go through QTensor, which requires a matrix.
    let f = |name: &str| -> Result<Vec<f32>, String> {
        crate::loader::load_f32(model, name, &crate::loader::Overlay::None)
    };

    // Two frequency tables, chosen per layer by whether it compresses. The
    // release's compress_rope_theta (160 000) is not in config.json — it
    // lives in inference/config.json — so it is pinned here with the other
    // constants the header cannot carry.
    let rope_of = |base: f32, yarn: bool| -> Vec<f32> {
        if yarn {
            crate::attention::yarn_inv_freq(cfg.rope_head_dim, base, 16.0, 65536, 32.0, 1.0)
        } else {
            crate::attention::rope_inv_freq(cfg.rope_head_dim, base)
        }
    };
    let globals = Dsv4Globals {
        inv_freq_compress: rope_of(160_000.0, true),
        inv_freq_window: rope_of(10_000.0, false),
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
        layers.push(load_layer(
            model,
            cfg,
            &format!("model.layers.{li}"),
            Scheme::Main,
        )?);
    }
    Ok((globals, layers))
}

/// Where a layer's tensors live in the file.
///
/// The MTP modules are the same layer as any other — attention, a
/// hyper-connection pair, a gated MoE over 256 experts — but the converter
/// wrote them under DeepSeek's internal names rather than the HF ones it used
/// for the trunk. Two schemes, one loader: a second copy would drift.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scheme {
    Main,
    Mtp,
}

impl Scheme {
    fn attn(self) -> &'static str {
        match self {
            Scheme::Main => "self_attn",
            Scheme::Mtp => "attn",
        }
    }
    fn attn_norm(self) -> &'static str {
        match self {
            Scheme::Main => "input_layernorm.weight",
            Scheme::Mtp => "attn_norm.weight",
        }
    }
    fn ffn_norm(self) -> &'static str {
        match self {
            Scheme::Main => "post_attention_layernorm.weight",
            Scheme::Mtp => "ffn_norm.weight",
        }
    }
    fn mlp(self) -> &'static str {
        match self {
            Scheme::Main => "mlp",
            Scheme::Mtp => "ffn",
        }
    }
    /// The router's per-expert bias. Absent on the trunk's hash layers, which
    /// is how they are recognised; always present on an MTP module.
    fn gate_bias(self) -> &'static str {
        match self {
            Scheme::Main => "expert_bias",
            Scheme::Mtp => "gate.bias",
        }
    }
    fn shared(self) -> &'static str {
        match self {
            Scheme::Main => "shared_expert",
            Scheme::Mtp => "shared_experts",
        }
    }
    /// gate, down, up — in that order, which is w1/w2/w3 upstream.
    fn w(self, i: u8) -> &'static str {
        match (self, i) {
            (Scheme::Main, 1) => "gate_proj.weight",
            (Scheme::Main, 2) => "down_proj.weight",
            (Scheme::Main, _) => "up_proj.weight",
            (Scheme::Mtp, 1) => "w1.weight",
            (Scheme::Mtp, 2) => "w2.weight",
            (Scheme::Mtp, _) => "w3.weight",
        }
    }
}

/// One layer, wherever it lives in the file.
pub fn load_layer(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    cfg: &Dsv4Cfg,
    p: &str,
    s: Scheme,
) -> Result<Dsv4Layer, String> {
    let q = |name: &str| -> Result<crate::qtensor::QTensor, String> {
        crate::qtensor::QTensor::from_model(model, name)
    };
    let f = |name: &str| -> Result<Vec<f32>, String> {
        crate::loader::load_f32(model, name, &crate::loader::Overlay::None)
    };
    let opt_f = |name: &str| -> Option<Vec<f32>> { f(name).ok() };
    let at = s.attn();
    let ml = s.mlp();
    {
        let scale3 = |name: &str| -> Result<[f32; 3], String> {
            let v = f(name)?;
            if v.len() < 3 {
                return Err(format!("{name}: expected 3 scales, got {}", v.len()));
            }
            Ok([v[0], v[1], v[2]])
        };
        // The compressor exists on every layer whose ratio is non-zero;
        // its presence in the file is the only signal we need.
        let compressor = match q(&format!("{p}.{at}.compressor.wkv.weight")) {
            Ok(wkv) => {
                let ape = f(&format!("{p}.{at}.compressor.ape"))?;
                // ape is [ratio, coff*head_dim]; coff is 2 when the windows
                // overlap, which the release does at ratio 4.
                let width = wkv.rows();
                let ratio = (ape.len() / width.max(1)).max(1);
                Some(Dsv4Compressor {
                    wkv,
                    wgate: q(&format!("{p}.{at}.compressor.wgate.weight"))?,
                    norm: f(&format!("{p}.{at}.compressor.norm.weight"))?,
                    ape,
                    ratio,
                    overlap: ratio == 4,
                })
            }
            Err(_) => None,
        };
        let indexer = match q(&format!("{p}.{at}.indexer.wq_b.weight")) {
            Ok(wq_b) => {
                let ape = f(&format!("{p}.{at}.indexer.compressor.ape"))?;
                let cwkv = q(&format!("{p}.{at}.indexer.compressor.wkv.weight"))?;
                let width = cwkv.rows();
                let ratio = (ape.len() / width.max(1)).max(1);
                Some(Dsv4Indexer {
                    wq_b,
                    weights_proj: q(&format!("{p}.{at}.indexer.weights_proj.weight"))?,
                    compressor: Dsv4Compressor {
                        wkv: cwkv,
                        wgate: q(&format!("{p}.{at}.indexer.compressor.wgate.weight"))?,
                        norm: f(&format!("{p}.{at}.indexer.compressor.norm.weight"))?,
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
            let ep = format!("{p}.{ml}.experts.{e}");
            experts.push(Dsv4Expert {
                w1: q(&format!("{ep}.{w}", w = s.w(1)))?,
                w2: q(&format!("{ep}.{w}", w = s.w(2)))?,
                w3: q(&format!("{ep}.{w}", w = s.w(3)))?,
            });
        }

        Ok(Dsv4Layer {
            attn_norm: f(&format!("{p}.{an}", an = s.attn_norm()))?,
            ffn_norm: f(&format!("{p}.{fnm}", fnm = s.ffn_norm()))?,
            wq_a: q(&format!("{p}.{at}.wq_a.weight"))?,
            q_norm: f(&format!("{p}.{at}.q_norm.weight"))?,
            wq_b: q(&format!("{p}.{at}.wq_b.weight"))?,
            wkv: q(&format!("{p}.{at}.wkv.weight"))?,
            kv_norm: f(&format!("{p}.{at}.kv_norm.weight"))?,
            wo_a: q(&format!("{p}.{at}.wo_a.weight"))?,
            wo_b: q(&format!("{p}.{at}.wo_b.weight"))?,
            attn_sink: f(&format!("{p}.{at}.attn_sink"))?,
            compressor,
            indexer,
            hc_attn_fn: f(&format!("{p}.hc_attn_fn"))?,
            hc_attn_base: f(&format!("{p}.hc_attn_base"))?,
            hc_attn_scale: scale3(&format!("{p}.hc_attn_scale"))?,
            hc_ffn_fn: f(&format!("{p}.hc_ffn_fn"))?,
            hc_ffn_base: f(&format!("{p}.hc_ffn_base"))?,
            hc_ffn_scale: scale3(&format!("{p}.hc_ffn_scale"))?,
            gate: q(&format!("{p}.{ml}.gate.weight"))?,
            // The bias is absent exactly on the hash layers, and the table
            // is present exactly there — the file itself says which is which.
            gate_bias: opt_f(&format!("{p}.{ml}.{b}", b = s.gate_bias())),
            tid2eid: opt_f(&format!("{p}.{ml}.tid2eid")),
            experts,
            mask: if model.tensor(&format!("{p}.{ml}.tid2eid")).is_some() {
                None
            } else {
                crate::loader::moe_task_mask(&format!("{p}."), cfg.n_routed_experts)
            },
            shared: Dsv4Expert {
                w1: q(&format!("{p}.{ml}.{sh}.{w}", sh = s.shared(), w = s.w(1)))?,
                w2: q(&format!("{p}.{ml}.{sh}.{w}", sh = s.shared(), w = s.w(2)))?,
                w3: q(&format!("{p}.{ml}.{sh}.{w}", sh = s.shared(), w = s.w(3)))?,
            },
        })
    }
}

/// One module of the speculation stack.
///
/// The release carries three, so the draft is three deep, and the last one
/// also holds a confidence head — the model scores its own proposals rather
/// than leaving acceptance to a threshold we would have to invent. Each
/// module is a full layer with its own 256 experts; what makes it an MTP
/// module rather than a 44th layer is `main_proj`, which folds the previous
/// hidden state into the next embedding before the layer runs.
pub struct Dsv4Mtp {
    pub layer: Dsv4Layer,
    /// Stage 0 only: the projection that turns the trunk's captured hidden
    /// states into the block's input. Later stages take the block from the
    /// stage before them, so they carry none.
    pub main_proj: Option<crate::qtensor::QTensor>,
    pub main_norm: Option<Vec<f32>>,
    /// Last module only: what turns a draft hidden state into logits.
    pub norm: Option<Vec<f32>>,
    pub hc_head_fn: Option<Vec<f32>>,
    pub hc_head_base: Option<Vec<f32>>,
    pub hc_head_scale: Option<f32>,
    pub confidence: Option<crate::qtensor::QTensor>,
    /// Last stage only: a rank-256 bigram table that biases the draft's
    /// logits, and whose embedding also feeds the confidence head. Cheap
    /// enough that the draft samples through it position by position while
    /// the network itself runs the whole block at once.
    pub markov_w1: Option<crate::qtensor::QTensor>,
    pub markov_w2: Option<crate::qtensor::QTensor>,
}

/// Load as much of the speculation stack as the file carries, up to
/// `max_depth`. Missing is not an error: a checkpoint without MTP simply
/// yields an empty stack, and the caller falls back to plain decoding.
pub fn load_mtp(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    cfg: &Dsv4Cfg,
    max_depth: usize,
) -> Vec<Dsv4Mtp> {
    let f = |name: &str| -> Option<Vec<f32>> {
        crate::loader::load_f32(model, name, &crate::loader::Overlay::None).ok()
    };
    let mut out = Vec::new();
    for d in 0..max_depth {
        let p = format!("model.mtp.{d}");
        // A stage is recognised by its attention, not by `main_proj`: only
        // stage 0 has that, and only the last has the head. Keying on either
        // end found one module of three.
        if model.tensor(&format!("{p}.attn.wq_a.weight")).is_none() {
            break;
        }
        let layer = match load_layer(model, cfg, &p, Scheme::Mtp) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("MTP {d}: пропущен, {e}");
                break;
            }
        };
        out.push(Dsv4Mtp {
            layer,
            main_proj: crate::qtensor::QTensor::from_model(model, &format!("{p}.main_proj.weight"))
                .ok(),
            main_norm: f(&format!("{p}.main_norm.weight")),
            norm: f(&format!("{p}.norm.weight")),
            hc_head_fn: f(&format!("{p}.hc_head_fn")),
            hc_head_base: f(&format!("{p}.hc_head_base")),
            hc_head_scale: f(&format!("{p}.hc_head_scale")).and_then(|v| v.first().copied()),
            confidence: crate::qtensor::QTensor::from_model(
                model,
                &format!("{p}.confidence_head.proj.weight"),
            )
            .ok(),
            markov_w1: crate::qtensor::QTensor::from_model(
                model,
                &format!("{p}.markov_head.markov_w1.weight"),
            )
            .ok(),
            markov_w2: crate::qtensor::QTensor::from_model(
                model,
                &format!("{p}.markov_head.markov_w2.weight"),
            )
            .ok(),
        });
    }
    if !out.is_empty() {
        let mp = out
            .iter()
            .find_map(|m| m.main_proj.as_ref())
            .map(|t| format!("[{}, {}]", t.rows(), t.cols()))
            .unwrap_or_else(|| "нет".into());
        eprintln!(
            "MTP: {} стади(я/и/й), main_proj {mp}, экспертов {}, \
             голова уверенности {}, марков {}",
            out.len(),
            out[0].layer.experts.len(),
            if out.iter().any(|m| m.confidence.is_some()) { "есть" } else { "нет" },
            if out.iter().any(|m| m.markov_w1.is_some()) { "есть" } else { "нет" },
        );
    }
    out
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
        let t = |rows: usize, cols: usize, seed: usize| {
            QTensor::from_f32(w(rows * cols, seed), rows, cols)
        };
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
                indexer: if li == 1 {
                    Some(Dsv4Indexer {
                        wq_b: t(2 * 16, cfg.q_lora_rank, 41),
                        weights_proj: t(2, dim, 43),
                        compressor: Dsv4Compressor {
                            wkv: t(2 * 16, dim, 45),
                            wgate: t(2 * 16, dim, 47),
                            norm: ones(16),
                            ape: vec![0.01; 4 * 2 * 16],
                            ratio: 4,
                            overlap: true,
                        },
                    })
                } else {
                    None
                },
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
                mask: None,
                shared: Dsv4Expert {
                    w1: t(cfg.moe_inter, dim, 25 + li),
                    w2: t(dim, cfg.moe_inter, 27 + li),
                    w3: t(cfg.moe_inter, dim, 29 + li),
                },
            });
        }
        let inv = |base: f32| -> Vec<f32> {
            (0..cfg.rope_head_dim / 2)
                .map(|i| 1.0 / base.powf(2.0 * i as f32 / cfg.rope_head_dim as f32))
                .collect()
        };
        let g = Dsv4Globals {
            inv_freq_compress: inv(160000.0),
            inv_freq_window: inv(10000.0),
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
            forward_token(
                &g,
                &layers,
                &cfg,
                &mut st,
                tok,
                &inv_freq,
                None,
                &mut logits,
            );
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
        // Every layer that HAS an indexer must have filled the indexer's own
        // cache: it is what decides which compressed positions attention
        // reads, and an empty one silently discards the whole long-range
        // memory rather than failing.
        for (li, l) in layers.iter().enumerate() {
            if l.indexer.is_some() {
                assert!(
                    !st.index_kv[li].is_empty(),
                    "layer {li} has an indexer but its cache stayed empty"
                );
            }
        }

        // Context must matter: the same token at position 0 of a fresh state
        // and at the end of a filled one cannot give identical logits.
        let mut fresh = Dsv4State::new(layers.len());
        let mut relogits = Vec::new();
        forward_token(
            &g,
            &layers,
            &cfg,
            &mut fresh,
            3,
            &inv_freq,
            None,
            &mut relogits,
        );
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
        assert!(
            (raw[1] - silu(50.0) * -50.0).abs() < 1e-3,
            "limit 0 must not clamp"
        );
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
        o_project(
            &attn,
            &row,
            per_group,
            &project,
            groups,
            lora,
            None,
            &mut serial,
        );

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
        assert!(
            serial.iter().any(|v| v.abs() > 1e-6),
            "test data is degenerate"
        );
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
        assert!(
            first.iter().all(|v| v.is_finite()),
            "first window: {first:?}"
        );
        assert!((first[0] - 40.0).abs() < 1e-3, "first dim0 = {}", first[0]);

        // And a previous window with real scores does pull the result.
        let mut both = vec![0.0f32; d];
        let strong_prev = vec![100.0f32; ratio * 2 * d];
        compress_window_overlap(
            &prev_kv,
            &strong_prev,
            &cur_kv,
            &cur_sc,
            ratio,
            d,
            &mut both,
        );
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
        let want_pre = [0.7310596, 0.8888268, 0.9525191, 0.97424865];
        let want_post = [1.9600224, 1.9534285, 1.9201256, 1.8160983];
        let want_comb = [
            0.5996052,
            0.28253591,
            0.09218107,
            0.025676856,
            0.17564717,
            0.22228767,
            0.27174541,
            0.33031881,
            0.029528176,
            0.12206022,
            0.32619134,
            0.5222193,
            0.19521846,
            0.37311527,
            0.30988118,
            0.12178412,
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
        route(&scores, Some(&bias), 2, 1.5, None, None, &mut idx, &mut w);
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
        sparse_attend(
            &q,
            &kv,
            &[0, usize::MAX],
            f32::NEG_INFINITY,
            1.0,
            hd,
            &mut b,
        );
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
        assert!(
            (out[0] - 2.0).abs() < 1e-5,
            "equal scores average: {}",
            out[0]
        );
        assert!(
            (out[1] - 20.0).abs() < 1e-3,
            "a dominant score wins: {}",
            out[1]
        );
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
        index_scores(&q, &kv, &w, nh, hd, 2, 2, None, &mut sc);
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
        index_scores(&q, &kv, &w, nh, hd, 3, 2, None, &mut sc);
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
            None,
            |_folded, out: &mut [f32]| out.iter_mut().for_each(|o| *o = 1.0),
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

    /// A task mask restricts SELECTION and nothing else: the weights still
    /// come from the pre-bias scores and still renormalize, now over what
    /// survives. Masking must never reroute — an expert the mask forbids has
    /// to be absent, not replaced by a neighbour with the wrong weight.
    #[test]
    fn a_task_mask_restricts_selection_and_renormalizes() {
        // Expert 3 scores highest, then 1, then 2, then 0.
        let scores = [0.1f32, 4.0, 1.0, 9.0];
        let (mut idx, mut w) = (Vec::new(), Vec::new());
        route(&scores, None, 2, 1.0, None, None, &mut idx, &mut w);
        assert_eq!(idx, vec![3, 1], "unmasked: the two best win");
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights must sum to route_scale");

        // Forbid the winner: the next two take its place and the weights
        // renormalize over them.
        let mask = [true, false, true, true];
        let (mut i2, mut w2) = (Vec::new(), Vec::new());
        route(&scores, None, 2, 1.0, None, Some(&mask), &mut i2, &mut w2);
        assert_eq!(i2, vec![3, 2], "masked expert must not be selected");
        let sum2: f32 = w2.iter().sum();
        assert!((sum2 - 1.0).abs() < 1e-5, "masked weights must renormalize");

        // A mask leaving fewer than top_k experts yields fewer, not garbage.
        let tight = [false, false, false, true];
        let (mut i3, mut w3) = (Vec::new(), Vec::new());
        route(&scores, None, 2, 1.0, None, Some(&tight), &mut i3, &mut w3);
        assert_eq!(i3, vec![3]);
        assert_eq!(w3.len(), 1);
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
        route(
            &scores,
            None,
            2,
            1.0,
            Some(&idx_forced),
            None,
            &mut idx,
            &mut w,
        );
        assert_eq!(idx, vec![0, 1], "the table must decide the experts");

        // The weights must be the table experts' own scores, normalized.
        let sp = |x: f32| (1.0 + x.exp()).ln().sqrt();
        let (s0, s1) = (sp(scores[0]), sp(scores[1]));
        let tot = s0 + s1;
        assert!(
            (w[0] - s0 / tot).abs() < 1e-6,
            "w[0]={} want {}",
            w[0],
            s0 / tot
        );
        assert!(
            (w[1] - s1 / tot).abs() < 1e-6,
            "w[1]={} want {}",
            w[1],
            s1 / tot
        );

        // And the top-k path is untouched: expert 3 still wins there.
        let (mut idx2, mut w2) = (Vec::new(), Vec::new());
        route(&scores, None, 2, 1.0, None, None, &mut idx2, &mut w2);
        assert_eq!(idx2[0], 3, "without a table the highest score still wins");
    }
}

