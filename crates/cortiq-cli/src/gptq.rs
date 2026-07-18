//! Layer-wise GPTQ — the **holographic transfer** for the `q1s` codec.
//!
//! Given a weight matrix `W [out, in]` and the calibration Hessian
//! `H = X·Xᵀ [in, in]` of its input activations, quantize the input
//! channels left-to-right and fold each column's rounding residual into
//! the not-yet-quantized columns through `H⁻¹` — the OBS/GPTQ update
//! `Π = Σ_PS·Σ_SS⁻¹`. That preserves the layer OUTPUT `W·x` over the
//! calibration distribution, not the weights, which is the only thing that
//! survives 1-bit (weight-space error diffusion / masking do not — measured).
//!
//! Emits the same `Q1S` bytes as `convert::encode_q1s` (a `q1` base with a
//! sparse f16 outlier overlay); the whole difference is *which* signs,
//! scales, and folded corrections the error-aware pass chooses.

use cortiq_core::quant::{f16_to_f32, f32_to_f16};

const GROUP_SIZE: usize = 32;

/// Read a calibration corpus: a `.json` array of `[prompt, text]` pairs
/// (the DTG-MA cache — texts concatenated) or a plain text file. Capped at
/// `budget_chars` so tokenization stays bounded.
fn read_calib_text(path: &str, budget_chars: usize) -> anyhow::Result<String> {
    let raw = std::fs::read_to_string(path)?;
    if !path.ends_with(".json") {
        return Ok(raw);
    }
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let mut out = String::new();
    if let Some(arr) = v.as_array() {
        for item in arr {
            match item {
                serde_json::Value::Array(pair) => {
                    for s in pair {
                        if let Some(t) = s.as_str() {
                            out.push_str(t);
                            out.push('\n');
                        }
                    }
                }
                serde_json::Value::String(s) => {
                    out.push_str(s);
                    out.push('\n');
                }
                _ => {}
            }
            if out.len() >= budget_chars {
                break;
            }
        }
    }
    Ok(out)
}

/// `cortiq quantize-gptq`: calibrate per-layer input Hessians on a corpus,
/// then GPTQ-quantize every captured linear to `q1s` (holographic fold +
/// two-field mask); copy the rest (norms/embeddings/lm_head) verbatim.
pub fn run_quantize_gptq(
    input: &str,
    calib: &str,
    output: &str,
    keep: f32,
    tokens: usize,
    lambda: f64,
) -> anyhow::Result<()> {
    use cortiq_core::format::{CmfModel, TensorSpec};
    use cortiq_core::quant::dequant_tensor;
    use cortiq_core::types::{QuantType, TensorDtype};
    use cortiq_engine::{Pipeline, SamplerConfig};
    use std::sync::Arc;

    // Calibration must run the CPU batched-prefill path so the matmat hook
    // fires; keep the GPU graph out of it. Set before any pipeline thread
    // starts (single-threaded here), so the edition-2024 unsafety is moot.
    unsafe {
        std::env::set_var("CMF_GPU", "0");
    }

    eprintln!("loading {input} …");
    let model = Arc::new(CmfModel::open_sharded(input)?);
    let mut pipe = Pipeline::from_model(&model, SamplerConfig::default())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    pipe.set_confidence(false);

    let text = read_calib_text(calib, tokens.saturating_mul(8).max(4096))?;
    let mut ids = pipe.tokenizer.encode(&text);
    ids.truncate(tokens.max(GROUP_SIZE));
    anyhow::ensure!(ids.len() >= 2, "calibration corpus produced too few tokens");
    eprintln!("calibrating Hessians on {} tokens …", ids.len());

    // The dense Hessian is only needed for the GPTQ fold (binary + λ<1e5).
    // Ternary and the fold-off mask path need only the diagonal (Σx²),
    // which is the only thing that fits for a 12B.
    let is_ternary = std::env::var("CMF_GPTQ_TERNARY").map(|v| v == "1").unwrap_or(false);
    let need_full_h = !is_ternary && lambda < 1e5;
    cortiq_engine::gptq_capture::begin(need_full_h);
    let _ = pipe.ppl_ids(&ids);
    let hess = cortiq_engine::gptq_capture::end();
    eprintln!("captured input Hessians for {} linears", hess.len());
    drop(pipe);

    // The Hessian inverse is O(cols³); very wide inputs (e.g. down_proj at
    // the intermediate size) are skipped past this cap and copied verbatim
    // until the blocked/parallel inverse lands. `CMF_GPTQ_MAXCOL` overrides.
    let max_col: usize = std::env::var("CMF_GPTQ_MAXCOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    // Ternary bulk (BitNet b1.58) instead of binary — no Hessian inverse,
    // so it ignores the column cap and runs on every linear.
    let ternary = std::env::var("CMF_GPTQ_TERNARY").map(|v| v == "1").unwrap_or(false);
    if ternary {
        eprintln!("bulk codec: ternary {{-s,0,+s}} (q1t)");
    }

    // Copy-verbatim tensors are filled now; the eligible linears are handed
    // to a work-stealing pool that dequantizes ONE tensor per worker at a
    // time. RAM stays bounded (≈ nthreads × one tensor) — holding every f32
    // weight of a 12B at once would be ~48 GB. `CMF_GPTQ_MAXCOL` leaves the
    // widest tensors (e.g. down_proj) at the input dtype for a size-smart
    // mixed model.
    let mut specs: Vec<Option<TensorSpec>> = (0..model.tensors.len()).map(|_| None).collect();
    let mut eligible: Vec<usize> = Vec::new();
    let mut n_copy = 0usize;
    for (slot, entry) in model.tensors.iter().enumerate() {
        let ok = entry.shape.len() == 2
            && entry.shape[1] % GROUP_SIZE == 0
            && entry.shape[1] <= max_col
            && hess
                .get(&entry.name)
                .map(|h| h.count > 0 && h.cols == entry.shape[1])
                .unwrap_or(false);
        if ok {
            eligible.push(slot);
        } else {
            specs[slot] = Some(TensorSpec {
                name: entry.name.clone(),
                dtype: entry.dtype,
                shape: entry.shape.clone(),
                data: model.entry_bytes(entry).to_vec(),
            });
            n_copy += 1;
        }
    }
    let n_gptq = eligible.len();
    eprintln!("  quantizing {n_gptq} linears (streamed, parallel), copying {n_copy} verbatim …");
    let out_dtype = if ternary { TensorDtype::Q1T } else { TensorDtype::Q1S };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let model_ref = &model;
    let hess_ref = &hess;
    let elig_ref = &eligible;
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let results: Vec<(usize, Vec<u8>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..nthreads)
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if idx >= elig_ref.len() {
                            break;
                        }
                        let slot = elig_ref[idx];
                        let entry = &model_ref.tensors[slot];
                        let (rows, cols) = (entry.shape[0], entry.shape[1]);
                        let mut w = vec![0f32; rows * cols];
                        if dequant_tensor(entry, model_ref.entry_bytes(entry), &mut w).is_err() {
                            continue;
                        }
                        let h = &hess_ref[&entry.name];
                        let bytes = if ternary {
                            quantize_q1t(&w, rows, cols, &h.rms(), keep)
                        } else {
                            gptq_quantize_q1s(&w, rows, cols, h.h.clone(), &h.rms(), keep, lambda)
                        };
                        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        eprint!("\r  quantized {n}/{n_gptq}   ");
                        out.push((slot, bytes));
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    for (slot, data) in results {
        let entry = &model.tensors[slot];
        specs[slot] = Some(TensorSpec {
            name: entry.name.clone(),
            dtype: out_dtype,
            shape: entry.shape.clone(),
            data,
        });
    }
    let specs: Vec<TensorSpec> = specs.into_iter().map(|s| s.unwrap()).collect();
    eprintln!(
        "\r  quantized {n_gptq} linears to {}, copied {n_copy} verbatim   ",
        out_dtype.name()
    );

    let mut header = model.header.clone();
    header.quant_type = QuantType::Vbit;
    header.section_hashes = None;
    CmfModel::write(output, &header, &specs, None, model.vocab.as_deref())
        .map_err(|e| anyhow::anyhow!("write {output}: {e}"))?;
    Ok(())
}

/// In-place lower Cholesky `A = L·Lᵀ` (upper triangle zeroed). Returns
/// false if `A` is not positive-definite (caller adds more damping).
fn cholesky_lower(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 0.0 {
            return false;
        }
        let ljj = d.sqrt();
        a[j * n + j] = ljj;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / ljj;
        }
        for k in (j + 1)..n {
            a[j * n + k] = 0.0;
        }
    }
    true
}

/// Inverse of a lower-triangular `L` (also lower-triangular).
fn invert_lower(l: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0 / l[i * n + i];
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i {
                s += l[i * n + k] * inv[k * n + j];
            }
            inv[i * n + j] = -s / l[i * n + i];
        }
    }
    inv
}

/// `H⁻¹` (dense, symmetric) from a symmetric PD `H`, with adaptive
/// diagonal damping `λ·mean(diag)` (raised until the Cholesky succeeds —
/// dead input channels leave `H` singular). Consumes `h`.
fn inverse_symmetric(mut h: Vec<f64>, n: usize, mut lambda: f64) -> Vec<f64> {
    let mean_diag = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n.max(1) as f64;
    let base = mean_diag.max(1e-6);
    loop {
        let mut a = h.clone();
        for i in 0..n {
            a[i * n + i] += lambda * base;
        }
        if cholesky_lower(&mut a, n) {
            let linv = invert_lower(&a, n);
            // H⁻¹ = Linvᵀ·Linv (Linv lower ⇒ only k ≥ max(i,j) contribute).
            let mut hinv = vec![0.0f64; n * n];
            for i in 0..n {
                for j in i..n {
                    let mut s = 0.0;
                    for k in j..n {
                        s += linv[k * n + i] * linv[k * n + j];
                    }
                    hinv[i * n + j] = s;
                    hinv[j * n + i] = s;
                }
            }
            return hinv;
        }
        lambda *= 10.0;
        if lambda > 1.0 {
            // Fully degenerate — fall back to a scaled identity (the fold
            // becomes a no-op, i.e. plain per-group sign quant).
            let mut hinv = vec![0.0f64; n * n];
            for i in 0..n {
                hinv[i * n + i] = 1.0 / base;
            }
            h.clear();
            return hinv;
        }
    }
}

/// Column (input-channel) outlier mask: spend the same weight budget on
/// whole high-`‖W[:,j]‖·RMS(x_j)` INPUT CHANNELS instead of scattered
/// weights. Activation outliers are per-channel (LLM.int8/AWQ/SpQR), so a
/// kept channel makes the dot product with the outlier activations exact —
/// and a channel list encodes far cheaper than per-weight indices.
fn column_mask(w0: &[f32], in_dim: usize, act_rms: &[f32], n_out: usize) -> Vec<bool> {
    let rows = w0.len() / in_dim.max(1);
    let n_cols_keep = (n_out / rows.max(1)).min(in_dim);
    let mut m = vec![false; w0.len()];
    if n_cols_keep == 0 || n_cols_keep >= in_dim {
        return m;
    }
    let mut cs: Vec<(f32, usize)> = (0..in_dim)
        .map(|j| {
            let mut mx = 0f32;
            for o in 0..rows {
                mx = mx.max(w0[o * in_dim + j].abs());
            }
            (mx * act_rms.get(j).copied().unwrap_or(1.0), j)
        })
        .collect();
    let k = in_dim - n_cols_keep;
    cs.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap());
    let keep: std::collections::HashSet<usize> = cs[k..].iter().map(|&(_, j)| j).collect();
    for o in 0..rows {
        for j in 0..in_dim {
            if keep.contains(&j) {
                m[o * in_dim + j] = true;
            }
        }
    }
    m
}

/// Two-field outlier mask: keep the `n_out` weights of highest
/// `|W| · RMS(x_col)` (amplitude 𝒲 × activation θ) at full precision.
/// `CMF_GPTQ_COLMASK=1` spends the same budget on whole channels instead.
fn two_field_mask(w0: &[f32], in_dim: usize, act_rms: &[f32], n_out: usize) -> Vec<bool> {
    if std::env::var("CMF_GPTQ_COLMASK").map(|v| v == "1").unwrap_or(false) {
        return column_mask(w0, in_dim, act_rms, n_out);
    }
    let total = w0.len();
    let mut m = vec![false; total];
    if n_out > 0 && n_out < total {
        let mut score: Vec<(f32, usize)> = (0..total)
            .map(|idx| {
                let col = idx % in_dim;
                (w0[idx].abs() * act_rms.get(col).copied().unwrap_or(1.0), idx)
            })
            .collect();
        let k = total - n_out;
        score.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap());
        for &(_, idx) in &score[k..] {
            m[idx] = true;
        }
    }
    m
}

/// Ternary (BitNet b1.58) quantization with the two-field outlier mask —
/// NO fold (the holographic fold backfires at extreme low-bit with a noisy
/// single-pass Hessian; ternary's zero level is the real win). Each group's
/// scale is the abs-mean of its non-outlier weights; a weight rounds to
/// {−1,0,+1}·s (|w| < 0.5·s ⇒ 0, capturing the near-zero mass exactly).
/// Emits `Q1T` bytes.
pub fn quantize_q1t(
    w0: &[f32],
    out_dim: usize,
    in_dim: usize,
    act_rms: &[f32],
    keep_frac: f32,
) -> Vec<u8> {
    assert_eq!(w0.len(), out_dim * in_dim);
    assert_eq!(in_dim % GROUP_SIZE, 0);
    let total = out_dim * in_dim;
    let n_out = (((total as f32) * keep_frac).round() as usize).min(total);
    let is_out = two_field_mask(w0, in_dim, act_rms, n_out);
    let groups_per_row = in_dim / GROUP_SIZE;
    let n_groups = total / GROUP_SIZE;

    // Pass 1: per-group abs-mean scale + ternary codes.
    let mut scale = vec![0.0f32; out_dim * groups_per_row];
    let mut codes = vec![0u8; n_groups * 8];
    for g in 0..n_groups {
        let base = g * GROUP_SIZE;
        let mut sum = 0.0f32;
        let mut cnt = 0usize;
        for k in 0..GROUP_SIZE {
            if !is_out[base + k] {
                sum += w0[base + k].abs();
                cnt += 1;
            }
        }
        let s = f16_to_f32(f32_to_f16(if cnt > 0 { sum / cnt as f32 } else { 0.0 }))
            .max(6.103_515_625e-5);
        scale[g] = s;
        for k in 0..GROUP_SIZE {
            let i = base + k;
            let code: u8 = if is_out[i] {
                0
            } else {
                let r = w0[i] / s;
                if r >= 0.5 {
                    1
                } else if r <= -0.5 {
                    2
                } else {
                    0
                }
            };
            codes[g * 8 + k / 4] |= code << ((k % 4) * 2);
        }
    }

    // Pass 2 — послойная докрутка (light FCD): rescale each output row by
    // the closed-form α that minimizes the activation-weighted output error
    // ‖α·Q(x) − W(x)‖²_d (d = per-channel activation power = RMS²). One
    // scalar per row, folded into that row's group scales — zero extra
    // storage. Disabled by CMF_GPTQ_NOCORRECT=1 for ablation.
    if !std::env::var("CMF_GPTQ_NOCORRECT").map(|v| v == "1").unwrap_or(false) {
        for o in 0..out_dim {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for gg in 0..groups_per_row {
                let g = o * groups_per_row + gg;
                let s = scale[g];
                let base = o * in_dim + gg * GROUP_SIZE;
                for k in 0..GROUP_SIZE {
                    let i = base + k;
                    if is_out[i] {
                        continue; // outliers are exact — not rescaled
                    }
                    let code = (codes[g * 8 + k / 4] >> ((k % 4) * 2)) & 0x3;
                    let q = match code {
                        1 => s,
                        2 => -s,
                        _ => continue,
                    } as f64;
                    let d = {
                        let r = act_rms.get(base - o * in_dim + k).copied().unwrap_or(1.0) as f64;
                        r * r
                    };
                    num += q * w0[i] as f64 * d;
                    den += q * q * d;
                }
            }
            if den > 1e-20 {
                let alpha = (num / den).clamp(0.5, 2.0) as f32;
                for gg in 0..groups_per_row {
                    scale[o * groups_per_row + gg] *= alpha;
                }
            }
        }
    }

    // Emit: [f16 scale][8B codes] per group, then the outlier overlay.
    let n_out_actual = is_out.iter().filter(|&&o| o).count();
    let mut out = Vec::with_capacity(n_groups * 10 + 4 + n_out_actual * 6);
    for g in 0..n_groups {
        out.extend_from_slice(&f32_to_f16(scale[g]).to_le_bytes());
        out.extend_from_slice(&codes[g * 8..g * 8 + 8]);
    }
    out.extend_from_slice(&(n_out_actual as u32).to_le_bytes());
    for (i, &o) in is_out.iter().enumerate() {
        if o {
            out.extend_from_slice(&(i as u32).to_le_bytes());
            out.extend_from_slice(&f32_to_f16(w0[i]).to_le_bytes());
        }
    }
    out
}

/// Quantize `W [out,in]` to `Q1S` bytes with the GPTQ holographic fold.
/// `h` is the input Hessian `X·Xᵀ [in,in]` (row-major f64), `act_rms[i]` =
/// `RMS(x_i)` over calibration (the activation field of the two-field
/// outlier score `|W|·RMS(x)`), `keep_frac` = outlier budget, `lambda` =
/// relative damping (0.01 is standard).
pub fn gptq_quantize_q1s(
    w0: &[f32],
    out_dim: usize,
    in_dim: usize,
    h: Vec<f64>,
    act_rms: &[f32],
    keep_frac: f32,
    lambda: f64,
) -> Vec<u8> {
    assert_eq!(w0.len(), out_dim * in_dim);
    assert_eq!(in_dim % GROUP_SIZE, 0);

    let n = in_dim;
    // Fold off (λ ≥ 1e5): skip the O(n³) inverse entirely — the pass
    // reduces to the two-field mask + per-group sign quant. Cheap enough to
    // run on the widest tensors (down_proj) and to sweep the mask budget.
    // (The dense Hessian is only present/needed when folding.)
    let fold = lambda < 1e5;
    if fold {
        assert_eq!(h.len(), in_dim * in_dim);
    }
    let hinv = if fold { inverse_symmetric(h, n, lambda) } else { Vec::new() };

    // Working copy (mutated by the error fold) and the two-field outlier
    // mask (chosen once, from the ORIGINAL weights × activation field).
    let mut w = w0.to_vec();
    let total = out_dim * in_dim;
    let n_out = (((total as f32) * keep_frac).round() as usize).min(total);
    let is_out = two_field_mask(w0, in_dim, act_rms, n_out);

    let groups_per_row = in_dim / GROUP_SIZE;
    // Quantized reconstruction levels: sign bits + per-(row,group) scale,
    // accumulated here, emitted in row-major group order afterwards.
    let mut sign_pos = vec![false; total]; // true ⇒ +s
    let mut scale = vec![0.0f32; out_dim * groups_per_row];

    // GPTQ sweep over input channels, one group of 32 at a time.
    for gi in 0..groups_per_row {
        let c0 = gi * GROUP_SIZE;
        // Per-output-row group scale from the CURRENT (folded) weights,
        // excluding outliers so a spike does not inflate the ±s level.
        for o in 0..out_dim {
            let mut sum = 0.0f32;
            let mut cnt = 0usize;
            for c in c0..c0 + GROUP_SIZE {
                if !is_out[o * in_dim + c] {
                    sum += w[o * in_dim + c].abs();
                    cnt += 1;
                }
            }
            let s = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
            scale[o * groups_per_row + gi] =
                f16_to_f32(f32_to_f16(s)).max(6.103_515_625e-5);
        }
        // Column-by-column quant + holographic fold into the remaining ones.
        for c in c0..c0 + GROUP_SIZE {
            let inv_d = if fold {
                let dinv = hinv[c * n + c];
                if dinv.abs() > 1e-12 { 1.0 / dinv } else { 0.0 }
            } else {
                0.0
            };
            for o in 0..out_dim {
                let idx = o * in_dim + c;
                if is_out[idx] {
                    sign_pos[idx] = w0[idx] >= 0.0; // hint only; overlay is exact
                    continue; // kept verbatim ⇒ no residual to fold
                }
                let s = scale[o * groups_per_row + gi];
                let pos = w[idx] >= 0.0;
                sign_pos[idx] = pos;
                if fold {
                    let q = if pos { s } else { -s };
                    let err = w[idx] - q;
                    // Fold: W[o, c+1:] -= err · H⁻¹[c, c+1:] / H⁻¹[c,c].
                    let coef = (err as f64) * inv_d;
                    if coef != 0.0 {
                        let hrow = &hinv[c * n..c * n + n];
                        let wrow = &mut w[o * in_dim..o * in_dim + in_dim];
                        for j in (c + 1)..n {
                            wrow[j] -= (coef * hrow[j]) as f32;
                        }
                    }
                }
            }
        }
    }

    // Emit Q1S: q1 base [f16 scale][4B bits] per (row, group), then the
    // sparse outlier overlay [u32 count][count × (u32 idx, f16 val)].
    let n_out_actual = is_out.iter().filter(|&&o| o).count();
    let mut out = Vec::with_capacity(out_dim * groups_per_row * 6 + 4 + n_out_actual * 6);
    for o in 0..out_dim {
        for gi in 0..groups_per_row {
            let s = scale[o * groups_per_row + gi];
            out.extend_from_slice(&f32_to_f16(s).to_le_bytes());
            let base = o * in_dim + gi * GROUP_SIZE;
            for jb in 0..GROUP_SIZE / 8 {
                let mut byte = 0u8;
                for k in 0..8 {
                    if sign_pos[base + jb * 8 + k] {
                        byte |= 1 << k;
                    }
                }
                out.push(byte);
            }
        }
    }
    out.extend_from_slice(&(n_out_actual as u32).to_le_bytes());
    for (idx, &o) in is_out.iter().enumerate() {
        if o {
            out.extend_from_slice(&(idx as u32).to_le_bytes());
            out.extend_from_slice(&f32_to_f16(w0[idx]).to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_core::quant::{dequant_q1s, dequant_q1t};

    /// Ternary roundtrip: near-zero weights decode to exactly 0, the rest to
    /// ±s, and kept outliers to their f16 value. The zero level is the whole
    /// point — it must be bit-exact.
    #[test]
    fn q1t_ternary_roundtrip_zeros_and_levels() {
        let (rows, cols) = (2usize, 64usize);
        // Mostly tiny (→ 0), a few clearly ±, one spike outlier.
        let mut vals: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.31).sin() * 0.02).collect();
        vals[10] = 0.9;
        vals[11] = -0.85;
        vals[70] = 5.0; // outlier
        let rms = vec![1.0f32; cols];
        let bytes = quantize_q1t(&vals, rows, cols, &rms, 1.0 / (rows * cols) as f32);
        let mut dec = vec![0f32; rows * cols];
        dequant_q1t(&bytes, &mut dec);
        // The spike is kept verbatim (f16).
        assert!((dec[70] - 5.0).abs() < 0.02, "outlier: {}", dec[70]);
        // Tiny weights collapse to exactly 0 (the ternary win).
        let zeros = dec.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > rows * cols / 3, "ternary must zero many weights, got {zeros}");
        // Clear ± weights keep their sign.
        assert!(dec[10] > 0.0 && dec[11] < 0.0, "signs: {} {}", dec[10], dec[11]);
    }

    /// The core claim: on a layer with CORRELATED input activations, the
    /// GPTQ holographic fold cuts the calibration OUTPUT error ‖(W−Ŵ)·X‖
    /// far below the naïve per-group sign quantizer — because it preserves
    /// W·x, not the weights. (Weight-space methods cannot; that is the
    /// whole point of the holographic transfer.)
    #[test]
    fn holographic_fold_beats_naive_on_output_error() {
        let (out_dim, in_dim, t) = (8usize, 64usize, 400usize);
        // Deterministic PRNG (no Math.random in this env / determinism).
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
        };
        // Correlated activations: a few latent factors drive all channels,
        // so the Hessian is far from diagonal (where the fold has power).
        let factors = 6usize;
        let load: Vec<f32> = (0..in_dim * factors).map(|_| rnd()).collect();
        let mut x = vec![0.0f32; in_dim * t]; // [in, t]
        for ti in 0..t {
            let z: Vec<f32> = (0..factors).map(|_| rnd()).collect();
            for i in 0..in_dim {
                let mut v = 0.15 * rnd();
                for f in 0..factors {
                    v += load[i * factors + f] * z[f];
                }
                x[i * t + ti] = v;
            }
        }
        let w: Vec<f32> = (0..out_dim * in_dim).map(|_| 0.4 * rnd()).collect();

        // H = X·Xᵀ and per-channel activation RMS.
        let mut h = vec![0.0f64; in_dim * in_dim];
        for i in 0..in_dim {
            for j in 0..in_dim {
                let mut s = 0.0f64;
                for ti in 0..t {
                    s += (x[i * t + ti] as f64) * (x[j * t + ti] as f64);
                }
                h[i * in_dim + j] = s;
            }
        }
        let act_rms: Vec<f32> = (0..in_dim)
            .map(|i| {
                ((0..t).map(|ti| (x[i * t + ti] as f64).powi(2)).sum::<f64>() / t as f64).sqrt()
                    as f32
            })
            .collect();

        // Output error ‖(W − Ŵ)·X‖²_F for a given reconstruction.
        let out_err = |wh: &[f32]| -> f64 {
            let mut e = 0.0f64;
            for o in 0..out_dim {
                for ti in 0..t {
                    let mut d = 0.0f64;
                    for i in 0..in_dim {
                        d += ((w[o * in_dim + i] - wh[o * in_dim + i]) as f64) * (x[i * t + ti] as f64);
                    }
                    e += d * d;
                }
            }
            e
        };

        // Naïve q1 reconstruction (no fold, no mask): ±mean|w| per group.
        let naive = {
            let mut d = vec![0f32; out_dim * in_dim];
            let groups = out_dim * in_dim / GROUP_SIZE;
            for g in 0..groups {
                let grp = &w[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
                let s = grp.iter().map(|v| v.abs()).sum::<f32>() / GROUP_SIZE as f32;
                for (k, &v) in grp.iter().enumerate() {
                    d[g * GROUP_SIZE + k] = if v >= 0.0 { s } else { -s };
                }
            }
            d
        };
        // GPTQ q1s with a modest 1% mask.
        let gptq = {
            let bytes = gptq_quantize_q1s(&w, out_dim, in_dim, h, &act_rms, 0.01, 0.01);
            let mut d = vec![0f32; out_dim * in_dim];
            dequant_q1s(&bytes, &mut d);
            d
        };
        let (e_naive, e_gptq) = (out_err(&naive), out_err(&gptq));
        assert!(
            e_gptq < e_naive * 0.6,
            "holographic fold must cut output error ≥40%: naive={e_naive:.3} gptq={e_gptq:.3}"
        );
    }
}
