//! Layer-wise GPTQ — an **error-feedback transfer** for the `q1s` codec.
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
/// then GPTQ-quantize every captured linear to `q1s` (error-feedback fold +
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
    let is_ternary = std::env::var("CMF_GPTQ_TERNARY")
        .map(|v| v == "1")
        .unwrap_or(false);
    let need_full_h = !is_ternary && lambda < 1e5;
    cortiq_engine::gptq_capture::begin(need_full_h);
    let score = pipe.ppl_ids(&ids);
    let hess = cortiq_engine::gptq_capture::end();
    score.map_err(|e| anyhow::anyhow!(e))?;
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
    let ternary = std::env::var("CMF_GPTQ_TERNARY")
        .map(|v| v == "1")
        .unwrap_or(false);
    if ternary {
        eprintln!("bulk codec: ternary {{-s,0,+s}} (q1t)");
    }
    // Extra outlier budget for the sensitive down_proj (1.0 = uniform).
    let down_mult: f32 = std::env::var("CMF_GPTQ_DOWN_KEEP")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.0)
        .max(1.0);

    // Copy-verbatim tensors are filled now; the eligible linears are handed
    // to a work-stealing pool that dequantizes ONE tensor per worker at a
    // time. RAM stays bounded (≈ nthreads × one tensor) — holding every f32
    // weight of a 12B at once would be ~48 GB. `CMF_GPTQ_MAXCOL` leaves the
    // widest tensors (e.g. down_proj) at the input dtype for a size-smart
    // mixed model.
    let mut specs: Vec<Option<TensorSpec>> = (0..model.tensors.len()).map(|_| None).collect();
    // Tensors kept at the input precision (not quantized). embed/lm_head are
    // the vocab-wide, most bit-sensitive projections (SpQR/AWQ practice);
    // adding `down_proj` (the gated-intermediate output — the next most
    // sensitive) is more efficient than flooding it with sparse outliers.
    // `CMF_GPTQ_SKIP` overrides the substring list.
    let skip_names: Vec<String> = std::env::var("CMF_GPTQ_SKIP")
        .unwrap_or_else(|_| "embed_tokens,lm_head".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let mut eligible: Vec<usize> = Vec::new();
    let mut n_copy = 0usize;
    for (slot, entry) in model.tensors.iter().enumerate() {
        let keep_precise = skip_names.iter().any(|s| entry.name.contains(s.as_str()));
        let ok = entry.shape.len() == 2
            && entry.shape[1] % GROUP_SIZE == 0
            && entry.shape[1] <= max_col
            && !keep_precise
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
    let out_dtype = if ternary {
        TensorDtype::Q1T
    } else {
        TensorDtype::Q1S
    };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let model_ref = &model;
    let hess_ref = &hess;
    let elig_ref = &eligible;
    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
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
                        // Per-tensor outlier budget: down_proj is the most
                        // low-bit-sensitive FFN tensor (its input is the
                        // gated intermediate), so give it a bigger mask when
                        // CMF_GPTQ_DOWN_KEEP > 1 (default 1 = uniform).
                        let tk = if entry.name.contains("down_proj") {
                            (keep * down_mult).min(0.25)
                        } else {
                            keep
                        };
                        let bytes = if ternary {
                            quantize_q1t(&w, rows, cols, &h.rms(), tk)
                        } else {
                            gptq_quantize_q1s(&w, rows, cols, h.h.clone(), &h.rms(), tk, lambda)
                        };
                        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        eprint!("\r  quantized {n}/{n_gptq}   ");
                        out.push((slot, bytes));
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
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
/// `|W| · RMS(x_col)` (weight magnitude × activation RMS) at full precision.
/// `CMF_GPTQ_COLMASK=1` spends the same budget on whole channels instead.
fn two_field_mask(w0: &[f32], in_dim: usize, act_rms: &[f32], n_out: usize) -> Vec<bool> {
    if std::env::var("CMF_GPTQ_COLMASK")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return column_mask(w0, in_dim, act_rms, n_out);
    }
    let total = w0.len();
    let mut m = vec![false; total];
    if n_out > 0 && n_out < total {
        let mut score: Vec<(f32, usize)> = (0..total)
            .map(|idx| {
                let col = idx % in_dim;
                (
                    w0[idx].abs() * act_rms.get(col).copied().unwrap_or(1.0),
                    idx,
                )
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
/// NO fold (the error-feedback fold backfires at extreme low-bit with a noisy
/// single-pass Hessian; ternary's zero level is the real win). Each group's
/// scale/support are selected between the historical abs-mean rounding and
/// an activation-weighted least-squares candidate. The lower-error candidate
/// wins independently in each group, preserving the near-zero mass exactly.
/// Emits `Q1T` bytes.
fn q1t_group_candidate(
    w: &[f32],
    outlier: &[bool],
    act_rms: &[f32],
    col0: usize,
) -> (f32, [u8; GROUP_SIZE]) {
    debug_assert_eq!(w.len(), GROUP_SIZE);
    debug_assert_eq!(outlier.len(), GROUP_SIZE);
    let mut sum = 0.0f32;
    let mut cnt = 0usize;
    for k in 0..GROUP_SIZE {
        if !outlier[k] {
            sum += w[k].abs();
            cnt += 1;
        }
    }
    let mean = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
    let legacy_scale = f16_to_f32(f32_to_f16(mean)).max(6.103_515_6e-5);

    // A sparse-support candidate. Its level is the exact diagonal-Hessian
    // least-squares solution over the selected support:
    //   s = Σ d_i |w_i| / Σ d_i, d_i = RMS(x_i)^2.
    // 0.7·mean_abs is only a cheap proposal; the error comparison below
    // retains the historical candidate whenever it is better.
    let threshold = 0.7 * mean;
    let (mut num, mut den, mut legacy_err) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..GROUP_SIZE {
        if outlier[k] {
            continue;
        }
        let rms = act_rms.get(col0 + k).copied().unwrap_or(1.0) as f64;
        let d = rms * rms;
        let a = w[k].abs() as f64;
        if w[k].abs() >= threshold {
            num += d * a;
            den += d;
        }
        let q = if w[k] >= 0.5 * legacy_scale {
            legacy_scale
        } else if w[k] <= -0.5 * legacy_scale {
            -legacy_scale
        } else {
            0.0
        } as f64;
        let e = w[k] as f64 - q;
        legacy_err += d * e * e;
    }
    let adaptive_scale = if den > 0.0 {
        f16_to_f32(f32_to_f16((num / den) as f32)).max(6.103_515_6e-5)
    } else {
        legacy_scale
    };
    let mut adaptive_err = 0.0f64;
    for k in 0..GROUP_SIZE {
        if outlier[k] {
            continue;
        }
        let rms = act_rms.get(col0 + k).copied().unwrap_or(1.0) as f64;
        let d = rms * rms;
        let q = if w[k].abs() >= threshold {
            adaptive_scale.copysign(w[k])
        } else {
            0.0
        } as f64;
        let e = w[k] as f64 - q;
        adaptive_err += d * e * e;
    }

    let adaptive = adaptive_err < legacy_err;
    let scale = if adaptive {
        adaptive_scale
    } else {
        legacy_scale
    };
    let mut code = [0u8; GROUP_SIZE];
    for k in 0..GROUP_SIZE {
        if outlier[k] {
            continue;
        }
        let nonzero = if adaptive {
            w[k].abs() >= threshold
        } else {
            w[k].abs() >= 0.5 * legacy_scale
        };
        if nonzero {
            code[k] = if w[k] >= 0.0 { 1 } else { 2 };
        }
    }
    (scale, code)
}

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

    // Pass 1: per-group abs-mean scale + ternary codes (base-3, 5 per byte).
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    let mut scale = vec![0.0f32; out_dim * groups_per_row];
    let mut codes = vec![0u8; n_groups * 7];
    for g in 0..n_groups {
        let base = g * GROUP_SIZE;
        let col0 = base % in_dim;
        let (s, group_codes) = q1t_group_candidate(
            &w0[base..base + GROUP_SIZE],
            &is_out[base..base + GROUP_SIZE],
            act_rms,
            col0,
        );
        scale[g] = s;
        for k in 0..GROUP_SIZE {
            // Code 0 at outlier positions is a KERNEL INVARIANT: the q1t
            // matvec adds `value·x` for each overlay entry without subtracting
            // a base, so the base here must contribute nothing. Do not change.
            codes[g * 7 + k / 5] += group_codes[k] * POW3[k % 5];
        }
    }

    // Pass 2 — послойная докрутка (light FCD): rescale each output row by
    // the closed-form α that minimizes the activation-weighted output error
    // ‖α·Q(x) − W(x)‖²_d (d = per-channel activation power = RMS²). One
    // scalar per row, folded into that row's group scales — zero extra
    // storage. Disabled by CMF_GPTQ_NOCORRECT=1 for ablation.
    if !std::env::var("CMF_GPTQ_NOCORRECT")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
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
                    let code = cortiq_core::quant::q1t_code(&codes[g * 7..g * 7 + 7], k);
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

    // Emit: [f16 scale][7B base-3 codes] per group, then the per-row overlay
    // — [u32 row_ptr[out_dim+1]] then [(u16 col, f16 val)] grouped by row.
    // col is a within-row index, so in_dim must fit u16 (holds for all
    // quantized attn/FFN tensors; the vocab-sized embed/lm_head are skipped).
    assert!(
        in_dim <= u16::MAX as usize + 1,
        "q1t overlay: in_dim {in_dim} exceeds u16"
    );
    let mut row_ptr = vec![0u32; out_dim + 1];
    for o in 0..out_dim {
        let c = is_out[o * in_dim..(o + 1) * in_dim]
            .iter()
            .filter(|&&b| b)
            .count();
        row_ptr[o + 1] = row_ptr[o] + c as u32;
    }
    let n_out_actual = row_ptr[out_dim] as usize;
    let mut out = Vec::with_capacity(n_groups * 9 + (out_dim + 1) * 4 + n_out_actual * 4);
    for g in 0..n_groups {
        out.extend_from_slice(&f32_to_f16(scale[g]).to_le_bytes());
        out.extend_from_slice(&codes[g * 7..g * 7 + 7]);
    }
    for &p in &row_ptr {
        out.extend_from_slice(&p.to_le_bytes());
    }
    for o in 0..out_dim {
        for j in 0..in_dim {
            let i = o * in_dim + j;
            if is_out[i] {
                out.extend_from_slice(&(j as u16).to_le_bytes());
                out.extend_from_slice(&f32_to_f16(w0[i]).to_le_bytes());
            }
        }
    }
    out
}

/// Quantize `W [out,in]` to `Q1S` bytes with the GPTQ error-feedback fold.
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
    let hinv = if fold {
        inverse_symmetric(h, n, lambda)
    } else {
        Vec::new()
    };

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
            scale[o * groups_per_row + gi] = f16_to_f32(f32_to_f16(s)).max(6.103_515_6e-5);
        }
        // Column-by-column quant + error-feedback fold into the remaining ones.
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

// ---------------------------------------------------------------------------
// GPTQ for q4tp
// ---------------------------------------------------------------------------

/// `cortiq quantize-gptq --codec q4tp`: a q4tp file whose linears are
/// GPTQ-rounded against calibration Hessians instead of round-to-nearest.
///
/// * `input` holds the weights to quantize — an f16 (or f32/bf16) export of
///   the model (`cortiq convert --quant f16`), so nothing is quantized twice.
/// * `calib_model` (default: `input`) is the file the calibration forward
///   runs on. The Hessian hook sees only memory-mapped quantized tensors
///   (f16 matrices are expanded to f32 at load and lose their names), so
///   pass the q8/q8_2f export of the same checkpoint here: its activations
///   are within noise of f16 and its tensor names match.
/// * The corpus is scored in `window`-token windows (fresh context each),
///   `tokens` in total.
/// * Tensors without a Hessian, the `CMF_GPTQ_SKIP` list (default
///   `embed_tokens,lm_head`) and everything else 2-D get the converter's
///   policy — `--tensor-quant` overrides first, else the q4tp profile —
///   round-to-nearest; 1-D tensors are copied. The header is the input's
///   (tokenizer, chat template, arch) relabelled as a q4tp file.
#[allow(clippy::too_many_arguments)]
pub fn run_quantize_gptq_q4tp(
    input: &str,
    calib_model: Option<&str>,
    calib: &str,
    output: &str,
    tokens: usize,
    window: usize,
    lambda: f64,
    threads: usize,
    act_order: bool,
    hessians: Option<&str>,
) -> anyhow::Result<()> {
    use cortiq_core::format::{CmfModel, TensorSpec};
    use cortiq_core::quant::dequant_tensor;
    use cortiq_core::types::{QuantType, TensorDtype};
    use cortiq_engine::{Pipeline, SamplerConfig};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    unsafe {
        std::env::set_var("CMF_GPU", "0");
    }
    let t0 = Instant::now();
    let cal_path = calib_model.unwrap_or(input);
    let cached = hessians.filter(|p| std::path::Path::new(p).exists());
    let hess = if let Some(p) = cached {
        eprintln!("Hessians from cache {p} (calibration skipped) …");
        load_hessians(p)?
    } else {
        eprintln!("calibration model {cal_path} …");
        let cmodel = Arc::new(CmfModel::open_sharded(cal_path)?);
        let mut pipe = Pipeline::from_model(&cmodel, SamplerConfig::default())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        pipe.set_confidence(false);
        let text = read_calib_text(calib, tokens.saturating_mul(8).max(4096))?;
        let mut ids = pipe.tokenizer.encode(&text);
        ids.truncate(tokens.max(2));
        let window = window.max(2);
        let n_win = ids.len().div_ceil(window);
        eprintln!(
            "calibrating Hessians on {} tokens ({n_win} windows of ≤{window}) …",
            ids.len()
        );
        cortiq_engine::gptq_capture::begin(true);
        let mut nll_sum = 0f64;
        for (k, win) in ids.chunks(window).enumerate() {
            if win.len() < 2 {
                continue;
            }
            let p = pipe.ppl_ids(win);
            match p {
                Ok(p) => nll_sum += p.ln(),
                Err(e) => {
                    let _ = cortiq_engine::gptq_capture::end();
                    anyhow::bail!("calibration window {k}: {e}");
                }
            }
            eprint!(
                "\r  window {}/{n_win}  mean ppl {:.3}  {:.0}s   ",
                k + 1,
                (nll_sum / (k + 1) as f64).exp(),
                t0.elapsed().as_secs_f64()
            );
        }
        eprintln!();
        let h = cortiq_engine::gptq_capture::end();
        if let Some(p) = hessians {
            save_hessians(p, &h)?;
            eprintln!("Hessians cached to {p}");
        }
        h
    };
    eprintln!(
        "captured input Hessians for {} linears ({:.0}s)",
        hess.len(),
        t0.elapsed().as_secs_f64()
    );

    let model = Arc::new(CmfModel::open_sharded(input)?);
    let arch = model.arch().clone();
    let skip_names: Vec<String> = std::env::var("CMF_GPTQ_SKIP")
        .unwrap_or_else(|_| "embed_tokens,lm_head".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // Eligible linears, grouped by identical Hessians (q/k/v and gate/up
    // read the same input), so the O(n³) fold operator is built once per
    // input rather than once per projection.
    let mut specs: Vec<Option<TensorSpec>> = (0..model.tensors.len()).map(|_| None).collect();
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    let mut group_of: HashMap<(usize, u64, u64), usize> = HashMap::new();
    let mut rtn: Vec<usize> = Vec::new();
    for (slot, entry) in model.tensors.iter().enumerate() {
        let two_d = entry.shape.len() == 2 && entry.shape[1] % GROUP_SIZE == 0;
        let keep = skip_names.iter().any(|s| entry.name.contains(s.as_str()));
        let h = hess.get(&entry.name).filter(|h| h.count > 0 && h.cols == entry.shape[1]);
        let quant = if two_d {
            crate::convert::effective_quant(&arch, crate::convert::Quant::Q4TiledP, &entry.name)
        } else {
            crate::convert::Quant::F16
        };
        let is_float_src = matches!(
            entry.dtype,
            TensorDtype::F16 | TensorDtype::F32 | TensorDtype::Bf16
        );
        if two_d
            && !keep
            && is_float_src
            && quant == crate::convert::Quant::Q4TiledP
            && !crate::convert::keeps_float(&entry.name)
            && let Some(h) = h
        {
            // Fingerprint: count, trace and a strided sample of entries.
            let n = h.cols;
            let mut fa = 0f64;
            let mut fb = 0f64;
            for i in (0..n).step_by(7) {
                fa += h.h[i * n + i];
                fb += h.h[i * n + (i * 31 + 17) % n] * (i as f64 + 1.0);
            }
            let key = (h.count, fa.to_bits(), fb.to_bits());
            let gi = *group_of.entry(key).or_insert_with(|| {
                groups.push((entry.name.clone(), Vec::new()));
                groups.len() - 1
            });
            groups[gi].1.push(slot);
        } else {
            rtn.push(slot);
        }
    }
    let n_gptq: usize = groups.iter().map(|g| g.1.len()).sum();
    eprintln!(
        "GPTQ: {n_gptq} linears in {} Hessian groups; {} tensors by the converter policy",
        groups.len(),
        rtn.len()
    );

    // Converter-policy tensors (overrides, q4tp RTN, f16 copies).
    for &slot in &rtn {
        let entry = &model.tensors[slot];
        let two_d = entry.shape.len() == 2 && entry.shape[1] % GROUP_SIZE == 0;
        let float_src = matches!(
            entry.dtype,
            TensorDtype::F16 | TensorDtype::F32 | TensorDtype::Bf16
        );
        let (dtype, data) = if two_d && float_src && !crate::convert::keeps_float(&entry.name) {
            let mut w = vec![0f32; entry.shape[0] * entry.shape[1]];
            dequant_tensor(entry, model.entry_bytes(entry), &mut w)
                .map_err(|e| anyhow::anyhow!("{}: {e}", entry.name))?;
            let q = crate::convert::effective_quant(
                &arch,
                crate::convert::Quant::Q4TiledP,
                &entry.name,
            );
            crate::convert::quantize_2d(q, &w, entry.shape[0], entry.shape[1])
        } else {
            (entry.dtype, model.entry_bytes(entry).to_vec())
        };
        specs[slot] = Some(TensorSpec {
            name: entry.name.clone(),
            dtype,
            shape: entry.shape.clone(),
            data,
        });
    }

    // Hessian groups: build the fold operator once, quantize its members.
    // Groups run `outer` at a time; each member's row sweep uses the rest.
    // The fold operator is single-threaded O(n³) and dominates, so run many
    // groups at once (each holds ~3·n² f64 at its peak) and give each row
    // sweep what is left.
    let outer = threads.clamp(1, 16);
    let inner = (threads / outer).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::<(usize, Vec<u8>)>::new());
    let mut hess = hess;
    // Take each group's Hessian out of the map up front (one copy per group).
    let group_h: Vec<std::sync::Mutex<Option<Vec<f64>>>> = groups
        .iter()
        .map(|(lead, _)| {
            std::sync::Mutex::new(hess.remove(lead).map(|h| h.h))
        })
        .collect();
    drop(hess);
    std::thread::scope(|s| {
        for _ in 0..outer {
            s.spawn(|| {
                loop {
                    let gi = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if gi >= groups.len() {
                        break;
                    }
                    let Some(h) = group_h[gi].lock().unwrap().take() else {
                        continue;
                    };
                    let n = model.tensors[groups[gi].1[0]].shape[1];
                    let hdiag: Vec<f32> = (0..n).map(|i| h[i * n + i] as f32).collect();
                    let perm = act_order.then(|| act_order_perm(&hdiag));
                    let (u, dead) = match &perm {
                        Some(p) => {
                            let hp = permute_sym(&h, n, p);
                            drop(h);
                            let (u, _) = gptq_fold_operator(hp, n, lambda);
                            (u, hdiag.iter().map(|&d| !(d > 0.0)).collect::<Vec<bool>>())
                        }
                        None => gptq_fold_operator(h, n, lambda),
                    };
                    for &slot in &groups[gi].1 {
                        let entry = &model.tensors[slot];
                        let (rows, cols) = (entry.shape[0], entry.shape[1]);
                        let mut w = vec![0f32; rows * cols];
                        if dequant_tensor(entry, model.entry_bytes(entry), &mut w).is_err() {
                            continue;
                        }
                        let bytes = match &perm {
                            Some(p) => gptq_quantize_q4tp_act_order(
                                &w, rows, cols, &u, &dead, p, inner,
                            ),
                            None => gptq_quantize_q4tp(&w, rows, cols, &u, &dead, &hdiag, inner),
                        };
                        results.lock().unwrap().push((slot, bytes));
                        let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        eprint!(
                            "\r  gptq {k}/{n_gptq}  {:.0}s   ",
                            t0.elapsed().as_secs_f64()
                        );
                    }
                }
            });
        }
    });
    eprintln!();
    for (slot, data) in results.into_inner().unwrap() {
        let entry = &model.tensors[slot];
        specs[slot] = Some(TensorSpec {
            name: entry.name.clone(),
            dtype: TensorDtype::Q4TiledP,
            shape: entry.shape.clone(),
            data,
        });
    }
    let missing: Vec<&str> = specs
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_none())
        .map(|(i, _)| model.tensors[i].name.as_str())
        .collect();
    anyhow::ensure!(missing.is_empty(), "tensors not produced: {missing:?}");
    let specs: Vec<TensorSpec> = specs.into_iter().map(|s| s.unwrap()).collect();

    let mut header = model.header.clone();
    header.quant_type = QuantType::Q4Block;
    header.section_hashes = None;
    let mut prov = header
        .provenance
        .clone()
        .unwrap_or_else(|| serde_json::json!({}));
    prov["weight_quant"] = serde_json::json!("q4tp");
    prov["gptq"] = serde_json::json!({
        "codec": "q4tp",
        "calibration_tokens": tokens,
        "window": window,
        "lambda": lambda,
        "act_order": act_order,
        "linears": n_gptq,
    });
    header.provenance = Some(prov);
    CmfModel::write(output, &header, &specs, None, model.vocab.as_deref())
        .map_err(|e| anyhow::anyhow!("write {output}: {e}"))?;
    eprintln!("done in {:.0}s", t0.elapsed().as_secs_f64());
    Ok(())
}

#[inline]
fn dot64(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0f64; 4];
    let n4 = a.len() / 4;
    for c in 0..n4 {
        for l in 0..4 {
            acc[l] += a[c * 4 + l] * b[c * 4 + l];
        }
    }
    let mut s = acc[0] + acc[1] + acc[2] + acc[3];
    for k in n4 * 4..a.len() {
        s += a[k] * b[k];
    }
    s
}

/// Row-oriented in-place lower Cholesky (reads the lower triangle only,
/// zeroes the upper). Same result as `cholesky_lower`, but every inner loop
/// is a contiguous dot product.
fn cholesky_lower_rows(a: &mut [f64], n: usize) -> bool {
    for i in 0..n {
        for j in 0..=i {
            let (ri, rj) = if i == j {
                (&a[i * n..i * n + j], &a[i * n..i * n + j])
            } else {
                // SAFETY-free split: row j < row i.
                let (lo, hi) = a.split_at(i * n);
                (&hi[..j], &lo[j * n..j * n + j])
            };
            let s = a[i * n + j] - dot64(ri, rj);
            if i == j {
                if !(s > 0.0) || !s.is_finite() {
                    return false;
                }
                a[i * n + i] = s.sqrt();
            } else {
                a[i * n + j] = s / a[j * n + j];
            }
        }
        for k in (i + 1)..n {
            a[i * n + k] = 0.0;
        }
    }
    true
}

/// The GPTQ fold operator: the UPPER Cholesky factor `U` of `H⁻¹`
/// (`H⁻¹ = Uᵀ·U`), row-major f32, from the input Hessian `H` with relative
/// damping `λ·mean(diag)` (raised ×10 until `H` factors). Channels that
/// never fired (`H_ii = 0`) are returned as `dead`: their weights do not
/// matter to the calibration output and GPTQ zeroes them.
///
/// With `U`, quantizing column `j` and folding its residual
/// `e = (w_j − q_j)/U_jj` into `w_{k>j} −= e·U_jk` is the exact OBS update
/// with `H⁻¹` re-conditioned after every eliminated column — the reference
/// GPTQ formulation (the raw `H⁻¹` row is only exact for the first column).
pub fn gptq_fold_operator(mut h: Vec<f64>, n: usize, lambda: f64) -> (Vec<f32>, Vec<bool>) {
    assert_eq!(h.len(), n * n);
    let dead: Vec<bool> = (0..n).map(|i| !(h[i * n + i] > 0.0)).collect();
    for i in 0..n {
        if dead[i] {
            for k in 0..n {
                h[i * n + k] = 0.0;
                h[k * n + i] = 0.0;
            }
            h[i * n + i] = 1.0;
        }
    }
    let mean_diag = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n.max(1) as f64;
    let base = mean_diag.max(1e-12);
    let mut lam = lambda.max(0.0);
    let mut a = loop {
        let mut a = h.clone();
        for i in 0..n {
            a[i * n + i] += lam * base;
        }
        if cholesky_lower_rows(&mut a, n) {
            break a;
        }
        lam = if lam == 0.0 { 1e-4 } else { lam * 10.0 };
        if lam > 10.0 {
            // Degenerate beyond repair: plain round-to-nearest (U = diag).
            let mut u = vec![0f32; n * n];
            for i in 0..n {
                u[i * n + i] = 1.0;
            }
            return (u, dead);
        }
    };
    drop(h);
    // Columns of L⁻¹, stored as rows: lt[j][i] = (L⁻¹)[i][j], i ≥ j.
    let mut lt = vec![0f64; n * n];
    let mut x = vec![0f64; n];
    for j in 0..n {
        x[j] = 1.0 / a[j * n + j];
        for i in (j + 1)..n {
            let s = dot64(&a[i * n + j..i * n + i], &x[j..i]);
            x[i] = -s / a[i * n + i];
        }
        lt[j * n + j..j * n + n].copy_from_slice(&x[j..n]);
    }
    // H⁻¹ = L⁻ᵀ·L⁻¹: (H⁻¹)[i][k] = Σ_{m ≥ i} lt[i][m]·lt[k][m] for k ≤ i.
    for i in 0..n {
        for k in 0..=i {
            a[i * n + k] = dot64(&lt[i * n + i..i * n + n], &lt[k * n + i..k * n + n]);
        }
    }
    drop(lt);
    if !cholesky_lower_rows(&mut a, n) {
        let mut u = vec![0f32; n * n];
        for i in 0..n {
            u[i * n + i] = 1.0;
        }
        return (u, dead);
    }
    // H⁻¹ = M·Mᵀ with M lower ⇒ U = Mᵀ.
    let mut u = vec![0f32; n * n];
    for j in 0..n {
        for c in 0..=j {
            u[c * n + j] = a[j * n + c] as f32;
        }
    }
    (u, dead)
}

/// GPTQ-quantize `W [rows, cols]` to `q4tp` bytes — the SAME layout, decoder
/// and kernels as `convert::encode_q4tp`, only better-chosen nibbles.
///
/// Each row's scale ladder (`lo`, `step`) is the converter's, fixed from the
/// original weights. Columns are then quantized left to right with the fold
/// `u` (from [`gptq_fold_operator`]); a tile's rung is picked when the sweep
/// reaches it, from the already-corrected weights, by the error weighted with
/// the channel energy `hdiag` (activation second moment). `dead` channels
/// are zeroed. Rows are independent given `u`, so blocks of rows run in
/// parallel on `threads` workers.
pub fn gptq_quantize_q4tp(
    w0: &[f32],
    rows: usize,
    cols: usize,
    u: &[f32],
    dead: &[bool],
    hdiag: &[f32],
    threads: usize,
) -> Vec<u8> {
    use cortiq_core::quant::{
        Q4TP_LMAX, Q4TP_NIB, q4tp_ladder, q4tp_put_code, q4tp_sections,
    };
    assert_eq!(w0.len(), rows * cols);
    assert_eq!(cols % GROUP_SIZE, 0);
    assert_eq!(u.len(), cols * cols);
    let gpr = cols / GROUP_SIZE;
    let mut w_init = w0.to_vec();
    for r in 0..rows {
        for (c, &d) in dead.iter().enumerate() {
            if d {
                w_init[r * cols + c] = 0.0;
            }
        }
    }
    // The converter's ladder, from the (dead-zeroed) original weights.
    let mut out = crate::convert::encode_q4tp(&w_init, rows, cols);
    let (params_off, codes_off, stride) = q4tp_sections(rows, cols);
    let params = out[params_off..params_off + rows * 4].to_vec();

    const RB: usize = 8;
    let (nib_all, rest) = out.split_at_mut(params_off);
    let codes_all = &mut rest[codes_off - params_off..codes_off - params_off + rows * stride];
    let jobs: Vec<(usize, &mut [u8], &mut [u8])> = nib_all
        .chunks_mut(RB * gpr * Q4TP_NIB)
        .zip(codes_all.chunks_mut(RB * stride))
        .enumerate()
        .map(|(b, (n, c))| (b * RB, n, c))
        .collect();
    let queue = std::sync::Mutex::new(jobs);
    let w_init = &w_init;
    let params = &params;
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                loop {
                    let job = queue.lock().unwrap().pop();
                    let Some((r0, nib, codes)) = job else { break };
                    let rb = (rows - r0).min(RB);
                    let mut w = w_init[r0 * cols..(r0 + rb) * cols].to_vec();
                    let tabs: Vec<[f32; 32]> = (0..rb).map(|k| q4tp_ladder(params, r0 + k)).collect();
                    let lo_st: Vec<(f32, f32)> = (0..rb)
                        .map(|k| {
                            let p = &params[(r0 + k) * 4..(r0 + k) * 4 + 4];
                            (
                                f16_to_f32(u16::from_le_bytes([p[0], p[1]])),
                                f16_to_f32(u16::from_le_bytes([p[2], p[3]])),
                            )
                        })
                        .collect();
                    codes.fill(0);
                    let mut sc = [0f32; RB];
                    let mut e = [0f32; RB];
                    for g in 0..gpr {
                        let c0 = g * GROUP_SIZE;
                        for k in 0..rb {
                            let tile = &w[k * cols + c0..k * cols + c0 + GROUP_SIZE];
                            let hd = &hdiag[c0..c0 + GROUP_SIZE];
                            let absmax = tile.iter().fold(0f32, |m, v| m.max(v.abs()));
                            let (lo, st) = lo_st[k];
                            let nominal = if absmax == 0.0 || st <= 0.0 {
                                0usize
                            } else {
                                (((absmax / 7.0).log2() - lo) / st)
                                    .round()
                                    .clamp(0.0, Q4TP_LMAX as f32)
                                    as usize
                            };
                            let mut best = (f32::INFINITY, nominal);
                            if absmax > 0.0 {
                                for cand in nominal.saturating_sub(2)..=(nominal + 1).min(Q4TP_LMAX) {
                                    let s_c = tabs[k][cand];
                                    if s_c <= 0.0 {
                                        continue;
                                    }
                                    let iv = 1.0 / s_c;
                                    let mut err = 0f32;
                                    for (&wv, &h) in tile.iter().zip(hd) {
                                        let q = (wv * iv).round_ties_even().clamp(-8.0, 7.0);
                                        let d = wv - q * s_c;
                                        err += h * d * d;
                                    }
                                    if err < best.0 {
                                        best = (err, cand);
                                    }
                                }
                            }
                            let c = best.1;
                            q4tp_put_code(&mut codes[k * stride..(k + 1) * stride], g, c);
                            sc[k] = tabs[k][c];
                        }
                        for j in c0..c0 + GROUP_SIZE {
                            let d = u[j * cols + j];
                            for k in 0..rb {
                                let wv = w[k * cols + j];
                                let q = if sc[k] > 0.0 {
                                    (wv / sc[k]).round_ties_even().clamp(-8.0, 7.0)
                                } else {
                                    0.0
                                };
                                let nb = (q as i8 + 8) as u8 & 0x0F;
                                let byte = &mut nib[(k * gpr + g) * Q4TP_NIB + (j - c0) / 2];
                                if (j - c0) % 2 == 0 {
                                    *byte = (*byte & 0xF0) | nb;
                                } else {
                                    *byte = (*byte & 0x0F) | (nb << 4);
                                }
                                e[k] = if d > 0.0 { (wv - q * sc[k]) / d } else { 0.0 };
                            }
                            let urow = &u[j * cols + j + 1..(j + 1) * cols];
                            for k in 0..rb {
                                let ek = e[k];
                                if ek == 0.0 {
                                    continue;
                                }
                                let wr = &mut w[k * cols + j + 1..(k + 1) * cols];
                                for (x, &uu) in wr.iter_mut().zip(urow) {
                                    *x -= ek * uu;
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    out
}

const HESS_MAGIC: &[u8; 8] = b"CMFHESS1";

/// Save captured Hessians so GPTQ variants (damping, act order, which
/// tensors) can be re-run without re-calibrating. Identical Hessians (q/k/v
/// and gate/up read one input) are stored once under all their names; only
/// the upper triangle is written. f64 throughout — bit-exact round trip.
pub fn save_hessians(
    path: &str,
    hess: &std::collections::HashMap<String, cortiq_engine::gptq_capture::HessianAcc>,
) -> anyhow::Result<()> {
    use std::io::Write;
    let mut names: Vec<&String> = hess.keys().collect();
    names.sort();
    // Group exact duplicates.
    let mut uniq: Vec<(Vec<&String>, &cortiq_engine::gptq_capture::HessianAcc)> = Vec::new();
    for n in names {
        let a = &hess[n];
        if let Some(u) = uniq
            .iter_mut()
            .find(|(_, b)| b.cols == a.cols && b.count == a.count && b.sumsq == a.sumsq && b.h == a.h)
        {
            u.0.push(n);
        } else {
            uniq.push((vec![n], a));
        }
    }
    let tmp = format!("{path}.tmp");
    let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp)?);
    f.write_all(HESS_MAGIC)?;
    f.write_all(&(uniq.len() as u64).to_le_bytes())?;
    for (ns, a) in &uniq {
        f.write_all(&(ns.len() as u32).to_le_bytes())?;
        for n in ns {
            f.write_all(&(n.len() as u32).to_le_bytes())?;
            f.write_all(n.as_bytes())?;
        }
        f.write_all(&(a.cols as u64).to_le_bytes())?;
        f.write_all(&(a.count as u64).to_le_bytes())?;
        f.write_all(&(a.h.len() as u64).to_le_bytes())?;
        for v in &a.sumsq {
            f.write_all(&v.to_le_bytes())?;
        }
        let n = a.cols;
        if a.h.len() == n * n {
            for i in 0..n {
                for v in &a.h[i * n + i..i * n + n] {
                    f.write_all(&v.to_le_bytes())?;
                }
            }
        }
    }
    f.flush()?;
    drop(f);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Inverse of [`save_hessians`].
pub fn load_hessians(
    path: &str,
) -> anyhow::Result<std::collections::HashMap<String, cortiq_engine::gptq_capture::HessianAcc>> {
    use std::io::Read;
    let mut f = std::io::BufReader::with_capacity(1 << 22, std::fs::File::open(path)?);
    let mut m = [0u8; 8];
    f.read_exact(&mut m)?;
    anyhow::ensure!(&m == HESS_MAGIC, "{path}: not a Hessian cache");
    let mut u64b = [0u8; 8];
    let mut u32b = [0u8; 4];
    let mut rd_u64 = |f: &mut std::io::BufReader<std::fs::File>| -> anyhow::Result<u64> {
        f.read_exact(&mut u64b)?;
        Ok(u64::from_le_bytes(u64b))
    };
    let n_uniq = rd_u64(&mut f)?;
    let mut out = std::collections::HashMap::new();
    for _ in 0..n_uniq {
        f.read_exact(&mut u32b)?;
        let nn = u32::from_le_bytes(u32b) as usize;
        let mut ns = Vec::with_capacity(nn);
        for _ in 0..nn {
            f.read_exact(&mut u32b)?;
            let mut s = vec![0u8; u32::from_le_bytes(u32b) as usize];
            f.read_exact(&mut s)?;
            ns.push(String::from_utf8(s)?);
        }
        let cols = rd_u64(&mut f)? as usize;
        let count = rd_u64(&mut f)? as usize;
        let hlen = rd_u64(&mut f)? as usize;
        let mut buf = vec![0u8; cols * 8];
        f.read_exact(&mut buf)?;
        let sumsq: Vec<f64> = buf
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut h = vec![0f64; hlen];
        if hlen == cols * cols {
            let mut row = vec![0u8; cols * 8];
            for i in 0..cols {
                let r = &mut row[..(cols - i) * 8];
                f.read_exact(r)?;
                for (k, c) in r.chunks_exact(8).enumerate() {
                    let v = f64::from_le_bytes(c.try_into().unwrap());
                    h[i * cols + i + k] = v;
                    h[(i + k) * cols + i] = v;
                }
            }
        }
        for n in ns {
            out.insert(
                n,
                cortiq_engine::gptq_capture::HessianAcc {
                    cols,
                    h: h.clone(),
                    sumsq: sumsq.clone(),
                    count,
                },
            );
        }
    }
    Ok(out)
}

/// Channel order for act-order GPTQ: descending input energy `H_ii`.
pub fn act_order_perm(hdiag: &[f32]) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..hdiag.len()).collect();
    perm.sort_by(|&a, &b| hdiag[b].total_cmp(&hdiag[a]).then(a.cmp(&b)));
    perm
}

/// `H[perm][perm]` — the Hessian in act-order channel order.
pub fn permute_sym(h: &[f64], n: usize, perm: &[usize]) -> Vec<f64> {
    let mut p = vec![0f64; n * n];
    for (a, &pa) in perm.iter().enumerate() {
        for (b, &pb) in perm.iter().enumerate() {
            p[a * n + b] = h[pa * n + pb];
        }
    }
    p
}

/// Act-order GPTQ into q4tp: channels are eliminated in `perm` order (the
/// loudest inputs first, while the most columns remain to absorb their
/// error), `u` being the fold operator of the PERMUTED Hessian. Because a
/// tile's channels are now visited out of order, its rung is fixed up front
/// — the converter's own choice from the original weights ("static
/// groups") — and only the nibbles move. Same layout as `encode_q4tp`.
pub fn gptq_quantize_q4tp_act_order(
    w0: &[f32],
    rows: usize,
    cols: usize,
    u: &[f32],
    dead: &[bool],
    perm: &[usize],
    threads: usize,
) -> Vec<u8> {
    use cortiq_core::quant::{Q4TP_NIB, q4tp_code, q4tp_ladder, q4tp_sections};
    assert_eq!(w0.len(), rows * cols);
    assert_eq!(cols % GROUP_SIZE, 0);
    assert_eq!(u.len(), cols * cols);
    assert_eq!(perm.len(), cols);
    let gpr = cols / GROUP_SIZE;
    let mut w_init = w0.to_vec();
    for r in 0..rows {
        for (c, &d) in dead.iter().enumerate() {
            if d {
                w_init[r * cols + c] = 0.0;
            }
        }
    }
    let mut out = crate::convert::encode_q4tp(&w_init, rows, cols);
    let (params_off, codes_off, stride) = q4tp_sections(rows, cols);
    let params = out[params_off..params_off + rows * 4].to_vec();
    let codes = out[codes_off..codes_off + rows * stride].to_vec();
    // `dead` is in original channel order; `u` is in permuted order.
    const RB: usize = 8;
    let nib_all = &mut out[..params_off];
    let jobs: Vec<(usize, &mut [u8])> = nib_all
        .chunks_mut(RB * gpr * Q4TP_NIB)
        .enumerate()
        .map(|(b, n)| (b * RB, n))
        .collect();
    let queue = std::sync::Mutex::new(jobs);
    let (w_init, params, codes) = (&w_init, &params, &codes);
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                loop {
                    let job = queue.lock().unwrap().pop();
                    let Some((r0, nib)) = job else { break };
                    let rb = (rows - r0).min(RB);
                    // Working rows in permuted channel order.
                    let mut w = vec![0f32; rb * cols];
                    let mut scl = vec![0f32; rb * cols];
                    for k in 0..rb {
                        let r = r0 + k;
                        let tab = q4tp_ladder(params, r);
                        let crow = &codes[r * stride..(r + 1) * stride];
                        for (a, &pa) in perm.iter().enumerate() {
                            w[k * cols + a] = w_init[r * cols + pa];
                            scl[k * cols + a] = tab[q4tp_code(crow, pa / GROUP_SIZE)];
                        }
                    }
                    nib.fill(0);
                    let mut e = [0f32; RB];
                    for a in 0..cols {
                        let pa = perm[a];
                        let (g, within) = (pa / GROUP_SIZE, pa % GROUP_SIZE);
                        let d = u[a * cols + a];
                        for k in 0..rb {
                            let wv = w[k * cols + a];
                            let s_ = scl[k * cols + a];
                            let q = if s_ > 0.0 {
                                (wv / s_).round_ties_even().clamp(-8.0, 7.0)
                            } else {
                                0.0
                            };
                            let nb = (q as i8 + 8) as u8 & 0x0F;
                            let byte = &mut nib[(k * gpr + g) * Q4TP_NIB + within / 2];
                            if within % 2 == 0 {
                                *byte |= nb;
                            } else {
                                *byte |= nb << 4;
                            }
                            e[k] = if d > 0.0 { (wv - q * s_) / d } else { 0.0 };
                        }
                        let urow = &u[a * cols + a + 1..(a + 1) * cols];
                        for k in 0..rb {
                            let ek = e[k];
                            if ek == 0.0 {
                                continue;
                            }
                            let wr = &mut w[k * cols + a + 1..(k + 1) * cols];
                            for (x, &uu) in wr.iter_mut().zip(urow) {
                                *x -= ek * uu;
                            }
                        }
                    }
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_core::quant::{dequant_q1s, dequant_q1t};

    /// GPTQ-q4tp decodes through the ordinary q4tp reader, and on correlated
    /// inputs its OUTPUT error ‖(W − Ŵ)·X‖ beats the converter's
    /// round-to-nearest encoding of the same weights with the same ladder.
    #[test]
    fn gptq_q4tp_beats_rtn_on_output_error() {
        use cortiq_core::quant::dequant_q4tp;
        let (rows, cols, t) = (24usize, 96usize, 400usize);
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) as f32 - 0.5
        };
        let w: Vec<f32> = (0..rows * cols).map(|_| rnd()).collect();
        // Correlated activations: a few latent factors plus noise, one
        // channel with a large scale (the outlier case GPTQ exists for).
        let lat: Vec<f32> = (0..t * 4).map(|_| rnd()).collect();
        let mix: Vec<f32> = (0..4 * cols).map(|_| rnd()).collect();
        let mut x = vec![0f32; t * cols];
        for s in 0..t {
            for c in 0..cols {
                let mut v = 0.2 * rnd();
                for f in 0..4 {
                    v += lat[s * 4 + f] * mix[f * cols + c];
                }
                x[s * cols + c] = if c == 5 { v * 8.0 } else { v };
            }
        }
        let mut h = vec![0f64; cols * cols];
        for s in 0..t {
            for i in 0..cols {
                for j in 0..cols {
                    h[i * cols + j] += x[s * cols + i] as f64 * x[s * cols + j] as f64;
                }
            }
        }
        let hdiag: Vec<f32> = (0..cols).map(|i| h[i * cols + i] as f32).collect();
        let (u, dead) = gptq_fold_operator(h, cols, 0.01);
        assert!(dead.iter().all(|d| !d));
        let g = gptq_quantize_q4tp(&w, rows, cols, &u, &dead, &hdiag, 3);
        let r = crate::convert::encode_q4tp(&w, rows, cols);
        assert_eq!(g.len(), r.len());
        let out_err = |bytes: &[u8]| -> f64 {
            let mut d = vec![0f32; rows * cols];
            dequant_q4tp(bytes, rows, cols, &mut d);
            let mut e = 0f64;
            for s in 0..t {
                for o in 0..rows {
                    let mut acc = 0f64;
                    for c in 0..cols {
                        acc += (w[o * cols + c] - d[o * cols + c]) as f64 * x[s * cols + c] as f64;
                    }
                    e += acc * acc;
                }
            }
            e
        };
        let (eg, er) = (out_err(&g), out_err(&r));
        assert!(eg < 0.8 * er, "gptq {eg} vs rtn {er}");
        // Act-order: same ladder AND same rungs as RTN (static groups).
        let perm = act_order_perm(&hdiag);
        assert_eq!(perm[0], 5, "the loud channel goes first");
        let mut h2 = vec![0f64; cols * cols];
        for s in 0..t {
            for i in 0..cols {
                for j in 0..cols {
                    h2[i * cols + j] += x[s * cols + i] as f64 * x[s * cols + j] as f64;
                }
            }
        }
        let (up, _) = gptq_fold_operator(permute_sym(&h2, cols, &perm), cols, 0.01);
        let ga = gptq_quantize_q4tp_act_order(&w, rows, cols, &up, &dead, &perm, 2);
        let ea = out_err(&ga);
        assert!(ea < 0.8 * er, "act-order {ea} vs rtn {er}");
        let (po2, _, _) = cortiq_core::quant::q4tp_sections(rows, cols);
        assert_eq!(&ga[po2..], &r[po2..], "act-order keeps the converter's ladder and rungs");
        // Same ladder parameters as the converter (only nibbles/codes move).
        let (po, _, _) = cortiq_core::quant::q4tp_sections(rows, cols);
        assert_eq!(&g[po..po + rows * 4], &r[po..po + rows * 4]);
    }

    #[test]
    fn hessian_cache_roundtrip_is_exact_and_dedups() {
        use cortiq_engine::gptq_capture::HessianAcc;
        let n = 5;
        let mk = |k: f64| {
            let mut h = vec![0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    h[i * n + j] = k * (1 + i.min(j)) as f64 + 0.125 * (i + j) as f64;
                }
            }
            HessianAcc { cols: n, h, sumsq: (0..n).map(|i| k + i as f64).collect(), count: 7 }
        };
        let mut m = std::collections::HashMap::new();
        m.insert("a.q".to_string(), mk(1.5));
        m.insert("a.k".to_string(), mk(1.5));
        m.insert("b".to_string(), mk(-2.25));
        let p = std::env::temp_dir().join(format!("cortiq-hess-{}.bin", std::process::id()));
        save_hessians(p.to_str().unwrap(), &m).unwrap();
        // two unique payloads: header + 2 × (names, cols, count, len, sumsq, upper)
        let back = load_hessians(p.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(back.len(), 3);
        for (k, v) in &m {
            let b = &back[k];
            assert_eq!((b.cols, b.count), (v.cols, v.count));
            assert_eq!(b.sumsq, v.sumsq);
            assert_eq!(b.h, v.h);
        }
    }

    /// With an identity Hessian GPTQ has nothing to fold: it must reproduce
    /// round-to-nearest within the ladder (codes may differ only where the
    /// converter's own rung search would tie).
    #[test]
    fn gptq_fold_operator_identity_is_identity() {
        let n = 40;
        let mut h = vec![0f64; n * n];
        for i in 0..n {
            h[i * n + i] = 2.0;
        }
        let (u, _) = gptq_fold_operator(h, n, 0.0);
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { (0.5f32).sqrt() } else { 0.0 };
                assert!((u[i * n + j] - want).abs() < 1e-6, "{i},{j}");
            }
        }
    }

    #[test]
    fn adaptive_ternary_group_never_worsens_weighted_proxy() {
        let w: Vec<f32> = (0..GROUP_SIZE)
            .map(|i| {
                let a = ((i * 17 + 5) % 31) as f32 / 31.0;
                if i % 3 == 0 {
                    a * 0.08
                } else {
                    (a + 0.15) * if i % 2 == 0 { 1.0 } else { -1.0 }
                }
            })
            .collect();
        let rms: Vec<f32> = (0..GROUP_SIZE)
            .map(|i| 0.3 + (i % 7) as f32 * 0.4)
            .collect();
        let outlier = vec![false; GROUP_SIZE];
        let (scale, codes) = q1t_group_candidate(&w, &outlier, &rms, 0);

        let legacy_scale = f16_to_f32(f32_to_f16(
            w.iter().map(|x| x.abs()).sum::<f32>() / GROUP_SIZE as f32,
        ))
        .max(6.103_515_6e-5);
        let err = |s: f32, c: &[u8]| -> f64 {
            (0..GROUP_SIZE)
                .map(|i| {
                    let q = match c[i] {
                        1 => s,
                        2 => -s,
                        _ => 0.0,
                    };
                    let e = (w[i] - q) as f64;
                    e * e * (rms[i] * rms[i]) as f64
                })
                .sum()
        };
        let legacy_codes: Vec<u8> = w
            .iter()
            .map(|&x| {
                if x >= 0.5 * legacy_scale {
                    1
                } else if x <= -0.5 * legacy_scale {
                    2
                } else {
                    0
                }
            })
            .collect();
        assert!(err(scale, &codes) < err(legacy_scale, &legacy_codes));
    }

    /// Ternary roundtrip: near-zero weights decode to exactly 0, the rest to
    /// ±s, and kept outliers to their f16 value. The zero level is the whole
    /// point — it must be bit-exact.
    #[test]
    fn q1t_ternary_roundtrip_zeros_and_levels() {
        let (rows, cols) = (2usize, 64usize);
        // Mostly tiny (→ 0), a few clearly ±, one spike outlier.
        let mut vals: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.31).sin() * 0.02)
            .collect();
        vals[10] = 0.9;
        vals[11] = -0.85;
        vals[70] = 5.0; // outlier
        let rms = vec![1.0f32; cols];
        let bytes = quantize_q1t(&vals, rows, cols, &rms, 1.0 / (rows * cols) as f32);
        let mut dec = vec![0f32; rows * cols];
        dequant_q1t(&bytes, rows, cols, &mut dec);
        // The spike is kept verbatim (f16).
        assert!((dec[70] - 5.0).abs() < 0.02, "outlier: {}", dec[70]);
        // Tiny weights collapse to exactly 0 (the ternary win).
        let zeros = dec.iter().filter(|&&v| v == 0.0).count();
        assert!(
            zeros > rows * cols / 3,
            "ternary must zero many weights, got {zeros}"
        );
        // Clear ± weights keep their sign.
        assert!(
            dec[10] > 0.0 && dec[11] < 0.0,
            "signs: {} {}",
            dec[10],
            dec[11]
        );
    }

    /// The core claim: on a layer with CORRELATED input activations, the
    /// GPTQ error-feedback fold cuts the calibration OUTPUT error ‖(W−Ŵ)·X‖
    /// far below the naïve per-group sign quantizer — because it preserves
    /// W·x, not the weights. (Weight-space methods cannot; that is the
    /// whole point of the error-feedback transfer.)
    #[test]
    fn error_feedback_fold_beats_naive_on_output_error() {
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
                        d += ((w[o * in_dim + i] - wh[o * in_dim + i]) as f64)
                            * (x[i * t + ti] as f64);
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
            "error-feedback fold must cut output error ≥40%: naive={e_naive:.3} gptq={e_gptq:.3}"
        );
    }
}
