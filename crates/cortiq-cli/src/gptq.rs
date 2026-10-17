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

/// One calibration corpus of `quantize-gptq --codec q4tp|q2tp` and its
/// relative share of the token budget (`--calib PATH[=SHARE]`, repeatable).
#[derive(Clone, Debug, PartialEq)]
pub struct CalibSource {
    pub path: String,
    pub share: f64,
}

impl CalibSource {
    /// `PATH` or `PATH=SHARE`. A suffix that does not parse as a number is
    /// part of the path; a source without a share weighs 1.
    pub fn parse(spec: &str) -> anyhow::Result<Self> {
        if let Some((p, s)) = spec.rsplit_once('=')
            && let Ok(v) = s.trim().parse::<f64>()
        {
            anyhow::ensure!(!p.is_empty(), "--calib '{spec}': empty path");
            anyhow::ensure!(
                v.is_finite() && v > 0.0,
                "--calib '{spec}': the share must be a positive number"
            );
            return Ok(Self { path: p.to_string(), share: v });
        }
        anyhow::ensure!(!spec.is_empty(), "--calib: empty path");
        Ok(Self { path: spec.to_string(), share: 1.0 })
    }
}

/// `tokens` split over the sources by share, rounded cumulatively so the
/// parts sum to exactly `tokens`.
pub fn split_calib_budget(sources: &[CalibSource], tokens: usize) -> Vec<usize> {
    let total: f64 = sources.iter().map(|s| s.share).sum();
    let (mut cum, mut prev) = (0f64, 0usize);
    let mut out: Vec<usize> = sources
        .iter()
        .map(|s| {
            cum += s.share;
            let upto = (((tokens as f64) * cum / total).round() as usize).clamp(prev, tokens);
            let n = upto - prev;
            prev = upto;
            n
        })
        .collect();
    if let Some(last) = out.last_mut() {
        *last += tokens - prev;
    }
    out
}

/// About `budget` bytes of calibration text, without loading a large corpus
/// whole: `.json` = an array of `[prompt, text]` pairs or strings (parsed
/// whole; that format is small by construction), `.jsonl` = one JSON value
/// per line (`{"text": …}` or a string — C4-style dumps), anything else =
/// the leading bytes of a plain-text file cut back to a line boundary.
/// Returns the text and whether the source ran out.
fn read_calib_text_budget(path: &str, budget: usize) -> anyhow::Result<(String, bool)> {
    use std::io::{BufRead, Read};
    if path.ends_with(".json") {
        let text = read_calib_text(path, budget)?;
        let exhausted = text.len() < budget;
        return Ok((text, exhausted));
    }
    let f = std::fs::File::open(path).map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
    if path.ends_with(".jsonl") {
        let mut rd = std::io::BufReader::new(f);
        let (mut out, mut line) = (String::new(), String::new());
        loop {
            line.clear();
            if rd.read_line(&mut line)? == 0 {
                return Ok((out, true));
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
                continue;
            };
            let t = match &v {
                serde_json::Value::String(s) => Some(s.as_str()),
                serde_json::Value::Object(o) => o.get("text").and_then(|t| t.as_str()),
                _ => None,
            };
            if let Some(t) = t {
                out.push_str(t);
                out.push('\n');
            }
            if out.len() >= budget {
                return Ok((out, false));
            }
        }
    }
    let mut buf = Vec::with_capacity(budget.min(1 << 28) + 1);
    f.take(budget as u64 + 1).read_to_end(&mut buf)?;
    let exhausted = buf.len() <= budget;
    if !exhausted {
        buf.truncate(budget);
        // Back to the last newline in the final 4 KiB, else to a char start,
        // so the cut never splits a word or a UTF-8 sequence.
        let tail = buf.len().saturating_sub(4096);
        if let Some(p) = buf[tail..].iter().rposition(|&c| c == b'\n') {
            buf.truncate(tail + p + 1);
        } else {
            while buf.last().is_some_and(|&c| c & 0xC0 == 0x80) {
                buf.pop();
            }
            if buf.last().is_some_and(|&c| c >= 0xC0) {
                buf.pop();
            }
        }
    }
    Ok((String::from_utf8_lossy(&buf).into_owned(), exhausted))
}

/// The first `want` tokens of one calibration source, reading only as much
/// text as that takes (≈8 bytes a token, doubled until it suffices). A cut
/// can only disturb the last few tokens, so a margin past `want` is required
/// before the prefix counts as final.
pub(crate) fn calib_source_tokens(
    path: &str,
    want: usize,
    encode: &dyn Fn(&str) -> Vec<u32>,
) -> anyhow::Result<Vec<u32>> {
    if want == 0 {
        return Ok(Vec::new());
    }
    let mut budget = want.saturating_mul(8).max(4096);
    loop {
        let (text, exhausted) = read_calib_text_budget(path, budget)?;
        let mut ids = encode(&text);
        if exhausted || ids.len() >= want + 64 {
            ids.truncate(want);
            return Ok(ids);
        }
        budget = budget.saturating_mul(2);
    }
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
    crate::knowledge::refuse_genome_rewrite(&model, "quantize gptq")?;
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

/// Options of `cortiq quantize-gptq --codec q4tp|q2tp`.
pub(crate) struct GptqTiledOpts<'a> {
    /// The weights to quantize — an f16 (or f32/bf16) export of the model
    /// (`cortiq convert --quant f16`), so nothing is quantized twice.
    pub input: &'a str,
    /// The file the calibration forward runs on (default `input`). The
    /// Hessian hook sees only memory-mapped quantized tensors (f16 matrices
    /// are expanded to f32 at load and lose their names), so pass the
    /// q8/q8_2f export of the same checkpoint: its activations are within
    /// noise of f16 and its tensor names match.
    pub calib_model: Option<&'a str>,
    /// Calibration corpora and their shares of `tokens`.
    pub calib: Vec<CalibSource>,
    pub output: &'a str,
    pub tokens: usize,
    /// Calibration window length (each window a fresh context).
    pub window: usize,
    pub lambda: f64,
    pub threads: usize,
    pub act_order: bool,
    /// Hessian cache: read when it exists (calibration skipped), written
    /// after calibrating when it does not. With `chunk_layers` one file per
    /// chunk, `PATH.L{first}-{last}`; an existing whole-model `PATH` serves
    /// any chunking (its entries are read lazily).
    pub hessians: Option<&'a str>,
    /// `Q4TiledP` or `Q2TiledP`: the profile whose per-tensor policy (and
    /// `--tensor-quant` overrides) decides each tensor's codec.
    pub profile: crate::convert::Quant,
    pub q2tp_rule: Q2tpRule,
    /// `--q2tp-rule-tensor PATTERN=RULE`: first match wins.
    pub q2tp_rule_overrides: Vec<(String, Q2tpRule)>,
    /// Remove each q2tp row's output shrinkage in the calibration metric
    /// (the CLI default: −11.5 % ppl on MiniCPM5's placed arm, zero bits).
    pub q2tp_debias: bool,
    /// The same for q4tp tensors (opt-in: the 4-bit shrinkage is small).
    pub q4tp_debias: bool,
    /// Capture this many layers per calibration pass (0 = all at once):
    /// bounds the Hessian RAM at the price of one forward per pass.
    pub chunk_layers: usize,
}

impl GptqTiledOpts<'_> {
    fn q2tp_rule_for(&self, name: &str) -> Q2tpRule {
        self.q2tp_rule_overrides
            .iter()
            .find(|(p, _)| crate::convert::glob_match(p, name))
            .map_or(self.q2tp_rule, |&(_, r)| r)
    }
}

/// Parse one `--q2tp-rule-tensor PATTERN=RULE`.
pub(crate) fn parse_q2tp_rule_override(spec: &str) -> anyhow::Result<(String, Q2tpRule)> {
    let (p, r) = spec
        .rsplit_once('=')
        .ok_or_else(|| anyhow::anyhow!("--q2tp-rule-tensor '{spec}': expected PATTERN=RULE"))?;
    anyhow::ensure!(!p.trim().is_empty(), "--q2tp-rule-tensor '{spec}': empty pattern");
    Ok((p.trim().to_string(), Q2tpRule::parse(r)?))
}

/// The calibration forward: one pipeline on the calibration model and the
/// token windows of every source, run once per layer chunk.
struct CalibRun {
    pipe: cortiq_engine::Pipeline,
    windows: Vec<Vec<u32>>,
    /// (file, share, tokens actually read) per source, for the provenance.
    used: Vec<(String, f64, usize)>,
}

impl CalibRun {
    fn new(o: &GptqTiledOpts) -> anyhow::Result<Self> {
        use cortiq_core::format::CmfModel;
        use cortiq_engine::{Pipeline, SamplerConfig};
        let cal_path = o.calib_model.unwrap_or(o.input);
        eprintln!("calibration model {cal_path} …");
        let cmodel = std::sync::Arc::new(CmfModel::open_sharded(cal_path)?);
        let mut pipe = Pipeline::from_model(&cmodel, SamplerConfig::default())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        pipe.set_confidence(false);
        let window = o.window.max(2);
        let budgets = split_calib_budget(&o.calib, o.tokens.max(2));
        let (mut windows, mut used) = (Vec::new(), Vec::new());
        for (src, &want) in o.calib.iter().zip(&budgets) {
            let ids = calib_source_tokens(&src.path, want, &|t| pipe.tokenizer.encode(t))?;
            if ids.len() < want {
                eprintln!("  note: {} holds only {} of its {want} tokens", src.path, ids.len());
            }
            used.push((src.path.clone(), src.share, ids.len()));
            windows.extend(ids.chunks(window).filter(|w| w.len() >= 2).map(<[u32]>::to_vec));
        }
        anyhow::ensure!(!windows.is_empty(), "calibration corpus produced too few tokens");
        let n: usize = windows.iter().map(Vec::len).sum();
        eprintln!(
            "calibrating Hessians on {n} tokens ({} windows of ≤{window}) …",
            windows.len()
        );
        Ok(Self { pipe, windows, used })
    }

    /// One forward over every window, layers `0..upto` only, with only
    /// `names` captured.
    fn capture(
        &mut self,
        names: std::collections::HashSet<String>,
        upto: usize,
        lazy: bool,
        t0: std::time::Instant,
    ) -> anyhow::Result<cortiq_engine::gptq_capture::Captured> {
        cortiq_engine::gptq_capture::begin_filtered(true, move |n| names.contains(n));
        // Positions per batched prefill step: the per-op device GEMMs of a
        // GPU calibration pay one round trip per projection per step.
        let chunk: usize = std::env::var("CMF_GPTQ_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let n_win = self.windows.len();
        let t_pass = std::time::Instant::now();
        for (k, win) in self.windows.iter().enumerate() {
            if let Err(e) = self.pipe.calib_forward(win, upto, chunk) {
                let _ = cortiq_engine::gptq_capture::end_shared();
                anyhow::bail!("calibration window {k}: {e}");
            }
            eprint!(
                "\r  window {}/{n_win}  {:.1}s/window  {:.0}s   ",
                k + 1,
                t_pass.elapsed().as_secs_f64() / (k + 1) as f64,
                t0.elapsed().as_secs_f64()
            );
        }
        eprintln!();
        let n_tok: usize = self.windows.iter().map(Vec::len).sum();
        cortiq_engine::cpuprof::report("calibration pass", n_tok);
        Ok(if lazy {
            cortiq_engine::gptq_capture::end_shared_lazy()
        } else {
            cortiq_engine::gptq_capture::end_shared()
        })
    }
}

/// Where one distinct Hessian lives until a worker needs it.
enum HSrc {
    Mem(std::sync::Mutex<Option<Vec<f64>>>),
    Cache { path: String, entry: HessCacheEntry },
    /// Still on the device (`gptq_capture::end_shared_lazy`).
    Device(std::sync::Mutex<Option<u64>>),
}

/// One distinct calibration Hessian and the linears that read it (q/k/v and
/// gate/up share their input, so the O(n³) fold operator is built once).
struct HGroup {
    slots: Vec<usize>,
    cols: usize,
    count: usize,
    sumsq: Vec<f64>,
    src: HSrc,
}

impl HGroup {
    fn take_h(&self) -> anyhow::Result<Vec<f64>> {
        match &self.src {
            HSrc::Mem(m) => m
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| anyhow::anyhow!("Hessian taken twice")),
            HSrc::Cache { path, entry } => Ok(read_hessian_entry(path, entry)?.h),
            HSrc::Device(m) => {
                let id = m.lock().unwrap().take().ok_or_else(|| anyhow::anyhow!("Hessian taken twice"))?;
                cortiq_engine::gpu::gram_take(id).ok_or_else(|| anyhow::anyhow!("device Hessian readback failed"))
            }
        }
    }
}

/// Group freshly captured Hessians by EXACT identity (shape, count, Σx² and
/// every entry), freeing duplicates as it goes. Members keep slot order, so
/// a group's lead is what it always was.
fn groups_from_capture(
    model: &cortiq_core::format::CmfModel,
    cap: cortiq_engine::gptq_capture::Captured,
    slots: &[usize],
    no_hessian: &mut Vec<usize>,
) -> Vec<HGroup> {
    let mut hess = cap.hess;
    let mut device = cap.device;
    let alias: std::collections::HashMap<String, String> = cap.alias.into_iter().collect();
    let mut groups: Vec<HGroup> = Vec::new();
    let mut group_of_owner: std::collections::HashMap<String, usize> = Default::default();
    let mut aliased: Vec<(usize, String)> = Vec::new();
    for &slot in slots {
        let entry = &model.tensors[slot];
        let cols = entry.shape[1];
        if let Some(owner) = alias.get(&entry.name) {
            aliased.push((slot, owner.clone()));
            continue;
        }
        let Some(acc) = hess.remove(&entry.name) else {
            no_hessian.push(slot);
            continue;
        };
        if let Some(id) = device.remove(&entry.name) {
            if acc.count == 0 || acc.cols != cols {
                no_hessian.push(slot);
                continue;
            }
            groups.push(HGroup {
                slots: vec![slot],
                cols,
                count: acc.count,
                sumsq: acc.sumsq,
                src: HSrc::Device(std::sync::Mutex::new(Some(id))),
            });
            group_of_owner.insert(entry.name.clone(), groups.len() - 1);
            continue;
        }
        if acc.count == 0 || acc.cols != cols || acc.h.len() != cols * cols {
            no_hessian.push(slot);
            continue;
        }
        let same = groups.iter_mut().find(|g| {
            g.cols == cols
                && g.count == acc.count
                && g.sumsq == acc.sumsq
                && matches!(&g.src, HSrc::Mem(m) if m.lock().unwrap().as_deref() == Some(&acc.h[..]))
        });
        match same {
            Some(g) => g.slots.push(slot),
            None => {
                groups.push(HGroup {
                    slots: vec![slot],
                    cols,
                    count: acc.count,
                    sumsq: acc.sumsq,
                    src: HSrc::Mem(std::sync::Mutex::new(Some(acc.h))),
                });
            }
        }
        if let Some(gi) = groups.iter().position(|g| g.slots.contains(&slot)) {
            group_of_owner.insert(entry.name.clone(), gi);
        }
    }
    // A projection that read its owner's very input shares the owner's
    // group (and fold operator).
    for (slot, owner) in aliased {
        match group_of_owner.get(&owner) {
            Some(&gi) if groups[gi].cols == model.tensors[slot].shape[1] => groups[gi].slots.push(slot),
            _ => no_hessian.push(slot),
        }
    }
    groups
}

/// Groups over a Hessian cache, read lazily: only the entries some slot
/// needs, one at a time when a worker reaches them.
fn groups_from_cache(
    model: &cortiq_core::format::CmfModel,
    path: &str,
    slots: &[usize],
    no_hessian: &mut Vec<usize>,
) -> anyhow::Result<Vec<HGroup>> {
    let index = index_hessian_cache(path)?;
    let mut by_name = std::collections::HashMap::new();
    for (i, e) in index.iter().enumerate() {
        for n in &e.names {
            by_name.insert(n.as_str(), i);
        }
    }
    let mut groups: Vec<HGroup> = Vec::new();
    let mut group_of: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &slot in slots {
        let entry = &model.tensors[slot];
        let cols = entry.shape[1];
        let hit = by_name.get(entry.name.as_str()).copied().filter(|&i| {
            let e = &index[i];
            e.count > 0 && e.cols == cols && e.hlen == cols * cols
        });
        let Some(i) = hit else {
            no_hessian.push(slot);
            continue;
        };
        let gi = *group_of.entry(i).or_insert_with(|| {
            groups.push(HGroup {
                slots: Vec::new(),
                cols,
                count: index[i].count,
                sumsq: Vec::new(),
                src: HSrc::Cache { path: path.to_string(), entry: index[i].clone() },
            });
            groups.len() - 1
        });
        groups[gi].slots.push(slot);
    }
    Ok(groups)
}

/// Write captured groups as a `CMFHESS1` cache (the format `save_hessians`
/// writes: each distinct Hessian once, under all its names).
fn save_hessian_groups(
    path: &str,
    model: &cortiq_core::format::CmfModel,
    groups: &[HGroup],
) -> anyhow::Result<()> {
    let guards: Vec<_> = groups
        .iter()
        .map(|g| match &g.src {
            HSrc::Mem(m) => Some(m.lock().unwrap()),
            // Never with a cache file: the capture reads device Hessians
            // back eagerly then.
            HSrc::Cache { .. } | HSrc::Device(_) => None,
        })
        .collect();
    let items: Vec<HessCacheItem> = groups
        .iter()
        .zip(&guards)
        .filter_map(|(g, guard)| {
            let h = guard.as_ref()?.as_deref()?;
            Some(HessCacheItem {
                names: g.slots.iter().map(|&s| model.tensors[s].name.as_str()).collect(),
                cols: g.cols,
                count: g.count,
                sumsq: &g.sumsq,
                h,
            })
        })
        .collect();
    write_hessian_cache(path, &items)
}

/// Layer index of a canonical tensor name (`…layers.{N}.…`).
fn layer_index(name: &str) -> Option<usize> {
    let rest = &name[name.find("layers.")? + "layers.".len()..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// GPTQ candidates split into calibration chunks of `chunk_layers` layers
/// (0 = one chunk). Tensors outside the layer stack ride with chunk 0.
fn gptq_layer_chunks(
    model: &cortiq_core::format::CmfModel,
    cand: &[usize],
    chunk_layers: usize,
) -> Vec<(usize, usize, Vec<usize>)> {
    if chunk_layers == 0 {
        return vec![(0, usize::MAX, cand.to_vec())];
    }
    let mut by_chunk: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for &slot in cand {
        let k = layer_index(&model.tensors[slot].name).map_or(0, |l| l / chunk_layers);
        by_chunk.entry(k).or_default().push(slot);
    }
    by_chunk
        .into_iter()
        .map(|(k, s)| (k * chunk_layers, (k + 1) * chunk_layers - 1, s))
        .collect()
}

/// Refuse a calibration pass whose dense Hessians cannot fit: `need` bytes
/// (q/k/v and gate/up are captured separately and deduplicated only after
/// the pass) against 90 % of `total`, the machine's RAM (`None` = unknown,
/// no check). The capture allocates on the first forward, so without this a
/// Qwen3.8-27B run at the default `--calib-chunk-layers 0` (~3.6 GB a layer,
/// ~230 GB) dies an OOM kill into its first CPU window. The error names a
/// chunk size whose pass takes ≤ 60 % of RAM, leaving the folds room.
fn check_capture_fits(need: f64, total: Option<f64>, layers: usize, chunk_layers: usize) -> anyhow::Result<()> {
    let Some(total) = total.filter(|&t| t > 0.0) else {
        return Ok(());
    };
    if need <= 0.9 * total {
        return Ok(());
    }
    let per_layer = need / layers.max(1) as f64;
    let fit = ((0.6 * total / per_layer) as usize).max(1);
    let now = if chunk_layers == 0 { "all layers in one pass".to_string() } else { format!("{chunk_layers} layers a pass") };
    anyhow::bail!(
        "calibration would capture ~{:.0} GB of dense Hessians ({now}, ~{:.1} GB a layer) but this machine has {:.0} GB: \
         pass --calib-chunk-layers {fit} or fewer",
        need / 1e9,
        per_layer / 1e9,
        total / 1e9
    )
}

/// How many Hessian groups to fold at once, given one group's in-flight
/// `peak` bytes. `CMF_GPTQ_MEM_GB` (`env_gb`) is the budget for the folds
/// alone, as it always was; otherwise 85 % of the machine's RAM (`total`)
/// less what the run already `held` (the chunk's Hessians, the tensors
/// written so far). Without that default the widest-first order starts the
/// widest folds together — sixteen 17408-wide `down_proj` folds of a 27B are
/// ~140 GB by this estimate on top of the chunk. `usize::MAX` = no cap.
fn fold_concurrency(peak: f64, held: f64, env_gb: Option<f64>, total: Option<f64>) -> usize {
    if !(peak > 0.0) {
        return usize::MAX;
    }
    let budget = match (env_gb, total) {
        (Some(gb), _) => gb * 1e9,
        (None, Some(t)) if t > 0.0 => 0.85 * t - held,
        _ => return usize::MAX,
    };
    ((budget / peak).max(0.0) as usize).max(1)
}

/// `cortiq quantize-gptq --codec q4tp|q2tp`: a q4tp or q2tp-profile file
/// whose linears are GPTQ-rounded against calibration Hessians instead of
/// round-to-nearest — the same layout, readers and kernels as `convert
/// --quant q4tp|q2tp`.
///
/// Every 2-D float tensor gets the converter's codec for the profile
/// (`--tensor-quant` overrides first): q4tp and q2tp linears with a Hessian
/// are GPTQ'd in that codec (so `--tensor-quant …gate_proj.weight=q4tp` on
/// a q2tp profile is an escort that stays GPTQ); the `CMF_GPTQ_SKIP` list
/// (default `embed_tokens,lm_head`), other codecs and tensors without a
/// Hessian are encoded by the converter; 1-D tensors are copied. The
/// header is the input's (tokenizer, chat template, arch) relabelled.
///
/// Deterministic: Hessians are summed in a fixed order, the fold operator
/// is single-threaded, and every row of every tensor is swept by one thread
/// in a fixed order — two runs write identical bytes whatever the thread
/// count.
pub(crate) fn run_quantize_gptq_tiled(o: &GptqTiledOpts) -> anyhow::Result<()> {
    use crate::convert::Quant;
    use cortiq_core::format::{CmfModel, TensorSpec};
    use cortiq_core::quant::dequant_tensor;
    use cortiq_core::types::{QuantType, TensorDtype};
    use std::sync::Arc;
    use std::time::Instant;

    anyhow::ensure!(
        matches!(o.profile, Quant::Q4TiledP | Quant::Q2TiledP),
        "quantize-gptq tiled path: the profile must be q4tp or q2tp"
    );
    // CMF_GPTQ_GPU=1 keeps the device for the calibration forward (a 27B
    // calibrates at ~9 tok/s on 27 host cores). The resident token graph
    // would hide every projection's input from the Hessian hook, so it is
    // switched off: the batched host prefill then sends each projection to
    // a per-op device GEMM with host-side input, and the hook folds that
    // input before the dispatch exactly as it does on the CPU.
    // `CMF_GPTQ_GPU=1|0` forces the choice; by default a build with a live
    // wgpu adapter calibrates on the card. macOS stays on the host: Metal's
    // chunk graph runs whole layer spans on the device, out of the hook's
    // sight.
    let gpu_calib = match std::env::var("CMF_GPTQ_GPU").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => {
            !cfg!(target_os = "macos")
                && std::env::var("CMF_GPU").as_deref() != Ok("0")
                && cortiq_engine::gpu::backend_available()
        }
    };
    eprintln!(
        "calibration device: {}",
        if gpu_calib { "GPU (per-op device GEMMs, device Hessians)" } else { "CPU" }
    );
    // …and the Hessians themselves accumulate on the card (CMF_GPTQ_GPU_HESS=0
    // keeps the host fold).
    cortiq_engine::gptq_capture::set_device(
        gpu_calib && std::env::var("CMF_GPTQ_GPU_HESS").as_deref() != Ok("0"),
    );
    unsafe {
        if !gpu_calib {
            std::env::set_var("CMF_GPU", "0");
        } else if std::env::var("CMF_GPU_WGPU_GRAPH").is_err() {
            std::env::set_var("CMF_GPU_WGPU_GRAPH", "0");
        }
    }
    let t0 = Instant::now();
    let codec = crate::convert::quant_name(o.profile);
    let model = Arc::new(CmfModel::open_sharded(o.input)?);
    let arch = model.arch().clone();
    let skip_names: Vec<String> = std::env::var("CMF_GPTQ_SKIP")
        .unwrap_or_else(|_| "embed_tokens,lm_head".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let quant_of = |name: &str| crate::convert::effective_quant(&arch, o.profile, name);

    // GPTQ candidates vs tensors left to the converter's policy.
    let (mut cand, mut rtn) = (Vec::new(), Vec::new());
    for (slot, entry) in model.tensors.iter().enumerate() {
        let two_d = entry.shape.len() == 2 && entry.shape[1] % GROUP_SIZE == 0;
        let keep = skip_names.iter().any(|s| entry.name.contains(s.as_str()));
        let float_src = matches!(
            entry.dtype,
            TensorDtype::F16 | TensorDtype::F32 | TensorDtype::Bf16
        );
        if two_d
            && !keep
            && float_src
            && matches!(quant_of(&entry.name), Quant::Q4TiledP | Quant::Q2TiledP)
            && !crate::convert::keeps_float(&entry.name)
        {
            cand.push(slot);
        } else {
            rtn.push(slot);
        }
    }
    if o.act_order && o.profile == Quant::Q2TiledP {
        // Act order must fix each group's rung before its channels arrive;
        // at 2 bits the rung chosen from the error-corrected weights is
        // worth more than the order (MiniCPM5: 42.2 vs 31.3 ppl).
        eprintln!("warning: --act-order fixes q2tp rungs up front — measured worse than natural order at 2 bits");
    }
    let chunks = gptq_layer_chunks(&model, &cand, o.chunk_layers);
    eprintln!(
        "GPTQ ({codec}): {} candidate linears in {} calibration pass(es); {} tensors by the converter policy",
        cand.len(),
        chunks.len(),
        rtn.len()
    );

    let mut calib: Option<CalibRun> = None;
    let mut no_hessian: Vec<usize> = Vec::new();
    let ram_total = cortiq_engine::fcd::physical_total_bytes().map(|b| b as f64);
    let results = std::sync::Mutex::new(Vec::<(usize, TensorDtype, Vec<u8>)>::new());
    let n_gptq = std::sync::atomic::AtomicUsize::new(0);
    let failed = std::sync::Mutex::new(Vec::<String>::new());
    for (lo, hi, slots) in &chunks {
        let missed_before = no_hessian.len();
        let cache = o.hessians.map(|p| {
            if o.chunk_layers == 0 {
                p.to_string()
            } else {
                format!("{p}.L{lo}-{hi}")
            }
        });
        // A whole-model cache serves any chunking: entries are read lazily.
        let existing = cache
            .as_deref()
            .filter(|p| std::path::Path::new(p).exists())
            .or(o.hessians.filter(|p| std::path::Path::new(p).exists()));
        let groups = match existing {
            Some(p) => {
                eprintln!("Hessians from cache {p} (calibration skipped) …");
                groups_from_cache(&model, p, slots, &mut no_hessian)?
            }
            _ => {
                let names: std::collections::HashSet<String> =
                    slots.iter().map(|&s| model.tensors[s].name.clone()).collect();
                let need: f64 = slots
                    .iter()
                    .map(|&s| (model.tensors[s].shape[1] as f64).powi(2) * 8.0)
                    .sum();
                let layers: std::collections::HashSet<Option<usize>> =
                    slots.iter().map(|&s| layer_index(&model.tensors[s].name)).collect();
                check_capture_fits(need, ram_total, layers.len(), o.chunk_layers)?;
                let gb = need / 1e9;
                if calib.is_none() {
                    calib = Some(CalibRun::new(o)?);
                }
                if o.chunk_layers > 0 {
                    eprintln!("layers {lo}..={hi}: capturing {} linears (≤{gb:.1} GB of Hessians) …", slots.len());
                } else {
                    eprintln!("capturing {} linears (≤{gb:.1} GB of Hessians) …", slots.len());
                }
                // Device Hessians stay on the card until their fold reads
                // them, unless a cache file wants them all on the host.
                let lazy = cache.is_none();
                let cap = calib.as_mut().unwrap().capture(names, hi.saturating_add(1), lazy, t0)?;
                anyhow::ensure!(
                    cap.broken.is_empty() || std::env::var("CMF_GPTQ_ALLOW_RTN").as_deref() == Ok("1"),
                    "calibration capture invalid for {:?} (CMF_GPTQ_ALLOW_RTN=1 encodes them by the converter)",
                    cap.broken
                );
                let groups = groups_from_capture(&model, cap, slots, &mut no_hessian);
                if let Some(p) = cache.as_deref() {
                    save_hessian_groups(p, &model, &groups)?;
                    eprintln!("Hessians cached to {p}");
                }
                groups
            }
        };
        eprintln!(
            "captured input Hessians for {} linears in {} groups ({:.0}s)",
            groups.iter().map(|g| g.slots.len()).sum::<usize>(),
            groups.len(),
            t0.elapsed().as_secs_f64()
        );
        // A candidate the calibration forward never reached falls back to
        // the converter's RTN; name them, so a forward path that bypasses
        // the hook cannot pass for GPTQ.
        let missed: Vec<&str> = no_hessian[missed_before..]
            .iter()
            .map(|&s| model.tensors[s].name.as_str())
            .collect();
        if !missed.is_empty() {
            eprintln!(
                "  WARNING: {} candidates got no Hessian (RTN): {}{}",
                missed.len(),
                missed.iter().take(40).copied().collect::<Vec<_>>().join(", "),
                if missed.len() > 40 { ", …" } else { "" }
            );
        }

        // Workers take whole groups: the O(n³) fold operator dominates, so
        // many groups run at once and each fold and row sweep gets the
        // threads left over. A group holds ~3·n² f64 + n² f32 at its peak
        // (1 GB at n = 6144, 8 GB at 17408), so [`fold_concurrency`] caps how
        // many run together by the RAM left (`CMF_GPTQ_MEM_GB` overrides),
        // trading parallel groups for parallel folds — the bytes are the same.
        let peak = groups.iter().map(|g| 28.0 * (g.cols as f64).powi(2)).fold(0.0, f64::max);
        let env_gb = std::env::var("CMF_GPTQ_MEM_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|&gb| gb > 0.0);
        // Already resident: the chunk's captured Hessians and every tensor
        // written so far (the output is assembled in RAM).
        let held = groups
            .iter()
            .filter(|g| matches!(g.src, HSrc::Mem(_)))
            .map(|g| 8.0 * (g.cols as f64).powi(2))
            .sum::<f64>()
            + results.lock().unwrap().iter().map(|r| r.2.len() as f64).sum::<f64>();
        let mem_cap = fold_concurrency(peak, held, env_gb, ram_total);
        let outer = o.threads.clamp(1, 16).min(groups.len().max(1)).min(mem_cap);
        let inner = (o.threads / outer).max(1);
        // Widest Hessians first (longest-processing-time order): a 6144-wide
        // fold costs ~27× a 2048-wide one, and one started last is the tail.
        // The order cannot change any byte — results are keyed by slot.
        // Device groups: a Hessian still on the card whose members are all
        // q4tp (natural order) is folded and swept there by one thread
        // (`gpu::gptq_fold_dev`); CMF_GPTQ_GPU_FOLD=0 keeps the host fold.
        let dev_fold = std::env::var("CMF_GPTQ_GPU_FOLD").as_deref() != Ok("0") && !o.act_order;
        let on_device = |g: &HGroup| {
            dev_fold
                && matches!(g.src, HSrc::Device(_))
                && g.slots.iter().all(|&s| {
                    let name = &model.tensors[s].name;
                    match quant_of(name) {
                        Quant::Q4TiledP => true,
                        Quant::Q2TiledP => o.q2tp_rule_for(name) == Q2tpRule::Placed,
                        _ => false,
                    }
                })
        };
        let mut order: Vec<usize> = (0..groups.len()).filter(|&g| !on_device(&groups[g])).collect();
        order.sort_by_key(|&g| std::cmp::Reverse(groups[g].cols));
        let mut dev_order: Vec<usize> = (0..groups.len()).filter(|&g| on_device(&groups[g])).collect();
        dev_order.sort_by_key(|&g| std::cmp::Reverse(groups[g].cols));
        if !dev_order.is_empty() {
            eprintln!("  {} Hessian groups fold on the device, {} on the host", dev_order.len(), order.len());
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let n_chunk: usize = groups.iter().map(|g| g.slots.len()).sum();
        let done = std::sync::atomic::AtomicUsize::new(0);
        // One Hessian group on the host: fold operator, then every member's
        // sweep (`inner` threads each).
        let host_group = |gi: usize, inner: usize| {
                        let grp = &groups[gi];
                        let h = match grp.take_h() {
                            Ok(h) => h,
                            Err(e) => {
                                failed.lock().unwrap().push(format!("group {gi}: {e}"));
                                return;
                            }
                        };
                        let n = grp.cols;
                        let hdiag: Vec<f32> = (0..n).map(|i| h[i * n + i] as f32).collect();
                        let quants: Vec<Quant> =
                            grp.slots.iter().map(|&s| quant_of(&model.tensors[s].name)).collect();
                        let keep_h = ((o.q2tp_debias && quants.contains(&Quant::Q2TiledP))
                            || (o.q4tp_debias && quants.contains(&Quant::Q4TiledP)))
                        .then(|| h.clone());
                        let perm = o.act_order.then(|| act_order_perm(&hdiag));
                        let (u, dead) = match &perm {
                            Some(p) => {
                                let hp = permute_sym(&h, n, p);
                                drop(h);
                                let (u, _) = gptq_fold_operator_par(hp, n, o.lambda, inner);
                                (u, hdiag.iter().map(|&d| !(d > 0.0)).collect::<Vec<bool>>())
                            }
                            None => gptq_fold_operator_par(h, n, o.lambda, inner),
                        };
                        for (&slot, &q) in grp.slots.iter().zip(&quants) {
                            let entry = &model.tensors[slot];
                            let (rows, cols) = (entry.shape[0], entry.shape[1]);
                            let mut w = vec![0f32; rows * cols];
                            if let Err(e) = dequant_tensor(entry, model.entry_bytes(entry), &mut w) {
                                failed.lock().unwrap().push(format!("{}: {e}", entry.name));
                                continue;
                            }
                            let out = match (q, &perm) {
                                (Quant::Q4TiledP, _) => {
                                    let mut b = match &perm {
                                        Some(p) => gptq_quantize_q4tp_act_order(&w, rows, cols, &u, &dead, p, inner),
                                        None => gptq_quantize_q4tp(&w, rows, cols, &u, &dead, &hdiag, inner),
                                    };
                                    if o.q4tp_debias
                                        && let Some(h) = &keep_h
                                    {
                                        q4tp_debias_gemm(&mut b, &w, rows, cols, h, inner);
                                    }
                                    (TensorDtype::Q4TiledP, b)
                                }
                                _ => {
                                    let rule = o.q2tp_rule_for(&entry.name);
                                    let mut b = gptq_quantize_q2tp(
                                        &w,
                                        rows,
                                        cols,
                                        &u,
                                        &dead,
                                        &hdiag,
                                        rule,
                                        perm.as_deref(),
                                        inner,
                                    );
                                    if o.q2tp_debias
                                        && let Some(h) = &keep_h
                                    {
                                        q2tp_debias_gemm(&mut b, &w, rows, cols, h, inner);
                                    }
                                    (TensorDtype::Q2TiledP, b)
                                }
                            };
                            results.lock().unwrap().push((slot, out.0, out.1));
                            n_gptq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                            eprint!("\r  gptq {k}/{n_chunk}  {:.0}s   ", t0.elapsed().as_secs_f64());
                        }
        };
        std::thread::scope(|s| {
            if !dev_order.is_empty() {
                s.spawn(|| {
                    for &gi in &dev_order {
                        let grp = &groups[gi];
                        let HSrc::Device(m) = &grp.src else { continue };
                        let Some(hid) = m.lock().unwrap().take() else { continue };
                        let n = grp.cols;
                        let quants: Vec<Quant> =
                            grp.slots.iter().map(|&s| quant_of(&model.tensors[s].name)).collect();
                        // q2tp debias reads the Hessian itself: take a host
                        // copy before the fold frees it.
                        let keep_h = if o.q2tp_debias && quants.contains(&Quant::Q2TiledP) {
                            cortiq_engine::gpu::gram_read(hid)
                        } else {
                            None
                        };
                        let Some(fid) = cortiq_engine::gpu::gptq_fold_dev(hid, n, o.lambda) else {
                            // Not on the card after all: hand the group to
                            // the host workers with its Hessian read back.
                            eprintln!("\n  group {gi} (n = {n}): device fold declined — host fold");
                            *m.lock().unwrap() = Some(hid);
                            host_group(gi, o.threads.max(1));
                            continue;
                        };
                        let (dead, hdiag, _lam) = cortiq_engine::gpu::gptq_fold_info(fid).unwrap_or_default();
                        for (&slot, &q) in grp.slots.iter().zip(&quants) {
                            let entry = &model.tensors[slot];
                            let (rows, cols) = (entry.shape[0], entry.shape[1]);
                            let mut w = vec![0f32; rows * cols];
                            if let Err(e) = dequant_tensor(entry, model.entry_bytes(entry), &mut w) {
                                failed.lock().unwrap().push(format!("{}: {e}", entry.name));
                                continue;
                            }
                            let out = if q == Quant::Q2TiledP {
                                gptq_quantize_q2tp_dev(&w, rows, cols, fid, &dead, &hdiag, o.threads).map(|mut b| {
                                    if let Some(h) = &keep_h {
                                        q2tp_debias_gemm(&mut b, &w, rows, cols, h, o.threads.max(1));
                                    }
                                    (TensorDtype::Q2TiledP, b)
                                })
                            } else {
                                gptq_quantize_q4tp_dev(&w, rows, cols, fid, &dead).map(|b| (TensorDtype::Q4TiledP, b))
                            };
                            match out {
                                Some((dt, b)) => {
                                    results.lock().unwrap().push((slot, dt, b));
                                    n_gptq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                                None => {
                                    failed.lock().unwrap().push(format!("{}: device sweep failed", entry.name));
                                    continue;
                                }
                            }
                            let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                            eprint!("\r  gptq {k}/{n_chunk}  {:.0}s   ", t0.elapsed().as_secs_f64());
                        }
                        cortiq_engine::gpu::gptq_fold_free(fid);
                    }
                });
            }
            for _ in 0..outer.min(order.len().max(1)) {
                s.spawn(|| {
                    loop {
                        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&gi) = order.get(k) else { break };
                        host_group(gi, inner);
                    }
                });
            }
        });
        eprintln!();
        let failed = failed.lock().unwrap();
        anyhow::ensure!(failed.is_empty(), "GPTQ failed: {failed:?}");
    }
    let calib_used = calib.as_ref().map(|c| c.used.clone());
    drop(calib);
    if !no_hessian.is_empty() {
        eprintln!(
            "  {} candidate linears had no Hessian — encoded by the converter policy",
            no_hessian.len()
        );
    }

    // Converter-policy tensors (overrides, the profile's RTN, f16 copies).
    let mut specs: Vec<Option<TensorSpec>> = (0..model.tensors.len()).map(|_| None).collect();
    rtn.extend(no_hessian);
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
            crate::convert::quantize_2d(quant_of(&entry.name), &w, entry.shape[0], entry.shape[1])
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
    for (slot, dtype, data) in results.into_inner().unwrap() {
        let entry = &model.tensors[slot];
        specs[slot] = Some(TensorSpec {
            name: entry.name.clone(),
            dtype,
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
    let n_gptq = n_gptq.into_inner();

    let mut header = model.header.clone();
    header.quant_type = QuantType::Q4Block;
    header.section_hashes = None;
    let mut prov = header
        .provenance
        .clone()
        .unwrap_or_else(|| serde_json::json!({}));
    prov["weight_quant"] = serde_json::json!(codec);
    let mut gptq = serde_json::json!({
        "codec": codec,
        "calibration_tokens": o.tokens,
        "window": o.window,
        "lambda": o.lambda,
        "act_order": o.act_order,
        "linears": n_gptq,
    });
    // Keys beyond the original q4tp record only when they carry information,
    // so a plain q4tp run keeps writing the header it always wrote.
    if o.profile == Quant::Q2TiledP {
        gptq["q2tp_rule"] = serde_json::json!(o.q2tp_rule.name());
        gptq["q2tp_debias"] = serde_json::json!(o.q2tp_debias);
        if !o.q2tp_rule_overrides.is_empty() {
            gptq["q2tp_rule_overrides"] = serde_json::json!(o
                .q2tp_rule_overrides
                .iter()
                .map(|(p, r)| format!("{p}={}", r.name()))
                .collect::<Vec<_>>());
        }
    }
    if o.q4tp_debias {
        gptq["q4tp_debias"] = serde_json::json!(true);
    }
    if o.calib.len() > 1 {
        if let Some(used) = &calib_used {
            gptq["calib"] = serde_json::json!(used
                .iter()
                .map(|(p, share, n)| serde_json::json!({
                    "file": std::path::Path::new(p).file_name().map(|f| f.to_string_lossy().into_owned()),
                    "share": share,
                    "tokens": n,
                }))
                .collect::<Vec<_>>());
        }
    }
    if o.chunk_layers > 0 {
        gptq["calib_chunk_layers"] = serde_json::json!(o.chunk_layers);
    }
    prov["gptq"] = gptq;
    header.provenance = Some(prov);
    CmfModel::write(o.output, &header, &specs, None, model.vocab.as_deref())
        .map_err(|e| anyhow::anyhow!("write {}: {e}", o.output))?;
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
/// is a contiguous dot product. The reference [`cholesky_lower_par`] must
/// reproduce bit for bit.
#[cfg(test)]
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

/// Raw view of a row-major f64 matrix shared by the fold workers. Within a
/// phase every element has at most one writer, and nobody reads an element
/// another worker writes in the same phase; phases are fenced by barriers.
#[derive(Clone, Copy)]
struct SharedMat(*mut f64);
// SAFETY: see the access discipline above; the pointer outlives the scope.
unsafe impl Send for SharedMat {}
unsafe impl Sync for SharedMat {}

/// Columns per block of the cache-blocked fold phases. While a block's rows
/// of `L` (or `L⁻¹`) stay cache-resident every other row streams past once
/// per block instead of once per column, so a 6144-wide Cholesky moves
/// ~10 GB from DRAM instead of ~310 GB — what made 13 concurrent folds
/// starve each other (one fold took ~3 min of a 7.5-min GPTQ pass).
const FOLD_BLOCK: usize = 32;

/// The row-oriented lower Cholesky (`cholesky_lower_rows`, kept as the test
/// reference), cache-blocked and on `threads` workers:
/// left-looking by blocks of [`FOLD_BLOCK`] columns — the block's diagonal
/// part on one worker, then the panel below it with rows dealt round-robin,
/// two barriers per block. Every entry is the same `dot64` over the same
/// slices, divided by the same pivot, as in the serial version, so the
/// factor is bit-identical whatever the thread count (the upper triangle is
/// zeroed at the end instead of row by row).
fn cholesky_lower_par(a: &mut [f64], n: usize, threads: usize) -> bool {
    let t = threads.clamp(1, n.max(1));
    let failed = std::sync::atomic::AtomicBool::new(false);
    let barrier = std::sync::Barrier::new(t);
    let m = SharedMat(a.as_mut_ptr());
    // SAFETY (all accesses below): row prefixes `0..j` read here are final
    // (written in an earlier block, or earlier in this block by the same
    // worker before a barrier); each entry is written by exactly one worker
    // — diagonal blocks by worker 0, panel rows by their owner `i % t`.
    let entry = move |i: usize, j: usize| -> f64 {
        let m = m;
        let (ri, rj) = unsafe {
            (
                std::slice::from_raw_parts(m.0.add(i * n), j),
                std::slice::from_raw_parts(m.0.add(j * n), j),
            )
        };
        let hij = unsafe { *m.0.add(i * n + j) };
        hij - dot64(ri, rj)
    };
    let (failed_r, barrier_r) = (&failed, &barrier);
    let work = move |w: usize| {
        let (m, failed, barrier) = (m, failed_r, barrier_r);
        let mut jb = 0;
        while jb < n {
            let je = (jb + FOLD_BLOCK).min(n);
            if w == 0 {
                'diag: for i in jb..je {
                    for j in jb..=i {
                        let s = entry(i, j);
                        if i == j {
                            if !(s > 0.0) || !s.is_finite() {
                                failed.store(true, std::sync::atomic::Ordering::Relaxed);
                                break 'diag;
                            }
                            unsafe { *m.0.add(i * n + i) = s.sqrt() };
                        } else {
                            unsafe { *m.0.add(i * n + j) = s / *m.0.add(j * n + j) };
                        }
                    }
                }
            }
            barrier.wait();
            if failed.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            let mut i = je + (w + t - je % t) % t;
            while i < n {
                for j in jb..je {
                    let s = entry(i, j);
                    unsafe { *m.0.add(i * n + j) = s / *m.0.add(j * n + j) };
                }
                i += t;
            }
            barrier.wait();
            jb = je;
        }
    };
    if t == 1 {
        work(0);
    } else {
        std::thread::scope(|s| {
            for w in 0..t {
                let work = &work;
                s.spawn(move || work(w));
            }
        });
    }
    if failed.into_inner() {
        return false;
    }
    for i in 0..n {
        a[i * n + i + 1..(i + 1) * n].fill(0.0);
    }
    true
}

/// Run `f(i)` for `i in 0..n`, dealt round-robin over `threads` workers
/// (`f` must only write state owned by `i`).
fn par_rows(n: usize, threads: usize, f: &(dyn Fn(usize) + Sync)) {
    let t = threads.clamp(1, n.max(1));
    if t == 1 {
        (0..n).for_each(f);
        return;
    }
    std::thread::scope(|s| {
        for w in 0..t {
            s.spawn(move || {
                let mut i = w;
                while i < n {
                    f(i);
                    i += t;
                }
            });
        }
    });
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
/// (Single-threaded form; the runner calls [`gptq_fold_operator_par`].)
#[cfg(test)]
pub fn gptq_fold_operator(h: Vec<f64>, n: usize, lambda: f64) -> (Vec<f32>, Vec<bool>) {
    gptq_fold_operator_par(h, n, lambda, 1)
}

/// [`gptq_fold_operator`] on `threads` workers, every phase cache-blocked
/// ([`FOLD_BLOCK`]) — bit-identical to the original serial operator for any
/// thread count (every entry is the same `dot64` over the same slices,
/// computed by one worker). It is O(n³): ~0.15 T multiply-adds for a
/// 6144-wide `down_proj`, ~3.5 T at 17408 (Qwen3.8-27B).
pub fn gptq_fold_operator_par(mut h: Vec<f64>, n: usize, lambda: f64, threads: usize) -> (Vec<f32>, Vec<bool>) {
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
        if cholesky_lower_par(&mut a, n, threads) {
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
    // Columns of L⁻¹, stored as rows: lt[j][i] = (L⁻¹)[i][j], i ≥ j. Each
    // column is its own forward substitution; a block of FOLD_BLOCK columns
    // is solved together (row i of L read once for all of them), blocks in
    // parallel. Each column's sequence of dot products is the serial one.
    let mut lt = vec![0f64; n * n];
    let nblk = n.div_ceil(FOLD_BLOCK);
    {
        let (av, ltp) = (&a, SharedMat(lt.as_mut_ptr()));
        par_rows(nblk, threads, &move |b| {
            let ltp = ltp;
            let (c0, c1) = (b * FOLD_BLOCK, ((b + 1) * FOLD_BLOCK).min(n));
            for i in c0..n {
                let ai = &av[i * n..i * n + i + 1];
                for j in c0..c1.min(i + 1) {
                    // SAFETY: block b writes only rows c0..c1 of `lt`.
                    let x = unsafe { std::slice::from_raw_parts_mut(ltp.0.add(j * n), n) };
                    let v = if i == j {
                        1.0 / ai[i]
                    } else {
                        -dot64(&ai[j..i], &x[j..i]) / ai[i]
                    };
                    x[i] = v;
                }
            }
        });
    }
    // H⁻¹ = L⁻ᵀ·L⁻¹: (H⁻¹)[i][k] = Σ_{m ≥ i} lt[i][m]·lt[k][m] for k ≤ i,
    // a block of FOLD_BLOCK rows at a time (each lt row k read once for the
    // whole block), blocks in parallel.
    {
        let (ltv, ap) = (&lt, SharedMat(a.as_mut_ptr()));
        par_rows(nblk, threads, &move |b| {
            let ap = ap;
            let (i0, i1) = (b * FOLD_BLOCK, ((b + 1) * FOLD_BLOCK).min(n));
            for k in 0..i1 {
                let lk = &ltv[k * n..(k + 1) * n];
                for i in i0.max(k)..i1 {
                    let v = dot64(&ltv[i * n + i..i * n + n], &lk[i..n]);
                    // SAFETY: block b writes only rows i0..i1 of `a`.
                    unsafe { *ap.0.add(i * n + k) = v };
                }
            }
        });
    }
    drop(lt);
    if !cholesky_lower_par(&mut a, n, threads) {
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

/// [`gptq_quantize_q4tp`] on the device against a device fold
/// (`gpu::gptq_fold_dev`): the same ladder (the converter's, from the
/// dead-zeroed weights), rung rule and nibble layout; f32 instead of the
/// host's f64 operator, so codes can differ at rounding boundaries.
pub(crate) fn gptq_quantize_q4tp_dev(w0: &[f32], rows: usize, cols: usize, fold: u64, dead: &[bool]) -> Option<Vec<u8>> {
    use cortiq_core::quant::{q4tp_ladder, q4tp_put_code, q4tp_sections};
    if cols % GROUP_SIZE != 0 || dead.len() != cols {
        return None;
    }
    let gpr = cols / GROUP_SIZE;
    let mut w_init = w0.to_vec();
    if dead.iter().any(|&d| d) {
        for row in w_init.chunks_exact_mut(cols) {
            for (v, &d) in row.iter_mut().zip(dead) {
                if d {
                    *v = 0.0;
                }
            }
        }
    }
    let mut out = crate::convert::encode_q4tp(&w_init, rows, cols);
    let (params_off, codes_off, stride) = q4tp_sections(rows, cols);
    if params_off != rows * cols / 2 {
        return None;
    }
    let params = out[params_off..params_off + rows * 4].to_vec();
    let mut tab = vec![0f32; rows * 32];
    let mut lost = vec![0f32; rows * 2];
    for r in 0..rows {
        tab[r * 32..r * 32 + 32].copy_from_slice(&q4tp_ladder(&params, r));
        let p = &params[r * 4..r * 4 + 4];
        lost[r * 2] = f16_to_f32(u16::from_le_bytes([p[0], p[1]]));
        lost[r * 2 + 1] = f16_to_f32(u16::from_le_bytes([p[2], p[3]]));
    }
    let mut codes = vec![0u32; rows * gpr];
    if !cortiq_engine::gpu::gptq_sweep_q4tp_dev(fold, &w_init, rows, &tab, &lost, &mut out[..params_off], &mut codes) {
        return None;
    }
    let cs = &mut out[codes_off..codes_off + rows * stride];
    cs.fill(0);
    for r in 0..rows {
        for g in 0..gpr {
            q4tp_put_code(&mut cs[r * stride..(r + 1) * stride], g, codes[r * gpr + g] as usize);
        }
    }
    Some(out)
}

/// [`gptq_quantize_q2tp`] with the `placed` rule on the device against a
/// device fold: ladders placed on the host (from the dead-zeroed weights, as
/// the host sweep places them), the sweep and its rung search on the card.
pub(crate) fn gptq_quantize_q2tp_dev(
    w0: &[f32],
    rows: usize,
    cols: usize,
    fold: u64,
    dead: &[bool],
    hdiag: &[f32],
    threads: usize,
) -> Option<Vec<u8>> {
    use crate::convert::{det_q2tp_ladder, q2tp_place_row};
    use cortiq_core::quant::{q2tp_sections, q4tp_put_code};
    if cols % GROUP_SIZE != 0 || dead.len() != cols || hdiag.len() != cols {
        return None;
    }
    let gpr = cols / GROUP_SIZE;
    let mut w_init = w0.to_vec();
    if dead.iter().any(|&d| d) {
        for row in w_init.chunks_exact_mut(cols) {
            for (v, &d) in row.iter_mut().zip(dead) {
                if d {
                    *v = 0.0;
                }
            }
        }
    }
    let hw: Vec<f32> = hdiag.iter().map(|&h| if h > 0.0 { h } else { 0.0 }).collect();
    let (params_off, codes_off, stride) = q2tp_sections(rows, cols);
    if params_off != rows * cols / 4 {
        return None;
    }
    let mut out = vec![0u8; codes_off + rows * stride];
    {
        let params = &mut out[params_off..params_off + rows * 4];
        let per = rows.div_ceil(threads.max(1));
        let (w_init, hw) = (&w_init, &hw);
        std::thread::scope(|s| {
            for (k, chunk) in params.chunks_mut(per * 4).enumerate() {
                s.spawn(move || {
                    let mut rungs = vec![0u8; gpr];
                    for (i, p) in chunk.chunks_exact_mut(4).enumerate() {
                        let r = k * per + i;
                        q2tp_place_row(&w_init[r * cols..(r + 1) * cols], hw, p, &mut rungs);
                    }
                });
            }
        });
    }
    let params = out[params_off..params_off + rows * 4].to_vec();
    let mut tab = vec![0f32; rows * 32];
    for r in 0..rows {
        tab[r * 32..r * 32 + 32].copy_from_slice(&det_q2tp_ladder(&params, r));
    }
    let mut codes = vec![0u32; rows * gpr];
    if !cortiq_engine::gpu::gptq_sweep_q2tp_dev(fold, &w_init, rows, &tab, &hw, &mut out[..params_off], &mut codes) {
        return None;
    }
    let cs = &mut out[codes_off..codes_off + rows * stride];
    cs.fill(0);
    for r in 0..rows {
        for g in 0..gpr {
            q4tp_put_code(&mut cs[r * stride..(r + 1) * stride], g, codes[r * gpr + g] as usize);
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// GPTQ for q2tp
// ---------------------------------------------------------------------------

/// How `quantize-gptq --codec q2tp` places each group's scale.
///
/// Measured on MiniCPM5-2B (gate/up q2tp, 32k calibration tokens, WikiText-2
/// ppl; bf16 18.9): `cortiq` 37.1, `placed` 31.7. The plain converter is 207.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q2tpRule {
    /// The q4tp rule moved to the 4-level grid: the v1 converter's ladder;
    /// when the sweep reaches a group its rung is the best of nominal−2..+1
    /// (nominal from the corrected group's absmax/1.5) by `H_ii`-weighted
    /// rounding error. Outer level near absmax: unbiased, but coarse.
    Cortiq,
    /// The ladder placed at the `H_ii`-weighted MSE-optimal group scales
    /// ([`crate::convert::q2tp_place_row`]) and a full 31-rung search when
    /// the sweep reaches each group.
    Placed,
}

impl Q2tpRule {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cortiq" | "absmax" => Ok(Self::Cortiq),
            "placed" | "full" => Ok(Self::Placed),
            other => anyhow::bail!("q2tp rule '{other}': expected cortiq or placed"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Cortiq => "cortiq",
            Self::Placed => "placed",
        }
    }
}

/// Columns per lazy-update block of the q2tp sweep (a multiple of 32).
const Q2_BLOCK: usize = 128;
/// Rows per work item. The trailing update streams one `Q2_BLOCK × cols`
/// slab of the fold operator per block; more rows per item reuse it more.
const Q2_ROWS: usize = 32;
/// Rows per register tile of the trailing update.
const Q2_SLAB: usize = 8;

/// GPTQ-quantize `W [rows, cols]` to `q2tp` bytes — the layout, reader and
/// kernels of `convert --quant q2tp`, only better-chosen codes, rungs and
/// (for [`Q2tpRule::Placed`]) ladders.
///
/// `u` is the fold operator of [`gptq_fold_operator`] (of the PERMUTED
/// Hessian when `perm` is given), `hdiag` the Hessian diagonal in natural
/// channel order, `dead` the channels that never fired (zeroed). Columns are
/// eliminated left to right — or in `perm` order (act order: loudest inputs
/// first), where a group's rung is fixed up front from the original weights
/// ("static groups") because its channels arrive out of order.
///
/// The sweep is blocked: inside a block of [`Q2_BLOCK`] columns each
/// residual is folded into the block at once, into the rest of the row once
/// per block by a register-tiled update. Rows are independent given `u`, so
/// work items of [`Q2_ROWS`] rows run on `threads` workers; each item is
/// computed by one thread in a fixed order, so the bytes do not depend on
/// the thread count.
#[allow(clippy::too_many_arguments)]
pub fn gptq_quantize_q2tp(
    w0: &[f32],
    rows: usize,
    cols: usize,
    u: &[f32],
    dead: &[bool],
    hdiag: &[f32],
    rule: Q2tpRule,
    perm: Option<&[usize]>,
    threads: usize,
) -> Vec<u8> {
    use cortiq_core::quant::{Q2TP_CHUNK, q2tp_sections};
    assert_eq!(w0.len(), rows * cols);
    assert_eq!(cols % GROUP_SIZE, 0);
    assert_eq!(u.len(), cols * cols);
    assert_eq!(hdiag.len(), cols);
    if let Some(p) = perm {
        assert_eq!(p.len(), cols);
    }
    let gpr = cols / GROUP_SIZE;
    let mut w_init = w0.to_vec();
    if dead.iter().any(|&d| d) {
        for row in w_init.chunks_exact_mut(cols) {
            for (x, &d) in row.iter_mut().zip(dead) {
                if d {
                    *x = 0.0;
                }
            }
        }
    }
    let hw: Vec<f32> = hdiag.iter().map(|&h| if h > 0.0 { h } else { 0.0 }).collect();
    let (params_off, codes_off, stride) = q2tp_sections(rows, cols);
    let mut out = vec![0u8; codes_off + rows * stride];
    // The cortiq rule starts from the v1 converter's ladder — and, in act
    // order, its rungs — on the dead-zeroed weights.
    let v1 = (rule == Q2tpRule::Cortiq)
        .then(|| crate::convert::encode_q2tp_v1(&w_init, rows, cols, true));
    let (chunks_all, rest) = out.split_at_mut(params_off);
    let (params_all, codes_all) = rest.split_at_mut(rows * 4);
    let jobs: Vec<(usize, &mut [u8], &mut [u8], &mut [u8])> = chunks_all
        .chunks_mut(Q2_ROWS * gpr * Q2TP_CHUNK)
        .zip(params_all.chunks_mut(Q2_ROWS * 4))
        .zip(codes_all.chunks_mut(Q2_ROWS * stride))
        .enumerate()
        .map(|(b, ((c, p), k))| (b * Q2_ROWS, c, p, k))
        .collect();
    let queue = std::sync::Mutex::new(jobs);
    let sweep = Q2Sweep {
        w_init: &w_init,
        rows,
        cols,
        u,
        hw: &hw,
        rule,
        perm,
        v1: v1.as_deref(),
    };
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                loop {
                    let job = queue.lock().unwrap().pop();
                    let Some((r0, chunks, params, codes)) = job else { break };
                    sweep.rows(r0, chunks, params, codes);
                }
            });
        }
    });
    out
}

/// Shared, read-only inputs of one tensor's q2tp sweep.
struct Q2Sweep<'a> {
    w_init: &'a [f32],
    rows: usize,
    cols: usize,
    u: &'a [f32],
    hw: &'a [f32],
    rule: Q2tpRule,
    perm: Option<&'a [usize]>,
    v1: Option<&'a [u8]>,
}

impl Q2Sweep<'_> {
    /// Quantize rows `r0..` (as many as `params` holds) into their planes.
    fn rows(&self, r0: usize, chunks: &mut [u8], params: &mut [u8], codes: &mut [u8]) {
        use crate::convert::{det_q2tp_ladder, q2_level, q2tp_place_row};
        use cortiq_core::quant::{Q2TP_CHUNK, q2tp_sections, q4tp_code, q4tp_put_code};
        let cols = self.cols;
        let gpr = cols / GROUP_SIZE;
        let stride = cortiq_core::quant::q4tp_code_stride(gpr);
        let rb = params.len() / 4;
        // Ladders, and the rungs a static (act-order) sweep keeps.
        let mut rungs = vec![0u8; rb * gpr];
        match self.rule {
            Q2tpRule::Cortiq => {
                let v1 = self.v1.expect("v1 ladder for the cortiq rule");
                let (po, co, _) = q2tp_sections(self.rows, cols);
                params.copy_from_slice(&v1[po + r0 * 4..po + (r0 + rb) * 4]);
                for k in 0..rb {
                    let crow = &v1[co + (r0 + k) * stride..co + (r0 + k + 1) * stride];
                    for g in 0..gpr {
                        rungs[k * gpr + g] = q4tp_code(crow, g) as u8;
                    }
                }
            }
            Q2tpRule::Placed => {
                for k in 0..rb {
                    q2tp_place_row(
                        &self.w_init[(r0 + k) * cols..(r0 + k + 1) * cols],
                        self.hw,
                        &mut params[k * 4..k * 4 + 4],
                        &mut rungs[k * gpr..(k + 1) * gpr],
                    );
                }
            }
        }
        let tabs: Vec<[f32; 32]> = (0..rb).map(|k| det_q2tp_ladder(params, k)).collect();
        let lo_st: Vec<(f32, f32)> = params
            .chunks_exact(4)
            .map(|p| {
                (
                    f16_to_f32(u16::from_le_bytes([p[0], p[1]])),
                    f16_to_f32(u16::from_le_bytes([p[2], p[3]])),
                )
            })
            .collect();
        // Working rows, in sweep order.
        let mut w = vec![0f32; rb * cols];
        for k in 0..rb {
            let src = &self.w_init[(r0 + k) * cols..(r0 + k + 1) * cols];
            let dst = &mut w[k * cols..(k + 1) * cols];
            match self.perm {
                Some(p) => {
                    for (d, &pa) in dst.iter_mut().zip(p) {
                        *d = src[pa];
                    }
                }
                None => dst.copy_from_slice(src),
            }
        }
        chunks.fill(0);
        codes.fill(0);
        let u = self.u;
        let mut e = vec![0f32; Q2_ROWS * Q2_BLOCK];
        let mut sc = vec![0f32; rb];
        for b0 in (0..cols).step_by(Q2_BLOCK) {
            let b1 = (b0 + Q2_BLOCK).min(cols);
            for j in b0..b1 {
                let col = self.perm.map_or(j, |p| p[j]);
                let (g, within) = (col / GROUP_SIZE, col % GROUP_SIZE);
                if self.perm.is_none() && j % GROUP_SIZE == 0 {
                    let hg: &[f32; GROUP_SIZE] = self.hw[j..j + GROUP_SIZE].try_into().unwrap();
                    for k in 0..rb {
                        let tile: &[f32; GROUP_SIZE] =
                            w[k * cols + j..k * cols + j + GROUP_SIZE].try_into().unwrap();
                        let c = q2tp_pick_rung(self.rule, tile, hg, &tabs[k], lo_st[k]);
                        rungs[k * gpr + g] = c as u8;
                        sc[k] = tabs[k][c];
                    }
                }
                let d = u[j * cols + j];
                for k in 0..rb {
                    let s = if self.perm.is_some() {
                        tabs[k][rungs[k * gpr + g] as usize]
                    } else {
                        sc[k]
                    };
                    let wv = w[k * cols + j];
                    let q = if s > 0.0 { q2_level(wv / s + 1.5) } else { 0.0 };
                    chunks[(k * gpr + g) * Q2TP_CHUNK + within / 4] |= (q as u8) << (2 * (within % 4));
                    e[k * Q2_BLOCK + (j - b0)] = if d > 0.0 { (wv - (q - 1.5) * s) / d } else { 0.0 };
                }
                let urow = &u[j * cols + j + 1..j * cols + b1];
                for k in 0..rb {
                    let ek = e[k * Q2_BLOCK + (j - b0)];
                    if ek == 0.0 {
                        continue;
                    }
                    let wr = &mut w[k * cols + j + 1..k * cols + b1];
                    for (x, &uu) in wr.iter_mut().zip(urow) {
                        *x -= ek * uu;
                    }
                }
            }
            if b1 < cols {
                q2_trailing_update(&mut w, rb, cols, &e, b0, b1, u);
            }
        }
        for k in 0..rb {
            let crow = &mut codes[k * stride..(k + 1) * stride];
            for g in 0..gpr {
                q4tp_put_code(crow, g, rungs[k * gpr + g] as usize);
            }
        }
    }
}

/// A group's rung when the sweep reaches it (its weights already carry the
/// folded residuals of every earlier column). 0 for an all-zero group.
#[inline]
fn q2tp_pick_rung(
    rule: Q2tpRule,
    tile: &[f32; GROUP_SIZE],
    hg: &[f32; GROUP_SIZE],
    tab: &[f32; 32],
    (lo, st): (f32, f32),
) -> usize {
    use crate::convert::{det_log2, q2tp_best_rung, q2tp_group_err};
    use cortiq_core::quant::Q2TP_LMAX;
    let absmax = tile.iter().fold(0f32, |m, v| m.max(v.abs()));
    if absmax == 0.0 {
        return 0;
    }
    match rule {
        Q2tpRule::Placed => q2tp_best_rung(tile, hg, tab).1,
        Q2tpRule::Cortiq => {
            let nominal = if st > 0.0 {
                1 + ((det_log2(absmax / 1.5) - lo) / st)
                    .round_ties_even()
                    .clamp(0.0, Q2TP_LMAX as f32) as usize
            } else {
                1
            };
            let mut best = (f32::INFINITY, nominal);
            for cand in nominal.saturating_sub(2).max(1)..=(nominal + 1).min(Q2TP_LMAX + 1) {
                let s = tab[cand];
                if !(s > 0.0) {
                    continue;
                }
                let e = q2tp_group_err(tile, hg, s);
                if e < best.0 {
                    best = (e, cand);
                }
            }
            best.1
        }
    }
}

/// The lazy GPTQ update of one block: `w[k, b1..] −= Σ_t e[k, t]·u[b0+t, b1..]`
/// over the block's scaled residuals `e` (row stride [`Q2_BLOCK`]).
/// Register-tiled (8 rows × 8 columns, the `u` tile hot across row slabs).
/// Per element the sum runs over `t` in order and is subtracted once, with
/// separate multiplies and adds (Rust never contracts them), so every SIMD
/// width and machine produces the same bits.
fn q2_trailing_update(w: &mut [f32], rb: usize, cols: usize, e: &[f32], b0: usize, b1: usize, u: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just detected.
            return unsafe { q2_trailing_avx2(w, rb, cols, e, b0, b1, u) };
        }
    }
    q2_trailing_impl(w, rb, cols, e, b0, b1, u)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q2_trailing_avx2(w: &mut [f32], rb: usize, cols: usize, e: &[f32], b0: usize, b1: usize, u: &[f32]) {
    q2_trailing_impl(w, rb, cols, e, b0, b1, u)
}

#[inline(always)]
fn q2_trailing_impl(w: &mut [f32], rb: usize, cols: usize, e: &[f32], b0: usize, b1: usize, u: &[f32]) {
    debug_assert!(rb <= Q2_ROWS && e.len() >= Q2_ROWS * Q2_BLOCK);
    let nb = b1 - b0;
    let mut t = b1;
    while t + 8 <= cols {
        let mut s0 = 0;
        while s0 < rb {
            let mut acc = [[0f32; 8]; Q2_SLAB];
            for jj in 0..nb {
                let ur: &[f32; 8] = u[(b0 + jj) * cols + t..(b0 + jj) * cols + t + 8].try_into().unwrap();
                for (k, a) in acc.iter_mut().enumerate() {
                    let ek = e[(s0 + k) * Q2_BLOCK + jj];
                    for l in 0..8 {
                        a[l] += ek * ur[l];
                    }
                }
            }
            for (k, a) in acc.iter().enumerate().take(rb - s0) {
                let wr = &mut w[(s0 + k) * cols + t..(s0 + k) * cols + t + 8];
                for l in 0..8 {
                    wr[l] -= a[l];
                }
            }
            s0 += Q2_SLAB;
        }
        t += 8;
    }
    for tt in t..cols {
        for k in 0..rb {
            let mut a = 0f32;
            for jj in 0..nb {
                a += e[k * Q2_BLOCK + jj] * u[(b0 + jj) * cols + tt];
            }
            w[k * cols + tt] -= a;
        }
    }
}

/// Output debias of a q2tp tensor in the calibration metric: per row the
/// regression slope `b = (Ŵ H Wᵀ)_rr / (W H Wᵀ)_rr` is removed through the
/// row's f16 `lo` ([`crate::convert::q2tp_unshrink_row`]) — zero bits;
/// codes, rungs and steps stay. MSE-optimal 2-bit rounding (the `placed`
/// rule) shrinks every output by its error energy, the same way in every
/// layer; on MiniCPM5 this removed 5–20 % of ppl on such arms.
pub fn q2tp_debias_hessian(bytes: &mut [u8], w0: &[f32], rows: usize, cols: usize, h: &[f64], threads: usize) {
    debias_tiled_hessian(bytes, w0, rows, cols, h, threads, true);
}

/// [`q2tp_debias_hessian`] for a q4tp tensor (the same `lo` shift: every
/// q4tp rung of a row is `2^(lo + c·step)`). The shrinkage is ~16× smaller
/// at 4 bits than at 2, so this is opt-in (`--q4tp-debias`).
pub fn q4tp_debias_hessian(bytes: &mut [u8], w0: &[f32], rows: usize, cols: usize, h: &[f64], threads: usize) {
    debias_tiled_hessian(bytes, w0, rows, cols, h, threads, false);
}

fn debias_tiled_hessian(
    bytes: &mut [u8],
    w0: &[f32],
    rows: usize,
    cols: usize,
    h: &[f64],
    threads: usize,
    two_bit: bool,
) {
    use cortiq_core::quant::{Q2TP_CHUNK, Q4TP_NIB, q2tp_sections, q4tp_code, q4tp_sections};
    assert_eq!(h.len(), cols * cols);
    assert_eq!(w0.len(), rows * cols);
    let gpr = cols / GROUP_SIZE;
    let (po, co, stride) = if two_bit { q2tp_sections(rows, cols) } else { q4tp_sections(rows, cols) };
    let chunk_b = if two_bit { Q2TP_CHUNK } else { Q4TP_NIB };
    let (plane, rest) = bytes.split_at_mut(po);
    let (params, codes) = rest.split_at_mut(co - po);
    let (plane, codes) = (&*plane, &*codes);
    let per = rows.div_ceil(threads.max(1)).max(1);
    // Rows per pass over H: each H row is streamed once for DB weight rows
    // and the multiply-adds vectorize across them.
    const DB: usize = 8;
    std::thread::scope(|s| {
        for (ci, chunk) in params.chunks_mut(per * 4).enumerate() {
            s.spawn(move || {
                let (mut wt, mut ys) = (vec![0f64; cols * DB], vec![0f64; cols * DB]);
                let mut wh = vec![0f32; cols];
                let nrows = chunk.len() / 4;
                let mut k0 = 0;
                while k0 < nrows {
                    let rb = (nrows - k0).min(DB);
                    let r0 = ci * per + k0;
                    wt.fill(0.0);
                    for k in 0..rb {
                        for (j, &v) in w0[(r0 + k) * cols..(r0 + k + 1) * cols].iter().enumerate() {
                            wt[j * DB + k] = v as f64;
                        }
                    }
                    // y = H·w for the block. Per (i, row) the sum runs over j
                    // in order from −0.0 — exactly `Iterator::sum` on that
                    // row alone — so the block changes no bit.
                    for i in 0..cols {
                        let mut acc = [-0.0f64; DB];
                        for (j, &hv) in h[i * cols..(i + 1) * cols].iter().enumerate() {
                            let wj: &[f64; DB] = wt[j * DB..(j + 1) * DB].try_into().unwrap();
                            for k in 0..DB {
                                acc[k] += hv * wj[k];
                            }
                        }
                        ys[i * DB..(i + 1) * DB].copy_from_slice(&acc);
                    }
                    for k in 0..rb {
                        let r = r0 + k;
                        let p = &mut chunk[(k0 + k) * 4..(k0 + k + 1) * 4];
                        let wr = &w0[r * cols..(r + 1) * cols];
                        // Ŵ_r through the encoder's own (machine-independent) ladder.
                        let tab = if two_bit {
                            crate::convert::det_q2tp_ladder(p, 0)
                        } else {
                            crate::convert::det_q4tp_ladder(p, 0)
                        };
                        let crow = &codes[r * stride..(r + 1) * stride];
                        for g in 0..gpr {
                            let sg = tab[q4tp_code(crow, g)];
                            let ch = &plane[(r * gpr + g) * chunk_b..(r * gpr + g + 1) * chunk_b];
                            let dst = &mut wh[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
                            for (b, &byte) in ch.iter().enumerate() {
                                if two_bit {
                                    for j in 0..4 {
                                        dst[b * 4 + j] = (((byte >> (2 * j)) & 3) as f32 - 1.5) * sg;
                                    }
                                } else {
                                    dst[b * 2] = ((byte & 0x0F) as f32 - 8.0) * sg;
                                    dst[b * 2 + 1] = (((byte >> 4) & 0x0F) as f32 - 8.0) * sg;
                                }
                            }
                        }
                        let num: f64 = wh.iter().enumerate().map(|(i, &a)| a as f64 * ys[i * DB + k]).sum();
                        let den: f64 = wr.iter().enumerate().map(|(i, &a)| a as f64 * ys[i * DB + k]).sum();
                        crate::convert::q2tp_unshrink_row(p, num, den);
                    }
                    k0 += rb;
                }
            });
        }
    });
}

/// [`q2tp_debias_hessian`] with `H·w` for every row precomputed (`hw`,
/// `rows × cols`, e.g. one GEMM `W·H` for the whole matrix).
pub fn q2tp_debias_rows(bytes: &mut [u8], w0: &[f32], rows: usize, cols: usize, hw: &[f32], threads: usize) {
    debias_tiled_rows(bytes, w0, rows, cols, hw, threads, true);
}

/// [`q2tp_debias_hessian`] with `H·w` for all rows from one f32 GEMM `W·H`
/// split over `threads` — what `quantize-gptq` and `quantize-moe` run: the
/// f64 per-row products cost rows·cols² scalar multiply-adds (1.5 T for a
/// 27B down projection). Same parameters up to f32 rounding of `H·w`.
pub fn q2tp_debias_gemm(bytes: &mut [u8], w0: &[f32], rows: usize, cols: usize, h: &[f64], threads: usize) {
    debias_gemm(bytes, w0, rows, cols, h, threads, true);
}

/// [`q4tp_debias_hessian`] in the GEMM form of [`q2tp_debias_gemm`].
pub fn q4tp_debias_gemm(bytes: &mut [u8], w0: &[f32], rows: usize, cols: usize, h: &[f64], threads: usize) {
    debias_gemm(bytes, w0, rows, cols, h, threads, false);
}

fn debias_gemm(bytes: &mut [u8], w0: &[f32], rows: usize, cols: usize, h: &[f64], threads: usize, two_bit: bool) {
    assert_eq!(h.len(), cols * cols);
    let h32: Vec<f32> = h.iter().map(|&v| v as f32).collect();
    let hw = gemm_rows_square(w0, rows, cols, &h32, threads);
    debias_tiled_rows(bytes, w0, rows, cols, &hw, threads, two_bit);
}

fn debias_tiled_rows(bytes: &mut [u8], w0: &[f32], rows: usize, cols: usize, hw: &[f32], threads: usize, two_bit: bool) {
    use cortiq_core::quant::{Q2TP_CHUNK, Q4TP_NIB, q2tp_sections, q4tp_code, q4tp_sections};
    assert_eq!(w0.len(), rows * cols);
    assert_eq!(hw.len(), rows * cols);
    let gpr = cols / GROUP_SIZE;
    let (po, co, stride) = if two_bit { q2tp_sections(rows, cols) } else { q4tp_sections(rows, cols) };
    let chunk_b = if two_bit { Q2TP_CHUNK } else { Q4TP_NIB };
    let (plane, rest) = bytes.split_at_mut(po);
    let (params, codes) = rest.split_at_mut(co - po);
    let (plane, codes) = (&*plane, &*codes);
    let per = rows.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in params.chunks_mut(per * 4).enumerate() {
            s.spawn(move || {
                let mut wh = vec![0f32; cols];
                for (k, p) in chunk.chunks_exact_mut(4).enumerate() {
                    let r = ci * per + k;
                    let wr = &w0[r * cols..(r + 1) * cols];
                    let y = &hw[r * cols..(r + 1) * cols];
                    // Ŵ_r through the encoder's own (machine-independent) ladder.
                    let tab = if two_bit { crate::convert::det_q2tp_ladder(p, 0) } else { crate::convert::det_q4tp_ladder(p, 0) };
                    let crow = &codes[r * stride..(r + 1) * stride];
                    for g in 0..gpr {
                        let sg = tab[q4tp_code(crow, g)];
                        let ch = &plane[(r * gpr + g) * chunk_b..(r * gpr + g + 1) * chunk_b];
                        let dst = &mut wh[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
                        for (b, &byte) in ch.iter().enumerate() {
                            if two_bit {
                                for j in 0..4 {
                                    dst[b * 4 + j] = (((byte >> (2 * j)) & 3) as f32 - 1.5) * sg;
                                }
                            } else {
                                dst[b * 2] = ((byte & 0x0F) as f32 - 8.0) * sg;
                                dst[b * 2 + 1] = (((byte >> 4) & 0x0F) as f32 - 8.0) * sg;
                            }
                        }
                    }
                    let num: f64 = wh.iter().zip(y).map(|(&a, &b)| a as f64 * b as f64).sum();
                    let den: f64 = wr.iter().zip(y).map(|(&a, &b)| a as f64 * b as f64).sum();
                    crate::convert::q2tp_unshrink_row(p, num, den);
                }
            });
        }
    });
}

/// `A·B` for row-major `a` (`m × k`) and a square `b` (`k × k`), rows split
/// across `threads`.
pub(crate) fn gemm_rows_square(a: &[f32], m: usize, k: usize, b: &[f32], threads: usize) -> Vec<f32> {
    assert!(a.len() >= m * k && b.len() >= k * k);
    let mut out = vec![0f32; m * k];
    let per = m.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * k).enumerate() {
            let rows = chunk.len() / k;
            let src = &a[ci * per * k..(ci * per + rows) * k];
            s.spawn(move || cortiq_engine::qtensor::sgemm_public(rows, k, k, 1.0, src, k, b, k, false, chunk, k));
        }
    });
    out
}

const HESS_MAGIC: &[u8; 8] = b"CMFHESS1";

/// Save captured Hessians so GPTQ variants (damping, act order, which
/// tensors) can be re-run without re-calibrating. Identical Hessians (q/k/v
/// and gate/up read one input) are stored once under all their names; only
/// the upper triangle is written. f64 throughout — bit-exact round trip.
/// (The GPTQ runner writes its deduplicated groups directly.)
#[cfg(test)]
pub fn save_hessians(
    path: &str,
    hess: &std::collections::HashMap<String, cortiq_engine::gptq_capture::HessianAcc>,
) -> anyhow::Result<()> {
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
    let items: Vec<HessCacheItem> = uniq
        .iter()
        .map(|(ns, a)| HessCacheItem {
            names: ns.iter().map(|n| n.as_str()).collect(),
            cols: a.cols,
            count: a.count,
            sumsq: &a.sumsq,
            h: &a.h,
        })
        .collect();
    write_hessian_cache(path, &items)
}

/// One distinct Hessian to write into a cache, under all its names.
struct HessCacheItem<'a> {
    names: Vec<&'a str>,
    cols: usize,
    count: usize,
    sumsq: &'a [f64],
    /// Dense `[cols·cols]`, or empty (diagonal-only capture).
    h: &'a [f64],
}

/// Write a `CMFHESS1` cache: magic, entry count, then per entry its names,
/// `cols`, `count`, `len(h)`, `Σx²` and the upper triangle of `H` by rows
/// (f64 LE throughout — a bit-exact round trip). Written to `PATH.tmp` and
/// renamed, so a crash never leaves a truncated cache that looks valid.
fn write_hessian_cache(path: &str, items: &[HessCacheItem]) -> anyhow::Result<()> {
    use std::io::Write;
    let tmp = format!("{path}.tmp");
    let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp)?);
    f.write_all(HESS_MAGIC)?;
    f.write_all(&(items.len() as u64).to_le_bytes())?;
    for a in items {
        f.write_all(&(a.names.len() as u32).to_le_bytes())?;
        for n in &a.names {
            f.write_all(&(n.len() as u32).to_le_bytes())?;
            f.write_all(n.as_bytes())?;
        }
        f.write_all(&(a.cols as u64).to_le_bytes())?;
        f.write_all(&(a.count as u64).to_le_bytes())?;
        f.write_all(&(a.h.len() as u64).to_le_bytes())?;
        for v in a.sumsq {
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

/// One distinct Hessian of a cache file: its names and where its payload
/// (`Σx²`, then the upper triangle) starts. GPTQ reads entries one at a time
/// through this index instead of materialising — and, per name,
/// duplicating — the whole cache (8.5 GB on disk for a 2B model, 3× that
/// expanded).
#[derive(Clone, Debug)]
pub struct HessCacheEntry {
    pub names: Vec<String>,
    pub cols: usize,
    pub count: usize,
    pub hlen: usize,
    pub offset: u64,
}

/// Index a `CMFHESS1` cache without reading its matrices.
pub fn index_hessian_cache(path: &str) -> anyhow::Result<Vec<HessCacheEntry>> {
    use std::io::{Read, Seek};
    let mut f = std::io::BufReader::with_capacity(1 << 16, std::fs::File::open(path)?);
    let mut m = [0u8; 8];
    f.read_exact(&mut m)?;
    anyhow::ensure!(&m == HESS_MAGIC, "{path}: not a Hessian cache");
    let rd_u64 = |f: &mut std::io::BufReader<std::fs::File>| -> anyhow::Result<u64> {
        let mut b = [0u8; 8];
        f.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    };
    let n_uniq = rd_u64(&mut f)?;
    let mut out = Vec::new();
    for _ in 0..n_uniq {
        let mut u32b = [0u8; 4];
        f.read_exact(&mut u32b)?;
        let nn = u32::from_le_bytes(u32b) as usize;
        let mut names = Vec::with_capacity(nn);
        for _ in 0..nn {
            f.read_exact(&mut u32b)?;
            let mut s = vec![0u8; u32::from_le_bytes(u32b) as usize];
            f.read_exact(&mut s)?;
            names.push(String::from_utf8(s)?);
        }
        let cols = rd_u64(&mut f)? as usize;
        let count = rd_u64(&mut f)? as usize;
        let hlen = rd_u64(&mut f)? as usize;
        let offset = f.stream_position()?;
        let tri = if hlen == cols * cols { cols * (cols + 1) / 2 } else { 0 };
        f.seek_relative(((cols + tri) * 8) as i64)?;
        out.push(HessCacheEntry { names, cols, count, hlen, offset });
    }
    Ok(out)
}

/// Read one indexed entry back as a dense symmetric Hessian.
pub fn read_hessian_entry(
    path: &str,
    e: &HessCacheEntry,
) -> anyhow::Result<cortiq_engine::gptq_capture::HessianAcc> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::io::BufReader::with_capacity(1 << 22, std::fs::File::open(path)?);
    f.seek(SeekFrom::Start(e.offset))?;
    let cols = e.cols;
    let mut buf = vec![0u8; cols * 8];
    f.read_exact(&mut buf)?;
    let sumsq: Vec<f64> = buf
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut h = vec![0f64; e.hlen];
    if e.hlen == cols * cols {
        for i in 0..cols {
            let r = &mut buf[..(cols - i) * 8];
            f.read_exact(r)?;
            for (k, c) in r.chunks_exact(8).enumerate() {
                let v = f64::from_le_bytes(c.try_into().unwrap());
                h[i * cols + i + k] = v;
                h[(i + k) * cols + i] = v;
            }
        }
    }
    Ok(cortiq_engine::gptq_capture::HessianAcc { cols, h, sumsq, count: e.count })
}

/// Inverse of [`save_hessians`] (every name gets its own copy).
#[cfg(test)]
pub fn load_hessians(
    path: &str,
) -> anyhow::Result<std::collections::HashMap<String, cortiq_engine::gptq_capture::HessianAcc>> {
    let mut out = std::collections::HashMap::new();
    for e in index_hessian_cache(path)? {
        let acc = read_hessian_entry(path, &e)?;
        for n in &e.names {
            out.insert(
                n.clone(),
                cortiq_engine::gptq_capture::HessianAcc {
                    cols: acc.cols,
                    h: acc.h.clone(),
                    sumsq: acc.sumsq.clone(),
                    count: acc.count,
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

    /// The device fold + q4tp sweep against the host's f64 operator and
    /// sweep on the same calibration data: U agrees to f32 rounding, and the
    /// device codes are as good (output error within 1 %). Skips without a
    /// wgpu device.
    #[test]
    fn device_fold_and_sweep_match_host() {
        let (n, rows, t) = (160usize, 48usize, 700usize);
        let Some(hid) = cortiq_engine::gpu::gram_alloc(n) else {
            eprintln!("no device — skipped");
            return;
        };
        // Correlated inputs with a few loud channels, as activations are.
        let mut x = vec![0f32; t * n];
        for r in 0..t {
            let base = ((r * 37 % 101) as f32 - 50.0) / 29.0;
            for c in 0..n {
                let v = ((r * 7919 + c * 104729) % 1009) as f32 / 1009.0 - 0.5;
                let loud = if c % 41 == 3 { 30.0 } else { 1.0 };
                x[r * n + c] = (v + 0.3 * base) * loud;
            }
        }
        for c in [17usize, 90] {
            for r in 0..t {
                x[r * n + c] = 0.0; // dead channels
            }
        }
        assert!(cortiq_engine::gpu::gram_add(hid, &x, t));
        let mut h = vec![0f64; n * n];
        for r in 0..t {
            for i in 0..n {
                for j in 0..n {
                    h[i * n + j] += x[r * n + i] as f64 * x[r * n + j] as f64;
                }
            }
        }
        let w: Vec<f32> = (0..rows * n).map(|k| (((k * 2654435761usize) % 2003) as f32 / 2003.0 - 0.5) * 0.08).collect();
        let fid = cortiq_engine::gpu::gptq_fold_dev(hid, n, 0.01).expect("device fold");
        let (dead, hd_dev, _) = cortiq_engine::gpu::gptq_fold_info(fid).unwrap();
        let u_dev = cortiq_engine::gpu::gptq_fold_read(fid).unwrap();
        let hdiag: Vec<f32> = (0..n).map(|i| h[i * n + i] as f32).collect();
        let (u, dead_h) = gptq_fold_operator(h.clone(), n, 0.01);
        assert_eq!(dead, dead_h);
        for i in 0..n {
            assert!((hd_dev[i] - hdiag[i]).abs() <= 1e-4 * hdiag[i].abs().max(1.0), "diag {i}");
        }
        let scale = u.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = u.iter().zip(&u_dev).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
        assert!(worst < 2e-3, "U relative deviation {worst:e}");
        let host = gptq_quantize_q4tp(&w, rows, n, &u, &dead_h, &hdiag, 1);
        let dev = gptq_quantize_q4tp_dev(&w, rows, n, fid, &dead).expect("device sweep");
        let host2 = gptq_quantize_q2tp(&w, rows, n, &u, &dead_h, &hdiag, Q2tpRule::Placed, None, 1);
        let dev2 = gptq_quantize_q2tp_dev(&w, rows, n, fid, &dead, &hd_dev, 2).expect("device q2tp sweep");
        cortiq_engine::gpu::gptq_fold_free(fid);
        assert_eq!(host.len(), dev.len());
        assert_eq!(host2.len(), dev2.len());
        let out_err2 = |bytes: &[u8]| {
            let mut wq = vec![0f32; rows * n];
            cortiq_core::quant::dequant_q2tp(bytes, rows, n, &mut wq);
            let mut e = 0f64;
            for r in 0..rows {
                for i in 0..n {
                    for j in 0..n {
                        let di = (w[r * n + i] - wq[r * n + i]) as f64;
                        let dj = (w[r * n + j] - wq[r * n + j]) as f64;
                        e += di * h[i * n + j] * dj;
                    }
                }
            }
            e
        };
        let (eh2, ed2) = (out_err2(&host2), out_err2(&dev2));
        eprintln!("q2tp output error: host {eh2:e} device {ed2:e}");
        assert!(ed2 <= eh2 * 1.05, "device q2tp output error {ed2:e} vs host {eh2:e}");
        let out_err = |bytes: &[u8]| {
            let mut wq = vec![0f32; rows * n];
            cortiq_core::quant::dequant_q4tp(bytes, rows, n, &mut wq);
            let mut e = 0f64;
            for r in 0..rows {
                for i in 0..n {
                    for j in 0..n {
                        let di = (w[r * n + i] - wq[r * n + i]) as f64;
                        let dj = (w[r * n + j] - wq[r * n + j]) as f64;
                        e += di * h[i * n + j] * dj;
                    }
                }
            }
            e
        };
        let (eh, ed) = (out_err(&host), out_err(&dev));
        eprintln!("q4tp output error: host {eh:e} device {ed:e}; U deviation {worst:e}");
        assert!(ed <= eh * 1.05, "device output error {ed:e} vs host {eh:e}");
        let same = host[..rows * n / 2].iter().zip(&dev[..rows * n / 2]).filter(|(a, b)| a == b).count();
        assert!(same * 10 >= rows * n / 2 * 8, "nibble bytes equal: {same}/{}", rows * n / 2);
    }
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
        // The lazy index sees the two distinct payloads and reads each back
        // exactly, wherever it sits in the file.
        let index = index_hessian_cache(p.to_str().unwrap()).unwrap();
        assert_eq!(index.len(), 2);
        for e in &index {
            let acc = read_hessian_entry(p.to_str().unwrap(), e).unwrap();
            for n in &e.names {
                assert_eq!(acc.h, m[n].h, "{n}");
                assert_eq!(acc.sumsq, m[n].sumsq, "{n}");
            }
        }
        let _ = std::fs::remove_file(&p);
        assert_eq!(back.len(), 3);
        for (k, v) in &m {
            let b = &back[k];
            assert_eq!((b.cols, b.count), (v.cols, v.count));
            assert_eq!(b.sumsq, v.sumsq);
            assert_eq!(b.h, v.h);
        }
    }

    /// A layer problem with CORRELATED inputs (a few latent factors, one
    /// loud channel) and heavy-tailed weights: `(w, x [t·cols], H = XᵀX)`.
    fn q2_problem(rows: usize, cols: usize, t: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f64>) {
        q2_problem_with(rows, cols, t, seed, 9.0)
    }

    fn q2_problem_with(
        rows: usize,
        cols: usize,
        t: usize,
        seed: u64,
        outlier: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f64>) {
        let mut s = seed;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) as f32 - 0.5
        };
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let v = rnd() * (1.0 + 2.0 * rnd().abs());
                if i % 211 == 0 { v * outlier } else { v }
            })
            .collect();
        let lat: Vec<f32> = (0..t * 6).map(|_| rnd()).collect();
        let mix: Vec<f32> = (0..6 * cols).map(|_| rnd()).collect();
        let mut x = vec![0f32; t * cols];
        for k in 0..t {
            for c in 0..cols {
                let mut v = 0.25 * rnd();
                for f in 0..6 {
                    v += lat[k * 6 + f] * mix[f * cols + c];
                }
                x[k * cols + c] = if c == 7 { v * 6.0 } else { v };
            }
        }
        let mut h = vec![0f64; cols * cols];
        for k in 0..t {
            let xr = &x[k * cols..(k + 1) * cols];
            for i in 0..cols {
                for j in 0..cols {
                    h[i * cols + j] += xr[i] as f64 * xr[j] as f64;
                }
            }
        }
        (w, x, h)
    }

    /// ‖(W − Ŵ)·X‖² over the calibration samples, Ŵ decoded by the reader.
    fn q2_out_err(w: &[f32], bytes: &[u8], x: &[f32], rows: usize, cols: usize) -> f64 {
        let mut d = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_q2tp(bytes, rows, cols, &mut d);
        let t = x.len() / cols;
        let mut e = 0f64;
        for k in 0..t {
            for o in 0..rows {
                let mut acc = 0f64;
                for c in 0..cols {
                    acc += (w[o * cols + c] - d[o * cols + c]) as f64 * x[k * cols + c] as f64;
                }
                e += acc * acc;
            }
        }
        e
    }

    /// GPTQ-q2tp decodes through the ordinary q2tp reader, and with either
    /// rule its calibration OUTPUT error beats round-to-nearest of the same
    /// weights on the same ladder; the placed rule beats the cortiq rule.
    #[test]
    fn gptq_q2tp_beats_rtn_and_placed_beats_cortiq() {
        let (rows, cols, t) = (40usize, 128usize, 600usize);
        let (w, x, h) = q2_problem(rows, cols, t, 0xC0FFEE);
        let hdiag: Vec<f32> = (0..cols).map(|i| h[i * cols + i] as f32).collect();
        let (u, dead) = gptq_fold_operator(h, cols, 0.01);
        let rtn_v1 = crate::convert::encode_q2tp_v1(&w, rows, cols, true);
        let rtn_v2 = crate::convert::encode_q2tp_v2(&w, rows, cols);
        let gc = gptq_quantize_q2tp(&w, rows, cols, &u, &dead, &hdiag, Q2tpRule::Cortiq, None, 3);
        let gp = gptq_quantize_q2tp(&w, rows, cols, &u, &dead, &hdiag, Q2tpRule::Placed, None, 3);
        assert!(cortiq_core::quant::validate_payload(
            cortiq_core::types::TensorDtype::Q2TiledP,
            &[rows, cols],
            &gp
        )
        .is_ok());
        let (ec, ep) = (q2_out_err(&w, &gc, &x, rows, cols), q2_out_err(&w, &gp, &x, rows, cols));
        let (e1, e2) = (q2_out_err(&w, &rtn_v1, &x, rows, cols), q2_out_err(&w, &rtn_v2, &x, rows, cols));
        println!("q2tp output error: rtn v1 {e1:.3e}, rtn v2 {e2:.3e}, gptq cortiq {ec:.3e}, gptq placed {ep:.3e}");
        assert!(ec < 0.8 * e1, "cortiq rule {ec:.3e} vs its RTN {e1:.3e}");
        assert!(ep < 0.8 * e2, "placed rule {ep:.3e} vs v2 RTN {e2:.3e}");
        assert!(ep < ec, "placed {ep:.3e} must beat the cortiq rule {ec:.3e}");
        // The cortiq rule keeps the converter's ladder (only codes and rungs move).
        let (po, _, _) = cortiq_core::quant::q2tp_sections(rows, cols);
        assert_eq!(&gc[po..po + rows * 4], &rtn_v1[po..po + rows * 4]);
    }

    /// Byte-identical whatever the thread count, in natural and act order.
    #[test]
    fn gptq_q2tp_is_deterministic_across_threads() {
        let (rows, cols, t) = (70usize, 256usize, 300usize);
        let (w, _, h) = q2_problem(rows, cols, t, 42);
        let hdiag: Vec<f32> = (0..cols).map(|i| h[i * cols + i] as f32).collect();
        let perm = act_order_perm(&hdiag);
        let (up, _) = gptq_fold_operator(permute_sym(&h, cols, &perm), cols, 0.01);
        let (u, dead) = gptq_fold_operator(h, cols, 0.01);
        for rule in [Q2tpRule::Cortiq, Q2tpRule::Placed] {
            let a = gptq_quantize_q2tp(&w, rows, cols, &u, &dead, &hdiag, rule, None, 1);
            let b = gptq_quantize_q2tp(&w, rows, cols, &u, &dead, &hdiag, rule, None, 7);
            assert_eq!(a, b, "{rule:?}: natural order depends on threads");
            let a = gptq_quantize_q2tp(&w, rows, cols, &up, &dead, &hdiag, rule, Some(&perm), 1);
            let b = gptq_quantize_q2tp(&w, rows, cols, &up, &dead, &hdiag, rule, Some(&perm), 5);
            assert_eq!(a, b, "{rule:?}: act order depends on threads");
        }
    }

    /// Act order fixes every group's rung up front — the converter's (cortiq
    /// rule) or the placed ladder's own RTN choice — and still beats RTN.
    #[test]
    fn gptq_q2tp_act_order_keeps_static_rungs() {
        let (rows, cols, t) = (24usize, 128usize, 500usize);
        let (w, x, h) = q2_problem(rows, cols, t, 9);
        let hdiag: Vec<f32> = (0..cols).map(|i| h[i * cols + i] as f32).collect();
        let perm = act_order_perm(&hdiag);
        assert_eq!(perm[0], 7, "the loud channel goes first");
        let (up, dead) = gptq_fold_operator(permute_sym(&h, cols, &perm), cols, 0.01);
        let v1 = crate::convert::encode_q2tp_v1(&w, rows, cols, true);
        let ga = gptq_quantize_q2tp(&w, rows, cols, &up, &dead, &hdiag, Q2tpRule::Cortiq, Some(&perm), 2);
        let (po, _, _) = cortiq_core::quant::q2tp_sections(rows, cols);
        assert_eq!(&ga[po..], &v1[po..], "act order keeps the converter's ladder and rungs");
        let (ea, e1) = (q2_out_err(&w, &ga, &x, rows, cols), q2_out_err(&w, &v1, &x, rows, cols));
        assert!(ea < 0.8 * e1, "act order {ea:.3e} vs RTN {e1:.3e}");
        let gp = gptq_quantize_q2tp(&w, rows, cols, &up, &dead, &hdiag, Q2tpRule::Placed, Some(&perm), 2);
        let hw: Vec<f32> = hdiag.iter().map(|&v| v.max(0.0)).collect();
        let gpr = cols / GROUP_SIZE;
        let stride = cortiq_core::quant::q4tp_code_stride(gpr);
        for r in 0..rows {
            let (mut p, mut rungs) = ([0u8; 4], vec![0u8; gpr]);
            crate::convert::q2tp_place_row(&w[r * cols..(r + 1) * cols], &hw, &mut p, &mut rungs);
            assert_eq!(&gp[po + r * 4..po + r * 4 + 4], &p, "row {r} ladder");
            let crow = &gp[po + rows * 4 + r * stride..po + rows * 4 + (r + 1) * stride];
            for g in 0..gpr {
                assert_eq!(cortiq_core::quant::q4tp_code(crow, g), rungs[g] as usize, "row {r} group {g}");
            }
        }
    }

    /// With a diagonal Hessian there is nothing to fold: the placed rule is
    /// then the placed ladder + full rung search + round-to-nearest, i.e.
    /// the v2 encoder before its debias step (codes may differ only where
    /// `w/s` and `w·(1/s)` land on opposite sides of a tie).
    #[test]
    fn gptq_q2tp_with_identity_hessian_is_placed_rtn() {
        let (rows, cols) = (16usize, 192usize);
        let (w, _, _) = q2_problem(rows, cols, 1, 3);
        let mut h = vec![0f64; cols * cols];
        for i in 0..cols {
            h[i * cols + i] = 2.0;
        }
        let hdiag = vec![2.0f32; cols];
        let (u, dead) = gptq_fold_operator(h, cols, 0.0);
        let g = gptq_quantize_q2tp(&w, rows, cols, &u, &dead, &hdiag, Q2tpRule::Placed, None, 2);
        let gpr = cols / GROUP_SIZE;
        let ones = vec![1f32; cols];
        let (po, co, stride) = cortiq_core::quant::q2tp_sections(rows, cols);
        let mut diff = 0usize;
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let (mut p, mut rungs) = ([0u8; 4], vec![0u8; gpr]);
            crate::convert::q2tp_place_row(row, &ones, &mut p, &mut rungs);
            assert_eq!(&g[po + r * 4..po + r * 4 + 4], &p, "row {r}: a constant metric places the same ladder");
            let crow = &g[co + r * stride..co + (r + 1) * stride];
            let tab = crate::convert::det_q2tp_ladder(&p, 0);
            for gi in 0..gpr {
                let c = cortiq_core::quant::q4tp_code(crow, gi);
                assert_eq!(c, rungs[gi] as usize, "row {r} group {gi}");
                for k in 0..GROUP_SIZE {
                    let col = gi * GROUP_SIZE + k;
                    let want = crate::convert::q2_level(row[col] * (1.0 / tab[c]) + 1.5) as u8;
                    let byte = g[(r * gpr + gi) * cortiq_core::quant::Q2TP_CHUNK + k / 4];
                    diff += ((byte >> (2 * (k % 4))) & 3 != want) as usize;
                }
            }
        }
        assert!(diff * 1000 <= rows * cols, "{diff} codes differ from RTN");
    }

    /// The Hessian debias brings every row's output slope back to ~1 (the
    /// gain clipped to 0.8..1.25) and touches nothing but the row `lo`.
    #[test]
    fn q2tp_debias_hessian_restores_unit_slope() {
        let (rows, cols, t) = (20usize, 128usize, 400usize);
        let (w, _, h) = q2_problem_with(rows, cols, t, 77, 2.0);
        let hdiag: Vec<f32> = (0..cols).map(|i| h[i * cols + i] as f32).collect();
        let (u, dead) = gptq_fold_operator(h.clone(), cols, 0.01);
        let g = gptq_quantize_q2tp(&w, rows, cols, &u, &dead, &hdiag, Q2tpRule::Placed, None, 2);
        let mut d = g.clone();
        q2tp_debias_hessian(&mut d, &w, rows, cols, &h, 3);
        let (po, co, _) = cortiq_core::quant::q2tp_sections(rows, cols);
        assert_eq!(&g[..po], &d[..po], "2-bit plane untouched");
        assert_eq!(&g[co..], &d[co..], "rungs untouched");
        let slope = |bytes: &[u8], r: usize| -> f64 {
            let mut deq = vec![0f32; rows * cols];
            cortiq_core::quant::dequant_q2tp(bytes, rows, cols, &mut deq);
            let (mut num, mut den) = (0f64, 0f64);
            for i in 0..cols {
                let yi: f64 = (0..cols).map(|j| h[i * cols + j] * w[r * cols + j] as f64).sum();
                num += deq[r * cols + i] as f64 * yi;
                den += w[r * cols + i] as f64 * yi;
            }
            num / den
        };
        let mut unit = 0usize;
        for r in 0..rows {
            assert_eq!(g[po + r * 4 + 2..po + r * 4 + 4], d[po + r * 4 + 2..po + r * 4 + 4], "step untouched");
            let (b0, b1) = (slope(&g, r), slope(&d, r));
            let want = b0 / b0.clamp(0.8, 1.25);
            assert!((b1 - want).abs() < 0.01, "row {r}: slope {b0:.4} → {b1:.4} (want {want:.4})");
            unit += ((b1 - 1.0).abs() < 0.01) as usize;
        }
        assert!(unit > rows / 2, "most rows must come back to a unit slope ({unit}/{rows})");
    }

    /// The GEMM debias (`H·w` for every row from one f32 `W·H`, as
    /// `quantize-gptq` and `quantize-moe` run it) lands on the same
    /// parameters as the f64 per-row form.
    #[test]
    fn q2tp_debias_rows_matches_hessian_form() {
        let (rows, cols, t) = (24usize, 256usize, 600usize);
        let (w, _, h) = q2_problem_with(rows, cols, t, 91, 2.0);
        let hdiag: Vec<f32> = (0..cols).map(|i| h[i * cols + i] as f32).collect();
        let (u, dead) = gptq_fold_operator(h.clone(), cols, 0.01);
        let g = gptq_quantize_q2tp(&w, rows, cols, &u, &dead, &hdiag, Q2tpRule::Placed, None, 2);
        let (mut a, mut b) = (g.clone(), g.clone());
        q2tp_debias_hessian(&mut a, &w, rows, cols, &h, 3);
        q2tp_debias_gemm(&mut b, &w, rows, cols, &h, 3);
        let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert!(differ <= rows / 8, "{differ} bytes differ between the f64 and the GEMM debias");
    }

    /// The row-blocked debias writes exactly what the plain per-row version
    /// (y = H·w by `Iterator::sum`, row at a time) writes, on both layouts,
    /// with a partial last block.
    #[test]
    fn blocked_debias_matches_per_row_reference() {
        let (rows, cols, t) = (13usize, 96usize, 300usize);
        let (w, _, h) = q2_problem_with(rows, cols, t, 404, 3.0);
        let hdiag: Vec<f32> = (0..cols).map(|i| h[i * cols + i] as f32).collect();
        let (u, dead) = gptq_fold_operator(h.clone(), cols, 0.01);
        let reference = |bytes: &mut [u8], two_bit: bool| {
            let mut deq = vec![0f32; rows * cols];
            let po = if two_bit {
                let d = bytes.to_vec();
                // decode through the det ladder, like the kernel
                let (po, co, stride) = cortiq_core::quant::q2tp_sections(rows, cols);
                for r in 0..rows {
                    let tab = crate::convert::det_q2tp_ladder(&d[po..], r);
                    for c in 0..cols {
                        let g = c / GROUP_SIZE;
                        let s = tab[cortiq_core::quant::q4tp_code(&d[co + r * stride..], g)];
                        let byte = d[(r * (cols / GROUP_SIZE) + g) * 8 + (c % GROUP_SIZE) / 4];
                        deq[r * cols + c] = (((byte >> (2 * (c % 4))) & 3) as f32 - 1.5) * s;
                    }
                }
                po
            } else {
                let d = bytes.to_vec();
                let (po, co, stride) = cortiq_core::quant::q4tp_sections(rows, cols);
                for r in 0..rows {
                    let tab = crate::convert::det_q4tp_ladder(&d[po..], r);
                    for c in 0..cols {
                        let g = c / GROUP_SIZE;
                        let s = tab[cortiq_core::quant::q4tp_code(&d[co + r * stride..], g)];
                        let byte = d[(r * (cols / GROUP_SIZE) + g) * 16 + (c % GROUP_SIZE) / 2];
                        let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                        deq[r * cols + c] = (nib as f32 - 8.0) * s;
                    }
                }
                po
            };
            for r in 0..rows {
                let wr = &w[r * cols..(r + 1) * cols];
                let y: Vec<f64> = (0..cols)
                    .map(|i| h[i * cols..(i + 1) * cols].iter().zip(wr).map(|(&a, &b)| a * b as f64).sum())
                    .collect();
                let num: f64 = deq[r * cols..(r + 1) * cols].iter().zip(&y).map(|(&a, &b)| a as f64 * b).sum();
                let den: f64 = wr.iter().zip(&y).map(|(&a, &b)| a as f64 * b).sum();
                crate::convert::q2tp_unshrink_row(&mut bytes[po + r * 4..po + r * 4 + 4], num, den);
            }
        };
        let g2 = gptq_quantize_q2tp(&w, rows, cols, &u, &dead, &hdiag, Q2tpRule::Placed, None, 1);
        let (mut a, mut b) = (g2.clone(), g2);
        q2tp_debias_hessian(&mut a, &w, rows, cols, &h, 2);
        reference(&mut b, true);
        assert_eq!(a, b, "q2tp");
        let g4 = gptq_quantize_q4tp(&w, rows, cols, &u, &dead, &hdiag, 1);
        let (mut a, mut b) = (g4.clone(), g4);
        q4tp_debias_hessian(&mut a, &w, rows, cols, &h, 3);
        reference(&mut b, false);
        assert_eq!(a, b, "q4tp");
    }

    /// The q4tp debias: same contract on the 4-bit layout — nibbles, rungs
    /// and steps untouched, each row's metric slope back to ~1.
    #[test]
    fn q4tp_debias_hessian_restores_unit_slope() {
        let (rows, cols, t) = (16usize, 128usize, 400usize);
        let (w, _, h) = q2_problem_with(rows, cols, t, 31, 2.0);
        let hdiag: Vec<f32> = (0..cols).map(|i| h[i * cols + i] as f32).collect();
        let (u, dead) = gptq_fold_operator(h.clone(), cols, 0.01);
        let g = gptq_quantize_q4tp(&w, rows, cols, &u, &dead, &hdiag, 2);
        let mut d = g.clone();
        q4tp_debias_hessian(&mut d, &w, rows, cols, &h, 3);
        let (po, co, _) = cortiq_core::quant::q4tp_sections(rows, cols);
        assert_eq!(&g[..po], &d[..po], "nibbles untouched");
        assert_eq!(&g[co..], &d[co..], "rungs untouched");
        let slope = |bytes: &[u8], r: usize| -> f64 {
            let mut deq = vec![0f32; rows * cols];
            cortiq_core::quant::dequant_q4tp(bytes, rows, cols, &mut deq);
            let (mut num, mut den) = (0f64, 0f64);
            for i in 0..cols {
                let yi: f64 = (0..cols).map(|j| h[i * cols + j] * w[r * cols + j] as f64).sum();
                num += deq[r * cols + i] as f64 * yi;
                den += w[r * cols + i] as f64 * yi;
            }
            num / den
        };
        for r in 0..rows {
            assert_eq!(g[po + r * 4 + 2..po + r * 4 + 4], d[po + r * 4 + 2..po + r * 4 + 4], "step untouched");
            let (b0, b1) = (slope(&g, r), slope(&d, r));
            let want = b0 / b0.clamp(0.8, 1.25);
            assert!((b1 - want).abs() < 0.005, "row {r}: slope {b0:.5} → {b1:.5}");
        }
    }

    /// The fold operator as it was before blocking and threads: row
    /// Cholesky, one column of L⁻¹ at a time, one row of H⁻¹ at a time.
    fn fold_reference(mut h: Vec<f64>, n: usize, lambda: f64) -> Vec<f32> {
        for i in 0..n {
            if !(h[i * n + i] > 0.0) {
                for k in 0..n {
                    h[i * n + k] = 0.0;
                    h[k * n + i] = 0.0;
                }
                h[i * n + i] = 1.0;
            }
        }
        let base = ((0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64).max(1e-12);
        let mut lam = lambda;
        let mut a = loop {
            let mut a = h.clone();
            for i in 0..n {
                a[i * n + i] += lam * base;
            }
            if cholesky_lower_rows(&mut a, n) {
                break a;
            }
            lam = if lam == 0.0 { 1e-4 } else { lam * 10.0 };
        };
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
        for i in 0..n {
            for k in 0..=i {
                a[i * n + k] = dot64(&lt[i * n + i..i * n + n], &lt[k * n + i..k * n + n]);
            }
        }
        assert!(cholesky_lower_rows(&mut a, n));
        let mut u = vec![0f32; n * n];
        for j in 0..n {
            for c in 0..=j {
                u[c * n + j] = a[j * n + c] as f32;
            }
        }
        u
    }

    /// The blocked, threaded fold operator is the original serial one bit
    /// for bit, at any thread count (sizes with a partial last block, a dead
    /// channel, a singular H that needs its damping).
    #[test]
    fn parallel_fold_operator_is_bit_identical() {
        for (n, samples) in [(97usize, 40usize), (130, 400)] {
            let (_, _, mut h) = q2_problem(1, n, samples, 5 + n as u64);
            for k in 0..n {
                h[13 * n + k] = 0.0;
                h[k * n + 13] = 0.0;
            }
            let want = fold_reference(h.clone(), n, 0.01);
            for t in [1usize, 2, 3, 8] {
                let (ut, dt) = gptq_fold_operator_par(h.clone(), n, 0.01, t);
                assert!(dt[13] && dt.iter().filter(|&&d| d).count() == 1);
                assert!(want.iter().zip(&ut).all(|(a, b)| a.to_bits() == b.to_bits()), "n {n}, {t} threads");
                let mut a1 = h.clone();
                for i in 0..n {
                    a1[i * n + i] += 1.0;
                }
                let mut at = a1.clone();
                assert_eq!(cholesky_lower_rows(&mut a1, n), cholesky_lower_par(&mut at, n, t));
                assert!(a1.iter().zip(&at).all(|(a, b)| a.to_bits() == b.to_bits()), "cholesky n {n}, {t}");
            }
        }
    }

    /// The register-tiled trailing update is bit-identical to the plain
    /// loop it replaces (same per-element order: Σ over t, then subtract).
    #[test]
    fn q2_trailing_update_matches_reference_bits() {
        // 99 trailing columns: twelve register tiles and a 3-column tail.
        let (rb, cols, b0, b1) = (13usize, 139usize, 0usize, 40usize);
        let f = |k: usize| ((k * 2654435761) % 1000) as f32 / 997.0 - 0.5;
        let u: Vec<f32> = (0..cols * cols).map(f).collect();
        let mut e = vec![0f32; Q2_ROWS * Q2_BLOCK];
        for k in 0..rb {
            for jj in 0..b1 - b0 {
                e[k * Q2_BLOCK + jj] = f(k * 31 + jj * 7 + 5);
            }
        }
        let w0: Vec<f32> = (0..rb * cols).map(|k| f(k + 11)).collect();
        let mut got = w0.clone();
        q2_trailing_update(&mut got, rb, cols, &e, b0, b1, &u);
        let mut want = w0;
        for k in 0..rb {
            for tt in b1..cols {
                let mut a = 0f32;
                for jj in 0..b1 - b0 {
                    a += e[k * Q2_BLOCK + jj] * u[(b0 + jj) * cols + tt];
                }
                want[k * cols + tt] -= a;
            }
        }
        assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn calib_sources_parse_and_split_exactly() {
        let a = CalibSource::parse("wiki.train.raw=0.75").unwrap();
        assert_eq!((a.path.as_str(), a.share), ("wiki.train.raw", 0.75));
        let b = CalibSource::parse("c4.jsonl").unwrap();
        assert_eq!((b.path.as_str(), b.share), ("c4.jsonl", 1.0));
        let c = CalibSource::parse("dir=x/file.txt").unwrap();
        assert_eq!(c.path, "dir=x/file.txt", "a non-numeric suffix belongs to the path");
        assert!(CalibSource::parse("a.txt=0").is_err());
        assert!(CalibSource::parse("a.txt=-1").is_err());
        let s = |v: &[f64]| -> Vec<CalibSource> {
            v.iter().map(|&share| CalibSource { path: "x".into(), share }).collect()
        };
        assert_eq!(split_calib_budget(&s(&[0.75, 0.25]), 131072), vec![98304, 32768]);
        assert_eq!(split_calib_budget(&s(&[1.0]), 32768), vec![32768]);
        let parts = split_calib_budget(&s(&[1.0, 1.0, 1.0]), 1000);
        assert_eq!(parts.iter().sum::<usize>(), 1000);
        assert!(parts.iter().all(|&p| (333..=334).contains(&p)));
    }

    /// Calibration text is read in budgeted prefixes: plain text stops at a
    /// line boundary, `.jsonl` takes the `text` fields line by line, and the
    /// token reader grows the budget until it has enough.
    #[test]
    fn calib_text_reads_only_what_the_budget_needs() {
        let dir = std::env::temp_dir().join(format!("cortiq-calib-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let txt = dir.join("c.txt");
        let body: String = (0..400).map(|i| format!("line {i} with some words\n")).collect();
        std::fs::write(&txt, &body).unwrap();
        let (t, ex) = read_calib_text_budget(txt.to_str().unwrap(), 100).unwrap();
        assert!(!ex && t.len() <= 100 && t.ends_with('\n') && body.starts_with(&t));
        let (t, ex) = read_calib_text_budget(txt.to_str().unwrap(), 1 << 20).unwrap();
        assert!(ex && t == body);
        let jl = dir.join("c.jsonl");
        std::fs::write(&jl, "{\"text\": \"alpha beta\"}\n\"gamma\"\n{\"other\": 1}\n{\"text\": \"delta\"}\n").unwrap();
        let (t, ex) = read_calib_text_budget(jl.to_str().unwrap(), 1 << 20).unwrap();
        assert!(ex);
        assert_eq!(t, "alpha beta\ngamma\ndelta\n");
        // A whitespace "tokenizer": the reader must come back with exactly
        // the first `want` tokens of the whole file.
        let enc = |s: &str| -> Vec<u32> { s.split_whitespace().map(|w| w.len() as u32).collect() };
        let all = enc(&body);
        let got = calib_source_tokens(txt.to_str().unwrap(), 700, &enc).unwrap();
        assert_eq!(got, all[..700].to_vec());
        let got = calib_source_tokens(txt.to_str().unwrap(), 1_000_000, &enc).unwrap();
        assert_eq!(got, all, "an exhausted source returns all it has");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The whole `quantize-gptq --codec q2tp` runner on a two-layer f16 file
    /// with a Hessian cache (no calibration forward): codec per tensor from
    /// the q2tp profile, a q4tp override that stays GPTQ, RTN for tensors the
    /// cache lacks, bytes equal to the kernels called directly, identical
    /// output for 1 and 4 threads, and a whole-model cache serving a chunked
    /// run.
    #[test]
    fn tiled_runner_from_cache_follows_profile_overrides_and_threads() {
        use crate::convert::Quant;
        use cortiq_core::format::{CmfModel, TensorSpec};
        use cortiq_core::types::TensorDtype;
        let _env = crate::convert::tests::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("cortiq-gptq-run-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = |f: &str| dir.join(f).to_str().unwrap().to_string();
        let (hid, inter) = (64usize, 96usize);
        let shapes: Vec<(String, Vec<usize>)> = (0..2)
            .flat_map(|l| {
                [
                    (format!("model.layers.{l}.self_attn.q_proj.weight"), vec![hid, hid]),
                    (format!("model.layers.{l}.mlp.gate_proj.weight"), vec![inter, hid]),
                    (format!("model.layers.{l}.mlp.up_proj.weight"), vec![inter, hid]),
                    (format!("model.layers.{l}.mlp.down_proj.weight"), vec![hid, inter]),
                    (format!("model.layers.{l}.input_layernorm.weight"), vec![hid]),
                ]
            })
            .collect();
        let mut weights = std::collections::HashMap::new();
        let mut specs = Vec::new();
        for (k, (name, shape)) in shapes.iter().enumerate() {
            let n: usize = shape.iter().product();
            let mut s = 100 + k as u64;
            // Heavy-ish tails, rounded through f16 so the runner reads exactly these values.
            let w: Vec<f32> = (0..n)
                .map(|i| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let v = ((s >> 33) as f64 / (1u64 << 31) as f64) as f32 - 0.5;
                    f16_to_f32(f32_to_f16(if i % 97 == 0 { v * 6.0 } else { v }))
                })
                .collect();
            specs.push(TensorSpec {
                name: name.clone(),
                dtype: TensorDtype::F16,
                shape: shape.clone(),
                data: crate::convert::encode_f16(&w),
            });
            weights.insert(name.clone(), w);
        }
        let header: cortiq_core::CmfHeader = serde_json::from_value(serde_json::json!({
            "format": "cmf",
            "version": cortiq_core::CMF_VERSION,
            "arch": {
                "arch_name": "llama", "hidden_size": hid, "intermediate_size": inter,
                "num_layers": 2, "num_attention_heads": 1, "num_kv_heads": 1, "head_dim": hid,
                "vocab_size": 16, "layer_types": ["FullAttention", "FullAttention"],
                "rms_norm_eps": 1e-6, "norm_style": "qwen", "hidden_act": "silu",
                "max_position_embeddings": 64
            },
            "quant_type": "Q4_BLOCK"
        }))
        .unwrap();
        CmfModel::write(&p("in.cmf"), &header, &specs, None, None).unwrap();
        // Layer 0's Hessians only: q_proj, gate+up (one shared input), down.
        let (_, _, h64) = q2_problem(1, hid, 300, 7);
        let (_, _, h96) = q2_problem(1, inter, 300, 8);
        let diag = |h: &[f64], n: usize| -> Vec<f64> { (0..n).map(|i| h[i * n + i]).collect() };
        let (d64, d96) = (diag(&h64, hid), diag(&h96, inter));
        write_hessian_cache(
            &p("h.hess"),
            &[
                HessCacheItem {
                    names: vec!["model.layers.0.self_attn.q_proj.weight"],
                    cols: hid,
                    count: 300,
                    sumsq: &d64,
                    h: &h64,
                },
                HessCacheItem {
                    names: vec!["model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight"],
                    cols: hid,
                    count: 300,
                    sumsq: &d64,
                    h: &h64,
                },
                HessCacheItem {
                    names: vec!["model.layers.0.mlp.down_proj.weight"],
                    cols: inter,
                    count: 300,
                    sumsq: &d96,
                    h: &h96,
                },
            ],
        )
        .unwrap();
        crate::convert::set_tensor_quant_overrides(&["model.layers.0.mlp.up_proj.weight=q4tp".to_string()])
            .unwrap();
        let run = |out: &str, threads: usize, chunk: usize| {
            let (input, output, hess) = (p("in.cmf"), p(out), p("h.hess"));
            run_quantize_gptq_tiled(&GptqTiledOpts {
                input: &input,
                calib_model: None,
                calib: vec![CalibSource { path: p("absent.txt"), share: 1.0 }],
                output: &output,
                tokens: 2048,
                window: 2048,
                lambda: 0.01,
                threads,
                act_order: false,
                hessians: Some(&hess),
                profile: Quant::Q2TiledP,
                q2tp_rule: Q2tpRule::Placed,
                q2tp_rule_overrides: Vec::new(),
                q2tp_debias: false,
                q4tp_debias: false,
                chunk_layers: chunk,
            })
            .unwrap();
            std::fs::read(&output).unwrap()
        };
        let (a, b) = (run("a.cmf", 1, 0), run("b.cmf", 4, 0));
        let _ = run("c.cmf", 3, 1);
        crate::convert::set_tensor_quant_overrides(&[]).unwrap();
        assert_eq!(a, b, "output depends on the thread count");
        let m = CmfModel::open(&p("a.cmf")).unwrap();
        let mc = CmfModel::open(&p("c.cmf")).unwrap();
        fn by<'a>(m: &'a CmfModel, n: &str) -> &'a cortiq_core::format::TensorEntry {
            m.tensors.iter().find(|t| t.name == n).unwrap()
        }
        let hd: Vec<f32> = d64.iter().map(|&v| v as f32).collect();
        let (u, dead) = gptq_fold_operator(h64.clone(), hid, 0.01);
        let w = |n: &str| weights[n].clone();
        let checks: [(&str, TensorDtype, Vec<u8>); 5] = [
            (
                "model.layers.0.mlp.gate_proj.weight",
                TensorDtype::Q2TiledP,
                gptq_quantize_q2tp(&w("model.layers.0.mlp.gate_proj.weight"), inter, hid, &u, &dead, &hd, Q2tpRule::Placed, None, 2),
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                TensorDtype::Q4TiledP,
                gptq_quantize_q4tp(&w("model.layers.0.mlp.up_proj.weight"), inter, hid, &u, &dead, &hd, 2),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                TensorDtype::Q4TiledP,
                gptq_quantize_q4tp(&w("model.layers.0.self_attn.q_proj.weight"), hid, hid, &u, &dead, &hd, 2),
            ),
            (
                "model.layers.1.mlp.gate_proj.weight",
                TensorDtype::Q2TiledP,
                crate::convert::encode_q2tp(&w("model.layers.1.mlp.gate_proj.weight"), inter, hid),
            ),
            (
                "model.layers.1.self_attn.q_proj.weight",
                TensorDtype::Q4TiledP,
                crate::convert::encode_q4tp(&w("model.layers.1.self_attn.q_proj.weight"), hid, hid),
            ),
        ];
        for (name, dtype, want) in checks {
            for mm in [&m, &mc] {
                let e = by(mm, name);
                assert_eq!(e.dtype, dtype, "{name}");
                assert_eq!(mm.entry_bytes(e), &want[..], "{name}: not the kernel's bytes");
            }
        }
        assert_eq!(by(&m, "model.layers.0.input_layernorm.weight").dtype, TensorDtype::F16);
        let prov = m.header.provenance.as_ref().unwrap();
        assert_eq!(prov["weight_quant"], "q2tp");
        assert_eq!(prov["gptq"]["q2tp_rule"], "placed");
        assert_eq!(prov["gptq"]["linears"], 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn layer_chunks_follow_the_layer_index() {
        assert_eq!(layer_index("model.layers.12.mlp.up_proj.weight"), Some(12));
        assert_eq!(layer_index("model.language_model.layers.3.self_attn.q_proj.weight"), Some(3));
        assert_eq!(layer_index("lm_head.weight"), None);
        assert_eq!(
            parse_q2tp_rule_override("model.layers.7.mlp.*=cortiq").unwrap(),
            ("model.layers.7.mlp.*".to_string(), Q2tpRule::Cortiq)
        );
        assert!(parse_q2tp_rule_override("x=bogus").is_err());
    }

    /// The RAM guards: a capture that cannot fit is refused with a chunk
    /// size that does, and the number of folds in flight follows the RAM
    /// left after what the run holds.
    #[test]
    fn ram_guards_refuse_impossible_captures_and_cap_folds() {
        let gb = 1e9;
        // Qwen3.8-27B-like: 64 layers × ~3.6 GB of Hessians on a 128 GB box.
        let err = check_capture_fits(64.0 * 3.6 * gb, Some(128.0 * gb), 64, 0).unwrap_err().to_string();
        assert!(err.contains("--calib-chunk-layers 21 "), "{err}");
        assert!(check_capture_fits(16.0 * 3.6 * gb, Some(128.0 * gb), 16, 16).is_ok());
        assert!(check_capture_fits(1e15, None, 64, 0).is_ok(), "unknown RAM: no check");
        let peak = 28.0 * 17408f64.powi(2); // one 27B down_proj fold, ~8.5 GB
        assert_eq!(fold_concurrency(peak, 0.0, None, None), usize::MAX, "unknown RAM: no cap");
        assert_eq!(fold_concurrency(peak, 0.0, Some(20.0), Some(128.0 * gb)), 2, "CMF_GPTQ_MEM_GB wins");
        // 0.85 · 128 − 50 GB held = 58.8 GB → 6 folds.
        assert_eq!(fold_concurrency(peak, 50.0 * gb, None, Some(128.0 * gb)), 6);
        assert_eq!(fold_concurrency(peak, 200.0 * gb, None, Some(128.0 * gb)), 1, "never below one");
        assert_eq!(fold_concurrency(0.0, 0.0, None, Some(128.0 * gb)), usize::MAX);
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
