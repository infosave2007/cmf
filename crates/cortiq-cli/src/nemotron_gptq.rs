//! `cortiq quantize-moe` for Nemotron 3.5 Transcribe (`nemotron3_5_transcribe`):
//! GPTQ q4tp for the routed experts, whose up/down MLP is NOT gated
//! (`down(relu(up·x)²)`).
//!
//! * Capture (`CMF_GPU=0`, the calibration CMF on the host path): every
//!   calibration recording is transcribed; each MoE block records its input
//!   rows (the block's RMSNorm output: audio tokens, the prompt and the
//!   generated transcript) and every row's routed experts. Saved as f16 rows
//!   + u16 picks per block under `--scratch`.
//! * Encode (a wgpu device): per expert, the gram of its routed rows folds
//!   into the GPTQ operator of `up`; the down input `relu(X·W_upᵀ)²` from the
//!   full-precision `up` gives the operator of `down`; both sweep into the
//!   converter's q4tp layout (`gptq_quantize_q4tp_dev`). Experts routed fewer
//!   than `--min-rows` rows keep the converter's round-to-nearest encoding.
//! * Output: every other tensor is copied from the calibration CMF (the
//!   q8_2f file: its skeleton IS the q4tp profile's), the experts replaced.
//!
//! The full-precision weights come from the release's `model.safetensors`
//! (`--hf <dir>`): the packed `(experts, in, out)` bf16 banks.

use crate::convert::{Quant, open_safetensors, quantize_2d, to_f32};
use crate::gptq::gptq_quantize_q4tp_dev;
use crate::moegptq::MoeOpts;
use cortiq_core::format::{CmfModel, CmfStreamWriter};
use cortiq_core::quant::{f16_to_f32, f32_to_f16};
use cortiq_core::types::TensorDtype;
use cortiq_engine::gpu;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

fn wavs(specs: &[String]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for s in specs {
        let p = Path::new(s.split('=').next().unwrap_or(s));
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav")))
                .collect();
            v.sort();
            out.extend(v);
        } else {
            out.push(p.to_path_buf());
        }
    }
    out
}

fn capture(cm: &Arc<CmfModel>, o: &MoeOpts, dir: &Path) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let tr = cortiq_engine::nemotron_asr::Transcriber::load(cm).map_err(anyhow::Error::msg)?;
    let pool = cortiq_engine::pool::Pool::from_env();
    tr.llm.start_capture();
    let mut tokens = 0usize;
    let files = wavs(o.calib);
    anyhow::ensure!(!files.is_empty(), "--calib names no WAV files");
    for (i, f) in files.iter().enumerate() {
        let samples = cortiq_engine::nemotron_asr::read_audio(f).map_err(anyhow::Error::msg)?;
        let out = tr
            .transcribe(&samples, None, 1024, pool.as_deref(), |_| {})
            .map_err(anyhow::Error::msg)?;
        tokens += out.prompt_tokens + out.ids.len();
        eprintln!(
            "  capture {}/{} {}: {} tokens (total {tokens}), {:.0}s",
            i + 1,
            files.len(),
            f.display(),
            out.prompt_tokens + out.ids.len(),
            t0.elapsed().as_secs_f64()
        );
        if tokens >= o.tokens {
            break;
        }
    }
    let cap = tr
        .llm
        .take_capture()
        .ok_or_else(|| anyhow::anyhow!("capture lost"))?;
    std::fs::create_dir_all(dir)?;
    let mut blocks = Vec::new();
    for (bi, (x, p)) in cap.x.iter().zip(&cap.picks).enumerate() {
        if x.is_empty() {
            continue;
        }
        let xb: Vec<u8> = x
            .iter()
            .flat_map(|v| f32_to_f16(*v).to_le_bytes())
            .collect();
        std::fs::write(dir.join(format!("block{bi}.x.f16")), xb)?;
        let pb: Vec<u8> = p.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("block{bi}.picks.u16")), pb)?;
        blocks.push(bi);
    }
    let meta = serde_json::json!({ "blocks": blocks, "tokens": tokens, "files": files.len() });
    std::fs::write(dir.join("nemotron-capture.json"), meta.to_string())?;
    eprintln!(
        "capture: {tokens} tokens, {} MoE blocks in {:.0}s → {}",
        blocks.len(),
        t0.elapsed().as_secs_f64(),
        dir.display()
    );
    Ok(())
}

/// One expert's GPTQ q4tp `[up, down]` payloads.
fn encode_expert(
    x: &[f32],
    n: usize,
    hidden: usize,
    inter: usize,
    up: &[f32],
    down: &[f32],
    lambda: f64,
) -> anyhow::Result<[Vec<u8>; 2]> {
    let sweep =
        |w: &[f32], rows: usize, cols: usize, xs: &[f32], m: usize| -> anyhow::Result<Vec<u8>> {
            let gid = gpu::gram_alloc(cols).ok_or_else(|| anyhow::anyhow!("gram alloc"))?;
            anyhow::ensure!(gpu::gram_add(gid, xs, m), "gram");
            let fid =
                gpu::gptq_fold_dev(gid, cols, lambda).ok_or_else(|| anyhow::anyhow!("fold"))?;
            let (dead, _, _) = gpu::gptq_fold_info(fid).unwrap_or_default();
            let b = gptq_quantize_q4tp_dev(w, rows, cols, fid, &dead)
                .ok_or_else(|| anyhow::anyhow!("q4tp sweep"));
            gpu::gptq_fold_free(fid);
            b
        };
    let ub = sweep(up, inter, hidden, x, n)?;
    // the down input: relu(X·W_upᵀ)² from the full-precision up projection
    let mut a = vec![0f32; n * inter];
    cortiq_engine::fcd_ops::gemm_nt(x, up, &mut a, n, hidden, inter, None);
    for v in a.iter_mut() {
        let r = v.max(0.0);
        *v = r * r;
    }
    let db = sweep(down, hidden, inter, &a, n)?;
    Ok([ub, db])
}

pub fn run(o: &MoeOpts) -> anyhow::Result<()> {
    let cm = Arc::new(CmfModel::open(o.calib_model)?);
    anyhow::ensure!(
        cm.header.arch.arch_name == cortiq_engine::nemotron_asr::ARCH_NAME,
        "not a Nemotron 3.5 Transcribe CMF"
    );
    anyhow::ensure!(
        o.codec == "q4tp",
        "nemotron_transcribe: only --codec q4tp is implemented"
    );
    let dir = Path::new(o.scratch);
    if !dir.join("nemotron-capture.json").exists() {
        capture(&cm, o, dir)?;
    }
    if !gpu::wgpu_active() {
        eprintln!(
            "capture is in {}; run the same command with a wgpu device for the GPTQ encode",
            dir.display()
        );
        return Ok(());
    }
    let output = o.output;
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("nemotron-capture.json"))?)?;
    let blocks: Vec<usize> = meta["blocks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_u64())
        .map(|v| v as usize)
        .collect();
    let llm = &cm
        .header
        .provenance
        .as_ref()
        .and_then(|p| p.get("nemotron_transcribe"))
        .ok_or_else(|| anyhow::anyhow!("no provenance"))?["llm_config"];
    let g = |k: &str| {
        llm.get(k)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| anyhow::anyhow!("llm config: {k}"))
    };
    let (hidden, inter, ne, top_k) = (
        g("hidden_size")?,
        g("moe_intermediate_size")?,
        g("n_routed_experts")?,
        g("num_experts_per_tok")?,
    );
    let src_dir = Path::new(o.hf);
    let st = open_safetensors(&src_dir.join("model.safetensors"))?;
    let find = |name: &str| {
        st.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow::anyhow!("source has no {name}"))
    };
    let t0 = Instant::now();
    let mut repl: HashMap<String, Vec<u8>> = HashMap::new();
    let (mut n_gptq, mut n_rtn) = (0usize, 0usize);
    for &bi in &blocks {
        let tb = Instant::now();
        let xb = std::fs::read(dir.join(format!("block{bi}.x.f16")))?;
        let x: Vec<f32> = xb
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let pb = std::fs::read(dir.join(format!("block{bi}.picks.u16")))?;
        let picks: Vec<u16> = pb
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let n = x.len() / hidden;
        anyhow::ensure!(
            picks.len() == n * top_k,
            "block {bi}: {} picks for {n} rows",
            picks.len()
        );
        let mut lists: Vec<Vec<usize>> = vec![Vec::new(); ne];
        for (j, &e) in picks.iter().enumerate() {
            lists[e as usize].push(j / top_k);
        }
        let ups = find(&format!(
            "llm.model.layers.{bi}.mixer.experts.gate_and_up_projs"
        ))?;
        let downs = find(&format!("llm.model.layers.{bi}.mixer.experts.down_projs"))?;
        let (upr, dnr) = (st.bytes(ups), st.bytes(downs));
        let per = hidden * inter * 2;
        for e in 0..ne {
            // packed (in, out) → [out, in]
            let tr = |raw: &[u8], din: usize, dout: usize| -> anyhow::Result<Vec<f32>> {
                let v = to_f32("BF16", &raw[e * per..(e + 1) * per])?;
                let mut t = vec![0f32; din * dout];
                for i in 0..din {
                    for o_ in 0..dout {
                        t[o_ * din + i] = v[i * dout + o_];
                    }
                }
                Ok(t)
            };
            let up = tr(upr, hidden, inter)?;
            let down = tr(dnr, inter, hidden)?;
            let rows = &lists[e];
            let [ub, db] = if rows.len() >= o.min_rows {
                let mut xs = vec![0f32; rows.len() * hidden];
                for (r, &t) in rows.iter().enumerate() {
                    xs[r * hidden..(r + 1) * hidden]
                        .copy_from_slice(&x[t * hidden..(t + 1) * hidden]);
                }
                n_gptq += 1;
                encode_expert(&xs, rows.len(), hidden, inter, &up, &down, o.lambda)?
            } else {
                n_rtn += 1;
                [
                    quantize_2d(Quant::Q4TiledP, &up, inter, hidden).1,
                    quantize_2d(Quant::Q4TiledP, &down, hidden, inter).1,
                ]
            };
            repl.insert(
                format!("model.layers.{bi}.mlp.experts.{e}.up_proj.weight"),
                ub,
            );
            repl.insert(
                format!("model.layers.{bi}.mlp.experts.{e}.down_proj.weight"),
                db,
            );
        }
        eprintln!(
            "  block {bi}: {n} rows, {:.1}s (total {:.0}s, {n_gptq} GPTQ / {n_rtn} RTN)",
            tb.elapsed().as_secs_f64(),
            t0.elapsed().as_secs_f64()
        );
    }
    // stream: the calibration file's tensors, experts replaced (q4tp)
    let gap = CmfStreamWriter::head_reserve_for(cm.tensors.len(), 64);
    let tmp = format!("{output}.part");
    let mut w =
        CmfStreamWriter::new(&tmp, gap).map_err(|e| anyhow::anyhow!("create {tmp}: {e}"))?;
    let mut replaced = 0usize;
    for e in &cm.tensors {
        if let Some(b) = repl.get(&e.name) {
            w.push(&e.name, TensorDtype::Q4TiledP, &e.shape, b)
                .map_err(|er| anyhow::anyhow!("{}: {er}", e.name))?;
            replaced += 1;
        } else if e.name.contains(".mlp.experts.") {
            // experts of blocks without capture (MTP): q4tp from the source codec
            let v = cortiq_engine::nemotron_h::tensor_values(&cm, &e.name)
                .map_err(anyhow::Error::msg)?;
            let (dt, b) = quantize_2d(Quant::Q4TiledP, &v, e.shape[0], e.shape[1]);
            w.push(&e.name, dt, &e.shape, &b)
                .map_err(|er| anyhow::anyhow!("{}: {er}", e.name))?;
        } else {
            w.push(&e.name, e.dtype, &e.shape, cm.entry_bytes(e))
                .map_err(|er| anyhow::anyhow!("{}: {er}", e.name))?;
        }
    }
    anyhow::ensure!(
        replaced == repl.len(),
        "{} of {} GPTQ experts found in the calibration file",
        replaced,
        repl.len()
    );
    let mut header = cm.header.clone();
    header.quant_type = cortiq_core::types::QuantType::Q4Block;
    if let Some(p) = header.provenance.as_mut().and_then(|p| p.as_object_mut()) {
        p.insert("quant".into(), serde_json::json!("q4tp"));
        p.insert(
            "experts_gptq".into(),
            serde_json::json!({ "calib_tokens": meta["tokens"], "calib_files": meta["files"], "lambda": o.lambda, "min_rows": o.min_rows, "gptq": n_gptq, "rtn": n_rtn }),
        );
    }
    w.finish(&header, None, cm.vocab.as_deref())
        .map_err(|e| anyhow::anyhow!("finish {tmp}: {e}"))?;
    std::fs::rename(&tmp, output)?;
    eprintln!(
        "{output}: {:.2} GB, {n_gptq} experts GPTQ, {n_rtn} RTN, {:.0}s",
        std::fs::metadata(output)?.len() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
