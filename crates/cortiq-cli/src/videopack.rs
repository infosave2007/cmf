//! Pack MiniMax-H3 — the joint audio-video DiT, its Qwen3-VL prompt
//! encoder and both VAE decoders — into ONE quantized .cmf, with the
//! 4-step Turbo LoRA merged in so the file IS the turbo model.
//!
//! Four sources, four passes, and `--in` carries a previous pass's file
//! through byte for byte: the stand that does this has less free disk
//! than the sum of the inputs, so each source is packed and deleted
//! before the next one lands.
//!
//! ## The adaLN collapse
//!
//! Forty per cent of the released DiT is one matrix per block:
//! `adaln_proj.linear` is [96768, 2688], 520 MB at bf16, 13 B parameters
//! over fifty blocks. It consumes ONE vector — the timestep embedding —
//! so its output over the whole schedule is a 1-D curve in R^96768, and
//! Comfy-Org's `pruned` checkpoints already ship it that way: an
//! `adaln_t_table` of [1025, 8] shared by every block, and per-block
//! weights of [96768, 8]. Rank eight, 1.55 MB, same arithmetic.
//!
//! The Turbo LoRA is written against the FULL matrix (`lora_A` is
//! [16, 2688]), which is why the ComfyUI node re-injects the time
//! conditioning at run time when the base is pruned. We do it once here
//! instead: the merged map is
//!
//! ```text
//! adaln(t) = W_p · u(t) + b + B · (A · silu(e(t)))
//!          = [W_p | B] · [u(t) ; A·silu(e(t))]
//! ```
//!
//! — a rank-24 curve, evaluated by table lookup. `--time-embedder` is
//! the four small tensors of the full checkpoint's `time_embedder` (64
//! MB out of 66 GB, a ranged read; `tools/mmh3_fetch.py`), needed only
//! to tabulate `A·silu(e(t))` on the pruned base's own 1025-point grid.
//! The result is 4.6 MB a block instead of 520.

use anyhow::{Context, anyhow};
use cortiq_core::CmfModel;
use cortiq_core::format::{CmfHeader, TensorSpec, TensorSpecRef};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::convert::{encode_f16, encode_q2tp, encode_q4tp, encode_q8_2f, encode_q8_row};
use cortiq_engine::pool::Pool;

/// Timestep-embedding grid of the pruned checkpoints. The runtime reads
/// the table with the same `t·(G−1)` lerp the reference uses, so this
/// must match `adaln_t_table.shape[0]`; it is asserted, not assumed.
const CURVE_GRID: usize = 1025;
/// Sinusoidal width of the time embedder (`timestep_input_dim`).
const FREQ_DIM: usize = 256;

// ── a memory-mapped safetensors file ────────────────────────────────

struct StEntry {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

struct StFile {
    map: Mmap,
    dir: HashMap<String, StEntry>,
    order: Vec<String>,
}

impl StFile {
    fn open(path: &Path) -> anyhow::Result<Self> {
        let f = std::fs::File::open(path).with_context(|| path.display().to_string())?;
        // SAFETY: the converter is the only writer of its inputs; a
        // truncation under us would be a torn read, same as any mmap.
        let map = unsafe { Mmap::map(&f) }.with_context(|| path.display().to_string())?;
        if map.len() < 8 {
            return Err(anyhow!("{}: truncated safetensors", path.display()));
        }
        let hlen = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(
            map.get(8..8 + hlen)
                .ok_or_else(|| anyhow!("{}: header past EOF", path.display()))?,
        )?;
        let base = 8 + hlen;
        let mut dir = HashMap::new();
        let mut order = Vec::new();
        for (name, meta) in header.as_object().ok_or_else(|| anyhow!("header"))? {
            if name == "__metadata__" {
                continue;
            }
            let offs = meta["data_offsets"]
                .as_array()
                .ok_or_else(|| anyhow!("{name}: data_offsets"))?;
            dir.insert(
                name.clone(),
                StEntry {
                    dtype: meta["dtype"].as_str().unwrap_or("F32").to_string(),
                    shape: meta["shape"]
                        .as_array()
                        .ok_or_else(|| anyhow!("{name}: shape"))?
                        .iter()
                        .map(|v| v.as_u64().unwrap_or(0) as usize)
                        .collect(),
                    start: offs[0].as_u64().unwrap_or(0) as usize + base,
                    end: offs[1].as_u64().unwrap_or(0) as usize + base,
                },
            );
            order.push(name.clone());
        }
        order.sort();
        Ok(Self { map, dir, order })
    }

    fn shape(&self, name: &str) -> Option<&[usize]> {
        self.dir.get(name).map(|e| e.shape.as_slice())
    }

    /// The tensor's bytes as stored. `tokenizer_json` is a byte blob
    /// wearing a tensor's clothes; decoding it as floats destroys it.
    fn raw(&self, name: &str) -> Option<&[u8]> {
        let e = self.dir.get(name)?;
        self.map.get(e.start..e.end)
    }

    fn has(&self, name: &str) -> bool {
        self.dir.contains_key(name)
    }

    /// Decode one tensor to f32. bf16/f16/f32 only — every MiniMax-H3
    /// release we read is one of those.
    fn get(&self, name: &str) -> anyhow::Result<Vec<f32>> {
        let e = self
            .dir
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))?;
        let raw = self
            .map
            .get(e.start..e.end)
            .ok_or_else(|| anyhow!("{name}: span past EOF"))?;
        Ok(match e.dtype.as_str() {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| cortiq_core::quant::f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16))
                .collect(),
            other => return Err(anyhow!("{name}: unsupported dtype {other}")),
        })
    }
}

// ── codecs ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Level {
    /// 4.16 bits a weight with the predicted per-row scale ladder.
    Q4tp,
    /// 2.16 bits, the same ladder one rung narrower. Reserved for the
    /// gate/up planes: DeepSeek-V4 measured 2 bits there at 1.3x the
    /// perplexity for 0.71x the file, and the down projection and the
    /// attention are where that trade stops paying.
    Q2tp,
    Q8,
    /// The two-field q8: `w = q·row[o]·col[i]`, eight bits with a second
    /// field along the INPUT axis. Same size as `q8_row` and strictly
    /// better on weights whose columns differ in scale — which is what an
    /// activation-outlier channel looks like from the weight side. The
    /// step up from `q4tp` for a machine with the memory to hold it.
    Q82f,
    F16,
    F32,
}

/// Which planes drop to two bits under `--quant q2tp`. Everything else
/// stays at four.
fn is_wide_plane(name: &str) -> bool {
    name.ends_with("mlp.fc1.weight")
        || name.ends_with("mlp.gate_proj.weight")
        || name.ends_with("mlp.up_proj.weight")
        || name.ends_with("ff.w1.weight")
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// One tensor at `level`. q4tp needs `cols % 32 == 0` and 2-D; anything
/// that misses either constraint falls to q8, then f16.
fn spec(name: String, vals: &[f32], shape: Vec<usize>, level: Level) -> TensorSpec {
    let two_d = shape.len() == 2;
    // q2tp is a policy, not a blanket: only the wide gate/up planes take
    // it, and only where the codec's column constraint is met.
    let level = match level {
        Level::Q2tp if two_d && shape[1] % 32 == 0 && is_wide_plane(&name) => Level::Q2tp,
        Level::Q2tp => Level::Q4tp,
        other => other,
    };
    let (dtype, data) = match level {
        Level::Q2tp => (TensorDtype::Q2TiledP, encode_q2tp(vals, shape[0], shape[1])),
        Level::Q4tp if two_d && shape[1] % 32 == 0 => {
            (TensorDtype::Q4TiledP, encode_q4tp(vals, shape[0], shape[1]))
        }
        // Two fields need both axes, and the column field is only
        // meaningful when there is more than one row to share it.
        Level::Q82f if two_d && shape[0] > 1 => {
            (TensorDtype::Q8_2f, encode_q8_2f(vals, shape[0], shape[1]))
        }
        Level::Q4tp | Level::Q8 | Level::Q82f if two_d => {
            (TensorDtype::Q8Row, encode_q8_row(vals, shape[0], shape[1]))
        }
        Level::F32 => (TensorDtype::F32, f32_bytes(vals)),
        _ => (TensorDtype::F16, encode_f16(vals)),
    };
    TensorSpec {
        name,
        dtype,
        shape,
        data,
    }
}

// ── LoRA ────────────────────────────────────────────────────────────

/// `W += scale · B·A`, in place, rows split across the pool. B is
/// [rows, r] and A is [r, cols] — the file's own orientation, so no
/// transpose is needed on either side.
fn lora_merge(w: &mut [f32], b: &[f32], a: &[f32], rows: usize, cols: usize, scale: f32) {
    let r = a.len() / cols;
    assert_eq!(b.len(), rows * r, "lora_B shape");
    let pool = Pool::from_env();
    let ptr = SendPtr(w.as_mut_ptr());
    let work = |lo: usize, hi: usize| {
        for row in lo..hi {
            // SAFETY: workers own disjoint row ranges of `w`.
            let dst = unsafe { ptr.row(row * cols, cols) };
            for k in 0..r {
                let c = scale * b[row * r + k];
                if c == 0.0 {
                    continue;
                }
                let arow = &a[k * cols..(k + 1) * cols];
                for (d, &av) in dst.iter_mut().zip(arow) {
                    *d += c * av;
                }
            }
        }
    };
    match pool.as_deref() {
        Some(p) => p.run_rows(rows, &work),
        None => work(0, rows),
    }
}

/// Row handout for pool workers. Rust-2021 closures capture the raw
/// pointer FIELD rather than the wrapper, which loses the `Sync` impl —
/// hence the accessor method (same shape as `dit::SendRows`).
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// SAFETY: caller guarantees disjoint `[off, off+len)` per worker.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

// ── the time-embedding curve ────────────────────────────────────────

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// `silu(time_embedder(t))` on the pruned grid: [CURVE_GRID, 2688].
/// This is the vector `adaln_proj.linear` consumes on a full
/// checkpoint, and therefore the vector the LoRA's `lora_A` consumes.
///
/// Two ways in. The Turbo node bundles this grid outright
/// (`h3_silu_temb_grid.safetensors`, 5.5 MB) — take it when it is
/// there, both because it is 5.5 MB against a 64 MB ranged read and
/// because it is the LoRA author's own tabulation. Otherwise rebuild it
/// from the full checkpoint's four `time_embedder.*` tensors.
fn time_curve(te: &StFile) -> anyhow::Result<(Vec<f32>, usize)> {
    if let Some(shape) = te.shape("silu_t_emb_grid") {
        let (grid, t_dim) = (shape[0], shape[1]);
        if grid != CURVE_GRID {
            return Err(anyhow!("silu_t_emb_grid grid {grid} != {CURVE_GRID}"));
        }
        return Ok((te.get("silu_t_emb_grid")?, t_dim));
    }
    let w_in = te.get("time_embedder.proj_in.weight")?; // [hidden, 256]
    let b_in = te.get("time_embedder.proj_in.bias")?;
    let w_out = te.get("time_embedder.proj_out.weight")?; // [t_dim, hidden]
    let b_out = te.get("time_embedder.proj_out.bias")?;
    let hidden = b_in.len();
    let t_dim = b_out.len();
    let half = FREQ_DIM / 2;
    let mut out = vec![0f32; CURVE_GRID * t_dim];
    for i in 0..CURVE_GRID {
        let t = i as f32 / (CURVE_GRID - 1) as f32;
        // cos before sin, fp32 throughout — the reference's order.
        let mut emb = vec![0f32; FREQ_DIM];
        for j in 0..half {
            let f = (-(10000f32.ln()) * j as f32 / half as f32).exp();
            let (s, c) = (t * f).sin_cos();
            emb[j] = c;
            emb[half + j] = s;
        }
        let mut h = vec![0f32; hidden];
        for (o, hv) in h.iter_mut().enumerate() {
            let row = &w_in[o * FREQ_DIM..(o + 1) * FREQ_DIM];
            *hv = silu(b_in[o] + row.iter().zip(&emb).map(|(&a, &b)| a * b).sum::<f32>());
        }
        let dst = &mut out[i * t_dim..(i + 1) * t_dim];
        for (o, d) in dst.iter_mut().enumerate() {
            let row = &w_out[o * hidden..(o + 1) * hidden];
            // The AdalnProj on a full checkpoint applies silu to the
            // embedding before its linear; curve-form checkpoints fold
            // that in, so the coordinate we tabulate must have it too.
            *d = silu(b_out[o] + row.iter().zip(&h).map(|(&a, &b)| a * b).sum::<f32>());
        }
    }
    Ok((out, t_dim))
}

/// One block's merged adaLN: the pruned rank-8 weight beside the LoRA's
/// rank-16 update, and the [grid, 24] coordinate table that drives them.
struct Adaln {
    weight: Vec<f32>, // [out, 8 + r]
    table: Vec<f32>,  // [CURVE_GRID, 8 + r]
    cols: usize,
}

fn merge_adaln(
    dit: &StFile,
    lora: Option<&StFile>,
    key: &str,
    base_table: &[f32],
    base_cols: usize,
    curve: &[f32],
    t_dim: usize,
    scale: f32,
) -> anyhow::Result<Adaln> {
    let w_p = dit.get(&format!("{key}.weight"))?;
    let shape = dit.shape(&format!("{key}.weight")).unwrap();
    let (out, k) = (shape[0], shape[1]);
    if k != base_cols {
        return Err(anyhow!(
            "{key}: rank {k} but adaln_t_table is {base_cols} wide"
        ));
    }
    let a = lora.and_then(|l| l.get(&format!("{key}.lora_A.weight")).ok());
    let b = lora.and_then(|l| l.get(&format!("{key}.lora_B.weight")).ok());
    let (a, b) = match (a, b) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            return Ok(Adaln {
                weight: w_p,
                table: base_table.to_vec(),
                cols: k,
            });
        }
    };
    let r = a.len() / t_dim;
    if b.len() != out * r {
        return Err(anyhow!("{key}: lora_B is {} not {out}x{r}", b.len()));
    }
    let cols = k + r;
    // [W_p | scale·B] — the scale rides on the weight rather than the
    // table so the LoRA-strength dial stays a single multiply here.
    let mut weight = vec![0f32; out * cols];
    for o in 0..out {
        weight[o * cols..o * cols + k].copy_from_slice(&w_p[o * k..(o + 1) * k]);
        for j in 0..r {
            weight[o * cols + k + j] = scale * b[o * r + j];
        }
    }
    // [u(t) | A·silu(e(t))]
    let mut table = vec![0f32; CURVE_GRID * cols];
    for i in 0..CURVE_GRID {
        let dst = &mut table[i * cols..(i + 1) * cols];
        dst[..k].copy_from_slice(&base_table[i * k..(i + 1) * k]);
        let e = &curve[i * t_dim..(i + 1) * t_dim];
        for j in 0..r {
            let arow = &a[j * t_dim..(j + 1) * t_dim];
            dst[k + j] = arow.iter().zip(e).map(|(&x, &y)| x * y).sum();
        }
    }
    Ok(Adaln {
        weight,
        table,
        cols,
    })
}

// ── components ──────────────────────────────────────────────────────

/// The DiT's per-block projections, in the order the runtime wants them.
const BLOCK_PROJ: [&str; 4] = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"];
const BLOCK_NORM: [&str; 4] = ["norm1", "norm2", "attn.q_norm", "attn.k_norm"];

/// Merge the LoRA into one projection and push it at `level`.
fn push_proj(
    specs: &mut Vec<TensorSpec>,
    src: &StFile,
    lora: Option<&StFile>,
    key: &str,
    out_name: &str,
    level: Level,
    scale: f32,
) -> anyhow::Result<()> {
    let name = format!("{key}.weight");
    let shape = src
        .shape(&name)
        .ok_or_else(|| anyhow!("missing {name}"))?
        .to_vec();
    let mut w = src.get(&name)?;
    if let Some(l) = lora {
        let (an, bn) = (
            format!("{key}.lora_A.weight"),
            format!("{key}.lora_B.weight"),
        );
        if l.has(&an) && l.has(&bn) {
            lora_merge(
                &mut w,
                &l.get(&bn)?,
                &l.get(&an)?,
                shape[0],
                shape[1],
                scale,
            );
        }
    }
    specs.push(spec(out_name.to_string(), &w, shape, level));
    Ok(())
}

fn push_exact(
    specs: &mut Vec<TensorSpec>,
    src: &StFile,
    key: &str,
    out_name: &str,
    level: Level,
) -> anyhow::Result<()> {
    let shape = src
        .shape(key)
        .ok_or_else(|| anyhow!("missing {key}"))?
        .to_vec();
    let v = src.get(key)?;
    // 5-D 1x1x1 convs (the VAE quant convs) are matrices wearing a hat.
    let shape = if shape.len() > 2 && shape[2..].iter().all(|&d| d == 1) {
        vec![shape[0], shape[1]]
    } else {
        shape
    };
    specs.push(spec(out_name.to_string(), &v, shape, level));
    Ok(())
}

fn pack_dit(
    specs: &mut Vec<TensorSpec>,
    path: &Path,
    lora_path: Option<&Path>,
    temb_path: Option<&Path>,
    scale: f32,
    level: Level,
) -> anyhow::Result<serde_json::Value> {
    let dit = StFile::open(path)?;
    let lora = match lora_path {
        Some(p) => Some(StFile::open(p)?),
        None => None,
    };
    let n_blocks = (0..)
        .take_while(|i| dit.has(&format!("blocks.{i}.attn.qkv_proj.weight")))
        .count();
    let n_refiner = (0..)
        .take_while(|i| dit.has(&format!("token_refiner.blocks.{i}.attn.qkv_proj.weight")))
        .count();
    if n_blocks == 0 {
        return Err(anyhow!("{}: no blocks", path.display()));
    }
    let hidden = dit
        .shape("video_patch_proj.weight")
        .ok_or_else(|| anyhow!("video_patch_proj"))?[0];
    let head_dim = dit
        .shape("blocks.0.attn.q_norm.weight")
        .ok_or_else(|| anyhow!("q_norm"))?[0];
    let heads = dit.shape("blocks.0.attn.qkv_proj.weight").unwrap()[0] / (3 * head_dim);
    let ffn = dit.shape("blocks.0.mlp.fc1.weight").unwrap()[0] / 2;
    let text_dim = dit.shape("condition_proj.weight").unwrap()[1];
    let latents_dim = dit.shape("final_layer.video_out.weight").unwrap()[0] / 4;
    let audio_dim = dit.shape("final_layer.audio_out.weight").unwrap()[0];

    // The curve. A pruned base is required: the whole point of this
    // packer is that adaLN never becomes 13 B parameters again.
    let base_table = dit
        .get("adaln_t_table")
        .map_err(|_| anyhow!("{}: not a pruned/curve checkpoint (no adaln_t_table) — pass minimax_h3_*_pruned_bf16.safetensors", path.display()))?;
    let base_cols = dit.shape("adaln_t_table").unwrap()[1];
    let grid = dit.shape("adaln_t_table").unwrap()[0];
    if grid != CURVE_GRID {
        return Err(anyhow!("adaln_t_table grid {grid} != {CURVE_GRID}"));
    }
    let (curve, t_dim) = match temb_path {
        Some(p) => time_curve(&StFile::open(p)?)?,
        None if lora.is_some() => {
            return Err(anyhow!(
                "--lora on a pruned base needs --time-embedder (the LoRA's adaLN update \
                 is written against the full 2688-d timestep embedding)"
            ));
        }
        None => (Vec::new(), 0),
    };

    for (i, prefix) in (0..n_blocks)
        .map(|i| (i, format!("blocks.{i}")))
        .chain((0..n_refiner).map(|i| (i, format!("token_refiner.blocks.{i}"))))
    {
        let _ = i;
        for p in BLOCK_PROJ {
            push_proj(
                specs,
                &dit,
                lora.as_ref(),
                &format!("{prefix}.{p}"),
                &format!("dit.{prefix}.{p}.weight"),
                level,
                scale,
            )?;
        }
        for n in BLOCK_NORM {
            push_exact(
                specs,
                &dit,
                &format!("{prefix}.{n}.weight"),
                &format!("dit.{prefix}.{n}.weight"),
                Level::F32,
            )?;
        }
        if prefix.starts_with("blocks.") {
            let ad = merge_adaln(
                &dit,
                lora.as_ref(),
                &format!("{prefix}.adaln_proj.linear"),
                &base_table,
                base_cols,
                &curve,
                t_dim,
                scale,
            )?;
            let rows = ad.weight.len() / ad.cols;
            specs.push(spec(
                format!("dit.{prefix}.adaln.weight"),
                &ad.weight,
                vec![rows, ad.cols],
                Level::F16,
            ));
            specs.push(spec(
                format!("dit.{prefix}.adaln.table"),
                &ad.table,
                vec![CURVE_GRID, ad.cols],
                Level::F32,
            ));
            push_exact(
                specs,
                &dit,
                &format!("{prefix}.adaln_proj.linear.bias"),
                &format!("dit.{prefix}.adaln.bias"),
                Level::F32,
            )?;
        }
        eprint!(
            "\rdit blocks {}/{}   ",
            specs.len() / 9,
            n_blocks + n_refiner
        );
    }
    eprintln!();

    // final layer + heads: the checkpoint's fp32 island, kept fp32.
    let ad = merge_adaln(
        &dit,
        lora.as_ref(),
        "final_layer.adaln_proj.linear",
        &base_table,
        base_cols,
        &curve,
        t_dim,
        scale,
    )?;
    let rows = ad.weight.len() / ad.cols;
    specs.push(spec(
        "dit.final_layer.adaln.weight".into(),
        &ad.weight,
        vec![rows, ad.cols],
        Level::F32,
    ));
    specs.push(spec(
        "dit.final_layer.adaln.table".into(),
        &ad.table,
        vec![CURVE_GRID, ad.cols],
        Level::F32,
    ));
    for (k, o, l) in [
        (
            "final_layer.adaln_proj.linear.bias",
            "dit.final_layer.adaln.bias",
            Level::F32,
        ),
        (
            "final_layer.norm.weight",
            "dit.final_layer.norm.weight",
            Level::F32,
        ),
        (
            "final_layer.video_out.weight",
            "dit.final_layer.video_out.weight",
            Level::F32,
        ),
        (
            "final_layer.video_out.bias",
            "dit.final_layer.video_out.bias",
            Level::F32,
        ),
        (
            "final_layer.audio_out.weight",
            "dit.final_layer.audio_out.weight",
            Level::F32,
        ),
        (
            "final_layer.audio_out.bias",
            "dit.final_layer.audio_out.bias",
            Level::F32,
        ),
        (
            "video_patch_proj.weight",
            "dit.video_patch_proj.weight",
            Level::F32,
        ),
        (
            "video_patch_proj.bias",
            "dit.video_patch_proj.bias",
            Level::F32,
        ),
        (
            "audio_patch_proj.weight",
            "dit.audio_patch_proj.weight",
            Level::F32,
        ),
        (
            "audio_patch_proj.bias",
            "dit.audio_patch_proj.bias",
            Level::F32,
        ),
        ("condition_proj.bias", "dit.condition_proj.bias", Level::F32),
        (
            "token_refiner.final_norm.weight",
            "dit.token_refiner.final_norm.weight",
            Level::F32,
        ),
        ("rope.inv_freq", "dit.rope_inv_freq", Level::F32),
    ] {
        push_exact(specs, &dit, k, o, l)?;
    }
    push_proj(
        specs,
        &dit,
        lora.as_ref(),
        "condition_proj",
        "dit.condition_proj.weight",
        level,
        scale,
    )?;

    Ok(serde_json::json!({
        "hidden_size": hidden,
        "num_layers": n_blocks,
        "token_refiner_num_layers": n_refiner,
        "num_attention_heads": heads,
        "attention_head_dim": head_dim,
        "ffn_hidden_size": ffn,
        "latents_dim": latents_dim,
        "audio_latents_dim": audio_dim,
        "text_dim": text_dim,
        "patch_size": [1, 2, 2],
        "adaln_curve_grid": CURVE_GRID,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5,
        "sigma_shift_video": 12.0,
        "sigma_shift_audio": 3.0,
    }))
}

fn pack_te(
    specs: &mut Vec<TensorSpec>,
    path: &Path,
    level: Level,
    embed_level: Level,
    keep: Option<usize>,
) -> anyhow::Result<serde_json::Value> {
    let te = StFile::open(path)?;
    // Qwen3-VL ships under two roots: a text-only truncation keeps the
    // plain `model.` of a Qwen3, while a full VL checkpoint (what the
    // ComfyUI single-files are) puts the LM under `model.language_model.`
    // beside `model.visual.`. Same tensors either way.
    let root = ["model.", "model.language_model."]
        .into_iter()
        .find(|r| te.has(&format!("{r}layers.0.self_attn.q_proj.weight")))
        .ok_or_else(|| {
            anyhow!(
                "{}: no layers under model. or model.language_model.",
                path.display()
            )
        })?;
    let have = (0..)
        .take_while(|i| te.has(&format!("{root}layers.{i}.self_attn.q_proj.weight")))
        .count();
    // The conditioning is a TAP, not a whole encoder: H3's own file is
    // the 32 B truncated at 50, and a ClipProj stand-in is truncated at
    // the layer its projection was fitted on. Layers above the tap are
    // never executed, so packing them is pure file size.
    let n = match keep {
        Some(k) if k == 0 || k > have => {
            return Err(anyhow!("--te-layers {k}: the checkpoint has {have} layers"));
        }
        Some(k) => k,
        None => have,
    };
    if n < have {
        eprintln!("te: tapping {n} of {have} layers");
    }
    let sh = |k: &str| te.shape(&format!("{root}{k}")).map(|s| s.to_vec());
    let hidden = sh("layers.0.self_attn.q_proj.weight").unwrap()[1];
    let head_dim = sh("layers.0.self_attn.q_norm.weight").unwrap()[0];
    let nh = sh("layers.0.self_attn.q_proj.weight").unwrap()[0] / head_dim;
    let nkv = sh("layers.0.self_attn.k_proj.weight").unwrap()[0] / head_dim;
    let inter = sh("layers.0.mlp.gate_proj.weight").unwrap()[0];
    let vocab = sh("embed_tokens.weight").unwrap()[0];

    push_exact(
        specs,
        &te,
        &format!("{root}embed_tokens.weight"),
        "te.embed_tokens.weight",
        embed_level,
    )?;
    for l in 0..n {
        let p = format!("{root}layers.{l}");
        for k in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            push_exact(
                specs,
                &te,
                &format!("{p}.{k}.weight"),
                &format!("te.layers.{l}.{k}.weight"),
                level,
            )?;
        }
        for k in [
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ] {
            push_exact(
                specs,
                &te,
                &format!("{p}.{k}.weight"),
                &format!("te.layers.{l}.{k}.weight"),
                Level::F32,
            )?;
        }
        eprint!("\rte layers {}/{n}   ", l + 1);
    }
    eprintln!();
    Ok(serde_json::json!({
        "hidden_size": hidden,
        "num_hidden_layers": n,
        "num_attention_heads": nh,
        "num_key_value_heads": nkv,
        "head_dim": head_dim,
        "intermediate_size": inter,
        "vocab_size": vocab,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000.0,
        // The conditioning is the UNNORMALIZED stream after the last
        // layer: the checkpoint is truncated there and ships no
        // final norm.
        "final_norm": false,
    }))
}

/// ClipProj: a SMALL Qwen3-VL stands in for the 32 B encoder, and a
/// fitted affine map carries its tapped hidden state into the space the
/// DiT was conditioned on —
///
/// ```text
/// cond = ((h - mean_in) / std_in) @ W * std_out + mean_out
/// ```
///
/// with the GELU residual, when the file carries one, added in the
/// STANDARDIZED space (before `std_out`/`mean_out`), and token 0 —
/// the attention sink, an outlier no regression fits — overwritten by
/// a stored vector rather than projected.
///
/// Every byte of it stays exact. It is 304 MB deciding whether a 13 GB
/// saving still reads as the same prompt; quantizing the regression
/// that carries the whole substitution would be a false economy.
fn pack_clip_proj(specs: &mut Vec<TensorSpec>, path: &Path) -> anyhow::Result<serde_json::Value> {
    pack_clip_proj_at(specs, path, "te.proj")
}

/// Same packing under an arbitrary prefix — `te.proj.vis` carries the
/// vision-row twin the runtime routes image spans through.
fn pack_clip_proj_at(
    specs: &mut Vec<TensorSpec>,
    path: &Path,
    prefix: &str,
) -> anyhow::Result<serde_json::Value> {
    let p = StFile::open(path)?;
    let w = p
        .shape("W")
        .ok_or_else(|| anyhow!("{}: no W — not a ClipProj file", path.display()))?
        .to_vec();
    if w.len() != 2 {
        return Err(anyhow!(
            "{}: W is {:?}, expected [d_in, d_out]",
            path.display(),
            w
        ));
    }
    let (d_in, d_out) = (w[0], w[1]);
    for k in ["W", "mean_in", "std_in", "mean_out", "std_out", "sink_out"] {
        push_exact(specs, &p, k, &format!("{prefix}.{k}"), Level::F32)?;
    }
    let mlp = p.has("mlp.0.weight");
    let mut mlp_hidden = 0usize;
    if mlp {
        mlp_hidden = p
            .shape("mlp.0.weight")
            .ok_or_else(|| anyhow!("clip-proj: mlp.0.weight has no shape"))?[0];
        for k in ["mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"] {
            push_exact(specs, &p, k, &format!("{prefix}.{k}"), Level::F32)?;
        }
    }
    eprintln!(
        "clip-proj: {d_in} -> {d_out}{}",
        if mlp {
            format!(" + GELU residual through {mlp_hidden}")
        } else {
            String::new()
        }
    );
    Ok(serde_json::json!({
        "d_in": d_in,
        "d_out": d_out,
        "mlp": mlp,
        "mlp_hidden": mlp_hidden,
    }))
}

/// Video VAE: the ViT3D decoder only. The 3-D conv encoder is a third
/// of the file and t2va never runs it.
fn pack_video_vae(
    specs: &mut Vec<TensorSpec>,
    path: &Path,
    level: Level,
    heads: usize,
) -> anyhow::Result<serde_json::Value> {
    let v = StFile::open(path)?;
    let n = (0..)
        .take_while(|i| {
            v.has(&format!(
                "decoder.transformer_blocks.{i}.attn.to_qkv.weight"
            ))
        })
        .count();
    if n == 0 {
        return Err(anyhow!("{}: no decoder blocks", path.display()));
    }
    let dim = v.shape("decoder.x_embedder.weight").unwrap()[0];
    let z = v.shape("decoder.x_embedder.weight").unwrap()[1];
    // Nothing in the checkpoint says how `dim` splits into heads; it is
    // an architecture constant of the release, overridable for the
    // parity toy, which is smaller.
    for l in 0..n {
        let p = format!("decoder.transformer_blocks.{l}");
        for k in ["attn.to_qkv", "attn.to_out", "ff.w1", "ff.w2"] {
            push_exact(
                specs,
                &v,
                &format!("{p}.{k}.weight"),
                &format!("vvae.blocks.{l}.{k}.weight"),
                level,
            )?;
            push_exact(
                specs,
                &v,
                &format!("{p}.{k}.bias"),
                &format!("vvae.blocks.{l}.{k}.bias"),
                Level::F32,
            )?;
        }
        for k in ["norm1", "norm2", "scale1", "scale2"] {
            let src = if k.starts_with("scale") {
                format!("{p}.{k}")
            } else {
                format!("{p}.{k}.weight")
            };
            push_exact(specs, &v, &src, &format!("vvae.blocks.{l}.{k}"), Level::F32)?;
        }
    }
    for (k, o) in [
        ("decoder.x_embedder.weight", "vvae.x_embedder.weight"),
        ("decoder.x_embedder.bias", "vvae.x_embedder.bias"),
        ("decoder.register_tokens", "vvae.register_tokens"),
        ("decoder.norm_out.weight", "vvae.norm_out.weight"),
        ("decoder.norm_out.bias", "vvae.norm_out.bias"),
        ("decoder.proj_out.bias", "vvae.proj_out.bias"),
        ("post_quant_conv.weight", "vvae.post_quant_conv.weight"),
        ("post_quant_conv.bias", "vvae.post_quant_conv.bias"),
        ("latents_mean", "vvae.latents_mean"),
        ("latents_std", "vvae.latents_std"),
    ] {
        push_exact(specs, &v, k, o, Level::F32)?;
    }
    push_exact(
        specs,
        &v,
        "decoder.proj_out.weight",
        "vvae.proj_out.weight",
        level,
    )?;

    // ── the encoder, one temporal tap of it ──
    //
    // fl2va conditions on single FRAMES, and a causal 3-D convolution
    // fed one frame pads its front with zeros — the reference's own
    // `autopad="causal_zero"` trims the kernel to `weight[:, :, -T:]`,
    // which at T = 1 is the last tap and nothing else. The other two
    // taps cannot be reached by a keyframe, so they are not packed: the
    // encoder lands at a third of its size. Encoding real video (ref2va)
    // would need them back.
    // Level count, blocks per level and which levels downsample all come
    // from the checkpoint; only the spatial stride is architectural (every
    // downsample in this encoder halves H and W).
    let n_levels = (0..)
        .take_while(|i| v.has(&format!("encoder.down.{i}.block.0.conv1.weight")))
        .count();
    let n_res = (0..)
        .take_while(|j| v.has(&format!("encoder.down.0.block.{j}.conv1.weight")))
        .count();
    let space_down: Vec<usize> = (0..n_levels)
        .map(|i| {
            if v.has(&format!("encoder.down.{i}.downsample.conv.weight")) {
                2
            } else {
                1
            }
        })
        .collect();
    for (i, &sd) in space_down.iter().enumerate() {
        for j in 0..n_res {
            let p = format!("encoder.down.{i}.block.{j}");
            for k in ["conv1", "conv2", "nin_shortcut"] {
                if v.has(&format!("{p}.{k}.weight")) {
                    push_last_tap(
                        specs,
                        &v,
                        &format!("{p}.{k}"),
                        &format!("vvae.enc.down.{i}.block.{j}.{k}"),
                        level,
                    )?;
                }
            }
            for k in ["norm1", "norm2"] {
                for t in ["weight", "bias"] {
                    push_exact(
                        specs,
                        &v,
                        &format!("{p}.{k}.{t}"),
                        &format!("vvae.enc.down.{i}.block.{j}.{k}.{t}"),
                        Level::F32,
                    )?;
                }
            }
        }
        if sd > 1 {
            push_last_tap(
                specs,
                &v,
                &format!("encoder.down.{i}.downsample.conv"),
                &format!("vvae.enc.down.{i}.downsample.conv"),
                level,
            )?;
        }
    }
    push_last_tap(specs, &v, "encoder.conv_in", "vvae.enc.conv_in", level)?;
    push_last_tap(specs, &v, "encoder.conv_out", "vvae.enc.conv_out", level)?;
    for t in ["weight", "bias"] {
        push_exact(
            specs,
            &v,
            &format!("encoder.norm_out.{t}"),
            &format!("vvae.enc.norm_out.{t}"),
            Level::F32,
        )?;
        push_exact(
            specs,
            &v,
            &format!("quant_conv.{t}"),
            &format!("vvae.quant_conv.{t}"),
            Level::F32,
        )?;
    }

    Ok(serde_json::json!({
        "num_layers": n, "dim": dim, "heads": heads, "dim_head": dim / heads,
        "z_channels": z, "patch_size": 16, "patch_size_t": 4,
        "rope_theta": 100.0, "rope_dim_ratio": 0.75,
        "num_register_tokens": 4, "eps": 1e-5,
        "space_down": space_down, "num_res_blocks": n_res, "num_levels": n_levels,
    }))
}

/// A 5-D causal convolution reduced to the one temporal tap a single
/// frame can reach: `[o, i, kt, kh, kw]` → `[o, i, kh, kw]`, plus its
/// bias unchanged.
fn push_last_tap(
    specs: &mut Vec<TensorSpec>,
    src: &StFile,
    key: &str,
    out_name: &str,
    level: Level,
) -> anyhow::Result<()> {
    let name = format!("{key}.weight");
    let shape = src
        .shape(&name)
        .ok_or_else(|| anyhow!("missing {name}"))?
        .to_vec();
    if shape.len() != 5 {
        return Err(anyhow!("{name}: expected a 5-D kernel, got {shape:?}"));
    }
    let (o, i, kt, kh, kw) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let w = src.get(&name)?;
    let mut tap = vec![0f32; o * i * kh * kw];
    for oi in 0..o {
        for ii in 0..i {
            let s = ((oi * i + ii) * kt + (kt - 1)) * kh * kw;
            let d = (oi * i + ii) * kh * kw;
            tap[d..d + kh * kw].copy_from_slice(&w[s..s + kh * kw]);
        }
    }
    // f16 unless the caller asked for exact — the parity gate does, or
    // it measures the codec instead of the port.
    let lv = if level == Level::F32 {
        Level::F32
    } else {
        Level::F16
    };
    specs.push(spec(
        format!("{out_name}.weight"),
        &tap,
        vec![o, i, kh, kw],
        lv,
    ));
    push_exact(
        specs,
        src,
        &format!("{key}.bias"),
        &format!("{out_name}.bias"),
        Level::F32,
    )
}

/// Audio VAE: `dec_in_proj` + BigVGAN. 90 M parameters — f16 all the
/// way; quantizing a vocoder buys 45 MB and costs audible hiss.
/// MiniMax-Music-3's AR stack: a Qwen3-8B backbone, three embedding
/// tables, the pruned audio head, and the 4-layer RVQ depth decoder.
///
/// Every projection in this checkpoint is FUSED where the rest of this
/// engine expects them split — `qkv_proj` instead of q/k/v,
/// `gate_up_proj` instead of gate/up — so the split happens once, here.
/// The LM's is a GQA fuse (32 query heads and 8 key/value heads of 128,
/// hence 6144 rows and not 3×4096); the depth decoder's is not
/// (12288 = 3×4096). Splitting the first as if it were the second gives
/// a model that loads, runs, and is wrong.
/// Split a fused `[sum(rows)·, cols]` projection into named pieces,
/// row-major. The fuse order is the reference's: q|k|v and gate|up.
fn split_fused(
    specs: &mut Vec<TensorSpec>,
    t: &StFile,
    level: Level,
    src: &str,
    cols: usize,
    parts: &[(&str, usize)],
) -> anyhow::Result<()> {
    let v = t.get(src)?;
    let mut off = 0usize;
    for (name, rows) in parts {
        let n = rows * cols;
        push_exact_shaped(specs, name, &v[off..off + n], vec![*rows, cols], level);
        off += n;
    }
    if off != v.len() {
        return Err(anyhow!("{src}: split covered {off} of {}", v.len()));
    }
    Ok(())
}

fn pack_music3_te(
    specs: &mut Vec<TensorSpec>,
    path: &Path,
    level: Level,
) -> anyhow::Result<(serde_json::Value, Option<Vec<u8>>)> {
    let t = StFile::open(path)?;
    let hidden = t
        .shape("model.layers.0.self_attn.o_proj.weight")
        .ok_or_else(|| anyhow!("{}: no LM layers", path.display()))?[0];
    let head_dim = t.shape("model.layers.0.self_attn.q_norm.weight").unwrap()[0];
    let qkv_rows = t.shape("model.layers.0.self_attn.qkv_proj.weight").unwrap()[0];
    let nh = hidden / head_dim;
    // qkv = (nh + 2·nkv)·head_dim, so nkv falls out rather than being assumed.
    let nkv = (qkv_rows / head_dim - nh) / 2;
    let inter = t.shape("model.layers.0.mlp.down_proj.weight").unwrap()[1];
    let n_lm = (0..)
        .take_while(|i| t.has(&format!("model.layers.{i}.self_attn.qkv_proj.weight")))
        .count();
    let n_dec = (0..)
        .take_while(|i| {
            t.has(&format!(
                "model.audio_decoder.layers.{i}.self_attn.qkv_proj.weight"
            ))
        })
        .count();
    let dec_inter = t
        .shape("model.audio_decoder.layers.0.mlp.down_proj.weight")
        .unwrap()[1];

    for l in 0..n_lm {
        let s = format!("model.layers.{l}");
        let d = format!("mte.layers.{l}");
        split_fused(
            specs,
            &t,
            level,
            &format!("{s}.self_attn.qkv_proj.weight"),
            hidden,
            &[
                (&format!("{d}.self_attn.q_proj.weight"), nh * head_dim),
                (&format!("{d}.self_attn.k_proj.weight"), nkv * head_dim),
                (&format!("{d}.self_attn.v_proj.weight"), nkv * head_dim),
            ],
        )?;
        split_fused(
            specs,
            &t,
            level,
            &format!("{s}.mlp.gate_up_proj.weight"),
            hidden,
            &[
                (&format!("{d}.mlp.gate_proj.weight"), inter),
                (&format!("{d}.mlp.up_proj.weight"), inter),
            ],
        )?;
        for k in ["self_attn.o_proj", "mlp.down_proj"] {
            push_exact(
                &mut *specs,
                &t,
                &format!("{s}.{k}.weight"),
                &format!("{d}.{k}.weight"),
                level,
            )?;
        }
        for k in [
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ] {
            push_exact(
                &mut *specs,
                &t,
                &format!("{s}.{k}.weight"),
                &format!("{d}.{k}.weight"),
                Level::F32,
            )?;
        }
        eprint!("\rmusic3 te layer {}/{n_lm}   ", l + 1);
    }
    for l in 0..n_dec {
        let s = format!("model.audio_decoder.layers.{l}");
        let d = format!("mte.audio_decoder.layers.{l}");
        split_fused(
            specs,
            &t,
            level,
            &format!("{s}.self_attn.qkv_proj.weight"),
            hidden,
            &[
                (&format!("{d}.self_attn.q_proj.weight"), hidden),
                (&format!("{d}.self_attn.k_proj.weight"), hidden),
                (&format!("{d}.self_attn.v_proj.weight"), hidden),
            ],
        )?;
        split_fused(
            specs,
            &t,
            level,
            &format!("{s}.mlp.gate_up_proj.weight"),
            hidden,
            &[
                (&format!("{d}.mlp.gate_proj.weight"), dec_inter),
                (&format!("{d}.mlp.up_proj.weight"), dec_inter),
            ],
        )?;
        push_exact(
            &mut *specs,
            &t,
            &format!("{s}.self_attn.o_proj.weight"),
            &format!("{d}.self_attn.o_proj.weight"),
            level,
        )?;
        push_exact(
            &mut *specs,
            &t,
            &format!("{s}.mlp.down_proj.weight"),
            &format!("{d}.mlp.down_proj.weight"),
            level,
        )?;
        for k in ["input_layernorm", "post_attention_layernorm"] {
            push_exact(
                &mut *specs,
                &t,
                &format!("{s}.{k}.weight"),
                &format!("{d}.{k}.weight"),
                Level::F32,
            )?;
        }
    }
    eprintln!();
    // Embeddings and heads take the codec; norms and the 16-row
    // positional table do not.
    for n in [
        "model.embed_tokens_prefill.weight",
        "model.embed_tokens_audio.weight",
        "model.audio_extra_embedding.weight",
        "model.lm_head_pruned.weight",
        "model.audio_decoder.projection.weight",
    ] {
        push_exact(
            &mut *specs,
            &t,
            n,
            &format!("mte.{}", n.trim_start_matches("model.")),
            level,
        )?;
    }
    for i in 0.. {
        let n = format!("model.audio_decoder.audio_heads.{i}.weight");
        if !t.has(&n) {
            break;
        }
        push_exact(
            &mut *specs,
            &t,
            &n,
            &format!("mte.audio_decoder.audio_heads.{i}.weight"),
            level,
        )?;
    }
    for n in [
        "model.norm.weight",
        "model.audio_decoder.norm.weight",
        "model.audio_decoder.pos_embedding.weight",
    ] {
        push_exact(
            &mut *specs,
            &t,
            n,
            &format!("mte.{}", n.trim_start_matches("model.")),
            Level::F32,
        )?;
    }

    let vocab = t.raw("tokenizer_json").map(|b| b.to_vec());
    let audio_vocab = t.shape("model.audio_decoder.audio_heads.0.weight").unwrap()[0];
    let codebooks = (0..)
        .take_while(|i| t.has(&format!("model.audio_decoder.audio_heads.{i}.weight")))
        .count()
        + 1;
    Ok((
        serde_json::json!({
            "kind": "minimax_music3_ar",
            "hidden_size": hidden,
            "num_hidden_layers": n_lm,
            "num_attention_heads": nh,
            "num_key_value_heads": nkv,
            "head_dim": head_dim,
            "intermediate_size": inter,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "decoder_num_layers": n_dec,
            "decoder_intermediate_size": dec_inter,
            "decoder_num_heads": 16,
            "audio_vocab_size": audio_vocab,
            "audio_num_codebooks": codebooks,
            "c0_vocab_size": t.shape("model.embed_tokens_audio.weight").unwrap()[0],
            // The AR loop's own constants, from ComfyUI's ar.py.
            "cfg_scale": 1.5,
            "top_k": 50,
            "audio_frames_per_second": 25,
            "max_audio_frames": 9000,
        }),
        vocab,
    ))
}

/// MiniMax-Music-3's flow-matching DiT.
///
/// Quantize the wide projections and leave everything else exact. The
/// exceptions are not taste: `project_out` is 128 rows carrying the whole
/// velocity, the two 1×1 convs are residual corrections whose whole job
/// is a small delta, and `inv_freq`/`cond_layer_*`/the norms are tiny and
/// decide geometry rather than magnitude. Together they are a few MB
/// against 1.3 GB.
fn pack_music3_dit(
    specs: &mut Vec<TensorSpec>,
    path: &Path,
    level: Level,
) -> anyhow::Result<serde_json::Value> {
    let d = StFile::open(path)?;
    let layers = (0..)
        .take_while(|i| {
            d.has(&format!(
                "diffusion_transformer.transformer.layers.{i}.self_attn.to_qkv.weight"
            ))
        })
        .count();
    if layers == 0 {
        return Err(anyhow!("{}: no DiT layers", path.display()));
    }
    let mut quantized = 0usize;
    let mut exact = 0usize;
    for name in d.order.clone() {
        if !(name.starts_with("diffusion_transformer.")
            || name.starts_with("latent_conditioners.")
            || name.starts_with("cond_layer_"))
        {
            continue;
        }
        let shape = d.shape(&name).unwrap().to_vec();
        let vals = d.get(&name)?;
        // The wide 2-D projections take the codec; the rest is exact.
        let wide = shape.len() == 2
            && shape[1] % 32 == 0
            && shape[0] >= 512
            && !name.ends_with("project_out.weight");
        let lv = if wide { level } else { Level::F32 };
        if wide {
            quantized += 1;
        } else {
            exact += 1;
        }
        push_exact_shaped(specs, &format!("mdit.{name}"), &vals, shape, lv);
    }
    eprintln!("music3 dit: {layers} layers, {quantized} projections quantized, {exact} exact");
    Ok(serde_json::json!({
        "kind": "minimax_music3_dit",
        "num_layers": layers,
        "hidden": 2048,
        "num_heads": 32,
        "head_dim": 64,
        "rotary_dim": 32,
        "ff_inner": 8192,
        "in_channels": 128,
        "condition_dim": 2048,
        "concat_channels": 2304,
        "fourier_dim": 256,
        "cond_layers": 8,
        // The reference windows the transformer over long latents rather
        // than attending across a whole song, and averages the overlap.
        "max_condition_frames": 200,
        "condition_hop_frames": 100,
        "audio_frames_per_second": 25,
    }))
}

/// `push_exact` reads from a file by name; this takes values already in
/// hand, applying the same "1×1 convs are matrices wearing a hat" rule.
fn push_exact_shaped(
    specs: &mut Vec<TensorSpec>,
    name: &str,
    vals: &[f32],
    shape: Vec<usize>,
    level: Level,
) {
    let shape = if shape.len() > 2 && shape[2..].iter().all(|&d| d == 1) {
        vec![shape[0], shape[1]]
    } else {
        shape
    };
    specs.push(spec(name.to_string(), vals, shape, level));
}

/// MiniMax-Music-3's DAV decoder.
///
/// Two things separate this from `pack_audio_vae`, and both are silent
/// if got wrong. Its convs carry PyTorch weight normalisation — a
/// `weight_g` magnitude and a `weight_v` direction, never a `weight` —
/// so the runtime would find nothing to load; they are folded here,
/// once, as `g · v / ‖v‖` over each output channel. And its Snake α is
/// used verbatim by the reference where H3's BigVGAN keeps α and β in
/// log scale, so it must NOT be packed into that decoder's names.
///
/// Everything stays exact. A vocoder is the one stack in these files
/// where quantization buys tens of megabytes and costs audible hiss —
/// the H3 conversion already paid to learn that.
fn pack_music3_vae(specs: &mut Vec<TensorSpec>, path: &Path) -> anyhow::Result<serde_json::Value> {
    let a = StFile::open(path)?;
    let mut folded = 0usize;
    let mut plain = 0usize;
    for name in a.order.clone() {
        if !(name.starts_with("decoder.") || name.starts_with("dec_in_proj.")) {
            continue;
        }
        if name.ends_with(".weight_g") {
            continue; // consumed with its weight_v
        }
        if let Some(stem) = name.strip_suffix(".weight_v") {
            let gname = format!("{stem}.weight_g");
            let shape = a
                .shape(&name)
                .ok_or_else(|| anyhow!("missing shape {name}"))?
                .to_vec();
            let v = a.get(&name)?;
            let g = a.get(&gname)?;
            let per = v.len() / shape[0];
            let mut w = vec![0f32; v.len()];
            for o in 0..shape[0] {
                let row = &v[o * per..(o + 1) * per];
                let norm = row
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>()
                    .sqrt();
                let s = if norm > 0.0 { g[o] as f64 / norm } else { 0.0 };
                for (d, &sv) in w[o * per..(o + 1) * per].iter_mut().zip(row) {
                    *d = (sv as f64 * s) as f32;
                }
            }
            specs.push(spec(format!("mvae.{stem}.weight"), &w, shape, Level::F32));
            folded += 1;
            continue;
        }
        let shape = a.shape(&name).unwrap().to_vec();
        let vals = a.get(&name)?;
        specs.push(spec(format!("mvae.{name}"), &vals, shape, Level::F32));
        plain += 1;
    }
    if folded == 0 {
        return Err(anyhow!(
            "{}: no weight-normalised convs — not a Music-3 DAV file",
            path.display()
        ));
    }
    eprintln!("music3 vae: {folded} convs folded, {plain} tensors copied");
    Ok(serde_json::json!({
        "kind": "minimax_music3_dav",
        "latent_channels": 128,
        "channels_per_side": 64,
        "upsampling_ratios": [8, 8, 4, 2],
        "hop": 512,
        "sampling_rate": 44100,
        "resblock_dilations": [1, 3, 9],
        "snake_log_scale": false,
    }))
}

fn pack_audio_vae(
    specs: &mut Vec<TensorSpec>,
    path: &Path,
    level: Level,
) -> anyhow::Result<serde_json::Value> {
    let a = StFile::open(path)?;
    let mut n = 0usize;
    for name in a.order.clone() {
        // The decoder is what renders. The encoder half rides along
        // because the DiT can take a reference soundtrack — the packed
        // layout already carries a reference-audio segment with its own
        // condition timestep — and without these tensors there is
        // nothing in the file to turn a .wav into latents. The runtime
        // path for it is not written yet; the weights are here so that
        // when it is, nobody re-downloads the container.
        if !(name.starts_with("decoder.")
            || name.starts_with("dec_in_proj.")
            || name.starts_with("latents_")
            || name.starts_with("encoder.")
            || name.starts_with("enc_")
            || name.starts_with("pre_block")
            || name.starts_with("mean")
            || name.starts_with("logs"))
        {
            continue;
        }
        let shape = a.shape(&name).unwrap().to_vec();
        let vals = a.get(&name)?;
        // Conv1d kernels are [out, in, k]; keep the shape. f16 unless
        // the caller asked for exact — the parity gate does, because
        // f16 weight noise sits above a toy vocoder's own output.
        let lv = if shape.len() == 1 || level == Level::F32 {
            Level::F32
        } else {
            Level::F16
        };
        specs.push(spec(format!("avae.{name}"), &vals, shape, lv));
        n += 1;
    }
    if n == 0 {
        return Err(anyhow!("{}: no decoder tensors", path.display()));
    }
    Ok(serde_json::json!({
        "sample_rate": 32000, "hop_length": 800, "latents_per_second": 40,
        "latent_dim": 2048, "decoder_dim": 1024, "vae_latent_channels": 32,
        "upsample_rates": [5, 5, 2, 2, 2, 2, 2],
        "upsample_kernel_sizes": [9, 9, 4, 4, 4, 4, 4],
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    }))
}

/// Qwen3-VL's vision tower — the half of `fl2va` that rides in the text
/// stream. It lives inside the prompt encoder's file under `visual.`,
/// or at the top level of a standalone export.
fn pack_vision(
    specs: &mut Vec<TensorSpec>,
    src: &StFile,
    level: Level,
    heads: usize,
    deepstack_at: &[usize],
) -> anyhow::Result<Option<serde_json::Value>> {
    let pre = if src.has("visual.patch_embed.proj.weight") {
        "visual."
    } else if src.has("patch_embed.proj.weight") {
        ""
    } else {
        return Ok(None);
    };
    let n = (0..)
        .take_while(|i| src.has(&format!("{pre}blocks.{i}.attn.qkv.weight")))
        .count();
    let hidden = src.shape(&format!("{pre}patch_embed.proj.weight")).unwrap()[0];
    let pe = src
        .shape(&format!("{pre}patch_embed.proj.weight"))
        .unwrap()
        .to_vec();
    // Conv3d over a whole patch IS a linear: flatten [o, in, t, ph, pw].
    let flat: usize = pe[1..].iter().product();
    let w = src.get(&format!("{pre}patch_embed.proj.weight"))?;
    specs.push(spec(
        "vis.patch_embed.weight".into(),
        &w,
        vec![hidden, flat],
        level,
    ));
    push_exact(
        specs,
        src,
        &format!("{pre}patch_embed.proj.bias"),
        "vis.patch_embed.bias",
        Level::F32,
    )?;
    push_exact(
        specs,
        src,
        &format!("{pre}pos_embed.weight"),
        "vis.pos_embed.weight",
        Level::F32,
    )?;
    for i in 0..n {
        let p = format!("{pre}blocks.{i}");
        for (k, o) in [
            ("attn.qkv", "attn.qkv"),
            ("attn.proj", "attn.proj"),
            ("mlp.linear_fc1", "mlp.linear_fc1"),
            ("mlp.linear_fc2", "mlp.linear_fc2"),
        ] {
            push_exact(
                specs,
                src,
                &format!("{p}.{k}.weight"),
                &format!("vis.blocks.{i}.{o}.weight"),
                level,
            )?;
            push_exact(
                specs,
                src,
                &format!("{p}.{k}.bias"),
                &format!("vis.blocks.{i}.{o}.bias"),
                Level::F32,
            )?;
        }
        for k in ["norm1", "norm2"] {
            for t in ["weight", "bias"] {
                push_exact(
                    specs,
                    src,
                    &format!("{p}.{k}.{t}"),
                    &format!("vis.blocks.{i}.{k}.{t}"),
                    Level::F32,
                )?;
            }
        }
    }
    let mut mergers = vec![(format!("{pre}merger"), "vis.merger".to_string())];
    let n_deep = (0..)
        .take_while(|k| src.has(&format!("{pre}deepstack_merger_list.{k}.norm.weight")))
        .count();
    for k in 0..n_deep {
        mergers.push((
            format!("{pre}deepstack_merger_list.{k}"),
            format!("vis.deepstack.{k}"),
        ));
    }
    for (from, to) in mergers {
        for t in ["weight", "bias"] {
            push_exact(
                specs,
                src,
                &format!("{from}.norm.{t}"),
                &format!("{to}.norm.{t}"),
                Level::F32,
            )?;
        }
        for k in ["linear_fc1", "linear_fc2"] {
            push_exact(
                specs,
                src,
                &format!("{from}.{k}.weight"),
                &format!("{to}.{k}.weight"),
                level,
            )?;
            push_exact(
                specs,
                src,
                &format!("{from}.{k}.bias"),
                &format!("{to}.{k}.bias"),
                Level::F32,
            )?;
        }
    }
    let out_hidden = src
        .shape(&format!("{pre}merger.linear_fc2.weight"))
        .unwrap()[0];
    let inter = src
        .shape(&format!("{pre}blocks.0.mlp.linear_fc1.weight"))
        .unwrap()[0];
    // The patch geometry is in the embedding kernel [o, in, t, ph, pw],
    // and the merge factor falls out of the merger's input width. The
    // head count and which layers feed the deepstack are architecture,
    // written nowhere in the weights — hence the flags.
    let (temporal, patch_size) = (pe[2], pe[3]);
    let merge_dim = src
        .shape(&format!("{pre}merger.linear_fc1.weight"))
        .unwrap()[1];
    let merge = ((merge_dim / hidden) as f64).sqrt().round() as usize;
    let deep: Vec<usize> = deepstack_at.iter().copied().filter(|&d| d < n).collect();
    if deep.len() != n_deep {
        return Err(anyhow!(
            "--vis-deepstack lists {} usable layers but the checkpoint has {n_deep} mergers",
            deep.len()
        ));
    }
    Ok(Some(serde_json::json!({
        "hidden_size": hidden, "intermediate_size": inter, "depth": n,
        "num_heads": heads, "patch_size": patch_size,
        "temporal_patch_size": temporal, "spatial_merge_size": merge,
        "out_hidden_size": out_hidden, "deepstack_visual_indexes": deep,
    })))
}

fn config_spec(prefix: &str, cfg: &serde_json::Value) -> TensorSpec {
    let raw = serde_json::to_vec(cfg).expect("config json");
    TensorSpec {
        name: format!("{prefix}.config_json"),
        dtype: TensorDtype::U8,
        shape: vec![raw.len()],
        data: raw,
    }
}

// ── driver ──────────────────────────────────────────────────────────

pub struct PackArgs<'a> {
    pub out: &'a str,
    pub carry: Option<&'a str>,
    pub dit: Option<&'a str>,
    pub lora: Option<&'a str>,
    pub time_embedder: Option<&'a str>,
    pub lora_scale: f32,
    pub te: Option<&'a str>,
    pub clip_proj: Option<&'a str>,
    pub clip_proj_vis: Option<&'a str>,
    pub music_vae: Option<&'a str>,
    pub music_dit: Option<&'a str>,
    pub music_te: Option<&'a str>,
    pub te_layers: Option<usize>,
    pub video_vae: Option<&'a str>,
    pub audio_vae: Option<&'a str>,
    pub tokenizer: Option<&'a str>,
    pub quant: &'a str,
    pub vvae_heads: usize,
    pub vision: Option<&'a str>,
    pub vis_heads: usize,
    pub vis_deepstack: Vec<usize>,
}

pub fn cmd_animate_pack(args: PackArgs<'_>) -> anyhow::Result<()> {
    let level = match args.quant {
        "q4tp" | "q4" => Level::Q4tp,
        "q2tp" | "q2" => Level::Q2tp,
        "q8" => Level::Q8,
        "q8_2f" | "q82f" => Level::Q82f,
        "f16" => Level::F16,
        // Exact, for the parity gate: any quantization noise floor sits
        // above the arithmetic difference the gate is looking for.
        "f32" => Level::F32,
        other => {
            return Err(anyhow!(
                "--quant {other}: expected q4tp, q2tp, q8_2f, q8, f16 or f32"
            ));
        }
    };
    let t0 = std::time::Instant::now();
    let mut specs: Vec<TensorSpec> = Vec::new();
    let mut prov = serde_json::Map::new();
    let mut clip_d_out = 0usize;

    if let Some(p) = args.te {
        let cfg = pack_te(&mut specs, Path::new(p), level, level, args.te_layers)?;
        specs.push(config_spec("te", &cfg));
        prov.insert("te".into(), serde_json::json!(p));
        eprintln!("te packed ({:.1}s)", t0.elapsed().as_secs_f64());
    } else if args.te_layers.is_some() {
        return Err(anyhow!("--te-layers only means something with --te"));
    }
    let mut music_vocab: Option<Vec<u8>> = None;
    if let Some(p) = args.music_te {
        let (cfg, vocab) = pack_music3_te(&mut specs, Path::new(p), level)?;
        specs.push(config_spec("mte", &cfg));
        music_vocab = vocab;
        prov.insert("music_te".into(), serde_json::json!(p));
        eprintln!("music3 te packed ({:.1}s)", t0.elapsed().as_secs_f64());
    }
    if let Some(p) = args.music_dit {
        let cfg = pack_music3_dit(&mut specs, Path::new(p), level)?;
        specs.push(config_spec("mdit", &cfg));
        prov.insert("music_dit".into(), serde_json::json!(p));
        eprintln!("music3 dit packed ({:.1}s)", t0.elapsed().as_secs_f64());
    }
    if let Some(p) = args.music_vae {
        let cfg = pack_music3_vae(&mut specs, Path::new(p))?;
        specs.push(config_spec("mvae", &cfg));
        prov.insert("music_vae".into(), serde_json::json!(p));
        eprintln!("music3 vae packed ({:.1}s)", t0.elapsed().as_secs_f64());
    }
    if let Some(p) = args.clip_proj {
        if args.te.is_none() && args.carry.is_none() {
            return Err(anyhow!(
                "--clip-proj projects FROM an encoder: pass --te, or --in a file that has one"
            ));
        }
        let cfg = pack_clip_proj(&mut specs, Path::new(p))?;
        clip_d_out = cfg["d_out"].as_u64().unwrap_or(0) as usize;
        specs.push(config_spec("te.proj", &cfg));
        prov.insert("clip_proj".into(), serde_json::json!(p));
    }
    if let Some(p) = args.clip_proj_vis {
        if args.clip_proj.is_none() && args.carry.is_none() {
            return Err(anyhow!(
                "--clip-proj-vis rides a base projection: pass --clip-proj or --in a file with one"
            ));
        }
        pack_clip_proj_at(&mut specs, Path::new(p), "te.proj.vis")?;
        prov.insert("clip_proj_vis".into(), serde_json::json!(p));
    }
    if let Some(p) = args.dit {
        let cfg = pack_dit(
            &mut specs,
            Path::new(p),
            args.lora.map(Path::new),
            args.time_embedder.map(Path::new),
            args.lora_scale,
            level,
        )?;
        specs.push(config_spec("dit", &cfg));
        prov.insert("dit".into(), serde_json::json!(p));
        if let Some(l) = args.lora {
            prov.insert("lora".into(), serde_json::json!(l));
            prov.insert("lora_scale".into(), serde_json::json!(args.lora_scale));
        }
        eprintln!("dit packed ({:.1}s)", t0.elapsed().as_secs_f64());
    }
    if let Some(p) = args.vision {
        let src = StFile::open(Path::new(p))?;
        match pack_vision(&mut specs, &src, level, args.vis_heads, &args.vis_deepstack)? {
            Some(cfg) => {
                specs.push(config_spec("vis", &cfg));
                prov.insert("vision".into(), serde_json::json!(p));
                eprintln!("vision tower packed ({:.1}s)", t0.elapsed().as_secs_f64());
            }
            None => return Err(anyhow!("{p}: no vision tower (no patch_embed.proj)")),
        }
    }
    if let Some(p) = args.video_vae {
        let cfg = pack_video_vae(&mut specs, Path::new(p), level, args.vvae_heads)?;
        specs.push(config_spec("vvae", &cfg));
        prov.insert("video_vae".into(), serde_json::json!(p));
        eprintln!("video vae packed ({:.1}s)", t0.elapsed().as_secs_f64());
    }
    if let Some(p) = args.audio_vae {
        let cfg = pack_audio_vae(&mut specs, Path::new(p), level)?;
        specs.push(config_spec("avae", &cfg));
        prov.insert("audio_vae".into(), serde_json::json!(p));
        eprintln!("audio vae packed ({:.1}s)", t0.elapsed().as_secs_f64());
    }

    // Carry a previous pass through by REFERENCE: its payloads stay in
    // the source mmap and stream to the new file page by page, so a
    // second component does not need the first one's bytes in RAM.
    let carried = match args.carry {
        Some(p) => Some(Arc::new(
            CmfModel::open(p).map_err(|e| anyhow!("{p}: {e}"))?,
        )),
        None => None,
    };
    // A projection that does not land in the DiT's conditioning width
    // is some other model's projection. Say so before writing 14 GB.
    if clip_d_out != 0 {
        let dcfg = specs
            .iter()
            .find(|s| s.name == "dit.config_json")
            .map(|s| s.data.clone())
            .or_else(|| {
                carried
                    .as_ref()
                    .and_then(|m| m.tensor_bytes("dit.config_json").ok())
                    .map(|b| b.to_vec())
            });
        let want = dcfg
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| v["text_dim"].as_u64())
            .unwrap_or(0) as usize;
        if want != 0 && want != clip_d_out {
            return Err(anyhow!(
                "--clip-proj emits {clip_d_out}-wide conditioning, this DiT takes {want}"
            ));
        }
    }
    let mut refs: Vec<TensorSpecRef> = Vec::new();
    let mut vocab_carried: Option<Vec<u8>> = None;
    // Re-running a component replaces it WHOLE. Name-by-name is not
    // enough: a 25-layer encoder packed over a carried 50-layer one
    // leaves `te.layers.25..49` of the old stack in the file — six
    // gigabytes of weight that `num_hidden_layers` then excludes from
    // the forward, so it costs disk and VRAM budget and never runs.
    // `--clip-proj` alone replaces only the projection, because there
    // the carried encoder is the thing it projects from.
    let mut drop_pre: Vec<&str> = Vec::new();
    if args.te.is_some() {
        drop_pre.push("te.");
    }
    if args.clip_proj_vis.is_some() {
        drop_pre.push("te.proj.vis.");
    }
    if args.te.is_none() && args.clip_proj.is_some() {
        drop_pre.push("te.proj.");
    }
    for (on, pre) in [
        (args.dit.is_some(), "dit."),
        (args.vision.is_some(), "vis."),
        (args.video_vae.is_some(), "vvae."),
        (args.audio_vae.is_some(), "avae."),
    ] {
        if on {
            drop_pre.push(pre);
        }
    }
    if let Some(m) = &carried {
        let names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();
        for e in &m.tensors {
            if names.iter().any(|n| n == &e.name) {
                continue; // a re-run of the same component wins
            }
            if drop_pre.iter().any(|p| e.name.starts_with(p)) {
                continue; // …and takes the rest of its component with it
            }
            refs.push(TensorSpecRef {
                name: e.name.clone(),
                dtype: e.dtype,
                shape: e.shape.clone(),
                data: m.entry_bytes(e),
            });
        }
        vocab_carried = m.vocab.clone();
    }
    for s in &specs {
        refs.push(TensorSpecRef {
            name: s.name.clone(),
            dtype: s.dtype,
            shape: s.shape.clone(),
            data: &s.data,
        });
    }
    refs.sort_by(|a, b| a.name.cmp(&b.name));

    let vocab = match args.tokenizer {
        Some(p) => Some(std::fs::read(p).with_context(|| p.to_string())?),
        None => music_vocab.or(vocab_carried),
    };

    // The header describes the PROMPT ENCODER — it is the only stack in
    // the file with an LLM shape, and `cortiq info` is read by people
    // who want to know what they are about to load.
    let te_cfg = refs
        .iter()
        .find(|r| r.name == "te.config_json")
        .map(|r| serde_json::from_slice::<serde_json::Value>(r.data))
        .transpose()?
        .unwrap_or_else(|| serde_json::json!({}));
    // A pass that has not packed the encoder yet still has to produce a
    // readable header, so every field falls back to zero rather than
    // to a null the arch deserializer rejects.
    let n = |k: &str| te_cfg[k].as_u64().unwrap_or(0);
    let nl = n("num_hidden_layers") as usize;
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "minimax-h3-av",
        "hidden_size": n("hidden_size"),
        "intermediate_size": n("intermediate_size"),
        "num_layers": nl,
        "num_attention_heads": n("num_attention_heads"),
        "num_kv_heads": n("num_key_value_heads"),
        "head_dim": n("head_dim"),
        "vocab_size": n("vocab_size"),
        "layer_types": vec!["FullAttention"; nl],
        "rms_norm_eps": te_cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6),
        "max_position_embeddings": 262144,
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
    }))?;
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch,
        quant_type: match level {
            Level::Q8 | Level::Q82f => QuantType::Q8Row,
            Level::Q2tp => QuantType::Q4Block,
            _ => QuantType::Q4Block,
        },
        provenance: Some(serde_json::json!({
            "pipeline": "minimax-h3",
            "components": {
                "te": "qwen3-vl-32b (50 layers)",
                "dit": "minimax-h3 audio-video dit",
                "vvae": "vit3d video vae decoder",
                "avae": "bigvgan audio vae decoder",
            },
            "sources": prov,
            "quant": args.quant,
        })),
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
    };
    CmfModel::write_ref(args.out, &header, &refs, None, vocab.as_deref())
        .map_err(|e| anyhow!("write {}: {e}", args.out))?;
    let size = std::fs::metadata(args.out)?.len();
    println!(
        "{}: {} tensors, {:.2} GB in {:.1}s",
        args.out,
        refs.len(),
        size as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
