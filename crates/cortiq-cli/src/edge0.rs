//! `cortiq edge0-import`: an Edge0 release folder (Edge0-145B / Edge0-120B,
//! the NVFP4 build) → `.cmf`.
//!
//! Edge0 is Qwen3.8-Flash-Next with latent routed experts: per layer, eight
//! D/U pairs project the token from the 2560-wide stream to a 1792-wide
//! (145B) or 1280-wide (120B) one and back, and each of the 512 experts works
//! in the latent width of its group. Everything outside the routed experts is
//! the base model's: the router rows are its bf16 weights bit for bit, and
//! attention, the shared expert and the hyper-connections are Q8_0 of its
//! weights to within a block's rounding.
//!
//! The import therefore takes a CMF of Qwen3.8-Flash-Next for the skeleton
//! (attention, GDN, hyper-connections, embeddings, head, the MTP head and,
//! unless `--ple edge0`, the n-gram table) and the folder for the rest:
//!
//! - routed experts: NVFP4 re-packed into `nvfp4` tensors without touching a
//!   value (`[gate | up | down]` blobs of ggml `block_nvfp4` + the three
//!   per-projection f32 scales);
//! - D and U: Q8_0 (`latent_dense_q8.bin`) → q8_2f, U as one `[hidden,
//!   groups·dim]` matrix whose column block `g` is `U_g`;
//! - the group labels: `mlp.latent_groups`, one byte per expert;
//! - the router: all 512 rows, bf16 → f16;
//! - the shared expert: Q8_0 → q8_2f;
//! - the n-gram table with `--ple edge0`: the folder's FP8 table → q8_2f in
//!   the template's shards (checked row by row against the template's copy).

use anyhow::{Context, Result, bail, ensure};
use cortiq_core::format::CmfStreamWriter;
use cortiq_core::{CmfModel, TensorDtype};
use serde_json::json;
use std::path::Path;

use crate::convert::{Quant, quantize_2d};
use crate::gguf::{Gguf, GgufTensor, parse};

const COPY_CHUNK: usize = 16 << 20;
const GGML_F32: u32 = 0;
const GGML_Q8_0: u32 = 8;
const GGML_BF16: u32 = 30;
const GGML_NVFP4: u32 = 40;

fn f16(h: u16) -> f32 {
    cortiq_core::quant::f16_to_f32(h)
}

/// Q8_0 blocks (`{f16 d; i8 q[32]}`) → f32.
fn q8_0(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    ensure!(n % 32 == 0 && raw.len() >= n / 32 * 34, "q8_0: {} bytes for {n} values", raw.len());
    let mut out = Vec::with_capacity(n);
    for b in raw[..n / 32 * 34].chunks_exact(34) {
        let d = f16(u16::from_le_bytes([b[0], b[1]]));
        out.extend(b[2..].iter().map(|&q| (q as i8) as f32 * d));
    }
    Ok(out)
}

fn bf16(raw: &[u8], n: usize) -> Vec<f32> {
    raw[..2 * n].chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect()
}

/// OCP E4M3 (fn) → f32: sign bit 7, exponent bits 6..3 (bias 7), mantissa
/// bits 2..0, `S.1111.111` NaN (read as 0: the table has none).
fn e4m3(b: u8) -> f32 {
    let (s, e, m) = (b >> 7, (b >> 3) & 0xF, b & 7);
    let v = if e == 0xF && m == 7 {
        0.0
    } else if e == 0 {
        m as f32 * 2f32.powi(-9)
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
    };
    if s == 1 { -v } else { v }
}

struct Native {
    dim: usize,
    ff: usize,
    /// (offset in `experts.bin`, bytes) of layer `l`'s expert 0; the layer's
    /// experts follow at `bytes` each.
    layers: Vec<(u64, usize)>,
}

/// `native_experts.txt`: `# strata native experts v6: latent A-only
/// (expert_embd D, expert_ff F)` then `layer gu_type d_type offset stride`.
fn read_native(dir: &Path) -> Result<Native> {
    let p = dir.join("native_experts.txt");
    let s = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
    let num = |key: &str| -> Result<usize> {
        let i = s.find(key).with_context(|| format!("{}: no `{key}`", p.display()))? + key.len();
        let d: String = s[i..].chars().take_while(|c| c.is_ascii_digit()).collect();
        d.parse().with_context(|| format!("{}: `{key}` is not a number", p.display()))
    };
    let (dim, ff) = (num("expert_embd ")?, num("expert_ff ")?);
    let mut layers = Vec::new();
    for line in s.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let f: Vec<&str> = line.split_whitespace().collect();
        ensure!(f.len() == 5, "{}: bad line `{line}`", p.display());
        let (li, gu, dt): (usize, u32, u32) = (f[0].parse()?, f[1].parse()?, f[2].parse()?);
        ensure!(li == layers.len(), "{}: layers out of order at `{line}`", p.display());
        if gu != GGML_NVFP4 || dt != GGML_NVFP4 {
            bail!(
                "{}: layer {li} experts are ggml types {gu}/{dt}; only the NVFP4 build (type {GGML_NVFP4}) imports without loss",
                p.display()
            );
        }
        layers.push((f[3].parse()?, f[4].parse()?));
    }
    Ok(Native { dim, ff, layers })
}

fn tensor<'a>(g: &'a Gguf, name: &str) -> Result<&'a GgufTensor> {
    g.tensors.iter().find(|t| t.name == name).with_context(|| format!("GGUF tensor {name} missing"))
}

fn raw<'a>(g: &'a Gguf, t: &GgufTensor, bytes: usize) -> Result<&'a [u8]> {
    let a = g.data_start + t.offset as usize;
    g.bytes.get(a..a + bytes).with_context(|| format!("{}: {bytes} bytes past the end of the GGUF", t.name))
}

/// A skeleton matrix as f32 `[rows, cols]` (GGUF dims are `[cols, rows]`).
fn matrix(g: &Gguf, name: &str) -> Result<(Vec<f32>, usize, usize)> {
    let t = tensor(g, name)?;
    ensure!(t.dims.len() == 2, "{name}: {:?} is not a matrix", t.dims);
    let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let n = rows * cols;
    let v = match t.ggml_type {
        GGML_Q8_0 => q8_0(raw(g, t, n / 32 * 34)?, n)?,
        GGML_BF16 => bf16(raw(g, t, 2 * n)?, n),
        GGML_F32 => raw(g, t, 4 * n)?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        ty => bail!("{name}: ggml type {ty} not expected in the skeleton"),
    };
    Ok((v, rows, cols))
}

fn mmap(p: &Path) -> Result<memmap2::Mmap> {
    let f = std::fs::File::open(p).with_context(|| format!("opening {}", p.display()))?;
    // SAFETY: read-only map of a file we just opened and do not modify.
    Ok(unsafe { memmap2::Mmap::map(&f)? })
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

/// `cortiq edge0-import`.
pub fn cmd_edge0_import(base_path: &str, dir: &str, output: &str, ple: &str) -> Result<()> {
    ensure!(matches!(ple, "base" | "edge0"), "--ple must be base or edge0");
    ensure!(
        !crate::tube::paths_collide(Path::new(output), Path::new(base_path)),
        "edge0-import: the output must differ from the base"
    );
    let dir = Path::new(dir);
    let base_arc = std::sync::Arc::new(CmfModel::open_sharded(base_path)?);
    let base: &CmfModel = &base_arc;
    let arch = base.arch().clone();
    ensure!(arch.arch_name == "qwen4_exp", "{base_path}: a Qwen3.8-Flash-Next CMF is the base, this is {}", arch.arch_name);
    let (nl, hidden) = (arch.num_layers, arch.hidden_size);
    let native = read_native(dir)?;
    ensure!(native.layers.len() == nl, "native_experts.txt lists {} layers, the base has {nl}", native.layers.len());
    let (dim, ff) = (native.dim, native.ff);
    ensure!(dim % 64 == 0 && ff % 64 == 0, "latent {dim} / expert ff {ff} not multiples of 64");
    let gu_bytes = ff * (dim / 64) * cortiq_core::nvfp4::GGML_BLOCK_BYTES;
    let d_bytes = dim * (ff / 64) * cortiq_core::nvfp4::GGML_BLOCK_BYTES;
    let blob = 2 * gu_bytes + d_bytes + 16;
    let skel = parse(&dir.join("model-skeleton.gguf"))?;
    let ne = tensor(&skel, "blk.0.ffn_gate_inp.weight")?.dims[1] as usize;
    for (li, &(off, stride)) in native.layers.iter().enumerate() {
        ensure!(stride == blob, "layer {li}: expert stride {stride}, NVFP4 {dim}/{ff} wants {blob}");
        ensure!(off == (li * ne * blob) as u64, "layer {li}: offset {off} is not layer-major");
    }
    let experts = mmap(&dir.join("experts.bin"))?;
    ensure!(experts.len() == nl * ne * blob, "experts.bin: {} bytes, want {}", experts.len(), nl * ne * blob);
    let labels_raw = std::fs::read(dir.join("labels.bin")).context("reading labels.bin")?;
    ensure!(labels_raw.len() == nl * ne * 4, "labels.bin: {} bytes for {nl} x {ne} labels", labels_raw.len());
    let labels: Vec<u8> = labels_raw
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .map(|v| u8::try_from(v).map_err(|_| anyhow::anyhow!("labels.bin: label {v} out of range")))
        .collect::<Result<_>>()?;
    let groups = labels.iter().copied().max().map_or(0, |g| g as usize + 1);
    ensure!(groups > 0, "labels.bin: no groups");
    let gl = groups * dim;
    let dense = mmap(&dir.join("latent_dense_q8.bin"))?;
    let d_layer = gl * (hidden / 32) * 34;
    let u_layer = groups * hidden * (dim / 32) * 34;
    ensure!(
        dense.len() == nl * (d_layer + u_layer),
        "latent_dense_q8.bin: {} bytes, {nl} layers of D {gl}x{hidden} and U {groups}x{hidden}x{dim} in Q8_0 want {}",
        dense.len(),
        nl * (d_layer + u_layer)
    );
    eprintln!("edge0-import: {nl} layers, {ne} experts in {groups} groups, latent {dim}, expert ff {ff}, NVFP4 blob {blob} B");

    // The n-gram table of the folder, when asked for.
    let ple_src = if ple == "edge0" {
        let g = parse(&dir.join("ple-fp8.gguf"))?;
        let t = tensor(&g, "per_layer_token_embd.weight")?;
        let scale = match g.md.get("strata.ple.scale") {
            Some(v) => v.as_f64().context("strata.ple.scale is not a number")? as f32,
            None => bail!("ple-fp8.gguf: no strata.ple.scale"),
        };
        let (w, rows) = (t.dims[0] as usize, t.dims[1] as usize);
        let off = g.data_start + t.offset as usize;
        Some((g, off, w, rows, scale))
    } else {
        None
    };

    let is_routed = |n: &str| -> bool {
        n.strip_prefix("model.layers.").is_some_and(|r| r.split_once('.').is_some_and(|(_, r)| r.starts_with("mlp.experts.")))
    };
    let replaced = |n: &str| -> bool {
        let Some(rest) = n.strip_prefix("model.layers.") else { return false };
        let Some((_, r)) = rest.split_once('.') else { return false };
        r == "mlp.gate.weight" || r.starts_with("mlp.shared_expert.") && r.ends_with("_proj.weight")
    };
    let is_ple_shard = |n: &str| n.contains(".ple.ple_embedding.ngram_embedding.shard_");
    let mut kept = 0usize;
    let n_new = nl * (3 * ne + 3 + 4);
    let avg = base.tensors.iter().map(|e| e.name.len()).sum::<usize>() / base.tensors.len().max(1);
    let mut w = CmfStreamWriter::new(output, CmfStreamWriter::head_reserve_for(base.tensors.len() + n_new, avg + 12))?;
    let t0 = std::time::Instant::now();
    let mut ple_rows_done = 0usize;
    let mut ple_cos_min = 1f64;
    for e in &base.tensors {
        if is_routed(&e.name) || replaced(&e.name) {
            continue;
        }
        if let Some((g, off, width, rows, scale)) = ple_src.as_ref().filter(|_| is_ple_shard(&e.name)) {
            ensure!(e.shape.len() == 2 && e.shape[1] == *width, "{}: {:?} vs the folder's rows of {width}", e.name, e.shape);
            let s: usize = e.name.rsplit("shard_").next().and_then(|v| v.split('.').next()).and_then(|v| v.parse().ok()).context("shard index")?;
            let r0 = s * e.shape[0];
            ensure!(r0 + e.shape[0] <= *rows, "{}: rows {r0}.. past the folder's {rows}", e.name);
            let bytes = &g.bytes[off + r0 * width..off + (r0 + e.shape[0]) * width];
            let vals: Vec<f32> = bytes.iter().map(|&b| e4m3(b) * scale).collect();
            // the same table: compare rows spread over the shard with the template's copy
            let mut tmpl = vec![0f32; *width];
            let q = cortiq_engine::qtensor::QTensor::from_model(&base_arc, &e.name).ok();
            if let Some(qt) = q.as_ref() {
                for k in 0..8 {
                    let r = (k * 7919 + 13) % e.shape[0];
                    qt.row_f32(r, &mut tmpl);
                    let c = cosine(&vals[r * width..(r + 1) * width], &tmpl);
                    ple_cos_min = ple_cos_min.min(c);
                    ensure!(c > 0.9, "{} row {r}: cosine {c:.3} with the base's table: not the same rows", e.name);
                }
            }
            let (dt, data) = quantize_2d(Quant::Q8_2f, &vals, e.shape[0], *width);
            w.push(&e.name, dt, &e.shape, &data)?;
            ple_rows_done += e.shape[0];
            continue;
        }
        w.push_bounded(&e.name, e.dtype, &e.shape, base.entry_bytes(e), COPY_CHUNK)?;
        kept += 1;
    }
    eprintln!("edge0-import: {kept} skeleton tensors copied, {:.0} s", t0.elapsed().as_secs_f64());
    let mut nv_bytes = 0usize;
    for li in 0..nl {
        let p = format!("model.layers.{li}.mlp.");
        // router: all experts, bf16 rows → f16
        let (r, rows, cols) = matrix(&skel, &format!("blk.{li}.ffn_gate_inp.weight"))?;
        ensure!(rows == ne && cols == hidden, "layer {li} router {rows}x{cols}");
        let (dt, data) = quantize_2d(Quant::F16, &r, rows, cols);
        w.push(&format!("{p}gate.weight"), dt, &[rows, cols], &data)?;
        // shared expert: Q8_0 → q8_2f
        for (gn, part) in [("ffn_gate_shexp", "gate_proj"), ("ffn_up_shexp", "up_proj"), ("ffn_down_shexp", "down_proj")] {
            let (v, rows, cols) = matrix(&skel, &format!("blk.{li}.{gn}.weight"))?;
            let (dt, data) = quantize_2d(Quant::Q8_2f, &v, rows, cols);
            w.push(&format!("{p}shared_expert.{part}.weight"), dt, &[rows, cols], &data)?;
        }
        // D [groups·dim, hidden] and U [hidden, groups·dim]
        let lo = li * (d_layer + u_layer);
        let dv = q8_0(&dense[lo..lo + d_layer], gl * hidden)?;
        let (dt, data) = quantize_2d(Quant::Q8_2f, &dv, gl, hidden);
        w.push(&format!("{p}latent_down.weight"), dt, &[gl, hidden], &data)?;
        drop(dv);
        let uv = q8_0(&dense[lo + d_layer..lo + d_layer + u_layer], groups * hidden * dim)?;
        let mut u = vec![0f32; hidden * gl];
        for g in 0..groups {
            for h in 0..hidden {
                let src = &uv[(g * hidden + h) * dim..(g * hidden + h + 1) * dim];
                u[h * gl + g * dim..h * gl + (g + 1) * dim].copy_from_slice(src);
            }
        }
        drop(uv);
        let (dt, data) = quantize_2d(Quant::Q8_2f, &u, hidden, gl);
        w.push(&format!("{p}latent_up.weight"), dt, &[hidden, gl], &data)?;
        drop(u);
        w.push(&format!("{p}latent_groups"), TensorDtype::U8, &[ne], &labels[li * ne..(li + 1) * ne])?;
        // routed experts: NVFP4 blobs re-packed
        for e in 0..ne {
            let b = &experts[(li * ne + e) * blob..(li * ne + e + 1) * blob];
            let tail = &b[blob - 16..];
            let sc = |i: usize| f32::from_le_bytes([tail[4 * i], tail[4 * i + 1], tail[4 * i + 2], tail[4 * i + 3]]);
            let parts = [
                ("gate_proj", &b[..gu_bytes], ff, dim, sc(0)),
                ("up_proj", &b[gu_bytes..2 * gu_bytes], ff, dim, sc(1)),
                ("down_proj", &b[2 * gu_bytes..2 * gu_bytes + d_bytes], dim, ff, sc(2)),
            ];
            for (part, bytes, rows, cols, s) in parts {
                let data = cortiq_core::nvfp4::repack_ggml(bytes, rows, cols, s)
                    .map_err(|m| anyhow::anyhow!("layer {li} expert {e} {part}: {m}"))?;
                nv_bytes += data.len();
                w.push(&format!("{p}experts.{e}.{part}.weight"), TensorDtype::Nvfp4, &[rows, cols], &data)?;
            }
        }
        if (li + 1) % 8 == 0 || li + 1 == nl {
            eprintln!("edge0-import: {}/{nl} layers, {:.0} s", li + 1, t0.elapsed().as_secs_f64());
        }
    }
    let mut header = base.header.clone();
    let q = header.arch.qwen4_exp.as_mut().context("base header has no qwen4_exp block")?;
    q.latent_dim = Some(dim);
    q.latent_groups = Some(groups);
    if let Some(m) = header.arch.moe.as_mut() {
        m.num_experts = ne;
    }
    let model_json: serde_json::Value = std::fs::read(dir.join("model.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(json!({}));
    let mut prov = json!({});
    prov["edge0"] = json!({
        "source": model_json.get("display_name").cloned().unwrap_or(json!(null)),
        "build": model_json.get("build").cloned().unwrap_or(json!(null)),
        "latent_dim": dim,
        "latent_groups": groups,
        "experts": "NVFP4, re-packed bit for bit",
        "latent_dense": "Q8_0 -> q8_2f",
        "router": "bf16 -> f16",
        "shared_expert": "Q8_0 -> q8_2f",
        "ngram_table": if ple == "edge0" { "FP8 -> q8_2f" } else { "the base's" },
        "skeleton": Path::new(base_path).file_name().and_then(|s| s.to_str()),
    });
    header.provenance = Some(prov);
    w.finish(&header, (!base.masks.masks.is_empty()).then_some(&base.masks), base.vocab.as_deref())?;
    let gb = |p: &str| std::fs::metadata(p).map(|m| m.len() as f64 / 1e9).unwrap_or(0.0);
    println!(
        "edge0-import: {output} {:.2} GB ({:.2} GB of NVFP4 experts){}",
        gb(output),
        nv_bytes as f64 / 1e9,
        if ple_rows_done > 0 { format!(", n-gram table from the folder: {ple_rows_done} rows, min cosine to the base {ple_cos_min:.4}") } else { String::new() }
    );
    Ok(())
}
