//! `qwen4exp` GGUF (Qwen3.8-Flash-Next as llama.cpp-derived engines store it:
//! the ISTA-DASLab GSQ-RCO files Strata runs, including the expert-pruned
//! Coder) → `.cmf`.
//!
//! The architecture carries hyper-connections, QSA attention with a sparse
//! indexer, GatedDeltaNet mixers and an n-gram PLE table whose CMF header
//! fields the GGUF metadata does not spell out, so the import is TEMPLATE
//! driven: a CMF of the same model (any quantization) supplies the header,
//! tokenizer and the codec of every tensor, and the GGUF supplies the
//! weights. Each tensor's layout is settled against the template, which is
//! the same network: of the candidate readings (as stored, llama.cpp's tiled
//! V heads, norms stored as 1+w, decay stored as −exp(A_log)) the one with
//! the highest cosine to the template's own copy wins, and a tensor whose
//! best reading stays below `MIN_COS` stops the import. Routed experts come
//! from the GGUF's banks in bank order with the router rows that score them,
//! so a pruned file (256 of 512 experts) imports as a cut model whose
//! renumbered experts the loader enumerates by presence.

use crate::convert;
use crate::gguf::{Gguf, GgufTensor, dequant, nbytes, parse, resolve_type, v_head_untile, v_head_untile_cols};
use anyhow::{Context, Result, bail, ensure};
use cortiq_core::{CmfModel, TensorDtype, TensorSpec};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// A non-expert tensor whose best reading agrees less than this with the
/// template is a mapping error, not quantization noise (a 2-bit codec still
/// scores above 0.9 against a 4-bit one; a wrong layout scores near 0).
const MIN_COS: f64 = 0.80;

/// The other shards of a split GGUF (`…-00001-of-00002.gguf`).
fn shard_paths(first: &Path, count: usize) -> Result<Vec<PathBuf>> {
    if count <= 1 {
        return Ok(vec![first.to_path_buf()]);
    }
    let name = first.file_name().and_then(|n| n.to_str()).context("shard path")?;
    let tag = format!("-of-{count:05}.gguf");
    let stem = name.strip_suffix(&tag).with_context(|| format!("{name}: a split GGUF names its shards …-00001{tag}"))?;
    let base = &stem[..stem.len().saturating_sub(5)];
    Ok((1..=count).map(|i| first.with_file_name(format!("{base}{i:05}{tag}"))).collect())
}

/// GGUF names of a template tensor. `None` for tensors the importer builds
/// itself (experts, the indexer's fused q/k, the PLE table shards).
fn gguf_name(name: &str) -> Option<String> {
    let direct = match name {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        "model.hyper_connection_mixer.hc_norm.weight" => return Some("output_hc_norm.weight".into()),
        "model.hyper_connection_mixer.input_mix_weight_down.weight" => return Some("output_hc_down.weight".into()),
        "model.hyper_connection_mixer.input_mix_weight_up.weight" => return Some("output_hc_up.weight".into()),
        _ => None::<String>,
    };
    if direct.is_some() {
        return direct;
    }
    let rest = name.strip_prefix("model.layers.")?;
    let (li, rest) = rest.split_once('.')?;
    let suffix = match rest {
        "attn_hyper_connection.block_inject_weight.weight" => "hc_attn_inject.weight",
        "attn_hyper_connection.hc_norm.weight" => "hc_attn_norm.weight",
        "attn_hyper_connection.input_mix_weight_down.weight" => "hc_attn_down.weight",
        "attn_hyper_connection.input_mix_weight_up.weight" => "hc_attn_up.weight",
        "mlp_hyper_connection.block_inject_weight.weight" => "hc_ffn_inject.weight",
        "mlp_hyper_connection.hc_norm.weight" => "hc_ffn_norm.weight",
        "mlp_hyper_connection.input_mix_weight_down.weight" => "hc_ffn_down.weight",
        "mlp_hyper_connection.input_mix_weight_up.weight" => "hc_ffn_up.weight",
        "linear_attn.A_log" => "ssm_a",
        "linear_attn.conv1d.weight" => "ssm_conv1d.weight",
        "linear_attn.dt_bias" => "ssm_dt.bias",
        "linear_attn.in_proj_a.weight" => "ssm_alpha.weight",
        "linear_attn.in_proj_b.weight" => "ssm_beta.weight",
        "linear_attn.in_proj_qkv.weight" => "attn_qkv.weight",
        "linear_attn.in_proj_z.weight" => "attn_gate.weight",
        "linear_attn.norm.weight" => "ssm_norm.weight",
        "linear_attn.out_proj.weight" => "ssm_out.weight",
        "mlp.gate.weight" => "ffn_gate_inp.weight",
        "mlp.shared_expert.down_proj.weight" => "ffn_down_shexp.weight",
        "mlp.shared_expert.gate_proj.weight" => "ffn_gate_shexp.weight",
        "mlp.shared_expert.up_proj.weight" => "ffn_up_shexp.weight",
        "mlp.shared_expert_gate.weight" => "ffn_gate_inp_shexp.weight",
        "ple.conv1d.weight" => "ple_conv1d.weight",
        "ple.key_proj.weight" => "ple_key.weight",
        "ple.value_proj.weight" => "ple_value.weight",
        "ple.norm_conv.weight" => "ple_norm_conv.weight",
        "ple.norm_key.weight" => "ple_norm_key.weight",
        "ple.norm_query.weight" => "ple_norm_query.weight",
        "self_attn.indexer.k_layernorm.weight" => "indexer.k_norm.weight",
        "self_attn.indexer.q_layernorm.weight" => "indexer.q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        _ => return None,
    };
    Some(format!("blk.{li}.{suffix}"))
}

/// "model.layers.L.mlp.experts.E.<proj>.weight" → (L, E, bank suffix).
fn expert_of(name: &str) -> Option<(usize, usize, &'static str)> {
    let t = name.strip_prefix("model.layers.")?;
    let (li, t) = t.split_once('.')?;
    let t = t.strip_prefix("mlp.experts.")?;
    let (e, proj) = t.split_once('.')?;
    let bank = match proj {
        "gate_proj.weight" => "ffn_gate_exps.weight",
        "up_proj.weight" => "ffn_up_exps.weight",
        "down_proj.weight" => "ffn_down_exps.weight",
        _ => return None,
    };
    Some((li.parse().ok()?, e.parse().ok()?, bank))
}

/// "model.layers.L.ple.ple_embedding.ngram_embedding.shard_K.weight" → K.
fn ple_shard(name: &str) -> Option<usize> {
    let (_, t) = name.split_once(".ple.ple_embedding.ngram_embedding.shard_")?;
    t.strip_suffix(".weight")?.parse().ok()
}

struct Shards {
    g: Vec<Gguf>,
    at: HashMap<String, (usize, usize)>,
}

impl Shards {
    fn get(&self, name: &str) -> Option<(&Gguf, &GgufTensor)> {
        let &(s, t) = self.at.get(name)?;
        Some((&self.g[s], &self.g[s].tensors[t]))
    }

    /// Elements `[from, from + n)` of a tensor, dequantized (`from` and `n`
    /// on block boundaries of its type).
    fn read(&self, name: &str, from: usize, n: usize) -> Result<Vec<f32>> {
        let (g, t) = self.get(name).with_context(|| format!("GGUF has no '{name}'"))?;
        let ty = resolve_type(g, t.ggml_type);
        let numel: usize = t.dims.iter().map(|&d| d as usize).product();
        ensure!(from + n <= numel, "{name}: elements {from}+{n} > {numel}");
        let skip = nbytes(ty, from)?;
        let len = nbytes(ty, n)?;
        ensure!(nbytes(ty, from + n)? == skip + len, "{name}: slice {from}+{n} is not on a block boundary");
        let base = g.data_start + t.offset as usize + skip;
        dequant(ty, &g.bytes[base..base + len], n)
    }

    fn numel(&self, name: &str) -> Option<usize> {
        self.get(name).map(|(_, t)| t.dims.iter().map(|&d| d as usize).product())
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    if aa == 0.0 && bb == 0.0 {
        return 1.0;
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

/// GatedDeltaNet head geometry, for llama.cpp's tiled V-head order.
#[derive(Clone, Copy)]
struct Gdn {
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
}

/// The candidate readings of one stored tensor: (label, values).
fn readings(name: &str, vals: Vec<f32>, shape: &[usize], gdn: Gdn) -> Vec<(&'static str, Vec<f32>)> {
    let Gdn { nk, nv, dk, dv } = gdn;
    let hid = *shape.last().unwrap_or(&0);
    let mut out: Vec<(&'static str, Vec<f32>)> = Vec::new();
    let tiled = nk > 0 && nv > nk;
    let untiled = |f: &dyn Fn(&mut Vec<f32>)| {
        let mut v = vals.clone();
        f(&mut v);
        v
    };
    if name.ends_with("linear_attn.in_proj_qkv.weight") && tiled {
        let voff = 2 * nk * dk * hid;
        out.push(("untiled", untiled(&|v| v_head_untile(&mut v[voff..], nv, dv * hid, nk))));
    } else if name.ends_with("linear_attn.in_proj_z.weight") && tiled {
        out.push(("untiled", untiled(&|v| v_head_untile(v, nv, dv * hid, nk))));
    } else if (name.ends_with("linear_attn.in_proj_a.weight") || name.ends_with("linear_attn.in_proj_b.weight")) && tiled {
        out.push(("untiled", untiled(&|v| v_head_untile(v, nv, hid, nk))));
    } else if name.ends_with("linear_attn.dt_bias") && tiled {
        out.push(("untiled", untiled(&|v| v_head_untile(v, nv, 1, nk))));
    } else if name.ends_with("linear_attn.A_log") {
        let ln = |v: &mut Vec<f32>| {
            for x in v.iter_mut() {
                *x = if *x < 0.0 { (-*x).ln() } else { *x };
            }
        };
        out.push(("ln(-a)", untiled(&ln)));
        if tiled {
            out.push(("untiled", untiled(&|v| v_head_untile(v, nv, 1, nk))));
            out.push(("ln(-a) untiled", untiled(&|v| {
                ln(v);
                v_head_untile(v, nv, 1, nk)
            })));
        }
    } else if name.ends_with("linear_attn.conv1d.weight") && tiled {
        let k = vals.len() / (2 * nk * dk + nv * dv).max(1);
        let coff = 2 * nk * dk * k;
        out.push(("untiled", untiled(&|v| v_head_untile(&mut v[coff..], nv, dv * k, nk))));
    } else if name.ends_with("linear_attn.out_proj.weight") && tiled {
        let rows = shape[0];
        out.push(("untiled", untiled(&|v| v_head_untile_cols(v, rows, nv, dv, nk))));
    }
    if name.contains("norm") {
        out.push(("w-1", untiled(&|v| v.iter_mut().for_each(|x| *x -= 1.0))));
    }
    out.insert(0, ("stored", vals));
    out
}

/// Encode `vals` in the template's codec for this tensor.
fn encode(dtype: TensorDtype, vals: &[f32], shape: &[usize]) -> Result<(TensorDtype, Vec<u8>)> {
    let two_d = shape.len() == 2;
    let (rows, cols) = if two_d { (shape[0], shape[1]) } else { (1, vals.len()) };
    Ok(match dtype {
        TensorDtype::F32 => (TensorDtype::F32, vals.iter().flat_map(|v| v.to_le_bytes()).collect()),
        TensorDtype::F16 => (TensorDtype::F16, convert::encode_f16(vals)),
        TensorDtype::Q8_2f if two_d => (TensorDtype::Q8_2f, convert::encode_q8_2f(vals, rows, cols)),
        TensorDtype::Q4TiledP if two_d => convert::quantize_2d(convert::Quant::Q4TiledP, vals, rows, cols),
        TensorDtype::Q2TiledP if two_d => convert::quantize_2d(convert::Quant::Q2TiledP, vals, rows, cols),
        other => bail!("template codec {other:?} for shape {shape:?} is not one the qwen4exp importer writes"),
    })
}

fn quant_dtype(q: convert::Quant) -> Result<TensorDtype> {
    Ok(match q {
        convert::Quant::Q4TiledP => TensorDtype::Q4TiledP,
        convert::Quant::Q2TiledP => TensorDtype::Q2TiledP,
        convert::Quant::Q8_2f => TensorDtype::Q8_2f,
        convert::Quant::F16 => TensorDtype::F16,
        other => bail!("--experts-quant {}: use q4tp, q2tp, q8_2f or f16", convert::quant_name(other)),
    })
}

/// Original expert indices of a pruned bank, from an RCO allocation listing
/// (`blk.L.ffn_gate_exps.weight[E]: TYPE`, an empty type = pruned): for the
/// cosine check of every expert against the template's own copy.
fn read_alloc(path: &str) -> Result<HashMap<usize, Vec<usize>>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    let mut kept: HashMap<usize, Vec<usize>> = HashMap::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("blk.") else { continue };
        let Some((li, rest)) = rest.split_once('.') else { continue };
        let Some(rest) = rest.strip_prefix("ffn_gate_exps.weight[") else { continue };
        let Some((e, ty)) = rest.split_once("]:") else { continue };
        if ty.trim().is_empty() {
            continue;
        }
        let (Ok(li), Ok(e)) = (li.parse::<usize>(), e.parse::<usize>()) else { continue };
        kept.entry(li).or_default().push(e);
    }
    for v in kept.values_mut() {
        v.sort_unstable();
    }
    Ok(kept)
}

pub(crate) fn run(
    first: Gguf,
    first_path: &Path,
    template: &str,
    experts_quant: Option<convert::Quant>,
    alloc: Option<&str>,
    output: &str,
) -> Result<()> {
    let count = first.md.get("split.count").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let paths = shard_paths(first_path, count)?;
    let mut g = vec![first];
    for p in &paths[1..] {
        g.push(parse(p).with_context(|| format!("opening shard {}", p.display()))?);
    }
    let mut at = HashMap::new();
    for (s, gg) in g.iter().enumerate() {
        for (i, t) in gg.tensors.iter().enumerate() {
            at.insert(t.name.clone(), (s, i));
        }
    }
    let src = Shards { g, at };
    let md = &src.g[0].md;
    let mdu = |k: &str| md.get(&format!("qwen4exp.{k}")).and_then(|v| v.as_u64()).map(|v| v as usize);
    let n_exp = mdu("expert_count").context("qwen4exp.expert_count")?;
    let gdn = Gdn {
        nk: mdu("ssm.group_count").unwrap_or(0),
        nv: mdu("ssm.time_step_rank").unwrap_or(0),
        dk: mdu("ssm.state_size").unwrap_or(0),
        dv: mdu("ssm.state_size").unwrap_or(0),
    };

    let tpl = CmfModel::open(template).with_context(|| format!("opening --template {template}"))?;
    ensure!(
        tpl.arch().arch_name == "qwen4_exp",
        "--template {template} is a '{}' file; a qwen4exp GGUF needs a Qwen3.8-Flash-Next CMF",
        tpl.arch().arch_name
    );
    let ple_rows = tpl
        .tensors
        .iter()
        .find(|e| ple_shard(&e.name).is_some())
        .map(|e| e.shape[0])
        .unwrap_or(0);
    let alloc = alloc.map(read_alloc).transpose()?;
    let xq = experts_quant.map(quant_dtype).transpose()?;

    // One job per template tensor that the output carries, in template order.
    struct Job {
        tpl_idx: usize,
        name: String,
        shape: Vec<usize>,
        dtype: TensorDtype,
    }
    let mut jobs: Vec<Job> = Vec::new();
    for (i, e) in tpl.tensors.iter().enumerate() {
        let mut shape = e.shape.to_vec();
        let mut dtype = e.dtype;
        if let Some((_, ex, _)) = expert_of(&e.name) {
            if ex >= n_exp {
                continue;
            }
            dtype = xq.unwrap_or(dtype);
        } else if e.name.ends_with(".mlp.gate.weight") {
            shape[0] = n_exp;
        }
        jobs.push(Job { tpl_idx: i, name: e.name.clone(), shape, dtype });
    }

    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).min(
        // a container's CPU quota, when it has one
        std::fs::read_to_string("/sys/fs/cgroup/cpu.max")
            .ok()
            .and_then(|s| {
                let mut it = s.split_whitespace();
                let q: f64 = it.next()?.parse().ok()?;
                let p: f64 = it.next()?.parse().ok()?;
                Some((q / p).ceil() as usize)
            })
            .unwrap_or(usize::MAX)
            .max(1),
    );
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let out: Mutex<Vec<Option<TensorSpec>>> = Mutex::new((0..jobs.len()).map(|_| None).collect());
    // (category, tensor, reading, cosine) for the report
    let checks: Mutex<Vec<(String, String, &'static str, f64)>> = Mutex::new(Vec::new());
    let fail: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let t0 = std::time::Instant::now();
    eprintln!("qwen4exp import: {} tensors, {n_exp} experts a layer, {threads} threads", jobs.len());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                loop {
                    let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if j >= jobs.len() || fail.lock().unwrap().is_some() {
                        break;
                    }
                    let job = &jobs[j];
                    let r = (|| -> Result<()> {
                        let te = &tpl.tensors[job.tpl_idx];
                        let n: usize = job.shape.iter().product();
                        let tpl_vals = |e: &cortiq_core::TensorEntry| -> Result<Vec<f32>> {
                            let mut v = vec![0f32; e.shape.iter().product()];
                            cortiq_core::quant::dequant_tensor(e, tpl.entry_bytes(e), &mut v).map_err(anyhow::Error::msg)?;
                            Ok(v)
                        };
                        let (vals, cat, reading, cos) = if let Some((li, ex, bank)) = expert_of(&job.name) {
                            let bname = format!("blk.{li}.{bank}");
                            let v = src.read(&bname, ex * n, n)?;
                            // with an allocation listing, the expert against its original
                            let cos = match alloc.as_ref().and_then(|a| a.get(&li)).and_then(|k| k.get(ex)) {
                                Some(&orig) => tpl
                                    .tensor(&job.name.replacen(&format!(".experts.{ex}."), &format!(".experts.{orig}."), 1))
                                    .map(|e| tpl_vals(e).map(|t| cosine(&v, &t)))
                                    .transpose()?,
                                None => None,
                            };
                            (v, "expert".to_string(), "bank", cos)
                        } else if let Some(k) = ple_shard(&job.name) {
                            let cols = job.shape[1];
                            let v = src.read("per_layer_token_embd.weight", k * ple_rows * cols, n)?;
                            let cos = cosine(&v, &tpl_vals(te)?);
                            (v, "ple_table".to_string(), "rows", Some(cos))
                        } else if job.name.ends_with("indexer.index_qk_proj.weight") {
                            let li = job.name.strip_prefix("model.layers.").and_then(|r| r.split('.').next()).context("layer")?;
                            let qn = format!("blk.{li}.indexer.q_proj.weight");
                            let kn = format!("blk.{li}.indexer.k_proj.weight");
                            let q = src.read(&qn, 0, src.numel(&qn).context("indexer q")?)?;
                            let k = src.read(&kn, 0, src.numel(&kn).context("indexer k")?)?;
                            ensure!(q.len() + k.len() == n, "{}: indexer q {} + k {} != {n}", job.name, q.len(), k.len());
                            let t = tpl_vals(te)?;
                            let qk: Vec<f32> = q.iter().chain(&k).copied().collect();
                            let kq: Vec<f32> = k.iter().chain(&q).copied().collect();
                            let (a, b) = (cosine(&qk, &t), cosine(&kq, &t));
                            if a >= b { (qk, "indexer".into(), "q;k", Some(a)) } else { (kq, "indexer".into(), "k;q", Some(b)) }
                        } else {
                            let gname = gguf_name(&job.name).with_context(|| format!("no GGUF source known for '{}'", job.name))?;
                            let have = src.numel(&gname).with_context(|| format!("GGUF has no '{gname}' (for '{}')", job.name))?;
                            ensure!(have == n, "{gname}: {have} elements, the template's '{}' has {n}", job.name);
                            let v = src.read(&gname, 0, n)?;
                            let cat = job.name.rsplit('.').nth(1).unwrap_or("").to_string();
                            if job.name.ends_with(".mlp.gate.weight") && n_exp != te.shape[0] {
                                // a pruned router: its rows score the bank's experts, compared
                                // row by row against the originals when the listing names them
                                let li: usize = job.name.strip_prefix("model.layers.").and_then(|r| r.split('.').next()).and_then(|s| s.parse().ok()).context("layer")?;
                                let cos = match alloc.as_ref().and_then(|a| a.get(&li)) {
                                    Some(orig) if orig.len() == n_exp => {
                                        let t = tpl_vals(te)?;
                                        let h = te.shape[1];
                                        let sel: Vec<f32> = orig.iter().flat_map(|&o| t[o * h..(o + 1) * h].to_vec()).collect();
                                        Some(cosine(&v, &sel))
                                    }
                                    _ => None,
                                };
                                (v, "router".into(), "stored", cos)
                            } else {
                                let t = tpl_vals(te)?;
                                let mut best: Option<(&'static str, Vec<f32>, f64)> = None;
                                for (label, cand) in readings(&job.name, v, &job.shape, gdn) {
                                    let c = cosine(&cand, &t);
                                    if best.as_ref().is_none_or(|b| c > b.2) {
                                        best = Some((label, cand, c));
                                    }
                                }
                                let (label, v, c) = best.context("no reading")?;
                                ensure!(c >= MIN_COS, "'{}' (from {gname}): best reading '{label}' has cos {c:.3} to the template — a layout this importer does not know", job.name);
                                (v, cat, label, Some(c))
                            }
                        };
                        if let Some(c) = cos {
                            checks.lock().unwrap().push((cat, job.name.clone(), reading, c));
                        }
                        let (dt, data) = encode(job.dtype, &vals, &job.shape)?;
                        out.lock().unwrap()[j] = Some(TensorSpec { name: job.name.clone(), dtype: dt, shape: job.shape.clone(), data });
                        Ok(())
                    })();
                    if let Err(e) = r {
                        *fail.lock().unwrap() = Some(e);
                        break;
                    }
                    let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if d % 512 == 0 || d == jobs.len() {
                        let el = t0.elapsed().as_secs_f64();
                        eprint!("\r  {d}/{} tensors · {:.0} s · ~{:.0} s left   ", jobs.len(), el, el / d as f64 * (jobs.len() - d) as f64);
                    }
                }
            });
        }
    });
    eprintln!();
    if let Some(e) = fail.into_inner().unwrap() {
        return Err(e);
    }

    // Agreement with the template, by category: the evidence the layout is right.
    let checks = checks.into_inner().unwrap();
    let mut by: std::collections::BTreeMap<String, (f64, f64, usize, String, HashMap<&'static str, usize>)> = Default::default();
    for (cat, name, reading, c) in &checks {
        let e = by.entry(cat.clone()).or_insert((f64::MAX, 0.0, 0, String::new(), HashMap::new()));
        if *c < e.0 {
            e.0 = *c;
            e.3 = name.clone();
        }
        e.1 += c;
        e.2 += 1;
        *e.4.entry(reading).or_default() += 1;
    }
    eprintln!("agreement with the template (cosine):");
    for (cat, (min, sum, n, worst, rd)) in &by {
        let mut rd: Vec<_> = rd.iter().map(|(k, v)| format!("{k}×{v}")).collect();
        rd.sort();
        eprintln!("  {cat:<28} n={n:<6} mean {:.4}  min {min:.4} ({worst})  [{}]", sum / *n as f64, rd.join(", "));
    }

    let tensors: Vec<TensorSpec> = out.into_inner().unwrap().into_iter().map(|t| t.expect("every job wrote")).collect();
    let mut header = tpl.header.clone();
    header.shard = None;
    let meta = |k: &str| md.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let prov = serde_json::json!({
        "tool": format!("cortiq import-gguf {} (qwen4exp, template)", env!("CARGO_PKG_VERSION")),
        "gguf": first_path.file_name().map(|n| n.to_string_lossy().into_owned()),
        "gguf_name": meta("general.name"),
        "gguf_repo": meta("general.repo_url"),
        "experts_per_layer": n_exp,
        "template": Path::new(template).file_name().map(|n| n.to_string_lossy().into_owned()),
    });
    match header.provenance.as_mut() {
        Some(serde_json::Value::Object(m)) => {
            m.insert("gguf_import".into(), prov);
        }
        _ => header.provenance = Some(serde_json::json!({ "gguf_import": prov })),
    }
    CmfModel::write(output, &header, &tensors, None, tpl.vocab.as_deref())?;
    let sz = std::fs::metadata(output)?.len() as f64 / 1e9;
    eprintln!("qwen4exp import: {} tensors -> {output} ({sz:.2} GB) in {:.0} s", tensors.len(), t0.elapsed().as_secs_f64());
    Ok(())
}
