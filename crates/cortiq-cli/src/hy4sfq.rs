//! `cortiq hy4-sfq`: SFq for the routed experts of a Hy4-flash file — GPTQ
//! on calibration rows, a per-channel scale of the experts' input and the
//! code refinement (`docs/Q2TP_SCALE_REFINE.md`), written in place into an
//! existing file of the same model.
//!
//! * `--capture-only`: the calibration forward. The calibration file (the
//!   q4tp one, its skeleton at 8 bits) runs every window through the card's
//!   prompt graph (CUDA hybrid) with the capture armed: per routed FFN the
//!   latent input rows (`latent_down · x`, the experts' own input) and the
//!   router's picks. Written to `--capture DIR` (`L{layer}.x` f16 rows,
//!   `.picks` u16, `.w` f32 mixing weights, `meta.json`).
//! * Without it: the encode. Per routed FFN, from the captured rows and the
//!   full-precision experts of `--hf` (the FP8 release, e4m3 with 128×128
//!   block scales, decoded exactly):
//!   1. the channel scale `t_j = σ_j^α / rms_j^(1−α)` over the latent
//!      channels (σ_j the RMS of column j over every expert's gate and up
//!      rows, rms_j the captured input's), unit geometric mean — the experts'
//!      gate/up columns are divided by `t`, the captured rows multiplied by
//!      it, and `latent_down`'s rows multiplied by it in the output, so the
//!      model is the same function before quantization;
//!   2. per expert GPTQ (`quantize-moe`'s encoder, on the wgpu device or the
//!      host): gate/up q2tp (placed rule) with the code refinement and the
//!      output debias, down q4tp (refined with `--refine-q4tp`), the SwiGLU
//!      clamp of the forward; experts routed fewer than `--min-rows` tokens
//!      take the converter's rounding of the scaled weights;
//!   3. the layer's experts and its rescaled `latent_down` recoded in place
//!      into `--target` (slots of the same layout and size).

use crate::convert::{Quant, quantize_2d};
use crate::gptq::{CalibSource, calib_source_tokens, chscale_vector, split_calib_budget};
use crate::hy4pack::Sources;
use crate::moegptq::{Codec, EncodeCfg, Engine, ExpertW, LayerJob, LayerRows, Part, RouteW, dequant_expert, encode_layer, expert_out, gather_rows, rtn_expert};
use cortiq_core::format::CmfModel;
use cortiq_core::quant::{f16_to_f32, f32_to_f16};
use cortiq_core::types::TensorDtype;
use std::path::Path;
use std::time::Instant;

pub struct Hy4SfqOpts<'a> {
    pub calib_model: &'a str,
    pub capture: &'a str,
    pub capture_only: bool,
    pub calib: &'a [String],
    pub tokens: usize,
    pub window: usize,
    pub hf: &'a str,
    pub bf16: Option<&'a str>,
    pub target: &'a str,
    pub alpha: f64,
    pub refine: usize,
    pub ridge: f64,
    pub refine_q4tp: bool,
    pub lambda: f64,
    pub min_rows: usize,
    pub threads: usize,
    pub layers: Option<(usize, usize)>,
}

fn write_raw(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let tmp = path.with_extension("part");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn le_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn run(o: &Hy4SfqOpts) -> anyhow::Result<()> {
    if o.capture_only {
        capture(o)
    } else {
        encode(o)
    }
}

fn capture(o: &Hy4SfqOpts) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let cmodel = std::sync::Arc::new(CmfModel::open_sharded(o.calib_model)?);
    anyhow::ensure!(cmodel.arch().arch_name == crate::hy4pack::ARCH_NAME, "{} is not a Hy4-flash file", o.calib_model);
    let mut p = cortiq_engine::Pipeline::from_model(&cmodel, cortiq_engine::SamplerConfig::default()).map_err(|e| anyhow::anyhow!("{e}"))?;
    p.set_confidence(false);
    let srcs: Vec<CalibSource> = o.calib.iter().map(|s| CalibSource::parse(s)).collect::<anyhow::Result<_>>()?;
    let budgets = split_calib_budget(&srcs, o.tokens);
    let mut windows: Vec<Vec<u32>> = Vec::new();
    for (src, &n) in srcs.iter().zip(&budgets) {
        let ids = calib_source_tokens(&src.path, n, &|t| p.tokenizer.encode(t))?;
        eprintln!("  {}: {} tokens", src.path, ids.len());
        windows.extend(ids.chunks(o.window.max(16)).filter(|w| w.len() >= 16).map(<[u32]>::to_vec));
    }
    let ntok: usize = windows.iter().map(Vec::len).sum();
    eprintln!("hy4-sfq capture: {ntok} tokens in {} windows of {}", windows.len(), o.window);
    let b = p.hy4.as_mut().ok_or_else(|| anyhow::anyhow!("the pipeline did not load the Hy4-flash stack"))?;
    let (g, l, c, st) = (&b.0, &b.1, &b.2, &mut b.3);
    let dir = Path::new(o.capture);
    std::fs::create_dir_all(dir)?;
    // appended window by window (the rows of a full capture outgrow a container)
    let mut files: std::collections::BTreeMap<usize, [std::io::BufWriter<std::fs::File>; 3]> = std::collections::BTreeMap::new();
    let mut top_k = 0usize;
    let mut sink = |cl: &cortiq_engine::hy4::CapLayer| -> Result<(), String> {
        use std::io::Write;
        let n = cl.x.len() / c.latent.max(1);
        top_k = cl.picks.len() / n.max(1);
        if !files.contains_key(&cl.li) {
            let open = |ext: &str| -> Result<std::io::BufWriter<std::fs::File>, String> {
                std::fs::File::create(dir.join(format!("L{}.{ext}.part", cl.li))).map(|f| std::io::BufWriter::with_capacity(1 << 22, f)).map_err(|e| e.to_string())
            };
            files.insert(cl.li, [open("x")?, open("picks")?, open("w")?]);
        }
        let f = files.get_mut(&cl.li).unwrap();
        f[0].write_all(&le_u16(&cl.x)).map_err(|e| e.to_string())?;
        f[1].write_all(&le_u16(&cl.picks)).map_err(|e| e.to_string())?;
        f[2].write_all(&le_f32(&cl.w)).map_err(|e| e.to_string())?;
        Ok(())
    };
    let ntok = cortiq_engine::hy4::capture(g, l, c, st, &windows, &mut sink).map_err(|e| anyhow::anyhow!("capture: {e}"))?;
    let mut lis = Vec::new();
    for (li, fs) in files {
        for (f, ext) in fs.into_iter().zip(["x", "picks", "w"]) {
            f.into_inner().map_err(|e| anyhow::anyhow!("{e}"))?.sync_all()?;
            std::fs::rename(dir.join(format!("L{li}.{ext}.part")), dir.join(format!("L{li}.{ext}")))?;
        }
        lis.push(li);
    }
    let meta = serde_json::json!({
        "model": o.calib_model,
        "tokens": ntok,
        "windows": windows.len(),
        "window": o.window,
        "calib": o.calib,
        "layers": lis,
        "top_k": top_k,
    });
    write_raw(&dir.join("meta.json"), serde_json::to_string_pretty(&meta)?.as_bytes())?;
    eprintln!("hy4-sfq capture: {} routed FFNs, {ntok} tokens into {} in {:.0} s", lis.len(), dir.display(), t0.elapsed().as_secs_f64());
    Ok(())
}

fn read_u16(path: &Path) -> anyhow::Result<Vec<u16>> {
    let b = std::fs::read(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

fn read_f32(path: &Path) -> anyhow::Result<Vec<f32>> {
    let b = std::fs::read(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn encode(o: &Hy4SfqOpts) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let dir = Path::new(o.capture);
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let ntok = meta["tokens"].as_u64().unwrap_or(0) as usize;
    let top_k = meta["top_k"].as_u64().unwrap_or(0) as usize;
    let lis: Vec<usize> = meta["layers"].as_array().map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect()).unwrap_or_default();
    anyhow::ensure!(ntok > 0 && top_k > 0 && !lis.is_empty(), "{}: an empty capture", dir.display());
    let target = CmfModel::open(o.target)?;
    anyhow::ensure!(target.arch().arch_name == crate::hy4pack::ARCH_NAME, "{} is not a Hy4-flash file", o.target);
    let prov = target.header.provenance.as_ref().and_then(|p| p.get("hy4_flash")).ok_or_else(|| anyhow::anyhow!("no hy4_flash record"))?;
    let tc = &prov["text_config"];
    let limit = tc["swiglu_limit"].as_f64().unwrap_or(0.0) as f32;
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(Path::new(o.hf).join("config.json"))?)?;
    let block = config
        .get("quantization_config")
        .and_then(|q| q.get("weight_block_size"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_u64())
        .map_or(128, |v| v as usize);
    let mut dirs: Vec<&Path> = vec![Path::new(o.hf)];
    if let Some(b) = o.bf16 {
        dirs.push(Path::new(b));
    }
    let src = Sources::open(&dirs)?;
    // the device engine (wgpu), as quantize-moe's
    let engine = if std::env::var("CMF_MOE_HOST").is_ok_and(|v| v != "0") {
        Engine::Host
    } else {
        match cortiq_engine::gpu_wgpu::train_ops::dev().filter(|_| cortiq_engine::gpu_wgpu::selected()) {
            Some(c) => Engine::Gpu(c),
            None => anyhow::bail!("hy4-sfq: no wgpu GPU for the encode (CMF_GPU=wgpu; CMF_MOE_HOST=1 runs on the host)"),
        }
    };
    let refine = (o.refine > 0).then_some((o.refine, o.ridge, o.refine_q4tp));
    let ec = EncodeCfg { codec: Codec::Q2tp, limit, lambda: o.lambda, threads: o.threads.max(1), refine };
    let (lo, hi) = o.layers.unwrap_or((0, usize::MAX));
    // CMF_HY4_SFQ_DRY=1: encode and time, write nothing (any file of the model as --target)
    let dry = std::env::var("CMF_HY4_SFQ_DRY").as_deref() == Ok("1");
    eprintln!(
        "hy4-sfq encode: {} routed FFNs, {ntok} calibration tokens, top-{top_k}, α {} (0 = no channel scale), refine {:?}, clamp {limit}",
        lis.len(),
        o.alpha,
        refine
    );
    for &li in lis.iter().filter(|&&li| li >= lo && li < hi) {
        let tl = Instant::now();
        let m = format!("model.layers.{li}.mlp.");
        let gu = format!("{m}experts.gate_up_proj");
        let dn = format!("{m}experts.down_proj");
        let mut x = read_u16(&dir.join(format!("L{li}.x")))?;
        let picks = read_u16(&dir.join(format!("L{li}.picks")))?;
        let w = read_f32(&dir.join(format!("L{li}.w")))?;
        let dim = x.len() / ntok;
        anyhow::ensure!(dim * ntok == x.len() && picks.len() == ntok * top_k, "layer {li}: capture sizes");
        let ne = target.tensors.iter().filter(|t| t.name.starts_with(&format!("{m}experts.")) && t.name.ends_with(".gate_proj.weight")).count();
        let first = src.bank_slice(&gu, 0, block)?;
        let inter = first.1 / 2;
        anyhow::ensure!(first.2 == dim, "layer {li}: experts read {} channels, the capture has {dim}", first.2);
        // 1. the channel scale over the latent channels
        let t: Vec<f32> = if o.alpha > 0.0 {
            let mut hdiag = vec![0f64; dim];
            for row in x.chunks_exact(dim) {
                for (h, &v) in hdiag.iter_mut().zip(row) {
                    let f = f16_to_f32(v) as f64;
                    *h += f * f;
                }
            }
            for h in &mut hdiag {
                *h /= ntok as f64;
            }
            // every expert's gate/up decoded once, in parallel (4.8 GB of e4m3 a layer)
            let nt = o.threads.max(1);
            let parts: Vec<anyhow::Result<Vec<f64>>> = std::thread::scope(|sc| {
                let hs: Vec<_> = (0..nt)
                    .map(|k| {
                        let (src, gu) = (&src, &gu);
                        sc.spawn(move || -> anyhow::Result<Vec<f64>> {
                            let mut cs = vec![0f64; dim];
                            for e in (k..ne).step_by(nt) {
                                let (v, _, _) = src.bank_slice(gu, e, block)?;
                                for row in v.chunks_exact(dim) {
                                    for (c, &a) in cs.iter_mut().zip(row) {
                                        *c += (a as f64) * (a as f64);
                                    }
                                }
                            }
                            Ok(cs)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("a column-energy worker panicked")))).collect()
            });
            let mut colsq = vec![0f64; dim];
            for p in parts {
                for (c, v) in colsq.iter_mut().zip(p?) {
                    *c += v;
                }
            }
            chscale_vector(&hdiag, &colsq, ne * 2 * inter, o.alpha)
        } else {
            vec![1.0; dim]
        };
        if o.alpha > 0.0 {
            for row in x.chunks_exact_mut(dim) {
                for (v, &s) in row.iter_mut().zip(&t) {
                    *v = f32_to_f16(f16_to_f32(*v) * s);
                }
            }
        }
        let (tmin, tmax) = t.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        // 2. the experts' GPTQ on the scaled rows and weights
        let lr = LayerRows { x, picks, w };
        let get = |part: Part, e: usize| -> anyhow::Result<ExpertW> {
            anyhow::ensure!(matches!(part, Part::Routed), "hy4-sfq encodes the routed experts only");
            let (v, rows, cols) = src.bank_slice(&gu, e, block)?;
            anyhow::ensure!(rows == 2 * inter && cols == dim, "expert {e}: gate_up {rows}x{cols}");
            let mut g = v[..inter * dim].to_vec();
            let mut u = v[inter * dim..].to_vec();
            for row in g.chunks_exact_mut(dim).chain(u.chunks_exact_mut(dim)) {
                for (a, &s) in row.iter_mut().zip(&t) {
                    *a /= s;
                }
            }
            let (d, drows, dcols) = src.bank_slice(&dn, e, block)?;
            anyhow::ensure!(drows == dim && dcols == inter, "expert {e}: down {drows}x{dcols}");
            Ok(ExpertW { g, u, d })
        };
        let held = vec![false; ntok];
        let mean = (ntok * top_k) as f64 / ne as f64;
        let prior_n0 = std::env::var("CMF_MOE_PRIOR").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(if mean < dim as f64 { dim as f64 } else { 0.0 });
        let job = LayerJob { li, hidden: dim, ne, inter, top_k, min_rows: o.min_rows, prior_n0, prior_s: 8192, held: &held, route_w: RouteW::Off };
        let out = encode_layer(engine, &ec, &job, &lr, true, false, None, None, &get)?;
        anyhow::ensure!(out.routed.len() == ne, "layer {li}: {} experts encoded of {ne}", out.routed.len());
        if dry {
            // the output error of the first experts with rows, on their own rows: RTN against this encode
            let mut lists: Vec<Vec<u32>> = vec![Vec::new(); ne];
            for (j, &e) in lr.picks.iter().enumerate() {
                if (e as usize) < ne {
                    lists[e as usize].push((j / top_k) as u32);
                }
            }
            let (mut base, mut e_rtn, mut e_enc, mut cnt) = (0f64, 0f64, 0f64, 0usize);
            for e in (0..ne).filter(|&e| lists[e].len() >= o.min_rows).take(16) {
                let w = get(Part::Routed, e)?;
                let xs = gather_rows(&lr.x, dim, &lists[e], 1.0, o.threads);
                let m = lists[e].len();
                let ofp = expert_out(&xs, m, dim, &w.g, &w.u, &w.d, inter, limit, o.threads);
                let err = |enc: &[Vec<u8>; 3]| -> f64 {
                    let [qg, qu, qd] = dequant_expert(enc, inter, dim, Codec::Q2tp);
                    let oq = expert_out(&xs, m, dim, &qg, &qu, &qd, inter, limit, o.threads);
                    oq.iter().zip(&ofp).map(|(a, b)| ((a - b) as f64).powi(2)).sum()
                };
                base += ofp.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
                e_rtn += err(&rtn_expert(Codec::Q2tp, &w, inter, dim));
                e_enc += err(&out.routed[e]);
                cnt += 1;
            }
            let hash = out.routed.iter().flatten().fold(0u64, |h, b| cortiq_core::hash::hash64(b) ^ h.rotate_left(7));
            eprintln!("  layer {li}: {cnt} experts' rel. output error: RTN {:.5}, this encode {:.5} ({:+.1} %); payload hash {hash:016x}", e_rtn / base, e_enc / base, (e_enc / e_rtn - 1.0) * 100.0);
            eprintln!(
                "  layer {li} (dry run, nothing written): {} GPTQ + {} rounded experts, t in [{tmin:.3}, {tmax:.3}], {:.0} s; stages (summed over workers) inputs {:.1} fold {:.1} sweeps+refine {:.1} down fold {:.1} down sweep+refine {:.1} s",
                out.n_gptq,
                out.n_rtn,
                tl.elapsed().as_secs_f64(),
                out.tim[0],
                out.tim[1],
                out.tim[2],
                out.tim[3],
                out.tim[4]
            );
            continue;
        }
        // 3. in place: the experts, then latent_down with its rows scaled
        let mut patches: Vec<(usize, TensorDtype, Vec<u8>)> = Vec::with_capacity(3 * ne + 1);
        let idx = |name: &str| -> anyhow::Result<usize> { target.tensors.iter().position(|t| t.name == name).ok_or_else(|| anyhow::anyhow!("{} has no {name}", o.target)) };
        for (e, enc) in out.routed.into_iter().enumerate() {
            for (k, (part, dt)) in [("gate_proj", TensorDtype::Q2TiledP), ("up_proj", TensorDtype::Q2TiledP), ("down_proj", TensorDtype::Q4TiledP)].into_iter().enumerate() {
                let name = format!("{m}experts.{e}.{part}.weight");
                let i = idx(&name)?;
                let te = &target.tensors[i];
                anyhow::ensure!(te.dtype == dt && te.nbytes as usize == enc[k].len(), "{name}: slot {:?} {} B, payload {dt:?} {} B", te.dtype, te.nbytes, enc[k].len());
                patches.push((i, dt, enc[k].clone()));
            }
        }
        if o.alpha > 0.0 {
            let name = format!("{m}latent_down.weight");
            let i = idx(&name)?;
            let te = &target.tensors[i];
            let (mut v, sh) = src.matrix(&format!("{m}routed_expert_down_proj.weight"), block)?;
            anyhow::ensure!(sh == vec![dim, sh[1]], "{name}: source {:?}", sh);
            for (r, row) in v.chunks_exact_mut(sh[1]).enumerate() {
                for a in row.iter_mut() {
                    *a *= t[r];
                }
            }
            let q = match te.dtype {
                TensorDtype::Q4TiledP => Quant::Q4TiledP,
                TensorDtype::Q8_2f => Quant::Q8_2f,
                TensorDtype::Q2TiledP => Quant::Q2TiledP,
                d => anyhow::bail!("{name}: a {d:?} slot"),
            };
            let (dt, bytes) = quantize_2d(q, &v, sh[0], sh[1]);
            anyhow::ensure!(dt == te.dtype && bytes.len() as u64 == te.nbytes, "{name}: {dt:?} {} B for a {:?} {} B slot", bytes.len(), te.dtype, te.nbytes);
            patches.push((i, dt, bytes));
        }
        CmfModel::recode_entries_in_place(o.target, &patches).map_err(|e| anyhow::anyhow!("{}: {e}", o.target))?;
        eprintln!(
            "  layer {li}: {} GPTQ + {} rounded experts, t in [{tmin:.3}, {tmax:.3}], {:.0} s (total {:.0} s)",
            out.n_gptq,
            out.n_rtn,
            tl.elapsed().as_secs_f64(),
            t0.elapsed().as_secs_f64()
        );
    }
    eprintln!("hy4-sfq: {} done in {:.0} s", o.target, t0.elapsed().as_secs_f64());
    Ok(())
}
