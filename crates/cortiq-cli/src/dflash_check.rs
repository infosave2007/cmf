//! `cortiq dflash-check`: the engine's DFlash drafter against a reference kit.
//!
//! The kit (`scripts/dory/q27_dflash_kit.py`) holds, for a few windows of
//! text, the target's tapped layer outputs and what the published drafter
//! made of them: layer 0's context keys and values, a block's rows after the
//! last norm for a few anchors, and the tokens drafted at every anchor. The
//! reference runs unrounded weights in bf16, the engine a quantized sidecar
//! in f32, so rows are compared by cosine, and tokens by what a round would
//! keep of them: the drafts in a row that are the target's own next tokens.

use anyhow::{Context, Result, anyhow, ensure};
use cortiq_core::CmfModel;
use cortiq_engine::dflash::Drafter;
use cortiq_engine::pool::Pool;
use cortiq_engine::qtensor::QTensor;
use std::path::Path;
use std::sync::Arc;

fn raw<T, const N: usize>(dir: &Path, name: &str, f: fn([u8; N]) -> T) -> Result<Vec<T>> {
    let p = dir.join(name);
    let b = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
    ensure!(b.len() % N == 0, "{}: {} bytes is not a whole number of values", p.display(), b.len());
    Ok(b.chunks_exact(N).map(|c| f(c.try_into().unwrap())).collect())
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += *x as f64 * *y as f64;
        aa += *x as f64 * *x as f64;
        bb += *y as f64 * *y as f64;
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

/// The drafts a round keeps: the leading ones the target would have chosen.
fn kept(drafts: &[u32], own: &[u32], anchor: usize, len: usize) -> usize {
    drafts.iter().enumerate().take_while(|&(j, d)| anchor + j < len && *d == own[anchor + j]).count()
}

pub fn run(model: &str, kit: &str, host: bool, limit: usize, head_rows: usize) -> Result<()> {
    let (model, kit) = (Path::new(model), Path::new(kit));
    let base = Arc::new(CmfModel::open(model).map_err(|e| anyhow!("{}: {e}", model.display()))?);
    let mut d = Drafter::beside(model).ok_or_else(|| anyhow!("{} has no DFlash sidecar beside it", model.display()))?.map_err(|e| anyhow!(e))?;
    if host {
        d.host_only();
    }
    let embed = QTensor::from_model(&base, "model.embed_tokens.weight").map_err(|e| anyhow!(e))?;
    let head = QTensor::from_model(&base, "lm_head.weight").map_err(|e| anyhow!(e))?;
    let pool = Pool::new(std::thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(1).max(1));
    let pool = Some(&pool);
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(kit.join("kit.json")).with_context(|| format!("reading {}/kit.json", kit.display()))?)?;
    let int = |k: &str| meta[k].as_u64().map(|v| v as usize).ok_or_else(|| anyhow!("kit.json has no {k}"));
    let g = d.g.clone();
    let (t, hs, n_starts, rows) = (int("window")?, int("hidden")?, int("starts")?, int("rows")?);
    let layers: Vec<usize> = meta["layers"].as_array().ok_or_else(|| anyhow!("kit.json has no layers"))?.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).collect();
    ensure!(layers == g.taps && hs == g.hidden && int("block")? == g.block, "the kit is of another drafter: layers {layers:?}, hidden {hs}");
    let (nl, hd, kv, drafts) = (layers.len(), g.head_dim, g.kv_heads, g.block - 1);
    let (mut worst, mut agree, mut total) = (1f64, 0usize, 0usize);
    // drafts kept: the engine's, its rows' own top tokens, the kit's, the kit's rows' own; and the rounds
    let (mut keep, mut rounds, mut block_ms, mut head_ms) = ([0usize; 4], 0usize, 0f64, 0f64);
    for w in meta["windows"].as_array().ok_or_else(|| anyhow!("kit.json has no windows"))?.iter().filter_map(|v| v.as_u64()) {
        let f = |s: &str| format!("w{w}.{s}");
        let taps = raw(kit, &f("taps.f16"), u16::from_le_bytes)?;
        ensure!(taps.len() == nl * t * hs, "window {w}: taps of {} values, expected {}", taps.len(), nl * t * hs);
        let (ids, own) = (raw(kit, &f("ids.u32"), u32::from_le_bytes)?, raw(kit, &f("own.u32"), u32::from_le_bytes)?);
        let len = meta["length"][w.to_string()].as_u64().map_or(t, |v| v as usize);
        let mut starts = raw(kit, &f("starts.u32"), u32::from_le_bytes)?;
        if limit > 0 {
            starts.truncate(limit.max(n_starts));
        }
        let (want_h, want_d, want_p) = (raw(kit, &f("hidden.f32"), f32::from_le_bytes)?, raw(kit, &f("drafts.u32"), u32::from_le_bytes)?, raw(kit, &f("plain.u32"), u32::from_le_bytes)?);
        let (key0, value0) = (raw(kit, &f("key0.f32"), f32::from_le_bytes)?, raw(kit, &f("value0.f32"), f32::from_le_bytes)?);
        ensure!(starts.len() >= n_starts && want_h.len() == n_starts * g.block * hs && want_d.len() >= starts.len() * drafts && want_p.len() == want_d.len() && key0.len() == kv * rows * hd, "window {w}: the kit's files do not match kit.json");
        // the target's rows up to the last anchor, a row's layers joined
        let need = rows.max(*starts.iter().max().unwrap() as usize + 1).min(t);
        let mut joined = vec![0f32; need * nl * hs];
        for r in 0..need {
            for l in 0..nl {
                let src = &taps[(l * t + r) * hs..][..hs];
                for (o, v) in joined[(r * nl + l) * hs..][..hs].iter_mut().zip(src) {
                    *o = cortiq_core::quant::f16_to_f32(*v);
                }
            }
        }
        d.reset();
        let t0 = std::time::Instant::now();
        d.push_context(0, &joined, pool);
        let ctx_ms = t0.elapsed().as_secs_f64() * 1e3 / need as f64;
        let (mut ck, mut cv) = (1f64, 1f64);
        for h in 0..kv {
            let (k, v) = d.context(0, h);
            for r in 0..rows {
                ck = ck.min(cos(&k[r * hd..][..hd], &key0[(h * rows + r) * hd..][..hd]));
                cv = cv.min(cos(&v[r * hd..][..hd], &value0[(h * rows + r) * hd..][..hd]));
            }
        }
        println!("window {w}: {need} context rows, {ctx_ms:.1} ms a row; layer 0's keys and values against the kit, lowest cosine of {} rows: {ck:.5} / {cv:.5}", kv * rows);
        worst = worst.min(ck).min(cv);
        for (si, &s) in starts.iter().enumerate() {
            let a = s as usize + 1;
            let mut h = vec![0f32; g.block * hs];
            embed.row_f32(ids[a] as usize, &mut h[..hs]);
            let (first, rest) = h.split_at_mut(hs);
            let _ = first;
            embed.row_f32(g.mask_token as usize, &mut rest[..hs]);
            for r in 2..g.block {
                h.copy_within(hs..2 * hs, r * hs);
            }
            let t0 = std::time::Instant::now();
            let out = d.block(a, h, pool);
            let t1 = std::time::Instant::now();
            // the head's first rows only (a shortlist of the vocabulary), or all of it
            let width = if head_rows == 0 { g.vocab } else { head_rows.min(g.vocab) };
            let mut logits = vec![0f32; drafts * width];
            let narrow = !host
                && head.mapped_q4tp().is_some_and(|(m, i)| {
                    cortiq_engine::gpu::q4tp_rows8_many(m, &[(i, g.vocab, hs, width)], &out[hs..], drafts, &mut [&mut logits[..]])
                });
            if !narrow {
                logits = vec![0f32; drafts * g.vocab];
                if host || !head.device_matmat(&out[hs..], drafts, &mut logits) {
                    head.matmat(&out[hs..], drafts, &mut logits, pool);
                }
            }
            let (chosen, plain) = d.select(&out[hs..], &logits, ids[a]);
            // the first round of a window pays for loading
            if si > 0 {
                block_ms += (t1 - t0).as_secs_f64() * 1e3;
                head_ms += t1.elapsed().as_secs_f64() * 1e3;
                rounds += 1;
            }
            let (wd, wp) = (&want_d[si * drafts..][..drafts], &want_p[si * drafts..][..drafts]);
            for (k, v) in keep.iter_mut().zip([&chosen[..], &plain[..], wd, wp]) {
                *k += kept(v, &own, a, len);
            }
            agree += chosen.iter().zip(wd).filter(|(a, b)| a == b).count();
            total += drafts;
            if si < n_starts {
                let low = (0..g.block).map(|r| cos(&out[r * hs..][..hs], &want_h[(si * g.block + r) * hs..][..hs])).fold(1f64, f64::min);
                println!("  anchor {a:4}: rows' lowest cosine {low:.5}; drafts {chosen:?}, the kit's {wd:?}");
                worst = worst.min(low);
            }
        }
    }
    let n = (total / drafts).max(1) as f64;
    println!("lowest cosine {worst:.5}; drafted tokens as the kit's {agree}/{total}");
    println!(
        "drafts a round keeps, every start alike, of {} starts: the engine {:.3} (its rows' own top tokens {:.3}); the kit {:.3} ({:.3})",
        total / drafts,
        keep[0] as f64 / n,
        keep[1] as f64 / n,
        keep[2] as f64 / n,
        keep[3] as f64 / n
    );
    println!("a round: block {:.1} ms, head and selector {:.1} ms", block_ms / rounds.max(1) as f64, head_ms / rounds.max(1) as f64);
    if let Some(line) = cortiq_engine::dflash::stage_report() {
        println!("{line}");
    }
    Ok(())
}
