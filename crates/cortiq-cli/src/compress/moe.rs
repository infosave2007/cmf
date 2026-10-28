//! Mixture-of-Experts models: the experts the task does not route to leave
//! the file. S1 counts how often the router picks each expert on the
//! teacher's answers (the counters `CMF_MOE_STATS` dumps), S2 finds the
//! routing-mass cover whose kept experts meet the size target, S3 writes
//! only those experts (`moe-defrag`: kept experts renumbered, router rows
//! sliced — the runtime then routes among them exactly as before).

use super::data::Data;
use super::stages::{open, path_of};
use super::Run;
use crate::tdata::TdataSplit;
use anyhow::{Context, Result};
use cortiq_core::CmfModel;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

/// "model.layers.L.mlp.experts.E.<rest>" → (L, E).
fn expert_of(name: &str) -> Option<(usize, usize)> {
    let t = name.strip_prefix("model.layers.")?;
    let (li, t) = t.split_once('.')?;
    let t = t.strip_prefix("mlp.experts.")?;
    let (e, _) = t.split_once('.')?;
    Some((li.parse().ok()?, e.parse().ok()?))
}

/// Expert count and parameters per expert, by layer.
pub fn layout(m: &CmfModel) -> Result<Value> {
    let moe = m.arch().moe.as_ref().context("not a MoE model")?;
    let mut per: BTreeMap<usize, u64> = BTreeMap::new();
    for e in &m.tensors {
        if let Some((l, 0)) = expert_of(&e.name) {
            *per.entry(l).or_default() += e.n_elems() as u64;
        }
    }
    anyhow::ensure!(!per.is_empty(), "no routed-expert tensors (model.layers.L.mlp.experts.E.*) — this MoE layout is not supported yet");
    Ok(json!({
        "experts": moe.num_experts,
        "top_k": moe.top_k,
        "per_expert": per.iter().map(|(l, n)| (l.to_string(), json!(n))).collect::<serde_json::Map<_, _>>(),
    }))
}

/// S1 for MoE: routing counts over the train records.
pub fn s1_routing(run: &Run) -> Result<()> {
    let s0 = run.require(0)?;
    let q8 = path_of(&s0, "q8").context("S0 output")?;
    let data = Data::load(&run.a.data)?;
    let budget = match run.a.illum_tokens {
        Some(0) => None,
        Some(n) => Some(n),
        None => Some(32768),
    };
    let mut recs: Vec<&super::data::Rec> = data.split(TdataSplit::Train).collect();
    recs.sort_by_cached_key(|r| crate::knowledge::sha256_hex(r.rid.as_bytes()));
    let (windows, total) = super::illum::windows(recs.into_iter(), run.a.window, budget);
    let m = open(&q8)?;
    let mut pipe = cortiq_engine::Pipeline::from_model(&m, cortiq_engine::SamplerConfig::default()).map_err(|e| anyhow::anyhow!("{e}"))?;
    pipe.set_confidence(false);
    run.log(&format!("S1: routing counts over {} train windows / {total} tokens", windows.len()));
    let t0 = std::time::Instant::now();
    let mut done = 0usize;
    for (i, w) in windows.iter().enumerate() {
        pipe.nll_ids_from(w, 0).map_err(|e| anyhow::anyhow!("window {i}: {e}"))?;
        done += w.len();
        let el = t0.elapsed().as_secs_f64();
        eprint!(
            "\r  routing: {done}/{total} tokens · {:.0} tok/s · ~{} left   ",
            done as f64 / el.max(1e-9),
            crate::fmt_eta((total - done) as f64 * el / done.max(1) as f64)
        );
    }
    eprintln!();
    let mut counts = serde_json::Map::new();
    for (li, lw) in pipe.weights.layers.iter().enumerate() {
        if let cortiq_engine::pipeline::FfnKind::Moe(moe) = &lw.ffn {
            counts.insert(li.to_string(), json!(moe.stats.borrow().clone()));
        }
    }
    anyhow::ensure!(!counts.is_empty(), "the forward recorded no routing (an expert stack the generic counters do not see)");
    let dir = run.stage_dir(1);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("routing.json");
    std::fs::write(&path, serde_json::to_vec(&Value::Object(counts))?)?;
    run.mark_done(1, json!({ "routing": path, "tokens": total }))
}

/// Experts kept per layer at routing-mass `cover` (most used first).
fn kept_at(counts: &[u64], cover: f64) -> usize {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return counts.len();
    }
    let mut c = counts.to_vec();
    c.sort_unstable_by(|a, b| b.cmp(a));
    let mut acc = 0u64;
    for (i, &x) in c.iter().enumerate() {
        acc += x;
        if acc as f64 >= cover * total as f64 {
            return i + 1;
        }
    }
    counts.len()
}

/// S2 for MoE: the cover that meets the size target.
pub fn s2_plan(run: &Run) -> Result<()> {
    let a = &run.a;
    let s0 = run.require(0)?;
    let s1 = run.require(1)?;
    let counts: BTreeMap<String, Vec<u64>> = serde_json::from_slice(&std::fs::read(path_of(&s1, "routing").context("S1 output")?)?)?;
    let params = s0["params"].as_u64().unwrap_or(0);
    let per: BTreeMap<String, u64> = serde_json::from_value(s0["moe"]["per_expert"].clone())?;
    let experts_params: u64 = counts.iter().map(|(l, c)| c.len() as u64 * per.get(l).copied().unwrap_or(0)).sum();
    let after = |cover: f64| -> u64 {
        params - counts.iter().map(|(l, c)| (c.len() - kept_at(c, cover)) as u64 * per.get(l).copied().unwrap_or(0)).sum::<u64>()
    };
    let target = if let Some(r) = a.ratio {
        anyhow::ensure!(r > 0.0 && r <= 1.0, "--ratio must be in (0, 1]");
        (params as f64 * r) as u64
    } else if let Some(sz) = &a.size {
        super::plan::parse_size(sz)?
    } else if let Some(f) = &a.ffn {
        // --ffn: the fraction of routed-expert parameters kept
        let v: f64 = f.parse().map_err(|_| anyhow::anyhow!("--ffn {f:?}: a fraction"))?;
        params - ((1.0 - v.clamp(0.0, 1.0)) * experts_params as f64) as u64
    } else {
        anyhow::bail!("no target (--ratio, --size or --ffn)")
    };
    // the largest cover (most routing mass kept) that is small enough
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    if after(1.0) <= target {
        lo = 1.0;
    } else {
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if after(mid) <= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }
    let cover = lo;
    anyhow::ensure!(
        cover > 0.0,
        "the target {:.3}B is below the parameters outside the routed experts ({:.3}B): a MoE model shrinks only by its experts",
        target as f64 / 1e9,
        (params - experts_params) as f64 / 1e9
    );
    let kept: BTreeMap<&String, usize> = counts.iter().map(|(l, c)| (l, kept_at(c, cover))).collect();
    let n_exp = counts.values().next().map_or(0, Vec::len);
    run.log(&format!(
        "S2: keep {:.1} % of the routing mass → {:.3}B of {:.3}B parameters; experts kept per layer (of {n_exp}): {}",
        cover * 100.0,
        after(cover) as f64 / 1e9,
        params as f64 / 1e9,
        kept.values().map(usize::to_string).collect::<Vec<_>>().join(" ")
    ));
    run.mark_done(2, json!({ "cover": cover, "params_after": after(cover), "kept": kept }))
}

/// S3 for MoE: write only the kept experts (f16 working copy and q8_2f).
pub fn s3_drop(run: &Run) -> Result<()> {
    let s0 = run.require(0)?;
    let s1 = run.require(1)?;
    let s2 = run.require(2)?;
    let stats = path_of(&s1, "routing").context("S1 output")?;
    let cover = s2["cover"].as_f64().context("S2 cover")?;
    let dir = run.stage_dir(3);
    std::fs::create_dir_all(&dir)?;
    let drop = |src: &Path, out: &Path| -> Result<()> {
        crate::moedefrag::cmd_moe_defrag(src.to_str().context("path")?, Some(stats.to_str().context("path")?), cover, out.to_str().context("path")?)
    };
    let q8_out = dir.join("fold-q8_2f.cmf");
    drop(&path_of(&s0, "q8").context("S0 q8")?, &q8_out)?;
    let f16_out = match path_of(&s0, "f16") {
        Some(f) => {
            let out = dir.join("fold-f16.cmf");
            drop(&f, &out)?;
            Some(out)
        }
        None => None,
    };
    let kept = CmfModel::open_sharded(&q8_out)?;
    run.log(&format!("S3: experts the task does not use are gone — {:.3}B parameters", kept.total_param_count() as f64 / 1e9));
    run.mark_done(3, json!({ "f16": f16_out.unwrap_or(q8_out.clone()), "q8": q8_out }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kept_experts_follow_the_routing_mass() {
        // 4 experts: 60 % / 30 % / 9 % / 1 % of the routes
        let c = [9u64, 60, 1, 30];
        assert_eq!(kept_at(&c, 0.5), 1);
        assert_eq!(kept_at(&c, 0.9), 2);
        assert_eq!(kept_at(&c, 0.95), 3);
        assert_eq!(kept_at(&c, 1.0), 4);
        // a layer the forward never routed keeps everything
        assert_eq!(kept_at(&[0, 0, 0], 0.5), 3);
    }

    #[test]
    fn expert_names_parse() {
        assert_eq!(expert_of("model.layers.3.mlp.experts.17.w1.weight"), Some((3, 17)));
        assert_eq!(expert_of("model.layers.3.mlp.gate.weight"), None);
    }
}
