//! `cortiq dflash-route-fit`: the router of a DFlash sidecar.
//!
//! A round goes either to the DFlash block or to the target's own MTP chain,
//! and which of them lands more tokens depends on the text. The target's row
//! at the tip knows the text. Rounds logged with `CMF_DFLASH_ROWS` carry that
//! row and both drafters' drafts; here each round is given to the drafter that
//! does better around it, a subspace of the rows is fitted for each drafter
//! from its own rounds alone (`router::fit_descriptor`, the engine's
//! recon-argmin routing), and the pair goes into the sidecar's header. At
//! decode a round goes to the drafter whose subspace holds the tip's row
//! better.

use anyhow::{Context, Result, anyhow, ensure};
use cortiq_core::CmfModel;
use cortiq_engine::dflash::{ROUTE_KEY, Route, RouteFit, sidecar_path};
use cortiq_engine::pipeline::{
    DFLASH_LEAN_BAND, DFLASH_LEAN_STEP, DflashStart, dflash_round_values, dflash_round_values_at, tail_repeat,
};
use cortiq_engine::router;
use std::path::Path;

/// One logged round.
struct Round {
    row: Vec<f32>,
    /// What a round here lands and costs in plain tokens: the block, the chain.
    value: ((f32, f32), (f32, f32)),
    /// The block does better around this round.
    block: bool,
    /// How many tokens before the round repeat an earlier stretch (`tail_repeat`).
    repeat: usize,
}

/// Another device's prices in plain tokens: the verify and commit of a round
/// of one to seven drafts, a block's drafts, one draft of the chain.
type Prices = ([f32; 7], f32, f32);

/// The rounds of one sequence (a file of `CMF_DFLASH_ROWS` records).
fn rounds(path: &Path, hidden: usize, window: usize, edge: f32, prices: Option<&Prices>) -> Result<Vec<Round>> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    let rec = 24 * 4 + hidden * 4;
    ensure!(!bytes.is_empty() && bytes.len() % rec == 0, "{}: not records of {rec} bytes", path.display());
    let recs: Vec<(Vec<u32>, Vec<f32>)> = bytes
        .chunks_exact(rec)
        .map(|r| {
            let head = r[..96].chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            let row = r[96..].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            (head, row)
        })
        .collect();
    // the stream up to the first logged round (`FILE.ids`): a lookup reads the prompt
    let ids_path = format!("{}.ids", path.display());
    let ids: Vec<u32> = std::fs::read(&ids_path)
        .with_context(|| format!("{ids_path} (rows are logged with it; `q27_dflash_rows.sh` collects both)"))?
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    ensure!(ids.last() == Some(&recs[0].0[1]), "{ids_path}: does not end at the first round's first token");
    // the sequence the rounds committed: a round's first token, the drafts it
    // kept, and the next round's first token after them
    let first = recs[0].0[0] as usize;
    let mut full: Vec<u32> = ids[..ids.len() - 1].to_vec();
    let base = full.len();
    for (h, _) in &recs {
        let (pos, kept) = (h[0] as usize, h[3] as usize);
        ensure!(pos == first + full.len() - base, "{}: a round at {pos} after {} tokens from {first}", path.display(), full.len() - base);
        full.push(h[1]);
        full.extend_from_slice(&h[4..4 + kept]);
    }
    let (mut starts, mut rows, mut repeats) = (Vec::new(), Vec::new(), Vec::new());
    for (h, row) in &recs {
        let at = base + h[0] as usize - first;
        // a round cut short by the end of the answer, or one the chain did not draft, says nothing
        if h[2] != 7 || h[11..18].contains(&u32::MAX) || full.len() < at + 8 {
            continue;
        }
        starts.push(DflashStart { seq: &full[..=at], after: &full[at + 1..], block: &h[4..11], chain: &h[11..18] });
        rows.push(row.clone());
        repeats.push(tail_repeat(&full[..=at]));
    }
    let values = match prices {
        Some((cost, block, step)) => dflash_round_values_at(&starts, cost, *block, *step),
        None => dflash_round_values(&starts),
    };
    let n = values.len();
    Ok((0..n)
        .map(|i| {
            // the drafter that lands more in a unit of time over the rounds around this one
            let (lo, hi) = (i.saturating_sub(window), (i + window + 1).min(n));
            let sum = |f: fn(&((f32, f32), (f32, f32))) -> (f32, f32)| {
                let (t, c) = values[lo..hi].iter().map(f).fold((0f32, 0f32), |s, v| (s.0 + v.0, s.1 + v.1));
                t / c
            };
            Round { row: rows[i].clone(), value: values[i], block: sum(|v| v.0) > sum(|v| v.1) * edge, repeat: repeats[i] }
        })
        .collect())
}

/// A sequence of rounds and the name of its file (`NAME_KIND`).
type Seq = (String, Vec<Round>);

/// What a policy lands in a plain token's time over `rounds` (tokens over
/// their cost): the block's rounds where `pick` says so, the chain's
/// elsewhere. (Every logged start counts as a round; a drafter's own rounds
/// would start elsewhere.)
fn rate<'a>(seqs: impl IntoIterator<Item = &'a Seq>, mut pick: impl FnMut(&Round, bool) -> bool) -> (f32, f32) {
    let (mut tokens, mut cost, mut rounds, mut to_block) = (0f32, 0f32, 0f32, 0usize);
    for (_, s) in seqs {
        let mut last = false;
        for r in s {
            last = pick(r, last);
            let v = if last { r.value.0 } else { r.value.1 };
            tokens += v.0;
            cost += v.1;
            to_block += last as usize;
            rounds += 1.0;
        }
    }
    (tokens / cost.max(1e-6), to_block as f32 / rounds.max(1.0))
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    model: &str,
    rows_dir: &str,
    rank: usize,
    window: usize,
    edge: f32,
    hold_every: usize,
    prices: Option<&str>,
    write: bool,
) -> Result<()> {
    let prices: Option<Prices> = match prices {
        Some(list) => {
            let v: Vec<f32> = list.split(',').map(|x| x.trim().parse::<f32>()).collect::<Result<_, _>>().context("--prices")?;
            ensure!(v.len() == 9, "--prices: nine numbers (seven rounds, the block, a chain draft), {} given", v.len());
            // a router fitted at another device's prices labels each round by
            // what it lands at those prices (the card it will run on)
            Some((v[..7].try_into().unwrap(), v[7], v[8]))
        }
        None => None,
    };
    let side = sidecar_path(Path::new(model));
    let hidden = {
        let m = CmfModel::open(&side).with_context(|| side.display().to_string())?;
        cortiq_engine::dflash::Geometry::of(&m).map_err(|e| anyhow!(e))?.hidden
    };
    let mut files: Vec<_> = std::fs::read_dir(rows_dir)
        .with_context(|| rows_dir.to_string())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "rows") && std::fs::metadata(p).is_ok_and(|m| m.len() > 0))
        // a tar from macOS brings `._name` twins along
        .filter(|p| !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')))
        .collect();
    files.sort();
    ensure!(files.len() >= 4, "{rows_dir}: {} files of rows, at least four are needed", files.len());
    let mut all = Vec::new();
    for f in &files {
        let r = rounds(f, hidden, window, edge, prices.as_ref())?;
        let name = f.file_stem().unwrap().to_string_lossy().to_string();
        let one = [(name.clone(), r)];
        let (b, c) = (rate(&one, |_, _| true).0, rate(&one, |_, _| false).0);
        let [(_, r)] = one;
        println!("{name}: {} rounds, the block {b:.2} and the chain {c:.2} tokens in a plain token's time, {} of them the block's", r.len(), r.iter().filter(|r| r.block).count());
        all.push((name, r));
    }
    // every `hold_every`-th sequence takes no part in the fit
    let (mut fit, mut held): (Vec<Seq>, Vec<Seq>) = (Vec::new(), Vec::new());
    for (i, r) in all.into_iter().enumerate() {
        if hold_every > 0 && i % hold_every == hold_every - 1 { held.push(r) } else { fit.push(r) }
    }
    let class = |seqs: &[Seq], block: bool| -> Vec<Vec<f32>> { seqs.iter().flat_map(|s| &s.1).filter(|r| r.block == block).map(|r| r.row.clone()).collect() };
    let build = |seqs: &[Seq]| -> Result<RouteFit> {
        let (b, c) = (class(seqs, true), class(seqs, false));
        ensure!(b.len() >= 8 && c.len() >= 8, "{} rounds are the block's and {} the chain's: too few of one to fit", b.len(), c.len());
        Ok(RouteFit {
            block: router::fit_descriptor(&b, 0, rank).ok_or_else(|| anyhow!("the block's rounds do not fit"))?,
            chain: router::fit_descriptor(&c, 0, rank).ok_or_else(|| anyhow!("the chain's rounds do not fit"))?,
            block_rows: b.len(),
            chain_rows: c.len(),
        })
    };
    let report = |name: &str, route: &Route, seqs: &[Seq]| {
        let line = |name: &str, seqs: &[&Seq]| {
            let n = seqs.iter().map(|s| s.1.len()).sum::<usize>().max(1);
            let right = seqs.iter().flat_map(|s| &s.1).filter(|r| (route.margin(&r.row) > 0.0) == r.block).count();
            let of = || seqs.iter().copied();
            let chain = rate(of(), |_, _| false).0;
            let block = rate(of(), |_, _| true).0;
            let best = rate(of(), |r, _| r.value.0.0 / r.value.0.1 > r.value.1.0 / r.value.1.1).0;
            let labels = rate(of(), |r, _| r.block).0;
            let raw = rate(of(), |r, _| route.margin(&r.row) > 0.0);
            // as the rounds use it: a running average over a sequence, the drafter in turn kept inside the band
            let mut lean = 0f32;
            let mut at = 0usize;
            let starts: Vec<usize> = seqs.iter().scan(0usize, |a, s| { let v = *a; *a += s.1.len(); Some(v) }).collect();
            let run = rate(of(), |r, last| {
                if starts.contains(&at) {
                    lean = 0.0;
                }
                at += 1;
                lean += DFLASH_LEAN_STEP * (route.margin(&r.row) - lean);
                if last { lean > -DFLASH_LEAN_BAND } else { lean > DFLASH_LEAN_BAND }
            });
            // the router, and the chain wherever the last tokens repeat the context that far
            let copy: Vec<String> = [3usize, 4, 6, 8, 12, 16]
                .iter()
                .map(|&m| {
                    let r = rate(of(), |r, _| r.repeat < m && route.margin(&r.row) > 0.0);
                    format!("{m}: {:.3} ({:.0} %)", r.0, 100.0 * r.1)
                })
                .collect();
            println!(
                "{name}: {n} rounds, the router agrees with {:.1} % of them; tokens in a plain token's time: the chain alone {chain:.3}, the block alone {block:.3}, the router a round at a time {:.3} ({:.0} % to the block), averaged {:.3} ({:.0} %), the fit's own split {labels:.3}, the better of the two every round {best:.3}; the router with the chain where the tail repeats N tokens or more: {}",
                100.0 * right as f32 / n as f32,
                raw.0,
                100.0 * raw.1,
                run.0,
                100.0 * run.1,
                copy.join(", ")
            );
        };
        line(name, &seqs.iter().collect::<Vec<_>>());
        // a file's kind is what follows the last `_` of its name
        let kind = |s: &Seq| s.0.rsplit('_').next().unwrap_or("").to_string();
        let mut kinds: Vec<String> = seqs.iter().map(kind).collect();
        kinds.sort();
        kinds.dedup();
        if kinds.len() > 1 {
            for k in kinds {
                line(&format!("  {k}"), &seqs.iter().filter(|s| kind(s) == k).collect::<Vec<_>>());
            }
        }
    };
    if !held.is_empty() {
        let part = build(&fit)?;
        report("sequences held out of the fit", &Route::of(&part, hidden).map_err(|e| anyhow!(e))?, &held);
    }
    // the router that is written is fitted on every sequence
    fit.extend(held);
    let whole = build(&fit)?;
    report("all sequences, fitted on them", &Route::of(&whole, hidden).map_err(|e| anyhow!(e))?, &fit);
    println!("rank {}, {} rounds of the block and {} of the chain", whole.block.rank, whole.block_rows, whole.chain_rows);
    if write {
        let value = serde_json::to_value(&whole)?;
        CmfModel::update_header_append(&side, move |h| {
            let p = h.provenance.get_or_insert_with(|| serde_json::json!({}));
            p[ROUTE_KEY] = value;
        })
        .map_err(|e| anyhow!("{}: {e}", side.display()))?;
        println!("written to {}", side.display());
    }
    Ok(())
}
