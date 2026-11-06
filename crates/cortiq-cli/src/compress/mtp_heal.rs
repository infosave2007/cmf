//! MTP draft trained on its own trunk after a structural compress.
//!
//! The draft predicts the token after next from the trunk's last hidden
//! state; the fold and the heal change that trunk, and the source's draft
//! stops agreeing with it (Qwen3.8 27B → 14B: 59–80 % of drafts accepted on
//! the source, 51–54 % after). Neither a ridge map of its input onto the
//! source's hidden rows nor a least-squares refit of its read-out moved the
//! acceptance (measured, 56/48 and 48/52 %): the draft has to be trained on
//! the compressed trunk's own predictions, the token its verify accepts.
//!
//! The trunk runs once on the trainer: per window its last hidden rows and
//! its next-token top-k. The draft's block is loaded as a one-layer model
//! (its attention and FFN as layer 0, its final norm, the shared head) that
//! starts from the block's input `eh_proj·[enorm(e_{p+1}); hnorm(h_p)]`, and
//! learns the trunk's distribution at `p+1` by KL — self-distillation. The
//! checkpoint whose chain keeps the most drafts a round on held-out rows is
//! written back into `model.mtp.*`.

use super::cmfio::{self, Encoding};
use super::data::Data;
use crate::tdata::TdataSplit;
use anyhow::{Context, Result, anyhow, ensure};
use cortiq_core::format::{CmfModel, CmfStreamWriter};
use cortiq_core::types::{LayerType, NormStyle, TensorDtype};
use cortiq_engine::gpu_wgpu::nvg_dev::{self as dv, DMat};
use cortiq_engine::nvg_train as nt;
use serde_json::json;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct Cfg {
    /// Chain steps trained (the runtime drafts 2–7 deep)
    pub chain: usize,
    /// Steps between rebuilds of the chain's inputs from the current weights
    pub refresh: usize,
    /// 0: every row starts a chain, and step j of a row reads step j rows of
    /// the rows before it. N: a chain starts every N rows and is laid out as
    /// a round sees it, the trunk's rows up to its start and its own after
    pub stride: usize,
    /// Tokens taken from a data set that names no count of its own
    pub tokens: usize,
    pub window: usize,
    /// 0 = no training: the trunk's pass, the chain's hold rates, the dump,
    /// and with a codec other than `keep` the draft as it is in that codec
    pub epochs: f64,
    pub lr: f64,
    pub topk: usize,
    pub batch: usize,
    pub eval_every: usize,
    pub patience: usize,
    /// Share of a target row's mass put on the trunk's own token: 0 = the
    /// trunk's distribution as it is, 1 = its choice alone, what a greedy
    /// verify accepts
    pub hard: f64,
    /// Directory that keeps the trunk's pass per data set (hidden rows and
    /// top-k): a later run on the same trunk, data and window reads it back
    pub trunk_cache: Option<PathBuf>,
    /// Codec of the trained draft in the output: `keep` (the file's own)
    /// or a codec's name
    pub codec: String,
    /// The same result with the draft in other codecs
    pub also: Vec<(String, PathBuf)>,
    /// Directory for the chain's rows per step (output hidden, top-4),
    /// before and after training, and the tensors a draft reads
    pub dump: Option<PathBuf>,
    /// Directory and trunk layers: the layers' output rows for the held-out
    /// windows, what a draft model built elsewhere reads from its target
    pub taps: Option<(PathBuf, Vec<usize>)>,
}

impl Cfg {
    /// The settings `compress` trains a draft with.
    pub fn stage(epochs: f64) -> Self {
        Cfg {
            chain: 3,
            refresh: 150,
            stride: 0,
            tokens: 393216,
            window: 2048,
            epochs,
            lr: 2e-5,
            topk: 64,
            batch: 2,
            eval_every: 25,
            patience: 6,
            hard: 0.0,
            trunk_cache: None,
            codec: "keep".into(),
            also: Vec::new(),
            dump: None,
            taps: None,
        }
    }
}

pub struct Report {
    pub steps: usize,
    pub best_step: usize,
    pub loss_start: f64,
    pub loss_best: f64,
    /// Per chain step, the share of held-out rows where the draft's token
    /// is the trunk's, among the rows where the steps before held
    pub hold_start: Vec<f64>,
    pub hold_best: Vec<f64>,
}

/// Top-k kept per row of a dumped chain step.
const KK: usize = 4;

const LAYER: [&str; 11] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// `x / rms(x)` times the gain (`w` or `1 + w`), in place.
fn rms_into(x: &[f32], w: &[f32], style: NormStyle, eps: f64, out: &mut [f32]) {
    let ms = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + eps).sqrt() as f32;
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = v * inv * if matches!(style, NormStyle::Gemma) { 1.0 + g } else { g };
    }
}

/// The draft's block as a one-layer model: its layer as `model.layers.0`,
/// its final norm as `model.norm`, the trunk's embedding and head. `init`
/// supplies the layer's starting weights (the source's own, in their
/// precision, when given).
fn draft_model(student: &CmfModel, init: &CmfModel, path: &Path) -> Result<()> {
    let tmp = path.with_extension("cmf.partial");
    let _ = std::fs::remove_file(&tmp);
    let mut w = CmfStreamWriter::new(&tmp, CmfStreamWriter::head_reserve_for(16, 48))?;
    for name in ["model.embed_tokens.weight", "lm_head.weight"] {
        let e = student.tensor(name).with_context(|| format!("mtp heal: no {name}"))?;
        w.push_bounded(name, e.dtype, &e.shape, student.entry_bytes(e), 1 << 28)?;
    }
    let e = init.tensor("model.mtp.norm.weight").context("mtp heal: no model.mtp.norm.weight")?;
    w.push("model.norm.weight", e.dtype, &e.shape, init.entry_bytes(e))?;
    let mut inter = 0;
    for t in LAYER {
        let src = format!("model.mtp.layers.0.{t}");
        let e = init.tensor(&src).with_context(|| format!("mtp heal: no {src}"))?;
        if t == "mlp.gate_proj.weight" {
            inter = e.shape[0];
        }
        let name = format!("model.layers.0.{t}");
        // the trainer exports f16/f32 only: a quantized draft starts from
        // its dequantized f16 copy
        if matches!(e.dtype, TensorDtype::F16 | TensorDtype::F32) {
            w.push(&name, e.dtype, &e.shape, init.entry_bytes(e))?;
        } else {
            let v = cmfio::tensor_f32(init, e)?;
            let b: Vec<u8> = v.iter().flat_map(|&x| cortiq_core::quant::f32_to_f16(x).to_le_bytes()).collect();
            w.push(&name, TensorDtype::F16, &e.shape, &b)?;
        }
    }
    let mut h = student.header.clone();
    h.section_hashes = None;
    h.segments.clear();
    h.arch.num_layers = 1;
    h.arch.layer_types = vec![LayerType::FullAttention];
    h.arch.intermediate_size = inter;
    h.arch.mtp = None;
    h.arch.attention_heads_per_layer = None;
    h.arch.kv_heads_per_layer = None;
    h.arch.no_rope_layers = None;
    w.finish(&h, None, student.vocab.as_deref())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// One window: its padded ids, its real length, the trunk's top-k at every
/// position, and per chain step the block's input rows (f16) — step 1 from
/// the trunk's hidden rows, step j > 1 from the draft's own output at step
/// j − 1, the way a speculative round chains the head on itself.
struct Win {
    ids: Vec<u32>,
    len: usize,
    ti: Vec<u32>,
    tp: Vec<f32>,
    h0: Vec<Vec<u16>>,
}

fn to_f16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| cortiq_core::quant::f32_to_f16(x)).collect()
}

fn from_f16(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&x| cortiq_core::quant::f16_to_f32(x)).collect()
}

fn fnv(mut h: u64, b: &[u8]) -> u64 {
    for &x in b {
        h = (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn put<T: Copy, const N: usize>(f: &mut impl Write, v: &[T], le: fn(T) -> [u8; N]) -> Result<()> {
    let mut b = Vec::with_capacity(v.len() * N);
    for &x in v {
        b.extend_from_slice(&le(x));
    }
    f.write_all(&b)?;
    Ok(())
}

fn get<T, const N: usize>(f: &mut impl Read, n: usize, le: fn([u8; N]) -> T) -> Result<Vec<T>> {
    let mut b = vec![0u8; n * N];
    f.read_exact(&mut b)?;
    Ok(b.chunks_exact(N).map(|c| le(c.try_into().unwrap())).collect())
}

fn create(dir: &Path, name: &str) -> Result<BufWriter<std::fs::File>> {
    let p = dir.join(name);
    Ok(BufWriter::with_capacity(1 << 22, std::fs::File::create(&p).with_context(|| format!("creating {}", p.display()))?))
}

/// Every tenth window is held out of training and is what a measure counts.
/// `CMF_MTP_HOLD_ALL=1` holds out all of them: a measure (`--epochs 0`) of text
/// no training read, such as answers to fresh prompts.
fn held_out(i: usize) -> bool {
    static ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALL.get_or_init(|| std::env::var("CMF_MTP_HOLD_ALL").as_deref() == Ok("1")) || i % 10 == 9
}

fn open(dir: &Path, name: &str) -> Result<BufReader<std::fs::File>> {
    let p = dir.join(name);
    Ok(BufReader::with_capacity(1 << 22, std::fs::File::open(&p).with_context(|| format!("opening {}", p.display()))?))
}

/// One data set of `spec` (`PATH.tdata[@TOKENS]`): its train records, in a
/// fixed shuffled order, as one stream cut into windows of `t`. A short
/// answer shares its window with the next record, so only the last window
/// carries padding.
fn packed_windows(item: &str, tokens: usize, t: usize, min: usize) -> Result<(String, Vec<Vec<u32>>)> {
    let (path, cap) = match item.rsplit_once('@') {
        Some((p, n)) if n.parse::<usize>().is_ok() => (p, n.parse::<usize>().unwrap()),
        _ => (item, tokens),
    };
    let data = Data::load(path)?;
    let mut recs: Vec<_> = data.split(TdataSplit::Train).collect();
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    for i in (1..recs.len()).rev() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        recs.swap(i, (rng % (i as u64 + 1)) as usize);
    }
    let (mut wins, mut cur, mut total) = (Vec::new(), Vec::with_capacity(t), 0usize);
    for r in recs {
        if total >= cap {
            break;
        }
        let take = r.ids.len().min(cap - total);
        total += take;
        for &id in &r.ids[..take] {
            cur.push(id);
            if cur.len() == t {
                wins.push(std::mem::replace(&mut cur, Vec::with_capacity(t)));
            }
        }
    }
    if cur.len() >= min {
        wins.push(cur);
    }
    let root = path.rfind(".tdata").map_or(path, |i| &path[..i]);
    let name = Path::new(root).file_name().and_then(|n| n.to_str()).unwrap_or("data").to_string();
    Ok((name, wins))
}

/// What tells one trunk from another: its depth and the first bytes of its
/// first and last MLP and of its head (a draft retrained in the same file
/// keeps the key).
fn trunk_key(m: &CmfModel) -> u64 {
    let nl = m.arch().num_layers;
    let mut h = fnv(0xcbf2_9ce4_8422_2325, &(nl as u64).to_le_bytes());
    for name in ["model.layers.0.mlp.down_proj.weight".to_string(), format!("model.layers.{}.mlp.down_proj.weight", nl.saturating_sub(1)), "lm_head.weight".into(), "model.embed_tokens.weight".into()] {
        if let Some(e) = m.tensor(&name) {
            let b = m.entry_bytes(e);
            h = fnv(h, &b[..b.len().min(1 << 16)]);
        }
    }
    h
}

/// Train `student`'s draft on its own trunk; write the student with the
/// trained draft to `out`. `init` (the source, before compress) gives the
/// draft's starting weights when it carries them in a higher precision.
///
/// The round chains the draft: its first step reads the trunk's hidden
/// row, every later one the block's own output. Trained on the first step
/// alone (Qwen3.8 14B) it accepted 88–90 % at depth 1 and still 54 % at the
/// runtime's depth 3–5, so every step of the chain is trained: step j's
/// input is rebuilt from the current weights' step j − 1 output every
/// `refresh` steps, its target the trunk's distribution j positions ahead.
///
/// `data_spec` names one or more data sets, comma-separated, each
/// `PATH.tdata[@TOKENS]`.
pub fn heal(student_path: &Path, init_path: Option<&Path>, data_spec: &str, cfg: &Cfg, out: &Path, threads: usize) -> Result<Report> {
    let c = cortiq_engine::gpu_wgpu::train_ops::dev().context("mtp heal: no wgpu GPU")?;
    let student = CmfModel::open_sharded(student_path).with_context(|| format!("opening {}", student_path.display()))?;
    super::heal_gpu::usable(&student).map_err(|e| anyhow!("mtp heal: {e}"))?;
    let init = match init_path {
        Some(p) => CmfModel::open_sharded(p).with_context(|| format!("opening {}", p.display()))?,
        None => CmfModel::open_sharded(student_path)?,
    };
    let arch = student.arch().clone();
    let (hs, t, k, nb, chain) = (arch.hidden_size, cfg.window, cfg.topk, cfg.batch.max(1), cfg.chain.max(1));
    for n in ["model.mtp.eh_proj.weight", "model.mtp.enorm.weight", "model.mtp.hnorm.weight"] {
        ensure!(student.tensor(n).is_some(), "mtp heal: {} carries no canonical MTP draft ({n})", student_path.display());
    }
    let pad = student.header.tokenizer_config.as_ref().and_then(|t| t.eos_token_ids.first().copied()).unwrap_or(0);

    // windows per data set; every tenth of all of them is held out
    let sets: Vec<(String, Vec<Vec<u32>>)> = data_spec.split(',').filter(|s| !s.is_empty()).map(|item| packed_windows(item, cfg.tokens, t, 16 + chain)).collect::<Result<_>>()?;
    let n_all: usize = sets.iter().map(|s| s.1.len()).sum();
    // window 0 is held out only when all of them are: a measure of a short text
    let need = if held_out(0) { 1 } else { 10 * nb };
    ensure!(n_all >= need, "mtp heal: {n_all} windows in {data_spec}, need {need} (every tenth is held out)");
    for (name, ws) in &sets {
        eprintln!("mtp heal: {name}: {} windows, {} tokens", ws.len(), ws.iter().map(Vec::len).sum::<usize>());
    }
    let total: usize = sets.iter().flat_map(|s| s.1.iter()).map(Vec::len).sum();
    let layout = if cfg.stride > 0 { format!(", a chain every {} rows, laid out as a round", cfg.stride) } else { String::new() };
    eprintln!("mtp heal: {n_all} windows, {total} tokens; chain depth {chain}{layout}");

    let embed = cmfio::tensor_f32(&student, student.tensor("model.embed_tokens.weight").context("mtp heal: no embedding")?)?;
    let enorm = cmfio::tensor_f32(&student, student.tensor("model.mtp.enorm.weight").unwrap())?;
    let hnorm = cmfio::tensor_f32(&student, student.tensor("model.mtp.hnorm.weight").unwrap())?;
    // the block input through the projection the draft will run with
    let eh_w = cmfio::tensor_f32(&student, student.tensor("model.mtp.eh_proj.weight").unwrap())?;
    let eh = DMat::from_f32(hs, 2 * hs, &eh_w);
    // step j's input rows: row p reads token p + j and `hid`'s row p (the
    // trunk's for j = 1, the block's own step j − 1 output after)
    let block_in = |ids: &[u32], len: usize, j: usize, hid: &[f32]| -> Vec<u16> {
        let mut cat = vec![0f32; t * 2 * hs];
        for p in 0..len.saturating_sub(j) {
            let tok = ids[p + j] as usize;
            let (ce, ch) = cat[p * 2 * hs..(p + 1) * 2 * hs].split_at_mut(hs);
            rms_into(&embed[tok * hs..(tok + 1) * hs], &enorm, arch.norm_style, arch.rms_norm_eps, ce);
            rms_into(&hid[p * hs..(p + 1) * hs], &hnorm, arch.norm_style, arch.rms_norm_eps, ch);
        }
        to_f16(&dv::mm(&DMat::from_f32(t, 2 * hs, &cat), false, &eh, true).to_f32())
    };

    // ── the trunk once per data set: hidden rows and the next-token top-k
    //    per window, kept in `trunk_cache` for the next run ──
    let mut wins: Vec<Win> = Vec::with_capacity(n_all);
    let mut kept: Vec<(PathBuf, usize)> = Vec::new();
    {
        let mut trunk: Option<(nt::Net, nt::Work)> = None;
        let tkey = fnv(fnv(trunk_key(&student), &(t as u64).to_le_bytes()), &(k as u64).to_le_bytes());
        for (name, ws) in &sets {
            let key = ws.iter().flatten().fold(tkey, |h, id| fnv(h, &id.to_le_bytes()));
            let dir = cfg.trunk_cache.as_ref().map(|d| d.join(format!("{name}-{key:016x}")));
            let t0 = Instant::now();
            match &dir {
                Some(d) if d.join("meta.json").exists() => {
                    let (mut fl, mut fi, mut fti, mut ftp, mut fh) = (open(d, "len.u32")?, open(d, "ids.u32")?, open(d, "ti.u32")?, open(d, "tp.f32")?, open(d, "h.f32")?);
                    for _ in 0..ws.len() {
                        let len = get(&mut fl, 1, u32::from_le_bytes)?[0] as usize;
                        let ids = get(&mut fi, t, u32::from_le_bytes)?;
                        let ti = get(&mut fti, t * k, u32::from_le_bytes)?;
                        let tp = get(&mut ftp, t * k, f32::from_le_bytes)?;
                        let h = get(&mut fh, t * hs, f32::from_le_bytes)?;
                        let h1 = block_in(&ids, len, 1, &h);
                        wins.push(Win { ids, len, ti, tp, h0: vec![h1] });
                    }
                    eprintln!("  trunk: {name}: {} windows read back from {} ({:.0}s)", ws.len(), d.display(), t0.elapsed().as_secs_f64());
                }
                _ => {
                    if trunk.is_none() {
                        let tnet = nt::Net::load(c, &student, true, false, None).map_err(|e| anyhow!("mtp heal: trunk: {e}"))?;
                        let tw = nt::Work::new(c, &tnet, 1, t);
                        trunk = Some((tnet, tw));
                        eprintln!("  trunk on the trainer ({:.0}s)", t0.elapsed().as_secs_f64());
                    }
                    let (tnet, tw) = trunk.as_ref().unwrap();
                    let mut files = match &dir {
                        Some(d) => {
                            std::fs::create_dir_all(d)?;
                            Some((create(d, "len.u32")?, create(d, "ids.u32")?, create(d, "ti.u32")?, create(d, "tp.f32")?, create(d, "h.f32")?))
                        }
                        None => None,
                    };
                    let (t0, mut done) = (Instant::now(), 0usize);
                    for (i, ids) in ws.iter().enumerate() {
                        let len = ids.len();
                        let mut pids = ids.clone();
                        pids.resize(t, pad);
                        let (ti, tp, h) = nt::teacher_topk_hidden(c, tnet, tw, &pids, k).with_context(|| format!("mtp heal: {name}: window {i}: trunk pass failed"))?;
                        if let Some((fl, fi, fti, ftp, fh)) = files.as_mut() {
                            put(fl, &[len as u32], u32::to_le_bytes)?;
                            put(fi, &pids, u32::to_le_bytes)?;
                            put(fti, &ti, u32::to_le_bytes)?;
                            put(ftp, &tp, f32::to_le_bytes)?;
                            put(fh, &h, f32::to_le_bytes)?;
                        }
                        let h1 = block_in(&pids, len, 1, &h);
                        wins.push(Win { ids: pids, len, ti, tp, h0: vec![h1] });
                        done += len;
                        eprint!("\r  trunk: {name}: {done} tokens · {:.0} tok/s   ", done as f64 / t0.elapsed().as_secs_f64().max(1e-9));
                    }
                    eprintln!();
                    if let (Some(d), Some((mut fl, mut fi, mut fti, mut ftp, mut fh))) = (&dir, files) {
                        for f in [&mut fl, &mut fi, &mut fti, &mut ftp, &mut fh] {
                            f.flush()?;
                        }
                        // written last: a directory without it is an interrupted pass
                        std::fs::write(d.join("meta.json"), serde_json::to_vec_pretty(&json!({ "set": name, "windows": ws.len(), "window": t, "topk": k, "hidden": hs, "tokens": done }))?)?;
                    }
                }
            }
            if let Some(d) = dir {
                kept.push((d, ws.len()));
            }
        }
        if let Some((d, layers)) = cfg.taps.as_ref().filter(|(d, _)| !d.join("meta.json").exists()) {
            std::fs::create_dir_all(d)?;
            if trunk.is_none() {
                let t0 = Instant::now();
                let tnet = nt::Net::load(c, &student, true, false, None).map_err(|e| anyhow!("mtp heal: trunk: {e}"))?;
                let tw = nt::Work::new(c, &tnet, 1, t);
                trunk = Some((tnet, tw));
                eprintln!("  trunk on the trainer ({:.0}s)", t0.elapsed().as_secs_f64());
            }
            let (tnet, tw) = trunk.as_ref().unwrap();
            let held: Vec<usize> = (0..wins.len()).filter(|&i| held_out(i)).collect();
            let (mut ft, mut fl, mut fi, mut fti) = (create(d, "taps.f16")?, create(d, "len.u32")?, create(d, "ids.u32")?, create(d, "ti.u32")?);
            for &i in &held {
                let w = &wins[i];
                let rows = nt::layer_outputs(c, tnet, tw, &w.ids, layers).with_context(|| format!("mtp heal: taps: window {i}: trunk pass failed"))?;
                for r in &rows {
                    put(&mut ft, &to_f16(r), u16::to_le_bytes)?;
                }
                put(&mut fl, &[w.len as u32], u32::to_le_bytes)?;
                put(&mut fi, &w.ids, u32::to_le_bytes)?;
                put(&mut fti, &w.ti.iter().step_by(k).copied().collect::<Vec<u32>>(), u32::to_le_bytes)?;
            }
            ft.flush()?;
            for f in [&mut fl, &mut fi, &mut fti] {
                f.flush()?;
            }
            // written last: a directory without it is an interrupted pass
            std::fs::write(
                d.join("meta.json"),
                serde_json::to_vec_pretty(&json!({
                    "windows": held, "window": t, "hidden": hs, "layers": layers,
                    "taps": "[window][layer][row][hidden] f16, a layer's output before any norm", "ti": "the trunk's own next token per row",
                }))?,
            )?;
            eprintln!("  taps: layers {layers:?} of {} held-out windows in {}", held.len(), d.display());
        }
    }

    // ── the draft as a one-layer student ──
    let dpath = out.with_extension("draft.cmf");
    draft_model(&student, &init, &dpath)?;
    let dm = CmfModel::open_sharded(&dpath)?;
    let snet = nt::Net::load_heal(c, &dm, nt::HealCfg { train_from: 0, ffn_only: false, prefix_cached: false }).map_err(|e| anyhow!("mtp heal: draft: {e}"))?;
    let sw = nt::Work::new(c, &snet, nb, t);
    let rw = nt::Work::new(c, &snet, 1, t);
    let stride = cfg.stride;
    ensure!(stride == 0 || stride >= chain, "mtp heal: --stride {stride} is shorter than the chain ({chain})");
    // A window as rounds see it: a chain starts every `stride` rows from
    // `off`; the row after a chain's row q reads the token at q + 2 and the
    // block's output at q, and every other row stays the trunk's. A row q
    // then predicts the token at q + 2 whatever its step is.
    let splice = |net: &nt::Net, w1: &nt::Work, w: &Win, off: usize| -> Result<Vec<f32>> {
        let mut x = from_f16(&w.h0[0]);
        for j in 2..=chain {
            let o = nt::hidden_out(c, net, w1, &w.ids, Some(&x)).context("mtp heal: chain forward failed")?;
            let b = block_in(&w.ids, w.len, 2, &o);
            let mut n = off;
            while n + j < w.len {
                let q = n + j - 1;
                x[q * hs..(q + 1) * hs].copy_from_slice(&from_f16(&b[(q - 1) * hs..q * hs]));
                n += stride;
            }
        }
        Ok(x)
    };
    // the chain's later steps from the current weights
    let refresh = |wins: &mut Vec<Win>, round: usize| -> Result<()> {
        let t0 = Instant::now();
        for (i, w) in wins.iter_mut().enumerate() {
            if stride > 0 {
                let x = splice(&snet, &rw, w, (i * 5 + round * 3) % stride)?;
                w.h0.truncate(1);
                w.h0.push(to_f16(&x));
                continue;
            }
            w.h0.truncate(1);
            for j in 2..=chain {
                let o = nt::hidden_out(c, &snet, &rw, &w.ids, Some(&from_f16(&w.h0[j - 2]))).context("mtp heal: chain forward failed")?;
                let x = block_in(&w.ids, w.len, j, &o);
                w.h0.push(x);
            }
        }
        if chain > 1 {
            eprintln!("  chain steps 2..{chain} rebuilt from the current draft ({:.0}s)", t0.elapsed().as_secs_f64());
        }
        Ok(())
    };
    let (val, train): (Vec<usize>, Vec<usize>) = (0..wins.len()).partition(|&i| held_out(i));
    // The chain of `net` over the windows `idx`, each step from the one
    // before: per step, of the held-out rows where the steps before held
    // and every token the chain was fed is the trunk's own choice (what a
    // greedy round feeds it), how many the draft's token equals the
    // trunk's. `dump`: every step's output rows and top-4 go to its files.
    let chain_pass = |net: &nt::Net, w1: &nt::Work, wins: &[Win], idx: &[usize], dump: Option<&Path>| -> Result<Vec<(u64, u64)>> {
        let mut cnt = vec![(0u64, 0u64); chain];
        let mut files = match dump {
            Some(d) => {
                std::fs::create_dir_all(d)?;
                Some((1..=chain).map(|j| Ok((create(d, &format!("x{j}.f16"))?, create(d, &format!("d{j}.u32"))?, create(d, &format!("dp{j}.f32"))?))).collect::<Result<Vec<_>>>()?)
            }
            None => None,
        };
        for &i in idx {
            let w = &wins[i];
            let mut x = from_f16(&w.h0[0]);
            let (mut held, mut on) = (vec![true; t], vec![true; t]);
            for j in 1..=chain {
                let (di, dp, o) = nt::topk_hidden_in(c, net, w1, &w.ids, Some(&x), KK).context("mtp heal: chain forward failed")?;
                let rows = w.len.saturating_sub(j);
                for p in 0..rows {
                    on[p] &= w.ids[p + j] == w.ti[(p + j - 1) * k];
                    if !(on[p] && held[p]) {
                        held[p] = false;
                        continue;
                    }
                    let hit = di[p * KK] == w.ti[(p + j) * k];
                    if held_out(i) {
                        cnt[j - 1].0 += 1;
                        cnt[j - 1].1 += hit as u64;
                    }
                    held[p] = hit;
                }
                held[rows..].fill(false);
                if let Some(f) = files.as_mut() {
                    put(&mut f[j - 1].0, &to_f16(&o), u16::to_le_bytes)?;
                    put(&mut f[j - 1].1, &di, u32::to_le_bytes)?;
                    put(&mut f[j - 1].2, &dp, f32::to_le_bytes)?;
                }
                if j < chain {
                    x = from_f16(&block_in(&w.ids, w.len, j + 1, &o));
                }
            }
        }
        if let Some(f) = files.as_mut() {
            for (a, b, d) in f.iter_mut() {
                a.flush()?;
                b.flush()?;
                d.flush()?;
            }
        }
        Ok(cnt)
    };
    // The same count over chains laid out as rounds (`splice`), two
    // offsets a window.
    let round_pass = |net: &nt::Net, w1: &nt::Work, wins: &[Win], idx: &[usize]| -> Result<Vec<(u64, u64)>> {
        let mut cnt = vec![(0u64, 0u64); chain];
        for &i in idx {
            let w = &wins[i];
            for off in [0, stride / 2] {
                let x = splice(net, w1, w, off)?;
                let (di, _, _) = nt::topk_hidden_in(c, net, w1, &w.ids, Some(&x), KK).context("mtp heal: chain forward failed")?;
                let mut n = off;
                while n + 1 < w.len {
                    for j in 1..=chain {
                        let q = n + j - 1;
                        if q + 1 >= w.len || w.ids[q + 1] != w.ti[q * k] {
                            break;
                        }
                        let hit = di[q * KK] == w.ti[(q + 1) * k];
                        cnt[j - 1].0 += 1;
                        cnt[j - 1].1 += hit as u64;
                        if !hit {
                            break;
                        }
                    }
                    n += stride;
                }
            }
        }
        Ok(cnt)
    };
    // rounds as a decode has them: every start's chain (one pass per offset
    // of the stride), then a walk where the next round starts after what
    // this one kept, so a round starts where the one before missed
    let order_pass = |net: &nt::Net, w1: &nt::Work, wins: &[Win], idx: &[usize]| -> Result<Vec<(u64, u64)>> {
        let mut cnt = vec![(0u64, 0u64); chain];
        for &i in idx {
            let w = &wins[i];
            let mut top: Vec<Vec<u32>> = Vec::with_capacity(stride);
            for off in 0..stride {
                let x = splice(net, w1, w, off)?;
                let (di, _, _) = nt::topk_hidden_in(c, net, w1, &w.ids, Some(&x), KK).context("mtp heal: chain forward failed")?;
                top.push(di.iter().step_by(KK).copied().collect());
            }
            let mut n = 0;
            while n + 1 < w.len {
                if w.ids[n + 1] != w.ti[n * k] {
                    n += 1;
                    continue;
                }
                let mut kept = 0;
                for j in 1..=chain {
                    let q = n + j - 1;
                    if q + 1 >= w.len || w.ids[q + 1] != w.ti[q * k] {
                        break;
                    }
                    let hit = top[n % stride][q] == w.ti[(q + 1) * k];
                    cnt[j - 1].0 += 1;
                    cnt[j - 1].1 += hit as u64;
                    if !hit {
                        break;
                    }
                    kept += 1;
                }
                n += kept + 1;
            }
        }
        Ok(cnt)
    };
    let rates = |cnt: &[(u64, u64)]| -> Vec<f64> { cnt.iter().map(|&(n, h)| h as f64 / n.max(1) as f64).collect() };
    let show = |cnt: &[(u64, u64)]| -> String { format!("{} (of {} rows)", cnt.iter().map(|&(n, h)| format!("{:.3}", h as f64 / n.max(1) as f64)).collect::<Vec<_>>().join(" / "), cnt.iter().map(|c| c.0.to_string()).collect::<Vec<_>>().join(" / ")) };
    // the drafts a round keeps, Σ_j Π_{i ≤ j} hold_i: what a checkpoint is chosen by
    let drafts_held = |cnt: &[(u64, u64)]| -> f64 {
        let mut a = 1.0;
        rates(cnt).iter().map(|r| { a *= r; a }).sum()
    };
    let all: Vec<usize> = (0..wins.len()).collect();
    // the dump's own files: what a draft built elsewhere reads
    if let Some(d) = &cfg.dump {
        std::fs::create_dir_all(d)?;
        if !d.join("meta.json").exists() {
            let (mut fl, mut fi, mut fti, mut ftp) = (create(d, "len.u32")?, create(d, "ids.u32")?, create(d, "ti.u32")?, create(d, "tp.f32")?);
            for w in &wins {
                put(&mut fl, &[w.len as u32], u32::to_le_bytes)?;
                put(&mut fi, &w.ids, u32::to_le_bytes)?;
                put(&mut fti, &w.ti, u32::to_le_bytes)?;
                put(&mut ftp, &w.tp, f32::to_le_bytes)?;
            }
            for f in [&mut fl, &mut fi, &mut fti, &mut ftp] {
                f.flush()?;
            }
            let f16 = |name: &str, v: &[f32]| -> Result<()> {
                let mut f = create(d, name)?;
                put(&mut f, &to_f16(v), u16::to_le_bytes)?;
                Ok(f.flush()?)
            };
            let f32s = |name: &str, tensor: &str| -> Result<()> {
                let mut f = create(d, name)?;
                put(&mut f, &cmfio::tensor_f32(&student, student.tensor(tensor).with_context(|| format!("mtp heal: no {tensor}"))?)?, f32::to_le_bytes)?;
                Ok(f.flush()?)
            };
            f16("embed.f16", &embed)?;
            f16("eh_proj.f16", &eh_w)?;
            match student.tensor("lm_head.weight") {
                Some(e) => f16("head.f16", &cmfio::tensor_f32(&student, e)?)?,
                None => f16("head.f16", &embed)?,
            }
            f32s("enorm.f32", "model.mtp.enorm.weight")?;
            f32s("hnorm.f32", "model.mtp.hnorm.weight")?;
            f32s("mtp_norm.f32", "model.mtp.norm.weight")?;
            f32s("norm.f32", "model.norm.weight")?;
            let sets: Vec<_> = kept.iter().map(|(p, n)| json!({ "dir": p.display().to_string(), "windows": n })).collect();
            std::fs::write(
                d.join("meta.json"),
                serde_json::to_vec_pretty(&json!({
                    "windows": wins.len(), "window": t, "topk": k, "draft_topk": KK, "chain": chain, "hidden": hs, "vocab": embed.len() / hs,
                    "norm_eps": arch.rms_norm_eps, "gemma_norm": matches!(arch.norm_style, NormStyle::Gemma), "heldout": "window index % 10 == 9", "trunk": sets,
                }))?,
            )?;
        }
    }
    // the hold rates as the layout trained has them; a dump is always of every row's chain
    let hold_pass = |net: &nt::Net, w1: &nt::Work, wins: &[Win], dump: Option<PathBuf>| -> Result<Vec<(u64, u64)>> {
        match (dump, stride) {
            (Some(d), 0) => chain_pass(net, w1, wins, &all, Some(&d)),
            (Some(d), _) => {
                chain_pass(net, w1, wins, &all, Some(&d))?;
                round_pass(net, w1, wins, &val)
            }
            (None, 0) => chain_pass(net, w1, wins, &val, None),
            (None, _) => round_pass(net, w1, wins, &val),
        }
    };
    let h0 = hold_pass(&snet, &rw, &wins, cfg.dump.as_ref().map(|d| d.join("before")))?;
    eprintln!("  held-out rows where the draft's token is the trunk's, per chain step, before training: {}", show(&h0));
    if stride > 0 && sets.len() > 1 {
        // the same per data set: a chain holds on code and on prose differently
        let mut from = 0;
        for (name, ws) in &sets {
            let idx: Vec<usize> = val.iter().copied().filter(|i| (from..from + ws.len()).contains(i)).collect();
            from += ws.len();
            if !idx.is_empty() {
                let h = round_pass(&snet, &rw, &wins, &idx)?;
                eprintln!("    {name}: {}, {:.2} drafts a round", show(&h), drafts_held(&h));
            }
        }
    }
    if stride > chain && cfg.epochs == 0.0 {
        let h = order_pass(&snet, &rw, &wins, &val)?;
        eprintln!("  rounds in order, each starting after what the one before kept: {}, {:.2} drafts a round", show(&h), drafts_held(&h));
        let mut from = 0;
        for (name, ws) in &sets {
            let idx: Vec<usize> = val.iter().copied().filter(|i| (from..from + ws.len()).contains(i)).collect();
            from += ws.len();
            if sets.len() > 1 && !idx.is_empty() {
                let h = order_pass(&snet, &rw, &wins, &idx)?;
                eprintln!("    {name}: {}, {:.2} drafts a round", show(&h), drafts_held(&h));
            }
        }
    }

    if cfg.epochs > 0.0 {
        refresh(&mut wins, 0)?;
    }
    let batches = |set: &[usize]| -> Vec<Vec<usize>> { set.chunks(nb).filter(|b| b.len() == nb).map(<[usize]>::to_vec).collect() };
    let (tb, vb) = (batches(&train), batches(&val));
    // a batch of the input rows `src` whose row p predicts the token at
    // p + j + 1: rows p with p + j < len, the trunk's top-k at p + j
    let join = |wins: &[Win], b: &[usize], src: usize, j: usize, hard: f32| -> (Vec<u32>, Vec<f32>, Vec<u32>, Vec<f32>, f64) {
        let mut o = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0f64);
        for &i in b {
            let w = &wins[i];
            o.0.extend_from_slice(&w.ids);
            o.1.extend(from_f16(&w.h0[src]));
            let (mut tid, mut tp) = (vec![pad; t * k], vec![0f32; t * k]);
            let rows = w.len.saturating_sub(j);
            for p in 0..rows {
                tid[p * k..(p + 1) * k].copy_from_slice(&w.ti[(p + j) * k..(p + j + 1) * k]);
                let row = &mut tp[p * k..(p + 1) * k];
                row.copy_from_slice(&w.tp[(p + j) * k..(p + j + 1) * k]);
                if hard > 0.0 {
                    let s: f32 = row.iter().sum();
                    row.iter_mut().for_each(|x| *x *= 1.0 - hard);
                    row[0] += hard * s;
                }
            }
            o.2.extend(tid);
            o.3.extend(tp);
            o.4 += rows as f64;
        }
        o
    };
    // What a step trains on, (input rows, target shift): every chain step's
    // rows of its own, or the trunk's rows and the rows laid out as rounds.
    let kinds: Vec<(usize, usize)> = if stride > 0 && chain > 1 { vec![(0, 1), (1, 1)] } else { (1..=chain).map(|j| (j - 1, j)).collect() };
    // held-out KL to the trunk's distribution per kind
    let val_loss = |wins: &[Win], net: &nt::Net| -> Result<Vec<f64>> {
        kinds
            .iter()
            .map(|&(src, j)| {
                let (mut s, mut n) = (0f64, 0f64);
                for b in &vb {
                    let (ids, h0, tid, tp, tot) = join(wins, b, src, j, 0.0);
                    s += nt::loss_sum_h(c, net, &sw, &ids, Some(&h0), &tid, &tp, k, true).context("mtp heal: eval failed")?;
                    n += tot;
                }
                Ok(s / n.max(1e-9))
            })
            .collect()
    };
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(" / ");
    // without training there is nothing a KL is compared with: the hold rates are the measure
    let e0 = if cfg.epochs > 0.0 {
        let v0 = val_loss(&wins, &snet)?;
        let per = if stride > 0 && chain > 1 { "the trunk's rows / rows laid out as rounds" } else { "per chain step" };
        eprintln!("  held-out KL to the trunk before training, {per}: {}", fmt(&v0));
        mean(&v0)
    } else {
        f64::NAN
    };
    let recode = cfg.codec != "keep" || !cfg.also.is_empty();
    if cfg.epochs <= 0.0 && !recode {
        drop(snet);
        let _ = std::fs::remove_file(&dpath);
        return Ok(Report { steps: 0, best_step: 0, loss_start: e0, loss_best: e0, hold_start: rates(&h0), hold_best: rates(&h0) });
    }
    let mut best = (drafts_held(&h0), nt::snapshot(c, &snet, false).context("snapshot")?, 0usize, e0);
    let mut done = 0usize;
    if cfg.epochs > 0.0 {
        let jobs: Vec<(usize, usize)> = (0..tb.len()).flat_map(|b| (0..kinds.len()).map(move |i| (b, i))).collect();
        let total_steps = ((cfg.epochs * jobs.len() as f64).ceil() as usize).max(1);
        let warm = (total_steps / 10).max(1);
        let shape = |d: usize| -> f64 {
            if d <= warm {
                d as f64 / warm as f64
            } else {
                let pr = (d - warm) as f64 / (total_steps - warm).max(1) as f64;
                0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * pr.min(1.0)).cos())
            }
        };
        let (mut skipped, mut gscale) = (0usize, 4096f32);
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut order: Vec<usize> = (0..jobs.len()).collect();
        let t0 = Instant::now();
        'outer: loop {
            for i in (1..order.len()).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                order.swap(i, (rng % (i as u64 + 1)) as usize);
            }
            for &oi in &order {
                if done >= total_steps {
                    break 'outer;
                }
                let (bi, (src, j)) = (jobs[oi].0, kinds[jobs[oi].1]);
                let (ids, h0, tid, tp, tot) = join(&wins, &tb[bi], src, j, cfg.hard as f32);
                let st = nt::step_wh(c, &snet, &sw, &ids, Some(&h0), &tid, &tp, k, gscale, None, Some(tot)).context("mtp heal: step failed")?;
                let gn = nt::grad_norm(c, &snet).unwrap_or(f64::NAN);
                if !st.finite || !gn.is_finite() {
                    nt::zero_grads(c, &snet);
                    gscale = (gscale / 2.0).max(1.0);
                    skipped += 1;
                    ensure!(skipped < 20, "mtp heal: 20 non-finite steps");
                    continue;
                }
                done += 1;
                let lr = cfg.lr * shape(done);
                let gmul = if gn > 1.0 { (1.0 / gn) as f32 } else { 1.0 };
                nt::adam(c, &snet, lr as f32, done as u32, 0.9, 0.999, 1e-8, gmul).context("mtp heal: adam failed")?;
                let sps = t0.elapsed().as_secs_f64() / done as f64;
                eprint!("\r  step {done}/{total_steps}  loss {:.4}  lr {lr:.2e}  {sps:.2} s/step · ~{} left   ", st.loss, crate::fmt_eta(sps * (total_steps - done) as f64));
                if chain > 1 && cfg.refresh > 0 && done % cfg.refresh == 0 && done < total_steps {
                    eprintln!();
                    refresh(&mut wins, done / cfg.refresh)?;
                }
                if done % cfg.eval_every == 0 || done == total_steps {
                    let v = val_loss(&wins, &snet)?;
                    let hh = hold_pass(&snet, &rw, &wins, None)?;
                    let g = drafts_held(&hh);
                    eprintln!("\n  step {done}: held-out KL {}; the chain holds {}: {g:.4} drafts a round (best {:.4})", fmt(&v), show(&hh), best.0.max(g));
                    if g > best.0 {
                        best = (g, nt::snapshot(c, &snet, false).context("snapshot")?, done, mean(&v));
                    } else if cfg.patience > 0 && done - best.2 >= cfg.patience * cfg.eval_every && done < total_steps {
                        eprintln!("  no new best in {} evaluations: stopping at step {done} of {total_steps}", cfg.patience);
                        break 'outer;
                    }
                }
            }
        }
        nt::restore(c, &snet, &best.1);
    }
    let h1 = if done > 0 {
        let h = hold_pass(&snet, &rw, &wins, cfg.dump.as_ref().map(|d| d.join("after")))?;
        eprintln!("  held-out rows where the draft's token is the trunk's, per chain step, after training: {}", show(&h));
        h
    } else {
        h0.clone()
    };
    let trained: std::collections::HashMap<String, Vec<f32>> = nt::export(c, &snet, &dm)
        .map_err(|e| anyhow!("mtp heal: export: {e}"))?
        .into_iter()
        .map(|(name, bytes)| {
            let v: Vec<f32> = match dm.tensor(&name).map(|e| e.dtype) {
                Some(TensorDtype::F32) => bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
                _ => bytes.chunks_exact(2).map(|b| cortiq_core::quant::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
            };
            let back = if name == "model.norm.weight" { "model.mtp.norm.weight".to_string() } else { name.replacen("model.layers.0.", "model.mtp.layers.0.", 1) };
            (back, v)
        })
        .collect();
    drop(snet);
    let _ = std::fs::remove_file(&dpath);
    let prov = json!({
        "data": data_spec, "tokens": total, "windows": wins.len(), "chain": chain, "steps": done, "best_step": best.2,
        "lr": cfg.lr, "topk": k, "hard": cfg.hard, "stride": stride, "heldout_kl_start": e0, "heldout_kl_best": best.3,
        "heldout_hold_start": rates(&h0), "heldout_hold_best": rates(&h1),
        "init": init_path.map(|p| p.display().to_string()),
    });
    // the output, and the same draft in every other codec asked for; each
    // one's chain is run again as its file holds it
    for (codec, path) in std::iter::once((cfg.codec.clone(), out.to_path_buf())).chain(cfg.also.iter().cloned()) {
        let enc = match codec.as_str() {
            "keep" => Encoding::Keep,
            o => Encoding::Replace(crate::convert::parse_quant(o)?),
        };
        cmfio::rewrite(
            &student,
            &path,
            enc,
            &|name| Ok(trained.get(name).map(|v| (student.tensor(name).map(|e| e.shape.clone()).unwrap_or_default(), v.clone()))),
            &|h| cmfio::provenance_set(h, "mtp_heal", prov.clone()),
            threads,
        )?;
        let m = CmfModel::open_sharded(&path)?;
        let qpath = path.with_extension("draftq.cmf");
        draft_model(&m, &m, &qpath)?;
        let qm = CmfModel::open_sharded(&qpath)?;
        let qnet = nt::Net::load(c, &qm, true, false, None).map_err(|e| anyhow!("mtp heal: {}: draft: {e}", path.display()))?;
        let qw = nt::Work::new(c, &qnet, 1, t);
        let hq = hold_pass(&qnet, &qw, &wins, None)?;
        let bytes: usize = m.tensors.iter().filter(|e| e.name.starts_with("model.mtp.")).map(|e| m.entry_bytes(e).len()).sum();
        eprintln!("  {}: the draft as {codec}, {:.0} MB: {}", path.display(), bytes as f64 / 1048576.0, show(&hq));
        drop(qnet);
        let _ = std::fs::remove_file(&qpath);
    }
    Ok(Report { steps: done, best_step: best.2, loss_start: e0, loss_best: best.3, hold_start: rates(&h0), hold_best: rates(&h1) })
}
