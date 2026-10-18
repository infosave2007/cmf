//! `cortiq quantize-moe`: GPTQ for the experts of a large MoE whose
//! calibration forward runs in cortiq — Qwen3.8-Flash-Next (`qwen4_exp`) and
//! DeepSeek-V4.1 (`deepseek_v41`). Routed and shared experts are encoded in
//! one of two layouts, the ones the converter writes and the device expert
//! banks accept:
//!
//! * `--codec q2tp` (default): gate/up q2tp (placed rule, output debias),
//!   down q4tp — every expert slot, the shared expert included;
//! * `--codec q4tp`: gate/up/down q4tp.
//!
//! Every other tensor is copied from the calibration file (or left in it:
//! `--in-place`).
//!
//! * Calibration: the calibration model (a q4tp file of the same model) runs
//!   its forward over the corpus; each captured layer records its MoE input
//!   rows and each token's routed experts — Flash-Next inside its device
//!   frames (top-k of the router logits, `Pipeline::qwen4_capture`), V4.1 in
//!   its per-token forward right after the host route (sqrt-softplus scores
//!   plus the selection bias, `Pipeline::dsv41_moe_capture`). The shared
//!   expert sees every token.
//! * Per expert: the input Hessian of gate/up from its rows, the down input
//!   `silu(X·W_gᵀ)⊙(X·W_uᵀ)` (V4.1: `silu(min(g, L))⊙clamp(u, −L, L)`, its
//!   `swiglu_limit` L) from the full-precision weights and its Hessian, both
//!   accumulated in row chunks; then the folds, the gate/up sweeps and the
//!   down q4tp sweep. On the GPU (device grams, folds, sweeps); without a
//!   wgpu adapter the run stops, `CMF_MOE_HOST=1` runs the same steps on
//!   the host.
//! * Full-precision weights come from the original safetensors shards, one
//!   layer at a time (the next layer's in the background), deleted when done
//!   — or read in place when `--hf` names a local directory. Flash-Next's are
//!   bf16; V4.1's routed experts are FP4 (E2M1 + one E8M0 scale per 32) and
//!   its shared expert FP8 (E4M3 + E8M0 per 32×32 tile), decoded by the
//!   converter's own unpackers so the GPTQ target is the exact source.
//! * Output: streamed into `--output` (neither the source checkpoint nor the
//!   result is ever held whole), or `--in-place` into the calibration file's
//!   own expert slots, one layer at a time (q4tp payloads fit their q4tp
//!   slots exactly; q2tp ones are smaller and the slack is released to the
//!   filesystem on Linux), for disks that cannot hold the model twice.
//! * `--patch-shared FILE` re-encodes only the shared experts of an existing
//!   output in place (their payloads fit the old slots).

use crate::convert::{Quant, fetch, open_safetensors, quantize_2d, to_f32, unpack_fp8_blocks, unpack_mxfp4};
use crate::gptq::{CalibSource, Q2tpRule, calib_source_tokens, gptq_fold_operator_par, gptq_quantize_q2tp, gptq_quantize_q2tp_dev, gptq_quantize_q4tp, gptq_quantize_q4tp_dev, q2tp_debias_gemm, q2tp_debias_rows, split_calib_budget};
use cortiq_core::format::{CmfModel, CmfStreamWriter};
use cortiq_core::quant::{f16_to_f32, f32_to_f16};
use cortiq_core::types::TensorDtype;
use cortiq_engine::gpu_wgpu::{Ctx, train_mm as tm, train_ops as to};
use cortiq_engine::{Pipeline, SamplerConfig};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub struct MoeOpts<'a> {
    pub calib_model: &'a str,
    pub hf: &'a str,
    pub calib: &'a [String],
    pub tokens: usize,
    pub window: usize,
    pub layers_per_pass: usize,
    pub lambda: f64,
    pub min_rows: usize,
    pub scratch: &'a str,
    pub threads: usize,
    pub output: &'a str,
    pub patch_shared: Option<&'a str>,
    /// `q2tp` (gate/up q2tp, down q4tp) or `q4tp` (all three q4tp).
    pub codec: &'a str,
    /// Write the experts into the calibration file's own slots.
    pub in_place: bool,
}

/// Input rows per device pass of one expert.
const CHUNK: usize = 16384;

/// The expert layout of the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Codec {
    /// gate/up q2tp (placed rule, debiased), down q4tp.
    Q2tp,
    /// gate/up/down q4tp.
    Q4tp,
}

impl Codec {
    pub(crate) fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "q2tp" => Ok(Self::Q2tp),
            "q4tp" => Ok(Self::Q4tp),
            other => anyhow::bail!("--codec {other}: expected q2tp or q4tp"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Q2tp => "q2tp",
            Self::Q4tp => "q4tp",
        }
    }

    /// Dtype and converter quant of gate/up (down is always q4tp).
    fn gate_up(self) -> (TensorDtype, Quant) {
        match self {
            Self::Q2tp => (TensorDtype::Q2TiledP, Quant::Q2TiledP),
            Self::Q4tp => (TensorDtype::Q4TiledP, Quant::Q4TiledP),
        }
    }

    /// The dtypes of an expert's gate, up and down.
    pub(crate) fn dtypes(self) -> [TensorDtype; 3] {
        let gu = self.gate_up().0;
        [gu, gu, TensorDtype::Q4TiledP]
    }
}

/// One layer's calibration record: `n` MoE input rows (f16, `hidden` wide,
/// row-major), `top_k` routed expert ids per token (`u16::MAX` = none) and
/// each pick's route weight (V4.1; empty when not captured — Flash-Next).
pub(crate) struct LayerRows {
    pub x: Vec<u16>,
    pub picks: Vec<u16>,
    pub w: Vec<f32>,
}

/// Route weights in the down Hessian (`CMF_MOE_ROUTE_W`). The reference
/// expert multiplies its SwiGLU activation by the route weight before
/// `w2`, so `w2`'s true input Hessian is `Σ w_t²·y_t y_tᵀ`; gate/up read the
/// unscaled input either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RouteW {
    /// Unweighted rows (the default; Flash-Next always).
    Off,
    /// Each down-input row scaled by its route weight.
    On,
    /// `CMF_MOE_DIAG` only: both encodes, side by side.
    Diag,
}

impl RouteW {
    fn parse(v: Option<&str>) -> anyhow::Result<Self> {
        match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("0") => Ok(Self::Off),
            Some("1") => Ok(Self::On),
            Some("diag") => Ok(Self::Diag),
            Some(other) => anyhow::bail!("CMF_MOE_ROUTE_W={other}: expected 0, 1 or diag"),
        }
    }
}

/// The encodes a `CMF_MOE_DIAG` run measures per expert: each listed prior,
/// with the unweighted and/or the route-weighted down Hessian.
pub(crate) fn diag_variants(priors: &[f64], route_w: RouteW) -> Vec<(f64, bool)> {
    priors
        .iter()
        .flat_map(|&n0| match route_w {
            RouteW::Off => vec![(n0, false)],
            RouteW::On => vec![(n0, true)],
            RouteW::Diag => vec![(n0, false), (n0, true)],
        })
        .collect()
}

/// Full-precision weights of one expert: gate and up `[inter, hidden]`,
/// down `[hidden, inter]`, row-major.
pub(crate) struct ExpertW {
    pub g: Vec<f32>,
    pub u: Vec<f32>,
    pub d: Vec<f32>,
}

/// How an expert is encoded.
#[derive(Clone, Copy)]
pub(crate) struct EncodeCfg {
    pub codec: Codec,
    /// DeepSeek `swiglu_limit` (0 = none): the down input is
    /// `silu(min(g, L))·clamp(u, −L, L)`, as in the forward.
    pub limit: f32,
    pub lambda: f64,
    pub threads: usize,
}

/// Where the per-expert linear algebra runs.
#[derive(Clone, Copy)]
pub(crate) enum Engine<'a> {
    Gpu(&'a Ctx),
    Host,
}

/// The model family of the calibration file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Qwen4,
    Dsv41,
}

pub fn run(o: &MoeOpts) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let codec = Codec::parse(o.codec)?;
    anyhow::ensure!(!(o.in_place && !o.output.is_empty()), "--output and --in-place are exclusive (--in-place writes into the calibration file)");
    // The host path is exact but far too slow for a full model: it runs
    // only when asked for, never as a silent fallback.
    let host_forced = std::env::var("CMF_MOE_HOST").is_ok_and(|v| v != "0");
    let engine = if host_forced {
        eprintln!("quantize-moe: CMF_MOE_HOST=1 — Hessians, folds and sweeps run on the host (exact, much slower)");
        Engine::Host
    } else {
        match to::dev() {
            // the device grams and GEMMs run on the matrix units
            // (cooperative matrices): without them the first expert would
            // fail after the whole capture
            Some(_) if !tm::matrix_units() => {
                anyhow::bail!("quantize-moe: the wgpu device has no matrix units (cooperative matrices) for the device encode (set CMF_MOE_HOST=1 to run on the host)")
            }
            Some(c) => Engine::Gpu(c),
            None => anyhow::bail!("quantize-moe: no wgpu GPU (set CMF_MOE_HOST=1 to run on the host)"),
        }
    };
    // q2tp in place releases each slot's slack by hole punching: probe the
    // filesystem now, not after the capture
    if o.in_place && codec == Codec::Q2tp && std::env::var_os("CMF_MOE_DIAG").is_none() {
        if no_punch() {
            eprintln!("quantize-moe: CMF_MOE_NO_PUNCH=1 — the q2tp slack stays allocated (the file keeps its q4tp size on disk until `cortiq compact`)");
        } else {
            let dir = Path::new(o.calib_model).parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
            if let Err(why) = probe_hole_punching(dir) {
                anyhow::bail!(
                    "--in-place --codec q2tp releases each expert slot's slack by hole punching, and {} does not: {why}. \
                     Set CMF_MOE_NO_PUNCH=1 to run anyway (the slack, about a third of the expert bytes, then stays allocated until `cortiq compact`)",
                    dir.display()
                );
            }
            eprintln!("quantize-moe: hole punching works on {}", dir.display());
        }
    }
    let cmodel = Arc::new(CmfModel::open_sharded(o.calib_model)?);
    // a new file is opened with truncation: never the calibration model
    // (mapped, read by the forward, the input of a later --in-place run)
    for f in calib_files(o.calib_model, &cmodel) {
        if !o.output.is_empty() && same_file(Path::new(o.output), &f) {
            anyhow::bail!("--output {} is the calibration file {} (writing it would truncate it): write to another path, or use --in-place", o.output, f.display());
        }
        if let Some(t) = o.patch_shared.filter(|t| same_file(Path::new(t), &f)) {
            anyhow::bail!("--patch-shared {t} is the calibration file {}: it patches an existing quantize-moe output (--in-place rewrites the calibration file)", f.display());
        }
    }
    let arch = cmodel.arch().clone();
    let nl = arch.num_layers;
    let hidden = arch.hidden_size;
    let mut pipe = Some(Pipeline::from_model(&cmodel, SamplerConfig::default()).map_err(|e| anyhow::anyhow!("{e}"))?);
    {
        let p = pipe.as_mut().expect("pipeline");
        p.set_confidence(false);
    }
    let p = pipe.as_ref().expect("pipeline");
    let family = if p.dsv41.is_some() {
        Family::Dsv41
    } else if p.qwen4_expert_dirs(0).is_some() {
        Family::Qwen4
    } else {
        anyhow::bail!("quantize-moe: {} is neither a qwen4_exp (Flash-Next-class) nor a deepseek_v41 model", o.calib_model);
    };
    // expert directory triples per layer, top-k, swiglu limit
    let mut layer_dirs: Vec<Vec<(usize, usize, usize)>> = Vec::with_capacity(nl);
    let mut shared_dirs: Vec<Option<(usize, usize, usize)>> = Vec::with_capacity(nl);
    let (top_k, limit, fp8_block) = match family {
        Family::Qwen4 => {
            for li in 0..nl {
                let (d, s) = p.qwen4_expert_dirs(li).ok_or_else(|| anyhow::anyhow!("layer {li}: no experts"))?;
                layer_dirs.push(d);
                shared_dirs.push(s);
            }
            (qwen4_top_k(&arch), 0.0f32, 0usize)
        }
        Family::Dsv41 => {
            let cfg = &p.dsv41.as_ref().expect("dsv41").2;
            anyhow::ensure!(cfg.dim == hidden, "deepseek_v41: runtime dim {} vs header hidden {hidden}", cfg.dim);
            for li in 0..nl {
                let (d, s) = dsv41_dirs(&cmodel, li, cfg.n_routed_experts)?;
                layer_dirs.push(d);
                shared_dirs.push(Some(s));
            }
            dsv41_settings(cfg, arch.deepseek_v41.as_ref())
        }
    };
    let ne = layer_dirs[0].len();
    anyhow::ensure!(ne > 0 && ne <= u16::MAX as usize && top_k > 0 && top_k <= ne, "{ne} experts, top-{top_k}: unsupported");
    let inter = cmodel.tensors[layer_dirs[0][0].0].shape[0];
    let inter_s = shared_dirs.iter().flatten().map(|&(g, _, _)| cmodel.tensors[g].shape[0]).max().unwrap_or(0);
    anyhow::ensure!(
        hidden % 32 == 0 && inter % 32 == 0 && inter_s % 32 == 0,
        "hidden {hidden} / expert width {inter} / shared width {inter_s} must be multiples of 32 (the q4tp/q2tp group)"
    );
    let patch = o.patch_shared;
    anyhow::ensure!(!(o.in_place && patch.is_some()), "--in-place and --patch-shared are exclusive");
    eprintln!(
        "quantize-moe ({family:?}, codec {}): {nl} layers, {ne} experts, top-{top_k}, hidden {hidden}, expert width {inter}{}{}{}",
        codec.name(),
        if limit > 0.0 { format!(", swiglu limit {limit}") } else { String::new() },
        if patch.is_some() { " — shared experts only, patched in place" } else { "" },
        if o.in_place { " — written into the calibration file in place" } else { "" },
    );
    if let Some(src) = cmodel.header.provenance.as_ref().and_then(|p| p.get("source_model")).and_then(|v| v.as_str()) {
        eprintln!("calibration file converted from {src}; expert weights from {}", o.hf);
    }

    // calibration windows
    let srcs: Vec<CalibSource> = o.calib.iter().map(|s| CalibSource::parse(s)).collect::<anyhow::Result<_>>()?;
    let budgets = split_calib_budget(&srcs, o.tokens);
    // V4.1: every window is a sequence of its own from position 0, so it
    // starts with BOS as every request does (not mid-text at position 0)
    let bos = match family {
        Family::Dsv41 => Some(p.tokenizer.bos_token_id.unwrap_or(0)),
        Family::Qwen4 => None,
    };
    let mut windows: Vec<Vec<u32>> = Vec::new();
    for (src, &n) in srcs.iter().zip(&budgets) {
        let ids = calib_source_tokens(&src.path, n, &|t| p.tokenizer.encode(t))?;
        windows.extend(calib_windows(&ids, o.window, bos));
    }
    if let Some(b) = bos {
        eprintln!("every calibration window starts with BOS (id {b})");
    }
    let ntok: usize = windows.iter().map(Vec::len).sum();
    eprintln!("calibrating on {ntok} tokens ({} windows)", windows.len());
    // CMF_MOE_DIAG="0,2560,…": per-expert output error on held-out windows
    // (every CMF_MOE_HOLDOUT-th, default 8th), round-to-nearest against GPTQ
    // with each listed prior (pseudo-rows, 0 = none); no file is written.
    // CMF_MOE_PRIOR = n0: the prior the production encode uses (n0/4 for
    // down), from CMF_MOE_PRIOR_S (8192) evenly spaced calibration tokens.
    // CMF_MOE_RANGE = lo..hi restricts the layers.
    let env_f = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let diag: Option<Vec<f64>> = std::env::var("CMF_MOE_DIAG").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
    let holdout = if diag.is_some() { env_f("CMF_MOE_HOLDOUT", 8.0) as usize } else { 0 };
    let prior_s = env_f("CMF_MOE_PRIOR_S", 8192.0) as usize;
    let mut held: Vec<bool> = Vec::with_capacity(ntok);
    for (wi, w) in windows.iter().enumerate() {
        held.extend(std::iter::repeat_n(holdout > 0 && wi % holdout == holdout - 1, w.len()));
    }
    let (range_lo, range_hi) = std::env::var("CMF_MOE_RANGE")
        .ok()
        .and_then(|v| v.split_once("..").and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?))))
        .unwrap_or((0, nl));
    let (range_lo, range_hi) = (range_lo.min(nl), range_hi.min(nl));
    anyhow::ensure!(range_lo < range_hi, "CMF_MOE_RANGE {range_lo}..{range_hi} is empty");
    // CMF_MOE_CAPTURE_ONLY=1: one calibration forward over the layers into
    // CMF_MOE_CAPTURE_DIR, then stop (no expert is encoded, no file written)
    let capture_only = std::env::var("CMF_MOE_CAPTURE_ONLY").is_ok_and(|v| v != "0");
    if capture_only {
        anyhow::ensure!(std::env::var_os("CMF_MOE_CAPTURE_DIR").is_some(), "CMF_MOE_CAPTURE_ONLY=1 needs CMF_MOE_CAPTURE_DIR");
        anyhow::ensure!(!o.in_place && patch.is_none() && diag.is_none(), "CMF_MOE_CAPTURE_ONLY=1 writes nothing: run it without --in-place, --patch-shared and CMF_MOE_DIAG");
        eprintln!("CMF_MOE_CAPTURE_ONLY=1: layers {range_lo}..{range_hi} are captured in one forward and saved; nothing is encoded or written");
    }
    anyhow::ensure!(
        capture_only || diag.is_some() || o.in_place || patch.is_some() || (range_lo, range_hi) == (0, nl),
        "CMF_MOE_RANGE {range_lo}..{range_hi} with --output would leave the other layers' experts out of the file \
         (use it with --in-place, --patch-shared or CMF_MOE_DIAG)"
    );
    if let Some(v) = &diag {
        eprintln!("DIAG: held-out every {holdout}th window, priors {v:?} (pseudo-rows), layers {range_lo}..{range_hi}; no file is written");
    }
    let n_cal = held.iter().filter(|&&h| !h).count();
    let prior_env = std::env::var("CMF_MOE_PRIOR").ok().and_then(|v| v.trim().parse::<f64>().ok());
    let (prior_n0, prior_why) = production_prior(family, prior_env, n_cal, top_k, ne, hidden);
    if diag.is_some() {
        // DIAG measures its own list of priors
    } else if prior_n0 > 0.0 {
        eprintln!("prior: n0 = {prior_n0} pseudo-rows (down {}) from {prior_s} calibration tokens — {prior_why}", prior_n0 / 4.0);
    } else {
        eprintln!("prior: none — {prior_why}");
    }
    let route_w = RouteW::parse(std::env::var("CMF_MOE_ROUTE_W").ok().as_deref())?;
    anyhow::ensure!(route_w == RouteW::Off || family == Family::Dsv41, "CMF_MOE_ROUTE_W: route weights are captured for DeepSeek-V4.1 only");
    anyhow::ensure!(route_w != RouteW::Diag || diag.is_some(), "CMF_MOE_ROUTE_W=diag compares both down Hessians under CMF_MOE_DIAG only");
    match route_w {
        RouteW::Off => eprintln!("down Hessian: unweighted rows (CMF_MOE_ROUTE_W=1 weights them by the route weight)"),
        RouteW::On => eprintln!("down Hessian: each row scaled by its route weight (Σ w²·y yᵀ, CMF_MOE_ROUTE_W=1)"),
        RouteW::Diag => eprintln!("DIAG: every prior with the unweighted and the route-weighted down Hessian (CMF_MOE_ROUTE_W=diag)"),
    }
    let variants = diag.as_deref().map(|v| diag_variants(v, route_w)).unwrap_or_default();
    let labels: Vec<String> = variants.iter().map(|&(n0, rw)| format!("{n0}{}", if rw { "+w" } else { "" })).collect();
    let nvar = variants.len();
    let mut diag_tot = vec![DiagBin::new(nvar); BINS.len()];

    // full-precision source
    let source = Source::open(o.hf, o.scratch)?;
    let shard_names = |li: usize| -> anyhow::Result<Vec<String>> {
        let mut names = Vec::new();
        match family {
            Family::Qwen4 => {
                if patch.is_none() {
                    names.extend(QWEN4_ROUTED.iter().map(|p| qwen4_src_name(&source, li, p)).collect::<anyhow::Result<Vec<_>>>()?);
                }
                if shared_dirs[li].is_some() {
                    names.extend(QWEN4_SHARED.iter().map(|p| qwen4_src_name(&source, li, p)).collect::<anyhow::Result<Vec<_>>>()?);
                }
            }
            Family::Dsv41 => names = dsv41_src_names(&source, li, ne, patch.is_none()),
        }
        source.shards_of(&names)
    };

    // output (the sink itself is created after the checks below: a new
    // file is opened with truncation)
    let writes = !(diag.is_some() || capture_only);
    let patch_target = match patch {
        Some(path) if writes => Some(CmfModel::open_sharded(path)?),
        _ => None,
    };
    let in_place = writes && patch.is_none() && o.in_place;
    let stream = writes && patch.is_none() && !o.in_place;
    if in_place {
        anyhow::ensure!(
            cmodel.tensors.iter().all(|t| t.shard == 0),
            "--in-place needs a single-file calibration model ({} has sibling shards)",
            o.calib_model
        );
    }
    if stream {
        anyhow::ensure!(!o.output.is_empty(), "--output is required (or --in-place / --patch-shared)");
    }
    let expert_slots: HashSet<usize> = layer_dirs.iter().flatten().chain(shared_dirs.iter().flatten()).flat_map(|&(g, u, d)| [g, u, d]).collect();
    // every slot this run rewrites must take the codec's payload: checked
    // now, not hours later when the first layer is written
    let written: Vec<(usize, usize, usize)> = (range_lo..range_hi)
        .flat_map(|li| if patch.is_none() { layer_dirs[li].clone() } else { Vec::new() }.into_iter().chain(shared_dirs[li]))
        .collect();
    if in_place {
        check_slots(&cmodel, &written, codec, None)?;
    } else if let (Some(path), Some(target)) = (patch, patch_target.as_ref()) {
        check_slots(&cmodel, &written, codec, Some((path, target)))?;
    }

    let ec = EncodeCfg { codec, limit, lambda: o.lambda, threads: o.threads.max(1) };
    let main_bufs = match engine {
        Engine::Gpu(c) => Some(wgpu_bufs::Bufs::new(c, CHUNK, hidden, inter.max(inter_s))),
        Engine::Host => None,
    };
    let mut stats = (0usize, 0usize);
    let mut prefetch: Option<std::thread::JoinHandle<anyhow::Result<()>>> = None;
    let passes = plan_passes(range_lo, range_hi, o.layers_per_pass, o.in_place || capture_only);
    // CMF_MOE_CAPTURE_DIR: captured rows are saved per layer and reused by
    // a later run on the same calibration tokens (another codec, a resume)
    let has_w = family == Family::Dsv41;
    let mut saved: Option<(PathBuf, CaptureMeta)> = match std::env::var_os("CMF_MOE_CAPTURE_DIR") {
        Some(d) => {
            let want = CaptureMeta {
                token_hash: token_hash(&windows),
                ntok,
                window: o.window,
                windows: windows.len(),
                hidden,
                top_k,
                ne,
                family: format!("{family:?}"),
                calib_model: o.calib_model.to_string(),
                calib_size: std::fs::metadata(o.calib_model).map(|m| m.len()).unwrap_or(0),
                has_w,
                layers: Vec::new(),
            };
            let dir = PathBuf::from(d);
            let meta = capture_open(&dir, &want)?;
            eprintln!(
                "capture directory {}: {} of the layers {range_lo}..{range_hi} saved, {:.2} GB per layer",
                dir.display(),
                (range_lo..range_hi).filter(|li| meta.layers.contains(li)).count(),
                meta.layer_bytes() as f64 / 1e9
            );
            Some((dir, meta))
        }
        None => None,
    };
    // free disk space for what this run will write, checked now: an --output
    // file gets its last ~200 GB (the non-expert tensors) at the very end
    {
        let mut needs: Vec<(PathBuf, String, u64)> = Vec::new();
        let mut credit: Vec<(PathBuf, u64)> = Vec::new();
        if stream {
            let gu = codec.gate_up().0;
            let slot_dt: HashMap<usize, TensorDtype> =
                layer_dirs.iter().flatten().chain(shared_dirs.iter().flatten()).flat_map(|&(g, u, d)| [(g, gu), (u, gu), (d, TensorDtype::Q4TiledP)]).collect();
            needs.push((PathBuf::from(o.output), format!("--output {}", o.output), stream_output_bytes(&cmodel, &slot_dt)?));
            // an earlier run's output is truncated when the new one opens it
            credit.push((PathBuf::from(o.output), sole_file_bytes(Path::new(o.output))));
        }
        if !capture_only && source.local.is_none() {
            // the layer being encoded and the next one, prefetched
            let first = shard_names(range_lo)?;
            match source.remote_sizes(&first) {
                Some(b) => {
                    let present: u64 = first.iter().map(|sh| std::fs::metadata(source.path(sh)).map(|m| m.len()).unwrap_or(0)).sum();
                    needs.push((source.scratch.clone(), format!("source shards in {}", source.scratch.display()), (2 * b).saturating_sub(present)));
                }
                None => eprintln!("free space: the size of the source shards is unknown (no size from the Hub); not counted"),
            }
        }
        if let Some((dir, meta)) = &saved {
            let left = (range_lo..range_hi).filter(|li| !meta.layers.contains(li)).count() as u64;
            needs.push((dir.clone(), format!("capture in {}", dir.display()), left * meta.layer_bytes()));
        }
        check_free_space(&needs, &credit, std::env::var("CMF_MOE_NO_SPACE_CHECK").is_ok_and(|v| v != "0"))?;
    }
    let mut sink = if !writes {
        Sink::Diag
    } else if let (Some(path), Some(target)) = (patch, patch_target) {
        Sink::Patch { path: path.to_string(), target, patches: Vec::new() }
    } else if in_place {
        Sink::InPlace { path: o.calib_model.to_string(), pending: Vec::new(), before: 0, after: 0, punched: 0 }
    } else {
        let avg_name = cmodel.tensors.iter().map(|e| e.name.len()).sum::<usize>() / cmodel.tensors.len().max(1);
        Sink::Stream(CmfStreamWriter::new(o.output, CmfStreamWriter::head_reserve_for(cmodel.tensors.len(), avg_name))?)
    };
    let is_saved =|saved: &Option<(PathBuf, CaptureMeta)>, lo: usize, hi: usize| saved.as_ref().is_some_and(|(_, m)| (lo..hi).all(|li| m.layers.contains(&li)));
    // the calibration forward is needed up to the last pass not on disk
    let last_forward = passes.iter().rposition(|&(lo, hi)| !is_saved(&saved, lo, hi));
    if last_forward.is_none() {
        eprintln!("every layer's capture is on disk: no calibration forward");
        pipe = None;
        cortiq_engine::gpu_wgpu::release_model_buffers(&cmodel);
    }
    for (pi, &(lo, hi)) in passes.iter().enumerate() {
        eprintln!(
            "  layers {lo}..{hi}: capture holds ~{:.1} GB of input rows",
            ((hi - lo) * ntok * (hidden * 2 + top_k * 2 + if has_w { top_k * 4 } else { 0 })) as f64 / 1e9
        );
        let tc = Instant::now();
        let rows: Vec<LayerRows> = if is_saved(&saved, lo, hi) {
            let (dir, meta) = saved.as_ref().expect("saved");
            let rows = (lo..hi).map(|li| capture_load_layer(dir, meta, li)).collect::<anyhow::Result<Vec<_>>>()?;
            eprintln!("  layers {lo}..{hi}: loaded from {} in {:.0}s", dir.display(), tc.elapsed().as_secs_f64());
            rows
        } else {
            // ── capture ──
            let nw = if has_w { ntok * top_k } else { 0 };
            let mut rows: Vec<LayerRows> = (lo..hi).map(|_| LayerRows { x: Vec::with_capacity(ntok * hidden), picks: Vec::with_capacity(ntok * top_k), w: Vec::with_capacity(nw) }).collect();
            {
                let p = pipe.as_mut().expect("pipeline alive while layers remain");
                for (wi, w) in windows.iter().enumerate() {
                    match family {
                        Family::Qwen4 => {
                            let cap = p.qwen4_capture(w, lo, hi).map_err(|e| anyhow::anyhow!("capture: {e}"))?;
                            for (k, r) in cap.into_iter().enumerate() {
                                qwen4_layer_rows(&r, hidden, ne, top_k, &mut rows[k]);
                            }
                        }
                        Family::Dsv41 => {
                            let cap = p.dsv41_moe_capture(w, lo, hi).map_err(|e| anyhow::anyhow!("capture: {e}"))?;
                            for (k, ((x, pk), w)) in cap.x.into_iter().zip(cap.picks).zip(cap.w).enumerate() {
                                rows[k].x.extend(x);
                                rows[k].picks.extend(pk);
                                rows[k].w.extend(w);
                            }
                        }
                    }
                    eprint!(
                        "\r  layers {lo}..{hi}: captured {}/{} windows  {:.1} tok/s   ",
                        wi + 1,
                        windows.len(),
                        rows[0].x.len() as f64 / hidden as f64 / tc.elapsed().as_secs_f64()
                    );
                }
            }
            eprintln!();
            if Some(pi) == last_forward {
                // the last forward: the model's device state is not needed any
                // more (its expert bank and cached weights live in the device
                // context, keyed by the model, and outlive the pipeline unless
                // released)
                pipe = None;
                cortiq_engine::gpu_wgpu::release_model_buffers(&cmodel);
            }
            if let Some((dir, meta)) = saved.as_mut() {
                eprintln!("  layers {lo}..{hi}: saving the capture to {} ({:.1} GB)", dir.display(), ((hi - lo) as u64 * meta.layer_bytes()) as f64 / 1e9);
                for (k, li) in (lo..hi).enumerate() {
                    capture_save_layer(dir, meta, li, &rows[k])?;
                    if !meta.layers.contains(&li) {
                        meta.layers.push(li);
                    }
                }
                meta.layers.sort_unstable();
                capture_save_meta(dir, meta)?;
            }
            rows
        };
        if capture_only {
            continue;
        }
        for li in lo..hi {
            let tl = Instant::now();
            let lr = &rows[li - lo];
            let n = lr.x.len() / hidden;
            anyhow::ensure!(lr.picks.len() == n * top_k && held.len() == n, "layer {li}: {n} rows, {} routes, {} tokens", lr.picks.len() / top_k, held.len());
            // full-precision weights of this layer (the next layer's shards
            // download in the background meanwhile)
            let tf = Instant::now();
            // a failed background download costs a synchronous one, never
            // the run (the capture before it can take hours)
            match prefetch.take().map(|h| h.join()) {
                None | Some(Ok(Ok(()))) => {}
                Some(Ok(Err(e))) => eprintln!("  layer {li}: background shard prefetch failed ({e:#}); fetching the shards again now"),
                Some(Err(_)) => eprintln!("  layer {li}: background shard prefetch panicked; fetching the shards again now"),
            }
            let shards = shard_names(li)?;
            let src = source.open_layer(&shards)?;
            if li + 1 < range_hi {
                prefetch = source.prefetch(shard_names(li + 1)?);
            }
            let t_fetch = tf.elapsed().as_secs_f64();
            let get = |part: Part, e: usize| -> anyhow::Result<ExpertW> {
                match family {
                    Family::Qwen4 => qwen4_expert(&source, &src, li, part, e, hidden, inter, inter_s),
                    Family::Dsv41 => dsv41_expert(&src, li, part, e, hidden, if matches!(part, Part::Shared) { inter_s } else { inter }, fp8_block),
                }
            };
            let job = LayerJob { li, hidden, ne, inter, top_k, min_rows: o.min_rows, prior_n0, prior_s, held: &held, route_w };
            let out = encode_layer(engine, &ec, &job, lr, patch.is_none(), shared_dirs[li].is_some() && diag.is_none(), main_bufs.as_ref(), diag.as_deref(), &get)?;
            let weighted_metric = !lr.w.is_empty();
            stats.0 += out.n_gptq;
            stats.1 += out.n_rtn;
            if diag.is_some() {
                for (bi, b) in out.diag.iter().enumerate() {
                    if b.count == 0 {
                        continue;
                    }
                    eprintln!("  DIAG layer {li} rows<{}: {} experts, rel. output error RTN {} | GPTQ {labels:?}: {}", bin_label(bi), b.count, b.rel(0), b.rels(nvar));
                    if weighted_metric {
                        eprintln!("  DIAG layer {li} rows<{}: route-weighted error (Σ w²·|Δ|²) RTN {} | GPTQ {labels:?}: {}", bin_label(bi), b.rel_w(0), b.rels_w(nvar));
                    }
                    diag_tot[bi].add(b);
                }
            }
            let (t_shared, tim) = (out.t_shared, out.tim);
            sink.emit_layer(&cmodel, codec, &layer_dirs[li], shared_dirs[li], out)?;
            // shards this layer was the last user of
            let keep: Vec<String> = if li + 1 < range_hi { shard_names(li + 1)? } else { Vec::new() };
            drop(src);
            for sh in &shards {
                if !keep.contains(sh) {
                    source.release(sh);
                }
            }
            eprintln!(
                "  layer {li}: {n} tokens, experts GPTQ {} / RTN {} so far, {:.0}s ({:.0}s total); fetch wait {t_fetch:.1}s, inputs {:.1} fold {:.1} gate/up {:.1} down fold {:.1} q4tp {:.1}, shared {:.1}s",
                stats.0,
                stats.1,
                tl.elapsed().as_secs_f64(),
                t0.elapsed().as_secs_f64(),
                tim[0],
                tim[1],
                tim[2],
                tim[3],
                tim[4],
                t_shared
            );
            if std::env::var("CMF_MOE_STOP_AFTER").ok().and_then(|v| v.parse::<usize>().ok()) == Some(li + 1) {
                anyhow::bail!(
                    "CMF_MOE_STOP_AFTER={}: stopped {}",
                    li + 1,
                    if o.in_place { "(layers done so far are already rewritten in place)" } else { "before writing the file" }
                );
            }
        }
    }
    if capture_only {
        eprintln!("capture of layers {range_lo}..{range_hi} saved in {:.0}s; no expert was encoded", t0.elapsed().as_secs_f64());
        return Ok(());
    }
    if diag.is_some() {
        eprintln!("DIAG total, layers {range_lo}..{range_hi}, relative expert output error on held-out rows (GPTQ variants: prior n0, +w = route-weighted down Hessian):");
        for (bi, b) in diag_tot.iter().enumerate() {
            if b.count == 0 {
                continue;
            }
            eprintln!("  rows<{}: {} experts | RTN {} | GPTQ {labels:?}: {}", bin_label(bi), b.count, b.rel(0), b.rels(nvar));
            if b.base_w > 0.0 {
                eprintln!("  rows<{}: route-weighted | RTN {} | GPTQ {labels:?}: {}", bin_label(bi), b.rel_w(0), b.rels_w(nvar));
            }
        }
        return Ok(());
    }
    let prov_info = serde_json::json!({
        "family": format!("{family:?}"),
        "experts": match codec {
            Codec::Q2tp => "routed and shared: gate/up q2tp (placed, debiased), down q4tp",
            Codec::Q4tp => "routed and shared: gate/up/down q4tp",
        },
        "codec": codec.name(),
        "swiglu_limit": limit,
        "calibration_tokens": ntok,
        "window": o.window,
        "lambda": o.lambda,
        "prior": prior_n0,
        "route_weighted_down_hessian": route_w == RouteW::On,
        "prior_source": prior_why,
        "prior_tokens": prior_s,
        "source": o.hf,
        "calibration_model": Path::new(o.calib_model).file_name().map(|f| f.to_string_lossy().into_owned()),
        "experts_gptq": stats.0,
        "experts_rtn": stats.1,
        "min_rows": o.min_rows,
    });
    sink.finish(&cmodel, &expert_slots, codec, prov_info, o.output, t0)
}

impl Sink {
    /// One layer's encoded experts into their slots (`dirs`: the routed
    /// experts' gate/up/down directory indices, `shared` the shared
    /// expert's), flushed to disk in place.
    fn emit_layer(&mut self, cm: &CmfModel, codec: Codec, dirs: &[(usize, usize, usize)], shared: Option<(usize, usize, usize)>, out: LayerOut) -> anyhow::Result<()> {
        let dts = codec.dtypes();
        anyhow::ensure!(out.routed.is_empty() || out.routed.len() == dirs.len(), "{} encoded experts for {} slots", out.routed.len(), dirs.len());
        for (enc, &(gi, ui, di)) in out.routed.into_iter().zip(dirs) {
            for ((slot, bytes), dt) in [gi, ui, di].into_iter().zip(enc).zip(dts) {
                self.emit(cm, slot, dt, bytes)?;
            }
        }
        if let (Some(enc), Some((gi, ui, di))) = (out.shared, shared) {
            for ((slot, bytes), dt) in [gi, ui, di].into_iter().zip(enc).zip(dts) {
                self.emit(cm, slot, dt, bytes)?;
            }
        }
        self.flush_layer(cm)
    }

    /// Complete the output: copy every non-expert tensor and write the
    /// header (a new file), or apply the collected patches (`--patch-shared`).
    fn finish(self, cmodel: &CmfModel, expert_slots: &HashSet<usize>, codec: Codec, prov_info: serde_json::Value, output: &str, t0: Instant) -> anyhow::Result<()> {
        match self {
            Sink::Diag => Ok(()),
            Sink::Patch { path, target, patches } => {
                drop(target);
                let n = patches.len();
                CmfModel::recode_entries_in_place(&path, &patches).map_err(|e| anyhow::anyhow!("patch {path}: {e}"))?;
                eprintln!("patched {n} shared-expert tensors in place in {path} in {:.0}s", t0.elapsed().as_secs_f64());
                Ok(())
            }
            Sink::InPlace { path, before, after, punched, .. } => {
                eprintln!(
                    "rewrote expert tensors in place in {path}: payload {:.2} GB → {:.2} GB, {:.2} GB of slack released; {:.0}s. \
                     The header provenance is the calibration file's ({prov_info})",
                    before as f64 / 1e9,
                    after as f64 / 1e9,
                    punched as f64 / 1e9,
                    t0.elapsed().as_secs_f64()
                );
                Ok(())
            }
            Sink::Stream(mut out) => {
                for (slot, e) in cmodel.tensors.iter().enumerate() {
                    if !expert_slots.contains(&slot) {
                        out.push_bounded(&e.name, e.dtype, &e.shape, cmodel.entry_bytes(e), 1 << 28)?;
                    }
                }
                let mut header = cmodel.header.clone();
                header.section_hashes = None;
                let mut prov = header.provenance.clone().unwrap_or_else(|| serde_json::json!({}));
                prov["weight_quant"] = serde_json::json!(codec.name());
                prov["gptq_moe"] = prov_info;
                header.provenance = Some(prov);
                out.finish(&header, None, cmodel.vocab.as_deref())?;
                eprintln!("wrote {output} in {:.0}s", t0.elapsed().as_secs_f64());
                Ok(())
            }
        }
    }
}

/// Flash-Next top-k from the header (10 when it has no `moe` block).
fn qwen4_top_k(arch: &cortiq_core::types::ModelArch) -> usize {
    arch.moe.as_ref().map(|m| m.top_k).unwrap_or(10)
}

/// A V4.1 model's top-k and SwiGLU limit (the runtime config; a negative
/// limit is none) and its FP8 weight block, from the preserved source
/// config's `quantization_config.weight_block_size` (32 when absent).
fn dsv41_settings(cfg: &cortiq_engine::dsv41::Dsv41Cfg, source: Option<&serde_json::Value>) -> (usize, f32, usize) {
    let block = source
        .and_then(|c| c.get("quantization_config"))
        .and_then(|q| q.get("weight_block_size"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.first()?.as_u64())
        .map(|b| b as usize)
        .unwrap_or(32);
    (cfg.top_k, cfg.swiglu_limit.max(0.0), block)
}

/// One source's calibration ids cut into windows of `window` tokens; with
/// `bos` each window starts with it and carries `window − 1` text tokens.
/// Windows under 16 tokens are dropped.
fn calib_windows(ids: &[u32], window: usize, bos: Option<u32>) -> Vec<Vec<u32>> {
    match bos {
        None => ids.chunks(window.max(1)).filter(|w| w.len() >= 16).map(<[u32]>::to_vec).collect(),
        Some(b) => ids
            .chunks(window.max(2) - 1)
            .map(|c| {
                let mut w = Vec::with_capacity(c.len() + 1);
                if c.first() != Some(&b) {
                    w.push(b);
                }
                w.extend_from_slice(c);
                w
            })
            .filter(|w| w.len() >= 16)
            .collect(),
    }
}

/// The layer prior of the production encode (`n0` pseudo-rows, 0 = none)
/// and why: `CMF_MOE_PRIOR` when set (0 turns it off); for V4.1, when the
/// mean expert is routed fewer calibration rows than its input width
/// (`n_cal·top_k/ne < hidden`: most experts' own Hessians are rank
/// deficient), one hidden width, the value measured best on Flash-Next.
fn production_prior(family: Family, env: Option<f64>, n_cal: usize, top_k: usize, ne: usize, hidden: usize) -> (f64, String) {
    let mean = (n_cal * top_k) as f64 / ne.max(1) as f64;
    match env {
        Some(v) => (v.max(0.0), format!("CMF_MOE_PRIOR={v}")),
        None if family == Family::Dsv41 && mean < hidden as f64 => {
            (hidden as f64, format!("V4.1 default: {mean:.0} routed rows per expert on average, below hidden {hidden} (CMF_MOE_PRIOR overrides)"))
        }
        None => (0.0, format!("{mean:.0} routed rows per expert on average (CMF_MOE_PRIOR sets one)")),
    }
}

/// Whether the slots `triples` (gate, up, down directory indices of the
/// calibration file) take this codec's payloads. In place (`target` None)
/// a slot keeps its directory entry and is rewritten with the codec's dtype:
/// it must be that dtype or q4tp (a q4tp slot holds either layout) and hold
/// at least the payload's bytes. `--patch-shared` rewrites an existing
/// output, whose layout the codec must match.
fn check_slots(cm: &CmfModel, triples: &[(usize, usize, usize)], codec: Codec, target: Option<(&str, &CmfModel)>) -> anyhow::Result<()> {
    let dts = codec.dtypes();
    for &(g, u, d) in triples {
        for (slot, want) in [g, u, d].into_iter().zip(dts) {
            let e = &cm.tensors[slot];
            let need = cortiq_core::quant::expected_nbytes(want, &e.shape)
                .ok_or_else(|| anyhow::anyhow!("{}: shape {:?} has no {} layout", e.name, e.shape, want.name()))? as u64;
            match target {
                None => {
                    anyhow::ensure!(
                        e.dtype == want || e.dtype == TensorDtype::Q4TiledP,
                        "--in-place --codec {}: {} is {} in the calibration file, and a {} payload fits only a {} or a q4tp slot",
                        codec.name(),
                        e.name,
                        e.dtype.name(),
                        want.name(),
                        want.name()
                    );
                    anyhow::ensure!(
                        e.nbytes >= need,
                        "--in-place --codec {}: the slot of {} holds {} B, its {} payload needs {need} B",
                        codec.name(),
                        e.name,
                        e.nbytes,
                        want.name()
                    );
                }
                Some((path, t)) => {
                    let te = t.tensor(&e.name).ok_or_else(|| anyhow::anyhow!("--patch-shared: {} is not in {path}", e.name))?;
                    let hint = match te.dtype {
                        TensorDtype::Q2TiledP => " (pass --codec q2tp)",
                        TensorDtype::Q4TiledP if want == TensorDtype::Q2TiledP => " (pass --codec q4tp)",
                        _ => "",
                    };
                    anyhow::ensure!(
                        te.dtype == want,
                        "--patch-shared --codec {}: {} is {} in {path}, the codec writes {}{hint}",
                        codec.name(),
                        e.name,
                        te.dtype.name(),
                        want.name()
                    );
                    anyhow::ensure!(te.shape == e.shape && te.nbytes >= need, "--patch-shared: {} has shape {:?} / {} B in {path}, want {:?} / {need} B", e.name, te.shape, te.nbytes, e.shape);
                }
            }
        }
    }
    Ok(())
}

/// Bytes a new `--output` file takes: every tensor of the calibration file,
/// the expert slots at their codec payload (`slot_dt`), each padded to the
/// data alignment, plus the head reserve and the vocabulary.
fn stream_output_bytes(cm: &CmfModel, slot_dt: &HashMap<usize, TensorDtype>) -> anyhow::Result<u64> {
    use cortiq_core::format::DATA_ALIGNMENT;
    let avg_name = cm.tensors.iter().map(|e| e.name.len()).sum::<usize>() / cm.tensors.len().max(1);
    let mut total = CmfStreamWriter::head_reserve_for(cm.tensors.len(), avg_name) + cm.vocab.as_ref().map_or(0, |v| v.len() as u64);
    for (slot, e) in cm.tensors.iter().enumerate() {
        let n = match slot_dt.get(&slot) {
            Some(&dt) => cortiq_core::quant::expected_nbytes(dt, &e.shape).ok_or_else(|| anyhow::anyhow!("{}: shape {:?} has no {} layout", e.name, e.shape, dt.name()))? as u64,
            None => e.nbytes,
        };
        total += n.div_ceil(DATA_ALIGNMENT) * DATA_ALIGNMENT;
    }
    Ok(total)
}

/// The bytes a regular file at `path` has allocated when it is its only
/// name (no other hard link keeps them), else 0: what truncating it frees.
fn sole_file_bytes(path: &Path) -> u64 {
    let Ok(m) = std::fs::symlink_metadata(path) else {
        return 0;
    };
    if !m.is_file() {
        return 0;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if m.nlink() == 1 { m.blocks() * 512 } else { 0 }
    }
    #[cfg(not(unix))]
    {
        m.len()
    }
}

/// The filesystem holding `path` (its nearest existing ancestor): its
/// device id and the bytes available to this user. `None` off unix, or
/// when it cannot be read.
fn fs_space(path: &Path) -> Option<(u64, u64)> {
    let base = std::iter::successors(Some(path), |p| p.parent()).find(|p| !p.as_os_str().is_empty() && p.exists()).unwrap_or(Path::new("."));
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::MetadataExt;
        let dev = std::fs::metadata(base).ok()?.dev();
        let c = std::ffi::CString::new(base.as_os_str().as_bytes()).ok()?;
        let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
            return None;
        }
        Some((dev, u64::from(st.f_bavail).saturating_mul(u64::from(st.f_frsize))))
    }
    #[cfg(not(unix))]
    {
        let _ = base;
        None
    }
}

/// Whether each filesystem has room for what the run writes to it: `needs`
/// are (path, what, bytes), `credit` (path, bytes) space the run frees
/// there first (an old output it truncates). A shortfall stops the run
/// unless `allow` (`CMF_MOE_NO_SPACE_CHECK=1`).
fn check_free_space(needs: &[(PathBuf, String, u64)], credit: &[(PathBuf, u64)], allow: bool) -> anyhow::Result<()> {
    let gb = |b: u64| b as f64 / 1e9;
    // per filesystem: (a path on it, available, needed, what)
    let mut fss: Vec<(u64, PathBuf, u64, u64, Vec<String>)> = Vec::new();
    for (path, what, bytes) in needs.iter().filter(|n| n.2 > 0) {
        let Some((dev, avail)) = fs_space(path) else {
            eprintln!("free space: cannot read the filesystem of {} ({what}); not checked", path.display());
            continue;
        };
        let i = match fss.iter().position(|f| f.0 == dev) {
            Some(i) => i,
            None => {
                fss.push((dev, path.clone(), avail, 0, Vec::new()));
                fss.len() - 1
            }
        };
        fss[i].3 += bytes;
        fss[i].4.push(format!("{what} {:.1} GB", gb(*bytes)));
    }
    for (path, bytes) in credit {
        if let Some(f) = fs_space(path).and_then(|(dev, _)| fss.iter_mut().find(|f| f.0 == dev)) {
            f.2 += bytes;
        }
    }
    let mut short = Vec::new();
    for (_, path, avail, need, what) in &fss {
        let line = format!("{}: needs {:.1} GB ({}), {:.1} GB available", path.display(), gb(*need), what.join(", "), gb(*avail));
        eprintln!("free space: {line}");
        if need > avail {
            short.push(line);
        }
    }
    if short.is_empty() {
        return Ok(());
    }
    if allow {
        eprintln!("free space: CMF_MOE_NO_SPACE_CHECK=1 — running although {}", short.join("; "));
        return Ok(());
    }
    anyhow::bail!("not enough free disk space: {}. Free some, or set CMF_MOE_NO_SPACE_CHECK=1 to run anyway", short.join("; "))
}

/// The files of the calibration model: the one given and, when it is shard
/// 1 of a sharded model, its siblings (`…-{no:05}-of-{count:05}.cmf`).
fn calib_files(path: &str, cm: &CmfModel) -> Vec<PathBuf> {
    let p = Path::new(path);
    let mut files = vec![p.to_path_buf()];
    if let Some(info) = cm.header.shard.as_ref() {
        let tag1 = format!("-{:05}-of-{:05}.cmf", 1, info.count);
        if let Some(stem) = p.file_name().and_then(|n| n.to_str()).and_then(|n| n.strip_suffix(&tag1)) {
            files.extend((2..=info.count).map(|no| p.with_file_name(format!("{stem}-{no:05}-of-{:05}.cmf", info.count))));
        }
    }
    files
}

/// Whether `a` and `b` are the same existing file, also through a symlink
/// or a hard link (unix: device and inode; elsewhere the canonical paths).
fn same_file(a: &Path, b: &Path) -> bool {
    let (Ok(ma), Ok(mb)) = (std::fs::metadata(a), std::fs::metadata(b)) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ma.dev() == mb.dev() && ma.ino() == mb.ino()
    }
    #[cfg(not(unix))]
    {
        let _ = (ma, mb);
        matches!((std::fs::canonicalize(a), std::fs::canonicalize(b)), (Ok(x), Ok(y)) if x == y)
    }
}

/// The expert triples (gate, up, down directory indices) of a V4.1 layer:
/// the converter's canonical names, or the upstream `ffn.*`/`w1..w3` ones
/// (the runtime loader accepts both).
fn dsv41_dirs(cm: &CmfModel, li: usize, ne: usize) -> anyhow::Result<(Vec<(usize, usize, usize)>, (usize, usize, usize))> {
    let find = |stem: &str, parts: [&str; 2]| -> anyhow::Result<usize> {
        for blk in ["mlp", "ffn"] {
            for p in parts {
                if let Some(i) = cm.tensor_index(&format!("model.layers.{li}.{blk}.{stem}.{p}.weight")) {
                    return Ok(i);
                }
            }
        }
        anyhow::bail!("layer {li}: no {stem}.{} in the calibration file", parts[0])
    };
    let triple = |stem: &str| -> anyhow::Result<(usize, usize, usize)> {
        Ok((find(stem, ["gate_proj", "w1"])?, find(stem, ["up_proj", "w3"])?, find(stem, ["down_proj", "w2"])?))
    };
    let routed = (0..ne).map(|e| triple(&format!("experts.{e}"))).collect::<anyhow::Result<Vec<_>>>()?;
    let shared = triple("shared_expert").or_else(|_| triple("shared_experts"))?;
    Ok((routed, shared))
}

// ── full-precision source ────────────────────────────────────────────────

/// The original checkpoint: a Hub repo (shards fetched into the scratch
/// directory one layer at a time and deleted after use) or a local
/// directory (read in place, never deleted).
struct Source {
    local: Option<PathBuf>,
    scratch: PathBuf,
    agent: ureq::Agent,
    base: String,
    token: Option<String>,
    wmap: serde_json::Map<String, serde_json::Value>,
}

impl Source {
    fn open(hf: &str, scratch: &str) -> anyhow::Result<Self> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(20))
            .timeout_read(std::time::Duration::from_secs(300))
            .build();
        let local = Path::new(hf).is_dir().then(|| PathBuf::from(hf));
        let scratch = PathBuf::from(scratch);
        let token = std::env::var("HF_TOKEN").ok();
        let base = format!("https://huggingface.co/{hf}/resolve/main");
        let idx_path = match &local {
            Some(d) => d.join("model.safetensors.index.json"),
            None => {
                std::fs::create_dir_all(&scratch)?;
                let p = scratch.join("model.safetensors.index.json");
                fetch(&agent, &format!("{base}/model.safetensors.index.json"), &p, token.as_deref(), true, 8)?;
                p
            }
        };
        let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx_path).map_err(|e| anyhow::anyhow!("{}: {e}", idx_path.display()))?)?;
        let wmap = idx["weight_map"].as_object().ok_or_else(|| anyhow::anyhow!("{}: no weight_map", idx_path.display()))?.clone();
        Ok(Self { local, scratch, agent, base, token, wmap })
    }

    fn shard_of(&self, name: &str) -> Option<&str> {
        self.wmap.get(name).and_then(|v| v.as_str())
    }

    /// The sorted, distinct shards holding `names`.
    fn shards_of(&self, names: &[String]) -> anyhow::Result<Vec<String>> {
        let mut v = Vec::new();
        for n in names {
            v.push(self.shard_of(n).ok_or_else(|| anyhow::anyhow!("{n} is not in the source index"))?.to_string());
        }
        v.sort();
        v.dedup();
        Ok(v)
    }

    fn path(&self, sh: &str) -> PathBuf {
        match &self.local {
            Some(d) => d.join(sh),
            None => self.scratch.join(sh),
        }
    }

    /// Fetch (remote) and map every shard of one layer. A failed download
    /// is retried for up to `fetch_wait()` (a pause of 20 s·k, at most 5
    /// minutes, after the k-th failure).
    fn open_layer(&self, shards: &[String]) -> anyhow::Result<LayerSrc> {
        let mut files = HashMap::new();
        let mut index = HashMap::new();
        for sh in shards {
            let p = self.path(sh);
            if self.local.is_none() {
                with_retries(fetch_wait(), std::time::Duration::from_secs(20), std::time::Duration::from_secs(300), sh, || {
                    fetch(&self.agent, &format!("{}/{sh}", self.base), &p, self.token.as_deref(), true, 16).map(|_| ())
                })?;
            }
            let st = open_safetensors(&p)?;
            for (i, m) in st.tensors.iter().enumerate() {
                index.insert(m.name.clone(), (sh.clone(), i));
            }
            files.insert(sh.clone(), st);
        }
        Ok(LayerSrc { files, index })
    }

    /// Download `shards` in the background (remote sources only).
    fn prefetch(&self, shards: Vec<String>) -> Option<std::thread::JoinHandle<anyhow::Result<()>>> {
        if self.local.is_some() {
            return None;
        }
        let (agent, base, token, scratch) = (self.agent.clone(), self.base.clone(), self.token.clone(), self.scratch.clone());
        Some(std::thread::spawn(move || -> anyhow::Result<()> {
            for sh in shards {
                fetch(&agent, &format!("{base}/{sh}"), &scratch.join(&sh), token.as_deref(), true, 16)?;
            }
            Ok(())
        }))
    }

    /// Total bytes of remote `shards` (a 1-byte range request each); `None`
    /// when the Hub reports no size for one of them.
    fn remote_sizes(&self, shards: &[String]) -> Option<u64> {
        shards.iter().map(|sh| crate::convert::probe_size(&self.agent, &format!("{}/{sh}", self.base), self.token.as_deref())).sum()
    }

    /// Delete a fetched shard (a local source is left alone).
    fn release(&self, sh: &str) {
        if self.local.is_none() {
            let _ = std::fs::remove_file(self.scratch.join(sh));
        }
    }
}

/// How long a synchronous shard download keeps being retried before the
/// run gives up: `CMF_MOE_FETCH_WAIT` minutes, default 240. A run that
/// stops after its capture, or halfway through an `--output` file (which
/// cannot resume), loses hours: waiting out a Hub outage is cheaper.
fn fetch_wait() -> std::time::Duration {
    let min = std::env::var("CMF_MOE_FETCH_WAIT").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|m| *m >= 0.0).unwrap_or(240.0);
    std::time::Duration::from_secs_f64(min * 60.0)
}

/// `f` until it succeeds, retried while the next pause still ends within
/// `budget` of the first attempt; the pause after the k-th failure is
/// `pause`·k, at most `max_pause`. The last error is returned.
fn with_retries(budget: std::time::Duration, pause: std::time::Duration, max_pause: std::time::Duration, what: &str, mut f: impl FnMut() -> anyhow::Result<()>) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let mut k = 1u32;
    loop {
        match f() {
            Ok(()) => return Ok(()),
            Err(e) => {
                let wait = (pause * k).min(max_pause);
                if t0.elapsed() + wait > budget {
                    return Err(e.context(format!("{what}: {k} attempts failed over {:.0} min (CMF_MOE_FETCH_WAIT sets the minutes)", t0.elapsed().as_secs_f64() / 60.0)));
                }
                eprintln!(
                    "  {what}: attempt {k} failed ({e:#}); retrying in {:.0}s (giving up {:.0} min after the first attempt)",
                    wait.as_secs_f64(),
                    budget.as_secs_f64() / 60.0
                );
                std::thread::sleep(wait);
                k += 1;
            }
        }
    }
}

/// The mapped source shards of one layer, with a name index.
pub(crate) struct LayerSrc {
    files: HashMap<String, crate::convert::SafeTensors>,
    index: HashMap<String, (String, usize)>,
}

impl LayerSrc {
    /// Map local shard files (tests, and local sources).
    #[cfg(test)]
    pub(crate) fn from_paths(paths: &[PathBuf]) -> anyhow::Result<Self> {
        let mut files = HashMap::new();
        let mut index = HashMap::new();
        for p in paths {
            let sh = p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
            let st = open_safetensors(p)?;
            for (i, m) in st.tensors.iter().enumerate() {
                index.insert(m.name.clone(), (sh.clone(), i));
            }
            files.insert(sh, st);
        }
        Ok(Self { files, index })
    }

    fn tensor(&self, name: &str) -> anyhow::Result<(&crate::convert::SafeTensors, &crate::convert::TensorMeta)> {
        let (sh, i) = self.index.get(name).ok_or_else(|| anyhow::anyhow!("{name} not in the layer's source shards"))?;
        let st = &self.files[sh];
        Ok((st, &st.tensors[*i]))
    }

    /// One V4.1 source matrix `[rows, cols]` decoded to f32 exactly as the
    /// converter decodes it: FP4 (`I8`/`U8` nibble pairs `[rows, cols/2]` +
    /// an E8M0 scale per 32 along the row), FP8 (`F8_E4M3` + an E8M0 scale
    /// per `block`×`block` tile), or a plain float tensor.
    pub(crate) fn dsv41_matrix(&self, stem: &str, rows: usize, cols: usize, block: usize) -> anyhow::Result<Vec<f32>> {
        let (st, m) = self.tensor(&format!("{stem}.weight"))?;
        let scale = || -> anyhow::Result<(&[u8], &crate::convert::TensorMeta)> {
            let (ss, sm) = self.tensor(&format!("{stem}.scale"))?;
            anyhow::ensure!(matches!(sm.dtype.as_str(), "F8_E8M0" | "U8"), "{stem}.scale: dtype {} (want F8_E8M0)", sm.dtype);
            Ok((ss.bytes(sm), sm))
        };
        match m.dtype.as_str() {
            "I8" | "U8" => {
                anyhow::ensure!(m.shape == [rows, cols / 2], "{stem}.weight: FP4 shape {:?}, want [{rows}, {}]", m.shape, cols / 2);
                let (sb, sm) = scale()?;
                anyhow::ensure!(sm.shape == [rows, cols / 32], "{stem}.scale: shape {:?}, want [{rows}, {}]", sm.shape, cols / 32);
                unpack_mxfp4(st.bytes(m), sb, rows, cols / 2)
            }
            "F8_E4M3" => {
                anyhow::ensure!(m.shape == [rows, cols], "{stem}.weight: FP8 shape {:?}, want [{rows}, {cols}]", m.shape);
                let (sb, sm) = scale()?;
                anyhow::ensure!(
                    sm.shape == [rows.div_ceil(block), cols.div_ceil(block)],
                    "{stem}.scale: shape {:?}, want [{}, {}] ({block}×{block} tiles)",
                    sm.shape,
                    rows.div_ceil(block),
                    cols.div_ceil(block)
                );
                unpack_fp8_blocks(st.bytes(m), sb, rows, cols, block)
            }
            _ => {
                anyhow::ensure!(m.shape == [rows, cols], "{stem}.weight: shape {:?}, want [{rows}, {cols}]", m.shape);
                to_f32(&m.dtype, st.bytes(m))
            }
        }
    }
}

/// The source tensors of a V4.1 layer's experts (`routed`: the routed ones
/// too, not only the shared expert): every weight, and its scale plane when
/// the index has one.
fn dsv41_src_names(source: &Source, li: usize, ne: usize, routed: bool) -> Vec<String> {
    let mut stems: Vec<String> = Vec::with_capacity(ne * 3 + 3);
    if routed {
        for e in 0..ne {
            stems.extend(["w1", "w2", "w3"].map(|w| format!("layers.{li}.ffn.experts.{e}.{w}")));
        }
    }
    stems.extend(["w1", "w2", "w3"].map(|w| format!("layers.{li}.ffn.shared_experts.{w}")));
    let mut names = Vec::with_capacity(stems.len() * 2);
    for s in stems {
        let scale = format!("{s}.scale");
        names.push(format!("{s}.weight"));
        if source.shard_of(&scale).is_some() {
            names.push(scale);
        }
    }
    names
}

/// Which expert of a layer.
#[derive(Clone, Copy)]
pub(crate) enum Part {
    Routed,
    Shared,
}

/// V4.1 expert `e` (or the shared expert) of layer `li` from its source
/// shard: `w1` gate, `w3` up, `w2` down.
pub(crate) fn dsv41_expert(src: &LayerSrc, li: usize, part: Part, e: usize, hidden: usize, inter: usize, block: usize) -> anyhow::Result<ExpertW> {
    let stem = match part {
        Part::Routed => format!("layers.{li}.ffn.experts.{e}"),
        Part::Shared => format!("layers.{li}.ffn.shared_experts"),
    };
    Ok(ExpertW {
        g: src.dsv41_matrix(&format!("{stem}.w1"), inter, hidden, block)?,
        u: src.dsv41_matrix(&format!("{stem}.w3"), inter, hidden, block)?,
        d: src.dsv41_matrix(&format!("{stem}.w2"), hidden, inter, block)?,
    })
}

const QWEN4_ROUTED: [&str; 2] = ["experts.gate_up_proj", "experts.down_proj"];
const QWEN4_SHARED: [&str; 3] = ["shared_expert.gate_proj.weight", "shared_expert.up_proj.weight", "shared_expert.down_proj.weight"];

/// The source name of `part` below `….layers.{li}.mlp.` (Flash-Next).
fn qwen4_src_name(source: &Source, li: usize, part: &str) -> anyhow::Result<String> {
    for prefix in ["model.language_model.layers", "model.layers"] {
        let n = format!("{prefix}.{li}.mlp.{part}");
        if source.shard_of(&n).is_some() {
            return Ok(n);
        }
    }
    anyhow::bail!("layer {li}: no mlp.{part} in the source index")
}

/// Flash-Next expert `e` (fused bf16 `gate_up_proj [ne, 2·inter, hidden]`,
/// gate first; `down_proj [ne, hidden, inter]`) or its shared expert.
#[allow(clippy::too_many_arguments)]
fn qwen4_expert(source: &Source, src: &LayerSrc, li: usize, part: Part, e: usize, hidden: usize, inter: usize, inter_s: usize) -> anyhow::Result<ExpertW> {
    match part {
        Part::Routed => {
            let (gst, gm) = src.tensor(&qwen4_src_name(source, li, QWEN4_ROUTED[0])?)?;
            let (dst, dm) = src.tensor(&qwen4_src_name(source, li, QWEN4_ROUTED[1])?)?;
            anyhow::ensure!(
                gm.shape.len() == 3 && gm.shape[1..] == [2 * inter, hidden] && dm.shape.len() == 3 && dm.shape[1..] == [hidden, inter] && e < gm.shape[0],
                "layer {li}: unexpected expert layout {:?} / {:?}",
                gm.shape,
                dm.shape
            );
            let esz = |shape: &[usize]| shape[1] * shape[2] * 2; // bf16
            let (gsz, dsz) = (esz(&gm.shape), esz(&dm.shape));
            let mut gu = to_f32(&gm.dtype, &gst.bytes(gm)[e * gsz..(e + 1) * gsz])?;
            let d = to_f32(&dm.dtype, &dst.bytes(dm)[e * dsz..(e + 1) * dsz])?;
            let u = gu.split_off(inter * hidden);
            Ok(ExpertW { g: gu, u, d })
        }
        Part::Shared => {
            let get = |p: &str, shape: [usize; 2]| -> anyhow::Result<Vec<f32>> {
                let (st, m) = src.tensor(&qwen4_src_name(source, li, p)?)?;
                anyhow::ensure!(m.shape == shape, "layer {li}: {p} has shape {:?}, want {shape:?}", m.shape);
                to_f32(&m.dtype, st.bytes(m))
            };
            Ok(ExpertW {
                g: get(QWEN4_SHARED[0], [inter_s, hidden])?,
                u: get(QWEN4_SHARED[1], [inter_s, hidden])?,
                d: get(QWEN4_SHARED[2], [hidden, inter_s])?,
            })
        }
    }
}

/// Flash-Next capture rows `[hidden + ne]` (input, router logits) → input
/// rows in f16 and the top-k of the f16-rounded logits.
fn qwen4_layer_rows(r: &[f32], hidden: usize, ne: usize, top_k: usize, out: &mut LayerRows) {
    let w = hidden + ne;
    let mut lg = vec![0f32; ne];
    let mut ord: Vec<usize> = Vec::with_capacity(ne);
    for row in r.chunks_exact(w) {
        out.x.extend(row[..hidden].iter().map(|&v| f32_to_f16(v)));
        for (d, &v) in lg.iter_mut().zip(&row[hidden..]) {
            *d = f16_to_f32(f32_to_f16(v));
        }
        ord.clear();
        ord.extend(0..ne);
        ord.select_nth_unstable_by(top_k - 1, |&a, &b| lg[b].partial_cmp(&lg[a]).unwrap_or(std::cmp::Ordering::Equal));
        out.picks.extend(ord[..top_k].iter().map(|&e| e as u16));
    }
}

/// The capture passes over layers `lo..hi`: `per` layers each, or one pass
/// over the whole range in place (every layer is captured before the first
/// is rewritten: a later pass would run over experts already replaced).
fn plan_passes(lo: usize, hi: usize, per: usize, in_place: bool) -> Vec<(usize, usize)> {
    let per = if in_place { (hi - lo).max(1) } else { per.max(1) };
    (lo..hi).step_by(per).map(|a| (a, (a + per).min(hi))).collect()
}

// ── saved capture (CMF_MOE_CAPTURE_DIR) ──────────────────────────────────
//
// Per layer: `L{li}.x` (the f16 input rows, raw little-endian u16),
// `L{li}.picks` (u16 expert ids), `L{li}.w` (f32 route weights, V4.1), and
// one `meta.json`. Every file is written to a `.tmp` sibling and renamed;
// `meta.json` lists a layer only after its files are complete.

/// What a saved capture was taken from, and which layers it holds.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CaptureMeta {
    /// SHA-256 of the calibration windows (each window's length, then its ids).
    pub token_hash: String,
    pub ntok: usize,
    pub window: usize,
    pub windows: usize,
    pub hidden: usize,
    pub top_k: usize,
    pub ne: usize,
    pub family: String,
    pub calib_model: String,
    pub calib_size: u64,
    /// Route weights saved (`L{li}.w`).
    pub has_w: bool,
    pub layers: Vec<usize>,
}

const CAPTURE_FORMAT: &str = "cortiq-moe-capture-1";

impl CaptureMeta {
    /// Bytes one layer takes on disk.
    pub(crate) fn layer_bytes(&self) -> u64 {
        (self.ntok * (self.hidden * 2 + self.top_k * 2 + if self.has_w { self.top_k * 4 } else { 0 })) as u64
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "format": CAPTURE_FORMAT,
            "token_hash": self.token_hash,
            "ntok": self.ntok,
            "window": self.window,
            "windows": self.windows,
            "hidden": self.hidden,
            "top_k": self.top_k,
            "ne": self.ne,
            "family": self.family,
            "calibration_model": self.calib_model,
            "calibration_model_bytes": self.calib_size,
            "route_weights": self.has_w,
            "layers": self.layers,
            "cortiq_version": env!("CARGO_PKG_VERSION"),
        })
    }

    fn from_json(v: &serde_json::Value) -> anyhow::Result<Self> {
        anyhow::ensure!(v["format"].as_str() == Some(CAPTURE_FORMAT), "format {:?}, want {CAPTURE_FORMAT}", v["format"]);
        let us = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or_else(|| anyhow::anyhow!("no {k}"));
        let st = |k: &str| v[k].as_str().map(str::to_string).ok_or_else(|| anyhow::anyhow!("no {k}"));
        Ok(Self {
            token_hash: st("token_hash")?,
            ntok: us("ntok")?,
            window: us("window")?,
            windows: us("windows")?,
            hidden: us("hidden")?,
            top_k: us("top_k")?,
            ne: us("ne")?,
            family: st("family")?,
            calib_model: st("calibration_model")?,
            calib_size: v["calibration_model_bytes"].as_u64().unwrap_or(0),
            has_w: v["route_weights"].as_bool().unwrap_or(false),
            layers: v["layers"].as_array().ok_or_else(|| anyhow::anyhow!("no layers"))?.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect(),
        })
    }

    /// Why this saved capture cannot serve a run described by `want` (the
    /// layers aside); `None` if it can.
    fn mismatch(&self, want: &CaptureMeta) -> Option<String> {
        let mut why = Vec::new();
        if self.token_hash != want.token_hash || self.ntok != want.ntok {
            why.push(format!("calibration tokens differ ({} tokens in {} windows saved, {} in {} now)", self.ntok, self.windows, want.ntok, want.windows));
        }
        for (k, a, b) in [("hidden", self.hidden, want.hidden), ("top_k", self.top_k, want.top_k), ("experts", self.ne, want.ne)] {
            if a != b {
                why.push(format!("{k} {a} saved, {b} now"));
            }
        }
        if self.family != want.family {
            why.push(format!("family {} saved, {} now", self.family, want.family));
        }
        if want.has_w && !self.has_w {
            why.push("no route weights saved".to_string());
        }
        (!why.is_empty()).then(|| why.join("; "))
    }
}

/// SHA-256 of the calibration windows: each window's length (u32 LE), then
/// its ids (u32 LE).
pub(crate) fn token_hash(windows: &[Vec<u32>]) -> String {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    for w in windows {
        h.update((w.len() as u32).to_le_bytes());
        for &id in w {
            h.update(id.to_le_bytes());
        }
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// The saved capture in `dir` for a run described by `want`: its meta (with
/// the layers it holds), or `want` with no layers for a new directory. A
/// capture of a different run is refused, never overwritten.
pub(crate) fn capture_open(dir: &Path, want: &CaptureMeta) -> anyhow::Result<CaptureMeta> {
    let mp = dir.join("meta.json");
    if !mp.exists() {
        std::fs::create_dir_all(dir).map_err(|e| anyhow::anyhow!("CMF_MOE_CAPTURE_DIR {}: {e}", dir.display()))?;
        return Ok(CaptureMeta { layers: Vec::new(), ..want.clone() });
    }
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&mp)?).map_err(|e| anyhow::anyhow!("{}: {e}", mp.display()))?;
    let have = CaptureMeta::from_json(&v).map_err(|e| anyhow::anyhow!("{}: {e}", mp.display()))?;
    if let Some(why) = have.mismatch(want) {
        anyhow::bail!(
            "CMF_MOE_CAPTURE_DIR {} holds the capture of another run ({why}); run with the same --calib/--tokens/--window, \
             or point CMF_MOE_CAPTURE_DIR at an empty directory",
            dir.display()
        );
    }
    if (have.calib_model.as_str(), have.calib_size) != (want.calib_model.as_str(), want.calib_size) {
        eprintln!(
            "note: the saved capture was taken with {} ({} B), this run calibrates with {} ({} B); the same tokens, so it is reused",
            have.calib_model, have.calib_size, want.calib_model, want.calib_size
        );
    }
    Ok(have)
}

/// `path` written through a `.tmp` sibling and renamed into place.
fn write_atomic(path: &Path, fill: impl FnOnce(&mut std::io::BufWriter<std::fs::File>) -> std::io::Result<()>) -> anyhow::Result<()> {
    let name = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = path.with_file_name(format!("{name}.tmp"));
    {
        let mut w = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp).map_err(|e| anyhow::anyhow!("{}: {e}", tmp.display()))?);
        fill(&mut w).map_err(|e| anyhow::anyhow!("{}: {e}", tmp.display()))?;
        w.into_inner().map_err(|e| anyhow::anyhow!("{}: {e}", tmp.display()))?.sync_all()?;
    }
    std::fs::rename(&tmp, path).map_err(|e| anyhow::anyhow!("{} → {}: {e}", tmp.display(), path.display()))?;
    Ok(())
}

/// Little-endian values in 1 MiB-element blocks.
fn write_le<T: Copy, const N: usize>(w: &mut impl std::io::Write, v: &[T], le: impl Fn(T) -> [u8; N]) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity((1 << 20) * N);
    for c in v.chunks(1 << 20) {
        buf.clear();
        buf.extend(c.iter().flat_map(|&x| le(x)));
        w.write_all(&buf)?;
    }
    Ok(())
}

/// `n` little-endian values of `N` bytes from `path` (its size must match).
fn read_le<T, const N: usize>(path: &Path, n: usize, from: impl Fn([u8; N]) -> T) -> anyhow::Result<Vec<T>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let len = f.metadata()?.len();
    anyhow::ensure!(len == (n * N) as u64, "{}: {len} B, want {} B (delete the layer's files to capture it again)", path.display(), n * N);
    let mut out = Vec::with_capacity(n);
    let mut buf = vec![0u8; (1 << 20) * N];
    let mut left = n;
    while left > 0 {
        let k = left.min(1 << 20);
        f.read_exact(&mut buf[..k * N])?;
        out.extend(buf[..k * N].chunks_exact(N).map(|b| from(b.try_into().expect("N bytes"))));
        left -= k;
    }
    Ok(out)
}

fn capture_path(dir: &Path, li: usize, ext: &str) -> PathBuf {
    dir.join(format!("L{li}.{ext}"))
}

/// Save one layer's captured rows (`meta.json` is written separately).
pub(crate) fn capture_save_layer(dir: &Path, meta: &CaptureMeta, li: usize, r: &LayerRows) -> anyhow::Result<()> {
    anyhow::ensure!(
        r.x.len() == meta.ntok * meta.hidden && r.picks.len() == meta.ntok * meta.top_k && (!meta.has_w || r.w.len() == r.picks.len()),
        "layer {li}: {} rows / {} picks / {} weights for {} tokens",
        r.x.len() / meta.hidden.max(1),
        r.picks.len(),
        r.w.len(),
        meta.ntok
    );
    write_atomic(&capture_path(dir, li, "x"), |w| write_le(w, &r.x, u16::to_le_bytes))?;
    write_atomic(&capture_path(dir, li, "picks"), |w| write_le(w, &r.picks, u16::to_le_bytes))?;
    if meta.has_w {
        write_atomic(&capture_path(dir, li, "w"), |w| write_le(w, &r.w, f32::to_le_bytes))?;
    }
    Ok(())
}

pub(crate) fn capture_save_meta(dir: &Path, meta: &CaptureMeta) -> anyhow::Result<()> {
    let bytes = serde_json::to_vec_pretty(&meta.to_json())?;
    write_atomic(&dir.join("meta.json"), |w| std::io::Write::write_all(w, &bytes))
}

/// One saved layer's rows.
pub(crate) fn capture_load_layer(dir: &Path, meta: &CaptureMeta, li: usize) -> anyhow::Result<LayerRows> {
    anyhow::ensure!(meta.layers.contains(&li), "layer {li} is not in the saved capture {}", dir.display());
    Ok(LayerRows {
        x: read_le(&capture_path(dir, li, "x"), meta.ntok * meta.hidden, u16::from_le_bytes)?,
        picks: read_le(&capture_path(dir, li, "picks"), meta.ntok * meta.top_k, u16::from_le_bytes)?,
        w: if meta.has_w { read_le(&capture_path(dir, li, "w"), meta.ntok * meta.top_k, f32::from_le_bytes)? } else { Vec::new() },
    })
}

// ── output ───────────────────────────────────────────────────────────────

enum Sink {
    /// A new file: the experts as they come, then every other tensor.
    Stream(CmfStreamWriter),
    /// The calibration file's own slots, rewritten after every layer.
    InPlace {
        path: String,
        pending: Vec<(usize, TensorDtype, Vec<u8>)>,
        before: u64,
        after: u64,
        punched: u64,
    },
    /// `--patch-shared`: the slots of an existing output, at the end.
    Patch {
        path: String,
        target: CmfModel,
        patches: Vec<(usize, TensorDtype, Vec<u8>)>,
    },
    /// Diagnostics: nothing is written.
    Diag,
}

impl Sink {
    /// One encoded tensor for directory slot `slot` of the calibration file.
    fn emit(&mut self, cm: &CmfModel, slot: usize, dt: TensorDtype, bytes: Vec<u8>) -> anyhow::Result<()> {
        let e = &cm.tensors[slot];
        match self {
            Sink::Stream(w) => w.push(&e.name, dt, &e.shape, &bytes).map_err(Into::into),
            Sink::InPlace { pending, .. } => {
                anyhow::ensure!(bytes.len() as u64 <= e.nbytes, "{}: new payload {} B larger than its slot {} B", e.name, bytes.len(), e.nbytes);
                pending.push((slot, dt, bytes));
                Ok(())
            }
            Sink::Patch { target, patches, .. } => {
                let i = target.tensor_index(&e.name).ok_or_else(|| anyhow::anyhow!("{} not in the patched file", e.name))?;
                anyhow::ensure!(target.tensors[i].shape == e.shape, "{}: shape differs in the patched file", e.name);
                anyhow::ensure!(bytes.len() as u64 <= target.tensors[i].nbytes, "{}: new payload larger than its slot", e.name);
                patches.push((i, dt, bytes));
                Ok(())
            }
            Sink::Diag => Ok(()),
        }
    }

    /// In place: write the layer's pending payloads into their slots and,
    /// where a payload shrank (q4tp slot → q2tp), release the slack.
    fn flush_layer(&mut self, cm: &CmfModel) -> anyhow::Result<()> {
        let Sink::InPlace { path, pending, before, after, punched } = self else {
            return Ok(());
        };
        if pending.is_empty() {
            return Ok(());
        }
        let mut holes = Vec::new();
        for (slot, _, bytes) in pending.iter() {
            let e = &cm.tensors[*slot];
            *before += e.nbytes;
            *after += bytes.len() as u64;
            if (bytes.len() as u64) < e.nbytes {
                let abs = cm.entry_abs_offset(e).ok_or_else(|| anyhow::anyhow!("{}: not in the primary file", e.name))? as u64;
                holes.push((abs + bytes.len() as u64, abs + e.nbytes));
            }
        }
        CmfModel::recode_entries_in_place(path, pending).map_err(|e| anyhow::anyhow!("in place {path}: {e}"))?;
        pending.clear();
        *punched += punch_holes(path, &holes)?;
        Ok(())
    }
}

/// `CMF_MOE_NO_PUNCH=1`: never punch holes (the q2tp slack in q4tp slots
/// stays allocated).
fn no_punch() -> bool {
    std::env::var("CMF_MOE_NO_PUNCH").is_ok_and(|v| v != "0")
}

/// Whether the filesystem of `dir` releases punched holes: a 1 MiB file is
/// written there, one hole punched into it, and its allocated blocks are
/// counted before and after. Linux only (`fallocate`); elsewhere never.
fn probe_hole_punching(dir: &Path) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        use std::io::Write;
        use std::os::unix::fs::MetadataExt;
        use std::os::unix::io::AsRawFd;
        const LEN: usize = 1 << 20;
        let p = dir.join(format!(".cortiq-moe-punch-probe-{}", std::process::id()));
        let probe = || -> Result<(), String> {
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&p)
                .map_err(|e| format!("create {}: {e}", p.display()))?;
            f.write_all(&vec![0xA5u8; LEN]).map_err(|e| format!("write {}: {e}", p.display()))?;
            f.sync_all().map_err(|e| format!("sync {}: {e}", p.display()))?;
            let before = f.metadata().map_err(|e| format!("stat {}: {e}", p.display()))?.blocks();
            let r = unsafe { libc::fallocate(f.as_raw_fd(), libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE, (LEN / 4) as libc::off_t, (LEN / 2) as libc::off_t) };
            if r != 0 {
                return Err(format!("fallocate(PUNCH_HOLE): {}", std::io::Error::last_os_error()));
            }
            f.sync_all().map_err(|e| format!("sync {}: {e}", p.display()))?;
            let md = f.metadata().map_err(|e| format!("stat {}: {e}", p.display()))?;
            if md.len() != LEN as u64 || md.blocks() >= before {
                return Err(format!(
                    "fallocate(PUNCH_HOLE) succeeded, but the probe file kept {} of {before} allocated 512-byte blocks (length {} of {LEN})",
                    md.blocks(),
                    md.len()
                ));
            }
            Ok(())
        };
        let r = probe();
        let _ = std::fs::remove_file(&p);
        r
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = dir;
        Err("hole punching is implemented on Linux only".to_string())
    }
}

/// Release whole filesystem blocks inside `[start, end)` ranges of `path`
/// (Linux `fallocate` hole punching; the file length is unchanged and the
/// ranges read back as zeros). Returns the bytes released (0 elsewhere, and
/// with `CMF_MOE_NO_PUNCH=1`).
fn punch_holes(path: &str, ranges: &[(u64, u64)]) -> anyhow::Result<u64> {
    if ranges.is_empty() || no_punch() {
        return Ok(0);
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        const BLK: u64 = 4096;
        let f = std::fs::OpenOptions::new().write(true).open(path)?;
        let mut total = 0u64;
        for &(s, e) in ranges {
            let (s, e) = (s.div_ceil(BLK) * BLK, e / BLK * BLK);
            if e <= s {
                continue;
            }
            let r = unsafe { libc::fallocate(f.as_raw_fd(), libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE, s as libc::off_t, (e - s) as libc::off_t) };
            if r != 0 {
                let err = std::io::Error::last_os_error();
                eprintln!("quantize-moe: hole punching unavailable on {path} ({err}); the slack stays allocated");
                return Ok(total);
            }
            total += e - s;
        }
        Ok(total)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        Ok(0)
    }
}

// ── one layer ────────────────────────────────────────────────────────────

/// Geometry and options of one layer's encode.
pub(crate) struct LayerJob<'a> {
    pub li: usize,
    pub hidden: usize,
    pub ne: usize,
    pub inter: usize,
    pub top_k: usize,
    pub min_rows: usize,
    pub prior_n0: f64,
    pub prior_s: usize,
    /// Per token: held out of the calibration (diagnostics).
    pub held: &'a [bool],
    pub route_w: RouteW,
}

/// Diagnostic accumulators of one row-count bin: experts, Σ‖out‖², Σ error
/// for RTN and for each variant, and the same with every held-out row
/// weighted by its route weight squared (`_w`; the error the MoE output
/// sees, when the capture has route weights).
#[derive(Clone)]
pub(crate) struct DiagBin {
    pub count: usize,
    pub base: f64,
    pub err: Vec<f64>,
    pub base_w: f64,
    pub err_w: Vec<f64>,
}

impl DiagBin {
    fn new(nvar: usize) -> Self {
        Self { count: 0, base: 0.0, err: vec![0.0; nvar + 1], base_w: 0.0, err_w: vec![0.0; nvar + 1] }
    }
    fn add(&mut self, o: &DiagBin) {
        self.count += o.count;
        self.base += o.base;
        self.base_w += o.base_w;
        for (a, b) in self.err.iter_mut().zip(&o.err) {
            *a += b;
        }
        for (a, b) in self.err_w.iter_mut().zip(&o.err_w) {
            *a += b;
        }
    }
    fn rel(&self, i: usize) -> String {
        format!("{:.4}", self.err[i] / self.base)
    }
    fn rel_w(&self, i: usize) -> String {
        format!("{:.4}", self.err_w[i] / self.base_w)
    }
    fn rels(&self, nvar: usize) -> String {
        (1..=nvar).map(|i| self.rel(i)).collect::<Vec<_>>().join(" ")
    }
    fn rels_w(&self, nvar: usize) -> String {
        (1..=nvar).map(|i| self.rel_w(i)).collect::<Vec<_>>().join(" ")
    }
}

/// Σ‖a_r − b_r‖² over the `m` rows (`width` wide), unweighted and weighted
/// by `w_r²` (`w` empty: the weighted sum is 0).
fn row_err(a: &[f32], b: &[f32], width: usize, w: &[f32]) -> (f64, f64) {
    let (mut e, mut ew) = (0f64, 0f64);
    for (r, (ra, rb)) in a.chunks_exact(width).zip(b.chunks_exact(width)).enumerate() {
        let v: f64 = ra.iter().zip(rb).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum();
        e += v;
        if let Some(&wr) = w.get(r) {
            ew += (wr as f64).powi(2) * v;
        }
    }
    (e, ew)
}

const BINS: [usize; 4] = [640, 2560, 10240, usize::MAX];

fn bin_label(bi: usize) -> String {
    if BINS[bi] == usize::MAX { "∞".into() } else { BINS[bi].to_string() }
}

/// What one layer produced.
pub(crate) struct LayerOut {
    /// Per routed expert: gate, up, down payloads (empty without routed work).
    pub routed: Vec<[Vec<u8>; 3]>,
    pub shared: Option<[Vec<u8>; 3]>,
    pub n_gptq: usize,
    pub n_rtn: usize,
    pub diag: Vec<DiagBin>,
    /// Stage seconds summed over workers: inputs, gate/up fold, gate/up sweeps, down fold, down sweep.
    pub tim: [f64; 5],
    pub t_shared: f64,
}

/// The routed experts (`routed`) and the shared expert (`shared`) of one
/// layer: per expert its rows from the captured routes, then GPTQ (or the
/// converter's encoding below `min_rows`). With `diag`, the routed experts
/// are measured on the held-out rows instead and nothing is returned.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_layer(
    engine: Engine,
    ec: &EncodeCfg,
    job: &LayerJob,
    lr: &LayerRows,
    routed: bool,
    shared: bool,
    main_bufs: Option<&wgpu_bufs::Bufs>,
    diag: Option<&[f64]>,
    get: &(dyn Fn(Part, usize) -> anyhow::Result<ExpertW> + Sync),
) -> anyhow::Result<LayerOut> {
    let (hidden, ne, inter, top_k) = (job.hidden, job.ne, job.inter, job.top_k);
    let n = lr.x.len() / hidden;
    let held = job.held;
    let variants = diag.map(|v| diag_variants(v, job.route_w)).unwrap_or_default();
    let nvar = variants.len();
    // route weights per pick, when captured
    let has_w = lr.w.len() == n * top_k && n > 0;
    anyhow::ensure!(job.route_w == RouteW::Off || has_w, "layer {}: route weighting needs the captured route weights ({} for {} picks)", job.li, lr.w.len(), n * top_k);
    let mut out = LayerOut { routed: Vec::new(), shared: None, n_gptq: 0, n_rtn: 0, diag: vec![DiagBin::new(nvar); BINS.len()], tim: [0.0; 5], t_shared: 0.0 };
    // the prior's sample: evenly spaced calibration tokens
    let cal: Vec<u32> = (0..n).filter(|&t| !held[t]).map(|t| t as u32).collect();
    let sample: Vec<u32> = cal.iter().step_by((cal.len() / job.prior_s.max(1)).max(1)).take(job.prior_s).copied().collect();
    if routed {
        // routing: each token's experts
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); ne];
        let mut lists_ho: Vec<Vec<u32>> = vec![Vec::new(); ne];
        // each list entry's route weight (when captured)
        let mut wl: Vec<Vec<f32>> = vec![Vec::new(); ne];
        let mut wl_ho: Vec<Vec<f32>> = vec![Vec::new(); ne];
        for t in 0..n {
            for j in t * top_k..(t + 1) * top_k {
                let e = lr.picks[j] as usize;
                if e >= ne {
                    continue;
                }
                let (l, w) = if held[t] { (&mut lists_ho[e], &mut wl_ho[e]) } else { (&mut lists[e], &mut wl[e]) };
                l.push(t as u32);
                if has_w {
                    w.push(lr.w[j]);
                }
            }
        }
        {
            let mut cnt: Vec<usize> = lists.iter().map(Vec::len).collect();
            cnt.sort_unstable();
            eprintln!(
                "  layer {}: routed rows per expert min {} median {} max {}; {} experts under {} rows",
                job.li,
                cnt[0],
                cnt[ne / 2],
                cnt[ne - 1],
                cnt.iter().filter(|&&v| v < job.min_rows).count(),
                job.min_rows
            );
        }
        // the prior with n0 pseudo-rows; route-weighted, the down prior's
        // rows carry the expert's mean squared route weight (pseudo-rows of
        // the average weighted row), so its share of the down Hessian stays
        let prior_for = |n0: f64, rw: Option<&[f32]>| -> Option<Prior> {
            let msw = rw.filter(|w| !w.is_empty()).map_or(1.0, |w| w.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / w.len() as f64);
            (n0 > 0.0).then(|| Prior { rows: &sample, n0, n0_down: n0 / 4.0 * msw })
        };
        let diag_layer = std::sync::Mutex::new(vec![DiagBin::new(nvar); BINS.len()]);
        // experts in parallel: each worker owns its device scratch and a
        // share of the host threads; results are kept in expert order
        let workers = std::env::var("CMF_MOE_WORKERS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4).clamp(1, 16);
        let wec = EncodeCfg { threads: (ec.threads / workers).max(1), ..*ec };
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: Vec<std::sync::Mutex<Option<[Vec<u8>; 3]>>> = (0..ne).map(|_| std::sync::Mutex::new(None)).collect();
        let timers = std::sync::Mutex::new([0f64; 5]);
        let (n_gptq, n_rtn) = (std::sync::atomic::AtomicUsize::new(0), std::sync::atomic::AtomicUsize::new(0));
        let worker = || -> anyhow::Result<()> {
            let mut wt = [0f64; 5];
            let mut bufs: Option<wgpu_bufs::Bufs> = None;
            loop {
                let e = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if e >= ne {
                    break;
                }
                if diag.is_some() {
                    let (ho, ce) = (&lists_ho[e], lists[e].len());
                    if ho.is_empty() || ce < job.min_rows {
                        continue;
                    }
                    let w = get(Part::Routed, e)?;
                    let xh = gather_rows(&lr.x, hidden, ho, 1.0, wec.threads);
                    let m = ho.len();
                    let o_fp = expert_out(&xh, m, hidden, &w.g, &w.u, &w.d, inter, ec.limit, wec.threads);
                    let (base, base_w) = row_err(&o_fp, &vec![0f32; o_fp.len()], hidden, &wl_ho[e]);
                    let err_of = |enc: &[Vec<u8>; 3]| -> (f64, f64) {
                        let [qg, qu, qd] = dequant_expert(enc, inter, hidden, ec.codec);
                        let oq = expert_out(&xh, m, hidden, &qg, &qu, &qd, inter, ec.limit, wec.threads);
                        row_err(&oq, &o_fp, hidden, &wl_ho[e])
                    };
                    let mut errs = vec![err_of(&rtn_expert(ec.codec, &w, inter, hidden))];
                    for &(n0, weighted) in &variants {
                        let rw = weighted.then(|| &wl[e][..]);
                        let enc = encode_expert(engine, &mut bufs, hidden, inter, &lr.x, &lists[e], rw, &w, &wec, prior_for(n0, rw), &mut wt)?;
                        errs.push(err_of(&enc));
                    }
                    let bin = BINS.iter().position(|&lim| ce < lim).unwrap_or(BINS.len() - 1);
                    let mut d = diag_layer.lock().unwrap();
                    d[bin].count += 1;
                    d[bin].base += base;
                    d[bin].base_w += base_w;
                    for (k, (v, vw)) in errs.into_iter().enumerate() {
                        d[bin].err[k] += v;
                        d[bin].err_w[k] += vw;
                    }
                    n_gptq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
                let w = get(Part::Routed, e)?;
                let enc = if lists[e].len() < job.min_rows {
                    n_rtn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    rtn_expert(ec.codec, &w, inter, hidden)
                } else {
                    n_gptq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let rw = (job.route_w == RouteW::On).then(|| &wl[e][..]);
                    encode_expert(engine, &mut bufs, hidden, inter, &lr.x, &lists[e], rw, &w, &wec, prior_for(job.prior_n0, rw), &mut wt)?
                };
                *results[e].lock().unwrap() = Some(enc);
            }
            let mut t = timers.lock().unwrap();
            for (a, b) in t.iter_mut().zip(wt) {
                *a += b;
            }
            Ok(())
        };
        std::thread::scope(|sc| -> anyhow::Result<()> {
            let hs: Vec<_> = (0..workers).map(|_| sc.spawn(&worker)).collect();
            for h in hs {
                h.join().map_err(|_| anyhow::anyhow!("expert worker panicked"))??;
            }
            Ok(())
        })?;
        out.tim = *timers.lock().unwrap();
        out.n_gptq = n_gptq.into_inner();
        out.n_rtn = n_rtn.into_inner();
        out.diag = diag_layer.into_inner().unwrap();
        if diag.is_none() {
            out.routed = results
                .into_iter()
                .enumerate()
                .map(|(e, s)| s.into_inner().unwrap().ok_or_else(|| anyhow::anyhow!("expert {e} missing")))
                .collect::<anyhow::Result<_>>()?;
        }
    }
    // the shared expert: every calibration token is its input
    if shared {
        let ts = Instant::now();
        let w = get(Part::Shared, 0)?;
        let is = w.g.len() / hidden;
        let mut st = [0f64; 5];
        let mut own: Option<wgpu_bufs::Bufs> = None;
        let enc = match (engine, main_bufs) {
            (Engine::Gpu(c), Some(b)) => encode_expert_gpu(c, b, hidden, is, &lr.x, &cal, None, &w, ec, None, &mut st)?,
            _ => encode_expert(engine, &mut own, hidden, is, &lr.x, &cal, None, &w, ec, None, &mut st)?,
        };
        out.shared = Some(enc);
        out.t_shared = ts.elapsed().as_secs_f64();
    }
    Ok(out)
}

/// The converter's encoding of an expert (round to nearest; q2tp v2).
pub(crate) fn rtn_expert(codec: Codec, w: &ExpertW, inter: usize, hidden: usize) -> [Vec<u8>; 3] {
    let q = codec.gate_up().1;
    [quantize_2d(q, &w.g, inter, hidden).1, quantize_2d(q, &w.u, inter, hidden).1, quantize_2d(Quant::Q4TiledP, &w.d, hidden, inter).1]
}

/// Layer-wide prior of the expert Hessians: `rows` (a sample of the layer's
/// tokens) enter both input Hessians as `n0` / `n0_down` pseudo-rows, so
/// directions an expert's own rows never visit carry the layer's statistics
/// instead of nothing (an expert routed fewer tokens than its input width
/// has a rank-deficient Hessian of its own).
#[derive(Clone, Copy)]
pub(crate) struct Prior<'a> {
    pub rows: &'a [u32],
    pub n0: f64,
    pub n0_down: f64,
}

/// GPTQ of one expert from its input rows (`rows` index the captured f16
/// rows `x`, `hidden` wide) on `engine`. `rw`: one route weight per row,
/// scaling that row's down input (`None`: unweighted). Returns the gate, up
/// and down payloads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_expert(
    engine: Engine,
    bufs: &mut Option<wgpu_bufs::Bufs>,
    hidden: usize,
    inter: usize,
    x: &[u16],
    rows: &[u32],
    rw: Option<&[f32]>,
    w: &ExpertW,
    ec: &EncodeCfg,
    prior: Option<Prior>,
    tim: &mut [f64; 5],
) -> anyhow::Result<[Vec<u8>; 3]> {
    anyhow::ensure!(rw.is_none_or(|v| v.len() == rows.len()), "{} route weights for {} rows", rw.map_or(0, <[f32]>::len), rows.len());
    match engine {
        Engine::Gpu(c) => {
            let b = bufs.get_or_insert_with(|| wgpu_bufs::Bufs::new(c, CHUNK, hidden, inter));
            encode_expert_gpu(c, b, hidden, inter, x, rows, rw, w, ec, prior, tim)
        }
        Engine::Host => encode_expert_host(hidden, inter, x, rows, rw, w, ec, prior, tim),
    }
}

/// [`encode_expert`] on the GPU: both input Hessians accumulate over
/// `CHUNK`-row device passes, then the gate/up fold and sweeps (q2tp:
/// placed rule + debias), the down fold and q4tp sweep.
#[allow(clippy::too_many_arguments)]
fn encode_expert_gpu(
    c: &Ctx,
    bufs: &wgpu_bufs::Bufs,
    hidden: usize,
    inter: usize,
    lr: &[u16],
    rows: &[u32],
    rw: Option<&[f32]>,
    w: &ExpertW,
    ec: &EncodeCfg,
    prior: Option<Prior>,
    tim: &mut [f64; 5],
) -> anyhow::Result<[Vec<u8>; 3]> {
    use cortiq_engine::gpu;
    let (wg, wu, wd) = (&w.g[..], &w.u[..], &w.d[..]);
    let threads = ec.threads;
    let ts = Instant::now();
    let hid = gpu::gram_alloc(hidden).ok_or_else(|| anyhow::anyhow!("gram alloc"))?;
    let did = gpu::gram_alloc(inter).ok_or_else(|| anyhow::anyhow!("gram alloc"))?;
    let (pg, pu) = (tm::f16_from_f32(c, wg), tm::f16_from_f32(c, wu));
    let gather = |xs: &mut [f32], part: &[u32], scale: f32| {
        par_rows(xs, hidden, threads, |r, row| {
            let t = part[r] as usize;
            for (d, &s) in row.iter_mut().zip(&lr[t * hidden..(t + 1) * hidden]) {
                *d = f16_to_f32(s) * scale;
            }
        });
    };
    // the down input of the m rows in bufs.x (gate/up in yg/yu) into yd,
    // times `scale` (and, with `per_row`, each row's weight in bufs.rw):
    // the clamped SwiGLU when the model has a limit
    let act = |m: usize, scale: f32, per_row: bool| -> anyhow::Result<()> {
        let d = if per_row {
            to::swiglu_lim_rows(c, m * inter, &bufs.yg, &bufs.yu, &bufs.yd, &bufs.rw, inter, ec.limit, scale)
        } else if ec.limit > 0.0 {
            to::swiglu_lim(c, m * inter, &bufs.yg, &bufs.yu, &bufs.yd, ec.limit, scale)
        } else {
            to::elt(c, "swiglu_fwd", m * inter, &bufs.yg, &bufs.yu, Some(&bufs.yd), None, None, None)
        };
        to::run(c, &[d.ok_or_else(|| anyhow::anyhow!("swiglu"))?]);
        Ok(())
    };
    let mut xe = vec![0f32; CHUNK.min(rows.len().max(prior.map_or(0, |p| p.rows.len()))) * hidden];
    for (ci, part) in rows.chunks(CHUNK).enumerate() {
        let m = part.len();
        let xs = &mut xe[..m * hidden];
        gather(xs, part, 1.0);
        to::write(c, &bufs.x, xs);
        anyhow::ensure!(gpu::gram_add_dev(hid, &bufs.x, m), "gram");
        tm::plane_linear(c, &pg, &bufs.x, &bufs.act16, &bufs.yg, m, inter, hidden, 1.0).ok_or_else(|| anyhow::anyhow!("gemm"))?;
        tm::plane_linear(c, &pu, &bufs.x, &bufs.act16, &bufs.yu, m, inter, hidden, 1.0).ok_or_else(|| anyhow::anyhow!("gemm"))?;
        if let Some(rw) = rw {
            to::write(c, &bufs.rw, &rw[ci * CHUNK..ci * CHUNK + m]);
        }
        act(m, 1.0, rw.is_some())?;
        anyhow::ensure!(gpu::gram_add_dev(did, &bufs.yd, m), "gram");
    }
    if let Some(pr) = prior.filter(|p| !p.rows.is_empty() && p.n0 > 0.0) {
        // S sample rows as n0 pseudo-rows: gram of a·x with a = √(n0/S); the
        // down input of the sample scaled by b = √(n0_down/S) — through
        // b·W_u without a limit (silu(g)⊙(b·u) = b·y), by the activation's
        // own output scale with one (the clamp does not commute with b)
        let s = pr.rows.len() as f64;
        let (a, b) = ((pr.n0 / s).sqrt() as f32, (pr.n0_down / s).sqrt() as f32);
        let pub_ = if ec.limit > 0.0 {
            None
        } else {
            let wub: Vec<f32> = wu.iter().map(|&v| v * b).collect();
            Some(tm::f16_from_f32(c, &wub))
        };
        for part in pr.rows.chunks(CHUNK) {
            let m = part.len();
            let xs = &mut xe[..m * hidden];
            gather(xs, part, a);
            to::write(c, &bufs.x, xs);
            anyhow::ensure!(gpu::gram_add_dev(hid, &bufs.x, m), "gram");
            gather(xs, part, 1.0);
            to::write(c, &bufs.x, xs);
            tm::plane_linear(c, &pg, &bufs.x, &bufs.act16, &bufs.yg, m, inter, hidden, 1.0).ok_or_else(|| anyhow::anyhow!("gemm"))?;
            tm::plane_linear(c, pub_.as_ref().unwrap_or(&pu), &bufs.x, &bufs.act16, &bufs.yu, m, inter, hidden, 1.0).ok_or_else(|| anyhow::anyhow!("gemm"))?;
            act(m, if pub_.is_some() { 1.0 } else { b }, false)?;
            anyhow::ensure!(gpu::gram_add_dev(did, &bufs.yd, m), "gram");
        }
    }
    drop(xe);
    // q2tp's debias needs H itself; the q4tp gate/up sweep only its fold
    let hgu = match ec.codec {
        Codec::Q2tp => Some(gpu::gram_read(hid).ok_or_else(|| anyhow::anyhow!("gram read"))?),
        Codec::Q4tp => None,
    };
    tim[0] += ts.elapsed().as_secs_f64();
    let ts = Instant::now();
    let fid = gpu::gptq_fold_dev(hid, hidden, ec.lambda).ok_or_else(|| anyhow::anyhow!("fold"))?;
    let (dead, hdiag, _) = gpu::gptq_fold_info(fid).unwrap_or_default();
    tim[1] += ts.elapsed().as_secs_f64();
    let ts = Instant::now();
    let (gb, ub) = match hgu {
        Some(hgu) => {
            let mut gb = gptq_quantize_q2tp_dev(wg, inter, hidden, fid, &dead, &hdiag, threads).ok_or_else(|| anyhow::anyhow!("q2tp sweep"))?;
            let mut ub = gptq_quantize_q2tp_dev(wu, inter, hidden, fid, &dead, &hdiag, threads).ok_or_else(|| anyhow::anyhow!("q2tp sweep"))?;
            gpu::gptq_fold_free(fid);
            // debias: H·w for every gate/up row, one device GEMM [W_g; W_u]·H
            let h32: Vec<f32> = hgu.iter().map(|&v| v as f32).collect();
            drop(hgu);
            let (hwg, hwu) = debias_products_dev(wg, wu, inter, hidden, &h32);
            drop(h32);
            q2tp_debias_rows(&mut gb, wg, inter, hidden, &hwg, threads);
            q2tp_debias_rows(&mut ub, wu, inter, hidden, &hwu, threads);
            (gb, ub)
        }
        None => {
            let gb = gptq_quantize_q4tp_dev(wg, inter, hidden, fid, &dead).ok_or_else(|| anyhow::anyhow!("q4tp sweep"))?;
            let ub = gptq_quantize_q4tp_dev(wu, inter, hidden, fid, &dead).ok_or_else(|| anyhow::anyhow!("q4tp sweep"))?;
            gpu::gptq_fold_free(fid);
            (gb, ub)
        }
    };
    tim[2] += ts.elapsed().as_secs_f64();
    let ts = Instant::now();
    let fd = gpu::gptq_fold_dev(did, inter, ec.lambda).ok_or_else(|| anyhow::anyhow!("fold"))?;
    let (dd, _, _) = gpu::gptq_fold_info(fd).unwrap_or_default();
    tim[3] += ts.elapsed().as_secs_f64();
    let ts = Instant::now();
    let db = gptq_quantize_q4tp_dev(wd, hidden, inter, fd, &dd).ok_or_else(|| anyhow::anyhow!("q4tp sweep"))?;
    gpu::gptq_fold_free(fd);
    tim[4] += ts.elapsed().as_secs_f64();
    Ok([gb, ub, db])
}

/// `W_g·H` and `W_u·H` (`inter × hidden` each, `h32` the `hidden²` gate/up
/// Hessian) for the q2tp debias, as one device GEMM `[W_g; W_u]·H` in exact
/// f32 FMA (the NVG encoder's GEMM; never the matrix units, whose operands
/// are f16/tf32 class) — the product the host path computes with an f32
/// GEMM too (`gemm_rows_square`), 2·inter·hidden² multiply-adds that cost
/// the host most of the q2tp gate/up stage.
pub(crate) fn debias_products_dev(wg: &[f32], wu: &[f32], inter: usize, hidden: usize, h32: &[f32]) -> (Vec<f32>, Vec<f32>) {
    use cortiq_engine::gpu_wgpu::nvg_dev::{DMat, gemm};
    let mut w = Vec::with_capacity(2 * inter * hidden);
    w.extend_from_slice(&wg[..inter * hidden]);
    w.extend_from_slice(&wu[..inter * hidden]);
    let a = DMat::from_f32(2 * inter, hidden, &w);
    drop(w);
    let h = DMat::from_f32(hidden, hidden, &h32[..hidden * hidden]);
    let c = DMat::zeros(2 * inter, hidden);
    gemm(1.0, &a, false, &h, false, 0.0, &c);
    let mut hwg = c.to_f32();
    let hwu = hwg.split_off(inter * hidden);
    (hwg, hwu)
}

/// [`encode_expert`] on the host: f64 Hessians from f32 chunk grams, the
/// host fold operator and the host sweeps (the reference the device path
/// approximates in f32).
#[allow(clippy::too_many_arguments)]
fn encode_expert_host(
    hidden: usize,
    inter: usize,
    lr: &[u16],
    rows: &[u32],
    rw: Option<&[f32]>,
    w: &ExpertW,
    ec: &EncodeCfg,
    prior: Option<Prior>,
    tim: &mut [f64; 5],
) -> anyhow::Result<[Vec<u8>; 3]> {
    const HCHUNK: usize = 4096;
    let threads = ec.threads;
    let ts = Instant::now();
    let mut hgu = vec![0f64; hidden * hidden];
    let mut hd = vec![0f64; inter * inter];
    // `pw`: per-row weights of the down input (route weights)
    let mut acc = |part: &[u32], pw: Option<&[f32]>, sx: f64, sy: f32| {
        let x = gather_rows(lr, hidden, part, 1.0, threads);
        let m = part.len();
        gram_add_host(&mut hgu, &x, m, hidden, sx, threads);
        let mut y = swiglu_rows(&x, m, hidden, &w.g, &w.u, inter, ec.limit, threads);
        match pw {
            Some(pw) => y.chunks_exact_mut(inter).zip(pw).for_each(|(row, &wr)| row.iter_mut().for_each(|v| *v *= wr * sy)),
            None if sy != 1.0 => y.iter_mut().for_each(|v| *v *= sy),
            None => {}
        }
        gram_add_host(&mut hd, &y, m, inter, 1.0, threads);
    };
    for (ci, part) in rows.chunks(HCHUNK).enumerate() {
        acc(part, rw.map(|v| &v[ci * HCHUNK..ci * HCHUNK + part.len()]), 1.0, 1.0);
    }
    if let Some(pr) = prior.filter(|p| !p.rows.is_empty() && p.n0 > 0.0) {
        let s = pr.rows.len() as f64;
        let (a2, b) = (pr.n0 / s, (pr.n0_down / s).sqrt() as f32);
        for part in pr.rows.chunks(HCHUNK) {
            acc(part, None, a2, b);
        }
    }
    tim[0] += ts.elapsed().as_secs_f64();
    Ok(encode_with_hessians(hidden, inter, w, hgu, hd, ec, tim))
}

/// The host encode of an expert from its two input Hessians (`hgu`
/// `hidden²`, `hd` `inter²`, f64 row-major): folds, gate/up sweeps (q2tp:
/// placed rule + output debias), down q4tp sweep.
pub(crate) fn encode_with_hessians(hidden: usize, inter: usize, w: &ExpertW, hgu: Vec<f64>, hd: Vec<f64>, ec: &EncodeCfg, tim: &mut [f64; 5]) -> [Vec<u8>; 3] {
    let threads = ec.threads;
    let ts = Instant::now();
    let hdiag: Vec<f32> = (0..hidden).map(|i| hgu[i * hidden + i] as f32).collect();
    let keep = (ec.codec == Codec::Q2tp).then(|| hgu.clone());
    let (u, dead) = gptq_fold_operator_par(hgu, hidden, ec.lambda, threads);
    tim[1] += ts.elapsed().as_secs_f64();
    let ts = Instant::now();
    let (gb, ub) = match &keep {
        Some(h) => {
            let mut gb = gptq_quantize_q2tp(&w.g, inter, hidden, &u, &dead, &hdiag, Q2tpRule::Placed, None, threads);
            let mut ub = gptq_quantize_q2tp(&w.u, inter, hidden, &u, &dead, &hdiag, Q2tpRule::Placed, None, threads);
            q2tp_debias_gemm(&mut gb, &w.g, inter, hidden, h, threads);
            q2tp_debias_gemm(&mut ub, &w.u, inter, hidden, h, threads);
            (gb, ub)
        }
        None => (
            gptq_quantize_q4tp(&w.g, inter, hidden, &u, &dead, &hdiag, threads),
            gptq_quantize_q4tp(&w.u, inter, hidden, &u, &dead, &hdiag, threads),
        ),
    };
    drop(u);
    tim[2] += ts.elapsed().as_secs_f64();
    let ts = Instant::now();
    let hddiag: Vec<f32> = (0..inter).map(|i| hd[i * inter + i] as f32).collect();
    let (ud, deadd) = gptq_fold_operator_par(hd, inter, ec.lambda, threads);
    tim[3] += ts.elapsed().as_secs_f64();
    let ts = Instant::now();
    let db = gptq_quantize_q4tp(&w.d, hidden, inter, &ud, &deadd, &hddiag, threads);
    tim[4] += ts.elapsed().as_secs_f64();
    [gb, ub, db]
}

/// `H += s·XᵀX` for the `m` rows of `x` (`cols` wide): one f32 GEMM of the
/// transposed chunk, accumulated in f64.
fn gram_add_host(h: &mut [f64], x: &[f32], m: usize, cols: usize, s: f64, threads: usize) {
    if m == 0 {
        return;
    }
    let mut xt = vec![0f32; cols * m];
    for r in 0..m {
        for (c, &v) in x[r * cols..(r + 1) * cols].iter().enumerate() {
            xt[c * m + r] = v;
        }
    }
    let g = gemm_nt(&xt, cols, m, &xt, cols, threads);
    for (a, &b) in h.iter_mut().zip(&g) {
        *a += s * b as f64;
    }
}

/// The f16 rows `part` of `x` (`hidden` wide) as f32, times `scale`.
fn gather_rows(x: &[u16], hidden: usize, part: &[u32], scale: f32, threads: usize) -> Vec<f32> {
    let mut out = vec![0f32; part.len() * hidden];
    par_rows(&mut out, hidden, threads, |r, row| {
        let t = part[r] as usize;
        for (d, &s) in row.iter_mut().zip(&x[t * hidden..(t + 1) * hidden]) {
            *d = f16_to_f32(s) * scale;
        }
    });
    out
}

/// `A·Wᵀ` for row-major `a` (`m × k`) and `w` (`n × k`), rows of `a` split
/// across `threads`.
fn gemm_nt(a: &[f32], m: usize, k: usize, w: &[f32], n: usize, threads: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    if m == 0 {
        return out;
    }
    let per = m.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * n).enumerate() {
            let rows = chunk.len() / n;
            let src = &a[ci * per * k..(ci * per + rows) * k];
            s.spawn(move || cortiq_engine::qtensor::sgemm_public(rows, n, k, 1.0, src, k, w, k, true, chunk, n));
        }
    });
    out
}

/// The down input `silu(min(g, L))·clamp(u, −L, L)` of the rows `x`
/// (`limit` L ≤ 0: no clamp).
#[allow(clippy::too_many_arguments)]
fn swiglu_rows(x: &[f32], m: usize, hidden: usize, wg: &[f32], wu: &[f32], inter: usize, limit: f32, threads: usize) -> Vec<f32> {
    let g = gemm_nt(x, m, hidden, wg, inter, threads);
    let u = gemm_nt(x, m, hidden, wu, inter, threads);
    g.iter()
        .zip(&u)
        .map(|(&g, &u)| {
            let (g, u) = if limit > 0.0 { (g.min(limit), u.clamp(-limit, limit)) } else { (g, u) };
            g / (1.0 + (-g).exp()) * u
        })
        .collect()
}

/// Expert output `swiglu(X·W_gᵀ, X·W_uᵀ)·W_dᵀ` of the rows `x` (`m × hidden`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_out(x: &[f32], m: usize, hidden: usize, wg: &[f32], wu: &[f32], wd: &[f32], inter: usize, limit: f32, threads: usize) -> Vec<f32> {
    let y = swiglu_rows(x, m, hidden, wg, wu, inter, limit, threads);
    gemm_nt(&y, m, inter, wd, hidden, threads)
}

/// Dequantized gate, up, down of an encoded expert.
pub(crate) fn dequant_expert(enc: &[Vec<u8>; 3], inter: usize, hidden: usize, codec: Codec) -> [Vec<f32>; 3] {
    let (mut g, mut u, mut d) = (vec![0f32; inter * hidden], vec![0f32; inter * hidden], vec![0f32; hidden * inter]);
    match codec {
        Codec::Q2tp => {
            cortiq_core::quant::dequant_q2tp(&enc[0], inter, hidden, &mut g);
            cortiq_core::quant::dequant_q2tp(&enc[1], inter, hidden, &mut u);
        }
        Codec::Q4tp => {
            cortiq_core::quant::dequant_q4tp(&enc[0], inter, hidden, &mut g);
            cortiq_core::quant::dequant_q4tp(&enc[1], inter, hidden, &mut u);
        }
    }
    cortiq_core::quant::dequant_q4tp(&enc[2], hidden, inter, &mut d);
    [g, u, d]
}

/// `f(row, row_slice)` over the `width`-wide rows of `v`, split across `threads`.
fn par_rows(v: &mut [f32], width: usize, threads: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let rows = v.len() / width.max(1);
    if rows == 0 {
        return;
    }
    let per = rows.div_ceil(threads.max(1)).max(1);
    let f = &f;
    std::thread::scope(|s| {
        for (ci, chunk) in v.chunks_mut(per * width).enumerate() {
            s.spawn(move || {
                for (k, row) in chunk.chunks_exact_mut(width).enumerate() {
                    f(ci * per + k, row);
                }
            });
        }
    });
}

/// Device scratch of the expert pass (`CHUNK` rows).
pub(crate) mod wgpu_bufs {
    use cortiq_engine::gpu_wgpu::{train_mm as tm, train_ops as to};
    pub struct Bufs {
        pub x: cortiq_engine::wgpu::Buffer,
        pub act16: cortiq_engine::wgpu::Buffer,
        pub yg: cortiq_engine::wgpu::Buffer,
        pub yu: cortiq_engine::wgpu::Buffer,
        pub yd: cortiq_engine::wgpu::Buffer,
        /// One route weight per row.
        pub rw: cortiq_engine::wgpu::Buffer,
    }
    impl Bufs {
        pub fn new(c: &cortiq_engine::gpu_wgpu::Ctx, rows: usize, hidden: usize, inter: usize) -> Self {
            let rows = tm::pad_rows(rows);
            Bufs {
                x: to::zeros(c, rows * hidden),
                act16: tm::f16_buf(c, rows * hidden),
                yg: to::zeros(c, rows * inter),
                yu: to::zeros(c, rows * inter),
                yd: to::zeros(c, rows * inter),
                rw: to::zeros(c, rows),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::tests::env_guard;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform in [−1, 1).
        fn uni(&mut self) -> f32 {
            ((self.next() >> 40) as f64 / (1u64 << 23) as f64 - 1.0) as f32
        }
        fn normal(&mut self) -> f32 {
            let u1 = ((self.next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
            let u2 = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        }
    }

    /// Weights whose 32-group amplitudes spread over a few powers of two,
    /// as real projection rows do.
    fn weights(rng: &mut Rng, rows: usize, cols: usize, scale: f32) -> Vec<f32> {
        let mut w = vec![0f32; rows * cols];
        for r in 0..rows {
            for g in 0..cols / 32 {
                let amp = scale * 2f32.powf(rng.uni() * 1.5);
                for c in g * 32..(g + 1) * 32 {
                    w[r * cols + c] = amp * rng.normal();
                }
            }
        }
        w
    }

    const FP4: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    /// The V4.1 routed-expert source format of a `[rows, cols]` matrix: FP4
    /// E2M1 nibble pairs `[rows, cols/2]` (low nibble = even element) and
    /// one E8M0 scale byte per 32 along the row.
    fn enc_fp4(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
        let (mut packed, mut scales) = (vec![0u8; rows * cols / 2], vec![0u8; rows * cols / 32]);
        for r in 0..rows {
            for g in 0..cols / 32 {
                let tile = &w[r * cols + g * 32..r * cols + g * 32 + 32];
                let amax = tile.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
                let k = if amax > 0.0 { ((amax / 6.0).log2().ceil() as i32 + 127).clamp(1, 254) } else { 127 };
                scales[r * cols / 32 + g] = k as u8;
                let s = 2f64.powi(k - 127);
                for (j, &v) in tile.iter().enumerate() {
                    let a = (v as f64 / s).abs();
                    let m = (0..8).min_by(|&x, &y| (FP4[x] - a).abs().partial_cmp(&(FP4[y] - a).abs()).unwrap()).unwrap() as u8;
                    let nib = m | if v < 0.0 { 8 } else { 0 };
                    let b = &mut packed[r * cols / 2 + g * 16 + j / 2];
                    *b |= if j % 2 == 0 { nib } else { nib << 4 };
                }
            }
        }
        (packed, scales)
    }

    /// Reference decode of [`enc_fp4`]'s format, in f64.
    fn dec_fp4(packed: &[u8], scales: &[u8], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                let b = packed[r * cols / 2 + c / 2];
                let nib = if c % 2 == 0 { b & 15 } else { b >> 4 };
                let v = FP4[(nib & 7) as usize] * 2f64.powi(scales[r * cols / 32 + c / 32] as i32 - 127);
                (if nib & 8 != 0 { -v } else { v }) as f32
            })
            .collect()
    }

    /// OCP E4M3 value of a byte, in f64 (no NaN codes are produced here).
    fn e4m3(b: u8) -> f64 {
        let (e, m) = (((b >> 3) & 15) as i32, (b & 7) as f64);
        let v = if e == 0 { m * 2f64.powi(-9) } else { (1.0 + m / 8.0) * 2f64.powi(e - 7) };
        if b & 0x80 != 0 { -v } else { v }
    }

    /// The V4.1 FP8 format: E4M3 bytes `[rows, cols]` and one E8M0 scale per
    /// `block`×`block` tile.
    fn enc_fp8(w: &[f32], rows: usize, cols: usize, block: usize) -> (Vec<u8>, Vec<u8>) {
        let (sr, sc) = (rows.div_ceil(block), cols.div_ceil(block));
        let mut scales = vec![127u8; sr * sc];
        for br in 0..sr {
            for bc in 0..sc {
                let mut amax = 0f64;
                for r in br * block..((br + 1) * block).min(rows) {
                    for c in bc * block..((bc + 1) * block).min(cols) {
                        amax = amax.max(w[r * cols + c].abs() as f64);
                    }
                }
                if amax > 0.0 {
                    scales[br * sc + bc] = ((amax / 448.0).log2().ceil() as i32 + 127).clamp(1, 254) as u8;
                }
            }
        }
        let bytes = (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                let v = w[i] as f64 / 2f64.powi(scales[(r / block) * sc + c / block] as i32 - 127);
                let code = (0u8..=0x7e).min_by(|&x, &y| (e4m3(x) - v.abs()).abs().partial_cmp(&(e4m3(y) - v.abs()).abs()).unwrap()).unwrap();
                code | if v < 0.0 { 0x80 } else { 0 }
            })
            .collect();
        (bytes, scales)
    }

    fn dec_fp8(bytes: &[u8], scales: &[u8], rows: usize, cols: usize, block: usize) -> Vec<f32> {
        let sc = cols.div_ceil(block);
        (0..rows * cols).map(|i| (e4m3(bytes[i]) * 2f64.powi(scales[(i / cols / block) * sc + (i % cols) / block] as i32 - 127)) as f32).collect()
    }

    fn write_safetensors(path: &Path, tensors: &[(String, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            header.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}));
        }
        let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&h);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// The source tensors of one V4.1 layer: `ne` FP4 routed experts and an
    /// FP8 shared expert (`w1` gate, `w3` up, `w2` down), with the exact
    /// decoded weights they stand for.
    fn dsv41_layer_source(rng: &mut Rng, li: usize, ne: usize, hidden: usize, inter: usize) -> (Vec<(String, &'static str, Vec<usize>, Vec<u8>)>, Vec<ExpertW>, ExpertW) {
        let mut t = Vec::new();
        let mut routed = Vec::new();
        let mats = |stem: String, fp4: bool, rng: &mut Rng, t: &mut Vec<(String, &'static str, Vec<usize>, Vec<u8>)>| -> ExpertW {
            let mut one = |w: &str, rows: usize, cols: usize| -> Vec<f32> {
                let v = weights(rng, rows, cols, 0.05);
                if fp4 {
                    let (p, s) = enc_fp4(&v, rows, cols);
                    let dec = dec_fp4(&p, &s, rows, cols);
                    t.push((format!("{stem}.{w}.weight"), "I8", vec![rows, cols / 2], p));
                    t.push((format!("{stem}.{w}.scale"), "F8_E8M0", vec![rows, cols / 32], s));
                    dec
                } else {
                    let (b, s) = enc_fp8(&v, rows, cols, 32);
                    let dec = dec_fp8(&b, &s, rows, cols, 32);
                    t.push((format!("{stem}.{w}.weight"), "F8_E4M3", vec![rows, cols], b));
                    t.push((format!("{stem}.{w}.scale"), "F8_E8M0", vec![rows.div_ceil(32), cols.div_ceil(32)], s));
                    dec
                }
            };
            ExpertW { g: one("w1", inter, hidden), u: one("w3", inter, hidden), d: one("w2", hidden, inter) }
        };
        for e in 0..ne {
            routed.push(mats(format!("layers.{li}.ffn.experts.{e}"), true, rng, &mut t));
        }
        let shared = mats(format!("layers.{li}.ffn.shared_experts"), false, rng, &mut t);
        (t, routed, shared)
    }

    /// Correlated MoE inputs: a low-rank signal with a decaying spectrum
    /// plus a little isotropic noise, the structure GPTQ exploits.
    fn inputs(rng: &mut Rng, n: usize, hidden: usize) -> Vec<f32> {
        let rank = hidden / 4;
        let basis: Vec<f32> = (0..rank * hidden).map(|_| rng.normal()).collect();
        let mut x = vec![0f32; n * hidden];
        for t in 0..n {
            let z: Vec<f32> = (0..rank).map(|k| rng.normal() / (1.0 + k as f32 * 0.3)).collect();
            for c in 0..hidden {
                let mut v = 0.05 * rng.normal();
                for (k, &zk) in z.iter().enumerate() {
                    v += zk * basis[k * hidden + c];
                }
                x[t * hidden + c] = v * 0.5;
            }
        }
        x
    }

    /// Flash-Next capture rows → f16 inputs and the top-k of the f16-rounded
    /// router logits (the routing the Flash-Next path always used).
    #[test]
    fn qwen4_rows_route_by_f16_logit_topk() {
        let (hidden, ne, top_k, n) = (8usize, 12usize, 3usize, 50usize);
        let mut rng = Rng(5);
        let r: Vec<f32> = (0..n * (hidden + ne)).map(|_| rng.normal()).collect();
        let mut out = LayerRows { x: Vec::new(), picks: Vec::new(), w: Vec::new() };
        qwen4_layer_rows(&r, hidden, ne, top_k, &mut out);
        assert_eq!((out.x.len(), out.picks.len()), (n * hidden, n * top_k));
        for t in 0..n {
            let row = &r[t * (hidden + ne)..(t + 1) * (hidden + ne)];
            assert!(out.x[t * hidden..(t + 1) * hidden].iter().zip(row).all(|(&h, &v)| h == f32_to_f16(v)));
            let lg: Vec<f32> = row[hidden..].iter().map(|&v| f16_to_f32(f32_to_f16(v))).collect();
            let mut want: Vec<usize> = (0..ne).collect();
            want.sort_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap());
            let mut want: Vec<u16> = want[..top_k].iter().map(|&e| e as u16).collect();
            let mut got = out.picks[t * top_k..(t + 1) * top_k].to_vec();
            want.sort_unstable();
            got.sort_unstable();
            assert_eq!(got, want, "token {t}");
        }
    }

    /// Settings read from the header: V4.1 top-k and SwiGLU limit from the
    /// runtime config, the FP8 block from the preserved quantization config;
    /// Flash-Next's top-k from its `moe` block.
    #[test]
    fn header_derived_settings() {
        let arch = |moe: Option<serde_json::Value>| -> cortiq_core::types::ModelArch {
            let mut v = serde_json::json!({
                "arch_name": "deepseek_v41", "hidden_size": 64, "intermediate_size": 32, "num_layers": 2,
                "num_attention_heads": 1, "num_kv_heads": 1, "head_dim": 32, "vocab_size": 16,
                "layer_types": ["FullAttention", "FullAttention"], "rms_norm_eps": 1e-6, "max_position_embeddings": 64,
                "linear_conv_kernel_dim": 0, "linear_num_key_heads": 0, "linear_num_value_heads": 0,
            });
            if let Some(m) = moe {
                v["moe"] = m;
            }
            serde_json::from_value(v).unwrap()
        };
        let a = arch(None);
        let src = |limit: f64, block: Option<serde_json::Value>| {
            let mut v = serde_json::json!({"text_config": {"hidden_size": 64, "num_hidden_layers": 2, "num_experts_per_tok": 6, "n_routed_experts": 384, "swiglu_limit": limit, "scoring_func": "sqrtsoftplus"}});
            if let Some(b) = block {
                v["quantization_config"] = serde_json::json!({"weight_block_size": b});
            }
            v
        };
        let settings = |s: &serde_json::Value| dsv41_settings(&cortiq_engine::dsv41::Dsv41Cfg::from_source(s, &a).unwrap().0, Some(s));
        assert_eq!(settings(&src(10.0, Some(serde_json::json!([128, 128])))), (6, 10.0, 128));
        assert_eq!(settings(&src(7.5, None)), (6, 7.5, 32), "no quantization_config: 32×32 tiles");
        assert_eq!(settings(&src(-1.0, Some(serde_json::json!([32, 32])))), (6, 0.0, 32), "a negative limit is none");
        assert_eq!(qwen4_top_k(&a), 10, "no moe block");
        let m = serde_json::json!({"num_experts": 512, "top_k": 8, "moe_intermediate_size": 128, "routed_scaling_factor": 1.0});
        assert_eq!(qwen4_top_k(&arch(Some(m))), 8);
    }

    /// The pass plan: `per` layers a pass, one pass in place.
    #[test]
    fn passes_cover_the_range() {
        assert_eq!(plan_passes(0, 40, 24, false), vec![(0, 24), (24, 40)]);
        assert_eq!(plan_passes(0, 40, 40, false), vec![(0, 40)]);
        assert_eq!(plan_passes(20, 22, 24, false), vec![(20, 22)]);
        assert_eq!(plan_passes(0, 40, 24, true), vec![(0, 40)]);
        assert_eq!(plan_passes(3, 10, 0, false).len(), 7, "per 0 means one layer a pass");
    }

    /// A saved capture loads back bit for bit; a directory holding the
    /// capture of another run (other tokens, other geometry) is refused,
    /// and no temporary file is left behind.
    #[test]
    fn capture_dir_round_trip_and_refusal() {
        let dir = std::env::temp_dir().join(format!("cortiq-moe-capture-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut rng = Rng(41);
        let windows: Vec<Vec<u32>> = (0..3).map(|k| (0..20u32).map(|i| i * 7 + k).collect()).collect();
        let (hidden, top_k, ne) = (16usize, 2usize, 5usize);
        let ntok = 60;
        let want = CaptureMeta {
            token_hash: token_hash(&windows),
            ntok,
            window: 20,
            windows: 3,
            hidden,
            top_k,
            ne,
            family: "Dsv41".into(),
            calib_model: "/pod/calib.cmf".into(),
            calib_size: 123,
            has_w: true,
            layers: Vec::new(),
        };
        let mut meta = capture_open(&dir, &want).unwrap();
        assert!(meta.layers.is_empty());
        let mk = |rng: &mut Rng| LayerRows {
            x: (0..ntok * hidden).map(|_| f32_to_f16(rng.normal())).collect(),
            picks: (0..ntok * top_k).map(|_| (rng.next() % ne as u64) as u16).collect(),
            w: (0..ntok * top_k).map(|_| rng.uni()).collect(),
        };
        let rows = [mk(&mut rng), mk(&mut rng)];
        for (k, li) in [3usize, 4].into_iter().enumerate() {
            capture_save_layer(&dir, &meta, li, &rows[k]).unwrap();
            meta.layers.push(li);
        }
        capture_save_meta(&dir, &meta).unwrap();
        assert_eq!(meta.layer_bytes(), (ntok * (hidden * 2 + top_k * 6)) as u64);
        assert_eq!(std::fs::metadata(dir.join("L3.x")).unwrap().len(), (ntok * hidden * 2) as u64);
        assert!(std::fs::read_dir(&dir).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().ends_with(".tmp")));
        // the same run (another calibration file path is only a note) loads it
        let again = capture_open(&dir, &CaptureMeta { calib_model: "/elsewhere.cmf".into(), ..want.clone() }).unwrap();
        assert_eq!(again.layers, vec![3, 4]);
        for (k, li) in [3usize, 4].into_iter().enumerate() {
            let got = capture_load_layer(&dir, &again, li).unwrap();
            assert_eq!((got.x, got.picks), (rows[k].x.clone(), rows[k].picks.clone()), "layer {li}");
            assert_eq!(got.w.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), rows[k].w.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        assert!(capture_load_layer(&dir, &again, 5).is_err(), "a layer not saved");
        // another run is refused
        let mut other = windows.clone();
        other[1][5] += 1;
        let e = capture_open(&dir, &CaptureMeta { token_hash: token_hash(&other), ..want.clone() }).unwrap_err().to_string();
        assert!(e.contains("calibration tokens differ"), "{e}");
        let e = capture_open(&dir, &CaptureMeta { hidden: 32, ..want.clone() }).unwrap_err().to_string();
        assert!(e.contains("hidden 16 saved, 32 now"), "{e}");
        let e = capture_open(&dir, &CaptureMeta { top_k: 3, ne: 6, ..want.clone() }).unwrap_err().to_string();
        assert!(e.contains("top_k") && e.contains("experts"), "{e}");
        // a truncated layer file is an error, not short rows
        std::fs::write(dir.join("L4.picks"), [0u8; 10]).unwrap();
        assert!(capture_load_layer(&dir, &again, 4).is_err());
        // a window split differently is another run, even with the same ids
        let flat: Vec<u32> = windows.concat();
        assert_ne!(token_hash(&[flat]), token_hash(&windows));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A shard download is retried until it succeeds, and gives up with
    /// the last error once the next pause would end past the time budget
    /// (growing pauses, capped).
    #[test]
    fn downloads_are_retried() {
        use std::time::Duration;
        let ms = Duration::from_millis;
        let mut calls = 0;
        with_retries(Duration::from_secs(60), Duration::ZERO, Duration::ZERO, "shard", || {
            calls += 1;
            if calls < 3 { anyhow::bail!("transient {calls}") } else { Ok(()) }
        })
        .unwrap();
        assert_eq!(calls, 3);
        // pauses 10, 20, 30, 30, … ms within a 100 ms budget: the pauses
        // after attempts 1..4 end at 10, 30, 60, 90 ms, the one after the
        // 5th would end at 120 > 100 (4 when the sleeps overshoot)
        let mut calls = 0;
        let e = with_retries(ms(100), ms(10), ms(30), "shard", || {
            calls += 1;
            anyhow::bail!("down {calls}")
        })
        .unwrap_err();
        assert!((4..=5).contains(&calls), "{calls} attempts");
        assert!(format!("{e:#}").contains(&format!("down {calls}")) && format!("{e:#}").contains("CMF_MOE_FETCH_WAIT"), "{e:#}");
        // a zero budget: one attempt
        let mut calls = 0;
        assert!(with_retries(Duration::ZERO, ms(10), ms(10), "shard", || {
            calls += 1;
            anyhow::bail!("down")
        })
        .is_err());
        assert_eq!(calls, 1);
    }

    /// Calibration windows: plain chunks without a BOS (Flash-Next, as
    /// before); with one, every window starts with it at `window` tokens.
    #[test]
    fn calib_windows_start_with_bos() {
        let ids: Vec<u32> = (1..=100).collect();
        let plain = calib_windows(&ids, 32, None);
        assert_eq!(plain, ids.chunks(32).take(3).map(<[u32]>::to_vec).collect::<Vec<_>>(), "the 4-token tail is dropped");
        let w = calib_windows(&ids, 32, Some(0));
        assert_eq!(w.len(), 3, "the 7-token tail (8 with BOS) is dropped");
        for (k, win) in w.iter().enumerate() {
            assert_eq!(win.len(), 32);
            assert_eq!(win[0], 0);
            assert_eq!(&win[1..], &ids[k * 31..(k + 1) * 31]);
        }
        // a chunk that already starts with BOS gets no second one
        let w = calib_windows(&[7u32; 40], 20, Some(7));
        assert!(w.iter().all(|x| x[0] == 7 && x.len() == 19));
    }

    /// The V4.1 prior defaults to one hidden width while the mean expert
    /// sees fewer rows than that; CMF_MOE_PRIOR always wins; Flash-Next keeps
    /// no default.
    #[test]
    fn production_prior_defaults() {
        let (n0, why) = production_prior(Family::Dsv41, None, 32768, 6, 384, 5120);
        assert_eq!(n0, 5120.0, "{why}");
        assert!(why.contains("512 routed rows"), "{why}");
        assert_eq!(production_prior(Family::Dsv41, None, 400_000, 6, 384, 5120).0, 0.0, "enough rows: no default");
        assert_eq!(production_prior(Family::Dsv41, Some(2560.0), 32768, 6, 384, 5120).0, 2560.0);
        assert_eq!(production_prior(Family::Dsv41, Some(0.0), 32768, 6, 384, 5120).0, 0.0, "CMF_MOE_PRIOR=0 turns it off");
        assert_eq!(production_prior(Family::Qwen4, None, 32768, 10, 512, 2560).0, 0.0, "Flash-Next unchanged");
    }

    /// `--output` / `--patch-shared` naming the calibration file are caught
    /// through a symlink and a hard link too; another file, or one that does
    /// not exist yet, is not the same file.
    #[test]
    fn output_naming_the_calibration_file_is_caught() {
        let dir = std::env::temp_dir().join(format!("cortiq-moe-same-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let calib = dir.join("calib.cmf");
        std::fs::write(&calib, b"calibration").unwrap();
        std::fs::write(dir.join("other.cmf"), b"calibration").unwrap();
        assert!(same_file(&calib, &calib));
        assert!(same_file(&dir.join(".").join("calib.cmf"), &calib));
        assert!(!same_file(&dir.join("other.cmf"), &calib));
        assert!(!same_file(&dir.join("new.cmf"), &calib));
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&calib, dir.join("link.cmf")).unwrap();
            std::fs::hard_link(&calib, dir.join("hard.cmf")).unwrap();
            assert!(same_file(&dir.join("link.cmf"), &calib));
            assert!(same_file(&dir.join("hard.cmf"), &calib));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The free-space check: what fits passes, a shortfall stops the run
    /// (unless allowed), needs on one filesystem add up, a file about to be
    /// truncated counts as free, and a path that does not exist yet is
    /// measured on its nearest existing ancestor.
    #[test]
    #[cfg(unix)]
    fn free_space_check_refuses_what_does_not_fit() {
        let dir = std::env::temp_dir().join(format!("cortiq-moe-space-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (_, avail) = fs_space(&dir.join("not/yet/out.cmf")).expect("statvfs");
        assert!(avail > 0);
        let need = |b: u64| (dir.join("out.cmf"), "--output".to_string(), b);
        check_free_space(&[need(1 << 20)], &[], false).unwrap();
        let e = check_free_space(&[need(avail + (1 << 40))], &[], false).unwrap_err().to_string();
        assert!(e.contains("not enough free disk space") && e.contains("CMF_MOE_NO_SPACE_CHECK=1"), "{e}");
        check_free_space(&[need(avail + (1 << 40))], &[], true).unwrap();
        // two needs on one filesystem are summed
        let half = avail / 2 + (1 << 30);
        assert!(check_free_space(&[need(half), (dir.join("cap"), "capture".into(), half)], &[], false).is_err());
        // an old output that the run truncates first is space it can use
        let old = dir.join("old.cmf");
        std::fs::write(&old, vec![1u8; 1 << 22]).unwrap();
        assert!(sole_file_bytes(&old) >= 1 << 22);
        assert_eq!(sole_file_bytes(&dir.join("missing.cmf")), 0);
        std::fs::hard_link(&old, dir.join("old-link.cmf")).unwrap();
        assert_eq!(sole_file_bytes(&old), 0, "another name keeps the blocks");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The hole-punching probe answers without leaving its file behind:
    /// unsupported off Linux, and on Linux whatever the temp filesystem does.
    #[test]
    fn hole_punch_probe_cleans_up() {
        let dir = std::env::temp_dir().join(format!("cortiq-moe-punch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let r = probe_hole_punching(&dir);
        eprintln!("hole punching in {}: {r:?}", dir.display());
        if cfg!(not(target_os = "linux")) {
            assert!(r.is_err());
        }
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "the probe file is removed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The FP4 and FP8 source tensors decode bit for bit to the weights
    /// they encode (the GPTQ target is the exact source, as the converter
    /// sees it).
    #[test]
    fn dsv41_source_decodes_fp4_and_fp8_exactly() {
        let dir = std::env::temp_dir().join(format!("cortiq-moe-src-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut rng = Rng(7);
        let (hidden, inter) = (96, 64);
        let (t, routed, shared) = dsv41_layer_source(&mut rng, 3, 2, hidden, inter);
        let shard = dir.join("model-00001-of-00001.safetensors");
        write_safetensors(&shard, &t);
        let src = LayerSrc::from_paths(&[shard]).unwrap();
        for (e, want) in routed.iter().enumerate() {
            let got = dsv41_expert(&src, 3, Part::Routed, e, hidden, inter, 32).unwrap();
            assert_eq!(got.g, want.g, "expert {e} gate");
            assert_eq!(got.u, want.u, "expert {e} up");
            assert_eq!(got.d, want.d, "expert {e} down");
            assert!(got.g.iter().any(|&v| v != 0.0));
        }
        let got = dsv41_expert(&src, 3, Part::Shared, 0, hidden, inter, 32).unwrap();
        assert_eq!((got.g, got.u, got.d), (shared.g.clone(), shared.u.clone(), shared.d.clone()), "shared expert");
        // a wrong geometry is refused, not reinterpreted
        assert!(dsv41_expert(&src, 3, Part::Routed, 0, hidden, inter + 32, 32).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With an identity Hessian GPTQ has nothing to trade between channels:
    /// no residual is fed forward and the encode is plain rounding. q2tp
    /// gate/up come out byte for byte as the converter's v2 encoder (placed
    /// ladder, nearest codes, weight-space debias). Every q4tp matrix keeps
    /// the converter's row ladders, rounds every weight to the nearest point
    /// of its group's grid, and picks the converter's rung in all but a
    /// handful of groups (the host sweep centres its rung window on the libm
    /// `log2` of absmax/7, the converter on its deterministic `log2` of the
    /// f16 scale; the error search then runs over windows one rung apart).
    #[test]
    fn identity_hessian_gptq_is_plain_rounding() {
        use cortiq_core::quant::{q4tp_code, q4tp_ladder, q4tp_sections};
        let _env = env_guard("4", "1");
        let mut rng = Rng(11);
        let (hidden, inter) = (128, 64);
        let w = ExpertW { g: weights(&mut rng, inter, hidden, 0.05), u: weights(&mut rng, inter, hidden, 0.05), d: weights(&mut rng, hidden, inter, 0.05) };
        let eye = |n: usize| -> Vec<f64> { (0..n * n).map(|i| if i / n == i % n { 1.0 } else { 0.0 }).collect() };
        let (mut groups, mut moved) = (0usize, 0usize);
        for codec in [Codec::Q4tp, Codec::Q2tp] {
            let ec = EncodeCfg { codec, limit: 10.0, lambda: 0.01, threads: 3 };
            let got = encode_with_hessians(hidden, inter, &w, eye(hidden), eye(inter), &ec, &mut [0.0; 5]);
            let want = rtn_expert(codec, &w, inter, hidden);
            for (k, (name, src)) in [("gate", &w.g), ("up", &w.u), ("down", &w.d)].into_iter().enumerate() {
                assert_eq!(got[k].len(), want[k].len(), "{codec:?} {name}: payload size");
                if codec == Codec::Q2tp && k < 2 {
                    assert_eq!(got[k], want[k], "{codec:?} {name}: the converter's q2tp v2 bytes");
                    continue;
                }
                let (rows, cols) = if k == 2 { (hidden, inter) } else { (inter, hidden) };
                let (po, co, stride) = q4tp_sections(rows, cols);
                assert_eq!(got[k][po..po + rows * 4], want[k][po..po + rows * 4], "{codec:?} {name}: row ladders");
                for r in 0..rows {
                    let tab = q4tp_ladder(&got[k][po..], r);
                    for g in 0..cols / 32 {
                        let c = q4tp_code(&got[k][co + r * stride..co + (r + 1) * stride], g);
                        groups += 1;
                        moved += (c != q4tp_code(&want[k][co + r * stride..co + (r + 1) * stride], g)) as usize;
                        let s = tab[c];
                        for j in 0..32 {
                            let col = g * 32 + j;
                            let byte = got[k][(r * cols + col) / 2];
                            let q = (if col % 2 == 0 { byte & 15 } else { byte >> 4 }) as f32 - 8.0;
                            let wv = src[r * cols + col];
                            let best = (-8..=7).map(|l| (wv - l as f32 * s).abs()).fold(f32::INFINITY, f32::min);
                            assert!((wv - q * s).abs() <= best + 1e-6 * s, "{codec:?} {name} [{r},{col}]: {q}·{s} is not the nearest grid point to {wv}");
                        }
                    }
                }
            }
        }
        eprintln!("q4tp rungs differing from the converter: {moved} of {groups} groups");
        assert!(moved * 100 <= groups, "{moved} of {groups} q4tp groups moved off the converter's rung");
    }

    /// On held-out rows of the same input distribution, GPTQ (with and
    /// without the layer prior) reconstructs the expert output better than
    /// round-to-nearest, in both layouts, with the SwiGLU clamp active.
    #[test]
    fn gptq_beats_rounding_on_held_out_rows() {
        let _env = env_guard("4", "1");
        let mut rng = Rng(23);
        let (hidden, inter, n_cal, n_ho) = (128, 64, 1024, 256);
        let x32 = inputs(&mut rng, n_cal + n_ho, hidden);
        let x: Vec<u16> = x32.iter().map(|&v| f32_to_f16(v)).collect();
        let xh: Vec<f32> = x[n_cal * hidden..].iter().map(|&h| f16_to_f32(h)).collect();
        let w = ExpertW { g: weights(&mut rng, inter, hidden, 0.15), u: weights(&mut rng, inter, hidden, 0.15), d: weights(&mut rng, hidden, inter, 0.1) };
        let limit = 1.0;
        let o_fp = expert_out(&xh, n_ho, hidden, &w.g, &w.u, &w.d, inter, limit, 2);
        // the clamp is exercised: some gate/up pre-activations exceed it
        let pre = gemm_nt(&xh, n_ho, hidden, &w.g, inter, 2);
        assert!(pre.iter().filter(|v| v.abs() > limit).count() > pre.len() / 50, "clamp not exercised");
        let err = |enc: &[Vec<u8>; 3], codec: Codec| -> f64 {
            let [g, u, d] = dequant_expert(enc, inter, hidden, codec);
            let o = expert_out(&xh, n_ho, hidden, &g, &u, &d, inter, limit, 2);
            o.iter().zip(&o_fp).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>() / o_fp.iter().map(|&v| (v as f64).powi(2)).sum::<f64>()
        };
        let rows: Vec<u32> = (0..n_cal as u32).collect();
        let sample: Vec<u32> = (0..n_cal as u32).step_by(4).collect();
        for codec in [Codec::Q4tp, Codec::Q2tp] {
            let ec = EncodeCfg { codec, limit, lambda: 0.01, threads: 3 };
            let rtn = err(&rtn_expert(codec, &w, inter, hidden), codec);
            let gptq = err(&encode_expert(Engine::Host, &mut None, hidden, inter, &x, &rows, None, &w, &ec, None, &mut [0.0; 5]).unwrap(), codec);
            let prior = Prior { rows: &sample, n0: hidden as f64, n0_down: hidden as f64 / 4.0 };
            let gptq_p = err(&encode_expert(Engine::Host, &mut None, hidden, inter, &x, &rows, None, &w, &ec, Some(prior), &mut [0.0; 5]).unwrap(), codec);
            eprintln!("{codec:?}: held-out relative output error RTN {rtn:.5}, GPTQ {gptq:.5}, GPTQ + prior {gptq_p:.5}");
            assert!(gptq < 0.9 * rtn, "{codec:?}: GPTQ {gptq} not better than RTN {rtn}");
            assert!(gptq_p < rtn, "{codec:?}: GPTQ with the prior {gptq_p} not better than RTN {rtn}");
        }
    }

    /// The down input with DeepSeek's clamp against hand values: the gate
    /// is clamped from above only (12 → 10, −12 stays), the up branch on both
    /// sides (−15 → −10, 3 stays); without a limit nothing is clamped.
    #[test]
    fn swiglu_rows_clamp_matches_hand_values() {
        let x = [1.0f32, 0.0, 0.0, 0.0]; // picks the first column of each weight row
        let wg = [12.0f32, 0.5, 0.5, 0.5, -12.0, 0.5, 0.5, 0.5];
        let wu = [-15.0f32, 0.5, 0.5, 0.5, 3.0, 0.5, 0.5, 0.5];
        let silu = |v: f64| v / (1.0 + (-v).exp());
        let close = |got: &[f32], want: [f64; 2]| {
            for (g, w) in got.iter().zip(want) {
                assert!((*g as f64 - w).abs() <= 1e-6 * w.abs(), "{got:?} vs {want:?}");
            }
        };
        close(&swiglu_rows(&x, 1, 4, &wg, &wu, 2, 10.0, 1), [silu(10.0) * -10.0, silu(-12.0) * 3.0]);
        close(&swiglu_rows(&x, 1, 4, &wg, &wu, 2, 0.0, 1), [silu(12.0) * -15.0, silu(-12.0) * 3.0]);
    }

    /// An expert routed fewer rows than its input width at hidden 256.
    struct LowRows {
        hidden: usize,
        inter: usize,
        x: Vec<u16>,
        xh: Vec<f32>,
        n_ho: usize,
        sample: Vec<u32>,
        w: ExpertW,
    }

    /// 2048 layer rows (the prior's sample: every 4th) and 256 held-out rows
    /// of one correlated distribution; the expert's own rows are the first
    /// rows of the layer.
    fn low_rows_case() -> LowRows {
        let mut rng = Rng(47);
        let (hidden, inter, n_layer, n_ho) = (256, 128, 2048, 256);
        let x32 = inputs(&mut rng, n_layer + n_ho, hidden);
        let x: Vec<u16> = x32.iter().map(|&v| f32_to_f16(v)).collect();
        let xh: Vec<f32> = x[n_layer * hidden..].iter().map(|&h| f16_to_f32(h)).collect();
        let w = ExpertW { g: weights(&mut rng, inter, hidden, 0.15), u: weights(&mut rng, inter, hidden, 0.15), d: weights(&mut rng, hidden, inter, 0.1) };
        LowRows { hidden, inter, x, xh, n_ho, sample: (0..n_layer as u32).step_by(4).collect(), w }
    }

    /// Held-out relative output error of RTN, GPTQ and GPTQ with the prior
    /// (n0 = hidden) for an expert of `m` rows, on `engine`.
    fn low_rows_errors(case: &LowRows, engine: Engine, m: usize, codec: Codec, limit: f32) -> (f64, f64, f64) {
        let LowRows { hidden, inter, ref x, ref xh, n_ho, ref sample, ref w } = *case;
        let o_fp = expert_out(xh, n_ho, hidden, &w.g, &w.u, &w.d, inter, limit, 2);
        let err = |enc: &[Vec<u8>; 3]| -> f64 {
            let [g, u, d] = dequant_expert(enc, inter, hidden, codec);
            let o = expert_out(xh, n_ho, hidden, &g, &u, &d, inter, limit, 2);
            o.iter().zip(&o_fp).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>() / o_fp.iter().map(|&v| (v as f64).powi(2)).sum::<f64>()
        };
        let ec = EncodeCfg { codec, limit, lambda: 0.01, threads: 3 };
        let rows: Vec<u32> = (0..m as u32).collect();
        let prior = Prior { rows: sample, n0: hidden as f64, n0_down: hidden as f64 / 4.0 };
        let rtn = err(&rtn_expert(codec, w, inter, hidden));
        let gptq = err(&encode_expert(engine, &mut None, hidden, inter, x, &rows, None, w, &ec, None, &mut [0.0; 5]).unwrap());
        let gptq_p = err(&encode_expert(engine, &mut None, hidden, inter, x, &rows, None, w, &ec, Some(prior), &mut [0.0; 5]).unwrap());
        (rtn, gptq, gptq_p)
    }

    /// Experts routed 48, 128 or 192 rows at hidden 256 have rank-deficient
    /// Hessians of their own (most V4.1 experts at 32k–65k calibration
    /// tokens: 512–1024 rows for 5120 inputs). With the layer prior GPTQ
    /// still beats rounding on held-out rows, in both layouts; plain GPTQ
    /// is reported, not required to win.
    #[test]
    fn prior_carries_experts_with_fewer_rows_than_hidden() {
        let _env = env_guard("4", "1");
        let case = low_rows_case();
        for m in [48, 128, 192] {
            for codec in [Codec::Q4tp, Codec::Q2tp] {
                let (rtn, gptq, gptq_p) = low_rows_errors(&case, Engine::Host, m, codec, 1.0);
                eprintln!("{m} rows, {codec:?}: held-out relative output error RTN {rtn:.5}, GPTQ {gptq:.5}, GPTQ + prior {gptq_p:.5}");
                assert!(gptq_p < rtn, "{m} rows {codec:?}: GPTQ with the prior {gptq_p} not better than RTN {rtn}");
            }
        }
    }

    /// Route weights that follow the input (as the router's do): rows of
    /// one input cluster weigh 1, rows of a louder second cluster 0.05. On
    /// held-out rows, in the error the MoE output sees (each row weighted by
    /// w²), the route-weighted down Hessian beats the unweighted one; both
    /// beat rounding. The gate/up Hessians are the same in both encodes.
    #[test]
    fn route_weighted_down_hessian_tracks_weighted_error() {
        let _env = env_guard("4", "1");
        let mut rng = Rng(37);
        let (hidden, inter, n_half, n_ho_half) = (128, 64, 512, 128);
        let xa = inputs(&mut rng, n_half + n_ho_half, hidden);
        let xb: Vec<f32> = inputs(&mut rng, n_half + n_ho_half, hidden).iter().map(|v| v * 4.0).collect();
        // calibration: a then b; held out: a then b
        let mut x32 = Vec::new();
        x32.extend_from_slice(&xa[..n_half * hidden]);
        x32.extend_from_slice(&xb[..n_half * hidden]);
        x32.extend_from_slice(&xa[n_half * hidden..]);
        x32.extend_from_slice(&xb[n_half * hidden..]);
        let x: Vec<u16> = x32.iter().map(|&v| f32_to_f16(v)).collect();
        let n_cal = 2 * n_half;
        let n_ho = 2 * n_ho_half;
        let xh: Vec<f32> = x[n_cal * hidden..].iter().map(|&h| f16_to_f32(h)).collect();
        let wt = |t: usize, half: usize| if t < half { 1.0f32 } else { 0.05 };
        let rw: Vec<f32> = (0..n_cal).map(|t| wt(t, n_half)).collect();
        let w_ho: Vec<f32> = (0..n_ho).map(|t| wt(t, n_ho_half)).collect();
        let w = ExpertW { g: weights(&mut rng, inter, hidden, 0.15), u: weights(&mut rng, inter, hidden, 0.15), d: weights(&mut rng, hidden, inter, 0.1) };
        let limit = 1.0;
        let o_fp = expert_out(&xh, n_ho, hidden, &w.g, &w.u, &w.d, inter, limit, 2);
        let (_, base_w) = row_err(&o_fp, &vec![0f32; o_fp.len()], hidden, &w_ho);
        // (full expert error, error of the down projection alone: exact gate/up)
        let err_w = |enc: &[Vec<u8>; 3], codec: Codec| -> (f64, f64) {
            let [g, u, d] = dequant_expert(enc, inter, hidden, codec);
            let o = expert_out(&xh, n_ho, hidden, &g, &u, &d, inter, limit, 2);
            let od = expert_out(&xh, n_ho, hidden, &w.g, &w.u, &d, inter, limit, 2);
            (row_err(&o, &o_fp, hidden, &w_ho).1 / base_w, row_err(&od, &o_fp, hidden, &w_ho).1 / base_w)
        };
        let rows: Vec<u32> = (0..n_cal as u32).collect();
        for codec in [Codec::Q4tp, Codec::Q2tp] {
            let ec = EncodeCfg { codec, limit, lambda: 0.01, threads: 3 };
            let rtn = err_w(&rtn_expert(codec, &w, inter, hidden), codec);
            let plain = err_w(&encode_expert(Engine::Host, &mut None, hidden, inter, &x, &rows, None, &w, &ec, None, &mut [0.0; 5]).unwrap(), codec);
            let weighted = err_w(&encode_expert(Engine::Host, &mut None, hidden, inter, &x, &rows, Some(&rw), &w, &ec, None, &mut [0.0; 5]).unwrap(), codec);
            eprintln!(
                "{codec:?}: route-weighted held-out error, expert (down alone): RTN {:.5} ({:.5}), GPTQ unweighted {:.5} ({:.5}), GPTQ route-weighted {:.5} ({:.5})",
                rtn.0, rtn.1, plain.0, plain.1, weighted.0, weighted.1
            );
            assert!(plain.0 < rtn.0 && weighted.0 < rtn.0, "{codec:?}: GPTQ not better than RTN: {rtn:?} {plain:?} {weighted:?}");
            assert!(weighted.1 < 0.95 * plain.1, "{codec:?}: the route-weighted down Hessian does not cut the weighted down error: {plain:?} vs {weighted:?}");
        }
        // the per-row weights must match the rows
        let ec = EncodeCfg { codec: Codec::Q4tp, limit, lambda: 0.01, threads: 3 };
        assert!(encode_expert(Engine::Host, &mut None, hidden, inter, &x, &rows, Some(&rw[1..]), &w, &ec, None, &mut [0.0; 5]).is_err());
    }

    /// The q2tp debias product `[W_g; W_u]·H` on the device against the
    /// host's f32 GEMM, at the V4.1 expert geometry (2304 × 5120, a Hessian
    /// of 512 correlated rows plus damping): relative error of the product,
    /// how many debiased q2tp rows come out different, and the speed of
    /// both. Opt-in (`CMF_MOE_GPU_TEST=1`; on macOS also `CMF_GPU=wgpu`);
    /// needs a wgpu device but not the matrix units, so it runs on the M4.
    #[test]
    fn gpu_debias_product_matches_host() {
        if std::env::var("CMF_MOE_GPU_TEST").ok().as_deref() != Some("1") {
            eprintln!("CMF_MOE_GPU_TEST unset — skipped");
            return;
        }
        assert!(to::dev().is_some(), "CMF_MOE_GPU_TEST=1 but no wgpu device (on macOS set CMF_GPU=wgpu)");
        let (inter, hidden, n) = (2304usize, 5120usize, 512usize);
        let threads = std::thread::available_parallelism().map_or(4, |v| v.get());
        let mut rng = Rng(43);
        let wg = weights(&mut rng, inter, hidden, 0.02);
        let wu = weights(&mut rng, inter, hidden, 0.02);
        let rank = 64;
        let basis: Vec<f32> = (0..rank * hidden).map(|_| rng.normal()).collect();
        let z: Vec<f32> = (0..n * rank).map(|_| rng.normal()).collect();
        let mut x = gemm_nt(&z, n, rank, &transpose(&basis, rank, hidden), hidden, threads);
        x.iter_mut().for_each(|v| *v += 0.1 * rng.normal());
        let mut h = vec![0f64; hidden * hidden];
        gram_add_host(&mut h, &x, n, hidden, 1.0, threads);
        let mean_d = (0..hidden).map(|i| h[i * hidden + i]).sum::<f64>() / hidden as f64;
        for i in 0..hidden {
            h[i * hidden + i] += 0.01 * mean_d;
        }
        let h32: Vec<f32> = h.iter().map(|&v| v as f32).collect();
        let flop = 2.0 * 2.0 * inter as f64 * hidden as f64 * hidden as f64;
        let _ = debias_products_dev(&wg[..32 * hidden], &wu[..32 * hidden], 32, hidden, &h32); // warm the pipeline
        let t = Instant::now();
        let (dg, du) = debias_products_dev(&wg, &wu, inter, hidden, &h32);
        let t_dev = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let hg = crate::gptq::gemm_rows_square(&wg, inter, hidden, &h32, threads);
        let hu = crate::gptq::gemm_rows_square(&wu, inter, hidden, &h32, threads);
        let t_host = t.elapsed().as_secs_f64();
        eprintln!(
            "[W_g; W_u]·H ({inter}×{hidden}·{hidden}²): device {t_dev:.3}s = {:.0} GFLOP/s, host f32 GEMM on {threads} threads {t_host:.3}s = {:.0} GFLOP/s",
            flop / t_dev / 1e9,
            flop / t_host / 1e9
        );
        let rel = |a: &[f32], b: &[f32]| -> f64 {
            let num: f64 = a.iter().zip(b).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum();
            (num / b.iter().map(|&y| (y as f64).powi(2)).sum::<f64>()).sqrt()
        };
        let (eg, eu) = (rel(&dg, &hg), rel(&du, &hu));
        // both against f64 on sampled entries (on the M4 the device kernel's
        // in-order FMA chain can match Accelerate's AMX bit for bit)
        let mut worst = 0f64;
        for _ in 0..64 {
            let (r, c) = ((rng.next() % inter as u64) as usize, (rng.next() % hidden as u64) as usize);
            let want: f64 = (0..hidden).map(|k| wg[r * hidden + k] as f64 * h32[k * hidden + c] as f64).sum();
            let scale: f64 = (0..hidden).map(|k| (wg[r * hidden + k] as f64 * h32[k * hidden + c] as f64).abs()).sum();
            worst = worst.max((dg[r * hidden + c] as f64 - want).abs() / scale.max(1e-30));
        }
        eprintln!("device product against f64 on 64 sampled entries: worst error {worst:.2e} of Σ|terms|");
        assert!(worst < 1e-5, "device product off against f64: {worst:e}");
        assert!(dg.iter().any(|&v| v != 0.0));
        // the debias reads H·w only through two dot products per row (its
        // f16 row parameters): count the rows the device product changes
        let mut changed = 0usize;
        for (w, d, hh) in [(&wg, &dg, &hg), (&wu, &du, &hu)] {
            let base = quantize_2d(Quant::Q2TiledP, w, inter, hidden).1;
            let (mut a, mut b) = (base.clone(), base);
            q2tp_debias_rows(&mut a, w, inter, hidden, d, threads);
            q2tp_debias_rows(&mut b, w, inter, hidden, hh, threads);
            let (po, _, _) = cortiq_core::quant::q2tp_sections(inter, hidden);
            assert_eq!(a[..po], b[..po], "the debias touches only the row parameters");
            changed += (0..inter).filter(|&r| a[po + r * 4..po + r * 4 + 4] != b[po + r * 4..po + r * 4 + 4]).count();
        }
        eprintln!("device vs host product: relative error gate {eg:.2e}, up {eu:.2e}; debiased q2tp rows that differ: {changed} of {}", 2 * inter);
        assert!(eg < 1e-5 && eu < 1e-5, "device product off: {eg:e} / {eu:e}");
        assert!(changed * 100 <= 2 * inter, "{changed} of {} debiased rows differ", 2 * inter);
    }

    /// Row-major transpose of a `rows × cols` matrix.
    fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut t = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                t[c * rows + r] = a[r * cols + c];
            }
        }
        t
    }

    /// The device encode against the host reference (the lead runs it on a
    /// GPU stand: `CMF_MOE_GPU_TEST=1`; device folds and sweeps are f32, so
    /// the comparison is by held-out error, not bytes). The device encode
    /// needs the matrix units (cooperative matrices): without them, or
    /// without a device, the opted-in test fails instead of passing green.
    #[test]
    fn gpu_encode_tracks_the_host_reference() {
        if std::env::var("CMF_MOE_GPU_TEST").ok().as_deref() != Some("1") {
            eprintln!("CMF_MOE_GPU_TEST unset — skipped");
            return;
        }
        let c = to::dev().expect("CMF_MOE_GPU_TEST=1 but no wgpu device (on macOS set CMF_GPU=wgpu)");
        assert!(tm::matrix_units(), "CMF_MOE_GPU_TEST=1 but the wgpu device has no matrix units (cooperative matrices): the device encode cannot run here");
        let _env = env_guard("4", "1");
        let mut rng = Rng(29);
        let (hidden, inter, n_cal, n_ho) = (256, 128, 2048, 256);
        let x32 = inputs(&mut rng, n_cal + n_ho, hidden);
        let x: Vec<u16> = x32.iter().map(|&v| f32_to_f16(v)).collect();
        let xh: Vec<f32> = x[n_cal * hidden..].iter().map(|&h| f16_to_f32(h)).collect();
        let w = ExpertW { g: weights(&mut rng, inter, hidden, 0.15), u: weights(&mut rng, inter, hidden, 0.15), d: weights(&mut rng, hidden, inter, 0.1) };
        let rows: Vec<u32> = (0..n_cal as u32).collect();
        let sample: Vec<u32> = (0..n_cal as u32).step_by(4).collect();
        for limit in [0.0f32, 1.0] {
            let o_fp = expert_out(&xh, n_ho, hidden, &w.g, &w.u, &w.d, inter, limit, 2);
            let err = |enc: &[Vec<u8>; 3], codec: Codec| -> f64 {
                let [g, u, d] = dequant_expert(enc, inter, hidden, codec);
                let o = expert_out(&xh, n_ho, hidden, &g, &u, &d, inter, limit, 2);
                o.iter().zip(&o_fp).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>() / o_fp.iter().map(|&v| (v as f64).powi(2)).sum::<f64>()
            };
            for codec in [Codec::Q4tp, Codec::Q2tp] {
                let ec = EncodeCfg { codec, limit, lambda: 0.01, threads: 4 };
                let prior = Some(Prior { rows: &sample, n0: hidden as f64, n0_down: hidden as f64 / 4.0 });
                let host = err(&encode_expert(Engine::Host, &mut None, hidden, inter, &x, &rows, None, &w, &ec, prior, &mut [0.0; 5]).unwrap(), codec);
                let dev = err(&encode_expert(Engine::Gpu(c), &mut None, hidden, inter, &x, &rows, None, &w, &ec, prior, &mut [0.0; 5]).unwrap(), codec);
                let rtn = err(&rtn_expert(codec, &w, inter, hidden), codec);
                eprintln!("limit {limit} {codec:?}: RTN {rtn:.5} host {host:.5} device {dev:.5}");
                assert!(dev < rtn && (dev - host).abs() < 0.05 * host + 1e-6, "limit {limit} {codec:?}: device {dev} vs host {host} (RTN {rtn})");
                // route-weighted down Hessian (per-row weights on the device)
                let rw: Vec<f32> = (0..n_cal).map(|t| 0.05 + (t % 7) as f32 * 0.2).collect();
                let host = err(&encode_expert(Engine::Host, &mut None, hidden, inter, &x, &rows, Some(&rw), &w, &ec, prior, &mut [0.0; 5]).unwrap(), codec);
                let dev = err(&encode_expert(Engine::Gpu(c), &mut None, hidden, inter, &x, &rows, Some(&rw), &w, &ec, prior, &mut [0.0; 5]).unwrap(), codec);
                eprintln!("limit {limit} {codec:?} route-weighted: host {host:.5} device {dev:.5}");
                assert!(dev < rtn && (dev - host).abs() < 0.05 * host + 1e-6, "limit {limit} {codec:?} route-weighted: device {dev} vs host {host} (RTN {rtn})");
            }
        }
        // experts with fewer rows than hidden: the prior carries the device
        // encode as it carries the host's
        let case = low_rows_case();
        for m in [48, 128, 192] {
            for codec in [Codec::Q4tp, Codec::Q2tp] {
                let (rtn, _, host_p) = low_rows_errors(&case, Engine::Host, m, codec, 1.0);
                let (_, dev, dev_p) = low_rows_errors(&case, Engine::Gpu(c), m, codec, 1.0);
                eprintln!("{m} rows, {codec:?}: RTN {rtn:.5}, device GPTQ {dev:.5}, GPTQ + prior host {host_p:.5} device {dev_p:.5}");
                assert!(dev_p < rtn && (dev_p - host_p).abs() < 0.1 * host_p + 1e-6, "{m} rows {codec:?}: device {dev_p} vs host {host_p} (RTN {rtn})");
            }
        }
    }

    /// A tiny V4.1 checkpoint (FP4 routed experts, FP8 shared expert) goes
    /// through the converter twice — the q4tp calibration file and the q2tp
    /// reference — then through the quantize-moe layer encode on synthetic
    /// captured rows. The new file, and the calibration file rewritten in
    /// place in either layout, carry exactly the converter's names, dtypes,
    /// shapes and payload sizes, verify clean, and keep every non-expert
    /// tensor byte for byte.
    #[test]
    fn dsv41_layers_write_the_converter_layout() {
        let _env = env_guard("4", "1");
        let root = std::env::temp_dir().join(format!("cortiq-moe-e2e-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (conv, srcdir) = (root.join("hf"), root.join("src"));
        std::fs::create_dir_all(&conv).unwrap();
        std::fs::create_dir_all(&srcdir).unwrap();
        let (nl, ne, top_k, hidden, inter, vocab) = (2usize, 4usize, 2usize, 64usize, 32usize, 16usize);
        let mut rng = Rng(31);
        let mut wmap = serde_json::Map::new();
        let mut layers_w = Vec::new();
        for li in 0..nl {
            let (mut t, routed, shared) = dsv41_layer_source(&mut rng, li, ne, hidden, inter);
            t.push((format!("layers.{li}.ffn.gate.weight"), "F32", vec![ne, hidden], f32_bytes(&weights(&mut rng, ne, hidden, 0.1))));
            t.push((format!("layers.{li}.ffn.gate.bias"), "F32", vec![ne], f32_bytes(&vec![0.0; ne])));
            t.push((format!("layers.{li}.attn_norm.weight"), "F32", vec![hidden], f32_bytes(&vec![1.0; hidden])));
            let name = format!("model-{:05}-of-{:05}.safetensors", li + 1, nl + 1);
            for (n, ..) in &t {
                wmap.insert(n.clone(), serde_json::json!(name));
            }
            write_safetensors(&conv.join(&name), &t);
            write_safetensors(&srcdir.join(&name), &t);
            layers_w.push((routed, shared));
        }
        let last = format!("model-{:05}-of-{:05}.safetensors", nl + 1, nl + 1);
        let g = vec![
            ("embed.weight".to_string(), "F32", vec![vocab, hidden], f32_bytes(&weights(&mut rng, vocab, hidden, 0.1))),
            ("norm.weight".to_string(), "F32", vec![hidden], f32_bytes(&vec![1.0; hidden])),
        ];
        for (n, ..) in &g {
            wmap.insert(n.clone(), serde_json::json!(last));
        }
        write_safetensors(&conv.join(&last), &g);
        write_safetensors(&srcdir.join(&last), &g);
        let config = serde_json::json!({
            "model_type": "deepseek_v41",
            "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
            "text_config": {
                "model_type": "deepseek_v41_text", "hidden_size": hidden, "moe_intermediate_size": inter,
                "intermediate_size": inter, "num_hidden_layers": nl, "num_attention_heads": 1, "num_key_value_heads": 1,
                "head_dim": 32, "vocab_size": vocab, "rms_norm_eps": 1e-6, "n_routed_experts": ne, "n_shared_experts": 1,
                "num_experts_per_tok": top_k, "swiglu_limit": 10.0, "scoring_func": "sqrtsoftplus"
            }
        });
        for d in [&conv, &srcdir] {
            std::fs::write(d.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
            std::fs::write(d.join("model.safetensors.index.json"), serde_json::to_vec(&serde_json::json!({"weight_map": wmap})).unwrap()).unwrap();
        }
        let (calib, ref2) = (root.join("calib-q4tp.cmf"), root.join("ref-q2tp.cmf"));
        crate::convert::run_convert_multi(
            conv.to_str().unwrap(),
            &[("q4tp".into(), calib.to_str().unwrap().into()), ("q2tp".into(), ref2.to_str().unwrap().into())],
            None,
            None,
            None,
            false,
            |_| {},
        )
        .unwrap();
        let cm = CmfModel::open(&calib).unwrap();
        let r2 = CmfModel::open(&ref2).unwrap();
        let dirs: Vec<_> = (0..nl).map(|li| dsv41_dirs(&cm, li, ne).unwrap()).collect();
        for (d, s) in &dirs {
            for &(gi, ui, di) in d.iter().chain(std::iter::once(s)) {
                assert_eq!([cm.tensors[gi].dtype, cm.tensors[ui].dtype, cm.tensors[di].dtype], [TensorDtype::Q4TiledP; 3], "calibration experts are q4tp");
                assert_eq!((cm.tensors[gi].shape.clone(), cm.tensors[di].shape.clone()), (vec![inter, hidden], vec![hidden, inter]));
            }
        }
        let slots: HashSet<usize> = dirs.iter().flat_map(|(d, s)| d.iter().chain(std::iter::once(s)).flat_map(|&(g, u, dn)| [g, u, dn])).collect();
        // the slot checks run before any capture: a q4tp file takes either
        // codec in place, a q2tp file only q2tp; --patch-shared needs the codec
        // of the patched file
        let triples = |m: &CmfModel| -> Vec<(usize, usize, usize)> { (0..nl).flat_map(|li| { let (d, s) = dsv41_dirs(m, li, ne).unwrap(); d.into_iter().chain([s]) }).collect() };
        let shared_of = |m: &CmfModel| -> Vec<(usize, usize, usize)> { (0..nl).map(|li| dsv41_dirs(m, li, ne).unwrap().1).collect() };
        check_slots(&cm, &triples(&cm), Codec::Q4tp, None).unwrap();
        check_slots(&cm, &triples(&cm), Codec::Q2tp, None).unwrap();
        check_slots(&r2, &triples(&r2), Codec::Q2tp, None).unwrap();
        let e = check_slots(&r2, &triples(&r2), Codec::Q4tp, None).unwrap_err().to_string();
        assert!(e.contains("fits only a q4tp"), "{e}");
        check_slots(&cm, &shared_of(&cm), Codec::Q2tp, Some(("ref2", &r2))).unwrap();
        let e = check_slots(&cm, &shared_of(&cm), Codec::Q4tp, Some(("ref2", &r2))).unwrap_err().to_string();
        assert!(e.contains("pass --codec q2tp"), "{e}");
        let e = check_slots(&r2, &shared_of(&r2), Codec::Q2tp, Some(("calib", &cm))).unwrap_err().to_string();
        assert!(e.contains("pass --codec q4tp"), "{e}");

        // synthetic captured rows and routes per layer
        let n = 640;
        let held = vec![false; n];
        let rows: Vec<LayerRows> = (0..nl)
            .map(|_| {
                let x: Vec<u16> = inputs(&mut rng, n, hidden).iter().map(|&v| f32_to_f16(v)).collect();
                let mut picks = Vec::with_capacity(n * top_k);
                for _ in 0..n {
                    let mut ids: Vec<usize> = (0..ne).collect();
                    for k in 0..top_k {
                        let j = k + (rng.next() as usize) % (ne - k);
                        ids.swap(k, j);
                        picks.push(ids[k] as u16);
                    }
                }
                // route weights in (0.05, 1.45]
                let w: Vec<f32> = (0..n * top_k).map(|_| 0.75 + 0.7 * rng.uni()).collect();
                LayerRows { x, picks, w }
            })
            .collect();
        // the local source directory, read in place
        let source = Source::open(srcdir.to_str().unwrap(), root.join("scratch").to_str().unwrap()).unwrap();
        let run_layers = |codec: Codec, sink: &mut Sink, cm: &CmfModel| {
            let ec = EncodeCfg { codec, limit: 10.0, lambda: 0.01, threads: 2 };
            for li in 0..nl {
                let names = dsv41_src_names(&source, li, ne, true);
                assert_eq!(names.len(), (ne + 1) * 6, "every weight with its scale plane");
                let shards = source.shards_of(&names).unwrap();
                assert_eq!(shards, vec![format!("model-{:05}-of-{:05}.safetensors", li + 1, nl + 1)]);
                let src = source.open_layer(&shards).unwrap();
                let get = |part: Part, e: usize| dsv41_expert(&src, li, part, e, hidden, inter, 32);
                let job = LayerJob { li, hidden, ne, inter, top_k, min_rows: 8, prior_n0: hidden as f64, prior_s: 128, held: &held, route_w: RouteW::Off };
                let out = encode_layer(Engine::Host, &ec, &job, &rows[li], true, true, None, None, &get).unwrap();
                assert_eq!((out.routed.len(), out.n_gptq + out.n_rtn), (ne, ne));
                // the GPTQ'd payloads are the encode of this source, not the calibration's bytes
                let (w0, _) = &layers_w[li];
                let rtn0 = rtn_expert(codec, &w0[0], inter, hidden);
                assert_eq!(out.routed[0][2].len(), rtn0[2].len());
                sink.emit_layer(cm, codec, &dirs[li].0, Some(dirs[li].1), out).unwrap();
            }
        };
        // every tensor of `want` exists in `got`; the experts carry its dtype, shape and size
        let same_layout = |got: &CmfModel, want: &CmfModel, what: &str| {
            assert!(got.verify().is_empty(), "{what}: verify {:?}", got.verify());
            assert_eq!(got.tensors.len(), want.tensors.len(), "{what}: tensor count");
            for (i, t) in cm.tensors.iter().enumerate() {
                let g = got.tensor(&t.name).unwrap_or_else(|| panic!("{what}: {} missing", t.name));
                let w = want.tensor(&t.name).unwrap_or_else(|| panic!("{what}: {} not in the reference", t.name));
                if slots.contains(&i) {
                    assert_eq!((g.dtype, &g.shape, g.nbytes), (w.dtype, &w.shape, w.nbytes), "{what}: {}", t.name);
                }
            }
        };
        let non_experts_kept = |got: &CmfModel, what: &str| {
            for (i, t) in cm.tensors.iter().enumerate() {
                if !slots.contains(&i) {
                    assert_eq!(got.entry_bytes(got.tensor(&t.name).unwrap()), cm.entry_bytes(t), "{what}: {} changed", t.name);
                }
            }
        };

        // the held-out diagnostic (CMF_MOE_DIAG) on the same layer: every 4th
        // block of 80 tokens held out, RTN against GPTQ without and with the
        // prior, per row-count bin; GPTQ is never worse than rounding
        // with CMF_MOE_ROUTE_W=diag each prior runs with the unweighted and
        // the route-weighted down Hessian, and the error is also reported
        // weighted by w² (what the MoE output sees)
        let held_d: Vec<bool> = (0..n).map(|t| (t / 80) % 4 == 3).collect();
        for (codec, route_w) in [(Codec::Q4tp, RouteW::Off), (Codec::Q2tp, RouteW::Off), (Codec::Q4tp, RouteW::Diag), (Codec::Q2tp, RouteW::Diag)] {
            let ec = EncodeCfg { codec, limit: 10.0, lambda: 0.01, threads: 2 };
            let names = dsv41_src_names(&source, 0, ne, true);
            let src = source.open_layer(&source.shards_of(&names).unwrap()).unwrap();
            let get = |part: Part, e: usize| dsv41_expert(&src, 0, part, e, hidden, inter, 32);
            let job = LayerJob { li: 0, hidden, ne, inter, top_k, min_rows: 8, prior_n0: 0.0, prior_s: 128, held: &held_d, route_w };
            let priors = [0.0, hidden as f64];
            let out = encode_layer(Engine::Host, &ec, &job, &rows[0], true, false, None, Some(&priors), &get).unwrap();
            assert!(out.routed.is_empty() && out.shared.is_none(), "diagnostics write nothing");
            let measured: usize = out.diag.iter().map(|b| b.count).sum();
            assert_eq!(measured, ne, "every expert measured");
            let vars = diag_variants(&priors, route_w);
            assert_eq!(vars.len(), if route_w == RouteW::Diag { 4 } else { 2 });
            for b in out.diag.iter().filter(|b| b.count > 0) {
                eprintln!("{codec:?} {route_w:?} diag: {} experts, RTN {} GPTQ {vars:?} {} | route-weighted RTN {} GPTQ {}", b.count, b.rel(0), b.rels(vars.len()), b.rel_w(0), b.rels_w(vars.len()));
                assert!(b.base_w > 0.0, "the weighted metric is measured");
                for (k, &(_, weighted)) in vars.iter().enumerate() {
                    let (e, rtn) = if weighted { (b.err_w[k + 1], b.err_w[0]) } else { (b.err[k + 1], b.err[0]) };
                    assert!(e <= rtn, "{codec:?} variant {:?}: GPTQ worse than rounding on held-out rows: {e} > {rtn}", vars[k]);
                }
            }
        }

        // a new q2tp file: the converter's q2tp layout
        let out2 = root.join("moe-q2tp.cmf");
        let mut sink = Sink::Stream(CmfStreamWriter::new(&out2, CmfStreamWriter::head_reserve_for(cm.tensors.len(), 64)).unwrap());
        run_layers(Codec::Q2tp, &mut sink, &cm);
        sink.finish(&cm, &slots, Codec::Q2tp, serde_json::json!({"test": true}), out2.to_str().unwrap(), Instant::now()).unwrap();
        let m2 = CmfModel::open(&out2).unwrap();
        same_layout(&m2, &r2, "q2tp output");
        // the free-space estimate of an --output file covers the file written
        let gu = Codec::Q2tp.gate_up().0;
        let slot_dt: HashMap<usize, TensorDtype> =
            dirs.iter().flat_map(|(r, s)| r.iter().chain(std::iter::once(s))).flat_map(|&(g, u, d)| [(g, gu), (u, gu), (d, TensorDtype::Q4TiledP)]).collect();
        let (est, len) = (stream_output_bytes(&cm, &slot_dt).unwrap(), std::fs::metadata(&out2).unwrap().len());
        assert!(est >= len && est <= len + cm.tensors.len() as u64 * 4096 + (4 << 20), "estimate {est} B for a {len} B file");
        non_experts_kept(&m2, "q2tp output");
        let prov = m2.header.provenance.as_ref().unwrap();
        assert_eq!((prov["weight_quant"].as_str(), prov["gptq_moe"]["test"].as_bool()), (Some("q2tp"), Some(true)));
        let (gi, _, _) = dirs[0].0[0];
        assert_ne!(m2.entry_bytes(m2.tensor(&cm.tensors[gi].name).unwrap()), r2.entry_bytes(r2.tensor(&cm.tensors[gi].name).unwrap()), "GPTQ changes the codes");

        // in place, q4tp: same slots, same layout as the calibration file
        let inplace4 = root.join("inplace-q4tp.cmf");
        std::fs::copy(&calib, &inplace4).unwrap();
        let c4 = CmfModel::open(&inplace4).unwrap();
        let mut sink = Sink::InPlace { path: inplace4.to_str().unwrap().into(), pending: Vec::new(), before: 0, after: 0, punched: 0 };
        run_layers(Codec::Q4tp, &mut sink, &c4);
        let Sink::InPlace { before, after, .. } = &sink else { unreachable!() };
        assert_eq!(before, after, "q4tp payloads fill their q4tp slots exactly");
        sink.finish(&c4, &slots, Codec::Q4tp, serde_json::json!({}), "", Instant::now()).unwrap();
        drop(c4);
        let m4 = CmfModel::open(&inplace4).unwrap();
        same_layout(&m4, &cm, "q4tp in place");
        non_experts_kept(&m4, "q4tp in place");
        assert_ne!(m4.entry_bytes(&m4.tensors[gi]), cm.entry_bytes(&cm.tensors[gi]), "the expert was rewritten");

        // in place, q2tp: smaller payloads in the q4tp slots, the q2tp layout
        let inplace2 = root.join("inplace-q2tp.cmf");
        std::fs::copy(&calib, &inplace2).unwrap();
        let c2 = CmfModel::open(&inplace2).unwrap();
        let mut sink = Sink::InPlace { path: inplace2.to_str().unwrap().into(), pending: Vec::new(), before: 0, after: 0, punched: 0 };
        run_layers(Codec::Q2tp, &mut sink, &c2);
        sink.finish(&c2, &slots, Codec::Q2tp, serde_json::json!({}), "", Instant::now()).unwrap();
        drop(c2);
        let m2i = CmfModel::open(&inplace2).unwrap();
        same_layout(&m2i, &r2, "q2tp in place");
        non_experts_kept(&m2i, "q2tp in place");
        for sh in source.shards_of(&dsv41_src_names(&source, 0, ne, true)).unwrap() {
            source.release(&sh);
            assert!(srcdir.join(&sh).exists(), "a local source is never deleted");
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
