//! `cortiq compress SOURCE DATA OUTPUT [QUANT]` — task-specific compression
//! by teacher data (docs/TZ_TEACHER_DATA_COMPRESS.ru.md §9).
//!
//! Stages, each in its own process (a clean device context and the
//! environment the stage needs), each cached in the work directory under a
//! key of its inputs, so a re-run resumes where the last one stopped:
//!
//!   S0 prepare     source → f16 working copy + q8_2f calibration sibling
//!   S1 illuminate  Grams of every FFN read-out input on the teacher data
//!   S2 plan        width of every layer from the refit-error curves
//!   S3 fold        keep-sets + closed-form refit of `down_proj`
//!   S4 heal        KD polish against the uncompressed source on the
//!                  teacher's answers
//!   S5 quant       GPTQ (q4tp/q2tp) or the converter's codec, calibrated
//!                  on the teacher data
//!   S6 eval        held-out (test) answers: NLL, generation, gates, report

pub mod cmfio;
pub mod data;
pub mod dtgma;
pub mod eval;
#[cfg(feature = "gpu")]
pub mod heal_gpu;
pub mod illum;
pub mod moe;
pub mod plan;
pub mod stages;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};

/// `cortiq compress`.
#[derive(clap::Args, Debug, Clone)]
pub struct CompressArgs {
    /// Source model: a .cmf (f16/bf16 or q8), a local Hugging Face directory
    /// or a hub repo id
    pub source: String,
    /// The task: a .jsonl of prompts (the teacher — the source, or
    /// `--teacher` — answers them first; those answers light up the parts
    /// of the model the task uses) or a finalized .tdata (`cortiq
    /// teacher-data generate`)
    pub data: String,
    /// Output .cmf
    pub output: String,
    /// Codec of the output (default: the source's own — quantize afterwards
    /// with `cortiq quantize-gptq` if you want a smaller file)
    pub quant: Option<String>,

    /// Prompts: who answers them — the source itself (default), another
    /// local .cmf with the same tokenizer, or a model behind any
    /// OpenAI-compatible API: `openrouter:qwen/qwen3.8-27b` (key in
    /// OPENROUTER_API_KEY), `openai:MODEL` (OPENAI_API_KEY), or
    /// `api:MODEL --api-base http://host:8000/v1` (vLLM, SGLang, llama.cpp,
    /// Ollama, …; key in CORTIQ_API_KEY if the server wants one) — no server
    /// to rent for a large teacher
    #[arg(long)]
    pub teacher: Option<String>,
    #[command(flatten)]
    pub api: crate::teacher_api::ApiArgs,
    /// Prompts: tokens each answer may take, reasoning included (default
    /// 2048 when the source answers; 8192 for an API teacher — its tokens
    /// are cheap and a large reasoning model thinks long)
    #[arg(long)]
    pub gen_max_tokens: Option<usize>,
    /// Prompts: answers per prompt
    #[arg(long, default_value_t = 1)]
    pub gen_samples: u32,
    /// Prompts: decoding — auto (the model's recommended sampling when its
    /// file carries one, else greedy) | greedy | file | temperature=…,top_p=…
    #[arg(long, default_value = "auto")]
    pub gen_sampling: String,
    /// Prompts: answer directly, without the model's reasoning
    #[arg(long)]
    pub no_think: bool,

    /// Target FFN width: a fraction of all FFN neurons (0.75) or a mean
    /// width in neurons (12288)
    #[arg(long)]
    pub ffn: Option<String>,
    /// Target parameter count without the MTP draft (12B, 900M)
    #[arg(long)]
    pub size: Option<String>,
    /// Target size as a fraction of the source (0.7 = 30 % smaller): the
    /// parameters kept, at the source's codec
    #[arg(long)]
    pub ratio: Option<f64>,
    /// fast | balanced | full | quant-only
    #[arg(long, default_value = "balanced")]
    pub recipe: String,

    /// Which neurons stay: `importance` (illumination: energy × read-out
    /// weight, one pass) or `dtgma` (Patent 2: a mask trained with the
    /// task's LM loss under L1 pressure decides per layer how many and which)
    #[arg(long, default_value = "importance")]
    pub select: String,
    /// Width per layer: `uniform` (measured best so far), `curve` (by each
    /// layer's relative refit error) or `abs` (that error weighted by the
    /// layer's output energy)
    #[arg(long, default_value = "uniform")]
    pub allocate: String,
    /// First layers kept at full width (measured at 75% FFN: protecting
    /// 1 + 2 edge layers narrows the rest further and costs more than it
    /// saves — Qwen3.5-2B fold 0.330 vs 0.320 unprotected, 0.8B healed
    /// 0.320 vs 0.313)
    #[arg(long, default_value_t = 0)]
    pub protect_first: usize,
    /// Last layers kept at full width
    #[arg(long, default_value_t = 0)]
    pub protect_last: usize,
    /// Lowest width a layer may get, as a fraction of its own
    #[arg(long, default_value_t = 0.25)]
    pub min_width: f64,
    /// FFN widths are multiples of this
    #[arg(long, default_value_t = 128)]
    pub align: usize,
    /// Refit of the read-out: auto (chosen per layer on the val split), on, off
    #[arg(long, default_value = "auto")]
    pub refit: String,
    /// Relative ridges tried by the refit (comma-separated)
    #[arg(long, default_value = "0.003,0.01,0.03,0.1")]
    pub ridge: String,
    /// Illumination window (tokens); longer records are cut with half overlap
    #[arg(long, default_value_t = 2048)]
    pub window: usize,
    /// Illumination tokens of the train split (val gets a quarter). Default:
    /// 20 per FFN neuron of the widest layer — the Gram of a refit needs
    /// ≳10 tokens per kept neuron (fewer measurably overfit), more only
    /// costs forward passes. 0 = every record
    #[arg(long)]
    pub illum_tokens: Option<usize>,

    /// Healing: none | fcd (default from the recipe)
    #[arg(long)]
    pub heal: Option<String>,
    /// Layers trained by the heal: all | deepN | auto
    #[arg(long, default_value = "auto")]
    pub heal_layers: String,
    /// Passes over the train records (default 2, `full` 3; on the device
    /// 6 and 12 — its steps are cheap: 0.8B deep8 2 → 6 epochs 0.339 → 0.328)
    #[arg(long)]
    pub heal_epochs: Option<f64>,
    /// Heal learning rate. Default: 1e-5 on the CPU (0.8B at 75% width:
    /// 5e-5 raised the val loss); on the device 1e-5 for a tail of up to
    /// 30% of the layers, falling to 3e-6 when every layer trains (0.8B:
    /// all layers at 1e-5 0.331, 5e-6 0.310, 3e-6 0.307)
    #[arg(long)]
    pub heal_lr: Option<f64>,
    /// KL(teacher‖student) weight of the heal loss (CE gets 1 − kd)
    #[arg(long, default_value_t = 0.7)]
    pub kd: f64,
    /// Heal window length
    #[arg(long, default_value_t = 512)]
    pub seq: usize,
    /// Heal windows per step
    #[arg(long, default_value_t = 2)]
    pub bs: usize,
    #[arg(long, default_value_t = 25)]
    pub eval_every: usize,
    /// Where the heal trains: auto (the device when it has the trainer's
    /// matrix-unit GEMMs — NVIDIA Vulkan — else the CPU) | gpu | cpu
    #[arg(long, default_value = "auto")]
    pub heal_device: String,
    /// Device heal: window length (each window a fresh context)
    #[arg(long, default_value_t = 2048)]
    pub heal_window: usize,
    /// Device heal: teacher distribution kept per token
    #[arg(long, default_value_t = 64)]
    pub heal_topk: usize,
    /// Device heal loss: mix=A (A·CE on the teacher data's text + (1−A)·KL
    /// to the source's top-k), kd (KL only) or ce (plain cross-entropy,
    /// distillation-free, Patent 12 §4.7). Qwen3.5-0.8B at 75 % FFN, test
    /// NLL after the heal: kd 0.303, ce 0.292, mix=0.5 0.281
    #[arg(long, default_value = "mix=0.5")]
    pub heal_loss: String,
    /// Device heal schedule: flat (every healed layer from the first step,
    /// warm-up + cosine) | genesis (the notebook's progressive unfreeze with
    /// per-layer rate decay; 0.8B, mix=0.5: 0.285 at 3e-6 vs flat 0.281)
    #[arg(long, default_value = "flat")]
    pub heal_schedule: String,
    /// Device heal: weights trained in the healed layers: ffn (the MLP the
    /// fold cut; 0.8B, all layers at 5e-6: 0.307 vs 0.310 training every
    /// linear, at a third of the device memory) | all
    #[arg(long, default_value = "ffn")]
    pub heal_params: String,
    /// Loss tokens: answer | generated (think + answer) | all
    #[arg(long, default_value = "generated")]
    pub loss_on: String,

    /// GPTQ calibration tokens (default: 65536, or all train tokens if fewer)
    #[arg(long)]
    pub calib_tokens: Option<usize>,
    #[arg(long, default_value_t = 2048)]
    pub calib_window: usize,
    /// Tensors kept at q8_2f in a low-bit output: kv (k/v of full attention),
    /// kvo, none, or comma-separated name globs
    #[arg(long, default_value = "kv")]
    pub escort: String,
    /// Per-tensor codec override, PATTERN=QUANT (as `convert --tensor-quant`)
    #[arg(long = "tensor-quant", value_name = "PATTERN=QUANT")]
    pub tensor_quant: Vec<String>,

    /// Test prompts generated from in the evaluation
    #[arg(long, default_value_t = 16)]
    pub eval_prompts: usize,
    /// Generation cap per evaluation prompt
    #[arg(long, default_value_t = 384)]
    pub eval_gen_tokens: usize,
    /// Plain text for a general-ppl check (forgetting outside the task)
    #[arg(long)]
    pub general_eval: Option<String>,
    /// Also score this model on the same test records — another compression,
    /// a quant-only file, the same task with another teacher (repeatable;
    /// extra report rows, the gates stay on the output)
    #[arg(long, value_name = "MODEL.cmf")]
    pub compare: Vec<String>,

    /// Work directory (default: <OUTPUT>.work)
    #[arg(long)]
    pub work_dir: Option<String>,
    /// Show the plan (architecture, widths, parameters) and stop
    #[arg(long)]
    pub plan_only: bool,
    /// Stop after this stage (S0…S6 or its name)
    #[arg(long)]
    pub until: Option<String>,
    /// Recompute from this stage on even if cached
    #[arg(long)]
    pub from: Option<String>,
    /// Discard the work directory and start over
    #[arg(long)]
    pub fresh: bool,
    #[arg(long)]
    pub threads: Option<usize>,
    /// Accept a 4-bit or lower source (folding on quantized weights adds
    /// their error to the fold's)
    #[arg(long)]
    pub allow_lossy_source: bool,
    /// Write the output even if a quality gate fails (as <OUTPUT>.rejected.cmf)
    #[arg(long)]
    pub write_on_fail: bool,
    /// Internal: run one stage in this process
    #[arg(long, hide = true)]
    pub stage: Option<String>,
}

impl CompressArgs {
    /// `--gen-max-tokens`, or its default for the teacher.
    pub fn gen_max_tokens(&self) -> usize {
        let api = self.teacher.as_deref().is_some_and(crate::teacher_api::is_api_spec);
        self.gen_max_tokens.unwrap_or(if api { 8192 } else { 2048 })
    }
}

pub const STAGES: [&str; 7] = ["S0.prepare", "S1.illuminate", "S2.plan", "S3.fold", "S4.heal", "S5.quant", "S6.eval"];

/// Index of a stage from `S3`, `3`, `fold` or `S3.fold`.
pub fn stage_index(s: &str) -> Result<usize> {
    let t = s.trim().to_ascii_lowercase();
    for (i, name) in STAGES.iter().enumerate() {
        let (code, word) = name.split_once('.').unwrap();
        if t == code.to_ascii_lowercase() || t == word || t == name.to_ascii_lowercase() || t == i.to_string() {
            return Ok(i);
        }
    }
    bail!("unknown stage {s:?}; expected one of {}", STAGES.join(", "))
}

/// The resolved run: arguments, recipe defaults, paths.
pub struct Run {
    pub a: CompressArgs,
    /// The prompts file when DATA is one (its teacher data is generated by
    /// S0 into `a.data`).
    pub prompts: Option<String>,
    pub work: PathBuf,
    pub quant: crate::convert::Quant,
    pub structural: bool,
    pub heal: bool,
    pub heal_epochs: f64,
    pub calib_tokens: Option<usize>,
    pub threads: usize,
}

impl Run {
    pub fn new(mut a: CompressArgs) -> Result<Self> {
        let quant = match a.quant.as_deref() {
            Some(q) => crate::convert::parse_quant(q)?,
            None => source_codec(&a.source)?,
        };
        let recipe = a.recipe.as_str();
        ensure!(
            matches!(recipe, "fast" | "balanced" | "full" | "quant-only"),
            "--recipe {recipe:?}: expected fast, balanced, full or quant-only"
        );
        let has_target = a.ffn.is_some() || a.size.is_some() || a.ratio.is_some();
        let structural = recipe != "quant-only";
        ensure!(
            !structural || has_target,
            "give a target (--ffn, --size or --ratio), or --recipe quant-only for calibration-only quantization"
        );
        ensure!(
            [a.ffn.is_some(), a.size.is_some(), a.ratio.is_some()].iter().filter(|&&x| x).count() <= 1,
            "give only one of --ffn, --size, --ratio"
        );
        let heal = match a.heal.as_deref() {
            Some("none") => false,
            Some("fcd") => true,
            Some(other) => bail!("--heal {other:?}: expected none or fcd"),
            None => structural && recipe != "fast",
        };
        let heal_epochs = a.heal_epochs.unwrap_or(if recipe == "full" { 3.0 } else { 2.0 });
        let threads = a.threads.unwrap_or_else(stages::cpu_budget);
        let work = PathBuf::from(a.work_dir.clone().unwrap_or_else(|| format!("{}.work", a.output)));
        let calib_tokens = a.calib_tokens.or(if recipe == "fast" { Some(32768) } else { None });
        // prompts in: S0 answers them with the source; every later stage
        // reads that teacher data
        let prompts = (!a.data.contains(".tdata")).then(|| a.data.clone());
        if prompts.is_some() {
            a.data = work.join(STAGES[0]).join("task.tdata").display().to_string();
        }
        Ok(Self { a, prompts, work, quant, structural, heal, heal_epochs, calib_tokens, threads })
    }

    pub fn stage_dir(&self, i: usize) -> PathBuf {
        self.work.join(STAGES[i])
    }

    pub fn done_path(&self, i: usize) -> PathBuf {
        self.stage_dir(i).join("done.json")
    }

    /// The finished stage's record, if its key still matches.
    pub fn done(&self, i: usize) -> Option<Value> {
        let v: Value = serde_json::from_slice(&std::fs::read(self.done_path(i)).ok()?).ok()?;
        let key = self.key(i).ok()?;
        (v.get("key").and_then(Value::as_str) == Some(key.as_str())).then_some(v)
    }

    /// Record a finished stage (its key plus what it produced).
    pub fn mark_done(&self, i: usize, mut info: Value) -> Result<()> {
        info["key"] = json!(self.key(i)?);
        info["stage"] = json!(STAGES[i]);
        info["finished"] = json!(now());
        let p = self.done_path(i);
        std::fs::create_dir_all(p.parent().unwrap())?;
        std::fs::write(&p, serde_json::to_vec_pretty(&info)?)?;
        Ok(())
    }

    pub fn require(&self, i: usize) -> Result<Value> {
        self.done(i).with_context(|| format!("{} has not been run (or its inputs changed)", STAGES[i]))
    }

    /// Identity of a stage: its own parameters and its inputs' keys.
    pub fn key(&self, i: usize) -> Result<String> {
        let a = &self.a;
        let own = match i {
            0 => json!({
                "source": source_identity(&a.source), "lossy": a.allow_lossy_source,
                "prompts": self.prompts.as_deref().map(|p| prompts_identity(p, a)),
            }),
            1 => json!({ "data": data_identity(&a.data), "window": a.window, "illum_tokens": a.illum_tokens, "curves": 2 }),
            2 => json!({
                "ffn": a.ffn, "size": a.size, "ratio": a.ratio, "allocate": a.allocate,
                "protect": [a.protect_first, a.protect_last], "min_width": a.min_width, "align": a.align,
                "select": a.select,
            }),
            3 => json!({ "refit": a.refit, "ridge": a.ridge }),
            4 => json!({
                "gate": "relative-v2",
                "heal": self.heal, "layers": a.heal_layers, "epochs": self.heal_epochs, "lr": a.heal_lr,
                "kd": a.kd, "seq": a.seq, "bs": a.bs, "eval_every": a.eval_every, "loss_on": a.loss_on,
                "data": data_identity(&a.data), "device": a.heal_device, "heal_window": a.heal_window,
                "heal_topk": a.heal_topk, "heal_params": a.heal_params, "heal_loss": a.heal_loss, "heal_schedule": a.heal_schedule,
            }),
            5 => json!({
                "o1": "off",
                "quant": crate::convert::quant_name(self.quant), "calib_tokens": self.calib_tokens,
                "calib_window": a.calib_window, "escort": a.escort, "tensor_quant": a.tensor_quant,
                "data": data_identity(&a.data),
            }),
            6 => json!({
                "o1": "off",
                "eval_prompts": a.eval_prompts, "eval_gen_tokens": a.eval_gen_tokens,
                "general_eval": a.general_eval, "data": data_identity(&a.data),
                "compare": a.compare,
            }),
            _ => bail!("stage index {i}"),
        };
        // upstream: the previous stage that this run actually executes
        let up = match i {
            0 => None,
            5 if !self.structural => Some(0),
            _ => Some(i - 1),
        };
        let up_key = match up {
            Some(u) => self.key(u)?,
            None => String::new(),
        };
        let all = json!({
            "stage": STAGES[i], "own": own, "up": up_key,
            "cortiq": env!("CARGO_PKG_VERSION"), "structural": self.structural,
        });
        Ok(crate::knowledge::sha256_hex(serde_json::to_string(&all)?.as_bytes())[..16].to_string())
    }

    /// Append a line to the run journal and print it.
    pub fn log(&self, line: &str) {
        println!("{line}");
        let _ = std::fs::create_dir_all(&self.work);
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(self.work.join("JOURNAL.md")) {
            let _ = writeln!(f, "{line}");
        }
    }

    /// The stages this run executes, in order.
    pub fn stages(&self) -> Vec<usize> {
        if self.structural { (0..7).collect() } else { vec![0, 5, 6] }
    }
}

pub fn now() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let (d, s) = (t / 86400, t % 86400);
    // civil date from days since 1970-01-01 (Howard Hinnant's algorithm)
    let z = d as i64 + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{day:02}T{:02}:{:02}:{:02}Z", s / 3600, s % 3600 / 60, s % 60)
}

/// Cheap identity of a source: path + size + mtime for files, config bytes
/// for an HF directory, the id itself for a hub repo.
fn source_identity(src: &str) -> String {
    let p = Path::new(src);
    if p.is_file() {
        let m = std::fs::metadata(p).ok();
        let mt = m.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
        return format!("{}:{}:{}", p.canonicalize().unwrap_or(p.into()).display(), m.map_or(0, |m| m.len()), mt.map_or(0, |d| d.as_secs()));
    }
    if p.is_dir() {
        let cfg = std::fs::read(p.join("config.json")).unwrap_or_default();
        return format!("{}:{}", p.canonicalize().unwrap_or(p.into()).display(), crate::knowledge::sha256_hex(&cfg));
    }
    src.to_string()
}

/// The output codec when none is given: the source's own (float sources
/// stay f16, 8-bit ones q8_2f).
fn source_codec(source: &str) -> Result<crate::convert::Quant> {
    use crate::convert::Quant;
    use cortiq_core::TensorDtype;
    if !source.ends_with(".cmf") {
        return Ok(Quant::F16);
    }
    let m = cortiq_core::CmfModel::open_sharded(source).with_context(|| format!("opening {source}"))?;
    let mut by: Vec<(TensorDtype, u64)> = Vec::new();
    for e in m.tensors.iter().filter(|e| e.shape.len() == 2) {
        match by.iter_mut().find(|(d, _)| *d == e.dtype) {
            Some((_, n)) => *n += e.n_elems() as u64,
            None => by.push((e.dtype, e.n_elems() as u64)),
        }
    }
    Ok(match by.into_iter().max_by_key(|&(_, n)| n).map(|(d, _)| d) {
        Some(TensorDtype::Q4TiledP) => Quant::Q4TiledP,
        Some(TensorDtype::Q2TiledP) => Quant::Q2TiledP,
        Some(TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32) | None => Quant::F16,
        _ => Quant::Q8_2f,
    })
}

/// Identity of a prompts file and of how its answers are generated.
fn prompts_identity(path: &str, a: &CompressArgs) -> String {
    let p = std::fs::read(path).unwrap_or_default();
    format!(
        "{}:{}:{}:{}:{}:{}",
        crate::knowledge::sha256_hex(&p),
        a.gen_max_tokens(),
        a.gen_samples,
        a.gen_sampling,
        a.no_think,
        a.teacher.as_deref().unwrap_or("source")
    ) + &match (&a.api.api_base, &a.api.api_extra) {
        (None, None) => String::new(),
        (b, e) => format!(":{}:{}", b.as_deref().unwrap_or(""), e.as_deref().unwrap_or("")),
    }
}

fn data_identity(spec: &str) -> String {
    let root = spec.rfind(".tdata").map(|i| &spec[..i + 6]).unwrap_or(spec);
    let m = std::fs::read(Path::new(root).join("manifest.json")).unwrap_or_default();
    format!("{spec}:{}", crate::knowledge::sha256_hex(&m))
}

/// One line on what a stage does, for the progress header.
fn stage_purpose(i: usize) -> &'static str {
    [
        "load the source; let it answer the task prompts",
        "light up the FFN neurons on the task (activation statistics)",
        "choose each layer's width",
        "keep the task's neurons, refit the rest into them",
        "heal: train the narrowed model back toward the source",
        "write the output codec",
        "score on held-out task answers; quality gates",
    ][i]
}

/// Entry point.
pub fn run(a: CompressArgs) -> Result<()> {
    if let Some(stage) = a.stage.clone() {
        let run = Run::new(a)?;
        stages::run_stage(&run, &stage)?;
        // The result is on disk; device teardown at exit took minutes on a
        // single thread (wgpu, 5090), so the child leaves without it.
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        unsafe { libc::_exit(0) }
    }
    orchestrate(Run::new(a)?)
}

fn orchestrate(run: Run) -> Result<()> {
    if run.a.fresh && run.work.exists() {
        std::fs::remove_dir_all(&run.work).with_context(|| format!("removing {}", run.work.display()))?;
    }
    std::fs::create_dir_all(&run.work)?;
    let a = run.a.clone();
    run.log(&format!(
        "\n## compress {} — {}\n\n`{}`\n",
        now(),
        a.output,
        std::env::args().collect::<Vec<_>>().join(" ")
    ));
    let order = run.stages();
    let until = match (&a.until, a.plan_only) {
        (_, true) => 0,
        (Some(u), _) => stage_index(u)?,
        (None, _) => 6,
    };
    let from = a.from.as_deref().map(stage_index).transpose()?;
    let job_t0 = std::time::Instant::now();
    let n_stages = order.iter().filter(|&&i| i <= until).count();
    for (k, &i) in order.iter().enumerate() {
        if i > until {
            break;
        }
        let forced = from.is_some_and(|f| i >= f);
        if !forced && let Some(v) = run.done(i) {
            run.log(&format!("[{}] cached ({})", STAGES[i], v.get("finished").and_then(Value::as_str).unwrap_or("")));
            continue;
        }
        let _ = std::fs::remove_file(run.done_path(i));
        let label = STAGES[i];
        run.log(&format!("[{label}] start {} — stage {}/{n_stages}: {}", now(), k + 1, stage_purpose(i)));
        let t0 = std::time::Instant::now();
        spawn_stage(&run, label)?;
        let v = run.done(i).with_context(|| format!("{label} finished without recording its result"))?;
        run.log(&format!(
            "[{label}] done in {:.0}s · {} since the start",
            t0.elapsed().as_secs_f64(),
            crate::fmt_eta(job_t0.elapsed().as_secs_f64())
        ));
        if i == 2 && v.get("plan").is_some() {
            stages::print_plan(&run, &v)?;
        }
    }
    if a.plan_only {
        // arithmetic only: the widths a uniform cut would give, no data pass
        return stages::print_arithmetic_plan(&run);
    }
    if until < 6 {
        run.log(&format!("stopped after {} (--until)", STAGES[until]));
        return Ok(());
    }
    let s6 = run.require(6)?;
    let pass = s6.get("pass").and_then(Value::as_bool).unwrap_or(false);
    eval::print_report(&run, &s6)?;
    let target = if pass {
        PathBuf::from(&a.output)
    } else if a.write_on_fail {
        PathBuf::from(format!("{}.rejected.cmf", a.output.trim_end_matches(".cmf")))
    } else {
        run.log("quality gates FAILED — no output written (see the report; --write-on-fail writes it anyway)");
        std::process::exit(5);
    };
    stages::publish(&run, &target)?;
    run.log(&format!("wrote {}", target.display()));
    if !pass {
        std::process::exit(5);
    }
    Ok(())
}

/// Run one stage as a child process of this binary with the environment it
/// needs. The Hessian hook of S1/S3/S5 only sees host-side inputs: macOS
/// calibrates on the CPU (Metal's chunk graph keeps whole layer spans on
/// the device), wgpu builds keep the device but not its resident graph.
fn spawn_stage(run: &Run, label: &str) -> Result<()> {
    // A rebuild during a long run replaces the binary: Linux then reports
    // the running image as "<path> (deleted)". The stage runs the binary
    // that is at that path now (the cache keys carry the version).
    let mut exe = std::env::current_exe()?;
    if !exe.exists()
        && let Some(s) = exe.to_str()
        && let Some(p) = s.strip_suffix(" (deleted)")
    {
        exe = PathBuf::from(p);
    }
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    argv.retain(|x| x != "--fresh" && x != "--plan-only");
    argv.push("--stage".into());
    argv.push(label.into());
    let mut cmd = std::process::Command::new(exe);
    cmd.args(&argv);
    let calib = matches!(label, "S1.illuminate" | "S2.plan" | "S3.fold" | "S5.quant");
    if calib {
        if cfg!(target_os = "macos") {
            cmd.env("CMF_GPU", "0");
        } else {
            cmd.env("CMF_GPU_WGPU_GRAPH", "0");
        }
    }
    cmd.env("CMF_GPU_PROBE", "0");
    // Exact attention everywhere: a provenance o1 hint (older FCD outputs
    // carried one) must not switch a stage to Nyström attention.
    cmd.env("CMF_O1", "off");
    // the engine's pool takes only the big cores of a heterogeneous CPU and
    // all host cores in a container; a stage uses the whole CPU budget
    if std::env::var_os("CMF_THREADS").is_none() {
        cmd.env("CMF_THREADS", run.threads.to_string());
    }
    let st = cmd.status().with_context(|| format!("starting {label}"))?;
    if !st.success() {
        run.log(&format!("[{label}] FAILED ({st}) — rerun the same command to resume"));
        bail!("{label} failed ({st})");
    }
    Ok(())
}
