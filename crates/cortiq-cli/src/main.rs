//! Cortiq CLI — sparse task-routed model inference.

mod convert;
mod gguf;
mod npy;

use clap::{Parser, Subcommand};
use cortiq_core::CmfModel;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Counting allocator: one relaxed increment per allocation — cheap
/// enough to keep always-on, precise enough for the roadmap's
/// «allocations/token in steady decode» counter (`bench --json`).
struct CountingAlloc;

static ALLOCS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, AtomicOrdering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, AtomicOrdering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAlloc = CountingAlloc;
use cortiq_engine::{CortiqRuntime, Pipeline, SamplerConfig};
use cortiq_server::{build_router, AppState};
use std::sync::Arc;

/// A frozen conversation state (B2, `.cmfstate`). v1 is LOGICAL: the token
/// prefix + active skill + seed + a model fingerprint. Resume replays the
/// prefix (bit-identical warm state). `kind` reserves the format for a
/// future PHYSICAL variant (serialized KV blobs → instant resume) without
/// a version bump — a reader rejects a kind it does not implement.
const STATE_MAGIC: &[u8; 4] = b"CMFS";
const STATE_KIND_LOGICAL: u32 = 0;

struct SessionState {
    kind: u32,
    /// (num_layers, hidden_size, vocab_size) — reject a wrong-model resume.
    fp: (u32, u32, u32),
    seed: Option<u64>,
    skill: Option<String>,
    tokens: Vec<u32>,
}

impl SessionState {
    fn fingerprint(arch: &cortiq_core::ModelArch) -> (u32, u32, u32) {
        (arch.num_layers as u32, arch.hidden_size as u32, arch.vocab_size as u32)
    }

    fn write(&self, path: &str) -> anyhow::Result<()> {
        let mut b = Vec::new();
        b.extend_from_slice(STATE_MAGIC);
        b.extend_from_slice(&1u32.to_le_bytes()); // version
        b.extend_from_slice(&self.kind.to_le_bytes());
        for v in [self.fp.0, self.fp.1, self.fp.2] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        match self.seed {
            Some(s) => {
                b.push(1);
                b.extend_from_slice(&s.to_le_bytes());
            }
            None => b.push(0),
        }
        let sk = self.skill.as_deref().unwrap_or("");
        b.extend_from_slice(&(sk.len() as u32).to_le_bytes());
        b.extend_from_slice(sk.as_bytes());
        b.extend_from_slice(&(self.tokens.len() as u32).to_le_bytes());
        for &t in &self.tokens {
            b.extend_from_slice(&t.to_le_bytes());
        }
        std::fs::write(path, b)?;
        Ok(())
    }

    fn read(path: &str) -> anyhow::Result<Self> {
        let b = std::fs::read(path)?;
        let mut p = 0usize;
        let take = |p: &mut usize, n: usize| -> anyhow::Result<&[u8]> {
            if *p + n > b.len() {
                anyhow::bail!("truncated .cmfstate");
            }
            let s = &b[*p..*p + n];
            *p += n;
            Ok(s)
        };
        let u32at = |p: &mut usize| -> anyhow::Result<u32> {
            Ok(u32::from_le_bytes(take(p, 4)?.try_into().unwrap()))
        };
        if take(&mut p, 4)? != STATE_MAGIC {
            anyhow::bail!("not a .cmfstate file (bad magic)");
        }
        let _version = u32at(&mut p)?;
        let kind = u32at(&mut p)?;
        let fp = (u32at(&mut p)?, u32at(&mut p)?, u32at(&mut p)?);
        let seed = if take(&mut p, 1)?[0] == 1 {
            Some(u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap()))
        } else {
            None
        };
        let sl = u32at(&mut p)? as usize;
        let skill = {
            let s = std::str::from_utf8(take(&mut p, sl)?)?.to_string();
            if s.is_empty() { None } else { Some(s) }
        };
        let n = u32at(&mut p)? as usize;
        let mut tokens = Vec::with_capacity(n);
        for _ in 0..n {
            tokens.push(u32at(&mut p)?);
        }
        Ok(SessionState { kind, fp, seed, skill, tokens })
    }
}

/// Bundled `--o1*` CLI flags for run/serve/bench. `spec = None` keeps
/// whatever env CMF_O1 / the file's converter hint resolved at load;
/// an explicit spec (including `off`) replaces it.
struct O1Flags {
    spec: Option<String>,
    m: Option<usize>,
    w: Option<usize>,
    sink: Option<usize>,
    rect: Option<String>,
}

impl O1Flags {
    /// Parsed rectifier; None = fall through to CMF_O1_RECT / default.
    fn rect(&self) -> anyhow::Result<Option<cortiq_engine::nystrom::O1Rect>> {
        match self.rect.as_deref() {
            None => Ok(None),
            Some(s) => cortiq_engine::nystrom::O1Cfg::parse_rect(s).map(Some).ok_or_else(|| {
                anyhow::anyhow!("--o1-rect '{s}' is not one of: agg | fm")
            }),
        }
    }

    /// The config this flag set resolves to, or None for `off`/absent.
    fn cfg(&self) -> anyhow::Result<Option<cortiq_engine::nystrom::O1Cfg>> {
        let rect = self.rect()?;
        Ok(self.spec.as_deref().and_then(|spec| {
            cortiq_engine::nystrom::O1Cfg::from_spec(spec, self.m, self.w, self.sink, rect)
        }))
    }

    fn apply(&self, pipeline: &mut Pipeline) {
        if let Some(spec) = self.spec.as_deref() {
            let rect = self.rect().unwrap_or(None);
            pipeline.set_o1(cortiq_engine::nystrom::O1Cfg::from_spec(
                spec, self.m, self.w, self.sink, rect,
            ));
        }
    }
}

/// `ppl --windows N --window-len L`: the val_ppl window discipline.
struct PplWindows {
    windows: Option<usize>,
    window_len: usize,
}

impl PplWindows {
    /// Offsets of the scored windows over `n` tokens: `windows` evenly
    /// spaced starts, `stride = (n - len - 1) / windows` — the exact
    /// selection of `heal_hybridk_06b.py::val_ppl(m, va, bs, n)`, whose
    /// N = n*bs windows sit at (j*bs + b)*stride, i.e. k*stride for
    /// k = 0..N-1. None = score one --tokens prefix instead.
    fn offsets(&self, n: usize) -> anyhow::Result<Option<Vec<usize>>> {
        let Some(w) = self.windows.filter(|&w| w > 0) else {
            return Ok(None);
        };
        anyhow::ensure!(
            n > self.window_len + 1,
            "corpus has {n} tokens < window_len+2 = {}",
            self.window_len + 2
        );
        let stride = (n - self.window_len - 1) / w;
        anyhow::ensure!(stride > 0, "{w} windows of {} do not fit in {n} tokens", self.window_len);
        Ok(Some((0..w).map(|k| k * stride).collect()))
    }
}

#[derive(Parser)]
#[command(name = "cortiq")]
#[command(about = "Sparse task-routed model inference engine")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference server with web dashboard
    Serve {
        /// Path to .cmf model file
        model: String,
        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,
        /// Host / interface to bind (use 127.0.0.1 for local-only)
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// Default task mask
        #[arg(short, long, default_value = "general")]
        task: String,
        /// Also listen on ollama-compatible port
        #[arg(long)]
        compat_port: Option<u16>,
        /// O(1) Nyström attention: replace KV-cache attention on the
        /// given layers (all | deepN | i,j,k | off). Overrides CMF_O1
        /// and the file's converter hint.
        #[arg(long)]
        o1: Option<String>,
        /// Landmark budget for --o1 (validated default 32)
        #[arg(long)]
        o1_m: Option<usize>,
        /// Exact-window width for --o1 (validated default 128)
        #[arg(long)]
        o1_window: Option<usize>,
        /// Permanent exact sink keys for --o1 (validated default 4)
        #[arg(long)]
        o1_sink: Option<usize>,
    },
    /// Convert a Hugging Face checkpoint to .cmf — native Rust, no Python
    Convert {
        /// HF model: a local dir (config.json + *.safetensors + tokenizer.json)
        /// or a hub repo id like `Qwen/Qwen2.5-0.5B-Instruct` (downloaded)
        #[arg(long)]
        model: String,
        /// Quantization for 2-D weights: q8 | q8_2f | q4 | f16 | vbit
        #[arg(long, default_value = "q8")]
        quant: String,
        /// Output .cmf path
        #[arg(long)]
        output: String,
        /// Hugging Face token (for gated/private repos)
        #[arg(long)]
        hf_token: Option<String>,
        /// Target mean bits for `--quant vbit` (3.0–8.0; default 4.25). Higher =
        /// better quality + larger file. Precision-sensitive architectures
        /// (e.g. GatedDeltaNet) may need 5.5–6 to stay coherent.
        #[arg(long, default_value = "4.25")]
        mean_bits: f32,
        /// Physically defragment (Patent 2 claims 9/10): drop pruned FFN
        /// neurons so they are neither stored nor computed. Points at a
        /// skill dir with baked FFN overlays (tensors/*.npy) and/or a
        /// keep-set (ffn_keep.npy); without ffn_keep.npy the keep-set is
        /// autodetected from zeroed down_proj columns. Drops masks. (spec §11)
        #[arg(long)]
        defrag: Option<String>,
        /// Record an O(1) Nyström attention hint (all | deepN | i,j,k):
        /// weights pass through UNCHANGED; the runtime reads the hint at
        /// load (override at serve time with --o1 off). Measured through
        /// the real runtime on Qwen3-0.6B (all 28 layers, wikitext-2):
        /// ×1.296 ppl zero-shot — reproduce with `cortiq ppl --o1 all`.
        #[arg(long)]
        o1: Option<String>,
        /// Landmark budget for the --o1 hint (validated default 32)
        #[arg(long)]
        o1_m: Option<usize>,
        /// Exact-window width for the --o1 hint (validated default 128)
        #[arg(long)]
        o1_window: Option<usize>,
        /// Permanent exact sink keys for the --o1 hint (validated default 4)
        #[arg(long)]
        o1_sink: Option<usize>,
    },
    /// Import a GGUF model to .cmf — native Rust (F32/F16/BF16/Q4_0..Q6_K + K-quants; llama/qwen2/qwen3)
    ImportGguf {
        /// A local .gguf file, an HF repo id (owner/name — best .gguf auto-picked), or owner/name/file.gguf
        gguf: String,
        /// Output .cmf path
        #[arg(long)]
        output: String,
        /// Quantization for 2-D weights: q8 | q8_2f | q4 | f16 | vbit
        #[arg(long, default_value = "q8")]
        quant: String,
        /// Hugging Face token for gated/private GGUF repos
        #[arg(long)]
        hf_token: Option<String>,
    },
    /// Chat with a model (applies the file's chat template), or one-shot
    /// with --prompt
    Run {
        /// Path to .cmf model file
        model: String,
        /// Task mask to use
        #[arg(short, long, default_value = "general")]
        task: String,
        /// Single prompt (non-interactive)
        #[arg(short, long)]
        prompt: Option<String>,
        /// Maximum number of tokens to generate
        #[arg(short = 'n', long, default_value = "256")]
        max_tokens: usize,
        /// Skill to overlay (spec §9): replacement tensors are read in
        /// place of backbone tensors
        #[arg(long)]
        skill: Option<String>,
        /// Greedy decoding (temperature 0) — gates and base models
        #[arg(long)]
        greedy: bool,
        /// Skip the model's chat template: feed the prompt to the model
        /// verbatim (completion mode). Default is to apply the template
        /// when the file carries one; base models without one always run raw.
        #[arg(long)]
        raw: bool,
        /// Render the chat template with enable_thinking=false — reasoning
        /// models (Qwen3/3.5) answer directly instead of emitting a <think>
        /// block.
        #[arg(long, conflicts_with = "raw")]
        no_think: bool,
        /// Soft blend: "auto" (top-2 softmax(−E/T), T=0.4) or "id:w,id:w"
        #[arg(long)]
        blend: Option<String>,
        /// Dynamic per-token skill routing with hysteresis (spec §9): the
        /// active skill switches mid-stream as the context evolves. Tune
        /// via CMF_ROUTE_EON/EOFF/MARGIN/PERIOD.
        #[arg(long)]
        route_dynamic: bool,
        /// After generation, reprint the answer with each token coloured
        /// by the model's confidence (Born mass): green = sure, red =
        /// guessing. The honest house — the model shows where it's unsure.
        #[arg(long)]
        confidence: bool,
        /// Emit the structured per-token telemetry trace (B4): id ·
        /// confidence · active skill · recon-coherence · switch. Add
        /// `--trace-json` for machine-readable JSONL on stderr.
        #[arg(long)]
        trace: bool,
        /// With --trace: also print each row as a JSON object on stderr.
        #[arg(long)]
        trace_json: bool,
        /// Resume a frozen session (B2): replay the `.cmfstate` token
        /// prefix + its skill before this prompt (bit-identical warm state).
        #[arg(long)]
        state: Option<String>,
        /// O(1) Nyström attention: replace KV-cache attention on the
        /// given layers (all | deepN | i,j,k | off). Overrides CMF_O1
        /// and the file's converter hint.
        #[arg(long)]
        o1: Option<String>,
        /// Landmark budget for --o1 (validated default 32)
        #[arg(long)]
        o1_m: Option<usize>,
        /// Exact-window width for --o1 (validated default 128)
        #[arg(long)]
        o1_window: Option<usize>,
        /// Permanent exact sink keys for --o1 (validated default 4)
        #[arg(long)]
        o1_sink: Option<usize>,
    },
    /// Freeze the current context into a `.cmfstate` (B2): the token prefix
    /// + active skill + seed + model fingerprint. Resume with `run --state`.
    Freeze {
        /// Path to .cmf model file
        model: String,
        /// Context text to freeze
        #[arg(short, long)]
        prompt: String,
        /// Output .cmfstate path
        #[arg(short, long)]
        out: String,
        /// Active skill to carry into the frozen session
        #[arg(long)]
        skill: Option<String>,
    },
    /// Show model information
    Info {
        /// Path to .cmf model file
        model: String,
    },
    /// List available masks
    Masks {
        /// Path to .cmf model file
        model: String,
    },
    /// Benchmark inference speed
    Bench {
        /// Path to .cmf model file
        model: String,
        /// Task to benchmark
        #[arg(short, long, default_value = "general")]
        task: String,
        /// Number of tokens to generate
        #[arg(long, default_value = "100")]
        tokens: u32,
        /// Machine-readable output: one JSON object with tok/s plus
        /// steady-state counters (allocations/token, pool
        /// dispatches/token) — the benchmark contract of the roadmap
        #[arg(long)]
        json: bool,
        /// Long-context mode: synthetic prompt of N tokens; reports
        /// prefill/decode at that depth plus the KV/state memory —
        /// O(context) KV for full-attention vs O(1) state for the
        /// linear core (spec §2, vmf_phase)
        #[arg(long)]
        ctx: Option<usize>,
        /// O(1) Nyström attention: replace KV-cache attention on the
        /// given layers (all | deepN | i,j,k | off). Overrides CMF_O1
        /// and the file's converter hint.
        #[arg(long)]
        o1: Option<String>,
        /// Landmark budget for --o1 (validated default 32)
        #[arg(long)]
        o1_m: Option<usize>,
        /// Exact-window width for --o1 (validated default 128)
        #[arg(long)]
        o1_window: Option<usize>,
        /// Permanent exact sink keys for --o1 (validated default 4)
        #[arg(long)]
        o1_sink: Option<usize>,
    },
    /// Score skills for a prompt (recon-argmin routing, spec 9)
    Route {
        /// Path to .cmf model file
        model: String,
        /// Prompt to route
        #[arg(short, long)]
        prompt: String,
    },
    /// Teacher-forced perplexity over a text file (quant gate)
    Ppl {
        /// Path to .cmf model file
        model: String,
        /// Text file to score
        #[arg(short, long)]
        file: String,
        /// Max tokens
        #[arg(long, default_value = "1024")]
        tokens: usize,
        /// Skill to overlay (claim-16 gate: overlaid vs backbone)
        #[arg(long)]
        skill: Option<String>,
        /// Soft blend "id:w,id:w" (claim 14 working tensors)
        #[arg(long)]
        blend: Option<String>,
        /// Dynamic per-window skill routing with hysteresis while scoring
        /// (VMF experiment: CMF_ROUTE_EON/EOFF/MARGIN/PERIOD).
        #[arg(long)]
        route_dynamic: bool,
        /// Score N evenly spaced windows of --window-len tokens instead of
        /// one --tokens prefix, combining them before the exp (the val_ppl
        /// discipline the published Qwen3-0.6B yardstick was measured with:
        /// 12 windows x 512 tokens).
        #[arg(long)]
        windows: Option<usize>,
        /// Token length of each --windows window
        #[arg(long, default_value = "512")]
        window_len: usize,
        /// Score the CONVERTED model: run the O(1) Nyström attention path
        /// (all | deepN | i,j,k | off) over the scored positions instead of
        /// exact attention. Each window's first --o1-prefill tokens run the
        /// exact prompt pass that freezes the landmarks, then every scored
        /// position goes through the real streaming decode kernel. The
        /// EXACT baseline over the identical tokens is printed next to it,
        /// so the ratio is apples-to-apples. Without this flag `ppl` scores
        /// the backbone exactly, even for a model carrying an --o1 hint.
        #[arg(long)]
        o1: Option<String>,
        /// Landmark budget for --o1 (default 32)
        #[arg(long)]
        o1_m: Option<usize>,
        /// Exact-window width for --o1 (default 128)
        #[arg(long)]
        o1_window: Option<usize>,
        /// Permanent exact sink keys for --o1 (default 4)
        #[arg(long)]
        o1_sink: Option<usize>,
        /// Skeleton rectifier for --o1: agg (clamp the aggregate far
        /// denominator) | fm (clamp F_u*M_u >= 0 — per-key non-negativity)
        #[arg(long)]
        o1_rect: Option<String>,
        /// Tokens per window that run the exact prompt pass before the O(1)
        /// seal (default: half the window). Landmarks are frozen from these
        /// tokens only — the runtime never sees the full-sequence landmarks
        /// the published torch probe used.
        #[arg(long)]
        o1_prefill: Option<usize>,
    },
    /// Tell the model's life story: origin, body, skills, integrity —
    /// the file's verifiable autobiography from its own header.
    Story {
        /// Path to .cmf model file
        model: String,
    },
    /// Semantic diff of two .cmf files: arch, quant, tensors (by hash64),
    /// skills — what changed between two model versions.
    Diff {
        /// Baseline .cmf
        a: String,
        /// Compared .cmf
        b: String,
    },
    /// Introspect WITHOUT generating: which skill recon-argmin picks (with
    /// E), and the first-token distribution + confidence — "how it would answer".
    Explain {
        /// Path to .cmf model file
        model: String,
        /// Prompt to introspect
        #[arg(short, long)]
        prompt: String,
        /// How many candidate first tokens to show
        #[arg(long, default_value = "8")]
        top: usize,
    },
    /// Measure confidence calibration (B1): is the model's Born-mass
    /// confidence a true property (80% ⇒ right 80%), or does it need a
    /// measured temperature? Reliability diagram + ECE + fitted T.
    Calibrate {
        /// Path to .cmf model file
        model: String,
        /// Held-out text file to measure on
        #[arg(short, long)]
        file: String,
        /// Skill to overlay while measuring
        #[arg(long)]
        skill: Option<String>,
        /// Max tokens
        #[arg(long, default_value = "800")]
        tokens: usize,
    },
    /// Verify file integrity: envelope, sections, per-tensor hashes
    Verify {
        /// Path to .cmf model file
        model: String,
    },
    /// FCD polish for O(1)-converted models: train the converted
    /// layers' LN gains + FFN against the exact-attention teacher
    /// (0.3·CE + 0.7·KL certified recipe), restore the best checkpoint,
    /// write `<model>.fcd.cmf` (docs/RUST_FCD.md)
    Fcd {
        /// Path to .cmf model file
        model: String,
        /// Plain-text training corpus (tokenized with the embedded tokenizer)
        #[arg(long)]
        corpus: String,
        /// Separate validation text (default: hold out the corpus tail)
        #[arg(long)]
        val_corpus: Option<String>,
        /// Layers to convert+polish: all | deepN | i,j,k (default: the
        /// file's converter hint, else all)
        #[arg(long)]
        o1: Option<String>,
        /// Landmark budget (validated default 32)
        #[arg(long)]
        o1_m: Option<usize>,
        /// Exact-window width (validated default 128)
        #[arg(long)]
        o1_window: Option<usize>,
        /// Permanent exact sink keys (validated default 4)
        #[arg(long)]
        o1_sink: Option<usize>,
        /// Training steps (certified: 300, best at 150)
        #[arg(long, default_value_t = 300)]
        steps: usize,
        /// AdamW learning rate
        #[arg(long, default_value_t = 5e-5)]
        lr: f64,
        /// KL(teacher‖student) weight in the loss
        #[arg(long, default_value_t = 0.7)]
        kl: f64,
        /// Quick-val cadence (steps)
        #[arg(long, default_value_t = 25)]
        eval_every: usize,
        /// Sequences per step
        #[arg(long, default_value_t = 2)]
        bs: usize,
        /// Window length in tokens
        #[arg(long, default_value_t = 512)]
        seq: usize,
        /// Output path (default: <model>.fcd.cmf)
        #[arg(long)]
        out: Option<String>,
        /// Run 3 greedy 400-token-prompt generations through the REAL
        /// streaming O(1) runtime before and after the polish (the loop
        /// gate)
        #[arg(long, default_value_t = false)]
        gen_check: bool,
        /// Generation-gated checkpoint selection (Patent 16 draft,
        /// claim 13): only checkpoints whose greedy generations stay
        /// under the loop-score gate are eligible for restore; if none
        /// passes, the zero-shot state is written (identity polish).
        /// Defaults ON when --gen-check is on.
        #[arg(long, default_value_t = false)]
        gen_gate: bool,
        /// Gate: max loop score per prompt (checkpoint fails above)
        #[arg(long, default_value_t = 0.35)]
        gate_threshold: f64,
        /// Gate: max loop-score increase over the zero-shot baseline
        #[arg(long, default_value_t = 0.10)]
        gate_slack: f64,
    },
}

/// Convert/import progress. `@PROGRESS <fraction>` is a marker for supervisors
/// that capture stdout (they parse it for a progress bar); on a terminal those
/// same hundreds of lines are noise, so paint one line in place instead.
fn progress_reporter(what: &'static str) -> impl FnMut(f32) {
    use std::io::IsTerminal;
    let tty = std::io::stdout().is_terminal();
    let mut done = false;
    move |f: f32| {
        use std::io::Write;
        if !tty {
            println!("@PROGRESS {f:.4}");
            return;
        }
        print!("\r  {what}: {:>5.1}%", f * 100.0);
        let _ = std::io::stdout().flush();
        if f >= 1.0 && !done {
            done = true;
            println!();
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // `run` hands the screen to the model: the loader's INFO chatter is noise
    // in front of an answer. Every other command keeps the informative
    // default. RUST_LOG overrides either way.
    let default_level = match &cli.command {
        Commands::Run { .. } => "warn",
        _ => "info",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| default_level.into()),
        )
        // Logs go to stderr: stdout carries the payload (generated text,
        // `bench --json`) and must stay machine-parseable.
        .with_writer(std::io::stderr)
        .init();

    match cli.command {
        Commands::Serve {
            model,
            port,
            host,
            task,
            compat_port,
            o1,
            o1_m,
            o1_window,
            o1_sink,
        } => {
            let o1 = O1Flags { spec: o1, m: o1_m, w: o1_window, sink: o1_sink, rect: None };
            cmd_serve(&model, &host, port, &task, compat_port, &o1).await
        }
        Commands::Convert {
            model, quant, output, hf_token, mean_bits, defrag,
            o1, o1_m, o1_window, o1_sink,
        } => {
            convert::set_vbit_mean_bits(mean_bits);
            // --o1: record the runtime hint in header provenance; the
            // weights pass through unchanged (this is metadata only).
            let o1_hint = match o1.as_deref() {
                None => None,
                Some(spec) => {
                    // The rectifier is a runtime knob, not a property of
                    // the weights — a file hint never pins it.
                    let cfg = cortiq_engine::nystrom::O1Cfg::from_spec(
                        spec, o1_m, o1_window, o1_sink, None,
                    )
                    .ok_or_else(|| {
                        anyhow::anyhow!("--o1 {spec}: expected all | deepN | i,j,k")
                    })?;
                    println!(
                        "o1 hint: layers {spec}, m={} w={} sink={} — weights unchanged; \
                         serve/run/bench read the hint automatically (disable with --o1 off)",
                        cfg.m, cfg.w, cfg.sink
                    );
                    Some(serde_json::json!({
                        "layers": spec, "m": cfg.m, "w": cfg.w, "sink": cfg.sink,
                    }))
                }
            };
            convert::run_convert(&model, &quant, &output, hf_token.as_deref(), defrag.as_deref(),
                                 o1_hint, progress_reporter("converting"))?;
            println!("✓ wrote {output}");
            Ok(())
        }
        Commands::ImportGguf { gguf, output, quant, hf_token } => {
            gguf::run_import_gguf(&gguf, &quant, &output, hf_token.as_deref(),
                                  progress_reporter("importing"))?;
            println!("✓ wrote {output}");
            Ok(())
        }
        Commands::Run {
            model,
            task,
            prompt,
            max_tokens,
            skill,
            greedy,
            raw,
            no_think,
            blend,
            route_dynamic,
            confidence,
            trace,
            trace_json,
            state,
            o1,
            o1_m,
            o1_window,
            o1_sink,
        } => {
            let o1 = O1Flags { spec: o1, m: o1_m, w: o1_window, sink: o1_sink, rect: None };
            cmd_run(&model, &task, prompt.as_deref(), max_tokens, skill.as_deref(), greedy,
                    raw, no_think, blend.as_deref(), route_dynamic, confidence, trace,
                    trace_json, state.as_deref(), &o1)
            .await
        }
        Commands::Freeze { model, prompt, out, skill } => {
            cmd_freeze(&model, &prompt, &out, skill.as_deref())
        }
        Commands::Route { model, prompt } => cmd_route(&model, &prompt),
        Commands::Ppl {
            model,
            file,
            tokens,
            skill,
            blend,
            route_dynamic,
            windows,
            window_len,
            o1,
            o1_m,
            o1_window,
            o1_sink,
            o1_rect,
            o1_prefill,
        } => cmd_ppl(
            &model,
            &file,
            tokens,
            skill.as_deref(),
            blend.as_deref(),
            route_dynamic,
            PplWindows { windows, window_len },
            &O1Flags { spec: o1, m: o1_m, w: o1_window, sink: o1_sink, rect: o1_rect },
            o1_prefill,
        ),
        Commands::Info { model } => cmd_info(&model).await,
        Commands::Story { model } => cmd_story(&model),
        Commands::Diff { a, b } => cmd_diff(&a, &b),
        Commands::Explain { model, prompt, top } => cmd_explain(&model, &prompt, top),
        Commands::Calibrate { model, file, skill, tokens } => {
            cmd_calibrate(&model, &file, skill.as_deref(), tokens)
        }
        Commands::Masks { model } => cmd_masks(&model).await,
        Commands::Bench {
            model,
            task,
            tokens,
            json,
            ctx,
            o1,
            o1_m,
            o1_window,
            o1_sink,
        } => {
            let o1 = O1Flags { spec: o1, m: o1_m, w: o1_window, sink: o1_sink, rect: None };
            cmd_bench(&model, &task, tokens, ctx, &o1, json).await
        }
        Commands::Verify { model } => cmd_verify(&model).await,
        Commands::Fcd {
            model,
            corpus,
            val_corpus,
            o1,
            o1_m,
            o1_window,
            o1_sink,
            steps,
            lr,
            kl,
            eval_every,
            bs,
            seq,
            out,
            gen_check,
            gen_gate,
            gate_threshold,
            gate_slack,
        } => cmd_fcd(
            &model,
            &corpus,
            val_corpus.as_deref(),
            o1.as_deref(),
            o1_m,
            o1_window,
            o1_sink,
            steps,
            lr,
            kl,
            eval_every,
            bs,
            seq,
            out.as_deref(),
            gen_check,
            gen_gate,
            gate_threshold,
            gate_slack,
        ),
    }
}

async fn cmd_serve(
    model_path: &str,
    host: &str,
    port: u16,
    default_task: &str,
    _compat_port: Option<u16>,
    o1: &O1Flags,
) -> anyhow::Result<()> {
    println!();
    println!("  ╔═══════════════════════════════════════╗");
    println!("  ║     Cortiq — Sparse Inference Engine   ║");
    println!("  ╚═══════════════════════════════════════╝");
    println!();

    // Load model + pipeline (real weights; fails loudly on a bad file).
    println!("  Loading model: {}", model_path);
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let arch = model.arch();
    println!(
        "    Architecture: {} | {}L | hidden={} | FFN={}",
        arch.arch_name, arch.num_layers, arch.hidden_size, arch.intermediate_size
    );
    println!("    Quantization: {:?}", model.header.quant_type);
    println!("    Masks: {}", model.masks.masks.len());

    // Slot pool (roadmap этап 5.1): N pipelines over ONE mmap — the
    // weights are shared zero-copy, each slot owns KV/state/workspace,
    // so up to N requests decode concurrently. CMF_SERVE_SLOTS
    // overrides; the default keeps ~4 pool threads per slot.
    let avail = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let slots = std::env::var("CMF_SERVE_SLOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| (avail / 4).clamp(1, 4));
    if std::env::var("CMF_THREADS").is_err() {
        // Split the cores between slots instead of oversubscribing
        // N pools × (cores−1) workers. Explicit CMF_THREADS wins.
        let per = (avail.saturating_sub(1) / slots).max(1);
        // SAFETY: single-threaded startup, before any pipeline/pool spawn.
        unsafe { std::env::set_var("CMF_THREADS", per.to_string()) };
    }
    let mut pipelines = Vec::with_capacity(slots);
    for _ in 0..slots {
        let mut pipeline = Pipeline::from_model(&model, SamplerConfig::default())?;
        o1.apply(&mut pipeline);
        pipelines.push(pipeline);
    }
    if pipelines[0].o1_active() {
        println!("    O(1) attention: nystrom (see load log for layers/params)");
    }
    println!(
        "    Pipeline: loaded ({:.2}B params) | {} slot(s) × {} thread(s)",
        model.total_param_count() as f64 / 1e9,
        slots,
        std::env::var("CMF_THREADS").unwrap_or_default(),
    );
    println!();

    // Create runtime
    let runtime = CortiqRuntime::new(model);
    if runtime.masks().get(default_task).is_some() {
        let _ = runtime.switch_task(default_task).await;
    }
    let tokenizer = pipelines[0].tokenizer.clone();
    let state = Arc::new(AppState {
        runtime,
        tokenizer,
        slots: cortiq_server::PipelinePool::new(pipelines),
    });

    // Build router
    let app = build_router(state);

    // Start server
    let addr = format!("{}:{}", host, port);
    println!("  ✓ API server:     http://{}:{}/v1/chat/completions", host, port);
    println!("  ✓ Web dashboard:  http://localhost:{}/", port);
    println!("  ✓ Status:         http://localhost:{}/v1/cortiq/status", port);
    println!();
    println!("  Press Ctrl+C to stop.");
    println!();

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Server listening on {}", addr);
    axum::serve(listener, app).await?;

    Ok(())
}

/// Claim 14 end-to-end: route → top-2 → softmax(−E/T), T=0.4
/// (the owner's validated default). Probe = first ≤128 tokens of the text.
fn auto_blend(model: &Arc<CmfModel>, text: &str) -> anyhow::Result<Vec<(String, f32)>> {
    let mut probe = Pipeline::from_model(model, SamplerConfig::default())?;
    let mut ids = probe.tokenizer.encode(text);
    ids.truncate(128);
    let routes = cortiq_engine::router::route(model, &mut probe, &ids);
    if routes.len() < 2 {
        anyhow::bail!("blend auto needs ≥2 routable skills");
    }
    let t = 0.4f32;
    let m = &routes[..2];
    let mx = -m[0].error / t;
    let ws: Vec<f32> = m.iter().map(|r| (-r.error / t - mx).exp()).collect();
    let sum: f32 = ws.iter().sum();
    Ok(m.iter()
        .zip(&ws)
        .map(|(r, w)| (r.id.clone(), w / sum))
        .collect())
}

fn parse_blend(spec: &str) -> anyhow::Result<Vec<(String, f32)>> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let (id, w) = part
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("blend format: id:w,id:w"))?;
        out.push((id.trim().to_string(), w.trim().parse::<f32>()?));
    }
    let sum: f32 = out.iter().map(|(_, w)| w).sum();
    for (_, w) in out.iter_mut() {
        *w /= sum.max(1e-9);
    }
    Ok(out)
}

/// Char caps for the FCD corpus — the certified recipe tokenized the
/// first 2M train / 200K val chars of wikitext-2-raw.
const FCD_TRAIN_CHARS: usize = 2_000_000;
const FCD_VAL_CHARS: usize = 200_000;

/// Three greedy generations from 400-token val prompts through the
/// REAL streaming O(1) runtime (the loop gate of the torch reference:
/// offsets L/10, L/2, 8L/10 of the val stream).
fn fcd_gen_check(
    model: &Arc<CmfModel>,
    o1: &cortiq_engine::nystrom::O1Cfg,
    va: &[u32],
    tag: &str,
) -> anyhow::Result<()> {
    let greedy = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        min_p: 0.0,
        seed: Some(0),
    };
    let mut pipeline = Pipeline::from_model(model, greedy)?;
    pipeline.set_o1(Some(o1.clone()));
    let l = va.len().saturating_sub(500);
    if l < 400 {
        println!("gen-check {tag}: val stream too short, skipped");
        return Ok(());
    }
    for off in [l / 10, l / 2, 8 * l / 10] {
        let prompt = &va[off..off + 400];
        let r = pipeline
            .generate_from_ids(prompt, 60, None, None)
            .map_err(|e| anyhow::anyhow!("generation: {e}"))?;
        println!(
            "GEN {tag} (off {off}, loop-score {:.2}): {}",
            cortiq_engine::fcd::loop_score(&r.token_ids),
            r.text.replace('\n', "\\n")
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_fcd(
    model_path: &str,
    corpus: &str,
    val_corpus: Option<&str>,
    o1: Option<&str>,
    o1_m: Option<usize>,
    o1_w: Option<usize>,
    o1_sink: Option<usize>,
    steps: usize,
    lr: f64,
    kl: f64,
    eval_every: usize,
    bs: usize,
    seq: usize,
    out: Option<&str>,
    gen_check: bool,
    gen_gate: bool,
    gate_threshold: f64,
    gate_slack: f64,
) -> anyhow::Result<()> {
    use cortiq_engine::fcd::{run_polish, FcdHyper, GenGateCfg};
    use cortiq_engine::nystrom::O1Cfg;

    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    // Layer set: explicit flag > file converter hint > all.
    let cfg = match o1 {
        Some(spec) => O1Cfg::from_spec(spec, o1_m, o1_w, o1_sink, None).ok_or_else(|| {
            anyhow::anyhow!("--o1 '{spec}' is off or malformed — nothing to polish")
        })?,
        None => model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("o1_attn"))
            .and_then(O1Cfg::from_json)
            .or_else(|| O1Cfg::from_spec("all", o1_m, o1_w, o1_sink, None))
            .expect("'all' always parses"),
    };

    // Tokenizer: embedded → sidecar. No byte-level fallback here — the
    // polish must train on the model's true token ids.
    let tokenizer = if let Some(vb) = &model.vocab {
        cortiq_engine::tokenizer::Tokenizer::from_bytes(vb)
            .map_err(|e| anyhow::anyhow!("embedded tokenizer: {e}"))?
    } else {
        let sidecar = std::path::Path::new(model_path).with_file_name("tokenizer.json");
        anyhow::ensure!(
            sidecar.exists(),
            "no tokenizer in the file or beside it — cannot tokenize the corpus"
        );
        cortiq_engine::tokenizer::Tokenizer::from_file(&sidecar)
            .map_err(|e| anyhow::anyhow!("sidecar tokenizer: {e}"))?
    };

    let cap = |s: String, n: usize| -> String {
        if s.len() > n {
            let mut end = n;
            while !s.is_char_boundary(end) {
                end += 1;
            }
            s[..end].to_string()
        } else {
            s
        }
    };
    let train_text = cap(std::fs::read_to_string(corpus)?, FCD_TRAIN_CHARS);
    println!("tokenizing corpus ({} chars)…", train_text.len());
    let mut tr = tokenizer.encode(&train_text);
    let va: Vec<u32> = match val_corpus {
        Some(p) => {
            let vt = cap(std::fs::read_to_string(p)?, FCD_VAL_CHARS);
            tokenizer.encode(&vt)
        }
        None => {
            // Hold out the corpus tail (never sampled for training).
            let cut = tr.len() - tr.len() / 10;
            tr.split_off(cut)
        }
    };
    println!("corpus: train {} tokens, val {} tokens", tr.len(), va.len());

    if gen_check {
        println!("── gen-check BEFORE polish (zero-shot O(1)) ──");
        fcd_gen_check(&model, &cfg, &va, "before")?;
    }

    let hp = FcdHyper { steps, lr, kl_w: kl, eval_every, bs, seq, seed: 0 };
    let out_path = out
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(format!("{model_path}.fcd.cmf")));
    // Gate default: on whenever --gen-check is on (claim 13 discipline).
    let gate_cfg = (gen_gate || gen_check)
        .then(|| GenGateCfg::standard(&va))
        .flatten()
        .map(|mut g| {
            g.threshold = gate_threshold;
            g.baseline_slack = gate_slack;
            g
        });
    let report = run_polish(&model, &cfg, &hp, &tr, &va, &out_path, gate_cfg.as_ref())
        .map_err(|e| anyhow::anyhow!("fcd polish: {e}"))?;

    println!("── FCD polish report ──");
    println!("converted layers : {:?}", report.converted);
    println!("teacher val-ppl  : {:.2}", report.teacher_ppl);
    println!("student ppl start: {:.2} (zero-shot O(1))", report.ppl_start);
    println!(
        "student ppl best : {:.2} (step {}), final {:.2}",
        report.ppl_best, report.best_step, report.ppl_final
    );
    println!(
        "steps            : {} ({:.1}s/step)",
        report.steps_run, report.sec_per_step
    );
    if let Some(gr) = &report.gate {
        println!("gen-gate baseline: {:?}", gr.baseline);
        for (st, ppl, scores, pass) in &gr.evals {
            println!(
                "gen-gate step {st}: ppl {ppl:.2} scores {scores:?} → {}",
                if *pass { "PASS" } else { "FAIL" }
            );
        }
        match gr.chosen {
            Some(st) => println!("gen-gate chose   : step {st}"),
            None => println!("gen-gate chose   : IDENTITY (polish rejected)"),
        }
    }
    println!("wrote            : {}", out_path.display());

    if gen_check {
        println!("── gen-check AFTER polish (streaming O(1) runtime) ──");
        let polished = Arc::new(CmfModel::open_sharded(
            out_path.to_str().unwrap_or_default(),
        )?);
        fcd_gen_check(&polished, &cfg, &va, "after")?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_ppl(
    model_path: &str,
    file: &str,
    max_tokens: usize,
    skill: Option<&str>,
    blend: Option<&str>,
    route_dynamic: bool,
    win: PplWindows,
    o1: &O1Flags,
    o1_prefill: Option<usize>,
) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let text = std::fs::read_to_string(file)?;
    let mut pipeline = match blend {
        Some("auto") => {
            let b = auto_blend(&model, &text)?;
            println!("blend auto: {b:?}");
            Pipeline::from_model_with_blend(&model, SamplerConfig::default(), &b)?
        }
        Some(spec) => {
            let b = parse_blend(spec)?;
            println!("blend: {b:?}");
            Pipeline::from_model_with_blend(&model, SamplerConfig::default(), &b)?
        }
        None => Pipeline::from_model_with_skill(&model, SamplerConfig::default(), skill)?,
    };
    // Windowed scoring keeps the RAW token stream: the val_ppl yardstick
    // slices windows out of the middle of the corpus, where a prepended
    // BOS would be a token the reference never scored.
    let windowed = win.offsets(pipeline.tokenizer.encode(&text).len())?;
    let mut ids = match &windowed {
        Some(_) => pipeline.tokenizer.encode(&text),
        None => pipeline.tokenizer.with_bos(pipeline.tokenizer.encode(&text)),
    };
    if windowed.is_none() {
        ids.truncate(max_tokens);
    }

    if let Some(offsets) = windowed {
        return ppl_windows(&mut pipeline, &ids, &offsets, win.window_len, o1, o1_prefill);
    }
    if let Some(cfg) = o1.cfg()? {
        // Single-prefix o1 scoring: seal after --o1-prefill (default:
        // half the sequence), score the rest through the O(1) kernel.
        pipeline.set_o1(Some(cfg));
        let prefill = o1_prefill.unwrap_or(ids.len() / 2).min(ids.len().saturating_sub(1));
        let (n_o1, c) = pipeline.nll_ids_o1(&ids, prefill);
        pipeline.set_o1(None);
        let (n_ex, _) = pipeline.nll_ids_from(&ids, prefill);
        report_o1_ppl(n_o1, n_ex, c, prefill, ids.len());
        dump_moe_stats(&pipeline)?;
        return Ok(());
    }
    if route_dynamic {
        let n = pipeline.enable_dynamic_routing();
        let (ppl, switches) = pipeline.ppl_ids_dynamic(&ids);
        println!(
            "PPL = {ppl:.3} over {} tokens | dynamic routing: {n} skills, {switches} switch(es)",
            ids.len()
        );
        dump_moe_stats(&pipeline)?;
        return Ok(());
    }
    let ppl = pipeline.ppl_ids(&ids);
    println!("PPL = {ppl:.3} over {} tokens", ids.len());

    // B-field of claim 12: router expert-selection frequencies on this
    // run → JSON {layer: [counts]} for the converter's flood-fill.
    dump_moe_stats(&pipeline)?;
    Ok(())
}

/// The O(1) score against its own exact baseline. `cnt` scored tokens
/// per sequence, positions `prefill..len-1`.
fn report_o1_ppl(nll_o1: f64, nll_exact: f64, cnt: usize, prefill: usize, len: usize) {
    let c = cnt.max(1) as f64;
    let (p_o1, p_ex) = ((nll_o1 / c).exp(), (nll_exact / c).exp());
    // Scored positions are prefill..len-1 EXCLUSIVE: position len-1 has
    // no next token to predict.
    println!(
        "PPL(o1, CONVERTED model) = {p_o1:.3} over {cnt} scored token(s) \
         [positions {prefill}..{} of {len}, per sequence]",
        len.saturating_sub(1)
    );
    println!("PPL(exact, same tokens)  = {p_ex:.3}");
    println!("ratio                    = x{:.3}", p_o1 / p_ex);
}

/// Score `offsets.len()` windows of `wlen` tokens, combining NLL across
/// windows BEFORE the exp (val_ppl discipline). With `--o1`, each window
/// is prefilled exactly, sealed, and its tail scored through the O(1)
/// kernel, next to the exact baseline over the identical tokens.
fn ppl_windows(
    pipeline: &mut Pipeline,
    ids: &[u32],
    offsets: &[usize],
    wlen: usize,
    o1: &O1Flags,
    o1_prefill: Option<usize>,
) -> anyhow::Result<()> {
    let cfg = o1.cfg()?;
    let prefill = match &cfg {
        Some(_) => o1_prefill.unwrap_or(wlen / 2).min(wlen.saturating_sub(1)),
        None => 0,
    };
    let (mut nll_o1, mut nll_ex, mut cnt) = (0f64, 0f64, 0usize);
    for &off in offsets {
        let w = &ids[off..off + wlen];
        if let Some(c) = &cfg {
            pipeline.set_o1(Some(c.clone()));
            let (n, k) = pipeline.nll_ids_o1(w, prefill);
            nll_o1 += n;
            pipeline.set_o1(None);
            let (n, k2) = pipeline.nll_ids_from(w, prefill);
            debug_assert_eq!(k, k2, "o1 and exact must score the same tokens");
            nll_ex += n;
            cnt += k;
        } else {
            let (n, k) = pipeline.nll_ids_from(w, 0);
            nll_ex += n;
            cnt += k;
        }
    }
    println!(
        "windows: {} x {wlen} tokens at stride {} ({} scored)",
        offsets.len(),
        offsets.get(1).copied().unwrap_or(0),
        cnt
    );
    match &cfg {
        Some(c) => {
            println!(
                "o1: layers {:?}, m={} w={} sink={} rect={:?}, prefill={prefill}",
                c.layers, c.m, c.w, c.sink, c.rect
            );
            report_o1_ppl(nll_o1, nll_ex, cnt, prefill, wlen);
        }
        None => println!("PPL = {:.3} (exact attention)", (nll_ex / cnt.max(1) as f64).exp()),
    }
    dump_moe_stats(pipeline)?;
    Ok(())
}

/// B-field of claim 12: expert-selection frequencies of this run →
/// JSON {layer: [counts]} (CMF_MOE_STATS=file). Works for both
/// teacher-forcing (ppl) and on-policy generation (run) —
/// VMF fireball principle: the observable = integral over the trajectory.
fn dump_moe_stats(pipeline: &Pipeline) -> anyhow::Result<()> {
    if let Ok(path) = std::env::var("CMF_MOE_STATS") {
        let mut parts = Vec::new();
        for (li, lw) in pipeline.weights.layers.iter().enumerate() {
            if let cortiq_engine::pipeline::FfnKind::Moe(m) = &lw.ffn {
                let st = m.stats.borrow();
                let counts: Vec<String> = st.iter().map(u64::to_string).collect();
                parts.push(format!("\"{li}\":[{}]", counts.join(",")));
            }
        }
        std::fs::write(&path, format!("{{{}}}", parts.join(",")))?;
        println!("router MoE stats → {path} ({} layers)", parts.len());
    }
    Ok(())
}

fn cmd_route(model_path: &str, prompt: &str) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let mut pipeline = Pipeline::from_model(&model, SamplerConfig::default())?;
    let ids = pipeline.tokenizer.encode(prompt);
    let routes = cortiq_engine::router::route(&model, &mut pipeline, &ids);
    if routes.is_empty() {
        println!("no routable skills in this container");
        return Ok(());
    }
    for r in &routes {
        println!("  {:<20} E = {:.4}", r.id, r.error);
    }
    println!("winner: {}", routes[0].id);
    Ok(())
}

/// Introspection without generation (ROADMAP A4): show recon-argmin skill
/// selection (with E) and the first-token distribution the routed model
/// would emit, plus its Born-mass confidence. Everything shown is a
/// quantity already computed by the runtime — no synthesis.
fn cmd_explain(model_path: &str, prompt: &str, top: usize) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let mut probe = Pipeline::from_model(&model, SamplerConfig::default())?;
    let ids = probe.tokenizer.encode(prompt);
    if ids.is_empty() {
        anyhow::bail!("empty prompt: nothing to explain");
    }
    println!("\n\x1b[1m🔍 explain: {model_path}\x1b[0m");
    println!("Prompt: {prompt:?}  ({} tokens)", ids.len());

    // ── Routing: which skill recon-argmin would pick ──
    let routes = cortiq_engine::router::route(&model, &mut probe, &ids);
    let winner: Option<String> = if routes.is_empty() {
        println!("\nSwarm: none (flat model) — no routing needed, the backbone answers.");
        None
    } else {
        println!("\n\x1b[1mRouting (recon-argmin, E=‖r−BBᵀr‖²/‖φ‖², lower = more coherent):\x1b[0m");
        let emax = routes.iter().map(|r| r.error).fold(0.0f32, f32::max).max(1e-6);
        for (i, r) in routes.iter().enumerate() {
            // Bar: shorter = lower E = more coherent (inverse scale).
            let fill = ((1.0 - r.error / emax) * 20.0).round() as usize;
            let bar = "█".repeat(fill);
            let mark = if i == 0 { "  \x1b[1m← chosen\x1b[0m" } else { "" };
            println!("  {:<12} E = {:.4}  {}{}", r.id, r.error, bar, mark);
        }
        Some(routes[0].id.clone())
    };

    // ── First token: distribution and confidence (Born mass) ──
    // Apply the chosen skill to show EXACTLY the routed answer.
    let mut pipeline = match &winner {
        Some(id) => Pipeline::from_model_with_skill(&model, SamplerConfig::default(), Some(id))?,
        None => Pipeline::from_model(&model, SamplerConfig::default())?,
    };
    let logits = pipeline.prefill_next_logits(&ids, None);
    let t = pipeline.calib_temp(); // B1: calibrated confidence if the file carries it
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let sum: f32 = logits.iter().map(|&v| ((v - max) / t).exp()).sum();
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, ((v - max) / t).exp() / sum))
        .collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let via = winner.as_deref().unwrap_or("backbone");
    println!("\n\x1b[1mFirst token (how it would start answering, via «{via}»):\x1b[0m");
    for (id, p) in probs.iter().take(top) {
        let piece = pipeline.tokenizer.decode_token(*id as u32).replace('\n', "⏎");
        let fill = (p * 30.0).round() as usize;
        println!("  {}  {:>5.1}%  {:?}", "█".repeat(fill.max(1)), p * 100.0, piece);
    }
    let top1 = probs[0].1;
    println!(
        "Confidence on the 1st token: {} (Born mass top-1)",
        conf_colour(&format!("{:.0}%", top1 * 100.0), top1)
    );
    Ok(())
}

/// Freeze a context into a `.cmfstate` (B2, logical v1): tokenize the
/// context exactly as generation would (no BOS — matches `generate`), and
/// store it with the active skill, seed, and a model fingerprint. Resume
/// via `run --state` replays these tokens (bit-identical warm state).
fn cmd_freeze(
    model_path: &str,
    prompt: &str,
    out: &str,
    skill: Option<&str>,
) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    if let Some(s) = skill {
        let known = model.header.skills.iter().any(|k| k.id == s)
            || model.skill_tensors(s).next().is_some();
        if !known {
            anyhow::bail!("skill '{s}' not in this container");
        }
    }
    let pipeline = Pipeline::from_model(&model, SamplerConfig::default())?;
    let tokens = pipeline.tokenizer.encode(prompt); // == generate()'s tokenization
    let st = SessionState {
        kind: STATE_KIND_LOGICAL,
        fp: SessionState::fingerprint(model.arch()),
        seed: SamplerConfig::default().seed,
        skill: skill.map(str::to_string),
        tokens,
    };
    st.write(out)?;
    println!(
        "frozen: {} tokens, skill {}, → {out}  (resume: cortiq run {model_path} --state {out} -p \"…\")",
        st.tokens.len(),
        skill.unwrap_or("—")
    );
    Ok(())
}

/// Expected Calibration Error over 10 equal-width bins of confidence:
/// Σ (n_bin/N)·|accuracy_bin − mean_confidence_bin|. Also returns the
/// per-bin (mean_conf, accuracy, count) for a reliability diagram.
fn ece_bins(conf: &[f32], correct: &[bool]) -> (f32, Vec<(f32, f32, usize)>) {
    let n = conf.len().max(1);
    let mut bins = vec![(0.0f64, 0usize, 0usize); 10]; // (sum_conf, n_correct, n)
    for (&c, &ok) in conf.iter().zip(correct) {
        let b = ((c * 10.0) as usize).min(9);
        bins[b].0 += c as f64;
        bins[b].1 += ok as usize;
        bins[b].2 += 1;
    }
    let mut ece = 0.0f64;
    let diagram = bins
        .iter()
        .map(|&(sc, nc, nb)| {
            if nb == 0 {
                (0.0, 0.0, 0)
            } else {
                let mc = sc / nb as f64;
                let acc = nc as f64 / nb as f64;
                ece += (nb as f64 / n as f64) * (acc - mc).abs();
                (mc as f32, acc as f32, nb)
            }
        })
        .collect();
    (ece as f32, diagram)
}

/// Measure confidence calibration (B1). Teacher-forces held-out text,
/// scores the model's Born-mass confidence against whether its argmax was
/// the real next token, over a temperature grid — reliability diagram +
/// ECE + the temperature that best calibrates. Honest: if already
/// calibrated, says so (no bytes needed).
fn cmd_calibrate(
    model_path: &str,
    file: &str,
    skill: Option<&str>,
    max_tokens: usize,
) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let text = std::fs::read_to_string(file)?;
    let mut pipeline = Pipeline::from_model_with_skill(&model, SamplerConfig::default(), skill)?;
    let mut ids = pipeline.tokenizer.with_bos(pipeline.tokenizer.encode(&text));
    ids.truncate(max_tokens);
    let temps: Vec<f32> = vec![0.5, 0.65, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2];
    let t1 = temps.iter().position(|&t| (t - 1.0).abs() < 1e-6).unwrap();

    println!("\n\x1b[1m🎯 Confidence calibration: {model_path}\x1b[0m");
    println!("Held-out: {file}  ({} tokens){}", ids.len(),
             skill.map(|s| format!(", skill {s}")).unwrap_or_default());
    let (correct, pmax) = pipeline.calib_ids(&ids, &temps);
    if correct.is_empty() {
        anyhow::bail!("too few tokens to calibrate");
    }
    let acc = correct.iter().filter(|&&c| c).count() as f32 / correct.len() as f32;

    // ECE at each temperature; pick the minimizer.
    let col = |ti: usize| -> Vec<f32> { pmax.iter().map(|r| r[ti]).collect() };
    let mut best = (t1, f32::INFINITY);
    let mut eces = Vec::new();
    for (ti, &t) in temps.iter().enumerate() {
        let (ece, _) = ece_bins(&col(ti), &correct);
        eces.push((t, ece));
        if ece < best.1 {
            best = (ti, ece);
        }
    }
    let (ece_raw, diag_raw) = ece_bins(&col(t1), &correct);
    let conf_raw: f32 = col(t1).iter().sum::<f32>() / correct.len() as f32;

    println!(
        "\nArgmax accuracy (top-1 == actual): {:.1}%   mean confidence (T=1): {:.1}%",
        acc * 100.0, conf_raw * 100.0
    );
    let verdict = if conf_raw > acc + 0.02 { "overconfident" }
                  else if conf_raw + 0.02 < acc { "underconfident" }
                  else { "well calibrated" };
    println!("Raw Born mass: \x1b[1m{verdict}\x1b[0m (ECE = {:.3})", ece_raw);

    // Reliability diagram at T=1: conf-bin vs actual accuracy.
    println!("\n  reliability diagram (T=1):  bin  conf   acc    n");
    for (b, &(mc, ac, nb)) in diag_raw.iter().enumerate() {
        if nb == 0 { continue; }
        let bar = "█".repeat((ac * 20.0).round() as usize);
        let sign = if mc > ac + 0.03 { "↑over" } else if ac > mc + 0.03 { "↓under" } else { "·" };
        println!("   {:>2}0%   {:>4.0}%  {:>4.0}%  {:>4}  {} {}",
                 b, mc * 100.0, ac * 100.0, nb, bar, sign);
    }

    let (bt, bece) = (temps[best.0], best.1);
    println!("\n  ECE by temperature:");
    for (t, e) in &eces {
        let mark = if (*t - bt).abs() < 1e-6 { "  ← best" } else { "" };
        println!("   T={:<4} ECE {:.3}{}", t, e, mark);
    }
    if (bt - 1.0).abs() < 1e-6 || bece + 0.005 > ece_raw {
        println!("\n\x1b[1mVerdict: already calibrated (T≈1).\x1b[0m No separate field needed — \
                  Born mass is itself the honest confidence.");
    } else {
        println!(
            "\n\x1b[1mVerdict: temperature T={bt} lowers ECE {:.3}→{:.3}\x1b[0m ({:.0}% of calibration error removed).",
            ece_raw, bece, (1.0 - bece / ece_raw.max(1e-6)) * 100.0
        );
        println!("Write into header (additive): \x1b[2mpython converter/set_calibration.py {model_path} --temperature {bt}\x1b[0m");
        println!("The runtime will apply it to --confidence/--trace/explain (calibrated Born mass).");
    }
    Ok(())
}

/// Colour a token by the model's confidence (Born mass): a 5-step ramp
/// from bright green (sure) to red (guessing). 24-bit ANSI.
fn conf_colour(text: &str, conf: f32) -> String {
    let (r, g, b) = if conf >= 0.8 {
        (80, 220, 100) // very sure — green
    } else if conf >= 0.55 {
        (150, 210, 90)
    } else if conf >= 0.35 {
        (220, 210, 80) // hesitant — yellow
    } else if conf >= 0.18 {
        (230, 150, 60) // shaky — orange
    } else {
        (230, 90, 80) // guessing — red
    };
    format!("\x1b[38;2;{r};{g};{b}m{text}\x1b[0m")
}

/// Render the structured per-token telemetry trace (B4). Every column is
/// a measured quantity: Born-mass confidence, the active skill, the
/// recon-coherence E (‖r−BBᵀr‖²/‖φ‖², low = coherent with the skill's
/// subspace), and a ▸ marker where the hysteresis router crossed a domain
/// boundary. With `json`, each row is also emitted as JSONL on stderr.
fn render_trace(traces: &[cortiq_engine::TokenTrace], pipeline: &Pipeline, json: bool) {
    if traces.is_empty() {
        return;
    }
    let has_routing = traces.iter().any(|t| t.active_skill.is_some() || t.recon.is_some());
    println!("\n\x1b[1mtrace ({} tokens):\x1b[0m", traces.len());
    if has_routing {
        println!("  {:>4}  {:<12}  {:>5}  {:<10}  {:>7}", "#", "token", "conf", "skill", "E");
    } else {
        println!("  {:>4}  {:<12}  {:>5}", "#", "token", "conf");
    }
    for tr in traces {
        let piece = pipeline.tokenizer.decode_token(tr.token_id);
        let shown: String = piece.chars().take(12).collect::<String>().replace('\n', "⏎");
        let conf = conf_colour(&format!("{:>4.0}%", tr.confidence * 100.0), tr.confidence);
        if has_routing {
            let skill = tr.active_skill.as_deref().unwrap_or("—");
            let sw = if tr.switched { " ▸" } else { "" };
            let e = tr.recon.map(|e| format!("{e:.4}")).unwrap_or_else(|| "—".into());
            println!("  {:>4}  {:<12}  {}  {:<10}  {:>7}{sw}", tr.t, shown, conf, skill, e);
        } else {
            println!("  {:>4}  {:<12}  {}", tr.t, shown, conf);
        }
        if json {
            let sk = tr.active_skill.as_deref().map(|s| format!("\"{s}\"")).unwrap_or_else(|| "null".into());
            let rc = tr.recon.map(|e| format!("{e:.6}")).unwrap_or_else(|| "null".into());
            eprintln!(
                "{{\"t\":{},\"token_id\":{},\"confidence\":{:.6},\"active_skill\":{},\"recon\":{},\"switched\":{}}}",
                tr.t, tr.token_id, tr.confidence, sk, rc, tr.switched
            );
        }
    }
}

/// Whether `run` renders the chat template. The file decides: no template
/// (base model) → completion, never the hardcoded ChatML fallback. `--raw`
/// asks for completion outright, and `--state` carries a RAW frozen prefix
/// (cmd_freeze encodes the context verbatim), so templating on top of it
/// would strand a BOS mid-sequence and break bit-identical replay (B2).
fn chat_mode(has_template: bool, raw: bool, resuming: bool) -> bool {
    has_template && !raw && !resuming
}

#[allow(clippy::too_many_arguments)]
async fn cmd_run(
    model_path: &str,
    task: &str,
    prompt: Option<&str>,
    max_tokens: usize,
    skill: Option<&str>,
    greedy: bool,
    raw: bool,
    no_think: bool,
    blend: Option<&str>,
    route_dynamic: bool,
    confidence: bool,
    trace: bool,
    trace_json: bool,
    state: Option<&str>,
    o1: &O1Flags,
) -> anyhow::Result<()> {
    println!("Loading model: {}", model_path);
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let mut skill = skill.map(str::to_string);

    // B2: resume a frozen session — replay its token prefix, carry its
    // skill/seed. The fingerprint guards against a wrong-model resume.
    let mut resume_prefix: Vec<u32> = Vec::new();
    let mut resume_seed: Option<u64> = None;
    if let Some(sp) = state {
        let st = SessionState::read(sp)?;
        if st.kind != STATE_KIND_LOGICAL {
            anyhow::bail!("unsupported .cmfstate kind {} (this build reads logical only)", st.kind);
        }
        if st.fp != SessionState::fingerprint(model.arch()) {
            anyhow::bail!(
                "state was frozen from a different model (fingerprint {:?} ≠ {:?})",
                st.fp, SessionState::fingerprint(model.arch())
            );
        }
        if skill.is_none() {
            skill = st.skill.clone();
        }
        resume_seed = st.seed;
        resume_prefix = st.tokens;
        println!(
            "resume: {} frozen tokens, skill {}",
            resume_prefix.len(),
            skill.as_deref().unwrap_or("—")
        );
    }
    if skill.as_deref() == Some("auto") {
        let mut probe = Pipeline::from_model(&model, SamplerConfig::default())?;
        let ids = probe.tokenizer.encode(prompt.unwrap_or(""));
        let routes = cortiq_engine::router::route(&model, &mut probe, &ids);
        skill = routes.first().map(|r| r.id.clone());
        println!("routed to skill: {}", skill.as_deref().unwrap_or("<none>"));
    }
    let mut sampler = SamplerConfig::default();
    if greedy {
        sampler.temperature = 0.0;
        sampler.repetition_penalty = 1.0;
    }
    if let Some(s) = resume_seed {
        sampler.seed = Some(s); // deterministic continuation of the frozen session
    }
    let mut pipeline = match blend {
        Some("auto") => {
            let b = auto_blend(&model, prompt.unwrap_or(""))?;
            println!("blend auto: {b:?}");
            Pipeline::from_model_with_blend(&model, sampler, &b)?
        }
        Some(spec) => {
            let b = parse_blend(spec)?;
            println!("blend: {b:?}");
            Pipeline::from_model_with_blend(&model, sampler, &b)?
        }
        None => Pipeline::from_model_with_skill(&model, sampler, skill.as_deref())?,
    };
    o1.apply(&mut pipeline);
    if route_dynamic {
        if skill.is_some() || blend.is_some() {
            println!(
                "note: --route-dynamic overrides --skill/--blend (routing starts from backbone)"
            );
        }
        let n = pipeline.enable_dynamic_routing();
        if n == 0 {
            println!("route dynamic: no routable skills in this container — running backbone");
        } else {
            println!("route dynamic: {n} skills, hysteresis on (φ EMA at router layer)");
        }
    }
    if trace {
        pipeline.set_trace(true);
    }
    let runtime = CortiqRuntime::new(model);

    if runtime.masks().get(task).is_some() {
        let _ = runtime.switch_task(task).await;
    }
    let mask = runtime.active_mask().await;

    let status = runtime.status().await;
    println!(
        "Ready: {} | Task: {} | Sparsity: {:.0}%",
        status.model_name,
        status.active_task,
        status.active_sparsity * 100.0
    );

    // The FILE decides chat behaviour (spec §6.1): a container that carries a
    // template is chatted with, one that doesn't is completed. Gate on the
    // template itself — apply_chat_template_opts() falls back to hardcoded
    // ChatML when there is none, which is NOT what a base model wants.
    let has_tpl = pipeline.tokenizer.chat_template.is_some();
    let use_template = chat_mode(has_tpl, raw, state.is_some());
    // `None` leaves enable_thinking undefined → the template's own default,
    // exactly as the server does (openai.rs: apply_chat_template_opts).
    let thinking: Option<bool> = if no_think { Some(false) } else { None };
    if state.is_some() && has_tpl && !raw {
        eprintln!("note: --state resumes a raw token prefix — chat template not applied");
    }
    if !has_tpl && !raw {
        tracing::info!("no chat template in this container — running completion mode");
    }

    let generate_and_print = |pipeline: &mut Pipeline,
                              ids: &[u32]|
     -> anyhow::Result<Option<String>> {
        use std::io::Write;
        // Stream silently when the confidence view will reprint coloured;
        // otherwise stream live as before.
        let cb: cortiq_engine::TokenCallback = if confidence {
            Box::new(|_tok: &str| true)
        } else {
            Box::new(|tok: &str| {
                print!("{tok}");
                let _ = std::io::stdout().flush();
                true
            })
        };
        let started = std::time::Instant::now();
        match pipeline.generate_from_ids(ids, max_tokens, mask.as_ref(), Some(cb)) {
            Ok(r) => {
                let secs = started.elapsed().as_secs_f64();
                // Confidence view: reprint token-by-token, coloured by the
                // model's Born mass on each emitted token.
                if confidence && !r.token_confidence.is_empty() {
                    print!("\n");
                    let mut lo = 1.0f32;
                    let mut sum = 0.0f32;
                    for (id, &c) in r.token_ids.iter().zip(&r.token_confidence) {
                        let piece = pipeline.tokenizer.decode_token(*id);
                        print!("{}", conf_colour(&piece, c));
                        lo = lo.min(c);
                        sum += c;
                    }
                    let _ = std::io::stdout().flush();
                    let avg = sum / r.token_confidence.len() as f32;
                    println!(
                        "\n\nconfidence: mean {:.0}% · min {:.0}%  \
                         (\x1b[38;2;80;220;100mknow\x1b[0m→\
                         \x1b[38;2;230;90;80mguess\x1b[0m)",
                        avg * 100.0,
                        lo * 100.0
                    );
                }
                println!(
                    "\n[{} tokens, {:.1} tok/s, finish: {}]",
                    r.tokens_generated,
                    r.tokens_generated as f64 / secs.max(1e-9),
                    r.finish_reason
                );
                let sw = pipeline.route_switches();
                if !sw.is_empty() {
                    println!("route: {} skill switch(es):", sw.len());
                    for (tok, from, to) in &sw {
                        println!(
                            "  @tok{tok}: {} → {}",
                            from.as_deref().unwrap_or("backbone"),
                            to.as_deref().unwrap_or("backbone")
                        );
                    }
                }
                if trace {
                    render_trace(&r.traces, pipeline, trace_json);
                }
                // `text` is the generated slice only (prompt excluded,
                // specials stripped) — exactly the assistant turn to carry
                // into the next render.
                return Ok(Some(r.text));
            }
            Err(e) => println!("error: {e}"),
        }
        Ok(None)
    };

    // B2: prepend the frozen prefix (empty when not resuming) so the
    // continuation runs from the warm context. Token-level replay ==
    // generate() on the concatenated ids.
    let build_ids = |pipeline: &Pipeline, history: &[(String, String)], text: &str| -> Vec<u32> {
        // An empty prompt stays empty: generate_from_ids answers it with
        // "empty prompt: nothing to generate from" as it does today. The
        // template would otherwise render its boilerplate and generate.
        if use_template && !text.is_empty() {
            pipeline.tokenizer.apply_chat_template_opts(history, thinking)
        } else {
            let mut ids = resume_prefix.clone();
            ids.extend(pipeline.tokenizer.encode(text));
            ids
        }
    };

    if let Some(p) = prompt {
        println!("\nPrompt: {p}\n");
        let history = vec![("user".to_string(), p.to_string())];
        let ids = build_ids(&pipeline, &history, p);
        generate_and_print(&mut pipeline, &ids)?;
    } else {
        println!("\nType your message (Ctrl+C to exit):\n");
        let stdin = std::io::stdin();
        let mut input = String::new();
        let mut history: Vec<(String, String)> = Vec::new();
        loop {
            print!("> ");
            use std::io::Write;
            std::io::stdout().flush()?;
            input.clear();
            if stdin.read_line(&mut input)? == 0 {
                break;
            }
            let text = input.trim();
            if text.is_empty() {
                continue;
            }
            history.push(("user".to_string(), text.to_string()));
            let mut ids = build_ids(&pipeline, &history, text);
            // The cache is cleared per turn and the prefill loop has no
            // length check (eviction only fires while decoding), so a long
            // chat would prefill past the RoPE range. Drop the oldest
            // exchanges — never a system turn — leaving room to decode.
            let budget = pipeline.kv_cache.max_seq_len / 2;
            while use_template && ids.len() > budget && history.len() > 1 {
                let Some(i) = history.iter().position(|(r, _)| r != "system") else {
                    break;
                };
                history.remove(i);
                // Drop the reply with its question, keeping user-first order.
                if history.get(i).is_some_and(|(r, _)| r == "assistant") {
                    history.remove(i);
                }
                eprintln!("note: context full — dropped the oldest exchange");
                ids = build_ids(&pipeline, &history, text);
            }
            // The terminal already echoed the user's line after "> ".
            if use_template {
                println!();
            }
            match generate_and_print(&mut pipeline, &ids)? {
                Some(reply) => history.push(("assistant".to_string(), reply)),
                // A failed turn leaves no dangling user message.
                None => {
                    history.pop();
                }
            }
            println!();
        }
    }

    dump_moe_stats(&pipeline)?;
    Ok(())
}

async fn cmd_info(model_path: &str) -> anyhow::Result<()> {
    let model = CmfModel::open_sharded(model_path)?;
    let arch = model.arch();

    let full = arch
        .layer_types
        .iter()
        .filter(|t| matches!(t, cortiq_core::LayerType::FullAttention))
        .count();
    println!("Model: {}", model_path);
    println!("  Format:      CMF v{}", model.header.version);
    println!("  Arch:        {}", arch.arch_name);
    println!(
        "  Layers:      {} ({} full / {} linear)",
        arch.num_layers,
        full,
        arch.num_layers - full
    );
    println!("  Hidden:      {}", arch.hidden_size);
    println!("  FFN:         {}", arch.intermediate_size);
    println!("  Heads:       {} (KV: {})", arch.num_attention_heads, arch.num_kv_heads);
    println!("  Vocab:       {}", arch.vocab_size);
    println!("  Quant:       {:?} (default; per-tensor in directory)", model.header.quant_type);
    println!("  Tensors:     {}", model.tensors.len());
    println!("  Params:      {:.2}B", model.total_param_count() as f64 / 1e9);
    println!("  Masks:       {}", model.masks.masks.len());
    println!(
        "  Tokenizer:   {}",
        if model.vocab.is_some() { "embedded" } else { "sidecar required" }
    );
    println!(
        "  MTP:         {}",
        match &arch.mtp {
            Some(m) => format!("{} block(s), shared embed+lm_head", m.num_layers),
            None => "—".to_string(),
        }
    );
    println!("  Sparse idx:  {} entries", model.sparse_index.len());

    Ok(())
}

/// The file's verifiable autobiography — narrated from its own header
/// (spec §2/§9) and directory. Everything here is IN the file; nothing
/// is inferred. "Opening someone else's .cmf, I am no longer blind."
fn cmd_story(model_path: &str) -> anyhow::Result<()> {
    let model = CmfModel::open_sharded(model_path)?;
    let arch = model.arch();
    let prov = model.header.provenance.as_ref();
    let sect = "─".repeat(60);

    // ── Who I am ──
    let full = arch
        .layer_types
        .iter()
        .filter(|t| matches!(t, cortiq_core::LayerType::FullAttention))
        .count();
    let linear = arch.num_layers - full;
    let body = match (&arch.moe, linear, full) {
        (Some(m), l, f) if l > 0 && f > 0 => format!(
            "hybrid: {l} linear + {f} full-attention layers, MoE ({} experts, top-{})",
            m.num_experts, m.top_k
        ),
        (Some(m), _, _) => format!("MoE transformer ({} experts, top-{})", m.num_experts, m.top_k),
        (None, l, f) if l > 0 && f > 0 => format!("hybrid: {l} linear + {f} full-attention layers"),
        _ => "dense transformer".to_string(),
    };
    println!("\n\x1b[1m📖 Model story: {model_path}\x1b[0m");
    println!("{sect}");
    println!(
        "I am \x1b[1m{}\x1b[0m, a {} with {:.2} billion parameters.",
        arch.arch_name,
        body,
        model.total_param_count() as f64 / 1e9
    );
    println!(
        "Body: {} layers, hidden {}, {} attention heads, vocab {}.",
        arch.num_layers, arch.hidden_size, arch.num_attention_heads, arch.vocab_size
    );

    // ── Where I come from ──
    if let Some(p) = prov {
        if let Some(src) = p.get("source_model").and_then(|v| v.as_str()) {
            let tool = p.get("tool").and_then(|v| v.as_str()).unwrap_or("?");
            println!("\nDescended from \x1b[1m{src}\x1b[0m — built by the tool {tool}.");
        }
        if let Some(lf) = p.get("linear_fold") {
            if let Some(from) = lf.get("from").and_then(|v| v.as_str()) {
                let thq = lf
                    .get("thq/thk")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                println!(
                    "My linear layers are folded from {from} onto the canonical vmf_phase core; \
                     thq/thk phase: {thq}."
                );
            } else if let Some(carried) = lf.get("carried").and_then(|v| v.as_str()) {
                println!("My linear core: {carried}.");
            }
        }
    }
    if let Some(lc) = &arch.linear_core {
        let extra = lc.nphase.map(|n| format!(", {n} phases/head")).unwrap_or_default();
        println!(
            "Linear attention = «{}» ({} heads{extra}).",
            lc.kind, lc.num_heads
        );
    }

    // ── What the body is made of (dtype histogram) ──
    let mut dtypes: std::collections::BTreeMap<String, (usize, u64)> = Default::default();
    for t in &model.tensors {
        let e = dtypes.entry(format!("{:?}", t.dtype)).or_default();
        e.0 += 1;
        e.1 += t.nbytes;
    }
    print!("\nBody assembled from {} tensors: ", model.tensors.len());
    let parts: Vec<String> = dtypes
        .iter()
        .map(|(d, (n, b))| format!("{d} ×{n} ({:.1} GB)", *b as f64 / 1e9))
        .collect();
    println!("{}.", parts.join(", "));
    if let Some(mtp) = &arch.mtp {
        println!("I carry an MTP head ({} block) — I can speculatively speed up.", mtp.num_layers);
    }

    // ── Which skills I carry (swarm) ──
    if !model.header.skills.is_empty() {
        let total: u64 = model.tensors.iter().map(|t| t.nbytes).sum();
        println!("\n\x1b[1mMy skill swarm ({}):\x1b[0m", model.header.skills.len());
        for sk in &model.header.skills {
            let sbytes: u64 = model.skill_tensors(&sk.id).map(|t| t.nbytes).sum();
            let name = sk.name.as_deref().unwrap_or(&sk.id);
            print!(
                "  • \x1b[1m{}\x1b[0m ({name}) — {:.0} MB, {:.1}% of the file, layers {:?}",
                sk.id,
                sbytes as f64 / 1e6,
                sbytes as f64 / total as f64 * 100.0,
                sk.layers
            );
            // Honest quality contract (claim 16).
            if let Some(q) = &sk.quality {
                let g = |k: &str| q.get(k).and_then(|v| v.as_f64());
                if let (Some(bb), Some(ov)) = (g("backbone"), g("overlaid").or(g("masked"))) {
                    let d = (ov - bb) / bb * 100.0;
                    print!("  | {} {bb:.2}→{ov:.2} ({d:+.1}%)",
                           q.get("metric").and_then(|v| v.as_str()).unwrap_or("quality"));
                }
            } else {
                print!("  | quality NOT measured");
            }
            println!();
        }
        println!(
            "Skill selection is by signal physics (recon-argmin), not by name; \
             storage = backbone + Σ deltas, not K copies."
        );
    }

    // ── How I speak ──
    print!("\nI speak: ");
    if model.vocab.is_some() {
        print!("tokenizer is embedded in the file (self-contained)");
    } else {
        print!("a sidecar tokenizer.json is required");
    }
    if let Some(tc) = &model.header.tokenizer_config {
        let n = tc.chat_template.as_deref().map(str::len).unwrap_or(0);
        if n > 0 {
            print!("; chat template {n} chars, {} stop token(s)", tc.eos_token_ids.len());
        }
    }
    println!(".");

    // ── How honest my confidence is (B1) ──
    if let Some(cal) = &model.header.calibration {
        print!("\nMy confidence is calibrated (T={:.2}", cal.temperature);
        if let (Some(a), Some(b)) = (cal.ece_before, cal.ece_after) {
            print!(", ECE {a:.3}→{b:.3}");
        }
        println!("): I show Born mass as a measured property, not a raw estimate.");
    }

    // ── Am I part of a whole ──
    if let Some(sh) = &model.header.shard {
        println!("I am shard {} of {} (the full model is in the neighbors).", sh.no, sh.count);
    }

    // ── Am I intact (verifiable) ──
    print!("\nIntegrity: ");
    let problems = model.verify();
    if problems.is_empty() {
        println!("\x1b[38;2;80;220;100mall {} hashes matched — I am not corrupted or tampered with.\x1b[0m",
                 model.tensors.len());
    } else {
        println!("\x1b[38;2;230;90;80m{} problem(s) — the file is corrupted:\x1b[0m", problems.len());
        for p in problems.iter().take(5) {
            println!("  ✗ {p}");
        }
    }
    println!("{sect}");
    Ok(())
}

/// Semantic diff of two .cmf files. Identity of a tensor = its name +
/// its `hash64` (spec §3): same name & same hash ⇒ verbatim-identical
/// bytes (the same primitive that makes cross-format dedup free). So the
/// diff is exact and grounded — no dequant, no ML claim. This is the
/// "compare two versions" half of skill-algebra (B3): it answers *what*
/// changed; `merge` (composing δ's) is not shipped because δ-arithmetic
/// composition is not yet demonstrated on measured skills.
fn cmd_diff(a_path: &str, b_path: &str) -> anyhow::Result<()> {
    let a = CmfModel::open_sharded(a_path)?;
    let b = CmfModel::open_sharded(b_path)?;
    let sect = "─".repeat(60);
    println!("\n\x1b[1mCMF diff\x1b[0m  \x1b[2m(a)\x1b[0m {a_path}  →  \x1b[2m(b)\x1b[0m {b_path}");
    println!("{sect}");

    // ── Header / arch ──
    let (aa, ba) = (a.arch(), b.arch());
    let mut hdr = Vec::new();
    let mut row = |label: &str, x: String, y: String| {
        if x != y {
            hdr.push(format!("  {label:<12} {x}  →  {y}"));
        }
    };
    row("format", format!("v{}", a.header.version), format!("v{}", b.header.version));
    row("arch", aa.arch_name.clone(), ba.arch_name.clone());
    row("layers", aa.num_layers.to_string(), ba.num_layers.to_string());
    row("hidden", aa.hidden_size.to_string(), ba.hidden_size.to_string());
    row("ffn", aa.intermediate_size.to_string(), ba.intermediate_size.to_string());
    row("vocab", aa.vocab_size.to_string(), ba.vocab_size.to_string());
    row("quant", format!("{:?}", a.header.quant_type), format!("{:?}", b.header.quant_type));
    row("params", format!("{:.3}B", a.total_param_count() as f64 / 1e9),
        format!("{:.3}B", b.total_param_count() as f64 / 1e9));
    if hdr.is_empty() {
        println!("Header/arch: identical.");
    } else {
        println!("Header/arch changed:");
        for h in &hdr {
            println!("{h}");
        }
    }

    // ── Tensors (identity = name + hash64) ──
    use std::collections::BTreeMap;
    let map = |m: &CmfModel| -> BTreeMap<String, (u64, String, u64)> {
        m.tensors
            .iter()
            .map(|t| (t.name.clone(), (t.hash, format!("{:?}", t.dtype), t.nbytes)))
            .collect()
    };
    let (ma, mb) = (map(&a), map(&b));
    let (mut added, mut removed, mut changed, mut same) = (Vec::new(), Vec::new(), Vec::new(), 0u64);
    for (name, (hb, db, nb)) in &mb {
        match ma.get(name) {
            None => added.push((name.clone(), db.clone(), *nb)),
            Some((ha, da, na)) => {
                if ha == hb {
                    same += 1;
                } else {
                    changed.push((name.clone(), da.clone(), db.clone(), *na, *nb));
                }
            }
        }
    }
    for (name, (_, da, na)) in &ma {
        if !mb.contains_key(name) {
            removed.push((name.clone(), da.clone(), *na));
        }
    }
    println!(
        "\nTensors: {} shared verbatim (hash matched), \
         \x1b[38;2;80;220;100m+{} new\x1b[0m, \
         \x1b[38;2;230;90;80m−{} removed\x1b[0m, \
         \x1b[38;2;230;190;80m~{} changed\x1b[0m.",
        same, added.len(), removed.len(), changed.len()
    );
    let show = |title: &str, rows: &[String]| {
        if rows.is_empty() {
            return;
        }
        println!("{title}");
        for r in rows.iter().take(20) {
            println!("    {r}");
        }
        if rows.len() > 20 {
            println!("    … {} more", rows.len() - 20);
        }
    };
    show(
        "  \x1b[38;2;80;220;100m+ new:\x1b[0m",
        &added.iter().map(|(n, d, b)| format!("{n}  [{d}, {:.1} MB]", *b as f64 / 1e6)).collect::<Vec<_>>(),
    );
    show(
        "  \x1b[38;2;230;90;80m− removed:\x1b[0m",
        &removed.iter().map(|(n, d, b)| format!("{n}  [{d}, {:.1} MB]", *b as f64 / 1e6)).collect::<Vec<_>>(),
    );
    show(
        "  \x1b[38;2;230;190;80m~ changed:\x1b[0m",
        &changed
            .iter()
            .map(|(n, da, db, na, nb)| {
                let dt = if da == db { da.clone() } else { format!("{da}→{db}") };
                let sz = if na == nb {
                    format!("{:.1} MB", *nb as f64 / 1e6)
                } else {
                    format!("{:.1}→{:.1} MB", *na as f64 / 1e6, *nb as f64 / 1e6)
                };
                format!("{n}  [{dt}, {sz}]")
            })
            .collect::<Vec<_>>(),
    );

    // ── Skills (swarm, Patent 15) ──
    let sid = |m: &CmfModel| -> BTreeMap<String, Vec<usize>> {
        m.header.skills.iter().map(|s| (s.id.clone(), s.layers.clone())).collect()
    };
    let (sa, sb) = (sid(&a), sid(&b));
    if !sa.is_empty() || !sb.is_empty() {
        let new_sk: Vec<_> = sb.keys().filter(|k| !sa.contains_key(*k)).cloned().collect();
        let del_sk: Vec<_> = sa.keys().filter(|k| !sb.contains_key(*k)).cloned().collect();
        let kept: Vec<_> = sb.keys().filter(|k| sa.contains_key(*k)).cloned().collect();
        print!("\nSwarm: {} shared", kept.len());
        if !new_sk.is_empty() {
            print!(", \x1b[38;2;80;220;100m+[{}]\x1b[0m", new_sk.join(","));
        }
        if !del_sk.is_empty() {
            print!(", \x1b[38;2;230;90;80m−[{}]\x1b[0m", del_sk.join(","));
        }
        println!(".");
    }
    println!("{sect}");
    Ok(())
}

async fn cmd_verify(model_path: &str) -> anyhow::Result<()> {
    println!("Verifying {} ...", model_path);
    // open() already enforces magic/version/features/section bounds.
    // Each shard is a self-contained valid .cmf (spec §10), so
    // verify opens the file as is, without merging neighbors.
    let model = CmfModel::open(model_path)?;
    println!("  ✓ envelope, sections, tensor directory ({} tensors)", model.tensors.len());

    let problems = model.verify();
    if problems.is_empty() {
        println!("  ✓ all tensor hashes match");
        println!("OK");
        Ok(())
    } else {
        for p in &problems {
            println!("  ✗ {}", p);
        }
        anyhow::bail!("{} tensor(s) corrupted", problems.len());
    }
}

async fn cmd_masks(model_path: &str) -> anyhow::Result<()> {
    let model = CmfModel::open_sharded(model_path)?;

    if model.masks.masks.is_empty() {
        println!("No masks in {}", model_path);
        return Ok(());
    }

    println!("Masks in {}:", model_path);
    println!("  {:<15} {:>8} {:>12} {:>6} {:>8}", "Name", "Sparsity", "Quality", "Layers", "Hot");
    println!("  {}", "-".repeat(56));
    for m in &model.masks.masks {
        let quality = match &m.quality {
            Some(q) => format!("{:.3} ({})", q.value, q.metric),
            None => "unmeasured".to_string(),
        };
        println!(
            "  {:<15} {:>7.0}% {:>12} {:>6} {:>8}",
            m.name,
            m.sparsity * 100.0,
            quality,
            m.active_layer_count(),
            if m.has_hot_pack { "hot" } else { "—" }
        );
    }

    Ok(())
}

async fn cmd_bench(
    model_path: &str,
    task: &str,
    tokens: u32,
    ctx: Option<usize>,
    o1: &O1Flags,
    json: bool,
) -> anyhow::Result<()> {
    if !json {
        println!("Benchmark: {} | task={} | tokens={}", model_path, task, tokens);
    }
    if let Some(n) = ctx {
        // Long-context mode must not silently evict mid-measurement:
        // raise the cap to cover prompt + generation unless the user
        // pinned it explicitly.
        if std::env::var("CMF_MAX_SEQ").is_err() {
            // SAFETY: single-threaded here — before any pipeline/pool spawn.
            unsafe { std::env::set_var("CMF_MAX_SEQ", (n + tokens as usize + 64).to_string()) };
        }
    }
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let mut pipeline = Pipeline::from_model(
        &model,
        SamplerConfig {
            temperature: 0.0, // greedy: benchmark must be deterministic
            seed: Some(42),
            ..Default::default()
        },
    )?;
    o1.apply(&mut pipeline);
    if pipeline.o1_active() {
        println!("  O(1):    nystrom attention on (KV replaced on flagged layers)");
    }
    let runtime = CortiqRuntime::new(model);
    if runtime.masks().get(task).is_some() {
        let _ = runtime.switch_task(task).await;
    }
    // "general" benches the dense path (enables MTP speculation);
    // named tasks bench masked sparse execution.
    let mask = if task == "general" {
        None
    } else {
        runtime.active_mask().await
    };

    // Warmup: touch every weight page once so the numbers below are
    // steady-state (a cold 14 GB mmap otherwise bills its first pass
    // to whichever phase runs first).
    let prompt = match ctx {
        // Long-context mode: repeat until the token budget is covered
        // (~9 tokens per sentence), then truncate exactly below.
        Some(n) => "The quick brown fox jumps over the lazy dog. ".repeat(n / 8 + 2),
        None => "The quick brown fox jumps over the lazy dog. ".repeat(4),
    };
    let mut prompt_ids = pipeline.tokenizer.encode(&prompt);
    if let Some(n) = ctx {
        prompt_ids.truncate(n);
        if prompt_ids.len() < n {
            anyhow::bail!("ctx {n}: synthetic prompt tokenized to only {} tokens", prompt_ids.len());
        }
    }
    let _ = pipeline
        .forward_ids(&prompt_ids[..2.min(prompt_ids.len())], mask.as_ref())
        .map_err(|e| anyhow::anyhow!(e))?;

    // Prefill benchmark.
    let t0 = std::time::Instant::now();
    let _ = pipeline
        .forward_ids(&prompt_ids, mask.as_ref())
        .map_err(|e| anyhow::anyhow!(e))?;
    let prefill_s = t0.elapsed().as_secs_f64();

    // Pair-fusion micro-bench: the memory-traffic win MTP verify rides
    // on. Skipped under o1 — forward_pair appends into the (sealed,
    // emptied) cache, so its numbers would be meaningless there.
    let (singles_ms, pair_ms) = if pipeline.o1_active() {
        (0.0, 0.0)
    } else {
        pipeline.measure_pair_fusion(8)
    };

    // Decode benchmark. Steady-state decode speed comes from the
    // inter-token timestamps: generation's own prefill (fused pairs; +
    // the one-off o1 seal) differs from the timed forward_ids prefill
    // above, so deriving decode by subtraction billed that difference
    // to the decode line — wrong for both arms, worst at long ctx.
    // Per-token stamps carry the counter snapshots too: steady-state
    // allocations/token and pool dispatches/token come from the same
    // inter-token deltas as the steady tok/s (roadmap этап 0).
    type Stamp = (std::time::Instant, u64, usize);
    let stamps: Arc<std::sync::Mutex<Vec<Stamp>>> = Arc::default();
    let st = stamps.clone();
    let cb: cortiq_engine::TokenCallback = Box::new(move |_tok| {
        st.lock().unwrap().push((
            std::time::Instant::now(),
            ALLOCS.load(AtomicOrdering::Relaxed),
            cortiq_engine::pool::dispatch_count(),
        ));
        true
    });
    let t1 = std::time::Instant::now();
    let result = pipeline
        .generate_from_ids(&prompt_ids, tokens as usize, mask.as_ref(), Some(cb))
        .map_err(|e| anyhow::anyhow!(e))?;
    let total_s = t1.elapsed().as_secs_f64();

    if !json {
        println!("  Prompt:  {} tokens | prefill {:.1} tok/s", prompt_ids.len(), prompt_ids.len() as f64 / prefill_s.max(1e-9));
    }
    let stamps = stamps.lock().unwrap();
    // stamp[0] fires right after generation's prefill (the first token
    // is sampled from the prefill hidden, no decode forward yet).
    let n_st = stamps.len();
    let decode_tps = if n_st >= 2 {
        (n_st - 1) as f64
            / (stamps[n_st - 1].0 - stamps[0].0).as_secs_f64().max(1e-9)
    } else {
        0.0
    };
    // Steady-state counters over the same inter-token window as tok/s.
    let (allocs_per_token, dispatches_per_token) = if n_st >= 2 {
        let steps = (n_st - 1) as f64;
        (
            (stamps[n_st - 1].1 - stamps[0].1) as f64 / steps,
            (stamps[n_st - 1].2 - stamps[0].2) as f64 / steps,
        )
    } else {
        (0.0, 0.0)
    };
    let ttft_s = stamps
        .first()
        .map(|s| s.0.duration_since(t1).as_secs_f64())
        .unwrap_or(0.0);
    // KV/state residency at the end of the run: full-attention layers
    // grow O(context); the linear core (vmf_phase/GDN) and the nystrom
    // override hold O(1) state — this line is the long-context memory
    // claim, measured.
    let total_mem = pipeline.kv_cache.total_memory_bytes();
    let nystrom_mem: usize = pipeline
        .kv_cache
        .layers
        .iter()
        .map(|l| l.o1_memory_bytes())
        .sum();
    if json {
        // llama-bench-compatible spirit: one flat JSON object, raw
        // numbers only — joinable without parsing human text.
        let obj = serde_json::json!({
            "model": model_path,
            "task": task,
            "ctx": ctx,
            "o1": pipeline.o1_active(),
            "threads_env": std::env::var("CMF_THREADS").ok(),
            "prompt_tokens": prompt_ids.len(),
            "prefill_tok_s": prompt_ids.len() as f64 / prefill_s.max(1e-9),
            "tokens_generated": result.tokens_generated,
            "decode_tok_s_steady": decode_tps,
            "decode_tok_s_incl_prefill": result.tokens_generated as f64 / total_s.max(1e-9),
            "ttft_s": ttft_s,
            "allocs_per_token": allocs_per_token,
            "pool_dispatches_per_token": dispatches_per_token,
            "pair_singles_ms": singles_ms,
            "pair_fused_ms": pair_ms,
            "kv_state_bytes": total_mem,
            "nystrom_state_bytes": nystrom_mem,
            "seq_len": pipeline.kv_cache.seq_len(),
            "mtp_drafted": result.mtp_drafted,
            "mtp_accepted": result.mtp_accepted,
            "finish_reason": result.finish_reason,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }
    println!(
        "  Decode:  {} tokens | {:.1} tok/s steady (TTFT {:.2}s, {:.1} incl. prefill)",
        result.tokens_generated,
        decode_tps,
        ttft_s,
        result.tokens_generated as f64 / total_s.max(1e-9)
    );
    println!(
        "  Steady:  {:.1} allocs/token | {:.1} pool dispatches/token",
        allocs_per_token, dispatches_per_token
    );
    if pair_ms > 0.0 {
        println!(
            "  Pair:    2 singles {:.2} ms vs fused {:.2} ms (×{:.2} cheaper second lane)",
            singles_ms,
            pair_ms,
            singles_ms / pair_ms.max(1e-9)
        );
    }
    if nystrom_mem > 0 {
        println!(
            "  Memory:  KV+state {:.1} MB (exact KV {:.1} MB + nystrom state {:.1} MB) at seq_len {}",
            total_mem as f64 / 1e6,
            (total_mem - nystrom_mem) as f64 / 1e6,
            nystrom_mem as f64 / 1e6,
            pipeline.kv_cache.seq_len()
        );
    } else {
        println!(
            "  Memory:  KV+state {:.1} MB at seq_len {}",
            total_mem as f64 / 1e6,
            pipeline.kv_cache.seq_len()
        );
    }
    if result.mtp_drafted > 0 {
        println!(
            "  MTP:     drafted {} | accepted {} ({:.0}%)",
            result.mtp_drafted,
            result.mtp_accepted,
            result.mtp_accepted as f64 / result.mtp_drafted as f64 * 100.0
        );
    }
    println!("  Finish:  {}", result.finish_reason);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_state_roundtrips() {
        let st = SessionState {
            kind: STATE_KIND_LOGICAL,
            fp: (24, 1024, 248320),
            seed: Some(42),
            skill: Some("ru".to_string()),
            tokens: vec![1, 5, 9, 100000, 3],
        };
        let dir = std::env::temp_dir();
        let path = dir.join(format!("cmfstate-test-{}.bin", std::process::id()));
        let p = path.to_str().unwrap();
        st.write(p).unwrap();
        let back = SessionState::read(p).unwrap();
        std::fs::remove_file(p).ok();
        assert_eq!(back.kind, st.kind);
        assert_eq!(back.fp, st.fp);
        assert_eq!(back.seed, st.seed);
        assert_eq!(back.skill, st.skill);
        assert_eq!(back.tokens, st.tokens);

        // None-skill / None-seed also round-trip.
        let st2 = SessionState { kind: 0, fp: (1, 2, 3), seed: None, skill: None, tokens: vec![7] };
        st2.write(p).unwrap();
        let b2 = SessionState::read(p).unwrap();
        std::fs::remove_file(p).ok();
        assert!(b2.seed.is_none() && b2.skill.is_none() && b2.tokens == vec![7]);
    }

    #[test]
    fn chat_mode_lets_the_file_decide() {
        // A container with a template is chatted with — the new default.
        assert!(chat_mode(true, false, false));
        // --raw opts out; a base model has nothing to opt out of.
        assert!(!chat_mode(true, true, false));
        assert!(!chat_mode(false, false, false));
        assert!(!chat_mode(false, true, false));
        // --state replays a raw prefix: raw whatever the file carries (B2).
        assert!(!chat_mode(true, false, true));
        assert!(!chat_mode(false, false, true));
    }

    #[test]
    fn session_state_rejects_bad_magic() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("cmfstate-badmagic-{}.bin", std::process::id()));
        let p = path.to_str().unwrap();
        std::fs::write(p, b"NOPEnot a state file").unwrap();
        let r = SessionState::read(p);
        std::fs::remove_file(p).ok();
        assert!(r.is_err(), "bad magic must be rejected");
    }
}
