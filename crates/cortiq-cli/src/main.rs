//! Cortiq CLI — sparse task-routed model inference.

mod convert;
mod gguf;

use clap::{Parser, Subcommand};
use cortiq_core::CmfModel;
use cortiq_engine::{CortiqRuntime, Pipeline, SamplerConfig};
use cortiq_server::{build_router, AppState};
use std::sync::Arc;
use tokio::sync::Mutex;

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
    /// Interactive chat mode
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            model,
            port,
            host,
            task,
            compat_port,
        } => cmd_serve(&model, &host, port, &task, compat_port).await,
        Commands::Convert { model, quant, output, hf_token } => {
            convert::run_convert(&model, &quant, &output, hf_token.as_deref(), |f| {
                println!("@PROGRESS {f:.4}");
            })?;
            println!("✓ wrote {output}");
            Ok(())
        }
        Commands::ImportGguf { gguf, output, quant, hf_token } => {
            gguf::run_import_gguf(&gguf, &quant, &output, hf_token.as_deref(), |f| {
                println!("@PROGRESS {f:.4}");
            })?;
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
            blend,
            route_dynamic,
            confidence,
            trace,
            trace_json,
            state,
        } => {
            cmd_run(&model, &task, prompt.as_deref(), max_tokens, skill.as_deref(), greedy,
                    blend.as_deref(), route_dynamic, confidence, trace, trace_json,
                    state.as_deref())
            .await
        }
        Commands::Freeze { model, prompt, out, skill } => {
            cmd_freeze(&model, &prompt, &out, skill.as_deref())
        }
        Commands::Route { model, prompt } => cmd_route(&model, &prompt),
        Commands::Ppl { model, file, tokens, skill, blend, route_dynamic } => {
            cmd_ppl(&model, &file, tokens, skill.as_deref(), blend.as_deref(), route_dynamic)
        }
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
        } => cmd_bench(&model, &task, tokens).await,
        Commands::Verify { model } => cmd_verify(&model).await,
    }
}

async fn cmd_serve(
    model_path: &str,
    host: &str,
    port: u16,
    default_task: &str,
    _compat_port: Option<u16>,
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

    let pipeline = Pipeline::from_model(&model, SamplerConfig::default())?;
    println!("    Pipeline: loaded ({:.2}B params)", model.total_param_count() as f64 / 1e9);
    println!();

    // Create runtime
    let runtime = CortiqRuntime::new(model);
    if runtime.masks().get(default_task).is_some() {
        let _ = runtime.switch_task(default_task).await;
    }
    let state = Arc::new(AppState {
        runtime,
        pipeline: Mutex::new(pipeline),
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

fn cmd_ppl(
    model_path: &str,
    file: &str,
    max_tokens: usize,
    skill: Option<&str>,
    blend: Option<&str>,
    route_dynamic: bool,
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
    let mut ids = pipeline.tokenizer.with_bos(pipeline.tokenizer.encode(&text));
    ids.truncate(max_tokens);
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

#[allow(clippy::too_many_arguments)]
async fn cmd_run(
    model_path: &str,
    task: &str,
    prompt: Option<&str>,
    max_tokens: usize,
    skill: Option<&str>,
    greedy: bool,
    blend: Option<&str>,
    route_dynamic: bool,
    confidence: bool,
    trace: bool,
    trace_json: bool,
    state: Option<&str>,
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

    let generate_and_print = |pipeline: &mut Pipeline, text: &str| -> anyhow::Result<()> {
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
        // B2: prepend the frozen prefix (empty when not resuming) so the
        // continuation runs from the warm context. Token-level replay ==
        // generate() on the concatenated ids.
        let mut ids = resume_prefix.clone();
        ids.extend(pipeline.tokenizer.encode(text));
        let outcome = if resume_prefix.is_empty() {
            pipeline.generate(text, max_tokens, mask.as_ref(), Some(cb))
        } else {
            pipeline.generate_from_ids(&ids, max_tokens, mask.as_ref(), Some(cb))
        };
        match outcome {
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
            }
            Err(e) => println!("error: {e}"),
        }
        Ok(())
    };

    if let Some(p) = prompt {
        println!("\nPrompt: {p}\n");
        generate_and_print(&mut pipeline, p)?;
    } else {
        println!("\nType your message (Ctrl+C to exit):\n");
        let stdin = std::io::stdin();
        let mut input = String::new();
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
            generate_and_print(&mut pipeline, text)?;
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
) -> anyhow::Result<()> {
    println!("Benchmark: {} | task={} | tokens={}", model_path, task, tokens);
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let mut pipeline = Pipeline::from_model(
        &model,
        SamplerConfig {
            temperature: 0.0, // greedy: benchmark must be deterministic
            seed: Some(42),
            ..Default::default()
        },
    )?;
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
    let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(4);
    let prompt_ids = pipeline.tokenizer.encode(&prompt);
    let _ = pipeline
        .forward_ids(&prompt_ids[..2.min(prompt_ids.len())], mask.as_ref())
        .map_err(|e| anyhow::anyhow!(e))?;

    // Prefill benchmark.
    let t0 = std::time::Instant::now();
    let _ = pipeline
        .forward_ids(&prompt_ids, mask.as_ref())
        .map_err(|e| anyhow::anyhow!(e))?;
    let prefill_s = t0.elapsed().as_secs_f64();

    // Pair-fusion micro-bench: the memory-traffic win MTP verify rides on.
    let (singles_ms, pair_ms) = pipeline.measure_pair_fusion(8);

    // Decode benchmark.
    let t1 = std::time::Instant::now();
    let result = pipeline
        .generate_from_ids(&prompt_ids, tokens as usize, mask.as_ref(), None)
        .map_err(|e| anyhow::anyhow!(e))?;
    let total_s = t1.elapsed().as_secs_f64();

    println!("  Prompt:  {} tokens | prefill {:.1} tok/s", prompt_ids.len(), prompt_ids.len() as f64 / prefill_s.max(1e-9));
    let decode_s = (total_s - prefill_s).max(1e-9);
    println!(
        "  Decode:  {} tokens | {:.1} tok/s pure ({:.1} incl. prefill)",
        result.tokens_generated,
        result.tokens_generated as f64 / decode_s,
        result.tokens_generated as f64 / total_s.max(1e-9)
    );
    println!(
        "  Pair:    2 singles {:.2} ms vs fused {:.2} ms (×{:.2} cheaper second lane)",
        singles_ms,
        pair_ms,
        singles_ms / pair_ms.max(1e-9)
    );
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
