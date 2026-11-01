//! Cortiq CLI — sparse task-routed model inference.
#![recursion_limit = "256"]

mod avout;
mod awnp;
mod choice;
mod convert;
mod decision;
mod dialog;
mod egemma2pack;
mod nemotron_pack;
mod embed;
mod flint;
mod flint_hybrid;
mod gguf;
mod gptq;
#[cfg(feature = "gpu")]
mod nvgtrain;
#[cfg(feature = "gpu")]
mod moegptq;
#[cfg(feature = "gpu")]
mod nemotron_gptq;
mod growth;
mod http_range;
mod imagepack;
mod primalvaepack;
mod vaecmd;
mod knowledge;
mod lookup;
mod ltxcmd;
mod ltxpack;
mod mimo_towers;
mod minimax_m3;
mod moedefrag;
mod music;
mod mossimagepack;
mod npy;
#[cfg(feature = "phonon")]
mod phononpack;
mod nvg;
mod nvg_linalg;
mod nvgbench;
#[cfg(feature = "gpu")]
mod nvg_gpu;
mod qwen_imagepack;
mod recall;
mod requant;
mod sign;
mod skill;
mod transcribe;
mod tube;
mod mtpcompact;
mod utility;
mod videopack;
mod zimagepack;
mod qi21pack;
mod sam3pack;
mod sam3seg;
mod tdata;
mod teacher_api;
mod teacher_data;
mod compress;

use clap::{Parser, Subcommand};
use cortiq_core::CmfModel;
use cortiq_core::types::TensorDtype;
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
use cortiq_engine::dsv41_encoding::{self, EncodeOptions, ReasoningEffort, ThinkingMode};
use cortiq_engine::dsv41_vision::{self, PreparedVlInputs, VisionConfig};
use cortiq_engine::{CortiqRuntime, Pipeline, SamplerConfig};
use cortiq_server::{AppState, build_router};
use std::sync::Arc;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

// wgpu's Vulkan/GL teardown can emit a driver event from a TLS destructor,
// after tracing's own per-thread context has already been destroyed.  Keep
// the normal subscriber for the command lifetime, then gate the layer off
// before Tokio/wgpu teardown begins so that late driver diagnostics cannot
// turn a successful inference into a TLS panic/abort.
static TRACING_LIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

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
        (
            arch.num_layers as u32,
            arch.hidden_size as u32,
            arch.vocab_size as u32,
        )
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
        Ok(SessionState {
            kind,
            fp,
            seed,
            skill,
            tokens,
        })
    }
}

/// Bundled `--o1*` CLI flags for run/serve/bench. `spec = None` keeps
/// whatever env CMF_O1 / the file's converter hint resolved at load;
/// an explicit spec (including `off`) replaces it.
#[derive(Clone)]
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
            Some(s) => cortiq_engine::nystrom::O1Cfg::parse_rect(s)
                .map(Some)
                .ok_or_else(|| anyhow::anyhow!("--o1-rect '{s}' is not one of: agg | fm")),
        }
    }

    /// The config this flag set resolves to, or None for `off`/absent.
    fn cfg(&self) -> anyhow::Result<Option<cortiq_engine::nystrom::O1Cfg>> {
        let rect = self.rect()?;
        Ok(self.spec.as_deref().and_then(|spec| {
            cortiq_engine::nystrom::O1Cfg::from_spec(spec, self.m, self.w, self.sink, rect)
        }))
    }

    /// Apply the flags; a natively bounded file refuses any `--o1`
    /// (the anchor is the file's operator, not a runtime knob).
    fn apply(&self, pipeline: &mut Pipeline) -> anyhow::Result<()> {
        if let Some(spec) = self.spec.as_deref() {
            let rect = self.rect()?;
            pipeline
                .try_set_o1(cortiq_engine::nystrom::O1Cfg::from_spec(
                    spec, self.m, self.w, self.sink, rect,
                ))
                .map_err(|e| anyhow::anyhow!(e))?;
        }
        Ok(())
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
        anyhow::ensure!(
            stride > 0,
            "{w} windows of {} do not fit in {n} tokens",
            self.window_len
        );
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
    /// Start the inference server with web dashboard (on a decision file:
    /// the decisions API, see `cortiq decision`)
    Serve {
        /// Path to .cmf model file
        model: String,
        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,
        /// Host / interface to bind [default: 0.0.0.0; 127.0.0.1 for a
        /// decision file]
        #[arg(long)]
        host: Option<String>,
        /// Default task mask [default: general]
        #[arg(short, long)]
        task: Option<String>,
        /// Decision file only: server configuration JSON (keys, limits,
        /// prices, oracle, cache, learning)
        #[arg(long)]
        decision_config: Option<String>,
        /// Decision file only: state directory (keys, usage and oracle
        /// ledgers, generations) [default: <MODEL>.state]
        #[arg(long)]
        state: Option<String>,
        /// Decision file only: remove a state LOCK left by a dead process
        /// where the state directory's filesystem has no advisory locks
        /// (elsewhere such a LOCK is taken over without it)
        #[arg(long)]
        break_lock: bool,
        /// Decision file only: expose the TypeSafe/Jev System One request
        /// adapter at POST /v1/systemone. The response always identifies the
        /// local CMF model, never Jev.
        #[arg(long)]
        jev_compatible: bool,
        /// Decision file only: shadow mode for switching production traffic
        /// from cortiq-router (router-API paths only). Every router-API
        /// request (/v1/route, /v1/route:batch, /v1/feedback,
        /// /v1/taxonomies, /v1/usage, /v1/escalations, /v1/healthz,
        /// /v1/readyz, /metrics, /v1/admin/keys) is forwarded unchanged to
        /// the old router at URL (https; http only to a loopback address),
        /// with the client's Authorization header, and the client gets the
        /// old router's answer byte for byte, errors included (502
        /// UPSTREAM_UNAVAILABLE only when it gives no answer). A /v1/route or
        /// /v1/route:batch the old router answered 200 is also decided
        /// locally after that answer — no oracle, no learning, no billing,
        /// 4 comparison slots of its own (64 more may wait; one past that
        /// is dropped) — and one line per input is appended to
        /// <state>/shadow.jsonl: {ts, request_id_old, text_hmac (HMAC-SHA256
        /// under <state>/shadow.key, never the text), taxonomy, old_label,
        /// new_label, agree, old_confident, new_confident, old_latency_ms,
        /// new_latency_ms, old_status, new_error}; other answers are neither
        /// decided nor written. GET /v1/admin/shadow (x-admin-token) returns
        /// the agreement overall, by confidence and per label
        #[arg(long, value_name = "URL")]
        shadow_of: Option<String>,
        /// Decision file only, with --shadow-of: deadline of one request
        /// forwarded to the old router, connect to last byte, in seconds
        /// (default 60, nginx's proxy_read_timeout; raise it for long
        /// batches the old router escalates input by input). An old router
        /// that does not answer in time gives 502 UPSTREAM_UNAVAILABLE
        #[arg(long, value_name = "SECONDS", requires = "shadow_of")]
        shadow_timeout_s: Option<u64>,
        #[command(flatten)]
        oracle: decision::OracleArgs,
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
        /// Network pipeline-split: address of a `cortiq worker` holding
        /// the tail layers of the SAME .cmf (verified by dir_hash).
        /// Forces a single serve slot — one worker holds one KV session.
        #[arg(long)]
        peer: Option<String>,
        /// First layer the peer/second GPU runs (default: half the stack)
        #[arg(long)]
        peer_split: Option<usize>,
        /// Shared secret for --peer (must match the worker's --token)
        #[arg(long, requires = "peer")]
        net_token: Option<String>,
        /// Wire payload dtype for --peer: f32 = bit-exact, f16 = half the bytes
        #[arg(long, default_value = "f32", requires = "peer")]
        net_dtype: String,
        /// Use N local GPUs. Replicas when the model fits one card —
        /// a full copy per card, N requests decoding at once, the mode
        /// that scales throughput; an in-process layer split when it
        /// does not. The server prints which mode it took and why.
        #[arg(long, conflicts_with = "peer")]
        gpus: Option<usize>,
        /// Router v2: also score quarantined skills in the per-request
        /// decision (gate measurement; debug)
        #[arg(long)]
        route_include_quarantine: bool,
        /// A request routed to a `lookup` record: answer (the table's text
        /// is the assistant message; default) | context (the card is
        /// prepended to the message and the backbone generates) | off
        /// (the table is ignored). Default: CMF_LOOKUP_MODE.
        #[arg(long)]
        lookup_mode: Option<String>,
    },
    /// d1 decision models (Liquid AI's System One on LFM3): named questions
    /// over a state and its pictures, answered from one forward pass as
    /// calibrated distributions; prints `{"answers": ..., "usage": ...}`
    SystemOne {
        /// Path to a d1 .cmf file
        model: String,
        /// The state: text, or `@file` (a `.json` file is read as JSON)
        #[arg(long)]
        state: Option<String>,
        /// Named questions as JSON: `{"refund": {"type": "noul",
        /// "instructions": "Is this a refund request?"}}`, or `@file`
        #[arg(long)]
        questions: String,
        /// Picture path, data URL or HTTP(S) URL (repeatable)
        #[arg(long = "image")]
        images: Vec<String>,
        /// d1-Omni audio clip: WAV path, data URL or HTTP(S) URL (repeatable)
        #[arg(long = "audio")]
        audios: Vec<String>,
    },
    /// Decide with a decision file: one text (-p) or a JSONL batch (--input);
    /// locally, and with --oracle MODEL (the key in OPENROUTER_API_KEY) an
    /// OpenRouter model answers only what the local model cannot decide
    Decide(decision::DecideArgs),
    /// Decision files: build (init, train, add-skill, learn), inspect (info,
    /// verify), generations (materialize, rollback), API keys and the oracle
    /// check (`decision oracle check`)
    Decision {
        #[command(subcommand)]
        cmd: decision::DecisionCmd,
    },
    /// Convert a Hugging Face checkpoint to .cmf — native Rust, no Python
    Convert {
        /// HF model: a local dir (config.json + *.safetensors + tokenizer.json)
        /// or a hub repo id like `Qwen/Qwen2.5-0.5B-Instruct` (downloaded)
        #[arg(long)]
        model: String,
        /// Quantization for 2-D weights: q8 | q8_2f | q4 | q4t | q4tp | q2tp | q1 | q1p | q1s | q1t | f16 | vbit | mxfp4 | fp4 | blr | auto.
        /// Default (or `auto`): q8, except MiMo-V2 (`mimo_v2`), which converts to
        /// q4tp experts with a q8_2f skeleton. `blr` = binary low-rank layer
        /// projections at `--bpw` (encoder `--blr-method`, docs/BLR_NVG.md).
        /// `fp4` = MoE experts as OCP MXFP4 (`mxfp4`, byte-identical to a
        /// DeepSeek-V4/V4.1 FP4 source), the rest as the q4tp profile
        /// (docs/FP4_NATIVE.md); `mxfp4` = every matrix in MXFP4.
        #[arg(long)]
        quant: Option<String>,
        /// Output .cmf path
        #[arg(long)]
        output: String,
        /// Additional profile and output, QUANT=PATH (repeatable). Every
        /// output shares one streamed source pass, avoiding a second download.
        #[arg(long = "also-output", value_name = "QUANT=PATH")]
        also_output: Vec<String>,
        /// Hugging Face token (for gated/private repos)
        #[arg(long)]
        hf_token: Option<String>,
        /// Target mean bits for `--quant vbit` (3.0–8.0; default 4.25). Higher =
        /// better quality + larger file. Precision-sensitive architectures
        /// (e.g. GatedDeltaNet) may need 5.5–6 to stay coherent.
        #[arg(long, default_value = "4.25")]
        mean_bits: f32,
        /// Per-tensor quantization override, PATTERN=QUANT (repeatable; first
        /// match wins; `*` is the only wildcard and spans dots). Applies to
        /// the 2-D weights the profile would quantize, e.g.
        /// `--tensor-quant lm_head.weight=q8_2f` keeps a q4tp file's output
        /// head at 8 bits; `--tensor-quant 'model.layers.*.mlp.down_proj.weight=q8_2f'`.
        #[arg(long = "tensor-quant", value_name = "PATTERN=QUANT")]
        tensor_quant: Vec<String>,
        /// Leave tensors out of the file: comma-separated globs over the
        /// CANONICAL output tensor names (repeatable; `*` is the only
        /// wildcard and spans dots). A source tensor whose canonical name
        /// matches is never decoded or written, in every output
        /// (`--also-output` included); a name only a split or a late rename
        /// produces is dropped at the writer. The source index is checked
        /// up front (a pattern matching nothing is flagged), the count and
        /// source bytes left out are printed, and the patterns are recorded
        /// in the header provenance (`excluded_tensors`); `--resume` must
        /// repeat them. E.g. `--exclude-tensors 'model.mtp.*'` drops a
        /// DeepSeek-V4/V4.1 MTP/DSpark stack (the V4.1 runtime never
        /// executes it): the 1017 `--quant fp4` payload goes from 504.51 to
        /// 496.96 GB (file ≈ 497.2 GB with its header), under Hugging Face's
        /// 500 GB per-file limit.
        #[arg(long = "exclude-tensors", value_name = "PATTERNS")]
        exclude_tensors: Vec<String>,
        /// Continue a conversion that was interrupted. Keeps the payloads
        /// already in <output> and skips every source shard its manifest
        /// records as done — the download included, which is where a
        /// restart otherwise spends its hours. Without the manifest
        /// (<output>.manifest) there is nothing to resume from and the
        /// conversion starts over.
        #[arg(long)]
        resume: bool,
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
        /// MiMo-V2 only: write JUST the multi-token-prediction draft layers
        /// (model_mtp.safetensors, `num_nextn_predict_layers` of them) as the
        /// sidecar `<stem>.mtp.cmf` beside --output — the file the runtime
        /// loads automatically for speculative decoding. The main file is
        /// not read or rewritten. Matrices at --quant (default q8_2f, the
        /// skeleton policy), norms and sinks f32.
        #[arg(long, conflicts_with = "mimo_towers")]
        mtp_sidecar: bool,
        /// Convert an official DSpark draft checkpoint into the separate
        /// `<stem>.dspark.cmf` draft sidecar beside an existing base --output
        /// CMF: MiniCPM-V-4.7's Qwen3 DFlash draft or MiniMax-M3.1's
        /// `DSparkMiniMaxDraftModel` — the base CMF's architecture selects
        /// the contract.  This is deliberately not an MTP sidecar and does
        /// not enable a draft runtime by itself.
        #[arg(long, conflicts_with = "mtp_sidecar")]
        dspark_sidecar: bool,
        /// MiMo-V2 towers: `text` (default) drops them; `mm-only` writes the
        /// companion `<stem>.mm.cmf` (vision, audio encoder, speech
        /// embeddings, audio tokenizer encoder — no text tensors; `auto`
        /// quant = q4tp matrices); `multimodal` writes one file with the text
        /// model and the towers. Needs a local checkpoint dir with
        /// `audio_tokenizer/`.
        #[arg(
            long = "mimo-towers",
            default_value = "text",
            value_name = "text|mm-only|multimodal"
        )]
        mimo_towers: String,
        /// Record the model's recommended sampling in the file, e.g.
        /// `temperature=1.0,top_p=0.95,top_k=0,min_p=0,repetition_penalty=1.0`
        /// (keys: temperature, top_p, top_k, min_p, repetition_penalty,
        /// presence_penalty). `run` and `serve` start from it; flags and
        /// request fields still override it.
        #[arg(long, value_name = "KEY=VALUE,...")]
        sampling: Option<String>,
        /// `--quant blr`: target bits per weight of the layer projections
        /// (strict container count; embeddings/lm_head stay q8_2f, norms float).
        /// Any value in (0, 16) is accepted: the rank rule turns it into a
        /// per-tensor rank, and the report states what was delivered.
        #[arg(long = "bpw", visible_alias = "blr-bpw", default_value = "0.55")]
        blr_bpw: f64,
        /// `--quant blr`: a bit-budget plan, one rule per line:
        /// `layer <n> <bpw>` (every tensor of the layer) or
        /// `tensor <name> <bpw>` (one tensor, wins over the layer rule).
        /// `#` starts a comment. The global `--bpw` is the default for
        /// everything the plan does not name.
        #[arg(long = "blr-plan", value_name = "FILE")]
        blr_plan: Option<String>,
        /// `--quant blr` encoder: `nvg` (data-aware, needs --calib; sequential
        /// per-layer calibration) or `dual-svid` (the data-free LittleBit
        /// initialiser, kept as the comparison baseline).
        #[arg(long = "blr-method", default_value = "nvg", value_name = "nvg|dual-svid")]
        blr_method: String,
        /// Calibration text for `--quant blr --blr-method nvg` (plain text;
        /// never the evaluation split).
        #[arg(long)]
        calib: Option<String>,
        /// K/V rank multiplier for GQA models (1 keeps the body at the named
        /// budget; LittleBit used 4).
        #[arg(long = "blr-kv-factor", default_value = "1.0")]
        blr_kv_factor: f64,
        /// Paths per tensor: auto (nvg: P=2 vs P=1 at the same bits, chosen on
        /// the held-out calibration slice; dual-svid: 2) | 1 | 2.
        #[arg(long = "blr-paths", default_value = "auto", value_name = "auto|1|2")]
        blr_paths: String,
        /// NVG calibration tokens used for fitting (512-token windows).
        #[arg(long = "blr-fit-tokens", default_value = "14848")]
        blr_fit_tokens: usize,
        /// NVG held-out calibration tokens for the P=1/P=2 selection.
        #[arg(long = "blr-sel-tokens", default_value = "3072")]
        blr_sel_tokens: usize,
        /// NVG relative Gram damping ρ.
        #[arg(long = "blr-rho", default_value = "0.03")]
        blr_rho: f64,
        /// Encoder threads (default: all cores).
        #[arg(long = "blr-threads")]
        blr_threads: Option<usize>,
        /// Keep the intermediate f16 source file (`<output>.src-f16.cmf`);
        /// an existing one is reused instead of reconverting.
        #[arg(long = "blr-keep-src")]
        blr_keep_src: bool,
        /// NVG encoder device: `gpu` (f32 on the wgpu device, calibration on
        /// per-op device GEMMs), `cpu` (the f64 reference), `auto` = gpu when
        /// a wgpu adapter is up (not on macOS).
        #[arg(long = "blr-device", default_value = "auto", value_name = "auto|cpu|gpu")]
        blr_device: String,
    },
    /// Embeddings from an EmbeddingGemma 2 .cmf (`cortiq convert --model
    /// <google/embeddinggemma-2 dir>`): texts, images (--image), videos
    /// (--video), audio (--audio) and interleaved text + media (--interleave)
    /// in one space — unit-length 768-d vectors, or a Matryoshka prefix with
    /// --dim 512|256|128.
    Embed {
        /// Path to the EmbeddingGemma 2 .cmf
        #[arg(long)]
        model: String,
        /// Texts to embed (one embedding each)
        texts: Vec<String>,
        /// More texts (repeatable)
        #[arg(long = "text")]
        text: Vec<String>,
        /// An audio file to embed on its own (repeatable; WAV, or any format
        /// ffmpeg reads; mixed to mono, resampled to 16 kHz, cut at 30 s)
        #[arg(long)]
        audio: Vec<String>,
        /// A file with one text per line
        #[arg(long)]
        file: Option<String>,
        /// JSON Lines: each line a string or {"text", "prompt_name"|"task", "title",
        /// "prompt", "image", "video", "audio": path | [paths], "image_tokens",
        /// "video_tokens"} (the text's <|image|> / <|video|> / <|audio|>
        /// placeholders take the media in order)
        #[arg(long)]
        jsonl: Option<String>,
        /// Task prompt: SearchQuery, Document, QuestionAnswering, FactChecking,
        /// CodeRetrieval, Classification, Clustering, SentenceSimilarity, …
        /// (--list-prompts). Default: none (the text as given).
        #[arg(long = "prompt-name", visible_alias = "task")]
        prompt_name: Option<String>,
        /// Title for the Document prompt (`title: {title} | text: …`)
        #[arg(long)]
        title: Option<String>,
        /// A raw prompt prefix instead of a named one
        #[arg(long)]
        prompt: Option<String>,
        /// Output dimension: 768, or a Matryoshka prefix 512 | 256 | 128 (re-normalized)
        #[arg(long, default_value = "768")]
        dim: usize,
        /// Print OpenAI-style JSON ({"data":[{"embedding":[…]}], "usage":…})
        #[arg(long)]
        json: bool,
        /// Write the embeddings as a float32 .npy [n, dim]
        #[arg(long)]
        npy: Option<String>,
        /// Print each input's token ids to stderr
        #[arg(long)]
        show_tokens: bool,
        /// List the task prompts the file carries and exit
        #[arg(long)]
        list_prompts: bool,
        /// Benchmark: run the forward N times, print each time (in-process)
        #[arg(long, default_value = "1", hide = true)]
        repeat: usize,
        /// An image to embed: a path or an http(s)/data URL (repeatable)
        #[arg(long = "image")]
        image: Vec<String>,
        /// A video to embed: mp4/webm/mov/… (needs ffmpeg on PATH), .y4m, or a
        /// directory of frames (repeatable). Sampled at 1 fps, at most 32 frames.
        #[arg(long = "video")]
        video: Vec<String>,
        /// Frame rate of a --video frame directory (without it the frames are
        /// taken as already sampled)
        #[arg(long = "video-fps")]
        video_fps: Option<f64>,
        /// Soft tokens per image: 70 | 140 | 280 (default) | 560 | 1120
        #[arg(long = "image-tokens")]
        image_tokens: Option<usize>,
        /// Soft tokens per video frame: 70 | 140 (default) | 280 | 560 | 1120
        #[arg(long = "video-tokens")]
        video_tokens: Option<usize>,
        /// One interleaved input: the text (its <|image|> / <|video|> /
        /// <|audio|> placeholders take the --image / --video / --audio media
        /// in order) instead of one embedding per item
        #[arg(long)]
        interleave: bool,
    },
    /// Transcribe audio with a Whisper or Nemotron 3.5 Transcribe CMF
    /// checkpoint (WAV, PCM or float).
    Transcribe {
        /// Path to a Whisper or Nemotron 3.5 Transcribe .cmf model
        model: String,
        /// Input WAV audio (Nemotron: several files are transcribed in one
        /// load, one `<file>: <text>` line each)
        #[arg(required = true, num_args = 1..)]
        audio: Vec<String>,
        /// Whisper language code, e.g. en, ru, de (default: en)
        #[arg(long, default_value = "en")]
        language: String,
        /// transcribe | translate (Whisper)
        #[arg(long, default_value = "transcribe")]
        task: String,
        /// Maximum generated text tokens (Whisper: per 30-second segment,
        /// default 440; Nemotron: per recording, default 16384)
        #[arg(long)]
        max_new_tokens: Option<usize>,
        /// Nemotron: the user instruction sent with the audio (default: the
        /// model's verbatim transcription prompt)
        #[arg(long)]
        prompt: Option<String>,
        /// Nemotron: print timings and token counts to stderr
        #[arg(long)]
        stats: bool,
        /// Nemotron: also print every word's start and end (seconds) and,
        /// in a speaker-tagged transcript, its speaker
        #[arg(long)]
        timestamps: bool,
    },
    /// Package a Gradium Phonon Q8 checkpoint as a self-contained CMF file.
    #[cfg(feature = "phonon")]
    PhononPack {
        /// Downloaded gradium/phonon-40m-multilingual directory containing
        /// config.json, model.q8.gguf, tokenizer.json and default-voice.safetensors.
        source: String,
        /// Destination .cmf file.
        #[arg(long)]
        output: String,
    },
    /// Synthesize multilingual Phonon speech from a Phonon CMF checkpoint.
    #[cfg(feature = "phonon")]
    Tts {
        /// Path to a Gradium Phonon `.cmf` model.
        model: String,
        /// Text to synthesize. Long input is split on sentence boundaries.
        text: String,
        /// Language used for text normalization: en, fr, de, es, or pt.
        #[arg(long, visible_alias = "language", default_value = "en")]
        lang: String,
        /// Output mono 24 kHz WAV path.
        #[arg(long, default_value = "speech.wav")]
        out: String,
        /// Optional precomputed Phonon voice-latent `.safetensors` file.
        /// Audio recordings are not supported because the speaker codec is not released.
        #[arg(long)]
        voice_file: Option<String>,
        /// Name assigned to --voice-file; without it only `default` is embedded.
        #[arg(long, default_value = "default")]
        voice: String,
        /// Flow-noise temperature; 0 is deterministic and 0.3 is the upstream default.
        #[arg(long, default_value_t = 0.3)]
        temperature: f32,
        /// Flow-noise seed for reproducible output.
        #[arg(long, default_value_t = 4_242_424_242_424_242u64)]
        seed: u64,
        /// Optional classifier-free guidance coefficient; 1 disables guidance.
        #[arg(long)]
        guidance: Option<f32>,
        /// Native speech-rate conditioning: negative is faster, positive is slower.
        #[arg(long, default_value_t = 0.0)]
        duration_delta: f32,
        /// Override the model's end-of-speech threshold.
        #[arg(long)]
        eos_threshold: Option<f32>,
        /// Text-token limit for each sentence chunk.
        #[arg(long)]
        max_tokens_per_chunk: Option<usize>,
    },
    /// Rewrite a container tightly, or calibrate/compact an existing MTP
    /// draft from a prompt set:
    ///
    /// `cortiq compact SOURCE.cmf PROMPTS.jsonl OUTPUT.cmf`
    ///
    /// The three-argument form uses SOURCE as its own teacher, creates or
    /// reuses the adjacent `OUTPUT.tdata` calibration cache, compacts only
    /// the MTP draft, and verifies the new runnable CMF. The trunk weights
    /// stay identical; the speculative verifier remains authoritative for
    /// emitted tokens. This is calibration/physical draft compaction, not
    /// generic task fine-tuning. The legacy
    /// `cortiq compact SOURCE.cmf --output OUTPUT.cmf` form only rewrites
    /// container layout (it does not train or compact a draft).
    Compact {
        /// Source .cmf model
        model: String,
        /// JSONL prompt set for the simple MTP-draft calibration form
        #[arg(value_name = "PROMPTS.jsonl")]
        prompts: Option<String>,
        /// New runnable .cmf with a compacted MTP draft in the simple form
        #[arg(value_name = "OUTPUT.cmf")]
        task_output: Option<String>,
        /// Output .cmf path for the legacy container-only rewrite form
        #[arg(long)]
        output: Option<String>,
        /// Lay out payloads in decode traversal order (including contiguous
        /// gate/up/down weights per MoE expert). This applies only to the
        /// legacy `--output` container rewrite; the default preserves the
        /// source directory order.
        #[arg(long)]
        execution_order: bool,
        /// Simple MTP form: fraction of each draft FFN width to retain
        /// (default: 0.90; groups are aligned to 32 channels)
        #[arg(long)]
        keep_ratio: Option<f64>,
        /// Simple MTP form: maximum direct-answer tokens the source teacher
        /// generates per prompt (default: 8192)
        #[arg(long)]
        max_tokens: Option<usize>,
        /// Simple MTP form: deterministic teacher responses per prompt
        /// (default: 1)
        #[arg(long)]
        samples: Option<u32>,
        /// Simple MTP form: seed for teacher generation and prompt split
        /// (default: 42)
        #[arg(long)]
        seed: Option<u64>,
        /// Simple MTP form: train/validation/test split (default: 90/5/5)
        #[arg(long)]
        split: Option<String>,
        /// Simple MTP form: use only the first N accepted prompt rows
        #[arg(long)]
        limit: Option<usize>,
        /// Simple MTP form: cap train records consumed by MTP compaction
        /// after the data integrity scan (0 = all)
        #[arg(long)]
        max_records: Option<usize>,
        /// Simple MTP form: cap one fresh-causal MTP calibration chunk
        /// (default: 4096 tokens)
        #[arg(long)]
        max_record_tokens: Option<usize>,
        /// Simple MTP form: retain teacher records that reach --max-tokens
        #[arg(long)]
        keep_truncated: bool,
        /// Simple MTP form: use a plain-greedy source teacher instead of its
        /// MTP/speculative path while making the reusable data cache
        #[arg(long)]
        disable_mtp: bool,
    },
    /// Recode an existing .cmf into a denser layout without the original
    /// checkpoint. `--quant q4tp` re-expresses q4_tiled/q4_block per-tile
    /// scales as rungs on a per-row ladder: same 4-bit grid, ~7% fewer bytes.
    /// Activation-Weighted Nullspace Projection: drop the weakest input
    /// channels of every MoE expert and refit the survivors to absorb what
    /// was removed. Needs calibration dumps from CMF_ACT_DUMP.
    Awnp {
        /// Source .cmf model
        model: String,
        /// Prefix of the CMF_ACT_DUMP files (`<prefix>.<layer>.f32`)
        #[arg(long)]
        acts: String,
        /// Output .cmf path
        #[arg(long)]
        output: String,
        /// Fraction of input channels to drop (0–0.9)
        #[arg(long, default_value_t = 0.25)]
        drop: f64,
        /// Ridge on C[S,S], relative to its mean diagonal
        #[arg(long, default_value_t = 1e-3)]
        ridge: f64,
        /// Variance-preserving rescale after the projection (patent 12)
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        rescale: bool,
    },
    Requant {
        /// Source .cmf model
        model: String,
        /// Output .cmf path (omit with --in-place)
        #[arg(long)]
        output: Option<String>,
        /// Target layout: blr-q8_2f (every blr tensor expanded to a dense q8_2f weight, for evaluation on the dense paths), q4tp (re-express scales), q4tp-quantize (float weights → q4tp), q2tp-draft (MTP inputs), q1g (lossless q1 repack: one scale per 128-group, 1.5 → 1.125 bpw), q1 (q1g back to q1 tiles, exact, for older readers), or embryo-q8-head (experimental tied Embryo vocabulary only; new output required)
        #[arg(long, default_value = "q4tp")]
        quant: String,
        /// Rewrite tensors inside the source file itself — for disks that
        /// cannot hold the model twice. New payloads must fit their slots.
        #[arg(long, default_value_t = false)]
        in_place: bool,
        /// q1g only: accept 128-groups whose four q1 tile scales are not
        /// byte-identical but agree within `--tol` (f16 rounding of the
        /// native scale). Without it any such group refuses the whole run.
        #[arg(long, default_value_t = false)]
        allow_tol: bool,
        /// q1g only: relative tolerance `|s_tile − s_group| / s_group` for
        /// `--allow-tol` (default 2^-8 = 0.39 %: a few f16 ulps). Tensors
        /// with a group beyond it stay q1, byte for byte.
        #[arg(long, default_value_t = 0.00390625)]
        tol: f64,
    },
    /// Bake a task's expert restriction as a switchable task mask
    /// (spec §5 expert fields): the full expert set stays, `run --task`
    /// narrows routing at inference — one file, many specialists.
    /// Defragment a dense FFN into task tubes: reorder the intermediate
    /// axis so each task's neurons form ONE contiguous run, cut the run
    /// into segments (core + tubes) and ship a task mask per tube set.
    /// A masked FFN only saves bytes once it is contiguous.
    TubeBake {
        /// Source .cmf (dense FFN)
        model: String,
        /// Plan JSON: {tasks, layers:[{order,widths}], active}
        #[arg(long)]
        plan: String,
        /// Output .cmf
        #[arg(long)]
        output: String,
    },
    /// Bake a task's skill into a model before compressing it (DTG-MA,
    /// Patent 2): soft gates on the FFN neurons of the deep layers trained
    /// with the task's LM loss at a small constant L1 (the held-out best),
    /// folded into down_proj; the width is kept (on the trainer's GPU)
    DtgmaBake {
        /// The model, f16/f32 (compress S0's src-f16.cmf of an HF checkpoint)
        model: String,
        /// The task's teacher data (.tdata: train and val splits)
        data: String,
        #[arg(long)]
        output: String,
        /// Gates on the layers from this one; the ones below run once, cached
        /// (default: the deeper half)
        #[arg(long)]
        from_layer: Option<usize>,
        #[arg(long, default_value_t = 300)]
        steps: usize,
        #[arg(long, default_value_t = 50)]
        eval_every: usize,
        #[arg(long, default_value_t = 0.1)]
        lr: f64,
        /// The constant L1 pull on the gates (a regulariser, not a target)
        #[arg(long, default_value_t = 0.003)]
        l1: f64,
        /// Initial gate logit (σ(4) = 0.98)
        #[arg(long, default_value_t = 4.0)]
        init: f32,
        #[arg(long, default_value_t = 2048)]
        window: usize,
        #[arg(long, default_value_t = 2)]
        batch: usize,
        /// Train batches cached and used (0 = all)
        #[arg(long, default_value_t = 0)]
        max_batches: usize,
        /// Loss tokens: answer | generated | all
        #[arg(long, default_value = "generated")]
        loss_on: String,
    },
    /// Train a compressed model's MTP draft on its own trunk: the trunk's
    /// hidden rows and next-token top-k once on the trainer, then the
    /// draft's block (a one-layer model from its own input) learns the
    /// trunk's distribution one position ahead — the token its verify
    /// accepts (KL, best held-out checkpoint)
    MtpHeal {
        /// The compressed model whose draft is trained
        model: String,
        /// The draft's starting weights in a higher precision (the source
        /// before compress, e.g. its f16 copy); default: the model's own
        #[arg(long)]
        init: Option<String>,
        /// Records to train on: a .tdata (its train split; every tenth
        /// window is held out)
        #[arg(long)]
        data: String,
        /// Chain steps trained: step 1 reads the trunk's hidden row, step j
        /// the draft's own step j − 1 output (the runtime drafts 3–5 deep)
        #[arg(long, default_value_t = 3)]
        chain: usize,
        /// Steps between rebuilds of the chain's inputs from the current weights
        #[arg(long, default_value_t = 150)]
        refresh: usize,
        #[arg(long, default_value_t = 393216)]
        tokens: usize,
        #[arg(long, default_value_t = 2048)]
        window: usize,
        #[arg(long, default_value_t = 3.0)]
        epochs: f64,
        #[arg(long, default_value_t = 2e-5)]
        lr: f64,
        /// The trunk's distribution kept per position
        #[arg(long, default_value_t = 64)]
        topk: usize,
        #[arg(long, default_value_t = 2)]
        batch: usize,
        #[arg(long, default_value_t = 25)]
        eval_every: usize,
        /// Evaluations without a new best before training stops (0 = never)
        #[arg(long, default_value_t = 6)]
        patience: usize,
        #[arg(long)]
        output: String,
    },
    /// Calibrate and physically compact a dense MTP draft FFN without
    /// rewriting the trunk. Statistics are teacher-forced through the
    /// actual draft stack; `--output` is always a new runnable main CMF.
    /// For a MiMo sidecar the matching `.mtp.cmf` is created beside it.
    MtpCompact {
        /// Source main .cmf (the MTP sidecar is discovered automatically)
        model: String,
        /// Calibration .tdata selector, for example calibration.tdata:train=0.5.
        /// If its directory does not exist, --teacher and --prompts generate
        /// it once before physical compaction starts.
        #[arg(long)]
        data: String,
        /// CMF teacher used only to create a missing --data directory. It may
        /// be the same file as MODEL, but must use compatible token ids.
        #[arg(long, requires = "prompts")]
        teacher: Option<String>,
        /// JSONL prompt source used only to create a missing --data directory.
        #[arg(long, requires = "teacher")]
        prompts: Option<String>,
        /// Preferred: new runnable main .cmf candidate; a MiMo `.mtp.cmf`
        /// sidecar is created next to it and the source backbone is
        /// hard-linked when possible. Compatibility: an explicit `.mtp.cmf`
        /// path writes only an external MiMo sidecar, not a runnable main.
        #[arg(long)]
        output: String,
        /// Fraction of each draft FFN width to retain; rounded down to 32-wide codec groups
        #[arg(long, default_value_t = 0.90)]
        keep_ratio: f64,
        /// Write the auditable v2 MTP plan here (default: <output>.mtp-plan.json)
        #[arg(long)]
        plan_out: Option<String>,
        /// Cap selected calibration records after the full `.tdata` integrity
        /// preflight (0 = all selected records)
        #[arg(long, default_value_t = 0)]
        max_records: usize,
        /// Maximum teacher-forced tokens per calibration sequence. Longer
        /// records are split into contiguous fresh-causal chunks so draft KV
        /// memory stays bounded (default: 4096).
        #[arg(long, default_value_t = 4096)]
        max_record_tokens: usize,
        /// Maximum direct-answer tokens generated by --teacher for each
        /// missing-data prompt (the generator records no thinking span).
        #[arg(long, default_value_t = 8192)]
        teacher_max_tokens: usize,
        /// Deterministic direct-answer samples generated per prompt when
        /// --data is missing.
        #[arg(long, default_value_t = 1)]
        teacher_samples: u32,
        /// Base seed for missing-data teacher generation.
        #[arg(long, default_value_t = 42)]
        teacher_seed: u64,
        /// Optional category-stratified split seed for missing-data generation
        /// (defaults to --teacher-seed).
        #[arg(long)]
        teacher_split_seed: Option<u64>,
        /// Train/validation/test split for missing-data generation, e.g.
        /// 90/5/5. Compaction consumes only the train split.
        #[arg(long, default_value = "90/5/5")]
        teacher_split: String,
        /// Retain missing-data records that hit --teacher-max-tokens; they
        /// remain marked truncated in the reusable dataset.
        #[arg(long)]
        teacher_keep_truncated: bool,
        /// Disable MTP/speculative decode while producing a missing-data
        /// teacher set; this is a reproducible plain-greedy control.
        #[arg(long)]
        teacher_disable_mtp: bool,
        /// Keep only the first N accepted prompt rows while generating a
        /// missing-data set.
        #[arg(long)]
        teacher_limit: Option<usize>,
    },
    /// Store every dense layer's `down_proj` a second time, transposed,
    /// so per-token neuron selection reads a neuron's down weights as a
    /// contiguous row instead of a strided column (`CMF_FFN_GATE_TOPK`).
    FfnTranspose {
        /// Source .cmf
        model: String,
        /// Output .cmf
        #[arg(long)]
        output: String,
    },
    MoeMask {
        /// Source .cmf model
        model: String,
        /// Expert-selection stats JSON (CMF_MOE_STATS dump)
        #[arg(long)]
        stats: String,
        /// Per-layer routing-mass fraction to keep (0–1]
        #[arg(long, default_value = "0.95")]
        cover: f64,
        /// Task name for the mask (activate with `run --task <name>`)
        #[arg(long)]
        name: String,
        /// Output .cmf path
        #[arg(long)]
        output: String,
    },
    /// Drop the MoE experts a task never routes to (physical defrag: kept
    /// experts renumbered, router rows sliced). Stats from CMF_MOE_STATS
    /// over a task-representative run; gate the result on `cortiq ppl`.
    MoeDefrag {
        /// Source .cmf model
        model: String,
        /// Expert-selection stats JSON (CMF_MOE_STATS dump); omit to
        /// reuse the routing counts embedded by a previous moe-defrag
        #[arg(long)]
        stats: Option<String>,
        /// Per-layer routing-mass fraction to keep (0–1]
        #[arg(long, default_value = "0.95")]
        cover: f64,
        /// Output .cmf path
        #[arg(long)]
        output: String,
    },
    /// Sign a model: detached <model>.sig (Ed25519 over the file's
    /// SHA-256). `cortiq verify` checks it automatically when present.
    Sign {
        /// Path to .cmf model file
        model: String,
        /// Signing-key file (32-byte hex seed; created on first use)
        #[arg(long, default_value = "cortiq-signing.key")]
        key: String,
    },
    /// Import a GGUF model to .cmf — native Rust (LLM GGUFs plus the Qwen Image diffusion transformer; F32/F16/BF16/Q4_0..Q6_K + K-quants; STQ1_0 ternary → exact q1t)
    ImportGguf {
        /// A local .gguf file, an HF repo id (owner/name — best .gguf auto-picked), or owner/name/file.gguf
        gguf: String,
        /// Output .cmf path
        #[arg(long)]
        output: String,
        /// Quantization for 2-D weights: q8 | q8_2f | q4 | q4t | q4tp | q2tp | q1 | q1p | q1s | q1t | f16 | vbit
        #[arg(long, default_value = "q8")]
        quant: String,
        /// Hugging Face token for gated/private GGUF repos
        #[arg(long)]
        hf_token: Option<String>,
        /// Directory with the vendor's HF tokenizer.json (+ chat_template.jinja /
        /// tokenizer_config.json) to embed verbatim instead of the tokenizer
        /// reconstructed from ggml metadata — needed when the pre-tokenizer is
        /// not plain byte-level BPE (HunYuan / Hy-MT2)
        #[arg(long)]
        tokenizer_dir: Option<String>,
    },
    /// 1-bit PTQ via error-feedback transfer (GPTQ). Calibrates each
    /// linear's input Hessian on a corpus, then quantizes it to `q1s`:
    /// binarize with the rounding error folded into the kept weights
    /// through H⁻¹ (`Σ_PS·Σ_SS⁻¹`) so the layer OUTPUT survives, plus a
    /// two-field `|W|·RMS(x)` outlier mask. Norms/embeddings/lm_head (no
    /// captured Hessian) are copied verbatim. Pair with `skill bake` (FCD)
    /// on the tail layers to recover the last of the quality.
    /// `--codec q4tp|q2tp` writes the converter's q4tp / q2tp profile instead,
    /// every linear GPTQ-rounded in its own codec (2 bits need ≥ 32k
    /// calibration tokens: on 2048 GPTQ-q2tp is worse than rounding).
    QuantizeGptq {
        /// Input .cmf (f16 or q8 — higher precision ⇒ a better fold)
        input: String,
        /// Calibration corpus: a `.txt`, a `.jsonl` (one `{"text": …}` or
        /// string per line), or a `.json` array of `[prompt, text]` pairs
        /// (the DTG-MA cache) whose texts concatenate. q4tp/q2tp: repeat it
        /// to mix corpora, `PATH=SHARE` giving each its share of --tokens
        /// (e.g. `--calib wiki.train.raw=0.75 --calib c4.jsonl=0.25`)
        #[arg(long, required = true)]
        calib: Vec<String>,
        /// Output .cmf path
        #[arg(long)]
        output: String,
        /// Outlier budget kept at f16 by the two-field mask (fraction)
        #[arg(long, default_value = "0.01")]
        keep: f32,
        /// Calibration tokens folded into the Hessians
        #[arg(long, default_value = "512")]
        tokens: usize,
        /// Relative Hessian damping λ (adds `λ·mean(diag)` to the diagonal)
        #[arg(long, default_value = "0.01")]
        lambda: f64,
        /// Target codec: q1s (default; the 1-bit error-feedback path), q4tp
        /// (GPTQ-rounded q4tp — same layout and kernels as `convert --quant
        /// q4tp`), or q2tp (the `convert --quant q2tp` profile: gate/up
        /// GPTQ'd into q2tp, the other linears into q4tp). Give q4tp/q2tp
        /// an f16 export as INPUT
        #[arg(long, default_value = "q1s")]
        codec: String,
        /// q4tp/q2tp: the file the calibration forward runs on (default
        /// INPUT). The Hessian hook only sees memory-mapped quantized
        /// tensors, so an f16 INPUT needs its q8/q8_2f sibling here
        #[arg(long)]
        calib_model: Option<String>,
        /// q4tp/q2tp: calibration window length (each window a fresh context)
        #[arg(long, default_value = "2048")]
        window: usize,
        /// q4tp/q2tp: per-tensor codec override, PATTERN=QUANT as in
        /// `convert --tensor-quant` (e.g. `lm_head.weight=q8_2f`). A q4tp or
        /// q2tp override stays GPTQ (an escort:
        /// `model.layers.7.mlp.gate_proj.weight=q4tp`); any other codec
        /// skips GPTQ and is encoded by the converter (the q8_2f escort
        /// lists from tools/quant/escort.py)
        #[arg(long = "tensor-quant", value_name = "PATTERN=QUANT")]
        tensor_quant: Vec<String>,
        /// q4tp/q2tp: worker threads (default: all cores)
        #[arg(long)]
        threads: Option<usize>,
        /// q4tp/q2tp: act-order GPTQ (channels by descending input energy,
        /// group rungs fixed up front from the original weights — at 2 bits
        /// that costs more than the order gains: not for q2tp)
        #[arg(long)]
        act_order: bool,
        /// q4tp/q2tp: Hessian cache file — loaded when it exists (calibration
        /// skipped), written after calibrating when it does not (one file
        /// per chunk, PATH.L{first}-{last}, with --calib-chunk-layers; an
        /// existing whole-model PATH serves any chunking)
        #[arg(long)]
        hessians: Option<String>,
        /// q4tp/q2tp: capture the Hessians of this many layers per
        /// calibration pass (0 = all at once). Bounds RAM — a dense Hessian
        /// is 8·cols² bytes on the host, ~0.5 GB per layer of a 2B model — at
        /// the price of one calibration forward per pass. With a wgpu GPU
        /// (the default where one is present; `CMF_GPTQ_GPU=0` keeps the
        /// CPU) the forward runs on the card and the Hessians accumulate
        /// there as packed f32 triangles, read back one group at a time by
        /// the fold: Qwen3.8-27B's 64 layers take ~50 GB, one pass
        #[arg(long, default_value = "0")]
        calib_chunk_layers: usize,
        /// q2tp: how group scales are placed — `placed` (default: ladder at
        /// the H-weighted MSE-optimal scales, full 31-rung search) or
        /// `cortiq` (the q4tp rule on the 4-level grid: converter ladder,
        /// rung from absmax over nominal−2..+1)
        #[arg(long)]
        q2tp_rule: Option<String>,
        /// q2tp: per-tensor rule override, PATTERN=RULE (first match wins),
        /// e.g. `model.layers.7.mlp.*=cortiq` for a layer whose outlier
        /// weights the MSE-optimal scales clip
        #[arg(long = "q2tp-rule-tensor", value_name = "PATTERN=RULE")]
        q2tp_rule_tensor: Vec<String>,
        /// q2tp: remove each row's output shrinkage in the calibration metric
        /// (slope (ŴHWᵀ)/(WHWᵀ), through the row's f16 `lo` — zero bits):
        /// `on` (default; −11 % ppl on MiniCPM5's placed arm) or `off`
        /// (the plain GPTQ arms of the q2tp study)
        #[arg(long, value_name = "on|off")]
        q2tp_debias: Option<String>,
        /// q4tp/q2tp: the same output debias for the q4tp tensors (opt-in:
        /// at 4 bits the shrinkage is ~16× smaller than at 2)
        #[arg(long)]
        q4tp_debias: bool,
    },
    /// GPTQ for the experts of a Flash-Next-class MoE (qwen4_exp), of
    /// DeepSeek-V4.1 (deepseek_v41) or of a Qwen3.5/3.6-MoE stack (Clover
    /// included), routed and shared: `--codec q2tp`
    /// (gate/up q2tp, down q4tp) or `q4tp`, on the GPU; the calibration file
    /// (a q4tp CMF of the model) supplies the activations and every
    /// non-expert tensor, the original safetensors checkpoint the
    /// full-precision expert weights (fetched one layer at a time).
    QuantizeMoe {
        /// Calibration .cmf (q4tp of the same model)
        calib_model: String,
        /// Original checkpoint: a Hub repo (e.g. Qwen/Qwen3.8-Flash-Next;
        /// HF_TOKEN from the environment), or a local directory with its
        /// index and shards (read in place, never deleted)
        #[arg(long)]
        hf: String,
        /// Calibration text, PATH or PATH=SHARE (repeatable)
        #[arg(long, required = true)]
        calib: Vec<String>,
        #[arg(long, default_value = "131072")]
        tokens: usize,
        #[arg(long, default_value = "2048")]
        window: usize,
        /// Layers captured per calibration pass (host RAM: 2·hidden bytes per
        /// token per layer — ~5 KB Flash-Next, ~10 KB DeepSeek-V4.1; one pass
        /// over all layers with --in-place)
        #[arg(long, default_value = "24")]
        layers_per_pass: usize,
        #[arg(long, default_value = "0.01")]
        lambda: f64,
        /// Experts routed fewer tokens than this keep the converter's encoding
        #[arg(long, default_value = "32")]
        min_rows: usize,
        /// Directory for the source shards in flight
        #[arg(long, default_value = "/tmp/cortiq-moe")]
        scratch: String,
        #[arg(long, default_value = "16")]
        threads: usize,
        #[arg(long, required_unless_present_any = ["patch_shared", "in_place"], conflicts_with = "in_place")]
        output: Option<String>,
        /// Re-encode only the shared experts of this existing quantize-moe
        /// output, in place (no --output file is written)
        #[arg(long)]
        patch_shared: Option<String>,
        /// Expert layout: q2tp (gate/up q2tp, down q4tp) or q4tp (all q4tp)
        #[arg(long, default_value = "q2tp")]
        codec: String,
        /// Rewrite the experts inside the calibration file itself, one layer
        /// at a time, for disks that cannot hold the model twice (all layers
        /// are captured in one pass first; q2tp slack is released on Linux)
        #[arg(long, default_value_t = false)]
        in_place: bool,
    },
    /// NVG Ť: light distillation training of a compressed model's metric
    /// (the h, ℓ, g scales of its binary low-rank tensors and every RMSNorm
    /// gain; with --train-signs also the signs; every other weight frozen)
    /// against its full-precision teacher, on the GPU. Qwen3.5-family models.
    NvgTrain {
        /// Student .cmf (binary low-rank tensors to train)
        student: String,
        /// Teacher .cmf (full precision, same architecture and tokenizer)
        #[arg(long)]
        teacher: String,
        /// Training text, PATH or PATH=SHARE (repeatable), as quantize-gptq
        #[arg(long, required = true)]
        calib: Vec<String>,
        /// Training tokens (per epoch)
        #[arg(long, default_value = "1048576")]
        tokens: usize,
        /// Window length (each window a fresh context)
        #[arg(long, default_value = "2048")]
        window: usize,
        /// Windows per step
        #[arg(long, default_value = "2")]
        batch: usize,
        #[arg(long, default_value = "1")]
        epochs: usize,
        /// Peak Adam learning rate (cosine decay after a 2 % warm-up)
        #[arg(long, default_value = "1e-4")]
        lr: f64,
        /// Teacher distribution kept per token
        #[arg(long, default_value = "64")]
        topk: usize,
        /// Held-out windows (best-checkpoint selection)
        #[arg(long, default_value = "8")]
        eval_windows: usize,
        /// Steps between held-out evaluations
        #[arg(long, default_value = "50")]
        eval_every: usize,
        /// Global gradient-norm clip
        #[arg(long, default_value = "1.0")]
        clip: f64,
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Output .cmf (the student with the trained metric)
        #[arg(long)]
        output: String,
        /// Also train the signs (straight-through latents, updated inside
        /// the reverse sweep; the best held-out snapshot's bits are exported)
        #[arg(long)]
        train_signs: bool,
        /// Sign optimizer: fact-mom (4 B/sign) | fact (2 B, no momentum:
        /// diverged on the 0.8B at every rate tried) | ref (12 B, small models)
        #[arg(long, default_value = "fact-mom")]
        sign_opt: String,
        /// RMS of the initial |latent| (magnitudes τ·|z|, sign from the bit)
        #[arg(long, default_value = "0.25")]
        sign_tau: f64,
        /// Peak sign learning rate, or auto = min(0.15·τ/√steps, 4e-3·τ)
        /// (the second term caps the per-step move at the stable rate)
        #[arg(long, default_value = "auto")]
        sign_lr: String,
        /// Sign steps at η = 0 while the second-moment statistics settle
        #[arg(long, default_value = "10")]
        sign_warm: usize,
        #[arg(long, default_value = "0.9")]
        sign_b1: f64,
        #[arg(long, default_value = "0.999")]
        sign_b2: f64,
        /// ε of the factored optimizer, relative to the gradient RMS
        #[arg(long, default_value = "1e-3")]
        sign_eps: f64,
        /// Clamp of the normalized sign step
        #[arg(long, default_value = "4")]
        sign_clamp: f64,
        /// Hysteresis of the bit against the latent (0 = sign of the latent)
        #[arg(long, default_value = "0")]
        sign_hyst: f64,
        /// Halve the sign rate when the per-step flip rate exceeds this 3
        /// steps running (a runaway guard: at the auto rate the latents near
        /// zero alone flip ~0.3 % per step on a 256-step run)
        #[arg(long, default_value = "2e-2")]
        sign_max_step_flip: f64,
        /// Halve the sign rate when this fraction of the signs has flipped
        #[arg(long, default_value = "0.15")]
        sign_max_total_flip: f64,
        /// Seed of the latent init and the stochastic rounding (default: --seed)
        #[arg(long)]
        sign_seed: Option<u64>,
        /// Cache file of the teacher top-k (reused when the teacher file and
        /// every data argument match)
        #[arg(long)]
        topk_cache: Option<String>,
        /// Directory for the resumable training state (metric + Adam moments,
        /// bits, sign latents and statistics, best snapshot, counters); use
        /// local disk, a 27B state is tens of GB
        #[arg(long)]
        sign_state: Option<String>,
        /// Steps between state saves
        #[arg(long, default_value = "500")]
        sign_state_every: usize,
        /// Delete the previous state before writing the next, so one state
        /// sits on disk instead of two (a 27B fact-mom state is ~57 GB); a
        /// crash during that save leaves nothing to resume from
        #[arg(long)]
        sign_state_replace: bool,
        /// Resume from the latest state in this directory (same arguments)
        #[arg(long)]
        resume: Option<String>,
        /// A bit-budget plan (`layer <n> <bpw>` / `tensor <name> <bpw>`): the
        /// student's BLR tensors are re-encoded at the planned budget before
        /// the training, with the teacher's weights as the encode target.
        /// The layers the plan does not name keep their payloads.
        #[arg(long, value_name = "FILE")]
        budget_plan: Option<String>,
        /// Directory with the cached input Grams of the re-encode
        /// (`L{layer}.{lead}.{kind}.gram`); captured from the student otherwise.
        #[arg(long, value_name = "DIR")]
        budget_gram_cache: Option<String>,
        /// The per-head row gauge (invE) on the re-encoded tensors.
        #[arg(long)]
        budget_rowweight: bool,
        /// The re-encode's calibration tokens (FIT / SEL).
        #[arg(long, default_value = "14848")]
        budget_fit_tokens: usize,
        #[arg(long, default_value = "3072")]
        budget_sel_tokens: usize,
        /// The device jobs of the re-encode.
        #[arg(long, default_value = "3")]
        budget_jobs: usize,
        /// DTG-MA before the metric: train a gate on every MLP neuron with
        /// the same distillation loss, keep this fraction of the neurons (the
        /// ones the gates rank highest, 128-aligned per layer, ≥ 25 % of each
        /// layer), cut the rest out of the binary low-rank tensors, then
        /// train the metric of what is left (NVG → DTG-MA)
        #[arg(long, value_name = "F")]
        mask_keep: Option<f64>,
    },
    /// One tensor through the NVG `blr` encoder with a per-stage profile, on
    /// the CPU reference path and/or the device path, from Grams captured
    /// once (layers 0..=l of the model's own forward) and cached. Without
    /// --tensor: list the encoder's body linears with their shapes.
    NvgBench {
        /// Model .cmf (the calibration forward runs on it)
        #[arg(long)]
        model: String,
        /// Tensor name (e.g. model.layers.0.mlp.down_proj.weight)
        #[arg(long)]
        tensor: Option<String>,
        /// Calibration text (needed until the --gram-cache files exist)
        #[arg(long)]
        calib: Option<String>,
        #[arg(long = "fit-tokens", default_value = "14848")]
        fit_tokens: usize,
        #[arg(long = "sel-tokens", default_value = "3072")]
        sel_tokens: usize,
        #[arg(long, default_value = "0.55")]
        bpw: f64,
        #[arg(long = "kv-factor", default_value = "1.0")]
        kv_factor: f64,
        #[arg(long, default_value = "auto", value_name = "auto|1|2")]
        paths: String,
        /// cpu | gpu | both
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(long, default_value = "16")]
        threads: usize,
        #[arg(long, default_value = "0.03")]
        rho: f64,
        /// Path prefix of the cached Grams (`<prefix>.fit.gram`, `.sel.gram`)
        #[arg(long = "gram-cache")]
        gram_cache: Option<String>,
        /// Encode every body linear of this layer instead of one tensor
        #[arg(long)]
        layer: Option<usize>,
        /// Tensors in flight on the device (--layer)
        #[arg(long, default_value = "3")]
        jobs: usize,
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
        /// Image path, data URL, or HTTP(S) URL for a V4.1, MiMo or LFM3-VL
        /// prompt. Repeat the flag for several images (MiMo and V4.1 place
        /// them after the text, LFM3-VL before it).
        #[arg(long = "image")]
        images: Vec<String>,
        /// MiMo or LFM3-VL video: local Y4M file or directory of image frames.
        #[arg(long = "video")]
        videos: Vec<String>,
        /// Frame rate of a frame directory (MiMo, LFM3-VL video).
        #[arg(long)]
        video_fps: Option<f64>,
        /// MiMo or LFM3-Audio clip: WAV (other formats through ffmpeg),
        /// data URL, or HTTP(S) URL; repeat for multiple clips.
        #[arg(long = "audio")]
        audios: Vec<String>,
        /// System message of the chat prompt (LFM3-Audio tasks read it:
        /// "Perform ASR in English.").
        #[arg(long)]
        system: Option<String>,
        /// LFM3-Audio: answer in speech and write it here (24 kHz WAV);
        /// the TTS system message picks the language and voice:
        /// "Perform TTS in American English using a female voice.".
        #[arg(long, value_name = "WAV")]
        speech_out: Option<std::path::PathBuf>,
        /// With --speech-out: `audio` (speech only, the TTS turn) or `mixed`
        /// (Liquid's interleaved text and speech; system message "Respond
        /// with interleaved text and audio.").
        #[arg(long, default_value = "audio", value_parser = ["audio", "mixed"])]
        speech_mode: String,
        /// Explicit MiMo multimodal companion (otherwise discovered beside text).
        #[arg(long)]
        mm: Option<std::path::PathBuf>,
        /// Maximum pixels per MiMo or LFM3-VL image (video frame) before
        /// patchification.
        #[arg(long)]
        image_max_pixels: Option<usize>,
        /// V4.1 reasoning budget: 1..=100 or low/high/max.
        #[arg(long, value_name = "1..100|low|high|max")]
        reasoning_effort: Option<String>,
        /// Maximum number of tokens to generate
        #[arg(short = 'n', long, default_value = "256")]
        max_tokens: usize,
        /// Skill to overlay (spec §9; a file with routable skills picks
        /// one automatically — `none` forces the backbone, an id pins):
        /// replacement tensors are read in
        /// place of backbone tensors
        #[arg(long)]
        skill: Option<String>,
        /// Greedy decoding (temperature 0) — gates and base models
        #[arg(long)]
        greedy: bool,
        /// Sampling temperature (default: the file's sampling defaults when
        /// it carries them, else 0.7; ignored with --greedy)
        #[arg(long, conflicts_with = "greedy")]
        temperature: Option<f32>,
        /// Repetition penalty (default: the file's, else 1.1; 1.0 disables —
        /// better for code, where repeated identifiers are the point, not a
        /// defect)
        #[arg(long, conflicts_with = "greedy")]
        rep_penalty: Option<f32>,
        /// Nucleus cutoff (default: the file's, else 0.9; Qwen3.8: 0.95
        /// thinking, 0.80 instruct)
        #[arg(long)]
        top_p: Option<f32>,
        /// Top-k cutoff, 0 = off (default: the file's, else 40; Qwen3.8: 20)
        #[arg(long)]
        top_k: Option<u32>,
        /// Min-p floor, 0 = off (default: the file's, else 0.05; Qwen3.8: 0.0)
        #[arg(long)]
        min_p: Option<f32>,
        /// Flat penalty on every already-seen token (Qwen3.8 instruct: 1.5)
        #[arg(long)]
        presence_penalty: Option<f32>,
        /// Fixed RNG seed for reproducible sampling
        #[arg(long)]
        seed: Option<u64>,
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
        /// Router v2: also score quarantined skills in the automatic
        /// decision (gate measurement; debug)
        #[arg(long)]
        route_include_quarantine: bool,
        /// A prompt routed to a `lookup` record: answer (the table's text
        /// is printed, nothing is generated; default) | context (the card
        /// is prepended to the prompt and the backbone generates) | off
        /// (the table is ignored). Default: CMF_LOOKUP_MODE.
        #[arg(long)]
        lookup_mode: Option<String>,
        /// After generation, reprint the answer with each token coloured
        /// by the model's confidence (softmax probability): green = sure, red =
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
        /// Network pipeline-split: address of a `cortiq worker` holding the
        /// tail layers of the SAME .cmf (e.g. 169.254.33.120:9911). The
        /// worker's file is verified by dir_hash — a different file is
        /// refused, never blended.
        #[arg(long)]
        peer: Option<String>,
        /// First layer the peer/second GPU runs (it holds
        /// [SPLIT..num_layers)). Default: half the stack. 0 = the peer
        /// runs every layer (this side keeps embed / head / sampler
        /// only). Works with both --peer and --gpus.
        #[arg(long)]
        peer_split: Option<usize>,
        /// Shared secret for --peer (must match the worker's --token).
        #[arg(long, requires = "peer")]
        net_token: Option<String>,
        /// Wire payload dtype for --peer: f32 = bit-exact, f16 = half the
        /// bytes (hidden states tolerate it; measure your model).
        #[arg(long, default_value = "f32", requires = "peer")]
        net_dtype: String,
        /// Hand the peer the final norm, lm_head and sampler: it answers
        /// token ids instead of hidden states. The head does not shrink
        /// with --peer-split, so on a weak coordinator it is the floor —
        /// worth 29 ms of a 73 ms token on an Android phone. With
        /// --peer-split 0 the whole per-token wire becomes 16 bytes.
        #[arg(long, requires = "peer")]
        peer_head: bool,
        /// Tokens the peer may return for ONE round trip. In head mode
        /// over the whole stack this side does nothing between tokens, so
        /// asking each time costs a round trip and buys nothing — and on
        /// Wi-Fi a round trip is 9 ms at the median and 95 at p99. Not
        /// speculation: the same sequential decode, identical output,
        /// fewer handshakes. Default 1.
        #[arg(long, requires = "peer_head", default_value_t = 1)]
        peer_run_ahead: u32,
        /// Prefill the prompt on the peer, pull its KV home, then decode
        /// HERE with the wire idle — pull the cable and the conversation
        /// continues. Prefill ships whole chunks (a few round trips for a
        /// whole prompt) while decode pays one per token, which is the
        /// asymmetry this trades on: 1800 positions cost 86.9 s on a
        /// phone and 11.3 s on a laptop. The state that comes back is
        /// ~224 KiB a position, so `--net-dtype f16` usually decides
        /// whether it pays. Needs `--peer-split 0`.
        #[arg(long, requires = "peer")]
        peer_prefill: bool,
        /// Split the layer stack across N local GPUs, in ONE process:
        /// segment i runs pinned to card i and only a hidden vector
        /// crosses the boundary. This is the CAPACITY mode — for a
        /// model that fits one card it costs nothing but gains nothing
        /// either; for throughput use `serve --gpus N` (replicas).
        /// `cortiq gpu` lists the cards.
        #[arg(long, conflicts_with = "peer")]
        gpus: Option<usize>,
    },
    /// Serve a layer span of a model to a network coordinator (`run --peer`).
    /// The worker holds the SAME .cmf as the coordinator (checked by
    /// dir_hash) and runs the layer range the coordinator assigns.
    Worker {
        /// Path to .cmf model file (same file as the coordinator's)
        model: String,
        /// Listen address; beyond loopback --token is REQUIRED
        #[arg(long, default_value = "127.0.0.1:9911")]
        listen: String,
        /// Shared secret the coordinator must present
        #[arg(long)]
        token: Option<String>,
    },
    /// List the `cortiq worker`s on this network. This broadcasts a small
    /// query to UDP 9910, re-sent about every 900 ms while it waits; each
    /// worker listening beyond loopback answers by unicast, and this prints
    /// what answered, with the address to hand to `--peer`. The answer
    /// carries identity and geometry, never the token. A worker started
    /// with `CMF_NET_BEACON=0` does not answer.
    Peers {
        /// Seconds to wait for answers (the query is re-sent about every 0.9 s)
        #[arg(long, default_value_t = 3.0)]
        wait: f64,
        /// Only workers holding THIS model — the dir_hash must match or
        /// the handshake would refuse anyway
        #[arg(long)]
        model: Option<String>,
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
    /// Dequantize one tensor of a .cmf into raw little-endian floats
    /// (`--dtype f32|bf16`), row-major — the bridge to offline tools
    /// (the FCD/fold experiments read a quantized model's exact numerics
    /// this way instead of re-implementing the codecs). `--name` may be a
    /// prefix with `--all`: one file per tensor under `--out` (a dir).
    /// `--dtype raw` writes the entry's ENCODED payload verbatim instead —
    /// the input of tools that re-encode payload fields themselves and
    /// store the result back with `patch-tensor --raw`.
    Dequant {
        /// Path to .cmf model file
        model: String,
        /// Tensor name (or prefix with --all)
        #[arg(long)]
        name: String,
        /// Output file (or directory with --all)
        #[arg(long)]
        out: String,
        /// f32 (default), bf16, or raw (the encoded payload verbatim)
        #[arg(long, default_value = "f32")]
        dtype: String,
        /// Treat --name as a prefix and dump every matching tensor
        #[arg(long, default_value_t = false)]
        all: bool,
    },
    /// Re-encode one 2-D tensor of a .cmf IN PLACE from raw f32 (row-major,
    /// rows×cols as in the directory) with the entry's OWN quantization —
    /// the payload keeps its slot; hashes are recomputed. Verify after.
    /// With `--raw` every `--set` file is an already ENCODED payload that
    /// is stored verbatim (its length is checked against the dtype/shape).
    PatchTensor {
        /// Path to .cmf model file (modified in place unless --output)
        model: String,
        /// `name=path.f32` — repeatable; raw f32 rows×cols little-endian
        /// (with --raw: the encoded payload)
        #[arg(long = "set")]
        sets: Vec<String>,
        /// Encode the patched tensors as this dtype instead of their own
        /// (f16 | q8_2f | q4tp | q2tp); needs --output when the payload
        /// grows. With --raw: the dtype the payloads are encoded in
        #[arg(long)]
        dtype: Option<String>,
        /// Write a new file (all other tensors copied verbatim) instead
        /// of patching in place
        #[arg(long)]
        output: Option<String>,
        /// Each `--set` path holds an already ENCODED payload (the entry's
        /// own dtype, or `--dtype`), stored verbatim — no re-encoding
        #[arg(long, default_value_t = false)]
        raw: bool,
    },
    /// Show model information
    Info {
        /// Path to .cmf model file
        model: String,
        /// List directory entries whose name starts with this prefix
        /// (name, dtype, shape, bytes, hash64 of the bytes), with a per-prefix total.
        #[arg(long)]
        tensors: Option<String>,
    },
    /// Repair legacy Recursive Flint CMF metadata in place without touching
    /// the tensor directory or quantized weight payloads.
    RepairFlint {
        /// Path to a Recursive Flint .cmf file
        model: String,
        /// Inspect and validate the file, but do not append the corrected header
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    /// Build a standalone Recursive Flint layer-hybrid CMF by borrowing q4tp
    /// routed gate/up payloads from a q4 donor and all remaining payloads
    /// from a q2 base.  The output path must not already exist.
    FlintHybrid {
        /// Standalone q2tp Flint base (routed gate/up q2tp, shared/down q4tp)
        q2_base: String,
        /// Standalone q4tp Flint donor
        q4_source: String,
        /// New output CMF path (refuses to overwrite)
        output: String,
        /// Explicit routed layer selection: 2,7-10,48; or all / none
        #[arg(long, value_name = "LAYER[,LAYER[-LAYER]...]")]
        layers: String,
        /// Verify inputs and print the exact payload plan, but do not write
        #[arg(long, default_value_t = false)]
        dry_run: bool,
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
        /// Measure the machine's two MoE-offload bandwidths instead of the
        /// model: host-RAM copy (the CPU arm's ceiling) and host-to-VRAM
        /// upload (the fetch arm's), each over expert-sized blocks. The
        /// hybrid split is arithmetic over exactly these two numbers, so
        /// they are measured once per machine, not guessed per run.
        #[arg(long)]
        bw: bool,
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
        /// Bench the in-process layer split across N local GPUs. The
        /// warmup generation is untimed (shader compile and weight
        /// upload belong there, not in the number). `--peer` benches a
        /// remote worker instead, with three repeats and a median.
        #[arg(long, conflicts_with = "peer")]
        gpus: Option<usize>,
        /// Bench against a remote `cortiq worker`
        #[arg(long)]
        peer: Option<String>,
        /// First layer the peer/second GPU runs (default: half)
        #[arg(long)]
        peer_split: Option<usize>,
        /// Shared secret for --peer
        #[arg(long)]
        net_token: Option<String>,
        /// Wire dtype: f32 (bit-exact) | f16
        #[arg(long, default_value = "f32")]
        net_dtype: String,
        /// Give the peer the final norm, lm_head and sampler: it answers
        /// token ids. Measures the thin-client topology.
        #[arg(long)]
        peer_head: bool,
        /// Core timing (llama-bench contract): greedy argmax without a
        /// working copy, no repetition penalty, no per-token confidence
        /// softmax. Default (off) measures the full production loop.
        #[arg(long)]
        core: bool,
        /// Keep generating past end-of-sequence (the EOS ids are
        /// suppressed in the sampler): a file whose greedy answer to the
        /// bench prompt is "stop" measures nothing otherwise.
        #[arg(long)]
        ignore_eos: bool,
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
        /// (tune with CMF_ROUTE_EON/EOFF/MARGIN/PERIOD).
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
    /// Synthetic recall probe (MQAR / NIAH-lite) through the RUNTIME: K
    /// random key→value token pairs, D filler tokens, one key — does the
    /// model put its value first? Accuracy per distance, per model file.
    /// Runs on whatever the environment selects (CMF_GPU=0 → CPU;
    /// CMF_GPU=wgpu CMF_GPU_PROBE=0 CMF_GPU_WGPU_GRAPH=1
    /// CMF_EMBRYO_RESIDENT=parallel → the resident Embryo graph).
    ProbeRecall {
        /// One or more .cmf files
        #[arg(required = true)]
        models: Vec<String>,
        /// Key→value pairs per trial
        #[arg(long, default_value = "4")]
        pairs: usize,
        /// Filler lengths between the pairs and the query
        #[arg(long, default_value = "256,1024,4096,16384")]
        distances: String,
        /// Trials per distance
        #[arg(long, default_value = "32")]
        trials: usize,
        #[arg(long, default_value = "1")]
        seed: u64,
        /// random (fresh ids per position) | repeat (one id repeated)
        #[arg(long, default_value = "random")]
        filler: String,
        #[arg(long)]
        json: bool,
    },
    /// Utility gate (S10) through the runtime under the corrected
    /// cmf-im-v1 chat contract: greedy answers to a frozen prompt set
    /// (default: the audit's 11 utility prompts), per prompt exact /
    /// keyword hit, loop metrics (distinct 5-grams in the last 128
    /// tokens, longest period ≤ 16 run), first-token top-5, TTFT, tok/s.
    ProbeUtility {
        /// One or more .cmf files
        #[arg(required = true)]
        models: Vec<String>,
        /// TSV prompt file: LABEL<TAB>EXPECTED[|ALT]<TAB>PROMPT (# comments)
        #[arg(long)]
        prompts: Option<String>,
        /// JSONL prompt file: {"lang","prompt","expect":[keywords],"src"} per line
        #[arg(long)]
        prompts_jsonl: Option<String>,
        #[arg(long, default_value = "128")]
        max_tokens: usize,
        /// Repetition penalty (1.0 = pure greedy)
        #[arg(long, default_value = "1.0")]
        rep_penalty: f32,
        /// Prepend the tokenizer's BOS (the contract has none)
        #[arg(long)]
        bos: bool,
        #[arg(long)]
        json: bool,
        /// Pipeline per prompt: auto (router-v2 decision) | backbone | <skill id>.
        /// Default: auto for a file with a router policy (ROUTER_V2), the
        /// plain pipeline (no routing) otherwise.
        #[arg(long)]
        route: Option<String>,
        /// Also score quarantined v2 skills (gate measurement; debug)
        #[arg(long)]
        include_quarantine: bool,
        /// A prompt routed to a `lookup` record: answer (the table's text
        /// is the answer, nothing is generated; default) | context (the
        /// card is prepended and the backbone generates) | off. Default:
        /// CMF_LOOKUP_MODE. Per row: `route.lookup_hit`, `lookup_key`, `field`.
        #[arg(long)]
        lookup_mode: Option<String>,
    },
    /// Conversation memory of a `lookup` record (spec §9.5.2) on scripted
    /// dialogs of USER turns, every turn decided exactly as serve's lookup
    /// pre-pass decides a chat: the φ router on the turn itself (prompt
    /// contract against the tokenizer's chat frame), the key from the most
    /// recent of the last 6 user turns that holds one (`lookup_turn` =
    /// turns back), the key_first policy on the turn's own strong key.
    /// Per turn: target, decided_by, hit, lookup_turn, entry and its Latin
    /// binomial vs expect_src, field, answer. Summary: first-turn and
    /// follow-up hit rates, follow-up correct-entry rate, per_lang by the
    /// script of EACH turn (= per_turn_lang; per_dialog_lang by the
    /// dialog's language), mean lookup_turn, the first 20 misses. answer
    /// mode generates nothing; context mode generates ≤ 32 greedy tokens
    /// on serve's transcript. --route auto on a file without a router
    /// policy warns: serve never routes such a file.
    ProbeDialog {
        /// .cmf file
        model: String,
        /// JSONL: {"src": "<latin>", "turns": ["…", …], "expect_src": "<latin>" |
        /// ["<latin>" | null per turn]} per line; optional "lang", "id"
        #[arg(long)]
        dialogs_jsonl: String,
        /// auto (router-v2 decision; the default on a ROUTER_V2 file) | backbone | <skill id>
        #[arg(long)]
        route: Option<String>,
        /// answer (default) | context | off. Default: CMF_LOOKUP_MODE
        #[arg(long)]
        lookup_mode: Option<String>,
        /// Also route to quarantined v2 skills (gate measurement; debug)
        #[arg(long)]
        include_quarantine: bool,
        /// Tokens generated per turn in context mode (capped at 32)
        #[arg(long, default_value = "32")]
        max_tokens: usize,
        #[arg(long)]
        json: bool,
    },
    /// Forced-choice fact probe for small models whose free generation is
    /// too weak for keyword matching: every prompt rendered as cmf-im-v1
    /// (as probe-utility), then log P(candidate | prefix) for every
    /// candidate string (sum of token log-probs; --norm mean = per-token
    /// mean) off ONE prefix forward per prompt — the candidates form a
    /// token trie and the host KV state is rolled back between siblings.
    /// Prediction = argmax; correct iff it equals expect[0]
    /// (case-insensitive). Candidates: the distinct expect[0] of the file
    /// or --candidates-jsonl. Reports top-1 / top-5 accuracy against the
    /// majority-class and chance baselines, mean log-prob of the correct
    /// vs the best wrong candidate, per-lang and per-row. Runs per-op on
    /// the host route; CMF_GPU / CMF_GROWTH / CMF_GROWTH_SHELL from the
    /// environment, as probe-utility.
    ProbeChoice {
        /// .cmf file
        model: String,
        /// JSONL prompt set: {"lang","prompt","expect":[answer, …],"src"} per line
        #[arg(long)]
        prompts_jsonl: String,
        /// Candidate strings, one {"text": …} per line (default: the
        /// distinct expect[0] values of the prompt set)
        #[arg(long)]
        candidates_jsonl: Option<String>,
        /// sum (log P of the whole candidate) | mean (per token)
        #[arg(long, default_value = "sum")]
        norm: String,
        /// shared (one prefix forward, KV rollback between candidates) |
        /// full (a fresh forward of prefix + candidate per candidate: the
        /// slow cross-check of the shared walk)
        #[arg(long, default_value = "shared")]
        scorer: String,
        /// Prepend the tokenizer's BOS (the contract has none)
        #[arg(long)]
        bos: bool,
        /// Cloze mode: score the prompt as a raw text prefix (no chat frame);
        /// candidates get a leading space. Routing still decides on the prompt.
        #[arg(long)]
        raw: bool,
        /// Pipeline per prompt: auto (router-v2 decision) | backbone | <skill id>.
        /// Default: auto for a file with a router policy (ROUTER_V2), the
        /// plain pipeline (no routing) otherwise.
        #[arg(long)]
        route: Option<String>,
        /// Also score quarantined v2 skills (gate measurement; debug)
        #[arg(long)]
        include_quarantine: bool,
        #[arg(long)]
        json: bool,
    },
    /// Router-v2 gate G3: the per-request decision on every prompt of a
    /// JSONL set against the expected target; accept / recall rate, the
    /// exact Clopper–Pearson 95 % upper bound of the error rate, per-src and
    /// per-lang breakdown, input sha256. On a file with a key_first lookup
    /// record the decision is serve's: the φ router, then the key_first
    /// step (a strong key takes a backbone decision) — a key_first take
    /// counts as an accept of the record (decided_by_counts); the summary
    /// says key_first_step: true, the gate skill-gate needs to activate a
    /// key_first record.
    RouteEval {
        /// .cmf file
        model: String,
        /// JSONL prompt set: {"prompt", "lang"?, "src"?} per line
        #[arg(long)]
        prompts_jsonl: String,
        /// Expected target of every prompt: backbone | <skill id>
        #[arg(long)]
        expect: String,
        /// Also score quarantined v2 skills (gate measurement)
        #[arg(long)]
        include_quarantine: bool,
        /// Measure the φ router alone, without the key_first step (a
        /// diagnostic: such a summary is not a gate for a key_first record)
        #[arg(long)]
        router_only: bool,
        #[arg(long)]
        json: bool,
    },
    /// Gate G2 data: every prompt rendered as cmf-im-v1 (as probe-utility),
    /// greedy --tokens tokens, and the logits of the last prompt position
    /// and of every generated position written as records
    /// {index u32, route u8, n_positions u32, vocab u32, f32[n_positions·vocab]}
    /// (LE; route 0 = backbone, 1+i = header.skills[i]).
    DumpLogits {
        /// .cmf file
        model: String,
        /// JSONL prompt set: {"prompt", …} per line
        #[arg(long)]
        prompts_jsonl: String,
        /// Greedy tokens per prompt
        #[arg(long, default_value = "64")]
        tokens: usize,
        /// auto | backbone | <skill id> (default: auto with a router
        /// policy, no routing otherwise)
        #[arg(long)]
        route: Option<String>,
        /// Also score quarantined v2 skills (gate measurement)
        #[arg(long)]
        include_quarantine: bool,
        /// Output file (the sidecar `<out>.meta.json` records the path,
        /// growth and shell modes for `logits-compare`)
        #[arg(long)]
        out: String,
        /// Forward path: per-op (every lane on the host route — what
        /// `growth-eval` measures; the default) | auto (the engine's own
        /// choice: the resident graph where the backend has one). Both
        /// dumps of a comparison and its `--only-indices` set must share it.
        #[arg(long)]
        path: Option<String>,
        /// A prompt routed to a `lookup` record: answer (the record carries
        /// the backbone's logits on the plain prompt under the skill's
        /// route code; default) | context (the backbone on the
        /// card-prepended prompt) | off. Default: CMF_LOOKUP_MODE.
        #[arg(long)]
        lookup_mode: Option<String>,
    },
    /// Compare two `dump-logits` files record by record (by index): matched,
    /// bit-identical, max|Δ|, per route of B; G2 = every backbone record of
    /// B (the candidate) bit-identical to A's. JSON on stdout.
    LogitsCompare {
        /// Reference dump (F0)
        a: String,
        /// Candidate dump (F1)
        b: String,
        /// Compare only these record indices: a JSON array, or the object
        /// `growth-eval --indices-out` writes (its no-hit set — G2 for a
        /// growth file: every listed record bit-identical to F0's)
        #[arg(long)]
        only_indices: Option<String>,
    },
    /// Set a v2 skill's measured gate and status by a header-only tail
    /// append (the payloads and the trunk are not touched), with a lineage
    /// event.
    SkillGate {
        /// .cmf file (updated in place, append-only)
        model: String,
        /// Skill id
        #[arg(long)]
        id: String,
        /// Gate JSON file: the route-eval summary of a non-vacuous run
        /// (auto-routing needs "status": "measured"; route-eval writes it)
        #[arg(long)]
        gate: String,
        /// active | quarantine | retired
        #[arg(long)]
        status: String,
        /// Commit --status active even when the gate is not measured (the
        /// record then stays NOT auto-routable; debugging only)
        #[arg(long)]
        allow_unmeasured: bool,
    },
    /// Gate G1: F1 is an append-only successor of F0 — equal trunk hash,
    /// every F0 directory entry unchanged, bytes [128, len(F0)) equal.
    GenomeVerify {
        /// The genome before the append
        f0: String,
        /// The file after it
        f1: String,
    },
    /// Build a `lookup` record (spec §9.5.2): an explicit key → card
    /// table (`skill.{id}.lookup.*`) appended to a COPY of the sealed
    /// genome, status quarantine, origin {user_corpus, dataset sha256,
    /// counts, dropped duplicates}. Keys are normalised (cmf-key-v2) and
    /// hashed; a key repeated inside an entry is dropped, a key two
    /// entries share goes by --on-duplicate (default: refuse), a key
    /// longer than the runtime's 4-word n-gram window is dropped as
    /// unreachable — all reported. JSON report on stdout.
    LookupBuild {
        /// The sealed genome (F0); never modified
        base: String,
        /// entries.jsonl: {"keys": [...], "<lang>": {"card": "...",
        /// "fields": {"family": "...", ...}}, ...} per line
        #[arg(long)]
        entries: String,
        /// Record id (skill.{id}.lookup.*)
        #[arg(long)]
        id: String,
        /// Output file: a copy of the base plus the record (must differ
        /// from the base; overwritten)
        #[arg(long)]
        out: String,
        /// Slot languages in order, e.g. ru,en (default: every language
        /// the entries carry, sorted by name)
        #[arg(long)]
        langs: Option<String>,
        /// Record name (default: the id)
        #[arg(long)]
        name: Option<String>,
        /// A key two entries share: first | last (the entry in file order
        /// that keeps it) | error (refuse the build — the default)
        #[arg(long, default_value = "error")]
        on_duplicate: String,
        /// Stop list: one key per line (# comments), left out of the table
        /// (generic words such as plant / растения / species)
        #[arg(long)]
        drop_keys: Option<String>,
        /// A prompt set (JSONL {"prompt"} or one per line) the built table
        /// is probed with through the runtime's key extraction: keys hit
        /// by more than --suspicious-share of the prompts are reported
        #[arg(long)]
        probe_prompts: Option<String>,
        /// Share of --probe-prompts above which a key is suspicious
        #[arg(long, default_value = "0.02")]
        suspicious_share: f64,
        /// A large GENERAL prompt set (JSONL {"prompt"} or one per line,
        /// e.g. the user turns of a chat corpus) the built table is probed
        /// with through the runtime's STRONG-key extraction: every strong
        /// key it hits is a general phrase that key_first would answer
        /// from the table (general_probe in the report)
        #[arg(long)]
        general_prompts: Option<String>,
        /// Write the keys the general probe hit (every stored spelling
        /// behind a hit), one per line, as a stop list for --drop-keys
        #[arg(long)]
        general_stop_out: Option<String>,
        /// Routing policy written to lookup.policy: router_and_key (the φ
        /// router decides; the default when omitted) | key_first (a strong
        /// key — ≥ 2 real words of ≥ 3 letters, no digits — takes a
        /// request the router sent to the backbone; one-word keys still
        /// need the router). Probe key_first with --general-prompts.
        #[arg(long)]
        policy: Option<String>,
    },
    /// Switch the routing policy of an existing `lookup` record (spec
    /// §9.5.2) by one header-only tail append with a lineage event — no
    /// rebuild; the table, the trunk and the router's calibration stay.
    /// router_and_key: the φ router decides, the key is looked up in what
    /// it sent. key_first: a STRONG key in the message (≥ 2 real words of
    /// ≥ 3 letters, no digits — an exact or stem n-gram, a Latin group in
    /// parentheses or capitalised) sends the request to the table even
    /// when the router
    /// picked the backbone; one-word keys still need the router. An ACTIVE
    /// record switched to key_first goes to stale_regate unless its gate
    /// already covers key_first (or --keep-gate). JSON report on stdout.
    LookupPolicy {
        /// .cmf file with the lookup record (updated in place, append-only)
        model: String,
        /// Lookup record id
        #[arg(long)]
        id: String,
        /// router_and_key | key_first
        #[arg(long)]
        policy: String,
        /// Switching an ACTIVE record to key_first sets it stale_regate
        /// (its gate was measured without the key_first step: re-gate with
        /// route-eval + skill-gate); --keep-gate keeps it active anyway
        #[arg(long)]
        keep_gate: bool,
    },
    /// Fit the request router (policy v2 backbone_gated, spec §9.4) for
    /// one v2 skill with the RUNTIME's φ (probe_phi_span over the
    /// canonical cmf-im-v1 span at --phi-layer, unit norm — what
    /// route_request scores): the skill's and the backbone's descriptors
    /// (mean + top-rank PCA rows, error statistics, the last 20 % of each
    /// prompt set held out), then header.router, header.routing
    /// (calibrate_v2 over every class of the file) and the record's
    /// selection by ONE header-only tail append. Prints the calibration
    /// summary (JSON on stdout): in-scope recall on the skill held-out,
    /// false-accept on the general held-out, θ, T.
    RouteFit {
        /// .cmf file with the skill record (updated in place, append-only)
        model: String,
        /// Skill id
        #[arg(long)]
        id: String,
        /// In-scope prompts: JSONL {"prompt"} / {"text"} / strings, or one
        /// prompt per line
        #[arg(long)]
        skill_prompts: String,
        /// General prompts (the backbone class), same formats
        #[arg(long)]
        general_prompts: String,
        /// φ = hidden after this layer (default: the file's router.phi.layer)
        #[arg(long)]
        phi_layer: Option<usize>,
        /// PCA rank of both descriptors (clamped to train − 1)
        #[arg(long, default_value = "16")]
        rank: usize,
        /// At most this many prompts per set, in file order (0 = all)
        #[arg(long, default_value = "4000")]
        max: usize,
        /// router.margin: a skill must beat the backbone by this much in
        /// unit error (default: the current router's, else 0.05)
        #[arg(long)]
        margin: Option<f32>,
        /// Target in-scope false-positive rate of the novelty flag (θ)
        #[arg(long, default_value = "0.05")]
        target_fpr: f32,
    },
    /// Growth records (`expert_append`): every prompt rendered as cmf-im-v1
    /// (as dump-logits), prefill + --max-tokens greedy steps, the MoE
    /// routing counters before/after → grown-expert wins per layer; per
    /// prompt hit = any grown win; hit rate, n, Clopper–Pearson 95 % upper
    /// bound, per-layer / per-record / per-src / per-lang breakdown.
    /// --indices-out writes the no-hit indices for
    /// `logits-compare --only-indices` (G2). Runs per-op (the counters
    /// live on the host route).
    GrowthEval {
        /// .cmf file with expert_append records (F1)
        model: String,
        /// JSONL prompt set: {"prompt", "lang"?, "src"?} per line
        #[arg(long)]
        prompts_jsonl: String,
        /// Greedy tokens per prompt (a dump-logits --tokens N record needs N)
        #[arg(long, default_value = "32")]
        max_tokens: usize,
        #[arg(long)]
        json: bool,
        /// Write the no-hit record indices (JSON) here
        #[arg(long)]
        indices_out: Option<String>,
        /// Apply the growth shell: on | off | both (default: CMF_GROWTH_SHELL)
        #[arg(long)]
        shell: Option<String>,
        /// Records to mount: active | all | off (default: CMF_GROWTH; a
        /// quarantined record needs `all`)
        #[arg(long)]
        growth: Option<String>,
        /// Forward path: per-op only (the counters live on the host route;
        /// the graph exports none) — the same flag as `dump-logits --path`,
        /// recorded in --indices-out; `logits-compare --only-indices`
        /// refuses dumps of another path
        #[arg(long)]
        path: Option<String>,
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
        /// Pin the route instead of deciding it: a skill id, or `none` for
        /// the backbone
        #[arg(long)]
        skill: Option<String>,
        /// Also score quarantined v2 skills (gate measurement; debug)
        #[arg(long)]
        route_include_quarantine: bool,
    },
    /// Measure confidence calibration (B1): is the model's softmax
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
    /// Swarm skills (spec §9): bake a skill from a real donor checkpoint,
    /// list what a file carries
    Skill {
        #[command(subcommand)]
        cmd: SkillCmd,
    },
    /// Compress a model for one task: `cortiq compress SOURCE.cmf
    /// PROMPTS.jsonl OUTPUT.cmf --ratio 0.8`. The teacher (the source, or
    /// `--teacher` — another .cmf or any OpenAI-compatible API) answers the
    /// prompts; the parts the task does not use leave the model (FFN width
    /// by illumination or DTG-MA masks, unused MoE experts), the rest is
    /// refit and healed on the answers, and held-out answers score it. The
    /// output keeps the source's codec — quantize it afterwards if you want.
    Compress(compress::CompressArgs),
    /// Work with finalized teacher-data (`.tdata`) directories
    TeacherData {
        #[command(subcommand)]
        cmd: TeacherDataCmd,
    },
    /// What the GPU backend can see: adapters, the one that would be
    /// chosen, and its limits. Answers "is the card visible at all?"
    /// without inferring it from a missing log line.
    Gpu,
    /// Verify file integrity: envelope, sections, per-tensor hashes
    Verify {
        /// Path to .cmf model file
        model: String,
    },
    /// Migrate Embryo files written before 0.8.1 IN PLACE: move the
    /// BOUNDED_STATE/GENOME bits from 7/8 (today's PRISM_HADAMARD/PRISM_AFFINE)
    /// to 12/13. Only envelope bytes 12..16 change, so appended records and
    /// the lineage survive (a re-export would drop them); a detached .sig
    /// must be renewed with `cortiq sign` afterwards
    MigrateEmbryoBits {
        /// .cmf files to migrate (files without the old bits are left alone)
        #[arg(required = true)]
        files: Vec<String>,
        /// Report what would change, write nothing
        #[arg(long)]
        dry_run: bool,
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
        /// Polish WITHOUT converting attention: `--o1` then only names
        /// which layers are TRAINABLE (their FFN and the two layer
        /// norms) and both teacher and student keep exact attention.
        /// This is the correction pass a compressed model wants — the
        /// damage is in the FFN, not in the attention kernel — and with
        /// `--kl 0` it is plain cross-entropy on the corpus.
        #[arg(long, default_value_t = false)]
        polish_only: bool,
        /// Distil from a SEPARATE, uncompressed .cmf instead of the
        /// model's own frozen state. For a compressed model that is the
        /// difference between anchoring on what it should do and
        /// anchoring on the damage; use it with `--kl` above 0.
        #[arg(long)]
        teacher: Option<String>,
    },
    /// Text-to-image with MOSS-Image, Z-Image / Z-Image-Turbo or Lumina
    /// CMFs, or edit reference images with Qwen Image CMFs.
    ///
    /// Z-Image (one .cmf from `imagine-pack`): every option defaults to the
    /// recipe stored in the file — Turbo: 1024x1024, 8 steps, guidance 0;
    /// Z-Image (base): 1024x1024, 28 steps, guidance 4 with CFG — so
    /// `cortiq imagine z-image-turbo.cmf --prompt "…"` needs no other flag.
    /// MOSS-Image (one .cmf from `imagine-pack`): the stored official recipe
    /// is 1024x1024, 50 Flow-Euler steps, CFG 3, shift 5, with its Qwen3.5
    /// text template and Flux2 decoder embedded.  Its text, DiT and VAE run
    /// as separate residency stages, so no component path is needed.
    /// Qwen-Image-2.1 (one .cmf from `imagine-pack`): 1024x1024, 40 steps, no
    /// CFG by default; PNG output keeps the alpha channel (ask for transparency
    /// in the prompt); --image adds condition images.
    /// Qwen Image Edit takes a transformer.cmf plus text_encoder.cmf and vae.cmf
    /// beside it; pass --image once per reference. Metal/Vulkan/DX12 are selected when available.
    Imagine {
        /// Model: a Z-Image/Lumina/Qwen .cmf, or a Lumina/Qwen root directory
        model_dir: String,
        /// Text prompt. MOSS also accepts exactly one --layout or --layout-file.
        #[arg(
            long,
            required_unless_present_any = ["layout", "layout_file"],
            conflicts_with_all = ["layout", "layout_file"]
        )]
        prompt: Option<String>,
        /// A JSON layout prompt for MOSS-Image (exclusive with --prompt).
        /// The object is passed unchanged through MOSS's official text template.
        #[arg(long, value_name = "JSON", conflicts_with_all = ["prompt", "layout_file"])]
        layout: Option<String>,
        /// Path to a JSON layout prompt for MOSS-Image (exclusive with --prompt).
        #[arg(long, value_name = "FILE", conflicts_with_all = ["prompt", "layout"])]
        layout_file: Option<String>,
        /// Image height in pixels (MOSS/Z-Image: multiple of 16; default from the file)
        #[arg(long)]
        height: Option<usize>,
        /// Image width in pixels (MOSS/Z-Image: multiple of 16; default from the file)
        #[arg(long)]
        width: Option<usize>,
        /// Denoising steps = DiT forwards per image (MOSS default 50 with
        /// CFG doing two forwards; Z-Image: Turbo 8/base 28; Lumina 30)
        #[arg(long)]
        steps: Option<usize>,
        /// Classifier-free guidance scale. Z-Image: 0 disables CFG (one DiT
        /// forward per step; the Turbo default), > 0 runs pos + g·(pos − neg)
        /// (base default 4). Lumina / Qwen Image Edit: default 4, ≤1 disables
        /// CFG. Qwen-Image-2.1: default 1 (off); true CFG needs --cfg > 1 AND
        /// --negative-prompt.
        #[arg(long, visible_alias = "guidance")]
        cfg: Option<f32>,
        /// Z-Image CFG renormalization: clip ‖pred‖ to C·‖pos‖ (the whole
        /// tensor norm). A bare flag means 1.0 (diffusers `True`); 0 = off.
        #[arg(long, num_args = 0..=1, default_missing_value = "1.0")]
        cfg_normalization: Option<f32>,
        /// Z-Image CFG truncation: CFG only while t_norm = 1 − σ ≤ this value
        /// (1.0 = every step; e.g. 0.5 drops CFG for the last half of the
        /// noise range)
        #[arg(long)]
        cfg_truncation: Option<f32>,
        /// MOSS/Z-Image scheduler shift override (default from the file)
        #[arg(long)]
        shift: Option<f32>,
        /// MOSS negative-branch blocks to skip during the active interval,
        /// as comma-separated zero-based indices. Use `none` or a bare flag
        /// to disable the source default (22).
        #[arg(long, value_name = "LAYERS", num_args = 0..=1, default_missing_value = "none")]
        uncond_skip_layers: Option<String>,
        /// MOSS positive-branch SLG blocks, as comma-separated zero-based indices.
        /// Together with --slg-scale this adds one DiT forward while active.
        #[arg(long, value_name = "LAYERS")]
        slg_layers: Option<String>,
        /// MOSS standard skip-layer-guidance correction scale (>= 0).
        #[arg(long)]
        slg_scale: Option<f32>,
        /// MOSS denoising-fraction interval start for source skip/SLG (0..=1).
        #[arg(long)]
        slg_start: Option<f32>,
        /// MOSS denoising-fraction interval end for source skip/SLG (0..=1).
        #[arg(long)]
        slg_stop: Option<f32>,
        /// MOSS/Z-Image prompt token cap after the chat template (MOSS default 4096)
        #[arg(long)]
        max_sequence_length: Option<usize>,
        /// Images to generate (MOSS/Z-Image); image i uses seed + i and the prompt
        /// is encoded once. Files: <out-stem>_<i>.<ext> when more than one.
        #[arg(long, default_value_t = 1)]
        num_images: usize,
        /// Random seed (MOSS/Z-Image: SplitMix64 + Box-Muller noise; CMF_INIT_LATENT
        /// applies only to compatible Z-Image latent geometry)
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Condition/reference image (Qwen-Image-2.1, Qwen Image Edit); repeat for multiple images
        #[arg(long = "image")]
        images: Vec<String>,
        /// Qwen2.5-VL CMF (default: text_encoder.cmf beside the transformer)
        #[arg(long)]
        text_encoder: Option<String>,
        /// Qwen Image VAE CMF (default: vae.cmf beside the transformer)
        #[arg(long)]
        vae: Option<String>,
        /// Negative prompt. MOSS: used whenever CFG != 1 (default "", with
        /// source negative-branch skip at block 22). Z-Image: used when guidance > 0
        /// (default "", the diffusers default). Qwen Image Edit: default a space, enabling true
        /// CFG. Qwen-Image-2.1: no default; CFG runs only when this is given
        /// and --cfg > 1.
        #[arg(long)]
        negative_prompt: Option<String>,
        /// Optional Qwen Image FlowMatch Euler scheduler JSON
        #[arg(long)]
        scheduler: Option<String>,
        /// Qwen Image reference area as side squared (Qwen-Image-2.1: condition
        /// images and the default output size); 1024 is the official profile
        #[arg(long, default_value_t = 1024)]
        reference_size: usize,
        /// Output image path (Z-Image/Qwen: PNG/JPEG/PPM by extension, Z-Image
        /// default out.png; Lumina: P6 PPM, default out.ppm)
        #[arg(long)]
        out: Option<String>,
    },
    /// Pack Meta SAM 3 (facebook/sam3) exports into ONE CMF: vision trunk +
    /// SAM2-style FPN + CLIP text tower + DETR detector (concept/point/box
    /// prompts, mask head) + the video tracker, with the vendor configs and
    /// tokenizer embedded.
    /// Run a standalone autoencoder CMF (Primal VAE, `cortiq convert` of
    /// ostris/primal_vae): encode an image to its latent (.npy), decode a
    /// latent to an image, or round-trip an image and print the PSNR.
    /// The whole forward runs on the GPU (Metal / Vulkan) when one is
    /// available.
    Vae {
        /// The VAE CMF
        #[arg(long)]
        model: String,
        /// encode | decode | roundtrip
        #[arg(long, default_value = "roundtrip")]
        mode: String,
        /// An image (encode, roundtrip) or a [1, C, h, w] latent .npy (decode)
        #[arg(long)]
        input: String,
        /// A latent .npy (encode) or a .png (decode, roundtrip)
        #[arg(long)]
        output: String,
    },
    Sam3Pack {
        /// Export directory: detector.safetensors (+ tracker.safetensors),
        /// config.json, detector_config.json, processor_config.json,
        /// tokenizer.json
        src: String,
        /// Projection codec for 2-D weights: q8_2f (default) or q4tp
        #[arg(long, default_value = "q8_2f")]
        quant: String,
        /// Output .cmf path
        #[arg(long)]
        out: String,
    },
    /// SAM3 inference: an image OR a video + a text prompt → mask overlays
    /// and JSON. Image input: one mask-overlay PNG and boxes/scores
    /// (preprocess and prompt framing are the reference ones — resize
    /// 1008², mean/std 0.5, CLIP frame padded to 32 — so the boxes line up
    /// with `tools/sam3_ref.py` fixtures).
    /// Video input — a directory of PNG frames, or an mp4/mkv/webm/… file
    /// (frames extracted with ffmpeg) — runs the memory-bank tracker
    /// (`trk.*`): a folder of overlay PNGs (mask + track id) and a JSON of
    /// tracks with per-frame boxes, scores and id stability.
    /// `cortiq segment model.cmf img.png --prompt "red square"`
    /// `cortiq segment model.cmf frames/ --prompt "red square" --out overlays/`
    Segment {
        /// Path to the SAM3 .cmf (from `sam3-pack`)
        model: String,
        /// Input: an image (PNG, JPEG, …), a directory of PNG frames, or a
        /// video file (mp4/mkv/webm/mov/avi — needs ffmpeg)
        input: String,
        /// Text prompt, e.g. "red square"
        #[arg(long)]
        prompt: String,
        /// Overlay PNG path (default: <image stem>_seg.png beside the input)
        #[arg(long)]
        out: Option<String>,
        /// Write the JSON here instead of stdout
        #[arg(long)]
        json: Option<String>,
        /// Keep at most this many objects, sorted by score
        #[arg(long, default_value = "10")]
        topk: usize,
        /// Score threshold (0.5 = the vendor's `score_threshold_detection`)
        #[arg(long, default_value = "0.5")]
        thr: f32,
        /// Mask binarization threshold on the sigmoid (HF default 0.5)
        #[arg(long = "mask-thr", default_value = "0.5")]
        mask_thr: f32,
    },
    /// Pack a Diffusers source into CMF. Qwen-Image-2.1, Z-Image and
    /// MOSS-Image-Base pipeline directories become ONE ready-to-run file
    /// (their DiT + text encoder + decoder + tokenizer + recipe) and a
    /// <out>.sha256. For the source Base string-prompt text-to-image route,
    /// MOSS retains the Qwen language tower and Flux2 decode path; Qwen vision
    /// and Flux2 encode tensors are not reachable by that route. Otherwise the
    /// default packs Lumina; --component packs a standalone Qwen component;
    /// --bundle merges retained Qwen component CMFs into one file.
    ImaginePack {
        /// Optional standalone Qwen component: qwen-text-encoder or qwen-vae
        #[arg(long)]
        component: Option<String>,
        /// Merge transformer.cmf, text_encoder.cmf and vae.cmf plus the
        /// scheduler into one mmap-served Qwen Image CMF without requantizing.
        #[arg(long, default_value_t = false)]
        bundle: bool,
        /// Diffusers root directory, or a pinned HF resolve base URL for Qwen
        root: String,
        /// Projection codec (Qwen-Image-2.1 DiT: q4tp (default)/q8/raw;
        /// Z-Image/MOSS DiT: q8 (=q8_2f, default)/q4tp/f16/bf16/raw;
        /// Qwen: q4tp/q4t/q8_2f/f16; Lumina: q4t (default)/q8)
        #[arg(long)]
        quant: Option<String>,
        /// Z-Image / Qwen-Image-2.1 / MOSS text-encoder projection codec:
        /// q8 (=q8_2f, default)/q4tp/f16/bf16/raw (embed_tokens stays q8_row
        /// unless raw/bf16/f16)
        #[arg(long)]
        te_quant: Option<String>,
        /// Z-Image/MOSS `te.embed_tokens` codec (default q8_row with a
        /// quantized --te-quant; raw/bf16/f16/f32 keep it 16/32-bit)
        #[arg(long)]
        te_embed_quant: Option<String>,
        /// Z-Image / Qwen-Image-2.1 / MOSS text-encoder projections kept at
        /// source precision (comma list: `layers.N.mlp.down_proj` or a suffix
        /// like `down_proj`; Z-Image defaults layer 6; MOSS/Qwen default none;
        /// a whole `layers.N` also works; `none` keeps nothing)
        #[arg(long, value_delimiter = ',')]
        te_keep: Vec<String>,
        /// Z-Image recipe stored in the file: turbo or base (default: from
        /// the scheduler shift — 3 = turbo, 6 = base)
        #[arg(long)]
        variant: Option<String>,
        /// Z-Image dev: pack only the first N main DiT layers (kernel tests)
        #[arg(long)]
        dit_layers: Option<usize>,
        /// Z-Image / Qwen-Image-2.1 / MOSS: skip hashing source shards into provenance
        #[arg(long, default_value_t = false)]
        no_source_sha: bool,
        /// Qwen-Image-2.1 vision-tower projection codec (default q8 = q8_2f)
        #[arg(long)]
        vis_quant: Option<String>,
        /// Qwen-Image-2.1 VAE weight codec (default f16; f32/bf16/raw)
        #[arg(long)]
        vae_quant: Option<String>,
        /// Qwen-Image-2.1/MOSS DiT projections kept at q8_2f over the
        /// --quant base. Qwen: `blocks.N`, a projection kind, or
        /// `blocks.N.<kind>`; MOSS: `layers.N`, `q`/`k`/`v`/`o`/
        /// `ffn_up_gate`/`ffn_down`, or `layers.N.<projection>`.
        #[arg(long, value_delimiter = ',')]
        dit_keep: Vec<String>,
        /// Qwen-Image-2.1: leave out the vision tower (text-to-image only)
        #[arg(long, default_value_t = false)]
        no_vision: bool,
        /// Output .cmf path
        #[arg(long)]
        out: String,
    },
    /// Pack MiniMax-H3 (+ the 4-step Turbo LoRA) into ONE quantized
    /// .cmf that `animate` runs straight off the mmap. Each source is
    /// optional and `--in` carries a previous pass through, so a stand
    /// with less disk than the sum of the inputs packs one at a time.
    /// MiniMax-Music-3 text → music with vocals, from a `.cmf` packed
    /// by `animate-pack --music-te/--music-dit/--music-vae`
    Music {
        /// Path to the packed .cmf
        model: String,
        /// Caption: style, instruments, mood, production
        #[arg(long)]
        prompt: String,
        /// Lyrics, with [verse]/[chorus] section tags
        #[arg(long, default_value = "")]
        lyrics: String,
        /// Seconds of audio to ask for; the model can stop earlier
        #[arg(long, default_value_t = 10.0)]
        seconds: f32,
        /// Euler steps for the latent
        #[arg(long, default_value_t = 8)]
        steps: usize,
        /// Same seed, same prompt → the same song
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Output .wav
        #[arg(long, default_value = "music.wav")]
        out: String,
    },
    /// Decode LTX-2.5 latents through the packed video VAE (`vvae.*`).
    LtxDecode {
        /// The .cmf packed by `ltx-pack`
        #[arg(long)]
        model: String,
        /// safetensors with a `latent` tensor [1,128,F,H,W] or [128,F,H,W]
        #[arg(long)]
        latent: String,
        /// Write frames as PPM stills into this directory
        #[arg(long)]
        out_dir: Option<String>,
        /// Write the decoded frames as a safetensors tensor
        #[arg(long)]
        out_tensors: Option<String>,
        /// Compare against a `frames` tensor in --latent and report the difference
        #[arg(long)]
        gate: bool,
        /// Write every intermediate stage to a safetensors (port debugging)
        #[arg(long)]
        dump_stages: Option<String>,
    },
    /// Run the LTX-2.5 AV DiT (`dit.*`) for one denoising step, from
    /// inputs captured off the reference — the port's numeric gate.
    LtxDit {
        /// The .cmf packed by `ltx-pack`
        #[arg(long)]
        model: String,
        /// safetensors of reference DiT inputs/outputs (see docs/LTX.md)
        #[arg(long)]
        oracle: String,
        /// Compare every captured stage against the oracle
        #[arg(long)]
        gate: bool,
        /// Write our stages to a safetensors
        #[arg(long)]
        dump: Option<String>,
    },
    /// Render an LTX-2.5 video on our engine: denoise with the AV DiT and
    /// decode through the video VAE, all from one .cmf.
    LtxRender {
        /// The .cmf packed by `ltx-pack`
        #[arg(long)]
        model: String,
        /// safetensors with `enc.video` / `enc.audio` prompt embeddings
        #[arg(long)]
        context: String,
        #[arg(long, default_value_t = 256)]
        height: usize,
        #[arg(long, default_value_t = 384)]
        width: usize,
        #[arg(long, default_value_t = 49)]
        frames: usize,
        #[arg(long, default_value_t = 24.0)]
        fps: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Write frames as PPM stills into this directory
        #[arg(long)]
        out_dir: Option<String>,
        /// Write the frames as one YUV4MPEG2 stream (feed it to ffmpeg)
        #[arg(long)]
        out_y4m: Option<String>,
        /// Write the denoised latent as safetensors
        #[arg(long)]
        out_latent: Option<String>,
        /// Stop after denoising, before the VAE
        #[arg(long)]
        skip_decode: bool,
    },
    /// Encode a prompt with the packed Gemma-4 encoder, the aggregate
    /// projections and the connectors — the context the DiT reads.
    LtxEncode {
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        /// safetensors of reference encoder activations, to compare against
        #[arg(long)]
        oracle: Option<String>,
        /// Write `enc.video` / `enc.audio` for `ltx-render --context`
        #[arg(long)]
        out: Option<String>,
    },
    /// Prompt in, video out: the whole LTX-2.5 pipeline on our engine.
    LtxVideo {
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        height: usize,
        #[arg(long, default_value_t = 384)]
        width: usize,
        #[arg(long, default_value_t = 49)]
        frames: usize,
        #[arg(long, default_value_t = 24.0)]
        fps: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Sample the way the distilled model was trained: half resolution,
        /// latent upscale, then a three-step refinement at full resolution
        #[arg(long)]
        two_stage: bool,
        /// Write the frames as one YUV4MPEG2 stream (feed it to ffmpeg)
        #[arg(long)]
        out: Option<String>,
        /// Write frames as PPM stills into this directory
        #[arg(long)]
        out_dir: Option<String>,
        /// Write the denoised latent as safetensors
        #[arg(long)]
        out_latent: Option<String>,
        /// Decode the soundtrack and write it as a 48 kHz stereo WAV
        #[arg(long)]
        out_audio: Option<String>,
        /// Start from a still (binary PPM): image-to-video
        #[arg(long)]
        image: Option<String>,
        /// Start from a clip (a directory of frame_*.ppm): video-to-video
        #[arg(long)]
        video: Option<String>,
        /// Hold the given picture fixed and write only its soundtrack
        #[arg(long)]
        video_to_audio: bool,
        /// Start from a soundtrack (16-bit WAV): audio-to-video
        #[arg(long)]
        audio_in: Option<String>,
        /// How far to re-noise a `--video` clip that covers the whole render:
        /// 1.0 keeps only its composition, 0.2 barely touches it
        #[arg(long, default_value_t = 0.6)]
        video_strength: f32,
        /// Encode the prompt every time instead of reusing a cached context
        #[arg(long)]
        no_context_cache: bool,
        /// Denoising steps (default 8 — the distilled ladder itself)
        #[arg(long)]
        steps: Option<usize>,
        /// Refinement steps in the second stage (default 3)
        #[arg(long)]
        steps2: Option<usize>,
        /// A LoRA adapter (.safetensors) applied at runtime
        #[arg(long)]
        lora: Option<String>,
        /// How hard to apply the adapter
        #[arg(long, default_value_t = 1.0)]
        lora_strength: f32,
        /// Reference still (binary PPM) for a multi-subject adapter; repeat up to 5
        #[arg(long = "ref")]
        refs: Vec<String>,
        /// Pixel frames each reference is held for: 25 or 33
        #[arg(long, default_value_t = 25)]
        ref_frames: usize,
    },
    /// Decode a saved audio latent through the audio VAE and vocoder.
    LtxAudio {
        #[arg(long)]
        model: String,
        /// safetensors carrying `audio_latent` (written by `ltx-video --out-latent`)
        #[arg(long)]
        latent: String,
        #[arg(long, default_value = "out.wav")]
        out: String,
        /// Print the statistics of every stage
        #[arg(long)]
        stats: bool,
        /// safetensors of reference `mel` / `waveform` to compare against
        #[arg(long)]
        oracle: Option<String>,
    },
    /// Pack LTX-2.5 — the 22B audio-video DiT, the Gemma-4 12B prompt
    /// encoder, both VAEs, the latent upscalers and the duration head —
    /// into ONE q4tp .cmf. Multi-pass: `--in` carries an earlier pass
    /// through byte for byte, so a stand with less disk than the sum of
    /// the sources packs one component at a time.
    LtxPack {
        /// Output .cmf path
        #[arg(long)]
        out: String,
        /// A .cmf from an earlier pass; its tensors are copied through
        #[arg(long = "in")]
        carry: Option<String>,
        /// The 22B transformer (ltx-2.5-22b-{dev,distilled}-transformer-bf16.safetensors)
        #[arg(long)]
        dit: Option<String>,
        /// Gemma-4 12B prompt encoder (gemma4-12b-with-proj-ltx-2.5-bf16.safetensors)
        #[arg(long)]
        te: Option<String>,
        /// Video VAE (ltx-2.5-video-vae-conv-bf16.safetensors)
        #[arg(long)]
        video_vae: Option<String>,
        /// Audio VAE (ltx-2.5-audio-vae-bf16.safetensors)
        #[arg(long)]
        audio_vae: Option<String>,
        /// Latent spatial upscaler x2
        #[arg(long)]
        spatial_upscaler: Option<String>,
        /// Latent temporal upscaler x2
        #[arg(long)]
        temporal_upscaler: Option<String>,
        /// Duration head
        #[arg(long)]
        duration_head: Option<String>,
        /// Codec for the big 2-D planes [default: q4tp]
        #[arg(long, default_value = "q4tp")]
        quant: String,
        /// Codec for convolutions (VAEs, upscalers): f16 keeps the
        /// decoder exact, q4tp folds kernels to [out, in·k·k·k]
        #[arg(long, default_value = "f16")]
        vae_quant: String,
        /// 2-D planes smaller than this many weights stay f16
        #[arg(long, default_value_t = 1 << 20)]
        min_q4tp: usize,
    },
    AnimatePack {
        /// Output .cmf path
        #[arg(long)]
        out: String,
        /// A .cmf from an earlier pass; its tensors are copied through
        #[arg(long = "in")]
        carry: Option<String>,
        /// MiniMax-H3 DiT — the PRUNED/curve checkpoint
        /// (minimax_h3_*_pruned_bf16.safetensors)
        #[arg(long)]
        dit: Option<String>,
        /// Turbo 4-step LoRA, merged into the packed weights
        #[arg(long)]
        lora: Option<String>,
        /// The silu(t_emb) curve the LoRA's adaLN update lives in —
        /// needed only with `--lora` on a pruned base. Either the Turbo
        /// node's bundled `h3_silu_temb_grid.safetensors` (5.5 MB) or
        /// the full checkpoint's four `time_embedder.*` tensors
        /// (tools/mmh3_fetch.py range-reads them, 64 MB)
        #[arg(long)]
        time_embedder: Option<String>,
        /// LoRA strength: down for over-sharp grain, up for smear
        #[arg(long, default_value_t = 1.0)]
        lora_scale: f32,
        /// Qwen3-VL-32B prompt encoder safetensors
        #[arg(long)]
        te: Option<String>,
        /// Pack only the first N encoder layers. The conditioning is a
        /// TAP, so everything above it never runs: H3's own file is the
        /// 32B cut at 50, and a ClipProj stand-in is cut at the layer
        /// its projection was fitted on
        #[arg(long)]
        te_layers: Option<usize>,
        /// MiniMax-Music-3's AR stack
        /// (`minimax_music3_text_encoder_pruned_bf16.safetensors`): the
        /// Qwen3-8B backbone, its embedding tables and the RVQ depth
        /// decoder. Carries the tokenizer, so `--tokenizer` is optional
        #[arg(long)]
        music_te: Option<String>,
        /// MiniMax-Music-3's flow-matching DiT
        /// (`minimax_music3_dit_fp16.safetensors`)
        #[arg(long)]
        music_dit: Option<String>,
        /// MiniMax-Music-3's DAV decoder (`minimax_music3_dav.safetensors`).
        /// Weight normalisation is folded at pack time and everything
        /// stays exact — a vocoder is where quantization costs hiss
        #[arg(long)]
        music_vae: Option<String>,
        /// ClipProj projection (`mmh3-<size>-ClipProj*.safetensors`):
        /// lets a SMALL Qwen3-VL stand in for the 32B encoder, its
        /// tapped hidden state mapped into the DiT's conditioning space
        /// by a fitted affine map. The encoder it projects from is
        /// whatever `--te` packs — pass the matching one
        #[arg(long)]
        clip_proj: Option<String>,
        /// Vision-row twin of --clip-proj (fitted on image activations);
        /// the runtime routes image-span rows through it
        #[arg(long)]
        clip_proj_vis: Option<String>,
        /// Video VAE (only the ViT3D decoder half is packed)
        #[arg(long)]
        video_vae: Option<String>,
        /// Audio VAE (only the BigVGAN decoder half is packed)
        #[arg(long)]
        audio_vae: Option<String>,
        /// Qwen3-VL tokenizer.json for the VOCAB section
        #[arg(long)]
        tokenizer: Option<String>,
        /// Projection codec: q4tp | q2tp | q8 | f16. q2tp drops the
        /// gate/up planes to two bits and leaves everything else at
        /// four; norms and the fp32 island stay exact either way
        #[arg(long, default_value = "q4tp")]
        quant: String,
        /// Qwen3-VL's vision tower — `fl2va` needs it, and it lives
        /// inside the prompt encoder's file under `visual.`
        #[arg(long)]
        vision: Option<String>,
        /// Attention heads of the vision tower — architecture, not in
        /// the checkpoint
        #[arg(long, default_value_t = 16)]
        vis_heads: usize,
        /// Vision layers whose features feed the deepstack mergers
        #[arg(long, default_value = "8,16,24")]
        vis_deepstack: String,
        /// Attention heads of the video VAE's ViT3D decoder — an
        /// architecture constant of the release, not in the checkpoint
        #[arg(long, default_value_t = 32)]
        vvae_heads: usize,
    },
    /// MiniMax-H3 text → video with synchronized stereo audio, from a
    /// `.cmf` packed by `animate-pack`. Writes an MJPEG+PCM AVI (and a
    /// .wav beside it) — no ffmpeg, nothing to install
    Animate {
        /// Packed .cmf
        model: String,
        #[arg(long)]
        prompt: String,
        /// Multiple of 32; the trained short edge is 768
        #[arg(long, default_value_t = 512)]
        width: usize,
        #[arg(long, default_value_t = 288)]
        height: usize,
        /// Frames at 24 fps, snapped up to the model's 17k+5 grid
        #[arg(long, default_value_t = 39)]
        frames: usize,
        /// The Turbo LoRA is trained for 4; more still helps a little
        #[arg(long, default_value_t = 4)]
        steps: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Integrate the audio on the VIDEO sigma grid, as a stock
        /// sampler does. Wrong at 4 steps; here to measure how wrong
        #[arg(long)]
        stock_sampler: bool,
        /// JPEG quality for the AVI's frames
        #[arg(long, default_value_t = 92)]
        quality: u32,
        /// First frame of the clip, as a binary P6 PPM. Turns the run
        /// into fl2va: the picture conditions the DiT as a latent AND
        /// enters the prompt as a vision block
        #[arg(long)]
        first_frame: Option<String>,
        /// Last frame, same format. Cover-cropped to the canvas where
        /// the first one is stretched, as the reference does
        #[arg(long)]
        last_frame: Option<String>,
        /// A directory of reference frames (`frame_0000.ppm` …): every
        /// `--video-stride`-th one becomes a condition pinned to its own
        /// moment in the clip. This is the release's `v2v` as this
        /// container can do it — the `fl2va` keyframe path with more than
        /// two frames, not a port of the reference's video node
        #[arg(long)]
        video: Option<String>,
        /// Take every n-th frame of `--video` (default 8: at 24 fps that
        /// is one reference every third of a second)
        #[arg(long, default_value_t = 8)]
        video_stride: usize,
        /// Chunk-causal (streaming) generation: latent frames per chunk.
        /// Each chunk is denoised seeing only the text, `--stream-sink`
        /// chunks from the start and a sliding `--stream-window` of recent
        /// ones — the protocol a streaming adapter (RAVEN) is trained for,
        /// and what stops the activation cache growing with clip length
        #[arg(long, default_value_t = 0)]
        stream_chunk: usize,
        #[arg(long, default_value_t = 2)]
        stream_sink: usize,
        #[arg(long, default_value_t = 2)]
        stream_window: usize,
        /// The published latent upscaler (`minimax_h3_latent_upscaler_3d_*.safetensors`):
        /// the denoised latent is resized by the learned net and the VAE
        /// decodes at the larger size — no decode → resize → encode round
        /// trip through the 5 B-parameter VAE, and none of the ghosting a
        /// plain interpolation puts there
        #[arg(long)]
        upscale: Option<String>,
        /// How far the latent upscaler takes it
        #[arg(long, default_value_t = 2.0)]
        upscale_by: f32,
        /// A LoRA adapter (.safetensors) applied at runtime — the
        /// community adapters for MiniMax-H3 as they ship
        #[arg(long)]
        lora: Option<String>,
        /// How hard to apply the adapter
        #[arg(long, default_value_t = 1.0)]
        lora_strength: f32,
        /// Output .avi (a .wav is written alongside)
        #[arg(long, default_value = "out.avi")]
        out: String,
        /// Also dump every frame as `frame_%04d.ppm` into this directory —
        /// the interchange format the LTX refine stage reads back
        /// (`cortiq ltx-video --video <dir> --video-strength …`), and the
        /// chunk-handoff format for a second machine
        #[arg(long)]
        frames_dir: Option<String>,
    },
}

/// Every `stride`-th `*.ppm` of a directory, each with the pixel index it
/// stands for in a render of `frames` frames. The source clip is mapped onto
/// the render's length by position, so a 100-frame reference conditions a
/// 39-frame render at the same *moments*, not the same indices.
fn read_frame_dir(
    dir: &str,
    stride: usize,
    frames: usize,
) -> anyhow::Result<Vec<((Vec<f32>, usize, usize), usize)>> {
    let mut names: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| anyhow::anyhow!("{dir}: {e}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("ppm")))
        .collect();
    names.sort();
    if names.is_empty() {
        return Err(anyhow::anyhow!("{dir}: no .ppm frames"));
    }
    let n = names.len();
    let mut out = Vec::new();
    for (i, path) in names.iter().enumerate().step_by(stride) {
        let idx = if n == 1 {
            0
        } else {
            i * (frames.saturating_sub(1)) / (n - 1)
        };
        out.push((read_ppm(&path.to_string_lossy())?, idx));
    }
    println!(
        "reference clip: {} of {n} frames, every {stride}th, mapped onto {frames}",
        out.len()
    );
    Ok(out)
}

/// A binary P6 PPM → RGB in [0, 1] as `[3, h, w]`.
///
/// PPM and not PNG on purpose: a decoder for either of the compressed
/// formats is a few hundred lines of table-driven code for something
/// every tool on the machine can already write, and `cortiq` earns its
/// "nothing to install" by not carrying code it does not need.
fn read_ppm(path: &str) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    let b = std::fs::read(path).map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
    let mut it = b.iter().copied().enumerate();
    let mut fields: Vec<usize> = Vec::new();
    let mut cur: Option<usize> = None;
    let mut start = 0usize;
    let mut comment = false;
    for (i, c) in it.by_ref() {
        if i == 0 || i == 1 {
            continue; // "P6"
        }
        if comment {
            if c == b'\n' {
                comment = false;
            }
            continue;
        }
        if c == b'#' {
            comment = true;
            continue;
        }
        if c.is_ascii_digit() {
            cur = Some(cur.unwrap_or(0) * 10 + (c - b'0') as usize);
        } else if let Some(v) = cur.take() {
            fields.push(v);
            if fields.len() == 3 {
                start = i + 1;
                break;
            }
        }
    }
    if &b[..2] != b"P6" || fields.len() != 3 {
        return Err(anyhow::anyhow!("{path}: not a binary P6 PPM"));
    }
    let (w, h, maxv) = (fields[0], fields[1], fields[2]);
    if maxv != 255 {
        return Err(anyhow::anyhow!("{path}: only 8-bit PPM (maxval 255)"));
    }
    let px = b
        .get(start..start + w * h * 3)
        .ok_or_else(|| anyhow::anyhow!("{path}: truncated ({} bytes short)", w * h * 3))?;
    let mut out = vec![0f32; 3 * h * w];
    for p in 0..h * w {
        for c in 0..3 {
            out[c * h * w + p] = px[p * 3 + c] as f32 / 255.0;
        }
    }
    Ok((out, h, w))
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

#[derive(Subcommand)]
enum TeacherDataCmd {
    /// Generate a finalized v1 `.tdata` directory from JSONL prompts.
    ///
    /// Greedy teacher answers, one hot pipeline for every record. A reasoning
    /// model thinks unless `--no-think`; its `</think>` token splits each
    /// record into exact reasoning and answer spans. An interrupted run is
    /// continued by the same command (the directory keeps every committed
    /// record). A generation that collapses into repetition is stopped early
    /// and recorded as unusable. `--disable-mtp` gives a plain-greedy control.
    Generate {
        /// Teacher: a local .cmf, or a model behind an API —
        /// `openrouter:qwen/qwen3.8-27b` (key in OPENROUTER_API_KEY; needs
        /// --tokenizer) or `api:MODEL` with CORTIQ_API_BASE
        teacher: String,
        /// JSONL prompts (`prompt` or `messages`; `id`/`category` optional)
        prompts: String,
        /// New output directory, which must end in `.tdata`
        output: String,
        /// Maximum generated tokens per record (unless a prompt row overrides it)
        #[arg(long, default_value_t = 8192)]
        max_tokens: usize,
        /// Responses per prompt (unless a prompt row overrides it)
        #[arg(long, default_value_t = 1)]
        samples: u32,
        /// Base seed; each record derives a deterministic independent seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Optional seed for the category-stratified prompt split (defaults to --seed)
        #[arg(long)]
        split_seed: Option<u64>,
        /// Train/validation/test ratio, e.g. 90/5/5
        #[arg(long, default_value = "90/5/5")]
        split: String,
        /// Teacher distribution width. Only 0 is valid until raw generated logits are observable.
        #[arg(long, default_value_t = 0)]
        topk: u32,
        /// Reserved for a future base-completion producer; the direct-answer
        /// v1 path rejects it rather than guessing reasoning-token boundaries.
        #[arg(long)]
        raw: bool,
        /// Render the chat template with enable_thinking=false (direct answers).
        /// Without it a reasoning model thinks first; the `</think>` token
        /// marks the exact reasoning / answer boundary of every record
        #[arg(long)]
        no_think: bool,
        /// Keep records that reached --max-tokens (they remain marked truncated)
        #[arg(long)]
        keep_truncated: bool,
        /// Disable MTP/speculative draft decoding for this job. The trunk
        /// remains on the selected GPU; this is a stable plain-greedy control,
        /// not a CPU fallback.
        #[arg(long)]
        disable_mtp: bool,
        /// Keep only the first N accepted, de-duplicated source prompts
        #[arg(long)]
        limit: Option<usize>,
        /// Generate only this share of the prompts, I/N (by prompt id hash;
        /// the split is computed over all prompts first, so shards merged
        /// with `teacher-data merge` equal one whole run)
        #[arg(long, value_name = "I/N")]
        shard: Option<String>,
        /// Decoding: `greedy` (default), `file` (the teacher's recommended
        /// sampling from `convert --sampling`), or
        /// `temperature=0.6,top_p=0.95,top_k=20,…`. Sampled records stay
        /// reproducible: each draws from its own seed. Greedy decoding of a
        /// reasoning model loops; use its recommended sampling there
        #[arg(long, default_value = "greedy")]
        sampling: String,
        /// API teacher: the model whose tokenizer and chat template turn the
        /// answers into token ids (the model the data will compress)
        #[arg(long)]
        tokenizer: Option<String>,
        #[command(flatten)]
        api: teacher_api::ApiArgs,
    },
    /// Join shards of one generation job (`generate --shard I/N`) into one
    /// finalized `.tdata`. Teacher, tokenizer, seed, split and generation
    /// settings must match; a record id may appear only once.
    Merge {
        /// Finalized shard directories
        #[arg(required = true)]
        parts: Vec<String>,
        /// New output directory ending in `.tdata`
        #[arg(short, long)]
        output: String,
    },
    /// Turn an existing answer cache into a finalized `.tdata`: a `.json`
    /// array of `[prompt, rendered ChatML]` pairs (the DTG-MA notebooks'
    /// cache) or JSONL rows with `prompt`/`messages` and `response`. Token ids
    /// come from the tokenizer of `--tokenizer`; repeated prompts become
    /// samples of one prompt; `</think>` splits reasoning from the answer.
    Import {
        /// The cache (.json pairs or .jsonl rows)
        source: String,
        /// New output directory ending in `.tdata`
        output: String,
        /// The .cmf whose tokenizer (and chat template) the ids are for —
        /// the model the data will compress
        #[arg(long)]
        tokenizer: String,
        /// Train/validation/test ratio
        #[arg(long, default_value = "90/5/5")]
        split: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Inspect a finalized `.tdata` data set without loading its token payload
    /// all at once.  The scan validates every record/index/token range.
    Inspect {
        /// Finalized `.tdata` directory
        tdata: String,
        /// Make an additional full integrity pass after computing the report.
        /// The ordinary inspection scan already validates every row while it
        /// counts it; this is useful when an explicit verify pass is required
        /// by an automation policy.
        #[arg(long)]
        verify: bool,
        /// Emit one stable JSON object to stdout.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum SkillCmd {
    /// Cut a baked specialist against its base into a STANDALONE skill
    /// file: only the tensors that changed, the mask catalog, and the
    /// identity keys binding it to the base's exact bytes
    Export {
        /// The baked specialist .cmf
        specialist: String,
        /// The base .cmf it was baked from
        #[arg(long)]
        base: String,
        /// Skill id ([A-Za-z0-9_-])
        #[arg(long)]
        id: String,
        /// Human-readable name
        #[arg(long)]
        name: Option<String>,
        /// Output skill file
        #[arg(short, long)]
        output: String,
    },
    /// Attach a standalone skill file to its base: verify the identity
    /// keys, overlay the tensors and masks, write the specialist
    Apply {
        /// The base .cmf the skill was cut against
        base: String,
        /// The skill file
        skill: String,
        /// Output specialist .cmf
        #[arg(short, long)]
        output: String,
        /// Attach even if the base's directory hash does not match the
        /// skill's key (the result is unsupported territory)
        #[arg(long)]
        force: bool,
    },
    /// Graft a skill from a donor HF checkpoint (same architecture):
    /// replacement tensors + routing subspace + measured quality
    Add {
        /// Backbone .cmf to grow
        model: String,
        /// Donor: HF repo id (downloaded+cached) or a local safetensors dir
        #[arg(long)]
        from: String,
        /// Skill id ([A-Za-z0-9_-])
        #[arg(long)]
        id: String,
        /// Human-readable name for the registry
        #[arg(long)]
        name: Option<String>,
        /// Layers to replace: `all`, `A-B`, or `i,j,k`
        #[arg(long, default_value = "all")]
        layers: String,
        /// Tensor families: ffn | attn | all
        #[arg(long, default_value = "ffn")]
        tensors: String,
        /// Example prompts (one per line) → recon-argmin routing subspace
        #[arg(long)]
        prompts: Option<String>,
        /// φ layer for routing (default: 2/3 of depth)
        #[arg(long)]
        phi_layer: Option<usize>,
        /// Routing subspace rank (clamped to prompts−1)
        #[arg(long, default_value = "2")]
        rank: usize,
        /// Held-out text: measure backbone vs overlaid PPL, record in the registry
        #[arg(long)]
        quality: Option<String>,
        /// Max tokens for the quality gate
        #[arg(long, default_value = "1024")]
        quality_tokens: usize,
        /// Optional binary decision threshold to carry in the skill record.
        /// The runtime applies this positive-class probability threshold
        /// automatically when the skill is the loaded artifact.
        #[arg(long)]
        decision_threshold: Option<f32>,
        /// Skip donor tensors whose relative change vs the backbone is
        /// below this (0 = keep everything): neurons the fine-tune
        /// never touched are not stored, the skill shrinks to its real
        /// delta. Try 0.02–0.05; verify with --quality
        #[arg(long, default_value = "0.0")]
        min_delta: f32,
        /// Store the skill's tensors in a cheaper encoding than the
        /// backbone (q8 | q8_2f | q4 | q4t | q4tp | q2tp | q1 | q1p |
        /// q1s | q1t | f16 | vbit): per-tensor dtypes are native to CMF,
        /// e.g. q4 halves an overlay on a q8 backbone. Verify with --quality
        #[arg(long)]
        skill_quant: Option<String>,
        /// Mean bits for --skill-quant vbit (3.0–8.0; small models
        /// usually need 5.5–6 to keep the fine-tune's gains)
        #[arg(long)]
        mean_bits: Option<f32>,
        /// DTG-MA sparse bake (Patent 2): keep this fraction of FFN
        /// neurons per layer, chosen by the task's own activation mass
        /// over --prompts; dead neurons are zeroed in the stored skill
        /// (vbit sinks them to its bit floor) and a task mask is baked
        /// and linked — `run --skill` activates it automatically
        #[arg(long)]
        sparse: Option<f32>,
        /// Output path (default: rewrite the model in place)
        #[arg(long)]
        output: Option<String>,
        /// Hugging Face token (gated/private donors)
        #[arg(long)]
        hf_token: Option<String>,
    },
    /// List a file's skills: tensors, size, layers, routing, quality
    List {
        /// Path to .cmf model file
        model: String,
    },
    /// Native DTG-MA bake (Patent 2), no Python: train the L1 neuron
    /// mask to its denoising bottom on your corpus, FCD-polish the last
    /// layers, and write a standalone defragged specialist — pruned
    /// neurons are neither stored nor computed
    Bake {
        /// Backbone .cmf (the embedded tokenizer is used)
        model: String,
        /// Task corpus: one or more plain-text files
        #[arg(long, num_args = 1.., required = true)]
        files: Vec<String>,
        /// Dedicated validation corpus for the held-out quality gate.
        /// When omitted, the first --held chunks of --files are used
        /// for backwards compatibility.
        #[arg(long, num_args = 1..)]
        held_files: Vec<String>,
        /// Output .cmf (the standalone specialist)
        #[arg(long)]
        output: String,
        /// Phase A steps (mask training)
        #[arg(long, default_value = "240")]
        steps_a: usize,
        /// Phase B steps (FCD polish)
        #[arg(long, default_value = "120")]
        steps_b: usize,
        /// Phase A Adam learning rate
        #[arg(long, default_value = "0.1")]
        lr_a: f64,
        /// Phase B FCD Adam learning rate
        #[arg(long, default_value = "0.00001")]
        lr_b: f64,
        /// Evaluate and checkpoint every N steps
        #[arg(long, default_value = "30")]
        eval_every: usize,
        /// How many of the last layers Phase B trains
        #[arg(long, default_value = "4")]
        fcd_layers: usize,
        /// Calibration chunk length in tokens
        #[arg(long, default_value = "256")]
        chunk: usize,
        /// Held-out chunks (the quality gate)
        #[arg(long, default_value = "12")]
        held: usize,
        /// Calibration chunks read from the corpus (consumed in file
        /// order, silently truncated past this). Raise it to feed a
        /// bigger corpus whole — more variety, same wall time: steps
        /// are what cost, chunks only feed them.
        #[arg(long, default_value = "112")]
        calib_chunks: usize,
        /// Independent fixed-length records per optimizer step. The focused
        /// loss is averaged across the batch; 1 preserves the original recipe.
        #[arg(long, default_value = "1")]
        batch: usize,
        /// Records per optimizer step in the native focused final-FFN cache.
        /// It does not multiply full-backbone activation memory.
        #[arg(long, default_value = "128")]
        fcd_batch: usize,
        /// Comma-separated single-token labels to score/train on while
        /// still forwarding the full chunk as context (for example
        /// DOWN,UP). Empty means ordinary all-token LM loss.
        #[arg(long)]
        focus_tokens: Option<String>,
        /// Start marker of one chat record in the corpus files (the
        /// rendered first turn): records split on it, never on token
        /// windows. ChatML by default; Spark-X2.5 renders
        /// "<｜start▁of▁sentence｜><|System|>".
        #[arg(long, default_value = "<|im_start|>system")]
        record_start: String,
        /// Score only the assistant's answer of each record: the tokens
        /// after the LAST occurrence of this rendered generation prompt
        /// ("<|im_start|>assistant\n", "<|Bot|></think>", …) through
        /// --answer-end. Multi-token answers; exclusive with --focus-tokens.
        #[arg(long, requires = "answer_end")]
        answer_after: Option<String>,
        /// The one-token turn terminator that closes an answer
        /// ("<|im_end|>", "<｜end▁of▁sentence｜>", …)
        #[arg(long, requires = "answer_after")]
        answer_end: Option<String>,
        /// Leave out (and count) chat records longer than --chunk instead
        /// of refusing the corpus; records are never cut.
        #[arg(long)]
        drop_oversize: bool,
        /// Pad chat records to the smallest of chunk/4, chunk/2, 3·chunk/4,
        /// chunk they fit instead of the full chunk; batches are drawn
        /// from equal-length records
        #[arg(long)]
        length_buckets: bool,
        /// Start from the mask of a previous run's --archaeology-report
        /// (its `selected_logits`); with --steps-a 0 the mask is kept and
        /// only Phase B runs.
        #[arg(long)]
        mask_from: Option<String>,
        /// Patent 2 Soft-Fold compilation: surviving neurons keep their
        /// learned gain σ(m), folded into down_proj (evaluated and
        /// FCD-polished as such); default snaps them to 1.
        #[arg(long)]
        soft_fold: bool,
        /// Keep the full FFN shape and ship the mask as a CMF task mask
        /// (`--task <name>` at run/serve time) instead of cutting dead
        /// neurons out of the tensors.
        #[arg(long)]
        no_defrag: bool,
        /// Name of the task mask written with --no-defrag (or a looped model)
        #[arg(long, default_value = "specialist")]
        mask_name: String,
        /// Full-layer FCD: Phase B also trains the attention projections,
        /// the projected head gate and both norm gains of the FCD layers
        #[arg(long)]
        fcd_attn: bool,
        /// Write each Phase-A evaluation's mask logits into this directory
        /// (phase-a-step-N.json, usable with --mask-from) as the run goes
        #[arg(long)]
        checkpoint_dir: Option<String>,
        /// JSONL with one object per record of --held-files, in record
        /// order, whose "family" names the record's group: held scores
        /// report per-family verdict accuracy and gate on its macro mean
        #[arg(long)]
        held_tags: Option<String>,
        /// Target sparsity (0.0–1.0): force at least this fraction of
        /// FFN neurons to be pruned. 0 = auto (denoising bottom)
        #[arg(long, default_value = "0.0")]
        target_sparsity: f64,
        /// L1 aggression multiplier: >1.0 = harder pruning push,
        /// <1.0 = softer (scales the L1 penalty schedule)
        #[arg(long, default_value = "1.0")]
        l1_aggression: f64,
        /// Initial L1 coefficient before the first evaluation window.
        #[arg(long, default_value = "0.01")]
        l1_init: f64,
        /// L1 coefficient increment after each evaluation window.
        #[arg(long, default_value = "0.005")]
        l1_step: f64,
        /// Hard-mask sigmoid threshold used for evaluation and defrag.
        #[arg(long, default_value = "0.5")]
        tau: f32,
        /// Effective mask logit at step zero. 2.0 is the current native
        /// recipe; 4.0 reproduces the older DTG-MA trading notebooks'
        /// near-identity initialization.
        #[arg(long, default_value = "2.0")]
        mask_init: f32,
        /// Reproduce the older DTG-MA sparsity penalty softplus(logit),
        /// whose gradient stays useful near an open gate.
        #[arg(long)]
        softplus_l1: bool,
        /// With --focus-tokens, checkpoint by hard held-out balanced
        /// accuracy (then raw accuracy/PPL) instead of PPL alone.
        #[arg(long)]
        checkpoint_accuracy: bool,
        /// Optional strict minimum raw accuracy for a focused checkpoint.
        /// The checkpoint must exceed this value on the held corpus.
        #[arg(long)]
        checkpoint_min_accuracy: Option<f64>,
        /// Optional strict minimum balanced accuracy for a focused
        /// checkpoint. Combine with --checkpoint-min-accuracy to guard
        /// against majority-only specialists.
        #[arg(long)]
        checkpoint_min_balanced_accuracy: Option<f64>,
        /// Rank checkpoints meeting the configured guards by raw accuracy
        /// first, then balanced accuracy and PPL.
        #[arg(long)]
        checkpoint_raw_priority: bool,
        /// Round each layer's kept-neuron count up to a multiple of
        /// this (keeps grouped codecs + SIMD kernels on the fast
        /// path; 1 = off)
        #[arg(long, default_value = "32")]
        ffn_align: usize,
        /// Force one FFN width across all layers — required by the
        /// whole-token GPU graphs (Metal/Vulkan); costs some sparsity
        /// on uneven layers
        #[arg(long)]
        uniform_inter: bool,
        /// Write the complete continuous logits and hard keep indices as
        /// JSON so archaeology runs can be compared neuron by neuron.
        #[arg(long)]
        archaeology_report: Option<String>,
        /// Run mask/FCD archaeology and write its report, but stop before
        /// building a specialist CMF. Useful for parameter sweeps.
        #[arg(long)]
        analysis_only: bool,
        /// Skip the expensive all-token real-runtime PPL gate after the
        /// specialist is written. Intended for controlled research runs;
        /// final artifacts should still be evaluated through the runtime.
        #[arg(long)]
        skip_runtime_gate: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // A usage error never echoes an argument that looks like a key.
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => decision::exit_on_clap_error(e),
    };

    // The engine cannot depend on the CLI, but the draft's device pack wants
    // the converter's q2tp encoder to requantize its experts at upload —
    // register it here, once, for whichever command ends up drafting.
    #[cfg(feature = "gpu")]
    let _ = cortiq_engine::dsv4::DSPARK_Q2TP_ENCODE.set(convert::encode_q2tp);

    // `run` hands the screen to the model: the loader's INFO chatter is noise
    // in front of an answer. Every other command keeps the informative
    // default. RUST_LOG overrides either way.
    let default_level = match &cli.command {
        Commands::Run { .. } => "warn",
        // `decide` prints its answer (and its batch totals on stderr), and
        // with --oracle its own hint: the service's is worded for a server.
        Commands::Decide(_) => "warn,cortiq_decision::service=error",
        _ => "info",
    };
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| default_level.into());
    // Past the shutdown gate; and for `serve --shadow-of`, never a
    // DEBUG/TRACE line of a target that prints forwarded request headers
    // (ureq's request prelude: the client keys and admin token the shadow
    // server forwards), whatever RUST_LOG says. Every other command keeps
    // ureq's debug lines (downloads).
    let forwards_secrets = matches!(
        &cli.command,
        Commands::Serve {
            shadow_of: Some(_),
            ..
        }
    );
    let shutdown_filter = tracing_subscriber::filter::filter_fn(move |m| {
        TRACING_LIVE.load(AtomicOrdering::Relaxed)
            && !(forwards_secrets
                && cortiq_decision::shadow::log_may_carry_secrets(m.target(), m.level()))
    });
    tracing_subscriber::registry()
        .with(env_filter)
        .with(
            tracing_subscriber::fmt::layer()
                // Logs go to stderr: stdout carries the payload (generated
                // text, `bench --json`) and must stay machine-parseable.
                .with_writer(std::io::stderr)
                .with_filter(shutdown_filter),
        )
        .init();

    let result = match cli.command {
        Commands::Serve {
            model,
            port,
            host,
            task,
            decision_config,
            state,
            break_lock,
            jev_compatible,
            shadow_of,
            shadow_timeout_s,
            oracle,
            compat_port,
            o1,
            o1_m,
            o1_window,
            o1_sink,
            peer,
            peer_split,
            net_token,
            net_dtype,
            gpus,
            route_include_quarantine,
            lookup_mode,
        } => {
            // The language-model flags a decision file refuses (spec §4.2).
            let llm_given: Vec<&'static str> = [
                ("--task", task.is_some()),
                ("--compat-port", compat_port.is_some()),
                ("--o1", o1.is_some()),
                ("--o1-m", o1_m.is_some()),
                ("--o1-window", o1_window.is_some()),
                ("--o1-sink", o1_sink.is_some()),
                ("--peer", peer.is_some()),
                ("--peer-split", peer_split.is_some()),
                ("--net-token", net_token.is_some()),
                ("--gpus", gpus.is_some()),
                ("--route-include-quarantine", route_include_quarantine),
                ("--lookup-mode", lookup_mode.is_some()),
            ]
            .into_iter()
            .filter_map(|(name, given)| given.then_some(name))
            .collect();
            let decision_flags = decision::ServeFlags {
                decision_config: decision_config.map(Into::into),
                state: state.map(Into::into),
                break_lock,
                jev_compatible,
                shadow_of,
                shadow_timeout_s,
                oracle,
            };
            let o1 = O1Flags {
                spec: o1,
                m: o1_m,
                w: o1_window,
                sink: o1_sink,
                rect: None,
            };
            cmd_serve(
                &model,
                host.as_deref(),
                port,
                task.as_deref(),
                &llm_given,
                &decision_flags,
                compat_port,
                &o1,
                peer.as_deref(),
                peer_split,
                net_token.as_deref(),
                &net_dtype,
                gpus,
                route_include_quarantine,
                lookup_mode.as_deref(),
            )
            .await
        }
        Commands::Decide(args) => decision::run_decide(&args),
        Commands::SystemOne { model, state, questions, images, audios } => {
            run_system_one(&model, state.as_deref(), &questions, &images, &audios)
        }
        Commands::Decision { cmd } => decision::run_decision(&cmd),
        Commands::Convert {
            model,
            quant,
            output,
            also_output,
            hf_token,
            mean_bits,
            tensor_quant,
            exclude_tensors,
            resume,
            defrag,
            o1,
            o1_m,
            o1_window,
            o1_sink,
            mtp_sidecar,
            dspark_sidecar,
            mimo_towers,
            sampling,
            blr_bpw,
            blr_plan,
            blr_method,
            calib,
            blr_kv_factor,
            blr_paths,
            blr_fit_tokens,
            blr_sel_tokens,
            blr_rho,
            blr_threads,
            blr_keep_src,
            blr_device,
        } => {
            // The usual Hub token variables when --hf-token is absent: an
            // anonymous multi-hundred-GB streamed conversion runs into the
            // Hub's rate limit, and a token on the command line shows in ps.
            let hf_token = hf_token.or_else(convert::hf_token_from_env);
            // A Primal VAE checkpoint packs as a standalone autoencoder.
            if primalvaepack::is_primal_vae_source(&model) {
                return primalvaepack::pack(&model, quant.as_deref(), &output);
            }
            if quant
                .as_deref()
                .is_some_and(|q| q.eq_ignore_ascii_case("blr"))
            {
                anyhow::ensure!(
                    also_output.is_empty()
                        && !mtp_sidecar
                        && !dspark_sidecar
                        && tensor_quant.is_empty()
                        && exclude_tensors.is_empty()
                        && !resume
                        && defrag.is_none()
                        && o1.is_none(),
                    "--quant blr runs its own encoder: --also-output, --mtp-sidecar, \
                     --dspark-sidecar, --tensor-quant, --exclude-tensors, --resume, --defrag \
                     and --o1 do not apply"
                );
                let method = match blr_method.to_ascii_lowercase().as_str() {
                    "nvg" => nvg::Method::Nvg,
                    "dual-svid" | "dualsvid" => nvg::Method::DualSvid,
                    other => anyhow::bail!("--blr-method {other}: expected nvg or dual-svid"),
                };
                let paths = match blr_paths.as_str() {
                    "auto" => nvg::PathsChoice::Auto,
                    "1" => nvg::PathsChoice::One,
                    "2" => nvg::PathsChoice::Two,
                    other => anyhow::bail!("--blr-paths {other}: expected auto, 1 or 2"),
                };
                anyhow::ensure!(
                    blr_bpw > 0.0 && blr_bpw < 16.0,
                    "--bpw {blr_bpw} out of range"
                );
                anyhow::ensure!(
                    blr_kv_factor.is_finite() && blr_kv_factor > 0.0,
                    "--blr-kv-factor {blr_kv_factor} must be finite and positive"
                );
                anyhow::ensure!(
                    blr_fit_tokens >= 2 && blr_sel_tokens >= 2,
                    "--blr-fit-tokens and --blr-sel-tokens must be at least 2"
                );
                let threads = blr_threads.unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4)
                });
                let plan = match &blr_plan {
                    Some(path) => {
                        let text = std::fs::read_to_string(path)
                            .map_err(|e| anyhow::anyhow!("--blr-plan {path}: {e}"))?;
                        nvg::Plan::parse(&text)
                            .map_err(|e| anyhow::anyhow!("--blr-plan {path}: {e}"))?
                    }
                    None => nvg::Plan::default(),
                };
                let opts = nvg::BlrOpts {
                    bpw: blr_bpw,
                    plan,
                    method,
                    calib,
                    kv_factor: blr_kv_factor,
                    paths,
                    fit_tokens: blr_fit_tokens,
                    sel_tokens: blr_sel_tokens,
                    window: 512,
                    rho: blr_rho,
                    threads,
                    keep_src: blr_keep_src,
                    device: nvg::BlrDevice::resolve(&blr_device)?,
                };
                nvg::run_convert_blr(&model, &output, hf_token.as_deref(), &opts)?;
                println!("✓ wrote {output}");
                return Ok(());
            }
            anyhow::ensure!(
                calib.is_none(),
                "--calib is the NVG calibration text of --quant blr"
            );
            anyhow::ensure!(
                !mtp_sidecar || also_output.is_empty(),
                "--also-output cannot be combined with --mtp-sidecar"
            );
            anyhow::ensure!(
                !dspark_sidecar || also_output.is_empty(),
                "--also-output cannot be combined with --dspark-sidecar"
            );
            anyhow::ensure!(
                !dspark_sidecar || mimo_towers == "text",
                "--mimo-towers cannot be combined with --dspark-sidecar"
            );
            let exclude_tensors = convert::parse_tensor_exclusions(&exclude_tensors)?;
            anyhow::ensure!(
                exclude_tensors.is_empty() || !dspark_sidecar,
                "--exclude-tensors cannot be combined with --dspark-sidecar (the sidecar is a fixed tensor set)"
            );
            if dspark_sidecar {
                let q = match quant.as_deref() {
                    None => "q8_2f",
                    Some(q) if q.eq_ignore_ascii_case(convert::AUTO_QUANT) => "q8_2f",
                    Some(q) => q,
                };
                let path = convert::write_dspark_sidecar(
                    std::path::Path::new(&model),
                    std::path::Path::new(&output),
                    q,
                )?;
                println!("✓ wrote DSpark sidecar {}", path.display());
                return Ok(());
            }
            let towers = convert::parse_mimo_towers(&mimo_towers)?;
            convert::set_sampling_defaults(sampling.as_deref())?;
            // Qwen3.8-Flash-Next: the MTP head is a hyper-connected hybrid
            // layer with its own 512 experts; it converts through the normal
            // walk (expert split, skeleton policy) from the 31-tensor source
            // directory `tools/qwen4_mtp_fetch.py` builds, into `<stem>.mtp.cmf`.
            let qwen4_mtp = mtp_sidecar
                && std::fs::read_to_string(std::path::Path::new(&model).join("config.json"))
                    .ok()
                    .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
                    .is_some_and(|c| c.get("model_type").and_then(|m| m.as_str()) == Some("qwen4_exp"));
            let output = if qwen4_mtp {
                // SAFETY: single-threaded here — before any pipeline/pool spawn.
                unsafe { std::env::set_var("CMF_CONVERT_QWEN4_MTP", "1") };
                cortiq_core::mtp_sidecar_path(std::path::Path::new(&output))
                    .to_string_lossy()
                    .into_owned()
            } else {
                output
            };
            if mtp_sidecar && !qwen4_mtp {
                anyhow::ensure!(
                    exclude_tensors.is_empty(),
                    "--exclude-tensors cannot be combined with --mtp-sidecar (the sidecar is a fixed tensor set)"
                );
                let q = match quant.as_deref() {
                    None => "q8_2f",
                    Some(q) if q.eq_ignore_ascii_case(convert::AUTO_QUANT) => "q8_2f",
                    Some(q) => q,
                };
                let path = convert::write_mimo_mtp_sidecar(
                    std::path::Path::new(&model),
                    std::path::Path::new(&output),
                    q,
                )?;
                println!("✓ wrote MTP sidecar {}", path.display());
                return Ok(());
            }
            convert::set_vbit_mean_bits(mean_bits);
            convert::set_tensor_quant_overrides(&tensor_quant)?;
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
                    .ok_or_else(|| anyhow::anyhow!("--o1 {spec}: expected all | deepN | i,j,k"))?;
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
            let mut outputs = vec![(
                quant.as_deref().unwrap_or(convert::AUTO_QUANT).to_string(),
                output.clone(),
            )];
            for spec in also_output {
                let (additional_quant, additional_output) = spec
                    .split_once('=')
                    .ok_or_else(|| anyhow::anyhow!("--also-output '{spec}': expected QUANT=PATH"))?;
                anyhow::ensure!(
                    !additional_quant.trim().is_empty() && !additional_output.trim().is_empty(),
                    "--also-output '{spec}': expected non-empty QUANT=PATH"
                );
                outputs.push((
                    additional_quant.trim().to_string(),
                    additional_output.to_string(),
                ));
            }
            convert::run_convert_multi_excluding(
                &model,
                &outputs,
                hf_token.as_deref(),
                defrag.as_deref(),
                o1_hint,
                resume,
                towers,
                &exclude_tensors,
                progress_reporter("converting"),
            )?;
            for (_, output) in &outputs {
                println!("✓ wrote {output}");
            }
            Ok(())
        }
        Commands::Embed {
            model,
            texts,
            text,
            audio,
            file,
            jsonl,
            prompt_name,
            title,
            prompt,
            dim,
            json,
            npy,
            show_tokens,
            list_prompts,
            repeat,
            image,
            video,
            video_fps,
            image_tokens,
            video_tokens,
            interleave,
        } => embed::run(embed::EmbedArgs {
            model,
            texts: texts.into_iter().chain(text).collect(),
            audio,
            file,
            jsonl,
            prompt_name,
            title,
            prompt,
            dim,
            json,
            npy,
            show_tokens,
            list_prompts,
            repeat,
            images: image,
            videos: video,
            video_fps,
            image_tokens,
            video_tokens,
            interleave,
        }),
        Commands::Transcribe {
            model,
            audio,
            language,
            task,
            max_new_tokens,
            prompt,
            stats,
            timestamps,
        } => {
            let model = Arc::new(CmfModel::open(&model)?);
            if model.header.arch.arch_name == cortiq_engine::nemotron_asr::ARCH_NAME {
                return transcribe::run_nemotron(
                    &model,
                    &audio,
                    prompt.as_deref(),
                    max_new_tokens.unwrap_or(16384),
                    stats,
                    timestamps,
                );
            }
            anyhow::ensure!(audio.len() == 1, "Whisper transcribes one file per call");
            let text = transcribe::run(&model, &audio[0], &language, &task, max_new_tokens.unwrap_or(440))?;
            println!("{text}");
            Ok(())
        }
        #[cfg(feature = "phonon")]
        Commands::PhononPack { source, output } => phononpack::cmd_phonon_pack(&source, &output),
        #[cfg(feature = "phonon")]
        Commands::Tts {
            model,
            text,
            lang,
            out,
            voice_file,
            voice,
            temperature,
            seed,
            guidance,
            duration_delta,
            eos_threshold,
            max_tokens_per_chunk,
        } => {
            let options = cortiq_engine::phonon::SynthesisOptions {
                language: &lang,
                voice_file: voice_file.as_deref().map(std::path::Path::new),
                voice_name: &voice,
                temperature,
                seed,
                guidance,
                duration_delta,
                eos_threshold,
                max_tokens_per_chunk,
            };
            let result = cortiq_engine::phonon::synthesize_to_wav(
                std::path::Path::new(&model),
                &text,
                std::path::Path::new(&out),
                &options,
            )?;
            println!(
                "✓ wrote {} ({:.2}s, {} Hz, {} samples)",
                out, result.seconds, result.sample_rate, result.samples
            );
            Ok(())
        }
        Commands::Compact {
            model,
            prompts,
            task_output,
            output,
            execution_order,
            keep_ratio,
            max_tokens,
            samples,
            seed,
            split,
            limit,
            max_records,
            max_record_tokens,
            keep_truncated,
            disable_mtp,
        } => match (prompts, task_output) {
            (Some(prompts), Some(task_output)) => {
                if output.is_some() || execution_order {
                    anyhow::bail!(
                        "compact: the task form is `compact SOURCE.cmf PROMPTS.jsonl OUTPUT.cmf`; \
                         --output/--execution-order belong to the legacy container-only form"
                    );
                }
                preflight_mtp_compact_model_input(&model)?;
                preflight_simple_mtp_compact_output(&task_output)?;
                let data = simple_mtp_compact_data_path(&task_output);
                eprintln!(
                    "compact: prompt-calibrated MTP draft compaction; source is its own teacher, calibration cache: {data}"
                );
                cmd_mtp_compact_with_data(
                    &model,
                    &data,
                    Some(&model),
                    Some(&prompts),
                    &task_output,
                    keep_ratio.unwrap_or(0.90),
                    None,
                    max_records.unwrap_or(0),
                    max_record_tokens.unwrap_or(4096),
                    TeacherDataGenerateOptions {
                        max_tokens: max_tokens.unwrap_or(8192),
                        samples: samples.unwrap_or(1),
                        seed: seed.unwrap_or(42),
                        split_seed: seed.unwrap_or(42),
                        split: split.unwrap_or_else(|| "90/5/5".to_string()),
                        topk: 0,
                        raw: false,
                        no_think: true,
                        keep_truncated,
                        disable_mtp,
                        limit,
                        shard: None,
                        sampling: "greedy".into(),
                        tokenizer: None,
                        api: Default::default(),
                    },
                )
            }
            (None, None) => {
                if keep_ratio.is_some()
                    || max_tokens.is_some()
                    || samples.is_some()
                    || seed.is_some()
                    || split.is_some()
                    || limit.is_some()
                    || max_records.is_some()
                    || max_record_tokens.is_some()
                    || keep_truncated
                    || disable_mtp
                {
                    anyhow::bail!(
                        "compact: MTP calibration options need `compact SOURCE.cmf PROMPTS.jsonl OUTPUT.cmf`; \
                         the legacy container-only form accepts only --output/--execution-order"
                    );
                }
                let output = output.ok_or_else(|| {
                    anyhow::anyhow!(
                        "compact: use either `compact SOURCE.cmf --output OUTPUT.cmf` to rewrite a container, \
                         or `compact SOURCE.cmf PROMPTS.jsonl OUTPUT.cmf` to make a prompt-calibrated MTP candidate"
                    )
                })?;
                moedefrag::cmd_compact(&model, &output, execution_order)
            }
            _ => anyhow::bail!(
                "compact: the prompt-calibrated form needs both PROMPTS.jsonl and OUTPUT.cmf: \
                 `compact SOURCE.cmf PROMPTS.jsonl OUTPUT.cmf`"
            ),
        },
        Commands::Awnp {
            model,
            acts,
            output,
            drop,
            ridge,
            rescale,
        } => awnp::cmd_awnp(&model, &acts, &output, drop, ridge, rescale),
        Commands::Requant {
            model,
            output,
            quant,
            in_place,
            allow_tol,
            tol,
        } => {
            if quant == "blr-q8_2f" {
                let out = output.as_deref().ok_or_else(|| anyhow::anyhow!("--quant blr-q8_2f needs --output"))?;
                requant::cmd_requant_blr_q8(&model, out)
            } else if quant == "q1g" {
                requant::cmd_requant_q1g(&model, output.as_deref(), in_place, allow_tol, tol)
            } else if quant == "q1" && !allow_tol {
                requant::cmd_requant_q1(&model, output.as_deref(), in_place)
            } else if allow_tol {
                Err(anyhow::anyhow!("--allow-tol only applies to --quant q1g"))
            } else {
                requant::cmd_requant(&model, output.as_deref(), &quant, in_place)
            }
        }
        Commands::TubeBake {
            model,
            plan,
            output,
        } => tube::cmd_tube_bake(&model, &plan, &output),
        Commands::MtpCompact {
            model,
            data,
            teacher,
            prompts,
            output,
            keep_ratio,
            plan_out,
            max_records,
            max_record_tokens,
            teacher_max_tokens,
            teacher_samples,
            teacher_seed,
            teacher_split_seed,
            teacher_split,
            teacher_keep_truncated,
            teacher_disable_mtp,
            teacher_limit,
        } => cmd_mtp_compact_with_data(
            &model,
            &data,
            teacher.as_deref(),
            prompts.as_deref(),
            &output,
            keep_ratio,
            plan_out.as_deref(),
            max_records,
            max_record_tokens,
            TeacherDataGenerateOptions {
                max_tokens: teacher_max_tokens,
                samples: teacher_samples,
                seed: teacher_seed,
                split_seed: teacher_split_seed.unwrap_or(teacher_seed),
                split: teacher_split,
                topk: 0,
                raw: false,
                no_think: true,
                keep_truncated: teacher_keep_truncated,
                disable_mtp: teacher_disable_mtp,
                limit: teacher_limit,
                shard: None,
                sampling: "greedy".into(),
                tokenizer: None,
                api: Default::default(),
            },
        ),
        Commands::FfnTranspose { model, output } => tube::cmd_ffn_transpose(&model, &output),
        Commands::MoeMask {
            model,
            stats,
            cover,
            name,
            output,
        } => moedefrag::cmd_moe_mask(&model, &stats, cover, &name, &output),
        Commands::MoeDefrag {
            model,
            stats,
            cover,
            output,
        } => moedefrag::cmd_moe_defrag(&model, stats.as_deref(), cover, &output),
        Commands::ImportGguf {
            gguf,
            output,
            quant,
            hf_token,
            tokenizer_dir,
        } => {
            gguf::run_import_gguf(
                &gguf,
                &quant,
                &output,
                hf_token.as_deref(),
                tokenizer_dir.as_deref(),
                progress_reporter("importing"),
            )?;
            println!("✓ wrote {output}");
            Ok(())
        }
        #[cfg(feature = "gpu")]
        Commands::QuantizeMoe { calib_model, hf, calib, tokens, window, layers_per_pass, lambda, min_rows, scratch, threads, output, patch_shared, codec, in_place } => {
            moegptq::run(&moegptq::MoeOpts {
                calib_model: &calib_model,
                hf: &hf,
                calib: &calib,
                tokens,
                window,
                layers_per_pass,
                lambda,
                min_rows,
                scratch: &scratch,
                threads,
                output: output.as_deref().unwrap_or(""),
                patch_shared: patch_shared.as_deref(),
                codec: &codec,
                in_place,
            })
        }
        #[cfg(not(feature = "gpu"))]
        Commands::QuantizeMoe { .. } => Err(anyhow::anyhow!("quantize-moe needs a build with the gpu feature")),
        #[cfg(feature = "gpu")]
        Commands::MtpHeal { model, init, data, chain, refresh, tokens, window, epochs, lr, topk, batch, eval_every, patience, output } => {
            let cfg = compress::mtp_heal::Cfg { chain, refresh, tokens, window, epochs, lr, topk, batch, eval_every, patience };
            let r = compress::mtp_heal::heal(
                std::path::Path::new(&model),
                init.as_deref().map(std::path::Path::new),
                &data,
                &cfg,
                std::path::Path::new(&output),
                compress::stages::cpu_budget(),
            )?;
            println!(
                "mtp heal: held-out KL to the trunk {:.5} → {:.5} (best step {} of {}); wrote {output}",
                r.loss_start, r.loss_best, r.best_step, r.steps
            );
            Ok(())
        }
        #[cfg(not(feature = "gpu"))]
        Commands::MtpHeal { .. } => Err(anyhow::anyhow!("mtp-heal needs a build with the gpu feature")),
        #[cfg(feature = "gpu")]
        Commands::DtgmaBake { model, data, output, from_layer, steps, eval_every, lr, l1, init, window, batch, max_batches, loss_on } => {
            let nl = cortiq_core::CmfModel::open_sharded(&model)?.arch().num_layers;
            let cfg = compress::dtgma_bake::Cfg { from: from_layer.unwrap_or(nl / 2), steps, eval_every, lr, l1, init, window, batch, max_batches };
            let r = compress::dtgma_bake::bake(
                std::path::Path::new(&model),
                &data,
                &loss_on,
                &cfg,
                std::path::Path::new(&output),
                compress::stages::cpu_budget(),
                &|l| println!("{l}"),
            )?;
            println!(
                "dtgma bake: held-out loss {:.4} → {:.4} (best step {} of {}); wrote {output}",
                r.nll_open, r.nll_best, r.step, r.steps
            );
            Ok(())
        }
        #[cfg(not(feature = "gpu"))]
        Commands::DtgmaBake { .. } => Err(anyhow::anyhow!("dtgma-bake needs a build with the gpu feature")),
        #[cfg(feature = "gpu")]
        Commands::NvgTrain {
            student,
            teacher,
            calib,
            tokens,
            window,
            batch,
            epochs,
            lr,
            topk,
            eval_windows,
            eval_every,
            clip,
            seed,
            output,
            train_signs,
            sign_opt,
            sign_tau,
            sign_lr,
            sign_warm,
            sign_b1,
            sign_b2,
            sign_eps,
            sign_clamp,
            sign_hyst,
            sign_max_step_flip,
            sign_max_total_flip,
            sign_seed,
            topk_cache,
            sign_state,
            sign_state_every,
            sign_state_replace,
            resume,
            budget_plan,
            budget_gram_cache,
            budget_rowweight,
            budget_fit_tokens,
            budget_sel_tokens,
            budget_jobs,
            mask_keep,
        } => {
            let signs = if train_signs {
                let opt = cortiq_engine::gpu_wgpu::train_sign::SignOpt::parse(&sign_opt).ok_or_else(|| anyhow::anyhow!("--sign-opt: fact | fact-mom | ref"))?;
                let lr = if sign_lr == "auto" { None } else { Some(sign_lr.parse::<f64>().map_err(|_| anyhow::anyhow!("--sign-lr: auto or a number"))?) };
                Some(nvgtrain::SignOpts {
                    opt,
                    tau: sign_tau,
                    lr,
                    warm: sign_warm,
                    b1: sign_b1,
                    b2: sign_b2,
                    eps: sign_eps,
                    clamp: sign_clamp,
                    hyst: sign_hyst,
                    max_step_flip: sign_max_step_flip,
                    max_total_flip: sign_max_total_flip,
                    seed: sign_seed.unwrap_or(seed),
                })
            } else {
                None
            };
            nvgtrain::run(&nvgtrain::NvgTrainOpts {
                student: &student,
                teacher: &teacher,
                calib: &calib,
                tokens,
                window,
                batch,
                epochs,
                lr,
                topk,
                eval_windows,
                eval_every,
                clip,
                seed,
                output: &output,
                signs,
                topk_cache: topk_cache.as_deref(),
                state_dir: sign_state.as_deref(),
                state_every: sign_state_every,
                state_replace: sign_state_replace,
                resume: resume.as_deref(),
                budget_plan: budget_plan.as_deref(),
                budget_gram_cache: budget_gram_cache.as_deref(),
                budget_rowweight,
                budget_fit_tokens,
                budget_sel_tokens,
                budget_jobs,
                mask_keep,
            })
        }
        #[cfg(not(feature = "gpu"))]
        Commands::NvgTrain { .. } => Err(anyhow::anyhow!("nvg-train needs a build with the gpu feature")),
        Commands::NvgBench {
            model,
            tensor,
            calib,
            fit_tokens,
            sel_tokens,
            bpw,
            kv_factor,
            paths,
            device,
            threads,
            rho,
            gram_cache,
            layer,
            jobs,
        } => {
            let paths = match paths.as_str() {
                "auto" => nvg::PathsChoice::Auto,
                "1" => nvg::PathsChoice::One,
                "2" => nvg::PathsChoice::Two,
                other => anyhow::bail!("--paths {other}: expected auto, 1 or 2"),
            };
            nvgbench::run(&nvgbench::BenchOpts {
                model: &model,
                tensor: tensor.as_deref(),
                calib: calib.as_deref(),
                fit_tokens,
                sel_tokens,
                bpw,
                kv_factor,
                paths,
                device: &device,
                threads,
                rho,
                gram_cache: gram_cache.as_deref(),
                layer,
                jobs,
            })
        }
        Commands::QuantizeGptq {
            input,
            calib,
            output,
            keep,
            tokens,
            lambda,
            codec,
            calib_model,
            window,
            tensor_quant,
            threads,
            act_order,
            hessians,
            calib_chunk_layers,
            q2tp_rule,
            q2tp_rule_tensor,
            q2tp_debias,
            q4tp_debias,
        } => {
            let codec = codec.to_ascii_lowercase();
            let q2tp_flags =
                q2tp_rule.is_some() || !q2tp_rule_tensor.is_empty() || q2tp_debias.is_some();
            let q2tp_debias = match q2tp_debias.as_deref().map(str::to_ascii_lowercase).as_deref() {
                None | Some("on" | "true" | "1") => true,
                Some("off" | "false" | "0") => false,
                Some(other) => anyhow::bail!("--q2tp-debias {other}: expected on or off"),
            };
            match codec.as_str() {
                "q4tp" | "q2tp" => {
                    let profile = convert::parse_quant(&codec)?;
                    anyhow::ensure!(
                        codec == "q2tp" || !q2tp_flags,
                        "--q2tp-rule / --q2tp-rule-tensor / --q2tp-debias apply to --codec q2tp"
                    );
                    convert::set_tensor_quant_overrides(&tensor_quant)?;
                    let threads = threads.unwrap_or_else(|| {
                        std::thread::available_parallelism()
                            .map(|n| n.get())
                            .unwrap_or(4)
                    });
                    let opts = gptq::GptqTiledOpts {
                        input: &input,
                        calib_model: calib_model.as_deref(),
                        calib: calib
                            .iter()
                            .map(|s| gptq::CalibSource::parse(s))
                            .collect::<anyhow::Result<Vec<_>>>()?,
                        output: &output,
                        tokens,
                        window,
                        lambda,
                        threads,
                        act_order,
                        hessians: hessians.as_deref(),
                        profile,
                        q2tp_rule: gptq::Q2tpRule::parse(q2tp_rule.as_deref().unwrap_or("placed"))?,
                        q2tp_rule_overrides: q2tp_rule_tensor
                            .iter()
                            .map(|s| gptq::parse_q2tp_rule_override(s))
                            .collect::<anyhow::Result<Vec<_>>>()?,
                        q2tp_debias,
                        q4tp_debias,
                        chunk_layers: calib_chunk_layers,
                    };
                    gptq::run_quantize_gptq_tiled(&opts)?;
                }
                "q1s" => {
                    anyhow::ensure!(
                        calib_model.is_none()
                            && tensor_quant.is_empty()
                            && !q2tp_flags
                            && !q4tp_debias
                            && calib_chunk_layers == 0,
                        "--calib-model / --tensor-quant / --calib-chunk-layers / --q2tp-* / --q4tp-debias apply to --codec q4tp|q2tp"
                    );
                    anyhow::ensure!(
                        calib.len() == 1,
                        "--codec q1s reads one --calib corpus"
                    );
                    gptq::run_quantize_gptq(&input, &calib[0], &output, keep, tokens, lambda)?
                }
                other => anyhow::bail!("--codec {other}: expected q1s, q4tp or q2tp"),
            }
            println!("✓ wrote {output}");
            Ok(())
        }
        Commands::Run {
            model,
            task,
            prompt,
            images,
            videos,
            video_fps,
            audios,
            system,
            speech_out,
            speech_mode,
            mm,
            image_max_pixels,
            reasoning_effort,
            max_tokens,
            skill,
            greedy,
            temperature,
            rep_penalty,
            top_p,
            top_k,
            min_p,
            presence_penalty,
            seed,
            raw,
            no_think,
            blend,
            route_dynamic,
            route_include_quarantine,
            confidence,
            trace,
            trace_json,
            state,
            o1,
            o1_m,
            o1_window,
            o1_sink,
            peer,
            peer_split,
            net_token,
            net_dtype,
            peer_head,
            peer_run_ahead,
            peer_prefill,
            gpus,
            lookup_mode,
        } => {
            let o1 = O1Flags {
                spec: o1,
                m: o1_m,
                w: o1_window,
                sink: o1_sink,
                rect: None,
            };
            cmd_run(
                &model,
                &task,
                prompt.as_deref(),
                &images,
                &videos,
                &audios,
                system.as_deref(),
                speech_out.as_deref().map(|p| (p, speech_mode.as_str())),
                &cortiq_engine::mimo_ingress::MediaOptions { companion: mm, image_max_pixels, video_fps },
                reasoning_effort.as_deref(),
                max_tokens,
                skill.as_deref(),
                greedy,
                temperature,
                rep_penalty,
                top_p,
                top_k,
                min_p,
                presence_penalty,
                seed,
                raw,
                no_think,
                blend.as_deref(),
                route_dynamic,
                route_include_quarantine,
                confidence,
                trace,
                trace_json,
                state.as_deref(),
                &o1,
                peer.as_deref(),
                peer_split,
                net_token.as_deref(),
                &net_dtype,
                peer_head,
                peer_run_ahead,
                peer_prefill,
                gpus,
                lookup_mode.as_deref(),
            )
            .await
        }
        Commands::Peers { wait, model } => {
            let mine = match model.as_deref() {
                Some(p) => {
                    let m = CmfModel::open_sharded(p)?;
                    Some((format!("{:016x}", m.dir_hash()), p.to_string()))
                }
                None => None,
            };
            eprintln!("listening {wait:.0}s for workers…");
            let found = cortiq_net::discover(std::time::Duration::from_secs_f64(wait))
                .map_err(|e| anyhow::anyhow!(e))?;
            let shown: Vec<_> = found
                .iter()
                .filter(|f| mine.as_ref().is_none_or(|(h, _)| &f.beacon.dir_hash == h))
                .collect();
            if shown.is_empty() {
                // Say which of the two silences this is.
                if found.is_empty() {
                    println!("no workers answered");
                } else {
                    println!(
                        "{} worker(s) answered, none holding {}",
                        found.len(),
                        mine.map(|(_, p)| p).unwrap_or_default()
                    );
                }
                return Ok(());
            }
            for f in shown {
                let b = &f.beacon;
                println!(
                    "{:<22} {} | {}L hidden {} | dir_hash {} | wire v{}{}{}",
                    f.peer_addr(),
                    b.model,
                    b.layers,
                    b.hidden,
                    b.dir_hash,
                    b.wire,
                    if b.token_required { " | token" } else { "" },
                    if b.wire == cortiq_net::WIRE_VERSION {
                        String::new()
                    } else {
                        format!(
                            " | INCOMPATIBLE, this build speaks v{}",
                            cortiq_net::WIRE_VERSION
                        )
                    },
                );
            }
            Ok(())
        }
        Commands::Worker {
            model,
            listen,
            token,
        } => cortiq_net::worker_serve(cortiq_net::WorkerConfig {
            model_path: model,
            listen,
            token,
        })
        .map_err(|e| anyhow::anyhow!(e)),
        Commands::Freeze {
            model,
            prompt,
            out,
            skill,
        } => cmd_freeze(&model, &prompt, &out, skill.as_deref()),
        Commands::Route { model, prompt } => cmd_route(&model, &prompt),
        Commands::ProbeUtility {
            models,
            prompts,
            prompts_jsonl,
            max_tokens,
            rep_penalty,
            bos,
            json,
            route,
            include_quarantine,
            lookup_mode,
        } => utility::cmd_probe_utility(utility::UtilityArgs {
            models: &models,
            prompts,
            prompts_jsonl,
            max_tokens: max_tokens.max(1),
            rep_penalty,
            bos,
            json,
            route,
            include_quarantine,
            lookup_mode,
        }),
        Commands::ProbeDialog {
            model,
            dialogs_jsonl,
            route,
            lookup_mode,
            include_quarantine,
            max_tokens,
            json,
        } => dialog::cmd_probe_dialog(dialog::DialogArgs {
            model: &model,
            dialogs_jsonl: &dialogs_jsonl,
            route,
            lookup_mode,
            include_quarantine,
            max_tokens,
            json,
        }),
        Commands::ProbeChoice {
            model,
            prompts_jsonl,
            candidates_jsonl,
            norm,
            scorer,
            bos,
            raw,
            route,
            include_quarantine,
            json,
        } => choice::cmd_probe_choice(choice::ChoiceArgs {
            model: &model,
            prompts_jsonl: &prompts_jsonl,
            candidates_jsonl: candidates_jsonl.as_deref(),
            norm: &norm,
            scorer: &scorer,
            bos,
            raw,
            route: route.as_deref(),
            include_quarantine,
            json,
        }),
        Commands::RouteEval {
            model,
            prompts_jsonl,
            expect,
            include_quarantine,
            router_only,
            json,
        } => knowledge::cmd_route_eval(knowledge::RouteEvalArgs {
            model: &model,
            prompts_jsonl: &prompts_jsonl,
            expect: &expect,
            include_quarantine,
            router_only,
            json,
        }),
        Commands::DumpLogits {
            model,
            prompts_jsonl,
            tokens,
            route,
            include_quarantine,
            out,
            path,
            lookup_mode,
        } => knowledge::cmd_dump_logits(knowledge::DumpArgs {
            model: &model,
            prompts_jsonl: &prompts_jsonl,
            tokens,
            route: route.as_deref(),
            include_quarantine,
            out: &out,
            path: path.as_deref(),
            lookup_mode: lookup_mode.as_deref(),
        }),
        Commands::LogitsCompare { a, b, only_indices } => {
            knowledge::cmd_logits_compare(&a, &b, only_indices.as_deref())
        }
        Commands::SkillGate {
            model,
            id,
            gate,
            status,
            allow_unmeasured,
        } => knowledge::cmd_skill_gate(&model, &id, &gate, &status, allow_unmeasured),
        Commands::GenomeVerify { f0, f1 } => knowledge::cmd_genome_verify(&f0, &f1),
        Commands::LookupBuild {
            base,
            entries,
            id,
            out,
            langs,
            name,
            on_duplicate,
            drop_keys,
            probe_prompts,
            suspicious_share,
            general_prompts,
            general_stop_out,
            policy,
        } => lookup::cmd_lookup_build(lookup::LookupBuildArgs {
            base: &base,
            entries: &entries,
            id: &id,
            out: &out,
            langs: langs.as_deref(),
            name: name.as_deref(),
            on_duplicate: &on_duplicate,
            drop_keys: drop_keys.as_deref(),
            probe_prompts: probe_prompts.as_deref(),
            suspicious_share,
            general_prompts: general_prompts.as_deref(),
            general_stop_out: general_stop_out.as_deref(),
            policy: policy.as_deref(),
        }),
        Commands::LookupPolicy {
            model,
            id,
            policy,
            keep_gate,
        } => lookup::cmd_lookup_policy(&model, &id, &policy, keep_gate),
        Commands::RouteFit {
            model,
            id,
            skill_prompts,
            general_prompts,
            phi_layer,
            rank,
            max,
            margin,
            target_fpr,
        } => lookup::cmd_route_fit(lookup::RouteFitArgs {
            model: &model,
            id: &id,
            skill_prompts: &skill_prompts,
            general_prompts: &general_prompts,
            phi_layer,
            rank,
            max,
            margin,
            target_fpr,
        }),
        Commands::GrowthEval {
            model,
            prompts_jsonl,
            max_tokens,
            json,
            indices_out,
            shell,
            growth,
            path,
        } => growth::cmd_growth_eval(growth::GrowthEvalArgs {
            model: &model,
            prompts_jsonl: &prompts_jsonl,
            max_tokens,
            json,
            indices_out: indices_out.as_deref(),
            shell: shell.as_deref(),
            growth: growth.as_deref(),
            path: path.as_deref(),
        }),
        Commands::ProbeRecall {
            models,
            pairs,
            distances,
            trials,
            seed,
            filler,
            json,
        } => {
            let distances: Vec<usize> = distances
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse::<usize>())
                .collect::<Result<_, _>>()
                .map_err(|e| anyhow::anyhow!("--distances: {e}"))?;
            recall::cmd_probe_recall(recall::RecallArgs {
                models: &models,
                pairs,
                distances,
                trials,
                seed,
                filler,
                json,
            })
        }
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
        } => {
            // The perplexity gate is the exactness contract: CPU == GPU bit
            // for bit, which needs the strict dot kernels. Generation keeps
            // the fast ones; an explicit CMF_SDOT still wins here.
            if std::env::var("CMF_SDOT").is_err() {
                unsafe { std::env::set_var("CMF_SDOT", "0") };
            }
            // The native Metal path is not held to the bit-parity contract
            // (10.217 vs the CPU/wgpu 10.224 on the 3B) — on macOS the
            // gate defaults to the CPU reference unless the caller picks a
            // backend explicitly. Linux/Windows keep the (proven-exact,
            // much faster) wgpu path.
            #[cfg(target_os = "macos")]
            if std::env::var("CMF_GPU").is_err() {
                unsafe { std::env::set_var("CMF_GPU", "0") };
            }
            cmd_ppl(
                &model,
                &file,
                tokens,
                skill.as_deref(),
                blend.as_deref(),
                route_dynamic,
                PplWindows {
                    windows,
                    window_len,
                },
                &O1Flags {
                    spec: o1,
                    m: o1_m,
                    w: o1_window,
                    sink: o1_sink,
                    rect: o1_rect,
                },
                o1_prefill,
            )
        }
        Commands::Info { model, tensors } => cmd_info(&model, tensors.as_deref()).await,
        Commands::RepairFlint { model, dry_run } => flint::cmd_repair_flint(&model, dry_run),
        Commands::FlintHybrid {
            q2_base,
            q4_source,
            output,
            layers,
            dry_run,
        } => flint_hybrid::cmd_flint_hybrid(&q2_base, &q4_source, &output, &layers, dry_run),
        Commands::Dequant {
            model,
            name,
            out,
            dtype,
            all,
        } => cmd_dequant(&model, &name, &out, &dtype, all),
        Commands::PatchTensor {
            model,
            sets,
            dtype,
            output,
            raw,
        } => cmd_patch_tensor(&model, &sets, dtype.as_deref(), output.as_deref(), raw),
        Commands::Story { model } => cmd_story(&model),
        Commands::Diff { a, b } => cmd_diff(&a, &b),
        Commands::Imagine {
            model_dir,
            prompt,
            layout,
            layout_file,
            height,
            width,
            steps,
            cfg,
            cfg_normalization,
            cfg_truncation,
            shift,
            uncond_skip_layers,
            slg_layers,
            slg_scale,
            slg_start,
            slg_stop,
            max_sequence_length,
            num_images,
            seed,
            images,
            text_encoder,
            vae,
            negative_prompt,
            scheduler,
            reference_size,
            out,
        } => {
            let arch = if std::path::Path::new(&model_dir).is_file() {
                CmfModel::open(&model_dir)
                    .map(|m| m.header.arch.arch_name.clone())
                    .unwrap_or_default()
            } else {
                String::new()
            };
            let zimage = arch == cortiq_engine::zimagegen::ARCH_NAME;
            let moss_image = arch == cortiq_engine::moss_imagegen::ARCH_NAME;
            let prompt = if moss_image {
                resolve_moss_prompt(prompt, layout, layout_file)?
            } else {
                anyhow::ensure!(
                    layout.is_none() && layout_file.is_none(),
                    "--layout/--layout-file apply only to MOSS-Image"
                );
                prompt.ok_or_else(|| anyhow::anyhow!("--prompt is required for this image model"))?
            };
            if arch == cortiq_engine::qwen_image21::ARCH_NAME {
                anyhow::ensure!(
                    text_encoder.is_none()
                        && vae.is_none()
                        && scheduler.is_none()
                        && cfg_normalization.is_none()
                        && cfg_truncation.is_none()
                        && shift.is_none()
                        && uncond_skip_layers.is_none()
                        && slg_layers.is_none()
                        && slg_scale.is_none()
                        && slg_start.is_none()
                        && slg_stop.is_none()
                        && max_sequence_length.is_none(),
                    "--text-encoder/--vae/--scheduler/--cfg-normalization/--cfg-truncation/--shift/--uncond-skip-layers/--slg-*/--max-sequence-length do not apply to Qwen-Image-2.1"
                );
                return cmd_qi21(
                    &model_dir,
                    &prompt,
                    Qi21Cli {
                        height,
                        width,
                        steps,
                        cfg,
                        num_images,
                        seed,
                        images,
                        negative_prompt,
                        reference_size,
                        out,
                    },
                );
            }
            if moss_image {
                anyhow::ensure!(
                    images.is_empty()
                        && text_encoder.is_none()
                        && vae.is_none()
                        && scheduler.is_none()
                        && cfg_normalization.is_none()
                        && cfg_truncation.is_none()
                        && reference_size == 1024,
                    "--image/--text-encoder/--vae/--scheduler/--cfg-normalization/--cfg-truncation/--reference-size do not apply to MOSS-Image"
                );
                cmd_moss_image(
                    &model_dir,
                    &prompt,
                    MossImageCli {
                        height,
                        width,
                        steps,
                        cfg,
                        shift,
                        uncond_skip_layers,
                        slg_layers,
                        slg_scale,
                        slg_start,
                        slg_stop,
                        max_sequence_length,
                        num_images,
                        seed,
                        negative_prompt,
                        out,
                    },
                )
            } else if zimage {
                anyhow::ensure!(
                    images.is_empty()
                        && text_encoder.is_none()
                        && vae.is_none()
                        && scheduler.is_none()
                        && uncond_skip_layers.is_none()
                        && slg_layers.is_none()
                        && slg_scale.is_none()
                        && slg_start.is_none()
                        && slg_stop.is_none()
                        && reference_size == 1024,
                    "--image/--text-encoder/--vae/--scheduler/--reference-size are Qwen Image options; --uncond-skip-layers/--slg-* are MOSS-Image options"
                );
                cmd_zimage(
                    &model_dir,
                    &prompt,
                    ZimageCli {
                        height,
                        width,
                        steps,
                        cfg,
                        cfg_normalization,
                        cfg_truncation,
                        shift,
                        max_sequence_length,
                        num_images,
                        seed,
                        negative_prompt,
                        out,
                    },
                )
            } else {
                anyhow::ensure!(
                    cfg_normalization.is_none()
                        && cfg_truncation.is_none()
                        && shift.is_none()
                        && uncond_skip_layers.is_none()
                        && slg_layers.is_none()
                        && slg_scale.is_none()
                        && slg_start.is_none()
                        && slg_stop.is_none()
                        && max_sequence_length.is_none()
                        && num_images == 1,
                    "--cfg-normalization/--cfg-truncation/--shift/--max-sequence-length/--num-images are Z-Image options; --uncond-skip-layers/--slg-* are MOSS-Image options"
                );
                cmd_imagine(
                    &model_dir,
                    &prompt,
                    height,
                    width,
                    steps,
                    cfg.unwrap_or(4.0),
                    seed,
                    out.as_deref().unwrap_or("out.ppm"),
                    &images,
                    text_encoder.as_deref(),
                    vae.as_deref(),
                    negative_prompt.as_deref(),
                    scheduler.as_deref(),
                    reference_size,
                )
            }
        }
        Commands::Vae { model, mode, input, output } => vaecmd::run(&model, &mode, &input, &output),
        Commands::Sam3Pack { src, quant, out } => sam3pack::pack(&src, &quant, &out),
        Commands::Segment {
            model,
            input,
            prompt,
            out,
            json,
            topk,
            thr,
            mask_thr,
        } => {
            let args = sam3seg::SegArgs {
                model,
                input,
                prompt,
                out,
                json,
                topk,
                thr,
                mask_thr,
            };
            if sam3seg::is_video_input(&args.input) {
                let res = sam3seg::run_video(&args)?;
                sam3seg::print_json_if_stdio(&res.json, args.json.as_deref())?;
                eprintln!(
                    "segment: video {} frame(s), {} track(s), {:.0} ms/frame -> {}",
                    res.frames,
                    res.tracks.len(),
                    res.mean_frame_ms,
                    res.overlay_dir.display()
                );
            } else {
                let res = sam3seg::run(&args)?;
                sam3seg::print_json_if_stdio(&res.json, args.json.as_deref())?;
                eprintln!(
                    "segment: {} object(s) of {}x{}, preprocess {:.0} ms + forward {:.0} ms (total {:.0} ms) -> {}",
                    res.detections.len(),
                    res.image_size.0,
                    res.image_size.1,
                    res.preprocess_ms,
                    res.forward_ms,
                    res.total_ms,
                    res.overlay_path.display()
                );
            }
            Ok(())
        }
        Commands::ImaginePack {
            root,
            quant,
            te_quant,
            te_embed_quant,
            te_keep,
            variant,
            dit_layers,
            no_source_sha,
            vis_quant,
            vae_quant,
            dit_keep,
            no_vision,
            out,
            component,
            bundle,
        } => {
            let zroot = std::path::Path::new(&root);
            if !bundle && component.is_none() && mossimagepack::is_moss_image_root(zroot) {
                anyhow::ensure!(
                    variant.is_none() && dit_layers.is_none(),
                    "--variant/--dit-layers are Z-Image options"
                );
                anyhow::ensure!(
                    vis_quant.is_none() && vae_quant.is_none() && !no_vision,
                    "--vis-quant/--vae-quant/--no-vision are Qwen-Image-2.1 options"
                );
                return mossimagepack::pack(
                    zroot,
                    &out,
                    &mossimagepack::PackOpts {
                        dit: zimagepack::parse_codec(
                            quant.as_deref().unwrap_or(mossimagepack::DEFAULT_DIT_CODEC),
                        )?,
                        dit_keep: dit_keep
                            .into_iter()
                            .filter(|key| key != "none")
                            .collect(),
                        te: zimagepack::parse_codec(
                            te_quant.as_deref().unwrap_or(mossimagepack::DEFAULT_TE_CODEC),
                        )?,
                        te_embed: te_embed_quant
                            .as_deref()
                            .map(zimagepack::parse_codec)
                            .transpose()?,
                        te_keep: te_keep.into_iter().filter(|key| key != "none").collect(),
                        source_sha: !no_source_sha,
                    },
                );
            }
            if !bundle && component.is_none() && qi21pack::is_qi21_root(zroot) {
                anyhow::ensure!(
                    te_embed_quant.is_none() && variant.is_none() && dit_layers.is_none(),
                    "--te-embed-quant/--variant/--dit-layers are Z-Image options"
                );
                return qi21pack::pack(
                    zroot,
                    &out,
                    &qi21pack::PackOpts {
                        dit: zimagepack::parse_codec(
                            quant.as_deref().unwrap_or(qi21pack::DEFAULT_DIT_CODEC),
                        )?,
                        te: zimagepack::parse_codec(
                            te_quant.as_deref().unwrap_or(qi21pack::DEFAULT_TE_CODEC),
                        )?,
                        vis: zimagepack::parse_codec(
                            vis_quant.as_deref().unwrap_or(qi21pack::DEFAULT_VIS_CODEC),
                        )?,
                        vae: zimagepack::parse_codec(
                            vae_quant.as_deref().unwrap_or(qi21pack::DEFAULT_VAE_CODEC),
                        )?,
                        dit_keep: qi21pack::parse_keep(&dit_keep),
                        te_keep: qi21pack::parse_keep(&te_keep),
                        vision: !no_vision,
                        source_sha: !no_source_sha,
                    },
                );
            }
            anyhow::ensure!(
                vis_quant.is_none() && vae_quant.is_none() && dit_keep.is_empty() && !no_vision,
                "--vis-quant/--vae-quant/--dit-keep/--no-vision are Qwen-Image-2.1 options"
            );
            if bundle {
                anyhow::ensure!(
                    component.is_none(),
                    "--bundle cannot be combined with --component"
                );
                qwen_imagepack::bundle(&root, &out)
            } else if let Some(component) = component {
                qwen_imagepack::pack(&root, &component, quant.as_deref().unwrap_or("q4t"), &out)
            } else if zimagepack::is_zimage_root(zroot) {
                zimagepack::pack(
                    zroot,
                    &out,
                    &zimagepack::PackOpts {
                        dit: zimagepack::parse_codec(
                            quant.as_deref().unwrap_or(zimagepack::DEFAULT_DIT_CODEC),
                        )?,
                        te: zimagepack::parse_codec(
                            te_quant.as_deref().unwrap_or(zimagepack::DEFAULT_TE_CODEC),
                        )?,
                        te_embed: te_embed_quant
                            .as_deref()
                            .map(zimagepack::parse_codec)
                            .transpose()?,
                        te_keep: if te_keep.is_empty() {
                            zimagepack::DEFAULT_TE_KEEP.iter().map(|s| s.to_string()).collect()
                        } else {
                            te_keep.into_iter().filter(|k| k != "none").collect()
                        },
                        layers: dit_layers,
                        variant,
                        source_sha: !no_source_sha,
                    },
                )
            } else {
                anyhow::ensure!(
                    te_quant.is_none()
                        && te_embed_quant.is_none()
                        && te_keep.is_empty()
                        && variant.is_none()
                        && dit_layers.is_none(),
                    "--te-quant/--variant/--dit-layers are Z-Image options"
                );
                imagepack::cmd_imagine_pack(&root, quant.as_deref().unwrap_or("q4t"), &out)
            }
        }
        Commands::Music {
            model,
            prompt,
            lyrics,
            seconds,
            steps,
            seed,
            out,
        } => music::cmd_music(&model, &prompt, &lyrics, seconds, steps, seed, &out),
        Commands::LtxDecode {
            model,
            latent,
            out_dir,
            out_tensors,
            gate,
            dump_stages,
        } => ltxcmd::cmd_ltx_decode(ltxcmd::DecodeArgs {
            model: &model,
            latent: &latent,
            out_dir: out_dir.as_deref(),
            out_tensors: out_tensors.as_deref(),
            gate,
            dump_stages: dump_stages.as_deref(),
        }),
        Commands::LtxDit {
            model,
            oracle,
            gate,
            dump,
        } => ltxcmd::cmd_ltx_dit(ltxcmd::DitArgs {
            model: &model,
            oracle: &oracle,
            gate,
            dump: dump.as_deref(),
        }),
        Commands::LtxRender {
            model,
            context,
            height,
            width,
            frames,
            fps,
            seed,
            out_dir,
            out_y4m,
            out_latent,
            skip_decode,
        } => ltxcmd::cmd_ltx_render(ltxcmd::RenderArgs {
            model: &model,
            context: &context,
            height,
            width,
            frames,
            fps,
            seed,
            out_dir: out_dir.as_deref(),
            out_y4m: out_y4m.as_deref(),
            out_latent: out_latent.as_deref(),
            skip_decode,
        }),
        Commands::LtxEncode {
            model,
            prompt,
            oracle,
            out,
        } => ltxcmd::cmd_ltx_encode(ltxcmd::EncodeArgs {
            model: &model,
            prompt: &prompt,
            oracle: oracle.as_deref(),
            out: out.as_deref(),
        }),
        Commands::LtxVideo {
            model,
            prompt,
            height,
            width,
            frames,
            fps,
            seed,
            two_stage,
            out,
            out_dir,
            out_latent,
            out_audio,
            image,
            video,
            video_to_audio,
            audio_in,
            video_strength,
            no_context_cache,
            steps,
            steps2,
            lora,
            lora_strength,
            refs,
            ref_frames,
        } => ltxcmd::cmd_ltx_video(ltxcmd::VideoArgs {
            model: &model,
            two_stage,
            prompt: &prompt,
            height,
            width,
            frames,
            fps,
            seed,
            out: out.as_deref(),
            out_dir: out_dir.as_deref(),
            out_latent: out_latent.as_deref(),
            out_audio: out_audio.as_deref(),
            image: image.as_deref(),
            video: video.as_deref(),
            video_to_audio,
            audio_in: audio_in.as_deref(),
            video_strength,
            no_context_cache,
            steps,
            steps2,
            lora: lora.as_deref(),
            lora_strength,
            refs,
            ref_frames,
        }),
        Commands::LtxAudio {
            model,
            latent,
            out,
            stats,
            oracle,
        } => ltxcmd::cmd_ltx_audio(ltxcmd::AudioArgs {
            model: &model,
            latent: &latent,
            out: &out,
            stats,
            oracle: oracle.as_deref(),
        }),
        Commands::LtxPack {
            out,
            carry,
            dit,
            te,
            video_vae,
            audio_vae,
            spatial_upscaler,
            temporal_upscaler,
            duration_head,
            quant,
            vae_quant,
            min_q4tp,
        } => ltxpack::cmd_ltx_pack(ltxpack::LtxPackArgs {
            out: &out,
            carry: carry.as_deref(),
            dit: dit.as_deref(),
            te: te.as_deref(),
            video_vae: video_vae.as_deref(),
            audio_vae: audio_vae.as_deref(),
            spatial_upscaler: spatial_upscaler.as_deref(),
            temporal_upscaler: temporal_upscaler.as_deref(),
            duration_head: duration_head.as_deref(),
            quant: &quant,
            vae_quant: &vae_quant,
            min_q4tp,
        }),
        Commands::AnimatePack {
            out,
            carry,
            dit,
            lora,
            time_embedder,
            lora_scale,
            te,
            te_layers,
            clip_proj,
            clip_proj_vis,
            music_vae,
            music_dit,
            music_te,
            video_vae,
            audio_vae,
            tokenizer,
            quant,
            vvae_heads,
            vision,
            vis_heads,
            vis_deepstack,
        } => videopack::cmd_animate_pack(videopack::PackArgs {
            out: &out,
            carry: carry.as_deref(),
            dit: dit.as_deref(),
            lora: lora.as_deref(),
            time_embedder: time_embedder.as_deref(),
            lora_scale,
            te: te.as_deref(),
            te_layers,
            clip_proj: clip_proj.as_deref(),
            clip_proj_vis: clip_proj_vis.as_deref(),
            music_vae: music_vae.as_deref(),
            music_dit: music_dit.as_deref(),
            music_te: music_te.as_deref(),
            video_vae: video_vae.as_deref(),
            audio_vae: audio_vae.as_deref(),
            tokenizer: tokenizer.as_deref(),
            quant: &quant,
            vvae_heads,
            vision: vision.as_deref(),
            vis_heads,
            vis_deepstack: vis_deepstack
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()?,
        }),
        Commands::Animate {
            model,
            prompt,
            width,
            height,
            frames,
            steps,
            seed,
            stock_sampler,
            quality,
            first_frame,
            last_frame,
            video,
            video_stride,
            stream_chunk,
            stream_sink,
            stream_window,
            upscale,
            upscale_by,
            lora,
            lora_strength,
            out,
            frames_dir,
        } => cmd_animate(
            &model,
            &prompt,
            cortiq_engine::videogen::AnimParams {
                width,
                height,
                frames,
                steps,
                seed,
                stock_sampler,
                first_frame: first_frame.as_deref().map(read_ppm).transpose()?,
                last_frame: last_frame.as_deref().map(read_ppm).transpose()?,
                mid_frames: match video.as_deref() {
                    None => Vec::new(),
                    Some(dir) => read_frame_dir(dir, video_stride.max(1), frames)?,
                },
                stream_chunk,
                stream_sink,
                stream_window,
                upscale,
                upscale_by,
                lora,
                lora_strength,
                ..Default::default()
            },
            quality,
            &out,
            frames_dir.as_deref(),
        ),
        Commands::Explain {
            model,
            prompt,
            top,
            skill,
            route_include_quarantine,
        } => cmd_explain(
            &model,
            &prompt,
            top,
            skill.as_deref(),
            route_include_quarantine,
        ),
        Commands::Calibrate {
            model,
            file,
            skill,
            tokens,
        } => cmd_calibrate(&model, &file, skill.as_deref(), tokens),
        Commands::Masks { model } => cmd_masks(&model).await,
        Commands::Bench {
            model,
            task,
            tokens,
            json,
            ctx,
            gpus,
            peer,
            peer_split,
            net_token,
            net_dtype,
            peer_head,
            core,
            ignore_eos,
            o1,
            o1_m,
            o1_window,
            o1_sink,
            bw,
        } => {
            if bw {
                return cmd_bench_bw(json, &model);
            }
            let o1 = O1Flags {
                spec: o1,
                m: o1_m,
                w: o1_window,
                sink: o1_sink,
                rect: None,
            };
            cmd_bench(
                &model,
                &task,
                tokens,
                ctx,
                &o1,
                json,
                core,
                ignore_eos,
                peer.as_deref(),
                peer_split,
                net_token.as_deref(),
                &net_dtype,
                peer_head,
                gpus,
            )
            .await
        }
        Commands::Skill { cmd } => match cmd {
            SkillCmd::Add {
                model,
                from,
                id,
                name,
                layers,
                tensors,
                prompts,
                phi_layer,
                rank,
                quality,
                quality_tokens,
                decision_threshold,
                min_delta,
                skill_quant,
                mean_bits,
                sparse,
                output,
                hf_token,
            } => skill::run_skill_add(
                &model,
                &from,
                &id,
                name.as_deref(),
                &layers,
                skill::Families::parse(&tensors)?,
                prompts.as_deref(),
                phi_layer,
                rank,
                quality.as_deref(),
                quality_tokens,
                decision_threshold,
                min_delta,
                skill_quant.as_deref(),
                mean_bits,
                sparse,
                output.as_deref(),
                hf_token.as_deref(),
            ),
            SkillCmd::List { model } => skill::run_skill_list(&model),
            SkillCmd::Export {
                specialist,
                base,
                id,
                name,
                output,
            } => skill::run_skill_export(&specialist, &base, &id, name.as_deref(), &output),
            SkillCmd::Apply {
                base,
                skill,
                output,
                force,
            } => skill::run_skill_apply(&base, &skill, &output, force),
            SkillCmd::Bake {
                model,
                files,
                held_files,
                output,
                steps_a,
                steps_b,
                lr_a,
                lr_b,
                eval_every,
                fcd_layers,
                chunk,
                held,
                calib_chunks,
                batch,
                fcd_batch,
                focus_tokens,
                record_start,
                answer_after,
                answer_end,
                drop_oversize,
                length_buckets,
                mask_from,
                soft_fold,
                no_defrag,
                mask_name,
                fcd_attn,
                checkpoint_dir,
                held_tags,
                target_sparsity,
                l1_aggression,
                l1_init,
                l1_step,
                tau,
                mask_init,
                softplus_l1,
                checkpoint_accuracy,
                checkpoint_min_accuracy,
                checkpoint_min_balanced_accuracy,
                checkpoint_raw_priority,
                ffn_align,
                uniform_inter,
                archaeology_report,
                analysis_only,
                skip_runtime_gate,
            } => skill::run_skill_bake(
                &model,
                &files,
                &held_files,
                &output,
                steps_a,
                steps_b,
                lr_a,
                lr_b,
                eval_every,
                fcd_layers,
                chunk,
                held,
                calib_chunks,
                batch,
                fcd_batch,
                focus_tokens.as_deref(),
                &record_start,
                answer_after.as_deref(),
                answer_end.as_deref(),
                drop_oversize,
                length_buckets,
                mask_from.as_deref(),
                soft_fold,
                no_defrag,
                &mask_name,
                fcd_attn,
                checkpoint_dir.as_deref(),
                held_tags.as_deref(),
                target_sparsity,
                l1_aggression,
                l1_init,
                l1_step,
                tau,
                mask_init,
                softplus_l1,
                checkpoint_accuracy,
                checkpoint_min_accuracy,
                checkpoint_min_balanced_accuracy,
                checkpoint_raw_priority,
                ffn_align,
                uniform_inter,
                archaeology_report.as_deref(),
                analysis_only,
                skip_runtime_gate,
            ),
        },
        Commands::Compress(args) => compress::run(args),
        Commands::TeacherData { cmd } => match cmd {
            TeacherDataCmd::Generate {
                teacher,
                prompts,
                output,
                max_tokens,
                samples,
                seed,
                split_seed,
                split,
                topk,
                raw,
                no_think,
                keep_truncated,
                disable_mtp,
                limit,
                shard,
                sampling,
                tokenizer,
                api,
            } => cmd_teacher_data_generate(
                &teacher,
                &prompts,
                &output,
                TeacherDataGenerateOptions {
                    max_tokens,
                    samples,
                    seed,
                    split_seed: split_seed.unwrap_or(seed),
                    split,
                    topk,
                    raw,
                    no_think,
                    keep_truncated,
                    disable_mtp,
                    limit,
                    shard: shard.as_deref().map(parse_shard).transpose()?,
                    sampling,
                    tokenizer,
                    api,
                },
            ),
            TeacherDataCmd::Merge { parts, output } => cmd_teacher_data_merge(&parts, &output),
            TeacherDataCmd::Inspect {
                tdata,
                verify,
                json,
            } => cmd_teacher_data_inspect(&tdata, verify, json),
            TeacherDataCmd::Import {
                source,
                output,
                tokenizer,
                split,
                seed,
            } => cmd_teacher_data_import(&source, &output, &tokenizer, &split, seed),
        },
        Commands::Gpu => {
            // `gpu_wgpu` is behind the `gpu` feature, and this arm called
            // into it unconditionally — so the CPU-only build the manifest
            // documents (`--no-default-features`) did not compile at all.
            #[cfg(feature = "gpu")]
            {
                for line in cortiq_engine::gpu_wgpu::adapter_report() {
                    println!("  {line}");
                }
                // What a round trip to this device costs with nothing in
                // it. On a laptop this is microseconds and the per-op
                // path is fine; where it is milliseconds, every kernel
                // number is really a latency number and tuning shaders
                // cannot help.
                if let Some((empty, one)) = cortiq_engine::gpu_wgpu::roundtrip_bench(30) {
                    println!(
                        "  round trip to the device: empty submit {empty:.2} ms · submit+dispatch+readback {one:.2} ms"
                    );
                }
                // CMF_GPU_DISPATCH_BENCH=1: what a dispatch costs, and whether
                // the cost is the launch or the barrier between two that touch
                // the same buffer. Every fusion decision hangs on which.
                if std::env::var("CMF_GPU_DISPATCH_BENCH").is_ok_and(|v| v != "0") {
                    for line in cortiq_engine::gpu_wgpu::dispatch_bench() {
                        println!("  {line}");
                    }
                }
            }
            #[cfg(not(feature = "gpu"))]
            println!("  built without the `gpu` feature — CPU only");
            Ok(())
        }
        Commands::Verify { model } => cmd_verify(&model).await,
        Commands::MigrateEmbryoBits { files, dry_run } => cmd_migrate_embryo_bits(&files, dry_run),
        Commands::Sign { model, key } => sign::cmd_sign(&model, &key),
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
            polish_only,
            teacher,
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
            polish_only,
            teacher.as_deref(),
        ),
    };
    // Stop accepting late tracing events before the owned GPU shutdown path:
    // timestamp/query destruction can enqueue driver work while contexts are
    // being dropped, and a logger teardown callback must not race that path.
    TRACING_LIVE.store(false, AtomicOrdering::Relaxed);
    // Release Vulkan contexts while Tokio and tracing are still alive.  The
    // engine keeps them resident during inference but must drain/drop them
    // before the NVIDIA background worker reaches process teardown.
    #[cfg(feature = "gpu")]
    cortiq_engine::gpu_wgpu::shutdown();
    result
}

/// Inspect a finalized teacher-data directory through the bounded streaming
/// reader.  The JSON shape deliberately contains only v1 fields owned by the
/// reader rather than the producer's open-ended manifest metadata, so scripts
/// have a stable contract even as producer provenance grows.
fn cmd_teacher_data_inspect(tdata_path: &str, verify: bool, json: bool) -> anyhow::Result<()> {
    let dataset = tdata::Dataset::open(tdata_path)?;
    // `inspect` streams every record, its fixed-width index row and its own
    // token payload in lockstep.  It is already an integrity check, while
    // `--verify` intentionally provides a distinct, explicit second pass.
    let summary = dataset.inspect()?;
    if verify {
        dataset.verify()?;
    }

    let split_json = |split: tdata::TdataSplitSummary| {
        serde_json::json!({
            "records": split.records,
            "usable": split.usable,
            "rejected": split.rejected,
            "tokens": split.tokens,
        })
    };
    let path = dataset.root().to_string_lossy().into_owned();
    let report = serde_json::json!({
        "schema": "cmf-tdata-inspect/v1",
        "path": path,
        "manifest": {
            "format": dataset.manifest.format.as_str(),
            "version": dataset.manifest.version,
            "complete": dataset.manifest.complete,
        },
        "integrity": {
            "stream_validated": true,
            "extra_verify_pass": verify,
        },
        "records": {
            "total": summary.records,
            "usable": summary.usable,
            "rejected": summary.rejected,
        },
        "tokens": summary.tokens,
        "splits": {
            "train": split_json(summary.train),
            "val": split_json(summary.val),
            "test": split_json(summary.test),
        },
    });

    if json {
        // One compact object makes the command safe in JSONL-style automation.
        println!("{}", serde_json::to_string(&report)?);
    } else {
        println!("schema: cmf-tdata-inspect/v1");
        println!("path: {tdata_path}");
        println!(
            "manifest: {} v{} ({})",
            dataset.manifest.format,
            dataset.manifest.version,
            if dataset.manifest.complete {
                "complete"
            } else {
                "incomplete"
            }
        );
        println!(
            "records: total={} usable={} rejected={}",
            summary.records, summary.usable, summary.rejected
        );
        println!("tokens: {}", summary.tokens);
        for (name, split) in [
            ("train", summary.train),
            ("val", summary.val),
            ("test", summary.test),
        ] {
            println!(
                "split.{name}: records={} usable={} rejected={} tokens={}",
                split.records, split.usable, split.rejected, split.tokens
            );
        }
        println!(
            "integrity: stream-validated{}",
            if verify {
                "; explicit verify pass complete"
            } else {
                ""
            }
        );
    }
    Ok(())
}

/// `cortiq teacher-data import`: a foreign answer cache as `.tdata`, its
/// text tokenized by the tokenizer of the model the data will compress.
/// `teacher-data generate openrouter:MODEL …`: the API answers the prompts
/// into `<output>.answers.jsonl` (resumable), then the answers become teacher
/// data through `--tokenizer`'s tokenizer and chat template.
fn teacher_data_generate_api(
    api: &teacher_api::ApiTeacher,
    spec: &str,
    prompts_path: &str,
    output_path: &str,
    options: &TeacherDataGenerateOptions,
) -> anyhow::Result<()> {
    let tokenizer = options
        .tokenizer
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("teacher-data generate {spec}: an API teacher needs --tokenizer MODEL.cmf (the model the data is for)"))?;
    anyhow::ensure!(output_path.ends_with(".tdata"), "teacher-data generate: output must end in .tdata");
    anyhow::ensure!(!std::path::Path::new(output_path).exists(), "teacher-data generate: {output_path} already exists");
    let (mut loaded, _) = parse_hashed_teacher_data_prompts(prompts_path)?;
    if let Some(limit) = options.limit {
        loaded.prompts.truncate(limit);
    }
    anyhow::ensure!(!loaded.prompts.is_empty(), "teacher-data generate: no prompts");
    let sampling = teacher_api::ApiSampling::parse(&options.sampling)?;
    let cache = std::path::PathBuf::from(format!("{output_path}.answers.jsonl"));
    teacher_api::answer_all(
        api,
        &loaded.prompts,
        options.samples,
        options.max_tokens,
        &sampling,
        !options.no_think,
        options.seed,
        options.api.api_concurrency,
        &cache,
    )?;
    cmd_teacher_data_import_as(
        cache.to_str().unwrap_or_default(),
        output_path,
        tokenizer,
        &options.split,
        options.split_seed,
        Some(serde_json::json!({
            "kind": "api",
            "model": api.model,
            "base": api.base,
            "extra": options.api.api_extra,
            "directory_hash": format!("api:{}", api.model),
            "sampling": options.sampling,
            "max_tokens": options.max_tokens,
            "thinking": !options.no_think,
            "note": "answers of the API model, tokenized with the student's tokenizer: text targets",
        })),
    )
}

fn cmd_teacher_data_import(
    source: &str,
    output: &str,
    tokenizer_model: &str,
    split: &str,
    seed: u64,
) -> anyhow::Result<()> {
    cmd_teacher_data_import_as(source, output, tokenizer_model, split, seed, None)
}

/// [`cmd_teacher_data_import`] with the manifest's teacher described by the caller.
fn cmd_teacher_data_import_as(
    source: &str,
    output: &str,
    tokenizer_model: &str,
    split: &str,
    seed: u64,
    teacher: Option<serde_json::Value>,
) -> anyhow::Result<()> {
    anyhow::ensure!(output.ends_with(".tdata"), "teacher-data import: output must end in .tdata");
    anyhow::ensure!(!std::path::Path::new(output).exists(), "teacher-data import: {output} already exists");
    // Only the tokenizer and template are needed: keep the device out.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let ratios = parse_teacher_data_split(split)?;
    let rows = teacher_data::parse_import_source(std::path::Path::new(source))?;
    anyhow::ensure!(!rows.is_empty(), "teacher-data import: {source} has no rows");
    let model = Arc::new(CmfModel::open_sharded(tokenizer_model)?);
    let pipeline = Pipeline::from_model(&model, teacher_data_greedy_sampler(seed))?;
    let tok = &pipeline.tokenizer;
    let eos = tok.eos_token_id.ok_or_else(|| anyhow::anyhow!("teacher-data import: the tokenizer has no EOS id"))?;
    // A chat answer ends with the turn terminator the model emits, which for
    // ChatML is `<|im_end|>`, not the document EOS.
    let im_end = {
        let ids = tok.encode("<|im_end|>");
        (ids.len() == 1).then(|| ids[0])
    };
    let chatml = tok.chat_template.as_deref().is_some_and(|t| t.contains("<|im_end|>"));
    let think_end = teacher_data::think_end_id(tok);
    // Repeated prompts are samples of one prompt (one split for all of them).
    let mut first: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut unique: Vec<teacher_data::TeacherPrompt> = Vec::new();
    let mut sample_of = Vec::with_capacity(rows.len());
    for r in &rows {
        let k = first.entry(r.prompt.id.clone()).or_insert_with(|| {
            unique.push(r.prompt.clone());
            unique.len() - 1
        });
        let n = sample_of.iter().filter(|(u, _)| u == k).count() as u32;
        sample_of.push((*k, n));
    }
    let splits = teacher_data::assign_stratified_splits(&unique, ratios, seed)?;
    let vocab = model.vocab.as_deref().ok_or_else(|| anyhow::anyhow!("teacher-data import: {tokenizer_model} has no embedded tokenizer"))?;
    let tokenizer_config_bytes = serde_json::to_vec(&model.header.tokenizer_config)?;
    let template_sha256 = tok.chat_template.as_deref().map(|t| knowledge::sha256_hex(t.as_bytes()));
    let source_sha = knowledge::sha256_hex(&std::fs::read(source)?);
    let manifest = serde_json::json!({
        "format": "cmf-tdata",
        "version": 1,
        "complete": false,
        "cortiq": {"version": env!("CARGO_PKG_VERSION")},
        "teacher": teacher.clone().unwrap_or_else(|| serde_json::json!({
            "kind": "imported",
            "file": std::path::Path::new(source).file_name().and_then(|n| n.to_str()),
            "directory_hash": format!("import:{}", &source_sha[..16]),
            "source_sha256": source_sha,
            "note": "answers written by another model; usable as text targets, not as this model's distribution",
        })),
        "tokenizer": {
            "vocab_size": tok.vocab_size(),
            "eos_id": eos,
            "vocab_sha256": knowledge::sha256_hex(vocab),
            "config_sha256": knowledge::sha256_hex(&tokenizer_config_bytes),
            "template_sha256": template_sha256,
            "from": std::path::Path::new(tokenizer_model).file_name().and_then(|n| n.to_str()),
        },
        "template": {"mode": "imported-text"},
        "seed": seed,
        "max_tokens": null,
        "topk": {"k": 0, "positions": "none"},
        "prompts": {"source": source, "sha256": source_sha, "count": unique.len(), "rows": rows.len()},
        "split": {"rule": "stratified-hash", "rule_version": 1, "seed": seed, "ratios": [ratios.train, ratios.val, ratios.test]},
        "generation": {"mode": "imported", "samples": "per-row"},
    });
    let mut writer = tdata::TdataWriter::create(output, manifest)?;
    for (p, &s) in unique.iter().zip(&splits) {
        writer.append_prompt(&teacher_data::canonical_prompt_json(p, s))?;
    }
    for (r, &(u, sample)) in rows.iter().zip(&sample_of) {
        // an answer cut off inside its reasoning has no `</think>` yet: the
        // API path marks it so the prompt still opens the reasoning
        let open_think = r.prompt.meta.get("open_think").and_then(|v| v.as_bool()) == Some(true);
        let has_think = open_think || r.response_text.contains("</think>");
        let prompt_ids = match &r.prefix_text {
            Some(prefix) => tok.encode(prefix),
            None => teacher_data::render_prompt(tok, &r.prompt, false, !has_think)?.token_ids,
        };
        let rendered = teacher_data::RenderedPrompt { token_ids: prompt_ids, template_applied: true, thinking: Some(has_think) };
        let mut generated = tok.encode(&r.response_text);
        let is_chatml = chatml || r.prefix_text.as_deref().is_some_and(|p| p.contains("<|im_start|>"));
        // an API answer cut by max_tokens is a truncated record (kept); it
        // gets no end token — the text did not end there
        let cut = r.prompt.meta.get("finish").and_then(|f| f.as_str()) == Some("length");
        if !cut {
            generated.push(if is_chatml { im_end.unwrap_or(eos) } else { eos });
        }
        let (think_n, answer_n) = match think_end {
            Some(end) if has_think => teacher_data::split_think_ids(&generated, end),
            _ => (0, generated.len()),
        };
        let think_text = (think_n > 0).then(|| tok.decode(&generated[..think_n]));
        let content = tok.decode(&generated[think_n..]);
        let response = teacher_data::ResponseParts::segmented(think_text, content.clone(), think_n as u32, answer_n as u32);
        let engine_finish = if cut { "max_tokens" } else { "stop" };
        let (finish, quality) = teacher_data::final_record_quality(engine_finish, &content, &generated, true);
        let prepared = teacher_data::build_canonical_record(teacher_data::CanonicalRecordInput {
            prompt: &unique[u],
            split: splits[u],
            sample,
            seed,
            rendered: &rendered,
            generated_token_ids: &generated,
            response,
            finish: &finish,
            quality,
        })?;
        writer.append_record(prepared.record, &prepared.tokens)?;
    }
    let stats = writer.stats();
    writer.finalize()?;
    println!(
        "teacher-data import: {output}: {} records ({} usable) for {} prompts from {source}",
        stats.records, stats.usable, unique.len()
    );
    Ok(())
}

/// Options kept together so the command dispatch remains readable and the
/// generation implementation cannot accidentally pick up a CLI default that
/// differs from the recorded manifest.
#[derive(Debug)]
pub(crate) struct TeacherDataGenerateOptions {
    pub(crate) max_tokens: usize,
    pub(crate) samples: u32,
    pub(crate) seed: u64,
    pub(crate) split_seed: u64,
    pub(crate) split: String,
    pub(crate) topk: u32,
    pub(crate) raw: bool,
    pub(crate) no_think: bool,
    pub(crate) keep_truncated: bool,
    pub(crate) disable_mtp: bool,
    pub(crate) limit: Option<usize>,
    /// `--shard I/N`: (I, N)
    pub(crate) shard: Option<(u32, u32)>,
    /// `--sampling`: greedy | file | K=V,…
    pub(crate) sampling: String,
    /// An API teacher's token ids come from this model's tokenizer.
    pub(crate) tokenizer: Option<String>,
    /// An API teacher's server, extra fields and concurrency.
    pub(crate) api: teacher_api::ApiArgs,
}

/// The decoding settings of `teacher-data generate --sampling`, as the
/// manifest records them (seed excluded — it is per record).
fn teacher_data_sampling(spec: &str, model: &CmfModel) -> anyhow::Result<serde_json::Value> {
    let greedy = serde_json::json!({
        "temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
        "repetition_penalty": 1.0, "presence_penalty": 0.0,
    });
    let mut v = greedy.clone();
    let pairs: Vec<(String, f64)> = match spec {
        "greedy" => return Ok(greedy),
        "file" => {
            let f = model.header.provenance.as_ref().and_then(|p| p.get("sampling")).ok_or_else(|| {
                anyhow::anyhow!("--sampling file: the teacher carries no recommended sampling; give it as temperature=…,top_p=…")
            })?;
            f.as_object()
                .into_iter()
                .flatten()
                .filter_map(|(k, x)| x.as_f64().map(|x| (k.clone(), x)))
                .collect()
        }
        list => list
            .split(',')
            .map(|kv| {
                let (k, x) = kv.split_once('=').ok_or_else(|| anyhow::anyhow!("--sampling {list:?}: expected KEY=VALUE pairs"))?;
                Ok((k.trim().to_string(), x.trim().parse::<f64>()?))
            })
            .collect::<anyhow::Result<_>>()?,
    };
    for (k, x) in pairs {
        anyhow::ensure!(v.get(&k).is_some(), "--sampling: unknown key {k:?}");
        v[k] = serde_json::json!(x);
    }
    Ok(v)
}

fn sampler_from(v: &serde_json::Value, seed: u64) -> SamplerConfig {
    let f = |k: &str| v[k].as_f64().unwrap_or(0.0) as f32;
    SamplerConfig {
        temperature: f("temperature"),
        top_p: f("top_p"),
        top_k: v["top_k"].as_f64().unwrap_or(0.0) as u32,
        min_p: f("min_p"),
        repetition_penalty: f("repetition_penalty"),
        presence_penalty: f("presence_penalty"),
        seed: Some(seed),
        ..Default::default()
    }
}

fn parse_shard(s: &str) -> anyhow::Result<(u32, u32)> {
    let (i, n) = s.split_once('/').ok_or_else(|| anyhow::anyhow!("--shard {s:?}: expected I/N"))?;
    let (i, n): (u32, u32) = (i.trim().parse()?, n.trim().parse()?);
    anyhow::ensure!(n > 0 && i < n, "--shard {s:?}: need 0 <= I < N");
    Ok((i, n))
}

/// Shard of a prompt id: the first bytes of its SHA-256.
fn shard_of(id: &str, n: u32) -> u32 {
    let h = knowledge::sha256_hex(id.as_bytes());
    (u64::from_str_radix(&h[..12], 16).unwrap_or(0) % n as u64) as u32
}

/// `cortiq teacher-data merge`: shards of one job as one `.tdata`.
fn cmd_teacher_data_merge(parts: &[String], output: &str) -> anyhow::Result<()> {
    anyhow::ensure!(output.ends_with(".tdata"), "teacher-data merge: output must end in .tdata");
    anyhow::ensure!(!std::path::Path::new(output).exists(), "teacher-data merge: {output} already exists");
    let readers: Vec<tdata::Dataset> = parts.iter().map(tdata::Dataset::open).collect::<anyhow::Result<_>>()?;
    let first = &readers[0].manifest.raw;
    for (r, p) in readers.iter().zip(parts).skip(1) {
        for ptr in ["/teacher/directory_hash", "/tokenizer/vocab_sha256", "/template/thinking", "/seed", "/max_tokens", "/prompts/sha256", "/split", "/sampling", "/generation/samples", "/generation/keep_truncated"] {
            anyhow::ensure!(
                r.manifest.raw.pointer(ptr) == first.pointer(ptr),
                "teacher-data merge: {p} differs from {} in {ptr} — not a shard of the same job",
                parts[0]
            );
        }
    }
    let mut manifest = first.clone();
    manifest["complete"] = serde_json::json!(false);
    manifest["shard"] = serde_json::Value::Null;
    manifest["merged_from"] = serde_json::json!(parts);
    if let Some(stats) = manifest.as_object_mut() {
        stats.remove("stats");
        stats.remove("files");
    }
    let mut writer = tdata::TdataWriter::create(output, manifest)?;
    let mut seen = std::collections::HashSet::new();
    for (r, p) in readers.iter().zip(parts) {
        let prompts = std::fs::read_to_string(r.root().join("prompts.jsonl"))?;
        for line in prompts.lines().filter(|l| !l.trim().is_empty()) {
            writer.append_prompt(&serde_json::from_str(line)?)?;
        }
        for sample in r.stream(tdata::TdataSelector::parse(&format!("{}:all", r.root().display()))?.include_rejected())? {
            let s = sample?;
            anyhow::ensure!(seen.insert(s.record.rid.clone()), "teacher-data merge: record {} appears twice ({p})", s.record.rid);
            writer.append_record(s.record, &s.tokens)?;
        }
    }
    let stats = writer.stats();
    writer.finalize()?;
    println!("teacher-data merge: {output}: {} records ({} usable) from {} parts", stats.records, stats.usable, parts.len());
    Ok(())
}

/// Resolve the two inputs needed to materialize a calibration set.  This is
/// deliberately a small pure gate: callers decide whether the `.tdata` root
/// already exists before opening a model or creating any output.
fn mtp_compact_generation_inputs<'a>(
    data: &str,
    teacher: Option<&'a str>,
    prompts: Option<&'a str>,
) -> anyhow::Result<(&'a str, &'a str)> {
    let teacher = teacher.ok_or_else(|| {
        anyhow::anyhow!(
            "mtp-compact: calibration data {data:?} does not exist; supply both --teacher and --prompts to generate it"
        )
    })?;
    let prompts = prompts.ok_or_else(|| {
        anyhow::anyhow!(
            "mtp-compact: calibration data {data:?} does not exist; supply both --teacher and --prompts to generate it"
        )
    })?;
    Ok((teacher, prompts))
}

/// The simple `compact SOURCE PROMPTS OUTPUT` form owns a visible reusable
/// cache beside its candidate. Replacing the final extension avoids an
/// ambiguous hidden directory while keeping multiple task outputs isolated.
fn simple_mtp_compact_data_path(output: &str) -> String {
    std::path::Path::new(output)
        .with_extension("tdata")
        .to_string_lossy()
        .into_owned()
}

/// Refuse a malformed final candidate name before an auto-generated teacher
/// data run can consume hours on a rented GPU.  The physical compactor repeats
/// its stronger collision/topology preflight after MTP discovery; this cheap
/// syntax fence applies equally to reused and missing `.tdata` inputs.
fn preflight_mtp_compact_output_suffix(output: &str) -> anyhow::Result<()> {
    let path = std::path::Path::new(output);
    anyhow::ensure!(
        path.extension().and_then(|extension| extension.to_str()) == Some("cmf"),
        "mtp-compact: --output must name a new .cmf candidate"
    );
    Ok(())
}

/// The model argument is always the runnable main CMF. Rejecting an MTP
/// sidecar here avoids generating a self-teacher dataset for an input the
/// physical compactor must later refuse for lacking its backbone.
fn preflight_mtp_compact_model_input(model: &str) -> anyhow::Result<()> {
    let path = std::path::Path::new(model);
    anyhow::ensure!(
        cortiq_core::mtp_sidecar_path(path) != path,
        "mtp-compact: MODEL must name the main .cmf, not an .mtp.cmf sidecar; pass its companion main CMF"
    );
    Ok(())
}

/// The short task form must always publish a runnable main CMF.  The extended
/// `mtp-compact` command retains the external-MiMo `.mtp.cmf` sidecar-only
/// compatibility spelling, which is why this is deliberately a separate
/// policy gate from the shared suffix check.
fn preflight_simple_mtp_compact_output(output: &str) -> anyhow::Result<()> {
    preflight_mtp_compact_output_suffix(output)?;
    let path = std::path::Path::new(output);
    if cortiq_core::mtp_sidecar_path(path) == path {
        anyhow::bail!(
            "compact: OUTPUT.cmf must name a runnable main CMF, not an .mtp.cmf sidecar; \
             use `mtp-compact` explicitly only when preserving a legacy MiMo sidecar-only workflow"
        );
    }
    Ok(())
}

/// A generated `.tdata` carries exact tokenizer byte/configuration hashes.
/// Check those inexpensive metadata contracts before spending a potentially
/// long teacher-generation run only to have MTP compaction reject its output.
fn ensure_mtp_compact_tokenizer_contracts_match(
    source_vocab: Option<&[u8]>,
    source_config: &[u8],
    teacher_vocab: Option<&[u8]>,
    teacher_config: &[u8],
) -> anyhow::Result<()> {
    let source_vocab = source_vocab.ok_or_else(|| {
        anyhow::anyhow!(
            "mtp-compact: source model has no embedded tokenizer bytes; a generated .tdata cannot be bound to it safely"
        )
    })?;
    let teacher_vocab = teacher_vocab.ok_or_else(|| {
        anyhow::anyhow!(
            "mtp-compact: teacher model has no embedded tokenizer bytes; cannot generate a reproducible .tdata tokenizer contract"
        )
    })?;
    anyhow::ensure!(
        knowledge::sha256_hex(source_vocab) == knowledge::sha256_hex(teacher_vocab),
        "mtp-compact: source and --teacher tokenizer vocabularies differ; refusing to generate calibration token ids that source cannot safely consume"
    );
    anyhow::ensure!(
        knowledge::sha256_hex(source_config) == knowledge::sha256_hex(teacher_config),
        "mtp-compact: source and --teacher tokenizer configurations differ; refusing to generate calibration token ids that source cannot safely consume"
    );
    Ok(())
}

/// Open only CMF metadata (not an all-tensor integrity scan) to reject an
/// incompatible teacher/source pair before a long missing-data generation.
/// The generator and physical compactor retain their full independent
/// integrity checks afterwards.
fn preflight_mtp_compact_generation_tokenizers(
    source_path: &str,
    teacher_path: &str,
) -> anyhow::Result<()> {
    let source = CmfModel::open_sharded(source_path).map_err(|error| {
        anyhow::anyhow!(
            "mtp-compact: opening source model {source_path:?} for teacher-data tokenizer preflight: {error}"
        )
    })?;
    let teacher = CmfModel::open_sharded(teacher_path).map_err(|error| {
        anyhow::anyhow!(
            "mtp-compact: opening teacher model {teacher_path:?} for tokenizer preflight: {error}"
        )
    })?;
    let source_config = serde_json::to_vec(&source.header.tokenizer_config).map_err(|error| {
        anyhow::anyhow!(
            "mtp-compact: serializing source tokenizer metadata for teacher-data preflight: {error}"
        )
    })?;
    let teacher_config = serde_json::to_vec(&teacher.header.tokenizer_config).map_err(|error| {
        anyhow::anyhow!(
            "mtp-compact: serializing teacher tokenizer metadata for preflight: {error}"
        )
    })?;
    ensure_mtp_compact_tokenizer_contracts_match(
        source.vocab.as_deref(),
        &source_config,
        teacher.vocab.as_deref(),
        &teacher_config,
    )
}

/// Reuse an immutable finalized `.tdata` set when it is already present, or
/// materialize it from the requested teacher exactly once before invoking the
/// physical MTP compactor.  The old `mtp-compact --data` path keeps its
/// semantics: the only new behavior is the missing-directory branch.
#[allow(clippy::too_many_arguments)]
fn cmd_mtp_compact_with_data(
    model: &str,
    data: &str,
    teacher: Option<&str>,
    prompts: Option<&str>,
    output: &str,
    keep_ratio: f64,
    plan_out: Option<&str>,
    max_records: usize,
    max_record_tokens: usize,
    generation: TeacherDataGenerateOptions,
) -> anyhow::Result<()> {
    // Validate the final artifact spelling before any optional teacher-data
    // generation. A missing `.cmf` suffix must fail in milliseconds rather
    // than after a costly self-teacher run. The extended command may still
    // use the `.mtp.cmf` legacy MiMo sidecar-only form; MTP discovery below
    // decides whether that form is valid for this source.
    preflight_mtp_compact_model_input(model)?;
    preflight_mtp_compact_output_suffix(output)?;
    // Parse selectors even before a directory exists: `calibration.tdata` and
    // `calibration.tdata:train` identify the same root, while generation must
    // write the root directory rather than accidentally create a colon-suffix
    // sibling. The compactor retains ownership of split/share validation.
    let selector = tdata::TdataSelector::parse(data).map_err(|error| {
        anyhow::anyhow!("mtp-compact: parsing --data selector {data:?}: {error}")
    })?;
    if selector.split != tdata::TdataSplitSelector::Train {
        anyhow::bail!(
            "mtp-compact: --data must select the train split; use PATH.tdata[:train][=SHARE]"
        );
    }
    let data_root = selector.root;
    if data_root.exists() {
        // Open-only is intentionally cheap. `cmd_mtp_compact` immediately
        // follows with the full stream/integrity preflight; opening here makes
        // an incomplete or malformed old job fail before any source-model
        // scan, and, critically, never overwrites it.
        tdata::TdataReader::open(&data_root).map_err(|error| {
            anyhow::anyhow!(
                "mtp-compact: existing calibration data {} must be finalized and readable: {error}",
                data_root.display()
            )
        })?;
        if teacher.is_some() || prompts.is_some() {
            eprintln!(
                "mtp compact: reusing existing calibration data {}; --teacher/--prompts and teacher-generation options are ignored",
                data_root.display()
            );
        } else {
            eprintln!(
                "mtp compact: reusing existing finalized calibration data {}",
                data_root.display()
            );
        }
    } else {
        let (teacher, prompts) = mtp_compact_generation_inputs(data, teacher, prompts)?;
        preflight_mtp_compact_generation_tokenizers(model, teacher)?;
        let data_root_utf8 = data_root.to_str().ok_or_else(|| {
            anyhow::anyhow!(
                "mtp-compact: generated calibration directory {} is not valid UTF-8",
                data_root.display()
            )
        })?;
        eprintln!(
            "mtp compact: calibration data {} is absent; generating it once from teacher {} and prompts {}",
            data_root.display(),
            teacher,
            prompts
        );
        cmd_teacher_data_generate(teacher, prompts, data_root_utf8, generation)
            .map_err(|error| anyhow::anyhow!("mtp-compact: generating missing calibration data: {error}"))?;
        // The writer's `finalize` is the commit point. Reopening prevents a
        // future generator change from silently flowing a partial directory
        // into the destructive/expensive compaction phase.
        tdata::TdataReader::open(&data_root).map_err(|error| {
            anyhow::anyhow!(
                "mtp-compact: generated calibration data {} was not finalized/readable: {error}",
                data_root.display()
            )
        })?;
    }

    mtpcompact::cmd_mtp_compact(
        model,
        data,
        output,
        keep_ratio,
        plan_out,
        max_records,
        max_record_tokens,
    )
}

/// Parse an integer train/validation/test ratio without a floating-point
/// threshold.  The deterministic splitter owns rounding after this point.
fn parse_teacher_data_split(value: &str) -> anyhow::Result<teacher_data::SplitRatios> {
    let parts: Vec<&str> = value.split('/').collect();
    if parts.len() != 3 || parts.iter().any(|part| part.is_empty()) {
        anyhow::bail!(
            "teacher-data generate: --split {value:?} must be train/val/test, for example 90/5/5"
        );
    }
    let parse = |part: &str, name: &str| -> anyhow::Result<u32> {
        part.parse::<u32>().map_err(|_| {
            anyhow::anyhow!(
                "teacher-data generate: --split {value:?}: {name} part {part:?} is not a non-negative integer"
            )
        })
    };
    teacher_data::SplitRatios::new(
        parse(parts[0], "train")?,
        parse(parts[1], "validation")?,
        parse(parts[2], "test")?,
    )
}

/// A `Read` adapter that records exactly the source bytes the JSONL parser
/// consumes.  Keeping the digest outside the adapter lets the parser own and
/// exhaust its buffered reader without a second source pass or a pathname
/// race between a pre-hash and parse.
struct HashingReader<'a, R> {
    inner: R,
    hasher: &'a mut sha2::Sha256,
}

impl<R: std::io::Read> std::io::Read for HashingReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        use sha2::Digest as _;

        let count = self.inner.read(buffer)?;
        self.hasher.update(&buffer[..count]);
        Ok(count)
    }
}

/// Parse and hash the prompt source in one forward pass without retaining a
/// second complete copy in RAM.  The manifest digest therefore describes the
/// actual byte stream given to the parser, even if a writer mutates the file
/// in place while the command is running.
fn parse_hashed_teacher_data_prompts(
    path: &str,
) -> anyhow::Result<(teacher_data::PromptLoad, String)> {
    use sha2::{Digest, Sha256};

    let file = std::fs::File::open(path)
        .map_err(|err| anyhow::anyhow!("opening prompts {path:?}: {err}"))?;
    let mut hasher = Sha256::new();
    let loaded = teacher_data::parse_jsonl_prompts(
        std::io::BufReader::new(HashingReader {
            inner: file,
            hasher: &mut hasher,
        }),
        teacher_data::PromptParseOptions::default(),
    )
    .map_err(|err| anyhow::anyhow!("parsing prompts {path:?}: {err}"))?;
    Ok((loaded, format!("{:x}", hasher.finalize())))
}

/// Sampling configuration for the first teacher-data slice.  Greedy decode
/// avoids a probability reconstruction pass and preserves the engine's MTP
/// fast path when the selected CMF has it.
fn teacher_data_greedy_sampler(seed: u64) -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        seed: Some(seed),
        ..Default::default()
    }
}

/// Generate a minimal but auditable `.tdata` v1 data set from JSONL prompts.
///
/// Deliberately absent in this first vertical slice: top-k logits, exact
/// hidden-reasoning spans, and public resume orchestration.  Each absence is
/// fail-closed rather than written as estimated data.  The hot `Pipeline`
/// remains alive for the whole job; only the request state and deterministic
/// sampler stream are reset between records.
/// `~3 h 05 min` / `~12 min` / `~40 s`: a remaining-time estimate.
pub(crate) fn fmt_eta(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s >= 3600 {
        format!("{} h {:02} min", s / 3600, (s % 3600) / 60)
    } else if s >= 90 {
        format!("{} min", (s + 30) / 60)
    } else {
        format!("{s} s")
    }
}

pub(crate) fn cmd_teacher_data_generate(
    teacher_path: &str,
    prompts_path: &str,
    output_path: &str,
    options: TeacherDataGenerateOptions,
) -> anyhow::Result<()> {
    teacher_data::require_topk_disabled(options.topk)?;
    if let Some(api) = teacher_api::ApiTeacher::parse(teacher_path, &options.api)? {
        return teacher_data_generate_api(&api, teacher_path, prompts_path, output_path, &options);
    }
    if options.raw {
        anyhow::bail!(
            "teacher-data generate: --raw is not available in the direct-answer v1 path; \
             it cannot prove an exact non-thinking response boundary without a carried chat template"
        );
    }
    if options.max_tokens == 0 {
        anyhow::bail!("teacher-data generate: --max-tokens must be positive");
    }
    if options.samples == 0 {
        anyhow::bail!("teacher-data generate: --samples must be positive");
    }
    if !output_path.ends_with(".tdata") {
        anyhow::bail!(
            "teacher-data generate: output {output_path:?} must end in .tdata to make the immutable dataset boundary explicit"
        );
    }
    // An existing directory is either finished (refuse) or an interrupted
    // run of this same job (continue it below).
    let resuming = std::path::Path::new(output_path).exists();
    if resuming {
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(std::path::Path::new(output_path).join("manifest.json"))
                .map_err(|e| anyhow::anyhow!("teacher-data generate: {output_path} exists but has no readable manifest: {e}"))?,
        )?;
        if manifest.get("complete").and_then(|v| v.as_bool()) == Some(true) {
            anyhow::bail!("teacher-data generate: {output_path:?} is already finalized; choose a new .tdata directory");
        }
    }
    let ratios = parse_teacher_data_split(&options.split)?;

    // Hash while parsing through one descriptor.  This is bounded by a JSONL
    // row, not a duplicate source-byte buffer, and the manifest digest names
    // the exact stream that supplied the accepted prompts.
    let (mut loaded, prompt_sha256) = parse_hashed_teacher_data_prompts(prompts_path)?;
    let accepted_before_limit = loaded.accepted();
    if let Some(limit) = options.limit {
        loaded.prompts.truncate(limit);
    }
    if loaded.prompts.is_empty() {
        anyhow::bail!(
            "teacher-data generate: no accepted prompts remain after validation{}",
            options
                .limit
                .map(|limit| format!(" and --limit {limit}"))
                .unwrap_or_default()
        );
    }
    let mut splits =
        teacher_data::assign_stratified_splits(&loaded.prompts, ratios, options.split_seed)?;
    debug_assert_eq!(splits.len(), loaded.prompts.len());
    // A shard keeps its prompts with the splits they got among all prompts.
    if let Some((i, n)) = options.shard {
        let keep: Vec<bool> = loaded.prompts.iter().map(|p| shard_of(&p.id, n) == i).collect();
        let mut k = keep.iter();
        loaded.prompts.retain(|_| *k.next().unwrap());
        let mut k = keep.iter();
        splits.retain(|_| *k.next().unwrap());
        anyhow::ensure!(!loaded.prompts.is_empty(), "teacher-data generate: shard {i}/{n} holds no prompts");
        eprintln!("teacher-data: shard {i}/{n}: {} prompts", loaded.prompts.len());
    }

    // The append-only writer cannot safely recover a prompt ledger row that
    // exceeds its bounded JSONL reader.  Validate every row before opening a
    // model or creating an incomplete output directory.
    for (prompt, &split) in loaded.prompts.iter().zip(&splits) {
        let ledger = teacher_data::canonical_prompt_json(prompt, split);
        tdata::serialize_prompt_jsonl_row(&ledger)?;
    }

    let planned_records = loaded.prompts.iter().try_fold(0_u64, |total, prompt| {
        total
            .checked_add(u64::from(prompt.samples.unwrap_or(options.samples)))
            .ok_or_else(|| {
                anyhow::anyhow!("teacher-data generate: planned record count overflows u64")
            })
    })?;

    // Do every failure-prone configuration step before creating an incomplete
    // output directory.  It is important that the one pipeline owns all
    // records: recreating it would repeatedly compile Metal graphs/upload
    // state and make the benchmark unnecessarily slow.
    let model = Arc::new(CmfModel::open_sharded(teacher_path)?);
    // `open_sharded` joins sibling tensor directories, but opening alone does
    // not rehash every payload.  A teacher-data manifest must not claim that
    // an unchecked sharded (or partially modified) teacher produced its
    // answers: validate all CMF tensor bytes once before any output exists.
    let integrity_problems = model.verify();
    if !integrity_problems.is_empty() {
        let preview = integrity_problems
            .iter()
            .take(3)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("; ");
        anyhow::bail!(
            "teacher-data generate: CMF integrity verification failed for {teacher_path:?} ({} problem(s)): {preview}",
            integrity_problems.len()
        );
    }
    // This fingerprint is calculated from the merged directory (including
    // sibling shards) after the all-tensor verification above.  `dir_hash`
    // remains useful for legacy/base identity, but is only the primary shard.
    let teacher_trunk_hash = model.trunk_hash();
    let sampling = teacher_data_sampling(&options.sampling, &model)?;
    let mut pipeline = Pipeline::from_model(&model, teacher_data_greedy_sampler(options.seed))?;
    if options.disable_mtp {
        // MTP is an optional speculative accelerator, not part of the
        // generated-token contract. This keeps the trunk on the selected GPU
        // while making a long calibration job plain greedy by request.
        pipeline.speculative = false;
    }
    let speculative_mtp = pipeline.speculative;
    pipeline.set_confidence(false);

    // A carried template is the only current mechanism for requesting
    // `enable_thinking=false`.  Base completion text could contain a model's
    // invisible reasoning convention, and treating every generated id as an
    // answer would silently corrupt answer-only losses.
    let template = pipeline.tokenizer.chat_template.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "teacher-data generate: teacher has no carried chat template; the direct-answer v1 path refuses raw/base completion until it can record exact reasoning boundaries"
        )
    })?;
    let rendered_prompts = loaded
        .prompts
        .iter()
        .map(|prompt| teacher_data::render_prompt(&pipeline.tokenizer, prompt, false, options.no_think))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let tokenizer_vocab = model.vocab.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "teacher-data generate: teacher has no embedded tokenizer bytes; refusing to write a token dataset without a reproducible tokenizer contract"
        )
    })?;
    let tokenizer_config_bytes = serde_json::to_vec(&model.header.tokenizer_config)
        .map_err(|err| anyhow::anyhow!("serializing teacher tokenizer configuration: {err}"))?;

    let source_name = std::path::Path::new(prompts_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("prompts.jsonl")
        .to_owned();
    let teacher_name = std::path::Path::new(teacher_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("teacher.cmf")
        .to_owned();
    let template_sha256 = knowledge::sha256_hex(template.as_bytes());
    let manifest = serde_json::json!({
        "format": "cmf-tdata",
        "version": 1,
        "complete": false,
        "cortiq": {"version": env!("CARGO_PKG_VERSION")},
        "teacher": {
            "kind": "cmf",
            "file": teacher_name,
            "directory_hash": format!("{:016x}", model.dir_hash()),
            "trunk_hash": format!("{:016x}", teacher_trunk_hash),
            "integrity": "verified-all-cmf-tensors-v1",
            "architecture": model.arch().arch_name.clone(),
            "quantization": format!("{:?}", model.header.quant_type),
        },
        "tokenizer": {
            "vocab_size": pipeline.tokenizer.vocab_size(),
            "eos_id": pipeline.tokenizer.eos_token_id,
            "vocab_sha256": knowledge::sha256_hex(tokenizer_vocab),
            "config_sha256": knowledge::sha256_hex(&tokenizer_config_bytes),
            "template_sha256": template_sha256,
        },
        "template": {
            "mode": "carried",
            "thinking": if options.no_think { "off" } else { "on" },
        },
        "sampling": sampling,
        "seed": options.seed,
        "max_tokens": options.max_tokens,
        "topk": {"k": 0, "positions": "none"},
        "prompts": {
            "source": source_name,
            "sha256": prompt_sha256,
            "source_lines": loaded.source_lines,
            "duplicates": loaded.duplicates,
            "skipped_short": loaded.skipped_short,
            "accepted_before_limit": accepted_before_limit,
            "count": loaded.accepted(),
        },
        "split": {
            "rule": "stratified-hash",
            "rule_version": 1,
            "seed": options.split_seed,
            "ratios": [ratios.train, ratios.val, ratios.test],
        },
        "generation": {
            "mode": if options.no_think { "direct-answer" } else { "think" },
            "samples": options.samples,
            "keep_truncated": options.keep_truncated,
            "speculative_mtp": if speculative_mtp { "enabled" } else { "disabled" },
            "mtp_control": if options.disable_mtp { "disabled-by-cli" } else { "engine-default" },
            "topk": "disabled-until-raw-logit-capture",
            "resume": "same-command",
            "loop_stop": "distinct-5gram<0.05 over the last 512 pieces",
        },
        "shard": options.shard.map(|(i, n)| serde_json::json!({"index": i, "count": n})),
        "input": {
            "format": "jsonl-v1",
            "minimum_visible_prompt_chars": teacher_data::DEFAULT_MIN_PROMPT_CHARS,
            "limit": options.limit,
        },
    });
    // The `</think>` id splits a reasoning model's records exactly.
    let think_end = if options.no_think {
        None
    } else {
        Some(teacher_data::think_end_id(&pipeline.tokenizer).ok_or_else(|| {
            anyhow::anyhow!(
                "teacher-data generate: the teacher's vocabulary has no single `</think>` token, so a reasoning span cannot be bounded exactly; use --no-think"
            )
        })?)
    };
    let mut done_rids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut writer = if resuming {
        let w = tdata::TdataWriter::resume(output_path)?;
        // Continue only the same job: every field that shapes a record must match.
        let old = w.manifest();
        let mut diff = Vec::new();
        for ptr in [
            "/teacher/directory_hash",
            "/tokenizer/vocab_sha256",
            "/template/thinking",
            "/seed",
            "/max_tokens",
            "/prompts/sha256",
            "/split",
            "/sampling",
            "/generation/samples",
            "/generation/keep_truncated",
        ] {
            if old.pointer(ptr) != manifest.pointer(ptr) {
                diff.push(format!("{ptr}: {} → {}", old.pointer(ptr).unwrap_or(&serde_json::Value::Null), manifest.pointer(ptr).unwrap_or(&serde_json::Value::Null)));
            }
        }
        if !diff.is_empty() {
            anyhow::bail!(
                "teacher-data generate: {output_path} is an interrupted run of a different job ({}); use a new directory",
                diff.join("; ")
            );
        }
        let records = std::fs::read_to_string(std::path::Path::new(output_path).join("records.jsonl"))?;
        for line in records.lines().filter(|l| !l.trim().is_empty()) {
            if let Some(rid) = serde_json::from_str::<serde_json::Value>(line).ok().and_then(|v| v.get("rid").and_then(|r| r.as_str()).map(str::to_owned)) {
                done_rids.insert(rid);
            }
        }
        eprintln!("teacher-data: continuing {output_path}: {} record(s) already committed", done_rids.len());
        w
    } else {
        let mut w = tdata::TdataWriter::create(output_path, manifest)?;
        for (prompt, &split) in loaded.prompts.iter().zip(&splits) {
            w.append_prompt(&teacher_data::canonical_prompt_json(prompt, split))?;
        }
        w
    };

    eprintln!(
        "teacher-data: {} accepted prompts → {planned_records} record(s); teacher={} ({}, {:?}); one hot pipeline; MTP={}",
        loaded.accepted(),
        teacher_name,
        model.arch().arch_name,
        model.header.quant_type,
        if speculative_mtp { "on" } else { "off" },
    );
    let job_started = std::time::Instant::now();
    let mut generated_tokens = 0_u64;
    let mut record_number = 0_u64;
    // records generated by this run (a resumed run skips the done ones):
    // the rate behind the "left" estimate
    let mut fresh = 0_u64;
    for ((prompt, &split), rendered) in loaded.prompts.iter().zip(&splits).zip(&rendered_prompts) {
        // Rendering was preflighted before creating output.  It neither
        // depends on the seed nor mutates KV, and the immutable ids are
        // reused for each sample exactly as they will be written to tokens.bin.
        let per_samples = prompt.samples.unwrap_or(options.samples);
        let per_max_tokens = prompt.max_tokens.unwrap_or(options.max_tokens);
        for sample in 0..per_samples {
            if done_rids.contains(&teacher_data::record_id(&prompt.id, sample)) {
                record_number += 1;
                continue;
            }
            let record_seed = teacher_data::record_seed(options.seed, &prompt.id, sample);
            pipeline.reset_session();
            pipeline.set_sampler_config(sampler_from(&sampling, record_seed));
            let started = std::time::Instant::now();
            let looped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let guard_flag = looped.clone();
            let mut guard = teacher_data::LoopGuard::new(0.05);
            // a live line while one long answer is generated (a reasoning
            // answer runs for minutes): tokens so far and their rate
            let (live_n, live_t0) = (record_number + 1, std::time::Instant::now());
            let mut live = (0_u64, std::time::Instant::now());
            let on_token: cortiq_engine::pipeline::TokenCallback = Box::new(move |piece: &str| {
                live.0 += 1;
                if live.1.elapsed().as_secs_f64() >= 5.0 {
                    live.1 = std::time::Instant::now();
                    let el = live_t0.elapsed().as_secs_f64();
                    eprint!("\r  record {live_n}/{planned_records}: {} tokens · {:.1} tok/s   ", live.0, live.0 as f64 / el.max(1e-9));
                }
                let go = guard.feed(piece);
                if !go {
                    guard_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                go
            });
            let result = pipeline
                .generate_from_ids(&rendered.token_ids, per_max_tokens, None, Some(on_token))
                .map_err(|err| {
                    anyhow::anyhow!(
                        "teacher-data generate: prompt {} (source line {}, sample {}): {err}",
                        prompt.id,
                        prompt.source_line,
                        sample,
                    )
                })?;
            if result.tokens_generated != result.token_ids.len() {
                anyhow::bail!(
                    "teacher-data generate: prompt {} sample {}: engine returned tokens_generated={} but {} token ids; refusing non-exact calibration payload",
                    prompt.id,
                    sample,
                    result.tokens_generated,
                    result.token_ids.len(),
                );
            }
            let (response, visible) = match think_end {
                Some(end) => {
                    let (think_n, answer_n) = teacher_data::split_think_ids(&result.token_ids, end);
                    let think_text = pipeline.tokenizer.decode(&result.token_ids[..think_n]);
                    let answer_text = pipeline.tokenizer.decode(&result.token_ids[think_n..]);
                    let visible = answer_text.clone();
                    (
                        teacher_data::ResponseParts::segmented(
                            Some(think_text),
                            answer_text,
                            think_n as u32,
                            answer_n as u32,
                        ),
                        visible,
                    )
                }
                None => {
                    let visible = result.text.clone();
                    (teacher_data::ResponseParts::direct(result.text.clone(), result.token_ids.len())?, visible)
                }
            };
            let engine_finish = if looped.load(std::sync::atomic::Ordering::Relaxed) {
                "loop"
            } else {
                result.finish_reason.as_str()
            };
            let (finish, quality) = teacher_data::final_record_quality(
                engine_finish,
                &visible,
                &result.token_ids,
                options.keep_truncated,
            );
            let prepared =
                teacher_data::build_canonical_record(teacher_data::CanonicalRecordInput {
                    prompt,
                    split,
                    sample,
                    seed: record_seed,
                    rendered,
                    generated_token_ids: &result.token_ids,
                    response,
                    finish: &finish,
                    quality,
                })?;
            let usable = prepared.record.usable;
            let elapsed = started.elapsed().as_secs_f64();
            let answer_tokens = result.token_ids.len() as u64;
            let mtp_drafted = result.mtp_drafted;
            let mtp_accepted = result.mtp_accepted;
            writer.append_record(prepared.record, &prepared.tokens)?;
            record_number += 1;
            fresh += 1;
            let left_secs = job_started.elapsed().as_secs_f64() / fresh as f64 * planned_records.saturating_sub(record_number) as f64;
            generated_tokens = generated_tokens.checked_add(answer_tokens).ok_or_else(|| {
                anyhow::anyhow!("teacher-data generate: generated token counter overflows u64")
            })?;
            eprintln!(
                "\rteacher-data: [{record_number}/{planned_records}] {}#{} split={} finish={} usable={} answer={} tok {:.2} tok/s{} · ~{} left",
                prompt.id,
                sample,
                split.as_str(),
                finish,
                usable,
                answer_tokens,
                answer_tokens as f64 / elapsed.max(f64::MIN_POSITIVE),
                if mtp_drafted > 0 {
                    format!(" · MTP {mtp_accepted}/{mtp_drafted}")
                } else {
                    String::new()
                },
                fmt_eta(left_secs),
            );
        }
    }
    let stats = writer.stats();
    let manifest = writer.finalize()?;
    let elapsed = job_started.elapsed().as_secs_f64();
    println!(
        "teacher-data: finalized {}: records={} usable={} rejected={} answer_tokens={} overall={:.2} tok/s",
        output_path,
        stats.records,
        stats.usable,
        stats.rejected,
        generated_tokens,
        generated_tokens as f64 / elapsed.max(f64::MIN_POSITIVE),
    );
    println!(
        "teacher-data: manifest {} v{} complete={}",
        manifest.format, manifest.version, manifest.complete
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
async fn cmd_serve(
    model_path: &str,
    host: Option<&str>,
    port: u16,
    default_task: Option<&str>,
    llm_given: &[&'static str],
    decision_flags: &decision::ServeFlags,
    _compat_port: Option<u16>,
    o1: &O1Flags,
    peer: Option<&str>,
    peer_split: Option<usize>,
    net_token: Option<&str>,
    net_dtype: &str,
    replicas: Option<usize>,
    route_include_quarantine: bool,
    lookup_mode: Option<&str>,
) -> anyhow::Result<()> {
    println!();
    println!("  ╔═══════════════════════════════════════╗");
    println!("  ║     Cortiq — Sparse Inference Engine   ║");
    println!("  ╚═══════════════════════════════════════╝");
    println!();

    // Load model + pipeline (real weights; fails loudly on a bad file).
    println!("  Loading model: {}", model_path);
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    // A decision file (DECISION bit) is served by the decisions server: no
    // Pipeline, no GPU, loopback unless --host says otherwise (spec §4.2).
    let is_decision = decision::is_decision_model(&model);
    decision::check_serve_flags(model_path, is_decision, llm_given, decision_flags)?;
    let host = decision::resolve_serve_host(host, is_decision);
    if is_decision {
        drop(model);
        return decision::serve(model_path, host, port, decision_flags).await;
    }
    // An EmbeddingGemma 2 file is an encoder: no Pipeline, no KV cache —
    // the embeddings server (POST /v1/embeddings) takes it.
    if cortiq_engine::egemma2::is_embedding_gemma2(&model) {
        let addr = std::net::ToSocketAddrs::to_socket_addrs(&(host, port))
            .map_err(|e| anyhow::anyhow!("bind address {host}:{port}: {e}"))?
            .next()
            .ok_or_else(|| anyhow::anyhow!("bind address {host}:{port}: no address"))?;
        return cortiq_server::embeddings::serve(model, model_path, addr).await;
    }
    let lookup_mode =
        cortiq_engine::lookup::LookupMode::resolve(lookup_mode).map_err(anyhow::Error::msg)?;
    let default_task = default_task.unwrap_or("general");
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
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut slots = std::env::var("CMF_SERVE_SLOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| (avail / 4).clamp(1, 4));
    // Replica mode (`--gpus N`): one slot PER CARD, each holding the
    // whole model on its own device. This is the multi-GPU mode that
    // actually scales — a layer split runs the cards in sequence and at
    // best matches one of them, replicas serve N requests at once
    // (2×RTX 5090, W2 34.7B: 232 tok/s aggregate against 119.8).
    let mut devices: Vec<usize> = Vec::new();
    let mut split_devices: Option<Vec<usize>> = None;
    if let Some(n) = replicas {
        let have = cortiq_engine::gpu::device_count();
        if n < 2 {
            anyhow::bail!("--gpus {n}: replicas need at least 2 cards");
        }
        if have < n {
            anyhow::bail!(
                "--gpus {n}, but {have} cards are visible (see `cortiq gpu`) —                  asking for more replicas than devices is not honest"
            );
        }
        if peer.is_some() {
            anyhow::bail!("--gpus and --peer are different modes: replicas versus a layer split");
        }
        // Replicas need the model to fit ONE card. When it does not, the
        // honest answer is the layer split, not a refusal and not N
        // copies that thrash: say which mode you are in and why.
        let budget = cortiq_engine::gpu::vram_budget();
        // A genome file budgets its trunk only (skill tensors replace trunk
        // tensors inside their lane — NF-7); other files the whole mapping.
        let weights = cortiq_engine::gpu::placement_weight_bytes(&model);
        if budget != u64::MAX && weights > budget {
            slots = 1;
            devices.clear();
            split_devices = Some((0..n).collect());
            println!(
                "    Split across cards: the {:.1} GB model does not fit one card's budget ({:.1} GB) — \
                 the layers are cut across {n} devices (replicas need a full copy per card)",
                weights as f64 / 1e9,
                budget as f64 / 1e9,
            );
        } else {
            // One slot per card by default. The frame profiler shows a
            // decode dispatch leaves this class of GPU mostly idle (a
            // 2048-hidden layer is 30-70 µs of work where the card
            // wants thousands of workgroups), which suggests a second
            // slot could interleave into the gaps — MEASURED, and it
            // does not: on 2×RTX 5090, four concurrent requests over
            // four slots ran 185 tok/s against 210 over two. They
            // contend for the queue and for the CPU pool instead of
            // overlapping. The knob stays for hardware that disagrees.
            let per_gpu = std::env::var("CMF_SLOTS_PER_GPU")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(1);
            slots = n * per_gpu;
            devices = (0..slots).map(|i| i % n).collect();
            println!(
                "    Replicas: {n} cards × the full model, {slots} slot(s) —                  {per_gpu} per card"
            );
        }
    }
    if peer.is_some() && slots != 1 {
        // One worker holds ONE KV session; more slots would interleave
        // sequences into it. Loud, not silent.
        println!("    note: --peer forces 1 slot (the worker holds one KV session)");
        slots = 1;
    }
    if slots != 1 && model.arch().arch_name == "deepseek_v41" {
        // Every slot's pipeline builds its own expert slot map, but the
        // GPU expert bank is one per model: two maps over the same
        // physical slots would let one slot compute with an expert the
        // other has just overwritten. One slot until the pool is shared.
        println!("    note: deepseek_v41 serves with 1 slot (one GPU expert bank per model)");
        slots = 1;
    }
    if std::env::var("CMF_THREADS").is_err() {
        // Split the cores between slots instead of oversubscribing
        // N pools × (cores−1) workers. Explicit CMF_THREADS wins.
        let per = (avail.saturating_sub(1) / slots).max(1);
        // SAFETY: single-threaded startup, before any pipeline/pool spawn.
        unsafe { std::env::set_var("CMF_THREADS", per.to_string()) };
    }
    let mut pipelines = Vec::with_capacity(slots);
    for i in 0..slots {
        // Load with the thread pinned to the slot's card, so every
        // device-resident allocation this pipeline makes lands there.
        if let Some(d) = devices.get(i) {
            cortiq_engine::gpu::set_current_device(*d);
        }
        let mut pipeline = Pipeline::from_model(&model, SamplerConfig::default())?;
        // No endpoint reports per-token confidence, and it is a full-vocab
        // softmax every token (~1 ms at Gemma's 262k vocab).
        // `CMF_SERVE_CONFIDENCE=1` keeps it.
        if std::env::var("CMF_SERVE_CONFIDENCE").as_deref() != Ok("1") {
            pipeline.set_confidence(false);
        }
        o1.apply(&mut pipeline)?;
        if let Some(devs) = &split_devices {
            pipeline
                .set_gpu_plan(Some(devs))
                .map_err(|e| anyhow::anyhow!("--gpus: {e}"))?;
        }
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

    // Network pipeline-split (serve --peer): connect and assign before
    // the listener starts — a bad worker fails the whole serve loudly.
    let mut remote = None;
    if let Some(addr) = peer {
        pipelines[0]
            .split_supported()
            .map_err(|e| anyhow::anyhow!(e))?;
        let nl = pipelines[0].num_layers;
        let split = peer_split.unwrap_or(nl / 2);
        if split >= nl {
            anyhow::bail!("--peer-split {split} out of range: the model has {nl} layers");
        }
        let dtype = match net_dtype {
            "f32" => cortiq_net::WireDtype::F32,
            "f16" => cortiq_net::WireDtype::F16,
            other => anyhow::bail!("--net-dtype {other}: expected f32 or f16"),
        };
        let rs = cortiq_net::RemoteSegment::connect(
            addr,
            net_token.unwrap_or(""),
            model.dir_hash(),
            &model.arch().arch_name,
            nl,
            pipelines[0].hidden_size,
            split,
            nl - 1,
            dtype,
            &cortiq_net::SessionSpec::default(),
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        println!(
            "    Peer: {addr} · layers {split}..{} remote · wire {net_dtype}",
            nl - 1
        );
        remote = Some(std::sync::Arc::new(std::sync::Mutex::new(rs)));
    }

    // Router policy v2: every request is decided on its last user message
    // (backbone-gated); the slots above are the backbone lane, each routed
    // skill gets its own lane on first use. φ runs on a separate backbone
    // pipeline so no slot's conversation state is reset by routing.
    let routing = if !cortiq_engine::router::is_router_v2(&model) {
        None
    } else if remote.is_some() {
        println!(
            "    Router v2: OFF under --peer (the worker holds the backbone span) — every \
             request runs the backbone"
        );
        None
    } else {
        let probe = Pipeline::from_model(&model, SamplerConfig::default())?;
        // The router was fitted with φ on one backend; φ here runs on
        // this process's — a different numerical class shifts E and the
        // measured gate does not transfer.
        knowledge::warn_phi_backend(&model, "serve");
        let lane_model = model.clone();
        let lane_o1 = o1.clone();
        let lane_split = split_devices.clone();
        let factory: cortiq_server::route::LaneFactory = Box::new(move |id: &str| {
            let mut p =
                Pipeline::from_model_with_skill(&lane_model, SamplerConfig::default(), Some(id))
                    .map_err(|e| e.to_string())?;
            lane_o1.apply(&mut p).map_err(|e| e.to_string())?;
            if let Some(devs) = &lane_split {
                p.set_gpu_plan(Some(devs))?;
            }
            Ok(p)
        });
        let r = cortiq_server::SkillRouter::new(
            model.clone(),
            probe,
            factory,
            route_include_quarantine,
        )
        .with_lookup_mode(lookup_mode);
        println!(
            "    Router v2: backbone_gated per request · routable skills {:?}{} · a skill lane \
             loads on its first request",
            r.routable_skills(),
            if route_include_quarantine {
                " (quarantine included — debug)"
            } else {
                ""
            }
        );
        if r.has_lookups() {
            println!(
                "    Lookup records: {:?} · mode {} (the table answers / the backbone runs with \
                 the card / ignored; --lookup-mode, CMF_LOOKUP_MODE)",
                r.lookup_ids(),
                lookup_mode.label()
            );
        }
        Some(Arc::new(r))
    };

    // Create runtime
    let runtime = CortiqRuntime::new(model.clone());
    if runtime.masks().get(default_task).is_some() {
        let _ = runtime.switch_task(default_task).await;
    }
    let tokenizer = pipelines[0].tokenizer.clone();
    let state = Arc::new(AppState {
        runtime,
        tokenizer,
        slots: if devices.is_empty() {
            cortiq_server::PipelinePool::new(pipelines)
        } else {
            cortiq_server::PipelinePool::with_devices(pipelines, devices)
        },
        remote,
        routing,
    });

    // Build router
    let app = build_router(state);

    // Start server
    let addr = format!("{}:{}", host, port);
    println!(
        "  ✓ API server:     http://{}:{}/v1/chat/completions",
        host, port
    );
    println!("  ✓ Web dashboard:  http://localhost:{}/", port);
    println!(
        "  ✓ Status:         http://localhost:{}/v1/cortiq/status",
        port
    );
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
        presence_penalty: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        min_p: 0.0,
        seed: Some(0),
        ..Default::default()
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
    polish_only: bool,
    teacher: Option<&str>,
) -> anyhow::Result<()> {
    use cortiq_engine::fcd::{FcdHyper, GenGateCfg, run_polish_distilled};
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

    // Teacher data: whole records, loss on the generated tokens, windows
    // inside records; the val split is the held-out set.
    if corpus.contains(".tdata") {
        use crate::compress::data::{Data, LossOn, Stream};
        use crate::tdata::TdataSplit;
        let data = Data::load(corpus)?;
        data.check_tokenizer(&model, "model")?;
        let eos = model.header.tokenizer_config.as_ref().and_then(|t| t.eos_token_ids.first().copied()).unwrap_or(0);
        let tr = Stream::build(data.split(TdataSplit::Train), LossOn::Generated, eos, seq + 1);
        let va = Stream::build(data.split(TdataSplit::Val), LossOn::Generated, eos, seq + 1);
        println!("corpus (teacher data): train {} scored tokens, val {}", tr.scored, va.scored);
        let hp = FcdHyper { steps, lr, kl_w: kl, eval_every, bs, seq, seed: 0, polish_only };
        let out_path = out
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(format!("{model_path}.fcd.cmf")));
        let tmodel = match teacher {
            Some(p) => Some(std::sync::Arc::new(cortiq_core::CmfModel::open_sharded(p)?)),
            None => None,
        };
        let report = cortiq_engine::fcd::run_polish_stream(
            &model,
            tmodel.as_ref(),
            &cfg,
            &hp,
            cortiq_engine::fcd::FcdStream { ids: &tr.ids, weight: Some(&tr.weight), starts: Some(&tr.starts) },
            cortiq_engine::fcd::FcdStream { ids: &va.ids, weight: Some(&va.weight), starts: Some(&va.starts) },
            &out_path,
            None,
            None,
        )
        .map_err(|e| anyhow::anyhow!("fcd polish: {e}"))?;
        println!(
            "fcd: val ppl teacher {:.3} · start {:.3} → final {:.3} (best step {}) → {}",
            report.teacher_ppl,
            report.ppl_start,
            report.ppl_final,
            report.best_step,
            out_path.display()
        );
        return Ok(());
    }
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

    if gen_check && !polish_only {
        println!("── gen-check BEFORE polish (zero-shot O(1)) ──");
        fcd_gen_check(&model, &cfg, &va, "before")?;
    }

    let hp = FcdHyper {
        steps,
        lr,
        kl_w: kl,
        eval_every,
        bs,
        seq,
        seed: 0,
        polish_only,
    };
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
    let tmodel = match teacher {
        Some(p) => Some(std::sync::Arc::new(cortiq_core::CmfModel::open_sharded(p)?)),
        None => None,
    };
    let report = run_polish_distilled(
        &model,
        tmodel.as_ref(),
        &cfg,
        &hp,
        &tr,
        &va,
        &out_path,
        gate_cfg.as_ref(),
    )
    .map_err(|e| anyhow::anyhow!("fcd polish: {e}"))?;

    println!("── FCD polish report ──");
    println!("converted layers : {:?}", report.converted);
    println!("teacher val-ppl  : {:.2}", report.teacher_ppl);
    println!(
        "student ppl start: {:.2} (zero-shot O(1))",
        report.ppl_start
    );
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
    // A natively bounded file scores through its own operator; the
    // post-hoc O(1) kernel does not apply to it.
    if o1.cfg()?.is_some() {
        if let Some(why) = pipeline.o1_refusal() {
            anyhow::bail!("{why}");
        }
    }
    // Windowed scoring keeps the RAW token stream: the val_ppl yardstick
    // slices windows out of the middle of the corpus, where a prepended
    // BOS would be a token the reference never scored.
    let windowed = win.offsets(pipeline.tokenizer.encode(&text).len())?;
    let mut ids = match &windowed {
        Some(_) => pipeline.tokenizer.encode(&text),
        None => pipeline
            .tokenizer
            .with_bos(pipeline.tokenizer.encode(&text)),
    };
    if windowed.is_none() {
        ids.truncate(max_tokens);
    }
    // CMF_PPL_IDS_OUT=path: the scored token ids, space-separated — for
    // diffing the tokenizer against a reference on the same corpus.
    if let Ok(p) = std::env::var("CMF_PPL_IDS_OUT") {
        let s: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        std::fs::write(&p, s.join(" "))?;
    }

    // CMF_PREFILL_PARITY=split[,k]: the host prefill, the batch graph and a
    // host prefix continued on the graph over the same tokens (diagnostic).
    if let Ok(spec) = std::env::var("CMF_PREFILL_PARITY") {
        let mut it = spec.split(',').map(|s| s.trim().parse::<usize>().ok());
        let split = it
            .next()
            .flatten()
            .ok_or_else(|| anyhow::anyhow!("CMF_PREFILL_PARITY=split[,k]"))?;
        let k = it.next().flatten().unwrap_or(64);
        let report = pipeline
            .prefill_parity_probe(&ids, split, k)
            .map_err(|e| anyhow::anyhow!(e))?;
        println!("{report}");
        return Ok(());
    }
    // CMF_PREFILL_PARITY_O1=k with --o1: the O(1) seal from a host prefix
    // against seals from a batch-graph prefix (diagnostic).
    if let Ok(k) = std::env::var("CMF_PREFILL_PARITY_O1") {
        let cfg = o1
            .cfg()?
            .ok_or_else(|| anyhow::anyhow!("CMF_PREFILL_PARITY_O1 needs --o1"))?;
        let k = k.trim().parse::<usize>().unwrap_or(64);
        let report = pipeline
            .o1_seal_parity_probe(&ids, cfg, o1_prefill.unwrap_or(256), k)
            .map_err(|e| anyhow::anyhow!(e))?;
        println!("{report}");
        return Ok(());
    }
    if let Some(offsets) = windowed {
        return ppl_windows(
            &mut pipeline,
            &ids,
            &offsets,
            win.window_len,
            o1,
            o1_prefill,
        );
    }
    if let Some(cfg) = o1.cfg()? {
        // Single-prefix o1 scoring: seal after --o1-prefill (default:
        // half the sequence), score the rest through the O(1) kernel.
        pipeline.set_o1(Some(cfg));
        let prefill = o1_prefill
            .unwrap_or(ids.len() / 2)
            .min(ids.len().saturating_sub(1));
        let (n_o1, c) = pipeline
            .nll_ids_o1(&ids, prefill)
            .map_err(|e| anyhow::anyhow!(e))?;
        pipeline.set_o1(None);
        let (n_ex, _) = pipeline
            .nll_ids_from(&ids, prefill)
            .map_err(|e| anyhow::anyhow!(e))?;
        report_o1_ppl(n_o1, n_ex, c, prefill, ids.len());
        dump_moe_stats(&pipeline)?;
        return Ok(());
    }
    if route_dynamic {
        let n = pipeline.enable_dynamic_routing();
        let (ppl, switches) = pipeline
            .ppl_ids_dynamic(&ids)
            .map_err(|e| anyhow::anyhow!(e))?;
        println!(
            "PPL = {ppl:.3} over {} tokens | dynamic routing: {n} skills, {switches} switch(es)",
            ids.len()
        );
        dump_moe_stats(&pipeline)?;
        return Ok(());
    }
    let ppl = pipeline.ppl_ids(&ids).map_err(|e| anyhow::anyhow!(e))?;
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
    let progress = std::env::var("CMF_PPL_PROGRESS").as_deref() == Ok("1");
    let started = std::time::Instant::now();
    for (wi, &off) in offsets.iter().enumerate() {
        let w = &ids[off..off + wlen];
        if let Some(c) = &cfg {
            pipeline.set_o1(Some(c.clone()));
            let (n, k) = pipeline
                .nll_ids_o1(w, prefill)
                .map_err(|e| anyhow::anyhow!(e))?;
            nll_o1 += n;
            pipeline.set_o1(None);
            let (n, k2) = pipeline
                .nll_ids_from(w, prefill)
                .map_err(|e| anyhow::anyhow!(e))?;
            debug_assert_eq!(k, k2, "o1 and exact must score the same tokens");
            nll_ex += n;
            cnt += k;
        } else {
            let (n, k) = pipeline
                .nll_ids_from(w, 0)
                .map_err(|e| anyhow::anyhow!(e))?;
            nll_ex += n;
            cnt += k;
        }
        if progress {
            eprintln!(
                "ppl-window: index={} offset={} scored={} elapsed_ms={} exact_nll={:.6}{}",
                wi + 1,
                off,
                cnt,
                started.elapsed().as_millis(),
                nll_ex,
                if cfg.is_some() {
                    format!(" o1_nll={:.6}", nll_o1)
                } else {
                    String::new()
                }
            );
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
        None => println!(
            "PPL = {:.3} ({})",
            (nll_ex / cnt.max(1) as f64).exp(),
            ppl_operator_label(pipeline)
        ),
    }
    dump_moe_stats(pipeline)?;
    Ok(())
}

/// What the exact-scoring line ran through: the file's natively bounded
/// anchor when it has one (there is no growing KV to be "exact" over),
/// else the full-KV attention.
fn ppl_operator_label(pipeline: &Pipeline) -> String {
    match pipeline.anchor_core.as_ref() {
        Some(ac) => format!(
            "bounded operator {} W={} S={}",
            ac.kind, ac.window, ac.sink
        ),
        None => "exact attention".to_string(),
    }
}

/// B-field of claim 12: expert-routing mass of this run →
/// JSON {layer: [counts]} (CMF_MOE_STATS=file). Works for both
/// teacher-forcing (ppl) and on-policy generation (run) —
/// Routing telemetry records the aggregate observable over the trajectory.
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
        // Only claim a dump when there is one: an architecture with its own
        // expert stack is handled below, and writing "{}" here first made
        // the log say 0 layers over a file that had forty.
        if !parts.is_empty() {
            std::fs::write(&path, format!("{{{}}}", parts.join(",")))?;
            println!("router MoE stats → {path} ({} layers)", parts.len());
        }
    }
    if let Ok(path) = std::env::var("CMF_MOE_STATS") {
        // Architectures with their own expert stack keep their counters
        // there, not on a MoeFfn the generic loop can see.
        let own = cortiq_engine::dsv4::take_route_counts();
        if !own.is_empty() {
            let parts: Vec<String> = own
                .iter()
                .enumerate()
                .filter(|(_, r)| !r.is_empty())
                .map(|(li, r)| {
                    let c: Vec<String> = r.iter().map(u64::to_string).collect();
                    format!("\"{li}\":[{}]", c.join(","))
                })
                .collect();
            std::fs::write(&path, format!("{{{}}}", parts.join(",")))?;
            println!("router MoE stats → {path} ({} layers, dsv4)", parts.len());
        }
    }
    if let Ok(path) = std::env::var("CMF_RMS_TRACE") {
        let mut parts = Vec::new();
        for (li, lw) in pipeline.weights.layers.iter().enumerate() {
            if let cortiq_engine::pipeline::FfnKind::Moe(m) = &lw.ffn {
                let a = m.act_sq.borrow();
                if a.is_empty() {
                    continue;
                }
                let v: Vec<String> = a.iter().map(|x| format!("{x:.6e}")).collect();
                parts.push(format!("\"{li}\":[{}]", v.join(",")));
            }
        }
        std::fs::write(&path, format!("{{{}}}", parts.join(",")))?;
        println!("RMS activation traces → {path} ({} layers)", parts.len());
    }
    if let Ok(spec) = std::env::var("CMF_ACT_DUMP") {
        // "<path>:<layer>,<layer>" — raw f32 rows per requested layer, so an
        // offline tool can form the activation covariance AWNP projects into.
        let (path, want) = spec.split_once(':').unwrap_or((spec.as_str(), ""));
        let want: Vec<usize> = want
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        for (li, lw) in pipeline.weights.layers.iter().enumerate() {
            if !want.is_empty() && !want.contains(&li) {
                continue;
            }
            if let cortiq_engine::pipeline::FfnKind::Moe(m) = &lw.ffn {
                let rows = m.act_rows.borrow();
                if rows.is_empty() {
                    continue;
                }
                let mut bytes = Vec::with_capacity(rows.len() * 4);
                for v in rows.iter() {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(format!("{path}.{li}.f32"), &bytes)?;
                println!(
                    "activations layer {li} → {path}.{li}.f32 ({} floats)",
                    rows.len()
                );
            }
        }
    }
    Ok(())
}

fn cmd_route(model_path: &str, prompt: &str) -> anyhow::Result<()> {
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let mut pipeline = Pipeline::from_model(&model, SamplerConfig::default())?;
    if cortiq_engine::router::is_router_v2(&model) {
        let d = cortiq_engine::router::route_request(&model, &mut pipeline, prompt);
        for s in &d.routing.scores {
            println!(
                "  {:<20} E = {:.4}  err = {:.3e}  p = {:.3}",
                s.id, s.error, s.raw_error, s.probability
            );
        }
        println!("{}", d.describe());
        return Ok(());
    }
    let ids = pipeline.tokenizer.encode(prompt);
    let tau = std::env::var("CMF_OOD_TAU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.30);
    let r = cortiq_engine::router::route_full(&model, &mut pipeline, &ids, tau);
    if r.scores.is_empty() {
        println!("no routable skills in this container");
        return Ok(());
    }
    for s in &r.scores {
        if r.calibrated {
            println!(
                "  {:<20} E = {:.4}  err = {:.3e}  p = {:.3}",
                s.id, s.error, s.raw_error, s.probability
            );
        } else {
            println!("  {:<20} E = {:.4}", s.id, s.error);
        }
    }
    if r.calibrated {
        println!(
            "winner: {}  confidence {:.3}  margin {:.3}  novelty {:.3} → {}",
            r.scores[0].id,
            r.confidence,
            r.margin,
            r.novelty,
            if r.is_novel {
                "OOD (novel)"
            } else {
                "in-scope"
            }
        );
    } else {
        println!(
            "winner: {}  ({}; uncalibrated file: E_min vs τ={tau})",
            r.scores[0].id,
            if r.is_novel { "OOD" } else { "in-scope" }
        );
    }
    Ok(())
}

/// Introspection without generation (ROADMAP A4): show recon-argmin skill
/// selection (with E) and the first-token distribution the routed model
/// would emit, plus its softmax confidence. Everything shown is a
/// quantity already computed by the runtime — no synthesis.
fn cmd_explain(
    model_path: &str,
    prompt: &str,
    top: usize,
    pin: Option<&str>,
    include_quarantine: bool,
) -> anyhow::Result<()> {
    use cortiq_core::knowledge::BACKBONE_CLASS_ID as BACKBONE;
    use cortiq_engine::router::{RouteDecision, RouteTarget};
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let mut probe = Pipeline::from_model(&model, SamplerConfig::default())?;
    let ids = probe.tokenizer.encode(prompt);
    if ids.is_empty() {
        anyhow::bail!("empty prompt: nothing to explain");
    }
    println!("\n\x1b[1m🔍 explain: {model_path}\x1b[0m");
    println!("Prompt: {prompt:?}  ({} tokens)", ids.len());

    // ── Routing: the per-request decision (router v2: backbone-gated;
    //    legacy: recon-argmin, a novel input runs the backbone) ──
    let v2 = cortiq_engine::router::is_router_v2(&model);
    let decision = match pin {
        Some("none") | Some("backbone") => {
            RouteDecision::forced(RouteTarget::Backbone, "pinned: --skill none")
        }
        Some(id) => {
            if !model.header.skills.iter().any(|s| s.id == id) {
                anyhow::bail!("--skill {id}: no such skill in {model_path}");
            }
            RouteDecision::forced(
                RouteTarget::Skill(id.to_string()),
                format!("pinned: --skill {id}"),
            )
        }
        None => cortiq_engine::router::route_request_with(
            &model,
            &mut probe,
            prompt,
            cortiq_engine::router::RouteOptions { include_quarantine },
        ),
    };
    // explain shows the first token of the RAW prompt; a cmf-im-v1 skill
    // is shown on the rendered turn it would really run on (PHI-1).
    let (decision, render_contract) = cortiq_engine::router::enforce_prompt_contract(
        &model.header,
        decision,
        cortiq_engine::router::PromptFrame::Raw,
        true,
    );
    let ids = if render_contract {
        probe
            .tokenizer
            .encode(&cortiq_engine::router::render_cmf_im_v1(prompt))
    } else {
        ids
    };
    if v2 || pin.is_some() || !decision.routing.scores.is_empty() {
        eprintln!("{}", decision.describe());
    }
    let routes = &decision.routing.scores;
    if routes.is_empty() {
        if pin.is_some() || v2 {
            println!(
                "\nRouting: {} — {}",
                decision.target_label(),
                decision.reason
            );
        } else {
            println!("\nSwarm: none (flat model) — no routing needed, the backbone answers.");
        }
    } else {
        if v2 {
            println!(
                "\n\x1b[1mRouting (router v2, backbone-gated; E = unit recon error of φ over the \
                 user text, lower = closer):\x1b[0m"
            );
        } else {
            println!(
                "\n\x1b[1mRouting (recon-argmin, E=‖r−BBᵀr‖²/‖φ‖², lower = more coherent):\x1b[0m"
            );
        }
        let emax = routes
            .iter()
            .map(|r| r.error)
            .fold(0.0f32, f32::max)
            .max(1e-6);
        let chosen = decision.skill().unwrap_or(BACKBONE);
        for r in routes.iter() {
            // Bar: shorter = lower E = more coherent (inverse scale).
            let fill = ((1.0 - r.error / emax) * 20.0).round() as usize;
            let bar = "█".repeat(fill);
            let mark = if r.id == chosen {
                "  \x1b[1m← chosen\x1b[0m"
            } else {
                ""
            };
            println!("  {:<12} E = {:.4}  {}{}", r.id, r.error, bar, mark);
        }
        println!(
            "Decision: {} — {}",
            decision.target_label(),
            decision.reason
        );
    }
    let winner: Option<String> = decision.skill().map(str::to_string);

    // ── First token: distribution and confidence (softmax probability) ──
    // Apply the chosen skill to show EXACTLY the routed answer.
    let mut pipeline = match &winner {
        Some(id) => Pipeline::from_model_with_skill(&model, SamplerConfig::default(), Some(id))?,
        // The backbone: the probe pipeline itself (plain, no overlay). No
        // second Pipeline: `probe` already holds the same backbone and
        // tokenizer, and keeping two GPU-backed pipelines alive made wgpu
        // teardown/resource reuse racy in this introspection-only command.
        None => probe,
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
        let piece = pipeline
            .tokenizer
            .decode_token(*id as u32)
            .replace('\n', "⏎");
        let fill = (p * 30.0).round() as usize;
        println!(
            "  {}  {:>5.1}%  {:?}",
            "█".repeat(fill.max(1)),
            p * 100.0,
            piece
        );
    }
    let top1 = probs[0].1;
    println!(
        "Confidence on the 1st token: {} (top-1 probability)",
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
/// scores the model's softmax confidence against whether its argmax was
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
    let mut ids = pipeline
        .tokenizer
        .with_bos(pipeline.tokenizer.encode(&text));
    ids.truncate(max_tokens);
    let temps: Vec<f32> = vec![0.5, 0.65, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2];
    let t1 = temps.iter().position(|&t| (t - 1.0).abs() < 1e-6).unwrap();

    println!("\n\x1b[1m🎯 Confidence calibration: {model_path}\x1b[0m");
    println!(
        "Held-out: {file}  ({} tokens){}",
        ids.len(),
        skill.map(|s| format!(", skill {s}")).unwrap_or_default()
    );
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
        acc * 100.0,
        conf_raw * 100.0
    );
    let verdict = if conf_raw > acc + 0.02 {
        "overconfident"
    } else if conf_raw + 0.02 < acc {
        "underconfident"
    } else {
        "well calibrated"
    };
    println!(
        "Raw confidence: \x1b[1m{verdict}\x1b[0m (ECE = {:.3})",
        ece_raw
    );

    // Reliability diagram at T=1: conf-bin vs actual accuracy.
    println!("\n  reliability diagram (T=1):  bin  conf   acc    n");
    for (b, &(mc, ac, nb)) in diag_raw.iter().enumerate() {
        if nb == 0 {
            continue;
        }
        let bar = "█".repeat((ac * 20.0).round() as usize);
        let sign = if mc > ac + 0.03 {
            "↑over"
        } else if ac > mc + 0.03 {
            "↓under"
        } else {
            "·"
        };
        println!(
            "   {:>2}0%   {:>4.0}%  {:>4.0}%  {:>4}  {} {}",
            b,
            mc * 100.0,
            ac * 100.0,
            nb,
            bar,
            sign
        );
    }

    let (bt, bece) = (temps[best.0], best.1);
    println!("\n  ECE by temperature:");
    for (t, e) in &eces {
        let mark = if (*t - bt).abs() < 1e-6 {
            "  ← best"
        } else {
            ""
        };
        println!("   T={:<4} ECE {:.3}{}", t, e, mark);
    }
    if (bt - 1.0).abs() < 1e-6 || bece + 0.005 > ece_raw {
        println!(
            "\n\x1b[1mVerdict: already calibrated (T≈1).\x1b[0m No separate field needed — \
                  softmax probability is the reported confidence."
        );
    } else {
        println!(
            "\n\x1b[1mVerdict: temperature T={bt} lowers ECE {:.3}→{:.3}\x1b[0m ({:.0}% of calibration error removed).",
            ece_raw,
            bece,
            (1.0 - bece / ece_raw.max(1e-6)) * 100.0
        );
        println!(
            "Write into header (additive): \x1b[2mpython converter/set_calibration.py {model_path} --temperature {bt}\x1b[0m"
        );
        println!(
            "The runtime will apply it to --confidence/--trace/explain (calibrated probability)."
        );
    }
    Ok(())
}

/// Colour a token by the model's confidence (softmax probability): a 5-step ramp
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
/// a measured quantity: softmax confidence, the active skill, the
/// recon-coherence E (‖r−BBᵀr‖²/‖φ‖², low = coherent with the skill's
/// subspace), and a ▸ marker where the hysteresis router crossed a domain
/// boundary. With `json`, each row is also emitted as JSONL on stderr.
fn render_trace(traces: &[cortiq_engine::TokenTrace], pipeline: &Pipeline, json: bool) {
    if traces.is_empty() {
        return;
    }
    let has_routing = traces
        .iter()
        .any(|t| t.active_skill.is_some() || t.recon.is_some());
    println!("\n\x1b[1mtrace ({} tokens):\x1b[0m", traces.len());
    if has_routing {
        println!(
            "  {:>4}  {:<12}  {:>5}  {:<10}  {:>7}",
            "#", "token", "conf", "skill", "E"
        );
    } else {
        println!("  {:>4}  {:<12}  {:>5}", "#", "token", "conf");
    }
    for tr in traces {
        let piece = pipeline.tokenizer.decode_token(tr.token_id);
        let shown: String = piece
            .chars()
            .take(12)
            .collect::<String>()
            .replace('\n', "⏎");
        let conf = conf_colour(&format!("{:>4.0}%", tr.confidence * 100.0), tr.confidence);
        if has_routing {
            let skill = tr.active_skill.as_deref().unwrap_or("—");
            let sw = if tr.switched { " ▸" } else { "" };
            let e = tr
                .recon
                .map(|e| format!("{e:.4}"))
                .unwrap_or_else(|| "—".into());
            println!(
                "  {:>4}  {:<12}  {}  {:<10}  {:>7}{sw}",
                tr.t, shown, conf, skill, e
            );
        } else {
            println!("  {:>4}  {:<12}  {}", tr.t, shown, conf);
        }
        if json {
            let sk = tr
                .active_skill
                .as_deref()
                .map(|s| format!("\"{s}\""))
                .unwrap_or_else(|| "null".into());
            let rc = tr
                .recon
                .map(|e| format!("{e:.6}"))
                .unwrap_or_else(|| "null".into());
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

/// The frame `run` generates a prompt in ([`chat_mode`] decides): the
/// file's chat template — cmf-im-v1 when it spells the `<|im_start|>` /
/// `<|im_end|>` frame, another template otherwise — or a raw completion.
fn run_prompt_frame(
    model: &CmfModel,
    raw: bool,
    resuming: bool,
) -> cortiq_engine::router::PromptFrame {
    use cortiq_engine::router::PromptFrame;
    let tpl = model
        .header
        .tokenizer_config
        .as_ref()
        .and_then(|t| t.chat_template.as_deref());
    if !chat_mode(tpl.is_some(), raw, resuming) {
        return PromptFrame::Raw;
    }
    match tpl {
        Some(t) if t.contains("<|im_start|>") && t.contains("<|im_end|>") => PromptFrame::CmfImV1,
        _ => PromptFrame::Other,
    }
}

/// Build the exact V4.1 harmony prompt and image records for the CLI. The
/// `run --prompt ... --image ...` path uses the same tagged text and image
/// block normalization as the OpenAI endpoint.
fn prepare_dsv41_cli_prompt(
    model: &CmfModel,
    pipeline: &Pipeline,
    prompt: &str,
    image_paths: &[String],
    no_think: bool,
    reasoning_effort: Option<&str>,
) -> anyhow::Result<PreparedVlInputs> {
    let tagged = dsv41_encoding::parse_tagged_text(prompt).map_err(|e| anyhow::anyhow!(e))?;
    let content = if image_paths.is_empty() {
        tagged
    } else {
        let mut blocks = match tagged {
            serde_json::Value::Array(blocks) => blocks,
            value => vec![serde_json::json!({"type":"text", "text":value})],
        };
        blocks.extend(image_paths.iter().map(|path| {
            serde_json::json!({
                "type":"image_url",
                "image_url":{"url":path},
            })
        }));
        serde_json::Value::Array(blocks)
    };
    prepare_dsv41_cli_messages(
        model,
        pipeline,
        &[serde_json::json!({"role":"user", "content":content})],
        no_think,
        reasoning_effort,
    )
}

fn prepare_dsv41_cli_messages(
    model: &CmfModel,
    pipeline: &Pipeline,
    messages: &[serde_json::Value],
    no_think: bool,
    reasoning_effort: Option<&str>,
) -> anyhow::Result<PreparedVlInputs> {
    let source = model
        .arch()
        .deepseek_v41
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DeepSeek-V4.1 source config is missing"))?;
    let vision = VisionConfig::from_source(source).map_err(|e| anyhow::anyhow!(e))?;
    let mode = if no_think || reasoning_effort.is_none() {
        ThinkingMode::Chat
    } else {
        ThinkingMode::Thinking
    };
    let effort = reasoning_effort
        .map(ReasoningEffort::parse)
        .transpose()
        .map_err(|e| anyhow::anyhow!(e))?;
    let messages = messages
        .iter()
        .map(|message| {
            let mut message = message.clone();
            if let Some(text) = message.get("content").and_then(|value| value.as_str()) {
                let tagged = dsv41_encoding::parse_tagged_text(text)
                    .map_err(|error| anyhow::anyhow!(error))?;
                if tagged.is_array() {
                    message["content"] = tagged;
                }
            }
            Ok::<_, anyhow::Error>(message)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let encoded = dsv41_encoding::encode_messages(
        &messages,
        &EncodeOptions {
            thinking_mode: mode,
            reasoning_effort: effort,
            ..EncodeOptions::default()
        },
    )
    .map_err(|e| anyhow::anyhow!(e))?;
    dsv41_vision::prepare_vl_inputs(
        &encoded.prompt,
        &encoded.images,
        &pipeline.tokenizer,
        &vision,
    )
    .map_err(|e| anyhow::anyhow!(e))
}

/// Token ids for `CMF_PROMPT_IDS`: a JSON array (`[1, 2, 3]`) or plain ids
/// separated by commas and/or whitespace. An empty list is an error.
fn parse_prompt_ids(text: &str) -> anyhow::Result<Vec<u32>> {
    let body = text.trim().trim_start_matches('[').trim_end_matches(']');
    let ids = body
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u32>()
                .map_err(|e| anyhow::anyhow!("CMF_PROMPT_IDS: bad token id {s:?}: {e}"))
        })
        .collect::<anyhow::Result<Vec<u32>>>()?;
    anyhow::ensure!(!ids.is_empty(), "CMF_PROMPT_IDS: no token ids");
    Ok(ids)
}

fn read_prompt_ids(path: &std::path::Path) -> anyhow::Result<Vec<u32>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("CMF_PROMPT_IDS: cannot read {}: {e}", path.display()))?;
    parse_prompt_ids(&text)
}

#[allow(clippy::too_many_arguments)]
async fn cmd_run(
    model_path: &str,
    task: &str,
    prompt: Option<&str>,
    images: &[String],
    videos: &[String],
    audios: &[String],
    system: Option<&str>,
    speech_out: Option<(&std::path::Path, &str)>,
    media_options: &cortiq_engine::mimo_ingress::MediaOptions,
    reasoning_effort: Option<&str>,
    max_tokens: usize,
    skill: Option<&str>,
    greedy: bool,
    temperature: Option<f32>,
    rep_penalty: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<u64>,
    raw: bool,
    no_think: bool,
    blend: Option<&str>,
    route_dynamic: bool,
    route_include_quarantine: bool,
    confidence: bool,
    trace: bool,
    trace_json: bool,
    state: Option<&str>,
    o1: &O1Flags,
    peer: Option<&str>,
    peer_split: Option<usize>,
    net_token: Option<&str>,
    net_dtype: &str,
    peer_head: bool,
    peer_run_ahead: u32,
    peer_prefill: bool,
    gpus: Option<usize>,
    lookup_mode: Option<&str>,
) -> anyhow::Result<()> {
    use cortiq_engine::lookup::{LookupMode, LookupOutcome, LookupTable, LookupTables};
    let lookup_mode = LookupMode::resolve(lookup_mode).map_err(anyhow::Error::msg)?;
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
            anyhow::bail!(
                "unsupported .cmfstate kind {} (this build reads logical only)",
                st.kind
            );
        }
        if st.fp != SessionState::fingerprint(model.arch()) {
            anyhow::bail!(
                "state was frozen from a different model (fingerprint {:?} ≠ {:?})",
                st.fp,
                SessionState::fingerprint(model.arch())
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
    // A file that carries routable skills routes by default (spec §9 —
    // selection is a property of the file): the prompt picks its
    // specialist with no flag at all. `--skill <id>` pins one,
    // `--skill none` forces the backbone, `--blend`/`--route-dynamic`
    // take their own paths.
    //
    // A file with router policy v2 (ROUTER_V2) routes per REQUEST with
    // the backbone-gated decision: the backbone is the default, a skill
    // runs only when the calibrated decision says so, and the decision
    // goes to stderr. The backbone runs the plain pipeline (no overlay).
    let v2 = cortiq_engine::router::is_router_v2(&model);
    let mut route_dynamic = route_dynamic;
    if route_dynamic && v2 {
        eprintln!(
            "route-dynamic: {model_path} declares router policy v2 (granularity \"request\") — \
             per-token switching is off; the request-level decision applies"
        );
        route_dynamic = false;
    }
    let routable = v2 || model.header.skills.iter().any(|s| s.selection.is_some());
    if skill.is_none() && blend.is_none() && !route_dynamic && routable && prompt.is_some() {
        skill = Some("auto".to_string());
    }
    if v2 && skill.is_none() && blend.is_none() && prompt.is_none() {
        eprintln!(
            "route: backbone | interactive session on a router-v2 file (pass --skill <id> to pin \
             a skill)"
        );
    }
    if matches!(skill.as_deref(), Some("none") | Some("backbone")) {
        if v2 {
            eprintln!("route: backbone | pinned: --skill none");
        }
        skill = None;
    }
    // The frame `run` generates with (the file's template, or a raw
    // completion) against the chosen skill's prompt_contract — one engine
    // check for run/explain/serve (PHI-1). A cmf-im-v1 skill on a raw
    // prompt runs on the rendered turn (the frame it was trained and gated
    // under); with a raw frozen `--state` prefix it cannot, and the
    // backbone runs.
    let frame = run_prompt_frame(&model, raw, state.is_some());
    let can_render = state.is_none();
    let mut render_contract = false;
    let mut decision: Option<cortiq_engine::router::RouteDecision> = None;
    // A pinned lookup record in an interactive session: the table is
    // consulted per turn (see the loop below); the backbone pipeline runs.
    let mut lookup_pinned: Option<LookupTable> = None;
    // The router's own decision (not a pinned skill): a `key_first` lookup
    // record may take it on a strong key of the prompt.
    let mut key_first_gate: Option<cortiq_engine::lookup::KeyFirstGate> = None;
    if skill.as_deref() == Some("auto") {
        let mut probe = Pipeline::from_model(&model, SamplerConfig::default())?;
        let route_opts = cortiq_engine::router::RouteOptions {
            include_quarantine: route_include_quarantine,
        };
        key_first_gate = Some(
            cortiq_engine::lookup::KeyFirstGate::new(route_opts, frame).can_render(can_render),
        );
        let d = cortiq_engine::router::route_request_with(
            &model,
            &mut probe,
            prompt.unwrap_or(""),
            route_opts,
        );
        let (d, render) =
            cortiq_engine::router::enforce_prompt_contract(&model.header, d, frame, can_render);
        render_contract = render;
        skill = d.skill().map(str::to_string);
        if !v2 {
            match skill.as_deref() {
                Some(id) => println!("routed to skill: {id}"),
                None => println!("routed to skill: <none> — {}", d.reason),
            }
        }
        decision = Some(d);
    } else if let Some(id) = skill.clone() {
        if v2 && !model.header.skills.iter().any(|s| s.id == id) {
            anyhow::bail!("--skill {id}: no such skill in {model_path}");
        }
        // A pinned skill obeys its prompt contract too.
        let pinned = cortiq_engine::router::RouteDecision::forced(
            cortiq_engine::router::RouteTarget::Skill(id.clone()),
            format!("pinned: --skill {id}"),
        );
        let (d, render) = cortiq_engine::router::enforce_prompt_contract(
            &model.header,
            pinned,
            frame,
            can_render,
        );
        if d.skill().is_none() {
            anyhow::bail!("--skill {id}: {}", d.reason);
        }
        render_contract = render;
        if prompt.is_none() && LookupTable::is_lookup(&model, &id) {
            if v2 {
                eprintln!("{}", d.describe());
            }
            eprintln!(
                "lookup: {id} | pinned in an interactive session — every turn is looked up \
                 (mode {}); the backbone pipeline runs",
                lookup_mode.label()
            );
            lookup_pinned = Some(LookupTable::open(&model, &id).map_err(anyhow::Error::msg)?);
            skill = None;
            render_contract = false;
        } else {
            decision = Some(d);
        }
    }
    // The lookup step (spec §9.5.2): a decision for a `lookup` record has
    // no lane. `answer` — the table's text is the answer and nothing is
    // generated; `context` — the backbone generates with the card
    // prepended; `off` or no key in the prompt — the backbone runs the
    // plain prompt, unchanged.
    let mut lookup_context: Option<String> = None;
    if let Some(d) = decision.take() {
        let tables = LookupTables::new(model.clone());
        let (d, outcome) = cortiq_engine::lookup::resolve_lookup_gated(
            &tables,
            d,
            key_first_gate,
            &[prompt.unwrap_or("")],
            lookup_mode,
        )
        .map_err(anyhow::Error::msg)?;
        if v2 {
            eprintln!("{}", d.describe());
        }
        if let Some(line) = outcome.describe() {
            eprintln!("{line}");
        }
        match outcome {
            LookupOutcome::NotLookup => {}
            LookupOutcome::Off { .. } | LookupOutcome::Miss { .. } => {
                skill = None;
                render_contract = false;
            }
            LookupOutcome::Answer(a) => {
                if let Some(p) = prompt {
                    println!("\nPrompt: {p}\n");
                }
                println!("{}", a.text);
                println!(
                    "\n[lookup '{}': key {:?} → entry {}, {}, field {}, decided by {}; answered \
                     from the table, nothing generated]",
                    a.id,
                    a.key.key,
                    a.key.entry,
                    a.lang,
                    a.field.as_deref().unwrap_or("— (whole card)"),
                    a.decided_by.label()
                );
                return Ok(());
            }
            LookupOutcome::Context(a) => {
                lookup_context = Some(a.card);
                skill = None;
                render_contract = false;
            }
        }
    }
    // The file's own sampling recommendation (`convert --sampling`) is the
    // starting point; every flag below still overrides it.
    let prov = model.header.provenance.as_ref();
    let mut sampler = SamplerConfig::from_provenance(prov);
    if SamplerConfig::file_has_defaults(prov) && !greedy {
        eprintln!(
            "sampling: the file's defaults (temperature {}, top-p {}, top-k {}, min-p {}, \
             repetition penalty {}); flags override",
            sampler.temperature,
            sampler.top_p,
            sampler.top_k,
            sampler.min_p,
            sampler.repetition_penalty
        );
    }
    if greedy {
        sampler.temperature = 0.0;
        sampler.repetition_penalty = 1.0;
    }
    if let Some(t) = temperature {
        sampler.temperature = t;
    }
    if let Some(r) = rep_penalty {
        sampler.repetition_penalty = r;
    }
    if let Some(v) = top_p {
        sampler.top_p = v;
    }
    if let Some(v) = top_k {
        sampler.top_k = v;
    }
    if let Some(v) = min_p {
        sampler.min_p = v;
    }
    if let Some(v) = presence_penalty {
        sampler.presence_penalty = v;
    }
    if let Some(s) = seed {
        sampler.seed = Some(s);
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
    o1.apply(&mut pipeline)?;
    // CMF_IGNORE_EOS=1: decode the full --max-tokens (backend parity runs
    // compare a fixed number of steps; `bench --ignore-eos` is the twin).
    if std::env::var("CMF_IGNORE_EOS").as_deref() == Ok("1") {
        pipeline.ignore_eos = true;
    }
    let is_dsv41 = model.arch().arch_name == "deepseek_v41";
    let is_mimo = model.arch().arch_name == "mimo_v2";
    let mimo_media = is_mimo && (!images.is_empty() || !videos.is_empty() || !audios.is_empty());
    let lfm3_vl = cortiq_engine::lfm3_vl::has_vision(&model);
    let lfm3_audio = cortiq_engine::lfm3_audio::has_audio(&model);
    // LFM3-Omni (vision + speech) takes every medium through the VL path.
    let lfm3_media = lfm3_vl && (!images.is_empty() || !videos.is_empty() || (lfm3_audio && !audios.is_empty()));
    let lfm3_audio_media = lfm3_audio && !lfm3_vl && !audios.is_empty();
    // Mage-VL (`llava_onevision2`): images through the OneVision tower.
    let ov2_media = cortiq_engine::onevision2::has_vision(&model) && !images.is_empty();
    // SenseNova-Loop: images through the Qwen3-VL tower, MRoPE rows.
    let sn_media = cortiq_engine::sensenova::has_vision(&model) && !images.is_empty();
    let ov2_media_or_sn = ov2_media || sn_media;
    // A text answer from a speech model: no speech or mixed turn.
    if lfm3_audio {
        let ids = cortiq_engine::lfm3_audio::text_only_suppress(&pipeline.tokenizer);
        pipeline.sampler_config.suppress_tokens.extend(ids);
    }
    anyhow::ensure!(is_dsv41 || reasoning_effort.is_none(), "--reasoning-effort requires DeepSeek-V4.1");
    anyhow::ensure!(is_dsv41 || is_mimo || lfm3_media || ov2_media_or_sn || images.is_empty(),
        "--image requires DeepSeek-V4.1, MiMo, an LFM3-VL, Mage-VL or SenseNova-Loop file");
    anyhow::ensure!(!ov2_media_or_sn || (!raw && state.is_none() && peer.is_none() && gpus.is_none()
        && std::env::var_os("CMF_PROMPT_IDS").is_none()),
        "Mage-VL images require a fresh local chat prompt (no --raw/--state/--peer/--gpus/CMF_PROMPT_IDS)");
    anyhow::ensure!(is_mimo || ((audios.is_empty() || lfm3_audio) && media_options.companion.is_none()
        && ((videos.is_empty() && media_options.video_fps.is_none() && media_options.image_max_pixels.is_none())
            || lfm3_vl
            || (ov2_media_or_sn && videos.is_empty() && media_options.video_fps.is_none()))),
        "--video/--audio/--mm and MiMo processing options require MiMo (--video and its options: or LFM3-VL; --audio: or LFM3-Audio)");
    anyhow::ensure!(!lfm3_audio_media || (!raw && state.is_none() && peer.is_none() && gpus.is_none()
        && std::env::var_os("CMF_PROMPT_IDS").is_none()),
        "LFM3-Audio clips require a fresh local chat prompt (no --raw/--state/--peer/--gpus/CMF_PROMPT_IDS)");
    anyhow::ensure!(!lfm3_media || (!raw && state.is_none() && peer.is_none() && gpus.is_none()
        && std::env::var_os("CMF_PROMPT_IDS").is_none()),
        "LFM3-VL images require a fresh local chat prompt (no --raw/--state/--peer/--gpus/CMF_PROMPT_IDS)");
    anyhow::ensure!((images.is_empty() && videos.is_empty() && audios.is_empty()) || prompt.is_some(),
        "media input requires --prompt so media order is unambiguous");
    anyhow::ensure!(!mimo_media || (!raw && state.is_none() && peer.is_none() && gpus.is_none()
        && std::env::var_os("CMF_PROMPT_IDS").is_none()),
        "MiMo media requires a fresh local chat prompt (no --raw/--state/--peer/--gpus/CMF_PROMPT_IDS)");
    // Same rule for the CLI: the vocabulary-wide softmax runs when its
    // output is going to be shown, not on every run.
    pipeline.set_confidence(confidence || trace);
    if let Some(n) = gpus {
        // In-process split: segment i runs pinned to card i and hands
        // the next one a hidden vector that never leaves this address
        // space — no worker process, no socket, no serialization.
        let have = cortiq_engine::gpu::device_count();
        if n < 2 {
            anyhow::bail!("--gpus {n}: needs ≥ 2 cards (pin a single one with CMF_GPU_ADAPTER)");
        }
        if have < n {
            anyhow::bail!("--gpus {n}, but {have} cards are visible (see `cortiq gpu`)");
        }
        let devs: Vec<usize> = (0..n).collect();
        pipeline
            .set_gpu_plan_at(Some(&devs), peer_split)
            .map_err(|e| anyhow::anyhow!("--gpus {n}: {e}"))?;
        if let Some(plan) = pipeline.gpu_plan() {
            let seg = plan
                .iter()
                .map(|(d, a, b)| format!("card {d}: layers {a}..={b}"))
                .collect::<Vec<_>>()
                .join(" · ");
            println!("gpus: {seg}");
        }
    }
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
    // Patent-2 link: a skill may carry its DTG-MA task mask
    // (input_mask_task) — activate it with the skill unless the user
    // pinned a task explicitly.
    let mut task = task.to_string();
    if task == "general" {
        if let Some(sid) = skill.as_deref() {
            if let Some(mt) = model
                .header
                .skills
                .iter()
                .find(|r| r.id == sid)
                .and_then(|r| r.input_mask_task.clone())
            {
                println!("skill '{sid}' carries task mask '{mt}' — sparse execution on");
                task = mt;
            }
        }
    }
    let task = task.as_str();
    let runtime = CortiqRuntime::new(model.clone());

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

    // ── Network pipeline-split (v1): one worker holds the tail layers ──
    // Backbone-only and mask-free on purpose: the worker verified it holds
    // the same FILE (dir_hash), but a skill/blend/mask is run-time state
    // this side only — running half a stack with it would be a chimera.
    let mut remote_opt: Option<cortiq_net::RemoteSegment> = None;
    if let Some(addr) = peer {
        if blend.is_some() || route_dynamic {
            anyhow::bail!(
                "--peer does not run --blend/--route-dynamic yet: a blend materializes \
                 mixed weights this side only, and dynamic routing switches mid-decode — \
                 both would desync the worker"
            );
        }
        if state.is_some() {
            anyhow::bail!("--peer does not resume .cmfstate sessions yet");
        }
        pipeline.split_supported().map_err(|e| anyhow::anyhow!(e))?;
        let nl = pipeline.num_layers;
        let split = peer_split.unwrap_or(nl / 2);
        if split >= nl {
            anyhow::bail!("--peer-split {split} out of range: the model has {nl} layers");
        }
        let dtype = match net_dtype {
            "f32" => cortiq_net::WireDtype::F32,
            "f16" => cortiq_net::WireDtype::F16,
            other => anyhow::bail!("--net-dtype {other}: expected f32 or f16"),
        };
        // The session spec the worker mirrors: skill overlay, mask task
        // (each side masks its own layers), o1 config over its span.
        let spec = cortiq_net::SessionSpec {
            skill: skill.clone(),
            task: mask.as_ref().map(|_| task.to_string()),
            o1: o1.spec.as_deref().map(|s| cortiq_net::O1Wire {
                spec: s.to_string(),
                m: o1.m.map(|v| v as u32),
                w: o1.w.map(|v| v as u32),
                sink: o1.sink.map(|v| v as u32),
                rect: o1.rect.clone(),
            }),
            head: peer_head,
            // The far side must sample with THIS run's settings —
            // greedy here and a default temperature there would be a
            // different answer with no error anywhere.
            sampler: if peer_head {
                Some(serde_json::to_string(&pipeline.sampler_config)?)
            } else {
                None
            },
            run_ahead: peer_run_ahead,
        };
        let mut rs = cortiq_net::RemoteSegment::connect(
            addr,
            net_token.unwrap_or(""),
            runtime.model().dir_hash(),
            &runtime.model().arch().arch_name,
            nl,
            pipeline.hidden_size,
            split,
            nl - 1,
            dtype,
            &spec,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        // What the peer is worth RIGHT NOW, not what it was worth once.
        // A phone whose governor let the clock fall serves the same span
        // at half the speed with no temperature change, so the clock
        // percentage here is the first thing to read when a split is slow.
        match rs.stats() {
            Ok(st) => println!("peer state: {}", st.summary()),
            Err(e) => eprintln!("peer state: unavailable ({e})"),
        }
        println!(
            "peer {addr}: layers {split}..{} remote, 0..{split} local \
             ({}), wire {net_dtype}{}{}{}",
            nl - 1,
            if peer_head {
                "head/sampler REMOTE — this side only tokenizes"
            } else {
                "embed/head/sampler local"
            },
            spec.skill
                .as_deref()
                .map(|s| format!(", skill {s}"))
                .unwrap_or_default(),
            spec.task
                .as_deref()
                .map(|t| format!(", task {t}"))
                .unwrap_or_default(),
            spec.o1
                .as_ref()
                .map(|w| format!(", o1 {}", w.spec))
                .unwrap_or_default(),
        );
        remote_opt = Some(rs);
    }

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

    let noninteractive_generate = prompt.is_some();
    let mut generate_and_print = |pipeline: &mut Pipeline,
                                  ids: &[u32],
                                  vl_inputs: Option<&PreparedVlInputs>,
                                  mimo_rows: Option<&[f32]>|
     -> anyhow::Result<Option<String>> {
        use std::io::Write;
        // Stream silently when the confidence view will reprint coloured;
        // otherwise stream live as before.
        // The moment the first token lands: everything before it is
        // prefill (plus, on a cold process, shader compile and the
        // weight upload); everything after is decode. One number over
        // the whole window read "6 tok/s" on a 30-token answer from
        // a model that decodes at 45 — users took it for a defect.
        let first_tok: std::sync::Arc<std::sync::Mutex<Option<std::time::Instant>>> =
            Default::default();
        let ft = first_tok.clone();
        let mark_first = move || {
            let mut g = ft.lock().unwrap();
            if g.is_none() {
                *g = Some(std::time::Instant::now());
            }
        };
        let cb: cortiq_engine::TokenCallback = if confidence {
            Box::new(move |_tok: &str| {
                mark_first();
                true
            })
        } else {
            Box::new(move |tok: &str| {
                mark_first();
                print!("{tok}");
                let _ = std::io::stdout().flush();
                true
            })
        };
        let started = std::time::Instant::now();
        if (vl_inputs.is_some() || mimo_rows.is_some()) && remote_opt.is_some() {
            anyhow::bail!(
                "DeepSeek-V4.1 multimodal generation is not supported with --peer; use a local pipeline"
            );
        }
        // Prefill offload: the peer absorbs the prompt, its state
        // comes home, and the wire goes idle for the rest of the
        // conversation. Done once — after it the segment is dropped,
        // so a later turn is a plain local generation.
        if peer_prefill {
            if let Some(rs) = remote_opt.as_mut() {
                match cortiq_net::prefill_on_peer(pipeline, rs, ids) {
                    Ok((bytes, pre_s, fetch_s)) => {
                        eprintln!(
                            "offload: {} positions prefilled on the peer in {:.2} s, \
                                 state home in {:.2} s ({:.1} MB, {:.0} MB/s) — decoding locally",
                            ids.len().saturating_sub(1),
                            pre_s,
                            fetch_s,
                            bytes as f64 / 1e6,
                            bytes as f64 / 1e6 / fetch_s.max(1e-9),
                        );
                    }
                    Err(e) => anyhow::bail!("prefill offload: {e}"),
                }
                remote_opt = None;
            }
        }
        let gen_res = match remote_opt.as_mut() {
            Some(rs) => {
                cortiq_net::generate_split(pipeline, rs, ids, max_tokens, mask.as_ref(), Some(cb))
                    .map(|(r, st)| {
                        // The numbers that decide whether this wire pays:
                        // measured, per generation, stderr.
                        if st.remote_steps > 0 {
                            let reused = r.prompt_tokens.saturating_sub(st.prefilled);
                            eprintln!(
                                "\nnet: prefill {:.0} ms ({} of {} pos{}) · {} round trips · \
                                     {:.2} ms avg rtt+remote · {:.0}% of decode wall",
                                st.prefill_s * 1e3,
                                st.prefilled,
                                r.prompt_tokens,
                                if reused > 0 {
                                    format!(", {reused} reused")
                                } else {
                                    String::new()
                                },
                                st.remote_steps,
                                st.net_s * 1e3 / st.remote_steps as f64,
                                100.0 * st.net_s / st.decode_s.max(1e-9),
                            );
                        }
                        r
                    })
            }
            None => match vl_inputs {
                Some(inputs) => {
                    pipeline.generate_from_vl(inputs, max_tokens, mask.as_ref(), Some(cb))
                }
                None => match mimo_rows {
                    Some(rows) => pipeline.generate_from_embeds(ids, rows, max_tokens, mask.as_ref(), Some(cb)),
                    None => pipeline.generate_from_ids(ids, max_tokens, mask.as_ref(), Some(cb)),
                },
            },
        };
        match gen_res {
            Ok(r) => {
                let secs = started.elapsed().as_secs_f64();
                // Confidence view: reprint token-by-token, coloured by the
                // model's softmax probability on each emitted token.
                if confidence && !r.token_confidence.is_empty() {
                    println!();
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
                // decode tok/s over the tokens AFTER the first one, and
                // the first token's latency on its own (prefill, plus
                // shader compile / weight upload on a cold process).
                let first = first_tok
                    .lock()
                    .unwrap()
                    .map(|t| t.duration_since(started).as_secs_f64());
                match first {
                    // Whole blocks arrive at once: the per-token decode
                    // rate of a causal model has no meaning here.
                    Some(ttft) if pipeline.is_block_diffusion() && pipeline.clover_stats().blocks > 0 => println!(
                        "\n[{} tokens · first block {:.1} s · overall {:.1} tok/s, finish: {}]",
                        r.tokens_generated,
                        ttft,
                        r.tokens_generated as f64 / secs.max(1e-9),
                        r.finish_reason
                    ),
                    Some(ttft) if r.tokens_generated >= 2 => println!(
                        "\n[{} tokens · first token {:.1} s · decode {:.1} tok/s · overall {:.1} tok/s, finish: {}]",
                        r.tokens_generated,
                        ttft,
                        (r.tokens_generated - 1) as f64 / (secs - ttft).max(1e-9),
                        r.tokens_generated as f64 / secs.max(1e-9),
                        r.finish_reason
                    ),
                    _ => println!(
                        "\n[{} tokens, {:.1} tok/s, finish: {}]",
                        r.tokens_generated,
                        r.tokens_generated as f64 / secs.max(1e-9),
                        r.finish_reason
                    ),
                }
                if r.mtp_drafted > 0 {
                    println!(
                        "[MTP draft: {} of {} accepted ({:.0}%)]",
                        r.mtp_accepted,
                        r.mtp_drafted,
                        100.0 * r.mtp_accepted as f64 / r.mtp_drafted as f64
                    );
                }
                if pipeline.is_block_diffusion() {
                    let st = pipeline.clover_stats();
                    let per = |x: f64| 1e3 * x / st.passes.max(1) as f64;
                    if st.drafted > 0 {
                        let rd = |x: f64| 1e3 * x / st.rounds as f64;
                        println!(
                            "[self-speculation: {} rounds · {} of {} drafts accepted ({:.0}%) · {:.2} tokens/round · {:.1} tok/s after the prompt · \
                             preview {:.0} ms + verify {:.0} ms + accept {:.1} ms a round · prompt {:.1} s]",
                            st.rounds,
                            st.accepted,
                            st.drafted,
                            100.0 * st.accepted as f64 / st.drafted.max(1) as f64,
                            r.tokens_generated as f64 / st.rounds as f64,
                            st.denoise_tok_s(r.tokens_generated),
                            rd(st.preview_s),
                            rd(st.verify_s),
                            rd(st.pick_s),
                            st.prompt_s
                        );
                    } else if st.blocks > 0 {
                    println!(
                        "[block diffusion: {} blocks · {} passes · {:.2} tokens/pass · {:.1} tok/s after the prompt · \
                         pass {:.0} ms + pick {:.1} ms · block commits {:.2} s · prompt {:.1} s · device layers {}]",
                        st.blocks,
                        st.passes,
                        r.tokens_generated as f64 / st.passes.max(1) as f64,
                        st.denoise_tok_s(r.tokens_generated),
                        per(st.preview_s),
                        per(st.pick_s),
                        st.commit_s,
                        st.prompt_s,
                        st.device_layers
                    );
                    }
                    if st.metal_passes > 0 {
                        let mp = |x: f64| 1e3 * x / st.metal_passes as f64;
                        println!(
                            "[Metal passes: {} · device {:.0} (executing {:.0}) · routing {:.1} · expert fills {:.1} ms/pass · expert cache {:.1}% hits]",
                            st.metal_passes,
                            mp(st.metal_gpu_s),
                            mp(st.metal_exec_s),
                            mp(st.metal_route_s),
                            mp(st.metal_fill_s),
                            100.0 * st.expert_hits as f64 / (st.expert_hits + st.expert_misses).max(1) as f64
                        );
                    }
                    if st.metal_stage_s.iter().any(|&x| x > 0.0) {
                        let mp = |x: f64| 1e3 * x / st.metal_passes.max(1) as f64;
                        let s = st.metal_stage_s;
                        println!(
                            "[Metal stages: gdn {:.0} · attention {:.0} · norm+router {:.0} · experts: gather {:.0} gate|up {:.0} act {:.0} down {:.0} combine {:.0} · head {:.0} ms/pass]",
                            mp(s[0]),
                            mp(s[1]),
                            mp(s[2]),
                            mp(s[3]),
                            mp(s[4]),
                            mp(s[5]),
                            mp(s[6]),
                            mp(s[7]),
                            mp(s[8])
                        );
                    }
                    if st.device_layers == 0 && st.passes > st.metal_passes {
                        println!(
                            "[host passes: gdn {:.0} · attention {:.0} · moe {:.0} · head {:.0} ms/pass]",
                            per(st.gdn_s),
                            per(st.attn_s),
                            per(st.moe_s),
                            per(st.head_s)
                        );
                    }
                }
                #[cfg(feature = "gpu")]
                cortiq_engine::dsv4::profile_report();
                cortiq_engine::dsv41::profile_report();
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
            Err(e) if noninteractive_generate => {
                return Err(anyhow::anyhow!("generation failed: {e}"));
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
        if render_contract && !text.is_empty() {
            // The routed skill's contract (cmf-im-v1) on a raw prompt: the
            // skill continues the frame it was trained and gated under —
            // exactly the rendering dump-logits / probe-utility measure.
            pipeline
                .tokenizer
                .encode(&cortiq_engine::router::render_cmf_im_v1(text))
        } else if use_template && !text.is_empty() {
            pipeline
                .tokenizer
                .apply_chat_template_opts(history, thinking)
        } else {
            let mut ids = resume_prefix.clone();
            ids.extend(pipeline.tokenizer.encode(text));
            // Fresh raw context: honor the tokenizer's BOS contract
            // (llama <s>, gemma <bos>) — word salad without it.
            if resume_prefix.is_empty() {
                ids = pipeline.tokenizer.with_bos(ids);
            }
            ids
        }
    };
    let build_dsv41_inputs = |pipeline: &Pipeline,
                              history: &[(String, String)]|
     -> anyhow::Result<PreparedVlInputs> {
        let messages = history
            .iter()
            .map(|(role, content)| serde_json::json!({"role":role, "content":content}))
            .collect::<Vec<_>>();
        let prepared =
            prepare_dsv41_cli_messages(&model, pipeline, &messages, no_think, reasoning_effort)?;
        Ok(prepared)
    };

    if let Some(p) = prompt {
        println!("\nPrompt: {p}\n");
        // `context` lookup mode: the backbone generates from the message
        // with the reference card prepended.
        let p_text = match &lookup_context {
            Some(card) => {
                eprintln!("lookup: the reference card is prepended to the prompt (mode context)");
                cortiq_engine::lookup::context_prompt(card, p)
            }
            None => p.to_string(),
        };
        let p = p_text.as_str();
        let mut history = vec![("user".to_string(), p.to_string())];
        if let Some(sys) = system {
            history.insert(0, ("system".to_string(), sys.to_string()));
        }
        let vl_inputs = is_dsv41
            .then(|| {
                prepare_dsv41_cli_prompt(&model, &pipeline, p, images, no_think, reasoning_effort)
            })
            .transpose()?;
        let mimo_inputs = if mimo_media {
            let mut content = vec![serde_json::json!({"type":"text", "text":p})];
            content.extend(images.iter().map(|path| serde_json::json!({"type":"image_url", "image_url":{"url":path}})));
            content.extend(videos.iter().map(|path| serde_json::json!({"type":"video", "video":{"path":path}})));
            content.extend(audios.iter().map(|path| serde_json::json!({"type":"audio", "audio":{"url":path}})));
            Some(cortiq_engine::mimo_ingress::prepare(&model, &pipeline,
                &[serde_json::json!({"role":"user", "content":content})], None, thinking, media_options)
                .map_err(anyhow::Error::msg)?)
        } else { None };
        // LFM3-VL: images, then videos, then the text, as in Liquid's examples.
        let lfm3_inputs = if lfm3_media {
            let mut content: Vec<serde_json::Value> = images.iter()
                .map(|path| serde_json::json!({"type":"image_url", "image_url":{"url":path}}))
                .collect();
            content.extend(videos.iter().map(|path| serde_json::json!({"type":"video", "video":{"path":path}})));
            if lfm3_audio {
                content.extend(audios.iter().map(|path| serde_json::json!({"type":"audio", "audio":path})));
            }
            if !p.trim().is_empty() {
                content.push(serde_json::json!({"type":"text", "text":p}));
            }
            let mut messages: Vec<serde_json::Value> = system
                .map(|s| serde_json::json!({"role":"system", "content":s}))
                .into_iter()
                .collect();
            messages.push(serde_json::json!({"role":"user", "content":content}));
            Some(cortiq_engine::lfm3_vl::prepare(&model, &pipeline,
                &messages, None, thinking,
                &cortiq_engine::lfm3_vl::VlOptions {
                    image_max_pixels: media_options.image_max_pixels,
                    video_fps: media_options.video_fps,
                })
                .map_err(anyhow::Error::msg)?)
        } else { None };
        // LFM3-Audio: the clips, then the text (none when empty).
        let lfm3_inputs = if lfm3_audio_media {
            let mut content: Vec<serde_json::Value> = audios.iter()
                .map(|path| serde_json::json!({"type":"audio", "audio":path}))
                .collect();
            if !p.trim().is_empty() {
                content.push(serde_json::json!({"type":"text", "text":p}));
            }
            let mut messages: Vec<serde_json::Value> = system
                .map(|s| serde_json::json!({"role":"system", "content":s}))
                .into_iter()
                .collect();
            messages.push(serde_json::json!({"role":"user", "content":content}));
            Some(cortiq_engine::lfm3_audio::prepare(&model, &pipeline, &messages, None, thinking)
                .map_err(anyhow::Error::msg)?)
        } else { lfm3_inputs };
        // Mage-VL: the images, then the text, as in the model card.
        let ov2_inputs = if ov2_media {
            let mut content: Vec<serde_json::Value> = images.iter()
                .map(|path| serde_json::json!({"type":"image_url", "image_url":{"url":path}}))
                .collect();
            content.push(serde_json::json!({"type":"text", "text":p}));
            let mut messages: Vec<serde_json::Value> = system
                .map(|s| serde_json::json!({"role":"system", "content":s}))
                .into_iter()
                .collect();
            messages.push(serde_json::json!({"role":"user", "content":content}));
            Some(cortiq_engine::onevision2::prepare(&model, &pipeline, &messages, None, thinking,
                media_options.image_max_pixels)
                .map_err(anyhow::Error::msg)?)
        } else { None };
        let sn_inputs = if sn_media {
            let mut content: Vec<serde_json::Value> = images.iter()
                .map(|path| serde_json::json!({"type":"image_url", "image_url":{"url":path}}))
                .collect();
            content.push(serde_json::json!({"type":"text", "text":p}));
            let mut messages: Vec<serde_json::Value> = system
                .map(|s| serde_json::json!({"role":"system", "content":s}))
                .into_iter()
                .collect();
            messages.push(serde_json::json!({"role":"user", "content":content}));
            let prepared = cortiq_engine::sensenova::prepare(&model, &pipeline, &messages, None, thinking,
                media_options.image_max_pixels)
                .map_err(anyhow::Error::msg)?;
            pipeline.prompt_mrope = prepared.mrope.clone();
            Some(prepared)
        } else { None };
        let mut ids = vl_inputs
            .as_ref()
            .map(|inputs| inputs.token_ids.clone())
            .or_else(|| ov2_inputs.as_ref().map(|inputs| inputs.token_ids.clone()))
            .or_else(|| sn_inputs.as_ref().map(|inputs| inputs.token_ids.clone()))
            .or_else(|| mimo_inputs.as_ref().map(|inputs| inputs.token_ids.clone()))
            .or_else(|| lfm3_inputs.as_ref().map(|inputs| inputs.token_ids.clone()))
            .unwrap_or_else(|| build_ids(&pipeline, &history, p));
        // CMF_PROMPT_IDS=<file>: feed exactly these token ids (a JSON array,
        // or ids separated by commas/whitespace) instead of the tokenized
        // prompt — parity runs against an oracle that tokenized elsewhere.
        if let Some(path) = std::env::var_os("CMF_PROMPT_IDS") {
            ids = read_prompt_ids(std::path::Path::new(&path))?;
            eprintln!(
                "CMF_PROMPT_IDS: {} ids from {}",
                ids.len(),
                path.to_string_lossy()
            );
        }
        // CMF_PROMPT_DUMP=1: the rendered prompt as the model sees it
        // (template applied, decoded back to text) — for template audits.
        if std::env::var("CMF_PROMPT_DUMP").is_ok() {
            eprintln!(
                "--- rendered prompt ({} tokens) ---\n{}\n--- end ---",
                ids.len(),
                pipeline.tokenizer.decode(&ids)
            );
            let head: Vec<String> = ids
                .iter()
                .take(12)
                .map(|&t| format!("{t}:{:?}", pipeline.tokenizer.decode(&[t])))
                .collect();
            let tail: Vec<String> = ids
                .iter()
                .rev()
                .take(12)
                .rev()
                .map(|&t| format!("{t}:{:?}", pipeline.tokenizer.decode(&[t])))
                .collect();
            eprintln!("head {}\ntail {}", head.join(" "), tail.join(" "));
        }
        let media_rows = mimo_inputs.as_ref().and_then(|p| p.rows.as_deref())
            .or_else(|| lfm3_inputs.as_ref().and_then(|p| p.rows.as_deref()))
            .or_else(|| ov2_inputs.as_ref().and_then(|p| p.rows.as_deref()))
            .or_else(|| sn_inputs.as_ref().and_then(|p| p.rows.as_deref()));
        // LFM3-Audio speech answer: the assistant turn opens with
        // <|audio_start|>, frames until the end-of-audio frame, the codec.
        if let Some((path, speech_mode)) = speech_out {
            use cortiq_engine::lfm3_speech as sp;
            anyhow::ensure!(sp::can_speak(&model),
                "--speech-out requires an LFM3-Audio file with its depth decoder and codec");
            let marker = if speech_mode == "mixed" { "<|mixed_start|>" } else { "<|audio_start|>" };
            let start = pipeline.tokenizer.token_to_id(marker)
                .ok_or_else(|| anyhow::anyhow!("the tokenizer has no {marker}"))?;
            let mut sids = ids.clone();
            let mut srows = media_rows.map(<[f32]>::to_vec);
            if sids.last() != Some(&start) {
                sids.push(start);
                if let Some(r) = srows.as_mut() {
                    r.extend(pipeline.embed_id(start));
                }
            }
            let pool = cortiq_engine::pool::Pool::from_env();
            let depth = sp::DepthDecoder::from_model(&model, pool.clone()).map_err(anyhow::Error::msg)?;
            let mut sampling = sp::AudioSampling::from_model(&model);
            if greedy {
                sampling.temperature = 0.0;
            }
            sampling.fixed_order = std::env::var("CMF_LFM3_MASKGIT_FIXED").as_deref() == Ok("1");
            let t0 = std::time::Instant::now();
            let out = sp::speak(&mut pipeline, &depth, &sids, srows.as_deref(), max_tokens, &sampling,
                seed.unwrap_or_else(|| std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)))
                .map_err(anyhow::Error::msg)?;
            let t_gen = t0.elapsed().as_secs_f64();
            let t_codec = std::time::Instant::now();
            let codec = sp::Codec::from_model(&model, pool).map_err(anyhow::Error::msg)?;
            let wave = codec.decode(&out.frames);
            if std::env::var_os("CMF_LFM3_SPEECH_PROF").is_some() {
                eprintln!("speech: codec (load + decode) {:.0} ms", t_codec.elapsed().as_secs_f64() * 1e3);
            }
            sp::write_wav(path, &wave, 24_000)?;
            if let Ok(dump) = std::env::var("CMF_LFM3_CODES_DUMP") {
                let flat: Vec<u32> = out.frames.iter().flatten().copied().collect();
                std::fs::write(&dump, serde_json::to_vec(&flat)?)?;
            }
            let text = pipeline.tokenizer.decode(&out.text_ids);
            if !text.trim().is_empty() {
                println!("{}", text.trim());
            }
            eprintln!("[{} frames · {:.2} s of speech · generated in {:.1} s · {}]",
                out.frames.len(), wave.len() as f64 / 24_000.0, t_gen, path.display());
            return Ok(());
        }
        generate_and_print(&mut pipeline, &ids, vl_inputs.as_ref(), media_rows)?;
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
            // A pinned lookup record: the turn is looked up first. `answer`
            // — the table's text is the reply, nothing is generated;
            // `context` — the backbone gets the card prepended; no key —
            // the plain turn runs.
            let mut turn_text = text.to_string();
            if let Some(t) = &lookup_pinned {
                match t.answer(text).map_err(anyhow::Error::msg)? {
                    Some(a) if lookup_mode == LookupMode::Answer => {
                        eprintln!("{}", a.describe());
                        println!("{}\n", a.text);
                        history.push(("user".to_string(), text.to_string()));
                        history.push(("assistant".to_string(), a.text));
                        continue;
                    }
                    Some(a) if lookup_mode == LookupMode::Context => {
                        eprintln!("{}", a.describe());
                        turn_text = cortiq_engine::lookup::context_prompt(&a.card, text);
                    }
                    _ => {}
                }
            }
            let text = turn_text.as_str();
            history.push(("user".to_string(), text.to_string()));
            let mut vl_inputs = is_dsv41
                .then(|| build_dsv41_inputs(&pipeline, &history))
                .transpose()?;
            let mut ids = vl_inputs
                .as_ref()
                .map(|inputs| inputs.token_ids.clone())
                .unwrap_or_else(|| build_ids(&pipeline, &history, text));
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
                vl_inputs = is_dsv41
                    .then(|| build_dsv41_inputs(&pipeline, &history))
                    .transpose()?;
                ids = vl_inputs
                    .as_ref()
                    .map(|inputs| inputs.token_ids.clone())
                    .unwrap_or_else(|| build_ids(&pipeline, &history, text));
            }
            // The terminal already echoed the user's line after "> ".
            if use_template {
                println!();
            }
            match generate_and_print(&mut pipeline, &ids, vl_inputs.as_ref(), None)? {
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

/// A local worker process holding the tail layers on the SECOND GPU.
/// Killed on drop — a run must never leave a model-sized process behind.
struct LocalGpuWorker {
    child: std::process::Child,
    addr: String,
    token: String,
}

impl Drop for LocalGpuWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// `--gpus N`: spawn `cortiq worker <model>` on 127.0.0.1 pinned to
/// adapter 1 (CMF_GPU_ADAPTER), pin THIS process to adapter 0, and let
/// the ordinary --peer machinery do the split. v1 is exactly two cards:
/// the coordinator speaks to ONE worker; chaining is a recorded next
/// step, and pretending otherwise would silently run N-2 cards idle.
fn spawn_local_gpu_worker(model: &str, n: usize) -> anyhow::Result<LocalGpuWorker> {
    if n != 2 {
        anyhow::bail!(
            "--gpus {n}: exactly 2 GPUs are supported for now (coordinator + local worker);              for N cards chain `cortiq worker` + --peer"
        );
    }
    // The coordinator takes card 0 unless already pinned — with two
    // identical cards `request_adapter` gives BOTH processes the same
    // "best" one. The worker then takes a DIFFERENT index: inheriting
    // the coordinator's pin verbatim put both processes on one card
    // (a 110 → 1-3 tok/s faceplant that looks like a slow wire).
    let coord_pin = std::env::var("CMF_GPU_ADAPTER").unwrap_or_else(|_| "0".into());
    if std::env::var("CMF_GPU_ADAPTER").is_err() {
        unsafe { std::env::set_var("CMF_GPU_ADAPTER", "0") };
    }
    let worker_pin = if coord_pin.trim() == "1" { "0" } else { "1" };
    // Let the OS pick a free port; the tiny bind→spawn race is local.
    let port = std::net::TcpListener::bind("127.0.0.1:0")?
        .local_addr()?
        .port();
    let addr = format!("127.0.0.1:{port}");
    let token = format!("local-gpus-{}", std::process::id());
    let child = std::process::Command::new(std::env::current_exe()?)
        .args(["worker", model, "--listen", &addr, "--token", &token])
        .env("CMF_GPU_ADAPTER", worker_pin)
        .stdout(std::process::Stdio::null())
        .spawn()
        .map_err(|e| anyhow::anyhow!("could not start the local worker: {e}"))?;
    let mut w = LocalGpuWorker { child, addr, token };
    eprintln!(
        "gpus: coordinator adapter {coord_pin}, worker adapter {worker_pin} — {}",
        w.addr
    );
    // Readiness = the listener answers. The worker binds before serving,
    // so this bounds model-load time, not just process start.
    for _ in 0..1800 {
        if let Some(st) = w.child.try_wait()? {
            anyhow::bail!("the local worker died at startup ({st}); its stderr is above");
        }
        if std::net::TcpStream::connect(&w.addr).is_ok() {
            return Ok(w);
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    anyhow::bail!("the local worker did not come up within 180 s")
}

#[derive(Clone, Copy, Default)]
struct GpuCounterSnapshot {
    metal_submits: u64,
    wgpu_submits: u64,
    wgpu_passes: u64,
    wgpu_upload_ns: u64,
    wgpu_upload_bytes: u64,
    /// Weight bytes dispatched so far (both backends) — sampled per token
    /// so the steady-window delta is a real bytes-per-token, not the
    /// process total (warmup + prefill + pair micro-bench) divided by
    /// the decode count. That quotient read 21.9 GB/token on a 15.4 GB
    /// Qwen3.8 file over wgpu — a 1.5x "amplification" that was the
    /// bench's own arithmetic.
    weight_bytes: u64,
}

/// Native-Metal execution receipts sampled around one `bench` generation.
///
/// The generic `GRAPH_TOK_*` counters describe the wgpu graph route.  Metal's
/// TokenGraph and the oversized-MoE selected-view fallback have separate,
/// completion-only counters; exposing their per-run deltas prevents a JSON
/// benchmark from claiming a pager result after a silent CPU fallback.
#[derive(Clone, Copy, Default)]
struct NativeMetalBenchCounters {
    graph_tokens: u64,
    graph_layers: u64,
    graph_errors: u64,
    sparse_moe_blocks: u64,
    sparse_moe_span_bytes: u64,
}

impl NativeMetalBenchCounters {
    fn since(self, before: Self) -> Self {
        Self {
            graph_tokens: self.graph_tokens.saturating_sub(before.graph_tokens),
            graph_layers: self.graph_layers.saturating_sub(before.graph_layers),
            graph_errors: self.graph_errors.saturating_sub(before.graph_errors),
            sparse_moe_blocks: self.sparse_moe_blocks.saturating_sub(before.sparse_moe_blocks),
            sparse_moe_span_bytes: self
                .sparse_moe_span_bytes
                .saturating_sub(before.sparse_moe_span_bytes),
        }
    }
}

/// Native Metal is target-specific, but retain zero-valued fields in portable
/// benchmark JSON so result-processing scripts do not need a platform branch.
fn native_metal_bench_counters() -> NativeMetalBenchCounters {
    let mut s = NativeMetalBenchCounters::default();
    #[cfg(target_os = "macos")]
    {
        use std::sync::atomic::Ordering;
        s.graph_tokens = cortiq_engine::pipeline::METAL_GRAPH_TOK_OK.load(Ordering::Relaxed);
        s.graph_layers = cortiq_engine::pipeline::METAL_GRAPH_LAYERS.load(Ordering::Relaxed);
        s.graph_errors = cortiq_engine::pipeline::METAL_GRAPH_ERRORS.load(Ordering::Relaxed);
        s.sparse_moe_blocks = cortiq_engine::gpu_metal::SPARSE_MOE_BLOCKS.load(Ordering::Relaxed);
        s.sparse_moe_span_bytes =
            cortiq_engine::gpu_metal::SPARSE_MOE_SPAN_BYTES.load(Ordering::Relaxed);
    }
    s
}

/// One coherent counter sample. Bench rates are computed from two samples in
/// the same inter-token window as steady tok/s; process-global totals include
/// warmup, prefill and the pair-fusion microbenchmark and are not per-token.
fn gpu_counter_snapshot() -> GpuCounterSnapshot {
    let mut s = GpuCounterSnapshot::default();
    #[cfg(all(target_os = "macos", feature = "gpu"))]
    {
        s.metal_submits = cortiq_engine::gpu_metal::METAL_SUBMITS
            .load(std::sync::atomic::Ordering::Relaxed) as u64;
    }
    #[cfg(feature = "gpu")]
    {
        use std::sync::atomic::Ordering;
        s.wgpu_submits = cortiq_engine::gpu_wgpu::SUBMITS.load(Ordering::Relaxed);
        s.wgpu_passes = cortiq_engine::gpu_wgpu::PASSES.load(Ordering::Relaxed);
        s.wgpu_upload_ns = cortiq_engine::gpu_wgpu::UPLOAD_NS.load(Ordering::Relaxed);
        s.wgpu_upload_bytes = cortiq_engine::gpu_wgpu::UPLOAD_BYTES.load(Ordering::Relaxed);
    }
    s.weight_bytes = cortiq_engine::gpu::weight_bytes_dispatched();
    s
}

/// Milliseconds spent pushing weights to the device, and the megabytes
/// that rode. A resident model uploads ONCE for the whole run; more than
/// the model's size means the cache is missing and the device is being
/// fed over the upload path, ~40 MB/s on this class of hardware.
fn wgpu_uploads() -> (f64, f64) {
    #[cfg(feature = "gpu")]
    {
        use std::sync::atomic::Ordering;
        (
            cortiq_engine::gpu_wgpu::UPLOAD_NS.load(Ordering::Relaxed) as f64 / 1e6,
            cortiq_engine::gpu_wgpu::UPLOAD_BYTES.load(Ordering::Relaxed) as f64 / 1e6,
        )
    }
    #[cfg(not(feature = "gpu"))]
    {
        (0.0, 0.0)
    }
}

fn cmd_dequant(
    model_path: &str,
    name: &str,
    out: &str,
    dtype: &str,
    all: bool,
) -> anyhow::Result<()> {
    let model = CmfModel::open_sharded(model_path)?;
    let names: Vec<String> = if all {
        model
            .tensors
            .iter()
            .filter(|e| e.name.starts_with(name))
            .map(|e| e.name.clone())
            .collect()
    } else {
        vec![name.to_string()]
    };
    if all {
        std::fs::create_dir_all(out)?;
    }
    for n in &names {
        let e = model
            .tensor(n)
            .ok_or_else(|| anyhow::anyhow!("no tensor '{n}'"))?;
        let bytes = model.tensor_bytes(n)?;
        let buf = dequant_dump_bytes(e, bytes, dtype)?;
        let path = if all {
            format!("{out}/{}.bin", n)
        } else {
            out.to_string()
        };
        std::fs::write(&path, &buf)?;
        if all {
            eprintln!("{n}\t{:?}\t{:?}", e.dtype, e.shape);
        }
    }
    if !all {
        let e = model.tensor(name).unwrap();
        println!("{} {:?} {:?} → {out}", name, e.dtype, e.shape);
    }
    Ok(())
}

/// The bytes `cortiq dequant --dtype DTYPE` writes for one tensor: the
/// decoded values as little-endian f32 or bf16 (round-to-nearest-even),
/// or with `raw` the entry's encoded payload verbatim (no decode at all).
fn dequant_dump_bytes<'a>(
    e: &cortiq_core::format::TensorEntry,
    bytes: &'a [u8],
    dtype: &str,
) -> anyhow::Result<std::borrow::Cow<'a, [u8]>> {
    if dtype == "raw" {
        return Ok(std::borrow::Cow::Borrowed(bytes));
    }
    if dtype != "f32" && dtype != "bf16" {
        anyhow::bail!("dtype {dtype}: f32, bf16 or raw");
    }
    let mut vals = vec![0f32; e.n_elems()];
    cortiq_core::quant::dequant_tensor(e, bytes, &mut vals)
        .map_err(|er| anyhow::anyhow!("dequant {}: {er}", e.name))?;
    let mut buf: Vec<u8> = Vec::with_capacity(vals.len() * 4);
    if dtype == "f32" {
        for v in &vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    } else {
        for v in &vals {
            // round-to-nearest-even to bf16
            let b = v.to_bits();
            let lsb = (b >> 16) & 1;
            let r = b.wrapping_add(0x7FFF + lsb) >> 16;
            buf.extend_from_slice(&(r as u16).to_le_bytes());
        }
    }
    Ok(std::borrow::Cow::Owned(buf))
}

/// Dtype of an already ENCODED payload for `patch-tensor --raw`, checked
/// against the entry's shape: the entry's own dtype, or — with `--dtype` —
/// the dtype the encoder emits for this shape (q4tp/q2tp fall back to
/// q8_2f when the input dim is not a multiple of the group, exactly as
/// `quantize_2d` does). The length check is `validate_payload`'s, i.e. the
/// reader's own rule, so a payload that passes here also passes `open()`.
/// The dtype a patched payload is stored in: a re-encodable quantization, or a
/// verbatim container dtype (the BLR code — its payload is produced by the
/// NVG encoder, not by `quantize_2d`).
enum PatchDtype {
    Quant(convert::Quant),
    Verbatim(TensorDtype),
}

fn raw_patch_dtype(
    e: &cortiq_core::format::TensorEntry,
    forced: Option<&PatchDtype>,
    payload: &[u8],
) -> anyhow::Result<TensorDtype> {
    let dt = match forced {
        None => e.dtype,
        Some(PatchDtype::Verbatim(dt)) => *dt,
        Some(PatchDtype::Quant(convert::Quant::F16)) => TensorDtype::F16,
        Some(PatchDtype::Quant(q)) => {
            anyhow::ensure!(
                e.shape.len() == 2,
                "{}: raw {q:?} wants a 2-D tensor (shape {:?})",
                e.name,
                e.shape
            );
            // A one-row probe picks the dtype without encoding the tensor.
            let cols = e.shape[1];
            convert::quantize_2d(*q, &vec![0f32; cols], 1, cols).0
        }
    };
    cortiq_core::quant::validate_payload(dt, &e.shape, payload)
        .map_err(|er| anyhow::anyhow!("{}: raw {dt:?} payload: {er}", e.name))?;
    Ok(dt)
}

fn cmd_patch_tensor(
    model_path: &str,
    sets: &[String],
    dtype: Option<&str>,
    output: Option<&str>,
    raw_payload: bool,
) -> anyhow::Result<()> {
    if sets.is_empty() {
        anyhow::bail!("nothing to patch: pass --set name=path.f32 (repeatable)");
    }
    let quant_of = |dt: TensorDtype| -> anyhow::Result<convert::Quant> {
        Ok(match dt {
            TensorDtype::Q4TiledP => convert::Quant::Q4TiledP,
            TensorDtype::Q2TiledP => convert::Quant::Q2TiledP,
            TensorDtype::Q4Tiled => convert::Quant::Q4Tiled,
            TensorDtype::Q8_2f => convert::Quant::Q8_2f,
            TensorDtype::Q8Row => convert::Quant::Q8Row,
            TensorDtype::F16 => convert::Quant::F16,
            other => anyhow::bail!("cannot re-encode dtype {other:?} here"),
        })
    };
    let forced: Option<PatchDtype> = match dtype {
        None => None,
        // The BLR code is produced by the NVG encoder and stored verbatim —
        // it has no `Quant` counterpart (the converter runs its own encoder).
        Some(q) if q.eq_ignore_ascii_case("blr") => {
            Some(PatchDtype::Verbatim(TensorDtype::BinaryLowRank))
        }
        Some(q) => Some(PatchDtype::Quant(convert::parse_quant(q)?)),
    };
    // (directory index, dtype, payload)
    let mut patches: Vec<(usize, TensorDtype, Vec<u8>)> = Vec::new();
    let model = CmfModel::open_sharded(model_path)?;
    knowledge::refuse_genome_rewrite(&model, "patch-tensor")?;
    for spec in sets {
        let (name, path) = spec
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--set wants name=path, got '{spec}'"))?;
        let idx = model
            .tensor_index(name)
            .ok_or_else(|| anyhow::anyhow!("no tensor '{name}'"))?;
        let e = &model.tensors[idx];
        let n = e.n_elems();
        let raw = std::fs::read(path)?;
        if raw_payload {
            // Verbatim payload: no re-encode, only the reader's own
            // length rule for the target dtype and the in-place slot.
            let dt = raw_patch_dtype(e, forced.as_ref(), &raw).map_err(|er| anyhow::anyhow!("{path}: {er}"))?;
            if output.is_none() {
                anyhow::ensure!(
                    e.shard == 0,
                    "{name} lives in a sibling shard; in-place needs shard 0"
                );
                anyhow::ensure!(
                    raw.len() as u64 <= e.nbytes,
                    "{name}: payload {} bytes > slot {} — use --output",
                    raw.len(),
                    e.nbytes
                );
            }
            eprintln!("{name}: {:?} → {dt:?} raw ({} bytes)", e.dtype, raw.len());
            patches.push((idx, dt, raw));
            continue;
        }
        if raw.len() != n * 4 {
            anyhow::bail!(
                "{path}: {} bytes, want {} (= {:?} f32)",
                raw.len(),
                n * 4,
                e.shape
            );
        }
        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let quant = match forced {
            Some(PatchDtype::Quant(q)) => q,
            // A verbatim dtype has no float source: it only exists as an
            // already-encoded payload (--raw).
            Some(PatchDtype::Verbatim(dt)) => anyhow::bail!(
                "--dtype {dt:?} stores an already-encoded payload: pass it with --raw"
            ),
            None => quant_of(e.dtype)?,
        };
        // Float targets take any shape (norms, biases, conv taps); the
        // quantized layouts are 2-D by definition.
        let (dt, data) = match quant {
            convert::Quant::F16 => (TensorDtype::F16, convert::encode_f16(&vals)),
            _ => {
                if e.shape.len() != 2 {
                    anyhow::bail!(
                        "{name}: {quant:?} wants a 2-D tensor (shape {:?}); use --dtype f16",
                        e.shape
                    );
                }
                convert::quantize_2d(quant, &vals, e.shape[0], e.shape[1])
            }
        };
        let (rows, cols) = (e.shape[0], e.shape.get(1).copied().unwrap_or(1));
        if output.is_none() {
            if e.shard != 0 {
                anyhow::bail!("{name} lives in a sibling shard; in-place needs shard 0");
            }
            if data.len() as u64 > e.nbytes {
                anyhow::bail!(
                    "{name}: payload {} bytes > slot {} — use --output",
                    data.len(),
                    e.nbytes
                );
            }
        }
        eprintln!(
            "{name}: {rows}×{cols} {:?} → {dt:?} ({} bytes)",
            e.dtype,
            data.len()
        );
        patches.push((idx, dt, data));
    }
    match output {
        None => {
            drop(model);
            CmfModel::recode_entries_in_place(model_path, &patches)?;
            println!("patched {} tensors in place in {model_path}", patches.len());
        }
        Some(out) => {
            let mut specs: Vec<cortiq_core::format::TensorSpec> =
                Vec::with_capacity(model.tensors.len());
            for (i, e) in model.tensors.iter().enumerate() {
                let (dtype, data) = match patches.iter().find(|p| p.0 == i) {
                    Some((_, dt, d)) => (*dt, d.clone()),
                    None => (e.dtype, model.entry_bytes(e).to_vec()),
                };
                specs.push(cortiq_core::format::TensorSpec {
                    name: e.name.clone(),
                    dtype,
                    shape: e.shape.clone(),
                    data,
                });
            }
            CmfModel::write(
                out,
                &model.header,
                &specs,
                Some(&model.masks),
                model.vocab.as_deref(),
            )?;
            println!("wrote {out} with {} patched tensors", patches.len());
        }
    }
    Ok(())
}

/// The `--exclude-tensors` patterns that left a declared MTP stack out of
/// this file: the header keeps `arch.mtp` (the source's depth), so `info`
/// asks whether the file holds `model.mtp.*` tensors and whether the
/// recorded exclusion covers them. None when the file carries the stack or
/// nothing in its provenance excluded it.
fn mtp_excluded_by(model: &CmfModel) -> Option<String> {
    if model.tensors.iter().any(|t| t.name.starts_with("model.mtp.")) {
        return None;
    }
    let pats: Vec<&str> = model
        .header
        .provenance
        .as_ref()?
        .get("excluded_tensors")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_str())
        .filter(|p| crate::convert::glob_match(p, "model.mtp.0.norm.weight"))
        .collect();
    (!pats.is_empty()).then(|| format!("'{}'", pats.join(",")))
}

async fn cmd_info(model_path: &str, tensors: Option<&str>) -> anyhow::Result<()> {
    let model = CmfModel::open_sharded(model_path)?;
    let arch = model.arch();

    if let Some(prefix) = tensors {
        let mut total = 0u64;
        let mut n = 0usize;
        for e in model.tensors.iter().filter(|e| e.name.starts_with(prefix)) {
            println!("{}\t{:?}\t{:?}\t{}\t{:016x}", e.name, e.dtype, e.shape, e.nbytes, e.hash);
            total += e.nbytes;
            n += 1;
        }
        println!(
            "# {n} tensors, {total} bytes ({:.2} GiB)",
            total as f64 / (1 << 30) as f64
        );
        return Ok(());
    }

    let count = |k: cortiq_core::LayerType| arch.layer_types.iter().filter(|&&t| t == k).count();
    let full = count(cortiq_core::LayerType::FullAttention);
    let sliding = count(cortiq_core::LayerType::SlidingAttention);
    let conv = count(cortiq_core::LayerType::ShortConv);
    let bounded = count(cortiq_core::LayerType::BoundedAttention);
    let linear = arch.num_layers - full - sliding - conv - bounded;
    println!("Model: {}", model_path);
    println!("  Format:      CMF v{}", model.header.version);
    println!("  Arch:        {}", arch.arch_name);
    let mut mix = format!("{full} full");
    if sliding > 0 {
        mix.push_str(&format!(" / {sliding} sliding"));
    }
    if conv > 0 {
        mix.push_str(&format!(" / {conv} conv"));
    }
    if bounded > 0 {
        mix.push_str(&format!(" / {bounded} bounded"));
    }
    if linear > 0 {
        mix.push_str(&format!(" / {linear} linear"));
    }
    println!("  Layers:      {} ({mix})", arch.num_layers);
    println!("  Hidden:      {}", arch.hidden_size);
    println!("  FFN:         {}", arch.intermediate_size);
    println!(
        "  Heads:       {} (KV: {})",
        arch.num_attention_heads, arch.num_kv_heads
    );
    println!("  Vocab:       {}", arch.vocab_size);
    println!(
        "  Quant:       {:?} (default; per-tensor in directory)",
        model.header.quant_type
    );
    // The head's own dtype, which the default rarely is and which decides
    // whether it has a GPU route at all — the largest single matvec of a
    // decode step should not be something one has to guess at.
    if let Some(e) = model
        .tensors
        .iter()
        .find(|e| e.name == "lm_head.weight")
        .or_else(|| {
            model
                .tensors
                .iter()
                .find(|e| e.name == "model.embed_tokens.weight")
        })
    {
        println!("  Head:        {:?} {:?}", e.dtype, e.shape);
    }
    println!("  Tensors:     {}", model.tensors.len());
    println!(
        "  Params:      {:.2}B",
        model.total_param_count() as f64 / 1e9
    );
    println!("  Masks:       {}", model.masks.masks.len());
    println!(
        "  Tokenizer:   {}",
        if model.vocab.is_some() {
            "embedded"
        } else {
            "sidecar required"
        }
    );
    if let Some(s) = model
        .header
        .provenance
        .as_ref()
        .and_then(|p| p.get("sampling"))
    {
        println!("  Sampling:    {s} (file defaults for run / serve)");
    }
    println!(
        "  MTP:         {}",
        match (&arch.mtp, mtp_excluded_by(&model)) {
            (Some(m), Some(pats)) => format!(
                "{} block(s) declared by the source, excluded from this file \
                 (--exclude-tensors {pats})",
                m.num_layers
            ),
            (Some(m), None) => format!("{} block(s), shared embed+lm_head", m.num_layers),
            (None, _) => "—".to_string(),
        }
    );
    if let Some(bd) = &arch.block_diffusion {
        println!(
            "  Diffusion:   {} over {}-token blocks · threshold {} · mask id {} · expert coreset {}",
            bd.algorithm,
            bd.block_size,
            bd.threshold,
            bd.mask_token_id,
            if bd.coreset_enabled {
                format!("top-p {}", bd.coreset_top_p)
            } else {
                "off".to_string()
            }
        );
        println!(
            "               response penalties: presence {} · frequency {} · repetition {} · \
             stop ramp {} from {} over {} · </think> ramp {} from {} over {}",
            bd.presence_penalty,
            bd.frequency_penalty,
            bd.repetition_penalty,
            bd.length_penalty,
            bd.length_penalty_start,
            bd.length_penalty_window,
            bd.think_end_bias,
            bd.think_end_start,
            bd.think_end_window
        );
    }
    println!("  Sparse idx:  {} entries", model.sparse_index.len());
    // The bounded anchor is an operator record, not a hint: print it as
    // the file states it, and the per-sequence state it fixes.
    if let Some(ac) = &arch.anchor_core {
        println!(
            "  Anchor:      {} window={} sink={} rope={} sink_scores={} train_windows={:?} far={}",
            ac.kind,
            ac.window,
            ac.sink,
            ac.rope,
            ac.sink_scores,
            ac.train_windows,
            ac.far
                .as_ref()
                .map(|f| format!("{{m={}, rect={}}}", f.m, f.rect))
                .unwrap_or_else(|| "null".into())
        );
        println!(
            "               ring per layer: {} B ({} kv_heads x {} x {} f32, K+V)",
            ac.ring_state_elems(arch.num_kv_heads, arch.head_dim) * 4,
            arch.num_kv_heads,
            ac.window,
            arch.head_dim
        );
    }
    match cortiq_engine::loader::per_sequence_state_bytes(&model) {
        Ok(st) => {
            let fixed = st.bounded_bytes + st.recurrent_bytes;
            if st.growing_layers == 0 {
                println!(
                    "  State/seq:   {fixed} B fixed ({} B bounded rings + {} B recurrent) — O(1) in context",
                    st.bounded_bytes, st.recurrent_bytes
                );
            } else {
                println!(
                    "  State/seq:   {fixed} B fixed ({} B bounded rings + {} B recurrent) + {} layer(s) of per-position KV",
                    st.bounded_bytes, st.recurrent_bytes, st.growing_layers
                );
            }
        }
        Err(e) => println!("  State/seq:   ? ({e})"),
    }

    Ok(())
}

fn qwen_component_paths(
    root: &std::path::Path,
    qwen_bundle: bool,
    text_encoder: Option<&str>,
    vae: Option<&str>,
    scheduler: Option<&str>,
) -> cortiq_engine::qwen_imagegen::QwenImagePaths {
    let parent = if root.is_dir() {
        root
    } else {
        root.parent().unwrap_or_else(|| std::path::Path::new("."))
    };
    cortiq_engine::qwen_imagegen::QwenImagePaths {
        transformer: if root.is_dir() {
            root.join("transformer.cmf")
        } else {
            root.to_path_buf()
        },
        text_encoder: text_encoder
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                if qwen_bundle {
                    root.to_path_buf()
                } else {
                    parent.join("text_encoder.cmf")
                }
            }),
        vae: vae.map(std::path::PathBuf::from).unwrap_or_else(|| {
            if qwen_bundle {
                root.to_path_buf()
            } else {
                parent.join("vae.cmf")
            }
        }),
        scheduler: scheduler.map(std::path::PathBuf::from).or_else(|| {
            if qwen_bundle {
                return None;
            }
            let path = parent.join("scheduler_config.json");
            path.is_file().then_some(path)
        }),
    }
}

/// `cortiq imagine` options that apply to a Qwen-Image-2.1 container.
struct Qi21Cli {
    height: Option<usize>,
    width: Option<usize>,
    steps: Option<usize>,
    cfg: Option<f32>,
    num_images: usize,
    seed: u64,
    images: Vec<String>,
    negative_prompt: Option<String>,
    reference_size: usize,
    out: Option<String>,
}

/// Qwen-Image-2.1: text-to-image (RGBA) or generation from condition images
/// (`--image`, repeatable); unset options come from the container.
fn cmd_qi21(model: &str, prompt: &str, o: Qi21Cli) -> anyhow::Result<()> {
    use cortiq_engine::qwen_image21gen::{generate_images, Qi21Defaults, Qi21Params};
    let d = Qi21Defaults::of(&CmfModel::open(model)?);
    let mut p = Qi21Params::from_defaults(&d);
    p.height = o.height;
    p.width = o.width;
    p.steps = o.steps.unwrap_or(d.steps);
    p.cfg = o.cfg.unwrap_or(d.cfg);
    p.seed = o.seed;
    p.num_images = o.num_images.max(1);
    p.images = o.images.iter().map(std::path::PathBuf::from).collect();
    p.output_resolution = o.reference_size;
    p.negative_prompt = o.negative_prompt;
    anyhow::ensure!(p.steps >= 2, "--steps must be at least 2");
    if p.cfg > 1.0 && p.negative_prompt.is_none() {
        eprintln!("note: --cfg > 1 needs --negative-prompt (Qwen-Image-2.1 samples without guidance by default)");
    }
    let out = o.out.unwrap_or_else(|| "out.png".into());
    let path = std::path::Path::new(&out);
    let names: Vec<std::path::PathBuf> = if p.num_images == 1 {
        vec![path.to_path_buf()]
    } else {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("png");
        (0..p.num_images)
            .map(|i| path.with_file_name(format!("{stem}_{i}.{ext}")))
            .collect()
    };
    eprintln!(
        "qwen-image-2.1: {} steps, cfg {}, seed {}, {} condition image(s), {} image(s)",
        p.steps,
        p.cfg,
        p.seed,
        p.images.len(),
        p.num_images
    );
    let t0 = std::time::Instant::now();
    let (imgs, tm) = generate_images(std::path::Path::new(model), prompt, &p, |img, i, n| {
        eprintln!(
            "image {}/{}: step {i}/{n} ({:.1}s)",
            img + 1,
            p.num_images,
            t0.elapsed().as_secs_f64()
        );
    })
    .map_err(anyhow::Error::msg)?;
    for (img, name) in imgs.iter().zip(&names) {
        img.save(name).map_err(anyhow::Error::msg)?;
        println!(
            "{}: {}x{}, seed {}{}",
            name.display(),
            img.width,
            img.height,
            img.seed,
            if img.has_alpha() { ", with transparency" } else { "" }
        );
    }
    println!(
        "{} image(s), {} steps in {:.1}s (text {:.1}s, dit load {:.1}s, prefill {:.1}s, median step {:.2}s, vae {:.1}s)",
        imgs.len(),
        p.steps,
        tm.total,
        tm.text_encode,
        tm.dit_load,
        tm.prefill,
        tm.median_step(),
        tm.vae
    );
    Ok(())
}

/// Parse zero-based MOSS transformer block indices without changing the
/// source's set-like semantics. `none`, `off` and an empty value deliberately
/// mean no skips, so a bare `--uncond-skip-layers` can disable the packaged
/// source default.
fn parse_moss_layer_list(raw: &str, flag: &str) -> anyhow::Result<Vec<usize>> {
    let raw = raw.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("none") || raw.eq_ignore_ascii_case("off") {
        return Ok(Vec::new());
    }
    let mut layers = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        anyhow::ensure!(!part.is_empty(), "{flag} needs comma-separated nonnegative block indices");
        let index = part.parse::<usize>().map_err(|_| {
            anyhow::anyhow!("{flag} has invalid block index {part:?}; expected comma-separated nonnegative integers")
        })?;
        // The official Python path turns the list into a frozenset. Removing
        // duplicates here has the same result and avoids needless contains
        // checks in the hot forward loop.
        if !layers.contains(&index) {
            layers.push(index);
        }
    }
    Ok(layers)
}

/// Keep a MOSS layout prompt byte-for-byte as supplied. The official pipeline
/// treats it as a normal string prompt, so validation catches only malformed
/// JSON / the wrong root value and never serializes it into a different form.
fn validate_moss_layout(raw: &str, source: &str) -> anyhow::Result<String> {
    let value: serde_json::Value = serde_json::from_str(raw)
        .map_err(|e| anyhow::anyhow!("{source} is not valid JSON: {e}"))?;
    anyhow::ensure!(value.is_object(), "{source} must contain a JSON object");
    Ok(raw.to_owned())
}

/// Select MOSS's single source prompt. `--layout` and `--layout-file` are a
/// convenience for the upstream JSON layout prompting mode; their content is
/// passed unchanged through the stored source chat template.
fn resolve_moss_prompt(
    prompt: Option<String>,
    layout: Option<String>,
    layout_file: Option<String>,
) -> anyhow::Result<String> {
    match (prompt, layout, layout_file) {
        (Some(prompt), None, None) => Ok(prompt),
        (None, Some(layout), None) => validate_moss_layout(&layout, "--layout"),
        (None, None, Some(path)) => {
            let layout = std::fs::read_to_string(&path)
                .map_err(|e| anyhow::anyhow!("cannot read --layout-file {path:?}: {e}"))?;
            validate_moss_layout(&layout, "--layout-file")
        }
        _ => anyhow::bail!("provide exactly one of --prompt, --layout or --layout-file for MOSS-Image"),
    }
}

/// `cortiq imagine` options that apply to a MOSS-Image container.
struct MossImageCli {
    height: Option<usize>,
    width: Option<usize>,
    steps: Option<usize>,
    cfg: Option<f32>,
    shift: Option<f32>,
    uncond_skip_layers: Option<String>,
    slg_layers: Option<String>,
    slg_scale: Option<f32>,
    slg_start: Option<f32>,
    slg_stop: Option<f32>,
    max_sequence_length: Option<usize>,
    num_images: usize,
    seed: u64,
    negative_prompt: Option<String>,
    out: Option<String>,
}

/// MOSS-Image: Qwen3.5 encoder → MOSS DiT → Flux2 decoder. Defaults come
/// from the CMF recipe and retain the official negative-pass skip at layer 22.
fn cmd_moss_image(model: &str, prompt: &str, o: MossImageCli) -> anyhow::Result<()> {
    use cortiq_engine::moss_imagegen::{generate_images, MossDefaults, MossParams};
    let d = MossDefaults::of(&CmfModel::open(model)?);
    let mut p = MossParams::from_defaults(&d);
    p.height = o.height.unwrap_or(d.height);
    p.width = o.width.unwrap_or(d.width);
    p.steps = o.steps.unwrap_or(d.steps);
    p.cfg = o.cfg.unwrap_or(d.cfg);
    p.shift = o.shift.unwrap_or(d.shift);
    p.max_tokens = o.max_sequence_length.unwrap_or(d.max_sequence_length);
    p.num_images = o.num_images;
    p.seed = o.seed;
    p.negative_prompt = o.negative_prompt;
    if let Some(layers) = o.uncond_skip_layers.as_deref() {
        p.uncond_skip_layers = Some(parse_moss_layer_list(layers, "--uncond-skip-layers")?);
    }
    if let Some(layers) = o.slg_layers.as_deref() {
        p.slg_layers = parse_moss_layer_list(layers, "--slg-layers")?;
    }
    p.slg_scale = o.slg_scale.unwrap_or(p.slg_scale);
    p.slg_start = o.slg_start.unwrap_or(p.slg_start);
    p.slg_stop = o.slg_stop.unwrap_or(p.slg_stop);
    anyhow::ensure!(
        p.height > 0 && p.width > 0 && p.height % 16 == 0 && p.width % 16 == 0,
        "--height/--width must be positive multiples of 16 (got {}x{})",
        p.width,
        p.height
    );
    anyhow::ensure!(
        p.steps > 0 && p.cfg.is_finite() && p.cfg >= 0.0,
        "--steps must be positive and --cfg must be finite and >= 0"
    );
    anyhow::ensure!(
        p.shift.is_finite() && p.shift > 0.0,
        "--shift must be finite and > 0"
    );
    anyhow::ensure!(
        p.slg_scale.is_finite() && p.slg_scale >= 0.0,
        "--slg-scale must be finite and >= 0"
    );
    anyhow::ensure!(
        p.slg_start.is_finite()
            && p.slg_stop.is_finite()
            && p.slg_start >= 0.0
            && p.slg_start <= p.slg_stop
            && p.slg_stop <= 1.0,
        "require 0 <= --slg-start <= --slg-stop <= 1"
    );
    anyhow::ensure!(p.num_images > 0, "--num-images must be at least 1");
    let out = o.out.unwrap_or_else(|| "out.png".into());
    let path = std::path::Path::new(&out);
    let names: Vec<std::path::PathBuf> = if p.num_images == 1 {
        vec![path.to_path_buf()]
    } else {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("png");
        (0..p.num_images)
            .map(|i| path.with_file_name(format!("{stem}_{i}.{ext}")))
            .collect()
    };
    eprintln!(
        "moss-image: {}x{}, {} requested steps, CFG {}, shift {}, seed {}, {} image(s)",
        p.width, p.height, p.steps, p.cfg, p.shift, p.seed, p.num_images
    );
    let started = std::time::Instant::now();
    let (images, timing) = generate_images(std::path::Path::new(model), prompt, &p, |image, step, total| {
        eprintln!(
            "image {}/{}: step {step}/{total} ({:.1}s)",
            image + 1,
            p.num_images,
            started.elapsed().as_secs_f64()
        );
    })
    .map_err(anyhow::Error::msg)?;
    for (image, name) in images.iter().zip(&names) {
        image.save(name).map_err(anyhow::Error::msg)?;
        println!("{}: {}x{}, seed {}", name.display(), image.width, image.height, image.seed);
    }
    println!(
        "{} MOSS image(s), {} actual steps in {:.1}s (text {:.1}s, DiT load {:.1}s, text prepare {:.1}s, median step {:.2}s, VAE {:.1}s)",
        images.len(),
        timing.steps.len() / images.len().max(1),
        timing.total,
        timing.text_encode,
        timing.dit_load,
        timing.text_prepare,
        timing.median_step(),
        timing.vae
    );
    Ok(())
}

/// `cortiq imagine` options that apply to a Z-Image container.
struct ZimageCli {
    height: Option<usize>,
    width: Option<usize>,
    steps: Option<usize>,
    cfg: Option<f32>,
    cfg_normalization: Option<f32>,
    cfg_truncation: Option<f32>,
    shift: Option<f32>,
    max_sequence_length: Option<usize>,
    num_images: usize,
    seed: u64,
    negative_prompt: Option<String>,
    out: Option<String>,
}

/// Z-Image / Z-Image-Turbo text-to-image: every unset option comes from the
/// recipe stored in the container (`zimage.config_json`).
fn cmd_zimage(model: &str, prompt: &str, o: ZimageCli) -> anyhow::Result<()> {
    use cortiq_engine::zimagegen::{generate_images, ZDefaults, ZParams};
    let d = ZDefaults::of(&CmfModel::open(model)?);
    let mut p = ZParams::from_defaults(&d);
    p.height = o.height.unwrap_or(d.height);
    p.width = o.width.unwrap_or(d.width);
    p.steps = o.steps.unwrap_or(d.steps);
    p.guidance = o.cfg.unwrap_or(d.guidance);
    p.cfg_normalization = o.cfg_normalization.unwrap_or(d.cfg_normalization);
    p.cfg_truncation = o.cfg_truncation.unwrap_or(d.cfg_truncation);
    p.shift = o.shift.unwrap_or(d.shift);
    p.max_tokens = o.max_sequence_length.unwrap_or(d.max_sequence_length);
    p.num_images = o.num_images.max(1);
    p.seed = o.seed;
    p.negative_prompt = o.negative_prompt;
    anyhow::ensure!(
        p.height % 16 == 0 && p.width % 16 == 0 && p.height > 0 && p.width > 0,
        "--height/--width must be positive multiples of 16 (got {}x{})",
        p.width,
        p.height
    );
    anyhow::ensure!(p.guidance >= 0.0, "--cfg must be >= 0");
    if p.guidance == 0.0 && p.negative_prompt.is_some() {
        eprintln!("note: --negative-prompt has no effect at guidance 0 (pass --cfg > 0)");
    }
    if d.variant == "turbo" && p.guidance > 0.0 {
        eprintln!("note: Z-Image-Turbo is distilled for guidance 0; CFG doubles the work");
    }
    let out = o.out.unwrap_or_else(|| "out.png".into());
    let path = std::path::Path::new(&out);
    let names: Vec<std::path::PathBuf> = if p.num_images == 1 {
        vec![path.to_path_buf()]
    } else {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("png");
        (0..p.num_images)
            .map(|i| path.with_file_name(format!("{stem}_{i}.{ext}")))
            .collect()
    };
    eprintln!(
        "z-image ({}): {}x{}, {} steps, guidance {}{}{}, shift {}, seed {}, {} image(s)",
        d.variant,
        p.width,
        p.height,
        p.steps,
        p.guidance,
        if p.guidance > 0.0 && p.cfg_normalization > 0.0 {
            format!(", cfg-normalization {}", p.cfg_normalization)
        } else {
            String::new()
        },
        if p.guidance > 0.0 && p.cfg_truncation < 1.0 {
            format!(", cfg-truncation {}", p.cfg_truncation)
        } else {
            String::new()
        },
        p.shift,
        p.seed,
        p.num_images
    );
    let t0 = std::time::Instant::now();
    let (imgs, tm) = generate_images(std::path::Path::new(model), prompt, &p, |img, i, n| {
        eprintln!(
            "image {}/{}: step {i}/{n} ({:.1}s)",
            img + 1,
            p.num_images,
            t0.elapsed().as_secs_f64()
        );
    })
    .map_err(anyhow::Error::msg)?;
    for (img, name) in imgs.iter().zip(&names) {
        img.save(name).map_err(anyhow::Error::msg)?;
        println!(
            "{}: {}x{}, seed {}",
            name.display(),
            img.width,
            img.height,
            img.seed
        );
    }
    println!(
        "{} image(s), {} steps in {:.1}s (text {:.1}s, dit load {:.1}s, median step {:.2}s, vae {:.1}s)",
        imgs.len(),
        p.steps,
        tm.total,
        tm.text_encode,
        tm.dit_load,
        tm.median_step(),
        tm.vae
    );
    Ok(())
}

/// Dispatch native image generation/editing from its CMF architecture.
#[allow(clippy::too_many_arguments)]
fn cmd_imagine(
    model_dir: &str,
    prompt: &str,
    height: Option<usize>,
    width: Option<usize>,
    steps: Option<usize>,
    cfg: f32,
    seed: u64,
    out: &str,
    images: &[String],
    text_encoder: Option<&str>,
    vae: Option<&str>,
    negative_prompt: Option<&str>,
    scheduler: Option<&str>,
    reference_size: usize,
) -> anyhow::Result<()> {
    let root = std::path::Path::new(model_dir);
    let (qwen_image, qwen_bundle) = if root.is_file() {
        let model = CmfModel::open(root)?;
        let arch = model.header.arch.arch_name.as_str();
        (
            arch == "qwen_image" || arch == "qwen_image_bundle",
            arch == "qwen_image_bundle" || model.tensor("image.bundle_config_json").is_some(),
        )
    } else {
        (root.join("transformer.cmf").is_file(), false)
    };
    if qwen_image || !images.is_empty() || text_encoder.is_some() || vae.is_some() {
        let paths = qwen_component_paths(root, qwen_bundle, text_encoder, vae, scheduler);
        if reference_size != 1024 {
            eprintln!(
                "Qwen Image: reference area {reference_size}² differs from the official 1024² profile"
            );
        }
        let defaults = cortiq_engine::qwen_imagegen::QwenImageParams::default();
        let params = cortiq_engine::qwen_imagegen::QwenImageParams {
            height: height.unwrap_or(defaults.height),
            width: width.unwrap_or(defaults.width),
            steps: steps.unwrap_or(defaults.steps),
            true_cfg_scale: cfg,
            seed,
            reference_size,
            negative_prompt: Some(negative_prompt.unwrap_or(" ").to_string()),
            ..Default::default()
        };
        let refs = images
            .iter()
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>();
        let t0 = std::time::Instant::now();
        let image = cortiq_engine::qwen_imagegen::edit_files(
            &paths,
            prompt,
            &refs,
            &params,
            |stage, i, n| {
                eprintln!("{stage}: {i}/{n} ({:.1}s)", t0.elapsed().as_secs_f64());
            },
        )
        .map_err(anyhow::Error::msg)?;
        image
            .save(std::path::Path::new(out))
            .map_err(anyhow::Error::msg)?;
        println!(
            "{out}: {}x{}, {} steps in {:.1}s",
            image.width,
            image.height,
            params.steps,
            t0.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    if negative_prompt.is_some() || scheduler.is_some() || reference_size != 1024 {
        anyhow::bail!("negative-prompt, scheduler, and reference-size are Qwen Image options");
    }
    let height = height.unwrap_or(512);
    let width = width.unwrap_or(512);
    let steps = steps.unwrap_or(30);
    let params = cortiq_engine::imagegen::GenParams {
        height,
        width,
        steps,
        guidance_scale: cfg,
        seed,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let img = cortiq_engine::imagegen::generate(
        std::path::Path::new(model_dir),
        prompt,
        &params,
        |i, n| {
            eprintln!("step {i}/{n} ({:.1}s)", t0.elapsed().as_secs_f64());
        },
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    // P6 PPM: header + interleaved u8 RGB.
    let mut buf = format!("P6\n{width} {height}\n255\n").into_bytes();
    let plane = height * width;
    for p in 0..plane {
        for ch in 0..3 {
            buf.push((img[ch * plane + p] * 255.0 + 0.5) as u8);
        }
    }
    std::fs::write(out, &buf)?;
    println!(
        "{out}: {width}x{height}, {steps} steps in {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn cmd_animate(
    model: &str,
    prompt: &str,
    params: cortiq_engine::videogen::AnimParams,
    quality: u32,
    out: &str,
    frames_dir: Option<&str>,
) -> anyhow::Result<()> {
    // A seeded render must come out the same twice. It did not: while a
    // probe class is undecided the arbitration ALTERNATES arms on real
    // user data, and whether it ever decides depends on how many samples
    // the cold-discard throws away — a race. Three runs of one binary,
    // one seed, gave two different files. Pinning the arms costs nothing
    // measurable here (57.4 s and 57.8 s against 57.5 s probing), so the
    // render pins them and an explicit CMF_GPU_PROBE still wins.
    if std::env::var_os("CMF_GPU_PROBE").is_none() {
        // SAFETY: single-threaded, before any engine or GPU init.
        unsafe { std::env::set_var("CMF_GPU_PROBE", "0") };
    }
    // GPU-vs-host is decided by the ENGINE's parity probe on this
    // file's own first qkv weight (videogen::mmh3_gpu_parity_probe):
    // the arm that renders is the arm that got probed, per stack —
    // measured wrong on an RTX PRO 6000, healthy on 2×RTX 5090.
    // CMF_MMH3_GPU=1/0 and an explicit CMF_GPU still force.
    //
    // The cooperative-matrix hold is LIFTED: the kernel now carries an
    // activation scale (its f16 operands can no longer overflow — that
    // overflow was the 512×288 NaN) and dequantizes the weight plane
    // once instead of per activation tile. The engine's parity probe
    // validates exactly this path on this file's own weights before
    // any frame is rendered, and the tensor cores are worth 1.5× on a
    // denoise step. CMF_COOP=0 still forces the scalar arm.

    let t0 = std::time::Instant::now();
    let anim = cortiq_engine::videogen::generate(
        std::path::Path::new(model),
        prompt,
        &params,
        |stage, i, n| {
            eprintln!("{stage} {i}/{n} ({:.1}s)", t0.elapsed().as_secs_f64());
        },
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let plane = anim.height * anim.width;
    let frames: Vec<Vec<u8>> = (0..anim.frames)
        .map(|f| {
            // The renderer's planes are [3, frames, h, w]; the encoder
            // wants one frame's three planes contiguous.
            let mut rgb = vec![0f32; 3 * plane];
            for c in 0..3 {
                let s = (c * anim.frames + f) * plane;
                rgb[c * plane..(c + 1) * plane].copy_from_slice(&anim.rgb[s..s + plane]);
            }
            avout::encode_jpeg(&rgb, anim.height, anim.width, quality)
        })
        .collect();
    if let Some(d) = frames_dir {
        // The renderer's planes are already the Vol layout ([3, F, H, W]),
        // so the LTX frame writer serialises them directly — this is the
        // interchange the refine stage (`ltx-video --video <dir>`) reads.
        // Anim.rgb is [0, 1]; the LTX frame codec speaks [-1, 1] (its VAE
        // decoder emits that, and `read_ppm_f32` maps back to it). Passing
        // [0, 1] through unmapped compresses every frame into the upper
        // half of the range — a washed-out clip, not an error.
        let vol = cortiq_engine::ltxvae::Vol {
            c: 3,
            f: anim.frames,
            h: anim.height,
            w: anim.width,
            data: anim.rgb.iter().map(|&v| v * 2.0 - 1.0).collect(),
        };
        crate::ltxcmd::write_frames(d, &vol)?;
    }
    let path = std::path::Path::new(out);
    avout::write_avi(
        path,
        &frames,
        anim.width,
        anim.height,
        cortiq_engine::videogen::FPS,
        &anim.audio,
        anim.samples,
        anim.sample_rate,
    )?;
    let wav = path.with_extension("wav");
    std::fs::write(
        &wav,
        avout::wav_bytes(&anim.audio, anim.samples, anim.sample_rate),
    )?;
    println!(
        "{out}: {}x{}, {} frames at {} fps ({:.2}s), {:.2}s of {} Hz stereo, {:.1} MB in {:.1}s",
        anim.width,
        anim.height,
        anim.frames,
        cortiq_engine::videogen::FPS,
        anim.frames as f64 / cortiq_engine::videogen::FPS as f64,
        anim.samples as f64 / anim.sample_rate as f64,
        anim.sample_rate,
        std::fs::metadata(path)?.len() as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    println!("{}: stereo PCM", wav.display());
    if let Some(rep) = cortiq_engine::audiovae::avae_time_report() {
        eprint!("{rep}");
    }
    if let Some(rep) = cortiq_engine::vae3d::vae3d_prof_report() {
        eprint!("{rep}");
    }
    if let Some(rep) = cortiq_engine::mmh3::mmh3_prof_report() {
        println!("{rep}");
    }
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    if let Some(rep) = cortiq_engine::gpu_wgpu::dit_phase_report() {
        println!("  {rep}");
    }
    Ok(())
}

fn cmd_story(model_path: &str) -> anyhow::Result<()> {
    let model = CmfModel::open_sharded(model_path)?;
    let arch = model.arch();
    let prov = model.header.provenance.as_ref();
    let sect = "─".repeat(60);

    // ── Who I am ──
    let full = arch
        .layer_types
        .iter()
        .filter(|t| {
            matches!(
                t,
                cortiq_core::LayerType::FullAttention | cortiq_core::LayerType::SlidingAttention
            )
        })
        .count();
    let conv = arch
        .layer_types
        .iter()
        .filter(|t| matches!(t, cortiq_core::LayerType::ShortConv))
        .count();
    // "mixer" layers = everything that isn't full softmax attention
    // (linear-attention cores and short-conv mixers alike).
    let linear = arch.num_layers - full;
    // Prefer the precise word when the non-full layers are conv mixers.
    let mixer = if conv > 0 { "conv" } else { "linear" };
    let body = match (&arch.moe, linear, full) {
        (Some(m), l, f) if l > 0 && f > 0 => format!(
            "hybrid: {l} {mixer} + {f} full-attention layers, MoE ({} experts, top-{})",
            m.num_experts, m.top_k
        ),
        (Some(m), _, _) => format!(
            "MoE transformer ({} experts, top-{})",
            m.num_experts, m.top_k
        ),
        (None, l, f) if l > 0 && f > 0 => {
            format!("hybrid: {l} {mixer} + {f} full-attention layers")
        }
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
                let thq = lf.get("thq/thk").and_then(|v| v.as_str()).unwrap_or("");
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
        let extra = lc
            .nphase
            .map(|n| format!(", {n} phases/head"))
            .unwrap_or_default();
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
        match mtp_excluded_by(&model) {
            Some(pats) => println!(
                "My source has an MTP head ({} block), but it was left out of this file \
                 (--exclude-tensors {pats}): no speculative speed-up.",
                mtp.num_layers
            ),
            None => println!(
                "I carry an MTP head ({} block) — I can speculatively speed up.",
                mtp.num_layers
            ),
        }
    }

    // ── Which skills I carry (swarm) ──
    if !model.header.skills.is_empty() {
        let total: u64 = model.tensors.iter().map(|t| t.nbytes).sum();
        println!(
            "\n\x1b[1mMy skill swarm ({}):\x1b[0m",
            model.header.skills.len()
        );
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
                    print!(
                        "  | {} {bb:.2}→{ov:.2} ({d:+.1}%)",
                        q.get("metric")
                            .and_then(|v| v.as_str())
                            .unwrap_or("quality")
                    );
                }
            } else {
                print!("  | quality NOT measured");
            }
            println!();
        }
        println!(
            "Skill selection is by reconstruction signal (recon-argmin), not by name; \
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
            print!(
                "; chat template {n} chars, {} stop token(s)",
                tc.eos_token_ids.len()
            );
        }
    }
    println!(".");

    // ── How honest my confidence is (B1) ──
    if let Some(cal) = &model.header.calibration {
        print!("\nMy confidence is calibrated (T={:.2}", cal.temperature);
        if let (Some(a), Some(b)) = (cal.ece_before, cal.ece_after) {
            print!(", ECE {a:.3}→{b:.3}");
        }
        println!("): I show softmax probability as a measured property, not a raw estimate.");
    }

    // ── Am I part of a whole ──
    if let Some(sh) = &model.header.shard {
        println!(
            "I am shard {} of {} (the full model is in the neighbors).",
            sh.no, sh.count
        );
    }

    // ── Am I intact (verifiable) ──
    print!("\nIntegrity: ");
    let problems = model.verify();
    if problems.is_empty() {
        println!(
            "\x1b[38;2;80;220;100mall {} hashes matched — I am not corrupted or tampered with.\x1b[0m",
            model.tensors.len()
        );
    } else {
        println!(
            "\x1b[38;2;230;90;80m{} problem(s) — the file is corrupted:\x1b[0m",
            problems.len()
        );
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
    row(
        "format",
        format!("v{}", a.header.version),
        format!("v{}", b.header.version),
    );
    row("arch", aa.arch_name.clone(), ba.arch_name.clone());
    row(
        "layers",
        aa.num_layers.to_string(),
        ba.num_layers.to_string(),
    );
    row(
        "hidden",
        aa.hidden_size.to_string(),
        ba.hidden_size.to_string(),
    );
    row(
        "ffn",
        aa.intermediate_size.to_string(),
        ba.intermediate_size.to_string(),
    );
    row(
        "vocab",
        aa.vocab_size.to_string(),
        ba.vocab_size.to_string(),
    );
    row(
        "linear_core",
        aa.linear_core_identity()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".into()),
        ba.linear_core_identity()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".into()),
    );
    row(
        "quant",
        format!("{:?}", a.header.quant_type),
        format!("{:?}", b.header.quant_type),
    );
    row(
        "params",
        format!("{:.3}B", a.total_param_count() as f64 / 1e9),
        format!("{:.3}B", b.total_param_count() as f64 / 1e9),
    );
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
    let (mut added, mut removed, mut changed, mut same) =
        (Vec::new(), Vec::new(), Vec::new(), 0u64);
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
        same,
        added.len(),
        removed.len(),
        changed.len()
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
        &added
            .iter()
            .map(|(n, d, b)| format!("{n}  [{d}, {:.1} MB]", *b as f64 / 1e6))
            .collect::<Vec<_>>(),
    );
    show(
        "  \x1b[38;2;230;90;80m− removed:\x1b[0m",
        &removed
            .iter()
            .map(|(n, d, b)| format!("{n}  [{d}, {:.1} MB]", *b as f64 / 1e6))
            .collect::<Vec<_>>(),
    );
    show(
        "  \x1b[38;2;230;190;80m~ changed:\x1b[0m",
        &changed
            .iter()
            .map(|(n, da, db, na, nb)| {
                let dt = if da == db {
                    da.clone()
                } else {
                    format!("{da}→{db}")
                };
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
        m.header
            .skills
            .iter()
            .map(|s| (s.id.clone(), s.layers.clone()))
            .collect()
    };
    let (sa, sb) = (sid(&a), sid(&b));
    if !sa.is_empty() || !sb.is_empty() {
        let new_sk: Vec<_> = sb
            .keys()
            .filter(|k| !sa.contains_key(*k))
            .cloned()
            .collect();
        let del_sk: Vec<_> = sa
            .keys()
            .filter(|k| !sb.contains_key(*k))
            .cloned()
            .collect();
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

/// `cortiq migrate-embryo-bits`: move the pre-0.8.1 Embryo bits 7/8 to
/// 12/13 in place (see `CmfModel::migrate_legacy_embryo_bits`), then prove
/// the result opens. Every file is reported; the command fails if any file
/// failed.
fn cmd_migrate_embryo_bits(files: &[String], dry_run: bool) -> anyhow::Result<()> {
    use cortiq_core::format::LegacyBitsMigration;
    let mut failed = 0usize;
    for path in files {
        match cortiq_core::CmfModel::migrate_legacy_embryo_bits(path, dry_run) {
            Ok(LegacyBitsMigration::NotCmf) => {
                println!("{path}: not a CMF file — skipped");
            }
            Ok(LegacyBitsMigration::NothingToDo { required_features }) => {
                println!("{path}: no pre-0.8.1 Embryo bits (required_features {required_features:#x}) — unchanged");
            }
            Ok(LegacyBitsMigration::Migrated { from, to }) if dry_run => {
                println!("{path}: would move required_features {from:#x} -> {to:#x}");
            }
            Ok(LegacyBitsMigration::Migrated { from, to }) => {
                println!("{path}: required_features {from:#x} -> {to:#x}");
                if let Err(e) = cortiq_core::CmfModel::open(path) {
                    failed += 1;
                    eprintln!(
                        "{path}: bits moved, but the file still does not open under this reader: {e}"
                    );
                }
                if std::path::Path::new(&format!("{path}.sig")).exists() {
                    println!("{path}: its detached .sig signed the old bytes — re-sign with `cortiq sign`");
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("{path}: {e}");
            }
        }
    }
    anyhow::ensure!(failed == 0, "{failed} of {} file(s) failed", files.len());
    Ok(())
}

async fn cmd_verify(model_path: &str) -> anyhow::Result<()> {
    println!("Verifying {} ...", model_path);
    // open() already enforces magic/version/features/section bounds.
    // Each shard is a self-contained valid .cmf (spec §10), so
    // verify opens the file as is, without merging neighbors.
    let model = CmfModel::open(model_path)?;
    println!(
        "  ✓ envelope, sections, tensor directory ({} tensors)",
        model.tensors.len()
    );

    let problems = model.verify();
    if problems.is_empty() {
        println!("  ✓ all tensor hashes match");
        // Authenticity on top of integrity: a detached <model>.sig
        // (cortiq sign) is verified when present — absence is not an
        // error, signing is opt-in.
        if sign::verify_detached(model_path, None)? {
            println!("  ✓ detached signature valid");
        }
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
    println!(
        "  {:<15} {:>8} {:>12} {:>6} {:>8}",
        "Name", "Sparsity", "Quality", "Layers", "Hot"
    );
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

#[allow(clippy::too_many_arguments)]
/// The two bandwidths the MoE hybrid split is arithmetic over, measured
/// on expert-sized blocks (13 MB class): host-RAM copy — the CPU arm's
/// ceiling — and host-to-VRAM upload — the fetch arm's. Once per machine.
fn cmd_bench_bw(json: bool, model: &str) -> anyhow::Result<()> {
    const BLOCK: usize = 13 * 1024 * 1024;
    const ROUNDS: usize = 24;
    // Storage read over the MODEL file itself, cache dropped per range —
    // a Gen5 NVMe reads at RAM-class rates and the offload defaults must
    // know that (a slow-disk assumption turns eviction and tiering into
    // anti-optimisations on fast machines, measured both ways).
    let storage_gbs = (|| -> Option<f64> {
        #[cfg(target_os = "linux")]
        use std::os::unix::io::AsRawFd;
        let f = std::fs::File::open(model).ok()?;
        let len = f.metadata().ok()?.len();
        if len < (BLOCK * 4) as u64 {
            return None;
        }
        let mut buf = vec![0u8; BLOCK];
        let mut total = 0usize;
        let t0 = std::time::Instant::now();
        for i in 0..8u64 {
            let off = (len / 9) * (i + 1) / BLOCK as u64 * BLOCK as u64;
            #[cfg(target_os = "linux")]
            unsafe {
                libc::posix_fadvise(
                    f.as_raw_fd(),
                    off as i64,
                    BLOCK as i64,
                    libc::POSIX_FADV_DONTNEED,
                );
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileExt;
                f.read_exact_at(&mut buf, off).ok()?;
            }
            #[cfg(not(unix))]
            {
                use std::io::{Read, Seek, SeekFrom};
                let mut fh = f.try_clone().ok()?;
                fh.seek(SeekFrom::Start(off)).ok()?;
                fh.read_exact(&mut buf).ok()?;
            }
            std::hint::black_box(&buf);
            total += BLOCK;
        }
        Some(total as f64 / t0.elapsed().as_secs_f64() / 1e9)
    })();
    let src = vec![7u8; BLOCK];
    let mut dst = vec![0u8; BLOCK];
    // Warm both buffers, then time the copies.
    dst.copy_from_slice(&src);
    let t0 = std::time::Instant::now();
    for _ in 0..ROUNDS {
        dst.copy_from_slice(&src);
        std::hint::black_box(&dst);
    }
    let host_gbs = (BLOCK * ROUNDS) as f64 / t0.elapsed().as_secs_f64() / 1e9;
    let pcie_gbs = cortiq_engine::gpu::upload_bandwidth_probe(BLOCK, ROUNDS);
    if json {
        println!(
            "{{\"host_copy_gbs\": {host_gbs:.2}, \"upload_gbs\": {:.2}, \"storage_gbs\": {:.2}}}",
            pcie_gbs.unwrap_or(0.0),
            storage_gbs.unwrap_or(0.0)
        );
    } else {
        println!("copy into RAM: {host_gbs:.1} GB/s (the ceiling of the CPU arm)");
        match pcie_gbs {
            Some(v) => println!("upload into VRAM: {v:.1} GB/s (the ceiling of the fetch arm)"),
            None => println!("upload into VRAM: no device"),
        }
        match storage_gbs {
            Some(v) => println!("storage read: {v:.1} GB/s (the source of cold bytes)"),
            None => println!("storage read: the file is too small to measure"),
        }
    }
    Ok(())
}

/// Select the prompt timing path used by the measured generation.  A
/// selected generation prefill must not be preceded by an unselected full
/// `forward_ids` pass: that pass is useful for the ordinary reference row,
/// but it is a misleading depth measurement for bounded O(1) or batched
/// graph ingestion.
fn bench_generation_prefill_path(
    o1_active: bool,
    task_mask_present: bool,
    prompt_len: usize,
    o1_prefill: Option<usize>,
    batch_k: Option<usize>,
    graph_prefill: bool,
) -> Option<&'static str> {
    if task_mask_present {
        return None;
    }
    if o1_active && o1_prefill.is_some_and(|p| p > 0 && p < prompt_len) {
        return Some("generation_ttft_bounded_o1");
    }
    if !o1_active && batch_k.is_some_and(|k| k > 0) {
        return Some(if graph_prefill {
            "generation_ttft_token_graph_prefill"
        } else {
            "generation_ttft_batched_graph"
        });
    }
    None
}

async fn cmd_bench(
    model_path: &str,
    task: &str,
    tokens: u32,
    ctx: Option<usize>,
    o1: &O1Flags,
    json: bool,
    core: bool,
    ignore_eos: bool,
    peer: Option<&str>,
    peer_split: Option<usize>,
    net_token: Option<&str>,
    net_dtype: &str,
    peer_head: bool,
    gpus: Option<usize>,
) -> anyhow::Result<()> {
    if !json {
        println!(
            "Benchmark: {} | task={} | tokens={}",
            model_path, task, tokens
        );
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
    // Keep every benchmark self-describing. Kernel conclusions are invalid
    // if the measured artifact no longer contains the dtype under test —
    // a whole Adreno tuning round was spent on a q8_2f kernel that this
    // file has zero tensors of. The FULL histogram, not a hand-picked few:
    // the q1 family alone is q1/q1s/q1t, and naming three dtypes here only
    // moves the same trap one requantization along.
    let tensor_count = model.tensors.len();
    let mut tensor_dtypes: std::collections::BTreeMap<&'static str, usize> = Default::default();
    for t in &model.tensors {
        *tensor_dtypes.entry(t.dtype.name()).or_default() += 1;
    }
    let dtype_n = |d: TensorDtype| tensor_dtypes.get(d.name()).copied().unwrap_or(0);
    let (tensor_q1, tensor_q8_2f, tensor_f16) = (
        dtype_n(TensorDtype::Q1),
        dtype_n(TensorDtype::Q8_2f),
        dtype_n(TensorDtype::F16),
    );
    let mut pipeline = Pipeline::from_model(
        &model,
        SamplerConfig {
            temperature: 0.0, // greedy: benchmark must be deterministic
            seed: Some(42),
            // --core: no penalty pass → the sampler's clone-free argmax
            repetition_penalty: if core {
                1.0
            } else {
                SamplerConfig::default().repetition_penalty
            },
            ..Default::default()
        },
    )?;
    if ignore_eos {
        // A loop flag, not a suppression: the earlier form put every EOS id
        // into `suppress_tokens`, which the engine counts as a penalty and
        // answers by switching the speculative round and the greedy burst
        // off — so `bench --ignore-eos` never measured either (mtp_drafted
        // stayed 0 on every card). The stream may now carry EOS ids; the
        // timing is what the flag is for.
        pipeline.ignore_eos = true;
    }
    if core {
        pipeline.set_confidence(false);
        if !json {
            println!("  Core timing: sampler/confidence excluded (llama-bench contract)");
        }
    }
    o1.apply(&mut pipeline)?;
    if let Some(n) = gpus {
        let have = cortiq_engine::gpu::device_count();
        if n < 2 {
            anyhow::bail!("--gpus {n}: needs ≥ 2 cards");
        }
        if have < n {
            anyhow::bail!("--gpus {n}, but {have} cards are visible");
        }
        let devs: Vec<usize> = (0..n).collect();
        pipeline
            .set_gpu_plan_at(Some(&devs), peer_split)
            .map_err(|e| anyhow::anyhow!("--gpus {n}: {e}"))?;
        if !json {
            if let Some(plan) = pipeline.gpu_plan() {
                println!("  GPUs:    {plan:?} (in-process split)");
            }
        }
    }
    if pipeline.o1_active() && !json {
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

    // ── Split bench (--gpus / --peer): the honest procedure — one
    // warmup generation (shader compile + weight upload + prefill land
    // here, untimed), then three measured repeats, median steady tok/s
    // from inter-token stamps. A benchmark where the graph refused or
    // weights re-uploaded mid-window FAILS instead of reporting CPU
    // numbers as GPU ones.
    if let Some(addr) = peer {
        pipeline.split_supported().map_err(|e| anyhow::anyhow!(e))?;
        let nl = pipeline.num_layers;
        let split = peer_split.unwrap_or(nl / 2);
        if split >= nl {
            anyhow::bail!("--peer-split {split}: the model has {nl} layers");
        }
        let dtype = match net_dtype {
            "f32" => cortiq_net::WireDtype::F32,
            "f16" => cortiq_net::WireDtype::F16,
            other => anyhow::bail!("--net-dtype {other}: f32 or f16"),
        };
        let spec = cortiq_net::SessionSpec {
            skill: None,
            task: mask.as_ref().map(|_| task.to_string()),
            o1: None,
            head: peer_head,
            sampler: if peer_head {
                Some(serde_json::to_string(&pipeline.sampler_config)?)
            } else {
                None
            },
            run_ahead: 1,
        };
        let mut rs = cortiq_net::RemoteSegment::connect(
            addr,
            net_token.unwrap_or(""),
            runtime.model().dir_hash(),
            &runtime.model().arch().arch_name,
            nl,
            pipeline.hidden_size,
            split,
            nl - 1,
            dtype,
            &spec,
        )
        .map_err(|e| anyhow::anyhow!(e))?;

        let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(4);
        let prompt_ids = pipeline.tokenizer.encode(&prompt);
        // Warmup: 32 tokens, untimed; its SplitStats carry the COLD
        // prefill (uploads included) — reported as such, not as steady.
        let (_wr, wst) = cortiq_net::generate_split(
            &mut pipeline,
            &mut rs,
            &prompt_ids,
            32,
            mask.as_ref(),
            None,
        )
        .map_err(|e| anyhow::anyhow!(e))?;

        let miss0 = cortiq_engine::pipeline::GRAPH_TOK_MISS.load(AtomicOrdering::Relaxed);
        let mut steady = Vec::new();
        let mut ttft = Vec::new();
        let mut net_share = Vec::new();
        let mut upload_delta = 0u64;
        let reps = 3usize;
        let meas = (tokens as usize).max(256);
        for _ in 0..reps {
            let up0 = cortiq_engine::gpu::upload_bytes();
            type Stamp = std::time::Instant;
            let stamps: Arc<std::sync::Mutex<Vec<Stamp>>> = Arc::default();
            let st = stamps.clone();
            let cb: cortiq_engine::TokenCallback = Box::new(move |_t| {
                st.lock().unwrap().push(std::time::Instant::now());
                true
            });
            let t0 = std::time::Instant::now();
            let (_r, sst) = cortiq_net::generate_split(
                &mut pipeline,
                &mut rs,
                &prompt_ids,
                meas,
                mask.as_ref(),
                Some(cb),
            )
            .map_err(|e| anyhow::anyhow!(e))?;
            upload_delta += cortiq_engine::gpu::upload_bytes() - up0;
            let stamps = stamps.lock().unwrap();
            let n = stamps.len();
            if n >= 34 {
                // Drop 32 ramp tokens from the window as well.
                steady.push((n - 33) as f64 / (stamps[n - 1] - stamps[32]).as_secs_f64().max(1e-9));
            }
            if let Some(first) = stamps.first() {
                ttft.push((*first - t0).as_secs_f64() * 1e3);
            }
            net_share.push(sst.net_s / sst.decode_s.max(1e-9));
        }
        let missn = cortiq_engine::pipeline::GRAPH_TOK_MISS.load(AtomicOrdering::Relaxed);
        steady.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ttft.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = |v: &Vec<f64>| v.get(v.len() / 2).copied().unwrap_or(0.0);
        let graph_miss = missn - miss0;
        let out = serde_json::json!({
            "mode": "split",
            "tensor_count": tensor_count,
            "tensor_dtypes": tensor_dtypes,
            "tensor_q1": tensor_q1,
            "tensor_q8_2f": tensor_q8_2f,
            "tensor_f16": tensor_f16,
            "split": split,
            "layers": nl,
            "wire": net_dtype,
            "adapter_local": std::env::var("CMF_GPU_ADAPTER").unwrap_or_default(),
            "measured_tokens": meas,
            "reps": reps,
            "prefill_cold_tok_s": wst.prefilled as f64 / wst.prefill_s.max(1e-9),
            "ttft_ms_median": med(&ttft),
            "decode_tok_s_steady_median": med(&steady),
            "decode_tok_s_steady_all": steady,
            "net_share_of_decode": net_share,
            "local_graph_miss_tokens": graph_miss,
            "steady_upload_bytes": upload_delta,
        });
        if json {
            println!("{out}");
        } else {
            println!(
                "split {split}/{nl} wire {net_dtype}: steady {:.1} tok/s (median of {reps}),                  ttft {:.0} ms, cold prefill {:.1} tok/s",
                med(&steady),
                med(&ttft),
                wst.prefilled as f64 / wst.prefill_s.max(1e-9),
            );
        }
        // Honest-bench contract: fast-path refusal or steady-window
        // re-upload disqualifies the number LOUDLY.
        if graph_miss > 0 {
            anyhow::bail!(
                "BENCH INVALID: the local segment declined the graph on {graph_miss} tokens —                  this is a CPU number, not a GPU one (CMF_GPU_DEBUG=1 shows why)"
            );
        }
        if upload_delta > 64 * 1024 * 1024 {
            anyhow::bail!(
                "BENCH INVALID: {upload_delta} bytes of weights arrived inside the steady window —                  eviction/re-upload, raise CMF_GPU_VRAM_MB"
            );
        }
        return Ok(());
    }

    // Warmup: compile and exercise the actual decode path before measuring.
    // `forward_ids` alone is not enough for checkpoints with a speculative
    // draft: DSpark's resident pack is intentionally built on first use, so
    // a forward-only warmup used to charge ~1.5 s of one-time upload to the
    // first inter-token "steady" window. A short untimed generation also
    // warms the ordinary sampler/head path and is the local equivalent of
    // the remote benchmark's existing 32-token warmup.
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
            anyhow::bail!(
                "ctx {n}: synthetic prompt tokenized to only {} tokens",
                prompt_ids.len()
            );
        }
    }
    let warm_ids = &prompt_ids[..2.min(prompt_ids.len())];
    let _ = pipeline
        .forward_ids(warm_ids, mask.as_ref())
        .map_err(|e| anyhow::anyhow!(e))?;
    let _ = pipeline
        .generate_from_ids(warm_ids, 8, mask.as_ref(), None)
        .map_err(|e| anyhow::anyhow!(e))?;

    // Prefill benchmark.  Bounded O(1) generation deliberately calibrates
    // only the requested prefix and streams the suffix through its sealed
    // state; batched graph generation similarly chooses a different ingest
    // path.  Running forward_ids here would silently do a second full
    // prefill before the real selected generation, so it would dominate
    // depth/TTFT and report the wrong path as `prefill_tok_s`.
    let o1_prefill_requested = std::env::var("CMF_O1_PREFILL")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    // The pipeline's own route (env or the discrete-card default), so the
    // reported prefill is the one generation takes.
    let batch_k_requested = Some(pipeline.generation_batch_k()).filter(|k| *k > 0);
    let prefill_measurement = bench_generation_prefill_path(
        pipeline.o1_active(),
        mask.is_some(),
        prompt_ids.len(),
        o1_prefill_requested,
        batch_k_requested,
        pipeline.generation_graph_prefill(),
    );
    let bounded_o1_prefill = prefill_measurement == Some("generation_ttft_bounded_o1");
    let batched_graph_prefill = prefill_measurement == Some("generation_ttft_batched_graph");
    let token_graph_prefill = prefill_measurement == Some("generation_ttft_token_graph_prefill");
    let generation_prefill = prefill_measurement.is_some();
    let prefill_s = if generation_prefill {
        None
    } else {
        cortiq_engine::cpuprof::reset();
        let t0 = std::time::Instant::now();
        let _ = pipeline
            .forward_ids(&prompt_ids, mask.as_ref())
            .map_err(|e| anyhow::anyhow!(e))?;
        let dt = t0.elapsed().as_secs_f64();
        cortiq_engine::cpuprof::report("prefill forward_ids", prompt_ids.len());
        Some(dt)
    };

    // Pair-fusion micro-bench: the memory-traffic win MTP verify rides
    // on. Skipped under o1 — forward_pair appends into the (sealed,
    // emptied) cache, so its numbers would be meaningless there — and on
    // a bounded-native file, whose sequence the resident graph may own:
    // the CPU pair walk would reset the device sequence and measure a
    // path the file never decodes on.
    let (singles_ms, pair_ms) = if pipeline.o1_active() || pipeline.bounded_native() {
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
    type Stamp = (std::time::Instant, u64, usize, GpuCounterSnapshot);
    let graph_ok0 = cortiq_engine::pipeline::GRAPH_TOK_OK.load(AtomicOrdering::Relaxed);
    let graph_prefix0 = cortiq_engine::pipeline::GRAPH_TOK_PREFIX.load(AtomicOrdering::Relaxed);
    let graph_full0 = cortiq_engine::pipeline::GRAPH_TOK_FULL.load(AtomicOrdering::Relaxed);
    let graph_miss0 = cortiq_engine::pipeline::GRAPH_TOK_MISS.load(AtomicOrdering::Relaxed);
    let native_metal0 = native_metal_bench_counters();
    let stamps: Arc<std::sync::Mutex<Vec<Stamp>>> = Arc::default();
    let st = stamps.clone();
    let cb: cortiq_engine::TokenCallback = Box::new(move |_tok| {
        st.lock().unwrap().push((
            std::time::Instant::now(),
            ALLOCS.load(AtomicOrdering::Relaxed),
            cortiq_engine::pool::dispatch_count(),
            gpu_counter_snapshot(),
        ));
        true
    });
    cortiq_engine::cpuprof::reset();
    let t1 = std::time::Instant::now();
    let result = pipeline
        .generate_from_ids(&prompt_ids, tokens as usize, mask.as_ref(), Some(cb))
        .map_err(|e| anyhow::anyhow!(e))?;
    let total_s = t1.elapsed().as_secs_f64();
    cortiq_engine::cpuprof::report("generation (prefill + decode)", result.tokens_generated);
    let graph_ok = cortiq_engine::pipeline::GRAPH_TOK_OK
        .load(AtomicOrdering::Relaxed)
        .saturating_sub(graph_ok0);
    let graph_prefix = cortiq_engine::pipeline::GRAPH_TOK_PREFIX
        .load(AtomicOrdering::Relaxed)
        .saturating_sub(graph_prefix0);
    let graph_full = cortiq_engine::pipeline::GRAPH_TOK_FULL
        .load(AtomicOrdering::Relaxed)
        .saturating_sub(graph_full0);
    let graph_miss = cortiq_engine::pipeline::GRAPH_TOK_MISS
        .load(AtomicOrdering::Relaxed)
        .saturating_sub(graph_miss0);
    let native_metal = native_metal_bench_counters().since(native_metal0);
    let gpu_vram_budget = cortiq_engine::gpu::vram_budget();
    let gpu_resident_weight = cortiq_engine::gpu::resident_bytes();
    let (o1_device_layers, o1_device_state_bytes) = pipeline.o1_device_stats();

    let stamps = stamps.lock().unwrap();
    // stamp[0] fires right after generation's prefill (the first token
    // is sampled from the prefill hidden, no decode forward yet).
    // Steady-window weight bytes: the delta between the first and last
    // per-token stamps, over the same inter-token window as tok/s. The
    // process total also holds the warmup, the timed prefill (one graph
    // pass per position on GDN hybrids) and the pair micro-bench.
    let n_st = stamps.len();
    let wb_total = if n_st >= 2 {
        stamps[n_st - 1]
            .3
            .weight_bytes
            .saturating_sub(stamps[0].3.weight_bytes)
    } else {
        cortiq_engine::gpu::weight_bytes_dispatched()
    };
    let wb_steps = if n_st >= 2 { n_st - 1 } else { n_st.max(1) };
    #[cfg(target_os = "macos")]
    let ffn_calls_per_token = cortiq_engine::gpu_metal::FFN_CALLS
        .load(std::sync::atomic::Ordering::Relaxed) as f64
        / n_st.max(1) as f64;
    #[cfg(not(target_os = "macos"))]
    let ffn_calls_per_token = 0.0f64;
    let decode_tps = if n_st >= 2 {
        (n_st - 1) as f64 / (stamps[n_st - 1].0 - stamps[0].0).as_secs_f64().max(1e-9)
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
    let (
        metal_submits_per_token,
        wgpu_submits_per_token,
        wgpu_passes_per_token,
        wgpu_steady_upload_ms_per_token,
        wgpu_steady_upload_mb_per_token,
    ) = if n_st >= 2 {
        let steps = (n_st - 1) as f64;
        let first = stamps[0].3;
        let last = stamps[n_st - 1].3;
        (
            last.metal_submits.saturating_sub(first.metal_submits) as f64 / steps,
            last.wgpu_submits.saturating_sub(first.wgpu_submits) as f64 / steps,
            last.wgpu_passes.saturating_sub(first.wgpu_passes) as f64 / steps,
            last.wgpu_upload_ns.saturating_sub(first.wgpu_upload_ns) as f64 / 1e6 / steps,
            last.wgpu_upload_bytes
                .saturating_sub(first.wgpu_upload_bytes) as f64
                / 1e6
                / steps,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    };
    let ttft_s = stamps
        .first()
        .map(|s| s.0.duration_since(t1).as_secs_f64())
        .unwrap_or(0.0);
    // For the bounded profile, TTFT is the only honest end-to-end prompt
    // measure because it includes exactly the prefix calibration plus the
    // O(1) suffix stream that production generation uses.  Keep the field
    // joinable with ordinary rows and label its provenance explicitly.
    let prefill_tok_s = prefill_s
        .map(|s| prompt_ids.len() as f64 / s.max(1e-9))
        .unwrap_or_else(|| prompt_ids.len() as f64 / ttft_s.max(1e-9));
    if !json {
        if bounded_o1_prefill {
            println!(
                "  Prompt:  {} tokens | bounded generation prefill {:.1} tok/s (TTFT)",
                prompt_ids.len(),
                prefill_tok_s
            );
        } else if token_graph_prefill {
            println!(
                "  Prompt:  {} tokens | token-graph generation prefill {:.1} tok/s (TTFT)",
                prompt_ids.len(),
                prefill_tok_s
            );
        } else if batched_graph_prefill {
            println!(
                "  Prompt:  {} tokens | batched generation prefill {:.1} tok/s (TTFT)",
                prompt_ids.len(),
                prefill_tok_s
            );
        } else {
            println!(
                "  Prompt:  {} tokens | prefill {:.1} tok/s",
                prompt_ids.len(),
                prefill_tok_s
            );
        }
    }
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
    // Bounded-anchor rings (host) and what the resident device graph
    // actually allocated for this sequence: both are the O(1) claim as
    // numbers, read from live state, never derived from geometry.
    let bounded_mem = pipeline.kv_cache.bounded_state_bytes();
    let device_state = pipeline.device_state_bytes();
    if json {
        // llama-bench-compatible spirit: one flat JSON object, raw
        // numbers only — joinable without parsing human text.
        let obj = serde_json::json!({
            "model": model_path,
            "tensor_count": tensor_count,
            "tensor_dtypes": tensor_dtypes,
            "tensor_q1": tensor_q1,
            "tensor_q8_2f": tensor_q8_2f,
            "tensor_f16": tensor_f16,
            "task": task,
            "ctx": ctx,
            "o1": pipeline.o1_active(),
            "o1_prefill_tokens": std::env::var("CMF_O1_PREFILL")
                .ok()
                .and_then(|v| v.parse::<usize>().ok()),
            "threads_env": std::env::var("CMF_THREADS").ok(),
            "prompt_tokens": prompt_ids.len(),
            "prefill_tok_s": prefill_tok_s,
            "prefill_forward_s": prefill_s,
            "prefill_measurement": prefill_measurement.unwrap_or("forward_ids"),
            "tokens_generated": result.tokens_generated,
            "metal_submits_per_token": metal_submits_per_token,
            "wgpu_submits_per_token": wgpu_submits_per_token,
            "wgpu_passes_per_token": wgpu_passes_per_token,
            "wgpu_upload_ms": wgpu_uploads().0,
            "wgpu_upload_mb": wgpu_uploads().1,
            "wgpu_steady_upload_ms_per_token": wgpu_steady_upload_ms_per_token,
            "wgpu_steady_upload_mb_per_token": wgpu_steady_upload_mb_per_token,
            "graph_tokens": graph_ok,
            "graph_full_tokens": graph_full,
            "graph_prefix_tokens": graph_prefix,
            "graph_miss_tokens": graph_miss,
            "native_metal_graph_tokens": native_metal.graph_tokens,
            "native_metal_graph_layers": native_metal.graph_layers,
            "native_metal_graph_errors": native_metal.graph_errors,
            "native_metal_sparse_moe_blocks": native_metal.sparse_moe_blocks,
            "native_metal_sparse_moe_span_bytes": native_metal.sparse_moe_span_bytes,
            "gpu_vram_budget_bytes": gpu_vram_budget,
            "gpu_resident_weight_bytes": gpu_resident_weight,
            "o1_device_layers": o1_device_layers,
            "o1_device_state_bytes": o1_device_state_bytes,
            "decode_tok_s_steady": decode_tps,
            "decode_tok_s_incl_prefill": result.tokens_generated as f64 / total_s.max(1e-9),
            "ttft_s": ttft_s,
            "allocs_per_token": allocs_per_token,
            "weight_gb_per_token": if n_st >= 2 { wb_total as f64 / wb_steps as f64 / 1e9 } else { 0.0 },
            "weight_gb_by_stage": cortiq_engine::gpu::weight_bytes_by().iter().map(|b| *b as f64 / n_st.max(1) as f64 / 1e9).collect::<Vec<_>>(),
            "ffn_calls_per_token": ffn_calls_per_token,
            "n_stamps": n_st,
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
        // Bounded-anchor / resident-device telemetry (kept outside the
        // macro: `json!` hits its recursion limit past ~40 keys).
        let mut obj = obj;
        obj["bounded_state_bytes"] = serde_json::json!(bounded_mem);
        obj["bounded_native"] = serde_json::json!(pipeline.bounded_native());
        obj["device_state_bytes"] = serde_json::json!(device_state.map(|(s, _)| s));
        obj["device_kv_bytes"] = serde_json::json!(device_state.map(|(_, k)| k));
        // Positions the bounded reuse record holds (the host cache's
        // seq_len stays 0 when the resident graph owns the sequence).
        obj["consumed_prefix_len"] = serde_json::json!(pipeline.kv_prefix.len());
        // Dynamic-tail prepared frame + global pool: what the MoE frame
        // spends per token (host encode vs GPU wait vs uploads) and how
        // often the pool admits a new expert. Without these, a frame that
        // trades CPU MoE for per-layer syncs is invisible in the report.
        // `gpu_wgpu` exists only when the optional GPU backend is enabled;
        // CPU-only builds deliberately omit these GPU-only metrics.
        #[cfg(feature = "gpu")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            let steps = (n_st.saturating_sub(1)).max(1) as f64;
            let enc = cortiq_engine::gpu_wgpu::MOE_ENC_NS.load(Relaxed) as f64 / 1e6;
            let wait = cortiq_engine::gpu_wgpu::MOE_WAIT_NS.load(Relaxed) as f64 / 1e6;
            let up = cortiq_engine::gpu_wgpu::MOE_UP_NS.load(Relaxed) as f64 / 1e6;
            let fills = cortiq_engine::gpu_wgpu::DSV4_FILLS.load(Relaxed);
            let fill_mb = cortiq_engine::gpu_wgpu::DSV4_FILL_BYTES.load(Relaxed) as f64 / 1e6;
            obj["moe_frame_enc_ms_per_tok"] = serde_json::json!(enc / steps);
            obj["moe_frame_wait_ms_per_tok"] = serde_json::json!(wait / steps);
            obj["moe_frame_up_ms_per_tok"] = serde_json::json!(up / steps);
            obj["moe_pool_fills"] = serde_json::json!(fills);
            obj["moe_pool_fill_mb"] = serde_json::json!(fill_mb);
            obj["moe_pool_fills_per_tok"] = serde_json::json!(fills as f64 / steps);
            obj["moe_pooled_calls_per_tok"] = serde_json::json!(
                cortiq_engine::gpu_wgpu::POOLED_CALLS.load(Relaxed) as f64 / steps
            );
            obj["moe_pooled_cold_per_tok"] = serde_json::json!(
                cortiq_engine::gpu_wgpu::POOLED_COLD_N.load(Relaxed) as f64 / steps
            );
            obj["m3_span_ms_per_tok"] = serde_json::json!(
                cortiq_engine::pipeline::M3_SPAN_NS.load(Relaxed) as f64 / 1e6 / steps
            );
            obj["m3_cold_ms_per_tok"] = serde_json::json!(
                cortiq_engine::pipeline::M3_COLD_NS.load(Relaxed) as f64 / 1e6 / steps
            );
            obj["m3_arm_ms_per_tok"] = serde_json::json!(
                cortiq_engine::pipeline::M3_ARM_NS.load(Relaxed) as f64 / 1e6 / steps
            );
            obj["m3_cold_dev_per_tok"] = serde_json::json!(
                cortiq_engine::pipeline::M3_COLD_DEV_N.load(Relaxed) as f64 / steps
            );
            obj["m3_cold_n_per_tok"] = serde_json::json!(
                cortiq_engine::pipeline::M3_COLD_N.load(Relaxed) as f64 / steps
            );
        }
        println!("{}", serde_json::to_string_pretty(&obj)?);
        cortiq_engine::prism::perf_report();
        cortiq_engine::linear_core::perf_report();
        #[cfg(feature = "gpu")]
        cortiq_engine::gpu_wgpu::perf_report();
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
    if gpu_vram_budget > 0 {
        println!(
            "  GPU:     graph full {} | prefix {} | miss {} | resident {:.1} MB / budget {:.1} MB",
            graph_full,
            graph_prefix,
            graph_miss,
            gpu_resident_weight as f64 / 1e6,
            gpu_vram_budget as f64 / 1e6
        );
    }
    // `bench` is where the dsv4 breakdown is actually wanted — `run` had it
    // and this did not, so every timing question needed a second command
    // measuring a different workload.
    cortiq_engine::prism::profile_report();
    cortiq_engine::prism::perf_report();
    cortiq_engine::linear_core::perf_report();
    #[cfg(feature = "gpu")]
    cortiq_engine::gpu_wgpu::perf_report();
    #[cfg(feature = "gpu")]
    cortiq_engine::dsv4::profile_report();
    cortiq_engine::dsv41::profile_report();
    if pair_ms > 0.0 {
        println!(
            "  Pair:    2 singles {:.2} ms vs fused {:.2} ms (×{:.2} cheaper second lane)",
            singles_ms,
            pair_ms,
            singles_ms / pair_ms.max(1e-9)
        );
    }
    if let Some((st, kv)) = device_state {
        println!(
            "  Device:  resident state {} B + KV/ring {} B (constant in context)",
            st, kv
        );
    }
    if bounded_mem > 0 {
        println!(
            "  Memory:  KV+state {:.1} MB (bounded rings {} B + recurrent {} B, no per-position KV) at seq_len {}",
            total_mem as f64 / 1e6,
            bounded_mem,
            pipeline.kv_cache.recurrent_state_bytes(),
            pipeline.kv_cache.seq_len()
        );
    } else if nystrom_mem > 0 {
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

/// `@file` → its contents, else the argument itself.
fn arg_or_file(v: &str) -> anyhow::Result<(String, bool)> {
    match v.strip_prefix('@') {
        Some(path) => Ok((std::fs::read_to_string(path)?, path.ends_with(".json"))),
        None => Ok((v.to_string(), false)),
    }
}

fn run_system_one(
    model: &str,
    state: Option<&str>,
    questions: &str,
    images: &[String],
    audios: &[String],
) -> anyhow::Result<()> {
    let cmf = Arc::new(CmfModel::open(model)?);
    anyhow::ensure!(
        cortiq_engine::d1::is_d1(&cmf),
        "{model} is not a d1 decision model (no provenance.d1)"
    );
    let pipeline = Pipeline::from_model(&cmf, SamplerConfig::default())?;
    let d1 = cortiq_engine::d1::D1::from_model(&cmf).map_err(anyhow::Error::msg)?;
    let state = match state {
        None => serde_json::Value::Null,
        Some(s) => {
            let (text, json_file) = arg_or_file(s)?;
            if json_file { serde_json::from_str(&text)? } else { serde_json::Value::String(text) }
        }
    };
    let (q, _) = arg_or_file(questions)?;
    let questions: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&q)
        .map_err(|e| anyhow::anyhow!("--questions must be a JSON object of named questions: {e}"))?;
    let req = cortiq_engine::d1::Request {
        state,
        questions,
        images: images.iter().map(|p| serde_json::json!({"url": p})).collect(),
        audios: audios.iter().map(|p| serde_json::json!(p)).collect(),
    };
    let t0 = std::time::Instant::now();
    let out = d1.system_one(&pipeline, &req).map_err(anyhow::Error::msg)?;
    println!("{}", serde_json::to_string_pretty(&out)?);
    eprintln!("system-one: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moss_layout_prompt_is_validated_but_not_reformatted() {
        let raw = "{\n  \"summary\": \"two foxes\", \"items\": []\n}";
        assert_eq!(
            resolve_moss_prompt(None, Some(raw.to_owned()), None).unwrap(),
            raw
        );
        assert!(resolve_moss_prompt(None, Some("[]".to_owned()), None).is_err());
        assert!(resolve_moss_prompt(None, Some("not json".to_owned()), None).is_err());
        assert!(resolve_moss_prompt(Some("text".to_owned()), Some("{}".to_owned()), None).is_err());
    }

    #[test]
    fn moss_skip_lists_match_source_set_behavior() {
        assert_eq!(parse_moss_layer_list("22, 24,22", "--slg-layers").unwrap(), [22, 24]);
        assert_eq!(parse_moss_layer_list("none", "--uncond-skip-layers").unwrap(), Vec::<usize>::new());
        assert_eq!(parse_moss_layer_list("", "--uncond-skip-layers").unwrap(), Vec::<usize>::new());
        assert!(parse_moss_layer_list("22,,24", "--slg-layers").is_err());
        assert!(parse_moss_layer_list("nope", "--slg-layers").is_err());
    }

    #[test]
    fn moss_imagine_parser_accepts_layout_and_source_controls() {
        // The full command tree needs the main CLI's larger stack in debug.
        let cli = std::thread::Builder::new()
            .name("moss-imagine-parser".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                Cli::try_parse_from([
                    "cortiq", "imagine", "moss.cmf", "--layout", "{\"summary\":\"cat\"}",
                    "--uncond-skip-layers", "none", "--slg-layers", "24,26", "--slg-scale", "3",
                    "--slg-start", "0.01", "--slg-stop", "1",
                ])
                .unwrap()
            })
            .unwrap()
            .join()
            .unwrap();
        match cli.command {
            Commands::Imagine {
                prompt,
                layout,
                layout_file,
                uncond_skip_layers,
                slg_layers,
                slg_scale,
                slg_start,
                slg_stop,
                ..
            } => {
                assert!(prompt.is_none());
                assert_eq!(layout.as_deref(), Some("{\"summary\":\"cat\"}"));
                assert!(layout_file.is_none());
                assert_eq!(uncond_skip_layers.as_deref(), Some("none"));
                assert_eq!(slg_layers.as_deref(), Some("24,26"));
                assert_eq!(slg_scale, Some(3.0));
                assert_eq!(slg_start, Some(0.01));
                assert_eq!(slg_stop, Some(1.0));
            }
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn mimo_run_media_flags_parse() {
        // Parsing the complete CLI command tree in an unoptimized build
        // exceeds libtest's 2 MiB worker stack on both Linux and macOS.
        // Match the CLI main-thread stack for this parser test only; keep
        // the actual parser and every assertion (and other tests) unchanged.
        let cli = std::thread::Builder::new()
            .name("mimo-cli-parser".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                Cli::try_parse_from([
                    "cortiq", "run", "text.cmf", "--prompt", "describe",
                    "--image", "one.png", "--image", "two.png", "--video", "clip.y4m",
                    "--video-fps", "2", "--audio", "clip.wav", "--mm", "full.mm.cmf",
                    "--image-max-pixels", "200704",
                ]).unwrap()
            })
            .unwrap()
            .join()
            .unwrap();
        match cli.command {
            Commands::Run { images, videos, audios, video_fps, mm, image_max_pixels, .. } => {
                assert_eq!(images, ["one.png", "two.png"]);
                assert_eq!(videos, ["clip.y4m"]);
                assert_eq!(audios, ["clip.wav"]);
                assert_eq!(video_fps, Some(2.0));
                assert_eq!(mm.unwrap(), std::path::PathBuf::from("full.mm.cmf"));
                assert_eq!(image_max_pixels, Some(200704));
            }
            _ => panic!("wrong subcommand"),
        }
    }

    /// `quantize-moe --in-place` writes into the calibration file, so an
    /// `--output` next to it is refused at parse time.
    #[test]
    fn moegptq_output_conflicts_with_in_place() {
        let parse = |args: &'static [&'static str]| {
            std::thread::Builder::new()
                .name("quantize-moe-parser".into())
                .stack_size(8 * 1024 * 1024)
                .spawn(move || Cli::try_parse_from(args).map(|_| ()).map_err(|e| e.kind()))
                .unwrap()
                .join()
                .unwrap()
        };
        let base = ["cortiq", "quantize-moe", "calib.cmf", "--hf", "src", "--calib", "c.txt"];
        let with = |extra: &[&'static str]| -> &'static [&'static str] { Box::leak([&base[..], extra].concat().into_boxed_slice()) };
        assert_eq!(parse(with(&["--in-place", "--output", "x.cmf"])), Err(clap::error::ErrorKind::ArgumentConflict));
        assert_eq!(parse(with(&["--in-place"])), Ok(()));
        assert_eq!(parse(with(&["--output", "x.cmf"])), Ok(()));
        assert_eq!(parse(with(&[])), Err(clap::error::ErrorKind::MissingRequiredArgument));
    }

    #[test]
    fn convert_also_output_profiles_parse() {
        let cli = std::thread::Builder::new()
            .name("convert-also-output-parser".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                Cli::try_parse_from([
                    "cortiq",
                    "convert",
                    "--model",
                    "openbmb/MiniCPM-V-4.7-35B-A3B",
                    "--quant",
                    "q4tp",
                    "--output",
                    "q4.cmf",
                    "--also-output",
                    "q8_2f=q8.cmf",
                    "--also-output",
                    "f16=reference.cmf",
                ])
                .unwrap()
            })
            .unwrap()
            .join()
            .unwrap();
        match cli.command {
            Commands::Convert {
                model,
                quant,
                output,
                also_output,
                ..
            } => {
                assert_eq!(model, "openbmb/MiniCPM-V-4.7-35B-A3B");
                assert_eq!(quant.as_deref(), Some("q4tp"));
                assert_eq!(output, "q4.cmf");
                assert_eq!(also_output, ["q8_2f=q8.cmf", "f16=reference.cmf"]);
            }
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn convert_blr_options_parse() {
        let cli = std::thread::Builder::new()
            .name("convert-blr-parser".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                Cli::try_parse_from([
                    "cortiq",
                    "convert",
                    "--model",
                    "hf08",
                    "--quant",
                    "blr",
                    "--output",
                    "m.cmf",
                    "--bpw",
                    "1.0",
                    "--blr-method",
                    "dual-svid",
                    "--calib",
                    "calib.txt",
                    "--blr-paths",
                    "2",
                    "--blr-rho",
                    "0.125",
                    "--blr-fit-tokens",
                    "1024",
                    "--blr-device",
                    "cpu",
                ])
                .unwrap()
            })
            .unwrap()
            .join()
            .unwrap();
        match cli.command {
            Commands::Convert {
                quant,
                blr_bpw,
                blr_method,
                calib,
                blr_paths,
                blr_rho,
                blr_fit_tokens,
                blr_sel_tokens,
                blr_kv_factor,
                blr_device,
                ..
            } => {
                assert_eq!(blr_device, "cpu");
                assert_eq!(nvg::BlrDevice::resolve(&blr_device).unwrap(), nvg::BlrDevice::Cpu);
                assert!(nvg::BlrDevice::resolve("tpu").is_err());
                assert_eq!(quant.as_deref(), Some("blr"));
                assert_eq!(blr_bpw, 1.0);
                assert_eq!(blr_method, "dual-svid");
                assert_eq!(calib.as_deref(), Some("calib.txt"));
                assert_eq!(blr_paths, "2");
                assert_eq!(blr_rho, 0.125);
                assert_eq!(blr_fit_tokens, 1024);
                assert_eq!(blr_sel_tokens, 3072);
                assert_eq!(blr_kv_factor, 1.0);
            }
            _ => panic!("wrong subcommand"),
        }
    }

    /// A small q4tp + f16 container for the raw dequant/patch round trips.
    fn raw_patch_fixture(dir: &std::path::Path) -> std::path::PathBuf {
        use cortiq_core::format::TensorSpec;
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join("src.cmf");
        let header: cortiq_core::CmfHeader = serde_json::from_value(serde_json::json!({
            "format": "cmf", "version": cortiq_core::CMF_VERSION,
            "arch": {"arch_name": "llama", "hidden_size": 64,
                "intermediate_size": 128, "num_layers": 1, "num_attention_heads": 1,
                "num_kv_heads": 1, "head_dim": 64, "vocab_size": 64,
                "layer_types": ["FullAttention"], "tie_word_embeddings": true,
                "rms_norm_eps": 1e-6, "max_position_embeddings": 64},
            "quant_type": "F32"
        }))
        .unwrap();
        let w: Vec<f32> = (0..96 * 64).map(|i| (i as f32 * 0.37).sin() * 0.05).collect();
        let (q_dt, q_bytes) = convert::quantize_2d(convert::Quant::Q4TiledP, &w, 96, 64);
        assert_eq!(q_dt, TensorDtype::Q4TiledP);
        let g: Vec<f32> = (0..64).map(|i| 1.0 + i as f32 / 128.0).collect();
        let specs = vec![
            TensorSpec {
                name: "model.layers.0.mlp.up_proj.weight".into(),
                dtype: q_dt,
                shape: vec![96, 64],
                data: q_bytes,
            },
            TensorSpec {
                name: "model.layers.0.input_layernorm.weight".into(),
                dtype: TensorDtype::F16,
                shape: vec![64],
                data: convert::encode_f16(&g),
            },
        ];
        CmfModel::write(path.to_str().unwrap(), &header, &specs, None, None).unwrap();
        path
    }

    #[test]
    fn dequant_raw_and_patch_raw_round_trip_payloads_verbatim() {
        let dir = std::env::temp_dir().join(format!("cmf-raw-patch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = raw_patch_fixture(&dir);
        let s = src.to_str().unwrap();
        let lin = "model.layers.0.mlp.up_proj.weight";
        let norm = "model.layers.0.input_layernorm.weight";

        // dequant --dtype raw: one file per tensor, the payload verbatim.
        let dump = dir.join("dump");
        cmd_dequant(s, "model.", dump.to_str().unwrap(), "raw", true).unwrap();
        let model = CmfModel::open(s).unwrap();
        for n in [lin, norm] {
            let got = std::fs::read(dump.join(format!("{n}.bin"))).unwrap();
            assert_eq!(got.as_slice(), model.tensor_bytes(n).unwrap(), "{n}");
        }
        // f32 / bf16 dumps are unchanged by the raw branch.
        let e = model.tensor(norm).unwrap();
        let f32s = dequant_dump_bytes(e, model.tensor_bytes(norm).unwrap(), "f32").unwrap();
        assert_eq!(f32s.len(), 64 * 4);
        assert_eq!(f32::from_le_bytes(f32s[4..8].try_into().unwrap()), 1.0 + 1.0 / 128.0);
        assert!(dequant_dump_bytes(e, model.tensor_bytes(norm).unwrap(), "f8").is_err());

        // Edit only payload FIELDS the way the offline refit does: the q4tp
        // row params (f16 lo of row 0) and one f16 norm gain.
        let mut q = model.tensor_bytes(lin).unwrap().to_vec();
        let nib = 96 * 2 * 16;
        q[nib] ^= 1;
        let mut g = model.tensor_bytes(norm).unwrap().to_vec();
        g[0] ^= 1;
        drop(model);
        let (qp, gp) = (dir.join("q.bin"), dir.join("g.bin"));
        std::fs::write(&qp, &q).unwrap();
        std::fs::write(&gp, &g).unwrap();
        let sets = vec![
            format!("{lin}={}", qp.display()),
            format!("{norm}={}", gp.display()),
        ];

        // --output: a new file with exactly these payloads, hashes valid.
        let out = dir.join("out.cmf");
        cmd_patch_tensor(s, &sets, None, Some(out.to_str().unwrap()), true).unwrap();
        let m = CmfModel::open(out.to_str().unwrap()).unwrap();
        assert!(m.verify().is_empty(), "{:?}", m.verify());
        assert_eq!(m.tensor_bytes(lin).unwrap(), q.as_slice());
        assert_eq!(m.tensor_bytes(norm).unwrap(), g.as_slice());
        assert_eq!(m.tensor(lin).unwrap().dtype, TensorDtype::Q4TiledP);
        drop(m);

        // In place (same sizes): identical payloads, hashes recomputed.
        let inplace = dir.join("inplace.cmf");
        std::fs::copy(&src, &inplace).unwrap();
        cmd_patch_tensor(inplace.to_str().unwrap(), &sets, None, None, true).unwrap();
        let m = CmfModel::open(inplace.to_str().unwrap()).unwrap();
        assert!(m.verify().is_empty(), "{:?}", m.verify());
        assert_eq!(m.tensor_bytes(lin).unwrap(), q.as_slice());
        assert_eq!(m.tensor_bytes(norm).unwrap(), g.as_slice());
        drop(m);

        // A payload of the wrong length is refused before anything is written.
        std::fs::write(&qp, &q[..q.len() - 1]).unwrap();
        let before = std::fs::read(&inplace).unwrap();
        let err = cmd_patch_tensor(inplace.to_str().unwrap(), &sets[..1], None, None, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("payload length"), "{err}");
        assert_eq!(std::fs::read(&inplace).unwrap(), before);

        // --dtype: a q8_2f payload for the q4tp entry. It does not fit the
        // slot in place; with --output the entry changes dtype.
        let w: Vec<f32> = (0..96 * 64).map(|i| (i as f32 * 0.11).cos() * 0.02).collect();
        let (dt8, q8) = convert::quantize_2d(convert::Quant::Q8_2f, &w, 96, 64);
        assert_eq!(dt8, TensorDtype::Q8_2f);
        std::fs::write(&qp, &q8).unwrap();
        let err = cmd_patch_tensor(inplace.to_str().unwrap(), &sets[..1], Some("q8_2f"), None, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("use --output"), "{err}");
        let out8 = dir.join("out8.cmf");
        cmd_patch_tensor(s, &sets[..1], Some("q8_2f"), Some(out8.to_str().unwrap()), true).unwrap();
        let m = CmfModel::open(out8.to_str().unwrap()).unwrap();
        assert!(m.verify().is_empty());
        assert_eq!(m.tensor(lin).unwrap().dtype, TensorDtype::Q8_2f);
        assert_eq!(m.tensor_bytes(lin).unwrap(), q8.as_slice());
        drop(m);
        // A q4tp-sized payload declared q8_2f is a length error.
        std::fs::write(&qp, &q).unwrap();
        assert!(cmd_patch_tensor(s, &sets[..1], Some("q8_2f"), Some(out8.to_str().unwrap()), true).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

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
        let st2 = SessionState {
            kind: 0,
            fp: (1, 2, 3),
            seed: None,
            skill: None,
            tokens: vec![7],
        };
        st2.write(p).unwrap();
        let b2 = SessionState::read(p).unwrap();
        std::fs::remove_file(p).ok();
        assert!(b2.seed.is_none() && b2.skill.is_none() && b2.tokens == vec![7]);
    }

    #[test]
    fn qwen_bundle_defaults_to_self_and_honors_component_overrides() {
        let bundle = std::path::Path::new("/tmp/qwen-image-edit-2509-q4tp.cmf");
        let paths = qwen_component_paths(bundle, true, None, None, None);
        assert_eq!(paths.transformer, bundle);
        assert_eq!(paths.text_encoder, bundle);
        assert_eq!(paths.vae, bundle);
        assert!(paths.scheduler.is_none());

        let paths = qwen_component_paths(
            bundle,
            true,
            Some("/tmp/override-text.cmf"),
            Some("/tmp/override-vae.cmf"),
            Some("/tmp/override-scheduler.json"),
        );
        assert_eq!(
            paths.text_encoder,
            std::path::PathBuf::from("/tmp/override-text.cmf")
        );
        assert_eq!(paths.vae, std::path::PathBuf::from("/tmp/override-vae.cmf"));
        assert_eq!(
            paths.scheduler,
            Some(std::path::PathBuf::from("/tmp/override-scheduler.json"))
        );
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

    #[test]
    fn bench_prefill_path_matches_selected_generation() {
        assert_eq!(
            bench_generation_prefill_path(false, false, 2048, None, Some(128), false),
            Some("generation_ttft_batched_graph")
        );
        assert_eq!(
            bench_generation_prefill_path(false, false, 2048, None, Some(128), true),
            Some("generation_ttft_token_graph_prefill")
        );
        assert_eq!(
            bench_generation_prefill_path(true, false, 2048, Some(256), None, false),
            Some("generation_ttft_bounded_o1")
        );
        // Invalid/full-prefix and task-masked paths retain the ordinary
        // forward_ids timing, so a caller cannot accidentally skip it.
        assert_eq!(
            bench_generation_prefill_path(true, false, 2048, Some(2048), None, false),
            None
        );
        assert_eq!(
            bench_generation_prefill_path(false, true, 2048, None, Some(128), true),
            None
        );
    }

    #[test]
    fn teacher_data_cli_split_and_greedy_contract_are_strict() {
        assert_eq!(
            parse_teacher_data_split("90/5/5").unwrap(),
            teacher_data::SplitRatios::new(90, 5, 5).unwrap()
        );
        assert_eq!(
            parse_teacher_data_split("0/50/50").unwrap(),
            teacher_data::SplitRatios::new(0, 50, 50).unwrap()
        );
        for invalid in ["90/10", "90//10", "x/5/5", "0/0/0", "90/5/5/0"] {
            assert!(
                parse_teacher_data_split(invalid).is_err(),
                "invalid split {invalid:?} must be rejected"
            );
        }

        let config = teacher_data_greedy_sampler(0xCAFE_BABE);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_p, 1.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.min_p, 0.0);
        assert_eq!(config.repetition_penalty, 1.0);
        assert_eq!(config.presence_penalty, 0.0);
        assert_eq!(config.seed, Some(0xCAFE_BABE));
    }

    #[test]
    fn teacher_data_disable_mtp_is_an_explicit_cli_opt_in() {
        // The complete CLI tree needs the main thread's larger stack in a
        // debug test binary; parse the actual public spelling rather than
        // duplicating its default in a helper.
        let cli = std::thread::Builder::new()
            .name("teacher-data-disable-mtp-parser".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                Cli::try_parse_from([
                    "cortiq",
                    "teacher-data",
                    "generate",
                    "teacher.cmf",
                    "prompts.jsonl",
                    "out.tdata",
                    "--no-think",
                    "--disable-mtp",
                ])
                .unwrap()
            })
            .unwrap()
            .join()
            .unwrap();
        match cli.command {
            Commands::TeacherData {
                cmd: TeacherDataCmd::Generate { disable_mtp, .. },
            } => assert!(disable_mtp),
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn mtp_compact_missing_data_requires_a_complete_teacher_pair() {
        assert_eq!(
            mtp_compact_generation_inputs(
                "calibration.tdata",
                Some("teacher.cmf"),
                Some("prompts.jsonl"),
            )
            .unwrap(),
            ("teacher.cmf", "prompts.jsonl")
        );
        for (teacher, prompts) in [
            (None, None),
            (Some("teacher.cmf"), None),
            (None, Some("prompts.jsonl")),
        ] {
            let error = mtp_compact_generation_inputs("calibration.tdata", teacher, prompts)
                .unwrap_err()
                .to_string();
            assert!(error.contains("--teacher") && error.contains("--prompts"), "{error}");
        }
    }

    #[test]
    fn compact_three_argument_form_is_the_simple_mtp_recipe() {
        let cli = std::thread::Builder::new()
            .name("compact-simple-mtp-parser".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                Cli::try_parse_from([
                    "cortiq",
                    "compact",
                    "source.cmf",
                    "prompts.jsonl",
                    "task-model.cmf",
                    "--keep-ratio",
                    "0.75",
                    "--max-tokens",
                    "512",
                    "--samples",
                    "2",
                    "--seed",
                    "99",
                    "--split",
                    "100/0/0",
                    "--limit",
                    "24",
                    "--max-records",
                    "20",
                    "--max-record-tokens",
                    "256",
                    "--keep-truncated",
                    "--disable-mtp",
                ])
                .unwrap()
            })
            .unwrap()
            .join()
            .unwrap();
        match cli.command {
            Commands::Compact {
                model,
                prompts,
                task_output,
                output,
                execution_order,
                keep_ratio,
                max_tokens,
                samples,
                seed,
                split,
                limit,
                max_records,
                max_record_tokens,
                keep_truncated,
                disable_mtp,
                ..
            } => {
                assert_eq!(model, "source.cmf");
                assert_eq!(prompts.as_deref(), Some("prompts.jsonl"));
                assert_eq!(task_output.as_deref(), Some("task-model.cmf"));
                assert!(output.is_none());
                assert!(!execution_order);
                assert_eq!(keep_ratio, Some(0.75));
                assert_eq!(max_tokens, Some(512));
                assert_eq!(samples, Some(2));
                assert_eq!(seed, Some(99));
                assert_eq!(split.as_deref(), Some("100/0/0"));
                assert_eq!(limit, Some(24));
                assert_eq!(max_records, Some(20));
                assert_eq!(max_record_tokens, Some(256));
                assert!(keep_truncated);
                assert!(disable_mtp);
            }
            _ => panic!("wrong subcommand"),
        }
        assert_eq!(
            simple_mtp_compact_data_path("task-model.cmf"),
            "task-model.tdata"
        );
        assert_eq!(
            simple_mtp_compact_data_path("models/task.model.cmf"),
            "models/task.model.tdata"
        );
    }

    #[test]
    fn prompt_calibrated_mtp_preflight_rejects_bad_paths_before_generation() {
        preflight_mtp_compact_model_input("source.cmf").unwrap();
        let sidecar_model = preflight_mtp_compact_model_input("source.mtp.cmf")
            .unwrap_err()
            .to_string();
        assert!(sidecar_model.contains("main .cmf"), "{sidecar_model}");

        preflight_mtp_compact_output_suffix("candidate.cmf").unwrap();
        // The extended command preserves this external-MiMo compatibility
        // spelling; source discovery decides its topology later.
        preflight_mtp_compact_output_suffix("candidate.mtp.cmf").unwrap();
        let bad_suffix = preflight_mtp_compact_output_suffix("candidate")
            .unwrap_err()
            .to_string();
        assert!(bad_suffix.contains("new .cmf candidate"), "{bad_suffix}");

        preflight_simple_mtp_compact_output("candidate.cmf").unwrap();
        let simple_sidecar = preflight_simple_mtp_compact_output("candidate.mtp.cmf")
            .unwrap_err()
            .to_string();
        assert!(simple_sidecar.contains("runnable main CMF"), "{simple_sidecar}");
    }

    #[test]
    fn compact_legacy_rewrite_form_stays_available() {
        let cli = std::thread::Builder::new()
            .name("compact-legacy-parser".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                Cli::try_parse_from([
                    "cortiq",
                    "compact",
                    "source.cmf",
                    "--output",
                    "rewritten.cmf",
                    "--execution-order",
                ])
                .unwrap()
            })
            .unwrap()
            .join()
            .unwrap();
        match cli.command {
            Commands::Compact {
                prompts,
                task_output,
                output,
                execution_order,
                ..
            } => {
                assert!(prompts.is_none());
                assert!(task_output.is_none());
                assert_eq!(output.as_deref(), Some("rewritten.cmf"));
                assert!(execution_order);
            }
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn mtp_compact_missing_data_preflights_exact_tokenizer_contract() {
        use super::ensure_mtp_compact_tokenizer_contracts_match;

        ensure_mtp_compact_tokenizer_contracts_match(
            Some(b"shared-tokenizer"),
            b"shared-config",
            Some(b"shared-tokenizer"),
            b"shared-config",
        )
        .unwrap();
        let vocab_error = ensure_mtp_compact_tokenizer_contracts_match(
            Some(b"source-tokenizer"),
            b"shared-config",
            Some(b"teacher-tokenizer"),
            b"shared-config",
        )
        .unwrap_err()
        .to_string();
        assert!(vocab_error.contains("vocabularies differ"), "{vocab_error}");
        let config_error = ensure_mtp_compact_tokenizer_contracts_match(
            Some(b"shared-tokenizer"),
            b"source-config",
            Some(b"shared-tokenizer"),
            b"teacher-config",
        )
        .unwrap_err()
        .to_string();
        assert!(
            config_error.contains("configurations differ"),
            "{config_error}"
        );
        let missing_error = ensure_mtp_compact_tokenizer_contracts_match(
            None,
            b"shared-config",
            Some(b"shared-tokenizer"),
            b"shared-config",
        )
        .unwrap_err()
        .to_string();
        assert!(missing_error.contains("source model has no embedded"));
    }

    #[test]
    fn mtp_compact_parser_accepts_reusable_teacher_data_recipe() {
        let cli = std::thread::Builder::new()
            .name("mtp-compact-auto-parser".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                Cli::try_parse_from([
                    "cortiq",
                    "mtp-compact",
                    "student.cmf",
                    "--data",
                    "calibration.tdata",
                    "--teacher",
                    "teacher.cmf",
                    "--prompts",
                    "prompts.jsonl",
                    "--output",
                    "candidate.cmf",
                    "--teacher-max-tokens",
                    "384",
                    "--teacher-samples",
                    "2",
                    "--teacher-limit",
                    "7",
                ])
                .unwrap()
            })
            .unwrap()
            .join()
            .unwrap();
        match cli.command {
            Commands::MtpCompact {
                model,
                data,
                teacher,
                prompts,
                output,
                teacher_max_tokens,
                teacher_samples,
                teacher_limit,
                ..
            } => {
                assert_eq!(model, "student.cmf");
                assert_eq!(data, "calibration.tdata");
                assert_eq!(teacher.as_deref(), Some("teacher.cmf"));
                assert_eq!(prompts.as_deref(), Some("prompts.jsonl"));
                assert_eq!(output, "candidate.cmf");
                assert_eq!(teacher_max_tokens, 384);
                assert_eq!(teacher_samples, 2);
                assert_eq!(teacher_limit, Some(7));
            }
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn teacher_data_prompt_hashing_tracks_the_stream_given_to_the_parser() {
        let path = std::env::temp_dir().join(format!(
            "cmf-teacher-data-prompts-{}-{}.jsonl",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let bytes = b"{\"prompt\":\"first prompt\"}\n{\"prompt\":\"second prompt\"}\n";
        std::fs::write(&path, bytes).unwrap();

        let (loaded, digest) = parse_hashed_teacher_data_prompts(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(digest, knowledge::sha256_hex(bytes));
        assert_eq!(loaded.accepted(), 2);
        assert_eq!(loaded.prompts[0].id, teacher_data::derived_prompt_id("first prompt"));
        assert_eq!(loaded.prompts[1].id, teacher_data::derived_prompt_id("second prompt"));
    }
}
