//! Cortiq CLI — sparse task-routed model inference.

mod avout;
mod awnp;
mod convert;
mod gguf;
mod gptq;
mod imagepack;
mod moedefrag;
mod tube;
mod music;
mod npy;
mod requant;
mod sign;
mod skill;
mod ltxcmd;
mod ltxpack;
mod videopack;

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
use cortiq_engine::{CortiqRuntime, Pipeline, SamplerConfig};
use cortiq_server::{AppState, build_router};
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
    },
    /// Convert a Hugging Face checkpoint to .cmf — native Rust, no Python
    Convert {
        /// HF model: a local dir (config.json + *.safetensors + tokenizer.json)
        /// or a hub repo id like `Qwen/Qwen2.5-0.5B-Instruct` (downloaded)
        #[arg(long)]
        model: String,
        /// Quantization for 2-D weights: q8 | q8_2f | q4 | q4t | q4tp | q2tp | q1 | q1p | q1s | q1t | f16 | vbit
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
    },
    /// Rewrite a container tightly: reclaim dead directory/header tails
    /// left by append-only skill growth (spec §9). Streams from mmap.
    Compact {
        /// Source .cmf model
        model: String,
        /// Output .cmf path
        #[arg(long)]
        output: String,
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
        /// Target layout: q4tp (re-express q4 scales), q4tp-quantize (REAL quantization of a float container: 2-D weights → q4tp), or q2tp-draft (MTP draft expert inputs)
        #[arg(long, default_value = "q4tp")]
        quant: String,
        /// Rewrite tensors inside the source file itself — for disks that
        /// cannot hold the model twice. New payloads must fit their slots.
        #[arg(long, default_value_t = false)]
        in_place: bool,
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
    /// Import a GGUF model to .cmf — native Rust (F32/F16/BF16/Q4_0..Q6_K + K-quants; llama/mistral/qwen2/qwen3/qwen3.5 incl. the qwen35moe GDN+MoE hybrids, gemma-3, phi-3/4, DeepSeek-R1 distills)
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
    },
    /// 1-bit PTQ via the holographic transfer (GPTQ). Calibrates each
    /// linear's input Hessian on a corpus, then quantizes it to `q1s`:
    /// binarize with the rounding error folded into the kept weights
    /// through H⁻¹ (`Σ_PS·Σ_SS⁻¹`) so the layer OUTPUT survives, plus a
    /// two-field `|W|·RMS(x)` outlier mask. Norms/embeddings/lm_head (no
    /// captured Hessian) are copied verbatim. Pair with `skill bake` (FCD)
    /// on the tail layers to recover the last of the quality.
    QuantizeGptq {
        /// Input .cmf (f16 or q8 — higher precision ⇒ a better fold)
        input: String,
        /// Calibration corpus: a `.txt`, or a `.json` array of
        /// `[prompt, text]` pairs (the DTG-MA cache) whose texts concatenate
        #[arg(long)]
        calib: String,
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
        /// Skill to overlay (spec §9; a file with routable skills picks
        /// one automatically — `none` forces the backbone, an id pins):
        /// replacement tensors are read in
        /// place of backbone tensors
        #[arg(long)]
        skill: Option<String>,
        /// Greedy decoding (temperature 0) — gates and base models
        #[arg(long)]
        greedy: bool,
        /// Sampling temperature (default 0.7; ignored with --greedy)
        #[arg(long, conflicts_with = "greedy")]
        temperature: Option<f32>,
        /// Repetition penalty (default 1.1; 1.0 disables — better for code,
        /// where repeated identifiers are the point, not a defect)
        #[arg(long, conflicts_with = "greedy")]
        rep_penalty: Option<f32>,
        /// Nucleus cutoff (Qwen3.8: 0.95 thinking, 0.80 instruct)
        #[arg(long)]
        top_p: Option<f32>,
        /// Top-k cutoff (Qwen3.8: 20)
        #[arg(long)]
        top_k: Option<u32>,
        /// Min-p floor (Qwen3.8: 0.0)
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
    /// List `cortiq worker`s shouting on this network. A worker listening
    /// beyond loopback announces itself every two seconds; this waits and
    /// prints what answered, with the address to hand to `--peer`. The
    /// beacon carries identity and geometry, never the token.
    Peers {
        /// Seconds to listen (a beacon repeats every 2)
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
    Dequant {
        /// Path to .cmf model file
        model: String,
        /// Tensor name (or prefix with --all)
        #[arg(long)]
        name: String,
        /// Output file (or directory with --all)
        #[arg(long)]
        out: String,
        /// f32 (default) or bf16
        #[arg(long, default_value = "f32")]
        dtype: String,
        /// Treat --name as a prefix and dump every matching tensor
        #[arg(long, default_value_t = false)]
        all: bool,
    },
    /// Re-encode one 2-D tensor of a .cmf IN PLACE from raw f32 (row-major,
    /// rows×cols as in the directory) with the entry's OWN quantization —
    /// the payload keeps its slot; hashes are recomputed. Verify after.
    PatchTensor {
        /// Path to .cmf model file (modified in place unless --output)
        model: String,
        /// `name=path.f32` — repeatable; raw f32 rows×cols little-endian
        #[arg(long = "set")]
        sets: Vec<String>,
        /// Encode the patched tensors as this dtype instead of their own
        /// (f16 | q8_2f | q4tp | q2tp); needs --output when the payload
        /// grows
        #[arg(long)]
        dtype: Option<String>,
        /// Write a new file (all other tensors copied verbatim) instead
        /// of patching in place
        #[arg(long)]
        output: Option<String>,
    },
    /// Show model information
    Info {
        /// Path to .cmf model file
        model: String,
        /// List directory entries whose name starts with this prefix
        /// (name, dtype, shape, bytes), with a per-prefix total.
        #[arg(long)]
        tensors: Option<String>,
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
    /// Swarm skills (spec §9): bake a skill from a real donor checkpoint,
    /// list what a file carries
    Skill {
        #[command(subcommand)]
        cmd: SkillCmd,
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
    /// Generate an image from text (Lumina-Image 2.0). Takes a packed
    /// `.cmf` from `imagine-pack` — one mmap for text encoder, DiT and
    /// VAE — or a raw diffusers directory (tokenizer/ text_encoder/
    /// transformer/ vae/) for the exact f32 path. `CMF_GPU=1` runs the
    /// DiT on the device (Metal); output is P6 PPM
    Imagine {
        /// Model root directory
        model_dir: String,
        /// Text prompt
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 512)]
        height: usize,
        #[arg(long, default_value_t = 512)]
        width: usize,
        /// Denoising steps
        #[arg(long, default_value_t = 30)]
        steps: usize,
        /// Guidance scale (≤1 disables CFG and halves the work)
        #[arg(long, default_value_t = 4.0)]
        cfg: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Output image path (.ppm, P6)
        #[arg(long, default_value = "out.ppm")]
        out: String,
    },
    /// Pack a Lumina-Image 2.0 diffusers directory into ONE quantized
    /// .cmf (te.* + dit.* + vae.* + tokenizer) that `imagine` runs
    /// straight off the mmap
    ImaginePack {
        /// Diffusers root (tokenizer/ text_encoder/ transformer/ vae/)
        root: String,
        /// Projection codec: q4t | q8 (modulation/embeddings stay q8,
        /// VAE f16, norms f32)
        #[arg(long, default_value = "q4t")]
        quant: String,
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
        /// Skip donor tensors whose relative change vs the backbone is
        /// below this (0 = keep everything): neurons the fine-tune
        /// never touched are not stored, the skill shrinks to its real
        /// delta. Try 0.02–0.05; verify with --quality
        #[arg(long, default_value = "0.0")]
        min_delta: f32,
        /// Store the skill's tensors in a cheaper encoding than the
        /// backbone (q8 | q8_2f | q4 | q4t | f16 | vbit): half the
        /// bytes with q4 on a q8 backbone. Verify with --quality
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
        /// Output .cmf (the standalone specialist)
        #[arg(long)]
        output: String,
        /// Phase A steps (mask training)
        #[arg(long, default_value = "240")]
        steps_a: usize,
        /// Phase B steps (FCD polish)
        #[arg(long, default_value = "120")]
        steps_b: usize,
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
        /// Target sparsity (0.0–1.0): force at least this fraction of
        /// FFN neurons to be pruned. 0 = auto (denoising bottom)
        #[arg(long, default_value = "0.0")]
        target_sparsity: f64,
        /// L1 aggression multiplier: >1.0 = harder pruning push,
        /// <1.0 = softer (scales the L1 penalty schedule)
        #[arg(long, default_value = "1.0")]
        l1_aggression: f64,
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
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

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
            peer,
            peer_split,
            net_token,
            net_dtype,
            gpus,
        } => {
            let o1 = O1Flags {
                spec: o1,
                m: o1_m,
                w: o1_window,
                sink: o1_sink,
                rect: None,
            };
            cmd_serve(
                &model,
                &host,
                port,
                &task,
                compat_port,
                &o1,
                peer.as_deref(),
                peer_split,
                net_token.as_deref(),
                &net_dtype,
                gpus,
            )
            .await
        }
        Commands::Convert {
            model,
            quant,
            output,
            hf_token,
            mean_bits,
            resume,
            defrag,
            o1,
            o1_m,
            o1_window,
            o1_sink,
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
            convert::run_convert(
                &model,
                &quant,
                &output,
                hf_token.as_deref(),
                defrag.as_deref(),
                o1_hint,
                resume,
                progress_reporter("converting"),
            )?;
            println!("✓ wrote {output}");
            Ok(())
        }
        Commands::Compact { model, output } => moedefrag::cmd_compact(&model, &output),
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
        } => requant::cmd_requant(&model, output.as_deref(), &quant, in_place),
        Commands::TubeBake {
            model,
            plan,
            output,
        } => tube::cmd_tube_bake(&model, &plan, &output),
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
        } => {
            gguf::run_import_gguf(
                &gguf,
                &quant,
                &output,
                hf_token.as_deref(),
                progress_reporter("importing"),
            )?;
            println!("✓ wrote {output}");
            Ok(())
        }
        Commands::QuantizeGptq {
            input,
            calib,
            output,
            keep,
            tokens,
            lambda,
        } => {
            gptq::run_quantize_gptq(&input, &calib, &output, keep, tokens, lambda)?;
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
        } => cmd_patch_tensor(&model, &sets, dtype.as_deref(), output.as_deref()),
        Commands::Story { model } => cmd_story(&model),
        Commands::Diff { a, b } => cmd_diff(&a, &b),
        Commands::Imagine {
            model_dir,
            prompt,
            height,
            width,
            steps,
            cfg,
            seed,
            out,
        } => cmd_imagine(&model_dir, &prompt, height, width, steps, cfg, seed, &out),
        Commands::ImaginePack { root, quant, out } => {
            imagepack::cmd_imagine_pack(&root, &quant, &out)
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
        Commands::LtxEncode { model, prompt, oracle, out } => {
            ltxcmd::cmd_ltx_encode(ltxcmd::EncodeArgs {
                model: &model,
                prompt: &prompt,
                oracle: oracle.as_deref(),
                out: out.as_deref(),
            })
        }
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
        Commands::LtxAudio { model, latent, out, stats, oracle } => {
            ltxcmd::cmd_ltx_audio(ltxcmd::AudioArgs {
                model: &model,
                latent: &latent,
                out: &out,
                stats,
                oracle: oracle.as_deref(),
            })
        }
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
        Commands::Explain { model, prompt, top } => cmd_explain(&model, &prompt, top),
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
                output,
                steps_a,
                steps_b,
                fcd_layers,
                chunk,
                held,
                calib_chunks,
                target_sparsity,
                l1_aggression,
                ffn_align,
                uniform_inter,
            } => skill::run_skill_bake(
                &model,
                &files,
                &output,
                steps_a,
                steps_b,
                fcd_layers,
                chunk,
                held,
                calib_chunks,
                target_sparsity,
                l1_aggression,
                ffn_align,
                uniform_inter,
            ),
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
                        "  круг до устройства: пустой сабмит {empty:.2} мс · сабмит+диспатч+readback {one:.2} мс"
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
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
async fn cmd_serve(
    model_path: &str,
    host: &str,
    port: u16,
    default_task: &str,
    _compat_port: Option<u16>,
    o1: &O1Flags,
    peer: Option<&str>,
    peer_split: Option<usize>,
    net_token: Option<&str>,
    net_dtype: &str,
    replicas: Option<usize>,
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
            anyhow::bail!("--gpus {n}: реплики имеют смысл от 2 карт");
        }
        if have < n {
            anyhow::bail!(
                "--gpus {n}, а карт видно {have} (см. `cortiq gpu`) —                  просить больше реплик, чем устройств, нечестно"
            );
        }
        if peer.is_some() {
            anyhow::bail!("--gpus и --peer — разные режимы: реплики против сплита слоёв");
        }
        // Replicas need the model to fit ONE card. When it does not, the
        // honest answer is the layer split, not a refusal and not N
        // copies that thrash: say which mode you are in and why.
        let budget = cortiq_engine::gpu::vram_budget();
        let weights = model.primary_bytes().len() as u64;
        if budget != u64::MAX && weights > budget {
            slots = 1;
            devices.clear();
            split_devices = Some((0..n).collect());
            println!(
                "    Сплит по картам: модель {:.1} ГБ не влезает в бюджет одной ({:.1} ГБ) — \
                 слои режутся на {n} устройств (реплики требуют полной копии на карту)",
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
                "    Реплики: {n} карт × полная модель, {slots} слот(ов) —                  {per_gpu} на карту"
            );
        }
    }
    if peer.is_some() && slots != 1 {
        // One worker holds ONE KV session; more slots would interleave
        // sequences into it. Loud, not silent.
        println!("    note: --peer forces 1 slot (the worker holds one KV session)");
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
        o1.apply(&mut pipeline);
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

    // Create runtime
    let runtime = CortiqRuntime::new(model);
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
        None => println!(
            "PPL = {:.3} (exact attention)",
            (nll_ex / cnt.max(1) as f64).exp()
        ),
    }
    dump_moe_stats(pipeline)?;
    Ok(())
}

/// B-field of claim 12: expert-routing mass of this run →
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
    let ids = pipeline.tokenizer.encode(prompt);
    let tau = std::env::var("CMF_OOD_TAU").ok().and_then(|v| v.parse().ok()).unwrap_or(0.30);
    let r = cortiq_engine::router::route_full(&model, &mut pipeline, &ids, tau);
    if r.scores.is_empty() {
        println!("no routable skills in this container");
        return Ok(());
    }
    for s in &r.scores {
        if r.calibrated {
            println!("  {:<20} E = {:.4}  err = {:.3e}  p = {:.3}", s.id, s.error, s.raw_error, s.probability);
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
            if r.is_novel { "OOD (novel)" } else { "in-scope" }
        );
    } else {
        println!("winner: {}  ({}; uncalibrated file: E_min vs τ={tau})", r.scores[0].id, if r.is_novel { "OOD" } else { "in-scope" });
    }
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
        println!(
            "\n\x1b[1mRouting (recon-argmin, E=‖r−BBᵀr‖²/‖φ‖², lower = more coherent):\x1b[0m"
        );
        let emax = routes
            .iter()
            .map(|r| r.error)
            .fold(0.0f32, f32::max)
            .max(1e-6);
        for (i, r) in routes.iter().enumerate() {
            // Bar: shorter = lower E = more coherent (inverse scale).
            let fill = ((1.0 - r.error / emax) * 20.0).round() as usize;
            let bar = "█".repeat(fill);
            let mark = if i == 0 {
                "  \x1b[1m← chosen\x1b[0m"
            } else {
                ""
            };
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
        "Raw Born mass: \x1b[1m{verdict}\x1b[0m (ECE = {:.3})",
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
                  Born mass is itself the honest confidence."
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
            "The runtime will apply it to --confidence/--trace/explain (calibrated Born mass)."
        );
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

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
async fn cmd_run(
    model_path: &str,
    task: &str,
    prompt: Option<&str>,
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
    let routable = model.header.skills.iter().any(|s| s.selection.is_some());
    if skill.is_none() && blend.is_none() && !route_dynamic && routable && prompt.is_some() {
        skill = Some("auto".to_string());
    }
    if matches!(skill.as_deref(), Some("none") | Some("backbone")) {
        skill = None;
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
    o1.apply(&mut pipeline);
    // Same rule for the CLI: the vocabulary-wide softmax runs when its
    // output is going to be shown, not on every run.
    pipeline.set_confidence(confidence || trace);
    if let Some(n) = gpus {
        // In-process split: segment i runs pinned to card i and hands
        // the next one a hidden vector that never leaves this address
        // space — no worker process, no socket, no serialization.
        let have = cortiq_engine::gpu::device_count();
        if n < 2 {
            anyhow::bail!("--gpus {n}: нужно ≥ 2 карт (пин одной — CMF_GPU_ADAPTER)");
        }
        if have < n {
            anyhow::bail!("--gpus {n}, а карт видно {have} (см. `cortiq gpu`)");
        }
        let devs: Vec<usize> = (0..n).collect();
        pipeline
            .set_gpu_plan_at(Some(&devs), peer_split)
            .map_err(|e| anyhow::anyhow!("--gpus {n}: {e}"))?;
        if let Some(plan) = pipeline.gpu_plan() {
            let seg = plan
                .iter()
                .map(|(d, a, b)| format!("карта {d}: слои {a}..={b}"))
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

    let mut generate_and_print = |pipeline: &mut Pipeline,
                                  ids: &[u32]|
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
            None => pipeline.generate_from_ids(ids, max_tokens, mask.as_ref(), Some(cb)),
        };
        match gen_res {
            Ok(r) => {
                let secs = started.elapsed().as_secs_f64();
                // Confidence view: reprint token-by-token, coloured by the
                // model's Born mass on each emitted token.
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
                #[cfg(feature = "gpu")]
                cortiq_engine::dsv4::profile_report();
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

    if let Some(p) = prompt {
        println!("\nPrompt: {p}\n");
        let history = vec![("user".to_string(), p.to_string())];
        let ids = build_ids(&pipeline, &history, p);
        // CMF_PROMPT_DUMP=1: the rendered prompt as the model sees it
        // (template applied, decoded back to text) — for template audits.
        if std::env::var("CMF_PROMPT_DUMP").is_ok() {
            eprintln!("--- rendered prompt ({} tokens) ---\n{}\n--- end ---", ids.len(), pipeline.tokenizer.decode(&ids));
            let head: Vec<String> = ids.iter().take(12).map(|&t| format!("{t}:{:?}", pipeline.tokenizer.decode(&[t]))).collect();
            let tail: Vec<String> = ids.iter().rev().take(12).rev().map(|&t| format!("{t}:{:?}", pipeline.tokenizer.decode(&[t]))).collect();
            eprintln!("head {}\ntail {}", head.join(" "), tail.join(" "));
        }
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
            "--gpus {n}: пока поддержаны ровно 2 GPU (координатор + локальный воркер);              для N карт — цепочка `cortiq worker` + --peer"
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
        .map_err(|e| anyhow::anyhow!("не удалось запустить локальный воркер: {e}"))?;
    let mut w = LocalGpuWorker { child, addr, token };
    eprintln!(
        "gpus: координатор adapter {coord_pin}, воркер adapter {worker_pin} — {}",
        w.addr
    );
    // Readiness = the listener answers. The worker binds before serving,
    // so this bounds model-load time, not just process start.
    for _ in 0..1800 {
        if let Some(st) = w.child.try_wait()? {
            anyhow::bail!("локальный воркер умер при старте ({st}); его stderr выше");
        }
        if std::net::TcpStream::connect(&w.addr).is_ok() {
            return Ok(w);
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    anyhow::bail!("локальный воркер не поднялся за 180 с")
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

fn cmd_dequant(model_path: &str, name: &str, out: &str, dtype: &str, all: bool) -> anyhow::Result<()> {
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
        let mut vals = vec![0f32; e.n_elems()];
        cortiq_core::quant::dequant_tensor(e, bytes, &mut vals)
            .map_err(|er| anyhow::anyhow!("dequant {n}: {er}"))?;
        let mut buf: Vec<u8> = Vec::with_capacity(vals.len() * 4);
        match dtype {
            "f32" => {
                for v in &vals {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            "bf16" => {
                for v in &vals {
                    // round-to-nearest-even to bf16
                    let b = v.to_bits();
                    let lsb = (b >> 16) & 1;
                    let r = b.wrapping_add(0x7FFF + lsb) >> 16;
                    buf.extend_from_slice(&(r as u16).to_le_bytes());
                }
            }
            other => anyhow::bail!("dtype {other}: f32 or bf16"),
        }
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

fn cmd_patch_tensor(
    model_path: &str,
    sets: &[String],
    dtype: Option<&str>,
    output: Option<&str>,
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
    let forced: Option<convert::Quant> = match dtype {
        None => None,
        Some(q) => Some(convert::parse_quant(q)?),
    };
    // (directory index, dtype, payload)
    let mut patches: Vec<(usize, TensorDtype, Vec<u8>)> = Vec::new();
    let model = CmfModel::open_sharded(model_path)?;
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
            Some(q) => q,
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
        eprintln!("{name}: {rows}×{cols} {:?} → {dt:?} ({} bytes)", e.dtype, data.len());
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

async fn cmd_info(model_path: &str, tensors: Option<&str>) -> anyhow::Result<()> {
    let model = CmfModel::open_sharded(model_path)?;
    let arch = model.arch();

    if let Some(prefix) = tensors {
        let mut total = 0u64;
        let mut n = 0usize;
        for e in model.tensors.iter().filter(|e| e.name.starts_with(prefix)) {
            println!("{}\t{:?}\t{:?}\t{}", e.name, e.dtype, e.shape, e.nbytes);
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
    let linear = arch.num_layers - full - sliding - conv;
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
#[allow(clippy::too_many_arguments)]
fn cmd_imagine(
    model_dir: &str,
    prompt: &str,
    height: usize,
    width: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    out: &str,
) -> anyhow::Result<()> {
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
        println!(
            "I carry an MTP head ({} block) — I can speculatively speed up.",
            mtp.num_layers
        );
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
        println!("): I show Born mass as a measured property, not a raw estimate.");
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
        #[allow(unused_imports)]
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
    let pcie_gbs = cortiq_engine::gpu_wgpu::upload_bandwidth_probe(BLOCK, ROUNDS);
    if json {
        println!(
            "{{\"host_copy_gbs\": {host_gbs:.2}, \"upload_gbs\": {:.2}, \"storage_gbs\": {:.2}}}",
            pcie_gbs.unwrap_or(0.0),
            storage_gbs.unwrap_or(0.0)
        );
    } else {
        println!("копия в RAM: {host_gbs:.1} ГБ/с (потолок CPU-плеча)");
        match pcie_gbs {
            Some(v) => println!("заливка в VRAM: {v:.1} ГБ/с (потолок fetch-плеча)"),
            None => println!("заливка в VRAM: устройства нет"),
        }
        match storage_gbs {
            Some(v) => println!("чтение хранилища: {v:.1} ГБ/с (источник холодных байтов)"),
            None => println!("чтение хранилища: файл мал для замера"),
        }
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
        // Every id the tokenizer treats as end-of-sequence, suppressed:
        // the greedy loop then never sees "stop" (a one-pass penalized
        // argmax, so the core number stays a core number).
        let vocab = pipeline.tokenizer.vocab_size() as u32;
        let eos: Vec<u32> = (0..vocab)
            .filter(|&id| pipeline.tokenizer.is_eos(id))
            .collect();
        let mut cfg = pipeline.sampler_config.clone();
        cfg.suppress_tokens = eos;
        pipeline.set_sampler_config(cfg);
    }
    if core {
        pipeline.set_confidence(false);
        if !json {
            println!("  Core timing: sampler/confidence excluded (llama-bench contract)");
        }
    }
    o1.apply(&mut pipeline);
    if let Some(n) = gpus {
        let have = cortiq_engine::gpu::device_count();
        if n < 2 {
            anyhow::bail!("--gpus {n}: нужно ≥ 2 карт");
        }
        if have < n {
            anyhow::bail!("--gpus {n}, а карт видно {have}");
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
            anyhow::bail!("--peer-split {split}: модель держит {nl} слоёв");
        }
        let dtype = match net_dtype {
            "f32" => cortiq_net::WireDtype::F32,
            "f16" => cortiq_net::WireDtype::F16,
            other => anyhow::bail!("--net-dtype {other}: f32 или f16"),
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
                "BENCH INVALID: локальный сегмент отказал графу на {graph_miss} токенах —                  это CPU-число, не GPU (CMF_GPU_DEBUG=1 покажет причину)"
            );
        }
        if upload_delta > 64 * 1024 * 1024 {
            anyhow::bail!(
                "BENCH INVALID: {upload_delta} байт весов доехало в steady-окне —                  вытеснение/перезаливка, поднимите CMF_GPU_VRAM_MB"
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
    type Stamp = (std::time::Instant, u64, usize, GpuCounterSnapshot);
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
    let t1 = std::time::Instant::now();
    let result = pipeline
        .generate_from_ids(&prompt_ids, tokens as usize, mask.as_ref(), Some(cb))
        .map_err(|e| anyhow::anyhow!(e))?;
    let total_s = t1.elapsed().as_secs_f64();

    if !json {
        println!(
            "  Prompt:  {} tokens | prefill {:.1} tok/s",
            prompt_ids.len(),
            prompt_ids.len() as f64 / prefill_s.max(1e-9)
        );
    }
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
            "tensor_count": tensor_count,
            "tensor_dtypes": tensor_dtypes,
            "tensor_q1": tensor_q1,
            "tensor_q8_2f": tensor_q8_2f,
            "tensor_f16": tensor_f16,
            "task": task,
            "ctx": ctx,
            "o1": pipeline.o1_active(),
            "threads_env": std::env::var("CMF_THREADS").ok(),
            "prompt_tokens": prompt_ids.len(),
            "prefill_tok_s": prompt_ids.len() as f64 / prefill_s.max(1e-9),
            "tokens_generated": result.tokens_generated,
            "metal_submits_per_token": metal_submits_per_token,
            "wgpu_submits_per_token": wgpu_submits_per_token,
            "wgpu_passes_per_token": wgpu_passes_per_token,
            "wgpu_upload_ms": wgpu_uploads().0,
            "wgpu_upload_mb": wgpu_uploads().1,
            "wgpu_steady_upload_ms_per_token": wgpu_steady_upload_ms_per_token,
            "wgpu_steady_upload_mb_per_token": wgpu_steady_upload_mb_per_token,
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
    // `bench` is where the dsv4 breakdown is actually wanted — `run` had it
    // and this did not, so every timing question needed a second command
    // measuring a different workload.
    #[cfg(feature = "gpu")]
    cortiq_engine::dsv4::profile_report();
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
