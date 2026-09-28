//! `cortiq-embryo` — the birth/growth trainer CLI (native Rust + Metal).

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "cortiq-embryo",
    about = "Cortiq Embryo trainer (native Rust + Metal)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Sub,
}

#[derive(Subcommand)]
enum Sub {
    /// Measure real TFLOPS of the training GEMM at Embryo shapes (docs §4.1).
    Bench {
        #[arg(long, default_value_t = 10)]
        reps: usize,
        #[arg(long)]
        no_verify: bool,
    },
    /// Print the Embryo-0 genome configuration and parameter counts.
    Config,
    /// Time full training steps of Embryo-0 (fwd + bwd + AdamW) on random tokens.
    StepBench {
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 1024)]
        seq: usize,
        #[arg(long, default_value_t = 5)]
        steps: usize,
        /// use the tiny genome instead of Embryo-0
        #[arg(long)]
        tiny: bool,
        /// JSON overrides merged into the genome config, e.g.
        /// '{"mixer":"gdn","anchor_layers":[3,7],"anchor_window":128,"anchor_sink":4}'
        #[arg(long)]
        cfg_json: Option<String>,
    },
    /// Turn a text file into a byte-level token shard (vocab 256) — smoke corpus.
    BytesShard { input: PathBuf, output: PathBuf },
    /// Train our byte-level BPE tokenizer (HF tokenizer.json, runtime-compatible).
    TrainTokenizer {
        #[arg(long, required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 32768)]
        vocab: usize,
        /// bytes of text to sample for the merge statistics
        #[arg(long, default_value_t = 400 << 20)]
        sample_mb_bytes: usize,
    },
    /// Encode corpus files (jsonl.gz / txt / parquet with --features data) into a u16 shard.
    Shard {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = usize::MAX)]
        max_tokens: usize,
    },
    /// Download corpus files (curl, resumable) into a directory.
    Fetch {
        #[arg(long)]
        dir: PathBuf,
        #[arg(required = true, num_args = 1..)]
        urls: Vec<String>,
    },
    /// Prepare fixed-width response-only OASST1 SFT shards (native Rust).
    SftPrepare {
        /// Official flat OASST1 `*.messages.jsonl.gz` export.
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        train_out: PathBuf,
        #[arg(long)]
        dev_out: PathBuf,
        #[arg(long)]
        final_out: PathBuf,
        #[arg(long)]
        manifest_out: PathBuf,
        #[arg(long, default_value_t = 512)]
        seq: usize,
        /// `oasst` (official flat export) or `messages` (plain
        /// `{"messages":[{role,content}..]}` JSONL, one record per assistant turn)
        #[arg(long, default_value = "oasst")]
        format: String,
        /// messages format: JSON field holding the subject/tree group used for
        /// the disjoint train/dev/final split (default: src_title, latin, plant,
        /// group, tree_id, title; inferred from Latin binomials when absent)
        #[arg(long)]
        group_field: Option<String>,
    },
    /// Prepare a fresh raw-LM replay shard from bounded native token prefixes.
    RawReplay {
        #[arg(long, required = true, num_args = 1..)]
        input: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 4_000_000)]
        max_tokens: usize,
    },
    /// Export a checkpoint (+ tokenizer.json) into a runtime-loadable .cmf.
    Export {
        #[arg(long)]
        ckpt: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Storage profile: exact f32, or f16 matrices with exact routing metadata/norms.
        #[arg(long, default_value = "f32", value_parser = ["f32", "f16"])]
        dtype: String,
        /// Stamp the frozen genome (bit GENOME, format v2): `header.genome`
        /// {id, generation 0, status, encoding, trunk_hash, master_trunk_hash}
        /// + lineage [birth]. v2 skills bind to it. Requires --genome-status.
        #[arg(long, requires = "genome_status")]
        genome_id: Option<String>,
        /// Genome status at birth (with --genome-id).
        #[arg(long, requires = "genome_id", value_parser = ["pre_chat", "candidate", "sealed"])]
        genome_status: Option<String>,
    },
    /// Dump held-out documents of a corpus file as text (for `cortiq ppl`).
    SampleText {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 0)]
        skip: usize,
        #[arg(long, default_value_t = 100)]
        docs: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Bake a skill from a genome checkpoint and append it to a copy of a .cmf (P2/P15).
    ///
    /// v2 (`--sft-train …`, format v2): an `ffn_replace` record over the
    /// FROZEN genome of `--base` (bit GENOME; `--ckpt` must be its exact
    /// checkpoint), response-only SFT (+ optional raw-LM rows), selected on
    /// the group dev answer NLL, routed by router v2 (backbone-gated,
    /// canonical span-mean φ), tail-appended to `--out` = copy of `--base`
    /// with status `quarantine`; prints one JSON line. Legacy (`--corpus`):
    /// flat LM corpus, v1 record, bases without a genome only.
    SkillBake {
        /// genome checkpoint (frozen)
        #[arg(long)]
        ckpt: PathBuf,
        /// tokenizer.json (ours). Legacy: required. v2: optional — the base's
        /// VOCAB section is used and this file must equal it.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// legacy task corpus files (txt / jsonl.gz / parquet)
        #[arg(long, num_args = 1..)]
        corpus: Vec<PathBuf>,
        /// base .cmf (v2: an exported genome — `export --genome-id`)
        #[arg(long)]
        base: PathBuf,
        /// output .cmf: a NEW file, never --base (the base is copied, then appended)
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        id: String,
        /// layers whose shared FFN the skill specialises (default: last two)
        #[arg(long, num_args = 1.., value_delimiter = ',')]
        layers: Option<Vec<usize>>,
        #[arg(long, default_value_t = 240)]
        steps_a: usize,
        #[arg(long, default_value_t = 120)]
        steps_b: usize,
        #[arg(long, default_value_t = 3e-2)]
        lr_a: f32,
        #[arg(long, default_value_t = 5e-5)]
        lr_b: f32,
        /// final L1 pressure on σ(m) (ramps from 0)
        #[arg(long, default_value_t = 2e-4)]
        l1: f32,
        #[arg(long, default_value_t = 0.5)]
        tau: f32,
        #[arg(long, default_value_t = 30)]
        eval_every: usize,
        #[arg(long, default_value_t = 4)]
        batch: usize,
        /// legacy window length (default 512); v2 uses the SFT shards' length
        /// and refuses a different value
        #[arg(long)]
        seq: Option<usize>,
        /// φ = hidden AFTER this layer (v2: must be < min(--layers); default
        /// min(2/3 depth, min(layers) − 1))
        #[arg(long)]
        phi_layer: Option<usize>,
        /// legacy: positions averaged for the routing φ
        #[arg(long, default_value_t = 48)]
        phi_len: usize,
        /// rank of the skill and backbone φ descriptors
        #[arg(long, default_value_t = 16)]
        rank: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// v2: response-only SFT training shard (`sft-prepare` output)
        #[arg(long)]
        sft_train: Option<PathBuf>,
        /// v2: group-disjoint dev SFT shard — the held-out selection set
        #[arg(long)]
        sft_dev: Option<PathBuf>,
        /// v2: terminal final SFT shard (reported, never used for selection)
        #[arg(long)]
        sft_final: Option<PathBuf>,
        /// v2: raw-LM u16 shard mixed into training batches (rows)
        #[arg(long)]
        lm_train: Option<PathBuf>,
        /// v2: raw-LM u16 dev shard (LM NLL reported next to the answer NLL)
        #[arg(long)]
        lm_dev: Option<PathBuf>,
        /// v2: fraction of each batch's rows drawn from --lm-train
        #[arg(long, default_value_t = 0.5)]
        lm_frac: f32,
        /// v2: dev evaluation size — 0 (default) = every record of --sft-dev;
        /// N > 0 = N·batch records spread evenly over the whole shard (never a
        /// prefix). Raw-LM dev: N batches (8 when 0)
        #[arg(long, default_value_t = 0)]
        dev_batches: usize,
        /// v2: the `sft-prepare --messages` manifest of the shards (verifies
        /// the group split; recorded in the record's quality)
        #[arg(long)]
        sft_manifest: Option<PathBuf>,
        /// v2: JSONL with "prompt" — the skill's questions (skill φ class)
        #[arg(long)]
        phi_prompts: Option<PathBuf>,
        /// v2: JSONL with "prompt" — general questions (backbone φ class);
        /// needed for the first record over a genome and with --refit-base
        #[arg(long)]
        general_prompts: Option<PathBuf>,
        /// v2: refit the backbone descriptor from --general-prompts although
        /// --base already routes (every active skill is then re-gated)
        #[arg(long)]
        refit_base: bool,
        /// v2: prompts per φ class (deterministic sample)
        #[arg(long, default_value_t = 4000)]
        phi_max: usize,
        /// v2: prompts per φ probe batch
        #[arg(long, default_value_t = 8)]
        phi_batch: usize,
        /// v2: longest canonical φ input in tokens (longer prompts skipped)
        #[arg(long, default_value_t = 512)]
        phi_max_len: usize,
        /// v2: a skill must beat the backbone by this much in unit error
        /// (default: the margin of --base's router, 0 for the first record)
        #[arg(long)]
        route_margin: Option<f32>,
        /// v2: target in-scope false-positive rate of the novelty flag
        #[arg(long, default_value_t = 0.02)]
        target_fpr: f32,
        /// no effect, kept for old scripts: v2 always trains and evaluates
        /// with dropless expert capacity (the runtime's operator)
        #[arg(long, hide = true)]
        dropless: bool,
    },
    /// Lane-only final-hidden teacher-residual pretraining (D22/D23).
    ResidualPretrain {
        /// token shards, `path[:weight]`, repeatable — same continuation mix as Birth
        #[arg(long, required = true, num_args = 1..)]
        shard: Vec<String>,
        /// held-out shard (defaults to each training shard's 0.5% tail)
        #[arg(long)]
        val: Option<PathBuf>,
        /// immutable pre-lane checkpoint (the lane is appended by name)
        #[arg(long)]
        resume: PathBuf,
        /// frozen all-softmax teacher checkpoint
        #[arg(long)]
        teacher: PathBuf,
        /// final checkpoint path; staged checkpoints append `-step-N.ckpt`
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 1024)]
        seq: usize,
        /// absolute optimizer step at which to stop (resume step + 100)
        #[arg(long, default_value_t = 61100)]
        steps: usize,
        #[arg(long, default_value_t = 6e-4)]
        lr: f32,
        #[arg(long, default_value_t = 0.1)]
        wd: f32,
        #[arg(long, default_value_t = 1.0)]
        clip: f32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Host-only layer-4 GQA donor graft into an anchor_every=4 candidate,
    /// followed by the fixed eight-batch immediate validation gate.
    AnchorGraft {
        /// immutable one-anchor student checkpoint
        #[arg(long)]
        student: PathBuf,
        /// immutable all-softmax twin checkpoint
        #[arg(long)]
        donor: PathBuf,
        /// candidate checkpoint to create
        #[arg(long)]
        out: PathBuf,
        /// held-out shard; if omitted, --shard tails provide validation
        #[arg(long)]
        val: Option<PathBuf>,
        /// training shards used only to derive the same deterministic tail
        #[arg(long, num_args = 1..)]
        shard: Vec<String>,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 1024)]
        seq: usize,
    },
    /// Phase-2 layer-only CE adaptation of the zero-output donor GQA lane.
    GqaLane {
        #[arg(long)]
        resume: PathBuf,
        #[arg(long)]
        donor: PathBuf,
        #[arg(long, required = true, num_args = 1..)]
        shard: Vec<String>,
        #[arg(long)]
        val: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 1024)]
        seq: usize,
        #[arg(long, default_value_t = 61100)]
        steps: usize,
        #[arg(long, default_value_t = 6e-4)]
        lr: f32,
        #[arg(long, default_value_t = 0.1)]
        wd: f32,
        #[arg(long, default_value_t = 1.0)]
        clip: f32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Sleep daemon: bake skills from the OOD buffer during idle time, gate, commit/rollback, journal.
    Sleep {
        #[arg(long)]
        ckpt: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        /// the served .cmf (skills are appended atomically)
        #[arg(long)]
        cmf: PathBuf,
        /// CMF_OOD_DIR of the server (buffer.jsonl, last_request, journal.jsonl)
        #[arg(long)]
        ood_dir: PathBuf,
        #[arg(long, default_value_t = 5.0)]
        idle_min: f64,
        #[arg(long, default_value_t = 4000)]
        min_tokens: usize,
        /// required held-out improvement (fraction of loss) to keep a skill
        #[arg(long, default_value_t = 0.02)]
        gate: f32,
        /// requant to q4tp when ppl grows by at most this fraction (0 = off)
        #[arg(long, default_value_t = 0.0)]
        requant_gate: f32,
        #[arg(long)]
        held_out: Option<PathBuf>,
        #[arg(long, default_value = "cortiq")]
        cortiq_bin: String,
        #[arg(long, num_args = 1.., value_delimiter = ',')]
        layers: Option<Vec<usize>>,
        #[arg(long, default_value_t = 240)]
        steps_a: usize,
        #[arg(long, default_value_t = 120)]
        steps_b: usize,
        #[arg(long, default_value_t = 4)]
        batch: usize,
        #[arg(long, default_value_t = 512)]
        seq: usize,
        /// one cycle then exit
        #[arg(long)]
        once: bool,
        /// ignore idle/min-tokens (demo)
        #[arg(long)]
        force: bool,
        #[arg(long, default_value_t = 30)]
        poll_secs: u64,
        /// try growth (new experts per layer) after N consecutive rejected nights (0 = off)
        #[arg(long, default_value_t = 3)]
        grow_after: usize,
    },
    /// Growth as records: K new experts per grown layer — --source-mode hottest: copies of the K trunk experts hottest ON THE GROWTH CORPUS, descriptors from the tokens they win; --source-mode novel: K-means clusters of the corpus tokens NOVEL for the trunk (best trunk reconstruction error above τ = the --novel-quantile of that error on --general), descriptors from the clusters, weights of the expert hottest on each cluster — only the new experts train on the corpus (bias frozen: 0 or the source's, --bias-mode; descriptors adapting or frozen, --desc-mode), gate on held-out, shells by --shell-mode (won-quantile | general-target), routing shift / coverage / novel coverage with the runtime formula, and the `expert_append` record appended to a copy of the genome file (--record-out). Prints one JSON summary line last.
    Grow {
        /// the genome checkpoint (exactly the checkpoint of --base when a record is written)
        #[arg(long)]
        ckpt: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        /// growth corpus files (jsonl(.gz) with "text", or .txt)
        #[arg(long, required = true, num_args = 1..)]
        corpus: Vec<PathBuf>,
        /// group held-out files (jsonl(.gz)/txt) instead of the corpus' 10 % tail
        #[arg(long, num_args = 1..)]
        held: Vec<PathBuf>,
        /// general token shard (.u16, e.g. dev-mix-v1) for routing_shift
        #[arg(long)]
        general: Option<PathBuf>,
        /// token budget of the trace passes (source wins, descriptor inits, shells): evenly spaced [seq] windows of the corpus, 0 = the whole corpus
        #[arg(long, default_value_t = 1_000_000)]
        trace_tokens: usize,
        /// document budget of the held-out coverage trace: evenly spaced documents, 0 = all
        #[arg(long, default_value_t = 1000)]
        trace_docs: usize,
        /// new experts per grown layer (K ≥ 1)
        #[arg(long, default_value_t = 1)]
        experts: usize,
        /// the grown layers, e.g. `4,5,6,7` (default: every MoE layer)
        #[arg(long, num_args = 1.., value_delimiter = ',')]
        layers: Option<Vec<usize>>,
        /// what the shells are calibrated on: `won-quantile` = the --shell-quantile of the errors of ALL the growth tokens an expert wins (as wide as the source's cluster); `general-target` = on --general, an expert capturing more than --shell-target-shift of a layer's general tokens gets the (target/share)-quantile of those captured errors, so its shell admits at most the target (needs --general)
        #[arg(long, default_value = "general-target", value_parser = ["won-quantile", "general-target"])]
        shell_mode: String,
        /// quantile of the won reconstruction errors that becomes each grown expert's shell (won-quantile, and general-target under the target)
        #[arg(long, default_value_t = 0.99)]
        shell_quantile: f32,
        /// general-target: the fraction of a layer's general tokens a grown expert's shell may admit
        #[arg(long, default_value_t = 0.005)]
        shell_target_shift: f32,
        /// the grown expert's balancing bias, frozen for the whole training and written into the record: `zero` (the copy out-scores its source against the trunk's negative biases) | `source` (the SOURCE trunk expert's bias: the copy ties with its source at insertion)
        #[arg(long, default_value = "zero", value_parser = ["zero", "source"])]
        bias_mode: String,
        /// where the K copies and their descriptors come from: `hottest` = the K trunk experts hottest on the growth corpus, descriptors (μ mean / U top-k eigenvectors) from the tokens they win; `novel` = the corpus tokens novel for the trunk (min trunk reconstruction error > τ, per layer the --novel-quantile of that error on --general) clustered by K-means, one descriptor per cluster and the weights of the trunk expert hottest on that cluster (needs --general; refuses a layer with < 64·K novel tokens)
        #[arg(long, default_value = "hottest", value_parser = ["hottest", "novel"])]
        source_mode: String,
        /// novel: τ per grown layer = this quantile of the general shard's min trunk reconstruction error (at most 1−q of the general tokens are novel for the trunk); the novelty pass runs and is reported whenever --general is given
        #[arg(long, default_value_t = 0.995)]
        novel_quantile: f32,
        /// whether the grown experts' descriptors (μ EMA) move while training: `adapt` | `frozen` (μ / U stay at the initialisation, the record's descriptor is exactly the cluster's); default: adapt for --source-mode hottest, frozen for novel
        #[arg(long, value_parser = ["adapt", "frozen"])]
        desc_mode: Option<String>,
        /// write the `expert_append` record to this NEW file (a copy of --base + tail append; never --base itself)
        #[arg(long)]
        record_out: Option<PathBuf>,
        /// the genome file F0 the record is appended to (with --record-out)
        #[arg(long)]
        base: Option<PathBuf>,
        /// record id (with --record-out)
        #[arg(long)]
        id: Option<String>,
        /// save the TRAINED grown checkpoint (E0+K experts in every layer; the input of `reshell`)
        #[arg(long)]
        out_ckpt: Option<PathBuf>,
        /// export the FULL grown genome as .cmf — NOT a genome record: it changes arch.moe.num_experts (layers outside --layers carry inert slots)
        #[arg(long)]
        export: Option<PathBuf>,
        #[arg(long, default_value_t = 300)]
        steps: usize,
        #[arg(long, default_value_t = 3e-4)]
        lr: f32,
        #[arg(long, default_value_t = 4)]
        batch: usize,
        #[arg(long, default_value_t = 512)]
        seq: usize,
        /// required held-out improvement over the GENOME (fraction of loss) to keep the growth (negative = never reject)
        #[arg(long, default_value_t = 0.005)]
        gate: f32,
        /// held-out budget: at most this many [batch, seq] batches of consecutive windows (0 = 16)
        #[arg(long, default_value_t = 16)]
        held_batches: usize,
        #[arg(long, default_value_t = 1e-3)]
        noise: f32,
        /// legacy descriptor shift (outside span(U)) for an expert whose trunk source wins no growth token; a record growth initialises descriptors from the corpus
        #[arg(long, default_value_t = 0.1)]
        shift: f32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// ONLY the post-training part of `grow` from a TRAINED grown checkpoint (`grow --out-ckpt`): the same traces on the corpus / held-out / general shard, shells by --shell-mode, coverage, routing shift, the `expert_append` record (--record-out) and the same one JSON summary line (with "reshell": true). The trunk sources are recomputed on the pre-growth instance (the same corpus wins); the bias mode is inferred from the checkpoint unless given.
    Reshell {
        /// the trained grown checkpoint (E0+K experts in every layer)
        #[arg(long)]
        ckpt: PathBuf,
        /// the pre-growth checkpoint of --base (F0's): required for an f16 genome, whose master trunk the grown checkpoint (served, rounded) cannot reproduce; an f32 genome recovers it from --ckpt
        #[arg(long)]
        genome_ckpt: Option<PathBuf>,
        #[arg(long)]
        tokenizer: PathBuf,
        /// growth corpus files (jsonl(.gz) with "text", or .txt) — the same as the growth's
        #[arg(long, required = true, num_args = 1..)]
        corpus: Vec<PathBuf>,
        /// group held-out files (jsonl(.gz)/txt) instead of the corpus' 10 % tail
        #[arg(long, num_args = 1..)]
        held: Vec<PathBuf>,
        /// general token shard (.u16, e.g. dev-mix-v1) for routing_shift (and the general-target shells)
        #[arg(long)]
        general: Option<PathBuf>,
        /// token budget of the trace passes (source wins, shells): evenly spaced [seq] windows of the corpus, 0 = the whole corpus
        #[arg(long, default_value_t = 1_000_000)]
        trace_tokens: usize,
        /// document budget of the held-out coverage trace: evenly spaced documents, 0 = all
        #[arg(long, default_value_t = 1000)]
        trace_docs: usize,
        /// the grown layers, e.g. `4,5,6,7` (default: the layers whose grown slots are live in --ckpt; must equal them when given)
        #[arg(long, num_args = 1.., value_delimiter = ',')]
        layers: Option<Vec<usize>>,
        /// what the shells are calibrated on (see `grow --help`)
        #[arg(long, default_value = "general-target", value_parser = ["won-quantile", "general-target"])]
        shell_mode: String,
        #[arg(long, default_value_t = 0.99)]
        shell_quantile: f32,
        #[arg(long, default_value_t = 0.005)]
        shell_target_shift: f32,
        /// the grown bias the record carries: `zero` | `source` (default: inferred from --ckpt and verified against the sources' biases)
        #[arg(long, value_parser = ["zero", "source"])]
        bias_mode: Option<String>,
        /// the growth's --source-mode (the sources / descriptor inits are recomputed by it; `novel` needs --general)
        #[arg(long, default_value = "hottest", value_parser = ["hottest", "novel"])]
        source_mode: String,
        /// the growth's --novel-quantile
        #[arg(long, default_value_t = 0.995)]
        novel_quantile: f32,
        /// the growth's --desc-mode: `adapt` | `frozen` (default: inferred — the grown μ / U of --ckpt equal to the recomputed initialisation bit for bit → frozen, else adapt; `frozen` given but differing → refused)
        #[arg(long, value_parser = ["adapt", "frozen"])]
        desc_mode: Option<String>,
        /// the growth's --seed (the novel clustering and the descriptor inits are seeded by it)
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// write the `expert_append` record to this NEW file (a copy of --base + tail append)
        #[arg(long)]
        record_out: Option<PathBuf>,
        /// the genome file F0 (E0, trunk binding, the record's base)
        #[arg(long)]
        base: PathBuf,
        /// record id (with --record-out)
        #[arg(long)]
        id: Option<String>,
        #[arg(long, default_value_t = 4)]
        batch: usize,
        #[arg(long, default_value_t = 512)]
        seq: usize,
    },
    /// Train K MTP heads (t+2, t+3, …) on the frozen trunk and append them to a .cmf.
    MtpTrain {
        #[arg(long)]
        ckpt: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, required = true, num_args = 1..)]
        corpus: Vec<PathBuf>,
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 2)]
        heads: usize,
        #[arg(long, default_value_t = 400)]
        steps: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f32,
        #[arg(long, default_value_t = 4)]
        batch: usize,
        #[arg(long, default_value_t = 512)]
        seq: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Native Vulkan training continuation. This command is available only
    /// in the opt-in Linux `vulkan` build; it keeps train and validation
    /// shards separate and writes the unchanged plain checkpoint format.
    #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
    VulkanTrain {
        #[arg(long)]
        ckpt: PathBuf,
        #[arg(long)]
        shard: PathBuf,
        /// Disjoint held-out shard (required; no validation leakage).
        #[arg(long)]
        val: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1)]
        batch: usize,
        #[arg(long, default_value_t = 64)]
        seq: usize,
        #[arg(long, default_value_t = 1)]
        steps: usize,
        /// Number of deterministic validation windows to average before and after training.
        /// The default keeps the original one-window smoke behavior; quality runs should
        /// pass a larger value on a frozen, disjoint shard.
        #[arg(long, default_value_t = 1)]
        val_batches: usize,
        #[arg(long, default_value_t = 1e-4)]
        lr: f32,
        #[arg(long, default_value_t = 0.1)]
        wd: f32,
        #[arg(long, default_value_t = 1.0)]
        clip: f32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Allocate enough expert capacity for every routed row. This is an
        /// evaluation/diagnostic profile; normal training retains capacity-2
        /// routing and may drop overflow rows.
        #[arg(long)]
        dropless: bool,
        /// Continue a LEGACY (full-causal anchor) checkpoint under the
        /// bounded anchor `swa_sink_v1`: exact window of this many keys.
        /// The arena is extended with the trained sink vectors (append-only:
        /// every legacy parameter and AdamW moment is copied bit-for-bit,
        /// as `birth --resume --anchor-window` does on macOS).
        #[arg(long)]
        anchor_window: Option<usize>,
        /// Trained NoPE sink vectors per KV head (with --anchor-window).
        #[arg(long)]
        anchor_sink: Option<usize>,
        /// Stochastic training windows, e.g. `64,128` (SWAX); empty = fixed.
        #[arg(long, value_delimiter = ',')]
        anchor_train_windows: Option<Vec<usize>>,
        /// carry the recurrent state / conv history / anchor keys across
        /// consecutive windows of one document stream (plan S6b)
        #[arg(long)]
        carry: bool,
        #[arg(long, default_value_t = 16)]
        carry_reset_every: usize,
        #[arg(long)]
        carry_eot: Option<u32>,
        /// print the held-out loss every N steps (0 = only before/after)
        #[arg(long, default_value_t = 0)]
        eval_every: usize,
        /// with --carry: print the fresh/carried window losses and the
        /// per-layer state rms/max every N steps (0 = off; summary always)
        #[arg(long, default_value_t = 10)]
        stats_every: usize,
    },
    /// Native Vulkan response-only SFT with a fixed 20% raw-LM replay mix.
    #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
    VulkanSft {
        #[arg(long)]
        ckpt: PathBuf,
        #[arg(long)]
        shard: PathBuf,
        #[arg(long)]
        val: PathBuf,
        /// Frozen raw-LM replay shard (training only).
        #[arg(long)]
        replay: PathBuf,
        /// Frozen raw-LM validation shard (evaluation only).
        #[arg(long)]
        raw_val: PathBuf,
        /// Frozen terminal response-only final shard.  It is evaluated only
        /// after training and is never used by the dev selection gate.
        #[arg(long)]
        final_val: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 10)]
        batch: usize,
        #[arg(long, default_value_t = 512)]
        seq: usize,
        /// Absolute optimizer step at which to stop. The initial run is 100
        /// steps; an extension beyond that must pass --allow-extended.
        #[arg(long, default_value_t = 61100)]
        steps: usize,
        #[arg(long)]
        allow_extended: bool,
        #[arg(long, default_value_t = 8)]
        val_batches: usize,
        #[arg(long, default_value_t = 8)]
        raw_val_batches: usize,
        #[arg(long, default_value_t = 2e-5)]
        lr: f32,
        #[arg(long, default_value_t = 0.1)]
        wd: f32,
        #[arg(long, default_value_t = 1.0)]
        clip: f32,
        #[arg(long, default_value_t = 17)]
        seed: u64,
        /// Number of already-consumed mixed batches in the deterministic
        /// sampler stream.  Required explicitly when extending a prior SFT
        /// stage so a resume cannot silently replay its prefix.
        #[arg(long, default_value_t = 0)]
        sampler_skip: usize,
        #[arg(long)]
        dropless: bool,
    },
    /// Synthetic recall probe (MQAR/NIAH-lite) through the trainer forward:
    /// K key→value pairs, real-text filler, one key repeated at distance D.
    ProbeRecall {
        #[arg(long)]
        ckpt: PathBuf,
        /// filler text (u16 shard)
        #[arg(long)]
        shard: PathBuf,
        #[arg(long, default_value_t = 4)]
        pairs: usize,
        /// query distances, e.g. `256,1024,4096,16384`
        #[arg(long, value_delimiter = ',', default_value = "256,1024,4096,16384")]
        dists: Vec<usize>,
        #[arg(long, default_value_t = 32)]
        trials: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// largest window T to allocate (the trainer materialises [T, T]
        /// anchor scores; 16448 needs ~17 GB at 8 anchor heads)
        #[arg(long, default_value_t = 16448)]
        max_seq: usize,
    },
    /// Depth profile: no-grad forward over one carried stream of N windows
    /// (mean NLL per window index, per-layer state rms/max after each).
    CarryProfile {
        #[arg(long)]
        ckpt: PathBuf,
        #[arg(long)]
        shard: PathBuf,
        #[arg(long, default_value_t = 16)]
        windows: usize,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 1024)]
        seq: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// also run the first N windows of rows 0-1 as one long forward
        /// (no carry) and print the per-window NLL next to the carried ones
        #[arg(long, default_value_t = 0)]
        one_pass: usize,
    },
    /// Birth: train from scratch (or resume) on a token shard.
    Birth {
        /// token shards (u16 LE), `path[:weight]`, repeatable — mixed by weight
        #[arg(long, required = true, num_args = 1..)]
        shard: Vec<String>,
        /// held-out shard for validation (defaults to the training shard's tail)
        #[arg(long)]
        val: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 1024)]
        seq: usize,
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        #[arg(long, default_value_t = 100)]
        warmup: usize,
        #[arg(long, default_value_t = 6e-4)]
        lr: f32,
        #[arg(long, default_value_t = 0.1)]
        wd: f32,
        #[arg(long, default_value_t = 1.0)]
        clip: f32,
        #[arg(long, default_value_t = 100)]
        eval_every: usize,
        #[arg(long, default_value_t = 500)]
        save_every: usize,
        /// refresh the expert descriptor subspaces (PCA of routed inputs) every N steps (0 = off)
        #[arg(long, default_value_t = 200)]
        pca_every: usize,
        /// disable online descriptor updates (preserves checkpoint extras)
        #[arg(long)]
        freeze_desc: bool,
        /// tiny genome (smoke test)
        #[arg(long)]
        tiny: bool,
        /// vocab override (e.g. 256 for byte shards)
        #[arg(long)]
        vocab: Option<usize>,
        /// every N-th layer is a softmax anchor (1 = the all-softmax twin)
        #[arg(long)]
        anchor_every: Option<usize>,
        /// short-conv taps before the mixer projections (0 = off)
        #[arg(long)]
        conv_k: Option<usize>,
        /// append the checkpoint-compatible one-head GDN correction lane
        #[arg(long)]
        gdn_lane: bool,
        /// use the parameter-neutral in-place Phase-Delta hybrid mixer
        #[arg(long)]
        phase_delta: bool,
        /// use Phase-Delta at exactly one zero-based hybrid layer (selection
        /// implies Phase-Delta and takes precedence over --phase-delta)
        #[arg(long, value_name = "LAYER")]
        phase_delta_layer: Option<usize>,
        /// use Phase-Delta at exactly two zero-based hybrid layers (for
        /// example `--phase-delta-layers 3,6`)
        #[arg(long, value_name = "LAYER,LAYER", value_delimiter = ',')]
        phase_delta_layers: Option<Vec<usize>>,
        /// causal four-position smoothing of resonance scores before top-1 routing
        #[arg(long)]
        router_smooth_k4: bool,
        /// fixed reconstruction-error margin for conditional top-2 routing
        /// (default off; full train-time backward integration is not yet enabled)
        #[arg(long, value_name = "MARGIN")]
        router_top2_margin: Option<f32>,
        /// bounded anchor swa_sink_v1: served exact window in keys (0 = the
        /// legacy full-causal anchor); with --resume it continues a legacy
        /// checkpoint under the mask
        #[arg(long, value_name = "W")]
        anchor_window: Option<usize>,
        /// trained NoPE sink vectors per KV head (needs --anchor-window;
        /// with --resume the arena is extended by the new sink tensors,
        /// legacy parameters and moments stay bit-identical)
        #[arg(long, value_name = "S")]
        anchor_sink: Option<usize>,
        /// SWAX stochastic training windows sampled per step, e.g. `64,128`
        #[arg(long, value_name = "W,W", value_delimiter = ',')]
        anchor_train_windows: Option<Vec<usize>>,
        /// explicit zero-based anchor schedule, e.g. `3,7` (default: every
        /// --anchor-every-th layer)
        #[arg(long, value_name = "LAYER,LAYER", value_delimiter = ',')]
        anchor_layers: Option<Vec<usize>>,
        /// mixer of the non-anchor layers: `hybrid_k` (legacy phase core) or
        /// `gdn` (gated delta-rule core, plan S7 variant B)
        #[arg(long, value_parser = ["hybrid_k", "gdn"])]
        mixer: Option<String>,
        /// GDN heads (nk = nv; default 4)
        #[arg(long)]
        gdn_heads: Option<usize>,
        /// GDN key head dim (default 128)
        #[arg(long)]
        gdn_dk: Option<usize>,
        /// GDN value head dim (default 128)
        #[arg(long)]
        gdn_dv: Option<usize>,
        /// JSON overrides merged into the genome config (ablations), e.g. '{"hidden":256,"layers":6}'
        #[arg(long)]
        cfg_json: Option<String>,
        /// warm-start: copy every name+size-matching tensor from this
        /// checkpoint (twin donor: embed/head/norms/FFN land, mixers stay fresh)
        #[arg(long)]
        init_from: Option<PathBuf>,
        /// feature distillation: teacher checkpoint whose final hidden the
        /// student's xf is pulled toward (MSE), on top of the CE loss
        #[arg(long)]
        distill_from: Option<PathBuf>,
        /// distillation weight w in L = CE + w·MSE(xf, xf_teacher)
        #[arg(long, default_value_t = 1.0)]
        distill_w: f32,
        /// teacher micro-batch for --distill-from (0 = --batch); 1–2 when the
        /// student already fills the card
        #[arg(long, default_value_t = 0)]
        distill_batch: usize,
        /// hold the donated tensors still for the first N steps
        #[arg(long, default_value_t = 0)]
        freeze_donor: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// carry the recurrent state / conv history / anchor keys across
        /// consecutive windows of one document stream (plan S6b, TBPTT-lite)
        #[arg(long)]
        carry: bool,
        /// restart every stream after N windows (0 = only at shard end / EOT)
        #[arg(long, default_value_t = 16)]
        carry_reset_every: usize,
        /// end-of-text token id: a window containing it ends the stream
        #[arg(long)]
        carry_eot: Option<u32>,
        /// with --resume: warmup+cosine over THIS run's steps (relative to
        /// the resume step) instead of the absolute step
        #[arg(long)]
        lr_restart: bool,
        /// dropless expert routing (capacity = every row)
        #[arg(long)]
        dropless: bool,
        /// extra held-out shards reported at every eval, `name=path` (repeatable)
        #[arg(long)]
        val_extra: Vec<String>,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Sub::Config => {
            let cfg = cortiq_embryo::model::EmbryoCfg::embryo0();
            let (total, active) = cfg.params();
            println!("{cfg:#?}");
            println!(
                "params: total {:.2} M, active/token {:.2} M",
                total as f64 / 1e6,
                active as f64 / 1e6
            );
            let lay = cortiq_embryo::model::Layout::new(&cfg);
            println!(
                "trainer arena now (shared expert, no routed experts yet): {:.2} M",
                lay.total as f64 / 1e6
            );
        }
        Sub::BytesShard { input, output } => {
            let text = std::fs::read(&input).expect("read input");
            let shard = cortiq_embryo::train::Shard::from_bytes(&text);
            shard.save(&output).expect("write shard");
            println!("{} tokens → {}", shard.tokens.len(), output.display());
        }
        Sub::TrainTokenizer {
            inputs,
            out,
            vocab,
            sample_mb_bytes,
        } => {
            cortiq_embryo::corpus::train_tokenizer(&inputs, &out, vocab, sample_mb_bytes);
        }
        Sub::Shard {
            tokenizer,
            inputs,
            out,
            max_tokens,
        } => {
            cortiq_embryo::corpus::shard(&tokenizer, &inputs, &out, max_tokens);
        }
        Sub::Export {
            ckpt,
            tokenizer,
            out,
            dtype,
            genome_id,
            genome_status,
        } => {
            // a genome file is never rewritten (export_genome checks again)
            if let Err(e) = cortiq_embryo::export::refuse_genome_overwrite(&out) {
                eprintln!("export: {e:#}");
                std::process::exit(1);
            }
            let ck = cortiq_embryo::train::load_checkpoint(&ckpt).expect("load checkpoint");
            let tj = std::fs::read(&tokenizer).expect("read tokenizer.json");
            let storage = match dtype.as_str() {
                "f16" => cortiq_core::types::TensorDtype::F16,
                _ => cortiq_core::types::TensorDtype::F32,
            };
            let genome = genome_id.map(|id| cortiq_embryo::export::ExportGenome {
                id,
                status: genome_status.expect("clap: --genome-status required"),
            });
            cortiq_embryo::export::export_genome(&ck, &tj, &out, storage, genome.as_ref())
                .expect("export");
            let (total, active) = ck.cfg.params();
            println!(
                "exported step {} → {} ({dtype}; {:.1} M params, {:.1} M active)",
                ck.step,
                out.display(),
                total as f64 / 1e6,
                active as f64 / 1e6
            );
            if genome.is_some() {
                let m = cortiq_core::format::CmfModel::open(&out).expect("re-open export");
                let g = m.header.genome.as_ref().expect("genome written");
                println!(
                    "genome '{}' gen {} status {} encoding {}: trunk_hash {} master_trunk_hash {}",
                    g.id, g.generation, g.status, g.encoding, g.trunk_hash, g.master_trunk_hash
                );
            }
        }
        Sub::SampleText {
            input,
            skip,
            docs,
            out,
        } => {
            cortiq_embryo::corpus::sample_text(&input, skip, docs, &out);
        }
        Sub::SkillBake {
            ckpt,
            tokenizer,
            corpus,
            base,
            out,
            id,
            layers,
            steps_a,
            steps_b,
            lr_a,
            lr_b,
            l1,
            tau,
            eval_every,
            batch,
            seq,
            phi_layer,
            phi_len,
            rank,
            seed,
            sft_train,
            sft_dev,
            sft_final,
            lm_train,
            lm_dev,
            lm_frac,
            dev_batches,
            sft_manifest,
            phi_prompts,
            general_prompts,
            refit_base,
            phi_max,
            phi_batch,
            phi_max_len,
            route_margin,
            target_fpr,
            dropless,
        } => {
            let _ = dropless; // v2 is always dropless; legacy never used it
            #[cfg(any(target_os = "macos", feature = "vulkan"))]
            {
                let fail = |e: anyhow::Error| -> ! {
                    eprintln!("skill-bake: {e:#}");
                    std::process::exit(1);
                };
                if let Some(sft_train) = sft_train {
                    // ---- v2: a record over the frozen genome ----
                    if !corpus.is_empty() {
                        fail(anyhow::anyhow!(
                            "--corpus (legacy) and --sft-train (v2) are exclusive"
                        ));
                    }
                    let need = |x: Option<PathBuf>, flag: &str| {
                        x.unwrap_or_else(|| fail(anyhow::anyhow!("v2 bake needs {flag}")))
                    };
                    let inputs = cortiq_embryo::skill::BakeV2Inputs {
                        ckpt: ckpt.clone(),
                        base: base.clone(),
                        out: out.clone(),
                        tokenizer,
                        sft_train: sft_train.clone(),
                        sft_dev: need(sft_dev, "--sft-dev"),
                        sft_final,
                        sft_manifest,
                        lm_train,
                        lm_dev,
                        phi_prompts: need(phi_prompts, "--phi-prompts"),
                        general_prompts,
                    };
                    // cheap header reads for the defaults (the bake re-checks everything)
                    let nl = cortiq_core::format::CmfModel::open(&base)
                        .unwrap_or_else(|e| fail(e.into()))
                        .header
                        .arch
                        .num_layers;
                    let layers =
                        layers.unwrap_or_else(|| vec![nl.saturating_sub(2), nl.saturating_sub(1)]);
                    let lmin = layers.iter().copied().min().unwrap_or(0);
                    let phi_layer = phi_layer.unwrap_or_else(|| {
                        if lmin == 0 {
                            fail(anyhow::anyhow!(
                                "--layers includes layer 0: no backbone layer precedes it for φ"
                            ));
                        }
                        (nl * 2 / 3).min(lmin - 1)
                    });
                    if let Some(sq) = seq {
                        let shard_seq = cortiq_embryo::sft::SftShard::load(&sft_train)
                            .unwrap_or_else(|e| fail(e))
                            .seq;
                        if sq != shard_seq {
                            fail(anyhow::anyhow!(
                                "--seq {sq} differs from the SFT shards' length {shard_seq}"
                            ));
                        }
                    }
                    let args = cortiq_embryo::skill::BakeV2Args {
                        id,
                        layers,
                        steps_a,
                        steps_b,
                        lr_a,
                        lr_b,
                        l1,
                        tau,
                        eval_every,
                        batch,
                        dev_batches,
                        lm_frac,
                        phi_layer,
                        phi_max,
                        phi_batch,
                        phi_max_len,
                        rank,
                        route_margin,
                        refit_base,
                        target_fpr,
                        seed,
                    };
                    let summary = cortiq_embryo::skill::bake_v2(&inputs, &args, &|| false)
                        .unwrap_or_else(|e| fail(e));
                    println!("{}", serde_json::to_string(&summary).expect("summary JSON"));
                } else {
                    // ---- legacy: flat corpus, v1 record, full rewrite ----
                    use cortiq_embryo::skill::{BakeArgs, append_to_cmf, bake};
                    if corpus.is_empty() {
                        fail(anyhow::anyhow!(
                            "give --sft-train/--sft-dev/--phi-prompts/--general-prompts (v2) or --corpus (legacy)"
                        ));
                    }
                    cortiq_embryo::skill::check_out_path(&base, &out).unwrap_or_else(|e| fail(e));
                    let base_model = cortiq_core::format::CmfModel::open(&base)
                        .unwrap_or_else(|e| fail(e.into()));
                    if base_model.header.genome.is_some() {
                        fail(anyhow::anyhow!(
                            "--base carries a GENOME: it takes only v2 records (--sft-train …); \
                             the legacy --corpus path rewrites the file"
                        ));
                    }
                    drop(base_model);
                    let tokenizer = tokenizer
                        .unwrap_or_else(|| fail(anyhow::anyhow!("legacy bake needs --tokenizer")));
                    let ck = cortiq_embryo::train::load_checkpoint(&ckpt).expect("load checkpoint");
                    // tokenize the corpus with our tokenizer
                    let bpe = cortiq_embryo::tokenizer::Bpe::load(&tokenizer).expect("tokenizer");
                    let eot = bpe.special_id(cortiq_embryo::tokenizer::EOT).unwrap_or(0) as u16;
                    let mut toks: Vec<u16> = Vec::new();
                    let mut cache = std::collections::HashMap::new();
                    for p in &corpus {
                        cortiq_embryo::data::for_each_doc(p, |text| {
                            let mut ids = Vec::new();
                            bpe.encode(text, &mut cache, &mut ids);
                            toks.extend(ids.iter().map(|&i| i as u16));
                            toks.push(eot);
                        })
                        .expect("read corpus");
                    }
                    println!("skill corpus: {} tokens", toks.len());
                    let shard = cortiq_embryo::train::Shard { tokens: toks };
                    let nl = ck.cfg.layers;
                    let layers =
                        layers.unwrap_or_else(|| vec![nl.saturating_sub(2), nl.saturating_sub(1)]);
                    let a = BakeArgs {
                        id: id.clone(),
                        layers: layers.clone(),
                        steps_a,
                        steps_b,
                        lr_a,
                        lr_b,
                        l1,
                        tau,
                        eval_every,
                        batch,
                        seq: seq.unwrap_or(512),
                        phi_layer: phi_layer.unwrap_or(nl * 2 / 3),
                        phi_len,
                        rank,
                        seed,
                    };
                    let (tensors, sel, kept, (l0, la, lb)) =
                        bake(&ck, &shard, &a, &|| false).expect("bake");
                    let quality = serde_json::json!({
                        "held_out_loss": {"base": l0, "mask": la, "mask+fcd": lb},
                        "held_out_ppl": {"base": l0.exp(), "mask": la.exp(), "mask+fcd": lb.exp()},
                        "kept_fraction": kept,
                    });
                    // a private temp (never an existing file), published
                    // without overwriting an --out that appeared meanwhile
                    let (tmp, f) = cortiq_embryo::skill::create_bake_tmp(&out)
                        .unwrap_or_else(|e| fail(e));
                    drop(f);
                    let unchanged =
                        match append_to_cmf(&base, &tmp, &id, &layers, &tensors, sel, quality) {
                            Ok(n) => n,
                            Err(e) => {
                                let _ = std::fs::remove_file(&tmp);
                                fail(e)
                            }
                        };
                    cortiq_embryo::skill::publish_new_file(&tmp, &out).unwrap_or_else(|e| fail(e));
                    println!(
                        "skill '{id}' appended → {} ({} tensors over layers {:?}; {unchanged} base tensors byte-identical; held-out ppl {:.1} → {:.1})",
                        out.display(),
                        tensors.len(),
                        layers,
                        l0.exp(),
                        lb.exp()
                    );
                    match cortiq_embryo::skill::calibrate_file(&out, 0.05) {
                        Ok(Some(c)) => println!(
                            "router calibrated over {} held-out φ: temperature {:.3e}, novelty θ {:.3} (fpr {:.2})",
                            c.samples, c.temperature, c.novelty_theta, c.target_fpr
                        ),
                        Ok(None) => println!("router calibration: no held-out φ in the file"),
                        Err(e) => eprintln!("router calibration failed: {e}"),
                    }
                }
            }
            #[cfg(not(any(target_os = "macos", feature = "vulkan")))]
            {
                let _ = (
                    ckpt, tokenizer, corpus, base, out, id, layers, steps_a, steps_b, lr_a, lr_b,
                    l1, tau, eval_every, batch, seq, phi_layer, phi_len, rank, seed, sft_train,
                    sft_dev, sft_final, lm_train, lm_dev, lm_frac, dev_batches, sft_manifest,
                    phi_prompts, general_prompts, refit_base, phi_max, phi_batch, phi_max_len,
                    route_margin, target_fpr,
                );
                eprintln!("skill-bake needs a GPU backend: Metal (macOS) or --features vulkan");
                std::process::exit(1);
            }
        }
        Sub::ResidualPretrain {
            shard,
            val,
            resume,
            teacher,
            out,
            batch,
            seq,
            steps,
            lr,
            wd,
            clip,
            seed,
        } => {
            #[cfg(target_os = "macos")]
            cortiq_embryo::cli::residual_pretrain(cortiq_embryo::cli::ResidualPretrainArgs {
                shard,
                val,
                resume,
                teacher,
                out,
                batch,
                seq,
                steps,
                lr,
                wd,
                clip,
                seed,
            });
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (
                    shard, val, resume, teacher, out, batch, seq, steps, lr, wd, clip, seed,
                );
                eprintln!("residual pretrain needs Metal (macOS)");
                std::process::exit(1);
            }
        }
        Sub::AnchorGraft {
            student,
            donor,
            out,
            val,
            shard,
            batch,
            seq,
        } => {
            #[cfg(target_os = "macos")]
            cortiq_embryo::cli::anchor_graft(cortiq_embryo::cli::AnchorGraftArgs {
                student,
                donor,
                out,
                shard,
                val,
                batch,
                seq,
            });
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (student, donor, out, val, shard, batch, seq);
                eprintln!("anchor graft needs Metal (macOS)");
                std::process::exit(1);
            }
        }
        Sub::GqaLane {
            resume,
            donor,
            shard,
            val,
            out,
            batch,
            seq,
            steps,
            lr,
            wd,
            clip,
            seed,
        } => {
            #[cfg(target_os = "macos")]
            cortiq_embryo::cli::gqa_lane(cortiq_embryo::cli::GqaLaneArgs {
                resume,
                donor,
                shard,
                val,
                out,
                batch,
                seq,
                steps,
                lr,
                wd,
                clip,
                seed,
            });
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (
                    resume, donor, shard, val, out, batch, seq, steps, lr, wd, clip, seed,
                );
                eprintln!("gqa lane needs Metal (macOS)");
                std::process::exit(1);
            }
        }
        Sub::Sleep {
            ckpt,
            tokenizer,
            cmf,
            ood_dir,
            idle_min,
            min_tokens,
            gate,
            requant_gate,
            held_out,
            cortiq_bin,
            layers,
            steps_a,
            steps_b,
            batch,
            seq,
            once,
            force,
            poll_secs,
            grow_after,
        } => {
            #[cfg(target_os = "macos")]
            {
                let nl = cortiq_embryo::train::load_checkpoint(&ckpt)
                    .map(|c| c.cfg.layers)
                    .unwrap_or(8);
                let layers = layers.unwrap_or_else(|| (nl.saturating_sub(3)..nl).collect());
                cortiq_embryo::sleep::run(cortiq_embryo::sleep::SleepArgs {
                    ckpt,
                    tokenizer,
                    cmf,
                    ood_dir,
                    idle_min,
                    min_tokens,
                    gate,
                    requant_gate,
                    held_out,
                    cortiq_bin,
                    layers,
                    steps_a,
                    steps_b,
                    batch,
                    seq,
                    once,
                    force,
                    poll_secs,
                    grow_after,
                })
                .expect("sleep");
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (
                    ckpt,
                    tokenizer,
                    cmf,
                    ood_dir,
                    idle_min,
                    min_tokens,
                    gate,
                    requant_gate,
                    held_out,
                    cortiq_bin,
                    layers,
                    steps_a,
                    steps_b,
                    batch,
                    seq,
                    once,
                    force,
                    poll_secs,
                    grow_after,
                );
                eprintln!("needs Metal (macOS)");
                std::process::exit(1);
            }
        }
        Sub::Grow {
            ckpt,
            tokenizer,
            corpus,
            held,
            general,
            trace_tokens,
            trace_docs,
            experts,
            layers,
            shell_mode,
            shell_quantile,
            shell_target_shift,
            bias_mode,
            source_mode,
            novel_quantile,
            desc_mode,
            record_out,
            base,
            id,
            out_ckpt,
            export,
            steps,
            lr,
            batch,
            seq,
            gate,
            held_batches,
            noise,
            shift,
            seed,
        } => {
            #[cfg(any(target_os = "macos", feature = "vulkan"))]
            {
                let shell_mode = cortiq_embryo::cli::ShellMode::parse(&shell_mode).unwrap_or_else(|e| {
                    eprintln!("grow: {e}");
                    std::process::exit(1);
                });
                let bias_mode = cortiq_embryo::cli::BiasMode::parse(&bias_mode).unwrap_or_else(|e| {
                    eprintln!("grow: {e}");
                    std::process::exit(1);
                });
                let source_mode = cortiq_embryo::cli::SourceMode::parse(&source_mode).unwrap_or_else(|e| {
                    eprintln!("grow: {e}");
                    std::process::exit(1);
                });
                let desc_mode = desc_mode.map(|d| {
                    cortiq_embryo::cli::DescMode::parse(&d).unwrap_or_else(|e| {
                        eprintln!("grow: {e}");
                        std::process::exit(1);
                    })
                });
                let a = cortiq_embryo::cli::GrowCli {
                    ckpt,
                    tokenizer,
                    corpus,
                    held,
                    general,
                    trace_tokens,
                    trace_docs,
                    experts,
                    layers,
                    shell_mode,
                    shell_quantile,
                    shell_target_shift,
                    bias_mode,
                    source_mode,
                    novel_quantile,
                    desc_mode,
                    record_out,
                    base,
                    id,
                    out_ckpt,
                    export,
                    steps,
                    lr,
                    batch,
                    seq,
                    gate,
                    held_batches,
                    noise,
                    shift,
                    seed,
                };
                match cortiq_embryo::cli::grow(&a) {
                    Ok(summary) => {
                        println!("{}", serde_json::to_string(&summary).expect("summary JSON"));
                    }
                    Err(e) if e.to_string().starts_with(cortiq_embryo::cli::GROW_REJECTED) => {
                        eprintln!("grow: {e:#}");
                        std::process::exit(2);
                    }
                    Err(e) => {
                        eprintln!("grow: {e:#}");
                        std::process::exit(1);
                    }
                }
            }
            #[cfg(not(any(target_os = "macos", feature = "vulkan")))]
            {
                let _ = (
                    ckpt, tokenizer, corpus, held, general, trace_tokens, trace_docs, experts, layers, shell_mode,
                    shell_quantile, shell_target_shift, bias_mode, source_mode, novel_quantile, desc_mode, record_out,
                    base, id, out_ckpt, export, steps, lr, batch, seq, gate, held_batches, noise, shift, seed,
                );
                eprintln!("grow needs a GPU backend: Metal (macOS) or --features vulkan");
                std::process::exit(1);
            }
        }
        Sub::Reshell {
            ckpt,
            genome_ckpt,
            tokenizer,
            corpus,
            held,
            general,
            trace_tokens,
            trace_docs,
            layers,
            shell_mode,
            shell_quantile,
            shell_target_shift,
            bias_mode,
            source_mode,
            novel_quantile,
            desc_mode,
            seed,
            record_out,
            base,
            id,
            batch,
            seq,
        } => {
            #[cfg(any(target_os = "macos", feature = "vulkan"))]
            {
                let shell_mode = cortiq_embryo::cli::ShellMode::parse(&shell_mode).unwrap_or_else(|e| {
                    eprintln!("reshell: {e}");
                    std::process::exit(1);
                });
                let bias_mode = bias_mode.map(|b| {
                    cortiq_embryo::cli::BiasMode::parse(&b).unwrap_or_else(|e| {
                        eprintln!("reshell: {e}");
                        std::process::exit(1);
                    })
                });
                let source_mode = cortiq_embryo::cli::SourceMode::parse(&source_mode).unwrap_or_else(|e| {
                    eprintln!("reshell: {e}");
                    std::process::exit(1);
                });
                let desc_mode = desc_mode.map(|d| {
                    cortiq_embryo::cli::DescMode::parse(&d).unwrap_or_else(|e| {
                        eprintln!("reshell: {e}");
                        std::process::exit(1);
                    })
                });
                let a = cortiq_embryo::cli::ReshellCli {
                    ckpt,
                    genome_ckpt,
                    tokenizer,
                    corpus,
                    held,
                    general,
                    trace_tokens,
                    trace_docs,
                    layers,
                    shell_mode,
                    shell_quantile,
                    shell_target_shift,
                    bias_mode,
                    source_mode,
                    novel_quantile,
                    desc_mode,
                    seed,
                    record_out,
                    base,
                    id,
                    batch,
                    seq,
                };
                match cortiq_embryo::cli::reshell(&a) {
                    Ok(summary) => {
                        println!("{}", serde_json::to_string(&summary).expect("summary JSON"));
                    }
                    Err(e) => {
                        eprintln!("reshell: {e:#}");
                        std::process::exit(1);
                    }
                }
            }
            #[cfg(not(any(target_os = "macos", feature = "vulkan")))]
            {
                let _ = (
                    ckpt, genome_ckpt, tokenizer, corpus, held, general, trace_tokens, trace_docs, layers, shell_mode,
                    shell_quantile, shell_target_shift, bias_mode, source_mode, novel_quantile, desc_mode, seed,
                    record_out, base, id, batch, seq,
                );
                eprintln!("reshell needs a GPU backend: Metal (macOS) or --features vulkan");
                std::process::exit(1);
            }
        }
        Sub::MtpTrain {
            ckpt,
            tokenizer,
            corpus,
            base,
            out,
            heads,
            steps,
            lr,
            batch,
            seq,
            seed,
        } => {
            #[cfg(target_os = "macos")]
            {
                let ck = cortiq_embryo::train::load_checkpoint(&ckpt).expect("load checkpoint");
                let bpe = cortiq_embryo::tokenizer::Bpe::load(&tokenizer).expect("tokenizer");
                let eot = bpe.special_id(cortiq_embryo::tokenizer::EOT).unwrap_or(0) as u16;
                let mut toks: Vec<u16> = Vec::new();
                let mut cache = std::collections::HashMap::new();
                for p in &corpus {
                    cortiq_embryo::data::for_each_doc(p, |text| {
                        let mut ids = Vec::new();
                        bpe.encode(text, &mut cache, &mut ids);
                        toks.extend(ids.iter().map(|&i| i as u16));
                        toks.push(eot);
                    })
                    .expect("read corpus");
                }
                let shard = cortiq_embryo::train::Shard { tokens: toks };
                let (_gpu, st, held) =
                    cortiq_embryo::mtp::train_mtp(&ck, &shard, heads, steps, lr, batch, seq, seed)
                        .expect("mtp");
                println!(
                    "mtp heads trained: held-out losses {:?} (ppl {:?})",
                    held,
                    held.iter()
                        .map(|l| format!("{:.1}", l.exp()))
                        .collect::<Vec<_>>()
                );
                let out_path = out.unwrap_or_else(|| base.clone());
                let tmp = out_path.with_extension("cmf.tmp");
                let kept = cortiq_embryo::mtp::append_to_cmf(&base, &tmp, &st).expect("append");
                std::fs::rename(&tmp, &out_path).expect("rename");
                println!(
                    "{} MTP tensors appended → {} ({kept} base tensors byte-identical)",
                    2 * heads,
                    out_path.display()
                );
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (
                    ckpt, tokenizer, corpus, base, out, heads, steps, lr, batch, seq, seed,
                );
                eprintln!("needs Metal (macOS)");
                std::process::exit(1);
            }
        }
        Sub::Fetch { dir, urls } => {
            cortiq_embryo::corpus::fetch(&urls, &dir);
        }
        Sub::SftPrepare {
            input,
            tokenizer,
            train_out,
            dev_out,
            final_out,
            manifest_out,
            seq,
            format,
            group_field,
        } => {
            match format.as_str() {
                "oasst" => cortiq_embryo::sft::prepare(
                    &input,
                    &tokenizer,
                    seq,
                    &train_out,
                    &dev_out,
                    &final_out,
                    &manifest_out,
                )
                .expect("prepare response-only SFT corpus"),
                "messages" => cortiq_embryo::sft::prepare_messages(
                    &input,
                    &tokenizer,
                    seq,
                    &train_out,
                    &dev_out,
                    &final_out,
                    &manifest_out,
                    group_field.as_deref(),
                )
                .expect("prepare response-only SFT corpus from messages JSONL"),
                other => {
                    eprintln!("unknown --format {other}: expected oasst or messages");
                    std::process::exit(2);
                }
            }
        }
        Sub::RawReplay {
            input,
            out,
            max_tokens,
        } => {
            cortiq_embryo::sft::make_raw_replay(&input, &out, max_tokens)
                .expect("prepare raw replay shard");
        }
        Sub::Bench { reps, no_verify } => {
            #[cfg(target_os = "macos")]
            {
                let Some(rows) = cortiq_embryo::bench::run(reps, !no_verify) else {
                    eprintln!("no Metal device");
                    std::process::exit(1);
                };
                println!(
                    "{:<44} {:>6} {:>6} {:>6} {:>9} {:>8} {:>10}",
                    "shape", "M", "N", "K", "gpu ms", "TFLOPS", "max|err|"
                );
                for r in &rows {
                    println!(
                        "{:<44} {:>6} {:>6} {:>6} {:>9.3} {:>8.2} {:>10.2e}",
                        r.name, r.m, r.n, r.k, r.gpu_ms, r.tflops, r.max_abs_err
                    );
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (reps, no_verify);
                eprintln!("bench needs Metal (macOS)");
                std::process::exit(1);
            }
        }
        Sub::CarryProfile {
            ckpt,
            shard,
            windows,
            batch,
            seq,
            seed,
            one_pass,
        } => {
            #[cfg(any(target_os = "macos", feature = "vulkan"))]
            cortiq_embryo::cli::carry_profile(cortiq_embryo::cli::CarryProfileArgs {
                ckpt,
                shard,
                windows,
                batch,
                seq,
                seed,
                one_pass,
            });
            #[cfg(not(any(target_os = "macos", feature = "vulkan")))]
            {
                let _ = (ckpt, shard, windows, batch, seq, seed, one_pass);
                eprintln!("needs a training device: Metal (macOS) or Vulkan (--features vulkan)");
                std::process::exit(1);
            }
        }
        Sub::ProbeRecall {
            ckpt,
            shard,
            pairs,
            dists,
            trials,
            seed,
            max_seq,
        } => {
            #[cfg(any(target_os = "macos", feature = "vulkan"))]
            cortiq_embryo::cli::probe_recall(cortiq_embryo::cli::ProbeRecallArgs {
                ckpt,
                shard,
                pairs,
                dists,
                trials,
                seed,
                max_seq,
            });
            #[cfg(not(any(target_os = "macos", feature = "vulkan")))]
            {
                let _ = (ckpt, shard, pairs, dists, trials, seed, max_seq);
                eprintln!("needs a training device: Metal (macOS) or Vulkan (--features vulkan)");
                std::process::exit(1);
            }
        }
        Sub::StepBench {
            batch,
            seq,
            steps,
            tiny,
            cfg_json,
        } => {
            #[cfg(any(target_os = "macos", feature = "vulkan"))]
            cortiq_embryo::cli::step_bench(batch, seq, steps, tiny, cfg_json.as_deref());
            #[cfg(not(any(target_os = "macos", feature = "vulkan")))]
            {
                let _ = (batch, seq, steps, tiny, cfg_json);
                eprintln!("needs a training device: Metal (macOS) or Vulkan (--features vulkan)");
                std::process::exit(1);
            }
        }
        #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
        Sub::VulkanTrain {
            ckpt,
            shard,
            val,
            out,
            batch,
            seq,
            steps,
            val_batches,
            lr,
            wd,
            clip,
            seed,
            dropless,
            anchor_window,
            anchor_sink,
            anchor_train_windows,
            carry,
            carry_reset_every,
            carry_eot,
            eval_every,
            stats_every,
        } => {
            use cortiq_embryo::cli::CarryAcc;
            use cortiq_embryo::model::EmbryoGpu;
            use cortiq_embryo::train::{Sampler, Shard, StreamSampler, load_checkpoint, save_checkpoint};

            // Never replace an input or an existing artifact.  The
            // checkpoint writer uses a sibling `.tmp` followed by rename;
            // reject both names up front so a failed/duplicate proof cannot
            // silently destroy a prior receipt.
            assert!(ckpt != out, "Vulkan train output must differ from input");
            assert!(
                !out.exists(),
                "refusing to overwrite checkpoint {}",
                out.display()
            );
            let out_tmp = out.with_extension("tmp");
            assert!(
                !out_tmp.exists(),
                "refusing to reuse an existing temporary checkpoint {}",
                out_tmp.display()
            );
            assert!(
                seq % 64 == 0,
                "Vulkan training requires --seq multiple of 64"
            );
            assert!(
                shard != val,
                "training and validation shards must be distinct"
            );
            let mut ck = load_checkpoint(&ckpt).expect("load checkpoint");
            // Bounded-anchor continuation (plan S4): a legacy full-causal
            // checkpoint is extended with trained sinks and then trained
            // under the band mask; a checkpoint that already carries a
            // bounded anchor is continued as it is (flags must agree).
            if let Some(window) = anchor_window {
                let sink = anchor_sink.unwrap_or(4);
                let train_windows = anchor_train_windows.clone().unwrap_or_default();
                if ck.cfg.anchor_window == 0 {
                    let before = ck.params.len();
                    ck = cortiq_embryo::train::append_anchor_sinks_checkpoint(
                        &ck,
                        window,
                        sink,
                        &train_windows,
                        seed,
                    )
                    .expect("bounded-anchor continuation");
                    println!(
                        "bounded anchor: window {window} sink {sink} train_windows {train_windows:?}; appended {} parameters; legacy prefix/moments copied",
                        ck.params.len() - before
                    );
                } else {
                    assert_eq!(
                        (ck.cfg.anchor_window, ck.cfg.anchor_sink),
                        (window, sink),
                        "checkpoint already carries a bounded anchor with a different geometry"
                    );
                    if !train_windows.is_empty() {
                        ck.cfg.anchor_train_windows = train_windows.clone();
                    }
                    println!(
                        "bounded anchor: continuing window {window} sink {sink} train_windows {:?}",
                        ck.cfg.anchor_train_windows
                    );
                }
            }
            let train_shard = Shard::load(&shard).expect("load training shard");
            let val_shard = Shard::load(&val).expect("load validation shard");
            let lay = cortiq_embryo::model::Layout::new(&ck.cfg);
            assert_eq!(
                ck.params.len(),
                lay.total,
                "checkpoint/layout arena mismatch"
            );
            let mut gpu = if carry {
                EmbryoGpu::new_carry(ck.cfg.clone(), batch, seq, &ck.params, dropless)
            } else if dropless {
                EmbryoGpu::new_eval_dropless(ck.cfg.clone(), batch, seq, &ck.params)
            } else {
                EmbryoGpu::new(ck.cfg.clone(), batch, seq, &ck.params)
            }
            .expect("native Vulkan adapter/context");
            if carry {
                eprintln!(
                    "carry: state carried across windows (reset every {carry_reset_every} windows, eot {carry_eot:?}); anchor keys carried: {}",
                    if gpu.carry_pad() > 0 { format!("{} columns", gpu.carry_pad()) } else { "no (legacy full-causal anchor)".into() }
                );
            }
            // Compact transfer checkpoints intentionally omit Adam moments:
            // their provenance step is not an optimizer step.  Only resume
            // the bias-correction counter when both moment arenas are present.
            if let (Some(m), Some(v)) = (&ck.m, &ck.v) {
                assert_eq!(m.len(), lay.total, "checkpoint m/layout arena mismatch");
                assert_eq!(v.len(), lay.total, "checkpoint v/layout arena mismatch");
                gpu.m.write_from(m);
                gpu.v.write_from(v);
                gpu.step = ck.step;
            } else {
                gpu.step = 0;
            }
            gpu.set_desc(&ck.extras);
            // Descriptor routing is a frozen, explicit limitation of this
            // first native bridge; updates must not silently diverge from the
            // compact checkpoint's routing descriptors.
            gpu.desc_updates.set(false);
            let mut sampler = Sampler::new(batch, seq, seed);
            let mut stream = carry.then(|| StreamSampler::new(batch, seq, seed, carry_reset_every, carry_eot));
            let mut carry_acc = carry.then(|| CarryAcc::new(ck.cfg.layers));
            let mut last_reset: Vec<bool> = vec![true; batch];
            let mut tok = Vec::new();
            let mut tgt = Vec::new();
            let mut vt = Vec::new();
            let mut vg = Vec::new();
            let val_window_tokens = batch * seq + 1;
            assert!(
                val_shard.tokens.len() > val_window_tokens,
                "held-out shard is shorter than one validation window"
            );
            let available_val_batches = val_shard.tokens.len() / val_window_tokens;
            let val_batches = val_batches.max(1).min(available_val_batches.max(1));
            let eval_val = |gpu: &EmbryoGpu, tokens: &mut Vec<u32>, targets: &mut Vec<u32>| {
                (0..val_batches)
                    .map(|i| {
                        Sampler::fixed_batch(&val_shard, batch, seq, i, tokens, targets);
                        gpu.eval_loss(tokens, targets)
                    })
                    .sum::<f32>()
                    / val_batches as f32
            };
            let before = eval_val(&gpu, &mut vt, &mut vg);
            assert!(
                before.is_finite(),
                "non-finite held-out loss before training"
            );
            eprintln!(
                "native Vulkan adapter active; held-out loss before={before:.6} batches={val_batches}"
            );
            for _ in 0..steps {
                if let Some(st) = stream.as_mut() {
                    let reset = st.batch(&train_shard, &mut tok, &mut tgt);
                    gpu.carry_begin(&reset);
                    last_reset = reset;
                } else {
                    sampler.batch(&train_shard, &mut tok, &mut tgt);
                }
                let (loss, gnorm, ms) = gpu.train_step(&tok, &tgt, lr, wd, clip);
                if let (Some(st), Some(acc)) = (stream.as_ref(), carry_acc.as_mut()) {
                    let per = gpu.per_position_loss();
                    gpu.carry_commit();
                    let stats = gpu.carry_state_stats();
                    let line = acc.push(gpu.step, &per, seq, &last_reset, &st.depths(), &stats);
                    if stats_every > 0 && gpu.step % stats_every as u32 == 0 {
                        eprintln!("{line}");
                    }
                }
                assert!(
                    loss.is_finite() && gnorm.is_finite(),
                    "non-finite Vulkan training result at step {}: loss={loss:?} grad_l2={gnorm:?}",
                    gpu.step
                );
                if dropless {
                    let cap = (batch * seq).div_ceil(64) * 64;
                    let drops: usize = gpu
                        .routing_counts()
                        .iter()
                        .flat_map(|counts| counts.iter())
                        .map(|&n| n.saturating_sub(cap as u32) as usize)
                        .sum();
                    assert_eq!(
                        drops, 0,
                        "dropless Vulkan route capacity overflow at step {}: cap={cap} drops={drops}",
                        gpu.step
                    );
                }
                // train_step has already reduced every gradient element on
                // device and rejects non-finite loss/norm before advancing
                // AdamW.  Do not mirror the full arena here: the old witness
                // readback copied ~224 MiB GPU -> CPU -> GPU per step.
                eprintln!(
                    "step={} train_loss={loss:.6} grad_l2={gnorm:.6} gpu_ms={ms:.3}",
                    gpu.step
                );
                if eval_every > 0 && gpu.step % eval_every as u32 == 0 {
                    let hl = eval_val(&gpu, &mut vt, &mut vg);
                    eprintln!("step={} heldout_loss={hl:.6} heldout_ppl={:.3}", gpu.step, hl.exp());
                }
            }
            let after = eval_val(&gpu, &mut vt, &mut vg);
            assert!(after.is_finite(), "non-finite held-out loss after training");
            if let Some(acc) = carry_acc.as_ref() {
                eprint!("{}", acc.report());
            }
            let desc = gpu.desc_host();
            let desc_refs: Vec<(&str, &[f32])> =
                desc.iter().map(|(n, x)| (*n, x.as_slice())).collect();
            let params = gpu.params_host();
            let moments_m = gpu.m.to_vec();
            let moments_v = gpu.v.to_vec();
            assert!(
                params.iter().all(|x| x.is_finite()),
                "refusing to save non-finite Vulkan parameters"
            );
            assert!(
                moments_m.iter().all(|x| x.is_finite()) && moments_v.iter().all(|x| x.is_finite()),
                "refusing to save non-finite Vulkan AdamW moments"
            );
            assert!(
                desc_refs
                    .iter()
                    .all(|(_, x)| x.iter().all(|v| v.is_finite())),
                "refusing to save non-finite frozen routing descriptors"
            );
            save_checkpoint(
                &out,
                &ck.cfg,
                gpu.step,
                &params,
                Some(&moments_m),
                Some(&moments_v),
                &desc_refs,
            )
            .expect("write Vulkan checkpoint");
            println!(
                "native Vulkan train receipt: step={} heldout_batches={val_batches} heldout_before={before:.6} heldout_after={after:.6} output={}",
                gpu.step,
                out.display()
            );
        }
        #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
        Sub::VulkanSft {
            ckpt,
            shard,
            val,
            replay,
            raw_val,
            final_val,
            out,
            batch,
            seq,
            steps,
            allow_extended,
            val_batches,
            raw_val_batches,
            lr,
            wd,
            clip,
            seed,
            sampler_skip,
            dropless,
        } => {
            use cortiq_embryo::model::EmbryoGpu;
            use cortiq_embryo::sft::{SftSampler, SftShard};
            use cortiq_embryo::train::{load_checkpoint, save_checkpoint, Sampler, Shard};

            assert!(ckpt != out, "Vulkan SFT output must differ from input");
            assert!(!out.exists(), "refusing to overwrite checkpoint {}", out.display());
            assert!(!out.with_extension("tmp").exists(), "temporary output already exists");
            assert!(shard != val, "SFT train and dev shards must be distinct");
            assert!(shard != raw_val, "SFT train and raw validation shards must be distinct");
            if let Some(final_path) = &final_val {
                assert!(final_path != &shard, "SFT train and final shards must be distinct");
                assert!(final_path != &val, "SFT dev and final shards must be distinct");
            }
            assert!(batch >= 5 && batch % 5 == 0, "SFT batch must be a multiple of 5");
            assert!(seq % 64 == 0, "Vulkan SFT requires --seq multiple of 64");
            let ck = load_checkpoint(&ckpt).expect("load checkpoint");
            assert!(steps > ck.step as usize, "SFT --steps must advance the checkpoint");
            assert!(
                steps <= ck.step as usize + 500,
                "SFT pilot is capped at 500 steps beyond the input checkpoint"
            );
            if steps > ck.step as usize + 100 {
                assert!(
                    allow_extended,
                    "SFT extension beyond the initial 100 steps requires --allow-extended"
                );
                assert!(
                    sampler_skip > 0,
                    "SFT extension requires explicit --sampler-skip to continue the data stream"
                );
            }
            let train_sft = SftShard::load(&shard).expect("load SFT training shard");
            let val_sft = SftShard::load(&val).expect("load SFT validation shard");
            let replay_shard = Shard::load(&replay).expect("load raw replay shard");
            let raw_val_shard = Shard::load(&raw_val).expect("load raw validation shard");
            let final_sft = final_val
                .as_ref()
                .map(|path| SftShard::load(path).expect("load terminal SFT final shard"));
            assert_eq!(train_sft.seq, seq, "SFT train sequence mismatch");
            assert_eq!(val_sft.seq, seq, "SFT validation sequence mismatch");
            if let Some(final_sft) = &final_sft {
                assert_eq!(final_sft.seq, seq, "SFT final sequence mismatch");
                assert!(final_sft.records >= batch && final_sft.valid_tokens() > 0);
            }
            assert!(train_sft.records >= batch && train_sft.valid_tokens() > 0);
            assert!(val_sft.records >= batch && val_sft.valid_tokens() > 0);
            assert!(replay_shard.tokens.len() > seq + 1);
            assert!(raw_val_shard.tokens.len() > seq + 1);
            let lay = cortiq_embryo::model::Layout::new(&ck.cfg);
            assert_eq!(ck.params.len(), lay.total, "checkpoint/layout arena mismatch");
            let mut gpu = if dropless {
                EmbryoGpu::new_eval_dropless(ck.cfg.clone(), batch, seq, &ck.params)
            } else {
                EmbryoGpu::new(ck.cfg.clone(), batch, seq, &ck.params)
            }
            .expect("native Vulkan adapter/context");
            if let (Some(m), Some(v)) = (&ck.m, &ck.v) {
                assert_eq!(m.len(), lay.total, "checkpoint m/layout arena mismatch");
                assert_eq!(v.len(), lay.total, "checkpoint v/layout arena mismatch");
                gpu.m.write_from(m);
                gpu.v.write_from(v);
                gpu.step = ck.step;
            } else {
                panic!("response-only SFT requires the full checkpoint AdamW moments");
            }
            gpu.set_desc(&ck.extras);
            gpu.desc_updates.set(false);
            let sft_val_batches = val_batches
                .max(1)
                .min((val_sft.records / batch).max(1));
            let final_val_batches = final_sft
                .as_ref()
                .map(|shard| val_batches.max(1).min((shard.records / batch).max(1)));
            let raw_window_tokens = batch * seq + 1;
            let raw_available = raw_val_shard.tokens.len() / raw_window_tokens;
            let raw_batches = raw_val_batches.max(1).min(raw_available.max(1));
            let mut vt = Vec::new();
            let mut vg = Vec::new();
            let mut rt = Vec::new();
            let mut rg = Vec::new();
            let eval_sft = |gpu: &EmbryoGpu, tokens: &mut Vec<u32>, targets: &mut Vec<u32>| {
                (0..sft_val_batches)
                    .map(|i| {
                        val_sft.fixed_batch(batch, i, tokens, targets);
                        gpu.eval_loss(tokens, targets)
                    })
                    .sum::<f32>()
                    / sft_val_batches as f32
            };
            let eval_raw = |gpu: &EmbryoGpu, tokens: &mut Vec<u32>, targets: &mut Vec<u32>| {
                (0..raw_batches)
                    .map(|i| {
                        Sampler::fixed_batch(&raw_val_shard, batch, seq, i, tokens, targets);
                        gpu.eval_loss(tokens, targets)
                    })
                    .sum::<f32>()
                    / raw_batches as f32
            };
            let response_before = eval_sft(&gpu, &mut vt, &mut vg);
            let raw_before = eval_raw(&gpu, &mut rt, &mut rg);
            assert!(response_before.is_finite() && raw_before.is_finite());
            eprintln!(
                "native Vulkan SFT adapter active; response_nll_before={response_before:.6} batches={sft_val_batches} raw_lm_before={raw_before:.6} batches={raw_batches} replay=20%"
            );
            let mut sampler = SftSampler::new(batch, seq, seed);
            sampler.skip_batches(sampler_skip);
            eprintln!(
                "sft_sampler seed={seed} skipped_batches={sampler_skip} continuation=explicit"
            );
            let mut tok = Vec::new();
            let mut tgt = Vec::new();
            for _ in 0..(steps - ck.step as usize) {
                sampler.batch(&train_sft, &replay_shard, &mut tok, &mut tgt);
                let (loss, gnorm, ms) = gpu.train_step(&tok, &tgt, lr, wd, clip);
                assert!(
                    loss.is_finite() && gnorm.is_finite(),
                    "non-finite Vulkan SFT result at step {}: loss={loss:?} grad_l2={gnorm:?}",
                    gpu.step
                );
                if dropless {
                    let cap = (batch * seq).div_ceil(64) * 64;
                    let drops: usize = gpu
                        .routing_counts()
                        .iter()
                        .flat_map(|counts| counts.iter())
                        .map(|&n| n.saturating_sub(cap as u32) as usize)
                        .sum();
                    assert_eq!(
                        drops, 0,
                        "dropless Vulkan SFT route overflow at step {}: cap={cap} drops={drops}",
                        gpu.step
                    );
                }
                eprintln!(
                    "sft_step={} train_response_replay_loss={loss:.6} grad_l2={gnorm:.6} gpu_ms={ms:.3}",
                    gpu.step
                );
            }
            let response_after = eval_sft(&gpu, &mut vt, &mut vg);
            let raw_after = eval_raw(&gpu, &mut rt, &mut rg);
            assert!(response_after.is_finite() && raw_after.is_finite());
            // The final split is deliberately evaluated only after the dev
            // gate is fixed.  Never feed this terminal number back into the
            // extension/selection decision above.
            let final_response_after = final_sft.as_ref().map(|shard| {
                let batches = final_val_batches.expect("final batch count");
                (0..batches)
                    .map(|i| {
                        shard.fixed_batch(batch, i, &mut vt, &mut vg);
                        gpu.eval_loss(&vt, &vg)
                    })
                    .sum::<f32>()
                    / batches as f32
            });
            if let Some(value) = final_response_after {
                assert!(value.is_finite(), "non-finite terminal final response loss");
                eprintln!(
                    "sft_terminal_final_response_nll={value:.6} batches={}",
                    final_val_batches.expect("final batch count")
                );
            }
            let response_gain = response_before - response_after;
            let raw_regression = raw_after - raw_before;
            let gate = response_gain >= 0.02 && raw_regression <= 0.05;
            eprintln!(
                "sft_gate response_nll_after={response_after:.6} gain={response_gain:.6} raw_lm_after={raw_after:.6} regression={raw_regression:.6} pass={gate}"
            );
            let desc = gpu.desc_host();
            let desc_refs: Vec<(&str, &[f32])> =
                desc.iter().map(|(n, x)| (*n, x.as_slice())).collect();
            let params = gpu.params_host();
            let moments_m = gpu.m.to_vec();
            let moments_v = gpu.v.to_vec();
            assert!(params.iter().all(|x| x.is_finite()));
            assert!(moments_m.iter().all(|x| x.is_finite()) && moments_v.iter().all(|x| x.is_finite()));
            assert!(desc_refs.iter().all(|(_, x)| x.iter().all(|v| v.is_finite())));
            save_checkpoint(
                &out,
                &ck.cfg,
                gpu.step,
                &params,
                Some(&moments_m),
                Some(&moments_v),
                &desc_refs,
            )
            .expect("write Vulkan SFT checkpoint");
            println!(
                "native Vulkan SFT receipt: step={} response_before={response_before:.6} response_after={response_after:.6} response_gain={response_gain:.6} raw_before={raw_before:.6} raw_after={raw_after:.6} raw_regression={raw_regression:.6} gate={gate} final_response_after={final_response_after:?} output={}",
                gpu.step,
                out.display()
            );
        }
        Sub::Birth {
            shard,
            val,
            out,
            resume,
            batch,
            seq,
            steps,
            warmup,
            lr,
            wd,
            clip,
            eval_every,
            save_every,
            pca_every,
            freeze_desc,
            tiny,
            vocab,
            anchor_every,
            conv_k,
            gdn_lane,
            phase_delta,
            phase_delta_layer,
            phase_delta_layers,
            router_smooth_k4,
            router_top2_margin,
            anchor_window,
            anchor_sink,
            anchor_train_windows,
            anchor_layers,
            mixer,
            gdn_heads,
            gdn_dk,
            gdn_dv,
            cfg_json,
            init_from,
            distill_from,
            distill_w,
            distill_batch,
            freeze_donor,
            seed,
            carry,
            carry_reset_every,
            carry_eot,
            lr_restart,
            dropless,
            val_extra,
        } => {
            #[cfg(any(target_os = "macos", feature = "vulkan"))]
            cortiq_embryo::cli::birth(cortiq_embryo::cli::BirthArgs {
                shard,
                val,
                out,
                resume,
                batch,
                seq,
                steps,
                warmup,
                lr,
                wd,
                clip,
                eval_every,
                save_every,
                pca_every,
                freeze_desc,
                tiny,
                vocab,
                anchor_every,
                conv_k,
                gdn_lane,
                phase_delta,
                phase_delta_layer,
                phase_delta_layers,
                router_smooth_k4,
                router_top2_margin,
                anchor_window,
                anchor_sink,
                anchor_train_windows,
                anchor_layers,
                mixer,
                gdn_heads,
                gdn_dk,
                gdn_dv,
                cfg_json,
                init_from,
                distill_from,
                distill_w,
                distill_batch,
                freeze_donor,
                seed,
                carry,
                carry_reset_every,
                carry_eot,
                lr_restart,
                dropless,
                val_extra,
            });
            #[cfg(not(any(target_os = "macos", feature = "vulkan")))]
            {
                let _ = (
                    shard,
                    val,
                    out,
                    resume,
                    batch,
                    seq,
                    steps,
                    warmup,
                    lr,
                    wd,
                    clip,
                    eval_every,
                    save_every,
                    pca_every,
                    tiny,
                    vocab,
                    anchor_every,
                    conv_k,
                    gdn_lane,
                    phase_delta,
                    phase_delta_layer,
                    phase_delta_layers,
                    router_smooth_k4,
                    router_top2_margin,
                    anchor_window,
                    anchor_sink,
                    anchor_train_windows,
                    anchor_layers,
                    mixer,
                    gdn_heads,
                    gdn_dk,
                    gdn_dv,
                    cfg_json,
                    init_from,
                    distill_from,
                    distill_w,
                    distill_batch,
                    freeze_donor,
                    carry,
                    carry_reset_every,
                    carry_eot,
                    lr_restart,
                    dropless,
                    val_extra,
                    seed,
                );
                eprintln!("needs Metal (macOS)");
                std::process::exit(1);
            }
        }
    }
}
