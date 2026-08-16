//! `cortiq-embryo` — the birth/growth trainer CLI (native Rust + Metal).

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cortiq-embryo", about = "Cortiq Embryo trainer (native Rust + Metal)")]
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
    },
    /// Turn a text file into a byte-level token shard (vocab 256) — smoke corpus.
    BytesShard {
        input: PathBuf,
        output: PathBuf,
    },
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
        /// tiny genome (smoke test)
        #[arg(long)]
        tiny: bool,
        /// vocab override (e.g. 256 for byte shards)
        #[arg(long)]
        vocab: Option<usize>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Sub::Config => {
            let cfg = cortiq_embryo::model::EmbryoCfg::embryo0();
            let (total, active) = cfg.params();
            println!("{cfg:#?}");
            println!("params: total {:.2} M, active/token {:.2} M", total as f64 / 1e6, active as f64 / 1e6);
            let lay = cortiq_embryo::model::Layout::new(&cfg);
            println!("trainer arena now (shared expert, no routed experts yet): {:.2} M", lay.total as f64 / 1e6);
        }
        Sub::BytesShard { input, output } => {
            let text = std::fs::read(&input).expect("read input");
            let shard = cortiq_embryo::train::Shard::from_bytes(&text);
            shard.save(&output).expect("write shard");
            println!("{} tokens → {}", shard.tokens.len(), output.display());
        }
        Sub::TrainTokenizer { inputs, out, vocab, sample_mb_bytes } => {
            cortiq_embryo::corpus::train_tokenizer(&inputs, &out, vocab, sample_mb_bytes);
        }
        Sub::Shard { tokenizer, inputs, out, max_tokens } => {
            cortiq_embryo::corpus::shard(&tokenizer, &inputs, &out, max_tokens);
        }
        Sub::Fetch { dir, urls } => {
            cortiq_embryo::corpus::fetch(&urls, &dir);
        }
        Sub::Bench { reps, no_verify } => {
            #[cfg(target_os = "macos")]
            {
                let Some(rows) = cortiq_embryo::bench::run(reps, !no_verify) else {
                    eprintln!("no Metal device");
                    std::process::exit(1);
                };
                println!("{:<44} {:>6} {:>6} {:>6} {:>9} {:>8} {:>10}", "shape", "M", "N", "K", "gpu ms", "TFLOPS", "max|err|");
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
        Sub::StepBench { batch, seq, steps, tiny } => {
            #[cfg(target_os = "macos")]
            cortiq_embryo::cli::step_bench(batch, seq, steps, tiny);
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (batch, seq, steps, tiny);
                eprintln!("needs Metal (macOS)");
                std::process::exit(1);
            }
        }
        Sub::Birth { shard, val, out, resume, batch, seq, steps, warmup, lr, wd, clip, eval_every, save_every, tiny, vocab, seed } => {
            #[cfg(target_os = "macos")]
            cortiq_embryo::cli::birth(cortiq_embryo::cli::BirthArgs {
                shard, val, out, resume, batch, seq, steps, warmup, lr, wd, clip, eval_every, save_every, tiny, vocab, seed,
            });
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (shard, val, out, resume, batch, seq, steps, warmup, lr, wd, clip, eval_every, save_every, tiny, vocab, seed);
                eprintln!("needs Metal (macOS)");
                std::process::exit(1);
            }
        }
    }
}
