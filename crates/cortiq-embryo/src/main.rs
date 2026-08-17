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
    /// Export a checkpoint (+ tokenizer.json) into a runtime-loadable .cmf.
    Export {
        #[arg(long)]
        ckpt: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        out: PathBuf,
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
    /// Bake a skill from a genome checkpoint on a task corpus and append it to a .cmf (P2/P15).
    SkillBake {
        /// genome checkpoint (frozen)
        #[arg(long)]
        ckpt: PathBuf,
        /// tokenizer.json (ours)
        #[arg(long)]
        tokenizer: PathBuf,
        /// task corpus files (txt / jsonl.gz / parquet)
        #[arg(long, required = true, num_args = 1..)]
        corpus: Vec<PathBuf>,
        /// base .cmf to append to (exported genome)
        #[arg(long)]
        base: PathBuf,
        /// output .cmf (defaults to overwrite --base)
        #[arg(long)]
        out: Option<PathBuf>,
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
        #[arg(long, default_value_t = 512)]
        seq: usize,
        /// φ-layer of the routing descriptor (default: 2/3 depth)
        #[arg(long)]
        phi_layer: Option<usize>,
        /// positions averaged for the routing φ (prompt-length statistics)
        #[arg(long, default_value_t = 48)]
        phi_len: usize,
        #[arg(long, default_value_t = 8)]
        rank: usize,
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
    /// Growth as records: add one expert per layer (copy of the hottest + shifted descriptor), train only the new experts on a corpus, gate on held-out, save the grown genome (+ optional .cmf export).
    Grow {
        #[arg(long)]
        ckpt: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, required = true, num_args = 1..)]
        corpus: Vec<PathBuf>,
        #[arg(long)]
        out_ckpt: PathBuf,
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
        /// required held-out improvement (fraction of loss) to keep the growth
        #[arg(long, default_value_t = 0.005)]
        gate: f32,
        #[arg(long, default_value_t = 1e-3)]
        noise: f32,
        #[arg(long, default_value_t = 0.1)]
        shift: f32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
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
        /// tiny genome (smoke test)
        #[arg(long)]
        tiny: bool,
        /// vocab override (e.g. 256 for byte shards)
        #[arg(long)]
        vocab: Option<usize>,
        /// every N-th layer is a softmax anchor (1 = the all-softmax twin)
        #[arg(long)]
        anchor_every: Option<usize>,
        /// JSON overrides merged into the genome config (ablations), e.g. '{"hidden":256,"layers":6}'
        #[arg(long)]
        cfg_json: Option<String>,
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
        Sub::Export { ckpt, tokenizer, out } => {
            let ck = cortiq_embryo::train::load_checkpoint(&ckpt).expect("load checkpoint");
            let tj = std::fs::read(&tokenizer).expect("read tokenizer.json");
            cortiq_embryo::export::export(&ck, &tj, &out).expect("export");
            let (total, active) = ck.cfg.params();
            println!("exported step {} → {} ({:.1} M params, {:.1} M active)", ck.step, out.display(), total as f64 / 1e6, active as f64 / 1e6);
        }
        Sub::SampleText { input, skip, docs, out } => {
            cortiq_embryo::corpus::sample_text(&input, skip, docs, &out);
        }
        Sub::SkillBake { ckpt, tokenizer, corpus, base, out, id, layers, steps_a, steps_b, lr_a, lr_b, l1, tau, eval_every, batch, seq, phi_layer, phi_len, rank, seed } => {
            #[cfg(target_os = "macos")]
            {
                use cortiq_embryo::skill::{BakeArgs, append_to_cmf, bake};
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
                let layers = layers.unwrap_or_else(|| vec![nl.saturating_sub(2), nl.saturating_sub(1)]);
                let a = BakeArgs {
                    id: id.clone(), layers: layers.clone(), steps_a, steps_b, lr_a, lr_b, l1, tau, eval_every, batch, seq,
                    phi_layer: phi_layer.unwrap_or(nl * 2 / 3), phi_len, rank, seed,
                };
                let (tensors, sel, kept, (l0, la, lb)) = bake(&ck, &shard, &a, &|| false).expect("bake");
                let quality = serde_json::json!({
                    "held_out_loss": {"base": l0, "mask": la, "mask+fcd": lb},
                    "held_out_ppl": {"base": l0.exp(), "mask": la.exp(), "mask+fcd": lb.exp()},
                    "kept_fraction": kept,
                });
                let out_path = out.unwrap_or_else(|| base.clone());
                let tmp = out_path.with_extension("cmf.tmp");
                let unchanged = append_to_cmf(&base, &tmp, &id, &layers, &tensors, sel, quality).expect("append");
                std::fs::rename(&tmp, &out_path).expect("rename");
                println!("skill '{id}' appended → {} ({} tensors over layers {:?}; {unchanged} base tensors byte-identical; held-out ppl {:.1} → {:.1})", out_path.display(), tensors.len(), layers, l0.exp(), lb.exp());
                match cortiq_embryo::skill::calibrate_file(&out_path, 0.05) {
                    Ok(Some(c)) => println!("router calibrated over {} held-out φ: temperature {:.3e}, novelty θ {:.3} (fpr {:.2})", c.samples, c.temperature, c.novelty_theta, c.target_fpr),
                    Ok(None) => println!("router calibration: no held-out φ in the file"),
                    Err(e) => eprintln!("router calibration failed: {e}"),
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (ckpt, tokenizer, corpus, base, out, id, layers, steps_a, steps_b, lr_a, lr_b, l1, tau, eval_every, batch, seq, phi_layer, phi_len, rank, seed);
                eprintln!("needs Metal (macOS)");
                std::process::exit(1);
            }
        }
        Sub::Sleep { ckpt, tokenizer, cmf, ood_dir, idle_min, min_tokens, gate, requant_gate, held_out, cortiq_bin, layers, steps_a, steps_b, batch, seq, once, force, poll_secs, grow_after } => {
            #[cfg(target_os = "macos")]
            {
                let nl = cortiq_embryo::train::load_checkpoint(&ckpt).map(|c| c.cfg.layers).unwrap_or(8);
                let layers = layers.unwrap_or_else(|| (nl.saturating_sub(3)..nl).collect());
                cortiq_embryo::sleep::run(cortiq_embryo::sleep::SleepArgs {
                    ckpt, tokenizer, cmf, ood_dir, idle_min, min_tokens, gate, requant_gate, held_out, cortiq_bin, layers, steps_a, steps_b, batch, seq, once, force, poll_secs, grow_after,
                })
                .expect("sleep");
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (ckpt, tokenizer, cmf, ood_dir, idle_min, min_tokens, gate, requant_gate, held_out, cortiq_bin, layers, steps_a, steps_b, batch, seq, once, force, poll_secs, grow_after);
                eprintln!("needs Metal (macOS)");
                std::process::exit(1);
            }
        }
        Sub::Grow { ckpt, tokenizer, corpus, out_ckpt, export, steps, lr, batch, seq, gate, noise, shift, seed } => {
            #[cfg(target_os = "macos")]
            {
                use cortiq_embryo::growth::{GrowArgs, grow_experts, train_new_experts};
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
                let (grown, sources) = grow_experts(&ck, noise, shift, seed);
                println!("grown: experts {} → {} in {} layers; sources per layer {:?}", ck.cfg.experts, grown.cfg.experts, grown.cfg.layers, sources);
                let a = GrowArgs { steps, lr, batch, seq, eval_every: 30, seed };
                let (trained, l0, l1) = train_new_experts(&grown, &shard, &a, &|| false).expect("train");
                let imp = (l0 - l1) / l0.max(1e-6);
                println!("held-out: before {l0:.4} (ppl {:.1}) → after {l1:.4} (ppl {:.1}), improvement {imp:.4} vs gate {gate}", l0.exp(), l1.exp());
                if imp < gate {
                    println!("growth REJECTED (gate) — nothing written");
                    std::process::exit(2);
                }
                let d: Vec<(&str, &[f32])> = trained.extras.iter().map(|(n, x)| (n.as_str(), x.as_slice())).collect();
                cortiq_embryo::train::save_checkpoint(&out_ckpt, &trained.cfg, trained.step, &trained.params, None, None, &d).expect("save");
                println!("grown genome saved → {}", out_ckpt.display());
                if let Some(e) = export {
                    let tj = std::fs::read(&tokenizer).expect("read tokenizer.json");
                    cortiq_embryo::export::export(&trained, &tj, &e).expect("export");
                    println!("exported → {}", e.display());
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (ckpt, tokenizer, corpus, out_ckpt, export, steps, lr, batch, seq, gate, noise, shift, seed);
                eprintln!("needs Metal (macOS)");
                std::process::exit(1);
            }
        }
        Sub::MtpTrain { ckpt, tokenizer, corpus, base, out, heads, steps, lr, batch, seq, seed } => {
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
                let (_gpu, st, held) = cortiq_embryo::mtp::train_mtp(&ck, &shard, heads, steps, lr, batch, seq, seed).expect("mtp");
                println!("mtp heads trained: held-out losses {:?} (ppl {:?})", held, held.iter().map(|l| format!("{:.1}", l.exp())).collect::<Vec<_>>());
                let out_path = out.unwrap_or_else(|| base.clone());
                let tmp = out_path.with_extension("cmf.tmp");
                let kept = cortiq_embryo::mtp::append_to_cmf(&base, &tmp, &st).expect("append");
                std::fs::rename(&tmp, &out_path).expect("rename");
                println!("{} MTP tensors appended → {} ({kept} base tensors byte-identical)", 2 * heads, out_path.display());
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (ckpt, tokenizer, corpus, base, out, heads, steps, lr, batch, seq, seed);
                eprintln!("needs Metal (macOS)");
                std::process::exit(1);
            }
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
        Sub::Birth { shard, val, out, resume, batch, seq, steps, warmup, lr, wd, clip, eval_every, save_every, pca_every, tiny, vocab, anchor_every, cfg_json, seed } => {
            #[cfg(target_os = "macos")]
            cortiq_embryo::cli::birth(cortiq_embryo::cli::BirthArgs {
                shard, val, out, resume, batch, seq, steps, warmup, lr, wd, clip, eval_every, save_every, pca_every, tiny, vocab, anchor_every, cfg_json, seed,
            });
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (shard, val, out, resume, batch, seq, steps, warmup, lr, wd, clip, eval_every, save_every, pca_every, tiny, vocab, anchor_every, cfg_json, seed);
                eprintln!("needs Metal (macOS)");
                std::process::exit(1);
            }
        }
    }
}
