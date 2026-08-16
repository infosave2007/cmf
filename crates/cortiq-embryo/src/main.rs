//! `cortiq-embryo` — the birth/growth trainer CLI.

use clap::{Parser, Subcommand};

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
        /// dispatches per shape inside one command buffer
        #[arg(long, default_value_t = 10)]
        reps: usize,
        /// skip the CPU spot-check of results
        #[arg(long)]
        no_verify: bool,
    },
    /// Print the Embryo-0 genome configuration and parameter counts.
    Config,
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Sub::Config => {
            let cfg = cortiq_embryo::model::EmbryoCfg::embryo0();
            let (total, active) = cfg.params();
            println!("{cfg:#?}");
            println!("params: total {:.2} M, active/token {:.2} M", total as f64 / 1e6, active as f64 / 1e6);
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
    }
}
