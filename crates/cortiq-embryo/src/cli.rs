//! Trainer commands (macOS/Metal): step timing and the birth loop.

use crate::model::{EmbryoCfg, EmbryoGpu, Layout, init_params};
use crate::train::{Mix, Sampler, Shard, load_checkpoint, lr_at, save_checkpoint};
use std::path::PathBuf;
use std::time::Instant;

pub fn step_bench(batch: usize, seq: usize, steps: usize, tiny: bool) {
    let cfg = if tiny { EmbryoCfg::tiny() } else { EmbryoCfg::embryo0() };
    let lay = Layout::new(&cfg);
    let (total, active) = cfg.params();
    println!(
        "genome: {} layers, hidden {}, vocab {}; arena {:.2} M params (§3 count {:.1} M / {:.1} M active)",
        cfg.layers, cfg.hidden, cfg.vocab, lay.total as f64 / 1e6, total as f64 / 1e6, active as f64 / 1e6
    );
    let t0 = Instant::now();
    let p = init_params(&cfg, &lay, 1);
    let mut gpu = EmbryoGpu::new(cfg.clone(), batch, seq, &p).expect("Metal");
    println!("alloc + init: {:.1} s", t0.elapsed().as_secs_f64());
    let m = batch * seq;
    let tokens: Vec<u32> = crate::ops::lcg_vec(3, m).iter().map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32).collect();
    let targets: Vec<u32> = crate::ops::lcg_vec(4, m).iter().map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32).collect();
    let mut best = f64::MAX;
    for s in 0..steps {
        let t = Instant::now();
        let (loss, gnorm, gpu_ms) = gpu.train_step(&tokens, &targets, 1e-4, 0.1, 1.0);
        let wall = t.elapsed().as_secs_f64() * 1e3;
        best = best.min(wall);
        println!(
            "step {s}: loss {loss:.4} |g| {gnorm:.3} gpu {gpu_ms:.0} ms wall {wall:.0} ms → {:.0} tok/s",
            m as f64 / (wall * 1e-3)
        );
        if s + 1 == steps {
            for (l, cnt) in gpu.routing_counts().iter().enumerate() {
                println!("  layer {l} experts: {cnt:?} (of {m}, cap {})", gpu.moe_cap);
            }
        }
    }
    if std::env::var("EMBRYO_PROFILE").is_ok() {
        println!("--- per-phase GPU ms ---");
        for (name, ms) in gpu.profile_step() {
            println!("{name:<28} {ms:>8.1} ms");
        }
        for (name, ms) in gpu.profile_hk_layer(0) {
            println!("{name:<44} {ms:>8.1} ms");
        }
    }
    let flop_per_tok = 6.0 * lay.total as f64;
    println!(
        "best step {best:.0} ms = {:.0} tok/s; 6·N·tok = {:.2} TFLOPS effective; 500 M tokens ≈ {:.1} h",
        m as f64 / (best * 1e-3),
        flop_per_tok * m as f64 / (best * 1e-3) / 1e12,
        500e6 / (m as f64 / (best * 1e-3)) / 3600.0
    );
}

pub struct BirthArgs {
    pub shard: Vec<String>,
    pub val: Option<PathBuf>,
    pub out: PathBuf,
    pub resume: Option<PathBuf>,
    pub batch: usize,
    pub seq: usize,
    pub steps: usize,
    pub warmup: usize,
    pub lr: f32,
    pub wd: f32,
    pub clip: f32,
    pub eval_every: usize,
    pub save_every: usize,
    pub pca_every: usize,
    pub tiny: bool,
    pub vocab: Option<usize>,
    pub seed: u64,
}

pub fn birth(a: BirthArgs) {
    // corpus: shards mixed by weight (`path[:weight]`); validation = the
    // held-out tail (0.5%) of every shard, or an explicit --val shard.
    let (train, val) = {
        let holdout = if a.val.is_some() { 0.0 } else { 0.005 };
        let (mix, tail) = Mix::load(&a.shard, holdout, a.seq).expect("load shards");
        let val = match &a.val {
            Some(v) => Shard::load(v).expect("load val shard"),
            None => tail,
        };
        (mix, val)
    };
    let (mut cfg, params, step0, m0, v0, extras) = match &a.resume {
        Some(p) => {
            let ck = load_checkpoint(p).expect("load checkpoint");
            println!("resumed {} at step {}", p.display(), ck.step);
            (ck.cfg, ck.params, ck.step, ck.m, ck.v, ck.extras)
        }
        None => {
            let mut cfg = if a.tiny { EmbryoCfg::tiny() } else { EmbryoCfg::embryo0() };
            if let Some(v) = a.vocab {
                assert!(v % 64 == 0, "vocab must be a multiple of 64");
                cfg.vocab = v;
            }
            let lay = Layout::new(&cfg);
            (cfg.clone(), init_params(&cfg, &lay, a.seed), 0, None, None, Vec::new())
        }
    };
    if let Some(v) = a.vocab {
        cfg.vocab = v;
    }
    let lay = Layout::new(&cfg);
    println!(
        "genome: {} layers, hidden {}, vocab {}, arena {:.2} M; train {} tok, val {} tok; B={} T={} steps={}",
        cfg.layers, cfg.hidden, cfg.vocab, lay.total as f64 / 1e6, train.total_tokens(), val.tokens.len(), a.batch, a.seq, a.steps
    );
    let mut gpu = EmbryoGpu::new(cfg.clone(), a.batch, a.seq, &params).expect("Metal");
    if let (Some(m), Some(v)) = (m0, v0) {
        gpu.m.write_from(&m);
        gpu.v.write_from(&v);
    }
    gpu.set_desc(&extras);
    gpu.step = step0;
    let mut sampler = Sampler::new(a.batch, a.seq, a.seed.wrapping_add(step0 as u64));
    let (mut tokens, mut targets) = (Vec::new(), Vec::new());
    let m = a.batch * a.seq;
    let val_batches = (val.tokens.len() / (m + 1)).clamp(1, 8);
    let eval = |gpu: &EmbryoGpu, tokens: &mut Vec<u32>, targets: &mut Vec<u32>| -> f32 {
        let mut s = 0.0f32;
        for i in 0..val_batches {
            Sampler::fixed_batch(&val, a.batch, a.seq, i, tokens, targets);
            s += gpu.eval_loss(tokens, targets);
        }
        s / val_batches as f32
    };
    let t_start = Instant::now();
    let mut ema = 0.0f32;
    let mut cov_ema: Vec<f32> = Vec::new();
    for step in step0 as usize..a.steps {
        sampler.batch_mix(&train, &mut tokens, &mut targets);
        let lr = lr_at(step, a.steps, a.warmup, a.lr, a.lr * 0.1);
        let t = Instant::now();
        let (loss, gnorm, _gpu_ms) = gpu.train_step(&tokens, &targets, lr, a.wd, a.clip);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        ema = if step == step0 as usize { loss } else { 0.98 * ema + 0.02 * loss };
        if a.pca_every > 0 && (step + 1) % a.pca_every == 0 {
            gpu.update_subspaces(&mut cov_ema, 0.9);
        }
        if step % 10 == 0 || step + 1 == a.steps {
            println!(
                "step {step:>6} loss {loss:.4} (ema {ema:.4}) |g| {gnorm:.3} lr {lr:.2e} {ms:.0} ms {:.0} tok/s  [{:.1} min]",
                m as f64 / (ms * 1e-3),
                t_start.elapsed().as_secs_f64() / 60.0
            );
            use std::io::Write as _;
            let _ = std::io::stdout().flush();
        }
        if !loss.is_finite() {
            eprintln!("loss is not finite at step {step} — stopping");
            break;
        }
        if (step + 1) % a.eval_every == 0 || step + 1 == a.steps {
            let vl = eval(&gpu, &mut tokens, &mut targets);
            println!("  val loss {vl:.4}  ppl {:.2}", vl.exp());
            let rc = gpu.routing_counts();
            if !rc.is_empty() {
                let cap = gpu.moe_cap;
                let summary: Vec<String> = rc
                    .iter()
                    .map(|c| {
                        let dropped: u32 = c.iter().map(|&n| n.saturating_sub(cap as u32)).sum();
                        format!("{:?}{}", c, if dropped > 0 { format!("(-{dropped})") } else { String::new() })
                    })
                    .collect();
                println!("  experts/layer: {}", summary.join(" "));
            }
        }
        if (step + 1) % a.save_every == 0 || step + 1 == a.steps {
            let p = gpu.params_host();
            let d = gpu.desc_host();
            let ex: Vec<(&str, &[f32])> = d.iter().map(|(n, x)| (*n, x.as_slice())).collect();
            save_checkpoint(&a.out, &cfg, gpu.step, &p, Some(gpu.m.as_slice()), Some(gpu.v.as_slice()), &ex).expect("save");
            println!("  saved {} (step {})", a.out.display(), gpu.step);
        }
    }
}

