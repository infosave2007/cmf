//! Legacy `FullAttention` Embryo genome (pre-S1 layout), a 4096-token
//! prefix scored on the CPU prefill-GEMM path under a 27-worker pool —
//! the shape of the two field failures on the S4 control export: a
//! SIGSEGV after ~8 s (`ppl --tokens 4096`) and a 47-minute spin
//! (`ppl --windows 8 --window-len 512`).  Both were one worker-pool
//! race: a worker not invited to a limited dispatch read the NEXT job's
//! descriptor with the previous epoch, ran it twice, and either wrapped
//! the barrier (the caller spins forever) or ran the closure after its
//! caller had returned (a dangling borrow — the segfault).  The pool's
//! own stress test pins the race; this test pins the end-to-end shape:
//! it must finish, score every position, and reproduce its own NLL bit
//! for bit on a second pass.  A watchdog turns a hang into a failure.

use cortiq_core::CmfModel;
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::sampler::SamplerConfig;
use std::sync::Arc;

#[path = "common/embryo_synth.rs"]
mod embryo_synth;

#[test]
fn legacy_full_attention_4096_prefix_scores_under_a_27_worker_pool() {
    // Before the first pipeline: the pool and the backend are chosen at
    // first use.  27 workers is the server's pool for this model.
    unsafe {
        std::env::set_var("CMF_GPU", "0");
        std::env::set_var("CMF_THREADS", "27");
        std::env::set_var("CMF_MAX_SEQ", "8192");
        std::env::remove_var("CMF_EMBRYO_RESIDENT");
    }
    let g = embryo_synth::SynthGeom::legacy_full_small();
    let path = embryo_synth::synth_genome_path(&g);
    let model = Arc::new(CmfModel::open(&path).expect("open legacy synthetic genome"));
    let mut p = Pipeline::from_model(&model, SamplerConfig::default()).expect("legacy file loads");
    assert!(!p.bounded_native(), "legacy genome must not carry anchor_core");
    let ids = embryo_synth::synth_ids(4097, 3, g.vocab);
    // Watchdog: the old pool spun forever here at 100% of one core.
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let d = done.clone();
    std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        while !d.load(std::sync::atomic::Ordering::SeqCst) {
            if t0.elapsed() > std::time::Duration::from_secs(600) {
                eprintln!("legacy 4096-token scoring hung for 600 s (pool barrier lost)");
                std::process::abort();
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });
    let (n1, c1) = p.nll_ids_from(&ids, 0).expect("nll scoring");
    let (n2, c2) = p.nll_ids_from(&ids, 0).expect("nll scoring");
    done.store(true, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(c1, 4096, "every position scored");
    assert_eq!(c2, 4096);
    assert!(n1.is_finite() && n1 > 0.0, "nll {n1}");
    assert_eq!(n1.to_bits(), n2.to_bits(), "pool scoring is not reproducible: {n1} vs {n2}");
    eprintln!(
        "legacy 4096: ppl {:.3}, pool {} workers",
        (n1 / c1 as f64).exp(),
        cortiq_engine::pool::Pool::effective_threads()
    );
}
