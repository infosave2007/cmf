//! Measure early-exit draft acceptance: how often argmax(final-norm +
//! lm_head over the hidden after L layers) equals the FULL model's own
//! greedy next token. This single number decides whether layer-skip
//! speculative decode can amortize the network split's round trips —
//! measured before any machinery is built, per the house rule.
//!
//! Usage: spec-probe <model.cmf> <prompt> [gen_tokens]
//! Run with CMF_GPU=0 for the exactness reference.

use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::sampler::SamplerConfig;

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: spec-probe <model.cmf> <prompt> [gen]");
    let prompt = args
        .next()
        .expect("usage: spec-probe <model.cmf> <prompt> [gen]");
    let gen_n: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(64);

    let model =
        std::sync::Arc::new(cortiq_core::CmfModel::open_sharded(&model_path).expect("open model"));
    let mut sampler = SamplerConfig::default();
    sampler.temperature = 0.0;
    sampler.repetition_penalty = 1.0;
    let mut p = Pipeline::from_model(&model, sampler).expect("load pipeline");
    p.split_supported().expect("arch supports spans");
    let nl = p.num_layers;

    // Reference: the full model's own greedy continuation.
    let ids = p.tokenizer.with_bos(p.tokenizer.encode(&prompt));
    let r = p
        .generate_from_ids(&ids, gen_n, None, None)
        .expect("reference generation");
    let mut seq = ids.clone();
    seq.extend(&r.token_ids);
    eprintln!(
        "reference: {} prompt + {} generated tokens ({} layers)",
        ids.len(),
        r.token_ids.len(),
        nl
    );
    if r.token_ids.len() < 8 {
        eprintln!("reference too short to measure — give a prompt the model answers longer");
        std::process::exit(1);
    }

    // Sweep early-exit depths. The draft keeps its OWN KV (layers 0..L),
    // teacher-forced over the reference sequence.
    let sweep: Vec<usize> = [nl / 6, nl / 4, nl / 3, nl / 2, nl * 2 / 3, nl * 5 / 6]
        .into_iter()
        .map(|l| l.max(1))
        .collect();
    println!("layer\tshare\taccept\taccept@prompt-end");
    for &l in sweep.iter() {
        p.reset_session();
        let mut hit = 0usize;
        let mut n = 0usize;
        for pos in 0..seq.len() - 1 {
            let emb = p.embed_id(seq[pos]);
            let h = p
                .forward_span(&emb, pos, 0, l - 1, None)
                .expect("draft span forward");
            if pos + 1 >= ids.len() {
                let logits = p.logits_from_hidden(&h);
                if argmax(&logits) == seq[pos + 1] as usize {
                    hit += 1;
                }
                n += 1;
            }
        }
        println!(
            "{l}\t{:.0}%\t{:.1}%\t({hit}/{n})",
            100.0 * l as f64 / nl as f64,
            100.0 * hit as f64 / n.max(1) as f64
        );
    }
}
