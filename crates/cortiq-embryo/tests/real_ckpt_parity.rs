//! Runtime vs trainer on a REAL checkpoint (EMBRYO_CKPT, EMBRYO_TOK env):
//! log-probs at several positions of a held-out window. Skipped otherwise.
#![cfg(target_os = "macos")]

use cortiq_embryo::model::{EmbryoGpu, Layout};

#[test]
fn real_checkpoint_runtime_parity() {
    let (Ok(ck_path), Ok(tok_path)) = (std::env::var("EMBRYO_CKPT"), std::env::var("EMBRYO_TOK")) else { return };
    let Some(_) = cortiq_embryo::metal::ctx() else { return };
    let ck = cortiq_embryo::train::load_checkpoint(std::path::Path::new(&ck_path)).unwrap();
    let cfg = ck.cfg.clone();
    let lay = Layout::new(&cfg);
    let t = 256usize;
    let gpu = EmbryoGpu::new(cfg.clone(), 1, t, &ck.params).unwrap();
    gpu.set_desc(&ck.extras);
    gpu.desc_updates.set(false);
    // tokens: a slice of held-out english
    let bpe = cortiq_embryo::tokenizer::Bpe::load(std::path::Path::new(&tok_path)).unwrap();
    let text = std::fs::read_to_string("/Users/oleg/embryo-data/heldout-en.txt").unwrap();
    let mut ids = Vec::new();
    let mut cache = std::collections::HashMap::new();
    bpe.encode(&text[..20000], &mut cache, &mut ids);
    let tokens: Vec<u32> = ids[..t].to_vec();
    let xf = gpu.forward_hidden(&tokens);
    let (h, v, ncl) = (cfg.hidden, cfg.vocab, cfg.head_clusters);
    let cs = v / ncl;
    let e = &ck.params[lay.embed..lay.embed + v * h];
    let cm = &ck.params[lay.head_clusters..lay.head_clusters + ncl * h];
    let logprobs = |x: &[f32]| -> Vec<f32> {
        let mut lc: Vec<f32> = (0..ncl).map(|c| (0..h).map(|j| cm[c * h + j] * x[j]).sum()).collect();
        let mx = lc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = mx + lc.iter().map(|z| (z - mx).exp()).sum::<f32>().ln();
        for z in &mut lc {
            *z -= lse;
        }
        let mut out = vec![0.0f32; v];
        for c in 0..ncl {
            let lg: Vec<f32> = (0..cs).map(|s| (0..h).map(|j| e[(c * cs + s) * h + j] * x[j]).sum()).collect();
            let bm = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let bl = bm + lg.iter().map(|z| (z - bm).exp()).sum::<f32>().ln();
            for s in 0..cs {
                out[c * cs + s] = lc[c] + lg[s] - bl;
            }
        }
        out
    };
    let cmf = std::env::var("EMBRYO_CMF").unwrap();
    let model = std::sync::Arc::new(cortiq_core::format::CmfModel::open(&cmf).unwrap());
    let mut pipe = cortiq_engine::pipeline::Pipeline::from_model(&model, cortiq_engine::sampler::SamplerConfig::default()).unwrap();
    // trainer's own loss on the window vs runtime's
    let mut tl = 0.0f64;
    let mut rl = 0.0f64;
    for pos in [0usize, 63, 127, 191, 200, 210, 220, 230, 240, 250, 254] {
        let want = logprobs(&xf[pos * h..(pos + 1) * h]);
        let got = pipe.prefill_next_logits(&tokens[..=pos], None);
        let d = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let nxt = tokens[(pos + 1).min(t - 1)] as usize;
        tl += -want[nxt] as f64;
        rl += -got[nxt] as f64;
        eprintln!("pos {pos:>3}: max|Δ| {d:.3e}  trainer lp(next) {:.3}  runtime lp(next) {:.3}", want[nxt], got[nxt]);
    }
    eprintln!("mean nll over probes: trainer {:.3} runtime {:.3}", tl / 11.0, rl / 11.0);
}
