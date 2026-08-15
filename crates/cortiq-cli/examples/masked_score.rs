//! Score a mask-carrying file BOTH ways on one text: bare, then with its
//! own default-task mask held active — the masked-inference fast path's
//! acceptance check. A specialist whose FCD weights were trained WITH
//! the mask should score better masked than bare.
//!
//!   cargo run --release -p cortiq-cli --example masked_score -- model.cmf text.txt [tokens]
use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_p = args.next().expect("model.cmf");
    let text_p = args.next().expect("text file");
    let ntok: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let text = std::fs::read_to_string(&text_p)?;
    let m = Arc::new(CmfModel::open(&model_p)?);
    let task = m.masks.default_task.clone();
    let mask = m
        .masks
        .masks
        .iter()
        .find(|x| x.name == task)
        .or_else(|| m.masks.masks.first())
        .cloned();
    let mut p = Pipeline::from_model(&m, SamplerConfig::default())?;
    let mut ids = p.tokenizer.encode(&text);
    ids.truncate(ntok);
    // The bake gate's shape: INDEPENDENT 256-token chunks, kv reset each.
    let score = |p: &mut Pipeline, mask: Option<&cortiq_core::TaskMask>| -> (f64, f64) {
        let t0 = std::time::Instant::now();
        let (mut nll, mut cnt) = (0f64, 0usize);
        for c in ids.chunks(256) {
            if c.len() < 2 {
                break;
            }
            let (l, k) = p.nll_ids_masked(c, 0, mask);
            nll += l;
            cnt += k;
        }
        ((nll / cnt.max(1) as f64).exp(), t0.elapsed().as_secs_f64())
    };
    let (bare, t_bare) = score(&mut p, None);
    let (masked, t_mask) = score(&mut p, mask.as_ref());
    // The decomposition probe: same chunks through the replica's f32 math.
    if std::env::var("MS_REPLICA").is_ok() {
        let chunks: Vec<Vec<u32>> = ids
            .chunks(256)
            .filter(|c| c.len() > 1)
            .map(|c| c.to_vec())
            .collect();
        let (rb, rm) = cortiq_engine::skillbake::replica_score_file_mask(&m, &chunks)?;
        println!("replica-on-written: bare {rb:.3} | masked {rm:.3}");
    }
    println!(
        "bare:   PPL = {bare:.3}  ({t_bare:.1}s)\nmasked: PPL = {masked:.3}  ({t_mask:.1}s, task '{}', {} mask(s))",
        task,
        m.masks.masks.len()
    );
    Ok(())
}
