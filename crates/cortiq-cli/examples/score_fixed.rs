use cortiq_core::CmfModel;
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::sampler::SamplerConfig;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args().nth(1).ok_or("model")?;
    let ids_path = std::env::args().nth(2).ok_or("ids")?;
    let ids: Vec<u32> = std::fs::read_to_string(ids_path)?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if ids.len() < 1025 {
        return Err(format!("need >=1025 ids, got {}", ids.len()).into());
    }
    let model = Arc::new(CmfModel::open_sharded(model_path)?);
    let mut p = Pipeline::from_model_with_skill(&model, SamplerConfig::default(), None)?;
    let mut nll = 0.0f64;
    let mut count = 0usize;
    for &(start, end) in &[(0usize, 513usize), (512usize, 1025usize)] {
        let (n, c) = p.nll_ids_from(&ids[start..end], 0)?;
        println!(
            "chunk start={start} input_len={} nll={n:.9} count={c}",
            end - start
        );
        nll += n;
        count += c;
    }
    let mean = nll / count as f64;
    let ppl = mean.exp();
    println!("fixed-reset-1024 nll={nll:.9} count={count} mean_nll={mean:.9} ppl={ppl:.9}");
    Ok(())
}
