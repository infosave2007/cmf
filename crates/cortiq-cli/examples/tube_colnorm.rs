//! Per-neuron down_proj column norms — the second half of the pruning
//! criterion. Dropping neuron n changes the layer output by
//! `a_n · W_down[:, n]`, so the honest importance is activation mass
//! TIMES the column's norm (Wanda's structured form), not mass alone.
//!
//!   cargo run --release -p cortiq-cli --example tube_colnorm -- model.cmf out.mass
//!
//! Writes the same `u32 layers, u32 inter, f32[...]` layout the mass
//! dumps use, so any tool that reads one reads the other.
use cortiq_core::CmfModel;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let model_p = a.next().expect("model.cmf");
    let out_p = a.next().expect("out.mass");
    let m = CmfModel::open_sharded(&model_p)?;
    let (nl, inter, hidden) = (
        m.arch().num_layers,
        m.arch().intermediate_size,
        m.arch().hidden_size,
    );
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out_p)?);
    f.write_all(&(nl as u32).to_le_bytes())?;
    f.write_all(&(inter as u32).to_le_bytes())?;
    for li in 0..nl {
        let name = format!("model.layers.{li}.mlp.down_proj.weight");
        let mut norms = vec![0f32; inter];
        if let Some(e) = m.tensors.iter().find(|t| t.name == name) {
            let mut w = vec![0f32; hidden * inter];
            cortiq_core::quant::dequant_tensor(e, m.tensor_bytes(&name)?, &mut w)?;
            let mut acc = vec![0f64; inter];
            for r in 0..hidden {
                let row = &w[r * inter..(r + 1) * inter];
                for (c, v) in row.iter().enumerate() {
                    acc[c] += (*v as f64) * (*v as f64);
                }
            }
            for (n, v) in acc.iter().enumerate() {
                norms[n] = v.sqrt() as f32;
            }
        } else {
            eprintln!("layer {li}: no dense down_proj (MoE?) — norms left at 0");
        }
        for v in &norms {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    f.flush()?;
    println!("wrote {out_p}: {nl} layers × {inter}");
    Ok(())
}
