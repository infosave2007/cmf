//! Repair tool: set every mask's head rows to all-ones (all heads
//! active) in an existing .cmf. The first per-visit specialist was
//! written with EMPTY head_masks, which the codec encodes as all-zero
//! rows — "no active heads" — so the loader forced f32 storage and the
//! head path masked attention itself away.
//!
//!     cargo run --release -p cortiq-cli --example fix_head_masks -- in.cmf out.cmf

use cortiq_core::{CmfModel, TensorSpec};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (inp, out) = (args.next().expect("in.cmf"), args.next().expect("out.cmf"));
    let m = CmfModel::open(&inp)?;
    let tensors: Vec<TensorSpec> = m
        .tensors
        .iter()
        .map(|t| TensorSpec {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: m.entry_bytes(t).to_vec(),
        })
        .collect();
    let mut catalog = m.masks.clone();
    let hb = m.header.arch.num_attention_heads.div_ceil(8);
    let tail = m.header.arch.num_attention_heads % 8;
    let mut row = vec![0xFFu8; hb];
    if tail != 0 {
        row[hb - 1] = (1u8 << tail) - 1;
    }
    for mask in &mut catalog.masks {
        mask.head_masks = vec![row.clone(); m.header.arch.num_layers];
    }
    // STRIP=1: write no catalog at all — the control that separates
    // "the mask catalog slows the runtime down" from "the rewritten
    // tensors do".
    if std::env::var("STRIP").as_deref() == Ok("1") {
        CmfModel::write(&out, &m.header, &tensors, None, m.vocab.as_deref())?;
        println!("wrote {out} with NO masks");
    } else {
        CmfModel::write(
            &out,
            &m.header,
            &tensors,
            Some(&catalog),
            m.vocab.as_deref(),
        )?;
        println!("wrote {out} with open head masks");
    }
    Ok(())
}
