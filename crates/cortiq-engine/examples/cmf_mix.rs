//! Dev-only: build a CMF that takes every tensor from A except those whose
//! name matches REGEX, which come from B (same names, any dtype). Used to
//! find which tensor group of a tower carries the quantization loss.
//! Also prints the weight-space error of every swapped tensor (B vs A).
//!
//!     cargo run --release -p cortiq-engine --example cmf_mix -- A.cmf B.cmf OUT.cmf REGEX

use cortiq_core::format::{CmfModel, TensorSpecRef};

fn deq(m: &CmfModel, name: &str) -> Vec<f32> {
    let e = m.tensor(name).unwrap();
    let mut out = vec![0f32; e.shape.iter().product()];
    cortiq_core::quant::dequant_tensor(e, m.entry_bytes(e), &mut out).unwrap();
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: cmf_mix A.cmf B.cmf OUT.cmf REGEX");
        std::process::exit(2);
    }
    let a = CmfModel::open(&args[1]).expect("open A");
    let b = CmfModel::open(&args[2]).expect("open B");
    let re = fancy_regex::Regex::new(&args[4]).expect("regex");
    let verbose = std::env::var("CMF_MIX_VERBOSE").is_ok();
    let mut specs = Vec::with_capacity(a.tensors.len());
    let mut swapped = 0usize;
    for e in &a.tensors {
        let take_b = re.is_match(&e.name).unwrap_or(false) && b.tensor(&e.name).is_some();
        let (m, entry) = if take_b {
            (&b, b.tensor(&e.name).unwrap())
        } else {
            (&a, e)
        };
        if take_b {
            swapped += 1;
            if verbose {
                let (wa, wb) = (deq(&a, &e.name), deq(&b, &e.name));
                let dot: f64 = wa.iter().zip(&wb).map(|(x, y)| *x as f64 * *y as f64).sum();
                let na: f64 = wa.iter().map(|x| (*x as f64).powi(2)).sum();
                let nb: f64 = wb.iter().map(|x| (*x as f64).powi(2)).sum();
                let err: f64 = wa
                    .iter()
                    .zip(&wb)
                    .map(|(x, y)| ((x - y) as f64).powi(2))
                    .sum();
                println!(
                    "{:<48} {:?} cos {:.6} rel-rms {:.4}",
                    e.name,
                    entry.dtype,
                    dot / (na * nb).sqrt(),
                    (err / na).sqrt()
                );
            }
        }
        specs.push(TensorSpecRef {
            name: e.name.clone(),
            dtype: entry.dtype,
            shape: entry.shape.clone(),
            data: m.entry_bytes(entry),
        });
    }
    let mut header = a.header.clone();
    header.section_hashes = None;
    CmfModel::write_ref(&args[3], &header, &specs, None, a.vocab.as_deref()).expect("write");
    println!("wrote {} ({swapped} tensors from B)", args[3]);
}
