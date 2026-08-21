//! Score a tube file the way it will be served: every task on its OWN
//! held-out text, with its OWN mask active, against the dense model.
//!
//!   CMF_GPU=0 cargo run --release -p cortiq-cli --example tube_eval -- \
//!       tubes.cmf eval_dir [dense.cmf] [tokens] [cross]
//!
//! `eval_dir/<task>.txt` must exist for every mask in the tube file.
//! Columns: the dense backbone, the tube file with every tube on (the
//! defrag's own cost — requantization only), and the tube file with
//! just that task's tubes (what the task actually pays for its width).
use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::sync::Arc;

fn ppl(p: &mut Pipeline, ids: &[u32], mask: Option<&cortiq_core::TaskMask>) -> f64 {
    let (mut nll, mut cnt) = (0f64, 0usize);
    for c in ids.chunks(256) {
        if c.len() < 2 {
            break;
        }
        let (l, k) = p.nll_ids_masked(c, 0, mask);
        nll += l;
        cnt += k;
    }
    (nll / cnt.max(1) as f64).exp()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let tube_p = a.next().expect("tubes.cmf");
    let eval_d = a.next().expect("eval dir");
    let dense_p = a.next();
    let ntok: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let cross = a.next().is_some();

    let tm = Arc::new(CmfModel::open_sharded(&tube_p)?);
    let masks = tm.masks.masks.clone();
    if masks.is_empty() {
        return Err("tube file carries no task masks".into());
    }
    let mut tp = Pipeline::from_model(&tm, SamplerConfig::default())?;

    let one_file = std::path::Path::new(&eval_d).is_file();
    let mut evals: Vec<(String, Vec<u32>)> = Vec::new();
    for m in &masks {
        // A directory scores every task on its OWN held-out text; a
        // single file scores every mask on the same text (the yardstick
        // run: wikitext through each width).
        let path = if one_file {
            eval_d.clone()
        } else {
            format!("{eval_d}/{}.txt", m.name)
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("no eval text for {} ({path})", m.name);
            continue;
        };
        let mut ids = tp.tokenizer.encode(&text);
        ids.truncate(ntok);
        evals.push((m.name.clone(), ids));
    }

    // Dense baseline first (a second mmap; scored on the same ids —
    // the tokenizers are the same file's).
    let mut dense: Vec<f64> = Vec::new();
    if let Some(dp) = &dense_p {
        let dm = Arc::new(CmfModel::open_sharded(dp)?);
        let mut p = Pipeline::from_model(&dm, SamplerConfig::default())?;
        for (_, ids) in &evals {
            dense.push(ppl(&mut p, ids, None));
        }
    }

    println!(
        "{:18} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "task", "dense", "tube-all", "tube-own", "active", "vs dense"
    );
    let (mut sd, mut sa, mut so) = (0f64, 0f64, 0f64);
    for (i, (task, ids)) in evals.iter().enumerate() {
        let m = masks.iter().find(|m| &m.name == task).unwrap();
        let all = ppl(&mut tp, ids, None);
        let own = ppl(&mut tp, ids, Some(m));
        let d = dense.get(i).copied().unwrap_or(f64::NAN);
        let act = 1.0 - m.sparsity as f64;
        println!(
            "{task:18} {d:9.3} {all:9.3} {own:9.3} {:7.1}% {:8.3}",
            act * 100.0,
            own / d
        );
        sd += d;
        sa += all;
        so += own;
    }
    let n = evals.len() as f64;
    println!(
        "{:18} {:9.3} {:9.3} {:9.3} {:>8} {:8.3}",
        "MEAN",
        sd / n,
        sa / n,
        so / n,
        "",
        so / sd
    );

    if cross {
        println!("\ncross matrix (rows = text, cols = task mask):");
        print!("{:18}", "");
        for m in &masks {
            print!("{:>9}", &m.name[..m.name.len().min(8)]);
        }
        println!();
        for (task, ids) in &evals {
            print!("{task:18}");
            for m in &masks {
                print!("{:9.3}", ppl(&mut tp, ids, Some(m)));
            }
            println!();
        }
    }
    Ok(())
}
